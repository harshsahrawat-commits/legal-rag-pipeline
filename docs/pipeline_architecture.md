# Pipeline Architecture — 9-Phase System

## Overview

```
                                        INGESTION PIPELINE
Raw Documents → [1. Acquisition] → [2. Parsing] → [3. Chunking] → [4. Enrichment] → [5. Embedding & Indexing] → [6. Knowledge Graph]

                                        QUERY-TIME PIPELINE
User Query → [0. Query Intelligence] → [7. Retrieval] → [8. Hallucination Mitigation] → [9. Quality Assurance]
```

Each phase is a standalone module. Phases communicate via filesystem (parsed files, chunk JSONs) or message queue (Redis/Celery for async). Every phase is idempotent — safe to re-run.

---

## Phase 0: Query Intelligence Layer (`src/query/`) — NEW

A pre-retrieval layer that decides *whether*, *how*, and *how deeply* to retrieve for each query. Three stacked components.

Full details in `docs/query_intelligence.md`.

**Component A: Semantic Query Cache** — Qdrant `query_cache` collection + Redis response store. 0.92 similarity threshold, 24hr TTL, amendment-triggered invalidation. Expected 30-50% hit rate, 123x latency reduction on hits.

**Component B: Adaptive RAG Query Router** — Rule-based classifier routing queries to 4 retrieval paths:
- SIMPLE → Knowledge Graph direct query (100-200ms)
- STANDARD → Full hybrid retrieval (500-800ms)
- COMPLEX → Hybrid + graph + RAPTOR (1-2s)
- ANALYTICAL → FLARE active retrieval (2-5s)

**Component C: Selective HyDE** — For COMPLEX/ANALYTICAL queries only, generate hypothetical answer via LLM and use its embedding for vector search. +18-29% NDCG improvement.

**End-to-end query flow:**
```
Query → Embed (20ms) → Cache Check (5ms) → Route (5ms) → [HyDE if complex] → Retrieval → Rerank → Parent Expand → LLM → Cache Store
```

---

## Phase 1: Agentic Acquisition (`src/acquisition/`) — COMPLETE

Three agents orchestrated by Celery:

**SourceDiscoveryAgent** — Maintains registry of Indian legal data sources. Probes for URL structure changes. Discovers new sources when tribunals are created.

**ChangeDetectionAgent** — Runs on schedule (daily for courts, weekly for gazette). Compares site state against last crawl. Outputs a queue of documents needing ingestion.

**LegalReviewAgent** — Classifies each new document (statute/judgment/notification/etc), extracts preliminary metadata, flags issues (scanned PDF, regional language, corruption).

Key sources: Indian Kanoon (HTML), India Code (HTML/PDF), Supreme Court (PDF), 25 High Courts (PDF/HTML), Gazette of India (PDF), SEBI/RBI/TRAI circulars.

Store source manifest in `configs/sources.yaml`. Version-controlled so we can track when source structures change.

---

## Phase 2: Parsing (`src/parsing/`) — COMPLETE

**Primary:** Docling (IBM, MIT License). Runs locally. Handles PDF (layout analysis + TableFormer), HTML (BeautifulSoup), scanned PDFs (Granite-Docling VLM for OCR).

**Fallback:** LlamaParse ($0.003/page) for high-volume batch. Tesseract for degraded scans.

**Quality validation after every document:**
1. Text completeness (extracted length vs expected from page count)
2. Section number sequence check (no gaps)
3. Table integrity (row/column alignment)
4. OCR confidence threshold (flag < 85%)

Output: `ParsedDocument` Pydantic model with raw text + structural markers.

---

## Phase 3: Chunking (`src/chunking/`)

**Tiered routing by document type AND condition** (not just type):

| Condition | Strategy | Chunk Boundary |
|-----------|----------|----------------|
| Well-structured statute | Structure-boundary | Section (with sub-sections, provisos, explanations) |
| Well-structured judgment | Structural | Facts / Issues / Per-issue reasoning / Holding / Order |
| Definitions section (S.2/3) | Proposition-based (LLM) | Each definition = atomic self-contained chunk |
| Partial structure | Recursive Semantic (RSC) | Structural split + semantic merge/split validation |
| Degraded scan (OCR < 80%) | Page-level | One page = one chunk, flagged for manual review |
| Unstructured text | Semantic (Max-Min) | Similarity-drop sentence boundaries |

**Max chunk size:** 1500 tokens. If a section exceeds this, split at sub-section boundaries.
**Overlap:** Not used for structure-boundary chunks (boundaries are semantic). 10-20% for semantic/RSC chunks.

**Enhancement layers (applied on top of base chunks):**
- **RAPTOR trees:** Per-Act recursive summary hierarchy. Level 0 = Act summary, Level 1 = Chapter summaries, Level 2 = Section chunks. Rebuilt per-Act when amendments arrive.
- **QuIM-RAG:** Pre-generate 3-5 questions per chunk using Claude Haiku. Store question embeddings as additional retrieval pathway.

Output: `LegalChunk` Pydantic model. See `docs/metadata_schema.md`.

Full strategy details in `docs/chunking_strategies.md`.

---

## Phase 4: Enrichment (`src/enrichment/`)

Two complementary techniques applied to ALL chunks:

**Contextual Retrieval (Anthropic technique):**
- For each chunk, call Claude Haiku to generate a 2-3 sentence context prefix
- Context includes: Act name, section number, chapter, legal concept, court
- USE PROMPT CACHING: cache full document text, only vary per-chunk query
- Enriched text used for BM25 index (improves keyword matching)
- Cost: ~$1.02 per million document tokens with prompt caching

**Late Chunking (Jina AI technique):**
- Embed entire document through long-context model (jina-embeddings-v3, 8192 tokens)
- THEN split token embeddings at chunk boundaries
- Mean pool each chunk's tokens → chunk embedding
- These embeddings carry inter-chunk context that naive embedding misses
- Cost: near-zero marginal (same compute as normal embedding, different order)

**Use both:** Late Chunking for vector embeddings, Contextual Retrieval for BM25 text.

---

## Phase 5: Embedding & Indexing (`src/embedding/`)

**Embedding model:** Fine-tuned BGE-m3 (multilingual, handles Hindi+English+legal Latin).
Fine-tuning uses QLoRA + NEFTune + DPO. Details in `docs/embedding_fine_tuning.md`.

**Dual Matryoshka Indexing:**

Store each chunk at two resolutions in Qdrant using named vectors:

```python
qdrant_client.create_collection(
    collection_name="legal_chunks",
    vectors_config={
        "fast": VectorParams(size=64, distance=Distance.COSINE),   # Broad candidate search
        "full": VectorParams(size=768, distance=Distance.COSINE)   # Precision re-scoring
    }
)

# Per chunk:
qdrant_client.upsert(
    collection_name="legal_chunks",
    points=[{
        "id": chunk.chunk_id,
        "vector": {
            "full": full_embedding.tolist(),      # 768-dim from Late Chunking
            "fast": full_embedding[:64].tolist()   # 64-dim Matryoshka slice
        },
        "payload": chunk.metadata  # Including parent_chunk_id, sibling_chunk_ids
    }]
)
```

**Indexing targets:**
1. **Qdrant** — Dual dense vectors (64-dim fast + 768-dim full) + BM25 sparse index (from contextualized text)
2. **Qdrant** — QuIM-RAG question embeddings (separate `quim_questions` collection, linked by chunk_id)
3. **Redis** — Parent document text store (key = parent_chunk_id, value = parent text + metadata)
4. **Neo4j** — Knowledge graph nodes/relationships

**Storage estimates:**
| Component | Size |
|-----------|------|
| 768-dim vectors (~3M chunks) | ~9.2 GB |
| 64-dim vectors (~3M chunks) | ~768 MB |
| Redis parent text store (~100K parents) | ~200 MB |
| **Total** | **~10.2 GB** |

---

## Phase 6: Knowledge Graph (`src/knowledge_graph/`)

Neo4j with custom schema for Indian law. Full schema in `docs/knowledge_graph_schema.md`.

Key capabilities this enables:
- **Point-in-time retrieval:** "What was Section 420 IPC as on January 1, 2020?"
- **Amendment cascading:** When new amendment ingested, auto-identify affected chunks
- **Citation traversal:** "Find all SC judgments interpreting Section 498A IPC"
- **Hierarchy navigation:** "Which sections fall under Chapter XVII of IPC?"
- **Direct query path:** SIMPLE queries routed by Phase 0 are answered by KG alone

---

## Phase 7: Retrieval (`src/retrieval/`)

**Retrieval pipeline (updated with funnel + parent expansion):**

```
Query Embedding (768-dim)
    │
    ├── Matryoshka Funnel Search:
    │   Stage 1: 64-dim fast search → top 1,000 candidates (15-20ms)
    │   Stage 2: 768-dim re-score candidates → top 100 (5-10ms)
    │
    ├── BM25 search on contextualized text → top 100
    │
    ├── QuIM search on quim_questions collection → top 50
    │
    ├── Graph traversal (Neo4j) → related chunks
    │
    └── Reciprocal Rank Fusion → Deduplicate → Top 150
        │
        └── Cross-encoder reranking (BGE-reranker-v2-m3) → Top 20
            │
            └── Parent-Document Context Expansion:
                ├── Fetch parent text from Redis for sub-section matches
                ├── Always include judgment header for judgment matches
                ├── Token-budget aware (max 30K context tokens)
                └── Deduplicate across matches
                    │
                    └── Pass to LLM with expanded context
```

**For ANALYTICAL queries only — FLARE Active Retrieval:**
- Generate response in 300-token segments with logprobs
- When token confidence drops below threshold → trigger targeted re-retrieval
- Cap at 5 additional retrievals per response
- Adds 200-500ms per trigger, 2-5s total for analytical queries

---

## Phase 8: Hallucination Mitigation (`src/hallucination/`)

Five sub-layers. Full details in `docs/hallucination_mitigation.md`.

1. **Citation verification** — Every case/section in LLM output verified against KG
2. **Temporal consistency** — Check all referenced laws are currently in force
3. **Confidence scoring** — Weighted score from retrieval relevance, source authority, citation verification rate
4. **GenGround verification** — Per-claim re-retrieval + alignment scoring. Tiered: GenGround for STANDARD+ queries, basic single-pass for SIMPLE queries
5. **Finetune-RAG** (Phase 3+) — Fine-tune generation model to resist hallucination, using synthetic adversarial training data + DPO

---

## Phase 9: Quality Assurance (`src/evaluation/`)

**Automated (RAGAS):** Context recall > 0.90, context precision > 0.85, faithfulness > 0.95, citation accuracy > 0.98, temporal accuracy > 0.99.

**Latency targets:**
| Query Type | TTFT Target |
|------------|-------------|
| SIMPLE | <200ms |
| STANDARD | <800ms |
| COMPLEX | <2s |
| ANALYTICAL | <5s |

**Query Intelligence metrics:** Cache hit rate > 30%, routing accuracy > 90%, GenGround claim verification rate > 95%.

**Human (lawyer evaluation):** 200 queries across 5 practice areas. Target 85%+ accuracy before launch.

Details in `docs/evaluation_framework.md`.

---

## Infrastructure Co-Location Requirement

At deployment, ALL components must be in the same availability zone:

```
Same machine or same AZ cluster:
├── Qdrant (vector search)
├── Neo4j (knowledge graph)
├── Redis (parent document store + semantic cache)
├── Embedding model (for query embedding at search time)
├── Reranker model (BGE-reranker-v2-m3)
└── Application server (Celery workers, API)

External API (only component that crosses network boundary):
└── LLM API (Claude) — for generation and hallucination mitigation
```

Inter-service latency target: <1ms between Qdrant/Neo4j/Redis/embedding model. This is achievable on a single server or within a cloud provider's AZ.
