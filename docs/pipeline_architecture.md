# Pipeline Architecture — 8-Phase Ingestion System

## Overview

```
Raw Documents → [1. Acquisition] → [2. Parsing] → [3. Chunking] → [4. Enrichment] → [5. Embedding & Indexing] → [6. Knowledge Graph] → [7. Hallucination Mitigation] → [8. Quality Assurance]
```

Each phase is a standalone module. Phases communicate via filesystem (parsed files, chunk JSONs) or message queue (Redis/Celery for async). Every phase is idempotent — safe to re-run.

## Phase 1: Agentic Acquisition (`src/acquisition/`)

Three agents orchestrated by Celery:

**SourceDiscoveryAgent** — Maintains registry of Indian legal data sources. Probes for URL structure changes. Discovers new sources when tribunals are created.

**ChangeDetectionAgent** — Runs on schedule (daily for courts, weekly for gazette). Compares site state against last crawl. Outputs a queue of documents needing ingestion.

**LegalReviewAgent** — Classifies each new document (statute/judgment/notification/etc), extracts preliminary metadata, flags issues (scanned PDF, regional language, corruption).

Key sources: Indian Kanoon (HTML), India Code (HTML/PDF), Supreme Court (PDF), 25 High Courts (PDF/HTML), Gazette of India (PDF), SEBI/RBI/TRAI circulars.

Store source manifest in `configs/sources.yaml`. Version-controlled so we can track when source structures change.

## Phase 2: Parsing (`src/parsing/`)

**Primary:** Docling (IBM, MIT License). Runs locally. Handles PDF (layout analysis + TableFormer), HTML (BeautifulSoup), scanned PDFs (Granite-Docling VLM for OCR).

**Fallback:** LlamaParse ($0.003/page) for high-volume batch. Tesseract for degraded scans.

**Quality validation after every document:**
1. Text completeness (extracted length vs expected from page count)
2. Section number sequence check (no gaps)
3. Table integrity (row/column alignment)
4. OCR confidence threshold (flag < 85%)

Output: `ParsedDocument` Pydantic model with raw text + structural markers.

## Phase 3: Chunking (`src/chunking/`)

**Route by document type:**

| Document Type | Strategy | Chunk Boundary |
|---|---|---|
| Statute/Act | Structure-boundary | Section (with sub-sections, provisos, explanations) |
| Judgment | Structural | Facts / Issues / Per-issue reasoning / Holding / Order |
| Notification/Circular | Semantic (Max-Min) | Similarity-drop sentence boundaries |
| Schedule/Table | Page-level | Full table or schedule entry |

**Max chunk size:** 1500 tokens. If a section exceeds this, split at sub-section boundaries.
**Overlap:** Not used for structure-boundary chunks (boundaries are semantic). 10-20% for semantic chunks.

**Enhancement layers (applied on top of base chunks):**
- **RAPTOR trees:** Recursive summary hierarchy per Act. Level 0 = Act summary, Level 1 = Chapter summaries, Level 2 = Section chunks.
- **QuIM-RAG:** Pre-generate 3-5 questions per chunk using Claude Haiku. Store question embeddings as additional retrieval pathway.

Output: `LegalChunk` Pydantic model. See `docs/metadata_schema.md`.

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

## Phase 5: Embedding & Indexing (`src/embedding/`)

**Embedding model:** Fine-tuned BGE-m3 (multilingual, handles Hindi+English+legal Latin).
Fine-tuning details in `docs/embedding_fine_tuning.md`.

**Indexing targets:**
1. **Qdrant** — Dense vectors (from Late Chunking) + BM25 sparse index (from contextualized text)
2. **Qdrant** — QuIM-RAG question embeddings (separate collection, linked by chunk_id)
3. **Neo4j** — Knowledge graph nodes/relationships

**Retrieval pipeline:**
```
Query → Dense search (top 100) + BM25 search (top 100) + QuIM search (top 50) + Graph traversal
      → Reciprocal Rank Fusion → Deduplicate → Top 150
      → Cross-encoder reranking (BGE-reranker-v2-m3) → Top 20
      → Pass to LLM
```

## Phase 6: Knowledge Graph (`src/knowledge_graph/`)

Neo4j with custom schema for Indian law. Full schema in `docs/knowledge_graph_schema.md`.

Key capabilities this enables:
- **Point-in-time retrieval:** "What was Section 420 IPC as on January 1, 2020?"
- **Amendment cascading:** When new amendment ingested, auto-identify affected chunks
- **Citation traversal:** "Find all SC judgments interpreting Section 498A IPC"
- **Hierarchy navigation:** "Which sections fall under Chapter XVII of IPC?"

## Phase 7: Hallucination Mitigation (`src/hallucination/`)

Five sub-layers. Full details in `docs/hallucination_mitigation.md`.

1. **Citation verification** — Every case/section in LLM output verified against KG
2. **Temporal consistency** — Check all referenced laws are currently in force
3. **Confidence scoring** — Weighted score from retrieval relevance, source authority, citation verification rate
4. **Grounded refinement** — Post-generation pass that cross-references claims against retrieved chunks
5. **Finetune-RAG** (Phase 2) — Fine-tune generation model to resist hallucination from noisy retrieval

## Phase 8: Quality Assurance (`src/evaluation/`)

**Automated (RAGAS):** Context recall > 0.90, context precision > 0.85, faithfulness > 0.95, citation accuracy > 0.98, temporal accuracy > 0.99.

**Human (lawyer evaluation):** 200 queries across 5 practice areas. Target 85%+ accuracy before launch.

Details in `docs/evaluation_framework.md`.
