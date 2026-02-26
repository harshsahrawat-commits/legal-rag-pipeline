# Phase 7: Retrieval — Implementation Plan

## Context

Phases 1-6 (the ingestion pipeline) are complete with 1052 tests. Phase 7 is the first **query-time** module — it processes individual queries, not batch documents. It sits between Phase 0 (Query Intelligence, not yet built) and Phase 8 (Hallucination Mitigation).

The retrieval module must:
1. Accept a user query and search across 4 parallel channels (dense vectors, BM25 sparse, QuIM questions, knowledge graph)
2. Fuse results via Reciprocal Rank Fusion → rerank with a cross-encoder → expand context from Redis
3. Expose a reusable `RetrievalEngine` that Phase 8 can call per-claim via `hybrid_search()`
4. Support FLARE active retrieval for analytical queries (optional, last subtask)

## Module Structure

```
src/retrieval/
├── __init__.py          # Public API: RetrievalConfig, RetrievalEngine, RetrievalPipeline, RetrievalResult
├── __main__.py          # python -m src.retrieval
├── run.py               # CLI: --query, --queries-file, --interactive, --dry-run, --config
├── pipeline.py          # RetrievalPipeline: batch/interactive orchestrator
├── _config.py           # load_retrieval_config() from YAML
├── _models.py           # QueryRoute, RetrievalQuery, ScoredChunk, FusedChunk, ExpandedContext, RetrievalResult, RetrievalSettings, RetrievalConfig
├── _exceptions.py       # RetrievalError hierarchy (7 exception types)
├── _engine.py           # RetrievalEngine: single-query handler, reusable by Phase 8
├── _searchers.py        # DenseSearcher, SparseSearcher, QuIMSearcher, GraphSearcher
├── _fusion.py           # ReciprocalRankFusion
├── _reranker.py         # CrossEncoderReranker (lazy transformers import)
├── _expander.py         # ParentDocumentExpander (Redis)
└── _flare.py            # FLAREActiveRetriever (optional, ANALYTICAL only)

configs/retrieval.yaml
tests/retrieval/         # ~215 tests across 10 test files
```

## Data Flow

```
Input: RetrievalQuery (text + optional embedding + route)
  │
  ├─ Phase 7 embeds query if no embedding provided (reuses LateChunkingEmbedder.embed_texts)
  │
  ├─ SIMPLE route → KG direct query via QueryBuilder → return
  │
  ├─ STANDARD/COMPLEX/ANALYTICAL:
  │   ├── Dense Matryoshka Funnel: 64-dim "fast" → top 1000 → 768-dim "full" rescore → top 100
  │   ├── BM25 Sparse: Qdrant sparse search on "bm25" named vector → top 100
  │   ├── QuIM: search quim_questions collection → map source_chunk_id → top 50
  │   ├── Graph Traversal (COMPLEX/ANALYTICAL only): Neo4j QueryBuilder → related chunk IDs
  │   └── Reciprocal Rank Fusion (deduplicate + fuse scores) → top 150
  │       └── Cross-Encoder Reranking (BGE-reranker-v2-m3) → top 20
  │           └── Parent-Document Expansion (Redis parent:{id}) → max 30K tokens
  │               └── [FLARE re-retrieval if ANALYTICAL and low confidence]
  │
Output: RetrievalResult (ranked ExpandedContext chunks + timings + metadata)
```

## Key Data Models (`_models.py`)

**QueryRoute** — `StrEnum`: SIMPLE, STANDARD, COMPLEX, ANALYTICAL

**RetrievalQuery** — Input to engine:
- `text: str`, `query_embedding: list[float] | None`, `query_embedding_fast: list[float] | None`
- `sparse_vector: SparseVector | None`, `route: QueryRoute = STANDARD`
- `hyde_text: str | None`, `metadata_filters: dict | None`, `reference_date: date | None`
- `max_results: int = 20`, `max_context_tokens: int = 30_000`

**ScoredChunk** — Intermediate per-channel result:
- `chunk_id: str`, `text: str`, `score: float`, `channel: str`, `payload: dict`

**FusedChunk** — Post-RRF result:
- `chunk_id: str`, `text: str`, `rrf_score: float`, `rerank_score: float | None`
- `channels: list[str]`, `payload: dict`

**ExpandedContext** — Final output chunk:
- `chunk_id: str`, `chunk_text: str`, `parent_text: str | None`
- `judgment_header_text: str | None`, `total_tokens: int`, `metadata: dict`

**RetrievalResult** — Full query result:
- `query: RetrievalQuery`, `route: QueryRoute`, `chunks: list[ExpandedContext]`
- `total_context_tokens: int`, `search_channels_used: list[str]`
- `timings: dict[str, float]`, `kg_direct_answer: dict | None`
- `flare_retrievals: int = 0`, `errors: list[str]`

## Core Architecture: Engine vs Pipeline

**RetrievalEngine** (`_engine.py`) — the reusable single-query handler:
- `retrieve(query: RetrievalQuery) -> RetrievalResult` — full pipeline for one query
- `hybrid_search(text, top_k=5) -> list[ScoredChunk]` — lightweight search for Phase 8 GenGround (dense + BM25 + RRF only, no rerank/expand)
- `kg_direct_query(query_text) -> dict | None` — SIMPLE route, KG only
- `load_models()` — load embedding model, reranker, BM25 vocab

**RetrievalPipeline** (`pipeline.py`) — batch/interactive orchestrator:
- `run(queries=None, queries_file=None, interactive=False) -> list[RetrievalResult]`
- Wraps `RetrievalEngine` for CLI usage

This separation ensures Phase 8 can call `engine.hybrid_search(claim)` without the pipeline overhead.

## Critical Integration Points

| Service | Phase 7 Usage | Interface | File |
|---------|--------------|-----------|------|
| Qdrant (sync client) | Dense, sparse, QuIM search | `QdrantClient.search()` | `src/embedding/_qdrant_indexer.py` |
| Embedder | Query embedding (768d) + Matryoshka slice (64d) | `LateChunkingEmbedder.embed_texts()`, `.matryoshka_slice()` | `src/embedding/_embedder.py` |
| BM25 | Query sparse encoding | `BM25SparseEncoder.encode()` + new `load_vocabulary()` | `src/embedding/_sparse.py` |
| Redis (async) | Parent chunk expansion | `redis.asyncio.get(parent:{id})` → JSON | `src/embedding/_redis_store.py` |
| Neo4j (async) | Graph traversal | `QueryBuilder.citation_traversal()`, etc. | `src/knowledge_graph/_queries.py` |
| Reranker | Cross-encoder scoring | `AutoModelForSequenceClassification` (lazy) | New: `_reranker.py` |

## Change to Existing Code

**One additive change to Phase 5** — `src/embedding/_sparse.py`:
- Add `save_vocabulary(path: Path) -> None` — serialize vocab, IDF, avg_dl to JSON
- Add `load_vocabulary(cls, path: Path) -> BM25SparseEncoder` — classmethod to load pre-built vocab
- ~30 lines, backward-compatible, 5 new tests in `tests/embedding/test_sparse.py`

---

## Subtask Breakdown (7 subtasks)

### Subtask 1: Foundation — Exceptions, Models, Config (~35 tests)

**Create:**
- `src/retrieval/_exceptions.py` — 7 exception classes: `RetrievalError`, `SearchError`, `RerankerError`, `RerankerNotAvailableError`, `ContextExpansionError`, `FLAREError`, `SearchNotAvailableError`
- `src/retrieval/_models.py` — `QueryRoute`, `RetrievalQuery`, `ScoredChunk`, `FusedChunk`, `ExpandedContext`, `RetrievalResult`, `RetrievalSettings`, `RetrievalConfig`
- `src/retrieval/_config.py` — `load_retrieval_config()`
- `configs/retrieval.yaml`
- `src/retrieval/__init__.py`, `src/retrieval/__main__.py`
- `tests/retrieval/__init__.py`, `tests/retrieval/conftest.py`
- `tests/retrieval/test_exceptions.py`, `test_models.py`, `test_config.py`

**Tests:** Exception hierarchy (7), model validation + defaults (20), config loading (8)
**Depends on:** Nothing

### Subtask 2: Searchers — Dense, BM25, QuIM, Graph (~50 tests)

**Create:**
- `src/retrieval/_searchers.py` — `DenseSearcher`, `SparseSearcher`, `QuIMSearcher`, `GraphSearcher`
- `tests/retrieval/test_searchers.py` (~45 tests)

**Modify (additive):**
- `src/embedding/_sparse.py` — add `save_vocabulary()` + `load_vocabulary()` (~30 lines)
- `tests/embedding/test_sparse.py` — add ~5 vocab persistence tests

**Test coverage:**
- DenseSearcher: Matryoshka 2-stage funnel, empty results, Qdrant errors, missing dep
- SparseSearcher: BM25 sparse search via Qdrant, empty vector, errors
- QuIMSearcher: question collection search, source_chunk_id → parent chunk mapping
- GraphSearcher: section/act reference extraction from query text, KG traversal via QueryBuilder, chunk ID resolution from Qdrant
- BM25 vocab save/load roundtrip

**Mocking:** `_build_mock_qdrant_module()` (same pattern as `tests/embedding/test_qdrant_indexer.py`), mock `Neo4jClient.run_query()`
**Depends on:** Subtask 1

### Subtask 3: Reciprocal Rank Fusion (~20 tests)

**Create:**
- `src/retrieval/_fusion.py` — `ReciprocalRankFusion`
- `tests/retrieval/test_fusion.py`

**Test coverage:** Single channel, disjoint results, overlapping results (dedup + score aggregation), 3+ channels, top-K truncation, empty input, tied scores, RRF k parameter effect
**Depends on:** Subtask 1

### Subtask 4: Cross-Encoder Reranker (~25 tests)

**Create:**
- `src/retrieval/_reranker.py` — `CrossEncoderReranker` (lazy `transformers` import)
- `tests/retrieval/test_reranker.py`

**Test coverage:** Model loading, missing dependency → `RerankerNotAvailableError`, rerank scoring + ordering, batch processing, empty input, top-K truncation, model not loaded error, graceful fallback (skip reranking → use RRF scores)
**Mocking:** `patch.dict(sys.modules)` for transformers, mock model forward pass
**Depends on:** Subtask 1

### Subtask 5: Parent Document Expander (~25 tests)

**Create:**
- `src/retrieval/_expander.py` — `ParentDocumentExpander`
- `tests/retrieval/test_expander.py`

**Test coverage:** Fetches parent text (statute with parent_chunk_id), fetches judgment header, token budget enforcement, deduplication (same parent referenced by multiple chunks), Redis miss (graceful), Redis connection error, missing dependency, empty chunks, chunks without parent_info, token counting with tiktoken
**Mocking:** `AsyncMock` for Redis client, return serialized JSON matching `RedisParentStore._serialize_parent` format
**Depends on:** Subtask 1

### Subtask 6: RetrievalEngine + Pipeline + CLI (~40 tests)

**Create:**
- `src/retrieval/_engine.py` — `RetrievalEngine`
- `src/retrieval/pipeline.py` — `RetrievalPipeline`
- `src/retrieval/run.py` — CLI (`--query`, `--queries-file`, `--interactive`, `--dry-run`, `--config`)
- Update `src/retrieval/__init__.py`
- `tests/retrieval/test_engine.py` (~20), `test_pipeline.py` (~12), `test_run.py` (~8)

**Test coverage:**
- Engine: full retrieve flow, SIMPLE/STANDARD/COMPLEX routes, `hybrid_search()` simplified interface, `kg_direct_query()`, query embedding computed when not provided, channel error isolation, timings populated
- Pipeline: process query list, load from file, interactive stdin, dry-run, per-query error isolation
- CLI: `--query`, `--queries-file`, `--dry-run`, `--config`, error handling

**Depends on:** Subtasks 2, 3, 4, 5

### Subtask 7: FLARE Active Retrieval + Integration Tests (~20 tests)

**Create:**
- `src/retrieval/_flare.py` — `FLAREActiveRetriever` (optional, lazy anthropic import)
- `tests/retrieval/test_flare.py` (~12 tests)
- `tests/retrieval/test_integration.py` (~8 tests)

**Test coverage:**
- FLARE: segment generation, low-confidence detection, re-retrieval trigger, max retrieval cap (5), all-high-confidence (no re-retrieval), re-retrieval adds new chunks, FLARE disabled → skip, LLM API error → graceful fallback
- Integration: full pipeline e2e for each route (SIMPLE, STANDARD, COMPLEX, ANALYTICAL), hybrid_search used by simulated Phase 8 consumer, error isolation across components, multiple queries batch

**Mocking:** Mock anthropic AsyncAnthropic for FLARE, mock engine.hybrid_search for re-retrieval
**Depends on:** Subtask 6

## Dependency Graph

```
Subtask 1: Foundation
    │
    ├── Subtask 2: Searchers ──────┐
    ├── Subtask 3: RRF Fusion ─────┤
    ├── Subtask 4: Reranker ───────┼── Subtask 6: Engine + Pipeline + CLI
    └── Subtask 5: Expander ───────┘           │
                                               └── Subtask 7: FLARE + Integration
```

Subtasks 2-5 can be built in parallel after Subtask 1.

## Test Summary (Actual)

| Subtask | New Tests | Cumulative (Phase 7) | Project Total |
|---------|-----------|---------------------|---------------|
| 1: Foundation | 54 | 54 | 1106 |
| 2: Searchers + BM25 vocab | 52 | 106 | 1158 |
| 3: RRF Fusion | 29 | 135 | 1187 |
| 4: Reranker | 28 | 163 | 1215 |
| 5: Expander | 30 | 193 | 1245 |
| 6: Engine + Pipeline + CLI | 27 | 220 | 1272 |
| 7: FLARE + Integration | 35 | 255 | 1313 |

**All 7 subtasks COMPLETE. Committed `27601e1`, pushed.**

## Acceptance Criteria

1. `python -m ruff check src/ tests/` — zero warnings
2. `python -m pytest tests/ -x` — all ~1267 tests pass
3. `python -m src.retrieval.run --query "What is Section 420 IPC?" --dry-run` — exits 0
4. `RetrievalEngine.retrieve()` handles all 4 routes correctly (SIMPLE/STANDARD/COMPLEX/ANALYTICAL)
5. `RetrievalEngine.hybrid_search()` returns scored chunks (reusable by Phase 8)
6. Matryoshka funnel: 64-dim fast search narrows candidates, 768-dim full rescores
7. RRF fuses results from multiple channels with correct deduplication
8. Cross-encoder reranker reorders by relevance score (graceful fallback if unavailable)
9. Parent document expansion fetches from Redis, respects 30K token budget
10. SIMPLE queries go through KG only (no vector search)
11. Channel failure is isolated — one channel down does not crash retrieval
12. All external services (Qdrant, Redis, Neo4j, transformers) are lazy-imported and mocked in unit tests
13. BM25 vocabulary can be saved/loaded across sessions
14. FLARE triggers re-retrieval for ANALYTICAL queries, capped at 5

## Verification

```bash
# Lint + format
python -m ruff check src/retrieval/ tests/retrieval/ && python -m ruff format --check src/retrieval/ tests/retrieval/

# Unit tests (fast, no external services)
python -m pytest tests/retrieval/ -x -v

# Full project test suite
python -m pytest tests/ -x -v

# CLI smoke test
python -m src.retrieval.run --query "What is Section 420 of IPC?" --dry-run
python -m src.retrieval.run --queries-file data/eval/test_queries.json --dry-run
```
