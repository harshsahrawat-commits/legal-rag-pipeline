# Phase 0: Query Intelligence Layer — Implementation Plan

## Overview

A pre-retrieval layer (`src/query/`) that sits between the user's query and Phase 7 (Retrieval). Three stacked components decide *whether*, *how*, and *how deeply* to retrieve.

**Source docs:** `docs/query_intelligence.md`, `docs/evaluation_framework.md`

---

## Scope

### Module: `src/query/`

```
src/query/
├── __init__.py              # Public exports: QueryIntelligenceLayer, QueryConfig, etc.
├── __main__.py              # python -m src.query
├── run.py                   # CLI entry point (argparse)
├── _exceptions.py           # QueryIntelligenceError hierarchy
├── _models.py               # Pydantic models: CacheEntry, RouterResult, HyDEResult, QuerySettings, QueryConfig
├── _config.py               # YAML config loader for configs/query.yaml
├── _cache.py                # SemanticQueryCache (Qdrant cache collection + Redis response store)
├── _router.py               # AdaptiveQueryRouter (rule-based classifier → 4 routes)
├── _hyde.py                 # SelectiveHyDE (LLM-generated hypothetical answer embeddings)
└── pipeline.py              # QueryIntelligenceLayer orchestrator (cache → route → HyDE → build RetrievalQuery)
```

### Config: `configs/query.yaml`

```yaml
settings:
  # Cache
  cache_enabled: true
  cache_similarity_threshold: 0.92
  cache_ttl_seconds: 86400        # 24h default
  cache_short_ttl_seconds: 3600   # 1h for rapidly-changing areas
  cache_collection: "query_cache"
  cache_redis_prefix: "qcache:"

  # Router
  router_version: "v1_rule_based"

  # HyDE
  hyde_enabled: true
  hyde_model: "claude-haiku"
  hyde_max_tokens: 200
  hyde_routes: ["complex", "analytical"]  # Only trigger for these routes

  # Embedding (reuses Phase 7 settings for model, but needs its own for query embedding)
  embedding_model: "BAAI/bge-m3"
  embedding_dim: 768
  matryoshka_dim: 64
  device: "cpu"

  # External services
  qdrant_host: "localhost"
  qdrant_port: 6333
  redis_url: "redis://localhost:6379/0"
```

---

## Dependencies

### From previous phases (already built):
- `src/retrieval/_models.py` — `QueryRoute`, `RetrievalQuery` (Phase 0 produces these)
- `src/retrieval/_searchers.py` — `extract_section_references()` (reused by router)
- `src/embedding/_embedder.py` — `LateChunkingEmbedder` (query embedding, lazy import)
- `src/utils/_exceptions.py` — `LegalRAGError` base
- `src/utils/_logging.py` — `get_logger()`

### External services (lazy-imported, mocked in tests):
- **Qdrant** — `qdrant_client` for cache collection (vector similarity search)
- **Redis** — `redis` for cached response storage (key-value with TTL)
- **Anthropic** — `anthropic` for HyDE generation (Claude Haiku)

### No new pip dependencies required
All external deps (qdrant-client, redis, anthropic) are already used by Phases 5-8.

---

## Data Flow

```
User query (str)
    │
    ▼
QueryIntelligenceLayer.process(query_text)
    │
    ├── 1. Embed query → 768-dim full + 64-dim fast vectors
    │
    ├── 2. SemanticQueryCache.get(embedding)
    │       ├── HIT  → return cached response (CacheHitResult)
    │       └── MISS → continue
    │
    ├── 3. AdaptiveQueryRouter.classify(query_text)
    │       → QueryRoute (SIMPLE / STANDARD / COMPLEX / ANALYTICAL)
    │
    ├── 4. SelectiveHyDE.maybe_generate(query_text, route)
    │       ├── SIMPLE/STANDARD → no-op, use original embedding
    │       └── COMPLEX/ANALYTICAL → LLM generates hypothetical, re-embed → hyde_embedding
    │
    └── 5. Build RetrievalQuery(text, embedding, route, hyde_text)
            │
            ▼
        Return to caller → feeds into RetrievalEngine.retrieve()
```

**Output:** `RetrievalQuery` (already defined in Phase 7) with populated fields:
- `text`: original query
- `query_embedding`: 768-dim full vector (or HyDE embedding for COMPLEX/ANALYTICAL)
- `query_embedding_fast`: 64-dim Matryoshka slice
- `route`: classified query route
- `hyde_text`: hypothetical answer text (if generated)

**Cache store:** After Phase 7+8 completes, caller invokes `cache.set(query, response)` to store result.

---

## Subtask Breakdown

**STATUS: ALL SUBTASKS COMPLETE (2026-02-27)**
- Subtask 1: DONE (40 tests) — Foundation
- Subtask 2: DONE (62 tests) — Cache (built by cache-builder agent)
- Subtask 3: DONE (68 tests) — Router (built by router-builder agent)
- Subtask 4: DONE (28 tests) — HyDE
- Subtask 5: DONE (65 tests) — Pipeline + CLI + Integration
- Quality audit: DONE — 1 critical bug fixed (tuple unpacking in run.py), batch delete optimization, type annotation tightening
- **Total: 263 tests, 1757 project-wide, all passing**

### Subtask 1: Foundation (models, config, exceptions) — SEQUENTIAL FIRST
**Estimated tests:** ~50

Build the shared infrastructure all other subtasks depend on.

**Deliverables:**
- `src/query/_exceptions.py` — Exception hierarchy:
  - `QueryIntelligenceError(LegalRAGError)` — base
  - `CacheError(QueryIntelligenceError)` — cache read/write failures
  - `RouterError(QueryIntelligenceError)` — classification failures
  - `HyDEError(QueryIntelligenceError)` — HyDE generation failures
  - `EmbeddingError(QueryIntelligenceError)` — query embedding failures
- `src/query/_models.py` — Pydantic models:
  - `CacheEntry` — stored cache record (query_text, embedding, response, acts_cited, cached_at, ttl)
  - `CacheResult` — cache lookup result (hit: bool, response: dict | None, similarity: float, cache_key: str | None)
  - `RouterResult` — classification output (route: QueryRoute, confidence: float, signals: list[str])
  - `HyDEResult` — HyDE output (hypothetical_text: str | None, hyde_embedding: list[float] | None, generated: bool)
  - `QueryIntelligenceResult` — full layer output (query_text, route, cache_hit, hyde_generated, retrieval_query: RetrievalQuery, timings: dict)
  - `QuerySettings` — all config fields as Pydantic model
  - `QueryConfig` — root config model (`settings: QuerySettings`)
- `src/query/_config.py` — YAML config loader (parallel to Phase 7's `_config.py`)
- `configs/query.yaml` — default config file
- `src/query/__init__.py` — public exports
- `src/query/__main__.py` — `python -m src.query` entry point

**Tests:** Models validation, config loading (valid/invalid/missing YAML), exception hierarchy, default values, serialization round-trip.

---

### Subtask 2: Semantic Query Cache — PARALLELIZABLE (after Subtask 1)
**Estimated tests:** ~45

**Deliverables:**
- `src/query/_cache.py` — `SemanticQueryCache` class:
  - `__init__(settings: QuerySettings)` — lazy Qdrant + Redis clients
  - `get(query_embedding: list[float]) -> CacheResult` — Qdrant similarity search → Redis response fetch
  - `set(query_text, query_embedding, response, acts_cited) -> str` — store in Qdrant + Redis with TTL
  - `invalidate_for_act(act_name: str) -> int` — scroll Qdrant for matching `acts_cited`, delete from both stores. Returns count deleted.
  - `invalidate_by_key(cache_key: str) -> bool` — delete a specific entry
  - `clear() -> int` — flush entire cache (admin operation)
  - `_ensure_collection()` — create `query_cache` collection if not exists (768-dim, cosine)
  - `is_available: bool` — property, checks Qdrant+Redis importability
  - Error isolation: all operations wrapped, failures return CacheResult(hit=False) or log warning

**Tests:** Cache miss, cache hit at various thresholds, TTL expiry (mocked time), act-based invalidation, key-based invalidation, clear, collection creation, Qdrant/Redis unavailable graceful fallback, serialization of complex response objects, concurrent access patterns.

---

### Subtask 3: Adaptive Query Router — PARALLELIZABLE (after Subtask 1)
**Estimated tests:** ~50

**Deliverables:**
- `src/query/_router.py` — `AdaptiveQueryRouter` class:
  - `__init__(settings: QuerySettings)` — compile regex patterns once
  - `classify(query_text: str) -> RouterResult` — rule-based classification:
    - **SIMPLE patterns:** "what does section X", "define Y", "text of section", "read section", "show me section", "what is Article X"
    - **ANALYTICAL signals:** "compare", "contrast", "evolution", "trace", "all grounds", "all provisions", "every", "comprehensive", "interplay between", "relationship between", "how has .* been interpreted", "trace the .* jurisprudence"
    - **COMPLEX detection:** >1 act reference, >2 section references, cross-jurisdictional signals ("Delhi and Mumbai", multiple court names)
    - **Default:** STANDARD
  - Uses `extract_section_references()` from `src/retrieval/_searchers.py` (import it)
  - New: `extract_act_references(query_text) -> list[str]` — regex for act name extraction
  - `confidence` field: 1.0 for pattern match, 0.7 for heuristic, 0.5 for default
  - `signals` field: list of which patterns/heuristics triggered

**Tests:** 10+ queries per route category (40+ query classification tests), edge cases (ambiguous queries, Hindi text, mixed language), act reference extraction, section reference counting, confidence scoring, empty query, very long query.

**Note:** The legal-domain-expert agent should review the regex patterns and provide realistic test queries per category.

---

### Subtask 4: Selective HyDE — SEQUENTIAL (after Subtask 3, uses route)
**Estimated tests:** ~30

**Deliverables:**
- `src/query/_hyde.py` — `SelectiveHyDE` class:
  - `__init__(settings: QuerySettings)` — lazy anthropic import
  - `maybe_generate(query_text: str, route: QueryRoute) -> HyDEResult` — only for COMPLEX/ANALYTICAL
    - Prompt: "You are an expert Indian lawyer. Given this legal research question, write a brief (2-3 sentence) hypothetical answer using proper Indian legal terminology, Act names, and section numbers."
    - Returns HyDEResult with hypothetical_text and hyde_embedding (re-embedded via embedder)
  - `is_available: bool` — property, checks anthropic importability + hyde_enabled setting
  - Graceful fallback: if LLM fails, return HyDEResult(generated=False), use original embedding
  - Embed the hypothetical text using the same embedding model as main search

**Tests:** HyDE skipped for SIMPLE/STANDARD, HyDE generated for COMPLEX/ANALYTICAL, LLM error graceful fallback, disabled setting, anthropic not available, embedding of hypothetical text, prompt content validation, max_tokens respected.

---

### Subtask 5: QueryIntelligenceLayer + CLI + Integration Tests — SEQUENTIAL (after all)
**Estimated tests:** ~40

**Deliverables:**
- `src/query/pipeline.py` — `QueryIntelligenceLayer` class:
  - `__init__(settings: QuerySettings)` — creates cache, router, hyde instances
  - `async process(query_text: str, embedder=None) -> QueryIntelligenceResult`:
    1. Embed query (use provided embedder or try lazy load)
    2. Check cache → if hit, return early with CacheHitResult
    3. Classify route
    4. Maybe HyDE → get hyde_embedding
    5. Build and return RetrievalQuery + QueryIntelligenceResult with timings
  - `async store_response(query_text, query_embedding, response, acts_cited)` — post-retrieval cache store
  - `async invalidate_for_act(act_name)` — delegate to cache
  - `from_config(config_path) -> QueryIntelligenceLayer` — classmethod

- `src/query/run.py` — CLI:
  - `--query TEXT` — process single query
  - `--classify-only` — just show route classification (no cache/HyDE)
  - `--cache-stats` — show cache hit rate stats
  - `--invalidate-act ACT_NAME` — invalidate cache for an act
  - `--dry-run` — show what would happen without side effects
  - `--config PATH` — config file path
  - `--log-level` / `--console-log` — logging options

- Integration tests:
  - E2E: query → cache miss → classify → HyDE → RetrievalQuery built correctly
  - E2E: query → cache hit → early return
  - Cache store → re-query → cache hit
  - Act invalidation → re-query → cache miss
  - All 4 routes produce correct RetrievalQuery
  - Phase 7 integration: QueryIntelligenceLayer output feeds RetrievalEngine.retrieve()
  - Error isolation: each component failure doesn't crash the layer
  - CLI smoke tests

---

## Parallelization Strategy (Agent Team)

```
Timeline:
─────────────────────────────────────────────────────
 Lead builds Subtask 1 (Foundation)         ~30 min
─────────────────────────────────────────────────────
      ┌─── Agent A: Subtask 2 (Cache)      ~40 min ───┐
      │                                                 │
      ├─── Agent B: Subtask 3 (Router)     ~40 min ───┤  PARALLEL
      │                                                 │
      └─── legal-domain-expert: Review      ~10 min ───┘
                     router patterns
─────────────────────────────────────────────────────
 Lead merges + builds Subtask 4 (HyDE)     ~30 min
─────────────────────────────────────────────────────
 Lead builds Subtask 5 (Pipeline+CLI+Integ) ~40 min
─────────────────────────────────────────────────────
 quality-auditor reviews full module        ~10 min
─────────────────────────────────────────────────────
```

**Subtasks 2 and 3 are fully independent** — they share only the foundation (Subtask 1) and do not import each other. Each teammate gets an isolated worktree.

---

## Acceptance Criteria

1. **All tests pass:** `python -m pytest tests/query/ -v` — 0 failures
2. **Full project tests pass:** `python -m pytest tests/ -x` — 1494 + new tests = ~1700+ total
3. **Lint clean:** `python -m ruff check src/query/ tests/query/` — 0 errors
4. **Format clean:** `python -m ruff format --check src/query/ tests/query/`
5. **Cache correctness:** Hit at threshold 0.92, miss below, TTL expiry works, act invalidation works
6. **Router accuracy:** All 40+ test queries classified to expected routes
7. **HyDE selective:** Only triggers for COMPLEX/ANALYTICAL, graceful fallback on LLM error
8. **Integration:** QueryIntelligenceLayer.process() produces a valid RetrievalQuery that Phase 7 can consume
9. **No regressions:** Existing 1494 tests still pass
10. **Lazy imports:** qdrant_client, redis, anthropic all lazy-imported, module loads without them installed

---

## Key Design Decisions

1. **QueryRoute stays in `src/retrieval/_models.py`** — Phase 0 imports it, doesn't redefine. The router produces it, the engine consumes it.
2. **Reuse `extract_section_references`** from `src/retrieval/_searchers.py` — don't duplicate regex logic.
3. **New `extract_act_references`** lives in `src/query/_router.py` — act detection logic is router-specific.
4. **Cache miss is the safe default** — any cache failure returns miss, never stale data.
5. **HyDE replaces the query embedding for vector search only** — BM25 and generation still use the original query text.
6. **Pipeline creates layer instances per-call** — consistent with Phase 8 pattern.
7. **All external deps lazy-imported** — module loads and tests run without qdrant/redis/anthropic installed.
