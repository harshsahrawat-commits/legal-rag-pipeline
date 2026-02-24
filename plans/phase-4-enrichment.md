# Phase 4: Enrichment — Implementation Plan

## Context

Phases 1-3 (Acquisition, Parsing, Chunking) are complete with 534 tests. Phase 3 outputs `LegalChunk` JSON arrays to `data/chunks/{source}/{doc_id}.json`. Phase 4 enriches these chunks with LLM-generated context for better retrieval.

The v2 architecture docs confirm Phase 4 design is unchanged: two LLM-powered text enrichment stages plus Late Chunking (deferred — see below).

## Scope

**In scope (Phase 4):**
1. **Contextual Retrieval** — For each chunk, call Claude Haiku to generate a 2-3 sentence context prefix situating it within the full document. Uses Anthropic prompt caching (cache full doc text, vary per chunk). Populates `chunk.contextualized_text` and `chunk.ingestion.contextualized = True`. Used for BM25 keyword search.
2. **QuIM-RAG** — For each chunk, generate 3-5 practical lawyer questions. Store as sidecar `.quim.json` file. Update `chunk.ingestion.quim_questions` count. Phase 5 will embed these questions into a separate Qdrant collection.

**Deferred to Phase 5 (Embedding & Indexing):**
- **Late Chunking** — This is fundamentally an embedding operation (embed full doc → split token embeddings at chunk boundaries → mean pool). It requires the actual embedding model (BGE-m3/jina) and produces vectors that go directly to Qdrant. Inseparable from Phase 5.

## Data Flow

```
Input:
  data/chunks/{source}/{doc_id}.json     — list[LegalChunk] (from Phase 3)
  data/parsed/{source}/{doc_id}.json     — ParsedDocument (full doc text for prompts)

Processing:
  Stage 1: ContextualRetrievalEnricher   — Claude Haiku + prompt caching
  Stage 2: QuIMRagEnricher               — Claude Haiku + prompt caching

Output:
  data/enriched/{source}/{doc_id}.json      — list[LegalChunk] with enriched fields
  data/enriched/{source}/{doc_id}.quim.json — QuIMDocument (questions per chunk)
```

## Module Structure

```
src/enrichment/
├── __init__.py              # Public API exports
├── __main__.py              # Module runner
├── _config.py               # YAML config loader
├── _exceptions.py           # EnrichmentError hierarchy
├── _models.py               # Pydantic models
├── pipeline.py              # EnrichmentPipeline orchestrator
├── run.py                   # CLI entry point
└── enrichers/
    ├── __init__.py
    ├── _base.py             # BaseEnricher ABC
    ├── _contextual.py       # ContextualRetrievalEnricher
    └── _quim.py             # QuIMRagEnricher

configs/enrichment.yaml      # Enrichment config
tests/enrichment/            # 10 test files, ~191 tests
```

## Key Design Decisions

1. **AsyncAnthropic client** — All LLM calls are I/O-bound. Use `anthropic.AsyncAnthropic()` with `asyncio.Semaphore(concurrency)` for concurrent chunk processing within a document.

2. **Prompt caching (CRITICAL for cost)** — Full document text in system message with `cache_control: {"type": "ephemeral"}`. Only the per-chunk query varies. Drops cost ~90%.

3. **Document windowing** — Indian statutes can exceed Haiku's 200K context. Documents beyond `context_window_tokens` are split into overlapping windows. Chunks are grouped by window; each window group shares one cached system prompt.

4. **Separate output directory** (`data/enriched/`) — Preserves original Phase 3 output. Follows "each stage is independent and idempotent" principle.

5. **QuIM questions as sidecar file** — `.quim.json` stored separately so Phase 5 can embed them independently without re-parsing the full chunk file.

6. **Per-chunk error isolation** — If one LLM call fails, remaining chunks still get enriched. Failed chunks retain `contextualized=False` / `quim_questions=0`.

## Subtask Breakdown

### Subtask 1: Foundation — Exceptions, Models, Config (~46 tests)

**Files:**
- `src/enrichment/_exceptions.py` — `EnrichmentError(LegalRAGError)`, `ContextualRetrievalError`, `QuIMGenerationError`, `EnricherNotAvailableError`, `LLMRateLimitError`, `DocumentTextTooLargeError`
- `src/enrichment/_models.py` — `EnrichmentSettings`, `EnrichmentConfig`, `EnrichmentResult`, `QuIMEntry`, `QuIMDocument`
- `src/enrichment/_config.py` — `load_enrichment_config()` (mirrors `load_chunking_config`)
- `configs/enrichment.yaml` — settings with `input_dir`, `output_dir`, `parsed_dir`, `model`, `concurrency`, `quim_questions_per_chunk`, `context_window_tokens`

**Tests:** `test_exceptions.py` (8), `test_models.py` (30), `test_config.py` (8)

**Dependencies:** None

### Subtask 2: BaseEnricher + ContextualRetrievalEnricher (~43 tests)

**Files:**
- `src/enrichment/enrichers/_base.py` — `BaseEnricher` ABC with `async enrich_document(chunks, parsed_doc) -> list[LegalChunk]` and `stage_name` property
- `src/enrichment/enrichers/_contextual.py` — Full implementation:
  - Lazy `AsyncAnthropic` init via `_ensure_client()`
  - Prompt: system = `<document>{full_text}</document>` with `cache_control`, user = `<chunk>{text}</chunk>`
  - Document windowing for >180K token docs
  - Concurrency via `asyncio.Semaphore`
  - Retry with exponential backoff for rate limits
  - Sets `chunk.contextualized_text = f"{context}\n\n{chunk.text}"` and `chunk.ingestion.contextualized = True`
- `src/enrichment/enrichers/__init__.py`

**Tests:** `test_base_enricher.py` (8), `test_contextual_enricher.py` (35) — all LLM calls mocked via `AsyncMock`

**Dependencies:** Subtask 1

### Subtask 3: QuIMRagEnricher (~30 tests)

**Files:**
- `src/enrichment/enrichers/_quim.py` — Same prompt caching pattern. Generates N questions per chunk. Accumulates `QuIMDocument` during `enrich_document()`. Exposed via `get_quim_document()`.
  - Prompt: includes Act name / case citation + section reference for context
  - Parses LLM output: one question per line, filters noise
  - Sets `chunk.ingestion.quim_questions = len(questions)`

**Tests:** `test_quim_enricher.py` (30) — all LLM calls mocked

**Dependencies:** Subtasks 1, 2

### Subtask 4: Pipeline Orchestrator (~35 tests)

**Files:**
- `src/enrichment/pipeline.py` — `EnrichmentPipeline`:
  - Constructor loads config, builds enrichers
  - `async run(source_name=None, stage=None, dry_run=False) -> EnrichmentResult`
  - Stage selection: `None` = both, `"contextual_retrieval"`, `"quim_rag"`
  - Discovery: scans `data/chunks/{source}/*.json`
  - Per-document: load chunks + parsed doc → run enrichers → save enriched chunks + quim file
  - Idempotency: skip if output already exists
  - Error isolation: per-document try/except
  - Helper functions: `_load_chunks()`, `_save_enriched_chunks()`, `_save_quim_document()`, `_load_parsed_document()`

**Tests:** `test_pipeline.py` (35) — mock enrichers, test discovery/routing/error handling/idempotency

**Dependencies:** Subtasks 1-3

### Subtask 5: CLI + Integration Tests (~37 tests)

**Files:**
- `src/enrichment/run.py` — CLI: `--source`, `--stage`, `--dry-run`, `--log-level`, `--console-log`, `--config`
- `src/enrichment/__main__.py` — `from src.enrichment.run import main; main()`
- `src/enrichment/__init__.py` — exports `EnrichmentConfig`, `EnrichmentPipeline`, `EnrichmentResult`
- `tests/enrichment/conftest.py` — shared fixtures (sample chunks, parsed docs, mock LLM, tmp dirs)

**Tests:** `test_run.py` (12), `test_integration.py` (25) — end-to-end with mocked LLM: statute+judgment enrichment, both stages, idempotency, error isolation, JSON round-trip

**Dependencies:** All previous subtasks

## Test Summary

| Subtask | Tests |
|---------|-------|
| 1: Foundation | ~46 |
| 2: Contextual Retrieval | ~43 |
| 3: QuIM-RAG | ~30 |
| 4: Pipeline | ~35 |
| 5: CLI + Integration | ~37 |
| **Total Phase 4** | **~191** |
| **Project total** | **~725** (534 + 191) |

## Acceptance Criteria

1. `python -m ruff check src/ tests/` — zero warnings
2. `python -m pytest tests/ -x` — all ~725 tests pass
3. `python -m src.enrichment.run --dry-run` — exits 0, discovers chunk files
4. `python -m src.enrichment.run --stage=contextual_retrieval --dry-run` — stage filtering works
5. Enriched chunk files have `contextualized_text` populated and `ingestion.contextualized = True`
6. QuIM sidecar files contain valid `QuIMDocument` with questions per chunk
7. Second run on same input skips all documents (idempotency)
8. Single LLM failure doesn't crash pipeline (error isolation)
9. No real LLM API calls in any test

## Critical Reference Files

- `src/chunking/pipeline.py` — pipeline orchestration pattern to mirror
- `src/chunking/chunkers/_proposition.py` — Anthropic client pattern (adapt to async)
- `src/chunking/_models.py` — `LegalChunk` fields Phase 4 populates
- `src/parsing/_models.py` — `ParsedDocument` model for loading full doc text
- `docs/enrichment_guide.md` — exact prompt templates and caching strategy
- `tests/chunking/test_proposition.py` — mock Anthropic pattern (adapt to `AsyncMock`)

## Verification

After implementation:
```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/
python -m pytest tests/ -x -v
python -m pytest tests/enrichment/ -v        # Phase 4 tests only
python -m src.enrichment.run --dry-run       # Smoke test CLI
```
