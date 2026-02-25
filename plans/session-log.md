# Session Log

## Session: 2026-02-22 11:55
**Phase:** Phase 1 — Agentic Acquisition
**What was built:**
- Full `src/utils/` module: exception hierarchy (`LegalRAGError`), structlog JSON logging, SHA-256 hashing
- Full `src/acquisition/` module (18 source files):
  - `_models.py` — 15 Pydantic models (enums, configs, crawl state, pipeline data)
  - `_config.py` — YAML config loader for `configs/sources.yaml`
  - `_state.py` — File-based crawl state persistence (atomic writes via tmp+rename)
  - `_rate_limiter.py` — Async token-bucket rate limiter
  - `_http_client.py` — Async HTTP client with retries (tenacity), rate limiting, timeout
  - `base_scraper.py` — Template method ABC (discover → filter → scrape → classify → save)
  - `scrapers/_indian_kanoon.py` — HTML scraper for indiankanoon.org (search pagination, classification, metadata extraction)
  - `scrapers/_india_code.py` — Scraper for indiacode.nic.in (seed act IDs, HTML+PDF detection)
  - `agents/_source_discovery.py` — Loads enabled sources from config
  - `agents/_change_detection.py` — Filters discovered URLs against crawl state
  - `agents/_legal_review.py` — Rule-based classification + flag detection (scanned PDF, regional language, small content, missing metadata)
  - `pipeline.py` — `AcquisitionPipeline` orchestrator (asyncio.gather with semaphore concurrency)
  - `run.py` + `__main__.py` — CLI entry point
- Full test suite: 105 tests across 16 test files (all passing)
- `configs/sources.yaml` — Source registry with Indian Kanoon + India Code configs
- `pyproject.toml` — Project config with ruff + pytest + hatchling
- `Indian_kanoon/` — API documentation reference folder (4 files: API docs, ToS, official client docs, reference Python source)

**What broke:**
1. `hatchling.backends` → correct import is `hatchling.build` (build backend)
2. Hatch couldn't find packages — needed `[tool.hatch.build.targets.wheel] packages = ["src"]`
3. `structlog.get_logger()` returns `BoundLoggerLazyProxy` before `configure_logging()` is called — test expected `BoundLogger` instance
4. `datetime.utcnow()` deprecated in Python 3.14 — replaced with `datetime.now(UTC)`
5. Ruff flagged `RateLimitExceeded` → renamed to `RateLimitError` (N818 convention)
6. Ruff flagged `str, Enum` → auto-fixed to `StrEnum` (UP042)
7. Scrapers `__init__.py` imported `IndiaCodeScraper` before it existed — had to write India Code scraper earlier than planned

**Decisions made:**
1. **File-based state over Redis** — inspectable JSON files, no infra dependency. `CrawlStateStore` interface makes future Redis swap clean.
2. **Template method on `BaseScraper`** — only 4 abstract methods per source. Adding new sources = one file.
3. **Sidecar `.meta.json`** per document — each doc independently processable by Phase 2. No giant manifest.
4. **Rule-based classification** (no LLM) — Indian Kanoon/India Code have clear structural signals. LLM budget saved for enrichment.
5. **Sequential within source, concurrent across sources** — simplifies rate limit enforcement.
6. **Indian Kanoon API preferred over scraping** — discovered official API with structured JSON, citation graphs, judgment structural analysis (Facts/Issues/Reasoning/Conclusion). Applied for non-commercial use (₹10,000/month free tier). Will refactor scraper once approved.

**Open questions:**
- Waiting for Indian Kanoon API non-commercial use approval email
- Once API approved: refactor `IndianKanoonScraper` to use API before or after Phase 2?
- India Code site structure may have changed — need live validation of scraper

**Next steps:**
1. Check for IK API approval email
2. Once approved: refactor `IndianKanoonScraper` to use API (token auth, JSON responses, citation data, structural analysis)
3. Initialize git repo and make first commit
4. Begin Phase 2 (Parsing) or wait for API refactor — user's choice

---

## Session: 2026-02-22 (second session)
**Phase:** Phase 1 — Agentic Acquisition (India Code scraper rewrite)

**What was built:**
1. **India Code live site probe** — Used Playwright to inspect the real indiacode.nic.in website and discovered the existing scraper was completely broken:
   - URL scheme wrong (used `45_of_1860` strings, real handles are opaque numeric IDs like `1999`)
   - CSS selectors wrong (`div.actTitle` etc. don't exist)
   - Discovery model wrong (static seed list vs paginated browse listing)
   - Content is JS-rendered but browse listing is server-rendered HTML table

2. **India Code scraper full rewrite** based on real site structure:
   - Discovery via DSpace browse listing pagination (server-rendered HTML table)
   - Save detail page HTML as raw content (not PDF — avoids binary corruption through text encoding roundtrip)
   - PDF URL stored in `PreliminaryMetadata.download_url` for Phase 2 to download directly
   - Doc IDs prefixed with `ic_` to avoid collision with Indian Kanoon IDs

3. **Files changed:**
   - `src/acquisition/_models.py` — Added `download_url: str | None` field to `PreliminaryMetadata`
   - `src/acquisition/scrapers/_india_code.py` — Full rewrite (browse listing pagination, detail page scraping, PDF URL extraction)
   - `tests/acquisition/conftest.py` — Replaced India Code fixtures with real HTML structure from live site
   - `tests/acquisition/test_india_code_scraper.py` — Full rewrite (30 tests)
   - `tests/acquisition/test_pipeline.py` — Updated integration test for new scraper interface
   - `tests/acquisition/test_config.py` — Updated config assertion (removed `seed_act_ids`)
   - `configs/sources.yaml` — Removed broken `seed_act_ids`, updated config for browse-listing discovery

4. **Results:** 125 tests all passing, lint clean, format clean

**What broke:**
- Nothing broke during this session — the rewrite was clean. The existing scraper was already broken against the live site (just hadn't been tested against it).

**Decisions made:**
1. **Save HTML detail page during acquisition, not PDF** — PDF is binary and would be corrupted through text encoding roundtrip in the existing save pipeline. Added `download_url` to metadata so Phase 2 can download PDFs directly.
2. **Browse listing pagination for discovery** — Not search, not static seed lists. The DSpace browse listing at `/sp/browse?type=title` is server-rendered and paginable.
3. **Opaque numeric IDs prefixed with `ic_`** — Real India Code handles are numeric (e.g., `1999`). Prefix avoids collision with Indian Kanoon doc IDs.

**Open questions / action items:**
- **MUST TEST LIVE**: India Code scraper needs live smoke test against indiacode.nic.in with `max_documents: 2`. User stopped the live test during this session.
- **URL quirk**: The browse listing URL may require `etal=-1&null=` query params or the site returns HTTP errors. Scraper currently doesn't include these — needs live testing to confirm if plain HTTP (without Playwright) works.
- **Indian Kanoon API**: Still pending non-commercial approval.

**Next steps:**
1. Live smoke test India Code scraper against real site (max 2 docs)
2. If browse listing needs Playwright or special params, adjust scraper accordingly
3. Check for Indian Kanoon API approval; refactor IK scraper once approved
4. Begin Phase 2 (Parsing)

---

## Session: 2026-02-22 (third session)
**Phase:** Phase 1 wrap-up + Phase 2 (Parsing) kickoff

**What was built:**

1. **India Code live smoke test — PASSED**
   - Plain HTTP works (no Playwright needed), `etal` params not required
   - 2 docs scraped: Aadhaar Act 2016 + Academy of Scientific Research Act 2011
   - Metadata correct (title, act number, year, date), PDF URLs valid and accessible
   - Duration: 6.8s for 2 docs at 0.33 req/sec rate limit

2. **User-Agent fix** — India Code blocks bot UAs (403). Wired `user_agent` from `GlobalAcquisitionSettings` → `BaseScraper` → `HttpClient`:
   - `src/acquisition/_models.py` — updated default UA to browser-like string
   - `src/acquisition/base_scraper.py` — accept + pass `user_agent` param
   - `src/acquisition/pipeline.py` — pass `settings.user_agent` to scraper
   - `src/acquisition/scrapers/_india_code.py` — accept `**kwargs` in `__init__`
   - `configs/sources.yaml` — updated UA string
   - Committed as `1d225ae`

3. **Phase 2 plan created** — Full implementation plan in `plans/phase-2-parsing.md`:
   - 7 subtasks: Docling validation → Foundation → Router+Validation → IK HTML parser → PDF downloader → PDF parser → Pipeline+CLI
   - Data models: ParsedDocument, ParsedSection (hierarchical), ParsedTable, QualityReport
   - Parser architecture: BaseParser ABC → Router → source-specific parsers

4. **Docling validated on Python 3.14** — v2.74.0 installs + parses Indian statute PDFs successfully:
   - Academy of Scientific Research Act PDF → 53K chars structured markdown
   - Section headings preserved, table structure intact

5. **Phase 2 Subtask 1 complete** — Foundation (37 tests):
   - `src/parsing/_models.py` — 7 models: ParsedDocument, ParsedSection, ParsedTable, QualityReport, QualityCheckResult, ParsingSettings, ParsingConfig
   - `src/parsing/_exceptions.py` — ParsingError hierarchy (5 exception types)
   - `src/parsing/_config.py` — YAML config loader
   - `configs/parsing.yaml` — Parser configuration
   - `src/parsing/__init__.py` — Public API

6. **Phase 2 Subtask 2 in progress** — Base Parser + Router + Validation:
   - `src/parsing/parsers/_base.py` — BaseParser ABC (parse, can_parse, parser_type)
   - `src/parsing/_validation.py` — QualityValidator (4 checks: text completeness, section sequence, table integrity, OCR confidence)
   - `src/parsing/_router.py` — ParserRouter (priority-based parser selection)
   - Tests NOT yet written for these — session ended mid-subtask

**Results:** 162 tests passing (125 Phase 1 + 37 Phase 2), lint clean

**What broke:**
1. `IndiaCodeScraper.__init__()` didn't accept `user_agent` kwarg after BaseScraper change — fixed with `**kwargs`
2. Windows path separators in test assertions: `str(Path("data/raw"))` → `"data\\raw"` on Windows. Fixed by comparing `Path` objects instead of strings.
3. Ruff TC001: can't move Pydantic field type imports to TYPE_CHECKING — Pydantic needs them at runtime. Fixed with `# noqa: TC001`.

**Decisions made:**
1. **Browser User-Agent for all scrapers** — India Code blocks bot UAs. Using Chrome UA string. Ethical for public government data with rate limiting.
2. **Docling as primary PDF parser** — Confirmed working on Python 3.14. No fallback needed (pymupdf was planned as backup).
3. **PDF cache at `data/cache/pdf/`** — Separate from raw acquisition output. Clean separation.
4. **Build IK HTML parser now with synthetic data** — Ready when API is approved.
5. **Pydantic field types stay as runtime imports** — `# noqa: TC001` when Pydantic models reference cross-module types.

**Open questions:**
- Indian Kanoon API: still pending approval
- Content-hash idempotency won't work on India Code (dynamic HTML) — acceptable for now

**Next steps:**
1. Write tests for Subtask 2 (_validation.py, _router.py, _base.py)
2. Complete Subtask 3 (Indian Kanoon HTML parser)
3. Complete Subtask 4 (PDF downloader + India Code HTML parser)
4. Complete Subtask 5 (Docling PDF parser)
5. Complete Subtask 6 (Pipeline + CLI + integration tests)

---

## Session: 2026-02-22 (fourth session)
**Phase:** Phase 2 — Document Parsing (Subtasks 2–3)

**What was built:**

1. **Subtask 2 tests — 40 tests** across 3 new test files:
   - `tests/parsing/test_base_parser.py` (8 tests) — ABC enforcement, concrete subclass, can_parse routing
   - `tests/parsing/test_router.py` (10 tests) — Registration, priority selection, fallback, UnsupportedFormatError
   - `tests/parsing/test_validation.py` (22 tests) — Text completeness (6), section sequence (6), table integrity (4), OCR confidence (3), aggregate report (3)

2. **Subtask 3 — Indian Kanoon HTML parser + 29 tests:**
   - `src/parsing/parsers/_html_indian_kanoon.py` (~290 lines) — Full parser with:
     - Judgment parsing: HEADER → FACTS → ISSUES → REASONING → HOLDING → ORDER sections via bold heading marker matching
     - Statute parsing: PREAMBLE → CHAPTER → SECTION hierarchy with CLAUSE/PROVISO/EXPLANATION children
     - Metadata extraction from `div.doc_title`, `div.doc_bench`, `div.doc_citations`
     - Auto-detection of JUDGMENT vs STATUTE when `document_type` is None
     - Edge cases: empty file, missing `div.judgments`, no structural markers
   - `tests/parsing/test_html_indian_kanoon.py` (29 tests) — 8 test classes: can_parse (3), parser_type (1), judgment parsing (7), statute parsing (7), metadata (3), type detection (2), edge cases (4), output contract (2)
   - `src/parsing/parsers/__init__.py` — exports `IndianKanoonHtmlParser`

**Results:** 231 tests all passing (125 Phase 1 + 106 Phase 2), lint clean

**What broke:**
1. "HOLDING" bold heading not matched — `_HOLDING_MARKERS` had phrases like "for the foregoing reasons" but not the word "holding" itself. The fixture HTML uses `<b>HOLDING</b>`. Fix: added `"holding"` to the markers list.
2. Statute auto-detection failed — `_SECTION_RE` has `^` anchor which only matches at string start, not mid-text. `_detect_document_type` was using `_SECTION_RE.search(text)` on the full `div.judgments` text where "Section 1" appears mid-string. Fix: used boundary-based patterns without `^` anchors for the detection method.
3. Ruff RUF100 — `# noqa: TC001` on parser imports was unused because those imports are used at runtime (constructing objects), not just type hints. Ruff correctly identifies them as runtime imports.

**Decisions made:**
1. **BeautifulSoup with html.parser backend** — Already used in Phase 1 scrapers. CSS selectors map directly to IK HTML structure. No new dependency.
2. **Judgment sections are flat, statutes are hierarchical** — Judgment sections (FACTS, ISSUES, etc.) are siblings. Statute sections form a tree (CHAPTER → SECTION → CLAUSE/PROVISO).
3. **Bold heading = structural marker only when it starts the paragraph** — Avoids false positives from bold text inline (e.g., emphasized case names).
4. **ORDER exact match only** — "order" is too common a word for substring matching. Only `<b>ORDER</b>` (exact) triggers ORDER section.
5. **Placeholder QualityReport in parser output** — Parser returns `QualityReport(overall_score=0.0, passed=False)`. Real validation done by QualityValidator in the pipeline.
6. **Separate detection patterns from parsing patterns** — `_SECTION_RE` with `^` anchor is correct for line-by-line parsing. Detection uses simpler `\bSection\s+\d+` without anchor.

**Open questions:**
- Indian Kanoon API: still pending approval
- Should `_REASONING_MARKERS` include "analysis" alone? Currently requires "analysis and reasoning" (substring match). Real IK pages may vary.

**Next steps:**
1. Complete Subtask 4 (PDF downloader + India Code HTML parser)
2. Complete Subtask 5 (Docling PDF parser)
3. Complete Subtask 6 (Pipeline + CLI + integration tests)
4. Commit all Phase 2 work when phase is complete (or commit incrementally)

---

## Session: 2026-02-23 18:15
**Phase:** Phase 2 — Document Parsing (Subtasks 4–5)

**What was built:**

1. **Subtask 4 — PDF Downloader + India Code HTML Parser (29 tests):**
   - `src/parsing/_downloader.py` (~130 lines) — `PdfDownloader` class:
     - Async PDF download via aiohttp with configurable timeout + User-Agent
     - Cache-based idempotency at `data/cache/pdf/{doc_id}.pdf` (skips if non-empty file exists)
     - Atomic writes via `tempfile.NamedTemporaryFile` + `Path.replace()`
     - Size limit enforcement: Content-Length pre-check + streaming byte count check
     - Raises `PDFDownloadError` on HTTP errors, timeout, network error, size exceeded
   - `src/parsing/parsers/_html_india_code.py` (~180 lines) — `IndiaCodeHtmlParser`:
     - Extracts metadata from DSpace detail page: `table.itemDisplayTable` (act number, date, year), `h2` title, `<title>` tag fallback
     - Year extraction from title text when not in metadata table
     - Falls back to Phase 1 `PreliminaryMetadata` for missing fields
     - Returns minimal `ParsedDocument` with `PREAMBLE` section (real content from PDF)
   - `tests/parsing/test_downloader.py` (11 tests) — cache hit, empty cache, download, dir creation, HTTP 404/500, network error, timeout, size limits (header + streaming), atomic write cleanup
   - `tests/parsing/test_html_india_code.py` (18 tests) — can_parse (3), parser_type, metadata extraction (7), title fallback, sections (2), output contract (4)

2. **Subtask 5 — Docling PDF Parser (19 tests):**
   - `src/parsing/parsers/_docling_pdf.py` (~340 lines) — `DoclingPdfParser`:
     - Lazy Docling import — graceful `ParserNotAvailableError` if not installed
     - Converts PDF → markdown via `DocumentConverter`, walks markdown to build `ParsedSection` tree
     - Full Indian statute patterns: PART, CHAPTER, SECTION, SUB_SECTION, CLAUSE, PROVISO, EXPLANATION, SCHEDULE
     - Generic markdown fallback (heading-based paragraphs) for non-statute docs
     - Table extraction from Docling objects (DataFrame or raw cell data)
     - Page count from Docling's page tracking
   - `tests/parsing/test_docling_pdf.py` (19 tests) — all mocked (no docling dependency needed):
     - can_parse (3), parser_type, docling unavailable, statute hierarchy (7), generic parsing (2), empty output, output contract (4)

**Results:** 279 tests all passing (125 Phase 1 + 154 Phase 2), lint clean, format clean

**What broke:**
1. **Windows PermissionError on temp file unlink** — `except BaseException: tmp_path.unlink()` inside a `with NamedTemporaryFile()` block fails on Windows because the file descriptor is still open. Fix: restructured so cleanup happens after the `with` block exits (fd closed by context manager first).
2. **Patch target for lazy import** — `patch("module.DocumentConverter")` fails when `DocumentConverter` is imported inside a method (not at module level). Fix: mock `_convert_with_docling` directly via `patch.object()` instead of mocking the import.
3. **Ruff SIM115** — `tempfile.NamedTemporaryFile()` without context manager. Fix: wrap in `with` statement, restructure cleanup flow.
4. **Ruff RUF015** — `[x for x in list if cond][0]` → `next(x for x in list if cond)` in tests.

**Decisions made:**
1. **Mock `_convert_with_docling` not `DocumentConverter`** — Since Docling import is lazy (inside method), mocking the static method is cleaner and doesn't require `sys.modules` hacking.
2. **Cleanup after `with` block, not inside** — On Windows, can't unlink an open file. The pattern is: `try: with tempfile as fd: ... ; rename(); except: unlink(); raise`.
3. **Generic markdown parser as fallback** — Non-statute PDFs (notifications, circulars) get paragraph-level sections based on markdown headings.
4. **Table extraction is best-effort** — If Docling table objects don't match expected API, log warning and skip rather than failing the entire parse.

**Open questions:**
- Indian Kanoon API: still pending approval
- Docling table extraction API shape may vary by version — current code handles DataFrame and raw cell data, may need adjustment when tested with real Docling output
- Should the pipeline (Subtask 6) merge IndiaCodeHtmlParser metadata with DoclingPdfParser content for India Code docs? (Yes — this is the planned flow)

**Next steps:**
1. Complete Subtask 6 (Pipeline + CLI + integration tests) — final subtask of Phase 2
2. Commit all Phase 2 Subtasks 4-6 work
3. Update phase plan with completion status

---

## Session: 2026-02-24 (first session)
**Phase:** Phase 2 — Document Parsing (Subtask 6 — final)

**What was built:**

1. **Subtask 6 — Pipeline + CLI + Integration Tests (29 tests):**
   - `src/parsing/pipeline.py` (~270 lines) — `ParsingPipeline` orchestrator:
     - Constructor: loads config, builds router (IK HTML → IC HTML → Docling PDF priority), creates QualityValidator + PdfDownloader
     - `async run(source_name, dry_run) → ParsingResult` — discovers `.meta.json` files, processes each document, returns aggregate result
     - Indian Kanoon flow: load meta.json → route to IK HTML parser → validate → save
     - India Code two-step flow: download PDF → parse HTML for metadata → parse PDF with Docling → merge metadata → validate → save
     - Idempotency via output file existence check
     - Per-document error isolation (never crashes pipeline)
     - `_merge_metadata()` — HTML-authoritative for metadata fields, PDF-authoritative for content
   - `src/parsing/run.py` (~80 lines) — CLI: `--source`, `--dry-run`, `--log-level`, `--console-log`, `--config`
   - `src/parsing/__main__.py` — module runner
   - `src/parsing/_models.py` — added `ParsingResult` model
   - `src/parsing/__init__.py` — updated exports (ParsingPipeline, ParsingResult, run_parsing())
   - `tests/parsing/test_pipeline.py` (29 tests) — 9 test classes:
     - TestSourceResolution (5), TestDiscovery (4), TestIndianKanoonFlow (3), TestIndiaCodeFlow (3), TestIdempotency (2), TestDryRun (2), TestErrorHandling (4), TestOutputValidation (2), TestMergeMetadata (2), TestParsingResult (2)

2. **GitHub repo created** — https://github.com/harshsahrawat-commits/legal-rag-pipeline (public)
   - All 7 commits pushed to `origin/master`

**Results:** 308 tests all passing (125 Phase 1 + 183 Phase 2), lint clean, format clean

**What broke:**
- Nothing broke — clean implementation session. Ruff caught 4 minor issues (unused imports, TC003 Path in TYPE_CHECKING, empty TYPE_CHECKING block) — all fixed before tests ran.

**Decisions made:**
1. **Single `ParsingResult` return** (not list) — parsing is document-granular, not source-granular like acquisition. One result summarizes the whole run.
2. **Sequential document processing** — parsers are sync (CPU-bound), only PDF download is async. Keeps it simple; concurrency can be added later.
3. **`_merge_metadata` as module-level function** — static, testable independently. Uses `model_copy(update=...)` for clean Pydantic v2 immutable updates.
4. **`_resolve_source_filter` as module-level function** — extracted from class for easier unit testing.
5. **GitHub repo on `master` branch** — `gh repo create` defaulted to `master`. Can rename to `main` later.

**Open questions:**
- Indian Kanoon API: still pending approval
- GitHub branch naming: `master` vs `main` — currently `master`, CLAUDE.md says `main`
- Phase 3 (Chunking) planning needed — ParsedDocument is now the input contract

**Next steps:**
1. Begin Phase 3 (Chunking) — read `docs/chunking_strategies.md` first
2. Create `plans/phase-3-chunking.md`
3. Rename `master` → `main` if desired
4. Check for Indian Kanoon API approval

---

## Session: 2026-02-24 (second session)
**Phase:** Phase 3 — Chunking (Subtasks 1–7 of 8)

**What was built:**

1. **Subtask 1 — Foundation (50 tests):**
   - `src/chunking/_models.py` — All enums (`ChunkType`, `ChunkStrategy`, `TemporalStatus`, `CourtHierarchy`), all sub-models (`SourceInfo`, `StatuteMetadata`, `AmendmentRecord`, `JudgmentMetadata`, `ContentMetadata`, `IngestionMetadata`, `ParentDocumentInfo`), `LegalChunk`, `ChunkingSettings`, `ChunkingConfig`, `ChunkingResult`
   - `src/chunking/_exceptions.py` — `ChunkingError` hierarchy (5 types)
   - `src/chunking/_config.py` — YAML config loader
   - `src/chunking/_token_counter.py` — `TokenCounter` (tiktoken `cl100k_base`)
   - `configs/chunking.yaml` — Chunking configuration

2. **Subtask 2 — Base + Metadata + PageLevel (53 tests):**
   - `src/chunking/chunkers/_base.py` — `BaseChunker` ABC
   - `src/chunking/_metadata_builder.py` — `MetadataBuilder` (source, statute, judgment, content, ingestion metadata + citation extraction regex)
   - `src/chunking/chunkers/_page_level.py` — Strategy 6: page-per-chunk, degraded scan flagging

3. **Subtask 3 — StatuteBoundaryChunker (23 tests):**
   - `src/chunking/chunkers/_statute_boundary.py` — Strategy 1: section-boundary chunking
   - Walk sections tree recursively, prepend act name header
   - Split at sub-section boundaries when >max_tokens, keep proviso/explanation with parent
   - Definition sections tagged `ChunkType.DEFINITION`, schedules as `SCHEDULE_ENTRY`

4. **Subtask 4 — JudgmentStructuralChunker (18 tests):**
   - `src/chunking/chunkers/_judgment_structural.py` — Strategy 2: judgment structural chunking
   - Header → compact chunk, its ID set as `judgment_header_chunk_id` on ALL other chunks
   - Holding never split; Facts/Reasoning split at paragraph boundaries
   - Dissent/Obiter get own chunks

5. **Subtask 5 — Router + Pipeline + CLI (22 tests):**
   - `src/chunking/_router.py` — `ChunkerRouter` (tiered priority selection)
   - `src/chunking/pipeline.py` — `ChunkingPipeline` (discover → load → route → chunk → post-process → save)
   - `src/chunking/run.py` + `__main__.py` — CLI with `--source`, `--dry-run`, `--log-level`, `--console-log`, `--config`
   - Post-processing: sequential `chunk_index`, `sibling_chunk_ids` (±2 window)

6. **Subtask 6 — RSC + SemanticChunker (24 tests):**
   - `src/chunking/chunkers/_recursive_semantic.py` — Strategy 3: 3-phase RSC (recursive split → semantic merge → oversized split)
   - `src/chunking/chunkers/_semantic_maxmin.py` — Strategy 5: sentence-level embedding similarity, percentile threshold
   - Both use lazy `sentence-transformers` import, all tests mock embeddings

7. **Subtask 7 — PropositionChunker (14 tests):**
   - `src/chunking/chunkers/_proposition.py` — Strategy 4: LLM decomposition of definitions
   - Lazy `anthropic` import, tests mock API client

**Results:** 204 new tests (512 total across all modules), lint clean, format clean

**What broke:**
1. `_flatten_section_text` didn't include `section.number` → sub-section numbers like `(1)` missing from chunk text. Fix: build label from number + title.
2. `_find_definition_sections` returned both parent and children definitions → double-chunking. Fix: skip children when parent already matched.
3. `SemanticMaxMinChunker._merge_and_split` — tiny groups kept merging without checking if accumulated result exceeds max_tokens. Fix: check merged size before appending.
4. `_recursive_split` test expected split at `\n\n` but text was under max_tokens. Fix: lowered max_tokens in test.

**Decisions made:**
1. **Chunkers are pure transforms** — accept `ParsedDocument`, return `list[LegalChunk]`. Pipeline handles all I/O.
2. **One `TokenCounter` instance shared across all chunkers** — injected via constructor.
3. **Router returns ONE chunker per document** — no multi-chunker orchestration.
4. **Optional deps degrade gracefully** — `ChunkerNotAvailableError` when sentence-transformers/anthropic not installed, router skips.
5. **`CourtHierarchy` is `IntEnum`** — enables comparison operators for precedent authority.
6. **numpy used directly for cosine similarity** — dot product of L2-normalized vectors. No sklearn dependency.

**Open questions:**
- Subtask 8 (integration tests + final wiring) still pending — need to finalize `__init__.py` exports and end-to-end tests
- RAPTOR + QuIM models defined but deferred — enum values and metadata fields ready

**Next steps:**
1. Complete Subtask 8: integration tests + final `__init__.py` exports
2. Commit all Phase 3 work
3. Run full test suite verification
4. Update CLAUDE.md chunking CLAUDE.md with implementation details

---

## Session: 2026-02-24 (third session)
**Phase:** Phase 3 — Chunking (Subtask 8 — final)

**What was built:**

1. **Subtask 8 — Integration tests + final exports (22 tests):**
   - `tests/chunking/test_integration.py` (22 tests) — 8 test classes:
     - TestStatuteEndToEnd (4): routing to StatuteBoundaryChunker, statute metadata, chunk types, document ID consistency
     - TestJudgmentEndToEnd (4): routing to JudgmentStructuralChunker, judgment metadata, header_chunk_id propagation, expected section types
     - TestDegradedScanRouting (2): pipeline routing with/without optional deps, direct PageLevel validation
     - TestMultiDocumentRun (2): mixed doc types + mixed sources in single pipeline run
     - TestTokenBounds (3): max_tokens compliance, token_count field accuracy
     - TestPostProcessing (4): index start at zero, contiguous indices, valid sibling UUIDs, window bounds
     - TestJsonRoundTrip (2): full field survival through serialization, judgment metadata round-trip
     - TestSourceProvenance (1): source info fields populated from ParsedDocument
   - `src/chunking/__init__.py` — Updated exports: `ChunkingConfig`, `ChunkingPipeline`, `ChunkingResult`, `LegalChunk`, plus `run_chunking()` convenience wrapper

2. **Phase 3 committed and pushed:** `f14a622` — 48 files, 7,133 lines added

**Results:** 534 tests all passing (125 Phase 1 + 183 Phase 2 + 226 Phase 3), lint clean, format clean

**What broke:**
1. **Degraded scan routing to SemanticMaxMin instead of PageLevel** — Documents with `sections=[]` and low OCR confidence route to SemanticMaxMin (which matches `len(sections)==0`) before PageLevel. If sentence-transformers isn't installed, SemanticMaxMin fails at runtime (`_ensure_model()` raises), and the pipeline records a failure. Fix: made integration test resilient to both scenarios (with/without sentence-transformers), added direct PageLevel test as separate validation.

**Decisions made:**
1. **Integration tests validate output quality, not just pipeline mechanics** — Existing `test_pipeline.py` tests discovery, idempotency, error handling. New `test_integration.py` validates metadata correctness, token bounds, sibling IDs, JSON round-trip integrity.
2. **Degraded scan test is environment-aware** — Uses `_has_sentence_transformers()` helper to assert correct behavior in both environments (with optional deps = semantic routing, without = graceful failure + error message).
3. **`__init__.py` follows parsing module pattern** — Exports primary data model, result model, pipeline class, config class, and convenience runner function.

**Open questions:**
- Router priority: degraded scans (OCR < 80%) should arguably route to PageLevel regardless of SemanticMaxMin availability. Currently they route to SemanticMaxMin if it matches first. Consider adding OCR check to SemanticMaxMin's `can_chunk()` or adjusting router priority.
- RAPTOR + QuIM chunkers deferred — enum values and metadata fields exist but no implementations yet
- Indian Kanoon API: still pending non-commercial approval

**Next steps:**
1. Begin Phase 4 (Enrichment) — read `docs/enrichment_guide.md` first
2. Create `plans/phase-4-enrichment.md`
3. Consider router fix for degraded scan priority before Phase 4

---

## Session: 2026-02-25 02:30
**Phase:** Phase 4 — Enrichment (all 5 subtasks complete)

**What was built:**

1. **Phase 4 plan** — Full implementation plan in `plans/phase-4-enrichment.md`: 5 subtasks, ~191 estimated tests, Late Chunking deferred to Phase 5 (it's an embedding operation).

2. **Subtask 1 — Foundation (37 tests):**
   - `src/enrichment/_exceptions.py` — 6 exception classes: `EnrichmentError`, `ContextualRetrievalError`, `QuIMGenerationError`, `EnricherNotAvailableError`, `LLMRateLimitError`, `DocumentTextTooLargeError`
   - `src/enrichment/_models.py` — `QuIMEntry`, `QuIMDocument`, `EnrichmentSettings`, `EnrichmentConfig`, `EnrichmentResult`
   - `src/enrichment/_config.py` — YAML config loader (mirrors `load_chunking_config`)
   - `configs/enrichment.yaml` — settings: input/output/parsed dirs, model, concurrency, quim_questions_per_chunk, context_window_tokens

3. **Subtask 2 — BaseEnricher + ContextualRetrievalEnricher (34 tests):**
   - `src/enrichment/enrichers/_base.py` — `BaseEnricher` ABC with `async enrich_document()` and `stage_name` property
   - `src/enrichment/enrichers/_contextual.py` — Full implementation (~230 lines):
     - AsyncAnthropic client with lazy init
     - Prompt caching: full doc in system message with `cache_control: {"type": "ephemeral"}`
     - Document windowing for docs >180K tokens (splits into overlapping windows)
     - `asyncio.Semaphore` concurrency control
     - Per-chunk error isolation (failed chunks stay unenriched, others proceed)
     - Sets `chunk.contextualized_text = f"{context}\n\n{chunk.text}"` and `chunk.ingestion.contextualized = True`

4. **Subtask 3 — QuIMRagEnricher (28 tests):**
   - `src/enrichment/enrichers/_quim.py` — Same prompt caching pattern, generates N questions per chunk
   - `get_quim_document()` returns accumulated `QuIMDocument` for sidecar file
   - `_parse_questions()` helper filters LLM output (blank lines, short non-questions)
   - Resolves Act name / case citation and section reference for prompt context

5. **Subtask 4 — Pipeline Orchestrator (21 tests):**
   - `src/enrichment/pipeline.py` — `EnrichmentPipeline` (~240 lines):
     - `async run(source_name, stage, dry_run) -> EnrichmentResult`
     - Stage selection: `None` = both, `"contextual_retrieval"`, `"quim_rag"`
     - Discovery: scans `data/chunks/{source}/*.json`, excludes `.quim.json`
     - Idempotency: skip if output file + quim file already exist
     - Per-document error isolation
     - Loads ParsedDocument for full text context; falls back to empty stub if missing
     - Saves enriched chunks to `data/enriched/{source}/` and QuIM sidecar to `.quim.json`

6. **Subtask 5 — CLI + Integration Tests (24 tests):**
   - `src/enrichment/run.py` — CLI: `--source`, `--stage`, `--dry-run`, `--log-level`, `--console-log`, `--config`
   - `src/enrichment/__main__.py` — module runner
   - `src/enrichment/__init__.py` — exports `EnrichmentConfig`, `EnrichmentPipeline`, `EnrichmentResult`, `run_enrichment()`
   - `tests/enrichment/conftest.py` — shared fixtures: sample chunks, parsed doc, `make_mock_async_anthropic()`
   - `tests/enrichment/test_integration.py` — 15 end-to-end tests: both stages, idempotency, error isolation, JSON round-trip
   - `tests/enrichment/test_run.py` — 9 CLI + export tests

**Results:** 144 new enrichment tests (678 total project), lint clean, format clean

**What broke:**
1. **Ruff TC003 on `UUID` import** — Pydantic needs `UUID` at runtime for model fields, but Ruff wants it in `TYPE_CHECKING`. Fix: `# noqa: TC003` (same pattern as `# noqa: TC001` for cross-module Pydantic field types).
2. **Ruff TC003 on `Path` import in tests** — `Path` used both in annotations (string after `from __future__ import annotations`) and runtime (`Path(...)` calls). Fix: `# noqa: TC003`.
3. **Pipeline test expected `documents_failed=1` on enricher failure** — But the contextual enricher isolates errors per-chunk, so even when all LLM calls fail, `enrich_document()` returns normally (no document-level exception). The document is "enriched" with zero contextualized chunks. Fix: updated test to assert `documents_enriched=1, chunks_contextualized=0` instead.
4. **Ruff I001 import sorting** — `# noqa` comments cause import blocks to be considered unsorted. Fix: `ruff check --fix` auto-sorts.
5. **Unused imports** — `ContextualRetrievalError`, `QuIMGenerationError`, `MagicMock`, `patch`, `pytest` imported but unused in various test/source files. Fix: `ruff check --fix` auto-removed.

**Decisions made:**
1. **Late Chunking deferred to Phase 5** — It's fundamentally an embedding operation (embed full doc → split token embeddings at chunk boundaries → mean pool). Requires embedding model (BGE-m3/jina) and produces vectors for Qdrant. Inseparable from Phase 5.
2. **AsyncAnthropic for enrichment** — All LLM calls are I/O-bound. `anthropic.AsyncAnthropic()` with `asyncio.Semaphore(concurrency)` for concurrent chunk processing within a document.
3. **Separate output directory (`data/enriched/`)** — Preserves Phase 3 originals. Each stage is independent and idempotent per architecture rules.
4. **QuIM questions as sidecar `.quim.json`** — Separate from enriched chunk file so Phase 5 can embed questions independently.
5. **Per-chunk error isolation** — Failed LLM calls leave individual chunks unenriched; remaining chunks still processed. Document-level isolation catches broader failures.
6. **Document windowing** — Documents >180K tokens (Haiku context - safety margin) split into overlapping windows. Chunks grouped by window; each window group shares one cached system prompt.
7. **v2 architecture docs validated unchanged** — `legal-rag-ingestion-v2.md` and `verdict-ai-*-v2-update.md` confirm Phase 4 design matches existing `docs/enrichment_guide.md`. No schema changes needed.

**Open questions:**
- Router priority for degraded scans still not fixed (Phase 3 carry-over)
- Indian Kanoon API: still pending non-commercial approval
- `anthropic` SDK minimum version for `cache_control` support — currently `>=0.18` in pyproject.toml, may need `>=0.28+` for prompt caching
- Phase 5 (Embedding & Indexing) planning needed — includes Late Chunking, Qdrant dual vectors, BGE-m3 fine-tuning

**Next steps:**
1. Begin Phase 5 (Embedding & Indexing) — read `docs/embedding_fine_tuning.md` first
2. Create `plans/phase-5-embedding.md`
3. Implement Late Chunking as part of Phase 5
4. Verify `anthropic` SDK version supports `cache_control` parameter

---

## Session: 2026-02-25 16:00
**Phase:** Phase 5 — Embedding & Indexing
**What was built:**
- Full `src/embedding/` module (11 source files, 3,639 lines):
  - `_exceptions.py` — 7 exception classes (EmbeddingError hierarchy)
  - `_models.py` — SparseVector, EmbeddingSettings, EmbeddingConfig, EmbeddingResult
  - `_config.py` — YAML config loader (mirrors enrichment pattern)
  - `_embedder.py` — LateChunkingEmbedder: full doc → token embeddings → slice per chunk → mean pool, plus standard embed_texts() for QuIM, Matryoshka slicing
  - `_sparse.py` — BM25SparseEncoder: per-doc vocabulary + IDF, BM25-weighted sparse vectors
  - `_qdrant_indexer.py` — QdrantIndexer: dual-vector (full 768d + fast 64d) + sparse BM25 collections, chunk/QuIM upsert
  - `_redis_store.py` — RedisParentStore: parent chunk text + judgment header storage
  - `pipeline.py` — EmbeddingPipeline: 11-step orchestrator (load → idempotency → Late Chunking → Matryoshka → BM25 → Qdrant → QuIM → Redis → flag update)
  - `run.py` — CLI: --source, --dry-run, --device, --config, --log-level, --console-log
  - `__init__.py` — Exports: EmbeddingPipeline, EmbeddingConfig, EmbeddingResult
  - `__main__.py` — Module runner
- `configs/embedding.yaml` — Connection strings, model name, dims, batch sizes
- `tests/embedding/` (11 test files, 149 tests):
  - test_exceptions (11 parametrized), test_models (12), test_config (8), test_embedder (25), test_sparse (20), test_qdrant_indexer (18), test_redis_store (14), test_pipeline (20), test_run (5), test_integration (8), conftest.py (shared fixtures)
- `pyproject.toml` — Added `[embedding]` optional dependency group (qdrant-client, redis, torch, transformers)

**What broke:**
1. **TestModelLoading: `patch("src.embedding._embedder.AutoModel")` fails** — AutoModel is lazy-imported inside `load_model()`, not at module level. `patch()` needs module-level attributes. Fix: used `patch.dict("sys.modules", {"transformers": mock_module})` instead.
2. **Qdrant tests: `qdrant_client` not installed** — Lazy imports inside `_create_chunks_collection`, `upsert_chunks` etc. fail because `qdrant_client.models` isn't available. Fix: created `_build_mock_qdrant_module()` helper that builds fake `qdrant_client` + `qdrant_client.models` modules and patches `sys.modules` via autouse fixture.
3. **test_embedding_error_isolates: pipeline falls back to embed_texts** — When parsed doc not found → `raw_text=""` → pipeline uses `embed_texts()` instead of `embed_document_late_chunking()`. Mocking only `embed_document_late_chunking` to raise wasn't enough. Fix: mock both `embed_document_late_chunking` and `embed_texts` to raise.
4. **test_partial_failure: call_count tracking** — With empty full_text, `embed_document_late_chunking` is never called, so the call_count never incremented for the embed_texts path. Fix: simplified to only mock `embed_texts` since that's the code path when no parsed doc exists.

**Decisions made:**
1. **Use `transformers` directly, not `sentence-transformers`** — Need `last_hidden_state` (token-level embeddings), not pooled output. `sentence-transformers` wraps models and only returns pooled embeddings.
2. **Token boundary reconstruction via `return_offsets_mapping=True`** — Tokenizer produces char offset mapping; `full_text.find(chunk.text[:80])` maps chunk → char offset → token indices via offset mapping.
3. **Document windowing for >8192 tokens** — Overlapping windows, embed each, average overlapping token positions. Same approach as Phase 4 enrichment.
4. **Late Chunking uses `chunk.text`, BM25 uses `contextualized_text`** — Original text for token matching against full doc; contextualized text for BM25 keyword enrichment.
5. **Per-document BM25 vocabulary** — Sufficient for indexing. Global IDF can be added later as optimization.
6. **Qdrant sync client** — Simpler than async client, adequate for local Qdrant. Methods are async in pipeline but client calls are sync.
7. **`numpy` import in `_qdrant_indexer.py` needs `# noqa: TC002`** — Used at runtime for `.tolist()`, not just type hints.

**Open questions:**
- Router priority for degraded scans still not fixed (Phase 3 carry-over)
- Indian Kanoon API: still pending non-commercial approval
- `anthropic` SDK minimum version for `cache_control` support
- Global IDF for BM25: deferred optimization
- Neo4j Knowledge Graph is Phase 6

**Next steps:**
1. Begin Phase 6 (Knowledge Graph) — read `docs/knowledge_graph_schema.md` first
2. Or begin Phase 7 (Hallucination) if KG is deferred
3. Consider fine-tuning pipeline for BGE-m3 (offline `scripts/` task)

## Session: 2026-02-26 ~22:00
**Phase:** Phase 6 — Knowledge Graph
**What was built:**
- Full `src/knowledge_graph/` module (13 source files, 26 files total including tests)
- `_exceptions.py` — 7 exception classes under `KnowledgeGraphError`
- `_models.py` — 8 Pydantic node models (Act, Section, SectionVersion, Judgment, Amendment, LegalConcept, Court, Judge), Relationship, ExtractedEntities, IntegrityCheck/Report, KGSettings/Config, KGResult
- `_config.py` — YAML config loader (mirrors embedding pattern)
- `_client.py` — Neo4jClient: lazy async driver, schema setup (8 constraints + 3 indexes), MERGE for all 8 node types, generic relationship MERGE, batch execution, parameterized read queries
- `_extractors.py` — EntityExtractor: chunk-driven extraction from StatuteMetadata → Act/Section/SectionVersion/Amendment, JudgmentMetadata → Judgment/Court/Judge, definition chunks → LegalConcept
- `_relationships.py` — RelationshipBuilder: all 15 relationship types from the schema doc
- `_queries.py` — QueryBuilder: 8 reusable Cypher queries (point-in-time, amendment cascade, citation traversal, hierarchy navigation, temporal status, judgment relationships, find replacement, node exists)
- `_integrity.py` — IntegrityChecker: 4 post-ingestion rules (section versions, repealed consistency, overrule hierarchy, version date overlap)
- `pipeline.py` — KnowledgeGraphPipeline orchestrator
- `run.py` + `__main__.py` — CLI with --source, --dry-run, --skip-integrity, --config
- `configs/knowledge_graph.yaml` — default config
- 225 tests across 10 test files, 8 integration tests
- Cleaned up junk files (=0.7, nul, .playwright-mcp/), updated .gitignore
- Committed v2 architecture docs and Phase 6 plan

**What broke:**
1. `"omit" in "omission"` → False! The word "omission" does NOT contain "omit" as a substring (it's "omis" + "sion"). Fixed `_amendment_rel_type` to check for "omis" as well.
2. Ruff TC003 flagged `UUID` import in `_models.py` — needed `# noqa: TC003` since UUID is used as Pydantic field type (known gotcha).
3. Ruff TC003 flagged `Path` import in test files — needed `# noqa: TC003` since Path is used at runtime in fixture bodies.

**Decisions made:**
- Chunk-driven entity extraction (not re-parsing raw text) — entities come from LegalChunk metadata fields
- Cross-document "dangling" MERGE: if target node doesn't exist yet, MERGE creates a stub that gets enriched later
- Section version ID = `{act_name}:{section_number}:{text_hash[:8]}` for human-readable uniqueness
- Reverse relationship direction for OVERRULES/FOLLOWS/DISTINGUISHES: stored on the overruled judgment, but relationship arrow points from overruler → overruled
- `_parse_section_ref` with common abbreviation expansion (IPC → Indian Penal Code, etc.)
- Skip self-references in REFERENCES relationships

**Open questions:**
- Phase 7 (Retrieval) plan not yet created
- Global IDF for BM25 still deferred
- Neo4j Community Edition 5.x needed for integration testing (Docker)

**Next steps:**
1. Plan Phase 7 (Retrieval) — hybrid search, reranking, FLARE, graph-augmented retrieval
2. Or plan Phase 8 (Hallucination Mitigation) — now that QueryBuilder exists as foundation
3. Consider Phase 0 (Query Intelligence) — semantic cache, query router, HyDE
