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
