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
