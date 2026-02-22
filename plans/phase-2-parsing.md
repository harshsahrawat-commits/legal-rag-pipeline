# Phase 2: Document Parsing — Implementation Plan

## Context

Phase 1 (Acquisition) is complete: 125 tests, lint clean, live-validated. It produces `data/raw/{source}/{doc_id}.html` + `{doc_id}.meta.json` sidecars. India Code detail pages don't contain statute text (JS-rendered) — the actual content is in PDFs pointed to by `download_url` in metadata. Phase 2 transforms these raw documents into structured `ParsedDocument` objects that Phase 3 (Chunking) can consume.

## Data Flow

```
Phase 1 Output                      Phase 2                           Phase 3 Input
─────────────────                    ───────                           ─────────────
data/raw/india_code/                                                  data/parsed/india_code/
  ic_2160.html          ──┐                                             ic_2160.json
  ic_2160.meta.json     ──┤  Load RawDocument                           (ParsedDocument)
                          ├─→ Download PDF from download_url
                          ├─→ Parse PDF (Docling or fallback)
                          ├─→ Detect structure (sections, chapters)
                          ├─→ Validate quality
                          └─→ Save ParsedDocument JSON

data/raw/indian_kanoon/
  doc_123.html          ──┐
  doc_123.meta.json     ──┤  Load RawDocument
                          ├─→ Parse HTML (BeautifulSoup)
                          ├─→ Detect structure (facts, issues, holding)
                          ├─→ Validate quality
                          └─→ Save ParsedDocument JSON
```

## Module Structure

```
src/parsing/
├── __init__.py                    # Exports: ParsingPipeline, ParsedDocument
├── __main__.py                    # python -m src.parsing
├── run.py                         # CLI entry point (argparse)
├── pipeline.py                    # Orchestrator
├── _models.py                     # ParsedDocument, ParsedSection, QualityReport
├── _exceptions.py                 # ParsingError hierarchy
├── _config.py                     # YAML config loader
├── _validation.py                 # Post-parse quality checks
├── _router.py                     # Select parser by source_type + content_format
├── _downloader.py                 # PDF downloader (for India Code download_url)
└── parsers/
    ├── __init__.py
    ├── _base.py                   # BaseParser ABC
    ├── _docling_pdf.py            # Docling PDF parser (primary)
    ├── _html_indian_kanoon.py     # Indian Kanoon HTML parser
    └── _html_india_code.py        # India Code detail page metadata extractor

configs/parsing.yaml               # Parser config (thresholds, paths, OCR settings)
tests/parsing/                     # Mirror structure
```

## Key Data Models

### ParsedSection (hierarchy node)
```python
class SectionLevel(StrEnum):
    PREAMBLE = "preamble"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUB_SECTION = "sub_section"
    CLAUSE = "clause"
    SUB_CLAUSE = "sub_clause"
    PROVISO = "proviso"
    EXPLANATION = "explanation"
    DEFINITION = "definition"
    SCHEDULE = "schedule"
    # Judgment-specific
    HEADER = "header"
    FACTS = "facts"
    ISSUES = "issues"
    REASONING = "reasoning"
    HOLDING = "holding"
    ORDER = "order"
    DISSENT = "dissent"
    OBITER = "obiter"

class ParsedSection(BaseModel):
    id: str                         # e.g., "s302", "ch17"
    level: SectionLevel
    number: str | None = None       # e.g., "302", "XVII"
    title: str | None = None        # e.g., "Punishment for murder"
    text: str                       # Full text of this section
    children: list[ParsedSection]   # Nested hierarchy
    parent_id: str | None = None
    token_count: int = 0
    page_numbers: list[int] = Field(default_factory=list)
```

### ParsedDocument (Phase 2 output)
```python
class ParsedDocument(BaseModel):
    document_id: UUID               # From RawDocument
    source_type: SourceType
    document_type: DocumentType
    content_format: ContentFormat

    raw_text: str                   # Full flattened text
    sections: list[ParsedSection]   # Hierarchical tree
    tables: list[ParsedTable]

    # Metadata (enriched from Phase 1 preliminary)
    title: str | None
    act_name: str | None
    act_number: str | None
    year: int | None
    date: str | None
    court: str | None
    case_citation: str | None
    page_count: int | None

    # Parsing provenance
    parser_used: ParserType
    ocr_applied: bool
    ocr_confidence: float | None
    parsing_duration_seconds: float

    # Quality
    quality: QualityReport

    # Lineage
    raw_content_path: str
    parsed_at: datetime
```

## Quality Validation (from docs/parsing_guide.md)

| Check | Method | Pass Threshold |
|-------|--------|---------------|
| Text completeness | `len(raw_text) / (page_count * 2000)` | > 0.5 (PDF only) |
| Section sequence | Regex extract section numbers, check gaps | No gaps |
| Table integrity | `row_count * col_count == cell_count` | Exact match |
| OCR confidence | Mean per-character confidence | > 0.85 |

## Structure Detection

Reuse regex patterns from `docs/chunking_strategies.md`:
```python
SECTION_PATTERN = r"^(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?)"
SUBSECTION_PATTERN = r"^\((\d+)\)"
CLAUSE_PATTERN = r"^\(([a-z])\)"
PROVISO_PATTERN = r"^Provided\s+that"
EXPLANATION_PATTERN = r"^Explanation\.?"
```

Judgment markers:
```python
FACTS_MARKERS = ["facts of the case", "brief facts", "factual matrix"]
ISSUES_MARKERS = ["issues for consideration", "questions of law"]
HOLDING_MARKERS = ["we hold that", "for the foregoing reasons", "appeal is"]
```

## Subtasks

### Subtask 0: Validate Docling on Python 3.14 (~15 min) -- DONE
- Docling v2.74.0 installs and works on Python 3.14.2
- Successfully parsed India Code statute PDF (53K chars structured markdown)

### Subtask 1: Foundation — Models, Exceptions, Config (~30 tests) -- DONE (37 tests)
**Files:** `_models.py`, `_exceptions.py`, `_config.py`, `__init__.py`, `configs/parsing.yaml`
- All Pydantic models (ParsedDocument, ParsedSection, ParsedTable, QualityReport)
- Exception hierarchy: `ParsingError` → `LegalRAGError`
  - `PDFDownloadError`, `DocumentStructureError`, `QualityValidationError`, `UnsupportedFormatError`
- YAML config loader (mirrors `src/acquisition/_config.py` pattern)
- Tests: model serialization, config loading, exception hierarchy

### Subtask 2: Base Parser + Router + Validation (~20 tests) -- IN PROGRESS (code written, tests pending)
**Files:** `parsers/_base.py`, `_router.py`, `_validation.py`
- `BaseParser` ABC: `parse()`, `can_parse()`, `parser_type` property
- `ParserRouter`: selects parser by source_type + content_format, handles ImportError for Docling
- `QualityValidator`: runs 4 checks, produces QualityReport
- Tests: routing logic, validation thresholds, fallback behavior

### Subtask 3: Indian Kanoon HTML Parser (~20 tests)
**Files:** `parsers/_html_indian_kanoon.py`
- Parse `div.judgments` content (judgments) and statute text (statutes)
- Detect structural sections using marker patterns
- Build ParsedSection hierarchy tree
- Tested with synthetic HTML from Phase 1 test fixtures
- Tests: judgment structure detection, statute section detection, metadata enrichment

### Subtask 4: PDF Downloader + India Code HTML Parser (~15 tests)
**Files:** `_downloader.py`, `parsers/_html_india_code.py`
- `PdfDownloader`: download from `download_url` → cache at `data/cache/pdf/{doc_id}.pdf`
  - Uses aiohttp (reuse Phase 1 pattern), atomic writes, cache-based idempotency
- `IndiaCodeHtmlParser`: extract metadata from detail page, flag need for PDF parsing
- Tests: download caching, timeout handling, HTTP errors, metadata extraction

### Subtask 5: PDF Parser — Docling or Fallback (~15 tests)
**Files:** `parsers/_docling_pdf.py` (or `_pymupdf_pdf.py` if Docling fails in Subtask 0)
- Wrap Docling `DocumentConverter` → walk `DoclingDocument` → build ParsedSection tree
- OR wrap pymupdf4llm + pdfplumber for text + table extraction
- Structure detection using statute/judgment regex patterns
- Tests: section hierarchy from PDF, table extraction, OCR confidence, `@pytest.mark.slow`

### Subtask 6: Pipeline + CLI + Integration Tests (~20 tests)
**Files:** `pipeline.py`, `run.py`, `__main__.py`
- `ParsingPipeline.run()`: scan input_dir → load RawDocument → route → parse → validate → save
- India Code flow: load meta.json → download PDF → parse → save ParsedDocument
- Indian Kanoon flow: load meta.json → parse HTML → save ParsedDocument
- Idempotent: skip already-parsed docs (check output file exists + content hash match)
- CLI: `python -m src.parsing.run --source="India Code" --log-level=INFO --console-log`
- Integration test: create temp dir with Phase 1 output structure, run pipeline, verify output

## Dependencies (pyproject.toml changes)

```toml
# Add to [project.dependencies] — always available
"pymupdf4llm>=0.0.17",     # Only if Docling fails in Subtask 0
"pdfplumber>=0.11",         # Only if Docling fails in Subtask 0

# Add to [project.optional-dependencies]
parsing = [
    "docling[easyocr]>=2.59",  # Optional: better PDF parsing with OCR
]
```

Note: exact deps depend on Subtask 0 outcome.

## Acceptance Criteria

| Criterion | How to Verify |
|-----------|---------------|
| `from src.parsing import ParsingPipeline, ParsedDocument` | Import test |
| ruff check + format clean | `python -m ruff check src/ tests/` |
| ~120 unit tests pass | `python -m pytest tests/parsing/ -x -v` |
| India Code PDF → ParsedDocument with sections | Integration test |
| Indian Kanoon HTML → ParsedDocument with structure | Parser test with synthetic data |
| Quality validation runs on every doc | Pipeline test |
| Idempotent (second run = 0 re-parses) | Pipeline idempotency test |
| ParsedDocument round-trips through JSON | Model serialization test |
| Docling/fallback gracefully handled | Router fallback test |
| Phase 3 can read ParsedDocument.sections tree | Model structure validation |

## Key Files to Reference

- `src/acquisition/_models.py` — RawDocument (Phase 2 input), reuse DocumentType/SourceType/ContentFormat enums
- `src/acquisition/_config.py` — Pattern for YAML config loading
- `src/acquisition/_exceptions.py` — Pattern for exception hierarchy
- `src/acquisition/base_scraper.py` — Pattern for template method ABC
- `docs/chunking_strategies.md` — Regex patterns for structure detection
- `docs/metadata_schema.md` — LegalChunk model (Phase 3 target, informs what ParsedDocument must carry)
- `docs/indian_legal_structure.md` — Indian statute/judgment hierarchy
