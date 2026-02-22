# Parsing Guide — Docling Configuration

## Primary: Docling (IBM, MIT License)

**Why Docling:**
- MIT license, no vendor lock-in
- Runs locally — legal data never leaves your servers
- Advanced PDF: layout analysis, reading order, table structure (TableFormer)
- Unified DoclingDocument format preserving hierarchy
- Native integrations: LangChain, LlamaIndex

**Installation:**
```bash
pip install docling[all]  # includes OCR, table detection, VLM
```

**Pipeline modes:**
1. **Standard** (digitally-born PDFs): Fast, uses DocLayNet for layout + TableFormer for tables
2. **VLM** (scanned PDFs): Uses Granite-Docling (258M params) for end-to-end OCR+structure
3. **HTML**: Built-in BeautifulSoup pipeline for Indian Kanoon pages

**Configuration for Indian legal docs:**
```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

# For clean PDFs (Supreme Court, recent legislation)
pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    table_structure_options={"mode": "accurate"},  # slower but better
)

# For scanned PDFs (old state Acts, some HC judgments)
ocr_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options={"lang": ["eng", "hin"]},  # Hindi + English
)

converter = DocumentConverter(
    format_options={PdfFormatOption.PIPELINE_OPTIONS: pipeline_options}
)
```

**Output:** Export to markdown (preserves headers) or JSON (programmatic access with metadata).

## Fallback: LlamaParse

For batch processing 100K+ files where speed matters more than perfect structure.
- $0.003/page, ~6s regardless of page count
- Use via API: `pip install llama-parse`
- Good for simpler documents (notifications, circulars)
- Less accurate on complex table structures

## Quality Validation

Run after every parsed document:

| Check | Method | Threshold |
|---|---|---|
| Text completeness | extracted_chars / (page_count × 2000) | > 0.5 |
| Section sequence | regex extract section numbers, check gaps | No gaps |
| Table integrity | verify row_count × col_count matches cell_count | Exact match |
| OCR confidence | Tesseract per-character confidence | > 85% mean |
| Cross-validation | Compare against Indian Kanoon HTML (for overlapping docs) | > 90% text similarity |

Flag documents below thresholds for manual review queue.

## Gotchas

_(Append new discoveries here using /learn command)_
