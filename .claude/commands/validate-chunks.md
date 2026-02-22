Run a quality validation pass on chunked output.

1. Read `docs/chunking_strategies.md` and `docs/metadata_schema.md`
2. Load chunks from the path specified by the user (or default: `data/chunks/`)
3. Validate each chunk against these rules:

**Schema validation:**
- Every chunk deserializes to a valid `LegalChunk` Pydantic model
- Required fields are present: id, document_id, text, document_type, chunk_type, source, content, ingestion

**Content validation:**
- No chunk exceeds 1500 tokens
- No chunk is below 50 tokens (suspiciously small)
- Statute chunks include section number in metadata
- Judgment chunks have case_citation in metadata
- `text` field is not empty or whitespace-only

**Structure validation (statutes):**
- Section numbers are sequential within each Act (no gaps)
- Every proviso/explanation is attached to a parent section (not orphaned)
- Definitions section has `chunk_type = "definition"`

**Cross-reference validation:**
- `sections_cited` contains valid section format strings
- `acts_cited` matches known Act names in the knowledge base

Report: total chunks, pass/fail counts per check, list of failed chunks with reasons.
Save report to `data/reports/chunk_validation_{timestamp}.json`.
