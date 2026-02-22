# Chunking Module

Read `docs/chunking_strategies.md` and `docs/indian_legal_structure.md` before making changes here.

## Key Constraint
**Never split a statute section's sub-sections, provisos, or explanations into separate chunks.** They are legally meaningless without their parent section. This is the #1 rule.

## Module Structure
- `base.py` — `BaseChunker` abstract class (all chunkers inherit from this)
- `statute_chunker.py` — Structure-boundary chunking for Acts/statutes
- `judgment_chunker.py` — Structural chunking for court judgments
- `semantic_chunker.py` — Max-Min semantic chunking (fallback for unstructured docs)
- `raptor.py` — RAPTOR summary tree builder (enhancement layer)
- `quim.py` — QuIM-RAG question generation (enhancement layer)
- `router.py` — Routes documents to appropriate chunker by type

## All Chunkers Must
1. Accept a `ParsedDocument` and return `list[LegalChunk]`
2. Respect 1500 token max per chunk
3. Populate all required metadata fields per `docs/metadata_schema.md`
4. Be idempotent (same input → same output)
