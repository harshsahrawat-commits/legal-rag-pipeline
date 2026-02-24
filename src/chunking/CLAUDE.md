# Chunking Module

Read `docs/chunking_strategies.md` and `docs/indian_legal_structure.md` before making changes here.

## Key Constraint
**Never split a statute section's sub-sections, provisos, or explanations into separate chunks.** They are legally meaningless without their parent section. This is the #1 rule.

## Module Structure
- `base.py` — `BaseChunker` abstract class (all chunkers inherit from this)
- `statute_chunker.py` — Structure-boundary chunking for Acts/statutes
- `judgment_chunker.py` — Structural chunking for court judgments
- `rsc_chunker.py` — Recursive Semantic Chunking (RSC) — hybrid structural+semantic fallback for partially-structured documents
- `proposition_chunker.py` — Proposition-based chunking for definitions sections (LLM-decomposed atomic chunks)
- `semantic_chunker.py` — Max-Min semantic chunking (fallback for fully unstructured docs)
- `page_chunker.py` — Page-level chunking for degraded scans and schedules
- `raptor.py` — RAPTOR summary tree builder (enhancement layer, per-Act trees)
- `quim.py` — QuIM-RAG question generation (enhancement layer)
- `router.py` — Tiered decision tree routing by document type AND condition

## Router Priority Order
1. Well-structured statute → `StatuteBoundaryChunker`
2. Well-structured judgment → `JudgmentStructuralChunker`
3. Definitions section (Section 2/3) → `PropositionChunker`
4. Partial structure detected → `RecursiveSemanticChunker`
5. Degraded scan (OCR < 80%) → `PageLevelChunker`
6. Schedule/table → `PageLevelChunker`
7. Default (unstructured) → `SemanticChunker`

## All Chunkers Must
1. Accept a `ParsedDocument` and return `list[LegalChunk]`
2. Respect 1500 token max per chunk
3. Populate all required metadata fields per `docs/metadata_schema.md`
4. Set `chunk_strategy` field to identify which strategy produced the chunk
5. Set `parent_info.parent_chunk_id` for sub-section and reasoning chunks
6. Set `parent_info.judgment_header_chunk_id` for all judgment chunks
7. Be idempotent (same input → same output)

## Dependencies
- `sentence-transformers` — for RSC and Max-Min semantic chunking (embedding similarity)
- `anthropic` — for PropositionChunker (LLM decomposition) and QuIM-RAG (question generation)
- `tiktoken` or `transformers` — for token counting
