# Chunking Strategies — Implementation Guide

## Router Logic

```python
def get_chunker(doc_type: DocumentType) -> BaseChunker:
    match doc_type:
        case DocumentType.STATUTE:
            return StatuteBoundaryChunker(max_tokens=1500)
        case DocumentType.JUDGMENT:
            return JudgmentStructuralChunker(max_tokens=1500)
        case DocumentType.SCHEDULE:
            return PageLevelChunker()
        case _:
            return SemanticChunker(method="max_min", max_tokens=1000, overlap=0.15)
```

## Statute Boundary Chunker

**Rules:**
1. Each Section = one chunk (including ALL sub-sections, provisos, explanations)
2. Section header MUST include: section number + Act name + chapter
3. If section > 1500 tokens: split at sub-section boundaries, keeping proviso/explanation with their immediate parent
4. Definitions section: each defined term = one chunk (with section header)
5. Cross-references within text (e.g., "subject to Section 12") → store as metadata, don't expand

**Section detection regex patterns for Indian statutes:**
```python
SECTION_PATTERN = r"^(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?)"
SUBSECTION_PATTERN = r"^\((\d+)\)"
CLAUSE_PATTERN = r"^\(([a-z])\)"
PROVISO_PATTERN = r"^Provided\s+that"
EXPLANATION_PATTERN = r"^Explanation\.?"
```

## Judgment Structural Chunker

**Rules:**
1. Detect structural sections via heading patterns and transitional phrases
2. Header/metadata = compact chunk (parties, court, date, bench)
3. Facts: split at paragraph boundaries if > 1500 tokens
4. Each issue + its full reasoning = one chunk. NEVER split an issue's reasoning.
5. Holding/ratio decidendi = always its own chunk
6. Dissenting opinions = separate chunks, clearly tagged

**Detection patterns:**
```python
FACTS_MARKERS = ["facts of the case", "brief facts", "factual matrix", "the case of the prosecution"]
ISSUES_MARKERS = ["issues for consideration", "questions of law", "points for determination"]
HOLDING_MARKERS = ["in view of the above", "for the foregoing reasons", "we hold that", "appeal is"]
DISSENT_MARKERS = ["per", "dissenting", "i am unable to agree"]
```

## Semantic Chunker (Max-Min Method)

For documents without clean structural markers. From Springer 2025 paper (AMI 0.85-0.90).

**Algorithm:**
1. Split into sentences (use spaCy sentence boundary detection — handles legal text better than nltk)
2. Embed each sentence
3. Compute cosine similarity between consecutive sentence pairs
4. Calculate split threshold: use percentile method (e.g., 25th percentile of similarities)
5. Split where similarity drops below threshold
6. Merge tiny chunks (< 100 tokens) with their neighbors

## RAPTOR Layer (Applied on Top of Base Chunks)

Build per-Act summary trees. Use Claude Haiku.

**Process:**
1. Collect all base chunks for one Act
2. Level 2 = base chunks (already exist)
3. Level 1 = cluster chunks by chapter → summarize each chapter cluster (one LLM call per chapter)
4. Level 0 = summarize all chapter summaries → Act-level summary (one LLM call)
5. Store summaries as additional chunks with `chunk_type = "raptor_summary"` and `raptor_level` metadata
6. During retrieval: search across all levels simultaneously
