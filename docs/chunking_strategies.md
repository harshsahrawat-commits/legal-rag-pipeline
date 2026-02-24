# Chunking Strategies — Implementation Guide

## Tiered Routing Logic

Documents are routed to chunkers based on type AND document condition, not type alone. The router evaluates conditions in priority order:

```
Document arrives for chunking (ParsedDocument from Phase 2)
    │
    ├── Structure detected (statute sections, judgment headings)?
    │   ├── STATUTE → StatuteBoundaryChunker
    │   └── JUDGMENT → JudgmentStructuralChunker
    │
    ├── Definitions section (Section 2/3 of most Acts)?
    │   └── YES → PropositionChunker (LLM-based, per-definition atomic chunks)
    │
    ├── Partial structure (some headings, inconsistent formatting)?
    │   └── YES → RecursiveSemanticChunker (RSC)
    │
    ├── Degraded scanned PDF (OCR confidence < 80%)?
    │   └── YES → PageLevelChunker (with requires_manual_review flag)
    │
    └── Unstructured text (circulars, notifications, law commission reports)?
        └── SemanticChunker (Max-Min method)
```

```python
def get_chunker(doc: ParsedDocument) -> BaseChunker:
    # Priority 1: Well-structured documents → structure-aware chunking
    if doc.document_type == DocumentType.STATUTE and has_section_structure(doc):
        return StatuteBoundaryChunker(max_tokens=1500)
    if doc.document_type == DocumentType.JUDGMENT and has_heading_structure(doc):
        return JudgmentStructuralChunker(max_tokens=1500)

    # Priority 2: Definitions sections → proposition-based atomic chunks
    if is_definitions_section(doc):
        return PropositionChunker(max_tokens=1500)

    # Priority 3: Partial structure → recursive semantic hybrid
    if has_partial_structure(doc):
        return RecursiveSemanticChunker(max_tokens=1500, similarity_threshold=0.75)

    # Priority 4: Degraded scans → page-level (preserve what OCR extracted)
    if doc.quality_report and doc.quality_report.ocr_confidence:
        if doc.quality_report.ocr_confidence.score < 0.80:
            return PageLevelChunker()

    # Priority 5: Schedules/tables
    if doc.document_type == DocumentType.SCHEDULE:
        return PageLevelChunker()

    # Default: Unstructured text → semantic chunking
    return SemanticChunker(method="max_min", max_tokens=1000, overlap=0.15)
```

---

## Strategy 1: Statute Boundary Chunker

**Rules:**
1. Each Section = one chunk (including ALL sub-sections, provisos, explanations)
2. Section header MUST include: section number + Act name + chapter
3. If section > 1500 tokens: split at sub-section boundaries, keeping proviso/explanation with their immediate parent
4. Definitions section: delegate to PropositionChunker (Strategy 4) for per-definition atomic chunks
5. Cross-references within text (e.g., "subject to Section 12") → store as metadata, don't expand

**Section detection regex patterns for Indian statutes:**
```python
SECTION_PATTERN = r"^(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+)?)"
SUBSECTION_PATTERN = r"^\((\d+)\)"
CLAUSE_PATTERN = r"^\(([a-z])\)"
PROVISO_PATTERN = r"^Provided\s+that"
EXPLANATION_PATTERN = r"^Explanation\.?"
```

---

## Strategy 2: Judgment Structural Chunker

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

---

## Strategy 3: Recursive Semantic Chunker (RSC)

**NEW — from Latif et al. (ICNLSP 2025)**

For documents with partial structure: High Court judgments with inconsistent formatting, tribunal orders without clear section markers, older notifications where OCR partially preserved structure.

RSC consistently outperformed both standard Recursive Character Text Splitter and standard Semantic Chunking on contextual relevancy scores. Compute cost sits between recursive splitting (low) and full semantic chunking (medium-high).

**Algorithm — three phases:**

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-m3")

def recursive_semantic_chunk(text, max_tokens=1500, similarity_threshold=0.75):
    # Phase 1: Recursive structural splitting
    # Respect structural separators in priority order
    separators = ["\n\n\n", "\n\n", "\n", ". "]
    initial_chunks = recursive_split(text, separators, max_tokens)

    # Phase 2: Semantic merge/split validation
    final_chunks = []
    buffer = initial_chunks[0]

    for i in range(1, len(initial_chunks)):
        emb_buffer = embed_model.encode(buffer)
        emb_next = embed_model.encode(initial_chunks[i])
        similarity = cosine_similarity(emb_buffer, emb_next)

        if similarity > similarity_threshold and token_count(buffer + initial_chunks[i]) <= max_tokens:
            # High similarity + fits budget → merge
            buffer = buffer + "\n" + initial_chunks[i]
        else:
            # Low similarity or exceeds budget → finalize, start new
            final_chunks.append(buffer)
            buffer = initial_chunks[i]

    final_chunks.append(buffer)

    # Phase 3: Split oversized chunks at lowest-similarity boundary
    validated_chunks = []
    for chunk in final_chunks:
        if token_count(chunk) > max_tokens:
            sub_chunks = split_at_lowest_similarity(chunk, embed_model, max_tokens)
            validated_chunks.extend(sub_chunks)
        else:
            validated_chunks.append(chunk)

    return validated_chunks
```

**When to use:** HC judgments with inconsistent formatting, tribunal orders without clear section markers, older notifications where OCR partially preserved structure.

**Why RSC over plain semantic chunking:** The semantic check only runs on chunks that need it (merge/split boundaries), not on every sentence. Produces chunks that are both structurally sound (no mid-sentence splits) and semantically coherent (no topic mixing).

---

## Strategy 4: Proposition-Based Chunker (For Definitions)

**NEW — from Dense X Retrieval (Chen et al., EMNLP 2024)**

Decomposes definitions sections into atomic, self-contained propositions. Each definition becomes independently retrievable with full context.

**Why for definitions:** Legal definitions sections contain discrete, independent entries. "Cognizable offence" has nothing to do with "Decree" in the same definitions chapter. Proposition-based chunking produces self-contained units where necessary context is included.

**Impact:** +10.1% Recall@20 (unsupervised), +5.9 to +7.8 Exact Match@100 across 6 dense retrievers.

**When to use:** Definitions sections in Acts (Section 2/3 in most Indian statutes), Schedule entries with discrete items, glossary sections in regulatory circulars.

```python
def proposition_chunk_definitions(definitions_text, act_name, section_number):
    prompt = f"""Extract each individual definition from this legal text as a
    self-contained proposition. Each proposition must:
    1. Include the defined term
    2. Include the full definition
    3. Include the Act name and section for context
    4. Be understandable without any surrounding text

    Legal text from {act_name}, {section_number}:
    {definitions_text}

    Return each definition as a separate proposition."""

    propositions = llm.generate(prompt)

    chunks = []
    for prop in propositions:
        chunks.append({
            "text": prop,
            "chunk_strategy": "proposition",
            "source_section": section_number,
            "source_act": act_name,
            "chunk_type": "definition"
        })
    return chunks
```

**Example output for CPC Section 2:**
- "Under Section 2(2) of the Code of Civil Procedure, 1908, 'decree' means the formal expression of an adjudication which, so far as regards the Court expressing it, conclusively determines the rights of the parties..."
- "Under Section 2(9) of the Code of Civil Procedure, 1908, 'judgment' means the statement given by the Judge of the grounds of a decree or order."

**Cost:** LLM call per definitions section. ~1-3% of any Act → ~$5-10 across all 800 Acts.

---

## Strategy 5: Semantic Chunker (Max-Min Method)

For documents without clean structural markers. From Springer 2025 paper (AMI 0.85-0.90).

**Algorithm:**
1. Split into sentences (use spaCy sentence boundary detection — handles legal text better than nltk)
2. Embed each sentence
3. Compute cosine similarity between consecutive sentence pairs
4. Calculate split threshold: use percentile method (e.g., 25th percentile of similarities)
5. Split where similarity drops below threshold
6. Merge tiny chunks (< 100 tokens) with their neighbors

**When to use:** Circulars, notifications, law commission reports, and any unstructured text without headings or sections.

---

## Strategy 6: Page-Level Chunker

Treats each PDF page as a single chunk. The simplest possible strategy.

NVIDIA's 2024 benchmark found page-level chunking achieved the highest average accuracy (0.648) with the lowest variance across financial and technical datasets.

**When to use:**
- Schedules and tables that span full pages
- Degraded scanned state legislation where OCR quality is below 80% and structure detection has failed (flagged by Phase 2 quality validation)

**For degraded scans:** These are documents that would fail structure detection anyway. Page-level chunking preserves what was extracted. Flag in metadata:

```json
{
  "chunk_strategy": "page_level",
  "ocr_confidence": 0.72,
  "requires_manual_review": true,
  "page_number": 14,
  "total_pages": 45
}
```

Chunks go into Qdrant with lower retrieval priority weight. When retrieved, display disclaimer: "This source was extracted from a degraded scan. Verify against the original document."

**Cost:** $0 — removes processing steps for documents that would fail anyway.

---

## RAPTOR Layer (Applied on Top of Base Chunks)

Build **per-Act** summary trees (NOT monolithic corpus trees). Use Claude Haiku.

**Process:**
1. Collect all base chunks for one Act
2. Level 2 = base chunks (already exist)
3. Level 1 = cluster chunks by chapter → summarize each chapter cluster (one LLM call per chapter)
4. Level 0 = summarize all chapter summaries → Act-level summary (one LLM call)
5. Store summaries as additional chunks with `chunk_type = "raptor_summary"` and `raptor_level` metadata
6. During retrieval: search across all levels simultaneously

### RAPTOR Tree Management — Amendment Handling

When an Act is amended, rebuild ONLY that Act's tree:

```python
def handle_amendment(amended_act_name, new_section_versions):
    # Step 1: Update section chunks in Qdrant
    for section_version in new_section_versions:
        update_chunk_in_qdrant(section_version)
        update_knowledge_graph(section_version)

    # Step 2: Rebuild RAPTOR tree for ONLY this Act
    act_chunks = get_all_chunks_for_act(amended_act_name)
    new_raptor_tree = build_raptor_tree(act_chunks)
    replace_raptor_tree(amended_act_name, new_raptor_tree)
    # Other Acts' trees are unaffected

def handle_new_judgment(judgment):
    # Judgments are additive — extend existing practice-area tree
    practice_area = classify_practice_area(judgment)
    existing_tree = get_raptor_tree(practice_area)
    # Add as leaf node, regenerate Level 1 summary only
    extend_raptor_tree(existing_tree, judgment.chunks, regenerate_level=1)
```

**Cost:** ~$0.50-1.00 per Act rebuild. Quarterly frequency for most Acts. ~$50-100/year for maintenance.

---

## QuIM-RAG Layer (Applied on Top of Base Chunks)

Pre-generate 3-5 questions per chunk. Embed questions. Match user queries against questions (tighter semantic match).

**Prompt:**
```
Given this legal text from {act_name}, {section_number}:
"{chunk_text}"

Generate 3-5 practical questions a lawyer might ask that this text could answer.
Focus on: how lawyers phrase real queries, not academic questions.
Format: one question per line, no numbering.
```

**Store:** Embed each question with the same model. Store in separate Qdrant collection `quim_questions` with `source_chunk_id` reference.

**Retrieval:** During search, also query `quim_questions` collection → map results back to source chunks.

---

## Universal Chunking Constraints

- **Max chunk size:** 1500 tokens for all strategies
- **Overlap:** Not used for structure-boundary chunks (boundaries are semantic). 10-20% for semantic/RSC chunks.
- **Never split a statute section's sub-sections, provisos, or explanations from their parent section.** They are legally meaningless without their parent. This is the #1 rule.
- **Every chunk must carry full metadata** per `docs/metadata_schema.md` — including `chunk_strategy` field identifying which strategy produced it.
