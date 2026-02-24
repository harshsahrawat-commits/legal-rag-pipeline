# Enrichment Guide — Late Chunking + Contextual Retrieval

## Why Two Techniques

Late Chunking improves **vector embeddings** (semantic search). Contextual Retrieval improves **BM25 text** (keyword search). Together they give 67%+ retrieval failure reduction.

## Late Chunking Implementation

**Concept:** Embed entire document first, THEN split token embeddings at chunk boundaries. Each chunk embedding carries context from surrounding text.

**Requirements:**
- Long-context embedding model (jina-embeddings-v3: 8192 tokens)
- Chunk boundaries must be token-aligned

**Algorithm:**
1. Tokenize full document
2. Pass all tokens through transformer → get token-level embeddings (shape: [N, dim])
3. Map chunk text boundaries to token indices
4. For each chunk: slice token embeddings → mean pool → chunk embedding
5. Store in Qdrant (dual vectors: 768-dim full + 64-dim fast Matryoshka slice)

**Gotchas:**
- Documents longer than model context (8192 tokens ~ 10 pages) must be windowed with overlap
- For Indian statutes with 200+ sections, process chapter-by-chapter with chapter context
- The chunk boundaries from Phase 3 must be preserved as token indices, not just text spans

**Cost:** Same as normal embedding. Just reorders operations.

## Contextual Retrieval Implementation

**Concept:** For each chunk, prepend LLM-generated context that situates it in the document.

**Prompt template:**
```
<document>
{FULL_DOCUMENT_TEXT}
</document>

Here is a chunk from the above document:
<chunk>
{CHUNK_TEXT}
</chunk>

Generate a brief (2-3 sentence) context for this chunk. Include:
- The Act/judgment name and section/paragraph number
- The legal topic or concept being addressed
- How this relates to the surrounding content
Do NOT repeat the chunk text. Only provide context.
```

**CRITICAL: Use prompt caching.** Cache the `<document>` block. Only the `<chunk>` varies per call. This drops cost by ~90%.

**Output format:** Prepend context to chunk text. Store as `contextualized_text` field.

**Use the contextualized text for:**
1. BM25 index (adds keywords like "cheating", "IPC", "Section 420" that may not appear in raw chunk)
2. Passing to LLM during generation (richer context for answer synthesis)

**Do NOT use contextualized text for:** Vector embedding (Late Chunking handles this better).

## QuIM-RAG Question Generation

**Concept:** Pre-generate 3-5 questions per chunk. Embed questions. Match user queries against questions (tighter semantic match).

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

## Prompt Caching Strategy — All LLM Calls

Anthropic Prompt Caching reduces latency by up to 85% and costs by up to 90% for long prompts. Apply the `cache_control` pattern to **every** LLM call in the pipeline, not just Contextual Retrieval.

### Pattern: Static Prefix + Dynamic Suffix

Structure every LLM call so static instructions are at the beginning (cached) and dynamic content is at the end (varies per call).

### 1. Contextual Retrieval (Ingestion Time)

```python
# Static: full document text (cached across all chunks from same document)
# Dynamic: individual chunk text (varies per chunk)
response = anthropic_client.messages.create(
    model="claude-haiku-4-5-20251001",
    system=[{
        "type": "text",
        "text": f"<document>\n{full_document_text}\n</document>\n\nYou generate brief context for document chunks.",
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": f"<chunk>\n{chunk_text}\n</chunk>\n\nGenerate context."}]
)
```

### 2. Response Generation (Query Time)

```python
# Static: system prompt with legal rules, court hierarchy, temporal info
system_prompt = """You are Verdict.ai, an expert Indian legal research assistant.

RULES:
1. Only cite statutes and cases that appear in the provided context
2. If you are unsure, say so explicitly — never fabricate citations
3. Always specify whether a cited law is currently in force or repealed
4. Include the relevant section numbers and Act names in every citation
5. If the question involves temporal aspects, clarify which version of the law applies
6. Structure your response: Brief Answer → Detailed Analysis → Relevant Provisions → Case Law → Caveats

COURT HIERARCHY (for precedent weight):
Supreme Court > High Court > District Court > Tribunal
Constitution Bench > Full Bench > Division Bench > Single Judge

CURRENT LEGAL TRANSITION (critical):
- IPC (1860) → Bharatiya Nyaya Sanhita (2023) — effective July 1, 2024
- CrPC (1973) → Bharatiya Nagarik Suraksha Sanhita (2023) — effective July 1, 2024
- Indian Evidence Act (1872) → Bharatiya Sakshya Adhiniyam (2023) — effective July 1, 2024
Always clarify which regime applies based on the date of the cause of action."""

# Dynamic: retrieved chunks + user query
response = anthropic_client.messages.create(
    model="claude-sonnet-4-6-20250514",
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}  # ~800 tokens cached
    }],
    messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"}]
)
```

### 3. GenGround Verification (Post-Generation)

```python
# Static: verification instructions (cached across all verification calls)
verification_prompt = """You are a legal accuracy auditor for Indian law.

For each claim in the AI-generated response:
1. Is it directly supported by the source documents? (Grounded/Ungrounded)
2. Is any source mischaracterized? (Accurate/Mischaracterized)
3. Are there important caveats from sources that were omitted? (Complete/Incomplete)
4. Are any cited laws repealed or amended? (Current/Outdated)

Return a revised response that removes ungrounded claims, corrects
mischaracterizations, and adds critical omitted caveats."""

# Dynamic: specific response + evidence being audited
response = anthropic_client.messages.create(
    model="claude-haiku-4-5-20251001",
    system=[{
        "type": "text",
        "text": verification_prompt,
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}"}]
)
```

### Cost Impact

| Component | Without Caching | With Caching | Savings |
|-----------|----------------|--------------|---------|
| LLM API (generation) | $200-500/mo | $30-100/mo | 80-85% |
| LLM API (hallucination) | Included above | Included above | Same |
| Contextual Retrieval (ingestion) | ~$500 one-time | ~$50 one-time | 90% |

The system prompt (~800 tokens) stays warm under any reasonable query volume (5-minute cache TTL). First query after warm-up has normal latency; subsequent queries get up to 85% faster prefill.
