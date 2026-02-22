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
5. Store in Qdrant

**Gotchas:**
- Documents longer than model context (8192 tokens ≈ 10 pages) must be windowed with overlap
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
