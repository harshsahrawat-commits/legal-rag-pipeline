# Verdict.ai Ingestion Pipeline — v2 Update Notes

**Date:** February 24, 2026
**Scope:** Incremental changes to Phases 1–6 (ingestion side only) of `legal-rag-ingestion-v2.md`
**Sources:** Cross-referenced against research on chunking strategies (Feb 2026), RAG latency optimization, RAG architecture variants, and LLM fine-tuning.

> **How to read this document:** Everything in the v2 pipeline that is NOT mentioned here remains unchanged. This document only covers what's new, what's upgraded, and what needs attention. Treat this as a patch on top of v2.

---

## What's Unchanged (Confirmed by New Research)

The following v2 decisions are **validated** by the new research and require no changes:

- **Phase 1 (Agentic Document Acquisition):** Fully sound. The three-agent architecture (Source Discovery, Change Detection, Legal Review) remains best practice. No new research contradicts this.

- **Phase 2 (Document Parsing):** Docling as primary parser is still the right choice. Fallback parsers (LlamaParse, Reducto, Tesseract) still appropriate. Quality validation benchmarks hold.

- **Phase 3 primary strategies:** Structure-boundary chunking for statutes and structural chunking for judgments are validated. The February 2026 taxonomy paper (arXiv 2602.16974) explicitly confirms: "simple structure-based methods often outperform sophisticated LLM-guided alternatives for standard corpus-level retrieval." Your primary chunking strategies are best-in-class.

- **Phase 4 (Dual Context Enrichment):** Late Chunking + Contextual Retrieval together remains cutting-edge. The chunking guide concludes: "Late chunking eliminates the information loss of boundary-agnostic splitting at near-zero marginal cost" and "contextual retrieval delivers dramatic retrieval improvements (67% failure reduction) as an additive layer." The latency supplement adds one nuance: "Contextual Retrieval preserves semantic coherence more effectively but requires greater computational resources. Late Chunking offers higher efficiency." Using both is still the right call.

- **Phase 5 (Metadata Schema & Knowledge Graph):** The metadata JSON schema and Neo4j graph schema are comprehensive. No changes needed.

- **QuIM-RAG (Strategy 5):** Still valuable, no changes needed.

- **Technology choices:** Qdrant, Neo4j, BGE-m3 base model, Celery for orchestration — all still appropriate.

---

## CHANGE 1: Phase 3 — Upgraded Fallback Chunking Strategy

### What's Changing

The v2 plan uses "Max-Min semantic chunking" as the sole fallback for documents without clean structural markers. New research identifies better, tiered alternatives.

### Why

The Vectara benchmark (cited in the chunking guide) found that "semantic chunking's computational costs are not consistently justified by performance gains over fixed-size chunking — benefits are highly task-dependent." The Superlinked HotpotQA evaluation confirmed that simple `SentenceSplitter` with ColBERT v2 outperformed standard `SemanticSplitter`. Max-Min chunking is better than basic semantic chunking, but it shouldn't be the only fallback.

### New Fallback Chain

Replace the single-fallback approach with a tiered system matched to document condition:

```
Document arrives for chunking
    │
    ├── Structure detected (statute sections, judgment headings)?
    │   └── YES → Structure-boundary chunking (unchanged from v2)
    │
    ├── Partial structure detected (some headings, inconsistent formatting)?
    │   └── YES → Recursive Semantic Chunking (RSC) [NEW]
    │
    ├── Definitions section or Schedule with discrete entries?
    │   └── YES → Proposition-based chunking [NEW]
    │
    ├── Degraded scanned PDF with failed structure detection?
    │   └── YES → Page-level chunking [NEW]
    │
    └── Unstructured text (circulars, notifications, law commission reports)?
        └── Max-Min semantic chunking (unchanged from v2)
```

### NEW Fallback A: Recursive Semantic Chunking (RSC)

**What it is:** A two-phase hybrid from Latif et al. (ICNLSP 2025) that combines structural awareness with semantic validation.

**How it works:**
1. First pass: Split using recursive character splitting (respects structural separators — paragraph breaks, sentence boundaries, headers)
2. Second pass: Semantic validation — if adjacent chunks are semantically similar above a threshold, merge them. If a single chunk contains semantically divergent content, split further.

**Why RSC over plain semantic chunking:**
- "Consistently outperformed both standard Recursive Character Text Splitter and standard Semantic Chunking on contextual relevancy scores" (ICNLSP 2025)
- Compute cost sits between recursive splitting (low) and full semantic chunking (medium-high) — the semantic check only runs on chunks that need it, not on every sentence
- Produces chunks that are both structurally sound (no mid-sentence splits) and semantically coherent (no topic mixing)

**When to use:** High Court judgments with inconsistent formatting, tribunal orders without clear section markers, older notifications where OCR partially preserved structure.

```python
# Pseudocode for Recursive Semantic Chunking
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-m3")

def recursive_semantic_chunk(text, max_tokens=1500, similarity_threshold=0.75):
    # Phase 1: Recursive structural splitting
    separators = ["\n\n\n", "\n\n", "\n", ". "]  # Hierarchy of boundaries
    initial_chunks = recursive_split(text, separators, max_tokens)
    
    # Phase 2: Semantic validation
    final_chunks = []
    buffer = initial_chunks[0]
    
    for i in range(1, len(initial_chunks)):
        # Embed current buffer and next chunk
        emb_buffer = embed_model.encode(buffer)
        emb_next = embed_model.encode(initial_chunks[i])
        similarity = cosine_similarity(emb_buffer, emb_next)
        
        if similarity > similarity_threshold and token_count(buffer + initial_chunks[i]) <= max_tokens:
            # High similarity + fits in token budget → merge
            buffer = buffer + "\n" + initial_chunks[i]
        else:
            # Low similarity or exceeds budget → finalize buffer, start new
            final_chunks.append(buffer)
            buffer = initial_chunks[i]
    
    final_chunks.append(buffer)  # Don't forget the last buffer
    
    # Phase 3: Split check — if any chunk is still >max_tokens and 
    # contains divergent content, split at lowest-similarity boundary
    validated_chunks = []
    for chunk in final_chunks:
        if token_count(chunk) > max_tokens:
            sub_chunks = split_at_lowest_similarity(chunk, embed_model, max_tokens)
            validated_chunks.extend(sub_chunks)
        else:
            validated_chunks.append(chunk)
    
    return validated_chunks
```

### NEW Fallback B: Proposition-Based Chunking (For Definitions)

**What it is:** Decomposes text into atomic, self-contained propositions — each encapsulating a single factoid. From the Dense X Retrieval paper (Chen et al., EMNLP 2024).

**Why for definitions:** Legal definitions sections contain discrete, independent entries. "Cognizable offence" has nothing to do with "Decree" in the same definitions chapter. Each definition should be independently retrievable. Proposition-based chunking does exactly this — it splits compound text into self-contained units where pronouns are replaced with full referents and necessary context is added.

**Impact:** Propositions improved Recall@20 by +10.1% (unsupervised) and Exact Match@100 by +5.9 to +7.8 points across 6 dense retrievers.

**When to use:** Definitions sections in Acts (Section 2 / Section 3 in most Indian statutes), Schedule entries with discrete items, glossary sections in regulatory circulars.

```python
# Pseudocode for proposition-based chunking of definitions
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
    
    # Each proposition becomes its own chunk
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

# Example output for CPC Section 2:
# Proposition 1: "Under Section 2(2) of the Code of Civil Procedure, 1908, 
#   'decree' means the formal expression of an adjudication which, so far as 
#   regards the Court expressing it, conclusively determines the rights of the 
#   parties with regard to all or any of the matters in controversy in the suit."
# Proposition 2: "Under Section 2(9) of the Code of Civil Procedure, 1908,
#   'judgment' means the statement given by the Judge of the grounds of a decree 
#   or order."
```

**Cost:** Same as QuIM-RAG question generation — an LLM call per definitions section. Since definitions sections are a small fraction of total chunks (~1-3% of any Act), cost is negligible (~$5-10 across all 800 Acts).

### NEW Fallback C: Page-Level Chunking (For Degraded Scans)

**What it is:** Treats each PDF page as a single chunk. The simplest possible strategy.

**Why:** NVIDIA's 2024 benchmark found page-level chunking achieved the highest average accuracy (0.648) with the lowest variance across financial and technical datasets — outperforming token-based chunking at every tested size (128–2,048 tokens). It's zero-cost and deterministic.

**When to use:** Very old scanned state legislation where OCR quality is below 80% and structure detection has failed. These documents are already flagged by your Phase 2 quality validation (OCR confidence <85%). Rather than losing them entirely, page-level chunking preserves what was extracted.

**Implementation:** No special code needed — most PDF loaders (Docling included) naturally return per-page output. Just don't run the structure detection or splitting logic on these documents. Flag them in metadata:

```json
{
  "chunk_strategy": "page_level",
  "ocr_confidence": 0.72,
  "requires_manual_review": true,
  "page_number": 14,
  "total_pages": 45
}
```

These chunks go into Qdrant with a lower retrieval priority weight. When retrieved, the system can display a disclaimer: "This source was extracted from a degraded scan. Verify against the original document."

**Cost:** $0 — this is actually removing processing steps for documents that would fail anyway.

### Updated Chunking Cost Impact

No meaningful cost change. RSC uses the same embedding model you're already loading. Proposition chunking adds ~$5-10 in LLM calls for definitions across all Acts. Page-level chunking costs nothing.

---

## CHANGE 2: Phase 5 — Add Parent-Document Storage for Retrieval Support

### What's Changing

Add a parent-child relationship layer in Qdrant alongside the Neo4j graph, so that at retrieval time the system can fetch surrounding context for any matched chunk.

### Why

The latency supplement identifies Parent-Document Retrieval as a standard production pattern: "Index small chunks for precise matching, but return the parent section or larger surrounding context to the LLM." The ARAGOG benchmark (April 2024) found that Sentence-Window Retrieval had the "highest retrieval precision" among tested techniques.

For legal text specifically: a sub-section might be the precise match, but the LLM needs the full section (with provisos and explanations) to generate a correct answer. Your structure-aware chunks already create natural parent-child relationships — this change just makes them retrievable.

### Implementation

At ingestion time, for every chunk, store the parent context in Qdrant metadata:

```python
# Pseudocode — add to chunk ingestion pipeline after Phase 3 chunking
def store_with_parent_context(chunk, all_chunks_from_same_document):
    # For statute chunks: parent = the full section (if chunk is sub-section)
    # or the full chapter (if chunk is a section)
    if chunk.document_type == "statute":
        parent_chunk = find_parent_section(chunk, all_chunks_from_same_document)
        # Store the parent's chunk_id for retrieval-time fetching
        chunk.metadata["parent_chunk_id"] = parent_chunk.chunk_id if parent_chunk else None
        # Also store a ±2 sibling window
        chunk.metadata["sibling_chunk_ids"] = get_sibling_chunks(
            chunk, all_chunks_from_same_document, window=2
        )
    
    # For judgment chunks: parent = the full issue section (if chunk is reasoning)
    # or the full judgment metadata (always included)
    elif chunk.document_type == "judgment":
        parent_chunk = find_parent_issue(chunk, all_chunks_from_same_document)
        chunk.metadata["parent_chunk_id"] = parent_chunk.chunk_id if parent_chunk else None
        chunk.metadata["judgment_header_chunk_id"] = get_header_chunk(
            all_chunks_from_same_document
        ).chunk_id
    
    # Store parent text in a separate key-value store (Redis or Qdrant payload)
    # for fast retrieval-time expansion
    if parent_chunk:
        parent_store.set(
            key=parent_chunk.chunk_id,
            value=parent_chunk.text,
            metadata=parent_chunk.metadata
        )
    
    return chunk
```

**Storage approach:** Use Qdrant's payload feature to store `parent_chunk_id` and `sibling_chunk_ids` as metadata on each vector. The actual parent text lives in a key-value store (Redis is ideal — fast lookups, low cost). At retrieval time, after matching a sub-section chunk, the system fetches the parent section text in 1-5ms.

### Storage Cost Impact

Minimal. You're adding a few string IDs to each Qdrant payload (~100 bytes per chunk) and storing parent text in Redis. For 500K chunks with ~100K unique parent chunks at ~2KB average:

| Component | Additional Storage | Additional Cost |
|-----------|-------------------|-----------------|
| Qdrant payload metadata | ~50MB | $0 (already self-hosted) |
| Redis parent text store | ~200MB | $0 (Redis runs on same server) |

---

## CHANGE 3: Phase 6 (Ingestion Side) — Upgraded Embedding Fine-Tuning Strategy

### What's Changing

The v2 fine-tuning plan is correct in direction but missing several practical techniques from the fine-tuning research that improve efficiency and quality.

### Five Specific Upgrades to the Fine-Tuning Plan

#### Upgrade A: Synthetic Data Generation for Training Pairs

**v2 approach:** "Collect 50K+ query-document pairs from Indian Kanoon search logs (if accessible), manually curated Q&A pairs, LLM-generated pairs from statutes."

**Problem:** Indian Kanoon search logs may not be accessible, and manual curation of 50K pairs is extremely time-consuming.

**New approach:** Use Claude or GPT-4 as your primary data generation engine. The fine-tuning guide identifies synthetic data generation as "your secret weapon" — "this is how many successful open-source models were trained (e.g., Alpaca, Vicuna, WizardLM)."

```python
# Pseudocode for synthetic training data generation
def generate_training_pairs(chunk, act_name, section_number):
    prompt = f"""You are an expert Indian lawyer. Given this legal text, generate 
    5 realistic query-document training pairs for embedding model fine-tuning.
    
    For each pair:
    - The query should be a natural legal question a practicing Indian lawyer would ask
    - The query should NOT use the exact same words as the text
    - Include 2 simple factual queries, 2 analytical queries, and 1 cross-referential query
    - Also generate 2 hard negative queries (questions that SEEM related but this 
      text does NOT answer — these are critical for fine-tuning quality)
    
    Legal text from {act_name}, Section {section_number}:
    {chunk.text}
    
    Return format:
    POSITIVE: [query] → [this chunk is the correct answer]
    HARD_NEGATIVE: [query] → [this chunk is NOT the correct answer, explain why]"""
    
    pairs = llm.generate(prompt)
    return parse_training_pairs(pairs)

# Generate for all chunks
training_data = []
for chunk in all_chunks:
    pairs = generate_training_pairs(chunk, chunk.act_name, chunk.section_number)
    training_data.extend(pairs)

# Manual quality review on 10-20% random sample
review_sample = random.sample(training_data, int(len(training_data) * 0.15))
# Flag and remove low-quality pairs
```

**Cost estimate:** At ~200 tokens per generation × 7 pairs per chunk × 100K key chunks, using Claude Haiku: ~$30-50 total. Much cheaper and faster than manual curation.

**Critical quality check:** The fine-tuning guide warns about specific red flags:
- Repetitive patterns (if many queries start the same way, the model learns that)
- Length bias (vary query lengths)
- Missing edge cases (include "I don't know" scenarios)
- Label noise (even 5% incorrect labels hurts meaningfully)

Review 10-20% manually before training.

#### Upgrade B: QLoRA for Hardware Efficiency

**v2 approach:** Standard fine-tuning on a single GPU with 10-12GB VRAM.

**New approach:** Use QLoRA (Quantized LoRA) if hardware is constrained. The fine-tuning guide rates QLoRA at "88-98% of full fine-tuning quality" while dramatically reducing VRAM requirements.

```
Standard fine-tuning BGE-m3: Requires ~24GB VRAM (full model in memory)
LoRA fine-tuning BGE-m3:     Requires ~12GB VRAM (model frozen, small adapters trained)
QLoRA fine-tuning BGE-m3:    Requires ~6-8GB VRAM (model quantized to 4-bit, adapters in bf16)
```

**Recommendation:** Start with QLoRA. If results are within 2% of your quality target, ship it. If not, move to LoRA. Full fine-tuning is overkill for embedding models unless you have unlimited GPU budget.

```python
# Pseudocode for QLoRA fine-tuning setup
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# Step 1: Quantize base model to 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

# Step 2: Configure LoRA adapters
lora_config = LoraConfig(
    r=16,                          # Rank — 16 is a good starting point for embeddings
    lora_alpha=32,                 # Scaling factor, typically 2x rank
    target_modules=["q_proj", "v_proj", "k_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none"
)

# Step 3: Train with MatryoshkaLoss (unchanged from v2)
# Hyperparameters from v2 still apply:
#   Epochs: 4, Batch: 32, LR: 2e-5, Scheduler: cosine
#   Add: NEFTune noise (see Upgrade C below)
```

**Cost saving:** QLoRA can fine-tune on a single T4 GPU ($0.35/hr on cloud) instead of requiring an A100. For 8 hours of training: ~$3 vs ~$50.

#### Upgrade C: NEFTune (Noisy Embeddings Fine-Tuning)

**What it is:** Add random noise to input embeddings during training. That's it.

**Why:** The fine-tuning guide reports this "consistently improves fine-tuned model quality across benchmarks" at zero additional cost or complexity. It acts as a regularizer — training with slightly noisy inputs makes the model more robust to variations in real queries.

```python
# Pseudocode — add to training loop
def add_neftune_noise(embeddings, noise_alpha=5.0):
    """Add uniform noise scaled by sequence length and embedding dimension."""
    dims = embeddings.shape  # (batch, seq_len, embed_dim)
    mag = noise_alpha / (dims[1] * dims[2]) ** 0.5
    noise = torch.zeros_like(embeddings).uniform_(-mag, mag)
    return embeddings + noise

# In training loop:
for batch in dataloader:
    input_embeddings = model.get_input_embeddings()(batch.input_ids)
    noisy_embeddings = add_neftune_noise(input_embeddings)  # One line added
    outputs = model(inputs_embeds=noisy_embeddings, ...)
    loss = compute_matryoshka_loss(outputs, batch.labels)
    loss.backward()
    optimizer.step()
```

**Cost:** $0 — one line of code.

#### Upgrade D: DPO for Preference Alignment (Post-Initial Training)

**What it is:** Direct Preference Optimization — show the model pairs of search results (one correct, one similar-but-wrong) and train it to prefer the correct one.

**Why for legal embeddings:** In law, semantic similarity ≠ legal equivalence. Section 420 IPC (cheating) and Section 420 BNS (cheating, post-2024) are semantically almost identical but legally different documents. DPO teaches the embedding model to distinguish between these cases — something standard contrastive learning struggles with.

**When to apply:** After initial fine-tuning, as a second training pass. Requires the hard negative pairs already generated in Upgrade A.

```python
# Pseudocode for DPO on embedding model
def create_dpo_dataset(training_pairs):
    """Convert training pairs into preference format."""
    dpo_examples = []
    for query, positive_chunk, hard_negative_chunk in training_pairs:
        dpo_examples.append({
            "query": query,
            "chosen": positive_chunk,      # Correct legal provision
            "rejected": hard_negative_chunk  # Similar but wrong provision
        })
    return dpo_examples

# Example:
# Query: "What is the punishment for cheating under current law?"
# Chosen: Section 318 BNS (current law, in force)
# Rejected: Section 420 IPC (repealed July 2024, semantically similar)
```

**Cost:** Same as initial fine-tuning — another 4-8 epochs on the same GPU. ~$3 with QLoRA.

#### Upgrade E: Data Quality Focus (The LIMA Principle)

The fine-tuning guide's strongest recommendation: "500 meticulously curated, diverse, high-quality examples will often produce a better fine-tuned model than 50,000 mediocre ones."

**Revised data strategy:**

```
Step 1: Generate 50K synthetic pairs (Upgrade A)          → ~$30-50 cost
Step 2: Auto-filter obvious low-quality pairs               → ~40K remaining
Step 3: Manual review of 15% random sample (~6K pairs)      → ~20 hours of work
Step 4: Remove pairs flagged in review                      → ~35K clean pairs
Step 5: Ensure diversity:
    ├── All 800+ Acts represented (not just popular ones)
    ├── Mix of simple factual + analytical + cross-referential queries
    ├── Mix of statutory text + judgment text + regulatory text
    ├── Hard negatives for every Act (similar but wrong provisions)
    └── Include Hindi-English mixed queries
Step 6: Split 85% train / 15% validation (stratified by Act)
Step 7: Train with QLoRA + NEFTune
Step 8: Evaluate on validation set → if <16% improvement over base, iterate on data
```

**Key insight from the guide:** "When in doubt, improve your data before tweaking hyperparameters. Data quality improvements almost always yield bigger gains than hyperparameter tuning."

### Updated Embedding Fine-Tuning Costs

| Component | v2 Estimate | Updated Estimate | Change |
|-----------|-------------|------------------|--------|
| Training data generation (synthetic) | Not estimated | ~$30-50 (LLM calls) | New |
| GPU time for fine-tuning | ~$50 (8 hrs on A100) | ~$6-12 (8 hrs on T4 with QLoRA) | ↓ 75-88% |
| DPO second pass | Not planned | ~$6-12 (8 hrs on T4) | New |
| Manual data review | Not estimated | ~20 hrs labor | New |
| **Total fine-tuning cost** | **~$50** | **~$50-75 + 20 hrs labor** | Similar $, better quality |

---

## CHANGE 4: Phase 6 (Ingestion Side) — Matryoshka Funnel Indexing

### What's Changing

Store embeddings at multiple Matryoshka dimensions in Qdrant to enable two-stage funnel retrieval at query time.

### Why

You're already planning Matryoshka Representation Learning for storage efficiency (the SEBI fine-tuning used it for "12x storage reduction"). But the latency optimization guide describes a retrieval pattern that exploits multiple resolutions. For this to work at query time, the multiple resolutions need to be stored at ingestion time.

### Implementation

At embedding time, store each chunk at two resolutions:

```python
# Pseudocode — modification to embedding storage in Phase 6
def store_matryoshka_embeddings(chunk, embedding_model):
    # Late Chunking produces full-dim embeddings (e.g., 768-dim for BGE-m3)
    full_embedding = late_chunk_embed(chunk)  # 768-dim, from v2 Phase 4
    
    # Truncate to low-dim for fast initial search
    low_dim_embedding = full_embedding[:64]   # 64-dim Matryoshka slice
    
    # Store both in Qdrant using named vectors
    qdrant_client.upsert(
        collection_name="legal_chunks",
        points=[{
            "id": chunk.chunk_id,
            "vector": {
                "full": full_embedding.tolist(),      # 768-dim for precision reranking
                "fast": low_dim_embedding.tolist()     # 64-dim for broad initial search
            },
            "payload": chunk.metadata  # Including parent_chunk_id from Change 2
        }]
    )
```

**Qdrant configuration for named vectors:**

```python
# Create collection with two named vector spaces
qdrant_client.create_collection(
    collection_name="legal_chunks",
    vectors_config={
        "fast": VectorParams(size=64, distance=Distance.COSINE),
        "full": VectorParams(size=768, distance=Distance.COSINE)
    }
)
```

### Storage Impact

| Resolution | Vectors | Size per Vector | Total Storage |
|------------|---------|-----------------|---------------|
| 64-dim (fast search) | ~3M | 256 bytes | ~768 MB |
| 768-dim (precision) | ~3M | 3,072 bytes | ~9.2 GB |
| **Total** | | | **~10 GB** |

vs. v2's single 768-dim approach: ~9.2 GB. The 64-dim layer adds only ~768 MB (~8% more storage) but enables significantly faster initial retrieval. See the Retrieval Plan for how these two layers are used at query time.

---

## CHANGE 5: RAPTOR Tree Management — Conflict Note

### What's Changing

No change to RAPTOR itself — the v2 implementation is correct. But the latency research identifies a specific conflict to be aware of:

> "RAPTOR indexing + real-time corpus updates: RAPTOR's tree must be rebuilt when documents change. Not suitable for corpora with frequent updates unless combined with incremental tree update strategies."

### Mitigation Strategy

**Build RAPTOR trees per-Act, not as a monolithic corpus tree.** When an amendment arrives:

1. Identify which Act was amended
2. Rebuild only that Act's RAPTOR tree (typically 50-200 sections → takes minutes, not hours)
3. New judgments don't modify existing trees — they're additive. Create judgment-level RAPTOR trees per practice area that can be extended incrementally

```python
# Pseudocode for incremental RAPTOR management
def handle_amendment(amended_act_name, new_section_versions):
    # Step 1: Update the section chunks in Qdrant
    for section_version in new_section_versions:
        update_chunk_in_qdrant(section_version)
        update_knowledge_graph(section_version)
    
    # Step 2: Rebuild RAPTOR tree for ONLY this Act
    act_chunks = get_all_chunks_for_act(amended_act_name)
    new_raptor_tree = build_raptor_tree(act_chunks)
    replace_raptor_tree(amended_act_name, new_raptor_tree)
    
    # Other Acts' RAPTOR trees are unaffected
    log(f"Rebuilt RAPTOR tree for {amended_act_name}. {len(act_chunks)} chunks.")

def handle_new_judgment(judgment):
    # Judgments are additive — extend existing practice-area tree
    practice_area = classify_practice_area(judgment)
    existing_tree = get_raptor_tree(practice_area)
    
    # Add new judgment as a leaf node, regenerate Level 1 summary only
    extend_raptor_tree(existing_tree, judgment.chunks, regenerate_level=1)
```

**Cost of RAPTOR rebuilds:**
- Per-Act rebuild: ~$0.50-1.00 in LLM calls (50-200 summaries)
- Frequency: Quarterly for most Acts (when amendments arrive via Gazette)
- Annual estimate for ongoing RAPTOR maintenance: ~$50-100

---

## CHANGE 6: Infrastructure — Co-Location Decision

### What's Changing

Adding an explicit infrastructure topology requirement.

### Why

The latency optimization guide states: "Co-location is the most impactful 'free' optimization. Network hops between RAG components add 10-100ms per hop. External vector DB SaaS in a different region can add 50-200ms per query."

### Requirement

At deployment time, ensure ALL ingestion and retrieval components are in the same availability zone:

```
Same machine or same AZ cluster:
├── Qdrant (vector search)
├── Neo4j (knowledge graph)
├── Redis (parent document store + semantic cache)
├── Embedding model (for query embedding at search time)
├── Reranker model (BGE-reranker-v2-m3)
└── Application server (Celery workers, API)

External API (only component that crosses network boundary):
└── LLM API (Claude/GPT) — for generation and hallucination mitigation
```

Inter-service latency target: <1ms between Qdrant/Neo4j/Redis/embedding model. This is achievable on a single server or within a cloud provider's AZ.

**Cost impact:** $0 — this is a topology decision, not additional infrastructure. Your v2 plan already self-hosts all these components. Just ensure they end up on the same machine/cluster.

---

## Updated Implementation Roadmap (Ingestion Phases Only)

Changes to the v2 timeline are marked with **[NEW]**.

### Phase 1: Foundation (Weeks 1-6)

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 1-2 | Set up Docling parsing pipeline. Ingest top 50 Central Acts. Structure-aware chunking for statutes. **[NEW] Implement proposition-based chunking for definitions sections.** | Chunk pipeline expanded |
| 3-4 | Ingest SC judgments (2023-2025). Structural chunking for judgments. **[NEW] Implement RSC fallback for judgments with inconsistent structure.** Set up Qdrant with hybrid search. | Fallback strategy added |
| 5-6 | Implement Contextual Retrieval for all chunks. **[NEW] Store parent-child chunk relationships in Qdrant payload + Redis.** **[NEW] Configure Qdrant with dual named vectors (64-dim fast + 768-dim full).** First demo. | Storage architecture upgraded |

**Cost:** ~$100 (Contextual Retrieval) + compute costs. Unchanged from v2.

### Phase 2: Intelligence Layer (Weeks 7-14)

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 7-8 | Build Neo4j knowledge graph. Ingest citation relationships. Implement temporal status tracking. | Unchanged |
| 9-10 | Implement Late Chunking. Re-embed all chunks. **[NEW] Store dual Matryoshka vectors.** Compare retrieval metrics before/after. | Dual indexing added |
| 11-12 | **[UPGRADED] Fine-tune embedding model using synthetic data generation + QLoRA + NEFTune.** Create training dataset using LLM-generated pairs from statutes. Manual review 15% sample. | Fine-tuning strategy upgraded |
| 13-14 | Implement QuIM-RAG question generation. Add RAPTOR summary trees for all 50 Acts. **[NEW] Build RAPTOR trees per-Act (not monolithic).** | RAPTOR scoping clarified |

**Cost:** ~$200 (QuIM + RAPTOR) + ~$50-75 (fine-tuning with QLoRA) + ~$30-50 (synthetic training data). Total: ~$280-325 vs. v2's ~$250.
**Savings:** GPU fine-tuning cost drops from ~$50 to ~$6-12. Net cost change is approximately zero, with better quality.

### Phase 4: Scale (Weeks 21-30) — Ingestion Side Only

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 21-24 | Scale to all Central Acts + all High Court judgments (last 5 years). **[NEW] Page-level chunking fallback for degraded scans from state legislatures.** Build agentic source discovery. | Degraded scan handling added |
| 25-28 | Multi-tenancy for firms. Continuous sync for new judgments/amendments. **[NEW] Implement incremental RAPTOR tree updates per-Act on amendment.** | RAPTOR maintenance added |

---

## Updated One-Time Ingestion Cost Projections

| Component | v2 Estimate | Updated Estimate | Notes |
|-----------|-------------|------------------|-------|
| Docling parsing (self-hosted) | $50 (GPU) | $50 | Unchanged |
| Contextual Retrieval (Claude Haiku) | ~$500 | ~$500 | Unchanged |
| QuIM-RAG question generation | ~$200 | ~$200 | Unchanged |
| RAPTOR summaries | ~$300 | ~$300 | Unchanged |
| Synthetic training data generation | — | ~$30-50 | **New** |
| Proposition chunking for definitions | — | ~$5-10 | **New** |
| Fine-tuned embedding inference | ~$100 | ~$50-75 | ↓ QLoRA savings |
| DPO second training pass | — | ~$6-12 | **New** |
| Neo4j (self-hosted) | $0 | $0 | Unchanged |
| Qdrant (self-hosted) | $0 | $0 | Unchanged |
| Redis (parent document store) | — | $0 | **New** (self-hosted) |
| **Total one-time ingestion** | **~$1,150** | **~$1,150-1,200** | ~Same cost, better quality |

---

## New Research Papers Added to Reading List

Add these to the v2 reading list under "Important (Informs Design Decisions)":

15. **"Every Chunking Strategy for RAG Pipelines in 2026"** (arXiv 2602.16974, Feb 2026)
    *Why:* Validates structure-based chunking as primary strategy. Introduces RSC as fallback. Confirms proposition-based chunking for definitions.

16. **"Recursive Semantic Chunking"** (Latif et al., ICNLSP 2025)
    *Why:* Your new fallback for partially-structured documents. Outperforms both recursive splitting and semantic chunking alone.

17. **"Dense X Retrieval: What Retrieval Granularity Should We Use?"** (Chen et al., EMNLP 2024)
    *Why:* The proposition-based chunking paper. +10.1% Recall@20 improvement.

18. **"The Complete Guide to Fine-Tuning Large Language Models"** (Feb 2026 compilation)
    *Why:* Practical guide informing QLoRA, NEFTune, DPO, and synthetic data strategies for your embedding fine-tuning.
