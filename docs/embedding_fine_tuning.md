# Embedding Fine-Tuning — Indian Legal Text

## Why Fine-Tune

Generic embedding models break on Indian legal text. A practitioner achieved **16% retrieval improvement** by fine-tuning BGE-base on SEBI regulatory text, with 12x storage reduction via Matryoshka Representation Learning.

**Failure modes of generic models:**
- "Debenture trustee" and "dematerialisation" have no semantic grounding
- "Section 23A refers to Section 12B" — cross-references are meaningless without domain knowledge
- Mixed English + Hindi + Latin ("mens rea", "suo motu", "locus standi") confuses tokenizers
- "Anticipatory bail", "lok adalat", "FIR" — India-specific concepts with no Western equivalent
- Section 420 IPC (cheating, repealed) vs Section 318 BNS (cheating, current) — semantically identical, legally different

## Base Model: BAAI/bge-m3

**Why bge-m3:** Multilingual (handles Hindi+English), supports 8192 token context (needed for Late Chunking), strong baseline performance, open-source.

**Alternative:** jina-embeddings-v3 — also multilingual, 8192 context, built-in Late Chunking support. Choose based on benchmarks on your specific test set.

## Training Data Creation — Synthetic Generation

**Target: ~35K clean query-document pairs** (after filtering from 50K generated).

Use Claude or GPT-4 as the primary data generation engine. Synthetic data generation is "your secret weapon" — many successful open-source models were trained this way (Alpaca, Vicuna, WizardLM).

```python
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
    POSITIVE: [query] -> [this chunk is the correct answer]
    HARD_NEGATIVE: [query] -> [this chunk is NOT the correct answer, explain why]"""

    pairs = llm.generate(prompt)
    return parse_training_pairs(pairs)
```

**Cost:** ~200 tokens per generation x 7 pairs per chunk x 100K key chunks, using Claude Haiku: ~$30-50 total. Much cheaper and faster than manual curation.

### Data Quality Pipeline (The LIMA Principle)

"500 meticulously curated, diverse, high-quality examples will often produce a better fine-tuned model than 50,000 mediocre ones."

```
Step 1: Generate 50K synthetic pairs                    → ~$30-50 cost
Step 2: Auto-filter obvious low-quality pairs            → ~40K remaining
Step 3: Manual review of 15% random sample (~6K pairs)   → ~20 hours of work
Step 4: Remove pairs flagged in review                   → ~35K clean pairs
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

**Red flags to check for:**
- Repetitive patterns (if many queries start the same way, the model learns that)
- Length bias (vary query lengths)
- Missing edge cases (include "I don't know" scenarios)
- Label noise (even 5% incorrect labels hurts meaningfully)

### Hard Negatives — Critical for Indian Law

Hard negatives teach the model to distinguish semantically similar but legally different text:
- Section 420 IPC (cheating, repealed July 2024) vs Section 318 BNS (cheating, current)
- Section 302 IPC (murder) vs Section 304 (culpable homicide not amounting to murder)
- Article 14 (equality before law) vs Article 15 (prohibition of discrimination)

Every training query must include at least one hard negative from a related-but-wrong provision.

## Fine-Tuning with QLoRA

Use QLoRA (Quantized LoRA) for hardware efficiency. Achieves 88-98% of full fine-tuning quality at dramatically reduced VRAM.

```
Standard fine-tuning BGE-m3: Requires ~24GB VRAM (full model in memory)
LoRA fine-tuning BGE-m3:     Requires ~12GB VRAM (model frozen, small adapters trained)
QLoRA fine-tuning BGE-m3:    Requires ~6-8GB VRAM (model quantized to 4-bit, adapters in bf16)
```

**Recommendation:** Start with QLoRA. If results are within 2% of quality target, ship it. If not, move to LoRA. Full fine-tuning is overkill for embedding models unless you have unlimited GPU budget.

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
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

# Step 3: Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="models/legal-bge-m3",
    num_train_epochs=4,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=16,  # effective batch: 512
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_torch_fused",
    bf16=True,
    tf32=True,
)

# Step 4: Matryoshka loss for multi-dimensional efficiency
loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[768, 512, 256, 128, 64],
)
```

**Hardware:** Single T4 GPU ($0.35/hr on cloud). ~8 hours training = ~$3.

## NEFTune — Noisy Embeddings Fine-Tuning

Add random noise to input embeddings during training. Consistently improves fine-tuned model quality across benchmarks at zero additional cost.

```python
def add_neftune_noise(embeddings, noise_alpha=5.0):
    """Add uniform noise scaled by sequence length and embedding dimension."""
    dims = embeddings.shape  # (batch, seq_len, embed_dim)
    mag = noise_alpha / (dims[1] * dims[2]) ** 0.5
    noise = torch.zeros_like(embeddings).uniform_(-mag, mag)
    return embeddings + noise

# In training loop:
for batch in dataloader:
    input_embeddings = model.get_input_embeddings()(batch.input_ids)
    noisy_embeddings = add_neftune_noise(input_embeddings)
    outputs = model(inputs_embeds=noisy_embeddings, ...)
    loss = compute_matryoshka_loss(outputs, batch.labels)
    loss.backward()
    optimizer.step()
```

**Cost:** $0 — one line of code. Acts as regularizer, making the model more robust to query variations.

## DPO — Preference Alignment (Post-Initial Training)

Direct Preference Optimization as a second training pass. Shows the model pairs of results (correct vs similar-but-wrong) and trains it to prefer the correct one.

**Why for Indian law:** Semantic similarity != legal equivalence. Section 420 IPC and Section 318 BNS are semantically almost identical but legally different. DPO teaches the embedding model to distinguish these.

```python
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

**Apply after initial QLoRA training.** Uses the hard negative pairs already generated. ~4-8 epochs on same GPU. ~$3-12 with QLoRA.

## Evaluation

Before deploying fine-tuned model, compare against base model on held-out test set:
- NDCG@10 on Indian legal retrieval queries
- MRR (Mean Reciprocal Rank) for factual queries
- Recall@20 for analytical queries (broader retrieval needed)

**Target:** >16% improvement over base model on all metrics (matching the SEBI fine-tuning result). If not achieved, iterate on training data quality first (not hyperparameters).

## Deployment

- Export to ONNX for faster CPU inference (if not using GPU for serving)
- Or serve via sentence-transformers directly
- Integrate with Late Chunking pipeline: the fine-tuned model must support long-context encoding
- Store dual Matryoshka vectors at ingestion time: `"fast"` (64-dim) + `"full"` (768-dim). See `docs/pipeline_architecture.md` Phase 5.

## Cost Summary

| Component | Cost |
|-----------|------|
| Synthetic training data generation | ~$30-50 (LLM calls) |
| GPU time for QLoRA fine-tuning | ~$3-12 (8 hrs on T4) |
| DPO second pass | ~$3-12 (8 hrs on T4) |
| Manual data review (15% sample) | ~20 hours labor |
| **Total** | **~$36-74 + 20 hrs labor** |
