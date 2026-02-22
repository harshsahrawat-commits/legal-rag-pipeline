# Embedding Fine-Tuning — Indian Legal Text

## Why Fine-Tune

Generic embedding models break on Indian legal text. A practitioner achieved **16% retrieval improvement** by fine-tuning BGE-base on SEBI regulatory text, with 12x storage reduction via Matryoshka Representation Learning.

**Failure modes of generic models:**
- "Debenture trustee" and "dematerialisation" have no semantic grounding
- "Section 23A refers to Section 12B" — cross-references are meaningless without domain knowledge
- Mixed English + Hindi + Latin ("mens rea", "suo motu", "locus standi") confuses tokenizers
- "Anticipatory bail", "lok adalat", "FIR" — India-specific concepts with no Western equivalent

## Base Model: BAAI/bge-m3

**Why bge-m3:** Multilingual (handles Hindi+English), supports 8192 token context (needed for Late Chunking), strong baseline performance, open-source.

**Alternative:** jina-embeddings-v3 — also multilingual, 8192 context, built-in Late Chunking support. Choose based on benchmarks on your specific test set.

## Training Data Creation

**Target: 50K+ query-document pairs**

Sources:
1. **Indian Kanoon search logs** (if accessible via their API) — real lawyer queries mapped to result pages
2. **LLM-generated pairs from statutes** — for each section, generate 3-5 natural language queries a lawyer might ask
3. **Manual curation** — have lawyers write 500-1000 queries across practice areas
4. **Hard negatives** — for each query, include semantically similar but wrong sections (e.g., Section 302 IPC murder vs Section 304 culpable homicide)
5. **Cross-referential pairs** — query about concept in Section X, answer is in Section Y that references X

**Format:** Triplets (query, positive_passage, negative_passage)

## Fine-Tuning Config

From the successful SEBI fine-tuning:

```python
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

model = SentenceTransformer("BAAI/bge-m3")

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

# Use Matryoshka loss for multi-dimensional efficiency
loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[768, 512, 256, 128, 64],
)
```

**Hardware:** Single GPU with 10-12GB VRAM minimum. A100 preferred for faster training.

## Evaluation

Before deploying fine-tuned model, compare against base model on held-out test set:
- NDCG@10 on Indian legal retrieval queries
- MRR (Mean Reciprocal Rank) for factual queries
- Recall@20 for analytical queries (broader retrieval needed)

**Target:** >10% improvement over base model on all metrics. If not achieved, iterate on training data quality.

## Deployment

- Export to ONNX for faster CPU inference (if not using GPU for serving)
- Or serve via sentence-transformers directly
- Integrate with Late Chunking pipeline: the fine-tuned model must support long-context encoding
