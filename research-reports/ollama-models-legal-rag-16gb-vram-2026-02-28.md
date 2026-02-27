# Ollama Models for Legal RAG Pipeline -- RTX 5060 Ti 16GB VRAM

**Research Date:** 2026-02-28
**Scope:** Models available on Ollama (ollama.com) suitable for a Legal RAG pipeline with 7 distinct task types, constrained to 16GB VRAM (NVIDIA RTX 5060 Ti). Covers parameter counts, quantization levels, VRAM estimates, benchmark scores, and structured output capabilities.
**Confidence Level:** High (based on multiple cross-referenced sources including benchmarks, VRAM calculators, and real-world user reports)

---

## Executive Summary

For a Legal RAG pipeline running on an RTX 5060 Ti with 16GB VRAM, the practical model range is **8B to 20B parameters at Q4_K_M quantization**. The 70B+ models (Llama 3.3 70B, Qwen 2.5 72B, DeepSeek-R1 full) are categorically excluded -- they require 35-48GB VRAM even at aggressive 4-bit quantization.

The top three recommendations for your specific use case are:

1. **Qwen3 14B (Q4_K_M)** -- Best overall for legal RAG tasks. Strongest instruction following, native structured JSON output, dual thinking/non-thinking mode, tool calling support, multilingual (100+ languages including Hindi), and fits comfortably at ~10.7GB VRAM. The thinking mode provides chain-of-thought for complex legal reasoning (GenGround, FLARE), while non-thinking mode delivers fast responses for simpler tasks (enrichment, question generation).

2. **Phi-4-Reasoning 14B (Q4_K_M)** -- Best pure reasoning quality per parameter. Outperforms DeepSeek-R1 Distill 70B (a 5x larger model) on reasoning benchmarks. Highest instruction-following score (IFBench 0.834-0.849) of any model in this VRAM class. Ideal for claim verification (GenGround) and confidence assessment (FLARE). ~11GB VRAM.

3. **GPT-OSS 20B (Q4_K_M)** -- Fastest inference at ~140 tokens/sec (fully GPU-resident at ~12-14GB). Native function calling and reasoning effort control. Strong coding and reasoning benchmarks. However, has a known structured output bug in Ollama (open issue #11691) that makes JSON schema enforcement unreliable as of February 2026.

The recommended strategy is a **multi-model approach**: use Qwen3 14B as the primary workhorse for 5 of 7 tasks, and swap in Phi-4-Reasoning for the two tasks requiring deepest analytical reasoning (GenGround claim extraction and FLARE confidence assessment). Both models fit in 16GB VRAM and can be hot-swapped via Ollama's model loading.

## Table of Contents

1. [Hardware Analysis](#1-hardware-analysis)
2. [VRAM Budget Breakdown](#2-vram-budget-breakdown)
3. [Quantization Primer](#3-quantization-primer)
4. [Complete Model Catalog (Fits 16GB)](#4-complete-model-catalog-fits-16gb)
5. [Detailed Model Analysis](#5-detailed-model-analysis)
6. [Task-to-Model Mapping](#6-task-to-model-mapping)
7. [Legal-Specific Models](#7-legal-specific-models)
8. [Multi-Model Strategy](#8-multi-model-strategy)
9. [Optimization Tips for RTX 5060 Ti](#9-optimization-tips-for-rtx-5060-ti)
10. [Key Findings and Insights](#10-key-findings-and-insights)
11. [Risks, Limitations, and Open Questions](#11-risks-limitations-and-open-questions)
12. [Final Ranked Recommendation](#12-final-ranked-recommendation)
13. [Sources and References](#13-sources-and-references)

---

## 1. Hardware Analysis

### RTX 5060 Ti 16GB Specifications

| Spec | Value |
|------|-------|
| GPU Architecture | NVIDIA Blackwell |
| VRAM | 16GB GDDR7 |
| Memory Bandwidth | ~448 GB/s (estimated) |
| FP16 Performance | ~30-35 TFLOPS |
| FP8 (native) | Yes -- Blackwell supports FP8 natively |
| CUDA Cores | ~4,608 |
| TDP | ~150W |

**Key advantage of RTX 5060 Ti over previous-gen 16GB cards (RTX 4060 Ti):** Native FP8 support via Blackwell architecture. This means models quantized with MXFP4 can run natively faster. Benchmark data shows GPT-OSS-20B with MXFP4 achieves 488 tokens/sec on RTX 5060 Ti for short-context workloads [1].

The RTX 5060 Ti 16GB with native FP8 support provides approximately 80-85% of the performance of RTX 3090 24GB (at FP16), while consuming significantly less power [2].

### System Context

- **CPU:** AMD Ryzen 5 9600X (6-core, 3.90 GHz) -- strong single-thread, good for prompt processing
- **RAM:** 32GB -- sufficient for CPU offloading overflow (Ollama can spill KV cache to system RAM)
- **OS:** Windows 64-bit (MINGW64)

### Hard Constraint

**16GB VRAM is the binding constraint.** The model weights + KV cache + CUDA overhead must total under ~15.5GB (leaving ~0.5GB for the OS/display driver). Models that exceed this will trigger CPU offloading, which degrades throughput by 3-11x based on benchmarks.

---

## 2. VRAM Budget Breakdown

The total VRAM consumption follows this formula [3]:

```
VRAM_total = Model_weights + KV_cache + CUDA_overhead

Where:
  Model_weights = Parameters(B) * bytes_per_weight
  KV_cache = 2 * num_layers * num_heads * head_dim * context_length * batch_size * kv_precision
  CUDA_overhead = 0.55 GB + (0.08 * Parameters_in_B) GB
```

### Bytes Per Weight by Quantization

| Quantization | Bits | Bytes/Param | Compression vs FP16 | Quality Retention |
|-------------|------|-------------|---------------------|-------------------|
| FP16 | 16 | 2.0 | 1x (baseline) | 100% |
| Q8_0 | 8 | 1.0 | 2x | 99.5% |
| Q6_K | 6.5 | 0.81 | 2.5x | 99.0% |
| Q5_K_M | 5.5 | 0.68 | 2.9x | 98.2% |
| **Q4_K_M** | **4.5** | **0.57** | **3.6x** | **97.1%** |
| Q3_K_M | 3.5 | 0.44 | 4.5x | 94-95% |
| Q2_K | 2.5 | 0.31 | 6.5x | 85-90% |

**Q4_K_M is the recommended quantization for 16GB VRAM.** It provides 97.1% quality retention -- the difference from FP16 is imperceptible for most tasks. Going lower to Q3_K_M introduces noticeable quality degradation and unpredictable behavior, especially for structured output [4].

### VRAM Examples at Q4_K_M (8K Context, FP16 KV Cache)

| Model Size | Weights | KV Cache (8K) | Overhead | Total | Fits 16GB? |
|-----------|---------|---------------|----------|-------|------------|
| 3B | 1.7 GB | ~0.5 GB | 0.8 GB | ~3.0 GB | Yes |
| 8B | 4.6 GB | ~1.0 GB | 1.2 GB | ~6.8 GB | Yes |
| 12B | 6.8 GB | ~1.5 GB | 1.5 GB | ~9.8 GB | Yes |
| 14B | 8.0 GB | ~1.7 GB | 1.7 GB | ~11.4 GB | Yes |
| 20B | 11.4 GB | ~2.0 GB | 2.1 GB | ~15.5 GB | Tight (8K only) |
| 24B | 13.7 GB | ~2.2 GB | 2.5 GB | ~18.4 GB | No (overflows) |
| 32B | 18.2 GB | ~2.5 GB | 3.1 GB | ~23.8 GB | No |
| 70B | 39.9 GB | ~5.0 GB | 6.2 GB | ~51.1 GB | No |

**Critical insight:** Context length dramatically affects VRAM. A 14B model at Q4_K_M uses ~11GB at 8K context but ~14GB at 32K context. For your legal RAG tasks, most operations (enrichment, question generation, claim extraction) use short prompts (2K-4K tokens), so 8K context is sufficient for most tasks. Only answer generation might benefit from longer context.

---

## 3. Quantization Primer

### Recommended Quantizations for 16GB VRAM

| Model Size | Best Quantization | VRAM (8K ctx) | Quality | Speed |
|-----------|------------------|---------------|---------|-------|
| 3B | Q8_0 | ~4.5 GB | Excellent | Very fast |
| 8B | Q5_K_M or Q4_K_M | 6.2-7.3 GB | Very good | Fast |
| 12B | Q4_K_M | ~9.0 GB | Good | Moderate |
| 14B | Q4_K_M | ~10.7-11 GB | Good | Moderate |
| 20B | Q4_K_M | ~12-14 GB | Good | Moderate |
| 24B | Q3_K_M or IQ4_NL | ~14-16 GB | Marginal | Slow (overflow) |

**Do not go below Q4_K_M** for legal tasks. Q3_K_M and lower quantizations show increased refusal rates and degraded coherence in complex reasoning tasks [5]. For legal reasoning where precision matters, Q4_K_M is the floor.

### KV Cache Quantization (Ollama Feature)

Ollama supports KV cache quantization via the `OLLAMA_KV_CACHE_TYPE` environment variable [6]:

| KV Cache Type | Memory vs FP16 | Quality Impact |
|--------------|----------------|----------------|
| f16 (default) | 1x | None |
| q8_0 | 0.5x | Negligible |
| q4_0 | 0.25x | Small, noticeable at long contexts |

**Recommendation:** Set `OLLAMA_KV_CACHE_TYPE=q8_0` to halve KV cache memory with no meaningful quality loss. This effectively gives you 1-2GB of extra headroom, potentially allowing a 20B model to run at 16K context instead of 8K.

---

## 4. Complete Model Catalog (Fits 16GB)

All models below are available on Ollama (ollama.com/library) and fit within 16GB VRAM at the specified quantization with 8K context.

### Tier 1: Primary Candidates (14B-20B, Best Quality)

| Model | Params | Quant | VRAM (8K) | tok/s (est.) | Context | Tool Calling | JSON Output |
|-------|--------|-------|-----------|-------------|---------|-------------|-------------|
| **Qwen3 14B** | 14B | Q4_K_M | 10.7 GB | 50-62 | 32K+ | Yes (native) | Yes (native) |
| **Phi-4-Reasoning 14B** | 14B | Q4_K_M | 11 GB | 45-55 | 16K | Partial | Yes (prompt) |
| **GPT-OSS 20B** | 20B (MoE) | Q4_K_M | 12-14 GB | 130-140 | 120K | Yes (native) | Buggy [7] |
| **DeepSeek-R1-Distill-Qwen 14B** | 14B | Q4_K_M | 11 GB | 45-55 | 64K | No | Yes (prompt) |
| **Phi-4 14B** (base) | 14B | Q4_K_M | 11 GB | 50-60 | 16K | Yes | Yes (prompt) |
| **Qwen 2.5 14B Instruct** | 14B | Q4_K_M | 10.7 GB | 50-60 | 128K | Yes | Yes (native) |

### Tier 2: Mid-Range (8B-12B, Good Balance)

| Model | Params | Quant | VRAM (8K) | tok/s (est.) | Context | Tool Calling | JSON Output |
|-------|--------|-------|-----------|-------------|---------|-------------|-------------|
| **Llama 3.1 8B Instruct** | 8B | Q4_K_M | 6.2 GB | 70-89 | 128K | Yes | Yes |
| **Llama 3.1 8B Instruct** | 8B | Q5_K_M | 7.3 GB | 65-80 | 128K | Yes | Yes |
| **Gemma 3 12B** | 12B | Q4_K_M | 10 GB | 40-55 | 128K | Via custom [8] | Yes |
| **Mistral Nemo 12B** | 12B | Q4_K_M | 9 GB | 50-65 | 128K | Yes | Yes |
| **Gemma 2 9B** | 9B | Q4_K_M | 5.7 GB | 55-70 | 8K | No | Yes (prompt) |
| **Llama 3.3 8B** | 8B | Q4_K_M | 6.2 GB | 70-89 | 128K | Yes | Yes |
| **DeepSeek-R1-Distill-Llama 8B** | 8B | Q4_K_M | 6.2 GB | 60-68 | 64K | No | Yes (prompt) |

### Tier 3: Lightweight (3B-4B, Speed Priority)

| Model | Params | Quant | VRAM (8K) | tok/s (est.) | Context | Tool Calling | JSON Output |
|-------|--------|-------|-----------|-------------|---------|-------------|-------------|
| **Qwen3 4B** | 4B | Q4_K_M | 4 GB | 120-150 | 32K | Yes | Yes |
| **Llama 3.2 3B** | 3B | Q4_K_M | 3.6 GB | 150-180 | 128K | Yes | Yes |
| **Gemma 3 4B** | 4B | Q4_K_M | ~4 GB | 120-140 | 128K | Via custom | Yes |
| **Phi-3 Mini 3.8B** | 3.8B | Q4_K_M | ~3.5 GB | 150+ | 128K | No | Yes (prompt) |

### Models That Do NOT Fit (Excluded)

| Model | Params | VRAM at Q4_K_M | Reason |
|-------|--------|---------------|--------|
| Llama 3.3 70B | 70B | ~40 GB | 2.5x over budget |
| Qwen 2.5 72B | 72B | ~41 GB | 2.5x over budget |
| Mistral Small 3.1/3.2 24B | 24B | ~19 GB | 3GB over (would require heavy CPU offload) |
| DeepSeek-R1 671B | 671B | ~380 GB | Requires a cluster |
| Gemma 3 27B | 27B | ~18-22 GB | Significant overflow; also has VRAM leak issues [9] |
| Qwen3 32B | 32B | ~22-24 GB | 6-8GB over budget |
| Devstral Small 24B | 24B | ~19 GB | 3GB over budget |
| Nemotron Nano 30B | 30B | ~25 GB | 9GB over budget |

**Note on Mistral Small 3.1 24B:** Some community sources claim it can fit in 16GB at Q4_K_M, but real-world measurements show 18-19GB usage. The discrepancy comes from not accounting for KV cache and CUDA overhead. At very short contexts (2K) with Q4_0 KV cache it might barely fit, but inference would be extremely slow due to partial CPU offloading (~18 tok/s vs ~50 tok/s for the 14B models) [10].

---

## 5. Detailed Model Analysis

### 5.1 Qwen3 14B -- RECOMMENDED PRIMARY MODEL

**Ollama tag:** `qwen3:14b`

| Attribute | Value |
|-----------|-------|
| Parameters | 14B |
| Architecture | Dense Transformer |
| VRAM (Q4_K_M, 8K ctx) | 10.7 GB |
| VRAM (Q4_K_M, 32K ctx) | 13.6 GB |
| Speed (Q4_K_M, 8K) | ~50-62 tok/s |
| Context Window | 32K+ |
| Languages | 100+ (including Hindi, Urdu) |
| License | Apache 2.0 |

**Benchmarks:**
- MMLU-Pro: 77.4% (surpasses GPT-OSS 20B's 74.8%)
- GPQA: 60.4%
- Math 500: 96.1%
- AIME: 76.3%
- IFEval: High (exact score not published, but Qwen3 family is top-tier)

**Key Strengths for Legal RAG:**
1. **Dual thinking/non-thinking mode:** Add `/think` for chain-of-thought reasoning (GenGround, FLARE) or `/no_think` for fast direct output (enrichment, question generation). This is uniquely valuable -- you get two models in one.
2. **Native structured output:** Ollama's structured output API works natively with Qwen3. Define a Pydantic schema, pass it as `format` parameter, and get reliable JSON [11].
3. **Native tool calling:** Qwen-Agent integration handles tool-calling templates internally, reducing prompt engineering overhead.
4. **Multilingual:** Critical for Indian legal documents that mix English, Hindi, and regional languages within the same judgment.
5. **Apache 2.0 license:** Full commercial use permitted.

**Weaknesses:**
- Intelligence Index (51.6% for Qwen3 family) is lower than GPT-OSS 20B (52.1%) on some aggregate benchmarks
- Thinking mode is verbose -- generates many reasoning tokens before the answer, increasing latency by 2-3x
- At 32K context, speed drops to ~9.6 tok/s as VRAM approaches the 16GB ceiling

**Verdict:** The best all-rounder for legal RAG. Handles all 7 tasks competently, with strong structured output and instruction following. The thinking/non-thinking toggle lets you trade speed for quality per-task.

### 5.2 Phi-4-Reasoning 14B -- BEST REASONING QUALITY

**Ollama tag:** `phi4-reasoning:14b`

| Attribute | Value |
|-----------|-------|
| Parameters | 14B |
| Architecture | Dense Transformer (Microsoft) |
| VRAM (Q4_K_M, 8K ctx) | ~11 GB |
| Speed (Q4_K_M) | ~45-55 tok/s |
| Context Window | 16K |
| Languages | Primarily English |
| License | MIT |

**Benchmarks:**
- MMLU: 84.8%
- GPQA: 56.1% (base), significantly higher for Reasoning variant
- IFBench: **0.834** (Reasoning), **0.849** (Reasoning Plus) -- **highest in class** [12]
- Outperforms DeepSeek-R1-Distill-Llama 70B on multiple reasoning benchmarks despite being 5x smaller

**Key Strengths for Legal RAG:**
1. **Instruction following:** Highest IFBench score of any model in this VRAM tier. Critical for tasks like claim extraction and confidence scoring where the model must follow a precise output schema.
2. **Reasoning depth:** Approaches full DeepSeek-R1 performance on logical reasoning. Ideal for GenGround (extracting individual claims from legal text and verifying each against evidence).
3. **Safety and predictability:** Trained specifically for compliant, predictable outputs -- important for legal applications where hallucination carries real risk.
4. **MIT license:** Full commercial use.

**Weaknesses:**
- 16K context window (vs. 32K+ for Qwen3, 128K for Llama 3.1) -- limits use for long legal documents
- Weaker multilingual support (primarily English-focused)
- No native tool calling in the Reasoning variant
- Structured JSON must be enforced via prompt engineering, not native schema

**Verdict:** Second-best pick. Exceptional for high-stakes reasoning tasks (GenGround, FLARE). Use as a specialist model alongside Qwen3.

### 5.3 GPT-OSS 20B -- FASTEST INFERENCE

**Ollama tag:** `gpt-oss:20b`

| Attribute | Value |
|-----------|-------|
| Parameters | 20B (MoE -- not all params active) |
| Architecture | Mixture of Experts (OpenAI) |
| VRAM (Q4_K_M, 8K ctx) | 12-14 GB |
| Speed (Q4_K_M) | **~140 tok/s** [13] |
| Context Window | Up to 120K |
| Languages | Multilingual |
| License | Apache 2.0 |

**Benchmarks:**
- MMLU-Pro: 74.8%
- LiveCodeBench: 77.7%
- GPQA: 68.8%
- Matches or exceeds OpenAI o3-mini on standard evaluations

**Key Strengths for Legal RAG:**
1. **Blazing fast:** 140 tok/s is 2-3x faster than the 14B dense models. For bulk enrichment of thousands of chunks, this speed advantage is significant.
2. **MoE architecture:** Only a subset of parameters is active per token, giving good quality-per-FLOP.
3. **Native function calling and reasoning effort control:** Can tune reasoning depth (low/medium/high) per request.
4. **120K context window:** Longest of any model in this tier.

**Weaknesses:**
- **CRITICAL: Structured output is broken in Ollama** as of February 2026. GitHub issue #11691 documents that the Harmony response format is incompatible with Ollama's JSON schema enforcement. Workarounds exist but are unreliable [7].
- VRAM usage is tight at 12-14GB -- only ~2-4GB headroom for KV cache. Context beyond 8K may trigger CPU offloading.
- Despite being 20B params, MoE means active params per token are lower, so raw reasoning quality may trail dense 14B models on some tasks.
- Relatively new model -- less community testing for edge cases.

**Verdict:** Would be the top pick if not for the structured output bug. For tasks that do not require JSON schema (e.g., free-text enrichment summaries, HyDE generation), it is excellent. For structured extraction tasks (claim extraction, RAGAS eval), avoid until the bug is resolved.

### 5.4 DeepSeek-R1-Distill-Qwen 14B -- BEST CHAIN-OF-THOUGHT

**Ollama tag:** `deepseek-r1:14b`

| Attribute | Value |
|-----------|-------|
| Parameters | 14B (distilled from 671B) |
| Architecture | Dense Transformer (Qwen2.5 base) |
| VRAM (Q4_K_M) | ~11 GB |
| Speed | ~45-55 tok/s |
| Context Window | 64K |
| License | MIT |

**Key Strengths:**
- Trained via distillation from DeepSeek-R1 (671B) -- captures reasoning patterns of a much larger model
- Excellent at mathematical and logical reasoning
- 64K context window
- Strong at step-by-step analysis (ideal for temporal consistency checking in legal texts)

**Weaknesses:**
- Forced `<think>` token generation adds latency (every response starts with reasoning chain)
- No native tool calling or structured output support
- Less instruction-following polish than Qwen3 or Phi-4
- Reasoning chains can be very verbose, consuming many output tokens

**Verdict:** Good specialist for deep reasoning tasks, but the forced thinking overhead and lack of structured output make it less practical as a general-purpose model for a 7-task pipeline.

### 5.5 Llama 3.1 8B Instruct -- BEST VALUE

**Ollama tag:** `llama3.1:8b`

| Attribute | Value |
|-----------|-------|
| Parameters | 8B |
| VRAM (Q4_K_M) | 6.2 GB |
| VRAM (Q5_K_M) | 7.3 GB |
| VRAM (Q8_0) | 8.5 GB |
| Speed (Q4_K_M) | ~70-89 tok/s |
| Context Window | 128K |
| License | Llama 3.1 Community License |

**Key Strengths:**
- Leaves ~10GB VRAM headroom -- could run alongside a second model or use very long contexts
- 128K context window
- Native tool calling and structured output
- Massive community support, tutorials, and fine-tuned variants
- Good multilingual support (trained on 15T+ tokens)

**Weaknesses:**
- Significantly weaker reasoning than 14B models (MMLU ~68% vs ~77-84% for 14B class)
- Legal analytical reasoning may be noticeably worse
- More prone to hallucination in knowledge-intensive legal tasks

**Verdict:** Excellent as a fast secondary model for low-complexity tasks (enrichment summaries, simple question generation). Not recommended as the sole model for high-stakes tasks like GenGround or answer generation.

### 5.6 Mistral Nemo 12B -- STRONG TOOL CALLING

**Ollama tag:** `mistral-nemo:12b`

| Attribute | Value |
|-----------|-------|
| Parameters | 12B |
| VRAM (Q4_K_M) | ~9 GB |
| Speed | ~50-65 tok/s |
| Context Window | 128K |
| License | Apache 2.0 |

**Key Strengths:**
- Built by Mistral AI + NVIDIA collaboration
- Trained with quantization awareness (FP8 runs without quality loss)
- Strong tool calling support
- 128K context window
- Good at structured output (Mistral models are noted for reliable JSON formatting [14])

**Weaknesses:**
- Older model (July 2024) -- superseded by Qwen3 and Phi-4 on most benchmarks
- Weaker reasoning than the 14B class
- No thinking mode / chain-of-thought support

**Verdict:** Solid mid-range option. The Mistral family's reliable JSON output is a genuine advantage for structured extraction tasks. Consider as a fallback if Qwen3 has issues with your specific prompts.

### 5.7 Gemma 3 12B -- MULTIMODAL OPTION

**Ollama tag:** `gemma3:12b`

| Attribute | Value |
|-----------|-------|
| Parameters | 12B |
| VRAM (Q4_K_M) | ~10 GB (but has reported VRAM issues) |
| Context Window | 128K |
| License | Gemma License (permissive) |

**Strengths:** Multimodal (text + images), 128K context, 140+ languages.

**Weaknesses:**
- **Reported VRAM leak:** Multiple Ollama issues (#9730, #10341) document that Gemma 3 12B Q4_K_M can consume 22-23GB VRAM with default settings or exhibit growing RAM usage across runs [9].
- No native tool calling in the base model (requires custom fork `orieg/gemma3-tools`)
- Less polished instruction following than Qwen3 or Phi-4

**Verdict:** Avoid for production use until VRAM issues are resolved. The multimodal capability is unnecessary for your text-only legal pipeline.

### 5.8 Qwen 2.5 14B Instruct -- PREDECESSOR, STILL STRONG

**Ollama tag:** `qwen2.5:14b-instruct`

| Attribute | Value |
|-----------|-------|
| Parameters | 14B |
| VRAM (Q4_K_M) | ~10.7 GB |
| Context Window | 128K |
| License | Apache 2.0 |

**Strengths:** 128K context (vs Qwen3's 32K), stable and well-tested, strong structured output, 29+ languages.

**Weaknesses:** Superseded by Qwen3 on all benchmarks. No thinking mode. Slightly weaker reasoning.

**Verdict:** A safe fallback if Qwen3 has compatibility issues. The 128K context is useful for answer generation over long retrieved contexts.

---

## 6. Task-to-Model Mapping

For each of the 7 Legal RAG tasks, here is the recommended model and configuration:

### Task 1: Text Enrichment (Context Summaries for Chunks)

**Requirement:** Generate a brief context paragraph situating each chunk within its parent document. Moderate reasoning, high throughput needed (thousands of chunks).

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | Qwen3 14B (`/no_think` mode) |
| **Alternative** | GPT-OSS 20B (if JSON not required) or Llama 3.1 8B |
| **Quantization** | Q4_K_M |
| **Context Needed** | 4K-8K (chunk + parent doc summary) |
| **Structured Output** | Not strictly needed (free text) |
| **Speed Priority** | HIGH (batch processing) |

**Rationale:** This is a high-throughput, moderate-complexity task. Qwen3 in non-thinking mode provides good quality at ~50-62 tok/s. For maximum speed, GPT-OSS 20B at ~140 tok/s is 2.5x faster, and since the output is free text (not structured JSON), the structured output bug is irrelevant.

### Task 2: Question Generation (QuIM-RAG)

**Requirement:** Generate 3-5 anticipatory questions per chunk that a lawyer might ask and that this chunk could answer.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | Qwen3 14B (`/no_think` mode) |
| **Alternative** | Phi-4 14B |
| **Quantization** | Q4_K_M |
| **Context Needed** | 4K-8K |
| **Structured Output** | Yes (JSON array of questions) |
| **Speed Priority** | MEDIUM (batch, but less volume than enrichment) |

**Rationale:** Requires understanding legal concepts well enough to anticipate relevant questions. The 14B models handle this significantly better than 8B. Qwen3's native JSON output ensures consistent question formatting.

### Task 3: HyDE (Hypothetical Document Generation)

**Requirement:** Given a user query, generate a hypothetical ideal answer document that would satisfy the query, for use as a search vector.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | Qwen3 14B (`/think` mode for complex queries, `/no_think` for simple) |
| **Alternative** | GPT-OSS 20B |
| **Quantization** | Q4_K_M |
| **Context Needed** | 2K-4K (query + instructions) |
| **Structured Output** | Not needed (free text) |
| **Speed Priority** | HIGH (latency-sensitive, in query path) |

**Rationale:** HyDE is on the critical query path -- every user query triggers it. Latency matters. For SIMPLE/STANDARD queries, use non-thinking mode. For COMPLEX/ANALYTICAL, thinking mode produces better hypothetical documents. GPT-OSS 20B is a strong alternative due to its speed.

### Task 4: FLARE (Retrieval Confidence Assessment + Follow-up Queries)

**Requirement:** Assess whether retrieved chunks sufficiently answer the query. If not, generate follow-up queries to fill knowledge gaps.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | **Phi-4-Reasoning 14B** |
| **Alternative** | Qwen3 14B (`/think` mode) |
| **Quantization** | Q4_K_M |
| **Context Needed** | 8K-16K (query + multiple retrieved chunks) |
| **Structured Output** | Yes (confidence score + follow-up queries as JSON) |
| **Speed Priority** | LOW (only triggers for ANALYTICAL queries) |

**Rationale:** This is a high-reasoning task that benefits from Phi-4-Reasoning's superior analytical capability and instruction following. The model needs to assess sufficiency, identify gaps, and generate targeted follow-up queries -- exactly the kind of instruction-following + reasoning task where Phi-4 excels (IFBench 0.834-0.849).

### Task 5: GenGround (Claim Extraction + Verification)

**Requirement:** Extract individual factual claims from a generated answer, then verify each claim against retrieved evidence chunks.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | **Phi-4-Reasoning 14B** |
| **Alternative** | Qwen3 14B (`/think` mode) |
| **Quantization** | Q4_K_M |
| **Context Needed** | 8K-16K (answer + evidence chunks) |
| **Structured Output** | Yes (claims array with verification status) |
| **Speed Priority** | LOW (post-generation, not latency-critical) |

**Rationale:** The most analytically demanding task in the pipeline. The model must decompose legal prose into discrete claims, then cross-reference each against evidence. Phi-4-Reasoning's top-tier reasoning and instruction following make it the best choice. Qwen3 in thinking mode is a close second.

### Task 6: Answer Generation (Legal Answer Synthesis)

**Requirement:** Synthesize a comprehensive legal answer from retrieved chunks, with proper citations to source statutes/judgments.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | Qwen3 14B (`/think` mode) |
| **Alternative** | Qwen 2.5 14B (if 128K context needed) |
| **Quantization** | Q4_K_M |
| **Context Needed** | 8K-32K (query + multiple retrieved chunks + system prompt) |
| **Structured Output** | Partial (structured citations within free text) |
| **Speed Priority** | MEDIUM (user-facing, but acceptable to wait 5-10 seconds) |

**Rationale:** Answer generation is the user-facing output. Quality is paramount. Qwen3's thinking mode provides visible chain-of-thought that can double as an "analysis" section. Its multilingual support handles mixed English/Hindi legal text. If the retrieval context is very long (20K+ tokens), Qwen 2.5 14B's 128K window provides more headroom.

### Task 7: RAGAS Evaluation Metrics

**Requirement:** Compute faithfulness, answer relevance, context precision/recall scores. Requires the model to judge quality of other model outputs.

| Attribute | Recommendation |
|-----------|---------------|
| **Primary Model** | Qwen3 14B (`/no_think` mode) |
| **Alternative** | Phi-4-Reasoning 14B |
| **Quantization** | Q4_K_M |
| **Context Needed** | 8K-16K (question + answer + context) |
| **Structured Output** | Yes (scores as JSON) |
| **Speed Priority** | LOW (batch evaluation, offline) |

**Rationale:** RAGAS evaluation requires the model to act as a judge -- assessing faithfulness and relevance. Qwen3 handles this well with its structured output capabilities. Phi-4-Reasoning is even better at the analytical judgment aspect but slower. Since evaluation is an offline batch process, either works.

### Summary Matrix

| Task | Primary Model | Mode | VRAM | Speed Priority |
|------|--------------|------|------|---------------|
| 1. Text Enrichment | Qwen3 14B | /no_think | 10.7 GB | HIGH |
| 2. Question Gen (QuIM-RAG) | Qwen3 14B | /no_think | 10.7 GB | MEDIUM |
| 3. HyDE | Qwen3 14B | Adaptive | 10.7 GB | HIGH |
| 4. FLARE | Phi-4-Reasoning 14B | Reasoning | 11 GB | LOW |
| 5. GenGround | Phi-4-Reasoning 14B | Reasoning | 11 GB | LOW |
| 6. Answer Generation | Qwen3 14B | /think | 10.7-13.6 GB | MEDIUM |
| 7. RAGAS Evaluation | Qwen3 14B | /no_think | 10.7 GB | LOW |

---

## 7. Legal-Specific Models

### Available on Ollama

1. **initium/law_model** -- A community-uploaded legal model on Ollama. Very limited documentation and unknown training data. Not recommended for production use.

2. **nawalkhan/legal-llm** -- Another community legal model. Insufficient benchmarking data available.

### Not on Ollama (Research Models)

1. **SaulLM-7B** -- Trained on 30B+ tokens of English legal text. Based on Mistral-7B. Would fit in 16GB at Q4_K_M (~5GB). However, it is a continued pre-training model, not instruction-tuned -- it excels at legal text completion but not at following instructions or generating structured output.

2. **SaulLM-54B / SaulLM-141B** -- Larger variants based on Mixtral. The 54B would not fit in 16GB. The 141B is out of the question.

3. **InternLM-Law** -- Chinese legal LLM. Not relevant for Indian law.

4. **LawLLM** -- Academic research model for US legal system. Not available on Ollama.

### Verdict on Legal-Specific Models

**Do not use legal-specific models.** The current generation of general-purpose 14B models (Qwen3, Phi-4) significantly outperform the legal-specific models on all relevant benchmarks. The legal-specific models are either:
- Too small (SaulLM-7B) and not instruction-tuned
- Too large (SaulLM-54B/141B) for 16GB VRAM
- Trained on non-Indian legal corpora (US/EU/Chinese law)
- Community uploads with no quality guarantees

Your pipeline already has domain-specific handling (Indian statute structure, temporal IPC/BNS transitions, citation extraction regex) in the code layer. The LLM just needs strong reasoning + instruction following, which the general-purpose models provide.

---

## 8. Multi-Model Strategy

### Recommended Configuration

Since Ollama can only run one model at a time per GPU (loading a new model unloads the previous one), and your 7 tasks are executed sequentially in the pipeline, use a **two-model rotation**:

```
Pipeline Stage        Model Loaded          VRAM Used    Approx Time
-----------------     ------------------    ---------    -----------
Text Enrichment       Qwen3 14B (Q4_K_M)   10.7 GB      Batch (bulk)
Question Gen          Qwen3 14B (Q4_K_M)   10.7 GB      Batch (bulk)
                      [no model swap needed]
--- Query Time ---
HyDE                  Qwen3 14B (Q4_K_M)   10.7 GB      Per-query
Answer Generation     Qwen3 14B (Q4_K_M)   10.7-13.6 GB Per-query
                      [no model swap needed]
FLARE (if triggered)  [swap] Phi-4-Reasoning 11 GB       Per-query
GenGround             Phi-4-Reasoning 14B    11 GB       Per-query
                      [swap back]
RAGAS Evaluation      [swap] Qwen3 14B      10.7 GB      Batch (offline)
```

**Model swap overhead:** Loading a new model in Ollama takes ~3-8 seconds on an NVMe SSD. This is acceptable since swaps only happen when transitioning between pipeline stages, not per-query.

### Alternative: Single-Model Simplicity

If you prefer to avoid model swapping entirely, **Qwen3 14B alone handles all 7 tasks** at 85-95% of the optimal quality. The only tasks where Phi-4-Reasoning provides a meaningful uplift are FLARE and GenGround, and even there, Qwen3 in `/think` mode is strong.

### Parallel Model Loading (Advanced)

If you need HyDE + answer generation + FLARE in the query path without swapping:
- Load **Llama 3.1 8B** (Q4_K_M, 6.2 GB) for HyDE (fast, simple task)
- Keep Qwen3 14B or Phi-4-Reasoning for the heavy tasks
- Total VRAM: 6.2 + 10.7 = 16.9 GB -- slightly over budget, but with `OLLAMA_KV_CACHE_TYPE=q8_0` and short contexts, it might fit

This is risky on 16GB and not recommended unless tested.

---

## 9. Optimization Tips for RTX 5060 Ti

### 1. Enable KV Cache Quantization

```bash
# Add to your environment (e.g., .env file or system env)
OLLAMA_KV_CACHE_TYPE=q8_0
```

This halves KV cache memory with negligible quality loss. Effectively gives you 1-2GB more headroom [6].

### 2. Set Appropriate Context Length

```bash
# In your Ollama Modelfile or API call
/set parameter num_ctx 8192   # For enrichment, question gen
/set parameter num_ctx 16384  # For FLARE, GenGround
/set parameter num_ctx 32768  # Only for answer generation with long contexts
```

Do not use the model's maximum context length by default. KV cache grows linearly -- 32K context for Qwen3 14B uses ~3GB more VRAM than 8K [3].

### 3. Use Flash Attention

Ollama enables flash attention by default on supported architectures. Verify with:
```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

This reduces memory usage for attention computation, especially at longer contexts.

### 4. Keep Models Warm

```bash
# Prevent Ollama from unloading the model after inactivity
OLLAMA_KEEP_ALIVE=-1   # Keep loaded indefinitely
# Or set a long timeout
OLLAMA_KEEP_ALIVE=60m   # Keep for 60 minutes
```

Model loading takes 3-8 seconds. For pipeline batch processing, keep the model loaded.

### 5. Structured Output Best Practices

When using Ollama's structured output with Qwen3:

```python
import ollama

response = ollama.chat(
    model='qwen3:14b',
    messages=[...],
    format={  # JSON schema
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence": {"type": "number"}
        },
        "required": ["claims", "confidence"]
    },
    options={
        "temperature": 0,  # Maximize schema adherence
        "num_ctx": 8192
    }
)
```

Add "Return as JSON" in your system prompt even when using the `format` parameter -- this helps the model understand the expectation [14].

### 6. Batch Processing Optimization

For bulk enrichment (thousands of chunks), process sequentially with keep-alive enabled rather than trying to parallelize. The GPU can only run one inference at a time, and the overhead of request queuing in Ollama is minimal.

---

## 10. Key Findings and Insights

1. **The 14B class is the sweet spot for 16GB VRAM.** The 8B models leave too much VRAM unused and sacrifice quality. The 20B+ models risk overflow. 14B at Q4_K_M uses ~10.7-11GB, leaving comfortable headroom for KV cache at 8K-16K context.

2. **Qwen3 14B's dual mode (/think vs /no_think) is uniquely valuable for a multi-task pipeline.** No other model in this VRAM class offers this flexibility. It lets you dynamically trade speed for quality depending on the task.

3. **GPT-OSS 20B's structured output bug is a dealbreaker for legal RAG.** Despite excellent speed and benchmarks, the inability to reliably produce JSON schemas via Ollama's API makes it unsuitable for 5 of your 7 tasks. Monitor GitHub issue #11691 for resolution.

4. **Legal-specific models are not worth using.** General-purpose 14B models outperform SaulLM-7B on all relevant tasks. Your domain expertise is better encoded in prompts and pipeline code than in a smaller, less capable legal-specific model.

5. **KV cache quantization (q8_0) is free performance.** Halves KV cache memory with no measurable quality loss. Every Ollama deployment on 16GB VRAM should enable this.

6. **Context length is the hidden VRAM killer.** A 14B model at 8K context uses 10.7GB; at 32K it uses 13.6GB. Set `num_ctx` per-task, not globally.

7. **The RTX 5060 Ti's native FP8 support is underutilized by Ollama.** Current Ollama quantizations are GGUF-based (integer quantization). As FP8 model formats mature in the llama.cpp ecosystem, the 5060 Ti will see significant additional performance gains.

8. **Model swapping overhead (3-8 seconds) is acceptable for pipeline use.** Only swap when changing pipeline stages, not per-query. The two-model strategy (Qwen3 primary + Phi-4-Reasoning specialist) provides optimal quality without meaningful latency penalty.

---

## 11. Risks, Limitations, and Open Questions

### Risks

1. **VRAM estimates are approximate.** Actual usage varies by ~5-10% depending on Ollama version, CUDA driver version, and specific quantization variant (Q4_K_M vs Q4_K_S vs Q4_0). Always test with `nvidia-smi` monitoring before committing to a configuration.

2. **Ollama on Windows has known quirks.** Some users report higher VRAM overhead on Windows vs Linux due to WDDM driver model. If VRAM is tighter than expected, consider WSL2 with Ollama.

3. **Quality degradation is task-dependent.** Q4_K_M's 97.1% quality retention is an average. For structured JSON extraction and legal citation tasks, the gap may be larger or smaller. Benchmark with your actual prompts.

4. **Gemma 3 VRAM issues are unresolved.** Multiple Ollama GitHub issues report VRAM leaks and unexpectedly high memory usage for Gemma 3 models. Avoid until patched.

### Open Questions

1. **Will Ollama fix GPT-OSS structured output?** If resolved, GPT-OSS 20B becomes the top recommendation for speed-critical tasks. The PR (#14288) is in progress.

2. **When will Qwen3 14B get a dedicated Q5_K_M Ollama tag?** The Q5_K_M quantization at ~12GB VRAM would be an attractive middle ground with 98.2% quality retention.

3. **How does the RTX 5060 Ti's GDDR7 bandwidth affect inference vs. RTX 4060 Ti GDDR6?** Higher bandwidth should improve token throughput, but real-world Ollama benchmarks on the 5060 Ti are still sparse.

4. **Is there a quality difference between Qwen3's /think mode and Phi-4-Reasoning for legal tasks specifically?** No published benchmark compares them on legal reasoning. You may need to run your own evaluation using the RAGAS pipeline.

---

## 12. Final Ranked Recommendation

### Rank 1: Qwen3 14B (Q4_K_M) -- PRIMARY MODEL

```bash
ollama pull qwen3:14b
# VRAM: 10.7 GB (8K ctx) / 13.6 GB (32K ctx)
# Speed: ~50-62 tok/s
# Use for: Tasks 1, 2, 3, 6, 7 (5 of 7 tasks)
```

**Why #1:** Best overall balance of reasoning quality, instruction following, structured output, multilingual support, and VRAM efficiency. The /think vs /no_think toggle is a killer feature for a multi-task pipeline. Apache 2.0 license.

### Rank 2: Phi-4-Reasoning 14B (Q4_K_M) -- SPECIALIST MODEL

```bash
ollama pull phi4-reasoning:14b-q4_K_M
# VRAM: ~11 GB
# Speed: ~45-55 tok/s
# Use for: Tasks 4 (FLARE) and 5 (GenGround)
```

**Why #2:** Highest instruction-following score in class (IFBench 0.849). Best pure reasoning quality for 14B. Ideal for the two tasks that demand the deepest analytical reasoning. MIT license.

### Rank 3: GPT-OSS 20B (Q4_K_M) -- SPEED MODEL (conditional)

```bash
ollama pull gpt-oss:20b
# VRAM: 12-14 GB
# Speed: ~140 tok/s
# Use for: Task 1 (bulk enrichment) IF structured JSON not required
# WAIT for structured output fix before broader use
```

**Why #3:** 2-3x faster than the 14B models. Excellent for batch processing thousands of chunks. But the structured output bug limits its utility to free-text tasks only. Revisit when issue #11691 is resolved.

### Rank 4: Llama 3.1 8B Instruct (Q5_K_M) -- LIGHTWEIGHT FALLBACK

```bash
ollama pull llama3.1:8b
# VRAM: 7.3 GB (Q5_K_M) / 6.2 GB (Q4_K_M)
# Speed: ~70-89 tok/s
# Use for: Fast HyDE generation, simple enrichment, testing/development
```

**Why #4:** Uses half the VRAM of 14B models. Good for development/testing and as a fast secondary model. The Q5_K_M quantization (7.3 GB) provides better quality than Q4_K_M at still-comfortable VRAM levels. 128K context window is generous.

### Rank 5: Mistral Nemo 12B (Q4_K_M) -- STRUCTURED OUTPUT FALLBACK

```bash
ollama pull mistral-nemo:12b
# VRAM: ~9 GB
# Speed: ~50-65 tok/s
# Use for: Backup if Qwen3 has issues with specific JSON schemas
```

**Why #5:** Mistral models are noted for particularly reliable JSON formatting. If Qwen3's structured output has edge cases with your specific schemas, Mistral Nemo is a solid alternative. 128K context. Apache 2.0. Tool calling support.

### Rank 6: Qwen 2.5 14B Instruct (Q4_K_M) -- LONG CONTEXT FALLBACK

```bash
ollama pull qwen2.5:14b-instruct
# VRAM: ~10.7 GB
# Speed: ~50-60 tok/s
# Use for: Answer generation when context exceeds 32K tokens
```

**Why #6:** 128K context window (vs Qwen3's 32K). If your retrieval pipeline returns very long contexts that exceed 32K tokens, this model handles them natively. Otherwise, Qwen3 14B is strictly better.

### Quick Reference: What to Install

For a production Legal RAG pipeline, install these two models:

```bash
# Primary (all-purpose)
ollama pull qwen3:14b

# Specialist (high-reasoning tasks)
ollama pull phi4-reasoning:14b-q4_K_M

# Set environment
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=60m
```

Total disk: ~18-20 GB for both models (only one loaded in VRAM at a time).

---

## 13. Sources and References

1. [RTX 5060 Ti LLM Benchmarks - Hardware Corner](https://www.hardware-corner.net/guides/dual-rtx-5060-ti-16gb-vs-rtx-3090-llm/) -- Dual 5060 Ti vs 3090 comparison, FP8/MXFP4 performance data
2. [Best Local LLMs for RTX 50 Series - APXML](https://apxml.com/posts/best-local-llms-for-every-nvidia-rtx-50-series-gpu) -- Model recommendations per GPU tier
3. [Ollama VRAM Requirements Complete 2026 Guide - LocalLLM.in](https://localllm.in/blog/ollama-vram-requirements-for-local-llms) -- VRAM formula, quantization tables, KV cache calculations
4. [AI Model Quantization 2025 Guide - Local AI Zone](https://local-ai-zone.github.io/guides/what-is-ai-quantization-q4-k-m-q8-gguf-guide-2025.html) -- Quantization quality retention percentages
5. [Best Local LLMs for 16GB VRAM - LocalLLM.in](https://localllm.in/blog/best-local-llms-16gb-vram) -- GPT-OSS 20B, Apriel, Qwen3 14B benchmark comparison with VRAM data
6. [KV Cache Quantization in Ollama - Mitja Martini](https://mitjamartini.com/posts/ollama-kv-cache-quantization/) -- q8_0/q4_0 KV cache settings and memory savings
7. [GPT-OSS Structured Output Bug - GitHub Issue #11691](https://github.com/ollama/ollama/issues/11691) -- Structured output incompatibility with Harmony response format
8. [Gemma 3 Tools for Ollama - orieg](https://ollama.com/orieg/gemma3-tools) -- Custom Gemma 3 models with tool calling support
9. [Gemma 3 12B VRAM Issues - GitHub Issue #9730](https://github.com/ollama/ollama/issues/9730) -- Reported 22-23GB VRAM usage for 12B model
10. [Comparing LLMs on 16GB VRAM GPU - Rost Glukhov](https://www.glukhov.org/post/2026/01/choosing-best-llm-for-ollama-on-16gb-vram-gpu/) -- Real-world benchmark data: GPT-OSS 20B (140 tok/s), Qwen3 14B (62 tok/s), Mistral Small 3.2 24B (18.5 tok/s with CPU offloading)
11. [Structured Output with Ollama and Qwen3 - Rost Glukhov](https://medium.com/@rosgluk/constraining-llms-with-structured-output-ollama-qwen3-python-or-go-2f56ff41d720) -- Practical guide to JSON schema enforcement with Qwen3
12. [Phi-4-Reasoning on HuggingFace](https://huggingface.co/microsoft/Phi-4-reasoning) -- Official benchmark scores including IFBench
13. [GPT-OSS 20B on Ollama](https://ollama.com/library/gpt-oss:20b) -- Official model page
14. [Ollama Structured Outputs Documentation](https://docs.ollama.com/capabilities/structured-outputs) -- Official structured output API guide
15. [Best Open Source LLMs for Legal Industry 2026 - SiliconFlow](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-legal-industry) -- Legal LLM comparison
16. [OpenAI GPT-OSS Benchmarks - Clarifai](https://www.clarifai.com/blog/openai-gpt-oss-benchmarks-how-it-compares-to-glm-4.5-qwen3-deepseek-and-kimi-k2) -- Cross-model benchmark comparison
17. [Phi-4-Reasoning Ollama Page](https://ollama.com/library/phi4-reasoning:14b-q4_K_M) -- Official Ollama model page with quantization options
18. [Qwen3 on Ollama](https://ollama.com/library/qwen3) -- Official Qwen3 model page
19. [Ollama Library](https://ollama.com/library) -- Complete model catalog
20. [SaulLM-7B Paper - ArXiv](https://arxiv.org/abs/2403.03883) -- Legal-specific LLM trained on 30B+ legal tokens
21. [RTX 5060 Ollama Benchmarks - DatabaseMart](https://www.databasemart.com/blog/ollama-gpu-benchmark-rtx5060) -- RTX 5060 performance data
22. [Best Ollama Models 2025 - Collabnix](https://collabnix.com/best-ollama-models-in-2025-complete-performance-comparison/) -- Performance comparison guide
23. [VRAM Calculator - APXML](https://apxml.com/tools/vram-calculator) -- Interactive VRAM estimator
24. [Best Open Source Small Language Models 2026 - BentoML](https://www.bentoml.com/blog/the-best-open-source-small-language-models) -- SLM comparison
25. [Bringing KV Context Quantisation to Ollama - smcleod.net](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/) -- Technical deep-dive on KV cache quantization implementation
