# NVIDIA NIM API — Free Model Catalog for Developers

**Research Date:** 2026-02-28
**Scope:** Complete catalog of LLM models available via NVIDIA NIM free API tier (build.nvidia.com), with focus on models suitable for a Legal RAG pipeline requiring text generation, JSON output, function calling, and analytical reasoning.
**Confidence Level:** High (sourced from official NVIDIA docs, developer forums, Mastra catalog, and liteLLM integration docs)

---

## Executive Summary

NVIDIA NIM (NVIDIA Inference Microservices) provides free API access to **70+ AI models** through build.nvidia.com. The free tier is rate-limited (approximately **40 requests/minute**) rather than credit-based — NVIDIA retired the old credit system in late 2025. No payment information is required; only phone verification. All endpoints are **OpenAI-compatible** (`/v1/chat/completions`) and work with both the OpenAI Python SDK (sync and async) and standard HTTP POST requests.

For a Legal RAG pipeline, the most relevant models are:
1. **Nemotron Ultra 253B** — Best reasoning quality, function calling, 131K context
2. **Nemotron Super 49B v1.5** — Best cost/quality balance for agentic tasks, function calling, 128K context
3. **Llama 3.3 70B Instruct** — Reliable workhorse, function calling, 128K context
4. **Mistral Large 3 675B** — Strongest Mistral, function calling, 262K context
5. **GPT-OSS 120B** — OpenAI's open MoE model, function calling, 128K context

**Critical limitation:** DeepSeek models on NIM do NOT support structured function calling (tool_calls field). Tool calls appear in response body text instead. Avoid DeepSeek for agentic/tool-calling tasks on NIM.

**For structured JSON output:** Use the `guided_json` parameter in the `nvext` body extension (NOT `response_format`), which constrains output to a JSON schema via XGrammar.

---

## Table of Contents

1. [API Access and Free Tier Details](#1-api-access-and-free-tier-details)
2. [API Format and Integration](#2-api-format-and-integration)
3. [Tier 1: Frontier-Class Models](#3-tier-1-frontier-class-models)
4. [Tier 2: Strong General-Purpose Models](#4-tier-2-strong-general-purpose-models)
5. [Tier 3: Efficient / Specialized Models](#5-tier-3-efficient--specialized-models)
6. [Tier 4: Compact / Edge Models](#6-tier-4-compact--edge-models)
7. [Embedding and Reranking Models](#7-embedding-and-reranking-models)
8. [Function Calling Support Matrix](#8-function-calling-support-matrix)
9. [Structured JSON Output](#9-structured-json-output)
10. [Recommendations for Legal RAG Pipeline](#10-recommendations-for-legal-rag-pipeline)
11. [Key Findings](#key-findings)
12. [Sources and References](#sources-and-references)

---

## 1. API Access and Free Tier Details

### How to Get Access
1. Create a free account at [build.nvidia.com](https://build.nvidia.com/)
2. Verify your phone number
3. Navigate to API Keys page and generate a key
4. No credit card required

### Free Tier Limits

| Parameter | Value |
|-----------|-------|
| **Pricing model** | Rate-limited (NOT credit-based; credits retired late 2025) |
| **Rate limit** | ~40 requests/minute (varies per model, not published per-model) |
| **Token limits** | Not published; models tend to be "context window limited" |
| **Daily/monthly caps** | No published hard caps; rate limit is the binding constraint |
| **Expiry** | No expiry — free forever for "prototyping, research, development, testing, learning" |
| **Production use** | Requires NVIDIA AI Enterprise license |

**Important notes:**
- NVIDIA explicitly stated they have "no plan to publish specific model limits, since the limits only apply to the APIs which are for trial experiences" (NVIDIA Developer Forums).
- The 40 RPM is the commonly observed rate across models. Some models may have slightly different limits visible in the top-right of build.nvidia.com when logged in.
- Context windows may be constrained below the model's theoretical maximum on the free tier.

### What "Free" Means
The NVIDIA API Catalog is a **trial experience**. It is intended for prototyping, not production. For production deployments, NVIDIA directs users to:
- NVIDIA AI Enterprise (self-hosted NIM containers)
- Third-party hosted endpoints (Together.ai, Baseten, Fireworks)
- DGX Cloud Serverless Inference

---

## 2. API Format and Integration

### Base URL
```
https://integrate.api.nvidia.com/v1
```

### OpenAI-Compatible Endpoint
```
POST https://integrate.api.nvidia.com/v1/chat/completions
Authorization: Bearer $NVIDIA_API_KEY
```

### Python (Sync)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxx"  # Your NVIDIA API key
)

response = client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    messages=[{"role": "user", "content": "Summarize Section 302 of BNS"}],
    temperature=0.2,
)
```

### Python (Async)
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="nvapi-xxx"
)

response = await client.chat.completions.create(
    model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
    messages=[{"role": "user", "content": "Summarize Section 302 of BNS"}],
    stream=True,
)
async for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Streaming
Set `stream=True` in the request. Works identically to OpenAI streaming.

### LiteLLM Integration
```python
import litellm
response = litellm.completion(
    model="nvidia_nim/meta/llama-3.3-70b-instruct",
    messages=[{"role": "user", "content": "..."}]
)
```

---

## 3. Tier 1: Frontier-Class Models

These models compete with Claude Opus / GPT-4o / Gemini Pro in reasoning quality.

### Nemotron Ultra 253B v1
| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/llama-3.1-nemotron-ultra-253b-v1` |
| **Parameters** | 253B total (MoE, reduced active via NAS) |
| **Context Window** | 131K tokens |
| **Function Calling** | YES (auto-enabled, requires `detailed_thinking: off`) |
| **JSON Output** | YES (via `guided_json` in `nvext`) |
| **Quality Comparison** | Competes with DeepSeek-R1; GPQA 76.0%. Meets/beats top open reasoning models |
| **Best For** | Complex legal reasoning, analytical queries, RAG synthesis |
| **Reasoning Mode** | Toggle on/off via system prompt |

### Mistral Large 3 675B
| Field | Value |
|-------|-------|
| **Model ID** | `mistralai/mistral-large-3-675b-instruct-2512` |
| **Parameters** | 675B total (673B language + 2.5B vision), 41B active (MoE) |
| **Context Window** | 262K tokens |
| **Function Calling** | YES (auto-enabled) |
| **JSON Output** | YES (via `guided_json`) |
| **Quality Comparison** | Comparable to GPT-4o class; strong instruction following |
| **Best For** | Long-context legal analysis, multi-document reasoning, 262K context |

### DeepSeek V3.2
| Field | Value |
|-------|-------|
| **Model ID** | `deepseek-ai/deepseek-v3.2` |
| **Parameters** | Large MoE (specifics vary) |
| **Context Window** | 164K tokens |
| **Function Calling** | **NO on NIM** (tool calls appear in response text, not structured) |
| **JSON Output** | Via prompt engineering only (no `guided_json` guarantee) |
| **Quality Comparison** | IMO gold medal; IOI gold medal; top reasoning |
| **Best For** | Pure text generation/reasoning where function calling is NOT needed |
| **Warning** | Do NOT use for agentic/tool-calling tasks on NIM |

### DeepSeek R1 / R1-0528
| Field | Value |
|-------|-------|
| **Model ID** | `deepseek-ai/deepseek-r1` / `deepseek-ai/deepseek-r1-0528` |
| **Parameters** | 671B (MoE) |
| **Context Window** | 128K tokens |
| **Function Calling** | **NO on NIM** |
| **Quality Comparison** | Top-tier reasoning; GPQA ~71%+ |
| **Best For** | Complex reasoning chains (non-agentic) |

### GPT-OSS 120B
| Field | Value |
|-------|-------|
| **Model ID** | `openai/gpt-oss-120b` |
| **Parameters** | 120B (MoE) |
| **Context Window** | 128K tokens |
| **Function Calling** | YES (auto-enabled) |
| **JSON Output** | YES (via `guided_json`) |
| **Quality Comparison** | Similar to OpenAI o3/o4 on open benchmarks; coding avg 8.3 |
| **Best For** | General-purpose reasoning, coding, tool use |
| **Reasoning** | Chain-of-thought with adjustable reasoning effort |

### Qwen3.5 397B-A17B
| Field | Value |
|-------|-------|
| **Model ID** | `qwen/qwen3.5-397b-a17b` |
| **Parameters** | 397B total, 17B active (MoE) |
| **Context Window** | Unknown (likely 128K+; hybrid MoE) |
| **Function Calling** | Likely YES (Qwen3 family supports it natively) |
| **Quality Comparison** | State-of-the-art multimodal; native vision-language |
| **Best For** | Multimodal RAG, vision+text workflows |
| **Added** | 2026-02-16 (very recent) |

### GLM-5
| Field | Value |
|-------|-------|
| **Model ID** | `z-ai/glm5` |
| **Parameters** | 744B total, 40B active (MoE with DSA) |
| **Context Window** | 203K tokens |
| **Function Calling** | Unknown (not in NIM function calling docs) |
| **Quality Comparison** | Best-in-class among open-source on reasoning, coding, agentic tasks |
| **Best For** | Long-context reasoning and agentic workflows |
| **Added** | 2026-02-17 (very recent) |

### Kimi K2.5
| Field | Value |
|-------|-------|
| **Model ID** | `moonshotai/kimi-k2.5` |
| **Parameters** | 1T total, 3.2% activation rate per token (MoE) |
| **Context Window** | 262K tokens |
| **Function Calling** | YES (per NVIDIA docs) |
| **Quality Comparison** | Strong multimodal (text + image + video) |
| **Best For** | Multimodal legal document analysis with images/tables |

---

## 4. Tier 2: Strong General-Purpose Models

These models are comparable to Claude Sonnet / GPT-4o-mini tier.

### Nemotron Super 49B v1.5 (RECOMMENDED)
| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/llama-3.3-nemotron-super-49b-v1.5` |
| **Parameters** | 49B (NAS-optimized from Llama-3.3-70B) |
| **Context Window** | 128K tokens |
| **Function Calling** | YES (auto-enabled, requires `detailed_thinking: off`) |
| **JSON Output** | YES (via `guided_json`) |
| **Quality Comparison** | MATH500 97.4%, AIME-2024 87.5%, IFEval strong. Fits on single H200 |
| **Best For** | Agentic RAG tasks, tool calling, instruction following |
| **Reasoning Mode** | Toggle on/off; "reasoning on" for complex, "off" for fast tool calls |

### Nemotron Super 49B v1
| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/llama-3.3-nemotron-super-49b-v1` |
| **Parameters** | 49B |
| **Context Window** | 128K tokens |
| **Function Calling** | YES |
| **Note** | v1.5 is strictly better; use v1.5 unless v1 is needed for compatibility |

### Llama 3.3 70B Instruct
| Field | Value |
|-------|-------|
| **Model ID** | `meta/llama-3.3-70b-instruct` |
| **Parameters** | 70B (dense) |
| **Context Window** | 128K tokens |
| **Function Calling** | YES (auto-enabled) |
| **JSON Output** | YES (via `guided_json`) |
| **Quality Comparison** | Strong general-purpose; reliable instruction following |
| **Best For** | Reliable text generation, context summaries, Q&A |

### Llama 3.1 405B Instruct
| Field | Value |
|-------|-------|
| **Model ID** | `meta/llama-3.1-405b-instruct` |
| **Parameters** | 405B (dense) |
| **Context Window** | 128K tokens |
| **Function Calling** | YES (auto-enabled) |
| **Quality Comparison** | Largest dense Llama; strong but older generation |
| **Best For** | Tasks needing maximum parameter count in dense architecture |

### Llama 4 Maverick 17B-128E
| Field | Value |
|-------|-------|
| **Model ID** | `meta/llama-4-maverick-17b-128e-instruct` |
| **Parameters** | 17B per expert, 128 experts (MoE) |
| **Context Window** | 128K tokens (supports up to 1M theoretically) |
| **Function Calling** | YES (Llama family auto-enabled) |
| **Quality Comparison** | Beats GPT-4o and Gemini 2.0 Flash on broad benchmarks |
| **Best For** | Multimodal tasks, high-throughput scenarios |

### Llama 4 Scout 17B-16E
| Field | Value |
|-------|-------|
| **Model ID** | `meta/llama-4-scout-17b-16e-instruct` |
| **Parameters** | 17B per expert, 16 experts (MoE) |
| **Context Window** | 128K tokens (supports up to 10M theoretically) |
| **Function Calling** | YES |
| **Quality Comparison** | Beats Gemma 3, Gemini 2.0 Flash-Lite, Mistral 3.1 |
| **Best For** | Best multimodal model in class; very long context |

### Kimi K2 Instruct
| Field | Value |
|-------|-------|
| **Model ID** | `moonshotai/kimi-k2-instruct` / `moonshotai/kimi-k2-instruct-0905` |
| **Parameters** | Large MoE |
| **Context Window** | 128K / 262K tokens |
| **Function Calling** | YES (per NVIDIA docs; tools passed per-request) |
| **Best For** | General instruction following with tool use |

### Qwen3 235B-A22B
| Field | Value |
|-------|-------|
| **Model ID** | `qwen/qwen3-235b-a22b` |
| **Parameters** | 235B total, 22B active (MoE) |
| **Context Window** | 131K tokens |
| **Function Calling** | Likely YES (Qwen3 family native support) |
| **Best For** | Multilingual tasks, strong reasoning |
| **Note** | Check for deprecation notices |

### MiniMax M2.1
| Field | Value |
|-------|-------|
| **Model ID** | `minimaxai/minimax-m2.1` |
| **Parameters** | 230B total, 10B active (MoE) |
| **Context Window** | 205K tokens |
| **Function Calling** | YES (native tool calling) |
| **Quality Comparison** | SWE-bench 74.0%, VIBE avg 88.6 |
| **Best For** | Agentic coding, long-context tool use, instruction following |
| **Special** | Built-in context management when >30% of context used |

### DeepSeek V3.1
| Field | Value |
|-------|-------|
| **Model ID** | `deepseek-ai/deepseek-v3.1` |
| **Parameters** | Large MoE |
| **Context Window** | 128K tokens |
| **Function Calling** | **LIMITED** — "strict function calling" advertised but historically unreliable on NIM |
| **Quality Comparison** | Strong hybrid think/non-think reasoning |
| **Warning** | Test function calling thoroughly before relying on it |

---

## 5. Tier 3: Efficient / Specialized Models

Comparable to Claude Haiku / GPT-4o-mini tier. Good for high-volume, lower-complexity tasks.

### Nemotron Nano 9B v2
| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/nvidia-nemotron-nano-9b-v2` |
| **Parameters** | 9B (hybrid Mamba-2 + MLP + 4 Attention layers) |
| **Context Window** | 131K tokens |
| **Function Calling** | YES (Nemotron family auto-enabled) |
| **Quality Comparison** | IFEval 90.3%, MATH500 97.8%, GPQA 64.0%. Better than Qwen3-8B |
| **Best For** | High-volume enrichment tasks, fast question generation |
| **Reasoning** | Toggle on/off; 6x faster throughput than comparable models |

### Nemotron 3 Nano 30B-A3B
| Field | Value |
|-------|-------|
| **Model ID** | `nvidia/nemotron-3-nano-30b-a3b` |
| **Parameters** | 30B total, 3B active (MoE + Mamba-2) |
| **Context Window** | 131K tokens |
| **Function Calling** | YES (with qwen3_coder parser) |
| **Quality Comparison** | More accurate than GPT-OSS-20B and Qwen3-30B-A3B-Thinking |
| **Best For** | Edge/efficient agentic tasks with tool calling |

### Mistral Small 3.1 24B
| Field | Value |
|-------|-------|
| **Model ID** | `mistralai/mistral-small-3.1-24b-instruct-2503` |
| **Parameters** | 24B (dense) |
| **Context Window** | 128K tokens |
| **Function Calling** | YES (Mistral family auto-enabled) |
| **Best For** | Fast instruction following, moderate-complexity tasks |

### Devstral 2 123B
| Field | Value |
|-------|-------|
| **Model ID** | `mistralai/devstral-2-123b-instruct-2512` |
| **Parameters** | 123B (dense) |
| **Context Window** | 262K tokens |
| **Function Calling** | YES (strong agentic/tool-calling capabilities) |
| **Best For** | Code-related tasks, repository analysis, 256K context |

### Ministral 14B
| Field | Value |
|-------|-------|
| **Model ID** | `mistralai/ministral-14b-instruct-2512` |
| **Parameters** | 14B |
| **Context Window** | 262K tokens |
| **Function Calling** | YES (Mistral family) |
| **Best For** | Long-context tasks on a budget |

### Qwen3 Coder 480B-A35B
| Field | Value |
|-------|-------|
| **Model ID** | `qwen/qwen3-coder-480b-a35b-instruct` |
| **Parameters** | 480B total, 35B active (MoE) |
| **Context Window** | 262K tokens |
| **Function Calling** | Likely YES |
| **Best For** | Code generation, structured output generation |

### Qwen3-Next 80B-A3B (Instruct / Thinking)
| Field | Value |
|-------|-------|
| **Model ID** | `qwen/qwen3-next-80b-a3b-instruct` / `qwen/qwen3-next-80b-a3b-thinking` |
| **Parameters** | 80B total, 3B active (hybrid MoE + Gated Delta Networks) |
| **Context Window** | 262K tokens |
| **Function Calling** | Likely YES |
| **Best For** | Ultra-efficient long-context tasks |

---

## 6. Tier 4: Compact / Edge Models

Suitable for simple classification, routing, or lightweight tasks.

| Model ID | Params | Context | Function Calling | Notes |
|----------|--------|---------|-----------------|-------|
| `meta/llama-3.2-1b-instruct` | 1B | 128K | YES | Smallest Llama; fast routing |
| `meta/llama3-8b-instruct` | 8B | 128K | YES | Good baseline |
| `meta/llama-3.1-70b-instruct` | 70B | 128K | YES | Older gen but solid |
| `google/gemma-3-1b-it` | 1B | 128K | No | Google's tiny model |
| `google/gemma-3-12b-it` | 12B | 128K | No | Mid-range Gemma |
| `google/gemma-3-27b-it` | 27B | 131K | No | Largest Gemma 3 |
| `google/gemma-3n-e2b-it` | ~2B | 128K | No | Efficient Gemma |
| `google/gemma-3n-e4b-it` | ~4B | 128K | No | Efficient Gemma |
| `microsoft/phi-4-mini-instruct` | Small | 131K | No (on NIM) | Microsoft's efficient model |
| `microsoft/phi-3-medium-128k-instruct` | Medium | 128K | No (on NIM) | Good for summarization |
| `qwen/qwq-32b` | 32B | 128K | Likely YES | Reasoning-focused |
| `qwen/qwen2.5-coder-32b-instruct` | 32B | 128K | Likely YES | Code-specialized |

### Legacy / Older Models Also Available
- `nvidia/nemotron-4-340b-instruct` (340B, 128K) — older Nemotron generation
- `nvidia/llama-3.1-nemotron-70b-instruct` (70B, 128K) — older Nemotron
- `nvidia/llama-3.1-nemotron-51b-instruct` (51B, 128K) — older Nemotron
- `meta/codellama-70b` (70B, 128K) — code-specialized
- Various Phi-3/3.5 models (4K-128K context)
- Various coding models (CodeGemma, Codestral, Mamba-Codestral)

---

## 7. Embedding and Reranking Models

These are also available on the NVIDIA NIM free tier (same rate limits apply).

| Model ID | Type | Dimensions | Max Tokens | Notes |
|----------|------|-----------|------------|-------|
| `nvidia/llama-embed-nemotron-8b` | Embedding | High-dim | 33K | Best NVIDIA embedding; retrieval + reranking + classification |
| `nvidia/llama-3.2-nv-embedqa-1b-v2` | Embedding | Standard | 8K | Multilingual (26 languages), optimized for QA retrieval |
| `nvidia/llama-3.2-nv-rerankqa-1b-v2` | Reranker | N/A | 8K | Cross-encoder reranker; pairs with embedqa for RAG |
| `nvidia/llama-nemotron-rerank-1b-v2` | Reranker | N/A | N/A | NeMo Retriever collection; state-of-the-art reranking |

**Important:** Some users have reported HTTP 402 (Payment Required) on embedding/reranker endpoints. The free tier availability for these may be more restricted than for LLM endpoints. Test before relying on them.

---

## 8. Function Calling Support Matrix

Based on official NVIDIA NIM documentation, function calling is **automatically enabled** for:

| Model Family | Function Calling | Notes |
|-------------|-----------------|-------|
| GPT-OSS (20B, 120B) | YES | Full tool_choice support |
| Llama 3.1 (all sizes) | YES | Auto-enabled |
| Llama 3.2 (all sizes) | YES | Auto-enabled |
| Llama 3.3 (all sizes) | YES | Auto-enabled |
| Llama 4 (Scout, Maverick) | YES | Auto-enabled |
| Mistral (all variants) | YES | Auto-enabled |
| Nemotron 3 Nano | YES | qwen3_coder parser |
| Nemotron Nano (all) | YES | Requires `detailed_thinking: off` |
| Nemotron Super (all) | YES | Requires `detailed_thinking: off` |
| Nemotron Ultra (all) | YES | Requires `detailed_thinking: off` |
| Kimi K2 / K2.5 | YES | Pass tools per-request |
| MiniMax M2/M2.1 | YES | Native tool calling |

### Models WITHOUT Function Calling on NIM

| Model Family | Function Calling | Workaround |
|-------------|-----------------|------------|
| DeepSeek R1 / R1-0528 | **NO** | Parse tool calls from response text |
| DeepSeek V3.1 | **LIMITED** | "Strict function calling" advertised; test carefully |
| DeepSeek V3.2 | **NO** | Tool calls appear in response body text |
| Google Gemma (all) | **NO** | Not listed in NIM function calling docs |
| Microsoft Phi (all) | **NO** | Not listed in NIM function calling docs |
| GLM-4.7 / GLM-5 | **Unknown** | Not in official docs; likely NO |
| Qwen3.5 | **Unknown** | Too new; not yet in function calling docs |

### Tool Choice Modes (for supported models)
- `"none"` — Never call tools
- `"auto"` — Model decides whether to call tools
- `{"type": "function", "function": {"name": "..."}}` — Force specific tool

---

## 9. Structured JSON Output

### Recommended: `guided_json` via `nvext`

NVIDIA explicitly recommends `guided_json` over `response_format={"type": "json_object"}` because the latter allows the model to generate any valid JSON (including empty JSON).

```python
response = client.chat.completions.create(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    messages=[{"role": "user", "content": "Extract citations from this text..."}],
    extra_body={
        "nvext": {
            "guided_json": {
                "type": "object",
                "properties": {
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "act": {"type": "string"},
                                "year": {"type": "integer"}
                            },
                            "required": ["section", "act"]
                        }
                    }
                },
                "required": ["citations"]
            }
        }
    }
)
```

### Backend
- **XGrammar** (fast, default) or **Outlines** (slower fallback)
- Works with any model that supports structured generation on NIM

---

## 10. Recommendations for Legal RAG Pipeline

### Primary Model Assignments

| Pipeline Task | Recommended Model | Rationale |
|--------------|-------------------|-----------|
| **Contextual Retrieval** (context summaries) | Nemotron Nano 9B v2 | High throughput, 131K context, good instruction following (IFEval 90.3%) |
| **Question Generation** (QuIM-RAG) | Nemotron Super 49B v1.5 | Strong reasoning + function calling for structured Q&A generation |
| **Answer Synthesis** (final response) | Nemotron Ultra 253B v1 | Best reasoning quality for complex legal analysis |
| **Query Classification** (router) | Nemotron Nano 9B v2 | Fast, cheap, good enough for classification |
| **HyDE** (hypothetical answers) | Nemotron Super 49B v1.5 | Good balance of quality and speed for generating hypothetical documents |
| **GenGround** (claim verification) | Llama 3.3 70B Instruct | Reliable, well-tested, function calling for evidence retrieval |
| **Citation Extraction** (structured JSON) | Any model + `guided_json` | Use schema-constrained output for reliable parsing |

### Fallback Strategy

Given the 40 RPM rate limit, a single-provider strategy may bottleneck a multi-stage pipeline. Consider:

1. **Primary:** NVIDIA NIM (Nemotron family for best quality)
2. **Fallback 1:** Groq (Llama 3.3 70B, 30 RPM, fast)
3. **Fallback 2:** Google Gemini Flash (10 RPM, 1M context, generous TPM)
4. **Fallback 3:** OpenRouter (free Qwen3 models)

### Rate Limit Mitigation
- Batch requests where possible
- Use smaller models (Nano 9B) for high-volume tasks
- Cache responses aggressively (semantic cache in your Phase 0)
- Stagger pipeline stages to avoid concurrent bursts

---

## Key Findings

1. **NVIDIA retired credits in late 2025.** The system is now purely rate-limited (~40 RPM). No token budgets, no monthly caps, no expiry. This is more generous than most free tiers for sustained development use.

2. **Nemotron family is the sweet spot for agentic RAG.** The Nano/Super/Ultra tiering maps perfectly to light/medium/heavy pipeline tasks, all with function calling and structured output support.

3. **DeepSeek models are a trap for agentic tasks on NIM.** Despite being top-tier reasoning models, they lack structured function calling on NIM. Tool calls appear in response text, breaking standard tool_choice workflows.

4. **262K context models are available for free.** Mistral Large 3 (675B), Devstral 2 (123B), Kimi K2.5, and several Qwen models offer 262K context on the free tier — useful for full-document legal analysis.

5. **Embedding/reranking models may require payment.** Some users report HTTP 402 errors. LLM endpoints appear more reliably free than embedding endpoints.

6. **`guided_json` is the correct JSON output mechanism.** Do NOT use `response_format={"type": "json_object"}`. Use `nvext.guided_json` with a JSON schema for reliable structured output.

7. **The catalog is rapidly expanding.** GLM-5 (added 2026-02-17), Qwen3.5 (added 2026-02-16), and DeepSeek V3.2 (added 2025-12-16) are recent additions. Check build.nvidia.com regularly for new models.

8. **Async support is native.** The OpenAI SDK's `AsyncOpenAI` works out of the box with NIM's base URL. Streaming works identically to OpenAI's API.

---

## Risks, Limitations & Open Questions

1. **Rate limits are unpublished per-model.** The ~40 RPM is community-observed, not guaranteed. NVIDIA may change limits without notice.
2. **Context windows may be truncated on free tier.** Models "tend to be context window limited" on trial endpoints (per community reports).
3. **No SLA or uptime guarantee.** This is a trial/prototyping service, not production infrastructure.
4. **Function calling for newer models is unconfirmed.** GLM-5, Qwen3.5, and some Kimi variants are not yet in the official function calling documentation.
5. **Production deployment requires NVIDIA AI Enterprise license.** The free tier is explicitly for prototyping only.

---

## Sources and References

1. [NVIDIA NIM for Developers](https://developer.nvidia.com/nim) — Official NIM landing page
2. [build.nvidia.com — Try NVIDIA NIM APIs](https://build.nvidia.com/) — Model catalog and API playground
3. [NVIDIA NIM Function Calling Documentation](https://docs.nvidia.com/nim/large-language-models/latest/function-calling.html) — Official list of models with function calling support
4. [NVIDIA NIM API Reference](https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html) — Endpoint specifications
5. [NVIDIA NIM Structured Generation](https://docs.nvidia.com/nim/large-language-models/latest/structured-generation.html) — `guided_json` and `nvext` documentation
6. [NVIDIA Developer Forums — Rate Limits](https://forums.developer.nvidia.com/t/request-more-4-000-credits-option-on-build-nvidia-com/344567) — Confirmation of rate-limit-based system replacing credits
7. [NVIDIA Developer Forums — Model Limits](https://forums.developer.nvidia.com/t/model-limits/331075) — ~40 RPM confirmed; per-model limits unpublished
8. [Mastra.ai NVIDIA Models Catalog](https://mastra.ai/models/providers/nvidia) — 71 models with context window sizes
9. [liteLLM NVIDIA NIM Provider](https://docs.litellm.ai/docs/providers/nvidia_nim) — Integration guide and model prefix format
10. [GitHub: free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources) — Community-maintained free API comparison
11. [NVIDIA Developer Forums — DeepSeek Tool Calling](https://forums.developer.nvidia.com/t/native-tool-calls-fail-on-deepseek-3-2/355587) — Confirmation DeepSeek V3.2 function calling fails on NIM
12. [Nemotron Ultra 253B Model Card](https://build.nvidia.com/nvidia/llama-3_1-nemotron-ultra-253b-v1/modelcard) — Benchmarks and capabilities
13. [Nemotron Super 49B v1.5 Model Card](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5/modelcard) — MATH500 97.4%, agentic benchmarks
14. [Nemotron Nano 9B v2 Model Card](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2/modelcard) — IFEval 90.3%, hybrid Mamba-2 architecture
15. [GPT-OSS 120B Model Card](https://build.nvidia.com/openai/gpt-oss-120b/modelcard) — OpenAI's open MoE model
16. [Kimi K2.5 Model Card](https://build.nvidia.com/moonshotai/kimi-k2.5/modelcard) — 1T params, 262K context
17. [MiniMax M2.1 Model Card](https://build.nvidia.com/minimaxai/minimax-m2_1/modelcard) — 230B/10B active, SWE-bench 74.0%
18. [GLM-5 Model Card](https://build.nvidia.com/z-ai/glm5/modelcard) — 744B/40B active, 203K context
19. [NVIDIA Blog — Nemotron Super v1.5](https://developer.nvidia.com/blog/build-more-accurate-and-efficient-ai-agents-with-the-new-nvidia-llama-nemotron-super-v1-5/) — Benchmark details
20. [NVIDIA Blog — Llama 4 on NIM](https://developer.nvidia.com/blog/nvidia-accelerates-inference-on-meta-llama-4-scout-and-maverick/) — Llama 4 availability and benchmarks
21. [NVIDIA Embedding/Reranker Forum Thread](https://forums.developer.nvidia.com/t/clarification-on-nvidia-embedding-reranker-api-access-and-costs/354320) — Free tier uncertainty for embedding models
22. [Analytics Vidhya — 15 Free LLM APIs 2026](https://www.analyticsvidhya.com/blog/2026/01/top-free-llm-apis/) — Comparative context
