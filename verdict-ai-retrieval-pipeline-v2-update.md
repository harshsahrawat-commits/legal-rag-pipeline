# Verdict.ai Retrieval & Generation Pipeline — v2 Update Notes

**Date:** February 24, 2026
**Scope:** Incremental changes to query-time components: Phase 6 (retrieval side), Phase 7 (hallucination mitigation), Phase 8 (evaluation), plus a NEW Phase 0 (query intelligence layer).
**Sources:** Cross-referenced against research on RAG latency optimization, RAG architecture variants (Feb 2026 addendum), and LLM fine-tuning.

> **How to read this document:** Everything in the v2 pipeline's query-time phases that is NOT mentioned here remains unchanged. This document only covers what's new, what's upgraded, and what needs attention. Treat this as a patch on top of v2's Phases 6-8.

---

## What's Unchanged (Confirmed by New Research)

The following v2 query-time decisions are **validated** by the new research:

- **Hybrid Search Architecture:** The four-pathway retrieval (dense + BM25 + QuIM-RAG questions + graph traversal) with Reciprocal Rank Fusion remains best practice. Every research file reinforces hybrid search over single-mode retrieval.

- **Vector Database (Qdrant):** Still the best fit. Self-hosted, native hybrid search, fast metadata filtering, free.

- **Reranker (BGE-reranker-v2-m3):** Still appropriate for v1 launch. The 150→20 reranking step is validated by Anthropic's research. (See Change 5 for a future upgrade path.)

- **Hallucination Mitigation Layers 7a-7c:** Citation verification, temporal consistency checking, and confidence scoring are all sound. No changes needed.

- **RAGAS Evaluation Framework:** Still the dominant open-source option. The metrics and targets from v2 remain valid.

- **Human Evaluation Protocol:** 5-10 lawyers, 200 test queries, four evaluation dimensions — all correct.

---

## NEW: Phase 0 — Query Intelligence Layer

### What This Is

A new layer that sits BEFORE retrieval — between the user's query and the hybrid search pipeline. Its job: decide *whether*, *how*, and *how deeply* to retrieve for each query.

### Why This Is the Highest-Priority Addition

The latency optimization guide is emphatic: "query routing and caching first, index and retrieval optimization second, context compression third, and generation infrastructure as the foundation." Your v2 pipeline has no query-phase intelligence — every query goes through the full hybrid retrieval pipeline regardless of complexity. The research shows this is where the highest-ROI improvements live.

Three components stack together:

### Component A: Semantic Query Cache

**What it does:** Stores (query embedding, response) pairs. When a new query's embedding is sufficiently similar to a cached query, returns the cached response immediately — bypassing the entire retrieval and generation pipeline.

**Impact:** Google Cloud measurements: uncached queries ~6,504ms vs. cache hits ~53ms — a 123× reduction. Production deployments report 18-60% cache hit rates, yielding 40-50% average latency reduction across all queries.

**Why this works especially well for legal queries:** Lawyers frequently ask variations of the same questions. "What is the limitation period for filing a civil suit?" and "What is the statute of limitations for civil cases in India?" and "How long do I have to file a suit under the Limitation Act?" are all the same query. Semantic caching catches these.

```python
# Pseudocode for semantic query cache
import redis
import numpy as np
from qdrant_client import QdrantClient

class SemanticCache:
    def __init__(self, similarity_threshold=0.92, ttl_seconds=86400):
        self.redis = redis.Redis()  # For response storage
        self.qdrant = QdrantClient()  # For embedding similarity search
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds  # Cache entries expire after 24 hours
    
    def get(self, query_embedding):
        """Check if a sufficiently similar query has been answered before."""
        results = self.qdrant.search(
            collection_name="query_cache",
            query_vector=query_embedding,
            limit=1,
            score_threshold=self.threshold
        )
        
        if results:
            cache_key = results[0].payload["cache_key"]
            cached_response = self.redis.get(cache_key)
            if cached_response:
                return {
                    "hit": True,
                    "response": json.loads(cached_response),
                    "similarity": results[0].score,
                    "original_query": results[0].payload["original_query"]
                }
        
        return {"hit": False}
    
    def set(self, query_text, query_embedding, response):
        """Cache a new query-response pair."""
        cache_key = f"cache:{uuid4()}"
        
        # Store response in Redis with TTL
        self.redis.setex(
            cache_key, 
            self.ttl,
            json.dumps(response)
        )
        
        # Store embedding in Qdrant for similarity search
        self.qdrant.upsert(
            collection_name="query_cache",
            points=[{
                "id": str(uuid4()),
                "vector": query_embedding,
                "payload": {
                    "cache_key": cache_key,
                    "original_query": query_text,
                    "cached_at": datetime.now().isoformat()
                }
            }]
        )
    
    def invalidate_for_act(self, act_name):
        """When an Act is amended, invalidate cached responses that cite it."""
        # Search cache entries mentioning this Act and delete them
        # This prevents serving stale legal information
        matching = self.qdrant.scroll(
            collection_name="query_cache",
            scroll_filter=Filter(
                must=[FieldCondition(key="acts_cited", match=MatchValue(value=act_name))]
            )
        )
        for point in matching:
            self.redis.delete(point.payload["cache_key"])
            self.qdrant.delete(collection_name="query_cache", points_selector=[point.id])
```

**Critical: Cache staleness for legal data.** The research warns about cache poisoning (SAFE-CACHE, Nature 2026: 52.77% adversarial attack success rate on GPTCache). For legal queries, staleness is a real risk — an amendment could make a cached answer wrong. Mitigations:

1. **TTL-based expiration:** Default 24-hour TTL. Queries about rapidly-changing areas (recent amendments, ongoing cases) get 1-hour TTL.
2. **Amendment-triggered invalidation:** When the ingestion pipeline detects an amendment (Phase 1 Change Detection Agent), invalidate all cache entries that reference the amended Act.
3. **Similarity threshold tuning:** Use 0.92 (high) to avoid false matches. "What's the punishment for cheating under IPC?" and "What's the punishment for cheating under BNS?" are semantically very similar but have different correct answers.

**Storage estimate for cache:**
- Assume 10,000 unique queries cached at any time
- Each entry: 768-dim embedding (3KB) + response payload (~5KB) = ~8KB
- Total: ~80MB — trivial

**Cost:** $0 (self-hosted Qdrant + Redis, already deployed)

### Component B: Adaptive RAG Query Router

**What it does:** A lightweight classifier that routes each query to the optimal retrieval path, avoiding the full pipeline for queries that don't need it.

**Impact:** Adaptive RAG (Jeong et al., NAACL 2024) reports 35% P50 latency reduction and 28% LLM API cost savings with 8% accuracy improvement. TARG (Training-free Adaptive Retrieval Gating, November 2025) reduces retrieval operations by 70-90% while matching accuracy.

**Four retrieval paths:**

```
User Query
    │
    ├── [Semantic Cache] → Cache hit? → Return cached response (~50ms)
    │
    └── [Query Router] → Classify query complexity
        │
        ├── SIMPLE (direct lookup)
        │   "What does Section 138 NI Act say?"
        │   "Define 'cognizable offence' under CrPC"
        │   → Knowledge Graph direct query (100-200ms)
        │   → Skip vector search entirely
        │
        ├── STANDARD (single-hop retrieval)
        │   "What is the punishment for cheating?"
        │   "Can a landlord evict a tenant for personal use?"
        │   → Standard hybrid retrieval pipeline from v2 (500-800ms)
        │
        ├── COMPLEX (multi-hop retrieval)
        │   "Compare the eviction grounds under Delhi and Mumbai rent control laws"
        │   "What is the interplay between Section 498A IPC and the DV Act?"
        │   → Full hybrid retrieval + graph traversal + RAPTOR (1-2s)
        │
        └── ANALYTICAL (multi-hop + active retrieval)
            "Trace the evolution of Section 377 jurisprudence from Naz Foundation to Navtej Johar"
            "What are ALL grounds for eviction under Delhi Rent Control Act and how has each been interpreted?"
            → FLARE-style active retrieval (see Change 3) (2-5s)
```

**Router implementation — start rule-based, evolve to learned:**

```python
# Pseudocode for query router — Phase 1: Rule-based
def route_query(query_text, query_embedding):
    # Step 1: Check cache first (always)
    cache_result = semantic_cache.get(query_embedding)
    if cache_result["hit"]:
        return {"path": "CACHED", "response": cache_result["response"]}
    
    # Step 2: Classify query complexity
    complexity = classify_complexity(query_text)
    return {"path": complexity}

def classify_complexity(query_text):
    """Rule-based classifier — v1 implementation."""
    query_lower = query_text.lower()
    
    # SIMPLE: Direct section/definition lookups
    simple_patterns = [
        r"what does section \d+",
        r"what is section \d+",
        r"define ['\"]?\w+['\"]?",
        r"text of section",
        r"read section",
        r"show me section"
    ]
    if any(re.search(p, query_lower) for p in simple_patterns):
        return "SIMPLE"
    
    # ANALYTICAL: Comparative, historical, exhaustive queries
    analytical_signals = [
        "compare", "contrast", "evolution", "trace", "all grounds",
        "all provisions", "every", "comprehensive", "interplay between",
        "relationship between", "how has .* been interpreted"
    ]
    if any(re.search(s, query_lower) for s in analytical_signals):
        return "ANALYTICAL"
    
    # COMPLEX: Multi-entity, multi-act, or cross-jurisdictional
    if len(extract_act_references(query_lower)) > 1:
        return "COMPLEX"
    if len(extract_section_references(query_lower)) > 2:
        return "COMPLEX"
    
    # Default: STANDARD
    return "STANDARD"
```

**Future upgrade (Phase 3+): Learned router using TARG.** TARG uses prefix logit statistics — mean token entropy, top-1/top-2 probability margins — to gate retrieval without any training. This eliminates 70-90% of unnecessary retrievals. Implement after you have sufficient query logs to validate the rule-based router.

**Cost:** $0 for rule-based router (pure Python logic, <10ms overhead). TARG requires access to LLM logits — free if using self-hosted model, adds ~10ms for API-based models.

### Component C: Selective Query Rewriting (HyDE)

**What it does:** For complex or ambiguous queries only, generates a hypothetical answer document via LLM, then uses its embedding for retrieval instead of the raw query embedding.

**Impact:** HyDE (Gao et al., 2022) yields +18-29% NDCG improvement by bridging the vocabulary gap between how lawyers ask questions and how legal text is written.

**When to use:** ONLY for COMPLEX and ANALYTICAL queries routed by the classifier. Never for SIMPLE or STANDARD queries — the 300-400ms overhead isn't justified.

**Why HyDE matters for Indian law:** A lawyer asks "Can a company director be held personally liable for company debts?" The legal text uses language like "lifting the corporate veil" and "personal guarantee under Section 2(58) of Companies Act." HyDE generates a hypothetical answer using legal terminology, making the embedding much closer to the actual statutory text.

```python
# Pseudocode for selective HyDE
def maybe_rewrite_query(query_text, route):
    """Only apply HyDE for complex/analytical queries."""
    if route not in ["COMPLEX", "ANALYTICAL"]:
        return query_text, embed(query_text)  # Return original
    
    # Generate hypothetical answer
    prompt = f"""You are an expert Indian lawyer. Given this legal research question, 
    write a brief (2-3 sentence) hypothetical answer using proper Indian legal 
    terminology, Act names, and section numbers. This will be used for retrieval, 
    so include the specific legal terms that would appear in the relevant statutes 
    and judgments.
    
    Question: {query_text}
    
    Hypothetical answer:"""
    
    hypothetical = llm.generate(prompt, model="claude-haiku", max_tokens=200)
    
    # Embed the hypothetical answer instead of the raw query
    hyde_embedding = embed(hypothetical)
    
    # Return both — use hyde_embedding for retrieval, original query for generation
    return query_text, hyde_embedding

# Cost: ~$0.001 per HyDE call (Claude Haiku, ~300 tokens)
# Only invoked for ~10-20% of queries (complex/analytical)
```

### Full Query Intelligence Layer — End-to-End Flow

```
User Query ("Can a landlord evict a tenant for personal use in Delhi?")
    │
    │── Step 1: Embed query (20ms)
    │
    │── Step 2: Check semantic cache (5ms)
    │   └── Miss → continue
    │
    │── Step 3: Route query → STANDARD (5ms)
    │
    │── Step 4: HyDE? → No (STANDARD queries skip HyDE)
    │
    │── Step 5: Proceed to hybrid retrieval (v2 pipeline)
    │
    │── [After generation] Step 6: Cache response for future queries
```

```
User Query ("Compare eviction grounds under Delhi and Mumbai rent control laws")
    │
    │── Step 1: Embed query (20ms)
    │
    │── Step 2: Check semantic cache (5ms)
    │   └── Miss → continue
    │
    │── Step 3: Route query → COMPLEX (5ms)
    │
    │── Step 4: HyDE → Generate hypothetical answer (300ms)
    │
    │── Step 5: Proceed to full hybrid retrieval + graph traversal 
    │   using HyDE embedding for vector search, original query for BM25
    │
    │── [After generation] Step 6: Cache response
```

### Cost and Latency Impact of Phase 0

| Component | Latency Added | Cost per Query | Expected Hit Rate |
|-----------|--------------|----------------|-------------------|
| Query embedding | 20ms | ~$0.0001 | 100% (always runs) |
| Semantic cache check | 5ms | $0 | 30-50% expected hit rate |
| Query router | 5ms | $0 | 100% (always runs) |
| HyDE (when invoked) | 300ms | ~$0.001 | 10-20% of queries |

**Net latency impact:** For cache hits (30-50% of queries): saves 1,500-2,000ms per query. For routed-simple queries: saves 500-800ms. For standard queries: adds ~30ms overhead (negligible). For complex queries with HyDE: adds ~300ms but improves retrieval quality significantly.

**Expected average latency improvement across all queries: 40-55%**

---

## CHANGE 1: Phase 6 (Retrieval Side) — Matryoshka Funnel Retrieval

### What's Changing

Replace single-stage vector search with a two-stage funnel using the dual Matryoshka embeddings stored at ingestion time (see Ingestion Plan, Change 4).

### How It Works

```
Query Embedding (768-dim)
    │
    │── Truncate to 64-dim
    │
    │── Stage 1: FAST SEARCH on 64-dim vectors
    │   Search ~3M vectors at 64 dimensions
    │   Return top 1,000 candidates
    │   Latency: ~15-20ms (vs. ~50-80ms for 768-dim search)
    │
    │── Stage 2: PRECISION RE-SCORE on 768-dim vectors
    │   Re-score only the 1,000 candidates at full 768 dimensions
    │   Return top 100 (feeds into BM25 fusion + reranking)
    │   Latency: ~5-10ms (only scoring 1,000 vectors, not searching 3M)
    │
    │── Continue to Reciprocal Rank Fusion with BM25 results (unchanged)
```

### Implementation

```python
# Pseudocode for Matryoshka funnel retrieval
def funnel_vector_search(query_embedding, top_k_final=100):
    # Stage 1: Fast broad search on 64-dim vectors
    query_64 = query_embedding[:64]  # Truncate query to match fast index
    
    broad_candidates = qdrant_client.search(
        collection_name="legal_chunks",
        query_vector=("fast", query_64),   # Search the 64-dim named vector
        limit=1000,                         # Broad candidate pool
        with_payload=False,                 # Skip payload for speed
        with_vectors=False
    )
    
    candidate_ids = [hit.id for hit in broad_candidates]
    
    # Stage 2: Precision re-score on 768-dim vectors
    # Fetch full vectors only for the 1,000 candidates
    precise_results = qdrant_client.search(
        collection_name="legal_chunks",
        query_vector=("full", query_embedding),  # Full 768-dim query
        limit=top_k_final,
        query_filter=Filter(
            must=[HasIdCondition(has_id=candidate_ids)]  # Only score candidates
        ),
        with_payload=True  # Now fetch metadata for downstream use
    )
    
    return precise_results
```

### Latency Impact

| Approach | Vector Search Latency | Quality |
|----------|----------------------|---------|
| v2: Single 768-dim search (top 100) | 50-80ms | Baseline |
| New: 64-dim broad (top 1000) → 768-dim rescore (top 100) | 20-30ms total | Same or better (larger initial candidate pool) |

The funnel is faster AND potentially more accurate because it considers 1,000 initial candidates instead of directly selecting the top 100.

---

## CHANGE 2: Phase 6 (Retrieval Side) — Parent-Document Context Expansion

### What's Changing

After retrieval and reranking, expand matched chunks with their parent context before passing to the LLM.

### Why

A sub-section might be the precise match, but the LLM needs the full section (with provisos, explanations, definitions) to generate a correct legal answer. The ARAGOG benchmark found Sentence-Window Retrieval had the "highest retrieval precision" among tested techniques. The 2025 CDC policy RAG study found optimized chunking achieved 0.79-0.82 faithfulness vs 0.47-0.51 for naive approaches — a 65% quality improvement.

### Implementation

```python
# Pseudocode — insert after reranking (after top 20 selection), before LLM generation
def expand_with_parent_context(reranked_chunks, parent_store, max_context_tokens=30000):
    """Expand matched chunks with parent section context."""
    expanded = []
    seen_parents = set()
    total_tokens = 0
    
    for chunk in reranked_chunks:
        # Always include the matched chunk itself
        expanded_entry = {
            "matched_chunk": chunk,
            "parent_text": None,
            "siblings": []
        }
        
        # Fetch parent context (if exists and not already included)
        parent_id = chunk.metadata.get("parent_chunk_id")
        if parent_id and parent_id not in seen_parents:
            parent_text = parent_store.get(parent_id)
            if parent_text:
                parent_tokens = token_count(parent_text)
                if total_tokens + parent_tokens <= max_context_tokens:
                    expanded_entry["parent_text"] = parent_text
                    seen_parents.add(parent_id)
                    total_tokens += parent_tokens
        
        # For judgment chunks: always include judgment header for case context
        header_id = chunk.metadata.get("judgment_header_chunk_id")
        if header_id and header_id not in seen_parents:
            header_text = parent_store.get(header_id)
            if header_text:
                expanded_entry["judgment_header"] = header_text
                seen_parents.add(header_id)
                total_tokens += token_count(header_text)
        
        expanded.append(expanded_entry)
        total_tokens += token_count(chunk.text)
    
    return expanded

def format_for_llm(expanded_chunks):
    """Format expanded chunks for the generation prompt."""
    formatted = []
    for entry in expanded_chunks:
        block = f"[Source: {entry['matched_chunk'].metadata['act_name']}, "
        block += f"{entry['matched_chunk'].metadata.get('section_number', 'N/A')}]\n"
        
        if entry.get("parent_text"):
            block += f"[Full Section Context:]\n{entry['parent_text']}\n\n"
            block += f"[Specifically Matched Passage:]\n{entry['matched_chunk'].text}"
        else:
            block += entry['matched_chunk'].text
        
        formatted.append(block)
    
    return "\n\n---\n\n".join(formatted)
```

### Latency Impact

Parent document fetching from Redis: 1-5ms total (batch fetch of ~20 parent IDs). Negligible vs. overall pipeline latency. The LLM receives 20-50% more context tokens, which slightly increases generation time, but the quality improvement (correct handling of provisos and explanations) reduces the need for hallucination-mitigation retries.

---

## CHANGE 3: Phase 6 (Retrieval Side) — FLARE-Style Active Retrieval for Analytical Queries

### What's Changing

For queries routed as ANALYTICAL by the query router, use active mid-generation retrieval instead of single-shot retrieval.

### Why

FLARE (Forward-Looking Active Retrieval, Jiang et al., EMNLP 2023) outperforms single-retrieval RAG on multi-hop QA tasks. For legal research, many important queries require synthesizing information from multiple, different sources that a single retrieval pass would miss.

**Example:** "What are all the grounds for eviction under the Delhi Rent Control Act and how have courts interpreted each?"

With single retrieval, you might get chunks covering the main 3-4 grounds but miss the less common ones. With FLARE, the model generates the first few grounds confidently, then when it reaches a ground it's uncertain about, it triggers targeted retrieval for that specific ground and its case law.

### How It Works

```python
# Pseudocode for FLARE-style active retrieval
def generate_with_active_retrieval(query, initial_chunks, confidence_threshold=0.4):
    """Generate response with mid-generation retrieval when confidence drops."""
    
    # Start with initial retrieval results
    context = format_for_llm(initial_chunks)
    generated_so_far = ""
    all_retrieved_chunks = set(c.chunk_id for c in initial_chunks)
    max_retrievals = 5  # Cap to prevent runaway retrieval loops
    retrieval_count = 0
    
    while retrieval_count < max_retrievals:
        # Generate next segment with token probabilities
        prompt = f"""Context: {context}
        
        Query: {query}
        
        Response so far: {generated_so_far}
        
        Continue the response:"""
        
        result = llm.generate(
            prompt, 
            max_tokens=300,  # Generate in segments
            return_logprobs=True
        )
        
        segment = result.text
        logprobs = result.logprobs
        
        # Check if any tokens have low confidence
        min_confidence = min(logprobs) if logprobs else 1.0
        
        if min_confidence < confidence_threshold:
            # Low confidence detected — model is uncertain
            # Use the uncertain segment as a retrieval query
            retrieval_query = extract_uncertain_claim(segment, logprobs)
            
            # Retrieve additional context
            new_chunks = hybrid_search(retrieval_query, exclude=all_retrieved_chunks)
            if new_chunks:
                context += "\n\n" + format_for_llm(new_chunks)
                all_retrieved_chunks.update(c.chunk_id for c in new_chunks)
                retrieval_count += 1
                
                # Re-generate the uncertain segment with new context
                continue  # Don't append the uncertain segment yet
        
        # Confidence is fine — append and continue
        generated_so_far += segment
        
        # Check if response is complete
        if result.finish_reason == "stop":
            break
    
    return generated_so_far
```

### When This Runs

Only for ANALYTICAL queries (~5-10% of total queries). The query router ensures standard queries never pay this latency cost.

### Latency Profile

Each FLARE retrieval trigger adds ~200-500ms (embedding + vector search + re-injection). With a cap of 5 additional retrievals:

| Query Type | Expected FLARE Triggers | Total Latency |
|------------|------------------------|---------------|
| Standard (never FLARE'd) | 0 | 500-800ms |
| Complex (never FLARE'd) | 0 | 1-2s |
| Analytical (FLARE active) | 2-4 triggers typical | 2-5s |

This is acceptable because analytical queries are inherently complex — a lawyer asking for a comprehensive analysis expects a slightly longer wait for a thorough answer.

### Cost

Each FLARE trigger costs one additional retrieval (free — self-hosted Qdrant) + one partial LLM generation (~$0.005 per trigger). For 5-10% of queries with 2-4 triggers each: adds ~$20-40/month at moderate query volume.

**Important caveat from the research:** "FLARE's dynamic retrieval + semantic caching: FLARE's mid-generation retrievals produce ephemeral sentence fragments unlikely to match cached queries. Semantic caching ROI is low for FLARE-heavy workloads." This is fine — FLARE only runs on analytical queries which are rarely cacheable anyway.

---

## CHANGE 4: Phase 7 — API Prompt Caching for Generation

### What's Changing

Restructure all LLM prompts to maximize API-level prompt caching.

### Why

The latency supplement reports: Anthropic Prompt Caching reduces latency by up to 85% and costs by up to 90% for long prompts. One developer reported going from $8,000/month to $800/month. For API-based RAG, this is "the single highest-ROI optimization available."

### How

Structure every LLM call (generation, hallucination mitigation, confidence scoring) so static content is at the beginning and dynamic content is at the end:

```python
# Pseudocode for prompt structure optimized for caching
def build_generation_prompt(query, retrieved_chunks):
    # STATIC PREFIX — cached across all queries (set cache breakpoint here)
    system_prompt = """You are Verdict.ai, an expert Indian legal research assistant.
    
    RULES:
    1. Only cite statutes and cases that appear in the provided context
    2. If you are unsure, say so explicitly — never fabricate citations
    3. Always specify whether a cited law is currently in force or repealed
    4. Include the relevant section numbers and Act names in every citation
    5. If the question involves temporal aspects, clarify which version of the 
       law applies and as of what date
    6. Structure your response with: Brief Answer → Detailed Analysis → 
       Relevant Provisions → Case Law (if applicable) → Caveats
    
    COURT HIERARCHY (for precedent weight):
    Supreme Court > High Court > District Court > Tribunal
    Constitution Bench > Full Bench > Division Bench > Single Judge
    
    CURRENT LEGAL TRANSITION (critical):
    - IPC (1860) → Bharatiya Nyaya Sanhita (2023) — effective July 1, 2024
    - CrPC (1973) → Bharatiya Nagarik Suraksha Sanhita (2023) — effective July 1, 2024
    - Indian Evidence Act (1872) → Bharatiya Sakshya Adhiniyam (2023) — effective July 1, 2024
    Always clarify which regime applies based on the date of the cause of action.
    """  # <-- CACHE BREAKPOINT: everything above this is identical for every query
    
    # DYNAMIC SUFFIX — varies per query
    context = format_for_llm(retrieved_chunks)
    user_message = f"""RETRIEVED LEGAL CONTEXT:
    {context}
    
    LAWYER'S QUESTION:
    {query}
    
    Provide a thorough, well-cited answer:"""
    
    return system_prompt, user_message

# For Anthropic API — use cache_control parameter
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5-20250514",
    system=[{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"}  # Mark for caching
    }],
    messages=[{"role": "user", "content": user_message}],
    max_tokens=4096
)
```

### Apply Same Pattern to Hallucination Mitigation Calls

Layer 7d (Grounded Refinement) also calls an LLM. Use the same caching pattern:

```python
# Grounded refinement — static instructions cached, dynamic response varies
refinement_system = """You are a legal accuracy auditor for Indian law.
    
    For each claim in the AI-generated response:
    1. Is it directly supported by the source documents? (Grounded/Ungrounded)
    2. Is any source mischaracterized? (Accurate/Mischaracterized)  
    3. Are there important caveats from sources that were omitted? (Complete/Incomplete)
    4. Are any cited laws repealed or amended? (Current/Outdated)
    
    Return a revised response that removes ungrounded claims, corrects 
    mischaracterizations, and adds critical omitted caveats."""
# <-- CACHE BREAKPOINT

# Dynamic part: the specific response + retrieved chunks being audited
```

### Cost Impact

| Component | v2 Monthly Estimate | With Prompt Caching | Savings |
|-----------|--------------------|--------------------|---------|
| LLM API (generation) | $200-500 | $30-100 | 80-85% |
| LLM API (hallucination mitigation) | Included above | Included above | Same |
| **Total LLM API** | **$200-500** | **$30-100** | **$170-400/month** |

This is not a projection — it's based on Anthropic's documented 85% latency reduction and 90% cost reduction for prompts with cacheable prefixes. Your system prompt + legal rules are ~800 tokens, well above the minimum for caching.

### Latency Impact

First query after cache warm-up: normal latency. Subsequent queries within the 5-minute cache TTL: up to 85% faster for the prefill phase. Since the system prompt is hit on every query, the cache will stay warm under any reasonable query volume.

---

## CHANGE 5: Phase 7 — GenGround Systematic Verification (Upgrade to Layer 7d)

### What's Changing

Upgrade the Grounded Refinement layer (7d) from a single-pass "is this grounded?" check to a systematic per-claim verification using the GenGround pattern.

### Why

GenGround (Shi et al., 2024) formalizes the generate-then-ground pattern with a key improvement: instead of asking an LLM "is this response grounded?" (which is a vague question), it treats each claim as a separate retrieval query and checks alignment with independently re-retrieved evidence. This catches misattributions that a single-pass check misses.

### Implementation

```python
# Pseudocode for GenGround-style verification (replaces v2 Layer 7d)
def genground_verification(response, retrieved_chunks, knowledge_graph):
    """Systematic per-claim verification with independent re-retrieval."""
    
    # Step 1: Extract individual claims from the response
    claims = extract_claims(response)
    # Example claims:
    # "Section 138 NI Act requires the cheque to be presented within 6 months"
    # "In Dashrath Rupsingh Rathod v. State (2014), the SC held that..."
    
    verification_results = []
    
    for claim in claims:
        # Step 2: For each claim, independently retrieve supporting evidence
        # This is the key GenGround insight — don't just check against the 
        # original retrieved chunks, re-retrieve specifically for this claim
        claim_evidence = hybrid_search(
            query=claim.text, 
            top_k=5,
            metadata_filter=claim.get_relevant_filters()  # e.g., specific Act
        )
        
        # Step 3: Also check the knowledge graph for citation accuracy
        if claim.contains_citation:
            kg_verification = knowledge_graph.verify_citation(
                citation=claim.citation,
                attributed_content=claim.attributed_content
            )
        else:
            kg_verification = None
        
        # Step 4: Score alignment between claim and evidence
        alignment = llm.check_alignment(
            claim=claim.text,
            evidence=[e.text for e in claim_evidence],
            kg_result=kg_verification,
            prompt="Does the evidence support this specific legal claim? "
                   "Consider: factual accuracy, correct attribution, "
                   "temporal validity, and jurisdictional applicability."
        )
        
        verification_results.append({
            "claim": claim,
            "evidence": claim_evidence,
            "kg_check": kg_verification,
            "verdict": alignment.verdict,  # SUPPORTED / UNSUPPORTED / PARTIALLY_SUPPORTED
            "confidence": alignment.confidence,
            "issues": alignment.issues  # List of specific problems found
        })
    
    # Step 5: Reconstruct response
    # - Remove UNSUPPORTED claims entirely
    # - Add caveats to PARTIALLY_SUPPORTED claims
    # - Keep SUPPORTED claims as-is
    revised_response = reconstruct_response(response, verification_results)
    
    # Step 6: Add transparency metadata
    revised_response.verification_summary = {
        "total_claims": len(claims),
        "supported": sum(1 for v in verification_results if v["verdict"] == "SUPPORTED"),
        "partially_supported": sum(1 for v in verification_results if v["verdict"] == "PARTIALLY_SUPPORTED"),
        "removed": sum(1 for v in verification_results if v["verdict"] == "UNSUPPORTED"),
        "verification_method": "GenGround per-claim re-retrieval"
    }
    
    return revised_response
```

### Cost vs. v2 Layer 7d

GenGround is more expensive per query because it does per-claim re-retrieval + per-claim LLM checks:

| | v2 Layer 7d | GenGround Upgrade |
|---|---|---|
| LLM calls per response | 1 (single verification pass) | 1 + N (N = number of claims, typically 5-15) |
| Retrieval calls per response | 0 | N (one per claim) |
| Accuracy | Good (catches obvious groundedness issues) | Better (catches subtle misattributions) |
| Cost per query | ~$0.005 | ~$0.02-0.05 |
| Latency added | ~500ms | ~1-2s |

**Recommendation:** Run GenGround on STANDARD, COMPLEX, and ANALYTICAL queries. For SIMPLE queries (direct lookups), the v2 Layer 7d single-pass check is sufficient.

---

## CHANGE 6: Phase 7 — Finetune-RAG Improvements from Fine-Tuning Research

### What's Changing

The v2 plan for Layer 7e (Finetune-RAG — training the generation model to resist hallucination) is correct in concept but missing practical implementation details from the fine-tuning guide.

### Specific Improvements

**Synthetic training data for hallucination resistance:**

```python
# Pseudocode for creating Finetune-RAG training data
def create_finetune_rag_dataset(real_queries, chunk_database):
    """Create training examples that teach the model to resist misleading context."""
    training_examples = []
    
    for query in real_queries:
        # Get the correct retrieval results
        correct_chunks = retrieve(query)
        correct_answer = generate_verified_answer(query, correct_chunks)
        
        # Create a "poisoned" version with deliberately wrong chunks mixed in
        wrong_chunks = get_similar_but_wrong_chunks(query, chunk_database)
        mixed_chunks = interleave(correct_chunks[:3], wrong_chunks[:2])
        
        # Training example: model should still give correct answer 
        # even when wrong chunks are present
        training_examples.append({
            "system": "You are a legal assistant. Answer based ONLY on provided context. "
                      "If context is contradictory, prefer the most authoritative source. "
                      "If unsure, say 'I cannot determine this from the provided sources.'",
            "context": format_chunks(mixed_chunks),
            "query": query,
            "expected_response": correct_answer  # Grounded in correct chunks only
        })
        
        # Also create "insufficient context" examples
        training_examples.append({
            "system": "...",
            "context": format_chunks(wrong_chunks),  # Only wrong chunks
            "query": query,
            "expected_response": "Based on the provided sources, I cannot reliably answer "
                                 "this question. The available context does not contain "
                                 "sufficient information about [specific gap]. I recommend "
                                 "checking [specific Act/Section] directly."
        })
    
    return training_examples
```

**Use DPO for preference learning:**

After initial Finetune-RAG training, use DPO with pairs:
- **Chosen:** Response grounded in correct sources, acknowledging uncertainty where appropriate
- **Rejected:** Response that confidently cites a wrong source or fabricates a citation

This teaches the model the *preference* for caution over confident hallucination.

**Dataset size recommendation:** The fine-tuning guide's LIMA finding applies here — 1,000 high-quality examples of hallucination resistance are more valuable than 50,000 generic legal Q&A pairs. Focus on:
- 200 examples with deliberately misleading context (model should resist)
- 200 examples with insufficient context (model should say "I don't know")
- 200 examples with correct context (model should answer normally)
- 200 examples with temporally outdated context (model should flag currency)
- 200 examples with misattributed citations (model should detect and correct)

### Timeline

This is a Phase 3 activity (Weeks 15-20) as in v2. The improvement is in how you build the training data, not when.

---

## CHANGE 7: Phase 8 — Updated Evaluation Metrics

### What's Changing

Add latency metrics and query-routing effectiveness to the RAGAS evaluation pipeline.

### New Metrics

Add these to the v2 evaluation table:

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| TTFT (Time to First Token) - Simple queries | <200ms | Query intelligence layer effectiveness |
| TTFT - Standard queries | <800ms | Core retrieval pipeline speed |
| TTFT - Complex queries | <2s | Multi-hop retrieval speed |
| TTFT - Analytical queries | <5s | FLARE active retrieval speed |
| Cache hit rate | >30% | Semantic cache effectiveness |
| Query routing accuracy | >90% | Router correctly classifies complexity |
| GenGround claim verification rate | >95% supported | Response groundedness |
| Parent context utilization | >80% of retrievals | Parent-doc expansion working correctly |

### Updated Continuous Improvement Loop

Add to the v2 weekly/monthly/quarterly cycle:

```
Weekly (add to existing):
├── Monitor cache hit rate and TTL effectiveness
├── Review query routing decisions (sample 50 queries, check classification)
├── Track FLARE trigger frequency (should be 5-10% of queries)
├── Monitor prompt cache hit rate (Anthropic API dashboard)

Monthly (add to existing):
├── Analyze cache staleness — any cases where cached responses became incorrect?
├── Tune query router rules based on misclassification patterns
├── Review FLARE retrieval patterns — are the additional retrievals finding useful context?
├── Evaluate GenGround verification rate — are too many claims being flagged?

Quarterly (add to existing):
├── Compare latency metrics against targets
├── Evaluate whether learned router (TARG) should replace rule-based router
└── Cost analysis: actual API spend vs. projections with prompt caching
```

---

## Updated Implementation Roadmap (Retrieval Phases Only)

Changes to the v2 timeline are marked with **[NEW]**.

### Phase 1: Foundation (Weeks 5-6) — Add Retrieval Basics

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 5-6 | Set up hybrid search. Basic reranking. **[NEW] Implement Matryoshka funnel retrieval (2-stage vector search).** **[NEW] Implement parent-document context expansion.** Minimal chat interface. First demo. | Retrieval architecture upgraded |

### Phase 2: Intelligence Layer (Weeks 7-14)

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 9-10 | **[NEW] Build semantic query cache (Redis + Qdrant cache collection).** **[NEW] Implement rule-based query router (4 paths).** | New — Query Intelligence Layer |
| 13-14 | **[NEW] Implement selective HyDE for complex/analytical queries.** **[NEW] Restructure all LLM prompts for API prompt caching.** | New — Query optimization |

### Phase 3: Trust Layer (Weeks 15-20)

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 15-16 | Citation verification (7a) and temporal checking (7b). Unchanged. | Unchanged |
| 17-18 | Confidence scoring (7c). **[UPGRADED] GenGround systematic verification replacing basic 7d.** RAGAS evaluation pipeline. **[NEW] Add latency and routing metrics.** | Hallucination mitigation upgraded |
| 19-20 | **[NEW] Implement FLARE active retrieval for analytical queries.** Lawyer evaluation (200 queries). Fix issues. | FLARE added |

### Phase 4: Scale (Weeks 21-30)

| Week | Deliverable | Status vs. v2 |
|------|------------|----------------|
| 25-28 | **[NEW] Create Finetune-RAG training dataset with DPO pairs.** **[NEW] Implement amendment-triggered cache invalidation.** Multi-tenancy. | Hallucination training + cache management |

---

## Updated Monthly Operating Costs

| Component | v2 Estimate | Updated Estimate | Notes |
|-----------|-------------|------------------|-------|
| GPU server | $100-200 | $100-200 | Unchanged |
| LLM API (generation + hallucination mitigation) | $200-500 | $30-100 | ↓ 80-85% from prompt caching |
| Cloud server (Qdrant + Neo4j + app) | $100-300 | $100-300 | Unchanged |
| Incremental ingestion | $50-100 | $50-100 | Unchanged |
| **[NEW] GenGround additional LLM calls** | — | $30-80 | Per-claim verification |
| **[NEW] FLARE additional retrievals + generation** | — | $20-40 | Only for analytical queries |
| **[NEW] HyDE query rewriting** | — | $5-10 | Only for complex/analytical queries |
| **Total monthly** | **$450-1,100** | **$335-830** | **↓ ~$100-270 savings from prompt caching** |

**Net result:** Despite adding GenGround, FLARE, and HyDE, the prompt caching savings more than compensate. Total monthly costs actually decrease.

**Updated break-even:** At ₹10,000/month per lawyer: **4-10 paying customers** (improved from v2's 5-12).

---

## Future Roadmap: Techniques to Watch (Don't Implement Yet)

### GFM-RAG for Knowledge Graph Retrieval (NeurIPS 2025)

An 8M-parameter Graph Neural Network pretrained on 60 knowledge graphs that reasons over graph structure for retrieval. Achieves SOTA on multi-hop QA and generalizes zero-shot to unseen datasets. Could replace manual Cypher queries with learned graph traversal. **Evaluate when your Neo4j graph is populated (Phase 3+).**

### Disco-RAG for Legal Reasoning (2026)

Makes RAG aware of rhetorical structure — claims, evidence, counterarguments, conclusions. Natural fit for court judgments where distinguishing "petitioner's argument" from "court's holding" is critical. **Very new, no production implementations yet.**

### PageIndex for Structured Navigation (Sep 2025)

98.7% accuracy on FinanceBench by building tree indices from document structure and using LLM reasoning to navigate them. Perfect for deeply hierarchical Indian statutes. **"Not production-ready as of late 2025."** But the principle should inform how you structure your RAPTOR trees.

### RankRAG Unified Reranking (NVIDIA, 2024)

Trains a single LLM to handle both context ranking and answer generation, eliminating the separate reranker. Outperforms GPT-4 on 9 benchmarks. **Include ranking capability in your Finetune-RAG training data (Phase 3+) to eventually replace the separate cross-encoder reranker.**

### TARG Learned Query Router (Nov 2025)

Replace the rule-based query router with TARG's prefix logit statistics for automatic retrieval gating. 70-90% retrieval reduction without training. **Implement after you have sufficient query logs (3-6 months post-launch) to validate the rule-based router first.**

---

## New Research Papers Added to Reading List

Add these to the v2 reading list:

15. **"Cutting Every Millisecond from RAG"** (Feb 2026 compilation)
    *Why:* Defines the query intelligence layer, semantic caching, and Adaptive RAG routing. Latency optimization for every pipeline stage.

16. **"RAG Latency Optimization Supplement"** (Feb 2026)
    *Why:* FLARE, IRCoT, API prompt caching, MoE models, VectorLiteRAG. Completes the latency picture.

17. **"RAG Architectures Guide — Addendum"** (Feb 2026)
    *Why:* CoRAG, PageIndex, GFM-RAG, RankRAG, DRAGIN, LongRAG, Disco-RAG. Informs future roadmap.

18. **"GenGround: Generate-Then-Ground"** (Shi et al., 2024)
    *Why:* The systematic per-claim verification pattern that upgrades your Layer 7d.

19. **"TARG: Training-free Adaptive Retrieval Gating"** (Nov 2025)
    *Why:* Future upgrade for the query router. 70-90% retrieval reduction.

20. **"FLARE: Forward-Looking Active Retrieval Augmented Generation"** (Jiang et al., EMNLP 2023)
    *Why:* Active mid-generation retrieval for complex analytical queries.
