# Query Intelligence Layer — Phase 0

A pre-retrieval layer that sits between the user's query and the hybrid search pipeline. Its job: decide *whether*, *how*, and *how deeply* to retrieve for each query.

This is the highest-ROI addition to the pipeline. The latency research is emphatic: "query routing and caching first, index and retrieval optimization second."

---

## Component A: Semantic Query Cache

Stores (query embedding, response) pairs. When a new query's embedding is sufficiently similar to a cached query, returns the cached response immediately — bypassing the entire retrieval and generation pipeline.

**Impact:** Uncached queries ~6,504ms vs cache hits ~53ms (123x reduction). Production deployments report 18-60% cache hit rates, yielding 40-50% average latency reduction.

**Why this works for legal queries:** Lawyers frequently ask variations of the same questions. "What is the limitation period for filing a civil suit?" and "How long do I have to file a suit under the Limitation Act?" are semantically the same query.

### Architecture

- **Query embeddings:** Stored in Qdrant `query_cache` collection (same embedding model as main search)
- **Response payloads:** Stored in Redis with TTL (fast key-value lookup)
- **Similarity threshold:** 0.92 (high, to avoid false matches between legally distinct queries)

```python
class SemanticCache:
    def __init__(self, similarity_threshold=0.92, ttl_seconds=86400):
        self.redis = redis.Redis()
        self.qdrant = QdrantClient()
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds  # 24-hour default

    def get(self, query_embedding):
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
                return {"hit": True, "response": json.loads(cached_response)}
        return {"hit": False}

    def set(self, query_text, query_embedding, response):
        cache_key = f"cache:{uuid4()}"
        self.redis.setex(cache_key, self.ttl, json.dumps(response))
        self.qdrant.upsert(
            collection_name="query_cache",
            points=[{
                "id": str(uuid4()),
                "vector": query_embedding,
                "payload": {
                    "cache_key": cache_key,
                    "original_query": query_text,
                    "acts_cited": response.get("acts_cited", []),
                    "cached_at": datetime.now().isoformat()
                }
            }]
        )

    def invalidate_for_act(self, act_name):
        """When an Act is amended, invalidate cached responses that cite it."""
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

### Cache Staleness Mitigations

Legal data has real staleness risks — an amendment makes a cached answer wrong.

1. **TTL-based expiration:** Default 24-hour TTL. Rapidly-changing areas (recent amendments, ongoing cases) get 1-hour TTL.
2. **Amendment-triggered invalidation:** When the ingestion pipeline detects an amendment (Phase 1 Change Detection Agent), invalidate all cache entries referencing the amended Act.
3. **High similarity threshold:** 0.92 prevents false matches. "What's the punishment for cheating under IPC?" and "under BNS?" are similar but have different correct answers.

**Storage:** ~80MB for 10K cached queries (768-dim embedding + ~5KB response each). Trivial.

---

## Component B: Adaptive RAG Query Router

A lightweight classifier that routes each query to the optimal retrieval path, avoiding the full pipeline for queries that don't need it.

**Impact:** Adaptive RAG reports 35% P50 latency reduction and 28% LLM API cost savings with 8% accuracy improvement.

### Four Retrieval Paths

```
User Query
    │
    ├── [Cache] → Hit? → Return cached response (~50ms)
    │
    └── [Router] → Classify complexity
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
        │   → Standard hybrid retrieval pipeline (500-800ms)
        │
        ├── COMPLEX (multi-hop retrieval)
        │   "Compare the eviction grounds under Delhi and Mumbai rent control laws"
        │   "What is the interplay between Section 498A IPC and the DV Act?"
        │   → Full hybrid retrieval + graph traversal + RAPTOR (1-2s)
        │
        └── ANALYTICAL (multi-hop + active retrieval)
            "Trace the evolution of Section 377 jurisprudence from Naz Foundation to Navtej Johar"
            "What are ALL grounds for eviction under Delhi Rent Control Act?"
            → FLARE-style active retrieval (2-5s)
```

### Rule-Based Router (v1 — Launch Implementation)

```python
def classify_complexity(query_text):
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

### Future Upgrade: TARG Learned Router

TARG (Training-free Adaptive Retrieval Gating, Nov 2025) uses prefix logit statistics to gate retrieval without training. 70-90% retrieval reduction while matching accuracy. **Implement after 3-6 months of query logs to validate rule-based router first.**

---

## Component C: Selective HyDE (Hypothetical Document Embeddings)

For complex or ambiguous queries, generates a hypothetical answer document via LLM, then uses its embedding for retrieval instead of the raw query embedding.

**Impact:** +18-29% NDCG improvement by bridging the vocabulary gap between how lawyers ask questions and how legal text is written.

**When to use:** ONLY for COMPLEX and ANALYTICAL queries routed by the classifier. Never for SIMPLE or STANDARD — the 300-400ms overhead isn't justified.

**Why HyDE matters for Indian law:** A lawyer asks "Can a company director be held personally liable for company debts?" The legal text uses "lifting the corporate veil" and "personal guarantee under Section 2(58) of Companies Act." HyDE generates a hypothetical answer using legal terminology, making the embedding closer to the actual statutory text.

```python
def maybe_rewrite_query(query_text, route):
    if route not in ["COMPLEX", "ANALYTICAL"]:
        return query_text, embed(query_text)

    prompt = f"""You are an expert Indian lawyer. Given this legal research question,
    write a brief (2-3 sentence) hypothetical answer using proper Indian legal
    terminology, Act names, and section numbers. This will be used for retrieval,
    so include the specific legal terms that would appear in the relevant statutes.

    Question: {query_text}

    Hypothetical answer:"""

    hypothetical = llm.generate(prompt, model="claude-haiku", max_tokens=200)
    hyde_embedding = embed(hypothetical)

    # Use hyde_embedding for vector search, original query for BM25 + generation
    return query_text, hyde_embedding
```

**Cost:** ~$0.001 per HyDE call (Claude Haiku, ~300 tokens). Only 10-20% of queries.

---

## End-to-End Query Flow

### Standard Query Example

```
"Can a landlord evict a tenant for personal use in Delhi?" (STANDARD)

Step 1: Embed query (20ms)
Step 2: Check semantic cache → Miss (5ms)
Step 3: Route → STANDARD (5ms)
Step 4: HyDE? → No (STANDARD queries skip)
Step 5: Matryoshka funnel search + BM25 + QuIM + graph (500-800ms)
Step 6: Rerank → top 20 → parent expand → LLM generation
Step 7: GenGround verification
Step 8: Cache response for future similar queries
```

### Complex Query Example

```
"Compare eviction grounds under Delhi and Mumbai rent control laws" (COMPLEX)

Step 1: Embed query (20ms)
Step 2: Check semantic cache → Miss (5ms)
Step 3: Route → COMPLEX (5ms)
Step 4: HyDE → Generate hypothetical answer (300ms)
Step 5: Full funnel search (using HyDE embedding) + BM25 (original query) + graph traversal (1-2s)
Step 6: Rerank → top 20 → parent expand → LLM generation
Step 7: GenGround verification
Step 8: Cache response
```

---

## Cost and Latency Summary

| Component | Latency Added | Cost per Query | Frequency |
|-----------|--------------|----------------|-----------|
| Query embedding | 20ms | ~$0.0001 | 100% |
| Semantic cache check | 5ms | $0 | 100% |
| Query router | 5ms | $0 | 100% |
| HyDE (when invoked) | 300ms | ~$0.001 | 10-20% |
| Cache hit (saves) | -1,500-2,000ms | $0 | 30-50% |

**Expected average latency improvement across all queries: 40-55%**
