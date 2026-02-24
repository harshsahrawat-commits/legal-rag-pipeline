# Evaluation Framework

## Automated Metrics (RAGAS)

Run weekly on sample queries and after every system update.

| Metric | Target | What It Measures |
|---|---|---|
| Context Recall | > 0.90 | Are all relevant chunks being retrieved? |
| Context Precision | > 0.85 | Are irrelevant chunks being filtered out? |
| Faithfulness | > 0.95 | Is the answer grounded in retrieved context? |
| Answer Relevancy | > 0.85 | Does the answer address the query? |

## Custom Legal Metrics

| Metric | Target | How Measured |
|---|---|---|
| Citation Accuracy | > 0.98 | % of cited sections/cases that exist and are correctly attributed |
| Temporal Accuracy | > 0.99 | % of referenced laws that are currently in force (unless historical query) |
| Section Completeness | > 0.90 | For "what does Section X say" queries, is the full section text retrieved? |
| Cross-reference Resolution | > 0.80 | When query requires multiple related sections, are all retrieved? |

## Latency Metrics

| Metric | Target | What It Measures |
|---|---|---|
| TTFT - Simple queries | < 200ms | Query intelligence + KG direct query speed |
| TTFT - Standard queries | < 800ms | Core retrieval pipeline speed |
| TTFT - Complex queries | < 2s | Multi-hop retrieval speed |
| TTFT - Analytical queries | < 5s | FLARE active retrieval speed |

## Query Intelligence Metrics

| Metric | Target | What It Measures |
|---|---|---|
| Cache hit rate | > 30% | Semantic cache effectiveness |
| Query routing accuracy | > 90% | Router correctly classifies query complexity |
| GenGround claim verification rate | > 95% supported | Response groundedness after verification |
| Parent context utilization | > 80% of retrievals | Parent-doc expansion working correctly |
| FLARE trigger frequency | 5-10% of queries | Active retrieval is calibrated correctly |

## Test Query Categories

Generate 200 queries across 5 practice areas x 4 query types:

**Practice areas:** Criminal, Civil/Contract, Corporate/Commercial, Tax, Constitutional

**Query types:**
1. **Factual** — "What is the punishment for cheating under BNS?"
2. **Analytical** — "Can anticipatory bail be granted in cases under Section 498A?"
3. **Cross-reference** — "How do Sections 34 and 149 IPC differ in establishing common intention?"
4. **Temporal** — "What was the bail provision under CrPC before BNSS replaced it?"

Store in `data/eval/test_queries.json` with expected answer and source citations.

## Human Evaluation Protocol

**Before launch and quarterly:**
1. Recruit 5-10 practicing Indian lawyers across practice areas
2. Each evaluates 40 queries (their domain) on 4 dimensions:
   - **Accuracy** (1-5): Is the legal information correct?
   - **Completeness** (1-5): Are all relevant provisions cited?
   - **Recency** (1-5): Is the law current?
   - **Usefulness** (1-5): Would this help build a case?
3. Target: 85%+ queries score >= 4 on accuracy
4. Compensation: Rs. 500-1000 per evaluation session

## Continuous Improvement Loop

```
After each evaluation run:
├── Identify worst-performing query categories
├── Trace failures to specific pipeline stage:
│   ├── Low context recall → chunking or embedding issue
│   ├── Low faithfulness → generation hallucination → strengthen Layer 7/8
│   ├── Low citation accuracy → KG coverage gap
│   └── Low temporal accuracy → amendment tracking gap
├── Fix the identified stage
├── Re-run evaluation to confirm improvement
└── Log the improvement in docs via /learn command
```

### Monitoring Cadence

```
Weekly:
├── Monitor cache hit rate and TTL effectiveness
├── Review query routing decisions (sample 50 queries, check classification)
├── Track FLARE trigger frequency (should be 5-10% of queries)
├── Monitor prompt cache hit rate (Anthropic API dashboard)
├── Run RAGAS on 50 sample queries
└── Check latency percentiles (p50, p95, p99) per query type

Monthly:
├── Analyze cache staleness — any cases where cached responses became incorrect?
├── Tune query router rules based on misclassification patterns
├── Review FLARE retrieval patterns — are the additional retrievals finding useful context?
├── Evaluate GenGround verification rate — are too many claims being flagged?
├── Full RAGAS evaluation on 200 queries
└── Cost analysis: actual API spend vs. projections with prompt caching

Quarterly:
├── Compare latency metrics against targets
├── Evaluate whether learned router (TARG) should replace rule-based router
├── Cost analysis: actual API spend vs. projections
├── Full human lawyer evaluation (200 queries, 5-10 lawyers)
├── Update training data for embedding model if performance has degraded
└── Review and update confidence scoring weights based on real accuracy data
```
