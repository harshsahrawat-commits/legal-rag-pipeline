# Hallucination Mitigation — 5-Layer Defense System

Stanford's 2025 study found even the best legal RAG tools hallucinate 17-33% of the time. This layer is what makes our system trustworthy.

## Layer 1: Citation Verification (Pre-Response)

Before ANY response reaches the user, extract all citations and verify against the knowledge graph.

```python
# Implementation pattern
async def verify_citations(response: str, kg: Neo4jClient) -> VerificationResult:
    citations = extract_all_citations(response)  # regex + NER

    for citation in citations:
        # Check existence
        exists = await kg.node_exists(citation)
        if not exists:
            flag(citation, "NOT_FOUND")
            continue

        # Check attribution accuracy
        source_text = await kg.get_source_text(citation)
        claim = extract_claim_for(response, citation)
        if not llm_verify_attribution(claim, source_text):
            flag(citation, "MISATTRIBUTION")
```

**Citation patterns to extract:**
- Section references: `Section 420 IPC`, `Section 13(1)(ia) HMA`, `Article 21`
- Case citations: `AIR 2023 SC 1234`, `(2023) 5 SCC 678`, `2023 SCC OnLine SC 890`
- Notification/circular numbers: `GSR 1234(E)`, `RBI/2023-24/45`

**If verification fails:** Remove the citation, add disclaimer, log for review.

## Layer 2: Temporal Consistency Check

India replaced three major criminal codes effective July 1, 2024:
- IPC → Bharatiya Nyaya Sanhita (BNS) 2023
- CrPC → Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
- Indian Evidence Act → Bharatiya Sakshya Adhiniyam (BSA) 2023

**The system MUST catch when an LLM cites repealed law.** This is worse than no answer.

```python
async def check_temporal(response: str, query_date: date | None = None):
    ref_date = query_date or date.today()

    for section_ref in extract_section_refs(response):
        status = await kg.temporal_status(section_ref, ref_date)

        if status.is_repealed:
            inject_warning(
                f"{section_ref} was repealed by {status.repealed_by} on {status.date_repealed}. "
                f"The current provision is {status.replacement}."
            )
```

**Also check for:** Amendments that changed section text, notifications that changed effective dates, ordinances that expired.

## Layer 3: Confidence Scoring

Every response gets a confidence score displayed to the user.

**Factors and weights:**
| Factor | Weight | How Measured |
|---|---|---|
| Retrieval relevance | 0.25 | Average cosine similarity of top-k chunks |
| Citation verification rate | 0.20 | verified_citations / total_citations |
| Source authority | 0.20 | Highest court level in sources (SC=1.0, HC=0.7, District=0.4) |
| Chunk agreement | 0.15 | Do top chunks agree or contradict? |
| Source recency | 0.10 | How recent are the cited sources? |
| Query specificity | 0.10 | Is the query specific enough for confident answer? |

**Display:** "Confidence: High (0.87) — Based on 3 Supreme Court judgments and current statutory text."

**If confidence < 0.6:** Prepend warning: "This response has lower confidence. Please verify independently."

## Layer 4: GenGround Systematic Verification (Post-Generation)

Upgraded from single-pass grounded refinement to **per-claim re-retrieval and verification** using the GenGround pattern (Shi et al., 2024). This catches subtle misattributions that a single-pass check misses.

**Tiered application:**
- SIMPLE queries (direct lookups): Basic single-pass grounded refinement (see below)
- STANDARD, COMPLEX, ANALYTICAL queries: Full GenGround verification

### Full GenGround Flow

```python
async def genground_verification(response: str, retrieved_chunks: list[LegalChunk], kg: Neo4jClient) -> VerifiedResponse:
    """Systematic per-claim verification with independent re-retrieval."""

    # Step 1: Extract individual claims from the response
    claims = extract_claims(response)
    # Example: "Section 138 NI Act requires the cheque to be presented within 6 months"
    # Example: "In Dashrath Rupsingh Rathod v. State (2014), the SC held that..."

    verification_results = []

    for claim in claims:
        # Step 2: Independently re-retrieve evidence for this specific claim
        # Key insight: don't just check against original chunks — re-retrieve
        claim_evidence = hybrid_search(
            query=claim.text,
            top_k=5,
            metadata_filter=claim.get_relevant_filters()  # e.g., specific Act
        )

        # Step 3: Also check knowledge graph for citation accuracy
        kg_verification = None
        if claim.contains_citation:
            kg_verification = await kg.verify_citation(
                citation=claim.citation,
                attributed_content=claim.attributed_content
            )

        # Step 4: Score alignment between claim and evidence
        alignment = await llm.check_alignment(
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
            "issues": alignment.issues
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

### Basic Single-Pass (For SIMPLE Queries Only)

```python
async def basic_grounded_refinement(response: str, chunks: list[LegalChunk]) -> str:
    audit_prompt = f"""You are a legal accuracy auditor. For each claim:
    1. Is it directly supported by the source documents?
    2. Is any source mischaracterized?
    3. Are important caveats omitted?

    Remove ungrounded claims. Correct mischaracterizations. Add omitted caveats.

    Response: {response}
    Sources: {format_chunks(chunks)}"""

    return await llm.generate(audit_prompt)
```

### Cost Comparison

| | Basic (SIMPLE) | GenGround (STANDARD+) |
|---|---|---|
| LLM calls per response | 1 | 1 + N (N = claims, typically 5-15) |
| Retrieval calls | 0 | N (one per claim) |
| Cost per query | ~$0.005 | ~$0.02-0.05 |
| Latency added | ~500ms | ~1-2s |
| Accuracy | Good (obvious issues) | Better (subtle misattributions) |

**Prompt caching:** Both the system prompt for GenGround alignment checks and the basic audit prompt should use Anthropic's `cache_control` parameter to cache the static instruction prefix. See `docs/enrichment_guide.md`.

## Layer 5: Finetune-RAG (Phase 3+)

Fine-tune the generation model itself to resist hallucination from noisy retrieval. Based on arXiv:2505.10792 (21.2% accuracy improvement).

### Synthetic Training Data — 5 Categories

The LIMA finding: "1,000 high-quality examples of hallucination resistance are more valuable than 50,000 generic legal Q&A pairs."

```python
def create_finetune_rag_dataset(real_queries, chunk_database):
    """Create training examples that teach the model to resist misleading context."""
    training_examples = []

    for query in real_queries:
        correct_chunks = retrieve(query)
        correct_answer = generate_verified_answer(query, correct_chunks)

        # Poisoned version: deliberately wrong chunks mixed in
        wrong_chunks = get_similar_but_wrong_chunks(query, chunk_database)
        mixed_chunks = interleave(correct_chunks[:3], wrong_chunks[:2])

        # Model should still give correct answer even with wrong chunks
        training_examples.append({
            "system": "Answer based ONLY on provided context. "
                      "If context is contradictory, prefer the most authoritative source. "
                      "If unsure, say 'I cannot determine this from the provided sources.'",
            "context": format_chunks(mixed_chunks),
            "query": query,
            "expected_response": correct_answer
        })

        # Also create "insufficient context" examples
        training_examples.append({
            "context": format_chunks(wrong_chunks),
            "query": query,
            "expected_response": "Based on the provided sources, I cannot reliably answer "
                                 "this question. The available context does not contain "
                                 "sufficient information about [specific gap]."
        })

    return training_examples
```

**Dataset composition (1,000 examples):**
| Category | Count | Purpose |
|----------|-------|---------|
| Deliberately misleading context | 200 | Model should resist and answer correctly |
| Insufficient context | 200 | Model should say "I don't know" |
| Correct context | 200 | Model should answer normally (baseline) |
| Temporally outdated context | 200 | Model should flag currency issues |
| Misattributed citations | 200 | Model should detect and correct |

### DPO for Preference Learning

After initial Finetune-RAG training, use DPO with pairs:
- **Chosen:** Response grounded in correct sources, acknowledging uncertainty where appropriate
- **Rejected:** Response that confidently cites a wrong source or fabricates a citation

This teaches the model the *preference* for caution over confident hallucination.

### Timeline

Phase 3+ activity (Weeks 19-20 in roadmap). Requires collecting production query logs and building the training dataset from real usage patterns where lawyers flag errors. Implement feedback loop from day one.
