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

## Layer 4: Grounded Refinement (Post-Generation)

After LLM generates response, run a SECOND LLM pass that acts as a legal auditor:

```python
async def grounded_refinement(response: str, chunks: list[LegalChunk]) -> str:
    audit_prompt = f"""You are a legal accuracy auditor. For each claim:
    1. Is it directly supported by the source documents? 
    2. Is any source mischaracterized?
    3. Are important caveats omitted?
    
    Remove ungrounded claims. Correct mischaracterizations. Add omitted caveats.
    
    Response: {response}
    Sources: {format_chunks(chunks)}"""
    
    return await llm.generate(audit_prompt)
```

**Cost consideration:** This doubles the LLM cost per query. Only enable for production responses, not during development/testing.

## Layer 5: Finetune-RAG (Phase 2 — After Launch)

Fine-tune the generation model itself to resist hallucination from noisy retrieval. Based on arXiv:2505.10792 (21.2% accuracy improvement).

**Training data needed:**
- Real Indian legal queries + correct retrieved chunks + some deliberately wrong chunks mixed in
- Labels: which claims are grounded, which are hallucinated
- ~10K examples minimum

**This requires collecting real usage data first.** Build the dataset from production queries where human lawyers flag errors. Implement feedback loop from day one.
