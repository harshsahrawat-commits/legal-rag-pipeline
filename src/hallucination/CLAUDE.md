# Hallucination Mitigation Module

Read `docs/hallucination_mitigation.md` before making changes here. This module is the most critical differentiator of the entire system.

## Module Structure
- `citation_verifier.py` — Extracts citations from LLM output, verifies against Neo4j KG
- `temporal_checker.py` — Verifies all referenced laws are currently in force
- `confidence_scorer.py` — Computes weighted confidence score per response
- `grounded_refiner.py` — Post-generation LLM pass that audits and corrects claims
- `pipeline.py` — Orchestrates all 4 layers in sequence

## Critical Rule
**This module MUST NOT silently pass through unverified citations.** If verification fails, the citation is removed and a disclaimer is added. Lawyers can be sanctioned for citing non-existent law — our system must never put them in that position.

## The IPC→BNS Problem
Always test with queries that reference both old (IPC) and new (BNS) provisions. The temporal checker must catch when a response cites "Section 420 IPC" without noting it was repealed on July 1, 2024.
