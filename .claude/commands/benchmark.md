Run the RAGAS evaluation benchmark against the current pipeline state.

1. Read `docs/evaluation_framework.md` for metric targets
2. Load test queries from `data/eval/test_queries.json`
   - If file doesn't exist, generate 20 starter queries covering:
     - 5 factual (e.g., "What is the punishment for cheating under IPC?")
     - 5 analytical (e.g., "Can anticipatory bail be granted for non-bailable offences?")
     - 5 cross-reference (e.g., "How does Section 34 IPC interact with Section 302?")
     - 5 temporal (e.g., "What is the current provision replacing Section 420 IPC?")
3. For each query, run the full retrieval + generation pipeline
4. Compute RAGAS metrics: context_recall, context_precision, faithfulness, answer_relevancy
5. Compute custom legal metrics: citation_accuracy, temporal_accuracy
6. Compare against targets from the evaluation framework doc
7. Print summary table with pass/fail per metric
8. Save detailed results to `data/reports/benchmark_{timestamp}.json`

Flag any metric below target threshold in red. Suggest specific pipeline improvements for failed metrics.
