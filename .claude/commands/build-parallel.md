Orchestrate parallel implementation of a pipeline phase using agent teams.

This command spawns specialized subagents to work on independent subtasks simultaneously.

1. Read the current phase plan from `plans/` directory
2. Identify subtasks that can be parallelized (no dependencies between them)
3. For each independent subtask, spawn a subagent with:
   - The relevant docs (only what that subtask needs)
   - The specific deliverable (module, class, function)
   - The test requirements
   - Instructions to commit work-in-progress to a feature branch

**Agent assignment rules:**
- **Parser tasks** → give agent `docs/parsing_guide.md` + `docs/indian_legal_structure.md`
- **Chunker tasks** → give agent `docs/chunking_strategies.md` + `docs/metadata_schema.md`
- **Enrichment tasks** → give agent `docs/enrichment_guide.md` + `docs/metadata_schema.md`
- **KG tasks** → give agent `docs/knowledge_graph_schema.md`
- **Hallucination tasks** → give agent `docs/hallucination_mitigation.md`
- **Test tasks** → give agent the module being tested + the relevant doc

4. After all agents complete, review their outputs:
   - Run `ruff check` on all new code
   - Run `pytest` on all new tests
   - Check for interface compatibility between modules
   - Merge into phase branch

5. Update the phase plan with completion status.

**Example usage:** "Build Phase 3 (chunking) in parallel — the statute chunker, judgment chunker, semantic chunker, and RAPTOR builder can all be built independently."
