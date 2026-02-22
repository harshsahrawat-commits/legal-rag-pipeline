You are starting a new implementation phase for the Legal RAG Ingestion Pipeline.

Before writing ANY code:

1. Read `CLAUDE.md` for project context
2. Read `docs/pipeline_architecture.md` for the full 8-phase design
3. Identify which specific phase docs are relevant (chunking, enrichment, KG, etc.) and read them
4. Read `docs/metadata_schema.md` — all data flows through `LegalChunk` Pydantic models

Then create an implementation plan:

1. **Scope:** What exactly will be built in this phase? List every module, class, and function.
2. **Dependencies:** What must exist before this phase works? (previous phases, external services, config)
3. **Data flow:** What comes in (format, source)? What goes out (format, destination)?
4. **Test plan:** What unit tests and integration tests will verify correctness?
5. **Acceptance criteria:** How do we know this phase is done? Be specific and measurable.
6. **Estimated subtasks:** Break into 3-8 subtasks, each completable in one session.

Save the plan to `plans/phase-{N}-{name}.md`. Commit it before starting implementation.

Ask the user which phase they want to plan. Reference the phase list from `docs/pipeline_architecture.md`.
