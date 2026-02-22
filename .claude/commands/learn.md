Capture a lesson learned from the current session and add it to the appropriate doc.

This is the self-improvement loop. After every significant debugging session or discovery, run this command to persist the knowledge.

1. Ask the user: "What did we learn?" (or use the description provided as argument)
2. Categorize the lesson:
   - **Parsing gotcha** → append to `docs/parsing_guide.md` under "## Gotchas"
   - **Chunking issue** → append to `docs/chunking_strategies.md` under "## Gotchas"
   - **Enrichment finding** → append to `docs/enrichment_guide.md` under "## Gotchas"
   - **KG pattern** → append to `docs/knowledge_graph_schema.md` under "## Gotchas"
   - **Indian law nuance** → append to `docs/indian_legal_structure.md`
   - **General convention** → if universally applicable, suggest adding to root `CLAUDE.md`
3. Format the lesson as:
   ```
   ### [Brief Title] (YYYY-MM-DD)
   **Problem:** What went wrong or was unexpected
   **Solution:** What fixed it or the correct approach
   **Example:** Code snippet or document example if applicable
   ```
4. Append to the relevant file
5. Commit with message: `docs: learned - {brief title}`

This ensures we never hit the same bug twice across sessions.
