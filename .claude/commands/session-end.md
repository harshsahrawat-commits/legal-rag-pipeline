Run this at the end of every work session. This is the persistent knowledge loop.

## Step 1: Session Summary

Review everything done in this session (git diff, files changed, conversations had). Create a session log entry:

```markdown
## Session: YYYY-MM-DD HH:MM
**Phase:** [which pipeline phase was worked on]
**What was built:** [modules, functions, tests created]
**What broke:** [bugs encountered and how they were fixed]
**Decisions made:** [architectural or technical choices with reasoning]
**Open questions:** [unresolved issues for next session]
**Next steps:** [what to do when starting the next session]
```

Save to `plans/session-log.md` (append, don't overwrite).

## Step 2: Extract Learnings

From the session, identify any:
- **Gotchas** — things that failed unexpectedly → append to relevant `docs/*.md` under `## Gotchas`
- **New patterns** — reusable approaches that worked well → append to relevant doc
- **Convention changes** — if we decided to do something differently going forward → update the relevant doc or CLAUDE.md

For each learning, use the same format as /learn:
```
### [Brief Title] (YYYY-MM-DD)
**Problem:** ...
**Solution:** ...
**Example:** (if applicable)
```

## Step 3: Update Implementation Plan

If a phase plan exists in `plans/phase-*.md`:
- Mark completed subtasks as done
- Update time estimates based on actual progress
- Add any new subtasks discovered during the session

## Step 4: Check CLAUDE.md Freshness

Review root `CLAUDE.md` against current reality:
- Are the bash commands still accurate?
- Is the project structure description still correct?
- Are there any new universal conventions that should be added?

If changes are needed, propose them (don't auto-edit — the user should approve CLAUDE.md changes).

## Step 5: Commit

Stage and commit all documentation updates with message: `docs: session log YYYY-MM-DD — [one-line summary]`
