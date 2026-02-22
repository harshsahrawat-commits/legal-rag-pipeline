Run this at the start of every work session. This restores context from previous sessions.

## Step 1: Load Last Session Context

Read `plans/session-log.md` and find the most recent entry. Display:
- **Last session date** and what was worked on
- **Open questions** from last session
- **Next steps** that were planned

## Step 2: Check Current State

Run these checks and report:
- `git status` — any uncommitted changes from last session?
- `git log --oneline -5` — last 5 commits for context
- Which phase plan is active? Read the latest `plans/phase-*.md` and show completion status.
- Are services running? Check if Qdrant, Neo4j, Redis are accessible (if applicable to current phase).

## Step 3: Identify Today's Work

Based on the session log and phase plan, propose what to work on today:
- List the next 2-3 uncompleted subtasks from the phase plan
- Flag any blockers or dependencies
- Suggest which docs to read before starting

## Step 4: Pre-load Context

Read the relevant docs for today's planned work (from the `docs/` directory). This ensures Claude has full context before the user gives their first real task.

Ask the user: "Here's what I suggest we work on today: [tasks]. Want to proceed with this, or do you have something else in mind?"
