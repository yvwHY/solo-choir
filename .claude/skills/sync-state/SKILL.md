---
name: sync-state
description: This skill should be used after every commit checkpoint in this repo, at the end of any working session, and whenever the user says "同步狀態", "更新現況", "sync state", or "收工". Keeps docs/STATE.md (the single source of truth for project state) from rotting — CLAUDE.md deliberately carries no state.
---

# Sync-state — state lives in STATE.md, and only there

`docs/STATE.md` is the single source of truth for "where the project is". It rots unless maintained; a stale state file actively misleads every future session (this happened: CLAUDE.md claimed "2 voices, plan not executed" while the repo had a 5-part live choir).

After a commit checkpoint or at session end:

1. **STATE.md** — update the affected lines only: status-per-track, next-steps order, the "last updated" date, and the ahead-of-origin note if it changed. Cite new commit hashes. Do not rewrite unaffected sections.
2. **Worklog** — for a substantive session, append/create `docs/worklogs/YYYY-MM-DD.md` (what was done, how it was verified, next step). Follow the existing worklogs' shape.
3. **GRAVEYARD.md / FINDINGS.md** — if the session killed an approach or validated a number, register it (see the `graveyard` skill).
4. **Never** re-add a "Current state" section to CLAUDE.md; it points at STATE.md by design.
5. If `ui-app` is far ahead of origin, remind the user to push (do not push unasked).
