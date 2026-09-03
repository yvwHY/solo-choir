---
name: writing-plans
description: "Use when you have a spec/requirements for a multi-step task, before touching code. Produces a bite-sized, file-by-file implementation plan. Trigger with /write-plan."
---

# Writing Plans

> Adapted from obra/superpowers for the Solo Choir project. Superpowers-internal
> execution sub-skills removed; the rigid RED-GREEN-test cycle is generalized to
> this project's verification (by ear, byte-identical diff, RTF/latency), since
> the audio engine is validated by listening, not unit tests. Plans save under
> `docs/plans/`.

## Overview

Write a comprehensive implementation plan assuming the implementer has zero
context for this codebase. Document which files to touch per task, the actual
code, and exactly how each step is verified. Bite-sized tasks. DRY, YAGNI,
frequent commits.

**Announce at start:** "I'm using the writing-plans skill to create the plan."

**Save to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Respect the project HARD RULES

Every plan in this repo inherits CLAUDE.md's hard rules — restate the relevant
ones in Global Constraints:
- Never modify the DSP — only add plumbing (telemetry tap, control input,
  stream settings, output mix).
- Engine-touching change must prove **bit-identical** output at defaults
  (max-abs-diff 0.0). Make that a verification step, not an afterthought.
- Never stage license-restricted binaries (`.bin/.pth/.onnx`, engine, models).
- Small steps, human review, commit working states before the next step.

## File Structure first

Before tasks, map which files are created/modified and each one's single
responsibility. Files that change together live together. Follow existing
patterns; don't unilaterally restructure.

## Task Right-Sizing

A task is the smallest unit that carries its own verification and is worth a
human's review gate. Fold setup/config/docs into the task whose deliverable
needs them. Each task ends with an independently checkable deliverable.

## Plan Document Header

```markdown
# [Feature Name] Implementation Plan

**Goal:** [one sentence]
**Architecture:** [2-3 sentences]
**Verification mode:** [by ear / byte-identical diff vs prior version / RTF & latency]

## Global Constraints
[Project-wide rules copied verbatim — the relevant CLAUDE.md HARD RULES,
version floors, the byte-identical non-regression requirement. Every task
implicitly includes this section.]

---
```

## Task Structure

Each task lists **Files** (exact paths, line ranges to modify), **Interfaces**
(what it consumes from earlier tasks / produces for later ones — exact
signatures), then bite-sized steps (2-5 min each). For this project a step's
verification is usually one of:

- **Listen:** run `python app/shell.py`, sing, confirm by ear ("You + 1 harmony
  still sums correctly").
- **Byte-diff:** render a fixed input through old vs new path, assert
  `max-abs-diff == 0.0` at defaults.
- **Numbers:** check RTF per pass / latency_ms in telemetry against the
  real-time ceiling (~0.3 RTF per converted part; live ≤ ~3 parts).
- **Commit:** `git add <paths> && git commit -m "..."` — and confirm no
  ignored binaries got staged (`git status`).

Show the actual code in every code step. Exact commands with expected result.

## No Placeholders

These are plan failures — never write them: "TBD", "TODO", "add appropriate
error handling", "handle edge cases", "similar to Task N" (repeat the code),
steps that say what without showing how, references to functions defined in no
task.

## Self-Review

After writing the plan, check it against the spec with fresh eyes:
1. **Coverage** — every spec requirement maps to a task; list gaps.
2. **Placeholder scan** — kill the red flags above.
3. **Name consistency** — types/signatures used in later tasks match earlier
   definitions.
Fix inline.

## Execution Handoff

After saving, summarize the plan and ask the user how to proceed — task by
task with a human checkpoint (and commit) between each, per the project's
"small steps, human review" rule.
