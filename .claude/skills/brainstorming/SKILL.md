---
name: brainstorming
description: "Use before any creative/implementation work — new features, components, behavior changes. Explores intent, requirements and design through one-question-at-a-time dialogue, ending in an approved design doc. Trigger with /brainstorm."
---

# Brainstorming Ideas Into Designs

Turn ideas into fully formed designs through natural collaborative dialogue, ending in a written, user-approved design doc.

> Adapted from obra/superpowers for the Solo Choir project. Superpowers-internal
> sub-skill references removed; verification framing matches this project's
> reality (verify by ear + byte-identical diff, not unit tests). Spec docs save
> under `docs/specs/`.

<HARD-GATE>
Do NOT write code, scaffold, or take any implementation action until you have
presented a design and the user has approved it. This applies to EVERY change
regardless of perceived simplicity.
</HARD-GATE>

## Anti-Pattern: "Too Simple To Need A Design"

Every change goes through this. "Simple" changes are where unexamined
assumptions cause the most wasted work. The design can be short (a few
sentences), but you MUST present it and get approval.

## Checklist (do in order)

1. **Explore project context** — read relevant files, the matching doc in
   `docs/`, recent commits. Respect the CLAUDE.md HARD RULES
   (never modify DSP; prove non-regression byte-for-byte).
2. **Ask clarifying questions — one at a time.** Wait for the answer before the
   next. Prefer multiple-choice. Focus on purpose, constraints, success
   criteria. If a question can be answered by reading the codebase, read it
   instead of asking.
3. **Propose 2-3 approaches** with trade-offs. Lead with your recommendation
   and why.
4. **Present the design in sections** scaled to complexity. Ask after each
   section whether it looks right. Cover: architecture, components, data flow
   (telemetry vs audio path), how it will be verified (by ear / byte-diff /
   RTF), and the real-time cost (each converted part = +1 inference pass).
5. **Write the design doc** to `docs/specs/YYYY-MM-DD-<topic>-design.md`.
6. **Spec self-review** — scan for placeholders (TBD/TODO), internal
   contradictions, scope creep, and any requirement open to two readings (pin
   one down). Fix inline.
7. **User reviews the written spec** — ask them to read it before proceeding.
8. **Transition to planning** — invoke the `writing-plans` skill. That is the
   ONLY next step; do not jump to implementation.

## Key Principles

- **One question at a time** — don't overwhelm.
- **YAGNI ruthlessly** — strip unrequested features from every design (pairs
  with the `ponytail` skill).
- **Explore alternatives** — always 2-3 approaches before settling.
- **Incremental validation** — approval after each section.
- **Stay surgical** — in this codebase, only add plumbing; never propose
  touching f0 estimation, conversion, resampling, or the diatonic maths.
- **Be flexible** — go back and clarify when something doesn't fit.

The terminal state is invoking `writing-plans`. Nothing else.
