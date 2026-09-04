# Instructions for AI agents (Codex, Cursor, and any non-Claude tool)

**Read `CLAUDE.md` in this directory first — it is the full instruction set for this repo** (despite the filename, it is tool-agnostic: hard rules, architecture, paths, data contracts). This file only adds the routing table below, which any model must follow.

## Read-before-acting routing (`docs/`)

| Situation | Read FIRST |
|---|---|
| How the project got here, phase by phase | `PROCESS.md` (the rolling state file was a working note and stayed in the development repository) |
| About to try a new approach (streaming, pitch, ensemble, timbre, transcription, hardware) | `GRAVEYARD.md` — 19 dead ends with root causes; do not re-run them |
| Need validated numbers (glide rates, latency, decorrelation, RTF, wiring) | `FINDINGS.md` |
| About to edit audio-path code (`server/`, live threading, bridges) | `VERIFICATION.md` — declare Regime A/B/C before editing; A must prove max-abs-diff 0.0 via `tools/render_diff.py` |
| A live-audio symptom (choppy, scrape, overflow, -10851, timbre collapse…) | `DEBUG_PLAYBOOK.md` — most symptoms already have a diagnosed root cause |
| Engine internals / hardware / UI spec / training / multi-voice | routing table in `CLAUDE.md` |

## Non-negotiables (short form — full text in CLAUDE.md)

1. Audio/DSP code in `server/` is validated and sensitive: understand it before editing, change the minimum, commit a working state first so `git checkout` reverts.
2. Never commit license-restricted binaries: no `.bin/.pth/.onnx`, no `beatrice_engine*`, no `paraphernalia_*`, no model dirs, nothing in `recordings/`.
3. Small steps with human by-ear checkpoints. Never self-certify audio quality — hand precise listening instructions to the user.
4. When an approach fails, register it in `GRAVEYARD.md` before moving on.
