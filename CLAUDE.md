# Solo Choir — Instructions for Claude Code

A real-time instrument that turns one singer into a choir: live voice → neural re-synthesis + diatonic harmony, low latency, on CPU. MA Computational Arts thesis (Goldsmiths).

## Reference docs — read on demand (not loaded by default)
When a task touches one of these areas, read the matching doc FIRST:
- **How the project got here → `docs/PROCESS.md`** (the rolling state file was a working note and stayed in the development repository)
- **About to run the show / need the frozen launch config, the pre-show self-check, or the gate fallback → `docs/VIVA_RUNBOOK.md`** (the one page that goes on stage; on conflict it and `git log` beat STATE)
- **About to try a new approach (streaming/pitch/ensemble/timbre/transcription/hardware) → `docs/GRAVEYARD.md`** (dead ends with root causes — never re-run them)
- **Validated numbers & recipes (glide, latency, decorrelation, RTF, wiring) → `docs/FINDINGS.md`**
- **About to edit audio-path code → `docs/VERIFICATION.md`** (declare Regime A/B/C first; A = max-abs-diff 0.0 via `tools/render_diff.py`)
- **Live-audio symptom (choppy/scrape/overflow/-10851/timbre collapse…) → `docs/DEBUG_PLAYBOOK.md`** (most symptoms already diagnosed)
- Engine internals / full architecture / data flow / design decisions / AI-assisted process → `docs/SOLO_CHOIR_TECH_DOC.md`
- Hardware / wearable (bone-conduction, haptic motors, intraoral, components, firmware) → `docs/HARDWARE_ROADMAP.md`
- UI build spec / bridge schema → `docs/UI_BUILD_SPEC.md`
- Voice-model training (collecting voices, step experiments, embedding-vs-model integration) → `docs/TRAINING_NOTES.md`
- **Whose voice is in the output / consent & corpus licences for the shipping models → `docs/VOICE_CREDITS.md`** (one non-corpus human voice is in the shipping Respond path — check before publishing, crediting, or changing part casting)
- Multi-voice / SATB status, what's live vs dormant, how to wire in the female model → `docs/MULTIVOICE_STATUS.md`

## HARD RULES — do not violate
1. **The core engine/DSP is editable — change it deliberately (updated 2026-06-21).** Modifying `server/` — incl. f0 estimation, conversion, resampling, the diatonic maths in `beatrice_converter.py` / `SoloChoir.py`, and the live audio threading — is ALLOWED when it serves a real goal (latency, quality, multi-voice). It is still validated, sensitive code: **read and understand it first, change the minimum needed, keep a way back** (commit a working state before, so a `git checkout` reverts). Prefer additive/guarded changes (a new flag or code path) over rewriting a validated path in place. (Prior policy was "never modify the DSP"; relaxed because the latency/quality work — e.g. Route A moving inference off the audio thread — requires it.)
2. **Use byte-identical as a regression guard, not a ban.** Where a path is *meant* to stay unchanged (the offline `--render` at defaults, or a refactor that should be behaviour-preserving), prove it bit-for-bit (render the same clip old vs new → max-abs-diff 0.0). Where a change is *meant* to alter live behaviour (Route A changes live timing on purpose), byte-identical does NOT apply — verify by ear / latency / underrun counts instead. Always know which kind of change you are making and say so.
3. **Never commit license-restricted binaries.** The Beatrice engine binary and the trained tenor model stay git-ignored and live outside the repo. Before committing, check `.gitignore` and `git ls-files` — no `.bin/.pth/.onnx`, no `beatrice_engine*`, no `paraphernalia_*`, no model dirs. (The head mesh `app/assets/head.glb` is also git-ignored since 2026-09-07: Sketchfab Standard licence — see `app/assets/CREDITS.md`.)
4. **Small steps, human review, commit checkpoints.** Do one bounded step, then stop for the human to verify by ear/eye. Commit working states before starting the next step so anything is reversible.

## Architecture (one paragraph)
Three layers with the audio path separated from control/telemetry. The **engine** runs as a subprocess and owns all live audio (mic → Beatrice conversion → diatonic harmony → output). The **bridge** (pywebview js_api) carries only telemetry (engine stdout → UI) and control (UI → engine stdin) — **audio never goes through the bridge**. The **UI** is a native macOS window via a pywebview shell. There are now **three apps** sharing this pattern (launch commands in `docs/VIVA_RUNBOOK.md`): the original live app (`app/shell.py`), the current 5-part live app **solo_min** (`app/solo_shell.py`), and the offline arrange **studio** (`studio/shell.py`).

## Key paths
- Live engines: `server/beatrice_solo_choir_live.py` (original harness; has `--render`) · `server/solo_min.py` (current 5-part live)
- Engine core (validated; edit deliberately — see HARD RULE 1): `server/beatrice_converter.py`, `server/voice_changer/SoloChoir.py`
- Original app: `app/shell.py` · `app/bridge.py` · `ui/solo_choir_ui_v10.html`
- solo_min app: `app/solo_shell.py` · `ui/solo_min.html`
- Studio app: `studio/shell.py` · `studio/studio_bridge.py` · `studio/ui/index.html` · engine API `server/studio_api.py` · arranger `server/arranger.py`
- Docs: `docs/` — see **Reference docs** above for which to read when

## Run / git
- Run: `$SOLO_CHOIR_PY_APP app/respond_shell.py` — the performing app. `app/shell.py` is the earlier single-voice app.
- Every path outside this repository is resolved in `config.py` from environment variables. Never hard-code an absolute path.
- This is the standalone submission repository, distilled from the working fork `yvwHY/solo-choir-vcclient` (branch `ui-app`, upstream `w-okada/voice-changer`). The full development history lives there.

## Data contracts
- Telemetry (engine→UI): `f0, midi, cents, level, rtf, latency_ms, voiced`
- Control (UI→engine): `convert, key, scale, intervals, harmonize, pitch, gain, gate, you`

## Current state & next steps
**This file deliberately carries no state — it rots.** Read `docs/PROCESS.md` (how the project got here, phase by phase: the three apps, what was tried, standing user preferences). Keep it fresh via the `sync-state` skill after every commit checkpoint. If STATE.md and `git log` disagree, trust `git log` and fix STATE.md.

Two standing display-honesty rules (behaviour, not state): the pitch slider shows the *applied* offset (0 means 0; never fake it), and per-voice octave/interval offsets are shown truthfully — the tenor's low character is the model's natural range, not an applied shift.

## Known gotchas
- **Audio symptoms** (input overflow, feedback, choppiness, scrape, -10851, timbre collapse…): all diagnosed symptoms live in `docs/DEBUG_PLAYBOOK.md` — look there BEFORE theorising about buffers.
- **Multi-voice cost**: each *converted* harmony = +1 inference pass, RTF ≈ 0.14/voice → live ceiling ~3–4 parts on CPU; fuller/fancier SATB goes through offline render (numbers in `FINDINGS.md` F6).
- **Terminal**: don't paste Chinese-comment lines into zsh (interactive comments are off → it tries to run `#`).

## Working style (general behaviour)
Distilled from Andrej Karpathy's notes on LLM coding pitfalls. The HARD RULES above always win on conflict; for trivial tasks use judgement.
1. **Think before coding.** Don't assume — state assumptions explicitly and ask when uncertain. If multiple interpretations exist, present them; don't pick silently. If a simpler approach exists, say so. If something is unclear, stop and name what's confusing.
2. **Simplicity first.** Minimum code that solves the problem, nothing speculative — no features beyond what was asked, no abstractions for single-use code, no unrequested "flexibility". If 200 lines could be 50, rewrite. (The `ponytail` skill enforces this; `/ponytail-review` audits a diff for over-engineering.)
3. **Surgical changes.** Touch only what you must — don't "improve" adjacent code, comments, or formatting; don't refactor what isn't broken; match existing style. Remove only the orphans *your* change created; note pre-existing dead code, don't delete it unasked. Every changed line should trace to the request. (This is the additions-only spirit of HARD RULE 2.)
4. **Goal-driven execution.** Turn a task into a verifiable goal and loop until it's met. In this repo "verify" means by ear, byte-identical diff (max-abs-diff 0.0), or RTF/latency numbers — not unit tests. State a brief plan with a per-step check for multi-step work.

## Skills installed (`.claude/skills/`)
Project-discipline skills (thin triggers; the knowledge bodies live in `docs/` so non-Claude tools inherit them too — see `AGENTS.md`):
- `graveyard` — BEFORE trying a new approach: check the dead-ends registry; AFTER an approach dies: register it.
- `audio-verify` — editing audio-path code: declare Regime A/B/C and prove it (`VERIFICATION.md`).
- `live-audio-debug` — any live-audio symptom: look up the diagnosed table first (`DEBUG_PLAYBOOK.md`).
- `sync-state` — after commit checkpoints / session end: update the state file and the worklog (both kept in the development repository).

General workflow skills:
- `/grill-me` (→ `grilling`) — relentless one-question-at-a-time interview to stress-test a plan before building.
- `/brainstorm` (`brainstorming`) — idea → approved design doc in `docs/specs/`; gate before any implementation. Adapted from obra/superpowers.
- `/write-plan` (`writing-plans`) — spec → bite-sized file-by-file plan in `docs/plans/`. Adapted from obra/superpowers.
- `ponytail` / `ponytail-review` — lazy-senior-dev minimalism + diff over-engineering audit (MIT, DietrichGebert/ponytail). Toggle off with "stop ponytail" / "normal mode".
