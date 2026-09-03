# VERIFICATION — How to Verify Audio Changes in This Repo

**Audience: AI assistants (Claude, Codex, any model).**
This repo has no unit-test culture for the audio path — and "it runs without crashing" is NOT verification here. Every audio change is verified by exactly one of three regimes. **Declare which regime your change belongs to BEFORE editing** (say it out loud in your plan); ambiguity about this is how validated DSP gets silently broken.

## Step 0 — always, before editing
- Commit the current working state first, so `git checkout` is the way back (HARD RULE 4 in CLAUDE.md).
- Search `GRAVEYARD.md` for your approach and `FINDINGS.md` for numbers you're about to re-derive.

## Regime A — behaviour-preserving change (refactor, new flag defaulting off, plumbing)
The claim is "nothing audible changed". **Prove it bit-for-bit:**
```bash
cd server
# before your change (or on the pre-change commit):
python beatrice_solo_choir_live.py --render IN.wav /tmp/before.wav   # + the same flags as the path you touched
# after your change:
python beatrice_solo_choir_live.py --render IN.wav /tmp/after.wav
python ../tools/render_diff.py /tmp/before.wav /tmp/after.wav        # must print max-abs-diff 0.0
```
`--render IN OUT` runs the exact live callback offline (flag defined at `server/beatrice_solo_choir_live.py:145`, render loop at ~`:741`). Requirement: **max-abs-diff 0.0**, not "small". Any nonzero diff means your refactor changed behaviour — find out why before proceeding. (`solo_min` has no `--render` flag as of 2026-07-06; for solo_min-path refactors, render through the harness above or add an equivalent offline entry first.)

## Regime B — intentional live-behaviour change (latency, glide, voice count, threading)
Byte-identical does NOT apply — say so explicitly. Verify instead with:
1. **Numbers:** RTF (printed by the engine; budget ≈ 0.14/converted voice, keep total < 1), underrun/overflow counts (must not grow vs baseline), measured latency if that's the claim (see FINDINGS F6 for reference points).
2. **Ears:** a live by-ear check is a human checkpoint — stop and hand it to Harry; do not self-certify audio quality. State precisely what he should listen for (e.g. "toggle Alto on/off 10× — no gaps", "sing a scale — no scrape on transitions").
3. If the symptom is intermittent, **measure, don't theorise**: record stems and inspect them (the 35 ms zero-gap discovery came from analysing a recorded stem, not from buffer theory — GRAVEYARD G5/G7).

## Regime C — offline render change (studio render, arranger, transcription)
Render the same fixed input clip before and after; listen to both; use `tools/render_diff.py` to characterise (a diff is expected — the question is whether it's the *intended* diff). For transcription changes, run the 10-take suite comparison rather than a single clip, and remember GRAVEYARD G15: residual errors may be true input ambiguity, not a bug.

## Environment gotchas that fake a failure (check before debugging "your" change)
- Two streams on one output device at **different sample rates** → `-10851`. Everything must be 48 kHz (FINDINGS F7).
- `input overflow` = two-clock drift, not your code (F7).
- Speaker playback → mic feedback / "reverb on You" = acoustic loop, not a bug. Verify on headphones or via Record→Play.
- Device-switch timbre weirdness → check control-state completeness first (GRAVEYARD G7).

## License guard (before every commit)
`git status` + `.gitignore` check: no `.bin/.pth/.onnx`, no `beatrice_engine*`, no `paraphernalia_*`, no model dirs, nothing under `recordings/`. (HARD RULE 3.)
