---
name: audio-verify
description: This skill should be used whenever editing audio-path code in this repo — anything under server/ (engine, DSP, f0, conversion, resampling, solo_min, studio_api, arranger) or the live threading/bridge — BEFORE the first edit and again before claiming completion. Also triggers on "驗證", "verify this change", "byte-identical", "regression check", "改 DSP", "改引擎". Enforces the three-regime verification doctrine.
---

# Audio-verify — declare the regime, then prove it

The full doctrine and commands live in `docs/VERIFICATION.md`. This skill is only the trigger. Read that file, then:

1. **Before editing:** commit the working state, and declare in your plan which regime the change is:
   - **A — behaviour-preserving** → must prove `--render` byte-identical (max-abs-diff **0.0**) via `tools/render_diff.py`.
   - **B — intentional live change** → byte-identical does NOT apply; verify with RTF / underrun counts / measured latency, plus a by-ear checkpoint handed to the user (never self-certify audio quality). Say explicitly that this is a Regime-B change.
   - **C — offline render change** → render the same fixed clip before/after, listen, characterise the diff as intended.
2. **Before claiming done:** show the actual evidence (the diff number, the RTF/underrun numbers, or the precise by-ear instructions you handed over). "Runs without crashing" is not verification here.
3. Check the environment-gotcha list in VERIFICATION.md before debugging a "failure" (sample-rate mismatch, two-clock overflow, speaker feedback all fake failures).
4. Run the license guard before committing (no engine binaries / models / recordings).
