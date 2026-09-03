# Multi-voice / SATB — current state & how to add the female model

_Last updated 2026-06-17. Companion to `TRAINING_NOTES.md` (the voice-asset track)._

## 1. What's live right now (the stable demo build)

**Single voice: You + Bass.** One inference pass → clean, ~47 ms, no crackle/echo.

- `app/bridge.py`: `self.satb = False` → the engine runs the **legacy single-voice path**
  (no `--voices`), proven **byte-identical** to the original validated engine (max-abs-diff 0.0).
- Default harmony = **Bass, an octave below** (`intervals = [-7]`). Toggling a head in the UI just
  picks which single interval sounds (Tenor = −2 third below, etc.). Only one harmony plays at a time.
- Device: an **Aggregate Device** is auto-picked for in+out (single clock → no `input overflow`).
- Everything else built this sprint is intact: **Record→Play** (mono mix + stereo stems + "Saved ✓"/
  Reveal), **live score** (real sung pitch), **RTF pill**, device hot-plug.

## 2. What's built but DORMANT (kept in the code, switched off)

All of this works in isolation / offline; it's off live because **stacking >1 inference pass adds
cumulative model noise on this rig** (and balancing many voices is subtle). Re-enable when the female
model + a thicker-stack strategy are ready.

- **Multi-voice SATB engine** — `server/beatrice_solo_choir_live.py`: a `VOICES` list, each part its
  own pass with `model / speaker / octave / interval / formant / on`. Off voices are skipped (RTF
  scales with active count). Enabled by launching with `--voices '<json>'`.
- **`--render IN OUT`** — offline file render through the SAME callback. Use it for full SATB without
  the real-time limit, and for the byte-identical regression.
- **Per-voice config** in `app/bridge.py` `SATB_VOICES` (+ UI heads carry the descriptors).
- **Per-head volume faders** and **per-voice `gain`/`you_gain`** existed (commit `81cd2a1a`) and were
  reverted (`18b15656`) — the engine mix-gain plumbing is easy to bring back from git if wanted.
- **JVS speaker probe** — `server/probe_jvs_speakers.py` (renders a clip through JVS base speakers).
- **In-app "Render SATB" action** — `app/bridge.py` `render_satb(stems_path)` (async, off-thread):
  for each *active* live voice (`self.control["voices"]` — the on-screen JVS choir + female Alto;
  switched from the `SATB_VOICES` constant 2026-06-22 so the bounce matches what's heard) it does a
  solo `--render --no-you` pass → a pure per-part stem (`<take>_satb_<part>.wav`), then one full
  `--render` (active voices + You) → a mix (`<take>_satb.wav`); the dry melody stem is
  `<take>_satb_you.wav`. All in `recordings/`. Streams `window.onRenderStatus` (rendering/done/failed).
  Carries the take's key/scale (`--key/--minor` from `self.control`) into each render so the diatonic
  harmony matches the key sung — takes store **no key metadata**, so control is the only source (same
  assumption the live engine uses; correct as long as the UI key wasn't changed since recording).
  Per-part stems are the DAW deliverable and the S/A swap seam for the female model. **UI button +
  3-state status are wired** (Task 3 done 2026-06-22): a "Render SATB" button appears after a take,
  with Reveal SATB on done.
  - **By-ear (take_20260619_143943, C major, current tenor model):** You and **Tenor** stems clean;
    the full **You+Bass+Tenor mix is coherent** (the scale comes through). The **Bass solo stem is
    rough** — octave-down (interval −7) sits below the tenor model's clean range — but it's largely
    **masked in the mix**, so it's a solo-stem artefact, not a mix-breaker. Fixing the always-C render
    (now keyed) is what made the mix read as a real C-major scale.
  - **Decision (2026-06-19) — Bass STAYS in the default mix** (resolves the earlier keep-vs-off
    question). The placeholder bass (tenor model, octave-down) adds audible low-end body and the
    choral character that *demonstrates the Solo Choir effect* — choral character > per-voice fidelity
    for now, and the mild muddiness is an acceptable placeholder cost. Decided via the A/B
    (`_satb.wav` = You+Bass+Tenor vs `_satb_youtenor.wav` = You+Tenor, rendered for
    take_20260619_143943). **Known limitation:** tenor-model-as-bass is out of range → rough solo
    stem, masked in the mix. **Upgrade path:** a dedicated bass model. (Default voices already
    You+Bass+Tenor → no behaviour change.) A separate **Bass-as-its-own-embodied-channel** routing
    (bone-conduction/haptic) remains a future option, not a blocker.
- **In-app "Export MIDI" action** (2026-06-22) — `app/bridge.py` `export_midi(stems_path)` (async,
  off-thread) → `server/score_export.py`. The *symbolic* counterpart to the audio render: it writes a
  multi-track `<take>_satb.mid` (You + each active live voice) — notes you can read, hand to a real
  choir, re-instrument, or edit. `librosa.pyin` detects the sung melody from `<take>_satb_you.wav`
  (extracts L from `_stems` if no prior render); each frame is **snapped to the key's scale** before
  segmentation (so a sung slide's chromatic passing tones collapse into their in-key neighbours and the
  whole score stays in-key); the SATB parts are `SoloChoir.diatonic_target_midi(...) + 12·octave`
  (`SoloChoir.py` reused unchanged). **No Beatrice inference / no DSP** — pitch detection + arithmetic,
  faster than the audio render. UI "Export MIDI" button + `window.onMidiStatus` (exporting/done/failed)
  + Reveal MIDI. **As-sung timing** (no quantization — a notation app meters it). *Known limit:* whole-
  semitone rounding → melisma/portamento are stepped approximations (clean for steady lines). Design +
  plan: `specs/2026-06-22-midi-export-design.md`, `plans/2026-06-22-midi-export.md`.

## 3. Key findings (why it is the way it is)

- **RTF is NOT the bottleneck on this machine** — offline 4 voices ≈ 0.5 RTF. The live noise was
  real-time **thread starvation / cumulative model noise**, not compute. (`OPENBLAS_NUM_THREADS=1`
  made it worse — single-threaded inference; reverted.)
- **`--latency low` (~47 ms) on the Aggregate Device** is right for single-voice. Raising the buffer
  fixed multi-voice crackle but added a ~150 ms **slap-back echo** (you hear your bone-conducted voice
  vs the delayed output) — not worth it.
- **The tenor model only sounds clean at/below its range.** Pitching it UP (Alto/Sop) → warble.
- **JVS base** (`beatrice_paraphernalia_jvs`, 100 speakers) gives real female timbre WITHOUT training
  and is clean as a SINGLE clean voice — BUT it's Japanese-trained, so the rounded Chinese **[u] "嗚"
  comes out as the wrong vowel** (everything else is correct). Non-commercial license. A stopgap only.

## 4. Adding the self-trained FEMALE model (when it lands)

Goal: real S/A from your own Chinese female model — correct vowels (fixes "嗚"), sings high, stable
like your tenor. The architecture already supports it; this is mostly config.

1. **Get the model dir** (the `paraphernalia_data_*` folder with the 5 `.bin` files). Keep it OUT of
   git (license-restricted, like the tenor) — back it up off-laptop. Note its absolute path.
2. **Register it in the engine** — `server/beatrice_solo_choir_live.py`, near `_JVS_MODEL`:
   add `_FEMALE_MODEL = "/abs/path/to/paraphernalia_..."`, and in `_build_voice` map
   `model == "female"` → `_FEMALE_MODEL` (same pattern as the `jvs` branch).
3. **Point S/A at it** — `app/bridge.py` `SATB_VOICES`: set Alto/Sop `"model": "female"`
   (drop the `jvs`/`tenor` placeholder). Mirror the same in `ui/solo_choir_ui_v10.html` `parts`
   (Alto/Sop `model:'female'`). target_speaker is `0` unless the model has multiple voices.
4. **Re-enable multi-voice carefully** — `self.satb = True`, but START with **You + Bass + Sop**
   (2 passes) and verify it's clean before adding more. Watch the RTF pill.
5. **Tune by ear** — Sop interval/octave/formant via the engine's existing per-voice control
   (the tune panel, if re-added) or directly in `SATB_VOICES`.
6. **Prove non-regression** — `python beatrice_solo_choir_live.py --render <clip> <out>` with the
   default 1-voice config must still be max-abs-diff 0.0 vs the prior build. No `server/` DSP math
   changes — only the model path + config.
7. **If stacking still adds noise**, fall back to **offline render** for the full SATB take
   (`--render`) and keep live to 1–2 voices.

## 5. Honest assessment for the viva/demo

The single-voice You + harmony is the **reliable, validated** instrument. Multi-voice SATB is real in
the code and the offline render, but **live multi-voice on this laptop is the rough edge** (cumulative
noise). The clean path forward is the female model + keeping live passes low; full thickness via the
offline render. Don't over-promise live 4-part SATB until the female model proves it stays clean.
