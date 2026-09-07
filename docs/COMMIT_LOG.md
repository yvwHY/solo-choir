# COMMIT LOG — the development history, June to September 2026

This repository is a condensed, English-language copy of the project, assembled
on 3–4 September 2026 for submission. Its own git history is therefore two days
long. The development itself happened in the working repositories listed below,
whose histories are exported here so that the timeline can be read without
access to those repositories.

The main working repository is a fork of `w-okada/voice-changer`
(`yvwHY/solo-choir-vcclient`, branch `ui-app`). Only commits authored for this
project (author `yvwHY`, from 15 June 2026) are listed; the upstream history is
not. Most commit messages from mid-July onwards are in Chinese, which was the
working language of the daily notes; the phase structure and the English
account of the same work are in [`PROCESS.md`](PROCESS.md).

## Commits per week (all working repositories)

| ISO week | Dates | Commits |
|---|---|---|
| 2026-W25 | 15 Jun – 21 Jun | 76 |
| 2026-W26 | 22 Jun – 28 Jun | 87 |
| 2026-W27 | 29 Jun – 5 Jul | 85 |
| 2026-W28 | 6 Jul – 12 Jul | 71 |
| 2026-W29 | 13 Jul – 19 Jul | 67 |
| 2026-W30 | 20 Jul – 26 Jul | 99 |
| 2026-W31 | 27 Jul – 2 Aug | 107 |
| 2026-W32 | 3 Aug – 9 Aug | 141 |
| 2026-W33 | 10 Aug – 16 Aug | 102 |
| 2026-W34 | 17 Aug – 23 Aug | 37 |
| 2026-W35 | 24 Aug – 30 Aug | 3 |
| 2026-W36 | 31 Aug – 6 Sep | 18 |

## Daily working notes

Fifty-three daily notes were written in Chinese in the main working repository
(`_solo_choir_docs/worklogs/`). They are condensed into [`PROCESS.md`](PROCESS.md).
Dates with a note:

2026-06-18, 2026-06-19, 2026-06-20, 2026-06-21, 2026-06-22, 2026-06-23, 
2026-06-24, 2026-06-26, 2026-06-27, 2026-07-01, 2026-07-02, 2026-07-03, 
2026-07-05, 2026-07-06, 2026-07-07, 2026-07-08, 2026-07-09, 2026-07-10, 
2026-07-11, 2026-07-12, 2026-07-13, 2026-07-14, 2026-07-15, 2026-07-16, 
2026-07-17, 2026-07-18, 2026-07-19, 2026-07-20, 2026-07-21, 2026-07-22, 
2026-07-23, 2026-07-24, 2026-07-25, 2026-07-26, 2026-07-27, 2026-07-28, 
2026-07-29, 2026-07-30, 2026-07-31, 2026-08-01, 2026-08-02, 2026-08-03, 
2026-08-04, 2026-08-05 (review), 2026-08-05, 2026-08-06, 2026-08-09, 
2026-08-13, 2026-08-16, 2026-08-17, 2026-08-18, 2026-08-23, 2026-08-24

## yvwHY/solo-choir-vcclient (branch ui-app) — the main working repository: engines, application, firmware, documentation (852 commits)

```
2026-06-15  5d55a94d  chore(mac): enable v1 voice-changer baseline to run from source on Apple Silicon
2026-06-15  a0d0a1f1  fix(mac): load RVC models from source on CPU (fairseq/pyworld + disable MPS)
2026-06-15  c42e3147  fix(mac): convert fp16 RVC onnx models to fp32 on CPU (was producing silence)
2026-06-15  992781e9  feat: Solo Choir mode — live diatonic harmony from the model's own f0 (RVC prototype)
2026-06-15  fe70f5de  fix: persist Solo Choir settings so they survive model reload/restart
2026-06-15  5d47f196  fix(solochoir): hysteresis on held note to stop sustained-note warble
2026-06-15  7ad35d1c  feat(beatrice): live mic->tenor + Solo Choir via Beatrice v2 rc.0 (editable harness)
2026-06-15  e4fe0761  feat(beatrice): optional output envelope tuning (default off); diagnosis notes
2026-06-15  c0643e8e  fix(beatrice): stateful (soxr) resampling — eliminates live buzz, VCClient parity
2026-06-15  6187c117  docs(beatrice): final-state writeup + recorded demo + harness --record
2026-06-15  907a3b05  chore: harden .gitignore against tracking Beatrice engine/model (license)
2026-06-15  49c5ab19  docs: Solo Choir project README
2026-06-15  d8b0770e  docs: fix SoloChoir.py path in README (server/voice_changer/SoloChoir.py)
2026-06-16  274f9976  wip: solo choir pywebview app + bridge + telemetry tap (steps 1-3b)
2026-06-16  613ad4ee  wip: live control → engine
2026-06-17  ce8c753a  feat(solo-choir): You passthrough, live control, latency + transport, mix safety
2026-06-17  2dc13198  docs: add Solo Choir technical doc (architecture + AI-assisted process); move project docs out of upstream web-client build folder
2026-06-17  91687d28  docs: add hardware roadmap (Track B); CLAUDE.md points to docs on demand
2026-06-17  b19f9423  feat(solo-choir): Record→Play persists take to WAV (Step A)
2026-06-17  2e99b291  feat(solo-choir): Record→Play stems — mono mix + stereo L=mel/R=choir (Step B)
2026-06-17  b881057c  feat(solo-choir): surface saved take in the UI — "Saved ✓" + Reveal
2026-06-17  21e9ecad  feat(solo-choir): part-soloed playback + gentler take normalize
2026-06-17  3477ef59  feat(solo-choir): launch engine at --latency low (bridge stream setting)
2026-06-17  2d24007c  feat(solo-choir): score shows the real sung take (live pitch capture)
2026-06-17  e4c899f9  chore: gitignore Claude Code agent memory dir
2026-06-17  6f481ec8  docs: add training notes; CLAUDE.md reflects multi-voice/timbre step, soft 6/29 framing, training track
2026-06-17  673fe66b  feat(solo-choir): 1a — JVS speaker probe tool (B′ female S/A path)
2026-06-17  c3e77a49  feat(solo-choir): 1b.1 — offline --render mode (drives the live callback, no device)
2026-06-17  9e9163b0  feat(solo-choir): 1b.2 — multi-voice SATB (per-voice model/speaker/octave/interval)
2026-06-17  45a7b037  feat(solo-choir): 1c.1 — per-voice on/off + live voices control
2026-06-17  3372c006  feat(solo-choir): 1c.2 — wire SATB to the app (heads + bridge --voices + tuning)
2026-06-17  b2e6941a  fix(solo-choir): auto-pick Aggregate Device + fixed-width readout (stability)
2026-06-17  4b006934  fix(solo-choir): Alto/Sop octave 0 (stable JVS range) + RTF on the LIVE pill
2026-06-17  2509aa4b  fix(solo-choir): gain-compensate the multi-voice choir sum (kills the clip)
2026-06-17  66288958  fix(solo-choir): stabilize multi-voice live — 1 BLAS thread/instance + bigger buffer
2026-06-17  b784fbfe  fix(solo-choir): drop JVS for now → all-tenor SATB at low latency (no echo)
2026-06-17  7195f7e1  fix(solo-choir): interim default = You+Bass+Tenor (clean tenor range; S/A off)
2026-06-17  054c096a  fix(solo-choir): drop the OPENBLAS_NUM_THREADS=1 pin (it caused the new buzz)
2026-06-17  92a39257  fix(solo-choir): live back to clean single-voice (You + 1 harmony); SATB dormant
2026-06-17  19b33d65  feat(solo-choir): live = You + one clean JVS female harmony
2026-06-17  f7171ad3  Revert "feat(solo-choir): live = You + one clean JVS female harmony"
2026-06-17  af9a37c0  feat(solo-choir): SATB mix — tenor low/mid + JVS soprano (You+Bass+Sop default)
2026-06-17  390d61a8  Revert "feat(solo-choir): SATB mix — tenor low/mid + JVS soprano (You+Bass+Sop default)"
2026-06-17  bf21561e  feat(solo-choir): distinguish Bass (octave below) from Tenor (third below)
2026-06-17  81cd2a1a  feat(solo-choir): per-head volume faders (voice gain → mix)
2026-06-17  2462c0c9  tweak(solo-choir): smaller head volume faders + wider range (−30..+12 dB)
2026-06-17  974f41f9  Revert "tweak(solo-choir): smaller head volume faders + wider range (−30..+12 dB)"
2026-06-17  18b15656  Revert "feat(solo-choir): per-head volume faders (voice gain → mix)"
2026-06-17  15c0c5ae  docs: multi-voice/SATB status + female-model integration steps
2026-06-18  f4c6832d  chore(skills): install grill-me, ponytail, adapted brainstorm/write-plan
2026-06-18  b3acfbf7  docs(spec): Render SATB on a take — offline multi-part design
2026-06-18  7a3169a6  docs(plan): add render-SATB design + 4-task plan; point CLAUDE.md at it
2026-06-18  5bc427b8  feat(engine): --no-you flag for harmony-only render (per-part SATB stems)
2026-06-19  a556425a  docs(worklog): 2026-06-18 — render-SATB design+plan, Task 1 (--no-you)
2026-06-19  5469d2e6  feat(bridge): render_satb — async offline SATB render to mix + per-part stems
2026-06-19  05383b64  docs(multivoice): Render SATB action + Bass-solo-rough/mix-coherent finding
2026-06-19  51f3b308  docs(multivoice): resolve Bass keep-vs-off → KEEP (intentional placeholder)
2026-06-19  c20afad9  test(eval): add training-step ablation harness (metrics + level-match + blind)
2026-06-19  9b5e2fde  feat(app): UI model selector (list_models/set_model); default back to 2k
2026-06-19  ebc6fd18  feat(ui): number keys 1–5 toggle voices (Bass/Tenor/You/Alto/Sop)
2026-06-19  74d26732  docs(worklog): 2026-06-19 — render-SATB Task 2 + key fix, step ablation, model selector + keyboard toggles
2026-06-20  f452eeb2  docs(training): F2–C5 protocol dataset beats old model (new25-2k); 5k overfits, combo worst
2026-06-20  ed2608ce  feat(engine): default to new25-2k tenor model (F2–C5 protocol)
2026-06-20  07e185dd  docs(spec+plan): F2–C5 recording-protocol experiment (design + plan)
2026-06-20  3e45b433  feat(satb): wire self-trained female model for live S/A (You+Bass+Sop)
2026-06-20  cf079710  fix(live): revert live to the clean single-voice path (satb=False)
2026-06-20  1760ef07  docs(worklog): 2026-06-20 — F2–C5 tenor shipped, female + 2-speaker models, live multi-voice limit accepted
2026-06-21  76683010  feat(engine): Route A pump — inference off the audio thread (lock-free ring)
2026-06-21  32f98659  feat(app): satb2 choir mode (Route A pump) + lite no-WebGL UI
2026-06-21  c85b047b  docs: relax HARD RULE 1 (core editable) + live-choir thickening design spec
2026-06-21  cacb30b4  fix(harmony): cross-frame median pitch smoothing — kills sustained-note crackle
2026-06-21  e3369d6b  docs(worklog): 2026-06-21 — Route A pump, core-editable, harmonizer crackle fix, feedback root causes
2026-06-21  f2ae5a1e  feat(app): live tenor/alto dual-model default + Bass/Sop deferred (coming soon)
2026-06-21  c8a5a28e  feat(engine): live 'church space' reverb (cheap stable delay-line Freeverb) + UI Space slider
2026-06-21  acdf8528  chore(ui): remove Spread + Ensemble controls (audio versions abandoned)
2026-06-21  a1cfc4f7  docs(worklog): 2026-06-21 session 2 — tenor/alto into app, live reverb, stereo abandoned
2026-06-22  35f21238  docs(claude.md): Later list — add MIDI-input dynamic harmony; mark ensemble/spread abandoned
2026-06-22  267c4f17  feat(live): JVS-sweep SATB default + fix part-toggle choppiness
2026-06-22  16615f39  fix(bridge): preserve voices `model` on restart + add female Alto to live JVS SATB
2026-06-22  31532420  docs(worklog): 2026-06-22 — device-switch timbre collapse SOLVED (model-drop on restart) + female Alto
2026-06-22  1688d086  feat(render): in-app Render SATB button — bounce the live choir to clean stems
2026-06-22  0ee876b3  docs(midi): approved design spec + implementation plan for MIDI/score export
2026-06-22  4459f766  feat(score): score_export — diatonic harmony notes + minimal MIDI writer
2026-06-22  6c71b8c4  feat(score): wav_to_notes — librosa.pyin melody -> note events
2026-06-22  dbc7c8c8  feat(bridge): export_midi — async take -> multi-track SATB MIDI
2026-06-22  76da83bf  feat(ui): Export MIDI button + 3-state status (exporting/✓/failed)
2026-06-22  d2d3e64b  feat(score): snap the MIDI melody to the key — drop slide passing tones
2026-06-22  f0d09c1a  docs(multivoice): document the Export MIDI take action (+ refresh Render SATB status)
2026-06-22  b90cd40a  fix(engine): parent-death watchdog — self-exit instead of orphaning on launcher death
2026-06-22  f534ea76  docs(worklog): 2026-06-22 — extend with Render button, MIDI export, watchdog
2026-06-22  362eb73f  docs(claude.md): demo status — 6/29 rough recorded; full demo (MIDI + hardware) toward viva
2026-06-23  c70dc39f  docs(vision): unify research roadmap (learn/perform); realign hardware roadmap; Track A ear-trainer spec
2026-06-23  1e5ab3e6  feat(track-a): harmony ear-trainer phase 1 — per-part solo playback + practice mode
2026-06-23  2d5639be  feat(track-a): phase 2 ear-trainer — note-match feedback; drop the cents needle
2026-06-23  19a1d651  feat(track-a): phase 3 — multi-select part tabs (You/SATB) + All; sum playback
2026-06-23  977b5290  feat(track-a): phase 4 — log practice attempts to a CSV for the study
2026-06-23  e036ecef  docs(worklog): 2026-06-23 — research vision unify + Track A ear-trainer (phases 1-4)
2026-06-25  4928afc9  docs(track-b): hardware bring-up — DAEX coupling finding + audio front-end build log; Pico W firmware skeleton
2026-06-25  07066113  feat(midi): engine — optional absolute pitch target + MIDI chord mute (guarded, byte-identical)
2026-06-25  d82fb302  feat(midi): bridge — guarded mido reader → midi control field
2026-06-25  3b5bc5a2  feat(midi): UI — MIDI panel (device status, A/B mode toggle, held-chord readout)
2026-06-25  fface521  docs(midi): approved design spec + file-by-file implementation plan
2026-06-25  3cdab9bc  fix(midi): move Held-chord readout into the Sung/Choir live block
2026-06-25  74bcdfc0  feat(midi): Mode B — reflect the MIDI-set key on the UI key chips
2026-06-25  25ad826b  feat(midi): Mode B → key triggers a diatonic triad (chord progression by keyboard)
2026-06-25  00deb84f  feat(midi): Mode B custom chords — 7 editable white-key chord slots
2026-06-25  082e31d5  perf(midi/live): 2-voice low-latency mode (≤50ms / RTF~0.3 target)
2026-06-25  7fdc106b  feat(arranger): v1 rule-based SATB arranger + approved spec/plan
2026-06-25  22f8e1c7  feat(arranger): offline arrange.py — melody wav -> voice-led SATB wav
2026-06-25  e0159741  feat(arranger): tight voicing + own-voice default ('choir of one')
2026-06-25  7debe77f  feat(arranger): wire the auto-arranger into the app (Arrange button)
2026-06-25  d81d9a86  revert(live): restore the post-MIDI 4-voice JVS choir (undo the 2-voice latency experiment)
2026-06-25  c990bc67  feat(live): all 4 voices on the user's own voice model ('choir of one')
2026-06-25  6a0d4140  fix(arranger): Play arrangement must start the play transport, not just set the source
2026-06-25  79989c62  fix(midi): a connected idle MIDI device must not silence the live choir
2026-06-25  574013a9  fix(midi): '— not connected —' must disconnect, not auto-reconnect
2026-06-25  5700c831  fix(devices): auto-pick the 'speaker_set' Aggregate Device on launch
2026-06-25  122a8987  docs(overdub): manual multitrack design + implementation plan
2026-06-25  0e3d2ad6  feat(overdub): engine capture/loop/monitor for manual multitrack (Task 1)
2026-06-25  5f596ae5  feat(overdub): export stacked layers to recordings/overdub_*.wav (Task 2)
2026-06-25  4d65ca21  feat(overdub): bridge mode-enter (single-voice relaunch) + od_* API (Task 3)
2026-06-25  e15891ad  fix(overdub): one-shot commands fire once via engine self-clear (not send-true-then-false)
2026-06-25  21b501e5  feat(overdub): Overdub panel UI — record-layer, loop, mute/solo/delete, export (Task 4)
2026-06-25  2346ff18  feat(arrange): style param + pad texture + arrange_with_chords (Task A1)
2026-06-25  dd22e9c8  docs(studio): harmony-line editor design + implementation plan
2026-06-25  bb87e51a  feat(studio): offline JSON API — transcribe/render_score/score_to_midi (Task A2)
2026-06-25  c0af362c  feat(studio): app skeleton — shell + offline bridge + boot UI + DESIGN.md (Task B1)
2026-06-26  0dfe1927  style(studio): light hybrid theme — frosted-glass editor + grey↔cyan animated gradient
2026-06-26  ad6e6208  feat(studio): Record melody -> Arrange into a score (Task B2)
2026-06-26  c0af4dab  feat(studio): draw score as coloured lines (C1) + free-length record & playback
2026-06-26  076b0471  feat(studio): drag a note's pitch with snap + chord-tone highlight (C2)
2026-06-26  b65a902b  feat(studio): drag onset + duration; hover halo + resize handle (C3)
2026-06-26  e2a8c20f  feat(studio): vowel preview + note ops (delete/split/connect) + undo + playback/transcription polish (C4+)
2026-06-26  252a870b  feat(studio): Basic Pitch transcription + style picker + mic selector (C5)
2026-06-26  7764efa8  feat(studio): Render-to-voice + Export MIDI (Phase D) + voice-select + SATB non-crossing
2026-06-26  f486d12b  feat(studio): focus-voice editing, wider voicing, voice balance, piano render
2026-06-26  32a8baef  feat(studio): CREPE transcription (isolated env) — beats Basic Pitch on flat-pitch multi-syllable
2026-06-26  319b1fe6  feat(studio): SwiftF0 transcription (in-process) — replaces the CREPE subprocess
2026-06-26  2a56555e  feat(studio): female SATB render via kNN-VC (offline) — Render dropdown 'female'
2026-06-26  935b8f5c  fix(studio): SwiftF0 transcription — stop over-splitting sustained notes
2026-06-26  292fbf5a  feat(studio): re-record overdub + per-voice solo/mix listen + louder/weighted render balance
2026-06-26  262c59ad  feat(studio): re-record polish (count-in, volume, revert), Export Mix, preview/render mutex, guide flow
2026-06-26  03353c56  docs(studio): 2026-06-26 worklog (SwiftF0 / kNN-VC female / re-record) + open-SVC survey
2026-06-26  741fc51e  feat(studio): parallel-thirds arranger style + re-record alignment offset + audible count-in
2026-06-26  891577e5  feat(live): free chord-aware harmony + low-latency Solo mode; fix speaker crash + JVS default
2026-06-26  69cce6de  fix(studio): coexist with the live app — 48k count-in + don't default output to BlackHole
2026-06-26  a0dcabf5  feat(live): overdub perform-over-loop — sing live (You + harmony) over the looping accompaniment
2026-06-26  c008fc2c  docs: 2026-06-26 worklog session 4 — live free harmony / Solo / overdub perform-over-loop / device coexistence
2026-06-27  c3388721  feat(training): voice-data preprocessing script — denoise + silence-trim + normalize + slice
2026-06-27  cb9ff2d2  feat(training): preprocess_voice --merge-gap — coalesce voiced fragments so less data is dropped
2026-06-27  980f4753  docs: spine positioning decision (2026-06-27)
2026-06-27  cd3046d5  feat(eval): practice-log report (study + before/after views)
2026-06-27  aab81979  feat(live): exciter on/off — mute R channel for learn/monitoring
2026-06-27  87616338  fix(live): stereo output so exciter R-mute actually works
2026-06-27  6fa415d0  docs: Test A (self-stacking) passed by ear + exciter fix worklog (2026-06-27)
2026-06-27  a2073ff7  docs: Test B — through-body DAEX path failed (predicted); perform-mode = cup spatial source
2026-06-27  ac80839e  feat(study): log subject + channel per practice attempt (子 study IVs)
2026-06-27  e635f11b  docs: worklog 2026-06-27 (study logging + publishing direction) + roadmap §8 enclosure
2026-06-27  5d8d0065  docs: correct who's-who — one supervisor (Daniel Berio); Yee-King is external advisor
2026-06-28  c45575ea  eval(ablation): compare new F2–C5 dataset checkpoints vs old engine model on held-out clip
2026-06-28  e79d2f4a  docs(hardware): enclosure changed from 3D print to wire armature + wound fibre
2026-06-28  9fd83ac5  docs(hardware): material decision — millinery wire (form) + elastic band (bone-conduction pressure)
2026-06-28  b04578b8  docs(hardware): fibre wrap + hardener — main shell vs DAEX cup
2026-06-29  4c3e5f4c  docs(hardware): final purchased material list + build process (wire + fibre mask)
2026-06-29  2ccb395e  style(ui): legibility + tactile controls + action hierarchy polish (CSS only)
2026-06-29  f2d7b2ea  style(studio): legibility + a11y polish and grouped toolbar (CSS/markup only)
2026-06-29  5865978c  feat(studio): render with the new 0627 model, without touching the live default
2026-06-30  91c0f506  docs(qr-roll): brainstorm spec + implementation plan for the QR score-roll
2026-06-30  257f1e94  feat(roll): shared score codec (quantise+delta+deflate+base45) + round-trip test
2026-06-30  b05b9f6b  fix(roll): URL-safe base64url payload (base45 had % and space — unsafe in a URL #fragment)
2026-06-30  777c2c67  feat(roll): vertical score-roll render from #hash (no audio yet)
2026-06-30  acc1ee66  feat(roll): bidirectional scrub + inertial coast (Task 3)
2026-06-30  7857cab4  feat(roll): choir soundfont + sustain playback with silent lead-in (Task 4)
2026-06-30  b7425dfa  feat(studio): Make-card button → QR of the arrangement's roll URL (Task 5)
2026-06-30  6f08f7e0  feat(roll): minimal add/delete + tail-drag length edit; sustain tracks visual length (Task 6)
2026-06-30  f7dd01c8  feat(studio): point Make-card QR at the public Pages deploy (yvwHY/solo-choir-roll)
2026-06-30  2dd8c5c0  fix(roll): harden iOS WebAudio unlock (resume + silent buffer in the gesture)
2026-06-30  2429668b  feat(roll): start hint — turn off silent mode + swipe up (fades on first touch)
2026-06-30  3b425a4e  style(roll): English-only start hint (drop Chinese)
2026-06-30  b2c696fd  fix(roll): pre-decode soundfont at load so the first swipe plays (no pre-tap)
2026-06-30  7d10e0c0  feat(roll): 'tap to start' overlay unlocks audio on first tap (no stray note added)
2026-06-30  bc3633e4  fix(roll): create AudioContext inside the tap gesture (iOS unlock; no second tap)
2026-06-30  5302757c  style(roll): cohesive gallery palette — cool #EEF2F7 + slate chrome + soft spot notes
2026-06-30  5e504597  feat(studio-render): offline choir-blend mix (_choir_mix); ensemble flag, dry path byte-identical
2026-06-30  02a931e3  feat(studio): Choir toggle → offline choir-blend on the voice render (Task 2)
2026-06-30  aecdb6ba  docs(render-choir-blend): brainstorm spec + implementation plan (feature shipped)
2026-06-30  a97d7949  feat(studio): linear step-flow wizard + flat visual redesign
2026-07-01  b3a7e133  docs(hardware): dual bone-conduction series wiring — by-ear confirmed (2026-07-01)
2026-07-01  c380fa62  feat(hardware/mask-frame): STEP 1 — import head.glb as a clean upright reference
2026-07-01  18efb7b6  feat(hardware/mask-frame): STEP 2 — component placement markers on the head
2026-07-02  472f33a9  feat(hardware/mask-frame): STEP 3 — front-of-face frame reference lines
2026-07-02  229ef959  docs(worklog): 2026-07-01 — mask frame reference series + Fusion bridge fix
2026-07-03  72f51cff  docs(worklog): 2026-07-02 — mask CAD marathon: hand-drawn concept → mask_concept11 in Fusion
2026-07-04  9b307792  docs(worklog): 2026-07-03 — concept12 QA+forehead pads; PIVOT to slim frame_A → STL v1
2026-07-04  7a925962  docs(specs+plans): robustness + ensemble/voice-leading + UI/UX design & plan
2026-07-04  e5b040ff  feat(app): Task 1.1 — engine watchdog + honest ENGINE pill + Reconnect
2026-07-04  b746f0d3  feat(app): Task 1.2 — honest per-voice choir note names
2026-07-04  9804ae9e  fix(dsp): Task 1.3 — hold harmony shift through unvoiced frames (kill crackle)
2026-07-04  c4726486  feat(app): Task 1.4 — honest Record->save failure state
2026-07-04  043eb6c8  feat(ui): Task 1.5 — remove the Calibrate placebo (Phase 1 done)
2026-07-04  cb96212b  fix(ui): You head reflects the engine's --no-you launch (honesty follow-up)
2026-07-05  d5cbec39  docs: Dreamtonics tech-stack analysis + Solo Choir's position
2026-07-05  8fcdd811  docs(plan): integrate choir research; Phase 1 done, Task 2.1 pivoted to offline
2026-07-05  50362e3e  feat(ui): Task 3.1 (contrast) — darken --dim/--dimmer for AA readability
2026-07-05  0925d49e  feat(ui): Task 3.5 — honest device empty-state (no stale fake options)
2026-07-05  220dd382  feat(ui): Task 3.2 — vendor three.js + GLTFLoader + Space Grotesk locally
2026-07-05  9cc61301  feat(ui): Task 3.4 — one-time hint that heads are clickable / keys 1-5 toggle
2026-07-05  06964def  docs(plan): mark 2.3 confirmed + Phase 3 progress (3.1-contrast/3.2/3.4/3.5 done)
2026-07-05  8f3415d9  feat(ui): Task 3.1 (font size) — raise smallest caption tier 9->10px
2026-07-05  e55e59da  feat(ui): Task 3.3 — Advanced disclosure (tuck Voice·tune + Study away)
2026-07-05  1a111172  docs(plan): Phase 3 UI/UX complete (3.1-3.5 done)
2026-07-05  981128f0  feat: Task 2.2 — chord-aware (voice-leading) harmony as the live default
2026-07-05  88f0c51d  docs(plan): Task 2.2 done — only 2.4 (f0-once) + offline-Choir remain
2026-07-05  a93a7a9d  perf(dsp): Task 2.4 — compute f0 once, share across voices (removes N-1 FFTs)
2026-07-05  f50ad51d  docs(plan): plan COMPLETE — Phase 1/2/3 all done; only offline-Choir remains
2026-07-05  c19c2b8e  docs(training-notes): document 'Study' — the ear-trainer practice-logging research layer
2026-07-05  c5bb6c37  docs(training-notes): Study — add methodology caveats (sequential blocks + counterbalance)
2026-07-05  40378a4f  test(eval): add vc_probe (minimal skinned-vcclient) + feed_ab pitch-transition harness
2026-07-05  b4b90ffd  feat(dsp): pitch glide — smooth per-note harmony transitions (scrape 1->8 by ear)
2026-07-05  312b7083  docs(spec+plan): solo rebuild — clean vc_probe core + two harmony modes
2026-07-05  0ae9a8b6  feat(solo-rebuild) Task 1: fork vc_probe -> server/solo_min.py (clean core)
2026-07-05  73617833  feat(solo-rebuild) Task 2: delay-align the dry You to the converted path
2026-07-05  a12167c6  feat(solo-rebuild) Task 3: Mode 1 Fixed transpose (--mode fixed --interval)
2026-07-05  07a2af19  feat(solo-rebuild) Task 4: Mode 2 Diatonic + glide + Free sub-toggle
2026-07-05  63b83a9e  feat(solo-rebuild) Task 5: SoloEngine refactor + minimal pywebview UI
2026-07-05  b6e8782e  docs(plan): solo rebuild — Tasks 1-5 built & committed, Task 6 deferred, pending user verify
2026-07-05  a5606408  feat(solo-rebuild): diatonic stability controls (hysteresis/smooth) + hide overfit model
2026-07-05  52d2f78e  feat(solo-rebuild): finalize — Pitch labels + live-tuned stability defaults (hys 1.0/smooth 30)
2026-07-05  1c0b9ade  docs(worklog): 2026-07-05 — vcclient diagnosis + pitch glide + solo_min rebuild (live-verified)
2026-07-05  121466c2  docs(plan): Task 6 two-factories — ATTEMPTED, dead end (real-time DSP shifter sounds bad)
2026-07-05  7a8f2590  feat(solo-rebuild): add a 2nd harmony voice (You + 2 = triad)
2026-07-05  db5c06cd  feat(solo-rebuild): per-voice model (--model2) — mix male+female voices
2026-07-05  edcbacc9  feat(solo-rebuild): per-voice speaker (--speaker2) — mix timbres crackle-free via one model
2026-07-05  b3ccdfe0  fix(solo-rebuild): kill the fixed-rhythm tick — QQ resampler + output cushion
2026-07-05  0837cff7  feat(solo-rebuild): 3rd harmony voice → You + 3 = 4-part choir
2026-07-05  97544345  feat(solo-rebuild): 4th harmony voice → You + 4 = 5-part choir
2026-07-05  85876392  feat(solo-rebuild): GUI = the 5-part male+female choir CLI
2026-07-05  ced8c773  feat(solo-rebuild): per-voice on/off (keep-warm)
2026-07-05  94e8e78e  fix(solo-rebuild): preserve voice on/off + dry state across model switch
2026-07-05  94b7b72c  docs(worklog): 2026-07-05 session 2 — solo_min → 5-part live choir + on/off + model-switch fix
2026-07-06  bd2292af  feat(solo-rebuild): solo_min UI/UX rebuild — v10-family visuals + live telemetry
2026-07-06  cc91deb9  fix(studio): stable render — score-to-score shift + grid pinning + gap-jump glide
2026-07-06  6cd59ee5  feat(studio): transcription splits legato + same-pitch re-attacks; SwiftF0-unified tracks
2026-07-06  ed3b447a  feat(studio): vowel-boundary splitting + honest note pitch — "8 syllables = 8 dots"
2026-07-06  8eef0dd3  feat(studio): absorb leap transits — undershoot/correction is a gesture, not notes
2026-07-06  cebc9d9b  fix(studio-ui): settings popover above the editor; player visible right after Record
2026-07-06  78358e33  feat(studio): quantise on the singer's tuning centre (θ) + post-leap settling absorb
2026-07-06  c5333588  fix(studio): drop the trailing release crack
2026-07-06  4652f4bf  fix(studio): trim notes to their voiced extent — faithful rhythm
2026-07-06  96a870dd  fix(studio): absorb the leading onset gesture (first-note fry/overshoot)
2026-07-06  76d3e15f  feat(studio): global Viterbi note decoding (_hmm_melody) — replaces local heuristics as default
2026-07-06  c8575fd0  fix(studio): HMM decoder — last-syllable cut margin + snap note starts to the energy attack
2026-07-06  85645aaf  docs(studio-ui): singing guide in the record-step hints
2026-07-07  91bac453  feat(studio): take-home QR — audience downloads the mix over local Wi-Fi
2026-07-07  0d2a21ea  docs(worklog): 2026-07-06/07 — solo_min UI/UX, render stability, Viterbi transcriber, take-home QR
2026-07-07  ef0235b3  fix(studio-ui): crooked QR — supersample and smooth-downscale
2026-07-07  4fe5854a  docs(institution): cross-model knowledge base — GRAVEYARD/FINDINGS/STATE/VERIFICATION/DEBUG_PLAYBOOK + 4 skills
2026-07-07  e18772a5  docs(worklog): 2026-07-07 session 2 — cross-model institution docs
2026-07-07  0d59b17d  feat(studio): design the take-home page as a keepsake card
2026-07-07  d23f1c03  docs(worklog): take-home QR fix + keepsake-card design pass
2026-07-07  f3cb70c2  fix(studio-ui): one QR entry point — merge Take-home into the Make-card overlay
2026-07-07  8d4f5242  docs(worklog): QR entry-point merge (user-verified)
2026-07-07  df6f6125  docs(state): ui-app pushed, in sync with origin
2026-07-07  b566d1c9  feat(studio): render with satb2 — Sop/Alto female by part name
2026-07-07  ed76275e  docs(state): studio on satb2; correct MIDI ownership (original app, not solo_min)
2026-07-07  745e9fa1  docs(state): ui-app pushed, in sync with origin
2026-07-07  0a6d5494  feat(studio): exports/ keepsake folder + block LAN directory listing
2026-07-07  b2601d0c  docs(state): exports keepsake folder + LAN listing block recorded
2026-07-07  0cc7c769  fix(solo_min): path fence on set_model — resolve + MODELS_ROOT containment
2026-07-07  e9b39a95  docs(worklog): security sweep results + set_model fence fix
2026-07-07  455d7510  docs(state): solo_min is the live dev target; MIDI port queued
2026-07-07  14f0dadf  docs(plan): MIDI port to solo_min — 4 tasks, byte-identical guard
2026-07-07  a0a8ef69  feat(solo_min): engine midi chord-hold control (additive, byte-identical when absent)
2026-07-07  cec75363  feat(solo_min): bridge MIDI I/O — device pick, A/B modes, shared chord slots
2026-07-07  aaf3d518  feat(solo_min): UI MIDI card — device, A/B mode, held chord, Mode B editor
2026-07-07  1b85e25c  docs(worklog): MIDI port Tasks 1-3 done, Task 4 pending by-ear
2026-07-07  4c136c63  docs(state): MIDI port to solo_min done — only Task-4 by-ear remains
2026-07-07  998289af  feat(solo_min): MIDI playability — 15ms chord coalesce, sustain pedal, latch
2026-07-07  856d39b3  Revert "feat(solo_min): MIDI playability — 15ms chord coalesce, sustain pedal, latch"
2026-07-07  76426920  Revert "feat(solo_min): UI MIDI card — device, A/B mode, held chord, Mode B editor"
2026-07-07  7194e563  Revert "feat(solo_min): bridge MIDI I/O — device pick, A/B modes, shared chord slots"
2026-07-07  0b7ac153  Revert "feat(solo_min): engine midi chord-hold control (additive, byte-identical when absent)"
2026-07-07  9389ecc3  docs: G20 MIDI-in-solo_min removal + CoreMIDI hot-plug gotcha + state sync
2026-07-07  6101de20  docs: llm-council decisions — G21 Beatrice-VST dead end; OSC/TD in exhibition batch
2026-07-07  c98da335  docs(state): 7-week viva plan — feature freeze, W1 deliverables, kill list
2026-07-07  4b800a2c  feat(solo_min): audio device selector — Track B test-session enabler
2026-07-07  e90b6501  docs(spec): baseline comparison + listening test design (draft for review)
2026-07-07  5f17b1ab  docs(playbook): audio-device hot-plug snapshot gotcha (throat device case)
2026-07-07  3c287180  feat(solo_min): master output gain 0..+18 dB — exciter drive for Track B
2026-07-07  31143448  docs(findings): exciter drive levels measured — +10 dB working point, +15 dB thermal limit
2026-07-07  3834f070  docs(hardware): power route B (LiPo+TP4056+MT3608), island connectors, Pico retires, PTT removal plan
2026-07-07  6159b134  docs(state): Track B parts ordered — W1 procurement item done
2026-07-08  208efbb0  docs(worklog): session 4 — Track B bench review, power route B, island connectors, parts ordered
2026-07-08  4252f7f8  docs(worklog): llm-council mask-frame verdict — three-piece + v1-first gate, pending ratification
2026-07-08  1d86517d  docs(hardware): 2026-07-08 school test + pivot — AI-Micro front-end, chest DML direction, headset metal parts ordered
2026-07-09  5a7a8f15  fix(solo_min): device rescan button — PortAudio re-init on rebuild
2026-07-09  cc6dc51e  docs(worklog+state): session 2 — rescan fix, school-test pivot, Masque research pointers
2026-07-10  554b02ac  docs(worklog+state+hardware): 2026-07-09 Fusion CAD day — frame redesign + assembly split, loft-collapse repair, 卡榫工單, wire-order correction
2026-07-10  ae1486ef  docs(worklog+state+graveyard+hardware): 2026-07-10 卡榫日 2 — WO-2 done, WO-1 converged to collar-self-axis pivot_mount (G22), WO-3 handed to Harry, no-trim + shoulder-bearing rules
2026-07-10  cdea6de5  docs: correction — wire material NOT decided (brass = recommendation only, Harry 2026-07-10); mark solder-dependent options conditional
2026-07-10  b7e69315  docs(worklog+state): 2026-07-10 evening — WO-3 pinch-collar door system complete, WO-4 BCE pendant v4 (front-of-ear, threaded eyelet), band goes hand-bent ⌀3 (2×1m rod), form rule round-in/angular-out
2026-07-10  5ee1704c  docs: WO-4 final is v5 snap-slit clamp (not threaded eyelet) per Harry's annotated drawing
2026-07-11  439baca7  docs(worklog+state): WO-4 v5→v10 late-night convergence — band-frame jaws clamp, double-strand band discovery, slit at main-strand midline; v10 pending morning review
2026-07-11  5e13dce4  docs(worklog+state): 2026-07-11 WO-4 DONE — jaws 2.89/slit 0.79 (torque halved), uniform r0.5 fillets, left mirror bce_mount_L; fillet recipe + incident lessons
2026-07-11  c1bd9b58  docs(state): 07-11 STL batch exported to SoloChoirCode/260711_prints (9 parts + print sheet)
2026-07-11  444fe36b  docs(state+worklog): wire material DECIDED — K&S 3955 brass 3.0mm×1m ×4; kerf-aware cut plan; Amazon leaded-brass/imperial traps recorded
2026-07-11  edad08b1  docs: wire numbers re-derived from live CAD (band 429 single-strand, yoke 654, all cold-bend, no anneal) — hairpin claim was stale; bend jigs shipped
2026-07-11  85aa538c  docs: orphan lower-strand step removed from bce_mount (phantom double-strand cleanup); anti-rotation now clamp-friction only — verify on print
2026-07-11  bf1e5ece  docs(worklog): step removal redone via deleteFaceFeatures heal (sketch-on-tilted-face gotcha recorded)
2026-07-11  a5b5fa3b  docs(state): K&S rods ordered; RODE AI-Micro arrived (checklist 1-4 pending)
2026-07-11  d1aa0cf5  docs(sync): 07-11 evening — STL batch v2 (14 files), harness cut list into HARDWARE_ROADMAP, manufacturing-phase next-step order
2026-07-13  498f3ab8  docs(sync): 07-12 late — BCE ear-side JST done; belt-pack SUPERSEDED by WO-5 integration box on WO-3 (X obsolete, L± re-measure on frame); exhibitor form filled, consent+submit pending
2026-07-14  ac9a91c9  feat(eval)+docs: AI-Micro arrival checklist ①-④ all passed → UGREEN retired; latency probe
2026-07-14  c9d65d82  docs(debug): playbook +2 rows — AI-Micro wedged-state crackle (power-cycle fix, measured 710→11 events); thin-You/low-input-level row; -9986 hotplug note
2026-07-14  91b2dccd  docs(sync): 07-14 bench-debug day — DAEX 粗糙 = 3 stacked causes (two-clock overflow re-hit; AI-Micro wedged state → power-cycle reset, 710→11 events; input gain low → RØDE Central); STATE + worklog
2026-07-14  b3f1bcc1  docs(debug): wedged-state row += show-day SOP (fixed power-up order; power-cycle first on unexplained anomalies)
2026-07-14  4ad23d8c  feat(eval): input_health.py — pre-flight mic-chain health check (LEVEL/CRACKLE/SNR/LR), distilled from 07-14 debug day
2026-07-14  8828a08a  docs(spec): automated mode design (Daniel 07-14) — voice-triggered arc runner, W5 timebox, no code until /write-plan; STATE W5 line updated
2026-07-14  7ed16478  feat(live): --split-lr spatial-separation experiment + eval/spatial_pilot.py
2026-07-14  11975b0e  docs(sync): F11 physical spatial separation (pilot + live verified); HARDWARE_ROADMAP 4-speaker exhibition rig plan; worklog/STATE 07-14 afternoon
2026-07-14  7bc5b334  docs(hardware): voice-unit modular concept (DAEX+phone-face units, frame- or stand-mounted) + per-voice output = channel mapping rule; 4 open questions before any WO
2026-07-14  2c44912e  docs(hardware): 07-14 late decisions — BCE per-ear channels (WO-5 → 2× PAM), 6ch central-amp doctrine, 7.1-card measure-first gate, screens frozen (OLED prototype first), PCB rejected on staging
2026-07-14  25bb47a2  docs(hardware): power/wiring diagram artifact 93627ded + two drawn-out constraints (AI-Micro 2-out limit in wearable; BTL floating units = loop-free); 7.1-card gate now latency AND mic-in
2026-07-14  b6ebd92a  docs(hardware): shopping pass with Harry — TP4056 owned (07-12 buy closed), collars→printed pinch-ring, grommets deferred to tap-test, 22AWG accepted; final buy list = DAEX25×3 + 7.1 card + screw kit (incl countersunk M3) + microUSB×4
2026-07-14  fdf22461  docs(hardware): purchases finalised (LEAGY 6ch card + countersunk kit + DAEX25x4); unit power = 4-core loom to VSYS, not batteries/per-unit USB; SUCESO splitter required for exhibition mic chain
2026-07-14  c8927768  docs(worklog): 07-14 evening — power-chain bring-up complete (battery→TP4056→MT3608 5.043V; cold-joint 1.693V diagnosed via meter bisect); BCE acceptance cleared (polarity + 4-4.5Ω single-unit spec); TP4056 confirmed USB-C
2026-07-15  6092d1a9  docs(worklog): WO-5 electrical prototype LIVE — battery rail → PAM 5V → BCE+DAEX playing from AI-Micro; MT3608 #1 casualty diagnosed & retired (spares now zero — reorder)
2026-07-15  c943811e  docs(worklog): 07-15 priority list — exhibitor form (deadline today) > baseline vocoder (behind, 0 lines) > pilot recruitment > checkout (+MT3608) > 2nd PAM > WO-5 Fusion
2026-07-15  d5443039  docs(worklog): todo +7 live-app per-voice channel routing (gated on CSL card arrival + gate tests; flag-guarded + Regime A/B if pulled ahead of W5)
2026-07-16  3634a5d2  docs(worklog+state): 2026-07-15 WO-5 box CAD day — sliding-lid box DONE (88×65×17.2, sim-verified snap detent); AI-Micro out of box onto lid J-hooks (card-swappable); 3×PAM = one 5V rail, current-budget question answered
2026-07-16  5f180b7d  feat(solo_min): --out-map per-voice output channel routing (todo +7 pulled ahead)
2026-07-16  7930068d  docs(worklog+state): 2026-07-16 wireless-audio research verdict (BT/WiFi dead, 2.4GHz RF only viable) + solo_min --out-map landed (5f180b7d, Regime A proven, live by-ear pending)
2026-07-16  393548a1  docs(state+worklog): out-map Regime B live-verified — 3-way spatial pilot PASSED (aggregate AI-Micro+Mac spk, 0 underruns, by-ear clean + spatial resonance)
2026-07-16  a7180c65  docs(worklog): 07-16 §C four-channel topology ratified — Mac 3.5mm + new PAM as self-powered optional voice-unit; PAM series-chaining forbidden (BTL); parts gap noted
2026-07-16  c6ccb0d4  docs(worklog): 07-16 §D purchase finalised — cart £202.80→£42.55; white-label exciter A/B plan (Dioche 50mm + 35mm 4Ω vs owned DAEX25CT-4); UMC404HD & VT-4 ×4 removed
2026-07-16  3275690b  docs(state+worklog): 07-16 order placed £42.55 (exciter A/B pair 4Ω verified + MT3608 + screws); STATE hands off next session = new-PAM 4-channel bench
2026-07-16  a44461d9  docs(state+worklog+findings): 07-16 §E WO-5 wrap-up CAD + §F new-PAM voice-unit built & 4-part/4-transducer live PASS (MT3608 rehabilitated 5.044V; F7 aggregate facts: ASCII names, launch-time device hit, in=out)
2026-07-16  86a9ac75  docs(state+worklog): 07-16 §E amendments — unit-power exit reworked as symmetric gate pair (mirror of speaker gate, both exit via pivot window; 15×7 high window removed) + rounding batch 2 (box perimeter/verticals r1.0, taper r1.5, lid tail r0.8; 3 knife-edge refusals left to print line-width)
2026-07-16  971697b4  eval: promote sm_render.py offline callback-feed harness from 07-16 scratchpad (Regime-A proof entry for solo_min paths)
2026-07-16  fe6c2890  docs(state+worklog+plan): 07-16 §H --voice-delay Tier-1 canon plan written (pending todo, NOT started) — Harry-ratified exhibition freeze-override; plan w/ shift-equivalence proof + ear gate
2026-07-16  8fdc119f  docs(state): Exhibitor Form was SUBMITTED (Harry 07-16) — stale 'not submitted' note corrected
2026-07-16  854fea43  docs(state+worklog): 07-16 todo ratified — UI voice→channel mapper in live app (W5); OLED Stage-5 deferred to 07-17 (Mac-side prepped in SoloChoirCode/260716_pico_oled)
2026-07-17  56955d9a  docs(state+worklog): 07-16/17 WO-4 window-seat saga closed — v6 1.0-frame chosen (Harry: print + hand-sand spur), both ears rebuilt symmetric; WO-5+WO-4 STL batch exported to 260717_prints; five build lessons recorded
2026-07-17  96f3c622  docs(worklog): 07-17 gcode batch sliced — 3 plates to 260717_prints (MINI/PETG, supports kept out of cavities by orientation + buildplate-only)
2026-07-17  e0a01651  docs(worklog): dedupe lesson list remnant in §G
2026-07-17  6dc625ef  docs(state+worklog): 07-17 gcode batch re-sliced PLA for HatchLabs — Generic PLA @MINIIS from vendor bundle, placement verified vs PETG, PETG held in gcode/petg_hold
2026-07-17  1cdfbebe  docs(state+worklog): 07-17 midday — bce v7 no-rail from lab feedback (side-rail cut, pad +0.5 → seat 8.6), P3v2 PLA sliced; lab notes (filament-type warning, first-layer fail)
2026-07-17  0cf753bb  docs(state+worklog): bce no-rail revision REVERTED on Harry's call — Fusion/files back to v6 baseline, P3 v1 gcode restored
2026-07-17  b4bd119a  docs(state+worklog): bce v7 landed — mouth-side corner post removed both ears, pad +0.5 via Harry's shared-param edit, P3v2 nopost sliced; mis-cut lesson recorded
2026-07-17  0a407b57  docs(state+worklog): 07-17 §F OLED/Pico W Stage-5 LIT on school carrier — GP4/GP5, hw-I2C dead on this bus, SoftI2C mandatory, ~29 fps, standalone on boot
2026-07-18  b12a5e53  feat(solo_shell): --face-udp — stream {level,voiced,f0} telemetry to the Pico OLED face over UDP broadcast (shell-level flag, engine argv untouched, sent from tele thread only)
2026-07-18  a1cbcfa3  docs(state): note --face-udp landed b12a5e53
2026-07-18  6bcbd31c  docs(state+worklog): 07-17 §G Stage-6 telemetry mouth VERIFIED by real singing — Pico self-AP pivot, adaptive envelope + fallback; next = Slate AX dual-mode boot; ADC tap = mouth end-state candidate
2026-07-18  475ab775  docs(hardware): voice-unit three-plane wiring + 5V rail tap rule
2026-07-18  ee487cad  feat(solo_min app): UI voice→channel mapper + rebuild hardening (out-map UI-isation)
2026-07-18  923c06f0  refactor(solo_min): retire --split-lr — superseded by --out-map (0,0,1,1 ≡ split-lr)
2026-07-18  6d96849a  docs(state+worklog): 07-18 §D — UI voice→channel mapper landed + --split-lr retired
2026-07-19  9fc51fb1  docs(worklog): WO-5 v8→v9 — C-channel slide fix, box stretch ×3 axes, centring pass
2026-07-19  ecb152a4  docs(state): sync to WO-5 v9 — tri-axis +10 box, centring pass, current print plates
2026-07-19  342e839d  docs(state+worklog+hardware): 07-19 §C box wiring built + 4-part live PASS; LEAGY death recorded
2026-07-19  3129dbde  docs(worklog): 07-19 §D–§I — 出線口雙向置中定案、黃銅通道分級加大(⌀4.3/⌀3.8)、交界修正案A、螺絲孔工程
2026-07-19  0e6695cb  docs(state+worklog): 07-19 §J 過帳 — mapper 真機耳測 PASS、--mon-ch 暫緩、exciter A/B 初判(Dioche 偏重)
2026-07-19  62b725fe  solo_min: --voice-delay per-voice output FIFO (Tier-1 canon/round; flag off = byte-identical, proven)
2026-07-19  5fcb27c6  docs(plan): baseline comparison + listening test — file-by-file W2/W3 catch-up plan (from 07-07 spec)
2026-07-19  78b09218  docs: ethics email 取消入檔（決策早已存在但未曾入文件致反覆重提）— spec 註記/STATE 劃銷/plan 收案
2026-07-19  423fd85c  docs(graveyard+state+worklog): --voice-delay live canon by-ear FAIL ×2 → G23 (DAF + spine violation); flag dormant, offline shape only
2026-07-19  80fd3e21  docs(worklog): voice-delay 處置定案 — Harry 拍板 flag 留休眠
2026-07-19  9abf25b1  solo_min+SoloChoir: --voice-lead Free sub-variant (common-tone hold, minimal leap, register anchor; off = byte-identical, proven)
2026-07-19  26026048  voice-lead: snap compensatory shift on common-tone hold (harmony pitch continuous; glide only real moves; off-path byte-identical, proven)
2026-07-19  03973742  docs(state+worklog+playbook): voice-lead 休眠（未 graveyard）、LiPo 低電 brownout 症狀入表、rail 總開關待辦
2026-07-19  e5d434af  studio: naive phase-vocoder baseline render branch (method=, default byte-identical, proven) — viva comparison layer
2026-07-19  8e7cb099  docs(worklog): Task 1 naive branch 收案 — Regime A 0.0；Regime C 移至 Task 5（demo 素材譜爛，鋼琴診斷確認非渲染問題）
2026-07-19  f00c8643  eval: blinded matched-stimulus batch script for the baseline listening test
2026-07-19  a6775798  eval: objective intonation metric (SwiftF0 cents-deviation vs score) for the baseline comparison
2026-07-19  ef4d258a  eval: predicted-MOS script (DNSMOS local) + static A/B listening-test page (blinded, JSON export)
2026-07-19  3eb06df1  docs(worklog): 聽測 Task 2-4 落地紀錄（stimuli/intonation/DNSMOS+test page）
2026-07-19  53172c84  docs(state): baseline listening-test Tasks 1-4 DONE; remaining = Task 5 (Harry) + main run
2026-07-19  518e96b9  docs(plan+worklog): Task 5 open questions resolved — 新錄 4 樂句；XM4 有線施測
2026-07-20  4beb6f41  docs(specs+worklog): interaction-mode pivot → wire-light; exhibition mode paused
2026-07-20  db549779  docs(wire-light): 2026-07-20 落地 — COB 選定+盤點+接線最簡化
2026-07-20  d8a5c2f5  docs(hardware): bench cable-mgmt — reversible-only until verified
2026-07-20  da938960  docs(wire-light): 拍板 光走電線非骨架 — 骨架剪長作廢
2026-07-20  d2610fbf  docs(state): 07-20 晚間 — 黃銅骨架彎成 + wire-light 定案(COB/光走電線)
2026-07-21  724aebd8  docs(state+worklog): 07-21 — WO-2 C口 / WO-3 整件重建 / 黃銅通道縮孔 / 全批重出 + voice unit v1
2026-07-21  dc607c00  chore(prints): 把 260719/260721 列印批次納入版本控制（gcode 除外）
2026-07-22  e00340b6  feat(solo_min): live naive-shifter baseline (RQ1 A/B) + fix click on every control press
2026-07-22  3c90803b  ui(solo_min): 對比/字級修正、A/B 控制、拆 Performance/Setup、移除 F/M
2026-07-22  4beaed86  docs(state+worklog+findings+graveyard): 07-22 — 聽測改走 live A/B、naive 基線、控制爆音修正
2026-07-22  240fed35  docs(state): NEXT SESSION 指標 — 展演模式（與聽測強相關，07-20 起的 paused 狀態將重啟）
2026-07-22  5320438a  prints+docs: 260722 批次（黃銅通道 +0.1、clip/col 加寬、支撐與配合修正）＋ 狀態同步
2026-07-22  95225995  feat(solo_min): 展演模式 —— entrance 開場序列、choir/you gain、輸出 gate 接線
2026-07-22  7b3ea244  docs(graveyard): 修正 G24 —— 把「編曲是根因」降級為未驗證假設，並補上統計效力問題
2026-07-23  ad98e873  docs(state+worklog+graveyard): 07-22 收尾同步（session 中斷未及 commit，07-23 補入）
2026-07-23  9824f903  prints+docs(P7): vu2 喇叭腔體一體件 —— C 口夾直接長在腔上，取代平板方案
2026-07-23  c27ba549  harmony/: Transformer counterpoint brain -> solo_min decision source (plan Tasks 0-2)
2026-07-23  d81bd963  docs: STATE + worklog 07-23 §G-§K — brain->solo_min Tasks 0-2 landed
2026-07-23  3472b484  harmony: Task 3 -- wire_live single sample clock for the live brain
2026-07-23  18994627  docs: Task 3 wire_live landed -- STATE + worklog 07-23 §K/§L
2026-07-23  6036e78e  gitignore: harmony/out/ (sustain-trigger render outputs)
2026-07-23  6510fed4  harmony/ear.py: torchcrepe-tiny ear + A/B harness -- verdict: parity with YIN on real stems (sustain metric 13/13 both), YIN stays default for design 2
2026-07-23  36adbe53  harmony/sustain.py: SustainDetector + --scan mode (design 2 Task 2, trigger tuning on real-voice stems)
2026-07-23  a441b9fb  harmony/sustain.py: per-note batch-WORLD mouth + offline render (design 2 Task 3, ear-gate material)
2026-07-23  40c01a38  docs: STATE + worklog 07-23 §M-§Q + FINDINGS F15 -- design 2 sustain-trigger Tasks 0-3, torchcrepe premise overturned
2026-07-23  d9453f2a  harmony: fix angel portamento -- absolute-note pinning replaces external steps
2026-07-23  65b18434  docs: worklog §R + STATE -- angel portamento root cause & abs-pinning fix
2026-07-23  3cf6e75b  eval: angel metrics -- glide meter, pitch accuracy vs brain targets, scrape meter
2026-07-23  9f75822a  docs: worklog §S + STATE -- objective scrape verification, glide-0 verdict stands
2026-07-23  931a392d  harmony/peek_probe.py: anticipation probe -- next-token logits as forecast of the singer's next note (measurement only)
2026-07-23  684454d6  harmony: peek_probe anticipation measurement -> G29 + second-chorus-partner proposal
2026-07-23  270bc830  harmony: Task 1 corpus scouting -- POP909 verified, BRIDGE-gap found
2026-07-23  8046505f  harmony: pairing-half feasibility x3 -- chorales in hand, silver_line.py, CPDL viable
2026-07-23  482548b3  docs: brain->solo_min line DORMANT by Harry's ear verdict (machine-voice timbre)
2026-07-23  326bcb0a  docs: wire-light spec control = Tier 2 ADC（07-20 §C 拍板，補漏改）＋ 07-23 定案八條；pico LED 落地 plan
2026-07-23  25b0f362  harmony: cpdl_scrape.py -- Partsongs x SATB corpus scraper, full run launched
2026-07-23  c8e2c5e6  harmony: silver_line v4 -- breath-bridging + line memory kill the parallel shadow
2026-07-23  1083e004  harmony: tokenizer v2 -- C-normalized chromatic tokens + beat-phase channel
2026-07-23  7e9765b3  docs: STATE sync -- silver v4 ear-passed, tokenizer v2 done, CPDL scraping
2026-07-23  d15f7d6a  firmware: pico_led v5.1 火流定稿 — bench 全項驗證（60fps、160/m、直連＋330Ω必裝、GP0 陣亡棄用）
2026-07-23  02e17276  docs: worklog §V + STATE — Pico LED Task 0–1 完成（8 題定案、火流 60fps bench PASS、GP0 陣亡教訓）
2026-07-23  18787230  harmony: cpdl_pairs.py -- CPDL (sop, alto) pair extraction, full run launched
2026-07-23  ee321251  harmony: corpus_split.py -- whole-song train/val + oblique filter (Task 3)
2026-07-23  54320439  harmony: train_v2 -- 12M/CTX4096 retrain on the 3-block corpus, launched
2026-07-23  3e2d0656  docs: STATE sync -- Task 3 done, v2 training running
2026-07-23  83981b82  harmony: v2 verified -- induction emerged (17% vs 27%), pitch forecast 2.4x Bach
2026-07-23  f50e520e  harmony: v2 brain behind solo_host -- three-accent renders at the ear gate
2026-07-23  9a7e33cf  docs: STATE sync -- v2 verified end to end, waiting at the ear gate
2026-07-24  b1bdd569  firmware: pico_led live 模式定稿 — Task 2 ADC 分接 bench 全 PASS（min-of-3 濾波、NOISE_FLOOR 500、大電容必裝）
2026-07-24  2d5fccfa  docs: worklog §V 續 + STATE — Task 2 ADC 分接 PASS、資安檢查記錄、PSK 輪換
2026-07-24  999eb5ff  harmony: three mouths for the v2 brain -- WORLD + Beatrice myvoice/girl
2026-07-24  66ef06e0  docs: ear verdict (WORLD wins) + 2x2 mouth experiment design for next session
2026-07-24  b0c37b98  harmony: v2 checkpoint (self-trained, 3h MPS -- same policy as v1 best.pt)
2026-07-24  a28cab3d  harmony: 2x2 mouth experiment -- world_mouth dual f0 modes + beatrice_mouth (Test A) + blind_pack
2026-07-24  be744e27  docs: sync STATE + worklog 07-24 §C -- 2x2 blind pack at Harry ear gate
2026-07-24  a1b53df9  docs: 2x2 blind verdict -- shift-style f0 is the main culprit (F16); Beatrice rehabilitated under clean pitch
2026-07-24  dce76b43  harmony: world_stream -- strict-causal streaming target-f0 WORLD (file-1a)
2026-07-24  892503cf  docs: parallel mouth lines -- world_stream at ear gate, DDSP-SVC training launched
2026-07-24  1cd529c2  docs: stream blind PASSED (stream beats batch, F17) + ddsp10k timbre confirmed
2026-07-24  469ba7b9  harmony: world_live -- live host for the streaming target-f0 WORLD mouth
2026-07-24  ee2fa581  docs: worklog §I world_live host ready, awaiting live mic test
2026-07-24  2ecf6717  docs: ddsp training stopped at 30k (val plateau), final 4-mouth blind pack ready
2026-07-24  416904c2  docs: final mouth blind -- DDSP-SVC wins (F18); next = live mic test + DDSP live port
2026-07-24  74665bf7  harmony: world_live fix -- lag 0.5 default (0.35 starved the ring), live status line + starvation counter
2026-07-24  d6a0e7eb  docs: worklog §K -- girl DDSP angel rendered (12k best), world_live latency-account fix
2026-07-24  61ba59ed  harmony: humanize lab -- the autotune feel is the target-f0 RECIPE, not the mouth
2026-07-24  fb07a3f6  harmony: gesture_lab (R1 corpus f0 transplant) + DDSPMouth in world_live
2026-07-24  e0e09b25  docs: worklog §L -- R1 gesture transplant + DDSP live mouth landed
2026-07-24  753ca9cc  harmony: DDSPMouth own 0.7s bound -- trims MPS cost spikes that starved the ring live
2026-07-24  766dfe95  harmony: SOLA splice for DDSPMouth + carrier verdict docs (F19)
2026-07-24  bfde7e25  docs: STATE sync -- evening ablation chain + carrier upgrade fork
2026-07-25  0c06e6f8  docs: STATE+worklog sync -- carrier upgrade ①mix+②reflow executed, blind pack at Harry's ear gate
2026-07-25  e2ca85f3  docs: carrier-upgrade verdict -- control holds, G30/G31 buried, F20 registered (data ceiling)
2026-07-25  7ff97798  docs: worklog §G -- SOLA retest verdict (halved, residual is chunking-layer)
2026-07-25  aedc2bf6  docs: goal bar pinned (in-context), recording-session plan, listening-test status flagged
2026-07-25  0595caef  docs: worklog 07-24 §N — wire-light 硬體場（Task 3 半截：COB 剪 4 段；Task 4 洞洞板開工：介面焊完＋兩軌佈局定案）
2026-07-25  d1e1a84d  docs: resolve listening-test contradiction -- cancelled 07-22 (RtD), stale Task 5/6 TODO struck
2026-07-25  3158a236  feat(harmony): option-0 sampler mouth v1 — real-recording unit library + varispeed micro-shift, no synthesis
2026-07-25  b4451cb9  feat(harmony): sampler mouth v2 — singing-gesture layer (still zero synthesis)
2026-07-25  e1fbc77b  fix(harmony): sampler mouth v4 — articulation imprint reaches conversion-carrier territory
2026-07-26  0a3788c3  docs: G32 — full-line concatenative sampler mouth dead (4x ear FAIL); option-0 demoted to sustain-role candidate
2026-07-26  925434fd  fix(harmony): sampler v5 ablation after G32 withdrawn — Harry rejected the quick verdict, measured attribution first
2026-07-26  e2725ab1  docs(prints): P2 postmortem — P8/P9 re-cut plates replace it (P9 flagged stale: straight-bore redesign not yet exported)
2026-07-26  0b25f15c  docs: worklog 07-25 §H + STATE — Task 4 佈局圖重修（類比小區重畫＋原理卡＋擬真焊法特寫）
2026-07-26  79148779  docs: STATE — push checkpoint 0b25f15c 記錄（eval/stimuli 素材夾暫不進版控）
2026-07-26  1a7fcb5d  feat(harmony): sampler v7 — run-based rendering: one unit bends through consecutive notes (true legato, zero seam)
2026-07-26  84a36554  docs: worklog 07-26 §C — instability elimination chain + v7 run-based legato
2026-07-26  6807ec4c  feat(harmony): night shift — key normalization (take is A major!), breath units, trio v1, anticipation pricing
2026-07-26  48e8b53a  docs: STATE — night-shift summary (key discovery, v7 legato, breaths, trio, anticipation pricing); all gates queued for 07-26 morning
2026-07-26  cdef6643  feat(harmony): direct mouth — his own voice into DDSP with the brain's f0, WORLD removed from the chain
2026-07-26  1d727dac  feat(harmony): anticipation mode + respond auto-key; G32 reinstated (option-0 fully dead, Part B cancelled)
2026-07-26  e479fe2c  firmware: pico_led — 1N4007 降燈條 VCC（3.3V 資料門檻翻案）＋ N=122 實測
2026-07-26  869506eb  firmware: pico_led — unit1 ADC 分接 live 驗收 PASS（NOISE_FLOOR 500→2000）
2026-07-26  f834388d  docs(pico_led): T1/T2 驗收 gate 通過 — 實唱三判準全過
2026-07-27  7f023fb0  docs: session close — Part B cancelled in the recording plan, STATE syncs morning verdicts + hardware gate
2026-07-27  dc368938  firmware: pico_led — 燈條自我耦合補償 + 反應調快；GND 虛焊血訓
2026-07-27  fe3033c8  docs: sync-state — wire-light unit1 完整運作（worklog §F、FINDINGS F21、STATE）
2026-07-27  e97e0b00  Docs: 07-27 ear-test verdicts + brain v3 spec (grilling, six decisions)
2026-07-27  48cf011e  harmony v3: three-line corpus pipeline (upper/lead/lower triples)
2026-07-27  dcb41757  harmony v3: dual-track trainer (joint vs serial), overnight run launched
2026-07-28  b8e4a86a  harmony v3: trio render inference + blind pack (joint/serial/v1 cells)
2026-07-28  879d90f9  docs: sync-state — v3 trio render + blind pack (worklog 07-28 §A, STATE)
2026-07-28  8c63fee3  direct_mouth: on-demand portamento (fix pervasive glide, 07-28 ear feedback)
2026-07-28  1aeb9494  docs: sync-state — on-demand porta fix + blind pack repack (worklog 07-28 §B)
2026-07-28  677acb08  render_v3: independence knob (suspension/passing-tone bias, joint track)
2026-07-28  82151e7d  docs: sync-state — round-2 verdict (joint wins, serial dies G33) + indep knob (worklog 07-28 §C)
2026-07-28  3b09d54e  render_v3/direct_mouth: stability knob + vibrato desync (07-28 round-4 feedback)
2026-07-28  ab109153  docs: worklog 07-28 §D — stability knob + vib desync round
2026-07-28  f0b4cb4e  direct_mouth: shift f0 mode — his real pitch contour, transposed (autotune root hunt)
2026-07-28  e2050530  docs: worklog 07-28 §E — shift f0 mode round
2026-07-28  8106fb88  docs: worklog 07-28 §F — shift dies (F16 holds on direct carrier), autotune ablation ladder
2026-07-28  4aa778b3  direct_mouth: texture f0 mode — target skeleton + his real micro-intonation
2026-07-28  3bd8646d  docs: worklog 07-28 §G — two-layer verdict + texture mode
2026-07-28  01bbaec9  docs: worklog 07-28 §H — texture dies, autotune floor = data ceiling, line-side closed
2026-07-28  642b081e  live_v3: two-angel live host (v3 brain + dual direct-drive DDSP mouths)
2026-07-28  3ad94107  docs: worklog 07-28 §I — live_v3 first block
2026-07-28  bd6a2588  live_v3: first-phrase key lock (keydet) — angels enter after the key locks
2026-07-28  1d81add0  docs: worklog 07-28 §J — live key lock
2026-07-28  4c0f901a  live_v3: anticipation mode — angels land WITH him (responsiveness feedback)
2026-07-28  436ba188  docs: worklog 07-28 §K — live first-wear feedback + anticipation revival
2026-07-28  175f6d9b  docs: sync-state — WO-5 盒體加高 40mm＋隔板精簡、exciter 黏著選型（worklog 07-28 §K/§L, STATE）
2026-07-28  c1eb073c  docs: worklog 07-28 — CAD/exciter 兩節改編號 §K/§L → §L/§M（與 live 線 §K 撞名）
2026-07-28  d7de3e99  docs: STATE — 交叉引用跟上 worklog 改編號（§K/§L → §L/§M）
2026-07-29  98c602c0  firmware/pico_led: unit2 實測定稿三參數 — MAX_LEVEL 0.30 / SPEED_MIN 100 / NOISE_FLOOR 3000
2026-07-29  22ba549e  docs: worklog 07-29 — unit 2 bring-up 全記錄＋STATE 同步
2026-07-29  8998b201  docs: unit 1 待推同版韌體＋迴授問題邊界釐清（worklog 07-29 §G）
2026-07-29  a9879436  docs: unit 1 韌體同版完成 — 兩 unit 皆 98c602c0
2026-07-29  84dab084  docs: worklog 07-28 §L — anticipation A/B round 1: lag 0.45 dies, latency floor is the mouth
2026-07-29  be4435ce  docs: worklog 07-28 §L — live anticipation hit rate 11% confirms G29 first-pass baseline
2026-07-29  2e77346c  live_v3: starved 計數改在播放頭判定，進場不再誤報斷供
2026-07-29  5323834a  harmony: rehearse_ab offline A/B scaffold for rehearsal mode
2026-07-29  65a15d38  live_v3: warm up brain + both mouth models before audio starts (MPS cold-start)
2026-07-30  261c0725  harmony: rehearse_ab ScoreTracker v2 — speed prior, lost/re-entry, shared octave
2026-07-30  9f2dcf84  harmony: loopback_lab.py — 自我迴環三格梯（原聲 / 真實 f0 注入 / 規則渲染 f0）
2026-07-30  07e7b710  direct_mouth: voicing gate — 無聲幀注入 f0=0，不再把吸氣/擦音唱出來
2026-07-30  b2ef0a55  direct_mouth: native-uv 注入路徑 + 多特徵合議 uv + 信心遲滯邊界
2026-07-30  b7f84d21  direct_mouth: uv 段源透傳——呼吸/氣音用他的原聲，不再合成
2026-07-30  d8da6cbe  direct_mouth: gate 錨定＋局部自適應門檻＋分窗透傳增益（修「12 秒後吸不了氣」）
2026-07-30  871eb800  direct_mouth: 輸入端 AGC——轉換前拉到訓練參考位準、轉換後還原
2026-07-30  819aa9d2  direct_mouth: gate 錨定改連鎖式——收乾淨起音前的垃圾 f0（金屬音根因）
2026-07-31  25318ce4  docs: 07-30 收官 checkpoint（worklog/STATE/rehearsal spec）
2026-07-31  79bb1adc  direct_mouth: f0-mode express——規則版表現層（兇手 (b) 的藥）
2026-07-31  8a49869f  harmony: rehearse_ab ScoreTracker v3——貝氏 forward filter（多假設）
2026-07-31  27b68d81  docs: 07-31 工程側兩案收官（表現層 express＋追蹤器 v3）——STATE/worklog
2026-07-31  01590bdc  docs: stim_3e 盲聽過關——兇手 (b) 結案、autotune 案收官（F21）
2026-07-31  ec1827c2  live_v3: 實戴收官儀器——mic dump＋命題一膝點分析
2026-07-31  645be192  world_live/live_v3: 表現層移植 live——ExpressiveF0 streaming 版（F21）
2026-07-31  b765fe3d  docs: 07-31 晚實戴場工程準備二連——worklog §D/STATE
2026-07-31  d55732c1  live_v3: file mode 音符線落檔＋--notes-json 重放——單一變因 A/B
2026-07-31  5e4357e6  live_v3: --out-map/--dry-ch per-voice 出力路由（F11 物理源分離上 v3）
2026-07-31  560c576f  live_v3: 排練模式緊檔 v1——ScoreTracker v3 對位、天使唱排練的線（spec 07-29）
2026-07-31  a8aee9f2  docs: 07-31 深夜工程加班二連——worklog §E/STATE
2026-07-31  b4c7b6ed  harmony: lab.py——live_v3 單檔 web 實驗台（測試 app v1）
2026-07-31  30181300  docs: v3 lab 測試 app——worklog §F/STATE
2026-07-31  183e53ad  lab/live_v3: 首玩三修——空路徑、keydet 警語、live 即時成本遙測
2026-07-31  036a3d21  docs: lab 首玩回收＋合成 live 測試兩發現——worklog §G
2026-07-31  ab524ee0  live_v3: ticks/hops 拆執行緒——腦爆預算不再拖死嘴（Regime B）
2026-07-31  795e721f  docs: 拆執行緒 Regime B 收官——worklog §H/STATE
2026-07-31  ccafe9b0  live_v3/world_live: 呼吸被唱案——DDSPMouth uv 出口遮罩（Regime B）
2026-07-31  81fb33ae  docs: 呼吸被唱案定罪與治癒——worklog §I/STATE
2026-07-31  74827444  live_v3/world_live: 嘴腦解耦 free-run——lag 0.45 翻案（Regime B）
2026-07-31  713d3473  docs: 嘴腦解耦 free-run 收官——worklog §J/STATE
2026-07-31  c5f58b30  world_live/live_v3: free-run v2 單流化＋keydet 拒鎖垃圾（Regime B）
2026-07-31  bd327fc7  docs: lab 二玩回收——worklog §K
2026-07-31  2eb1ec72  live_v3/lab: --hop-ms/--bound-ms 低延遲旋鈕＋lab 進階參數欄（Regime B）
2026-07-31  db36a95c  docs: 低延遲旋鈕掃描——worklog §L
2026-07-31  b697d5d4  world_live/live_v3: units 共享＋hop 分段遙測——lag 0.30 進實驗檔（Regime B）
2026-07-31  fb43b61e  docs: units 共享收官——worklog §M
2026-07-31  0379bce5  live_v3: live 預設換用 260730 重訓嘴（Harry 拍板「換換看」）
2026-07-31  5affcb2c  world_live/live_v3/lab: 波波聲＝接縫能量凹陷——PV 交叉淡化＋Stop 失靈修（Regime B）
2026-07-31  61f462f1  docs: lab 三玩回收——worklog §N
2026-07-31  36d73079  world_live/live_v3: StreamMouth 真串流嘴——相位續接、零 SOLA（Regime B）
2026-07-31  c2378f76  docs: StreamMouth——worklog §O
2026-07-31  49caf9a6  live_v3: io 旗標遙測＋latency high——「重複輸入」假說的法醫儀器（Regime B）
2026-07-31  572fe177  lab_ui: 狀態列加 IO 丟失格（Harry「IO 在哪裡」）
2026-07-31  417280f9  lab_ui: 狀態列 9 格 grid 修正
2026-08-01  f5cc96fe  hardware: 展示結構七迭代收官——一件式純梯形錐體 vu2_pyramid
2026-08-01  d0ec0438  live_v3/world_live: 波波聲結案——uv 遮罩抖動定罪、live 預設關閉（Regime B）
2026-08-01  ef0db77d  docs: 波波聲結案——worklog §P
2026-08-01  8c2028ad  harmony: 排練預渲染立線——prerender_stems 離線 express stems＋live_v3 --pre-stems 薄對位層（離線品質上 live，Regime B；worklog 08-01）
2026-08-01  6403cf40  docs: STATE 08-01 排練預渲染段（順帶落盤硬體 session 掛單的 WO-5 加高一行）
2026-08-01  52301fa5  harmony: reh_polish 排練檔譜面後處理——縱向持續不和諧修協和（18.0→12.3% 貼基準；worklog 08-01 §F–§G）
2026-08-01  5e3d68ad  docs: worklog 08-01 §H 調性案——pair06 實為 A 大調、吸附循環論證血訓
2026-08-01  55833fea  docs: worklog 08-01 §I 調性案終判——A 大調確立、pair06 結案、NEXT 應答式 v2
2026-08-01  364c98f5  docs: STATE 08-01 補三段判決——B/P 判死歸因對位、組成層工具鏈、調性案 A 大調結案
2026-08-01  d2aa8796  harmony: 應答式 v2 落地——phrase_render 行程內常駐渲染器（6.5× 快、品質等價）＋respond2 應答主體（worklog 08-01 §J）
2026-08-01  6e6fd705  firmware+harmony: 應答式實體鍵 Pico 路線——pico_button 韌體（UDP 廣播＋雙模開機）＋ tap_listen 接收端（selftest PASS；worklog 08-01 §K）
2026-08-02  5ef0be8e  hardware: 穿戴新方向打樣——auxetic 網布×剛性板漸變 coupon
2026-08-02  ad1136db  docs: worklog 08-01 §L-§M 展場前端轉向（空氣麥克風）＋麥克風盤點＋明日優先序
2026-08-02  7d6f8171  docs: STATE 08-01 補應答式 v2 落地、實體鍵線、展場前端轉向＋明日優先序
2026-08-02  c7bdeab5  respond2: live 模式＋忙碌窗硬靜音（空氣麥克風路線的地基）
2026-08-02  7918ca9d  respond2 live: 斷句門檻開場自動校準＋實體鍵接上
2026-08-02  789eac81  docs: STATE 08-02 應答式 v2 上 live（硬靜音／開場校準／實體鍵）＋worklog
2026-08-02  bf1e23cc  key_probe（真歌定調儀器固化）＋respond2 --octave（釘住 auto-octave）
2026-08-02  939f2303  respond2 --expr：把表現層開成旗標（08-02 量到它是「怪」的頭號嫌犯）
2026-08-02  37b82bfe  表現層拆兩半：兩天使共用音準參考（--tune-lock），音程誤差 25.8→3.4c
2026-08-02  aaf53eb1  respond2 --f0-mode target：把 blind_v3 的嘴接回應答式
2026-08-02  6dcd5616  --f0-mode target 翻成預設；F21 規則修正＋08-02 worklog/STATE
2026-08-02  258aad3e  prerender_stems: --f0-mode target/express ＋ --ceiling（08-01 歸因重審）
2026-08-02  a212ab73  docs: 08-02 §H 重審結果——ceiling 平反、對位版維持判死
2026-08-02  c22824b8  respond2 live: 實體鍵連上時，能量斷句退居安全網（--gap-tap）
2026-08-03  ea54bd14  docs: 08-02 論文素材——四張圖（driver 固化）＋五段可改寫的草稿
2026-08-03  2230d3d1  docs: 08-02 §I 首次真人實測（K669B）——三個發現
2026-08-03  b5c70e66  sync-state: STATE 補完 08-02 後半（重審／實體鍵／真人實測／論文素材）
2026-08-03  e847cf49  sync-state: NeurIPS 截止 8/3→8/10 AoE，並記下 non-archival 與雪梨到場的代價
2026-08-03  8d1bea36  兩張嘴共用「只看 x」的特徵：每句 harvest 4 次→1 次，RTF 0.47→0.23
2026-08-03  3f24b098  管線化：他還在唱的時候就先算，等待 p50 4.47s → 2.16s（9.8s 的句子也只等 2.4s）
2026-08-03  4c3f98aa  開場校準加有聲提示：站著唱的人看不到 console
2026-08-03  933dafe5  docs: 08-03 管線化——worklog、STATE、FINDINGS F22
2026-08-03  bb81c4c8  docs: 「輸入 SNR 對天使音質」結案——素材是 08-02 那支 252s dump，時序確認跑的是 target
2026-08-03  3aa2d683  殘響：合成 IR 的卷積殘響（預設關，等 Harry 耳測）
2026-08-03  d695a385  prints: mesh coupon 換 auxetic 家族——v3/v4/v5＋無板對照組
2026-08-03  cbc2c48d  sync-state: 08-03 coupon 診斷與展籤二校
2026-08-03  1b9859e3  gitignore: 聽測 stimuli 不進版控
2026-08-03  842537c0  殘響與混音平衡定案（Harry 耳測四輪）＋尾巴截斷＋削峰警告
2026-08-03  86caf9a4  docs: 08-03 §G 殘響與混音平衡定案（耳測四輪）
2026-08-03  aea96738  score_notes：阿卡貝拉 MIDI → 三聲部 tick 網格（譜這一側，400 個真編曲 93% 吃得下）
2026-08-03  d385ac66  score_align：他唱的那句 → 譜上的位置（離線 DTW，和聲一致率 87%）
2026-08-03  3484dbbb  perform：演出模式（伴奏帶＋他唱 lead，同一個空間）＋即時卷積殘響
2026-08-03  b3713289  auto-octave 改用校準唱段鎖（舊的只差 2 個半音就翻八度）
2026-08-03  11198dfb  docs: 08-03 §H MIDI 和聲／演出模式／auto-octave
2026-08-04  a92336f5  修 polish 的音高空間 bug：真實調上的不和諧率 29–33% → 10–15%
2026-08-04  c2d8dcf4  polish 改對「他真實唱的音高」判協和：錯調的傷害 37.7% → 17.9%
2026-08-04  7083202c  --key auto：逐句用 raw-f0 音級分布重估旋轉角度
2026-08-04  8e8c7be7  殘響早期反射改擴散簇：離散 taps 把他的開口子音複製成 slap echo
2026-08-04  af35b0ec  體檢後兩修：stab 2→1（錯開 562ms→0）、uv 透傳 respond2 預設 0（+7.7dB→+0.2）
2026-08-04  d0e1461c  預設值改成 08-03 深夜實戴拍板的組合（dry 0.944／天使 0.75／女聲 +1／indep 0.15）
2026-08-04  a0a717f5  docs: 08-04 凌晨場 worklog＋STATE（兩 bug、體檢兩修、三條方法論血訓）
2026-08-04  b68c7c6c  reh_polish CLI 補同空間換算＋08-01 polish 數字重審（兩掛起項收案）
2026-08-04  75a390dd  C 案分離度量測管線：clap_probe.py（rise×hf 二維判準）＋worklog §H＋STATE
2026-08-04  8ad07250  docs: 撞號修正——C 案分離度段 §H → §I（同日兩場並行寫入）
2026-08-04  de1928fa  STATE：C 案段落引用跟上撞號修正 §H→§I
2026-08-04  469f6034  live 終判過關（stab1／key auto）＋dump 改關流前落檔（CoreAudio 掛死救援）
2026-08-04  adb22998  應答式 app 介面：respond_shell＋respond_live.html（08-02 §I 血訓的正解）
2026-08-04  4b9b77f8  docs: worklog §K＋STATE——應答式 app 介面
2026-08-04  a19f29be  respond2 --out-map 分路（dry/上/下天使各自聲道）＋UI 英文版
2026-08-04  6ddde5dc  docs: worklog §K 二輪——英文 UI＋out-map 分路
2026-08-04  f613161a  實體鍵 A 案定案＋USB serial 整條鏈驗通：main_usb.py＋SerialTapListener（C 案退場，worklog §L）
2026-08-04  7cb5f658  UI fix: Stop 按鈕永不出現——style.display='' 掉回 CSS 的 none
2026-08-04  6b1f5301  docs: §L 和諧掛起格收案——16:00 有調 dump 審計 11.2%＝優於語料；互撞 9.6%＝stab1 代價在真歌上消失
2026-08-04  15e386e8  docs: 撞號修正——實體鍵晚場段 §L → §M
2026-08-04  7385d665  docs: STATE——選歌方向定案（自寫主曲＋合唱段，三首備選）＋NEXT 同步 C 案否決
2026-08-04  7494c591  VocalSet male3 試訓：14k checkpoint＋耳測樣本（worklog §N）
2026-08-04  aa5e524c  GRAVEYARD G33: VocalSet 練習語料轉真歌判死——死因＝詞素材庫存（跨歌手鑑別定罪）
2026-08-04  8af8d512  docs: §N 續——跨歌手鑑別定罪、G33、轉進 GTSinger ZH-Tenor-1
2026-08-04  a5a1c1cb  docs: 08-04 §O 實體鍵殼建模出圖＋焊槍盤點＋gcode PLA 合規重切
2026-08-05  6dad2806  docs: §N 續 2——M4Singer Bass-1 訓成（11k val 0.974）、耳測樣本備妥
2026-08-05  570dffba  prosody 移植 prototype：phrase_render prosody_cents＋A/B/B2/C 樣本
2026-08-05  77f8eafe  docs: §O 判決——prosody A/B/B2/C 全數不過；下一假設 D＝連續 f0 直接移調
2026-08-05  75d1ae3e  prosody D 版：連續 f0×音程比（含空間換算修正）——幀間活動 10.5c 最接近人
2026-08-05  f2d2ccb0  E 版：天使歌手化——短 uv 空隙橋接（sustain_ms 守衛參數）疊 D 連續 f0
2026-08-05  bee6afdd  docs: §O 判決 2——D/E 亦不過、交接下一 session（剩餘嫌疑＝合成器層，建議裸測鑑別）
2026-08-05  84df1b42  docs: STATE——autotune 案置頂交接
2026-08-05  190b91ff  docs: 全面 review 報告落檔（7 bugs＋守衛驗證全過）＋STATE 指標
2026-08-05  26390b14  autotune 案：合成器單一變因裸測（bare_test.py，08-04 §O 判決 2 的下一刀）
2026-08-05  59b8619d  autotune 案：裸測判未過（f0 曲線層全滅）→ 正典 identity 對照渲染
2026-08-05  93e3d4ac  harmony: respond2 實體鍵改接 SerialTapListener（USB serial）
2026-08-05  f842bbe5  docs: sync-state——實體鍵接線收尾（worklog 08-05 §C）
2026-08-05  9647bf92  docs: 實體鍵按壓導通測試過（TAP 即時、序號累積正常）
2026-08-05  9a2babe0  autotune 案素材翻案：舊選段選到回應段（三聲部混音）＝A–E/裸測/正典作廢
2026-08-05  439e1abc  worklog 08-05：實體鍵段撞號 §C→§D（素材翻案保留 §C）
2026-08-05  6c3a5e3e  harmony: gap-tap 預設 2s→6s（Harry 08-05 定）；修 --help 裸 % 崩潰
2026-08-05  123186f6  docs: STATE 補 gap-tap 6s 裁定
2026-08-05  ed41d5c9  autotune 案破口：病在音色層（CombSub 裸輸出），enhancer 守衛式入 phrase_render
2026-08-05  b5d4c59a  autotune 案收案：Aenh 過→respond2 接線（Bass-1 lower＋enhance=True）
2026-08-05  24cda139  Soprano-3 訓成（val 谷底 10k）＋耳測樣本；worklog §G 續
2026-08-05  1808a05b  S 入應答式 live：respond2 四聲部（--sop 守衛、預設開）＋實體鍵可測
2026-08-05  a4b852ab  韌體合併：火流 v5.1＋實體鍵一板同跑（非阻塞去彈跳）；worklog §I
2026-08-05  c3edb52e  worklog §I 驗收：火流＋按鈕全鏈通
2026-08-05  95675f8b  worklog：Soprano-3 過耳＝SATB 四聲部到齊
2026-08-05  92c53f12  STATE：SATB 到齊＋live 全鏈通
2026-08-05  f15cf3e4  app：reverb（悠遠）＋tail s（悠長）兩顆旋鈕進介面
2026-08-05  b1dfb084  Alto-5 訓成＋girl A/B 樣本；worklog §J
2026-08-05  1ad47a39  worklog §J 判決：平手＝girl 續任、Alto-5 認證備胎
2026-08-05  231d667a  混音：Bass-1 下聲部 +1.5dB（Harry 聽 satb_4part 定；--lower-db 預設 0→1.5）
2026-08-05  30ae2d2a  應答 UI：中央狀態文字加入按鈕提示（三態：收句/作廢重唱/中止回應）
2026-08-05  aef9bff5  應答 UI：mic/out 改下拉選單（引擎 venv 即時查裝置表；AI-Micro 預選）
2026-08-05  460417b5  應答 UI：map D/U/L/S 改四路聲道下拉（選項隨 out 裝置聲道數；留—＝不分路）
2026-08-05  e9a10209  燈條靈敏度：NOISE_FLOOR 3000→2500（Harry 要更靈敏；對實測雜訊 2100 留 400 餘裕）
2026-08-05  c4d55207  sustain_live v0：長音觸發同步 live 原型（G28 案②；S+B 八度、enhancer 同款音路；兩嘴塊渲染實測 301ms）
2026-08-05  2692c1d3  GRAVEYARD G33b：batch 長音觸發 live FAIL（延遲地板＋回授）；下一注＝DDSP 滑窗串流 spike
2026-08-05  c332c2ce  worklog §K：batch 觸發陣亡記錄＋滑窗串流 spike 立案
2026-08-05  2f8a881a  STATE：即時線 spike 立案＋混音/旋鈕/韌體收尾同步
2026-08-05  f8e81b53  spike_stream：DDSP 滑窗串流 spike（gui.py 核心 headless 化、MPS；bench infer p95 39ms/1.0s 視窗＝block 0.10s）
2026-08-05  0153c3eb  spike_stream：接縫改 phase vocoder（gui.py 原方）治跳針
2026-08-05  a6b24b19  spike_stream v2：units 凍結快取＋512 對齊 block（hubert 全窗自注意力＝滑窗重寫 units p50 0.38＝跳針根因；每幀窗尾兩塊修訂一次後凍進絕對網格）
2026-08-05  269bf8ab  spike_stream v3：和聲層上線（S=sop3+12/B=bass1−12、共用 encoder/f0/enhancer；93ms/197ms 塊）；--solo 保留單聲部檢驗模式
2026-08-05  4f1f58e4  spike_stream v4：每聲部各自 SOLA＋接縫（混音 SOLA 對不齊兩個週期＝和音跳針根因）
2026-08-05  c22a36c6  spike_stream v5：f0 凍結網格＋固定 shift=0（拔掉 SOLA 搜尋的相對時序抖）
2026-08-05  6360eda3  spike_stream v7：全特徵凍結＋每聲部 SOLA 回歸（儀器：對齊後 corr 0.959＝縫只剩相位、SOLA 本職）
2026-08-05  a04af217  spike_stream v8：音量門檻 -45→-60（症狀重定性＝斷斷續續＝門檻切輸出，非接縫）
2026-08-05  aaed9527  worklog §L：串流 spike v1-v8 診斷鏈（solo 過耳＝核心成立；和聲斷續留下場）
2026-08-05  4b234bbf  STATE：串流 spike 戰果與斷續留局同步
2026-08-05  98272c8d  spike_stream v9：推論移出音訊 callback（v1-v8 斷續真兇＝110ms MPS 工作在 CoreAudio 死線上；callback 只搬記憶體、worker 執行緒推論、status 旗標入帳）
2026-08-05  a32b0211  即時和聲串流成立（Harry 終判 v9 乾淨）；血訓入 worklog §L＋DEBUG_PLAYBOOK
2026-08-05  655869b0  STATE：即時和聲串流收案
2026-08-05  1ee1e7b9  spike_stream v10：diatonic 腦上線（--key 大調映射；S+7步/中聲部=他的模型−2步三度下/B−7步；量化只決定音程、曲線是他的＝D 版哲學；150ms/197ms 三聲部）
2026-08-05  462d96e5  spike_stream v11：長音穩定包（遲滯音符追蹤 0.65半音/35ms、f0 到幀即凍、音程進快取＝相位帳精確、crossfade 0.08）
2026-08-05  26837887  spike_stream v12：長音 units EMA 釘住（streak≥70ms→α0.12；換音α1）＝「唱嗚不再漂成別的母音」
2026-08-05  b47a2138  spike_stream v13：天使腦接入（EarV3 tick 流獨立執行緒；girl=上聲部線/bass1=下聲部線/S=+8ve；KeyTracker 自動追調；--no-brain 退回 diatonic）
2026-08-05  4cdbea3b  spike_stream v14：腦音符直饋（tracker.push 的 pitch 重算全割＝+55ms 歸零；keylock 外部化；block 回 0.20）
2026-08-05  6a0f2101  worklog §M：v10-v14 串流進化史＋上下文機制路線圖
2026-08-05  ecdcf49c  STATE：串流 v14 腦接入＋上下文路線圖
2026-08-05  0755a3fe  應答 app 整合即時天使模式：mode 下拉（應答/即時）→ spike_stream 子行程、串流狀態/缺塊入 UI 遙測
2026-08-05  c9f0f64f  worklog §M 續：即時模式入 app
2026-08-05  159dfed7  worklog：即時模式 app 終判過＝08-05 收工
2026-08-05  2a24b780  docs: worklog §N K669B 夾環殼 v1→v3（實測筒徑 47.0）＋STATE 同步
2026-08-05  e43b5029  spike v15: 樂句橋接（§M③ E 版 sustain live 化；--sustain 守衛預設 0）
2026-08-06  473424ef  spike v16-v18: 橋接判準三迭代＋review bass 三病灶（--sustain 守衛）
2026-08-06  c5582215  spike v19: 腦線抖動雙修（音程遲滯＋k_shift 鎖）＋dump 診斷儀器
2026-08-06  826ae265  spike: enhancer 三聲部合批（§M 刀位；等價性過 MPS 自噪地板、實測 −12ms）
2026-08-06  6c2bbd4c  spike v20: bass −1 破案＝吸附差補償（腦 feed 層；girl 音程散同顆 bug）
2026-08-06  5b9318e0  docs: worklog 08-06（四刀紀錄）＋STATE 同步
2026-08-06  b5fc5a8c  spike v21: --floor-db 音量地板（治喉麥垃圾幀觸發怪詞怪音）
2026-08-06  ec63d0d2  spike v22: --replay 離線重播＋--agc 串流 AGC（皆 guarded 預設關）
2026-08-06  f1174fd1  spike: 診斷旋鈕三件（--dump-stems 逐聲部 stem／--ema-alpha／--ema-streak；預設全＝原行為）
2026-08-06  9048ab8c  docs: worklog 08-06 §E 晚場（v21/v22、迭代迴圈、聽單交棒）
2026-08-06  ca7df7f4  spike v23/v24: enh-tail 翻案上線＋EMA-CPU＋rehearse 譜模式接線（皆 guarded）
2026-08-07  96a6bf25  spike v25/v26: lookahead 判負＋enc-win 左上下文＝咬字主兇的藥（guarded）
2026-08-07  b130b842  docs: worklog 08-06 §F 補記（v23–v26 從 commit 回填）＋STATE 同步＋論文貝多芬查證素材
2026-08-07  3ff7f931  docs: worklog 08-06 §F（咬字主兇＝hubert 窗長；enc-win 候選版數據）
2026-08-07  bd535c68  docs: §F 收官（enc-win 3 甜蜜點；block 0.12 live 判死；聽單 E 候選）
2026-08-07  87f0316e  docs: §G 事故記錄（symlink 穿透蓋檔＋復原）＋戰場轉移至 reflow
2026-08-07  a59a65ad  docs: §H reflow 天花板成立＋bass1 訓練點火＋聽單_reflow
2026-08-07  ea6bca20  docs: §H 補記（ODE 步數掃描＝即時可行）
2026-08-07  681f49b5  docs: Harry 裁 reflow D 過/E 壓喉嚨；DNSMOS 音域盲點血訓
2026-08-07  d3a2358e  docs: STATE 同步（reflow pivot＋bass1 訓練中＋watcher）
2026-08-07  428530d2  docs: §I bass 過耳＋girl/sop3 訓練佇列啟動
2026-08-08  1e76c2cb  docs: §I 收工＝三張 reflow 嘴到齊＋樣本記分
2026-08-08  1fccf363  spike_stream6: reflow 嘴版滑窗串流跑通（v26 機制全繼承、只換嘴）
2026-08-08  f1652ba5  docs: §J spike_stream6 跑通紀錄
2026-08-08  343911ab  spike6 v27: --bridge-decay／--sop-db＋§K 氣音破案（sop3 訓練不足）
2026-08-08  e64c357c  docs: STATE 同步（spike_stream6 主線／sop3 續訓／live 指令與血訓）
2026-08-08  f44aee21  spike6 v28: --t-start／--seed 旗標＋移除空的 --no-enhance；mic_check 收音體檢
2026-08-08  8cc1b79e  spike6 v29: --bridge-hold 治「唱到一半被切」＋--f0 抽取器旗標
2026-08-09  22b6a945  docs: mesh 後處理入檔（§8a 接合/收邊）＋08-06 §L 氣音三連翻案＋STATE 同步
2026-08-09  43802c83  spike6 v30: --bridge-units（橋接期音量還在就不凍 units）＋dump bu
2026-08-09  cce45bfa  spike6: --prof 分段預算（MPS synchronize；預設關）
2026-08-09  717e5a1e  spike6 v31: --drop（真的拿掉聲部，不是把增益歸零）
2026-08-09  3d8bf89d  docs: 08-09 Track A 入檔（worklog §F–§J＋STATE 同步）
2026-08-09  b7df6d09  spike6 v32: --voc-tail（只 vocode 尾段；預設關）
2026-08-10  f4bd60a7  docs: 08-09 深夜場入檔（worklog §K–§O＋STATE 同步）
2026-08-10  017338d1  docs: 08-10 場入檔（回授三翻案＋AEC 平反＋DTD＋音質三鍋＋G34-36）
2026-08-10  4bd2f000  spike6 v33: --noise-grid 凍結噪聲網格（預設關；Regime A 證 0.0）
2026-08-10  1e407274  spike6 v34: --sola-cont / --f0-mature / --seam-amp ＋ SOLA shift 儀器（皆預設關；Regime A 證 0.0）
2026-08-11  69ee0d53  spike6 v35: per-voice spk_id（多人模型）＋ girl 槽換裝 M4Singer Alto-6
2026-08-11  9e8d21a2  spike6 v36: --units-mature 轉換幀成熟刷新（預設關；Regime A 證 0.0）
2026-08-11  ccb6aaf8  score_live v1: 天使照譜唱、他當指揮（獨立腳本，不動 live 主路徑）
2026-08-11  dc8ee115  docs: 08-11 STATE 同步（alto3 驗收/配方轉正/score_live/嘴部感測器）
2026-08-11  9b770574  score_live v3–v4: 順序模式＋嘴部門控＋tutti 段表＋佇列紀律（08-11 live 四輪機械過；音樂性總評未收）
2026-08-11  9880a20d  08-11 晚場：score_live v7/v7.1（依賴反轉＋stems 播放）、spike_stream6 v37（--aah/--pad/--swell/--wet）、respond_live 新增、respond2 --aah、50ms spec
2026-08-11  87e6a736  solo_min 優化 #1+#2：接上 f0_override（Task 2.4 遺留接口）＋ xrun 報告移出 audio callback
2026-08-11  13c78fd9  respond2 space 鍵：serial 沒插退回 UDP tap，app space＝實體鍵（focus 才算連線）
2026-08-11  4706e5ed  bank_live v3.2：G 音庫取樣器（音高空間預渲＋嘴部門控＋人味離散＋悠遠層；Harry 判 ok）
2026-08-11  eee14523  bank_live v4（voice-leading＋portamento）＋STATE 08-11 深夜同步＋Beatrice×M4Singer 前置計畫
2026-08-11  2ca4997e  bank_live v5：--file 離線台架＋已知音符拒斥（第二道回授防線）
2026-08-11  349e1c12  bank_live v5.1：錯落起音（三聲部起音時值各異＝進場綻開）；台架驗過
2026-08-11  fe33079d  bank_live v6：快慢層合體＋搶拍
2026-08-11  9b683cd7  bank_live v7：立體聲站位（成分四 lite）＋呼吸預備窗
2026-08-11  bd3d3ca7  bank_live v8＋melody_lm v0：第 2 階地基——Harry 旋律習慣模型接進搶拍
2026-08-11  d67f80d0  G29 補註：melody_lm/v8 部分重開（資料換他自己＋用途降級為 commit 加速）＋血訓自記
2026-08-12  1fdee1bd  bank_live v10：穩定性總 review 修復場（雙 fresh 審查、S 級全修）
2026-08-12  009962af  bank_live v12：R2 雙審修復（拒斥層重設計＋DSP 等功率/整數週期迴圈）
2026-08-12  95233e88  bank_live v13：R3 終驗收斂——拒斥層整層移除（實測假保護）＋鏡頭看門狗
2026-08-12  343b8ea4  respond2 系穩定性修復場：三 fresh 審查 + 修 13 條 S/A + fresh 驗收
2026-08-12  adc0a1dc  bank_live v15.1：R4 雙審修復（看門狗盲區/法醫記生效值/合理性門/porta 相干縫）
2026-08-12  2218504f  bank_live v15.2：R5 認證收官（大跳三票/看門狗貼齊/法醫 I1 正確不變量/view 10fps）
2026-08-12  8ce7b4e5  docs：五輪 bank_live review 報告＋低延遲調查＋solo_min 盤點入庫（原在 session tmp）
2026-08-12  60d206b1  docs(STATE)：雙線續跑場——Colab 8000 步跑完但 ckpt 全滅、v2 斷線免疫 cells 備好；A 線耳測 runbook 入地圖
2026-08-12  1b7e2b5e  docs(STATE)：A 線判決場收官——v15.2 預設全過耳零改動；新尾巴 SIGINT 收場卡＋idle xrun 爬升
2026-08-12  30bc7693  bank_live：SIGINT 無條件裝回 default handler（背景 & 啟動繼承 SIG_IGN＝收場全聾、dump 陪葬；08-12 r3/r4 實案）
2026-08-12  ef498b67  docs(STATE)：兩條判決場尾巴結案——SIGINT=啟動方式鍋已修、xrun 爬升=本底非蟲
2026-08-12  c4bc6cc2  bank_live：--frame-b64 N 旗標（嘴部畫面 base64 出 stdout 給 shell UI；預設 0=行為不變）
2026-08-12  57f3225c  respond app：即時模式換 bank_live＋嘴部畫面進 UI＋第三模式（應答＋即時並行）
2026-08-12  069bd939  respond app：審查修復 S1/S2/A3/A4/A5/B6/B7/B8（fresh opus 審查，假行程台架 4 案 PASS）
2026-08-12  133271ed  docs(STATE)：viva 整合＋審查修復收官；多母音庫轉明暗軸設計
2026-08-12  0143915e  bank_live v16：多母音五層（--vowels，預設關＝現版原樣）——0619 開採素材、mic tilt 最近鄰選層＋遲滯、換音時生效、快取分目錄可退回
2026-08-12  4e53407e  docs(STATE)：選層四版追兇收官→v17 連續母音場定案（Harry 第一性定調）
2026-08-13  f4570fca  bank_live v16.2/16.3 選層基底存檔（聲音指紋＋嘴形二維最近鄰；範式已判死、v17 改建前檢查點）
2026-08-13  18aa53cf  bank_live v17：連續母音場——嘴形反距離權重逐 hop 混五層（Harry 第一性定調）
2026-08-13  3599eb33  bank_live v17.1：整圈嘴特徵（Harry「為什麼不測整圈嘴」）＋校準工具
2026-08-13  1f9ffe4d  bank_live v18：個人母音模型當混層權重（405 參數 logistic；ML 判準實驗附證）
2026-08-13  62228d89  bank_live v19：融合母音場（整圈嘴 80D＋MFCC 13D）——補唇的物理盲點
2026-08-13  18453f3a  bank_live v20：--layers 母音子集＋--vw-tau 權重平滑（Harry「轉換變來變去不好聽」）
2026-08-13  452de2ee  bank_live v21：音庫依母音重挑＋渲後驗收（喔判死＝模型吐不出圓唇後母音）
2026-08-13  127b3359  layer_vowel_audit：層庫目錄改用 mtime 挑最新（＋--dir 指定）
2026-08-13  58602005  bank_live v22：改用「渲後」挑素材，四層全驗過；喔正式判死
2026-08-13  10bdf953  bank_live v23：挑素材加音質判準（HNR），四層全面超越 v21/v22
2026-08-13  96adf937  bank_live：母音層改回 v21 那組（Harry 耳裁），--layers 2,3,4＝咿/嗚/啊
2026-08-13  54697bbd  bank_live v24：音高遲滯死區＋回跳窗參數化（Harry「轉音會上下上下」）
2026-08-13  099baed2  bank_live v24.2：死區只保護『剛換過去的音』（Harry「稍微遲鈍」的解）
2026-08-13  057e9e8a  bank_live v26：死區重做（三審查判死 v24）＋滑音斜坡＋延遲儀器＋假八度補洞；含並列 session 的引擎全英文化
2026-08-13  85cabf78  bank_live v27：開口門檻參數化＋dump 記開口值（Harry「微小張嘴就觸發」）
2026-08-13  ffaa2161  bank_live v28：嘟嘴保持（唱嗚不再被門控切掉）＋門檻用實測定，不用猜
2026-08-13  ec40c119  bank_live v29：監看畫面改畫整圈嘴唇＋十字（Harry：一條線很奇怪）
2026-08-13  8c860b05  bank_live v30：門控改用 mediapipe blendshape（Harry：有沒有更聰明的方式）
2026-08-13  69f86c60  bank_live v30.1：pucker 門檻 0.60→0.75＋開門連 2 幀防抖（Harry：閉著嘴還是會觸發）
2026-08-13  77c44edc  bank_live v31：--tenor 多開一個男聲部（Harry：現在是不是只有 bass）
2026-08-13  3d5e7ed3  bank_live v32：--balance 依實測響度拉齊聲部＋--trim 耳朵微調
2026-08-13  0ea33a1f  bank_live v33：偵測器加音量門檻（Harry：張嘴不唱也出聲、閉嘴觸發還是敏感）
2026-08-13  4e7c81c0  bank_live v34：門控改用訓練出來的「在不在唱」模型（Harry：唱咿會變紅嘴）
2026-08-13  9b99b342  bank_live v34.1：門控模型只吃嘴/下巴維度（Harry 螢幕證據：閉著嘴 sing 1.00）
2026-08-13  3f552755  bank_live v35：修 fresh 審查的 6 條（含 --tenor 引進的 pads 迴歸）
2026-08-13  1c782b45  bank_live v35.1：修 v35 引進的 UnboundLocalError（鏡頭執行緒整場死掉、無畫面、門控全開）
2026-08-13  9241f8b3  bank_live v36：合唱配置重寫（--vl 2）＋tenor 音域收到模型驗證範圍
2026-08-13  d9f3062a  bank_live v37：顫音＋音域折八度＋狀態列協定修復（第四、五份審查）
2026-08-13  46848929  docs(STATE)：08-13 夜五審查場收官（v26→v37）與 viva 前待辦
2026-08-14  689819cd  docs(STATE)：頂端加「下次開工三件事」定序（app 旗標寫死／B 線耳裁／viva 凍結配置）
2026-08-14  41546e31  docs(worklog 08-13)：補引擎線（v17→v37、五審查場、感測層血訓）
2026-08-14  be827b7c  STATE: B 線耳裁結案＝收在 3000（B 線凍結，不續訓 8000）
2026-08-14  07e086c5  solo_min 神經即時模式接進應答 app；cache thrash 判活（--blocksize 960）
2026-08-14  3b642246  solo_min 嘴部門控（--mouth-gate，預設關）；SOLO_PY 換 6x venv
2026-08-14  5fca665a  STATE: 門控待辦改成「程式完成、live 未驗」，附四步實測清單
2026-08-14  94d1c13a  STATE: 嘴部門控 live 實測通過＝結案（神經即時模式可以上台）
2026-08-14  ba48b437  bank_live v38：--per-part 多人聲部，團員優先吃真的別的歌手
2026-08-14  d696b825  STATE: 16 種音色專案開跑（fem8 訓練中）；記下一個壞掉的對照
2026-08-14  34e137ed  respond_shell：_spawn_bank 補上 viva 凍結配置（排練/演出不一致點關閉）
2026-08-14  f81f7b12  STATE: app 旗標寫死結案（逐位證明排練＝上台）；待辦剩 viva 凍結文件那條
2026-08-14  06b56eed  app：鍵盤/滑鼠無縫換模式、Live+neural 組合、dry 直通關閉、UI review 修復
2026-08-14  bb83b00f  solo_min 直通聲加回來：兇手是 40ms 延遲不是直通本身
2026-08-14  06dc3e61  STATE: 演出操作段（1-4/滑鼠切換、duo、三個平衡欄）；fem8 40k 待耳裁
2026-08-15  7645d948  STATE: fem8 八嗓耳裁全過；我用質心做的三條預測全部作廢
2026-08-15  6d6dbd72  STATE: 下個 session 交接段（等回答 4 / 要做 4 / 已完成但未驗 3）
2026-08-15  8165b81d  bank_live v39：團員可借別顆模型的歌手（16 種音色接上台）
2026-08-15  d0a007a7  bank_live v39b：團員音庫逐音對齊領唱響度（Harry live「女聲蓋過男聲」）
2026-08-15  a23dbd76  FINDINGS F25 + playbook：bank_live 的雜音源是監看視窗，而兩支儀器都照不到
2026-08-15  a26af033  bank_live v39c：借來的 12 位耳裁只活 4 位（probe 的合格不轉移到音庫）
2026-08-16  fe102bbd  門控 v40：自校＋運動否決＋誠實標籤＋sp/mo 儀器（**預設不啟用**，證據不足）
2026-08-16  fff0a1d6  STATE: 08-16 交接（男聲模型到手、16 音色接上；門控鏡頭路判死；viva 文件仍未寫）
2026-08-16  ecb6e776  包絡相關在 bank_live：回授路徑可量到（279ms 峰）但耦合不是常數，判別力仍不足
2026-08-16  05fb550b  修正 F26 診斷：耦合沒問題，死穴是殘差的語意
2026-08-16  f28f82b1  VIVA_RUNBOOK：上台那張紙（清單上掛最久的原始待辦，結案）
2026-08-16  c1a92168  凍結配置加 --per-part 4（Harry 08-16 三段耳裁：16 人要、真人音色不加分）
2026-08-16  cc6abb48  v41 音域重配：bass 0→-8、tenor +7→+6，並關掉 pad（Harry 08-16 耳裁）
2026-08-16  b974f70d  回退 v41 音域（Harry live「有節奏的雜音又回來了」），pad 維持關閉
2026-08-16  0ff0b216  定罪 --frame-b64：app 的「一下一下的雜音」，凍結配置改 0
2026-08-16  8b76904e  定位並修掉「一下一下的雜音」：元兇是團員借來的真人音色（--mem-real 0）
2026-08-16  2d995640  runbook 校正：frame-b64 的誤判、五支瞎掉的儀器、音域現況
2026-08-16  ccdac75a  --trim 定案 bass=+7,tenor=+6；--pad 維持關閉（Harry 08-16 耳裁）
2026-08-16  725e6b25  bass 音域定案 -8（Harry 乾淨條件下 live 判「這個可以」）
2026-08-16  b0d7b0e3  STATE: 08-16 完整收工（凍結配置全部定案；雜音元兇＝--mem-real；三項耳裁結案）
2026-08-16  55ea4ed4  worklog 2026-08-16：門控收案、runbook、三項耳裁結案、雜音定位
2026-08-17  eeea5b3f  bank_live 建庫韌性：cache key 只含渲染欄位、npz 原子寫入＋載入驗證、渲染進度列印
2026-08-17  042169cf  bank_live 引擎防護：worker 例外不再殺整場、dump 移入關流前＋上限、GATE DEAD 心跳路可見、ready 移入開流後
2026-08-17  3c8b4a93  bank_live 語意修正：--mem-real 預設 0＋領唱排除、mem-real 0 跳過建庫、vlfloor 記實際發聲音、註解翻案
2026-08-17  d70d0032  respond_shell：引擎退場走 registry＋兜底 SIGKILL、ngain 0 不再反轉、看門狗改活動計時、gate/八度/調種子跨重生
2026-08-17  b151725e  respond2：gate 重用鏈修復（%.6g＋下限拒收＋--key-seed）、tail 後 resync、rotation 行不縮排
2026-08-17  a54ab963  UI：bank 模式讀數解隱藏＋dropped/GATE DEAD/WORKER DEAD 上警告條、mode 下拉 blur＋拒絕時回復
2026-08-17  efbb94e1  docs：runbook 跟上 08-17 修復（自檢指令/mem-norm 消失/dump 時序/冷 cache）＋STATE/worklog 08-17
2026-08-17  608e78ca  vlfloor 耳裁結案：Harry 判「after 可以，過」
2026-08-17  a467a180  凍結配置：--frame-b64 開回 5（冤案翻案＋live 實測免費）；runbook 自檢第 3 步回 app 內
2026-08-17  3e444f99  修切模式後鏡頭殘影（exit 降級的迴歸）＋ neural 模式改調內三度（待耳裁）
2026-08-17  60c784b3  切換 UX 成熟化：點擊瞬間回應——SWITCHING 大字、殘影即collect、下拉鎖定、失敗自動回復
2026-08-18  4bbf3ae5  切換失敗語意跟上「一動作一動」：舊引擎點擊即停，失敗＝全停回 Start
2026-08-18  66961920  bank live 門控權威改麥克風（v43 --mic-gate，重開 G34，Harry 耳測 PASS）
2026-08-18  529e0cad  neural 量化天使＋free 四部制（Harry 全鏈耳裁 PASS）；UI 鏡頭顯示退役
2026-08-18  dce8dae1  duo 平衡凍結 0.5/0.5（Harry 終判）：bank 欄留空按模式取預設
2026-08-18  292110e4  STATE 08-18 收工：鏡頭退役＋neural 量化 free 四部制全鏈 PASS
2026-08-18  316446a7  UI 欄位稽核修復：ngain/you 裁決預設首次真正生效＋按模式灰欄＋統計清空＋key 降級警告
2026-08-18  a4178aa1  分聲道輸出開到 stream/duo/solo（Harry 裁決）：bank_live 新增 --out-map、solo_min 補濕層洞、UI 加 B·T·A·S
2026-08-18  9960a145  docs：分聲道 live 首測記錄——路由 PASS；LiPo unit 跳保護已診斷（rig 改插電）
2026-08-18  fc0a7587  修路由回歸：mic 門回授地板 _OLV 在 out-map 下改立體聲等效尺度（4ch 原本讀低 3dB）
2026-08-18  4e822813  tap_listen：UDP 去重按送端分流——app space 與 WiFi 按鈕序號互吞＝實體鍵無徵兆失聰
2026-08-18  abfe5358  實體鍵藍牙化：免配對 GATT 通知版（HID 死案入墓園 G38）＋app 內建 bleak 接收
2026-08-18  a46f3069  STATE/worklog：藍牙按鈕 Harry live 終驗 PASS＝結案；分聲道 live PASS 入 STATE
2026-08-19  f5bc0ae0  四模式音量門檻全面 +6 dB（Harry「應答app音量門檻都拉更高」）
2026-08-19  74c2ffbb  VOICE_CREDITS：出貨路徑上每顆會發聲的模型＝誰的聲音、憑什麼用
2026-08-21  9add6d56  RUNBOOK 開場自檢五步→七步：分聲道欄位（第 6，補提交）＋展場螢幕（第 7）
2026-08-21  0e31ec69  260821: 應答 app 雙擊啟動器（.command）＋runbook §0 改成預設開法
2026-08-23  ee54b17f  260822: 疊層底床（loop_deck）＝現場錄 loop 疊層的插件，主 app 一行不動
2026-08-23  24b01381  260823: 疊層底床＝預備拍＋等待判定重寫＋輸出裝置現場可換
2026-08-24  a6bd39dd  260824: viva 音訊路由定案（應答→viva-out／疊層底床→speakers4）＋RUNBOOK §1.5
2026-08-25  09999f5e  260825: 藍牙實體鍵的鍵帽顯示修正（BUTTON）＋訊息區不再印 undefined
2026-08-25  e14a8e4d  260825: RUNBOOK 自檢七步→八步（第 8＝實體鍵連上）＋第 3 步改寫（鏡頭 08-18 已退役）
```

## harmony_brain — the harmony language model (later merged into harmony/) (3 commits)

```
2026-07-23  bcfa8f6  Freeze harmony_brain working baseline (brain+YIN+live loop+solo_host bridge, best.pt val 0.474)
2026-07-23  25432b5  Snapshot respond.py: design 1 call-and-response angel (offline+live modes; ear-gate pending)
2026-07-26  f6d7337  feat: --key auto (scale-membership detection); respond_take is D# major, not C
```

## solo-choir-roll — the score-roll web prototype (10 commits)

```
2026-06-30  d49656d  Solo Choir score-roll player (Pages deploy)
2026-06-30  b8e0b20  fix: harden iOS WebAudio unlock (resume + silent buffer in the gesture)
2026-06-30  d064739  add: start hint (turn off silent mode · swipe up), fades on first touch
2026-06-30  10b72dc  copy: English-only start hint
2026-06-30  52e3c0b  fix: pre-decode soundfont at load so the FIRST swipe plays (no pre-tap)
2026-06-30  95ddfc8  add: 'tap to start' overlay that unlocks audio (intercepts the first tap, no stray note)
2026-06-30  5fdcd77  fix: create AudioContext inside the tap gesture (iOS won't unlock one made at load)
2026-06-30  21343e7  style: cohesive gallery palette (cool #EEF2F7 + slate chrome + soft spot notes)
2026-08-18  fcca8ef  add: four short-path redirects (j/c/h/s) for per-song business cards
2026-08-18  fd3e43f  add: f/ redirect (Farewell 送別) for fifth business card
```

## DDSP-SVC 6.x checkout — local changes for the sampled choir (1 commits)

```
2026-08-10  6e03f85  reflow: init_noise 守門參數（None＝原行為；spike6 v33 凍結噪聲網格用）
```

## yvwHY/solo-choir-prints — 3D-print batches, hardware documents (9 commits)

```
2026-07-20  71f832f  Init print-batch repo; add 260719 re-slice (4 plates for MINI)
2026-07-20  56e861f  Fix P1 wo5_shell support: organic -> grid + 5mm brim
2026-08-04  ffef49a  260804: K669B 按鈕殼＋試環（PLA 合規版；PETG 留自帶料路線）
2026-08-18  2f573db  260816: Solo Choir 名片五版（雷切雙層＋外框化＋25張排料）
2026-08-18  22a49e0  260816: 切割線寬 0.025→0.01mm（Hatchlab 判定規則）；README 補 Hatchlab 規定
2026-08-19  c0c3bc0  260819: 觀眾說明卡定稿（A4 全模式版）＋送件檔
2026-08-19  577f07a  260819: 說明卡改 140×250mm＋寫進實體鍵；送件檔重出
2026-08-20  819dd09  260820: 說明卡改 120×190mm；字級換間距（判「有點擠」後翻轉配比）
2026-09-03  36696fc  積欠的列印/送件檔一次補上（08-04 k669 批次、08-16 名片、08-24 提示牌）
```

## solo-choir-document — the written submission (3 commits)

```
2026-08-19  fc0a4d0  建庫：Final Project 繳交文件與圖片素材
2026-08-31  e81fabe  260831: 最終文件影片重剪成 2:00 ＋ 4K 版；剪接全面腳本化
2026-09-03  02f5d4d  260903: 正文改寫（形態演進三段、引用補齊、Concept/Technical 重寫）；Figure 4 改直式
```

## yvwHY/solo-choir — this repository (15 commits)

```
2026-09-03  23156d5  Solo Choir — standalone submission repository
2026-09-03  9a134a4  README: add documentation video link
2026-09-03  a9e961c  Use Yu-Ting Liao as the byline
2026-09-04  e646e15  docs: finish English pass, condense process, rename launchers
2026-09-04  e5b4f7b  research/prints: condense to one English README, drop derived files
2026-09-04  a82e71d  research/prints: keep the slicer profiles the batches used
2026-09-04  8e069d2  research: translate score_live and spike_stream, tidy eval/studio device names
2026-09-04  2dc8e9b  research: translate score_align, sampler_mouth, loopback_lab
2026-09-04  3334dd2  research: translate score_notes, perform, clap_probe, angel_audit, key_probe, prosody_ab
2026-09-04  aac69f6  research: translate the probe and lab scripts; fix two stale stdout patterns
2026-09-04  a0a29c1  research: translate the last top-level experiment scripts and the lab UI
2026-09-04  384f556  research/scratchpad: translate the feedback probe and the gate trainers
2026-09-04  c78dffb  research/scratchpad: translate the vowel and melody probes
2026-09-04  059d0d0  research: add index READMEs for harmony_experiments and prints
2026-09-04  59b7ba2  Attribution: cite DDSP-SVC, HuBERT-Soft, NSF-HiFiGAN, MediaPipe and POP909
```

