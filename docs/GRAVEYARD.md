# GRAVEYARD — Dead Ends Registry

**Audience: AI assistants (Claude, Codex, any model) and humans working on this repo.**
Every entry below is an approach that was actually tried and killed, with the root cause of death. Days of work are buried here — do not dig them up by accident.

## Rules of this file

1. **Before building anything new** in these domains — live streaming/mixing, pitch shifting, ensemble/choir sound, female timbre, transcription, per-voice timbre, wearable hardware — search this file first. If your idea matches an entry, do NOT re-run it; read *Instead* and attack the recorded root cause.
2. **When an approach dies, register it here BEFORE moving on.** Format: Tried / Died because / Evidence / Instead. A one-line burial saves the next model a day of excavation.
3. **Entries are never deleted.** A superseded direction gets a status note, not removal.
4. A verdict here is a **root-cause** conclusion, not "didn't work once". If you believe the premise has changed (new engine API, new model, GPU available), say so explicitly in your plan and cite what changed — that is the only legitimate way past an entry.

---

## Live engine / streaming

### G1. Mixing or splicing two engine output streams (all variants)
- **Tried:** dual-stream crossfade (scored 3/10 by ear), 20 ms fade splice (blurred), energy-valley splice + 5 ms fade (blurred). ~12 prototype iterations.
- **Died because:** the Beatrice waveform generator's phase free-runs per stream. Two streams of the *same* voice are phase-incoherent, so any crossfade combs/blurs. This is a property of the engine, not of the fade length.
- **Evidence:** 2026-07-05 session; prototypes `feed_ab.py` v0–v11 (scratchpad); `docs/worklogs/2026-07-05.md`.
- **Instead:** one stateful stream per voice, change pitch *within* the stream via glide (see FINDINGS F2).

### G2. vcclient-style chunk + context re-feed applied to dynamic harmony
- **Tried:** re-feeding context audio per chunk (how vcclient batches) while the harmony shift changes.
- **Died because:** re-feeding interrupts the stateful stream → ~10 Hz tremolo on the output.
- **Evidence:** 2026-07-05 diagnosis (same worklog as G1). vcclient gets away with it because its pitch shift is *static*.
- **Instead:** FINDINGS F2 (glide inside one stateful stream).

### G3. "Two factories": neural engine at fixed pitch + external real-time DSP pitch shifter
- **Tried:** delay-line 2-tap-crossfade real-time shifter applied after the engine.
- **Died because:** the shifter itself sounds bad (warble/metallic) even offline.
- **Evidence:** commit `121466c2` (docs: Task 6 ATTEMPTED, dead end), 2026-07-05.
- **Instead:** FINDINGS F2. The 7→9 quality gap needs a feature-domain custom engine binding (post-viva project).

### G4. Jumping `set_pitch_shift_semitone` directly to the new target
- **Tried:** setting the new shift value in one step on note changes.
- **Died because:** every *change* (regardless of size) scrapes the engine's internal state; a melody makes ~18 changes in 8 s → constant instability ("scrape").
- **Evidence:** 2026-07-05; decompiled vcclient 2.2.2 confirms it glides via tiny `set_config` cent-steps and never jumps.
- **Instead:** glide 3–6 ¢ per 10 ms hop + 80 ms debounce (FINDINGS F2). Note the rc0 binding quantises pitch with a ~5–10 ¢ deadzone, so micro-detune via this API is a no-op (see G8).

### G5. Skipping inference for toggled-off voices to save CPU
- **Tried:** `if not voice.on: continue` in the voice loop.
- **Died because:** a cold converter resumed with ~35 ms of zeros per re-enable → the "choppy toggle" symptom. 39 exact 35 ms digital-silence gaps were measured in a recorded stem; pump underruns ≈ 2 and PortAudio overflows = 0, proving the earlier buffer/underrun theory wrong.
- **Evidence:** 2026-06-22; offline A/B: skip-mode 27.0 % spurious zero-fill vs always-run 0.1 %.
- **Instead:** always run inference for every live voice (keep warm), gate only at the mix sum. Cost: RTF no longer shrinks with voices off (~0.375 for 3 parts — acceptable). Offline render may still skip.

### G6. Real-time input-side pre-shift (+12 into the model) for female timbre
- **Tried:** granular real-time pitch shifter before the engine (`server/female_timbre_test.py`).
- **Died because:** granular shifting destroys phoneme content; +12 clips and feeds garbage to the model. Output-side TUNE (octave-up) already achieves the same goal.
- **Evidence:** 2026-06-21 by-ear.
- **Instead:** output-side TUNE / per-voice octave; offline kNN-VC for genuine female timbre (FINDINGS F5).

### G7. Blaming device-switch timbre collapse on "input overflow"
- **Tried:** treating the post-device-switch collapse (all voices turn male) as an input/clock corruption problem.
- **Died because:** a clean independent mic stream still collapsed. Real cause: the UI's `controlState()` sent `voices` without the `model` field; `bridge.set_control` did a whole-dict `update()`; `_restart_engine` then rebuilt every voice on the default `"tenor"` model.
- **Evidence:** 2026-06-22; fix in `app/bridge.py set_control`.
- **Instead:** when live timbre changes inexplicably, check control-state completeness across restarts *first* (DEBUG_PLAYBOOK), not audio-clock theories.

### G8. Neural-API micro-detune for ensemble spread
- **Tried:** small per-voice detunes (±5–15 ¢) through the engine pitch API.
- **Died because:** the rc0 binding quantises pitch with a ~5–10 ¢ deadzone — small detunes are silently dropped.
- **Evidence:** 2026-07-04/05 sessions.
- **Instead:** do decorrelation offline in the signal domain (WORLD recipe, FINDINGS F3).

## Timbre / ensemble

### G9. "Distinct speakers ⇒ clean ensemble" (DISPROVEN)
- **Tried:** giving each of 6 copies a different `target_speaker` expecting a choir.
- **Died because:** time-aligned copies fuse into "one person" regardless of timbre. Timbre difference is neither sufficient nor necessary for ensemble.
- **Evidence:** 2026-06-22 by-ear; corrected 2026-07-04 by the decorrelation work.
- **Instead:** pitch + timing decorrelation, independent vibrato, stereo pan (FINDINGS F3). Speaker variety is optional seasoning.

### G10. Per-voice formant spread on one model (Bass −1 … Sop +1)
- **Tried:** spreading `formant_shift` across parts to differentiate the "choir of one".
- **Died because:** with tight voicing on a single model, spreads of 0 / 1.0 / 1.6 were indistinguishable in the mix (user A/B). Single-model timbre is inherently homogeneous; formant can't fix it.
- **Evidence:** 2026-06-25, feature reverted same day.
- **Instead:** real timbre diversity needs different models/speakers (which re-opens the electronic-timbre problem) or offline kNN-VC (FINDINGS F5).

### G11. Live stereo "ensemble/spread" rework
- **Tried:** live stereo widening of the parts (2026-06-21).
- **Died because:** Alto sounded wrong; abandoned, UI controls removed.
- **Instead:** stereo belongs in the offline decorrelation recipe (FINDINGS F3), not in the live path.

### G12. kNN-VC soprano via pitch-shifted input
- **Tried:** (a) librosa phase-vocoder +12 → formants shift too → chipmunk, worse WavLM matching; (b) WORLD/pyworld formant-preserving shift (f0×2, envelope kept) → input sounds natural but soprano output still bad.
- **Died because:** (b) failing proves the root cause is kNN-VC/WavLM being out-of-distribution at high register — not shift quality.
- **Evidence:** 2026-06-26.
- **Instead:** in this arranger Sop carries the melody in the *male* register, so Sop simply uses the **alto reference pool** — it was a register-mismatch problem, not a high-note problem. See FINDINGS F5.

### G13. Expecting Beatrice male→female conversion to sound female
- **Tried:** male input through the trained female / satb2 female speaker expecting a female voice.
- **Died because:** Beatrice preserves substantial source vocal-tract character; male→female yields "neutral", not female. This is documented source-speaker leakage (arXiv 2504.15822), not a bug in this repo, and cannot be fixed at the mix level.
- **Evidence:** 2026-06-21 by-ear; literature survey 2026-06-24.
- **Instead:** (i) push output pitch ≥ ~A4 and female timbre emerges (use TUNE/octave); (ii) offline kNN-VC render (FINDINGS F5); (iii) convincing *live* female needs GPU inference — out of scope on CPU.
- **⚠ 2026-08-14 — the verdict stands, the 07-22 numbers below do NOT. Do not cite them.** Re-measured through `260811_bt/render_bt.py` (same take, same satb2 model, only `target_speaker` changed), at **both** registers because the 07-22 run used `female_timbre_test.py`, whose default is `PITCH=0.0` — i.e. the male register this very entry says female timbre cannot appear in:
  - 0 semitones: correlation **0.575**, centroid **980 vs 773 Hz = 21.1 %** (07-22 recorded 0.955 / 2 %)
  - +12 semitones: correlation **0.673**, centroid **1215 vs 955 Hz = 21.4 %**, log-spectral distance 8.82 dB
  - Whole-file centroid (no frame gating) gives 1051 vs 867 = 17.5 % — the discrepancy is not a framing artefact.
  The two speakers differ ~10× more than recorded. Most likely the 07-22 measurement was taken through a path where the speaker never actually switched — this entry itself documents that that control chain had `except: pass` swallowing failures and was a silent no-op on single-speaker models. **Lesson: a number measured through a path with a known silent-failure mode is not evidence; re-measure through a path that has been proven to apply the setting.**
  **Harry's ear, 2026-08-14, loudness-aligned A/B at +12 (`ear/AB_satb2_spk0_vs_spk1_up12.wav`): "都是女聲" — both female.** So the conclusion holds by ear (the speaker index is not a voice-type control) even though the supporting metric was wrong, and metric-vs-ear conflicts still resolve to the ear. Live SATB got solved another way — see FINDINGS **F23** (two models coexist at `--blocksize 960`), which removes the reason anyone wanted joint multi-speaker training in the first place.
- **Measured 2026-07-22 (satb2, same take through spk0 vs spk1) — SUPERSEDED, see the warning above:** correlation **0.955**, difference RMS 30 % of signal, spectral centroid **904 vs 886 Hz** — a 2 % shift. The switch *works*; it is simply not a gender change. **Consequence: the UI's F/M control was removed** (`3c90803b`) — it claimed male/female, delivered a faint variant, was a silent no-op on the six single-speaker models (`get_num_speakers() == 1`), and its failure was swallowed by `except: pass`. Four layers of dishonesty stacked into "is this a bug?". Engine-side `--speaker*` launch args stay.

## Transcription (studio)

### G14. CREPE (and madmom)
- **Tried:** CREPE for f0 (separate ~5 GB `vcclient-crepe` conda env with TF/numpy2); madmom for onsets.
- **Died because:** SwiftF0 (MIT) matched CREPE's low-register accuracy without octave jumps, runs in-process, no TF. madmom won't compile on numpy2 at all. Retired a second time when the heuristic onset stack itself was replaced (G15).
- **Evidence:** commit `319b1fe6` (SwiftF0 swap); `server/transcribe_crepe.py` removed; `conda env remove -n vcclient-crepe` is safe.
- **Instead:** `_hmm_melody` global Viterbi decode (default since `76d3e15f`), `_swiftf0_melody` heuristics as fallback.

### G15. Piling more onset/segmentation heuristics onto transcription
- **Tried:** rounds of heuristics — vowel-boundary MFCC splits, tuning-centre θ quantisation, leap-transit absorb, fry/crack trimming (commits `ed3b447a`, `78358e33`, `c5333588`, `4652f4bf`, `96a870dd`, `c8575fd0`).
- **Died because (ceiling declared 2026-07-06):** on the 10-take suite the remaining errors are *true input ambiguity* — same-pitch legato syllables with no energy/spectral cue, and notes sung on an x.5-semitone boundary that honestly quantise to either neighbour. No heuristic can decide these.
- **Instead:** the drag-to-edit editor is the designed fix for the residual. Do NOT add heuristic rounds; if transcription "looks wrong", first check whether it is one of these two ambiguity classes.

### G24. Offline arranger-rendered stimuli for the baseline listening test — 2026-07-22

- **Tried:** `eval/render_stimuli.py` → `studio_api.transcribe()` → `arranger.arrange_with_chords()` → render Beatrice + naive from the same score, as the A/B material for the viva-keystone listening test.
- **Died because — ONE sufficient reason, plus one hypothesis that is NOT established:**
  1. **(SUFFICIENT, no listening data needed) It does not represent the instrument.** The live app harmonises by fixed diatonic scale-steps (`SoloChoir.py`, `--free` adds causal chord inference); the studio arranger does chord inference + voice-leading + bloom + per-voice register tables. They are different instruments — testing one says nothing about the other. **This alone kills the route**; it is verifiable from the code and does not depend on anything below.
  2. **(HYPOTHESIS — NOT VERIFIED, corrected 2026-07-22 evening)** The original entry claimed the arranger's voicing "swamped the variable under test" and called it a root cause. That overstates the evidence. **What was actually measured:** soprano shifted **+15 … +17 semitones** on the 4 real takes; the two conditions are objectively different (correlation 0.27–0.41, naive's spectral centroid consistently +124–249 Hz — the phase-vocoder formant-shift fingerprint, so rendering did not fail); Harry scored **2/4** on a blind guess. **What was never measured:** that the register shift is *why* it was inaudible. The obvious control — re-render at a sane soprano register and blind-listen again — was never run; the route was abandoned instead.
- **The bigger hole the original entry missed: the test had no statistical power.** 4 binary trials, one listener. Under pure guessing, P(exactly 2/4) = **0.375** — the observed result is the single most likely outcome of a coin flip. A listener who could genuinely hear the difference 75 % of the time still scores 2/4 about **21 %** of the time. Worse, **the design could not have produced a significant positive result under any outcome**: even a perfect 4/4 gives a one-tailed p of **0.0625**. So **2/4 supports no conclusion at all** — not "the chipmunk masked it", not "the difference is inaudible", not anything. The honest reading is *"this test returned no interpretable result"*.
- **Alternatives never ruled out** (any of these could equally explain 2/4): the Beatrice-vs-naive difference may simply be hard to hear **inside a 4-part mix** (it was audible in earlier solo comparisons — a mix masks); procedure/blinding effects; or plain lack of power.
- **Carry-forward for RQ1 (this is why the correction matters).** RQ1 ("must the harmony sound like a human voice?") is the *same comparison*, just delivered live. If the difference is hard to hear in a mix, the live A/B fails the same way — and repeating "one listener, four trials" on 07-27 will again yield nothing interpretable. **Before running it, fix the trial count and n, and decide how to avoid mix masking.** A cheap prior check exists with material already on disk: `eval/stimuli_260721_private/_render_*/` holds **per-part stems for both conditions** (`studio_alto.wav` vs `naive_alto.wav` …), so a single-part A/B tests audibility with no arrangement and no mix confound, without re-rendering anything. **Do not delete those directories** until that question is settled. They are **untracked and deliberately not committed** (65 MB of wav; and `_private/KEY.csv` is the unblinding map — the generating script warns "never serve it", so it must not ride along in a pushed repo).
- **Also found:** transcription octave errors survive into the stimuli (phrase 1 had two isolated MIDI-82 notes, 34 semitones above the median) — `render_stimuli.py` transcribes internally and has **no injection point for a corrected score**, so the "EDIT the score" step in the plan has no implementation.
- **Instead:** live A/B inside solo_min — `--naive` swaps only the per-voice timbre generation, keeping f0, the diatonic decision and the harmony identical (`e00340b6`, FINDINGS F13). Supervisor-sanctioned direction (Matthew meeting: build a real-time shifter to compare against). The baseline algorithm is the author's own prior instrument, `yvwHY/Harmonizing` (Tone.js `PitchShift`), so it is a citable tool rather than an ad-hoc straw man.

## Superseded directions (project-level)

### G16. Physical punched harmony card take-home
- Superseded 2026-06-30 by the **QR score-roll** web take-home (spec `specs/2026-06-29-qr-score-roll-design.md`). Deprioritised, not disproven.

### G17. "Studio/arrange is the core of the project" (2026-06-25 pivot)
- Reversed 2026-06-27 by the spine decision: **the live embodied harmony loop is the hero**; studio = offline quality/artifact/stem factory serving it; ear-trainer = research vehicle. The still-valid parts of the 06-25 pivot: keep `arrange(notes, key, style)` style-abstraction, keep note-level drag editing between arranger and render.

### G21. Beatrice-based DAW plugin (VST/AU) — rejected before any code (2026-07-07)
- **Tried:** proposal only (llm-council, 5-lens blind review — unanimous reject); no code written.
- **Died because:** triple kill, each sufficient alone. (1) License: Beatrice is non-commercial / NO redistribution — a plugin's whole point is distributing the engine to other machines; (2) thesis: G20 amplified — the instrument becomes a producer-workflow effect on a DAW timeline, dismantling the live embodied loop ("so it's just a voice plugin?" is a free viva attack); (3) engineering: Python in-process engine → C++/JUCE real-time rewrite, weeks-to-months.
- **Evidence:** council transcript 2026-07-07 (aggregate: First-principles > Rigorist > Long-horizon > Red-teamer > Pragmatist, all five reject (a)).
- **Instead:** DAW interop IS ALREADY SHIPPED as studio's stems + MIDI export — say it as a thesis choice ("deliberately not a plugin"), not a capability gap. Any future plugin/product form (2027+, company track) requires an OWN-model engine first — post-viva voice training (FINDINGS F9) is the real prerequisite, not JUCE code.

### G23. `--voice-delay` Tier-1 live canon (聲部分離 Tier-1「與過去的自己輪唱」) — by-ear FAIL 2026-07-19
- **Tried:** `--voice-delay` per-voice output FIFO in solo_min (commit `62b725fe`, plan `plans/2026-07-16-voice-delay-canon.md`). All objective verification PASSED: Regime A 0.0 (flag off AND `0,0,0,0`), shift-equivalence bit-exact (1 s delay = exactly 48 000 samples, first second silent). Two live configs same evening: ① `0,0.6,1.2,1.8` with normal harmonized intervals; ② `0,0,2,4` with V3/V4 at unison (steps 0) — a true phrase-level canon.
- **Died because:** by-ear, both configs (Harry). ① word-length delays + harmony = stutter-echo,「每一個字都會跑 delay，怎麼樣都不成調」— four "past yous" fight the current line. ② even clean phrase-level unison canon 還是不好. Root causes: (1) **DAF** — singing live against delayed copies of your own voice is delayed-auditory-feedback interference, hostile to the singer; (2) free singing is not canon-structured material; (3) conceptually a delayed voice **stops following the current voice** — the same spine violation that killed G20 (the choir must follow the voice, now).
- **Evidence:** worklog 2026-07-19 §K.
- **Instead:** live canon is dead; the flag itself is dormant and harmless (off = proven bit-identical). The compatible shape is OFFLINE/AUTOMATED: canon delays in a studio render or the automated-mode presentation, where nobody is singing live against the delayed parts. Do not re-attempt live canon without re-reading this entry.

### G20. MIDI chord-hold inside solo_min (removed 2026-07-07, same day it was ported)
- **Tried:** full port of the original app's MIDI input harmony into solo_min (engine `midi` control + bridge I/O + UI card + playability pass: 15 ms coalesce / sustain / latch). Commits `a0a8ef69`, `cec75363`, `aaf3d518`, `998289af`; plan `plans/2026-07-07-midi-port-solo-min.md`. All objective verification PASSED (byte-diff 0.0, contract tests).
- **Died because:** concept mismatch, not a technical failure — Harry's by-ear verdict: "跟 solo choir 不搭". Chord-hold hands harmony authority to the keyboard, so the choir stops following the voice; that contradicts the thesis spine (body as interface / the live embodied harmony loop). A known release-mute click existed but was NOT the reason.
- **Evidence:** reverted the four commits 2026-07-07 evening (see that day's worklog); the working implementation survives in git history and in the ORIGINAL app (`app/bridge.py:610-756` + v10 UI), untouched.
- **Instead:** solo_min stays voice-driven (diatonic/Free). If MIDI ever returns, the compatible shapes are (a) audience-plays-chords two-person exhibition interaction, or (b) the original spec's set-key variant (keyboard re-keys the diatonic harmony; the choir still follows the voice). Re-porting = `git revert` of the four revert commits, then fix the release click (gain ramp).

## Hardware / CAD (Track B)

### G18. Fitting the mask by multi-section surface loft on the head scan
- Self-intersects persistently (`Rail misses profile`); judged "one of Fusion's hardest techniques, don't start here". Thicken on the hand-drawn surface also fails (`ASM_LOP_OFF_NO_SURF`). Full-coverage `mask_concept12` (824 cm³ + voronoi + straps) collapsed under its own complexity — each fix spawned the next problem.
- **Instead:** lightweight ear-hook skeleton **frame_A** (v1 STL shipped: `SoloChoirCode/260703_frame_a/frame_A_v1.stl`). 2026-07-02/03 worklogs.

### G19. Test B: DAEX exciter through-body ("perform mode" transduction)
- Surface exciter output is absorbed by soft tissue — inaudible as intended. Expected negative result (2026-06-27).
- **Instead:** cup-shaped spatial-source design (validated 2026-06-24: choir sounds co-located with the singer).

### G22. Nape pivot as a separate hinge (through-pin at tube ends / whole-yoke rotation about (0,146,−31.5)) — 2026-07-10
- **Tried (4 versions in one day):** rectangular blocks → ⌀18 hinge barrel + ⌀3 through-pin (press-fit, then bayonet front-entry) → boss-and-fork "eyeglass hinge". All CAD-verified individually, all killed by two facts in combination:
  1. **The pin corridor is owned by the wire.** The nape tubes' lower ends pass 1.2 mm from the pivot axis, so any axial member (pin, screw, pilot hole) at x ±(8.5..11.65) collides with them. Trimming the wire to clear it is FORBIDDEN (Harry's standing rule 2026-07-10: the wire is never cut to accommodate a part — parts accommodate the wire).
  2. **Whole-yoke rotation about the tube-end axis cannot don.** In the real donning direction (chest panel lifts over the face — the direction whose ear corridor was tuned 07-09) the collar/sleeve hits the nape tubes after only ~20° (tube "spoke" sits at θ≈138°, collar rest at 164° → 26° margin). The clear direction (+130°) sweeps the panel through the wearer's torso. The 07-09 assembly articulation "verified" this joint only because Fusion joints do no collision checking, and only arms-vs-ears was ever swept.
- **Instead (v4, SHIPPED):** **the collar's own straight rear section IS the axis** — the ⌀3 wire rotates inside a printed ⌀3.3 bearing channel (`pivot_mount`, single fixed part, no metal pin, no printed joint). Collar sweep radius = 0, so the tube collision vanishes identically; panel arc unchanged. Do not reintroduce a discrete nape hinge without re-reading this entry.

### G25. 「全機黃銅通道統一縮孔到 0.05 mm 單邊」(2026-07-21) — 實印 FAIL 2026-07-22
- **Tried:** 為了消除鬆動，把全機吃 ⌀3.0 黃銅棒的通道**一律**縮到 0.05 mm 單邊餘隙（`band_clamp` 髮箍隧道與**兩條中柱道 4.3 → 3.1**、`pivot_mount` 3.8 → 3.1、`door_col` 3.8 → 3.2、`wo4_cup` 3.6 → 3.2、J 鉤座 3.3 → 3.1）。CAD 上 `measureMinimumDistance` 逐條驗過 0.0500，數字漂亮。
- **Died because:** 實印後 **`band_clamp` 的兩條中柱道穿不過去**，其餘可以。差別不在孔徑（中柱道與 `pivot_mount` 都是 ⌀3.10／0.05 單邊），而在**通道彎不彎**：中柱道面 `f1/f2` 的 bbox 是 3.100 × **5.474** × 20.000（直通道的 Y 幅寬應等於孔徑），中柱軸在 band_clamp 的 20 mm 高度內橫擺 **2.374 mm**；`pivot_mount` 是單一 Cylinder 面的**直孔**。而中柱是 07-20 **徒手冷彎**的（worklog 07-20 §D）——手彎的桿不可能跟 CAD 樣條在 2.3 mm 擺幅上吻合到 0.05 mm。**⌀4.3 的舊值（0.65 單邊）本來就是在吸收這個誤差，統一縮孔把它拿掉了。**
- **不是熱縮。** Harry 與我最初都歸因於熱收縮；PLA 線性收縮率 0.3–0.5% 換算到 ⌀3.1 只有 **0.012 mm**，比層高小一個數量級。**歸因前先算數量級。**
- **Evidence:** worklog 2026-07-22 §G；`prints/260722_prints/README.md`。
- **Instead:** **餘隙要看通道形狀，不能全機一個數。** 直孔 0.05–0.10 單邊可行；**彎通道必須另計**（本批放到 ⌀3.30 ＝ 0.15 單邊試水溫）。若 ⌀3.30 仍穿不過，代表手彎誤差 > 0.15 mm，**下一版把中柱道改成 C 形喉口從側面卡入**（`band_clamp` 髮箍座與 `vu1_panel` 夾都已驗證這招），而不是繼續放大到會鬆。不要再對「全機統一某個餘隙」這種說法照單全收。

## 和音天使 harmony_brain（260722 新線）

### G26. Granular dual-tap 移調器當天使的嘴（ShiftVoice）— 真嗓 FAIL 2026-07-22
- **Tried:** whammy 式環形緩衝 dual-tap 移調（`260722_harmony_brain/live.py` ShiftVoice）＋凍結延音＋auto-octave 音域折返。合成鋸齒驗收全過（移調準度 46.0/44.1、凍結延音、9.2x→226x RT）。
- **Died because:** 真嗓三輪 by-ear 全敗（狂抖、無語調、低音泥、不像本人）。量測定罪：56s 內斷點 52 次、段內 f0 std 0.65 半音（穩定人聲 0.1–0.3）、1–8Hz AM 偽影佔包絡能量 59%（dual-tap 交叉淡化的固有掃頻拍頻）、ratio 綁即時 f0 導致追蹤閃爍直通輸出。
- **Lesson（血價）：合成測試訊號過關 ≠ 真人聲過關。人聲管線的驗收訊號必須是真嗓錄音**（此後建立 take.wav stems ＋ 離線 replay 閉環）。
- **Instead:** WORLD 分析重合成（batch 版 Harry 首次認可質感）。

### G27. vcclient 外部控制當即時音高通道（REST convert_chunk / slot PUT）— 2026-07-22
- **Tried:** headless 啟動 `model/VC/dist`（port 18000）；逆向 convert_chunk 格式（**16k in / 48k out / hop 160**，此數字可再用）；full-slot PUT `pitch_shifts[dst_id]`（部分欄位 PUT 無效，需整包）。
- **Died because:** ① convert_chunk 是無上下文/無 crossfade 的 debug 通道——真嗓過去只剩 0–2 個 voiced 幀，餵預移調音源音高傳遞全滅（斜率 −0.02、相關 −0.10）＝**不能用它評判模型**；② native 視窗管線與 server REST 半脫鉤（REST 說 slot 3、實際在跑 slot 0 JVS；voice-changer-manager 回報 null），full-slot PUT 觸發重載斷流「要重新開口才有聲」＝不能當每格控制通道。
- **Evidence:** 正規路徑（BlackHole 播入 GUI 管線）實測**乾淨**——WORLD 天使經 myvoice 轉換無電子音、轉換完整。引擎沒問題，是這版 build 沒有可靠的外部即時控制面。
- **Instead:** Beatrice 降級為可選音色鍍膜；Harry 美學裁決偏好 WORLD 保留的自身聲音顆粒（voice-ownership 發現，論文素材）。

### G28. Naive 流式 WORLD（46ms hop ＋ YIN 塊級 f0）— by-ear FAIL 2026-07-22
- **Tried:** `world_rt.py`：hop 2048、context 4096、cheaptrick/d4c 逐塊、10ms 接縫交叉淡化、音高保持振幅釋放的收尾。3.5x RT；客觀指標過（接縫頻段 4%、段內 std 0.18）。
- **Died because:** Harry「很糟糕」。離線版好的三個前提正是流式丟掉的：harvest 級逐幀 f0（YIN 塊級常數 f0 餵 cheaptrick → 頻譜就錯）、完整上下文、幀的自然演化（小窗＋凍結重複＝死質感）。客觀指標沒抓到的＝質感維度。
- **Instead:**（設計已定未動工）①應答式（樂句後應和，耳口全用離線品質零件）②長音觸發同步（只在穩定母音進場、每音符單次 batch 合成）③DDSP-SVC 並行賭注。耳升級 torchcrepe tiny 為兩案前置。

### G29. JSB 腦的預先換音（pre-commit anticipation）— probe 量測 FAIL 2026-07-23
- **Tried:** 交錯 (sop, alto) causal LM 的 next-token logits 本身就是「Harry 下一個音」的預報 → 設想信心閘門＋預先送 steps 讓天使在換和弦時與他同時落地。動工前先建量測 `harmony/peek_probe.py`（重播真嗓 take、餵入真 token 前先取預報；不載引擎、live.py 零改動）。
- **部分重開（2026-08-11，melody_lm v0／bank_live v8）**：死因①③被移除後的降級重試——資料換成**他自己的 51 檔錄音**（top-1 39.3%/top-3 64.3% vs 本碑的 11-16%＝換對資料後可學是量出來的）；用途從「預先押注換音」降級成「**已偵測音的 commit 加速**」（模型不產生內容，猜錯代價＝多等 35ms 不會唱錯）；時機預測（本碑 0% 那半）完全不碰。⚠ 血訓自記：建模前沒查本碑（08-11 第二次「動工前沒查」同型犯規）。
- **Died because:** 兩段真嗓 take 數字全滅：換音**時機** 0%（模型任何時刻都高信心押 HOLD → 信心閘門無效，因為它錯得很有信心）；換音**音高**（遮 HOLD/REST）top-1 11–16% ≈ 亂猜水位（音域內 ≈12%）。根因：①它的 schema 是 Bach chorale 的 soprano，不是 Harry 的自由唱——預感的對象錯了；②自由節奏無拍子 → 時機預測 ill-posed（分布不匹配：模型活在量化格線）；③無 veridical（不認識這首歌）。預先押注 84% 會錯，比「慢半拍但永遠正確」更糟。
- **Lesson:** 預感可分解（Huron）＝曲風 × 這首歌 × 拍子 × 身體訊號 × 對位；此腦只有「錯的曲風＋對的對位」。**對位（反應式）那半從未壞** — solo_min 整合線價值不受影響。單一 LM 兼任預感器＋對位器,修預感靠換資料不必拆架構。
- **Instead:** proposal `specs/2026-07-23-second-chorus-partner-proposal.md`（阿卡貝拉/barbershop MIDI 重訓＋長 context induction＋節奏提示）；`peek_probe.py` 留作該線驗收尺（Bach 基線數字在此）。

### G30. DDSP 嘴加料重訓用舊 session 料（tenor 25min 混 0619）— 盲聽 FAIL 2026-07-25
- **Tried:** tenor.wav（舊 session）過同一條 `preprocess_voice.py` 出 211 clips/13.2min，混 F9 clean set 成 431 clips/27.6min，combsub 同 config 訓 30k（val 原封同 5 顆保可比）。test_loss 確實較低（1.04–1.07 vs 舊平台 ~1.09）。
- **Died because:** 三格盲聽（blind_carrier_upgrade_0725）Harry「2>3>1」＝混料版**墊底**，連原 14.9min 版都輸。F9「old+new mix was verified worst」在 DDSP 架構下由耳朵重新確認——舊 session 的錄音條件差異污染音色一致性，val loss 降低量不到耳感門檻（**loss 是弱代理**的再一次實證）。
- **Instead:** 加料只能加**同條件新錄的料**（同 mic/房間/level 的 fresh session）；或走選項⓪取樣嘴（見 worklog 07-25 §A）。

### G31. DDSP-SVC 6.x reflow 取代 combsub（14.9min 資料）— 盲聽＋RTF 雙 FAIL 2026-07-25
- **Tried:** 6.x RectifiedFlow（50-step euler＋nsf-hifigan），encoder/f0/資料與 combsub 全對齊隔離架構軸；兩次訓練（谷底 4k；6.x 刪舊 ckpt 政策吃掉 run1 谷底 → run2 全程快照拿回）。
- **Died because:** ①盲聽谷底顆仍小輸 combsub 原版（2>3，差距極小）＝架構升級在此資料量下不付錢——大模型 4k 就過擬合，吃不飽；②**CPU RTF ≈ 2.05**（combsub 0.09 的 23 倍）＝live 資格出局。
- **Instead:** 資料量先解決（錄新料）再議大架構；reflow 若復活只當離線 render 嘴（降 step／MPS 加速未試）。6.x 血訓三雷已記 worklog 07-25 §E。

### G32. 選項⓪取樣嘴當全旋律線嘴（concatenative full-line sampling）— by-ear FAIL ×4 2026-07-25/26
> **✅ 恢復生效 2026-07-26 上午（消融證據完備後 Harry 重審全敗）**：v7 run 制（真 legato、接縫 34→14、過渡區 33→23c）耳測「好了一些，但有幾個段落還是斷斷的尤其前面」；v8（＋換氣單元）「跟 v7 差不多」＝換氣無感；Harry 真嗓 v8keyed「不自然」。**且 Harry 否決降級後的長音定位**：「只做長音不太符合定位——一個只會長音的天使（還不是跟上的那種），還是和音天使嗎？」→ **選項⓪全線判死（含長音候補角色），Part B 母音錄音取消**，分鐘數改餵 Part A 訓練料。消融本身留為方法論資產（下方撤回段的歸因數字全部有效，證明判死不是參數沒調好）。
> **（歷史）撤回 2026-07-26 凌晨（Harry 駁回速判：「直接判死好嗎，先好好分析原因吧」）**：量測歸因（worklog 07-26 §B）顯示四輪耳測混有三個**未隔離的可修因子**——girl 版主犯＝母音/音色亂跳（相鄰單元 MFCC cos 距 0.125，選取從未約束）、harry 版主犯＝短單元硬循環（55% 事件需循環、p90 2.94x）＋跨 session 拼貼（25/52 次換音跨房間）、v4 全熨平殺死載體表情（「音色皮」的可能主因）。消融 v5（樂句內音色一致＋夠長單元偏好＋熨平降 0.35）三數字 → 0.031／8% 循環／17 跨 source。**本條在 v5 耳測重審前不生效**；若重審仍敗，恢復判死並補消融證據。唯「咬字/詞無處繼承」的機制論證（下方 Died because ②）不受撤回影響——那是拼貼機制的定義，非實驗結果。
- **Tried:** `harmony/sampler_mouth.py`（build＝穩定 sustain 單元收割、render＝腦 notes.json 驅動拼貼）。girl 母音料 126 單元（E3–G5）先零成本驗管線；Harry 0619＋tenor 歌聲收割 805 單元（43–70 全半音階）。三代引擎：v1 靜態拼接＋等功率接縫；v2 歌唱手勢層（scoop −60c、legato varispeed 彎音 ≤4st＋接縫藏穩定段、句尾垂降、呼吸跟隨）；v4 熨平載體（單元除自身 50ms 包絡）＋零相位咬字蓋印（20ms 窗）。設計規格得證：全音階庫＋腦線 → 音高微修全 ≤50c（girl 中位 8c／Harry 中位 21c）。
- **Died because:** 四輪耳測全敗：girl v1「像鋼琴，沒有唱的過程」→ v2（手勢層、音高軌跡驗證有真滑音）「還是一樣，無對詞一起」→ v4（音節帶包絡相關 0.02→0.67→**0.878**，逼近轉換載體參照 0.951）「鬆鬆的音色皮，音跟音接的都不好」→ Harry 真嗓 805 單元版「成果不好，很不穩定」。根因三層皆結構性：①**無過渡體**——歌唱 legato 是發聲器官連續運動的錄音，兩段靜態 sustain 之間任何數學過渡（彎音/交叉淡接）都不是它（商用人聲庫靠真錄音程過渡樣本解此題）；②**無咬字繼承**——拼貼機制的輸入只有音符清單，現場人聲不進去，詞無處可繼承；包絡蓋印做到 r=0.878 仍被聽成「皮」；③歌聲收割單元先天不穩（音準散、中位 0.76s 靠循環撐、跨 session 拼貼感）。
- **Lesson（血價）：** 轉換載體（輸入＝現場人聲，詞/legato 免費繼承）與拼貼載體（輸入＝音符清單，一切要自建）是兩種機制——「對詞」只能來自前者。**音節帶包絡相關是必要非充分**：r≥0.88 耳朵仍判「音色皮」——蓋印的振幅敘事與載體的發聲機制無因果，質感維度再次逃過客觀指標（同 G28 教訓的變奏）。
- **Instead:** 選項⓪降級為**長音場景候補部件**（與②長音觸發天使婚配——你拉長音它用真嗓單元進場，該場景無過渡/咬字問題；未判死，但 Part B 錄音投資待 Harry 裁）。全旋律線天使唯一結構解＝轉換載體：現任 combsub＋(a) 同條件新料重訓。存活資產：`sampler_mouth.py` 全管線、單元庫索引（`out/units_{girl,harry}.json`）、音節帶(2–10Hz)包絡相關量測法（咬字耦合的客觀代理，`ddspgirl=0.951` 為轉換載體基準）。

### G33. Brain v3 串聯式條件生成（brain_v2 upper → v3_serial lower）— 盲聽墊底 2026-07-28
- **Tried:** `train_v3.py --serial`（同資料同架構，loss 只算 lower 位置＝專職第二天使；4,000 步 best val 0.3743）＋ `render_v3.py` serial cell（推論期 brain_v2 出 upper token 流 → v3_serial 以 (lead, upper) 為條件逐 tick 生 lower）。
- **Died because:** Harry 盲聽排序 **joint ≥ v1 > serial**＝墊底，連互不聞的 v1 都輸。統計同向歸因：黏線 28.1%（joint 14.2%、真實語料 16.8%）、upper 掉到 lead 下方 24.9%（joint 3.2%）——①外供的 v2-upper 本就不是「上聲部腦」也不知道 lower 存在；②訓練吃真 upper、推論吃生成 upper 的分佈落差（train_v3 docstring 本來就標記「誠實量測 serial 弱點」，量到了）。
- **Lesson:** 條件鏈的品質上限被最弱外供環節鎖死；「訓練真料、推論生成料」的 exposure bias 不是免費的。互撞感知要 by construction（同 tick 因果可見），不能靠事後串接。
- **Instead:** 聯合式＝v3 主線（joint interleave [lead,upper,lower]，chain rule 內建互撞感知）；`v3_serial/best.pt` 留檔供對照，不再投資。

## G33 — VocalSet 母音練習語料當 DDSP-SVC 音色來源（轉真歌用）（2026-08-04）
**死因＝詞素材庫存，不是方法。** 21.8 min 訓 male3（配方照抄 girl、val 谷底
14k＝1.007）：域內像人、跨歌手（male1 同曲）也像人＝**HuBERT units 不帶說話
人、有學到的詞會泛化**；但整包只有三首短歌的詞（義/拉丁/英童謠，2–4 min），
Harry 真歌一超出音素庫存＝decoder 硬外插＝「完全不像人」。VocalSet 擠不出更
多有詞素材 → 轉真歌的音色語料**必須是真歌**（girl 26 min 真歌＝反例證明）。
數字漂亮耳朵說不對的又一例：val loss 量在同分布練習音上。
**重開條件**：只拿它訓「無詞聲部」（哼鳴/母音 pad 類角色）或當真歌語料的補充
料——主料必須含詞。工件留用：`260804_vocalset/`（CC BY）、
`data_vs_male3/`、`exp/combsub-vs-male3/`、`vs_male3_trial.py`（鑑別配方）。

### G33b. 長音觸發同步 batch 塊式 live（sustain_live v0）— live FAIL 2026-08-05
- **Tried:** `harmony/sustain_live.py`：parselmouth 滑窗 f0 → 穩定 ~0.26s 觸發 → 母音回文循環＋HuBERT→CombSub→enhancer 1.2s 塊、續塊跟目標、150ms 釋放。工程數字全過（兩嘴塊渲染 301ms、音路＝應答式終判同款）。
- **Died because:** Harry live 三刀：①不及時（體感 ~1s：偵測窗實際 0.4–0.6s＋渲染 0.3s＋起音滑行；架構地板 ~0.4–0.5s＝修到底仍難及格）②不穩（回授螺旋：mic 聽到自己的 bass 99–106Hz 又觸發重定目標，73 次自進場；疊塊間換目標階梯）③不消失（回授令 f0 永遠有聲＝永不釋放）。回授與穩定可修，**延遲地板是架構性的**——batch 觸發做不出「一起唱」。
- **Instead:** 即時路線的下一注＝G28 案③ DDSP-SVC 滑窗串流（gui.py 架構、為 RT VC 而生、從未在天使上試過；先做延遲 spike 再決定蓋不蓋）。樂句層的應答式仍是本體。重開條件：若「陪唱」改為明確的展演段落（觀眾知道天使晚半秒進場＝美學設計而非缺陷）可復活。

### G34. 單支麥克風、能量域的回授判別（包絡閘門）— 量測 FAIL 2026-08-10
- **Tried:** 目標＝不動接線就分出「他的直達聲」與「天使繞回來的回授」，好讓腦不吃到天使的 f0（`GRAVEYARD:233`／G33b 同一機制在串流架構重現）。做法：我們一個樣本都不差地知道送去喇叭的 `y`，故估耦合增益 k 與播放延遲 d，判準＝`mic_env < thr·k·angel_env[t−d]` 即視為回授、不餵腦。理由：08-09 §N 量到包絡相關 r=+0.950 而相干性僅 0.0133＝**兩個時鐘毀掉的是相位不是包絡**，所以相位級 AEC 判負不蘊含包絡級判別不可行。工件 `harmony/scratchpad/fb_duck_probe.py`。
- **Died because:** **結構性，不是參數沒調好。** 迴路是 `angel(t)≈G·mic(t)`（系統輸出正比於輸入，diag17 實測 G 恆定：他不唱 0.97:1、他在唱也 0.97:1）而 `mic(t)=direct(t)+k·angel(t−d)` → **整個迴路的包絡自相似，任何 mic/angel 能量比都趨近常數**。實證：延遲掃 0–697ms 分離度**單調上升、無峰**（＝找不到物理回授路徑，只是「延遲越久相關越弱」的平凡效果），最佳分離僅 1.68×；門檻掃描擋掉 68% 回授要賠上 **26% 誤殺真唱**。附帶血訓：`dmp["mic"]` 與 `dmp["out"]` 在同一個 `step()` 裡 append（`spike_stream6.py:1020-1021`）＝dump 的 angel 是**產生時間軸**、不是喇叭播出的時間軸，直接互相關會得到「延遲 0ms、r=0.864」的假結果（那是輸入驅動輸出的因果相關）。
- **Instead:** 換維度而非換參數——能量以外的資訊。已試 CPP（見 G35）亦死。真正的下一步是**先錄一段乾淨標定素材**（他完全不出聲、天使照放喇叭 30–60s），才拿得到不含直達聲的 k／d／回授頻譜。**重開條件**：麥克風與輸出走同一時鐘（同一介面／Aggregate＋drift correction）後，相位級 AEC 本來就該重測（`scratchpad/aec_probe.py` 留著），屆時本條的能量域近似不再必要。
- **2026-08-16 更新（bank_live 架構下重測，判決不變但死因要限縮）：** 本條的結構性死因「迴路 `angel≈G·mic` 自相似 → 能量比恆定」**只對即時轉換載體成立**。bank_live 的 `y` 是音庫取樣播放（`bank_live.py:1864`），out 不是 mic 的函數，那條因果邊不存在——重測結果也確實不同：去趨勢後**延遲掃描有峰（279 ms, r=+0.249）**，負延遲側乾淨負相關，中位分離 **6.76×**（本條當年只有 1.68×）。**但仍不足以當門控**，而死因**不是**標定不準：本條與 G35 更正開的處方（另錄乾淨標定素材）其實 08-10 就已經做到了（`CALIB_*`），它反而證明耦合平坦（k(f) 跨 3.1 dB）、可標（−17.5 dB）、與盲估一致。真正的死穴是**殘差的語意**——用標定 k 算，他不出聲時 mic 仍比預測回授高 **+2.7 dB 中位**＝麥克風裡有與回授同量級的**非回授能量**（呼吸、氣音、動作聲、環境）。所以最佳門檻誤把回授當發聲 **36.6 %** 是定義上的、不是量測誤差，且不比已判死的鏡頭門控（閉嘴誤觸 20–35 %）好。詳見 **FINDINGS F26**、腳本 `scratchpad/bank_fb_envcorr.py`。**本條對串流架構維持判死**；bank_live 上是「未判死但判別力不足」，重開條件已改成——**換維度**（能分辨人聲發聲與呼吸／動作聲的資訊），而不是把同一個能量比標得更準。

### G35. CPP「單音 vs 三聲部和弦」當回授判別特徵 — 量測 FAIL 2026-08-10
- **Tried:** 承 G34 換維度。特徵理由來自 08-09 §N 的失敗歸因本身：「parselmouth 是單音追蹤器、hubert 是單人歌聲訓的，餵它們三聲部和弦＝分布外」——那個分布外就是判別特徵：**他唱＝單音（倒譜一根尖峰），天使繞回來＝三聲部和弦（峰被攤掉）**。和弦度不隨音量縮放，理應殺不掉於迴路自相似。工件 `harmony/scratchpad/fb_spec_probe.py`（93ms 窗、quefrency 70–400Hz）。
- **Died because:** 表面數字漂亮——最佳門檻 CPP=0.13dB、正確率 **88.9%**（baseline 全猜「在唱」＝70.0%，抓到真唱 96.9%／擋掉回授 70.4%）——但**它只是音量的代理**：CPP 與 log-RMS 相關 **r=+0.602**，且在 RMS 重疊區內按音量分箱後，同音量的 CPP 差只剩 **+0.027 dB**（回授 0.145 vs 真唱 0.172）。既然只是重講音量，就跟 `bridge_feed` 現有的音量判準沒有差別——而那個判準已經被回授撐住而失效。
- **Instead:** 同 G34：**瓶頸是素材不是特徵**。diag17 的「他不唱」ground truth 本身是用能量分位切出來的，兩組 RMS 差 10 倍且幾乎不重疊 → 任何特徵都與能量糾纏、無法解耦，這份素材原理上判不了任何判別法的生死。先錄乾淨標定素材（他不出聲、天使照放），CPP 與其他頻譜特徵才值得重測。**這是「數字漂亮耳朵/對照說不對」的又一例**（同 G33 VocalSet val loss、08-08 null 介入 +5.9% 假差異）：先問 ground truth 是怎麼切出來的。

> **G34／G35 同日更正（2026-08-10 晚，標定素材到手後）**：兩條的死亡判決維持，但**論據要分開看**。G34 的死因「迴路 `angel≈G·mic` 自相似 → 能量比恆定」不依賴任何 ground truth，**仍然成立**。G35 的「CPP 只是音量代理（同音量差 +0.027dB）」則是**用壞標籤算的**——那批 quiet 是拿 mic 能量 30 分位切的，其中「回授幀」音量中位 −22.6dB，而標定素材算出真回授應在 −39.7dB ＝ 標籤裡混的是他在唱的幀。**G35 應降級為「未判定」而非「已判死」**，要重測需一段「天使大聲且他確實不出聲」夠長的素材（diag17 裡這種時段只有 5.5 秒，因為橋接只撐 ~0.5s、天使很快衰減）。**更重要的是兩條的前提已經變了**：標定量到時鐘漂移僅 −31 ppm（1kHz 要 16 秒才跑掉半週期），**AEC 死於濾波器長度而非時鐘**（`B*K=24576` 樣本被 24151 樣本的純延遲吃光，只剩 425 樣本描述房間）→ 相位級 AEC 值得用「已知 547ms 延遲補償＋短濾波器」重做，做成了就不需要 G34/G35 這類能量／頻譜的近似判別。**血訓（跨這兩條）：先問 ground truth 是怎麼切出來的**——用音量切的標籤，不能拿來證明音量以外的特徵好不好，也不能拿來定音量門檻。

### G36. 樂句邊界雜音的信號級後處理（輸入整形×3＋輸出動態×2）— 耳測 FAIL×4＋指標紅燈 2026-08-10
- **Tried:** Harry 開模型層後，copy-synthesis 先判 vocoder 無罪（真 mel 過 NSF-HiFiGAN，Harry 耳測乾淨）→ 鎖定 mel 生成側。竊竊私語一度以「空隙靜音」實驗歸因於輸入空隙垃圾——**同日 Harry 推翻：他那輪的原話是「不算氣音竊竊私語，就是開頭以及結尾都有雜音」＝修正描述而非報告改變；且現象跨素材不變（clean30 不同素材同樣有開頭吸氣聲）、五路清輸入全數無效＝不是素材的問題**，主現象自始就是邊界渲染雜音（本條正文）。漏音存在是事實（Harry 在 copy-syn 裡聽到）但其可聽貢獻未證。剩下的樂句開頭/結尾雜音連試五路：①空隙數位靜音（gated）②空隙換循環房間底噪（cn）③換氣壓制＋餘弦起音坡＋150ms 釋放（v3）④輸入鍵控下行擴展器（v4）⑤音量解耦渲染（vd/vd2：vol 條件夾底讓嘴不進低音量區、渲後把真包絡乘回）。工件：`scratchpad/canon_take17_{gated,cn,v2,v3,vd}.py`、`boundary_metric.py`、`boundary_forensics.py`。
- **Died because:** 根因是**嘴在低音量區輸出 SNR 崩潰**（v27 早量過：音量掉時諧波掉 5.5×、噪聲只掉 1.5×；37.1s 時頻驗屍＝輸入 60ms 斜坡被嘴抹成 ~200ms 全頻帶霧）——樂句進出必穿低音量區，**信號級處理只能改霧的形狀不能除病**：坡放慢＝霧變長、坡加快／擴展器＝增益急變（Harry:「更多了」）、循環底噪＝週期假訊號、vd 快起音＝40ms 階差 33dB > 原版 25（同 v4 死因，被指標攔下未送耳）。**方法論教訓兩條**：⑴中位數軌跡會把偶發爆點洗掉＝尺與耳脫鉤，要逐事件驗屍＋逐事件對照輸入（子音本來就是高頻爆，「輸出有爆而輸入沒有」才是雜音）⑵能量指標聽不到「增益急變」，平滑度（短窗能量階差）要一起量——v4 敗後補的這把尺成功預測了 vd 也會敗。
- **Instead:** 病在模型＝解在模型：**重訓時做音量增強**（同 clip 多增益複製，讓嘴學會音色與音量解耦）或查 6.x 訓練配置是否已支援 vol aug；順帶目標＝onset 固有 +5dB 氣聲（輸入整形不可除，同屬渲染性格）。**live 不受此案影響**——橋接 gate 本來就在做同件事且 Harry 未在 live 抱怨邊界（住在正典/應答式管線的問題）。**重開條件**：換 vocoder 或嘴重訓後，本案五路的量測工具（boundary_metric/forensics）直接複用來驗收。

### G37. 凍結噪聲網格治「音不穩」（v33 --noise-grid）— 對照量測否決假設 2026-08-10
- **Tried:** 08-10 單聲部 live 輪 Harry 三嘴皆報「音不穩」。分項量測：live angel 對命令 f0 偏差 std 31c/12%>30c，而 live 追蹤誤差僅 10c、離線正典同嘴僅 9c ＝ live 渲染層的鍋。假設＝reflow 每窗獨立 randn（08-08 血訓「同輸入兩次不一樣」）令同一長音每 244ms 換一條音高路徑。實作 v33：三層守門參數（reflow.py `init_noise`／vocoder.py 透傳／spike_stream6 `--noise-grid`＋每聲部固定種子 16384 幀噪聲庫按絕對幀位查表）。旗標關＝Regime A 已證（canon 同種子重渲 max-abs-diff 0.0×4 檔）。
- **Died because（假設層面；程式碼保留）:** 重疊窗對照實驗（同段音訊、兩個錯位 0.244s 的窗各渲一次、量重疊區音高一致性）：**ts85 OFF 2.2c vs ON 2.6c、ts35（噪聲佔 65%）OFF 2.1c vs ON 2.2c ＝噪聲實現在任何工作點都不驅動音高**——ODE 速度場把不同噪聲拉回同一條音高軌跡。**真兇另有其人（同夜定位）**：|偏差| 按命令音高斜率分箱＝平段 7.1c（跟正典一樣準，對上 Harry「長音拉著是一條」）、陡段 74c、隱含時間錯位 ≈1.3 幀＝**15ms**＝**換音瞬間被接縫機械（SOLA/phase_vocoder）時間抹泥＋凍結網格把窗尾早期粗 f0 估計永久凍住**（正典用全上下文估計＝無此病）。
- **Instead:** 靶心移到**換音瞬間的接縫/凍結層**：候選＝「延遲凍結」（新幀先渲、成熟 m 塊後才凍，讓 f0 估計用到更多上下文；要對 v6 跳針的凍結初衷做取捨）、或換音段的 SOLA 行為檢討。`--noise-grid` 程式碼保留（無害、Regime A 已證），それ本身不解本案。**方法論**：實作前的機制假設要先用最小對照實驗打一次（本案 8 對重疊窗 20 分鐘就否決了，省下一輪 live 耳測）。

> **G34 追記（2026-08-18，Harry 本人重開 → 耳測 PASS）**：Harry 裁「live 只用麥克風就可以觸發」＝重開 08-16 判決。實作 bank_live v43 `--mic-gate`（能量域門，權威從鏡頭改 mic）。首版帶 CALIB 值上機翻車（門開 88.6%）；用首場 dump 調參（門檻 -26dBFS、本機當日耦合 -13.5）後二測，**Harry 判「我覺得可以」**。F26 的定義性死穴（呼吸/動作聲誤判 ~37%）沒有被繞過——是門檻拉高後誤開降到耳朵可接受（sim 誤開 9%）。本條對**串流架構**的判死不變；bank_live 上的狀態改為「能量域門控 live 服役中」。詳 worklog 08-18。

### G38. BLE HID 鍵盤按鈕（Pico W 官方 MicroPython）— 韌體缺配對支援 2026-08-18
- **Tried:** 實體鍵改藍牙路線（Harry 裁「那就藍牙」）：Pico W 上寫完整 BLE HID 鍵盤（HID service 0x1812＋report map＋bond 金鑰快閃持久化），按下＝送 Space，Mac 端零改動。工件：`firmware/pico_led_button/main_ble.py` 的第一版（git 歷史可考）。
- **Died because:** **官方 rp2 MicroPython（實測 v1.28.0）沒編入配對/bonding**——`ble.config(bond/le_secure/mitm/io)` 全部 `ValueError: unknown config param`（`MICROPY_PY_BLUETOOTH_ENABLE_PAIRING_BONDING` 未開），而 macOS 對 HID 輸入裝置強制加密配對＝無 bond 就不吃。CircuitPython 同板（HCI _bleio）同樣不支援 bonding。要走通得自編韌體＝viva 前不值得。
- **Instead:** **免配對自訂 GATT 通知服務**（同檔 `main_ble.py` 現行版）：Pico notify `TAP n`，`respond_shell._ble_loop`（bleak）收下轉 `self.tap()`＝與 UI space 同一條路同一個流水號。端到端 67/67 taps 實測通。**重開條件**：哪天官方韌體編入 pairing（或願意自編韌體），HID 版第一版碼在 git 裡直接撈回來。
