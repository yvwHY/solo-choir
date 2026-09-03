# Solo Choir — Voice-Model Training Notes (the voice-asset track)

*Parallel human track that feeds better timbres into the software instrument. The app's multi-pass architecture stays the same regardless of where a voice comes from — this track just improves the **quality** of each part. Draft + running log; record experiments here as they happen.*

---

## 1. Goal

Real, convincing **male (Tenor/Bass) and female (Soprano/Alto)** timbres for the full SATB demo and the viva, rather than only approximating female voices by formant-shifting a male model. Trained Beatrice voices are the quality ceiling; the embedding-probe + formant path is the provisional stand-in until they exist.

## 2. Baseline already done

- Own **tenor** model: ~**25 minutes** of own singing → **Beatrice V2** → **2000 steps** (`paraphernalia_00002000`). This is the validated voice in the engine today.
- A working training pipeline already exists, so adding voices and re-training is a known, low-risk process — not a research unknown.

## 3. Two-stage plan

1. **Provisional (now, free, unblocks the form video):** probe `target_speaker` in the existing Beatrice embeddings (~229 slots, multi-speaker base incl. female) + `formant_shift` fine-tune. No training, no DSP change. Get a rough male/female SATB working and validate the multi-pass architecture + UI.
2. **Trained (toward the full demo / viva 8/26):** collect and train real voices for the parts that need them (female S/A; optionally a distinct male B). Swap them into the same architecture once they sound good.

## 4. Collecting voices

- ~25 min of singing per voice (matching the proven pipeline), ideally with **varied pitch, vowels, and dynamics** so the model generalises beyond the exact takes.
- **Consistent recording conditions** (same mic, room, level) across a voice's session.
- **Consent + attribution:** other people's voices will appear in a thesis / public demo — get explicit permission and record it (e.g. a `CREDITS` entry), the same discipline used for the head mesh.
- **Preprocessing → cleaner clips (2026-06-26):** `tools/preprocess_voice.py INPUT OUTDIR` does denoise
  (noisereduce, gentle ~0.8) → drop silence/breaths (voiced intervals only) → RMS loudness-normalise
  (peak-safe) → slice into 10s training clips. NO dereverb (record fairly dry). For a *passable* model the
  lever is **clean + consistent**: run it on ONE consistent session (same mic/room/level) — clean beats
  more messy hours (see the `combo50` mixed-session result below). Tunables: `--top-db` (silence), `--denoise`,
  `--target-dbfs`, `--clip-s`. noisereduce/librosa/soxr already in the env.

## 5. Step-count experiment (2000 → 5000?)

More steps is **not automatically better** — on a small (~25 min) dataset, too many steps **overfits**: the model reproduces the training audio well but can degrade on pitches/vowels/dynamics it didn't see.

Method:
- Train with **checkpoints saved along the way** (e.g. 2000 / 3500 / 5000).
- **A/B by ear on held-out singing** (material NOT in the training set); pick the best-sounding checkpoint, not the highest number.
- Keep the **2000-step model as the baseline** to compare against.
- Watch for overfitting signs: metallic/buzzy artefacts, pitch instability, over-smoothed timbre.
- Often **more / more-varied data** helps generalisation more than simply increasing steps on the same 25 min.

## 6. Key integration question (confirm before committing to many voices)

Does a trained voice plug in as:
- **(A) a new speaker embedding inside the same base model** — switchable via `target_speaker`, light, fits the multi-pass architecture cleanly (preferred); or
- **(B) a separate model file per voice** — heavier (memory + load), and fewer can run simultaneously live.

This determines how many distinct trained voices can be active **live** vs only in offline render. Confirm against Beatrice v2 before investing in many models. (Also worth testing: can **one** female model + per-part formant/octave cover **both** S and A, instead of two separate models?)

## 7. Constraints that do NOT change with training

- **Real-time pass limit is unchanged.** Each converted part = +1 inference pass (RTF ~0.3/pass). Better models improve *timbre*, not the pass count → **live ≤ ~3 parts; full SATB via offline render** (Record→Play, stems ready).

## 8. Licensing & storage

- Trained models are **license-restricted** like the tenor model → **kept out of git** (git-ignored), backed up separately (off-laptop copy).
- Record donor consent for any non-own voices.
- **JVS base model** (`beatrice_paraphernalia_jvs`, used for the female S/A voices in the B′ path):
  trained on the **JVS corpus + JVS-MuSiC** — **non-commercial / academic use only** (unauthorized
  commercial use prohibited). Kept out of git (gitignored, referenced by absolute path), credited here.
  Sources: JVS corpus & JVS-MuSiC, Shinnosuke Takamichi et al. Swap in a self-trained female model
  before any commercial use.

---

## Experiment log
*(append results here — date, voice, steps, dataset length, A/B verdict, artefacts noticed)*

- _2000 steps, own tenor, ~25 min — current baseline in engine._
- _2026-06-20, own tenor, **new F2–C5 recording-protocol dataset** (0619.wav ~21 min,
  sliced into 10 s clips for training). Held-out high-note blind A/B (60 s, C major) across
  rounds: **new25-2k > new25-5k > old2k**, and **combo50 (old+new ~50 min) worst**. Ear and
  metrics agree (new25-2k tenor F0-err p90=90, in-tune=82.6% — both best; 5k regressed →
  overfit). **Conclusions: (1) the F2–C5 protocol data beats the old model (steadier highs);
  (2) sweet spot ≈ 2000 steps, 5000 overfits on ~25 min; (3) merging old+new data (combo) is
  counter-productive. Adopted `new25-2k` as the new tenor baseline.** Eval harness:
  `eval/ablation_eval.py`; design `specs/2026-06-19-recording-protocol-design.md`._

---

## Appendix — "Study" / practice logging (ear-trainer research vehicle — NOT voice training)

Different track from the rest of this doc (kept here for lack of a better home). "Study" is the
**data-logging layer for the pitch-matching ear-trainer** — the thesis's *listening-test / comparison*
layer (per the Yee-King direction). It is a research tool, not a performance feature; that is why the
live UI now keeps it behind the dock's **Advanced ⌄** disclosure (Task 3.3).

**The two independent variables** (`#studyCtl` in `ui/solo_choir_ui_v10.html`):
- **subject** — free-text label for *who* is practising (participant ID).
- **channel** — `bone` / `headphone` / `speaker`: *how the target reference is delivered to the ear*.
  This is the key variable — bone-conduction (the wearable mask) vs headphone vs speaker.

**Research question it measures:** does hearing the target pitch through **bone-conduction** let a
singer match pitch *better* than through headphone/speaker? (= "body as interface" — the wearable
mask as a pitch-feedback channel.)

**How a practice attempt is captured & logged:**
1. Record a take → **Render** (produces per-part stems).
2. In the part tabs, **solo ONE part** → enters *practice mode* (`state.practice`, `_playSel()`): the
   app plays that part's stem as the target and analyses your live mic against it.
3. UI feedback: your sung note turns **green + ✓** when within **40 ¢** of the target; else **♯ high /
   ♭ low** (`ui` tick, `state.practicePart`).
4. While playing+singing, `_attempt` accumulates `voiced`, `matched` (on-target time), `absC` (Σ|cents|·dt).
5. On leaving/switching the part, `flushAttempt()` → `pywebview.api.log_practice(...)` appends **one row** to
   `recordings/practice_log.csv` (gitignored) via `app/bridge.py:log_practice`.

**CSV columns** (`recordings/practice_log.csv`):
`part, key, scale, voiced_s, match_pct (=matched/voiced), mean_abs_cents (=absC/voiced), on_target_s, subject, channel`.

**To run a study:** set subject + channel → solo a part → sing to match → repeat across
`channel` (bone→headphone→speaker) and subjects → compare `match_pct` / `mean_abs_cents` per channel.
Analysis is offline over the CSV (no in-app charting).

**Methodology caveats (the delivery channels are compared SEQUENTIALLY, not simultaneously):**
- The machine has **one 3.5 mm jack**, and the mask already uses both sides of it (L = bone-conduction,
  R = exciter). You therefore **cannot** monitor bone + headphone
  at once, and you don't need to: this is a **blocked, within-subject** comparison. Do one channel as a
  block (plug device → pick it in Output device → set `channel` label → several attempts), then physically
  **swap** the device and start the next block. Only ever one output live at a time.
- **No guard-rail on the label:** `channel` is free-set text; if you forget to update it after swapping the
  device, the block is mislabelled. Set it *before* each block and double-check.
- **Counterbalance to kill the learning-effect confound:** a singer simply gets more accurate with practice,
  independent of channel — so a fixed bone→headphone→speaker order would make the *last* channel look best
  just from warm-up. **Randomise / counterbalance the channel order across subjects** (e.g. Latin square).
  Expect this to be asked at viva.
- Simultaneous same-session A/B would need multiple outputs (a multi-out USB interface or an aggregate
  device splitting streams) — **not required** by the blocked design; noted only so it isn't reinvented.
