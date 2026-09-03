# Open-license SVC / singer-model / transcription survey (2026-06-26)

Research sweep (GitHub + Hugging Face, web-verified) for the Solo Choir "one singer → choir" project.
Goal lens: free/open license usable in thesis + 6/29 exhibition + possible public release; two layers —
**(A) live CPU-real-time** and **(B) offline/GPU high-quality**. Licenses flagged per item.
Current baseline: Beatrice v2 (restricted-license engine) + rule-based diatonic SATB + CREPE transcription.

## Headline findings
1. **No open engine gives CPU-realtime + zero-shot together** (mid-2026). Beatrice's CPU-realtime niche is
   still ~unique. Everything zero-shot (Seed-VC, kNN-VC, FreeVC, OpenVoice) is GPU/offline in practice.
2. **The male→female / "neutral" problem is a *mechanism* problem, not a model problem.** Beatrice is
   source-dependent ([[beatrice-source-dependent-conversion]]). Reference-copying engines (kNN-VC) or
   per-part *trained* models (RVC / so-vits) sound genuinely female where Beatrice goes neutral. Leakage
   never fully disappears — documented field-wide.
3. **"4 distinct SATB members from one singer" = speaker-embedding interpolation / reference-pool blending.**
   Open paths: so-vits `spk_mix`, RVC model-fusion, kNN-VC pool-mixing, FreeVC/embedding SLERP. This is the
   open analog of the commercial Vocoflex morph.
4. **Flat-pitch multi-syllable splitting is an ONSET problem, not f0.** No pitch model fixes it. Best cheap
   win = improve onsets (superflux + RMS-novelty) on top of the current CREPE pipeline. Best low-register
   f0 upgrade = **SwiftF0** (MIT, ONNX, no TF, CPU-realtime).

---

## Layer A — live CPU-real-time-capable

| Tool | What | License | Notes for us |
|---|---|---|---|
| **Beatrice v2** (current) | realtime SVC | **restricted, non-MIT** (git-ignored) | the baseline; CPU-realtime is its point |
| **DDSP-SVC** (yxlllc) | lightweight realtime SVC (diff DSP) | **MIT** (verify LICENSE text) | lightest non-Beatrice CPU per-pass; needs training; best candidate for an extra live converted voice |
| **RVC** (RVC-Project) | trained-per-voice VC + retrieval index | **MIT** ✅ | closest workflow match to Beatrice; CPU marginal, GPU preferred; supports model-fusion to fabricate voices |
| **so-vits-svc-fork** (voicepaw) | so-vits + realtime mic | license **unconfirmed** (parent AGPL) ⚠️ | **unmaintained since ~2023**; AGPL contamination risk — lower priority |
| **SwiftF0** (f0, see transcription) | f0 estimator | **MIT** | ONNX, no TF/PyTorch, CPU-realtime — drop-in CREPE replacement |

## Layer B — offline / GPU high-quality (fine for the SATB render path)

| Tool | What | License | Notes for us |
|---|---|---|---|
| **kNN-VC** (bshall) | any-to-any, reference-pool, no training | **MIT** ✅ | **top "supplement" pick** — copies a *reference* timbre (not a trained neutral speaker) → directly attacks male→female; blend pools to fabricate SATB; CPU works (slow), MIT-clean |
| **Seed-VC** (Plachtaa) | zero-shot VC + **singing**, diffusion | **GPL-3.0** ⚠️ + **archived Nov 2025** | best zero-shot quality; copyleft = caution on distribution; GPU |
| **FreeVC / FreeSVC** (OlaWod) | one-shot VC; FreeSVC = singing variant | engine **MIT** ✅ (FreeSVC weights unverified) | MIT-clean; older quality; embedding interpolable |
| **Vevo** (Amphion) | zero-shot imitation, disentangled | **MIT** ✅ | bleeding-edge research; offline |
| **OpenVoice V2** / **CosyVoice 2/3** | tone-color / TTS w/ VC mode | **MIT** / **Apache-2.0** ✅ | TTS-oriented, weak fit for *sung* harmony; use for generating timbre references only |
| **Diff-SVC**, **so-vits-svc** (orig) | predecessors | GPL-ish / **AGPL-3.0** ⚠️ | superseded + unmaintained — skip |

---

## Singer / female / choir models & datasets

**Honest bottom line:** there is **no clearly-permissive, ready-to-ship female *singing* checkpoint** to drop
in. Clean path = MIT engine (kNN-VC / FreeVC) + open female/SATB data (below), or self-train RVC/so-vits.

**Permissive data (thesis + exhibition + release safe):**
- **M4Singer** (HF `umoubuton/m4singer`, GitHub) — **MIT** ✅ — ~700 songs, 20 pros, **explicit SATB labels**
  (`m4_soprano/alto/tenor/bass`) + scores. **Top recommendation** for per-part target voices / kNN-VC pools.
- **Choral Singing Dataset (CSD)** Zenodo 1286570 — **CC BY 4.0** — real SATB, 4/section, isolated tracks + MIDI.
- **Cantoría** Zenodo 5851070 — **CC BY 4.0** — pro SATB quartet, individual tracks.
- **Dagstuhl ChoirSet** Zenodo 4618287 — **CC BY 4.0** — multitrack SATB + F0/score annotations.
- **ESMUC Choir** Zenodo 5848990 — **CC BY** (verify version).
- **VocalSet** (Zenodo / HF `stemsai/vocalset`) — **CC BY 4.0** (verify) — 20 pros (9 F), vowels/techniques; great kNN-VC reference for a controllable female timbre.
- **PJS** — **CC BY-SA 4.0** — solo JP singing (single voice).

**Non-commercial — use for experiments/thesis, DON'T ship (flagged):**
- **OpenSinger** — **CC BY-NC-SA** ⚠️ — biggest female volume (30 h / 41 female singers).
- **Opencpop** (HF `espnet/ace-opencpop-segments`) — **CC BY-NC** ⚠️ — clean solo female.
- **so-vits-svc 4.x base** (`Sucial/...`) — **CC BY-NC-SA** ⚠️.

**Avoid for the exhibition:** community RVC "singer" checkpoints (voice-models.com etc.) — licenses
unspecified / IP-tainted (vtuber/celebrity/Splice). Inspiration only, not shippable.

---

## Transcription (the active pipeline — CREPE + librosa onsets)

**Octave jumps in the low male register → swap the f0 model:**
- **SwiftF0** (lars76) — **MIT** — ONNX + ONNXRuntime, **no TF/PyTorch**, CPU-realtime (~42× CREPE), range
  down to ~47 Hz, strong octave accuracy. **Best fit** (tiny isolated dep vs our TF env). Replace CREPE's f0.
- **RMVPE** (Dream-High) — **Apache-2.0** — gold-standard vocal f0 robustness; PyTorch, heavier.
- **FCPE** (torchfcpe) — **MIT** — fast, PyTorch; CPU RTF unverified.
- **torchcrepe** — **MIT** — CREPE weights without TF (PyTorch); doesn't fix octaves (same model).
- **pYIN** (`librosa.pyin`) — **ISC** — zero new deps, decent with `fmin~65Hz`; cheap fallback.

**Flat-pitch syllable splitting = onset work (no f0 model fixes it):**
- Stay on **librosa** (**ISC**, numpy-2 safe): switch flux to **`superflux`** + add an **RMS/energy-novelty**
  onset channel and OR them — syllable re-attacks on a held note show as amplitude dips. Cheapest real gain.
- Borrow **CREPE Notes** (xavriley) *idea* (pitch-gradient + inverse-confidence onsets) but **NOT the package**
  — it's **GPL-3.0 + pulls madmom (non-commercial + numpy-2 broken)** ⚠️.
- **madmom** = best learned onsets but **non-commercial clause + np.float numpy-2 breakage** ⚠️ (why we skipped it).
- **ROSVOT** (RickyL-2000) — **MIT** — purpose-built sung-note transcriber, **GPU/offline** — option for the
  offline "score mode" Record→transcribe path.
- Basic Pitch — **Apache-2.0** (permissive, not restricted) — our prior baseline; smears low/flat-multisyllable.

---

## Timbre-morph / "fabricate SATB from one singer"
- **Commercial north-star:** Dreamtonics **Vocoflex** — real-time latent morph, 10 s target, **closed/paid
  ($159+), KYC + watermarked** ⚠️ — comparison baseline only ([[vocoflex-research]]).
- **Open, usable today:** so-vits **`spk_mix`** (AGPL ⚠️) / **RVC model-fusion** (MIT ✅) for explicit
  embedding mixing; **kNN-VC pool-blend** (MIT ✅, simplest CPU experiment); **FreeVC embedding SLERP** (MIT ✅).
  SLERP > linear for fewer artifacts (VoxMorph).
- **Research-only:** NANSY++ (best "voice designing" idea, **no usable code/weights**); embedding-interp papers
  (method is reusable inside RVC/FreeVC/so-vits).
- **Skip for SATB voices:** RAVE (**CC-BY-NC** ⚠️ + texture, not speaker-faithful), DDSP (instrument-only).

---

## Recommended next experiments (mapped to our project)
1. **kNN-VC (MIT) offline experiment** — feed M4Singer/CSD per-part (S/A/T/B) reference pools; test whether a
   real female reference beats Beatrice's neutral female on the *render* path. Lowest code, license-clean.
2. **SwiftF0 (MIT) f0 swap** in `transcribe_crepe.py` — tiny ONNX dep, may kill the low-register octave
   errors AND drop the heavy TF/numpy-2 env entirely (no separate env needed). High value / low risk.
3. **Onset upgrade** — add superflux + RMS-novelty to the librosa onset step (cheap, numpy-2 safe).
4. **Embedding-interpolation (SLERP)** to fabricate 4 SATB identities — prototype in RVC (MIT) model-fusion.

All four are additive and don't touch the validated live engine. Constraint per [[post-meeting-direction-2026-06-23]]:
frame as a research-comparison layer; don't switch the live engine before viva.
