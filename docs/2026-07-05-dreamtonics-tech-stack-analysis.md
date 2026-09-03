# Dreamtonics tech-stack analysis + our position

- **Date**: 2026-07-05
- **Purpose**: reverse-engineer Dreamtonics' audio-AI stack (Synthesizer V Studio 2 Pro, Vocoflex, Choir Voice Collections, their vocoder) from a senior ML/audio-engineer view, and locate Solo Choir relative to it.
- **Provenance**: two `deep-research` harness runs (fan-out web search → fetch → adversarial verify). The automated *synthesis* step failed both times on the account session limit, so this doc is **hand-synthesized** from the verified claims + the run's official-source claims + prior repo memory. Confidence is tagged per line: **✅ verified** (3-0 / 2-1 adversarial vote), **⚠ official-but-unverified** (from Dreamtonics/press, verification blocked by session limit), **🔍 inferred** (my analysis, not a source).
- **Related memory**: [[choir-decorrelation-recipe]], [[vocoflex-research]], [[ensemble-comb-tradeoff]].

---

## A. Company & research lineage
- ✅ **Kanru Hua (华侃如)** — founder of Dreamtonics, creator of Synthesizer V / SynthV Studio; Tokyo-based; studied at UIUC. (CCRMA talk listing)
- ✅ **His DSP foundation is open-source**: [`libllsm2` / LLSM v2](https://github.com/Dreamtonics/libllsm2) — a **classical DSP two-layer source-filter speech model**: layer 0 = harmonic-plus-noise (harmonic amplitudes/phases + noise PSD); layer 1 = source-filter re-interpretation (smooth spectral envelope ≈ vocal-tract transfer function + glottal-model params). Synthesis = **Pulse-by-Pulse (PbP)**: each glottal period generated in the frequency domain, filtered by the layer-1 vocal-tract response, back to time via **overlap-add IFFT**. This is his analysis/synthesis engine for pitch/timbre manipulation.
- ✅ His work "**bridges speech signal processing with generative models**" → the stack is **hybrid DSP + neural**, not pure end-to-end. (audio.dev speaker bio)
- ✅ **Attribution correction (do not misstate in the thesis)**: the **Neural Homomorphic Vocoder (NHV)** patent [US11842722B2](https://patents.google.com/patent/US11842722B2) is by **Kai Yu, Zhijun Liu, Kuan Chen (SJTU / AISpeech)** — **NOT** Kanru Hua / Dreamtonics. NHV is the *academic template* for source-filter neural vocoders, not Dreamtonics IP.

## B. Acoustic model + vocoder (Synthesizer V Studio 2 "AI")
- ✅ **The offline voice-generation path is an end-to-end trained neural network** — evidence: it embeds **inaudible watermarks** directly into generated audio (implies a single learned model, not DSP concatenation). (audio.dev)
- ✅ **Entirely offline, local CPU, ~300% faster than v1, no GPU / accel hardware** — via algorithm redesign + multi-threading. (SynthV2 announce; 2-1)
- ✅ **Smart Pitch Controls**: user enters key pitch locations → system generates natural pitch curves in near-real-time (a learned pitch-curve prior). (SynthV2 announce)
- ✅ **AI Retakes**: the models preserve finer human voice *dynamics*; those dynamics feed AI Retakes to generate more diverse alternate takes. (SynthV2 announce; 2-1)
- 🔍 Exact vocoder not disclosed, but a **source-filter neural vocoder in the NHV family (~15 kFLOPs/sample)** is what makes CPU real-time feasible, and matches his LLSM/DSP lineage → **inferred: source-filter neural hybrid**.

## C. Timbre representation & conversion (Vocoflex)  — light-verified 2026-07-05 (WebFetch, MusicTech + Sound on Sound reviews)
- ✅ Build a target voice from a sample **as short as 10 s** ("as little as 10 seconds… far less than competing products that require 15 minutes"), OR randomly generate from gender + tone.
- ✅ **XY pad = gender (X) × tone (Y)**; each target voice shown as a **curve made of dots (small portions of the analysed audio)**; with multiple targets, "curves/dots are distributed around the screen, with **similar voices placed nearer each other to make morphing smoother**"; morph **interpolates between voice models → new voices**. (🔍 extrapolation: inferred, not explicitly confirmed.)
- ✅ Fastest **"Realtime mode" ≈ 35 ms**; users disable Realtime for higher fidelity. (⚠ the "≈105 ms highest-quality" figure came from an earlier source, not re-confirmed here.) ⚠ CPU-only/offline stated as "runs on their own local computer" but CPU-vs-GPU not spelled out in these reviews (prior Dreamtonics claim = local CPU).
- 🔍 Closest open equivalents: **kNN-VC, Seed-VC, StreamVC** (global speaker embedding via FiLM — in principle interpolable), **VoxMorph** (Slerp over embeddings). Per [[vocoflex-research]]: **no open stack achieves the full combo (interpolable embedding + CPU real-time + singing/choir) → that combo is a genuine novel contribution.**

## D. Choir Voice Collections (Synthesizer V, score/lyric-driven — NOT a live voice-changer)  — light-verified 2026-07-05 (WebFetch, Bedroom Producers Blog)
- ✅ Built via **deep learning + spatial signal processing**: record full choir sections with a **microphone array** → dissect into individual soloists while retaining the choral timbre → **rebuild a controllable model**. ("recorded full choir sections using a microphone array … combines deep learning with spatial signal processing" — described as a **"hybrid modeling approach"**.)
- ✅ Each collection = **16 individual voices, four per part (4 S / 4 A / 4 T / 4 B)**; ensemble **scales 1 soloist → 16 singers**.
- ✅ Controls (**correction: legato, NOT vibrato**): *"adjust **pitch and timing coherency** to dial in **tightness or lushness**, and fine-tune **consonant articulation, brightness, and legato**"* — per part.
- ⚠ Whether it is a generative model vs sample-triggering is **not stated** (called a "hybrid modeling approach"); do NOT assert "generative, not sample-based" as fact.
- 🔍 Their stated principle — *"isolated choir singers sing loose in pitch+time; recombined it snaps into place"* — is the **same perceptual mechanism Solo Choir independently reproduced offline via WORLD** (see appendix). Their edge = real-choir-trained inter-singer interaction; ours = same effect from decorrelating one converted voice.

## E. Training data & engineering
- ✅ The through-line is **offline CPU real-time**, achieved by **cheap source-filter vocoders + algorithm/multithread optimization** (no GPU). Data scale/licensing not public (🔍 proprietary recorded voice DBs + mic-array choir sessions).

## F. Reproducibility — map to open source + difficulty
| Capability | Closest open stack | Reproducibility | Hardest part |
|---|---|---|---|
| **Choir dispersion** | our **WORLD per-voice pitch/timing decorrelation** (built this session) | 🟢 ~80% perceptually | matching real-choir-trained inter-singer naturalness |
| **SynthV acoustic + vocoder** | **`libllsm2` (their own open DSP)** / NNSVS / DiffSinger + source-filter neural vocoder | 🟡 medium | end-to-end neural quality + large curated singing data |
| **Vocoflex morph** | kNN-VC / Seed-VC / StreamVC (FiLM global embedding) | 🔴 hard | interpolable-embedding **+** CPU real-time **+** quality, all at once |

---

## Three takeaways for Solo Choir
1. **`libllsm2` is Dreamtonics' own open-source high-quality DSP source-filter vocoder** → a candidate **replacement for WORLD** in our per-voice pitch/timing/formant decorrelation (same-ecosystem author, purpose-built for this manipulation). Evaluate.
2. Dreamtonics' moat = **source-filter (DSP-guided) neural + CPU real-time + large curated data**. Solo Choir already lives in the **CPU-real-time source-filter** space (Beatrice + WORLD).
3. **Solo Choir independently reproduced their core choir technique** (pitch+timing dispersion). The differentiator is the thesis angle **"a choir of one" — the crowd is all *you*** (one person decorrelated into many), where Dreamtonics uses many real singers. That is a defensible, original framing.

## Sources
- [libllsm2 (Dreamtonics, LLSM v2 DSP vocoder)](https://github.com/Dreamtonics/libllsm2)
- [Kanru Hua — audio.dev speaker](https://conference.audio.dev/speakers/kanru-hua/)
- [Kanru Hua — CCRMA talk](https://ccrma.stanford.edu/events/singing-voice-synthesis-towards-state-of-art-dreamtonics-inc-tokyo-japan)
- [Announcing Synthesizer V Studio 2 Pro](https://dreamtonics.com/announcing-synthesizer-v-studio-2-pro/)
- [Neural Homomorphic Vocoder (Interspeech 2020)](https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html) · [NHV patent US11842722B2](https://patents.google.com/patent/US11842722B2)
- [Announcing Vocoflex](https://dreamtonics.com/announcing-vocoflex-real-time-voice-morphing-plugin/) · [Vocoflex](https://dreamtonics.com/vocoflex/)
- [Real Choir Voice Collections](https://dreamtonics.com/choir-voices/) · [Choir Voice Collections plugin](https://dreamtonics.com/choir-voice-collections/)
- Light-verify reviews (2026-07-05): [MusicTech Vocoflex review](https://musictech.com/reviews/plug-ins/dreamtonics-vocoflex-review-vocal-plugin-ai/) · [Sound on Sound Vocoflex](https://www.soundonsound.com/news/real-time-voice-morphing-dreamtonics-vocoflex) · [Bedroom Producers — Choir Voice Collections](https://bedroomproducersblog.com/2026/02/26/dreamtonics-choir-voice-collections/)

## Appendix — Solo Choir's own choir finding (2026-07-04, for cross-reference)
Turning one voice into "many people" needs **independent per-voice PITCH *and* TIMING decorrelation (both; either alone insufficient; timbre NOT necessary)** + independent vibrato (after cancelling the original) + stereo panning. Neural detune fails (~5–10¢ quantization deadzone; SATB = distinct pitches, no beating). Numbers: real-choir F0 dev 0–50¢ (mean ~20¢), timing dev ~100 ms (synth caps onset std ~20–40 ms), ~7 differentiated voices ≈ 10 real singers. Working offline recipe = WORLD resynthesis per clone. Full detail in [[choir-decorrelation-recipe]]; prototype in this session's scratchpad (`choir_decorr2.py`).

> **Note on completeness**: the automated adversarial-verify + synthesis of C/D/F was blocked by the account session limit (three run attempts). Angle A/B = adversarially verified by the harness (3-0 / 2-1). **C/D were then light-verified by hand (2026-07-05) via direct WebFetch of MusicTech / Sound on Sound / Bedroom Producers reviews** — most ⚠ claims are now ✅ with quotes, and one error was corrected (Choir control is **legato**, not vibrato). Still open/soft: Vocoflex CPU-vs-GPU specifics, the ≈105 ms figure, generative-vs-sample for the choir. Re-run `deep-research` (resume `wf_775d4cec-812`) with budget headroom to fully close these.
