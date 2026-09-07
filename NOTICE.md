# Attribution

## Upstream code

Solo Choir began as a fork of **[w-okada/voice-changer](https://github.com/w-okada/voice-changer)**,
MIT licence, © 2022 Wataru Okada. The upstream licence text is kept as
[`LICENSE-upstream-voice-changer`](LICENSE-upstream-voice-changer).

This repository contains only the files this project added or changed. The
upstream server application, the React web client, the trainer, the Docker
tooling and the tutorials are not included. The parts of `server/` kept here are
this project's own files: `beatrice_converter.py`, `solo_min.py`,
`mouth_gate.py`, `score_export.py`, `arranger.py`, `studio_api.py`,
`voice_changer/SoloChoir.py` and `voice_changer/naive_shift.py`. The full fork,
with its history, is at
[`yvwHY/solo-choir-vcclient`](https://github.com/yvwHY/solo-choir-vcclient)
(branch `ui-app`).

## Voice conversion engine

**Beatrice v2**, used as a compiled engine driven from a Python harness
(`server/beatrice_converter.py`). Academic / non-commercial terms. The engine
build and its models are not redistributed here.

## Voice synthesis code

The Respond and sampled-choir engines import and drive **DDSP-SVC**
(**[yxlllc/DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)**, MIT licence) from a
checkout outside this repository: `harmony/phrase_render.py`,
`harmony/world_live.py` and `harmony/spike_stream6.py` call its `ddsp.vocoder`,
`ddsp.core`, `enhancer` and `encoder` modules. Its checkout path is configured
through `SOLO_CHOIR_DDSP` / `SOLO_CHOIR_DDSP6X`; see
[`models/README.md`](models/README.md). DDSP-SVC in turn brings in:

- **HuBERT-Soft** — [bshall/hubert](https://github.com/bshall/hubert), MIT
  licence, used as the content encoder. Method: van Niekerk et al., *A comparison
  of discrete and soft speech units for improved voice conversion*, ICASSP 2022.
- **NSF-HiFiGAN** — the vocoder and the enhancer. HiFi-GAN: Kong, Kim and Bae,
  NeurIPS 2020; the neural source-filter model: Wang, Takaki and Yamagishi,
  *IEEE/ACM TASLP* 28, 2020.
- The **DDSP** formulation itself: Engel, Hantrakul, Gu and Roberts, ICLR 2020.

**MediaPipe Face Landmarker** ([google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe),
Apache 2.0) provides the blendshapes the camera gate and the vowel work were
built on. Lugaresi et al., arXiv:1906.08172.

## Singing corpora

- **M4Singer** — Zhang et al., *NeurIPS 2022 Datasets and Benchmarks*.
  CC BY-NC-SA. Six of the shipping voice models are trained on it.
- **JVS corpus** and **JVS-MuSiC** — Takamichi et al. Academic /
  non-commercial. Base corpus for the Beatrice-family models.
- **CPDL** (Choral Public Domain Library) — public-domain choral scores.
- The **JSB chorales** distributed with **music21** (Cuthbert and Ariza, MIT
  licence) — public-domain scores.
- **POP909** — Wang et al., *ISMIR 2020*, released for research use. Only the
  melody and chord annotations were used, to derive a second voice
  (`research/harmony_experiments/silver_line.py`); the dataset itself is not
  redistributed here.

### Which corpus is in which harmony-brain checkpoint

The checkpoints in `harmony/checkpoints/` are this project's own weights and
contain no voice data, but they are not all trained on the same corpus:

| Checkpoint | Trained on |
|---|---|
| `v3_joint/`, `v3_serial/` (the shipping models) | JSB chorales and CPDL |
| `v2/` | JSB chorales, CPDL, and lines derived from POP909 |
| `best.pt` (v1) | JSB chorales |

## Human voices

One model in the shipping Respond path is trained on a friend's voice, recorded
in person with consent, anonymous by their request. One model is the author's own
voice. Both are documented in [`docs/VOICE_CREDITS.md`](docs/VOICE_CREDITS.md).

## Third-party assets in this repository

- `three.min.js`, `GLTFLoader.js`, Space Grotesk web fonts — see
  [`app/assets/CREDITS.md`](app/assets/CREDITS.md). The head mesh the early
  interface used (`head.glb`, iamsunroy on Sketchfab, Sketchfab Standard
  licence) is not redistributed; the same file says where to get it.
- `research/web/roll/sf/` — WebAudioFont player and soundfont; see
  [`research/web/roll/sf/CREDITS.md`](research/web/roll/sf/CREDITS.md).
- `research/harmony_experiments/` and `.claude/skills/` include workflow skills
  adapted from third-party sources; see
  [`_reference/skills/ATTRIBUTION.md`](_reference/skills/ATTRIBUTION.md) where
  present.

## Libraries

PyTorch, NumPy, SciPy, librosa, soundfile, soxr, sounddevice, praat-parselmouth,
pyworld, torchcrepe, torchfcpe, MediaPipe, OpenCV, pywebview, bleak, three.js.
Each under its own licence.
