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

## Singing corpora

- **M4Singer** — Zhang et al., *NeurIPS 2022 Datasets and Benchmarks*.
  CC BY-NC-SA. Six of the shipping voice models are trained on it.
- **JVS corpus** and **JVS-MuSiC** — Takamichi et al. Academic /
  non-commercial. Base corpus for the Beatrice-family models.
- **CPDL** (Choral Public Domain Library) — public-domain chorale scores, used
  to train the harmony-brain models in `harmony/checkpoints/`.

## Human voices

One model in the shipping Respond path is trained on a friend's voice, recorded
in person with consent, anonymous by their request. One model is the author's own
voice. Both are documented in [`docs/VOICE_CREDITS.md`](docs/VOICE_CREDITS.md).

## Third-party assets in this repository

- `app/assets/head.glb`, `three.min.js`, `GLTFLoader.js`, Space Grotesk web
  fonts — see [`app/assets/CREDITS.md`](app/assets/CREDITS.md).
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
