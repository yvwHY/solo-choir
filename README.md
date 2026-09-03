# Solo Choir

A real-time instrument that turns one singer into a choir. You sing a phrase into
a microphone; the system re-synthesises your voice and answers with harmony
voices, on CPU, low latency, around a table.

**Yu-Ting (Harry) Liao** — MA Computational Arts, Goldsmiths, University of
London, 2026. Final project.

- Documentation video: **https://vimeo.com/1223687812**
- Technical documentation: [`docs/SOLO_CHOIR_TECH_DOC.md`](docs/SOLO_CHOIR_TECH_DOC.md)
- Research framing: [`docs/SOLO_CHOIR_RESEARCH_VISION.md`](docs/SOLO_CHOIR_RESEARCH_VISION.md)
- Voice provenance and consent: [`docs/VOICE_CREDITS.md`](docs/VOICE_CREDITS.md)

---

## The piece

Solo Choir sits on a table. Four speakers are placed around it, with a head model
and a boom microphone in the middle. A wearable version of the same instrument is
shown next to the table.

The performing application (`app/respond_shell.py`) has four modes, switched live
with keys 1–4:

| # | Mode | What it does |
|---|---|---|
| 1 | **Respond** | Sing a phrase, then pause (or tap the button on the mic). A choir answers, in your own voice. |
| 2 | **Live** | Keep singing; a sampled choir harmonises with you in real time. |
| 3 | **Live + neural** | The sampled choir joined by two neural voices. |
| 4 | **Live · neural** | Neural voices only, following you as you sing. |

Modes 1 and 2–4 are served by different engines, spawned as subprocesses by the
same shell: `harmony/respond2.py` (Respond), `harmony/bank_live.py` (sampled
choir), `server/solo_min.py` (neural). See
[`docs/VIVA_RUNBOOK.md`](docs/VIVA_RUNBOOK.md) for the frozen launch
configuration actually used in performance, flag by flag, with the reason each
flag is set.

## Architecture in one paragraph

Three layers, with the audio path deliberately separated from control and
telemetry. The **engine** runs as a subprocess and owns all live audio (mic →
conversion → diatonic harmony → output). The **bridge** (pywebview `js_api`)
carries only telemetry (engine stdout → UI) and control (UI → engine stdin);
**audio never passes through the bridge**. The **UI** is a native macOS window
served from local HTML. This keeps the interface free to be a rich WebGL surface
without ever entering the audio-critical path.

## Repository layout

```
app/           Application shells (pywebview) and the double-click launchers
harmony/       Live engines: Respond, sampled choir, harmony brain, pitch, reverb
  checkpoints/   Trained harmony-brain models (this project's own weights)
  scratchpad/    Small calibration models loaded at run time
server/        Beatrice wrapper, the 5-part neural engine, the diatonic harmoniser
ui/            The four interface pages (plain HTML/JS/three.js)
firmware/      MicroPython for the Raspberry Pi Pico: the BLE phrase-end button
hardware/      Fusion 360 scripts for the wearable mask frame
docs/          Technical documentation, findings, dead ends, runbook, worklogs
research/      Work that is not part of the performing instrument:
               evaluation scripts, the offline arrange studio, experiment scripts,
               the score-roll web prototype, and the 3D-print batches
config.py      Every path that points outside this repository, in one place
models/        Not in git — see models/README.md
```

## Running it

The instrument needs two Python environments, on purpose. The UI shell needs
`pywebview` and `bleak`; the engines need `torch`, `mediapipe` and a NumPy 2.x
build. Installing both into one environment forces a NumPy downgrade that changes
the engines, so the shell spawns the engines as subprocesses in their own
interpreter. Both are Python 3.10.

```bash
# 1. UI shell environment
python3.10 -m venv .venv-app
.venv-app/bin/pip install -r requirements-app.txt

# 2. Engine environment — this is a DDSP-SVC checkout with its own venv;
#    see models/README.md for which fork and how it is set up.

# 3. Tell the code where everything lives
export SOLO_CHOIR_MODELS=/path/to/solo-choir-models
export SOLO_CHOIR_PY_APP=$PWD/.venv-app/bin/python

# 4. Run
$SOLO_CHOIR_PY_APP app/respond_shell.py
```

On the machine the piece was built on, `app/開啟應答app.command` is
double-clicked instead. It is a `.command` and not a `.app` deliberately: the
terminal window has to stay open, because Ctrl-C is the recovery path during a
performance and the start-up self-check prints there, and because macOS
microphone permission follows Terminal.app.

Every path that leaves this repository is resolved in [`config.py`](config.py),
from environment variables with defaults under `models/`. Nothing else in the
runtime hard-codes an absolute path.

## Models

**The voice models and the conversion engine are not in this repository.** They
are either too large for git (five 207 MB checkpoints) or not ours to
redistribute (a non-commercial engine build, models derived from
non-commercial corpora, and one model of a private individual's voice).

[`models/README.md`](models/README.md) lists every model the instrument loads:
what it is, whose voice is in it, which corpus it was trained on, and under what
licence. [`docs/VOICE_CREDITS.md`](docs/VOICE_CREDITS.md) records consent for the
one non-corpus human voice in the shipping path.

The trained harmony-brain weights in `harmony/checkpoints/` **are** included —
they are this project's own models, trained on public-domain chorale scores, and
they contain no voice data.

## Documentation

| Read this | When |
|---|---|
| [`docs/SOLO_CHOIR_TECH_DOC.md`](docs/SOLO_CHOIR_TECH_DOC.md) | Architecture, data flow, design decisions, the AI-assisted process |
| [`docs/SOLO_CHOIR_RESEARCH_VISION.md`](docs/SOLO_CHOIR_RESEARCH_VISION.md) | The research framing: the body as an interface |
| [`docs/VIVA_RUNBOOK.md`](docs/VIVA_RUNBOOK.md) | The frozen performance configuration and pre-show check (Chinese) |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | Measured numbers: latency, glide, decorrelation, RTF |
| [`docs/GRAVEYARD.md`](docs/GRAVEYARD.md) | Approaches that were tried and died, with root causes |
| [`docs/DEBUG_PLAYBOOK.md`](docs/DEBUG_PLAYBOOK.md) | Diagnosed live-audio symptoms |
| [`docs/TRAINING_NOTES.md`](docs/TRAINING_NOTES.md) | Voice-model training: data, steps, experiments |
| [`docs/HARDWARE_ROADMAP.md`](docs/HARDWARE_ROADMAP.md) | The wearable: bone conduction, haptics, components |
| [`docs/VERIFICATION.md`](docs/VERIFICATION.md) | How audio-path changes were proved not to change the sound |
| [`docs/worklogs/`](docs/worklogs) | 53 dated development logs, June–August 2026 |

Some documents are in Chinese; they were written as working notes. The technical
documentation, research framing, findings and dead-end registry are in English.

## Attribution and licence

Solo Choir is built on a fork of [`w-okada/voice-changer`](https://github.com/w-okada/voice-changer)
(MIT). The upstream server, web client, trainer and Docker tooling are **not**
included here — only the files this project added or changed. Full attribution
for the upstream code, the conversion engine, and the singing corpora is in
[`NOTICE.md`](NOTICE.md).

The code in this repository is released under the MIT licence
([`LICENSE`](LICENSE)). The models it loads are not, and are not distributed with
it.
