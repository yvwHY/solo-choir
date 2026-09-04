# research — not part of the performing instrument

Everything here was made during the project but does not run in the piece. It is
kept because it is the evidence behind the decisions in `docs/`.

| Directory | What it is |
|---|---|
| `eval/` | Measurement scripts: latency probe, intonation metric, predicted MOS, ablation, the listening-test page |
| `harmony_experiments/` | Probes, labs, corpus scrapers and abandoned engine lines. Most reference paths that only existed on the development machine |
| `studio/` | An offline arrange application. Built alongside the instrument, not shown with it |
| `web/roll/` | A browser score-roll prototype |
| `prints/` | 3D-print batches for the table rig and the wearable frame: STL, DXF, G-code and per-batch notes |

These files are deliberately **not** kept working against `config.py`. They ran
once, against the machine they were written on, and the numbers they produced are
recorded in `docs/FINDINGS.md` and `docs/PROCESS.md`.
