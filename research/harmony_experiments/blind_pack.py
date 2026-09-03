"""Blind-pack angel stems for ear tests: one mix recipe for every cell.

Each stem is mixed against the take with the SAME recipe (0.5 take +
0.8 peak-normalized stem, 48 kHz, peak 0.9), shuffled to stim_N.wav, and
the mapping written to key.json -- listen first, peek after.

Run (vcclient-dev):
  .../python blind_pack.py <take.wav> <outdir> name=stem.wav[=trim_ms] ...

trim_ms drops the stem's head to cancel a known render latency (e.g. the
Beatrice engine path lags its input ~50 ms; WORLD stems have none) so no
column drags behind the take for a reason that isn't the variable under test.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 48000


def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != TARGET_SR:
        x = soxr.resample(x, sr, TARGET_SR)
    return x / max(1e-9, np.abs(x).max())


def main():
    take = load(sys.argv[1])
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    cells = [a.split("=") for a in sys.argv[3:]]
    order = list(range(len(cells)))
    random.shuffle(order)
    key = {}
    for stim_i, cell_i in enumerate(order, 1):
        name, stem_path = cells[cell_i][0], cells[cell_i][1]
        trim_ms = float(cells[cell_i][2]) if len(cells[cell_i]) > 2 else 0.0
        stem = load(stem_path)[int(TARGET_SR * trim_ms / 1000):]
        n = min(len(take), len(stem))
        mix = 0.5 * take[:n] + 0.8 * stem[:n]
        mix = mix / max(1e-9, np.abs(mix).max()) * 0.9
        sf.write(outdir / f"stim_{stim_i}.wav", mix, TARGET_SR, subtype="PCM_16")
        key[f"stim_{stim_i}"] = {"cell": name, "stem": stem_path, "trim_ms": trim_ms}
    json.dump(key, open(outdir / "key.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote {len(cells)} stimuli to {outdir}/ (key.json = answers, peek AFTER)")


if __name__ == "__main__":
    main()
