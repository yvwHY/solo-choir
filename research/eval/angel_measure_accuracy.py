"""Did the angel actually sing the brain's notes?
Compares the rendered angel's YIN track against <out>.targets.json.
Only scores hold windows (target == previous tick's target, skipping the
first 60ms of each tick) so transitions don't count against accuracy.
Usage: python measure_accuracy.py <angel.wav>
"""
import json
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "harmony"))
from pitch import yin_f0, hz_to_midi  # noqa: E402

FRAME, HOP = 2048, 512
wav = sys.argv[1]
meta = json.load(open(wav + ".targets.json"))
tick_sec, targets = meta["tick_sec"], meta["targets"]

x, sr = sf.read(wav, dtype="float32", always_2d=True)
x = x[:, 0].astype(np.float64)
midi = []
for i in range(0, len(x) - FRAME, HOP):
    f0 = yin_f0(x[i : i + FRAME], sr)
    midi.append(hz_to_midi(f0) if f0 else np.nan)
midi = np.array(midi)
dt = HOP / sr

errs, scored = [], 0
for k in range(1, len(targets)):
    t = targets[k]
    if t is None or targets[k - 1] != t:
        continue  # rests and transition ticks don't score
    lo = int((k * tick_sec + 0.06) / dt)
    hi = int(((k + 1) * tick_sec) / dt)
    w = midi[lo:hi]
    w = w[~np.isnan(w)]
    if len(w) < 3:
        continue
    scored += 1
    err = np.median(w) - t
    err -= 12 * round(err / 12)  # octave-agnostic (YIN octave blips)
    errs.append(err * 100)  # cents
errs = np.array(errs)
print(f"scored hold-ticks: {scored}")
print(f"|err| median {np.median(np.abs(errs)):.0f} cents, "
      f"within +-50c: {(np.abs(errs) <= 50).mean():.0%}, "
      f"within +-100c: {(np.abs(errs) <= 100).mean():.0%}")
