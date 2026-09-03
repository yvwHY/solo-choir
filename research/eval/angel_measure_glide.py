"""Objective glide meter for the angel-portamento symptom.
Usage: python measure_glide.py <wav> [label]
Reports: voiced%, glide-time% (median-smoothed |slope| > 4 st/s),
note-hold segments (>=200ms within +-0.5 st) count + median length.
"""
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "harmony"))
from pitch import yin_f0, hz_to_midi  # noqa: E402

FRAME, HOP = 2048, 512  # @48k: 42.7ms window, 10.7ms hop

wav, label = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
x, sr = sf.read(wav, dtype="float32", always_2d=True)
x = x[:, 0].astype(np.float64)

midi = []
for i in range(0, len(x) - FRAME, HOP):
    f0 = yin_f0(x[i : i + FRAME], sr)
    midi.append(hz_to_midi(f0) if f0 else np.nan)
midi = np.array(midi)
dt = HOP / sr
voiced = ~np.isnan(midi)

# median smooth ~150ms to kill vibrato, then slope
k = max(3, int(0.15 / dt) | 1)
sm = np.full_like(midi, np.nan)
for i in np.flatnonzero(voiced):
    w = midi[max(0, i - k // 2) : i + k // 2 + 1]
    w = w[~np.isnan(w)]
    if len(w) >= k // 2:
        sm[i] = np.median(w)
slope = np.abs(np.diff(sm)) / dt  # st/s, nan where unvoiced
gliding = slope > 4.0

# note-hold segments: contiguous voiced runs staying within +-0.5 st of their median
holds = []
run = []
for i in range(len(sm)):
    if not np.isnan(sm[i]):
        run.append(sm[i])
        if np.ptp(run) > 1.0:  # +-0.5 around center
            run = [sm[i]]
    else:
        if len(run) * dt >= 0.2:
            holds.append(len(run) * dt)
        run = []
if len(run) * dt >= 0.2:
    holds.append(len(run) * dt)

v = voiced.sum()
g = np.nansum(gliding & ~np.isnan(slope))
vs = (~np.isnan(slope)).sum()
print(f"[{label}] dur {len(x)/sr:.1f}s  voiced {v/len(midi):.0%}")
print(f"[{label}] glide-time {g/max(1,vs):.0%} of voiced (smoothed |slope|>4 st/s)")
print(f"[{label}] holds>=200ms: n={len(holds)}, median {np.median(holds) if holds else 0:.2f}s, "
      f"total {sum(holds):.1f}s")
