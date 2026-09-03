"""Objective scrape/click meter for shift-jump artifacts (F2 risk).

Self-controlled: compares high-frequency spectral flux in TRANSITION windows
(the exact tick times where the brain moved the target, from targets.json)
against HOLD windows of the same file. A --glide 0 scrape would show up as
localized broadband transients right after shift jumps.

Detector validation: --inject writes a copy with known 1ms discontinuity
clicks inside hold regions; the meter must flag those (proves sensitivity).

Usage: python scrape_meter.py <angel.wav> [--inject]
"""
import json
import sys

import numpy as np
import soundfile as sf

NFFT, HOP = 512, 128
HF_HZ = 3000.0

wav = sys.argv[1]
inject = "--inject" in sys.argv
meta = json.load(open(wav.replace("_clickcheck", "") + ".targets.json"))
tick_sec, targets = meta["tick_sec"], meta["targets"]

x, sr = sf.read(wav, dtype="float32", always_2d=True)
x = x[:, 0].astype(np.float64)

# classify each tick: transition (target changed, incl. onset) vs hold
trans_t, hold_t = [], []
for k in range(1, len(targets) - 1):
    t0 = k * tick_sec
    if targets[k] is None:
        continue
    if targets[k] != targets[k - 1]:
        trans_t.append(t0)
    elif targets[k + 1] == targets[k]:
        hold_t.append(t0 + tick_sec / 2)  # mid-hold, far from any jump

if inject:
    y = x.copy()
    n_inj = 0
    for t in hold_t[::2]:
        i = int((t + 0.02) * sr)
        if i + 48 < len(y) and np.abs(y[i : i + 48]).max() > 1e-3:
            y[i : i + 48] = -y[i : i + 48]  # 1ms polarity flip = a real click
            n_inj += 1
    out = wav.replace(".wav", "_clickcheck.wav")
    sf.write(out, y.astype(np.float32), sr)
    print(f"injected {n_inj} clicks into hold regions -> {out}")
    x = y

# HF spectral flux
win = np.hanning(NFFT)
frames = []
for i in range(0, len(x) - NFFT, HOP):
    frames.append(np.abs(np.fft.rfft(x[i : i + NFFT] * win)))
S = np.array(frames)
hf = S[:, int(HF_HZ * NFFT / sr):]
flux = np.sqrt(np.sum(np.maximum(0.0, np.diff(hf, axis=0)) ** 2, axis=1))
fdt = HOP / sr


def window_max(t0, t1):
    a, b = int(t0 / fdt), int(t1 / fdt)
    return flux[a:b].max() if b > a and b <= len(flux) else None


tw = [m for t in trans_t if (m := window_max(t - 0.02, t + 0.08)) is not None]
hw = [m for t in hold_t if (m := window_max(t - 0.05, t + 0.05)) is not None]
tw, hw = np.array(tw), np.array(hw)
thresh = np.percentile(hw, 99)
print(f"windows: transition n={len(tw)}, hold n={len(hw)}")
print(f"hold   flux: median {np.median(hw):.4f}, p99 {thresh:.4f} (= click threshold)")
print(f"trans  flux: median {np.median(tw):.4f}, p95 {np.percentile(tw,95):.4f}, max {tw.max():.4f}")
flagged = (tw > thresh).sum()
print(f"transitions over hold-p99: {flagged}/{len(tw)} ({flagged/len(tw):.0%}) "
      f"[chance level ~1% if transitions are artifact-free]")
