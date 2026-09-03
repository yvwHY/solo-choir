"""input_health — pre-flight health check for the mic→interface input chain.

Born from the 2026-07-14 bench-debug day (DEBUG_PLAYBOOK rows: wedged-state crackle,
thin-You/low-gain): records the interface's RAW input directly (bypasses the app),
then checks the three things that failed that day:

  LEVEL    performance-volume peaks should land ~0.2–0.8 FS
           (0.03 = the undetected low-gain state; >0.85 = clipping risk)
  CRACKLE  impulsive events per voiced second (4 kHz high-pass, 8-sigma)
           (healthy take measured ~0.6/s; wedged AI-Micro measured ~35/s)
  SNR      voiced RMS vs floor RMS — run in a quiet room with the app CLOSED,
           otherwise monitoring bleed lands in the floor and the number lies

Run (vcclient-dev), then SING AT PERFORMANCE VOLUME for the whole recording:
  python eval/input_health.py                     # AI-Micro, 12 s
  python eval/input_health.py --device throat --seconds 20

Exit code 0 = all PASS/WARN, 1 = any FAIL (usable in a pre-session script).
If CRACKLE fails: power-cycle the interface (full unplug→replug) and re-run
BEFORE theorising — see DEBUG_PLAYBOOK.
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48000


def pick(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i
    raise SystemExit(f"input device not found: {name}")


def verdict(label: str, value: str, level: str, hint: str = "") -> bool:
    mark = {"PASS": "✅ PASS", "WARN": "⚠️  WARN", "FAIL": "❌ FAIL"}[level]
    print(f"  {label:<8} {value:<28} {mark}" + (f"  — {hint}" if hint else ""))
    return level != "FAIL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="AI-Micro")
    ap.add_argument("--seconds", type=float, default=12.0)
    args = ap.parse_args()

    dev = pick(args.device)
    info = sd.query_devices(dev)
    ch = min(2, info["max_input_channels"])
    print(f"device: [{dev}] {info['name']}  ({ch} ch)")
    print(f"🔴 recording {args.seconds:.0f}s — SING AT PERFORMANCE VOLUME the whole time")
    for n in (3, 2, 1):
        print(f"  {n}…", flush=True)
        time.sleep(1)
    x = sd.rec(int(args.seconds * SR), samplerate=SR, channels=ch, device=dev, dtype="float32")
    sd.wait()
    L = x[:, 0]

    # frame RMS (100 ms) → voiced = top quartile, floor = bottom 15%
    n = int(0.1 * SR)
    rms = np.sqrt((L[: len(L) // n * n].reshape(-1, n) ** 2).mean(axis=1))
    voiced_mask = rms > np.percentile(rms, 75)
    voiced = rms[voiced_mask].mean()
    floor = rms[rms < np.percentile(rms, 15)].mean()
    voiced_s = voiced_mask.sum() * 0.1

    peak = float(np.abs(L).max())
    snr = 20 * np.log10(voiced / floor) if floor > 0 else np.inf

    # impulsive crackle: 4 kHz high-pass, 8-sigma outliers, grouped within 10 ms
    from scipy.signal import butter, filtfilt

    b, a = butter(4, 4000 / (SR / 2), "high")
    hp = filtfilt(b, a, L)
    idx = np.where(np.abs(hp) > 8 * np.std(hp))[0]
    events = len(np.split(idx, np.where(np.diff(idx) > SR * 0.01)[0] + 1)) if len(idx) else 0
    rate = events / voiced_s if voiced_s > 0 else 0.0

    print("\nresults:")
    ok = True
    if peak < 0.1:
        ok &= verdict("LEVEL", f"peak {peak:.3f} FS", "FAIL", "no/tiny signal — mic coupling? gain? (RØDE Central)")
    elif peak < 0.2 or peak > 0.85:
        ok &= verdict("LEVEL", f"peak {peak:.3f} FS", "WARN", "target 0.2–0.8; adjust interface gain")
    else:
        ok &= verdict("LEVEL", f"peak {peak:.3f} FS", "PASS")

    if rate > 8:
        ok &= verdict("CRACKLE", f"{events} events, {rate:.1f}/voiced-s", "FAIL", "power-cycle the interface, re-run (playbook)")
    elif rate > 2:
        ok &= verdict("CRACKLE", f"{events} events, {rate:.1f}/voiced-s", "WARN", "re-run; if persistent, power-cycle the interface")
    else:
        ok &= verdict("CRACKLE", f"{events} events, {rate:.1f}/voiced-s", "PASS")

    if snr < 20:
        ok &= verdict("SNR", f"{snr:.1f} dB (voiced {20*np.log10(voiced):.1f} / floor {20*np.log10(floor):.1f} dBFS)", "FAIL", "noise before the ADC — but check the room/app first")
    elif snr < 30:
        ok &= verdict("SNR", f"{snr:.1f} dB", "WARN", "only meaningful with app closed + quiet room")
    else:
        ok &= verdict("SNR", f"{snr:.1f} dB", "PASS")

    if ch == 2:
        R = x[:, 1]
        corr = float(np.corrcoef(L, R)[0, 1]) if np.std(R) > 0 else 0.0
        if corr < 0.9 and np.sqrt((R**2).mean()) < 0.2 * np.sqrt((L**2).mean() + 1e-12):
            verdict("L/R", f"corr {corr:.2f}", "WARN", "channels differ — engine reads ch1 only; check input port")
        else:
            verdict("L/R", f"corr {corr:.2f}", "PASS")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
