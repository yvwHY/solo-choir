"""Separation measurement for using a clap or a finger snap as an acoustic cue.

The problem: if the "your turn" signal in the exhibition is a clap, the detector
has to tell a clap apart from the singer still singing. The energy route was
already dead, with only 2 dB of headroom. This tries attack shape instead: a
wideband sharp attack (clap or snap) against a harmonic slow attack (singing)
against a slow fade-in (the parts bleeding back in).

The honest standard, the same one used to measure bleed: not clap against average
singing, but running the same detector across all the singing material to find the
moment that most resembles a clap - a plosive consonant is the likeliest false
positive - and measuring its headroom against the weakest clap. It passes at 6 dB
or more.

The detector uses two dimensions, deliberately simple enough to compute per hop
inside a callback:
  rise = how far the short-window energy in dB climbs within about 12 ms
  hf   = the share of spectral energy above 4 kHz, that is wideband and inharmonic
  trigger = rise >= R and hf >= H. R and H come from a grid search for an operating
  point that takes every clap with zero false triggers on singing and bleed, and
  the headroom is reported.

Usage (conda environment vcclient-dev):
  record claps: python clap_probe.py --record out/claps_$(date +%y%m%d).wav
                (standing, at exhibition distance; ten claps after the countdown,
                more than 1 s apart)
  record snaps: as above with a snaps_* filename
  analyse:      python clap_probe.py --analyze out/claps_*.wav out/snaps_*.wav
                (the singing and bleed material is fixed; --sing and --bleed override)
"""
import argparse
import sys

import numpy as np
import soundfile as sf

SR = 44100
WIN, HOP = 256, 128            # about 5.8 ms window, 2.9 ms hop
RISE_HOPS = 4                  # rise spans 4 hops, about 11.6 ms
HF_CUT_HZ = 4000
EPS = 1e-10

SING_DEFAULT = ["out/resp2_live_260802_234841.wav"]        # 252 s of real singing, standing
BLEED_DEFAULT = [                                          # worn, with the parts playing
    "out/live_mic_260731_225804.wav",
    "out/live_mic_260731_230150.wav",
    "out/live_mic_260731_233041.wav",
]


def features(x):
    """Compute (rise_db, hf_ratio, db) per hop. Fully vectorised, so 252 s takes about a second."""
    n = (len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, WIN), strides=(x.strides[0] * HOP, x.strides[0]))
    # a -80 dBFS floor: digital silence is -200 dB, and without a floor the rise
    # out of silence inflates to over 100 dB and drowns the attack-shape difference
    # this is actually measuring
    db = np.maximum(
        20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + EPS), -80.0)
    rise = np.concatenate([np.zeros(RISE_HOPS), db[RISE_HOPS:] - db[:-RISE_HOPS]])
    spec = np.abs(np.fft.rfft(frames * np.hanning(WIN), axis=1)) ** 2
    cut = int(HF_CUT_HZ * WIN / SR)
    hf = spec[:, cut:].sum(axis=1) / (spec.sum(axis=1) + EPS)
    return rise, hf, db


def pick_events(rise, hf, db, k, min_gap_s=0.5):
    """Pick k events from a clap file: peaks in rise, at least min_gap apart, returning (rise, hf) for each."""
    gap = int(min_gap_s * SR / HOP)
    order = np.argsort(rise)[::-1]
    picked = []
    for i in order:
        if all(abs(i - j) >= gap for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    # an event's hf is the maximum within 2 hops of the peak, since the rise peak and the spectral peak can be half a window apart
    return [(float(rise[i]),
             float(hf[max(0, i - 2):i + 3].max()),
             float(db[i])) for i in sorted(picked)]


def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == SR, f"{path}: {sr} != {SR}"
    return np.ascontiguousarray(x)


def analyze(cue_paths, sing_paths, bleed_paths, n_events):
    cues = []                                        # [(rise, hf, db, path)]
    for p in cue_paths:
        r, h, d = features(load(p))
        cues += [(*e, p) for e in pick_events(r, h, d, n_events)]
        print(f"[cue] {p}: took {n_events} events, rise "
              f"{min(e[0] for e in cues[-n_events:]):.1f}–"
              f"{max(e[0] for e in cues[-n_events:]):.1f} dB")

    bg = {}                                          # path -> (rise, hf, db)
    for p in sing_paths + bleed_paths:
        bg[p] = features(load(p))
        kind = "sing" if p in sing_paths else "bleed"
        print(f"[{kind}] {p}: {len(bg[p][0])} hops, "
              f"max rise {bg[p][0].max():.1f} dB")

    # grid search: R from 6 to 40 dB, H from 0.05 to 0.6, for "every clap taken, zero false triggers on background"
    best = None
    for R in np.arange(6, 40.5, 0.5):
        for H in np.arange(0.05, 0.61, 0.01):
            if not all(r >= R and h >= H for r, h, _, _ in cues):
                continue                             # a missed clap fails
            fp = sum(int(np.sum((rr >= R) & (hh >= H)))
                     for rr, hh, _ in bg.values())
            # headroom = the weakest clap's rise in dB above the largest background rise among hops with hf >= H
            bg_max = max((rr[hh >= H].max() if (hh >= H).any() else -np.inf)
                         for rr, hh, _ in bg.values())
            margin = min(r for r, _, _, _ in cues) - max(bg_max, R)
            cand = (fp == 0, margin, R, H, fp)
            if best is None or cand > best:
                best = cand
    if best is None:
        print("\nverdict: fail - no (R,H) takes every clap event")
        return
    ok, margin, R, H, fp = best
    print(f"\noperating point R={R:.1f} dB, H={H:.2f}  false triggers {fp} hops")
    weakest = min(cues, key=lambda c: c[0])
    print(f"weakest clap rise {weakest[0]:.1f} dB (hf {weakest[1]:.2f}, "
          f"level {weakest[2]:.1f} dBFS, {weakest[3]})")
    if ok:
        print(f"verdict: {'pass' if margin >= 6 else 'marginal (under 6 dB; measure the false-trigger rate for real)'}"
              f" - headroom {margin:.1f} dB")
    else:
        print(f"verdict: fail - no zero-false-trigger operating point exists (at best {fp} false hops)")


def record(path, secs):
    import sounddevice as sd
    dev = sd.query_devices(kind="input")
    print(f"input device: {dev['name']} (check this is the right microphone)")
    for t in (3, 2, 1):
        print(f"  {t}...", flush=True)
        sd.sleep(1000)
    print(f"recording {secs}s - clap ten times, more than a second apart")
    x = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="float64")
    sd.wait()
    peak = float(np.abs(x).max())
    sf.write(path, x, SR)
    print(f"saved {path}  peak {20*np.log10(peak+EPS):.1f} dBFS"
          + ("  clipped" if peak > 0.99 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", metavar="OUT_WAV")
    ap.add_argument("--secs", type=float, default=20)
    ap.add_argument("--analyze", nargs="+", metavar="CUE_WAV")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--sing", nargs="*", default=SING_DEFAULT)
    ap.add_argument("--bleed", nargs="*", default=BLEED_DEFAULT)
    a = ap.parse_args()
    if a.record:
        record(a.record, a.secs)
    elif a.analyze:
        analyze(a.analyze, a.sing, a.bleed, a.events)
    else:
        ap.print_help()
        sys.exit(1)
