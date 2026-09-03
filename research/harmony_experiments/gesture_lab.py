"""R1 root fix: transplant REAL pitch gestures from Harry's corpus.

Harry (07-24): every parametric humanize variant sounded the same -- the
autotune feel is structural, because the f0 contour is SCRIPTED. This lab
removes the script entirely: every note the brain sings gets its contour
from a real sung note in his corpus (onset scoop, sustain life, vibrato,
release), selected by matching approach-interval and duration, time-warped
onset-anchored, re-centered on the target note. Zero synthetic curves --
the only numbers left are selection weights, not gesture shapes.

Stage 1 (mine, cached):  0619 clean clips -> note segments ->
    cents-deviation contours (vs the segment's own median, so his tuning
    error is not transplanted -- F16: the note center stays the target's).
Stage 2 (assemble): brain notes.json -> per-note unit pick -> warped dev
    contour -> f0 curve -> WORLD mouth render (reuses humanize_lab.analyse).

Run (retraining venv):
  venv/bin/python gesture_lab.py <take.wav> out/angel_v2_world2_notes.json
Outputs out/gest_r1_{angel,mix}.wav + gesture cache out/gesture_corpus.npz
"""

import glob
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pyworld

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from humanize_lab import analyse  # noqa: E402  (WORLD analysis + borrow policy)

FRAME_MS = 5.0
FPS = 1000.0 / FRAME_MS  # 200 frames/s
CORPUS = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260618/test/man/0619_clean_clips"
CACHE = HERE / "out" / "gesture_corpus.npz"
MIN_SEG_S = 0.22
SEED = 20260724


def mine_corpus():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return list(z["units"])
    units = []
    files = sorted(glob.glob(f"{CORPUS}/*.wav"))
    print(f"mining {len(files)} clips...")
    for i, path in enumerate(files):
        with wave.open(path) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
        x = x.astype(np.float64)
        f0, t = pyworld.harvest(x, sr, frame_period=FRAME_MS)
        midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
        # median-smooth the rounded track to segment on stable notes
        r = np.full(len(midi), -99.0)
        ok = np.isfinite(midi)
        r[ok] = np.round(midi[ok])
        seg_lo = None
        prev_center = None
        prev_end = -1e9
        k = 0
        while k <= len(r):
            same = k < len(r) and seg_lo is not None and r[k] == r[seg_lo]
            if seg_lo is None:
                if k < len(r) and r[k] > 0:
                    seg_lo = k
            elif not same:
                seg = midi[seg_lo:k]
                if len(seg) >= MIN_SEG_S * FPS and np.isfinite(seg).all():
                    center = float(np.median(seg))
                    dev = np.clip(100.0 * (seg - center), -180, 180)
                    gap_s = (seg_lo - prev_end) / FPS
                    itv = (None if prev_center is None or gap_s > 0.3
                           else float(np.round(center - prev_center)))
                    units.append({"dev": dev.astype(np.float32),
                                  "dur": len(seg) / FPS, "itv": itv})
                    prev_center = float(np.median(seg))
                    prev_end = k
                seg_lo = k if k < len(r) and r[k] > 0 else None
            k += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)}: {len(units)} units")
    np.savez(CACHE, units=np.array(units, dtype=object))
    return units


def pick(units, dur_t, itv_t, rng, recent):
    best, best_s = None, 1e9
    for j, u in enumerate(units):
        s = abs(np.log(max(u["dur"], 0.05) / max(dur_t, 0.05)))
        if itv_t is None:
            s += 0.0 if u["itv"] is None else 0.8
        elif u["itv"] is None:
            s += 0.8
        else:
            s += 0.25 * abs(u["itv"] - itv_t)
        if j in recent:
            s += 0.6
        s += 0.05 * rng.random()  # tie-break variety
        if s < best_s:
            best, best_s = j, s
    return best


def warp(dev, n_target):
    """onset-anchored: first 250 ms natural speed; middle tiled/stretched;
    last 100 ms kept as the release."""
    n = len(dev)
    if n == n_target:
        return dev.copy()
    on = min(int(0.25 * FPS), n // 3, n_target)
    rel = min(int(0.10 * FPS), n // 4, max(0, n_target - on))
    mid = dev[on : n - rel]
    n_mid_t = n_target - on - rel
    if n_mid_t <= 0:
        return np.concatenate([dev[:on], dev[n - rel:]])[:n_target]
    if len(mid) < 2:
        mid_t = np.full(n_mid_t, dev[on - 1] if on else 0.0)
    elif n_mid_t <= 1.5 * len(mid):
        mid_t = np.interp(np.linspace(0, len(mid) - 1, n_mid_t),
                          np.arange(len(mid)), mid)
    else:  # tile with 100 ms crossfades: vibrato keeps its natural rate
        xfn = min(int(0.10 * FPS), len(mid) // 3)
        out = list(mid)
        while len(out) < n_mid_t + xfn:
            f = np.linspace(0, 1, xfn)
            for i in range(xfn):
                out[-xfn + i] = out[-xfn + i] * (1 - f[i]) + mid[i] * f[i]
            out.extend(mid[xfn:])
        mid_t = np.array(out[:n_mid_t])
    return np.concatenate([dev[:on], mid_t, dev[n - rel:]])


def main():
    take_path, notes_path = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.load(open(notes_path))
    notes, step_samps, sr = meta["notes"], meta["step_samps"], meta["sr"]

    units = mine_corpus()
    durs = [u["dur"] for u in units]
    print(f"corpus: {len(units)} gesture units, dur p50 {np.median(durs):.2f}s "
          f"max {max(durs):.2f}s, with-interval {sum(u['itv'] is not None for u in units)}")

    with wave.open(str(take_path)) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    # target segments from the tick timeline
    segs = []  # (note, tick_lo, n_ticks)
    for k, n in enumerate(notes):
        if n is None:
            continue
        if segs and segs[-1][0] == n and segs[-1][1] + segs[-1][2] == k:
            segs[-1][2] += 1
        else:
            segs.append([n, k, 1])

    tick_f = step_samps / sr * FPS  # frames per tick (37.5)
    n_frames = int(len(mic) / (sr * FRAME_MS / 1000.0)) + 1
    f0_new = np.zeros(n_frames)
    rng = np.random.default_rng(SEED)
    recent = []
    prev_note = None
    prev_end_tick = -99
    for note, lo, nt in segs:
        gap_ticks = lo - prev_end_tick
        itv_t = None if prev_note is None or gap_ticks > 2 else float(note - prev_note)
        dur_t = nt * step_samps / sr
        j = pick(units, dur_t, itv_t, rng, recent)
        recent = (recent + [j])[-8:]
        f_lo, f_hi = int(lo * tick_f), min(int((lo + nt) * tick_f), n_frames)
        dev = warp(np.asarray(units[j]["dev"], dtype=np.float64), f_hi - f_lo)
        hz = 440 * 2 ** ((note - 69) / 12)
        f0_new[f_lo:f_hi] = hz * 2 ** (dev[: f_hi - f_lo] / 1200.0)
        prev_note, prev_end_tick = note, lo + nt

    print("WORLD analysis...")
    _, _, sp2, ap2, _ = analyse(mic, sr)
    print("synthesize...")
    angel = pyworld.synthesize(np.ascontiguousarray(f0_new[: len(sp2)]),
                               sp2, ap2, sr, frame_period=FRAME_MS)
    angel = angel[: len(mic)] if len(angel) >= len(mic) else np.pad(
        angel, (0, len(mic) - len(angel)))
    a_up, a_dn = 1 - np.exp(-FRAME_MS / 40.0), 1 - np.exp(-FRAME_MS / 150.0)
    tgt = (f0_new[: len(sp2)] > 0).astype(float)
    g, env = 0.0, np.empty(len(tgt))
    for k in range(len(tgt)):
        g += (tgt[k] - g) * (a_up if tgt[k] > g else a_dn)
        env[k] = g
    hop = int(sr * FRAME_MS / 1000)
    gate = np.repeat(env, hop)[: len(angel)]
    gate = np.pad(gate, (0, len(angel) - len(gate)),
                  constant_values=gate[-1] if len(gate) else 0)
    angel *= gate
    for suffix, sig in [("angel", angel), ("mix", 0.5 * mic + 0.8 * angel)]:
        peak = max(1e-9, np.abs(sig).max())
        with wave.open(str(HERE / "out" / f"gest_r1_{suffix}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
    print("wrote out/gest_r1_angel.wav, out/gest_r1_mix.wav")


if __name__ == "__main__":
    main()
