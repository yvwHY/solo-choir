"""Objective intonation: per harmony stem, SwiftF0-track the rendered audio and measure deviation
in cents from the score's target notes (spec 2026-07-07 §6.2). Reports per-condition medians —
the intonation axis's ground truth, independent of any human listener.

Usage: python eval/intonation_metric.py STIMULI_DIR   (a folder made by render_stimuli.py)
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import soxr
import swift_f0

VOICES = ["sop", "alto", "tenor", "bass"]
EDGE_S = 0.08          # skip note onsets/offsets (entry scoops/falls, same margin as _melody_tracks)


def stem_cents(wav, notes):
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    y = y[:, 0]
    if sr != 16000:
        y = np.asarray(soxr.resample(y, sr, 16000), dtype=np.float32)
    r = swift_f0.SwiftF0().detect_from_array(y, 16000)
    t = np.asarray(r.timestamps)
    hz = np.asarray(r.pitch_hz)
    ok = (np.asarray(r.confidence) > 0.5) & (hz > 0)
    midi = 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0)
    cents = []
    for (s, e, tgt) in notes:
        m = ok & (t >= s + EDGE_S) & (t < e - EDGE_S)
        d = 100.0 * (midi[m] - tgt)
        cents.extend(d[np.abs(d) <= 600].tolist())   # >6 st = octave/tracker error, not intonation
    return np.asarray(cents)


def main():
    d = sys.argv[1]
    per_cond = {"beatrice": [], "naive": []}
    for sj in sorted(glob.glob(os.path.join(d, "phrase*_score.json"))):
        k = os.path.basename(sj).split("_")[0]
        score = json.load(open(sj))
        rd = os.path.join(d, "_render_" + k.replace("phrase", ""))
        for cond, base in (("beatrice", "studio"), ("naive", "naive")):
            for v in VOICES:
                p = os.path.join(rd, f"{base}_{v}.wav")
                if not os.path.exists(p):
                    continue
                notes = [(n["start"], n["start"] + n["dur"], n["midi"])
                         for n in score["voices"][v.capitalize()]]
                c = stem_cents(p, notes)
                if len(c):
                    per_cond[cond].append(c)
                print(f"{k} {cond:9s} {v:6s} median|dev| {np.median(np.abs(c)) if len(c) else float('nan'):6.1f} cents  n={len(c)}")
    for cond, cs in per_cond.items():
        allc = np.concatenate(cs) if cs else np.zeros(0)
        if len(allc):
            q25, q75 = np.percentile(np.abs(allc), [25, 75])
            print(f"TOTAL {cond:9s} median|dev| {np.median(np.abs(allc)):6.1f} cents  IQR {q75 - q25:6.1f}  n={len(allc)}")
        else:
            print(f"TOTAL {cond:9s} no voiced frames matched")


if __name__ == "__main__":
    main()
