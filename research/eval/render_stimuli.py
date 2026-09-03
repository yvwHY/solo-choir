"""Render matched A/B stimuli for the baseline listening test (spec 2026-07-07 §4).
Per take: transcribe -> render Beatrice + naive from the SAME score -> RMS-normalize both mixes
to a common target (pair-shared peak guard keeps their loudness EQUAL) -> write blinded pairs +
manifest.js for the test page + KEY.csv (the unblinding map — never serve it).

Usage: python eval/render_stimuli.py OUT_DIR TAKE1.wav [TAKE2.wav ...] [--seed 7]
"""
import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "server"))
import studio_api  # noqa: E402
from beatrice_solo_choir_live import _SATB2_MODEL  # noqa: E402

TARGET_RMS = 0.05          # common loudness target (linear)


def _load_norm(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y[:, 0]
    r = float(np.sqrt(np.mean(y * y)))
    if r > 1e-6:
        y = y * (TARGET_RMS / r)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--seed", type=int, default=7, help="blinding shuffle seed (recorded via KEY.csv anyway)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    key_rows, manifest = [], []
    for k, take in enumerate(a.takes, 1):
        score = studio_api.transcribe(take)
        json.dump(score, open(os.path.join(a.out_dir, f"phrase{k}_score.json"), "w"))
        tmp = os.path.join(a.out_dir, f"_render_{k}")
        mixes = {"beatrice": studio_api.render_score(score, take, tmp, model=_SATB2_MODEL)["mix"],
                 "naive":    studio_api.render_score(score, take, tmp, model=_SATB2_MODEL,
                                                     method="naive")["mix"]}
        ys = {c: _load_norm(p) for c, p in mixes.items()}
        peak = max(float(np.max(np.abs(y))) for y in ys.values())
        scale = min(1.0, 0.99 / peak) if peak > 0 else 1.0    # shared guard → pair loudness stays equal
        labels = ["a", "b"]
        rng.shuffle(labels)                                   # per-phrase blinded position
        for (cond, y), lab in zip(ys.items(), labels):
            fn = f"phrase{k}_{lab}.wav"
            sf.write(os.path.join(a.out_dir, fn), y * scale, 48000)
            key_rows.append([f"phrase{k}", lab, cond])
        manifest.append({"phrase": f"phrase{k}",
                         "a": f"phrase{k}_a.wav", "b": f"phrase{k}_b.wav"})
    with open(os.path.join(a.out_dir, "KEY.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["phrase", "label", "condition"]] + key_rows)
    with open(os.path.join(a.out_dir, "manifest.js"), "w") as f:
        f.write("const MANIFEST = " + json.dumps(manifest) + ";\n")
    print(f"[stimuli] {len(a.takes)} phrase(s) -> {a.out_dir} (KEY.csv is the unblinding map — never serve it)")


if __name__ == "__main__":
    main()
