"""Predicted naturalness for the baseline comparison (spec 2026-07-07 §6.2): DNSMOS (Microsoft
speechmos package, ONNX, fully local) on each blinded mix, unblinded via KEY.csv. ALWAYS report
these as PREDICTED MOS (a speech-trained neural estimator applied to singing — a proxy, stated
as such in the thesis), never as measured human MOS.

Usage: python eval/predicted_mos.py STIMULI_DIR    (a folder made by render_stimuli.py)
First run downloads the ONNX models; needs `pip install --no-deps speechmos` (onnxruntime/librosa
already in vcclient-dev — never let pip touch this env's torch).
"""
import csv
import glob
import os
import sys

import numpy as np
import soundfile as sf
import soxr


def main():
    try:
        from speechmos import dnsmos
    except ImportError:
        sys.exit("pip install --no-deps speechmos  (Microsoft DNSMOS; deps already in vcclient-dev)")
    d = sys.argv[1]
    key = {(r["phrase"], r["label"]): r["condition"]
           for r in csv.DictReader(open(os.path.join(d, "KEY.csv")))}
    per_cond = {}
    for p in sorted(glob.glob(os.path.join(d, "phrase*_[ab].wav"))):
        name = os.path.basename(p)[:-4]
        phrase, lab = name.rsplit("_", 1)
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y[:, 0]
        if sr != 16000:
            y = np.asarray(soxr.resample(y, sr, 16000), dtype=np.float32)
        r = dnsmos.run(y, sr=16000)
        cond = key[(phrase, lab)]
        per_cond.setdefault(cond, []).append(r["p808_mos"])
        print(f"{name}  {cond:9s}  predicted p808_mos {r['p808_mos']:.2f}  ovrl {r['ovrl_mos']:.2f}  sig {r['sig_mos']:.2f}")
    for cond, ms in sorted(per_cond.items()):
        print(f"MEAN {cond:9s} p808_mos {np.mean(ms):.2f}  (n={len(ms)})")


if __name__ == "__main__":
    main()
