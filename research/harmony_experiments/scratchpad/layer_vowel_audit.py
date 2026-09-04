"""layer_vowel_audit - which vowel does each of the five layers actually render?

The verdict that one layer sounded like a different vowel hit an assumption nobody
had checked since v16.3: the layer-to-vowel mapping **was only ever applied in order
of brightness** (the corpus was sorted by spectral tilt and the labels assumed from
there). What vowel the material actually sings was never examined. The mouth shape
is recognised correctly and the right bank is chosen, but **the bank itself is
mislabelled**, so mouth and sound do not agree.

Two things here:
 1. Objective: LPC formants of the loop each layer renders (downsampled to 16k,
    order 18), giving F1 and F2, compared against the standard vowel space as a
    hypothesis.
 2. Subjective, and authoritative: produce a BLIND LISTENING file, layer_audit.wav,
    announcing only the layer number and never the vowel name (against mishearing
    and against suggestion), two seconds per layer, for the ear to label.
    The mapping is then rebuilt from those answers.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/layer_vowel_audit.py [--voice alto] [--midi 62]
"""
import argparse
import glob
import os
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

SR = 44100
# Standard vowel F1/F2 for a male voice, in Hz. A reference only, never the verdict.
# The keys are the Chinese vowel labels that key bank_live.VOWEL_LAYERS.
REF = {"啊": (730, 1090), "喔": (500, 700), "嗚": (320, 800),
       "欸": (530, 1840), "咿": (270, 2290)}


def lpc_formants(x, order=18, fs=16000):
    x = x - x.mean()
    x = lfilter([1, -0.97], [1], x)                 # pre-emphasis
    w = x * np.hanning(len(x))
    r = np.correlate(w, w, "full")[len(w) - 1:][:order + 1]
    if r[0] <= 0:
        return []
    a, e = np.zeros(order + 1), r[0]                # Levinson-Durbin
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / e
        an = a.copy()
        for j in range(1, i):
            an[j] = a[j] + k * a[i - j]
        an[i] = k
        a = an
        e *= (1 - k * k)
        if e <= 0:
            break
    rts = [z for z in np.roots(a) if np.imag(z) > 0.01]
    f = sorted(float(np.arctan2(np.imag(z), np.real(z)) * fs / (2 * np.pi))
               for z in rts)
    bw = {}
    for z in rts:
        fz = float(np.arctan2(np.imag(z), np.real(z)) * fs / (2 * np.pi))
        bw[round(fz)] = -0.5 * (fs / (2 * np.pi)) * np.log(abs(z))
    return [x_ for x_ in f if 200 < x_ < 4000 and bw.get(round(x_), 1e9) < 500]


def nearest_vowel(f1, f2):
    d = {k: ((np.log(f1 / v[0])) ** 2 + (np.log(f2 / v[1])) ** 2) ** 0.5
         for k, v in REF.items()}
    o = sorted(d, key=d.get)
    return o[0], o[1], d[o[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="alto")
    ap.add_argument("--midi", type=int, default=62)
    ap.add_argument("--out", default="scratchpad/layer_audit.wav")
    ap.add_argument("--dir", default="", help="layer bank directory (empty = the newest)")
    a = ap.parse_args()

    cands = glob.glob(f"scratchpad/bank_*/{a.voice}_L0.npz")
    assert cands, "no layer bank found (run once with --vowels 1 first)"
    # Pick the newest by mtime: the directory names are hashes, so lexical order has
    # nothing to do with age. The blind listening file was once nearly built from an
    # old bank. Use --dir to be explicit.
    bd = a.dir or os.path.dirname(max(cands, key=os.path.getmtime))
    print(f"layer bank directory {bd}\n")

    print(f"{'layer':<7}{'F1':>6}{'F2':>7}   {'nearest vowel':<16}{'second':<10}")
    segs = []
    for li in range(5):
        z = np.load(f"{bd}/{a.voice}_L{li}.npz")
        key = str(a.midi) if str(a.midi) in z.files else z.files[
            len(z.files) // 2]
        loop = z[key].astype("float64")
        x16 = resample_poly(loop, 16000, SR)
        n = min(len(x16), 1600)
        fs_ = [lpc_formants(x16[i:i + n]) for i in range(0, len(x16) - n, n)]
        fs_ = [f for f in fs_ if len(f) >= 2]
        F1 = float(np.median([f[0] for f in fs_])) if fs_ else float("nan")
        F2 = float(np.median([f[1] for f in fs_])) if fs_ else float("nan")
        v1, v2, _d = nearest_vowel(F1, F2)
        print(f"L{li:<3}{F1:>6.0f}{F2:>7.0f}   {v1:<10}{v2:<8}")

        # the blind file announces the layer number only, never the vowel name, so nothing is suggested
        aif = f"/tmp/_lay{li}.aiff"
        subprocess.run(["say", "-o", aif, f"第{li + 1}個"],   # spoken: "number N"
                       check=True)
        sp, sr_ = sf.read(aif, dtype="float32", always_2d=True)
        sp = resample_poly(sp[:, 0], SR, sr_).astype("float32")
        body = np.tile(loop, int(2.0 * SR / len(loop)) + 1)[:int(2.0 * SR)]
        body = (body / (np.abs(body).max() + 1e-9) * 0.5).astype("float32")
        segs += [sp * 0.6, np.zeros(int(0.25 * SR), "float32"), body,
                 np.zeros(int(0.6 * SR), "float32")]
        os.remove(aif)
    sf.write(a.out, np.concatenate(segs), SR)
    print(f"\nblind listening file -> {a.out} (each entry: the layer number, then 2 seconds of that layer)")


if __name__ == "__main__":
    main()
