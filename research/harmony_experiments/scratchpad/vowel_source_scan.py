"""vowel_source_scan - choose material by VOWEL rather than by brightness

The post-mortem (layer_vowel_audit) found that the five current layers were chosen
by sorting on spectral tilt, and the vowels they actually sing measure as
[u/o, o, i, o, a]: **no e at all, and o holding three of the five places**. The
labels were purely an assumption about order. Every one of the three listening
verdicts lines up with that: the pair that "worked" really is i and a; the vowel that
"could not be told apart" has no layer at all; the pair that "never came through" is
actually the same vowel twice; and the layer that "sounded like a different vowel"
had its F2 pushed from 785 to 1322 by rendering.

The key point: **rendering changes the vowel** (one source layer at F2 829 came out
at 1877 after rendering, which is a different vowel). So material cannot be chosen at
the source alone; it has to be **checked again after rendering**. This file does only
the first step, source-side candidates; the render check is in vowel_layer_rebuild.py.

What it does: scan the corpus, measure F1 and F2 over every 300 ms window (his own
voice, low register, where LPC is reliable), then pick the most similar, steadiest
and longest candidates for each of the five target vowels and save them as JSON for
rendering.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_source_scan.py
"""
import glob
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ns = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _ns)
lpc_formants, REF = _ns["lpc_formants"], _ns["REF"]
C = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
     "260618/test/man/0619_clean_clips")
WIN, HOP = 0.30, 0.15                    # seconds


def dist(f1, f2, v):
    r = REF[v]
    return float(np.hypot(np.log(f1 / r[0]), np.log(f2 / r[1])))


def main():
    files = sorted(glob.glob(f"{C}/*.wav"))
    print(f"scanning {len(files)} clips...", flush=True)
    rows = []
    for fp in files:
        x, fs = sf.read(fp, always_2d=True)
        x = x[:, 0].astype("float64")
        if len(x) < int(WIN * fs):
            continue
        x16 = resample_poly(x, 16000, fs)
        w, h = int(WIN * 16000), int(HOP * 16000)
        for i in range(0, len(x16) - w, h):
            seg = x16[i:i + w]
            rms = float(np.sqrt((seg ** 2).mean()))
            if rms < 0.02:                     # skip silent and breathy windows
                continue
            # steadiness: the formants of the two halves must agree, so transitions are excluded
            fa = lpc_formants(seg[:w // 2])
            fb = lpc_formants(seg[w // 2:])
            if len(fa) < 2 or len(fb) < 2:
                continue
            if (abs(fa[0] - fb[0]) > 60 or abs(fa[1] - fb[1]) > 180):
                continue
            F1, F2 = (fa[0] + fb[0]) / 2, (fa[1] + fb[1]) / 2
            best = min(REF, key=lambda v: dist(F1, F2, v))
            rows.append({"clip": fp, "t0": i / 16000.0,
                         "t1": (i + w) / 16000.0, "F1": F1, "F2": F2,
                         "vowel": best, "d": dist(F1, F2, best),
                         "rms": rms})
    print(f"{len(rows)} valid windows", flush=True)
    out = {}
    for v in REF:
        # take 30, not 8: the criterion for material is how it sounds AFTER rendering,
        # and the source is only a first sieve, so the candidate pool has to be large
        # enough for vowel_render_select to draw from (the lesson: a source-side
        # d=0.11 still came out wrong, while an ordinary corpus window came out clean,
        # which makes this something of a lottery)
        cs = sorted([r for r in rows if r["vowel"] == v],
                    key=lambda r: r["d"])[:30]
        out[v] = cs
        print(f"\n{v} (target F1 {REF[v][0]} F2 {REF[v][1]}), {len(cs)} candidates:")
        for r in cs[:4]:
            print(f"  {r['clip'].split('/')[-1]} {r['t0']:.1f}-{r['t1']:.1f}s"
                  f"  F1 {r['F1']:.0f} F2 {r['F2']:.0f}  d {r['d']:.2f}")
    json.dump(out, open("scratchpad/vowel_source_cands.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n→ scratchpad/vowel_source_cands.json")


if __name__ == "__main__":
    main()
