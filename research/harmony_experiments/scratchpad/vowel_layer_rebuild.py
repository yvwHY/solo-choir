"""vowel_layer_rebuild - re-pick the five layers of material by vowel (groundwork for v21)

vowel_source_scan confirmed that all five vowels are present in the corpus (one at
F1 519 / F2 1824, another at 319/796); it was picking by brightness that ended up
with three copies of the same vowel.

This picks a CONTINUOUS STEADY STRETCH rather than a single window. The render
encoder takes context of plus or minus 40 hops (about 0.93 s) around the centre of
the window, so a 300 ms window swallows the neighbouring phoneme with it, which was
the hidden trap in the old method. So: adjacent windows of the same vowel are merged
into a run, the run of at least 0.7 s with the smallest mean distance is taken, and
its centre plus or minus 0.5 s becomes the layer window.

Output: a suggested VOWEL_LAYERS block to paste into bank_live.py, plus JSON.
After rendering, layer_vowel_audit MUST be run to check it - **rendering changes the
vowel**, and being right at the source does not mean being right after rendering
(one layer went from F2 829 to 1877, which is a different vowel; that is exactly how
this broke).
"""
import glob
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, REF = _a["lpc_formants"], _a["REF"]
_s = {}
exec(compile(open("scratchpad/vowel_source_scan.py").read()
             .split("\ndef main()")[0], "vss", "exec"), _s)
dist, C, WIN, HOP = _s["dist"], _s["C"], _s["WIN"], _s["HOP"]
ORDER = ["喔", "欸", "咿", "嗚", "啊"]      # must be in the same order as the calibration labels
MINRUN = 0.7


def windows(fp):
    x, fs = sf.read(fp, always_2d=True)
    x = x[:, 0].astype("float64")
    if len(x) < int(WIN * fs):
        return []
    x16 = resample_poly(x, 16000, fs)
    w, h = int(WIN * 16000), int(HOP * 16000)
    out = []
    for i in range(0, len(x16) - w, h):
        seg = x16[i:i + w]
        if float(np.sqrt((seg ** 2).mean())) < 0.02:
            continue
        fa, fb = lpc_formants(seg[:w // 2]), lpc_formants(seg[w // 2:])
        if len(fa) < 2 or len(fb) < 2:
            continue
        if abs(fa[0] - fb[0]) > 60 or abs(fa[1] - fb[1]) > 180:
            continue
        F1, F2 = (fa[0] + fb[0]) / 2, (fa[1] + fb[1]) / 2
        v = min(REF, key=lambda k: dist(F1, F2, k))
        out.append((i / 16000.0, v, dist(F1, F2, v), F1, F2))
    return out


def runs_of(fp):
    ws = windows(fp)
    out, cur = [], None
    for t, v, d, F1, F2 in ws:
        if cur and cur["v"] == v and t - cur["last"] <= HOP + 1e-6:
            cur["last"] = t
            cur["ds"].append(d)
            cur["f"].append((F1, F2))
        else:
            if cur:
                out.append(cur)
            cur = {"clip": fp, "v": v, "t0": t, "last": t, "ds": [d],
                   "f": [(F1, F2)]}
    if cur:
        out.append(cur)
    for r in out:
        r["t1"] = r["last"] + WIN
        r["len"] = r["t1"] - r["t0"]
        r["d"] = float(np.mean(r["ds"]))
        r["F1"] = float(np.median([x[0] for x in r["f"]]))
        r["F2"] = float(np.median([x[1] for x in r["f"]]))
        del r["f"], r["ds"], r["last"]
    return out


def main():
    files = sorted(glob.glob(f"{C}/*.wav"))
    print(f"scanning {len(files)} clips, merging continuous stretches...", flush=True)
    R = []
    for fp in files:
        R += runs_of(fp)
    pick = {}
    for v in ORDER:
        cs = [r for r in R if r["v"] == v and r["len"] >= MINRUN]
        if not cs:
            cs = [r for r in R if r["v"] == v]
            print(f"{v}: no steady stretch of at least {MINRUN}s; falling back to the longest")
        cs.sort(key=lambda r: (r["d"], -r["len"]))
        pick[v] = cs[0]
    print(f"\n{'vowel':<7}{'clip':<16}{'window':<14}{'len':>5}{'F1':>6}{'F2':>7}"
          f"{'d':>6}")
    lines = []
    for v in ORDER:
        r = pick[v]
        mid = (r["t0"] + r["t1"]) / 2
        t0, t1 = max(0.0, mid - 0.5), mid + 0.5
        nm = r["clip"].split("/")[-1]
        print(f"{v:<4}{nm:<16}{r['t0']:.1f}-{r['t1']:.1f}s{'':<3}"
              f"{r['len']:>5.1f}{r['F1']:>6.0f}{r['F2']:>7.0f}{r['d']:>6.2f}")
        lines.append(f'                ("{v}", f"{{_C0619}}/{nm}", '
                     f'{t0:.2f}, {t1:.2f}, 0.0),')
    print("\nVOWEL_LAYERS to paste into bank_live.py:")
    print("VOWEL_LAYERS = [" + "\n".join(lines)[16:] + "]")
    json.dump({v: pick[v] for v in ORDER},
              open("scratchpad/vowel_layer_pick.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
