"""POP909 feasibility probe for the second-chorus-partner corpus (Task 1).

Questions:
 1. Is BRIDGE a usable second voice? (coverage: % of melody time with a
    concurrent bridge note; register relative to melody)
 2. Does the repetition structure survive tokenization? (repetition score:
    fraction of 16-step melody n-grams on the 16th-note beat grid that
    recur later in the song -- the induction-learning signal)
 3. Ranges vs the brain's 36..84 world.
"""

import sys
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("POP909-Dataset-master/POP909")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100

REST = -1


def grid_tokens(notes, grid):
    """Sounding pitch at each grid time (16th notes), REST when silent."""
    toks = np.full(len(grid), REST, dtype=int)
    for n in notes:
        i0, i1 = np.searchsorted(grid, [n.start, n.end])
        toks[i0:i1] = n.pitch
    return toks


def rep_score(toks, w=16):
    """Fraction of length-w windows (with sound) that recur exactly later."""
    seen, rec, tot = {}, 0, 0
    for i in range(0, len(toks) - w, w // 2):
        key = tuple(toks[i : i + w])
        if all(t == REST for t in key):
            continue
        tot += 1
        if key in seen:
            rec += 1
        seen[key] = True
    return rec / tot if tot else 0.0


stats = []
for d in sorted(ROOT.iterdir())[:N]:
    mid = d / f"{d.name}.mid"
    beats_f = d / "beat_midi.txt"
    if not (mid.exists() and beats_f.exists()):
        continue
    try:
        pm = pretty_midi.PrettyMIDI(str(mid))
    except Exception:
        continue
    tracks = {t.name.upper(): t for t in pm.instruments}
    if "MELODY" not in tracks or "BRIDGE" not in tracks:
        continue
    mel, bri = tracks["MELODY"].notes, tracks["BRIDGE"].notes
    if not mel:
        continue
    beats = np.array([float(l.split()[0]) for l in beats_f.read_text().splitlines() if l.strip()])
    if len(beats) < 8:
        continue
    # 16th grid: 4 subdivisions per annotated beat
    grid = np.concatenate([np.linspace(beats[i], beats[i + 1], 4, endpoint=False)
                           for i in range(len(beats) - 1)])
    mt, bt = grid_tokens(mel, grid), grid_tokens(bri, grid)
    voiced = mt != REST
    both = voiced & (bt != REST)
    below = np.mean(bt[both] < mt[both]) if both.any() else np.nan
    stats.append(dict(
        song=d.name,
        mel_notes=len(mel), bri_notes=len(bri),
        mel_lo=min(n.pitch for n in mel), mel_hi=max(n.pitch for n in mel),
        bri_lo=min((n.pitch for n in bri), default=0), bri_hi=max((n.pitch for n in bri), default=0),
        coverage=float(both.sum() / max(1, voiced.sum())),
        bri_below=float(below) if not np.isnan(below) else -1,
        rep=rep_score(mt),
        rep_pair=rep_score(np.stack([mt, bt]).T.reshape(-1), w=32),
        ticks=len(grid),
    ))

if not stats:
    print("NO USABLE SONGS")
    sys.exit(1)

arr = lambda k: np.array([s[k] for s in stats], dtype=float)
print(f"parsed {len(stats)}/{N} songs OK")
print(f"melody notes/song: median {np.median(arr('mel_notes')):.0f}  "
      f"bridge notes/song: median {np.median(arr('bri_notes')):.0f}")
print(f"melody range: p5 {np.percentile(arr('mel_lo'),5):.0f} .. p95 {np.percentile(arr('mel_hi'),95):.0f} "
      f"(brain world = 36..84)")
print(f"bridge range: p5 {np.percentile(arr('bri_lo'),5):.0f} .. p95 {np.percentile(arr('bri_hi'),95):.0f}")
cov = arr("coverage")
print(f"BRIDGE coverage of melody time: median {np.median(cov):.0%}, "
      f">=50% in {np.mean(cov>=0.5):.0%} of songs, >=80% in {np.mean(cov>=0.8):.0%}")
bb = arr("bri_below"); bb = bb[bb >= 0]
print(f"bridge below melody (when both sound): median {np.median(bb):.0%}")
rep = arr("rep")
print(f"melody repetition score (16-step 16th-note windows recurring): "
      f"median {np.median(rep):.0%}, p25 {np.percentile(rep,25):.0%}, p75 {np.percentile(rep,75):.0%}")
print(f"ticks/song median {np.median(arr('ticks')):.0f} 16ths "
      f"-> interleaved tokens ~{2*np.median(arr('ticks')):.0f} (CTX needed)")
lo = np.mean(cov < 0.3)
print(f"\nverdict inputs: songs with BRIDGE coverage <30% = {lo:.0%} "
      f"(these are 'fills-only' bridges -- unusable as a duet partner line)")
