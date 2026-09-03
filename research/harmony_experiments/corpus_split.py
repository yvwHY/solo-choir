"""Corpus assembly for the second-chorus-partner retrain. Proposal Task 3.

Doesn't copy data: reads the three token blocks (pop909_v2 / chorale
tokens.npz / cpdl_v2 when present), applies the quality filter, and writes
corpus_split.json -- whole-song train/val lists the trainer consumes.

Quality filter (silver block only): oblique-motion %, the metric that
reproduced Harry's independence ranking (worklog 07-23 §T cont. 4). Real
pairs (chorale, CPDL) are not filtered. Whole-song split prevents chorus
repeats leaking into val (the induction probe would be meaningless).

Run:  python corpus_split.py <tokens_v2_dir> <chorale_tokens.npz> [--min-oblique 0.35]
"""

import json
import sys
from pathlib import Path

import numpy as np

REST, HOLD, PITCH_OFFSET = 0, 1, 2
VAL_FRAC, SEED = 0.05, 7


def decode_col(toks):
    out = np.full(len(toks), -1, dtype=int)
    prev = None
    for i, t in enumerate(toks):
        if t == REST:
            prev = None
        elif t == HOLD:
            out[i] = -1 if prev is None else prev
        else:
            prev = t
            out[i] = t
    return out


def oblique_pct(tok):
    m, s = decode_col(tok[:, 0]), decode_col(tok[:, 1])
    idx = np.where((m >= 0) & (s >= 0))[0]
    if len(idx) < 8:
        return 0.0
    dm, ds = np.diff(m[idx].astype(float)), np.diff(s[idx].astype(float))
    mv = (dm != 0) | (ds != 0)
    if not mv.any():
        return 0.0
    dm, ds = dm[mv], ds[mv]
    return float(np.mean((dm != 0) & (ds == 0)) + np.mean((dm == 0) & (ds != 0)))


def main():
    tv2 = Path(sys.argv[1])
    chorale_npz = Path(sys.argv[2])
    min_obl = (float(sys.argv[sys.argv.index("--min-oblique") + 1])
               if "--min-oblique" in sys.argv else 0.35)

    songs = []  # (source, id, ticks)
    dropped = []

    pop = np.load(tv2 / "pop909_v2.npz")
    obls = {}
    for k in pop.files:
        if not k.startswith("tok_"):
            continue
        sid = k[4:]
        o = oblique_pct(pop[k])
        obls[sid] = o
        (songs if o >= min_obl else dropped).append(("pop909", sid, int(len(pop[k]))))
    ov = np.array(list(obls.values()))
    print(f"pop909: {len(obls)} songs, oblique median {np.median(ov):.0%} "
          f"(p25 {np.percentile(ov,25):.0%} / p75 {np.percentile(ov,75):.0%}); "
          f"filter >={min_obl:.0%} keeps {sum(1 for s in songs if s[0]=='pop909')}, "
          f"drops {len(dropped)}  <- NOT silent: dropped list in the json")

    cho = np.load(chorale_npz, allow_pickle=True)
    for k in cho.files:
        songs.append(("chorale", k, int(len(cho[k]))))
    print(f"chorale: {len(cho.files)} works (unfiltered, gold pairs)")

    cpdl_p = tv2 / "cpdl_v2.npz"
    if cpdl_p.exists():
        cpdl = np.load(cpdl_p)
        n = 0
        for k in cpdl.files:
            if k.startswith("tok_"):
                songs.append(("cpdl", k[4:], int(len(cpdl[k]))))
                n += 1
        print(f"cpdl: {n} works (unfiltered, real pairs)")
    else:
        print("cpdl: npz not present yet -- re-run after cpdl_pairs.py finishes")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(songs))
    n_val = max(1, int(len(songs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())
    split = {
        "train": [songs[i][:2] for i in range(len(songs)) if i not in val_idx],
        "val": [songs[i][:2] for i in val_idx],
        "dropped_low_oblique": [d[:2] for d in dropped],
        "min_oblique": min_obl, "seed": SEED,
    }
    out = tv2 / "corpus_split.json"
    out.write_text(json.dumps(split))
    tot = sum(s[2] for s in songs)
    print(f"-> {out}: train {len(split['train'])} / val {len(split['val'])} songs, "
          f"{tot} ticks ~ {2*tot} interleaved tokens total")


if __name__ == "__main__":
    main()
