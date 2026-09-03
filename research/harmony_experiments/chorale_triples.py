"""JSB chorales (music21 corpus) -> (upper, lead, lower) token triples, brain v3.

Same machinery as cpdl_triples.py (shared extract_score): voices ranked by
mean pitch, adjacent windows of 3, middle voice = lead. A 4-voice chorale
yields (S,A,T) lead=A and (A,T,B) lead=T. Same filters + C/Am transposition.

Output: <out>/chorale_v3.npz {tok_ch<i>w<j>: (T,3) int64 [upper,lead,lower],
        phase_ch<i>w<j>: (T,)} + chorale_v3_meta.json (title per chorale).
Run (RETRAINING venv): venv/bin/python chorale_triples.py <out_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from music21 import corpus

from cpdl_triples import extract_score


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays, meta, skips = {}, {}, Counter()
    for i, chorale in enumerate(corpus.chorales.Iterator()):
        sid = f"ch{i:03d}"
        try:
            r, why = extract_score(chorale)
        except Exception:
            skips["parse_fail"] += 1
            continue
        if r is None:
            skips[why] += 1
            continue
        wins, shift = r
        for j, tok, phase in wins:
            arrays[f"tok_{sid}w{j}"], arrays[f"phase_{sid}w{j}"] = tok, phase
        meta[sid] = {"title": str(chorale.metadata.title), "shift": shift,
                     "windows": len(wins),
                     "ticks": int(sum(len(t) for _, t, _ in wins))}
        if (i + 1) % 50 == 0:
            print(f"[{i+1}] chorales {len(meta)} triples "
                  f"{sum(m['windows'] for m in meta.values())}", flush=True)
    np.savez_compressed(out_dir / "chorale_v3.npz", **arrays)
    (out_dir / "chorale_v3_meta.json").write_text(json.dumps({"songs": meta}, indent=0))
    triples = sum(m["windows"] for m in meta.values())
    ticks = sum(m["ticks"] for m in meta.values())
    print(f"{len(meta)} chorales -> {triples} triples ({ticks} ticks ~ "
          f"{3*ticks} interleaved tokens); skips: {dict(skips)}", flush=True)


if __name__ == "__main__":
    main()
