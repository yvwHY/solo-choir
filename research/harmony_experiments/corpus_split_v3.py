"""v3 corpus split: chorale_v3 + cpdl_v3 triples -> corpus_split_v3.json.

Whole-SONG split (5% val, seed 7, same policy as v2): all windows of a song
land on the same side -- adjacent windows share two voices, so a per-window
split would leak val voices into train. No quality filter: both sources are
real scores (v2's oblique filter only ever applied to machine-derived
silver lines, corpus_split.py docstring).

Output: <tokens_v3_dir>/corpus_split_v3.json
        {"train"/"val": [[source, window_key], ...], "seed": 7}
        window_key e.g. "ch012w0" / "100338w1" -> npz key tok_<window_key>.
Run:  python corpus_split_v3.py <tokens_v3_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

VAL_FRAC, SEED = 0.05, 7
SOURCES = {"chorale": "chorale_v3", "cpdl": "cpdl_v3"}


def main():
    tv3 = Path(sys.argv[1])
    songs = []  # (source, sid, [window_key...], ticks)
    for source, stem in SOURCES.items():
        meta = json.loads((tv3 / f"{stem}_meta.json").read_text())["songs"]
        npz = np.load(tv3 / f"{stem}.npz")
        keys = [k[4:] for k in npz.files if k.startswith("tok_")]
        for sid, m in meta.items():
            wins = sorted(k for k in keys if k.rsplit("w", 1)[0] == sid)
            assert len(wins) == m["windows"], (source, sid)
            songs.append((source, sid, wins, m["ticks"]))
        print(f"{source}: {len(meta)} songs, "
              f"{sum(m['windows'] for m in meta.values())} triples")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(songs))
    n_val = max(1, int(len(songs) * VAL_FRAC))
    val_idx = set(order[:n_val].tolist())
    split = {
        "train": [[s, w] for i, (s, _, ws, _) in enumerate(songs)
                  if i not in val_idx for w in ws],
        "val": [[s, w] for i, (s, _, ws, _) in enumerate(songs)
                if i in val_idx for w in ws],
        "seed": SEED, "val_frac": VAL_FRAC, "split_unit": "song",
    }
    out = tv3 / "corpus_split_v3.json"
    out.write_text(json.dumps(split))
    tot = sum(t for _, _, _, t in songs)
    print(f"-> {out}: {len(songs)} songs ({len(songs)-n_val} train / {n_val} val) "
          f"= {len(split['train'])} / {len(split['val'])} triples, "
          f"{tot} ticks ~ {3*tot} interleaved tokens total")


if __name__ == "__main__":
    main()
