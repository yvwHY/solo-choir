"""CPDL corpus -> (upper, lead, lower) token triples for brain v3.

Spec: docs/specs/2026-07-27-brain-v3-two-voice.md (wrap voicing:
the user is the MIDDLE voice, angels above and below). Reuses the v2 pairing
machinery from cpdl_pairs.py unchanged: voices ranked by mean pitch, same
filters (<20 notes, near-dup parts, polyphony, off-grid, C/Am transposition).
Every window of 3 ADJACENT ranked voices becomes one triple -- an SATB piece
yields (S,A,T) lead=A and (A,T,B) lead=T, doubling usable data.

Output: <out>/cpdl_v3.npz  {tok_<sid>w<j>: (T,3) int64 cols [upper,lead,lower],
        phase_<sid>w<j>: (T,)}  + cpdl_v3_meta.json (per-song shift/windows).
Run (RETRAINING venv, music21):
  venv/bin/python cpdl_triples.py <corpus_dir> <out_dir> [--limit N]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from music21 import converter

from cpdl_pairs import (STEP, events_to_grid, grid_to_tokens, measure_phase,
                        part_events, transpose_iv)


def extract_score(score):
    """Score -> ([(win_idx, (T,3) tokens, phase)], shift) or (None, reason)."""
    cands = []
    for part in score.parts:
        evs = part_events(part)
        if evs and len(evs) >= 20:
            cands.append((np.mean([e[2] for e in evs]), len(evs), evs))
    dedup = []
    for mp, n, evs in sorted(cands, reverse=True):
        if any(abs(mp - mp2) < 0.5 and abs(n - n2) <= 2 for mp2, n2, _ in dedup):
            continue
        dedup.append((mp, n, evs))
    if len(dedup) < 3:
        return None, "lt3_parts"
    shift = transpose_iv(score)
    wins = []
    for j in range(len(dedup) - 2):  # adjacent triples, middle voice = lead
        trio = dedup[j:j + 3]
        n_steps = int(max(e[0] + e[1] for _, _, evs in trio for e in evs) / STEP) + 1
        grids, ogs = zip(*(events_to_grid(evs, n_steps, shift) for _, _, evs in trio))
        if max(ogs) > 0.2:
            continue
        tok = np.stack([grid_to_tokens(g) for g in grids], axis=1)
        wins.append((j, tok, measure_phase(score, n_steps)))
    if not wins:
        return None, "off_grid"
    return (wins, shift), None


def extract(path):
    return extract_score(converter.parse(str(path)))


def main():
    corpus, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    files = sorted(p for p in corpus.iterdir()
                   if p.suffix.lower() in (".mxl", ".mid", ".midi"))
    if limit:
        files = files[:limit]
    arrays, meta, skips = {}, {}, Counter()
    for i, f in enumerate(files):
        sid = f.stem.split("_")[0]
        if sid in meta:  # one score per CPDL page is enough
            skips["dup_page"] += 1
            continue
        try:
            r, why = extract(f)
        except Exception:
            skips["parse_fail"] += 1
            continue
        if r is None:
            skips[why] += 1
            continue
        wins, shift = r
        for j, tok, phase in wins:
            arrays[f"tok_{sid}w{j}"], arrays[f"phase_{sid}w{j}"] = tok, phase
        meta[sid] = {"file": f.name, "shift": shift, "windows": len(wins),
                     "ticks": int(sum(len(t) for _, t, _ in wins))}
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(files)}] songs {len(meta)} triples "
                  f"{sum(m['windows'] for m in meta.values())}", flush=True)
    np.savez_compressed(out_dir / "cpdl_v3.npz", **arrays)
    (out_dir / "cpdl_v3_meta.json").write_text(json.dumps({"songs": meta}, indent=0))
    triples = sum(m["windows"] for m in meta.values())
    ticks = sum(m["ticks"] for m in meta.values())
    print(f"{len(meta)} songs -> {triples} triples ({ticks} ticks ~ "
          f"{3*ticks} interleaved tokens) from {len(files)} files; "
          f"skips: {dict(skips)}", flush=True)


if __name__ == "__main__":
    main()
