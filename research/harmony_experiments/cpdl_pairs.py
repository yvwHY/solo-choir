"""CPDL corpus -> (soprano, alto) token pairs. Proposal Task 3, CPDL half.

Runs in the RETRAINING venv (260722_harmony_brain/venv: music21 10.x), not
vcclient-dev -- music21 parses both .mxl and .mid uniformly.

Pairing policy (from the 2026-07-23 agent probe): track names are unreliable
(3/9 semantic), so voices are ranked by mean pitch; require >=3 plausible
vocal parts (SATB confidence), take the top two as S and A. Filters: <20
notes, near-duplicate parts, heavy internal polyphony, >20% off-grid onsets
(triplet/compound pieces), out-of-range after C/Am transposition.

Output: <out>/cpdl_v2.npz  {tok_<id>: (T,2) int64, phase_<id>: (T,)}
        + cpdl_meta.json (title, key, shift, skip-reason stats).
Run:  venv/bin/python cpdl_pairs.py <corpus_dir> <out_dir> [--limit N]
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from music21 import converter, interval, pitch as m21pitch

REST_G = -1  # grid sentinel (matches silver_line.REST)
REST, HOLD, PITCH_OFFSET, PITCH_LO, PITCH_HI = 0, 1, 2, 36, 84
STEP = 0.25  # 16th note in quarterLength


def transpose_iv(score):
    k = score.analyze("key")
    tonic = "C" if k.mode == "major" else "A"
    iv = interval.Interval(k.tonic, m21pitch.Pitch(tonic))
    if iv.semitones > 6:
        iv = interval.Interval(iv.semitones - 12)
    elif iv.semitones < -6:
        iv = interval.Interval(iv.semitones + 12)
    return int(iv.semitones)


def part_events(part):
    """[(offset_ql, dur_ql, midi)] taking chord tops; None if too polyphonic."""
    evs, poly = [], 0
    for n in part.flatten().notes:
        if n.isChord:
            poly += 1
            p = max(x.midi for x in n.pitches)
        else:
            p = n.pitch.midi
        evs.append((float(n.offset), float(n.quarterLength), p))
    if not evs or poly / len(evs) > 0.3:
        return None
    return evs


def events_to_grid(evs, n_steps, shift):
    grid = np.full(n_steps, REST_G, dtype=int)
    off_grid = 0
    for off, dur, midi in evs:
        s = off / STEP
        if abs(s - round(s)) > 0.1:
            off_grid += 1
            continue
        s = int(round(s))
        e = min(n_steps, s + max(1, int(round(dur / STEP))))
        m = midi + shift
        while m > PITCH_HI:
            m -= 12
        while m < PITCH_LO:
            m += 12
        grid[s:e] = m
    return grid, off_grid / len(evs)


def grid_to_tokens(grid):
    toks = np.full(len(grid), REST, dtype=np.int64)
    prev = None
    for i, p in enumerate(grid):
        if p == REST_G:
            prev = None
            continue
        toks[i] = HOLD if p == prev else p - PITCH_LO + PITCH_OFFSET
        prev = p
    return toks


def measure_phase(score, n_steps):
    """Per-16th phase = position within its measure (mod 16)."""
    phase = np.zeros(n_steps, dtype=np.int64)
    part = score.parts[0] if score.parts else score
    for meas in part.getElementsByClass("Measure"):
        s = int(round(float(meas.offset) / STEP))
        ln = int(round(float(meas.quarterLength) / STEP)) or 16
        for j in range(s, min(n_steps, s + ln)):
            phase[j] = (j - s) % 16
    return phase


def extract(path):
    score = converter.parse(str(path))
    parts = list(score.parts)
    cands = []
    for part in parts:
        evs = part_events(part)
        if evs and len(evs) >= 20:
            cands.append((np.mean([e[2] for e in evs]), len(evs), evs))
    # dedupe near-identical parts (repeated tracks in some editions)
    dedup = []
    for mp, n, evs in sorted(cands, reverse=True):
        if any(abs(mp - mp2) < 0.5 and abs(n - n2) <= 2 for mp2, n2, _ in dedup):
            continue
        dedup.append((mp, n, evs))
    if len(dedup) < 3:
        return None, "lt3_parts"
    shift = transpose_iv(score)
    # dedup is sorted by mean pitch descending: [0] = soprano, [1] = alto
    n_steps = int(max(e[0] + e[1] for _, _, evs in dedup[:2] for e in evs) / STEP) + 1
    sop, og1 = events_to_grid(dedup[0][2], n_steps, shift)
    alto, og2 = events_to_grid(dedup[1][2], n_steps, shift)
    if max(og1, og2) > 0.2:
        return None, "off_grid"
    tok = np.stack([grid_to_tokens(sop), grid_to_tokens(alto)], axis=1)
    return (tok, measure_phase(score, n_steps), shift), None


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
        if f"tok_{sid}" in arrays:  # one score per CPDL page is enough
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
        tok, phase, shift = r
        arrays[f"tok_{sid}"], arrays[f"phase_{sid}"] = tok, phase
        meta[sid] = {"file": f.name, "shift": shift, "ticks": int(len(tok))}
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(files)}] paired {len(meta)}", flush=True)
    np.savez_compressed(out_dir / "cpdl_v2.npz", **arrays)
    (out_dir / "cpdl_meta.json").write_text(json.dumps({"songs": meta}, indent=0))
    ticks = sum(m["ticks"] for m in meta.values())
    print(f"paired {len(meta)} works ({ticks} ticks ~ {2*ticks} interleaved tokens) "
          f"from {len(files)} files; skips: {dict(skips)}")


if __name__ == "__main__":
    main()
