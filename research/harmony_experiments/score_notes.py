"""score_notes.py - an a cappella MIDI file to a three-part tick grid (the score side).

Alongside the harmony the model improvises, this opens a second route where the
harmony is decided by an arrangement written in advance.

**Nothing downstream has to change**: the format for an external note line already
exists - the rehearsal file `prerender_stems.py` consumes,
`{"lead": [...], "upper": [...], "lower": [...]}`, one sounding MIDI pitch or None
per tick. The two voices, the target f0 recipe, the polish, the reverb and the mix
balance have all been judged by ear already; this file only turns a score into
that dict.

This handles ONLY the score side, on the score's own timeline. Aligning it to a
phrase he sang and putting it in his rhythm is the next stage, DTW, and that is
where the risk is. They are deliberately separate, because the score side can be
validated in bulk against the 4016 CPDL files while alignment can only be
validated against his own recordings.

What the entry layer has to guess, since a found arrangement does not follow your
format: which tracks are voices, which is the melody, which two go to the parts,
and whether to transpose. **Every one can be overridden by a flag**: a score you
wrote yourself works by default, and a found one is specified when the guess is
wrong. One path, not two.

This runs under vcclient-dev, where pretty_midi lives, not the DDSP venv. There is
no conflict: it is an offline step whose output is json, and rendering still runs
under the DDSP venv.

Run（vcclient-dev）:
  python score_notes.py song.mid --out out/song_notes.json      # guess
  python score_notes.py song.mid --melody 0 --upper 1 --lower 3 --transpose -5
  python score_notes.py --survey '../../260723_corpus/cpdl_sample/*.mid'
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pitch import SR  # noqa: E402
from render_v3 import GIRL_RANGE, HARRY_RANGE, fold  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402

TICK_S = TICK_SAMPS / SR                      # 0.1875 s, the same cell as the model's tick
# His comfortable range, from the voicing_mask note: 110-300 Hz, that is A2 to D4,
# taken slightly wider. This is the target for transposing the lead: a score's
# melody will not happen to sit in his range, and a line he cannot reach is not
# worth discussing.
LEAD_RANGE = (45, 64)                         # A2–E4


def load_parts(path, max_poly=0.15):
    """MIDI to a list of monophonic parts, each [(start, end, pitch)], ordered by median pitch from high to low.

    A found score contains anything: piano reductions, four parts crammed onto one
    track, empty tracks. The filters are:
      - drop tracks with too few notes (title tracks, click tracks)
      - **drop tracks whose chord share exceeds max_poly**, which means a reduction
        or several parts on one track, not a single line
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    parts = []
    for inst in pm.instruments:
        if inst.is_drum or len(inst.notes) < 8:
            continue
        ns = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        # overlap: an onset before the previous note ends means they sound together
        ov = sum(1 for a, b in zip(ns, ns[1:]) if b.start < a.end - 1e-3)
        if ov / max(1, len(ns) - 1) > max_poly:
            continue
        parts.append({
            "name": inst.name.strip() or f"track{len(parts)}",
            "notes": [(n.start, n.end, n.pitch) for n in ns],
            "med": float(np.median([n.pitch for n in ns])),
        })
    parts.sort(key=lambda p: -p["med"])        # high to low, that is S A T B
    return parts, pm.get_end_time()


def pick(parts, melody=None, upper=None, lower=None):
    """Choose the indices of (lead, upper, lower).

    By default the MELODY IS THE HIGHEST PART, which in a found arrangement is
    almost always what a person sings, and the parts take the two nearest lines
    above and below it, the bracketing arrangement of the v3 spec. When the melody
    is the highest part there is nothing above it, so a line is borrowed from below
    and folded up an octave past the lead; the bracketing swapping over at the
    moment of folding is a cost the spec already accepts."""
    n = len(parts)
    if n < 3:
        raise ValueError(f"only {n} monophonic parts found, at least 3 are needed "
                         f"(--melody/--upper/--lower can specify them, or this score is a reduction)")
    li = 0 if melody is None else melody
    if upper is not None and lower is not None:
        return li, upper, lower
    rest = [i for i in range(n) if i != li]
    above = [i for i in rest if parts[i]["med"] > parts[li]["med"]]
    below = [i for i in rest if parts[i]["med"] <= parts[li]["med"]]
    ui = (above[-1] if above else below[0]) if upper is None else upper
    lo = [i for i in below if i != ui]
    li_ = (lo[0] if lo else [i for i in rest if i != ui][0]) if lower is None else lower
    return li, ui, li_


def to_ticks(notes, n_ticks, tick_s=TICK_S):
    """(start, end, pitch) to one sounding pitch per tick, None for a rest.
    Where a cell holds more than one note, the one occupying most of it wins. A tick
    of 0.1875 s is longer than a sixteenth, so a fast run gets swallowed; that is an
    existing cost of the tick grid, not something introduced here."""
    out = []
    for k in range(n_ticks):
        t0, t1 = k * tick_s, (k + 1) * tick_s
        best, best_ov = None, 0.0
        for s, e, p in notes:
            if e <= t0:
                continue
            if s >= t1:
                break
            ov = min(e, t1) - max(s, t0)
            if ov > best_ov:
                best, best_ov = p, ov
        out.append(best)
    return out


def build(path, melody=None, upper=None, lower=None, transpose=None):
    parts, dur = load_parts(path)
    li, ui, lo = pick(parts, melody, upper, lower)
    n_ticks = int(np.ceil(dur / TICK_S))
    raw = {k: to_ticks(parts[i]["notes"], n_ticks)
           for k, i in (("lead", li), ("upper", ui), ("lower", lo))}
    if transpose is None:
        # automatic transposition: put the melody's median pitch into his
        # comfortable range, moving the whole score by the same amount so the three
        # lines keep their harmonic relationship
        med = float(np.median([p for p in raw["lead"] if p is not None]))
        tgt = 0.5 * (LEAD_RANGE[0] + LEAD_RANGE[1])
        transpose = int(round((tgt - med) / 12.0)) * 12      # octaves only, so the key does not change
    out = {k: [None if p is None else p + transpose for p in v]
           for k, v in raw.items()}
    # fold the parts into the measured range of their own voices, the same fold render_v3 uses, with the same cost and reasoning
    out["upper"] = fold(out["upper"], *GIRL_RANGE)
    out["lower"] = fold(out["lower"], *HARRY_RANGE)
    info = {"parts": [p["name"] for p in parts],
            "med": [round(p["med"], 1) for p in parts],
            "picked": {"lead": li, "upper": ui, "lower": lo},
            "transpose": transpose, "ticks": n_ticks,
            "dur_s": round(dur, 1)}
    return out, info


def survey(pattern, limit=400):
    """Use the CPDL batch as test material: how many found arrangements the entry layer can actually take."""
    fs = sorted(glob.glob(pattern))[:limit]
    ok = bad_parse = few = 0
    npart = []
    for f in fs:
        try:
            parts, _ = load_parts(f)
        except Exception:
            bad_parse += 1
            continue
        npart.append(len(parts))
        if len(parts) < 3:
            few += 1
        else:
            ok += 1
    npart = np.array(npart) if npart else np.zeros(1)
    print(f"{len(fs)} files: usable {ok} ({100*ok/max(1,len(fs)):.0f}%)"
          f" | parse failed {bad_parse} | fewer than 3 monophonic parts {few}")
    print(f"monophonic parts, median {np.median(npart):.0f}  "
          f"distribution {np.bincount(npart.astype(int))[:9].tolist()} (index = count)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", nargs="?")
    ap.add_argument("--out", default=None, help="write a notes json in rehearsal-file format")
    ap.add_argument("--melody", type=int, default=None,
                    help="which part is the melody (0 = highest); specify when the guess is wrong")
    ap.add_argument("--upper", type=int, default=None)
    ap.add_argument("--lower", type=int, default=None)
    ap.add_argument("--transpose", type=int, default=None,
                    help="semitones; omitted, the melody is moved into his range automatically, by octaves only")
    ap.add_argument("--survey", default=None, help="glob, to check the usable rate in bulk")
    a = ap.parse_args()
    if a.survey:
        return survey(a.survey)
    if not a.mid:
        ap.error("give a .mid, or use --survey")
    try:
        out, info = build(a.mid, a.melody, a.upper, a.lower, a.transpose)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"parts, high to low: {list(zip(info['parts'], info['med']))}")
    print(f"picked lead={info['picked']['lead']} upper={info['picked']['upper']} "
          f"lower={info['picked']['lower']} | transposed {info['transpose']:+d} semitones"
          f"｜{info['dur_s']}s = {info['ticks']} ticks")
    for k in ("lead", "upper", "lower"):
        v = [p for p in out[k] if p is not None]
        print(f"  {k:6s} sounding {100*len(v)/len(out[k]):3.0f}%  "
              f"range {min(v) if v else '-'}-{max(v) if v else '-'}  first 12 {out[k][:12]}")
    if a.out:
        json.dump({**out, "k_shift": 0, "step_samps": TICK_SAMPS},
                  open(a.out, "w"))
        print(f"wrote {a.out} (rehearsal-file format, ready for prerender_stems or respond2)")


if __name__ == "__main__":
    main()
