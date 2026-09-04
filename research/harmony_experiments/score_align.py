"""score_align.py - align a sung phrase to a score (the alignment side of MIDI harmony).

`score_notes.py` produces the score's own timeline. To use it as the note line
for the parts, two questions have to be answered first: which passage of the
score the singer just sang, and how their rhythm maps onto the score's.

**Why there is an opening here where the tracker of 2026-08-01 died** (37.9% hit
rate against a material ceiling of 49.4%, which condemned the pre-rendered live
half): that tracker had to be causal and immediate, because the parts sing at the
same time as the singer, so it had to guess the position while the phrase was
still going, and a wrong guess was a wrong note there and then. The answering
mode is entirely different:

  - Alignment happens AFTER the phrase ends, with the whole phrase in hand, so it
    can look at the future; DTW is not an online algorithm.
  - The score is FIXED, unlike "the same song he sang last time", which drifts on
    its own - and that drift is exactly where the 49.4% ceiling came from: two
    takes are sung differently, so no amount of alignment accuracy helps.
  - A phrase boundary is a breath, which is usually a phrase boundary in the score
    too, so the search starts roughly in the right place.

So this uses subsequence DTW, free to start and end anywhere in the score because
the singer sings a small part of the whole, with a cursor carried between phrases:
where the last phrase ended is where the next one starts looking, which is the
monotone progress the answering mode gives for free.

Transposition is not guessed once: every candidate semitone shift is run and the
cheapest wins, since the key actually sung need not match what score_notes chose.

**Fall back to the model when confidence is low**: when the alignment does not
convince, let the model improvise rather than paste the wrong harmony on. That
fallback is the precondition for this feature going on stage, not a safety net.

Run (self-check, two takes of the same song standing in for score against live):
  .../python score_align.py --selftest
"""
import numpy as np

REST = 2.0          # cost of one side sounding while the other rests, in semitones
CAP = 6.0           # ceiling on pitch distance: an outlier that fits nothing should not dominate the path
BIG = 1e9


def _cost(a, b):
    """Distance between two sounding MIDI notes, where None is a rest."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return REST
    return min(abs(a - b), CAP)


def dtw(sung, score, free_ends=True, skip=0.5):
    """Subsequence DTW: all of the sung line must be used, the score only in part.

    Returns (cost_per_tick, j_of_i), where j_of_i[i] is the score cell the
    singer's i-th tick aligns to. Steps of (1,1), (1,0) and (0,1) are allowed, so
    the singer may go faster or slower and may skip notes in the score, but a
    non-diagonal step PAYS a skip cost. Without that penalty it collapses: the
    first version glued a 23-tick phrase onto 2 cells of the score at a cost of
    0.00, having found a rest to stuff the whole phrase into, and still printed a
    100% hit rate. A degenerate path is not caught by the cost, only by the shape
    of the steps."""
    n, m = len(sung), len(score)
    if n == 0 or m == 0:
        return BIG, []
    D = np.full((n + 1, m + 1), BIG)
    D[0, 0] = 0.0
    if free_ends:
        D[0, :] = 0.0                     # any cell of the score may be the start
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        ci = sung[i - 1]
        for j in range(1, m + 1):
            c = _cost(ci, score[j - 1])
            d, k = D[i - 1, j - 1], 0     # diagonal: one tick to one cell
            if D[i - 1, j] + skip < d:
                d, k = D[i - 1, j] + skip, 1   # down: the singer holds and the score waits
            if D[i, j - 1] + skip < d:
                d, k = D[i, j - 1] + skip, 2   # right: the singer is ahead and skips a cell
            D[i, j] = d + c
            ptr[i, j] = k
    j_end = int(np.argmin(D[n, 1:])) + 1 if free_ends else m
    total = D[n, j_end]
    j_of_i = [0] * n
    i, j = n, j_end
    while i > 0:
        j_of_i[i - 1] = j - 1
        k = ptr[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
        if j < 1:                          # against the left edge: the rest sticks to the first cell
            while i > 0:
                j_of_i[i - 1] = 0
                i -= 1
            break
    return total / max(1, n), j_of_i


def align(sung, score, cursor=0, back=8, ahead=None, offsets=range(-12, 13)):
    """This phrase to its place in the score.

    cursor is where the last phrase aligned. The answering mode is monotone by
    nature, since the singer works through the piece, so the window only opens
    `back` cells backwards (8 by default, about 1.5 s, tolerating an overshoot at
    the end of the last phrase) rather than searching freely: at back=40 the third
    phrase jumped behind the second and the sixth behind the fifth, and with any
    repeated material it gets pulled away. The window cannot be dropped either:
    scanning the whole score for every phrase is slow and more easily pulled to a
    distant repeat, which was the known weakness of the 08-01 tracker.

    Returns a dict: j_of_i, cost, offset, start, end. cost is the mean per tick and
    doubles as confidence: above a threshold, fall back to the model."""
    lo = max(0, cursor - back)
    hi = len(score) if ahead is None else min(len(score), cursor + ahead)
    win = score[lo:hi]
    best = None
    for o in offsets:
        sh = [None if p is None else p + o for p in win]
        c, path = dtw(sung, sh)
        if best is None or c < best[0]:
            best = (c, path, o)
    c, path, o = best
    j = [p + lo for p in path]
    return {"j_of_i": j, "cost": c, "offset": o,
            "start": j[0] if j else lo, "end": j[-1] if j else lo}


def map_parts(a, score_parts, keys=("upper", "lower")):
    """Apply the alignment to the other parts: when the singer sings their i-th
    tick, the parts sing the same cell of the score. The transposition has to move
    with the lead, or the harmonic relationship is skewed."""
    j = a["j_of_i"]
    return {k: [None if score_parts[k][x] is None else score_parts[k][x] + a["offset"]
                for x in j] for k in keys}


def confidence(sung, score, a):
    """Confidence between 0 and 1, used to decide whether to fall back to the model.

    **The cost cannot be used as confidence.** Measured on the synthetic check, the
    badly aligned group had a median cost of 0.15, LOWER than the 0.31 of the well
    aligned group, because the typical failure is collapsing onto a rest, and rest
    against rest costs 0: the more degenerate the path, the better the cost looks.
    I got this wrong in the first version and it is left here as a warning.

    Three things that fail together instead:
      agree  the share of ticks where the sung note is within 1 semitone of the
             score cell, counting only ticks where both sound
      onrest the share of sung notes landing on a rest in the score, the direct
             fingerprint of a collapse
      slope  score cells advanced over singer ticks, about 1 when healthy and far
             below 1 in a collapse
    """
    j = a["j_of_i"]
    o = a["offset"]
    vv = miss = onrest = 0
    for i, p in enumerate(sung):
        if p is None:
            continue
        q = score[j[i]]
        if q is None:
            onrest += 1
            continue
        vv += 1
        miss += abs(p - (q + o)) > 1.0
    n_v = vv + onrest
    agree = (vv - miss) / max(1, vv)
    onrest_r = onrest / max(1, n_v)
    slope = (j[-1] - j[0] + 1) / max(1, len(j))
    conf = agree * (1 - onrest_r) * min(1.0, slope / 0.6)
    return {"conf": conf, "agree": agree, "onrest": onrest_r, "slope": slope}


def hit_rate(sung, score_lead, a, tol=1.0):
    """After alignment, the share of ticks whose sung note is within tol semitones of the score cell.

    Read with care: this is the quantity DTW minimises, so it is HIGH by
    construction and cannot on its own be evidence of a successful alignment. Its
    use is comparison against the oracle below; the gap is the information."""
    ok = tot = 0
    for i, p in enumerate(sung):
        if p is None:
            continue
        q = score_lead[a["j_of_i"][i]]
        tot += 1
        ok += q is not None and abs(p - (q + a["offset"])) <= tol
    return ok / max(1, tot), tot


def oracle_rate(sung, score_lead, offset, lo=0, hi=None, tol=1.0):
    """The ceiling: ignoring time order, every sung note takes the best cell in the
    window. This falls when the material itself does not contain the note, rather
    than the alignment being at fault."""
    hi = len(score_lead) if hi is None else hi
    cand = [q + offset for q in score_lead[lo:hi] if q is not None]
    if not cand:
        return 0.0
    cand = np.array(cand)
    hit = [np.min(np.abs(cand - p)) <= tol for p in sung if p is not None]
    return float(np.mean(hit)) if hit else 0.0


def _selftest():
    """Two takes of the same song standing in for score against live singing, which
    is the real difficulty: take 1's note line is the score and take 2 is what is
    being sung now. This is the same material the real-time tracker of 08-01 ran on
    (37.9% against a material ceiling of 49.4%), so the numbers compare directly:
    the only difference is offline whole-phrase DTW against causal real-time tracking."""
    import sys
    import time
    from pathlib import Path
    import soundfile as sf
    import torch
    H = Path(__file__).parent
    sys.path.insert(0, str(H))
    import respond2 as R
    from live_v3 import EarV3
    from pitch import SR

    base = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
            "SoloChoirCode/260730_recording/pairs")

    def lead_of(path):
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        x = np.ascontiguousarray(x[:, 0])
        assert sr == SR
        torch.manual_seed(1234)
        ear = EarV3(indep=0.0, stab=2, key=3)
        ear.v2t.shift = 0
        out = []
        for s, e in R.find_phrases(x, 0.35, 0.5):
            lead, _, _ = R.phrase_notes(ear, x[s:e])
            out.append(lead)
        return out

    print("extracting note lines...", flush=True)
    score_ph = lead_of(f"{base}/pair06_take1.wav")     # standing in for the score
    sung_ph = lead_of(f"{base}/pair06_take2.wav")      # standing in for live singing
    score = [p for ph in score_ph for p in ph]         # the score is one continuous piece
    print(f"score {len(score)} ticks (take1, {len(score_ph)} phrases) | "
          f"sung {len(sung_ph)} phrases (take2)")

    cursor, hits, oras, costs, t0 = 0, [], [], [], time.perf_counter()
    for i, sung in enumerate(sung_ph):
        a = align(sung, score, cursor=cursor,
                  ahead=cursor + 4 * len(sung) + 60)
        h, n = hit_rate(sung, score, a)
        lo = max(0, a["start"] - 10)
        o = oracle_rate(sung, score, a["offset"], lo, a["end"] + 10)
        cursor = a["end"]
        hits.append(h * n)
        oras.append(o * n)
        costs.append(a["cost"])
        print(f"  phrase {i+1:2d} {len(sung):3d} ticks -> score [{a['start']:3d},{a['end']:3d}]"
              f" offset {a['offset']:+3d} | cost {a['cost']:.2f} | hit {h*100:3.0f}%"
              f" (ceiling for that window {o*100:3.0f}%)")
    ntot = sum(len([p for p in s if p is not None]) for s in sung_ph)
    print(f"\ntotal: hit {100*sum(hits)/ntot:.1f}% | material ceiling "
          f"{100*sum(oras)/ntot:.1f}% | {1000*(time.perf_counter()-t0)/len(sung_ph):.0f}ms per phrase")
    print("for reference, the causal real-time tracker on the same material: 37.9% against a ceiling of 49.4%")
    print("Three warnings; miss one and these numbers read as something they are not:")
    print("  1. The hit rate is the quantity DTW itself minimises, so it is high by")
    print("     construction and is not on its own evidence of success.")
    print("  2. The ceiling is defined differently here: this oracle takes any cell in")
    print("     the window, which is far looser than the per-tick definition used on")
    print("     08-01, so the two ceilings are not comparable. What is comparable is the")
    print("     contrast itself, offline whole-phrase against causal real-time on the")
    print("     same material.")
    print("  3. What is not measured at all is RHYTHM: the hit rate only asks whether")
    print("     the pitch is right, and where the harmony lands in time is for the ear.")


def _synth(trials=200, seed=20260803):
    """Synthetic check: an alignment test WITH a ground truth.

    Two takes of the same song cannot test an aligner - they are two different
    improvisations with no correct answer, and what gets measured mixes alignment
    accuracy with the takes simply differing. This does the opposite: take a passage
    of the score, distort it into "what a singer would do" in a KNOWN way, and see
    whether the aligner can send every tick back where it came from.

    The four distortions follow real singing: local tempo between 0.7 and 1.4 times,
    pitch wobble of half a semitone with the occasional whole semitone error, dropped
    words and breaths turning into rests, and a wrong key as an overall shift.
    """
    import glob
    import sys
    from pathlib import Path
    H = Path(__file__).parent
    sys.path.insert(0, str(H))
    rng = np.random.default_rng(seed)
    fs = sorted(glob.glob("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
                          "SoloChoirCode/260723_corpus/cpdl_partsongs/*.mid"))
    import score_notes as SN
    from score_notes import TICK_S
    scores = []
    for f in fs:
        if len(scores) >= 40:
            break
        try:
            out, _ = SN.build(f)
        except Exception:
            continue
        if sum(p is not None for p in out["lead"]) > 60:
            scores.append(out)
    print(f"score library: {len(scores)} pieces (CPDL, real arrangements)")

    errs, exact, costs, confs, harm = [], [], [], [], []
    for t in range(trials):
        full = scores[rng.integers(len(scores))]
        sc = full["lead"]
        a0 = int(rng.integers(0, max(1, len(sc) - 40)))
        b0 = min(len(sc), a0 + int(rng.integers(12, 40)))
        off = int(rng.integers(-5, 6))
        sung, truth = [], []
        j = a0
        while j < b0:
            rep = 1 + (rng.random() < 0.25) - (rng.random() < 0.25)   # 0/1/2
            for _ in range(max(0, rep)):
                p = sc[j]
                if p is not None:
                    p = p + off + int(rng.random() < 0.10) * rng.choice([-1, 1])
                    if rng.random() < 0.08:
                        p = None                      # a breath, or a dropped word
                sung.append(p)
                truth.append(j)
            j += 1
        if len(sung) < 8:
            continue
        a = align(sung, sc, cursor=a0, ahead=a0 + 4 * len(sung) + 60)
        e = np.abs(np.array(a["j_of_i"]) - np.array(truth))
        errs.append(np.median(e))
        exact.append(float(np.mean(e <= 1)))
        costs.append(a["cost"])
        confs.append(confidence(sung, sc, a))
        # What actually matters: the audience hears harmony, not indices. Aligning to
        # a repeat puts the position "wrong" while the harmony there is often the
        # same, so position error overstates the real damage.
        got = map_parts(a, full)
        agree = []
        for i, jt in enumerate(truth):
            for k in ("upper", "lower"):
                t_, g_ = full[k][jt], got[k][i]
                if t_ is None and g_ is None:
                    agree.append(1.0)
                elif t_ is None or g_ is None:
                    agree.append(0.0)
                else:
                    agree.append(float((t_ + a["offset"] - g_) % 12 == 0))
        harm.append(float(np.mean(agree)) if agree else 0.0)
    errs, exact, costs = np.array(errs), np.array(exact), np.array(costs)
    print(f"{len(errs)} trials: median position error **{np.median(errs):.1f} ticks**"
          f"（{np.median(errs)*TICK_S*1000:.0f}ms）｜p90 {np.percentile(errs,90):.1f}")
    print(f"share within +/-1 tick: median {100*np.median(exact):.0f}%"
          f" | overall {100*exact.mean():.0f}%")
    print(f"cost (confidence) median {np.median(costs):.2f}  p90 {np.percentile(costs,90):.2f}")
    good = exact > 0.7
    print(f"well aligned (over 70% within +/-1 tick): {100*good.mean():.0f}%")
    print(f"  cost           good {np.median(costs[good]):.2f} / "
          f"bad {np.median(costs[~good]):.2f}  <- INVERTED, unusable as confidence")
    cf = np.array(confs)
    for k in ("conf", "agree", "onrest", "slope"):
        g, b = np.median([c[k] for c, m in zip(cf, good) if m]), \
               np.median([c[k] for c, m in zip(cf, good) if not m])
        print(f"  {k:8s}      good {g:.2f} / bad {b:.2f}"
              f"{'   <- usable' if k == 'conf' and g > b + 0.15 else ''}")
    thr = 0.5
    keep = np.array([c["conf"] >= thr for c in cf])
    if keep.any():
        harm = np.array(harm)
    print(f"\n**harmonic agreement (interval class mod 12, what the parts actually "
          f"sing against the ground truth)**: median {100*np.median(harm):.0f}% | "
          f"mean {100*harm.mean():.0f}% | well aligned {100*harm[good].mean():.0f}% | "
          f"badly aligned {100*harm[~good].mean():.0f}%")
    print("  -> position error overstates the damage: aligning to a repeat usually gives the same harmony.")
    print(f"  at conf >= {thr}: {100*keep.mean():.0f}% of phrases kept, "
              f"of which {100*good[keep].mean():.0f}% are well aligned "
              f"(against {100*good.mean():.0f}% with no threshold)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="two takes of one song; no ground truth, shape only")
    ap.add_argument("--synth", action="store_true",
                    help="synthetic check: with a ground truth, measuring the aligner itself")
    a = ap.parse_args()
    if a.synth:
        _synth()
    elif a.selftest:
        _selftest()
