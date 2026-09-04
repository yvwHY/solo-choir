"""reh_polish.py — post-processing for a rehearsal score: resolve sustained
vertical dissonance in place (2026-08-01)

The dividend of pre-rendering: a rehearsal file is a fixed offline score, so the
chord content can be cleaned by rule before rendering, without the model. The
rules are deliberately conservative:

- Only dissonance held against the lead for MIN_RUN ticks or more is touched,
  that is, a clash sustained for 375 ms or longer. A single-tick passing note or
  suspension is vocabulary and is left alone.
- The repair moves the whole run of that part to the nearest candidate note that
  is in key (the root decides, see below), consonant with the modal note of the
  lead over that run, ideally not clashing with the other part either, and no
  more than 5 semitones away. If nothing qualifies, it gives up.

**The caller must guarantee that d["lead"] and d["upper"]/d["lower"] are in the
same pitch space.** The bug caught on 2026-08-04: respond2 compared a lead in
model space with parts in microphone space, offset by key plus octave. It was
invisible only when the offset happened to be a multiple of 12, so `--key 0` was
correct and any real key compared wrongly throughout. The same applies to root:
the older version hard-coded absolute C major, so any `--key` pulled the parts
onto the wrong scale. root is the pitch class of the tonic in microphone space;
`--key N` gives root = (-N) % 12.
- The statistics print before and after, in the mod-12 terms used by
  render_v3.

Run:
  python reh_polish.py out/pair06_reh0_notes.json out/pair06_reh0p_notes.json
"""
import json
import sys
from collections import Counter

MAJ = {0, 2, 4, 5, 7, 9, 11}
DISS = {1, 2, 6, 10, 11}
MIN_RUN = 2


def stats(d):
    n = diss = n2 = d2 = 0
    for L, U, Lo in zip(d["lead"], d["upper"], d["lower"]):
        if L is not None:
            for a in (U, Lo):
                if a is not None:
                    n += 1
                    diss += abs(int(a) - int(L)) % 12 in DISS
        if U is not None and Lo is not None:
            n2 += 1
            d2 += abs(int(U) - int(Lo)) % 12 in DISS
    return 100.0 * diss / max(n, 1), 100.0 * d2 / max(n2, 1)


def runs(ns):
    """Runs of equal (start, end, note); None runs are skipped."""
    out, i = [], 0
    while i < len(ns):
        if ns[i] is None:
            i += 1
            continue
        j = i
        while j < len(ns) and ns[j] == ns[i]:
            j += 1
        out.append((i, j, int(ns[i])))
        i = j
    return out


def polish(d, voice, other, root=0):
    ns, lead, ot = d[voice], d["lead"], d[other]
    fixed = 0
    for a, b, m in runs(ns):
        clash = [t for t in range(a, b) if lead[t] is not None
                 and abs(m - int(lead[t])) % 12 in DISS]
        if len(clash) < MIN_RUN:
            continue
        lm = Counter(int(lead[t]) for t in clash).most_common(1)[0][0]
        oth = [int(ot[t]) for t in range(a, b) if ot[t] is not None]
        om = Counter(oth).most_common(1)[0][0] if oth else None

        def ok(c, strict):
            if (c - root) % 12 not in MAJ or abs(c - lm) % 12 in DISS:
                return False
            return not (strict and om is not None and abs(c - om) % 12 in DISS)

        for strict in (True, False):
            cand = next((m + s * k for k in range(1, 6) for s in (-1, 1)
                         if ok(m + s * k, strict)), None)
            if cand is not None:
                break
        if cand is None:
            continue
        for t in range(a, b):
            ns[t] = cand
        fixed += 1
    return fixed


def main(src, dst):
    d = json.load(open(src))
    # A lead written by live_v3 is in model space while the parts are in
    # microphone space, the offline shape of the 2026-08-04 bug. v2t.shift is
    # always a multiple of 12, so in mod-12 terms the two spaces differ only by
    # k_shift, and since polish and the statistics compare in mod-12 throughout,
    # subtracting k_shift puts them in the same space. Only the view used for
    # comparison is converted; the lead written to file stays in model space,
    # which is what downstream consumers such as ScoreTracker expect.
    k = int(d.get("k_shift") or 0)
    root = (-k) % 12
    v = dict(d, lead=[None if x is None else int(x) - k
                      for x in d.get("lead", [])]) if k else d
    print(f"space: k_shift {k:+d} -> root {root} (lead converted to microphone space for comparison)"
          if k else "space: k_shift 0, the two spaces coincide in mod-12, root 0")
    b = stats(v)
    fu = polish(v, "upper", "lower", root)
    fl = polish(v, "lower", "upper", root)
    a = stats(v)
    json.dump(d, open(dst, "w"))
    print(f"polish: {fu} runs repaired in upper, {fl} in lower")
    print(f"parts vs lead dissonance {b[0]:.1f}% -> {a[0]:.1f}% | "
          f"upper vs lower {b[1]:.1f}% → {a[1]:.1f}%")
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
