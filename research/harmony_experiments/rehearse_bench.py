"""rehearse_bench.py - score-tracker scoring over 12 pairs of double takes.

The driver for the three earlier scoring rounds was ad hoc and never kept, so it is
written down from here on: one harness (--key 0 --shared-octave, which has already
reproduced v2 at 41.5% with an r-hat of 1.04 on the first pair) runs all 12 pairs
against N trackers in one go and prints the comparison table.

Run（DDSP venv）:
  .../260724_ddsp_svc/venv/bin/python rehearse_bench.py [--trackers v2,v3]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rehearse_ab import lead_stream, run_A, sounding  # noqa: E402

PAIRS = HERE / "../../../260730_recording/pairs"

# The ceiling and the reliability come from oracle_table.txt, the frozen DTW ground-truth upper bound
CEIL = {  # pair: (conf, oracle_voiced%, tick_ratio)
    "pair01": ("high", 81.1, 1.04), "pair02": ("high", 65.3, 1.03),
    "pair03": ("high", 48.1, 1.02), "pair04": ("medium", 63.8, 0.99),
    "pair05": ("high", 44.8, 0.99), "pair06": ("medium", 49.4, 1.21),
    "pair07": ("medium", 30.7, 1.22), "pair08": ("high", 37.2, 1.09),
    "pair09": ("medium", 38.1, 1.27), "pair10": ("medium", 45.0, 1.06),
    "pair11": ("high", 34.1, 1.26), "pair12": ("medium", 29.8, 1.20),
}


def main(a):
    trackers = a.trackers.split(",")
    rows = []
    for pair, (conf, ceil, ratio) in CEIL.items():
        reh, _, oct_r = lead_stream(str(PAIRS / f"{pair}_take1.wav"), "0")
        test, _, _ = lead_stream(str(PAIRS / f"{pair}_take2.wav"), "0",
                                 octave=oct_r)
        score = sounding(reh)
        res = {}
        for tr in trackers:
            h = run_A(test, score, tr)
            res[tr] = (100.0 * h.hit / h.n, 100.0 * h.hit_on / h.n_on,
                       h.tracker.jumps)
        rows.append((pair, conf, ceil, ratio, res))
        cells = "  ".join(f"{tr} {res[tr][0]:5.1f}%/{res[tr][1]:5.1f}%"
                          f" j{res[tr][2]}" for tr in trackers)
        print(f"{pair} {conf:<7} ceil {ceil:4.1f}  {cells}", flush=True)

    print(f"\n{'pair':<8}{'conf':<8}{'ratio':>6}{'ceil%':>7}", end="")
    for tr in trackers:
        print(f"{'A ' + tr:>8}{'on%':>7}{'A/ceil':>8}{'jmp':>5}", end="")
    print()
    print("-" * (29 + 28 * len(trackers)))
    for pair, conf, ceil, ratio, res in rows:
        print(f"{pair:<8}{conf:<8}{ratio:>6.2f}{ceil:>7.1f}", end="")
        for tr in trackers:
            v, o, j = res[tr]
            print(f"{v:>8.1f}{o:>7.1f}{v / ceil:>8.2f}{j:>5}", end="")
        print()
    print("-" * (29 + 28 * len(trackers)))
    for label, keys in (
            ("MEAN all 12", list(CEIL)),
            ("MEAN high-conf 6", [p for p, v in CEIL.items() if v[0] == "high"])):
        sel = [r for r in rows if r[0] in keys]
        mc = sum(r[2] for r in sel) / len(sel)
        print(f"{label:<22}{mc:>7.1f}", end="")
        for tr in trackers:
            mv = sum(r[4][tr][0] for r in sel) / len(sel)
            mo = sum(r[4][tr][1] for r in sel) / len(sel)
            mr = sum(r[4][tr][0] / r[2] for r in sel) / len(sel)
            print(f"{mv:>8.1f}{mo:>7.1f}{mr:>8.2f}{'':>5}", end="")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", default="v2,v3")
    main(ap.parse_args())
