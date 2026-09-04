#!/usr/bin/env python
"""Practice-log report for the learning study: bone conduction against headphones, and
single-learner before/after.

ADDITIVE, READ-ONLY: reads recordings/practice_log.csv (written by bridge.log_practice,
gitignored) and prints two voiced-time-weighted tables. Touches no engine/app code.

The log's per-attempt columns are:
    time, part, key, scale, voiced_s, match_pct, mean_abs_cents, on_target_s
(see specs/2026-06-23-track-a-ear-trainer-design.md Phase 4). The study's INDEPENDENT
VARIABLE — delivery `channel` (bone / headphone / speaker), and `subject` — are NOT in
the log yet (added to logging only when Test A passes and the study runs; see
specs/2026-06-27-spine-positioning-decision.md). This script already groups by them IF
present, so it needs no change once they're added.

Run from the repo root (conda env vcclient-dev):
    python eval/practice_report.py
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG = REPO_ROOT / "recordings" / "practice_log.csv"
OUT = REPO_ROOT / "recordings" / "practice_report.md"   # generated, gitignored

# Optional independent-variable / grouping columns, used only if present in the log.
IV_COLS = ["subject", "channel"]


def aggregate(rows, group_cols):
    """Voiced-time-weighted aggregate per group. Returns list of dict rows, sorted."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        try:
            v = float(r["voiced_s"])
            cents = float(r["mean_abs_cents"])
            match = float(r["match_pct"])
            on_t = float(r["on_target_s"])
        except (KeyError, ValueError):
            continue
        if v <= 0:
            continue
        key = tuple(r.get(c, "") for c in group_cols)
        g = groups.setdefault(key, {"n": 0, "voiced_s": 0.0, "on_target_s": 0.0,
                                    "_cents_w": 0.0, "_match_w": 0.0})
        g["n"] += 1
        g["voiced_s"] += v
        g["on_target_s"] += on_t
        g["_cents_w"] += cents * v      # voiced-time-weighted
        g["_match_w"] += match * v
    out = []
    for key, g in groups.items():
        v = g["voiced_s"]
        row = dict(zip(group_cols, key))
        row.update({
            "n": g["n"],
            "voiced_s": round(v, 1),
            "mean_abs_cents": round(g["_cents_w"] / v, 1),
            "match_pct": round(100.0 * g["_match_w"] / v, 1),
            "on_target_pct": round(100.0 * g["on_target_s"] / v, 1),
        })
        out.append(row)
    return sorted(out, key=lambda r: tuple(str(r[c]) for c in group_cols))


def md_table(title, note, rows, cols):
    lines = [f"## {title}", "", note, ""]
    if not rows:
        lines += ["_(no data)_", ""]
        return "\n".join(lines)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    if not LOG.exists():
        print(f"No practice log yet: {LOG}\n(Practise a single harmony part in the live app to write rows.)")
        return
    with open(LOG, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        print(f"Practice log is empty: {LOG}")
        return

    present_iv = [c for c in IV_COLS if c in header]

    # Table 1 — the study view: group by present IVs (channel/subject) × part.
    g1_cols = present_iv + ["part"]
    t1 = aggregate(rows, g1_cols)
    metric_cols = ["n", "voiced_s", "mean_abs_cents", "match_pct", "on_target_pct"]
    if present_iv:
        note1 = (f"Independent variable(s) present: **{', '.join(present_iv)}**. "
                 f"Lower `mean_abs_cents` / higher `on_target_pct` = better held the part.")
    else:
        note1 = ("No `channel`/`subject` column in the log yet → grouped by part only. "
                 "Add those to logging when the bone-vs-headphone study runs "
                 "(see specs/2026-06-27-spine-positioning-decision.md).")

    # Table 2 — single-learner before/after: part × date.
    for r in rows:
        r["date"] = (r.get("time", "") or "")[:10]
    t2 = aggregate(rows, ["part", "date"])

    parts = [
        f"# Practice report\n\nSource: `{LOG.relative_to(REPO_ROOT)}` ({len(rows)} attempts). "
        "Voiced-time-weighted. `mean_abs_cents` = pitch error from the target note; "
        "`on_target_pct` = share of sung time within tolerance.\n",
        md_table("Study view — by " + " × ".join(g1_cols), note1, t1, g1_cols + metric_cols),
        md_table("Before/after — by part × date",
                 "Trend for one learner across sessions (lower cents / higher on-target over dates = learning).",
                 t2, ["part", "date"] + metric_cols),
    ]
    report = "\n".join(parts)
    OUT.write_text(report)
    print(report)
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
