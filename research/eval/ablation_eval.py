#!/usr/bin/env python
"""Tenor training-step ablation eval harness (2k/5k/10k/15k checkpoints).

ADDITIVE — outside the committed engine. Imports the engine/config READ-ONLY and
shells out to the existing `--render` CLI; touches no DSP and modifies nothing in
server/ or app/. Models stay gitignored; all outputs land in recordings/ablation/
(gitignored) — nothing here should ever be committed.

What it does, for every CHECKPOINT × TAKE (config block below):
  1. Render the take's melody (L channel of *_stems.wav) through the checkpoint with
     the CURRENT pipeline — C major, the same SATB_VOICES config the app uses:
     one solo --no-you render per active part (clean per-part stem) + one full mix.
  2. Metrics on the RAW renders (natural level): F0 tracking error vs the intended
     diatonic interval (librosa.pyin → the engine's own diatonic_target_midi),
     RMS, spectral centroid → metrics.csv + metrics.md.
  3. Level-match every render to equal RMS (no clipping) → matched/  (for fair A/B).
  4. Blind set: matched renders copied under coded names → blind/ + a key CSV you
     open only AFTER scoring.

Run (conda env vcclient-dev), from the repo root:
    python eval/ablation_eval.py
Missing checkpoints/takes are skipped with a warning, so you can run incrementally
(e.g. now with only 2k+5k and one take) and re-run as you extract 10k/15k and
record the rest of the eval set.
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "app"))
sys.path.insert(0, str(REPO_ROOT / "server"))
from bridge import SATB_VOICES, SERVER_DIR, RECORDINGS_DIR, DEFAULT_MODEL  # noqa: E402
from voice_changer.SoloChoir import diatonic_target_midi, MAJOR_SCALE_PCS, f0_to_midi  # noqa: E402

# ============================== CONFIG (edit me) ==============================
# Checkpoints to compare: (label, paraphernalia model dir). Keep them side-by-side;
# never overwrite. 2k = the committed default; 5k extracted earlier. Fill 10k/15k
# once you unzip them (same man/ folder pattern).
NEWDATA = Path("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260618/test/model/data/2000")
NEW5K = Path("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260620/result")
CHECKPOINTS = [
    ("old2k",    Path(DEFAULT_MODEL)),                       # current engine model (old ~25min data)
    ("new25-2k", NEWDATA / "paraphernalia_data_00002000"),   # new F2–C5 protocol dataset, 2000 steps
    ("new25-5k", NEW5K / "paraphernalia_data_00005000"),     # same new dataset, 5000 steps
]

# Eval set: (label, take_*_stems.wav). The melody is the L channel. Add the fixed
# set as you record it: scale / sustained vowel / consonant phrase / out-of-dist.
TAKES = [
    ("heldout", Path("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260618/test/man/held-out.wav")),
]

KEY, MINOR = "C", 0          # render key/scale (the eval set is sung in C major)
NORM_CEIL = 0.97             # peak ceiling when equal-RMS level-matching (headroom, no clip)
BLIND_SEED = 20260619        # fixed → reproducible blind codes (don't read the key file early)
# pyin range: melody is C3->C4, but Bass sits ~2 octaves down (interval -7 + octave -1)
PYIN_FMIN = float(librosa.note_to_hz("C1"))   # ~32.7 Hz
PYIN_FMAX = float(librosa.note_to_hz("C6"))   # ~1046 Hz
SR = 22050                   # analysis sample rate
HOP = 512
MAX_LAG_FRAMES = 6           # ±frames searched to absorb the engine's ~25ms output latency
# =============================================================================

OUT = RECORDINGS_DIR / "ablation"
RENDERS, MATCHED, BLIND = OUT / "renders", OUT / "matched", OUT / "blind"


def active_voices():
    return [v for v in SATB_VOICES if v.get("on")]


def render_one(model_dir, mel_path, out_path, voices, no_you):
    """Shell out to the committed engine --render (read-only use). Returns True on ok."""
    cmd = [sys.executable, "beatrice_solo_choir_live.py",
           "--model", str(model_dir),
           "--render", str(mel_path), str(out_path),
           "--key", KEY, "--minor", str(MINOR),
           "--voices", json.dumps(voices)]
    if no_you:
        cmd.append("--no-you")
    r = subprocess.run(cmd, cwd=str(SERVER_DIR), capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    ! render failed ({out_path.name}): {r.stderr.strip().splitlines()[-1:] }")
        return False
    return True


def extract_melody(stems_path, mel_path):
    """L channel of the stereo take (= raw You melody) → mono source for --render."""
    data, sr = sf.read(str(stems_path), dtype="float32", always_2d=True)
    sf.write(str(mel_path), data[:, 0], sr)


def track_f0(wav_path):
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    f0, _, _ = librosa.pyin(y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=SR,
                            frame_length=2048, hop_length=HOP)
    midi = np.where(np.isfinite(f0) & (f0 > 0), 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-9) / 440.0), np.nan)
    return midi  # NaN where unvoiced


def intended_midi(in_midi, interval, octave):
    """The pitch the engine TARGETS per frame: diatonic shift (engine's own maths)
    of the rounded input note + octave*12, in C major."""
    out = np.full_like(in_midi, np.nan)
    for i, m in enumerate(in_midi):
        if np.isfinite(m):
            tgt = diatonic_target_midi(int(round(m)), 0, MAJOR_SCALE_PCS, int(interval))
            out[i] = tgt + 12.0 * int(octave)
    return out


def f0_error(in_midi, out_midi, interval, octave):
    """Median/p90 |cents| between measured output F0 and the intended target, with a
    small lag search to absorb output latency. Returns (med, p90, in_tune_pct, voiced_pct, lag_ms)."""
    want = intended_midi(in_midi, interval, octave)
    n = min(len(want), len(out_midi))
    want, meas = want[:n], out_midi[:n]
    out_voiced_pct = 100.0 * np.mean(np.isfinite(meas)) if n else 0.0
    best = None
    # the engine output only ever LAGS the input (~25ms internal latency); search a
    # physical range [-1, +MAX] frames — a large negative alignment would be spurious.
    for lag in range(-1, MAX_LAG_FRAMES + 1):
        w = want[max(0, -lag): n - max(0, lag)]
        m = meas[max(0, lag): n - max(0, -lag)]
        ok = np.isfinite(w) & np.isfinite(m)
        if ok.sum() < 5:
            continue
        cents = np.abs((m[ok] - w[ok]) * 100.0)
        med = float(np.median(cents))
        if best is None or med < best[0]:
            in_tune = 100.0 * np.mean(cents <= 50.0)
            best = (med, float(np.percentile(cents, 90)), in_tune, lag)
    if best is None:
        return (np.nan, np.nan, 0.0, out_voiced_pct, 0.0)
    med, p90, in_tune, lag = best
    return (med, p90, in_tune, out_voiced_pct, lag * HOP / SR * 1000.0)


def measure(wav_path, in_midi, interval, octave, is_solo):
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR))) if y.size else 0.0
    row = {"dur_s": round(len(y) / SR, 2), "rms": round(rms, 5), "peak": round(peak, 4),
           "centroid_hz": round(centroid, 1),
           "f0_err_med_cents": "", "f0_err_p90_cents": "", "in_tune_pct": "",
           "voiced_pct": "", "lag_ms": ""}
    if is_solo and in_midi is not None:
        out_midi = track_f0(wav_path)
        med, p90, in_tune, voiced, lag = f0_error(in_midi, out_midi, interval, octave)
        row.update({"f0_err_med_cents": round(med, 1), "f0_err_p90_cents": round(p90, 1),
                    "in_tune_pct": round(in_tune, 1), "voiced_pct": round(voiced, 1),
                    "lag_ms": round(lag, 1)})
    return row, rms, peak


def main():
    for d in (RENDERS, MATCHED, BLIND):
        d.mkdir(parents=True, exist_ok=True)
    ckpts = [(lbl, p) for lbl, p in CHECKPOINTS if p.exists()]
    takes = [(lbl, p) for lbl, p in TAKES if p.exists()]
    for lbl, p in CHECKPOINTS:
        if not p.exists():
            print(f"[skip] checkpoint {lbl}: missing {p}")
    for lbl, p in TAKES:
        if not p.exists():
            print(f"[skip] take {lbl}: missing {p}")
    if not ckpts or not takes:
        print("Nothing to do — need at least one checkpoint and one take present.")
        return
    voices = active_voices()
    print(f"Checkpoints: {[c[0] for c in ckpts]}  Takes: {[t[0] for t in takes]}  "
          f"Active voices: {[v['part'] for v in voices]}  Key: {KEY} {'minor' if MINOR else 'major'}")

    # melody source per take (once) + its input-F0 track (reused across checkpoints)
    in_midi = {}
    for tlbl, stems in takes:
        mel = RENDERS / f"{tlbl}__melody.wav"
        extract_melody(stems, mel)
        in_midi[tlbl] = track_f0(mel)

    rows, files = [], []   # files: (raw_path, rms, peak, ckpt, take, stem)
    for clbl, model in ckpts:
        for tlbl, _ in takes:
            mel = RENDERS / f"{tlbl}__melody.wav"
            print(f"[render] {clbl} × {tlbl}")
            jobs = [(v["part"].lower(), [dict(x, on=(x["part"] == v["part"])) for x in SATB_VOICES],
                     True, v["interval"], v["octave"]) for v in voices]
            jobs.append(("mix", SATB_VOICES, False, None, None))
            for stem, vv, no_you, interval, octave in jobs:
                out = RENDERS / f"{tlbl}__{clbl}__{stem}.wav"
                if not render_one(model, mel, out, vv, no_you):
                    continue
                row, rms, peak = measure(out, in_midi[tlbl], interval, octave, is_solo=(stem != "mix"))
                row = {"ckpt": clbl, "take": tlbl, "stem": stem, **row}
                rows.append(row)
                files.append((out, rms, peak, clbl, tlbl, stem))

    if not files:
        print("No renders produced.")
        return

    # ---- equal-RMS level-match (no clip): T <= NORM_CEIL * min(rms/peak) over all files
    valid = [(r, pk) for _, r, pk, *_ in files if r > 1e-5 and pk > 0]
    target_rms = NORM_CEIL * min(r / pk for r, pk in valid) if valid else 0.0
    for raw, rms, peak, clbl, tlbl, stem in files:
        y, sr = sf.read(str(raw), dtype="float32")
        g = (target_rms / rms) if rms > 1e-5 else 1.0
        sf.write(str(MATCHED / raw.name), y * g, sr)
    print(f"[level-match] equal RMS → {target_rms:.4f} (peak ceiling {NORM_CEIL}); {len(files)} files")

    # ---- blind set: coded copies + a key opened only after scoring
    coded = list(files)
    random.Random(BLIND_SEED).shuffle(coded)
    key_rows = []
    for i, (raw, *_rest, clbl, tlbl, stem) in enumerate(coded):
        code = f"X{i:03d}"
        shutil.copyfile(str(MATCHED / raw.name), str(BLIND / f"{code}.wav"))
        key_rows.append({"code": code, "take": tlbl, "ckpt": clbl, "stem": stem})
    key_path = OUT / "blind_key__DO_NOT_OPEN_until_scored.csv"
    with open(key_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "take", "ckpt", "stem"])
        w.writeheader(); w.writerows(key_rows)

    # ---- metrics out (CSV + markdown)
    cols = ["ckpt", "take", "stem", "dur_s", "rms", "peak", "centroid_hz",
            "f0_err_med_cents", "f0_err_p90_cents", "in_tune_pct", "voiced_pct", "lag_ms"]
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    with open(OUT / "metrics.md", "w") as f:
        f.write("# Tenor step-ablation metrics\n\n")
        f.write("F0 error = |cents| of measured output F0 vs the engine's intended diatonic "
                "target (librosa.pyin; lag-aligned). RMS/centroid on the RAW renders "
                "(natural level). Mix rows have no F0 (chord). Lower err / higher in-tune% = better.\n\n")
        f.write("> Caveat: the placeholder **Bass** sits ~2 octaves down (≈32–65 Hz), where pyin "
                "octave-errors badly — treat Bass F0 columns as LOW CONFIDENCE and judge bass by "
                "blind listening + centroid/RMS, not F0. Tenor F0 (in pyin's reliable band) is the "
                "trustworthy pitch metric.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

    print(f"\nDone.\n  renders : {RENDERS}\n  matched : {MATCHED}\n  blind   : {BLIND}\n"
          f"  key     : {key_path}\n  metrics : {OUT/'metrics.md'} (+ .csv)")


if __name__ == "__main__":
    main()
