"""key_probe.py - find the key of a real take from the raw f0 pitch-class distribution.

Why this exists: `--key` has to be a manual integer, and manual does not mean
guessed. A whole evening was spent calling one take C major and being wrong, and
the root cause was a blind instrument: the model's representation SNAPS notes into
the C scale, so "the lead is 100% within C" is circular. Looking at the UNSNAPPED
raw f0 instead, the pitch-class distribution pointed at the A and E family
immediately, and A major was settled. That analysis was one-off and never went into
the repository, which is the same instrument lesson as live_probe; this is it
written down.

Two criteria, the two actually used:
  1. **Coverage**: how much of the sounding time the seven pitch classes of each
     major key account for, weighted by voiced frames.
  2. **Final notes**: the pitch-class distribution of phrase endings. A sense of
     cadence is harder evidence than a statistic.
If the two disagree, choose a different song: it means the take is dirty in any
single major key, which is typical of free improvisation (one take covered only
about 70%).

The shift printed is exactly what `--key` wants, following the same convention as
EarV3._keylock: shift = (0 - root) % 12, minus 12 above 6, so the parts' score
lands back in C major coordinates.

Usage (DDSP venv, from harmony/):
    python key_probe.py <take.wav>
    python key_probe.py <take.wav> --hop 512     # denser, and slower
"""
import argparse

import numpy as np
import soundfile as sf

from pitch import FRAME, SR, hz_to_midi, yin_f0
from respond2 import find_phrases

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]          # major scale degrees, relative to the tonic


def raw_pitch_classes(x, hop):
    """Unsnapped f0 to (sample position, pitch class) per frame. Deliberately avoids
    VoiceToTokens and every representation inside the model: those snap notes to the
    scale, and what comes out is then only your own assumption."""
    out = []
    for i in range(0, len(x) - FRAME, hop):
        f = yin_f0(x[i:i + FRAME])
        if f is not None:
            out.append((i, int(round(hz_to_midi(f))) % 12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--tail", type=float, default=0.30,
                    help="how many seconds at the end of a phrase count as the final note")
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--gate", type=float, default=0.02)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    assert sr == SR, (a.wav, sr)

    pcs = raw_pitch_classes(x, a.hop)
    if not pcs:
        raise SystemExit("no voiced frames (silent file? wrong sample rate?)")
    hist = np.bincount([p for _, p in pcs], minlength=12).astype(float)
    hist /= hist.sum()
    print(f"{a.wav} | {len(x)/SR:.1f}s, {len(pcs)} voiced frames")
    print("raw f0 pitch-class distribution (unsnapped):")
    for i in np.argsort(hist)[::-1][:6]:
        print(f"  {NAMES[i]:2s} {hist[i]*100:5.1f}%")

    import respond2
    respond2.RMS_GATE = a.gate
    ph = find_phrases(x, a.gap, 0.5)
    ends = []
    for s, e in ph:
        seg = [p for i, p in pcs if e - a.tail * SR <= i < e]
        if seg:
            ends.append(np.bincount(seg, minlength=12).argmax())
    eh = np.bincount(ends, minlength=12).astype(float) if ends else np.zeros(12)

    print(f"\n{len(ph)} phrases; pitch class of the final {a.tail}s:")
    for i in np.argsort(eh)[::-1][:4]:
        if eh[i]:
            print(f"  {NAMES[i]:2s} {int(eh[i])} phrases")

    print("\ncandidate major keys (coverage = share of sounding time in the key's seven classes):")
    rows = []
    for root in range(12):
        cov = sum(hist[(root + d) % 12] for d in MAJOR)
        fin = sum(eh[(root + d) % 12] for d in MAJOR) / max(1, len(ends))
        shift = (0 - root) % 12
        shift = shift - 12 if shift > 6 else shift
        rows.append((cov, root, shift, fin, eh[root]))
    rows.sort(reverse=True)
    for cov, root, shift, fin, tonic_ends in rows[:4]:
        print(f"  {NAMES[root]:2s} major  coverage {cov*100:5.1f}%  "
              f"endings in key {fin*100:5.0f}%  ending on the tonic {int(tonic_ends)} phrases  "
              f"→ --key {shift:+d}")

    cov, root, shift, _, _ = rows[0]
    print(f"\nverdict: **{NAMES[root]} major -> --key {shift:+d}**")
    if cov < 0.85:
        print(f"coverage is only {cov*100:.0f}%: this take is dirty within any single "
              f"major key, which is the level that runs a pipeline but cannot judge quality.")
    if rows[0][0] - rows[1][0] < 0.03:
        print(f"first and second differ by only {(rows[0][0]-rows[1][0])*100:.1f}%, so they "
              f"cannot be separated; read the endings column, or pick a tonally cleaner song.")


if __name__ == "__main__":
    main()
