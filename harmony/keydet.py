"""keydet.py — key detection (task 6: leaving C major)

A sliding-window pitch histogram correlated against Krumhansl-Schmuckler major
profiles over 24 keys (12 roots by major and minor profile). This project's
harmony model understands only the major system, so the detector's output is a
major key root.
Uses:
  offline  python keydet.py <wav...>          # key and confidence per file
  module   from keydet import KeyDetector     # live: push(f0_hz) -> key

Design:
- The input is f0 alone, which both YIN and the telemetry provide, and never a
  spectrum, so it is not coupled to the existing detectors.
- A sliding window, 8 s by default as a trustworthy starting point live, plus a
  whole-take accumulator. Confidence is the gap between the best and second-best
  correlation.
- The output is the root in semitones (C = 0), for normalising the input to C
  before the model and rotating the output back.
"""
import math, sys
import numpy as np

# Krumhansl-Kessler major/minor key profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


SCALE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)  # major scale template


def key_from_hist(hist):
    """A 12-bin pitch-class histogram to (root, is_major, confidence).

    The criterion is scale membership, that is, which major scale set holds the
    most histogram mass, not the K-S tonic profile. What the model needs is
    which major scale the melody fits into; C major and A minor are the same set
    of notes, and the argument about which is the tonic does not affect
    normalisation. In the first test, K-S called an A-centred melody in C major
    "A".
    is_major is a compatibility field and is always True. conf is the gap
    between the best and second-best membership, from 0 to 1."""
    tot = hist.sum()
    if tot <= 0:
        return None
    member = np.array([np.dot(np.roll(SCALE, r), hist) for r in range(12)]) / tot
    order = np.argsort(member)[::-1]
    r1, r2 = int(order[0]), int(order[1])
    return r1, True, float(member[r1] - member[r2])


class KeyDetector:
    """Live accumulator: push each frame's f0, in Hz or None, and read key()."""

    def __init__(self, win_frames=None):
        self.hist = np.zeros(12)
        self.win = win_frames          # None = accumulate without limit
        self.buf = []

    def push(self, f0):
        if not f0:
            return
        pc = int(round(69 + 12 * math.log2(f0 / 440.0))) % 12
        self.hist[pc] += 1
        if self.win:
            self.buf.append(pc)
            if len(self.buf) > self.win:
                self.hist[self.buf.pop(0)] -= 1

    def key(self):
        r = key_from_hist(self.hist)
        if r is None:
            return None
        root, is_maj, conf = r
        return {"root": root, "major": is_maj, "conf": conf,
                "name": NAMES[root] + ("" if is_maj else "m")}


def track_file(path):
    import soundfile as sf, soxr
    from pitch import yin_f0
    SR = 44100
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    kd = KeyDetector()
    for t in range(0, len(x) - 2048, 1024):
        kd.push(yin_f0(x[t:t + 2048].astype(np.float32), SR))
    return kd.key()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        k = track_file(p)
        print(f"{p}: {k['name']}  conf={k['conf']:.3f}" if k else f"{p}: no pitch")
