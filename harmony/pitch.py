"""Lightweight YIN pitch tracker for the live path (numpy only).

This is the project's own implementation of the YIN algorithm (de Cheveigne
and Kawahara, 2002): a frame of audio is compared with delayed copies of
itself, and the delay at which it best matches gives the period, hence the
pitch. Designed for singing voice: 2048-sample frames at 44.1 kHz (46 ms),
pitch range 100-1000 Hz.

Two layers: yin_f0() estimates one frame; PitchTracker wraps it for streaming
input and adds the cleaning a live singer needs (median smoothing, octave-jump
correction, rejection of vocal fry and breath noise).

Run directly for a self-test on synthetic tones:  venv/bin/python pitch.py
"""

import numpy as np

SR = 44100
FRAME = 2048
FMIN, FMAX = 100.0, 1000.0  # 100 Hz floor keeps vocal fry out of the band
YIN_THRESHOLD = 0.15
RMS_GATE = 0.005  # below this the frame counts as silence (headset mics run quiet)


def yin_f0(frame, sr=SR):
    """Estimate the pitch of one frame. Returns Hz, or None if silent/unvoiced.

    Steps: remove DC and gate on level; build the YIN difference function
    over the candidate delay range; normalise it (CMNDF) so the first true
    dip is comparable across levels; take the first dip under YIN_THRESHOLD
    and walk to its local minimum; refine the delay with a parabola through
    the three points around it. The frame must be long enough to hold the
    longest candidate period plus 512 samples of overlap.
    """
    x = frame.astype(np.float64)
    x = x - x.mean()
    if np.sqrt(np.mean(x * x)) < RMS_GATE:
        return None
    tau_min = int(sr / FMAX)
    tau_max = int(sr / FMIN)
    w = len(x) - tau_max
    if w < 512:
        raise ValueError("frame too short for FMIN")

    # difference function d[tau], then cumulative-mean-normalized (CMNDF)
    d = np.empty(tau_max + 1)
    d[0] = 0.0
    for tau in range(1, tau_max + 1):
        diff = x[:w] - x[tau : tau + w]
        d[tau] = diff @ diff
    cum = np.cumsum(d[1:])
    cmndf = np.empty(tau_max + 1)
    cmndf[0] = 1.0
    cmndf[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(cum, 1e-12)

    # first dip under threshold (absolute-threshold step of YIN)
    below = np.flatnonzero(cmndf[tau_min:tau_max] < YIN_THRESHOLD)
    if len(below) == 0:
        return None
    tau = tau_min + below[0]
    while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
        tau += 1

    # parabolic interpolation around the minimum
    if 1 <= tau < tau_max:
        a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            tau = tau + 0.5 * (a - c) / denom
    return sr / tau


def hz_to_midi(f):
    """Convert a frequency in Hz to a fractional MIDI note number (A4 = 69)."""
    return 69 + 12 * np.log2(f / 440.0)


class PitchTracker:
    """Streaming wrapper: push arbitrary-size chunks, poll latest (midi, voiced)."""

    def __init__(self, sr=SR, hop=1024, median=5):
        """hop is the step between analysed frames in samples; median is how
        many recent frame estimates are pooled before a note is accepted."""
        self.sr = sr
        self.hop = hop
        self.buf = np.zeros(0, dtype=np.float64)
        self.recent = []  # median filter over last few frames kills octave blips
        self.median = median
        self.latest = None  # rounded MIDI int or None
        self.latest_hz = None  # unrounded, for pitch-shift ratio computation
        self.last_voiced = None
        self.register = []  # recent accepted notes; rejects fry/breath outliers
        self.rejected = []

    def push(self, samples):
        """Feed audio in; return the current note as a MIDI integer, or None.

        Samples are buffered and analysed one frame per hop. A note is only
        reported when at least half of the last `median` frames were voiced,
        and the median of those frames is used, which removes single-frame
        blips. Two further corrections follow: a jump of almost exactly one
        octave from the last note is treated as a tracking error and folded
        back, and a note far below the recent register is held back as fry
        or breath noise unless it persists for three frames, in which case
        the register follows the singer down. self.latest_hz keeps the
        unrounded frequency for pitch-shift ratios.
        """
        self.buf = np.concatenate([self.buf, np.asarray(samples, dtype=np.float64)])
        while len(self.buf) >= FRAME:
            f0 = yin_f0(self.buf[:FRAME], self.sr)
            self.buf = self.buf[self.hop :]
            self.recent.append(f0)
            self.recent = self.recent[-self.median :]
            voiced = [f for f in self.recent if f is not None]
            if len(voiced) >= (self.median + 1) // 2:
                m = int(round(hz_to_midi(float(np.median(voiced)))))
                # near-exact octave jump from the last voiced note is almost
                # always a sub- or super-harmonic tracking error, not a real leap
                if self.last_voiced is not None:
                    d = m - self.last_voiced
                    if 11 <= abs(d) <= 13:
                        m -= 12 * int(np.sign(d))
                # fry/breath rejection: a sudden note far below the recent
                # register is noise -- unless it persists (a real register drop)
                if len(self.register) >= 8 and m < np.median(self.register) - 9:
                    self.rejected.append(m)
                    if len(self.rejected) >= 3 and np.ptp(self.rejected[-3:]) <= 2:
                        self.register = self.rejected[-3:]  # he really went down
                    else:
                        self.latest = None
                        continue
                self.rejected = []
                self.register = (self.register + [m])[-32:]
                self.latest = m
                self.latest_hz = 440.0 * 2 ** ((m - 69) / 12) if abs(
                    hz_to_midi(float(np.median(voiced))) - m) > 1 else float(np.median(voiced))
                self.last_voiced = m
            else:
                self.latest = None
                self.latest_hz = None
        return self.latest


def _selftest():
    """Check yin_f0 on seven synthetic voice-like tones and on near-silence."""
    rng = np.random.default_rng(0)
    print("synthetic-tone self-test (target -> detected):")
    errs = []
    for midi in [48, 55, 60, 64, 67, 72, 79]:
        f = 440 * 2 ** ((midi - 69) / 12)
        t = np.arange(FRAME) / SR
        # saw-ish tone with vibrato + noise, closer to voice than a pure sine
        sig = sum((0.6 / h) * np.sin(2 * np.pi * f * h * t + 0.1 * h) for h in range(1, 5))
        sig += 0.01 * rng.standard_normal(FRAME)
        f0 = yin_f0(sig)
        got = int(round(hz_to_midi(f0))) if f0 else None
        errs.append(got == midi)
        print(f"  MIDI {midi} ({f:6.1f} Hz) -> {got}")
    silent = yin_f0(0.001 * rng.standard_normal(FRAME))
    print(f"  silence -> {silent} (want None)")
    assert all(errs) and silent is None, "SELF-TEST FAILED"
    print("self-test passed")


if __name__ == "__main__":
    _selftest()
