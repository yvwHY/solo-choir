"""Naive real-time pitch shifter — the baseline harmony voice for the live A/B.

RQ1 asks whether the harmony has to sound like a *human voice*: Beatrice re-synthesises the
harmony line as a voice, this shifter just moves the singer's own recording in pitch. Everything
else in the chain stays identical (same f0, same diatonic decision, same latency), so the only
variable a listener hears is the timbre.

Ported from the author's own browser instrument **Harmonizing** (`yvwHY/Harmonizing`, 2026-05),
which uses Tone.js `PitchShift({windowSize: 0.1, delayTime: 0, feedback: 0, wet: 1})`. That is a
delay-line granular shifter: two read taps half a window apart, each ramping through the window
and wrapping, mixed with an equal-power crossfade so the wrap discontinuity is masked. The same
parameters are used here on purpose — the baseline must be the tool that was actually built, not
an ad-hoc shifter written for the comparison.
"""
import numpy as np


class GranularShifter:
    """Streaming 2-tap delay-line pitch shifter (Tone.PitchShift equivalent).

    process(x, semitones) -> same-length output. `semitones` may change every hop; the phase is
    continuous across calls so there is no click when the harmony moves.
    """

    def __init__(self, sr: int, window_s: float = 0.1):
        self.sr = int(sr)
        self.W = max(2, int(round(window_s * sr)))     # grain/window length in samples
        self._hist = np.zeros(2 * self.W, dtype=np.float32)   # >= W of past + this hop
        self._phase = 0.0                              # read-pointer drift within [0, W)

    def reset(self) -> None:
        self._hist[:] = 0.0
        self._phase = 0.0

    def process(self, x: np.ndarray, semitones: float) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = len(x)
        if n == 0:
            return x
        W = self.W
        hist = np.concatenate([self._hist, x])          # [... past ..., this hop]
        base = len(hist) - n                            # index of x[0] inside hist

        ratio = float(2.0 ** (semitones / 12.0))        # read speed; 1.0 = no shift
        # drift of the read pointer relative to the write pointer, one value per output sample
        p = (self._phase + np.arange(n, dtype=np.float64) * (ratio - 1.0)) % W
        self._phase = float((self._phase + n * (ratio - 1.0)) % W)

        idx = base + np.arange(n, dtype=np.float64) - W   # read a whole window in the past
        outs = []
        for tap in (0.0, 0.5):                            # two taps, half a window apart
            pt = (p + tap * W) % W
            r = idx + pt
            i0 = np.floor(r).astype(np.int64)
            frac = (r - i0).astype(np.float32)
            i0 = np.clip(i0, 0, len(hist) - 2)
            y = hist[i0] * (1.0 - frac) + hist[i0 + 1] * frac      # linear interpolation
            w = np.sin(np.pi * pt / W).astype(np.float32)           # equal-power crossfade:
            outs.append(y * w)                                      # tap weights are sin / cos
        out = (outs[0] + outs[1]).astype(np.float32)

        self._hist = hist[-2 * W:] if len(hist) >= 2 * W else np.concatenate(
            [np.zeros(2 * W - len(hist), dtype=np.float32), hist])
        return out
