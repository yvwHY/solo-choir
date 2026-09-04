"""reverb.py — convolution reverb from a synthesised impulse response (2026-08-03)

**Why convolution and not an algorithmic reverb**: Schroeder and FDN designs need
a set of mutually prime comb delays tuned by hand, and the failure mode of a bad
tuning is a metallic ring — which is exactly what this project spent three days
killing (the sentinel harmonic column of 2026-07-30 sections F to I). Gaussian
noise under an exponential decay is already a maximum-density diffuse field,
**with no comb peaks by construction**, so there is nothing to tune wrongly.

**What makes it sound like a room rather than an effect is band-split decay**: in
a real space the high frequencies die first, through air absorption and soft
surfaces, and a white-noise tail with a single T60 hisses. Here three bands each
get their own T60.

The three-part structure follows a real impulse response: the direct sound (not
in the IR; it travels as the dry signal), then the predelay silence, then early
reflections (sparse, discrete, and deliberately irregular in spacing, because
regular spacing is flutter echo), then the late diffuse tail (faded in, because
the echo density of a real room **grows** rather than starting full).

No purchases and no new dependency: scipy is already in the venv.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
# The early-reflection window, in ms. Changed on 2026-08-04 to a **diffuse
# cluster**. It used to be 10 discrete taps between 11 and 84 ms at a gain of
# 0.45, which reads as space on a sustained note but as an audible slap echo on
# **the consonant of an opening**; worn, it was heard as "the opening repeated",
# and it vanished at --reverb 0, which convicted it. Discrete repeats can be
# picked out tap by tap, so they were replaced by dense random micro-taps over
# the same window under an exponential envelope: no single reflection is
# identifiable, while the room's near-wall cue remains.
ER_SPAN_MS = (8.0, 90.0)


def _trim(ir, trim_db, fade_ms=60.0, sr=SR):
    """Truncate the tail once it falls below -trim_db, with a fade before the
    cut so the join does not click.

    Why truncate: in the answering mode **the length of the response is exactly
    how long the singer cannot sound**, since the hard mute covers the whole
    playback. The tail is therefore not free decoration but a cost paid straight
    out of the interaction. On 2026-08-03, T60 at 1.8, 2.2, 2.5 and 2.8 all
    sounded "about the same" by ear, so tail length is not an audible control in
    this context and there is no reason to keep paying for it. The threshold uses
    the measured envelope rather than the theoretical value, because the three
    bands have their own T60 and the low band decays slowest."""
    w = int(0.02 * sr)
    env = np.array([np.sqrt(np.mean(ir[i:i + w] ** 2))
                    for i in range(0, len(ir) - w, w)])
    # Search only after the peak: the IR begins with the predelay silence, and
    # searching from the start would truncate at 20 ms.
    pk = int(np.argmax(env))
    thr = env[pk] * 10 ** (-trim_db / 20.0)
    below = np.flatnonzero(env[pk:] < thr)
    if not len(below):
        return ir
    end = min(len(ir), (pk + below[0] + 1) * w)
    f = max(1, int(fade_ms / 1000.0 * sr))
    out = ir[:end].copy()
    if end > f:
        out[end - f:] *= np.cos(np.linspace(0, np.pi / 2, f)) ** 2
    return out


def make_ir(t60=1.8, predelay_ms=20.0, hf_damp=0.40, lf_boost=1.20,
            er_gain=0.45, trim_db=0.0, seed=20260803, sr=SR):
    """Return a mono IR, energy-normalised so that rms after convolution is
    about the rms of the input.

    t60 is the seconds for the mid band to fall 60 dB (1.2 a small room, 1.8 an
    ordinary hall, 2.8 a church). hf_damp and lf_boost are the T60 multipliers of
    the high and low bands against the mid. It is deterministic, from a fixed
    seed, so the same parameters are always the same space, which is what makes
    an A/B comparison clean."""
    rng = np.random.default_rng(seed)
    n = int(t60 * 1.3 * sr)                      # keep the tail down to -78 dB before truncating
    t = np.arange(n) / sr
    lo = butter(2, 500, "lowpass", fs=sr, output="sos")
    mid = butter(2, (500, 4000), "bandpass", fs=sr, output="sos")
    hi = butter(2, 4000, "highpass", fs=sr, output="sos")
    noise = rng.standard_normal(n)
    tail = np.zeros(n)
    for sos, mult in ((lo, lf_boost), (mid, 1.0), (hi, hf_damp)):
        # each band has its own T60: -60 dB is 10^-3, so the exponent is 6.908/T60
        tail += sosfilt(sos, noise) * np.exp(-6.908 * t / (t60 * mult))
    # Echo density grows: the first 30 ms of the tail fades in, or the IR opens
    # with a gunshot.
    ramp = np.clip(t / 0.030, 0.0, 1.0)
    tail *= ramp
    tail /= np.sqrt(np.mean(tail ** 2)) * np.sqrt(n)   # normalise the tail energy first

    ir = np.zeros(int(predelay_ms / 1000.0 * sr) + n)
    pre = int(predelay_ms / 1000.0 * sr)
    ir[pre:pre + n] = tail
    a0, a1 = (int(v / 1000.0 * sr) for v in ER_SPAN_MS)
    er = rng.standard_normal(a1 - a0) * np.exp(-np.arange(a1 - a0) / (0.022 * sr))
    er *= er_gain / np.sqrt(np.sum(er ** 2))           # total cluster energy is er_gain squared
    ir[pre + a0:pre + a1] += er
    if trim_db > 0:
        ir = _trim(ir, trim_db, sr=sr)
    return ir / np.sqrt(np.sum(ir ** 2))               # energy normalisation


def wet(x, ir):
    """Return the wet part only, of length len(x) + len(ir) - 1. The dry signal
    is not in it, so the caller decides how much to send; at wet 0 the whole path
    is skipped and the result is byte-identical to having no reverb."""
    return fftconvolve(x, ir)


def _demo():
    """Listen to the IR itself as an impulse test and print the decay every
    100 ms, to confirm there are no comb peaks or other anomalies."""
    import sys
    import soundfile as sf
    t60 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.8
    ir = make_ir(t60)
    print(f"IR {len(ir)/SR:.2f}s  peak {np.abs(ir).max():.4f}  "
          f"energy {np.sum(ir**2):.4f}")
    e = [20 * np.log10(np.sqrt(np.mean(ir[i:i + SR // 10] ** 2)) + 1e-12)
         for i in range(0, len(ir) - SR // 10, SR // 2)]
    print("rms dB every 0.5 s:", " ".join(f"{v:.0f}" for v in e))
    sf.write(f"out/ir_t60_{t60}.wav", ir / np.abs(ir).max() * 0.9, SR,
             subtype="PCM_16")
    print(f"wrote out/ir_t60_{t60}.wav")


if __name__ == "__main__":
    _demo()


class ConvLive:
    """Live convolution reverb by uniformly partitioned overlap-save, for
    performance.

    The offline `wet()` is a single fftconvolve over the whole signal and cannot
    be used live, since it would wait for the singing to finish. Here the IR is
    cut into partitions of length B, and each callback does one FFT, P complex
    multiply-accumulates in the frequency domain, and one IFFT.
    **Output block k depends only on input blocks up to k, so it is causal and
    adds no latency** beyond the callback's own block. That matters, because any
    delay on the wet signal is heard as a longer predelay, and the predelay is
    already 20 ms. What must not be delayed is the dry signal, and the dry signal
    never enters the computer at all: it travels a direct path while the computer
    outputs wet only, which keeps delayed auditory feedback and feedback loops
    out.

    Cost with B = 1024 and an IR of 0.88 s, so P = 38: 38 complex
    multiply-accumulates of length 1025 per block, which is negligible.
    """

    def __init__(self, ir, block=1024):
        self.B = int(block)
        self.N = 2 * self.B
        P = int(np.ceil(len(ir) / self.B))
        pad = np.zeros(P * self.B)
        pad[:len(ir)] = ir
        self.H = np.stack([np.fft.rfft(
            np.concatenate([pad[p * self.B:(p + 1) * self.B],
                            np.zeros(self.B)]), self.N) for p in range(P)])
        self.X = np.zeros((P, self.N // 2 + 1), dtype=complex)   # frequency-domain history
        self.prev = np.zeros(self.B)
        self.P = P

    def __call__(self, x):
        """One input block of length B to one wet block of length B."""
        if len(x) != self.B:                      # the final block may be short
            y = np.zeros(self.B)
            y[:len(x)] = x
            x = y
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = np.fft.rfft(np.concatenate([self.prev, x]), self.N)
        self.prev = x
        acc = np.sum(self.H * self.X, axis=0)
        return np.fft.irfft(acc, self.N)[self.B:]   # overlap-save: take the second half
