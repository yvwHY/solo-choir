"""CREPE ear: torchcrepe-tiny streaming pitch tracker (upgrade over YIN).

Worklog 07-22 §H: the self-written YIN ear heard only 45-68% of real voiced
frames live -- the invisible cause of every stuttering mouth. torchcrepe tiny
is the agreed upgrade, the shared prerequisite for designs 2 (sustain-trigger)
and 3 (DDSP-SVC). Same push()/latest contract as pitch.PitchTracker; pitch.py
itself stays untouched (it is validated code shared with solo_host).

A/B self-test vs YIN over a real-voice stem (offline harvest = reference):
    /opt/anaconda3/envs/vcclient-dev/bin/python ear.py \
        --file ../../../260722_harmony_brain/data/take.wav
"""

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torchcrepe

from pitch import SR, RMS_GATE, PitchTracker, hz_to_midi

CREPE_SR = 16000
WIN = 1024                                  # torchcrepe window: 64 ms @ 16 kHz
CTX16 = 2048                                # analysis context @ 16 kHz
CTX44 = int(round(CTX16 * SR / CREPE_SR))   # same context @ 44.1 kHz (5644)
FMIN, FMAX = 80.0, 1000.0
CONF_MIN = 0.55                             # periodicity below this = unvoiced


class CrepeTracker:
    """Streaming wrapper: push arbitrary chunks, poll latest (midi, hz).
    Smoothing block copied verbatim from pitch.PitchTracker.push -- kept as
    duplication on purpose so the validated YIN path is never edited."""

    def __init__(self, sr=SR, hop=1024, median=5, device="cpu"):
        self.sr = sr
        self.hop = hop
        self.median = median
        self.device = device
        self.buf = np.zeros(0, dtype=np.float64)
        self.recent = []
        self.latest = None
        self.latest_hz = None
        self.last_voiced = None
        self.register = []
        self.rejected = []
        self.push_times = []                # per-hop seconds, for the RTF check
        self._f0(np.ones(CTX44) * 0.01)     # warm-up: first live hop not slow

    def _f0(self, win44):
        """One (f0 Hz | None) estimate from the trailing CTX44 samples."""
        if np.sqrt(np.mean(win44 * win44)) < RMS_GATE:
            return None
        x16 = np.interp(np.linspace(0, len(win44) - 1, CTX16),
                        np.arange(len(win44)), win44)
        audio = torch.tensor(x16, dtype=torch.float32, device=self.device)[None]
        f0, pd = torchcrepe.predict(
            audio, CREPE_SR, hop_length=WIN, fmin=FMIN, fmax=FMAX,
            model="tiny", return_periodicity=True, batch_size=8,
            device=self.device)
        mid = f0.shape[1] // 2              # centre frame: fully in real data
        return float(f0[0, mid]) if float(pd[0, mid]) >= CONF_MIN else None

    def push(self, samples):
        self.buf = np.concatenate([self.buf, np.asarray(samples, dtype=np.float64)])
        while len(self.buf) >= CTX44:
            t0 = time.perf_counter()
            f0 = self._f0(self.buf[:CTX44])
            self.buf = self.buf[self.hop:]
            # ---- below: verbatim smoothing block from pitch.PitchTracker ----
            self.recent.append(f0)
            self.recent = self.recent[-self.median:]
            voiced = [f for f in self.recent if f is not None]
            if len(voiced) >= (self.median + 1) // 2:
                m = int(round(hz_to_midi(float(np.median(voiced)))))
                if self.last_voiced is not None:
                    d = m - self.last_voiced
                    if 11 <= abs(d) <= 13:
                        m -= 12 * int(np.sign(d))
                if len(self.register) >= 8 and m < np.median(self.register) - 9:
                    self.rejected.append(m)
                    if len(self.rejected) >= 3 and np.ptp(self.rejected[-3:]) <= 2:
                        self.register = self.rejected[-3:]
                    else:
                        self.latest = None
                        self.push_times.append(time.perf_counter() - t0)
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
            self.push_times.append(time.perf_counter() - t0)
        return self.latest


def load_mono(path):
    """wav -> mono float64 @ SR (stereo takes: channel L = the mic stem)."""
    with wave.open(str(path)) as w:
        nch, sr_in = w.getnchannels(), w.getframerate()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    x = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)
    if sr_in != SR:
        x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * SR / sr_in)),
                      np.arange(len(x)), x)
    return x


def main():
    import pyworld
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    a = ap.parse_args()
    x = load_mono(a.file)
    f0r, _ = pyworld.harvest(x, SR, frame_period=5.0)   # offline reference ear

    blk = 1024
    yin, crepe = PitchTracker(), CrepeTracker(device=a.device)
    stats = {"ref_v": 0, "yin_hit": 0, "crepe_hit": 0,
             "ref_u": 0, "yin_false": 0, "crepe_false": 0,
             "cents": [], "oct": 0, "cmp_n": 0}
    for i in range(0, len(x) - blk, blk):
        b = x[i:i + blk]
        yin.push(b)
        crepe.push(b)
        # reference over the trailing 100 ms (matches tracker smoothing lag)
        j1 = int(i / SR * 200) + 1
        seg = f0r[max(0, j1 - 20):j1]
        v = seg[seg > 0]
        if len(seg) == 0:
            continue
        if len(v) >= len(seg) / 2:
            ref_m = hz_to_midi(float(np.median(v)))
            stats["ref_v"] += 1
            stats["yin_hit"] += yin.latest is not None
            stats["crepe_hit"] += crepe.latest is not None
            if crepe.latest is not None:
                stats["cmp_n"] += 1
                d = abs(crepe.latest - ref_m)
                stats["oct"] += 11 <= d <= 13
                stats["cents"].append(100 * min(d, abs(d - 12), abs(d - 24)))
        else:
            stats["ref_u"] += 1
            stats["yin_false"] += yin.latest is not None
            stats["crepe_false"] += crepe.latest is not None

    yh = 100 * stats["yin_hit"] / max(1, stats["ref_v"])
    ch = 100 * stats["crepe_hit"] / max(1, stats["ref_v"])
    print(f"voiced blocks (harvest ref): {stats['ref_v']}")
    print(f"heard%   YIN {yh:5.1f}   CREPE {ch:5.1f}   (target: CREPE >= YIN+15)")
    print(f"false-voiced%   YIN {100*stats['yin_false']/max(1,stats['ref_u']):5.1f}"
          f"   CREPE {100*stats['crepe_false']/max(1,stats['ref_u']):5.1f}")
    print(f"CREPE octave-error% {100*stats['oct']/max(1,stats['cmp_n']):4.1f}"
          f"   median |cents| {np.median(stats['cents']):5.1f}")
    pt = np.array(crepe.push_times) * 1000
    print(f"CREPE per-hop ms: mean {pt.mean():.2f}  p99 {np.percentile(pt,99):.2f}"
          f"   (hop is {1000*1024/SR:.1f} ms -- must stay well under)")


if __name__ == "__main__":
    main()
