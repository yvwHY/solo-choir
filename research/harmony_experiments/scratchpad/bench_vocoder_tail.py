"""Margin sweep for making the vocoder tail-only: vocode just the tail plus m frames
of margin, compare it against the same stretch from a full-window vocode, and find
the m at which the maximum absolute difference reaches zero.

Why ask: the window is 0.71 s, but SOLA only takes the tail, blk + cf +
sola_search = 0.29 s, so **59% of the synthesis work is thrown away** - and that 59%
falls on NSF-HiFiGAN, the most expensive thing in the system (55% of a block). v23's
`--enh-tail` proved the same principle on the CombSub line, worth 35 ms a block, but
when it was ported it was tied to the name "enhancer" and discarded with it
(spike_stream6.py:275, "enh_tail is accepted and ignored") - reflow has no enhancer,
but it does have a vocoder.

NSF-HiFiGAN is a convolutional upsampler and therefore has a finite receptive field.
If m exceeds that field, vocoding the tail alone must be BIT IDENTICAL to the tail of
the full window - which makes it Regime A, needing no listening test.

Run: 260724_ddsp_svc_6x/venv/bin/python scratchpad/bench_vocoder_tail.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spike_stream6 as S  # noqa: E402

SR, HOP = S.SR, 512
BLK = int(round(0.24 * SR / HOP)) * HOP
CF, SOLA_SEARCH, LAST_DELAY = int(0.04 * SR), int(0.01 * SR), int(0.02 * SR)
WIN = ((max(int(0.7 * SR), BLK + CF + SOLA_SEARCH + 2 * LAST_DELAY)
        // HOP + 1) * HOP)
USED = BLK + CF + SOLA_SEARCH + LAST_DELAY      # the tail SOLA actually touches
USED_F = -(-USED // HOP)                        # in frames, rounded up
MARGINS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


voices = [(f"{S.DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.85),
          (f"{S.DDSP}/exp/reflow-harry-run1/model_14000.pt", -3.0, 0.9),
          (f"{S.DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
svc = S.Svc(voices, step=2, t_start=0.85)
model = svc.voices[0][0]

x, sr = sf.read("scratchpad/_ab30.wav", dtype="float32", always_2d=True)
assert sr == SR
x = x[:, 0]

print(f"window {WIN} samples = {WIN/SR:.2f}s = {WIN//HOP} frames")
print(f"tail used by SOLA {USED} samples = {USED/SR:.2f}s = {USED_F} frames"
      f" ({100*USED/WIN:.0f}% of the window)\n")

# take 5 windows where he is singing, avoiding silence
starts = []
for st in range(0, len(x) - WIN, WIN):
    if float(np.abs(x[st:st + WIN]).mean()) > 0.01:
        starts.append(st)
    if len(starts) == 5:
        break
print(f"sampled {len(starts)} voiced windows\n")

torch.manual_seed(0)
worst = {m: 0.0 for m in MARGINS}
with torch.no_grad():
    for st in starts:
        xb = x[st:st + WIN].astype("float64")
        f0_np, vol_t, mask = svc.prep(xb, -60.0)
        f0 = torch.from_numpy(f0_np).float().to(svc.device)[None, :, None]
        units = svc.encode(xb)
        n = min(units.size(1), f0.size(1), vol_t.size(1))
        fv = f0[:, :n] * 2 ** (12.0 / 12.0)

        wav, _ = model.ddsp_model(units[:, :n], fv, vol_t[:, :n],
                                  spk_id=svc.spk, infer=True)
        dm = svc.vocoder.extract(wav)
        # fixed seed, so the mel is reproducible (reflow calls randn; a lesson learned)
        torch.manual_seed(1234)
        mel = model.reflow_model(dm, gt_spec=dm, infer=True, infer_step=2,
                                 method="euler", t_start=0.85, use_tqdm=False)
        f0m = fv[:, -mel.shape[1]:]
        full = svc.vocoder.infer(mel, f0m).reshape(-1)

        for m in MARGINS:
            k = min(USED_F + m, mel.shape[1])
            tail = svc.vocoder.infer(mel[:, -k:], f0m[:, -k:]).reshape(-1)
            d = float((tail[-USED:] - full[-USED:]).abs().max())
            worst[m] = max(worst[m], d)

print(f"{'margin m (frames)':<20}{'m in seconds':>14}{'max-abs-diff':>16}")
for m in MARGINS:
    print(f"{m:<14}{m*HOP/SR:>10.3f}{worst[m]:>16.3e}"
          + ("   <- bit identical" if worst[m] == 0.0 else ""))

# how much it saves: timed on the first window
zero = [m for m in MARGINS if worst[m] == 0.0]
pick = zero[0] if zero else MARGINS[-1]
k = USED_F + pick
xb = x[starts[0]:starts[0] + WIN].astype("float64")
f0_np, vol_t, mask = svc.prep(xb, -60.0)
f0 = torch.from_numpy(f0_np).float().to(svc.device)[None, :, None]
units = svc.encode(xb)
n = min(units.size(1), f0.size(1), vol_t.size(1))
fv = f0[:, :n] * 2 ** (12.0 / 12.0)
with torch.no_grad():
    wav, _ = model.ddsp_model(units[:, :n], fv, vol_t[:, :n],
                              spk_id=svc.spk, infer=True)
    dm = svc.vocoder.extract(wav)
    mel = model.reflow_model(dm, gt_spec=dm, infer=True, infer_step=2,
                             method="euler", t_start=0.85, use_tqdm=False)
    f0m = fv[:, -mel.shape[1]:]

    def bench(fn, rep=15):
        fn(); sync()
        ts = []
        for _ in range(rep):
            sync(); t = time.perf_counter(); fn(); sync()
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts))

    t_full = bench(lambda: svc.vocoder.infer(mel, f0m))
    t_tail = bench(lambda: svc.vocoder.infer(mel[:, -k:], f0m[:, -k:]))
print(f"\nmargin {pick} frames ({pick*HOP/SR:.3f}s) -> vocode {k}/{mel.shape[1]} frames")
print(f"one voice, vocoder: full window {t_full:.1f}ms -> tail only {t_tail:.1f}ms"
      f" (saving {t_full-t_tail:.1f}ms, {100*(t_full-t_tail)/t_full:.0f}%)")
print(f"three voices save about {3*(t_full-t_tail):.0f}ms per block"
      f" (note: the bench calls mps.synchronize, so absolutes read high; only the ratio matters)")
