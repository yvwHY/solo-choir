"""vs_male3_trial.py - listening samples from a trial model trained on VocalSet male3.

**The material was withdrawn.** This uses the same `[0::2]` selection as
prosody_ab, which picked the RESPONSE segments out of the dump, that is a
three-part mix. Its output is void and the male3 verdict has to be re-measured on
clean material. Do not run it.

It extracts his sung phrases from a session dump and renders: male3 singing his
line (a timbre check), male3 on the lower part line (its real position), the same
line on the existing voice for comparison, and a dry-plus-male3 mix. The extraction
and the render recipe are copied from angel_audit.py so the measure is the same.
The verdict is by ear.

Run (DDSP venv):  python vs_male3_trial.py
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

H = Path(__file__).parent
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402
from live_v3 import EarV3  # noqa: E402

DUMP = H / "out/resp2_live_260804_160058.wav"
M3 = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
      "260724_ddsp_svc/exp/combsub-vs-male3/model_14000.pt")
OUT = H / "out" / "vs_male3_trial"
OUT.mkdir(exist_ok=True)
per = TICK_SAMPS / dm.HOP

x, _ = sf.read(str(DUMP), dtype="float64", always_2d=True)
x = np.ascontiguousarray(x[:, 0])
R.RMS_GATE = 0.02
segs = []
for s, e in R.find_phrases(x, 0.35, 0.8):
    if s / SR < 12:
        continue
    f0 = dm.harvest_f0(np.ascontiguousarray(x[s:e]))
    v = f0[f0 > 0]
    if len(v) < 30:
        continue
    if float(np.mean(v > 340.0)) < 0.05:
        segs.append(np.ascontiguousarray(x[s:e]))
segs = segs[0::2]
segs = sorted(segs, key=len, reverse=True)[:2]     # the two longest phrases, which have the most content
print(f"material: {len(segs)} phrases, {', '.join(f'{len(s)/SR:.1f}s' for s in segs)}")

torch.manual_seed(1234)
ear = EarV3(indep=0.15, stab=1, key=0)
ear.v2t.shift = 12
kt = R.KeyTracker()
rend = {
    "m3": PhraseRenderer(R.DDSP, M3, expr_seed=20260804),
    "girl": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                           expr_seed=R.VOICES["upper"][1]),
}
for r in rend.values():
    r.render(np.zeros(SR), [60], TICK_SAMPS)

def true_line(seg, n):
    f0 = dm.harvest_f0(seg)
    out = []
    for k in range(n):
        w = f0[int(k * per):int((k + 1) * per)]
        w = w[w > 0]
        out.append(None if len(w) < 3 else
                   int(round(float(np.median(69 + 12 * np.log2(w / 440.0))))))
    return out

def norm(y):
    return y / (np.abs(y).max() + 1e-12) * 0.7

kw = dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0, uv_dry=0.0,
          vib_hz=R.VIB["lower"][0], vib_phase=R.VIB["lower"][1])
for i, seg in enumerate(segs):
    kt.push(dm.harvest_f0(seg))
    r = kt.best()
    if r is not None and r[2] >= 0.015:
        ear.k_shift = r[0]
    lead, up, lo = R.phrase_notes(ear, seg)
    ident = true_line(seg, len(lead))
    for tag, notes, mouth in (("identity", ident, "m3"),
                              ("lower", lo, "m3"),
                              ("lower_girl", lo, "girl")):
        sh = {}
        y = rend[mouth].render(seg, notes, TICK_SAMPS, shared=sh, **kw)
        n = min(len(seg), len(y))
        sf.write(str(OUT / f"seg{i}_{tag}.wav"), norm(y[:n]), SR, subtype="PCM_16")
        if tag == "lower":
            mix = 0.944 * seg[:n] + 0.75 * y[:n]
            sf.write(str(OUT / f"seg{i}_mix_dry+m3.wav"), norm(mix), SR,
                     subtype="PCM_16")
    sf.write(str(OUT / f"seg{i}_dry.wav"), norm(seg), SR, subtype="PCM_16")
    print(f"seg{i} done（key {ear.k_shift or 0}）")
print("wrote", OUT)
