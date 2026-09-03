"""vs_male3_trial.py — VocalSet male3 試訓模型的耳測樣本（08-04）

⚠ 08-05 素材翻案：本檔與 prosody_ab 同一套 `[0::2]` 選段＝選到 dump 的
回應段（三聲部混音）；產物作廢、male3 判決需在乾淨素材上重測
（worklog 08-05 §C）。勿再執行。

從 16:00 場 dump 抽 Harry 實唱句，渲染：male3 唱他的線（音色檢查）、
male3 唱下天使線（實戰位）、girl 同線對照、dry+male3 混音。
素材抽取與渲染配方照抄 angel_audit.py（同一口徑）。判決＝Harry 耳朵。

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
segs = sorted(segs, key=len, reverse=True)[:2]     # 取最長兩句＝最有內容
print(f"素材：{len(segs)} 句, {', '.join(f'{len(s)/SR:.1f}s' for s in segs)}")

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
