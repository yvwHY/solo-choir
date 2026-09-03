"""a_enh.py — A 版配方＋NSF-HiFiGAN enhancer（08-05 §F；接 clean_ab 判決）

Harry 定位：「enh 檔其實還好，我說的是非 enh 的 bass wav」＝autotune 感
不在音高曲線（canon/bare 的 f0 已量測＝他的曲線 −12、偏差 1–2c）、在
**低音域 CombSub 裸輸出的死平諧波**；NSF-HiFiGAN enhancer 重整後消失。

本檔＝把 enhancer 掛上現行配方（phrase_render `enhance=True` 守衛參數）
渲染 A+enh 版，素材同 clean_ab 兩句（xcorr 選段）：
  seg{i}_Aenh_mix.wav / seg{i}_Aenh_bass1_lower.wav
過耳＝enhancer 入配方（再量 live RTF 預算）；不過＝enhancer 也救不了
配方層，回頭查骨架×enhancer 的交互。

Run (DDSP venv):  python a_enh.py
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

H = Path(__file__).parent
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402
from prosody_ab import B1, GU, GL, DRY  # noqa: E402
from bare_test import OUT, DUMP, his_phrases, wr  # noqa: E402


def main():
    import torch
    x, _ = sf.read(str(DUMP), dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs, _ = his_phrases(x)

    torch.manual_seed(1234)
    rend = {"upper": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                    expr_seed=R.VOICES["upper"][1],
                                    enhance=True),
            "lower": PhraseRenderer(R.DDSP, B1, expr_seed=20260805,
                                    enhance=True)}
    for r_ in rend.values():
        r_.render(np.zeros(SR), [60], TICK_SAMPS)

    for i, seg in enumerate(segs):
        sh = {"f0m": dm.harvest_f0(seg)}
        # 音符線＝讀 bare_test 落檔的同一條（EarV3 隨機性對呼叫序敏感，
        # 重算會岔線＝不是單一變因；實測岔線後同音幀只剩 11–37%）。
        d = json.load(open(OUT / f"seg{i}_notes.json"))
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        t0 = time.perf_counter()
        ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh, **kw[v])
               for v in R.VOICES}
        dt = time.perf_counter() - t0
        n = min(len(seg), *(len(a) for a in ang.values()))
        wr(f"seg{i}_Aenh_mix.wav",
           DRY * seg[:n] + GU * ang["upper"][:n] + GL * ang["lower"][:n])
        wr(f"seg{i}_Aenh_bass1_lower.wav", ang["lower"][:n])
        print(f"seg{i}: 兩聲部含 enhancer {dt:.2f}s／{len(seg)/SR:.1f}s 音訊"
              f" = RTF {dt/(len(seg)/SR):.2f}")
    print("wrote", OUT, "(seg*_Aenh_*.wav)")


if __name__ == "__main__":
    main()
