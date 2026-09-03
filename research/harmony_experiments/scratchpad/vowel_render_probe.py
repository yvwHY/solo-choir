"""vowel_render_probe — 模型到底渲不渲得出圓唇後母音？（08-13）

新庫驗收：喔（源 479/700）與嗚（源 317/818）渲完雙雙變成啊（F1 被撐到
~750）。兩種可能：①我挑的那一段剛好不好 ②模型（他自己的嗓訓的 reflow）
根本吐不出圓唇後母音。本檔用多個候選段各渲一顆音來分辨——若八個候選
全塌＝是模型的性格，誠實砍掉這兩層，不再浪費他的耳朵。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_render_probe.py
"""
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, nearest_vowel = _a["lpc_formants"], _a["nearest_vowel"]
SR, HOP = 44100, 512
MIDI = 48
MODEL = "reflow-bass1/model_32000.pt"


def meas(x):
    x16 = resample_poly(x, 16000, SR)
    n = 1600
    f = [lpc_formants(x16[i:i + n]) for i in range(0, max(1, len(x16) - n), n)]
    f = [q for q in f if len(q) >= 2]
    if not f:
        return float("nan"), float("nan")
    return (float(np.median([q[0] for q in f])),
            float(np.median([q[1] for q in f])))


def main():
    import torch
    import spike_stream6 as S
    cands = json.load(open("scratchpad/vowel_source_cands.json"))
    svc = S.Svc([(f"{S.DDSP}/exp/{MODEL}", 0.0, 1.0, 1)], step=2,
                t_start=0.85)
    NF = int(2.0 * SR / HOP)
    f0 = np.full(NF, 440.0 * 2 ** ((MIDI - 69) / 12.0))
    vol = np.full(NF, 0.06)
    vol[:4] = np.linspace(0, 0.06, 4)
    vol_t = torch.from_numpy(vol).float().to(svc.device)[None, :, None]
    mask = torch.ones(1, NF * HOP, device=svc.device)
    for v in ("喔", "嗚", "啊", "咿"):        # 後兩個＝對照組（已知會活）
        print(f"\n{v}（源 → 渲後）")
        for r in cands[v][:4]:
            x, fs = sf.read(r["clip"], always_2d=True)
            mid = (r["t0"] + r["t1"]) / 2
            a0 = max(0, int((mid - 0.5) * fs))
            seg = x[a0:a0 + int(1.0 * fs), 0].astype("float64")
            if fs != SR:
                seg = resample_poly(seg, SR, fs)
            with torch.no_grad():
                UU = svc.encode(seg)[:, 8:-8]
                nb = UU.size(1)
                torch.manual_seed(1234 + MIDI)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask),
                               ratios=[None])[0].cpu().numpy()
            F1, F2 = meas(au[int(0.5 * SR):int(1.8 * SR)].astype("float64"))
            v1, v2, _d = nearest_vowel(F1, F2)
            print(f"  {r['clip'].split('/')[-1]} {r['t0']:.1f}s  "
                  f"源 {r['F1']:.0f}/{r['F2']:.0f} → 渲 {F1:.0f}/{F2:.0f}"
                  f"  ＝{v1}（次{v2}）{'✓' if v1 == v else '✗'}")


if __name__ == "__main__":
    main()
