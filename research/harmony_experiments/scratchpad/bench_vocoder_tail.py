"""vocoder 尾段化的邊距掃描：只 vocode 尾段 + m 幀邊距，跟整窗 vocode 的
同一段比對，找 max-abs-diff 歸零的 m。

為什麼問這個（08-09 §K）：視窗 0.71s，但 SOLA 只取尾巴 blk+cf+sola_search
＝0.29s ＝ **59% 的合成工作被丟掉**，而那 59% 落在全系統最貴的 NSF-HiFiGAN
上（整塊 55%）。v23 `--enh-tail` 在 CombSub 線驗證過同一原理（−35ms/塊），
但移植 reflow 時綁在「enhancer」這個名字上一起被丟掉了（spike_stream6.py:275
「enh_tail 參數收下但不用」）——reflow 沒有 enhancer，但有 vocoder。

NSF-HiFiGAN 是卷積上採樣器＝感受野有限。若 m 大於感受野，尾段單獨 vocode
應與整窗的尾段**逐位元相同** → 那是 Regime A，不必耳測。

跑：260724_ddsp_svc_6x/venv/bin/python scratchpad/bench_vocoder_tail.py
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
USED = BLK + CF + SOLA_SEARCH + LAST_DELAY      # SOLA 真正碰到的尾段
USED_F = -(-USED // HOP)                        # 換算成幀（進位）
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

print(f"視窗 {WIN} 樣本 = {WIN/SR:.2f}s = {WIN//HOP} 幀")
print(f"SOLA 用到的尾段 {USED} 樣本 = {USED/SR:.2f}s = {USED_F} 幀"
      f"（{100*USED/WIN:.0f}% of 視窗）\n")

# 取 5 個有唱的視窗（避開靜音）
starts = []
for st in range(0, len(x) - WIN, WIN):
    if float(np.abs(x[st:st + WIN]).mean()) > 0.01:
        starts.append(st)
    if len(starts) == 5:
        break
print(f"取樣 {len(starts)} 個有聲視窗\n")

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
        # 種子固定＝mel 可重現（reflow 的 randn 見 08-08 血訓）
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

print(f"{'邊距 m（幀）':<14}{'m 的秒數':>10}{'max-abs-diff':>16}")
for m in MARGINS:
    print(f"{m:<14}{m*HOP/SR:>10.3f}{worst[m]:>16.3e}"
          + ("   ← 逐位元相同" if worst[m] == 0.0 else ""))

# 省多少：以第一個視窗計時
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
print(f"\n邊距 {pick} 幀（{pick*HOP/SR:.3f}s）→ vocode {k}/{mel.shape[1]} 幀")
print(f"一張嘴 vocoder：整窗 {t_full:.1f}ms → 尾段 {t_tail:.1f}ms"
      f"（省 {t_full-t_tail:.1f}ms，{100*(t_full-t_tail)/t_full:.0f}%）")
print(f"三張嘴合計約省 {3*(t_full-t_tail):.0f}ms/塊"
      f"（註：台架有 mps.synchronize＝絕對值偏高，只看比例）")
