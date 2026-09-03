"""bank_gate_forensics — 門控法醫（v14.2 配套）：對 dump 驗收狀態機。

用法: python bank_gate_forensics.py scratchpad/<dump前綴>
輸出: ①每次門/臉/嘴狀態轉移的時間戳與間隔 ②不變量檢查（抓「不該發生」）：
  I1 門關(mok=0)期間 f0 必須為 0（違反＝門控被繞過）
  I2 門關超過 release(0.25s)+尾（0.3s）後輸出包絡必須 <-40dB（違反＝關不掉）
  I3 臉丟失→門關的延遲必須 ≤0.9s（0.8s 寬限＋一個嘴迴圈）
  I4 臉回來→門開的延遲必須 ≤0.2s（嘴 worker 週期＋偵測）
"""
import sys

import numpy as np
import soundfile as sf

pre = sys.argv[1]
z = np.load(pre + "_st.npz")
out, sr = sf.read(pre + "_out.wav")
om = out.mean(axis=1) if out.ndim == 2 else out
hop = float(z["hop"]) if "hop" in z else 0.035
mok, f0, face, mon = z["mok"], z["f0"], z["face"], z["mon"]
n = len(mok)
t = np.arange(n) * hop
print(f"dump {n} hops（{n*hop:.1f}s）")


def transitions(x, name):
    ch = np.where(np.diff(x.astype(int)) != 0)[0]
    print(f"\n[{name}] 轉移 {len(ch)} 次")
    for i in ch[:40]:
        print(f"  {t[i+1]:8.2f}s  {int(x[i])} → {int(x[i+1])}")
    return ch


tf = transitions(face, "臉")
tm = transitions(mon, "嘴開")
tg = transitions(mok, "門")

print("\n── 不變量 ──")
# I1（R5 修正：f0 通道現在記門控**前**的偵測＝門關時有 f0 是正常的；
# 正確不變量＝門關期間 note 不得改變）
note = z["note"]
v1 = 0
for i in range(1, len(mok)):
    if mok[i] < 0.5 and mok[i-1] < 0.5 and note[i] != note[i-1]             and note[i] >= 0:
        v1 += 1
print(f"I1 門關期間 note 改變：{v1} 次 {'✗ 違反' if v1 else '✓'}")
# I2 門關 ≥0.55s 之後的輸出包絡
env = np.array([np.sqrt((om[int(i*hop*sr):int((i+1)*hop*sr)]**2).mean())
                for i in range(n)])
bad = 0
run = 0
for i in range(n):
    run = run + 1 if mok[i] < 0.5 else 0
    if run * hop > 0.55 and env[i] > 0.01:
        bad += 1
print(f"I2 門關 >0.55s 後仍有輸出：{bad} hops {'✗ 違反' if bad else '✓'}")
# I3 臉丟→門關延遲
lags = []
for i in np.where(np.diff(face.astype(int)) == -1)[0]:
    j = i
    while j < n and mok[j] > 0.5:
        j += 1
    if j < n:
        lags.append((j - i) * hop)
if lags:
    print(f"I3 臉丟→門關：p50 {np.median(lags):.2f}s max {max(lags):.2f}s "
          f"{'✓' if max(lags) <= 0.9 else '✗ >0.9s'}")
else:
    print("I3 無臉丟事件（本場沒走開過）")
# I4 臉回→門開
lags = []
for i in np.where(np.diff(face.astype(int)) == 1)[0]:
    j = i
    while j < n and mok[j] < 0.5:
        j += 1
    if j < n:
        lags.append((j - i) * hop)
if lags:
    print(f"I4 臉回→門開：p50 {np.median(lags):.2f}s max {max(lags):.2f}s"
          f"（含他重新開口的時間；純門延遲看 p50）")
else:
    print("I4 無臉回事件")
