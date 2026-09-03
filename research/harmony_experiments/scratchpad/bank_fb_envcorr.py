"""bank_live 的包絡域回授偵測：地基測試（08-16）

**為什麼這不是 G34 重跑。** G34（`fb_duck_probe.py`）死於結構性論據：
串流架構的迴路是 `angel(t) ≈ G·mic(t)`（即時轉換載體，diag17 實測 G 恆定
0.97:1），整個迴路的包絡自相似 → 任何 mic/out 能量比都趨近常數 → 延遲掃
0–697ms「單調上升、無峰」。**bank_live 沒有那條因果邊**：`y` 是音庫取樣
播放（`bank_live.py:1864` `p.render(hopN, r)`）＋pad＋殘響，只透過離散的
`note` 決策與 mic 耦合。他閉嘴之後音庫還會響 ~3 秒（STATE `--pad-hold`）。
所以 out 不是 mic 的函數，G34 的自相似前提在這裡不成立。

**這支只答一個問題（地基）**：mic 與 out 的能量包絡之間，存不存在一個
**非零延遲的相關峰**？
  有峰 → 物理回授路徑存在、延遲可估 → 才有資格談判別。
  無峰 → 跟 G34 一樣沒有可估的路徑，這條在 bank_live 也死。

**兩條血訓（GRAVEYARD G34/G35）已內建**：
1. `dmp["mic"]`／`dmp["out"]` 在同一個 callback 同一處 append
   （`bank_live.py:1913-1914`）＝out 是**產生時間軸**、不是喇叭播出的時間
   軸。所以峰**必須**落在非零延遲；0ms 附近的峰是時間軸假象，不是回授。
2. 不用 mic 能量分位切 quiet/sing 標籤——G35 更正判定那是壞標籤。這支
   完全不需要標籤：互相關是描述性的。
"""
import sys

import numpy as np
import soundfile as sf

SR, HOP = 44100, 512
DUMP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
        "SoloChoirCode/260811_bt/ear/dump_0815/")
WIN = 0.2                                  # 能量包絡窗 ~200ms（STATE 指定）


def env(path):
    """逐 HOP 的 RMS 包絡（out 走 mono 匯流排＝喇叭實際發出的合成聲）。"""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: sr={sr}"
    ch = x.shape[1]
    x = x.mean(1)
    m = len(x) // HOP
    e = np.sqrt((x[:m * HOP].reshape(m, HOP) ** 2).mean(1) + 1e-20)
    return e, ch, len(x)


mic, mic_ch, mic_n = env(DUMP + "noise_mic.wav")
out, out_ch, out_n = env(DUMP + "noise_out.wav")
st = np.load(DUMP + "noise_st.npz")
m = min(len(mic), len(out))
mic, out = mic[:m], out[:m]

print(f"mic {mic_ch}ch {mic_n} smp {mic_n / SR:6.1f}s  "
      f"out {out_ch}ch {out_n} smp {out_n / SR:6.1f}s")
print(f"共同 {m} 幀 @ {HOP / SR * 1000:.1f}ms = {m * HOP / SR:.1f}s")
print(f"npz: {', '.join(f'{k}{st[k].shape}' for k in st.files)}")
nfr = len(st["mok"])
print(f"npz 幀數 {nfr}（每 callback 一筆）→ callback hop "
      f"= {m / nfr * HOP / SR * 1000:.1f}ms\n")

# ── 包絡平滑到 ~200ms，取 log 後去均值（比值域會被 G34 的恆定比陷阱吃掉）──
w = max(1, int(round(WIN * SR / HOP)))
sm = lambda x: np.convolve(x, np.ones(w) / w, "same")
lm, lo_ = np.log(sm(mic) + 1e-9), np.log(sm(out) + 1e-9)

# out 真的在響的幀才算——靜音段的相關無意義（G34 的 `ref > 1e-5` 同理）
live = sm(out) > np.percentile(sm(out), 40)
print(f"out 有響的幀 {int(live.sum())}/{m} = {live.mean() * 100:.0f}%")

# ── 去趨勢（必要）──────────────────────────────────────────────
# 第一輪未去趨勢的結果：曲線從 d=0 的 0.302 一路爬到掃描邊緣 1393ms 的
# 0.372 還沒轉頭＝兩軌共有一個**數十秒尺度的慢趨勢**（整段錄音的段落／
# 音量結構），它把每一個延遲的 r 都墊高，正是 G34 說的「延遲越久相關越
# 弱／強」那種平凡效果。回授是短尺度耦合，所以減掉 DETREND 秒的移動平均
# 只留音節／樂句尺度的調變，峰才照得出來。
DETREND = 3.0
wd = max(3, int(round(DETREND * SR / HOP)) | 1)
hp = lambda x: x - np.convolve(x, np.ones(wd) / wd, "same")
a, b = hp(lm), hp(lo_)
edge = wd                                  # 移動平均在兩端不完整，切掉
a = a - a[live].mean()
b = b - b[live].mean()
live = live.copy()
live[:edge] = live[-edge:] = False

# ── 延遲掃描：正延遲＝out 領先 mic（回授必然如此）；負延遲當對照組 ──
# 物理預期：輸出緩衝（--sblock）＋喇叭→麥空氣路徑 ≈ 十幾到數百 ms。
# 負延遲＝mic 領先 out，物理上不可能是回授＝對照組（該明顯較低）。
print(f"\n掃延遲（正 = out 領先 mic；已去 {DETREND}s 趨勢）：")
print("   延遲ms    r        n")
rows = []
for d in range(-86, 173, 2):              # 幀；-1.0s … +2.0s
    if d >= 0:
        x, y = a[d:], b[:m - d] if d else b
        msk = live[:m - d] if d else live
    else:
        x, y = a[:m + d], b[-d:]
        msk = live[-d:]
    n = int(msk.sum())
    if n < 200:
        continue
    x, y = x[msk], y[msk]
    r = float(np.corrcoef(x, y)[0, 1])
    rows.append((d, r, n))

for d, r, n in rows:
    bar = "#" * max(0, int(r * 60))
    print(f"  {d * HOP / SR * 1000:7.0f}  {r:+.3f}  {n:6d}  {bar}")

pos = [t for t in rows if t[0] > 0]
dbest, rbest, _ = max(pos, key=lambda t: t[1])
r0 = next(r for d, r, _ in rows if d == 0)
redge = rows[-1][1]
print(f"\n最佳正延遲 d={dbest} 幀 = {dbest * HOP / SR * 1000:.0f}ms, r={rbest:+.3f}")
print(f"對照：d=0 r={r0:+.3f}（時間軸假象基準）、掃描邊緣 r={redge:+.3f}")
print("判讀：峰要落在非零延遲、且明顯高於 d=0 與邊緣，才算物理回授路徑。")
print("      若曲線單調（無峰）＝ 跟 G34 同樣沒有可估的路徑 → 這條也死。")

# ══ 第二階段：估回授量、取殘差 ════════════════════════════════════
# 模型走**功率域**（回授是聲學疊加，功率才可加；log 域不可加）：
#     mic_pow[t] ≈ direct_pow[t] + k · out_pow[t−d] + floor
# 要答的就是 STATE 那句：mic 的能量能不能用 out 解釋？
#     能 → direct ≈ 0 ＝ 喇叭繞回來的
#     不能 → direct > 0 ＝ 有外來聲源 ＝ 他在發聲
#
# **k 怎麼估而不用標籤**（G35 血訓：用音量切的標籤不能拿來定音量門檻）：
# direct_pow ≥ 0 恆成立 → 回授是 mic 能量的**下界**。所以 k 要用**低分位
# 迴歸**估：擬合線由「他不出聲」的那些幀自己決定，不需要誰去標它們。
# 最小平方會被他在唱的幀往上拉（而且 bank_live 的 note 由他的 f0 觸發＝
# out 的起止與他的唱段相關，這是 G34 自相似在本架構的殘影），分位迴歸不會。
TAU = 0.10
mp = sm(mic) ** 2
op = np.concatenate([np.zeros(dbest), sm(out) ** 2])[:m]
ok = (op > np.percentile(op, 40)) & (np.arange(m) >= edge)

ks = np.linspace(0, 2.0, 401)[1:]
loss = [float(np.mean(np.maximum(TAU * (mp[ok] - k * op[ok]),
                                 (TAU - 1) * (mp[ok] - k * op[ok])))) for k in ks]
k = float(ks[int(np.argmin(loss))])
direct = np.maximum(mp - k * op, 0.0)
frac = direct / np.maximum(mp, 1e-20)

print(f"\n── 回授量（{TAU:.0%} 分位迴歸，無標籤）──")
print(f"耦合 k={k:.3f}（out 功率的 {k:.1%} 繞回麥克風）"
      f" = {10 * np.log10(k + 1e-12):+.1f} dB")
print("mic 能量之中「解釋不掉」的比例分布（1.0＝完全外來聲源、0＝全是回授）：")
for p in (5, 10, 25, 50, 75, 90, 95):
    print(f"   {p:2d} 分位  {np.percentile(frac[ok], p):.3f}")
print(f"   解釋不掉 <10% 的幀（＝純回授）：{float((frac[ok] < 0.1).mean()):.1%}")
print(f"   解釋不掉 >50% 的幀（＝有外來聲源）：{float((frac[ok] > 0.5).mean()):.1%}")
print(f"   ⚠ 上面第一個數字約等於 TAU={TAU} 是**定義後果**（分位迴歸讓約 τ 的"
      "幀落在線下），不是獨立發現。")

# ── k 可不可信，取決於素材裡有沒有「喇叭大聲而他不出聲」的時段 ──────
# G34/G35 的更正就是栽在這：diag17 裡這種時段只有 5.5 秒，k 根本標定不出來。
# 交叉驗證用 npz 的 `f0`（門控**前**的偵測，未經已判死的鏡頭門控）：
# 若耦合真的只有 −20dB，回授不足以讓 f0 起振 → 「音庫在播 ∧ f0 靜默」
# 就是可信的「他不出聲」時段，而且完全不靠 mic 音量切標籤。
f0 = st["f0"]
vn = st["vn"]
nfr = len(f0)
idx = np.minimum((np.arange(m) * nfr // m), nfr - 1)   # 包絡幀 → npz 幀
playing = (vn[:, :] >= 0).any(1)[idx]                  # 音庫任一聲部在播
silent = (f0[idx] <= 0)                                # 偵測不到他的音高
cand = playing & silent & ok

runs, cur = [], 0
for v in cand:
    if v:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs = np.array(runs) * HOP / SR

print(f"\n── 標定素材夠不夠（交叉驗證，用 npz f0 而非 mic 音量）──")
print(f"音庫在播的幀 {float(playing[ok].mean()):.1%}；"
      f"其中 f0 靜默＝他不出聲的候選 {float(cand.sum()) * HOP / SR:.1f}s "
      f"（{float(cand.mean()):.1%} of {m * HOP / SR:.0f}s）")
if len(runs):
    print(f"連續時段 {len(runs)} 段，最長 {runs.max():.1f}s、"
          f"中位 {np.median(runs):.2f}s、≥1s 的有 {int((runs >= 1).sum())} 段"
          f"（共 {runs[runs >= 1].sum():.1f}s）")
    print(f"   對照 G34/G35 的 diag17：這種時段只有 5.5s → k 標不出來")
    kk = float(np.median(mp[cand] / np.maximum(op[cand], 1e-20))) if cand.sum() else float("nan")
    print(f"   只用這些幀直接估 k = {kk:.4f} "
          f"({10 * np.log10(kk + 1e-12):+.1f} dB)，"
          f"vs 分位迴歸的 {k:.3f} ({10 * np.log10(k + 1e-12):+.1f} dB)")
else:
    print("   一段都沒有 → k 標不出來，跟 G34/G35 同樣的困境")

# ── 決定性檢查：有沒有**乾淨**的標定幀？ ────────────────────────────
# 兩法差 10dB 的原因假設：候選段中位 0.10s 比 WIN=200ms 的平滑窗還短，
# 平滑把他自己的聲音抹進候選幀 → k 被高估。而且 f0 掉了不等於他真的停
# （換氣、字間、句尾殘響都會掉）。所以：
#   ① 兩端各留 GUARD 的保護帶（腐蝕），只取段落中心
#   ② 包絡窗縮到 SHORT，短於保護後的段長
# 若腐蝕後幾乎不剩幀 → 這段素材裡**沒有**可用來標定的乾淨時刻，
# 那就是 G34/G35 更正指定的處方：得另錄乾淨標定素材，這裡不可能算出 k。
GUARD, SHORT = 0.15, 0.05
g = max(1, int(round(GUARD * SR / HOP)))
ws = max(1, int(round(SHORT * SR / HOP)))
ero = cand.copy()
for s in range(1, g + 1):                  # 二值腐蝕：兩端各縮 GUARD
    ero &= np.concatenate([cand[s:], np.zeros(s, bool)])
    ero &= np.concatenate([np.zeros(s, bool), cand[:-s]])

sms = lambda x: np.convolve(x, np.ones(ws) / ws, "same")
mp_s = sms(mic) ** 2
op_s = np.concatenate([np.zeros(dbest), sms(out) ** 2])[:m]

print(f"\n── 決定性檢查：保護帶 {GUARD * 1000:.0f}ms、包絡窗 {SHORT * 1000:.0f}ms ──")
print(f"腐蝕後剩 {int(ero.sum())} 幀 = {ero.sum() * HOP / SR:.2f}s"
      f"（腐蝕前 {cand.sum() * HOP / SR:.1f}s）")
if ero.sum() >= 30:
    kc = float(np.median(mp_s[ero] / np.maximum(op_s[ero], 1e-20)))
    q1 = float(np.percentile(mp_s[ero] / np.maximum(op_s[ero], 1e-20), 25))
    q3 = float(np.percentile(mp_s[ero] / np.maximum(op_s[ero], 1e-20), 75))
    print(f"乾淨幀估 k = {kc:.4f} ({10 * np.log10(kc + 1e-12):+.1f} dB)，"
          f"四分位 {10 * np.log10(q1 + 1e-12):+.1f} … "
          f"{10 * np.log10(q3 + 1e-12):+.1f} dB")
    print(f"三法對照：分位迴歸 {10 * np.log10(k + 1e-12):+.1f} dB / "
          f"未腐蝕候選 {10 * np.log10(kk + 1e-12):+.1f} dB / "
          f"乾淨幀 {10 * np.log10(kc + 1e-12):+.1f} dB")
    print("  → 三法收斂（相差 <3dB）＝ k 可信，可以往判別走；"
          "發散＝素材標不出 k。")
else:
    print("→ 乾淨幀不足 30 幀。這段素材裡沒有「他確實不出聲而喇叭在響」的")
    print("  可用時刻，k 在此標不出來（與 G34/G35 更正的診斷一致）。")
    print("  處方（GRAVEYARD 早就寫下）：另錄 30–60s 乾淨標定素材——")
    print("  他完全不出聲、音庫照原音量放喇叭、同一支麥同一個擺位。")

# ── 最終：分離度（G34 的同一把尺，>2× 才算可行）──────────────────
# 兩堆都**不是**用 mic 音量切的（G35 血訓）：
#   不出聲＝上面腐蝕後的乾淨幀；在唱＝f0 有聲且同樣腐蝕過的段中心。
# f0 當標籤在這裡站得住，因為耦合只有 ~−15dB，音庫不足以讓 f0 起振
#（自洽檢查：乾淨幀依定義 f0=0）。
sings = (f0[idx] > 0) & ok
es = sings.copy()
for s in range(1, g + 1):
    es &= np.concatenate([sings[s:], np.zeros(s, bool)])
    es &= np.concatenate([np.zeros(s, bool), sings[:-s]])

pred = np.maximum(kc * op_s, 1e-20) if ero.sum() >= 30 else np.maximum(k * op_s, 1e-20)
ratio = mp_s / pred
rq = 10 * np.log10(ratio[ero] + 1e-12)
rs = 10 * np.log10(ratio[es] + 1e-12)

print(f"\n── 分離度（G34 同一把尺）──")
print(f"不出聲 {int(ero.sum())} 幀 / 在唱 {int(es.sum())} 幀")
print(f"  不出聲 mic/預測回授：中位 {np.median(rq):+.1f} dB  "
      f"(25–75% {np.percentile(rq, 25):+.1f} … {np.percentile(rq, 75):+.1f})")
print(f"  在唱   mic/預測回授：中位 {np.median(rs):+.1f} dB  "
      f"(25–75% {np.percentile(rs, 25):+.1f} … {np.percentile(rs, 75):+.1f})")
sep = 10 ** ((np.median(rs) - np.median(rq)) / 10)
print(f"  中位分離 {np.median(rs) - np.median(rq):+.1f} dB = {sep:.2f}×"
      f"   （G34 最佳 1.68×；>2× 才算可行）")

# 掃門檻看實際代價——這才是能不能上台的判準
print("\n  門檻dB   抓到在唱   誤擋不出聲")
best_t = None
for t in np.arange(-6, 25, 2.0):
    tp = float((rs > t).mean())
    fp = float((rq > t).mean())
    if best_t is None or (tp - fp) > best_t[0]:
        best_t = (tp - fp, t, tp, fp)
    print(f"   {t:+5.0f}    {tp:7.1%}    {fp:9.1%}")
_, t, tp, fp = best_t
print(f"\n最佳門檻 {t:+.0f}dB：抓到在唱 {tp:.1%}、誤把回授當成他在唱 {fp:.1%}")
print(f"重疊率 {1 - (tp - fp):.1%} —— 門控要的是「他有沒有在發聲」的二元答案，")
print("這個重疊就是它答錯的頻率。")

# ── 對照：改用 08-10 標定素材量到的 k（不是從這段污染素材估的）──────
# `fb_calib_ana.py` 在 CALIB 上量到 k(f) = −21.8/−21.6/−23.2/−20.1/−20.7 dB
#（80Hz–12kHz 只跨 3.1 dB ＝ **耦合其實很平坦**），寬頻 k = 0.133 (−17.5 dB)。
# 這推翻了「13 dB 散布來自頻率相關耦合」的假設，也給了一個獨立標定的 k：
# 它正好落在本段兩個無標籤估計（−20.0 / −14.8 dB）中間 ＝ 三者其實一致。
# 注意**只移植 k 不移植延遲**：CALIB 量到 547.6 ms 是「播放檔時間軸→mic」，
# 與 dump 的「產生時間軸→mic」定義不同，延遲仍用本段自己量到的 279 ms。
K_CALIB = 0.133 ** 2               # ⚠ 標定印的 0.133 是**振幅**比（20log10=−17.5dB）；
                                   # 這裡 mp_s/op_s 都是功率，係數要平方才對得上。
r2 = mp_s / np.maximum(K_CALIB * op_s, 1e-20)
rq2 = 10 * np.log10(r2[ero] + 1e-12)
rs2 = 10 * np.log10(r2[es] + 1e-12)
print(f"\n── 對照：改用標定 k={K_CALIB} ({10 * np.log10(K_CALIB):+.1f} dB) ──")
print(f"  不出聲 中位 {np.median(rq2):+.1f} dB (25–75% {np.percentile(rq2, 25):+.1f} … "
      f"{np.percentile(rq2, 75):+.1f})")
print(f"  在唱   中位 {np.median(rs2):+.1f} dB (25–75% {np.percentile(rs2, 25):+.1f} … "
      f"{np.percentile(rs2, 75):+.1f})")
print(f"  中位分離 {np.median(rs2) - np.median(rq2):+.1f} dB = "
      f"{10 ** ((np.median(rs2) - np.median(rq2)) / 10):.2f}×（k 只平移，分離度不變）")
bt2 = max(((float((rs2 > t).mean()) - float((rq2 > t).mean())), t,
           float((rs2 > t).mean()), float((rq2 > t).mean()))
          for t in np.arange(-12, 25, 1.0))
print(f"  最佳門檻 {bt2[1]:+.0f}dB：抓到在唱 {bt2[2]:.1%}、誤當發聲 {bt2[3]:.1%}")

# ── 散布的真正來源：回授有沒有埋進底噪？ ───────────────────────────
# CALIB 的回授 RMS 0.00681 / 底噪 0.000581 ＝ SNR 21.4 dB（那是演出音量）。
# 若 dump_0815 的音庫較小聲，k·out 會掉到 mic 底噪附近，比值自然亂跳——
# 那 13 dB 散布就是量測 SNR 不足，不是耦合不穩，兩者的處方完全不同。
floor = float(np.percentile(mp_s, 2) ** 0.5)
fb_rms = np.sqrt(K_CALIB * op_s[ero])
print(f"\n── 13 dB 散布是耦合不穩、還是 SNR 不足？──")
print(f"  mic 底噪估計（2 分位）{20 * np.log10(floor + 1e-12):+.1f} dBFS")
print(f"  乾淨幀的預測回授 RMS：中位 {20 * np.log10(np.median(fb_rms) + 1e-12):+.1f} dBFS"
      f"（{np.percentile(20 * np.log10(fb_rms + 1e-12), 10):+.1f} … "
      f"{np.percentile(20 * np.log10(fb_rms + 1e-12), 90):+.1f}）")
print(f"  → 回授相對底噪 SNR 中位 {20 * np.log10(np.median(fb_rms) / (floor + 1e-12)):+.1f} dB"
      f"（CALIB 演出音量時是 +21.4 dB）")
print(f"  低於 +6dB SNR 的乾淨幀：{float((fb_rms < floor * 2).mean()):.1%}"
      "  ← 這些幀的比值本來就不可能穩")
