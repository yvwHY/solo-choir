"""angel_audit.py — 天使表現層體檢（08-04 固化）

Harry：「我要你全面檢查：和諧、和聲、音量、殘響，所有對於天使表現層面的」。
八軸一張成績單，**每項對「聽得到的參照」量，不對內部表徵**——同日兩次教訓：
內部一致性指標在使用者說「不對」的同時可以很漂亮（polish 空間 bug、C 吸附
自洽）。素材＝實戴 session dump 的實唱句。

用法（DDSP venv）:  python angel_audit.py [dump.wav] [--gate 0.0474]

⚠ dump＝他的句與回應交錯，且回應裡他的乾聲最大聲（harvest 主抓他、girl 佔比
判不到）→ 靠「他的句永遠在回應之前」取偶數位去重。句長成對＝污染的指紋。"""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch

H = Path("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer/harmony")
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm, reverb as rv, reh_polish
from phrase_render import PhraseRenderer
from pitch import SR
from world_live import TICK_SAMPS
from live_v3 import EarV3

# ── 現行設定（他 08-03 深夜拍板「先以目前數據為主」）──
DRY, ANGEL, UP_DB, LO_DB, GAIN, INDEP = 0.944, 0.75, 1.0, 0.0, 0.5, 0.15
GU, GL = ANGEL * 10 ** (UP_DB / 20), ANGEL * 10 ** (LO_DB / 20)
WET = 0.20
DISS = reh_polish.DISS
per = TICK_SAMPS / dm.HOP

# ── 0. 素材：從 dump 抽他的實唱句 ──
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("dump", nargs="?", default=str(H / "out/resp2_live_260803_232426.wav"))
_ap.add_argument("--gate", type=float, default=0.0474)
_a = _ap.parse_args()
x, _ = sf.read(_a.dump, dtype="float64", always_2d=True)
x = np.ascontiguousarray(x[:, 0])
R.RMS_GATE = _a.gate
segs = []
for s, e in R.find_phrases(x, 0.35, 0.8):
    if s / SR < 12:                                   # 校準區
        continue
    f0 = dm.harvest_f0(np.ascontiguousarray(x[s:e]))
    v = f0[f0 > 0]
    if len(v) < 30:
        continue
    hi = float(np.mean(v > 340.0))                    # girl 線＝回應的指紋
    if hi < 0.05:
        segs.append(np.ascontiguousarray(x[s:e]))
# dump＝他的句與回應交錯，而回應裡他的乾聲最大聲＝harvest 主抓他、girl 佔比
# 判不到 → 上一版把回應也收進來（句長成對＝指紋）。他的句永遠在回應之前：取偶數位。
segs = segs[0::2]
print(f"素材：{len(segs)} 句他的實唱（K669B）, "
      f"{', '.join(f'{len(s)/SR:.1f}s' for s in segs)}")

def true_line(seg, n):
    f0 = dm.harvest_f0(seg)
    out = []
    for k in range(n):
        w = f0[int(k * per):int((k + 1) * per)]; w = w[w > 0]
        out.append(None if len(w) < 3 else float(np.median(69 + 12 * np.log2(w / 440.0))))
    return out

# ── 渲染（現行配方，逐句收 stems）──
torch.manual_seed(1234)
ear = EarV3(indep=INDEP, stab=1, key=0); ear.v2t.shift = 12
kt = R.KeyTracker()
rend = {v: PhraseRenderer(R.DDSP, m, expr_seed=sd) for v, (m, sd) in R.VOICES.items()}
for r in rend.values():
    r.render(np.zeros(SR), [60], TICK_SAMPS)

rows = []
for seg in segs:
    kt.push(dm.harvest_f0(seg))
    r = kt.best()
    if r is not None and r[2] >= 0.015:
        ear.k_shift = r[0]
    lead, up, lo = R.phrase_notes(ear, seg)
    d = {"lead": lead, "upper": up, "lower": lo}
    T = true_line(seg, len(lead))
    d["lead"] = [None if t is None else int(round(t)) for t in T]
    root = (-(ear.k_shift or 0)) % 12
    reh_polish.polish(d, "upper", "lower", root)
    reh_polish.polish(d, "lower", "upper", root)
    sh = {}
    kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0, uv_dry=0.0,
                  vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1]) for v in R.VOICES}
    ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh, **kw[v]) for v in R.VOICES}
    n = min(len(seg), *(len(a) for a in ang.values()))
    rows.append(dict(seg=seg[:n], T=T, d=d, u=ang["upper"][:n], l=ang["lower"][:n],
                     vm=sh["vm"], key=ear.k_shift or 0))
print(f"定調軌跡：{[r['key'] for r in rows]}")

print("\n════ 體檢 ════")
# 1. 和諧：天使 vs 他真實音高
n = di = 0
for r in rows:
    for t, u, l in zip(r["T"], r["d"]["upper"], r["d"]["lower"]):
        if t is None: continue
        for a in (u, l):
            if a is None: continue
            n += 1; di += round(abs(a - t)) % 12 in DISS
print(f"1 和諧      對他真實音高不和諧 {100*di/max(n,1):5.1f}%   [語料 12.1／pair06 修後 17.3]")

# 2. 和聲內容：兩天使互相
sm = st_ = dd = m2 = 0
for r in rows:
    U, L = r["d"]["upper"], r["d"]["lower"]
    for u, l in zip(U, L):
        if u is None or l is None: continue
        m2 += 1
        sm += (u - l) % 12 == 0
        dd += abs(u - l) % 12 in DISS
print(f"2 和聲內容  兩天使同音 {100*sm/max(m2,1):5.1f}%  互撞 {100*dd/max(m2,1):5.1f}%   [語料 同音19.9 不和諧12.1]")

# 3. 音準執行＋F21 音程誤差
errs, iv = [], []
for r in rows:
    n_h = len(r["u"]) // dm.HOP
    fu = dm.harvest_f0(np.ascontiguousarray(r["u"]))[:n_h]
    fl = dm.harvest_f0(np.ascontiguousarray(r["l"]))[:n_h]
    for k in range(n_h):
        t = min(int(k * dm.HOP / TICK_SAMPS), len(r["d"]["upper"]) - 1)
        mu, ml = r["d"]["upper"][t], r["d"]["lower"][t]
        t_v = r["vm"][k] > 0.5 if k < len(r["vm"]) else False
        if not t_v:
            continue                    # uv 段的 stem 是他的透傳原聲＝量到的不是天使
        if fu[k] > 0 and mu is not None:
            c = 1200 * np.log2(fu[k] / (440 * 2 ** ((mu - 69) / 12)))
            if abs(c) < 250: errs.append(abs(c))
        if fu[k] > 0 and fl[k] > 0 and mu is not None and ml is not None:
            got = 1200 * np.log2(fu[k] / fl[k]); want = 100.0 * (mu - ml)
            if abs(got - want) < 300: iv.append(got - want)
print(f"3 音準執行  |cents 誤差| 中位 {np.median(errs):5.1f}c   音程誤差 SD {np.std(iv):5.1f}c   [F21 target: 11c／>20c 7.9%]")

# 4. 錯開：他換音 → 天使換音的延遲
lags = []
for r in rows:
    Tn = [None if t is None else int(round(t)) for t in r["T"]]
    ch_h = [k for k in range(1, len(Tn)) if Tn[k] is not None and Tn[k] != Tn[k-1]]
    U = r["d"]["upper"]
    ch_a = [k for k in range(1, len(U)) if U[k] is not None and U[k] != U[k-1]]
    for c in ch_h:
        after = [a for a in ch_a if a >= c]
        if after: lags.append((after[0] - c) * TICK_SAMPS / SR)
print(f"4 錯開      他換音→天使跟進 中位 {np.median(lags)*1000:4.0f}ms  p90 {np.percentile(lags,90)*1000:4.0f}ms   [tick 網格＝187.5ms 一格]")

# 5. 子音疊加：uv 段他的聲音在回應裡的等效增益
gains = []
for r in rows:
    v = np.repeat(np.asarray(r["vm"]) > 0.5, dm.HOP)[:len(r["seg"])]
    uvm = ~v
    if uvm.sum() < SR * 0.05: continue
    d0 = r["seg"][uvm]
    tot = DRY * d0 + GU * r["u"][:len(v)][uvm] + GL * r["l"][:len(v)][uvm]
    gains.append(20 * np.log10(np.sqrt(np.mean(tot**2)) / (np.sqrt(np.mean((DRY*d0)**2)) + 1e-12)))
print(f"5 子音疊加  uv 段（換氣/子音）比只有乾聲多 {np.mean(gains):+5.1f} dB   [0＝無疊加；透傳×2 聲部]")

# 6. 音量平衡（現行增益）
dd_ = aa_ = 0
for r in rows:
    v = np.repeat(np.asarray(r["vm"]) > 0.5, dm.HOP)[:len(r["seg"])]
    dd_ += np.sum((DRY * r["seg"][v]) ** 2)
    aa_ += np.sum((GU * r["u"][:len(v)][v] + GL * r["l"][:len(v)][v]) ** 2)
print(f"6 音量平衡  兩天使合計 − 他 = {10*np.log10(aa_/dd_):+5.1f} dB   [他最終手調的組合＝天使領先約 +0.7 dB]")

# 7. 殘響：濕/直比＋削峰
ir = rv.make_ir(1.8, trim_db=35)
pks, wr = [], []
for r in rows:
    mix = DRY * r["seg"] + GU * r["u"] + GL * r["l"]
    w = rv.wet(mix, ir)
    y = (np.concatenate([mix, np.zeros(len(w) - len(mix))]) + WET * w) * GAIN
    pks.append(np.abs(y).max())
    wr.append(10 * np.log10(np.sum((WET*w)**2) / np.sum(mix**2)))
print(f"7 殘響      濕/直 {np.mean(wr):+5.1f} dB   尾巴 +0.88s/句   ER crest 5.0（擴散）  [送出量 0.20 未在房間裁過]")
print(f"8 削峰      gain {GAIN} 下峰值 max {max(pks):.2f}   [>1.0＝clip]")
