"""C 案（拍手／彈指＝聲學暗號）的分離度量測 — 08-03 §J 拍板的離線判準。

問題：展場「換你」訊號若用拍手，偵測器必須把拍手跟「他還在唱」分開。
08-01 §K 已證能量餘裕只剩 2 dB＝能量這條路死。C 案改走起音形狀：
寬頻銳起音（拍手/彈指）vs 諧波慢起音（歌聲）vs 慢淡入（天使 bleed）。

誠實標準（與 08-01 量 bleed 同一招）：不是比「拍手 vs 歌聲平均」，
是拿同一偵測器掃過全部歌聲素材，找出最像拍手的瞬間（塞音子音 /t/ /k/
是最可能的假陽性），量它與最弱一下拍手的餘裕。餘裕 >= 6 dB 才算過。

偵測器（兩維規則，故意簡單到能在 callback 裡逐 hop 算）：
  rise = 短窗能量 dB 在 ~12ms 內的爬升量（銳起音）
  hf   = 4kHz 以上頻譜能量佔比（寬頻、非諧波）
  觸發 = rise >= R 且 hf >= H；R、H 由 grid search 找「收下全部拍手、
  歌聲+bleed 零誤觸」的工作點，回報餘裕。

用法（conda env vcclient-dev）：
  錄拍手：python clap_probe.py --record out/claps_$(date +%y%m%d).wav
          （K669B 站姿、展場距離；倒數後拍 10 下，每下隔 >1s）
  錄彈指：同上換檔名 snaps_*
  分析：  python clap_probe.py --analyze out/claps_*.wav out/snaps_*.wav
          （歌聲/bleed 素材已寫死為 §J 指定檔，--sing/--bleed 可換）
"""
import argparse
import sys

import numpy as np
import soundfile as sf

SR = 44100
WIN, HOP = 256, 128            # ~5.8ms 窗 / ~2.9ms hop
RISE_HOPS = 4                  # rise 跨 4 hop ~= 11.6ms
HF_CUT_HZ = 4000
EPS = 1e-10

SING_DEFAULT = ["out/resp2_live_260802_234841.wav"]        # 252s K669B 站姿實唱
BLEED_DEFAULT = [                                          # 07-31 實戴、天使在播
    "out/live_mic_260731_225804.wav",
    "out/live_mic_260731_230150.wav",
    "out/live_mic_260731_233041.wav",
]


def features(x):
    """逐 hop 算 (rise_db, hf_ratio, db)。全 numpy 向量化，252s 約 1s 算完。"""
    n = (len(x) - WIN) // HOP
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, WIN), strides=(x.strides[0] * HOP, x.strides[0]))
    # -80 dBFS 地板：數位靜音是 -200 dB，不設地板的話「從靜音跳出來」的
    # rise 會被灌成 100+ dB，淹沒真正要量的起音形狀差異
    db = np.maximum(
        20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + EPS), -80.0)
    rise = np.concatenate([np.zeros(RISE_HOPS), db[RISE_HOPS:] - db[:-RISE_HOPS]])
    spec = np.abs(np.fft.rfft(frames * np.hanning(WIN), axis=1)) ** 2
    cut = int(HF_CUT_HZ * WIN / SR)
    hf = spec[:, cut:].sum(axis=1) / (spec.sum(axis=1) + EPS)
    return rise, hf, db


def pick_events(rise, hf, db, k, min_gap_s=0.5):
    """拍手檔裡挑 k 個事件：rise 峰值、彼此隔 min_gap，回傳各事件的 (rise, hf)。"""
    gap = int(min_gap_s * SR / HOP)
    order = np.argsort(rise)[::-1]
    picked = []
    for i in order:
        if all(abs(i - j) >= gap for j in picked):
            picked.append(i)
            if len(picked) == k:
                break
    # 事件的 hf 取峰值附近 ±2 hop 的最大（rise 峰與頻譜峰可差半個窗）
    return [(float(rise[i]),
             float(hf[max(0, i - 2):i + 3].max()),
             float(db[i])) for i in sorted(picked)]


def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == SR, f"{path}: {sr} != {SR}"
    return np.ascontiguousarray(x)


def analyze(cue_paths, sing_paths, bleed_paths, n_events):
    cues = []                                        # [(rise, hf, db, path)]
    for p in cue_paths:
        r, h, d = features(load(p))
        cues += [(*e, p) for e in pick_events(r, h, d, n_events)]
        print(f"[cue] {p}: 取 {n_events} 事件, rise "
              f"{min(e[0] for e in cues[-n_events:]):.1f}–"
              f"{max(e[0] for e in cues[-n_events:]):.1f} dB")

    bg = {}                                          # path -> (rise, hf, db)
    for p in sing_paths + bleed_paths:
        bg[p] = features(load(p))
        kind = "sing" if p in sing_paths else "bleed"
        print(f"[{kind}] {p}: {len(bg[p][0])} hops, "
              f"max rise {bg[p][0].max():.1f} dB")

    # grid search：R 掃 6..40 dB、H 掃 0.05..0.6，找「全收拍手、背景零誤觸」
    best = None
    for R in np.arange(6, 40.5, 0.5):
        for H in np.arange(0.05, 0.61, 0.01):
            if not all(r >= R and h >= H for r, h, _, _ in cues):
                continue                             # 漏拍手＝不合格
            fp = sum(int(np.sum((rr >= R) & (hh >= H)))
                     for rr, hh, _ in bg.values())
            # 餘裕＝最弱拍手 rise 對「背景中 hf>=H 者的最大 rise」的 dB 差
            bg_max = max((rr[hh >= H].max() if (hh >= H).any() else -np.inf)
                         for rr, hh, _ in bg.values())
            margin = min(r for r, _, _, _ in cues) - max(bg_max, R)
            cand = (fp == 0, margin, R, H, fp)
            if best is None or cand > best:
                best = cand
    if best is None:
        print("\n判決：不通過 — 沒有任何 (R,H) 能收下全部拍手事件")
        return
    ok, margin, R, H, fp = best
    print(f"\n工作點 R={R:.1f} dB, H={H:.2f}  誤觸 {fp} hop")
    weakest = min(cues, key=lambda c: c[0])
    print(f"最弱拍手 rise {weakest[0]:.1f} dB (hf {weakest[1]:.2f}, "
          f"level {weakest[2]:.1f} dBFS, {weakest[3]})")
    if ok:
        print(f"判決：{'通過' if margin >= 6 else '勉強（<6dB，建議實測誤觸率）'}"
              f" — 餘裕 {margin:.1f} dB")
    else:
        print(f"判決：不通過 — 零誤觸工作點不存在（最少 {fp} 個誤觸 hop）")


def record(path, secs):
    import sounddevice as sd
    dev = sd.query_devices(kind="input")
    print(f"輸入裝置：{dev['name']}（確認是 K669B！）")
    for t in (3, 2, 1):
        print(f"  {t}...", flush=True)
        sd.sleep(1000)
    print(f"錄音中 {secs}s — 拍 10 下，每下隔 1 秒以上")
    x = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="float64")
    sd.wait()
    peak = float(np.abs(x).max())
    sf.write(path, x, SR)
    print(f"存 {path}  peak {20*np.log10(peak+EPS):.1f} dBFS"
          + ("  ⚠ 削波" if peak > 0.99 else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", metavar="OUT_WAV")
    ap.add_argument("--secs", type=float, default=20)
    ap.add_argument("--analyze", nargs="+", metavar="CUE_WAV")
    ap.add_argument("--events", type=int, default=10)
    ap.add_argument("--sing", nargs="*", default=SING_DEFAULT)
    ap.add_argument("--bleed", nargs="*", default=BLEED_DEFAULT)
    a = ap.parse_args()
    if a.record:
        record(a.record, a.secs)
    elif a.analyze:
        analyze(a.analyze, a.sing, a.bleed, a.events)
    else:
        ap.print_help()
        sys.exit(1)
