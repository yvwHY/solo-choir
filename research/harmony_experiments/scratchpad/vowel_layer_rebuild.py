"""vowel_layer_rebuild — 依母音重挑五層素材（08-13，v21 前置）

vowel_source_scan 證實 0619 語料五個母音齊全（欸 clip_0103 F1 519/F2
1824、嗚 clip_0035 319/796），原挑法照亮度撈才會撈到三個喔。

本檔挑的是**連續穩定段**不是單一窗——渲染 encode 會抓窗中心 ±40 hop
（≈0.93s）的上下文，300ms 的窗會把隔壁音素一起吃進去（原挑法的隱形
坑）。做法：同母音的相鄰窗併成 run → 取 ≥0.7s 且平均距離最小的 run →
中心 ±0.5s 當層窗。

輸出：建議的 VOWEL_LAYERS 區塊（人工貼進 bank_live.py）＋ JSON。
渲完必須跑 layer_vowel_audit 驗收——**渲染會改變母音**，源端對不代表
渲出來對（L3 喔 F2 829 → 渲後 1877＝欸，就是這樣壞的）。
"""
import glob
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, REF = _a["lpc_formants"], _a["REF"]
_s = {}
exec(compile(open("scratchpad/vowel_source_scan.py").read()
             .split("\ndef main()")[0], "vss", "exec"), _s)
dist, C, WIN, HOP = _s["dist"], _s["C"], _s["WIN"], _s["HOP"]
ORDER = ["喔", "欸", "咿", "嗚", "啊"]      # 必須與校準 labels 同序
MINRUN = 0.7


def windows(fp):
    x, fs = sf.read(fp, always_2d=True)
    x = x[:, 0].astype("float64")
    if len(x) < int(WIN * fs):
        return []
    x16 = resample_poly(x, 16000, fs)
    w, h = int(WIN * 16000), int(HOP * 16000)
    out = []
    for i in range(0, len(x16) - w, h):
        seg = x16[i:i + w]
        if float(np.sqrt((seg ** 2).mean())) < 0.02:
            continue
        fa, fb = lpc_formants(seg[:w // 2]), lpc_formants(seg[w // 2:])
        if len(fa) < 2 or len(fb) < 2:
            continue
        if abs(fa[0] - fb[0]) > 60 or abs(fa[1] - fb[1]) > 180:
            continue
        F1, F2 = (fa[0] + fb[0]) / 2, (fa[1] + fb[1]) / 2
        v = min(REF, key=lambda k: dist(F1, F2, k))
        out.append((i / 16000.0, v, dist(F1, F2, v), F1, F2))
    return out


def runs_of(fp):
    ws = windows(fp)
    out, cur = [], None
    for t, v, d, F1, F2 in ws:
        if cur and cur["v"] == v and t - cur["last"] <= HOP + 1e-6:
            cur["last"] = t
            cur["ds"].append(d)
            cur["f"].append((F1, F2))
        else:
            if cur:
                out.append(cur)
            cur = {"clip": fp, "v": v, "t0": t, "last": t, "ds": [d],
                   "f": [(F1, F2)]}
    if cur:
        out.append(cur)
    for r in out:
        r["t1"] = r["last"] + WIN
        r["len"] = r["t1"] - r["t0"]
        r["d"] = float(np.mean(r["ds"]))
        r["F1"] = float(np.median([x[0] for x in r["f"]]))
        r["F2"] = float(np.median([x[1] for x in r["f"]]))
        del r["f"], r["ds"], r["last"]
    return out


def main():
    files = sorted(glob.glob(f"{C}/*.wav"))
    print(f"掃 {len(files)} clips（併連續段）…", flush=True)
    R = []
    for fp in files:
        R += runs_of(fp)
    pick = {}
    for v in ORDER:
        cs = [r for r in R if r["v"] == v and r["len"] >= MINRUN]
        if not cs:
            cs = [r for r in R if r["v"] == v]
            print(f"⚠ {v} 沒有 ≥{MINRUN}s 的穩定段，退而取最長的")
        cs.sort(key=lambda r: (r["d"], -r["len"]))
        pick[v] = cs[0]
    print(f"\n{'母音':<4}{'clip':<16}{'窗':<14}{'長':>5}{'F1':>6}{'F2':>7}"
          f"{'d':>6}")
    lines = []
    for v in ORDER:
        r = pick[v]
        mid = (r["t0"] + r["t1"]) / 2
        t0, t1 = max(0.0, mid - 0.5), mid + 0.5
        nm = r["clip"].split("/")[-1]
        print(f"{v:<4}{nm:<16}{r['t0']:.1f}-{r['t1']:.1f}s{'':<3}"
              f"{r['len']:>5.1f}{r['F1']:>6.0f}{r['F2']:>7.0f}{r['d']:>6.2f}")
        lines.append(f'                ("{v}", f"{{_C0619}}/{nm}", '
                     f'{t0:.2f}, {t1:.2f}, 0.0),')
    print("\n貼進 bank_live.py 的 VOWEL_LAYERS：")
    print("VOWEL_LAYERS = [" + "\n".join(lines)[16:] + "]")
    json.dump({v: pick[v] for v in ORDER},
              open("scratchpad/vowel_layer_pick.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
