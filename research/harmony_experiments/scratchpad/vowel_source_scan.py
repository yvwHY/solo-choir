"""vowel_source_scan — 依「母音」挑素材，不再依亮度（08-13）

驗屍結論（layer_vowel_audit）：現行五層是照 spectral tilt 排序挑的，
素材端實測母音＝[嗚/喔, 喔, 咿, 喔, 啊]——**沒有欸、喔佔三席**，
標籤（喔欸咿嗚啊）純屬順序假設。Harry 的三次耳判逐條對上：
「咿啊順」（L2 真的是咿、L4 真的是啊）、「欸分不出來」（根本沒有欸層）、
「喔嗚沒試出來」（兩層其實是同一類）、「喔聽起來像欸」（L0 渲後 F2
785→1322 被推亮）。

⚠ 關鍵：**渲染會改變母音**（源 L3 喔 F2 829 → 渲後 1877＝欸）。所以
挑素材不能只看源端，要**渲完再驗一次**（本檔只做第一步：源端候選；
第二步渲染驗收在 vowel_layer_rebuild.py）。

本檔：掃 0619 語料，對每個 300ms 窗量 F1/F2（他自己的嗓、低音域＝LPC
可靠）→ 對五個目標母音各挑最像、最穩、夠長的候選段 → 存 JSON 供渲染。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_source_scan.py
"""
import glob
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_ns = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _ns)
lpc_formants, REF = _ns["lpc_formants"], _ns["REF"]
C = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
     "260618/test/man/0619_clean_clips")
WIN, HOP = 0.30, 0.15                    # 秒


def dist(f1, f2, v):
    r = REF[v]
    return float(np.hypot(np.log(f1 / r[0]), np.log(f2 / r[1])))


def main():
    files = sorted(glob.glob(f"{C}/*.wav"))
    print(f"掃 {len(files)} 個 clip…", flush=True)
    rows = []
    for fp in files:
        x, fs = sf.read(fp, always_2d=True)
        x = x[:, 0].astype("float64")
        if len(x) < int(WIN * fs):
            continue
        x16 = resample_poly(x, 16000, fs)
        w, h = int(WIN * 16000), int(HOP * 16000)
        for i in range(0, len(x16) - w, h):
            seg = x16[i:i + w]
            rms = float(np.sqrt((seg ** 2).mean()))
            if rms < 0.02:                     # 無聲/氣音窗跳過
                continue
            # 穩定性：前後半段的共振峰要一致（過渡段不要）
            fa = lpc_formants(seg[:w // 2])
            fb = lpc_formants(seg[w // 2:])
            if len(fa) < 2 or len(fb) < 2:
                continue
            if (abs(fa[0] - fb[0]) > 60 or abs(fa[1] - fb[1]) > 180):
                continue
            F1, F2 = (fa[0] + fb[0]) / 2, (fa[1] + fb[1]) / 2
            best = min(REF, key=lambda v: dist(F1, F2, v))
            rows.append({"clip": fp, "t0": i / 16000.0,
                         "t1": (i + w) / 16000.0, "F1": F1, "F2": F2,
                         "vowel": best, "d": dist(F1, F2, best),
                         "rms": rms})
    print(f"有效窗 {len(rows)}", flush=True)
    out = {}
    for v in REF:
        # 取 30（不是 8）——挑素材的判準是**渲後**像不像，源頭只是初篩，
        # 所以候選池要夠大給 vowel_render_select 抽（08-13 血訓：源端
        # d=0.11 的嗚渲完照樣歪，語料某段反而乾淨＝這件事像抽籤）
        cs = sorted([r for r in rows if r["vowel"] == v],
                    key=lambda r: r["d"])[:30]
        out[v] = cs
        print(f"\n{v}（目標 F1 {REF[v][0]} F2 {REF[v][1]}）候選 {len(cs)}：")
        for r in cs[:4]:
            print(f"  {r['clip'].split('/')[-1]} {r['t0']:.1f}-{r['t1']:.1f}s"
                  f"  F1 {r['F1']:.0f} F2 {r['F2']:.0f}  d {r['d']:.2f}")
    json.dump(out, open("scratchpad/vowel_source_cands.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n→ scratchpad/vowel_source_cands.json")


if __name__ == "__main__":
    main()
