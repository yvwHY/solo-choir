"""vowel_render_select — 用「渲染後」的母音挑素材（08-13，v22）

血訓（今天最貴的一課）：**源端像不像不是判準**。Harry 現場錄的嗚源端
d=0.11（全場最準）渲完仍歪到 F2 1330；語料裡某段源端普通，渲完卻是
乾淨的嗚（221/907）。哪一段的 units 能活過模型比較像抽籤——那就別猜，
直接把候選全渲一次、用渲後距離挑贏家。

流程：vowel_source_cands.json（每母音 30 候選）＋現場錄音 → 每個渲一顆
midi 48 → 量 F1/F2 → 每母音取渲後距離最小者 → 印出 VOWEL_LAYERS。
沒有任何候選渲後落在該母音＝誠實判死（喔的候選人已陣亡 5 次）。

跑: cd harmony && PYTHONPATH=. ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_render_select.py [--per 30]
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, nearest_vowel, REF = (_a["lpc_formants"], _a["nearest_vowel"],
                                    _a["REF"])
_s = {}
exec(compile(open("scratchpad/vowel_source_scan.py").read()
             .split("\ndef main()")[0], "vss", "exec"), _s)
dist = _s["dist"]
SR, HOP, MIDI = 44100, 512, 48
F0HZ = 440.0 * 2 ** ((MIDI - 69) / 12.0)
DMAX = 0.25                              # 母音距離門檻（過了就比乾淨度）
MODEL = "reflow-bass1/model_32000.pt"
ORDER = ["喔", "欸", "咿", "嗚", "啊"]


def hnr(x, f0):
    """諧波對噪音比（dB）＝音質判準。08-13 血訓：v22 只用共振峰距離挑，
    四層 HNR 全面掉 1-6dB，Harry 當場聽出「不如上個」——**準確度與乾淨度
    要一起管**（同一個 clip 只差窗口位置就差 6dB）。"""
    x = x - x.mean()
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    hb = np.zeros(len(fr), bool)
    k = 1
    while k * f0 < 8000:
        hb |= np.abs(fr - k * f0) < max(f0 * 0.06, 12)
        k += 1
    band = (fr > 60) & (fr < 8000)
    h = X[hb & band].sum()
    return float(10 * np.log10(h / max(X[band].sum() - h, 1e-20)))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30)
    a = ap.parse_args()
    import torch
    import spike_stream6 as S
    cands = json.load(open("scratchpad/vowel_source_cands.json"))
    # v21 用過的窗（Harry 耳判較好的那版）明確放進候選池，讓它有機會
    # 憑 HNR 贏回來——不然搜索永遠只在新候選裡打轉
    for v, clip, t in (("欸", "clip_0143.wav", 2.17), ("咿", "clip_0155.wav",
                       3.97), ("嗚", "clip_0035.wav", 3.25),
                       ("啊", "clip_0009.wav", 1.43)):
        cands.setdefault(v, []).insert(0, {
            "clip": f"{_s['C']}/{clip}", "t0": t - 0.15, "t1": t + 0.15,
            "F1": 0.0, "F2": 0.0, "vowel": v, "d": 0.0, "rms": 0.0})
    # 現場錄音（vowel_record_ritual 產）也丟進候選池：整段掃窗
    for p in sorted(glob.glob("scratchpad/vowel_takes/*.wav")):
        v = os.path.basename(p)[:-4]
        if v not in cands:
            continue
        x, fs = sf.read(p, always_2d=True)
        dur = len(x) / fs
        for t0 in np.arange(0.3, max(0.4, dur - 1.2), 0.35):
            cands[v].append({"clip": p, "t0": float(t0),
                             "t1": float(t0 + 0.3), "F1": 0.0, "F2": 0.0,
                             "vowel": v, "d": 0.0, "rms": 0.0})

    svc = S.Svc([(f"{S.DDSP}/exp/{MODEL}", 0.0, 1.0, 1)], step=2,
                t_start=0.85)
    NF = int(2.0 * SR / HOP)
    f0 = np.full(NF, 440.0 * 2 ** ((MIDI - 69) / 12.0))
    vol = np.full(NF, 0.06)
    vol[:4] = np.linspace(0, 0.06, 4)
    vol_t = torch.from_numpy(vol).float().to(svc.device)[None, :, None]
    mask = torch.ones(1, NF * HOP, device=svc.device)

    best, lines = {}, []
    for v in ORDER:
        rows = []
        for r in cands[v][:a.per + 20]:
            x, fs = sf.read(r["clip"], always_2d=True)
            mid = (r["t0"] + r["t1"]) / 2
            a0 = max(0, int((mid - 0.5) * fs))
            seg = x[a0:a0 + int(1.0 * fs), 0].astype("float64")
            if len(seg) < int(0.6 * fs):
                continue
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
            body = au[int(0.5 * SR):int(1.8 * SR)].astype("float64")
            F1, F2 = meas(body)
            if not np.isfinite(F1) or not np.isfinite(F2):
                continue
            rows.append({"clip": r["clip"], "mid": mid, "F1": F1, "F2": F2,
                         "d": dist(F1, F2, v), "hnr": hnr(body, F0HZ),
                         "v": nearest_vowel(F1, F2)[0]})
        # 判準二選一不行：先用母音正確＋距離門檻篩，**再用 HNR 挑最乾淨**
        hit = sorted([q for q in rows if q["v"] == v and q["d"] <= DMAX],
                     key=lambda q: -q["hnr"])
        rows.sort(key=lambda q: q["d"])
        print(f"\n{v}：渲了 {len(rows)} 個，母音對且 d≤{DMAX} 的有 "
              f"{len(hit)} 個（依 HNR 排）", flush=True)
        for q in hit[:3]:
            print(f"   {os.path.basename(q['clip'])} @{q['mid']:.1f}s  "
                  f"渲後 {q['F1']:.0f}/{q['F2']:.0f}  d {q['d']:.2f}  "
                  f"HNR {q['hnr']:.1f}dB")
        if hit:
            b = hit[0]
            best[v] = b
            lines.append(f'                ("{v}", "{b["clip"]}", '
                         f'{b["mid"] - 0.5:.2f}, {b["mid"] + 0.5:.2f}, 0.0),'
                         f'  # 渲 {b["F1"]:.0f}/{b["F2"]:.0f} '
                         f'HNR {b["hnr"]:.0f}')
    print("\n── 渲後選出的層 ──")
    print("\n".join(lines) if lines else "（無）")
    for v in ORDER:
        if v not in best:
            print(f"⚠ {v}：{a.per}+ 個候選渲後沒有一個是{v}＝這顆嘴做不到")
    json.dump(best, open("scratchpad/vowel_render_best.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
