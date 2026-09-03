"""prosody_ab.py — prosody 移植 A/B/B2（08-05；Harry:「有 auto tune 就是不行」）

⚠ 08-05 素材翻案（Harry 抓到）：本檔 `[0::2]` 選段選到的是 dump 裡的
**回應段**（他的重播＋兩天使同軌）＝三聲部混音被當成他的句子；
out/prosody_ab/ 全部產物與量測作廢。重審走 bare_test.py（xcorr 選段，
out/clean_ab/），詳 worklog 08-05 §C。本檔僅留案底，勿再執行。

診斷（08-05 凌晨，儀器＋girl 對照 Harry 已確認）：autotune 不在模型層，在
表現層——天使 f0 是量化譜位合成的（每音平線＋33ms 跳接＋定速顫音），
他真實的音高曲線被整條丟棄（幀間活動 13c → 5c）。

修法＝把他真實 f0 的偏差曲線移植到骨架上（`phrase_render prosody_cents`）：
- dev(t) ＝ 他的 f0（MIDI）− 他自己的量化線（hold 補洞）＝勾滑、漂、樂句
- 移動平均去掉他的顫音（快成分仍由既有去同步顫音模板出）＋clamp ±60c
- **兩聲部共用同一條**＝音程恆定，08-02 express 的死因（各偏各的）構造上不可能
- B＝0.29s 窗（只留樂句慢成分；首輪量測顯示勾滑被砍半＝送到一半的人味）
  B2＝0.12s 窗（保留 0.1–0.3s 的勾滑帶）

產出（素材＝16:00 場兩句，同前兩輪 trial 可對聽）：
  segN_mix_{A,B,B2}.wav          A＝現行配方
  segN_bass1_lower_{A,B,B2}.wav  裸 stem 近聽用

Run (DDSP venv):  python prosody_ab.py
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

H = Path(__file__).parent
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402
from live_v3 import EarV3  # noqa: E402

B1 = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
      "260724_ddsp_svc/exp/combsub-m4-bass1/model_11000.pt")
OUT = H / "out" / "prosody_ab"
GU = 0.75 * 10 ** (1.0 / 20.0)     # 現行混音：上聲部 +1 dB
GL = 0.75
DRY = 0.944
MA_SLOW, MA_SCOOP = 25, 10         # 0.29s／0.12s 移動平均窗
CLAMP = 60.0


def prosody_curve(f0m, n_hops, ma=MA_SLOW):
    """他的 f0（harvest, per-hop Hz）→ 偏差曲線（cents, per-hop）。"""
    m = np.full(n_hops, np.nan)
    k = min(n_hops, len(f0m))
    v = f0m[:k] > 0
    m[:k][v] = 69 + 12 * np.log2(f0m[:k][v] / 440.0)
    line = np.copy(m)
    last = np.nan
    for i in range(n_hops):
        if np.isnan(line[i]):
            line[i] = last
        else:
            last = line[i]
    line = np.round(line)
    dev = (m - line) * 100.0
    dev[np.isnan(dev)] = 0.0        # 無聲段＝不加料
    ker = np.ones(ma) / ma
    dev = np.convolve(dev, ker, mode="same")
    return np.clip(dev, -CLAMP, CLAMP)


def motion(w):
    f0 = dm.harvest_f0(np.ascontiguousarray(w.astype(np.float64)))
    v = f0[f0 > 0]
    mm = 69 + 12 * np.log2(v / 440.0)
    return float(np.median(np.abs(np.diff(mm)) * 100))


def main():
    OUT.mkdir(exist_ok=True)
    import torch
    x, _ = sf.read(str(H / "out/resp2_live_260804_160058.wav"),
                   dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs = []
    for s, e in R.find_phrases(x, 0.35, 0.8):
        if s / SR < 12:
            continue
        f0 = dm.harvest_f0(np.ascontiguousarray(x[s:e]))
        v = f0[f0 > 0]
        if len(v) < 30:
            continue
        if float(np.mean(v > 340.0)) < 0.05:
            segs.append(np.ascontiguousarray(x[s:e]))
    segs = sorted(segs[0::2], key=len, reverse=True)[:2]

    torch.manual_seed(1234)
    ear = EarV3(indep=0.15, stab=1, key=0)
    ear.v2t.shift = 12
    kt = R.KeyTracker()
    rend = {"upper": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                    expr_seed=R.VOICES["upper"][1]),
            "lower": PhraseRenderer(R.DDSP, B1, expr_seed=20260805)}
    for r_ in rend.values():
        r_.render(np.zeros(SR), [60], TICK_SAMPS)

    for i, seg in enumerate(segs):
        kt.push(dm.harvest_f0(seg))
        rb = kt.best()
        if rb is not None and rb[2] >= 0.015:
            ear.k_shift = rb[0]
        lead, up, lo = R.phrase_notes(ear, seg)
        d = {"upper": up, "lower": lo}
        f0m = dm.harvest_f0(seg)
        n_hops = len(seg) // dm.HOP + 1
        curves = {"A": None,
                  "B": prosody_curve(f0m, n_hops, MA_SLOW),
                  "B2": prosody_curve(f0m, n_hops, MA_SCOOP),
                  "C": prosody_curve(f0m, n_hops, 1),   # 全移植含他的顫音
                  "D": "dlead",                          # 連續 f0×音程比（見下）
                  "E": "dlead+sus"}                      # D＋短空隙橋接 250ms
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        res = {}
        for ver, pc in curves.items():
            sus = isinstance(pc, str) and "sus" in pc
            if isinstance(pc, str):      # D/E：dev＝他的連續 f0 − 腦的 lead 線
                m = np.full(n_hops, np.nan)
                kk = min(n_hops, len(f0m)); vv = f0m[:kk] > 0
                m[:kk][vv] = 69 + 12*np.log2(f0m[:kk][vv]/440.0)
                ll = np.full(n_hops, np.nan)
                off = ear.v2t.shift + (ear.k_shift or 0)   # 腦空間→mic 空間
                for h in range(n_hops):
                    t_i = min(int(h*dm.HOP/TICK_SAMPS), len(lead)-1)
                    if lead[t_i] is not None: ll[h] = lead[t_i] - off
                last = np.nan
                for h in range(n_hops):
                    if np.isnan(ll[h]): ll[h] = last
                    else: last = ll[h]
                pc = (m - ll) * 100.0
                pc[np.isnan(pc)] = 0.0
                pc = np.clip(pc, -250, 250)   # 只擋八度抓錯，不砍滑行
            extra = {} if pc is None else {"prosody_cents": pc}
            if sus:
                extra["sustain_ms"] = 250.0
            kw_v = kw
            if ver in ("C", "D", "E"):   # 顫音已在他的曲線裡＝關掉合成模板
                kw_v = {v: dict(kw[v], vib_semi=0.0) for v in R.VOICES}
            sh_v = {"f0m": f0m}
            ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh_v,
                                     **kw_v[v], **extra) for v in R.VOICES}
            n = min(len(seg), *(len(a) for a in ang.values()))
            mix = DRY * seg[:n] + GU * ang["upper"][:n] + GL * ang["lower"][:n]
            sf.write(str(OUT / f"seg{i}_mix_{ver}.wav"),
                     mix / (np.abs(mix).max() + 1e-9) * 0.7, SR,
                     subtype="PCM_16")
            st = ang["lower"][:n]
            sf.write(str(OUT / f"seg{i}_bass1_lower_{ver}.wav"),
                     st / (np.abs(st).max() + 1e-9) * 0.7, SR,
                     subtype="PCM_16")
            res[ver] = st
        print(f"seg{i}: 幀間 f0 活動 dry {motion(seg):.1f}c ｜ "
              + " ".join(f"{v} {motion(w):.1f}c" for v, w in res.items())
              + f" ｜ 注入幅度 p50 B {np.median(np.abs(curves['B'])):.1f}c"
                f" B2 {np.median(np.abs(curves['B2'])):.1f}c")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
