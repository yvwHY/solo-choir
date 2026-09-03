"""bare_test.py — autotune 案重審：乾淨素材 × 全階梯（08-05 v2）

**素材翻案（Harry 抓到的）**：respond2 的 session dump 是「他被聽見的＋
播出去的同軌」（respond2.py:850）＝他的獨唱句與回應段（他的重播＋兩天使）
在同一條 mono 軌上交錯。舊選段 `[0::2]` 的奇偶對位被前置過濾打歪——
互相關鑑定（回應段必含上一句的數位重播＝xcorr 高）證實：prosody_ab／
vs_male3_trial／裸測 v1 選中的 seg0/seg1（103.7s、123.7s）**都是回應段**
＝三聲部混音被當成他的句子餵進 harvest/HuBERT＝「不穩、亂掉」的來源。
A–E／裸測 v1／正典 v1 的所有判決與量測作廢（worklog 08-05 §C）。

本檔 v2＝在真獨唱句上重跑整條階梯（產物 `out/clean_ab/`）：
  seg{i}_dry.wav                 他的獨唱句（先確認：單人聲、乾淨）
  seg{i}_A_mix / _A_bass1_lower  現行配方基線（骨架量化；原始 autotune 被告）
  seg{i}_bare_harry_ident        他的 harvest f0 ×1 直進他自己的模型（dm 馬具）
  seg{i}_bare_bass1_oct / _girl_up / _bare_mix   八度下/上與裸混音
  （canon＝main.py 直跑另由 shell 產：seg{i}_canon_*.wav）

判讀階梯：dry → canon（正典上限）→ bare（dm 馬具）→ A（全配方）——
autotune／亂掉在哪一階出現，罪就在那一階新增的東西上。

Run (DDSP venv):  python bare_test.py
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
from prosody_ab import B1, GU, GL, DRY, motion  # noqa: E402

OUT = H / "out" / "clean_ab"
DUMP = H / "out/resp2_live_260804_160058.wav"


def his_phrases(x, k=2, min_s=3.0, xc_resp=0.5, xc_self=0.35):
    """挑真獨唱句。分類器＝與前一段的正規化互相關峰值：回應段含上一句的
    數位重播（實測 0.56–0.95），他的句子與前段無關（實測 ≤0.30）。
    再要求「下一段是回應」＝這句真的被腦聽見並回應過。"""
    from scipy.signal import fftconvolve
    ph = [(s, e) for s, e in R.find_phrases(x, 0.35, 0.8)]
    xc = [0.0]
    for i in range(1, len(ph)):
        a = x[ph[i][0]:ph[i][1]]
        b = x[ph[i - 1][0]:ph[i - 1][1]]
        a = a / (np.sqrt((a ** 2).mean()) + 1e-9)
        b = b / (np.sqrt((b ** 2).mean()) + 1e-9)
        c = fftconvolve(a, b[::-1], mode="full")
        xc.append(float(np.abs(c).max() / min(len(a), len(b))))
    picks = [i for i in range(len(ph) - 1)
             if xc[i] <= xc_self and xc[i + 1] >= xc_resp
             and (ph[i][1] - ph[i][0]) / SR >= min_s]
    picks = sorted(picks, key=lambda i: ph[i][0] - ph[i][1])[:k]
    return [np.ascontiguousarray(x[ph[i][0]:ph[i][1]]) for i in picks], \
        [(ph[i][0] / SR, ph[i][1] / SR) for i in picks]


def bare_render(r, x, ratio, sh):
    """他的 f0 × 固定倍率直接進模型。phrase_render.render 拆掉骨架後的殘骸：
    每一步（AGC/vm/哨兵/vol mask/÷agc_g/uv 靜音）逐行對齊，只差 f0 來源。"""
    torch = r.torch
    n_hops = len(x) // dm.HOP + 1
    if "f0m" not in sh:
        sh["f0m"] = dm.harvest_f0(x)
    f0m = sh["f0m"]
    if "agc" not in sh:
        sh["agc"] = dm.input_agc(x, f0m=f0m)
    x_in, agc_g = sh["agc"]
    if "vm" not in sh:
        sh["vm"] = dm.voicing_mask(x, n_hops, f0m=f0m)
    vm = sh["vm"]
    f0 = np.zeros(n_hops)
    k = min(n_hops, len(f0m))
    f0[:k] = f0m[:k] * ratio
    f0 = np.where(f0 * vm > 0, f0 * vm, 1200.0)   # uv 哨兵，同 phrase_render
    with torch.no_grad():
        if sh.get("units") is None:
            au = torch.from_numpy(x_in).float().unsqueeze(0).to(r.device)
            sh["units"] = r.encoder.encode(au, SR, dm.HOP)
            sh["vol"] = r.vol_ex.extract(x_in)
        units, vol_all = sh["units"], sh["vol"]
        n = min(units.size(1), n_hops)
        vol = vol_all[:n]
        mask = r._vol_mask(vol)[:n]
        f0_t = torch.from_numpy(f0[:n]).float().to(r.device)[None, :, None]
        vol_t = torch.from_numpy(np.asarray(vol)).float().to(r.device)[None, :, None]
        out, _, _ = r.model(units[:, :n], f0_t, vol_t, spk_id=r.spk)
        wav = out.squeeze().cpu().numpy().astype(np.float64)
    m_up = np.repeat(mask, dm.HOP)[: len(wav)]
    wav[: len(m_up)] *= m_up
    kk = min(len(wav), len(agc_g))
    wav = wav[:kk] / agc_g[:kk]
    ang, _ = dm.uv_passthrough(wav, x[:kk], vm, dry_gain=0.0)  # 同 AB stems
    return ang


def wr(name, w):
    sf.write(str(OUT / name), w / (np.abs(w).max() + 1e-9) * 0.7, SR,
             subtype="PCM_16")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "src").mkdir(exist_ok=True)
    import torch
    x, _ = sf.read(str(DUMP), dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs, ts = his_phrases(x)
    for (a, b) in ts:
        print(f"seg @ {a:.1f}-{b:.1f}s（獨唱句，xcorr 鑑定）")

    torch.manual_seed(1234)
    ear = EarV3(indep=0.15, stab=1, key=0)
    ear.v2t.shift = 12
    kt = R.KeyTracker()
    rend = {"harry": PhraseRenderer(R.DDSP, R.VOICES["lower"][0]),
            "girl": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                   expr_seed=R.VOICES["upper"][1]),
            "bass1": PhraseRenderer(R.DDSP, B1, expr_seed=20260805)}
    plan = [("harry_ident", "harry", 1.0),
            ("bass1_oct", "bass1", 0.5),
            ("girl_up", "girl", 2.0)]

    for i, seg in enumerate(segs):
        sf.write(str(OUT / "src" / f"seg{i}_src.wav"), seg, SR, subtype="FLOAT")
        wr(f"seg{i}_dry.wav", seg)
        sh = {"f0m": dm.harvest_f0(seg)}
        # --- 裸測（dm 馬具、零骨架） ---
        res = {}
        for tag, mdl, ratio in plan:
            ang = bare_render(rend[mdl], seg, ratio, sh)
            res[tag] = ang
            wr(f"seg{i}_bare_{tag}.wav", ang)
        n = min(len(seg), *(len(a) for a in res.values()))
        wr(f"seg{i}_bare_mix.wav",
           DRY * seg[:n] + GU * res["girl_up"][:n]
           + GL * res["bass1_oct"][:n])
        # --- A 版基線（現行配方＝骨架量化；同 prosody_ab A） ---
        kt.push(sh["f0m"])
        rb = kt.best()
        if rb is not None and rb[2] >= 0.015:
            ear.k_shift = rb[0]
        lead, up, lo = R.phrase_notes(ear, seg)
        d = {"upper": up, "lower": lo}
        import json
        json.dump({v: [None if t is None else int(t) for t in d[v]]
                   for v in d},
                  open(OUT / f"seg{i}_notes.json", "w"))
        # 音符線落檔＝a_enh.py 等後續 A/B 吃同一條線（EarV3 有隨機性、對
        # 呼叫序敏感——單一變因對照必須鎖線）。
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        mouth = {"upper": "girl", "lower": "bass1"}
        angA = {v: rend[mouth[v]].render(seg, d[v], TICK_SAMPS, shared=sh,
                                         **kw[v]) for v in R.VOICES}
        n = min(len(seg), *(len(a) for a in angA.values()))
        wr(f"seg{i}_A_mix.wav",
           DRY * seg[:n] + GU * angA["upper"][:n] + GL * angA["lower"][:n])
        wr(f"seg{i}_A_bass1_lower.wav", angA["lower"][:n])
        print(f"seg{i}: 幀間 f0 活動 dry {motion(seg):.1f}c ｜ "
              + " ".join(f"{t} {motion(w):.1f}c" for t, w in res.items())
              + f" ｜ A_lower {motion(angA['lower'][:n]):.1f}c")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
