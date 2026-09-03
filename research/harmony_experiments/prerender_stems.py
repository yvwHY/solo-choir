"""prerender_stems.py — 排練檔 → 離線 express 品質天使 stems（預渲染的離線半場）

方向（2026-08-01 Harry「優化天使、不是優化 live_v3」）：排練過的歌，天使
音訊可以完全離開即時約束——用離線正典 direct_mouth（--f0-mode express＝
F21 配方、uv 透傳、輸入 AGC 全預設）把排練檔的兩條天使線渲染成與排練
take 同時間軸的 stems。live 端（live_v3 --pre-stems）按譜追蹤器對位、
寫進未來格＝離線天花板品質直接上 live，嘴管線的即時妥協全數不存在。

driver 固化（07-30 血訓：agent 臨時 driver 沒留＝跨表無裁決效力）。

Run（DDSP venv）:
  ../../../260724_ddsp_svc/venv/bin/python prerender_stems.py \
      --rehearse out/pair06_reh_notes.json \
      --take ../../../260730_recording/pairs/pair06_take1.wav --tag pair06_pre

產物：out/{tag}_{voice}_angel.wav（44.1k mono，時間軸＝take）
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DDSP = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260724_ddsp_svc"
TICK_SAMPS = int((60.0 / 80.0) * 0.25 * 44100)   # 8268＝world_live 的腦脈搏

# 同 live_v3 預設嘴（upper=girl、lower=260730 重訓）＋ per-voice seed 去同步
VOICES = {"upper": (f"{DDSP}/exp/combsub-girl/model_30000.pt", 20260731),
          "lower": (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt", 20260732)}
# target 模式的顫音去同步（render_v3.py:339-340 原值）。08-02 F21 修正：
# express 只在獨唱驗過，兩個聲部同時發聲會拆掉和聲（音程誤差 SD 40.7c）。
VIB = {"upper": (5.3, 0.0), "lower": (4.6, 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", required=True,
                    help="live_v3 反應式 file 落的 *_notes.json（upper/lower/lead）")
    ap.add_argument("--take", required=True, help="排練 take（stems 與它同時間軸）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expr-gain", type=float, default=1.0)
    ap.add_argument("--f0-mode", choices=["target", "express"], default="target",
                    help="target＝骨架＋固定顫音去同步（08-02 F21 修正後的雙"
                         "聲部配方）；express＝07-31 表現層（獨唱才對）")
    ap.add_argument("--ceiling", action="store_true",
                    help="順手輸出天花板混音 out/{tag}_ceiling_mix.wav"
                         "（take 0.5＋兩天使各 0.6，同 live_v3 收官配方）＝"
                         "對位不參與的純 stems 品質判準")
    a = ap.parse_args()
    reh = json.load(open(a.rehearse))
    for v, (model, seed) in VOICES.items():
        nj = os.path.join(HERE, "out", f"{a.tag}_{v}_notes.json")
        json.dump({"notes": reh[v], "step_samps": TICK_SAMPS}, open(nj, "w"))
        cmd = [sys.executable, os.path.join(HERE, "direct_mouth.py"),
               "--notes", nj, "--take", a.take, "--tag", f"{a.tag}_{v}",
               "--model", model, "--key", "0", "--f0-mode", a.f0_mode, "--run"]
        if a.f0_mode == "express":
            cmd += ["--expr-gain", str(a.expr_gain), "--expr-seed", str(seed)]
        else:
            cmd += ["--vib-hz", str(VIB[v][0]), "--vib-phase", str(VIB[v][1]),
                    "--vib-onset-ms", "250"]
        print(">>", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode:
            sys.exit(r.returncode)
    print("stems:", ", ".join(f"out/{a.tag}_{v}_angel.wav" for v in VOICES))

    if a.ceiling:
        import numpy as np
        import soundfile as sf
        take, sr = sf.read(a.take, dtype="float64", always_2d=True)
        take = take[:, 0]
        ang = [sf.read(os.path.join(HERE, "out", f"{a.tag}_{v}_angel.wav"),
                       dtype="float64")[0] for v in VOICES]
        n = min(len(take), *(len(w) for w in ang))
        mix = 0.5 * take[:n] + 0.6 * ang[0][:n] + 0.6 * ang[1][:n]
        p = os.path.join(HERE, "out", f"{a.tag}_ceiling_mix.wav")
        sf.write(p, mix / (np.abs(mix).max() + 1e-12) * 0.9, sr, subtype="PCM_16")
        print("ceiling:", p)


if __name__ == "__main__":
    main()
