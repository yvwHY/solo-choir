"""layer_vowel_audit — 五層渲出來到底是什麼母音？（08-13）

Harry 三層輪判「喔聲音聽起來像是誒」＝踩到 v16.3 起沒人驗過的假設：
層↔母音對應**只是按亮度順序套上去的**（0619 素材依 spectral tilt 排序
→ 直接假設 dk2=喔 dk1=欸 mid=咿 br1=嗚 br2=啊），素材本身唱的是什麼
母音從來沒查。嘴形認得對、庫選得對，但**庫本身標錯**＝口型與聲音對不上。

本檔兩件事：
 1. 客觀：對每層渲出的 loop 做 LPC 共振峰（降 16k、order 18）→ F1/F2，
    對照標準母音空間給假設。
 2. 主觀（權威）：產生**盲聽包** layer_audit.wav——只報層號不報母音名
    （防聽混/防暗示，08-13 血訓），每層 2 秒，讓 Harry 耳標。
    → 他標完，映射照他的答案重建（音質歸他）。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/layer_vowel_audit.py [--voice alto] [--midi 62]
"""
import argparse
import glob
import os
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import lfilter, resample_poly

SR = 44100
# 標準母音 F1/F2（男聲概略，Hz）——只當對照，不當判決
REF = {"啊": (730, 1090), "喔": (500, 700), "嗚": (320, 800),
       "欸": (530, 1840), "咿": (270, 2290)}


def lpc_formants(x, order=18, fs=16000):
    x = x - x.mean()
    x = lfilter([1, -0.97], [1], x)                 # 預強調
    w = x * np.hanning(len(x))
    r = np.correlate(w, w, "full")[len(w) - 1:][:order + 1]
    if r[0] <= 0:
        return []
    a, e = np.zeros(order + 1), r[0]                # Levinson-Durbin
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = r[i] + sum(a[j] * r[i - j] for j in range(1, i))
        k = -acc / e
        an = a.copy()
        for j in range(1, i):
            an[j] = a[j] + k * a[i - j]
        an[i] = k
        a = an
        e *= (1 - k * k)
        if e <= 0:
            break
    rts = [z for z in np.roots(a) if np.imag(z) > 0.01]
    f = sorted(float(np.arctan2(np.imag(z), np.real(z)) * fs / (2 * np.pi))
               for z in rts)
    bw = {}
    for z in rts:
        fz = float(np.arctan2(np.imag(z), np.real(z)) * fs / (2 * np.pi))
        bw[round(fz)] = -0.5 * (fs / (2 * np.pi)) * np.log(abs(z))
    return [x_ for x_ in f if 200 < x_ < 4000 and bw.get(round(x_), 1e9) < 500]


def nearest_vowel(f1, f2):
    d = {k: ((np.log(f1 / v[0])) ** 2 + (np.log(f2 / v[1])) ** 2) ** 0.5
         for k, v in REF.items()}
    o = sorted(d, key=d.get)
    return o[0], o[1], d[o[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="alto")
    ap.add_argument("--midi", type=int, default=62)
    ap.add_argument("--out", default="scratchpad/layer_audit.wav")
    ap.add_argument("--dir", default="", help="指定層庫目錄（空=最新）")
    a = ap.parse_args()

    cands = glob.glob(f"scratchpad/bank_*/{a.voice}_L0.npz")
    assert cands, "找不到層庫（先跑一次 --vowels 1）"
    # ⚠ 用 mtime 挑最新——檔名是雜湊，字典序跟新舊無關（08-13 踩過：
    # 盲聽包差點用舊庫生成）。要指定就給 --dir。
    bd = a.dir or os.path.dirname(max(cands, key=os.path.getmtime))
    print(f"層庫目錄 {bd}\n")

    print(f"{'層':<4}{'F1':>6}{'F2':>7}   {'最近母音':<10}{'次近':<8}")
    segs = []
    for li in range(5):
        z = np.load(f"{bd}/{a.voice}_L{li}.npz")
        key = str(a.midi) if str(a.midi) in z.files else z.files[
            len(z.files) // 2]
        loop = z[key].astype("float64")
        x16 = resample_poly(loop, 16000, SR)
        n = min(len(x16), 1600)
        fs_ = [lpc_formants(x16[i:i + n]) for i in range(0, len(x16) - n, n)]
        fs_ = [f for f in fs_ if len(f) >= 2]
        F1 = float(np.median([f[0] for f in fs_])) if fs_ else float("nan")
        F2 = float(np.median([f[1] for f in fs_])) if fs_ else float("nan")
        v1, v2, _d = nearest_vowel(F1, F2)
        print(f"L{li:<3}{F1:>6.0f}{F2:>7.0f}   {v1:<10}{v2:<8}")

        # 盲聽包：只報層號（不報母音名＝不暗示）
        aif = f"/tmp/_lay{li}.aiff"
        subprocess.run(["say", "-o", aif, f"第{li + 1}個"],
                       check=True)
        sp, sr_ = sf.read(aif, dtype="float32", always_2d=True)
        sp = resample_poly(sp[:, 0], SR, sr_).astype("float32")
        body = np.tile(loop, int(2.0 * SR / len(loop)) + 1)[:int(2.0 * SR)]
        body = (body / (np.abs(body).max() + 1e-9) * 0.5).astype("float32")
        segs += [sp * 0.6, np.zeros(int(0.25 * SR), "float32"), body,
                 np.zeros(int(0.6 * SR), "float32")]
        os.remove(aif)
    sf.write(a.out, np.concatenate(segs), SR)
    print(f"\n盲聽包 → {a.out}（每段：報層號 → 2 秒該層聲音）")


if __name__ == "__main__":
    main()
