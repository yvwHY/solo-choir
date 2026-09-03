"""vowel_record_ritual — 刻意錄母音長音當層素材，錄完立刻渲立刻驗（08-13）

Harry 提案「那我再錄音一次嗚呢」。分辨兩件事：
  ①模型性格：這顆嘴吐不出圓唇後母音（喔 0/4、嗚 1/4 存活）
  ②素材品質：0619 是唱歌中途的母音，有共構牽連、段落短
刻意錄的穩定長音是最快的分辨器——而且對每個母音都是升級（現行欸的
素材只是勉強及格 d=0.15）。

流程（血訓內建）：
  報幕用描述詞（防聽混）→ 錄 4s → **立刻驗**（RMS 有聲＋共振峰對目標
  的距離）→ 不合格當場重錄（最多 3 次）→ 全部錄完後**逐個渲一顆音再量**
  （渲染會改變母音，源端對不算數）→ 只留渲後仍是該母音的，印出可貼進
  bank_live 的 VOWEL_LAYERS。

跑: cd harmony && PYTHONPATH=. ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_record_ritual.py [--vowels 喔,嗚] [--seconds 4]
"""
import argparse
import os
import subprocess
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, nearest_vowel, REF = (_a["lpc_formants"], _a["nearest_vowel"],
                                    _a["REF"])
SR, HOP = 44100, 512
MIDI = 48
MODEL = "reflow-bass1/model_32000.pt"
ORDER = ["喔", "欸", "咿", "嗚", "啊"]
SAY = {"喔": "圓嘴的喔，像 oh", "欸": "扁嘴的欸，像 eh",
       "咿": "咧嘴的咿，像 ee", "嗚": "嘟嘴的嗚，像 oo",
       "啊": "張大的啊，像 ah"}
OUT = "scratchpad/vowel_takes"


def say(t):
    subprocess.run(["say", t])


def meas(x, fs=SR, lo=0.0, hi=None):
    x16 = resample_poly(x, 16000, fs)
    n = 1600
    f = [lpc_formants(x16[i:i + n]) for i in range(0, max(1, len(x16) - n), n)]
    f = [q for q in f if len(q) >= 2]
    if not f:
        return float("nan"), float("nan")
    return (float(np.median([q[0] for q in f])),
            float(np.median([q[1] for q in f])))


def steadiest(x, win=1.0):
    """回最穩的 win 秒窗（相鄰 0.25s 段共振峰變異最小）。"""
    w, h = int(win * SR), int(0.25 * SR)
    best, bi = None, 0
    for i in range(0, max(1, len(x) - w), h):
        seg = x[i:i + w]
        fs_ = [lpc_formants(resample_poly(seg[j:j + int(0.25 * SR)],
                                          16000, SR))
               for j in range(0, w - int(0.25 * SR), int(0.25 * SR))]
        fs_ = [q for q in fs_ if len(q) >= 2]
        if len(fs_) < 3:
            continue
        v = (np.std([q[0] for q in fs_]) / 60.0
             + np.std([q[1] for q in fs_]) / 180.0)
        if best is None or v < best:
            best, bi = v, i
    return bi / SR, (bi + w) / SR


def record(vow, seconds, dev):
    import sounddevice as sd
    for k in range(3):
        say(SAY[vow])
        time.sleep(0.3)
        say("唱")
        buf = []

        def cb(ind, n, ti, st):
            buf.append(ind[:, 0].copy())
        with sd.InputStream(samplerate=SR, channels=1, device=dev,
                            dtype="float32", callback=cb):
            time.sleep(seconds)
        x = np.concatenate(buf).astype("float64") if buf else np.zeros(1)
        rms = float(np.sqrt((x ** 2).mean()))
        F1, F2 = meas(x[int(0.5 * SR):])
        v1, _v2, d = nearest_vowel(F1, F2)
        print(f"[{vow}] RMS {20*np.log10(rms+1e-9):.1f}dBFS  "
              f"F1 {F1:.0f} F2 {F2:.0f} → 像 {v1}（對目標 d={d:.2f}）",
              flush=True)
        if rms < 0.005:
            say("沒收到聲音，重來")
            continue
        if d > 0.35:
            say("這個聽起來不太對，再一次")
            continue
        say("好")
        return x
    say("三次都不合格，跳過")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vowels", default="喔,欸,咿,嗚,啊")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--in-name", default="USB PnP")
    a = ap.parse_args()
    vows = [v.strip() for v in a.vowels.split(",") if v.strip()]
    os.makedirs(OUT, exist_ok=True)

    takes = {}
    say("母音素材錄音。每個拉長四秒，音高穩定就好。")
    for v in vows:
        x = record(v, a.seconds, a.in_name)
        if x is None:
            continue
        p = f"{OUT}/{v}.wav"
        sf.write(p, x.astype("float32"), SR)
        takes[v] = p
    if not takes:
        print("沒有可用素材")
        return

    print("\n── 渲染驗收（渲一顆 midi 48 再量）──", flush=True)
    import torch
    import spike_stream6 as S
    svc = S.Svc([(f"{S.DDSP}/exp/{MODEL}", 0.0, 1.0, 1)], step=2,
                t_start=0.85)
    NF = int(2.0 * SR / HOP)
    f0 = np.full(NF, 440.0 * 2 ** ((MIDI - 69) / 12.0))
    vol = np.full(NF, 0.06)
    vol[:4] = np.linspace(0, 0.06, 4)
    vol_t = torch.from_numpy(vol).float().to(svc.device)[None, :, None]
    mask = torch.ones(1, NF * HOP, device=svc.device)
    lines = []
    for v, p in takes.items():
        x, _sr = sf.read(p, always_2d=True)
        x = x[:, 0].astype("float64")
        t0, t1 = steadiest(x)
        seg = x[int(t0 * SR):int(t1 * SR)]
        with torch.no_grad():
            UU = svc.encode(seg)[:, 8:-8]
            nb = UU.size(1)
            torch.manual_seed(1234 + MIDI)
            au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                           units_override=UU[:, np.arange(NF) % nb],
                           feats=(f0, vol_t, mask), ratios=[None]
                           )[0].cpu().numpy()
        F1, F2 = meas(au[int(0.5 * SR):int(1.8 * SR)].astype("float64"))
        v1, v2, _d = nearest_vowel(F1, F2)
        ok = "✓" if v1 == v else ("~" if v2 == v else "✗")
        print(f"  {v}  穩定窗 {t0:.1f}-{t1:.1f}s  渲後 F1 {F1:.0f} "
              f"F2 {F2:.0f} → {v1}（次{v2}）{ok}", flush=True)
        if ok == "✓":
            lines.append(f'                ("{v}", "{os.path.abspath(p)}", '
                         f'{t0:.2f}, {t1:.2f}, 0.0),')
    print("\n渲後仍正確的層（可貼進 bank_live VOWEL_LAYERS）：")
    print("\n".join(lines) if lines else "  （無）")


if __name__ == "__main__":
    main()
