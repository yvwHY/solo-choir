"""vowel_acoustic_probe — 聲學母音路線的離線可行性實驗（08-13）

問題：視覺路線天花板＝舌位看不見（欸/咿 同唇形）＋live 嘴形落在錨點外
＝場糊掉（v171_r1 驗屍：熵 0.83、嗚 56%）。聲學通道帶完整母音資訊，
但 v16.2 死於天使漏音。未試的救法＝**頻譜扣除**：天使輸出是我們自己
生成的＝完全已知，從 mic 頻譜扣掉再認母音。

實驗設計（零新錄音）：
  訓練錨點：lip_sweep2.npz 的五母音音檔（08-13 凌晨錄，乾淨）
  測試資料：lip_ring_sweep.npz 的五母音音檔（08-13 中午錄，乾淨、獨立場次）
  三態：A 乾淨 test → 分類
        B test + 天使(取自 v171_r1_out) 1:1 RMS 混入 → 分類（模擬 live 漏音）
        C 同 B 但先做頻譜幅度扣除（已知天使訊號；含 ±20ms 失準敏感度）
  特徵：log-mel 帶能量（300–5000Hz、24 帶）逐幀減均值（位準不變）
  分類：對五錨點（訓練集逐幀特徵平均）cos 相似度最近鄰
判準：A 高＝聲學特徵本身能認他的母音（跨場次泛化）；B 崩＝復現漏音病；
C 恢復到接近 A＝扣除路線可行 → 值得上 live。
"""
import numpy as np
import soundfile as sf

SR = 44100
NFFT = 2048
HOPF = 1470                      # 30Hz 幀率（對齊鏡頭/控制通道節奏）
LABELS = ["喔", "欸", "咿", "嗚", "啊"]


def melfb(nmel=24, fmin=300.0, fmax=5000.0):
    def m(f):
        return 2595 * np.log10(1 + f / 700)

    def im(x):
        return 700 * (10 ** (x / 2595) - 1)
    pts = im(np.linspace(m(fmin), m(fmax), nmel + 2))
    bins = np.floor((NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((nmel, NFFT // 2 + 1))
    for i in range(nmel):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        if b > a:
            fb[i, a:b] = np.linspace(0, 1, b - a)
        if c > b:
            fb[i, b:c] = np.linspace(1, 0, c - b)
    return fb


FB = melfb()
WIN = np.hanning(NFFT)


def frames(x):
    n = (len(x) - NFFT) // HOPF
    return np.stack([x[i * HOPF:i * HOPF + NFFT] * WIN for i in range(n)])


def feats_from_power(P):
    """P=(T,nfft/2+1) 功率譜 → log-mel 減均值特徵。"""
    E = P @ FB.T
    L = np.log(E + 1e-10)
    return L - L.mean(1, keepdims=True)


def feats(x):
    F = np.fft.rfft(frames(x), axis=1)
    return feats_from_power(np.abs(F) ** 2)


def voiced(x):
    fr = frames(x)
    r = np.sqrt((fr ** 2).mean(1))
    return r >= 0.25 * np.percentile(r, 75)


def classify(F, anchors):
    A = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    return (Fn @ A.T).argmax(1)


def acc_report(tag, preds_true):
    ok = np.concatenate([(p == vi) for vi, p in enumerate(preds_true)])
    per = [f"{LABELS[vi]}{(p == vi).mean():.2f}"
           for vi, p in enumerate(preds_true)]
    print(f"{tag}: 總 {ok.mean():.2f}  逐母音 {' '.join(per)}", flush=True)
    return ok.mean()


def main():
    tr = dict(np.load("scratchpad/lip_sweep2.npz"))
    te = dict(np.load("scratchpad/lip_ring_sweep.npz"))
    ang, _ = sf.read("scratchpad/v171_r1_out.wav", always_2d=True)
    ang = ang.mean(1).astype("float64")
    # 挑天使有聲的一段（避免抽到靜音＝假輕鬆）
    ar = np.sqrt(np.convolve(ang ** 2, np.ones(4410) / 4410, "same"))
    ai = int(np.argmax(ar > np.percentile(ar, 90)))

    anchors = np.array([feats(tr[f"a{v}"].astype("float64"))[
        voiced(tr[f"a{v}"].astype("float64"))].mean(0) for v in range(5)])

    predA, predB, predC, predC20 = [], [], [], []
    rng = np.random.RandomState(20260813)
    for v in range(5):
        x = te[f"a{v}"].astype("float64")
        vm = voiced(x)
        # 天使段：每母音換一段（噪聲多樣性），等 RMS 混入
        seg = ang[ai + v * len(x): ai + (v + 1) * len(x)]
        if len(seg) < len(x):
            seg = np.tile(ang[ai:ai + len(x)], 2)[:len(x)]
        g = np.sqrt((x ** 2).mean() / max((seg ** 2).mean(), 1e-12))
        mix = x + seg * g
        predA.append(classify(feats(x)[vm], anchors))
        predB.append(classify(feats(mix)[vm], anchors))
        # C：功率譜扣除（天使訊號已知、逐幀對齊）；floor 0
        for lag, bucket in ((0, predC), (int(0.02 * SR), predC20)):
            known = np.roll(seg * g, lag)      # 模擬 live 對齊誤差
            Pm = np.abs(np.fft.rfft(frames(mix), axis=1)) ** 2
            Pk = np.abs(np.fft.rfft(frames(known), axis=1)) ** 2
            bucket.append(classify(
                feats_from_power(np.maximum(Pm - Pk, 0.0))[vm], anchors))

    acc_report("A 乾淨（跨場次泛化上限）", predA)
    acc_report("B 混天使 1:1（live 現況）", predB)
    acc_report("C 扣除・完美對齊", predC)
    acc_report("C 扣除・失準 20ms", predC20)


if __name__ == "__main__":
    main()
