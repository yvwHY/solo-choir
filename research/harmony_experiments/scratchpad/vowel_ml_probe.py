"""vowel_ml_probe — 「要精細是不是要 ML」的離線判準實驗（08-13）

背景：兩個通道、兩種特徵，反距離最近鄰都卡在同一個天花板——
  視覺 v17.1 live 熵 0.83（糊）、聲學 mel 最近鄰跨場次 0.60。
假設：瓶頸不是感測資訊不足，是「五個原型點＋算距離」這個模型太弱。
本實驗：同樣的資料、同樣的切分，把最近鄰換成小分類器，看差多少。

⚠ 方法論（先寫死，免得自己騙自己）：
  1. **時間分塊 CV**，不做隨機切分——相鄰幀高度相關，隨機切分會把
     同一個嘴形同時放進 train/test＝準確率灌水。每段掃音由低到高，
     分塊等於也在考「換音高還認不認得」。
  2. **跨場次**才算真泛化：聲學支線 train=lip_sweep2（08-13 凌晨）、
     test=lip_ring_sweep（08-13 中午）。視覺支線只有中午一場有整圈
     landmark ⇒ 只能分塊 CV，報告時明講這條沒有跨場次證據。
  3. **基線用同一套切分**跑最近鄰（現行 v17.1 幾何），差值才可歸因。
  4. 真正要的是**權重**不是硬標籤 ⇒ 除了 accuracy 也報「真類機率
     平均」（soft），那才是餵給混層的東西。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_ml_probe.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SR = 44100
NFFT, HOPF = 2048, 1470                  # 30Hz 幀率（對齊控制通道節奏）
LABELS = ["喔", "欸", "咿", "嗚", "啊"]
NB = 5                                   # 時間分塊數

_src = open("scratchpad/lip_ring_calib.py").read()
_ns = {}
exec(compile(_src.split("def main()")[0], "lrc", "exec"), _ns)
ring_vec = _ns["ring_vec"]


def melfb(nmel=26, fmin=200.0, fmax=6000.0):
    def m(f):
        return 2595 * np.log10(1 + f / 700)

    def im(x):
        return 700 * (10 ** (x / 2595) - 1)
    pts = im(np.linspace(m(fmin), m(fmax), nmel + 2))
    b = np.floor((NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((nmel, NFFT // 2 + 1))
    for i in range(nmel):
        lo, mid, hi = b[i], b[i + 1], b[i + 2]
        if mid > lo:
            fb[i, lo:mid] = np.linspace(0, 1, mid - lo)
        if hi > mid:
            fb[i, mid:hi] = np.linspace(1, 0, hi - mid)
    return fb


FB = melfb()
WIN = np.hanning(NFFT)


def mfcc(x):
    """→ (T,13) MFCC c1-c13（丟 c0＝位準；DCT 低倒頻＝包絡、抑制音高梳。"""
    n = (len(x) - NFFT) // HOPF
    if n <= 0:
        return np.zeros((0, 13)), np.zeros(0)
    fr = np.stack([x[i * HOPF:i * HOPF + NFFT] * WIN for i in range(n)])
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    L = np.log(P @ FB.T + 1e-10)
    K = L.shape[1]
    D = np.cos(np.pi / K * (np.arange(K) + 0.5)[None, :]
               * np.arange(1, 14)[:, None])
    return L @ D.T, np.sqrt((fr ** 2).mean(1))


def blocks(n, nb=NB):
    """連續時間分塊索引（0..nb-1）。"""
    return np.minimum((np.arange(n) * nb) // max(n, 1), nb - 1)


def nn_baseline(Xtr, ytr, Xte):
    """現行 v17.1 幾何：類別均值當錨點、正規化 L1 距離最近鄰。"""
    A = np.stack([Xtr[ytr == c].mean(0) for c in range(5)])
    S = np.maximum(Xtr.std(0), 1e-6)
    d = np.abs((Xte[:, None, :] - A[None]) / S).mean(2)
    w = 1.0 / (d * d + 1e-6)
    P = w / w.sum(1, keepdims=True)
    return P.argmax(1), P


def fit_eval(Xtr, ytr, Xte, yte, kind):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "nn":
        pred, P = nn_baseline(Xtr, ytr, Xte)
    else:
        clf = (LogisticRegression(max_iter=2000, C=1.0)
               if kind == "lr" else
               MLPClassifier(hidden_layer_sizes=(32,), max_iter=1200,
                             random_state=0, alpha=1e-2))
        clf.fit(Xtr, ytr)
        P = clf.predict_proba(Xte)
        pred = P.argmax(1)
    return ((pred == yte).mean(), P[np.arange(len(yte)), yte].mean(),
            pred)


def blocked_cv(X, y, blk, kind):
    accs, softs, preds, trues = [], [], [], []
    for b in range(NB):
        te = blk == b
        if te.sum() == 0 or len(np.unique(y[~te])) < 5:
            continue
        a, s, p = fit_eval(X[~te], y[~te], X[te], y[te], kind)
        accs.append(a)
        softs.append(s)
        preds.append(p)
        trues.append(y[te])
    return (float(np.mean(accs)), float(np.mean(softs)),
            np.concatenate(preds), np.concatenate(trues))


def load_ring():
    z = dict(np.load("scratchpad/lip_ring_sweep.npz"))
    asp = float(z["aspect"])
    V, Aa, Y, B = [], [], [], []
    for vi in range(5):
        LA, ts, au = z[f"l{vi}"], z[f"t{vi}"], z[f"a{vi}"].astype("float64")
        M, rms = mfcc(au)
        thr = 0.25 * np.percentile(rms, 75)
        fi = np.clip((ts * SR / HOPF).astype(int), 0, len(M) - 1)
        ok = rms[fi] >= thr
        V.append(np.array([ring_vec(x, asp) for x in LA[ok]]))
        Aa.append(M[fi[ok]])
        Y.append(np.full(int(ok.sum()), vi))
        B.append(blocks(int(ok.sum())))
    return (np.concatenate(V), np.concatenate(Aa), np.concatenate(Y),
            np.concatenate(B))


def load_sweep2():
    z = dict(np.load("scratchpad/lip_sweep2.npz"))
    Aa, Y = [], []
    for vi in range(5):
        M, rms = mfcc(z[f"a{vi}"].astype("float64"))
        ok = rms >= 0.25 * np.percentile(rms, 75)
        Aa.append(M[ok])
        Y.append(np.full(int(ok.sum()), vi))
    return np.concatenate(Aa), np.concatenate(Y)


def confusion(pred, true):
    C = np.zeros((5, 5), int)
    for t, p in zip(true, pred):
        C[t, p] += 1
    return C


def main():
    Vr, Ar, yr, br = load_ring()
    As, ys = load_sweep2()
    print(f"資料：中午場（整圈+聲學）{len(yr)} 幀、凌晨場（聲學）{len(ys)} 幀\n")

    print("── 跨場次（train 凌晨 → test 中午）：唯一的真泛化證據 ──")
    for kind, nm in (("nn", "最近鄰（現行）"), ("lr", "logistic"),
                     ("mlp", "MLP-32")):
        a, s, pred = fit_eval(As, ys, Ar, yr, kind)
        print(f"  聲學 {nm:<14} acc {a:.2f}  真類機率 {s:.2f}")
        if kind == "mlp":
            print("   混淆（列=真 喔欸咿嗚啊）:\n",
                  confusion(pred, yr))

    print("\n── 場次內時間分塊 CV（單場、掃音低→高分塊＝順帶考跨音高）──")
    for tag, X in (("視覺整圈 80D", Vr), ("聲學 MFCC 13D", Ar),
                   ("融合 93D", np.hstack([Vr, Ar]))):
        row = []
        for kind in ("nn", "lr", "mlp"):
            a, s, pred, true = blocked_cv(X, yr, br, kind)
            row.append(f"{kind} acc {a:.2f}/soft {s:.2f}")
            if kind == "mlp" and tag.startswith("融合"):
                Cm = confusion(pred, true)
        print(f"  {tag:<14} " + "   ".join(row))
    print("  融合 MLP 混淆（列=真 喔欸咿嗚啊）:\n", Cm)


if __name__ == "__main__":
    main()
