"""vowel_ml_probe - an offline test of whether finer vowel control needs machine learning

Background: two channels and two kinds of feature, and inverse-distance nearest
neighbour hits the same ceiling in both - the visual channel at a live entropy of
0.83 (blurred) and the acoustic mel nearest neighbour at 0.60 across sessions.
Hypothesis: the bottleneck is not a lack of sensor information but that "five
prototype points and a distance" is too weak a model.
This experiment: the same data and the same splits, with the nearest neighbour
replaced by a small classifier, to see how much difference it makes.

Methodology, written down in advance so as not to fool myself:
  1. **Cross-validate over blocks of time**, never a random split: adjacent frames
     are highly correlated, and a random split puts the same mouth shape in both
     train and test, inflating accuracy. Each take sweeps from low to high, so
     blocking also tests whether it still recognises the vowel at a new pitch.
  2. Only **across sessions** counts as real generalisation: the acoustic branch
     trains on one session and tests on another. The visual branch has full lip-ring
     landmarks from one session only, so it can only be block cross-validated, and
     the report says plainly that it has no across-session evidence.
  3. **The baseline runs on the same splits** as the current nearest neighbour, so
     the difference can be attributed.
  4. What is actually wanted is WEIGHTS, not hard labels, so alongside accuracy it
     reports the mean probability of the true class, which is what feeds the mixing
     layer.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_ml_probe.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SR = 44100
NFFT, HOPF = 2048, 1470                  # a 30 Hz frame rate, matching the control channel
LABELS = ["喔", "欸", "咿", "嗚", "啊"]   # vowel labels; they key bank_live.VOWEL_LAYERS
NB = 5                                   # number of time blocks

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
    """To (T,13) MFCCs c1 to c13, dropping c0 which is level; the low DCT quefrencies are the envelope and suppress the pitch comb."""
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
    """Contiguous time-block indices, 0 to nb-1."""
    return np.minimum((np.arange(n) * nb) // max(n, 1), nb - 1)


def nn_baseline(Xtr, ytr, Xte):
    """The current v17.1 geometry: class means as anchors, nearest neighbour by normalised L1 distance."""
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
    print(f"data: session with the full ring and audio {len(yr)} frames, audio-only session {len(ys)} frames\n")

    print("-- across sessions (train on the earlier, test on the later): the only real generalisation evidence --")
    for kind, nm in (("nn", "nearest neighbour (current)"), ("lr", "logistic"),
                     ("mlp", "MLP-32")):
        a, s, pred = fit_eval(As, ys, Ar, yr, kind)
        print(f"  acoustic {nm:<26} acc {a:.2f}  true-class probability {s:.2f}")
        if kind == "mlp":
            print("   confusion (rows = true, in LABELS order):\n",
                  confusion(pred, yr))

    print("\n-- within-session cross-validation over time blocks (one session, swept low to high, so it also tests across pitch) --")
    for tag, X in (("visual full ring 80D", Vr), ("acoustic MFCC 13D", Ar),
                   ("fused 93D", np.hstack([Vr, Ar]))):
        row = []
        for kind in ("nn", "lr", "mlp"):
            a, s, pred, true = blocked_cv(X, yr, br, kind)
            row.append(f"{kind} acc {a:.2f}/soft {s:.2f}")
            if kind == "mlp" and tag.startswith("fused"):
                Cm = confusion(pred, true)
        print(f"  {tag:<14} " + "   ".join(row))
    print("  fused MLP confusion (rows = true, in LABELS order):\n", Cm)


if __name__ == "__main__":
    main()
