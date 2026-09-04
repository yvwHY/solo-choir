"""vowel_fusion_train - train the fused vowel model (v19)

The basis is in vowel_fusion_probe: fusion against visual alone takes the first pair
the live v18 verdict complained about **from 0.79 to 0.94**, overall accuracy from
0.89 to 0.95 and the weight from 0.86 to 0.94; and a 1:1 bleed from the parts costs
only 0 to 1 percentage point, so spectral subtraction is dropped, saving an entire
alignment pipeline.

The 93 features are the 80-dimensional full lip ring (visual, EMA on the camera
thread) plus 13 MFCCs (acoustic, per hop on the audio thread). It writes
scratchpad/vowel_fusion_model.npz:
  W (5,93), b (5,), mean, scale, feat='ring+mfcc' and the MFCC parameters, which the
  live end checks: a mismatch refuses to load, so the two MFCC definitions cannot
  quietly diverge.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_p = open("scratchpad/vowel_ml_probe.py").read()
_ns = {}
exec(compile(_p.split("\ndef main()")[0], "vmp", "exec"), _ns)
load_ring, blocked_cv, LABELS = (_ns["load_ring"], _ns["blocked_cv"],
                                 _ns["LABELS"])


def main():
    V, A, y, b = load_ring()
    X = np.hstack([V, A])
    acc, soft, _p1, _t1 = blocked_cv(X, y, b, "lr")
    print(f"block cross-validation (fused 93D): acc {acc:.2f}  true-class probability {soft:.2f}")

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    W, bb = clf.coef_.astype("float64"), clf.intercept_.astype("float64")
    assert W.shape == (5, 93), W.shape

    Z = (X - sc.mean_) / sc.scale_
    L = Z @ W.T + bb
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    err = float(np.abs(P - clf.predict_proba(sc.transform(X))).max())
    print(f"numpy inference against sklearn, max error {err:.2e}")
    assert err < 1e-9

    np.savez("scratchpad/vowel_fusion_model.npz", W=W, b=bb,
             mean=sc.mean_, scale=sc.scale_, labels=np.array(LABELS),
             feat=np.array("ring+mfcc"), nfft=np.int64(_ns["NFFT"]),
             hopf=np.int64(_ns["HOPF"]), nmel=np.int64(26),
             fmin=np.float64(200.0), fmax=np.float64(6000.0),
             ncep=np.int64(13), cv_acc=np.float64(acc),
             cv_soft=np.float64(soft))
    print(f"{len(y)} training frames, model -> scratchpad/vowel_fusion_model.npz "
          f"({W.size + bb.size} parameters)")


if __name__ == "__main__":
    main()
