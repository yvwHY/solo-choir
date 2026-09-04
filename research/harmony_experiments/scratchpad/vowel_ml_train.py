"""vowel_ml_train - train the personal vowel weight model (v18)

The experimental basis is in vowel_ml_probe.py: on the same data and the same splits,
logistic regression against nearest neighbour took visual accuracy from 0.81 to 0.89
and **the true-class probability from 0.57 to 0.86**, and the probability is what is
actually wanted.

This file freezes that logistic model into a file the live end can load. **Visual
full-ring only**: the acoustic channel scores higher (0.96) but needs spectral
subtraction first to survive the parts bleeding back in, which is another layer of
engineering and risk, scheduled for v19. The mouth cannot see the speakers, so this
version adds no new risk at all.

Writes scratchpad/vowel_ring_model.npz:
  W (5,80), b (5,), mean (80,), scale (80,) - the live end computes softmax(W.z+b)
  with z = (ring_vec - mean) / scale, five lines of pure numpy and no sklearn
  dependency. cv_acc and cv_soft record the block cross-validation score at the time
  of training, so the numbers are traceable.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_ml_train.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_p = open("scratchpad/vowel_ml_probe.py").read()
_ns = {}
exec(compile(_p.split("\ndef main()")[0], "vmp", "exec"), _ns)   # split at the start of a line only:
#   the probe file also contains the literal "def main()" in its text, so a bare string split would cut in the wrong place
load_ring, blocked_cv, LABELS = _ns["load_ring"], _ns["blocked_cv"], _ns["LABELS"]


def main():
    V, _A, y, b = load_ring()
    acc, soft, _p1, _t1 = blocked_cv(V, y, b, "lr")
    print(f"block cross-validation (visual full ring): acc {acc:.2f}  true-class probability {soft:.2f}")

    sc = StandardScaler().fit(V)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(V), y)
    W = clf.coef_.astype("float64")
    bb = clf.intercept_.astype("float64")
    assert W.shape == (5, V.shape[1]) and bb.shape == (5,), (W.shape, bb.shape)

    # read-back self-check: the pure numpy inference must match sklearn to the bit, since the live end uses the former
    Z = (V - sc.mean_) / sc.scale_
    L = Z @ W.T + bb
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    err = float(np.abs(P - clf.predict_proba(sc.transform(V))).max())
    print(f"numpy inference against sklearn, max error {err:.2e} (must be under 1e-9)")
    assert err < 1e-9

    np.savez("scratchpad/vowel_ring_model.npz", W=W, b=bb,
             mean=sc.mean_, scale=sc.scale_, labels=np.array(LABELS),
             cv_acc=np.float64(acc), cv_soft=np.float64(soft))
    print(f"{len(y)} training frames, model -> scratchpad/vowel_ring_model.npz "
          f"({W.size + bb.size} parameters)")
    print("true-class probability on the training set itself", float(P[np.arange(len(y)), y].mean()).__round__(3))


if __name__ == "__main__":
    main()
