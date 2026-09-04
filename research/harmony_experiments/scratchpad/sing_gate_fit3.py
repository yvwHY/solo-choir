"""sing_gate_fit3 - retrain over three sessions with per-session baseline self-calibration, producing sing_gate_model3.npz

No new recording needed. It consumes the two existing data files:
  scratchpad/sing_gate_data.npz   the first session (52 dimensions, taking [23:50])
  scratchpad/sing_gate_data2.npz  two A/B sessions (already 27 dimensions plus 2
                                  dark columns; the first 27 are taken)

Why this exists - three things that were measured:
 1. The current model collapses across sessions. The same person in the same posture
    had a resting jawOpen of 0.031 on one day and 0.017 two days later. A per-frame
    logistic model on standardised absolute values can be flipped wholesale by that
    much baseline shift. Measured: training on the first session plus A and testing
    on B gave 44.3% false triggers on a closed mouth and 36.4% on a slightly open
    one. **Self-calibrating per session**, zeroing the features against that
    session's own resting mean, brings those to 10.6% and 0.0%.
 2. Speech cannot be excluded. Across every combination of sessions and every
    threshold, speech triggers 63 to 97% of the time. Mouth shape does not separate
    singing from speech: the information is not there, and no model can fix that.
 3. But MOVEMENT does separate them. The mean |delta jawOpen| over a 0.5 s causal
    window is 0.0173, 0.0130 and 0.0079 for speech across the three sessions, while
    the highest vowel in singing reaches only 0.0023. A difference does not depend on
    the baseline, so it holds across sessions. Used as a VETO - the model says this
    looks like singing, but the jaw is opening and closing rapidly, so it is speech -
    it takes speech from 97.5% to 0.0%.
 Cost: 13 to 17% of singing frames are missed (the state machine holds the gate open
 for 0.8 s, so a missed frame is not a missed note).
 The threshold was swept on a single held-out session, so it carries a risk of
 overfitting; the real acceptance test is live.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/sing_gate_fit3.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NB, FPS, JAW, WIN = 27, 29.4, 2, 15      # JAW=names[2]='jawOpen'；WIN≈0.5s
BS_LO = 23                               # where the mouth and jaw block starts within the 52 dimensions
STATES = [("rest", 0), ("ajar", 0), ("talk", 0),
          ("ah", 1), ("yi", 1), ("wu", 1), ("ei", 1)]
TH_GRID = (0.0040, 0.0030, 0.0025, 0.0020)


def motion(r, w=WIN):
    """Mean absolute frame difference over a 0.5 s causal window (past only), that is the syllable rate of speech."""
    j = r[:, JAW].astype("float64")
    d = np.abs(np.diff(j, prepend=j[0]))
    return np.array([d[max(0, i - w + 1):i + 1].mean() for i in range(len(r))])


def sim(p, veto, so=0.7, sh=0.3, nfr=2, hold_s=0.8):
    """A copy of bank_live's gate state machine: hysteresis, consecutive frames to open, and a 0.8 s hold."""
    on, arm, last, out = False, 0, -1e9, []
    for i, pi in enumerate(p):
        ok = pi > (sh if on else so) and not veto[i]
        if on:
            nv = ok
            if not nv:
                arm = 0
        else:
            arm = arm + 1 if ok else 0
            nv = arm >= nfr
        if nv:
            last = i / FPS
        on = (i / FPS - last) < hold_s
        out.append(on)
    return np.array(out)


def load():
    o = np.load("scratchpad/sing_gate_data.npz")
    n = np.load("scratchpad/sing_gate_data2.npz")
    names = [str(x) for x in o["names"]][BS_LO:BS_LO + NB]
    assert names == [str(x) for x in n["names"]], "the two data files order their columns differently"
    S = {"0813": {k: o[f"bs_{k}"][:, BS_LO:BS_LO + NB] for k, _ in STATES},
         "0815A": {k: n[f"A_{k}"][:, :NB] for k, _ in STATES},
         "0815B": {k: n[f"B_{k}"][:, :NB] for k, _ in STATES}}
    return names, S


def pack(sess):
    """Each session self-calibrates against its own resting mean; training and live preprocessing have to match."""
    base = sess["rest"].mean(0)
    X, y, tag, mo = [], [], [], []
    for k, lab in STATES:
        r = sess[k]
        X.append(r - base)
        y.append(np.full(len(r), lab))
        tag += [k] * len(r)
        mo.append(motion(r))
    return np.vstack(X), np.concatenate(y), np.array(tag), np.concatenate(mo)


def report(clf, sc, te, label):
    X, y, tag, mo = te
    p = 1.0 / (1.0 + np.exp(-(((X - sc.mean_) / sc.scale_) @ clf.coef_[0]
                              + float(clf.intercept_[0]))))
    print(f"  {label}")
    print(f"  {'motion veto':>12}{'closed':>8}{'ajar':>8}{'speech':>8}{'sung missed':>12}")
    rows = []
    for th in (None,) + TH_GRID:
        veto = np.zeros(len(mo), bool) if th is None else (mo > th)
        g = sim(p, veto)
        f = lambda ks: g[np.isin(tag, ks)].mean() * 100
        r = (f(["rest"]), f(["ajar"]), f(["talk"]),
             (1 - g[y == 1].mean()) * 100)
        rows.append((th, r))
        print(f"  {('none' if th is None else f'{th:.4f}'):>12}"
              f"{r[0]:>7.1f}%{r[1]:>7.1f}%{r[2]:>7.1f}%{r[3]:>8.1f}%")
    return rows


def main():
    names, S = load()
    packs = {k: pack(v) for k, v in S.items()}

    # the honest score is holding out a whole session; same-session cross-validation
    # is not validation, which is the lesson from the first session
    print("leave-one-session-out validation")
    holdout_rows = []
    for te_k in S:
        tr = [packs[k] for k in S if k != te_k]
        Xt = np.vstack([t[0] for t in tr])
        yt = np.concatenate([t[1] for t in tr])
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xt), yt)
        holdout_rows.append(report(clf, sc, packs[te_k],
                                   f"train {'+'.join(k for k in S if k != te_k)}"
                                   f" -> test {te_k}"))

    # the shipping model trains on all three; the leave-one-out figure is what to expect of it
    Xa = np.vstack([packs[k][0] for k in S])
    ya = np.concatenate([packs[k][1] for k in S])
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xa), ya)
    W = clf.coef_[0].astype("float64")
    b0 = float(clf.intercept_[0])
    Z = (Xa - sc.mean_) / sc.scale_
    p = 1.0 / (1.0 + np.exp(-(Z @ W + b0)))
    assert abs(p - clf.predict_proba(sc.transform(Xa))[:, 1]).max() < 1e-9, \
        "the numpy inference disagrees with sklearn"

    # the cv field holds the LEAVE-ONE-SESSION-OUT mean accuracy, not same-session CV, which is exactly what misled the first round
    accs = []
    for te_k in S:
        tr = [packs[k] for k in S if k != te_k]
        Xt = np.vstack([t[0] for t in tr])
        yt = np.concatenate([t[1] for t in tr])
        s2 = StandardScaler().fit(Xt)
        c2 = LogisticRegression(max_iter=5000).fit(s2.transform(Xt), yt)
        X, y, _t, _m = packs[te_k]
        accs.append((c2.predict(s2.transform(X)) == y).mean())
    cv = float(np.mean(accs))

    np.savez("scratchpad/sing_gate_model3.npz",
             W=W, b=np.float64(b0), mean=sc.mean_, scale=sc.scale_,
             names=np.array(names),
             idx=np.arange(BS_LO, BS_LO + NB),
             cv=np.float64(cv),
             # the v40 contract: the live end must do the same preprocessing or the model is useless
             calib=np.int64(1),           # 1 means subtract the opening baseline from the features first
             jaw_col=np.int64(JAW),       # which column the movement is computed on
             motion_win=np.int64(WIN),    # causal window in frames (about 0.5 s at 29 fps)
             motion_th=np.float64(0.0025))
    print(f"\nmodel -> scratchpad/sing_gate_model3.npz")
    print(f"  leave-one-session-out mean acc {cv:.3f} (ACROSS sessions, not same-session CV)")
    print(f"  {W.size + 1} parameters; the contract adds calib, jaw_col, motion_win and motion_th")


if __name__ == "__main__":
    main()
