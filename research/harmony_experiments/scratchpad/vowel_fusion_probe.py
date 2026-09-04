"""vowel_fusion_probe - feasibility of fusing the visual and acoustic channels, including subtracting the parts bleeding back in

The live verdict on v18: two vowels could not be told apart, another pair never
came through, and one pair was fine. Those are exactly the two weakest pairs in the
calibration geometry (0.32 and 0.55 apart, against 0.98 for the pair that worked).
The physical explanation: one pair differs in tongue position, which the lips do not
show, and the other in lip protrusion, which a front-on camera cannot see because it
only has x and y. **Both are audible** - in the across-session acoustic confusion
matrix those two score 0.94 and 0.87, precisely the visual blind spot; and the
acoustic channel's own weak vowel scores 0.32 across sessions where the visual
channel gets it right 100% of the time within a session. Hypothesis: the two
channels' blind spots are complementary, and fusing them solves both complaints.

The risk: live, the acoustic channel picks up the parts bleeding back in.
vowel_acoustic_probe already showed spectral subtraction works (1:1 bleed costs only
4 percentage points, subtraction recovers all of it, and it stays stable with 20 ms
of misalignment). This file puts that subtraction INSIDE the evaluation loop rather
than assuming it is free.

Evaluation (cross-validation over 5 time blocks; training always on clean audio and
only the test set given bleed, which simulates live):
  V   visual 80D (that is, v18 as it stands)
  A   acoustic MFCC 13D
  F   fused 93D
  F+leak      test audio mixed 1:1 with the parts, no subtraction
  F+leak+sub  test audio mixed 1:1, with spectral subtraction at 20 ms of
              misalignment, which is the honest version
The headline metric is the TWO-CLASS discrimination on the two complained-about
pairs, not overall accuracy alone.
"""
import numpy as np
import soundfile as sf

_p = open("scratchpad/vowel_ml_probe.py").read()
_ns = {}
exec(compile(_p.split("\ndef main()")[0], "vmp", "exec"), _ns)
(ring_vec, mfcc, blocks, fit_eval, LABELS, SR, HOPF, NB,
 FB, WIN, NFFT) = (_ns["ring_vec"], _ns["mfcc"], _ns["blocks"],
                   _ns["fit_eval"], _ns["LABELS"], _ns["SR"], _ns["HOPF"],
                   _ns["NB"], _ns["FB"], _ns["WIN"], _ns["NFFT"])
PAIRS = [(1, 2), (0, 3), (2, 4)]          # the two problem pairs, plus one control


def spec(x):
    n = (len(x) - NFFT) // HOPF
    fr = np.stack([x[i * HOPF:i * HOPF + NFFT] * WIN for i in range(n)])
    return np.abs(np.fft.rfft(fr, axis=1)) ** 2


def mfcc_from_power(P):
    L = np.log(P @ FB.T + 1e-10)
    K = L.shape[1]
    D = np.cos(np.pi / K * (np.arange(K) + 0.5)[None, :]
               * np.arange(1, 14)[:, None])
    return L @ D.T


def build(angel, lag_ms):
    """To a dict of feature matrices (clean, bleed, subtracted) plus y and blk. Training always uses the clean ones."""
    z = dict(np.load("scratchpad/lip_ring_sweep.npz"))
    asp = float(z["aspect"])
    out = {k: [] for k in ("V", "A", "Aleak", "Asub")}
    Y, B = [], []
    ar = np.sqrt(np.convolve(angel ** 2, np.ones(4410) / 4410, "same"))
    ai = int(np.argmax(ar > np.percentile(ar, 90)))
    for vi in range(5):
        LA, ts, au = z[f"l{vi}"], z[f"t{vi}"], z[f"a{vi}"].astype("float64")
        M, rms = mfcc(au)
        fi = np.clip((ts * SR / HOPF).astype(int), 0, len(M) - 1)
        ok = rms[fi] >= 0.25 * np.percentile(rms, 75)
        seg = angel[ai + vi * len(au): ai + (vi + 1) * len(au)]
        if len(seg) < len(au):
            seg = np.tile(angel[ai:ai + len(au)], 2)[:len(au)]
        g = np.sqrt((au ** 2).mean() / max((seg ** 2).mean(), 1e-12))
        seg = seg * g
        mix = au + seg
        known = np.roll(seg, int(lag_ms * 1e-3 * SR))   # live alignment error
        Pm, Pk = spec(mix), spec(known)
        Ml = mfcc_from_power(Pm)
        Ms = mfcc_from_power(np.maximum(Pm - Pk, 0.0))
        out["V"].append(np.array([ring_vec(x, asp) for x in LA[ok]]))
        out["A"].append(M[fi[ok]])
        out["Aleak"].append(Ml[np.clip(fi[ok], 0, len(Ml) - 1)])
        out["Asub"].append(Ms[np.clip(fi[ok], 0, len(Ms) - 1)])
        Y.append(np.full(int(ok.sum()), vi))
        B.append(blocks(int(ok.sum())))
    return ({k: np.concatenate(v) for k, v in out.items()},
            np.concatenate(Y), np.concatenate(B))


def cv(Xtr_all, Xte_all, y, blk):
    """Train on clean features, test on the (possibly contaminated) ones; returns acc, soft and pred."""
    P = np.zeros((len(y), 5))
    for b in range(NB):
        te = blk == b
        if te.sum() == 0:
            continue
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr_all[~te])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr_all[~te]),
                                                    y[~te])
        P[te] = clf.predict_proba(sc.transform(Xte_all[te]))
    pred = P.argmax(1)
    return ((pred == y).mean(), P[np.arange(len(y)), y].mean(), pred)


def pair_acc(pred, y, i, j):
    m = (y == i) | (y == j)
    return float((pred[m] == y[m]).mean())


def main():
    ang, _sr = sf.read("scratchpad/v18_r1_out.wav", always_2d=True)
    ang = ang.mean(1).astype("float64")
    X, y, blk = build(ang, lag_ms=20.0)
    combos = [
        ("V visual (v18 as it stands)", X["V"], X["V"]),
        ("A acoustic (clean)", X["A"], X["A"]),
        ("F fused (clean)", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["A"]])),
        ("F fused + bleed, no subtraction", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["Aleak"]])),
        ("F fused + subtraction (20ms off)", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["Asub"]])),
    ]
    print(f"{'condition':<34}{'acc':>7}{'weight':>8}"
          + "".join(f"{LABELS[i]}/{LABELS[j]:>4}" for i, j in PAIRS))
    for nm, Xtr, Xte in combos:
        acc, soft, pred = cv(Xtr, Xte, y, blk)
        pr = "".join(f"{pair_acc(pred, y, i, j):>8.2f}" for i, j in PAIRS)
        print(f"{nm:<22}{acc:>7.2f}{soft:>7.2f}{pr}")


if __name__ == "__main__":
    main()
