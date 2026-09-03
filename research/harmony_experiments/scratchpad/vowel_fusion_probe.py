"""vowel_fusion_probe — 視覺＋聲學融合（含天使漏音扣除）可行性（v19 前置）

Harry v18 live 判決：「欸分不出來、喔嗚也沒試出來、咿啊順」。
＝正好是校準幾何最弱的兩對（欸↔咿 0.32、喔↔嗚 0.55；咿↔啊 0.98）。
物理解釋：欸/咿 差在舌位（唇看不見）、喔/嗚 差在嘴唇前突（正面鏡頭
只有 x/y＝深度丟失）。**兩者都聽得見**——聲學跨場次混淆表裡欸 0.94、
咿 0.87，正是視覺的盲點；反過來聲學的弱項是嗚（跨場次 0.32），而視覺
場次內嗚 100%。⇒ 假設：兩通道盲點互補，融合同時解掉他抱怨的兩對。

風險：聲學在 live 會吃天使漏音。vowel_acoustic_probe 已證頻譜扣除可行
（1:1 漏音只吃 4pp、扣除全救回、失準 20ms 仍穩）——本檔把扣除**放進
融合的評估迴圈**，而不是假設它免費。

評估（時間分塊 CV，5 塊；訓練永遠用乾淨、測試才加漏音＝模擬 live）：
  V   視覺 80D（＝v18 現況）
  A   聲學 MFCC 13D
  F   融合 93D
  F+leak      測試音混 1:1 天使、不扣除
  F+leak+sub  測試音混 1:1 天使、頻譜扣除（延遲失準 20ms＝誠實版）
重點指標＝他抱怨的兩對的**兩類分辨率**，不只總 acc。
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
PAIRS = [(1, 2), (0, 3), (2, 4)]          # 欸/咿、喔/嗚、咿/啊（對照組）


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
    """→ dict of 特徵矩陣（乾淨/漏音/扣除）＋ y/blk。訓練恆用乾淨。"""
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
        known = np.roll(seg, int(lag_ms * 1e-3 * SR))   # live 對齊誤差
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
    """訓練用乾淨特徵、測試用（可能被污染的）特徵；回 acc/soft/pred。"""
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
        ("V 視覺（v18 現況）", X["V"], X["V"]),
        ("A 聲學（乾淨）", X["A"], X["A"]),
        ("F 融合（乾淨）", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["A"]])),
        ("F 融合＋漏音不扣", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["Aleak"]])),
        ("F 融合＋扣除(20ms失準)", np.hstack([X["V"], X["A"]]),
         np.hstack([X["V"], X["Asub"]])),
    ]
    print(f"{'條件':<22}{'總acc':>7}{'權重':>7}"
          + "".join(f"{LABELS[i]}/{LABELS[j]:>4}" for i, j in PAIRS))
    for nm, Xtr, Xte in combos:
        acc, soft, pred = cv(Xtr, Xte, y, blk)
        pr = "".join(f"{pair_acc(pred, y, i, j):>8.2f}" for i, j in PAIRS)
        print(f"{nm:<22}{acc:>7.2f}{soft:>7.2f}{pr}")


if __name__ == "__main__":
    main()
