"""vowel_fusion_train — 訓練融合母音模型（v19；08-13）

判準見 vowel_fusion_probe：融合對視覺，**欸/咿 0.79→0.94**（Harry v18
live 抱怨的第一對）、總 acc 0.89→0.95、權重 0.86→0.94；天使 1:1 漏音
只吃 0-1pp ⇒ 頻譜扣除不做（省一整層對齊管線）。

特徵 93D＝整圈嘴 80D（視覺、鏡頭執行緒 EMA）＋MFCC 13D（聲學、音訊
執行緒逐 hop）。輸出 scratchpad/vowel_fusion_model.npz：
  W (5,93) b (5,) mean scale + feat='ring+mfcc' + MFCC 參數（live 端
  驗證用：參數不符＝拒載，免得兩份 MFCC 定義悄悄分家）。
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
    print(f"分塊 CV（融合 93D）：acc {acc:.2f}  真類機率 {soft:.2f}")

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X), y)
    W, bb = clf.coef_.astype("float64"), clf.intercept_.astype("float64")
    assert W.shape == (5, 93), W.shape

    Z = (X - sc.mean_) / sc.scale_
    L = Z @ W.T + bb
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    err = float(np.abs(P - clf.predict_proba(sc.transform(X))).max())
    print(f"numpy 推論 vs sklearn max err {err:.2e}")
    assert err < 1e-9

    np.savez("scratchpad/vowel_fusion_model.npz", W=W, b=bb,
             mean=sc.mean_, scale=sc.scale_, labels=np.array(LABELS),
             feat=np.array("ring+mfcc"), nfft=np.int64(_ns["NFFT"]),
             hopf=np.int64(_ns["HOPF"]), nmel=np.int64(26),
             fmin=np.float64(200.0), fmax=np.float64(6000.0),
             ncep=np.int64(13), cv_acc=np.float64(acc),
             cv_soft=np.float64(soft))
    print(f"訓練幀 {len(y)}、模型 → scratchpad/vowel_fusion_model.npz "
          f"（{W.size + bb.size} 參數）")


if __name__ == "__main__":
    main()
