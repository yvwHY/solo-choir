"""vowel_ml_train — 訓練個人母音權重模型（v18；08-13）

實驗判準見 vowel_ml_probe.py：同資料同切分，logistic 對最近鄰＝
視覺 acc 0.81→0.89、**真類機率 0.57→0.86**（權重才是要的東西）。

本檔＝把那個 logistic 固化成 live 可載的檔案。**只用視覺整圈嘴**：
聲學分數更高（0.96）但天使漏音要先做頻譜扣除＝另一層工程與風險，
排在 v19；嘴看不到喇叭＝這版零新風險。

輸出 scratchpad/vowel_ring_model.npz：
  W (5,80) b (5,) mean (80,) scale (80,)  ——  live 端 softmax(W·z+b)、
  z=(ring_vec−mean)/scale，純 numpy 五行，不進 sklearn 依賴。
  cv_acc/cv_soft ＝訓練當下的分塊 CV 成績（來源可追）。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_ml_train.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_p = open("scratchpad/vowel_ml_probe.py").read()
_ns = {}
exec(compile(_p.split("\ndef main()")[0], "vmp", "exec"), _ns)   # 行首才切
#   （probe 檔內文也有 "def main()" 字面值＝不能用裸字串切）
load_ring, blocked_cv, LABELS = _ns["load_ring"], _ns["blocked_cv"], _ns["LABELS"]


def main():
    V, _A, y, b = load_ring()
    acc, soft, _p1, _t1 = blocked_cv(V, y, b, "lr")
    print(f"分塊 CV（視覺整圈）：acc {acc:.2f}  真類機率 {soft:.2f}")

    sc = StandardScaler().fit(V)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(V), y)
    W = clf.coef_.astype("float64")
    bb = clf.intercept_.astype("float64")
    assert W.shape == (5, V.shape[1]) and bb.shape == (5,), (W.shape, bb.shape)

    # read-back 自驗：純 numpy 推論要與 sklearn 逐位元近似（live 端用前者）
    Z = (V - sc.mean_) / sc.scale_
    L = Z @ W.T + bb
    P = np.exp(L - L.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    err = float(np.abs(P - clf.predict_proba(sc.transform(V))).max())
    print(f"numpy 推論 vs sklearn max err {err:.2e}（要 <1e-9）")
    assert err < 1e-9

    np.savez("scratchpad/vowel_ring_model.npz", W=W, b=bb,
             mean=sc.mean_, scale=sc.scale_, labels=np.array(LABELS),
             cv_acc=np.float64(acc), cv_soft=np.float64(soft))
    print(f"訓練幀 {len(y)}、模型 → scratchpad/vowel_ring_model.npz "
          f"（{W.size + bb.size} 參數）")
    print("訓練集自身真類機率", float(P[np.arange(len(y)), y].mean()).__round__(3))


if __name__ == "__main__":
    main()
