"""sing_gate_fit3 — 三場次重訓＋每場基線自校，產 sing_gate_model3.npz

不需要重錄。吃現有的兩個資料檔：
  scratchpad/sing_gate_data.npz   08-13 那場（52 維，取 [23:50]）
  scratchpad/sing_gate_data2.npz  08-15 兩場 A/B（已是 27 維＋2 欄暗區，取前 27）

為什麼要這支（08-15 量出來的三件事）：
 ① 現行模型跨場次會垮：同一個人同一個動作，jawOpen 靜止值 08-13 是 0.031、
    08-15 是 0.017。逐幀 logistic 吃標準化後的絕對值，基線位移足以整批翻盤。
    實測「08-13＋A 訓 → B 測」閉嘴誤觸 44.3%、微張 36.4%。
    **每場自校**（用該場自己的 rest 均值把特徵歸零）→ 10.6% / 0.0%。
 ② 講話擋不掉：任何場次組合、任何門檻，講話誤觸都是 63-97%。嘴型分不開
    唱歌與講話——資訊本身不夠，不是模型不夠好。
 ③ 但**運動量**分得開：0.5s 因果窗的 |ΔjawOpen| 平均，講話在三場是
    0.0173/0.0130/0.0079，唱歌最高的「啊」只有 0.0023。差分量不吃基線，
    所以跨場次站得住。當**否決票**用（模型說像在唱、但嘴在高速開闔＝講話）
    → 講話 97.5% → 0.0%。
 ⚠ 代價：唱歌漏接 13-17%（狀態機有 0.8s 保持開啟，漏幀不等於漏音）。
 ⚠ 門檻是在單一 held-out 場次（08-15 B）上掃出來的＝有調參過擬風險，
    真正驗收在 Harry live。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/sing_gate_fit3.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NB, FPS, JAW, WIN = 27, 29.4, 2, 15      # JAW=names[2]='jawOpen'；WIN≈0.5s
BS_LO = 23                               # 52 維裡嘴/下巴區塊的起點
STATES = [("rest", 0), ("ajar", 0), ("talk", 0),
          ("ah", 1), ("yi", 1), ("wu", 1), ("ei", 1)]
TH_GRID = (0.0040, 0.0030, 0.0025, 0.0020)


def motion(r, w=WIN):
    """0.5s 因果窗的平均絕對幀差（只看過去）＝說話的音節開闔速率。"""
    j = r[:, JAW].astype("float64")
    d = np.abs(np.diff(j, prepend=j[0]))
    return np.array([d[max(0, i - w + 1):i + 1].mean() for i in range(len(r))])


def sim(p, veto, so=0.7, sh=0.3, nfr=2, hold_s=0.8):
    """複製 bank_live 的門控狀態機（遲滯＋連續幀開門＋0.8s 保持）。"""
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
    assert names == [str(x) for x in n["names"]], "兩份資料的欄位順序不一致"
    S = {"0813": {k: o[f"bs_{k}"][:, BS_LO:BS_LO + NB] for k, _ in STATES},
         "0815A": {k: n[f"A_{k}"][:, :NB] for k, _ in STATES},
         "0815B": {k: n[f"B_{k}"][:, :NB] for k, _ in STATES}}
    return names, S


def pack(sess):
    """每場用自己的 rest 均值自校＝訓練與 live 的前處理必須一致。"""
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
    print(f"  {'運動否決':>10}{'閉嘴':>8}{'微張':>8}{'講話':>8}{'唱漏接':>9}")
    rows = []
    for th in (None,) + TH_GRID:
        veto = np.zeros(len(mo), bool) if th is None else (mo > th)
        g = sim(p, veto)
        f = lambda ks: g[np.isin(tag, ks)].mean() * 100
        r = (f(["rest"]), f(["ajar"]), f(["talk"]),
             (1 - g[y == 1].mean()) * 100)
        rows.append((th, r))
        print(f"  {('無' if th is None else f'{th:.4f}'):>10}"
              f"{r[0]:>7.1f}%{r[1]:>7.1f}%{r[2]:>7.1f}%{r[3]:>8.1f}%")
    return rows


def main():
    names, S = load()
    packs = {k: pack(v) for k, v in S.items()}

    # 誠實的成績＝拿一整場當沒看過的考卷（同場次 CV 不算驗證，08-13 血訓）
    print("留一場驗證（每次拿一整場當沒看過的考卷）")
    holdout_rows = []
    for te_k in S:
        tr = [packs[k] for k in S if k != te_k]
        Xt = np.vstack([t[0] for t in tr])
        yt = np.concatenate([t[1] for t in tr])
        sc = StandardScaler().fit(Xt)
        clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xt), yt)
        holdout_rows.append(report(clf, sc, packs[te_k],
                                   f"訓 {'+'.join(k for k in S if k != te_k)}"
                                   f" → 測 {te_k}"))

    # 出貨模型＝三場全訓（留一的數字才是它的期望表現）
    Xa = np.vstack([packs[k][0] for k in S])
    ya = np.concatenate([packs[k][1] for k in S])
    sc = StandardScaler().fit(Xa)
    clf = LogisticRegression(max_iter=5000).fit(sc.transform(Xa), ya)
    W = clf.coef_[0].astype("float64")
    b0 = float(clf.intercept_[0])
    Z = (Xa - sc.mean_) / sc.scale_
    p = 1.0 / (1.0 + np.exp(-(Z @ W + b0)))
    assert abs(p - clf.predict_proba(sc.transform(Xa))[:, 1]).max() < 1e-9, \
        "numpy 推論與 sklearn 不一致"

    # cv 欄位：**留一場**的平均 acc（不是同場次 CV——那正是 08-13 被騙的東西）
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
             # v40 新契約：live 端必須做同樣的前處理，否則模型是廢的
             calib=np.int64(1),           # 1＝特徵要先減開場基線
             jaw_col=np.int64(JAW),       # 運動量算在第幾欄
             motion_win=np.int64(WIN),    # 因果窗幀數（≈0.5s @29fps）
             motion_th=np.float64(0.0025))
    print(f"\n模型 → scratchpad/sing_gate_model3.npz")
    print(f"  留一場平均 acc {cv:.3f}（**跨場次**，不是同場次 CV）")
    print(f"  參數 {W.size + 1}；契約新增 calib/jaw_col/motion_win/motion_th")


if __name__ == "__main__":
    main()
