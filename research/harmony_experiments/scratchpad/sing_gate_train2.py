"""sing_gate_train2 — 在 27 維嘴/下巴 blendshape 之外加「唇內暗區佔比」當第二票

為什麼：現行門控（08-13，27 維）跨場次 acc 0.934、閉嘴/微張誤觸 0%，但
**講話 11% 誤觸**——那是誠實邊界，不是 bug。blendshape 描述的是「嘴的形狀」，
而唱與講話的形狀重疊；**唱歌時嘴會持續張開露出口腔內部（暗）**，講話則是
快速開闔，暗區佔比的時間分布不同。所以加的是一個 blendshape 拿不到的通道。

暗區佔比＝唇內多邊形裡「比周圍皮膚暗很多」的像素比例。**一定要用周圍皮膚
亮度正規化**，否則它量到的是打光不是嘴（換一盞燈就全盤失效）。

紀律（08-13 血訓，違反過一次就夠了）：
  **同場次 CV 不算驗證。** 52 維那版同場次 CV 0.915 看起來很漂亮，跨場次
  誤觸 47.5%，Harry 螢幕實證「閉嘴 sing 1.00」。所以這支**錄兩個場次**：
  場次 A 訓練、場次 B 當沒看過的考卷，中間要求受試者起身走動改變姿勢/角度。
  報出來的數字一律是「A 訓 → B 測」，同場次 CV 只印出來當對照，不當結論。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/sing_gate_train2.py
    已經錄過要重訓（不用再錄）: 同上 + --retrain
"""
import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEC, WARM = 8.0, 3.0
BS_LO, BS_HI = 23, 50            # 現行模型的嘴/下巴區塊（idx 23..49）
STATES = [("rest", 0, "閉著嘴，放鬆"), ("ajar", 0, "嘴巴微微張開，不出聲"),
          ("talk", 0, "隨便講幾句話"), ("ah", 1, "唱啊，拉長"),
          ("yi", 1, "唱咿，拉長"), ("wu", 1, "唱嗚，拉長"),
          ("ei", 1, "唱欸，拉長")]
DATA = "scratchpad/sing_gate_data2.npz"
MODEL = "scratchpad/sing_gate_model2.npz"

# MediaPipe 478 點網格的**內唇**環（唱歌時這圈裡面就是口腔）
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
             308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def dark_feats(frame_bgr, landmarks):
    """→ (暗區佔比, 唇內中位亮度/皮膚中位亮度)。取不到就 (0, 1)＝中性值。

    皮膚參考取唇環膨脹後的「環帶」（嘴周圍那圈皮膚），不是整張臉——臉頰
    與額頭的打光跟嘴周圍差很多，用整張臉正規化等於引進一個跟嘴無關的變數。
    """
    h, w = frame_bgr.shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in INNER_LIP], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    inside = int((mask > 0).sum())
    if inside < 30:                       # 嘴閉著＝多邊形退化，沒有內部
        return 0.0, 1.0
    k = max(5, int(np.sqrt(inside) * 0.8)) | 1
    ring = cv2.dilate(mask, np.ones((k, k), np.uint8)) - mask
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    skin = float(np.median(g[ring > 0])) if (ring > 0).sum() > 30 else 0.0
    if skin < 5:
        return 0.0, 1.0
    vin = g[mask > 0].astype("float64")
    return float((vin < 0.6 * skin).mean()), float(np.median(vin) / skin)


def open_cam():
    best = None
    for ci in range(3):
        c = cv2.VideoCapture(ci)
        if not c.isOpened():
            continue
        time.sleep(0.4)
        okf, fr = c.read()
        b = float(fr.mean()) if okf else -1
        c.release()
        if best is None or b > best[1]:
            best = (ci, b)
    assert best and best[1] > 10, f"找不到有畫面的鏡頭 {best}"
    return cv2.VideoCapture(best[0])


def record():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="scratchpad/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True)
    cap = open_cam()
    out = {}
    names = None
    for sess in ("A", "B"):
        if sess == "A":
            subprocess.run(["say", "場次 A，訓練用。七個狀態，各八秒"])
        else:
            subprocess.run(["say", "場次 A 錄完。請站起來走一下、換個姿勢和角度，"
                                   "這是驗證用的第二場次，十秒後開始"])
            time.sleep(10)
            subprocess.run(["say", "場次 B，驗證用"])
        lmk = vision.FaceLandmarker.create_from_options(opts)
        t0 = time.time()
        while time.time() - t0 < WARM:          # 每場次各自暖機
            okf, frame = cap.read()
            if okf:
                lmk.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    int((time.time() - t0) * 1000) + 1)
        lmk.close()
        for key, lab, prompt in STATES:
            lmk = vision.FaceLandmarker.create_from_options(opts)
            subprocess.run(["say", prompt])
            rows = []
            t0 = time.time()
            while time.time() - t0 < SEC:
                okf, frame = cap.read()
                if not okf:
                    continue
                now = time.time()
                res = lmk.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    int((now - t0) * 1000) + 1)
                if not res.face_blendshapes or not res.face_landmarks:
                    continue
                bs = res.face_blendshapes[0]
                if names is None:
                    names = [c.category_name for c in bs][BS_LO:BS_HI]
                dr, rb = dark_feats(frame, res.face_landmarks[0])
                rows.append([c.score for c in bs][BS_LO:BS_HI] + [dr, rb])
                h2 = int(frame.shape[0] * 480 / frame.shape[1])
                fr2 = cv2.resize(frame, (480, h2))
                cv2.putText(fr2, f"{sess} {key} ({'sing' if lab else 'not'})"
                            f"  {SEC - (now - t0):.0f}s  dark {dr:.2f}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if lab else (0, 165, 255), 2)
                cv2.imshow("sing gate 2", fr2)
                cv2.waitKey(1)
            lmk.close()
            out[f"{sess}_{key}"] = np.array(rows)
            print(f"[{sess} {key}] 幀 {len(rows)}  暗區中位 "
                  f"{np.median(np.array(rows)[:, -2]):.3f}", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    np.savez(DATA, names=np.array(names), **out)
    print(f"\n原始資料 → {DATA}")


def fit(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=4000).fit(sc.transform(Xtr), ytr)
    W, b0 = clf.coef_[0].astype("float64"), float(clf.intercept_[0])
    pr = lambda X: 1.0 / (1.0 + np.exp(-(((X - sc.mean_) / sc.scale_) @ W + b0)))
    return sc, W, b0, pr, float((pr(Xte) > 0.5).astype(int).__eq__(yte).mean())


def train():
    z = np.load(DATA)
    names = [str(x) for x in z["names"]]
    keys = [k for k in z.files if k != "names"]

    def pack(sess, cols):
        X, y, tag = [], [], []
        for key, lab, _ in STATES:
            r = z[f"{sess}_{key}"]
            X.append(r[:, cols])
            y.append(np.full(len(r), lab))
            tag += [key] * len(r)
        return np.vstack(X), np.concatenate(y), np.array(tag)

    NB = len(names)
    VARIANTS = [("27 維（現行做法）", list(range(NB))),
                ("27+暗區", list(range(NB)) + [NB]),
                ("27+暗區+相對亮度", list(range(NB)) + [NB, NB + 1])]
    print(f"{'方案':<22}{'A訓→B測 acc':>13}{'講話誤觸':>10}{'閉嘴/微張誤觸':>15}"
          f"{'唱漏接':>9}")
    best = None
    for label, cols in VARIANTS:
        Xa, ya, _ = pack("A", cols)
        Xb, yb, tb = pack("B", cols)
        sc, W, b0, pr, acc = fit(Xa, ya, Xb, yb)
        p = pr(Xb)
        talk = float((p[tb == "talk"] > 0.5).mean())
        quiet = float((p[np.isin(tb, ["rest", "ajar"])] > 0.5).mean())
        miss = float((p[yb == 1] <= 0.5).mean())
        print(f"{label:<22}{acc:>13.3f}{talk*100:>9.1f}%{quiet*100:>14.1f}%"
              f"{miss*100:>8.1f}%")
        if best is None or (talk, -acc) < best[0]:
            best = ((talk, -acc), label, cols)
    print(f"\n跨場次最好的是：{best[1]}")
    return z, names, best


if __name__ == "__main__":
    if "--retrain" not in sys.argv:
        record()
    train()
