"""sing_gate_train2 - add "the share of dark area inside the lips" as a second vote alongside the 27 mouth and jaw blendshapes

Why: the current gate (27 dimensions) reaches 0.934 accuracy across sessions with
no false triggers on a closed or slightly open mouth, but **11% on speech** - an
honest boundary, not a bug. Blendshapes describe the SHAPE of the mouth, and the
shapes of singing and speaking overlap; but **singing holds the mouth open, showing
the dark inside**, while speech opens and closes rapidly, so the dark share has a
different distribution in time. This adds a channel the blendshapes cannot reach.

The dark share is the fraction of pixels inside the lip polygon much darker than the
surrounding skin. It MUST be normalised against the surrounding skin brightness, or
it measures the lighting rather than the mouth and fails entirely when a lamp changes.

Discipline (a lesson learned once, which was enough):
  **Same-session cross-validation is not validation.** The 52-dimension version
  scored 0.915 in same-session CV, which looked excellent, and had 47.5% false
  triggers across sessions, demonstrated on screen with a closed mouth reading 1.00.
  So this records TWO sessions: session A trains, session B is the unseen paper, and
  in between the subject stands up and moves to change their posture and angle.
  Every reported figure is "train on A, test on B"; same-session CV is printed only
  as a control and is never the conclusion.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/sing_gate_train2.py
    to retrain from an existing recording: the same, plus --retrain
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
BS_LO, BS_HI = 23, 50            # the mouth and jaw block of the current model, indices 23 to 49
# The prompt strings stay in Chinese: they are spoken to the singer by macOS `say`.
STATES = [("rest", 0, "閉著嘴，放鬆"), ("ajar", 0, "嘴巴微微張開，不出聲"),
          ("talk", 0, "隨便講幾句話"), ("ah", 1, "唱啊，拉長"),
          ("yi", 1, "唱咿，拉長"), ("wu", 1, "唱嗚，拉長"),
          ("ei", 1, "唱欸，拉長")]
DATA = "scratchpad/sing_gate_data2.npz"
MODEL = "scratchpad/sing_gate_model2.npz"

# The INNER lip ring of the MediaPipe 478-point mesh; while singing, what is inside it is the mouth cavity
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
             308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def dark_feats(frame_bgr, landmarks):
    """Returns (dark share, median brightness inside the lips over median skin brightness). (0, 1), a neutral value, if it cannot be taken.

    The skin reference is the BAND around the dilated lip ring, that is the skin just
    around the mouth, not the whole face: the cheeks and forehead are lit quite
    differently from the mouth, and normalising against the whole face introduces a
    variable that has nothing to do with the mouth.
    """
    h, w = frame_bgr.shape[:2]
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in INNER_LIP], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    inside = int((mask > 0).sum())
    if inside < 30:                       # mouth closed: the polygon degenerates and has no interior
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
    assert best and best[1] > 10, f"no camera with a picture {best}"
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
            subprocess.run(["say", "場次 A，訓練用。七個狀態，各八秒"])   # spoken prompt
        else:
            subprocess.run(["say", "場次 A 錄完。請站起來走一下、換個姿勢和角度，"
                                   "這是驗證用的第二場次，十秒後開始"])   # spoken prompt
            time.sleep(10)
            subprocess.run(["say", "場次 B，驗證用"])   # spoken prompt
        lmk = vision.FaceLandmarker.create_from_options(opts)
        t0 = time.time()
        while time.time() - t0 < WARM:          # each session warms up on its own
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
            print(f"[{sess} {key}] {len(rows)} frames  median dark share "
                  f"{np.median(np.array(rows)[:, -2]):.3f}", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    np.savez(DATA, names=np.array(names), **out)
    print(f"\nraw data -> {DATA}")


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
    VARIANTS = [("27 dimensions (current)", list(range(NB))),
                ("27 + dark share", list(range(NB)) + [NB]),
                ("27 + dark share + relative brightness", list(range(NB)) + [NB, NB + 1])]
    print(f"{'variant':<38}{'A->B acc':>10}{'speech FP':>11}{'closed/ajar FP':>16}"
          f"{'sung missed':>12}")
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
    print(f"\nbest across sessions: {best[1]}")
    return z, names, best


if __name__ == "__main__":
    if "--retrain" not in sys.argv:
        record()
    train()
