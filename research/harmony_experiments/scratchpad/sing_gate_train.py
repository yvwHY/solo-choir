"""sing_gate_train - train a gate for "is he singing" from blendshapes

The complaint that started it: singing one particular vowel turned the indicator
red, that is, the gate decided he was not singing. It has the same shape as another
vowel: the jaw barely opens (jawOpen low) and the lips are not pursed (pucker low),
so both hand-picked criteria fail. Hand-picking a third coefficient is whack-a-mole,
one vowel at a time, so **let the data decide**: record a few seconds each of
singing (four vowels) and not singing (closed, slightly open, speaking), fit one
logistic model on the 52 blendshapes, and use its probability as the gate.

The same method and discipline as the v18 vowel model: cross-validation over blocks
of time (adjacent frames are correlated, so no random split), raw data saved to an
npz, the numpy inference checked bit for bit against sklearn, and five lines of pure
numpy at the live end.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/sing_gate_train.py
"""
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEC, WARM, NB = 5.0, 3.0, 5
# The prompt strings stay in Chinese: they are spoken to the singer by macOS `say`.
STATES = [("rest", 0, "閉著嘴，放鬆"), ("ajar", 0, "嘴巴微微張開，不出聲"),
          ("talk", 0, "隨便講幾句話"), ("ah", 1, "唱啊，拉長"),
          ("yi", 1, "唱咿，拉長"), ("wu", 1, "唱嗚，拉長"),
          ("ei", 1, "唱欸，拉長")]


def main():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="scratchpad/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True)
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
    cap = cv2.VideoCapture(best[0])
    lmk = vision.FaceLandmarker.create_from_options(opts)
    subprocess.run(["say", "暖機三秒"])       # spoken: warming up, three seconds
    t0 = time.time()
    while time.time() - t0 < WARM:
        okf, frame = cap.read()
        if okf:
            lmk.detect_for_video(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int((time.time() - t0) * 1000) + 1)
    lmk.close()

    names, X, y, blk, raw = None, [], [], [], {}
    subprocess.run(["say", "七個狀態，各五秒"])   # spoken: seven states, five seconds each
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
            if not res.face_blendshapes:
                continue
            bs = res.face_blendshapes[0]
            if names is None:
                names = [c.category_name for c in bs]
            rows.append([c.score for c in bs])
            h2 = int(frame.shape[0] * 480 / frame.shape[1])
            fr2 = cv2.resize(frame, (480, h2))
            cv2.putText(fr2, f"{key} ({'sing' if lab else 'not'})  "
                        f"{SEC - (now - t0):.0f}s", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if lab else (0, 165, 255), 2)
            cv2.imshow("sing gate", fr2)
            cv2.waitKey(1)
        lmk.close()
        r = np.array(rows)
        raw[key] = r
        X.append(r)
        y.append(np.full(len(r), lab))
        blk.append(np.minimum((np.arange(len(r)) * NB) // max(len(r), 1),
                              NB - 1))
        print(f"[{key}] {len(r)} frames", flush=True)
    cap.release()
    cv2.destroyAllWindows()
    X = np.vstack(X)
    y = np.concatenate(y)
    blk = np.concatenate(blk)
    np.savez("scratchpad/sing_gate_data.npz", names=np.array(names),
             **{f"bs_{k}": v for k, v in raw.items()})

    accs = []
    for b in range(NB):
        te = blk == b
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[~te]),
                                                    y[~te])
        accs.append((clf.predict(sc.transform(X[te])) == y[te]).mean())
    print(f"\ncross-validation accuracy over blocks of time {np.mean(accs):.3f}")

    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    W = clf.coef_[0].astype("float64")
    b0 = float(clf.intercept_[0])
    Z = (X - sc.mean_) / sc.scale_
    p = 1.0 / (1.0 + np.exp(-(Z @ W + b0)))
    assert abs(p - clf.predict_proba(sc.transform(X))[:, 1]).max() < 1e-9
    print("median probability of singing, per state:")
    for key, lab, _pr in STATES:
        Zk = (raw[key] - sc.mean_) / sc.scale_
        pk = 1.0 / (1.0 + np.exp(-(Zk @ W + b0)))
        print(f"  {key:<5}({'sing' if lab else 'not '})  "
              f"p50 {np.median(pk):.3f}  p10 {np.percentile(pk, 10):.3f}  "
              f"p90 {np.percentile(pk, 90):.3f}")
    np.savez("scratchpad/sing_gate_model.npz", W=W, b=np.float64(b0),
             mean=sc.mean_, scale=sc.scale_, names=np.array(names),
             cv=np.float64(np.mean(accs)))
    print(f"\nmodel -> scratchpad/sing_gate_model.npz ({W.size + 1} parameters)")
    # Choosing the threshold: singing must pass and not-singing must not. Read the
    # TAILS, not the medians.
    pos = np.concatenate([1.0 / (1.0 + np.exp(-(((raw[k] - sc.mean_)
                                                 / sc.scale_) @ W + b0)))
                          for k, l, _ in STATES if l])
    neg = np.concatenate([1.0 / (1.0 + np.exp(-(((raw[k] - sc.mean_)
                                                 / sc.scale_) @ W + b0)))
                          for k, l, _ in STATES if not l])
    for t in np.arange(0.3, 0.95, 0.05):
        if (neg > t).mean() <= 0.01:
            print(f"threshold {t:.2f}: singing passes {(pos > t).mean()*100:.0f}%, "
                  f"not singing false triggers {(neg > t).mean()*100:.1f}%")
            break


if __name__ == "__main__":
    main()
