"""mouth_blendshape_probe - replace hand-coded geometry with mediapipe blendshapes

The question: the rounded vowel is still not reliably detected, is there a smarter
way? There is. We had been computing two distance ratios by hand (opening height and
mouth width), but mediapipe's FaceLandmarker can already output **52 blendshapes**,
among them `mouthPucker`, `mouthFunnel` and `jawOpen`, which are trained quantities
describing exactly this and are already normalised for head angle and distance. The
FaceBlendshapesGraph in the start-up log is that; we simply had its output turned
off.

This measures four states (rest, ajar, the rounded vowel, and an open vowel as a
positive control), 6 s each including a 3 s warm-up, with the values shown live -
numbers nobody has looked at must not set a parameter - and **the raw data saved to
an npz**, which the previous version forgot, so recomputing meant recording again.
Output: the blendshapes ranked by separation, plus a suggested gate rule.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/mouth_blendshape_probe.py
"""
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# The prompts stay in Chinese: they are spoken to the singer by macOS `say`.
STATES = [("rest", "閉著嘴，放鬆"), ("ajar", "嘴巴微微張開，不要出聲"),
          ("wu", "唱嗚，嘟嘴，拉長"), ("ah", "唱啊，拉長")]
SEC, WARM = 6.0, 3.0
WATCH = ["jawOpen", "mouthPucker", "mouthFunnel", "mouthClose",
         "mouthPressLeft", "mouthShrugUpper"]


def main():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="scratchpad/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True)          # this line is the smarter way
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
    subprocess.run(["say", "暖機三秒"])           # spoken: warming up, three seconds
    t0 = time.time()
    while time.time() - t0 < WARM:
        okf, frame = cap.read()
        if okf:
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lmk.detect_for_video(img, int((time.time() - t0) * 1000) + 1)
    lmk.close()

    names, out = None, {}
    subprocess.run(["say", "四個狀態，各六秒"])     # spoken: four states, six seconds each
    for key, prompt in STATES:
        lmk = vision.FaceLandmarker.create_from_options(opts)
        subprocess.run(["say", prompt])
        rows, geo = [], []
        t0 = time.time()
        while time.time() - t0 < SEC:
            okf, frame = cap.read()
            if not okf:
                continue
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int((now - t0) * 1000) + 1)
            if not res.face_landmarks or not res.face_blendshapes:
                continue
            bs = res.face_blendshapes[0]
            if names is None:
                names = [c.category_name for c in bs]
            rows.append([c.score for c in bs])
            f = res.face_landmarks[0]
            geo.append([abs(f[13].y - f[14].y)
                        / (abs(f[10].y - f[152].y) + 1e-9),
                        abs(f[61].x - f[291].x)
                        / (abs(f[234].x - f[454].x) + 1e-9)])
            h2 = int(frame.shape[0] * 480 / frame.shape[1])
            fr2 = cv2.resize(frame, (480, h2))
            d = dict(zip(names, rows[-1]))
            cv2.putText(fr2, f"{key}  {SEC - (now - t0):.0f}s", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(fr2, f"jawOpen {d['jawOpen']:.2f}  "
                        f"pucker {d['mouthPucker']:.2f}  "
                        f"funnel {d['mouthFunnel']:.2f}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("blendshape probe", fr2)
            cv2.waitKey(1)
        lmk.close()
        out[key] = (np.array(rows), np.array(geo))
        d = dict(zip(names, np.median(np.array(rows), 0)))
        print(f"[{key}] {len(rows)} frames  " + "  ".join(
            f"{w} {d[w]:.3f}" for w in WATCH), flush=True)
    cap.release()
    cv2.destroyAllWindows()
    np.savez("scratchpad/mouth_blendshapes.npz", names=np.array(names),
             **{f"bs_{k}": v[0] for k, v in out.items()},
             **{f"geo_{k}": v[1] for k, v in out.items()})

    print("\n-- what best separates the rounded vowel from closed and ajar --")
    wu = out["wu"][0]
    neg = np.vstack([out["rest"][0], out["ajar"][0]])
    sc = []
    for i, nm in enumerate(names):
        s = ((np.median(wu[:, i]) - np.median(neg[:, i]))
             / (np.std(np.concatenate([wu[:, i], neg[:, i]])) + 1e-6))
        sc.append((abs(s), nm, np.median(wu[:, i]), np.median(neg[:, i])))
    for s, nm, a_, b_ in sorted(sc, reverse=True)[:6]:
        print(f"  {nm:<22} separation {s:>5.2f}   vowel {a_:.3f} vs closed/ajar {b_:.3f}")
    # the geometric baseline, which the current gate uses
    g_wu, g_neg = out["wu"][1], np.vstack([out["rest"][1], out["ajar"][1]])
    for j, nm in enumerate(["opening (geometric)", "mouth width (geometric)"]):
        s = ((np.median(g_wu[:, j]) - np.median(g_neg[:, j]))
             / (np.std(np.concatenate([g_wu[:, j], g_neg[:, j]])) + 1e-6))
        print(f"  {nm:<22} separation {abs(s):>5.2f}")
    print("\n-- single-threshold sweep (vowel pass rate / closed-ajar false triggers) --")
    for _s, nm, a_, b_ in sorted(sc, reverse=True)[:3]:
        i = names.index(nm)
        for t in np.arange(0.05, 0.95, 0.05):
            hi = a_ > b_
            ok = (wu[:, i] > t).mean() if hi else (wu[:, i] < t).mean()
            bad = (neg[:, i] > t).mean() if hi else (neg[:, i] < t).mean()
            if ok >= 0.9 and bad <= 0.05:
                print(f"  {nm} {'>' if hi else '<'} {t:.2f}："
                      f"vowel passes {ok*100:.0f}%, false triggers {bad*100:.0f}%")
                break


if __name__ == "__main__":
    main()
