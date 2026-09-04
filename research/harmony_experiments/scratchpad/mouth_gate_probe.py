"""mouth_gate_probe - measure three states (closed, ajar, singing one rounded vowel) to set the threshold

The question that started it: raising the threshold to 0.05 breaks the rounded
vowel, whose mouth shape falls below it. Measured on his own calibration data, that
vowel has a median opening of **0.018**, BELOW the current closed-mouth threshold of
0.020, while the other vowels sit between 0.050 and 0.098. So **on opening height
alone, that vowel and a closed mouth cannot be separated**, and no threshold helps.
What distinguishes it is that the mouth is at its NARROWEST (0.297, pursed).

This measures three states, 5 s each with a spoken prompt: resting closed, slightly
open, and singing the rounded vowel. It reports the distributions of the opening
value and the mouth width, and searches for a separable rule:
  gate open = val > T_open  or  (val > T_min and wd < T_narrow)
If the three states are inseparable in those two dimensions, it says so honestly,
which means that route needs a different channel, such as adding audio.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/mouth_gate_probe.py
"""
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# State names and prompts stay in Chinese: the prompt is spoken by macOS `say`, and
# the names key the results dict below.
STATES = [("閉嘴休息", "閉著嘴，放鬆"),
          ("微微張嘴", "嘴巴微微張開，不要出聲"),
          ("唱嗚", "唱嗚，嘟嘴，拉長")]
SEC = 6.0
WARM = 3.0        # camera warm-up: while exposure and focus are still settling the mesh is unstable and the first take is all wrong


def main():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="scratchpad/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
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
    out = {}
    subprocess.run(["say", "先暖機三秒，請看畫面上的數字"])   # spoken prompt
    _w = vision.FaceLandmarker.create_from_options(opts)
    t0 = time.time()
    while time.time() - t0 < WARM:          # discard the warm-up frames: the first take
        okf, frame = cap.read()             # once measured 0.076, higher than ajar, which was false data
        if okf:
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _w.detect_for_video(img, int((now - t0) * 1000) + 1)
    _w.close()
    subprocess.run(["say", "嘴部門檻量測，三個狀態各六秒"])   # spoken prompt
    for name, prompt in STATES:
        lmk = vision.FaceLandmarker.create_from_options(opts)
        subprocess.run(["say", prompt])
        vals, wds = [], []
        t0 = time.time()
        while time.time() - t0 < SEC:
            okf, frame = cap.read()
            if not okf:
                continue
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int((now - t0) * 1000) + 1)
            if not res.face_landmarks:
                continue
            f = res.face_landmarks[0]
            v_ = abs(f[13].y - f[14].y) / (abs(f[10].y - f[152].y) + 1e-9)
            w_ = abs(f[61].x - f[291].x) / (abs(f[234].x - f[454].x) + 1e-9)
            vals.append(v_)
            wds.append(w_)
            # shown live, so he can see what is being measured; never set a threshold from numbers nobody has looked at
            h2 = int(frame.shape[0] * 480 / frame.shape[1])
            fr2 = cv2.resize(frame, (480, h2))
            cv2.putText(fr2, f"{name}  {SEC - (now - t0):.0f}s", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(fr2, f"open {v_:.3f}   width {w_:.3f}", (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("mouth probe", fr2)
            cv2.waitKey(1)
        lmk.close()
        v, w = np.array(vals), np.array(wds)
        out[name] = (v, w)
        print(f"[{name}] {len(v)} frames  opening median {np.median(v):.3f} "
              f"(p10 {np.percentile(v, 10):.3f} p90 {np.percentile(v, 90):.3f})"
              f"  mouth width median {np.median(w):.3f}", flush=True)
    cap.release()
    cv2.destroyAllWindows()

    rest, ajar, wu = out["閉嘴休息"], out["微微張嘴"], out["唱嗚"]
    print("\n-- searching for a rule --")
    # the goal: the rounded vowel passes, closed and ajar do not
    cands = []
    for t_open in np.arange(0.03, 0.09, 0.005):
        for t_min in np.arange(0.005, 0.03, 0.0025):
            for t_narrow in np.arange(0.26, 0.36, 0.005):
                def gate(v, w):
                    return (v > t_open) | ((v > t_min) & (w < t_narrow))
                ok_wu = gate(*wu).mean()
                bad = max(gate(*rest).mean(), gate(*ajar).mean())
                if ok_wu >= 0.85 and bad <= 0.10:
                    cands.append((ok_wu - bad, t_open, t_min, t_narrow,
                                  ok_wu, bad))
    if not cands:
        print("These two dimensions cannot separate them (either the vowel fails or ajar "
              "triggers). No threshold fixes this; it needs a different channel, such as "
              "adding audio energy or pitch to the gate, or a shape model for that vowel.")
        for nm in out:
            v, w = out[nm]
            print(f"   {nm}: val p50 {np.median(v):.3f} / wd p50 "
                  f"{np.median(w):.3f}")
        return
    cands.sort(reverse=True)
    _s, to, tm, tn, ok, bad = cands[0]
    print(f"suggested rule: gate open = val > {to:.3f}  or  (val > {tm:.4f} and "
          f"wd < {tn:.3f})")
    print(f"  the rounded vowel passes {ok*100:.0f}%, closed/ajar false triggers {bad*100:.0f}%")
    print(f"  → bank_live.py --mouth-open {to:.3f} --mouth-min {tm:.4f} "
          f"--mouth-narrow {tn:.3f}")


if __name__ == "__main__":
    main()
