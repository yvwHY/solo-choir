"""mouth_gate_probe — 閉嘴／微張／唱嗚 三態實測，定門檻用（08-13）

Harry：「門檻改 0.05，但唱嗚時嘴型又會小於門檻，怎麼辦」。
他的校準資料實測：嗚 開口值中位 **0.018**（比現行閉嘴門檻 0.020 還低），
其餘母音 0.050-0.098 ⇒ **單看開口高度，唱嗚與閉嘴不可分**，門檻怎麼調
都救不了。嗚的區別特徵是**嘴最窄**（0.297，嘟起來）。

本檔量三態（各 5s，語音報幕）：閉嘴休息／微微張嘴／唱嗚 → 回報
開口值 val 與嘴寬 wd 的分布，並自動找一條可分的規則：
  門開 = val > T_open  或  （val > T_min 且 wd < T_narrow）
若三態在這兩維上不可分，就誠實說不可分（那條路要換通道，例如接聲音）。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/mouth_gate_probe.py
"""
import subprocess
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

STATES = [("閉嘴休息", "閉著嘴，放鬆"),
          ("微微張嘴", "嘴巴微微張開，不要出聲"),
          ("唱嗚", "唱嗚，嘟嘴，拉長")]
SEC = 6.0
WARM = 3.0        # 鏡頭暖機（自動曝光/對焦未收斂時網格不穩＝第一段全歪）


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
    assert best and best[1] > 10, f"找不到有畫面的鏡頭 {best}"
    cap = cv2.VideoCapture(best[0])
    out = {}
    subprocess.run(["say", "先暖機三秒，請看畫面上的數字"])
    _w = vision.FaceLandmarker.create_from_options(opts)
    t0 = time.time()
    while time.time() - t0 < WARM:          # 丟掉暖機幀（血訓：08-13 第一
        okf, frame = cap.read()             # 段量到 0.076 比微張還高＝假資料）
        if okf:
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            _w.detect_for_video(img, int((now - t0) * 1000) + 1)
    _w.close()
    subprocess.run(["say", "嘴部門檻量測，三個狀態各六秒"])
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
            # 即時顯示＝他自己能看到量到什麼（血訓：拿沒看過的數字定門檻）
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
        print(f"[{name}] 幀 {len(v)}  開口值 中位 {np.median(v):.3f} "
              f"(p10 {np.percentile(v, 10):.3f} p90 {np.percentile(v, 90):.3f})"
              f"  嘴寬 中位 {np.median(w):.3f}", flush=True)
    cap.release()
    cv2.destroyAllWindows()

    rest, ajar, wu = out["閉嘴休息"], out["微微張嘴"], out["唱嗚"]
    print("\n── 找規則 ──")
    # 目標：嗚 要過、閉嘴與微張 要不過
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
        print("⚠ 這兩維分不開（嗚 過不了 或 微張會誤觸）——門檻救不了，"
              "要換通道（例如：門控加聲音能量/音高，或嗚 專屬形狀模型）")
        for nm in out:
            v, w = out[nm]
            print(f"   {nm}: val p50 {np.median(v):.3f} / wd p50 "
                  f"{np.median(w):.3f}")
        return
    cands.sort(reverse=True)
    _s, to, tm, tn, ok, bad = cands[0]
    print(f"建議規則：門開 = val > {to:.3f}  或  (val > {tm:.4f} 且 "
          f"wd < {tn:.3f})")
    print(f"  唱嗚通過率 {ok*100:.0f}%、閉嘴/微張誤觸 {bad*100:.0f}%")
    print(f"  → bank_live.py --mouth-open {to:.3f} --mouth-min {tm:.4f} "
          f"--mouth-narrow {tn:.3f}")


if __name__ == "__main__":
    main()
