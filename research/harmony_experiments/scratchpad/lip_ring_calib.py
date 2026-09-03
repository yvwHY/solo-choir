"""lip_ring_calib — v17.1 整圈嘴校準儀式（Harry 08-13「為什麼不測整圈嘴」）

v16.3/v17 只量 (開口高, 嘴角寬) 2D＝欸↔咿 幾乎重疊（錨距 0.37）。這裡
改存**整圈唇形**：內外唇 40 點 → 去平移（唇心）/去旋轉（太陽穴連線）/
去尺度（太陽穴距離）→ 80 維向量。母音靠圓唇度/嘴角形狀/上下唇曲率分，
不只高寬。

儀式（血訓內建）：五母音各唱 ~7s 由低掃到高（錨點平均掉音高變化、
scales 吃到變異）；語音報幕用**描述詞**防聽混（圓嘴的喔≠張大的啊）；
每段錄完**立刻驗有聲**（RMS 門檻，不過就重唱該母音——lip_sweep v1
空資料血案）。

產出：
  lip_ring_sweep.npz   原始資料（全 478 landmark/幀＋音檔）＝之後改特徵
                       定義可離線重算，不用重錄
  vowel_ring_calib.npz anchors (5,80) + scales (80,) + labels
                       —— bank_live 有此檔＝自動走整圈嘴（沒有＝退 2D）

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/lip_ring_calib.py [--seconds 7] [--in-name "USB PnP"]
重算不重錄: 加 --recompute
"""
import argparse
import subprocess
import time

import numpy as np

SR = 44100
LABELS = ["喔", "欸", "咿", "嗚", "啊"]      # 順序＝VOWEL_LAYERS 層 0-4
ANNOUNCE = ["圓嘴的喔", "扁嘴的欸", "咧嘴的咿", "嘟嘴的嗚", "張大的啊"]

# ⚠ 與 bank_live.py 的 _ring_vec/LIP_RING 是**刻意複製**，改一處要改兩處
# （_render_layer 同款紀律：校準與 live 特徵定義必須逐位元同）
LIP_RING = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
            314, 17, 84, 181, 91, 146,           # 外圈 20
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95]            # 內圈 20


def ring_vec(LA, aspect):
    """LA=(478,2) 正規化座標 → 80 維整圈嘴形。x 乘長寬比＝幾何等距。"""
    pts = LA[LIP_RING].copy()
    pts[:, 0] *= aspect
    e0 = np.array([LA[234, 0] * aspect, LA[234, 1]])
    e1 = np.array([LA[454, 0] * aspect, LA[454, 1]])
    ax = e1 - e0
    sc = float(np.hypot(ax[0], ax[1])) + 1e-9
    th = float(np.arctan2(ax[1], ax[0]))
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c, -s], [s, c]])
    return ((pts - pts.mean(0)) @ R.T / sc).ravel()


def say(txt):
    subprocess.run(["say", txt])


def capture_vowel(cap, lmk, mp, cv2, in_name, seconds, label):
    """一個母音的擷取：回 (landmarks (T,478,2), t (T,), audio (N,))。"""
    import sounddevice as sd
    frames, ts, audio = [], [], []

    def _cb(indata, n, tinfo, status):
        audio.append(indata[:, 0].copy())

    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, device=in_name,
                        dtype="float32", callback=_cb):
        while time.time() - t0 < seconds:
            okf, frame = cap.read()
            if not okf:
                continue
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int((now - t0) * 1000) + 1)
            if res.face_landmarks:
                f = res.face_landmarks[0]
                LA = np.array([[p.x, p.y] for p in f])
                frames.append(LA)
                ts.append(now - t0)
                h2 = int(frame.shape[0] * 480 / frame.shape[1])
                fr2 = cv2.resize(frame, (480, h2))
                for i in LIP_RING:
                    cv2.circle(fr2, (int(f[i].x * 480), int(f[i].y * h2)),
                               1, (0, 255, 255), -1)
                cv2.putText(fr2, f"{label}  {seconds - (now - t0):.0f}s",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imshow("lip ring calib", fr2)
                cv2.waitKey(1)
    au = np.concatenate(audio) if audio else np.zeros(1, dtype="float32")
    return (np.array(frames) if frames else np.zeros((0, 478, 2)),
            np.array(ts), au)


def voiced_mask(ts, au):
    """逐幀有聲判定：該幀 ±100ms 音訊 RMS ≥ 全段 75 分位的 1/4。"""
    if not len(ts):
        return np.zeros(0, dtype=bool)
    n = len(au)
    c = np.concatenate([[0.0], np.cumsum(au.astype("float64") ** 2)])
    w = int(0.1 * SR)
    i = np.clip((ts * SR).astype(int), 0, n - 1)   # ts 與音訊同起點（秒）
    lo = np.clip(i - w, 0, n - 1)
    hi = np.clip(i + w, 1, n)
    rms = np.sqrt((c[hi] - c[lo]) / np.maximum(hi - lo, 1))
    return rms >= 0.25 * np.percentile(rms, 75)


def compute_calib(raw):
    """raw dict → anchors/scales＋分離度報表。"""
    vecs = []
    for vi in range(5):
        LA, ts, au = raw[f"l{vi}"], raw[f"t{vi}"], raw[f"a{vi}"]
        vm = voiced_mask(ts, au)
        keep = LA[vm]
        assert len(keep) >= 30, (LABELS[vi], "有聲幀不足", int(vm.sum()))
        vecs.append(np.array([ring_vec(x, float(raw["aspect"])) for x in keep]))
    A = np.array([v.mean(0) for v in vecs])
    S = np.maximum(np.concatenate(vecs).std(0), 1e-4)
    print("配對分離度（mean |Δ|/scale；越大越好）：")
    for i in range(5):
        for j in range(i + 1, 5):
            d = float(np.abs((A[i] - A[j]) / S).mean())
            print(f"  {LABELS[i]}↔{LABELS[j]}  {d:.2f}")
    return A, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--cam", type=int, default=-1)
    ap.add_argument("--raw", default="scratchpad/lip_ring_sweep.npz")
    ap.add_argument("--out", default="scratchpad/vowel_ring_calib.npz")
    ap.add_argument("--recompute", action="store_true",
                    help="不錄，直接用 --raw 重算 anchors/scales")
    args = ap.parse_args()

    if args.recompute:
        raw = dict(np.load(args.raw))
        A, S = compute_calib(raw)
        np.savez(args.out, anchors=A, scales=S, labels=np.array(LABELS))
        print(f"重算完成 → {args.out}")
        return

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="scratchpad/face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    if args.cam >= 0:
        cap = cv2.VideoCapture(args.cam)
    else:                                    # bank_live 同款：挑最亮的
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
        print(f"鏡頭 index {best[0]}（亮度 {best[1]:.0f}）", flush=True)
        cap = cv2.VideoCapture(best[0])
    assert cap.isOpened()

    raw = {}
    aspect = None
    say("整圈嘴校準。每個母音由低唱到高，唱滿提示秒數。")
    for vi, (lab, ann) in enumerate(zip(LABELS, ANNOUNCE)):
        for attempt in range(3):
            lmk = vision.FaceLandmarker.create_from_options(opts)
            say(ann)
            time.sleep(0.5)
            say("唱")
            LA, ts, au = capture_vowel(cap, lmk, mp, cv2, args.in_name,
                                       args.seconds, lab)
            lmk.close()
            if aspect is None and len(LA):
                okf, fr0 = cap.read()
                aspect = fr0.shape[1] / fr0.shape[0] if okf else 4 / 3
            rms = float(np.sqrt((au ** 2).mean()))
            vm = voiced_mask(ts, au)
            print(f"[{lab}] 幀 {len(LA)}（有聲 {int(vm.sum())}）"
                  f" 音 RMS {20*np.log10(rms+1e-9):.1f} dBFS", flush=True)
            if rms >= 0.01 and int(vm.sum()) >= 30 and len(LA) >= 60:
                raw[f"l{vi}"], raw[f"t{vi}"], raw[f"a{vi}"] = LA, ts, au
                say("好")
                break
            say("沒收到聲音或臉，重來")
        else:
            raise SystemExit(f"{lab} 三次都不合格，中止（血訓：不拿空資料算）")
    cap.release()
    cv2.destroyAllWindows()

    raw["aspect"] = np.float64(aspect or 4 / 3)
    np.savez(args.raw, **raw)
    print(f"原始資料 → {args.raw}")
    A, S = compute_calib(raw)
    np.savez(args.out, anchors=A, scales=S, labels=np.array(LABELS))
    say("校準完成")
    print(f"校準 → {args.out}（anchors {A.shape}）")


if __name__ == "__main__":
    main()
