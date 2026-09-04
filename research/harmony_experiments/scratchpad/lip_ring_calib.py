"""lip_ring_calib - the full lip-ring mouth calibration ritual (v17.1).

v16.3 and v17 measured only two dimensions, mouth height and corner width, and two
of the vowels almost overlapped there (anchor distance 0.37). This stores the WHOLE
LIP RING instead: 40 points on the inner and outer lips, with translation removed
(the lip centre), rotation removed (the line between the temples) and scale removed
(the distance between the temples), giving an 80-dimensional vector. Vowels then
separate by lip rounding, corner shape and the curvature of both lips, not by
height and width alone.

The ritual, with its hard-won lessons built in: each of the five vowels is sung for
about 7 s sweeping from low to high, so the anchor averages out pitch and the scales
capture the variation; the spoken prompt uses a DESCRIPTION of the mouth shape so
two vowels cannot be confused by ear; and each take is checked for voicing
immediately (an RMS threshold, and a failed vowel is sung again) after an earlier
version silently recorded empty data.

Output:
  lip_ring_sweep.npz   the raw data: all 478 landmarks per frame plus the audio, so
                       changing the feature definition later can be recomputed
                       offline without recording again
  vowel_ring_calib.npz anchors (5,80) + scales (80,) + labels
                       - if bank_live finds this file it uses the full ring; without it, it falls back to two dimensions

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/lip_ring_calib.py [--seconds 7] [--in-name "USB PnP"]
Recompute without recording: add --recompute
"""
import argparse
import subprocess
import time

import numpy as np

SR = 44100
# The five vowel labels stay in Chinese: they must match bank_live.VOWEL_NAMES,
# which is compared by string and feeds the bank cache key.
LABELS = ["喔", "欸", "咿", "嗚", "啊"]      # the order matches VOWEL_LAYERS layers 0-4
ANNOUNCE = ["圓嘴的喔", "扁嘴的欸", "咧嘴的咿", "嘟嘴的嗚", "張大的啊"]  # spoken as a mouth-shape description

# Deliberately duplicated from _ring_vec and LIP_RING in bank_live.py: change one
# and change the other. Same discipline as _render_layer - the calibration and the
# live feature definition have to be identical to the bit.
LIP_RING = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
            314, 17, 84, 181, 91, 146,           # 20 on the outer ring
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95]            # 20 on the inner ring


def ring_vec(LA, aspect):
    """LA of shape (478,2) in normalised coordinates to an 80-dimensional lip ring. x is multiplied by the aspect ratio so distances are geometric."""
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
    """Capture one vowel: returns (landmarks (T,478,2), t (T,), audio (N,))."""
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
    """Voicing per frame: the audio RMS within 100 ms of the frame is at least a quarter of the take's 75th percentile."""
    if not len(ts):
        return np.zeros(0, dtype=bool)
    n = len(au)
    c = np.concatenate([[0.0], np.cumsum(au.astype("float64") ** 2)])
    w = int(0.1 * SR)
    i = np.clip((ts * SR).astype(int), 0, n - 1)   # ts shares its origin with the audio, in seconds
    lo = np.clip(i - w, 0, n - 1)
    hi = np.clip(i + w, 1, n)
    rms = np.sqrt((c[hi] - c[lo]) / np.maximum(hi - lo, 1))
    return rms >= 0.25 * np.percentile(rms, 75)


def compute_calib(raw):
    """raw dict to anchors and scales, plus a separation report."""
    vecs = []
    for vi in range(5):
        LA, ts, au = raw[f"l{vi}"], raw[f"t{vi}"], raw[f"a{vi}"]
        vm = voiced_mask(ts, au)
        keep = LA[vm]
        assert len(keep) >= 30, (LABELS[vi], "too few voiced frames", int(vm.sum()))
        vecs.append(np.array([ring_vec(x, float(raw["aspect"])) for x in keep]))
    A = np.array([v.mean(0) for v in vecs])
    S = np.maximum(np.concatenate(vecs).std(0), 1e-4)
    print("pairwise separation (mean |delta| over scale; larger is better):")
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
                    help="do not record; recompute anchors and scales from --raw")
    args = ap.parse_args()

    if args.recompute:
        raw = dict(np.load(args.raw))
        A, S = compute_calib(raw)
        np.savez(args.out, anchors=A, scales=S, labels=np.array(LABELS))
        print(f"recomputed -> {args.out}")
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
    else:                                    # as bank_live does: pick the brightest
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
        print(f"camera index {best[0]} (brightness {best[1]:.0f})", flush=True)
        cap = cv2.VideoCapture(best[0])
    assert cap.isOpened()

    raw = {}
    aspect = None
    say("整圈嘴校準。每個母音由低唱到高，唱滿提示秒數。")  # spoken prompt, Chinese
    for vi, (lab, ann) in enumerate(zip(LABELS, ANNOUNCE)):
        for attempt in range(3):
            lmk = vision.FaceLandmarker.create_from_options(opts)
            say(ann)
            time.sleep(0.5)
            say("唱")        # "sing"
            LA, ts, au = capture_vowel(cap, lmk, mp, cv2, args.in_name,
                                       args.seconds, lab)
            lmk.close()
            if aspect is None and len(LA):
                okf, fr0 = cap.read()
                aspect = fr0.shape[1] / fr0.shape[0] if okf else 4 / 3
            rms = float(np.sqrt((au ** 2).mean()))
            vm = voiced_mask(ts, au)
            print(f"[{lab}] {len(LA)} frames ({int(vm.sum())} voiced)"
                  f" audio RMS {20*np.log10(rms+1e-9):.1f} dBFS", flush=True)
            if rms >= 0.01 and int(vm.sum()) >= 30 and len(LA) >= 60:
                raw[f"l{vi}"], raw[f"t{vi}"], raw[f"a{vi}"] = LA, ts, au
                say("好")     # "good"
                break
            say("沒收到聲音或臉，重來")   # "no sound or face, again"
        else:
            raise SystemExit(f"{lab} failed three times, stopping (the lesson: never compute on empty data)")
    cap.release()
    cv2.destroyAllWindows()

    raw["aspect"] = np.float64(aspect or 4 / 3)
    np.savez(args.raw, **raw)
    print(f"raw data -> {args.raw}")
    A, S = compute_calib(raw)
    np.savez(args.out, anchors=A, scales=S, labels=np.array(LABELS))
    say("校準完成")   # "calibration complete"
    print(f"calibration -> {args.out} (anchors {A.shape})")


if __name__ == "__main__":
    main()
