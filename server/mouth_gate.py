"""mouth_gate — the camera mouth gate used by solo_min. Added 2026-08-14, after
"the speakers are sounding and I will also be talking".

This is a **twin** of `harmony/bank_live.py::_mouth_worker`, not a shared module.
Why it was copied rather than factored out: bank_live is the certified rehearsal
and performance engine, and touching it twelve days before the viva to remove a
duplication risks more than it gains. **A change to the gate logic on either
side has to be read against the other**; both carry this note. The shared
refactor waits until after the viva.

The two gates do different things, and their meanings should not be mixed:
  bank_live is a **sampler**. It detects the singer's pitch to trigger the
    sample bank, so its gate answers "does this f0 belong to the singer?" The
    parts coming from the speakers are perfectly periodic, and the detector
    would take them for the singer.
  solo_min is a **converter**. It turns what it hears straight into sound, so
    its gate answers "should the output sound at all?" A closed mouth means
    silent parts, which means nothing returns to the microphone to be converted
    again, and the feedback loop is broken here.

Why a level gate (solo_min's `--gate-floor`) cannot save this: it looks at the
**input level**, and while feedback is happening the microphone is loudly
hearing the speakers, so the level gate decides the singer is singing and holds
the gate open. It cannot help, structurally. A camera can, because **feedback
cannot fool the lips** (2026-08-12).

The model is `harmony/scratchpad/sing_gate_model.npz`, the one trained for
bank_live v34.1: 27 dimensions, mouth and jaw blendshapes only, 0.934 accuracy
across sessions, and 0.0% false triggers with the mouth closed or slightly open.
It uses the camera alone and has nothing to do with the sounding engine, so it
can be borrowed unchanged.
**Honest boundary: speech triggers it 11% of the time.** Picture alone cannot
separate singing from speech, since the mouth is open either way. Pushing that
to zero needs audio-side features (singing holds a pitch plateau, speech drifts
continuously in short runs), which is another piece of engineering.
"""
from __future__ import annotations
import os
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HARMONY = os.path.join(os.path.dirname(_HERE), "harmony")
DEF_MODEL = os.path.join(_HARMONY, "scratchpad", "sing_gate_model.npz")
DEF_TASK = os.path.join(_HARMONY, "scratchpad", "face_landmarker.task")


class MouthGate:
    """A camera thread plus a per-sample ramp. The audio thread calls only
    ramp(), which is arithmetic alone."""

    def __init__(self, model=DEF_MODEL, task=DEF_TASK, cam=-1,
                 sing_open=0.7, sing_hold=0.3, frames=2, grace=0.8,
                 ramp_s=0.12, fail_open=True, sr=48000, log=print):
        self.model_p, self.task_p, self.cam = model, task, cam
        self.sing_open, self.sing_hold, self.frames = sing_open, sing_hold, frames
        self.grace, self.fail_open, self.sr = grace, fail_open, sr
        self.log = log
        self._step = 1.0 / max(1e-6, ramp_s) / sr    # gain change per sample
        self._g = 0.0                                # current gain; closed at the start
        self.M = {"on": False, "arm": 0, "last_open": 0.0, "ok": False,
                  "sp": -1.0, "hb": None, "dead": None}
        self._die = False
        self._W = None

    # ---------- audio thread: this is the only method the callback uses -----
    def ramp(self, n):
        """Return a gain ramp of length n and advance the state. A ramp rather
        than a hard cut, which would click."""
        tgt = 1.0 if self._target_open() else 0.0
        g0 = self._g
        if g0 == tgt:
            self._g = tgt
            return np.full(n, np.float32(tgt), dtype=np.float32)
        d = self._step * n
        g1 = min(tgt, g0 + d) if tgt > g0 else max(tgt, g0 - d)
        out = np.linspace(g0, g1, n, dtype=np.float32)
        self._g = float(g1)
        return out

    def _target_open(self):
        if self.M["dead"] is not None:          # the thread has died
            return self.fail_open
        hb = self.M["hb"]
        if hb is None:
            # Before the camera has produced its first frame the gate stays
            # closed. Failing open would send the opening seconds straight into
            # feedback, the same mine as bank_live review R5#1.
            return False
        if time.time() - hb > 0.85:             # watchdog: read() has stopped, so the camera is dead
            if not self.M.get("_warned"):
                self.M["_warned"] = True
                self.log("⚠ [mouth] camera dead — gate %s"
                         % ("open (ungated)" if self.fail_open else "closed"))
            return self.fail_open
        if self.M.get("_warned"):
            self.M["_warned"] = False
            self.log("[gate] camera back")
        return self.M["ok"]

    # ---------- camera thread ----------
    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
        return self

    def stop(self):
        self._die = True

    def _load_model(self):
        z = np.load(self.model_p)
        W, b = z["W"], float(z["b"])
        mean, scale, idx = z["mean"], z["scale"], z["idx"]
        names = [str(x) for x in z["names"]]
        # The same validity check as bank_live: an invalid model is disabled
        # rather than guessed at (review S1/A5).
        # idx must be compared with W by **shape, not size**. Writing
        # `idx.shape != W.size` compares a tuple with an int, which is always
        # true, so the model was rejected every time. That happened on
        # 2026-08-14 and was only visible through the GATE DEAD line.
        if (W.shape != mean.shape or scale.shape != mean.shape
                or len(names) != W.size or idx.shape != W.shape
                or not np.isfinite(W).all() or (scale <= 0).any()):
            raise ValueError("sing_gate_model.npz failed validation")
        self._W, self._B, self._M_, self._S, self._I = W, b, mean, scale, idx
        self._CHK = names
        self.log(f"[gate] sing model loaded ({W.size + 1} params, "
                 f"mouth/jaw only, cross-session acc {float(z['cv']):.3f})")

    def _worker(self):
        try:
            self._load_model()
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            lmk = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=self.task_p),
                    running_mode=vision.RunningMode.VIDEO, num_faces=1,
                    output_face_blendshapes=True))
            cap = self._open_cam(cv2)
            t0 = now = time.time()
            last_det = 0.0
            while not self._die:
                okf, frame = cap.read()
                now = time.time()
                if okf:
                    # The heartbeat means a frame was really read. If read()
                    # fails, because the camera was taken or dropped, the
                    # heartbeat stops and the watchdog fires, rather than the
                    # gate closing silently for good.
                    self.M["hb"] = now
                if okf and now - last_det >= 0.033:            # ~30Hz
                    last_det = now
                    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = lmk.detect_for_video(img, int((now - t0) * 1000))
                    if not res.face_landmarks or not res.face_blendshapes:
                        # A lost face starts again rather than holding the old
                        # value. From the bank_live review: holding it would let
                        # turning away and back with the mouth closed open the
                        # gate.
                        self.M["on"], self.M["arm"] = False, 0
                    else:
                        self._score(res.face_blendshapes[0], now)
                # No face means the singer is not there, so the gate closes
                # (decided at v14.1). A camera that has really died is a
                # different case, handled by the watchdog above.
                self.M["ok"] = now - self.M["last_open"] < self.grace
                if not okf:
                    time.sleep(0.01)
            cap.release()
        except Exception as e:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.M["dead"] = str(e)
            self.log(f"⚠ [mouth] GATE DEAD ({e}) — gate "
                     f"{'open (ungated)' if self.fail_open else 'closed'}")

    def _open_cam(self, cv2):
        if self.cam >= 0:
            c = cv2.VideoCapture(self.cam)
            assert c.isOpened(), f"camera {self.cam} won't open"
            return c
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
        self.log(f"[gate] camera index {best[0]} (brightness {best[1]:.0f})")
        return cv2.VideoCapture(best[0])

    def _score(self, bs, now):
        if self._CHK is not None:
            # A change of mediapipe version silently shifts the layout and the
            # model reads different dimensions. Check once, for real, on the
            # first frame, and disable on a mismatch. Same as bank_live, whose
            # comment once claimed this was verified when it was not.
            live = [bs[i].category_name for i in self._I]
            if live != self._CHK:
                self.log("⚠ [gate] blendshape order != model — gate disabled")
                self._W = None
            self._CHK = None
        if self._W is None:
            return
        v = np.array([c.score for c in bs])[self._I]
        p = 1.0 / (1.0 + np.exp(-(np.dot((v - self._M_) / self._S, self._W)
                                  + self._B)))
        self.M["sp"] = float(p)
        if self.M["on"]:
            nv = p > self.sing_hold           # hysteresis: the threshold is lower while singing
            if not nv:
                self.M["arm"] = 0
        else:
            # Opening needs N consecutive frames, so a single noisy frame
            # cannot push the gate open.
            self.M["arm"] = self.M["arm"] + 1 if p > self.sing_open else 0
            nv = self.M["arm"] >= self.frames
        if nv != self.M["on"]:
            self.log(f"[gate] {'open' if nv else 'close'}  p {p:.3f}")
        self.M["on"] = nv
        if nv:
            self.M["last_open"] = now
