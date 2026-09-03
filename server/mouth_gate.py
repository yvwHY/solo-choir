"""mouth_gate — 鏡頭嘴部門控（solo_min 用；08-14 Harry 裁「喇叭出聲、會講話」）。

⚠ 這是 `harmony/bank_live.py::_mouth_worker` 的**雙胞胎**，不是共用模組。
為什麼複製而不抽共用：bank_live 是已認證的排練/演出引擎，viva 前 12 天為了
消除重複去動它，風險大於收益。**任一邊改門控邏輯，另一邊要跟著看**——
兩邊都標了這段話。共用重構等 viva 之後。

兩邊的門控在做不同的事，別把它們的語意混起來：
  bank_live 是**取樣器**，偵測他的音高去觸發音庫 → 門控回答「這個 f0
    算不算他的」（喇叭裡的天使是完美週期訊號，偵測器會當成他）。
  solo_min 是**轉換器**，把聽到的直接變成聲音 → 門控回答「輸出要不要
    出聲」。他閉嘴＝天使閉嘴＝沒有東西繞回麥克風被再轉換一次，回授
    迴圈斷在這裡。

為什麼音量門（solo_min 的 `--gate-floor`）救不了回授：它看的是**輸入
音量**，而回授正在發生的時候麥克風正大聲聽著喇叭 ⇒ 音量門判定「他在
唱」把門開著。結構上救不了。鏡頭可以，因為**回授騙不了嘴唇**（08-12）。

模型＝`harmony/scratchpad/sing_gate_model.npz`（bank_live v34.1 訓的那顆，
27 維、只吃嘴/下巴 blendshape、跨場次 acc 0.934、閉嘴與微張誤觸 0.0%）。
它純粹吃鏡頭、與發聲引擎無關，所以可以原樣借用。
**誠實邊界：講話誤觸 11%**——純畫面分不出唱與講（嘴都是開的）。要壓到 0
得加音訊側特徵（唱＝音高踩住平台、講＝一直漂且每段短），那是另一個工程。
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
    """鏡頭執行緒＋逐樣本斜坡。audio thread 只呼叫 ramp()（純算術）。"""

    def __init__(self, model=DEF_MODEL, task=DEF_TASK, cam=-1,
                 sing_open=0.7, sing_hold=0.3, frames=2, grace=0.8,
                 ramp_s=0.12, fail_open=True, sr=48000, log=print):
        self.model_p, self.task_p, self.cam = model, task, cam
        self.sing_open, self.sing_hold, self.frames = sing_open, sing_hold, frames
        self.grace, self.fail_open, self.sr = grace, fail_open, sr
        self.log = log
        self._step = 1.0 / max(1e-6, ramp_s) / sr    # 每樣本的增益變化量
        self._g = 0.0                                # 目前增益（開場＝關）
        self.M = {"on": False, "arm": 0, "last_open": 0.0, "ok": False,
                  "sp": -1.0, "hb": None, "dead": None}
        self._die = False
        self._W = None

    # ---------- 音訊執行緒：只有這個會被 callback 呼叫 ----------
    def ramp(self, n):
        """回傳長度 n 的增益斜坡並前進狀態。斜坡＝不硬切（硬切會喀）。"""
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
        if self.M["dead"] is not None:          # 執行緒炸了
            return self.fail_open
        hb = self.M["hb"]
        if hb is None:
            # 開場鏡頭還沒吐第一幀：**關著**。fail-open 會讓開場那幾秒
            # 直接進回授（bank_live 審查 R5#1 的同一顆地雷）。
            return False
        if time.time() - hb > 0.85:             # 看門狗：read() 停了＝鏡頭死
            if not self.M.get("_warned"):
                self.M["_warned"] = True
                self.log("⚠ [mouth] camera dead — gate %s"
                         % ("open (ungated)" if self.fail_open else "closed"))
            return self.fail_open
        if self.M.get("_warned"):
            self.M["_warned"] = False
            self.log("[gate] camera back")
        return self.M["ok"]

    # ---------- 鏡頭執行緒 ----------
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
        # bank_live 同款合格檢查：不合格＝停用模型而不是猜（審查 S1/A5）
        # ⚠ idx 要跟 W **比 shape 不是比 size**——寫成 `idx.shape != W.size`
        # 是 tuple 比 int＝恆真＝模型每次都被判不合格（08-14 實案，靠
        # GATE DEAD 那行才看見）。
        if (W.shape != mean.shape or scale.shape != mean.shape
                or len(names) != W.size or idx.shape != W.shape
                or not np.isfinite(W).all() or (scale <= 0).any()):
            raise ValueError("sing_gate_model.npz 不合格")
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
                    # 心跳＝真的讀到 frame。read() 失敗（鏡頭被搶/掉線）
                    # ＝心跳停＝看門狗開火，而不是無聲地永遠關門。
                    self.M["hb"] = now
                if okf and now - last_det >= 0.033:            # ~30Hz
                    last_det = now
                    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = lmk.detect_for_video(img, int((now - t0) * 1000))
                    if not res.face_landmarks or not res.face_blendshapes:
                        # 臉丟＝重新來過（不是維持舊值）。bank_live 審查：
                        # 停在舊值會讓「轉頭再回正、閉著嘴」也開門。
                        self.M["on"], self.M["arm"] = False, 0
                    else:
                        self._score(res.face_blendshapes[0], now)
                # 臉不在＝他不在＝門關（v14.1 Harry 裁決）。鏡頭真死是另一
                # 個情境，由上面的看門狗處理。
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
            # mediapipe 換版就會靜默錯位＝模型讀到別的維度。首幀真的對一次，
            # 不符就停用（bank_live 同款；那邊的註解一度宣稱驗過但沒驗）。
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
            nv = p > self.sing_hold           # 遲滯：唱著的時候門檻低
            if not nv:
                self.M["arm"] = 0
        else:
            # 開門要連續 N 幀＝防單幀雜訊把門推開
            self.M["arm"] = self.M["arm"] + 1 if p > self.sing_open else 0
            nv = self.M["arm"] >= self.frames
        if nv != self.M["on"]:
            self.log(f"[gate] {'open' if nv else 'close'}  p {p:.3f}")
        self.M["on"] = nv
        if nv:
            self.M["last_open"] = now
