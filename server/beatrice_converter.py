"""Beatrice v2 (rc.0) realtime converter with Solo Choir diatonic harmony.

Drives the user's trained tenor through the Beatrice v2 rc.0 engine
(`SimpleBeatrice`) from editable Python, per 10ms frame:

    mic frame (160 @16k)
      -> [external f0 detector] -> input pitch (Hz)
      -> [Solo Choir] f0 -> MIDI -> diatonic transpose (semitones)
      -> SimpleBeatrice.set_pitch_shift_semitone(transpose)
      -> SimpleBeatrice.convert(frame) -> tenor harmony out (@24k)

rc.0 hides the model's internal f0, so we estimate the input pitch ourselves
(lightweight autocorrelation on a short rolling window) and feed the shared
voice_changer/SoloChoir.py logic. The diatonic transpose is applied via the
engine's own per-call semitone TUNE — fully in this editable layer.

Engine binaries are NOT in the repo (Beatrice license). Point BEATRICE_ENGINE_DIR
at the extracted rc.0 engine and pass the model dir of paraphernalia .bin files.
"""
from __future__ import annotations

import os
import sys
import numpy as np

from voice_changer.SoloChoir import (SoloChoirHarmonizer,
                                     diatonic_target_midi, scale_pcs_for)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEATRICE_ENGINE as _BEATRICE_ENGINE  # noqa: E402
_DEFAULT_ENGINE = str(_BEATRICE_ENGINE)


def estimate_f0_autocorr(buf: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 600.0) -> float:
    """Fast monophonic f0 via normalized autocorrelation. 0.0 if unvoiced/quiet."""
    x = buf.astype(np.float64)
    x = x - x.mean()
    energy = np.sqrt(np.mean(x * x))
    if energy < 1e-3:  # silence
        return 0.0
    tau_min = int(sr / fmax)
    tau_max = min(int(sr / fmin), len(x) - 1)
    if tau_max <= tau_min:
        return 0.0
    # autocorrelation via FFT
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n)
    ac = np.fft.irfft(f * np.conj(f), n)[: tau_max + 1]
    ac0 = ac[0] if ac[0] != 0 else 1e-9
    seg = ac[tau_min:tau_max + 1]
    if seg.size == 0:
        return 0.0
    k = int(np.argmax(seg)) + tau_min
    if ac[k] / ac0 < 0.3:  # weak periodicity -> treat as unvoiced
        return 0.0
    # parabolic interpolation around the peak
    if 1 <= k < tau_max:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        denom = (a - 2 * b + c)
        if denom != 0:
            k = k + 0.5 * (a - c) / denom
    return sr / k if k > 0 else 0.0


class BeatriceSoloChoir:
    def __init__(
        self,
        model_dir: str,
        engine_dir: str | None = None,
        version: str = "2.0.0-rc.0",
        target_speaker: int = 0,
        formant_shift: float = 0.0,
        min_source_pitch: float = 40.0,
        max_source_pitch: float = 80.0,
        vq_num_neighbors: int = 8,
        f0_window: int = 2048,
        harmonizer: SoloChoirHarmonizer | None = None,
        output_gate_floor: float = 0.0,   # 0 = off (envelope tuning helps only in silent gaps)
        output_gate_smooth: float = 0.3,
        output_gate_expon: float = 1.0,   # attenuation curve: 1 = linear (historical); 2 = square-law,
                                          # twice the dB cut below the floor, identical at/above it.
                                          # Needed when a downstream choir gain lifts the voices: a
                                          # linear gate's cut is exactly cancelled by +18 dB of gain.
    ):
        engine_dir = engine_dir or os.environ.get("BEATRICE_ENGINE_DIR", _DEFAULT_ENGINE)
        if engine_dir not in sys.path:
            sys.path.insert(0, engine_dir)
        import beatrice  # from the extracted engine dir

        self.m = beatrice.load_beatrice(version)
        self.in_sr = self.m.IN_SAMPLE_RATE      # 16000
        self.out_sr = self.m.OUT_SAMPLE_RATE    # 24000
        self.hop = self.m.IN_HOP_LENGTH         # 160

        def _p(name):
            return os.path.join(model_dir, name)

        self.sb = self.m.SimpleBeatrice(
            _p("phone_extractor.bin"),
            _p("pitch_estimator.bin"),
            _p("embedding_setter.bin"),
            _p("waveform_generator.bin"),
            _p("speaker_embeddings.bin"),
        )
        if not self.sb.is_ready():
            raise RuntimeError(f"Beatrice model not ready (last_error={self.sb.last_error()})")
        self.sb.set_config(
            target_speaker=target_speaker,
            formant_shift=formant_shift,
            pitch_shift_semitone=0.0,
            min_source_pitch=min_source_pitch,
            max_source_pitch=max_source_pitch,
            vq_num_neighbors=vq_num_neighbors,
        )

        self.harmonizer = harmonizer or SoloChoirHarmonizer()
        self._f0buf = np.zeros(f0_window, dtype=np.float32)
        self._cur_shift = 0.0
        # Output envelope (volume) tuning: scale the converted output by the input
        # level so the model's noise floor (and any mic hiss converted with it) is
        # suppressed during quiet/non-voiced moments, while actual singing passes at
        # full level. This is the dominant fix for the live "buzz" (mic hiss exposed
        # by the always-on vocoder); mirrors the VCClient's sqrt volume tuning.
        self.output_gate_floor = output_gate_floor   # input RMS at/above which output is full
        self.output_gate_smooth = output_gate_smooth  # EMA on the gain to avoid pumping/clicks
        self.output_gate_expon = output_gate_expon   # 1 = linear (historical), >1 = steeper below floor
        self._env_gain = 0.0
        # Optional READ-ONLY telemetry sink, set by the app bridge (None = no-op).
        # Does not affect any DSP: it only reports values process_frame already computes.
        self.telemetry_tap = None
        # Live "Pitch" control (semitones, formant-preserving): added to the engine TUNE.
        # 0.0 → identical to before; does NOT change the diatonic/f0 math.
        self.pitch_offset = 0.0
        # Pitch GLIDE (kill the per-note transition scrape): instead of jumping the engine TUNE to the
        # new target, ramp toward it in <=pitch_glide_step semitones/hop, and only accept a new target
        # after it persists pitch_hold_frames hops (debounce). 0/0 → OFF → byte-identical (jump as before).
        # Decompiling vcclient proved its clean pitch-drag = exactly this fine ramp (set_config at cent
        # increments, never a per-block jump). Sweet spot by ear: step 0.03–0.06, hold 8. See memory
        # pitch-transition-scrape-diagnosis.
        self.pitch_glide_step = 0.0       # semitones per hop; 0 = disabled
        self.pitch_hold_frames = 0        # debounce: hops a new target must persist before accepted
        self._glide_cand = None
        self._glide_cand_cnt = 0
        self._glide_target = 0.0
        # Quantized-angel mode (2026-08-18, Harry「和諧度不如 ddsp 離線版」):
        # OFF (default) = the voice is the singer's live contour + an integer
        # shift, so his every intonation wobble is copied onto every voice and
        # the chord has NO in-tune anchor. ON = the voice sings the ABSOLUTE
        # snapped target (held note + diatonic shift): the singer's cents
        # deviation is cancelled per hop, and the voice carries its OWN slow
        # vibrato instead (a flat robot note reads as autotune — hard no).
        # This is bank_live's recipe (quantized note bank + --vib) at the
        # neural engine's frame rate. All attrs poked by the wrapper; nothing
        # in the app sets them → default path byte-identical.
        self.quantize = False
        self.q_glide = 0.25               # target ramp, st/hop（換音 ~140ms 內完成＝
                                          # 圓滑線速度；0.05 版被 Harry 判「滑音嚴重」）
        self.q_rest_hops = 20             # 靜默超過此 hop 數（200ms）＝新句：目標
                                          # 直接落點，不從上一句的音滑過來
        self._q_uv = 0
        self.vib_cents = 0.0              # own vibrato depth (peak, cents)
        self.vib_hz = 5.0                 # own vibrato rate
        self.vib_onset_hops = 30          # 起音先直、顫音 ~300ms 漸開（真人唱法；
                                          # 純正弦一開口就滿幅＝合成感）
        self._vib_ph = 0.0
        self._vib_am_ph = 0.0             # 深度慢速呼吸（~0.5Hz ±15%）
        self._q_age = 0                   # 距上次換音的 hop 數（顫音漸開用）
        self._qtgt = None                 # ramped absolute target (MIDI float)
        self.q_snap = 3.5                 # 目標跳動超過此半音數＝直接落點不滑
                                          # （音級圓滑線最多 ~四五度；更大＝偵測
                                          # 修正或真大跳，滑過去＝「極速滑音」實案）
        self._q_cont3 = []                # 句首進場檢查：連續 3 hop 音高一致
        # 補償項平滑（08-18 Harry「聲部還是會抖」）：applied=目標−cont，
        # cont 的逐 hop 估計噪音（幾 cents）會原封印在天使音上。中值(5)＋
        # EMA(0.35)＝~45ms 延遲換掉估計噪音；長音穩定拿到、真移動照樣跟。
        self.q_comp_smooth = True
        self._q_med5 = []
        self._q_cs = None
        self.q_commit_hops = 4            # 新目標要連續站穩 40ms 才採納＝圓滑
                                          # 路過的中間音（<40ms）不再被咬住
        self._q_held = None               # quantize 自管的已提交音符
        self._q_raw_held = None           # 目標快取（held 換了才重算）
        self._q_raw_tgt = None
        self._q_far = 0                   # 連續「遠離 held」hop 數：1–2＝飛點
                                          # （凍結）；≥3＝他真的在移動（釘住舊
                                          # 目標，別讓輸出跟著走——165c 走音實案）
        # 純律（08-18 和諧度）：天使相對「他的音」取純律比值而非平均律格。
        # 四度只差 2c，但三度差 13.7c＝換音程時這裡就是「鎖住」與「將就」的差。
        self.q_ji = False
        # 平台跟他調音：共享 dict（各聲部指到同一個）＝和弦內部鎖死、整體
        # 慢慢貼上他的音準中心（τ ≈ 2.5s；只在他站在音上時更新；夾 ±0.35st）。
        self.q_tune = None                # {"off": st}；None＝關
        self.q_tune_lead = False          # 只有 leader 聲部更新（其餘只讀）

    def process_frame(self, seg16k: np.ndarray, f0_override: float | None = None) -> np.ndarray:
        """seg16k: 160 float32 @16k. Returns the tenor (harmony) output @24k.
        f0_override (Task 2.4): reuse a shared f0 estimate instead of recomputing the autocorr FFT.
        All voices see the SAME 16k input, so their f0 buffers/estimates are identical — computing it
        once on the reference voice and injecting it here is byte-identical, just cheaper."""
        seg = np.ascontiguousarray(seg16k, dtype=np.float32)
        vol = float(np.sqrt(np.mean(seg * seg)))  # input level this frame
        # roll the f0 window and append this frame
        self._f0buf = np.roll(self._f0buf, -len(seg))
        self._f0buf[-len(seg):] = seg

        f0 = 0.0       # input pitch (Hz); 0 = unvoiced/disabled. Only computed when harmonizing.
        diatonic = 0.0  # diatonic harmony shift (semitones), 0 when not harmonizing
        if self.harmonizer.enabled:
            f0 = estimate_f0_autocorr(self._f0buf, self.in_sr) if f0_override is None else float(f0_override)
            if not self.quantize:
                # ALWAYS call compute_shift: on an unvoiced frame representative_f0() returns None and
                # compute_shift holds the last shift (SoloChoir.py:199-201) instead of the old `if f0 > 0
                # else None`, which forced diatonic=0 and jumped the TUNE mid-sustain → neural crackle.
                shift = self.harmonizer.compute_shift(np.array([f0], dtype=np.float32))
                diatonic = float(shift) if shift is not None else 0.0
            # quantize 模式**不跑**舊判音鏈：它的輸出這裡用不到，而它的副作
            # 用（median+遲滯的 held、auto_chord 的和弦狀態）會跟量化的判音
            # 搶同一個和弦記憶＝雙重驅動（08-18 free 接線時定的雷）。
        self._last_f0 = f0   # Task 2.4: expose the f0 used this frame so peer voices can reuse it
        if self.quantize and self.harmonizer.enabled:
            # Quantized angel（08-18，v2 重寫）：天使唱「他所站音符」的調內
            # 絕對目標＋自有顫音；他的連續音高逐 hop 被抵消。音符提交不再走
            # harmonizer 的 median+遲滯鏈——那條鏈會把圓滑「路過值」提交成
            # held（實案：降全音時 held 卡 46、他站 45、差 1.0 恰在遲滯縫上
            # ＝天使永遠不跟）。這裡的唯一提交來源＝「站穩的音高」（3 hop
            # 範圍 <0.35st）：路過值天生站不穩＝進不來；落地 30ms 即提交。
            if f0 <= 0.0:
                self._q_uv += 1               # 靜默過門檻＝下一句從頭起音
                if self._q_uv > self.q_rest_hops and self._q_held is not None:
                    self._q_held = None
                    self._qtgt = None
                    self._q_cont3 = []
                    self._q_med5 = []
                    self._q_cs = None
            else:
                self._q_uv = 0
                cont = 69.0 + 12.0 * float(np.log2(f0 / 440.0))
                self._q_cont3 = (self._q_cont3 + [cont])[-3:]
                med3 = float(np.median(self._q_cont3))
                stab = (len(self._q_cont3) == 3
                        and max(self._q_cont3) - min(self._q_cont3) < 0.35)
                if stab and (self._q_held is None
                             or abs(med3 - self._q_held) > 0.6):
                    self._q_held = int(round(med3))   # 0.6＝提交遲滯（顫音/
                                                      # 飄移不觸發、換音必觸發）
                if self._q_held is None:
                    pass                              # 句首還沒站穩＝天使不進場
                elif abs(cont - self._q_held) > 2.5:
                    self._q_far += 1
                    if self._q_far >= 3 and self._qtgt is not None:
                        # 持續遠離＝他真的在大幅移動：用當下 cont 釘住舊目標
                        # ＝守音；凍結會讓輸出跟著他走出走音（165c 實案）
                        tune = (self.q_tune["off"]
                                if self.q_tune is not None else 0.0)
                        ns = self._qtgt + tune + self.pitch_offset - cont
                        if ns != self._cur_shift:
                            self.sb.set_pitch_shift_semitone(ns)
                            self._cur_shift = ns
                    # 前 2 hop＝可能是飛點：維持上一個 shift
                else:
                    self._q_far = 0
                    if self._q_raw_held != self._q_held:
                        # held 換了才重算一次（auto_chord 有狀態：和弦記憶＋
                        # voice-lead 的自家上一音；逐 hop 重呼叫既浪費又可能
                        # 在邊界打抖）。和弦狀態的**唯一**驅動源＝量化 held。
                        if getattr(self.harmonizer, "auto_chord", False):
                            self._q_raw_tgt = float(
                                self.harmonizer._auto_chord_target(self._q_held))
                        else:
                            self._q_raw_tgt = float(diatonic_target_midi(
                                self._q_held, self.harmonizer.key_root,
                                scale_pcs_for(self.harmonizer.minor),
                                self.harmonizer.interval_steps))
                        self._q_raw_held = self._q_held
                    tgt = self._q_raw_tgt
                    if self._qtgt is None:
                        self._qtgt = tgt              # 新句＝直接落點，零滑音
                        self._q_age = 0
                    d = tgt - self._qtgt
                    if abs(d) > 0.3:
                        self._q_age = 0               # 換音＝顫音重新漸開
                    else:
                        self._q_age += 1
                    if abs(d) > self.q_snap:
                        self._qtgt = tgt              # 大跳/修正＝直接落點
                    elif abs(d) <= self.q_glide:
                        self._qtgt = tgt
                    else:
                        self._qtgt += self.q_glide if d > 0 else -self.q_glide
                    self._vib_ph += 2.0 * np.pi * self.vib_hz * 0.01
                    self._vib_am_ph += 2.0 * np.pi * 0.5 * 0.01
                    env = min(1.0, self._q_age / max(1, self.vib_onset_hops))
                    depth = (self.vib_cents / 100.0) * env \
                        * (1.0 + 0.15 * float(np.sin(self._vib_am_ph)))
                    vib = depth * float(np.sin(self._vib_ph))
                    ji = 0.0
                    if self.q_ji:
                        iv = int(round(tgt - self._q_held))
                        k, oc = abs(iv) % 12, abs(iv) // 12
                        _PURE = (0.0, 111.7, 203.9, 315.6, 386.3, 498.0,
                                 590.2, 702.0, 813.7, 884.4, 996.1, 1088.3)
                        pure = oc * 1200.0 + _PURE[k]
                        ji = ((pure if iv >= 0 else -pure) - iv * 100.0) / 100.0
                    tune = 0.0
                    if self.q_tune is not None:
                        dev = cont - self._q_held
                        if self.q_tune_lead and abs(dev) < 0.5:
                            o = self.q_tune["off"]
                            o += 0.004 * (dev - o)            # τ≈2.5s
                            self.q_tune["off"] = max(-0.35, min(0.35, o))
                        tune = self.q_tune["off"]
                    comp = cont
                    if self.q_comp_smooth:
                        self._q_med5 = (self._q_med5 + [cont])[-5:]
                        m = float(np.median(self._q_med5))
                        if self._q_cs is None or abs(cont - self._q_cs) > 0.35:
                            self._q_cs = cont             # 死區：大動即時跟
                            self._q_med5 = [cont]
                        else:
                            self._q_cs += 0.35 * (m - self._q_cs)
                        comp = self._q_cs
                    new_shift = self._qtgt + ji + tune + vib \
                        + self.pitch_offset - comp
                    if new_shift != self._cur_shift:
                        self.sb.set_pitch_shift_semitone(new_shift)
                        self._cur_shift = new_shift
        else:
            # engine TUNE = diatonic harmony shift + live Pitch control. pitch_offset defaults 0
            # → byte-identical to before; the diatonic math above is unchanged.
            new_shift = diatonic + self.pitch_offset
            if self.pitch_glide_step <= 0.0:
                # OFF → original behaviour (byte-identical): jump straight to the target
                if new_shift != self._cur_shift:
                    self.sb.set_pitch_shift_semitone(new_shift)
                    self._cur_shift = new_shift
            elif getattr(self.harmonizer, "snap_shift_hint", False) and new_shift != self._cur_shift:
                # voice-lead common-tone hold: the harmony PITCH is unchanged (the shift change merely
                # compensates the singer's move) — jump so the held note stays continuous; gliding here
                # made the harmony swoop with the singer then slide back. False everywhere but voice_lead.
                self._glide_target = new_shift; self._glide_cand = None; self._glide_cand_cnt = 0
                self._cur_shift = new_shift
                self.sb.set_pitch_shift_semitone(new_shift)
            else:
                # debounce the target, then glide the ENGINE value toward it in small per-hop steps
                if new_shift != self._glide_target:
                    if new_shift == self._glide_cand:
                        self._glide_cand_cnt += 1
                    else:
                        self._glide_cand = new_shift; self._glide_cand_cnt = 1
                    if self._glide_cand_cnt >= max(1, self.pitch_hold_frames):
                        self._glide_target = new_shift; self._glide_cand = None; self._glide_cand_cnt = 0
                if self._cur_shift != self._glide_target:
                    d = self._glide_target - self._cur_shift
                    step = self.pitch_glide_step if abs(d) > self.pitch_glide_step else abs(d)
                    self._cur_shift += step if d > 0 else -step
                    self.sb.set_pitch_shift_semitone(self._cur_shift)

        # convert() returns (out_block @24k, n). Emit the FULL block: it is exactly
        # OUT_HOP_LENGTH per IN_HOP_LENGTH input (10ms->10ms, time ratio 1.0). The
        # returned n is an internal count, NOT the emit length (using it underruns).
        y, _n = self.sb.convert(seg)
        y = np.asarray(y, dtype=np.float32).ravel()

        # output envelope tuning (smoothed): full level when singing, suppressed when quiet
        if self.output_gate_floor > 0:
            target = min(1.0, vol / self.output_gate_floor)
            if self.output_gate_expon != 1.0:        # steeper cut below the floor; ratio >= 1 unchanged
                target = target ** self.output_gate_expon
            a = self.output_gate_smooth
            self._env_gain = a * target + (1.0 - a) * self._env_gain
            y = y * self._env_gain

        # READ-ONLY telemetry: report the values just computed; never alters y or the flow.
        if self.telemetry_tap is not None:
            out_rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
            self.telemetry_tap(f0=float(f0), in_level=float(vol), out_level=out_rms, voiced=bool(f0 > 0.0))
        return y
