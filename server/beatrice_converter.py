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
        # Quantized-parts mode (2026-08-18, after "less in tune than the offline ddsp version"):
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
        self.q_glide = 0.25               # target ramp in semitones per hop; a note change
                                          # completes in about 140 ms, the speed of a
                                          # smooth line. The 0.05 version was judged
                                          # to have heavy portamento.
        self.q_rest_hops = 20             # silence longer than this many hops (200 ms)
                                          # starts a new phrase: the target lands
                                          # directly instead of gliding from the
                                          # last note of the previous phrase
        self._q_uv = 0
        self.vib_cents = 0.0              # own vibrato depth (peak, cents)
        self.vib_hz = 5.0                 # own vibrato rate
        self.vib_onset_hops = 30          # straight at the attack, with vibrato opening
                                          # over about 300 ms, as a singer does. A
                                          # pure sine at full depth from the first
                                          # instant sounds synthetic.
        self._vib_ph = 0.0
        self._vib_am_ph = 0.0             # slow breathing in depth, about 0.5 Hz at +/-15%
        self._q_age = 0                   # hops since the last note change, for the vibrato ramp
        self._qtgt = None                 # ramped absolute target (MIDI float)
        self.q_snap = 3.5                 # a target jump larger than this many semitones
                                          # lands directly instead of gliding. A
                                          # smooth diatonic line moves at most a
                                          # fourth or fifth; anything larger is a
                                          # detection correction or a real leap, and
                                          # gliding it produced the extremely fast
                                          # slide that was reported.
        self._q_cont3 = []                # entry check at the start of a phrase: the pitch
                                          # must agree over three consecutive hops
        # Smoothing the compensation term, after "the parts still wobble"
        # (2026-08-18). applied = target - cont, so the few cents of per-hop
        # estimation noise in cont would print straight onto the part. A median
        # of 5 with an EMA of 0.35 trades about 45 ms of delay for that noise:
        # sustained notes come out steady, and real movement is still followed.
        self.q_comp_smooth = True
        self._q_med5 = []
        self._q_cs = None
        self.q_commit_hops = 4            # a new target is adopted only after holding for
                                          # 40 ms, so passing notes inside a smooth
                                          # line, which last under 40 ms, are no
                                          # longer latched onto
        self._q_held = None               # the committed note, owned by quantize
        self._q_raw_held = None           # cached target, recomputed only when held changes
        self._q_raw_tgt = None
        self._q_far = 0                   # consecutive hops far from held: 1-2 is an
                                          # outlier and is frozen; 3 or more means the
                                          # singer really is moving, so the old target
                                          # is pinned rather than followed, which is
                                          # what caused the 165-cent excursion
        # Just intonation (2026-08-18): the parts take just ratios against the
        # singer's note rather than the equal-tempered grid. A fourth differs by
        # only 2 cents, but a third by 13.7, which is the difference between a
        # chord that locks and one that merely gets by.
        self.q_ji = False
        # The ensemble tunes to the singer through a shared dict that every
        # part points at, so the chord locks internally while the whole slowly
        # tracks the singer's tuning centre (time constant about 2.5 s, updated
        # only while they are holding a note, clamped to +/-0.35 semitones).
        self.q_tune = None                # {"off": st}; None disables it
        self.q_tune_lead = False          # only the leader part updates it; the others read

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
            # In quantize mode the old note chain does not run: its output is
            # unused here, and its side effects, the median-plus-hysteresis held
            # and auto_chord's chord state, would compete with the quantised
            # note detection for the same chord memory, which is the double
            # drive identified when free was wired in on 2026-08-18.
        self._last_f0 = f0   # Task 2.4: expose the f0 used this frame so peer voices can reuse it
        if self.quantize and self.harmonizer.enabled:
            # Quantised parts (2026-08-18, rewritten as v2): each part sings
            # the absolute in-key target of the note the singer is standing on,
            # with its own vibrato, and the singer's continuous pitch is
            # cancelled hop by hop. Notes are no longer committed through the
            # harmoniser's median-plus-hysteresis chain, which committed passing
            # values of a smooth line as held. In one case, descending a whole
            # tone, held stuck at 46 while the singer stood on 45, and the
            # difference of 1.0 fell exactly in the hysteresis gap, so the part
            # never followed. The only source of a commit here is a pitch that
            # holds still, within 0.35 semitones over three hops. A passing value
            # cannot hold still and so cannot enter; a landed note commits after
            # 30 ms.
            if f0 <= 0.0:
                self._q_uv += 1               # silence past the threshold: the next phrase attacks afresh
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
                    self._q_held = int(round(med3))   # 0.6 is the commit hysteresis: vibrato
                                                      # and drift do not trigger it, a
                                                      # note change always does
                if self._q_held is None:
                    pass                              # not settled yet at the start of a phrase, so the
                                                      # parts stay out
                elif abs(cont - self._q_held) > 2.5:
                    self._q_far += 1
                    if self._q_far >= 3 and self._qtgt is not None:
                        # Persistently far away means the singer really is
                        # moving a long way: pin the old target using the current
                        # cont and hold the note. Freezing instead would let the
                        # output follow them out of tune, which is the 165-cent
                        # case.
                        tune = (self.q_tune["off"]
                                if self.q_tune is not None else 0.0)
                        ns = self._qtgt + tune + self.pitch_offset - cont
                        if ns != self._cur_shift:
                            self.sb.set_pitch_shift_semitone(ns)
                            self._cur_shift = ns
                    # the first two hops may be outliers: keep the previous shift
                else:
                    self._q_far = 0
                    if self._q_raw_held != self._q_held:
                        # Recomputed only when held changes. auto_chord carries
                        # state, the chord memory and voice-lead's own previous
                        # note, so calling it every hop would waste work and
                        # could chatter at a boundary. The quantised held is the
                        # only driver of the chord state.
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
                        self._qtgt = tgt              # a new phrase lands directly, with no glide
                        self._q_age = 0
                    d = tgt - self._qtgt
                    if abs(d) > 0.3:
                        self._q_age = 0               # a note change restarts the vibrato ramp
                    else:
                        self._q_age += 1
                    if abs(d) > self.q_snap:
                        self._qtgt = tgt              # a leap or a correction lands directly
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
                            self._q_cs = cont             # dead zone: a large move is followed at once
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
