"""Live host for the streaming target-f0 WORLD mouth (file-1a live).

mic -> ear (YIN tracker + v2 brain, 0.1875 s ticks) -> per-tick ABSOLUTE note
    -> bounded WORLD resynthesis (world_stream recipe, ear-validated F17)
    -> speakers, LAG behind the voice (default 0.5 s: tick 187.5 + hop 150
       + compute ~80 + hold 40 ms stack worst-case -- 0.35 starved the ring,
       first live lesson; the angel is another person, DAF toxicity does not
       apply -- G23 is about delayed SELF).

Same-tick semantics as the approved stimuli: the note decided from tick k is
sung over tick k's frames, so the voice/angel relationship Harry blessed in
blind_stream_0724 is preserved exactly, just LAG later. The mouth interface
is (mic ring, per-tick note) -- a trained singing mouth (DDSP-SVC) can swap
in without touching the host.

Run live:    venv/bin/python world_live.py [--in-name X] [--out-name Y]
             [--lag 0.35] [--gain 1.5]
File check:  venv/bin/python world_live.py --file take.wav out.wav
             [--notes-json out/angel_v2_world2_notes.json]
(--notes-json bypasses the ear/brain with a replayed line = deterministic
mouth-equivalence check against world_stream.)
"""

import argparse
import copy
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pyworld

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from live import VoiceToTokens, token_to_midi  # noqa: E402
from pitch import SR, PitchTracker  # noqa: E402

FRAME_MS = 5.0
VIB_HZ, VIB_SEMI = 5.0, 0.12
PORTA = 0.15
HOP_FRAMES = 30
GUARD_FRAMES = 10
HOLD_FRAMES = 8
CTX_FRAMES = 240
TAIL_FRAMES = 80
REST_CLOSE = 40
BOUND_FRAMES = 200
FRAME_HOP = SR * FRAME_MS / 1000.0
TICK_SAMPS = int((60.0 / 80.0) * 0.25 * SR)  # 8268 @44.1k, the brain pulse


class Ear:
    """tick audio in -> mic-space absolute note (or None). world_mouth logic."""

    def __init__(self, accent="cpdl", legato_gap=1, reg_lo=9.0):
        from brain_v2 import BrainV2
        self.brain = BrainV2(accent=accent)
        self.v2t = VoiceToTokens()
        self.tracker = PitchTracker()
        self.prev = None
        self.legato = {"gap": 0, "note": None}
        self.legato_gap = legato_gap
        self.reg_lo = reg_lo

    def tick(self, chunk):
        self.tracker.push(chunk)
        s = self.v2t.token(self.tracker.latest)
        m = token_to_midi(self.brain.step(s), self.prev)
        self.prev = m
        if m is None and self.legato["note"] is not None and self.legato["gap"] < self.legato_gap:
            self.legato["gap"] += 1
            m = self.legato["note"]
        else:
            self.legato["gap"] = 0
            self.legato["note"] = m
        if m is None:
            return None
        n = m - (self.v2t.shift or 0)
        ref = float(np.median(self.tracker.register)) if self.tracker.register else 55.0
        while n < ref - self.reg_lo:
            n += 12
        while n > ref + 9:
            n -= 12
        return n


class ExpressiveF0:
    """The streaming form of the rule-based expression layer (F21, passed by ear
    on 2026-07-31; the parameters share their source with direct_mouth.EXPR and
    the measurements are cited in the comments there). Each frame, step() emits
    a deviation in cents: a per-note bias, a slow drift, a fast tremor, an
    irregular vibrato whose rate performs a random walk, and a scoop into the
    first note of a phrase. A fixed seed makes it deterministic, and a different
    seed per voice desynchronises the parts naturally, which cured the
    mechanically synchronised vibrato at one phase and rate found on 2026-07-28.
    Differences from the offline version: live porta applies to every note
    change, with no leap_snap, so the scoop is added only at the head of a phrase
    where cur == 0, and steps and leaps are left to porta without a second bend.
    The noise state keeps running through rests, so the drift is continuous
    across phrases, as offline."""

    BIAS_SD, BIAS_CLIP = 22.0, 35.0
    DRIFT_SD, DRIFT_FC = 13.0, 1.5
    JIT_SD, JIT_FC = 6.0, 10.0
    VIB_HZ0, VIB_WALK_SD, VIB_WALK_FC = 5.9, 0.9, 0.8
    VIB_DEPTH, VIB_DEPTH_SD = 20.0, 6.0
    VIB_DEPTH_LO, VIB_DEPTH_HI = 8.0, 32.0
    BEND_MAG, BEND_DOWN = 22.0, 0.59
    BEND_TAU_LO, BEND_TAU_HI = 0.015, 0.060

    def __init__(self, seed, gain=1.0):
        self.rng = np.random.default_rng(seed)
        self.gain = gain
        self.dt = FRAME_MS / 1000.0
        self._lp = {}          # name -> [one-pole coefficient, steady-state std for normalisation, state]
        for k, fc in (("drift", self.DRIFT_FC), ("jit", self.JIT_FC),
                      ("walk", self.VIB_WALK_FC)):
            a = 1.0 - np.exp(-2 * np.pi * fc * self.dt)
            self._lp[k] = [a, np.sqrt(a / (2.0 - a)), 0.0]
        self.phase = self.bias = self.bend = 0.0
        self.bend_a = 0.0
        self.depth = self.depth_t = self.VIB_DEPTH
        self.a_dep = 1.0 - np.exp(-self.dt / 0.03)   # 30 ms smoothing of the note-change depth

    def note_on(self, snapped):
        """A new note instance. snapped means the skeleton landed instantly, at
        the head of a phrase, which calls for a scoop."""
        r = self.rng
        self.bias = float(np.clip(r.normal(0.0, self.BIAS_SD),
                                  -self.BIAS_CLIP, self.BIAS_CLIP))
        self.depth_t = float(np.clip(r.normal(self.VIB_DEPTH, self.VIB_DEPTH_SD),
                                     self.VIB_DEPTH_LO, self.VIB_DEPTH_HI))
        if snapped:
            sgn = -1.0 if r.random() < self.BEND_DOWN else 1.0
            self.bend = sgn * min(60.0, r.lognormal(np.log(self.BEND_MAG), 0.5))
            self.bend_a = np.exp(-self.dt / r.uniform(self.BEND_TAU_LO,
                                                      self.BEND_TAU_HI))

    def step(self):
        vals = {}
        for k, s in self._lp.items():
            s[2] += s[0] * (self.rng.standard_normal() - s[2])
            vals[k] = s[2] / s[1]
        self.phase += (self.VIB_HZ0 + self.VIB_WALK_SD * vals["walk"]) * self.dt
        self.depth += (self.depth_t - self.depth) * self.a_dep
        self.bend *= self.bend_a
        return self.gain * (self.bias + self.DRIFT_SD * vals["drift"]
                            + self.JIT_SD * vals["jit"]
                            + self.depth * np.sin(2 * np.pi * self.phase)
                            + self.bend)


def pv_crossfade(a, b, fade_in):
    """Phase-vocoder crossfade, a line-by-line numpy port of upstream
    gui.phase_vocoder, from the pulsing case of 2026-07-31. A linear sin^2/cos^2
    crossfade between two segments whose phases are not fully aligned, from the
    SOLA residual and combsub restarting its phase every window, loses energy at
    the centre of the join. That happens once per hop, giving a 6.7 Hz amplitude
    pumping, which is the pulsing that was heard: measured, the streaming version
    had 5-9 Hz AM at 6.4-7.4 against 1.8 offline.
    The phase vocoder reconstructs the shared spectrum of the two segments with
    interpolated phase as a cross term, which puts the energy back."""
    n = len(a)
    fade_out = 1.0 - fade_in
    window = np.sqrt(fade_out * fade_in)
    fa = np.fft.rfft(a * window)
    fb = np.fft.rfft(b * window)
    absab = np.abs(fa) + np.abs(fb)
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = np.angle(fa)
    dphi = np.angle(fb) - phia
    dphi = dphi - 2 * np.pi * np.floor(dphi / (2 * np.pi) + 0.5)
    w = 2 * np.pi * np.arange(n // 2 + 1) + dphi
    t = np.arange(n)[:, None] / n
    return (a * fade_out ** 2 + b * fade_in ** 2
            + (absab * np.cos(w * t + phia)).sum(-1) * window / n)


class Mouth:
    """shared machinery: target-f0 build, gate, phrase state, splice-emit."""

    def __init__(self, out_ring, expr=None):
        self.out = out_ring
        self.expr = expr   # ExpressiveF0 or None; None is the fixed-vibrato behaviour from before 2026-07-31
        self.prev_note = None
        nf = int(len(out_ring) / FRAME_HOP) + 1
        self.accepted = 0
        self.phrase = None
        self.cur = 0.0     # porta state
        self.env_g = 0.0   # gate state
        self.f0_t = np.zeros(nf)
        self.env = np.zeros(nf)
        self.built = 0     # target frames built so far
        self.a_up = 1 - np.exp(-FRAME_MS / 40.0)
        self.a_dn = 1 - np.exp(-FRAME_MS / 150.0)
        self.xf = int(0.030 * SR)
        self.hold = int(round(HOLD_FRAMES * FRAME_HOP))
        self.frontier = 0
        self.t_hop = []

    def _build_target(self, upto_f, notes):
        for k in range(self.built, upto_f):
            tick = min(int(k * FRAME_HOP / TICK_SAMPS), len(notes) - 1)
            note = notes[tick] if tick < len(notes) else None
            tk = k * FRAME_MS / 1000.0
            target = 0.0 if note is None else 440 * 2 ** ((note - 69) / 12)
            if target > 0:
                if self.expr is not None:
                    if note != self.prev_note:
                        self.expr.note_on(snapped=self.cur == 0)
                    self.cur = target if self.cur == 0 \
                        else self.cur + (target - self.cur) * PORTA
                    self.f0_t[k] = self.cur * 2 ** (self.expr.step() / 1200.0)
                else:
                    self.cur = target if self.cur == 0 else self.cur + (target - self.cur) * PORTA
                    self.f0_t[k] = self.cur * 2 ** (VIB_SEMI * np.sin(2 * np.pi * VIB_HZ * tk) / 12)
            else:
                self.cur = 0.0
                self.f0_t[k] = 0.0
                if self.expr is not None:
                    self.expr.step()   # the noise state keeps running through rests, so the drift is continuous across phrases
            self.prev_note = note
            g = self.f0_t[k] > 0
            self.env_g += (float(g) - self.env_g) * (self.a_up if g > self.env_g else self.a_dn)
            self.env[k] = self.env_g
        self.built = max(self.built, upto_f)

    def _gated(self, wav, lo_f, hi_f):
        gate = np.interp(np.arange(len(wav)) / SR,
                         np.arange(hi_f - lo_f) * FRAME_MS / 1000.0, self.env[lo_f:hi_f])
        return wav * gate

    def advance_phrase(self, new_lo, can_open=True):
        for gf in range(new_lo, self.accepted):
            resting = self.f0_t[gf] == 0
            if self.phrase is None:
                if resting or not can_open:
                    continue
                self.phrase = {"start": gf, "len": 0, "rest": 0, "n": 0}
            self.phrase["len"] += 1
            self.phrase["rest"] = self.phrase["rest"] + 1 if resting else 0
        return self.phrase

    def splice_emit(self, wav_b, s0, s0b, hold):
        """crossfade-splice wav_b (starting at abs sample s0b) into the ring;
        phrase['n'] tracks emitted samples relative to phrase start s0."""
        rel_prev = self.phrase["n"] - (s0b - s0)
        n_emit = max(rel_prev, len(wav_b) - hold)
        if rel_prev > 0:
            fl = min(self.xf, rel_prev, len(wav_b))
            fade = np.sin(0.5 * np.pi * np.linspace(0, 1, fl)) ** 2
            lo = rel_prev - fl
            self.out[s0b + lo : s0b + rel_prev] = pv_crossfade(
                self.out[s0b + lo : s0b + rel_prev], wav_b[lo:rel_prev], fade)
        self.out[s0b + max(rel_prev, 0) : s0b + n_emit] = wav_b[max(rel_prev, 0):n_emit]
        self.frontier = max(self.frontier, s0b + n_emit)
        self.phrase["n"] = (s0b - s0) + n_emit


class WorldMouth(Mouth):
    """world_stream bounded mode, incremental (F17-validated)."""

    def __init__(self, out_ring, expr=None):
        super().__init__(out_ring, expr)
        fft = pyworld.get_cheaptrick_fft_size(SR)
        dim = fft // 2 + 1
        nf = len(self.f0_t)
        self.borrow_sp = np.zeros((nf, dim))
        self.borrow_ap = np.ones((nf, dim))
        self.have_good = False
        self.last_sp = self.last_ap = None

    def hop(self, mic, t_end, notes, n_ticks, closing=False):
        """process audio up to input sample t_end; notes decided for n_ticks."""
        t0 = time.perf_counter()
        w0 = max(0, t_end - int(round(CTX_FRAMES * FRAME_HOP)))
        win = np.ascontiguousarray(mic[w0:t_end], dtype=np.float64)
        if len(win) < 1024:
            return
        base_f = int(round(w0 / FRAME_HOP))
        f0w, tw = pyworld.dio(win, SR, frame_period=FRAME_MS)
        f0w = pyworld.stonemask(win, f0w, tw, SR)
        tl0 = max(0, len(win) - int(round(TAIL_FRAMES * FRAME_HOP)))
        tail = win[tl0:]
        tf0_lo = int(round(tl0 / FRAME_HOP))
        f0t = f0w[tf0_lo : tf0_lo + int(len(tail) / FRAME_HOP) + 1]
        tt = np.arange(len(f0t)) * FRAME_MS / 1000.0
        spt = pyworld.cheaptrick(tail, f0t, tt, SR)
        apt = pyworld.d4c(tail, f0t, tt, SR)
        tail_base_f = base_f + tf0_lo

        # frames usable once guard-settled AND their tick's note is decided
        lim = t_end if closing else min(
            int(t_end - GUARD_FRAMES * FRAME_HOP), n_ticks * TICK_SAMPS)
        lim_f = int(lim / FRAME_HOP)
        self._build_target(min(lim_f, len(self.f0_t)), notes)
        midi_w = np.where(f0w > 0, 69 + 12 * np.log2(np.maximum(f0w, 1) / 440), np.nan)
        new_lo = self.accepted
        for gf in range(self.accepted, min(lim_f, tail_base_f + len(f0t), len(self.f0_t))):
            lt, lf = gf - tail_base_f, gf - base_f
            if lt < 0:
                continue
            voiced = f0w[lf] > 0
            wm = midi_w[max(0, lf - 4) : lf + 5]
            med = np.nan if np.isnan(wm).all() else np.nanmedian(wm)
            oct_err = voiced and not np.isnan(med) and abs(midi_w[lf] - med) > 6
            if voiced and apt[lt].mean() < 0.8 and not oct_err:
                self.last_sp, self.last_ap, self.have_good = spt[lt], apt[lt], True
            if self.have_good:
                self.borrow_sp[gf], self.borrow_ap[gf] = self.last_sp, self.last_ap
            self.accepted = gf + 1
        if self.accepted == new_lo:
            self.t_hop.append(time.perf_counter() - t0)
            return

        if self.advance_phrase(new_lo, can_open=self.have_good) is None:
            self.t_hop.append(time.perf_counter() - t0)
            return
        closing_p = self.phrase["rest"] >= REST_CLOSE or closing
        hold = 0 if closing_p else self.hold
        s = self.phrase["start"]
        e = s + self.phrase["len"]
        b = max(0, self.phrase["len"] - BOUND_FRAMES)
        wav_b = pyworld.synthesize(
            np.ascontiguousarray(self.f0_t[s + b : e]),
            np.ascontiguousarray(self.borrow_sp[s + b : e]),
            np.ascontiguousarray(self.borrow_ap[s + b : e]), SR, frame_period=FRAME_MS)
        wav_b = self._gated(wav_b, s + b, e)
        self.splice_emit(wav_b, int(round(s * FRAME_HOP)),
                         int(round((s + b) * FRAME_HOP)), hold)
        if closing_p:
            self.phrase = None
        self.t_hop.append(time.perf_counter() - t0)


class DDSPMouth(Mouth):
    """DDSP-SVC combsub as the singing mouth (F18 blind winner).

    content = hubert units from HIS live audio (vowels follow the mouth),
    f0 = the brain's target curve (target-style by construction),
    volume = his mic energy gated by the target envelope (angel breathes
    with him; silent when he is silent -- unlike WorldMouth's borrow-hold).
    Same bounded window + splice emission as WorldMouth (F17 machinery).
    Needs the DDSP venv (torch, transformers); see --mouth ddsp in main.
    """

    BLOCK = 512  # model hop @44.1k
    BOUNDF = 140  # 0.7 s resynth window (vs WORLD's 1.0 s): trims the MPS
                  # hubert+model cost spikes that starved the ring live
    UV_W, UV_TAU_LO, UV_TAU_HI = 2048, 55, 551  # autocorrelation window, and the lag band for 80-800 Hz
    UV_HI, UV_LO, UV_ENTER = 0.60, 0.35, 8      # Schmitt hysteresis; 8 frames, 40 ms, to enter uv
    # v2 anti-chatter, after the pulsing was pinned on the mask chattering late
    # on 2026-07-31 and the --uv-gate 0 A/B confirmed it:
    UV_EXIT = 3        # leaving uv also needs 3 consecutive strongly periodic frames; the
                       # single frame of v1 was half the cause of the fast switching
    UV_DWELL = 40      # minimum dwell of 200 ms after a change, which caps the chatter rate
    UV_WEAK = 0.5      # in the border band (LO_DEEP to LO), entering uv needs
                       # rms below 0.5 of the recent voiced rms. Decisively
                       # aperiodic frames, below LO_DEEP, which is the signature
                       # of a breath, are not level-limited, so a loud breath
                       # still closes the gate while a soft quavering note is not
                       # closed by mistake
    UV_LO_DEEP = 0.25  # threshold for decisively aperiodic
    UV_FLOOR = 0.05    # soft mask: uv is pushed down 26 dB rather than hard
                       # zeroed, over a 25 ms ramp, which turns the residual
                       # switching from a click into a dip. Switching in the
                       # active region fell from 1.86 to 1.14 per second. At
                       # 0.15 the breath_v% rebounded to 40.9, too loose; 0.05
                       # measured it back down

    def __init__(self, out_ring, repo, model_path, device="mps", expr=None,
                 uv_gate=False, free_run=False, bound_f=None):
        super().__init__(out_ring, expr)
        if bound_f:            # re-render window in frames: the hop cost is proportional to the
                               # window length, so a shorter window means a shallower rewind
            self.BOUNDF = int(bound_f)
        self.uv_gate = uv_gate
        self.vuv = np.ones(len(self.f0_t))  # per-frame sounding mask (1 = voiced)
        self.vuv_done = 0
        self.vuv_state, self.vuv_run = True, 0
        self.vuv_xrun, self.vuv_dwell, self.vuv_vr = 0, 999, 0.05
        self.free_run = free_run
        self.t_enc, self.t_fwd = [], []   # per-stage timing inside a hop: hubert and model
        self.emitted = 0   # the emission front, in absolute samples, monotonic apart from a rewind on divergence
        self.eff = []      # tick -> the note actually used when built, including the hold assumption; the single stream of truth
        self.snap = {}     # tick -> the complete state before the first frame of that tick, the rewind point
        self.checked = 0   # ticks whose assumption has been reconciled
        import torch
        sys.path.insert(0, str(repo))
        from ddsp.vocoder import Units_Encoder, Volume_Extractor, load_model
        self.torch = torch
        self.device = device
        self.model, margs = load_model(str(model_path), device=device)
        enc_ckpt = Path(repo) / margs.data.encoder_ckpt
        self.encoder = Units_Encoder(
            margs.data.encoder, str(enc_ckpt), margs.data.encoder_sample_rate,
            margs.data.encoder_hop_size, device=device)
        self.vol_ex = Volume_Extractor(self.BLOCK)
        self.spk = torch.LongTensor([[1]]).to(device)
        assert int(margs.data.block_size) == self.BLOCK
        assert int(margs.data.sampling_rate) == SR

    def _update_vuv(self, mic, upto_f):
        """Per-frame sounding detection on the singer's microphone, from the
        "breaths get sung" case of 2026-07-31: with the same note line and the
        same model, the live voice measured breath_v% 54.1 against 13.9 offline.
        Volume is microphone energy times the note envelope, a breath has energy,
        and the model's f0 always carries a pitch, so breaths and consonants were
        rendered as pitched notes. The voicing gate that fixed this offline in
        (a) had never been ported; the live version only needs silence and not
        pass-through, because the singer's real breath is already in the air.

        Autocorrelation peak periodicity plus an RMS threshold (0.005, the same
        terms as pitch.RMS_GATE) plus causal hysteresis: uv is entered only after
        UV_ENTER consecutive aperiodic frames, and the marking is applied
        retroactively. The bounded resynthesis window writes that correction back
        into the ring, and since the lag of 0.5-0.6 s exceeds the hop rate of
        150 ms, it is corrected before it is heard. A single strongly periodic
        frame returns to voiced, giving priority to not opening a hole inside a
        phrase, the same asymmetric cost principle as the offline anchoring.
        On seg_take it agrees with the offline voicing_mask 95% of the time, and
        the threshold surface is flat: HI and LO moved by 0.05 change nothing.
        The mask multiplies the model's output, after the SOLA inside the hop,
        and does not touch the vol condition; the reason is in the hop
        comments."""
        lo, hi = self.vuv_done, min(upto_f, len(self.vuv))
        if hi <= lo:
            return
        idx = ((np.arange(lo, hi)[:, None] * FRAME_HOP).astype(int)
               + np.arange(self.UV_W))
        segs = mic[np.minimum(idx, len(mic) - 1)]
        segs = segs - segs.mean(axis=1, keepdims=True)
        rms = np.sqrt((segs ** 2).mean(axis=1))
        F = np.fft.rfft(segs, n=2 * self.UV_W, axis=1)
        r = np.fft.irfft((F * np.conj(F)).real, axis=1)[:, :self.UV_W]
        peak = (r[:, self.UV_TAU_LO:self.UV_TAU_HI].max(axis=1)
                / np.maximum(r[:, 0], 1e-12))
        for i, k in enumerate(range(lo, hi)):
            hard = rms[i] < 0.005
            if peak[i] > self.UV_HI and not hard:      # track the recent voiced level
                self.vuv_vr += (1 - np.exp(-1 / 100.0)) * (rms[i] - self.vuv_vr)
            self.vuv_dwell += 1
            if self.vuv_state:
                if hard or peak[i] < self.UV_LO_DEEP or (
                        peak[i] < self.UV_LO
                        and rms[i] < self.UV_WEAK * self.vuv_vr):
                    self.vuv_run += 1
                    if (self.vuv_run >= self.UV_ENTER
                            and self.vuv_dwell >= self.UV_DWELL):
                        self.vuv_state, self.vuv_run, self.vuv_dwell = False, 0, 0
                        self.vuv[max(0, k - self.UV_ENTER + 1):k] = 0.0
                else:
                    self.vuv_run = 0
            else:
                if not hard and peak[i] > self.UV_HI:
                    self.vuv_xrun += 1
                    if (self.vuv_xrun >= self.UV_EXIT
                            and self.vuv_dwell >= self.UV_DWELL):
                        self.vuv_state, self.vuv_xrun, self.vuv_dwell = True, 0, 0
                else:
                    self.vuv_xrun = 0
            self.vuv[k] = 1.0 if self.vuv_state else 0.0
        self.vuv_done = hi

    def _uv_gain(self, pos):
        """vuv to an output gain curve: a Hann smoothing in the frame domain,
        about a 25 ms ramp, over the soft floor UV_FLOOR."""
        lo = max(0, int(pos[0]) - 8)
        hi = min(len(self.vuv), int(pos[-1]) + 8)
        kern = np.hanning(7)
        sm = np.convolve(self.vuv[lo:hi], kern / kern.sum(), mode="same")
        return (self.UV_FLOOR + (1.0 - self.UV_FLOOR)
                * np.interp(pos, np.arange(lo, hi), sm))

    def _build_hold(self, upto_f, notes, n_ticks):
        """Single-stream construction with a snapshot at each tick boundary
        (decoupling v2, 2026-07-31). v1 had a settled region and a provisional
        one, with the provisional region restarted from a snapshot every hop, and
        listening to a sustained note exposed the structural error: on a
        sustained note the pitch does not change and the whole impression rests
        on the expression texture, while each emitted block of about 150 ms
        carried its own restarted vibrato and drift phase, which sounded like the
        same note being played over and over.
        In v2 the state (expression, porta, envelope, phrase) advances once per
        frame, so the texture is continuous by construction. An undecided tick
        uses the last decided note, a hold, and records it in eff. When the new
        tick is settled it is reconciled, and only if the assumption is overturned
        does it rewind to that tick's snapshot, rebuild, and rewind the emission
        front to re-lay the audio, which lands the bend. When the assumption
        holds, there is zero rework."""
        lim_t = min(n_ticks, len(self.eff))
        for t in range(self.checked, lim_t):
            if self.eff[t] != notes[t]:
                self.emitted = min(self.emitted, t * TICK_SAMPS)
                (self.expr, self.cur, self.prev_note, self.env_g,
                 self.accepted, self.phrase, self.built) = self.snap[t]
                del self.eff[t:]
                for k in [k for k in self.snap if k >= t]:
                    del self.snap[k]
                break
        self.checked = n_ticks
        last = notes[-1] if notes else None
        while self.built < upto_f:
            t = int(self.built * FRAME_HOP / TICK_SAMPS)
            if t == len(self.eff):
                self.snap[t] = (copy.deepcopy(self.expr), self.cur,
                                self.prev_note, self.env_g, self.accepted,
                                copy.deepcopy(self.phrase), self.built)
                self.eff.append(notes[t] if t < n_ticks else last)
                self.snap.pop(t - 16, None)   # the decision lag is at most a few ticks; older snapshots are dropped
            end_f = min(upto_f, int(np.ceil((t + 1) * TICK_SAMPS / FRAME_HOP)))
            new_lo = self.accepted
            self._build_target(end_f, self.eff)
            self.accepted = max(self.accepted, self.built)
            self.advance_phrase(new_lo)

    def _sola(self, wav_b, s0b, rel_prev, R=256, W=1024):
        """best sample offset (+-R) aligning wav_b's overlap to the ring"""
        if rel_prev < R + 128 or len(wav_b) < rel_prev + R:
            return 0
        W = min(W, rel_prev - R)
        ref = self.out[s0b + rel_prev - W : s0b + rel_prev]
        best_d, best_c = 0, -1e18
        for d in range(-R, R + 1, 4):
            seg = wav_b[rel_prev - W + d : rel_prev + d]
            c = float(np.dot(ref, seg)) / (np.sqrt(float(np.dot(seg, seg))) + 1e-9)
            if c > best_c:
                best_c, best_d = c, d
        return best_d

    def hop(self, mic, t_end, notes, n_ticks, closing=False, shared=None):
        t0 = time.perf_counter()
        lim = t_end if closing else int(t_end - GUARD_FRAMES * FRAME_HOP)
        if not (self.free_run or closing):
            lim = min(lim, n_ticks * TICK_SAMPS)
        lim_f = int(lim / FRAME_HOP)
        if self.uv_gate:
            self._update_vuv(mic, min(lim_f, len(self.vuv)))
        if self.free_run:
            # Decoupling (2026-07-31): the voice does not wait for the model.
            # The render front is no longer hard-locked by the note decision, so
            # a note change becomes a suspension-like residue of about 200 ms
            # followed by a bend in, rather than a 0.5 s wait for everything. The
            # mechanism is in _build_hold, the v2 single-stream form. Known cost:
            # entering a phrase still waits for the model, since a held rest does
            # not open a phrase, so the bend only rescues note changes inside a
            # phrase.
            self._build_hold(min(lim_f, len(self.f0_t)), notes, n_ticks)
            if self.phrase is None:
                self.t_hop.append(time.perf_counter() - t0)
                return
            s0p = int(round(self.phrase["start"] * FRAME_HOP))
            self.phrase["n"] = max(self.phrase["n"], self.emitted - s0p)
        else:
            self._build_target(min(lim_f, len(self.f0_t)), notes)
            new_lo = self.accepted
            self.accepted = max(new_lo, min(lim_f, len(self.f0_t)))
            if self.accepted == new_lo:
                self.t_hop.append(time.perf_counter() - t0)
                return
            if self.advance_phrase(new_lo) is None:
                self.t_hop.append(time.perf_counter() - t0)
                return
        closing_p = self.phrase["rest"] >= REST_CLOSE or closing
        hold = 0 if closing_p else self.hold
        s = self.phrase["start"]
        e = s + self.phrase["len"]
        b = max(0, self.phrase["len"] - self.BOUNDF)
        s0 = int(round(s * FRAME_HOP))
        w_lo = (int(round((s + b) * FRAME_HOP)) // self.BLOCK) * self.BLOCK
        # The window start is aligned to the absolute 512 grid, so the units of
        # the two voices can be sliced and shared; see below.
        n_blocks = (int(round(e * FRAME_HOP)) - w_lo) // self.BLOCK
        if n_blocks < 2:
            self.t_hop.append(time.perf_counter() - t0)
            return
        w_hi = w_lo + n_blocks * self.BLOCK
        audio = np.ascontiguousarray(mic[w_lo:w_hi], dtype=np.float32)
        torch = self.torch
        with torch.no_grad():
            t1 = time.perf_counter()
            # Shared units (2026-07-31 section M): the content of both voices
            # comes from the same span of the singer's microphone through the
            # same HuBERT encoder, so it is computed once per hop cycle. The
            # encoder is two thirds of a voice's cost, measured at 16 ms against
            # 8 ms for the forward pass. The window is already aligned to the 512
            # grid, so the second voice slices it directly; when the window is
            # not contained in the cache, which happens at the start of a phrase
            # where the lengths differ, it computes its own and updates the
            # cache.
            if (shared is not None and shared.get("units") is not None
                    and shared["lo"] <= w_lo
                    and w_lo + n_blocks * self.BLOCK <= shared["hi"]):
                b0 = (w_lo - shared["lo"]) // self.BLOCK
                units = shared["units"][:, b0:b0 + n_blocks]
            else:
                audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                units = self.encoder.encode(audio_t, SR, self.BLOCK)
                if shared is not None:
                    shared.update(lo=w_lo, hi=w_hi, units=units)
            # Reading a value is the synchronisation boundary for timing.
            # torch.mps.synchronize collides with the MPS encode on the tick
            # thread and raises a Metal assertion, which happened in practice;
            # synchronising by reading a value, through .cpu() or .item(), has
            # been verified to coexist with the existing concurrency.
            _ = float(units[0, -1, 0])
            self.t_enc.append(time.perf_counter() - t1)
            # target f0 at block centres; rests forward-filled (uv f0=0 is
            # out-of-distribution for the model; volume mutes them anyway)
            centres = (w_lo + (np.arange(n_blocks) + 0.5) * self.BLOCK) / FRAME_HOP
            f0 = np.interp(centres, np.arange(len(self.f0_t)), self.f0_t)
            nz = f0 > 0
            if nz.any():
                idx = np.maximum.accumulate(np.where(nz, np.arange(len(f0)), 0))
                f0 = np.where(nz, f0, f0[idx])
                f0[f0 <= 0] = f0[nz][0]
            else:
                f0[:] = 200.0
            vol = self.vol_ex.extract(audio.astype(np.float64))[:n_blocks]
            env_b = np.interp(centres, np.arange(len(self.env)), self.env)
            vol = vol * env_b
            f0_t = torch.from_numpy(f0).float().to(self.device).unsqueeze(0).unsqueeze(-1)
            vol_t = torch.from_numpy(vol).float().to(self.device).unsqueeze(0).unsqueeze(-1)
            n = min(units.size(1), n_blocks)
            t2 = time.perf_counter()
            out, _, _ = self.model(units[:, :n], f0_t[:, :n], vol_t[:, :n],
                                   spk_id=self.spk)
            wav_b = out.squeeze(0).cpu().numpy().astype(np.float64)
            self.t_fwd.append(time.perf_counter() - t2)
        # SOLA: each run's harmonic phase restarts at its window start, so a
        # fixed-position crossfade against the ring warbles every 150 ms
        # ("pitch sliding around", 07-24 live). Align to the ring first.
        rel_prev = self.phrase["n"] - (w_lo - s0)
        d = self._sola(wav_b, w_lo, rel_prev)
        if d > 0:
            wav_b = wav_b[d:]
        elif d < 0:
            wav_b = np.concatenate([np.full(-d, wav_b[0]), wav_b[:d]])
        if self.uv_gate:
            # Output mask: vol is only a conditioning input to unit2ctrl, whose
            # output is a comb tooth through a predicted filter plus noise
            # through a filter, so zeroing it does not silence the model; the
            # first version went through vol and measured only an 8% drop in
            # energy over uv spans. The mask is applied after the SOLA, so the
            # overlap used for alignment is not punched through, and np.interp
            # over the 0/1 frame grid gives a 5 ms linear ramp.
            pos = (w_lo + np.arange(len(wav_b))) / FRAME_HOP
            wav_b = wav_b * self._uv_gain(pos)
        self.splice_emit(wav_b, s0, w_lo, hold)
        if self.free_run:
            self.emitted = max(self.emitted, s0 + self.phrase["n"])
        if closing_p:
            self.phrase = None
        self.t_hop.append(time.perf_counter() - t0)


class StreamMouth(DDSPMouth):
    """A genuinely streaming DDSP voice, written late on 2026-07-31 after the
    question "why is solo_min both immediate and smooth?"

    solo_min is smooth because Beatrice is an assembly line: it synthesises once
    per sample, its state is continuous across frames, and there are no joins.
    DDSPMouth is a batch renderer forced into a live setting: every hop it
    recomputes a whole 0.45-0.7 s window, does a SOLA and crossfades the tail on,
    which is 6.7 joins per second. The pulsing is the energy dip at those joins,
    and the fragmentation is joins landing late under load.

    This class opens combsub into an assembly line. The initial_phase entry of
    forward (vocoder.py:655) carries the exciter phase across blocks, so it
    synthesises once per sample with no SOLA and no re-rendering, and blocks are
    joined by a 256-sample linear fade; with the phase continuous the signals are
    correlated, so a linear fade loses no energy. The phase is advanced by
    recomputing core.upsample plus a float32 cumsum, bit-identical to the model.
    Bends are free: frames are built at the last possible moment before
    rendering, and a decision that arrives late is slid in by porta from the
    current pitch, which is the vocabulary of a suspension. None of the
    free-running eff, snapshot and rewind machinery is needed. Rests skip the
    model entirely, advancing the phase analytically and outputting silence, so
    the MPS cost is proportional to new sounding audio."""

    E = 2        # extra blocks rendered at the left edge, eaten by the istft window and used only for the fade
    CTXB = 28    # HuBERT left-context blocks (0.325 s, for unit quality)
    CHUNK_MAX = 43   # ceiling on one hop's rendering (0.5 s), protecting against a catch-up burst

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        sys.path.insert(0, str(kw.get("repo") or args[1]))
        from ddsp.core import upsample as _up
        self._up = _up
        self.next_b = None   # the next absolute block to emit
        self.phase = 0.0     # exciter phase in radians, at the start of the rendered chunk

    def hop(self, mic, t_end, notes, n_ticks, closing=False, shared=None):
        t0 = time.perf_counter()
        lim = t_end if closing else int(t_end - GUARD_FRAMES * FRAME_HOP)
        lim_f = int(lim / FRAME_HOP)
        if self.uv_gate:
            self._update_vuv(mic, min(lim_f, len(self.vuv)))
        self._build_target(min(lim_f, len(self.f0_t)), notes)  # built at the last moment
        lim_b = lim // self.BLOCK
        if self.next_b is None:
            self.next_b = max(self.E, lim_b - 1)
        b0 = self.next_b
        n_new = min(lim_b - b0, self.CHUNK_MAX)
        if n_new <= 0:
            self.t_hop.append(time.perf_counter() - t0)
            return
        b1, r0 = b0 + n_new, b0 - self.E
        r1 = min(b1 + self.E, len(mic) // self.BLOCK)  # blocks rendered at the right edge, for the istft tail window
        # Without a right neighbour, the overlap-add carves an amplitude dip out
        # of the last block of every chunk, which was the source of the residual
        # AM in the first test. Render E extra blocks and discard them; the
        # emitted span is unchanged.
        centres = ((np.arange(r0, r1) + 0.5) * self.BLOCK) / FRAME_HOP
        f0 = np.interp(centres, np.arange(len(self.f0_t)), self.f0_t)
        nz = f0 > 0
        if nz.any():
            idx = np.maximum.accumulate(np.where(nz, np.arange(len(f0)), 0))
            f0 = np.where(nz, f0, f0[idx])
            f0[f0 <= 0] = f0[nz][0]
        else:
            f0[:] = 200.0
        # Output mask curve (env times vuv). The vol condition cannot silence
        # the model, which is the lesson of the breath case of 2026-07-31.
        pos = (r0 * self.BLOCK + np.arange((r1 - r0) * self.BLOCK)) / FRAME_HOP
        g = np.interp(pos, np.arange(len(self.env)), self.env)
        if self.uv_gate:
            g = g * self._uv_gain(pos)
        torch = self.torch
        if g.max() < 1e-3:
            # Rest: skip the model and advance the phase analytically.
            # Continuity is inaudible in silence, so float64 is enough.
            self.phase = (self.phase + 2 * np.pi * float(f0[: n_new].sum())
                          * self.BLOCK / SR) % (2 * np.pi)
            self.out[b0 * self.BLOCK: b1 * self.BLOCK] = 0.0
            self.frontier = max(self.frontier, b1 * self.BLOCK)
            self.next_b = b1
            self.t_hop.append(time.perf_counter() - t0)
            return
        a_lo = r0 * self.BLOCK
        audio = np.ascontiguousarray(mic[a_lo: r1 * self.BLOCK],
                                     dtype=np.float32)
        with torch.no_grad():
            t1 = time.perf_counter()
            e_lo = max(0, r0 - self.CTXB) * self.BLOCK
            e_hi = min(len(mic), (r1 + 2) * self.BLOCK,
                       (t_end // self.BLOCK) * self.BLOCK)
            if (shared is not None and shared.get("units") is not None
                    and shared["lo"] <= a_lo
                    and r1 * self.BLOCK <= shared["hi"]):
                bb = (a_lo - shared["lo"]) // self.BLOCK
                units = shared["units"][:, bb: bb + (r1 - r0)]
            else:
                ctx = torch.from_numpy(np.ascontiguousarray(
                    mic[e_lo:e_hi], dtype=np.float32)).unsqueeze(0).to(self.device)
                units_all = self.encoder.encode(ctx, SR, self.BLOCK)
                if shared is not None:
                    shared.update(lo=e_lo, hi=e_hi, units=units_all)
                bb = (a_lo - e_lo) // self.BLOCK
                units = units_all[:, bb: bb + (r1 - r0)]
            _ = float(units[0, -1, 0])
            self.t_enc.append(time.perf_counter() - t1)
            t2 = time.perf_counter()
            n = min(units.size(1), r1 - r0)
            f0_t = torch.from_numpy(f0[:n]).float().to(self.device)[None, :, None]
            vol = self.vol_ex.extract(audio.astype(np.float64))[:n]
            env_b = np.interp(centres[:n], np.arange(len(self.env)), self.env)
            vol_t = torch.from_numpy(vol * env_b).float().to(self.device)[None, :, None]
            out, _, _ = self.model(units[:, :n], f0_t, vol_t, spk_id=self.spk,
                                   initial_phase=torch.tensor(float(self.phase)))
            wav = out.squeeze(0).cpu().numpy().astype(np.float64)
            # The encoder occasionally returns one block fewer than requested,
            # at a resampling boundary, so the emission end point converges to
            # what was actually produced.
            b1a = min(b1, r0 + n)
            if b1a <= b0:
                self.t_hop.append(time.perf_counter() - t0)
                return
            # Advance the phase by the same formula as the model, core.upsample
            # plus a float32 cumsum, taking the accumulated phase at the start of
            # the next chunk's rendering, b1a - E.
            f0_up = self._up(f0_t.float().cpu(), self.BLOCK)
            x = torch.cumsum(f0_up.float() / SR, 1)
            k = (b1a - self.E - r0) * self.BLOCK - 1
            self.phase = (self.phase
                          + 2 * np.pi * float(x[0, max(0, k), 0])) % (2 * np.pi)
            self.t_fwd.append(time.perf_counter() - t2)
        ne = (b1a - r0) * self.BLOCK
        seg = wav[self.E * self.BLOCK: ne] * g[self.E * self.BLOCK: ne]
        xfl = min(256, b0 * self.BLOCK)
        if xfl:
            head = (wav[self.E * self.BLOCK - xfl: self.E * self.BLOCK]
                    * g[self.E * self.BLOCK - xfl: self.E * self.BLOCK])
            w = np.linspace(0.0, 1.0, xfl)
            s = b0 * self.BLOCK
            self.out[s - xfl: s] = self.out[s - xfl: s] * (1 - w) + head * w
        self.out[b0 * self.BLOCK: b1a * self.BLOCK] = seg
        self.frontier = max(self.frontier, b1a * self.BLOCK)
        self.next_b = b1a
        self.t_hop.append(time.perf_counter() - t0)


def run(a):
    cap = int((a.minutes * 60 + 5) * SR)
    mic_ring = np.zeros(cap, dtype=np.float64)
    out_ring = np.zeros(cap, dtype=np.float64)
    if a.mouth == "ddsp":
        mouth = DDSPMouth(out_ring, a.ddsp_repo, a.ddsp_model)
    else:
        mouth = WorldMouth(out_ring)
    notes = []
    if a.notes_json:
        meta = json.load(open(a.notes_json))
        notes = list(meta["notes"])
        ear = None
    else:
        ear = Ear(accent=a.accent)
    st = {"in": 0, "ticks": len(notes), "hop_next": int(round(HOP_FRAMES * FRAME_HOP)),
          "under": 0, "done": False}
    lag = int(a.lag * SR)
    lock = threading.Lock()

    def worker():
        hop_samps = int(round(HOP_FRAMES * FRAME_HOP))
        while not st["done"]:
            with lock:
                n_in = st["in"]
            while ear is not None and (st["ticks"] + 1) * TICK_SAMPS <= n_in:
                k = st["ticks"]
                note = ear.tick(mic_ring[k * TICK_SAMPS : (k + 1) * TICK_SAMPS])
                notes.append(note)
                st["ticks"] += 1
            if n_in >= st["hop_next"]:
                t_end = st["hop_next"]
                mouth.hop(mic_ring, t_end, notes if notes else [None],
                          st["ticks"], closing=False)
                st["hop_next"] = t_end + hop_samps
            else:
                time.sleep(0.005)

    if a.file:
        # headless: pump the file through the same worker path
        with wave.open(a.file[0]) as w:
            nch = w.getnchannels()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
        mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)
        mic_ring[: len(mic)] = mic
        if ear is not None:
            while (st["ticks"] + 1) * TICK_SAMPS <= len(mic):
                k = st["ticks"]
                notes.append(ear.tick(mic_ring[k * TICK_SAMPS : (k + 1) * TICK_SAMPS]))
                st["ticks"] += 1
        hop_samps = int(round(HOP_FRAMES * FRAME_HOP))
        for t_end in range(hop_samps, len(mic) + hop_samps, hop_samps):
            mouth.hop(mic_ring, min(t_end, len(mic)), notes if notes else [None],
                      st["ticks"] if ear is not None else len(notes),
                      closing=t_end >= len(mic))
        angel = out_ring[: len(mic)]
        th = np.array(mouth.t_hop) * 1000
        print(f"hops {len(th)} | p50 {np.percentile(th,50):.0f} p95 "
              f"{np.percentile(th,95):.0f} ms vs hop {HOP_FRAMES*FRAME_MS:.0f}")
        for name, sig in [(a.file[1], angel),
                          (a.file[1].replace(".wav", "_mix.wav"),
                           0.5 * mic + 0.8 * angel)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
            print("wrote", name)
        return

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return

    def cb(indata, outdata, frames, t, status):
        n = st["in"]
        mic_ring[n : n + frames] = indata[:, 0]
        pos = n + frames - lag
        if pos <= 0:
            outdata[:] = 0
        else:
            seg = out_ring[max(0, pos - frames) : pos] * a.gain
            outdata[:, 0] = np.pad(seg, (frames - len(seg), 0))
            if outdata.shape[1] > 1:
                outdata[:, 1] = outdata[:, 0]
            if pos > getattr(mouth, "frontier", 0) and (notes and any(
                    x is not None for x in notes[-3:])):
                st["under"] += frames  # playhead ahead of the emit frontier
        with lock:
            st["in"] = n + frames

    threading.Thread(target=worker, daemon=True).start()
    dev = (a.in_name, a.out_name)
    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, 2),
                   device=dev, callback=cb):
        print(f"live. lag {a.lag*1000:.0f} ms, C major -- Ctrl-C stops.", flush=True)
        try:
            while st["in"] < cap - SR * 10:
                time.sleep(2)
                cur = next((x for x in reversed(notes) if x is not None), None) \
                    if notes else None
                if mouth.phrase is None:
                    mtxt = "  rest"  # frontier legitimately frozen: no meaning
                else:
                    margin = (mouth.frontier - (st["in"] - lag)) / SR
                    mtxt = f"{margin*1000:+5.0f} ms"
                print(f"in {st['in']/SR:6.1f}s | ticks {st['ticks']} | note {cur} "
                      f"| margin {mtxt} | starved {st['under']/SR:.1f}s",
                      flush=True)
        except KeyboardInterrupt:
            pass
    st["done"] = True
    th = np.array(mouth.t_hop) * 1000 if mouth.t_hop else np.zeros(1)
    print(f"\nhops {len(th)} | p50 {np.percentile(th,50):.0f} "
          f"p95 {np.percentile(th,95):.0f} ms vs hop {HOP_FRAMES*FRAME_MS:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--notes-json", default=None)
    ap.add_argument("--accent", default="cpdl", choices=["chorale", "pop909", "cpdl"])
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--lag", type=float, default=0.5,
                help="playback delay behind the voice; must cover tick 187.5 + hop 150 + compute ~80 + hold 40 ms (see worklog 07-24)")
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--mouth", choices=["world", "ddsp"], default="world",
                    help="ddsp needs the DDSP venv: 260724_ddsp_svc/venv/bin/python")
    import pathlib as _pl
    sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
    from config import DDSP as _DDSP
    ap.add_argument("--ddsp-repo", default=str(_DDSP))
    ap.add_argument("--ddsp-model",
                    default=str(_DDSP / "exp/combsub-harry/model_30000.pt"))
    a = ap.parse_args()
    run(a)
