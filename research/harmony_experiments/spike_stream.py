"""spike_stream.py - a sliding-window streaming spike for the DDSP carrier.

The problem: the real-time line died on batch triggering, at a latency floor.
This tests two make-or-break numbers for continuous sliding-window conversion:
how the end-to-end latency feels live, and whether the streamed sound carries the
dead quality that killed the earlier naive resynthesis (G28).

The architecture copies the real-time core of DDSP-SVC's gui.py - rolling
window, SOLA alignment, crossfaded splices - with three differences: it is
headless, the device is MPS rather than CUDA or CPU, and f0 comes from
parselmouth because the rmvpe weights are not present. A single-part identity
conversion comes first (--pitch transposes); the harmony layer goes on top only
once that passes.

Run (DDSP venv):
  python spike_stream.py                          # identity through his own model
  python spike_stream.py --pitch -12 --model exp/combsub-m4-bass1/model_11000.pt
  python spike_stream.py --sustain 250            # v15 phrase bridging
Ctrl-C to stop. Each block prints its inference time; above the block duration
the sound breaks up.
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
                   "SoloChoirCode/260724_ddsp_svc")
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, \
    Units_Encoder  # noqa: E402
from ddsp.core import upsample  # noqa: E402
from enhancer import Enhancer  # noqa: E402

SR = 44100
DDSP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
        "SoloChoirCode/260724_ddsp_svc")
MAJ = [0, 2, 4, 5, 7, 9, 11]


def dia_from_note(note, steps, root):
    """A settled note to a semitone offset, that many diatonic steps away."""
    rel = int(note) - root
    oc, pc = divmod(rel, 12)
    di = min(range(7), key=lambda i: min((MAJ[i] - pc) % 12,
                                         (pc - MAJ[i]) % 12))
    to, ti = divmod(di + steps, 7)
    return float(root + (oc + to) * 12 + MAJ[ti] - int(note))


class NoteTracker:
    """Note tracking with hysteresis (the core of v11 sustain stability): the
    note only changes after deviating more than 0.65 semitones for at least 3
    frames, about 35 ms, so vibrato and small excursions across a boundary no
    longer chatter. Frame by frame and causal, so it is consistent across windows."""

    def __init__(self):
        self.cur = None
        self.pend = 0
        self.hist = []
        self.streak = 0

    def feed(self, f0_hz):
        """Returns (notes, streaks), where streak is how many consecutive frames
        have held the same note (0 means it just changed, or silence). This is the
        gate for the v12 units smoothing: pin when stable, follow instantly on a change."""
        out = np.zeros(len(f0_hz))
        stk = np.zeros(len(f0_hz), dtype=int)
        for h, f in enumerate(f0_hz):
            if f > 0:
                m = 69 + 12 * np.log2(f / 440.0)
                self.hist = (self.hist + [m])[-5:]
                mm = float(np.median(self.hist))
                if self.cur is None:
                    self.cur = round(mm)
                    self.streak = 0
                elif abs(mm - self.cur) > 0.65:
                    self.pend += 1
                    if self.pend >= 3:
                        self.cur, self.pend = round(mm), 0
                        self.streak = 0
                else:
                    self.pend = 0
                    self.streak += 1
            else:
                self.streak = 0
            out[h] = self.cur if self.cur is not None else 0.0
            stk[h] = self.streak
        return out, stk


def dia_offsets(f0c, steps, root):
    """Frozen f0 in Hz per hop to a per-frame semitone offset, the diatonic
    interval. Quantisation decides the interval only; the curve itself does not
    move. Deterministic: it depends only on a local median of the frozen f0, so
    it is consistent across windows and needs no separate cache."""
    v = f0c > 0
    m = np.where(v, 69 + 12 * np.log2(np.maximum(f0c, 1.0) / 440.0), 0.0)
    # a local median over 5 frames damps vibrato jitter, so note boundaries do not chatter
    k = 2
    mm = np.copy(m)
    for h in range(len(m)):
        w = m[max(0, h - k):h + k + 1]
        w = w[w > 0]
        if len(w):
            mm[h] = np.median(w)
    off = np.zeros(len(m))
    for h in range(len(m)):
        if mm[h] <= 0:
            off[h] = off[h - 1] if h else 0.0
            continue
        note = int(round(mm[h]))
        rel = note - root
        oc, pc = divmod(rel, 12)
        di = min(range(7), key=lambda i: min((MAJ[i] - pc) % 12,
                                             (pc - MAJ[i]) % 12))
        to, ti = divmod(di + steps, 7)
        target = root + (oc + to) * 12 + MAJ[ti]
        off[h] = float(target - note)
    return off


def phase_vocoder(a, b, fade_out, fade_in):
    """Straight from gui.py:15: phase continuity across the splice, the cure for skipping."""
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(
        deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    return (a * (fade_out ** 2) + b * (fade_in ** 2)
            + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n)


class Svc:
    """A trimmed SvcDDSP from gui.py, extended to several parts: one encoder, f0,
    volume and enhancer shared across N voices.
    voices: [(model_path, semitones, gain), ...]"""

    def __init__(self, voices, device="mps", enhance=True):
        self.device = device
        self.voices = []
        for path, semi, gain in voices:
            model, args = load_model(path, device=device)
            self.voices.append((model, semi, gain))
        self.args = args                      # one pipeline, so the data parameters are shared
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            f"{DDSP}/{self.args.data.encoder_ckpt}",
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size, device=device)
        self.enhancer = (Enhancer(self.args.enhancer.type,
                                  f"{DDSP}/{self.args.enhancer.ckpt}",
                                  device=device) if enhance else None)
        self.vol_ex = Volume_Extractor(self.args.data.block_size)
        self.spk = torch.LongTensor([[1]]).to(device)

    def prep(self, audio, threhold=-60.0, want_uv=False):
        """Returns (f0_np, vol_t, mask[, uv]). f0 stays numpy so the caller can put
        it on the frozen grid. want_uv also returns parselmouth's real voicing (the
        frames where f0 is 0): with a throat microphone a consonant's level never
        falls below the -60 dB threshold, so the vocal-fold decision is the honest
        one. The interpolation follows vocoder.py:142-146 byte for byte."""
        hop = self.args.data.block_size
        pe = F0_Extractor("parselmouth", SR, hop, 65.0, 800.0)
        if want_uv:
            f0_np = pe.extract(audio, uv_interp=False, device=self.device)
            uv = f0_np == 0
            if len(f0_np[~uv]) > 0:
                f0_np[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0],
                                      f0_np[~uv])
            f0_np[f0_np < 65.0] = 65.0
        else:
            f0_np = pe.extract(audio, uv_interp=True, device=self.device)
        vol = self.vol_ex.extract(audio)
        mask = (vol > 10 ** (threhold / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n:n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device)[None, :, None]
        mask = upsample(mask, hop).squeeze(-1)
        vol_t = torch.from_numpy(vol).float().to(self.device)[None, :, None]
        if want_uv:
            return f0_np, vol_t, mask, uv
        return f0_np, vol_t, mask

    def infer(self, audio, pitch_adjust=0.0, threhold=-60.0,
              units_override=None, feats=None, phases=None, ratios=None,
              enh_tail=0):
        """One output per part: shared units, f0 and volume, each with its own
        transposition and gain. feats = (f0_np, vol_t, mask) supplied from outside
        so nothing is recomputed, for the frozen grid. phases are each part's
        absolute phase in radians, so the comb source stays in phase across windows
        (the fix for v6's skipping). ratios are per-frame semitone offsets per
        part; None uses the fixed semitone value."""
        hop = self.args.data.block_size
        f0_np, vol_t, mask = feats if feats is not None \
            else self.prep(audio, threhold)
        f0 = torch.from_numpy(f0_np).float().to(self.device)[None, :, None]
        units = units_override
        if units is None:
            units = self.encode(audio)
        n = min(units.size(1), f0.size(1), vol_t.size(1))
        outs, fvs = [], []
        with torch.no_grad():
            for vi, (model, semi, gain) in enumerate(self.voices):
                if ratios is not None and ratios[vi] is not None:
                    r = torch.from_numpy(
                        ratios[vi][:n]).float().to(self.device)[None, :, None]
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + r) / 12.0)
                else:
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + semi) / 12.0)
                ip = (None if phases is None else torch.tensor(
                    [[[phases[vi]]]], dtype=torch.float32,
                    device=self.device))
                out, _, _ = model(units[:, :n], fv, vol_t[:, :n],
                                  spk_id=self.spk, initial_phase=ip)
                out *= mask[:, :n * hop]
                outs.append(out)
                fvs.append(fv)
            if self.enhancer is not None:
                # all three parts go through hifigan in one batch; calling it per
                # part costs three kernel launches. At 44.1k/512 with adaptive_key
                # 0, enhance() is transparent to batching along its whole path.
                ob, fb = torch.cat(outs, 0), torch.cat(fvs, 0)
                if enh_tail and enh_tail < ob.size(-1):
                    # v23: enhance only the tail the step actually consumes plus a
                    # margin, worth about 28 ms. The NSF phase integrates from the
                    # call's start, so a tail-only call shifts the phase - but the
                    # phase restarts every window anyway, and SOLA plus the phase
                    # vocoder are what treat the splice. Whether it is audible is
                    # settled by the replay splice metric and by ear.
                    tl = (enh_tail // hop + 1) * hop
                    eb, esr = self.enhancer.enhance(
                        ob[:, -tl:], self.args.data.sampling_rate,
                        fb[:, -tl // hop:], hop, adaptive_key=0)
                    assert esr == SR
                    L = min(tl, eb.size(-1), ob.size(-1))
                    ob = ob.clone()
                    ob[:, -L:] = eb.reshape(eb.size(0), -1)[:, -L:]
                else:
                    eb, esr = self.enhancer.enhance(
                        ob, self.args.data.sampling_rate, fb, hop,
                        adaptive_key=0)
                    assert esr == SR
                    ob = eb.reshape(eb.size(0), -1)[:, :ob.size(-1)] \
                        if eb.dim() > 2 else eb
                outs = [ob[vi:vi + 1] for vi in range(len(self.voices))]
        return [out.squeeze() * gain for out, (_, _, gain)
                in zip(outs, self.voices)]
        # returned per part, so each splice aligns on its own; SOLA cannot align two periods in a mix

    def encode(self, audio):
        au = torch.from_numpy(audio).float()[None].to(self.device)
        return self.units_encoder.encode(au, SR, self.args.data.block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=f"{DDSP}/exp/combsub-harry-260730/model_30000.pt",
                    help="model used by --solo mode")
    ap.add_argument("--solo", action="store_true",
                    help="single-part identity, for judging timbre and latency; "
                         "the default is harmony (S = Soprano-3 +12, B = Bass-1 -12)")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--no-enhance", dest="enh", action="store_false")
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--out-name", default="AI-Micro")
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--thr", type=float, default=-60.0,
                    help="volume threshold in dB. The recipe uses -60; gui.py's -45 "
                         "is too high for a throat microphone and chops the output up")
    ap.add_argument("--key", type=int, default=0,
                    help="major tonic pitch class (0=C to 11=B), for the diatonic mapping")
    ap.add_argument("--no-brain", dest="brain", action="store_false",
                    help="turn the harmony model off and fall back to the v12 fixed diatonic mapping")
    ap.add_argument("--octave", type=int, default=12,
                    help="model space shift, as in respond2")
    ap.add_argument("--block", type=float, default=0.10,
                    help="seconds per block, the dominant latency term; the inference "
                         "time has to stay under it (benched: a 1.0 s window has p95 39 ms)")
    ap.add_argument("--crossfade", type=float, default=0.04)
    ap.add_argument("--extra", type=float, default=1.0,
                    help="seconds of context prepended: timbre against compute")
    ap.add_argument("--sustain", type=float, default=0.0,
                    help="v15 phrase bridging in ms: an unvoiced gap this short or "
                         "shorter is sung through on the vowel; 0 disables it")
    ap.add_argument("--release", type=float, default=120.0,
                    help="a real rest, longer than sustain: fade out over this many ms "
                         "instead of chopping with a hard gate")
    ap.add_argument("--floor-db", type=float, default=0.0,
                    help="v21 volume floor in dBFS (0 disables): frames below this are "
                         "forced to unvoiced, so the gate can close and the model takes a "
                         "rest. This treats breath and friction on a throat microphone being "
                         "called voiced by parselmouth: 61% of gate-open frames peaked below "
                         "a tenth of the singing level with a median f0 of 562 Hz, which is "
                         "the raw material of the strange sound. -36 killed 100% of that "
                         "with no false positives")
    ap.add_argument("--dump", default="",
                    help="on exit write <dump>_{mic,angel}.wav and _feats.npz, with "
                         "per-frame f0, voicing, gate, bridge, volume and interval")
    ap.add_argument("--enc-win", type=float, default=0.0,
                    help="v26 hubert encoding window in seconds (0 uses the render "
                         "window): encoding a longer past buys articulation as left "
                         "context at no latency cost. Dose response: 0.7s gives 0.524, "
                         "1.5s 0.694, 3s 0.788, against 0.813 offline, costing 15 to 40 ms")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="v25 lookahead in frames: the render and consumption window "
                         "shifts back by N frames, giving that region N frames of right "
                         "context for hubert, the EMA and the note, at a cost of N x 11.6 ms "
                         "of output latency. 0 disables it")
    ap.add_argument("--rehearse", default="",
                    help="v24 rehearsal-score mode: notes.json (lead/upper/lower, "
                         "absolute pitch in microphone space, on a 186 ms tick grid, "
                         "produced from a dump by scratchpad/make_score.py). ScoreTracker "
                         "v3 aligns to the lead and the parts sing the score, so the harmony "
                         "is guaranteed by the score and the model is bypassed entirely")
    ap.add_argument("--reh-rate", type=float, default=1.0,
                    help="tempo prior for the rehearsal alignment; a fixed value beats self-estimation")
    ap.add_argument("--enh-tail", action="store_true",
                    help="v23: run the enhancer only over the tail SOLA consumes plus a "
                         "0.1 s margin, worth about 28 ms")
    ap.add_argument("--dump-stems", action="store_true",
                    help="add per-part stems to the dump, for attribution; three times the file size")
    ap.add_argument("--ema-alpha", type=float, default=0.12,
                    help="v12 units EMA coefficient while the note is stable. Higher "
                         "follows the vowel faster and articulates more sharply, lower pins harder")
    ap.add_argument("--ema-streak", type=int, default=6,
                    help="frames of stability before the EMA starts pinning (6 is about 70 ms)")
    ap.add_argument("--agc", action="store_true",
                    help="v22 streaming AGC, the causal version of direct_mouth's input "
                         "AGC: pull the voiced level towards the training reference RMS of "
                         "0.0566 before the model and divide it back out afterwards, so the "
                         "level structure is unchanged. This treats conversion degrading when "
                         "the level falls outside the training distribution, the main suspect "
                         "for mushy, weak articulation (measured degrading monotonically from "
                         "14.3 to 18.3 dB). Off by default")
    ap.add_argument("--replay", default="",
                    help="v22 offline replay: read a wav and run it block by block down "
                         "the same step and model path, with no audio device open and the "
                         "model draining synchronously each block, so it lags by one block "
                         "exactly as it does live. With --dump it produces something directly "
                         "comparable to a live dump, so quality can be iterated without singing again")
    a = ap.parse_args()

    print("loading models…", flush=True)
    ear = kt = None
    REH = None
    if a.rehearse:
        # v24: in score mode the model is bypassed. All three voices share the configuration; the note lines come from the score tracker.
        import json
        from rehearse_ab import ScoreTracker
        d = json.load(open(a.rehearse))
        REH = {"upper": d["upper"], "lower": d["lower"],
               "tracker": ScoreTracker(d["lead"], mode="v3",
                                       rate0=a.reh_rate)}
        print(f"[rehearse] score {len(d['lead'])} ticks", flush=True)
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.8),
                  (f"{DDSP}/exp/combsub-girl/model_30000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]
    elif a.solo:
        voices = [(a.model, 0.0, 1.0)]
        dia_steps = [None]
        vmode = [None]
    elif a.brain:
        # v13: the harmony model runs on the EarV3 tick stream, which is a live
        # machine already. The upper line goes to one voice, the lower to Bass-1,
        # the soprano an octave above the melody. The model's output is converted
        # into an interval relative to the singer's pitch before it enters the
        # cache, so the interval is constant.
        import respond2 as R2
        from live_v3 import EarV3
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.8),
                  (f"{DDSP}/exp/combsub-girl/model_30000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]        # the soprano is fixed at +12; the other two follow the model
        ear = EarV3(indep=0.15, stab=1, key=0)
        ear.v2t.shift = a.octave
        ear.k_shift = 0        # the key lock is maintained externally by KeyTracker
        kt = R2.KeyTracker()
    else:
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt",
                   -3.0, 0.9),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [7, -2, -7]
        vmode = [None, None, None]
    svc = Svc(voices, enhance=a.enh)
    HOP = svc.args.data.block_size
    blk = max(1, round(a.block * SR / HOP)) * HOP     # aligned to 512, so the grid does not drift
    cf = int(a.crossfade * SR)
    sola_search = int(0.01 * SR)
    last_delay = int(0.02 * SR)
    LA = a.lookahead * HOP              # v25 lookahead in samples; 0 disables
    input_frame = ((max(int(a.extra * SR),
                        blk + cf + sola_search + 2 * last_delay + LA)
                    // HOP + 1) * HOP)
    nb = blk // HOP
    buf = np.zeros(input_frame, dtype="float32")
    ENC_WIN = (max(int(a.enc_win * SR) // HOP * HOP, input_frame)
               if a.enc_win else 0)
    buf3 = np.zeros(ENC_WIN, dtype="float32") if ENC_WIN else None
    sola_bufs = [torch.zeros(cf, device=svc.device) for _ in svc.voices]
    fade_in = torch.sin(np.pi * torch.arange(0, 1, 1 / cf,
                                             device=svc.device) / 2) ** 2
    fade_out = 1 - fade_in
    stats = {"n": 0, "ms": [], "late": 0}
    st = {"uc": None, "fc": None, "offc": None, "tb": np.zeros(0),
          "cur": {"up": 12.0, "lo": -12.0}, "ktn": 0,
          "hole": 10 ** 9, "env": 0.0, "nbh": 0, "nbf": 0,
          "f0q": [], "hf0": 0.0, "ufifo": [], "inbr": False,
          "ivp": {"up": None, "lo": None}, "klock": False,
          "agc_g": 1.0, "agc_ema": None}
    accs = [0.0] * len(svc.voices)      # per-part absolute phase accumulator, in radians
    tracker = NoteTracker()             # note hysteresis (v11 sustain stability)
    from collections import defaultdict
    dmp = defaultdict(list) if a.dump else None
    sus_f = int(round(a.sustain / 1000.0 * SR / HOP))   # bridging limit, in frames
    rel_f = max(1, int(round(a.release / 1000.0 * SR / HOP)))
    ENV_DK = float(np.exp(-(HOP / SR) / 0.15))   # volume inertia, tau about 150 ms

    def bridge_feed(vol_np, uv, f0_np):
        """v18 phrase bridging, causal and frame by frame. Returns (gate, vol,
        bridged, f0_hold), fed only new frames so the result can be frozen into the
        grid and stay consistent across windows.

        A gap opens when the vocal folds read unvoiced AND the level falls below a
        quarter of the envelope (-12 dB), which rejects the false unvoiced readings
        at the tail of a sliding window. Whether the gap CONTINUES depends on
        voicing alone: in v17, during a long rest the envelope decays to the noise
        floor, the relative test stops working, the gate reopens, and the bass sings
        a low hum on interpolated f0.

        On holding: the frame at the mouth of the gap is already contaminated by the
        consonant (f0 guessed at the window tail, dirty units), so it cannot be the
        one held. f0 is taken from a safe frame about 45 ms earlier, the oldest of a
        four-frame FIFO, which is lookahead after the fact and free. The volume comes
        from the envelope, following the onset instantly and releasing at tau about
        150 ms. A short gap holds the gate open; a real rest holds to the limit and
        then fades out over rel_f frames with a cosine."""
        g = np.zeros(len(vol_np))
        v = np.copy(vol_np)
        b = np.zeros(len(vol_np), dtype=bool)
        vf0 = np.copy(f0_np)
        for j, x in enumerate(vol_np):
            st["env"] = max(float(x), st["env"] * ENV_DK)
            if not (uv[j] and (st["hole"] > 0 or x < st["env"] * 0.25)):
                st["hole"] = 0
                g[j] = 1.0
                st["f0q"] = (st["f0q"] + [float(f0_np[j])])[-4:]
            else:
                st["hole"] += 1
                if st["hole"] == 1:
                    st["hf0"] = st["f0q"][0] if st["f0q"] else float(f0_np[j])
                if st["hole"] <= sus_f:
                    g[j], v[j], b[j] = 1.0, st["env"], True
                    vf0[j] = st["hf0"]
                    st["nbf"] += 1
                    if st["hole"] == 1:
                        st["nbh"] += 1
                elif st["hole"] <= sus_f + rel_f:
                    r = (st["hole"] - sus_f) / rel_f
                    g[j] = float(np.cos(r * np.pi / 2) ** 2)
                    v[j], b[j] = st["env"], True
                    vf0[j] = st["hf0"]
        return g, v, b, vf0

    # warm up with the same window size and the same ratios path, so the MPS kernels are compiled before the stream opens
    nfrm = input_frame // HOP + 1
    svc.infer(buf.astype("float64") + 1e-6, a.pitch, a.thr,
              ratios=[np.zeros(nfrm) if (s is not None or vm) else None
                      for s, vm in zip(dia_steps, vmode)])
    print(f"ready  block {blk/SR*1000:.0f}ms / window {input_frame/SR:.2f}s",
          flush=True)

    import threading
    in_q, out_q = [], []
    qlock = threading.Lock()
    ev = threading.Event()
    st2 = {"under": 0, "flags": 0, "die": False}

    def cb(indata, outdata, frames, tinfo, status):
        # v9: the callback only moves memory, in microseconds. Inference happens in
        # the worker. Putting 110 ms of MPS work here was the real cause of the
        # break-up in v1 to v8: the CoreAudio deadline is hard, and meeting it on
        # average is not enough.
        if status:
            st2["flags"] += 1
        with qlock:
            in_q.append(indata[:, 0].copy())
            have = sum(len(q) for q in out_q)
            outdata[:] = 0
            if have >= frames:
                need = frames
                col = []
                while need > 0:
                    q = out_q[0]
                    take = min(need, len(q))
                    col.append(q[:take])
                    if take == len(q):
                        out_q.pop(0)
                    else:
                        out_q[0] = q[take:]
                    need -= take
                y = np.concatenate(col)
                outdata[:len(y), 0] = y
                if outdata.shape[1] > 1:
                    outdata[:len(y), 1] = y
            else:
                st2["under"] += 1
        ev.set()

    def worker():
        t_pending = np.zeros(0, dtype="float32")
        while not st2["die"]:
            ev.wait(0.5)
            ev.clear()
            with qlock:
                if in_q:
                    t_pending = np.concatenate([t_pending] + in_q)
                    in_q.clear()
            while len(t_pending) >= blk:
                chunk, t_pending = t_pending[:blk], t_pending[blk:]
                step(chunk)

    bq, block = [], threading.Lock()
    TICK_FRAMES = 16                    # 16 x 512 / 44100, about 186 ms, the original tick

    def ear_note_tick(f_in):
        """v14 note feed-through: ear.tick minus tracker.push and the key lock,
        which were the whole 55 ms cost. k_shift is maintained externally by
        KeyTracker and f_in comes from the frozen f0. The reactive path matches
        live_v3.EarV3.tick line for line.

        v24 rehearsal: the score tracker replaces the model. Absolute pitch in
        microphone space goes straight through, with no tokens, no snapping and no
        key shift; when the singer leaves the score, the parts anchor to it."""
        if REH is not None:
            tr = REH["tracker"]
            tr.observe(None if f_in is None else int(f_in))
            if f_in is not None:
                p = tr.p
                for part, line in (("up", REH["upper"]),
                                   ("lo", REH["lower"])):
                    note = line[p] if 0 <= p < len(line) else None
                    if note is not None:
                        st["cur"][part] = float(int(note) - int(f_in))
            return
        from live_v3 import token_to_midi
        real = f_in                 # the note actually sung, before snapping or folding
        if f_in is not None and ear.k_shift:
            f_in = int(f_in) + ear.k_shift
        s = ear.v2t.token(f_in)
        toks = ear.brain.step(s)
        u, l = ear._to_notes(toks)
        ear.lead_prev = token_to_midi(s, ear.lead_prev)
        ear.lead_hist.append(ear.lead_prev)
        lead = ear.lead_prev
        offsp = ear.v2t.shift + (ear.k_shift or 0)
        if lead is not None:
        # v18: the reference is the note actually sung. The old lead minus offset
        # was an internal representation after snapping and folding, and in C major
        # snapping put it a semitone out, which fired on 40% of ticks and made the
        # bass sing a semitone sharp.
        # v20: that reference then has to lose the same snapping offset d, because
        # the model writes harmony entirely in the snapped space. Computing the
        # interval against the unsnapped real note gives the model's interval minus
        # d: a bass aiming at the unison lands on -1, right against the voice, and
        # a fifth becomes a tritone. Measured: P(bass = -1) is 24% snapped against
        # 4% unsnapped; correcting it took the -1 share from 11.3% to 4.2%.
            if sus_f and real is not None:
                d_snap = 1 if (real + offsp) % 12 in (1, 3, 6, 8, 10) else 0
                mic = real - d_snap
            else:
                mic = lead - offsp
            for part, note in (("up", u), ("lo", l)):
                if note is None:
                    continue
                iv = float(note - mic)
                if not sus_f:
                    st["cur"][part] = iv
                    continue
                # v19 interval hysteresis: the harmony was changing note 7 times a
                # second with a median dwell of 12 ms, which is why there was no sense
                # of harmony. A new interval now has to hold for two consecutive ticks,
                # putting the dwell floor at about 370 ms, so a single roll of the dice
                # never reaches the output.
                if iv == st["cur"][part]:
                    st["ivp"][part] = None
                elif st["ivp"][part] == iv:
                    st["cur"][part] = iv
                    st["ivp"][part] = None
                else:
                    st["ivp"][part] = iv

    brain_acc = []

    def brain_drain():
        """Consume the frozen f0 frames accumulated in bq into ticks. Live this is
        called from a thread; in replay it is called synchronously after each step,
        so the same logic runs and the model lags by one block either way."""
        with block:
            if bq:
                brain_acc.extend(bq)
                bq.clear()
        while len(brain_acc) >= TICK_FRAMES:
            w = np.array(brain_acc[:TICK_FRAMES])
            del brain_acc[:TICK_FRAMES]
            v = w[w > 0]
            f_in = (int(round(float(np.median(
                69 + 12 * np.log2(v / 440.0)))))
                if len(v) >= TICK_FRAMES // 2 else None)
            try:
                ear_note_tick(f_in)
            except Exception as e:  # noqa: BLE001
                print("brain err:", e, flush=True)

    def brain_worker():
        # the model thread consumes frozen f0 frames and never touches audio, so it costs almost nothing
        while not st2["die"]:
            time.sleep(0.02)
            brain_drain()

    def step(chunk):
        t0 = time.perf_counter()
        buf[:-blk] = buf[blk:]
        buf[-blk:] = chunk
        if ENC_WIN:
            buf3[:-blk] = buf3[blk:]
            buf3[-blk:] = chunk
        try:
                # v11 frozen f0 grid: freeze a frame as it arrives (the tail estimate
                # is good to p95 4.3 cents), and freeze the hysteretic note, its
                # interval and its stability streak with it, so rendering and the phase
                # accounting use exactly the same values.
            xb = buf.astype("float64")
            if a.agc:
                xb = xb * st["agc_g"]      # v22: the model sees a levelled signal, divided back out later
            if sus_f:
                f0f, vol_t, mask, uvf = svc.prep(xb, a.thr, want_uv=True)
                if a.floor_db < 0:
                    # v21: frames below the floor count as unvoiced, so the downstream
                    # bridge gate and the model's rest detection both follow from one
                    # place. The threshold is defined on the raw level, so with the AGC
                    # on it is scaled by the same gain.
                    uvf = uvf | (vol_t[0, :, 0].cpu().numpy()
                                 < 10 ** (a.floor_db / 20.0) * st["agc_g"])
            else:
                f0f, vol_t, mask = svc.prep(xb, a.thr)
            if a.agc:
                # gain update, causal and effective one block later: an EMA over the
                # raw level of the voiced frames (tau about 2 s), giving g = ref / ema,
                # clamped to [0.25, 16] as the offline input AGC is.
                vn = vol_t[0, -nb:, 0].cpu().numpy() / st["agc_g"]
                act = vn > 10 ** (a.thr / 20.0)
                if act.any():
                    al = 1 - float(np.exp(-(blk / SR) / 2.0))
                    m = float(np.median(vn[act]))
                    st["agc_ema"] = m if st["agc_ema"] is None else \
                        (1 - al) * st["agc_ema"] + al * m
                    st["agc_g"] = float(np.clip(
                        0.0566 / max(st["agc_ema"], 1e-6), 0.25, 16.0))
            fc = st["fc"]
            if fc is None or len(fc) != len(f0f):
                if sus_f:
                    g, v, b, vf0 = bridge_feed(
                        vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                    st["gt"], st["vc"], st["br"] = g, v, b
                    st["fc"] = vf0      # a bridged frame freezes the safe f0; a voiced frame keeps its own
                else:
                    st["fc"] = f0f
                notes, stks = tracker.feed(f0f)
                st["stk"] = stks
                st["offc"] = []
                for vi, s in enumerate(dia_steps):
                    if s is not None:
                        st["offc"].append(np.array(
                            [0.0 if n <= 0 else dia_from_note(n, s, a.key)
                             for n in notes]))
                    elif vmode[vi]:
                        st["offc"].append(
                            np.full(len(f0f), st["cur"][vmode[vi]]))
                    else:
                        st["offc"].append(None)
            else:
                # advance the absolute phase: the nb frames leaving the window, times the interval that was rendered
                for vi, (_, semi, _) in enumerate(svc.voices):
                    if st["offc"][vi] is not None:
                        adv = float((fc[:nb] * 2 ** (
                            (a.pitch + st["offc"][vi][:nb]) / 12.0)).sum())
                    else:
                        adv = float(fc[:nb].sum()) \
                            * 2 ** ((a.pitch + semi) / 12.0)
                    accs[vi] = (accs[vi] + 2 * np.pi * adv * HOP / SR) \
                        % (2 * np.pi)
                if sus_f:
                    if st.get("gt") is None:    # self-healing, so an exception during init does not wedge it permanently
                        g, v, b, vf0 = bridge_feed(
                            vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                        st["gt"], st["vc"], st["br"] = g, v, b
                        st["fc"] = vf0
                    else:
                        g, v, b, vf0 = bridge_feed(
                            vol_t[0, -nb:, 0].cpu().numpy(),
                            uvf[-nb:], f0f[-nb:])
                        st["gt"] = np.concatenate([st["gt"][nb:], g])
                        st["vc"] = np.concatenate([st["vc"][nb:], v])
                        st["br"] = np.concatenate([st["br"][nb:], b])
                        st["fc"] = np.concatenate([fc[nb:], vf0])
                else:
                    st["fc"] = np.concatenate([fc[nb:], f0f[-nb:]])
                if ear is not None or REH is not None:
                    with block:
                        # v18: unvoiced frames send 0, so the model receives a rest.
                        # The old code fed interpolated f0, which produced 0 rests in
                        # 484 ticks and took the token stream outside its training
                        # distribution.
                        bq.extend((np.where(uvf[-nb:], 0.0, f0f[-nb:])
                                   if sus_f else f0f[-nb:]).tolist())
                notes, stks = tracker.feed(f0f[-nb:])
                st["stk"] = np.concatenate([st["stk"][nb:], stks])
                for vi, s in enumerate(dia_steps):
                    if s is not None:
                        new = np.array(
                            [0.0 if n <= 0 else dia_from_note(n, s, a.key)
                             for n in notes])
                    elif vmode[vi]:
                        tgt = st["cur"][vmode[vi]]
                        new = np.full(nb, tgt)
                        pv = float(st["offc"][vi][-1])
                        if sus_f and pv != tgt:
                            # v18: slide about 35 ms into a new interval rather than
                            # jumping a whole block, in the spirit of the offline porta.
                            k3 = min(nb, 3)
                            new[:k3] = np.linspace(pv, tgt, k3 + 2)[1:k3 + 1]
                    else:
                        continue
                    st["offc"][vi] = np.concatenate(
                        [st["offc"][vi][nb:], new])
                if kt is not None:
                    st["ktn"] += 1
                    if st["ktn"] % 10 == 0:        # push the key tracker about every 2 s
                        kt.push(st["fc"])
                        rb = kt.best()
                        if rb is not None and rb[2] >= 0.015 and \
                                not (sus_f and st["klock"]):
                            # v19: once locked, k_shift never moves again (the same
                            # discipline as respond2); re-estimating continuously shifts
                            # every interval the moment it drifts.
                            ear.k_shift = rb[0]
                            st["klock"] = True
            # frozen units cache (v2) plus v12 sustain smoothing. On a held vowel,
            # hubert guesses a different vowel frame by frame at the window tail and
            # the output wanders. Once the note is stable (streak >= 6, about 70 ms),
            # units follow an EMA with alpha 0.12 (tau about 90 ms) and pin; on a note
            # change or a consonant the streak resets and alpha goes to 1, so the onset
            # is not blurred. The smoothing runs causally on the absolute grid, so it
            # is consistent across windows.
            if ENC_WIN:
                # v26: encode 3 s of the past as free left context for articulation,
                # then take only the tail frames matching the render window, so nothing
                # downstream can tell the difference.
                xbe = buf3.astype("float64")
                if a.agc:
                    xbe = xbe * st["agc_g"]
                fresh = svc.encode(xbe)[:, -(input_frame // HOP + 1):]
            else:
                fresh = svc.encode(xb)
            uc = st["uc"]
            if uc is None or uc.size(1) != fresh.size(1):
                st["uc"] = fresh
                st["usm"] = fresh.clone()
            else:
                keep = uc[:, nb:]
                m = 2 * nb
                st["uc"] = torch.cat(
                    [keep[:, :fresh.size(1) - m], fresh[:, -m:]], 1)
                us = st["usm"][:, nb:]
                # v23: the per-frame EMA loop runs on the CPU in numpy (256 dimensions
                # by nb frames costs nothing there); it used to launch two or three MPS
                # kernels per frame, which was the bulk of the step's hidden cost. One
                # download, one upload, numerically identical to a float32 lerp.
                fr = fresh[:, -nb:].detach().cpu().numpy()
                prev = us[:, -1].detach().cpu().numpy()
                news = np.empty_like(fr)
                for j in range(nb):
                    if sus_f and st["br"][-nb + j]:
                        if not st["inbr"]:
                            # v18 rewind at the mouth of the gap: that frame is already
                            # contaminated by the consonant, so the frozen vowel comes
                            # from the oldest FIFO entry, about 45 ms earlier
                            if st["ufifo"]:
                                prev = st["ufifo"][0]
                            st["inbr"] = True
                        al = 0.0
                    else:
                        st["inbr"] = False
                        al = a.ema_alpha \
                            if st["stk"][-nb + j] >= a.ema_streak else 1.0
                    prev = (1 - al) * prev + al * fr[:, j]
                    if sus_f and not st["inbr"]:
                        st["ufifo"] = (st["ufifo"] + [prev])[-4:]
                    news[:, j] = prev
                st["usm"] = torch.cat(
                    [us, torch.from_numpy(news).to(us)], 1)
            if sus_f:
                # v18: the bridge gate replaces the original mask outright. A throat
                # microphone's noise floor at -55 dB sits above the -60 dB threshold, so
                # the original mask is always open (20 minutes produced 3 gaps); taking
                # the max of the two is the same as having no gate, real rests never
                # close, and the bass hums. The bridge gate already means "voiced =
                # open". The volume comes from the frozen cache, holding the envelope
                # inside a gap.
                gu = upsample(torch.from_numpy(st["gt"]).float().to(
                    svc.device)[None, :, None], HOP).squeeze(-1)
                k2 = min(mask.size(1), gu.size(1))
                mask = gu[:, :k2]
                vol_t = torch.from_numpy(st["vc"]).float().to(
                    svc.device)[None, :, None]
            aus = svc.infer(xb, a.pitch, a.thr,
                            units_override=st["usm"],
                            feats=(st["fc"], vol_t, mask), phases=accs,
                            ratios=st["offc"],
                            enh_tail=(blk + cf + sola_search + last_delay
                                      + LA + int(0.1 * SR))
                            if a.enh_tail else 0)
            y = None
            stems = [] if (dmp is not None and a.dump_stems) else None
            for vi, au in enumerate(aus):    # SOLA and splice each part separately, mix afterwards
                # with every feature frozen, the overlap correlates at 0.96, so the
                # splice is only a phase shift, which is SOLA's actual job; with the
                # content identical the shift is stable and no longer chatters
                tw = au[-blk - cf - sola_search - last_delay - LA:
                        -last_delay - LA if last_delay + LA else None]
                ci = tw[None, None, :cf + sola_search]
                num = F.conv1d(ci, sola_bufs[vi][None, None, :])
                den = torch.sqrt(F.conv1d(ci ** 2, torch.ones(
                    1, 1, cf, device=svc.device)) + 1e-8)
                shift = int(torch.argmax(num[0, 0] / den[0, 0]))
                tw = tw[shift: shift + blk + cf].clone()
                tw[:cf] = phase_vocoder(sola_bufs[vi], tw[:cf],
                                        fade_out, fade_in)
                sola_bufs[vi] = tw[-cf:]
                w = tw[:-cf].cpu().numpy()
                if stems is not None:
                    stems.append(w.copy())
                y = w if y is None else y[:len(w)] + w[:len(y)]
            if a.agc:
                y = y / st["agc_g"]        # restore the level structure, as the offline input AGC does
            y = np.clip(y * a.gain, -1.0, 1.0).astype("float32")
            with qlock:
                out_q.append(y)
            if dmp is not None:
                dmp["mic"].append(chunk.copy())
                dmp["out"].append(y.copy())
                if stems is not None:
                    for vi, w2 in enumerate(stems):
                        dmp[f"stem{vi}"].append(w2)
                dmp["f0"].append(st["fc"][-nb:].copy())
                dmp["f0raw"].append(f0f[-nb:].copy())
                dmp["stk"].append(st["stk"][-nb:].copy())
                dmp["vol"].append(
                    vol_t[0, -nb:, 0].detach().cpu().numpy().copy())
                if sus_f and st.get("gt") is not None:
                    dmp["uv"].append(uvf[-nb:].astype(float).copy())
                    for kk in ("gt", "vc", "br"):
                        dmp[kk].append(
                            np.asarray(st[kk][-nb:], dtype=float).copy())
                for vi in range(len(svc.voices)):
                    if st["offc"][vi] is not None:
                        dmp[f"off{vi}"].append(st["offc"][vi][-nb:].copy())
                if a.agc:
                    dmp["agc"].append(np.full(nb, st["agc_g"]))
        except Exception as e:  # noqa: BLE001
            print("infer err:", e, flush=True)
        ms = (time.perf_counter() - t0) * 1000
        stats["n"] += 1
        stats["ms"] = stats["ms"][-40:] + [ms]
        if ms > blk / SR * 1000:
            stats["late"] += 1
        if stats["n"] % 20 == 0:
            m = np.array(stats["ms"][-20:])
            print(f"infer p50 {np.median(m):.0f}ms p95 "
                  f"{np.percentile(m, 95):.0f}ms / block {blk/SR*1000:.0f}ms"
                  f" | late {stats['late']} under {st2['under']}"
                  f" flags {st2['flags']}"
                  + (f" bridge {st['nbh']}holes/{st['nbf']}frames" if sus_f else ""),
                  flush=True)

    def flush_dump():
        """Write the dump, on exit and every 10 s, so a hard kill loses at most 10 s."""
        import soundfile as sf
        sf.write(a.dump + "_mic.wav", np.concatenate(list(dmp["mic"])), SR)
        sf.write(a.dump + "_angel.wav", np.concatenate(list(dmp["out"])), SR)
        np.savez(a.dump + "_feats.npz",
                 **{k: np.concatenate(list(v)) for k, v in list(dmp.items())
                    if k not in ("mic", "out") and len(v)})

    if a.replay:
        import soundfile as sf
        x, sr_in = sf.read(a.replay, dtype="float32", always_2d=True)
        assert sr_in == SR, f"replay file must be {SR}Hz, got {sr_in}"
        x = x[:, 0]
        nblk = (len(x) - blk) // blk + 1
        print(f"replay {a.replay}  {len(x)/SR:.0f}s / {nblk} blocks",
              flush=True)
        t0 = time.perf_counter()
        for i in range(nblk):
            step(x[i * blk:(i + 1) * blk])
            with qlock:
                out_q.clear()          # nothing is playing, so do not let the queue eat memory
            if ear is not None or REH is not None:
                brain_drain()
        el = time.perf_counter() - t0
        print(f"replay done {el:.0f}s（RTF {el/(nblk*blk/SR):.2f}）",
              flush=True)
        if dmp is not None and dmp["mic"]:
            flush_dump()
            print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz", flush=True)
        return

    print(f"stream on in={a.in_name!r} out={a.out_name!r} pitch {a.pitch:+.0f}"
          f" enhancer {'on' if a.enh else 'off'} (Ctrl-C to stop)", flush=True)
    wt = threading.Thread(target=worker, daemon=True)
    wt.start()
    if ear is not None or REH is not None:
        threading.Thread(target=brain_worker, daemon=True).start()

    import signal

    def _term(*_):
        raise KeyboardInterrupt      # a background process blocks SIGINT, so TERM takes the dump path too

    signal.signal(signal.SIGTERM, _term)
    with sd.Stream(samplerate=SR, blocksize=blk, channels=(1, 2),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency="high", callback=cb):
        try:
            tick_s = 0
            while True:
                time.sleep(1)
                tick_s += 1
                if dmp is not None and tick_s % 10 == 0 and dmp["mic"]:
                    flush_dump()
        except KeyboardInterrupt:
            st2["die"] = True
            if dmp is not None and dmp["mic"]:
                time.sleep(0.3)          # let the worker finish, so nothing is written half-way
                flush_dump()
                print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz",
                      flush=True)
            print("\nbye")


if __name__ == "__main__":
    main()
