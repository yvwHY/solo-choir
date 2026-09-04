"""spike_stream6.py — sliding-window streaming with the reflow voices
(2026-08-08; the 6.x port of spike_stream v26)

The whole streaming mechanism of spike_stream (CombSub plus the enhancer) is
kept unchanged — the sliding window, the frozen grid, SOLA, the direct model feed
of v19/v20, bridging, floor and enc-win — and only the voices are replaced:
  - three reflow voices (rectified flow plus NSF-HiFiGAN; the
    reflow-{harry,girl,bass1,sop3} models trained on 2026-08-07 and 08), which
    reach the realism ceiling at the 3.0-3.5 level
  - no enhancer, since reflow contains its own vocoder; and no initial_phase
    accounting, since NSF's phase cannot be controlled, so the joins rest
    entirely on SOLA plus the phase vocoder, as in gui_reflow
  - ODE sampling through --step (2 to 4 steps already saturate; swept
    2026-08-07)

Run (in the 6x venv):
  260724_ddsp_svc_6x/venv/bin/python spike_stream6.py --sustain 250 \
      --block 0.17 --extra 0.7 --floor-db -36 --ema-alpha 0.3 --enc-win 3
Ctrl-C to finish. Each block prints its inference time in ms; above the block
time the sound breaks up.
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP6X as _DDSP6X  # noqa: E402
sys.path.insert(0, str(_DDSP6X))
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from ddsp.vocoder import F0_Extractor, Volume_Extractor, \
    Units_Encoder  # noqa: E402
from ddsp.core import upsample  # noqa: E402
from reflow.vocoder import load_model_vocoder  # noqa: E402

SR = 44100
DDSP = str(_DDSP6X)
MAJ = [0, 2, 4, 5, 7, 9, 11]


def dia_from_note(note, steps, root):
    """A settled note to the semitone offset of diatonic steps."""
    rel = int(note) - root
    oc, pc = divmod(rel, 12)
    di = min(range(7), key=lambda i: min((MAJ[i] - pc) % 12,
                                         (pc - MAJ[i]) % 12))
    to, ti = divmod(di + steps, 7)
    return float(root + (oc + to) * 12 + MAJ[ti] - int(note))


class NoteTracker:
    """Note tracking with hysteresis, the core of v11's stability on sustained
    notes: a note changes only after a deviation over 0.65 semitones has held for
    3 frames, about 35 ms, so vibrato and micro-deviations crossing a boundary no
    longer chatter. It is causal frame by frame, so it agrees across windows."""

    def __init__(self):
        self.cur = None
        self.pend = 0
        self.hist = []
        self.streak = 0

    def feed(self, f0_hz):
        """Returns (notes, streaks), where a streak is the number of consecutive
        frames on the same note; 0 means the note has just changed or there is no
        sound. This gates the unit smoothing of v12: pin only when stable, and
        follow a note change at once."""
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
    """Frozen f0 in Hz per hop to a per-frame semitone offset, the in-key interval
    of a number of diatonic steps.
    Quantisation decides the interval only and never touches the curve, which is
    the philosophy of version D, the aesthetic verified on 2026-08-05.
    Deterministic: it depends only on a local neighbourhood of the frozen fc, the
    median over plus or minus 2 frames, so it agrees across windows and needs no
    separate cache."""
    v = f0c > 0
    m = np.where(v, 69 + 12 * np.log2(np.maximum(f0c, 1.0) / 440.0), 0.0)
    # A local median over 5 frames suppresses vibrato jitter and stops chattering
    # at a note boundary.
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
    """As in gui.py:15: make the phase continuous across a join, which is the
    main treatment for skipping."""
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
    """A trimmed gui.py SvcDDSP with multiple parts: the encoder, f0, volume and
    enhancer are shared, and there are N voices.
    voices: [(model_path, semitones, gain), ...]"""

    NGRID_BANK = 16384      # length of the frozen noise bank in frames, a loop of about 190 s

    def __init__(self, voices, device="mps", step=4, t_start=0.7,
                 f0_extractor="parselmouth", noise_grid=False):
        self.device = device
        self.voc_tail = 0                    # v32: vocode only the last few frames; 0 is off
        self.f0_extractor = f0_extractor
        self.step = step                     # ODE sampling steps; 2 to 4 already saturate
        self.t_start = t_start               # ODE starting point (see --t-start)
        self.voices = []
        self.vocoder = None
        import os
        cwd = os.getcwd()
        os.chdir(DDSP)          # vocoder.ckpt in config.yaml is a relative path
        try:
            self.spks = []      # v35 per-voice spk_id for multi-speaker models; the default is 1
            for v in voices:
                path, semi, gain = v[:3]
                spk = v[3] if len(v) > 3 else 1
                model, vocoder, args = load_model_vocoder(path, device=device)
                if self.vocoder is None:
                    self.vocoder = vocoder   # one NSF-HiFiGAN, shared
                self.voices.append((model, semi, gain))
                self.spks.append(
                    torch.LongTensor([[spk]]).to(device))
        finally:
            os.chdir(cwd)
        # The vocoder loads lazily, calling torch.load only on the forward pass,
        # and stores a relative path, so it is made absolute here.
        vp = getattr(self.vocoder, "vocoder", self.vocoder)
        if hasattr(vp, "model_path") and not os.path.isabs(vp.model_path):
            vp.model_path = os.path.join(DDSP, vp.model_path)
        self.args = args                      # the same pipeline, so the data parameters are shared
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            f"{DDSP}/{self.args.data.encoder_ckpt}",
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size, device=device)
        self.enhancer = None                 # reflow contains its own vocoder, so this is not needed
        self.vol_ex = Volume_Extractor(
            self.args.data.block_size,
            int(getattr(self.args.data, "volume_smooth_size", 1024)))
        self.spk = torch.LongTensor([[1]]).to(device)
        # v33 frozen noise grid: reflow's starting noise is now looked up by
        # absolute frame position, one table per part from a fixed seed, so
        # overlapping windows re-rendering the same frame get the same noise. That
        # removes the pitch-path jitter caused by an independent randn per window
        # (measured 2026-08-10: live std 31 cents against 9 for the reference).
        # None turns it off and restores the randn behaviour, as does calling
        # infer without frame0.
        self.ngrid = None
        if noise_grid:
            g = torch.Generator().manual_seed(20260810)
            M = self.vocoder.dimension      # Vocoder.__init__ stores this as an int attribute
            self.ngrid = [torch.randn(self.NGRID_BANK, M, generator=g)
                          .to(device) for _ in self.voices]
        # v29: F0_Extractor used to be rebuilt for every block, which nobody
        # noticed while parselmouth was cheap. rmvpe is a 181 MB network and must
        # be built once. Its weight path is relative, so the working directory has
        # to be DDSP at construction time.
        if self.f0_extractor == "rmvpe":
            # The vendored RMVPE calls torch.load without map_location and the
            # checkpoint was saved on a CUDA machine, so it fails outright on a
            # Mac. It is built here and put into the F0_KERNEL cache, which
            # F0_Extractor only fills itself when the key is missing, so the
            # vendored repository need not be changed.
            from ddsp.vocoder import F0_KERNEL
            if "rmvpe" not in F0_KERNEL:
                from encoder.rmvpe import RMVPE
                # 2026-08-08: pretrain/rmvpe/model.pt is the older architecture
                # and has no unet.tf.*, while the vendored E2E0 needs those 50
                # layers, so load_state_dict(strict=False) would silently leave
                # them random and the f0 would be rubbish. Fail loudly instead.
                _sd = torch.load(f"{DDSP}/pretrain/rmvpe/model.pt",
                                 map_location="cpu")
                if not any(".tf." in k for k in _sd):
                    raise SystemExit(
                        "pretrain/rmvpe/model.pt does not match the vendored RMVPE "
                        "architecture: unet.tf.* is missing, so 50 layers would be "
                        "random weights. Using rmvpe needs weights of the matching "
                        "version; for now use --f0 parselmouth or fcpe.")
                _tl = torch.load

                def _shim(*ar, **kw):
                    # Two mismatches: the checkpoint was saved on CUDA and needs
                    # map_location, and these weights are a raw state_dict in RVC
                    # format while the vendored code reads ckpt['model'].
                    o = _tl(*ar, **{**kw, "map_location": "cpu"})
                    return o if isinstance(o, dict) and "model" in o \
                        else {"model": o}

                torch.load = _shim
                try:
                    F0_KERNEL["rmvpe"] = RMVPE(
                        f"{DDSP}/pretrain/rmvpe/model.pt", hop_length=160)
                finally:
                    torch.load = _tl
        if self.f0_extractor == "fcpe":
            # The vendored code hard-codes 'cuda' if available else 'cpu', which
            # falls to CPU on a Mac (RTF 2.13, so not real-time). It is likewise
            # pre-filled into the cache and pushed to MPS.
            from ddsp.vocoder import F0_KERNEL
            if "fcpe" not in F0_KERNEL:
                from torchfcpe import spawn_bundled_infer_model
                F0_KERNEL["fcpe"] = spawn_bundled_infer_model(device=device)
        cwd = os.getcwd()
        os.chdir(DDSP)
        try:
            self.pe = F0_Extractor(self.f0_extractor, SR,
                                   self.args.data.block_size, 65.0, 800.0)
            if self.f0_extractor == "fcpe":
                self.pe.device_fcpe = device
        finally:
            os.chdir(cwd)

    def prep(self, audio, threhold=-60.0, want_uv=False):
        """Returns (f0_np, vol_t, mask[, uv]). f0 stays as numpy so the caller can
        place it on the frozen grid.
        want_uv also returns parselmouth's real voicing, the frames where f0 == 0.
        That is the criterion for bridging in v16: with a throat microphone the
        level of a consonant never falls below the -60 dB threshold, so the vocal
        fold decision is the true version of vm in variant E.
        The interpolation section reproduces vocoder.py:142-146, so f0 is
        byte-identical to uv_interp=True."""
        hop = self.args.data.block_size
        pe = self.pe                       # v29: built once; it used to be rebuilt per block
        # The resampling kernel of the neural extractors (fcpe, crepe, rmvpe) is
        # float32, while parselmouth still takes float64, so the default path is
        # bit-identical.
        au = audio if self.f0_extractor in ("parselmouth", "dio", "harvest") \
            else audio.astype("float32")
        if want_uv:
            f0_np = pe.extract(au, uv_interp=False, device=self.device)
            uv = f0_np == 0
            if len(f0_np[~uv]) > 0:
                f0_np[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0],
                                      f0_np[~uv])
            f0_np[f0_np < 65.0] = 65.0
        else:
            f0_np = pe.extract(au, uv_interp=True, device=self.device)
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
              enh_tail=0, frame0=None):
        """One output per part: shared units, f0 and volume, with each part's own
        transposition and gain.
        feats is (f0_np, vol_t, mask) supplied from outside so it is not
        recomputed, which the frozen grid needs.
        phases is the absolute phase per part in radians, keeping the comb source
        in phase across windows, which is the real fix for the skipping of v6.
        ratios is a per-frame array of semitone offsets per part, diatonic; None
        uses the fixed semi.
        frame0 is the absolute frame position of this window's first frame, used to
        look up the frozen noise grid of v33; None, or an unbuilt grid, restores
        the randn behaviour."""
        hop = self.args.data.block_size
        f0_np, vol_t, mask = feats if feats is not None \
            else self.prep(audio, threhold)
        f0 = torch.from_numpy(f0_np).float().to(self.device)[None, :, None]
        units = units_override
        if units is None:
            units = self.encode(audio)
        n = min(units.size(1), f0.size(1), vol_t.size(1))
        # The reflow voices accept phases and enh_tail but do not use them: there
        # is no phase accounting and no enhancer.
        ws = []
        with torch.no_grad():
            for vi, (model, semi, gain) in enumerate(self.voices):
                if ratios is not None and ratios[vi] is not None:
                    r = torch.from_numpy(
                        ratios[vi][:n]).float().to(self.device)[None, :, None]
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + r) / 12.0)
                else:
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + semi) / 12.0)
                # v32: the mel and the vocoder are called separately. At
                # voc_tail = 0 this is equivalent to the original
                # return_wav=True, since the model internally calls
                # vocoder.infer(mel, f0[-T:]), which makes regime A provable.
                # Above 0, only the tail plus a margin is vocoded.
                # The window is 0.71 s while SOLA takes only the last 0.31 s, so
                # 59% of the vocoder work is thrown away.
                # Cost is about 3.8 ms plus 0.29 ms per frame (measured, see
                # scratchpad/bench_vocoder_tail).
                # Not bit-identical: NSF produces its excitation phase by
                # integrating f0, so changing the starting point changes the
                # phase (the "NSF phase cannot be controlled" of the docstring).
                # That is regime B, judged by ear.
                nz = None
                if self.ngrid is not None and frame0 is not None:
                    # 8 frames of margin, since the mel may have more frames than
                    # n; the reflow side trims to the real T
                    idx = (frame0 + torch.arange(n + 8)) % self.NGRID_BANK
                    # [T,M] to [1,1,M,T], the x shape reflow expects
                    nz = self.ngrid[vi][idx].t()[None, None]
                mel = model(units[:, :n], fv, vol_t[:, :n],
                            spk_id=self.spks[vi], vocoder=self.vocoder,
                            infer=True, return_wav=False,
                            infer_step=self.step, method="euler",
                            t_start=self.t_start, use_tqdm=False,
                            init_noise=nz)
                f0m = fv[:, -mel.size(1):]
                if self.voc_tail:
                    kf = min(mel.size(1), self.voc_tail)
                    out = self.vocoder.infer(
                        mel[:, -kf:], f0m[:, -kf:]).reshape(1, -1)
                    # Pad with zeros at the front back to the full window length,
                    # so nothing downstream has to change: the mask and SOLA's
                    # negative slices index as before, and the padded region is
                    # thrown away by the tail slice anyway.
                    pad = mel.size(1) * hop - out.size(1)
                    if pad > 0:
                        out = F.pad(out, (pad, 0))
                else:
                    out = self.vocoder.infer(mel, f0m).reshape(1, -1)
                k = min(out.size(1), mask.size(1), n * hop)
                out = out[:, :k] * mask[:, :k]
                ws.append(out.squeeze() * gain)
        return ws
        # Return each part separately so the joins align individually; SOLA over
        # the mix cannot align two different periods.

    def encode(self, audio):
        au = torch.from_numpy(audio).float()[None].to(self.device)
        return self.units_encoder.encode(au, SR, self.args.data.block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=f"{DDSP}/exp/reflow-harry-run1/model_14000.pt",
                    help="the model used by --solo mode")
    ap.add_argument("--solo", action="store_true",
                    help="single-part identity, for checking quality and latency; the default is harmony, with S = Soprano-3 at +12 and B = Bass-1 at -12 following along")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--step", type=int, default=4,
                    help="reflow ODE sampling steps (swept 2026-08-07: 2 steps scored 3.27 and 4 steps saturated at 3.30; more steps cost more computation)")
    ap.add_argument("--t-start", type=float, default=0.7,
                    help="the reflow ODE starting point, 0 to 1; the default of 0.7 is the value hard-coded before 2026-08-08. In reflow.py:80 the initial mel is "
                         "t_start times the DDSP mel plus (1 - t_start) times Gaussian noise, with dt = (1 - t_start) / step. Higher injects less "
                         "noise and stays closer to the DDSP base; lower moves towards the generative end. All three voice configs use 0.0 with "
                         "50 steps, which is the offline reference working point, so streaming and the reference already differ along this axis")
    ap.add_argument("--f0", default="parselmouth",
                    choices=["parselmouth", "rmvpe", "fcpe", "harvest",
                             "crepe", "dio"],
                    help="the voicing and pitch detector; parselmouth by default, which is the original behaviour. Diagnosis of 2026-08-08: the whole "
                         "sustain, bridge and decay apparatus exists to clean up after unreliable voicing. In diag12, 23 of the bridges were "
                         "cases where they were singing at full level and parselmouth called 174 ms of it silent. The rmvpe weights are already "
                         "in pretrain/rmvpe and are far steadier on singing than an autocorrelation method")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the reflow sampling noise. The randn in reflow.py:80 has no seed, so the same input with the same flags differs between runs. "
                         "Without a seed an offline A/B has no resolution: measured, even a null intervention produced a difference of plus or "
                         "minus 6%")
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--out-name", default="AI-Micro")
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--thr", type=float, default=-60.0,
                    help="volume threshold in dB; the settled recipe is -60. The -45 of gui.py is too high for a throat microphone and chops the output into pieces")
    ap.add_argument("--key", type=int, default=0,
                    help="pitch class of the major tonic (0 = C to 11 = B), used by the diatonic mapping")
    ap.add_argument("--no-brain", dest="brain", action="store_false",
                    help="turn the harmony model off and fall back to the fixed diatonic mapping of v12")
    ap.add_argument("--octave", type=int, default=12,
                    help="shift in model space, as in respond2 (the 16:00 session used 12)")
    ap.add_argument("--block", type=float, default=0.10,
                    help="seconds per block, which is the dominant latency term; the inference time in ms must be below it (bench: a 1.0 s window had an "
                         "inference p95 of 39 ms, so 0.10 leaves 2.5 times the margin)")
    ap.add_argument("--crossfade", type=float, default=0.04)
    ap.add_argument("--extra", type=float, default=1.0,
                    help="seconds of context prepended, trading quality against computation")
    ap.add_argument("--sustain", type=float, default=0.0,
                    help="v15 phrase bridging in ms: unvoiced gaps up to this length are sung through on a sustained vowel (the live form of variant E's sustain; "
                         "0 is off, the old path)")
    ap.add_argument("--sop-db", type=float, default=0.0,
                    help="gain of the S part in dB (-99 mutes it). Diagnosis of 2026-08-08: sop3 at 20000 steps has a high-to-low frequency ratio of 0.016 while "
                         "singing, three times that of girl and bass, which is the source of the whispering breathiness; it is undertrained, only "
                         "20000 steps on 710 clips. While it is being trained further, -99 turns it off so the other three can be heard")
    ap.add_argument("--bridge-decay", type=float, default=0.0,
                    help="v27 bridge decay time constant in ms (0 is off, the old behaviour): during a sustain the gate follows exp(-t/tau), so a consonant does "
                         "not break the note while a quiet breathy tail is not dragged out. This treats the gasp after every phrase; 80 is suggested")
    ap.add_argument("--bridge-hold", type=float, default=0.0,
                    help="v29 sustain freeze threshold, as a proportion of the envelope (0 is off, the old behaviour): while the level inside a gap is still above "
                         "the envelope times this value, the clock of the decay and sustain budget does not advance, so the gate holds. This treats "
                         "being cut off mid-phrase. Diagnosis (diag12, 2026-08-08): in 13 of 23 bridges the level inside the gap had already returned "
                         "to a quarter of the envelope or more, but v17 measured gap duration by unvoiced frames alone and could not leave, so the "
                         "gate decayed to zero. The gate is not reopened, which would let interpolated false f0 in, the low bass hum v17 exists to "
                         "prevent; it simply stops falling. 0.5 is suggested")
    ap.add_argument("--units-mature", type=int, default=0,
                    help="v36 mature refresh of transition units: within the last M frames, transition frames where alpha == 1 are overwritten each window with this "
                         "window's fresh values, which have more right context; pinned frames are untouched (0 is off, the old behaviour). 21-42, "
                         "that is 1-2 blocks, is suggested")
    ap.add_argument("--aah", action="store_true",
                    help="v37, the parts sing 'ah' rather than carrying the words (2026-08-11): the units are replaced by the take-17 vowel loop, aligned to the "
                         "absolute frame grid, while f0, volume and the harmony still follow them live. The encode and unit-freezing chain is "
                         "bypassed, so the RTF falls as well. Off is bit-identical to the old behaviour")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="v37 slow-layer texture (stage 0, H): the upper and lower parts sing a phrase-level chord pad instead, pinned to the absolute pitch of the "
                         "anchor note plus interval at the moment the chord changes, rather than following the melody line. A chord must hold for at "
                         "least this many seconds before it may change. The soprano still follows the melody at +12, as a shadow part. 0 is off, the "
                         "old behaviour")
    ap.add_argument("--swell", type=float, default=0.0,
                    help="v37 soft attack, part of the distance layer: rate-limit the rise of the volume envelope with a time constant in ms, giving a choral swell, "
                         "while the fall is untouched so endings stay natural. Cached on the absolute frame grid so it agrees across windows. 0 is "
                         "off, the old behaviour; 120-200 is suggested")
    ap.add_argument("--wet", type=float, default=0.0,
                    help="v37 distance layer: how much of the parts is sent into the space, through the synthesised IR of reverb.py convolved with streaming "
                         "overlap-add. 0 is off, the old behaviour; 0.3-0.4 is suggested, since respond2's 0.20, chosen by ear, was for a close-up "
                         "context")
    ap.add_argument("--seam-amp", action="store_true",
                    help="v34: join with a plain amplitude crossfade instead (off by default, which uses the phase vocoder). Use with --sola-cont; see the comments "
                         "in the join section")
    ap.add_argument("--f0-mature", type=int, default=0,
                    help="v34 deferred freezing: the f0 of the last M frames is refreshed each window with the latest estimate (0 is off, freezing on arrival, the "
                         "old behaviour). An estimate at the end of a window lacks right context and is 37.6 cents out over a note change. Pair it "
                         "with --lookahead M, which plays frames M frames later, so the estimate has matured by the time it sounds. The cost is "
                         "LA times 11.6 ms of latency")
    ap.add_argument("--sola-cont", type=float, default=0.0,
                    help="v34 SOLA shift continuity bias lambda (0 is off, the old behaviour): score times (1 - lambda * |s - prev| / search) picks the nearest "
                         "good-enough peak to the previous block. This treats unstable pitch: the shift jumps between correlation peaks a period "
                         "apart (p90 7 ms), and over a note change that timebase jitter times the slope gives 50-74 cents of deviation on a steep "
                         "segment. 0.2-0.4 is suggested; too large locks onto the old shift and joins wrongly when the content really changes")
    ap.add_argument("--noise-grid", action="store_true",
                    help="v33 frozen noise grid (off by default, the old behaviour): reflow's starting noise is looked up by absolute frame position in a "
                         "fixed-seed noise bank, one per part over a 190 s loop, so overlapping windows re-rendering the same frame use the same "
                         "noise. This treats unstable pitch: with an independent randn per window, one sustained note changes pitch path every "
                         "244 ms (measured 2026-08-10: live deviates from the commanded f0 with an std of 31 cents and 12% of frames over 30, "
                         "against 9 cents and 1% for the same voice in the offline reference). This is regime B, since it changes the live output; "
                         "with the default off the randn path is bit-identical, so regime A remains provable")
    ap.add_argument("--voc-tail", type=int, default=0,
                    help="v32: vocode only the tail plus this margin, in frames (0 is off, the old behaviour). The window is 0.71 s while SOLA takes only the last "
                         "0.31 s, so 59% of the vocoder work is discarded, and the vocoder is the most expensive stage in the system, 55% of a "
                         "block. On the bench, three voices together save about 20 ms per block, a 13% reduction. **Not bit-identical**: NSF derives "
                         "its excitation phase by integrating f0, so changing the starting point changes the phase, and a margin sweep confirmed the "
                         "difference does not fall with the margin. That is regime B and needs a listening test. But the phase origin of adjacent "
                         "blocks already disagrees, which is exactly what SOLA and the phase vocoder exist to stitch, so this is not a new class of "
                         "problem. 4 is suggested; listen for the joins becoming rougher")
    ap.add_argument("--drop", type=int, nargs="*", default=[],
                    help="drop part N (0 = soprano, 1 = the friend's voice in the middle, 2 = bass; empty drops nothing, the old behaviour). The difference from "
                         "--sop-db is that the latter only zeroes the gain while the voice still runs inference; this really does not load or compute "
                         "it. Measured, each voice costs about 41 ms, of which the vocoder is 28, which makes it the single largest lever on latency")
    ap.add_argument("--prof", action="store_true",
                    help="stage timing (0 is off, the old behaviour): split each block's milliseconds into six stages, prep, bridge and model, encode, units, "
                         "inference and SOLA, and print a budget table on exit. MPS is an asynchronous queue, so mps.synchronize() has to be inserted "
                         "between stages to measure real values, which itself raises the total slightly. **The absolute RTF under --prof cannot be "
                         "compared with a run without it**; look only at the proportions")
    ap.add_argument("--bridge-units", type=float, default=0.0,
                    help="v30 threshold for unfreezing units during a bridge, as a proportion of the envelope (0 is off, the old behaviour): while the level inside "
                         "a gap is still above the envelope times this value, the units are not frozen and follow the EMA and streak as usual, so "
                         "consonants keep up. This treats consonants such as /t/ not coming out. Diagnosis of 2026-08-08: --bridge-hold pushed the "
                         "share of consonants falling inside the frozen-unit region from 20.1% to 29.3% and made them louder. It is symmetrical with "
                         "what --bridge-hold does to the gate: that one keeps the gate from falling, this one keeps the units from freezing. It also "
                         "moves the trigger of v18's gap rewind from a false gap, a voicing error, to a real fall in level. 0.5 is suggested")
    ap.add_argument("--release", type=float, default=120.0,
                    help="a real rest beyond the sustain: milliseconds of fade-out, replacing a hard gate cut")
    ap.add_argument("--floor-db", type=float, default=0.0,
                    help="v21 volume floor in dBFS (0 is off, the old path): frames whose volume falls below this are forced to count as unvoiced, so the gate can "
                         "close and the model takes a rest. This treats breath and friction on a throat microphone being called voiced by "
                         "parselmouth (diag4: 61% of the frames with the gate open were rubbish whose microphone peak was under a tenth of the "
                         "singing, with a median f0 of 562 Hz, which is the raw material of the strange sounds; -36 measured killed 100% of the "
                         "voiced rubbish with no false positives)")
    ap.add_argument("--dump", default="",
                    help="on exit, write <dump>_{mic,angel}.wav plus _feats.npz, holding per-frame f0, unvoiced flags, gate, bridge, volume and interval, for "
                         "diagnosis")
    ap.add_argument("--enc-win", type=float, default=0.0,
                    help="v26 HuBERT encoding window in seconds (0 is off, matching the render window): the encoder takes a longer stretch of past audio, buying "
                         "diction with left context at no latency cost. The dose response in diag5: 0.7 s gave 0.524, 1.5 s gave 0.694 and 3 s gave "
                         "0.788, against 0.813 for the reference, at a cost rising from 15 to 40 ms. 3.0 is suggested")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="v25 lookahead in frames (roadmap item 1): the render and take windows move back N frames as a whole, giving HuBERT, the EMA and the note "
                         "tracker N more frames of right context there, which sharpens the diction, at the cost of N times 11.6 ms of output "
                         "latency. 0 is off, the original behaviour. Spending what enh-tail saves on this buys better diction at unchanged latency")
    ap.add_argument("--rehearse", default="",
                    help="v24 rehearsal score mode (roadmap item 7, thesis level): a notes.json holding lead, upper and lower as absolute notes in microphone space "
                         "on a 186 ms tick grid, produced from a dump by scratchpad/make_score.py. ScoreTracker v3 aligns to their lead and the parts "
                         "sing the score, so the harmony is guaranteed by the score and the whole model layer, including the token snapping, is "
                         "bypassed")
    ap.add_argument("--reh-rate", type=float, default=1.0,
                    help="speed prior r-hat for rehearsal alignment, the same control as in live_v3; a fixed value beats estimating it")
    ap.add_argument("--enh-tail", action="store_true",
                    help="v23: the enhancer computes only the tail SOLA takes plus a 0.1 s margin, saving of the order of 28 ms; the quality of the joins is judged "
                         "by the replay metric and by ear")
    ap.add_argument("--dump-stems", action="store_true",
                    help="add a per-part stem waveform to the dump, for attribution; the file is three times the size")
    ap.add_argument("--ema-alpha", type=float, default=0.12,
                    help="v12 EMA smoothing coefficient for the units while a note is stable; higher makes the vowel follow faster and the diction sharper, lower "
                         "pins it harder. 0.12 is the original value")
    ap.add_argument("--ema-streak", type=int, default=6,
                    help="how many stable frames before the EMA starts pinning (6, about 70 ms, is the original value)")
    ap.add_argument("--agc", action="store_true",
                    help="v22 streaming AGC, the causal form of direct_mouth's input_agc: raise the voiced level towards TRAIN_REF_RMS 0.0566 before the model and "
                         "divide it back out of the output, so the level structure is unchanged. This treats the level falling outside the training "
                         "distribution and degrading the conversion"
                         "(the prime suspect for smeared, weak diction; measured 2026-07-30, it worsens monotonically from 14.3 to 18.3 dB). "
                         "Off by default, the old path")
    ap.add_argument("--replay", default="",
                    help="v22 offline replay: read a wav and run it block by block through the same step and model path, without opening an audio device. "
                         "The model is drained synchronously each block, so it lags by one block exactly as live does. With --dump it "
                         "produces something comparable to a live dump, which is the foundation for iterating on quality without singing again")
    a = ap.parse_args()

    print("loading models...", flush=True)
    ear = kt = None
    REH = None
    if a.rehearse:
        # v24: score mode bypasses the model. The three voices keep the same
        # configuration; the note line comes from score tracking.
        import json
        from rehearse_ab import ScoreTracker
        d = json.load(open(a.rehearse))
        REH = {"upper": d["upper"], "lower": d["lower"],
               "tracker": ScoreTracker(d["lead"], mode="v3",
                                       rate0=a.reh_rate)}
        print(f"[rehearse] score of {len(d['lead'])} ticks", flush=True)
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.8),
                  # v35 (2026-08-11): the friend's voice is replaced by
                  # M4Singer Alto-6, trained jointly on three singers for 40k
                  # steps, spk2. On the replay bench against the friend's voice it
                  # is 8-10 dB cleaner of haze with equal tuning and diction, and
                  # Alto-6 was chosen by ear from three voices. The friend's model
                  # at 20000 steps is kept on file for a rollback.
                  (f"{DDSP}/exp/reflow-alto3/model_40000.pt", 12.0, 0.85, 2),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]
    elif a.solo:
        voices = [(a.model, 0.0, 1.0)]
        dia_steps = [None]
        vmode = [None]
    elif a.brain:
        # v13 harmony model: the EarV3 tick stream, which is a live machine
        # already. The alto voice sings the upper line and Bass-1 the lower, with
        # indep, stab and legato all inherited, and S doubles the melody an octave
        # up. The model's output is converted into an interval relative to their
        # pitch before entering the cache, so the interval stays constant, which
        # is the philosophy of variant D.
        import respond2 as R2
        from live_v3 import EarV3
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.8),
                  # v35 (2026-08-11): the friend's voice is replaced by
                  # M4Singer Alto-6, trained jointly on three singers for 40k
                  # steps, spk2. On the replay bench against the friend's voice it
                  # is 8-10 dB cleaner of haze with equal tuning and diction, and
                  # Alto-6 was chosen by ear from three voices. The friend's model
                  # at 20000 steps is kept on file for a rollback.
                  (f"{DDSP}/exp/reflow-alto3/model_40000.pt", 12.0, 0.85, 2),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]        # soprano is fixed at +12; the alto and
                                          # bass parts follow the model
        ear = EarV3(indep=0.15, stab=1, key=0)
        ear.v2t.shift = a.octave
        ear.k_shift = 0        # the key lock is maintained externally by
                               # KeyTracker (the direct feed of v14)
        kt = R2.KeyTracker()
    else:
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/reflow-harry-run1/model_14000.pt",
                   -3.0, 0.9),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
        dia_steps = [7, -2, -7]
        vmode = [None, None, None]
    if a.sop_db and voices:
        p0, s0, g0 = voices[0]
        voices[0] = (p0, s0, 0.0 if a.sop_db <= -90 else
                     g0 * 10 ** (a.sop_db / 20.0))
    if a.drop:
        # v31: really remove the voice. --sop-db only zeroes the gain, so
        # inference still runs and nothing is saved. All three lists, voices,
        # dia_steps and vmode, must be filtered together to keep the indices
        # aligned.
        keep = [i for i in range(len(voices)) if i not in a.drop]
        voices = [voices[i] for i in keep]
        dia_steps = [dia_steps[i] for i in keep]
        vmode = [vmode[i] for i in keep]
        print(f"drop {sorted(a.drop)} -> {len(voices)} parts left", flush=True)
    if a.seed is not None:
        torch.manual_seed(a.seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(a.seed)
    svc = Svc(voices, step=a.step, t_start=a.t_start, f0_extractor=a.f0,
              noise_grid=a.noise_grid)
    HOP = svc.args.data.block_size
    blk = max(1, round(a.block * SR / HOP)) * HOP     # aligned to 512 so the grid does not drift
    cf = int(a.crossfade * SR)
    sola_search = int(0.01 * SR)
    last_delay = int(0.02 * SR)
    LA = a.lookahead * HOP              # v25 lookahead in samples; 0 is off
    input_frame = ((max(int(a.extra * SR),
                        blk + cf + sola_search + 2 * last_delay + LA)
                    // HOP + 1) * HOP)
    if a.voc_tail:
        # The tail SOLA actually touches, in samples converted to frames and
        # rounded up, plus the margin given by the user.
        used = blk + cf + sola_search + last_delay + LA
        svc.voc_tail = -(-used // HOP) + a.voc_tail
        print(f"voc-tail: vocoding the last {svc.voc_tail} of {input_frame // HOP} frames"
              f" (margin {a.voc_tail})", flush=True)
    nb = blk // HOP
    buf = np.zeros(input_frame, dtype="float32")
    ENC_WIN = (max(int(a.enc_win * SR) // HOP * HOP, input_frame)
               if a.enc_win else 0)
    buf3 = np.zeros(ENC_WIN, dtype="float32") if ENC_WIN else None
    VOW = None
    if a.aah:
        # v37 vowel bank, as in score_sing and score_live: about 0.9 s of units
        # from the middle of the longest voiced span of take 17, looped, which is
        # the parts' "ah". Built once, then looked up per window by absolute frame
        # position.
        import soundfile as _sf
        mic0, _ = _sf.read("scratchpad/take17.wav", dtype="float64",
                           always_2d=True)
        mic0 = mic0[:, 0]
        _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
        runs, _i = [], 0
        while _i < len(uv0):
            if not uv0[_i]:
                _j = _i
                while _j < len(uv0) and not uv0[_j]:
                    _j += 1
                runs.append((_i, _j))
                _i = _j
            else:
                _i += 1
        _a0, _b0 = max(runs, key=lambda r: r[1] - r[0])
        _mid = (_a0 + _b0) // 2
        with torch.no_grad():
            VOW = svc.encode(
                mic0[max(0, (_mid - 40) * HOP):(_mid + 40) * HOP])[:, 8:-8]
        print(f"[ah] vowel bank of {VOW.size(1)} frames from take 17; the parts do not carry the words",
              flush=True)
    sola_bufs = [torch.zeros(cf, device=svc.device) for _ in svc.voices]
    prev_shift = [0] * len(svc.voices)   # for v34 --sola-cont: the previous block's shift
    RVIR, RVT = None, None
    if a.wet > 0:
        # v37 distance layer: the synthesised IR from reverb.py, the one that
        # killed the metallic quality, convolved by streaming overlap-add. Only
        # the parts are wetted; y is the parts alone, and the dry voice never
        # comes through here.
        import reverb as rv
        from scipy.signal import fftconvolve as _fftc
        RVIR = rv.make_ir(2.4, trim_db=35.0)
        RVT = [np.zeros(len(RVIR) - 1)]
        print(f"[wet] IR {len(RVIR)/SR:.1f}s, send {a.wet}", flush=True)
    fade_in = torch.sin(np.pi * torch.arange(0, 1, 1 / cf,
                                             device=svc.device) / 2) ** 2
    fade_out = 1 - fade_in
    stats = {"n": 0, "ms": [], "late": 0}
    PROF = a.prof
    prof = {}       # a plain dict; the defaultdict import is further down
    if PROF:
        _sync = "mps" in str(svc.device)    # device is a string, not a torch.device

        def pt(key, t):
            """Stage timing. MPS is an asynchronous queue: without synchronising,
            all that is measured is the time to enqueue the work, a few tens of
            microseconds, and the real cost piles up at the next synchronisation
            point."""
            if _sync:
                torch.mps.synchronize()
            now = time.perf_counter()
            prof[key] = prof.get(key, 0.0) + (now - t) * 1000.0
            return now
    else:
        def pt(key, t):
            return t
    st = {"uc": None, "fc": None, "offc": None, "tb": np.zeros(0),
          "cur": {"up": 12.0, "lo": -12.0}, "ktn": 0,
          "hole": 10 ** 9, "dk": 10 ** 9, "env": 0.0, "nbh": 0, "nbf": 0,
          "f0q": [], "hf0": 0.0, "ufifo": [], "inbr": False,
          "ivp": {"up": None, "lo": None}, "klock": False,
          "agc_g": 1.0, "agc_ema": None,
          "padt": {"up": 0.0, "lo": 0.0}, "pada": None,   # v37 --pad
          "abs": 0}     # v33: total frames streamed so far, the absolute frame position
                     # into the frozen noise grid
    accs = [0.0] * len(svc.voices)      # per-part absolute phase accumulator, in radians
    tracker = NoteTracker()             # note tracking with hysteresis (v11 stability on sustained notes)
    from collections import defaultdict
    dmp = defaultdict(list) if a.dump else None
    sus_f = int(round(a.sustain / 1000.0 * SR / HOP))   # bridging ceiling, in frames
    rel_f = max(1, int(round(a.release / 1000.0 * SR / HOP)))
    bdk = a.bridge_decay / 1000.0            # v27 bridge decay time constant in seconds; 0 is off
    bhold = a.bridge_hold                    # v29 sustain freeze threshold; 0 is off
    buh = a.bridge_units                     # v30 unit unfreeze threshold; 0 is off
    ENV_DK = float(np.exp(-(HOP / SR) / 0.15))   # volume inertia, time constant about 150 ms

    def bridge_feed(vol_np, uv, f0_np):
        """v18 phrase bridging, causal and frame by frame; the history is in the
        worklog sections O and P. Returns (gate, vol, bridged, f0_hold, unfreeze),
        and only newly arrived frames are fed in, so the result can be frozen into
        the grid and agrees across windows.
        unfreeze is v30: within a bridged frame, the level still being present
        means they really are still singing and parselmouth has misjudged the
        vocal folds, so the units downstream should not be frozen.
        Entering a gap requires both that the folds are called unvoiced and that
        the level has fallen below a quarter of the envelope, -12 dB, which blocks
        the false unvoiced frames at the end of a sliding window that caused the
        trouble in v16. The **duration** of a gap is measured by unvoiced frames
        alone: the trap of v17 was that over a long rest the envelope decays to
        the noise floor, the relative criterion fails, the gate reopens, and the
        bass sings a low hum on interpolated false f0 (review F1).
        The hold philosophy, the lesson of "all three were wrong" in v17: the
        frame at the mouth of the gap is already contaminated by the consonant,
        with f0 guessed at the end of the window and dirty units, so the mouth of
        the gap cannot be held. f0 is taken from a safe frame about 45 ms earlier,
        the oldest value in a 4-frame FIFO, which is free retrospective lookahead,
        and the volume is taken from the envelope, which follows the attack
        instantly and releases with a time constant of about 150 ms.
        A short gap, up to sus_f, holds the gate at 1; a real rest holds it fully
        and then fades out over rel_f frames with a cosine."""
        g = np.zeros(len(vol_np))
        v = np.copy(vol_np)
        b = np.zeros(len(vol_np), dtype=bool)
        h = np.zeros(len(vol_np), dtype=bool)
        vf0 = np.copy(f0_np)
        for j, x in enumerate(vol_np):
            st["env"] = max(float(x), st["env"] * ENV_DK)
            if not (uv[j] and (st["hole"] > 0 or x < st["env"] * 0.25)):
                st["hole"] = 0
                st["dk"] = 0
                g[j] = 1.0
                st["f0q"] = (st["f0q"] + [float(f0_np[j])])[-4:]
            else:
                st["hole"] += 1
                if st["hole"] == 1:
                    st["hf0"] = st["f0q"][0] if st["f0q"] else float(f0_np[j])
                    st["dk"] = 0
                # v29: the level is still present, so they are still singing and
                # parselmouth has misjudged the folds; the decay clock does not
                # advance. At bhold = 0, dk increases every frame, which is
                # frame-for-frame identical to the old behaviour.
                if not (bhold > 0 and x > st["env"] * bhold):
                    st["dk"] += 1
                # v30: the criterion for unfreezing the units, on its own
                # threshold so it can be A/B tested separately from the gate's
                # bhold.
                hv = buh > 0 and x > st["env"] * buh
                if st["dk"] <= sus_f:
                    # v27 (2026-08-08, "it sounds like a gasp after every
                    # phrase"): during a bridge the gate decays exponentially
                    # rather than holding at 1.0 throughout. Diagnosis: the
                    # high-to-low frequency ratio over bridged and fading spans is
                    # 0.040 against 0.011 while singing, 3.6 times higher, because
                    # at a low level reflow renders a frozen vowel as breath; the
                    # harmonics fall 5.5 times while the noise falls only 1.5.
                    # Within the first 50 ms or so of the time constant it is still
                    # nearly fully open, which preserves the original intent of not
                    # breaking the note on a consonant.
                    gd = (1.0 if bdk <= 0 else
                          float(np.exp(-(st["dk"] - 1) * (HOP / SR) / bdk)))
                    g[j], v[j], b[j] = gd, st["env"] * gd, True
                    h[j] = hv
                    vf0[j] = st["hf0"]
                    st["nbf"] += 1
                    if st["hole"] == 1:
                        st["nbh"] += 1
                elif st["dk"] <= sus_f + rel_f:
                    r = (st["dk"] - sus_f) / rel_f
                    g[j] = float(np.cos(r * np.pi / 2) ** 2)
                    v[j], b[j] = st["env"], True
                    h[j] = hv
                    vf0[j] = st["hf0"]
        return g, v, b, vf0, h

    # Warm-up with the same window size and the same ratios path, so the MPS
    # kernels are compiled before the stream opens.
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
        # v9: the callback only moves memory, at microsecond scale. Inference runs
        # on the worker; 110 ms of MPS
        # Doing work here was the real cause of the break-up in v1 to v8: the
        # CoreAudio deadline is hard, so meeting it on average is worthless, and
        # the status flag used to be ignored, which gave false reassurance.
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
    TICK_FRAMES = 16                    # 16 x 512 / 44100 is about 186 ms, the original tick rate

    def ear_note_tick(f_in):
        """v14 direct note feed: ear.tick minus tracker.push and the key lock,
        which together were the whole 55 ms cost. k_shift is maintained externally
        by KeyTracker and f_in comes from the frozen f0.
        The reactive path is aligned line for line with live_v3.EarV3.tick.
        v24 rehearse: score tracking replaces the model. Absolute notes in
        microphone space pass straight through, and the token, snapping and
        k_shift layers never enter. When they leave the script, the parts are the
        score anchor, with the interval computed against what they really
        sing."""
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
        real = f_in                 # the note they really sang, before snapping or folding; the v18 interval reference
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
            # v18 (review F6): the reference is the note they really sang. The
            # old lead - offsp was the internal representation after snapping and
            # folding, which differs by a semitone when snapped to C major;
            # measured, 40% of ticks triggered it and the bass sang a semitone
            # sharp. respond2.py:419 carries the same lesson, and the streaming
            # version walked into it again.
            # v20: the real-singing reference must also subtract the same snapping
            # difference d that token() applies. The model writes harmony entirely
            # in snapped space, so computing the interval against the unsnapped
            # real singing gives the model's interval minus d: the bass aiming for
            # a unison becomes -1 and sits on top of them, and the alto's fifth
            # becomes a tritone (diag1: P(bass = -1 | snapped) is 24% against 4%
            # unsnapped; correcting it in simulation took the share of -1 from
            # 11.3% to 4.2% and the alto from 4/6 to 5/7).
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
                # v19 interval hysteresis (proved by diag1: 7 note changes per
                # second with a median dwell of 12 ms, so the harmony line was
                # chattering, which is the main reason it did not sound like
                # harmony): a new interval is adopted only after two consecutive
                # ticks agree, giving a floor of about 370 ms on the dwell, so a
                # one-off roll of the dice never reaches the output.
                if iv == st["cur"][part]:
                    st["ivp"][part] = None
                elif st["ivp"][part] == iv:
                    if a.pad > 0 and (time.time() - st["padt"][part]
                                      < a.pad):
                        # v37 harmonic rhythm limit: do not change before the
                        # dwell is complete; try again next tick, keeping ivp.
                        pass
                    else:
                        st["cur"][part] = iv
                        st["ivp"][part] = None
                        if a.pad > 0:
                            st["padt"][part] = time.time()
                            st["pada"] = float(mic)   # the pad anchor: their note at the moment the chord changes
                else:
                    st["ivp"][part] = iv

    brain_acc = []

    def brain_drain():
        """Consume the frozen f0 frames accumulated in bq into ticks. Live calls
        this from a thread and replay calls it synchronously after each step, so
        the logic is identical and the model lags by one block in both."""
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
        # The model thread consumes frozen f0 frames and no longer touches audio,
        # so its cost approaches zero.
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
            # v11 frozen f0 grid: freeze on arrival, since the p95 of the
            # end-of-window estimate is 4.3 cents, which is good enough, plus
            # hysteretic notes. The interval and the stability streak enter the
            # cache too, so rendering and the phase accounting share one exact
            # value.
            xb = buf.astype("float64")
            if a.agc:
                xb = xb * st["agc_g"]      # v22: the model sees the levelled signal; the output divides it back out
            if sus_f:
                f0f, vol_t, mask, uvf = svc.prep(xb, a.thr, want_uv=True)
                if a.floor_db < 0:
                    # v21: a frame below the floor counts as unvoiced. Both the
                    # gate in bridge_feed downstream and the rest decision fed to
                    # the model read uvf, so one place governs. The threshold is
                    # defined against the original level, and with the AGC on the
                    # volume has already been multiplied by g, so the threshold is
                    # multiplied too.
                    uvf = uvf | (vol_t[0, :, 0].cpu().numpy()
                                 < 10 ** (a.floor_db / 20.0) * st["agc_g"])
            else:
                f0f, vol_t, mask = svc.prep(xb, a.thr)
            tp = pt("prep", t0)
            if a.agc:
                # Gain update, causal and taking effect one block later: an EMA
                # with a time constant of about 2 s over the original level of the
                # active spans of the new frames gives g = REF / ema, clamped to
                # [0.25, 16], the same bounds as input_agc.
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
            # v33 absolute frame position: on the first window every frame is new,
            # and afterwards it advances by nb per block.
            st["abs"] = len(f0f) if (fc is None or len(fc) != len(f0f)) \
                else st["abs"] + nb
            if fc is None or len(fc) != len(f0f):
                if sus_f:
                    g, v, b, vf0, h = bridge_feed(
                        vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                    st["gt"], st["vc"], st["br"], st["bu"] = g, v, b, h
                    st["fc"] = vf0      # a bridged frame freezes the safe f0; a voiced frame keeps f0f
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
                # Advance the absolute phase: the nb frames leaving the window
                # times the interval used to render them, taken from the cache.
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
                    if st.get("gt") is None:    # self-healing: an exception during init must not deadlock for good
                        g, v, b, vf0, h = bridge_feed(
                            vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                        st["gt"], st["vc"], st["br"], st["bu"] = g, v, b, h
                        st["fc"] = vf0
                    else:
                        g, v, b, vf0, h = bridge_feed(
                            vol_t[0, -nb:, 0].cpu().numpy(),
                            uvf[-nb:], f0f[-nb:])
                        st["gt"] = np.concatenate([st["gt"][nb:], g])
                        st["vc"] = np.concatenate([st["vc"][nb:], v])
                        st["br"] = np.concatenate([st["br"][nb:], b])
                        st["bu"] = np.concatenate([st["bu"][nb:], h])
                        st["fc"] = np.concatenate([fc[nb:], vf0])
                else:
                    st["fc"] = np.concatenate([fc[nb:], f0f[-nb:]])
                # v34 --f0-mature: deferred freezing. The last M+nb frames, not
                # yet played or only just played, are refreshed each window with
                # the latest estimate. At the end of a window parselmouth has no
                # right context and is 37.6 cents out over a note change (measured
                # by replaying against a re-measurement with full context); one
                # more window of context matures the estimate. Bridged frames are
                # not refreshed, keeping the freezing philosophy, since hf0 is
                # deliberate. Use it with --lookahead M, so frames play M frames
                # later and the estimate has matured by then; without lookahead it
                # only improves the part that is later re-read as left context.
                # 0 is off and bit-identical to the old behaviour.
                if a.f0_mature > 0:
                    M = min(a.f0_mature + nb, len(f0f))
                    keep = st["br"][-M:] if sus_f else \
                        np.zeros(M, dtype=bool)
                    st["fc"][-M:] = np.where(keep, st["fc"][-M:], f0f[-M:])
                if ear is not None or REH is not None:
                    with block:
                        # v18 (review F7): unvoiced frames send 0 so the model
                        # can receive a rest. The old version fed interpolated f0,
                        # which produced 0 rests in 484 ticks and took the token
                        # stream outside the training distribution. The w > 0
                        # filter in brain_worker already handles it.
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
                        if a.pad > 0 and st["pada"] is not None:
                            # v37 --pad: the parts are pinned to the absolute
                            # pitch of the anchor note plus the interval, which is
                            # a chord pad. offc cancels their f0 movement frame by
                            # frame, so melisma and vibrato do not carry over, and
                            # unvoiced frames fall back to the relative interval.
                            # A jump, from a chord change or entering and leaving
                            # silence, glides in at up to 0.8 semitones per frame,
                            # about 200 ms of portamento.
                            fw = f0f[-nb:]
                            mw = 69 + 12 * np.log2(
                                np.maximum(fw, 1.0) / 440.0)
                            new = np.where(fw > 0,
                                           st["pada"] + tgt - mw, tgt)
                            pv = float(st["offc"][vi][-1])
                            for j3 in range(nb):
                                pv += float(np.clip(new[j3] - pv, -0.8, 0.8))
                                new[j3] = pv
                        else:
                            new = np.full(nb, tgt)
                            pv = float(st["offc"][vi][-1])
                            if sus_f and pv != tgt:
                                # v18 (review F8): a note change glides into the
                                # new interval over about 35 ms rather than jumping
                                # for a whole block, in the spirit of the offline
                                # target_f0 portamento.
                                k3 = min(nb, 3)
                                new[:k3] = np.linspace(pv, tgt,
                                                       k3 + 2)[1:k3 + 1]
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
                            # v19: once k_shift is locked it never moves again,
                            # the same discipline as respond2. Continuous
                            # re-estimation means one drift shifts every interval
                            # at once, which is a suspect in the chattering of the
                            # model line.
                            ear.k_shift = rb[0]
                            st["klock"] = True
            tp = pt("bridge+brain", tp)
            # v2 frozen unit cache plus v12 smoothing on sustained notes: while
            # they sing a closed vowel, HuBERT guesses a different vowel frame by
            # frame at the end of the window and the output vowel wanders. Once
            # the note is stable, a streak of 6 frames or about 70 ms, the units
            # are pinned by an EMA at alpha 0.12, a time constant of about 90 ms;
            # on a note change or a consonant the streak resets and alpha goes to
            # 1, so it follows at once and the attack is not smeared. The
            # smoothing runs causally on the absolute grid, so it agrees across
            # windows.
            if VOW is not None:
                # v37 --aah: the material becomes the take-17 vowel loop on the
                # absolute frame grid, so it agrees across windows, and the
                # encode, EMA freezing and unit maturing are all bypassed. The
                # vowel bank does not chatter, so the frame-by-frame HuBERT
                # guessing that chain defends against does not exist, and not
                # running HuBERT saves the largest share of all. f0 (st["fc"]),
                # the volume, the mask and the harmony ratio (offc) are unchanged,
                # so it still follows them live.
                Lw = input_frame // HOP + 1
                idx = (st["abs"] - Lw + np.arange(Lw)) % VOW.size(1)
                st["usm"] = VOW[:, torch.from_numpy(idx).to(VOW.device)]
                fresh = None
            elif ENC_WIN:
                # v26: the encoder takes 3 s of past audio, buying diction with
                # left context at no cost, and only the tail frames corresponding
                # to the render window are taken, so uc, usm and infer downstream
                # cannot tell the difference.
                xbe = buf3.astype("float64")
                if a.agc:
                    xbe = xbe * st["agc_g"]
                fresh = svc.encode(xbe)[:, -(input_frame // HOP + 1):]
            else:
                fresh = svc.encode(xb)
            tp = pt("encode", tp)
            uc = st["uc"]
            if fresh is None:
                pass                                 # --aah: usm is already ready
            elif uc is None or uc.size(1) != fresh.size(1):
                st["uc"] = fresh
                st["usm"] = fresh.clone()
                st["ua1"] = np.ones(fresh.size(1), dtype=bool)  # v36
            else:
                keep = uc[:, nb:]
                m = 2 * nb
                st["uc"] = torch.cat(
                    [keep[:, :fresh.size(1) - m], fresh[:, -m:]], 1)
                us = st["usm"][:, nb:]
                # v23: the per-frame EMA loop runs on the CPU in numpy. At 256
                # dimensions times nb frames the CPU cost is nil, while the old
                # version launched 2-3 MPS kernels per frame, which was the bulk
                # of the hidden cost of a step. One download, the computation, one
                # upload, numerically the same as a float32 lerp.
                fr = fresh[:, -nb:].detach().cpu().numpy()
                prev = us[:, -1].detach().cpu().numpy()
                news = np.empty_like(fr)
                ua1_new = np.zeros(nb, dtype=bool)
                for j in range(nb):
                    # v30: a bridged frame whose level is still present, st["bu"],
                    # means they are still singing, so the units are not frozen and
                    # follow as usual, which treats consonants such as /t/ not
                    # coming out. The side effect is welcome: v18's rewind to the
                    # mouth of the gap is now triggered by a real fall in level and
                    # no longer rewinds from a false gap.
                    if sus_f and st["br"][-nb + j] and not st["bu"][-nb + j]:
                        if not st["inbr"]:
                            # v18 rewind to the mouth of the gap: that frame is
                            # already contaminated by the consonant, so the frozen
                            # vowel comes from the oldest value in the FIFO, about
                            # 45 ms earlier, following the same philosophy as the
                            # f0 hold.
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
                    ua1_new[j] = (al == 1.0)
                st["ua1"] = np.concatenate([st["ua1"][nb:], ua1_new])
                st["usm"] = torch.cat(
                    [us, torch.from_numpy(news).to(us)], 1)
                # v36 --units-mature: mature refresh of transition frames.
                # Diagnosis on the bench, 2026-08-11: live transition frames have a
                # unit cosine of 0.768 against 0.899 for the reference, so the gap
                # lies entirely in streaming, while the enc-win, t_start and ema
                # controls are all flat within plus or minus 0.01. The mechanism is
                # that HuBERT has no right context at the end of a window, the
                # transitions are the blurriest part, and once they enter usm they
                # are frozen for good. The refresh rule: within the last M frames,
                # before this block's new frames, transition frames where alpha == 1
                # are overwritten with this window's mature estimate, which now has
                # about M x 11.6 ms of right context. Pinned frames with alpha < 1
                # are untouched, so the v12 protection of sustained notes is
                # intact. 0 is off and bit-identical to the old behaviour.
                if a.units_mature > 0:
                    L = st["usm"].size(1)
                    lo = max(0, L - a.units_mature - nb)
                    zone = np.zeros(L, dtype=bool)
                    zone[lo:L - nb] = st["ua1"][lo:L - nb]
                    if zone.any():
                        zi = torch.from_numpy(np.where(zone)[0]).to(
                            st["usm"].device)
                        st["usm"][:, zi] = fresh[:, zi]
            if sus_f:
                # v18: the bridge gate replaces the original mask outright. A
                # throat microphone's noise floor of -55 dB is above the -60
                # threshold, so the original mask was permanently open (review F1:
                # three gaps in twenty minutes), taking the maximum amounted to
                # having no gate, a real rest could not close it, and the bass hummed
                # at low frequency. The bridge gate already contains "voiced means
                # open". The volume comes from the frozen cache, holding the
                # envelope through a gap.
                gu = upsample(torch.from_numpy(st["gt"]).float().to(
                    svc.device)[None, :, None], HOP).squeeze(-1)
                k2 = min(mask.size(1), gu.size(1))
                mask = gu[:, :k2]
                vsrc = st["vc"]
                if a.swell > 0:
                    # v37 --swell: rate-limit the rise with an asymmetric
                    # first-order filter, leaving the fall instantaneous so endings
                    # stay natural. Computed once for new frames and cached, so it
                    # agrees across windows, following the frozen-grid philosophy.
                    vs = st.get("vsw")
                    if vs is None or len(vs) != len(vsrc):
                        st["vsw"] = np.asarray(vsrc, dtype=float).copy()
                    else:
                        head = vs[nb:]
                        p3 = float(head[-1]) if len(head) else float(vsrc[0])
                        au_ = 1 - float(np.exp(-(HOP / SR)
                                               / (a.swell / 1000.0)))
                        seg3 = np.empty(nb)
                        for j3 in range(nb):
                            t3 = float(vsrc[-nb + j3])
                            p3 += (au_ if t3 > p3 else 1.0) * (t3 - p3)
                            seg3[j3] = p3
                        st["vsw"] = np.concatenate([head, seg3])
                    vsrc = st["vsw"]
                vol_t = torch.from_numpy(vsrc).float().to(
                    svc.device)[None, :, None]
            tp = pt("units", tp)
            aus = svc.infer(xb, a.pitch, a.thr,
                            units_override=st["usm"],
                            feats=(st["fc"], vol_t, mask), phases=accs,
                            ratios=st["offc"],
                            enh_tail=(blk + cf + sola_search + last_delay
                                      + LA + int(0.1 * SR))
                            if a.enh_tail else 0,
                            frame0=st["abs"] - len(st["fc"]))
            tp = pt("infer", tp)
            y = None
            stems = [] if (dmp is not None and a.dump_stems) else None
            for vi, au in enumerate(aus):    # SOLA and join each part separately, mixing only once they are assembled
                # With all the features frozen, the overlap correlates at 0.96,
                # shown by measurement, so all that is left at the join is a phase
                # shift, which is exactly SOLA's job. With the content identical,
                # the shift is stable and no longer chatters.
                tw = au[-blk - cf - sola_search - last_delay - LA:
                        -last_delay - LA if last_delay + LA else None]
                ci = tw[None, None, :cf + sola_search]
                num = F.conv1d(ci, sola_bufs[vi][None, None, :])
                den = torch.sqrt(F.conv1d(ci ** 2, torch.ones(
                    1, 1, cf, device=svc.device)) + 1e-8)
                score = num[0, 0] / den[0, 0]
                # v34 --sola-cont: continuity bias on the shift. Diagnosis by
                # replay, 2026-08-10: argmax jumps between correlation peaks a
                # fundamental period apart, with a p90 |delta shift| of 7 ms and no
                # preference for any passage. On a flat passage the jump is
                # inaudible because the periods align; over a note change the
                # deviation is the slope times delta t, and on a steep segment that
                # is 50-74 cents, which is the main cause of unstable pitch. The
                # bias weights peaks near the previous block's shift, choosing the
                # nearest good-enough peak rather than the global maximum. 0 is off
                # and bit-identical to the old behaviour.
                if a.sola_cont > 0:
                    dist = torch.abs(torch.arange(
                        score.numel(), device=score.device, dtype=score.dtype)
                        - prev_shift[vi])
                    score = score * (1 - a.sola_cont * dist / sola_search)
                shift = int(torch.argmax(score))
                prev_shift[vi] = shift
                if dmp is not None:      # v33 instrument: evidence for tracing timebase jitter at the joins
                    dmp[f"shift{vi}"].append(np.array([shift]))
                tw = tw[shift: shift + blk + cf].clone()
                # v34 --seam-amp: a plain amplitude crossfade, skipping the phase
                # vocoder. Diagnosis: crossfade length has a monotonic dose response
                # on tuning degradation (0.02, 0.04 and 0.08 give 60, 73 and 82
                # cents on a steep segment), which shows that phase interpolation
                # bends the instantaneous frequency. Once the shift is aligned by
                # --sola-cont, an amplitude crossfade does not bend the pitch. Off
                # by default, the old behaviour.
                if a.seam_amp:
                    tw[:cf] = sola_bufs[vi] * fade_out + tw[:cf] * fade_in
                else:
                    tw[:cf] = phase_vocoder(sola_bufs[vi], tw[:cf],
                                            fade_out, fade_in)
                sola_bufs[vi] = tw[-cf:]
                w = tw[:-cf].cpu().numpy()
                if stems is not None:
                    stems.append(w.copy())
                y = w if y is None else y[:len(w)] + w[:len(y)]
            tp = pt("sola+mix", tp)
            if a.agc:
                y = y / st["agc_g"]        # restore the level structure, the same philosophy as input_agc
            if RVIR is not None:
                # v37 streaming convolution: the full response of this block, with
                # the head emitted inside the block and the tail accumulated into RVT
                wf = _fftc(y.astype(float), RVIR)
                wet_ = wf[:len(y)]
                tl = RVT[0]
                wet_ += tl[:len(y)]
                nt = wf[len(y):]
                rest = tl[len(y):]
                nt[:len(rest)] += rest
                RVT[0] = nt
                y = y + a.wet * wet_
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
                    for kk in ("gt", "vc", "br", "bu"):
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
                  f" | late {stats['late']} underruns {st2['under']}"
                  f" flags {st2['flags']}"
                  + (f" bridges {st['nbh']} gaps / {st['nbf']} frames" if sus_f else ""),
                  flush=True)

    def prof_report():
        """The --prof budget table. Note that synchronising itself costs
        something, so the total is higher than the same configuration without
        --prof; read only the proportions and never compare the absolute values
        with the RTF."""
        n = max(stats["n"], 1)
        tot = sum(prof.values())
        out = [f"\nStage budget (--prof; {stats['n']} blocks, block {blk/SR*1000:.0f}ms)",
               f"{'stage':<13}{'ms/block':>11}{'share':>9}"]
        for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
            out.append(f"{k:<13}{v/n:9.1f}{100*v/max(tot, 1e-9):8.1f}%")
        out.append(f"{'total':<13}{tot/n:11.1f}{100.0:8.1f}%")
        return "\n".join(out)

    def flush_dump():
        """Write the dump, on exit and every 10 s, so a hard kill loses at most
        10 s."""
        import soundfile as sf
        sf.write(a.dump + "_mic.wav", np.concatenate(list(dmp["mic"])), SR)
        sf.write(a.dump + "_angel.wav", np.concatenate(list(dmp["out"])), SR)
        np.savez(a.dump + "_feats.npz",
                 **{k: np.concatenate(list(v)) for k, v in list(dmp.items())
                    if k not in ("mic", "out") and len(v)})

    if a.replay:
        import soundfile as sf
        x, sr_in = sf.read(a.replay, dtype="float32", always_2d=True)
        assert sr_in == SR, f"a replay file must be {SR}Hz, got {sr_in}"
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
        if PROF:
            print(prof_report(), flush=True)
        if dmp is not None and dmp["mic"]:
            flush_dump()
            print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz", flush=True)
        return

    print(f"opening stream in={a.in_name!r} out={a.out_name!r} pitch {a.pitch:+.0f}"
          f" t_start {a.t_start} step {a.step}"
          f" seed {a.seed if a.seed is not None else 'random'}"
          f" (Ctrl-C to finish)", flush=True)
    wt = threading.Thread(target=worker, daemon=True)
    wt.start()
    if ear is not None or REH is not None:
        threading.Thread(target=brain_worker, daemon=True).start()

    import signal

    def _term(*_):
        raise KeyboardInterrupt      # a nohup background process blocks SIGINT; TERM also takes the dump path

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
