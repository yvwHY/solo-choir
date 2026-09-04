"""solo_min — a minimal, clean live Solo app: You (dry) + 1 converted harmony voice.

Built on vc_probe's clean core: direct-callback inference (NO pump/worker/ring),
relaxed buffer, one BeatriceSoloChoir instance = the harmony voice, raw mic = the
dry You (delay-aligned + summed). Two switchable modes:
  --mode fixed     : harmony = your voice + a STATIC N-semitone shift → 9/10 clean
                     (5ths/8ves stay in-key; thirds drift off-key — that's expected)
  --mode diatonic  : harmony snaps to a key/scale and follows the melody (in-key),
                     smoothed by the committed pitch-glide → 8/10 (add --free for chord-aware)

Coexists with the choir app (server/beatrice_solo_choir_live.py) — this touches none of it.

Run (conda env vcclient-dev):
    cd server && python solo_min.py --model <paraphernalia_dir> --mode fixed --interval -5 \
        --in-name speaker_set --out-name speaker_set
Ctrl-C to stop.
"""
from __future__ import annotations
import argparse, sys, os, time
import numpy as np
import sounddevice as sd
import soxr

# solo_min lives IN server/, so its siblings import directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voice_changer.SoloChoir import SoloChoirHarmonizer
from voice_changer.naive_shift import GranularShifter
from beatrice_converter import BeatriceSoloChoir

DEV_SR = 48000
NOTE_TO_PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
              "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def pick(name, kind):
    """first device whose name contains `name` (substring), for 'input'/'output'."""
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch > 0 and name.lower() in d["name"].lower():
            return i
    return None


def configure_voice(conv, harm, cfg, steps=None, interval=None):
    """Apply mode settings from cfg to an EXISTING conv/harm. steps/interval override cfg's (so each
    voice in a multi-voice stack can have its own interval while sharing key/scale/glide/stability).
    Reused by build_voices (initial) and SoloEngine.set_control (live).
      fixed    : harmonizer OFF, static pitch_offset = interval → 9/10 clean
      diatonic : harmonizer ON, harmony from compute_shift + glide → 8/10 (Free = chord-aware)
    """
    iv = cfg.interval if interval is None else interval
    st = cfg.steps if steps is None else steps
    if cfg.mode == "fixed":
        harm.enabled = False
        conv.pitch_offset = float(iv)              # STATIC transpose → no scrape
        conv.pitch_glide_step = 0.0
    else:  # diatonic
        harm.enabled = True
        conv.pitch_offset = 0.0                    # harmony comes from compute_shift
        harm.key_root = NOTE_TO_PC[cfg.key.upper()]
        harm.minor = bool(cfg.minor)
        harm.interval_steps = int(st)
        harm.auto_chord = bool(cfg.free)           # Free = chord-aware sub-variant
        harm.voice_lead = bool(getattr(cfg, "voice_lead", False))   # Free sub-variant: independent lines
        harm.note_hysteresis = float(cfg.hysteresis)     # bigger -> less snap jitter, slower note changes
        harm.note_smooth_frames = int(cfg.smooth)        # bigger → steadier pitch, more lag on genuine changes
        conv.pitch_glide_step = float(cfg.glide) if cfg.glide > 0.0 else 0.0
        conv.pitch_hold_frames = 8


def _voice_specs(args):
    """Per-voice (steps, interval, model, speaker) list — one entry per harmony voice (You is the dry
    passthrough, not in this list). Voice N reads steps/steps2/steps3…, interval/interval2/…, etc.
    A voice is included when its steps attr exists. Each voice = +1 inference pass.
    NO cache-thrash when voices share ONE model and differ only by target_speaker (self-trained satb2:
    speaker 0 = female, 1 = male) — the crackle-free way to mix timbres. DIFFERENT models crackle."""
    specs = []
    for suf in ("", "2", "3", "4"):   # add more suffixes here to grow the choir
        st = getattr(args, "steps" + suf, None)
        if st is None:
            continue
        iv = getattr(args, "interval" + suf, args.interval)
        m = getattr(args, "model" + suf, None) or args.model
        s = getattr(args, "speaker" + suf, None)
        s = args.speaker if s is None else s
        specs.append((st, iv, m, s))
    n = int(getattr(args, "voices", 4) or 4)
    specs = specs[:max(1, n)]
    return specs


def build_voices(args):
    """Create + configure the harmony-voice stack. Returns [(conv, harm), ...].
    Extracted so smoke tests + the UI reuse it."""
    voices = []
    for (st, iv, model, spk) in _voice_specs(args):
        harm = SoloChoirHarmonizer(enabled=(args.mode == "diatonic"))
        conv = BeatriceSoloChoir(model, engine_dir=args.engine, harmonizer=harm,
                                 target_speaker=int(spk), formant_shift=float(args.formant),
                                 output_gate_floor=float(args.gate_floor),
                                 output_gate_expon=float(args.gate_expon))
        configure_voice(conv, harm, args, steps=st, interval=iv)
        voices.append((conv, harm))
    return voices


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="paraphernalia dir (voice 1)")
    ap.add_argument("--model2", default=None, help="voice-2 model (default = same as --model). Two different models may cache-thrash and crackle")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--speaker", type=int, default=0, help="voice-1 target speaker (satb2: 0=female, 1=male)")
    ap.add_argument("--speaker2", type=int, default=None, help="voice-2 target speaker (default = --speaker). Same model + different speaker = no cache-thrash way to mix timbres")
    ap.add_argument("--mode", choices=["fixed", "diatonic"], default="fixed",
                    help="fixed = static transpose (9/10 clean, 5ths/8ves in-key); "
                         "diatonic = in-key dynamic harmony w/ glide (8/10)")
    ap.add_argument("--interval", type=float, default=-5.0,
                    help="fixed mode: semitones to transpose the harmony voice (-5 = a fifth below)")
    # diatonic-mode controls
    ap.add_argument("--key", default="C", help="diatonic mode: root note (C..B)")
    ap.add_argument("--minor", action="store_true", help="diatonic mode: natural minor (default major)")
    ap.add_argument("--free", action="store_true", help="diatonic mode: chord-aware (Free) vs fixed interval-in-scale")
    ap.add_argument("--voice-lead", action="store_true",
                    help="Free-mode sub-variant: per-voice voice-leading (hold common tones, minimal "
                         "leap on chord changes, register-anchored) instead of parallel-anchored "
                         "chord tones. Needs --free. Blob-Opera-style independent lines.")
    ap.add_argument("--steps", type=int, default=-2, help="diatonic mode: voice-1 interval in scale-steps (-2 = a third below)")
    ap.add_argument("--steps2", type=int, default=2, help="diatonic: voice-2 interval in scale-steps (+2 = a third above)")
    ap.add_argument("--interval2", type=float, default=7.0, help="fixed: voice-2 transpose in semitones (+7 = a fifth above)")
    ap.add_argument("--steps3", type=int, default=-4, help="diatonic: voice-3 interval in scale-steps (-4 = a fifth below, bass anchor)")
    ap.add_argument("--interval3", type=float, default=-12.0, help="fixed: voice-3 transpose in semitones (-12 = an octave below, bass)")
    ap.add_argument("--model3", default=None, help="voice-3 model (default = --model)")
    ap.add_argument("--speaker3", type=int, default=None, help="voice-3 target speaker (default = --speaker; satb2: 1=male bass)")
    ap.add_argument("--steps4", type=int, default=4, help="diatonic: voice-4 interval in scale-steps (+4 = a fifth above, top)")
    ap.add_argument("--interval4", type=float, default=12.0, help="fixed: voice-4 transpose in semitones (+12 = an octave above)")
    ap.add_argument("--model4", default=None, help="voice-4 model (default = --model)")
    ap.add_argument("--speaker4", type=int, default=None, help="voice-4 target speaker (default = --speaker; satb2: 0=female top)")
    ap.add_argument("--glide", type=float, default=0.04, help="diatonic mode: pitch glide semitones/hop (0 = jump)")
    ap.add_argument("--hysteresis", type=float, default=1.0, help="diatonic: snap hysteresis (bigger = less jitter, slower changes)")
    ap.add_argument("--smooth", type=int, default=30, help="diatonic: pitch median-smoothing frames (bigger = steadier)")
    ap.add_argument("--formant", type=float, default=0.0)
    # --- mouth gate (2026-08-14) — see server/mouth_gate.py for why this exists ---
    # Default OFF: with the flag absent solo_min behaves exactly as before, and
    # nothing imports cv2/mediapipe (they live in the 6x venv, not vcclient-dev).
    ap.add_argument("--mouth-gate", type=int, default=0,
                    help="camera mouth gate on the CONVERTED voices (the dry You is never gated). "
                         "You stop singing -> the choir stops -> nothing loops back into the mic. "
                         "Needs cv2+mediapipe, i.e. run this from the 6x venv. 0 = off (unchanged)")
    ap.add_argument("--gate-fail", choices=["open", "close"], default="open",
                    help="what the gate does if the CAMERA DIES mid-show. 'open' = keep singing "
                         "ungated (matches bank_live's 08-12 ruling; on speakers that means feedback "
                         "risk); 'close' = go silent (safe, but the piece stops). Startup is always "
                         "closed until the first frame arrives.")
    ap.add_argument("--sing-open", type=float, default=0.7, help="gate: P(singing) to OPEN (needs --gate-frames in a row)")
    ap.add_argument("--sing-hold", type=float, default=0.3, help="gate: P(singing) to STAY open (hysteresis)")
    ap.add_argument("--gate-frames", type=int, default=2, help="gate: consecutive frames above --sing-open to open")
    ap.add_argument("--gate-grace", type=float, default=0.8, help="gate: keep open this long after the mouth closes (consonants, breaths)")
    ap.add_argument("--gate-ramp", type=float, default=0.12, help="gate: open/close ramp in seconds (0 would click)")
    ap.add_argument("--cam", type=int, default=-1, help="gate: camera index (-1 = auto-pick the brightest)")
    # --- mic gate (2026-08-18: neural triggers without the camera, following bank) ---
    # The twin of --mic-gate in harmony/bank_live.py v43: gate authority is the
    # microphone level and the camera is never opened, with neither cv2 nor
    # mediapipe imported. A change to the mic gate logic on either side has to
    # be read against the other, the same discipline as mouth_gate.py. While
    # this is on, --mouth-gate gives way.
    ap.add_argument("--mic-gate", type=int, default=0,
                    help="mic-level gate on the CONVERTED voices (0 = off). Camera-free twin of "
                         "bank_live v43: opens when the input hop RMS beats max(--mic-open, recent "
                         "output max + --bleed + --bleed-margin) for --mic-frames hops in a row; "
                         "closes after --mic-hold s below the floor. Takes authority over --mouth-gate. "
                         "Honest edge (F26): an energy gate can mistake feedback for singing — "
                         "the tuned floor keeps that at bay, ears are the judge.")
    ap.add_argument("--mic-open", type=float, default=-45.0,
                    help="mic gate: absolute open level in dBFS (hop RMS). 08-18 bank tuning on this "
                         "mic/room landed at -26 (the -45 default was defeated by room+feedback noise "
                         "sitting at -34~-28 — the shell passes the tuned value)")
    ap.add_argument("--mic-frames", type=int, default=7,
                    help="mic gate: consecutive 10 ms hops above the floor to open (7 = 70 ms, same "
                         "arm time as bank_live's 2 x 35 ms) — single-hop transients don't open it")
    ap.add_argument("--mic-hold", type=float, default=0.8,
                    help="mic gate: seconds below the floor before closing — consonants and word gaps "
                         "don't chop the phrase (same 0.8 s grace as the camera gate)")
    ap.add_argument("--bleed", type=float, default=0.0,
                    help="mic gate: speaker->mic coupling in dB (negative; 0 = floor is --mic-open "
                         "only). Floor is raised to recent-output-max + bleed + margin so the choir "
                         "in the mic doesn't count as singing. This rig measured median -13.4 (08-18)")
    ap.add_argument("--bleed-margin", type=float, default=6.0,
                    help="mic gate: dB above the bleed estimate that counts as real singing")
    # --- quantised parts (2026-08-18 tuning work; all off by default, so the
    #     original path is bit-identical) ---
    ap.add_argument("--voices", type=int, default=4,
                    help="build only the first N parts (4 = the original behaviour); with 1-2 parts the inference cost falls with it")
    ap.add_argument("--quantize", type=int, default=0,
                    help="quantised parts: each part sings the absolute in-key target of the note the singer stands on "
                         "(a note is committed only after holding for 30 ms, so passing notes cannot enter), and the "
                         "singer's intonation movement is cancelled frame by frame. Includes just-intonation offsets and "
                         "a shared tuning centre (time constant about 2.5 s tracking the singer, with voice 1 as leader). "
                         "0 = off, the parts follow the singer's continuous pitch, which is the old behaviour")
    ap.add_argument("--q-glide", type=float, default=0.4,
                    help="quantize note-change ramp in semitones per hop; a target jump larger than --q-snap lands directly")
    ap.add_argument("--q-snap", type=float, default=3.5,
                    help="a quantize target jump larger than this many semitones lands directly instead of gliding "
                         "(from the smeared note changes of 2026-08-18: smaller is crisper, at the cost of a possible "
                         "stepped quality on large leaps)")
    ap.add_argument("--vib", default="",
                    help="per-part vibrato for quantize, 'cents@Hz,...' such as '14@5.2,12@4.6'; parts beyond the list "
                         "reuse it cyclically, and empty means no vibrato. Opens over 300 ms at the attack and breathes in depth")
    ap.add_argument("--wet-send", type=float, default=0.0,
                    help="choir reverb send (0 = off). A synthesised IR from harmony/reverb.py through ConvLive "
                         "partitioned convolution, wetting the converted parts only; the dry voice stays out of the "
                         "reverb, the same meaning as [wet] in bank")
    ap.add_argument("--wet-t60", type=float, default=1.2, help="reverb T60 in seconds")
    ap.add_argument("--no-dry", action="store_true", help="drop the dry You passthrough (harmony voice only)")
    ap.add_argument("--dry-delay-ms", type=float, default=40.0,
                    help="delay the dry You to align with the converted voice's algorithmic latency (tune by ear)")
    ap.add_argument("--cushion-ms", type=float, default=20.0,
                    help="output buffer cushion to absorb residual jitter; adds this much latency")
    ap.add_argument("--resample-quality", default="QQ", choices=["QQ", "LQ", "MQ", "HQ", "VHQ"],
                    help="soxr quality. QQ (linear, ~zero latency) avoids the bursty-production tick; HQ is higher quality but bursty (needs a big cushion / pump)")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--blocksize", type=int, default=480, help="device frames/callback @48k (480=10ms)")
    ap.add_argument("--latency", default="low")
    ap.add_argument("--out-map", default=None,
                    help="per-voice output channel map, comma-separated, one entry per voice "
                         "(e.g. '0,0,1,2' = voices 1+2 → ch0, voice 3 → ch1, voice 4 → ch2). Needs ONE "
                         "output device with max(map)+1 channels — multi-out interface or Aggregate "
                         "Device with Drift Correction; never two streams (FINDINGS F7). Routing only, "
                         "no widening (GRAVEYARD G11).")
    ap.add_argument("--dry-ch", type=int, default=-1,
                    help="with --out-map: which channel gets the dry You (-1 = all channels)")
    ap.add_argument("--voice-delay", default=None,
                    help="per-voice output delay in SECONDS, comma-separated, one entry per voice "
                         "(e.g. '0,0.6,1.2,1.8' = voice 1 live, voices 2-4 enter staggered). Tier-1 "
                         "canon/round (the part-separation roadmap): each delayed voice is 'you N seconds ago', "
                         "harmonized as of then. FIFO on the converted output only — the dry You stays "
                         "live and leads; inference cost unchanged. Static per launch. Composes with "
                         "the mono and --out-map paths (delay applied before routing).")
    ap.add_argument("--naive", action="store_true",
                    help="BASELINE MODE (research A/B, RQ1): harmony voices are the singer's own audio "
                         "pitch-shifted by a granular delay-line shifter (the author's Harmonizing / "
                         "Tone.PitchShift algorithm) instead of Beatrice re-synthesis. Same f0 and the "
                         "same diatonic decision — the variable under test is whether the harmony has to "
                         "sound like a human voice. NOT strictly latency-matched, but close enough not to "
                         "confound it: measured onset delay is 4-50 ms depending on shift amount and "
                         "grain phase (mean 10-33 ms) — the two taps sit half a window apart so one is "
                         "always reading recent audio. Same order as the engine path; it drifts, which "
                         "is part of the granular character. "
                         "Toggles live from the UI ({\"naive\": true}).")
    ap.add_argument("--gate-floor", type=float, default=0.0,
                    help="output envelope on the converted voices: input RMS at/above which they play at "
                         "full level, below it they scale down (0 = off, the pre-2026-07-22 solo_min "
                         "behaviour; ~0.01 suppresses the model's noise floor in silent gaps). The "
                         "original app has had this as --gate-floor all along; solo_min never wired it "
                         "up, which only became audible once --choir-gain could lift the choir 18 dB.")
    ap.add_argument("--gate-expon", type=float, default=3.0,
                    help="gate attenuation curve (only used when --gate-floor > 0): dB cut below the "
                         "floor is multiplied by this. 1 = the original app's linear gate — which "
                         "+18 dB of --choir-gain cancels exactly (measured 2026-07-22); 3 keeps ~2x "
                         "the cut even after that gain.")
    ap.add_argument("--choir-gain", type=float, default=1.0,
                    help="linear gain on the converted voices only (the dry You is untouched) — the "
                         "choir/you balance. 1.0 = the original mix, so the default path is unchanged")
    ap.add_argument("--you-gain", type=float, default=1.0,
                    help="linear gain on the dry You only. The RIGHT knob for 'the choir is buried "
                         "under my own voice': pulling You down shifts the balance without amplifying "
                         "the converted voices' residual noise the way --choir-gain does. 1.0 = "
                         "unchanged default path.")
    ap.add_argument("--entrance", action="store_true",
                    help="EXHIBITION OPENING: don't start in full live mode. The dry You is there from "
                         "the first note; the converted voices fade in one at a time (V1 first) and once "
                         "all are up it IS the normal live instrument — nothing else changes. Armed at "
                         "launch: the sequence starts when singing is detected (input RMS over "
                         "--entrance-thresh for --entrance-hold seconds), and re-arms itself after "
                         "--entrance-rearm seconds of silence so the next person gets the same opening. "
                         "Level-only VAD on purpose: f0/voiced is not computed in fixed mode, so it "
                         "would never trigger there. Gain envelope only — no extra inference, no change "
                         "to the audio path once the voices are up (each gain = 1.0 → original mix).")
    ap.add_argument("--entrance-stagger", type=float, default=2.0, help="--entrance: seconds between voice entries")
    ap.add_argument("--entrance-fade", type=float, default=1.0, help="--entrance: seconds each voice takes to fade in")
    ap.add_argument("--entrance-thresh", type=float, default=0.004,
                    help="--entrance: absolute RMS floor under which nothing ever counts as singing "
                         "(the relative test is --entrance-over; whichever is higher wins)")
    ap.add_argument("--entrance-over", type=float, default=4.0,
                    help="--entrance: how many times the tracked noise floor the input must reach to "
                         "count as singing (4x = 12 dB over the room; a false start costs one empty "
                         "opening, a missed one costs the whole experience — so lean low)")
    ap.add_argument("--entrance-hold", type=float, default=0.5, help="--entrance: seconds over threshold before the sequence starts")
    ap.add_argument("--entrance-rearm", type=float, default=8.0, help="--entrance: seconds of silence that reset it back to the opening")
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    return ap


class MicGate:
    """Mic-level gate (08-18) — camera-free twin of bank_live v43's --mic-gate.

    Everything runs ON THE AUDIO THREAD (pure arithmetic: no I/O, no locks,
    no threads) — _cb feeds it the input hop + the output block, ramp() is the
    same hook MouthGate exposes. main() polls .on/.lvl/.fl on its own clock to
    print transitions (printing inside the callback is a drop inducer, 08-05
    playbook). A change to the mic gate logic in bank_live has to be read against
    this, the same discipline as mouth_gate."""

    def __init__(self, open_db, frames, hold_s, bleed, margin,
                 ramp_s, sr, hop_s, out_win):
        self.open_db, self.frames = float(open_db), int(frames)
        self.bleed, self.margin = float(bleed), float(margin)
        self.hold_hops = max(1, int(round(float(hold_s) / hop_s)))
        self._step = 1.0 / max(1e-6, float(ramp_s)) / sr   # per-sample gain step
        self._g = 0.0                        # closed at the start; the singer has not begun
        self._arm = 0                        # consecutive hops above the floor
        self._run = 10 ** 6                  # hops since last above the floor
        # F26: the feedback peak sits ~280 ms behind the output — keep a window
        # of recent output levels and take the MAX (a single point would gamble
        # on the latency estimate; the window only over-blocks, never under).
        self._olv = [-120.0] * max(1, int(out_win))
        self.on, self.lvl, self.fl = False, -120.0, self.open_db   # snapshot read by main()
        self.n_hops = self.n_open = 0        # cumulative open fraction (tuning aid)

    def note_output(self, y):
        """One callback block of what actually left for the speakers (incl. dry)."""
        lvl = 20.0 * np.log10(float(np.sqrt(np.mean(y * y))) + 1e-12)
        self._olv = self._olv[1:] + [lvl]

    def note_input_hop(self, hop):
        """One 10 ms input hop -> advance the open/close state machine."""
        self.lvl = 20.0 * np.log10(float(np.sqrt(np.mean(hop * hop))) + 1e-12)
        fl = self.open_db
        if self.bleed < 0:
            fl = max(fl, max(self._olv) + self.bleed + self.margin)
        self.fl = fl
        if self.lvl > fl:
            self._arm += 1
            if self._arm >= self.frames:
                self._run = 0
        else:
            self._arm = 0
            self._run += 1
        self.on = self._run < self.hold_hops
        self.n_hops += 1
        self.n_open += int(self.on)

    def ramp(self, n):
        """Same contract as MouthGate.ramp: length-n gain ramp, advances state."""
        tgt = 1.0 if self.on else 0.0
        g0 = self._g
        if g0 == tgt:
            return np.full(n, np.float32(tgt), dtype=np.float32)
        d = self._step * n
        g1 = min(tgt, g0 + d) if tgt > g0 else max(tgt, g0 - d)
        out = np.linspace(g0, g1, n, dtype=np.float32)
        self._g = float(g1)
        return out

    def stop(self):                          # interface parity with MouthGate
        pass


class SoloEngine:
    """Owns the audio stream + the single harmony voice. CLI main() and the Phase-2 pywebview UI both
    drive it. set_control(dict) mutates mode/params live (no rebuild) by re-running configure_voice."""

    def __init__(self, args):
        self.args = args
        self.voices = build_voices(args)                       # [(conv, harm), ...] — You + these
        c0 = self.voices[0][0]
        # Resampler quality: HQ/MQ/LQ have large internal latency → they emit in BURSTS (0 samples for a
        # few callbacks then a big dump) → out48 underruns → fixed-rhythm tick. QQ (linear) has ~zero
        # latency → exactly the expected samples every callback → no burst, no tick. Linear resampling is
        # perceptually fine here (Beatrice is the quality anchor; this is a live instrument). Configurable.
        q = args.resample_quality
        self.in_rs = soxr.ResampleStream(DEV_SR, c0.in_sr, 1, dtype="float32", quality=q)    # 48k->16k
        self.out_rs = [soxr.ResampleStream(c0.out_sr, DEV_SR, 1, dtype="float32", quality=q) for _ in self.voices]
        self.HOP = c0.in_sr // 100   # 160 @16k = 10ms
        self.in16 = np.zeros(0, dtype=np.float32)
        # --naive baseline: the singer's own audio at the converter's OUTPUT rate, so the shifted
        # result drops into the existing out_rs chain unchanged. Kept warm (fed every hop even when
        # the mode is off) so toggling live in the middle of a phrase doesn't cold-start — same
        # "stay warm" reasoning as the per-voice on/off above. Default audio path is untouched.
        self.naive_rs = soxr.ResampleStream(c0.in_sr, c0.out_sr, 1, dtype="float32", quality=q)
        self.shifters = [GranularShifter(c0.out_sr) for _ in self.voices]
        # A/B switching swaps one signal for another → an instant swap clicks. Both branches are
        # computed every hop anyway, so crossfade between them (equal power, ~25 ms) — the toggle
        # stays usable mid-phrase, which is the whole point of a live A/B.
        self._ab = 0.0                                        # 0 = Beatrice, 1 = shifter
        self._ab_step = 1.0 / max(1, int(round(0.025 * c0.out_sr)))
        # The converter emits exactly OUT_HOP samples per input hop; the resampler does not (237 on
        # the first hop, 240 after). Buffer it and emit the same fixed count, so the two branches are
        # always sample-aligned — otherwise the crossfade truncates the Beatrice block (a click) and
        # the output cushion drifts in naive mode.
        self.OUT_HOP = c0.out_sr // 100
        self._dry24 = np.zeros(0, dtype=np.float32)
        # set_target_speaker is the expensive half of a control message (the rest of configure_voice
        # is attribute assignment). Remember what each voice already has so a control press only pays
        # for the voice that actually changed — build_voices already applied these at construction.
        self._spk = [int(sp) for (_st, _iv, _m, sp) in _voice_specs(args)]
        # Output cushion: production is bursty (whole-hop draining) so out48 momentarily dips below the
        # device's n and the tail zero-fills → a fixed-rhythm 'tick'. Prime out48 with a cushion of
        # silence so it stays ahead and absorbs the jitter (no pump needed). The cushion equally delays
        # BOTH converted + dry, so the tuned dry alignment (relative offset = dry_delay_ms) is preserved.
        self.CUSHION = int(round(args.cushion_ms * DEV_SR / 1000.0))
        self.out48 = np.zeros(self.CUSHION, dtype=np.float32)
        self.voice_on = [True] * len(self.voices)   # per-voice on/off; OFF voices stay WARM (still run) but aren't summed → no cold-resume glitch on re-enable
        self.out_map = None                          # per-voice→channel routing; None = legacy mono path
        self.n_out = 2
        if getattr(args, "out_map", None):
            self.out_map = [int(t) for t in str(args.out_map).split(",")]
            if len(self.out_map) != len(self.voices) or min(self.out_map) < 0:
                raise SystemExit(f"--out-map needs {len(self.voices)} non-negative entries (one per voice)")
            self.n_out = max(self.out_map) + 1
            self.out48_ch = [np.zeros(self.CUSHION, dtype=np.float32) for _ in range(self.n_out)]
            self.dry_ch = int(getattr(args, "dry_ch", -1))
        self.delay_fifo = [None] * len(self.voices)  # --voice-delay: per-voice output FIFO; None = undelayed
        if getattr(args, "voice_delay", None):
            vals = [float(t) for t in str(args.voice_delay).split(",")]
            if len(vals) != len(self.voices) or min(vals) < 0:
                raise SystemExit(f"--voice-delay needs {len(self.voices)} non-negative entries (one per voice)")
            self.delay_fifo = [np.zeros(int(round(v * DEV_SR)), dtype=np.float32) if v > 0 else None
                               for v in vals]
        # --entrance (exhibition opening): per-voice fade-in gain, applied just before the mix. All 1.0
        # unless the flag is on, and 1.0 again once the last voice is up → the steady state is the
        # unmodified mix. Times are held in samples so the envelope is driven by the audio clock.
        self.ent_gain = [1.0] * len(self.voices)
        self.ENT_STAG = int(round(args.entrance_stagger * DEV_SR))
        self.ENT_FADE = max(1, int(round(args.entrance_fade * DEV_SR)))
        self.ENT_HOLD = int(round(args.entrance_hold * DEV_SR))
        self.ENT_REARM = int(round(args.entrance_rearm * DEV_SR))
        self.ent_trig = float(args.entrance_thresh)   # level that counts as singing; UI shows it on the meter
        self._ent_arm()
        self.choir_gain = float(args.choir_gain)     # converted voices only (dry untouched); 1.0 = original mix
        self.you_gain = float(args.you_gain)         # dry You only; 1.0 = original mix
        self.out_gain = 1.0                          # master output gain (linear); 1.0 → gain branch skipped, byte-identical
        self.dry_on = not args.no_dry
        self.DRY_DELAY = int(round(args.dry_delay_ms * DEV_SR / 1000.0))
        self.dry_hist = np.zeros(self.DRY_DELAY + self.CUSHION, dtype=np.float32)   # +cushion → stays aligned
        # Mouth gate (08-14). Imported ONLY when asked: cv2/mediapipe live in the
        # 6x venv, and vcclient-dev must keep working with the flag absent.
        if int(getattr(args, "quantize", 0)):
            # Wiring the quantised parts (2026-08-18): these are all converter
            # attributes, set per part.
            vibs = []
            for tok in str(getattr(args, "vib", "") or "").split(","):
                tok = tok.strip()
                if tok:
                    c_, h_ = tok.split("@")
                    vibs.append((float(c_), float(h_)))
            tune = {"off": 0.0}               # shared, so the chord locks internally while the
                                              # whole tracks the singer
            for vi, (conv, _h) in enumerate(self.voices):
                conv.quantize = True
                conv.q_glide = float(args.q_glide)
                conv.q_snap = float(args.q_snap)
                conv.q_ji = True
                conv.q_tune = tune
                conv.q_tune_lead = (vi == 0)
                if vibs:
                    vc, vh = vibs[vi % len(vibs)]
                    conv.vib_cents, conv.vib_hz = vc, vh
        # Optional shared-room wet on the CONVERTED mix (2026-08-18 tuning
        # work): callable(block48) -> block48, with None leaving the original
        # path bit-identical. It sits before the dry sum, so the dry voice, the
        # singer themselves, stays out of the reverb, the same meaning as [wet]
        # in bank.
        self.wet_fn = None
        if float(getattr(args, "wet_send", 0.0) or 0.0) > 0.0:
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "harmony"))
            import reverb
            _ir = reverb.make_ir(t60=float(args.wet_t60), predelay_ms=20.0,
                                 trim_db=60.0, sr=DEV_SR)
            _rv = reverb.ConvLive(_ir, block=int(args.blocksize))
            _send = float(args.wet_send)
            self.wet_fn = (lambda y: (y + _send * _rv(y)).astype(np.float32))
            # the out-map path computes its own wet, feeding a mono bus back
            # into each channel; the parts are kept here for that
            self._wet_rv, self._wet_send = _rv, _send
        self.gate = None
        self.gate_mic = None
        if int(getattr(args, "mic_gate", 0)):
            # 08-18: mic-level gate — camera never opens, cv2/mediapipe never
            # imported. Output window covers the ~280 ms feedback peak (F26).
            self.gate_mic = MicGate(
                open_db=args.mic_open, frames=args.mic_frames,
                hold_s=args.mic_hold, bleed=args.bleed,
                margin=args.bleed_margin, ramp_s=float(args.gate_ramp),
                sr=DEV_SR, hop_s=self.HOP / c0.in_sr,
                out_win=int(np.ceil(0.28 * DEV_SR / args.blocksize)))
            self.gate = self.gate_mic
        elif int(getattr(args, "mouth_gate", 0)):
            import mouth_gate
            self.gate = mouth_gate.MouthGate(
                cam=int(args.cam), sing_open=float(args.sing_open),
                sing_hold=float(args.sing_hold), frames=int(args.gate_frames),
                grace=float(args.gate_grace), ramp_s=float(args.gate_ramp),
                fail_open=(args.gate_fail == "open"), sr=DEV_SR,
                log=lambda s: print(s, file=sys.stderr, flush=True)).start()
        self._stream = None
        # xrun accounting OUT of the callback (DEBUG_PLAYBOOK 08-05: printing inside the audio
        # callback is itself a drop inducer). _cb only bumps the counter + keeps the last flags
        # object (reference assignment, no formatting); main() reports on its own clock.
        self.xruns = 0
        self.xrun_last = None

    def _ent_arm(self):
        """--entrance: back to the opening — converted voices silent, waiting for someone to sing."""
        self._ent_t = None          # samples since the sequence started; None = armed, not started
        self._ent_hot = 0           # consecutive samples over threshold (arming)
        self._ent_quiet = 0         # consecutive samples under threshold (re-arming)
        self._ent_cal = DEV_SR      # samples of room-measuring before it can trigger (see _ent_step)
        self._ent_floor = None      # tracked input noise floor (RMS); None = not measured yet
        self.ent_gain = [0.0 if self.args.entrance else 1.0] * len(self.voices)

    def ent_state(self):
        """--entrance status for the UI (None = mode off). armed = still waiting for singing; up = how
        many voices have faded in so far; trig = the input level that starts it (drawn on the meter)."""
        if not self.args.entrance:
            return None
        return {"armed": self._ent_t is None, "up": round(float(sum(self.ent_gain)), 2),
                "n": len(self.voices), "trig": round(float(self.ent_trig), 4)}

    def _ent_step(self, mono, frames):
        """--entrance state machine, one call per callback. Sets self.ent_gain.
        "Singing" is measured RELATIVE to the noise floor, not as a fixed RMS: measured floors differ
        by 16x between mics (RØDE AI-Micro 0.0008, AirPods 0.0137), so any fixed number is either
        permanently triggered on one and unreachable on the other."""
        rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
        if self._ent_cal > 0:                # first second after arming: measure the room, never trigger.
            self._ent_cal -= frames          # taking the MINIMUM, so a cough during it doesn't deafen us
            self._ent_floor = rms if self._ent_floor is None else min(self._ent_floor, rms)
            return
        if self._ent_t is None:              # track the floor only while armed — during the sequence the
            if rms < self._ent_floor:        # singing itself would drag it up and break the next re-arm
                self._ent_floor = rms                              # falls instantly to a quieter room
            else:
                self._ent_floor += (rms - self._ent_floor) * 5e-4  # rises over ~20 s, so singing barely moves it
        self.ent_trig = max(self.args.entrance_thresh, self._ent_floor * self.args.entrance_over)
        loud = rms >= self.ent_trig
        if self._ent_t is None:                       # armed: wait for sustained singing
            self._ent_hot = self._ent_hot + frames if loud else 0
            if self._ent_hot < self.ENT_HOLD:
                return
            self._ent_t = 0                           # start the fade at zero — crediting the hold time
                                                      # would drop V1 in at half gain, an audible step
        else:
            self._ent_t += frames
            self._ent_quiet = 0 if loud else self._ent_quiet + frames
            if self._ent_quiet >= self.ENT_REARM:     # next person gets the same opening
                self._ent_arm()
                return
        self.ent_gain = [min(1.0, max(0.0, (self._ent_t - i * self.ENT_STAG) / self.ENT_FADE))
                         for i in range(len(self.voices))]

    def _cb(self, indata, outdata, frames, tinfo, status):
        if status:
            self.xruns += 1          # report from main(): no I/O on the audio thread
            self.xrun_last = status
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if self.args.entrance:
            self._ent_step(mono, frames)
        if self.dry_on:
            self.dry_hist = np.concatenate([self.dry_hist, mono])
        x16 = np.asarray(self.in_rs.resample_chunk(mono), dtype=np.float32)
        self.in16 = np.concatenate([self.in16, x16])
        while len(self.in16) >= self.HOP:
            hop = self.in16[:self.HOP]; self.in16 = self.in16[self.HOP:]
            if self.gate_mic is not None:
                self.gate_mic.note_input_hop(hop)
            # process EVERY voice (keep warm: converter + resampler state stays live → no glitch on
            # re-enable), then sum only the ON voices with constant-power gain over the on-count.
            y48s = []
            self._dry24 = np.concatenate([self._dry24,                              # kept warm; used by --naive
                                          np.asarray(self.naive_rs.resample_chunk(hop), dtype=np.float32)])
            if len(self._dry24) >= self.OUT_HOP:                                    # emit a fixed block
                dry24, self._dry24 = self._dry24[:self.OUT_HOP], self._dry24[self.OUT_HOP:]
            else:                                                                   # first hop only
                dry24 = np.concatenate([np.zeros(self.OUT_HOP - len(self._dry24), dtype=np.float32), self._dry24])
                self._dry24 = np.zeros(0, dtype=np.float32)
            target = 1.0 if bool(getattr(self.args, "naive", False)) else 0.0
            ab = None
            if self._ab != target:                                   # ramp only while switching
                n = len(dry24)
                ab = np.clip(self._ab + np.arange(1, n + 1) * (self._ab_step if target > self._ab else -self._ab_step),
                             0.0, 1.0).astype(np.float32)
                self._ab = float(ab[-1])
            f0_shared = None      # Task 2.4 finally wired: voice 0 computes f0, peers reuse.
            for vi, (conv, _harm) in enumerate(self.voices):   # +1 inference pass per voice (even if off)
                y24 = conv.process_frame(hop, f0_override=f0_shared)
                if vi == 0 and conv.harmonizer.enabled:
                    # only reuse an estimate that was actually computed — if voice 0 isn't
                    # harmonizing, _last_f0 is a constant 0.0, not a measurement. A computed
                    # 0.0 (unvoiced frame) IS forwarded: peers would measure the same 0.0.
                    f0_shared = float(getattr(conv, "_last_f0", 0.0))
                # Baseline: same shift the engine just decided (post-glide), applied to the singer's own
                # audio instead of re-synthesising a voice. process_frame still runs either way, so f0,
                # the harmony decision and the telemetry are identical between the two modes.
                # ALWAYS run the shifter — it carries a one-window history, and a cold one plays ~100 ms
                # of stale buffer on the first hop after a switch (that was an audible click).
                yn = self.shifters[vi].process(dry24, float(getattr(conv, "_cur_shift", 0.0)))
                if self._ab == 1.0 and ab is None:
                    y24 = yn
                elif ab is not None:                          # equal-power crossfade during the switch
                    k = min(len(y24), len(yn), len(ab))
                    g = ab[:k]
                    y24 = (y24[:k] * np.cos(g * (np.pi / 2)) + yn[:k] * np.sin(g * (np.pi / 2))).astype(np.float32)
                y48s.append(np.asarray(self.out_rs[vi].resample_chunk(np.ascontiguousarray(y24, dtype=np.float32)), dtype=np.float32))
            for vi in range(len(self.voices)):   # --voice-delay: push this hop, pop the hop from N s ago
                f = self.delay_fifo[vi]
                if f is not None:
                    buf = np.concatenate([f, y48s[vi]])
                    k = len(y48s[vi])
                    y48s[vi], self.delay_fifo[vi] = buf[:k], buf[k:]
            if self.gate is not None:
                # Mouth gate on the CONVERTED voices only — the dry You is the singer's
                # own voice and stays in the PA. Applied here (after --voice-delay, before
                # routing) so the mono and --out-map paths are both covered and the ramp
                # advances exactly once per hop. Ramp, never a hard cut (a cut clicks).
                gr = self.gate.ramp(max(len(v) for v in y48s))
                y48s = [v * gr[:len(v)] for v in y48s]
            if self.out_map is not None:
                # per-voice→channel routing (generalises the retired --split-lr experiment; same
                # per-group constant-power maths).
                for c in range(self.n_out):
                    gon = [i for i in range(len(self.voices)) if self.out_map[i] == c and self.voice_on[i]]
                    if gon:
                        L = min(len(y48s[i]) for i in gon)
                        m = np.zeros(L, dtype=np.float32)
                        for i in gon:
                            g = self.ent_gain[i]                 # 1.0 unless --entrance is fading it in
                            m += y48s[i][:L] if g == 1.0 else y48s[i][:L] * g
                        w = sum(self.ent_gain[i] for i in gon)   # see the mono path: weighted by presence
                        if w > 1.0:
                            m *= 1.0 / np.sqrt(w)
                        if self.choir_gain != 1.0:
                            m = np.clip(m * self.choir_gain, -1.0, 1.0)
                    else:
                        m = np.zeros(len(y48s[0]), dtype=np.float32)
                    self.out48_ch[c] = np.concatenate([self.out48_ch[c], m])
                continue
            on = [i for i in range(len(self.voices)) if self.voice_on[i]]
            if on:
                L = min(len(y48s[i]) for i in on)
                mix48 = np.zeros(L, dtype=np.float32)
                for i in on:
                    g = self.ent_gain[i]                         # 1.0 unless --entrance is fading it in
                    mix48 += y48s[i][:L] if g == 1.0 else y48s[i][:L] * g
                # constant-power over the ON voices — but weighted by how much of each voice is actually
                # present, so a voice entering alone is a whole voice against the (undimmed) dry You
                # instead of a quarter of one. Steady state w == len(on) → the original scaling.
                w = sum(self.ent_gain[i] for i in on)
                if w > 1.0:
                    mix48 *= 1.0 / np.sqrt(w)
                if self.choir_gain != 1.0:          # choir/you balance; 1.0 → untouched mix
                    mix48 = np.clip(mix48 * self.choir_gain, -1.0, 1.0)   # +18 dB can hit the rail
            else:
                mix48 = np.zeros(len(y48s[0]), dtype=np.float32)   # all off → silence, keep timing
            self.out48 = np.concatenate([self.out48, mix48])
        n = frames
        if self.out_map is not None:
            dry = None
            if self.dry_on and len(self.dry_hist) >= n:
                dry = self.dry_hist[:n]; self.dry_hist = self.dry_hist[n:]
                if self.you_gain != 1.0:             # You/choir balance; 1.0 → untouched
                    dry = dry * self.you_gain
            chans = []
            for c in range(self.n_out):
                y = np.zeros(n, dtype=np.float32)
                t = min(n, len(self.out48_ch[c])); y[:t] = self.out48_ch[c][:t]; self.out48_ch[c] = self.out48_ch[c][t:]
                chans.append(y)
            if getattr(self, "_wet_rv", None) is not None:
                # Audit fix, 2026-08-18: the out-map path used to skip the
                # choir reverb entirely, so turning on channel routing silently
                # removed the certified reverb. The wet layer now runs through a
                # mono bus, the sum of all converted parts, fed back into every
                # channel, which is the diffuse-field meaning of [wet] in bank.
                # The dry voice still stays out of the reverb.
                wet = (self._wet_send
                       * self._wet_rv(np.sum(chans, axis=0))).astype(np.float32)
                chans = [y + wet for y in chans]
            for c, y in enumerate(chans):
                if dry is not None and self.dry_ch in (-1, c):
                    y = y + dry
                if self.out_gain != 1.0:
                    y = np.clip(y * self.out_gain, -1.0, 1.0)
                outdata[:, c] = y
            if self.gate_mic is not None:    # feedback floor sees the loudest channel
                self.gate_mic.note_output(np.max(np.abs(outdata), axis=1))
            return
        y = np.zeros(n, dtype=np.float32)
        take = min(n, len(self.out48)); y[:take] = self.out48[:take]; self.out48 = self.out48[take:]
        if self.wet_fn is not None:          # 2026-08-18: choir reverb, converted parts only
            y = self.wet_fn(y)
        if self.dry_on and len(self.dry_hist) >= n:
            d = self.dry_hist[:n]; self.dry_hist = self.dry_hist[n:]   # oldest n = delayed You
            y = y + (d if self.you_gain == 1.0 else d * self.you_gain)
        if self.out_gain != 1.0:                     # master gain (low-power exciter drive); clip guards the DAC
            y = np.clip(y * self.out_gain, -1.0, 1.0)
        outdata[:, 0] = y; outdata[:, 1] = y
        if self.gate_mic is not None:
            self.gate_mic.note_output(y)

    def set_control(self, d: dict):
        """Live control from the UI. Updates args + re-applies mode config to the running conv/harm."""
        for k, v in (d or {}).items():
            if hasattr(self.args, k):
                setattr(self.args, k, v)
        if "no_dry" in d or "dry" in d:
            self.dry_on = (not self.args.no_dry) if "no_dry" in d else bool(d["dry"])
        if "gain" in d:                              # linear master gain from the UI (1.0 = unity)
            self.out_gain = max(0.0, float(d["gain"]))
        if "choir_gain" in d:                        # choir/you balance from the UI (1.0 = original mix)
            self.choir_gain = max(0.0, float(d["choir_gain"]))
        if "you" in d:                               # dry You level (the data contract's `you` key)
            self.you_gain = max(0.0, float(d["you"]))
        if "gate" in d:                              # noise gate on the converted voices (0 = off)
            for (conv, _h) in self.voices:
                conv.output_gate_floor = max(0.0, float(d["gate"]))
        for i in range(len(self.voices)):        # per-voice on/off: keys on1..onN (1-indexed = Voice 1..N)
            k = "on" + str(i + 1)
            if k in d:
                self.voice_on[i] = bool(d[k])
        # Re-applying every voice's config costs ~12 ms — longer than the 10 ms callback period, so it
        # starves the audio thread and the dropout is audible as a click on EVERY control press
        # (measured 2026-07-22: dry 7.8–12.4 ms, naive 12.0–15.3 ms). Toggles that cannot change voice
        # config skip it entirely; every other key keeps the original full re-apply.
        if "entrance" in d:      # on → back to the silent opening; off → every gain straight back to 1.0
            self._ent_arm()
        CHEAP = {"dry", "no_dry", "gain", "choir_gain", "you", "gate", "naive", "entrance"} | {"on" + str(i + 1) for i in range(len(self.voices))}
        if d and set(d).issubset(CHEAP):
            return True
        specs = _voice_specs(self.args)
        for i, ((conv, harm), (st, iv, _m, spk)) in enumerate(zip(self.voices, specs)):  # each voice its own interval + speaker
            configure_voice(conv, harm, self.args, steps=st, interval=iv)
            if int(spk) != self._spk[i]:                  # unchanged → skip the expensive setter
                try:                                      # speaker is a live engine setter (no rebuild)
                    conv.sb.set_target_speaker(int(spk))
                    self._spk[i] = int(spk)
                except Exception:  # noqa: BLE001 — invalid speaker for this model → keep the current one
                    pass
        return True

    def start(self):
        in_dev = pick(self.args.in_name, "input")
        out_dev = pick(self.args.out_name, "output")
        print(f"[solo_min] model={os.path.basename(self.args.model)} mode={self.args.mode} "
              f"dry={self.dry_on} dry_delay={self.args.dry_delay_ms}ms in={in_dev} out={out_dev}", file=sys.stderr)
        self._stream = sd.Stream(samplerate=DEV_SR, blocksize=self.args.blocksize, dtype="float32",
                                 channels=(1, self.n_out), device=(in_dev, out_dev), latency=self.args.latency,
                                 callback=self._cb)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop(); self._stream.close(); self._stream = None


def main():
    args = build_argparser().parse_args()
    if args.list:
        print(sd.query_devices()); return
    eng = SoloEngine(args)
    eng.start()
    print("[solo_min] running (Ctrl-C to stop)", file=sys.stderr)
    last_x = 0
    last_g = None                        # mic gate: print transitions from HERE,
    try:                                 # never from the audio callback (08-05)
        while True:
            time.sleep(0.5)
            g = eng.gate_mic
            if g is not None and g.on != last_g:
                last_g = g.on
                pct = 100.0 * g.n_open / max(1, g.n_hops)
                print(f"[gate] mic {'open' if g.on else 'close'}  lvl {g.lvl:.1f}"
                      f"  floor {g.fl:.1f}  open {pct:.0f}%", file=sys.stderr,
                      flush=True)
            if eng.xruns != last_x:      # xrun report moved OUT of the audio callback
                print(f"[solo_min] xruns {eng.xruns} ({eng.xrun_last})",
                      file=sys.stderr)
                last_x = eng.xruns
    except KeyboardInterrupt:
        eng.stop()
        print(f"\n[solo_min] stopped (xruns {eng.xruns})", file=sys.stderr)


if __name__ == "__main__":
    main()
