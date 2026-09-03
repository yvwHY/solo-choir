"""Steps 4+5: the live loop. Mic -> YIN -> 16th grid -> brain -> sine alto out.

Live mode (headphones strongly recommended -- speakers feed back into YIN):
    venv/bin/python live.py --bpm 70 --bars 16
File mode (same pipeline, wav in, wav out -- how this repo self-tests):
    venv/bin/python live.py --file data/sop_only.wav --truth 0

The brain was trained in C major / A minor: sing in C for sensible harmony.
Auto-octave: your register is aligned into soprano range (MIDI 60-84) using
the first few voiced notes, so singing an octave down is fine.
"""

import argparse
import json
import queue
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import torch

from pitch import SR, PitchTracker, RMS_GATE, hz_to_midi
from train import HarmonyTransformer, CTX

HERE = Path(__file__).parent
REST, HOLD, PITCH_OFFSET, PITCH_LO, PITCH_HI = 0, 1, 2, 36, 84
TEMPERATURE, TOP_K = 0.9, 8


class Brain:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(HERE / "checkpoints" / "best.pt",
                        map_location=self.device, weights_only=True)
        self.model = HarmonyTransformer(ck["vocab"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.ctx = []
        # warm up MPS kernels so the first live tick isn't slow
        with torch.no_grad():
            self.model(torch.zeros(1, 8, dtype=torch.long, device=self.device))

    @torch.no_grad()
    def step(self, sop_token):
        self.ctx.append(sop_token)
        x = torch.tensor([self.ctx[-CTX:]], device=self.device)
        logits = self.model(x)[0, -1] / TEMPERATURE
        if sop_token != REST:
            logits[REST] = float("-inf")  # while you sing, the angel sings
        k = torch.topk(logits, TOP_K)
        alto = int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])
        self.ctx.append(alto)
        return alto


class VoiceToTokens:
    """Rounded MIDI stream -> REST/HOLD/onset tokens, with auto octave shift."""

    def __init__(self):
        self.prev = None
        self.shift = None
        self.warmup = []

    def token(self, midi):
        if midi is None:
            self.prev = None
            return REST
        if self.shift is None:
            self.warmup.append(midi)
            if len(self.warmup) < 6:
                return REST
            med = float(np.median(self.warmup))
            # aim the median at 65, not the soprano mean 71: leaves headroom
            # above so higher phrases don't overflow the training range
            self.shift = int(12 * round((65 - med) / 12))
            print(f"[auto-octave] shift = {self.shift:+d} semitones")
        m = midi + self.shift
        while m > PITCH_HI:  # fold strays back in by octaves instead of muting
            m -= 12
        while m < PITCH_LO:
            m += 12
        if m % 12 in (1, 3, 6, 8, 10):
            m -= 1  # snap chromatic blips to the C-major scale the brain knows
        if not (PITCH_LO <= m <= PITCH_HI):
            self.prev = None
            return REST
        tok = HOLD if m == self.prev else m - PITCH_LO + PITCH_OFFSET
        self.prev = m
        return tok


def token_to_midi(tok, prev):
    if tok >= PITCH_OFFSET:
        return tok - PITCH_OFFSET + PITCH_LO
    return prev if tok == HOLD else None


class SineVoice:
    """Phase-continuous sine with 5 ms ramps; render(n) yields n samples."""

    def __init__(self, gain=0.25):
        self.freq = 0.0
        self.target = 0.0
        self.phase = 0.0
        self.amp = 0.0
        self.gain = gain

    def set_midi(self, midi):
        self.target = 0.0 if midi is None else 440 * 2 ** ((midi - 69) / 12)

    def render(self, n):
        out = np.empty(n)
        ramp = self.gain / (0.005 * SR)
        for i in range(n):
            if self.target > 0:
                self.freq = self.target
                self.amp = min(self.gain, self.amp + ramp)
            else:
                self.amp = max(0.0, self.amp - ramp)
            self.phase += 2 * np.pi * self.freq / SR
            out[i] = self.amp * np.sin(self.phase)
        self.phase %= 2 * np.pi
        return out


class ShiftVoice:
    """Angel sings its notes with grains of YOUR live voice (whammy-style
    dual-tap ring buffer). While you breathe, the buffer freezes and the last
    vowel loops, so the angel can hold notes through your rests.

    Output register: the brain thinks in soprano space, but we render the
    angel relative to your actual octave (target minus auto-octave shift),
    keeping the transposition ratio small and the formant damage low.
    """

    def __init__(self, tracker, gain=0.9):
        self.tracker = tracker
        self.L = 1 << 16
        self.ring = np.zeros(self.L)
        self.w = 0
        self.N = 4096  # grain window (~93 ms)
        self.phase = 0.0
        self.vw = None  # virtual write head: advances 1/sample during render
        self.ratio = 1.0
        self.target_hz = 0.0
        self.amp = 0.0
        self.gain = gain
        self.frozen = False

    def set_midi(self, midi):
        if midi is None:
            self.target_hz = 0.0
            return
        # register guard: fold the note to within a 6th of YOUR current voice,
        # so the ratio stays in the sweet zone and never hits its clamps
        src_midi = hz_to_midi(float(np.median(self.src_hist))) if getattr(self, "src_hist", None) else 55.0
        while midi - src_midi < -9:
            midi += 12
        while midi - src_midi > 9:
            midi -= 12
        self.target_hz = 440 * 2 ** ((midi - 69) / 12)

    def push_mic(self, block):
        if self.tracker.latest is None:
            self.frozen = True  # hold the last vowel; don't write breath noise
            return
        self.frozen = False
        n = len(block)
        idx = (self.w + np.arange(n)) % self.L
        self.ring[idx] = block
        self.w = (self.w + n) % self.L

    def render(self, n):
        # Reference = rolling median of your f0 (~0.75 s): tracking flicker is
        # absorbed, your true vibrato still rides through on the audio itself.
        src = self.tracker.latest_hz
        if src:
            self.src_hist = (getattr(self, "src_hist", []) + [src])[-32:]
        if self.target_hz > 0 and getattr(self, "src_hist", None):
            self.goal = float(np.clip(
                self.target_hz / float(np.median(self.src_hist)), 0.5, 2.0))
        # fully vectorized: the per-sample python loop overflowed the callback
        ramp = self.gain / (0.005 * SR)
        steps = np.arange(1, n + 1)
        if self.target_hz > 0:
            amp = np.minimum(self.gain, self.amp + ramp * steps)
        else:
            amp = np.maximum(0.0, self.amp - ramp * steps)
        self.amp = float(amp[-1])
        if amp.max() <= 0:
            self.vw = None  # resync to the freshest audio at next note-on
            return np.zeros(n)
        if self.vw is None:
            self.vw = float(self.w)
        # one-pole glide toward the goal ratio, exact per-sample trajectory
        goal = getattr(self, "goal", 1.0)
        ratios = goal + (self.ratio - goal) * (1.0 - 0.002) ** steps
        self.ratio = float(ratios[-1])
        # designed vibrato: 5 Hz, ±0.12 semitone -- musical intonation that
        # doesn't depend on how clean the pitch tracking happens to be
        vt = (getattr(self, "vib_ph", 0) + steps) / SR
        self.vib_ph = getattr(self, "vib_ph", 0) + n
        ratios = ratios * 2.0 ** (0.12 * np.sin(2 * np.pi * 5.0 * vt) / 12.0)
        if not self.frozen:
            vws = self.vw + steps.astype(np.float64)
            phases = (self.phase + np.cumsum(1.0 - ratios)) % self.N
        else:
            vws = np.full(n, self.vw)
            phases = (self.phase - np.cumsum(ratios)) % self.N
        self.vw = float(vws[-1])
        self.phase = float(phases[-1])
        s = np.zeros(n)
        for k in (0.0, 0.5):
            d = (phases + k * self.N) % self.N
            pos = (vws - 1.0 - d) % self.L
            j = pos.astype(np.int64)
            frac = pos - j
            sample = self.ring[j] * (1 - frac) + self.ring[(j + 1) % self.L] * frac
            s += sample * (0.5 - 0.5 * np.cos(2 * np.pi * d / self.N))
        return amp * s


def run_file(path, truth_idx, bpm):
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1, "mono wav please"
        sr_in = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    if sr_in != SR:
        n_out = int(len(audio) * SR / sr_in)
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                          np.arange(len(audio)), audio)

    brain, v2t, tracker = Brain(), VoiceToTokens(), PitchTracker()
    step_samps = int((60.0 / bpm) * 0.25 * SR)
    n_ticks = len(audio) // step_samps
    sop_toks, alto_toks = [], []
    pos = 0
    for _ in range(n_ticks):
        tracker.push(audio[pos : pos + step_samps])
        pos += step_samps
        s = v2t.token(tracker.latest)
        sop_toks.append(s)
        alto_toks.append(brain.step(s))

    # render alto on top of the input
    voice, out = SineVoice(), np.zeros(len(audio))
    prev = None
    for t, tok in enumerate(alto_toks):
        m = token_to_midi(tok, prev)
        prev = m
        voice.set_midi(m)
        seg = voice.render(step_samps)
        out[t * step_samps : t * step_samps + len(seg)] = seg
    mix = 0.6 * audio + out
    mix /= max(1.0, np.abs(mix).max())
    outpath = HERE / "data" / "live_filemode.wav"
    with wave.open(str(outpath), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    print(f"wrote {outpath}")

    if truth_idx is not None:
        d = np.load(HERE / "data" / "tokens.npz")
        true_sop = [d[k] for k in d.files][truth_idx][:, 0]
        n = min(len(true_sop), len(sop_toks))
        got = np.array(sop_toks[:n]); ref = true_sop[:n]
        # compare sounding pitch per tick (token-level HOLD/onset offsets are timing noise)
        def sounding(col):
            out_, cur = np.full(len(col), -1), -1
            for i, tk in enumerate(col):
                if tk >= PITCH_OFFSET: cur = tk
                elif tk == REST: cur = -1
                out_[i] = cur
            return out_
        sg, sref = sounding(got), sounding(ref)
        # auto-octave re-registers the input by v2t.shift; compensate before comparing
        sref = np.where(sref >= 0, sref + (v2t.shift or 0), sref)
        acc = np.mean(sg == sref)
        print(f"pitch-tracking accuracy vs ground truth (sounding pitch/tick): {acc:.1%}")


class Conductor:
    """Drives a running vcclient (Beatrice) instead of synthesizing audio:
    your live voice goes mic -> vcclient -> headphones, and we steer its
    pitch_shift knob per tick. Rests are done by muting vcclient's output."""

    BASE = "http://127.0.0.1:18000"

    def __init__(self, slot=3):
        import requests
        self.s = requests.Session()
        self.slot = slot
        self.cur = None
        self.muted = None
        # the server only honors a FULL slot-object PUT, and Beatrice reads
        # pitch_shifts[dst_id], not the scalar -- learned the hard way
        self.obj = self.s.get(f"{self.BASE}/api/slot-manager/slots/{slot}", timeout=5).json()

    def set(self, shift):
        try:
            if shift is None:
                if self.muted is not True:
                    self.s.put(f"{self.BASE}/api/configuration-manager/configuration",
                               json={"audio_output_device_gain": 0.0}, timeout=2)
                    self.muted = True
                return
            if self.muted is not False:
                self.s.put(f"{self.BASE}/api/configuration-manager/configuration",
                           json={"audio_output_device_gain": 1.0}, timeout=2)
                self.muted = False
            shift = int(round(shift))
            if shift != self.cur:
                self.obj["pitch_shift"] = shift
                dst = self.obj.get("dst_id", 0)
                shifts = self.obj.get("pitch_shifts") or [0]
                while len(shifts) <= dst:
                    shifts.append(0)
                shifts[dst] = shift
                self.obj["pitch_shifts"] = shifts
                self.s.put(f"{self.BASE}/api/slot-manager/slots/{self.slot}",
                           json=self.obj, timeout=2)
                self.cur = shift
        except Exception as e:
            print(f"[conductor] API error: {e}", flush=True)


def run_replay(path, bpm, tag="replay"):
    """Offline re-run of the FULL live pipeline over a recorded mic stem
    (data/take.wav channel L, or any mono wav). Lets us iterate on the voice
    engine against Harry's real singing without him re-singing every attempt.
    Writes data/<tag>_angel.wav (stem) and data/<tag>_mix.wav (mic + angel).
    """
    with wave.open(str(path)) as w:
        sr_in, nch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    a = raw.reshape(-1, nch)[:, 0] if nch > 1 else raw
    if sr_in != SR:
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * SR / sr_in)),
                      np.arange(len(a)), a)

    brain, v2t, tracker = Brain(), VoiceToTokens(), PitchTracker()
    voice = ShiftVoice(tracker)
    step_samps = int((60.0 / bpm) * 0.25 * SR)
    legato = {"gap": 0, "note": None}
    prev = None
    next_tick = 0
    out = []
    for i in range(0, len(a) - 1024, 1024):
        blk = a[i : i + 1024]
        tracker.push(blk)
        voice.push_mic(blk)
        while next_tick < i + 1024:
            s = v2t.token(tracker.latest)
            al = brain.step(s)
            m = token_to_midi(al, prev)
            prev = m
            if m is None and legato["note"] is not None and legato["gap"] < 1:
                legato["gap"] += 1
                m_voice = legato["note"]
            else:
                legato["gap"] = 0
                legato["note"] = m
                m_voice = m
            voice.set_midi(None if m_voice is None else m_voice - (v2t.shift or 0))
            next_tick += step_samps
        out.append(voice.render(1024))
    angel = np.concatenate(out)
    mic = a[: len(angel)]
    for name, sig in [(f"{tag}_angel", angel), (f"{tag}_mix", 0.5 * mic + angel)]:
        peak = max(1.0, np.abs(sig).max())
        with wave.open(str(HERE / "data" / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((sig / peak * 32767).astype(np.int16).tobytes())
    print(f"wrote data/{tag}_angel.wav, data/{tag}_mix.wav")


def run_live(bpm, bars, in_dev, out_dev, voice_mode="shift"):
    import sounddevice as sd

    brain, v2t, tracker = Brain(), VoiceToTokens(), PitchTracker()
    step_samps = int((60.0 / bpm) * 0.25 * SR)
    total_ticks = bars * 16
    conductor = Conductor() if voice_mode == "conduct" else None
    voice = ShiftVoice(tracker) if voice_mode == "shift" else SineVoice()
    click_env = np.exp(-np.arange(int(0.01 * SR)) / (0.002 * SR))
    tick_q = queue.Queue()
    state = {"sample": 0, "start": None, "click": None, "done": False}
    log = {"sop": [], "alto": []}
    rec = {"mic": [], "angel": []}  # both stems archived for offline diagnosis

    def audio_cb(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        tracker_in.put(indata[:, 0].copy())
        if isinstance(voice, ShiftVoice):
            voice.push_mic(indata[:, 0])
        block = voice.render(frames)
        rec["mic"].append(indata[:, 0].copy())
        rec["angel"].append(block.copy())  # pre-click: pure angel stem
        s0 = state["sample"]
        quarter = step_samps * 4
        # voice-activated start: click idles until you sing, then bars begin
        # on the next quarter so the remote-launch race can't eat the session
        if state["start"] is None and tracker.latest is not None:
            state["start"] = (s0 // quarter + 1) * quarter
            print("voice detected -- bar 1 starts on the next beat", flush=True)
        for i in range(frames):
            s = s0 + i
            if s % quarter == 0:
                state["click"] = (s, 1.4 if s % (quarter * 4) == 0 else 1.0)
            if state["click"]:
                cs, amp = state["click"]
                j = s - cs
                if j < len(click_env):
                    block[i] += 0.25 * amp * click_env[j] * np.sin(2 * np.pi * 1200 * j / SR)
                else:
                    state["click"] = None
            if state["start"] is not None and s >= state["start"] and (s - state["start"]) % step_samps == 0:
                tick_q.put((s - state["start"]) // step_samps)
        outdata[:, 0] = block
        state["sample"] += frames
        if state["start"] is not None and state["sample"] >= state["start"] + (total_ticks + 2) * step_samps:
            state["done"] = True
            raise sd.CallbackStop

    tracker_in = queue.Queue()

    def pitch_worker():
        while not state["done"]:
            try:
                tracker.push(tracker_in.get(timeout=0.2))
            except queue.Empty:
                pass

    def brain_worker():
        prev = None
        lock = {"heard": None, "n": 0}
        legato = {"gap": 0, "note": None}
        while not state["done"]:
            try:
                tick = tick_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if tick >= total_ticks:
                voice.set_midi(None)
                continue
            raw = tracker.latest  # capture once: it updates concurrently
            s = v2t.token(raw)
            a = brain.step(s)
            m = token_to_midi(a, prev)
            prev = m
            # legato: ride through single-tick dropouts instead of chopping
            if m is None and legato["note"] is not None and legato["gap"] < 1:
                legato["gap"] += 1
                m_voice = legato["note"]
            else:
                legato["gap"] = 0
                legato["note"] = m
                m_voice = m
            if conductor is not None:
                if m_voice is None or raw is None:
                    conductor.set(None)
                else:
                    conductor.set((m_voice - (v2t.shift or 0)) - raw)
            else:
                # shift voice renders in your own octave: brain-space minus auto-shift
                voice.set_midi(m_voice if (m_voice is None or isinstance(voice, SineVoice))
                               else m_voice - (v2t.shift or 0))
            log["sop"].append(s); log["alto"].append(a)
            # feedback guard: input frozen on the angel's own pitch (mod octave)
            heard = v2t.prev
            if (heard is not None and m is not None
                    and heard == lock["heard"] and (heard - m) % 12 == 0):
                lock["n"] += 1
                if lock["n"] == 16:
                    print("[warning] input is locked to my own output -- "
                          "speakers feeding back? use headphones", flush=True)
            else:
                lock["n"] = 0
            lock["heard"] = heard
            if tick % 4 == 0:
                sung = raw if raw else "--"
                heard = v2t.prev if v2t.prev is not None else "--"
                print(f"bar {tick//16 + 1:2d} beat {(tick%16)//4 + 1} | you: {sung} (brain hears {heard}) | angel: {m}", flush=True)

    threading.Thread(target=pitch_worker, daemon=True).start()
    threading.Thread(target=brain_worker, daemon=True).start()

    print(f"{bpm} bpm, {bars} bars. Click is idling -- start singing (C major) "
          "whenever you're ready and bar 1 begins.", flush=True)
    try:
        with sd.Stream(samplerate=SR, blocksize=1024, channels=1, dtype="float32",
                       device=(in_dev, out_dev), callback=audio_cb):
            while not state["done"]:
                sd.sleep(100)
    finally:
        state["done"] = True  # stop workers even on Ctrl-C / stream errors
        if conductor is not None:
            conductor.set(0)  # unmute vcclient and zero the pitch knob
        (HERE / "data" / "live_session.json").write_text(json.dumps(log))
        if rec["mic"]:
            stems = np.stack([np.concatenate(rec["mic"]), np.concatenate(rec["angel"])], axis=1)
            peak = max(1.0, np.abs(stems).max())
            with wave.open(str(HERE / "data" / "take.wav"), "wb") as w:
                w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((stems / peak * 32767).astype(np.int16).tobytes())
            print("stems (L=mic, R=angel) -> data/take.wav", flush=True)
        print(f"done. {len(log['alto'])} ticks logged -> data/live_session.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="wav in instead of mic (validation mode)")
    ap.add_argument("--truth", type=int, help="chorale index for accuracy check (file mode)")
    ap.add_argument("--bpm", type=float, default=70)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--in-dev", default=None, help="input device name/index (e.g. 'throat')")
    ap.add_argument("--out-dev", default=None, help="output device name/index")
    ap.add_argument("--voice", choices=["shift", "sine", "conduct"], default="shift",
                    help="'shift' = built-in transposer, 'sine' = test tone, "
                         "'conduct' = steer a running vcclient (Beatrice) via its pitch knob")
    ap.add_argument("--replay", type=Path,
                    help="re-run pipeline offline over a recorded mic stem (e.g. data/take.wav)")
    a = ap.parse_args()
    if a.replay:
        run_replay(a.replay, a.bpm)
    elif a.file:
        run_file(a.file, a.truth, a.bpm)
    else:
        run_live(a.bpm, a.bars, a.in_dev, a.out_dev, a.voice)
