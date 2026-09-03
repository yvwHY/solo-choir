"""Design 2: sustain-trigger synchronous angel.

While you sing, the ear + Transformer brain listen every 16th tick; nothing
sounds on short or moving notes. Hold a stable vowel for --hold seconds and
ONE batch-WORLD synthesis (the recipe Harry approved 07-22: harvest f0 +
good-frame borrowing + portamento/vibrato + attack envelope) renders that
very vowel at the brain's harmony note and joins you until you release
(150 ms fade). One note = one batch call: the G28 streaming artifact surface
never opens.

Ear: the validated YIN PitchTracker by default; --ear crepe switches to
torchcrepe tiny (measured parity on real stems, kept as an option -- see
ear.py commit message).

Trigger tuning (no synthesis):
    /opt/anaconda3/envs/vcclient-dev/bin/python sustain.py --scan \
        --file ../../../260722_harmony_brain/data/take.wav
Offline render (ear-gate material):        same command without --scan
Live (headphones; sing in --key, default C):
    /opt/anaconda3/envs/vcclient-dev/bin/python sustain.py --live \
        --in-name 外接麥克風 --out-name 外接耳機
"""

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import pyworld

from ear import CrepeTracker, load_mono
from live import Brain, VoiceToTokens, token_to_midi
from pitch import SR, PitchTracker, hz_to_midi

HERE = Path(__file__).parent
OUT = HERE / "out"
FRAME_MS = 5.0
VIB_HZ, VIB_SEMI = 5.0, 0.12
PORTA = 0.15            # one-pole per 5 ms frame ~ 30 ms glide (world_replay recipe)
FADE_S = 0.15           # release fade when you let go of the note
NOTES = "C C# D D# E F F# G G# A A# B".split()

KEYS = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
        "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10,
        "BB": 10, "B": 11}


def key_offset(name):
    """Key name -> semitones above C; minor keys map to the relative major
    (that is what the C/Am-trained brain actually knows)."""
    name = name.strip().upper().replace("♯", "#").replace("♭", "B")
    minor = name.endswith("M")
    root = KEYS[name[:-1] if minor else name]
    return (root + 3) % 12 if minor else root


def note_name(m):
    m = int(round(m))
    return f"{NOTES[m % 12]}{m // 12 - 1}"


def make_tracker(a):
    return CrepeTracker() if a.ear == "crepe" else PitchTracker()


class SustainDetector:
    """Fires once per held note: the last hold_s seconds were continuously
    voiced and within +/- cents_tol of their median. Stays 'active' through
    vibrato; releases when the voice stops for gap_s or moves > move_semi."""

    def __init__(self, hold_s=0.4, cents_tol=50.0, gap_s=0.15, move_semi=1.5):
        self.hold_s, self.cents_tol = hold_s, cents_tol
        self.gap_s, self.move_semi = gap_s, move_semi
        self.hist = []              # (t, midi_float) while continuously voiced
        self.unvoiced_since = None
        self.active = None          # median midi of the note owning a trigger

    def update(self, t, midi_f):
        if midi_f is None:
            if self.unvoiced_since is None:
                self.unvoiced_since = t
            if t - self.unvoiced_since >= self.gap_s:
                self.hist = []
                if self.active is not None:
                    self.active = None
                    return "release"
            return None
        self.unvoiced_since = None
        if self.active is not None:
            if abs(midi_f - self.active) > self.move_semi:
                self.active = None          # note changed: re-arm immediately
                self.hist = [(t, midi_f)]
                return "release"
            return None
        self.hist.append((t, midi_f))
        self.hist = [(tt, mm) for tt, mm in self.hist if t - tt <= self.hold_s + 0.02]
        if t - self.hist[0][0] + 1e-9 < self.hold_s or len(self.hist) < 8:
            return None
        vals = np.array([mm for _, mm in self.hist])
        med = float(np.median(vals))
        if np.abs(vals - med).max() * 100.0 <= self.cents_tol:
            self.active = med
            return "trigger"
        return None


def run_scan(a):
    """Offline trigger-only pass over a real-voice stem: prints the event
    table so --hold / --cents-tol can be tuned before any synthesis exists."""
    x = load_mono(a.file)
    tracker = make_tracker(a)
    det = SustainDetector(a.hold, a.cents_tol)
    blk = 1024
    events = []
    for i in range(0, len(x) - blk, blk):
        tracker.push(x[i:i + blk])
        t = (i + blk) / SR
        mf = hz_to_midi(tracker.latest_hz) if tracker.latest is not None else None
        ev = det.update(t, mf)
        if ev == "trigger":
            rms = float(np.sqrt(np.mean(x[i + blk - int(a.hold * SR):i + blk] ** 2)))
            events.append([t, det.active, None, rms])
            print(f"{t:7.2f}s  TRIGGER  {note_name(det.active):4s} "
                  f"({det.active:6.2f})  snippet rms {rms:.4f}")
        elif ev == "release" and events and events[-1][2] is None:
            events[-1][2] = t
            print(f"{t:7.2f}s  release  (held {t - events[-1][0]:.2f}s)")
    if events and events[-1][2] is None:
        events[-1][2] = len(x) / SR
    print(f"\n{len(events)} sustained notes; "
          f"durations {[round(e[2] - e[0], 2) for e in events]}")
    assert all(e[3] > 0.005 for e in events), "trigger fired on near-silence!"
    return events


def synth_sustain(snip, target_midi, dur_s):
    """ONE batch-WORLD render: your held vowel re-sung at the harmony note.
    Good frames (voiced, low aperiodicity, near the median pitch) are
    ping-pong tiled to dur_s so the texture keeps evolving -- no static loop
    point; f0 is a fresh continuous contour (portamento-in + designed
    vibrato). Full-context batch synthesis, never streamed (G28 red line)."""
    f0, t = pyworld.harvest(snip, SR, frame_period=FRAME_MS)
    if not (f0 > 0).any():
        return None
    sp = pyworld.cheaptrick(snip, f0, t, SR)
    ap = pyworld.d4c(snip, f0, t, SR)
    apm = ap.mean(axis=1)
    midi_f0 = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
    med = np.nanmedian(midi_f0)
    good = (f0 > 0) & (apm < 0.8) & (np.abs(np.nan_to_num(midi_f0, nan=99.0) - med) <= 2.0)
    gi = np.flatnonzero(good)
    if len(gi) < 8:
        gi = np.flatnonzero(f0 > 0)
    n = int(dur_s * 1000 / FRAME_MS)
    pp = np.concatenate([gi, gi[::-1]])
    idx = pp[np.arange(n) % len(pp)]
    tt = np.arange(n) * FRAME_MS / 1000.0
    tgt = 440.0 * 2 ** ((target_midi - 69) / 12)
    f0n, cur = np.empty(n), 0.0
    for k in range(n):
        cur = tgt if cur == 0.0 else cur + (tgt - cur) * PORTA
        f0n[k] = cur * 2 ** (VIB_SEMI * np.sin(2 * np.pi * VIB_HZ * tt[k]) / 12)
    angel = pyworld.synthesize(f0n, sp[idx], ap[idx], SR, frame_period=FRAME_MS)
    attack = 1.0 - np.exp(-np.arange(len(angel)) / (0.04 * SR))   # 40 ms in
    return angel * attack


def fold_register(m_brain, shift, register, reg_lo):
    """Brain space -> your octave (angel_line recipe from respond.py):
    subtract the auto-octave shift, then fold within [-reg_lo, +9] semitones
    of your recent median so the angel sits beside your actual voice."""
    m = m_brain - (shift or 0)
    ref = float(np.median(register)) if register else 55.0
    while m < ref - reg_lo:
        m += 12
    while m > ref + 9:
        m -= 12
    return m


def fade_out(buf, at, fade_s=FADE_S):
    """In-place linear fade starting at sample `at`; truncates after it."""
    at = max(0, min(len(buf), at))
    f = min(len(buf) - at, int(fade_s * SR))
    if f > 0:
        buf[at:at + f] *= np.linspace(1.0, 0.0, f)
    return buf[:at + f]


def run_render(a):
    """Full offline chain over a real-voice stem. The brain hears every 16th
    tick (counterpoint context stays real); each trigger costs exactly one
    batch synthesis, placed at trigger time + measured synth wall time, so
    the mix previews honest live latency."""
    x = load_mono(a.file)
    key = key_offset(a.key)
    tracker = make_tracker(a)
    det = SustainDetector(a.hold, a.cents_tol)
    brain, v2t = Brain(), VoiceToTokens()
    register, legato, prev = [], {"gap": 0, "note": None}, None
    m_voice = None
    step = int((60.0 / a.bpm) * 0.25 * SR)
    next_tick = 0
    blk = 1024
    angel_track = np.zeros(len(x) + int(a.dur * SR))
    open_ev = None                     # (start_sample, end_limit_sample)
    n_ev, synth_ms = 0, []

    for i in range(0, len(x) - blk, blk):
        tracker.push(x[i:i + blk])
        now = i + blk
        while next_tick < now:         # brain listens on the 16th grid
            raw = tracker.latest
            s = v2t.token(None if raw is None else raw - key)
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
            if raw is not None:
                register.append(raw - key)
            next_tick += step
        mf = hz_to_midi(tracker.latest_hz) if tracker.latest is not None else None
        ev = det.update(now / SR, mf)
        if ev == "trigger" and m_voice is not None:
            snip = x[max(0, now - int(a.hold * SR)):now]
            target = fold_register(m_voice, v2t.shift, register, a.reg_lo) + key
            t0 = time.perf_counter()
            angel = synth_sustain(snip, target, a.dur)
            dt = time.perf_counter() - t0
            if angel is None:
                det.active = None
                continue
            angel *= min(0.9, float(np.abs(snip).max())) / max(1e-9, float(np.abs(angel).max()))
            start = now + int(dt * SR)             # honest live-latency preview
            angel_track[start:start + len(angel)] += angel[:len(angel_track) - start]
            open_ev = (start, start + len(angel))
            n_ev += 1
            synth_ms.append(dt * 1000)
            print(f"{now/SR:7.2f}s  you {note_name(det.active):4s} -> angel "
                  f"{note_name(target):4s}  synth {dt*1000:5.0f} ms  "
                  f"(enters {a.hold + dt:.2f}s after note onset)")
        elif ev == "release" and open_ev is not None:
            s0, s1 = open_ev
            faded = fade_out(angel_track[s0:s1].copy(), now - s0)
            angel_track[s0:s1] = 0.0
            angel_track[s0:s0 + len(faded)] = faded
            open_ev = None

    angel_track = angel_track[:len(x)]
    mix = 0.5 * x + 0.8 * angel_track
    for name, sig in [(f"{a.tag}_mix", mix), (f"{a.tag}_angel", angel_track)]:
        peak = max(1e-9, np.abs(sig).max())
        with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
    print(f"{n_ev} sustained notes -> out/{a.tag}_mix.wav, out/{a.tag}_angel.wav")
    if synth_ms:
        print(f"synth wall ms: mean {np.mean(synth_ms):.0f}  max {np.max(synth_ms):.0f}")
    assert n_ev > 0, "no triggers on this stem -- check --hold/--cents-tol"


def run_live(a):
    raise SystemExit("Task 4 not implemented yet -- use --scan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path,
                    default=HERE / "../../../260722_harmony_brain/data/take.wav")
    ap.add_argument("--scan", action="store_true", help="trigger events only, no synthesis")
    ap.add_argument("--ear", choices=["yin", "crepe"], default="yin",
                    help="pitch tracker (yin = validated default; crepe measured at parity)")
    ap.add_argument("--hold", type=float, default=0.4, help="stable seconds before the angel enters")
    ap.add_argument("--cents-tol", type=float, default=50.0, help="max deviation from median (cents)")
    ap.add_argument("--dur", type=float, default=5.0, help="max sustain length (s)")
    ap.add_argument("--bpm", type=float, default=70, help="brain tick grid")
    ap.add_argument("--key", default="C")
    ap.add_argument("--reg-lo", type=float, default=9.0)
    ap.add_argument("--tag", default="sustain")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.scan:
        run_scan(a)
    elif a.live:
        run_live(a)
    else:
        run_render(a)


if __name__ == "__main__":
    main()
