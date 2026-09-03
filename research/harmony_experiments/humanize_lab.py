"""f0-contour humanize lab: the autotune feel lives in the TARGET RECIPE.

Harry (07-24 live): every target-f0 mouth (WORLD, ddsp30k, ddspgirl) shares
an "auto-tune" quality -- the common factor is the synthetic f0 contour
(30 ms one-pole porta + fixed 5 Hz/0.12 st sine vibrato + dead-flat
sustains), not the timbre engine. This lab renders contour variants through
the SAME WORLD mouth on the SAME take + brain line for a blind ranking:

  H1 stochastic  -- vibrato onset ramp (delay + grow-in per note), slow
                    rate wobble, 1/f-ish micro-drift. No dependence on his
                    live pitch.
  H2 transplant  -- his own micro-deviation (cents from HIS rounded note,
                    clamped, scaled ALPHA) grafted onto the target note.
                    Note placement stays target-式 (F16); only the micro
                    texture is his. No synthetic vibrato.
  H3 hybrid      -- H2 at lower alpha + H1's onset-ramped vibrato at half
                    depth + light drift.

Run (retraining venv):
  venv/bin/python humanize_lab.py <take.wav> out/angel_v2_world2_notes.json
Outputs out/hlab_{h1,h2,h3}_{angel,mix}.wav (control = angel_v2_world2_target).
"""

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pyworld

HERE = Path(__file__).parent
FRAME_MS = 5.0
PORTA = 0.15
SEED = 20260724


def analyse(mic, sr):
    """WORLD analysis + borrow-frame policy, verbatim from world_mouth."""
    f0_mic, t = pyworld.harvest(mic, sr, frame_period=FRAME_MS)
    sp = pyworld.cheaptrick(mic, f0_mic, t, sr)
    ap = pyworld.d4c(mic, f0_mic, t, sr)
    voiced = f0_mic > 0
    apm = ap.mean(axis=1)
    midi_f0 = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0_mic, 1) / 440), np.nan)
    med = np.copy(midi_f0)
    for k in np.flatnonzero(voiced):
        med[k] = np.nanmedian(midi_f0[max(0, k - 4) : k + 5])
    octave_err = voiced & (np.abs(midi_f0 - med) > 6)
    good = voiced & (apm < 0.8) & ~octave_err
    if not good.any():
        good = voiced
    first_good = np.flatnonzero(good)[0]
    idx = np.zeros(len(f0_mic), dtype=int)
    last = first_good
    for k in range(len(f0_mic)):
        if good[k]:
            last = k
        idx[k] = last
    return f0_mic, t, sp[idx], np.clip(ap[idx], 0.0, 1.0), midi_f0


def note_per_frame(notes, t, sr, step_samps):
    out = np.full(len(t), -1.0)
    for k in range(len(t)):
        tick = min(int(t[k] * sr / step_samps), len(notes) - 1)
        n = notes[tick]
        out[k] = -1.0 if n is None else n
    return out


def porta_track(nframes_note):
    """one-pole portamento toward the (possibly humanized) note track, in Hz"""
    f = np.zeros(len(nframes_note))
    cur = 0.0
    for k, n in enumerate(nframes_note):
        target = 0.0 if n < 0 else 440 * 2 ** ((n - 69) / 12)
        if target > 0:
            cur = target if cur == 0 else cur + (target - cur) * PORTA
            f[k] = cur
        else:
            cur = 0.0
    return f


def seg_starts(nf):
    """frame indices where a new note segment begins (note change or entry)"""
    starts = np.zeros(len(nf), dtype=bool)
    prev = -1.0
    for k, n in enumerate(nf):
        if n >= 0 and n != prev:
            starts[k] = True
        prev = n
    return starts


def onset_age(nf):
    """seconds since the current note segment began, per frame"""
    age = np.zeros(len(nf))
    a = 0.0
    prev = -1.0
    for k, n in enumerate(nf):
        a = 0.0 if (n != prev or n < 0) else a + FRAME_MS / 1000.0
        age[k] = a
        prev = n
    return age


def drift(nframes, rng, sigma_c=9.0, tau_s=0.8):
    """1/f-ish micro-drift in cents: OU random walk, low-passed"""
    a = np.exp(-FRAME_MS / 1000.0 / tau_s)
    x, out = 0.0, np.empty(nframes)
    s = sigma_c * np.sqrt(1 - a * a)
    for k in range(nframes):
        x = a * x + s * rng.standard_normal()
        out[k] = x
    return out


def main():
    take_path, notes_path = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.load(open(notes_path))
    notes, step_samps, sr = meta["notes"], meta["step_samps"], meta["sr"]

    with wave.open(str(take_path)) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    print("WORLD analysis...")
    f0_mic, t, sp2, ap2, midi_mic = analyse(mic, sr)
    nf = note_per_frame(notes, t, sr, step_samps)
    base_hz = porta_track(nf)
    age = onset_age(nf)
    rng = np.random.default_rng(SEED)
    tsec = t

    # his micro-deviation from HIS OWN rounded note, cents; 0 where unvoiced
    dev = np.where(np.isfinite(midi_mic),
                   100.0 * (midi_mic - np.round(midi_mic)), 0.0)
    dev = np.clip(np.nan_to_num(dev), -80, 80)
    # smooth the handoff where voicing flickers (20 ms box)
    k5 = np.ones(4) / 4
    dev = np.convolve(dev, k5, "same")

    def vib_ramped(depth_st, delay=0.25, grow=0.6):
        ramp = np.clip((age - delay) / grow, 0, 1)
        rate = 5.0 + 0.45 * np.sin(2 * np.pi * 0.31 * tsec + 1.7)
        phase = np.cumsum(2 * np.pi * rate * FRAME_MS / 1000.0)
        return depth_st * ramp * np.sin(phase)  # semitones

    variants = {
        "h1": 2.0 ** ((vib_ramped(0.12) + drift(len(nf), rng) / 100.0) / 12.0),
        "h2": 2.0 ** ((0.45 * dev / 100.0) / 12.0),
        "h3": 2.0 ** ((0.30 * dev / 100.0 + vib_ramped(0.06)
                       + 0.6 * drift(len(nf), rng, sigma_c=6.0) / 100.0) / 12.0),
    }

    a_up, a_dn = 1 - np.exp(-FRAME_MS / 40.0), 1 - np.exp(-FRAME_MS / 150.0)
    for name, mul in variants.items():
        f0_new = np.where(base_hz > 0, base_hz * mul, 0.0)
        print(f"synthesize {name}...")
        angel = pyworld.synthesize(np.ascontiguousarray(f0_new), sp2, ap2, sr,
                                   frame_period=FRAME_MS)
        angel = angel[: len(mic)] if len(angel) >= len(mic) else np.pad(
            angel, (0, len(mic) - len(angel)))
        tgt = (f0_new > 0).astype(float)
        g, env = 0.0, np.empty(len(tgt))
        for k in range(len(tgt)):
            g += (tgt[k] - g) * (a_up if tgt[k] > g else a_dn)
            env[k] = g
        hop = int(sr * FRAME_MS / 1000)
        gate = np.repeat(env, hop)[: len(angel)]
        gate = np.pad(gate, (0, len(angel) - len(gate)),
                      constant_values=gate[-1] if len(gate) else 0)
        angel *= gate
        for suffix, sig in [("angel", angel), ("mix", 0.5 * mic + 0.8 * angel)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(str(HERE / "out" / f"hlab_{name}_{suffix}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
        print(f"wrote out/hlab_{name}_angel.wav, out/hlab_{name}_mix.wav")


if __name__ == "__main__":
    main()
