"""Solo Choir — MIDI / score export (engine-agnostic).

Turns a take's sung melody + the diatonic SATB harmony into a multi-track MIDI
file. Pure: no torch, no audio device, no Beatrice. Reuses voice_changer.SoloChoir
for the diatonic maths (UNCHANGED). See specs/2026-06-22-midi-export-design.md.
"""
from __future__ import annotations

import struct

from voice_changer.SoloChoir import (diatonic_target_midi, scale_pcs_for,
                                     snap_to_scale, f0_to_midi)


def harmony_note(melody_midi: int, key_root: int, minor: bool,
                 interval_steps: int, octave: int) -> int:
    """A SATB part's MIDI note for one melody note: the in-key diatonic target
    plus the part's octave offset — mirrors the live engine (harmony off the
    snapped note; octave applied via TUNE)."""
    target = diatonic_target_midi(melody_midi, key_root, scale_pcs_for(minor), interval_steps)
    return target + 12 * octave


def _vlq(n: int) -> bytes:
    """MIDI variable-length quantity (delta-times)."""
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _meta(delta: int, mtype: int, data: bytes) -> bytes:
    return _vlq(delta) + b"\xFF" + bytes([mtype]) + _vlq(len(data)) + data


def _track_chunk(events: bytes) -> bytes:
    events = events + _vlq(0) + b"\xFF\x2F\x00"          # end-of-track
    return b"MTrk" + struct.pack(">I", len(events)) + events


def write_midi(out_path, tracks, ppq=480, bpm=120, velocity=80):
    """Type-1 MIDI. tracks = [(name, gm_program, [(start_s,end_s,midi), ...]), ...].
    Free timing: bpm ONLY scales seconds->ticks (no quantization); notes stay at
    their real onsets and a notation app meters them on import."""
    tps = ppq * bpm / 60.0                               # ticks per second
    chunks = []
    tempo = int(round(60_000_000 / bpm))
    meta = _meta(0, 0x03, b"Solo Choir") + _meta(0, 0x51, struct.pack(">I", tempo)[1:])
    chunks.append(_track_chunk(meta))                    # track 0: name + tempo
    for ch, (name, program, notes) in enumerate(tracks):
        ev = bytearray()
        ev += _meta(0, 0x03, name.encode("ascii", "replace"))
        ev += _vlq(0) + bytes([0xC0 | (ch & 0x0F), program & 0x7F])   # program change
        pairs = []                                       # (tick, is_on, midi)
        for (s, e, m) in notes:
            pairs.append((int(round(s * tps)), 1, int(m)))
            pairs.append((int(round(e * tps)), 0, int(m)))
        pairs.sort(key=lambda x: (x[0], x[1]))           # note-off before note-on at same tick
        last = 0
        for (tick, on, m) in pairs:
            ev += _vlq(tick - last) + bytes([(0x90 if on else 0x80) | (ch & 0x0F),
                                             m & 0x7F, velocity if on else 0])
            last = tick
        chunks.append(_track_chunk(bytes(ev)))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), ppq)
    with open(out_path, "wb") as f:
        f.write(header + b"".join(chunks))


def wav_to_notes(wav_path, key_root=0, minor=False, voiced_floor_hz=1.0, min_note_s=0.09):
    """Detect a monophonic melody's note events from a mono WAV via librosa.pyin.
    Returns [(start_s, end_s, midi_int), ...]: consecutive same-pitch frames merge
    into a note; unvoiced gaps become rests; notes shorter than min_note_s drop.

    Each frame is snapped to the key's scale (snap_to_scale) BEFORE merging, so a
    slide's chromatic passing tones collapse into their in-key neighbours (a sung
    "do→re" portamento stops emitting a spurious do# between them) and the whole
    score stays in-key, consistent with the SATB harmony."""
    import numpy as np
    import librosa
    scale_pcs = scale_pcs_for(minor)
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    hop = 512
    f0, voiced, _ = librosa.pyin(y, sr=sr, hop_length=hop,
                                 fmin=librosa.note_to_hz("C2"),
                                 fmax=librosa.note_to_hz("C6"))
    t_per = hop / sr
    frames = []                                          # in-key MIDI per frame, or None
    for i, hz in enumerate(f0):
        if voiced[i] and not np.isnan(hz) and hz > voiced_floor_hz:
            frames.append(snap_to_scale(int(round(f0_to_midi(float(hz)))), key_root, scale_pcs))
        else:
            frames.append(None)
    notes, run_note, run_start = [], None, 0
    for i, m in enumerate(frames + [None]):              # trailing None flushes the last run
        if m != run_note:
            if run_note is not None:
                notes.append((run_start * t_per, i * t_per, run_note))
            run_note, run_start = m, i
    return [(s, e, m) for (s, e, m) in notes if (e - s) >= min_note_s]
