"""Solo Choir — diatonic harmony logic (engine-agnostic).

Phase 1 implements the music logic exactly as specified in
solo_choir_vcclient_phase1_spec.md. It is deliberately kept free of any
voice-changer / torch coupling so it can be unit-tested in isolation and later
ported from the RVC prototype to the Beatrice v2 engine unchanged.

Per-chunk contract (see SoloChoirHarmonizer.compute_shift):
    raw input f0 contour (Hz, from the model's own pitch extractor)
        -> representative input MIDI note
        -> diatonic target MIDI (root + scale + signed scale-step interval)
        -> transpose in semitones = target - input

The single converted output, shifted by that transpose, is the diatonic
harmony line.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# semitone offsets from the root, one entry per scale degree (index 0..6)
MAJOR_SCALE_PCS = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE_PCS = (0, 2, 3, 5, 7, 8, 10)  # natural minor

# note name -> pitch class (C = 0 ... B = 11)
NOTE_NAME_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def f0_to_midi(f0_hz: float) -> float:
    """m = 69 + 12*log2(f0 / 440). Returns a float (un-rounded)."""
    return 69.0 + 12.0 * math.log2(f0_hz / 440.0)


def scale_pcs_for(is_minor: bool) -> tuple[int, ...]:
    return MINOR_SCALE_PCS if is_minor else MAJOR_SCALE_PCS


def snap_to_scale(m: int, root_pc: int, scale_pcs: tuple[int, ...]) -> int:
    """Snap MIDI note ``m`` to the nearest pitch in the key (nearest scale tone).

    On an exact tie (input is equidistant between two scale tones, e.g. a note
    a semitone away from both neighbours) the lower note is chosen, for
    determinism.
    """
    best_note = m
    best_dist = None
    # candidates within +/- an octave are more than enough to cover the nearest.
    for note in range(m - 7, m + 8):
        if (note - root_pc) % 12 in scale_pcs:
            dist = abs(note - m)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_note = note
    return best_note


def diatonic_target_midi(m: int, root_pc: int, scale_pcs: tuple[int, ...], interval_steps: int) -> int:
    """Diatonic shift of MIDI note ``m`` by ``interval_steps`` scale-steps.

    1. snap m to the nearest scale tone
    2. find its scale-degree index i (0..6) and octave
    3. shift by N steps, carrying octaves (every wrap of 7 steps = +/-12 semitones)
    """
    snapped = snap_to_scale(m, root_pc, scale_pcs)
    rel = snapped - root_pc
    octave = rel // 12
    pc = rel % 12
    i = scale_pcs.index(pc)  # snapped is guaranteed in-scale

    new_index_total = i + interval_steps
    new_degree = new_index_total % 7
    octave_carry = new_index_total // 7  # floor division: correct for negatives

    target = root_pc + scale_pcs[new_degree] + 12 * (octave + octave_carry)
    return target


@dataclass
class SoloChoirHarmonizer:
    """Holds the key/interval and turns a per-chunk f0 contour into a transpose.

    Attributes:
        enabled:        Solo Choir mode on/off.
        key_root:       root pitch class, C=0 .. B=11.
        minor:          True = natural minor, False = major.
        interval_steps: signed diatonic interval in scale-steps. Default -2
                        (a third below). +2 = third above, -5 = sixth below,
                        +5 = sixth above.
        voiced_floor_hz: f0 values at/below this are treated as unvoiced.
    """
    enabled: bool = False
    key_root: int = 0
    minor: bool = False
    interval_steps: int = -2
    voiced_floor_hz: float = 1.0

    # Held-note stabilisation (kills the sustained-note "oo-ee" warble):
    # the snapped input note only changes once the pitch moves past the half-step
    # boundary by this extra margin (in semitones). Larger = stickier held notes.
    # ~0.35 keeps a steady note rock-solid while a real semitone step (>=1.0)
    # still switches promptly, so melodic moves stay responsive (no added lag).
    note_hysteresis: float = 0.35

    # Cross-frame pitch smoothing. The converter feeds compute_shift ONE f0 per hop, so the
    # per-chunk median in representative_f0 can't reject vibrato. A median over this many recent
    # frames (~100 hop/s) holds a sustained note's snapped pitch rock-steady under vibrato → the
    # applied semitone shift stays CONSTANT. (Every shift change glitches the neural pitch-shift;
    # a held note flip-flopping across a scale boundary on vibrato was the squeal on sustained notes.)
    # ~12 ≈ 120ms. Adds a small lag (~half the window) to genuine note changes.
    note_smooth_frames: int = 12

    # MIDI-driven absolute target (None → diatonic interval, the default/unchanged path).
    # When set to a MIDI note number, compute_shift targets that ABSOLUTE pitch instead of
    # a diatonic interval off the sung note: shift = abs_target - held_input_midi. This is
    # how MIDI chord-hold (Mode A) re-points each voice. Set/cleared per block by the engine
    # callback from the `midi` control field; never touched on the diatonic (no-MIDI) path.
    abs_target_midi: int | None = None

    # Auto-chord "free harmony" (off by default → byte-identical). When True, compute_shift infers
    # the current chord LIVE from the sung note (hold-until-doesn't-fit nearest diatonic triad, the
    # arranger's rule) and snaps the diatonic-interval target to the nearest CHORD tone — so the
    # interval bends with the harmony (a 3rd here, a 4th there) instead of a rigid parallel interval.
    auto_chord: bool = False

    # Voice-leading sub-variant of Free/auto_chord (off by default → auto_chord unchanged).
    # When True each voice keeps its own LINE: hold common tones, move to the nearest tone of the
    # new chord on chord changes (minimal leap), with the parallel-interval anchor as register
    # gravity — candidates are clamped to anchor±5 so a line can never drift out of its register
    # (which also keeps the voices from crossing in practice). Only consulted when auto_chord is on.
    voice_lead: bool = False

    # Per-frame hint for the converter (voice_lead only, else always False): True on the frame where
    # the sung note moved but the harmony TARGET pitch did not (common-tone hold) — the shift change
    # is purely compensatory, so the converter should JUMP it (output pitch stays continuous) instead
    # of gliding, which made the held harmony swoop with the singer and slide back (07-19 live).
    snap_shift_hint: bool = False

    # internal state (persists across chunks; not constructor-relevant)
    _held_midi: int | None = None
    _last_shift: int | None = None
    _midi_ring: list | None = None
    _chord_degree: int | None = None
    _last_target: int | None = None
    _last_m: int | None = None

    def reset(self) -> None:
        """Forget the held note (e.g. when re-enabling). Next voiced chunk re-acquires."""
        self._held_midi = None
        self._last_shift = None
        self._midi_ring = None
        self._chord_degree = None
        self._last_target = None
        self._last_m = None
        self.snap_shift_hint = False

    def _triad_pcs(self, degree: int) -> tuple[int, ...]:
        """Pitch classes of the diatonic triad on `degree` (0=I..6=vii) in the current key."""
        scale = scale_pcs_for(self.minor)
        return tuple((self.key_root + scale[(degree + k) % 7]) % 12 for k in (0, 2, 4))

    def _auto_chord_target(self, m: int) -> int:
        """Chord-aware target: infer the chord from sung note `m` (hold current chord until it no
        longer fits, then jump to the nearest triad containing the note), then snap the diatonic
        target to the nearest tone of that chord."""
        if self._chord_degree is None:
            self._chord_degree = 0
        pc = m % 12
        if pc not in self._triad_pcs(self._chord_degree):                 # hold-until-doesn't-fit
            cands = [d for d in range(7) if pc in self._triad_pcs(d)]
            if cands:                                                     # prefer the nearest chord change
                self._chord_degree = min(cands, key=lambda d: min((d - self._chord_degree) % 7,
                                                                   (self._chord_degree - d) % 7))
        tones = self._triad_pcs(self._chord_degree)
        anchor = diatonic_target_midi(m, self.key_root, scale_pcs_for(self.minor), self.interval_steps)
        if self.voice_lead and self._last_target is not None:
            cands = [p for p in range(anchor - 5, anchor + 6) if p % 12 in tones]
            if cands:                                     # minimal leap from OWN previous note,
                target = min(cands,                       # anchor as soft register gravity
                             key=lambda p: (abs(p - self._last_target) + 0.35 * abs(p - anchor), p))
                self._last_target = target
                return target
        target = min((anchor + off for off in (0, -1, 1, -2, 2, -3, 3, -4, 4)),
                     key=lambda p: (p % 12) not in tones)                 # nearest chord tone to the anchor
        self._last_target = target
        return target

    def representative_f0(self, f0_contour: np.ndarray) -> float | None:
        """Median of the voiced frames, or None if the chunk has no pitch.

        The median across voiced frames already rejects per-frame f0 jitter and
        most transient octave errors within a chunk.
        """
        voiced = f0_contour[f0_contour > self.voiced_floor_hz]
        if voiced.size == 0:
            return None
        return float(np.median(voiced))

    def _held_input_midi(self, midi_cont: float) -> int:
        """Apply hysteresis to pick the committed input note from a continuous MIDI.

        Keep the currently-held note unless the pitch has clearly moved past the
        half-step boundary by an extra ``note_hysteresis``. This stops a steady
        note whose f0 wobbles across a scale-tone boundary from flip-flopping
        between two notes, without delaying genuine note changes.
        """
        if self._held_midi is None or abs(midi_cont - self._held_midi) > 0.5 + self.note_hysteresis:
            self._held_midi = int(round(midi_cont))
        return self._held_midi

    def compute_shift(self, f0_contour: np.ndarray) -> int | None:
        """Per-chunk transpose (semitones) for this f0 contour.

        Returns an integer semitone shift = target - input. During brief unvoiced
        gaps it holds the last shift (so voicing flicker mid-note doesn't drop the
        harmony); returns None only when disabled or before any voiced frame.
        Diatonic intervals are whole semitones, so the result is an int.
        """
        if not self.enabled:
            return None
        self.snap_shift_hint = False            # refreshed every frame; True only set below
        f0_rep = self.representative_f0(f0_contour)
        if f0_rep is None:
            return self._last_shift  # hold harmony through short unvoiced gaps
        # cross-frame median smoothing rejects vibrato swing (and stray octave blips) so a sustained
        # note's snapped pitch — and thus the applied shift — stays constant (no glitch-y stepping).
        if self._midi_ring is None:
            self._midi_ring = []
        self._midi_ring.append(f0_to_midi(f0_rep))
        if len(self._midi_ring) > self.note_smooth_frames:
            self._midi_ring.pop(0)
        m = self._held_input_midi(float(np.median(self._midi_ring)))
        if self.abs_target_midi is not None:
            target = int(self.abs_target_midi)          # MIDI Mode A: absolute pitch
        elif self.auto_chord:
            prev_target = self._last_target
            target = self._auto_chord_target(m)         # free harmony: chord-aware (auto-inferred)
            if (self.voice_lead and prev_target is not None and target == prev_target
                    and self._last_m is not None and m != self._last_m):
                self.snap_shift_hint = True             # common-tone hold: compensate instantly
        else:
            scale_pcs = scale_pcs_for(self.minor)       # diatonic path — byte-identical when off
            target = diatonic_target_midi(m, self.key_root, scale_pcs, self.interval_steps)
        self._last_m = m
        shift = int(target - m)
        self._last_shift = shift
        return shift
