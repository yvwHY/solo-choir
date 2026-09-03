"""Spec-conformance tests for the Solo Choir diatonic harmony logic.

Run: python -m voice_changer.test_solo_choir   (from the server/ dir)
No test framework required; plain asserts so it runs anywhere.
"""
import numpy as np

from voice_changer.SoloChoir import (
    MAJOR_SCALE_PCS,
    diatonic_target_midi,
    f0_to_midi,
    SoloChoirHarmonizer,
)

A4_HZ = 440.0
A4_MIDI = 69


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_f0_to_midi():
    assert approx(f0_to_midi(440.0), 69.0)
    assert approx(f0_to_midi(880.0), 81.0)   # octave up
    assert approx(f0_to_midi(220.0), 57.0)   # octave down
    assert approx(round(f0_to_midi(261.63)), 60)  # ~middle C


def test_spec_presets_in_C_major():
    # Sung note A4 (MIDI 69) in C major. Verify the four common presets from
    # the spec produce the diatonically-correct target and transpose.
    root, scale = 0, MAJOR_SCALE_PCS
    cases = {
        -2: (65, -4),   # third below  -> F4
        +2: (72, +3),   # third above  -> C5
        -5: (60, -9),   # sixth below  -> C4
        +5: (77, +8),   # sixth above  -> F5
    }
    for steps, (exp_target, exp_transpose) in cases.items():
        target = diatonic_target_midi(A4_MIDI, root, scale, steps)
        assert target == exp_target, (steps, target, exp_target)
        assert target - A4_MIDI == exp_transpose, (steps, target - A4_MIDI, exp_transpose)


def test_octave_wrap_is_12_semitones():
    # +7 scale steps = exactly one octave up regardless of starting degree.
    for m in (60, 62, 64, 65, 67, 69, 71):
        assert diatonic_target_midi(m, 0, MAJOR_SCALE_PCS, 7) == m + 12
        assert diatonic_target_midi(m, 0, MAJOR_SCALE_PCS, -7) == m - 12


def test_snap_off_scale_input():
    # C#5 (MIDI 61) is not in C major; a third below (-2) should snap then shift.
    # Snap 61 -> 60 (C, lower on tie vs D), degree 0; -2 steps -> A3 (57).
    assert diatonic_target_midi(61, 0, MAJOR_SCALE_PCS, -2) == 57


def test_harmonizer_compute_shift_third_below():
    h = SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=-2)
    # voiced contour around A4 with a couple of unvoiced (0) frames
    contour = np.array([0.0, A4_HZ, A4_HZ, A4_HZ * 1.001, 0.0])
    assert h.compute_shift(contour) == -4  # A4 -> F4


def test_harmonizer_disabled_and_unvoiced_return_none():
    off = SoloChoirHarmonizer(enabled=False, interval_steps=-2)
    assert off.compute_shift(np.array([A4_HZ, A4_HZ])) is None
    on = SoloChoirHarmonizer(enabled=True, interval_steps=-2)
    assert on.compute_shift(np.zeros(5)) is None  # fully unvoiced


def _midi_to_contour(midi, n=8):
    """A voiced f0 contour (Hz) at a given (possibly fractional) MIDI note."""
    hz = 440.0 * 2 ** ((midi - 69) / 12.0)
    return np.full(n, hz, dtype=np.float32)


def test_held_note_jitter_gives_constant_shift():
    # Sustained note whose f0 wobbles +/-0.4 semitone across a scale-tone boundary.
    # With hysteresis the committed note must not flip, so the shift stays constant
    # (this is the fix for the sustained-note "oo-ee" warble).
    h = SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=-2)
    jitter = [60.0, 59.6, 60.4, 59.7, 60.3, 60.0, 59.8]
    shifts = [h.compute_shift(_midi_to_contour(m)) for m in jitter]
    assert len(set(shifts)) == 1, f"held note flip-flopped: {shifts}"


def test_real_note_change_switches_promptly():
    # A genuine step of >= 1 semitone must switch on the very next chunk (no lag).
    h = SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=-2)
    h.compute_shift(_midi_to_contour(60.0))            # hold C4
    s_after_step = h.compute_shift(_midi_to_contour(62.0))  # step to D4
    assert h._held_midi == 62
    # D4 third-below in C major -> B3: shift -3
    assert s_after_step == -3


def test_unvoiced_holds_last_shift():
    h = SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=-2)
    s = h.compute_shift(_midi_to_contour(69.0))        # A4 -> -4
    assert s == -4
    assert h.compute_shift(np.zeros(5)) == -4          # unvoiced gap holds harmony


def test_minor_key_third_below():
    # A minor (root=9). Sing E5 (MIDI 76). A natural-minor third below E is C.
    # E(76) in A minor: degrees A,B,C,D,E,F,G -> E is degree 4. -2 -> degree 2 = C5 (72).
    from voice_changer.SoloChoir import MINOR_SCALE_PCS
    assert diatonic_target_midi(76, 9, MINOR_SCALE_PCS, -2) == 72


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} Solo Choir spec tests passed.")


if __name__ == "__main__":
    main()
