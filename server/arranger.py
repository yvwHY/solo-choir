"""v1 auto-arranger (offline, rule-based): one melody -> 4 voice-led SATB target lines.
Pure logic, no audio. Design: specs/2026-06-25-auto-arranger-v1-design.md.

Chord inference = hold-until-doesn't-fit (chords sustain; change only when the melody leaves the
chord, to the nearest diatonic triad containing the new note). Voice-leading = each voice takes the
chord tone nearest its previous note (bass = root; soprano tracks the melody). Bloom = a note held
>= SUSTAIN_S enriches the triad with its diatonic 7th and lifts the soprano onto it. Major key only.
"""
from voice_changer.SoloChoir import MAJOR_SCALE_PCS

VOICES = ["Bass", "Tenor", "Alto", "Sop"]
# octave-placement ranges per voice (MIDI), low->high
RANGES = {"Bass": (40, 60), "Tenor": (48, 67), "Alto": (55, 74), "Sop": (60, 81)}
SUSTAIN_S = 0.8   # bloom threshold (seconds) — by-ear tunable


def _triad_pcs(key_root, degree):
    """Pitch classes of the diatonic triad on `degree` (0=I..6=vii) in the major key."""
    return [(key_root + MAJOR_SCALE_PCS[(degree + k) % 7]) % 12 for k in (0, 2, 4)]


def _seventh_pc(key_root, degree):
    """Pitch class of the diatonic 7th above the root of `degree`'s chord."""
    return (key_root + MAJOR_SCALE_PCS[(degree + 6) % 7]) % 12


def _pick_chord(pc, key_root, prev_degree):
    """Nearest diatonic triad containing pitch-class `pc`: most shared tones with the previous
    chord, tie-break toward tonic/dominant. Falls back to prev if pc is non-diatonic."""
    cands = [d for d in range(7) if pc in _triad_pcs(key_root, d)]
    if not cands:
        return prev_degree
    prev = set(_triad_pcs(key_root, prev_degree))
    return max(cands, key=lambda d: (len(prev & set(_triad_pcs(key_root, d))), d in (0, 4)))


def _nearest_midi(pc, anchor, lo, hi):
    """MIDI note of pitch-class `pc` within [lo,hi] nearest to `anchor`."""
    cands = [n for n in range(lo, hi + 1) if n % 12 == pc % 12]
    if not cands:                                  # range < 1 octave: widen
        cands = [n for n in range(lo - 12, hi + 13) if n % 12 == pc % 12]
    return min(cands, key=lambda n: abs(n - anchor))


def _arrange_core(notes, key_root, max_shift, style):
    """Place 4 voices per melody note + record the chord. style='choral' keeps the ORIGINAL placement
    (byte-identical notes); 'pad' always enriches with the 7th and lifts the soprano onto a colour
    tone (the harmony then sustains via _hold in arrange_with_chords)."""
    out = {v: [] for v in VOICES}
    chords = []
    prev = {v: (RANGES[v][0] + RANGES[v][1]) // 2 for v in VOICES}   # seed mid-range
    degree = 0                                     # start on I
    for (s, e, m) in notes:
        def rng(v):                                # this voice's placement window for this note
            lo, hi = RANGES[v]
            if max_shift is not None:
                lo, hi = max(lo, m - max_shift), min(hi, m + max_shift)
                if lo > hi:                        # window misses the voice range → clamp toward melody
                    lo = hi = min(RANGES[v][1], max(RANGES[v][0], m))
            return lo, hi
        pc = m % 12
        if pc not in _triad_pcs(key_root, degree):
            degree = _pick_chord(pc, key_root, degree)
        chords.append((s, e, degree))
        tones = _triad_pcs(key_root, degree)
        bloom = (e - s) >= SUSTAIN_S
        if style == "pad":                         # Pad: always enrich (7th); harmony will sustain
            tones = tones + [_seventh_pc(key_root, degree)]
        elif bloom:
            tones = tones + [_seventh_pc(key_root, degree)]
        targets = {}
        targets["Bass"] = _nearest_midi(tones[0], prev["Bass"], *rng("Bass"))   # root
        for v in ("Tenor", "Alto"):                # inner voices: nearest chord tone to themselves
            pc_v = min(tones, key=lambda p: abs(_nearest_midi(p, prev[v], *rng(v)) - prev[v]))
            targets[v] = _nearest_midi(pc_v, prev[v], *rng(v))
        sop_pc = tones[-1] if (bloom or style == "pad") else min(   # soprano tracks the melody (or colour tone)
            tones, key=lambda p: abs(_nearest_midi(p, m, *rng("Sop")) - m))
        targets["Sop"] = _nearest_midi(sop_pc, m, *rng("Sop"))
        ordered = sorted(targets[v] for v in VOICES)   # enforce SATB order (no voice crossing): lowest=Bass … highest=Sop.
        for v, p in zip(VOICES, ordered):              # same pitch SET (chord/audio unchanged) — just keeps the lines from crossing.
            targets[v] = p
        for v in VOICES:
            out[v].append((s, e, targets[v]))
            prev[v] = targets[v]
    return out, chords


def _scale_index(midi, key_root):
    """Absolute diatonic index of the scale tone nearest `midi` (7 per octave)."""
    octv, pc = divmod(midi - key_root, 12)
    di = min(range(7), key=lambda i: abs(MAJOR_SCALE_PCS[i] - pc))
    return octv * 7 + di


def _diatonic_shift(midi, key_root, steps):
    """Shift `midi` by `steps` diatonic scale degrees in the major key (negative = down)."""
    octv, di = divmod(_scale_index(midi, key_root) + steps, 7)
    return key_root + octv * 12 + MAJOR_SCALE_PCS[di]


def _nearest_chord_tone(midi, tones):
    """Nudge `midi` to the nearest pitch whose pitch-class is a chord tone (≤ a few semitones)."""
    return min((midi + d for d in (0, -1, 1, -2, 2, -3, 3)), key=lambda p: (p % 12) not in tones)


# parallel-thirds voicing: Sop=melody, Alto a diatonic 3rd below, Tenor a 5th below, Bass = chord root.
_THIRDS_SHIFT = {"Sop": 0, "Alto": -2, "Tenor": -4}


def _arrange_thirds(notes, key_root, max_shift):
    """Parallel-thirds style (harvestmusician method 2): the harmony tracks the melody contour at fixed
    diatonic intervals (parallel motion), with a 4th substituted for a 3rd when it clashes with the
    chord. Bass is the chord root for grounding. Simpler/brighter than the voice-led 'choral' style."""
    out = {v: [] for v in VOICES}
    chords = []
    prevbass = (RANGES["Bass"][0] + RANGES["Bass"][1]) // 2
    degree = 0
    for (s, e, m) in notes:
        pc = m % 12
        if pc not in _triad_pcs(key_root, degree):
            degree = _pick_chord(pc, key_root, degree)
        chords.append((s, e, degree))
        tones = _triad_pcs(key_root, degree)
        targets = {}
        for v in ("Sop", "Alto", "Tenor"):
            p = _diatonic_shift(m, key_root, _THIRDS_SHIFT[v])
            if v != "Sop" and (p % 12) not in tones:           # dissonant 3rd → 4th (nearest chord tone)
                p = _nearest_chord_tone(p, tones)
            lo, hi = RANGES[v]
            while p < lo: p += 12
            while p > hi: p -= 12
            targets[v] = p
        targets["Bass"] = _nearest_midi(tones[0], prevbass, *RANGES["Bass"]); prevbass = targets["Bass"]
        ordered = sorted(targets[v] for v in VOICES)           # enforce SATB order (no crossing)
        for v, p in zip(VOICES, ordered):
            out[v].append((s, e, p))
    return out, chords


def _hold(score):
    """Merge consecutive same-pitch notes into one sustained note (Pad: harmony stops moving on every
    melody note → a held-chord texture instead of homorhythmic block chords)."""
    merged = []
    for (s, e, mid) in score:
        if merged and merged[-1][2] == mid and abs(merged[-1][1] - s) < 1e-3:
            merged[-1] = (merged[-1][0], e, mid)
        else:
            merged.append((s, e, mid))
    return merged


def arrange_with_chords(notes, key_root, max_shift=None, style="choral"):
    """SATB + the inferred chord track. style: 'choral' (voice-led block, original) | 'pad' (sustained,
    7th-enriched) | 'thirds' (parallel-thirds, brighter). Returns (voices, chords)."""
    if style == "thirds":
        return _arrange_thirds(notes, key_root, max_shift)
    voices, chords = _arrange_core(notes, key_root, max_shift, style)
    if style == "pad":
        voices = {v: _hold(voices[v]) for v in VOICES}   # harmony sustains; the melody still moves
    return voices, chords


def arrange(notes, key_root, max_shift=None, style="choral"):
    """Back-compatible: voices only. style='choral' returns the SAME notes as before."""
    return arrange_with_chords(notes, key_root, max_shift, style)[0]
