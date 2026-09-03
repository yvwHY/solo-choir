"""Tokenizer v2: POP909 (melody, silver) -> chromatic C-normalized token pairs
with a per-tick beat-phase channel. Proposal Task 2 (second-chorus-partner).

Changes vs the chorale pipeline (data_pipeline.py):
  - per-song transposition to C major / A minor from the key annotation
    (dominant key when the song modulates), NO chromatic snapping after --
    borrowed-chord and blue notes stay real tokens (vocab 36..84 already
    covers all semitones)
  - beat phase 0..15 per 16th tick (from the annotated beats + downbeats),
    written alongside the tokens for the model's phase embedding (Task 4)
  - out-of-range pitches fold back by octaves (same policy as the live ear)

Output: <out>/pop909_v2.npz  {tok_<song>: (T,2) int64, phase_<song>: (T,)}
        + meta.json (per-song key/shift/stats). Round-trip proof: --verify
        re-decodes one song and checks grid equality + writes audition wav.
Run:  python tokenize_v2.py <POP909 root> <out_dir> [--verify 004]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from silver_line import REST as GREST, derive, chord_grid, melody_grid, song_paths  # noqa: E402
import pretty_midi  # noqa: E402

REST, HOLD, PITCH_OFFSET, PITCH_LO, PITCH_HI = 0, 1, 2, 36, 84
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
C_MAJ = {0, 2, 4, 5, 7, 9, 11}


def dominant_key(key_file):
    """Longest-duration annotated key -> (tonic pc, mode)."""
    best, dur = None, -1
    for line in Path(key_file).read_text().splitlines():
        if not line.strip():
            continue
        a, b, sym = line.split()
        if sym == "N":
            continue
        if float(b) - float(a) > dur:
            dur, best = float(b) - float(a), sym
    if best is None:
        return None
    root_s, _, mode = best.partition(":")
    pc = (PC[root_s[0]] + root_s[1:].count("#") - root_s[1:].count("b")) % 12
    return pc, mode


def shift_to_c(tonic_pc, mode):
    """Semitone shift moving the tonic to C (major) / A (minor), minimal move."""
    target = 0 if mode == "maj" else 9
    s = (target - tonic_pc) % 12
    return s - 12 if s > 6 else s


def grid_to_tokens(grid_pitches, shift):
    """Absolute-pitch grid -> REST/HOLD/onset tokens, transposed + folded."""
    toks = np.full(len(grid_pitches), REST, dtype=np.int64)
    prev = None
    for i, p in enumerate(grid_pitches):
        if p == GREST:
            prev = None
            continue
        m = int(p) + shift
        while m > PITCH_HI:
            m -= 12
        while m < PITCH_LO:
            m += 12
        toks[i] = HOLD if m == prev else m - PITCH_LO + PITCH_OFFSET
        prev = m
    return toks


def tokens_to_grid(toks):
    out = np.full(len(toks), GREST, dtype=int)
    prev = None
    for i, t in enumerate(toks):
        if t == REST:
            prev = None
        elif t == HOLD:
            out[i] = prev if prev is not None else GREST
        else:
            prev = t - PITCH_OFFSET + PITCH_LO
            out[i] = prev
    return out


def beat_phase(beat_file, n_ticks):
    """Per-16th-tick position in bar (0..15) from beat + downbeat annotations."""
    rows = [l.split() for l in Path(beat_file).read_text().splitlines() if l.strip()]
    down = np.array([float(r[2]) for r in rows]) == 1.0
    beat_in_bar = np.zeros(len(rows), dtype=int)
    b = 0
    for i in range(len(rows)):
        b = 0 if down[i] else b + 1
        beat_in_bar[i] = b % 4
    phase = np.repeat(beat_in_bar * 4, 4) + np.tile(np.arange(4), len(rows))
    return phase[:n_ticks] % 16


def tokenize_song(root, name):
    mid, beats_f, chords_f = song_paths(root, name)
    pm = pretty_midi.PrettyMIDI(str(mid))
    mel_notes = {t.name.upper(): t for t in pm.instruments}["MELODY"].notes
    beats = np.array([float(l.split()[0]) for l in beats_f.read_text().splitlines() if l.strip()])
    grid = np.concatenate([np.linspace(beats[i], beats[i + 1], 4, endpoint=False)
                           for i in range(len(beats) - 1)])
    mel = melody_grid(mel_notes, grid)
    sil = derive(mel, chord_grid(chords_f, grid))
    key = dominant_key(Path(root) / name / "key_audio.txt")
    if key is None:
        return None
    shift = shift_to_c(*key)
    tok = np.stack([grid_to_tokens(mel, shift), grid_to_tokens(sil, shift)], axis=1)
    phase = beat_phase(beats_f, len(grid))
    return tok, phase, {"key": f"{key[0]}:{key[1]}", "shift": shift}


def main():
    root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    verify = sys.argv[sys.argv.index("--verify") + 1] if "--verify" in sys.argv else None

    arrays, meta = {}, {}
    chroma_counts = total_counts = 0
    songs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    for name in songs:
        try:
            r = tokenize_song(root, name)
        except Exception as e:
            print(f"  {name} FAIL: {e}", flush=True)
            continue
        if r is None:
            continue
        tok, phase, m = r
        arrays[f"tok_{name}"], arrays[f"phase_{name}"] = tok, phase
        meta[name] = m
        pitched = tok[tok >= PITCH_OFFSET] - PITCH_OFFSET + PITCH_LO
        chroma_counts += int(np.sum(~np.isin(pitched % 12, list(C_MAJ))))
        total_counts += len(pitched)

    np.savez_compressed(out_dir / "pop909_v2.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps(
        {"vocab": {"REST": REST, "HOLD": HOLD, "PITCH_OFFSET": PITCH_OFFSET,
                   "PITCH_LO": PITCH_LO, "PITCH_HI": PITCH_HI, "vocab_size": 51,
                   "phase_size": 16},
         "songs": meta}, indent=1))
    print(f"{len(meta)}/{len(songs)} songs -> {out_dir}/pop909_v2.npz")
    print(f"chromatic (non-C-scale) onset tokens after transposition: "
          f"{chroma_counts/max(1,total_counts):.1%}  (these are the real "
          f"borrowed/blue notes the old snap would have destroyed)")

    if verify and f"tok_{verify}" in arrays:
        # round-trip: decode -> compare against the transposed source grid
        mid, beats_f, chords_f = song_paths(root, verify)
        pm = pretty_midi.PrettyMIDI(str(mid))
        mel_notes = {t.name.upper(): t for t in pm.instruments}["MELODY"].notes
        beats = np.array([float(l.split()[0]) for l in beats_f.read_text().splitlines() if l.strip()])
        grid = np.concatenate([np.linspace(beats[i], beats[i + 1], 4, endpoint=False)
                               for i in range(len(beats) - 1)])
        mel = melody_grid(mel_notes, grid)
        shift = meta[verify]["shift"]
        ref = mel.copy()
        for i, p in enumerate(ref):
            if p != GREST:
                m = int(p) + shift
                while m > PITCH_HI:
                    m -= 12
                while m < PITCH_LO:
                    m += 12
                ref[i] = m
        dec = tokens_to_grid(arrays[f"tok_{verify}"][:, 0])
        print(f"round-trip {verify}: decode==source-grid -> {bool(np.array_equal(dec, ref))}")


if __name__ == "__main__":
    main()
