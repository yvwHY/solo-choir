"""Silver harmony lines for POP909: melody + chord annotations -> a derived
second voice, giving the pop half of the corpus the (melody, harmony) pair
shape it lacks (BRIDGE is fills-only -- see corpus_probe_pop909.py).

Rules (deliberately simple, chorale-ish voice-leading):
  - candidates = chord tones in a band below the melody (3rd..octave)
  - hold the common tone if it's still a chord tone, else minimal motion
  - never unison/2nd against the melody; rest when the melody rests
"silver" = machine-derived labels: good enough to teach pop idiom +
repetition, not a substitute for real counterpoint (chorales supply that).

Run:  python silver_line.py <POP909 root> [n_songs] [--midi 001,004]
      --midi exports audition files melody+silver to harmony/out/
"""

import sys
from pathlib import Path

import numpy as np
import pretty_midi

REST = -1
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUAL = {  # chord quality -> pitch classes above root (triad core only)
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "7": (0, 4, 7, 10),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9), "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10), "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "maj9": (0, 4, 7, 11), "min9": (0, 3, 7, 10), "9": (0, 4, 7, 10),
}


def chord_pcs(sym):
    """'C#:maj7/5' -> pitch-class set, or None for 'N'."""
    if sym == "N":
        return None
    root_s, _, qual = sym.partition(":")
    qual = qual.split("/")[0]  # slash bass irrelevant for pc set
    root = PC[root_s[0]] + root_s[1:].count("#") - root_s[1:].count("b")
    pcs = QUAL.get(qual) or QUAL.get(qual.rstrip("0123456789()")) or QUAL["maj"]
    return frozenset((root + i) % 12 for i in pcs)


def melody_grid(notes, grid):
    toks = np.full(len(grid), REST, dtype=int)
    for n in notes:
        i0, i1 = np.searchsorted(grid, [n.start, n.end])
        toks[i0:i1] = n.pitch
    return toks


def chord_grid(chord_file, grid):
    spans = []
    for line in Path(chord_file).read_text().splitlines():
        if line.strip():
            a, b, sym = line.split("\t")
            spans.append((float(a), float(b), chord_pcs(sym)))
    out = [None] * len(grid)
    for a, b, pcs in spans:
        i0, i1 = np.searchsorted(grid, [a, b])
        for i in range(i0, min(i1, len(grid))):
            out[i] = pcs
    return out


GAP_MAX = 4  # 16th ticks (= 1 beat) a silver tone may ring through a melody rest


def derive(mel, chords):
    """Melody + chord tokens -> silver harmony line (same grid).

    Two rules beyond the candidate/hold basics, both from the 001-audition
    post-mortem (choppy melodies produced a parallel shadow):
      - breath-bridging: short melody rests (<= GAP_MAX) don't silence the
        silver tone while it stays a chord tone -- a partner sings through
        your commas; long rests (phrase ends) still go silent
      - line memory: `prev` survives rests as the voice-leading anchor, so
        the next phrase connects by minimal motion instead of re-anchoring
        a parallel 3rd under the melody
    """
    silver = np.full(len(mel), REST, dtype=int)
    prev = None  # last sounded pitch: anchor persists across rests
    gap = 0
    for i, (m, pcs) in enumerate(zip(mel, chords)):
        if m == REST or pcs is None:
            gap += 1
            if (prev is not None and gap <= GAP_MAX
                    and pcs is not None and prev % 12 in pcs):
                silver[i] = prev  # breath-bridging
            continue
        gap = 0
        cands = [p for p in range(m - 12, m - 2) if p % 12 in pcs]
        cands = [p for p in cands if (m - p) % 12 not in (0, 1, 2, 11)]
        if not cands:
            continue
        if prev in cands:
            choice = prev  # common-tone hold
        else:
            # minimal motion from the anchor, mild 3rds/6ths preference
            def cost(p):
                c = abs(p - prev) if prev is not None else abs((m - p) - 4)
                return c + (0 if (m - p) % 12 in (3, 4, 8, 9) else 1.5)
            choice = min(cands, key=cost)
        silver[i] = choice
        prev = choice
    return silver


def song_paths(root, name):
    d = Path(root) / name
    return d / f"{name}.mid", d / "beat_midi.txt", d / "chord_midi.txt"


def load_song(root, name):
    mid, beats_f, chords_f = song_paths(root, name)
    pm = pretty_midi.PrettyMIDI(str(mid))
    mel_notes = {t.name.upper(): t for t in pm.instruments}["MELODY"].notes
    beats = np.array([float(l.split()[0]) for l in beats_f.read_text().splitlines() if l.strip()])
    grid = np.concatenate([np.linspace(beats[i], beats[i + 1], 4, endpoint=False)
                           for i in range(len(beats) - 1)])
    mel = melody_grid(mel_notes, grid)
    silver = derive(mel, chord_grid(chords_f, grid))
    return grid, mel, silver


def export_midi(grid, mel, silver, out_path):
    pm = pretty_midi.PrettyMIDI()
    for name, toks, prog in (("MELODY", mel, 52), ("SILVER", silver, 53)):
        inst = pretty_midi.Instrument(program=prog, name=name)
        i = 0
        while i < len(toks):
            if toks[i] == REST:
                i += 1
                continue
            j = i
            while j < len(toks) and toks[j] == toks[i]:
                j += 1
            end = grid[j] if j < len(grid) else grid[-1] + 0.2
            inst.notes.append(pretty_midi.Note(90, int(toks[i]), grid[i], end))
            i = j
        pm.instruments.append(inst)
    pm.write(str(out_path))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
    audition = []
    if "--midi" in sys.argv:
        audition = sys.argv[sys.argv.index("--midi") + 1].split(",")

    cov, ivals, steps, holds = [], [], [], []
    songs = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())[:n]
    for name in songs:
        try:
            grid, mel, silver = load_song(root, name)
        except Exception:
            continue
        voiced = mel != REST
        both = voiced & (silver != REST)
        cov.append(both.sum() / max(1, voiced.sum()))
        iv = (mel[both] - silver[both]) % 12
        ivals.extend(iv.tolist())
        d = np.diff(silver[silver != REST])
        steps.extend(np.abs(d[d != 0]).tolist())
        holds.append(np.mean(d == 0) if len(d) else 0)

    cov = np.array(cov)
    ivals = np.array(ivals)
    print(f"{len(cov)} songs | silver coverage of melody time: median {np.median(cov):.0%} "
          f"(>=80% in {np.mean(cov>=0.8):.0%} of songs)")
    frac = lambda s: np.isin(ivals, list(s)).mean()
    print(f"interval vs melody: 3rds {frac({3,4}):.0%}  6ths {frac({8,9}):.0%}  "
          f"4th/5th {frac({5,7}):.0%}  other {1-frac({3,4,8,9,5,7}):.0%}")
    print(f"silver motion: median step {np.median(steps):.0f} st, hold ratio median {np.median(holds):.0%}")

    if audition:
        out = Path(__file__).parent / "out"
        out.mkdir(exist_ok=True)
        for name in audition:
            grid, mel, silver = load_song(root, name)
            export_midi(grid, mel, silver, out / f"silver_{name}.mid")
            print(f"audition -> {out}/silver_{name}.mid")
