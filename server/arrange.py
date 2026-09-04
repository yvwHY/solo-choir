"""Offline auto-arranger render: melody wav -> voice-led SATB arrangement wav.
Usage: python arrange.py <melody.wav> [--key C] [--model <dir>]
Reuses BeatriceSoloChoir + harmonizer.abs_target_midi (the validated MIDI Mode A path); no DSP/engine
change. Design: specs/2026-06-25-auto-arranger-v1-design.md."""
import argparse
import os
import sys

import numpy as np
import soundfile as sf
import soxr

import arranger
import score_export
from voice_changer.SoloChoir import SoloChoirHarmonizer
from beatrice_converter import BeatriceSoloChoir
from beatrice_solo_choir_live import _SATB2_MODEL          # SATB male/female single model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from bridge import DEFAULT_MODEL as USER_MODEL             # the user's own trained voice ("choir of one")

DEV_SR = 48000
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# satb2 speaker by PLACED pitch, not by part: sp1 (male) is natural LOW, sp0 (female) natural HIGH.
# Picking by where a voice actually sits avoids "female model forced low = bad timbre" (the tight-
# voicing problem) — a low voice gets the male timbre, a high voice the female timbre.
SPEAKER_SPLIT_MIDI = 60   # >= this -> female (sp0); below -> male (sp1)


def _speaker_for(median_midi):
    return 0 if median_midi >= SPEAKER_SPLIT_MIDI else 1


def _detect_key(x16k):
    """Best-fit MAJOR key root (0-11): histogram sung pitch-classes, pick the scale covering the most."""
    from beatrice_converter import estimate_f0_autocorr
    from voice_changer.SoloChoir import MAJOR_SCALE_PCS
    W, H = 2048, 512
    hist = np.zeros(12)
    for i in range(0, max(0, len(x16k) - W), H):
        f = estimate_f0_autocorr(np.ascontiguousarray(x16k[i:i + W]), 16000)
        if f > 0:
            hist[int(round(69 + 12 * np.log2(f / 440.0))) % 12] += 1
    if hist.sum() == 0:
        return 0
    return max(range(12), key=lambda r: sum(hist[(r + s) % 12] for s in MAJOR_SCALE_PCS))


def _tuning_offset(midi_frames):
    """Take-wide tuning offset θ ∈ [-0.5, 0.5) st: circular mean of the voiced frames' distance to the
    integer grid. A singer centred between two semitones (e.g. 52.5) makes independent per-note
    rounding FLIP between neighbours (the 2026-07-06 E3 take: 8 equal syllables came out 52/53/52…);
    quantising against (grid + θ) keeps the whole take consistent and preserves intervals."""
    fr = np.asarray(midi_frames, dtype=np.float64)
    fr = fr[~np.isnan(fr)]
    if fr.size == 0:
        return 0.0
    ang = 2.0 * np.pi * (fr - np.floor(fr))
    return float(np.angle(np.mean(np.exp(1j * ang))) / (2.0 * np.pi))


def _melody_tracks(mel16k, mel_score, smooth_frames=25, edge_s=0.08, max_rate=0.02):
    """Offline SwiftF0 lookahead over the SUNG melody. Returns (dev, voiced) per 10ms hop.
    SwiftF0 (not pyin): it is the SAME tracker transcription trusts — two different trackers made the
    correction absorb their disagreement noise — and it is ~10x faster (pyin cost ~0.3x audio length).

    voiced[i]: is the singer voicing at hop i — unvoiced gaps (rests, consonants) are where a shift
    change can JUMP inaudibly instead of gliding through sounding audio.

    dev[i] (semitones): smoothed, RATE-LIMITED deviation of the sung pitch from its transcribed note.
    Subtracting it from the score-to-score shift pins the harmony to the grid (kills the slow wander)
    WITHOUT chasing fast movements: frames within edge_s of a note boundary are excluded (entry
    scoops/falls), and the correction slews at most max_rate st/hop (2 st/s) — a lagging correction
    that chases a fast slide produces an audible COUNTER-slide in the harmony (found 2026-07-06)."""
    import swift_f0
    from scipy.signal import medfilt
    r = swift_f0.SwiftF0().detect_from_array(np.asarray(mel16k, dtype=np.float32), 16000)
    st = np.asarray(r.timestamps)
    hz = np.asarray(r.pitch_hz)
    ok = (np.asarray(r.confidence) > 0.5) & (hz > 0)
    n = len(mel16k) // 160
    all_midi = np.where(ok, 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-6) / 440.0), np.nan)
    theta = _tuning_offset(all_midi)           # pin to the SINGER'S tuning centre, not A440: correcting
    dev = np.zeros(n, dtype=np.float32)        # toward the raw grid would hold the harmony up to half a
    voiced = np.zeros(n, dtype=bool)           # semitone off the dry lead
    j = 0
    for i in range(n):
        t = i * 0.01
        while j + 1 < len(st) and abs(st[j + 1] - t) <= abs(st[j] - t):   # nearest SwiftF0 frame
            j += 1
        if len(st) == 0 or abs(st[j] - t) > 0.02 or not ok[j]:
            continue
        voiced[i] = True
        note = next((mid for (s, e, mid) in mel_score if s + edge_s <= t < e - edge_s), None)
        if note is None:                       # rest or note edge: no correction reference
            continue
        d = float(all_midi[j]) - (note + theta)
        if abs(d) <= 2.0:                      # beyond ±2 st = octave/track error, not intonation
            dev[i] = d
    dev = medfilt(dev, smooth_frames | 1).astype(np.float32)
    for i in range(1, len(dev)):               # slew limit: correct drift, never chase slides
        step = float(np.clip(dev[i] - dev[i - 1], -max_rate, max_rate))
        dev[i] = dev[i - 1] + step
    return dev, voiced


def _render_voice(mel16k, score, model, speaker, glide=0.4, hold=1, mel_score=None, dev=None,
                  voiced=None):
    """Render one voice: feed the melody @16k frame-by-frame, drive the pitch shift from `score`
    (list of (start_s,end_s,target_midi)) by frame time. Returns 48k float32.

    mel_score (melody notes, same tuple format): when given, the shift is computed SCORE-TO-SCORE —
    target note − melody note, constant within a note — instead of target − per-frame detected pitch.
    Offline we know both sides, so the singer's jitter/vibrato transfers in PARALLEL into the harmony
    (intervals against the dry lead stay exact) rather than modulating the shift (the wandering
    instability). Also skips f0 autocorr entirely. None = old frame-level tracking path."""
    score_shift = mel_score is not None
    h = SoloChoirHarmonizer(enabled=not score_shift)   # score_shift path needs no pitch tracking
    conv = BeatriceSoloChoir(model, harmonizer=h, target_speaker=int(speaker))
    if glide > 0.0:
        # Pitch glide (same validated values as solo_min live): without it every note boundary
        # jumps the engine TUNE -> per-note transition scrape (the render "instability").
        # glide=0 -> old jump behaviour, for A/B.  See memory pitch-transition-scrape-diagnosis.
        conv.pitch_glide_step = float(glide)
        conv.pitch_hold_frames = int(hold)
    rs = soxr.ResampleStream(conv.out_sr, DEV_SR, 1, dtype="float32", quality="HQ")  # 24k->48k
    HOP = conv.hop                                  # 160 @16k = 10ms
    out24 = []
    last = None
    nf = len(mel16k) // HOP
    if score_shift:
        # Precompute the per-frame shift, then BACKFILL rests with the NEXT note's shift: the engine
        # glide then ramps during the silence and the shift is already in place when the note starts
        # (gliding on the note onset = an audible scoop). Offline knows the future — use it.
        offs = np.full(nf, np.nan, dtype=np.float32)
        for i in range(nf):
            t = i * HOP / conv.in_sr
            tgt = next((mid for (s, e, mid) in score if s <= t < e), None)
            mel = next((mid for (s, e, mid) in mel_score if s <= t < e), None)
            if tgt is not None and mel is not None:
                offs[i] = float(tgt - mel) - (float(dev[i]) if dev is not None and i < len(dev) else 0.0)
        nxt = np.nan
        for i in range(nf - 1, -1, -1):              # rests take the upcoming note's shift
            if np.isnan(offs[i]):
                offs[i] = nxt
            else:
                nxt = offs[i]
        prev = 0.0
        for i in range(nf):                          # trailing rest (no upcoming note) holds the last
            if np.isnan(offs[i]):
                offs[i] = prev
            else:
                prev = offs[i]
    for i in range(nf):
        t = i * HOP / conv.in_sr                     # frame start time (s)
        if score_shift:
            if voiced is not None and glide > 0.0:
                # unvoiced gap (rest/consonant): no pitch to hear -> JUMP the shift instantly there,
                # so legato note boundaries land ready instead of gliding through sounding audio
                conv.pitch_glide_step = 0.0 if (i >= len(voiced) or not voiced[i]) else float(glide)
            conv.pitch_offset = float(offs[i])
        else:
            tgt = next((mid for (s, e, mid) in score if s <= t < e), last)
            last = tgt
            h.abs_target_midi = tgt                  # None outside any note (input is silent there)
        out24.append(np.asarray(conv.process_frame(mel16k[i*HOP:(i+1)*HOP]), dtype=np.float32))
    y24 = np.concatenate(out24) if out24 else np.zeros(0, dtype=np.float32)
    return np.asarray(rs.resample_chunk(y24), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("melody")
    ap.add_argument("--key", default="auto", help="major key root, or 'auto' to detect from the melody")
    # default = the user's own voice ("choir of one"): tight voicing in one consistent timbre sounded
    # natural & un-electronic by ear (2026-06-25). Pass --model <satb2 dir> for male/female SATB.
    ap.add_argument("--model", default=USER_MODEL)
    ap.add_argument("--max-shift", type=int, default=8,
                    help="clamp every voice within ±N semitones of the melody (tight voicing → small "
                         "shifts → less 'electronic'). Default 8 (validated by ear). Big value = wide SATB.")
    ap.add_argument("--glide", type=float, default=0.4,
                    help="pitch glide semitones/hop at note transitions (0.4 = fast ~by-ear winner "
                         "2026-07-06; slower swoops audibly, 0 = instant jump)")
    args = ap.parse_args()

    # melody audio -> 48k mono -> 16k for the converter input (loaded first; key auto-detect needs it)
    mel, sr = sf.read(args.melody, dtype="float32", always_2d=True)
    mel48 = mel.mean(axis=1)
    if sr != DEV_SR:
        mel48 = np.asarray(soxr.resample(mel48, sr, DEV_SR, quality="HQ"), dtype=np.float32)
    mel16 = np.asarray(soxr.resample(mel48, DEV_SR, 16000, quality="HQ"), dtype=np.float32)

    key_root = _detect_key(mel16) if args.key == "auto" else (KEYS.index(args.key) if args.key in KEYS else 0)
    print(f"[arrange] key = {KEYS[key_root]} major", file=sys.stderr)

    notes = score_export.wav_to_notes(args.melody, key_root, False)   # major key
    if not notes:
        print("no notes found in melody", file=sys.stderr); sys.exit(1)
    scores = arranger.arrange(notes, key_root, max_shift=args.max_shift)

    base = os.path.splitext(args.melody)[0]
    dev, voiced = _melody_tracks(mel16, notes)      # grid-pinning correction + jump-in-gaps info
    voices48 = {}
    for v in arranger.VOICES:
        med = sorted(t for _, _, t in scores[v])[len(scores[v]) // 2]   # this voice's median pitch
        # the male/female split only applies to the 2-speaker satb2 model; a single-speaker model
        # (e.g. the user's own tenor voice → "a choir of one") always uses speaker 0.
        spk = _speaker_for(med) if args.model == _SATB2_MODEL else 0
        y = _render_voice(mel16, scores[v], args.model, spk, glide=args.glide, mel_score=notes,
                          dev=dev, voiced=voiced)
        voices48[v] = y
        sf.write(f"{base}_arr_{v.lower()}.wav", y, DEV_SR)         # per-voice stem
        tone = (("female" if spk == 0 else "male") if args.model == _SATB2_MODEL
                else os.path.basename(args.model.rstrip("/")))
        print(f"[arrange] rendered {v} (median MIDI {med}, {tone})", file=sys.stderr)

    L = min([len(mel48)] + [len(y) for y in voices48.values()])
    mix = mel48[:L].copy()                                         # dry melody on top
    harm = np.sum([voices48[v][:L] for v in arranger.VOICES], axis=0)
    mix = mix + harm / np.sqrt(len(arranger.VOICES))               # constant-power harmony sum
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    out = f"{base}_arranged.wav"
    sf.write(out, mix, DEV_SR)
    print(out)


if __name__ == "__main__":
    main()
