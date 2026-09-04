"""Offline JSON API for the studio (arrange) app. Wraps the validated backend; no engine/DSP change.
Editing lives in the frontend on a JSON 'score'; this module only transcribes a hummed melody,
renders EDITED notes into the user's voice, and writes MIDI.

score = {
  "key":    {"root": 0..11, "minor": false},
  "chords": [ {"start": s, "end": e, "degree": 0..6} … ],   # guidance/highlight only
  "melody": [ {"start": s, "dur": d, "midi": m} … ],         # the sung lead (first-class)
  "voices": { "Bass":[…], "Tenor":[…], "Alto":[…], "Sop":[…] }   # notes {start,dur,midi}
}
"""
import os
import sys

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, os.path.dirname(__file__))                # server/ on path
import arranger
import score_export
from arrange import _render_voice, _melody_tracks, _tuning_offset, _detect_key, DEV_SR
from beatrice_solo_choir_live import _SATB2_MODEL
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from bridge import DEFAULT_MODEL as USER_MODEL                # the user's own voice ("choir of one")


def _load_mel(melody_wav):
    """melody wav -> (48k mono, 16k mono). 16k feeds the converter; 48k is the passthrough melody."""
    mel, sr = sf.read(melody_wav, dtype="float32", always_2d=True)
    mel48 = mel.mean(axis=1)
    if sr != DEV_SR:
        mel48 = np.asarray(soxr.resample(mel48, sr, DEV_SR, quality="HQ"), dtype=np.float32)
    mel16 = np.asarray(soxr.resample(mel48, DEV_SR, 16000, quality="HQ"), dtype=np.float32)
    return mel48, mel16


def _pack(seq):                                              # (s,e,midi) -> {start,dur,midi}
    return [{"start": s, "dur": e - s, "midi": int(m)} for (s, e, m) in seq]


def _unpack(seq):                                           # {start,dur,midi} -> (s,e,midi)
    return [(n["start"], n["start"] + n["dur"], int(n["midi"])) for n in seq]


_BP_MODEL = None


def _bp_model():
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


def _bp_melody(melody_wav):
    """FAITHFUL monophonic melody via Spotify Basic Pitch (a trained onset/offset note-transcription
    model) + monophonic winner-take-all (resolve overlapping notes by amplitude -> one melodic line)
    + octave correction. Chromatic (NOT snapped to a detected key) -> reproduce what was actually
    sung, fast or slow. Returns [(start,end,midi)…]."""
    import os as _os
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from basic_pitch.inference import predict
    _, _, ev = predict(melody_wav, _bp_model(), onset_threshold=0.6, frame_threshold=0.3,
                       minimum_note_length=70, minimum_frequency=80, maximum_frequency=700,
                       melodia_trick=True)
    raw = [(float(s), float(e), int(p), float(a)) for (s, e, p, a, *_) in ev]
    if not raw:
        return []
    raw.sort(key=lambda n: -n[3])                          # loudest first
    kept = []
    for n in raw:
        if not any(n[0] < k[1] and k[0] < n[1] for k in kept):   # keep only non-overlapping → monophonic line
            kept.append(n)
    kept.sort(key=lambda n: n[0])
    pitches = sorted(p for _, _, p, _ in kept)
    med = pitches[len(pitches) // 2]
    out = []
    for s, e, p, a in kept:                                # pull octave-error outliers back toward the median
        if med - p >= 8 and abs(med - (p + 12)) < abs(med - p):
            p += 12
        elif p - med >= 8 and abs(med - (p - 12)) < abs(med - p):
            p -= 12
        out.append([s, e, p])
    # despike ONLY (no merge!): a brief ≤2-semitone blip between two EQUAL pitches is vibrato, not a
    # syllable → snap its pitch to the neighbours but KEEP its onset. Repeated same-pitch syllables
    # (the syllables yi-er-san sung on one note = 3 notes) are preserved: each
    # onset stays its own note. Chinese characters below name the actual sung
    # takes these numbers were measured on, and are kept as identifiers.
    for i in range(1, len(out) - 1):
        if (out[i][1] - out[i][0]) < 0.11 and out[i-1][2] == out[i+1][2] and abs(out[i][2] - out[i-1][2]) <= 2:
            out[i][2] = out[i-1][2]
    return [tuple(n) for n in out]


_SF0_RANGE = (36, 84)          # C2..C6 vocal range — drops sub-bass/whistle noise
_SF0_MIN_VOICED_FR, _SF0_MIN_NOTE_S = 3, 0.05
_SF0_HOP = 512
_SF0_DIP_KEEP = 0.85           # a same-pitch boundary is a real re-articulation (split kept) when the
                              # energy dips below this fraction of the surrounding level; otherwise the
                              # energy is continuous = one held note → merge. Lower = split less.
                              # 0.85 measured 2026-07-06 on the "la-la-la" takes: real re-attacks (/l/) dip to
                              # 0.60-0.81, mid-vowel wobble onsets stay >=0.92; 0.55 merged every "la".
_SF0_SPEC_SPLIT = 45.0         # 2nd channel for VOWEL-boundary re-attacks (yi -> er: no consonant, energy
                              # never dips): MFCC distance across the boundary. Measured on the counting take
                              # yi-er-san-si-wu-liu-qi-ba:
                              # real syllable changes 49-159, held-vowel wobble onsets 21-29.


def _sf0_onsets(y, sr):
    """Onset times = superflux + RMS-novelty (the RMS channel catches same-pitch re-attacks). Raw and
    over-eager on purpose; _swiftf0_melody decides which boundaries are real via the energy dip."""
    import librosa, numpy as _np
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_SF0_HOP, lag=2, max_size=3)
    nov = _np.maximum(0.0, _np.diff(librosa.feature.rms(y=y, hop_length=_SF0_HOP)[0], prepend=0))
    frs = set(librosa.onset.onset_detect(onset_envelope=flux, sr=sr, hop_length=_SF0_HOP, backtrack=True))
    frs |= set(librosa.onset.onset_detect(onset_envelope=nov, sr=sr, hop_length=_SF0_HOP, backtrack=True))
    out = []
    for fr in sorted(frs):
        ti = round(float(librosa.frames_to_time(fr, sr=sr, hop_length=_SF0_HOP)), 3)
        if not out or ti - out[-1] >= 0.06:                    # dedup attacks <60ms apart
            out.append(ti)
    return out


def _pitch_change_bounds(t, midi, min_run_s=0.12, theta=0.0):
    """Boundaries where the sung pitch SETTLES on a new semitone: a legato note change carries no
    onset (energy continuous) so _sf0_onsets misses it and the old segmentation merged both notes
    into one median pitch (found 2026-07-06: two ~1 st intra-note stretches on a clean take). A run
    of >=min_run_s voiced frames holding one pitch (±0.6 st around a slow-EMA ref), settling >=1 st
    away from the previously settled pitch -> boundary at the run start."""
    import numpy as _np
    out, settled = [], None
    dt = float(_np.median(_np.diff(t))) if len(t) > 1 else 0.01
    need = max(3, int(round(min_run_s / dt)))
    for i in range(need - 1, len(midi)):
        win = midi[i - need + 1:i + 1]
        if _np.isnan(win).any():                 # windowed (not run-based): an EMA reference GLIDES
            continue                             # along a slow slide and never re-settles (v1 bug)
        if float(win.max() - win.min()) > 0.8:   # window not stable on one pitch yet
            continue
        pc = int(round(float(_np.median(win)) - theta))   # θ: quantise on the singer's tuning centre —
        if settled is not None and pc != settled:         # wobble across a raw x.5 boundary otherwise
            out.append(round(float(t[i - need + 1]), 3))  # fires phantom note changes
        settled = pc
    return out


def _swiftf0_melody(melody_wav):
    """FAITHFUL monophonic melody via SwiftF0 (ONNX f0, accurate low male register, no octave jumps)
    + onset splitting. Runs in-process (no TF env). A note is split at a pitch change, or at a same-
    pitch boundary only when the ENERGY DIPS (a real re-articulation — new syllable/breath); a held
    note with continuous energy stays one note (no vibrato/wobble over-split). So yi-er-san on one pitch
    splits (each syllable dips) but a sustained vowel doesn't. Returns [(start,end,midi)…]; [] on any
    failure so the caller falls back to Basic Pitch."""
    import numpy as _np, librosa, swift_f0
    try:
        y, sr = sf.read(melody_wav)
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(_np.float32)
        r = swift_f0.SwiftF0().detect_from_array(y, sr)
        t = _np.asarray(r.timestamps); f = _np.asarray(r.pitch_hz); c = _np.asarray(r.confidence)
        midi = _np.full(len(f), _np.nan)
        ok = (c > 0.5) & (f > 0)
        midi[ok] = 69 + 12 * _np.log2(f[ok] / 440.0)
        rms = librosa.feature.rms(y=y, hop_length=_SF0_HOP)[0]
        rms_t = librosa.frames_to_time(_np.arange(len(rms)), sr=sr, hop_length=_SF0_HOP)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=_SF0_HOP, n_mfcc=13)[1:]  # drop c0 (energy)
        theta = _tuning_offset(midi)           # take-wide tuning centre (see arrange._tuning_offset)
        bounds = sorted(set([0.0] + _sf0_onsets(y, sr) + _pitch_change_bounds(t, midi, theta=theta)
                            + [float(t[-1])]))
    except Exception:
        return []

    def _reart(boundary, s0, s1):                              # real re-articulation at `boundary`?
        win = (rms_t >= boundary - 0.04) & (rms_t <= boundary + 0.04)
        around = (rms_t >= s0) & (rms_t <= s1)
        if not win.any() or not around.any():
            return True
        level = float(_np.median(rms[around]))
        if level > 1e-6 and float(rms[win].min()) / level < _SF0_DIP_KEEP:
            return True                                        # energy dips -> consonant re-attack
        ia = (rms_t >= boundary - 0.10) & (rms_t <= boundary - 0.02)
        ib = (rms_t >= boundary + 0.02) & (rms_t <= boundary + 0.10)
        if not ia.any() or not ib.any():
            return False
        jump = float(_np.linalg.norm(mfcc[:, ia].mean(1) - mfcc[:, ib].mean(1)))
        return jump > _SF0_SPEC_SPLIT                          # vowel changed -> re-attack w/o a dip

    def _seg_pitch(a, b):                                      # trimmed-median MIDI of [a,b), or None
        lo = a + min(0.08, (b - a) * 0.25)                     # trim the entry scoop / exit fall so
        hi = b - min(0.04, (b - a) * 0.15)                     # slides don't drag the median off-pitch
        m = midi[(t >= lo) & (t < hi)]; m = m[~_np.isnan(m)]
        if len(m) < _SF0_MIN_VOICED_FR:                        # too short after trim -> use the full segment
            m = midi[(t >= a) & (t < b)]; m = m[~_np.isnan(m)]
        if len(m) < _SF0_MIN_VOICED_FR:
            return None
        return int(round(float(_np.median(m)) - theta))   # quantise on the singer's tuning centre

    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < _SF0_MIN_NOTE_S:
            continue
        nm = _seg_pitch(a, b)
        if nm is None or not (_SF0_RANGE[0] <= nm <= _SF0_RANGE[1]):
            continue
        # trim the note to its VOICED extent: bounds are onset-to-onset, so a short sung syllable
        # followed by a rest was drawn as long as the whole gap (a 0.1 s "yi" became a 0.53 s note and
        # skewed the visual rhythm, 2026-07-06). Interior unvoiced flickers are untouched.
        sel = (t >= a) & (t < b)
        vt = t[sel][~_np.isnan(midi[sel])]
        if len(vt) < _SF0_MIN_VOICED_FR:
            continue
        a, b = max(a, float(vt[0]) - 0.01), min(b, float(vt[-1]) + 0.02)
        if b - a < _SF0_MIN_NOTE_S:
            continue
        if out and abs(nm - out[-1][2]) <= 1 and not _reart(a, out[-1][0], b):
            out[-1][1] = round(float(b), 3)                    # ~same pitch + continuous energy → one note
            merged = _seg_pitch(out[-1][0], float(b))          # RE-EVALUATE the merged span: keeping the
            if merged is not None:                             # first fragment's pitch let a 0.1s entry
                out[-1][2] = merged                            # scoop mislabel a 7s note (2026-07-06)
        else:
            out.append([round(float(a), 3), round(float(b), 3), nm])
    # despike (same rule as _bp_melody): a brief ≤2-st blip between two EQUAL pitches is a consonant's
    # pitch bend, not a note → snap its pitch to the neighbours but KEEP its onset (the syllable stays)
    for i in range(1, len(out) - 1):
        if (out[i][1] - out[i][0]) < 0.11 and out[i-1][2] == out[i+1][2] and abs(out[i][2] - out[i-1][2]) <= 2:
            out[i][2] = out[i-1][2]
    # absorb leap transits (right-to-left): a <0.15s DIFFERENT-pitch fragment within 3 st of a longer
    # note it runs straight into is the singer's undershoot/correction landing on a leap (measured on
    # on "jiu": 51 -> dive 44.4 -> settle 45.8 became 3 phantom notes), not notes -> merge forward KEEPING the
    # landing pitch (a median over the transit would drag the note flat). Same-pitch shorts are NOT
    # touched here: those are real fast syllables (the fast "la" take has genuine 0.13 s syllables).
    changed = True
    while changed:                                     # to convergence: an unabsorbed fragment can mask
        changed = False                                # the leap that the settling test looks back for
        i = len(out) - 2
        while i >= 0:
            cur, nxt = out[i], out[i + 1]
            transit = ((cur[1] - cur[0]) < 0.15 and abs(cur[2] - nxt[2]) <= 3
                       and (nxt[1] - nxt[0]) > (cur[1] - cur[0]))
            # post-leap settling can take ~0.3 s (21:14 on "jiu": 0.28 s of correction before landing): absorb
            # a longer near-landing fragment ONLY when the preceding note confirms a big leap is in
            # progress (no "landing must be longer" here: it can rival the landing's length)
            settling = (i > 0 and abs(out[i - 1][2] - nxt[2]) >= 3
                        and abs(cur[2] - nxt[2]) <= 1 and (cur[1] - cur[0]) < 0.3)
            if (transit or settling) and cur[2] != nxt[2] and (nxt[0] - cur[1]) < 0.05:
                out[i + 1] = [cur[0], nxt[1], nxt[2]]
                del out[i]
                changed = True
            i -= 1
    # leading onset gesture: the FIRST note has no previous note for the transit/settling rules to see,
    # so a short opening fragment survives them. Measured (22:17 take): fry 43.5 → overshoot 55 →
    # settle 52 all inside 0.19s — any median of that chaos is garbage. Merge it into the following
    # longer note (keep the start time: the syllable does begin there).
    if (len(out) >= 2 and (out[0][1] - out[0][0]) < 0.2
            and (out[1][1] - out[1][0]) > (out[0][1] - out[0][0])
            and abs(out[0][2] - out[1][2]) >= 2 and (out[1][0] - out[0][1]) < 0.05):
        out[1] = [out[0][0], out[1][1], out[1][2]]
        out.pop(0)
    # trailing release crack: a final <0.2s fragment jumping >=2 st straight off a >=2x longer note is
    # the voice cracking on the release (measured on the "jiu" tail: 0.16 s at +3.6 st), not a sung note -> drop it
    if (len(out) >= 2 and (out[-1][1] - out[-1][0]) < 0.2
            and abs(out[-1][2] - out[-2][2]) >= 2
            and (out[-2][1] - out[-2][0]) >= 2 * (out[-1][1] - out[-1][0])
            and (out[-1][0] - out[-2][1]) < 0.05):
        out.pop()
    # trailing same-pitch echo: a final <0.12s same-pitch piece right off the last note (a release
    # re-attack) has no following note for the sliver rule to merge it into → merge it backward
    if (len(out) >= 2 and (out[-1][1] - out[-1][0]) < 0.12
            and out[-1][2] == out[-2][2] and (out[-1][0] - out[-2][1]) < 0.05):
        out[-2] = [out[-2][0], out[-1][1], out[-2][2]]
        out.pop()
    # absorb slivers: a <0.12s same-pitch fragment squeezed between two onsets is the NEXT syllable's
    # attack transient, not a note of its own → merge it forward (into the note it starts). Cleans the
    # score/MIDI ("8 syllables = 8 dots"); render-neutral (same pitch = same shift).
    merged = []
    for seg in out:
        if merged and seg[2] == merged[-1][2] and (merged[-1][1] - merged[-1][0]) < 0.12 \
                and abs(seg[0] - merged[-1][1]) < 0.05:
            merged[-1] = [merged[-1][0], seg[1], seg[2]]       # sliver + following note → one note
        else:
            merged.append(list(seg))
    return [(a, b, nm) for a, b, nm in merged]


# ── HMM note decoding (default transcriber since 2026-07-06) ─────────────────────────────────────
# The heuristic _swiftf0_melody above decides every boundary/pitch LOCALLY against hard thresholds —
# 9 takes of whack-a-mole showed each new take lands on the wrong side of some threshold. Here the
# note skeleton is decoded GLOBALLY (Tony/pYIN style): states = rest + one per semitone, emission =
# SwiftF0 frame pitch around note+θ, one-off penalty per note change, Viterbi optimum. Fry blips,
# leap transits and boundary flips become path costs instead of edge cases. Same-pitch syllable
# splitting (energy dip + MFCC jump — the proven part) and voiced-extent trim are applied on top.
_HMM_SIGMA = 0.4        # st: emission width around note+θ (wobble tolerance)
_HMM_CHANGE = -18.0     # log-penalty per note→note change: a 0.2s excursion 1.5 st away can't win a
                        # state; a 0.2s real note >=1 st away still can (~3 logit/frame emission gain)
_HMM_REST = -6.0        # note↔rest transition
_HMM_V_REST = -8.0      # voiced frame while in rest
_HMM_UV_NOTE = -2.5     # unvoiced frame while in a note (interior flicker tolerance)


def _hmm_melody(melody_wav):
    """Monophonic melody via global Viterbi note decoding. Returns [(start,end,midi)…]; [] on failure
    so transcribe() falls back to the heuristic path."""
    import numpy as _np, librosa
    try:
        import swift_f0
        y, sr = sf.read(melody_wav)
        if y.ndim > 1:
            y = y[:, 0]
        y = y.astype(_np.float32)
        r = swift_f0.SwiftF0().detect_from_array(y, sr)
        t = _np.asarray(r.timestamps); hz = _np.asarray(r.pitch_hz); c = _np.asarray(r.confidence)
        ok = (c > 0.5) & (hz > 0)
        midi = _np.where(ok, 69 + 12 * _np.log2(_np.maximum(hz, 1e-6) / 440.0), _np.nan)
        theta = _tuning_offset(midi)
        n = len(t)
        if n < 10 or not ok.any():
            return []
        LO, HI = _SF0_RANGE
        P = HI - LO + 1
        grid = _np.arange(LO, HI + 1) + theta
        E = _np.empty((n, P + 1))
        for i in range(n):
            if ok[i]:
                d = midi[i] - grid
                E[i, 1:] = -0.5 * (d / _HMM_SIGMA) ** 2
                E[i, 0] = _HMM_V_REST
            else:
                E[i, 1:] = _HMM_UV_NOTE
                E[i, 0] = 0.0
        dp = E[0].copy(); bp = _np.zeros((n, P + 1), dtype=_np.int32)
        for i in range(1, n):
            nd = dp[1:]
            o1 = int(_np.argmax(nd)); v1 = nd[o1]
            tmp = nd.copy(); tmp[o1] = -_np.inf
            o2 = int(_np.argmax(tmp)); v2 = tmp[o2]
            cand = _np.empty(P + 1); prev = _np.empty(P + 1, dtype=_np.int32)
            if dp[0] >= v1 + _HMM_REST:
                cand[0] = dp[0]; prev[0] = 0
            else:
                cand[0] = v1 + _HMM_REST; prev[0] = 1 + o1
            from_rest = dp[0] + _HMM_REST
            for k in range(1, P + 1):
                fo_v, fo_i = (v1, 1 + o1) if (k - 1) != o1 else (v2, 1 + o2)
                best, pv = dp[k], k
                if from_rest > best:
                    best, pv = from_rest, 0
                if fo_v + _HMM_CHANGE > best:
                    best, pv = fo_v + _HMM_CHANGE, fo_i
                cand[k] = best; prev[k] = pv
            dp = cand + E[i]; bp[i] = prev
        path = _np.empty(n, dtype=_np.int32); path[-1] = int(_np.argmax(dp))
        for i in range(n - 1, 0, -1):
            path[i - 1] = bp[i, path[i]]
        runs, s = [], 0
        for i in range(1, n + 1):
            if i == n or path[i] != path[s]:
                if path[s] != 0 and (t[min(i, n - 1)] - t[s]) >= 0.05:
                    runs.append([float(t[s]), float(t[i - 1]) + 0.01, LO + int(path[s]) - 1])
                s = i
        if not runs:
            return []
        # leading onset gesture: Viterbi's INITIAL state is free, so an opening scoop can win a state
        # that _HMM_CHANGE forbids mid-take — merge a short opening run into its adjacent follower
        if (len(runs) >= 2 and (runs[0][1] - runs[0][0]) < 0.25
                and (runs[1][0] - runs[0][1]) < 0.1 and abs(runs[0][2] - runs[1][2]) <= 3):
            runs[1] = [runs[0][0], runs[1][1], runs[1][2]]
            runs.pop(0)
        # post-leap settling across a state boundary: keep the LATER (settled) pitch
        i = 0
        while i + 1 < len(runs):
            a, b = runs[i], runs[i + 1]
            leap = i == 0 or abs(runs[i - 1][2] - b[2]) >= 3
            if leap and (b[0] - a[1]) < 0.1 and abs(a[2] - b[2]) <= 1 and (a[1] - a[0]) < 0.35:
                runs[i] = [a[0], b[1], b[2]]
                del runs[i + 1]
            else:
                i += 1
        # same-pitch syllable split inside each run (proven: onsets validated by dip OR MFCC jump)
        on = _sf0_onsets(y, sr)
        rms = librosa.feature.rms(y=y, hop_length=_SF0_HOP)[0]
        rms_t = librosa.frames_to_time(_np.arange(len(rms)), sr=sr, hop_length=_SF0_HOP)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=_SF0_HOP, n_mfcc=13)[1:]

        def _reart(boundary, s0, s1):
            win = (rms_t >= boundary - 0.04) & (rms_t <= boundary + 0.04)
            around = (rms_t >= s0) & (rms_t <= s1)
            if not win.any() or not around.any():
                return True
            level = float(_np.median(rms[around]))
            if level > 1e-6 and float(rms[win].min()) / level < _SF0_DIP_KEEP:
                return True
            ia = (rms_t >= boundary - 0.10) & (rms_t <= boundary - 0.02)
            ib = (rms_t >= boundary + 0.02) & (rms_t <= boundary + 0.10)
            if not ia.any() or not ib.any():
                return False
            return float(_np.linalg.norm(mfcc[:, ia].mean(1) - mfcc[:, ib].mean(1))) > _SF0_SPEC_SPLIT

        notes = []
        for (a, b, m) in runs:
            # end margin 0.02 not 0.05: a last syllable right before the run boundary ("ba" at 2.66 with
            # the run ending 2.71) must still get its cut
            cuts = [a] + [o for o in on if a + 0.03 < o < b - 0.02 and _reart(o, a, b)] + [b]
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                notes.append([s0, s1, m])
        # absorb same-pitch slivers forward (attack transients), then voiced-extent trim
        merged = []
        for seg in notes:
            if merged and seg[2] == merged[-1][2] and (merged[-1][1] - merged[-1][0]) < 0.12 \
                    and abs(seg[0] - merged[-1][1]) < 0.05:
                merged[-1] = [merged[-1][0], seg[1], seg[2]]
            else:
                merged.append(list(seg))
        out = []
        for (a, b, m) in merged:
            sel = (t >= a) & (t < b)
            vt = t[sel][~_np.isnan(midi[sel])]
            if len(vt) < _SF0_MIN_VOICED_FR:
                continue
            a2, b2 = max(a, float(vt[0]) - 0.01), min(b, float(vt[-1]) + 0.02)
            # snap the start back to the energy attack: pitch confidence rises on the VOWEL, ~20-60ms
            # after the perceptual onset (consonant) — the "starts feel late" report
            pre = [o for o in on if a2 - 0.06 <= o <= a2 + 0.005]
            if pre:
                a2 = max(pre[-1], out[-1][1] if out else 0.0)
            if b2 - a2 >= _SF0_MIN_NOTE_S:
                out.append([round(a2, 3), round(b2, 3), int(m)])
        # trailing release: crack (short jump off a longer final note → drop) / same-pitch echo (merge back)
        if (len(out) >= 2 and (out[-1][1] - out[-1][0]) < 0.2
                and abs(out[-1][2] - out[-2][2]) >= 2
                and (out[-2][1] - out[-2][0]) >= 2 * (out[-1][1] - out[-1][0])
                and (out[-1][0] - out[-2][1]) < 0.08):
            out.pop()
        if (len(out) >= 2 and (out[-1][1] - out[-1][0]) < 0.12
                and out[-1][2] == out[-2][2] and (out[-1][0] - out[-2][1]) < 0.08):
            out[-2] = [out[-2][0], out[-1][1], out[-2][2]]
            out.pop()
        return [(a, b, m) for a, b, m in out]
    except Exception:
        return []


def transcribe(melody_wav, key_root=None, style="choral"):
    """Hummed melody wav -> score JSON (key auto-detected if key_root is None)."""
    _, mel16 = _load_mel(melody_wav)
    if key_root is None:
        key_root = _detect_key(mel16)
    notes = (_hmm_melody(melody_wav)                            # global Viterbi decode (default)
             or _swiftf0_melody(melody_wav)                     # heuristic pipeline (fallback)
             or _bp_melody(melody_wav)
             or score_export.wav_to_notes(melody_wav, key_root, False))
    voices, chords = arranger.arrange_with_chords(notes, key_root, max_shift=12, style=style)  # wider voicing = clearer harmony (8 clustered the voices)
    return {"key": {"root": int(key_root), "minor": False},
            "chords": [{"start": s, "end": e, "degree": int(d)} for (s, e, d) in chords],
            "melody": _pack(notes),
            "voices": {v: _pack(voices[v]) for v in arranger.VOICES}}


def _render_voice_naive(mel48, score, mel_score):
    """BASELINE render (spec 2026-07-07): per harmony note, phase-vocoder pitch-shift the SAME
    melody segment by (target − sung) semitones — SmartHarmonizerV2's algorithm family, score-
    driven. No formant preservation, no inference: that IS the naive baseline. 5 ms edge fades
    are splice plumbing (click removal), not enhancement."""
    import librosa
    if not mel_score:
        raise ValueError("naive method needs a transcribed melody in the score")
    out = np.zeros(len(mel48), dtype=np.float32)
    fade = int(0.005 * DEV_SR)
    for (s, e, tgt) in score:
        best, mel = 0.0, None
        for (ms, me, mid) in mel_score:                       # melody note with max overlap
            ov = min(e, me) - max(s, ms)
            if ov > best:
                best, mel = ov, mid
        i0, i1 = int(s * DEV_SR), min(int(e * DEV_SR), len(mel48))
        if mel is None or i1 <= i0:
            continue
        y = librosa.effects.pitch_shift(mel48[i0:i1], sr=DEV_SR, n_steps=float(tgt - mel))
        y = np.asarray(y, dtype=np.float32)
        f = min(fade, len(y) // 2)
        if f > 0:
            y[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
            y[len(y) - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        out[i0:i0 + len(y)] += y
    return out


def render_score(score, melody_wav, out_dir, model=USER_MODEL, parts=None, ensemble=False,
                 method="beatrice"):
    """Render the EDITED score into the user's voice. Melody = raw passthrough; the harmony voices
    are the melody recording pitch-shifted (abs_target_midi) per their edited notes. `parts` (e.g.
    ["Sop","Bass"]) limits which harmony voices to render; None = all. Returns paths.
    method="naive" swaps ONLY the per-voice conversion for the phase-vocoder baseline
    (_render_voice_naive) — same score, same timing, same mix, identical dry You."""
    mel48, mel16 = _load_mel(melody_wav)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "studio" if method == "beatrice" else "naive")
    voicelist = [v for v in arranger.VOICES if (not parts or v in parts)]
    mel_notes = _unpack(score.get("melody") or []) or None     # score-to-score shift (see _render_voice)
    dev = voiced = None
    if mel_notes:                                   # grid-pinning correction + jump-in-gaps (offline pyin)
        dev, voiced = _melody_tracks(mel16, mel_notes)
    stems, voices48 = {}, {}
    for v in voicelist:
        sc = _unpack(score["voices"][v])
        if method == "naive":
            y = _render_voice_naive(mel48, sc, mel_notes)
        else:
            # satb2: split by PART NAME (Sop/Alto female spk0, Tenor/Bass male spk1), matching live —
            # pitch-median split never fired female on male-range takes (all medians < 60).
            spk = (0 if v in ("Sop", "Alto") else 1) if model == _SATB2_MODEL else 0
            y = _render_voice(mel16, sc, model, spk, mel_score=mel_notes, dev=dev, voiced=voiced)
        voices48[v] = y
        p = f"{base}_{v.lower()}.wav"
        sf.write(p, y, DEV_SR)
        stems[v] = p
    # stems are written raw; the mix RMS-balances each voice + boosts the harmony under the lead.
    mixp = f"{base}_mix.wav"
    mix = _choir_mix(mel48, voices48) if ensemble else _balanced_mix(mel48, voices48)  # ensemble→stereo choir; off→unchanged dry
    sf.write(mixp, mix, DEV_SR)
    return {"stems": stems, "mix": mixp}


# Each SATB voice → which female reference pool to copy. Sop carries the lead melody (sung male range,
# NOT high), so the ALTO reference matches it best — soprano's own (high) pool gave poor kNN neighbours.
_VOICE_REF = {"Sop": "alto", "Alto": "alto", "Tenor": "tenor", "Bass": "bass"}


def render_female_score(score, melody_wav, out_dir, parts=None, model=USER_MODEL):
    """Like render_score, but re-timbres every SATB harmony stem toward a real female reference via
    kNN-VC (offline, MIT). Soprano uses the alto reference (it sits in the male range, not high —
    verified by ear 2026-06-26). Needs server/knn_assets/cache/{alto,tenor,bass}.pt (run
    knn_vc_render.precompute()). Returns paths like render_score. Offline only — never on the live path."""
    import knn_vc_render as knn
    res = render_score(score, melody_wav, out_dir, model=model, parts=parts)
    mel48, _ = _load_mel(melody_wav)
    base = os.path.join(out_dir, "studiofem")
    voices48, stems = {}, {}
    for v, stem in res["stems"].items():
        p = f"{base}_{v.lower()}.wav"
        knn.retimbre(stem, _VOICE_REF.get(v, "alto"), p)                   # 16kHz female stem
        y, sr = sf.read(p)
        y = np.asarray(soxr.resample(y, sr, DEV_SR, quality="HQ"), dtype=np.float32) if sr != DEV_SR else np.asarray(y, np.float32)
        sf.write(p, y, DEV_SR)
        voices48[v] = y
        stems[v] = p
    mixp = f"{base}_mix.wav"; sf.write(mixp, _balanced_mix(mel48, voices48), DEV_SR)
    return {"stems": stems, "mix": mixp}


def render_piano_score(score, out_dir, parts=None):
    """Synthesize the score (melody + selected harmony voices) as a simple PIANO — a clean,
    balanced, neural-free render. The voice render goes thin on big up-shifts (low male voice
    shifted up an octave); piano is a faithful, even alternative for hearing the arrangement.
    Returns paths (same shape as render_score)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "studio")
    voicelist = [v for v in arranger.VOICES if (not parts or v in parts)]
    lines = [("Melody", score["melody"])] + [(v, score["voices"][v]) for v in voicelist]
    end = max([n["start"] + n["dur"] for _, seq in lines for n in seq] + [0.1])
    N = int((end + 0.4) * DEV_SR)

    def piano_note(midi, dur):
        f = 440.0 * 2 ** ((midi - 69) / 12.0)
        n = max(1, int(min(max(dur, 0.1), 2.5) * DEV_SR))         # cap the ring at 2.5s
        t = np.arange(n, dtype=np.float32) / DEV_SR
        y = np.zeros(n, dtype=np.float32)
        for h, a in ((1, 1.0), (2, 0.55), (3, 0.33), (4, 0.2), (5, 0.12), (6, 0.07)):
            y += a * np.sin(2 * np.pi * f * h * t, dtype=np.float32)
        env = np.exp(-3.2 * t, dtype=np.float32)                  # piano-ish exp decay
        att = np.minimum(1.0, t / 0.006)                          # short attack
        return y * env * att

    stems, mix = {}, np.zeros(N, dtype=np.float32)
    for name, seq in lines:
        line = np.zeros(N, dtype=np.float32)
        for nt in seq:
            w = piano_note(int(nt["midi"]), float(nt["dur"]))
            i0 = int(nt["start"] * DEV_SR); i1 = min(N, i0 + len(w))
            line[i0:i1] += w[:i1 - i0]
        r = float(np.sqrt(np.mean(line * line))) if len(line) else 0.0
        if r > 1e-4:
            line *= 0.12 / r                                      # balance every line to the same level
        if name != "Melody":
            p = f"{base}_{name.lower()}.wav"; sf.write(p, line, DEV_SR); stems[name] = p
        mix += line
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix /= peak
    mixp = f"{base}_mix.wav"
    sf.write(mixp, mix, DEV_SR)
    return {"stems": stems, "mix": mixp}


_LEAD_FWD = 1.0     # melody level vs the per-voice reference (was 1.3 = lead pushed forward; 1.0 lets
                    # the harmony sit louder under it, per request)
_HARM_GAIN = 1.3    # extra boost on the harmony sum so the parts read clearly under the lead
_VOICE_LEVEL = {"Sop": 1.15, "Alto": 1.05, "Tenor": 0.92, "Bass": 0.85}  # high parts slightly forward of low


def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _balanced_mix(mel48, voices48):
    """melody passthrough + RMS-balanced, gained harmony sum → mix array. Each voice is normalised to
    a common level then weighted by _VOICE_LEVEL (high parts slightly louder); the harmony is boosted
    (_HARM_GAIN) and the lead no longer pushed forward (_LEAD_FWD) so the voices sit louder."""
    rmss = {v: _rms(voices48[v]) for v in voices48}
    ref = float(np.median([r for r in rmss.values() if r > 1e-4])) if rmss else 0.0
    vv = {v: (voices48[v] * min(ref / rmss[v], 4.0) if ref > 1e-4 and rmss[v] > 1e-4 else voices48[v])
             * _VOICE_LEVEL.get(v, 1.0)
          for v in voices48}
    L = min((len(y) for y in vv.values()), default=len(mel48))
    harm = (np.sum([vv[v][:L] for v in vv], axis=0) / np.sqrt(max(1, len(vv))) * _HARM_GAIN
            if vv else np.zeros(L, dtype=np.float32))
    mix = np.zeros(L, dtype=np.float32)
    Lm = min(L, len(mel48)); mel = mel48[:Lm].copy(); mr = _rms(mel)
    if ref > 1e-4 and mr > 1e-4:
        mel = mel * min((ref * _LEAD_FWD) / mr, 4.0)
    mix[:Lm] += mel; mix += harm
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix /= peak
    return mix


# --- choir-blend (ensemble): make each part a small section; fully offline, no extra inference ---
_ENS_DETUNE = (-9.0, 9.0)        # cents for the two added copies per voice (small group)
_ENS_DELAY  = (0.018, 0.024)     # seconds: micro-timing jitter for the two copies
_ENS_COPY_GAIN = 0.8             # added copies sit under the dry original
_ENS_WET = 0.22                  # reverb wet fraction
_ENS_DECAY = 4.5                 # hall IR decay (1/s)
_ENS_IR_SEC = 1.5


def _detune(y, cents):
    r = 2.0 ** (cents / 1200.0)                                  # >1 = up; resample to DEV_SR/r then read at DEV_SR
    z = np.asarray(soxr.resample(y, DEV_SR, DEV_SR / r, quality="HQ"), dtype=np.float32)
    if len(z) < len(y):
        z = np.pad(z, (0, len(y) - len(z)))
    return z[:len(y)]


def _thicken(y):                                                # dry + 2 detuned/jittered copies (mono) → a fuller voice
    out = y.astype(np.float32).copy()
    for c, d in zip(_ENS_DETUNE, _ENS_DELAY):
        z = _detune(y, c) * _ENS_COPY_GAIN
        n = int(d * DEV_SR)
        z = np.concatenate([np.zeros(n, dtype=np.float32), z])[:len(y)]
        out += z
    return out


def _hall_ir():                                                 # deterministic, decorrelated L/R → space + width
    rng = np.random.RandomState(7)
    n = int(_ENS_IR_SEC * DEV_SR)
    env = np.exp(-np.arange(n) / DEV_SR * _ENS_DECAY).astype(np.float32)
    ir = np.stack([rng.randn(n).astype(np.float32) * env,
                   rng.randn(n).astype(np.float32) * env], axis=1)
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True) + 1e-9)  # unit energy per channel → predictable wet level
    return ir


def _choir_mix(mel48, voices48):
    from scipy.signal import fftconvolve
    thick = {v: _thicken(voices48[v]) for v in voices48}
    mono = _balanced_mix(mel48, thick)                          # reuse the EXACT dry balancing (lead/harmony levels)
    ir = _hall_ir()
    wetL = fftconvolve(mono, ir[:, 0])[:len(mono)].astype(np.float32)
    wetR = fftconvolve(mono, ir[:, 1])[:len(mono)].astype(np.float32)
    out = np.stack([mono * (1.0 - _ENS_WET) + wetL * _ENS_WET,
                    mono * (1.0 - _ENS_WET) + wetR * _ENS_WET], axis=1).astype(np.float32)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out /= peak
    return out


def _stem_audio(path):
    y, _ = sf.read(path)
    return np.asarray(y if y.ndim == 1 else y[:, 0], dtype=np.float32)


def others_backing(stems, melody_wav, exclude_voice, out_path):
    """Backing for re-recording `exclude_voice`: melody + every OTHER stem, balanced & mixed. The
    singer overdubs the muted part against this. Writes out_path (DEV_SR). Returns out_path."""
    mel48, _ = _load_mel(melody_wav)
    voices48 = {v: _stem_audio(p) for v, p in stems.items() if v != exclude_voice}
    sf.write(out_path, _balanced_mix(mel48, voices48), DEV_SR)
    return out_path


def remix_with_take(stems, melody_wav, replaced_voice, take_wav, out_dir):
    """Replace one voice's rendered stem with the RAW re-recorded take (the singer's real voice, as-is)
    and re-mix. Other stems are reused unchanged. Returns paths like render_score."""
    mel48, _ = _load_mel(melody_wav)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, "studio")
    voices48, out_stems = {}, {}
    for v, stem in stems.items():
        if v == replaced_voice:
            y, sr = sf.read(take_wav)
            y = y.mean(axis=1) if y.ndim > 1 else y
            if sr != DEV_SR:
                y = soxr.resample(y, sr, DEV_SR, quality="HQ")
            y = np.asarray(y, dtype=np.float32)
        else:
            y = _stem_audio(stem)
        p = f"{base}_{v.lower()}.wav"; sf.write(p, y, DEV_SR)
        voices48[v] = y; out_stems[v] = p
    mixp = f"{base}_mix.wav"; sf.write(mixp, _balanced_mix(mel48, voices48), DEV_SR)
    return {"stems": out_stems, "mix": mixp}


def score_to_midi(score, out_path):
    """Write the score (melody + 4 voices) as a Type-1 MIDI (free timing). Returns out_path."""
    GM = {"Bass": 42, "Tenor": 40, "Alto": 40, "Sop": 40}    # placeholder GM voices (cello/strings)
    tracks = [("Melody", 53, _unpack(score["melody"]))]      # 53 = Voice Oohs
    tracks += [(v, GM[v], _unpack(score["voices"][v])) for v in arranger.VOICES]
    score_export.write_midi(out_path, tracks)
    return out_path
