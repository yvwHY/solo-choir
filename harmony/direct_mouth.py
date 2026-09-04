"""direct_mouth.py — the direct-drive voice: the singer's own audio goes into
DDSP with f0 replaced by the model's target line, bypassing WORLD

Motivation (2026-07-26, after every version was judged "auto-tuned" in a full
listening pass): the current chain is **model, then WORLD synthesising the parts,
then DDSP converting**. F19 had already shown by ablation that the auto-tuned
quality is not the f0 curve recipe, since H1, H2 and H3 humanisation all failed,
but the carrier — and **what had never been tested is that the carrier is the
WORLD stage in the middle of the chain**. DDSP was only faithfully converting a
signal that already carried WORLD's buzz.

This voice removes WORLD entirely:
  input audio = the singer's original take, so the content units carry their real
                diction, phonemes and breath
  f0          = the model's target line (F16: the parts always sound the target
                pitch)
  output      = their timbre, singing their words, at the parts' pitch

The implementation requires no change to the DDSP repository: main.py:177-199 has
an f0 cache keyed by the input md5 and the parameters, so writing the target f0
into the cache file makes main.py load it instead of extracting pitch.

Usage (the vcclient-dev environment, from harmony/):
  python direct_mouth.py --notes out/angel_v2_world4keyed_notes.json \
      --take ../../../260722_harmony_brain/data/take.wav --tag direct30k
  (this prints the DDSP command to run next; --run runs it directly)

Since 2026-07-30 the voicing gate is on by default (`--respect-unvoiced`, see
voicing_mask): the singer's unvoiced frames are given f0 = 0, so breaths and
fricatives are no longer sung. The old behaviour is `--no-respect-unvoiced`.
"""
import argparse, hashlib, json, math, os, subprocess, sys
import numpy as np
import soundfile as sf

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
HOP = 512            # DDSP combsub config: block_size 512 @ 44100
PORTA = 0.15         # the same one-pole portamento as world_mouth, per hop
VIB_HZ, VIB_SEMI = 5.0, 0.12


def load_notes(path):
    d = json.load(open(path))
    if "notes" in d:
        return d["notes"], int(d["step_samps"]), d.get("heard")
    return d["targets"], int(round(d["tick_sec"] * SR)), None


def target_f0(notes, tick, n_hops, porta_ms=33.0, leap_snap=4,
              vib_hz=VIB_HZ, vib_phase=0.0, vib_onset_ms=0.0, vib_semi=VIB_SEMI):
    """The model's note line to a target f0 in Hz per hop, holding the previous
    value through silence, which is DDSP's uv_interp convention: the f0 curve may
    not contain zero holes, and the volume mask handles silence separately.

    Portamento on demand, changed after the listening feedback of 2026-07-28 that
    "the female voice slides on every note in all three cells": only steps of
    leap_snap semitones or fewer inside a phrase glide, through a one-pole with a
    time constant of porta_ms, which is world_mouth's 33 ms. The old
    PORTA = 0.15 per hop, on an 11.6 ms hop, gave 2.3 times world's glide. Leaps
    and note changes across a rest land directly; gliding across a rest scooped
    from the last note of the previous phrase into the head of the new one, by up
    to a measured 10 semitones.

    vib_hz, vib_phase in cycles and vib_onset_ms desynchronise the vibrato across
    parts and delay its onset, from the auto-tune feedback of 2026-07-28: two
    parts on sines at the same phase and rate are mechanically synchronised,
    while a real singer holds a note straight before the vibrato arrives. The
    defaults are the original behaviour."""
    hop_ms = 1000.0 * HOP / SR
    alpha = 1.0 - math.exp(-hop_ms / porta_ms) if porta_ms > 0 else 1.0
    f0 = np.zeros(n_hops)
    cur, prev_m, gap, age = 0.0, None, False, 0
    for k in range(n_hops):
        t = min(int(k * HOP / tick), len(notes) - 1)
        m = notes[t]
        if m is None:
            f0[k] = cur          # hold the last pitch; unvoiced spans are silenced by the volume mask
            gap = gap or prev_m is not None
            continue
        tgt = 440.0 * 2 ** ((m - 69) / 12.0)
        if m != prev_m or gap:
            if (cur == 0.0 or prev_m is None or gap
                    or (leap_snap and abs(m - prev_m) > leap_snap)):
                cur = tgt        # land directly: the head of a phrase, across a rest, or a leap
            prev_m, age = m, 0
        gap = False
        age += 1
        cur = cur + (tgt - cur) * alpha
        depth = vib_semi * (min(1.0, age * hop_ms / vib_onset_ms)
                            if vib_onset_ms > 0 else 1.0)
        vib = 2 ** (depth * math.sin(
            2 * math.pi * (vib_hz * k * HOP / SR + vib_phase)) / 12)
        f0[k] = cur * vib
    if f0[0] == 0:
        nz = np.flatnonzero(f0)
        if len(nz):
            f0[:nz[0]] = f0[nz[0]]
    return f0


def _runs(mask, val):
    """Runs of equal value, as half-open intervals [a, b)."""
    i, out = 0, []
    while i < len(mask):
        if mask[i] == val:
            j = i
            while j < len(mask) and mask[j] == val:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def harvest_f0(x):
    """The single entry point to pyworld.harvest. **The most expensive item per
    phrase**, measured at an RTF of 0.114 on 2026-08-03, and input_agc and
    voicing_conf each called it once while both voices ran the whole thing again,
    so the same audio was harvested four times. Both need the same thing, which
    frames have a fundamental, so it is computed once and passed in.
    Passing f0m is pure caching: the result is byte-identical, not
    approximate."""
    import pyworld
    f0m, _ = pyworld.harvest(np.ascontiguousarray(x), SR,
                             frame_period=1000.0 * HOP / SR)
    return f0m


def extract_f0_uv(x, n_hops, floor_hz=0.0):
    """The singer's real pitch to (f0, voiced), where **unvoiced frames are zero
    throughout**.

    Architecture change of 2026-07-30: the old version filled unvoiced spans with
    the previous value and then had the gate punch holes back through them, which
    destroys and then repairs, leaving two versions of the truth that disagree.
    There is now one source of truth: harvest's voiced and unvoiced flags travel
    in pairs, the octave median repair takes the median of neighbours **within a
    single voiced run only**, never borrowing across an unvoiced span and never
    interpolating, and unvoiced frames are never filled.
    """
    import pyworld
    x = np.ascontiguousarray(x)
    f0m, _ = pyworld.harvest(x, SR, frame_period=1000.0 * HOP / SR)
    v = f0m > 0
    midi = 69 + 12 * np.log2(np.maximum(f0m, 1) / 440.0)
    out = np.zeros(len(f0m))
    for a, b in _runs(v, True):                    # octave repair stays inside a run
        seg = midi[a:b]
        med = np.array([np.median(seg[max(0, i - 4):i + 5])
                        for i in range(len(seg))])
        fixed = np.where(np.abs(seg - med) > 6, med, seg)
        out[a:b] = 440.0 * 2 ** ((fixed - 69) / 12.0)
    if floor_hz > 0:                     # octave fold at the curve layer: the singer's phrase ends sink, and the shift is negative
        low = (out > 0) & (out < floor_hz)  # would fall outside the voice's range; measured, about 5% of hops are below 65 Hz
        while low.any():
            out[low] *= 2.0
            low = (out > 0) & (out < floor_hz)
    f0 = np.zeros(n_hops)
    voiced = np.zeros(n_hops, dtype=bool)
    n = min(n_hops, len(out))
    f0[:n], voiced[:n] = out[:n], v[:n]
    return f0, voiced


def shift_f0(notes, heard, tick, n_hops, x, floor_hz=0.0):
    """Shift mode, the direct-drive port of world_mouth 3b, from tracing the
    auto-tune quality on 2026-07-28: f0 is not synthesised at all. The singer's
    own pitch curve is extracted and transposed tick by tick by the interval
    (part note minus their note). The micro-deviations, the drift, the real
    vibrato and the note transitions are all theirs, sourced from a person and
    not a synthesiser.
    f0 is always zero over unvoiced spans (since 2026-07-30, see extract_f0_uv).
    The porta and vibrato parameters do not apply in this mode."""
    f0_base, _ = extract_f0_uv(x, n_hops, floor_hz)
    shifts, last = [], None
    for n, h in zip(notes, heard):
        if n is None:
            shifts.append(None)
        else:
            if h is not None:
                last = n - h
            shifts.append(last)
    out = np.zeros(n_hops)
    for k in range(n_hops):
        s = shifts[min(int(k * HOP / tick), len(shifts) - 1)]
        if s is not None and f0_base[k] > 0:
            out[k] = f0_base[k] * 2 ** (s / 12.0)   # the rest stay 0, which is native uv
    return out


PER_K_LO, PER_K_HI = 0.45, 0.80
LOCAL_HALF = int(round(3.0 * SR / HOP))      # half-window of the local reference: 3 seconds


def _ramp(v, lo, hi):
    return np.clip((v - lo) / (np.asarray(hi) - np.asarray(lo) + 1e-12), 0.0, 1.0)


def _local_ref(vals, mask, half, q):
    """The q-th percentile of vals over frames where mask is true, within a
    window of plus or minus half; with no samples it falls back to the whole-take
    value."""
    idx = np.flatnonzero(mask)
    glob = np.percentile(vals[idx], q) if len(idx) else 0.0
    out = np.full(len(vals), glob, dtype=float)
    if not len(idx):
        return out
    for k in range(len(vals)):
        w = idx[(idx >= k - half) & (idx <= k + half)]
        if len(w) >= 8:
            out[k] = np.percentile(vals[w], q)
    return out


def voicing_frames(x, k0, k1, win=2048):
    """Per-frame voicing features (per, flat, rms) over the hop range [k0, k1).

    Factored out for pipelining (2026-08-03): each frame looks only at the 2048
    samples in [k*HOP - win/2, k*HOP + win/2), which is **purely local**, so it
    can be computed as soon as the window fills, and the result is byte-identical
    to computing the whole phrase at once, being the same samples through the
    same formula. That means it can be computed while the singer is still
    singing.
    Frames whose window runs past x are left at 0, which matches the semantics of
    the original whole-take loop, where the last few frames were zero anyway."""
    q0, q1 = int(SR / 800), int(SR / 65)
    w = np.hanning(win)
    per, flat, rms = (np.zeros(k1 - k0) for _ in range(3))
    for k in range(k0, k1):
        s = k * HOP - win // 2
        if s < 0 or s + win > len(x):
            continue
        seg = x[s:s + win] * w
        e = float(np.dot(seg, seg))
        rms[k - k0] = np.sqrt((seg ** 2).mean())
        if e < 1e-12:
            continue
        per[k - k0] = (np.correlate(seg, seg, "full")[win - 1:] / e)[q0:q1].max()
        p = np.abs(np.fft.rfft(seg)) ** 2 + 1e-12
        flat[k - k0] = np.exp(np.mean(np.log(p))) / np.mean(p)
    return per, flat, rms


def frames_ready(n_samps, win=2048):
    """With n_samps samples available, how many hops have a full window and can
    therefore be computed already."""
    return max(0, (n_samps - win // 2) // HOP + 1)


def voicing_conf(x, n_hops, win=2048, f0m=None, frames=None):
    """Per-frame voicing confidence in [0, 1], decided by several features
    together, **resolving doubt as unvoiced**.

    Trusting harvest alone is not enough: it calls a fair proportion of breaths
    and fricatives voiced (the p5 periodicity of the take-voiced frames in this
    span is only 0.35, so that group is misclassified). Three features vote:
      per  the autocorrelation periodicity peak over the 65-800 Hz lag band, the
           strongest discriminator (voiced p10 0.72, unvoiced p90 0.62) and the
           dominant vote
      flat spectral flatness, which rises for noise-like frames
      db   frame energy, relative to the p90 of voiced spans
    conf = harvest ? min(per_score, max(flat_score, db_score)) : 0
    min lets the weakest link decide, resolving doubt as unvoiced, while max lets
    a quiet but clean note ending be rescued by flat, so genuinely soft singing is
    not killed. The thresholds were set from the labelled statistics of the
    three-step material.
    """
    per, flat, rms = (voicing_frames(x, 0, n_hops, win) if frames is None
                      else frames)
    db = 20 * np.log10(rms + 1e-9)
    if f0m is None:
        f0m = harvest_f0(x)
    hv = np.zeros(n_hops, dtype=bool)
    n = min(n_hops, len(f0m))
    hv[:n] = f0m[:n] > 0
    # The reference is local rather than whole-take: within one take the level can
    # fall by 5-13 dB, which is the root cause of "after twelve seconds it sounds
    # like singing without being able to breathe" (2026-07-30). Quieter spans have
    # a worse signal-to-noise ratio, so both periodicity and energy fall together,
    # and a fixed whole-take threshold calls the **real singing** of the second
    # half unvoiced. The pass-through then covers what they are actually singing
    # with their raw audio, which is the choking quality. A local reference lets
    # the threshold follow the passage.
    ref_db = _local_ref(db, hv, LOCAL_HALF, 90)
    ref_per = _local_ref(per, hv, LOCAL_HALF, 50)
    conf = np.minimum(_ramp(per, PER_K_LO * ref_per, PER_K_HI * ref_per),
                      np.maximum(_ramp(-flat, -0.010, -0.002),
                                 _ramp(db, ref_db - 26, ref_db - 14)))
    return np.where(hv, conf, 0.0), hv, db - ref_db


def voicing_mask(x, n_hops, min_uv_ms=40.0, guard_ms=0.0, hi=0.60, lo=0.35,
                 anchor_ms=25.0, weak_db=-12.0, f0m=None, frames=None):
    """Voicing confidence to the hard gate used for injection (from the listening
    of 2026-07-30: "cells 2 and 3 both sound clearly auto-tuned on breaths and on
    /s/-type fricatives").

    Why it is hard: the forward pass of CombSubSuperFast takes only units, f0 and
    volume (`ddsp/vocoder.py:735`), and the harmonic-to-noise ratio is predicted
    by the network itself (`split_map`, line 729), so **there is no per-frame
    aperiodicity input**. A genuinely continuous soft gate would mean changing the
    DDSP repository, which is not done here. The next best thing is free: the
    upsample in `ddsp/core.py:66` is linear interpolation, so f0 transitions
    linearly within that block and every gate boundary already has an 11.6 ms
    fade of the exciter. There is no need, and no reason, to ramp f0 by hand;
    a longer ramp becomes an audible downward slide.

    Unvoiced frames are injected with a **1200 Hz sentinel rather than 0**
    (`--uv-f0`). Hard zeroing is acoustically best (a breath measures CPP 1.838
    against 1.855 for the raw audio), but combtooth is sinc(sr*x/(f0+1e-3)): as
    f0 approaches 0 the phase x almost freezes and the argument approaches 0/0, so
    whenever a ramp's phase wrap lands near zero, sinc returns to 1 and produces
    **a full-scale burst** (measured peak 1.0000 with a crest factor of 31.7,
    twice in 26 s). The sentinel keeps the phase advancing one unit per sample and
    the argument always well-conditioned (crest 5.8), while 1200 Hz is far above
    the singer's range of 110-300, so the model's harmonic filter cannot render a
    convincing pitch there. Measured, the wideband CPP of a breath over
    800-4000 Hz is 1.583 against 1.581 for the raw audio, so no sentinel hum
    smuggles itself in. The internal representation still treats the (f0, uv) pair
    as the truth; the sentinel is substituted only at the final injection, as an
    expedient for conditioning the vocoder.

    Confidence is turned into a state by Schmitt hysteresis, entering voiced at hi
    and unvoiced at lo, so the boundaries are decided by the actual acoustics.
    That replaces the old fixed guard inset, which was the main cause of the
    residue, since boundary frames were forced back to voiced. min_uv_ms filters
    out isolated drop-outs inside a sustained note, which are the source of the
    gate's grainy quality.
    """
    hop_ms = 1000.0 * HOP / SR
    conf, hv, rel_db = voicing_conf(x, n_hops, f0m=f0m, frames=frames)
    v = np.zeros(n_hops, dtype=bool)
    state = True
    for k in range(n_hops):
        if state and conf[k] < lo:
            state = False
        elif not state and conf[k] > hi:
            state = True
        v[k] = state
    # Anchoring, the root-cause fix for "after twelve seconds it sounds like
    # singing without being able to breathe" (2026-07-30): the gate may only fall
    # where the singer is **really not sounding**, that is, on harvest's unvoiced
    # frames and within anchor_ms of them. The joint vote could previously decide
    # a frame was unvoiced in the middle of a continuous phrase; measured over a
    # 14 s window, their real unvoiced share was 1.7% (they sang throughout) while
    # the gate marked 14.5%, so the pass-through inserted 151 ms of raw audio into
    # a phrase where they had not taken a breath at all, which is the choking
    # quality. The pass-through turns the cost of over-gating from "slightly fewer
    # harmonics" into "the raw voice covering the part", so the asymmetry
    # reverses: better to miss a breath than to open a hole inside a phrase.
    if anchor_ms >= 0:
        allow = ~hv
        # Grow outwards from genuinely unvoiced frames along a chain of
        # consecutive low-confidence frames (third round, 2026-07-30, after "one
        # of the notes sounds metallic"). Very quiet audio just before an attack,
        # rms below 0.005, is still called voiced by harvest, but it produces
        # rubbish f0 jumping 2-8 semitones per hop, and combtooth renders 4-8 hops
        # from it, giving an inharmonic scrape at the head of the phrase (measured
        # inharmonic energy there is 75-99% against 1-6% for the raw audio). That
        # rubbish extends 35-90 ms past the harvest-unvoiced frames, which a fixed
        # 25 ms dilation cannot cover, so it was anchored back to voiced.
        # The rule is now that the whole conf < lo chain connected to genuinely
        # unvoiced frames may be gated: an isolated low-confidence frame in the
        # middle of a phrase still cannot move, so no hole is opened, while the
        # chain of rubbish before an attack is cleared entirely.
        # The chain may only pass through quiet frames: a frame at a normal level
        # is not crossed even at low confidence, or the chain digs into the
        # phrase; unconstrained, it dug a measured 81 ms.
        low = (conf < lo) & (rel_db < weak_db)
        for a, b in _runs(low, True):
            if allow[a:b].any() or (a > 0 and allow[a - 1]) or \
                    (b < n_hops and allow[b]):
                allow[a:b] = True
        d = int(round(anchor_ms / hop_ms))
        if d:
            grown = allow.copy()
            for a, b in _runs(allow, True):
                grown[max(0, a - d):min(n_hops, b + d)] = True
            allow = grown
        v = v | ~allow
    for a, b in _runs(v, False):                   # fill back unvoiced gaps that are too short
        if (b - a) * hop_ms < min_uv_ms:
            v[a:b] = True
    g = max(0, int(round(guard_ms / hop_ms)))
    if g:
        keep = v.copy()
        for a, b in _runs(v, False):
            keep[a:min(b, a + g)] = True
            keep[max(a, b - g):b] = True
        v = keep
    return v.astype(float)


TRAIN_REF_RMS = 0.0566      # median of the upper half of the frames in 260724_ddsp_svc/data/train/volume


def input_agc(x, ref_rms=TRAIN_REF_RMS, win_s=2.0, lo=0.25, hi=16.0, f0m=None):
    """Bring the level of voiced spans up to the training reference before
    conversion, and divide it back out afterwards.

    2026-07-30: with the pass-through having fixed the breaths, it still sounded
    "after twelve seconds like singing without being able to breathe, forced". The
    root cause was neither the gate nor the breaths but **the conversion of the
    voiced spans themselves**: the median voiced rms of this take is 0.0134, below
    the p10 of the training clips (0.0188), and the later part falls to 0.005,
    which is an eighth of the training median. DDSP's volume is one of the model's
    conditioning inputs (`ddsp/vocoder.py:735`, volume_frames), and once the level
    falls outside the training distribution the conversion degrades.

    Evidence: the spectral distance between the converted output and the raw audio
    worsens monotonically with the singer's level (14.3 dB at rms .0195, 18.3 dB
    at .0055), while the CPP of **the raw audio itself** barely changes between the
    two halves (2.42 to 2.34). They did not sing worse; the model was fed an
    unfamiliar level.

    Voiced rms is measured in windows and interpolated linearly into a smooth
    gain, which avoids pumping on breaths. Returns (x_agc, gain); the output side
    multiplies by 1/gain and the level structure is fully restored.
    """
    f0 = harvest_f0(x) if f0m is None else f0m
    v = np.repeat(f0 > 0, HOP)
    v = (np.concatenate([v, np.zeros(len(x) - len(v), dtype=bool)])
         if len(v) < len(x) else v[:len(x)])
    W = int(win_s * SR)
    cs, gs = [], []
    for s in range(0, len(x), W):
        sl = slice(s, min(len(x), s + W))
        m = v[sl]
        if m.sum() < 0.1 * SR:
            continue
        r = np.sqrt((x[sl][m] ** 2).mean())
        if r > 1e-9:
            cs.append(0.5 * (s + min(len(x), s + W)))
            gs.append(ref_rms / r)
    if not gs:
        return x.copy(), np.ones(len(x))
    g = (np.interp(np.arange(len(x)), cs, gs) if len(gs) >= 2
         else np.full(len(x), gs[0]))
    g = np.clip(g, lo, hi)
    peak = float(np.abs(x * g).max())
    if peak > 0.95:                     # scale the whole thing back proportionally; the gain curve keeps its shape
        g = g * (0.95 / peak)
    return x * g, g


def uv_passthrough(ang, x, vm, xfade_ms=15.0, dry_gain=1.0):
    """Use the singer's raw audio directly over unvoiced spans instead of anything
    synthesised (third listening round, 2026-07-30: the 1200 Hz sentinel "turns
    into a harsh, scraping, metallic sound" — the harmonic branch renders a column
    of overtones on a 1200 Hz fundamental, which the CPP instrument over
    800-4000 Hz did not catch and the ear did).

    The direct-drive chain already takes their original take, so breaths and
    breathy consonants **need no synthesis at all**: voiced spans use the
    converted result, unvoiced spans use the raw audio, and the boundary uses the
    gate's hysteresis decision with a raised-cosine crossfade. This takes the whole
    question of what silence should sound like away from the model.

    Level: the raw audio is scaled to the magnitude of the converted output by the
    rms ratio of the voiced spans, which preserves the proportion of breath to
    singing in their own recording rather than forcing breaths to some absolute
    value.
    Honest caveat: the upper part has the friend's timbre while the passed-through
    breath has the singer's. Breaths carry little speaker identity, so this is
    assumed acceptable for now, which is exactly what the listening test is for.

    dry_gain (2026-08-04) is the proportion of raw audio passed through. **Stems
    played on their own, in live_v3 or perform, keep 1.0.** In respond2's response
    mix the singer's dry voice is already present, and two stems each passing
    another copy through would stack the same breath three times; measured,
    breaths and consonants came out 7.7 dB above the dry voice alone. respond2
    therefore passes 0, so the stems are silent over unvoiced spans. The
    raised-cosine ramp is unchanged and the sentinel output is still suppressed, so
    the mine of 2026-07-30 does not come back.
    """
    n = min(len(ang), len(x))
    ang, x = ang[:n], x[:n]
    g = np.repeat(np.asarray(vm, dtype=float), HOP)
    g = (np.concatenate([g, np.full(n - len(g), g[-1] if len(g) else 1.0)])
         if len(g) < n else g[:n])
    half = max(1, int(SR * xfade_ms / 2000.0))
    k = np.hanning(2 * half + 1)
    g = np.convolve(g, k / k.sum(), mode="same")     # binary mask to a raised-cosine ramp
    v = np.repeat(np.asarray(vm, dtype=float), HOP)
    v = (np.concatenate([v, np.ones(n - len(v))]) if len(v) < n else v[:n]) > 0.5
    # Windowed gain: within one take the voiced rms measured a fall from 0.037 to
    # 0.0075, about 13 dB, so a single gain over the whole take would misplace the
    # passed-through breaths of the later part. It is measured every 2 s and
    # interpolated linearly into a smooth curve.
    W = int(2.0 * SR)
    cs, gs = [], []
    for s0 in range(0, n, W):
        sl = slice(s0, min(n, s0 + W))
        m = v[sl]
        if m.sum() < 0.1 * SR:
            continue
        den = np.sqrt((x[sl][m] ** 2).mean())
        if den > 1e-12:
            cs.append(0.5 * (s0 + min(n, s0 + W)))
            gs.append(np.sqrt((ang[sl][m] ** 2).mean()) / den)
    if len(gs) >= 2:
        s = np.clip(np.interp(np.arange(n), cs, gs), 0.25, 4.0)
    else:
        s = np.full(n, gs[0] if gs else 1.0)
    return g * ang + dry_gain * (1.0 - g) * s * x, float(np.median(s))


# Expression-layer parameters, measured on 2026-07-31 over 26.1 s of
# seg_take.wav and 19 long sustained runs; the calibration script is in that
# day's worklog. A sustained cents_sd of 32.57 decomposes into four components
# plus the scoop into a note:
#   bias   per-note offset, sd 22 cents (median |bias| 20, range -37 to +34);
#          the largest contributor
#   drift  slow drift within a note, sd 13 cents, in the band up to 2.5 Hz
#   jit    fast tremor, sd 6 cents (the 9-20 Hz band holds 19% of the residual
#          energy and includes harvest's own measurement noise, so a conservative
#          value is taken)
#   vib    depth 20 cents peak (p25 16.5, p75 29.4), rate 5.9 Hz, rate jitter
#          about 0.9 Hz (zero crossings measured 1.71 but include noise, so it is
#          halved), onset delay about 0 ms (the measured median)
#   bend   the scoop into a note, median |22 cents|, hooked in from below 59% of
#          the time, settling in a median of 35 ms (p75 75 ms)
EXPR = dict(bias_sd=22.0, bias_clip=35.0, drift_sd=13.0, drift_fc=1.5,
            jit_sd=6.0, jit_fc=10.0, vib_hz=5.9, vib_walk_sd=0.9,
            vib_depth=20.0, vib_depth_sd=6.0, vib_depth_lo=8.0,
            vib_depth_hi=32.0, bend_mag=22.0, bend_down=0.59,
            bend_tau_lo_ms=15.0, bend_tau_hi_ms=60.0,
            tune_fc=0.25)   # rate of change of the choir's shared tuning reference, used by tune_lock


def _lp_noise(rng, n, fc, sd):
    """One-pole low-passed white noise to a slowly varying curve of the target
    sd. Deterministic: it takes an external rng."""
    alpha = 1.0 - math.exp(-2 * math.pi * fc * HOP / SR)
    y = np.empty(n)
    acc = 0.0
    for i, w in enumerate(rng.standard_normal(n)):
        acc += (w - acc) * alpha
        y[i] = acc
    s = y.std()
    return y * (sd / s) if s > 1e-9 else y


def expressive_cents(notes, tick, n_hops, leap_snap=4, seed=20260731, p=EXPR,
                     tune_seed=None, tune_lock=0.0):
    """The rule-based expression layer (2026-07-31), the remedy for cause (b),
    "the f0 is twice as straight as a real singer's": a curve of cents deviation
    from the target skeleton made of a per-note offset, a slow drift, a fast
    tremor, an irregular vibrato and a scoop into the note. The parameters copy
    the measured human statistics (see the EXPR comment), and a fixed seed makes a
    render reproducible.

    How this differs from the three failures of F19 (H1, H2 and H3 humanisation):
    at the time, the metallic quality of cause (a) covered everything and contour
    differences were inaudible. Only once (a) was closed could (b) be heard on its
    own, and the second cell of the three-step comparison of 2026-07-30 passed
    blind listening, which showed the f0 texture really is a lever.
    The scoop is added only where the skeleton lands instantly, at the head of a
    phrase, across a rest, or on a leap; steps already have a porta bend and are
    not doubled.

    tune_lock (2026-08-02, additive; 0 is byte-identical to the old behaviour) is
    the choir tuning to itself. This layer used to let each part deviate on its
    own, so the **pitch centres** of the two parts did not agree: the measured
    interval error had an SD of 40 cents and exceeded 20 cents, audibly out of
    tune, 61.6% of the time. It was judged "strange", which also matches the
    residual strangeness that remained on 2026-08-01 after the key work. A real
    choir does not work this way: singers listen to each other and share a tuning
    centre, and what differs between them is the vibrato and the jitter. So the
    layer splits in two:

      slow (the per-note bias plus the drift, that is, the pitch centre) follows a
           shared rng according to tune_lock
      fast (jitter, vibrato rate and phase, and the scoop, that is, the texture)
           is always independent per part

    Passing both parts the same tune_seed with tune_lock = 1 makes the slow layer
    entirely shared, so the pitch-centre error is zero, while the vibrato stays
    desynchronised, preserving the lesson of 2026-07-28 that mechanically
    synchronised vibrato is one source of the auto-tuned quality.
    Energy is conserved: sqrt(L) times the shared component plus sqrt(1-L) times
    the part's own, so the total SD does not change with L and only the error
    **between parts** does. The shared component is drawn from a slow curve in the
    time domain and sampled at the onset of each note, so the two parts agree even
    though they change note at different times; singers tune by clock time, not by
    bar.

    The order of rng draws is deliberately unchanged: the shared component always
    comes from a separate trng, so at tune_lock = 0 the order and number of rng
    calls match the old version exactly, which keeps regime A verifiable."""
    rng = np.random.default_rng(seed)
    trng = np.random.default_rng(seed if tune_seed is None else tune_seed)
    hop_s = HOP / SR
    # Walk the note rhythm of target_f0 once: the note instance id per hop plus
    # the list of instant-landing onsets.
    note_id = np.full(n_hops, -1)
    snaps = []
    nid, prev_m, gap = -1, None, False
    for k in range(n_hops):
        m = notes[min(int(k * HOP / tick), len(notes) - 1)]
        if m is None:
            gap = gap or prev_m is not None
            note_id[k] = nid
            continue
        if m != prev_m or gap:
            if prev_m is None or gap or (leap_snap and abs(m - prev_m) > leap_snap):
                snaps.append(k)
            nid += 1
            prev_m = m
        gap = False
        note_id[k] = nid
    n_notes = nid + 1
    # per-note draws, in fixed order, so it is deterministic
    bias = np.clip(rng.normal(0.0, p["bias_sd"], n_notes),
                   -p["bias_clip"], p["bias_clip"])
    depth = np.clip(rng.normal(p["vib_depth"], p["vib_depth_sd"], n_notes),
                    p["vib_depth_lo"], p["vib_depth_hi"])
    # continuous components
    drift = _lp_noise(rng, n_hops, p["drift_fc"], p["drift_sd"])
    common = 0.0
    if tune_lock:                              # the choir tunes together: the slow layer moves to the shared rng
        L = tune_lock
        # The shared component must be a **continuous** curve and cannot be
        # sampled per note: the two parts change note at different times, so
        # taking different moments of the same curve amounts to drawing
        # separately. Measured on 2026-08-02, a per-note version only brought the
        # slow component down from 27.7 to 14.8 cents, which did not fix it.
        # The whole curve is now shared and each part's own per-note offset fades
        # out with L, because **what sounds human lives in the movement — drift,
        # jitter, vibrato, the scoop — and not in a fixed offset**. In a solo, a
        # constant offset sounds human; across two parts, each offset on its own
        # is simply out of tune.
        common = L * _lp_noise(trng, n_hops, p["tune_fc"], p["bias_sd"])
        sdrift = _lp_noise(trng, n_hops, p["drift_fc"], p["drift_sd"])
        bias = (1.0 - L) * bias
        drift = (1.0 - L) * drift + L * sdrift
    jit = _lp_noise(rng, n_hops, p["jit_fc"], p["jit_sd"])
    rate = p["vib_hz"] + _lp_noise(rng, n_hops, 0.8, p["vib_walk_sd"])
    vib = np.sin(2 * math.pi * np.cumsum(rate) * hop_s)
    dep = np.where(note_id >= 0, depth[np.clip(note_id, 0, None)], 0.0)
    a = 1.0 - math.exp(-hop_s / 0.03)          # 30 ms smoothing of the depth across a note change
    for i in range(1, n_hops):
        dep[i] = dep[i - 1] + (dep[i] - dep[i - 1]) * a
    out = np.where(note_id >= 0, bias[np.clip(note_id, 0, None)], 0.0) \
        + drift + jit + dep * vib + common
    # The scoop: where the skeleton lands instantly, hook in from below (59% of
    # the time) or from above, settling exponentially.
    for k0 in snaps:
        sgn = -1.0 if rng.random() < p["bend_down"] else 1.0
        mag = min(60.0, rng.lognormal(math.log(p["bend_mag"]), 0.5))
        tau = rng.uniform(p["bend_tau_lo_ms"], p["bend_tau_hi_ms"]) / 1000.0
        span = min(n_hops - k0, int(5 * tau / hop_s) + 1)
        t = np.arange(span) * hop_s
        out[k0:k0 + span] += sgn * mag * np.exp(-t / tau)
    return out


def texture_cents(x, n_hops):
    """The singer's real f0 to a micro-tuning texture: the deviation from the
    nearest semitone in cents, naturally bounded at plus or minus 50.
    The synthesis chosen after the diagnostic ladder of 2026-07-28 (B slightly
    auto-tuned, C heavier): the stability of the target skeleton with their human
    quality, without the instability of shift. That instability comes from the
    transitions, so the texture is zeroed on hops where they change note quickly,
    above 35 cents per hop, and only the micro-deviation, drift and real vibrato
    of sustained notes are printed over."""
    import pyworld
    x = np.ascontiguousarray(x)
    f0, _ = pyworld.harvest(x, SR, frame_period=1000.0 * HOP / SR)
    midi = 69 + 12 * np.log2(np.maximum(f0, 1) / 440)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    ok = (f0 > 0) & (dm < 0.35)
    res = np.where(ok, dev, 0.0)
    res = np.convolve(res, np.ones(3) / 3, mode="same")  # 3-hop smoothing of the extracted jitter
    out = np.zeros(n_hops)
    n = min(n_hops, len(res))
    out[:n] = res[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--tag", default="direct30k")
    ap.add_argument("--model", default=f"{DDSP}/exp/combsub-harry/model_30000.pt")
    ap.add_argument("--key", default="0")
    ap.add_argument("--pe", default="parselmouth")
    ap.add_argument("--f0-min", default="65")
    ap.add_argument("--f0-max", default="800")
    ap.add_argument("--porta-ms", type=float, default=33.0,
                    help="portamento time constant in ms for steps (0 lands every note directly)")
    ap.add_argument("--leap-snap", type=int, default=4,
                    help="note changes larger than this many semitones land directly (0 glides everything)")
    ap.add_argument("--vib-hz", type=float, default=VIB_HZ)
    ap.add_argument("--vib-phase", type=float, default=0.0,
                    help="vibrato phase offset in cycles, 0 to 1")
    ap.add_argument("--vib-onset-ms", type=float, default=0.0,
                    help="vibrato onset delay: each note is straight before the vibrato arrives (0 = off)")
    ap.add_argument("--f0-mode", choices=["target", "shift", "texture", "express"],
                    default="target",
                    help="shift transposes the singer's real f0 tick by tick (the notes json must contain heard); "
                         "texture is the target skeleton with their micro-tuning texture printed over it; "
                         "express is the target skeleton plus the rule-based expression layer, whose parameters copy the human "
                         "statistics, needs no f0 from them, and can be ported to live")
    ap.add_argument("--texture-gain", type=float, default=1.0)
    ap.add_argument("--expr-gain", type=float, default=1.0,
                    help="overall multiplier of the express expression layer (0 returns to the bare skeleton)")
    ap.add_argument("--expr-seed", type=int, default=20260731,
                    help="RNG seed of the express expression layer; fixed makes a render reproducible")
    ap.add_argument("--respect-unvoiced", dest="respect_unvoiced",
                    action="store_true", default=True,
                    help="inject f0 = 0 on the singer's unvoiced frames, breaths and fricatives, so only DDSP's noise branch runs (on by default)")
    ap.add_argument("--no-respect-unvoiced", dest="respect_unvoiced",
                    action="store_false", help="turn the voicing gate off, the behaviour before 2026-07-30, for comparison")
    ap.add_argument("--uv-min-ms", type=float, default=40.0,
                    help="unvoiced gaps shorter than this are not gated, which removes the grain")
    ap.add_argument("--uv-guard-ms", type=float, default=0.0,
                    help="inset at both ends of a gated span (0 leaves the boundary to the hysteresis)")
    ap.add_argument("--uv-hi", type=float, default=0.60,
                    help="upper hysteresis threshold of the voicing confidence, entering voiced")
    ap.add_argument("--uv-lo", type=float, default=0.35,
                    help="lower hysteresis threshold of the voicing confidence, entering unvoiced")
    ap.add_argument("--uv-anchor-ms", type=float, default=25.0,
                    help="the gate may only fall on frames harvest calls unvoiced, and within this neighbourhood of them "
                         "(-1 disables anchoring, letting the joint vote open holes inside a phrase)")
    ap.add_argument("--uv-f0", type=float, default=1200.0,
                    help="the sentinel f0 injected on unvoiced frames, purely to prevent bursts; the output is covered by the pass-through anyway. "
                         "0 hard-zeroes it, which occasionally produces a full-scale burst")
    ap.add_argument("--uv-passthrough", dest="uv_passthrough",
                    action="store_true", default=True,
                    help="pass the singer's raw audio through over unvoiced spans instead of synthesising them (on by default)")
    ap.add_argument("--no-uv-passthrough", dest="uv_passthrough",
                    action="store_false",
                    help="turn the pass-through off, so unvoiced spans are still synthesised by DDSP, for comparison")
    ap.add_argument("--uv-xfade-ms", type=float, default=15.0,
                    help="length of the raised-cosine crossfade at the pass-through boundary")
    ap.add_argument("--input-agc", dest="input_agc", action="store_true",
                    default=True,
                    help="raise the input level to the training reference before conversion and restore it afterwards (on by default)")
    ap.add_argument("--no-input-agc", dest="input_agc", action="store_false",
                    help="turn the input AGC off, so the model takes their raw level directly, for comparison")
    ap.add_argument("--agc-ref", type=float, default=TRAIN_REF_RMS,
                    help="target rms of the AGC; the default is the measured reference level of the training clips")
    ap.add_argument("--run", action="store_true", help="run DDSP directly")
    a = ap.parse_args()

    # 1) The take to mono at 44.1 kHz for DDSP to read. The md5 has to match, so
    #    a real file is written out.
    x, sr = sf.read(a.take, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    in_path = os.path.join(HERE, "out", f"{a.tag}_in.wav")
    # The AGC applies only to the copy fed to DDSP. f0 extraction, the voicing
    # decision and the pass-through all use the original x, and the output is
    # multiplied by 1/agc_g, so the level structure is untouched.
    if a.input_agc:
        x_in, agc_g = input_agc(x, a.agc_ref)
        print(f"input AGC: gain {agc_g.min():.2f}–{agc_g.max():.2f}× "
              f"(ref rms {a.agc_ref:.4f}, peak {np.abs(x_in).max():.3f})")
    else:
        x_in, agc_g = x, None
    sf.write(in_path, x_in, SR, subtype="PCM_16")

    # 2) The target f0 written into DDSP's f0 cache; the file-name convention is
    #    in main.py:178.
    # The cache file-name convention (main.py:165-178): hop_size is the result of
    # a float computation, so it appears as "512.0", and the md5 is taken over the
    # input file's bytes. Get the name wrong and main.py silently extracts their
    # own pitch instead, then saves it under the correct name and poisons later
    # runs. That happened on 2026-07-26.
    notes, tick, heard = load_notes(a.notes)
    md5 = hashlib.md5(open(in_path, "rb").read()).hexdigest()
    hop_name = float(HOP)
    cache = os.path.join(DDSP, "cache",
                         f"{a.pe}_{hop_name}_{a.f0_min}_{a.f0_max}_{md5}.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    n_hops = int(np.load(cache).shape[0]) if os.path.exists(cache) \
        else len(x) // HOP + 1
    if a.f0_mode == "shift":
        if heard is None:
            sys.exit("--f0-mode shift needs heard in the notes json, which only newer render_v3 versions write")
        f0 = shift_f0(notes, heard, tick, n_hops, x, float(a.f0_min))
    elif a.f0_mode == "texture":
        skel = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                         vib_semi=0.0)   # the real vibrato is in the texture, so the artificial one is off
        f0 = skel * 2 ** (a.texture_gain * texture_cents(x, n_hops) / 1200.0)
    elif a.f0_mode == "express":
        skel = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                         vib_semi=0.0)   # the vibrato comes from the expression layer, in its irregular form
        cents = expressive_cents(notes, tick, n_hops, a.leap_snap, a.expr_seed)
        f0 = skel * 2 ** (a.expr_gain * cents / 1200.0)
        print(f"express layer: cents sd {cents.std():.1f} "
              f"(gain {a.expr_gain}, seed {a.expr_seed})")
    else:
        f0 = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                       a.vib_hz, a.vib_phase, a.vib_onset_ms)
    vm = None
    if a.respect_unvoiced:
        vm = voicing_mask(x, n_hops, a.uv_min_ms, a.uv_guard_ms,
                          a.uv_hi, a.uv_lo, a.uv_anchor_ms)
        f0 = f0 * vm
        print(f"voicing gate: {100 * (1 - vm.mean()):.1f}% hops uv "
              f"(hi/lo {a.uv_hi}/{a.uv_lo}, min_uv {a.uv_min_ms:.0f}ms, "
              f"guard {a.uv_guard_ms:.0f}ms)")
    # Both native unvoiced zeros, from extract_f0_uv, and the gate's zeros are
    # replaced by the sentinel: any f0 = 0 drives combtooth's sinc argument
    # towards 0/0 and occasionally produces a full-scale burst (see
    # voicing_mask).
    n_uv = int((f0 <= 0).sum())
    f0 = np.where(f0 > 0, f0, a.uv_f0)
    print(f"uv sentinel: {n_uv} hops ({100.0 * n_uv / n_hops:.1f}%) → {a.uv_f0:.0f}Hz")
    np.save(cache, f0.astype(np.float32), allow_pickle=False)
    print(f"f0 cache written: {os.path.basename(cache)}  "
          f"hops={n_hops} range={f0[f0>0].min():.0f}-{f0.max():.0f}Hz")

    out_path = os.path.join(HERE, "out", f"{a.tag}_angel.wav")
    cmd = [f"{DDSP}/venv/bin/python", f"{DDSP}/main.py",
           "-m", a.model, "-i", in_path, "-o", out_path,
           "-k", a.key, "-pe", a.pe, "-e", "false", "-d", "mps",
           "-fmin", a.f0_min, "-fmax", a.f0_max]
    print("\n" + " ".join(cmd) + "\n")
    if a.run:
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        r = subprocess.run(cmd, cwd=DDSP, env=env)
        if r.returncode:
            sys.exit(r.returncode)
        ang, _ = sf.read(out_path, dtype="float64")
        if ang.ndim > 1:
            ang = ang[:, 0]
        if agc_g is not None:                    # undo the AGC, restoring the level structure
            m = min(len(ang), len(agc_g))
            ang = ang[:m] / agc_g[:m]
        if a.uv_passthrough and vm is not None:
            ang, s = uv_passthrough(ang, x, vm, a.uv_xfade_ms)
            sf.write(out_path, ang, SR)          # only after the pass-through is this the voice's final output
            print(f"uv passthrough: raw audio restored over {100 * (1 - vm.mean()):.1f}% of hops "
                  f"(xfade {a.uv_xfade_ms:.0f}ms, take gain {s:.3f})")
        n = min(len(x), len(ang))
        mix = 0.5 * x[:n] + 0.8 * ang[:n]
        mix = mix / (np.max(np.abs(mix)) + 1e-12) * 0.9
        sf.write(os.path.join(HERE, "out", f"{a.tag}_mix.wav"), mix, SR)
        print(f"wrote out/{a.tag}_angel.wav / out/{a.tag}_mix.wav")


if __name__ == "__main__":
    main()
