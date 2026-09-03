"""Streaming target-f0 WORLD mouth -- strict-causal chunk simulation (file-1a).

The 150-250 ms lookahead reframe (worklog 07-24 §B) makes a slow angel legal.
This simulates that mouth honestly: audio arrives in HOP-sized chunks and a
frame may only be used once GUARD of audio exists after it, and synthesized
audio is only emitted once it is HOLD clear of WORLD's end-boundary effect
(measured: the last ~28 ms of a synthesis differ vs a longer one; before
that, sample-exact). Structural lag = HOP + GUARD + HOLD = 240 ms + compute.

G28 killed naive streaming WORLD for three root causes; each is countered:
  * block-constant YIN f0 -> per-frame dio+stonemask over a 1.2 s window
    (dio 7 ms/s vs harvest 103 ms/s; cheaptrick gets true per-frame f0)
  * tiny context           -> f0 context 1.2 s; sp/ap on a 0.4 s tail window
  * frozen/repeated frames -> frames are analyzed once, appended, never reused
Seams: WORLD synthesis is deterministic, so re-synthesizing the whole phrase
each hop reproduces the already-emitted prefix sample-exactly (checked at
runtime, reported) -- zero seams by construction. Per-hop cost grows within a
phrase (offline fine; live would use WORLD's C++ incremental synthesizer), so
a bounded live variant (resynth last 1 s, 30 ms equal-power splice) is
rendered too. If bounded ~= exact by ear, the live path is proven.

Pitch is placed by TARGET (F16 rule), same recipe as world_mouth target mode,
notes replayed from world_mouth's {tag}_notes.json (same brain line).

Run (retraining venv):
  venv/bin/python world_stream.py <take.wav> out/angel_v2_world2_notes.json [tag]
"""

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pyworld

HERE = Path(__file__).parent
FRAME_MS = 5.0
VIB_HZ, VIB_SEMI = 5.0, 0.12
PORTA = 0.15
HOP_FRAMES = 30      # 150 ms per chunk
GUARD_FRAMES = 10    # 50 ms settle before a frame may be used
HOLD_FRAMES = 8      # 40 ms synthesis tail held back (end-boundary ~28 ms)
CTX_FRAMES = 240     # 1.2 s f0 window
TAIL_FRAMES = 80     # 0.4 s sp/ap window
REST_CLOSE = 40      # 200 ms of rest closes a phrase (covers the gate tail)
BOUND_FRAMES = 200   # bounded mode: resynth last 1.0 s only


def synth(f0, sp, ap, sr):
    return pyworld.synthesize(np.ascontiguousarray(f0),
                              np.ascontiguousarray(sp),
                              np.ascontiguousarray(ap), sr, frame_period=FRAME_MS)


def main():
    take_path, notes_path = Path(sys.argv[1]), Path(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "angel_v2_wstream"

    meta = json.load(open(notes_path))
    notes, step_samps, sr = meta["notes"], meta["step_samps"], meta["sr"]
    xf = int(0.030 * sr)
    hold_samps = int(round(HOLD_FRAMES * sr * FRAME_MS / 1000))

    with wave.open(str(take_path)) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    frame_hop = sr * FRAME_MS / 1000.0  # 220.5 at 44.1k; grid kept in frame units
    n_frames = int(len(mic) / frame_hop) + 1

    # per-frame TARGET f0 on the global grid -- identical recipe to
    # world_mouth target mode, and causal (one-pole porta, absolute-time vib)
    f0_target = np.zeros(n_frames)
    cur = 0.0
    for k in range(n_frames):
        tk = k * FRAME_MS / 1000.0
        tick = min(int(tk * sr / step_samps), len(notes) - 1)
        note = notes[tick]
        target = 0.0 if note is None else 440 * 2 ** ((note - 69) / 12)
        if target > 0:
            cur = target if cur == 0 else cur + (target - cur) * PORTA
            f0_target[k] = cur * 2 ** (VIB_SEMI * np.sin(2 * np.pi * VIB_HZ * tk) / 12)
        else:
            cur = 0.0

    # causal singer gate on the same grid (40 ms attack / 150 ms release)
    a_up, a_dn = 1 - np.exp(-FRAME_MS / 40.0), 1 - np.exp(-FRAME_MS / 150.0)
    g, env = 0.0, np.empty(n_frames)
    tgt = (f0_target > 0).astype(float)
    for k in range(n_frames):
        g += (tgt[k] - g) * (a_up if tgt[k] > g else a_dn)
        env[k] = g

    def gated(wav_, lo_f, hi_f):
        gate = np.interp(np.arange(len(wav_)) / sr,
                         (np.arange(hi_f - lo_f)) * FRAME_MS / 1000.0,
                         env[lo_f:hi_f])
        return wav_ * gate

    # ---- the stream loop ----
    fft_size = pyworld.get_cheaptrick_fft_size(sr)
    sp_dim = fft_size // 2 + 1
    out = {"exact": np.zeros(len(mic) + sr), "bounded": np.zeros(len(mic) + sr)}
    borrow_sp = np.zeros((n_frames, sp_dim))
    borrow_ap = np.ones((n_frames, sp_dim))
    have_good = False
    last_sp = None
    last_ap = None
    accepted = 0
    phrase = None            # dict(start, f0, rest, n{mode: emitted samples})
    prefix_diff = 0.0
    t_an, t_syn_b, t_syn_x = [], [], []

    hop_samps = int(round(HOP_FRAMES * frame_hop))
    for t_end in range(hop_samps, len(mic) + hop_samps, hop_samps):
        t_end = min(t_end, len(mic))
        closing_take = t_end == len(mic)
        w0 = max(0, t_end - int(round(CTX_FRAMES * frame_hop)))
        win = mic[w0:t_end]
        base_f = int(round(w0 / frame_hop))
        t0 = time.perf_counter()
        f0w, tw = pyworld.dio(win, sr, frame_period=FRAME_MS)
        f0w = pyworld.stonemask(win, f0w, tw, sr)
        # sp/ap only on the tail the new frames live in (analysis cost bound)
        tl0 = max(0, len(win) - int(round(TAIL_FRAMES * frame_hop)))
        tail = win[tl0:]
        tf0_lo = int(round(tl0 / frame_hop))
        f0t = f0w[tf0_lo : tf0_lo + int(len(tail) / frame_hop) + 1]
        tt = np.arange(len(f0t)) * FRAME_MS / 1000.0
        spt = pyworld.cheaptrick(tail, f0t, tt, sr)
        apt = pyworld.d4c(tail, f0t, tt, sr)
        tail_base_f = base_f + tf0_lo
        t_an.append(time.perf_counter() - t0)

        # accept frames older than the guard, never re-visit older ones
        lim_f = n_frames if closing_take else int(
            (t_end - GUARD_FRAMES * frame_hop) / frame_hop)
        midi_w = np.where(f0w > 0, 69 + 12 * np.log2(np.maximum(f0w, 1) / 440), np.nan)
        new_lo = accepted
        for gf in range(accepted, min(lim_f, n_frames, tail_base_f + len(f0t))):
            lt = gf - tail_base_f
            lf = gf - base_f
            if lt < 0:
                continue
            voiced = f0w[lf] > 0
            apm = apt[lt].mean()
            win_m = midi_w[max(0, lf - 4) : lf + 5]
            med = np.nan if np.isnan(win_m).all() else np.nanmedian(win_m)
            oct_err = voiced and not np.isnan(med) and abs(midi_w[lf] - med) > 6
            if voiced and apm < 0.8 and not oct_err:
                last_sp, last_ap, have_good = spt[lt], apt[lt], True
            if have_good:
                borrow_sp[gf], borrow_ap[gf] = last_sp, last_ap
            accepted = gf + 1

        if accepted == new_lo:
            continue

        # feed accepted frames into the phrase machine
        for gf in range(new_lo, accepted):
            resting = f0_target[gf] == 0
            if phrase is None:
                if resting or not have_good:
                    continue
                phrase = {"start": gf, "f0": [], "rest": 0,
                          "n": {"exact": 0, "bounded": 0}}
            phrase["f0"].append(f0_target[gf])
            phrase["rest"] = phrase["rest"] + 1 if resting else 0

        if phrase is None:
            continue
        closing = phrase["rest"] >= REST_CLOSE or closing_take
        hold = 0 if closing else hold_samps
        pf = len(phrase["f0"])
        s, e = phrase["start"], phrase["start"] + pf
        f0p = np.array(phrase["f0"])
        s0 = int(round(s * frame_hop))

        # exact: whole-phrase resynth; deterministic prefix = no seams
        t0 = time.perf_counter()
        wav = gated(synth(f0p, borrow_sp[s:e], borrow_ap[s:e], sr), s, e)
        t_syn_x.append(time.perf_counter() - t0)
        n_prev = phrase["n"]["exact"]
        n_emit = max(n_prev, len(wav) - hold)
        if n_prev:
            prefix_diff = max(prefix_diff, float(np.abs(
                out["exact"][s0 : s0 + n_prev] - wav[:n_prev]).max()))
        out["exact"][s0 + n_prev : s0 + n_emit] = wav[n_prev:n_emit]
        phrase["n"]["exact"] = n_emit

        # bounded: resynth only the last BOUND_FRAMES, equal-power splice
        t0 = time.perf_counter()
        b = max(0, pf - BOUND_FRAMES)
        wav_b = gated(synth(f0p[b:], borrow_sp[s + b : e], borrow_ap[s + b : e], sr),
                      s + b, e)
        s0b = int(round((s + b) * frame_hop))
        n_prev_b = phrase["n"]["bounded"]
        rel_prev = n_prev_b - (s0b - s0)      # already-emitted samples inside bound
        n_emit_b = max(rel_prev, len(wav_b) - hold)
        if rel_prev > 0:
            fl = min(xf, rel_prev, len(wav_b))
            fade = np.sin(0.5 * np.pi * np.linspace(0, 1, fl)) ** 2
            lo = rel_prev - fl
            out["bounded"][s0b + lo : s0b + rel_prev] = (
                out["bounded"][s0b + lo : s0b + rel_prev] * (1 - fade)
                + wav_b[lo:rel_prev] * fade)
        out["bounded"][s0b + max(rel_prev, 0) : s0b + n_emit_b] = \
            wav_b[max(rel_prev, 0):n_emit_b]
        phrase["n"]["bounded"] = (s0b - s0) + n_emit_b
        t_syn_b.append(time.perf_counter() - t0)
        if closing:
            phrase = None

    # ---- report + write ----
    an = np.array(t_an) * 1000
    sb = np.array(t_syn_b) * 1000 if t_syn_b else np.zeros(1)
    sx = np.array(t_syn_x) * 1000 if t_syn_x else np.zeros(1)
    hop_ms = HOP_FRAMES * FRAME_MS
    live_ms = an[: len(sb)] + sb if len(sb) <= len(an) else an + sb[: len(an)]
    print(f"hops {len(an)} | analysis ms p50 {np.percentile(an,50):.0f} "
          f"p95 {np.percentile(an,95):.0f} | bounded synth p50 {np.percentile(sb,50):.0f} "
          f"p95 {np.percentile(sb,95):.0f} | exact synth p95 {np.percentile(sx,95):.0f} (offline only)")
    print(f"LIVE path per-hop (analysis+bounded) p50 {np.percentile(live_ms,50):.0f} "
          f"p95 {np.percentile(live_ms,95):.0f} ms vs hop {hop_ms:.0f} ms; "
          f"latency = {hop_ms + (GUARD_FRAMES+HOLD_FRAMES)*FRAME_MS:.0f} + compute")
    print(f"exact-mode prefix re-emit max diff {prefix_diff:.2e} (0 = zero seams)")
    for mode in ("exact", "bounded"):
        angel = out[mode][: len(mic)]
        for name, sig in [(f"{tag}_{mode}_angel", angel),
                          (f"{tag}_{mode}_mix", 0.5 * mic + 0.8 * angel)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(str(HERE / "out" / f"{name}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
        print(f"wrote out/{tag}_{mode}_angel.wav, out/{tag}_{mode}_mix.wav")


if __name__ == "__main__":
    main()
