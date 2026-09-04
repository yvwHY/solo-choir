"""loopback_lab.py - a three-cell self-loop ladder: make the pipeline sing the singer's own melody back.

Motivation: after retraining on 2.7 times the data, blind listening judged the
result "much the same" and the auto-tuned quality had not moved, which ACQUITS the
voice, the DDSP converter. Suspicion moved to the seam - the layer where the
model's note line is rendered into an f0 curve, with its flat quantised target,
rule-based portamento and artificial vibrato.

The ladder removes the variable "which note the model chose" entirely: all three
cells sing HIS OWN melody.
  cell 1 = the original take, untouched
  cell 2 = the direct-drive voice with his own real f0 injected verbatim (shift
           mode, offset always 0) - conversion alone
  cell 3 = the direct-drive voice with the same melody quantised to notes and
           drawn into f0 by the portamento and vibrato rules - seam plus voice
Same passage, same checkpoint, same level. Reading it: if 2 sounds like 1 and 3
is auto-tuned, the seam is convicted; if 2 is already auto-tuned, it is complicit.

Usage (vcclient-dev environment, from harmony/):
  python loopback_lab.py prep   # cut the passage, write seg_notes.json (notes = heard = his line)
  # ...run direct_mouth.py twice in between; prep prints the commands...
  python loopback_lab.py pack   # level-match, then stim_1..3.wav + key.json + statistics
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "loopback_0730")
TAKE = os.path.join(HERE, "..", "..", "..", "260722_harmony_brain", "data", "take.wav")
NOTES = os.path.join(HERE, "out", "angel_v2_world4keyed_notes.json")
SR, HOP = 44100, 512
T0, T1 = 131, 270          # tick range: starts on an onset after a rest, ends at the next rest (26.1 s)
DDSP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
        "260724_ddsp_svc")
CKPT = f"{DDSP}/exp/combsub-harry/model_30000.pt"


def prep():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(NOTES))
    heard, ss = d["heard"], int(d["step_samps"])
    a, b = T0 * ss, T1 * ss
    x, sr = sf.read(TAKE, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == SR, sr
    seg = np.ascontiguousarray(x[a:b])
    sf.write(os.path.join(OUT, "seg_take.wav"), seg, SR, subtype="PCM_16")
    # notes = heard = his own lead notes, through the pipeline's existing
    # transcription. Shift mode holds the offset at 0, and target mode quantises
    # and renders the same line, so both cells have identical melodic content.
    line = heard[T0:T1]
    json.dump({"bpm": d["bpm"], "sr": SR, "step_samps": ss,
               "notes": line, "heard": line},
              open(os.path.join(OUT, "seg_notes.json"), "w"), indent=1)
    v = [m for m in line if m is not None]
    print(f"seg: ticks {T0}-{T1}  {a/SR:.2f}-{b/SR:.2f}s  {len(seg)/SR:.2f}s  "
          f"voiced {len(v)}/{len(line)} ticks  midi {min(v)}-{max(v)}")
    for tag, mode in (("lb2_ownf0", "shift"), ("lb3_rulef0", "target")):
        print(f"\npython direct_mouth.py --notes out/loopback_0730/seg_notes.json "
              f"--take out/loopback_0730/seg_take.wav "
              f"--tag loopback_0730/{tag} --model {CKPT} "
              f"--f0-mode {mode} --run")


def _load(p):
    y, sr = sf.read(p, dtype="float64")
    if y.ndim > 1:
        y = y[:, 0]
    assert sr == SR, (p, sr)
    return np.ascontiguousarray(y)


def _stats(y):
    """Intonation statistics over voiced hops: deviation from the nearest semitone in cents, and vibrato depth."""
    import pyworld
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    v = f0 > 0
    midi = 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    sus = v & (dm < 0.35)                      # sustained stretches, excluding transitions between notes
    # vibrato depth: how far the sustained cents swing about a 5-hop moving average, with the slow drift removed
    ma = np.convolve(dev, np.ones(5) / 5, mode="same")
    rip = (dev - ma)[sus]
    return dict(voiced=float(v.mean()),
                cents_sd=float(np.std(dev[sus])) if sus.any() else 0.0,
                vib_cents=float(np.std(rip) * np.sqrt(2)) if sus.any() else 0.0,
                f0_med=float(np.median(f0[v])) if v.any() else 0.0)


def pack():
    cells = [("1_take_raw", os.path.join(OUT, "seg_take.wav"),
              "the original: sung by him, with no conversion at all"),
             ("2_ownf0", os.path.join(OUT, "lb2_ownf0_angel.wav"),
              "the direct-drive voice with his real f0 injected verbatim (shift mode, offset always 0) - conversion alone"),
             ("3_rulef0", os.path.join(OUT, "lb3_rulef0_angel.wav"),
              "the direct-drive voice with the same melody quantised to notes and rendered into f0 by "
              "33 ms portamento, leap-snap 4 and 5 Hz 0.12 st vibrato - seam plus voice")]
    ys = [_load(p) for _, p, _ in cells]
    n = min(len(y) for y in ys)
    ys = [y[:n] for y in ys]
    rms = [float(np.sqrt((y ** 2).mean())) for y in ys]
    tgt = float(np.median(rms))
    ys = [y * (tgt / r) for y, r in zip(ys, rms)]
    g = 0.9 / max(float(np.abs(y).max()) for y in ys)   # one gain for all of them, so the relative levels survive
    key = {}
    for i, ((name, src, desc), y) in enumerate(zip(cells, ys), 1):
        y = y * g
        sf.write(os.path.join(OUT, f"stim_{i}.wav"), y, SR, subtype="PCM_16")
        st = _stats(y)
        key[f"stim_{i}"] = dict(cell=name, source=os.path.basename(src), desc=desc,
                                dur_s=round(n / SR, 3),
                                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                                peak=round(float(np.abs(y).max()), 3), **
                                {k: round(v, 3) for k, v in st.items()})
        print(f"stim_{i} {name:11s} rms {key[f'stim_{i}']['rms']:.5f} "
              f"cents_sd {st['cents_sd']:6.2f}  vib {st['vib_cents']:5.2f}c  "
              f"voiced {st['voiced']:.2f}  f0med {st['f0_med']:.1f}Hz")
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote 3 stimuli + key.json to {OUT}/")


def _take_uv():
    """Mask of the take's unvoiced frames (breaths, fricatives), the reference truth for scoring."""
    import pyworld
    y = _load(os.path.join(OUT, "seg_take.wav"))
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    return f0 > 0


def _cpp(y, hops, n=2048):
    """Cepstral peak prominence of the given frames, that is, the strength of the harmonic column; lower is more noise-like.

    Spectral flatness was tried first, but it is contaminated by spectral tilt (the
    learned envelope of the noise branch after the gate is itself quite coloured, so
    flatness FALLS), which gives the opposite conclusion. CPP looks only at the
    cepstral peak caused by periodicity and is immune to envelope tilt, which makes
    it the direct answer to "is there a harmonic column". Truly silent frames are
    excluded as meaningless.
    """
    q0, q1 = int(SR / 800), int(SR / 65)
    w = np.hanning(n)
    idx = np.arange(q0, q1)
    A = np.vstack([idx, np.ones_like(idx)]).T
    vals = []
    for k in hops:
        s = k * HOP - n // 2
        if s < 0 or s + n > len(y):
            continue
        seg = y[s:s + n] * w
        if np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        lp = np.log(np.abs(np.fft.rfft(seg)) ** 2 + 1e-12)
        lc = np.log(np.abs(np.fft.irfft(lp, n=n))[q0:q1] + 1e-12)
        b = np.linalg.lstsq(A, lc, rcond=None)[0]
        vals.append(float(np.max(lc - A @ b)))
    return float(np.mean(vals)) if vals else 0.0


def _score(y, ref_v):
    """Three numbers for the gate: overall voiced percentage, the share of breaths sung out loud, and intonation regression over voiced frames."""
    import pyworld
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    v = f0 > 0
    n = min(len(v), len(ref_v))
    v, r = v[:n], ref_v[:n]
    midi = 69 + 12 * np.log2(np.maximum(f0[:n], 1) / 440.0)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    sus = v & r & (dm < 0.35)                       # only sustained stretches where the take is voiced too
    return dict(voiced=round(float(v.mean()), 3),
                breath_voiced=round(float(v[~r].mean()), 3),
                cents_sd=round(float(np.std(dev[sus])), 2) if sus.any() else 0.0,
                uv_cpp=round(_cpp(y, np.flatnonzero(~r)), 3),
                v_cpp=round(_cpp(y, np.flatnonzero(r)), 3))


def packg():
    """Before and after the gate: reuse stim_1's RMS as the common level; stim_1..3 are untouched."""
    ref_v = _take_uv()
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2g_ownf0_gated", "lb2g_ownf0_angel.wav", "stim_2g.wav"),
            ("3g_rulef0_gated", "lb3g_rulef0_angel.wav", "stim_3g.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':17s} {'voiced%':>8s} {'breath_v%':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        print(f"{name:17s} {st['voiced']*100:8.1f} {st['breath_voiced']*100:10.1f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="the same as cell " + name[0] + ", the only difference being "
                     "--respect-unvoiced, which injects f0=0 on his unvoiced frames",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2g.wav / stim_3g.wav + key.json to {OUT}/")


def packh():
    """Native unvoiced plus consensus plus hysteresis (2h/3h): writes 2h and 3h only, overwriting no existing stimulus.

    Two extra diagnostic columns:
      interior_v%   the share of breath frames at least 2 hops from the nearest
                    take-voiced frame that get sung. Excluding the boundary matters:
                    harvest's analysis window on the output is over 23 ms and smears
                    adjacent voiced frames in anyway, and that part is measurement
                    bleed rather than breath being sung
      cut_on_v%     the share of take-voiced frames the gate cuts by mistake, the
                    upper bound on the cost of calling a doubtful frame unvoiced
    """
    import direct_mouth as dm
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    vm = dm.voicing_mask(take, len(ref_v)) > 0
    print(f"gate: {100 * (1 - vm.mean()):.1f}% hops uv | "
          f"cut_on_voiced {100 * (ref_v & ~vm).sum() / ref_v.sum():.1f}%")
    vidx = np.flatnonzero(ref_v)
    interior = np.array([np.min(np.abs(vidx - k)) >= 2 if len(vidx) else True
                         for k in range(len(ref_v))]) & ~ref_v
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2g_gated", "stim_2g.wav", None),
            ("3g_gated", "stim_3g.wav", None),
            ("2h_native_uv", "lb2h_ownf0_angel.wav", "stim_2h.wav"),
            ("3h_native_uv", "lb3h_rulef0_angel.wav", "stim_3h.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':15s} {'voiced%':>8s} {'breath_v%':>10s} {'interior_v%':>12s} "
          f"{'cents_sd':>9s} {'uv_CPP':>7s} {'v_CPP':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        import pyworld
        f0o, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
        m = min(len(f0o), len(interior))
        iv = float((f0o[:m] > 0)[interior[:m]].mean())
        print(f"{name:15s} {st['voiced']*100:8.1f} {st['breath_voiced']*100:10.1f} "
              f"{iv*100:12.1f} {st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} "
              f"{st['v_cpp']:6.3f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="native unvoiced injection (extraction gives (f0, uv) as a pair), "
                     "multi-feature consensus on uv, and confidence hysteresis at the boundary",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                interior_voiced=round(iv, 3), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2h.wav / stim_3h.wav + key.json to {OUT}/")


def _lsd(y, ref, hops, n=2048):
    """Log-spectral distance from the original over the breath stretches, in dB; a pass-through should approach 0."""
    w = np.hanning(n)
    vals = []
    for k in hops:
        s = k * HOP - n // 2
        if s < 0 or s + n > min(len(y), len(ref)):
            continue
        a = 20 * np.log10(np.abs(np.fft.rfft(y[s:s + n] * w)) + 1e-9)
        b = 20 * np.log10(np.abs(np.fft.rfft(ref[s:s + n] * w)) + 1e-9)
        vals.append(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.mean(vals)) if vals else 0.0


def packp():
    """Source pass-through (2p/3p): unvoiced stretches take the original. Writes 2p and 3p only."""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2h_sentinel", "stim_2h.wav", None),
            ("3h_sentinel", "stim_3h.wav", None),
            ("2p_passthru", "lb2p_ownf0_angel.wav", "stim_2p.wav"),
            ("3p_passthru", "lb3p_rulef0_angel.wav", "stim_3p.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':14s} {'breath_v%':>10s} {'breathLSD_dB':>13s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        # log-spectral distance has to be compared at matched level: scale the original to this cell's voiced RMS
        m = min(len(y), len(take))
        vm_s = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vm_s] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vm_s] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:14s} {st['breath_voiced']*100:10.1f} {lsd:13.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="unvoiced stretches pass his original through (15 ms raised-cosine "
                     "crossfade, level matched to the converted output); voiced stretches are converted",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2p.wav / stim_3p.wav + key.json to {OUT}/")


def packq():
    """Anchored (2q/3q): fixes the gate opening a hole in the middle of a phrase, the "cannot breathe after 12 seconds" problem. Writes 2q and 3q only."""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2p_passthru", "stim_2p.wav", None),
            ("3p_passthru", "stim_3p.wav", None),
            ("2q_anchored", "lb2q_ownf0_angel.wav", "stim_2q.wav"),
            ("3q_anchored", "lb3q_rulef0_angel.wav", "stim_3q.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="source pass-through, the gate anchored within 25 ms of a silence "
                     "harvest agrees on so it never opens a hole mid-phrase, per-window "
                     "pass-through gain, and a locally adaptive consensus threshold",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2q.wav / stim_3q.wav + key.json to {OUT}/")


def packr():
    """Input AGC (2r/3r): pull the level to the training reference before conversion and restore it after. Writes 2r and 3r only."""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2q_anchored", "stim_2q.wav", None),
            ("3q_anchored", "stim_3q.wav", None),
            ("2r_agc", "lb2r_ownf0_angel.wav", "stim_2r.wav"),
            ("3r_agc", "lb3r_rulef0_angel.wav", "stim_3r.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="the q settings plus an input AGC to the training reference RMS of "
                     "0.0566 before conversion, multiplied back by 1/gain afterwards so the "
                     "level structure is restored",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2r.wav / stim_3r.wav + key.json to {OUT}/")


def packe():
    """Expression layer (3e): the r settings plus f0-mode express, a rule-based
    per-note offset, drift, irregular vibrato and scoop, with parameters taken from
    measurements of real singing. Writes 3e only - cell 2 needs no expression layer,
    since its f0 is already a real singer's."""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2r_agc", "stim_2r.wav", None),
            ("3r_agc", "stim_3r.wav", None),
            ("3e_express", "lb3e_exprf0_angel.wav", "stim_3e.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="the r settings plus f0-mode express: a target skeleton with a "
                     "rule-based expression layer (per-note offset, slow drift, fast tremor, "
                     "irregular vibrato and a scoop, with parameters taken from measurements "
                     "of real singing, seed 20260731)",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_3e.wav + key.json to {OUT}/")


if __name__ == "__main__":
    {"prep": prep, "pack": pack, "packg": packg, "packh": packh,
     "packp": packp, "packq": packq, "packr": packr, "packe": packe}[
        sys.argv[1] if len(sys.argv) > 1 else "prep"]()
