"""bare_test.py - the auto-tune case re-tried on clean material, across the whole ladder (v2).

**The material was withdrawn.** respond2's session dump is "what he was heard
saying plus what was played out, on one track" (respond2.py:850): his solo phrases
and the responses - his own playback plus both parts - interleave on a single mono
track. The old `[0::2]` selection had its odd/even alignment thrown off by a
pre-filter. Cross-correlation settles it, since a response necessarily contains a
digital replay of the phrase before it and so correlates highly: the segments
chosen by prosody_ab, vs_male3_trial and the first bare test (at 103.7 s and
123.7 s) were ALL responses, so a three-part mix was fed to harvest and HuBERT as
if it were his phrase, which is where "unsteady, a mess" came from. Every verdict
and measurement from cells A to E, the first bare test and the first reference run
is void.

This v2 re-runs the whole ladder on real solo phrases, into `out/clean_ab/`:
  seg{i}_dry.wav                 his solo phrase (confirmed first: one voice, clean)
  seg{i}_A_mix / _A_bass1_lower  the current recipe as a baseline (quantised
                                 skeleton, the original defendant)
  seg{i}_bare_harry_ident        his harvest f0 at x1 straight into his own model
  seg{i}_bare_bass1_oct / _girl_up / _bare_mix   an octave down and up, and the bare mix
  (the reference cells come from running main.py directly, via the shell: seg{i}_canon_*.wav)

Reading the ladder: dry, then reference (the ceiling), then bare, then A (the full
recipe). Whichever rung the auto-tuned or unsteady quality appears on, the blame
belongs to what that rung added.

Run (DDSP venv):  python bare_test.py
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

H = Path(__file__).parent
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402
from live_v3 import EarV3  # noqa: E402
from prosody_ab import B1, GU, GL, DRY, motion  # noqa: E402

OUT = H / "out" / "clean_ab"
DUMP = H / "out/resp2_live_260804_160058.wav"


def his_phrases(x, k=2, min_s=3.0, xc_resp=0.5, xc_self=0.35):
    """Pick real solo phrases. The classifier is the peak normalised
    cross-correlation with the previous segment: a response contains a digital
    replay of the phrase before it (measured 0.56 to 0.95), while his own phrase is
    unrelated to what precedes it (measured 0.30 or below). It also requires that
    the NEXT segment be a response, which proves the model heard this phrase and
    answered it."""
    from scipy.signal import fftconvolve
    ph = [(s, e) for s, e in R.find_phrases(x, 0.35, 0.8)]
    xc = [0.0]
    for i in range(1, len(ph)):
        a = x[ph[i][0]:ph[i][1]]
        b = x[ph[i - 1][0]:ph[i - 1][1]]
        a = a / (np.sqrt((a ** 2).mean()) + 1e-9)
        b = b / (np.sqrt((b ** 2).mean()) + 1e-9)
        c = fftconvolve(a, b[::-1], mode="full")
        xc.append(float(np.abs(c).max() / min(len(a), len(b))))
    picks = [i for i in range(len(ph) - 1)
             if xc[i] <= xc_self and xc[i + 1] >= xc_resp
             and (ph[i][1] - ph[i][0]) / SR >= min_s]
    picks = sorted(picks, key=lambda i: ph[i][0] - ph[i][1])[:k]
    return [np.ascontiguousarray(x[ph[i][0]:ph[i][1]]) for i in picks], \
        [(ph[i][0] / SR, ph[i][1] / SR) for i in picks]


def bare_render(r, x, ratio, sh):
    """His f0 times a fixed ratio, straight into the model: what is left of
    phrase_render.render once the skeleton is removed. Every step (AGC, voicing
    mask, sentinel, volume mask, dividing the AGC gain back out, silencing unvoiced)
    matches line for line; only the source of f0 differs."""
    torch = r.torch
    n_hops = len(x) // dm.HOP + 1
    if "f0m" not in sh:
        sh["f0m"] = dm.harvest_f0(x)
    f0m = sh["f0m"]
    if "agc" not in sh:
        sh["agc"] = dm.input_agc(x, f0m=f0m)
    x_in, agc_g = sh["agc"]
    if "vm" not in sh:
        sh["vm"] = dm.voicing_mask(x, n_hops, f0m=f0m)
    vm = sh["vm"]
    f0 = np.zeros(n_hops)
    k = min(n_hops, len(f0m))
    f0[:k] = f0m[:k] * ratio
    f0 = np.where(f0 * vm > 0, f0 * vm, 1200.0)   # the unvoiced sentinel, as in phrase_render
    with torch.no_grad():
        if sh.get("units") is None:
            au = torch.from_numpy(x_in).float().unsqueeze(0).to(r.device)
            sh["units"] = r.encoder.encode(au, SR, dm.HOP)
            sh["vol"] = r.vol_ex.extract(x_in)
        units, vol_all = sh["units"], sh["vol"]
        n = min(units.size(1), n_hops)
        vol = vol_all[:n]
        mask = r._vol_mask(vol)[:n]
        f0_t = torch.from_numpy(f0[:n]).float().to(r.device)[None, :, None]
        vol_t = torch.from_numpy(np.asarray(vol)).float().to(r.device)[None, :, None]
        out, _, _ = r.model(units[:, :n], f0_t, vol_t, spk_id=r.spk)
        wav = out.squeeze().cpu().numpy().astype(np.float64)
    m_up = np.repeat(mask, dm.HOP)[: len(wav)]
    wav[: len(m_up)] *= m_up
    kk = min(len(wav), len(agc_g))
    wav = wav[:kk] / agc_g[:kk]
    ang, _ = dm.uv_passthrough(wav, x[:kk], vm, dry_gain=0.0)  # the same stems as the A/B
    return ang


def wr(name, w):
    sf.write(str(OUT / name), w / (np.abs(w).max() + 1e-9) * 0.7, SR,
             subtype="PCM_16")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "src").mkdir(exist_ok=True)
    import torch
    x, _ = sf.read(str(DUMP), dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs, ts = his_phrases(x)
    for (a, b) in ts:
        print(f"seg at {a:.1f}-{b:.1f}s (a solo phrase, confirmed by cross-correlation)")

    torch.manual_seed(1234)
    ear = EarV3(indep=0.15, stab=1, key=0)
    ear.v2t.shift = 12
    kt = R.KeyTracker()
    rend = {"harry": PhraseRenderer(R.DDSP, R.VOICES["lower"][0]),
            "girl": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                   expr_seed=R.VOICES["upper"][1]),
            "bass1": PhraseRenderer(R.DDSP, B1, expr_seed=20260805)}
    plan = [("harry_ident", "harry", 1.0),
            ("bass1_oct", "bass1", 0.5),
            ("girl_up", "girl", 2.0)]

    for i, seg in enumerate(segs):
        sf.write(str(OUT / "src" / f"seg{i}_src.wav"), seg, SR, subtype="FLOAT")
        wr(f"seg{i}_dry.wav", seg)
        sh = {"f0m": dm.harvest_f0(seg)}
        # --- the bare test: the harness only, no skeleton ---
        res = {}
        for tag, mdl, ratio in plan:
            ang = bare_render(rend[mdl], seg, ratio, sh)
            res[tag] = ang
            wr(f"seg{i}_bare_{tag}.wav", ang)
        n = min(len(seg), *(len(a) for a in res.values()))
        wr(f"seg{i}_bare_mix.wav",
           DRY * seg[:n] + GU * res["girl_up"][:n]
           + GL * res["bass1_oct"][:n])
        # --- the A baseline: the current recipe, that is the quantised skeleton ---
        kt.push(sh["f0m"])
        rb = kt.best()
        if rb is not None and rb[2] >= 0.015:
            ear.k_shift = rb[0]
        lead, up, lo = R.phrase_notes(ear, seg)
        d = {"upper": up, "lower": lo}
        import json
        json.dump({v: [None if t is None else int(t) for t in d[v]]
                   for v in d},
                  open(OUT / f"seg{i}_notes.json", "w"))
        # write the note line out, so later A/Bs such as a_enh.py consume the same
        # one; the ear has randomness and is sensitive to call order, so a
        # single-variable comparison has to lock the line.
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        mouth = {"upper": "girl", "lower": "bass1"}
        angA = {v: rend[mouth[v]].render(seg, d[v], TICK_SAMPS, shared=sh,
                                         **kw[v]) for v in R.VOICES}
        n = min(len(seg), *(len(a) for a in angA.values()))
        wr(f"seg{i}_A_mix.wav",
           DRY * seg[:n] + GU * angA["upper"][:n] + GL * angA["lower"][:n])
        wr(f"seg{i}_A_bass1_lower.wav", angA["lower"][:n])
        print(f"seg{i}: frame-to-frame f0 activity, dry {motion(seg):.1f}c | "
              + " ".join(f"{t} {motion(w):.1f}c" for t, w in res.items())
              + f" ｜ A_lower {motion(angA['lower'][:n]):.1f}c")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
