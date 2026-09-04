"""prosody_ab.py - A/B/B2 for transplanting prosody.

**Superseded, and its results are void.** The `[0::2]` selection here picked the
RESPONSE segments out of the dump - his playback and both parts on one track - so a
three-part mix was treated as his own phrase. Everything under out/prosody_ab/ and
every number it produced is withdrawn. The re-run is bare_test.py, which selects by
cross-correlation into out/clean_ab/. Kept as a record; do not run it.

The diagnosis it followed: the auto-tuned quality is not in the model layer but in
the expression layer. The parts' f0 is synthesised from quantised score positions -
a flat line per note, a 33 ms jump between them, vibrato at a fixed rate - and his
real pitch curve is discarded entirely (frame-to-frame activity falls from 13 to 5
cents).

The fix transplants the deviation curve of his real f0 onto that skeleton
(`phrase_render prosody_cents`):
- dev(t) = his f0 in MIDI minus his own quantised line, with gaps held: the scoops,
  the drift and the phrasing
- a moving average removes his vibrato, since the fast component still comes from
  the existing desynchronised vibrato template, and it is clamped to +/-60 cents
- **both parts share one curve**, so the interval is constant and the failure of the
  earlier express layer - each part deviating on its own - is structurally impossible
- B uses a 0.29 s window, keeping only the slow phrasing component (the first
  measurement showed the scoops halved, so half the human quality arrives)
  B2 uses 0.12 s, keeping the 0.1 to 0.3 s scoop band

Output (two phrases of material, comparable by ear with the two earlier trials):
  segN_mix_{A,B,B2}.wav          A is the current recipe
  segN_bass1_lower_{A,B,B2}.wav  bare stems for close listening

Run (DDSP venv):  python prosody_ab.py
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

B1 = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
      "260724_ddsp_svc/exp/combsub-m4-bass1/model_11000.pt")
OUT = H / "out" / "prosody_ab"
GU = 0.75 * 10 ** (1.0 / 20.0)     # the current mix: upper part +1 dB
GL = 0.75
DRY = 0.944
MA_SLOW, MA_SCOOP = 25, 10         # moving-average windows of 0.29 s and 0.12 s
CLAMP = 60.0


def prosody_curve(f0m, n_hops, ma=MA_SLOW):
    """His f0 (harvest, Hz per hop) to a deviation curve in cents per hop."""
    m = np.full(n_hops, np.nan)
    k = min(n_hops, len(f0m))
    v = f0m[:k] > 0
    m[:k][v] = 69 + 12 * np.log2(f0m[:k][v] / 440.0)
    line = np.copy(m)
    last = np.nan
    for i in range(n_hops):
        if np.isnan(line[i]):
            line[i] = last
        else:
            last = line[i]
    line = np.round(line)
    dev = (m - line) * 100.0
    dev[np.isnan(dev)] = 0.0        # add nothing over silence
    ker = np.ones(ma) / ma
    dev = np.convolve(dev, ker, mode="same")
    return np.clip(dev, -CLAMP, CLAMP)


def motion(w):
    f0 = dm.harvest_f0(np.ascontiguousarray(w.astype(np.float64)))
    v = f0[f0 > 0]
    mm = 69 + 12 * np.log2(v / 440.0)
    return float(np.median(np.abs(np.diff(mm)) * 100))


def main():
    OUT.mkdir(exist_ok=True)
    import torch
    x, _ = sf.read(str(H / "out/resp2_live_260804_160058.wav"),
                   dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs = []
    for s, e in R.find_phrases(x, 0.35, 0.8):
        if s / SR < 12:
            continue
        f0 = dm.harvest_f0(np.ascontiguousarray(x[s:e]))
        v = f0[f0 > 0]
        if len(v) < 30:
            continue
        if float(np.mean(v > 340.0)) < 0.05:
            segs.append(np.ascontiguousarray(x[s:e]))
    segs = sorted(segs[0::2], key=len, reverse=True)[:2]

    torch.manual_seed(1234)
    ear = EarV3(indep=0.15, stab=1, key=0)
    ear.v2t.shift = 12
    kt = R.KeyTracker()
    rend = {"upper": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                    expr_seed=R.VOICES["upper"][1]),
            "lower": PhraseRenderer(R.DDSP, B1, expr_seed=20260805)}
    for r_ in rend.values():
        r_.render(np.zeros(SR), [60], TICK_SAMPS)

    for i, seg in enumerate(segs):
        kt.push(dm.harvest_f0(seg))
        rb = kt.best()
        if rb is not None and rb[2] >= 0.015:
            ear.k_shift = rb[0]
        lead, up, lo = R.phrase_notes(ear, seg)
        d = {"upper": up, "lower": lo}
        f0m = dm.harvest_f0(seg)
        n_hops = len(seg) // dm.HOP + 1
        curves = {"A": None,
                  "B": prosody_curve(f0m, n_hops, MA_SLOW),
                  "B2": prosody_curve(f0m, n_hops, MA_SCOOP),
                  "C": prosody_curve(f0m, n_hops, 1),   # full transplant, including his vibrato
                  "D": "dlead",                          # continuous f0 times the interval ratio, below
                  "E": "dlead+sus"}                      # D plus bridging gaps up to 250 ms
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        res = {}
        for ver, pc in curves.items():
            sus = isinstance(pc, str) and "sus" in pc
            if isinstance(pc, str):      # D and E: dev = his continuous f0 minus the model's lead line
                m = np.full(n_hops, np.nan)
                kk = min(n_hops, len(f0m)); vv = f0m[:kk] > 0
                m[:kk][vv] = 69 + 12*np.log2(f0m[:kk][vv]/440.0)
                ll = np.full(n_hops, np.nan)
                off = ear.v2t.shift + (ear.k_shift or 0)   # model space to microphone space
                for h in range(n_hops):
                    t_i = min(int(h*dm.HOP/TICK_SAMPS), len(lead)-1)
                    if lead[t_i] is not None: ll[h] = lead[t_i] - off
                last = np.nan
                for h in range(n_hops):
                    if np.isnan(ll[h]): ll[h] = last
                    else: last = ll[h]
                pc = (m - ll) * 100.0
                pc[np.isnan(pc)] = 0.0
                pc = np.clip(pc, -250, 250)   # catch octave errors only, without cutting the glides
            extra = {} if pc is None else {"prosody_cents": pc}
            if sus:
                extra["sustain_ms"] = 250.0
            kw_v = kw
            if ver in ("C", "D", "E"):   # the vibrato is already in his curve, so turn the synthetic template off
                kw_v = {v: dict(kw[v], vib_semi=0.0) for v in R.VOICES}
            sh_v = {"f0m": f0m}
            ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh_v,
                                     **kw_v[v], **extra) for v in R.VOICES}
            n = min(len(seg), *(len(a) for a in ang.values()))
            mix = DRY * seg[:n] + GU * ang["upper"][:n] + GL * ang["lower"][:n]
            sf.write(str(OUT / f"seg{i}_mix_{ver}.wav"),
                     mix / (np.abs(mix).max() + 1e-9) * 0.7, SR,
                     subtype="PCM_16")
            st = ang["lower"][:n]
            sf.write(str(OUT / f"seg{i}_bass1_lower_{ver}.wav"),
                     st / (np.abs(st).max() + 1e-9) * 0.7, SR,
                     subtype="PCM_16")
            res[ver] = st
        print(f"seg{i}: frame-to-frame f0 activity, dry {motion(seg):.1f}c | "
              + " ".join(f"{v} {motion(w):.1f}c" for v, w in res.items())
              + f" | injected magnitude p50 B {np.median(np.abs(curves['B'])):.1f}c"
                f" B2 {np.median(np.abs(curves['B2'])):.1f}c")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
