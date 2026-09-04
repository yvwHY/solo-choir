"""a_enh.py - the A recipe plus the NSF-HiFiGAN enhancer.

The verdict this follows: "the enhanced files are actually fine; I meant the
un-enhanced bass wav". So the auto-tuned quality is not in the pitch curve - the f0
of the reference and bare cells was measured as his own curve at -12 with 1 to 2
cents of deviation - but in **the dead flat harmonics of the bare CombSub output in
the low register**, which disappear once the NSF-HiFiGAN enhancer rebuilds them.

This file hangs the enhancer on the current recipe (the `enhance=True` guard
parameter of phrase_render) and renders an A-plus-enhancer version, on the same two
phrases as clean_ab, selected by cross-correlation:
  seg{i}_Aenh_mix.wav / seg{i}_Aenh_bass1_lower.wav
If it passes by ear, the enhancer joins the recipe and its live real-time budget is
measured next. If not, the enhancer cannot save the recipe layer either, and the
interaction between the skeleton and the enhancer is where to look.

Run (DDSP venv):  python a_enh.py
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

H = Path(__file__).parent
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402
from prosody_ab import B1, GU, GL, DRY  # noqa: E402
from bare_test import OUT, DUMP, his_phrases, wr  # noqa: E402


def main():
    import torch
    x, _ = sf.read(str(DUMP), dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    R.RMS_GATE = 0.02
    segs, _ = his_phrases(x)

    torch.manual_seed(1234)
    rend = {"upper": PhraseRenderer(R.DDSP, R.VOICES["upper"][0],
                                    expr_seed=R.VOICES["upper"][1],
                                    enhance=True),
            "lower": PhraseRenderer(R.DDSP, B1, expr_seed=20260805,
                                    enhance=True)}
    for r_ in rend.values():
        r_.render(np.zeros(SR), [60], TICK_SAMPS)

    for i, seg in enumerate(segs):
        sh = {"f0m": dm.harvest_f0(seg)}
        # the note line is read from what bare_test wrote, not recomputed: the ear
        # has randomness and is sensitive to call order, so recomputing forks the
        # line and it stops being a single-variable comparison (measured: after a
        # fork only 11 to 37% of frames hold the same note).
        d = json.load(open(OUT / f"seg{i}_notes.json"))
        kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0,
                      uv_dry=0.0, vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1])
              for v in R.VOICES}
        t0 = time.perf_counter()
        ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh, **kw[v])
               for v in R.VOICES}
        dt = time.perf_counter() - t0
        n = min(len(seg), *(len(a) for a in ang.values()))
        wr(f"seg{i}_Aenh_mix.wav",
           DRY * seg[:n] + GU * ang["upper"][:n] + GL * ang["lower"][:n])
        wr(f"seg{i}_Aenh_bass1_lower.wav", ang["lower"][:n])
        print(f"seg{i}: both parts with the enhancer, {dt:.2f}s for {len(seg)/SR:.1f}s of audio"
              f" = RTF {dt/(len(seg)/SR):.2f}")
    print("wrote", OUT, "(seg*_Aenh_*.wav)")


if __name__ == "__main__":
    main()
