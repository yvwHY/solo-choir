"""prerender_stems.py - a rehearsal file to offline-quality part stems (the pre-rendered offline half).

The direction was to improve the parts rather than the live engine. For a rehearsed
song the parts' audio can leave the real-time constraint entirely: the offline
reference direct_mouth (--f0-mode express, the F21 recipe, unvoiced pass-through and
the default input AGC) renders the two part lines into stems on the same timeline as
the rehearsal take. The live end (live_v3 --pre-stems) aligns them with the score
tracker and writes them into future cells, so offline ceiling quality reaches the
live output and none of the voice pipeline's real-time compromises exist.

The driver is written down here because an ad-hoc driver that was not kept carries no authority across a comparison table.

Run（DDSP venv）:
  ../../../260724_ddsp_svc/venv/bin/python prerender_stems.py \
      --rehearse out/pair06_reh_notes.json \
      --take ../../../260730_recording/pairs/pair06_take1.wav --tag pair06_pre

Output: out/{tag}_{voice}_angel.wav, 44.1k mono, on the take's timeline
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DDSP = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260724_ddsp_svc"
TICK_SAMPS = int((60.0 / 80.0) * 0.25 * 44100)   # 8268, the model's pulse in world_live

# the same default voices as live_v3, with a per-voice seed for desynchronisation
VOICES = {"upper": (f"{DDSP}/exp/combsub-girl/model_30000.pt", 20260731),
          "lower": (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt", 20260732)}
# vibrato desynchronisation for target mode (the original values from
# render_v3.py:339-340). The F21 correction: express was only validated on a solo
# voice, and two parts sounding at once pull the harmony apart (interval error SD 40.7 cents).
VIB = {"upper": (5.3, 0.0), "lower": (4.6, 0.5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", required=True,
                    help="the *_notes.json written by live_v3 in reactive file mode (upper/lower/lead)")
    ap.add_argument("--take", required=True, help="the rehearsal take; the stems share its timeline")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expr-gain", type=float, default=1.0)
    ap.add_argument("--f0-mode", choices=["target", "express"], default="target",
                    help="target is the skeleton with fixed vibrato desynchronisation, the "
                         "two-part recipe after the F21 correction; express is the expression "
                         "layer, which is only right for a solo voice")
    ap.add_argument("--ceiling", action="store_true",
                    help="also write the ceiling mix out/{tag}_ceiling_mix.wav (take at 0.5 "
                         "and each part at 0.6, the same final recipe as live_v3), which judges "
                         "stem quality with alignment taken out of it")
    a = ap.parse_args()
    reh = json.load(open(a.rehearse))
    for v, (model, seed) in VOICES.items():
        nj = os.path.join(HERE, "out", f"{a.tag}_{v}_notes.json")
        json.dump({"notes": reh[v], "step_samps": TICK_SAMPS}, open(nj, "w"))
        cmd = [sys.executable, os.path.join(HERE, "direct_mouth.py"),
               "--notes", nj, "--take", a.take, "--tag", f"{a.tag}_{v}",
               "--model", model, "--key", "0", "--f0-mode", a.f0_mode, "--run"]
        if a.f0_mode == "express":
            cmd += ["--expr-gain", str(a.expr_gain), "--expr-seed", str(seed)]
        else:
            cmd += ["--vib-hz", str(VIB[v][0]), "--vib-phase", str(VIB[v][1]),
                    "--vib-onset-ms", "250"]
        print(">>", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode:
            sys.exit(r.returncode)
    print("stems:", ", ".join(f"out/{a.tag}_{v}_angel.wav" for v in VOICES))

    if a.ceiling:
        import numpy as np
        import soundfile as sf
        take, sr = sf.read(a.take, dtype="float64", always_2d=True)
        take = take[:, 0]
        ang = [sf.read(os.path.join(HERE, "out", f"{a.tag}_{v}_angel.wav"),
                       dtype="float64")[0] for v in VOICES]
        n = min(len(take), *(len(w) for w in ang))
        mix = 0.5 * take[:n] + 0.6 * ang[0][:n] + 0.6 * ang[1][:n]
        p = os.path.join(HERE, "out", f"{a.tag}_ceiling_mix.wav")
        sf.write(p, mix / (np.abs(mix).max() + 1e-12) * 0.9, sr, subtype="PCM_16")
        print("ceiling:", p)


if __name__ == "__main__":
    main()
