"""spatial_pilot — does PHYSICAL source separation de-fuse the choir? (Step 0 pilot)

Hypothesis under test (2026-07-14, pre-purchase gate for the 4-speaker exhibition rig):
the "choir sounds like one person" fusion (GRAVEYARD G9 = correlation, FINDINGS F3)
is broken by giving voices physically separate sources. This plays existing studio
per-voice stems — NO signal processing whatsoever (G11 buried in-path widening;
this script only routes) — through one vs two devices for an A/B by ear.

  A (baseline): all 4 stems mixed → ONE device
  B (spatial):  sop+alto → device A, tenor+bass → device B, spaced 1–2 m apart

Run (vcclient-dev; QUIT or mute the live app first — its mic will pick this up):
  python eval/spatial_pilot.py recordings/studio_render_20260707_114554 --mode mono
  python eval/spatial_pilot.py recordings/studio_render_20260707_114554 --mode split

Pass = B has a clear "several singers in the room" quality A lacks.
Fail = no meaningful difference → bury the 4-speaker idea in GRAVEYARD, done cheap.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf

STEMS = ("studio_sop.wav", "studio_alto.wav", "studio_tenor.wav", "studio_bass.wav")


def pick_out(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return i
    raise SystemExit(f"output device not found: {name}")


def load_stems(folder: pathlib.Path):
    xs, srs = [], set()
    for f in STEMS:
        x, sr = sf.read(folder / f, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        xs.append(x)
        srs.add(sr)
    if len(srs) != 1:
        raise SystemExit(f"stems disagree on sample rate: {srs}")
    n = max(len(x) for x in xs)
    xs = [np.pad(x, (0, n - len(x))) for x in xs]
    return xs, srs.pop()


def play(dev: int, mono: np.ndarray, sr: float, barrier: threading.Barrier):
    stereo = np.repeat(mono[:, None], 2, axis=1)
    with sd.OutputStream(samplerate=sr, device=dev, channels=2, dtype="float32") as stream:
        barrier.wait()
        stream.write(stereo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem_dir", help="a recordings/studio_render_* folder")
    ap.add_argument("--mode", choices=("mono", "split"), required=True)
    ap.add_argument("--dev-a", default="AI-Micro", help="device A (mono baseline plays here)")
    ap.add_argument("--dev-b", default="揚聲器",  # the macOS speaker device name on a
                    # zh-Hant system, matched literally against the device list
                    help="device B (split mode: tenor+bass)")
    ap.add_argument("--gain", type=float, default=1.0, help="overall gain")
    ap.add_argument("--gain-b", type=float, default=1.0, help="extra gain on device B (level-match by ear)")
    ap.add_argument("--seconds", type=float, default=0, help="limit playback length (0 = full)")
    args = ap.parse_args()

    xs, sr = load_stems(pathlib.Path(args.stem_dir))
    if args.seconds:
        n = int(args.seconds * sr)
        xs = [x[:n] for x in xs]
    # 0.35 per stem keeps a 4-stem sum inside ±1 with headroom; identical gain both modes
    g = 0.35 * args.gain
    dev_a = pick_out(args.dev_a)

    if args.mode == "mono":
        mix = g * sum(xs)
        print(f"A/baseline: all 4 stems → [{dev_a}] {sd.query_devices(dev_a)['name']}")
        barrier = threading.Barrier(1)
        play(dev_a, mix, sr, barrier)
    else:
        dev_b = pick_out(args.dev_b)
        mix_a = g * (xs[0] + xs[1])                    # sop + alto
        mix_b = g * args.gain_b * (xs[2] + xs[3])      # tenor + bass
        print(f"B/spatial: sop+alto → [{dev_a}] {sd.query_devices(dev_a)['name']}")
        print(f"           tenor+bass → [{dev_b}] {sd.query_devices(dev_b)['name']}")
        barrier = threading.Barrier(2)
        ta = threading.Thread(target=play, args=(dev_a, mix_a, sr, barrier))
        ta.start()
        play(dev_b, mix_b, sr, barrier)
        ta.join()
    print("done")


if __name__ == "__main__":
    main()
