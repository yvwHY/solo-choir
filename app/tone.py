#!/usr/bin/env python3
"""tone.py — play a tone through every output device, so you can hear which
loudspeaker carries which device name.

    python app/tone.py              # step through every output device, 1.5 s each
    python app/tone.py speaker_set  # one device only (name or index)
    python app/tone.py --ch 0 spk   # left channel only (--ch 1 = right only)

Left and right carry different pitches by default, 440 Hz left and 660 Hz
right, so the sides are told apart by ear.
"""
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48000
DUR = 1.5


def tone(dev, name, only_ch=None):
    n = np.arange(int(SR * DUR))
    env = np.minimum(1.0, np.minimum(n, len(n) - n) / (0.02 * SR))   # fade in and out at the ends
    l = 0.25 * np.sin(2 * np.pi * 440.0 * n / SR) * env
    r = 0.25 * np.sin(2 * np.pi * 660.0 * n / SR) * env
    if only_ch == 0:
        r = r * 0
    elif only_ch == 1:
        l = l * 0
    sig = np.stack([l, r], axis=1).astype(np.float32)
    tag = "" if only_ch is None else f" ({'left' if only_ch == 0 else 'right'} channel only)"
    print(f"  > {dev:2d} {name}{tag}   left=440Hz right=660Hz")
    try:
        sd.play(sig, SR, device=dev, blocking=True)
    except Exception as e:
        print(f"     x could not play: {e}")
    time.sleep(0.3)


def main():
    args = [a for a in sys.argv[1:]]
    only_ch = None
    if "--ch" in args:
        i = args.index("--ch")
        only_ch = int(args[i + 1])
        del args[i:i + 2]

    devs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] >= 2]
    if args:
        want = args[0].lower()
        try:
            devs = [(int(want), sd.query_devices(int(want))["name"])]
        except ValueError:
            devs = [(i, n) for i, n in devs if want in n.lower()]
        if not devs:
            sys.exit(f"output device not found: {args[0]}")

    print("Playing in order. Whichever speaker sounds is the one that name refers to. Ctrl-C to stop.\n")
    for i, n in devs:
        if "blackhole" in n.lower():
            print(f"  - {i:2d} {n} (skipped: virtual device, makes no sound)")
            continue
        tone(i, n, only_ch)
    print("\nDone.")


if __name__ == "__main__":
    main()
