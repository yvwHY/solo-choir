"""latency_probe — electrical-loopback device RTT measurement (F6 re-run harness).

Cable: device OUT → same device IN (AI-Micro: headphone out → input 2, 3.5mm male-male).
Opens one duplex sd.Stream (same params as the F6 sweet-spot row), emits short clicks at known
output frame positions, records the input, and reports per-click round-trip = device output
buffer + wire + input buffer, on ONE clock (no drift ambiguity).

Full-chain mouth-to-ear estimate = RTT + engine hop (10 ms) + cushion + per-voice compute.

Run (vcclient-dev):  python eval/latency_probe.py --device "AI-Micro"
Keep Mac output volume moderate (~30-50%) so the click doesn't clip the mic input.
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48000


def pick(name: str) -> int:
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0 and d["max_output_channels"] > 0:
            return i
    raise SystemExit(f"duplex device not found: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="AI-Micro")
    ap.add_argument("--blocksize", type=int, default=240, help="F6 sweet-spot row = 240")
    ap.add_argument("--latency", type=float, default=0.004, help="F6 sweet-spot row = 0.004")
    ap.add_argument("--clicks", type=int, default=8)
    ap.add_argument("--gap-s", type=float, default=0.5)
    args = ap.parse_args()

    dev = pick(args.device)
    print(f"device[{dev}] = {sd.query_devices(dev)['name']}", file=sys.stderr)

    click_len = int(0.001 * SR)                       # 1 ms burst
    click = (0.8 * np.sin(2 * np.pi * 4000 * np.arange(click_len) / SR)).astype(np.float32)
    gap_frames = int(args.gap_s * SR)

    rec: list[np.ndarray] = []
    click_out_pos: list[int] = []
    state = {"out_count": 0, "next_click": SR // 2}   # first click after 0.5 s of settling

    def cb(indata, outdata, frames, t, status):
        if status:
            print(status, file=sys.stderr)
        rec.append(indata[:, 0].copy())
        outdata.fill(0.0)
        oc = state["out_count"]
        if len(click_out_pos) < args.clicks and oc + frames > state["next_click"]:
            off = max(0, state["next_click"] - oc)
            n = min(click_len, frames - off)
            outdata[off:off + n, 0] = click[:n]
            if outdata.shape[1] > 1:
                outdata[off:off + n, 1] = click[:n]
            click_out_pos.append(oc + off)
            state["next_click"] = oc + off + gap_frames
        state["out_count"] = oc + frames

    total_s = 1.0 + args.clicks * args.gap_s + 0.5
    with sd.Stream(samplerate=SR, blocksize=args.blocksize, dtype="float32", channels=(1, 2),
                   device=(dev, dev), latency=args.latency, callback=cb) as st:
        print(f"PortAudio reported latency: in={st.latency[0]*1000:.1f} ms  out={st.latency[1]*1000:.1f} ms",
              file=sys.stderr)
        time.sleep(total_s)

    x = np.concatenate(rec)
    thresh = max(0.05, float(np.max(np.abs(x))) * 0.4)
    if float(np.max(np.abs(x))) < 0.02:
        raise SystemExit("no signal captured — is the loopback cable in, and Mac output volume up?")

    rtts = []
    for pos in click_out_pos:
        seg = np.abs(x[pos:pos + gap_frames])
        hits = np.nonzero(seg > thresh)[0]
        if len(hits):
            rtts.append(hits[0] / SR * 1000.0)
    if not rtts:
        raise SystemExit("clicks emitted but none detected in the recording — check cable/level.")

    rtts_a = np.array(rtts)
    print(f"clicks detected: {len(rtts_a)}/{len(click_out_pos)}")
    print(f"device RTT: median {np.median(rtts_a):.1f} ms  (min {rtts_a.min():.1f} / max {rtts_a.max():.1f})")
    print(f"full-chain estimate (RTT + hop 10 ms + cushion 3 ms + ~1.5 ms compute): "
          f"~{np.median(rtts_a) + 14.5:.0f} ms mouth-to-ear")


if __name__ == "__main__":
    main()
