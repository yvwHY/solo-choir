"""vc_probe — a minimal 'skinned vcclient' for fair A/B: ONE model, ONE voice, PURE conversion,
STATIC pitch (like vcclient's slider). No harmony, no multi-voice, no diatonic. Same Beatrice engine.

Purpose: isolate the engine/feeding path from Solo Choir's harmony machinery. If this sounds ==
vcclient, the quality gap is 100% the dynamic-harmony layer, not the engine or our feeding.

Run (conda env vcclient-dev):
    python eval/vc_probe.py --model <paraphernalia_dir> [--pitch 0] [--formant 0] [--speaker 0]
                            [--glide 0] [--dry] [--in-name USB] [--out-name speaker_set] [--list]
Keys while running: Ctrl-C to stop. Pitch/formant are set ONCE at start (static, like vcclient).
"""
from __future__ import annotations
import argparse, sys, os, time
import numpy as np
import sounddevice as sd
import soxr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from voice_changer.SoloChoir import SoloChoirHarmonizer
from beatrice_converter import BeatriceSoloChoir

DEV_SR = 48000


def pick(name, kind):
    """first device whose name contains `name` (substring), for the given kind ('input'/'output')."""
    if not name:
        return None
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch > 0 and name.lower() in d["name"].lower():
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="paraphernalia dir")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--pitch", type=float, default=0.0, help="STATIC pitch shift (semitones), like vcclient's slider")
    ap.add_argument("--formant", type=float, default=0.0)
    ap.add_argument("--glide", type=float, default=0.0, help="if >0, ramp pitch changes at this semitone/hop (only matters if you change pitch live via --sweep)")
    ap.add_argument("--dry", action="store_true", help="also pass the raw mic through (default: converted only, like vcclient)")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--blocksize", type=int, default=480, help="device frames/callback @48k (480=10ms)")
    ap.add_argument("--latency", default="low")
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = ap.parse_args()

    if args.list:
        print(sd.query_devices()); return

    conv = BeatriceSoloChoir(args.model, engine_dir=args.engine,
                             harmonizer=SoloChoirHarmonizer(enabled=False),  # NO harmony → pure conversion
                             target_speaker=int(args.speaker), formant_shift=float(args.formant))
    conv.pitch_offset = float(args.pitch)   # STATIC, set once (vcclient-style)
    if args.glide > 0.0:
        conv.pitch_glide_step = float(args.glide); conv.pitch_hold_frames = 1

    in_rs = soxr.ResampleStream(DEV_SR, conv.in_sr, 1, dtype="float32", quality="HQ")   # 48k -> 16k
    out_rs = soxr.ResampleStream(conv.out_sr, DEV_SR, 1, dtype="float32", quality="HQ")  # 24k -> 48k
    HOP = conv.in_sr // 100   # 160 @16k = 10ms

    in16 = np.zeros(0, dtype=np.float32)
    out48 = np.zeros(0, dtype=np.float32)
    dry_hist = np.zeros(0, dtype=np.float32)

    def cb(indata, outdata, frames, tinfo, status):
        nonlocal in16, out48, dry_hist
        if status:
            print(status, file=sys.stderr)
        mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if args.dry:
            dry_hist = np.concatenate([dry_hist, mono])
        x16 = np.asarray(in_rs.resample_chunk(mono), dtype=np.float32)
        in16 = np.concatenate([in16, x16])
        while len(in16) >= HOP:
            hop = in16[:HOP]; in16 = in16[HOP:]
            y24 = conv.process_frame(hop)
            y48 = np.asarray(out_rs.resample_chunk(np.ascontiguousarray(y24, dtype=np.float32)), dtype=np.float32)
            out48 = np.concatenate([out48, y48])
        n = frames
        y = np.zeros(n, dtype=np.float32)
        take = min(n, len(out48)); y[:take] = out48[:take]; out48 = out48[take:]
        if args.dry:
            d = np.zeros(n, dtype=np.float32); dt = min(n, len(dry_hist)); d[:dt] = dry_hist[:dt]; dry_hist = dry_hist[dt:]
            y = y + d
        outdata[:, 0] = y; outdata[:, 1] = y

    in_dev = pick(args.in_name, "input")
    out_dev = pick(args.out_name, "output")
    print(f"[vc_probe] model={os.path.basename(args.model)} speaker={args.speaker} pitch={args.pitch}st "
          f"formant={args.formant} glide={args.glide} dry={args.dry}", file=sys.stderr)
    print(f"[vc_probe] in={in_dev} out={out_dev} block={args.blocksize} (Ctrl-C to stop)", file=sys.stderr)
    with sd.Stream(samplerate=DEV_SR, blocksize=args.blocksize, dtype="float32",
                   channels=(1, 2), device=(in_dev, out_dev), latency=args.latency, callback=cb):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[vc_probe] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
