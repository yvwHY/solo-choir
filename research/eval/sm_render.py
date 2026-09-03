"""Offline feed of solo_min.SoloEngine._cb — Regime A proof harness for solo_min-path changes.
Feeds a fixed wav through the exact live callback (no audio device) and writes the multi-channel
output, so before/after edits can be diffed bit-for-bit (tools/render_diff.py must print 0.0).

Usage: python sm_render.py IN.wav OUT.wav [extra solo_min flags...]
Engine args = solo_shell._default_args() launch set (satb2 5-part), minus device names.
"""
import sys, os
import numpy as np
import soundfile as sf
import soxr

REPO = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer"
sys.path.insert(0, os.path.join(REPO, "server"))
import solo_min  # noqa: E402

MODEL = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/model/VC/"
         "dist/model_dir/1/model/paraphernalia_data_satb2")
SECONDS = 10.0

def main():
    in_wav, out_wav = sys.argv[1], sys.argv[2]
    extra = sys.argv[3:]
    args = solo_min.build_argparser().parse_args([
        "--model", MODEL, "--mode", "diatonic", "--key", "C",
        "--steps", "-2", "--steps2", "2", "--steps3", "-4", "--steps4", "4",
        "--speaker", "0", "--speaker2", "0", "--speaker3", "1", "--speaker4", "0",
    ] + extra)
    eng = solo_min.SoloEngine(args)
    nch = int(getattr(eng, "n_out", 2))  # pre-change engines have no n_out → 2

    x, sr = sf.read(in_wav, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != solo_min.DEV_SR:
        x = soxr.resample(x, sr, solo_min.DEV_SR).astype(np.float32)
    x = x[: int(SECONDS * solo_min.DEV_SR)]
    bs = int(args.blocksize)
    if len(x) % bs:
        x = np.concatenate([x, np.zeros(bs - len(x) % bs, dtype=np.float32)])

    outs = []
    for i in range(0, len(x), bs):
        indata = x[i:i + bs].reshape(-1, 1)
        outdata = np.zeros((bs, nch), dtype=np.float32)
        eng._cb(indata, outdata, bs, None, None)
        outs.append(outdata.copy())
    y = np.concatenate(outs)
    sf.write(out_wav, y, solo_min.DEV_SR)
    print(f"[sm_render] wrote {out_wav} ch={nch} frames={len(y)} "
          f"peak={float(np.max(np.abs(y))):.4f} per-ch-peak={[round(float(np.max(np.abs(y[:, c]))), 4) for c in range(nch)]}")

if __name__ == "__main__":
    main()
