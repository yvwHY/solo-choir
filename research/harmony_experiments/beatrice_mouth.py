"""Test A mouth (2x2, worklog 07-24 §B): Beatrice timbre x clean target f0.

Feed the WORLD target-f0 resynthesis of Harry's take into Beatrice at
shift 0 -- pure timbre conversion, the pitch work already done upstream.
Beatrice has no absolute-f0 control, but we control the f0 of what it
hears: "rewriting Beatrice without touching the binary". Also the fourth
mouth candidate (offline; a streaming version would be file 1 of §B).

Run (vcclient-dev):
  /opt/anaconda3/envs/vcclient-dev/bin/python beatrice_mouth.py \
      out/angel_v2_world2_target_angel.wav out/angel_v2_beatr_target.wav
Extra solo_min flags pass through (e.g. --speaker 1, --model-dir ...).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import solo_host  # noqa: E402  (inserts <repo>/server on sys.path)
import solo_min  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_wav")
    ap.add_argument("out_wav")
    ap.add_argument("--model-dir", default=None)
    a, extra = ap.parse_known_args()

    import soundfile as sf
    import soxr

    eng = solo_host.build_engine(extra, a.model_dir)  # --steps 0 = shift stays 0
    eng.set_control({"you": 0.0})  # converted stem only

    x, sr = sf.read(a.in_wav, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != solo_min.DEV_SR:
        x = soxr.resample(x, sr, solo_min.DEV_SR).astype(np.float32)
    bs = int(eng.args.blocksize)
    if len(x) % bs:
        x = np.concatenate([x, np.zeros(bs - len(x) % bs, dtype=np.float32)])
    nch = int(getattr(eng, "n_out", 2))

    outs = []
    for b in range(0, len(x), bs):
        indata = x[b : b + bs].reshape(-1, 1)
        outdata = np.zeros((bs, nch), dtype=np.float32)
        eng._cb(indata, outdata, bs, None, None)
        outs.append(outdata.copy())
    angel = np.concatenate(outs).mean(axis=1)
    sf.write(a.out_wav, angel / max(1e-9, np.abs(angel).max()) * 0.9, solo_min.DEV_SR)
    print(f"wrote {a.out_wav} ({len(angel)/solo_min.DEV_SR:.1f}s)")


if __name__ == "__main__":
    main()
