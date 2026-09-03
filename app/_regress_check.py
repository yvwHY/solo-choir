"""Capture the converter's output on a fixed tone with DEFAULT controls, to a .npy.

Used by the regression check: run once with the current code, once after `git stash`
(original DSP), then compare the two .npy for byte-identity. Default controls =
harmonizer enabled, C major, interval -2, pitch 0 — i.e. the 3b baseline.

    python app/_regress_check.py /tmp/out.npy
"""
import sys
from pathlib import Path

import numpy as np

SERVER = Path(__file__).resolve().parent.parent / "server"
sys.path.insert(0, str(SERVER))

from beatrice_converter import BeatriceSoloChoir            # noqa: E402
from voice_changer.SoloChoir import SoloChoirHarmonizer     # noqa: E402

sys.path.insert(0, str(SERVER.parent))
from config import BEATRICE_MODELS  # noqa: E402
MODEL = str(BEATRICE_MODELS / "paraphernalia_data_00002000")


def main():
    out_path = sys.argv[1]
    conv = BeatriceSoloChoir(MODEL, harmonizer=SoloChoirHarmonizer(
        enabled=True, key_root=0, minor=False, interval_steps=-2))  # 3b defaults
    n = 60
    total = np.arange(n * conv.hop) / conv.in_sr
    sig = (0.3 * np.sin(2 * np.pi * 220.0 * total)).astype(np.float32)   # continuous A3
    ys = [conv.process_frame(sig[i * conv.hop:(i + 1) * conv.hop].copy()) for i in range(n)]
    np.save(out_path, np.concatenate(ys))
    print(f"wrote {out_path}: {sum(len(y) for y in ys)} samples")


if __name__ == "__main__":
    main()
