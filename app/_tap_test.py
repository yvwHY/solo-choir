"""Verify the server-side read-only telemetry tap (UI_BUILD_SPEC §6 step 3a).

No GUI, no mic: feed a synthetic 220 Hz tone (A3) through the REAL engine and check
  1. the tap reports the engine's real f0 / input level / voiced, and
  2. enabling the tap does NOT change the DSP output (y is bit-identical with/without).

    /opt/anaconda3/envs/vcclient-dev/bin/python app/_tap_test.py
"""
from __future__ import annotations

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
F = 220.0  # A3 → midi 57


def make_conv():
    return BeatriceSoloChoir(MODEL, harmonizer=SoloChoirHarmonizer(enabled=True))


def sine_frames(conv, n):
    # ONE continuous, phase-continuous tone sliced into frames (mimics a real mic),
    # not per-frame sines (those would inject a 160-sample = 100Hz boundary artifact).
    total = np.arange(n * conv.hop) / conv.in_sr
    sig = (0.3 * np.sin(2 * np.pi * F * total)).astype(np.float32)
    return [sig[i * conv.hop:(i + 1) * conv.hop].copy() for i in range(n)]


def main() -> None:
    # 1) tap reports real engine values
    conv = make_conv()
    last = {}
    conv.telemetry_tap = lambda **kw: last.update(kw)
    ys_tap = [conv.process_frame(s) for s in sine_frames(conv, 30)]  # warm the 2048 f0 window

    expected_rms = float(np.sqrt(np.mean((0.3 * np.sin(np.linspace(0, 2 * np.pi * F * conv.hop / conv.in_sr, conv.hop))) ** 2)))
    f0_ok = abs(last.get("f0", 0) - F) < 4.0
    lvl_ok = abs(last.get("in_level", 0) - expected_rms) < 0.03
    voiced_ok = last.get("voiced") is True
    print(f"tap: f0={last.get('f0'):.2f}Hz (exp~{F})  in_level={last.get('in_level'):.3f} "
          f"(exp~{expected_rms:.3f})  out_level={last.get('out_level'):.3f}  voiced={last.get('voiced')}")
    print(f"     f0_ok={f0_ok}  level_ok={lvl_ok}  voiced_ok={voiced_ok}")

    # 2) tap does not alter the DSP output: fresh conv, no tap, same input → identical y
    conv2 = make_conv()
    ys_notap = [conv2.process_frame(s) for s in sine_frames(conv2, 30)]
    identical = all(np.array_equal(a, b) for a, b in zip(ys_tap, ys_notap))
    print(f"DSP unchanged by tap (y identical with/without tap): {identical}")

    ok = f0_ok and lvl_ok and voiced_ok and identical
    print(f"=== TAP: {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
