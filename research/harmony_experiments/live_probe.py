"""A synthetic live probe: BlackHole as a virtual microphone and a feeder playing a
wav, so the whole three-thread chain with a real sd.Stream can be tested with
nobody in the room. The output gain is 0, so the desk stays silent and everything
is measured from the closing report and the microphone dump.

Usage (DDSP venv, from harmony/):
  .../260724_ddsp_svc/venv/bin/python live_probe.py out/loopback_0730/seg_take.wav \
      --lag 0.45 -- --free-run 0        # anything after -- is passed straight through

A lesson learned the hard way: a process started with a shell `&` has SIGINT set to
ignored by POSIX, so pkill -INT does nothing at all. It has to be Popen with no
shell plus proc.send_signal(SIGINT).
Feeding starts the instant the engine prints its opening line, which removes the
silent lead-in that would otherwise contaminate the measurement.
"""
import argparse
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--lag", default="0.6")
ap.add_argument("--key", default="0")
ap.add_argument("--in-name", default="BlackHole")
a, rest = ap.parse_known_args()   # unrecognised arguments pass straight through to the engine
if rest and rest[0] == "--":
    rest = rest[1:]
a.rest = rest

x, sr = sf.read(a.wav, dtype="float32", always_2d=True)
assert sr == 44100, sr

cmd = [sys.executable, "live_v3.py", "--in-name", a.in_name,
       "--gain", "0", "--lag", a.lag, "--key", a.key, *a.rest]
print("probe:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)

lines, started = [], threading.Event()


def reader():
    for ln in proc.stdout:
        lines.append(ln.rstrip())
        print("  |", ln.rstrip(), flush=True)
        if ln.startswith("live v3."):
            started.set()


threading.Thread(target=reader, daemon=True).start()
if not started.wait(120):
    proc.kill()
    sys.exit("the engine never started (120s)")

time.sleep(0.3)
print(f"probe: feeding {len(x) / sr:.1f}s into {a.in_name}", flush=True)
sd.play(x, sr, device=a.in_name, blocking=True)
time.sleep(float(a.lag) + 1.0)          # let the tail play out, so the closing statistics cover all the material
proc.send_signal(signal.SIGINT)
proc.wait(timeout=60)
print("probe: done", flush=True)
