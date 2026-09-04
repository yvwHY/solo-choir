"""A synthetic probe for respond2 --live: BlackHole as a virtual microphone, so the whole chain runs with no hardware.

The difference from live_probe.py: **the answering mode has to wait.** After a
phrase ends, respond2 hard-mutes its input all the way through the response and its
reverb tail. If the feeder keeps playing, his "next phrase" lands entirely inside
that mute window and is swallowed, so the phrase lengths and waits measured are
fictional (this is the other reason an earlier synthetic probe underestimated by a
factor of three). So this reads the mute-window duration respond2 prints and PAUSES
THE PLAY HEAD, simulating a singer waiting for the response before the next phrase.

It measures three things: whether the chain runs end to end; input overflow and
underflow (pipelining starts the model and the voicing while he is still singing,
which is the only risk of input being squeezed out); and, per phrase, the time from
him stopping to the response starting.

Usage (DDSP venv, from harmony/):
  .../260724_ddsp_svc/venv/bin/python resp2_probe.py take.wav --key 3
  ... resp2_probe.py take.wav --key 3 -- --no-pipeline      # anything after -- goes to respond2

A lesson learned the hard way: a process started with a shell `&` has SIGINT set to
ignored, so this must use Popen with no shell plus proc.send_signal(SIGINT).
"""
import argparse
import re
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
ap.add_argument("--key", default="3")
ap.add_argument("--gate", default="0.02", help="fixed, so the opening calibration is skipped")
ap.add_argument("--in-name", default="BlackHole", help="respond2's virtual microphone")
a, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

x, sr = sf.read(a.wav, dtype="float32", always_2d=True)
assert sr == 44100, sr
x = x[:, :1]

cmd = [sys.executable, "respond2.py", "--live", "--in-name", a.in_name,
       "--key", a.key, "--gate", a.gate, "--gain", "0", "--no-tap", *rest]
print("probe:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True)

started = threading.Event()
st = {"pos": 0, "mute_until": 0.0}
RE = re.compile(r"brain\+render\s+([\d.]+)s.*mute window\s+([\d.]+)s")


def reader():
    for ln in proc.stdout:
        ln = ln.rstrip()
        print("  |", ln, flush=True)
        if ln.startswith("respond2 live."):
            started.set()
        m = RE.search(ln)
        if m:                      # what is left of the mute window is its total minus the model and render already spent
            st["mute_until"] = time.perf_counter() + max(
                0.0, float(m.group(2)) - float(m.group(1)))


threading.Thread(target=reader, daemon=True).start()
if not started.wait(180):
    proc.kill()
    sys.exit("respond2 never started (180s)")

done = threading.Event()


def cb(outdata, frames, t, status):
    outdata[:] = 0
    if time.perf_counter() < st["mute_until"]:
        return                      # he is listening to the response: do not advance the play head
    p = st["pos"]
    s = x[p:p + frames]
    if not len(s):
        done.set()
        return
    outdata[:len(s), 0] = s[:, 0]
    if outdata.shape[1] > 1:
        outdata[:len(s), 1] = s[:, 0]
    st["pos"] = p + len(s)


time.sleep(0.3)
print(f"probe: feeding {len(x) / sr:.1f}s into {a.in_name}"
      f" (the play head pauses while a response plays)", flush=True)
with sd.OutputStream(samplerate=sr, blocksize=1024, channels=2,
                     device=a.in_name, callback=cb):
    t0 = time.perf_counter()
    while not done.wait(0.2):
        if time.perf_counter() - t0 > len(x) / sr * 4 + 120:
            print("probe: timed out, stopping", flush=True)
            break
    time.sleep(max(0.0, st["mute_until"] - time.perf_counter()) + 1.0)

proc.send_signal(signal.SIGINT)
proc.wait(timeout=120)
print("probe: done", flush=True)
