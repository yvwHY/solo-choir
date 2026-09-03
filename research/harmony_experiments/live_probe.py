"""合成 live probe：BlackHole 當虛擬 mic、feeder 播 wav，live_v3 真 sd.Stream
三執行緒全鏈無人測試（07-31 §G/§H 用過；當時沒進 repo＝儀器教訓再犯一次，
本檔固化）。輸出 gain 0＝桌面無聲，量測全靠收官報表＋mic dump。

用法（DDSP venv，於 harmony/ 下）:
  .../260724_ddsp_svc/venv/bin/python live_probe.py out/loopback_0730/seg_take.wav \
      --lag 0.45 -- --free-run 0        # -- 之後原樣傳給 live_v3

血訓（07-31 §H）：shell `&` 背景起的行程 SIGINT 被 POSIX 設成忽略，
pkill -INT 全空包——必須 Popen（無 shell）＋ proc.send_signal(SIGINT)。
即餵：live_v3 印出開場行後立刻開始播，消除靜音前導污染。
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
a, rest = ap.parse_known_args()   # 未認得的參數原樣傳給 live_v3
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
    sys.exit("live_v3 沒開場（120s）")

time.sleep(0.3)
print(f"probe: feeding {len(x) / sr:.1f}s into {a.in_name}", flush=True)
sd.play(x, sr, device=a.in_name, blocking=True)
time.sleep(float(a.lag) + 1.0)          # 讓尾巴播完、收官統計含完整素材
proc.send_signal(signal.SIGINT)
proc.wait(timeout=60)
print("probe: done", flush=True)
