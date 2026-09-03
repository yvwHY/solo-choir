"""respond2 --live 的合成 probe：BlackHole 當虛擬 mic，無硬體跑完整條鏈。

與 live_probe.py（live_v3 用）的差別＝**應答式要會等**：respond2 斷句後輸入
硬靜音，一路到回應播完＋殘響尾巴。feeder 若一路播下去，他「下一句」會整段
落在靜音窗裡被吃掉，量到的句長與等待全是假的（08-02 合成 probe 低估 3 倍的
另一個成因）。所以這裡讀 respond2 自己印的「靜音窗」秒數，**播放頭原地暫停**
——模擬他等回應播完才唱下一句。

量的是三件事：①端到端跑不跑得完 ②io overflow/underflow（管線化在他唱的時候
就開始跑腦與 voicing，這是輸入被擠掉的唯一風險）③每句「他停唱到回應開播」。

用法（DDSP venv，於 harmony/ 下）:
  .../260724_ddsp_svc/venv/bin/python resp2_probe.py take.wav --key 3
  ... resp2_probe.py take.wav --key 3 -- --no-pipeline      # -- 之後傳給 respond2

血訓（07-31 §H）：shell `&` 背景起的行程 SIGINT 被設成忽略 → 必須 Popen
（無 shell）＋ proc.send_signal(SIGINT)。
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
ap.add_argument("--gate", default="0.02", help="給死＝跳過開場校準")
ap.add_argument("--in-name", default="BlackHole", help="respond2 的虛擬 mic")
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
RE = re.compile(r"腦＋渲染\s+([\d.]+)s.*靜音窗\s+([\d.]+)s")


def reader():
    for ln in proc.stdout:
        ln = ln.rstrip()
        print("  |", ln, flush=True)
        if ln.startswith("respond2 live."):
            started.set()
        m = RE.search(ln)
        if m:                      # 靜音窗剩下的部分＝總長減掉已經花掉的腦＋渲染
            st["mute_until"] = time.perf_counter() + max(
                0.0, float(m.group(2)) - float(m.group(1)))


threading.Thread(target=reader, daemon=True).start()
if not started.wait(180):
    proc.kill()
    sys.exit("respond2 沒開場（180s）")

done = threading.Event()


def cb(outdata, frames, t, status):
    outdata[:] = 0
    if time.perf_counter() < st["mute_until"]:
        return                      # 他在聽回應：播放頭不前進
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
      f"（回應播放期間暫停播放頭）", flush=True)
with sd.OutputStream(samplerate=sr, blocksize=1024, channels=2,
                     device=a.in_name, callback=cb):
    t0 = time.perf_counter()
    while not done.wait(0.2):
        if time.perf_counter() - t0 > len(x) / sr * 4 + 120:
            print("probe: 超時，收工", flush=True)
            break
    time.sleep(max(0.0, st["mute_until"] - time.perf_counter()) + 1.0)

proc.send_signal(signal.SIGINT)
proc.wait(timeout=120)
print("probe: done", flush=True)
