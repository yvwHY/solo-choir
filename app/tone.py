#!/usr/bin/env python3
"""tone.py — 對著每個輸出裝置播一聲，用耳朵確認哪顆喇叭是哪個名字。

    python app/tone.py              # 依序播過所有輸出裝置，每個 1.5 秒
    python app/tone.py speaker_set  # 只播指定裝置（名字或編號都可以）
    python app/tone.py --ch 0 spk   # 只從左聲道播（--ch 1 ＝只右聲道）

左右分辨：預設左聲道 440Hz、右聲道 660Hz，兩個音高不一樣，一聽就知道
哪邊是左哪邊是右。
"""
import sys
import time

import numpy as np
import sounddevice as sd

SR = 48000
DUR = 1.5


def tone(dev, name, only_ch=None):
    n = np.arange(int(SR * DUR))
    env = np.minimum(1.0, np.minimum(n, len(n) - n) / (0.02 * SR))   # 頭尾淡入淡出
    l = 0.25 * np.sin(2 * np.pi * 440.0 * n / SR) * env
    r = 0.25 * np.sin(2 * np.pi * 660.0 * n / SR) * env
    if only_ch == 0:
        r = r * 0
    elif only_ch == 1:
        l = l * 0
    sig = np.stack([l, r], axis=1).astype(np.float32)
    tag = "" if only_ch is None else f"（只有{'左' if only_ch == 0 else '右'}聲道）"
    print(f"  ▶ {dev:2d} {name}{tag}   左=440Hz 右=660Hz")
    try:
        sd.play(sig, SR, device=dev, blocking=True)
    except Exception as e:
        print(f"     ✗ 播不出來：{e}")
    time.sleep(0.3)


def main():
    args = [a for a in sys.argv[1:]]
    only_ch = None
    if "--ch" in args:
        i = args.index("--ch")
        only_ch = int(args[i + 1])
        del args[i:i + 2]

    devs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] >= 2]
    if args:
        want = args[0].lower()
        try:
            devs = [(int(want), sd.query_devices(int(want))["name"])]
        except ValueError:
            devs = [(i, n) for i, n in devs if want in n.lower()]
        if not devs:
            sys.exit(f"找不到輸出裝置：{args[0]}")

    print("依序播。聽到哪顆喇叭響，就知道那個名字對應哪顆。Ctrl-C 停。\n")
    for i, n in devs:
        if "blackhole" in n.lower():
            print(f"  – {i:2d} {n}（跳過：假喇叭，本來就不會出聲）")
            continue
        tone(i, n, only_ch)
    print("\n完成。")


if __name__ == "__main__":
    main()
