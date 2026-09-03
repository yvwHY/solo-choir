"""mic_check.py — 開唱前 30 秒的收音體檢（08-08）

為什麼有這支：diag8 那一唱的 SNR 只有 29dB（0802 同一支 USB 麥是 42dB），
三張嘴把麥克風的雜訊底一起唱出來＝「氣音竊竊私語」。當時沒人知道底片已經
壞了，於是整輪耳測與量測都建在壞素材上。這支就是不讓那件事再發生一次。

輸入位準沒有任何軟體增益（spike_stream6.py:486 直接吃 indata），所以唯一
的旋鈕是 macOS 的輸入音量（系統設定→聲音→輸入）。

Run (DDSP venv):  python mic_check.py            # 預設 12 秒
                  python mic_check.py --sec 20
流程：前 3 秒安靜（量雜訊底）→ 之後正常唱一句。
"""
import argparse
import numpy as np
import sounddevice as sd

SR = 44100
HOP = 512
QUIET_S = 3.0                 # 開頭安靜段長度
TRAIN_REF_RMS = 0.0566        # direct_mouth.TRAIN_REF_RMS（模型訓練參考位準）
SNR_GOOD = 40.0               # 0802 那捲好底片 = 42dB
SNR_MIN = 35.0


def db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--sec", type=float, default=12.0)
    ap.add_argument("--file", default="",
                    help="改判讀既有 wav（不錄音）＝可離線驗算這支工具本身")
    a = ap.parse_args()

    if a.file:
        import soundfile as sf
        x, _ = sf.read(a.file)
        if x.ndim > 1:
            x = x.mean(1)
        x = x.astype(np.float64)
        print(f"判讀 {a.file}（{len(x) / SR:.1f}s）")
    else:
        print(f"錄音 {a.sec:.0f} 秒 — 前 {QUIET_S:.0f} 秒請安靜，之後正常唱一句")
        buf = sd.rec(int(a.sec * SR), samplerate=SR, channels=1,
                     dtype="float32", device=a.in_name)
        for s in range(int(a.sec), 0, -1):
            sd.sleep(1000)
            print(f"  {s}…", end="", flush=True)
        sd.wait()
        print()
        x = buf[:, 0].astype(np.float64)
    n = len(x) // HOP
    r = np.sqrt((x[:n * HOP].reshape(n, HOP) ** 2).mean(1))
    nq = int(QUIET_S * SR / HOP)

    if a.file:
        # 既有檔沒有約定的安靜段：用全檔分位數（與 08-08 診斷同一把尺）
        floor, voice = np.percentile(r, 10), np.percentile(r, 90)
    else:
        floor = np.percentile(r[:nq], 90)    # 安靜段的上緣＝真正的底
        voice = np.percentile(r[nq:], 90)    # 唱歌段的 p90
    snr = db(voice) - db(floor)
    peak = np.abs(x[nq * HOP:]).max()

    print(f"\n  雜訊底 {db(floor):6.1f} dBFS")
    print(f"  歌聲   {db(voice):6.1f} dBFS   (峰值 {db(peak):.1f})")
    print(f"  SNR    {snr:6.1f} dB")
    print(f"  vs 訓練參考 {db(TRAIN_REF_RMS):.1f} dBFS："
          f"{db(voice) - db(TRAIN_REF_RMS):+.1f} dB")

    print()
    if peak > 0.95:
        print("  ✗ 削峰了 — 把輸入音量調低")
    elif snr < SNR_MIN:
        print(f"  ✗ SNR {snr:.0f}dB 太低（diag8 那捲壞底片就是 29dB）。"
              "把 macOS 輸入音量調高再測一次；\n"
              "    若調到底仍不足，換麥距／線材／USB 埠再試。")
    elif snr < SNR_GOOD:
        print(f"  △ SNR {snr:.0f}dB 堪用，但不如 0802 那捲（42dB）。"
              "還有調高的空間就調。")
    else:
        print(f"  ✓ SNR {snr:.0f}dB — 可以開唱")

    if abs(db(voice) - db(TRAIN_REF_RMS)) > 6 and snr >= SNR_MIN:
        print("  △ 歌聲位準離訓練分布 >6dB（模型在分布外會退化）")


if __name__ == "__main__":
    main()
