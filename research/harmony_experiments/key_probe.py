"""key_probe.py — 真歌定調：raw f0 音級分布（08-01 §H 的儀器，固化）

為什麼要有這支：`--key` 必須是手動整數，而「手動」不等於「用猜的」。
08-01 判 pair06 是 C 大調判了一整晚都不對，根因是**儀器盲點**——腦的表徵
把音吸附進 C 音階，於是「lead 100% 在 C 調內」是循環論證。改看**未經吸附
的 raw f0**，音級分布立刻指向 A/E 家族，A 大調確立。當晚這支是一次性分析
沒進 repo（同 live_probe 的儀器教訓），今天補上。

判準（照 08-01 實際用的兩件）：
  1. **涵蓋率**：每個大調的 7 個音級佔了多少發聲時間（有聲 frame 加權）。
  2. **終止音**：樂句尾音的音級分布——調性的終止感比統計量更硬。
兩者不一致就別選歌了，那是「這首在任何單一大調內都髒」的訊號（pair06 涵蓋
僅 ~70%，自由唱素材的典型）。

輸出的 shift 直接就是 `--key` 要填的數（與 EarV3._keylock 同慣例：
shift = (0 - root) % 12，>6 減 12，讓天使的譜落回 C 大調座標）。

用法（DDSP venv，於 harmony/ 下）:
    python key_probe.py <take.wav>
    python key_probe.py <take.wav> --hop 512     # 更密（慢）
"""
import argparse

import numpy as np
import soundfile as sf

from pitch import FRAME, SR, hz_to_midi, yin_f0
from respond2 import find_phrases

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR = [0, 2, 4, 5, 7, 9, 11]          # 大調音級（相對主音）


def raw_pitch_classes(x, hop):
    """未吸附的 f0 → 每 frame 的 (樣本位置, 音級)。刻意不經 VoiceToTokens／
    腦的任何表徵：那些會把音吸附進音階，量出來的東西就只是自己的假設。"""
    out = []
    for i in range(0, len(x) - FRAME, hop):
        f = yin_f0(x[i:i + FRAME])
        if f is not None:
            out.append((i, int(round(hz_to_midi(f))) % 12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--tail", type=float, default=0.30,
                    help="樂句尾音取最後幾秒（終止感）")
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--gate", type=float, default=0.02)
    a = ap.parse_args()

    x, sr = sf.read(a.wav, dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    assert sr == SR, (a.wav, sr)

    pcs = raw_pitch_classes(x, a.hop)
    if not pcs:
        raise SystemExit("沒有有聲 frame（檔案是靜音？取樣率不對？）")
    hist = np.bincount([p for _, p in pcs], minlength=12).astype(float)
    hist /= hist.sum()
    print(f"{a.wav}｜{len(x)/SR:.1f}s，有聲 frame {len(pcs)}")
    print("raw f0 音級分布（未吸附）:")
    for i in np.argsort(hist)[::-1][:6]:
        print(f"  {NAMES[i]:2s} {hist[i]*100:5.1f}%")

    import respond2
    respond2.RMS_GATE = a.gate
    ph = find_phrases(x, a.gap, 0.5)
    ends = []
    for s, e in ph:
        seg = [p for i, p in pcs if e - a.tail * SR <= i < e]
        if seg:
            ends.append(np.bincount(seg, minlength=12).argmax())
    eh = np.bincount(ends, minlength=12).astype(float) if ends else np.zeros(12)

    print(f"\n樂句 {len(ph)} 句，尾音（最後 {a.tail}s）音級:")
    for i in np.argsort(eh)[::-1][:4]:
        if eh[i]:
            print(f"  {NAMES[i]:2s} {int(eh[i])} 句")

    print("\n候選大調（涵蓋率＝該調 7 個音級佔的發聲時間）:")
    rows = []
    for root in range(12):
        cov = sum(hist[(root + d) % 12] for d in MAJOR)
        fin = sum(eh[(root + d) % 12] for d in MAJOR) / max(1, len(ends))
        shift = (0 - root) % 12
        shift = shift - 12 if shift > 6 else shift
        rows.append((cov, root, shift, fin, eh[root]))
    rows.sort(reverse=True)
    for cov, root, shift, fin, tonic_ends in rows[:4]:
        print(f"  {NAMES[root]:2s} 大調  涵蓋 {cov*100:5.1f}%  "
              f"尾音在調內 {fin*100:5.0f}%  結在主音 {int(tonic_ends)} 句  "
              f"→ --key {shift:+d}")

    cov, root, shift, _, _ = rows[0]
    print(f"\n判：**{NAMES[root]} 大調 → --key {shift:+d}**")
    if cov < 0.85:
        print(f"⚠ 涵蓋僅 {cov*100:.0f}%：這首在單一大調內就髒（pair06 ~70% 是"
              f"「跑通管線用、不能拿來裁決品質」的等級）。選歌時這是扣分項。")
    if rows[0][0] - rows[1][0] < 0.03:
        print(f"⚠ 第一與第二名只差 {(rows[0][0]-rows[1][0])*100:.1f}%＝分不開，"
              f"別靠這支硬選；看尾音那欄，或換一首調性乾淨的。")


if __name__ == "__main__":
    main()
