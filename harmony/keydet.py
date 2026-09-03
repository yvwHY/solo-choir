"""keydet.py — 調性偵測（任務6：走出 C 大調）

滑動窗音高直方圖 × Krumhansl-Schmuckler 大調輪廓相關 → 24 調（12 大調×
major/minor profile，本專案腦只懂大調系統，偵測輸出＝大調 key root）。
用途：
  離線  python keydet.py <wav...>              # 每檔報 key＋信心
  模組  from keydet import KeyDetector          # live 用：push(f0_hz) → key

設計：
- 輸入只吃 f0（YIN/遙測都有），不吃頻譜——與現有耳朵零耦合。
- 滑動窗預設 8s（live 可信起點）＋全曲累積版；信心＝最佳/次佳相關差。
- 輸出 root 半音數（C=0），供「輸入歸一到 C 餵腦、輸出轉回」用。
"""
import math, sys
import numpy as np

# Krumhansl-Kessler major/minor key profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


SCALE = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)  # 大調音階模板


def key_from_hist(hist):
    """12-bin 音高類直方圖 → (root, is_major, confidence)。

    判準＝音階隸屬度（直方圖質量落在哪個大調音階集合內最多），不是 K-S
    主音輪廓——腦要的是「哪個大調音階裝得下這段歌」（C 大調＝A 小調同一組
    音，主音之爭與歸一無關；首測 K-S 把 A 中心的 C 大調旋律判成 A）。
    is_major 僅為相容欄位，恆 True；conf＝最佳/次佳隸屬度差（0–1）。"""
    tot = hist.sum()
    if tot <= 0:
        return None
    member = np.array([np.dot(np.roll(SCALE, r), hist) for r in range(12)]) / tot
    order = np.argsort(member)[::-1]
    r1, r2 = int(order[0]), int(order[1])
    return r1, True, float(member[r1] - member[r2])


class KeyDetector:
    """live 累積器：push 每幀 f0（Hz 或 None），read key()。"""

    def __init__(self, win_frames=None):
        self.hist = np.zeros(12)
        self.win = win_frames          # None = 無限累積
        self.buf = []

    def push(self, f0):
        if not f0:
            return
        pc = int(round(69 + 12 * math.log2(f0 / 440.0))) % 12
        self.hist[pc] += 1
        if self.win:
            self.buf.append(pc)
            if len(self.buf) > self.win:
                self.hist[self.buf.pop(0)] -= 1

    def key(self):
        r = key_from_hist(self.hist)
        if r is None:
            return None
        root, is_maj, conf = r
        return {"root": root, "major": is_maj, "conf": conf,
                "name": NAMES[root] + ("" if is_maj else "m")}


def track_file(path):
    import soundfile as sf, soxr
    from pitch import yin_f0
    SR = 44100
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    kd = KeyDetector()
    for t in range(0, len(x) - 2048, 1024):
        kd.push(yin_f0(x[t:t + 2048].astype(np.float32), SR))
    return kd.key()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        k = track_file(p)
        print(f"{p}: {k['name']}  conf={k['conf']:.3f}" if k else f"{p}: no pitch")
