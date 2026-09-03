"""reverb.py — 合成 IR 的卷積殘響（08-03）

**為什麼是卷積、不是演算法殘響**：Schroeder/FDN 那類要調一組互質的梳狀延遲，
調不好的失敗模式就是**金屬感**——而金屬感正是這個專案花了三天才殺掉的東西
（07-30 §F–§I 的哨兵泛音柱）。指數衰減的高斯噪音本身就是一個最大密度的擴散
場，**天生沒有梳狀峰**，沒有可以調壞的東西。

**讓它像房間而不像效果器的，是分頻帶衰減**：真實空間裡高頻先死（空氣吸收＋
吸音材），單一 T60 的白噪音尾巴會「嘶」。這裡分三帶各給各的 T60。

三段結構照真實脈衝響應：直達聲（不在 IR 裡，走乾聲）→ predelay 靜默 → 早期
反射（稀疏離散、間隔刻意不規則，規則間隔＝flutter echo）→ 晚期擴散尾巴
（淡入，因為真實房間的回聲密度是**長**出來的，不是一開始就滿）。

零採購零新依賴：scipy 已在 venv 裡（`盤點` 原則）。
"""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
# 早期反射區間（ms）。08-04 改成**擴散簇**：原本是 10 個離散 tap（11–84ms、
# 增益 0.45），對持續音是「空間感」，對**開口的子音**卻是一串可辨的 slap echo
# ——Harry 實戴聽成「重複開口」，--reverb 0 消失＝定罪。離散重複來自 tap 可以
# 被逐一聽出；改成同區間的密集隨機微 tap（指數衰減包絡），單一反射不可辨、
# 房間的「近牆」線索仍在。
ER_SPAN_MS = (8.0, 90.0)


def _trim(ir, trim_db, fade_ms=60.0, sr=SR):
    """尾巴掉到 −trim_db 之後直接截斷（前面加淡出，免得斷面變成一聲喀噠）。

    為什麼要截：在應答式裡**回應多長，他就多久不能出聲**（硬靜音涵蓋整個播放
    期），所以尾巴不是免費的裝飾，是直接付出去的互動成本。08-03 Harry 耳測
    T60 1.8/2.2/2.5/2.8 四格「差不多」＝尾巴長度在這個情境下不是可聞旋鈕，
    那就沒有理由留著付錢。門檻用實測包絡而不是理論值——三個頻帶各有各的
    T60，低頻掉得最慢。"""
    w = int(0.02 * sr)
    env = np.array([np.sqrt(np.mean(ir[i:i + w] ** 2))
                    for i in range(0, len(ir) - w, w)])
    # 從峰值**之後**才找：IR 開頭是 predelay 的靜默，不從峰後找會截在 20ms。
    pk = int(np.argmax(env))
    thr = env[pk] * 10 ** (-trim_db / 20.0)
    below = np.flatnonzero(env[pk:] < thr)
    if not len(below):
        return ir
    end = min(len(ir), (pk + below[0] + 1) * w)
    f = max(1, int(fade_ms / 1000.0 * sr))
    out = ir[:end].copy()
    if end > f:
        out[end - f:] *= np.cos(np.linspace(0, np.pi / 2, f)) ** 2
    return out


def make_ir(t60=1.8, predelay_ms=20.0, hf_damp=0.40, lf_boost=1.20,
            er_gain=0.45, trim_db=0.0, seed=20260803, sr=SR):
    """回傳單聲道 IR（能量正規化：卷積後 rms ≈ 輸入 rms）。

    t60＝中頻衰減到 −60 dB 的秒數（1.2 小房間／1.8 一般廳／2.8 教堂）。
    hf_damp／lf_boost＝高低頻相對中頻的 T60 倍率。deterministic（固定 seed）
    ＝同一組參數永遠是同一個空間，A/B 才乾淨。"""
    rng = np.random.default_rng(seed)
    n = int(t60 * 1.3 * sr)                      # 尾巴留到 −78 dB 才截斷
    t = np.arange(n) / sr
    lo = butter(2, 500, "lowpass", fs=sr, output="sos")
    mid = butter(2, (500, 4000), "bandpass", fs=sr, output="sos")
    hi = butter(2, 4000, "highpass", fs=sr, output="sos")
    noise = rng.standard_normal(n)
    tail = np.zeros(n)
    for sos, mult in ((lo, lf_boost), (mid, 1.0), (hi, hf_damp)):
        # 每帶自己的 T60：−60 dB ＝ 10^(-3)，故指數係數 6.908/T60
        tail += sosfilt(sos, noise) * np.exp(-6.908 * t / (t60 * mult))
    # 回聲密度是長出來的：尾巴前 30ms 淡入，否則 IR 開頭是一發槍響
    ramp = np.clip(t / 0.030, 0.0, 1.0)
    tail *= ramp
    tail /= np.sqrt(np.mean(tail ** 2)) * np.sqrt(n)   # 先把尾巴能量正規化

    ir = np.zeros(int(predelay_ms / 1000.0 * sr) + n)
    pre = int(predelay_ms / 1000.0 * sr)
    ir[pre:pre + n] = tail
    a0, a1 = (int(v / 1000.0 * sr) for v in ER_SPAN_MS)
    er = rng.standard_normal(a1 - a0) * np.exp(-np.arange(a1 - a0) / (0.022 * sr))
    er *= er_gain / np.sqrt(np.sum(er ** 2))           # 簇總能量＝er_gain²
    ir[pre + a0:pre + a1] += er
    if trim_db > 0:
        ir = _trim(ir, trim_db, sr=sr)
    return ir / np.sqrt(np.sum(ir ** 2))               # 能量正規化


def wet(x, ir):
    """只回傳濕的部分（長度 len(x)+len(ir)-1）。乾聲不在裡面＝呼叫端自己決定
    送多少進去，wet 0 時整條路徑不執行＝逐 byte 等於沒有殘響。"""
    return fftconvolve(x, ir)


def _demo():
    """聽 IR 本身（衝擊測試）＋印每 100ms 的衰減，確認沒有梳狀峰與異常。"""
    import sys
    import soundfile as sf
    t60 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.8
    ir = make_ir(t60)
    print(f"IR {len(ir)/SR:.2f}s  peak {np.abs(ir).max():.4f}  "
          f"能量 {np.sum(ir**2):.4f}")
    e = [20 * np.log10(np.sqrt(np.mean(ir[i:i + SR // 10] ** 2)) + 1e-12)
         for i in range(0, len(ir) - SR // 10, SR // 2)]
    print("每 0.5s 的 rms dB:", " ".join(f"{v:.0f}" for v in e))
    sf.write(f"out/ir_t60_{t60}.wav", ir / np.abs(ir).max() * 0.9, SR,
             subtype="PCM_16")
    print(f"wrote out/ir_t60_{t60}.wav")


if __name__ == "__main__":
    _demo()


class ConvLive:
    """即時卷積殘響（均勻分割 overlap-save）——演出模式用。

    離線那支 `wet()` 是整段 fftconvolve，live 上不能用（要等整段唱完）。這裡把
    IR 切成 B 長的分割，每個 callback 只做「一次 FFT ＋ P 次頻域乘加 ＋ 一次
    IFFT」。**輸出第 k 塊只依賴第 ≤k 塊的輸入＝因果、零額外延遲**（除了 callback
    本身那一塊）——這點很重要，因為濕聲晚出來多少，聽感上就等於 predelay 變長，
    而 predelay 本來就是 20ms；真正不能延遲的是乾聲，而乾聲根本不進電腦
    （送出/回送：乾聲走直路，電腦只出濕聲，DAF 與回授都不引進來）。

    成本（B=1024、IR 0.88s ⇒ P=38）：每塊 38 次長度 1025 的複數乘加，可忽略。
    """

    def __init__(self, ir, block=1024):
        self.B = int(block)
        self.N = 2 * self.B
        P = int(np.ceil(len(ir) / self.B))
        pad = np.zeros(P * self.B)
        pad[:len(ir)] = ir
        self.H = np.stack([np.fft.rfft(
            np.concatenate([pad[p * self.B:(p + 1) * self.B],
                            np.zeros(self.B)]), self.N) for p in range(P)])
        self.X = np.zeros((P, self.N // 2 + 1), dtype=complex)   # 頻域歷史
        self.prev = np.zeros(self.B)
        self.P = P

    def __call__(self, x):
        """一塊輸入（長度 B）→ 一塊濕聲（長度 B）。"""
        if len(x) != self.B:                      # 收官那塊可能不滿
            y = np.zeros(self.B)
            y[:len(x)] = x
            x = y
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = np.fft.rfft(np.concatenate([self.prev, x]), self.N)
        self.prev = x
        acc = np.sum(self.H * self.X, axis=0)
        return np.fft.irfft(acc, self.N)[self.B:]   # overlap-save：取後半
