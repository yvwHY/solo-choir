# firmware/pico_led/main.py — wire-light v5.1「火流」: 能量從盒端流向發聲體
# Pico W / MicroPython v1.28。開機自跑。
# 效果＝火焰式推進（隨機濺入→定向推進→衰減），2026-07-23 bench 定稿。
# 效能結構是硬要求（Pico 純 Python 浮點只有 5.5fps）：整數熱場 bytearray
# ＋ gamma LUT ＋ @micropython.native ＋ 只推實際段長的緩衝 → 60fps@160px。
#
# Strip A: DATA=GP1, env=ADC0(GP26)=PAM-in L 分接
# Strip B: DATA=GP2, env=ADC1(GP27)=PAM-in R 分接
# GP0 於 07-23 bench 被燈條垮壓倒灌打死——永久棄用；DATA 一律串 330Ω 保護。
import math
import random
import time

import micropython
import neopixel
from machine import ADC, Pin

MODE = "live"        # "live"=ADC envelope 驅動（正式）/ "demo"=自跑火流（bring-up／拍攝 fallback）
N = (122, 122)       # 每條可定址像素數（實測 160/m；07-26 色帶尺實測兩段皆 122px ≈ 76cm）
DATA_PINS = (1, 2)   # GP0 陣亡，勿用
ADC_CH = (0, 1)      # ADC0=GP26, ADC1=GP27
MAX_LEVEL = 0.30     # 亮度上限（07-29 unit2 實測目判：0.45 太亮收斂到 0.30）
WARM = (255, 120, 30)   # 暖白
GAMMA = 1.5
DECAY = 1.30         # 熱場衰減率 /s（07-27：0.55 收聲要 8s 才暗，太拖）
SPEED_MIN, SPEED_MAX = 100.0, 170.0  # 推進速度 px/s（07-29：60 時連音的火包停在頭段像每顆音重跑，抬到 100 低 env 也持續行進）
SPAWN_MIN, SPAWN_MAX = 0.15, 0.90    # 濺入機率 /frame（env 0→1）
NOISE_FLOOR = 3000   # ADC 峰對峰低於此視為靜音（07-29 unit2：邊緣爆發 ~2100 擦 2000 舊門檻，抬到 3000；真唱歌 span 19k+ 餘裕大）
COUPLING = 4300      # 燈全亮時自己灌進 ADC 的雜訊量，依 env 等比扣除
# ── 燈條自我耦合的補償（2026-07-27）──────────────────────────────
# 實測可重現：燈全黑底噪 ~1000、燈全亮 ~5300（黑→亮→黑回到原值）。燈條 PWM 電流
# 脈衝經共地耦回 ADC。單一固定門檻會自鎖（燈亮→底噪破門檻→判定有聲→燈續亮，
# 音停也不熄）；改用高低兩段遲滯則會在中等音量振盪、反應變鈍（Harry 07-27 耳判）。
# 這裡改成「扣掉」而不是「躲開」：耦合量正比於燈亮度，而 env 就是亮度，所以
#   真實訊號 = span − COUPLING × env
# 燈暗不扣（全靈敏度），燈全亮剛好抵消它自己造的雜訊 → 門檻可維持在低點 2000，
# 不需遲滯、沒有振盪、任何 Mac 音量下都成立。
# （麵包板 Task 2 沒發作＝當時 1 條燈、1 路分接、燈條直吃低阻抗軌；07-26 為修資料
#  誤碼加的 1N4007 在供電路徑上加了阻抗，脈衝才變成電壓波動。根治仍在硬體：
#  大電解要在 1N4007 之後的燈條側，必要時加大到 470µF，屆時 COUPLING 要重量。）

_lutR = bytearray(256)
_lutG = bytearray(256)
_lutB = bytearray(256)
for _h in range(256):
    _lvl = (_h / 255.0) ** GAMMA * MAX_LEVEL
    _lutR[_h] = int(WARM[0] * _lvl + 0.5)
    _lutG[_h] = int(WARM[1] * _lvl + 0.5)
    _lutB[_h] = int(WARM[2] * _lvl + 0.5)


@micropython.native
def _step(heat, buf, decay_k, shifts, n, lutR, lutG, lutB):
    """衰減 → 推進 → 熱場經 LUT 寫入 neopixel 緩衝（GRB 傳輸序）。"""
    for i in range(n):
        heat[i] = (heat[i] * decay_k) >> 8
    for _ in range(shifts):
        i = n - 1
        while i > 1:
            heat[i] = (heat[i - 1] * 2 + heat[i - 2]) // 3
            i -= 1
    for i in range(n):
        h = heat[i]
        j = 3 * i
        buf[j] = lutG[h]
        buf[j + 1] = lutR[h]
        buf[j + 2] = lutB[h]


class Envelope:
    """峰對峰＋自適應峰值正規化；量 (max-min) → 對偏壓中點誤差免疫。快攻慢放。
    min-of-3 連續窗最小值＝尖峰濾波（07-23 bench：燈條電流脈衝經共地耦回 ADC，
    單發＋兩窗寬的爆發都擋掉；攻擊延遲 +2 幀無感）。"""

    def __init__(self, ch):
        self.adc = ADC(ch)
        self.peak = 4000.0
        self.env = 0.0
        self.s1 = 0
        self.s2 = 0

    def read(self, nsamp=80):
        lo, hi = 65535, 0
        rd = self.adc.read_u16
        for _ in range(nsamp):
            v = rd()
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        s0 = hi - lo
        span = s0
        if self.s1 < span:
            span = self.s1
        if self.s2 < span:
            span = self.s2
        self.s2 = self.s1
        self.s1 = s0
        span -= int(COUPLING * self.env)   # 扣掉燈條自己灌進來的雜訊
        if span < NOISE_FLOOR:
            span = 0
        self.peak = max(span, self.peak * 0.9995, 2000.0)
        raw = span / self.peak
        self.env = raw if raw > self.env else self.env * 0.90 + raw * 0.10
        return self.env


class Strip:
    def __init__(self, pin, n):
        # 上電先把可能殘留亂態的整卷/長段刷黑，再換實際段長的短緩衝（write 時間省一半以上）
        full = neopixel.NeoPixel(Pin(pin), 800)
        full.fill((0, 0, 0))
        full.write()
        self.np = neopixel.NeoPixel(Pin(pin), n)
        self.n = n
        self.heat = bytearray(n)
        self.adv = 0.0

    def frame(self, e, dt, rnd):
        self.adv += (SPEED_MIN + (SPEED_MAX - SPEED_MIN) * e) * dt
        shifts = int(self.adv)
        self.adv -= shifts
        if e > 0.02 and rnd() < (SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * e):
            j = int(rnd() * 3)
            v = self.heat[j] + int(90 + 165 * rnd())
            self.heat[j] = 255 if v > 255 else v
        _step(self.heat, self.np.buf, int(256 * (1.0 - DECAY * dt)),
              shifts, self.n, _lutR, _lutG, _lutB)
        self.np.write()


strips = [Strip(p, n) for p, n in zip(DATA_PINS, N)]
envs = [Envelope(c) for c in ADC_CH]
rnd = random.random
t_last = time.ticks_ms()
demo_t = 0.0

while True:
    now = time.ticks_ms()
    dt = time.ticks_diff(now, t_last) / 1000.0
    t_last = now
    if MODE == "demo":
        demo_t += dt
    for k, s in enumerate(strips):
        if MODE == "live":
            e = envs[k].read()
        else:  # 兩層不成整數比的正弦假 envelope，相位錯開
            e = 0.35 + 0.35 * math.sin(demo_t * 0.9 + k * 1.7) \
                + 0.25 * math.sin(demo_t * 2.7 + 1.3 + k * 0.8)
            e = min(1.0, max(0.0, e))
        s.frame(e, dt, rnd)
    time.sleep_ms(5)
