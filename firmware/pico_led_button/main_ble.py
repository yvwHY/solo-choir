# firmware/pico_led_button/main_ble.py — 火流 v5.1 ＋ 實體鍵 **BLE 通知**（08-18）
#
# 08-18 Harry 裁「那就藍牙」。原計畫＝BLE HID 鍵盤，**死因：官方 rp2 韌體沒
# 編入配對/bond（ble.config(bond=…) → ValueError: unknown config param），而
# macOS 對 HID 強制加密配對** → 已入 GRAVEYARD。改走免配對路線：自訂 GATT
# 服務＋通知——按下＝notify "TAP <seq>"，Mac 端由 respond_shell 內建的 bleak
# 接收執行緒收下、走既有 tap 路徑（等同按 Space）。無線、Mac 不斷網、免配對。
# 火流本體與參數逐行照抄 main.py（08-05 合併版）不動；USB serial 的 TAP/HB
# 照印＝插線即備援（tap_listen 直接吃）。
#
# 板載 LED 語意：**恆亮＝BLE 已連上（app 在收）；慢閃(1Hz)＝在廣播等連線**；
# 按下熄 0.2s 不變。回 USB serial 版：mpremote fs cp main.py :main.py
import math
import random
import time

import bluetooth
import micropython
import neopixel
from machine import ADC, Pin

MODE = "live"        # "live"=ADC envelope 驅動（正式）/ "demo"=自跑火流
N = (122, 122)
DATA_PINS = (1, 2)   # GP0 陣亡，勿用
ADC_CH = (0, 1)
MAX_LEVEL = 0.30
WARM = (255, 120, 30)
GAMMA = 1.5
DECAY = 1.30
SPEED_MIN, SPEED_MAX = 100.0, 170.0
SPAWN_MIN, SPAWN_MAX = 0.15, 0.90
NOISE_FLOOR = 2500   # 08-05 Harry 要更靈敏：3000→2500（實測雜訊爆發 ~2100
                     # ＝仍有 400 餘裕；再低就會自燃，見 07-29 紀錄）
COUPLING = 4300
DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

# ---------------- BLE 通知按鍵（IRQ 驅動，免配對，不佔主迴圈） ----------------
_IRQ_CENTRAL_CONNECT = 1
_IRQ_CENTRAL_DISCONNECT = 2

# 自訂 128-bit UUID（"SOLOCH" 嵌在字串裡，Mac 端 respond_shell 以此掃描）
_SVC_UUID = bluetooth.UUID("0f9a0001-1e0e-4c7a-9a4e-534f4c4f4348")
_TAP_UUID = bluetooth.UUID("0f9a0002-1e0e-4c7a-9a4e-534f4c4f4348")


class BleKey:
    def __init__(self, name="SoloChoir Key"):
        self.conn = None
        b = bluetooth.BLE()
        self.b = b
        b.irq(self._irq)
        b.active(True)
        b.config(gap_name=name)
        ((self.h_tap,),) = b.gatts_register_services(
            ((_SVC_UUID, ((_TAP_UUID,
                           bluetooth.FLAG_READ | bluetooth.FLAG_NOTIFY),)),))
        b.gatts_write(self.h_tap, b"HB 0")
        # 128-bit UUID 佔 18B ＋ flags 3B ＝ 廣播包放不下名字 → 名字放
        # scan response（bleak 兩包都看得到）
        self._adv = b"\x02\x01\x06" + bytes((17, 0x07)) + bytes(_SVC_UUID)
        nm = name.encode()
        self._resp = bytes((len(nm) + 1, 0x09)) + nm
        self._advertise()

    def _advertise(self):
        try:
            self.b.gap_advertise(100_000, adv_data=self._adv,
                                 resp_data=self._resp)
            print("[ble] advertising")
        except OSError as e:
            print("[ble] adv fail", e)

    def _irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            self.conn, _, _ = data
            print("[ble] connected")
        elif event == _IRQ_CENTRAL_DISCONNECT:
            self.conn = None
            print("[ble] disconnected")
            self._advertise()

    def send(self, line):
        """按鍵/心跳走同一條：notify 一行文字（協定同 serial 的 TAP/HB）。"""
        if self.conn is None:
            return False
        try:
            self.b.gatts_notify(self.conn, self.h_tap, line)
            return True
        except OSError as e:
            print("[ble] notify fail", e)
            return False


# ---------------- 以下火流本體＝main.py 逐行照抄（勿動） ----------------
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
        span -= int(COUPLING * self.env)
        if span < NOISE_FLOOR:
            span = 0
        self.peak = max(span, self.peak * 0.9995, 2000.0)
        raw = span / self.peak
        self.env = raw if raw > self.env else self.env * 0.90 + raw * 0.10
        return self.env


class Strip:
    def __init__(self, pin, n):
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


button = Pin(15, Pin.IN, Pin.PULL_UP)
led = Pin("LED", Pin.OUT)
ble_key = BleKey()

strips = [Strip(p, n) for p, n in zip(DATA_PINS, N)]
envs = [Envelope(c) for c in ADC_CH]
rnd = random.random
t_last = time.ticks_ms()
demo_t = 0.0

seq, last = 0, 1
t_hb = time.ticks_ms()
t_tap = 0                                 # 非阻塞去彈跳：上次按下時刻
led_off_at = 0                            # 板載燈按下熄 0.2s（非阻塞）
t_blink = time.ticks_ms()
blink_on = True

while True:
    now = time.ticks_ms()
    dt = time.ticks_diff(now, t_last) / 1000.0
    t_last = now

    v = button.value()
    if last == 1 and v == 0 and time.ticks_diff(now, t_tap) > DEBOUNCE_MS:
        seq += 1
        print("TAP", seq)                 # USB 備援：插線＝serial 路照走
        ble_key.send(b"TAP %d" % seq)
        t_tap = now
        led.off()
        led_off_at = now
    last = v
    # LED：恆亮＝BLE 已連上；慢閃(1Hz)＝在廣播等連線；按下熄 0.2s 優先
    if led_off_at:
        if time.ticks_diff(now, led_off_at) > DEBOUNCE_MS:
            led_off_at = 0
    elif ble_key.conn is not None:
        led.on()
    else:
        if time.ticks_diff(now, t_blink) > 500:
            t_blink = now
            blink_on = not blink_on
        led.value(1 if blink_on else 0)
    if time.ticks_diff(now, t_hb) > int(HEARTBEAT_S * 1000):
        print("HB", seq)
        ble_key.send(b"HB %d" % seq)
        t_hb = now

    if MODE == "demo":
        demo_t += dt
    for k, s in enumerate(strips):
        if MODE == "live":
            e = envs[k].read()
        else:
            e = 0.35 + 0.35 * math.sin(demo_t * 0.9 + k * 1.7) \
                + 0.25 * math.sin(demo_t * 2.7 + 1.3 + k * 0.8)
            e = min(1.0, max(0.0, e))
        s.frame(e, dt, rnd)
    time.sleep_ms(5)
