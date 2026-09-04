# firmware/pico_led_button/main.py - fire flow v5.1 plus the physical button over
# USB serial, merged 2026-08-05
#
# Flashing the button firmware (pico_button/main_usb.py) on 2026-08-04
# overwrote the fire flow (pico_led/main.py), which left the unit's strips dark.
# The pins never clashed (strips on GP1/GP2 with ADC26/27, button on GP15), so
# the two are merged into one file: the same Pico serves the button, with the
# TAP/HB protocol unchanged so tap_listen.py reads it directly, and both
# wire-lights. Button debouncing is now non-blocking; the original
# sleep_ms(200) froze the fire flow.
# The fire flow itself and its parameters are copied line for line from
# pico_led/main.py v5.1, settled on the 2026-07-23 bench. The only changes are
# the button and heartbeat sections inserted into the loop.
import math
import random
import time

import micropython
import neopixel
from machine import ADC, Pin

MODE = "live"        # "live" = driven by the ADC envelope (the real one),
                     # "demo" = self-running
N = (122, 122)
DATA_PINS = (1, 2)   # GP0 is dead; do not use
ADC_CH = (0, 1)
MAX_LEVEL = 0.30
WARM = (255, 120, 30)
GAMMA = 1.5
DECAY = 1.30
SPEED_MIN, SPEED_MAX = 100.0, 170.0
SPAWN_MIN, SPAWN_MAX = 0.15, 0.90
NOISE_FLOOR = 2500   # made more sensitive on 2026-08-05, 3000 to 2500. Measured
                     # noise bursts reach about 2100, leaving 400 of margin; any
                     # lower and it self-ignites, see the 2026-07-29 notes.
COUPLING = 4300

DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

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
led.on()                                  # steady on from power-up means alive

strips = [Strip(p, n) for p, n in zip(DATA_PINS, N)]
envs = [Envelope(c) for c in ADC_CH]
rnd = random.random
t_last = time.ticks_ms()
demo_t = 0.0

seq, last = 0, 1
t_hb = time.ticks_ms()
t_tap = 0                                 # non-blocking debounce: time of the last press
led_off_at = 0                            # the on-board LED goes out for 0.2 s on a press,
                                          # without blocking

while True:
    now = time.ticks_ms()
    dt = time.ticks_diff(now, t_last) / 1000.0
    t_last = now

    v = button.value()
    if last == 1 and v == 0 and time.ticks_diff(now, t_tap) > DEBOUNCE_MS:
        seq += 1
        print("TAP", seq)
        t_tap = now
        led.off()
        led_off_at = now
    last = v
    if led_off_at and time.ticks_diff(now, led_off_at) > DEBOUNCE_MS:
        led.on()
        led_off_at = 0
    if time.ticks_diff(now, t_hb) > int(HEARTBEAT_S * 1000):
        print("HB", seq)
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
