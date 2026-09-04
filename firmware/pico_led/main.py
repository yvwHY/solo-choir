# firmware/pico_led/main.py - wire-light v5.1, "fire flow": energy travels from
# the box towards the sounding body
# Pico W / MicroPython v1.28. Runs on boot.
# The effect is a flame-like advance: random spawn, directed advance, decay.
# Settled on the bench, 2026-07-23.
# The performance structure is a hard requirement: plain Python floats on the
# Pico manage only 5.5 fps. An integer heat field in a bytearray, a gamma LUT,
# @micropython.native and a buffer only as long as the real strip give
# 60 fps at 160 px.
#
# Strip A: DATA=GP1, env=ADC0(GP26), tapped from PAM-in L
# Strip B: DATA=GP2, env=ADC1(GP27), tapped from PAM-in R
# GP0 was killed on the 2026-07-23 bench by a back-feed when the strip's supply
# collapsed; it is abandoned for good. Every DATA line carries a 330 ohm series
# resistor.
import math
import random
import time

import micropython
import neopixel
from machine import ADC, Pin

MODE = "live"        # "live" = driven by the ADC envelope (the real one),
                     # "demo" = self-running, for bring-up and as a filming fallback
N = (122, 122)       # addressable pixels per strip; measured 160/m, and on
                     # 2026-07-26 both runs measured 122 px, about 76 cm
DATA_PINS = (1, 2)   # GP0 is dead; do not use
ADC_CH = (0, 1)      # ADC0=GP26, ADC1=GP27
MAX_LEVEL = 0.30     # brightness ceiling; judged by eye on unit 2, 2026-07-29,
                     # where 0.45 was too bright and it settled at 0.30
WARM = (255, 120, 30)   # warm white
GAMMA = 1.5
DECAY = 1.30         # heat-field decay per second; at 0.55 on 2026-07-27 it took
                     # 8 s to go dark after the sound stopped, which dragged
SPEED_MIN, SPEED_MAX = 100.0, 170.0  # advance speed in px/s; at 60 on 2026-07-29 the flame
                     # packets of a legato line stalled near the start and looked
                     # restarted on every note, so 100 keeps them moving even at
                     # a low envelope
SPAWN_MIN, SPAWN_MAX = 0.15, 0.90    # spawn probability per frame, for envelope 0 to 1
NOISE_FLOOR = 3000   # peak-to-peak ADC below this counts as silence; on unit 2,
                     # 2026-07-29, edge bursts of about 2100 grazed the old 2000
                     # threshold, so it rose to 3000. Real singing spans 19k and
                     # more, with plenty of margin
COUPLING = 4300      # the noise the strip injects into the ADC at full
                     # brightness, subtracted in proportion to the envelope
# -- compensating the strip's self-coupling (2026-07-27) ----------------------
# Reproducible measurement: the noise floor is about 1000 with the strip dark
# and about 5300 with it fully lit, and returns to the first value when it goes
# dark again. The strip's PWM current pulses couple back into the ADC through
# the shared ground. A single fixed threshold latches: the strip lights, the
# floor crosses the threshold, that reads as sound, the strip stays lit, and it
# does not go dark when the sound stops. Two-level hysteresis instead
# oscillates at middling levels and dulls the response, judged by ear on
# 2026-07-27.
# This subtracts the coupling rather than avoiding it. The coupled amount is
# proportional to the brightness, and the envelope is the brightness, so
#   real signal = span - COUPLING * envelope
# Nothing is subtracted while the strip is dark, which keeps full sensitivity,
# and at full brightness it exactly cancels the noise the strip makes, so the
# threshold can stay low at 2000, with no hysteresis, no oscillation, and it
# holds at any Mac output level.
# (It did not appear in breadboard task 2, which had one strip, one tap and the
#  strip drawing straight from a low-impedance rail. The 1N4007 added on
#  2026-07-26 to fix data corruption put impedance in the supply path, which is
#  what turned the pulses into voltage swings. The real cure is still hardware:
#  the bulk electrolytic belongs on the strip side of the 1N4007, raised to
#  470 uF if necessary, and COUPLING must then be measured again.)

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
    """Decay, advance, then write the heat field through the LUT into the
    neopixel buffer, in GRB transmission order."""
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
    """Peak-to-peak with adaptive peak normalisation. Measuring (max - min) is
    immune to bias-point error. Fast attack, slow release.
    The minimum of three consecutive windows is a spike filter: on the
    2026-07-23 bench the strip's current pulses coupled back into the ADC
    through the shared ground, and this rejects both single events and bursts
    two windows wide. The two extra frames of attack delay are imperceptible."""

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
        span -= int(COUPLING * self.env)   # subtract the noise the strip injects into itself
        if span < NOISE_FLOOR:
            span = 0
        self.peak = max(span, self.peak * 0.9995, 2000.0)
        raw = span / self.peak
        self.env = raw if raw > self.env else self.env * 0.90 + raw * 0.10
        return self.env


class Strip:
    def __init__(self, pin, n):
        # Clear the whole reel, or a long run, at power-up in case it holds a
        # random state, then switch to a buffer the length of the real strip,
        # which more than halves the write time.
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
        else:  # a fake envelope from two sines in a non-integer ratio, out of phase
            e = 0.35 + 0.35 * math.sin(demo_t * 0.9 + k * 1.7) \
                + 0.25 * math.sin(demo_t * 2.7 + 1.3 + k * 0.8)
            e = min(1.0, max(0.0, e))
        s.frame(e, dt, rnd)
    time.sleep_ms(5)
