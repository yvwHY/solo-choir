# main_usb.py - the USB serial version of the button, the wired route for the
# exhibition, decided 2026-08-04
#
# 2026-08-04: the button is mounted on the K669B, a two-core cable runs to a
# Pico beside the Mac, and one USB cable carries both power and communication.
# WiFi leaves entirely, taking packet loss, the access point and heartbeat
# timeouts with it. The WiFi version (main.py) is kept for the worn mode, where
# no cable can be run and it is still the right answer.
#
# The protocol is the same as the UDP version, one message per line, TAP n and
# HB n, and SerialTapListener in tap_listen.py on the Mac reads the printed
# output directly. Serial is reliable, so the three redundant sends are not
# needed.
#
# Wiring as in main.py: the two diagonal legs of the button to GP15 and GND,
# internal pull-up, pressed reads 0.

import time

from machine import Pin

DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

button = Pin(15, Pin.IN, Pin.PULL_UP)
led = Pin("LED", Pin.OUT)
led.on()                                  # steady on from power-up means alive

seq, last = 0, 1
t_hb = time.ticks_ms()
while True:
    v = button.value()
    if last == 1 and v == 0:              # falling edge, active low
        seq += 1
        print("TAP", seq)
        led.off()                         # a brief blink on press, as visual feedback
        time.sleep_ms(DEBOUNCE_MS)
        led.on()
    last = v
    if time.ticks_diff(time.ticks_ms(), t_hb) > int(HEARTBEAT_S * 1000):
        print("HB", seq)
        t_hb = time.ticks_ms()
    time.sleep_ms(5)
