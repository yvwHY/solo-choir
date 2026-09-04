# main.py - Solo Choir physical button ("your turn" key for the answering mode)
#
# Role: one momentary button on the body, broadcast to the Mac over UDP. This is
# the control plane and never touches audio.
#
# Why it exists (2026-08-01): phrase-end detection in the answering mode v2 is
# energy-only, which has two weaknesses.
#   1. It has to wait a full 0.35 s of silence to know the singer has stopped,
#      which is a floor on the latency.
#   2. The choir returns to the microphone through the body and the air; the
#      measured p90 of 0.016 is already close to the gate at 0.02, leaving about
#      2 dB of margin. Any louder in the room and it reads as "still singing",
#      so the response never triggers.
# A physical button is immune to both, at about 6 ms of latency (pinged on the
# OLED line). Energy detection stays as the fallback.
#
# Hardware: one leg of the button to GP15, the other to GND, with the internal
# pull-up, so pressed reads 0. No other parts are needed.
#   (Same wiring as firmware/pico_w/pico_w_solo_choir.py, which is the HTTP
#    skeleton. This file uses UDP instead: no TCP handshake, non-blocking, and it
#    implements the dual-mode boot left open in STATE.)
#
# Network: try STA first, against a router such as the Slate AX in the
#   equipment room; if that fails, raise an access point `SoloChoir` at
#   192.168.4.1, the configuration verified on the OLED line on 2026-07-17, and
#   the Mac joins it. Both modes broadcast over UDP, so the Mac never needs the
#   other side's IP and a change of venue needs no reconfiguration.
#
# Reliability: UDP drops packets, so each press sends three datagrams 10 ms
#   apart carrying a rising sequence number, which the Mac de-duplicates. A
#   heartbeat every 2 s lets the Mac show the connection state, which is checked
#   before going on stage.

import network
import socket
import time

from machine import Pin

STA_SSID, STA_PASS = "", ""          # STA is tried only when filled in; empty means go straight to AP
AP_SSID, AP_PASS = "SoloChoir", "solochoir"   # AP mode: the Mac joins this network
PORT = 8766
DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

button = Pin(15, Pin.IN, Pin.PULL_UP)   # pressed = grounded = 0
led = Pin("LED", Pin.OUT)


def net_up():
    """Dual mode: use STA if it connects, otherwise raise an AP. Returns the
    broadcast address."""
    if STA_SSID:
        w = network.WLAN(network.STA_IF)
        w.active(True)
        w.connect(STA_SSID, STA_PASS)
        for _ in range(60):             # wait at most 6 s
            if w.isconnected():
                ip, _, _, _ = w.ifconfig()
                print("STA ok:", ip)
                led.on()
                return ip.rsplit(".", 1)[0] + ".255"
            led.toggle()
            time.sleep(0.1)
        w.active(False)
        print("STA failed, falling back to AP")
    ap = network.WLAN(network.AP_IF)
    ap.config(essid=AP_SSID, password=AP_PASS)
    ap.active(True)
    while not ap.active():
        time.sleep(0.1)
    print("AP ok:", ap.ifconfig()[0], "(join", AP_SSID, "on the Mac)")
    led.on()
    return "192.168.4.255"


def main():
    bcast = net_up()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    seq, last, t_hb = 0, 1, time.ticks_ms()
    while True:
        v = button.value()
        if last == 1 and v == 0:                  # falling edge, active low
            seq += 1
            for _ in range(3):                    # three redundant datagrams against packet loss
                s.sendto(b"TAP %d" % seq, (bcast, PORT))
                time.sleep_ms(10)
            led.off()
            time.sleep_ms(DEBOUNCE_MS)
            led.on()
        last = v
        if time.ticks_diff(time.ticks_ms(), t_hb) > HEARTBEAT_S * 1000:
            s.sendto(b"HB %d" % seq, (bcast, PORT))
            t_hb = time.ticks_ms()
        time.sleep_ms(5)


main()
