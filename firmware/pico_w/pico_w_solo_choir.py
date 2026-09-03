# pico_w_solo_choir.py — Solo Choir wearable CONTROL plane (NOT audio).
#
# Role: on-body button -> send a control command to the Mac engine over WiFi,
#       plus a status LED. Audio goes Mac -> PAM8403 -> DAEX, never through here.
#       The Pico W is a thin, stateless controller; all audio/harmony logic
#       stays on the Mac (see HARDWARE_ROADMAP.md section 8).
#
# NOTE: the Mac-side endpoint (http://<mac-ip>:8765/control) does NOT exist yet.
#       It would be a small listener added to app/bridge.py later that feeds the
#       received dict into set_control. This file is the firmware skeleton so the
#       structure is clear -- it is not yet a runnable end-to-end integration.

import network, time, urequests
from machine import Pin

WIFI_SSID = "your-network"
WIFI_PASS = "your-password"
MAC_URL   = "http://192.168.1.50:8765/control"   # Mac running the Solo Choir app

button = Pin(15, Pin.IN, Pin.PULL_UP)   # footswitch / momentary button -> GND
led    = Pin("LED", Pin.OUT)            # onboard LED = status


def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASS)
    while not wlan.isconnected():
        led.toggle(); time.sleep(0.2)   # blink while connecting
    led.on()
    print("wifi ok:", wlan.ifconfig()[0])


def send(cmd):
    # cmd is a control dict matching the engine's contract, e.g.
    #   {"harmonize": False}  or  {"you": False}  (perform-mode)
    try:
        r = urequests.post(MAC_URL, json=cmd); r.close()
    except Exception as e:
        print("send failed:", e)


def main():
    wifi_connect()
    harmonize_on = True
    last = button.value()
    while True:
        v = button.value()
        if last == 1 and v == 0:          # press edge (active-low)
            harmonize_on = not harmonize_on
            send({"harmonize": harmonize_on})
            led.toggle()
            time.sleep_ms(200)            # debounce
        last = v
        time.sleep_ms(10)


main()
