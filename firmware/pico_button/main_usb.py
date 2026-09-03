# main_usb.py — 實體鍵 USB serial 版（08-04 展場有線路線）
#
# 08-04 決定：展場按鈕貼 K669B、兩芯長線拉到 Mac 旁的 Pico，USB 一條線
# 兼供電＋通訊 → WiFi 整個退場（掉包/AP/heartbeat 逾時全部消失）。
# WiFi 版（main.py）保留給穿戴模式——身上沒線可拉，它仍是對的。
#
# 協定與 UDP 版相同（TAP n / HB n 一行一則），Mac 端 tap_listen.py 的
# SerialTapListener 直接吃 print 輸出。serial 是可靠傳輸＝不需冗餘 3 封。
#
# 接線同 main.py：按鈕對角兩腳 → GP15＋GND，內部上拉，按下＝0。

import time

from machine import Pin

DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

button = Pin(15, Pin.IN, Pin.PULL_UP)
led = Pin("LED", Pin.OUT)
led.on()                                  # 上電恆亮＝活著

seq, last = 0, 1
t_hb = time.ticks_ms()
while True:
    v = button.value()
    if last == 1 and v == 0:              # 按下緣（active-low）
        seq += 1
        print("TAP", seq)
        led.off()                         # 按下瞬間熄一下＝視覺回饋
        time.sleep_ms(DEBOUNCE_MS)
        led.on()
    last = v
    if time.ticks_diff(time.ticks_ms(), t_hb) > int(HEARTBEAT_S * 1000):
        print("HB", seq)
        t_hb = time.ticks_ms()
    time.sleep_ms(5)
