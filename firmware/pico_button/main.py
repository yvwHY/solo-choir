# main.py — Solo Choir 實體鍵（應答式「換你」鍵）韌體
#
# 角色：身上一顆 momentary 按鈕 → UDP 廣播給 Mac。控制平面，不碰音訊。
#
# 為什麼要它（08-01）：應答式 v2 的句尾偵測是純能量判斷，兩個弱點——
#   ① 必須等滿 0.35s 靜默才知道你停了（延遲的下限）
#   ② 天使聲音經身體/空氣回到麥克風，實測 p90 0.016 已逼近 gate 0.02（只剩
#      2dB 餘裕）；展場音量再大就會誤判成「他還在唱」，回應永遠不觸發
# 實體鍵對兩者免疫，且延遲 ~6ms（OLED 線實測 ping）。能量偵測留作 fallback。
#
# 硬體：按鈕一腳 GP15、另一腳 GND（內部上拉，按下＝0）。無需其他零件。
#   （沿用 firmware/pico_w/pico_w_solo_choir.py 的接法；該檔是 HTTP 版骨架，
#    本檔改 UDP＝免 TCP 握手、不阻塞，並實作 STATE 掛著的「雙模開機」待辦。）
#
# 網路：先試 STA（器材室 Slate AX 之類的路由器），連不上就自建 AP
#   `SoloChoir` 192.168.4.1（＝ OLED 線 07-17 驗過的組態，Mac 加入該 AP）。
#   兩種模式都走 UDP 廣播 → Mac 端不必知道對方 IP、換場地免改設定。
#
# 可靠度：UDP 會掉包 → 每次按下連送 3 封（間隔 10ms）帶遞增序號，Mac 端以
#   序號去重。另每 2s 送一次 heartbeat＝Mac 端可顯示連線狀態（上台前確認）。

import network
import socket
import time

from machine import Pin

STA_SSID, STA_PASS = "", ""          # 填了才會試 STA；留空＝直接開 AP
AP_SSID, AP_PASS = "SoloChoir", "solochoir"   # AP 模式（Mac 加入這個網路）
PORT = 8766
DEBOUNCE_MS = 200
HEARTBEAT_S = 2.0

button = Pin(15, Pin.IN, Pin.PULL_UP)   # 按下＝接地＝0
led = Pin("LED", Pin.OUT)


def net_up():
    """雙模：STA 連得上就用，否則自建 AP。回傳廣播位址。"""
    if STA_SSID:
        w = network.WLAN(network.STA_IF)
        w.active(True)
        w.connect(STA_SSID, STA_PASS)
        for _ in range(60):             # 最多等 6s
            if w.isconnected():
                ip, _, _, _ = w.ifconfig()
                print("STA ok:", ip)
                led.on()
                return ip.rsplit(".", 1)[0] + ".255"
            led.toggle()
            time.sleep(0.1)
        w.active(False)
        print("STA 失敗 → 退回 AP")
    ap = network.WLAN(network.AP_IF)
    ap.config(essid=AP_SSID, password=AP_PASS)
    ap.active(True)
    while not ap.active():
        time.sleep(0.1)
    print("AP ok:", ap.ifconfig()[0], "(Mac 請加入", AP_SSID, ")")
    led.on()
    return "192.168.4.255"


def main():
    bcast = net_up()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    seq, last, t_hb = 0, 1, time.ticks_ms()
    while True:
        v = button.value()
        if last == 1 and v == 0:                  # 按下緣（active-low）
            seq += 1
            for _ in range(3):                    # 冗餘送 3 封抗掉包
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
