"""tap_listen.py — Mac 端收實體鍵（`firmware/pico_button`）的 UDP 訊號

應答式 v2 的「換你」鍵接收端。用法：

    tap = TapListener()          # 背景執行緒，非阻塞
    ...
    if tap.take():               # 有新的一下（消費掉）→ 立刻斷句
        ...
    tap.alive                    # heartbeat 在不在（上台前確認連線）

去重：韌體每按一次冗餘送 3 封帶同序號 → 以序號去重（UDP 掉包容忍）。
不擋路原則：收不到就當沒這顆鍵，能量偵測照常運作（08-01 設計：按鈕負責
「結束」這個有歧義的一半，開始仍由能量判定；按鈕沒按到系統不會卡死）。

自測（不需硬體）：
    python tap_listen.py --selftest     # 自己送自己收，量往返延遲
監看（接上 Pico 後確認整條鏈）：
    python tap_listen.py --monitor              # WiFi/UDP 版（穿戴模式）
    python tap_listen.py --monitor --serial     # USB serial 版（展場有線，08-04）
"""
import argparse
import socket
import threading
import time

PORT = 8766


class TapListener:
    def __init__(self, port=PORT, hb_timeout=6.0):
        self.port = port
        self.hb_timeout = hb_timeout
        self._seqs = {}          # 已見過的最大序號，**按送端各記各的**（去重）
        self._pending = 0        # 尚未被消費的按鍵數
        self._t_last = 0.0       # 最後一次收到任何封包
        self._proto = False      # 收過一行合法協定（TAP/HB）＝對面真的是按鈕
        self.dead = False        # 讀取執行緒已結束（serial 拔線）＝該退回 UDP
        self._lock = threading.Lock()
        self.err = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("", port))
        except OSError as e:     # 埠被佔用＝不擋路，退化成「沒有這顆鍵」
            self.err = e
            return
        self._sock.settimeout(0.5)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                data, addr = self._sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                return
            self._ingest(data, addr[0])

    def _ingest(self, data, src="serial"):
        parts = data.split()
        if not parts:
            return
        with self._lock:
            self._t_last = time.time()
            if parts[0] in (b"TAP", b"HB"):
                self._proto = True           # 認得的協定行＝對面是按鈕（見 wait_proto）
            if parts[0] == b"TAP" and len(parts) > 1:
                try:
                    seq = int(parts[1])
                except ValueError:
                    return
                # 冗餘 3 封只算一次＝序號要往前。但序號**明顯回落**＝送端重置
                # （Pico 重開機／app 重啟，兩者都從 0 起算）：舊碼的單調比大小
                # 會讓那顆鍵整場失聰，而 HB 照收 ⇒ alive 仍是 True、gap 仍是
                # 6s ⇒ 錯得毫無徵兆（08-12 審查 S3/A4）。回落就接受並重新同步。
                # 08-18 Harry live「實體鍵沒反應、HB 卻活著」：UDP 埠上有**兩個
                # 送端**（app 的 space＝127.0.0.1、WiFi 按鈕＝區網 IP），各自
                # 一套流水號——混記一個 _seq 時，兩邊號碼落在 30 窗內就會把
                # 落後的一方整批當重複吞掉。去重改成**按送端各記各的**；
                # serial 版單一送端（src="serial"）行為不變。
                last = self._seqs.get(src, 0)
                if seq > last or seq < last - 30:
                    self._seqs[src] = seq
                    self._pending += 1

    @property
    def alive(self):
        """heartbeat 還在＝Pico 連著（上台前的連線指示）。"""
        with self._lock:
            return (time.time() - self._t_last) < self.hb_timeout

    def take(self):
        """有沒有新的一下（消費掉）。多下只算一下＝不會塞積。"""
        with self._lock:
            if self._pending:
                self._pending = 0
                return True
        return False

    def wait_proto(self, timeout):
        """限時等一行合法協定（TAP/HB）。開得起來 ≠ 是那顆按鈕（serial 只認
        /dev/cu.usbmodem*），沒等到就該關掉退回別的來源（respond2 live）。"""
        t0 = time.time()
        while True:
            with self._lock:
                if self._proto:
                    return True
            if time.time() - t0 >= timeout:
                return False
            time.sleep(0.05)

    def close(self):
        """釋放埠／裝置（不擋路：關不掉就算了）。"""
        try:
            self._sock.close()
        except Exception:        # noqa: BLE001 — 收尾不能再拋
            pass


class SerialTapListener(TapListener):
    """USB serial 版（08-04 展場有線路線：按鈕→長線→Mac 旁 Pico→USB）。

    介面與 TapListener 完全相同（take()/alive/err/不擋路）；差別只在
    傳輸層——讀 `firmware/pico_button/main_usb.py` 的序列輸出，協定同
    （TAP n / HB n）。dev=None 時自動抓第一個 /dev/cu.usbmodem*。
    """

    def __init__(self, dev=None, hb_timeout=6.0):
        import glob

        import serial
        self.hb_timeout = hb_timeout
        self._seqs = {}
        self._pending = 0
        self._t_last = 0.0
        self._proto = False
        self.dead = False
        self._lock = threading.Lock()
        self.err = None
        self.dev = None
        if dev is None:
            hits = sorted(glob.glob("/dev/cu.usbmodem*"))
            if not hits:                 # 沒插＝不擋路，退化成「沒有這顆鍵」
                self.err = FileNotFoundError("no /dev/cu.usbmodem*")
                return
            dev = hits[0]
        try:
            self._ser = serial.Serial(dev, 115200, timeout=0.5)
        except (OSError, serial.SerialException) as e:
            self.err = e
            return
        self.dev = dev
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                line = self._ser.readline()
            except Exception:
                self.dead = True         # 拔線＝呼叫端退回 UDP（不擋路）
                return
            if line:
                self._ingest(line)

    def close(self):
        try:
            self._ser.close()
        except Exception:        # noqa: BLE001 — 沒開成功／已關掉
            pass


def _send(port, seq, n=3):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    for _ in range(n):
        s.sendto(b"TAP %d" % seq, ("127.0.0.1", port))
        time.sleep(0.01)
    s.close()


def selftest(port=PORT):
    """不需硬體：自己送自己收，驗去重＋量本機延遲。"""
    tap = TapListener(port)
    if tap.err:
        raise SystemExit(f"bind {port} 失敗: {tap.err}")
    time.sleep(0.2)
    lat = []
    for seq in (1, 2, 3):
        t0 = time.perf_counter()
        # 背景送：冗餘封包間的 10ms 睡眠不能算進延遲（量的是第一封到達）
        threading.Thread(target=_send, args=(port, seq), daemon=True).start()
        while not tap.take():
            if time.perf_counter() - t0 > 2:
                raise SystemExit(f"seq {seq} 沒收到")
            time.sleep(0.001)
        lat.append((time.perf_counter() - t0) * 1000)
    assert not tap.take(), "去重失敗：同一下被算了兩次"
    _send(port, 3)                       # 舊序號重送＝不該再觸發
    time.sleep(0.1)
    assert not tap.take(), "去重失敗：舊序號被當成新按鍵"
    print(f"selftest PASS｜3 下全收到、冗餘封包去重正確、舊序號不重觸發")
    print(f"本機延遲 {min(lat):.1f}–{max(lat):.1f} ms"
          f"（本機迴路；上身後加 WiFi 單程 ~6ms，OLED 線實測）")
    print(f"heartbeat 判定: alive={tap.alive}")


def monitor(port=PORT, serial_dev=None, use_serial=False):
    if use_serial:
        tap = SerialTapListener(serial_dev)
        if tap.err:
            raise SystemExit(f"serial 開啟失敗: {tap.err}")
        print(f"監看 serial {tap.dev}（Ctrl-C 停）。應每 2s 有 heartbeat。")
    else:
        tap = TapListener(port)
        if tap.err:
            raise SystemExit(f"bind {port} 失敗: {tap.err}")
        print(f"監看 UDP :{port}（Ctrl-C 停）。Pico 上電後應每 2s 有 heartbeat。")
    n, was = 0, None
    while True:
        if tap.alive != was:
            was = tap.alive
            print(f"[{time.strftime('%H:%M:%S')}] 連線 {'ON' if was else 'OFF'}")
        if tap.take():
            n += 1
            print(f"[{time.strftime('%H:%M:%S')}] TAP #{n}")
        time.sleep(0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--serial", nargs="?", const=True, default=None,
                    help="serial 模式；可帶裝置路徑，預設抓 /dev/cu.usbmodem*")
    ap.add_argument("--port", type=int, default=PORT)
    a = ap.parse_args()
    if a.selftest:
        selftest(a.port)
    elif a.monitor:
        monitor(a.port,
                serial_dev=a.serial if isinstance(a.serial, str) else None,
                use_serial=a.serial is not None)
    else:
        ap.print_help()
