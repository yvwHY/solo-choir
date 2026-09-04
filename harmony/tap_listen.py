"""tap_listen.py — receives the physical button (`firmware/pico_button`) on the
Mac over UDP

The receiving end of the "your turn" key of the answering mode v2. Usage:

    tap = TapListener()          # background thread, non-blocking

    if tap.take():               # a new press, consumed; end the phrase now

    tap.alive                    # is the heartbeat present (checked before going on stage)

De-duplication: the firmware sends three redundant datagrams with the same
sequence number per press, so duplicates are removed by sequence number, which
tolerates UDP packet loss.
Nothing blocks: if nothing arrives, treat the button as absent and let energy
detection work as before. The design of 2026-08-01: the button owns "end", the
ambiguous half, while the start is still decided by energy, and a missed press
cannot lock the system up.

Self-test (no hardware needed):
    python tap_listen.py --selftest     # sends to itself and measures round-trip latency
Monitor (with a Pico attached, to check the whole chain):
    python tap_listen.py --monitor              # the WiFi/UDP version, for the worn mode
    python tap_listen.py --monitor --serial     # the USB serial version, wired for the exhibition (2026-08-04)
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
        self._seqs = {}          # the highest sequence number seen, kept per sender, for de-duplication
        self._pending = 0        # presses not yet consumed
        self._t_last = 0.0       # when any packet last arrived
        self._proto = False      # a valid protocol line (TAP/HB) has been seen, so the other end really is the button
        self.dead = False        # the reader thread has ended (the serial cable was pulled), so fall back to UDP
        self._lock = threading.Lock()
        self.err = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind(("", port))
        except OSError as e:     # the port is taken: block nothing, and degrade to "there is no button"
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
                self._proto = True           # a recognised protocol line means the other end is the button (see wait_proto)
            if parts[0] == b"TAP" and len(parts) > 1:
                try:
                    seq = int(parts[1])
                except ValueError:
                    return
                # The three redundant datagrams count once, so the sequence must
                # move forward. But a sequence that clearly falls back means the
                # sender reset, either the Pico rebooting or the app restarting,
                # both of which start from 0. The old code's monotonic comparison
                # then left the button deaf for the rest of the evening, while
                # the heartbeat still arrived, so alive stayed True and the gap
                # stayed 6 s: wrong with no symptom at all (review S3/A4,
                # 2026-08-12). A fall-back is accepted and resynchronised.
                # 2026-08-18, "the button does nothing while the heartbeat is
                # alive": there are **two senders** on the UDP port, the app's
                # space key from 127.0.0.1 and the WiFi button from a LAN
                # address, each with its own sequence. Keeping one _seq for both
                # meant that whenever the two numbers fell inside the window of
                # 30, the one behind was swallowed wholesale as duplicates.
                # De-duplication is now kept **per sender**; the serial version
                # has a single sender (src="serial") and is unchanged.
                last = self._seqs.get(src, 0)
                if seq > last or seq < last - 30:
                    self._seqs[src] = seq
                    self._pending += 1

    @property
    def alive(self):
        """The heartbeat is present, so the Pico is connected. This is the
        connection indicator checked before going on stage."""
        with self._lock:
            return (time.time() - self._t_last) < self.hb_timeout

    def take(self):
        """Whether there is a new press, consuming it. Several presses count as
        one, so they cannot pile up."""
        with self._lock:
            if self._pending:
                self._pending = 0
                return True
        return False

    def wait_proto(self, timeout):
        """Wait a bounded time for one valid protocol line (TAP or HB). Opening
        successfully does not mean this is the button; the serial version only
        recognises /dev/cu.usbmodem*. If nothing arrives, close it and fall back
        to another source (respond2 live)."""
        t0 = time.time()
        while True:
            with self._lock:
                if self._proto:
                    return True
            if time.time() - t0 >= timeout:
                return False
            time.sleep(0.05)

    def close(self):
        """Release the port or device. Nothing blocks: if it will not close,
        let it be."""
        try:
            self._sock.close()
        except Exception:        # noqa: BLE001 - clean-up must not raise
            pass


class SerialTapListener(TapListener):
    """The USB serial version, the wired route for the exhibition chosen on
    2026-08-04: button, long cable, a Pico beside the Mac, then USB.

    The interface is identical to TapListener (take(), alive, err, and blocking
    nothing); only the transport differs. It reads the serial output of
    `firmware/pico_button/main_usb.py` under the same protocol, TAP n and HB n.
    With dev=None it takes the first /dev/cu.usbmodem* it finds.
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
            if not hits:                 # nothing plugged in: block nothing and degrade to "no button"
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
                self.dead = True         # the cable was pulled, so the caller falls back to UDP
                return
            if line:
                self._ingest(line)

    def close(self):
        try:
            self._ser.close()
        except Exception:        # noqa: BLE001 - never opened, or already closed
            pass


def _send(port, seq, n=3):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    for _ in range(n):
        s.sendto(b"TAP %d" % seq, ("127.0.0.1", port))
        time.sleep(0.01)
    s.close()


def selftest(port=PORT):
    """No hardware needed: send to itself and receive, checking de-duplication
    and measuring local latency."""
    tap = TapListener(port)
    if tap.err:
        raise SystemExit(f"bind {port} failed: {tap.err}")
    time.sleep(0.2)
    lat = []
    for seq in (1, 2, 3):
        t0 = time.perf_counter()
        # Send in the background: the 10 ms sleep between redundant datagrams
        # must not count as latency, since what is measured is the first arrival.
        threading.Thread(target=_send, args=(port, seq), daemon=True).start()
        while not tap.take():
            if time.perf_counter() - t0 > 2:
                raise SystemExit(f"seq {seq} never arrived")
            time.sleep(0.001)
        lat.append((time.perf_counter() - t0) * 1000)
    assert not tap.take(), "de-duplication failed: one press counted twice"
    _send(port, 3)                       # resending an old sequence must not trigger again
    time.sleep(0.1)
    assert not tap.take(), "de-duplication failed: an old sequence read as a new press"
    print("selftest PASS: all 3 presses received, redundant datagrams de-duplicated, old sequences do not retrigger")
    print(f"local latency {min(lat):.1f}-{max(lat):.1f} ms"
          " (local loopback; worn, add about 6 ms one way over WiFi, measured on the OLED line)")
    print(f"heartbeat: alive={tap.alive}")


def monitor(port=PORT, serial_dev=None, use_serial=False):
    if use_serial:
        tap = SerialTapListener(serial_dev)
        if tap.err:
            raise SystemExit(f"could not open serial: {tap.err}")
        print(f"monitoring serial {tap.dev} (Ctrl-C to stop). A heartbeat should arrive every 2 s.")
    else:
        tap = TapListener(port)
        if tap.err:
            raise SystemExit(f"bind {port} failed: {tap.err}")
        print(f"monitoring UDP :{port} (Ctrl-C to stop). Once the Pico is powered a heartbeat should arrive every 2 s.")
    n, was = 0, None
    while True:
        if tap.alive != was:
            was = tap.alive
            print(f"[{time.strftime('%H:%M:%S')}] link {'ON' if was else 'OFF'}")
        if tap.take():
            n += 1
            print(f"[{time.strftime('%H:%M:%S')}] TAP #{n}")
        time.sleep(0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--serial", nargs="?", const=True, default=None,
                    help="serial mode; a device path may be given, otherwise the first /dev/cu.usbmodem*")
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
