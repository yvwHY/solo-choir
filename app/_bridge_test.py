"""Verify the §6.2/§6.3a bridge: engine→UI telemetry mapping + UI→engine control.

No mic/engine subprocess here — we inject raw engine telemetry into the Bridge and
check the UI receives the EXACT §2 contract (7 keys), with correct midi/cents/voiced,
and that flipping a control in the UI lands on the Python Bridge. (Real engine f0 is
covered by _tap_test.py; mic-follows-along is the user's app run.)

    /opt/anaconda3/envs/vcclient-dev/bin/python app/_bridge_test.py
"""
from __future__ import annotations

import json
import threading
import time

import webview

from bridge import Bridge
from shell import REPO_ROOT, UI_REL, start_static_server

TEL_KEYS = {"f0", "midi", "cents", "level", "rtf", "latency_ms", "voiced"}
CTRL_KEYS = {"convert", "key", "scale", "intervals", "harmonize", "pitch", "gain", "gate"}


def read_tel(window):
    return json.loads(window.evaluate_js("JSON.stringify(window.__telemetry||null)") or "null")


def run_checks(window, bridge: Bridge) -> None:
    time.sleep(1.0)
    results = []

    # engine→UI : inject a voiced 220Hz reading, expect the mapped contract
    with bridge._lock:
        bridge._engine_tel = {"f0": 220.0, "in_level": 0.2, "out_level": 0.1, "voiced": True}
    time.sleep(0.3)
    t = read_tel(window)
    voiced_ok = (isinstance(t, dict) and set(t) == TEL_KEYS and t["voiced"] is True
                 and abs(t["midi"] - 57.0) < 0.1 and abs(t["cents"]) < 5 and abs(t["level"] - 0.2) < 0.01)
    print(f"engine→UI voiced : {t}")
    print(f"   keys=={'exact' if set(t)==TEL_KEYS else set(t)^TEL_KEYS}  -> {'PASS' if voiced_ok else 'FAIL'}")
    results.append(voiced_ok)

    # engine→UI : unvoiced (silence) → f0/midi/cents zero, voiced false, level still reported
    with bridge._lock:
        bridge._engine_tel = {"f0": 0.0, "in_level": 0.03, "out_level": 0.0, "voiced": False}
    time.sleep(0.3)
    t = read_tel(window)
    unvoiced_ok = (set(t) == TEL_KEYS and t["voiced"] is False and t["f0"] == 0.0
                   and abs(t["level"] - 0.03) < 0.01)
    print(f"engine→UI unvoiced: {t}  -> {'PASS' if unvoiced_ok else 'FAIL'}")
    results.append(unvoiced_ok)

    # UI→engine : flip Convert + key + scale in the UI, push, check Python Bridge
    window.evaluate_js("state.convert=false; state.keyRoot=5; state.minor=true;")
    window.evaluate_js("window.__solochoir_sendControl()")
    time.sleep(0.3)
    ctrl = dict(bridge.control)
    ctrl_ok = (CTRL_KEYS.issubset(ctrl) and ctrl["convert"] is False
               and ctrl["key"] == 5 and ctrl["scale"] == "minor")
    print(f"UI→engine control : {ctrl}  -> {'PASS' if ctrl_ok else 'FAIL'}")
    results.append(ctrl_ok)

    print(f"=== BRIDGE: {'PASS' if all(results) else 'FAIL'} ===")
    window.destroy()


def main() -> None:
    base = start_static_server(REPO_ROOT)
    bridge = Bridge()
    win = webview.create_window("Solo Choir (bridge test)", url=f"{base}/{UI_REL}",
                                width=1280, height=820, background_color="#c6d0d9",
                                js_api=bridge)
    bridge._attach(win)
    win.events.closed += bridge._on_closed

    def boot() -> None:
        # only the telemetry push loop (NOT _start_engine — no mic in this test)
        threading.Thread(target=bridge._telemetry_loop, daemon=True).start()
        run_checks(win, bridge)

    webview.start(boot)


if __name__ == "__main__":
    main()
