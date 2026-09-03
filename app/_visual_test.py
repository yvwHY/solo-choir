"""Verify §6 step 3b: the UI reads window.__telemetry (no demo loop, no fake sims).

No engine — we inject telemetry straight into window.__telemetry and read the DOM the
tick() loop produces:
  - no demo loop: with NO telemetry, readings stay '—' (don't cycle)
  - voiced: Sung/Hz/cents/needle/input-level reflect the telemetry values
  - unvoiced: readings go '—', input level falls back toward empty

    /opt/anaconda3/envs/vcclient-dev/bin/python app/_visual_test.py
"""
from __future__ import annotations

import time

import webview

from shell import REPO_ROOT, UI_REL, start_static_server

READ = """(function(){var $=function(i){return document.getElementById(i)};
return JSON.stringify({
  sung: $('roSung').textContent, hz: $('roHz').textContent, harm: $('roHarm').textContent,
  cents: $('centsV').textContent, needleLeft: $('needle').style.left,
  inBarRight: $('inBar').style.right, inLvl: $('inLvlV').textContent
});})()"""


def read(win):
    import json
    return json.loads(win.evaluate_js(READ))


def checks(win):
    time.sleep(2.0)
    res = []

    # 1) no demo loop: no telemetry → Sung stays '—' across time
    win.evaluate_js("window.__telemetry=null;")
    time.sleep(0.4); a = read(win)
    time.sleep(0.4); b = read(win)
    nodemo = a["sung"] == "—" and b["sung"] == "—" and a["harm"] == "—"
    print(f"no-demo (telemetry=null): sung={a['sung']!r}/{b['sung']!r} harm={a['harm']!r} -> {'PASS' if nodemo else 'FAIL'}")
    res.append(nodemo)

    # 2) voiced telemetry → readings reflect it (C4 = midi 60, 261.6Hz, +6¢, level .3)
    win.evaluate_js("window.__telemetry={f0:261.63,midi:60.0,cents:6.0,level:0.3,rtf:0,latency_ms:0,voiced:true};")
    time.sleep(0.8); v = read(win)
    needle = float(v["needleLeft"].rstrip("%"))   # 50 + cents → ~56 for +6¢
    voiced_ok = (v["sung"] == "C4" and v["hz"].startswith("262") and v["cents"] == "+6¢"
                 and 55.0 < needle < 57.0 and v["harm"] == "Tenor A3"
                 and v["inBarRight"] != "100%" and "dB" in v["inLvl"])
    print(f"voiced: {v}  -> {'PASS' if voiced_ok else 'FAIL'}")
    res.append(voiced_ok)

    # 3) unvoiced → readings go '—', input level falls back
    win.evaluate_js("window.__telemetry={f0:0,midi:0,cents:0,level:0.0,rtf:0,latency_ms:0,voiced:false};")
    time.sleep(0.8); u = read(win)
    unvoiced_ok = u["sung"] == "—" and u["harm"] == "—" and u["cents"] == "—" and u["inLvl"] == "−∞ dB"
    print(f"unvoiced: {u}  -> {'PASS' if unvoiced_ok else 'FAIL'}")
    res.append(unvoiced_ok)

    print(f"=== 3b VISUAL: {'PASS' if all(res) else 'FAIL'} ===")
    win.destroy()


def main():
    base = start_static_server(REPO_ROOT)
    win = webview.create_window("Solo Choir (3b test)", url=f"{base}/{UI_REL}",
                                width=1280, height=820, background_color="#c6d0d9")
    webview.start(lambda: checks(win))


if __name__ == "__main__":
    main()
