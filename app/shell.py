"""Solo Choir — pywebview desktop shell (UI_BUILD_SPEC §6 step 1).

Opens the v10 prototype (ui/solo_choir_ui_v10.html) in a native macOS
WKWebView window so we can confirm three.js / WebGL render correctly.

This is the OUTER layer only. It does not import or touch the audio DSP in
server/. The bundled HTTP server here is *static asset serving* (so the page can
fetch app/assets/head.glb without file:// sandbox/CORS limits).

The control/telemetry bridge (§6 step 2) is wired here via pywebview js_api — see
app/bridge.py. The bridge's telemetry is still a MOCK source until step 3.

Run (from repo root, using the engine env):
    /opt/anaconda3/envs/vcclient-dev/bin/python app/shell.py
"""
from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import webview

from bridge import Bridge

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_REL = "ui/solo_choir_ui_v10.html"  # heavy 3D head UI (has device selection). Lite no-WebGL: ui/solo_choir_ui_simple.html


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request logging
        pass

    def end_headers(self) -> None:         # never let WKWebView serve a stale UI (always load latest)
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()


def start_static_server(root: Path) -> str:
    """Serve `root` on a background thread; return the base URL (http://127.0.0.1:port)."""
    handler = partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)  # port 0 = pick a free one
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def main() -> None:
    ui = REPO_ROOT / UI_REL
    if not ui.exists():
        raise SystemExit(f"UI file not found: {ui}")

    base = start_static_server(REPO_ROOT)
    bridge = Bridge()
    window = webview.create_window(
        "Solo Choir",
        url=f"{base}/{UI_REL}",
        width=1280,
        height=820,
        min_size=(960, 640),
        background_color="#c6d0d9",  # matches the UI's top gradient (no white flash)
        js_api=bridge,               # exposes bridge.set_control as window.pywebview.api.set_control
    )
    bridge._attach(window)
    window.events.closed += bridge._on_closed
    # debug=True → right-click → Inspect Element opens WKWebView devtools
    webview.start(bridge._start, debug=True)   # start engine + telemetry once the GUI is ready


if __name__ == "__main__":
    main()
