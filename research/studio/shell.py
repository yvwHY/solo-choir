"""Studio (arrange) — pywebview desktop shell. Mirrors app/shell.py but loads the harmony-line
editor (studio/ui/index.html) and the OFFLINE studio_bridge (no live engine to start).

Run (from repo root, engine env):  /opt/anaconda3/envs/vcclient-dev/bin/python studio/shell.py
"""
from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import webview

from studio_bridge import Bridge

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_REL = "studio/ui/index.html"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def start_static_server(root: Path) -> str:
    handler = partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def main() -> None:
    ui = REPO_ROOT / UI_REL
    if not ui.exists():
        raise SystemExit(f"UI file not found: {ui}")
    base = start_static_server(REPO_ROOT)
    bridge = Bridge()
    window = webview.create_window(
        "Solo Choir — Studio",
        url=f"{base}/{UI_REL}",
        width=1280,
        height=820,
        min_size=(960, 640),
        background_color="#ffffff",   # light canvas (no flash)
        js_api=bridge,
    )
    bridge._attach(window)
    webview.start(debug=True)         # no engine to start; debug=True → right-click → Inspect


if __name__ == "__main__":
    main()
