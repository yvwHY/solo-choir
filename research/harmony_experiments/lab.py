"""lab.py — live_v3 實驗台（單檔 web 控制台，2026-07-31）

不走 app/ 的 pywebview 三層模式：live_v3 是引擎子行程（音訊它自己管），
這裡只是 stdlib HTTP server——一頁 UI＋SSE 把引擎 stdout 串進瀏覽器＋
start/stop 兩個 POST。沒有新依賴，每個端點可 curl，玩壞了殺掉重開就好。

Run（DDSP venv，於 harmony/ 下）:
  .../260724_ddsp_svc/venv/bin/python lab.py     # 自動開瀏覽器
Stop 按鈕送 SIGINT＝引擎跑完收官報表（mic dump＋命題一數字）才退。
"""
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PY = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
      "260724_ddsp_svc/venv/bin/python")
PORT = 8321

_clients = set()
_clients_lock = threading.Lock()
_hist = []              # 引擎輸出歷史（晚進的瀏覽器補看）
_engine = {"proc": None}


def _emit(line):
    _hist.append(line)
    del _hist[:-500]
    with _clients_lock:
        for q in _clients:
            q.put(line)


def _pump(proc):
    for raw in proc.stdout:
        _emit(raw.rstrip("\n"))
    proc.wait()
    _emit(f"__state__ stopped rc={proc.returncode}")
    _engine["proc"] = None


def _start(cfg):
    if _engine["proc"] is not None:
        return {"err": "已經在跑了，先 Stop"}
    cmd = [PY, "-u", str(HERE / "live_v3.py")]
    if cfg.get("mode") == "file":
        src = cfg.get("file_in") or ""
        # Path("") == Path(".") 會存在——必須先擋空字串再驗檔案（07-31 Harry 踩到）
        if not src or not Path(src).is_file():
            return {"err": f"file 模式要先填輸入 wav 路徑（收到：{src!r}）"}
        cmd += ["--file", src, str(HERE / "out" / "lab_render.wav")]
    else:
        for k, f in (("in_name", "--in-name"), ("out_name", "--out-name")):
            if cfg.get(k):
                cmd += [f, cfg[k]]
        cmd += ["--lag", str(cfg.get("lag", 0.6)),
                "--gain", str(cfg.get("gain", 1.5))]
        if cfg.get("out_map"):
            cmd += ["--out-map", cfg["out_map"]]
            if cfg.get("dry_ch") not in (None, ""):
                cmd += ["--dry-ch", str(cfg["dry_ch"])]
    cmd += ["--key", str(cfg.get("key", "auto")),
            "--expr-gain", str(cfg.get("expr_gain", 1.0))]
    if cfg.get("anticipate"):
        cmd += ["--anticipate"]
    if cfg.get("extra"):          # 進階參數原樣傳（如 --hop-ms 100 --bound-ms 450）
        cmd += str(cfg["extra"]).split()
    if cfg.get("rehearse"):
        cmd += ["--rehearse", cfg["rehearse"],
                "--reh-rate", str(cfg.get("reh_rate", 1.0))]
    _emit("__state__ starting")
    _emit("$ " + " ".join(cmd[2:]))
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    _engine["proc"] = proc
    threading.Thread(target=_pump, args=(proc,), daemon=True).start()
    return {"ok": True}


def _stop():
    p = _engine["proc"]
    if p is None:
        return {"err": "沒有在跑"}
    p.send_signal(signal.SIGINT)     # ＝Ctrl-C：跑完收官報表才退

    def _escalate(proc):             # SIGINT 被忽略/卡死的保險（07-31 Stop 失靈案）
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _emit("⚠ SIGINT 15s 未退，改送 SIGTERM")
            proc.terminate()
    threading.Thread(target=_escalate, args=(p,), daemon=True).start()
    return {"ok": True}


def _devices():
    import sounddevice as sd
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append({"i": i, "name": d["name"],
                    "in": d["max_input_channels"],
                    "out": d["max_output_channels"]})
    return out


def _rehearse_files():
    out = []
    for p in sorted((HERE / "out").glob("*_notes.json")):
        try:
            if "lead" in json.load(open(p)):
                out.append(str(p.relative_to(HERE)))
        except Exception:
            pass
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = (HERE / "lab_ui.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/devices":
            self._json(_devices())
        elif self.path == "/rehearse-files":
            self._json(_rehearse_files())
        elif self.path == "/state":
            self._json({"running": _engine["proc"] is not None})
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q = queue.Queue()
            for line in _hist:
                q.put(line)
            with _clients_lock:
                _clients.add(q)
            try:
                while True:
                    try:
                        line = q.get(timeout=15)
                        self.wfile.write(
                            f"data: {json.dumps(line, ensure_ascii=False)}\n\n"
                            .encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _clients_lock:
                    _clients.discard(q)
        else:
            self._json({"err": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        cfg = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/start":
            self._json(_start(cfg))
        elif self.path == "/stop":
            self._json(_stop())
        else:
            self._json({"err": "not found"}, 404)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    url = f"http://127.0.0.1:{PORT}"
    print(f"v3 lab: {url}  (Ctrl-C 關閉；引擎若在跑會一併收掉)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        p = _engine["proc"]
        if p is not None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
