"""solo_shell — minimal pywebview window for the solo_min rebuild (You + 1 harmony voice).

Mirrors the choir app/shell.py pattern (static server + webview + js_api bridge) but drives the clean
solo_min.SoloEngine instead of the choir engine. Standalone — does not import or touch the choir app.

Run (conda env vcclient-dev):  python app/solo_shell.py
"""
from __future__ import annotations
import sys, os, threading, json, time, socket
from functools import partial
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse

import webview
import sounddevice as sd

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
UI_REL = "ui/solo_min.html"
sys.path.insert(0, str(SERVER_DIR))
import solo_min  # noqa: E402  (server/solo_min.py — the SoloEngine)

# validated tenor baseline (same as the choir app's default); UI can switch via the model dropdown
sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
MODELS_ROOT = config.BEATRICE_MODELS
# default = the self-trained satb2 (2-speaker: 0=female, 1=male) so the UI opens on the full 5-part
# male+female choir the CLI produces (symlinked into MODELS_ROOT as paraphernalia_data_satb2).
DEFAULT_MODEL = str(MODELS_ROOT / "paraphernalia_data_satb2")
MODEL_BINS = ("embedding_setter.bin", "speaker_embeddings.bin", "pitch_estimator.bin",
              "phone_extractor.bin", "waveform_generator.bin")


# --face-udp: shell-level option (telemetry → Pico OLED face, worklog 07-17 §F/§G). Stripped here so
# the engine parser never sees it — server/ stays untouched. Bare --face-udp broadcasts on the
# Pico-AP subnet (the Pico is the access point, 192.168.4.0/24); pass HOST:PORT to override.
_face_p = argparse.ArgumentParser(add_help=False)
_face_p.add_argument("--face-udp", nargs="?", const="192.168.4.255:5005", default=None,
                     metavar="HOST:PORT")
FACE_ARGS, _ENGINE_ARGV = _face_p.parse_known_args(sys.argv[1:])


def _default_args():
    """Launch defaults = the verified 5-part male+female choir (satb2, per-voice speaker):
    You(mid) + 3rd below + 3rd above(f) + 5th-below-bass(m) + 5th-above-top(f)."""
    a = solo_min.build_argparser().parse_args([
        "--model", DEFAULT_MODEL, "--mode", "diatonic", "--key", "C",
        "--steps", "-2", "--steps2", "2", "--steps3", "-4", "--steps4", "4",
        "--speaker", "0", "--speaker2", "0", "--speaker3", "1", "--speaker4", "0",
        "--in-name", "speaker_set", "--out-name", "speaker_set",
    ] + _ENGINE_ARGV)   # CLI overrides/extends the defaults (e.g. --out-map); no argv → identical
    return a


def _is_model_dir(p: Path) -> bool:
    return p.is_dir() and all((p / b).exists() for b in MODEL_BINS)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def start_static_server(root: Path) -> str:
    handler = partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}"


class Bridge:
    """js_api: window.pywebview.api.* — control the SoloEngine live."""

    def __init__(self):
        self.args = _default_args()
        self.engine: solo_min.SoloEngine | None = None
        self._win = None
        # telemetry: read-only taps (beatrice_converter.telemetry_tap) → latest values here → pushed
        # to the UI every ~100ms by _tele_loop. Never touches the audio path.
        self._tele = {"f0": 0.0, "in_level": 0.0, "voiced": False}
        self._vout: list[float] = []
        self._tele_thread: threading.Thread | None = None
        # face UDP (off unless --face-udp): tiny {level, voiced, f0} datagram per tele tick, sent from
        # this thread only — the audio path never blocks on the network.
        self._face_sock = self._face_addr = None
        if FACE_ARGS.face_udp:
            host, port = FACE_ARGS.face_udp.rsplit(":", 1)
            self._face_addr = (host, int(port))
            self._face_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._face_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def _attach(self, window):
        self._win = window

    def _attach_taps(self):
        eng = self.engine
        self._vout = [0.0] * len(eng.voices)

        def make_tap(i):
            def tap(f0, in_level, out_level, voiced):
                self._vout[i] = out_level
                if i == 0:   # all voices see the same input; voice 0 reports it
                    self._tele.update(f0=f0, in_level=in_level, voiced=voiced)
            return tap

        for i, (conv, _harm) in enumerate(eng.voices):
            conv.telemetry_tap = make_tap(i)

    def _tele_loop(self):
        while True:
            time.sleep(0.1)
            eng, win = self.engine, self._win
            if eng is None or win is None:
                continue
            try:
                voices = [{"out": round(self._vout[i], 4),
                           "shift": round(float(conv._cur_shift), 2),
                           "on": bool(eng.voice_on[i])}
                          for i, (conv, _h) in enumerate(eng.voices)]
                payload = {"f0": round(float(self._tele["f0"]), 2),
                           "level": round(float(self._tele["in_level"]), 4),
                           "voiced": bool(self._tele["voiced"]),
                           "dry": bool(eng.dry_on), "voices": voices,
                           "ent": eng.ent_state()}   # None unless --entrance; drives the Entrance button
                win.evaluate_js("window.soloTele&&window.soloTele(%s)" % json.dumps(payload))
                if self._face_sock:
                    self._face_sock.sendto(json.dumps({"level": payload["level"],
                                                       "voiced": payload["voiced"],
                                                       "f0": payload["f0"]}).encode(),
                                           self._face_addr)
            except Exception:   # noqa: BLE001 — engine mid-rebuild / window closing: skip this tick
                pass

    def _start(self):
        self.engine = solo_min.SoloEngine(self.args)
        self.engine.start()
        self._attach_taps()
        if self._tele_thread is None:
            self._tele_thread = threading.Thread(target=self._tele_loop, daemon=True)
            self._tele_thread.start()

    def set_control(self, ctrl: dict):
        if self.engine:
            self.engine.set_control(ctrl or {})
        return True

    def get_state(self):
        """Current control state so the UI initialises truthful (matches launch args + live toggles)."""
        a, eng = self.args, self.engine
        spk = [a.speaker,
               a.speaker if a.speaker2 is None else a.speaker2,
               a.speaker if a.speaker3 is None else a.speaker3,
               a.speaker if a.speaker4 is None else a.speaker4]
        return {"mode": a.mode, "key": a.key, "minor": bool(a.minor), "free": bool(a.free),
                "naive": bool(getattr(a, "naive", False)),   # A/B: harmony timbre = voice | shifter
                "entrance": bool(getattr(a, "entrance", False)),   # exhibition opening (voices fade in)
                "choir_gain": float(eng.choir_gain) if eng else float(getattr(a, "choir_gain", 1.0)),
                "gate": float(eng.voices[0][0].output_gate_floor) if eng else float(getattr(a, "gate_floor", 0.0)),
                "you_gain": float(eng.you_gain) if eng else float(getattr(a, "you_gain", 1.0)),
                "steps": [a.steps, a.steps2, a.steps3, a.steps4],
                "intervals": [a.interval, a.interval2, a.interval3, a.interval4],
                "speakers": spk,
                "glide": a.glide, "hysteresis": a.hysteresis, "smooth": a.smooth,
                "dry": bool(eng.dry_on) if eng else (not a.no_dry),
                "on": list(eng.voice_on) if eng else [True, True, True, True]}

    def _notify(self, msg: str):
        """Push a rebuild error ('' = clear) to the UI status line. Never raises."""
        try:
            if self._win:
                self._win.evaluate_js("window.soloErr&&window.soloErr(%s)" % json.dumps(str(msg)))
        except Exception:  # noqa: BLE001 — window closing
            pass

    def _rebuild(self, reinit_portaudio=False):
        """Stop + restart the engine on current args, carrying live toggle state across the
        rebuild so the UI stays truthful (G7: control-state completeness on device/model switch).
        reinit_portaudio=True re-enumerates devices between stop and start — PortAudio's device
        snapshot is frozen at init, so a hot-(re)plugged device is invisible without it.
        Engine start can fail (device gone, out_map needing more channels than the device has →
        PortAudio error / SystemExit): catch it, leave engine=None and surface the error in the UI
        instead of letting the exception blow through the js_api and kill audio silently."""
        prev = self.engine
        if prev:
            prev.stop()
        self.engine = None
        if reinit_portaudio:
            sd._terminate()
            sd._initialize()
        try:
            self._start()
        except (Exception, SystemExit) as e:  # noqa: BLE001
            self.engine = None   # _start assigns engine before .start() — drop the half-built one
            self._notify(f"engine start failed: {e}")
            return False
        if prev is not None:
            self.engine.voice_on = list(prev.voice_on)[:len(self.engine.voice_on)]
            self.engine.dry_on = prev.dry_on
            self.engine.out_gain = prev.out_gain   # was silently reset to 1.0 on every rebuild
        self._notify("")
        return True

    def set_model(self, name: str):
        """Switch model → rebuild the engine on the new model (stop old stream, start new)."""
        cand = (MODELS_ROOT / str(name)).resolve()
        if MODELS_ROOT.resolve() not in cand.parents or not _is_model_dir(cand):
            return False
        self.args.model = str(cand)
        return self._rebuild()

    def _out_dev_channels(self):
        """Max output channels of the CURRENTLY selected output device (None → system default)."""
        try:
            dev = solo_min.pick(self.args.out_name, "output")
            return int(sd.query_devices(dev, "output")["max_output_channels"])
        except Exception:  # noqa: BLE001 — device vanished mid-query
            return 2

    def set_devices(self, in_name, out_name):
        """UI picked audio devices → rebuild the engine on them (exact device names from
        list_devices; substring-matched by solo_min.pick, empty → system default).
        An active out_map that needs more channels than the new device has is dropped honestly
        (notified, UI repaints from list_devices) — never a silent dead engine."""
        self.args.in_name = (in_name or "").strip() or None
        self.args.out_name = (out_name or "").strip() or None
        dropped = None
        if self.args.out_map:
            need = max(int(t) for t in str(self.args.out_map).split(",")) + 1
            have = self._out_dev_channels()
            if need > have:
                self.args.out_map = None
                self.args.dry_ch = -1
                dropped = f"routing off: map needed {need} ch, device has {have}"
        ok = self._rebuild()
        if ok and dropped:
            self._notify(dropped)
        return ok

    def set_out_map(self, out_map, dry_ch=-1):
        """UI routing row → per-voice output channel on the ONE selected output device
        (F7: one multi-channel device, one clock — never per-voice devices/streams).
        out_map = 'a,b,c,d' (one 0-based channel per voice) or None/'' = legacy mono mix.
        Validated here so a bad pick is a UI error line, not a dead engine; the running
        engine is left untouched on rejection."""
        m = str(out_map or "").strip()
        if not m:
            self.args.out_map = None
            self.args.dry_ch = -1
            return self._rebuild()
        n_voices = len(solo_min._voice_specs(self.args))
        try:
            chans = [int(t) for t in m.split(",")]
        except ValueError:
            self._notify(f"bad routing map: {m!r}")
            return False
        if len(chans) != n_voices or min(chans) < 0:
            self._notify(f"routing map needs {n_voices} non-negative entries, got {m!r}")
            return False
        n_out = max(chans) + 1
        have = self._out_dev_channels()
        if n_out > have:
            self._notify(f"map needs {n_out} ch, output device has {have}")
            return False
        d = int(dry_ch)
        if not (-1 <= d < n_out):   # engine has no upper-bound check → dry would vanish silently
            self._notify(f"dry channel {d} outside map range 0..{n_out - 1}")
            return False
        self.args.out_map = ",".join(str(c) for c in chans)
        self.args.dry_ch = d
        return self._rebuild()

    def rescan_devices(self):
        """UI ↻ button: pick up hot-(re)plugged devices without restarting the app —
        rebuild with a PortAudio re-init, then hand back the fresh device list."""
        self._rebuild(reinit_portaudio=True)
        return self.list_devices()

    # models to hide from the UI dropdown (overfit/unwanted); filesystem untouched, reversible
    _EXCLUDE = {"paraphernalia_data_00010000"}

    def list_models(self):
        cur = os.path.basename(self.args.model)
        names = sorted(p.name for p in MODELS_ROOT.glob("paraphernalia_data_*")
                       if _is_model_dir(p) and p.name not in self._EXCLUDE)
        return {"models": names, "current": cur}

    def list_devices(self):
        ins = sorted({d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0})
        outs: dict[str, int] = {}
        for d in sd.query_devices():
            if d["max_output_channels"] > 0:
                outs[d["name"]] = max(outs.get(d["name"], 0), int(d["max_output_channels"]))
        return {"inputs": ins,
                "outputs": [{"name": n, "ch": c} for n, c in sorted(outs.items())],
                "in_name": self.args.in_name, "out_name": self.args.out_name,
                "out_map": self.args.out_map, "dry_ch": int(getattr(self.args, "dry_ch", -1))}

    def _on_closed(self):
        if self.engine:
            self.engine.stop()


def main():
    ui = REPO_ROOT / UI_REL
    if not ui.exists():
        raise SystemExit(f"UI file not found: {ui}")
    base = start_static_server(REPO_ROOT)
    bridge = Bridge()
    window = webview.create_window(
        "Solo · min", url=f"{base}/{UI_REL}", width=920, height=680,
        min_size=(760, 540), background_color="#c6d0d9", js_api=bridge,
    )
    bridge._attach(window)
    window.events.closed += bridge._on_closed
    webview.start(bridge._start, debug=True)


if __name__ == "__main__":
    main()
