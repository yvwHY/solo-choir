"""Studio (arrange) bridge — OFFLINE. js_api for the harmony-line editor. No live Beatrice engine.

Records the mic to a wav, then calls server/studio_api.py (transcribe / render_score / score_to_midi)
in-process. Named studio_bridge (NOT bridge) to avoid colliding with app/bridge.py, which studio_api
imports as `bridge` for DEFAULT_MODEL.

Run (from repo root, engine env):  /opt/anaconda3/envs/vcclient-dev/bin/python studio/shell.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "server"
REC = REPO / "recordings"
EXPORTS = REPO / "exports"               # what Finder reveals on export — keepsakes only, never the working folder
_BACKING_GAIN = 0.25                     # re-record guide level — quiet so it doesn't drag the singer's pitch


def _countin(beats=4, gap=0.8, dur=0.09, sr=48000):
    """A 4-beat click track (last beat higher) → audible count-in before re-recording. sr=48000 to
    MATCH the engine/devices so studio can share one output device with the live app (a mismatched
    rate on a shared device → PortAudio -10851 = the re-record-while-live-runs failure)."""
    import numpy as np
    n = int((beats - 1) * gap * sr) + int(dur * sr) + sr // 20
    out = np.zeros(n, dtype="float32")
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-7 * t).astype("float32")
    for i in range(beats):
        beep = (0.35 * np.sin(2 * np.pi * (990 if i == beats - 1 else 660) * t)).astype("float32") * env
        i0 = int(i * gap * sr)
        out[i0:i0 + len(beep)] += beep
    return out, sr
sys.path.insert(0, str(SERVER))
import studio_api   # noqa: E402  (after sys.path setup)

# Studio renders OFFLINE with the satb2 multi-speaker model (same as solo_min live). Must be
# string-equal to _SATB2_MODEL: studio_api/arrange gate the per-part male/female speaker split on
# model == _SATB2_MODEL. Falls back to the live default if the dir is missing (moved disk) → a bad
# path degrades to a working render.
STUDIO_MODEL = studio_api._SATB2_MODEL
if not Path(STUDIO_MODEL).is_dir():
    STUDIO_MODEL = studio_api.USER_MODEL


# ── take-home QR: LAN download server (exhibition) ────────────────────────────────────────────────
_DL_PORT = [None]                    # started once per app run


def _lan_ip():
    """This Mac's LAN address (works without internet: UDP connect sends nothing). None if only lo."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return None if ip.startswith("127.") else ip
    except Exception:  # noqa: BLE001
        return None


def _ensure_download_server() -> int:
    """One ThreadingHTTPServer on the LAN interface serving ONLY recordings/ (the UI's own static
    server stays loopback-only — the repo is never exposed). Fixed port 8899 so hotspot setups keep
    a stable URL; falls back to an ephemeral port if taken. macOS will ask to allow incoming
    connections once — accept it at the exhibition."""
    if _DL_PORT[0]:
        return _DL_PORT[0]
    import threading
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class _H(SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def list_directory(self, path):  # noqa: N802 (base-class name)
            # No browsing: a visitor trimming the QR URL to "/" must NOT get an index of
            # recordings/. Direct timestamped file URLs (the QR's own links) still work.
            self.send_error(404, "No listing")
            return None

    handler = partial(_H, directory=str(REC))
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", 8899), handler)
    except OSError:
        httpd = ThreadingHTTPServer(("0.0.0.0", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _DL_PORT[0] = httpd.server_address[1]
    return _DL_PORT[0]


# Take-home page design notes (2026-07-07): minimal single column, one CTA (ui-ux-pro-max pattern);
# Solo Choir brand (teal accent, mono letterspaced labels) over the skill's generic palette; SYSTEM
# fonts only — the page is served over an OFFLINE LAN, webfonts would never load. Deliberately a
# single dark look (a keepsake card, not an app). Contrast AA-checked on #10141a.
_TAKEAWAY_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#10141a">
<title>Solo Choir — your recording</title>
<style>
 :root{--bg:#10141a;--ink:#eef3f6;--dim:#93a1ad;--faint:#7a8794;--acc:#7fd0c9;--hair:rgba(238,243,246,.14)}
 *{box-sizing:border-box}
 body{margin:0;min-height:100dvh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
      font-family:-apple-system,system-ui,"Segoe UI",sans-serif;text-align:center;
      padding:max(24px,env(safe-area-inset-top)) max(20px,env(safe-area-inset-right))
              max(24px,env(safe-area-inset-bottom)) max(20px,env(safe-area-inset-left))}
 .card{width:min(360px,100%);animation:in .35s ease-out}
 @keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
 @media (prefers-reduced-motion:reduce){.card{animation:none}}
 .dot{width:8px;height:8px;border-radius:50%;background:var(--acc);margin:0 auto 18px;
      box-shadow:0 0 12px rgba(127,208,201,.7)}
 h1{font-size:19px;font-weight:600;letter-spacing:.34em;margin:0 0 10px;padding-left:.34em}
 .sub{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--dim);letter-spacing:.18em;
      text-transform:uppercase;margin:0 0 8px}
 .date{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--faint);letter-spacing:.14em;
      margin:0 0 30px}
 audio{width:100%;display:block;margin:0 auto 26px}
 a.cta{display:block;color:#10141a;background:var(--acc);text-decoration:none;font-weight:650;
   font-size:16px;line-height:1;padding:16px 26px;border-radius:999px;transition:filter .2s}
 a.cta:active{filter:brightness(.88)}
 a.cta:focus-visible{outline:3px solid var(--ink);outline-offset:3px}
 .hint{margin-top:14px;font-size:12px;color:var(--dim);line-height:1.5}
 .foot{margin-top:34px;padding-top:16px;border-top:1px solid var(--hair);
       font-family:ui-monospace,Menlo,monospace;font-size:10.5px;color:var(--faint);letter-spacing:.12em}
</style></head><body><main class="card">
<div class="dot" aria-hidden="true"></div>
<h1>SOLO&nbsp;CHOIR</h1>
<p class="sub">one voice &middot; a choir &middot; recorded live</p>
<p class="date">{DATE}</p>
<audio controls preload="metadata" src="{AUDIO}" aria-label="your choir recording"></audio>
<a class="cta" href="{AUDIO}" download>Save this recording</a>
<p class="hint">iPhone: long-press &rarr; &ldquo;Download Linked File&rdquo;, or play and share</p>
<footer class="foot">MA&nbsp;COMPUTATIONAL&nbsp;ARTS &middot; GOLDSMITHS</footer>
</main></body></html>
"""


class Bridge:
    def __init__(self) -> None:
        self._win = None
        self._mel: str | None = None        # last recorded melody wav
        self._score: dict | None = None     # last (possibly edited) score
        self._stream = None                 # active recording InputStream (start/stop)
        self._rec_frames: list = []
        self._rec_sr = 48000
        self._render: dict | None = None    # last render: {"stems": {voice: path}, "mix": path}
        self._render_orig: dict | None = None  # the ORIGINAL rendered stems (for re-record revert)
        self._rerec_voice: str | None = None  # voice currently being re-recorded
        self._rerec_align = 0.0              # ms to advance(+)/delay(-) the take vs the score (latency comp)

    # ---- lifecycle ----
    def _attach(self, window) -> None:
        self._win = window

    # ---- B1: smoke ----
    def ping(self) -> dict:
        return {"ok": True}

    def list_devices(self) -> dict:
        try:
            import sounddevice as sd
            if self._stream is None:                       # safe only when not recording
                try:
                    sd._terminate(); sd._initialize()      # re-scan PortAudio so hot-plugged devices appear
                except Exception:  # noqa: BLE001
                    pass
            devs = sd.query_devices()
            ins = [d["name"] for d in devs if d["max_input_channels"] > 0]
            outs = [d["name"] for d in devs if d["max_output_channels"] > 0]
        except Exception as e:  # noqa: BLE001 — no audio host → empty list, app still runs
            return {"inputs": [], "outputs": [], "error": str(e)}
        return {"inputs": ins, "outputs": outs}

    # ---- B2: start/stop record a hummed melody (free length) + play it back, then arrange ----
    def start_record(self, device=None) -> dict:
        """Begin recording mono mic audio in the background (non-blocking). Stop with stop_record."""
        try:
            import sounddevice as sd
            self._rec_frames = []
            self._rec_sr = 48000

            def _cb(indata, frames, t, status):       # audio thread: just stash a copy
                self._rec_frames.append(indata.copy())

            self._stream = sd.InputStream(samplerate=self._rec_sr, channels=1, dtype="float32",
                                          device=(device or None), callback=_cb)
            self._stream.start()
        except Exception as e:  # noqa: BLE001 — bad device / no host
            self._stream = None
            return {"error": str(e)}
        return {"recording": True}

    def stop_record(self) -> dict:
        """Stop recording -> recordings/studio_mel_*.wav. Returns a /recordings/ URL for playback."""
        import numpy as np
        import soundfile as sf
        if self._stream is None:
            return {"error": "not recording"}
        try:
            self._stream.stop(); self._stream.close()
        finally:
            self._stream = None
        x = np.concatenate(self._rec_frames) if self._rec_frames else np.zeros((0, 1), dtype="float32")
        REC.mkdir(parents=True, exist_ok=True)
        name = f"studio_mel_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        p = str(REC / name)
        sf.write(p, x[:, 0], self._rec_sr)
        self._mel = p
        return {"wav": p, "url": "/recordings/" + name, "seconds": round(len(x) / self._rec_sr, 1)}

    def arrange(self, style="choral") -> dict:
        """Transcribe the last recorded melody + arrange into a score (key auto-detected)."""
        if not self._mel:
            return {"error": "no melody recorded yet"}
        try:
            self._score = studio_api.transcribe(self._mel, None, str(style))
        except Exception as e:  # noqa: BLE001 — transcription/arrange failure → report, don't crash
            return {"error": str(e)}
        return self._score

    # ---- Phase D: render the EDITED score into the user's voice + export ----
    def set_score(self, score) -> dict:
        """UI pushes the current (possibly edited) score here before render/export."""
        if score:
            self._score = score
        return {"ok": True}

    def make_takeaway_qr(self) -> dict:
        """Exhibition take-home: QR the audience scans to DOWNLOAD the rendered mix over the local
        Wi-Fi. Starts (once) a download server bound to the LAN interface serving ONLY recordings/,
        converts the mix to .m4a via macOS afconvert (≈10x smaller, phone-friendly; falls back to a
        wav copy), writes a minimal player/download page, QRs its URL. No cloud, works offline on a
        shared hotspot — the audience just needs to be on the same network."""
        if not self._render or not self._render.get("mix"):
            return {"error": "render first"}
        ip = _lan_ip()
        if not ip:
            return {"error": "no LAN address — join a Wi-Fi network (or start a hotspot) first"}
        try:
            port = _ensure_download_server()
        except Exception as e:  # noqa: BLE001
            return {"error": f"download server: {e}"}
        import segno, shutil, subprocess
        stamp = time.strftime("%Y%m%d_%H%M%S")
        audio = f"takeaway_{stamp}.m4a"
        try:
            subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", self._render["mix"],
                            str(REC / audio)], check=True, capture_output=True, timeout=60)
        except Exception:  # noqa: BLE001 — afconvert missing/failed → serve the wav as-is
            audio = f"takeaway_{stamp}.wav"
            shutil.copy(self._render["mix"], REC / audio)
        page = f"takeaway_{stamp}.html"
        (REC / page).write_text(_TAKEAWAY_HTML.replace("{AUDIO}", audio)
                                .replace("{DATE}", time.strftime("%-d %B %Y · %H:%M")),
                                encoding="utf-8")
        url = f"http://{ip}:{port}/{page}"
        qr = f"studio_takeaway_{stamp}.png"
        try:
            # scale 20 → ≥600px: the UI shows it at 300px; DOWNscaling keeps modules even (8 gave
            # 264px stretched up to 300 with pixelated rendering = crooked-looking modules)
            segno.make(url, error="l").save(str(REC / qr), scale=20, border=2)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        return {"url": url, "qr_url": "/recordings/" + qr}

    def make_card_qr(self, url) -> dict:
        """Render `url` (the roll page link with the score in its #fragment) to a QR PNG in
        recordings/. Returns a /recordings/ URL the studio UI shows as <img>. No score logic here —
        the JS codec already built the URL; this only draws the QR."""
        if not url or not isinstance(url, str):
            return {"error": "no url"}
        try:
            import segno
            name = "studio_card_" + time.strftime("%Y%m%d_%H%M%S") + ".png"
            segno.make(url, error="l").save(str(REC / name), scale=12, border=2)   # supersample: the UI
            # shows QRs at 300px; rendering above display size keeps modules even (see make_takeaway_qr)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        return {"url": "/recordings/" + name}

    def render(self, parts=None, instrument="voice", ensemble=True) -> dict:
        """Render the (edited) score → recordings/studio_render_*/. instrument='voice' (the user's
        own voice — needs a recording) or 'piano' (neural-free synth, even & never thin). `parts` =
        harmony voices to include. Returns a /recordings/ URL for the mix."""
        if not self._score:
            return {"error": "arrange first"}
        pl = (list(parts) if parts else None)
        out = str(REC / ("studio_render_" + time.strftime("%Y%m%d_%H%M%S")))
        try:
            if instrument == "piano":
                r = studio_api.render_piano_score(self._score, out, parts=pl)
            elif instrument == "female":
                if not self._mel:
                    return {"error": "record a melody first (female render needs your recording)"}
                r = studio_api.render_female_score(self._score, self._mel, out, parts=pl, model=STUDIO_MODEL)  # SATB → kNN female (Sop via alto ref); slower, offline
            else:
                if not self._mel:
                    return {"error": "record a melody first (voice render needs your recording)"}
                r = studio_api.render_score(self._score, self._mel, out, parts=pl, model=STUDIO_MODEL, ensemble=bool(ensemble))
        except Exception as e:  # noqa: BLE001 — render failure → report, don't crash
            return {"error": str(e)}
        self._render = {"stems": dict(r["stems"]), "mix": r["mix"]}
        self._render_orig = dict(r["stems"])        # snapshot for re-record revert
        return self._render_payload(r)

    def _render_payload(self, r) -> dict:
        rel = lambda p: "/recordings/" + os.path.relpath(p, str(REC))   # noqa: E731
        return {"mix_url": rel(r["mix"]),
                "stems": list(r["stems"].keys()),
                "stem_urls": {v: rel(p) for v, p in r["stems"].items()}}

    # ---- per-voice solo-listen is just stem_urls above; re-record overdubs one part ----
    def rerecord_start(self, voice, device=None, out_device=None, backing_gain=None, align_ms=0) -> dict:
        """Overdub one rendered part: play the OTHER parts as backing (to out_device — pick headphones,
        at backing_gain volume) while recording the mic. align_ms advances(+)/delays(-) the take to
        compensate playback+capture latency. Stop with rerecord_stop, which replaces `voice`."""
        self._rerec_align = float(align_ms or 0)
        if not self._render or voice not in self._render["stems"]:
            return {"error": "render first"}
        if not self._mel:
            return {"error": "no melody"}
        try:
            import sounddevice as sd
            import soundfile as sf
            backing = str(REC / ("studio_backing_" + time.strftime("%Y%m%d_%H%M%S") + ".wav"))
            studio_api.others_backing(self._render["stems"], self._mel, voice, backing)
            ci, cisr = _countin()                         # audible 3·2·1 → headphones, BEFORE recording starts
            sd.play(ci, cisr, device=(out_device or None)); sd.wait()
            self._rec_frames = []
            self._rec_sr = 48000

            def _cb(indata, frames, t, status):
                self._rec_frames.append(indata.copy())

            self._stream = sd.InputStream(samplerate=self._rec_sr, channels=1, dtype="float32",
                                          device=(device or None), callback=_cb)
            self._stream.start()
            bdata, bsr = sf.read(backing, dtype="float32")
            g = _BACKING_GAIN if backing_gain is None else max(0.0, min(1.0, float(backing_gain)))
            sd.play(bdata * g, bsr, device=(out_device or None))   # quiet guide (don't pull the singer); → headphones, non-blocking
            self._rerec_voice = voice
        except Exception as e:  # noqa: BLE001
            self._stream = None
            self._rerec_voice = None
            return {"error": str(e)}
        return {"recording": True, "voice": voice}

    def rerecord_stop(self) -> dict:
        """Stop the overdub → replace `voice`'s stem with the raw take, re-mix, return new urls."""
        import numpy as np
        import soundfile as sf
        import sounddevice as sd
        voice = self._rerec_voice
        if voice is None or self._stream is None:
            return {"error": "not re-recording"}
        try:
            sd.stop()                                    # stop backing playback
        except Exception:  # noqa: BLE001
            pass
        try:
            self._stream.stop(); self._stream.close()
        finally:
            self._stream = None
        self._rerec_voice = None
        x = np.concatenate(self._rec_frames) if self._rec_frames else np.zeros((0, 1), dtype="float32")
        mono = x[:, 0]
        n = int(round(self._rerec_align / 1000.0 * self._rec_sr))   # latency compensation
        if n > 0:
            mono = mono[n:]                                         # advance: drop the latency lead-in
        elif n < 0:
            mono = np.concatenate([np.zeros(-n, dtype=mono.dtype), mono])  # delay: pad the front
        REC.mkdir(parents=True, exist_ok=True)
        take = str(REC / (f"studio_take_{voice.lower()}_" + time.strftime("%Y%m%d_%H%M%S") + ".wav"))
        sf.write(take, mono, self._rec_sr)
        out = str(REC / ("studio_render_" + time.strftime("%Y%m%d_%H%M%S")))
        try:
            r = studio_api.remix_with_take(self._render["stems"], self._mel, voice, take, out)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        self._render = {"stems": dict(r["stems"]), "mix": r["mix"]}
        return self._render_payload(r)

    def rerecord_revert(self, voice) -> dict:
        """Undo a re-record: restore `voice` to its ORIGINAL rendered stem, re-mix."""
        if not self._render or not self._render_orig or voice not in self._render_orig:
            return {"error": "nothing to revert"}
        if not self._mel:
            return {"error": "no melody"}
        out = str(REC / ("studio_render_" + time.strftime("%Y%m%d_%H%M%S")))
        try:
            r = studio_api.remix_with_take(self._render["stems"], self._mel, voice,
                                           self._render_orig[voice], out)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        self._render = {"stems": dict(r["stems"]), "mix": r["mix"]}
        return self._render_payload(r)

    def export_mix(self) -> dict:
        """Copy the current render mix into exports/ and reveal THAT copy in Finder — the
        audience never sees the recordings/ working folder."""
        if not self._render or not self._render.get("mix"):
            return {"error": "render first"}
        p = self._render["mix"]
        try:
            import shutil
            import subprocess
            EXPORTS.mkdir(parents=True, exist_ok=True)
            dst = str(EXPORTS / (Path(p).parent.name.replace("studio_render_", "solo_choir_") + ".wav"))
            shutil.copy(p, dst)
            subprocess.run(["open", "-R", dst], check=False)
        except Exception:  # noqa: BLE001
            pass
        return {"mix": p, "mix_url": "/recordings/" + os.path.relpath(p, str(REC))}

    def export_midi(self) -> dict:
        if not self._score:
            return {"error": "arrange first"}
        EXPORTS.mkdir(parents=True, exist_ok=True)
        p = str(EXPORTS / ("solo_choir_" + time.strftime("%Y%m%d_%H%M%S") + ".mid"))
        try:
            studio_api.score_to_midi(self._score, p)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        try:
            import subprocess
            subprocess.run(["open", "-R", p], check=False)   # reveal the .mid in Finder
        except Exception:  # noqa: BLE001
            pass
        return {"midi": p}
