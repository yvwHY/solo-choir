"""Solo Choir control/telemetry bridge — UI_BUILD_SPEC §6 step 2 + 3a + device routing.

Transport: pywebview js_api (single process for the GUI).
  - UI -> engine : window.pywebview.api.set_control / list_devices / set_devices
  - engine -> UI : Bridge pushes window.onTelemetry({...}) via evaluate_js, ~30/s

Telemetry source is the REAL engine, run via the existing harness CLI
(server/beatrice_solo_choir_live.py) with the read-only --emit-telemetry flag.
Device routing: the UI lists real sounddevice devices and selecting one (re)launches
the engine subprocess with the harness's own --in-name/--out-name flags. No DSP is
touched here; server/ only gains the read-only telemetry tap.

Contracts (from §2, UNCHANGED):
  telemetry (engine -> UI): {f0, midi, cents, level, rtf, latency_ms, voiced}
  control   (UI -> engine): {convert, key, scale, intervals, harmonize, pitch, gain, gate}
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import mido
    _MIDI_OK = True
except Exception:   # noqa: BLE001 — mido/rtmidi absent → MIDI feature simply off, app runs normally
    _MIDI_OK = False

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "server"
RECORDINGS_DIR = REPO_ROOT / "recordings"   # where the engine writes take_*.wav (gitignored)
MIDI_CHORDS_PATH = RECORDINGS_DIR / "midi_chords.json"   # persisted Mode B custom chords (gitignored)
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
DEFAULT_MODEL = str(config.BEATRICE_MODELS / "paraphernalia_data_new25_2k")
# Models live as paraphernalia_data_* dirs in MODELS_ROOT (below); the UI lists them and
# set_model() switches live. new25_2k (F2–C5 protocol dataset, 2000 steps) is the validated
# default — it beat the old 2k on held-out high-note blind A/B (TRAINING_NOTES 2026-06-20);
# the old paraphernalia_data_00002000 stays in the folder, still selectable.
MODELS_ROOT = Path(DEFAULT_MODEL).parent
MODEL_BINS = ("embedding_setter.bin", "speaker_embeddings.bin", "pitch_estimator.bin",
              "phone_extractor.bin", "waveform_generator.bin")


def _is_model_dir(p: Path) -> bool:
    return p.is_dir() and all((p / b).exists() for b in MODEL_BINS)
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TELEMETRY_HZ = 12   # UI push rate; lowered from 30 → fewer evaluate_js round-trips = less GUI CPU contention with the engine (helps --pump underruns)
# SATB voice config (B′ path): T/B = the user's tenor model; S/A = the self-trained female model.
# NOTE (2026-06-22): render_satb now bounces the CURRENT live voices (self.control["voices"], the
# on-screen JVS choir + female Alto), NOT this constant — so SATB_VOICES is currently UNWIRED. Kept
# as the alternative "offline female-model S/A" preset (flip Alto/Sop on + point _render_satb_worker
# here) if the female model ever beats JVS for S/A by ear.
SATB_VOICES = [
    # SATB: LIVE default = clean all-tenor (You + Bass + Tenor on the validated tenor model) — the
    # only combo this laptop runs cleanly real-time. A 2nd model OR the female pitched up live →
    # crackle + warble (cache / real-time limit; MULTIVOICE_STATUS §3/§5/§7, TRAINING_NOTES 2026-06-20).
    # Real male+female S/A is delivered via the OFFLINE render (clean): flip Alto/Sop on (they use the
    # self-trained female model) for render_satb. The 2-speaker satb2 model stays available by path.
    {"part": "Bass",  "model": "tenor",  "speaker": 0, "octave": -1, "interval": -7, "formant": 0.0, "on": True},
    {"part": "Tenor", "model": "tenor",  "speaker": 0, "octave": 0,  "interval": -2, "formant": 0.0, "on": True},
    {"part": "Alto",  "model": "female", "speaker": 0, "octave": 0,  "interval": 2,  "formant": 0.0, "on": False},
    {"part": "Sop",   "model": "female", "speaker": 0, "octave": 0,  "interval": 4,  "formant": 0.0, "on": False},
]
# LIVE "choir mode" (2026-06-21) — validated better-sounding by ear: the female S/A come through
# and aren't buried by Bass (vs the single-voice You+Bass default). ONE self-trained satb2 model
# serves every part via target_speaker (sp1 = male T/B, sp0 = female S/A) → a single weight set in
# CPU cache, so the live multi-pass stays clean (no crackle thrash). Formant-preserving thickening is
# the engine's native neural conversion (= Harmonizr's core; no DSP pitch-shifter needed). Runs with
# the big-buffer + no-You config (see _start_engine). Separate from SATB_VOICES so the offline
# render_satb path (female model) is untouched. Deferred: stereo spacing (SynthV) + true low latency
# (Route A — inference off the audio callback). Revert: git checkout.
CHOIR_VOICES = [
    # With Route A (--pump, inference off the audio thread), 'low' latency runs 3 passes clean (the
    # in-callback path could only do 1-2 before crackling). Default You+Bass+Tenor+Sop with Bass+Tenor+Sop
    # on; Alto off but toggleable (a 4th pass is fine on pump as long as RTF stays < 1).
    {"part": "Bass",  "model": "satb2", "speaker": 1, "octave": -1, "interval": -7, "formant": 0.0, "on": True},
    {"part": "Tenor", "model": "satb2", "speaker": 1, "octave": 0,  "interval": -2, "formant": 0.0, "on": True},
    {"part": "Alto",  "model": "satb2", "speaker": 0, "octave": 0,  "interval": 2,  "formant": 0.0, "on": False},
    {"part": "Sop",   "model": "satb2", "speaker": 0, "octave": 0,  "interval": 4,  "formant": 0.0, "on": True},
]
# LIVE "tenor/alto" arrangement (2026-06-22) — the dual-model combo validated by ear on 6/21
# (worklog 2026-06-21 §E): You(--no-you) + Tenor(3rd BELOW, the tenor model) + Alto(3rd ABOVE, the
# self-trained female model). 2 inference passes, no crossfade (the tenor↔alto crossfade was dropped
# for cost). This is xfade_test.py distilled to its clean core, wired through the existing engine
# per-voice "model" routing (tenor→--model, female→_FEMALE_MODEL — no DSP touched). Swap back to the
# satb2 single-model live default by pointing __init__'s voices at CHOIR_VOICES (or git checkout).
TENOR_ALTO_VOICES = [
    {"part": "Tenor", "model": "tenor",  "speaker": 0, "octave": 0, "interval": -2, "formant": 0.0, "on": True},
    {"part": "Alto",  "model": "female", "speaker": 0, "octave": 0, "interval": 2,  "formant": 0.0, "on": True},
]
# LIVE JVS SATB (2026-06-22) — sweep-picked JVS timbres (server/sweep_speakers.py) for the male
# parts: S=jvs61 (bright), T=jvs69 (mid), B=jvs47 (dark), all on ONE jvs model via target_speaker.
# Alto (added 2026-06-22 per user) is the self-trained FEMALE model — so this is now 4 passes across
# TWO weight sets (jvs + female), no longer the single-cache CHOIR_VOICES win: re-verify by ear it
# stays clean on --pump (watch RTF + crackle cache-thrash; drop Alto if it crackles). JVS base =
# academic/non-commercial license (gitignored, by path).
# Must stay in lockstep with the UI `parts` array (model is set HERE; the UI only sends
# on/interval/octave/speaker/formant, matched by part name). Revert: point __init__'s voices
# back at TENOR_ALTO_VOICES (or git checkout this + the UI parts array).
JVS_SATB_VOICES = [
    {"part": "Bass",  "model": "jvs",    "speaker": 47, "octave": -1, "interval": -4, "formant": 0.0, "on": True},
    {"part": "Tenor", "model": "jvs",    "speaker": 69, "octave": 0,  "interval": -2, "formant": 0.0, "on": True},
    # Alto = the self-trained FEMALE model (_FEMALE_MODEL, single-speaker → 0), +2 above the melody —
    # the female setting validated by ear on 6/21 (TENOR_ALTO_VOICES). This adds a SECOND model
    # alongside jvs → 2 weight sets in CPU cache; watch for crackle cache-thrash + the 4th pass on RTF.
    {"part": "Alto",  "model": "female", "speaker": 0,  "octave": 0,  "interval": 2,  "formant": 0.0, "on": True},
    # Sop bumped +2 → +4 so it sits above the new Alto (standard SATB ascending stack; matches the
    # offline SATB_VOICES voicing). Bass/Tenor unchanged.
    {"part": "Sop",   "model": "jvs",    "speaker": 61, "octave": 0,  "interval": 4,  "formant": 0.0, "on": True},
]
# LOW-LATENCY 2-voice live (2026-06-25) — for the ≤50ms / RTF~0.3 target. ONLY 2 inference passes
# run (live keeps every CONFIGURED voice warm, so RTF is set by this list's length, not on/off), and
# both sit on ONE jvs model → single CPU-cache weight set (no thrash) → fast, low RTF variance → a
# small cushion stays clean. ~0.16 RTF/pass × 2 ≈ 0.32. Tenor (jvs69, −2) + Sop (jvs61, +4): a
# low+high 2-part. Full SATB still via offline render. Revert to 4-voice: point __init__ at
# JVS_SATB_VOICES (or git checkout). Timbres/parts here are by-ear tweakable.
LOWLAT_VOICES = [
    {"part": "Tenor", "model": "jvs", "speaker": 69, "octave": 0, "interval": -2, "formant": 0.0, "on": True},
    {"part": "Sop",   "model": "jvs", "speaker": 61, "octave": 0, "interval": 4,  "formant": 0.0, "on": True},
]
# LIVE "all my voice" (2026-06-25, per user) — every part on the USER'S OWN trained model ("tenor"
# routes to args.model = DEFAULT_MODEL), single speaker → ONE weight set in CPU cache (clean, no
# cross-model thrash). A literal choir of the user's own voice. Same SATB intervals/octaves as
# JVS_SATB_VOICES; only the model/speaker changed. Revert → JVS_SATB_VOICES (or git checkout).
MYVOICE_VOICES = [
    {"part": "Bass",  "model": "tenor", "speaker": 0, "octave": -1, "interval": -4, "formant": 0.0, "on": True},
    {"part": "Tenor", "model": "tenor", "speaker": 0, "octave": 0,  "interval": -2, "formant": 0.0, "on": True},
    {"part": "Alto",  "model": "tenor", "speaker": 0, "octave": 0,  "interval": 2,  "formant": 0.0, "on": True},
    {"part": "Sop",   "model": "tenor", "speaker": 0, "octave": 0,  "interval": 4,  "formant": 0.0, "on": True},
]
# output devices we must NEVER auto-pick (built-in speaker -> mic feedback loop)
# The two Chinese entries are macOS device names on a zh-Hant system
# ("Built-in" / "Speakers"). They are match targets, not prose: do not translate.
SPEAKER_HINTS = ("built-in", "macbook", "speaker", "internal", "內建", "揚聲器")


def _f0_to_midi(f0: float) -> float:
    return 69.0 + 12.0 * math.log2(f0 / 440.0)


# Mode B custom chords: each WHITE key (pitch class) → a chord (root pc + quality). The bridge
# builds the absolute MIDI notes and sends them down the existing Mode A chord path (engine
# unchanged). Anchored so the root sits in the C4 octave; played at a fixed octave regardless of
# which octave the key is pressed (the slot is keyed by pitch class).
_CHORD_QUALITIES = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "7": [0, 4, 7, 10], "sus4": [0, 5, 7],
}
_CHORD_BASE = 60                                  # root anchored in the C4 octave (MIDI 60..71)
_WHITE_PCS = [0, 2, 4, 5, 7, 9, 11]               # C D E F G A B
# default = C-major diatonic chords on each white key (works out of the box; every slot editable)
_DEFAULT_CHORDS = {
    0: {"root": 0, "quality": "maj"},   2: {"root": 2, "quality": "min"},
    4: {"root": 4, "quality": "min"},   5: {"root": 5, "quality": "maj"},
    7: {"root": 7, "quality": "maj"},   9: {"root": 9, "quality": "min"},
    11: {"root": 11, "quality": "dim"},
}


def _build_chord(root_pc: int, quality: str) -> list[int]:
    """Absolute MIDI notes of the chord (root pc + quality), anchored in the C4 octave."""
    ivs = _CHORD_QUALITIES.get(quality, _CHORD_QUALITIES["maj"])
    root = _CHORD_BASE + (int(root_pc) % 12)
    return [root + i for i in ivs]


class Bridge:
    def __init__(self, model: str | None = None) -> None:
        self.control: dict = {
            "convert": True, "key": 0, "scale": "major",
            "intervals": [-7], "harmonize": True,   # default harmony = Bass, an octave below (≠ Tenor's −2)
            "pitch": 0, "gain": 0, "gate": -48, "auto_chord": False,  # free chord-aware harmony toggle
            "you": False,                               # choir mode: monitor the choir only (no raw You) → big buffer adds no slap-back
            "voices": [dict(v) for v in JVS_SATB_VOICES],  # live: JVS SATB (S=jvs61/T=jvs69/B=jvs47, Alto=female) — matches the UI parts' speakers; revert → MYVOICE_VOICES ('choir of one')
        }
        self.model = model or DEFAULT_MODEL
        self.in_name: str | None = None
        self.out_name: str | None = None
        # PortAudio suggested latency. 'low' (~47ms mouth-to-ear on the Aggregate Device) — low enough
        # that you don't hear your own (bone-conducted) voice slap-back against the processed output.
        # 'high' is stable but the ~150ms lag is an audible echo. With the all-tenor SATB (no JVS) the
        # multi-voice passes stay within the low buffer + BLAS=1, so 'low' is the right default again.
        self.latency = "low"   # choir mode: Tier-1 finding — 'low' on the Aggregate Device gives ~77ms for 3 passes (vs ~207ms at 0.2); test by ear that it stays clean
        # SATB multi-voice live is OFF — this laptop only runs the single-voice path cleanly in real
        # time (live multi-pass / a 2nd model / female pitched up = crackle + warble; see
        # TRAINING_NOTES 2026-06-20, MULTIVOICE_STATUS §3/§5/§7). Live = the proven, byte-identical
        # You + one harmony. Full male+female SATB is delivered via the OFFLINE render (clean).
        self.satb = True   # choir mode ON (satb2 single-model multi-voice; CHOIR_VOICES). Revert: set False
        self.overdub = False   # manual overdub mode (separate transport); True → _start_engine launches single-voice interval-0
        self.lowlat = False   # low-latency SOLO mode: You + ONE converted voice, small buffers (~46ms on the aggregate)
        self.od_perform_interval = -2   # overdub PLAY = perform You + this harmony (−2 = a 3rd below) over the loop
        self._win = None
        self._closed = False
        self._engine: subprocess.Popen | None = None
        self._engine_gen = 0
        self._engine_tel: dict | None = None
        self._engine_alive = False    # watchdog: true once launched; false when the read loop ends (engine died)
        self._stop_reason = ""        # why the engine stopped (shown in the UI pill)
        self._control_dirty = False   # has the UI sent any control beyond the launch defaults?
        self._lock = threading.Lock()
        self._mix_seq = 0             # Track A multi-part practice: temp-mix filename counter
        self._mix_prev = None         # previous temp mix to delete on the next play_mix
        # MIDI input (Mode A chord-hold / Mode B sets key). Absent device → never sends the
        # `midi` control field → engine stays byte-identical.
        self.midi_mode = "chord"      # "chord" (Mode A) | "key" (Mode B: white key → custom chord)
        self.midi_chords = self._load_midi_chords()   # {white-key pc: {root, quality}}
        self._midi_held: set[int] = set()
        self._midi_port = None
        self._midi_name: str | None = None
        self._midi_thread: threading.Thread | None = None
        self._midi_stop = False

    # ===== exposed to JS (window.pywebview.api.*) =====
    def set_control(self, ctrl: dict | None):
        if ctrl:
            # `model` is bridge-owned: the UI's controlState() sends voices WITHOUT it (it only
            # carries on/interval/octave/formant/speaker — see JVS_SATB_VOICES note + ui controlState).
            # A plain update() would drop model from self.control["voices"], so the next device/model
            # RESTART rebuilds --voices model-less → engine _build_voice() defaults to "tenor" → the JVS
            # female timbre collapses to the male model until app restart. Re-stamp model per part here.
            if isinstance(ctrl.get("voices"), list):
                prev = {v.get("part"): v.get("model") for v in self.control.get("voices", [])}
                for v in ctrl["voices"]:
                    if "model" not in v and prev.get(v.get("part")):
                        v["model"] = prev[v.get("part")]
            self.control.update(ctrl)
        self._control_dirty = True
        self._send_control()          # push live to the running engine (stdin)
        return True

    def record(self, on):
        """§6.5 transport — start/stop recording the live mix (JS: pywebview.api.record(bool))."""
        self._send_json({"record": bool(on)})
        return True

    def reconnect(self):
        """UI 'Reconnect' after an engine stop — relaunch on the current devices/model."""
        self._restart_engine()
        return True

    def play(self, on):
        """§6.5 transport — play back / stop the recorded take (no live mic → no feedback)."""
        self._send_json({"play": bool(on)})
        return True

    def exciter(self, on):
        """Hardware: R channel = exciter (perform). off → mute R for learn/monitor; L=bone unaffected."""
        self._send_json({"exciter": bool(on)})
        return True

    def set_part(self, part):
        """UI part tab (Melody/Choir/Both) → engine picks which stem to play back."""
        self._send_json({"play_part": str(part)})
        return True

    def play_file(self, path):
        """Track A ear-trainer — solo-play an on-disk rendered stem (recordings/*_satb_*.wav).
        Empty path → back to the in-memory take. Restricted to the recordings dir for safety."""
        p = (path or "").strip()
        if p:
            try:
                rp = Path(p).resolve()
                if not (rp.exists() and RECORDINGS_DIR.resolve() in rp.parents):
                    return False
                p = str(rp)
            except Exception:  # noqa: BLE001 — bad path; never crash the bridge
                return False
        self._send_json({"play_file": p})
        return True

    def play_mix(self, paths):
        """Track A — play the SUM of several rendered part stems (multi-select practice: turn parts on/off;
        leaving your own part off = hold-against). Empty → back to the take. Sums off the audio thread into
        a fresh temp file (new name each call so the engine reloads), deleting the previous one. recordings/ only."""
        items = [p for p in (paths or []) if p]
        if not items:
            return self.play_file("")
        try:
            import soundfile as sf, numpy as np
            bufs, sr = [], 48000
            for p in items:
                rp = Path(p).resolve()
                if not (rp.exists() and RECORDINGS_DIR.resolve() in rp.parents):
                    continue
                d, sr = sf.read(str(rp), dtype="float32", always_2d=True)
                bufs.append(d[:, 0])
            if not bufs:
                return self.play_file("")
            L = min(len(b) for b in bufs)
            mix = np.sum([b[:L] for b in bufs], axis=0)
            peak = float(np.max(np.abs(mix))) if mix.size else 0.0
            if peak > 1.0:
                mix = mix / peak                      # guard against clipping when summing parts
            self._mix_seq += 1
            out = RECORDINGS_DIR / f"_practice_mix_{self._mix_seq}.wav"
            sf.write(str(out), mix, sr)
            if self._mix_prev:
                try: Path(self._mix_prev).unlink()
                except Exception: pass                # noqa: BLE001 — stale temp already gone
            self._mix_prev = str(out)
            return self.play_file(str(out))
        except Exception:  # noqa: BLE001 — bad stems; never crash the bridge
            return False

    # ===== Manual overdub (separate transport; never touches record/play/play_buf) =====
    def od_enter(self, on):
        """Enter/leave overdub mode → relaunch the engine single-voice (interval 0, no-You)."""
        want = bool(on)
        if want != self.overdub:
            self.overdub = want
            self.control["od_on"] = want            # engine gates the overdub path on this (re-sent on restart)
            if not want:                             # leaving → clear transient overdub controls
                for k in ("od_rec", "od_play", "od_export"):
                    self.control[k] = False
            if self._engine is not None:
                self._restart_engine()
        return {"overdub": self.overdub}

    def set_lowlat(self, on):
        """Toggle low-latency SOLO mode (You + ONE converted voice, ~46ms on the aggregate) vs the
        full choir. Mutually exclusive with the SATB choir → relaunch the engine."""
        want = bool(on)
        if want != self.lowlat:
            self.lowlat = want
            self.satb = not want      # solo ON → choir OFF; solo OFF → back to the 4-voice choir
            if self._engine is not None:
                self._restart_engine()
        return {"lowlat": self.lowlat, "satb": self.satb}

    def od_record(self, on):
        # record a CLEAN layer: the exact sung note (interval 0), no You → layers stack into the
        # harmony accompaniment without doubling your raw voice.
        msg = {"od_on": True, "od_rec": bool(on)}
        if on:
            msg["intervals"] = [0]; msg["you"] = False
        self._send_json(msg); return True

    def od_play(self, on):
        # PERFORM over the loop: the recorded layers are the backing; your live voice (You + a 3rd-below
        # harmony) sings on top. Stop → leave the live voice settings (You/harmony) as the user left them.
        msg = {"od_on": True, "od_play": bool(on)}
        if on:
            msg["intervals"] = [self.od_perform_interval]; msg["you"] = True
        self._send_json(msg); return True

    def od_mute(self, indices):
        self._send_json({"od_on": True, "od_mute": [int(i) for i in (indices or [])]}); return True

    def od_delete(self, i):
        self._send_json({"od_on": True, "od_del": int(i)})   # engine self-clears (fires once)
        return True

    def od_clear(self):
        self._send_json({"od_on": True, "od_clear": True})   # engine self-clears (fires once)
        return True

    def od_export(self):
        self._send_json({"od_on": True, "od_export": True})  # engine self-clears (fires once)
        return True

    def log_practice(self, entry):
        """Track A Phase 4 — record one practice attempt's aggregates to recordings/practice_log.csv
        (gitignored local study data). The UI sends per-attempt stats; we stamp the time and write.
        subject + channel are the (sub-)study's independent variables — eval/practice_report.py groups
        by them. Rewrites the whole (small) log each attempt so an older header (pre-subject/channel)
        is migrated by back-filling blanks; atomic via a temp file + replace."""
        try:
            import csv, datetime
            e = entry or {}
            path = RECORDINGS_DIR / "practice_log.csv"
            cols = ["time", "part", "key", "scale", "voiced_s",
                    "match_pct", "mean_abs_cents", "on_target_s", "subject", "channel"]
            row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
                   "part": e.get("part", ""), "key": e.get("key", ""), "scale": e.get("scale", ""),
                   "voiced_s": round(float(e.get("voiced_s", 0)), 1),
                   "match_pct": round(float(e.get("match_pct", 0)), 3),
                   "mean_abs_cents": round(float(e.get("mean_abs_cents", 0)), 1),
                   "on_target_s": round(float(e.get("on_target_s", 0)), 1),
                   "subject": e.get("subject", ""), "channel": e.get("channel", "")}
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            prior = []
            if path.exists():
                with open(path, newline="") as f:
                    prior = list(csv.DictReader(f))   # migrates an older header by reading by name
            tmp = path.with_name("practice_log.csv.tmp")
            with open(tmp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in prior:
                    w.writerow({c: r.get(c, "") for c in cols})
                w.writerow(row)
            tmp.replace(path)
            return str(path)
        except Exception:  # noqa: BLE001 — logging must never crash the bridge
            return False

    def reveal(self, path):
        """Reveal a saved take in Finder (macOS). Restricted to the recordings dir for safety."""
        try:
            p = Path(path).resolve()
            if p.exists() and RECORDINGS_DIR.resolve() in p.parents:
                subprocess.run(["open", "-R", str(p)], check=False)
                return True
        except Exception:  # noqa: BLE001 — bad path / open failed; never crash the bridge
            pass
        return False

    def render_satb(self, stems_path):
        """Offline-render a take's raw 'You' melody through the active SATB voices.
        Writes <take>_satb.wav (mix) + <take>_satb_<part>.wav (per active part) to
        recordings/. Async; pushes window.onRenderStatus({state,...})."""
        threading.Thread(target=self._render_satb_worker, args=(str(stems_path),),
                         daemon=True).start()
        return True

    def _render_satb_worker(self, stems_path):
        import soundfile as sf  # local import: render path only, keeps cold start lean
        try:
            stems = Path(stems_path).resolve()
            if not (stems.exists() and RECORDINGS_DIR.resolve() in stems.parents):
                self._push_render_status({"state": "failed", "why": "bad take path"}); return
            self._push_render_status({"state": "rendering"})
            # --render averages all input channels (rx.mean(axis=1)); the stems file is
            # stereo L=melody R=choir, so extract L → a temp mono melody for a clean source.
            data, sr = sf.read(str(stems), dtype="float32", always_2d=True)
            mel = data[:, 0]
            base = stems.name.replace("_stems.wav", "")          # take_YYYYmmdd_HHMMSS
            # the dry You stem; doubles as the render SOURCE. Named _satb_you so it completes the
            # coherent 5-stem package (Bass/Tenor/Alto/Sop + You) when dragged into a DAW.
            mel_path = RECORDINGS_DIR / f"{base}_satb_you.wav"
            sf.write(str(mel_path), mel, sr)
            # render the CURRENT live voices (what's on screen — Bass/Tenor/Sop on jvs + the female
            # Alto), so the bounce matches the choir the user hears, not the separate SATB_VOICES
            # preset. `model` survives on self.control["voices"] (re-stamped in set_control); each
            # voice routes to its own model offline (jvs/female/tenor). Fall back to the JVS default.
            voices = [dict(v) for v in (self.control.get("voices") or JVS_SATB_VOICES)]
            active = [v for v in voices if v.get("on")]
            outputs = []
            # per active part: solo render (only that voice on), harmony only (--no-you)
            for v in active:
                solo = [dict(x, on=(x["part"] == v["part"])) for x in voices]
                out = RECORDINGS_DIR / f"{base}_satb_{v['part'].lower()}.wav"
                self._render_once(mel_path, out, solo, no_you=True)
                outputs.append(str(out))
            # full mix (all active on, with You) for instant in-app playback
            mix_out = RECORDINGS_DIR / f"{base}_satb.wav"
            self._render_once(mel_path, mix_out, voices, no_you=False)
            # `you` = the dry melody stem, so the UI can offer "You" as a togglable part alongside SATB.
            self._push_render_status({"state": "done", "mix": str(mix_out), "stems": outputs, "you": str(mel_path)})
        except Exception as e:  # noqa: BLE001 — never crash the bridge
            self._push_render_status({"state": "failed", "why": str(e)})

    def _render_once(self, in_path, out_path, voices, no_you):
        # carry the take's key/scale into the render so the diatonic harmony matches the key
        # it was sung in. Takes store no key metadata, so self.control is the only source —
        # same assumption the live engine uses (it reflects the take's key unless changed since).
        c = self.control
        cmd = [sys.executable, "beatrice_solo_choir_live.py",
               "--model", self.model,
               "--render", str(in_path), str(out_path),
               "--key", KEYS[int(c.get("key", 0)) % 12],
               "--minor", "1" if c.get("scale") == "minor" else "0",
               "--voices", json.dumps(voices)]
        if no_you:
            cmd += ["--no-you"]
        r = subprocess.run(cmd, cwd=str(SERVER_DIR))
        if r.returncode != 0:
            raise RuntimeError(f"render exited {r.returncode}: {str(out_path)}")

    def _push_render_status(self, d):
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onRenderStatus && window.onRenderStatus({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def export_midi(self, stems_path):
        """Export a take's melody + diatonic SATB harmony as a multi-track .mid.
        Async; pushes window.onMidiStatus({state,...})."""
        threading.Thread(target=self._export_midi_worker, args=(str(stems_path),),
                         daemon=True).start()
        return True

    def _export_midi_worker(self, stems_path):
        import soundfile as sf
        if str(SERVER_DIR) not in sys.path:           # import score_export from server/
            sys.path.insert(0, str(SERVER_DIR))
        try:
            import score_export
            stems = Path(stems_path).resolve()
            if not (stems.exists() and RECORDINGS_DIR.resolve() in stems.parents):
                self._push_midi_status({"state": "failed", "why": "bad take path"}); return
            self._push_midi_status({"state": "exporting"})
            base = stems.name.replace("_stems.wav", "")
            mel_path = RECORDINGS_DIR / f"{base}_satb_you.wav"
            if not mel_path.exists():                 # works even without a prior audio render
                data, sr = sf.read(str(stems), dtype="float32", always_2d=True)
                sf.write(str(mel_path), data[:, 0], sr)
            c = self.control
            key_root, minor = int(c.get("key", 0)) % 12, (c.get("scale") == "minor")
            notes = score_export.wav_to_notes(str(mel_path), key_root, minor)   # melody snapped in-key
            voices = [v for v in (c.get("voices") or JVS_SATB_VOICES) if v.get("on")]
            tracks = [("You", 52, notes)]             # 52 = GM Choir Aahs
            for v in voices:
                part = [(s, e, score_export.harmony_note(m, key_root, minor,
                          int(v.get("interval", 0)), int(v.get("octave", 0))))
                        for (s, e, m) in notes]
                tracks.append((v["part"], 52, part))
            out = RECORDINGS_DIR / f"{base}_satb.mid"
            score_export.write_midi(str(out), tracks)
            self._push_midi_status({"state": "done", "midi": str(out)})
        except Exception as e:  # noqa: BLE001 — never crash the bridge
            self._push_midi_status({"state": "failed", "why": str(e)})

    def _push_midi_status(self, d):
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onMidiStatus && window.onMidiStatus({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def arrange_take(self, stems_path):
        """Auto-arrange a take's melody into a voice-led SATB wav (offline, server/arrange.py).
        Async; pushes window.onArrangeStatus({state,...})."""
        threading.Thread(target=self._arrange_worker, args=(str(stems_path),), daemon=True).start()
        return True

    def _arrange_worker(self, stems_path):
        import soundfile as sf  # local import: arrange path only
        try:
            stems = Path(stems_path).resolve()
            if not (stems.exists() and RECORDINGS_DIR.resolve() in stems.parents):
                self._push_arrange_status({"state": "failed", "why": "bad take path"}); return
            self._push_arrange_status({"state": "arranging"})
            data, sr = sf.read(str(stems), dtype="float32", always_2d=True)
            base = stems.name.replace("_stems.wav", "")
            mel_path = RECORDINGS_DIR / f"{base}_arr_mel.wav"
            sf.write(str(mel_path), data[:, 0], sr)          # L = dry melody = the arrange source
            # auto-detect key; tight + the user's own voice are arrange.py's validated defaults
            r = subprocess.run([sys.executable, "arrange.py", str(mel_path), "--key", "auto"],
                               cwd=str(SERVER_DIR))
            if r.returncode != 0:
                raise RuntimeError(f"arrange exited {r.returncode}")
            out = RECORDINGS_DIR / f"{base}_arr_mel_arranged.wav"
            self._push_arrange_status({"state": "done", "arranged": str(out)})
        except Exception as e:  # noqa: BLE001 — never crash the bridge
            self._push_arrange_status({"state": "failed", "why": str(e)})

    def _push_arrange_status(self, d):
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onArrangeStatus && window.onArrangeStatus({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def _send_json(self, d: dict) -> None:
        eng = self._engine
        if eng and eng.stdin and eng.poll() is None:
            try:
                eng.stdin.write(json.dumps(d) + "\n")
                eng.stdin.flush()
            except Exception:  # noqa: BLE001 — engine closing
                pass

    def _send_control(self) -> None:
        self._send_json(self.control)

    def list_devices(self):
        """Real input/output device names + the currently selected ones (for the UI dropdowns)."""
        ins, outs = self._device_names()
        return {"inputs": ins, "outputs": outs, "in_name": self.in_name, "out_name": self.out_name}

    def set_devices(self, in_name: str | None, out_name: str | None):
        """UI picked a device → relaunch the engine on it (stop old, start new)."""
        changed = (in_name and in_name != self.in_name) or (out_name and out_name != self.out_name)
        if in_name:
            self.in_name = in_name
        if out_name:
            self.out_name = out_name
        if changed and self._engine is not None:   # engine already running → re-route
            self._restart_engine()
        return {"in_name": self.in_name, "out_name": self.out_name}

    def list_models(self):
        """Selectable models = paraphernalia_data_* dirs in MODELS_ROOT (+ the current one)."""
        names = []
        if MODELS_ROOT.exists():
            names = sorted(p.name for p in MODELS_ROOT.iterdir()
                           if p.name.startswith("paraphernalia_data_") and _is_model_dir(p))
        cur = Path(self.model).name
        if cur not in names:
            names.append(cur)
        return {"models": names, "current": cur}

    def set_model(self, name: str | None):
        """UI picked a model → relaunch the engine on it (stop old, start new). Mirrors set_devices."""
        if name:
            cand = (MODELS_ROOT / name).resolve()
            if MODELS_ROOT.resolve() in cand.parents and _is_model_dir(cand) and str(cand) != self.model:
                self.model = str(cand)
                if self._engine is not None:        # engine already running → reload on the new model
                    self._restart_engine()
        return {"current": Path(self.model).name}

    # ===== MIDI input (exposed to JS) =====
    def list_midi(self):
        """UI MIDI panel — available input ports + the connected one + current mode."""
        names = []
        if _MIDI_OK:
            try:
                names = list(mido.get_input_names())
            except Exception:  # noqa: BLE001
                names = []
        return {"available": _MIDI_OK, "devices": names,
                "connected": self._midi_name, "mode": self.midi_mode}

    def set_midi_device(self, name):
        """UI picked a MIDI port ('' = '— not connected —' → disconnect, do NOT auto-pick)."""
        clean = (name or "").strip()
        if clean:
            self._open_midi(clean)
        else:
            self._close_midi()         # explicit disconnect → engine gets {"on":False} → diatonic choir
        return {"connected": self._midi_name}

    def set_midi_mode(self, mode):
        """UI A/B toggle. 'key' = Mode B (white key → custom chord); else Mode A (chord-hold)."""
        self.midi_mode = "key" if mode == "key" else "chord"
        self._emit_midi_control()      # re-emit the held notes under the new mode
        return {"mode": self.midi_mode}

    def get_midi_chords(self):
        """Mode B editor — the 7 white-key chord slots (ordered) + the quality choices."""
        slots = [{"pc": pc, "root": self.midi_chords[pc]["root"],
                  "quality": self.midi_chords[pc]["quality"]}
                 for pc in _WHITE_PCS if pc in self.midi_chords]
        return {"slots": slots, "qualities": list(_CHORD_QUALITIES.keys())}

    def set_midi_chords(self, slots):
        """UI edited a slot → update + persist. Each slot: {pc, root, quality}."""
        for s in (slots or []):
            pc = int(s["pc"]) % 12
            if pc in self.midi_chords:
                q = str(s.get("quality", "maj"))
                self.midi_chords[pc] = {"root": int(s["root"]) % 12,
                                        "quality": q if q in _CHORD_QUALITIES else "maj"}
        self._save_midi_chords()
        self._emit_midi_control()      # if a key is held, reflect the edit immediately
        return True

    def _load_midi_chords(self) -> dict:
        try:
            if MIDI_CHORDS_PATH.exists():
                raw = json.loads(MIDI_CHORDS_PATH.read_text())
                m = {int(k): {"root": int(v["root"]) % 12, "quality": str(v["quality"])}
                     for k, v in raw.items() if int(k) in _WHITE_PCS}
                if m:
                    return {**{k: dict(v) for k, v in _DEFAULT_CHORDS.items()}, **m}
        except Exception:  # noqa: BLE001 — bad/missing file → defaults
            pass
        return {k: dict(v) for k, v in _DEFAULT_CHORDS.items()}

    def _save_midi_chords(self) -> None:
        try:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            MIDI_CHORDS_PATH.write_text(
                json.dumps({str(k): v for k, v in self.midi_chords.items()}))
        except Exception:  # noqa: BLE001 — disk/path issue; never crash the bridge
            pass

    def _open_midi(self, name: str | None) -> None:
        self._close_midi()
        if not _MIDI_OK:
            return
        try:
            names = list(mido.get_input_names())
        except Exception:  # noqa: BLE001
            names = []
        if not name:   # auto-pick the Launchkey NOTE port (prefer "MIDI", avoid the "DAW" surface)
            low = [(n, n.lower()) for n in names]
            name = (next((n for n, l in low if ("launchkey" in l or "mini" in l)
                          and "midi" in l and "daw" not in l), None)
                    or next((n for n, l in low if "launchkey" in l or "mini" in l), None)
                    or (names[0] if names else None))
        if not name:
            return
        try:
            self._midi_port = mido.open_input(name)
        except Exception as e:  # noqa: BLE001 — bad/busy port; feature stays off
            print(f"[bridge] midi open failed: {e}", file=sys.stderr)
            return
        self._midi_name = name
        self._midi_held = set()
        self._midi_stop = False
        self._midi_thread = threading.Thread(target=self._midi_loop, daemon=True)
        self._midi_thread.start()
        print(f"[bridge] midi in: {name!r}", file=sys.stderr)

    def _midi_loop(self) -> None:
        port = self._midi_port
        if port is None:
            return
        try:
            for msg in port:                       # blocks per message; exits when port closes
                if self._midi_stop:
                    break
                if msg.type == "note_on" and msg.velocity > 0:
                    self._midi_held.add(msg.note)
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    self._midi_held.discard(msg.note)
                else:
                    continue                       # ignore CC / clock / aftertouch
                self._emit_midi_control()
        except Exception:  # noqa: BLE001 — port closed / device unplugged
            pass

    def _emit_midi_control(self) -> None:
        held = sorted(self._midi_held)
        if self.midi_mode == "key":
            # Mode B: lowest held WHITE key → its custom chord slot, sent as a chord (reuses the
            # Mode A engine path → engine unchanged). Black keys / unset slots → silent. Release
            # → empty → silent.
            slot = self.midi_chords.get(held[0] % 12) if held else None
            notes = _build_chord(slot["root"], slot["quality"]) if slot else []
        else:                                      # Mode A: the held notes ARE the chord
            notes = held
        self.control["midi"] = {"on": True, "mode": "chord", "notes": notes}
        self._send_control()
        self._push_midi_ui(notes)                  # show the actual chord the choir will sing

    def _close_midi(self) -> None:
        self._midi_stop = True
        p, self._midi_port = self._midi_port, None
        if p is not None:
            try:
                p.close()
            except Exception:  # noqa: BLE001
                pass
        if self._midi_name is not None:
            self._midi_name = None
            self.control["midi"] = {"on": False}   # tell the engine MIDI is off → diatonic resumes
            self._send_control()
        self._midi_held = set()

    def _push_midi_ui(self, notes) -> None:
        if self._closed or self._win is None:
            return
        d = {"available": _MIDI_OK, "connected": self._midi_name,
             "mode": self.midi_mode, "notes": list(notes)}
        try:
            self._win.evaluate_js(f"window.onMidi && window.onMidi({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    # ===== lifecycle (underscored → not exposed to JS) =====
    def _attach(self, window) -> None:
        self._win = window

    def _on_closed(self) -> None:
        self._closed = True
        self._close_midi()
        self._stop_engine()

    def _start(self) -> None:
        self._pick_default_devices()
        self._start_engine()
        self._open_midi(None)          # auto-pick a connected Launchkey; no device → no-op
        self._telemetry_loop()

    # ===== devices =====
    def _device_names(self) -> tuple[list[str], list[str]]:
        try:
            import sounddevice as sd
            # refresh PortAudio's cached device list so hot-plugged devices appear without a
            # restart. Safe here: the GUI process holds no audio stream (engine is a subprocess).
            try:
                sd._terminate(); sd._initialize()
            except Exception:  # noqa: BLE001
                pass
            devs = sd.query_devices()
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] query_devices failed: {e}", file=sys.stderr)
            return [], []

        def uniq(seq):
            seen, out = set(), []
            for n in seq:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
            return out

        ins = uniq(d["name"] for d in devs if d.get("max_input_channels", 0) > 0)
        outs = uniq(d["name"] for d in devs if d.get("max_output_channels", 0) > 0)
        return ins, outs

    def _pick_default_devices(self) -> None:
        ins, outs = self._device_names()

        def prefer(names, want, avoid=()):
            for n in names:
                low = n.lower()
                if any(w in low for w in want) and not any(a in low for a in avoid):
                    return n
            return None

        def first_not(names, avoid):
            return next((n for n in names if not any(a in n.lower() for a in avoid)), None)

        # PREFER an Aggregate Device on BOTH in and out: one clock domain → no 'input overflow'
        # drift and low latency hold even with multi-voice (SATB). This is the single biggest
        # stability win — without it, USB-mic + a separate output are two clocks and overflow,
        # which cascades into crackle + flickering telemetry + UI jitter.
        # "speaker_set" = this rig's Aggregate Device (it isn't named "Aggregate"); a single Core
        # Audio clock domain, verified healthy for live 4-voice (RTF ~0.45, zero 'input overflow').
        # Chinese entries are the localised macOS name for an Aggregate Device.
        # Match targets, not prose: do not translate.
        agg_hints = ("aggregate", "聚集", "聚合", "speaker_set")
        agg_in, agg_out = prefer(ins, agg_hints), prefer(outs, agg_hints)
        if agg_in and agg_out:
            self.in_name, self.out_name = agg_in, agg_out
            print(f"[bridge] using Aggregate Device for in+out: {agg_in!r}", file=sys.stderr)
            return
        # input: prefer the USB PnP mic, else first available
        self.in_name = prefer(ins, ("usb pnp", "usb")) or (ins[0] if ins else None)
        # output: prefer headphones/bone-conduction; NEVER the built-in speaker (feedback)
        # "耳機" is the localised macOS word for headphones. Match target, not prose.
        self.out_name = (prefer(outs, ("headphone", "耳機", "bone", "airpod", "bluetooth", "blackhole"))
                         or first_not(outs, SPEAKER_HINTS) or (outs[0] if outs else None))
        print(f"[bridge] default devices: in={self.in_name!r} out={self.out_name!r} "
              f"(no Aggregate Device found — two clocks may overflow; make one for stable multi-voice)",
              file=sys.stderr)

    # ===== real engine (existing harness CLI; read-only telemetry) =====
    def _start_engine(self) -> None:
        c = self.control
        interval = 0 if self.overdub else (c.get("intervals") or [-2])[0]   # overdub: sing the actual note (no shift)
        cmd = [
            sys.executable, "beatrice_solo_choir_live.py",
            "--model", self.model, "--emit-telemetry",
            "--key", KEYS[int(c.get("key", 0)) % 12],
            "--minor", "1" if c.get("scale") == "minor" else "0",
            "--interval", str(int(interval)),
            "--latency", str(self.latency),
        ]
        if not self.overdub:   # Task 2.2: chord-aware (voice-leading) harmony is the live default; overdub sings the actual note (no snap)
            cmd += ["--auto-chord"]
            cmd += ["--pitch-glide", "0.04"]   # smooth per-note pitch transitions (kills the scrape; vcclient's clean pitch = this fine ramp). --render omits it → byte-identical

        if self.overdub:   # overdub: ONE voice, Route A pump. ~56ms: cushion 5 + latency 0.005 (the device
            cmd += ["--no-you", "--pump", "--cushion-ms", "5", "--blocksize", "240", "--latency", "0.005"]  # buffer at 0.005 is more forgiving than the 0.004 that dropped the loop while recording — test by ear).
        elif self.lowlat:  # low-latency solo: You + ONE converted voice (engine default 1 voice), Route A pump +
            cmd += ["--pump", "--cushion-ms", "3", "--blocksize", "240", "--latency", "0.004"]  # small buffers → ~46ms on speaker_set (measured); later --latency wins over the one above
        elif self.satb:   # choir mode: satb2 single-model multi-voice + no-You + Route A pump (see __init__)
            cmd += ["--voices", json.dumps(c.get("voices") or CHOIR_VOICES), "--no-you",
                    "--pump", "--cushion-ms", "30"]   # 30ms prebuffer absorbs GUI-contention bursts
        if self.in_name:
            cmd += ["--in-name", self.in_name]
        if self.out_name:
            cmd += ["--out-name", self.out_name]
        self._engine_gen += 1
        gen = self._engine_gen
        try:
            self._engine = subprocess.Popen(
                cmd, cwd=str(SERVER_DIR),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1,
            )
            self._engine_alive = True
            self._stop_reason = ""
            threading.Thread(target=self._read_engine, args=(gen,), daemon=True).start()
            # carry live control to the new engine (e.g. after a device restart). On the very
            # first launch nothing is dirty → engine runs the launch flags == 3b baseline.
            if self._control_dirty:
                self._send_control()
            print(f"[bridge] engine gen{gen}: in={self.in_name!r} out={self.out_name!r}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] engine launch failed: {e}", file=sys.stderr)

    def _read_engine(self, gen: int) -> None:
        eng = self._engine
        if not eng or not eng.stdout:
            return
        for line in eng.stdout:               # blocks per line; daemon thread
            if gen != self._engine_gen:        # superseded by a device restart
                return
            line = line.strip()
            if not line or line[0] != "{":     # skip the harness's plain log lines
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "f0" in d:
                with self._lock:
                    self._engine_tel = d
            elif "saved" in d:                 # engine finalized a take → tell the UI
                self._push_saved(d)
            elif "od_saved" in d:              # engine exported an overdub stack → tell the UI
                self._push_overdub_saved(d)
            elif "save_failed" in d:           # engine failed to write a take → UI shows 'Save failed'
                self._push_save_failed(d)
        # the stdout iterator ended → the engine process exited. If this generation is still current
        # (NOT superseded by a deliberate device/model restart), it died unexpectedly → flag it so the
        # telemetry loop reports STOPPED and the UI offers Reconnect.
        if gen == self._engine_gen and not self._closed:
            rc = eng.poll()
            self._engine_alive = False
            self._stop_reason = ("audio device unavailable / sample-rate mismatch"
                                 if rc not in (0, None) else "engine stopped")

    def _push_saved(self, d: dict) -> None:
        """Engine reported a finalized take → UI shows 'Saved ✓ …' + a Reveal button."""
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onSaved && window.onSaved({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def _push_overdub_saved(self, d: dict) -> None:
        """Engine exported an overdub stack → UI shows 'Saved ✓ …'."""
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onOverdubSaved && window.onOverdubSaved({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def _push_save_failed(self, d: dict) -> None:
        """Engine failed to write a take → UI shows 'Save failed' (no false Saved/Reveal)."""
        if self._closed or self._win is None:
            return
        try:
            self._win.evaluate_js(f"window.onSaveFailed && window.onSaveFailed({json.dumps(d)})")
        except Exception:  # noqa: BLE001 — window closing
            pass

    def _stop_engine(self) -> None:
        # terminate() is ASYNC (SIGTERM returns immediately). We MUST block until the old engine
        # has actually exited and released its duplex audio stream — otherwise _start_engine opens
        # a SECOND stream on the same device while the old one still holds it → the two contend and
        # PortAudio comes up in a permanent 'input overflow' state that survives until a clean
        # restart. (This is why a device change or a delay 'fixed' it: the old engine had time to
        # die / the new stream landed on a different device.) Bounded wait, then hard kill.
        eng = self._engine
        self._engine = None
        if eng and eng.poll() is None:
            eng.terminate()
            try:
                eng.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                eng.kill()
                try:
                    eng.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

    def _restart_engine(self) -> None:
        self._stop_engine()
        with self._lock:
            self._engine_tel = None            # drop stale telemetry from the old device
        self._start_engine()

    # ===== telemetry push (engine -> UI), mapped into the unchanged JS contract =====
    def _telemetry_loop(self) -> None:
        period = 1.0 / TELEMETRY_HZ
        while not self._closed and self._win is not None:
            with self._lock:
                e = dict(self._engine_tel) if self._engine_tel else None
            tel = self._to_contract(e)
            tel["engine_alive"] = self._engine_alive
            tel["stop_reason"] = self._stop_reason
            try:
                self._win.evaluate_js(
                    f"window.onTelemetry && window.onTelemetry({json.dumps(tel)})"
                )
            except Exception:  # noqa: BLE001 — window closing
                pass
            time.sleep(period)

    @staticmethod
    def _to_contract(e: dict | None) -> dict:
        """Map raw engine telemetry -> the §2 JS contract (+ Track A play_t/play_len for the playhead)."""
        level = round(min(1.0, float(e["in_level"])), 3) if e and "in_level" in e else 0.0
        rtf = round(float(e.get("rtf", 0.0)), 3) if e else 0.0           # real per-block compute load
        latency_ms = round(float(e.get("latency_ms", 0.0)), 1) if e else 0.0  # real est. mouth-to-ear
        # Track A: real playback position (seconds into the loop) + loop length so the UI playhead can
        # track the engine exactly. play_t < 0 → the engine is not looping a take/stem right now.
        play_t = round(float(e.get("play_t", -1.0)), 3) if e else -1.0
        play_len = round(float(e.get("play_len", 0.0)), 3) if e else 0.0
        # overdub playhead/layer count (0/absent when not in overdub mode → UI panel just stays idle)
        od = {"od_count": int(e.get("od_count", 0)) if e else 0,
              "od_rec": bool(e.get("od_rec", False)) if e else False,
              "od_t": round(float(e.get("od_t", 0.0)), 3) if e else 0.0,
              "od_len_s": round(float(e.get("od_len_s", 0.0)), 3) if e else 0.0}
        voices = (e.get("voices") if e else None) or []   # per-voice honest note names (Task 1.2)
        if not e or not e.get("voiced") or float(e.get("f0", 0.0)) <= 0.0:
            return {"f0": 0.0, "midi": 0.0, "cents": 0.0, "level": level,
                    "rtf": rtf, "latency_ms": latency_ms, "voiced": False,
                    "play_t": play_t, "play_len": play_len, "voices": voices, **od}
        f0 = float(e["f0"])
        midi = _f0_to_midi(f0)
        nearest = round(midi)
        return {"f0": round(f0, 2), "midi": round(midi, 3),
                "cents": round((midi - nearest) * 100.0, 1), "level": level,
                "rtf": rtf, "latency_ms": latency_ms, "voiced": True,
                "play_t": play_t, "play_len": play_len, "voices": voices, **od}
