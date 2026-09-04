"""respond_shell — pywebview shell: the exhibition and test interface for the
answering mode (respond2).

The lesson from 2026-08-02 §I: calibration and status printed only to the
console are invisible to someone standing up and singing. This shell turns
respond2's stdout into large on-screen status (whether you may sing,
calibration instructions, phrase count, waiting, warnings). The engine runs as
a subprocess, following the architecture in CLAUDE.md: respond2 lives in the
ddsp venv, the shell in vcclient-dev, all audio stays inside the subprocess and
the bridge carries text only. respond2's behaviour is unchanged; what is parsed
here is the same output a person reads, plus one extra marker line
("captured → rendering").

Run (conda env vcclient-dev):  python app/respond_shell.py
"""
from __future__ import annotations
import json
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import webview

REPO_ROOT = Path(__file__).resolve().parent.parent
HARMONY = REPO_ROOT / "harmony"
UI_FILE = REPO_ROOT / "ui" / "respond_live.html"
# respond2 lives in the ddsp venv (vcclient-dev has no parselmouth, worklog 2026-08-04 §J)
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO_ROOT))
import config  # noqa: E402
ENGINE_PY = config.PY_RESPOND
# bank_live lives in the 6x venv (the certified launch path in the runbook)
BANK_PY = config.PY_ENGINE
# solo_min (the neural live mode, 2026-08-14) runs in the 6x venv, the same one
# as bank_live. It used conda vcclient-dev until 08-14, when it moved: the mouth
# gate needs cv2 and mediapipe, which exist only in the 6x venv, and installing
# them into vcclient-dev would drag numpy from 1.23.5 to 2.x. The certified
# environment is not touched before the viva. Measured: Beatrice in the 6x venv
# is bit-identical (same input, rms 0.05540 on both sides, RTF 0.116 vs 0.122);
# both are Python 3.10.20.
SOLO_PY = BANK_PY
SERVER = REPO_ROOT / "server"
# The two voices settled by ear on 2026-08-14: the upper part is the B-line
# self-trained Soprano-3 (bt, 3000 steps, judged "stop at 3000"), the lower part
# is the tenor that has been in use since June.
SOLO_SOP = config.BEATRICE_MODELS / "paraphernalia_data_00003000"
SOLO_BASS = config.BEATRICE_MODELS / "paraphernalia_data_new25_2k"
SOLO_ALT = config.BEATRICE_MODELS / "paraphernalia_data_satb2"

# stdout to UI events. Only the human-readable lines are matched; anything
# unrecognised still goes to the log panel.
_PATTERNS = [
    ("cal_quiet", re.compile(r"^1\) Calibration: stay silent")),
    ("cal_sing",  re.compile(r"^2\) Calibration: sing")),
    ("sep",       re.compile(r"separation ([\d.]+) dB")),
    # Printed only when calibration succeeds; failure prints a different line.
    # The value is captured so that a later mode switch can pass it back with
    # `--gate`, which means switching into the answering mode mid-performance
    # does not require calibrating again. Calibration asks the singer for three
    # seconds of silence and then five seconds of singing, which cannot be done
    # on stage.
    # Review #16, 2026-08-17: scientific notation is allowed. On a quiet chain
    # the gate can be of the order 1e-05, so the engine now prints %.6g; both
    # 0.0424 and 1.2e-05 are captured whole. The old %.4f printed a small gate
    # as 0.0000, and reusing that as --gate 0.0 makes lv>0 always true, so the
    # phrase never ends.
    ("calgate",   re.compile(r"→ gate ([\d.eE+-]+) \(")),
    ("key",       re.compile(r"^\[key\] (.+)")),
    ("octave",    re.compile(r"^\[auto-octave\] (.+)")),
    ("live",      re.compile(r"^respond2 live\.")),
    ("listen",    re.compile(r"^listening +([\d.]+)s .*phrase (\d+).*io (\d+)/(\d+)")),
    ("render",    re.compile(r"^  captured → rendering")),
    ("phrase",    re.compile(
        r"^  phrase (\d+) .*response +([\d.]+)s .*wait +([\d.]+)s"
        r".*mute window +([\d.]+)s")),
    ("warn",      re.compile(r"^\s*⚠ (.+)")),
    # Button status (connected / disconnected / fallen back to UDP). If it is
    # not on the status bar it may as well not exist: someone standing and
    # singing cannot read a 10.5px log panel (review A3/B7, 2026-08-12).
    # Failure lines carry a warning prefix from the engine and are caught by the
    # warn pattern above, which raises the large warning bar.
    ("tap",       re.compile(r"^\[tap\] (.+)")),
    ("dump",      re.compile(r"^session dump: (.+?) \(")),
    ("done",      re.compile(r"^total (\d+) phrases \| (.+)")),
    # The live streaming mode (spike_stream, 2026-08-05 §M)
    ("stream_on", re.compile(r"^stream on ")),
    ("stream_st", re.compile(
        r"^infer p50 (\d+)ms.*under (\d+) flags (\d+)")),
    # Live mode v2 is bank_live, which replaced spike_stream on 2026-08-12.
    # A dead gate risks feedback in the both mode, so it must raise the large
    # warning bar and not sit in the 10.5px log (review A4).
    ("warn",      re.compile(r"^⚠? ?\[mouth\] GATE DEAD")),
    # The neural live mode (solo_min, 2026-08-14). solo_min prints these lines
    # to stderr, but Popen merges stderr into stdout, so they arrive here and
    # solo_min itself needs no change.
    ("solo_on",   re.compile(r"^\[solo_min\] running")),
    ("solo_st",   re.compile(r"^\[solo_min\] xruns (\d+)")),
    ("bank_on",   re.compile(r"^ready \(sample bank")),
    # backlog and dropped are optional groups. dropped is the only instrument
    # that sees audio being lost (F25); the earlier regex did not capture it, so
    # a 35 ms hole under duo saturation was audible but invisible (review #4,
    # 2026-08-17). The line may also end with GATE DEAD or WORKER DEAD, which
    # the UI sniffs separately.
    ("bank_st",   re.compile(
        r"^note (\S+) +cents +([+\-][\d.]+) +mouth +(open|closed) +xrun (\d+)"
        r"(?: +backlog (\d+)(?:/dropped (\d+))?)?")),
]


TAP_PORT = 8766     # tap_listen.TapListener port; respond2 falls back to this
                    # when the serial button is not plugged in


class Bridge:
    def __init__(self):
        self._win = None
        self._proc: subprocess.Popen | None = None
        self._proc2: subprocess.Popen | None = None   # bank_live for the both mode
        self._both_pending: dict | None = None        # bank starts only after respond2 has calibrated
        self._tap_seq = 0        # same protocol as the firmware: rising sequence, de-duplicated at the receiver
        self._tap_on = False     # heartbeat is sent only while the window has focus (blur = button disconnected)
        self._stopping = False   # SIGINT already sent (guards against double-clicking Stop)
        self._stop_t = 0.0       # time of the last SIGINT, for the guard window
        self._lock = threading.Lock()   # atomic take of _both_pending (review S2)
        self._mode = None        # the mode currently sounding (used by keyboard switching)
        self._opts: dict = {}    # the devices and parameters chosen at Start, reused across switches
        self._switch: dict | None = None   # a switch in progress: {mode, old[], new}
        self._gate: str | None = None      # the phrase-end gate from this run's successful calibration, reused on switch
        self._octave: int | None = None    # the octave locked this run, pinned with --octave on respawn
        self._key_shift: int | None = None  # the latest key shift, seeded back with --key-seed on respawn
        self._retired: list = []           # engines sent SIGINT but not yet confirmed dead
        self._retired_procs: set = set()   # engines retired on purpose, so their exit event is downgraded
        threading.Thread(target=self._hb_loop, daemon=True).start()
        threading.Thread(target=self._ble_loop, daemon=True).start()

    def _ble_loop(self):
        """Receive the physical button over BLE.

        The Pico firmware is firmware/pico_led_button/main_ble.py: a custom
        notification service that needs no pairing. The BLE HID keyboard route
        died because the official rp2 firmware is built without pairing; see
        GRAVEYARD. A received TAP calls self.tap(), the same path and the same
        running number as the UI space key, so the two cannot swallow each
        other. Disconnection triggers a rescan and reconnect.
        Nothing blocks: if bleak is missing, Bluetooth permission is refused, or
        the button is absent, the thread exits quietly and the app runs on.
        """
        try:
            import asyncio

            from bleak import BleakClient, BleakScanner
        except ImportError:
            return
        SVC = "0f9a0001-1e0e-4c7a-9a4e-534f4c4f4348"
        TAP_C = "0f9a0002-1e0e-4c7a-9a4e-534f4c4f4348"

        async def run():
            while True:
                try:
                    dev = await BleakScanner.find_device_by_filter(
                        lambda d, a: SVC in (a.service_uuids or []),
                        timeout=10.0)
                    if dev is None:
                        await asyncio.sleep(3)
                        continue
                    lost = asyncio.Event()
                    async with BleakClient(
                            dev, disconnected_callback=lambda c: lost.set()
                    ) as cl:
                        self._push("tap", {"m": [
                            "connected (BLE key) — button or SPACE ends "
                            "the phrase"]})

                        def _cb(_h, data):
                            if bytes(data).startswith(b"TAP"):
                                self.tap()
                        await cl.start_notify(TAP_C, _cb)
                        await lost.wait()
                    self._push("tap", {"m": ["disconnected (BLE key) — "
                                             "SPACE still works"]})
                except Exception:  # noqa: BLE001 - Bluetooth off, permission refused, or device gone
                    await asyncio.sleep(5)

        try:
            asyncio.new_event_loop().run_until_complete(run())
        except Exception:  # noqa: BLE001 - catch-all: a dead BLE layer must not take the app down
            pass

    def _tap_send(self, msg):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(msg, ("127.0.0.1", TAP_PORT))
            s.close()
        except OSError:
            pass                 # nothing blocks: if it cannot be sent, treat the button as absent

    def _hb_loop(self):
        while True:
            if self._tap_on and self.running():
                self._tap_send(b"HB 0")
            time.sleep(2.0)

    def tap(self):
        """The UI space key acts as the physical button: send TAP three times
        for redundancy; the receiver de-duplicates by sequence number."""
        self._tap_seq += 1
        for _ in range(3):
            self._tap_send(b"TAP %d" % self._tap_seq)
        return True

    def tap_presence(self, on):
        """UI focus and blur turn the heartbeat on and off. After 6 s without a
        heartbeat the engine falls back to energy-only phrase detection."""
        self._tap_on = bool(on)
        return True

    def _attach(self, window):
        self._win = window

    def _push(self, kind, payload):
        try:
            if self._win:
                self._win.evaluate_js(
                    "window.respTele&&window.respTele(%s)"
                    % json.dumps({"kind": kind, **payload}, ensure_ascii=False))
        except Exception:  # noqa: BLE001 - the window is closing
            pass

    def _reader(self, proc, who):
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # 2026-08-17: an engine that is retiring, or being switched away
            # from, feeds the log only and no longer drives the UI. Once exit
            # was downgraded, status lines from the old engine inside its <=8 s
            # shutdown window still pushed to the screen, which froze a stale
            # camera image after switching away from bank. By the same
            # argument, the old engine should leave the screen the moment the
            # switch is requested: the sound continues seamlessly, but the UI
            # follows the performer's intent rather than the old engine.
            sw0 = self._switch
            if (proc in self._retired_procs
                    or (sw0 and proc in sw0.get("old", []))):
                if not line.startswith("FRAME "):
                    self._push("log", {"line": line})
                continue
            if line.startswith("FRAME "):
                # 2026-08-18: the camera view was removed from the UI, so
                # images no longer reach it. The frozen configuration does not
                # pass --frame-b64; this is the backstop. Any FRAME line, from
                # an older engine or a hand-set flag, is dropped without
                # reaching the log or the pattern scan.
                continue
            ev = {"line": line}
            kind = "log"
            for name, pat in _PATTERNS:
                m = pat.search(line)
                if m:
                    kind, ev["m"] = name, list(m.groups())
                    break
            if kind == "calgate":
                # Record the gate from a successful calibration; a failure
                # prints a different line. Review #15, 2026-08-17: accept the
                # value only from the current engine or the one being switched
                # in. A late line from a retiring engine, which may live up to
                # 8 s, must not overwrite it.
                sw0 = self._switch
                if proc is self._proc or (sw0 and sw0.get("new") is proc):
                    self._gate = ev["m"][0]
            elif kind == "octave":
                # Review #16, 2026-08-17: remember the locked octave, so an
                # engine respawned with --gate can pin it with --octave instead
                # of falling back to the first-six-ticks estimator. Measured on
                # 2026-08-02, the same material locked to both +0 and +12; a
                # wrong lock live is wrong for the whole performance.
                mo = re.search(r"(?:locked|pinned to) ([+-]?\d+)", line)
                if mo and (proc is self._proc
                           or (self._switch or {}).get("new") is proc):
                    self._octave = int(mo.group(1))
            elif kind == "key":
                # As above: the latest key shift, the last signed integer on
                # the seeded or rotation line.
                mk = re.findall(r"[+-]\d+", line)
                if mk and (proc is self._proc
                           or (self._switch or {}).get("new") is proc):
                    self._key_shift = int(mk[-1])
            self._push(kind, ev)
            # The incoming engine reporting ready retires the old one. This
            # must run before both_pending, so that switching to both attaches
            # _proc2 under the new respond2.
            self._switch_ready(proc, kind)
            if kind == "live" and proc is self._proc:
                # In the both mode, bank enters only once respond2 has
                # calibrated and started listening. Entering earlier would feed
                # harmony into respond2's silence and singing calibration
                # windows (plan of 2026-08-12, step 3).
                # Race (review S2): take pending atomically, then spawn. If
                # Stop arrives during the spawn, finish it and stop it at once.
                # `proc is self._proc` (review #15, 2026-08-17): a late live
                # line from a retiring respond2 must not claim pending and
                # attach bank to the wrong engine.
                with self._lock:
                    o, self._both_pending = self._both_pending, None
                if o is not None and not self._stopping:
                    self._proc2 = self._spawn_bank(o, boot_kind="log")
                    if self._stopping and self._proc2.poll() is None:
                        self._retire([self._proc2])
        code = proc.wait()
        with self._lock:
            sw = self._switch
            # The incoming engine died before reporting ready, so the switch
            # is cancelled and the old engine keeps singing. Without cancelling,
            # _switch would stay set for good and every later key press would be
            # read as "a switch is in progress" and silently ignored, which
            # kills the keyboard for the rest of the performance.
            dead_new = bool(sw and sw["new"] is proc)
            if dead_new:
                self._switch = None
        if dead_new:
            # Since the one-action-one-move change of 2026-08-17 the old engine
            # is already retired at the moment of the click, so a failure is not
            # "staying on" but a full stop; switch_fail returns the UI to Start.
            self._push("warn", {"line": f"⚠ switch to {sw['mode']} died "
                                        f"({code}) — engines stopped, press "
                                        f"Start", "m": [sw["mode"]]})
            self._push("switch_fail", {"line": "press Start",
                                       "m": [sw["mode"]]})
            with self._lock:
                self._mode = None
        if proc is self._proc:
            self._both_pending = None   # the main engine died, so the both schedule is void (review A3)
        # Review S1: exit carries which engine died and whether any is still
        # alive, so the UI can decide whether to return the button to Start.
        # Returning it while the other engine is still singing would show a
        # false ENDED and leave an orphan.
        # Review #19, 2026-08-17: an engine retired on purpose sends no exit.
        # Every successful switch used to raise the UI's yellow "another engine
        # is still running" false alarm, which held the warning bar until the
        # next Start. That is the same bar as GATE DEAD, so it trained the
        # performer to ignore it.
        if proc in self._retired_procs:
            self._push("log", {"line": f"{who} retired ({code})"})
        else:
            self._push("exit", {"line": f"{who} exited ({code})", "code": code,
                                "who": who, "still": self.running()})

    def _spawn_bank(self, o, boot_kind="boot"):
        """The bank_live subprocess, in the frozen viva configuration
        (completed 2026-08-14; it previously carried only --attack 0.06).

        What this fixed: everything built on 2026-08-13 sat behind flags that
        default to off, so the app was running pre-v26 behaviour. Rehearsing
        from the CLI and performing from the app meant two different
        instruments, and this was the only rehearsal-to-performance mismatch at
        the time. Where each flag comes from:
          (--sing-gate 1 was here: a 27-dimension mouth and jaw model, 0.934
                         accuracy across sessions, 0% false trigger on a closed
                         or slightly open mouth (v34.1). Removed on 2026-08-18
                         with the camera; the gate authority is now microphone
                         level, see --mic-gate and --mouth 0.)
          --vl 2         The default of 1 had the upper three parts singing the
                         same note 79-100% of the time, so four parts were
                         really three in unison. At 2, unison drops to 0%, full
                         triads reach 43-50%, and each part is pulled out of the
                         model's break region (tenor 66-87% to 4-9%).
          --tenor 1      Adds the tenor part (v31). From v42 the lead voice is
                         reflow-male8 spk7 and no longer the same voice as the
                         bass, because bass1 is broken in the 65-68 range.
          --trim bass=+7,tenor=+6
                         Balance settled by ear on 2026-08-16 (tenor was +3).
                         The note was that this is a resonance problem and that
                         bass and tenor could be louder. A-weighted measurement
                         put alto loudest, with bass 4.7 dB and tenor 4.0 dB
                         below it in equal-loudness terms, and both were asked
                         to stand out more, so each gained 2 dB.
                         Use A-weighting, never flat rms: once the bass is an
                         octave down the two disagree by 13.3 dB (tenor 8.7,
                         alto 5.3, soprano 1.3). Flat rms calls the bass loud
                         enough while the ear cannot hear it, which is exactly
                         why --balance was abandoned on 2026-08-13.
          --vib 25       The default of 0 means no vibrato at all, which is the
                         strongest fingerprint of a corrected sound. The 3-8 Hz
                         share rises from 11.6% to 19.9% (a human singer here
                         measures 22.7%).
          --min-level -44
                         The default of -100 leaves the level gate open. With
                         the mouth closed at -54 dBFS, 43% of frames still
                         report a pitch, because autocorrelation is blind to
                         level. Raised from -50 to -44 (+6 dB) on 2026-08-19.
          --deadzone 0.35
                         The default of 0 flickers between notes 32% of the
                         time. At 0.35: flicker 19%, real notes missed 0 of 41,
                         commit latency 0 ms. The upper bound is clamped at
                         0.45 in the code.
          --maxlag 2     Ceiling of the output-latency ratchet (default 3).
          --per-part 4   The default of 1 gives four parts and four voices. A
                         three-way blind listening on 2026-08-16 (A = four
                         voices, B = sixteen detuned copies, C = sixteen
                         including real timbres) found B and C hard to tell
                         apart and both more like a group of people, so sixteen
                         voices are wanted and real timbres add nothing. C could
                         not win structurally: MEM_SPK holds only one surviving
                         singer per part (4 of 12 passed the ear test), so
                         per-part 4 is lead plus one real singer plus two
                         detuned copies, which is 4 real voices out of 16. CPU
                         verified on the offline bench: median 2.7 ms against a
                         35 ms budget, no overruns (F25).
          --mouth 0      2026-08-18: the camera view was removed from the UI and
                         the camera retired entirely. Gate authority had moved
                         to the microphone the same day (see --mic-gate); with
                         the view gone the camera had no function left, so the
                         mediapipe thread never starts. That saves CPU and
                         removes a failure mode, since GATE DEAD warnings can no
                         longer come from the camera. --sing-gate 1 and
                         --frame-b64 5 went with it, being dead flags without a
                         camera. To bring the camera back: drop --mouth 0,
                         restore those two flags, and take mouthCam from the
                         pre-2026-08-18 respond_live.html.
          --mem-real 0   2026-08-16: this was the cause of the intermittent
                         noise. The engine default flipped to 0 on 2026-08-17;
                         it is still written out here, because the frozen
                         configuration does not rely on defaults.
                         The old default of 1 let a part borrow a real singer
                         from another model (v39c). Two different recordings of
                         the same note laid on top of each other always differ
                         slightly in pitch, which is audible but invisible to a
                         spectrum, because it is interference and not an added
                         component.
                         Located by turning all seven processing layers off to
                         a bare version and adding them back one at a time:
                           per-part 1 + vibrato        -> clean
                           per-part 4 + vibrato        -> present
                           per-part 4, identical copies -> clean
                           per-part 4, real timbre removed only -> clean
                         Musically nothing is lost: the same day's ear test
                         found T2 (detuned copies) and T3 (with real timbres)
                         hard to tell apart, so the sense of sixteen singers
                         rests on detuning.
                         All five instruments are blind to it (dropped, xrun,
                         high-frequency energy, inharmonic energy, roughness);
                         the ear is the only judge.
          --pad 0        The default of 1.5 leaves the slow chord pad on. On
                         2026-08-16, listening to the four parts in isolation
                         turned up "something like an extra low melody besides
                         the part itself", which was the pad: the anchor note
                         holds for 1.5 s before the chord changes, so it walks a
                         line nobody is singing. Measured, the pad adds only
                         +0.8 dB while the singer sings and +5.2 dB while they
                         do not (the share of frames above -40 dB in silence
                         falls from 97.3% to 53.9%), so it mostly holds itself
                         up in the gaps. Judged: turn it off.
                         Turning the pad off does not fix the three seconds of
                         sound after the singer stops: the two decay curves are
                         almost identical (-0.8 vs -1.1 dB after 3 s), so that
                         tail has another source. Offline --file runs have no
                         camera and therefore an open gate, so this number is
                         only accurate live.
        --vowels is deliberately absent (that is, 0, everything on "ah"). After
        four multi-vowel versions on the afternoon of 2026-08-13 the final
        judgement by ear was to go back to a single vowel and keep the whole
        multi-vowel line out of the viva. That evening's note lists
        `--vowels 1 --layers 2,3,4` while marking the vowel switch as still to
        be judged by ear, that is, not settled, so the settled option is used
        here. To reverse it, change this and record the decision in STATE.

        --gain is not passed: the version that passed by ear defaults to 1.0,
        and the UI gain field carries respond semantics, so forcing 0.5 would
        depart from the certified configuration.
        boot_kind="log" is for the both mode: a boot event would cover the UI
        with LOADING, and respond2's heartbeat is silent while the singer sings,
        so LOADING would sit over the whole first phrase (review A5). It
        therefore goes to the log only.
          --mic-gate 1 --mic-open -18 --bleed -13.5
                         2026-08-18: live should be triggered by the microphone
                         alone, so gate authority moved from the camera to
                         microphone level (bank_live v43). The gate opens when
                         the hop RMS beats max(-18 dBFS, the last 8 hops of
                         output -13.5 + 6 dB) for two consecutive hops, and
                         closes after 0.8 s below the floor. The threshold went
                         -26 to -24 later on 2026-08-18 after an ear check
                         called for something stricter (singing level is at or
                         above -22 and the valleys sit between -24 and -22, so
                         -24 hugs the valley floor), then -24 to -18 (+6 dB) on
                         2026-08-19. To roll back, set --mic-open here and in
                         _spawn_solo to -24.
                         The first version carried -45 and -17.5 (CALIB k,
                         2026-08-10) and failed on contact: it triggered
                         constantly, and the first session dump
                         (app_bank_0818_134915) measured the gate open 88.6% of
                         the time. Two causes: the room plus feedback noise
                         floor sits at -34 to -28 dBFS, which makes a -45
                         threshold meaningless; and the coupling measured on
                         this machine that day had a median of -13.4 dB (p90
                         -7.9), leaking more than CALIB's -17.5. The new values
                         come from a parameter sweep over that session's
                         recording: gate open 15.9%, 96% of sung hops open,
                         9% false opens from feedback (the old values were
                         88 / 100 / 88).
                         (--sing-gate and --frame-b64 stayed briefly as a
                         picture-only UI feature; they were removed with
                         --mouth 0 later on 2026-08-18, see above.)
                         Honest boundary (F26): in the energy domain, feedback
                         can still be mistaken for singing. That reopens the
                         2026-08-16 finding that the gate is not reliable, and
                         the decision rests on the ear.
        Known boundary (review B10): bank always writes ch0/1, so when respond
        routes parts with --out-map they pile onto channel D. Agree the channel
        layout before performing the both mode with split routing."""
        o = o or {}
        cmd = [str(BANK_PY), "bank_live.py",
               "--tenor", "1", "--vl", "2", "--trim", "bass=+7,tenor=+6",
               "--vib", "25", "--deadzone", "0.35", "--maxlag", "2",
               "--min-level", "-44", "--attack", "0.06",
               "--per-part", "4", "--mem-real", "0", "--pad", "0",
               "--view", "0", "--mouth", "0",
               # bank level, from 2026-08-18: "in duo, bank covers neural".
               # Driven by the UI bank field; the default of 1.0 is the value
               # that passed by ear, and passing 1.0 explicitly is bit-identical
               # to the earlier practice of omitting --gain. The engine reads
               # its flags at Start and at a switch, so a change takes effect
               # only after pressing Start again or switching mode; it is not a
               # live control. Empty or non-numeric falls back to 1.0.
               "--gain", str(o["bank_gain"]
                             if o.get("bank_gain") not in (None, "")
                             else 1.0),
               "--mic-gate", "1", "--mic-open", "-18", "--bleed", "-13.5",
               "--dump", "scratchpad/app_bank_" + time.strftime("%m%d_%H%M%S")]
        # Channel routing, 2026-08-18: neural, live and live+neural should all
        # be routable. The UI map B-T-A-S is the bank_live VOICES order
        # (bass, tenor, alto, soprano) and is passed through unchanged. If the
        # four fields are not all set, bankMapValue returns empty, no flag is
        # passed, and the original stereo pan runs.
        _bm = str(o.get("bank_map") or "").strip()
        if _bm:
            cmd += ["--out-map", _bm]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        p = subprocess.Popen(
            cmd, cwd=str(HARMONY), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "bank"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_solo(self, o, boot_kind="boot"):
        """The solo_min subprocess: the neural live mode, quantised version,
        adopted on 2026-08-18.

        The frozen configuration that a full evening of listening converged on
        (engineering detail in the worklog for the evening of 2026-08-18):
          Quantised parts (--quantize): each part sings the absolute in-key
            target of the note the singer is standing on, which gives the chord
            something accurate to stand against. The older version copied the
            singer's own intonation, so when they drifted the whole choir
            drifted, which was the root of "less in tune than the offline ddsp
            version". Includes just intonation and a shared tuning centre, with
            a time constant of about 2.5 s tracking the singer's own centre.
            A note is committed only from a pitch that has held for 30 ms, so
            passing notes cannot enter; this cured both "the descending part
            does not move" and "an extremely fast slide".
          Casting: alto (satb2 spk0, female) a fourth above, tenor (new25_2k,
            the singer's own voice) a fourth below, plus the dry voice, giving
            three parts. Decided in sequence on 2026-08-18; the older four-part
            soprano and bass arrangement is retired, and returning to it means
            this function as it stood before 2026-08-18 in git.
          --vib 14@5.2,12@4.6: each part has its own vibrato at a different
            rate, so they are independent, opening over 300 ms at the attack and
            breathing in depth, which is not a synthetic sine.
          --wet-send 0.28: a shared reverb layer, the same hall, which is the
            glue of a choir; the dry voice stays dry.
          --voice-delay 25/40 ms: staggered attacks, the small lag of a choir
            behind a lead singer.
          --blocksize 960 (20 ms): two different models running together thrash
            the cache; 480 was judged by ear on 2026-08-14 as crackling. Only a
            single model can use 480.
        Gate: microphone level (see the --mic-gate note below), camera off."""
        o = o or {}
        # key: diatonic needs a key name. If the UI key field holds C..B,
        # with sharps or flats, it is used; auto, empty or an integer starts on
        # C and logs it. Minor keys need --minor, which is not yet in the UI.
        # The intervals of 2026-08-18 are the end point of that day's chain of
        # decisions (octave, then third, then symmetrical fourths). To change
        # them, touch only --steps and --steps2; the note and just-intonation
        # logic of quantize follows automatically.
        _k = str(o.get("key") or "").strip()
        if not re.fullmatch(r"[A-Ga-g][#b]?", _k):
            _k = "C"
        cmd = [str(SOLO_PY), "solo_min.py",
               "--model", str(SOLO_ALT), "--speaker", "0",
               "--mode", "diatonic", "--key", _k,
               # 2026-08-18, "tall building but no roof": the upper part's
               # anchor moves from +3 (a fourth) to +7 (the octave region), so
               # free picks its chord tones an octave above the singer and
               # actually caps the texture. If it sits too high or too sharp,
               # fall back to +5 (a sixth). The tenor anchor is unchanged.
               "--steps", "7",
               # 2026-08-18, "a third voice to fill the middle": the middle
               # layer is the same satb2 female model anchored at +3 (the
               # fourth region, where the upper part used to sit, which makes it
               # the middle floor). --model2 is not passed, so it inherits
               # --model. Three voices from two models at a 960 block is the
               # same load verified on 2026-08-14.
               "--steps2", "3",
               "--model3", str(SOLO_BASS), "--steps3", "-3",
               "--voices", "3",
               # 2026-08-18, "a fourth-based texture does not suit most
               # songs": free chord inference plus voice-lead independent
               # lines, passed by ear the same evening together with the
               # four-part and note-transition values. The parts take chord
               # tones rather than fixed intervals, and steps +/-3 are demoted
               # to range anchors. Verified on the bench: chord tones assigned
               # correctly, common tones held, voice motion within 5 semitones,
               # and quantised held is the only driver of chord state, since the
               # old note chain is broken under quantize, so there is no double
               # drive. To return to fixed fourths, remove these two flags.
               "--free", "--voice-lead",
               # 2026-08-18, "note changes sometimes smear together": tighten
               # the transitions. Glide 0.4 to 0.7 (a third completes in about
               # 43 ms), jumps over 2.5 semitones land directly, reverb t60 1.2
               # to 0.9, attack stagger 25/33/40 to 20/28/36 ms. The sustained-
               # note criteria (stable, locked) are unchanged.
               "--quantize", "1", "--q-glide", "0.7", "--q-snap", "2.5",
               "--vib", "14@5.2,13@4.9,12@4.6",
               "--wet-send", "0.28", "--wet-t60", "0.9",
               "--voice-delay", "0.02,0.028,0.036",
               # Level of the neural parts, driven by the UI neural field. The
               # default of 1.0 is the +5 dB accounting of 2026-08-18:
               # (choir/sqrt2)/you = (1/1.414)/0.4 = 1.77 = +5 dB, judged as
               # "the two parts 5 dB above me". The duo mode halves this again
               # by default. Review #17, 2026-08-17: the accept-only-if-non-empty
               # rule is unchanged.
               "--choir-gain", str(o["ngain"]
                                   if o.get("ngain") not in (None, "")
                                   else 1.0),
               "--gate-floor", "0.02",
               "--blocksize", "960", "--cushion-ms", "20",
               # Gate. 2026-08-18: neural should trigger without the camera,
               # like bank, following the bank_live v43 decision. Authority
               # moves from the camera to microphone level, and solo_min never
               # opens the camera at all (cv2 and mediapipe are not loaded). The
               # gate opens when the hop RMS beats
               # max(-26 dBFS, peak output over the last 280 ms - 13.5 + 6 dB)
               # for 70 ms, and closes after 0.8 s below the floor. The -26 and
               # -13.5 are taken straight from the sweep bank ran on this
               # machine the same day, with the same microphone in the same
               # room. bank's first version used the -45 default and failed on
               # contact by triggering constantly, so do not go back.
               # To return to the camera version, replace this line with
               # "--mouth-gate 1 --gate-fail open" (the 2026-08-14 version).
               # Honest boundary, as for bank (F26): in the energy domain
               # feedback can be mistaken for singing, and the ear decides.
               # The camera version's 11% false trigger on speech becomes
               # "speech always triggers", because a microphone gate cannot tell
               # singing from speech. Press Stop, or pull neural to 0, before
               # talking to an audience.
               "--mic-gate", "1", "--mic-open", "-18", "--bleed", "-13.5"]
        # The dry pass-through stays. Removing it on 2026-08-14 left "no
        # audible harmony, just two dry voices": harmony needs a reference to
        # exist against. The echo was not caused by the pass-through itself but
        # by `--dry-delay-ms 40`. That delay aligned the pass-through with the
        # converted harmony, which only makes sense on headphones, where the
        # singer is not in the air. Performing through speakers the singer is
        # present in the room, so the pass-through should align with their real
        # voice, which means zero delay; anything else is slap-back.
        # The cost is that the harmony trails the singer by about 40 ms, which
        # is what a choir following a lead singer does anyway.
        # Setting the you field to 0 removes the pass-through entirely, which is
        # only right on headphones or with split PA routing.
        _you = float(o.get("you_gain") if o.get("you_gain") not in (None, "")
                     else 0.3)   # 2026-08-18, judged in the app as "my own
                                 # voice is a bit loud": 0.4 to 0.3, which puts
                                 # the parts +7.4 dB over the dry voice instead
                                 # of +5.
        cmd += ["--dry-delay-ms", "0"]
        cmd += ["--no-dry"] if _you <= 0 else ["--you-gain", str(_you)]
        # Channel routing. 2026-08-18: neural, live and live+neural should all
        # be routable. In neural, the UI map D-U-L-S means U = the roof (+7),
        # S = the middle layer (+3), L = tenor (-3), which is the v1,v2,v3 order
        # of solo_min --out-map; D is the dry voice, --dry-ch. Leaving S empty
        # puts it on the same channel as U, as in respond. An incomplete
        # selection returns an empty mapValue, so no flag is passed and the
        # original mix path runs, bit-identical at the engine.
        _om = str(o.get("out_map") or "").strip()
        if _om:
            _p = _om.split(",")
            _d, _u, _l = _p[0], _p[1], _p[2]
            _s = _p[3] if len(_p) > 3 else _u
            cmd += ["--out-map", f"{_u},{_s},{_l}", "--dry-ch", _d]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        p = subprocess.Popen(
            cmd, cwd=str(SERVER), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "solo"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_respond(self, o, boot_kind="boot"):
        """The respond2 subprocess: the main engine of the answering mode, and
        the one that leads in the both mode.

        `--gate` reuses the threshold already calibrated in this session when
        switching into the answering mode mid-performance. Without it, every
        switch would rerun calibration, three seconds of silence and five of
        singing, which cannot be done on stage; during calibration the singer
        has to follow the machine rather than sing their piece, which is not
        seamless. A value exists only if calibration has succeeded in this
        session, since a failure prints a different line that is not matched.
        With no value, calibration runs as before. This holds within one
        session, as long as the microphone and the room are unchanged.
        """
        cmd = [str(ENGINE_PY), "respond2.py", "--live",
               # 2026-08-19, "raise all the level thresholds": calibration
               # threshold doubled (+6 dB). The engine default of 1.0 is the
               # original behaviour, so rolling back means removing these two
               # arguments.
               "--gate-boost", "2.0",
               "--key", str(o.get("key") or "auto"),
               "--gain", str(o.get("gain") or 0.5),
               "--stab", str(o.get("stab") or 1),
               "--max-min", str(o.get("max_min") or 40)]
        if o.get("gap"):
            # 2026-08-14: "SING switches to HOLD too quickly, there is not
            # enough time to sing". The engine default of 0.35 s sits inside the
            # measured breath range of 0.3-0.5 s, so a slightly long breath is
            # read as the end of the phrase. Exposed in the UI to be set by the
            # performer rather than fixed at a guessed value.
            cmd += ["--gap", str(o["gap"])]
        if o.get("gate"):
            cmd += ["--gate", str(o["gate"])]
        elif self._gate:
            cmd += ["--gate", self._gate]
            self._push("log", {"line": f"[switch] reusing calibrated gate "
                                       f"{self._gate} — no recalibration"})
            # Review #16, 2026-08-17: respawning with --gate skips calibration,
            # which used to fall back to the first-six-voiced-ticks octave
            # estimator (on 2026-08-02 the same material locked to both +0 and
            # +12, and a wrong lock live is wrong all evening), and left
            # --key auto without its calibration seed, starting from 0.
            # Seed back what this session already locked: --octave pins the
            # octave, --key-seed seeds the key, and the engine still corrects
            # phrase by phrase. This hangs only off the gate-reuse path; a Start
            # with fresh calibration locks its own.
            if self._octave is not None:
                cmd += ["--octave", str(self._octave)]
            if (self._key_shift is not None
                    and str(o.get("key") or "auto").lower() == "auto"):
                cmd += ["--key-seed", str(self._key_shift)]
        if o.get("in_name"):
            cmd += ["--in-name", str(o["in_name"])]
        if o.get("out_name"):
            cmd += ["--out-name", str(o["out_name"])]
        if o.get("out_map"):                 # 'D,U,L' -> respond2 --out-map (split routing)
            cmd += ["--out-map", str(o["out_map"])]
        if o.get("reverb") not in (None, ""):    # distance: reverb send (default 0.20)
            cmd += ["--reverb", str(o["reverb"])]
        if o.get("tail_s") not in (None, ""):    # length: seconds the parts sustain after the singer stops
            cmd += ["--tail-s", str(o["tail_s"])]
        p = subprocess.Popen(
            cmd, cwd=str(HARMONY), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(p, "respond"),
                         daemon=True).start()
        self._push(boot_kind, {"line": " ".join(cmd[1:])})
        return p

    def _spawn_mode(self, mode, o, boot_kind="boot"):
        """Spawn the engines for a mode and return (main engine, [others
        started alongside it]).

        Start and keyboard switching share this one path. It was factored out
        deliberately when switching was added on 2026-08-14: the lesson from the
        hard-coded flags in `_spawn_bank` is that two paths to the same thing
        eventually become two different instruments.
        The main engine is the one that reports ready, and the switch retires
        the old engines against it."""
        if mode == "stream":
            # Live mode v2 is bank_live, which replaced spike_stream on
            # 2026-08-12. To bring the old route back, set cmd to
            # [ENGINE_PY, "spike_stream.py", "--gain", "0.6", "--block",
            # "0.20", "--crossfade", "0.08", "--extra", "0.7"], which is v14 of
            # 2026-08-05 §M unchanged.
            return self._spawn_bank(o, boot_kind), []
        if mode == "solo":
            # Neural live, added on 2026-08-14 so the audience never sees the
            # application being changed.
            return self._spawn_solo(o, boot_kind), []
        if mode == "duo":
            # 2026-08-14, "try live neural plus live": the sampled four parts,
            # the two neural voices and the dry voice all sounding at once.
            # Neither side opens the camera any more (2026-08-18: gate authority
            # moved to microphone level, and later the same day the camera view
            # was removed from the UI, retiring bank's picture-only camera too).
            # bank is taken as the main engine because its ready arrives later,
            # having to load the sample bank, so both are already singing when
            # the switch retires the old ones.
            # The duo starting point halves the neural side again, measured on
            # 2026-08-14 as much louder than the sampler. The UI neural field
            # still applies; only its default is halved in the duo mode, and a
            # value typed in is used as given.
            od = dict(o)
            # Review #17, 2026-08-17: `or` cannot be used here. The UI sends
            # strings, and 0 is falsy after float(), so setting neural to 0
            # would be revived by _spawn_solo's default to 0.5 and not halved,
            # turning a request for silence into double the level. The you_gain
            # idiom next to it is the correct one.
            od["ngain"] = float(o["ngain"] if o.get("ngain") not in (None, "")
                                else 0.5) * 0.5
            partner = self._spawn_solo(od, "log")
            # Duo balance settled on 2026-08-18 as 0.5/0.5: with the bank field
            # empty, duo uses 0.5, while the neural half reaches 0.25 through
            # the halving above. Running bank alone (mode 2) with the field
            # empty still uses the certified 1.0, from _spawn_bank's fallback.
            # A value typed in is used as given in both modes.
            ob = dict(o)
            if ob.get("bank_gain") in (None, ""):
                ob["bank_gain"] = 0.5
            return self._spawn_bank(ob, boot_kind), [partner]
        return self._spawn_respond(o, boot_kind), []

    def start(self, opts: dict):
        """opts: {mode, gain, stab, key, gap, in_name, out_name, out_map,
        max_min, reverb, tail_s, ngain, bank_gain, you_gain} - all have
        defaults. Which fields are read by which engine is in the _spawn_*
        methods; respond, bank and solo each take what they need and ignore the
        rest."""
        if self.running():
            return False
        self._stopping = False
        # Void any schedule left over from the previous run (review A3): if
        # respond2 died during calibration and the answering mode alone was
        # restarted, an old pending would make bank appear out of nowhere.
        self._both_pending = None
        o = opts or {}
        # A different input device makes the previous calibration's threshold
        # invalid for this microphone, so it is discarded and calibration runs
        # again. Otherwise it would be reused silently all evening, and the
        # symptom, phrases not ending or a whole song read as one phrase, looks
        # unrelated to having changed the microphone.
        if o.get("in_name") != (self._opts or {}).get("in_name"):
            self._gate = None
        self._opts = dict(o)        # the same devices and parameters are reused across switches
        self._mode = o.get("mode") or "respond"
        if self._mode == "both":
            # The third mode, decided on 2026-08-12: respond2 goes first, and
            # _reader starts bank_live only on "respond2 live.", which marks the
            # end of calibration.
            self._both_pending = o
        self._proc, extra = self._spawn_mode(self._mode, o)
        self._proc2 = extra[0] if extra else None
        return True

    def _retire(self, procs):
        """Retire an engine with SIGINT and make sure it really dies
        (review #15, 2026-08-17).

        Retiring on a switch used to send one bare SIGINT and drop the reference
        into a local variable. When the engine hung closing a CoreAudio stream,
        a known entry in the playbook, it held the microphone, the output and
        the camera for good, and neither stop() nor _on_closed could see it.
        That reproduced, word for word, the failure _on_closed's docstring warns
        about: the space key being dead at the next performance.
        Here the process is recorded in the _retired registry, which stop() and
        _on_closed also collect, marked in _retired_procs so its exit event is
        downgraded, and given an 8 s reaper thread that falls back to SIGKILL.
        The budget matches stop(), which is enough for the dump to be written."""
        alive = [p for p in procs if p and p.poll() is None]
        if not alive:
            return
        with self._lock:
            self._retired = [q for q in self._retired
                             if q.poll() is None] + alive
            self._retired_procs.update(alive)
        for p in alive:
            p.send_signal(signal.SIGINT)     # the engine finishes writing its dump before leaving

        def _r():
            t0 = time.time()
            while any(p.poll() is None for p in alive) and time.time() - t0 < 8:
                time.sleep(0.25)
            for p in alive:
                if p.poll() is None:
                    self._push("warn", {"line": "⚠ retired engine hung in "
                                        "stream close → killed (dump was "
                                        "written before the close)",
                                        "m": ["kill"]})
                    p.kill()
        threading.Thread(target=_r, daemon=True).start()

    # Mode to the event that means "this engine is really singing". A switch
    # waits for that event before stopping the old engines.
    # both still works but was removed from the UI on 2026-08-14; the mechanism
    # is kept rather than deleted, so bringing it back is one UI option.
    _READY = {"respond": "live", "both": "live",
              "stream": "bank_on", "solo": "solo_on", "duo": "bank_on"}

    def switch_mode(self, mode, opts=None):
        """Switching by keys 1-4 or the drop-down: the old engine stops on the
        click, nothing sounds while the new one loads (the UI holds SWITCHING),
        and it sings once ready.

        Decided on 2026-08-17, one action one move. The old design started the
        new engine and killed the old one only on ready, which kept the sound
        seamless, but the old engine went on responding to the singer during the
        switch, so the instrument did not obey the action. The overlap is gone.
        A failed switch leaves nothing sounding and returns to Start, with the
        switch_fail event restoring the UI."""
        if mode not in self._READY:
            return False
        if not self.running() or self._stopping:
            return False            # switching before the piece starts is
                                    # meaningless; press Start
        with self._lock:
            if self._switch is not None:
                # A switch is in progress: repeated presses are ignored rather
                # than queued, so a string of switches cannot pile up and run on
                # their own.
                return False
            if mode == self._mode:
                return True
            old = [p for p in (self._proc, self._proc2)
                   if p and p.poll() is None]
            self._switch = {"mode": mode, "old": old, "new": None, "extra": []}
        # A switch takes the UI field values as they are now (neural, gap,
        # you and so on). Using only the Start set would mean that adjusting
        # neural and then pressing a key appears to do nothing, while the number
        # on screen has clearly changed, so display and behaviour disagree.
        # Device fields still come from the Start set; changing audio devices
        # mid-performance is out of scope for this path.
        o = dict(self._opts or {})
        for k, v in (opts or {}).items():
            if k not in ("mode", "in_name", "out_name", "out_map"):
                o[k] = v
        o["mode"] = mode
        try:
            p, extra = self._spawn_mode(mode, o, boot_kind="log")  # a boot event would cover the UI with LOADING
        except Exception as e:      # noqa: BLE001 - if it cannot be spawned, the old engine keeps singing
            with self._lock:
                self._switch = None
            self._push("warn", {"line": f"⚠ switch to {mode} failed ({e})",
                                "m": [str(e)]})
            return False
        with self._lock:
            cancelled = self._switch is None    # stop() arrived during the spawn
            if not cancelled:
                self._switch["new"] = p
                self._switch["extra"] = extra
                old_now = list(self._switch["old"])
                if mode == "both":
                    self._both_pending = o  # bank follows only after ready
        if cancelled:
            # Also stop the partner of a combined mode. _retire takes the lock
            # itself, so it must be called outside the lock.
            self._retire([p] + extra)
            return False
        # One action one move, 2026-08-17: the old engine retires at the moment
        # of the click, leaving once its dump is written, rather than singing on
        # until the new engine is ready. The silence while the new one loads is
        # covered by SWITCHING in the UI.
        self._retire(old_now)
        self._push("switching", {"line": f"switching to {mode}…",
                                 "m": [mode]})
        threading.Thread(target=self._switch_watchdog, args=(p, mode),
                         daemon=True).start()
        return True

    SWITCH_TIMEOUT = 30.0

    def _switch_watchdog(self, proc, mode):
        """Watchdog for a new engine that is alive but never reports ready.

        Without it, an engine that hangs part-way through rendering the sample
        bank, opening a CoreAudio stream, or claiming the camera would leave
        `_switch` set for good, and every later key press would be read as "a
        switch is in progress" and silently ignored. The keyboard would be dead
        for the rest of the performance, with nothing on stage to say why.
        _reader covers the case where the process dies; a hang can only be
        caught by time.

        Review #3, 2026-08-17: the clock runs 30 s from the last line of output,
        not 30 s from launch. Re-rendering the bank from a cold cache takes
        several minutes, but the engine prints a progress line every 8 notes, so
        there is activity throughout; a real hang produces no output at all and
        is collected after 30 s. t_act is stamped by _switch_ready on every
        line."""
        t0 = time.time()
        while True:
            time.sleep(0.5)
            with self._lock:
                sw = self._switch
                if sw is None or sw["new"] is not proc:
                    return              # already ready, or cancelled
                last = max(t0, sw.get("t_act", t0))
            if time.time() - last >= self.SWITCH_TIMEOUT:
                break
        with self._lock:
            sw = self._switch
            if not sw or sw["new"] is not proc:
                return
            self._switch = None
            victims = [sw["new"]] + sw["extra"]
        # The hung incoming engine has to be collected, with the 8 s SIGKILL
        # backstop, or it holds the audio device and the camera.
        self._retire(victims)
        self._push("warn", {"line": f"⚠ switch to {mode} timed out "
                                    f"({self.SWITCH_TIMEOUT:g}s with no "
                                    f"output) — engines stopped, press Start",
                            "m": [mode]})
        self._push("switch_fail", {"line": "press Start", "m": [mode]})
        with self._lock:
            self._mode = None

    def _switch_ready(self, proc, kind):
        """The incoming engine reporting ready retires the old ones. Returns
        True when this line has completed the switch."""
        with self._lock:
            sw = self._switch
            if sw and sw["new"] is proc:
                # Activity stamp for the watchdog: rendering the bank from a
                # cold cache prints progress lines, so it is not killed at 30 s.
                sw["t_act"] = time.time()
            if not sw or sw["new"] is not proc or kind != self._READY[sw["mode"]]:
                return False
            self._switch = None
            self._mode = sw["mode"]
            # The partner of a combined mode must be handed to _proc2, or it
            # falls outside the view of stop() and running() and becomes an
            # orphan holding the audio device and the camera.
            self._proc, self._proc2 = proc, (sw["extra"] or [None])[0]
            old = sw["old"]
        # Since 2026-08-17 the old engines retire at the click, so this is only
        # a backstop; _retire touches live processes only, and a hang is
        # collected by the reaper.
        self._retire(old)
        self._push("switched", {"line": f"now {self._mode}", "m": [self._mode]})
        return True

    def devices(self):
        """List audio devices for the UI drop-downs. sounddevice is queried
        from the engine's own venv, so this is the same table respond2 sees when
        it opens a stream."""
        try:
            r = subprocess.run(
                [str(ENGINE_PY), "-c",
                 "import sounddevice as sd, json;"
                 "d = sd.query_devices();"
                 "print(json.dumps({"
                 "'in': [x['name'] for x in d if x['max_input_channels'] > 0],"
                 "'out': [{'name': x['name'], 'ch': x['max_output_channels']}"
                 " for x in d if x['max_output_channels'] > 0]"
                 "}, ensure_ascii=False))"],
                capture_output=True, text=True, timeout=15)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001 - showing the error in the UI is enough
            return {"in": [], "out": [], "err": str(e)}

    def stop(self):
        """SIGINT lets the dump be written first (fixed 2026-08-04). Closing a
        CoreAudio stream can hang, a known playbook issue, so anything still
        alive after 8 s is sent SIGKILL; the dump is already safe by then and no
        lldb rescue is needed.

        Guard against double clicks (review A7, 2026-08-12): a second SIGINT
        would land inside the sf.write that is writing the dump and truncate the
        file. If one has already been sent, this only waits."""
        self._both_pending = None
        with self._lock:
            sw, self._switch = self._switch, None   # cancel the switch and stop the incoming engine too
            # Review #15, 2026-08-17: engines that are retiring but may not yet
            # be dead must be collected too, or a hang leaves an orphan holding
            # the device.
            retired = list(self._retired)
        procs = [p for p in ([self._proc, self._proc2] + retired
                             + ([sw["new"]] + sw["extra"] if sw else []))
                 if p and p.poll() is None]
        self._last_stop_procs = procs       # _on_closed waits on this same list
        if not procs:
            return True
        # The double-click guard is a time window, not a permanent latch: if
        # the stop/spawn race in the both mode leaks an orphan bank, a second
        # Stop after the window still collects it rather than spinning for good.
        # The window is the reaper budget of 8 s plus a 4 s margin (review B8:
        # the two are tied, so changing the budget means changing this. A window
        # shorter than the budget lets a second SIGINT land in the sf.write that
        # is writing the dump and truncate the file).
        if self._stopping and time.time() - self._stop_t < 8 + 4:
            return True
        self._stopping = True
        self._stop_t = time.time()
        for p in procs:
            p.send_signal(signal.SIGINT)

        def _reap():
            t0 = time.time()
            while any(p.poll() is None for p in procs) and time.time() - t0 < 8:
                time.sleep(0.25)
            for p in procs:
                if p.poll() is None:
                    self._push("warn", {"line": ("⚠ stream close hung (known playbook issue) → forcing shutdown; "
                         "the dump was already written before the close"), "m": ["kill"]})
                    p.kill()
        threading.Thread(target=_reap, daemon=True).start()
        return True

    def running(self):
        sw = self._switch
        return bool((self._proc and self._proc.poll() is None)
                    or (self._proc2 and self._proc2.poll() is None)
                    # An engine being switched in also counts as running.
                    # Missing it would let the UI return the button to Start
                    # during the overlap, and Stop would not collect it, leaving
                    # an orphan holding the audio device: the false ENDED of
                    # review S1.
                    or (sw and any(q and q.poll() is None
                                   for q in [sw["new"]] + sw["extra"])))

    def _on_closed(self):
        """Closing the window means the interpreter is about to exit, which
        takes the daemon reaper thread in stop() with it, so the SIGKILL after
        8 s is never sent. An engine stuck closing a CoreAudio stream then
        becomes an orphan holding port 8766 and the audio device, and the space
        key is dead at the next performance (review S1/A5, 2026-08-12). So the
        collection happens synchronously here: SIGINT, bounded wait, kill."""
        self.stop()
        # Review #15, 2026-08-17: wait on the list stop() actually aimed at,
        # including engines being switched in or retired. Snapshotting only
        # (_proc, _proc2) missed an engine mid-switch when the window closed,
        # so the interpreter exited part-way through the SIGINT, the reaper died
        # with it, and an orphan was left.
        procs = [p for p in getattr(self, "_last_stop_procs", []) if p]
        # 8 s is the same budget as the reaper in stop(). Closing the window
        # mid-render means SIGINT has to wait for torch to return to Python, and
        # the dump's sf.write can be hundreds of megabytes; 3 s would SIGKILL an
        # engine that is still writing and truncate the whole recording
        # (verified N1, 2026-08-12).
        # Both engines share one deadline (review B7: 8 s each would block the
        # GUI main thread for up to 16 s, with every evaluate_js stalled).
        deadline = time.time() + 8
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()


def main():
    if not UI_FILE.exists():
        raise SystemExit(f"UI file not found: {UI_FILE}")
    if not ENGINE_PY.exists():
        raise SystemExit(f"engine python not found: {ENGINE_PY}")
    bridge = Bridge()
    window = webview.create_window(
        "Solo Choir - Respond", url=str(UI_FILE), width=760, height=640,
        min_size=(560, 480), background_color="#10141a", js_api=bridge)
    bridge._attach(window)
    window.events.closed += bridge._on_closed
    webview.start()


if __name__ == "__main__":
    main()
