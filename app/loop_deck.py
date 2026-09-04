#!/usr/bin/env python3
"""loop_deck.py — a live layered bed for Solo Choir. A plug-in: not one line
of the main app changes.

Signal path
-----------
    main app --> multi-output device (speakers + BlackHole 2ch)
                                  |
                                  v
                          loop_deck records it --> speakers (loop layer only)

The main app does not know this program exists. It simply sends its output to a
device that also contains BlackHole. loop_deck takes what the main app has just
sent from BlackHole, records it into a loop layer, and sends that to the
speakers from its own output.

Why there is no passthrough
    A passthrough would send the main app's output to BlackHole alone and have
    this program forward it, which means that if this program dies there is no
    sound at all. With a multi-output device, this program dying only removes
    the bed, and the main app still reaches the speakers. The cost of a rescue
    is very different.

Keys (pressed in this program's terminal window)
    space  first press starts recording; the second stops, fixes the loop
           length and starts looping at once. After that, each press records the
           next layer, starting from the current position and joining the loop
           when it comes round.
    u      undo the last layer
    c      clear everything and return to no bed
    m      mute and unmute: the bed disappears at once, for a rescue
    q      quit

Feedback safety
    The input is BlackHole, that is, the main app's output, not a microphone.
    Layering does not pick up the room and cannot feed back without limit. This
    holds as long as this program's output device is not itself a device
    containing BlackHole, which would record itself. That setting is refused at
    start-up.
"""
import argparse
import queue
import sys
import termios
import threading
import time
import tty

import numpy as np
import sounddevice as sd

SR_DEFAULT = 48000
MAX_LOOP_S = 90.0          # longest first layer; recording stops automatically past this


class LoopDeck:
    def __init__(self, sr, ch, block, max_loop_s, lat_ms, gain):
        self.sr, self.ch, self.block = sr, ch, block
        self.gain = float(gain)
        self.lat = int(round(lat_ms * 1e-3 * sr))   # input latency compensation, in samples
        self.cap = int(max_loop_s * sr)

        self.L = 0                                  # loop length in samples; 0 = not yet fixed
        self.mix = None                             # (L, ch): the sum of committed layers
        self.layers = []                            # one (L, ch) per layer, kept for undo
        self.layer_on = []                          # per-layer on/off; off keeps the layer but silences it
        self.beats = 0                              # beats per loop; 0 = no beat grid
        self.bpm = 0.0                              # tempo; 0 = free length, fixed by the stop press
        self.clk = 0                                # free-running beat clock, in samples
        self.armed = False                          # record pressed, waiting for the next beat 1
        self.cin_beats = 4                          # count-in beats; 0 = record on the press
        self.cin = 0                                # samples left in the count-in
        self.cin_all = 0                            # total samples in this count-in
        self.p = 0                                  # playback position

        self.rec = False                            # recording
        self.first = True                           # whether the next recording is the first layer
        self.pend = np.zeros((self.cap, ch), np.float32)
        self.pend_n = 0                             # samples written so far in the first layer
        self.wpos = 0                               # write position from layer 2 on
        self.wdone = 0                              # samples written from layer 2 on

        self.muted = False
        self.in_n = 0                               # how many times the record callback has run
        self.xruns = 0
        self.click_hi = self._mk_click(1800.0)      # beat 1
        self.click_lo = self._mk_click(1100.0)      # other beats
        self.click_cur = self.click_lo
        self.click_i = -1                           # position in the click; -1 = not clicking
        self.lock = threading.Lock()
        self.msg = queue.Queue()                    # the callback never prints; it posts messages instead

    # ---- the count-in click ---------------------------------------------
    def _mk_click(self, f):
        """A 30 ms blip. It appears only in the count-in, never while recording."""
        n = int(self.sr * 0.03)
        t = np.arange(n, dtype=np.float32) / self.sr
        w = np.sin(2 * np.pi * f * t) * np.exp(-t * 70.0) * 0.30
        return np.repeat(w.astype(np.float32)[:, None], self.ch, axis=1)

    def beat_len(self):
        """Samples per beat; 0 when no tempo is set."""
        return int(round(self.sr * 60.0 / self.bpm)) if self.bpm > 0 else 0

    def _emit_click(self, outdata, frames):
        """Finish a click left over from the previous block at the start of this one."""
        if self.click_i < 0:
            return
        c = self.click_cur
        n = min(frames, c.shape[0] - self.click_i)
        if n <= 0:
            self.click_i = -1
            return
        outdata[:n] += c[self.click_i:self.click_i + n]
        self.click_i += n
        if self.click_i >= c.shape[0]:
            self.click_i = -1

    def _start_click(self, outdata, frames, off, accent):
        c = self.click_hi if accent else self.click_lo
        self.click_cur = c
        n = min(frames - off, c.shape[0])
        outdata[off:off + n] += c[:n]
        self.click_i = n if n < c.shape[0] else -1

    # ---- audio callbacks: no allocation, no printing ---------------------
    def rec_cb(self, indata, frames, tinfo, status):
        if status:
            self.xruns += 1
        self.in_n += 1                              # heartbeat: proof the input device is really delivering
        if not self.rec:
            return
        if self.first:
            n = min(frames, self.cap - self.pend_n)
            if n > 0:
                self.pend[self.pend_n:self.pend_n + n] = indata[:n]
                self.pend_n += n
            if self.pend_n >= self.cap:             # the ceiling was reached, so stop
                self.rec = False
                self.msg.put(("auto_stop", None))
            return
        # From layer 2 on the length is known: write into the ring and finish after one lap
        L = self.L
        n = min(frames, L - self.wdone)
        if n <= 0:
            return
        w = self.wpos
        end = w + n
        if end <= L:
            self.pend[w:end] = indata[:n]
        else:
            k = L - w
            self.pend[w:L] = indata[:k]
            self.pend[0:end - L] = indata[k:n]
        self.wpos = end % L
        self.wdone += n
        if self.wdone >= L:
            self.rec = False
            self.msg.put(("commit", None))

    def play_cb(self, outdata, frames, tinfo, status):
        if status:
            self.xruns += 1
        outdata[:] = 0.0
        self._emit_click(outdata, frames)

        if self.cin > 0:                         # -- count-in: the clock has not started yet
            bl = self.beat_len()
            done = self.cin_all - self.cin
            if bl:
                k = (done + bl - 1) // bl
                off = k * bl - done
                if off < frames:
                    self._start_click(outdata, frames, off, k == 0)
            self.cin -= frames
            if self.cin <= 0:                    # count finished: this instant is beat 1
                self.cin = 0
                L = self.L
                if not L:                        # the main thread just cancelled, so do not start
                    return
                self.clk = 0
                self.p = 0
                self.wpos = self.lat % L
                self.wdone = 0
                self.rec = True
                self.msg.put(("rec_bar1", None))
            return

        base = self.clk
        self.clk = base + frames                 # the clock always runs, so the metronome need not wait for recording
        L = self.L
        if self.armed and L:
            # Waiting for beat 1. The clock runs in this callback, so the moment it
            # wraps is judged here most accurately. It used to be judged in the record
            # callback as "position in the loop < two blocks", which was another thread
            # and another clock, with a window of a few milliseconds; missing it cost a
            # whole lap, or every lap, so the press never started a recording.
            bl = self.beat_len()
            if bl and self.cin_beats and (L - base % L) <= min(self.cin_beats * bl, L):
                pos = base % L                   # click for a few beats before the start so the player can join
                k = (pos + bl - 1) // bl
                off = k * bl - pos
                if off < frames:
                    self._start_click(outdata, frames, off, False)
            if (base % L) + frames >= L:
                self.armed = False
                self.wpos = self.lat % L
                self.wdone = 0
                self.rec = True
                self.msg.put(("rec_layer", None))
        if L and self.bpm > 0:
            self.p = base % L                    # with a tempo, playback is tied to the clock and cannot drift
        if self.mix is None or self.muted or L == 0:
            return
        p = self.p
        end = p + frames
        if end <= L:
            outdata[:] += self.mix[p:end]
        else:
            k = L - p
            outdata[:k] += self.mix[p:L]
            outdata[k:] += self.mix[0:end - L]
        if self.bpm <= 0:
            self.p = end % L

    # ---- control, on the main thread -------------------------------------
    def target_L(self):
        """Loop length in samples when tempo and beats are set; 0 means free length."""
        if self.bpm > 0 and self.beats > 0:
            return int(round(self.sr * 60.0 / self.bpm * self.beats))
        return 0

    def toggle(self):
        if self.cin > 0:                            # cancelled part-way through the count-in
            self.cin = self.cin_all = 0
            self.rec = False                        # also stop a recording the audio side has just begun
            self.L = 0                              # the loop length is not fixed yet, so return to idle
            self.first = True
            self.msg.put(("cancel_arm", None))
            return
        if self.armed:                              # cancelled before recording began
            self.armed = False
            self.msg.put(("cancel_arm", None))
            return
        if self.rec:
            if self.first:
                self._commit_first()
            else:
                self.rec = False                    # stopped early, so the incomplete lap is discarded
                self.msg.put(("cancel", None))
            return

        tl = self.target_L()
        if tl and not self.layers:
            # Metronome mode with no track yet: the loop length is known and nothing
            # there is no existing beat to align to. Do not wait: the
            # press is beat 1, the clock resets, and one full lap still ends it.
            # Waiting for the next lap only makes sense once a bed exists.
            with self.lock:
                self.L = tl
                self.first = False
                self.clk = 0
                self.p = 0
                bl = self.beat_len()
                if self.cin_beats and bl:
                    self.cin_all = self.cin = self.cin_beats * bl
                    self.msg.put(("countin", self.cin_beats))
                else:
                    self.wpos = self.lat % tl
                    self.wdone = 0
                    self.rec = True
                    self.msg.put(("rec_bar1", None))
            return

        if self.first:
            self.pend_n = 0
            self.rec = True
            self.msg.put(("rec_first", None))
        elif self.bpm > 0:
            if self.L == 0:
                return
            self.armed = True
            self.msg.put(("armed", None))
        else:
            if self.L == 0:
                return
            self.wpos = (self.p + self.lat) % self.L   # align to the playback position and compensate input latency
            self.wdone = 0
            self.rec = True
            self.msg.put(("rec_layer", None))

    def _rebuild_locked(self):
        """Recompute the mix from the layers still on. All off produces zeros, not
        None, which would read as "nothing recorded" and send the UI back to idle."""
        if not self.layers:
            self.mix = None
            return
        acc = None
        for lay, on in zip(self.layers, self.layer_on):
            if not on:
                continue
            acc = lay.copy() if acc is None else acc + lay
        self.mix = (acc * self.gain) if acc is not None \
            else np.zeros((self.L, self.ch), np.float32)

    def _commit_first(self):
        self.rec = False
        n = self.pend_n
        if n < int(0.25 * self.sr):
            self.msg.put(("too_short", None))
            return
        n = max(0, n - self.lat)
        with self.lock:
            layer = self.pend[self.lat:self.lat + n].copy() if self.lat else self.pend[:n].copy()
            self.L = layer.shape[0]
            self.layers = [layer]
            self.layer_on = [True]
            self.p = 0
            self.first = False
            self._rebuild_locked()
        self.msg.put(("first_done", self.L / self.sr))

    def commit_layer(self):
        with self.lock:
            self.layers.append(self.pend[:self.L].copy())
            self.layer_on.append(True)
            self._rebuild_locked()
        self.msg.put(("layer_done", len(self.layers)))

    def toggle_layer(self, i):
        with self.lock:
            if 0 <= i < len(self.layers):
                self.layer_on[i] = not self.layer_on[i]
                self._rebuild_locked()
                return self.layer_on[i]
        return None

    def drop(self, i):
        """Delete one layer. Deleting the last returns to the un-recorded state."""
        with self.lock:
            if not (0 <= i < len(self.layers)):
                return len(self.layers)
            self.layers.pop(i)
            self.layer_on.pop(i)
            if not self.layers:
                self.clear_locked()
                return 0
            self._rebuild_locked()
            return len(self.layers)

    def undo(self):
        return self.drop(len(self.layers) - 1)

    def clear_locked(self):
        self.layers = []
        self.layer_on = []
        self.mix = None
        self.L = 0
        self.p = 0
        self.first = True
        self.rec = False
        self.armed = False
        self.cin = self.cin_all = 0

    def clear(self):
        with self.lock:
            self.clear_locked()


def _resolve(name_or_idx, kind):
    """Device name, as a partial string, or index, to an index."""
    if name_or_idx is None:
        return None
    try:
        return int(name_or_idx)
    except ValueError:
        pass
    want = str(name_or_idx).lower()
    hits = []
    for i, d in enumerate(sd.query_devices()):
        chans = d["max_input_channels"] if kind == "in" else d["max_output_channels"]
        if chans > 0 and want in d["name"].lower():
            hits.append(i)
    if not hits:
        sys.exit(f"{'input' if kind == 'in' else 'output'} device not found: {name_or_idx}")
    return hits[0]



# -- message handling, shared by both interfaces -----------------------
_hb = {"n": -1, "t": 0.0, "warned": False}     # record heartbeat: the last count seen and when

def pump(deck, max_loop_s, say):
    """Consume the messages posted by the callbacks. say(text) decides where
    they are shown."""
    # If the input device stops delivering, the display would sit on "recording"
    # for ever. Say so rather than let the player wait.
    now = time.monotonic()
    if deck.in_n != _hb["n"]:
        _hb.update(n=deck.in_n, t=now, warned=False)
    elif now - _hb["t"] > 2.0 and not _hb["warned"]:
        _hb["warned"] = True
        say("! no input for 2 s (microphone permission? BlackHole not running?)")
    while not deck.msg.empty():
        kind, val = deck.msg.get()
        if kind == "rec_first":
            say("* recording the first layer... (press again to stop)")
        elif kind == "countin":
            say(f"count-in {val} beats... (recording starts on its own)")
        elif kind == "rec_bar1":
            say("* recording layer 1... (this is beat 1; one full lap ends it)")
        elif kind == "rec_layer":
            say(f"* recording layer {len(deck.layers) + 1}... (joins after one full lap)")
        elif kind == "first_done":
            say(f"loop length {val:.2f}s, looping")
        elif kind == "commit":
            deck.commit_layer()
        elif kind == "layer_done":
            say(f"layer {val} added")
        elif kind == "armed":
            say("waiting for the next beat 1...")
        elif kind == "cancel_arm":
            say("recording cancelled")
        elif kind == "cancel":
            say("layer cancelled (did not complete a lap)")
        elif kind == "too_short":
            say("too short (<0.25 s), discarded")
        elif kind == "auto_stop":
            say(f"! hit the {max_loop_s:.0f}s ceiling, recording stopped")
            deck._commit_first()


def _act(deck, c):
    """One key, one action. Returns the text to show, None when there is
    nothing to say, or "QUIT"."""
    if c == " ":
        deck.toggle(); return None
    if c == "u":
        n = deck.undo(); return f"undo: {n} layers left" if n else "undo: now empty"
    if c == "c":
        deck.clear(); return "cleared"
    if c == "m":
        deck.muted = not deck.muted
        return "muted" if deck.muted else "unmuted"
    if c in ("q", "\x03"):
        return "QUIT"
    return None


# -- interface 1: window, the default ----------------------------------
def run_ui(deck, a, i_name, o_name, rs, ps):
    import tkinter as tk

    BG, PANEL, FG, DIM = "#14161a", "#1c2027", "#e8eaed", "#7a8290"
    IDLE, REC, PLAY, MUTE = "#5a6472", "#e5484d", "#3dd68c", "#e8a33d"

    root = tk.Tk()
    root.title("Layered bed")
    root.configure(bg=BG)
    root.geometry("470x600")
    root.minsize(430, 520)
    if not a.no_top:
        root.attributes("-topmost", True)

    # -- status --
    head = tk.Frame(root, bg=BG)
    head.pack(pady=(16, 0))
    lamp = tk.Label(head, text="", font=("Helvetica", 40, "bold"), bg=BG, fg=IDLE)
    lamp.pack(side="left")
    # a flash on every wrap makes the start of a lap visible, so it need not be counted
    flash = tk.Label(head, text="●", font=("Helvetica", 26), bg=BG, fg=BG)
    flash.pack(side="left", padx=(10, 0))

    sub = tk.Label(root, text="", font=("Helvetica", 14), bg=BG, fg=DIM)
    sub.pack()

    cv = tk.Canvas(root, height=26, bg="#20242b", highlightthickness=0)
    cv.pack(fill="x", padx=24, pady=(14, 2))
    bar = cv.create_rectangle(0, 0, 0, 26, fill=IDLE, width=0)
    grid_ids = []

    # Beat lamps: one per beat, with beat 1 accented in another colour. The
    # first track can be sung against them.
    dots = tk.Canvas(root, height=26, bg=BG, highlightthickness=0)
    dots.pack(fill="x", padx=24, pady=(6, 0))
    dot_ids = []

    beatrow = tk.Frame(root, bg=BG)
    beatrow.pack(pady=(4, 6))
    tk.Label(beatrow, text="Tempo", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    pv = tk.StringVar(value=f"{deck.bpm:.0f}")
    ent_bpm = tk.Entry(beatrow, textvariable=pv, width=4, justify="center",
                       font=("Helvetica", 13))
    ent_bpm.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="x loop", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    bv = tk.StringVar(value=str(deck.beats))
    ent = tk.Entry(beatrow, textvariable=bv, width=4, justify="center",
                   font=("Helvetica", 13))
    ent.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="beats", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    tk.Label(beatrow, text="  count-in", font=("Helvetica", 12), bg=BG,
             fg=DIM).pack(side="left")
    cinv = tk.StringVar(value=str(deck.cin_beats))
    ent_cin = tk.Entry(beatrow, textvariable=cinv, width=3, justify="center",
                       font=("Helvetica", 13))
    ent_cin.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="beats", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    beatlbl = tk.Label(beatrow, text="", font=("Helvetica", 11), bg=BG, fg=DIM)
    beatlbl.pack(side="left", padx=(8, 0))

    # -- track list --
    tk.Label(root, text="Tracks", font=("Helvetica", 12), bg=BG, fg=DIM,
             anchor="w").pack(fill="x", padx=24, pady=(6, 2))
    tracks = tk.Frame(root, bg=PANEL)
    tracks.pack(fill="both", expand=True, padx=24)

    # -- buttons --
    big = tk.Button(root, text="Record / Stop  (space)", height=2,
                    highlightbackground=BG, command=lambda: press(" "))
    big.pack(fill="x", padx=24, pady=(10, 4))
    btns = tk.Frame(root, bg=BG)
    btns.pack(pady=(2, 6))
    for text, key in (("Undo (u)", "u"), ("Clear (c)", "c"), ("Mute (m)", "m")):
        tk.Button(btns, text=text, width=9, highlightbackground=BG,
                  command=lambda k=key: press(k)).pack(side="left", padx=4)

    # -- output device, changeable during a performance --
    # An aggregate device whose members were unplugged stays in the list with no
    # output channels, so list only devices that can really sound. BlackHole is
    # excluded outright: sending the bed back into it records itself.
    outs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0
            and "blackhole" not in d["name"].lower()]
    olabels = [f"{i}｜{nm}" for i, nm in outs]
    cur = next((l for l, (i, nm) in zip(olabels, outs) if nm == o_name),
               olabels[0] if olabels else o_name)

    devrow = tk.Frame(root, bg=BG)
    devrow.pack(side="bottom", pady=(2, 6))
    tk.Label(devrow, text=f"in: {i_name}   out:", font=("Helvetica", 10),
             bg=BG, fg=DIM).pack(side="left")
    ovar = tk.StringVar(value=cur)
    om = tk.OptionMenu(devrow, ovar, *(olabels or [cur]),
                       command=lambda v: switch_out(v))
    om.configure(font=("Helvetica", 10), highlightthickness=0)
    om.pack(side="left")

    stream = {"ps": ps, "label": cur}

    def switch_out(label):
        """Change speakers: stop the old stream, then open the new one. If the new
        one will not open, the old one is restored, so there is never silence."""
        if label == stream["label"]:
            return
        idx = int(label.split("｜")[0])
        old = stream["ps"]
        try:
            old.stop()
        except Exception:
            pass
        try:
            new = sd.OutputStream(device=idx, channels=a.ch, samplerate=a.sr,
                                  blocksize=a.block, dtype="float32",
                                  callback=deck.play_cb)
            new.start()
        except Exception as e:
            try:
                old.start()
            except Exception:
                pass
            ovar.set(stream["label"])
            say(f"could not switch ({e.__class__.__name__}); keeping the previous device")
            return
        try:
            old.close()
        except Exception:
            pass
        stream["ps"], stream["label"] = new, label
        nm = label.split("｜")[1]
        if "bh" in nm.lower():                  # a name containing bh sends the bed back into BlackHole
            say(f"! {nm} sends the bed back into BlackHole; the next track would record itself")
        else:
            say(f"now sounding through {nm}")

    msg = {"t": ""}
    # The canvas width is still 1 when the window has just opened, before layout,
    # so the grid and the beat lamps remember the width they were drawn at and
    # redraw when it changes; otherwise everything is squeezed out of sight.
    ui = {"sig": None, "last_p": 0, "flash": 0, "gw": 0, "dw": 0}

    def say(t):
        msg["t"] = t

    def press(c):
        """A button press. Always acts, and takes focus off the beats field; without
        that, space would still be swallowed by the field after a button press."""
        root.focus_set()
        r = _act(deck, c)
        if r == "QUIT":
            close()
        elif r:
            say(r)

    def on_key(c):
        """A key press. Only this path yields to the beats field: while the caret is
        in the field, space types rather than records. Mouse buttons do not come
        through here and are unaffected."""
        if root.focus_get() in (ent, ent_bpm, ent_cin) \
                and c in (" ", "u", "c", "m", "q"):
            return
        press(c)

    def close():
        try:
            rs.stop(); rs.close()
            stream["ps"].stop(); stream["ps"].close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    for k in (" ", "u", "c", "m", "q"):
        root.bind(f"<KeyPress-{'space' if k == ' ' else k}>", lambda e, k=k: on_key(k))
    # Enter in the beats field means it is filled in: focus is handed back and
    # space records again at once
    for e_ in (ent, ent_bpm, ent_cin):
        e_.bind("<Return>", lambda e: root.focus_set())
        e_.bind("<Escape>", lambda e: root.focus_set())

    def rebuild_tracks():
        for w in tracks.winfo_children():
            w.destroy()
        if not deck.layers:
            tk.Label(tracks, text="no tracks yet", font=("Helvetica", 12),
                     bg=PANEL, fg=DIM).pack(pady=14)
            return
        for i in range(len(deck.layers)):
            on = deck.layer_on[i]
            row = tk.Frame(tracks, bg=PANEL)
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=f"{i + 1}", font=("Helvetica", 13, "bold"), width=2,
                     bg=PANEL, fg=(FG if on else DIM)).pack(side="left")
            tk.Label(row, text=f"{deck.layers[i].shape[0] / deck.sr:.2f}s",
                     font=("Helvetica", 12), bg=PANEL,
                     fg=(FG if on else DIM)).pack(side="left", padx=(6, 0))
            tk.Button(row, text="del", width=3, highlightbackground=PANEL,
                      command=lambda i=i: (deck.drop(i), say(f"deleted track {i + 1}"))
                      ).pack(side="right", padx=2)
            tk.Button(row, text=("on" if on else "off"), width=3,
                      highlightbackground=PANEL,
                      command=lambda i=i: (deck.toggle_layer(i), say(""))
                      ).pack(side="right", padx=2)

    def draw_grid(n, w):
        for gid in grid_ids:
            cv.delete(gid)
        grid_ids.clear()
        for k in range(1, max(n, 0)):
            x = w * k / n
            grid_ids.append(cv.create_line(x, 0, x, 26, fill="#3a4150", width=1))

    def tick():
        pump(deck, a.max_loop_s, say)
        sr = deck.sr

        try:
            deck.beats = max(0, min(64, int(bv.get() or 0)))
        except ValueError:
            deck.beats = 0
        try:
            deck.cin_beats = max(0, min(16, int(cinv.get() or 0)))
        except ValueError:
            deck.cin_beats = 0
        if not deck.layers:            # the tempo locks once a track exists, or the grid would break
            try:
                deck.bpm = max(0.0, min(300.0, float(pv.get() or 0)))
            except ValueError:
                deck.bpm = 0.0
            ent_bpm.configure(state="normal")
        elif str(ent_bpm.cget("state")) != "readonly":
            ent_bpm.configure(state="readonly")

        if deck.cin > 0:
            bl = deck.beat_len() or 1
            col, txt = MUTE, f"count-in {-(-deck.cin // bl)}"
            done, total = (deck.cin_all - deck.cin) / sr, (deck.cin_all or 1) / sr
            sub.configure(text="recording starts when the count ends  (press again to cancel)")
        elif deck.armed:
            col, txt = MUTE, "waiting for beat 1"
            total = deck.L / sr if deck.L else 1
            done = (deck.clk % deck.L) / sr if deck.L else 0
            sub.configure(text=f"recording starts at the top of the next lap  (press again to cancel)")
        elif deck.rec:
            col, txt = REC, "recording"
            if deck.first:
                done, total = deck.pend_n / sr, a.max_loop_s
                sub.configure(text=f"first track  {done:.1f}s  (space to stop)")
            else:
                done, total = deck.wdone / sr, deck.L / sr
                lab = f"layer {len(deck.layers) + 1}" if deck.layers else "track 1"
                sub.configure(text=f"{lab}  {total - done:.1f}s left")
        elif deck.muted:
            col, txt = MUTE, "muted"
            done, total = (deck.p / sr, deck.L / sr) if deck.L else (0, 1)
            sub.configure(text="the bed is gone for now; press m to bring it back")
        elif deck.L:
            col, txt = PLAY, "looping"
            done, total = deck.p / sr, deck.L / sr
            if deck.beats:
                b = int(done / total * deck.beats) + 1
                sub.configure(text=f"beat {b} / {deck.beats}  -  {done:.1f} / {total:.1f}s")
            else:
                sub.configure(text=f"{done:.1f} / {total:.1f}s")
        elif deck.bpm > 0 and deck.beats > 0:
            # Nothing recorded yet but the metronome is running: sing along and a
            # press to record will line up
            col, txt = IDLE, "metronome"
            tl = deck.target_L() or 1
            done, total = (deck.clk % tl) / sr, tl / sr
            sub.configure(text=f"{deck.bpm:.0f} BPM  loop {total:.2f}s  press record and sing along")
        else:
            col, txt = IDLE, "idle"
            done, total = 0, 1
            sub.configure(text="set a tempo and beats, or just press space to record freely")

        lamp.configure(text=txt, fg=col)

        # a wrap flashes
        if deck.L and deck.p < ui["last_p"]:
            ui["flash"] = 4
        ui["last_p"] = deck.p
        if ui["flash"] > 0:
            flash.configure(fg=col)
            ui["flash"] -= 1
        else:
            flash.configure(fg=BG)

        w = max(cv.winfo_width(), 1)
        cv.coords(bar, 0, 0, w * min(done / total if total else 0, 1.0), 26)
        cv.itemconfigure(bar, fill=col)
        if len(grid_ids) != max(deck.beats - 1, 0) or ui["gw"] != w:
            draw_grid(deck.beats, w)
            ui["gw"] = w
        for gid in grid_ids:
            cv.tag_raise(gid)

        sig = (len(deck.layers), tuple(deck.layer_on))
        if sig != ui["sig"]:
            ui["sig"] = sig
            rebuild_tracks()

        # -- beat lamps --
        n = deck.beats
        span = deck.L or deck.target_L()
        dw = max(dots.winfo_width(), 1)
        if len(dot_ids) != n or ui["dw"] != dw:
            for did in dot_ids:
                dots.delete(did)
            dot_ids.clear()
            ui["dw"] = dw
            for k in range(n):
                x = dw * (k + 0.5) / n
                dot_ids.append(dots.create_oval(x - 7, 5, x + 7, 19,
                                                fill=BG, outline="#3a4150", width=2))
        if n and span:
            if deck.cin > 0:
                bl = deck.beat_len() or 1
                cur = ((deck.cin_all - deck.cin) // bl) % n
            else:
                pos = (deck.clk % span) if deck.bpm > 0 else deck.p
                cur = int(pos / span * n) % n
            for k, did in enumerate(dot_ids):
                if k == cur:
                    dots.itemconfigure(did, fill=(FG if k == 0 else col),
                                       outline=(FG if k == 0 else col))
                else:
                    dots.itemconfigure(did, fill=BG, outline="#3a4150")

        on_n = sum(1 for x in deck.layer_on if x)
        if deck.layers:
            beatlbl.configure(text=f"{on_n}/{len(deck.layers)} tracks sounding  -  tempo locked")
        else:
            beatlbl.configure(text="0 = free length")

        root.after(50, tick)

    rebuild_tracks()
    tick()
    root.mainloop()
    print(f"[loop_deck] finished (xruns {deck.xruns})")


# -- interface 2: terminal keys (--no-ui) ------------------------------
def run_keys(deck, a, rs, ps):
    import select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            pump(deck, a.max_loop_s, print)
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            out = _act(deck, sys.stdin.read(1))
            if out == "QUIT":
                break
            if out:
                print(out)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        rs.stop(); rs.close()
        ps.stop(); ps.close()
        print(f"\n[loop_deck] finished (xruns {deck.xruns})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dev", default="BlackHole", help="input source (default BlackHole)")
    ap.add_argument("--out-dev", default=None, help="where the bed is sent (default: the system output)")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--ch", type=int, default=2)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--max-loop-s", type=float, default=MAX_LOOP_S)
    ap.add_argument("--lat-ms", type=float, default=0.0,
                    help="input latency compensation; raise it if layers sound late")
    ap.add_argument("--gain", type=float, default=1.0, help="overall level of the bed")
    ap.add_argument("--list", action="store_true", help="list the devices and exit")
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="tempo; setting it gives a metronome and computes the first track length (0 = free length)")
    ap.add_argument("--beats", type=int, default=0, help="beats per loop")
    ap.add_argument("--no-ui", action="store_true", help="no window; use terminal keys")
    ap.add_argument("--no-top", action="store_true", help="do not keep the window on top")
    a = ap.parse_args()

    if a.list:
        for i, d in enumerate(sd.query_devices()):
            print(f"{i:2d} | {d['name']:32s} | in {d['max_input_channels']} | out {d['max_output_channels']}")
        return

    i_dev = _resolve(a.in_dev, "in")
    o_dev = _resolve(a.out_dev, "out") if a.out_dev else sd.default.device[1]
    i_name = sd.query_devices(i_dev)["name"]
    o_name = sd.query_devices(o_dev)["name"]

    if "blackhole" in o_name.lower():
        sys.exit(f"output device {o_name} would record itself. Choose another output device.")

    # Terminal mode is meaningless without a keyboard; window mode has its own
    # and is unaffected. Running in the background or in a pipe raises a
    # traceback from tcgetattr, so it is caught here and said plainly.
    if a.no_ui and not sys.stdin.isatty():
        sys.exit("no keyboard available (this is not a terminal window).\n"
                 "   Open it with the loop_deck launcher, or run it in a terminal.")

    deck = LoopDeck(a.sr, a.ch, a.block, a.max_loop_s, a.lat_ms, a.gain)
    deck.bpm = max(0.0, a.bpm)
    deck.beats = max(0, a.beats)

    print(f"[loop_deck] in:  {i_name}")
    print(f"[loop_deck] out: {o_name}")
    print(f"[loop_deck] {a.sr}Hz {a.ch}ch block={a.block} latency-comp={a.lat_ms}ms ceiling={a.max_loop_s:.0f}s")
    if deck.target_L():
        print(f"[loop_deck] metronome {deck.bpm:.0f} BPM x {deck.beats} beats "
              f"= loop {deck.target_L() / a.sr:.2f}s")
    print("[loop_deck] space=record/stop  u=undo  c=clear  m=mute  q=quit")

    rs = sd.InputStream(device=i_dev, channels=a.ch, samplerate=a.sr,
                        blocksize=a.block, dtype="float32", callback=deck.rec_cb)
    ps = sd.OutputStream(device=o_dev, channels=a.ch, samplerate=a.sr,
                         blocksize=a.block, dtype="float32", callback=deck.play_cb)
    rs.start()
    ps.start()
    print("[loop_deck] running")

    if a.no_ui:
        run_keys(deck, a, rs, ps)
    else:
        run_ui(deck, a, i_name, o_name, rs, ps)


if __name__ == "__main__":
    main()
