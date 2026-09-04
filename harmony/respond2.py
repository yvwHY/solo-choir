"""respond2.py — the answering mode v2: sing a phrase, and the response plays
offline-quality harmony together with the phrase they have just sung

The user experience asked for on 2026-08-01: "the user sings a phrase as input,
and the output is the rendered harmony of the parts' response plus the phrase the
user has just sung".

Why this shape survives, while it sidesteps both of the things that killed the
pre-rendered rehearsal line the same day:
  - **There is no alignment problem**: the harmony is made for the phrase they
    have just finished, so the notes are always right (rehearsal mode died at a
    tracking hit rate of 37.9%).
  - **There is no simultaneity pressure**: the response comes after them by
    definition, which is call-and-response grammar rather than latency.
  - **The gap between phrases is long enough to run the offline reference
    render** (phrase_render keeps the model resident; its RTF is documented
    there).
  - Their voice is inside the response, which is the answering form of the hero
    loop, hearing yourself become a choir.

Red line (G "--voice-delay" was killed on 2026-07-16: playing a delayed copy of
their own voice is delayed auditory feedback, and every listening test failed):
**the response plays only while they are not singing, and fades out of the way
the moment they open their mouth**. This is a hard rule in live mode, not an
option. The difference is that voice-delay layered while they sang, which is the
premise of delayed auditory feedback, while this answers in the gaps.

The hard mute in live mode (settled with the front end for the exhibition on
2026-08-01, an air microphone, a FIFINE K669B): **from the instant a phrase ends,
the input is completely dead until the response has played and its reverb tail
has decayed.** This is not "dodge once feedback is detected" but a structure in
which singing and playback are never alive at the same time, which removes
feedback, the speakers being mistaken for singing, and the response bleeding into
the next phrase, all at once. The cost is that the microphone cannot hear them
during that window, so **the physical button is their only channel while muted**
(`tap_listen`): pressed while listening it ends the phrase at once without
waiting for the gap, and pressed while muted it cancels the response and returns
to listening immediately. With no button, it degrades to energy detection alone
and blocks nothing.

Phrase-end detection shares one set of criteria with file mode
(`find_phrases` and `PhraseGate` mirror each other): a 10 ms RMS window over the
gate counts as voiced, silence of gap or longer ends the phrase, and islands
shorter than min-phrase are discarded. In live mode the gate is **measured on the
spot** at the start, taking the midpoint in the log domain between the p95 of the
noise floor in silence and the p75 of the singing, rather than inheriting the
0.02 of the throat-microphone material. Inherited parameters have to be verified;
that lesson was paid for twice on 2026-08-01.
The model is BrainV3, continuous across phrases so it remembers the song, with
indep 0 and reh_polish (the composition layer of 2026-08-01 section G).
The key must be given as an integer by hand (the lesson of 2026-08-01 section H:
keydet is unusable on real songs, and the bench terms must not be contaminated).

  python respond2.py --file take.wav out/resp2.wav --key 3      # offline
  python respond2.py --live --key 3 --in-name K669B             # live
  python respond2.py --live --list-devices                      # list device names
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import direct_mouth as dm  # noqa: E402
import reverb as rv  # noqa: E402
from live_v3 import EarV3  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from reh_polish import polish  # noqa: E402
from tap_listen import SerialTapListener, TapListener  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
# The phrase gate cannot use pitch.py's 0.005, which answers "is there a pitch".
# Breath noise passes 0.005, so the whole take left only two gaps of 0.6 s or
# more and produced a single "phrase" of 34.6 s (measured 2026-08-01).
# Singing has a median rms of 0.15 and breaths sit near 0.01, so 0.02 separates
# them; with a gap of 0.35 that gives phrases with a median length of 2.4 s. Their
# breaths measure 0.3-0.5 s; the 0.6 s came from the older respond on different
# material.
RMS_GATE = 0.02
# After the serial port opens, wait this long for one TAP or HB line; otherwise
# treat it as "not that button" and fall back to UDP. The firmware sends a
# heartbeat every 2 s, so a real button usually proves itself in under 2 s.
TAP_WAIT = 5.0
VOICES = {"upper": (f"{DDSP}/exp/combsub-girl/model_30000.pt", 20260731),
          "lower": (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", 20260805)}
# The recipe finally passed on 2026-08-05 section F: lower is Bass-1, a real bass
# from M4Singer, and both parts go through the NSF-HiFiGAN enhancer. That is the
# conclusion of the auto-tune case: the fault was the flat, lifeless harmonics of
# the bare CombSub output in the low range, the enhancer cures it, and f0 and the
# skeleton were innocent. The old lower, their own model, was:
#   "lower": (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt", 20260732)
# The S part, 2026-08-05 section G, guarded by --sop; at 0 the settled two-part
# recipe is byte-identical. M4Singer Soprano-3, at the validation minimum of 10k.
# Its line is their microphone pitch plus 12, since the melody on top is the choral
# convention. S3 has a p10 of 67 and their median of 54+12 = 66 sits at the bottom
# of that range, so listen before adjusting.
SOP = (f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 20260806)
# Vibrato desynchronisation, at the original values of render_v3.py:339-340. Two
# parts at the same phase and rate are mechanically synchronised, which is one
# source of the auto-tuned quality (2026-07-28). Used by --f0-mode target.
VIB = {"upper": (5.3, 0.0), "lower": (4.6, 0.5), "sop": (4.9, 0.75)}


def find_phrases(x, gap_s=0.6, min_s=0.5, win=441):
    """Sample ranges as (start, end). The criteria of the older
    respond.py:50-68, verified against a real singer: a 10 ms RMS window over the
    gate counts as voiced, silence of gap_s or longer ends the phrase, and islands
    shorter than min_s are discarded."""
    n = len(x) // win
    loud = np.array([np.sqrt(np.mean(x[i * win:(i + 1) * win] ** 2)) > RMS_GATE
                     for i in range(n)])
    gap_w, out, i = int(gap_s * SR / win), [], 0
    while i < n:
        if not loud[i]:
            i += 1
            continue
        j = i
        run = 0
        while j < n and run < gap_w:
            j += 1
            run = run + 1 if j < n and not loud[j] else 0
        end = j - run
        if (end - i) * win >= min_s * SR:
            out.append((i * win, min(len(x), end * win)))
        i = j
    return out


def phrase_notes(ear, seg):
    """One phrase of audio to (lead, upper, lower) per tick. The model is
    continuous across phrases, so it remembers the song; only voiced spans are fed
    in, and silence never enters its context."""
    up, lo = [], []
    n0 = len(ear.lead_hist)
    for k in range(len(seg) // TICK_SAMPS):
        for u, l in ear.tick(seg[k * TICK_SAMPS:(k + 1) * TICK_SAMPS]):
            up.append(u)
            lo.append(l)
    return ear.lead_hist[n0:], up, lo


class PhraseGate:
    """The streaming form of find_phrases: push(chunk) returns the phrase audio,
    with its trailing silence removed, once a phrase is complete, and None
    otherwise. The criteria match find_phrases item by item and the parameters come
    from the same CLI, so the two cannot drift apart (the lesson of 2026-08-01:
    inherited parameters have to be verified)."""

    WIN = 441                                    # 10 ms at 44.1 kHz, as in find_phrases

    def __init__(self, gate, gap_s, min_s):
        self.gate = gate
        self.min_w = int(min_s * SR / self.WIN)
        self.lv = []                             # history of window rms, for display and start-up calibration
        self.set_gap(gap_s)
        self.reset()

    def set_gap(self, gap_s):
        """Seconds of silence needed to end a phrase. With the button connected
        this is stretched into a safety net; see live()."""
        self.gap_s = gap_s
        self.gap_w = int(gap_s * SR / self.WIN)

    def reset(self):
        self.res = np.zeros(0)                   # the remainder that does not fill a window
        self.buf = []                            # the windows of this phrase, trailing silence included
        self.run = 0                             # consecutive silent windows
        self.on = False
        self.taken = 0                           # windows already handed to Prefetch

    def committed(self):
        """Samples not yet handed over that are **guaranteed to stay in this
        phrase**, for pipelining (2026-08-03).

        Two guarantees; without either, pipelining would receive audio belonging
        to another phrase:
          1. Trailing trim: what is trimmed is always the **current** run of
             trailing silence. push trims one more, the last voiced window, so
             run+1, while force_end trims run. The trim point is therefore always
             at or after `len(buf) - run - 1`: any window already before that
             bound cannot be trimmed later, whether the silence continues, leaving
             the trim point where it is, or they sing again, moving it later.
             **gap_w cannot be used as the reserve**: with the button connected,
             gap_w is 6.0 s (--gap-tap), so a phrase shorter than 6.5 s could hand
             over nothing at all and pipelining would effectively be off (review
             A5/A4/A2, 2026-08-12; the 6.0 s safety net for automatic phrase ends
             is unaffected).
          2. This phrase will certainly be handed over: once its length passes
             min_w, end is at least min_w on both the push and force_end paths, so
             it cannot take the "too short, treat as a mis-press, discard" branch."""
        if not self.on:
            return np.zeros(0)
        end = len(self.buf) - self.run - 1
        if end < max(1, self.min_w) or end <= self.taken:
            return np.zeros(0)
        out = np.concatenate(self.buf[self.taken:end])
        self.taken = end
        return out

    def force_end(self):
        """The physical button: end the phrase now without waiting for the gap,
        discarding trailing silence. Too short counts as a mis-press, so the phrase
        in hand is discarded and listening continues; by the principle that the
        button never blocks, a wrong press cannot lock the system up."""
        if not self.on:
            return None
        end = len(self.buf) - self.run
        seg = (np.concatenate(self.buf[:end])
               if end >= max(1, self.min_w) else None)
        self.on, self.buf, self.run, self.taken = False, [], 0, 0
        return seg

    def push(self, x):
        x = np.concatenate([self.res, x])
        n = len(x) // self.WIN
        self.res = x[n * self.WIN:]
        for i in range(n):
            w = x[i * self.WIN:(i + 1) * self.WIN]
            self.lv.append(float(np.sqrt(np.mean(w ** 2))))
            loud = self.lv[-1] > self.gate
            if not self.on:
                if loud:
                    self.on, self.buf, self.run = True, [w], 0
                continue
            self.buf.append(w)
            self.run = 0 if loud else self.run + 1
            if self.run < self.gap_w:
                continue
            # find_phrases has end = j - run, which is one voiced window, 10 ms,
            # short. That -1 is copied here deliberately: both modes must cut the
            # same phrase boundary, or the listening conclusions from file mode
            # would not hold for live. Change both sides together or neither.
            end = len(self.buf) - self.run - 1
            seg = (np.concatenate(self.buf[:end])
                   if end >= max(1, self.min_w) else None)
            self.on, self.buf, self.run, self.taken = False, [], 0, 0
            if seg is not None:
                self.res = x[(i + 1) * self.WIN:]    # unprocessed windows are kept for next time
                return seg
        return None


class KeyTracker:
    """Accumulate the raw-f0 pitch-class distribution phrase by phrase into a
    rotation angle (2026-08-04, `--key auto`).

    **Why this can be automatic now when it was killed on 2026-07-30 and
    2026-07-31**: neither of those was killed for being inaccurate but because
    **being wrong destroyed the whole evening** (on 2026-07-31 a confidence of
    0.01 was hard-locked at +4 semitones and the key was wrong all evening). Once
    polish's reference changed to their real pitch on 2026-08-04, the cost of a
    wrong angle fell from 37.7% to 17.9%, against 17.3% for the right key. Total
    ruin became almost painless, which is what makes automatic estimation worth
    it. The model could always handle every key, since harmony is invariant under
    transposition and the corpus is normalised to C and A minor, rotating in and
    back out; the angle was the only thing missing.

    What is measured is harvest's raw f0, which pipelining computes anyway, at no
    extra cost, and **deliberately not the output of VoiceToTokens**: that layer
    snaps notes onto a scale, so measuring the key from it is circular reasoning.
    The claim that "the lead is 100% inside C" on 2026-08-01 section H came about
    exactly that way."""

    MAJOR = (0, 2, 4, 5, 7, 9, 11)
    MIN_FRAMES = 150                  # judge only after about 1.7 s of sounding

    def __init__(self):
        self.h = np.zeros(12)

    def push(self, f0m):
        v = np.asarray(f0m)
        v = v[v > 0]
        if len(v):
            np.add.at(self.h, np.round(69 + 12 * np.log2(v / 440.0)).astype(int) % 12, 1.0)

    def best(self):
        """(key_shift, coverage, lead over the runner-up in percentage points);
        None when there is too little data."""
        t = self.h.sum()
        if t < self.MIN_FRAMES:
            return None
        cov = np.array([sum(self.h[(r + d) % 12] for d in self.MAJOR) / t
                        for r in range(12)])
        o = np.argsort(cov)[::-1]
        root = int(o[0])
        sh = (0 - root) % 12
        return (sh - 12 if sh > 6 else sh), float(cov[root]), float(cov[o[0]] - cov[o[1]])


class Prefetch:
    """Pipelining (2026-08-03): compute the parts that are byte-identical whether
    computed early or late, while they are still singing.

    The breakdown of 2026-08-03, on a 5.16 s phrase for one part: harvest 0.59 s,
    per-frame voicing 0.41 s, HuBERT encode 0.54 s, and the model forward pass only
    0.06 s. The money is in the pre-processing, not the model. Of those:
      **can be computed early** (computing in segments is byte-identical to
      computing the whole)
        1. the model (EarV3), which already streams tick by tick, with the same
           samples in the same order and therefore the same result
        2. the per-frame voicing features, since each frame looks only at plus or
           minus 1024 samples (dm.voicing_frames)
      **cannot be computed early** (non-causal over the whole span; cutting it
      gives a different voice)
        harvest, whose path search spans the whole phrase; the HuBERT encode, with
        full attention across the transformer; and _local_ref, over a window of
        plus or minus 3 s. These stay at the end of the phrase and run over the
        whole span; better to wait another 0.6 s than to change a sound that has
        already passed a listening test. The model forward pass is not cut for the
        same reason, so there are no joins and no phase jumps.

    It consumes only the samples handed over by PhraseGate.committed(), which are
    guaranteed to belong to this phrase; see that method.
    It runs on a background thread while the main loop goes on capturing and
    detecting phrase ends, which must not become sluggish because a computation is
    running.
    """

    def __init__(self, ear):
        self.ear = ear
        self.q = queue.Queue()
        self.done = threading.Event()
        self.out = None
        self.fed = 0                      # samples fed in by the main thread, for the statistics
        self._reset()
        threading.Thread(target=self._run, daemon=True).start()

    def _reset(self):
        self.x = np.zeros(0)              # the samples of this phrase so far
        self.n0 = len(self.ear.lead_hist)  # the model's starting point, the n0 of phrase_notes
        self.up, self.lo = [], []
        self.n_tick = 0                   # ticks already fed to the model
        self.fr = [np.zeros(0)] * 3       # per / flat / rms
        self.n_fr = 0                     # voicing frames already computed

    def feed(self, x):
        self.fed += len(x)
        self.q.put(("feed", x))

    def abandon(self):
        """This phrase does not count, for example after a cancelling press, so
        the half-finished work in hand is discarded. What the model has already
        heard is kept: they really did sing those ticks, and a context that is
        continuous across phrases ought to remember them."""
        self._sync(("reset", None))
        self.fed = 0

    def resync(self):
        """Re-snapshot the model's starting point, n0 and the rest. Used after
        the tail loop in make_response: tail_tick keeps appending to ear.lead_hist
        after finish(), and without updating n0 the next phrase's lead would begin
        with the previous phrase's n_tail trailing notes (review #20,
        2026-08-17)."""
        self._sync(("reset", None))
        self.fed = 0

    def finish(self, seg):
        """End of phrase: finish the remaining work and return (notes, shared)."""
        self._sync(("finish", seg))
        d, sh = self.out
        self._sync(("reset", None))
        self.fed = 0
        return d, sh

    def _sync(self, msg):
        self.done.clear()
        self.q.put(msg)
        self.done.wait()

    def _run(self):
        while True:
            kind, payload = self.q.get()
            try:
                if kind == "feed":
                    self._feed(payload)
                    continue
                if kind == "finish":
                    self.out = self._finish(payload)
                else:
                    self._reset()
            except Exception as e:                 # pipelining blocks nothing: fall back to recomputing the whole phrase
                print(f"⚠ prefetch failed ({e!r}) → recomputing this phrase",
                          flush=True)
                self._reset()
                self.out = (None, {})
            self.done.set()

    def _feed(self, x):
        self.x = np.concatenate([self.x, x])
        self._brain(len(self.x) // TICK_SAMPS)
        self._frames(self.x, dm.frames_ready(len(self.x)))

    def _brain(self, k):
        while self.n_tick < k:
            s = self.x[self.n_tick * TICK_SAMPS:(self.n_tick + 1) * TICK_SAMPS]
            for u, l in self.ear.tick(s):
                self.up.append(u)
                self.lo.append(l)
            self.n_tick += 1

    def _frames(self, x, k1):
        if k1 <= self.n_fr:
            return
        new = dm.voicing_frames(x, self.n_fr, k1)
        self.fr = [np.concatenate([a, b]) for a, b in zip(self.fr, new)]
        self.n_fr = k1

    def _finish(self, seg):
        n = len(self.x)
        if n > len(seg) or not np.array_equal(seg[:n], self.x):
            # The two guarantees of committed() prevent this; if it happens, the
            # code here is wrong.
            print(f"⚠ prefetch prefix mismatch ({n} vs {len(seg)}) → dropping it,"
                      f" recomputing", flush=True)
            self._reset()
        self.x = seg
        self._brain(len(seg) // TICK_SAMPS)
        n_hops = len(seg) // dm.HOP + 1
        self._frames(seg, n_hops)                  # frames whose window is short at the end stay 0, the original semantics
        f0m = dm.harvest_f0(seg)
        sh = {"n": len(seg), "f0m": f0m,
              "agc": dm.input_agc(seg, f0m=f0m),
              "vm": dm.voicing_mask(seg, n_hops, f0m=f0m,
                                    frames=tuple(f[:n_hops] for f in self.fr))}
        d = {"lead": self.ear.lead_hist[self.n0:],
             "upper": self.up, "lower": self.lo}
        return d, sh


def tail_carrier(seg, vm, n_tail, fade=0.45):
    """Carrier audio for the tail: they have stopped sounding, but the voice needs
    content, since units, f0 and volume all come from audio. Looping **the last
    voiced span** of this phrase as the carrier makes the parts sustain the same
    vowel, which is the standard way a choir ends a phrase anyway.

    Only the content is borrowed; the pitch comes from the note line, since f0 is
    injected, so what loops is the timbre and the breath and not the melody. An
    equal-power crossfade hides the join, and the last fade proportion fades it
    out."""
    v = np.repeat(np.asarray(vm, dtype=float), dm.HOP)[:len(seg)] > 0.5
    idx = np.flatnonzero(v)
    if not len(idx):
        return np.zeros(n_tail)
    end = idx[-1] + 1
    src = seg[max(0, end - int(0.5 * SR)):end]        # the last 0.5 s of voiced audio
    if len(src) < int(0.05 * SR):
        return np.zeros(n_tail)
    xf = int(0.03 * SR)
    out, w = [], np.linspace(0, 1, xf)
    cur = src.copy()
    while sum(len(c) for c in out) < n_tail + len(src):
        if out:
            out[-1][-xf:] = out[-1][-xf:] * (1 - w) + cur[:xf] * w
            out.append(cur[xf:].copy())
        else:
            out.append(cur.copy())
    y = np.concatenate(out)[:n_tail]
    k = int(len(y) * (1 - fade))
    if k < len(y):
        y[k:] *= np.cos(np.linspace(0, np.pi / 2, len(y) - k)) ** 2
    return y


def make_response(seg, ear, rend, a, pre=None):
    """One of their phrases to the audio to play back, their dry phrase plus the
    two parts, and the render time in seconds.
    Shared by file and live mode, so there is only one recipe for the response.

    pre is the Prefetch of live pipelining: the model's note line and the x-only
    features were computed while they were singing, so only harvest, encode and the
    forward pass remain. Without it, the whole phrase is computed from scratch as
    before."""
    d, sh = (None, {}) if pre is None else pre.finish(seg)
    if d is None:
        lead, up, lo = phrase_notes(ear, seg)
        d = {"lead": lead, "upper": up, "lower": lo}
    n_tail = int(a.tail_s * SR / TICK_SAMPS)
    if n_tail:
        # Tail: they have stopped and the parts move on their own (2026-08-04,
        # item 2). The lead holds their last note.
        last = next((p for p in reversed(d["lead"]) if p is not None), None)
        for _ in range(n_tail):
            u, l = ear.tail_tick()
            d["lead"].append(last)
            d["upper"].append(u)
            d["lower"].append(l)
        if pre is not None:
            # Review #20, 2026-08-17: tail_tick appends to ear.lead_hist on every
            # step, while the reset in finish() snapshots n0 **before** the tail
            # loop. Without re-snapshotting, from the second phrase on, d["lead"]
            # begins with the previous phrase's n_tail trailing notes while up and
            # lo belong to the new one, so the soprano line, derived from the lead
            # and the most exposed of all, lags the whole phrase by tail_s seconds.
            # It appeared only with the length control (--tail-s) on, and produced
            # no error to trace.
            pre.resync()
    if "sop" in rend:
        # The S line is their microphone pitch plus 12. The lead is in model
        # space, so subtracting off converts back to microphone space; the lesson
        # of section B is to ask which coordinate system a quantity is in before
        # comparing across spaces. It is taken before polish rewrites d["lead"].
        # A shift of None means auto-octave has not locked yet, because
        # calibration was skipped or failed, or the first phrase was too short:
        # VoiceToTokens needs six voiced ticks. Treating None as 0, no shift, makes
        # the S line simply the microphone pitch plus 12, which is the only
        # defensible default here. The old code raised a TypeError and died on the
        # spot, taking the session dump with it (review S2/S4, 2026-08-12).
        off = (ear.v2t.shift or 0) + (ear.k_shift or 0)
        d["sop"] = [None if p is None else int(p) - off + 12
                    for p in d["lead"]]
    if a.polish:
        # polish's reference changed to **the pitch they actually sang**
        # (2026-08-04, after "should it not be key-free?"). It used to compare
        # against lead_hist, the line already snapped into C major by
        # VoiceToTokens and then transposed, which is the system's internal
        # representation and not what they sang. That line is self-consistent with
        # the parts, so the measured dissonance rate always looked good, 10.1% in
        # the wrong key against 10.6% in the right one, indistinguishable, while
        # what the ear heard was the parts clashing with **the real pitch**.
        # After changing the reference, over 8 phrases of pair06 and counted
        # against the real pitch:
        #     right key +3, old reference     17.3%
        #     wrong key 0, old reference      37.7%   <- what the ear heard
        #     wrong key 0, **new reference**  17.9%   <- almost level with the right key
        # That is, **the parts do not need to be told the key; they only need not
        # to clash with the notes actually being sung**. --key falls from "wrong
        # once, ruined entirely" to a hint worth 2.4 percentage points on the scale
        # term; ignoring the scale altogether costs 20.3%.
        if "f0m" not in sh:
            sh["n"], sh["f0m"] = len(seg), dm.harvest_f0(seg)
        if getattr(a, "keytrack", None) is not None:
            a.keytrack.push(sh["f0m"])       # reuses the same harvest, at no extra cost
        per = TICK_SAMPS / dm.HOP
        ref = []
        for k in range(len(d["lead"])):
            w = sh["f0m"][int(k * per):int((k + 1) * per)]
            w = w[w > 0]
            ref.append(None if len(w) < 3 else
                       int(round(float(np.median(69 + 12 * np.log2(w / 440.0))))))
        d["lead"] = ref + [None] * max(0, len(d["lead"]) - len(ref))
        root = (-(ear.k_shift or 0)) % 12
        polish(d, "upper", "lower", root)
        polish(d, "lower", "upper", root)
    t0 = time.perf_counter()
    # sh is the whole layer that looks only at x and is shared by both voices:
    # harvest, agc, voicing_mask, units and vol. Under pipelining it arrives
    # already carrying the f0m, agc and vm computed by prefetch.
    if a.f0_mode == "target":     # the blind_v3 recipe: the skeleton plus fixed desynchronised vibrato
        kw = {v: dict(expr_gain=0.0, vib_semi=a.vib_semi, vib_onset_ms=250.0,
                      vib_hz=VIB[v][0], vib_phase=VIB[v][1]) for v in rend}
    else:                         # express: the vibrato and the rest of the texture all come from the expression layer
        kw = {v: dict(expr_gain=a.expr, tune_seed=20260802,
                      tune_lock=a.tune_lock) for v in rend}
    for v in rend:                # the dry voice is already in the response mix, so the stems no longer pass it through (audit item 5, 2026-08-04)
        kw[v]["uv_dry"] = a.uv_dry
    x = seg
    if a.aah:
        # A plain "ah" (2026-08-11: "the immediate version should sing a plain
        # ah, not the words"). The parts' material is no longer their words:
        # tail_carrier extends from the tail to the whole phrase, looping the last
        # voiced vowel of this phrase as the carrier for all of it, so the parts
        # sing "ah" in their current timbre while the pitch still follows the
        # model's harmony line.
        # shared cannot be reused: it was computed over their original phrase
        # (harvest, agc, vm), while the carrier is different audio of the same
        # length, which the guard cannot catch, so it must be cleared here.
        sh = {}
        vm0 = dm.voicing_mask(seg, len(seg) // dm.HOP + 1)
        n_all = len(seg) + n_tail * TICK_SAMPS
        x = tail_carrier(seg, vm0, n_all,
                         fade=(0.45 * n_tail * TICK_SAMPS / n_all)
                         if n_tail else 0.0)
    elif n_tail:
        # The f0m, agc and vm computed early by prefetch are for **the original
        # phrase length** and no longer apply once the tail is added;
        # PhraseRenderer's guard catches that, and the guard is right. They are
        # discarded and recomputed. The model's note line is still pre-computed,
        # and that is the large part of what prefetch saves (RTF 0.31 against
        # 0.08).
        sh = {}
        vm0 = dm.voicing_mask(seg, len(seg) // dm.HOP + 1)
        x = np.concatenate([seg, tail_carrier(seg, vm0,
                                              n_tail * TICK_SAMPS)])
    ang = {v: rend[v].render(x, d[v], TICK_SAMPS, shared=sh, **kw[v])
           for v in rend}
    dt = time.perf_counter() - t0
    n = min(len(x), *(len(ang[v]) for v in rend))
    seg = np.concatenate([seg, np.zeros(n - len(seg))]) if n > len(seg) else seg
    # Do not change the order of addition on this line: --reverb 0 must be
    # byte-identical to the version before reverb existed, and splitting it into
    # dry plus parts changes the floating-point association order. The difference
    # is 1e-17, but it is no longer the same file.
    gu = a.angel_gain * 10 ** (a.upper_db / 20.0)
    gl = a.angel_gain * 10 ** (a.lower_db / 20.0)
    gs = a.angel_gain * 10 ** (getattr(a, "sop_db", 0.0) / 20.0)
    resp = (a.dry_gain * seg[:n] + gu * ang["upper"][:n]
            + gl * ang["lower"][:n])
    if "sop" in ang:              # appended at the end, so the bit path of --sop 0 is unchanged
        resp = resp + gs * ang["sop"][:n]
    if a.reverb > 0:
        angels = gu * ang["upper"][:n] + gl * ang["lower"][:n]
        if "sop" in ang:
            angels = angels + gs * ang["sop"][:n]
        # Send and return: the dry level is untouched and only the wet is added.
        # --reverb-all sends their phrase into the space too, gluing it into one
        # choir; by default only the parts are sent, so they are in front and the
        # parts are in the space.
        w = rv.wet(resp if a.reverb_all else angels, a.ir)
        resp = np.concatenate([resp, np.zeros(len(w) - len(resp))]) + a.reverb * w
    parts = None
    if getattr(a, "out_map", None):
        # Channel routing (live --out-map): each stem carries its own reverb.
        # Convolution is linear, so the sum is the same sound as the mixed version,
        # only placed on its own channel. The order is [dry, upper, lower], with
        # later parts appended. The mixed resp is still computed unchanged, so the
        # dump and file terms do not move.
        parts = [a.dry_gain * seg[:n], gu * ang["upper"][:n],
                 gl * ang["lower"][:n]]
        if "sop" in ang:
            parts.append(gs * ang["sop"][:n])
        if a.reverb > 0:
            parts = [q if (i == 0 and not a.reverb_all) else
                     np.concatenate([q, np.zeros(len(w) - len(q))])
                     + a.reverb * rv.wet(q, a.ir)
                     for i, q in enumerate(parts)]
        L = max(len(q) for q in parts)
        parts = [np.concatenate([q, np.zeros(L - len(q))]) for q in parts]
    return resp, dt, parts


def octave_from_calib(x):
    """The calibration singing to the auto-octave shift (2026-08-03).

    The original estimator took only **the first six voiced ticks, 1.1 seconds**,
    and that 1.1 seconds is their first breath into the song, where the breath,
    the attack rubbish and any low preparatory note all live. In file mode a wrong
    lock can be rerun; **on stage a wrong lock is wrong all evening** (a probe on
    2026-08-02 locked the same material at both +0 and +12).

    The second stage of calibration is already "sing for 5 seconds at performance
    level", which is 27 ticks of **deliberate singing**, and it happens before any
    response. The same estimator, 4.5 times the data, and a cleaner moment to
    sample. This is not a change of algorithm; the formula is word for word the
    same, aiming the median at 65 and moving only by octaves. Only what is fed to
    it changes.

    The shift is quantised to whole octaves, so an estimation error has to exceed
    six semitones to change the result, which makes it immune to small differences
    in the median. Using harvest rather than the model's ticks relies on that
    quantisation to absorb the difference."""
    if x is None or len(x) < SR:
        return None
    f0 = dm.harvest_f0(np.ascontiguousarray(x))
    v = f0[f0 > 0]
    if len(v) < 40:                       # under about 0.5 s of voiced audio means they did not sing; do not lock
        return None
    med = float(np.median(69 + 12 * np.log2(v / 440.0)))
    return int(12 * round((65 - med) / 12)), med


def build_ear_and_mouths(a):
    ear = EarV3(indep=a.indep, stab=a.stab, key=a.key)
    if a.octave is not None:
        # auto-octave (live.py:69-77) is decided from the median of the first six
        # voiced ticks and never changes again in that process: in file mode a
        # wrong lock can be rerun, while on stage it puts the parts an octave out
        # all evening. It also makes an A/B unclean, because moving the key moves
        # the median, and crossing a rounding boundary changes the result
        # (measured 2026-08-02: --key -2 locked +12 and --key 5 locked +0).
        # Pinning it keeps a single variable.
        ear.v2t.shift = a.octave
        print(f"[auto-octave] pinned to {a.octave:+d} semitones (--octave)")
    rend = {v: PhraseRenderer(DDSP, m, expr_seed=sd, enhance=True)
            for v, (m, sd) in VOICES.items()}
    if getattr(a, "sop", 0):
        rend["sop"] = PhraseRenderer(DDSP, SOP[0], expr_seed=SOP[1],
                                     enhance=True)
    for r in rend.values():                       # warm up the MPS cold start (2026-07-29 section G)
        r.render(np.zeros(SR), [60], TICK_SAMPS)
    return ear, rend


def calibrate(drain, a, beep=None):
    """Two measurements at the start giving the phrase-end threshold. The 0.02
    came from throat-microphone material (see the respond2 comment) and does not
    hold with an air microphone, a different room or a different gain:
    **inherited parameters have to be verified**, a lesson paid for twice on
    2026-08-01. So nothing is guessed and it is measured on the spot: the midpoint
    in the log domain between the p95 of the noise floor and the p75 of the
    singing.

    This also answers whether the input signal-to-noise ratio holds up, the open
    item of 2026-08-01: the separation is printed here.

    The beeps are audible prompts (2026-08-02 section I: on the first real test
    with a person, the process was put in the background, so **"sing now" could
    not be seen**, nobody sang during the second measurement, and the guard
    correctly fell back to the default. Standing and singing at an exhibition, the
    console is equally invisible, so the prompt has to come out of the speakers).
    The signals are:
      one low tone, the silent measurement begins; two high tones, sing now;
      three rising tones, calibration complete; one long low tone, failed.
    """
    raw = {}

    def measure(secs, label, cue=None, keep=None):
        print(label, flush=True)
        if beep is not None and cue is not None:
            beep(*cue)
            drain()                            # the prompt tone entered the microphone itself, so discard it
        lv, res, t0, buf = [], np.zeros(0), time.perf_counter(), []
        while time.perf_counter() - t0 < secs:
            x = np.concatenate([res, drain()])
            if keep is not None:
                # Keep only what arrived this round. res is the previous round's
                # remainder and was already appended, so appending it again pushes
                # a 0-10 ms fragment in twice; measured, that inflated the audio by
                # 55%, and both octave and key are harvested from it (review A1,
                # 2026-08-12).
                buf.append(x[len(res):])
            n = len(x) // PhraseGate.WIN
            res = x[n * PhraseGate.WIN:]
            lv += [float(np.sqrt(np.mean(x[i * PhraseGate.WIN:
                                           (i + 1) * PhraseGate.WIN] ** 2)))
                   for i in range(n)]
            time.sleep(0.01)
        if keep is not None:
            raw[keep] = np.concatenate(buf) if buf else np.zeros(0)
        return np.array(lv) if len(lv) else np.zeros(1)

    drain()                                    # discard the noise at the start
    q = measure(a.calib_quiet, f"1) Calibration: stay silent for "
                               f"{a.calib_quiet:.0f} s (measuring the noise "
                               f"floor)… [beep x1 = quiet starts]",
                cue=(440.0, 200, 1))
    s = measure(a.calib_sing, f"2) Calibration: sing for "
                              f"{a.calib_sing:.0f} s at normal performance "
                              f"volume (measuring your level)… [beep x2 = sing]",
                cue=(880.0, 160, 2), keep="sing")
    # Noise floor: a digitally silent input, from a muted interface or an
    # unplugged cable, makes the gate come out as 0, so everything counts as
    # voiced and no phrase can ever end. A real microphone is never 0, but an
    # unplugged one is. The floor itself must be **far below any real signal**:
    # 1e-4, which is -80 dBFS, condemned a perfectly good session at a low level
    # (review S2, 2026-08-12), while 1e-6 still catches an all-zero input.
    n95 = max(float(np.percentile(q, 95)), 1e-6)
    s75 = float(np.percentile(s, 75))
    if s75 <= n95:                             # they did not sing, or the microphone is not open
        # The fallback cannot be the absolute 0.02: at a genuinely low level it
        # sits 30-50 dB above the singing, so gate.on never becomes true and even
        # the physical button cannot rescue it, since force_end returns None while
        # not on. It is derived from **the measured noise floor** instead:
        # n95 times 3, or +9.5 dB, which is at least the same order of
        # magnitude.
        g = float(n95 * 3.0) * a.gate_boost
        print(f"⚠ calibration failed: singing level {s75:.4f} <= noise floor "
              f"{n95:.4f} (mic not picking anything up?) → using the "
              f"noise-derived gate {g:.4f} (n95x3). Splitting will be "
              f"touchy; rerun calibration or set --gate before the show",
              flush=True)
        if beep is not None:
            beep(220.0, 700, 1)                # failure is one long low tone, clearly different from success
        return g, raw.get("sing")
    g = float(np.sqrt(n95 * s75)) * a.gate_boost   # the midpoint in the log domain, equidistant from both;
    # 2026-08-19, "raise all the level thresholds": --gate-boost, whose default of
    # 1.0 is the original behaviour.
    db = 20 * np.log10(s75 / n95)
    s10 = float(np.percentile(s, 10))          # the quiet windows inside the singing: breaths and gaps between words
    # The gate prints with %.6g and not %.4f (review #16, 2026-08-17):
    # respond_shell reuses the string scraped from this line. On a quiet chain the
    # gate can be of the order 1e-05, which %.4f prints as 0.0000, and a respawned
    # engine given --gate 0.0 makes lv > 0 always true, so the phrase never ends.
    print(f"   noise floor p95 {n95:.4f} (p50 {np.median(q):.4f}) | "
          f"singing p75 {s75:.4f} (p50 {np.median(s):.4f}) | "
          f"singing p10 {s10:.4f} (breaths/gaps) | "
          f"separation {db:.1f} dB"
          f"\n   → gate {g:.6g} "
          f"({20 * np.log10(g / n95):.1f} dB above the noise floor, "
          f"{20 * np.log10(s75 / g):.1f} dB below the singing level"
          f"{'' if a.gate_boost == 1.0 else f'; boost x{a.gate_boost:g}'})",
          flush=True)
    # What really has to be excluded is the breath, not the room noise floor; the
    # hand-tuned 0.02 was set against breaths at about 0.01. A silent measurement
    # cannot see a breath, so err on the high side: a gate that is too high cuts
    # phrases into pieces, and the response still works, while a gate that is too
    # low cannot end a phrase at all and turns a whole song into one (the measured
    # 34.6 s "phrase" of 2026-08-01). The latter is the fatal one.
    if g < s10:
        print(f"   ⚠ gate {g:.4f} is below the quiet windows inside the singing "
              f"{s10:.4f} = breaths clear the threshold and phrases may not "
              f"split. If phrases run long, raise --gate by hand.",
              flush=True)
    if db < 12:
        print("   ⚠ separation <12 dB: breaths and the noise floor crowd the "
              "threshold = splitting will be unreliable. Fix mic distance / "
              "gain / room noise first.", flush=True)
    if beep is not None:
        beep(660.0, 110, 3)                    # three rising tones: calibration complete, singing may begin
    return g, raw.get("sing")


def live(a):
    """Live: listen, end the phrase, hard mute, render, play the response, unmute."""
    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return

    ear, rend = build_ear_and_mouths(a)
    pre = None if a.no_pipeline else Prefetch(ear)

    def udp_tap():
        """Fall back to UDP, which carries the app space key and the wireless
        button under the same protocol; if it will not open, treat the button as
        absent."""
        t = TapListener()
        if t.err:
            print(f"⚠ [tap] could not open the UDP port ({t.err}) → treating this as "
                  f"no physical key (an orphan engine may still hold the port)",
                  flush=True)
            return None
        return t

    tap = None if a.no_tap else SerialTapListener()
    if tap is not None and tap.err:            # nothing plugged in
        print(f"[tap] could not open serial ({tap.err}) → listening on UDP "
              f"instead (app space key)", flush=True)
        tap = udp_tap()
    elif tap is not None and not tap.wait_proto(TAP_WAIT):
        # Opening it does not prove it is the button: the glob may match a
        # printer, another board, or a Pico sitting in its REPL. The old code
        # looked only at whether opening failed, so in those cases UDP never
        # opened and the space key silently stopped working (review A3/A1,
        # 2026-08-12). Without one TAP or HB line, close it and fall back to
        # UDP.
        print(f"⚠ [tap] no TAP/HB from serial {tap.dev} within {TAP_WAIT:g}s"
              " = not the button → closing it, listening on UDP "
              "(app space key)", flush=True)
        tap.close()
        tap = udp_tap()
    cap, lock = [], threading.Lock()
    cap_n = 0
    st = {"muted": False, "play": None, "ppos": 0, "tail": 0, "n": 0,
          "iov": 0, "oun": 0, "phr": 0, "wait": [], "fade": False,
          "abort": False}
    sess = np.zeros(int(a.max_min * 60 * SR), dtype=np.float32)
    # --out-map 'D,U,L' routes to channels (F7: one multi-channel device). None is
    # the old behaviour, and all the routing code below hangs off omap, so the
    # default path is unchanged line for line.
    omap = None
    if a.out_map:
        omap = [int(x) for x in str(a.out_map).split(",")]
        if len(omap) not in (3, 4) or min(omap) < 0:
            raise SystemExit(f"--out-map needs 3 or 4 non-negative channels "
                             f"(dry,upper,lower[,sop]; without the fourth, sop shares "
                             f"upper's channel), got {a.out_map!r}")
    n_out = 2 if omap is None else max(max(omap) + 1, 2)

    def cb(indata, outdata, frames, t, status):
        if status:                       # a PortAudio flag is direct evidence of pulsing in the input chain
            if status.input_overflow:
                st["iov"] += 1
            if status.output_underflow:
                st["oun"] += 1
        outdata[:] = 0
        with lock:
            muted = st["muted"]
            if not muted:                # hard mute: this audio never reaches the phrase detector at all
                cap.append(indata[:, 0].astype(np.float64))
            p = st["play"]
            if p is not None:
                k = st["ppos"]
                s = p[k:k + frames]
                if p.ndim == 2:          # routed response: each stem is already on its own channel
                    outdata[:len(s), :s.shape[1]] = s
                elif omap is not None:   # a mono signal such as a beep in routed mode goes to every channel
                    outdata[:len(s), :] = s[:, None]
                else:
                    outdata[:len(s), 0] = s
                st["ppos"] = k + len(s)
                if st["fade"]:               # cancelled by the button: fade out one block before stopping, to avoid a click
                    outdata[:, 0] *= np.linspace(1, 0, frames)
                    if omap is not None:     # routed: the other channels fade with it
                        outdata[:, 1:] *= np.linspace(1, 0, frames)[:, None]
                    st["play"], st["fade"] = None, False
                    st["tail"] = int(0.05 * SR)   # they are about to sing; the tail only holds off the reverb
                elif st["ppos"] >= len(p):   # finished playing: stay muted a little longer for the reverb and speaker decay
                    # max(1, ...): with --mute-tail 0, tail would always be 0, so
                    # the `elif st["tail"] > 0` below could never be true, muted
                    # could never be released, and everything after the first
                    # phrase would deadlock (review S5, 2026-08-12). One sample
                    # means the next callback block releases it.
                    st["play"] = None
                    st["tail"] = max(1, int(a.mute_tail * SR))
            elif st["tail"] > 0:
                st["tail"] = max(0, st["tail"] - frames)
                if st["tail"] == 0:
                    st["muted"] = False
            if omap is None and outdata.shape[1] > 1:
                outdata[:, 1] = outdata[:, 0]
            n = st["n"]                  # session dump: their input is written as zeros while muted
            if n + frames <= len(sess):
                mix = outdata[:, 0] if omap is None else outdata.sum(axis=1)
                sess[n:n + frames] = (indata[:, 0] * (0.0 if muted else 1.0)
                                      + mix)
                st["n"] = n + frames
            else:
                st["sess_full"] = True   # stop writing once full; the main loop warns once

    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, n_out),
                   device=(a.in_name, a.out_name), callback=cb,
                   latency=a.io_latency):
        def drain():
            with lock:
                b, cap[:] = list(cap), []
            return np.concatenate(b) if b else np.zeros(0)

        def beep(hz, ms, n, gap_s=0.10):
            """A prompt tone out of the speakers, along the existing play path
            without opening another output.
            2026-08-02 section I: the one real test with a person failed because
            the prompt was text only. The console was not visible, so nobody sang
            during the second calibration stage. Standing and singing at an
            exhibition, it is equally invisible."""
            t = np.arange(int(ms / 1000.0 * SR)) / SR
            one = 0.25 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
            sig = np.tile(np.concatenate([one, np.zeros(int(gap_s * SR))]), n)
            with lock:
                st["play"], st["ppos"] = sig * a.gain, 0
            time.sleep(len(sig) / SR + a.mute_tail + 0.2)   # wait for it to finish playing, plus the reverb

        if a.gate is not None and a.gate <= 1e-6:
            # Review #16, 2026-08-17: a gate of 0, or one that has fallen to the
            # digital-silence floor, makes every window count as voiced, so a
            # phrase never ends and only the physical button can close one. Such a
            # value can only come from a broken reuse chain, where a quiet
            # calibration printed 0.0000 and it was scraped back. Refuse it and
            # recalibrate on the spot.
            print(f"⚠ --gate {a.gate:g} is at/below the digital-silence floor "
                  f"= phrases would never end → ignoring it, recalibrating",
                  flush=True)
            a.gate = None
        if a.gate is not None:
            g, sing = a.gate, None
        else:
            g, sing = calibrate(drain, a, beep)
        if a.keytrack is not None and sing is not None and len(sing) > SR:
            # The calibration singing is already 5 s at performance level and
            # happens before the first phrase, so seeding the key estimate from it
            # stops the first phrase starting from 0, for the same reason as
            # auto-octave.
            a.keytrack.push(dm.harvest_f0(np.ascontiguousarray(sing)))
            r = a.keytrack.best()
            if r is not None:
                ear.k_shift = r[0]
                print(f"[key] calibration segment seeded --key {r[0]:+d}"
                      f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                      flush=True)
        elif a.keytrack is not None and a.key_seed is not None:
            # Review #16, 2026-08-17: when respawned with --gate, which skips
            # calibration, the shell seeds the key locked by the previous engine
            # through --key-seed; otherwise auto starts from 0 and the first
            # phrase is harmonised in the wrong key. The per-phrase rotation
            # correction still applies afterwards.
            ear.k_shift = a.key_seed
            print(f"[key] seeded --key {a.key_seed:+d} from the previous "
                  f"engine (--key-seed)", flush=True)
        if a.octave is None:              # an explicit --octave always wins
            r = octave_from_calib(sing)
            if r is not None:
                ear.v2t.shift, med = r
                print(f"[auto-octave] locked {r[0]:+d} semitones from the calibration "
                      f"segment (median MIDI {med:.1f}, {len(sing)/SR:.1f}s)",
                      flush=True)
            else:
                # Review #16, 2026-08-17: this line used to say only "too short".
                # With --gate skipping calibration, sing is None and lands here
                # too, so the message must honestly cover that case. The app's
                # switch path supplies --octave, so this only appears on a bare CLI
                # with --gate.
                print("[auto-octave] no usable calibration segment (too "
                      "short, or skipped because --gate was given) → falling "
                      "back to \"first 6 voiced ticks\" (a wrong lock ruins "
                      "the whole set; --octave pins it)", flush=True)
        drain()          # the closing prompt tone entered the microphone too; do not take it for their first phrase
        gate = PhraseGate(g, a.gap, a.min_phrase)
        # With the button connected, they decide where a phrase ends and energy
        # detection retreats to a safety net, with the gap stretched only to stop
        # a missed press from locking things up. Otherwise a 0.35 s gap cuts the
        # phrase at a mid-phrase breath before the button has any chance to wait
        # for them; how long to sing, and whether to pause in the middle, is what
        # really returns to their hands (2026-08-02).
        def sync_gap():
            want = a.gap_tap if (tap is not None and tap.alive) else a.gap
            if want != gate.gap_s:
                gate.set_gap(want)
                print(f"[tap] {'connected' if want == a.gap_tap else 'disconnected'}"
                      f" → phrase gap {want}s"
                      f" ({'phrases end on the button, energy split is only a safety net' if want == a.gap_tap else 'back to pure energy detection'})",
                      flush=True)

        sync_gap()
        t_txt = "off" if tap is None else (
            "connected" if tap.alive else "not connected")
        # Honest display: print the gap actually in force, which is --gap-tap when
        # the button is connected, not --gap.
        how = (f"button ends phrases (auto after {gate.gap_s}s = safety net)"
               if tap is not None and tap.alive
               else f"sing, then {gate.gap_s}s of silence")
        print(f"respond2 live. key {a.key} st, gate {g:.4f} gap {gate.gap_s}s, "
              f"response = dry {a.dry_gain} + angel {a.angel_gain}"
              f" (upper {a.upper_db:+.1f} lower {a.lower_db:+.1f} dB), "
              f"gain {a.gain}"
              f" | key {t_txt} | {how} → muted render → response -- Ctrl-C stops.",
              flush=True)
        t_say = time.perf_counter()
        try:
            while True:
                with lock:
                    muted, blocks, cap[:] = st["muted"], list(cap), []
                if tap is not None and tap.dead:   # the serial reader thread has died
                    print("⚠ [tap] serial read interrupted (cable unplugged?) → "
                          "listening on UDP (app space key)", flush=True)
                    tap.close()
                    tap = udp_tap()
                tapped = tap.take() if tap is not None else False
                if tap is not None and not gate.on:   # do not change the gap mid-phrase
                    sync_gap()
                if muted:
                    if tapped:               # their only channel while muted: cancel the response
                        with lock:
                            hot = st["play"] is not None
                            if hot:
                                st["fade"] = True
                        # play is None means the response has finished and it is
                        # inside the mute_tail; during rendering the main loop does
                        # not poll here at all. So there is nothing to cancel.
                        # The old code latched st["abort"] here, and that flag
                        # survived until **the next phrase** had rendered, so
                        # pressing the button after the parts finished, meaning "my
                        # turn", cost the whole of the next response. All three
                        # reviews of 2026-08-12 caught it. Ignore it.
                        print(f"  (key: {'response aborted' if hot else 'response already finished, ignored'})",
                              flush=True)
                    time.sleep(0.01)
                    continue
                if not blocks and not tapped:
                    if time.perf_counter() - t_say > 2:
                        t_say = time.perf_counter()
                        lv = np.array(gate.lv[-400:]) if gate.lv else np.zeros(1)
                        print(f"listening {cap_n / SR:6.1f}s | window rms p50 "
                              f"{np.median(lv):.3f} p95 {np.percentile(lv, 95):.3f}"
                              f" (gate {g:.4f}) | phrase {st['phr']}"
                              f"{'' if tap is None else ' | key ' + ('ON' if tap.alive else 'off')}"
                              f" | io {st['iov']}/{st['oun']}", flush=True)
                        if st.get("sess_full") and not st.get("sess_full_said"):
                            st["sess_full_said"] = True
                            print(f"⚠ dump buffer full (--max-min {a.max_min:g})"
                                  " = later audio no longer enters the session "
                                  "dump (responses carry on)", flush=True)
                    time.sleep(0.01)
                    continue
                if len(blocks):
                    chunk = np.concatenate(blocks)
                    cap_n += len(chunk)
                    seg = gate.push(chunk)
                else:
                    seg = None
                by_tap = False
                if tapped and seg is None:   # a press ends the phrase now, without waiting for the gap
                    seg, by_tap = gate.force_end(), True
                    print(f"  (key: {'phrase ended' if seg is not None else 'not a full phrase yet, ignored'})",
                          flush=True)
                    if seg is None and pre is not None and pre.fed:
                        pre.abandon()
                if seg is None:
                    if pre is not None:      # they are still singing, so compute the settled part now
                        c = gate.committed()
                        if len(c):
                            pre.feed(c)
                    continue
                with lock:               # the phrase has ended: go completely dead, including during rendering
                    # abort is always cleared here: it should apply to this render
                    # only, and left standing it becomes the latch above that ate a
                    # whole response.
                    st["muted"], st["abort"] = True, False
                print("  captured → rendering…", flush=True)   # the UI knows the hard mute has begun from this line
                t_det = time.perf_counter()
                pre_s = 0.0 if pre is None else pre.fed / SR
                try:
                    resp, dt, parts = make_response(seg, ear, rend, a, pre=pre)
                except Exception as e:   # noqa: BLE001 - one bad phrase must not cost the whole session
                    # The old code caught only KeyboardInterrupt, so any phrase
                    # that raised propagated out and even the session dump was
                    # skipped (review A6/S2/S4, 2026-08-12).
                    print(f"  ⚠ this phrase failed to render ({e!r}) → dropping it, "
                          f"back to listening", flush=True)
                    if pre is not None:
                        pre.abandon()
                    gate.reset()
                    with lock:
                        st["muted"], st["abort"] = False, False
                    continue
                busy = time.perf_counter() - t_det     # model plus polish plus render
                if omap is None:
                    peak = float(np.abs(resp).max()) if len(resp) else 0.0
                    out = np.clip(np.concatenate(
                        [np.zeros(int(a.pause * SR)), resp]) * a.gain, -1, 1)
                else:                    # routed: each stem to its own channel, clipped per channel
                    pz = int(a.pause * SR)
                    out = np.zeros((pz + len(resp), n_out))
                    om = omap + [omap[1]] * (len(parts) - len(omap))
                    for q, ch in zip(parts, om):   # the fourth stem, sop, shares upper by default
                        out[pz:pz + len(q), ch] += q * a.gain
                    peak = (float(np.abs(out).max()) / a.gain) if a.gain else 0.0
                    np.clip(out, -1, 1, out=out)
                st["phr"] += 1
                # The time from them stopping to the response starting must be
                # counted from their last voiced sample: the gap seconds are paid
                # first before the end can be detected, then the model and the
                # render, and only then the pause. Reporting pause plus render
                # alone omits the gap and the model; the probe of 2026-08-02
                # measured the mute window 0.2-0.6 s longer than that.
                # Honest display: a press pays no gap at all, and an energy end
                # pays whatever gate.gap_s is in force, which is --gap-tap when the
                # button is connected, rather than always printing --gap.
                g_paid = 0.0 if by_tap else gate.gap_s
                st["wait"].append(g_paid + busy + a.pause)
                print(f"  phrase {st['phr']} {len(seg) / SR:4.1f}s (pre "
                      f"{pre_s:4.1f}s) → brain+render "
                      f"{busy:4.2f}s (mouth {dt:4.2f}s, RTF "
                      f"{dt / (len(seg) / SR):.2f}) → response {len(resp) / SR:4.1f}s "
                      f"peak {peak * a.gain:.2f} | **wait "
                      f"{g_paid + busy + a.pause:4.2f}s** (gap {g_paid:.2f}"
                      f"{' = key' if by_tap else ''}) | mute window "
                      f"{busy + a.pause + len(resp) / SR + a.mute_tail:4.2f}s",
                      flush=True)
                if peak * a.gain > 1.0:
                    # Clipping must not be swallowed silently by np.clip. The
                    # response's peak is tied directly to their input level, since
                    # the parts are restored to their level, so no single default
                    # gain is right for every microphone; all that can be done is
                    # to measure it and say so. Measured 2026-08-03: the
                    # throat-microphone material at dry 1.1 peaked at a median of
                    # 1.35 and a maximum of 2.07, and the K669B is an order of
                    # magnitude lower.
                    print(f"  ⚠ clipping: peak {peak * a.gain:.2f} > 1.0, this phrase got cut"
                          f" (suggest --gain {0.89 / peak:.2f})", flush=True)
                if a.keytrack is not None:
                    r = a.keytrack.best()
                    if r is not None and r[0] != (ear.k_shift or 0) and r[2] >= a.key_margin:
                        # No indent at the start of the line (2026-08-17):
                        # respond_shell's key pattern is ^\[key\], so an indented
                        # rotation line is invisible to the UI, the KEY tile
                        # freezes at the seed value, and --key-seed on a switch
                        # cannot carry the latest one either.
                        print(f"[key] rotation {ear.k_shift:+d} → {r[0]:+d}"
                              f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                              flush=True)
                        ear.k_shift = r[0]
                gate.reset()             # the input does not exist while muted, so the detector state is cleared too
                with lock:
                    if st["abort"]:      # a press during rendering means they want to sing on, so this phrase is void
                        st["abort"], st["muted"] = False, False
                        print("  (key: response discarded, back to listening)", flush=True)
                    elif len(resp):
                        st["play"], st["ppos"] = out.astype(np.float64), 0
                    else:                # the model called the whole phrase a rest, so there is nothing to play; go back to listening
                        st["muted"] = False
        except KeyboardInterrupt:
            pass
        finally:
            # The dump and the statistics are written **before** the stream is
            # closed. On the morning of 2026-08-04, after Ctrl-C, PortAudio's
            # FinishStoppingStream hung inside CoreAudio once and the dump very
            # nearly went with the process; it was rescued by injecting through
            # lldb. sf.write is safe while the stream is still open: n is
            # snapshotted first and the callback only writes past n.
            # **finally**: exceptions other than Ctrl-C must be written out too, or
            # one uncaught error costs the whole recording (review A6/S2,
            # 2026-08-12).
            n = st["n"]
            if n > SR:
                stamp = time.strftime("%y%m%d_%H%M%S")
                p = HERE / "out" / f"resp2_live_{stamp}.wav"
                p.parent.mkdir(exist_ok=True)
                sf.write(str(p), sess[:n], SR, subtype="PCM_16")
                print(f"\nsession dump: {p} ({n / SR:.1f}s, what was heard from him + "
                      f"what was played, same track; input is 0 through the "
                      f"mute = proof the mute is hard)", flush=True)
            w = np.array(st["wait"]) if st["wait"] else np.zeros(1)
            print(f"total {st['phr']} phrases | wait p50 {np.median(w):.2f}s "
                  f"p95 {np.percentile(w, 95):.2f}s"
                  f" (pipelining {'off' if a.no_pipeline else 'on'}; gap as "
                  f"actually paid)"
                  f" | io overflow/underflow {st['iov']}/{st['oun']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--live", action="store_true",
                    help="live mode: microphone in, speakers out, with the input hard-muted while a response plays")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-map", default=None,
                    help="live channel routing: 'D,U,L' are the 0-based output channels for the dry voice, the upper part and the lower part "
                         "(F7: one multi-channel device; later parts are appended). Omitted, it mixes to stereo, the old behaviour. "
                         "File mode ignores it")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="output level of the response (live)")
    ap.add_argument("--mute-tail", type=float, default=0.25,
                    help="seconds to stay muted after the response has played, absorbing the reverb and the speaker decay so its own tail is not "
                         "mistaken for them starting to sing")
    ap.add_argument("--io-latency", default="high",
                    help="sd.Stream latency: high gives the driver a large buffer, resisting GIL spikes")
    ap.add_argument("--max-min", type=float, default=10.0,
                    help="ceiling on the session dump, in minutes")
    ap.add_argument("--key-margin", type=float, default=0.015,
                    help="--key auto: change the angle only with this much lead over the runner-up, which avoids chattering phrase by phrase")
    ap.add_argument("--key-seed", type=int, default=None,
                    help="initial key shift in semitones for --key auto. When respawned with --gate, which skips calibration, respond_shell seeds the "
                         "key locked by the previous engine; the per-phrase rotation correction still applies afterwards. With neither "
                         "calibration nor a seed it starts from 0, the old behaviour")
    ap.add_argument("--key",
                    help="an integer transposition, or **auto**, re-estimated per phrase from the raw-f0 pitch-class distribution. Required (the lesson of "
                         "2026-08-01: a real song can be neither left on auto nor inherit the bench 0; fix the key from the raw-f0 "
                         "pitch-class distribution first)")
    ap.add_argument("--gap", type=float, default=0.35,
                    help="seconds of silence that end a phrase; their breaths measure 0.3-0.5 s. Used when the physical button is not connected")
    ap.add_argument("--gap-tap", type=float, default=6.0,
                    help="the phrase gap while the button is connected, as a safety net: the button decides where a phrase ends, so a deliberate pause "
                         "between phrases is not cut short, and this only stops a missed press locking things up (6 s, set 2026-08-05, "
                         "covering their deliberate pauses)")
    ap.add_argument("--gate", type=float, default=None,
                    help="the rms threshold for voiced, used to end phrases; singing has a median of about 0.15 and breaths about 0.01. Left out in live "
                         f"mode it calibrates at the start; left out in file mode it uses {RMS_GATE}")
    ap.add_argument("--gate-boost", type=float, default=1.0,
                    help="multiply the calibrated threshold by this; above 1 makes it harder to trigger. The default of 1.0 is the behaviour before "
                         "2026-08-19 and is bit-identical. 2.0 is +6 dB. Erring high cuts phrases into pieces, and the response still "
                         "works; erring low means no phrase ever ends and the whole song becomes one, so erring high is the safer side")
    ap.add_argument("--calib-quiet", type=float, default=3.0,
                    help="first calibration stage: seconds spent measuring the noise floor (live)")
    ap.add_argument("--calib-sing", type=float, default=5.0,
                    help="second calibration stage: seconds spent measuring the singing level (live)")
    ap.add_argument("--no-tap", action="store_true",
                    help="do not receive the physical button (tap_listen over USB serial, the wired exhibition route of 2026-08-04)")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="turn pipelining off (live), so the model and the voicing features are computed from scratch at the end of a phrase. Pipelining "
                         "only computes early the parts that are byte-identical whether computed in segments or in one pass, so the sound "
                         "does not change; this flag is the escape route if something goes wrong")
    ap.add_argument("--min-phrase", type=float, default=0.5)
    ap.add_argument("--pause", type=float, default=0.35,
                    help="the interval between them finishing and the response starting, which gives it breath")
    ap.add_argument("--dry-gain", type=float, default=0.944,
                    help="the proportion of their own phrase in the response. **Set by hand while wearing the instrument late on 2026-08-03**: dry 0.944 "
                         "with the parts at 0.75, the upper part a further +1 dB, so the two parts together lead them by about +0.7 dB. "
                         "The 1.1 chosen by an earlier offline A/B is void: that round was not blind (the label read 'you lead by 3 dB'), "
                         "it used pair06 as material, and it was judged at a desk, which breaks the in-context rule of 2026-07-25")
    ap.add_argument("--f0-mode", choices=["express", "target"], default="target",
                    help="the f0 recipe for the voices. **target is the default**, the blind_v3 line of 2026-07-28: a bare skeleton with fixed vibrato, the "
                         "two parts desynchronised at 5.3 and 4.6 Hz with offset phase. On 2026-08-02 both were judged acceptable while the "
                         "express version was judged strange. express is the expression layer of 2026-07-31, **verified on a solo but it "
                         "pulls the harmony apart across two parts**")
    ap.add_argument("--vib-semi", type=float, default=0.12,
                    help="vibrato depth in target mode, in semitones; 0.12 is plus or minus 12 cents, the original direct_mouth.VIB_SEMI")
    ap.add_argument("--tune-lock", type=float, default=0.0,
                    help="the choir tuning to itself (2026-08-02): the two parts share the slow component of the expression layer, the per-note bias and "
                         "the drift, which is the pitch centre, while the vibrato and the jitter stay independent. 0 is the old behaviour "
                         "and 1 makes the pitch centres identical. Nothing human is lost; only the tuning comes back together")
    ap.add_argument("--expr", type=float, default=1.0,
                    help="strength of the expression layer (expressive_cents, added 2026-07-31). 1.0 is the current setting and 0 is the bare skeleton. "
                         "Measured on 2026-08-02, the interval error between the two parts had an SD of 40.7 cents and exceeded 20 cents "
                         "61.6%% of the time, so the harmony was simply not on the same pitch (the per-note bias has an sd of 22 cents and "
                         "the two voices use different seeds, so each deviates on its own)")
    ap.add_argument("--octave", type=int, default=None,
                    help="pin the auto-octave shift in semitones; left out, it locks automatically from the first six voiced ticks. Set it for a "
                         "single-variable A/B, or on stage when you would rather not gamble on the first phrase being heard an octave out")
    ap.add_argument("--angel-gain", type=float, default=0.75,
                    help="the level of each part, set by hand while wearing the instrument on 2026-08-03; see --dry-gain")
    ap.add_argument("--uv-dry", type=float, default=0.0,
                    help="how much of their raw audio the stems pass through over unvoiced spans. The dry voice is already in the response mix, so passing "
                         "it through again stacks the same breath three times (audit of 2026-08-04: breaths and consonants +7.7 dB), hence "
                         "the default of 0, which silences the stems over unvoiced spans. 1.0 is the old behaviour")
    ap.add_argument("--stab", type=int, default=1,
                    help="the model's shortest note, in ticks of 187.5 ms. Audit of 2026-08-04: the median time from them changing note to the parts "
                         "following was 562 ms, one of the main causes of the parts feeling out of step, and the floor imposed by stab 2 "
                         "alone is 375 ms, so the default drops to 1. 2 is the old behaviour, the one listened to on 2026-08-02")
    ap.add_argument("--aah", action="store_true",
                    help="the parts sing only 'ah' rather than carrying the words (2026-08-11): the material is a whole-phrase carrier looping the voiced "
                         "vowel at the end of their phrase, while the pitch still follows the model's harmony line. Off leaves the original "
                         "path byte-identical")
    ap.add_argument("--tail-s", type=float, default=0.0,
                    help="seconds the parts go on singing on their own after they finish (2026-08-04). 0 is off. The carrier loops the last voiced audio of "
                         "their phrase, since the voice needs content, and the pitch is still generated by the model, so the parts keep "
                         "moving while they are silent")
    ap.add_argument("--upper-db", type=float, default=1.0,
                    help="correction to the upper part (the friend's voice) relative to --angel-gain, in dB (2026-08-03: 'the female voice can take +1')")
    ap.add_argument("--lower-db", type=float, default=1.5,
                    help="correction to the lower part (Bass-1) relative to --angel-gain, in dB (+1.5, set on 2026-08-05 after listening to satb_4part; "
                         "previously 0.0)")
    ap.add_argument("--sop", type=int, default=1,
                    help="the S part: Soprano-3 singing their microphone pitch plus 12 (2026-08-05 section G). 0 leaves the settled two-part recipe "
                         "(Aenh, 2026-08-05) byte-identical")
    ap.add_argument("--sop-db", type=float, default=0.0,
                    help="correction to the S part relative to --angel-gain, in dB")
    ap.add_argument("--reverb", type=float, default=0.20,
                    help="reverb send (0 is off, and the whole path is skipped, so it is byte-identical to having no reverb). Set to 0.20 by ear on "
                         "2026-08-03; the synthesised IR is in reverb.py")
    ap.add_argument("--reverb-s", type=float, default=1.8,
                    help="reverb T60 in seconds. **Listened to on 2026-08-03, 1.8, 2.2, 2.5 and 2.8 all sounded about the same, so the length is not an "
                         "audible control in the answering mode**; the IR is energy-normalised, so a longer tail only spreads the same "
                         "energy thinner. The shortest is therefore taken, because the tail is paid for directly in their mute window. "
                         "What really changes the wetness is --reverb")
    ap.add_argument("--reverb-trim", type=float, default=35.0,
                    help="truncate the tail once it falls this many dB (0 does not truncate). The length of the response is exactly how long they cannot "
                         "sound, so the tail is paid straight out of the interaction")
    ap.add_argument("--reverb-angels-only", dest="reverb_all",
                    action="store_false",
                    help="send only the parts into the space and leave their voice in front. **The default sends everything**, from 2026-08-03: 'there is "
                         "a sense of space, and I go into it too', which glues it into one choir")
    ap.add_argument("--no-polish", dest="polish", action="store_false")
    ap.add_argument("--indep", type=float, default=0.15,
                    help="independence of the parts. 0, 0.15, 0.3 and 1.0 were tried while wearing the instrument on 2026-08-03 without a final decision; "
                         "0.15 is provisional. In the short phrases of the answering mode there are few triggers, because it hangs off "
                         "their note events and the shape of the phrase is the real ceiling; see the worklog for 2026-08-04")
    a = ap.parse_args()
    if a.list_devices:
        return live(a)                  # live() prints the device table at its start and returns
    if a.live == bool(a.file):
        ap.error("choose either --file or --live")
    if a.key is None:
        ap.error("--key is required; fix the key of a real song from the raw-f0 pitch-class distribution first")
    a.ir = (rv.make_ir(a.reverb_s, trim_db=a.reverb_trim)
            if a.reverb > 0 else None)
    a.keytrack = None
    if str(a.key).lower() == "auto":
        a.keytrack, a.key = KeyTracker(), "0"     # start from 0 and begin correcting after the first phrase
    if a.live:
        return live(a)

    x, sr = sf.read(a.file[0], dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    assert sr == SR, (a.file[0], sr)
    if a.gate is None:                  # file mode has nothing live to measure, so it keeps the existing threshold
        a.gate = RMS_GATE
    globals()['RMS_GATE'] = a.gate
    ph = find_phrases(x, a.gap, a.min_phrase)
    print(f"split: {len(ph)} phrases / {len(x)/SR:.1f}s"
          f" ({', '.join(f'{(b-s)/SR:.1f}s' for s, b in ph[:8])}"
          f"{' …' if len(ph) > 8 else ''})", flush=True)

    ear, rend = build_ear_and_mouths(a)

    pieces, t_r = [], []
    for i, (s, e) in enumerate(ph):
        seg = x[s:e]
        resp, dt, _ = make_response(seg, ear, rend, a)
        t_r.append(dt)
        pieces += [seg, np.zeros(int(a.pause * SR)), resp,
                   np.zeros(int(a.pause * SR))]
        print(f"  phrase {i+1} {len(seg)/SR:4.1f}s → render {t_r[-1]:4.2f}s "
              f"(RTF {t_r[-1]/(len(seg)/SR):.2f})", flush=True)
        if a.keytrack is not None:
            r = a.keytrack.best()
            if r is not None and r[0] != (ear.k_shift or 0) and r[2] >= a.key_margin:
                print(f"    [key] rotation {ear.k_shift:+d} → {r[0]:+d}"
                      f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                      flush=True)
                ear.k_shift = r[0]

    out = np.concatenate(pieces)
    sf.write(a.file[1], out / (np.abs(out).max() + 1e-12) * 0.9, SR,
             subtype="PCM_16")
    tr = np.array(t_r)
    print(f"\nrender: p50 {np.median(tr):.2f}s p95 {np.percentile(tr, 95):.2f}s "
          f"| his stop → response start ≈ gap {a.gap} + brain + mouth + "
          f"pause {a.pause}"
          f" (>={a.gap + np.median(tr) + a.pause:.2f}s; brain is not inside the "
          f"render seconds, see the closing line of --live)")
    print(f"wrote {a.file[1]}  ({len(out)/SR:.1f}s, [his phrase][response]x{len(ph)})")


if __name__ == "__main__":
    import signal as _sig
    # A parent process launched from a background shell has SIGINT set to SIG_IGN
    # and that is inherited, so Ctrl-C and Stop do nothing at all (the lesson of
    # 2026-07-31 section H; live_v3.py:529 has the same fix). Restore it
    # explicitly.
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)
    main()
