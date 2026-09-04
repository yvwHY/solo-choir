"""live_v3.py — the v3 two-part live host (the first piece of item #4 in the spec)

mic -> EarV3 (a YIN tracker plus the v3_joint model, sampling upper and lower
       every tick)
    -> two direct-drive DDSP voices (upper through combsub-girl, lower through
       combsub-harry, each with its own output ring; the bounded window, SOLA and
       splice machinery of world_live is inherited whole)
    -> speakers (LAG behind the voice; the two parts mixed at 0.6 each)

The same model class as the offline render_v3 (BrainV3, with the indep and stab
controls), with a sliding window added for live use: 768 tokens by default, which
is 48 s of memory, measured at 38 ms for two forward passes per tick; 1536 gives
81 ms, still inside the 187.5 ms budget. The key assumption is that of
world_live, C major. Locking on the first phrase with keydet came later; it is
the live form of the decision of 2026-07-27 that the parts are the tonal anchor.

File check (no audio device; end-to-end verification):
  .../python live_v3.py --file take.wav out/live_v3_check.wav
"""
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from keydet import KeyDetector  # noqa: E402
from live import HOLD, REST, VoiceToTokens, token_to_midi  # noqa: E402
from pitch import SR, PitchTracker, yin_f0  # noqa: E402
from render_v3 import GIRL_RANGE, HARRY_RANGE, BrainV3, fold  # noqa: E402
from rehearse_ab import ScoreTracker  # noqa: E402
from world_live import (TICK_SAMPS, DDSPMouth, StreamMouth, ExpressiveF0,  # noqa: E402
                        FRAME_HOP, HOP_FRAMES)

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)

KNEE_RMS = 0.019   # the knee where soft singing degrades (worklog 2026-07-30:
                   # distortion 14.3 dB at rms .0195, 18.3 dB at .0055)


def mic_level_stats(x):
    """The instrument for falsifying proposition one (worklog 2026-07-30 section
    E): the distribution of voiced input rms against the knee.
    Voicing is decided as in the live front end, by yin_f0, with rms below 0.005
    counted as silence, so the dangerous band from 0.005 to 0.019 is not eaten by
    the instrument itself. It prints numbers only; the judgement is left to a
    person."""
    v = []
    for s in range(0, len(x) - 2048, 1024):
        seg = x[s:s + 2048]
        if yin_f0(seg.astype(np.float32)) is not None:
            v.append(float(np.sqrt(np.mean(seg * seg))))
    if not v:
        print("mic level (proposition 1): no voiced frames")
        return
    v = np.array(v)
    print(f"mic level (proposition 1): voiced {len(v)} frames | rms p10 "
          f"{np.percentile(v, 10):.4f} "
          f"p25 {np.percentile(v, 25):.4f} p50 {np.percentile(v, 50):.4f} | "
          f"below knee {KNEE_RMS}: {float((v < KNEE_RMS).mean()):.0%}", flush=True)


class EarV3:
    """One tick of audio in, an (upper, lower) pair of absolute notes in
    microphone space out, or None.
    The v3 form of world_live.Ear: one lead token stream, two part outputs, each
    filling its own legato gaps and folded by octave into the measured range of
    its voice.

    key="auto" locks on the first phrase, the live form of the decision of
    2026-07-27 that the parts are the tonal anchor: over the first few seconds of
    singing, keydet hears the major scale and locks k_shift, which then does not
    follow any drift. Before the lock the parts are silent, which is the solo
    opening of the demo arc. key=<int> is a fixed transposition, with 0 the old
    C major behaviour."""

    LOCK_MIN, LOCK_CONF, LOCK_MAX = 130, 0.03, 300  # voiced windows: about 3 s, with a ceiling of about 7 s

    def __init__(self, indep=0.0, stab=1, window=768, legato_gap=1, key="auto",
                 anticipate=False, ant_conf=0.25, rehearse=None, reh_rate=1.0):
        self.brain = BrainV3("joint", indep=indep, stab=stab, window=window)
        self.lead_hist = []          # the lead sounding pitch per tick, recorded for a rehearsal file
        self.reh = None              # rehearsal mode (the tight version v1): the model is bypassed and the parts sing the rehearsed line
        if rehearse:
            self.reh = json.load(open(rehearse))
            self.reh_tracker = ScoreTracker(self.reh["lead"], "v3", rate0=reh_rate)
            print(f"[rehearse] {Path(rehearse).name}: score {len(self.reh['lead'])} "
                  f"ticks, r̂={reh_rate} (v3 forward filter)", flush=True)
        self.v2t = VoiceToTokens()
        self.tracker = PitchTracker()
        self.prev = {"upper": None, "lower": None}
        self.legato = {v: {"gap": 0, "note": None} for v in self.prev}
        self.legato_gap = legato_gap
        self.rng = {"upper": GIRL_RANGE, "lower": HARRY_RANGE}
        self.ant = anticipate
        self.ant_conf = ant_conf
        self.pending = None          # anticipation: the (predicted lead,) already settled on the previous tick
        self.n_ant = self.n_agree = 0
        self.lead_prev = None        # the singer's sounding pitch, for the hit-rate statistics
        if key == "auto":
            self.k_shift, self.kd = None, KeyDetector()
        else:
            self.k_shift, self.kd = int(key), None

    def _keylock(self, chunk):
        for t in range(0, len(chunk) - 2048, 1024):
            self.kd.push(yin_f0(chunk[t:t + 2048].astype(np.float32), SR))
        n, k = self.kd.hist.sum(), self.kd.key()
        if not k:
            return
        if n >= self.LOCK_MIN and k["conf"] >= self.LOCK_CONF:
            self.k_shift = (0 - k["root"]) % 12
            if self.k_shift > 6:
                self.k_shift -= 12
            print(f"[keydet] locked {k['name']} -> shift {self.k_shift:+d} st "
                  f"(conf {k['conf']:.2f})", flush=True)
        elif n >= self.LOCK_MAX:
            # Timed out with too little confidence means a flat histogram, that
            # is, keydet is unreliable on real songs (the dead case of
            # 2026-07-30; on 2026-07-31 a live confidence of 0.01 was hard-locked
            # at +4 semitones and the harmony was in the wrong key all evening).
            # Refuse to lock on rubbish: fall back to shift 0 and warn loudly.
            # The real answer is still to enter an integer key by hand.
            self.k_shift = 0
            print(f"[keydet] ⚠ conf {k['conf']:.2f} < {self.LOCK_CONF} → refusing to "
                  f"lock {k['name']}, falling back to shift 0 (= treat as C "
                  f"major). For a real song stop and set key to an integer "
                  f"(07-30 rule)", flush=True)

    def _to_notes(self, toks):
        """(upper_tok, lower_tok) to a pair of absolute notes in microphone
        space, filling legato gaps and folding by octave."""
        out = {}
        for v, tok in zip(("upper", "lower"), toks):
            m = token_to_midi(tok, self.prev[v])
            self.prev[v] = m
            lg = self.legato[v]
            if m is None and lg["note"] is not None and lg["gap"] < self.legato_gap:
                lg["gap"] += 1
                m = lg["note"]
            else:
                lg["gap"] = 0
                lg["note"] = m
            if m is None:
                out[v] = None
                continue
            out[v] = fold([m - (self.v2t.shift or 0) - self.k_shift],
                          *self.rng[v])[0]
        return out["upper"], out["lower"]

    def tail_tick(self):
        """Let the parts take one more step of their own after the singer has
        finished (2026-08-04).

        Observed while wearing the instrument on 2026-08-03: "they only seem
        independent on a sustained note". Both triggers of `indep`, the singer
        holding and the singer changing note, hang off **the singer's** note
        events, so in the short phrases of the answering mode the control has no
        chance to fire most of the time. The tail of a response is where that
        limit can be sidestepped: during it the singer is already hard-muted and
        silent, so the parts can move on their own.

        Feeding HOLD as the lead treats the singer as sustaining their last note,
        so the model continues to move over the same harmony rather than opening
        a new phrase. It touches neither the tracker nor v2t, so the singer's note
        line and octave lock are entirely unaffected."""
        self.lead_hist.append(self.lead_prev)
        return self._to_notes(self.brain.step(HOLD))

    def tick(self, chunk):
        """Handle one completed tick and return the new note pairs to append to
        notes.
        Reactive returns [this tick]; anticipatory returns [this tick, the first
        time, then the note already settled for the next tick], so the next
        tick's note exists before its audio arrives and the parts land together
        with the singer."""
        self.tracker.push(chunk)
        if self.k_shift is None:            # the first phrase: listen for the key only; the parts stay out
            self._keylock(chunk)
            if self.k_shift is None:
                self.lead_hist.append(None)
                return [(None, None)]
        f_in = self.tracker.latest
        if f_in is not None and self.k_shift:
            f_in = int(f_in) + self.k_shift
        s = self.v2t.token(f_in)
        if self.reh is not None:            # rehearsal mode: alignment replaces the model
            self.lead_prev = token_to_midi(s, self.lead_prev)
            self.lead_hist.append(self.lead_prev)
            self.reh_tracker.observe(self.lead_prev)
            j = int(round(self.reh_tracker.p + self.reh_tracker.rate))
            if 0 <= j < len(self.reh["lead"]):
                return [(self.reh["upper"][j], self.reh["lower"][j])]
            return [(None, None)]           # the score has run out, so the parts stop
        new = []
        if self.ant and self.pending is not None:
            fore = self.pending[0]          # this tick's note was settled last tick; only check the answer
            self.brain.commit_lead(s)
            # A hit means agreement at the pitch layer. While the singer holds,
            # the real token is HOLD and the prediction is forced to emit a pitch
            # token, so a literal comparison could never match; comparing sounding
            # pitch is the honest test.
            if s != REST:
                self.n_ant += 1
                self.n_agree += int(token_to_midi(fore, self.lead_prev)
                                    == token_to_midi(s, self.lead_prev))
            self.lead_prev = token_to_midi(s, self.lead_prev)
        else:
            new.append(self._to_notes(self.brain.step(s)))
            self.lead_prev = token_to_midi(s, self.lead_prev)
        if self.ant:
            nxt = self.brain.step_anticipate(voiced_hint=s != REST,
                                             conf=self.ant_conf)
            if nxt is not None:
                fore, toks = nxt
                self.pending = (fore,)
                new.append(self._to_notes(toks))
            else:
                self.pending = None
        self.lead_hist.append(self.lead_prev)
        return new


def _warmup_pair(fn, label):
    """Run fn() twice and time the first call, which pays for MPS kernel
    compilation, and the warmed call."""
    t0 = time.perf_counter(); fn(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); fn(); warm = time.perf_counter() - t0
    print(f"[warmup] {label} first hop {cold * 1000:.0f}ms -> "
          f"post-warmup {warm * 1000:.0f}ms", flush=True)


def warmup(ear, mouths):
    """During start-up, before the audio stream opens, run the model once and one
    hop of each voice (worklog 2026-07-29 section G, "MPS cold start 239 ms"), to
    move the kernel compilation cost of the first forward pass out of the first
    real hop."""
    import torch
    brain = ear.brain
    window = brain.window or 768
    x = torch.zeros((1, window), dtype=torch.long, device=brain.device)
    p = torch.zeros((1, window), dtype=torch.long, device=brain.device)
    with torch.no_grad():
        _warmup_pair(lambda: brain.model(x, p, brain.src), "brain")
    for v, m in mouths.items():
        n = max(2, round(HOP_FRAMES * FRAME_HOP / m.BLOCK))
        au = m.torch.zeros((1, n * m.BLOCK), device=m.device)
        z = m.torch.zeros((1, n, 1), device=m.device)
        with m.torch.no_grad():
            _warmup_pair(lambda: m.model(m.encoder.encode(au, SR, m.BLOCK)[:, :n],
                                         z, z, spk_id=m.spk), f"{v} mouth")


def run(a):
    out_map = None
    if a.out_map:
        out_map = [int(t) for t in str(a.out_map).split(",")]
        if len(out_map) != 2 or min(out_map) < 0:
            raise SystemExit("--out-map needs two non-negative integers, upper and lower, such as '0,1'")
    if a.dry_ch is not None and out_map is None:
        raise SystemExit("--dry-ch requires --out-map")
    pre = None
    if a.pre_stems:
        if not a.rehearse:
            raise SystemExit("--pre-stems requires --rehearse")
        import soundfile as _sf
        pre = {}
        for v, p in zip(("upper", "lower"), a.pre_stems.split(",")):
            s, psr = _sf.read(p, dtype="float64", always_2d=True)
            assert psr == SR, (p, psr)
            pre[v] = np.ascontiguousarray(s[:, 0])
        print(f"[pre-stems] offline stems, aligned playback (no mouth, ahead "
              f"{a.pre_ahead} ticks): {a.pre_stems}", flush=True)
    cap = int((a.minutes * 60 + 5) * SR)
    mic_ring = np.zeros(cap, dtype=np.float64)
    rings = {"upper": np.zeros(cap, dtype=np.float64),
             "lower": np.zeros(cap, dtype=np.float64)}
    # The expression layer (F21, 2026-07-31): a different seed per voice
    # desynchronises the parts; --expr-gain 0 turns it off.
    ex = (lambda seed: ExpressiveF0(seed, a.expr_gain)) if a.expr_gain > 0 \
        else (lambda seed: None)
    bf = int(round(a.bound_ms / 5.0)) if a.bound_ms else None
    MouthCls = StreamMouth if a.mouth == "stream" else DDSPMouth
    mouths = {} if pre is not None else \
        {"upper": MouthCls(rings["upper"], a.ddsp_repo, a.girl_model,
                           expr=ex(20260731), uv_gate=bool(a.uv_gate),
                           free_run=bool(a.free_run), bound_f=bf),
         "lower": MouthCls(rings["lower"], a.ddsp_repo, a.harry_model,
                           expr=ex(20260732), uv_gate=bool(a.uv_gate),
                           free_run=bool(a.free_run), bound_f=bf)}
    ear = EarV3(indep=a.indep, stab=a.stab, window=a.window, key=a.key,
                anticipate=a.anticipate, ant_conf=a.ant_conf,
                rehearse=a.rehearse, reh_rate=a.reh_rate)
    if a.rehearse:
        rk = ear.reh.get("k_shift")
        if a.key == "auto":
            print("⚠ rehearse mode with --key auto: if the live lock differs from the "
                  "rehearsal file the whole alignment shifts — the demo rule is "
                  "manual --key (07-30 checklist)")
        elif rk is not None and rk != ear.k_shift:
            print(f"⚠ rehearsal file k_shift {rk:+d} != this run {ear.k_shift:+d}, the "
              f"whole alignment will shift")
    warmup(ear, mouths)
    notes = {"upper": [], "lower": []}
    hop_samps = int(round(a.hop_ms / 1000.0 * SR))
    st = {"in": 0, "ticks": 0, "hop_next": hop_samps,
          "under": 0, "done": False}
    lag = int(a.lag * SR)
    lock = threading.Lock()
    t_tick = []

    # Aligned playback state for pre-stems: one read position into the stem,
    # since both parts share a time base, plus a resync counter.
    ps = {"cur": None, "jumps": 0, "quiet": 0}
    XF = int(0.010 * SR)

    def do_pre(k):
        """Tick k has been captured, with the microphone front at (k+1) * TICK.
        Write the stem at the tracker's score position into live slot k+ahead, so
        the audio is in place ahead of the beat. A continuous score position reads
        on seamlessly; only a deviation over the threshold re-locates, with a
        10 ms crossfade. Two or more ticks of silence write zeros, so the parts
        stop with the singer."""
        tr = ear.reh_tracker
        b0 = (k + a.pre_ahead) * TICK_SAMPS
        if b0 < 0 or b0 + TICK_SAMPS > cap:
            return
        ps["quiet"] = ps["quiet"] + 1 if ear.lead_prev is None else 0
        # the predicted score position at the start of live slot k+ahead, as a
        # float tick, converted to a sample position in the stem
        tgt = (tr.p + (a.pre_ahead - 1) * tr.rate) * TICK_SAMPS
        n_stem = min(len(pre["upper"]), len(pre["lower"]))
        if ps["quiet"] >= 2 or not (0 <= tgt < n_stem - TICK_SAMPS):
            for v in ("upper", "lower"):
                rings[v][b0:b0 + TICK_SAMPS] = 0.0
            ps["cur"] = None
            return
        # Musical placement of the re-locate (v2): once the drift exceeds the
        # threshold it does not jump immediately but waits until the target score
        # position falls on a note change, since a lead note change is the singer
        # re-attacking in the take and hides the edit. A drift of 4 times the
        # threshold is a hard ceiling and jumps even without a note change,
        # because with tempo-mismatched material, such as pair06 at a ratio of
        # 1.21, the drift of reading 1:1 is structural, and the 60 ms soft
        # threshold alone would re-locate every two ticks.
        thr = a.pre_resync_ms * SR / 1000.0
        drift = None if ps["cur"] is None else abs(ps["cur"] - tgt)
        lead = ear.reh["lead"]
        jt = int(tgt // TICK_SAMPS)
        onset = 0 < jt < len(lead) and lead[jt] != lead[jt - 1]
        jump = ps["cur"] is None or (drift > thr and (onset or drift > 4 * thr))
        c0 = int(round(tgt if jump else ps["cur"]))
        if c0 + TICK_SAMPS > n_stem:   # the read cursor has run past the end of the stem, from a deferred re-locate
            jump, c0 = True, int(round(tgt))
        ramp = np.linspace(0.0, 1.0, XF)
        for v in ("upper", "lower"):
            seg = pre[v][c0:c0 + TICK_SAMPS].copy()
            if jump:
                if ps["cur"] is not None and int(round(ps["cur"])) + XF <= n_stem:
                    tail = pre[v][int(round(ps["cur"])):int(round(ps["cur"])) + XF]
                    seg[:XF] = seg[:XF] * ramp + tail * (1.0 - ramp)
                else:
                    seg[:XF] *= ramp          # entering after silence: fade in
            rings[v][b0:b0 + TICK_SAMPS] = seg
        if jump and ps["cur"] is not None:
            ps["jumps"] += 1
        ps["cur"] = c0 + TICK_SAMPS

    def do_ticks(n_in):
        while (st["ticks"] + 1) * TICK_SAMPS <= n_in:
            k = st["ticks"]
            t0 = time.perf_counter()
            new = ear.tick(mic_ring[k * TICK_SAMPS: (k + 1) * TICK_SAMPS])
            t_tick.append(time.perf_counter() - t0)
            for up, lo in new:
                notes["upper"].append(up)
                notes["lower"].append(lo)
            if pre is not None:
                do_pre(k)
            st["ticks"] += 1

    def do_hops(t_end, closing=False):
        if pre is not None:      # stems mode: no voices; do_pre writes the audio into the ring directly
            return
        shared = {}          # the units cache for this round; both voices share the content, see DDSPMouth.hop
        for v in ("upper", "lower"):
            mouths[v].hop(mic_ring, t_end, notes[v] if notes[v] else [None],
                          len(notes[v]), closing=closing, shared=shared)

    # Ticks and hops run on separate threads (the starvation case of 2026-07-31
    # section G, regime B). A single worker used to run all pending ticks and then
    # one hop in order, and the model tick p95 rose to 242-351 ms under desktop
    # load, over the 187.5 ms budget, doubling again with two forward passes for
    # anticipation. The voice could produce a hop in 25 ms but was queued behind
    # the model and starved: measured, 21.9 s starved out of 44, with a margin of
    # -16 s. Split apart, the voice no longer waits for the model.
    # Safety: notes only grows, appended on the tick thread and indexed on the hop
    # thread after snapshotting its length, which is safe under the GIL. The
    # voice's lim already limits itself to n_ticks, so when the model falls behind
    # the voice simply builds a little less and catches up on the next hop, which
    # matches the file-mode semantics. st["hop_next"] is touched only by the hop
    # thread. Concurrent MPS use on the two threads is serialised by the command
    # queue.
    def tick_worker():
        while not st["done"]:
            with lock:
                n_in = st["in"]
            do_ticks(n_in)
            time.sleep(0.005)

    def hop_worker():
        while not st["done"]:
            with lock:
                n_in = st["in"]
            if n_in >= st["hop_next"]:
                t_end = st["hop_next"]
                do_hops(t_end)
                st["hop_next"] = t_end + hop_samps
            else:
                time.sleep(0.005)

    def report():
        tt = np.array(t_tick) * 1000 if t_tick else np.zeros(1)
        for v, m in mouths.items():
            th = np.array(m.t_hop) * 1000 if m.t_hop else np.zeros(1)
            te = np.array(m.t_enc) * 1000 if m.t_enc else np.zeros(1)
            tf = np.array(m.t_fwd) * 1000 if m.t_fwd else np.zeros(1)
            print(f"{v}: hops {len(th)} p50 {np.percentile(th, 50):.0f} "
                  f"p95 {np.percentile(th, 95):.0f} ms "
                  f"(enc p50 {np.percentile(te, 50):.0f} / "
                  f"fwd p50 {np.percentile(tf, 50):.0f})", flush=True)
        print(f"brain tick: p50 {np.percentile(tt, 50):.0f} "
              f"p95 {np.percentile(tt, 95):.0f} ms vs tick {TICK_SAMPS / SR * 1000:.1f}")
        if ear.n_ant:
            print(f"anticipation: {ear.n_agree}/{ear.n_ant} predictions hit "
                  f"({ear.n_agree / ear.n_ant:.0%}), angel lands with him")
        if ear.reh is not None:
            tr = ear.reh_tracker
            print(f"rehearse: score pos {tr.p}/{len(ear.reh['lead'])} "
                  f"r̂={tr.rate:.2f} MAP jumps x{tr.jumps}", flush=True)
        if pre is not None:
            print(f"pre-stems: resync x{ps['jumps']}"
                  f" (ahead {a.pre_ahead} ticks, threshold {a.pre_resync_ms:.0f}ms)",
                  flush=True)

    if a.file:
        import soundfile as _sf                  # the stdlib wave module cannot read float wav; the pairs are float32
        raw, in_sr = _sf.read(a.file[0], dtype="float64", always_2d=True)
        assert in_sr == SR, (a.file[0], in_sr)
        mic = np.ascontiguousarray(raw[:, 0])
        mic_ring[: len(mic)] = mic
        if a.notes_json:      # replay an existing note line, skipping the model, for a single-variable A/B
            d = json.load(open(a.notes_json))
            notes["upper"], notes["lower"] = d["upper"], d["lower"]
            print(f"notes replay: {a.notes_json} (brain off, harmony composition locked)")
        else:
            do_ticks(len(mic))
        for t_end in range(hop_samps, len(mic) + hop_samps, hop_samps):
            do_hops(min(t_end, len(mic)), closing=t_end >= len(mic))
        report()
        angels = 0.6 * rings["upper"][: len(mic)] + 0.6 * rings["lower"][: len(mic)]
        for name, sig in [(a.file[1], angels),
                          (a.file[1].replace(".wav", "_mix.wav"), 0.5 * mic + angels)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
            print("wrote", name)
        if not a.notes_json:      # write the note line out, so it can be replayed unchanged with --notes-json
            nj = a.file[1].replace(".wav", "_notes.json")
            d = dict(notes)
            if not a.anticipate and a.rehearse is None:
                # reactive means notes and lead align tick by tick, so this can
                # serve as a rehearsal file for --rehearse
                d["lead"], d["k_shift"] = ear.lead_hist, ear.k_shift
            json.dump(d, open(nj, "w"))
            print("wrote", nj + (" (includes the lead line = usable as a rehearsal file)"
                                 if "lead" in d else ""))
        mic_level_stats(mic)
        return

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return

    def cb(indata, outdata, frames, t, status):
        if status:                 # a PortAudio flag is direct evidence of pulsing in the input chain
            if status.input_overflow:
                st["cb_iov"] = st.get("cb_iov", 0) + 1
            if status.output_underflow:
                st["cb_oun"] = st.get("cb_oun", 0) + 1
        n = st["in"]
        mic_ring[n: n + frames] = indata[:, 0]
        pos = n + frames - lag
        outdata[:] = 0
        if pos > 0:
            lo = max(0, pos - frames)
            if out_map is None:                      # the old path: the two parts mixed, the same content on both channels
                seg = (0.6 * rings["upper"][lo:pos]
                       + 0.6 * rings["lower"][lo:pos]) * a.gain
                outdata[:, 0] = np.pad(seg, (frames - len(seg), 0))
                if outdata.shape[1] > 1:
                    outdata[:, 1] = outdata[:, 0]
            else:                                    # per-voice routing (F11, physical source separation)
                for v, c in zip(("upper", "lower"), out_map):
                    seg = rings[v][lo:pos] * (0.6 * a.gain)
                    outdata[:, c] += np.pad(seg, (frames - len(seg), 0))
            # starved = a mouth owes audio AT THE PLAYHEAD, so ask the note
            # of the tick pos sits in -- not the newest notes at the mic
            # head, one lag ahead: that gate charged the angel's entry as
            # starved while pos was still in the pre-entry silence where
            # the ring is legitimately zero (07-28 §K), and charged every
            # phrase ending the same way (the "slow climb" after it).
            tk = pos // TICK_SAMPS
            if any(pos > m.frontier and tk < len(notes[v])
                   and notes[v][tk] is not None for v, m in mouths.items()):
                st["under"] += frames
        if a.dry_ch is not None:                     # dry pass-through, live and undelayed
            if a.dry_ch < 0:
                outdata += indata[:, :1]
            else:
                outdata[:, a.dry_ch] += indata[:, 0]
        with lock:
            st["in"] = n + frames

    threading.Thread(target=tick_worker, daemon=True).start()
    threading.Thread(target=hop_worker, daemon=True).start()
    dev = (a.in_name, a.out_name)
    n_out = 2 if out_map is None else 1 + max(
        out_map + ([a.dry_ch] if a.dry_ch is not None and a.dry_ch >= 0 else []))
    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, n_out),
                   device=dev, callback=cb, latency=a.io_latency):
        key_txt = "key auto-lock (listen for the key on the first phrase; the parts enter after)" if a.key == "auto" \
            else f"key shift {a.key} st"
        print(f"live v3. lag {a.lag * 1000:.0f} ms, {key_txt}, "
              f"indep {a.indep} stab {a.stab} -- Ctrl-C stops.", flush=True)
        try:
            while st["in"] < cap - SR * 10:
                time.sleep(2)
                cur = {v: next((x for x in reversed(notes[v]) if x is not None), None)
                       for v in notes}
                if any(m.phrase is not None for m in mouths.values()):
                    margin = (min(m.frontier for m in mouths.values())
                              - (st["in"] - lag)) / SR
                    mtxt = f"{margin * 1000:+5.0f} ms"
                else:
                    mtxt = "  rest"      # the frontier is legitimately frozen, so the margin is meaningless
                bt = np.array(t_tick[-32:]) * 1000 if t_tick else np.zeros(1)
                hp = max(((np.percentile(np.array(m.t_hop[-32:]) * 1000, 95)
                           if m.t_hop else 0.0) for m in mouths.values()),
                         default=0.0)
                print(f"in {st['in'] / SR:6.1f}s | ticks {st['ticks']} "
                      f"| U {cur['upper']} L {cur['lower']} "
                      f"| margin {mtxt} "
                      f"| starved {st['under'] / SR:.1f}s "
                      f"| brain {np.percentile(bt, 95):.0f}ms hop {hp:.0f}ms"
                      f" | io {st.get('cb_iov', 0)}/{st.get('cb_oun', 0)}",
                      flush=True)
        except KeyboardInterrupt:
            pass
    st["done"] = True
    report()
    n = st["in"]
    if n > SR:                    # end of a worn session: write the microphone out and the proposition-one numbers
        stamp = time.strftime("%y%m%d_%H%M%S")
        path = HERE / "out" / f"live_mic_{stamp}.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(mic_ring[:n], -1, 1)
                           * 32767).astype(np.int16).tobytes())
        print(f"mic dump: {path}", flush=True)
        if a.dump_out:            # write the output ring out, as material for diagnosing pulsing and joins
            sig = 0.6 * rings["upper"][:n] + 0.6 * rings["lower"][:n]
            op = HERE / "out" / f"live_out_{stamp}.wav"
            with wave.open(str(op), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((np.clip(sig, -1, 1) * 32767)
                              .astype(np.int16).tobytes())
            print(f"out dump: {op}", flush=True)
        mic_level_stats(mic_ring[:n])


if __name__ == "__main__":
    import signal as _sig
    # A parent process launched from a background shell, such as a lab server
    # under nohup, has SIGINT set to SIG_IGN, and that is inherited, so pressing
    # Stop did nothing at all (reported 2026-07-31; a variant of the lesson in
    # section H). Restore it explicitly.
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--notes-json", default=None,
                    help="file mode: replay an existing note line, the *_notes.json written automatically by a previous file run. "
                         "The model does not run and the chord content is locked, for a single-variable A/B")
    ap.add_argument("--rehearse", default=None,
                    help="rehearsal mode (the tight version v1, spec of 2026-07-29): load a rehearsal file, the *_notes.json with a lead "
                         "written by a reactive file run. ScoreTracker v3 aligns to your lead, the parts sing the rehearsed line and the "
                         "model is bypassed; off the script, the parts are the rehearsal anchor. The key must match the rehearsal file")
    ap.add_argument("--reh-rate", type=float, default=1.0,
                    help="the speed prior r-hat for rehearsal alignment, in score ticks per real tick. A fixed value until live has an external tempo "
                         "estimator; the ablation of 2026-07-30 found fixed better than estimating online")
    ap.add_argument("--pre-stems", default=None,
                    help="pre-rendered rehearsal stems 'upper.wav,lower.wav', produced by prerender_stems.py on the rehearsal take's time base. "
                         "The parts become aligned playback of offline express-quality audio and no voice is loaded, which puts the offline "
                         "ceiling on stage (2026-08-01: the work is optimising the parts, not live_v3). Requires --rehearse")
    ap.add_argument("--pre-ahead", type=int, default=1,
                    help="how far ahead to pre-write, in ticks: once tick k is captured, the stem at score position p-hat is written into live slot k+N, "
                         "so the audio is in place ahead of the beat. 1 is the next slot, and the lag then only has to cover tick alignment jitter, "
                         "about 0.05 s")
    ap.add_argument("--pre-resync-ms", type=float, default=60.0,
                    help="re-locate only when the stem read position deviates from the tracker's score position by more than this, with a 10 ms crossfade; "
                         "below it, reading continues seamlessly")
    ap.add_argument("--indep", type=float, default=0.15)
    ap.add_argument("--stab", type=int, default=2)
    ap.add_argument("--window", type=int, default=768,
                    help="the model sliding window in tokens (768 = 48 s at 38 ms, 1536 = 96 s at 81 ms)")
    ap.add_argument("--key", default="auto",
                    help="auto locks with keydet on the first phrase; an integer is a fixed transposition, with 0 the old C major behaviour")
    ap.add_argument("--anticipate", action="store_true",
                    help="anticipation mode: predict the singer's next note so the parts land together with them (use with --lag 0.45)")
    ap.add_argument("--ant-conf", type=float, default=0.25,
                    help="anticipation confidence threshold: below this probability, do not move early and assume the singer holds (0 = off)")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--lag", type=float, default=0.6,
                    help="seconds the playback lags the voice; it must cover tick 187.5 + hop 150 + model 38-81 + two voices at 25 ms")
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--mouth", choices=["stream", "splice"], default="stream",
                    help="stream is genuine streaming, with the phase carried across, one synthesis per sample and no joins (late on 2026-07-31); "
                         "splice is the old bounded re-render with a SOLA tail")
    ap.add_argument("--bound-ms", type=float, default=None,
                    help="the voice re-render window in ms, 700 by default. The hop cost is proportional to the window length, so a shorter window saves "
                         "computation but a rewind on divergence can only reach inside the window. The decision lag is about 200-400 ms, and below "
                         "400 there is a risk of not reaching far enough")
    ap.add_argument("--hop-ms", type=float, default=150.0,
                    help="the voice render period in ms. The lag floor is about the 50 ms guard plus this plus the render time; 75 aims at a lag of "
                         "0.3-0.35, which doubles MPS occupancy, as the starvation telemetry showed")
    ap.add_argument("--free-run", type=int, default=1,
                    help="decouple the voice from the model (2026-07-31): the voice does not wait for the tick decision but holds its current note, and when "
                         "the model's decision arrives it rewinds and bends in, which lets the lag fall to 0.35-0.45. 0 is the old behaviour, where "
                         "the render front is hard-locked by the note decision")
    ap.add_argument("--io-latency", choices=["high", "low"], default="high",
                    help="sd.Stream latency: high gives the driver a large buffer, resisting the input loss under GIL and load spikes that is suspected of "
                         "causing the pulsing; low is the old behaviour")
    ap.add_argument("--dump-out", action="store_true",
                    help="write the parts' output ring out at the end of a live run, for diagnosing pulsing and joins")
    ap.add_argument("--uv-gate", type=int, default=0,
                    help="pull the parts down over breaths and consonants (the v2c anti-chatter version). 0 by default, that is, off: the A/B late on "
                         "2026-07-31 found the mask chattering to be the main cause of the pulsing, and with it off the sound was clean with no "
                         "complaint about breaths, since the expression layer and the streaming voice had improved its premise. Set it to 1 again if "
                         "sung breaths return when the instrument is worn")
    ap.add_argument("--expr-gain", type=float, default=1.0,
                    help="expression-layer multiplier (F21, the rule-based version of 2026-07-31: bias, drift, irregular vibrato and a scoop, with a different "
                         "seed per part to desynchronise them); 0 returns to the old fixed-vibrato behaviour")
    ap.add_argument("--out-map", default=None,
                    help="per-voice output channels 'upper,lower', for example '0,1' sending upper to channel 0 and lower to channel 1, so the rig's units "
                         "are picked up separately (F11, physical source separation). The same convention as solo_min: a single multi-channel device "
                         "(AI-Micro or an aggregate), and never two streams (F7). The default is the current stereo mix")
    ap.add_argument("--dry-ch", type=int, default=None,
                    help="with --out-map: which channel the dry signal, your microphone passed through, goes to. -1 sends it to all; omitting it sends it to "
                         "none, which is the rig convention of 2026-07-29, where the lights watch the parts only")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--ddsp-repo", default=DDSP)
    ap.add_argument("--girl-model", default=f"{DDSP}/exp/combsub-girl/model_30000.pt")
    # On 2026-07-31 live moved to the voice retrained on 2026-07-30 (40.8 min of
    # data; the blind listening of 2026-07-30 was a tie, and it was settled while
    # wearing the instrument on "let us try the other one"). The old voice is
    # --harry-model {DDSP}/exp/combsub-harry/model_30000.pt. The offline
    # direct_mouth default is unchanged, so historical cells stay reproducible.
    ap.add_argument("--harry-model",
                    default=f"{DDSP}/exp/combsub-harry-260730/model_30000.pt")
    a = ap.parse_args()
    run(a)
