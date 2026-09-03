"""Harmony brain -> solo_min: the transformer decides, the proven engine sings.

Runs IN solo_min's environment and drives SoloEngine in-process (sm_render
pattern) -- zero changes to the voice-changer repo. The engine supplies the
ear (its f0 telemetry) and the mouth (Beatrice satb2 female voice, ~46 ms
live path); the brain supplies the counterpoint as ABSOLUTE notes pinned via
the harmonizer's existing abs_target_midi (the MIDI-Mode-A mechanism, unused
by solo_min itself): the engine recomputes the compensating shift every frame
against its own held-note estimate, so the angel holds its pitch while Harry
moves. The engine runs --glide 0 here: with an external conductor re-aiming
the target, the anti-scrape glide (F2, 4 st/s) never converged -- every note
became a portamento chasing a moving target (worklog 07-23 §R). Rests ride
the CHEAP on1 switch (keep-warm, no reconfigure click).

Offline proof:  /opt/anaconda3/envs/vcclient-dev/bin/python solo_host.py \
                    --render data/take.wav data/solomin_angel.wav
Live:           .../python solo_host.py --in-name <mic> --out-name <phones>
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = str(HERE.parent)  # lives in <voice-changer>/harmony/
sys.path.insert(0, os.path.join(REPO, "server"))
sys.path.insert(0, str(HERE))
import solo_min  # noqa: E402
from live import Brain, VoiceToTokens, token_to_midi  # noqa: E402

MODEL = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/model/VC/"
         "dist/model_dir/1/model/paraphernalia_data_satb2")
TICK_SEC = 0.1875  # 16th note at 80 bpm, the brain's native pulse


class Conductor:
    """One brain tick: engine f0 -> token -> alto -> absolute-note target."""

    def __init__(self, eng, brain=None):
        self.eng = eng
        self.brain = brain or Brain()
        self.v2t = VoiceToTokens()
        self.prev = None
        self.legato = {"gap": 0, "note": None}
        self.cur_on = None
        self.f0 = 0.0  # updated by telemetry tap
        self.ctl_ms = []

    def tap(self, f0=0.0, in_level=0.0, out_level=0.0, voiced=False):
        self.f0 = float(f0) if voiced else 0.0
        self.in_lv = float(in_level)
        self.out_lv = float(out_level)  # angel's actual output RMS

    def _set(self, d):
        t0 = time.perf_counter()
        self.eng.set_control(d)
        self.ctl_ms.append((time.perf_counter() - t0) * 1000)

    def tick(self):
        midi = int(round(69 + 12 * np.log2(self.f0 / 440.0))) if self.f0 > 0 else None
        s = self.v2t.token(midi)
        heard = self.v2t.prev  # sounding note in brain space (white keys)
        m = token_to_midi(self.brain.step(s), self.prev)
        self.prev = m
        if m is None and self.legato["note"] is not None and self.legato["gap"] < 2:
            self.legato["gap"] += 1
            m = self.legato["note"]
        else:
            self.legato["gap"] = 0
            self.legato["note"] = m
        harm = self.eng.voices[0][1]
        if m is None or heard is None:
            harm.abs_target_midi = None
            if self.cur_on is not False:
                self._set({"on1": False})
                self.cur_on = False
            return None
        # fold the angel's ABSOLUTE note into a band mostly above his voice
        # (female descant separates perceptually; below him reads as mud)
        while m < heard - 3:
            m += 12
        while m > heard + 12:
            m -= 12
        # brain space -> engine space, then pin: the harmonizer recomputes the
        # compensating shift every frame against its own held-note estimate --
        # one pitch reference, nothing to chase (the old external-steps path
        # aimed off a second, instantaneous estimate and re-sent steps on every
        # boundary wobble; see worklog 07-23 §R)
        target = m - (self.v2t.shift or 0)
        harm.abs_target_midi = int(target)
        if self.cur_on is not True:
            self._set({"on1": True})
            self.cur_on = True
        return target


def build_engine(extra, model_dir=None):
    args = solo_min.build_argparser().parse_args([
        "--model", model_dir or MODEL, "--mode", "diatonic", "--key", "C",
        "--steps", "0", "--speaker", "0",  # single female voice = the angel
        "--glide", "0",  # conductor re-aims the target; F2 glide would chase it forever (§R)
    ] + extra)
    eng = solo_min.SoloEngine(args)
    # voice 1 is the angel; park any other spawned voices (keep-warm off)
    eng.set_control({f"on{i}": False for i in range(2, len(eng.voices) + 1)})
    # abs-pinning wants a FRESH input-note estimate: the diatonic defaults
    # (hysteresis 1.0 -> 1.5 st dead zone, ~300 ms median) never re-lock on
    # Harry's 1-semitone steps (E->F, B->C), leaving the angel a whole tone
    # off until he leaps. Chatter is harmless here: held_m flips are exactly
    # cancelled by the recomputed shift, the absolute target never moves.
    harm = eng.voices[0][1]
    harm.note_hysteresis = 0.05
    harm.note_smooth_frames = 4
    print(f"engine up: {len(eng.voices)} voice(s), angel = voice 1")
    return eng


def make_brain(a):
    if a.brain == "v2":
        from brain_v2 import BrainV2
        return BrainV2(accent=a.accent)
    return None  # Conductor defaults to the v1 Brain


def run_render(in_wav, out_wav, extra, brain=None, model_dir=None):
    import soundfile as sf
    import soxr

    eng = build_engine(extra, model_dir)
    cond = Conductor(eng, brain=brain)
    eng.voices[0][0].telemetry_tap = cond.tap
    eng.set_control({"you": 0.0})  # angel stem only; mix made below

    x, sr = sf.read(in_wav, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != solo_min.DEV_SR:
        x = soxr.resample(x, sr, solo_min.DEV_SR).astype(np.float32)
    bs = int(eng.args.blocksize)
    if len(x) % bs:
        x = np.concatenate([x, np.zeros(bs - len(x) % bs, dtype=np.float32)])
    nch = int(getattr(eng, "n_out", 2))
    # same sample clock as wire_live: remainder carry, no drift (int() flooring
    # to whole blocks gave 0.180s ticks vs the brain's 0.1875 grid)
    tick_samps = int(TICK_SEC * 16000)
    hop = bs // 3  # engine samples per device block (16k vs 48k)

    outs, targets, acc = [], [], tick_samps  # start with an immediate tick
    for b in range(0, len(x), bs):
        indata = x[b : b + bs].reshape(-1, 1)
        outdata = np.zeros((bs, nch), dtype=np.float32)
        eng._cb(indata, outdata, bs, None, None)
        outs.append(outdata.copy())
        acc += hop
        if acc >= tick_samps:
            acc -= tick_samps
            targets.append(cond.tick())
    angel = np.concatenate(outs).mean(axis=1)
    mix = 0.5 * x[: len(angel)] + 1.0 * angel
    sf.write(out_wav, angel / max(1e-9, np.abs(angel).max()) * 0.9, solo_min.DEV_SR)
    mix_path = str(out_wav).replace(".wav", "_mix.wav")
    sf.write(mix_path, mix / max(1e-9, np.abs(mix).max()) * 0.9, solo_min.DEV_SR)
    import json
    json.dump({"tick_sec": TICK_SEC, "targets": targets},
              open(str(out_wav) + ".targets.json", "w"))  # for pitch-accuracy checks
    active = [t for t in targets if t is not None]
    print(f"wrote {out_wav} + {mix_path}")
    print(f"ticks {len(targets)}, angel active {len(active)/max(1,len(targets)):.0%}, "
          f"targets MIDI {min(active) if active else '-'}..{max(active) if active else '-'}")
    print(f"set_control: n={len(cond.ctl_ms)}, median {np.median(cond.ctl_ms):.2f}ms, "
          f"max {np.max(cond.ctl_ms):.2f}ms (audio callback budget 10ms)")


def wire_live(eng, cond):
    """Single clock: the engine's telemetry stream (one tap per 10 ms hop)
    drives the brain pulse -- replaces the old wall-clock sleep loop.
    tap runs ON THE AUDIO THREAD: count + enqueue only, never block; brain
    inference (~6 ms MPS) stays on the brain thread. Between ticks the engine
    holds the angel on its absolute note by itself (abs_target_midi is
    re-evaluated every frame), so there is nothing else to schedule."""
    import queue
    import threading

    q = queue.Queue(maxsize=64)
    tick_samps = int(TICK_SEC * 16000)  # brain pulse in engine samples
    hop = int(eng.HOP)                  # 160 @16k = one tap per 10 ms
    st = {"acc": 0}

    def tap(f0=0.0, in_level=0.0, out_level=0.0, voiced=False):
        cond.tap(f0=f0, in_level=in_level, out_level=out_level, voiced=voiced)
        st["acc"] += hop
        if st["acc"] >= tick_samps:
            st["acc"] -= tick_samps  # keep the remainder: no drift (18.75 hops/tick)
            try:
                q.put_nowait("tick")
            except queue.Full:
                pass  # brain behind; drop is safe, next tick re-syncs

    eng.voices[0][0].telemetry_tap = tap

    def brain_loop():
        n = 0
        while True:
            q.get()
            t = cond.tick()
            n += 1
            if n % 8 == 0:
                conv, harm = eng.voices[0]
                print(f"you {cond.f0:5.0f}Hz in {getattr(cond,'in_lv',0):.3f} | "
                      f"angel target {t} (abs={harm.abs_target_midi}) "
                      f"shift={float(conv._cur_shift):+.1f}st on={eng.voice_on[0]} "
                      f"out {getattr(cond,'out_lv',0):.3f}", flush=True)

    return threading.Thread(target=brain_loop, daemon=True)


def run_live(extra, brain=None, model_dir=None):
    eng = build_engine(extra, model_dir)
    cond = Conductor(eng, brain=brain)
    # quiet headset mic: open the output gate fully, lift the choir
    eng.set_control({"you": 0.0, "gate": 0.0, "choir_gain": 2.0})
    wire_live(eng, cond).start()
    eng.start()
    print("live. sing in C major -- Ctrl-C stops.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--brain", choices=["v1", "v2"], default="v1")
    ap.add_argument("--accent", choices=["chorale", "pop909", "cpdl"], default="cpdl")
    ap.add_argument("--model-dir", default=None,
                    help="Beatrice paraphernalia dir (default: satb2)")
    a, extra = ap.parse_known_args()
    if a.render:
        run_render(a.render[0], a.render[1], extra, brain=make_brain(a), model_dir=a.model_dir)
    else:
        dev = []
        if a.in_name:
            dev += ["--in-name", a.in_name]
        if a.out_name:
            dev += ["--out-name", a.out_name]
        run_live(dev + extra, brain=make_brain(a), model_dir=a.model_dir)
