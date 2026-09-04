"""render_v3.py — the three-part render inference runner (the first half of item
#3 in the brain v3 spec)

The same take and the same direct-drive voice, with only the model changed
across three cells:
  v1      brain_v2 run twice independently, the trio v1 recipe, deaf to each other
  joint   v3_joint, a single model producing (upper, lower) per tick, where lower
          can see the upper of the same tick
  serial  brain_v2 produces upper, then v3_serial produces lower conditioned on
          (lead, upper)

Output, in out/: {tag}_upper_notes.json and {tag}_lower_notes.json in
direct_mouth format, plus {tag}_stats.json (clash and line-sticking statistics,
used for attribution and never to decide pass or fail). With --run it also drives
the voices twice, upper through combsub-girl and lower through combsub-harry, and
writes {tag}_stem.wav (the two parts) and {tag}_trio.wav (the take at 0.5 with
each part at 0.6).

The blind listening pack is finally built by blind_pack.py from the three stems,
mixed by the same recipe and randomised.

  --baseline prints the same statistics over the real distribution of the CPDL
  and chorale corpus, which is the reference for end point 2 of the spec.
"""
import argparse, json, subprocess, sys, wave
from pathlib import Path

import numpy as np
import torch

from live import HOLD, PITCH_OFFSET, REST, VoiceToTokens, token_to_midi
from brain_v2 import BrainV2
from pitch import SR, PitchTracker, yin_f0
from keydet import KeyDetector
from train_v3 import CORPUS, SRC, HarmonyTransformerV3

HERE = Path(__file__).parent
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
TEMPERATURE, TOP_K = 0.9, 8          # as brain_v2: the A/B changes the model only, not the sampling recipe
BPM = 80.0                            # tick shared by v2 and v3 (a sixteenth at 80 BPM = 187.5 ms)
GIRL_RANGE = (52, 79)                 # E3-G5, the measured range of the girl unit bank (2026-07-26)
HARRY_RANGE = (43, 70)                # the measured range of the harry unit bank
DISSONANT = {1, 2, 6, 10, 11}         # interval classes counted as a clash, on the same terms as D3 (2026-07-26)
STICKY_RUN = 4                        # sticking = the same note for 4 ticks or more (0.75 s)


class BrainV3:
    """A per-tick sampler over a v3 checkpoint. joint: step(lead) -> (upper,
    lower); serial: step_serial(lead, upper) -> lower. The whole context is fed,
    since a take is much shorter than CTX.

    indep is the independence control, a personality layer at inference time,
    from the feedback of 2026-07-28: "they move together too much, I want them
    slightly apart". At the moment the singer changes note, the part suspends
    with probability indep, boosting HOLD so it resolves a beat or two later;
    while the singer holds a note, it takes a passing note with probability
    indep, suppressing HOLD. Each part rolls its own dice, and the choice of
    pitch is still the model's: this biases rhythmic behaviour only and never
    writes the harmony."""

    INDEP_BIAS = 3.0

    def __init__(self, mode, accent="cpdl", indep=0.0, stab=1, window=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(HERE / "checkpoints" / f"v3_{mode}" / "best.pt",
                        map_location=self.device, weights_only=True)
        self.model = HarmonyTransformerV3(ck["vocab"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.src = torch.tensor([SRC[accent]], device=self.device)
        self.ctx, self.phases, self.tick = [], [], 0
        self.indep = indep
        self.stab = stab                 # shortest note in ticks: HOLD is forced after an onset
        self.window = window             # live sliding window in tokens; None uses the whole context
        self.last = {0: REST, 1: REST}   # each part's token from the previous tick
        self.age = {0: 999, 1: 999}      # ticks the current note of each part has lasted

    @torch.no_grad()
    def _sample(self, mask_rest, hold_bias=0.0):
        start = 0
        if self.window and len(self.ctx) > self.window:
            # The cut must be a multiple of 3: the voice embedding uses idx % 3,
            # so an off-by-one cut puts every part out of phase.
            start = (len(self.ctx) - self.window + 2) // 3 * 3
        x = torch.tensor([self.ctx[start:]], device=self.device)
        p = torch.tensor([self.phases[start:]], device=self.device)
        logits = self.model(x, p, self.src)[0, -1] / TEMPERATURE
        if mask_rest:
            logits[REST] = float("-inf")  # the parts sound while the singer sounds, as in the v2 rule
        if hold_bias:
            logits[HOLD] = logits[HOLD] + hold_bias
        k = torch.topk(logits, TOP_K)
        return int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])

    def _push(self, tok, phase):
        self.ctx.append(tok)
        self.phases.append(phase)

    def _indep_bias(self, lead_tok, voice):
        """The dice for a suspension or a passing note. It does not intervene
        while the part itself is at REST, where HOLD continues the rest and stays
        silent."""
        if not self.indep or self.last[voice] == REST:
            return 0.0
        if lead_tok >= PITCH_OFFSET and float(torch.rand(1)) < self.indep:
            return +self.INDEP_BIAS      # the singer changes note and this part stays put: a suspension
        if lead_tok == HOLD and float(torch.rand(1)) < self.indep:
            return -self.INDEP_BIAS      # the singer holds and this part moves: a passing note
        return 0.0

    def _voices_for(self, lead_tok, phase):
        out = []
        for voice in (0, 1):
            # The stability control, from the "notes are too short" feedback of
            # 2026-07-28: HOLD is forced for stab ticks after an onset, and the
            # model only decides again once that commitment has run out. It is
            # not forced while the singer has stopped.
            if (self.stab > 1 and self.age[voice] < self.stab
                    and lead_tok != REST and self.last[voice] != REST):
                tok = HOLD
            else:
                tok = self._sample(lead_tok != REST,
                                   self._indep_bias(lead_tok, voice))
            self._push(tok, phase)
            if tok >= PITCH_OFFSET:
                self.age[voice] = 1
            elif tok == HOLD:
                self.age[voice] += 1
            else:
                self.age[voice] = 999
            self.last[voice] = tok
            out.append(tok)
        return out[0], out[1]

    def step(self, lead_tok):
        phase = self.tick % 16
        self.tick += 1
        self._push(lead_tok, phase)
        return self._voices_for(lead_tok, phase)

    @torch.no_grad()
    def step_anticipate(self, voiced_hint=True, conf=0.0):
        """The anticipation step, the v3 form of the brain_v2 recipe. Restarted
        after the live feedback of 2026-07-28 that the parts "pull on the
        singer": the root cause, a glide on every note, was fixed by gliding only
        on demand, and a wrong prediction is now corrected by an instant jump.
        Rather than waiting for the next tick's lead, the singer's note is
        predicted by argmax (voiced_hint means pitch only, never whether they are
        sounding, with REST and HOLD masked), and both parts are sampled from
        that, so they land together with the singer.
        When the real token arrives, commit_lead() writes it back, keeping the
        parts' own choices."""
        if not self.ctx:
            return None
        phase = self.tick % 16
        self.tick += 1
        start = 0
        if self.window and len(self.ctx) > self.window:
            start = (len(self.ctx) - self.window + 2) // 3 * 3
        x = torch.tensor([self.ctx[start:]], device=self.device)
        p = torch.tensor([self.phases[start:]], device=self.device)
        lg = self.model(x, p, self.src)[0, -1].clone()
        if voiced_hint:
            lg[REST] = float("-inf")
            lg[HOLD] = float("-inf")
        fore = int(torch.argmax(lg))
        # Do not anticipate at low confidence: when unsure, assume the singer
        # holds, so the parts continue their current note. On a melody heard for
        # the first time the hit rate is only about 10-16% (the G29 baseline),
        # and guessing damages the harmony.
        if conf and float(torch.softmax(lg, -1)[fore]) < conf:
            fore = HOLD
        self._push(fore, phase)
        return fore, self._voices_for(fore, phase)

    def commit_lead(self, lead_tok):
        """Replace the predicted lead of the previous anticipation tick with the
        real token, keeping the context honest."""
        self.ctx[-3] = lead_tok

    def step_serial(self, lead_tok, upper_tok):
        phase = self.tick % 16
        self.tick += 1
        self._push(lead_tok, phase)
        self._push(upper_tok, phase)
        lo = self._sample(lead_tok != REST)
        self._push(lo, phase)
        return lo


def fold(notes, lo, hi):
    out = []
    for n in notes:
        if n is not None:
            while n < lo:
                n += 12
            while n > hi:
                n -= 12
        out.append(n)
    return out


def line_stats(a, b):
    """Clash and line-sticking statistics over two sounding MIDI lines, with
    None for silence.
    Everything is counted in interval classes, mod 12, because doubling at the
    octave is still heard as sticking to one line. After the v1 girl voice was
    shifted +12, same-note could never be detected, so the terms had to change
    for the cells to be comparable."""
    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not both:
        return {"ticks_both": 0}
    unison = [(x - y) % 12 == 0 for x, y in both]
    diss = [abs(x - y) % 12 in DISSONANT for x, y in both]
    runs, r = [], 0
    for u in unison + [False]:          # flush at the end
        if u:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    sticky = sum(x for x in runs if x >= STICKY_RUN)
    return {"ticks_both": len(both),
            "unison": sum(unison) / len(both),
            "dissonant": sum(diss) / len(both),
            "sticky": sticky / len(both),
            "longest_unison_run": max(runs, default=0)}


def corpus_baseline():
    """The real upper and lower distribution of the CPDL and chorale corpus:
    end point 2 of the spec, the reference for learning the real distribution."""
    split = json.loads((CORPUS / "corpus_split_v3.json").read_text())
    blocks = {"chorale": np.load(CORPUS / "chorale_v3.npz"),
              "cpdl": np.load(CORPUS / "cpdl_v3.npz")}
    agg = {"n": 0, "unison": 0.0, "dissonant": 0.0, "sticky": 0.0}
    for source, wkey in split["train"] + split["val"]:
        tok = blocks[source][f"tok_{wkey}"]  # (T,3) [upper, lead, lower]
        lines = []
        for ch in (0, 2):
            prev, line = None, []
            for t in tok[:, ch]:
                m = token_to_midi(int(t), prev)
                prev = m
                line.append(m)
            lines.append(line)
        s = line_stats(*lines)
        n = s.get("ticks_both", 0)
        if not n:
            continue
        agg["n"] += n
        for k in ("unison", "dissonant", "sticky"):
            agg[k] += s[k] * n
    for k in ("unison", "dissonant", "sticky"):
        agg[k] /= max(1, agg["n"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v1", "joint", "serial"])
    ap.add_argument("--take", default=str(HERE / "../../../260722_harmony_brain/data/take.wav"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--indep", type=float, default=0.0,
                    help="independence: the per-event probability of a suspension or passing note (joint only)")
    ap.add_argument("--stab", type=int, default=1,
                    help="stability: the shortest note in ticks (1 is the current behaviour; joint only)")
    ap.add_argument("--f0-mode", choices=["target", "shift", "texture"], default="target",
                    help="f0 source for the voice: target is the synthesised target line, shift transposes the singer's real f0, "
                         "texture is the target skeleton with the real f0 micro-texture printed over it")
    ap.add_argument("--legato-gap", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="also drive both voices and mix the trio")
    ap.add_argument("--baseline", action="store_true", help="print only the corpus distribution statistics")
    a = ap.parse_args()
    if a.baseline:
        print(json.dumps(corpus_baseline(), indent=1))
        return
    if not a.mode:
        ap.error("--mode required (or --baseline)")
    tag = a.tag or f"trio_{a.mode}"
    torch.manual_seed(a.seed)

    with wave.open(a.take) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    # Key normalisation, as in world_mouth "auto": the model understands only a
    # C framework, and the take measures as A major.
    kd = KeyDetector()
    for ts in range(0, len(mic) - 2048, 1024):
        f = yin_f0(mic[ts:ts + 2048].astype(np.float32), SR)
        kd.push(float(f) if f else None)
    k = kd.key()
    root = k["root"] if k else 0
    k_shift = (0 - root) % 12
    if k_shift > 6:
        k_shift -= 12
    print(f"key detect: root {k['name'] if k else '?'} -> normalize {k_shift:+d} st")

    # Model pass: the same tick loop as live, with one lead token stream feeding
    # every model.
    if a.mode == "v1":
        brains = [BrainV2(accent="cpdl"), BrainV2(accent="cpdl")]
    elif a.mode == "joint":
        v3 = BrainV3("joint", indep=a.indep, stab=a.stab)
    else:
        v2, v3 = BrainV2(accent="cpdl"), BrainV3("serial")
    v2t, tracker = VoiceToTokens(), PitchTracker()
    step_samps = int((60.0 / BPM) * 0.25 * SR)
    lines = {"upper": [], "lower": []}
    prev = {"upper": None, "lower": None}
    legato = {v: {"gap": 0, "note": None} for v in lines}
    lead = []                            # the singer's sounding line, for the lead-against-parts statistics
    for tick_start in range(0, len(mic) - 1024, step_samps):
        tracker.push(mic[tick_start: tick_start + step_samps])
        f_in = tracker.latest
        lead.append(None if f_in is None else int(f_in))
        if f_in is not None and k_shift:
            f_in = int(f_in) + k_shift
        s = v2t.token(f_in)
        if a.mode == "v1":
            toks = {"upper": brains[0].step(s), "lower": brains[1].step(s)}
        elif a.mode == "joint":
            up, lo = v3.step(s)
            toks = {"upper": up, "lower": lo}
        else:
            up = v2.step(s)
            toks = {"upper": up, "lower": v3.step_serial(s, up)}
        for v in lines:
            m = token_to_midi(toks[v], prev[v])
            prev[v] = m
            lg = legato[v]
            if m is None and lg["note"] is not None and lg["gap"] < a.legato_gap:
                lg["gap"] += 1
                m = lg["note"]
            else:
                lg["gap"] = 0
                lg["note"] = m
            lines[v].append(None if m is None else m - (v2t.shift or 0) - k_shift)

    # Range handling (end point 4 of the spec: fold by octaves, accepting that
    # the enclosure can invert at the moment of a fold).
    # The v1 cell is faithful to the trio v1 recipe: both lines fold into the
    # singer's register, and the girl voice is raised an octave by -k 12.
    ref = float(np.median(tracker.register)) if tracker.register else 55.0
    if a.mode == "v1":
        girl_key = 12
        sound = {v: fold(lines[v], ref - 9, ref + 9) for v in lines}
        notes = dict(sound)
        sound["upper"] = [None if n is None else n + 12 for n in sound["upper"]]
    else:
        girl_key = 0
        notes = sound = {"upper": fold(lines["upper"], *GIRL_RANGE),
                         "lower": fold(lines["lower"], *HARRY_RANGE)}

    for v in lines:
        path = HERE / "out" / f"{tag}_{v}_notes.json"
        json.dump({"bpm": BPM, "sr": SR, "step_samps": step_samps,
                   "notes": notes[v], "heard": lead}, open(path, "w"))
    stats = {"mode": a.mode, "seed": a.seed, "ticks": len(lines["upper"]),
             "upper_vs_lower": line_stats(sound["upper"], sound["lower"]),
             "lead_vs_upper": line_stats(lead, sound["upper"]),
             "lead_vs_lower": line_stats(lead, sound["lower"]),
             "inversion": _inversion(lead, sound),
             "motion": {v: _motion(lead, sound[v]) for v in sound}}
    json.dump(stats, open(HERE / "out" / f"{tag}_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

    if a.run:
        # Vibrato desynchronisation: different rates and phases for the two
        # parts, plus a delayed onset. Mechanically synchronised vibrato is one
        # of the sources of a corrected sound.
        vib = {"upper": ("5.3", "0.0"), "lower": ("4.6", "0.5")}
        fmin = {"upper": "140", "lower": "80"}   # shift mode: floor of the curve-layer fold
        for v, model, key in [("upper", "combsub-girl/model_30000.pt", girl_key),
                              ("lower", "combsub-harry/model_30000.pt", 0)]:
            r = subprocess.run([sys.executable, str(HERE / "direct_mouth.py"),
                                "--notes", str(HERE / "out" / f"{tag}_{v}_notes.json"),
                                "--take", a.take, "--tag", f"{tag}_{v}",
                                "--model", f"{DDSP}/exp/{model}",
                                "--vib-hz", vib[v][0], "--vib-phase", vib[v][1],
                                "--vib-onset-ms", "250",
                                "--f0-mode", a.f0_mode,
                                "--f0-min", fmin[v] if a.f0_mode == "shift" else "65",
                                "--key", str(key), "--run"])
            if r.returncode:
                sys.exit(r.returncode)
        import soundfile as sf
        up, _ = sf.read(HERE / "out" / f"{tag}_upper_angel.wav", dtype="float64")
        lo, _ = sf.read(HERE / "out" / f"{tag}_lower_angel.wav", dtype="float64")
        n = min(len(mic), len(up), len(lo))
        stem = 0.5 * up[:n] + 0.5 * lo[:n]
        stem = stem / (np.max(np.abs(stem)) + 1e-12) * 0.9
        sf.write(HERE / "out" / f"{tag}_stem.wav", stem, SR)
        trio = 0.5 * mic[:n] + 0.6 * up[:n] + 0.6 * lo[:n]
        trio = trio / (np.max(np.abs(trio)) + 1e-12) * 0.9
        sf.write(HERE / "out" / f"{tag}_trio.wav", trio, SR)
        print(f"wrote out/{tag}_stem.wav, out/{tag}_trio.wav")


def _motion(lead, line):
    """Attribution for independence: how often the parts stay put when the
    singer changes note (a suspension), and how often they move while the singer
    holds (a passing note)."""
    sus = mov = lead_chg = lead_hold = 0
    for t in range(1, len(lead)):
        if None in (lead[t], lead[t - 1], line[t], line[t - 1]):
            continue
        if lead[t] != lead[t - 1]:
            lead_chg += 1
            sus += line[t] == line[t - 1]
        else:
            lead_hold += 1
            mov += line[t] != line[t - 1]
    return {"hold_on_lead_change": sus / max(1, lead_chg),
            "move_on_lead_hold": mov / max(1, lead_hold)}


def _inversion(lead, sound):
    """How often the enclosure breaks: the share of ticks where upper falls
    below the lead or lower rises above it."""
    pairs_u = [(l, x) for l, x in zip(lead, sound["upper"]) if l is not None and x is not None]
    pairs_l = [(l, x) for l, x in zip(lead, sound["lower"]) if l is not None and x is not None]
    return {"upper_below_lead": sum(x < l for l, x in pairs_u) / max(1, len(pairs_u)),
            "lower_above_lead": sum(x > l for l, x in pairs_l) / max(1, len(pairs_l))}


if __name__ == "__main__":
    main()
