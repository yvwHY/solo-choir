"""rehearse_ab.py — an offline scaffold for deciding between rehearsal-mode
mechanisms (spec of 2026-07-29, the mechanism fork)

It takes two wav files, a rehearsal take and a test take, has each mechanism
predict the singer's next note, and prints a table of hit rates. **This decides
mechanisms only and makes no sound**: it measures on the model side and never
touches the voice or DDSP.

The mechanisms, each in its smallest workable form rather than as a large system:
  baseline  a cold model with no prior, which is the current
            live_v3 --anticipate; the offline reproduction of the 11% baseline
            of section L
  A score   the tick pitch line of the rehearsal take is used as a score, aligned
            by a monotonic pointer, and the prediction is the next note on it
  B1 warm   the rehearsal take's token stream is run into BrainV3's sliding
            window first (768 tokens, 48 s, which cannot hold a whole song; the
            spec already flagged this limit, and a long test take pushes it out)
  B2 bias   after aligning the position, lambda is added to the logit of the
            rehearsed note. **Offline, the position alignment simply uses A's
            tracking result, which is the shape of option C.** B2 on its own
            would need its own locator.

Terms (the pitch layer of the worklog for 2026-07-28 section L, aligned word for
word with the statistics in live_v3.EarV3.tick): every non-REST tick of the
singer is checked against the answer, comparing sounding pitch rather than the
literal token, because while the singer holds, the prediction is forced to emit
a pitch token and a literal comparison could never match, which would be unfair.
A separate onset column counts only the ticks where the singer really changes
note, the terms of the 11-16% figure in G29, pitch on note changes with HOLD and
REST masked.

Run (in the DDSP venv, as live_v3):

Smoke test on the same take (knowingly inflated; it only checks the scaffold):
  ... --rehearse .../260722_harmony_brain/data/take.wav --test <the same file>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from keydet import KeyDetector  # noqa: E402
from live import (HOLD, PITCH_HI, PITCH_LO, PITCH_OFFSET, REST,  # noqa: E402
                  VoiceToTokens, token_to_midi)
from pitch import SR, PitchTracker, yin_f0  # noqa: E402
from render_v3 import BrainV3  # noqa: E402

TICK_SAMPS = int((60.0 / 80.0) * 0.25 * SR)      # 8268 at 44.1 kHz, the same model pulse as world_live
LOCK_MIN, LOCK_CONF, LOCK_MAX = 130, 0.03, 300   # the first-phrase lock thresholds of EarV3


# ------------------------------------------------------- front end (an EarV3 replica)

def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x.astype(np.float64)


def lead_stream(path, key="auto", max_ticks=None, octave=None):
    """A wav to a lead token stream. A word-for-word replica of the EarV3 front
    end: the YIN tracker, then keydet locking k_shift on the first phrase without
    following any drift, then VoiceToTokens. Ticks before the lock do not enter
    the stream, which is the span in EarV3 before the parts come in. Returns
    (tokens, k_shift, octave).

    A non-None octave pins VoiceToTokens' auto-octave directly and skips its own
    warm-up. live.py:69-77 decides the octave from the median of the first six
    notes of that take, so two takes of the same song decide separately and were
    measured 12 semitones apart, which puts A's score and B's prior an octave out
    as a whole. main's --shared-octave uses this to let the test take reuse the
    rehearsal take's octave."""
    x = load(path)
    tracker, v2t = PitchTracker(), VoiceToTokens()
    if octave is not None:
        v2t.shift = int(octave)
    kd = KeyDetector() if key == "auto" else None
    k_shift = None if key == "auto" else int(key)
    toks = []
    for b in range(0, len(x) - TICK_SAMPS + 1, TICK_SAMPS):
        chunk = x[b:b + TICK_SAMPS]
        tracker.push(chunk)
        if k_shift is None:
            for t in range(0, len(chunk) - 2048, 1024):
                kd.push(yin_f0(chunk[t:t + 2048].astype(np.float32), SR))
            n, k = kd.hist.sum(), kd.key()
            if k and (n >= LOCK_MAX or (n >= LOCK_MIN and k["conf"] >= LOCK_CONF)):
                k_shift = (0 - k["root"]) % 12
                if k_shift > 6:
                    k_shift -= 12
                print(f"[keydet] {Path(path).name}: locked {k['name']} -> "
                      f"shift {k_shift:+d} st (conf {k['conf']:.2f})", flush=True)
            if k_shift is None:
                continue
        f_in = tracker.latest
        if f_in is not None and k_shift:
            f_in = int(f_in) + k_shift
        toks.append(v2t.token(f_in))
        if max_ticks and len(toks) >= max_ticks:
            break
    return toks, k_shift, v2t.shift


def sounding(toks):
    """A token stream to the sounding pitch per tick; HOLD continues and REST is
    None."""
    out, prev = [], None
    for t in toks:
        prev = token_to_midi(t, prev)
        out.append(prev)
    return out


def midi_to_tok(m):
    if m is None or not (PITCH_LO <= m <= PITCH_HI):
        return None
    return int(m) - PITCH_LO + PITCH_OFFSET


# --------------------------------------------------- scoring (the terms of section L)

class Hits:
    def __init__(self):
        self.n = self.hit = self.n_on = self.hit_on = 0
        self.prev = None                      # the singer's sounding pitch on the previous tick

    def add(self, pred_midi, actual_tok):
        act = token_to_midi(actual_tok, self.prev)
        if actual_tok != REST:
            ok = int(pred_midi is not None and pred_midi == act)
            self.n += 1
            self.hit += ok
            if actual_tok >= PITCH_OFFSET:    # a tick where the singer really changes note, the G29 terms
                self.n_on += 1
                self.hit_on += ok
        self.prev = act

    def add_tok(self, fore, actual_tok):
        self.add(token_to_midi(fore, self.prev), actual_tok)

    def skip(self, actual_tok):
        """No prediction to check, on the first tick: only prev advances, as in
        the else branch of live_v3."""
        self.prev = token_to_midi(actual_tok, self.prev)

    def row(self):
        p = f"{self.hit}/{self.n}" if self.n else "-"
        q = f"{self.hit_on}/{self.n_on}" if self.n_on else "-"
        return (f"{p:>10} {self.hit / self.n:6.1%}" if self.n else f"{p:>10}       ",
                f"{q:>10} {self.hit_on / self.n_on:6.1%}" if self.n_on else f"{q:>10}       ")


# --------------------------------------------------------------- A: score tracking

class ScoreTracker:
    """The smallest workable alignment, without building a large system: the tick
    pitch line of the rehearsal take is the score, a pointer moves monotonically
    forward, and on each tick the position in the window
    [p+1-BACK, p+1+FWD] that best fits the singer's current sounding pitch is
    chosen, ties going to the position nearest p+1, which is the prior of moving
    at a constant rate. When it loses the place, that is, when no zero-cost
    candidate is in the window, it still advances on minimum cost; v1 has no
    lost-place strategy, which the spec already listed as A's risk.

    v2, added on 2026-07-30 and now the default (`--track v1` returns to the
    original behaviour above), fixes three measured faults, all additively:
      speed prior   The original tie-break was |j-(p+1)|, which prefers advancing
                    exactly one step per tick. But the second pass is generally
                    sung faster than the first: across 12 pairs the ratio of
                    score to test ticks measured 0.99-1.27, mean 1.12, so the
                    score should advance r-hat steps per tick. With strophic
                    material there are also many tied candidates, so it advanced
                    slightly too little each tick and accumulated a permanent lag;
                    the median deviation reached 192 ticks, about 36 s, and 10 of
                    12 pairs were behind more than 70% of the time. It now anchors
                    at p+r-hat and breaks ties on |j-(p+r-hat)|, with r-hat
                    estimated online by an EMA of the actual advance.
      re-entry      When the moving average of the local cost stays over a
                    threshold, the place is declared lost, and the observations of
                    the last BUF ticks are correlated against the whole score to
                    re-enter. Jumping backwards is allowed, since strophic
                    material may return to an earlier pass.
      wider window  BACK from 2 to 8: the original could never come back once it
                    had run ahead.
    """

    BACK, FWD = 2, 6                      # the v1 window, kept at its original value for --track v1
    BACK_V2, FWD_V2 = 8, 10               # the v2 window: running ahead must be recoverable
    R_LO, R_HI, R_A = 0.6, 2.0, 0.05      # bounds on r-hat and the EMA coefficient
    LOST_WIN, LOST_TH, LOST_HOLD = 12, 2.6, 40   # lost place: a mean cost over 12 ticks above 2.6 semitones
    BUF = 24                              # observation buffer for re-entry (about 4.5 s)

    R_WIN = 48                            # r-hat is estimated from the slope within a window, not a per-tick EMA

    # Parameters of v3, an HMM forward filter. On 2026-07-31, 12 pairs over a
    # 27-point grid gave a flat surface (mean12 31.7-34.4), so the gain comes
    # from the mechanism and not from tuning. This set is the highest on hi6
    # (41.6) while the same-take regression still holds (99.5 of 100). At
    # tau >= 1.2, multiple hypotheses settle on the wrong equivalent passage in
    # self-similar material and the held notes match wrongly, and same-take falls
    # to 95.6.
    V3_SIGMA = 1.0    # emission softness in semitones: exp(-cost/sigma)
    V3_TAU = 0.8      # spread of the transition around r-hat, in ticks
    V3_EPS = 1e-3     # a uniform leak per tick, which is probabilistic re-entry, replacing v2's triggered jump
    V3_DMAX = 4       # the largest advance in one tick
    V3_JUMP = 4       # a MAP displacement over this counts as a jump; purely diagnostic, to match the v2 report

    def __init__(self, score, mode="v2", rate0=1.0, adapt=False):
        self.score = score
        self.mode = mode
        self.adapt = adapt
        self.p = -1
        self.last = next((m for m in score if m is not None), None)
        self.rate0 = min(max(rate0, self.R_LO), self.R_HI)
        self.rate = self.rate0
        self.obs, self.hist, self.ptrs = [], [], []
        self.tick = self.last_jump = 0
        self.jumps = 0
        if mode == "v3":
            n = len(score)
            self._s = np.array([np.nan if m is None else float(m)
                                for m in score])
            # Starting prior: the same assumption as v1 and v2, that the test
            # take begins at the head of the material, with some slope margin.
            self.alpha = np.exp(-np.arange(n) / 10.0)
            self.alpha /= self.alpha.sum()
            d = np.arange(self.V3_DMAX + 1)
            w = np.exp(-((d - self.rate) ** 2) / (2 * self.V3_TAU ** 2))
            self._tw = w / w.sum()
            self.p = 0

    @staticmethod
    def _cost(a, b):
        if a is None and b is None:
            return 0
        if a is None or b is None:
            return 10
        return min(abs(a - b), 9)

    def observe(self, m):
        if self.mode == "v1":
            return self._observe_v1(m)
        if self.mode == "v3":
            return self._observe_v3(m)
        return self._observe_v2(m)

    def _observe_v3(self, m):
        """A discrete Bayesian forward filter, the exact form of the multiple
        hypothesis particles of 2026-07-30 section D2. The state is the position
        in the score, with the full posterior updated every tick. The ambiguity of
        repeated strophic passages is held as a multi-modal posterior and resolves
        itself as evidence arrives, replacing v2's triggered re-entry, where a
        single pointer was forced to change its mind (5 of 12 pairs did so three
        times or more).
        The state space is discrete and small, a score of about 300-830 steps, so
        the forward recursion is exact inference and needs no particle
        approximation. It is pure numpy with no randomness, and therefore
        deterministic. The speed prior enters the transition kernel, with d from
        0 to DMAX and the weight concentrated at r-hat, and the uniform leak
        V3_EPS is permanent probabilistic re-entry."""
        n = len(self._s)
        prop = np.zeros(n)
        for d, w in enumerate(self._tw):               # transition: advance d steps
            if d == 0:
                prop += w * self.alpha
            else:
                prop[d:] += w * self.alpha[:-d]
        prop = (1.0 - self.V3_EPS) * prop / max(prop.sum(), 1e-300) \
            + self.V3_EPS / n
        if m is None:                                   # emission: the same cost table
            cost = np.where(np.isnan(self._s), 0.0, 10.0)
        else:
            cost = np.where(np.isnan(self._s), 10.0,
                            np.minimum(np.abs(self._s - m), 9.0))
        self.alpha = prop * np.exp(-cost / self.V3_SIGMA)
        s = self.alpha.sum()
        if s < 1e-300:                                  # protection against total collapse: fall back to the propagated prior
            self.alpha = prop
            s = self.alpha.sum()
        self.alpha /= s
        prev = self.p
        self.p = int(np.argmax(self.alpha))
        if abs(self.p - prev) > self.V3_JUMP:
            self.jumps += 1
        if self.score[self.p] is not None:
            self.last = self.score[self.p]

    def _observe_v1(self, m):
        lo = max(0, self.p + 1 - self.BACK)
        hi = min(len(self.score), self.p + 1 + self.FWD + 1)
        if lo >= hi:
            self.p = len(self.score) - 1
            return
        self.p = min(range(lo, hi),
                     key=lambda j: (self._cost(self.score[j], m),
                                    abs(j - (self.p + 1))))
        if self.score[self.p] is not None:
            self.last = self.score[self.p]

    def _observe_v2(self, m):
        self.tick += 1
        self.obs.append(m)
        if len(self.obs) > self.BUF:
            self.obs.pop(0)
        anchor = self.p + self.rate                    # speed prior: this is where it should land
        c = int(round(anchor))
        lo = max(0, c - self.BACK_V2)
        hi = min(len(self.score), c + self.FWD_V2 + 1)
        if lo >= hi:
            self.p = len(self.score) - 1
            return
        prev = self.p
        self.p = min(range(lo, hi),
                     key=lambda j: (self._cost(self.score[j], m),
                                    abs(j - anchor)))
        if self.score[self.p] is not None:
            self.last = self.score[self.p]
        # r-hat is not adapted online by default. Estimating the speed from the
        # pointer's own advance is self-fulfilling: what it measures is exactly
        # the lag it ought to correct, so the estimate only comes out lower. A
        # per-tick EMA settled at 0.60-0.75, and the windowed slope did the same,
        # against a true value of 0.99-1.27. Measured ablation: with r-hat fixed
        # at the tick ratio, A reached 0.574 of the ceiling, and online adaptation
        # dropped that to 0.537. --rate-adapt turns adaptation back on.
        if self.adapt:
            self.ptrs.append(self.p)
            if len(self.ptrs) > self.R_WIN:
                self.ptrs.pop(0)
            if len(self.ptrs) == self.R_WIN:
                slope = (self.ptrs[-1] - self.ptrs[0]) / (self.R_WIN - 1)
                self.rate = min(max(0.5 * self.rate0 + 0.5 * slope,
                                    self.R_LO), self.R_HI)
        if m is not None and self.score[self.p] is not None:
            self.hist.append(self._cost(self.score[self.p], m))
            if len(self.hist) > self.LOST_WIN:
                self.hist.pop(0)
            if (len(self.hist) == self.LOST_WIN
                    and sum(self.hist) / self.LOST_WIN > self.LOST_TH
                    and self.tick - self.last_jump >= self.LOST_HOLD):
                self._reenter()

    def _reenter(self):
        """Losing the place means relocating against the whole score using the
        observations of the last BUF ticks. Jumping backwards is allowed, since
        strophic material may genuinely return to an earlier pass, which v1's
        monotonic pointer could not do."""
        n = len(self.score)
        if sum(o is not None for o in self.obs) < 6:
            return
        best, bs = None, None
        for s in range(n):
            tot = cnt = 0
            for k, o in enumerate(self.obs):
                if o is None:
                    continue
                j = int(round(s + k * self.rate))
                if j >= n:
                    break
                tot += self._cost(self.score[j], o)
                cnt += 1
            if cnt >= 6:
                v = tot / cnt
                if best is None or v < best:
                    best, bs = v, s
        if bs is None:
            return
        self.p = min(max(int(round(bs + (len(self.obs) - 1) * self.rate)), 0), n - 1)
        if self.score[self.p] is not None:
            self.last = self.score[self.p]
        self.hist.clear()
        self.last_jump = self.tick
        self.jumps += 1

    def predict(self):
        """The next note on the score. Where the score rests, the last real note
        is held, so the parts hold rather than change, because the scoring counts
        only ticks where the singer sounds and returning None would be an
        automatic miss, which is not honest.
        v2 uses p+r-hat, the score position corresponding to the next test tick;
        v1 used a fixed p+1."""
        j = self.p + 1 if self.mode == "v1" else int(round(self.p + self.rate))
        if 0 <= j < len(self.score) and self.score[j] is not None:
            return self.score[j]
        return self.last


def _rate0(score, test_toks, rate0=None):
    """The starting value of r-hat. An offline decision tool knows the length of
    both streams, so using the ratio of score to test ticks as a prior is honest.
    A live version would have to estimate it from the rehearsal take's tempo or
    from its own first few seconds."""
    if rate0 is not None:
        return float(rate0)
    return len(score) / max(len(test_toks), 1)


def run_A(test_toks, score, mode="v2", rate0=None, adapt=False):
    hits, tr, pending = Hits(), ScoreTracker(
        score, mode, _rate0(score, test_toks, rate0), adapt), None
    for s in test_toks:
        if pending is not None:
            hits.add(pending, s)
        else:
            hits.skip(s)
        tr.observe(hits.prev)
        pending = tr.predict()
    hits.tracker = tr                     # lets main report r-hat and the number of re-entries
    return hits


# ------------------------------------------------------ B: the model takes a prior

@torch.no_grad()
def anticipate(brain, voiced_hint, conf, bias_tok=None, lam=0.0):
    """A biased form of BrainV3.step_anticipate. With lam=0 and bias_tok=None it
    is line-for-line equivalent to the original method; render_v3.py is unchanged
    and only the smallest necessary section is replicated here to add lambda."""
    if not brain.ctx:
        return None
    phase = brain.tick % 16
    brain.tick += 1
    start = 0
    if brain.window and len(brain.ctx) > brain.window:
        start = (len(brain.ctx) - brain.window + 2) // 3 * 3
    x = torch.tensor([brain.ctx[start:]], device=brain.device)
    p = torch.tensor([brain.phases[start:]], device=brain.device)
    lg = brain.model(x, p, brain.src)[0, -1].clone()
    if voiced_hint:
        lg[REST] = float("-inf")
        lg[HOLD] = float("-inf")
    if bias_tok is not None and lam:
        lg[bias_tok] = lg[bias_tok] + lam
    fore = int(torch.argmax(lg))
    if conf and float(torch.softmax(lg, -1)[fore]) < conf:
        fore = HOLD
    brain._push(fore, phase)
    return fore, brain._voices_for(fore, phase)


def brain_run(test_toks, brain, ant_conf, tracker=None, lam=0.0):
    """The offline form of the anticipation branch of live_v3.EarV3.tick:
    predict, check against the answer on the next tick, then commit_lead writes
    the real token back into the context. A non-None tracker supplies B2's
    position alignment."""
    hits, pending = Hits(), None
    for s in test_toks:
        if pending is not None:
            hits.add_tok(pending, s)
            brain.commit_lead(s)
        else:
            brain.step(s)
            hits.skip(s)
        bias_tok = None
        if tracker is not None:
            tracker.observe(hits.prev)
            bias_tok = midi_to_tok(tracker.predict())
        nxt = anticipate(brain, s != REST, ant_conf, bias_tok, lam)
        pending = nxt[0] if nxt is not None else None
    return hits


def fresh_brain(a, preheat=None):
    torch.manual_seed(0)                      # reproducible sampling of the parts, so mechanisms are comparable
    b = BrainV3("joint", indep=a.indep, stab=a.stab, window=a.window)
    if preheat:
        for s in preheat:
            b.step(s)
        b.tick = 0    # the test take returns to the same phase grid as the baseline: one variable
    return b


# ---------------------------------------------------------------- main

def main(a):
    same = Path(a.rehearse).resolve() == Path(a.test).resolve()
    reh, _, oct_r = lead_stream(a.rehearse, a.key, a.max_ticks)
    if same:
        test, oct_t = reh, oct_r
    else:
        test, _, oct_t = lead_stream(a.test, a.key, a.max_ticks,
                                     octave=oct_r if a.shared_octave else None)
    if oct_r != oct_t:
        print(f"! auto-octave disagrees: rehearse {oct_r:+d} / test {oct_t:+d} semitones. "
              f"The score and the prior will be {oct_t - oct_r:+d} semitones out as a whole, "
              "so the A and B numbers decide nothing. Add --shared-octave to make the test "
              "take reuse the rehearsal octave.", flush=True)
    score = sounding(reh)
    print(f"rehearse {Path(a.rehearse).name}: {len(reh)} ticks "
          f"({len(reh) * TICK_SAMPS / SR:.1f}s) | "
          f"test {Path(a.test).name}: {len(test)} ticks "
          f"({len(test) * TICK_SAMPS / SR:.1f}s)", flush=True)

    warn = "SAME-TAKE, INFLATED" if same else ""
    rows = []
    if "base" in a.mechs:
        rows.append(("baseline, cold model, no prior", brain_run(test, fresh_brain(a), a.ant_conf),
                     "section L / G29 baseline about 11%" if not same else warn))
    if "A" in a.mechs:
        hA = run_A(test, score, a.track, adapt=a.rate_adapt)
        note = "the same take should score about 100%, otherwise the tracker has a bug" if same else "no model, alignment alone"
        if a.track != "v1":
            note += f"; r-hat={hA.tracker.rate:.2f}, re-entries x{hA.tracker.jumps}"
        rows.append(("A, score tracking", hA, note))
    if "B1" in a.mechs:
        rows.append(("B1, context warmed", brain_run(test, fresh_brain(a, reh), a.ant_conf),
                     f"window {a.window} tok = {a.window / 3 * TICK_SAMPS / SR:.0f}s"
                     + (f"；{warn}" if same else "")))
    if "B2" in a.mechs:
        for lam in a.lams:
            rows.append((f"B2, logit bias lambda={lam:g}",
                         brain_run(test, fresh_brain(a), a.ant_conf,
                                   ScoreTracker(score, a.track,
                                                _rate0(score, test),
                                                a.rate_adapt), lam),
                         "position alignment borrowed from A, the shape of option C" + (f"; {warn}" if same else "")))

    print("\nTerms as in the worklog for 2026-07-28 section L, pitch layer: the predicted\n"
          "note against the test take's actual sounding pitch on the next tick, with the\n"
          "singer's REST ticks excluded.\n")
    print(f"{'mechanism':<30}{'voiced hits':>16}{'onset hits':>16}   note")
    print("-" * 96)
    for name, h, note in rows:
        v, o = h.row()
        print(f"{name:<24}{v:>18}{o:>18}   {note}")
    if same:
        print(f"\n! {warn}: rehearsal and test are the same file, so the B mechanisms are\n"
              "checking their own answers and the numbers decide nothing. Fair material is\n"
              "two takes of the same song.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", required=True, help="rehearsal take wav, the source of the score and the prior")
    ap.add_argument("--test", required=True, help="test take wav, the one checked against the answer")
    ap.add_argument("--mechs", default="base,A,B1,B2", help="mechanisms to run, comma separated")
    ap.add_argument("--lams", default="1,3,6", help="lambda values to sweep for B2")
    ap.add_argument("--window", type=int, default=768, help="the model sliding window in tokens, as in live_v3")
    ap.add_argument("--indep", type=float, default=0.15)
    ap.add_argument("--stab", type=int, default=2)
    ap.add_argument("--ant-conf", type=float, default=0.25, help="anticipation confidence threshold, as in live_v3")
    ap.add_argument("--key", default="auto", help="auto locks on each take's own first phrase; an integer is a fixed transposition")
    ap.add_argument("--track", default="v2", choices=["v1", "v2", "v3"],
                    help="score tracker for A and B2: v2 (default) is the speed prior plus lost-place re-entry plus a wider window; "
                         "v1 is the original; v3 is a Bayesian forward filter, holding the ambiguity of repeated passages as a "
                         "multi-modal posterior")
    ap.add_argument("--rate-adapt", action="store_true",
                    help="adapt r-hat online from the windowed slope. Measured worse than the fixed tick ratio, because the "
                         "pointer's own advance is exactly the lag it should be correcting")
    ap.add_argument("--shared-octave", action="store_true",
                    help="the test take reuses the rehearsal take's auto-octave. live.py:69-77 lets each take decide for itself, "
                         "and two takes of one song can differ by 12 semitones, which makes A and B wrong throughout")
    ap.add_argument("--max-ticks", type=int, default=None, help="run only the first N ticks, for a smoke test")
    a = ap.parse_args()
    a.mechs = set(a.mechs.split(","))
    a.lams = [float(x) for x in a.lams.split(",")]
    main(a)
