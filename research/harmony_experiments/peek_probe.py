"""Anticipation probe: can the brain predict Harry's NEXT note before he sings it?

The context is interleaved (sop, alto) causal-LM tokens, so after each alto the
model's next-token logits ARE a forecast of the next soprano token. This script
replays a real take through the production ear (PitchTracker -> VoiceToTokens),
asks for that forecast before feeding each actual token, and reports hit rates.
Measurement only -- no engine, no audio out, live.py untouched.

Run:  /opt/anaconda3/envs/vcclient-dev/bin/python peek_probe.py <take.wav> [...]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, str(Path(__file__).parent))
from live import Brain, VoiceToTokens, REST, HOLD, PITCH_OFFSET, PITCH_LO  # noqa: E402
from pitch import PitchTracker, SR as YIN_SR  # noqa: E402

TICK_SEC = 0.1875  # solo_host's brain pulse


def note_name(tok):
    if tok == REST:
        return "REST"
    if tok == HOLD:
        return "HOLD"
    m = tok - PITCH_OFFSET + PITCH_LO
    return "CDEFGAB"[[0, 2, 4, 5, 7, 9, 11].index(m % 12)] + str(m // 12 - 1)


def probe(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != YIN_SR:
        x = soxr.resample(x, sr, YIN_SR).astype(np.float32)

    torch.manual_seed(0)  # reproducible alto sampling
    brain, v2t, tracker = Brain(), VoiceToTokens(), PitchTracker()
    tick = int(round(TICK_SEC * YIN_SR))

    rows = []  # (actual_tok, top1_tok, hit3, p_top1, pitch_top3)
    for b in range(0, len(x) - tick, tick):
        tracker.push(x[b : b + tick])
        actual = v2t.token(tracker.latest)
        if brain.ctx:  # ctx ends on an alto -> next-token logits forecast the sop
            with torch.no_grad():
                t = torch.tensor([brain.ctx[-512:]], device=brain.device)
                p = torch.softmax(brain.model(t)[0, -1], -1)
            top3 = torch.topk(p, 3).indices.tolist()
            # timing belongs to the body (pin sees his onset in ~10 ms); the
            # brain only owes the PITCH -- mask HOLD/REST, keep pitch ranking
            pp = p.clone()
            pp[REST] = pp[HOLD] = 0.0
            ptop3 = torch.topk(pp, 3).indices.tolist()
            rows.append((actual, top3[0], actual in top3, float(p[top3[0]]), ptop3))
        brain.step(actual)

    def stats(sel, label):
        if not sel:
            print(f"  {label:22s} n=0")
            return
        h1 = sum(a == t for a, t, _, _, _ in sel) / len(sel)
        h3 = sum(h for _, _, h, _, _ in sel) / len(sel)
        print(f"  {label:22s} n={len(sel):4d}  top1 {h1:5.1%}  top3 {h3:5.1%}")

    onset = [r for r in rows if r[0] >= PITCH_OFFSET]
    print(f"\n=== {Path(path).name}: {len(rows)} ticks "
          f"(REST {sum(r[0] == REST for r in rows)}, "
          f"HOLD {sum(r[0] == HOLD for r in rows)}, onset {len(onset)})")
    stats(rows, "all ticks")
    stats([r for r in rows if r[0] == HOLD], "HOLD")
    stats(onset, "onset, raw forecast")

    if onset:
        h1 = sum(a == pt[0] for a, _, _, _, pt in onset) / len(onset)
        h3 = sum(a in pt for a, _, _, _, pt in onset) / len(onset)
        print(f"  {'onset, pitch-masked':22s} n={len(onset):4d}  top1 {h1:5.1%}  "
              f"top3 {h3:5.1%}   <- the money number")
        errs = [abs(a - pt[0]) for a, _, _, _, pt in onset if a != pt[0]]
        if errs:
            print(f"  pitch-masked misses: median |err| {np.median(errs):.0f} semitones; "
              + ", ".join(f"{note_name(pt[0])}->{note_name(a)}"
                          for a, _, _, _, pt in onset if a != pt[0])[:120])

    print("  confidence gate (onset-only, raw): coverage -> top1 precision")
    for th in (0.3, 0.5, 0.7, 0.9):
        g = [r for r in onset if r[3] >= th]
        if g:
            print(f"    p>={th}: {len(g)/len(onset):5.1%} of onsets -> "
                  f"{sum(a == t for a, t, _, _, _ in g)/len(g):5.1%} hit")
        else:
            print(f"    p>={th}: no onsets pass")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        probe(p)
