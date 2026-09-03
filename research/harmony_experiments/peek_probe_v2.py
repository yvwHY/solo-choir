"""Anticipation probe for the v2 brain -- same protocol as peek_probe.py so
the numbers land directly against the G29 Bach baseline.

Caveat (honest): free-rubato takes have no annotated beat, so phase is fed
as a wall-clock 16th cycle from tick 0 (the 4/4-at-80bpm assumption baked
into TICK_SEC). Real beat alignment would need a click take -- treat these
numbers as a lower bound for on-grid singing.

Run (vcclient-dev): python peek_probe_v2.py <take.wav> [...] [--src pop909]
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

sys.path.insert(0, str(Path(__file__).parent))
from live import REST, HOLD, PITCH_OFFSET, PITCH_LO, VoiceToTokens  # noqa: E402
from pitch import PitchTracker, SR as YIN_SR  # noqa: E402
from train_v2 import CTX, SRC, HarmonyTransformerV2  # noqa: E402

TICK_SEC = 0.1875


class BrainV2:
    def __init__(self, src="pop909"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(Path(__file__).parent / "checkpoints" / "v2" / "best.pt",
                        map_location=self.device, weights_only=True)
        self.model = HarmonyTransformerV2(ck["vocab"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.ctx, self.phases = [], []
        self.src = torch.tensor([SRC[src]], device=self.device)

    @torch.no_grad()
    def logits_next(self):
        x = torch.tensor([self.ctx[-CTX:]], device=self.device)
        p = torch.tensor([self.phases[-CTX:]], device=self.device)
        return self.model(x, p, self.src)[0, -1]

    @torch.no_grad()
    def step(self, sop_token, tick):
        phase = tick % 16
        self.ctx.append(sop_token)
        self.phases.append(phase)
        logits = self.logits_next() / 0.9
        if sop_token != REST:
            logits[REST] = float("-inf")
        k = torch.topk(logits, 8)
        alto = int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])
        self.ctx.append(alto)
        self.phases.append(phase)
        return alto


def probe(path, src):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != YIN_SR:
        x = soxr.resample(x, sr, YIN_SR).astype(np.float32)
    torch.manual_seed(0)
    brain, v2t, tracker = BrainV2(src), VoiceToTokens(), PitchTracker()
    tick = int(round(TICK_SEC * YIN_SR))

    rows = []
    for n, b in enumerate(range(0, len(x) - tick, tick)):
        tracker.push(x[b : b + tick])
        actual = v2t.token(tracker.latest)
        if brain.ctx:
            p = torch.softmax(brain.logits_next(), -1)
            top3 = torch.topk(p, 3).indices.tolist()
            pp = p.clone()
            pp[REST] = pp[HOLD] = 0.0
            ptop3 = torch.topk(pp, 3).indices.tolist()
            rows.append((actual, top3[0], actual in top3, float(p[top3[0]]), ptop3))
        brain.step(actual, n)

    onset = [r for r in rows if r[0] >= PITCH_OFFSET]
    print(f"\n=== {Path(path).name} (src={src}): {len(rows)} ticks, onset {len(onset)}")

    def line(sel, label):
        if sel:
            h1 = sum(a == t for a, t, _, _, _ in sel) / len(sel)
            h3 = sum(h for _, _, h, _, _ in sel) / len(sel)
            print(f"  {label:22s} top1 {h1:5.1%}  top3 {h3:5.1%}   (G29 Bach: 0.0%)")

    line(onset, "onset, raw forecast")
    if onset:
        h1 = sum(a == pt[0] for a, _, _, _, pt in onset) / len(onset)
        h3 = sum(a in pt for a, _, _, _, pt in onset) / len(onset)
        print(f"  {'onset, pitch-masked':22s} top1 {h1:5.1%}  top3 {h3:5.1%}   "
              f"(G29 Bach: 11-16% / 43-49% ~ chance)")


if __name__ == "__main__":
    src = sys.argv[sys.argv.index("--src") + 1] if "--src" in sys.argv else "pop909"
    for p in [a for a in sys.argv[1:] if not a.startswith("--") and a != src]:
        probe(p, src)
