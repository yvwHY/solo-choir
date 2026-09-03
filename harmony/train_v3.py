"""Brain v3 two-angel training. Dual-track per the 2026-07-27 spec:

JOINT (default): one model, 3-voice interleave [lead, upper, lower] per tick.
  Causal order gives lower sight of (lead, upper) at the same tick -- the
  chain rule P(u,l|lead) = P(u|lead) * P(l|lead,u), collision-awareness by
  construction. Loss on all three voices (the model also learns the lead's
  world, same as v2 learning both voices).

SERIAL (--serial): same data/architecture, loss masked to LOWER positions
  only -- the model is a dedicated second angel conditioned on (lead, upper);
  at inference upper comes from brain_v2. Measures the serial weakness
  honestly (trained on real uppers, fed generated ones).

vs train_v2: corpus = tokens_v3 triples (chorale+cpdl, whole-song split),
voice embedding 2 -> 3, no induction probe (no POP909 in v3 corpus).
Checkpoints: checkpoints/v3_joint/best.pt | checkpoints/v3_serial/best.pt.

Run (retraining venv):  venv/bin/python train_v3.py [--serial] [--steps N] [--smoke]
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TOKENS_V3 as CORPUS  # noqa: E402

CTX = 4096
D_MODEL, N_LAYERS, N_HEADS = 384, 6, 6
BATCH = 2
ACCUM = 3
LR = 3e-4
VAL_EVERY = 200
SEED = 7
VOCAB = 51
PAD = -100
SRC = {"chorale": 0, "cpdl": 2}  # ids match v2 accent conditioning


class HarmonyTransformerV3(nn.Module):
    def __init__(self, vocab=VOCAB):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.voice = nn.Embedding(3, D_MODEL)   # 0 lead / 1 upper / 2 lower
        self.pos = nn.Embedding(CTX, D_MODEL)
        self.phase = nn.Embedding(16, D_MODEL)
        self.src = nn.Embedding(3, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)

    def forward(self, x, phase, src):  # x/phase: (B,T), src: (B,)
        B, T = x.shape
        idx = torch.arange(T, device=x.device)
        h = (self.tok(x) + self.voice(idx % 3) + self.pos(idx)
             + self.phase(phase) + self.src(src)[:, None, :])
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def load_corpus():
    split = json.loads((CORPUS / "corpus_split_v3.json").read_text())
    blocks = {"chorale": np.load(CORPUS / "chorale_v3.npz"),
              "cpdl": np.load(CORPUS / "cpdl_v3.npz")}

    def get(source, wkey):
        tok = blocks[source][f"tok_{wkey}"]        # (T,3) [upper, lead, lower]
        phase = blocks[source][f"phase_{wkey}"]
        seq = tok[:, [1, 0, 2]].reshape(-1).astype(np.int64)  # -> lead,upper,lower
        ph = np.repeat(phase.astype(np.int64), 3)[: len(seq)]
        return seq, ph, SRC[source]

    train = [get(s, w) for s, w in split["train"]]
    val = [get(s, w) for s, w in split["val"]]
    return train, val


def sample_batch(seqs, batch, rng, serial):
    xs = np.zeros((batch, CTX + 1), dtype=np.int64)
    ph = np.zeros((batch, CTX + 1), dtype=np.int64)
    sr = np.zeros(batch, dtype=np.int64)
    ys = np.full((batch, CTX), PAD, dtype=np.int64)
    for b in range(batch):
        seq, phase, src = seqs[rng.integers(len(seqs))]
        sr[b] = src
        if len(seq) <= CTX + 1:
            xs[b, : len(seq)], ph[b, : len(phase)] = seq, phase
            ys[b, : len(seq) - 1] = seq[1:]
        else:
            st = rng.integers(0, (len(seq) - CTX - 1) // 3) * 3
            xs[b], ph[b] = seq[st : st + CTX + 1], phase[st : st + CTX + 1]
            ys[b] = xs[b, 1:]
    if serial:  # loss only where the TARGET is a lower-voice token
        keep = (np.arange(CTX) % 3) == 1  # target of position t is token t+1; t%3==1 -> lower
        ys[:, ~keep] = PAD
    x, p, y = torch.from_numpy(xs[:, :-1]), torch.from_numpy(ph[:, :-1]), torch.from_numpy(ys)
    return x, p, torch.from_numpy(sr), y


@torch.no_grad()
def evaluate(model, val, device, rng, serial, iters=8):
    model.eval()
    losses = []
    for _ in range(iters):
        x, p, sr, y = sample_batch(val, BATCH, rng, serial)
        logits = model(x.to(device), p.to(device), sr.to(device))
        losses.append(F.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1).to(device), ignore_index=PAD).item())
    model.train()
    return float(np.mean(losses))


def main():
    serial = "--serial" in sys.argv
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 4000
    smoke = "--smoke" in sys.argv
    if smoke:
        steps = 30
    mode = "serial" if serial else "joint"
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train, val = load_corpus()
    print(f"mode={mode} device={device} train={len(train)} val={len(val)}", flush=True)
    model = HarmonyTransformerV3().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  CTX={CTX}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(SEED)
    ckdir = HERE / "checkpoints" / f"v3_{mode}"
    ckdir.mkdir(parents=True, exist_ok=True)
    best, t0 = math.inf, time.time()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            x, p, sr, y = sample_batch(train, BATCH, rng, serial)
            logits = model(x.to(device), p.to(device), sr.to(device))
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1).to(device),
                                   ignore_index=PAD) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % (5 if smoke else VAL_EVERY) == 0 or step == steps:
            vl = evaluate(model, val, device, rng, serial, iters=2 if smoke else 8)
            tok_s = step * BATCH * ACCUM * CTX / (time.time() - t0)
            flag = ""
            if vl < best and not smoke:
                best = vl
                torch.save({"model": model.state_dict(), "vocab": VOCAB, "mode": mode,
                            "config": {"CTX": CTX, "D_MODEL": D_MODEL,
                                       "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS,
                                       "phase": 16, "src": 3, "voices": 3}},
                           ckdir / "best.pt")
                flag = "  <- saved"
            print(f"step {step:5d}  train {loss.item()*ACCUM:.4f}  val {vl:.4f}  "
                  f"[{tok_s/1000:.1f}k tok/s]{flag}", flush=True)

    print(f"done. best val {best:.4f} ({ckdir/'best.pt'})", flush=True)


if __name__ == "__main__":
    main()
