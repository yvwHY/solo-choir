"""Step 2: small causal Transformer over interleaved (soprano, alto) tokens.

Sequence layout: [s0, a0, s1, a1, ...] -- strictly alternating voices.
At runtime the model sees the soprano token for the current 16th step
(from pitch tracking) and samples the alto token right after it, so the
causal mask matches the live setting exactly.

Trains on MPS if available. ~2M params, minutes not hours.
Saves best-val checkpoint to checkpoints/best.pt.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
CTX = 512          # tokens = 256 grid steps = 16 bars of 4/4
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
BATCH = 64
STEPS = 6000
LR = 3e-4
VAL_EVERY = 250
SEED = 7


class HarmonyTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.voice = nn.Embedding(2, D_MODEL)   # 0 = soprano, 1 = alto
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=0.1, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)

    def forward(self, x):  # x: (B, T) interleaved tokens
        B, T = x.shape
        idx = torch.arange(T, device=x.device)
        h = self.tok(x) + self.voice(idx % 2) + self.pos(idx)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def load_data():
    d = np.load(HERE / "data" / "tokens.npz")
    seqs = [d[k] for k in d.files]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(seqs))
    n_val = max(1, len(seqs) // 10)
    val_idx = set(order[:n_val].tolist())
    flat = [s.reshape(-1) for s in seqs]  # (T,2) -> interleaved (2T,)
    train = [flat[i] for i in range(len(flat)) if i not in val_idx]
    val = [flat[i] for i in range(len(flat)) if i in val_idx]
    (HERE / "data" / "split.json").write_text(json.dumps(
        {"val_chorale_indices": sorted(val_idx)}))
    return train, val


def sample_batch(seqs, batch, rng):
    xs = np.zeros((batch, CTX + 1), dtype=np.int64)
    for b in range(batch):
        s = seqs[rng.integers(len(seqs))]
        if len(s) <= CTX + 1:
            xs[b, : len(s)] = s  # pad with 0 (REST) -- harmless filler
        else:
            start = rng.integers(0, (len(s) - CTX - 1) // 2) * 2  # keep voice parity
            xs[b] = s[start : start + CTX + 1]
    t = torch.from_numpy(xs)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, val, device, rng, iters=20):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = sample_batch(val, BATCH, rng)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def main():
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab = json.loads((HERE / "data" / "vocab.json").read_text())["vocab_size"]
    train, val = load_data()
    print(f"device={device}  vocab={vocab}  train={len(train)} val={len(val)} chorales")

    model = HarmonyTransformer(vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    rng = np.random.default_rng(SEED)
    ckdir = HERE / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    best = math.inf

    for step in range(1, STEPS + 1):
        x, y = sample_batch(train, BATCH, rng)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % VAL_EVERY == 0 or step == STEPS:
            vl = evaluate(model, val, device, rng)
            flag = ""
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "vocab": vocab,
                            "config": {"CTX": CTX, "D_MODEL": D_MODEL,
                                       "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS}},
                           ckdir / "best.pt")
                flag = "  <- saved"
            print(f"step {step:5d}  train {loss.item():.4f}  val {vl:.4f}{flag}", flush=True)

    print(f"done. best val loss {best:.4f}  ({ckdir/'best.pt'})")


if __name__ == "__main__":
    main()
