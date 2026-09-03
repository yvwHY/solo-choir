"""Second-chorus-partner retrain (proposal Task 4). New file; train.py/v1 kept.

vs v1: corpus = chorale + POP909/silver + CPDL (corpus_split.json, whole-song
val split), CTX 512 -> 4096 (a full song), + beat-phase embedding (0..15),
+ source embedding (0 chorale / 1 silver / 2 cpdl -- inference can pick the
angel's accent), padding masked out of the loss, checkpoints/v2/best.pt.

Induction probe (every eval): on val POP909 songs, sop-onset top-1 accuracy
at REPEATED 16-tick windows vs FIRST-PASS positions. repeat >> first-pass
is the "breathes with you by the second chorus" signal appearing.

Run (retraining venv):  venv/bin/python train_v2.py [--steps N] [--smoke]
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
from config import TOKENS_V2 as CORPUS, CHORALE_NPZ  # noqa: E402

CTX = 4096
D_MODEL, N_LAYERS, N_HEADS = 384, 6, 6
BATCH = 2
ACCUM = 3
LR = 3e-4
VAL_EVERY = 200
SEED = 7
VOCAB = 51
REST, HOLD, PITCH_OFFSET = 0, 1, 2
PAD = -100
SRC = {"chorale": 0, "pop909": 1, "cpdl": 2}


class HarmonyTransformerV2(nn.Module):
    def __init__(self, vocab=VOCAB):
        super().__init__()
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.voice = nn.Embedding(2, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        self.phase = nn.Embedding(16, D_MODEL)  # beat position within the bar
        self.src = nn.Embedding(3, D_MODEL)     # corpus accent conditioning
        layer = nn.TransformerEncoderLayer(
            D_MODEL, N_HEADS, dim_feedforward=4 * D_MODEL,
            dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)

    def forward(self, x, phase, src):  # x/phase: (B,T), src: (B,)
        B, T = x.shape
        idx = torch.arange(T, device=x.device)
        h = (self.tok(x) + self.voice(idx % 2) + self.pos(idx)
             + self.phase(phase) + self.src(src)[:, None, :])
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


def load_corpus():
    split = json.loads((CORPUS / "corpus_split.json").read_text())
    blocks = {
        "pop909": np.load(CORPUS / "pop909_v2.npz"),
        "cpdl": np.load(CORPUS / "cpdl_v2.npz"),
        "chorale": np.load(CHORALE_NPZ),
    }

    def get(source, sid):
        if source == "chorale":
            tok = blocks["chorale"][sid]
            phase = (np.arange(len(tok)) % 16)  # 4/4 assumption, worklog §T5
        else:
            tok = blocks[source][f"tok_{sid}"]
            phase = blocks[source][f"phase_{sid}"]
        seq = tok.reshape(-1).astype(np.int64)             # (2T,) interleaved
        ph = np.repeat(phase.astype(np.int64), 2)[: len(seq)]
        return seq, ph, SRC[source]

    train = [get(s, i) for s, i in split["train"]]
    val = [get(s, i) for s, i in split["val"]]
    val_pop = [(s, i) for s, i in split["val"] if s == "pop909"]
    return train, val, val_pop, get


def sample_batch(seqs, batch, rng):
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
            st = rng.integers(0, (len(seq) - CTX - 1) // 2) * 2
            xs[b], ph[b] = seq[st : st + CTX + 1], phase[st : st + CTX + 1]
            ys[b] = xs[b, 1:]
    x, p, y = torch.from_numpy(xs[:, :-1]), torch.from_numpy(ph[:, :-1]), torch.from_numpy(ys)
    return x, p, torch.from_numpy(sr), y


def repeat_mask(mel_toks, w=16):
    """Per-tick bool: does the 16-tick melody window ENDING here occur earlier?"""
    seen, out = set(), np.zeros(len(mel_toks), dtype=bool)
    for i in range(w, len(mel_toks)):
        key = tuple(mel_toks[i - w : i])
        out[i] = key in seen
        seen.add(key)
    return out


@torch.no_grad()
def induction_probe(model, val_pop, get, device, max_songs=16):
    hits = {True: [0, 0], False: [0, 0]}  # repeated? -> [hit, total]
    for s, i in val_pop[:max_songs]:
        seq, ph, src = get(s, i)
        seq, ph = seq[:CTX], ph[:CTX]
        x = torch.from_numpy(seq[None, :]).to(device)
        p = torch.from_numpy(ph[None, :]).to(device)
        sr = torch.tensor([src], device=device)
        pred = model(x, p, sr)[0].argmax(-1).cpu().numpy()  # pred[t] = token t+1
        mel = seq[0::2]
        rep = repeat_mask(mel)
        for t in range(1, len(mel)):
            tgt = mel[t]
            if tgt < PITCH_OFFSET:      # onsets only
                continue
            hit = pred[2 * t - 1] == tgt  # position 2t-1 predicts sop at tick t
            hits[bool(rep[t])][0] += int(hit)
            hits[bool(rep[t])][1] += 1
    f = lambda k: hits[k][0] / max(1, hits[k][1])
    return f(False), f(True), hits[False][1], hits[True][1]


@torch.no_grad()
def evaluate(model, val, device, rng, iters=8):
    model.eval()
    losses = []
    for _ in range(iters):
        x, p, sr, y = sample_batch(val, BATCH, rng)
        logits = model(x.to(device), p.to(device), sr.to(device))
        losses.append(F.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1).to(device), ignore_index=PAD).item())
    model.train()
    return float(np.mean(losses))


def main():
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 4000
    smoke = "--smoke" in sys.argv
    if smoke:
        steps = 30
    torch.manual_seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train, val, val_pop, get = load_corpus()
    print(f"device={device} train={len(train)} val={len(val)} (pop in val: {len(val_pop)})",
          flush=True)
    model = HarmonyTransformerV2().to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  CTX={CTX}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(SEED)
    ckdir = HERE / "checkpoints" / "v2"
    ckdir.mkdir(parents=True, exist_ok=True)
    best, t0 = math.inf, time.time()

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):  # micro-batches: CTX 4096 doesn't fit MPS otherwise
            x, p, sr, y = sample_batch(train, BATCH, rng)
            logits = model(x.to(device), p.to(device), sr.to(device))
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1).to(device),
                                   ignore_index=PAD) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % (5 if smoke else VAL_EVERY) == 0 or step == steps:
            vl = evaluate(model, val, device, rng, iters=2 if smoke else 8)
            a_first, a_rep, n1, n2 = induction_probe(model, val_pop, get, device,
                                                     max_songs=4 if smoke else 16)
            tok_s = step * BATCH * ACCUM * CTX / (time.time() - t0)
            flag = ""
            if vl < best and not smoke:
                best = vl
                torch.save({"model": model.state_dict(), "vocab": VOCAB,
                            "config": {"CTX": CTX, "D_MODEL": D_MODEL,
                                       "N_LAYERS": N_LAYERS, "N_HEADS": N_HEADS,
                                       "phase": 16, "src": 3}},
                           ckdir / "best.pt")
                flag = "  <- saved"
            print(f"step {step:5d}  train {loss.item()*ACCUM:.4f}  val {vl:.4f}  "
                  f"induction first {a_first:.1%}({n1}) vs repeat {a_rep:.1%}({n2})  "
                  f"[{tok_s/1000:.1f}k tok/s]{flag}", flush=True)

    print(f"done. best val {best:.4f} ({ckdir/'best.pt'})", flush=True)


if __name__ == "__main__":
    main()
