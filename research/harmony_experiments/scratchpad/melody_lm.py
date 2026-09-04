"""melody_lm v0 - groundwork for stage 2: a transition model of the singer's melodic habits

From the existing recordings (microphone dumps, deduplicated, plus two later
sessions) it extracts note events, turns them into interval sequences, and fits a
trigram with backoff. Self-checked on a held-out split at file level: top-1 and
top-3 hit rates against three baselines - repeat the last note, step by plus or minus
2, and uniform within the key. The model is written to melody_lm_v0.json, and later
feeds bank_live's anticipation, upgrading it from a generic guess-a-step to his own
habits.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import parselmouth
import soundfile as sf

SR = 44100


def notes_from_wav(path, max_s=180):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[: int(max_s * sr), 0]
    if len(x) < sr:
        return []
    p = parselmouth.Sound(x, sr).to_pitch(
        time_step=0.01, pitch_floor=65, pitch_ceiling=800)
    f = p.selected_array["frequency"]
    m = np.where(f > 0, 69 + 12 * np.log2(np.maximum(f, 1) / 440.0), np.nan)
    ev, cur, run = [], None, 0
    for v in m:
        q = None if np.isnan(v) else int(round(v))
        if q == cur:
            run += 1
        else:
            if cur is not None and run >= 8:      # at least 80 ms counts as a note
                if not ev or ev[-1] != cur:
                    ev.append(cur)
            cur, run = q, 1
    if cur is not None and run >= 8 and (not ev or ev[-1] != cur):
        ev.append(cur)
    return [n for n in ev if n and 36 <= n <= 84]


# corpus: microphone dumps deduplicated by file size (identical sizes mean the same material replayed) plus the main recordings
seen, files = set(), []
for pat in ["scratchpad/*_mic.wav", "scratchpad/take17.wav",
            "../../../SoloChoirCode/260730_recording/260730.wav",
            "../../../SoloChoirCode/260802_recording/0802_test.wav"]:
    for f in sorted(glob.glob(pat)):
        sz = os.path.getsize(f)
        if sz in seen or sz < 2_000_000:
            continue
        seen.add(sz)
        files.append(f)
print(f"corpus: {len(files)} files after deduplication")

seqs = []
for f in files:
    ev = notes_from_wav(f)
    if len(ev) >= 12:
        seqs.append((f, ev))
print(f"{len(seqs)} usable sequences, {sum(len(e) for _, e in seqs)} notes in total")

# 85/15 split at file level
rng = np.random.RandomState(20260811)
idx = rng.permutation(len(seqs))
ncut = max(2, int(0.15 * len(seqs)))
test_i = set(idx[:ncut].tolist())
CLIP = 12


def to_iv(ev):
    return [int(np.clip(b - a, -CLIP, CLIP)) for a, b in zip(ev, ev[1:])]


tri = defaultdict(lambda: defaultdict(float))
bi = defaultdict(lambda: defaultdict(float))
uni = defaultdict(float)
for i, (_f, ev) in enumerate(seqs):
    if i in test_i:
        continue
    iv = to_iv(ev)
    for j, v in enumerate(iv):
        uni[v] += 1
        if j >= 1:
            bi[iv[j - 1]][v] += 1
        if j >= 2:
            tri[(iv[j - 2], iv[j - 1])][v] += 1

VOC = list(range(-CLIP, CLIP + 1))


def predict(ctx):
    """Return the next interval's probability ranking (trigram to bigram to unigram backoff)."""
    sc = {}
    t = tri.get(tuple(ctx[-2:])) if len(ctx) >= 2 else None
    b = bi.get(ctx[-1]) if len(ctx) >= 1 else None
    for v in VOC:
        sc[v] = ((t.get(v, 0.0) * 100 if t else 0.0)
                 + (b.get(v, 0.0) * 10 if b else 0.0)
                 + uni.get(v, 0.0) * 0.01 + 1e-6)
    return sorted(sc, key=sc.get, reverse=True)


hit1 = hit3 = tot = 0
b_rep = b_step = 0
for i in test_i:
    iv = to_iv(seqs[i][1])
    for j in range(2, len(iv)):
        pred = predict(iv[:j])
        tot += 1
        hit1 += pred[0] == iv[j]
        hit3 += iv[j] in pred[:3]
        b_rep += iv[j] == 0
        b_step += iv[j] in (-2, -1, 1, 2)
print(f"held-out, {tot} predictions: top-1 {hit1/tot:.1%}  top-3 {hit3/tot:.1%}")
print(f"baselines: repeat the same note {b_rep/tot:.1%} | step (+/-1, 2 as top-4) {b_step/tot:.1%}"
      f" | uniform within the key about 14%")
json.dump({"tri": {f"{k[0]},{k[1]}": dict(v) for k, v in tri.items()},
           "bi": {str(k): dict(v) for k, v in bi.items()},
           "uni": dict(uni)},
          open("scratchpad/melody_lm_v0.json", "w"))
print("model -> scratchpad/melody_lm_v0.json")
