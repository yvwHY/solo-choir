"""melody_lm v0 — 第 2 階地基：Harry 旋律習慣的音程轉移模型（08-11 夜）

從既有錄音（mic dumps 去重＋take17＋260730/0802）抽音符事件 → 音程序列
→ trigram＋backoff。held-out 檔案級切分自驗：top-1/top-3 命中率 vs 三個
baseline（重複上一音／級進 ±2／調內均勻）。模型落 melody_lm_v0.json，
之後接 bank_live 搶拍＝從「通用級進猜測」升級成「他的習慣」。
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
            if cur is not None and run >= 8:      # ≥80ms 算一個音
                if not ev or ev[-1] != cur:
                    ev.append(cur)
            cur, run = q, 1
    if cur is not None and run >= 8 and (not ev or ev[-1] != cur):
        ev.append(cur)
    return [n for n in ev if n and 36 <= n <= 84]


# 語料：mic dumps 依檔案大小去重（同尺寸＝同素材 replay）＋主要錄音
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
print(f"語料 {len(files)} 檔（去重後）")

seqs = []
for f in files:
    ev = notes_from_wav(f)
    if len(ev) >= 12:
        seqs.append((f, ev))
print(f"可用序列 {len(seqs)} 條、音符總數 {sum(len(e) for _, e in seqs)}")

# 檔案級 85/15 切分
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
    """回傳 next-interval 機率排序（trigram→bigram→unigram backoff）。"""
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
print(f"held-out {tot} 次預測：top-1 {hit1/tot:.1%}  top-3 {hit3/tot:.1%}")
print(f"baseline：重複同音 {b_rep/tot:.1%}｜級進(±1,2 當 top-4) {b_step/tot:.1%}"
      f"｜調內均勻 ~14%")
json.dump({"tri": {f"{k[0]},{k[1]}": dict(v) for k, v in tri.items()},
           "bi": {str(k): dict(v) for k, v in bi.items()},
           "uni": dict(uni)},
          open("scratchpad/melody_lm_v0.json", "w"))
print("模型 → scratchpad/melody_lm_v0.json")
