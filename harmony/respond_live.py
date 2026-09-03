"""respond_live — 及時 live 應答式（08-11 Harry「試試你剛剛的及時live的應答式」）

你唱一句 → 句尾 ~1s 內天使用「啊」答你：你的 f0 輪廓原樣還你（melisma
一起），三部＝bass 原八度＋alto +12＋sop 上方全音階三度（KeyTracker auto
定調，respond2 08-04 驗過的那顆）。

為什麼這條能「及時」而 respond2 要等 ~5s：
  - 材料固定＝take17 母音循環（score_sing/v7.1 stems 同款、Harry 過耳），
    不用 harvest/hubert 整句前處理——v7.1 實測正典塊渲 RTF ~0.04/聲部，
    4s 句 × 3 聲部 ≈ 0.5s 渲完。
  - f0 用 parselmouth 整句抽（~0.1s），和聲＝純平移＝零成本。
  - G33b 的重開條件在此成立：「天使晚一秒進場＝call-response 語法」，
    延遲不是缺陷是樂句結構。

斷句走**音高門**不走能量門（08-11 血訓：USB PnP 輸入僅 ~−61dBFS、與底噪
差 0.3dB＝能量 gate 全滅；parselmouth 對位準無感）。輸出音量＝注入 vol
0.06（stems 同款）＝與他的輸入位準脫鉤。

回應播放期間輸入硬靜音（respond2 紅線：唱/播結構上不同時活著＝回授、
DAF、殘響滲句三殺）。

跑: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python respond_live.py \
      [--key auto] [--in-name "USB PnP"] [--file in.wav out.wav]
"""
import argparse
import threading
import time

import numpy as np
import parselmouth
import soundfile as sf
import torch

import spike_stream6 as S

SR, HOP = 44100, 512
MAJ = [0, 2, 4, 5, 7, 9, 11]

ap = argparse.ArgumentParser()
ap.add_argument("--in-name", default="USB PnP")
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")
ap.add_argument("--block", type=float, default=0.24, help="斷句判定粒度")
ap.add_argument("--sblock", type=int, default=512)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--key", default="auto",
                help="整數移調或 auto（音級分布逐句估，respond2 同款）")
ap.add_argument("--gap", type=int, default=2,
                help="斷句靜默塊數（2 塊 ≈ 0.5s）")
ap.add_argument("--min-voiced", type=int, default=2,
                help="最短句（有聲塊數）；短於此當清嗓丟掉")
ap.add_argument("--max-s", type=float, default=15.0, help="單句上限秒數")
ap.add_argument("--mute-tail", type=float, default=0.25,
                help="回應播完再靜音的秒數（吃喇叭衰減）")
ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"),
                help="離線驗證：整檔當一句渲回應")
a = ap.parse_args()

# 三部（現役 6x 嘴）：(名, 模型, spk, gain, 半音位移, 是否全音階三度)
VOICES = [("bass", f"{S.DDSP}/exp/reflow-bass1/model_32000.pt", 1, 1.0,
           0, False),
          ("alto", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt", 2, 0.85,
           12, False),
          ("sop", f"{S.DDSP}/exp/reflow-sop3/model_20000.pt", 1, 0.7,
           12, True)]

print("載入模型…", flush=True)
svcs = [S.Svc([(p, 0.0, g, sp)], step=2, t_start=0.85)
        for _, p, sp, g, *_x in VOICES]

# 母音 units（同 score_sing/v7.1）
mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
mic0 = mic0[:, 0]
_f, _v, _m, uv0 = svcs[0].prep(mic0[:60 * SR], -60.0, want_uv=True)
runs, _i = [], 0
while _i < len(uv0):
    if not uv0[_i]:
        _j = _i
        while _j < len(uv0) and not uv0[_j]:
            _j += 1
        runs.append((_i, _j))
        _i = _j
    else:
        _i += 1
_a0, _b0 = max(runs, key=lambda r: r[1] - r[0])
_mid = (_a0 + _b0) // 2
with torch.no_grad():
    UU = svcs[0].encode(
        mic0[max(0, (_mid - 40) * HOP):(_mid + 40) * HOP])[:, 8:-8]
NBANK = UU.size(1)
print(f"母音庫 {NBANK} 幀", flush=True)


class KeyTracker:
    """respond2 同款（08-04 驗過）：raw f0 音級分布 → 大調旋轉角。"""

    MIN_FRAMES = 150

    def __init__(self):
        self.h = np.zeros(12)

    def push(self, f0):
        v = np.asarray(f0)
        v = v[v > 0]
        if len(v):
            np.add.at(self.h, np.round(
                69 + 12 * np.log2(v / 440.0)).astype(int) % 12, 1.0)

    def root(self):
        """大調主音 pitch class；資料不足回 None。"""
        if self.h.sum() < self.MIN_FRAMES:
            return None
        cov = [sum(self.h[(r + d) % 12] for d in MAJ) for r in range(12)]
        return int(np.argmax(cov))


kt = KeyTracker() if str(a.key).lower() == "auto" else None
FIXED_ROOT = 0 if kt is not None else (-int(a.key)) % 12


def dia_third(notes, root):
    """逐幀：大調 root 上、note 的上方全音階三度＝半音位移陣列（中值平滑）。"""
    sh = np.empty(len(notes))
    for i, m in enumerate(notes):
        rel = (int(round(m)) - root) % 12
        idx = int(np.argmin([min(abs(rel - s), 12 - abs(rel - s))
                             for s in MAJ]))
        sh[i] = MAJ[(idx + 2) % 7] + 12 * ((idx + 2) // 7) - MAJ[idx]
    if len(sh) >= 9:                     # 換音邊界的抖動抹掉
        from scipy.ndimage import median_filter
        sh = median_filter(sh, size=9, mode="nearest")
    return sh


def render_answer(seg):
    """一句音訊 → (回應音訊, 渲染秒數) 或 None（沒唱到東西）。"""
    p = parselmouth.Sound(seg, SR).to_pitch(
        time_step=HOP / SR, pitch_floor=60, pitch_ceiling=800)
    f0 = p.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.sum() < 20:                # <0.25s 有聲＝清嗓，不答
        return None
    n = len(f0)
    if kt is not None:
        kt.push(f0)
        root = kt.root()
        root = 0 if root is None else root
    else:
        root = FIXED_ROOT
    # f0 內插過無聲段＋平滑（score_sing 同款）；vol＝有聲 0.06、無聲 0
    idx = np.where(voiced)[0]
    f0i = np.interp(np.arange(n), idx, f0[idx])
    f0i = np.convolve(f0i, np.ones(3) / 3, "same")
    vol = np.convolve(np.where(voiced, 0.06, 0.0), np.ones(7) / 7, "same")
    notes = 69 + 12 * np.log2(f0i / 440.0)
    t0 = time.perf_counter()
    y = None
    CHF = 800
    with torch.no_grad():
        for vi, (nm, _p2, _sp, _g, semi, third) in enumerate(VOICES):
            off = np.full(n, float(semi))
            if third:
                off += dia_third(notes + semi, root)
            f0v = f0i * 2 ** (off / 12.0)
            outs = []
            for fo in range(0, n, CHF):
                k = min(CHF, n - fo)
                if k <= 1:
                    break
                vw = vol[fo:fo + k]
                vol_t = torch.from_numpy(vw).float().to(
                    svcs[vi].device)[None, :, None]
                mask = S.upsample(
                    torch.from_numpy((vw > 0.005).astype(float)).float()
                    .to(svcs[vi].device)[None, :, None], HOP).squeeze(-1)
                un = UU[:, (fo + np.arange(k)) % NBANK]
                au = svcs[vi].infer(np.zeros(k * HOP), 0.0, -60.0,
                                    units_override=un,
                                    feats=(f0v[fo:fo + k], vol_t, mask),
                                    ratios=[None])[0]
                outs.append(au.cpu().numpy())
            w = np.concatenate(outs)
            y = w if y is None else y[:len(w)] + w[:len(y)]
    y = np.clip(y * a.gain, -1, 1).astype("float32")
    return y, time.perf_counter() - t0


# ── 離線驗證模式 ──
if a.file:
    x, sr = sf.read(a.file[0], dtype="float64", always_2d=True)
    assert sr == SR
    r = render_answer(x[:, 0])
    assert r is not None, "整檔無有聲段"
    y, dt = r
    sf.write(a.file[1], y, SR)
    print(f"{len(x)/SR:.1f}s → 渲染 {dt:.2f}s（RTF {dt/(len(x)/SR):.2f}）"
          f" → {a.file[1]}", flush=True)
    raise SystemExit


def tick_note(x):
    try:
        p = parselmouth.Sound(x, SR).to_pitch(
            time_step=0.05, pitch_floor=65, pitch_ceiling=800)
        f = p.selected_array["frequency"]
        f = f[f > 0]
        if len(f) < 2:
            return None
        return float(np.median(f))
    except Exception:
        return None


blk = int(a.block * SR)
in_q, out_q = [], []
qlock = threading.Lock()
st = {"die": False, "mode": "listen", "cap": [], "vc": 0, "sq": 0,
      "tickbuf": np.zeros(0), "tail": 0, "phr": 0, "wait": [],
      "tstop": 0.0, "xrun": 0}
dmp = {"mic": [], "ans": []}


def worker():
    # res＝不滿一塊的餘數。v1 bug：sblock 512 的小塊沒累積、永遠湊不滿
    # 0.24s 判定塊＝斷句器全程沒跑（08-11 live 首試「沒反應」的根因）。
    res = np.zeros(0, dtype="float32")
    while not st["die"]:
        with qlock:
            pend = np.concatenate(in_q) if in_q else None
            in_q.clear()
            playing = bool(out_q) or st["tail"] > 0
        if pend is None:
            time.sleep(0.005)
            continue
        if st["mode"] == "play":
            res = np.zeros(0, dtype="float32")
            if playing:
                continue                 # 硬靜音：播放期輸入不存在
            st["mode"] = "listen"
            st["cap"], st["vc"], st["sq"] = [], 0, 0
            st["tickbuf"] = np.zeros(0)
            print("（聽……）", flush=True)
            continue
        # LISTEN：逐塊斷句（音高門）
        res = np.concatenate([res, pend])
        nblk = len(res) // blk
        chunks, res = ([res[i * blk:(i + 1) * blk] for i in range(nblk)],
                       res[nblk * blk:])
        for chunk in chunks:
            st["tickbuf"] = np.concatenate([st["tickbuf"], chunk])[-blk * 2:]
            note = tick_note(st["tickbuf"][-int(0.25 * SR):])
            if note is not None:
                if st["vc"] == 0 and not st["cap"]:
                    pass
                st["vc"] += 1
                st["sq"] = 0
            elif st["cap"]:
                st["sq"] += 1
            if st["cap"] or note is not None:
                st["cap"].append(chunk)
            over = len(st["cap"]) * a.block > a.max_s
            if st["cap"] and (st["sq"] >= a.gap or over):
                seg = np.concatenate(st["cap"])
                nv = st["vc"]
                st["cap"], st["vc"], st["sq"] = [], 0, 0
                if nv < a.min_voiced:
                    continue             # 清嗓，不答
                st["mode"] = "play"
                st["tstop"] = time.time()
                print(f"收句（{len(seg)/SR:.1f}s）→ 渲染…", flush=True)
                r = render_answer(seg)
                if r is None:
                    st["mode"] = "listen"
                    continue
                y, dt = r
                st["phr"] += 1
                wait = time.time() - st["tstop"]
                st["wait"].append(wait)
                dmp["mic"].append(seg.copy())
                dmp["ans"].append(y.copy())
                with qlock:
                    for j in range(0, len(y), blk):
                        out_q.append(y[j:j + blk])
                    st["tail"] = int(a.mute_tail * SR)
                print(f"句{st['phr']} {len(seg)/SR:4.1f}s → 渲 {dt:.2f}s"
                      f"（RTF {dt/(len(seg)/SR):.2f}）→ 回應 {len(y)/SR:.1f}s"
                      f"｜句尾到開播 {wait:.2f}s", flush=True)
                break                    # 本批剩的輸入屬於靜音期，丟


OB = [np.zeros(0, dtype="float32")]


def cb(indata, outdata, frames, tinfo, status):
    if status:
        st["xrun"] += 1
    with qlock:
        in_q.append(indata[:, 0].copy())
        buf = OB[0]
        while out_q and len(buf) < frames:
            buf = np.concatenate([buf, out_q.pop(0)])
        if not out_q and len(buf) <= frames and st["tail"] > 0:
            st["tail"] = max(0, st["tail"] - frames)
    if len(buf) >= frames:
        outdata[:, 0] = buf[:frames]
        outdata[:, 1] = buf[:frames]
        OB[0] = buf[frames:]
    else:
        outdata.fill(0)
        if len(buf):
            outdata[:len(buf), 0] = buf
            outdata[:len(buf), 1] = buf
        OB[0] = np.zeros(0, dtype="float32")


import sounddevice as sd  # noqa: E402

# 預熱（kernel 編譯；不然第一句多付 ~2s）
with torch.no_grad():
    for v_ in svcs:
        v_.infer(np.zeros(61 * HOP), 0.0, -60.0,
                 units_override=UU[:, np.arange(61) % NBANK],
                 feats=(np.full(61, 220.0),
                        torch.full((1, 61, 1), 0.05, device=v_.device),
                        torch.ones(1, 61 * HOP, device=v_.device)),
                 ratios=[None])
threading.Thread(target=worker, daemon=True).start()
print(f"ready（唱一句、停 {a.gap * a.block:.1f}s＝收句；天使用「啊」答你；"
      f"Ctrl-C 結束）", flush=True)
try:
    with sd.Stream(samplerate=SR, blocksize=a.sblock, channels=(1, 2),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency="low", callback=cb):
        while True:
            time.sleep(2)
except KeyboardInterrupt:
    st["die"] = True
    if dmp["mic"]:
        pairs_m, pairs_a = [], []
        for m, y in zip(dmp["mic"], dmp["ans"]):
            L = max(len(m), len(y))
            pairs_m.append(np.pad(m, (0, L - len(m))))
            pairs_a.append(np.pad(y.astype("float64"), (0, L - len(y))))
        sf.write("scratchpad/rlive_mic.wav", np.concatenate(pairs_m), SR)
        sf.write("scratchpad/rlive_ans.wav", np.concatenate(pairs_a), SR)
        print("\ndump: scratchpad/rlive_mic.wav / rlive_ans.wav（逐句對齊）",
              flush=True)
    w = np.array(st["wait"]) if st["wait"] else np.zeros(1)
    print(f"共 {st['phr']} 句｜句尾到開播 p50 {np.median(w):.2f}s｜"
          f"xrun {st['xrun']}", flush=True)
    print("bye")
