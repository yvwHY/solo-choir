"""respond_live — the immediate live answering mode (2026-08-11, "try your
immediate live answering idea")

You sing a phrase; within about a second of its end the parts answer on "ah",
returning your own f0 contour, melisma included. Three parts: bass at your
octave, alto at +12, and soprano a diatonic third above (with KeyTracker on
auto, the one verified in respond2 on 2026-08-04).

Why this path can be immediate while respond2 waits about 5 s:
  - The material is fixed, the take-17 vowel loop, the same as the score_sing
    v7.1 stems, already passed by ear. There is no harvest or HuBERT
    pre-processing of the whole phrase; v7.1 measured a reference block render
    at an RTF of about 0.04 per part, so a 4 s phrase across 3 parts renders in
    roughly 0.5 s.
  - f0 is extracted for the whole phrase with parselmouth, about 0.1 s, and the
    harmony is pure transposition, which is free.
  - The condition for reopening G33b holds here: a part entering a second later
    is call-and-response grammar, so the latency is not a defect but the
    structure of the phrase.

Phrases are ended by a **pitch gate rather than an energy gate** (the lesson of
2026-08-11: a USB plug-and-play input sat at about -61 dBFS, only 0.3 dB above
its noise floor, so every energy gate failed; parselmouth is indifferent to
level). The output level is the injected vol of 0.06, as in the stems, which
decouples it from the singer's input level.

The input is hard-muted while a response plays (respond2's red line: singing and
playback must not be alive at the same time, or feedback, delayed auditory
feedback and reverb bleeding into the phrase all follow).

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python respond_live.py \
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
# The default is a macOS device name on a zh-Hant system; it is a match target,
# not prose, so it is not translated.
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")
ap.add_argument("--block", type=float, default=0.24, help="granularity of the phrase-end decision")
ap.add_argument("--sblock", type=int, default=512)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--key", default="auto",
                help="an integer transposition, or auto, which estimates from the pitch-class distribution per phrase, as respond2 does")
ap.add_argument("--gap", type=int, default=2,
                help="blocks of silence that end a phrase (2 blocks is about 0.5 s)")
ap.add_argument("--min-voiced", type=int, default=2,
                help="shortest phrase, in voiced blocks; anything shorter is treated as clearing the throat and discarded")
ap.add_argument("--max-s", type=float, default=15.0, help="maximum seconds in one phrase")
ap.add_argument("--mute-tail", type=float, default=0.25,
                help="seconds to stay muted after a response finishes, to cover the speaker decay")
ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"),
                help="offline check: treat the whole file as one phrase and render a response")
a = ap.parse_args()

# The three parts, on the current 6x voices: (name, model, spk, gain, semitone shift, diatonic third or not)
VOICES = [("bass", f"{S.DDSP}/exp/reflow-bass1/model_32000.pt", 1, 1.0,
           0, False),
          ("alto", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt", 2, 0.85,
           12, False),
          ("sop", f"{S.DDSP}/exp/reflow-sop3/model_20000.pt", 1, 0.7,
           12, True)]

print("loading models...", flush=True)
svcs = [S.Svc([(p, 0.0, g, sp)], step=2, t_start=0.85)
        for _, p, sp, g, *_x in VOICES]

# vowel units, as in score_sing v7.1
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
print(f"vowel bank: {NBANK} frames", flush=True)


class KeyTracker:
    """As in respond2, verified 2026-08-04: the pitch-class distribution of the
    raw f0 to a major-key rotation."""

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
        """The pitch class of the major tonic; None when there is too little data."""
        if self.h.sum() < self.MIN_FRAMES:
            return None
        cov = [sum(self.h[(r + d) % 12] for d in MAJ) for r in range(12)]
        return int(np.argmax(cov))


kt = KeyTracker() if str(a.key).lower() == "auto" else None
FIXED_ROOT = 0 if kt is not None else (-int(a.key)) % 12


def dia_third(notes, root):
    """Per frame: the diatonic third above note within the major root, as an
    array of semitone shifts, median-smoothed."""
    sh = np.empty(len(notes))
    for i, m in enumerate(notes):
        rel = (int(round(m)) - root) % 12
        idx = int(np.argmin([min(abs(rel - s), 12 - abs(rel - s))
                             for s in MAJ]))
        sh[i] = MAJ[(idx + 2) % 7] + 12 * ((idx + 2) // 7) - MAJ[idx]
    if len(sh) >= 9:                     # smooth out the jitter at note boundaries
        from scipy.ndimage import median_filter
        sh = median_filter(sh, size=9, mode="nearest")
    return sh


def render_answer(seg):
    """One phrase of audio to (response audio, render seconds), or None when
    nothing was sung."""
    p = parselmouth.Sound(seg, SR).to_pitch(
        time_step=HOP / SR, pitch_floor=60, pitch_ceiling=800)
    f0 = p.selected_array["frequency"]
    voiced = f0 > 0
    if voiced.sum() < 20:                # under 0.25 s of voiced audio is clearing the throat; no answer
        return None
    n = len(f0)
    if kt is not None:
        kt.push(f0)
        root = kt.root()
        root = 0 if root is None else root
    else:
        root = FIXED_ROOT
    # f0 is interpolated across unvoiced spans and smoothed, as in score_sing;
    # vol is 0.06 where voiced and 0 where not
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


# -- offline check mode --
if a.file:
    x, sr = sf.read(a.file[0], dtype="float64", always_2d=True)
    assert sr == SR
    r = render_answer(x[:, 0])
    assert r is not None, "no voiced span in the whole file"
    y, dt = r
    sf.write(a.file[1], y, SR)
    print(f"{len(x)/SR:.1f}s -> rendered in {dt:.2f}s (RTF {dt/(len(x)/SR):.2f})"
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
    # res is the remainder that does not fill a block. The v1 bug: small
    # sblock 512 chunks were never accumulated, so the 0.24 s decision block was
    # never filled and the phrase detector never ran at all, which is why the
    # first live attempt on 2026-08-11 did nothing.
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
                continue                 # hard mute: during playback the input does not exist
            st["mode"] = "listen"
            st["cap"], st["vc"], st["sq"] = [], 0, 0
            st["tickbuf"] = np.zeros(0)
            print("(listening...)", flush=True)
            continue
        # LISTEN: end the phrase block by block, on the pitch gate
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
                    continue             # clearing the throat; no answer
                st["mode"] = "play"
                st["tstop"] = time.time()
                print(f"phrase captured ({len(seg)/SR:.1f}s), rendering...", flush=True)
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
                print(f"phrase {st['phr']} {len(seg)/SR:4.1f}s -> render {dt:.2f}s"
                      f" (RTF {dt/(len(seg)/SR):.2f}) -> response {len(y)/SR:.1f}s"
                      f" | phrase end to playback {wait:.2f}s", flush=True)
                break                    # the rest of this batch belongs to the muted period; discard


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

# Warm-up so the kernels compile; without it the first phrase pays about 2 s more
with torch.no_grad():
    for v_ in svcs:
        v_.infer(np.zeros(61 * HOP), 0.0, -60.0,
                 units_override=UU[:, np.arange(61) % NBANK],
                 feats=(np.full(61, 220.0),
                        torch.full((1, 61, 1), 0.05, device=v_.device),
                        torch.ones(1, 61 * HOP, device=v_.device)),
                 ratios=[None])
threading.Thread(target=worker, daemon=True).start()
print(f"ready: sing a phrase and pause {a.gap * a.block:.1f}s to end it; the parts answer on 'ah'; "
      "Ctrl-C to finish", flush=True)
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
        print("\ndump: scratchpad/rlive_mic.wav / rlive_ans.wav (aligned phrase by phrase)",
              flush=True)
    w = np.array(st["wait"]) if st["wait"] else np.zeros(1)
    print(f"{st['phr']} phrases | phrase end to playback p50 {np.median(w):.2f}s | "
          f"xrun {st['xrun']}", flush=True)
    print("bye")
