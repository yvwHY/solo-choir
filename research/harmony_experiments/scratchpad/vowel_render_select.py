"""vowel_render_select - choose material by the vowel AFTER rendering (v22)

The most expensive lesson of that day: **how it sounds at the source is not the
criterion.** A vowel recorded deliberately, the closest of the whole session at
d=0.11 at the source, still came out at F2 1330 after rendering; an ordinary-looking
stretch of the corpus rendered into a clean version of the same vowel (221/907).
Which stretch's units survive the model is close to a lottery, so do not guess:
render every candidate once and pick the winner by its distance after rendering.

The flow: vowel_source_cands.json (30 candidates per vowel) plus the deliberate
recordings, each rendered as one note at MIDI 48, F1 and F2 measured, and the
smallest post-render distance taken per vowel, printing VOWEL_LAYERS.
If no candidate lands on the vowel after rendering, that is an honest death (one
vowel has now lost five sets of candidates).

Run: cd harmony && PYTHONPATH=. ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_render_select.py [--per 30]
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, nearest_vowel, REF = (_a["lpc_formants"], _a["nearest_vowel"],
                                    _a["REF"])
_s = {}
exec(compile(open("scratchpad/vowel_source_scan.py").read()
             .split("\ndef main()")[0], "vss", "exec"), _s)
dist = _s["dist"]
SR, HOP, MIDI = 44100, 512, 48
F0HZ = 440.0 * 2 ** ((MIDI - 69) / 12.0)
DMAX = 0.25                              # vowel distance threshold; past it, cleanliness decides
MODEL = "reflow-bass1/model_32000.pt"
ORDER = ["喔", "欸", "咿", "嗚", "啊"]      # vowel labels; they key bank_live.VOWEL_LAYERS


def hnr(x, f0):
    """Harmonic-to-noise ratio in dB, the criterion for quality. The lesson: v22
    picked on formant distance alone and all four layers lost 1 to 6 dB of HNR, which
    was heard immediately as worse than the previous version. **Accuracy and
    cleanliness have to be managed together** - the same clip differs by 6 dB with
    nothing but the window position changed."""
    x = x - x.mean()
    n = len(x)
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    hb = np.zeros(len(fr), bool)
    k = 1
    while k * f0 < 8000:
        hb |= np.abs(fr - k * f0) < max(f0 * 0.06, 12)
        k += 1
    band = (fr > 60) & (fr < 8000)
    h = X[hb & band].sum()
    return float(10 * np.log10(h / max(X[band].sum() - h, 1e-20)))


def meas(x):
    x16 = resample_poly(x, 16000, SR)
    n = 1600
    f = [lpc_formants(x16[i:i + n]) for i in range(0, max(1, len(x16) - n), n)]
    f = [q for q in f if len(q) >= 2]
    if not f:
        return float("nan"), float("nan")
    return (float(np.median([q[0] for q in f])),
            float(np.median([q[1] for q in f])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=30)
    a = ap.parse_args()
    import torch
    import spike_stream6 as S
    cands = json.load(open("scratchpad/vowel_source_cands.json"))
    # The window used by v21, the one judged better by ear, goes into the candidate
    # pool explicitly so it has a chance to win back on HNR; otherwise the search only
    # ever circles among new candidates.
    for v, clip, t in (("欸", "clip_0143.wav", 2.17), ("咿", "clip_0155.wav",
                       3.97), ("嗚", "clip_0035.wav", 3.25),
                       ("啊", "clip_0009.wav", 1.43)):
        cands.setdefault(v, []).insert(0, {
            "clip": f"{_s['C']}/{clip}", "t0": t - 0.15, "t1": t + 0.15,
            "F1": 0.0, "F2": 0.0, "vowel": v, "d": 0.0, "rms": 0.0})
    # the deliberate recordings (from vowel_record_ritual) go into the pool too, sweeping the whole take
    for p in sorted(glob.glob("scratchpad/vowel_takes/*.wav")):
        v = os.path.basename(p)[:-4]
        if v not in cands:
            continue
        x, fs = sf.read(p, always_2d=True)
        dur = len(x) / fs
        for t0 in np.arange(0.3, max(0.4, dur - 1.2), 0.35):
            cands[v].append({"clip": p, "t0": float(t0),
                             "t1": float(t0 + 0.3), "F1": 0.0, "F2": 0.0,
                             "vowel": v, "d": 0.0, "rms": 0.0})

    svc = S.Svc([(f"{S.DDSP}/exp/{MODEL}", 0.0, 1.0, 1)], step=2,
                t_start=0.85)
    NF = int(2.0 * SR / HOP)
    f0 = np.full(NF, 440.0 * 2 ** ((MIDI - 69) / 12.0))
    vol = np.full(NF, 0.06)
    vol[:4] = np.linspace(0, 0.06, 4)
    vol_t = torch.from_numpy(vol).float().to(svc.device)[None, :, None]
    mask = torch.ones(1, NF * HOP, device=svc.device)

    best, lines = {}, []
    for v in ORDER:
        rows = []
        for r in cands[v][:a.per + 20]:
            x, fs = sf.read(r["clip"], always_2d=True)
            mid = (r["t0"] + r["t1"]) / 2
            a0 = max(0, int((mid - 0.5) * fs))
            seg = x[a0:a0 + int(1.0 * fs), 0].astype("float64")
            if len(seg) < int(0.6 * fs):
                continue
            if fs != SR:
                seg = resample_poly(seg, SR, fs)
            with torch.no_grad():
                UU = svc.encode(seg)[:, 8:-8]
                nb = UU.size(1)
                torch.manual_seed(1234 + MIDI)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask),
                               ratios=[None])[0].cpu().numpy()
            body = au[int(0.5 * SR):int(1.8 * SR)].astype("float64")
            F1, F2 = meas(body)
            if not np.isfinite(F1) or not np.isfinite(F2):
                continue
            rows.append({"clip": r["clip"], "mid": mid, "F1": F1, "F2": F2,
                         "d": dist(F1, F2, v), "hnr": hnr(body, F0HZ),
                         "v": nearest_vowel(F1, F2)[0]})
        # the two criteria cannot be traded off: sieve on the correct vowel and the distance threshold FIRST, then pick the cleanest by HNR
        hit = sorted([q for q in rows if q["v"] == v and q["d"] <= DMAX],
                     key=lambda q: -q["hnr"])
        rows.sort(key=lambda q: q["d"])
        print(f"\n{v}: rendered {len(rows)}, of which {len(hit)} have the right vowel "
              f"and d<={DMAX} (ordered by HNR)", flush=True)
        for q in hit[:3]:
            print(f"   {os.path.basename(q['clip'])} @{q['mid']:.1f}s  "
                  f"rendered {q['F1']:.0f}/{q['F2']:.0f}  d {q['d']:.2f}  "
                  f"HNR {q['hnr']:.1f}dB")
        if hit:
            b = hit[0]
            best[v] = b
            lines.append(f'                ("{v}", "{b["clip"]}", '
                         f'{b["mid"] - 0.5:.2f}, {b["mid"] + 0.5:.2f}, 0.0),'
                         f'  # rendered {b["F1"]:.0f}/{b["F2"]:.0f} '
                         f'HNR {b["hnr"]:.0f}')
    print("\n-- layers selected after rendering --")
    print("\n".join(lines) if lines else "(none)")
    for v in ORDER:
        if v not in best:
            print(f"{v}: none of the {a.per}+ candidates renders as that vowel; this voice cannot do it")
    json.dump(best, open("scratchpad/vowel_render_best.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
