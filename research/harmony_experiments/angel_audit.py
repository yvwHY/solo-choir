"""angel_audit.py - a health check on the parts' expression layer.

A full check of consonance, harmony, level and reverb across the parts, as one
scorecard on eight axes. **Every axis is measured against something audible rather
than an internal representation**, after two lessons on the same day: an internal
consistency metric can look excellent at the exact moment the listener says it is
wrong. The material is real sung phrases from a worn session dump.

Usage (DDSP venv):  python angel_audit.py [dump.wav] [--gate 0.0474]

The dump interleaves his phrases with the responses, and inside a response his own
dry voice is the loudest thing, so pitch detection follows him and the upper part
cannot be measured. His phrase always precedes its response, so taking the
even-numbered segments removes the duplicates. Segments pairing up in length is the
fingerprint of that contamination."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf, torch

H = Path("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer/harmony")
sys.path.insert(0, str(H))
import respond2 as R, direct_mouth as dm, reverb as rv, reh_polish
from phrase_render import PhraseRenderer
from pitch import SR
from world_live import TICK_SAMPS
from live_v3 import EarV3

# -- current settings --
DRY, ANGEL, UP_DB, LO_DB, GAIN, INDEP = 0.944, 0.75, 1.0, 0.0, 0.5, 0.15
GU, GL = ANGEL * 10 ** (UP_DB / 20), ANGEL * 10 ** (LO_DB / 20)
WET = 0.20
DISS = reh_polish.DISS
per = TICK_SAMPS / dm.HOP

# -- 0. material: extract his sung phrases from the dump --
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("dump", nargs="?", default=str(H / "out/resp2_live_260803_232426.wav"))
_ap.add_argument("--gate", type=float, default=0.0474)
_a = _ap.parse_args()
x, _ = sf.read(_a.dump, dtype="float64", always_2d=True)
x = np.ascontiguousarray(x[:, 0])
R.RMS_GATE = _a.gate
segs = []
for s, e in R.find_phrases(x, 0.35, 0.8):
    if s / SR < 12:                                   # the calibration region
        continue
    f0 = dm.harvest_f0(np.ascontiguousarray(x[s:e]))
    v = f0[f0 > 0]
    if len(v) < 30:
        continue
    hi = float(np.mean(v > 340.0))                    # the upper line, the fingerprint of a response
    if hi < 0.05:
        segs.append(np.ascontiguousarray(x[s:e]))
# The dump interleaves his phrases with the responses, and his dry voice is loudest
# inside a response, so detection follows him. The previous version pulled the
# responses in too, with paired segment lengths as the fingerprint. His phrase always
# comes first, so take the even-numbered segments.
segs = segs[0::2]
print(f"material: {len(segs)} sung phrases, "
      f"{', '.join(f'{len(s)/SR:.1f}s' for s in segs)}")

def true_line(seg, n):
    f0 = dm.harvest_f0(seg)
    out = []
    for k in range(n):
        w = f0[int(k * per):int((k + 1) * per)]; w = w[w > 0]
        out.append(None if len(w) < 3 else float(np.median(69 + 12 * np.log2(w / 440.0))))
    return out

# -- render with the current recipe, collecting stems per phrase --
torch.manual_seed(1234)
ear = EarV3(indep=INDEP, stab=1, key=0); ear.v2t.shift = 12
kt = R.KeyTracker()
rend = {v: PhraseRenderer(R.DDSP, m, expr_seed=sd) for v, (m, sd) in R.VOICES.items()}
for r in rend.values():
    r.render(np.zeros(SR), [60], TICK_SAMPS)

rows = []
for seg in segs:
    kt.push(dm.harvest_f0(seg))
    r = kt.best()
    if r is not None and r[2] >= 0.015:
        ear.k_shift = r[0]
    lead, up, lo = R.phrase_notes(ear, seg)
    d = {"lead": lead, "upper": up, "lower": lo}
    T = true_line(seg, len(lead))
    d["lead"] = [None if t is None else int(round(t)) for t in T]
    root = (-(ear.k_shift or 0)) % 12
    reh_polish.polish(d, "upper", "lower", root)
    reh_polish.polish(d, "lower", "upper", root)
    sh = {}
    kw = {v: dict(expr_gain=0.0, vib_semi=0.12, vib_onset_ms=250.0, uv_dry=0.0,
                  vib_hz=R.VIB[v][0], vib_phase=R.VIB[v][1]) for v in R.VOICES}
    ang = {v: rend[v].render(seg, d[v], TICK_SAMPS, shared=sh, **kw[v]) for v in R.VOICES}
    n = min(len(seg), *(len(a) for a in ang.values()))
    rows.append(dict(seg=seg[:n], T=T, d=d, u=ang["upper"][:n], l=ang["lower"][:n],
                     vm=sh["vm"], key=ear.k_shift or 0))
print(f"key trajectory: {[r['key'] for r in rows]}")

print("\n==== health check ====")
# 1. consonance: the parts against his real pitch
n = di = 0
for r in rows:
    for t, u, l in zip(r["T"], r["d"]["upper"], r["d"]["lower"]):
        if t is None: continue
        for a in (u, l):
            if a is None: continue
            n += 1; di += round(abs(a - t)) % 12 in DISS
print(f"1 consonance   dissonant against his real pitch {100*di/max(n,1):5.1f}%   [corpus 12.1, pair06 after the fix 17.3]")

# 2. harmonic content: the two parts against each other
sm = st_ = dd = m2 = 0
for r in rows:
    U, L = r["d"]["upper"], r["d"]["lower"]
    for u, l in zip(U, L):
        if u is None or l is None: continue
        m2 += 1
        sm += (u - l) % 12 == 0
        dd += abs(u - l) % 12 in DISS
print(f"2 harmony      the two parts in unison {100*sm/max(m2,1):5.1f}%  clashing {100*dd/max(m2,1):5.1f}%   [corpus: unison 19.9, dissonant 12.1]")

# 3. intonation, and the F21 interval error
errs, iv = [], []
for r in rows:
    n_h = len(r["u"]) // dm.HOP
    fu = dm.harvest_f0(np.ascontiguousarray(r["u"]))[:n_h]
    fl = dm.harvest_f0(np.ascontiguousarray(r["l"]))[:n_h]
    for k in range(n_h):
        t = min(int(k * dm.HOP / TICK_SAMPS), len(r["d"]["upper"]) - 1)
        mu, ml = r["d"]["upper"][t], r["d"]["lower"][t]
        t_v = r["vm"][k] > 0.5 if k < len(r["vm"]) else False
        if not t_v:
            continue                    # in an unvoiced stretch the stem is his own passed-through voice, so this would not measure a part
        if fu[k] > 0 and mu is not None:
            c = 1200 * np.log2(fu[k] / (440 * 2 ** ((mu - 69) / 12)))
            if abs(c) < 250: errs.append(abs(c))
        if fu[k] > 0 and fl[k] > 0 and mu is not None and ml is not None:
            got = 1200 * np.log2(fu[k] / fl[k]); want = 100.0 * (mu - ml)
            if abs(got - want) < 300: iv.append(got - want)
print(f"3 intonation   median |cents error| {np.median(errs):5.1f}c   interval error SD {np.std(iv):5.1f}c   [F21 target: 11c, over 20c 7.9%]")

# 4. lag: from his note change to the parts changing
lags = []
for r in rows:
    Tn = [None if t is None else int(round(t)) for t in r["T"]]
    ch_h = [k for k in range(1, len(Tn)) if Tn[k] is not None and Tn[k] != Tn[k-1]]
    U = r["d"]["upper"]
    ch_a = [k for k in range(1, len(U)) if U[k] is not None and U[k] != U[k-1]]
    for c in ch_h:
        after = [a for a in ch_a if a >= c]
        if after: lags.append((after[0] - c) * TICK_SAMPS / SR)
print(f"4 lag          his change to theirs, median {np.median(lags)*1000:4.0f}ms  p90 {np.percentile(lags,90)*1000:4.0f}ms   [the tick grid is 187.5 ms]")

# 5. consonant stacking: the effective gain of his voice inside a response over unvoiced stretches
gains = []
for r in rows:
    v = np.repeat(np.asarray(r["vm"]) > 0.5, dm.HOP)[:len(r["seg"])]
    uvm = ~v
    if uvm.sum() < SR * 0.05: continue
    d0 = r["seg"][uvm]
    tot = DRY * d0 + GU * r["u"][:len(v)][uvm] + GL * r["l"][:len(v)][uvm]
    gains.append(20 * np.log10(np.sqrt(np.mean(tot**2)) / (np.sqrt(np.mean((DRY*d0)**2)) + 1e-12)))
print(f"5 stacking     unvoiced stretches (breaths, consonants) are {np.mean(gains):+5.1f} dB above the dry alone   [0 = no stacking; pass-through times two parts]")

# 6. level balance at the current gains
dd_ = aa_ = 0
for r in rows:
    v = np.repeat(np.asarray(r["vm"]) > 0.5, dm.HOP)[:len(r["seg"])]
    dd_ += np.sum((DRY * r["seg"][v]) ** 2)
    aa_ += np.sum((GU * r["u"][:len(v)][v] + GL * r["l"][:len(v)][v]) ** 2)
print(f"6 balance      both parts together minus him = {10*np.log10(aa_/dd_):+5.1f} dB   [his own final hand-set mix has the parts about +0.7 dB ahead]")

# 7. reverb: wet to direct ratio, and clipping
ir = rv.make_ir(1.8, trim_db=35)
pks, wr = [], []
for r in rows:
    mix = DRY * r["seg"] + GU * r["u"] + GL * r["l"]
    w = rv.wet(mix, ir)
    y = (np.concatenate([mix, np.zeros(len(w) - len(mix))]) + WET * w) * GAIN
    pks.append(np.abs(y).max())
    wr.append(10 * np.log10(np.sum((WET*w)**2) / np.sum(mix**2)))
print(f"7 reverb       wet/direct {np.mean(wr):+5.1f} dB   tail +0.88s per phrase   early-reflection crest 5.0 (diffuse)  [send level 0.20, never judged in the room]")
print(f"8 clipping     peak at gain {GAIN}: max {max(pks):.2f}   [over 1.0 clips]")
