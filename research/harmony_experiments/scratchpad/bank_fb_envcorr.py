"""Envelope-domain feedback detection for the sampled choir: the groundwork test.

**Why this is not a rerun of G34.** G34 (`fb_duck_probe.py`) died on a structural
argument: in the streaming architecture the loop is `angel(t) = G * mic(t)` - a
live converting carrier, with G measured constant at 0.97:1 - so the envelope of
the whole loop is self-similar, any mic-to-output energy ratio tends to a constant,
and a delay sweep from 0 to 697 ms rises monotonically with no peak.
**bank_live has no such causal edge**: `y` is sampled playback from the banks
(`bank_live.py:1864`, `p.render(hopN, r)`) plus the pad and the reverb, coupled to
the microphone only through the discrete `note` decision. The banks keep sounding
for about three seconds after he stops. So the output is not a function of the
microphone, and G34's self-similarity premise does not hold here.

**This script answers one question**: between the energy envelopes of the
microphone and the output, is there a correlation peak at a NON-ZERO delay?
  A peak means a physical feedback path exists and its delay can be estimated, which
  is the precondition for discriminating at all.
  No peak means there is no estimable path, as in G34, and this route dies here too.

**Two hard-won lessons (GRAVEYARD G34 and G35) are built in:**
1. `dmp["mic"]` and `dmp["out"]` are appended in the same callback at the same place
   (`bank_live.py:1913-1914`), so the output track is on the GENERATION timeline,
   not the timeline of what the speakers emitted. The peak MUST therefore fall at a
   non-zero delay; a peak near 0 ms is a timeline artefact, not feedback.
2. Do not cut quiet/singing labels by percentiles of microphone energy - the
   correction to G35 established that those are bad labels. This script needs no
   labels at all: cross-correlation is descriptive.
"""
import sys

import numpy as np
import soundfile as sf

SR, HOP = 44100, 512
DUMP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
        "SoloChoirCode/260811_bt/ear/dump_0815/")
WIN = 0.2                                  # energy envelope window, about 200 ms


def env(path):
    """RMS envelope per hop. The output is the mono bus, that is the mixed sound the speakers actually emit."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: sr={sr}"
    ch = x.shape[1]
    x = x.mean(1)
    m = len(x) // HOP
    e = np.sqrt((x[:m * HOP].reshape(m, HOP) ** 2).mean(1) + 1e-20)
    return e, ch, len(x)


mic, mic_ch, mic_n = env(DUMP + "noise_mic.wav")
out, out_ch, out_n = env(DUMP + "noise_out.wav")
st = np.load(DUMP + "noise_st.npz")
m = min(len(mic), len(out))
mic, out = mic[:m], out[:m]

print(f"mic {mic_ch}ch {mic_n} smp {mic_n / SR:6.1f}s  "
      f"out {out_ch}ch {out_n} smp {out_n / SR:6.1f}s")
print(f"{m} frames in common at {HOP / SR * 1000:.1f}ms = {m * HOP / SR:.1f}s")
print(f"npz: {', '.join(f'{k}{st[k].shape}' for k in st.files)}")
nfr = len(st["mok"])
print(f"{nfr} frames in the npz (one per callback) -> callback hop "
      f"= {m / nfr * HOP / SR * 1000:.1f}ms\n")

# -- smooth the envelopes to about 200 ms, take the log and remove the mean; a ratio would fall into G34's constant-ratio trap --
w = max(1, int(round(WIN * SR / HOP)))
sm = lambda x: np.convolve(x, np.ones(w) / w, "same")
lm, lo_ = np.log(sm(mic) + 1e-9), np.log(sm(out) + 1e-9)

# only count frames where the output is actually sounding; correlation over silence is meaningless (the same reasoning as G34's `ref > 1e-5`)
live = sm(out) > np.percentile(sm(out), 40)
print(f"frames with the output sounding {int(live.sum())}/{m} = {live.mean() * 100:.0f}%")

# -- detrending (necessary) --
# Without it, the first run climbed from 0.302 at d=0 to 0.372 at the edge of the
# sweep at 1393 ms and had still not turned: both tracks share a slow trend on the
# scale of TENS OF SECONDS (the sectional and level structure of the whole
# recording) which lifts r at every delay, exactly the trivial effect G34 described.
# Feedback is a short-scale coupling, so subtracting a moving average over DETREND
# seconds leaves only syllable and phrase scale modulation, and the peak becomes
# visible.
DETREND = 3.0
wd = max(3, int(round(DETREND * SR / HOP)) | 1)
hp = lambda x: x - np.convolve(x, np.ones(wd) / wd, "same")
a, b = hp(lm), hp(lo_)
edge = wd                                  # the moving average is incomplete at both ends; trim them
a = a - a[live].mean()
b = b - b[live].mean()
live = live.copy()
live[:edge] = live[-edge:] = False

# -- delay sweep: a positive delay means the output leads the microphone, which
# feedback necessarily does; negative delays are the control --
# Physically expected: the output buffer (--sblock) plus the speaker-to-microphone
# air path, so tens to hundreds of ms.
# A negative delay has the microphone leading the output, which cannot physically be
# feedback, so it is the control and should be clearly lower.
print(f"\ndelay sweep (positive = output leads microphone; {DETREND}s trend removed):")
print("   delay ms   r        n")
rows = []
for d in range(-86, 173, 2):              # frames, -1.0 s to +2.0 s
    if d >= 0:
        x, y = a[d:], b[:m - d] if d else b
        msk = live[:m - d] if d else live
    else:
        x, y = a[:m + d], b[-d:]
        msk = live[-d:]
    n = int(msk.sum())
    if n < 200:
        continue
    x, y = x[msk], y[msk]
    r = float(np.corrcoef(x, y)[0, 1])
    rows.append((d, r, n))

for d, r, n in rows:
    bar = "#" * max(0, int(r * 60))
    print(f"  {d * HOP / SR * 1000:7.0f}  {r:+.3f}  {n:6d}  {bar}")

pos = [t for t in rows if t[0] > 0]
dbest, rbest, _ = max(pos, key=lambda t: t[1])
r0 = next(r for d, r, _ in rows if d == 0)
redge = rows[-1][1]
print(f"\nbest positive delay d={dbest} frames = {dbest * HOP / SR * 1000:.0f}ms, r={rbest:+.3f}")
print(f"controls: d=0 r={r0:+.3f} (the timeline-artefact baseline), sweep edge r={redge:+.3f}")
print("Reading it: the peak has to fall at a non-zero delay and clearly above both d=0")
print("      and the edge to count as a physical feedback path. A monotone curve with no")
print("      peak means no estimable path, as in G34, and this route dies too.")

# == stage two: estimate the feedback and take the residual ==
# The model works in the POWER domain, because feedback is acoustic superposition
# and only power adds; the log domain does not.
#     mic_pow[t] ≈ direct_pow[t] + k · out_pow[t−d] + floor
# The question is whether the microphone's energy can be explained by the output:
#     it can    -> direct is about 0 -> it came back from the speakers
#     it cannot -> direct is above 0 -> there is an external source, that is, he is sounding
#
# **How k is estimated without labels** (the G35 lesson: labels cut by level cannot
# then be used to set a level threshold): direct power is always at least 0, so
# feedback is a LOWER BOUND on the microphone's energy. k is therefore estimated by
# LOW-QUANTILE REGRESSION, where the fit is decided by the frames in which he is not
# sounding, with nobody having to label them. Least squares would be dragged up by
# the frames he is singing (and bank_live's notes are triggered by his f0, so the
# output's starts and stops correlate with his phrases, which is the residue of
# G34's self-similarity in this architecture); quantile regression is not.
TAU = 0.10
mp = sm(mic) ** 2
op = np.concatenate([np.zeros(dbest), sm(out) ** 2])[:m]
ok = (op > np.percentile(op, 40)) & (np.arange(m) >= edge)

ks = np.linspace(0, 2.0, 401)[1:]
loss = [float(np.mean(np.maximum(TAU * (mp[ok] - k * op[ok]),
                                 (TAU - 1) * (mp[ok] - k * op[ok])))) for k in ks]
k = float(ks[int(np.argmin(loss))])
direct = np.maximum(mp - k * op, 0.0)
frac = direct / np.maximum(mp, 1e-20)

print(f"\n-- feedback level ({TAU:.0%} quantile regression, no labels) --")
print(f"coupling k={k:.3f} ({k:.1%} of the output power returns to the microphone)"
      f" = {10 * np.log10(k + 1e-12):+.1f} dB")
print("distribution of the share of microphone energy that cannot be explained (1.0 = entirely an external source, 0 = all feedback):")
for p in (5, 10, 25, 50, 75, 90, 95):
    print(f"   {p:2d}th percentile  {np.percentile(frac[ok], p):.3f}")
print(f"   frames under 10% unexplained (pure feedback): {float((frac[ok] < 0.1).mean()):.1%}")
print(f"   frames over 50% unexplained (an external source): {float((frac[ok] > 0.5).mean()):.1%}")
print(f"   note: the first of those roughly equalling TAU={TAU} is a CONSEQUENCE OF THE "
      "DEFINITION - quantile regression puts about tau of the frames under the line - "
      "not an independent finding.")
# -- whether k can be trusted depends on whether the material contains stretches
# where the speakers are loud and he is not sounding --
# This is exactly what the corrections to G34 and G35 fell on: the earlier material
# had only 5.5 s of such stretches, so k could not be calibrated at all.
# The cross-check uses the npz `f0`, that is detection BEFORE the gate, untouched by
# the already-condemned camera gate: if the coupling really is only -20 dB, feedback
# cannot make f0 fire, so "the banks are playing AND f0 is silent" is a trustworthy
# "he is not sounding" stretch, and it does not cut labels by microphone level.
f0 = st["f0"]
vn = st["vn"]
nfr = len(f0)
idx = np.minimum((np.arange(m) * nfr // m), nfr - 1)   # envelope frame to npz frame
playing = (vn[:, :] >= 0).any(1)[idx]                  # any part is playing from the banks
silent = (f0[idx] <= 0)                                # his pitch is not detected
cand = playing & silent & ok

runs, cur = [], 0
for v in cand:
    if v:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs = np.array(runs) * HOP / SR

print(f"\n-- is there enough calibration material? (cross-check, using npz f0 rather than microphone level) --")
print(f"frames with the banks playing {float(playing[ok].mean()):.1%}; "
      f"of those, f0 silent, that is candidates for him not sounding, {float(cand.sum()) * HOP / SR:.1f}s "
      f"（{float(cand.mean()):.1%} of {m * HOP / SR:.0f}s）")
if len(runs):
    print(f"{len(runs)} continuous stretches, longest {runs.max():.1f}s, "
          f"median {np.median(runs):.2f}s, {int((runs >= 1).sum())} of them at least 1s"
          f" ({runs[runs >= 1].sum():.1f}s in total)")
    print(f"   for comparison, the material behind G34 and G35 had only 5.5s, so k could not be calibrated")
    kk = float(np.median(mp[cand] / np.maximum(op[cand], 1e-20))) if cand.sum() else float("nan")
    print(f"   estimating k from those frames alone gives {kk:.4f} "
          f"({10 * np.log10(kk + 1e-12):+.1f} dB)，"
          f"against {k:.3f} from the quantile regression ({10 * np.log10(k + 1e-12):+.1f} dB)")
else:
    print("   not one stretch, so k cannot be calibrated: the same predicament as G34 and G35")

# -- the decisive check: are there any CLEAN calibration frames? --
# Hypothesis for the 10 dB gap between the two methods: the candidate stretches have
# a median length of 0.10 s, shorter than the 200 ms smoothing window, so smoothing
# smears his own voice into the candidate frames and k is overestimated. And f0
# dropping out does not mean he really stopped - breaths, gaps between words and
# reverb at the end of a phrase all drop it. So:
#   1. erode a GUARD band from each end and keep only the middle of each stretch
#   2. shrink the envelope window to SHORT, shorter than the eroded stretch
# If erosion leaves almost nothing, this material contains NO moment usable for
# calibration, which is the prescription written into the corrections to G34 and
# G35: clean calibration material has to be recorded separately, and k cannot be
# computed here.
GUARD, SHORT = 0.15, 0.05
g = max(1, int(round(GUARD * SR / HOP)))
ws = max(1, int(round(SHORT * SR / HOP)))
ero = cand.copy()
for s in range(1, g + 1):                  # binary erosion: shrink GUARD from each end
    ero &= np.concatenate([cand[s:], np.zeros(s, bool)])
    ero &= np.concatenate([np.zeros(s, bool), cand[:-s]])

sms = lambda x: np.convolve(x, np.ones(ws) / ws, "same")
mp_s = sms(mic) ** 2
op_s = np.concatenate([np.zeros(dbest), sms(out) ** 2])[:m]

print(f"\n-- decisive check: guard band {GUARD * 1000:.0f}ms, envelope window {SHORT * 1000:.0f}ms --")
print(f"{int(ero.sum())} frames survive erosion = {ero.sum() * HOP / SR:.2f}s"
      f" (from {cand.sum() * HOP / SR:.1f}s before erosion)")
if ero.sum() >= 30:
    kc = float(np.median(mp_s[ero] / np.maximum(op_s[ero], 1e-20)))
    q1 = float(np.percentile(mp_s[ero] / np.maximum(op_s[ero], 1e-20), 25))
    q3 = float(np.percentile(mp_s[ero] / np.maximum(op_s[ero], 1e-20), 75))
    print(f"clean frames give k = {kc:.4f} ({10 * np.log10(kc + 1e-12):+.1f} dB), "
          f"quartiles {10 * np.log10(q1 + 1e-12):+.1f} to "
          f"{10 * np.log10(q3 + 1e-12):+.1f} dB")
    print(f"three methods: quantile regression {10 * np.log10(k + 1e-12):+.1f} dB / "
          f"uneroded candidates {10 * np.log10(kk + 1e-12):+.1f} dB / "
          f"clean frames {10 * np.log10(kc + 1e-12):+.1f} dB")
    print("  -> agreement within 3 dB means k is trustworthy and discrimination is worth "
          "attempting; divergence means the material cannot calibrate k.")
else:
    print("-> fewer than 30 clean frames. This material contains no usable moment where")
    print("  he is definitely not sounding while the speakers are, so k cannot be calibrated")
    print("  here, which matches the diagnosis in the corrections to G34 and G35.")
    print("  The prescription, written down long ago: record 30 to 60 s of clean calibration")
    print("  material with him silent, the banks at performance level, same microphone and placement.")

# -- finally: separation, measured on G34's own scale, where more than 2x is viable --
# Neither group is cut by microphone level (the G35 lesson):
#   not sounding = the clean eroded frames above; singing = f0 voiced and eroded the
#   same way.
# f0 holds up as a label here because the coupling is only about -15 dB, so the banks
# cannot make f0 fire (self-consistency check: clean frames have f0 = 0 by definition).
sings = (f0[idx] > 0) & ok
es = sings.copy()
for s in range(1, g + 1):
    es &= np.concatenate([sings[s:], np.zeros(s, bool)])
    es &= np.concatenate([np.zeros(s, bool), sings[:-s]])

pred = np.maximum(kc * op_s, 1e-20) if ero.sum() >= 30 else np.maximum(k * op_s, 1e-20)
ratio = mp_s / pred
rq = 10 * np.log10(ratio[ero] + 1e-12)
rs = 10 * np.log10(ratio[es] + 1e-12)

print(f"\n-- separation (G34's scale) --")
print(f"not sounding {int(ero.sum())} frames / singing {int(es.sum())} frames")
print(f"  not sounding, microphone over predicted feedback: median {np.median(rq):+.1f} dB  "
      f"(25–75% {np.percentile(rq, 25):+.1f} … {np.percentile(rq, 75):+.1f})")
print(f"  singing,     microphone over predicted feedback: median {np.median(rs):+.1f} dB  "
      f"(25–75% {np.percentile(rs, 25):+.1f} … {np.percentile(rs, 75):+.1f})")
sep = 10 ** ((np.median(rs) - np.median(rq)) / 10)
print(f"  median separation {np.median(rs) - np.median(rq):+.1f} dB = {sep:.2f}x"
      f"   (G34's best was 1.68x; more than 2x is viable)")

# sweep the threshold for the real cost, which is what decides whether this can go on stage
print("\n  threshold dB   singing caught   not-sounding wrongly blocked")
best_t = None
for t in np.arange(-6, 25, 2.0):
    tp = float((rs > t).mean())
    fp = float((rq > t).mean())
    if best_t is None or (tp - fp) > best_t[0]:
        best_t = (tp - fp, t, tp, fp)
    print(f"   {t:+5.0f}    {tp:7.1%}    {fp:9.1%}")
_, t, tp, fp = best_t
print(f"\nbest threshold {t:+.0f}dB: singing caught {tp:.1%}, feedback mistaken for singing {fp:.1%}")
print(f"overlap {1 - (tp - fp):.1%} - the gate needs a binary answer to 'is he sounding',")
print("and this overlap is how often it gets that wrong.")

# -- control: use the k measured on the separate calibration material rather than
# estimated from this contaminated recording --
# `fb_calib_ana.py` measured k(f) on CALIB as -21.8, -21.6, -23.2, -20.1 and
# -20.7 dB, spanning only 3.1 dB from 80 Hz to 12 kHz, which means **the coupling is
# in fact flat**, with a wideband k of 0.133 (-17.5 dB). That overturns the
# hypothesis that the 13 dB spread came from frequency-dependent coupling, and it
# gives an independently calibrated k which falls exactly between this recording's
# two label-free estimates (-20.0 and -14.8 dB), so all three agree.
# Note that only k transfers, not the delay: CALIB's 547.6 ms is measured from the
# playback file's timeline to the microphone, a different definition from the dump's
# generation timeline, so the delay stays the 279 ms measured here.
K_CALIB = 0.133 ** 2               # the calibration prints 0.133 as an AMPLITUDE ratio
                                   # (20log10 = -17.5 dB); mp_s and op_s here are both
                                   # power, so the coefficient has to be squared to match.
rq2 = 10 * np.log10(r2[ero] + 1e-12)
rs2 = 10 * np.log10(r2[es] + 1e-12)
print(f"\n-- control: using the calibrated k={K_CALIB} ({10 * np.log10(K_CALIB):+.1f} dB) --")
print(f"  not sounding median {np.median(rq2):+.1f} dB (25-75% {np.percentile(rq2, 25):+.1f} to "
      f"{np.percentile(rq2, 75):+.1f})")
print(f"  singing     median {np.median(rs2):+.1f} dB (25-75% {np.percentile(rs2, 25):+.1f} to "
      f"{np.percentile(rs2, 75):+.1f})")
print(f"  median separation {np.median(rs2) - np.median(rq2):+.1f} dB = "
      f"{10 ** ((np.median(rs2) - np.median(rq2)) / 10):.2f}x (k only shifts; the separation is unchanged)")
bt2 = max(((float((rs2 > t).mean()) - float((rq2 > t).mean())), t,
           float((rs2 > t).mean()), float((rq2 > t).mean()))
          for t in np.arange(-12, 25, 1.0))
print(f"  best threshold {bt2[1]:+.0f}dB: singing caught {bt2[2]:.1%}, mistaken for sounding {bt2[3]:.1%}")

# -- where the spread really comes from: is the feedback buried in the noise floor? --
# On CALIB the feedback RMS was 0.00681 against a noise floor of 0.000581, an SNR of
# 21.4 dB, at performance level. If the banks were quieter in this dump, k times the
# output falls near the microphone's noise floor and the ratio jumps around on its
# own - in which case the 13 dB spread is insufficient measurement SNR rather than
# unstable coupling, and the two have entirely different prescriptions.
floor = float(np.percentile(mp_s, 2) ** 0.5)
fb_rms = np.sqrt(K_CALIB * op_s[ero])
print(f"\n-- is the 13 dB spread unstable coupling, or insufficient SNR? --")
print(f"  microphone noise floor estimate (2nd percentile) {20 * np.log10(floor + 1e-12):+.1f} dBFS")
print(f"  predicted feedback RMS over the clean frames: median {20 * np.log10(np.median(fb_rms) + 1e-12):+.1f} dBFS"
      f"（{np.percentile(20 * np.log10(fb_rms + 1e-12), 10):+.1f} … "
      f"{np.percentile(20 * np.log10(fb_rms + 1e-12), 90):+.1f}）")
print(f"  -> feedback above the noise floor, median SNR {20 * np.log10(np.median(fb_rms) / (floor + 1e-12)):+.1f} dB"
      f" (CALIB at performance level gave +21.4 dB)")
print(f"  clean frames below +6dB SNR: {float((fb_rms < floor * 2).mean()):.1%}"
      "  <- the ratio cannot possibly be stable on those")
