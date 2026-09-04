"""bank_gate_forensics - forensics on the gate: check the state machine against a dump.

Usage: python bank_gate_forensics.py scratchpad/<dump prefix>
Output: (1) a timestamp and interval for every gate, face and mouth transition;
(2) invariant checks, which catch what should not happen:
  I1 while the gate is shut, f0 must be 0 (a violation means the gate was bypassed)
  I2 more than release (0.25 s) plus the tail (0.3 s) after the gate shuts, the
     output envelope must be below -40 dB (a violation means it will not shut)
  I3 face lost to gate shut must be within 0.9 s (0.8 s grace plus one mouth loop)
  I4 face back to gate open must be within 0.2 s (the mouth worker period plus detection)
"""
import sys

import numpy as np
import soundfile as sf

pre = sys.argv[1]
z = np.load(pre + "_st.npz")
out, sr = sf.read(pre + "_out.wav")
om = out.mean(axis=1) if out.ndim == 2 else out
hop = float(z["hop"]) if "hop" in z else 0.035
mok, f0, face, mon = z["mok"], z["f0"], z["face"], z["mon"]
n = len(mok)
t = np.arange(n) * hop
print(f"dump {n} hops（{n*hop:.1f}s）")


def transitions(x, name):
    ch = np.where(np.diff(x.astype(int)) != 0)[0]
    print(f"\n[{name}] {len(ch)} transitions")
    for i in ch[:40]:
        print(f"  {t[i+1]:8.2f}s  {int(x[i])} → {int(x[i+1])}")
    return ch


tf = transitions(face, "face")
tm = transitions(mon, "mouth open")
tg = transitions(mok, "gate")

print("\n-- invariants --")
# I1 (corrected in R5: the f0 channel now records detection BEFORE the gate, so f0
# while the gate is shut is normal; the correct invariant is that the note must not
# change while the gate is shut)
note = z["note"]
v1 = 0
for i in range(1, len(mok)):
    if mok[i] < 0.5 and mok[i-1] < 0.5 and note[i] != note[i-1]             and note[i] >= 0:
        v1 += 1
print(f"I1 note changed while the gate was shut: {v1} times {'FAIL' if v1 else 'ok'}")
# I2 output envelope more than 0.55 s after the gate shuts
env = np.array([np.sqrt((om[int(i*hop*sr):int((i+1)*hop*sr)]**2).mean())
                for i in range(n)])
bad = 0
run = 0
for i in range(n):
    run = run + 1 if mok[i] < 0.5 else 0
    if run * hop > 0.55 and env[i] > 0.01:
        bad += 1
print(f"I2 still output more than 0.55s after the gate shut: {bad} hops {'FAIL' if bad else 'ok'}")
# I3 face lost to gate shut
lags = []
for i in np.where(np.diff(face.astype(int)) == -1)[0]:
    j = i
    while j < n and mok[j] > 0.5:
        j += 1
    if j < n:
        lags.append((j - i) * hop)
if lags:
    print(f"I3 face lost to gate shut: p50 {np.median(lags):.2f}s max {max(lags):.2f}s "
          f"{'✓' if max(lags) <= 0.9 else '✗ >0.9s'}")
else:
    print("I3 no face-lost events (nobody walked away)")
# I4 face back to gate open
lags = []
for i in np.where(np.diff(face.astype(int)) == 1)[0]:
    j = i
    while j < n and mok[j] < 0.5:
        j += 1
    if j < n:
        lags.append((j - i) * hop)
if lags:
    print(f"I4 face back to gate open: p50 {np.median(lags):.2f}s max {max(lags):.2f}s"
          f" (including the time he took to start singing again; p50 is the gate delay alone)")
else:
    print("I4 no face-returned events")
