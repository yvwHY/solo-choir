"""bank_live — the sampled choir (stage 1, line A of the two-track start of
2026-08-11)

The principle generalises v7.1's "the score is fixed, so do not render" into
**the pitch space is fixed, so do not render live**: the possible outputs of the
parts singing "ah" in chromatic harmony are a finite set, so at start-up each
part's whole range is pre-rendered semitone by semitone into loopable banks, at
the reference quality of reflow and cached on disk. Live then only:
  detects their note (autocorrelation over a window of about 46 ms), quantises to
  semitones with hysteresis, looks up and triggers three parts, loops them with a
  30 ms crossfade on a note change, applies attack and release envelopes, and
  mixes.
There is no machine learning in the loop, so the latency is about 36 ms of
detection, measured, plus I/O, giving roughly **50-60 ms** at reference quality.

v10 (the stability review of 2026-08-12): two fresh reviewers, one on correctness
and concurrency and one on behaviour and symptoms, produced over 25 findings and
every severe one was fixed — the order of clipping outside the range (a restart
loop at 28.6 Hz), the complementary crossfade (a +6 dB overshoot), a refractory
clock that was never written (dead code from v9), v9's admission test for a new
pitch (a bass unison or a pad chord tone vetoed itself, which was the main cause
of the stumbling; it returns to v2, where a closed mouth kills it), pads being
brought into the rejection layer (the main cause of pads that would not stop),
the camera gate freeze fix plus 30 Hz, underrun fade-out, subharmonic protection,
cents decay in silence, unbinding the pre-empt from the key (a cold start), the
fast path filling in the history, and moving allocation out of the callback lock.
See the review file and the worklog for 2026-08-12.
Bench limitation (review A15): --file does not go through the mouth gate path, so
everything passing offline does not mean the gate timing is sound. Any change to
the gate must be verified live.

v17 (the continuous vowel field of 2026-08-13, settled from first principles):
the whole paradigm of reactive discrete layer selection — v16 tilt, then 16.1 lip
distance, then 16.2 a voice fingerprint, then 16.3 a two-dimensional nearest
neighbour on mouth shape — was abandoned. A mirror is not a partner, and it is
always half a vowel late. A vowel is a continuous shape of the mouth and not a
symbol: mediapipe's continuous mouth (height, width) values feed inverse-distance
weights over five vowel anchors, and five loop layers are mixed continuously, per
hop and per sample. Because the layers share a midi and are the same length, the
samples align and there is no switching event at all, and the whole class of
illnesses — classifier, hysteresis, dwell — disappears. Known and unverified:
mixing across layers may add a slight chorus, since the same seed at the same f0
should be close; to be judged by ear.

The parts (four voices from v42; the numbers in VOICES are authoritative):
  bass1 at -8 (the C3 region, settled 2026-08-16); tenor from male8 spk7;
  alto3-40k (spk2) at +12; sop3 at +12 and then a diatonic third above, with
  KeyTracker on auto, the one verified in respond2 on 2026-08-04.

Chord locking (v6, item 1) comes free: their tuning deviation, the cents
difference from the quantised note smoothed by an EMA, multiplies the playback
rate of every part, so the ensemble bends with their tuning and the beating
disappears. --lock 0 turns it off.

The level follows voiced-or-not plus an envelope rather than the microphone
level; the lesson of 2026-08-11 is that a USB plug-and-play input at about
-61 dBFS makes the level untrustworthy, while autocorrelation detection is
indifferent to it.

v5 (the night of 2026-08-11, worked through independently): --file gives an
offline bench along the same per-hop pipeline, so offline conclusions hold live,
plus rejection of notes already sounding, which is the second line of defence
against feedback: a detected pitch sitting on a sounding note more than six
semitones outside their range is the alto or soprano coming back from the
speakers and is blocked, while a bass unison is left to the mouth gate. Bench
verification: the sung passage at rms 0.062 stays voiced, and a simulated alto
feedback passage at 0.015 no longer self-oscillates. The algorithmic latency
measured from a synthetic attack is 36 ms, giving about 55-60 ms end to end with
I/O.

v1 measured bare (2026-08-11): feedback proved outright. As soon as they stop,
the parts coming from the speakers, being perfectly periodic, are taken by the
detector for the singer and it sings on by itself. v2 added the **mouth gate**,
the same as score_live v3.1: mediapipe mouth opening with hysteresis and a 0.8 s
grace period, so a closed mouth means the pitch is not theirs. Feedback cannot
fool the lips. If the camera dies, or with --mouth 0, it falls back to running
bare, which is fine on headphones.

The gate state machine (specified explicitly in v14.2; the forensic script
bank_gate_forensics.py checks a dump against it):
  Signals: face, meaning mediapipe found a face this frame; mouth_on, meaning the
  opening passed the threshold, with hysteresis at 0.015 to open and 0.008 to
  close; hb, the camera worker heartbeat, alive if under 1 s old.
  States, that is, whether f0 counts as theirs:
    S1 singing available  face and mouth_on          -> f0 valid (last_open refreshed)
    S2 grace              last_open under 0.8 s ago  -> f0 valid (a consonant closing
                                                       the lips, or looking down,
                                                       does not break the line)
    S3 closed             last_open 0.8 s or older,  -> f0 forced to 0 (immune to
                          including walking away        feedback and room sound)
    S4 camera dead        hb absent for over 1 s     -> run bare (all f0 accepted)
                                                       plus a warning in the terminal
  Transition time constants: S1 to S3 is exactly 0.8 s; S3 to S1 is at most one
  mouth worker period, about 33 ms, plus the detection window; entering and
  leaving S4 each print one line. Design decision (2026-08-12): no face means the
  gate closes, rather than failing open.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python bank_live.py \
      [--in-name "USB PnP"] [--lock 0.5] [--gain 1.0]
"""
import argparse
import hashlib
import os
import signal
import threading
import time

# When started in the background with `&` from a non-interactive shell, SIGINT is
# inherited as SIG_IGN, so CPython never installs a KeyboardInterrupt handler and
# Ctrl-C and kill -INT are both deaf, taking the closing dump with them (cases r3
# and r4 of the judgement session on 2026-08-12). It is reinstalled here
# unconditionally, so the dump is always reached.
signal.signal(signal.SIGINT, signal.default_int_handler)

import numpy as np
import soundfile as sf

SR, HOP = 44100, 512
MAJ = [0, 2, 4, 5, 7, 9, 11]

ap = argparse.ArgumentParser()
ap.add_argument("--in-name", default="USB PnP")
# The default is a macOS device name on a zh-Hant system: a match target, not
# prose, so it is not translated.
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")
ap.add_argument("--sblock", type=int, default=512)
ap.add_argument("--hop", type=float, default=0.035,
                help="detection and synthesis granularity in seconds. The window is twice the hop, so 35 ms gives a 70 ms window, which still covers "
                     "two periods at 65 Hz; a note change takes about 70 ms in total, against about 92 ms when the hop was 46 ms")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--out-map", default=None,
                help="one output channel per part, comma separated, in the current VOICES order (with --tenor 1 that is bass, tenor, alto, sop, "
                     "for example '0,1,2,3'). A whole part, the lead voice, the ensemble members and the pad, goes to one channel; the "
                     "reverb runs on a mono bus fed back into every channel. This routes only and never widens (G11), and needs a single "
                     "multi-channel device (F7). None leaves the original stereo pan path bit-identical")
ap.add_argument("--lock", type=float, default=0.5,
                help="how much the chord locks: 0 keeps the parts in equal temperament, 1 follows their cents exactly")
ap.add_argument("--attack", type=float, default=0.12, help="attack, in seconds")
ap.add_argument("--release", type=float, default=0.25, help="release, in seconds")
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--vowels", type=int, default=0,
                help="v17 continuous vowel field (0 is off, and the behaviour is completely unchanged): five extra vowel-layer banks are rendered "
                     "per part, and live, mediapipe's continuous mouth-shape values give inverse-distance weights over the five anchors "
                     "and the five layers are mixed per hop, so the parts follow the mouth continuously. Setting this flag uses a "
                     "separate bank_ cache directory")
ap.add_argument("--layers", default="",
                help="v20: keep only some vowel layers, by comma-separated index; empty keeps all. From v22 the indices are 0 for eh, 1 for ee, "
                     "2 for oo and 3 for ah; oh could not be rendered and was removed. From 2026-08-13: the sound changing back and forth "
                     "to match the mouth shape does not sound good, so fewer vowels mean fewer changes. For example, --layers 1,3 keeps "
                     "only ee and ah")
ap.add_argument("--vw-tau", type=float, default=0.0,
                help="v20 weight smoothing time constant in seconds (0 is off, following the mouth as fast as it moves). If the timbre still "
                     "seems to wander after reducing the layers, set this to 0.15-0.3")
ap.add_argument("--young", type=float, default=0.0,
                help="v26 young-note protection, in seconds (0 is off, the old behaviour): for this long after a note change, moving away needs "
                     "--young-need more votes and cannot use the single-vote pre-empt channel. Bouncing only happens while a note is very "
                     "young, and once it is held steadily nothing is affected, so this cuts the flapping without costing responsiveness. "
                     "It replaces the v24 pitch dead zone, which three reviews condemned for leaking notes in silence, sticking on a note "
                     "for good, and driving the glide rate up")
ap.add_argument("--young-need", type=int, default=2,
                help="how many extra votes a young note needs before it may change, used with --young")
ap.add_argument("--deadzone", type=float, default=0.0,
                help="v26 pitch dead zone in semitones (0 is off): anything within 0.5 plus this of the current note counts as still being on that "
                     "note, clearing the candidate votes. **Capped at 0.5**, so the escape threshold stays at or below 1.0 semitone, the "
                     "centre of the neighbouring note, which guarantees it is reachable and avoids v24 sticking on a note or leaking one. "
                     "What it blocks is slow drift across a grid line, the kind votes cannot stop")
ap.add_argument("--maxlag", type=int, default=3,
                help="v25 output backlog cap (blocks; 1 block = 35ms). 3 = "
                     "old behaviour: one worker stall permanently adds up to "
                     "105ms and xrun cannot see it. 1 = latency clamped to "
                     "35ms, at the cost of a 5ms fade-in when a block drops")
ap.add_argument("--rebound", type=float, default=0.25,
                help="v24 back-jump guard window in seconds: returning to the previous note within this time counts as vibrato chatter and needs "
                     "four votes, with the pre-empt blocked. Review A4: raising it to 0.6 adds about 70 ms of delay to returning to the "
                     "previous note after a breath, and covers a neighbouring-note figure of quavers at 100 bpm, so the default stays at "
                     "0.25 and any change should be measured on the bench first")
ap.add_argument("--rebuild", action="store_true", help="force the banks to be re-rendered")
ap.add_argument("--mouth", type=int, default=1, help="mouth gate (0 is off)")
ap.add_argument("--mouth-open", type=float, default=0.015,
                help="v27 opening threshold, the inner lip distance over the face height: the mouth must open this far to count as singing. From "
                     "2026-08-13, a very slight opening triggered it, so raise this. It used to be hard-coded at 0.015")
ap.add_argument("--mouth-close", type=float, default=0.008,
                help="v27 closing threshold, the lower edge of the hysteresis, which must be below --mouth-open. It used to be hard-coded at 0.008")
ap.add_argument("--min-level", type=float, default=-100.0,
                help="v33 detection level threshold in dBFS (-100 is off, the old behaviour). Autocorrelation measures periodicity only and is "
                     "indifferent to level: measured from a dump on 2026-08-13, with the mouth closed and the microphone at only "
                     "-54 dBFS, 43%% of frames still reported a pitch, so as soon as the gate entered its grace period the room noise "
                     "set the parts singing. They sing at -35 dBFS, so -45 separates the two cleanly. (-100 is a leftover from the era "
                     "of 2026-08-11 when the microphone was broken at -61 dBFS; once it was repaired it became a hole.)")
ap.add_argument("--bleed", type=float, default=0.0,
                help="v33 coupling from the parts back into the microphone, in dB and negative (0 is off); measured at -18. Detection requires the "
                     "microphone to exceed the parts' current output plus this coupling by a further --bleed-margin dB before it counts "
                     "as singing, so bleed from the speakers cannot trigger it while the mouth is open but silent")
ap.add_argument("--bleed-margin", type=float, default=6.0,
                help="how many dB above the bleed it must be to count")
ap.add_argument("--balance", type=int, default=0,
                help="v32: level the parts automatically from the measured loudness of their banks (0 is off). The render gain that was set does "
                     "not match the real output, because each voice is naturally a different loudness: measured 2026-08-13, the soprano "
                     "was 7.0 dB below the bass, the tenor 3.8 dB below and the alto 0.5 dB above")
ap.add_argument("--trim", default="",
                help="v32 per-part trim in dB, comma separated, such as sop=+2,bass=-1. This is the layer the ear decides, laid over --balance")
ap.add_argument("--tenor", type=int, default=0,
                help="v31: add a tenor part; 0 is off and returns to three parts. From v42 the lead voice is reflow-male8 spk7 and no longer the "
                     "same voice as the bass, because bass1 is broken in the 65-68 range. Its range is 43-68 with a shift of +6; under "
                     "--vl 2 the shift of an upper part is only read as a fallback, since the actual note is decided by the voice "
                     "leading")
ap.add_argument("--gate-log", type=int, default=0,
                help="v34: print one line every time the gate opens or closes, including the current probability, so a false trigger can be located "
                     "afterwards rather than recalled")
ap.add_argument("--mic-gate", type=int, default=0,
                help="v43 microphone gate (0 is off; from 2026-08-18, live should be triggerable by the microphone alone): gate authority moves from "
                     "the camera to the microphone level. The hop RMS must beat max(--mic-open, the output level over the last 8 hops "
                     "plus --bleed plus --bleed-margin) for --mic-frames consecutive hops before the gate opens, and it closes after "
                     "--mic-hold seconds below the floor. While it is on, the camera still runs but only as a picture (view and "
                     "frame-b64); the watchdog and the sing gate no longer govern the gate. Unlike the camera gate, this also applies "
                     "under --file, so the gate can be verified offline. Honest boundary (F26): in the energy domain it catches real "
                     "singing 89%% of the time and mistakes feedback for singing 37%% of the time, so the decision rests on listening")
ap.add_argument("--mic-open", type=float, default=-45.0,
                help="absolute opening level of the microphone gate in dBFS, on the hop RMS. The noise floor with the mouth closed measures -54 and "
                     "--min-level is frozen at -50, so the default leaves a further 5 dB of margin")
ap.add_argument("--mic-frames", type=int, default=2,
                help="opening requires N consecutive hops above the floor (2 x 35 ms = 70 ms), so a single-hop transient such as clothing rustle or "
                     "a click does not open it; the same meaning as --bs-frames")
ap.add_argument("--mic-hold", type=float, default=0.8,
                help="how long below the floor before the gate closes, in seconds, so the brief dips between words and on consonants do not cut the "
                     "phrase. The value is taken from the camera gate's 0.8 s grace period. Internally it is counted in hops, so the "
                     "--file bench is unaffected by wall-clock time")
ap.add_argument("--sing-gate", type=int, default=0,
                help="v34: use a trained 'is this person singing' model as the gate (53 parameters, produced by "
                     "scratchpad/sing_gate_train.py). Hand-picked coefficients were whack-a-mole, one vowel at a time: jawOpen misses ee, "
                     "since the jaw does not open, and pucker misses ah. Measured probabilities across seven states: mouth closed 0.000, "
                     "slightly open 0.019, speaking 0.008, against singing ah 0.986, ee 1.000, oo 1.000 and eh 0.996, so at a threshold of "
                     "0.30 singing passes 100%% of the time with 0.7%% false triggers")
ap.add_argument("--gate-model", default="auto",
                help="v40 gate model path; auto uses sing_gate_model3.npz if it exists, which was trained across three sessions with self-calibration "
                     "and a movement veto, and otherwise falls back to sing_gate_model, the single session of 2026-08-13. To force the "
                     "older one, give its path explicitly")
ap.add_argument("--gate-calib", type=float, default=5.0,
                help="v40 seconds of baseline self-calibration at the start (**keep the mouth closed** during it; the gate stays shut). This corrects "
                     "session drift: the same person making the same movement measured a resting jawOpen of 0.031 on 2026-08-13 and "
                     "0.017 on 2026-08-15, which gave 44%% false triggers with the mouth closed. Only models on the new contract "
                     "(calib=1) use it")
ap.add_argument("--gate-motion-th", type=float, default=-1.0,
                help="v40 movement veto threshold, the mean |delta jawOpen| over a 0.5 s window; below 0 uses the model's built-in value of 0.0025. "
                     "Speaking measures 0.0079-0.0173 and singing peaks at 0.0023. Lowering it blocks speech better but drops more "
                     "singing: at 0.0020, speech triggered 0%% of the time and 17.5%% of singing was missed")
ap.add_argument("--sing-open", type=float, default=0.7,
                help="v34 threshold to start singing, as a probability")
ap.add_argument("--sing-hold", type=float, default=0.3,
                help="v34 threshold to keep singing, the lower edge of the hysteresis")
ap.add_argument("--bs-gate", type=int, default=0,
                help="v30: use mediapipe blendshapes as the gate (0 is off and uses hand-computed geometry). Measured across four states, "
                     "mouthPucker on oo reads 0.932 against 0.415 with the mouth closed or slightly open, a separation of 2.25, which is "
                     "the highest of all 52 blendshapes and beats the hand-computed mouth width at 2.07 and the opening at 1.39. At a "
                     "threshold of 0.60, oo passes 94%% of the time with 3%% false triggers. **Side effect: starting on oo can now open "
                     "the gate**, since pucker is in place before the mouth opens")
ap.add_argument("--bs-open", type=float, default=0.20,
                help="v30 jawOpen threshold to start singing (measured: slightly open 0.131, singing ah 0.343)")
ap.add_argument("--bs-close", type=float, default=0.08,
                help="v30 jawOpen threshold to keep singing, the lower edge of the hysteresis")
ap.add_argument("--bs-pucker", type=float, default=0.75,
                help="v30.1 pucker threshold. Set thresholds from **the tail, not the median**: with the mouth closed, pucker has a median of 0.493 "
                     "but a maximum of 0.682, so at a threshold of 0.60 some frame crosses it every second, and **one frame opens the "
                     "gate for 0.8 s**, which is why it still triggered with the mouth closed. At 0.70 and above, a closed or slightly "
                     "open mouth gives 0.0%% false triggers while oo still passes 80%% of the time")
ap.add_argument("--bs-frames", type=int, default=2,
                help="v30.1 opening debounce: this many consecutive frames must hold before the gate opens. A single noisy frame opening it for 0.8 s "
                     "was the failure mode of the previous version. Holding the gate open is unaffected")
ap.add_argument("--mouth-narrow", type=float, default=0.0,
                help="v28 pucker hold, the mouth width over the face width (0 is off): **while already singing**, the gate stays open as long as the "
                     "mouth is narrower than this. Singing oo gives an opening of only 0.008, indistinguishable from 0.006 with the mouth "
                     "closed, but a mouth width of 0.279 against 0.341 closed, which is 18%% narrower and the only dimension that "
                     "separates them (measured across three states on 2026-08-13). Opening the gate still looks only at the opening, so "
                     "this adds no false-trigger risk. The cost is that **starting on oo still cannot open the gate**; the honest "
                     "boundary is that the camera cannot see the difference between a pucker and a closed mouth, only between wide and "
                     "narrow")
ap.add_argument("--view", type=int, default=1,
                help="mouth monitor window: the camera, the inner lip line, the opening and the gate state (0 is off)")
ap.add_argument("--cam", type=int, default=-1, help="camera index (-1 selects automatically)")
ap.add_argument("--vl", type=int, default=1,
                # v36: 2 is the newer choral arrangement, with the ranges pulled
                # towards the centre, unisons forbidden and crossing forbidden

                help="v4 voice leading: the alto and soprano each take the chord tone nearest their previous note, from candidates a third, a fifth "
                     "or an octave away, so the parts move in small steps. 0 is off and they jump in parallel at a fixed +12 or a "
                     "third")
ap.add_argument("--mem-real", type=int, default=0,
                help="v44: whether ensemble members may borrow a real singer from another model (0 always copies the lead, the v38 behaviour). "
                     "**The default changed to 0 after the judgement of 2026-08-16**: laying the same note from two different recordings "
                     "on top of each other always leaves a slight pitch difference, which is the intermittent noise, an interference that "
                     "all five instruments are blind to, and musically nothing is lost, since real timbres and detuned copies were hard "
                     "to tell apart by ear. The frozen configuration is also 0. Trying real timbres requires writing 1 explicitly and "
                     "re-verifying the MEM_SPK list note by note first")
ap.add_argument("--vib-sync", type=int, default=0,
                help="v43: ensemble members share the lead's vibrato phase (1 synchronises them). 0 gives each a random phase, the original behaviour, "
                     "so copies of the same note have instantaneous frequencies offset from each other and, on the high harmonics, fall "
                     "into the 15-300 Hz difference band, which is roughness: audible but invisible to a spectrum, because it is not a new "
                     "component but interference")
ap.add_argument("--porta", type=float, default=0.08,
                help="v15 portamento: glide only when the previous note was held for 0.3 s or more and the step is 2 semitones or fewer, so a slow "
                     "transition is an expressive glide while normal singing changes notes cleanly. Linear and of fixed length, in "
                     "seconds (0 is off)")
ap.add_argument("--fold", type=int, default=1,
                help="v37: fold by octaves outside the range, keeping the pitch class, rather than clipping, which changes it")
ap.add_argument("--vib", type=float, default=0.0,
                help="v37 vibrato depth in cents (0 is off). The review measured the parts' energy in the 3-8 Hz band at only 2.2%% of theirs, that "
                     "is, no vibrato at all, which is the strongest fingerprint of a corrected sound. 20-30 is suggested")
ap.add_argument("--vib-rate", type=float, default=5.5, help="vibrato rate in Hz")
ap.add_argument("--vib-delay", type=float, default=0.35,
                help="the vibrato fades in only once a note is older than this, so a note change itself stays clean")
ap.add_argument("--human", type=float, default=6.0,
                help="v3 human dispersion: each part drifts slowly in tuning on its own, in plus or minus cents (0 is off), which kills the uniform "
                     "quality of the pad (v6, item 2, humanisation)")
ap.add_argument("--per-part", type=int, default=1,
                help="v38: how many singers per part (1 is the current state). The members **share one bank**: nothing is re-rendered and VOICES is "
                     "untouched, since the cache key is repr(VOICES) and changing it would re-render all fifteen layer banks. Only extra "
                     "playback cursors are opened. What differs is a static detuning, a staggered attack, each member's own slow drift "
                     "and vibrato phase, and a slightly scattered position. FINDINGS F3: about 7 decorrelated lines sound like 10 real "
                     "people, and more brings diminishing returns")
ap.add_argument("--spread-cents", type=float, default=25.0,
                help="v38 static detuning between members, as an std in cents. F3 puts the sweet spot at about 25; in a real ensemble the spread of "
                     "F0 between members is 0-50 cents with a mean of about 20")
ap.add_argument("--spread-ms", type=float, default=20.0,
                help="v38 stagger of the members' attacks in ms, uniform over 0 to twice this. F3 puts the sweet spot at about 20 ms and finds over "
                     "40 ms worse. It is also a delay line, reading the same loop at different phases, which does most of the "
                     "decorrelation; detuning alone only produces beating")
ap.add_argument("--spread-pan", type=float, default=0.18,
                help="v38: how far the members are scattered around the part's centre, plus or minus, in -1 to 1 coordinates")
ap.add_argument("--wet", type=float, default=0.3,
                help="v3 distance layer: reverb send, convolving the reverb.py IR in a stream (0 is off)")
ap.add_argument("--pad", type=float, default=1.5,
                help="v6 slow-layer chord pad, built from the banks with no machine learning: the anchor note must hold for at least this many "
                     "seconds before the chord may change; 0 is off, leaving only the fast layer")
ap.add_argument("--pad-gain", type=float, default=0.45, help="level of the slow layer")
ap.add_argument("--pad-hold", type=float, default=1.5,
                help="how long after you stop before the slow layer releases; the fast layer releases in 0.25 s, so the pad outlives you")
ap.add_argument("--predict", default="scratchpad/melody_lm_v0.json",
                help="v8 personalised pre-empt: a model of this singer's melodic habits (melody_lm). A note change within the model's top three "
                     "commits on one vote, which covers the habitual leaps outside stepwise motion. An empty string turns it off and "
                     "falls back to the stepwise shortcut")
ap.add_argument("--fast", type=int, default=1,
                help="v6 pre-empt: a stepwise change of 2 semitones or fewer that stays in key commits on one vote, about 35 ms; 0 is off and every "
                     "change needs two votes, about 70 ms")
ap.add_argument("--frame-b64", type=float, default=0,
                help="print the mouth image to stdout N frames per second as \'FRAME <base64 jpeg>\', for the respond_shell UI (0 is off and the "
                     "behaviour is unchanged)")
ap.add_argument("--dump", default="", help="record the microphone and the output to <path>_mic.wav and <path>_out.wav")
ap.add_argument("--dump-max-min", type=float, default=30.0,
                help="v44 ceiling on the dump, in minutes. The buffer used to be unbounded, holding about 32 MB per minute, and concatenating it at "
                     "the end of a long session either exhausted memory or overran the shell\'s 8 s SIGKILL budget, taking the whole "
                     "recording with it. Once full it stops recording and prints one line; the performance is unaffected")
ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"),
                help="offline bench: run a whole file through the same per-hop pipeline, with no audio device and no camera, for self-verification "
                     "during development")
a = ap.parse_args()
# Review B2: absurd flag values used to pass silently. With --hys 12, a measured
# 60 s produced only 7 note changes with a longest note of 18.45 s and the program
# said nothing. The range checks live here so it dies at start-up, rather than
# becoming "it sounds strange and nobody can find out why".
assert 0 <= a.young <= 2.0, "--young must be between 0 and 2 seconds"
# **0.5 is a hard ceiling, not a matter of taste**: the dead-zone escape threshold
# is 0.5 + deadzone, and above 0.5 it stops intersecting the plausibility gate,
# which allows at most 0.35 off the grid, so the note just left becomes
# unreachable (review S2 measured it stuck for 5 seconds). From 0.55 it also
# silently drops semitone steps (S3 measured 3 notes lost per pass). At 0.5 it
# already eats real notes, 4 of 41 stable plateaus, so the ceiling is 0.45.
assert 1 <= a.per_part <= 8, "--per-part must be between 1 and 8 (F3: beyond 7 lines the returns diminish)"
assert 0 <= a.spread_cents <= 60, "--spread-cents must be between 0 and 60 (F3: a real ensemble spans 0-50 cents)"
assert 0 <= a.spread_ms <= 40, "--spread-ms is capped at 40 (F3: staggering the attacks beyond 40 ms sounds worse)"
assert 0 <= a.spread_pan <= 0.5, "--spread-pan must be between 0 and 0.5"
assert 0 <= a.deadzone <= 0.45, "--deadzone is capped at 0.45 (see the note in the code)"
assert 0 <= a.young_need <= 6, "--young-need must be between 0 and 6 votes"
assert 0 <= a.rebound <= 2.0, "--rebound must be between 0 and 2 seconds"
assert 1 <= a.maxlag <= 16, "--maxlag must be between 1 and 16 blocks"
# Review #11, 2026-08-17: --attack or --release at 0 divides by zero in render's
# n/(atk*SR), which kills the worker thread and silences the output for good while
# every indicator stays green. This file's convention that 0 means off does not
# apply to these two: the envelope always exists and cannot be turned off.
assert a.attack > 0, "--attack must be above 0 (0 divides by zero and kills the audio worker)"
assert a.release > 0, "--release must be above 0 (0 divides by zero and kills the audio worker)"
# Review: --bs-frames 0 makes `arm >= 0` always true, so the gate is permanently
# open regardless of the probability.
assert 1 <= a.bs_frames <= 10, "--bs-frames must be at least 1 (0 leaves the gate permanently open)"
assert a.sing_hold <= a.sing_open, "--sing-hold must be at or below --sing-open (hysteresis)"
assert -120 <= a.min_level <= 0, "--min-level must be between -120 and 0 dBFS"   # -100 turns it off
assert 0 < a.mouth_close < a.mouth_open < 0.5, \
    "--mouth-close must be below --mouth-open (the two edges of the hysteresis)"

# A part is (name, model, spk, gain, bank MIDI range, semitone shift, whether to
# add a diatonic third).
# The cache key uses only the first five fields, the name, model, spk, gain and
# range (see _ckv), so changing the shift does not re-render.
# v41 (2026-08-16): the bass shift changed from 0 to **-8**. It was briefly
# reverted that day because of live noise, but the noise turned out to be caused by
# **--mem-real**, the interference of laying different recordings on top of each
# other (see that flag's help; --frame-b64 was wrongly blamed and has been
# cleared), and had nothing to do with the shift, so it went back in.
# The complaint was that the male resonance of the bass and tenor never came
# through. It was neither the timbre nor the level but **the range**: measured
# from the 2026-08-15 dump, the bass had a median of G#3 (56), **exactly the same
# as the singer** (both at 207.7 Hz), and the tenor A#3 (58). Two male parts two
# semitones apart are not two lines but one thickened line, and both sat in the
# middle register with nobody in the bass resonance region of G2 to C3.
# Why -8 and not -12: -12 was tried first and judged "there it is", but sweeping
# every shift showed that -12 puts only 54.8% inside the standard bass core of
# G2-C4 and drops 8.9% below the bank's lower limit, where it is folded by an
# octave. **-8 is the best**: 86.4% inside the core, 0.4% dropping out, and it
# never enters the broken 71-75 region. The low-frequency energy is almost the
# same either way (13.2% against 13.8% below 120 Hz), so -8 gets the same
# resonance without the octave-folding defect.
VOICES = [("bass", "reflow-bass1/model_32000.pt", 1, 1.0, (36, 68), -8, False),
          ("alto", "reflow-alto3/model_40000.pt", 2, 0.85, (50, 80), 12,
           False),
          ("sop", "reflow-sop3/model_20000.pt", 1, 0.7, (53, 84), 12, True)]
# v31, from the question "can another male part be added, is there only a bass at
# the moment?" — yes: of the three voices in service only the bass was male. There
# was no second male weight on hand (bass1 has n_spk=1, and only alto3 was trained
# jointly on three voices with a selectable spk_id), so the shortest path was to
# **open a second part on the same bass voice singing a different line**. A real
# choir already divides parts within one voice type, and with each part's own
# human drift (HUM) and position (PAN) they become two people. A genuinely new
# timbre would mean training M4Singer's Tenor-1 to 7 or Bass-2 and 3, none of
# which has been used, and that needs a GPU.
if a.tenor:
    # The upper limit is 68 and not 75: it was set in the bass1 era, where
    # rendering 71-75 was a broken region (at MIDI 75 the pitch within a loop had
    # an SD of 59.9 cents and a periodicity of 0.51, against 0.99 for a healthy
    # note, and the tenor once spent 37% of its time there). After v42 moved to
    # male8 spk7, 65-68 measures clean (periodicity 0.987-0.990), but **nothing
    # above 68 has been measured**. Raising the limit means sweeping note by note
    # first; do not simply inherit the old limit and do not widen it blindly.
    # v42 (2026-08-16): the lead voice moved from **bass1 to male8 spk7
    # (Tenor-5)**.
    # Live, the report was "noise on B3, none on C4, none on A3" together with "B3
    # is also fine sometimes". Sweeping the bank note by note found the cause:
    # **bass1 is broken at MIDI 65-68** (periodicity 0.790-0.881, pitch SD 14-25
    # cents, against 0.99 for a healthy note), and voice leading chooses which note
    # to take from the harmonic context, so the same sung note sometimes lands on a
    # broken one and sometimes does not, which is the intermittency.
    # Measured from a dump, the share of notes landing on broken ones was 18.04%
    # for the bass and 5.15% for the tenor in the old configuration, and after v41
    # it was 0% for the bass and 3.65% for the tenor. So v41 had already fixed the
    # bass, and what remained was the tenor.
    # Why change the model rather than narrow the range: lowering the upper limit
    # to 64 would also reach zero, at the cost of 3.65% of notes being folded by an
    # octave, which trades one defect for another. **male8 spk7 is clean over the
    # same range** (65-68, periodicity 0.987-0.990, SD 5.0-7.6 cents), and that
    # weight is already on disk, since the ensemble members have been using it, so
    # it costs nothing extra.
    # v31 used bass1 because there was no second male weight on hand; once male8
    # was trained on 2026-08-15 that premise no longer held, and this is the change
    # that did not follow at the time.
    VOICES.insert(1, ("tenor", "reflow-male8/model_40000.pt", 7, 0.8,
                      # v41 (2026-08-16): +7 became **+6**. The sweep showed +6
                      # puts the highest share inside the core E3-E4, 96.3%
                      # against 94.9% for +7, and the interval from the bass at -8
                      # is 14 semitones (corrected 2026-08-17; the original note
                      # of "+9" was an arithmetic error).
                      # Review #10, 2026-08-17: **under --vl 2 this shift is
                      # inert**. sh is read only for vi == 0, the bass, and as a
                      # fallback when the candidate set is empty; which note an
                      # upper part actually sings is decided by voice leading from
                      # the range and the scoring. So that 96.3% describes the
                      # sweeper's terms and not the current system. To move the
                      # tenor's register, the real control is the range (43, 68)
                      # and its centre.
                      (43, 68), 6, False))
# --out-map (2026-08-18: neural, live and live+neural should all be routable). It
# must be parsed after VOICES is settled, that is, after --tenor has inserted its
# part, so the lengths match. None skips every OMAP branch below and leaves the
# original path bit-identical (regime A, verified by comparing --file runs before
# and after).
OMAP = None
NCH = 2
if a.out_map:
    OMAP = [int(t) for t in str(a.out_map).split(",")]
    if len(OMAP) != len(VOICES) or min(OMAP) < 0:
        raise SystemExit(f"--out-map needs {len(VOICES)} non-negative channels, in the order "
                         f"{','.join(nm for nm, *_x in VOICES)}），"
                         f"got {a.out_map!r}")
    NCH = max(max(OMAP) + 1, 2)
NOTE_S = 2.0                             # seconds rendered per note
LOOP_A, LOOP_B = int(0.5 * SR), int(1.8 * SR)   # the loop region
XF = int(0.05 * SR)                      # crossfade at the loop wrap

# v16 multiple vowel layers, used only with --vowels: five layers from the
# 2026-06-19 material spanning dark to bright, approved by ear on 2026-08-13.
# Each is (tag, src_wav, t0, t1, src_tilt_dB). The src_tilt field has had no
# consumer since v17, when selecting a layer by tilt was abandoned, but the repr
# of the whole tuple enters the cache hash below (_ck), so changing any field
# re-renders all fifteen layer banks. Do not change them.
# This path's string enters the repr of VOWEL_LAYERS and then the cache hash in
# _ck below. Moving to another machine changes the cache key once and re-renders
# the layer banks once; the frozen configuration does not pass --vowels, so this
# does not affect a performance.
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import VOWEL_CLIPS as _VOWEL_CLIPS  # noqa: E402
_C0619 = str(_VOWEL_CLIPS)
# v21 (re-selected after the post-mortem of 2026-08-13): **chosen by vowel, no
# longer by brightness**. The older five layers were taken by sorting on spectral
# tilt, and measured they were [oo, oh, ee, oh, ah]: no eh at all, and oh took
# three of the five slots, so the labels "oh, eh, ee, oo, ah" were purely an
# assumption about the ordering. Three rounds of listening confirmed it item by
# item: ee and ah were right, eh could not be told apart, oo and oh were never
# reached, and oh sounded like eh. The new selection is vowel_layer_rebuild:
# measure F1 and F2 over 300 ms windows of the corpus, merge consecutive spans of
# the same vowel, and take the centre plus or minus 0.5 s of the most
# representative run of 0.7 s or longer. The rendering encoder takes plus or minus
# 0.93 s of context around the window centre, so a short window mixes in the
# neighbouring phoneme.
# The values in brackets are that span's measured (F1, F2); the order must match
# the calibration labels, so that layer i corresponds to class i.
# v22 changed the selection to **choosing after rendering**
#   (vowel_render_select): how close the source is turns out not to be the
#   criterion. An oo recorded live measured d = 0.11 at the source, the closest of
#   the whole session, and still came out wrong after rendering, while an ordinary
#   span of the corpus rendered cleanly. Which units survive the model is close to
#   a lottery, so every candidate is rendered and chosen on the rendered distance.
#   Survival rates, from 30-38 candidates per vowel: ah 33, eh 29, ee 29, oo 6,
#   and **oh 0, which is fatal** (even the best candidate rendered as oo, 381 of
#   943). This is a boundary of the model and not a problem of pronunciation, so
#   stop testing it by ear.
# The lesson of v23 ("this version is worse than the last", and all four layers
#   measured 1-6 dB down in harmonic-to-noise ratio on the spot): selecting on
#   formant distance alone optimised accuracy and paid for it in cleanliness; the
#   same clip differs by 6 dB with only the window position changed. The criterion
#   became **two-stage**: first require the vowel to be correct with d at or below
#   0.25, then take the highest harmonic-to-noise ratio. Two of the current four
#   layers come from the live recordings of 2026-08-13, the eh and the oo, and they
#   won on sound quality.
# The vowel tags below stay in Chinese on purpose: the repr of each tuple enters
# the cache hash (_ck), so changing one character would re-render all fifteen
# layer banks. They read oh, eh, ee, oo, ah.
VOWEL_LAYERS = [("喔", f"{_C0619}/clip_0022.wav", 1.75, 2.75, 0.0),   # 479/700
                ("欸", f"{_C0619}/clip_0143.wav", 1.67, 2.67, 0.0),   # 609/1749
                ("咿", f"{_C0619}/clip_0155.wav", 3.47, 4.47, 0.0),   # 264/2331
                ("嗚", f"{_C0619}/clip_0035.wav", 2.75, 3.75, 0.0),   # 320/794
                ("啊", f"{_C0619}/clip_0009.wav", 0.93, 1.93, 0.0)]   # 717/1101
# Judged by ear on 2026-08-13: **return to this v21 set**. The harmonic-to-noise
# measurement said the v23 set was 1-3 dB cleaner per layer, and this set was still
# preferred by ear. When the metric and the ear disagree, the ear decides and the
# measurement only guards against engineering regressions.
# The v23 set (eh, ee, oo, ah, two of them from live recordings) is preserved in
# commit 10bdf953 and can be restored.
# The classifier has five classes, in the same order as lip_ring_calib.LABELS in
# the calibration ritual. There may be fewer layers than classes, since oh cannot
# be rendered, so the layer-to-class mapping is written out explicitly rather than
# relying on the indices happening to match.
# Class labels, oh / eh / ee / oo / ah, kept in Chinese to match VOWEL_LAYERS.
VOWEL_NAMES = ["喔", "欸", "咿", "嗚", "啊"]
L_MID = 4                                # the fallback layer with no camera: ah, the same
                                         # as the reference bank
# v20 vowel subset (--layers): the model still produces five class probabilities,
# and taking a subset and renormalising gives the conditional probability of which
# of these vowels is being sung. Not one byte of the banks or the cache changes, so
# switching subsets needs no re-rendering.
_LSEL = list(range(len(VOWEL_LAYERS)))
if a.layers.strip():
    _LSEL = sorted({int(s) for s in a.layers.split(",") if s.strip() != ""})
    assert _LSEL and all(0 <= i < len(VOWEL_LAYERS) for i in _LSEL), _LSEL
_NSEL = len(_LSEL)
# v22: layer to classifier class. There are fewer layers than classes, since oh
# has no layer, so its probability is renormalised away.
_LCLS = [VOWEL_NAMES.index(t) for (t, *_r) in VOWEL_LAYERS]
assert len(set(_LCLS)) == len(_LCLS), "two layers map to the same vowel class"
_LDEF = _LSEL.index(L_MID) if L_MID in _LSEL else _NSEL // 2   # fallback layer
# v17: with no camera or no calibration file, the weights fall back to a one-hot
# vector on the default layer.
_VW_MID = np.zeros(_NSEL, dtype="float32")
_VW_MID[_LDEF] = 1.0


def _sub(p5):
    """The classifier's five class probabilities to weights over the layers in
    use: take the corresponding classes and renormalise, falling back to the
    default layer when they are near zero. oh has no layer, so its probability is
    zeroed here, which makes this the conditional probability of which of the
    achievable vowels is being sung."""
    q = p5[[_LCLS[i] for i in _LSEL]]
    s = float(q.sum())
    return (q / s).astype("float32") if s > 1e-6 else _VW_MID


_SEL_NAMES = [VOWEL_NAMES[_LCLS[i]] for i in _LSEL]   # layer index to vowel name
if a.vowels:
    print(f"[field] vowel set {'/'.join(_SEL_NAMES)}"
          + (f", weight smoothing {a.vw_tau}s" if a.vw_tau > 0 else ""),
          flush=True)

# With vowels on, the cache key includes the layer definitions plus "vowels1", so
# it lands in **a different bank_ directory**, not one byte of the existing cache
# changes, and --vowels 0 can be used to go back at any time.
# Review #3, 2026-08-17: the key takes only the fields that **enter the render**,
# that is, the name, model, spk, gain and range. sh and th are harmony logic in the
# playback layer, and changing them leaves the rendered banks byte-identical. The
# old key used the whole repr(VOICES), so adjusting the shift in v41 and v42
# re-rendered all eight banks, about a minute each, and the first switch into bank
# mode then hit respond_shell's 30 s switch watchdog.
_ckv = [(nm, mdl, sp, g, rng) for nm, mdl, sp, g, rng, _sh, _th in VOICES]
_ck = hashlib.md5((repr(_ckv) + a.vowel_src + str(NOTE_S) + "perloop1"
                   + (repr(VOWEL_LAYERS) + "vowels1" if a.vowels else ""))
                  .encode()).hexdigest()[:10]
BANK_DIR = f"scratchpad/bank_{_ck}"
BANKS = {}                               # name -> {midi: np.ndarray of the loop span}
LBANKS = {}                              # v16: name -> [layer][midi] -> ndarray
XBANKS = {}                              # v39: (name, model, spk) -> {midi: …}
# v38 ensemble timbres (--per-part): **another speaker really is another singer**,
# since these are different M4Singer people trained jointly. Reviewed by ear on
# 2026-08-14 for reflow-alto3 (n_spk=3):
#   spk1 and spk2 are two people, both clean ("they do sound like different
#     people")
#   spk3 was **rejected**: "clearly non-vocal noise and distorted pitch".
#     Corroborating evidence: it has the least training data, 1182 files against
#     1890 and 1933, and the highest spectral centroid of the three at 3564 Hz,
#     which is high-frequency rubbish. It was undertrained.
#
# v39 (2026-08-15): the field widened from spk to **(model, spk)**, so a borrowed
# singer need not share the lead's model. v38's remark that the bass and soprano
# models have n_spk=1 and therefore no second person to borrow is now void:
# reflow-fem8 and reflow-male8, each 8 real M4Singer people at 40000 steps, have
# been trained, and on 2026-08-15 fem8 was judged "all usable" and male8 "all pass
# for now, distinguishable but not markedly".
# **The lead voices are deliberately unchanged**, by decision: the three current
# voices each passed by ear along the way, and eleven days before the viva the
# existing path is not touched. So this is purely additive: the leads stay as they
# are while the ensemble members are upgraded from a detuned copy of the lead to
# genuinely other people.
#
# Who is borrowed is matched by **range**, not by the number of files (the lesson
# of 2026-08-15: file count is not the criterion, and Soprano-2 with 455 files is
# usable at 40k):
#   bass  (36-68) <- the three basses of male8
#   tenor (43-68) <- male8's Tenor-1, 3 and 5 (medians 61, 57 and 52, well spread)
#   alto  (50-80) <- fem8's Alto-1, 4 and 5, avoiding Alto-6, who is the same
#                    person as the lead alto3 spk2
#   sop   (53-84) <- the three sopranos of fem8
# The members' banks are rendered over **that part's range**, taking lo and hi
#   from the lead's entry, and not over the borrowed singer's comfortable region.
#   F24: the range is what decides the type of timbre. Borrowing a high voice for a
#   low part pushes it into the broken region, which is why the assignments above
#   follow the medians rather than taking any three.
_M8 = "reflow-male8/model_40000.pt"   # 1-3 Bass-1/2/3；4-8 Tenor-1/2/3/5/7
_F8 = "reflow-fem8/model_40000.pt"    # 1-3 Soprano-1/2/3；4-8 Alto-1/4/5/6/7
# Judged by ear on 2026-08-15 (v39c): the first version borrowed three singers per
# part and was judged noisy live. Listening to each one separately offline, with a
# span rendered over **that part's range** at matched loudness, found **8 of the 12
# to be bad**:
#   bad  Bass-1, Bass-3 / Tenor-1, Tenor-3 / Alto-1, Alto-4 / Soprano-1, Soprano-2
#   kept Bass-2        / Tenor-5          / Alto-5         / Soprano-3
# **The same people had all passed the probe that morning** (fem8 "all usable",
# male8 "all pass for now"). The probe uses one shared set of notes while a bank
# runs the whole range of a part, so **passing the probe does not transfer to the
# bank**, and a new singer must always be listened to again on the bank path
# (F26).
# The range hypothesis was overturned by its own data: the range statistics of
# Bass-1, rejected, and Bass-2, kept, are almost identical, and Tenor-5, pushed
# furthest from its own region, is the one that was kept. The only thing that
# correlates with the judgement is **the number of training files**: all four
# survivors have 1165 or more, and all eight rejected have 1073 or fewer except
# Bass-1 at 1758. Treat it as a hint when choosing candidates and never as a
# predictor; three instruments have already been fooled today, periodicity,
# spectral flatness and the dump.
MEM_SPK = {"bass":  [(_M8, 2)],      # Bass-2, 1656 files. C2-D2 contains broken notes;
           #   that region is reachable after the bass moved to -8 in v41, so
           #   re-verify note by note before using --mem-real 1
           "tenor": [(_M8, 7)],      # Tenor-5, 1224 files. From v42 this is the same person as the lead
           #   (the lead voice became male8 spk7). The playback side excludes it
           #   automatically (see the lead exclusion in _avail); it stays on the
           #   list only as a record. Every other tenor in male8 was rejected by
           #   ear, so the tenor currently has **no** real singer to borrow.
           "alto":  [(_F8, 6)],      # Alto-5, 1934 files
           "sop":   [(_F8, 3)]}      # Soprano-3, 1165 files


def _savez_atomic(path, bank):
    """Write the cache atomically: write a temporary file, then os.replace.
    An np.savez interrupted half-way, by the switch watchdog's SIGINT, a shutdown
    or a full disk, leaves half a zip, and every later start then fails inside
    np.load. The worse form is "incomplete but valid": a zip entry cut at its
    boundary loads silently as a short bank, and one part spends the whole evening
    folded into whatever low octave survived, with no error at all. Review #14,
    2026-08-17, reproduced all three failure modes."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, **{str(k): v for k, v in bank.items()})
    os.replace(tmp, path)


def _load_bank(path, lo, hi, what):
    """Load the cache and verify it: a hit requires that it loads and that the
    notes cover lo..hi exactly. Otherwise it returns None and is treated as a cache
    miss and re-rendered; a corrupt file is overwritten atomically by
    _savez_atomic."""
    try:
        z = np.load(path)
        bank = {int(k): z[k] for k in z.files}
    except Exception as e:                   # BadZipFile, a truncated entry, permissions and so on
        print(f"⚠ [bank] {what} cache unreadable ({e}) → re-rendering",
              flush=True)
        return None
    if sorted(bank) != list(range(lo, hi + 1)):
        print(f"⚠ [bank] {what} cache incomplete ({len(bank)}/{hi - lo + 1} "
              f"notes) → re-rendering", flush=True)
        return None
    return bank


def _render_layer(svc, UU, lo, hi):
    """v16: render the whole range from the given units UU into {midi: the loop
    span alone}; used only by --vowels.

    The body is a **deliberate line-for-line copy** of the main render loop in
    _build_banks below. "No behavioural change on the default path" is a hard
    constraint that outranks avoiding repetition: with not one character of the
    existing loop touched, this change cannot possibly reach the reference quality.
    If the rendering (f0, vol, mask, seed, loop trimming) is ever changed, both
    places must change.
    """
    import torch
    nb = UU.size(1)
    NF = int(NOTE_S * SR / HOP)
    bank = {}
    with torch.no_grad():
        for m in range(lo, hi + 1):
            f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
            vol = np.full(NF, 0.06)
            vol[:4] = np.linspace(0, 0.06, 4)      # remove the onset spike
            vol_t = (torch.from_numpy(vol).float()
                     .to(svc.device)[None, :, None])
            mask = torch.ones(1, NF * HOP, device=svc.device)
            torch.manual_seed(1234 + m)
            au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                           units_override=UU[:, np.arange(NF) % nb],
                           feats=(f0, vol_t, mask), ratios=[None]
                           )[0].cpu().numpy()
            seg = au[LOOP_A:LOOP_B + XF].astype("float32")
            T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
            L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
            xfs = int(0.005 * SR)
            lin = np.linspace(0, 1, xfs, dtype="float32")
            out_ = seg[:L_].copy()
            out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
            bank[m] = out_
            del au
    return bank


def _build_mem_banks():
    """v38: the "other singer" banks used by the ensemble members; built only when
    --per-part is 2 or more and that model has n_spk above 1.

    The rendering body is a **deliberate line-for-line copy** of the main loop in
    _build_banks, for the same reason as _render_layer: not one character of the
    existing loop is touched, so this change cannot reach the reference quality. If
    the rendering (f0, vol, mask, seed, loop trimming) is ever changed, all three
    places must change.
    The cache file name carries the model and the spk, `<nm>_<model>_s<spk>.npz`,
    with the model added in v39, so it can never overwrite the lead's bank or the
    same spk number from another model, and the BANK_DIR hash need not change:
    switching model or spk simply changes the file name, so nothing stale is ever
    read.
    """
    import torch
    import spike_stream6 as S
    pairs = [(nm, m2, s2) for nm, *_x in VOICES
             for (m2, s2) in MEM_SPK.get(nm, [])]
    if not pairs:
        return
    # v39: the model now comes from the MEM_SPK entry, so only the lead's gain and
    # **range** are kept here.
    spec = {nm: (g, lo, hi)
            for nm, _mdl, _sp, g, (lo, hi), _sh, _th in VOICES}
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    NF = int(NOTE_S * SR / HOP)
    UU = None
    for nm, mdl, sp2 in pairs:
        tag = mdl.split("/")[0]              # reflow-male8 / reflow-fem8 …
        path = f"{BANK_DIR}/{nm}_{tag}_s{sp2}.npz"
        g, lo, hi = spec[nm]
        if os.path.exists(path) and not a.rebuild:
            bk0 = _load_bank(path, lo, hi, f"{nm} {tag} spk{sp2}")
            if bk0 is not None:
                XBANKS[(nm, mdl, sp2)] = bk0
                print(f"[bank] {nm} {tag} spk{sp2} cache hit "
                      f"({len(bk0)} notes)", flush=True)
                continue
        svc = S.Svc([(f"{S.DDSP}/exp/{mdl}", 0.0, g, sp2)],
                    step=2, t_start=0.85)
        if UU is None:
            _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
            runs, i = [], 0
            while i < len(uv0):
                if not uv0[i]:
                    j = i
                    while j < len(uv0) and not uv0[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            a0, b0 = max(runs, key=lambda r: r[1] - r[0])
            mid = (a0 + b0) // 2
            with torch.no_grad():
                UU = svc.encode(
                    mic0[max(0, (mid - 40) * HOP):(mid + 40) * HOP])[:, 8:-8]
        nb = UU.size(1)
        bank = {}
        t0 = time.time()
        with torch.no_grad():
            for m in range(lo, hi + 1):
                f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
                vol = np.full(NF, 0.06)
                vol[:4] = np.linspace(0, 0.06, 4)
                vol_t = (torch.from_numpy(vol).float()
                         .to(svc.device)[None, :, None])
                mask = torch.ones(1, NF * HOP, device=svc.device)
                torch.manual_seed(1234 + m)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask), ratios=[None]
                               )[0].cpu().numpy()
                seg = au[LOOP_A:LOOP_B + XF].astype("float32")
                T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
                L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
                xfs = int(0.005 * SR)
                lin = np.linspace(0, 1, xfs, dtype="float32")
                out_ = seg[:L_].copy()
                out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
                bank[m] = out_
                del au
                if (m - lo) % 8 == 7:        # progress printing, which feeds the shell's switch watchdog
                    print(f"[bank] {nm} {tag} spk{sp2} rendering "
                          f"{m - lo + 1}/{hi - lo + 1}", flush=True)
        _savez_atomic(path, bank)
        XBANKS[(nm, mdl, sp2)] = bank
        print(f"[bank] {nm} {tag} spk{sp2} rendered {len(bank)} notes "
              f"({time.time() - t0:.0f}s) → {path}", flush=True)


def _norm_layers(nm):
    """v17 (review A4): at a given midi note, align the rms of every layer to the
    middle layer. Measured, the loudness between layers differs by as much as 4 dB,
    so even with the weights summing to 1, changing the mouth shape changes the
    level. This touches memory only and never the cache files, and the middle layer
    is left byte-identical, so degenerating to one-hot is still bit-identical to a
    single layer."""
    for m_, ref in LBANKS[nm][L_MID].items():
        r0 = float(np.sqrt((ref ** 2).mean()))
        for li in range(len(VOWEL_LAYERS)):
            if li == L_MID:
                continue
            b_ = LBANKS[nm][li][m_]
            r_ = float(np.sqrt((b_ ** 2).mean()))
            if r_ > 1e-9:
                LBANKS[nm][li][m_] = (b_ * (r0 / r_)).astype("float32")


def _build_banks():
    """Render the banks at reference quality: about a minute the first time, and
    seconds from cache afterwards."""
    import torch
    import spike_stream6 as S
    os.makedirs(BANK_DIR, exist_ok=True)
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    UU = None
    NF = int(NOTE_S * SR / HOP)
    for nm, mdl, sp, g, (lo, hi), _sh, _th in VOICES:
        path = f"{BANK_DIR}/{nm}.npz"
        # v16: each layer bank is stored in its own file, and a hit requires all
        # of them; if one is missing the whole voice is re-rendered.
        lp = ([f"{BANK_DIR}/{nm}_L{li}.npz"
               for li in range(len(VOWEL_LAYERS))] if a.vowels else [])
        if (os.path.exists(path) and all(os.path.exists(p) for p in lp)
                and not a.rebuild):
            bk0 = _load_bank(path, lo, hi, nm)
            lb0 = []
            if bk0 is not None and a.vowels:
                for li, p_ in enumerate(lp):
                    b_ = _load_bank(p_, lo, hi, f"{nm} L{li}")
                    if b_ is None:
                        lb0 = None           # any bad layer means the whole voice is re-rendered
                        break
                    lb0.append(b_)
            if bk0 is not None and lb0 is not None:
                BANKS[nm] = bk0
                if a.vowels:
                    LBANKS[nm] = lb0
                    _norm_layers(nm)
                print(f"[bank] {nm} cache hit ({len(bk0)} notes)", flush=True)
                continue
        svc = S.Svc([(f"{S.DDSP}/exp/{mdl}", 0.0, g, sp)],
                    step=2, t_start=0.85)
        if UU is None:
            _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
            runs, i = [], 0
            while i < len(uv0):
                if not uv0[i]:
                    j = i
                    while j < len(uv0) and not uv0[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            a0, b0 = max(runs, key=lambda r: r[1] - r[0])
            mid = (a0 + b0) // 2
            with torch.no_grad():
                UU = svc.encode(
                    mic0[max(0, (mid - 40) * HOP):(mid + 40) * HOP])[:, 8:-8]
        nb = UU.size(1)
        bank = {}
        t0 = time.time()
        with torch.no_grad():
            for m in range(lo, hi + 1):
                f0 = np.full(NF, 440.0 * 2 ** ((m - 69) / 12.0))
                vol = np.full(NF, 0.06)
                vol[:4] = np.linspace(0, 0.06, 4)      # remove the onset spike
                vol_t = (torch.from_numpy(vol).float()
                         .to(svc.device)[None, :, None])
                mask = torch.ones(1, NF * HOP, device=svc.device)
                torch.manual_seed(1234 + m)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask), ratios=[None]
                               )[0].cpu().numpy()
                # Take the loop region and hide the wrap join with a crossfade,
                # then store the loop span alone.
                seg = au[LOOP_A:LOOP_B + XF].astype("float32")
                # R2B v2: an integer-period loop. The correlation between the head
                # and the tail measures 0.96, so the material is nearly periodic and
                # an equal-power join actually adds a +3 dB bump. Taking the loop
                # length as a whole number of fundamental periods aligns the phase,
                # and the residual noise component is closed with a 5 ms linear
                # join.
                T_ = SR / (440.0 * 2 ** ((m - 69) / 12.0))
                L_ = int(round(round((LOOP_B - LOOP_A) / T_) * T_))
                xfs = int(0.005 * SR)
                lin = np.linspace(0, 1, xfs, dtype="float32")
                out_ = seg[:L_].copy()
                out_[:xfs] = seg[:xfs] * lin + seg[L_:L_ + xfs] * (1 - lin)
                bank[m] = out_
                del au
                if (m - lo) % 8 == 7:        # progress printing, which feeds the shell's switch watchdog
                    print(f"[bank] {nm} rendering {m - lo + 1}/{hi - lo + 1}",
                          flush=True)
        _savez_atomic(path, bank)
        BANKS[nm] = bank
        print(f"[bank] {nm} rendered {len(bank)} notes "
              f"({time.time() - t0:.0f}s) → {path}", flush=True)
        if a.vowels:
            # v16: the same voice and the same rendering, changing only the source
            # of the units, gives five banks from dark to bright.
            LBANKS[nm] = []
            for li, (tag, src, lt0, lt1, _tl) in enumerate(VOWEL_LAYERS):
                lpath = f"{BANK_DIR}/{nm}_L{li}.npz"
                if os.path.exists(lpath) and not a.rebuild:
                    lb_ = _load_bank(lpath, lo, hi, f"{nm} L{li} {tag}")
                    if lb_ is not None:
                        LBANKS[nm].append(lb_)
                        print(f"[layer] {nm} L{li} {tag} cache hit", flush=True)
                        continue
                mv, _sr = sf.read(src, dtype="float64", always_2d=True)
                mv = mv[:, 0]
                mid = int((lt0 + lt1) / 2 * SR / HOP)   # the centre of the window, in hop coordinates
                with torch.no_grad():
                    UL = svc.encode(mv[max(0, (mid - 40) * HOP):
                                       (mid + 40) * HOP])[:, 8:-8]
                tl0 = time.time()
                lb = _render_layer(svc, UL, lo, hi)
                _savez_atomic(lpath, lb)
                LBANKS[nm].append(lb)
                print(f"[layer] {nm} L{li} {tag} rendered {len(lb)} notes"
                      f" ({time.time() - tl0:.0f}s) → {lpath}", flush=True)
            _norm_layers(nm)
        del svc


def _norm_mem_banks():
    """v39, from a live comment on 2026-08-15 ("the female voices seem to cover the
    male ones; at least I could not hear the men"): align the rms of a borrowed
    member bank to the lead bank of that part, **note by note**.

    Why it is needed: the MIXG part balance below is measured from **the lead**
    banks, and the borrowed banks never entered that calculation. Measured on the
    mean rms of bank_853950ed4f, the three sopranos borrowed are louder than the
    lead by
    Why it is needed: the MIXG part balance below is measured from the lead banks,
    and the borrowed banks never entered that calculation. Measured on the mean rms
    of bank_853950ed4f, the three borrowed sopranos are 4.3, 4.8 and 4.9 dB louder
    than the lead while the two borrowed tenors are 1.9 and 2.0 dB quieter, so the
    soprano part sits about +3.9 dB and the tenor about -1 dB, opening a 5 dB gap.
    This is the same trap as the lesson that killed --balance, measuring the wrong
    bank; the difference is that this time the newly added half was left out.
    Note by note, not overall: different singers have different loudness curves at
    the ends of a range, and aligning overall makes one of them vanish in the high
    register. The same reason _norm_layers works note by note.
    This touches memory only and never the cache files, as _norm_layers does, so
    the lead and the cache stay byte-identical.
    """
    for (nm, mdl, sp), bk in XBANKS.items():
        ref_bank = BANKS.get(nm)
        if not ref_bank:
            continue
        ds = []
        for m_, b_ in bk.items():
            ref = ref_bank.get(m_)
            if ref is None:
                continue
            r0 = float(np.sqrt((ref ** 2).mean()))
            r_ = float(np.sqrt((b_ ** 2).mean()))
            if r0 > 1e-9 and r_ > 1e-9:
                bk[m_] = (b_ * np.float32(r0 / r_)).astype("float32")
                ds.append(20.0 * np.log10(r0 / r_))
        if ds:
            # Print it, so whether this step really happened is visible; running
            # bare is a silent failure.
            print(f"[mem-norm] {nm} {mdl.split('/')[0]} spk{sp}: "
                  f"{np.mean(ds):+.1f}dB (per note {np.min(ds):+.1f} to "
                  f"{np.max(ds):+.1f})", flush=True)


_build_banks()
if a.per_part > 1 and a.mem_real and not a.vowels:
    # --mem-real 0 borrows nobody, which is the frozen configuration, and skips
    # rendering and loading entirely. Review #3, 2026-08-17: it used to instantiate
    # fem8 and male8 anyway, render or load 122 notes and keep about 28 MB
    # resident, while _avail = [] at the output meant they could never sound; on an
    # exhibition machine without the fem8 weights it even died at start-up.
    # With --vowels on, no other singer is borrowed: LBANKS holds layers only for
    # the lead's spk, so a member would get someone else's voice against its own
    # layers and the mixing would not line up. Supporting both would mean rendering
    # the members' layer banks first, which is another matter. The vowel line was
    # judged on 2026-08-13 as "back to ah throughout", so it does not block this.
    _build_mem_banks()
    _norm_mem_banks()
# v32 part balance: level using the **measured** rms of the banks, plus the ear
# layer of --trim.
MIXG = {nm: 1.0 for nm, *_x in VOICES}
if a.balance or a.trim:
    _rms = {nm: float(np.mean([np.sqrt((b ** 2).mean())
                               for b in BANKS[nm].values()]))
            for nm, *_x in VOICES}
    _ref = _rms[VOICES[0][0]]
    if a.balance:
        MIXG = {nm: _ref / max(v, 1e-9) for nm, v in _rms.items()}
    for _t in a.trim.split(","):
        if "=" in _t:
            _k, _v = _t.split("=")
            _k = _k.strip()
            assert _k in MIXG, f"--trim does not recognise the part {_k}"
            MIXG[_k] *= 10 ** (float(_v) / 20.0)
    print("[mix] " + "  ".join(
        f"{nm} {20 * np.log10(MIXG[nm]):+.1f}dB" for nm, *_x in VOICES),
          flush=True)
print(f"sample bank ready ({sum(len(b) for b in BANKS.values())} notes)",
          flush=True)


class KeyTracker:
    MIN = 120

    def __init__(self):
        self.h = np.zeros(12)

    def push(self, midi_f):
        self.h[int(round(midi_f)) % 12] += 1.0

    def root(self):
        if self.h.sum() < self.MIN:
            return 0
        cov = [sum(self.h[(r + d) % 12] for d in MAJ) for r in range(12)]
        return int(np.argmax(cov))


def dia_step(m, root, deg):
    """Within the major root, the semitone offset to the note deg diatonic steps
    above m (deg 2 is a third, 4 is a fifth)."""
    rel = (m - root) % 12
    idx = int(np.argmin([min(abs(rel - s), 12 - abs(rel - s)) for s in MAJ]))
    return MAJ[(idx + deg) % 7] + 12 * ((idx + deg) // 7) - MAJ[idx]


def dia_third(m, root):
    return dia_step(m, root, 2)


def f0_autocorr(x):
    """f0 by normalised autocorrelation over a window of about 46 ms, from 65 to
    800 Hz; silence returns 0. No machine learning, and indifferent to level."""
    x = x - x.mean()
    e = float(np.sqrt(np.mean(x * x)))
    if e < 1e-5:
        return 0.0
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo, hi = int(SR / 800), min(int(SR / 65), n - 1)
    k0 = lo + int(np.argmax(ac[lo:hi]))
    if ac[k0] < 0.45:                     # voicing is decided on the raw peak alone (R2A#4:
        return 0.0                        # the guard must not demote a weak voice to silence)
    k = k0
    # Review A14, subharmonic protection: if the peak at k/2 is almost as high, the
    # real fundamental is an octave up.
    while k >= 2 * lo and ac[int(round(k / 2))] > 0.85 * ac[k]:
        k = int(round(k / 2))
    if ac[k] < 0.45:
        k = k0                            # the halved candidate is too weak, so keep the original peak
    # parabolic interpolation to refine it
    if 0 < k < n - 1:
        d = (ac[k - 1] - ac[k + 1]) / (2 * (ac[k - 1] - 2 * ac[k]
                                            + ac[k + 1]) + 1e-12)
        k = k + float(np.clip(d, -1, 1))
    return SR / k


hopN = max(1, int(a.hop * SR))
# v17 two-dimensional mouth-shape anchors, calibrated by the full-range
# lip_sweep2 sweep; row i of anchors corresponds to layer i of VOWEL_LAYERS, the
# mapping established in v16.3, and is not changed. A missing file pins the
# weights to a one-hot on the middle layer and warns, rather than guessing.
# (The v16.2 voice fingerprint vowel_calib.npz is no longer loaded as a fallback:
# it died of contamination by bleed from the parts, post-mortem in the daily note
# for 2026-08-13. The mouth cannot see the speakers.)
if a.vowels:
    try:
        _zl = np.load("scratchpad/vowel_lip_calib.npz")
        _VL_A, _VL_S = _zl["anchors"], _zl["scales"]
        # Review S1/A5: if the number of anchors does not equal the number of
        # layers, the weight shape blows up the worker, which then appears alive
        # while silent; a scale at or below 0, or non-finite, amplifies NaN three
        # times over. An invalid calibration file disables the feature rather than
        # guessing.
        if (_VL_A.shape != (len(VOWEL_NAMES), 2)
                or not np.isfinite(_VL_A).all()
                or not np.isfinite(_VL_S).all() or (_VL_S <= 0).any()):
            print(f"⚠ [field] calibration file invalid (anchors {_VL_A.shape}, "
                  f"scales {_VL_S}) = vowel field disabled (fixed mid)",
                  flush=True)
            _VL_A = None
        else:
            print(f"[field] mouth-shape anchors loaded ({len(_VL_A)} vowels)", flush=True)
    except FileNotFoundError:
        _VL_A = None
        print("⚠ [field] no vowel_lip_calib.npz = vowel field disabled (fixed mid)",
              flush=True)
    # v17.1 full lip-ring anchors, produced by scratchpad/lip_ring_calib.py after
    # the question "why not measure the whole ring of the mouth?" on 2026-08-13.
    # If the file exists the ring is preferred and the 2D version is the fallback.
    # Validation follows the same S1/A5 standard, shape, finiteness and scale above
    # 0, and an invalid file falls back to 2D rather than guessing.
    _VR_A = None
    try:
        _zr = np.load("scratchpad/vowel_ring_calib.npz")
        _VR_A, _VR_S = _zr["anchors"], _zr["scales"]
        if (_VR_A.ndim != 2 or _VR_A.shape[0] != len(VOWEL_NAMES)
                or _VR_S.shape != (_VR_A.shape[1],)
                or not np.isfinite(_VR_A).all()
                or not np.isfinite(_VR_S).all() or (_VR_S <= 0).any()):
            print(f"⚠ [field] full-ring mouth calibration invalid "
                  f"(anchors {_VR_A.shape}) = falling back to 2D", flush=True)
            _VR_A = None
        else:
            print(f"[field] full-ring mouth anchors loaded ({_VR_A.shape[0]} vowels x"
                  f" {_VR_A.shape[1]} dims)", flush=True)
    except FileNotFoundError:
        pass                                 # no ring calibration, so use the 2D version

# LIP_RING and ring_vec here are a **deliberate copy** of the ones in
# scratchpad/lip_ring_calib.py; changing one means changing both. The same
# discipline as _render_layer: the calibration and the live features must be
# bit-identical.
LIP_RING = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405,
            314, 17, 84, 181, 91, 146,           # outer ring, 20 points
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402,
            317, 14, 87, 178, 88, 95]            # inner ring, 20 points


def _ring_vec(LA, aspect):
    """LA, normalised (478, 2) coordinates, to an 80-dimensional lip-ring shape:
    translation removed by the lip centre, rotation by the line between the
    temples, and scale by the distance between them. x is multiplied by the aspect
    ratio so the geometry is isometric."""
    pts = LA[LIP_RING].copy()
    pts[:, 0] *= aspect
    e0 = np.array([LA[234, 0] * aspect, LA[234, 1]])
    e1 = np.array([LA[454, 0] * aspect, LA[454, 1]])
    ax = e1 - e0
    sc = float(np.hypot(ax[0], ax[1])) + 1e-9
    th = float(np.arctan2(ax[1], ax[0]))
    c, s = np.cos(-th), np.sin(-th)
    R = np.array([[c, -s], [s, c]])
    return ((pts - pts.mean(0)) @ R.T / sc).ravel()


# v18 personal vowel model, produced by scratchpad/vowel_ml_train.py, a logistic
# model with 405 parameters. If the file exists its probabilities are used directly
# as the layer-mixing weights, in preference to the anchor geometry. Experimental
# criterion (vowel_ml_probe, same data, blocked cross-validation over time):
# nearest neighbour scored an accuracy of 0.81 with a **true-class probability of
# 0.57**, this model 0.89 and **0.86**. The blur heard live is that 0.57 measured.
# Validation follows the same S1/A5 standard.
_VM_W = None
if a.vowels:
    try:
        _zm = np.load("scratchpad/vowel_ring_model.npz")
        _VM_W, _VM_B = _zm["W"], _zm["b"]
        _VM_M, _VM_S = _zm["mean"], _zm["scale"]
        _D = len(LIP_RING) * 2
        if (_VM_W.shape != (len(VOWEL_NAMES), _D)
                or _VM_B.shape != (len(VOWEL_NAMES),)
                or _VM_M.shape != (_D,) or _VM_S.shape != (_D,)
                or not all(np.isfinite(x).all()
                           for x in (_VM_W, _VM_B, _VM_M, _VM_S))
                or (_VM_S <= 0).any()):
            print(f"⚠ [field] vowel model invalid (W {_VM_W.shape}) = back to anchors",
                  flush=True)
            _VM_W = None
        else:
            print(f"[field] vowel model loaded ({_VM_W.size + _VM_B.size} params, "
                  f"CV acc {float(_zm['cv_acc']):.2f})", flush=True)
    except FileNotFoundError:
        pass                                 # no model, so use the anchor geometry

# v19 fused vowel model, produced by scratchpad/vowel_fusion_train.py: the
# 80-dimensional lip ring plus 13 MFCC dimensions. Live on v18 the judgement was
# "eh cannot be told apart, oh and oo were never reached, ee and ah are fine",
# which is a blind spot of lip physics: eh and ee differ in tongue position, and oh
# and oo in protrusion, neither of which a frontal camera has depth for. The
# acoustics cover exactly those two (fusion probe: eh and ee rose from 0.79 to
# 0.94, overall accuracy from 0.89 to 0.95, weight 0.94).
# Measured, 1:1 bleed from the parts costs only 0-1 percentage points, so **no
# spectral subtraction is done**, which saves a whole alignment pipeline.
# The MFCC below is a **deliberate copy** of scratchpad/vowel_ml_probe.py::mfcc.
# Its parameters are stored in the model file and checked item by item, and a
# mismatch refuses the load, so the two definitions cannot quietly diverge.
_SG_W = None
_SG_BASE = None          # v40 opening baseline; not None only once calibration has finished
_SG_CAL = []             # features accumulated during v40 calibration
_SG_JAWQ = []            # sliding window of jawOpen, for the movement measure
_SG_MTH = 0.0            # v40 movement veto threshold (0 does not veto, the old behaviour)
_SG_JC = 2
_SG_MW = 15
_SG_NEEDCAL = False
if a.sing_gate:
    try:
        # v40: auto **deliberately still points at the older model**, so this
        # commit changes no current behaviour. model3, trained across three
        # sessions with self-calibration and a movement veto, takes effect only if
        # its path is written out: measured live on 2026-08-15 it still triggered
        # repeatedly by mistake, with sp saturating at 1.00 on both a closed mouth
        # and a hand over the mouth, so it does not qualify as the default. To try
        # it: --gate-model scratchpad/sing_gate_model3.npz
        _sgp = (a.gate_model if a.gate_model != "auto"
                else "scratchpad/sing_gate_model.npz")
        _zs = np.load(_sgp)
        _SG_W, _SG_B = _zs["W"], float(_zs["b"])
        _SG_M, _SG_S, _SG_N = _zs["mean"], _zs["scale"], list(_zs["names"])
        # v34.1: the model takes only the **mouth and jaw** dimensions (idx). The
        # full 52-dimension version also learned the eyebrows and cheeks, which
        # carried 49% of the total weight, and scored 98.8% accuracy within a
        # session but **only 72.2% across sessions with 47.5% false triggers while
        # not singing**, which is why a closed mouth showed sing 1.00. Keeping only
        # the mouth and jaw gives 93.4% across sessions with 0.0% false triggers.
        _SG_I = _zs["idx"]
        if (_SG_W.shape != _SG_M.shape or _SG_S.shape != _SG_M.shape
                or len(_SG_N) != _SG_W.size or _SG_I.shape != _SG_W.shape
                or not np.isfinite(_SG_W).all() or (_SG_S <= 0).any()):
            print("! [gate] the singing model failed validation; falling back to the geometric gate", flush=True)
            _SG_W = None
        else:
            _SG_CHK = [str(x) for x in _SG_N]   # checked against the first frame, see below
            # v40: the four fields of the new contract. An older model has none of
            # them and falls back to the old behaviour throughout, with calibration
            # and the veto off, so **one piece of code supports both models**
            # without branching into two paths.
            _SG_NEEDCAL = bool(int(_zs["calib"])) if "calib" in _zs else False
            _SG_JC = int(_zs["jaw_col"]) if "jaw_col" in _zs else 2
            _SG_MW = int(_zs["motion_win"]) if "motion_win" in _zs else 15
            _SG_MTH = (float(_zs["motion_th"]) if "motion_th" in _zs else 0.0)
            if a.gate_motion_th >= 0:            # the flag overrides it, adjustable live
                _SG_MTH = a.gate_motion_th
            # What cv means: the older model stores blocked cross-validation
            #   **within one session** while this line used to print it as
            #   cross-session, a mislabelling found on 2026-08-15. The newer model
            #   stores a **leave-one-session-out** mean. So the label follows the
            #   model rather than always printing cross-session.
            _cvlab = "leave-one-session-out" if _SG_NEEDCAL else "same-session"
            print(f"[gate] sing model loaded ({_SG_W.size + 1} params, "
                  f"mouth/jaw only, {_cvlab} acc {float(_zs['cv']):.3f}) "
                  f"← {os.path.basename(_sgp)}", flush=True)
            if _SG_NEEDCAL:
                print(f"[gate] v40: {a.gate_calib:.0f}s baseline self-calibration at the start "
                      f"(**keep the mouth closed**) plus a movement veto at {_SG_MTH:.4f}", flush=True)
    except FileNotFoundError:
        print("! [gate] no sing_gate_model.npz; the gate falls back to the blendshape geometric thresholds",
              flush=True)
    except Exception as _e:                      # Review: a corrupt npz or a missing key
        _SG_W = None                             #   used to raise and prevent start-up
        print(f"! [gate] could not load the singing model ({_e}); falling back to the blendshape geometric thresholds",
              flush=True)

_VF_W = None
if a.vowels:
    try:
        _zf = np.load("scratchpad/vowel_fusion_model.npz")
        _VF_W, _VF_B = _zf["W"], _zf["b"]
        _VF_M, _VF_S = _zf["mean"], _zf["scale"]
        _FN, _FH = int(_zf["nfft"]), int(_zf["hopf"])
        _NM, _NC = int(_zf["nmel"]), int(_zf["ncep"])
        _F0, _F1 = float(_zf["fmin"]), float(_zf["fmax"])
        _DIM = len(LIP_RING) * 2 + _NC
        if (str(_zf["feat"]) != "ring+mfcc"
                or _VF_W.shape != (len(VOWEL_NAMES), _DIM)
                or _VF_B.shape != (len(VOWEL_NAMES),)
                or _VF_M.shape != (_DIM,) or _VF_S.shape != (_DIM,)
                or not all(np.isfinite(x).all()
                           for x in (_VF_W, _VF_B, _VF_M, _VF_S))
                or (_VF_S <= 0).any()):
            print(f"⚠ [field] fusion model invalid (W {_VF_W.shape}) = back to vision",
                  flush=True)
            _VF_W = None
        else:
            def _melfb(nmel, fmin, fmax, nfft):
                def m(f):
                    return 2595 * np.log10(1 + f / 700)

                def im(x):
                    return 700 * (10 ** (x / 2595) - 1)
                pts = im(np.linspace(m(fmin), m(fmax), nmel + 2))
                bb_ = np.floor((nfft + 1) * pts / SR).astype(int)
                fb = np.zeros((nmel, nfft // 2 + 1))
                for i in range(nmel):
                    lo_, mid_, hi_ = bb_[i], bb_[i + 1], bb_[i + 2]
                    if mid_ > lo_:
                        fb[i, lo_:mid_] = np.linspace(0, 1, mid_ - lo_)
                    if hi_ > mid_:
                        fb[i, mid_:hi_] = np.linspace(1, 0, hi_ - mid_)
                return fb

            _FFB = _melfb(_NM, _F0, _F1, _FN)
            _FWIN = np.hanning(_FN)
            _FDCT = np.cos(np.pi / _NM * (np.arange(_NM) + 0.5)[None, :]
                           * np.arange(1, _NC + 1)[:, None])
            _ABUF = [np.zeros(_FN, dtype="float32")]   # the most recent NFFT samples
            print(f"[field] fusion model loaded ({_VF_W.size + _VF_B.size} params, "
                  f"CV acc {float(_zf['cv_acc']):.2f}, vision+acoustic)",
                  flush=True)
    except FileNotFoundError:
        pass                                 # no fused model, so use the visual one
kt = KeyTracker()
st = {"die": False, "xrun": 0, "cents": 0.0, "note": None, "cand": None,
      "cc": 0, "quiet": 0, "lastm": None, "pada": None, "padt": -1e9,
      "hist": [], "tsw": -1e9, "pn": None, "porta_ok": True}

# -- the mouth gate, as in score_live v3.1: a closed mouth means the pitch is not
#    yours, which kills feedback --
M = {"ok": a.mouth == 0 or bool(a.file), "on": False, "last_open": 0.0,
     "val": -1.0, "t_on": -1e9, "face": False}


def _mouth_worker():
    # These two are assigned inside the function, to disable the model when it
    # fails validation, so they must be declared global. Otherwise the whole
    # function treats them as local, the first frame raises UnboundLocalError, the
    # thread dies inside a broad except, and the result is **no picture and the
    # gate wide open all evening** (a real case on 2026-08-13).
    # _SG_BASE below is **assigned**, not mutated, so without this line it is
    # likewise an UnboundLocalError, the camera thread dies for the whole session
    # and the gate stays open. v35 was bitten by exactly this.
    global _SG_W, _SG_CHK, _SG_BASE
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path="scratchpad/face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=bool(a.bs_gate or a.sing_gate))
        lmk = vision.FaceLandmarker.create_from_options(opts)
        if a.cam >= 0:
            cap = cv2.VideoCapture(a.cam)
            assert cap.isOpened(), f"camera {a.cam} will not open"
        else:
            best = None
            for ci in range(3):
                c = cv2.VideoCapture(ci)
                if not c.isOpened():
                    continue
                time.sleep(0.4)
                okf, fr = c.read()
                b = float(fr.mean()) if okf else -1
                c.release()
                if best is None or b > best[1]:
                    best = (ci, b)
            assert best and best[1] > 10, f"no camera with a picture was found {best}"
            print(f"[mouth] camera index {best[0]} (brightness {best[1]:.0f})",
                  flush=True)
            cap = cv2.VideoCapture(best[0])
        t0 = time.time()
        last_face = time.time()
        last_det = 0.0
        while not st["die"]:
            okf, frame = cap.read()
            now = time.time()
            M["face"] = now - last_face < 0.3     # the face channel, for the dump and the picture
            if okf:
                M["hb"] = now    # R4A#1: the heartbeat means a frame was really read.
                #   If read() fails, which happens routinely when the camera is
                #   taken or dropped, the heartbeat stops, the watchdog fires and
                #   it runs bare with a warning, rather than closing the gate for
                #   good in silence.
            if okf and now - last_det >= 0.033:      # about 30 Hz, previously 20
                last_det = now
                img = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = lmk.detect_for_video(img, int((now - t0) * 1000))
                if (a.view or a.frame_b64 > 0) and not res.face_landmarks:
                    # R4A#8: update the picture even when the face is lost; it
                    # used to freeze on the last frame with a face.
                    h2 = int(frame.shape[0] * 480 / frame.shape[1])
                    fr2 = cv2.resize(frame, (480, h2))
                    cv2.putText(fr2, "NO FACE - gate closing", (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    M["frame"] = fr2
                if not res.face_landmarks:
                    # Review: with the face lost, M["on"] and M["arm"] used to hold
                    # their old values, so when the face returned it took the
                    # "holding" branch, with a lower threshold and no debounce, and
                    # turning away and back with the mouth closed opened the gate.
                    # A lost face starts again.
                    M["on"] = False
                    M["arm"] = 0
                if res.face_landmarks:
                    last_face = now
                    f = res.face_landmarks[0]
                    val = abs(f[13].y - f[14].y) / (abs(f[10].y - f[152].y)
                                                    + 1e-9)
                    M["val"] = val
                    wd_ = (abs(f[61].x - f[291].x)
                           / (abs(f[234].x - f[454].x) + 1e-9))
                    M["wd"] = wd_
                    bs = (res.face_blendshapes[0]
                          if ((a.bs_gate or a.sing_gate)
                              and res.face_blendshapes) else None)
                    if bs is not None and _SG_W is not None:
                        # v34: a trained "is this person singing" probability as
                        # the gate, 52 blendshape dimensions through a logistic
                        # model. The negative samples include **speech**, which
                        # hand-written criteria never managed to block. The order
                        # of the names is verified at start-up.
                        if _SG_CHK is not None:
                            # Review: the comment claimed the order of the names
                            # was verified at start-up, and it was not. A change of
                            # mediapipe version silently shifts the layout. Check
                            # once, for real, on the first frame, and disable the
                            # model on a mismatch rather than guessing.
                            _live = [bs[i].category_name for i in _SG_I]
                            if _live != _SG_CHK:
                                print("! [gate] the blendshape name order does not match the model; "
                                      "the singing model is disabled", flush=True)
                                _SG_W = None
                            _SG_CHK = None
                        if _SG_W is None:
                            bs = None
                    if bs is not None and _SG_W is not None:
                        _v = np.array([c.score for c in bs])[_SG_I]
                        # v40, item 1: baseline self-calibration at the start. On
                        # 2026-08-15 the same person making the same movement
                        # measured a resting jawOpen of 0.031 on 2026-08-13 and
                        # 0.017 on 2026-08-15, so a per-frame logistic model taking
                        # absolute values flips wholesale, with 44% false triggers
                        # on a closed mouth. During calibration **the gate stays
                        # shut**, since the singer is asked to keep the mouth
                        # closed, and it opens only once calibration is done.
                        # _p must be given a value first: on the frame that
                        # **completes** calibration, this branch fills in
                        # _SG_BASE, so the test below is no longer "calibrating",
                        # execution falls into the else and reads _p, which is an
                        # UnboundLocalError, the camera thread dies for the whole
                        # session and the gate stays open. That really happened
                        # once.
                        _p = 0.0
                        if _SG_NEEDCAL and _SG_BASE is None:
                            _SG_CAL.append(_v)
                            # Use **time** and not a frame count: the frame rate
                            # varies with the machine and the light, so counting
                            # frames makes the real calibration time disagree with
                            # the seconds printed.
                            if now - M.setdefault("calt0", now) >= a.gate_calib \
                                    and len(_SG_CAL) >= 20:
                                _SG_BASE = np.mean(_SG_CAL, axis=0)
                                print(f"[gate] baseline calibration complete ({len(_SG_CAL)} frames / "
                                      f"{now - M['calt0']:.1f}s); the gate is now active",
                                      flush=True)
                            M["sp"] = 0.0
                            M["on"] = False
                            M["arm"] = 0
                            nv = False
                        else:
                            if _SG_BASE is not None:
                                _v = _v - _SG_BASE
                            # v40, item 2: the movement veto. Speech opens and
                            # closes the mouth at 3-8 Hz per syllable, while
                            # singing holds a posture. Measured mean |delta
                            # jawOpen| over a 0.5 s window: speech 0.0173, 0.0130
                            # and 0.0079 across three sessions, while the highest
                            # for singing, on ah, is only 0.0023. A difference
                            # measure does not depend on the baseline, so it holds
                            # across sessions.
                            _SG_JAWQ.append(float(bs[_SG_I[_SG_JC]].score))
                            del _SG_JAWQ[:-(_SG_MW + 1)]
                            _mo = (float(np.abs(np.diff(_SG_JAWQ)).mean())
                                   if len(_SG_JAWQ) > 1 else 0.0)
                            M["mo"] = _mo
                            _veto = _SG_MTH > 0 and _mo > _SG_MTH
                            _p = 1.0 / (1.0 + np.exp(
                                -(np.dot((_v - _SG_M) / _SG_S, _SG_W) + _SG_B)))
                            M["sp"] = float(_p)
                            if _veto:            # you are speaking, so it never counts
                                _p = 0.0
                        if _SG_NEEDCAL and _SG_BASE is None:
                            pass                 # calibrating; nv is already False
                        elif M["on"]:
                            nv = _p > a.sing_hold
                            if not nv:
                                M["arm"] = 0
                        else:
                            M["arm"] = (M.get("arm", 0) + 1
                                        if _p > a.sing_open else 0)
                            nv = M["arm"] >= a.bs_frames
                    elif bs is not None:
                        # v30: the blendshape gate. jawOpen covers the opening and
                        # mouthPucker the pucker. Both are trained quantities,
                        # normalised for head angle and distance, so unlike a
                        # hand-computed ratio they are not affected by where the
                        # camera sits.
                        _b = {c.category_name: c.score for c in bs}
                        jaw = _b.get("jawOpen", 0.0)
                        puc = _b.get("mouthPucker", 0.0)
                        M["jaw"], M["puc"] = jaw, puc
                        raw = (jaw > (a.bs_close if M["on"] else a.bs_open)
                               or puc > a.bs_pucker)
                        if M["on"]:
                            nv = raw
                            if not raw:
                                M["arm"] = 0
                        else:
                            # opening needs N consecutive frames, so a single
                            # noisy frame cannot open it for 0.8 s
                            M["arm"] = (M.get("arm", 0) + 1) if raw else 0
                            nv = M["arm"] >= a.bs_frames
                    elif M["on"]:
                        # v28 geometric fallback, while already singing: the mouth
                        # is either open enough or narrow enough
                        nv = (val > a.mouth_close
                              or (a.mouth_narrow > 0 and wd_ < a.mouth_narrow
                                  and val > 0.004))
                    else:
                        nv = val > a.mouth_open      # starting to sing always looks at the opening
                    if nv != M["on"] and a.gate_log:
                        print(f"[gate] {'open' if nv else 'close'}"
                              f"  p {M.get('sp', -1):.3f}"
                              f"  open {val:.3f}"
                              f"  t {time.time() - t0:.1f}s",
                              flush=True)
                    if nv and not M["on"]:
                        M["t_on"] = now  # v7 breath preparation: the instant the mouth opens is the cue
                    M["on"] = nv
                    if M["on"]:
                        M["last_open"] = now
                    # v19: with the fused model, the weights are computed per hop
                    # on the audio thread, since they need the current MFCC, and
                    # this only maintains the mouth-shape EMA for it to read.
                    if a.vowels and _VF_W is not None:
                        _rv = _ring_vec(
                            np.array([[p.x, p.y] for p in f]),
                            frame.shape[1] / frame.shape[0])
                        M["rv"] = (_rv if M.get("rv") is None else
                                   M["rv"] + 0.35 * (_rv - M["rv"]))
                    elif a.vowels and (_VM_W is not None
                                       or _VR_A is not None):
                        # v17.1 the full lip ring (2026-08-13): 40 points on the
                        # inner and outer lips into 80 dimensions. Vowels are
                        # separated by roundedness, the shape of the corners and
                        # the curvature of the lips, not by height and width alone;
                        # this is the answer to eh and ee overlapping at 0.37 in
                        # two dimensions.
                        _rv = _ring_vec(
                            np.array([[p.x, p.y] for p in f]),
                            frame.shape[1] / frame.shape[0])
                        M["rv"] = (_rv if M.get("rv") is None else
                                   M["rv"] + 0.35 * (_rv - M["rv"]))
                        if _VM_W is not None:
                            # v18: the personal model's class probabilities are
                            # the layer-mixing weights, five lines of plain numpy
                            # at about 30 microseconds, so the render loop still
                            # contains no machine learning.
                            _l = _VM_W @ ((M["rv"] - _VM_M) / _VM_S) + _VM_B
                            _e = np.exp(_l - _l.max())
                            M["vw"] = _sub(_e / _e.sum())
                        else:
                            _dl = np.abs((_VR_A - M["rv"]) / _VR_S).mean(1)
                            _w = 1.0 / (_dl * _dl + 1e-6)
                            M["vw"] = _sub(_w / _w.sum())
                    elif a.vowels and _VL_A is not None:
                        # v17 continuous vowel field, the 2D fallback: normalised
                        # L1 distances from the mouth shape (height, width) to the
                        # five anchors give inverse-square weights, so on an anchor
                        # it is nearly one-hot and between anchors it mixes
                        # continuously. No classifier, no hysteresis, no dwell: the
                        # discrete layer-selection paradigm was abandoned on
                        # 2026-08-13, because a mirror is always half a vowel
                        # late.
                        _wd = (abs(f[61].x - f[291].x)
                               / (abs(f[234].x - f[454].x) + 1e-9))
                        _hw = np.array([val, _wd])
                        M["hw"] = (_hw if M.get("hw") is None else
                                   M["hw"] + 0.35 * (_hw - M["hw"]))
                        _dl = np.abs((_VL_A - M["hw"]) / _VL_S).sum(1)
                        _w = 1.0 / (_dl * _dl + 1e-6)
                        M["vw"] = _sub(_w / _w.sum())
                    if a.view or a.frame_b64 > 0:
                        # v14 monitor: the inner lip line plus the numbers. The
                        # main thread calls imshow; this only draws.
                        h2 = int(frame.shape[0] * 480 / frame.shape[1])
                        fr2 = cv2.resize(frame, (480, h2))
                        # v29, after "a single line across the teeth looks odd,
                        # could it be a cross, or the ring of the lips?": draw
                        # **the whole lip ring plus a cross**. The vertical line is
                        # the opening height and the horizontal one the mouth
                        # width, and the gate really uses both, since from v28 the
                        # pucker hold reads the width. A single vertical line was a
                        # leftover from v14, when only opening and closing
                        # mattered.
                        def _pt(i):
                            return (int(f[i].x * 480), int(f[i].y * h2))
                        col = (0, 255, 0) if M["on"] else (0, 0, 255)
                        for _seg in (LIP_RING[:20], LIP_RING[20:]):
                            cv2.polylines(fr2,
                                          [np.array([_pt(i) for i in _seg],
                                                    np.int32)],
                                          True, col, 1, cv2.LINE_AA)
                        cv2.line(fr2, _pt(13), _pt(14), col, 2)    # opening
                        cv2.line(fr2, _pt(61), _pt(291), col, 2)   # mouth width
                        _wdv = M.get("wd", -1.0)
                        cv2.putText(fr2,
                                    f"FACE {'Y' if M.get('face') else 'N'} "
                                    + (f"sing {M['sp']:.2f} "
                                       if "sp" in M else
                                       f"jaw {M.get('jaw', 0):.2f} puck "
                                       f"{M['puc']:.2f} "
                                       if "jaw" in M else
                                       f"open {val:.3f} wide {_wdv:.3f} ") +
                                    f"{'ON' if M['on'] else 'off'}"
                                    f"  gate "
                                    f"{'OK' if M.get('eff', M['ok']) else 'X'}",
                                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (255, 255, 255), 2)
                        if a.vowels and M.get("vw") is not None:
                            # v17/v20: overlay the live layer weights, the vowel
                            # name and a percentage
                            cv2.putText(fr2, " ".join(
                                f"{nm_}{int(round(w * 100)):02d}"
                                for nm_, w in zip(_SEL_NAMES, M["vw"])),
                                (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1)
                        M["frame"] = fr2
            # Review A7: M["ok"] must be recomputed every pass, so a stalled
            # camera does not freeze it. v14.1, decided at acceptance: no face
            # means the singer is not there, so the gate **closes**. The earlier
            # "fail open after 3 s without a face" let an empty room be triggered
            # by feedback or noise. A camera that has really died, with the
            # heartbeat gone, is handled separately by the watchdog, which runs
            # bare and warns.
            M["ok"] = now - M["last_open"] < 0.8
            if not okf:
                time.sleep(0.01)
        cap.release()
    except Exception as e:
        M["ok"] = True
        M["dead"] = str(e)          # Review item 5: it used to print one line and
                                    # then run bare in silence for the whole session
        import traceback
        traceback.print_exc()
        print(f"⚠ [mouth] GATE DEAD ({e}) = running ungated",
              flush=True)


if a.mouth and not a.file:
    threading.Thread(target=_mouth_worker, daemon=True).start()


class VoicePlay:
    """Sample playback for one part: the current note, the crossfade on a note
    change, and the attack and release envelopes."""

    def __init__(self, nm, atk=None, rel=None, count=True, bank=None):
        self.nm = nm
        self.count = count               # v38: False marks an ensemble member, excluded from the forensic counts
        self.atk = atk if atk is not None else a.attack * self.ATK_MULT.get(nm, 1.0)
        self.rel = rel if rel is not None else a.release
        self.bank = BANKS[nm] if bank is None else bank   # v38: a member may use another singer
        self.cur = None                  # (midi, buf, pos)
        self.old = None                  # the previous note, fading out through a change
        self.env = 0.0
        self.tgt = 0.0
        self.gl = 0.0                    # v15 porta: the starting semitone offset
        self.glt = 0                     # total glide length, in samples
        self.glr = 0                     # glide remaining, in samples
        self.xfn = int(0.03 * SR)
        self.xin = 0                     # fade-in remaining on a note change (review A2)
        self.vw = _VW_MID                # v17: the previous hop's layer weights, for interpolating the mix

    def set_note(self, m):
        if m is None:
            self.tgt = 0.0
            return
        self.tgt = 1.0
        # Review A1: the clip must come before the comparison, or a target note
        # outside the range retriggers every hop. Above 68 the bass once entered a
        # 28.6 Hz restart loop, a comb-like drone and the largest source of
        # bursts.
        lo_b, hi_b = min(self.bank), max(self.bank)
        if a.fold:
            # Outside the range, **fold by octaves** rather than clip: clipping
            # also changes the pitch class, and measured, 19-26% of pad notes
            # became non-chord tones, the soprano pad stuck on F3 while the singer
            # was low, and the tenor once went from 83 to 68. Folding keeps the
            # pitch class and changes only the octave.
            while m < lo_b:
                m += 12
            while m > hi_b:
                m -= 12
        m = int(np.clip(m, lo_b, hi_b))
        if self.cur is not None and self.cur[0] == m:
            return
        if self.cur is not None:
            self.old = [self.cur, self.xfn]        # [(midi, buf, pos), remaining]
            self.xin = self.xfn                    # review A2: the new note fades in
            self.xlin = False
            if (a.porta > 0 and abs(m - self.cur[0]) <= 2
                    and st.get("porta_ok", True)):
                # v15: glide only if the previous note was held long enough, 0.3 s
                # or more, which makes it an expressive glide; a fast phrase cuts
                # cleanly.
                self.gl = float(self.cur[0] - m)
                self.glt = self.glr = max(1, int(a.porta * SR))
                if self.count:      # v38: only the lead counts. If members counted
                    #   too, the glide count would be multiplied by the number of
                    #   people (measured, --per-part 4 took it from 53 to 209),
                    #   and that number is the ruler for the glide-rate forensics
                    #   (the 46.6% to 68.9% of v26 was measured with it), so the
                    #   instrument would be quietly contaminated.
                    st["porta_n"] = st.get("porta_n", 0) + 1   # v25 instrument
                self.xlin = True   # R4A#12: porta starts from the same fundamental,
                #   so the signals are coherent and an equal-power join bumps
                #   (measured sigma 2.19 dB); use a linear join instead.
        # v17: the five layers are stacked as (5, L). At a given midi note every
        # layer's loop is the same length, so the samples align; the length depends
        # only on f0, see _render_layer. The mixing happens per sample in render
        # and no layer is selected here, so v16's boundary of "do not change layer
        # during a sustained note" disappears along with the paradigm. Every part
        # follows the field; the alto-only isolation of v16.3b was an experiment
        # under the dead paradigm and is not kept. Range clipping still uses
        # self.bank, since the layer banks share lo and hi.
        buf = (np.stack([LBANKS[self.nm][li][m] for li in _LSEL])
               if a.vowels else self.bank[m])
        self.cur = [m, buf, 0.0]

    ATK_MULT = {"bass": 0.6, "alto": 1.0, "sop": 1.6}   # v5 staggered attack:
    #   the three parts have different attack times, so the entry unfolds rather
    #   than arriving as a wall (the simple form of the v6 item 2 stagger)

    def render(self, n, rate):
        out = np.zeros(n, dtype="float32")
        if a.vowels:
            # v17: this hop's per-sample layer weights interpolate linearly from
            # the previous hop's weights to the latest, which kills the zipper
            # stepping of a 30 Hz weight update. cur and old share one set, so both
            # ends of a note-change crossfade hear the same voice. With no value
            # from the camera it pins to a one-hot on the middle layer.
            w1 = M.get("vw")
            if w1 is None:
                w1 = _VW_MID
            g5 = np.linspace(self.vw, w1, n, dtype="float32")   # (n,5)
            self.vw = w1
        else:
            g5 = None
        if self.cur is not None and (self.env > 1e-4 or self.tgt > 0):
            if self.glr > 0:
                # v15 linear glide of fixed length: it **arrives exactly** after
                # porta seconds. The earlier exponential tail was always still on
                # its way, which was one of the main reasons normal singing speed
                # turned into constant portamento.
                # v25 (review S1): a single rate used to be applied to a whole
                # block, so **the glide was a staircase rather than a ramp**. With
                # porta 0.08 and a hop of 0.035, the first 35 ms block stayed
                # entirely on the old pitch and the remaining three steps were up
                # to about 90 cents each. It now interpolates per sample within a
                # block, and _read accepts a per-sample rate array.
                r0 = self.glr / self.glt
                r1 = max(0, self.glr - n) / self.glt
                g = 2 ** (self.gl * np.linspace(r0, r1, n, endpoint=False,
                                                dtype="float64") / 12.0)
                self.glr = max(0, self.glr - n)
            else:
                g = 1.0
            w = self._read(self.cur, n, rate * g, g5)
            if self.xin > 0:
                # Review A2: a complementary crossfade, with the new note fading
                # in linearly and the old one out, keeps the total amplitude
                # roughly constant through a change. It used to start with the new
                # note at full amplitude on top of the old, a +6 dB overshoot.
                k = min(n, self.xin)
                w = w.copy()
                ramp = np.linspace(1 - self.xin / self.xfn,
                                   1 - (self.xin - k) / self.xfn,
                                   k, dtype="float32")
                w[:k] *= ramp if getattr(self, "xlin", False) \
                    else np.sqrt(ramp)
                self.xin -= k
            out += w

        if self.old is not None:
            w = self._read(self.old[0], n, rate, g5)
            k = min(n, self.old[1])
            framp = np.linspace(self.old[1] / self.xfn,
                                (self.old[1] - k) / self.xfn,
                                k).astype("float32")
            fade = framp if getattr(self, "xlin", False) else np.sqrt(framp)
            out[:k] += w[:k] * fade    # equal power, or linear while porta keeps them coherent
            self.old[1] -= k
            if self.old[1] <= 0:
                self.old = None
        # attack and release envelopes, linear within a block
        aatk = n / (self.atk * SR)
        arel = n / (self.rel * SR)
        e0 = self.env
        e1 = (min(1.0, e0 + aatk) if self.tgt > 0 else max(0.0, e0 - arel))
        self.env = e1
        return out * np.linspace(e0, e1, n, dtype="float32")

    def _read(self, slot, n, rate, g5=None):
        """rate may be a scalar or **a per-sample array** (the v25 glide varies
        the rate sample by sample within a block)."""
        _m, buf, pos = slot
        L = buf.shape[-1]
        if np.ndim(rate) == 0:
            idx = (pos + np.arange(n, dtype="float64") * rate) % L
            end = pos + n * rate
        else:
            # The displacement is the cumulative sum of the rate: idx[0] is pos,
            # and each later sample has its own rate.
            c = np.cumsum(rate, dtype="float64")
            idx = (pos + c - rate[0]) % L
            end = pos + c[-1]
        i0 = idx.astype(np.int64)
        fr = (idx - i0).astype("float32")
        i1 = (i0 + 1) % L
        slot[2] = float(end % L)
        if buf.ndim == 1:
            return buf[i0] * (1 - fr) + buf[i1] * fr
        # v17: read the five layers in phase as (5, n) and mix them with
        # per-sample weights. The layers are the same length so one cursor serves
        # all of them; they were rendered from the same seed at the same f0, so how
        # coherent the cross-layer mix sounds is still to be judged by ear.
        y5 = buf[:, i0] * (1 - fr) + buf[:, i1] * fr
        return (y5 * g5.T).sum(axis=0)


players = [VoicePlay(nm) for nm, *_x in VOICES]
# v7 positions (a light version of component four): in the fast layer the bass
# sits centre-left, the alto right and the soprano left, with the pad pulled to the
# outside.
# Positions expand with the number of parts; they used to be hard-coded for three,
# and the tenor is inserted to the right of the bass.
_PANF = {"bass": -0.15, "tenor": 0.15, "alto": 0.35, "sop": -0.35}
_PANP = {"bass": -0.6, "tenor": 0.3, "alto": 0.6, "sop": 0.15}
PAN_F = [_PANF[nm] for nm, *_x in VOICES]
PAN_P = [_PANP[nm] for nm, *_x in VOICES]


def _pan(p_):
    th = (p_ + 1) * np.pi / 4
    return np.float32(np.cos(th)), np.float32(np.sin(th))
# v6 slow layer: the same banks with long attack and release, 0.8 s to open and
# 1.2 s to close, which makes a chord pad that outlives your phrase.
_PADATK = {"bass": 0.8, "tenor": 0.85, "alto": 0.9, "sop": 1.0}
pads = ([VoicePlay(nm, atk=_PADATK[nm], rel=0.9)
         for nm, *_x in VOICES] if a.pad > 0 else [])
# v3 human quality: two slow LFOs per part, out of phase with each other, sum to
# an independent drift of plus or minus human cents.
_rng = np.random.RandomState(20260811)
HUM = [( _rng.uniform(0.06, 0.16), _rng.uniform(0, 6.28),
         _rng.uniform(0.15, 0.3), _rng.uniform(0, 6.28))
       for _ in range(2 * len(VOICES))]
VIBPH = [_rng.uniform(0, 6.28) for _ in range(2 * len(VOICES))]
# v38 several singers per part (--per-part). **The members are drawn after
# VIBPH**, so the existing HUM and VIBPH values above do not shift because this
# flag was enabled: --per-part 1 must stay byte-identical, and the first member of
# a four-person version must sound like the same person as the one-person
# version.
PP = int(a.per_part)
MEMG = np.float32(1.0 / np.sqrt(PP))     # at PP=1 this is exactly 1.0, so the multiplication is a no-op
MEMS = [[] for _ in VOICES]
if PP > 1:
    _mrng = np.random.RandomState(20260814)
    _real = 0
    for _vi, (_nm, _mdl, _sp, *_x) in enumerate(VOICES):
        # Members take **a genuinely different singer** first, another spk of the
        # same model, and fall back to a detuned copy of the lead only once those
        # run out. From 2026-08-14: "I would rather it used a different timbre."
        # v39: that column holds (model, spk), so the key is a triple.
        # v43 --mem-real 0 makes every member a copy of the lead's bank, the v38
        # behaviour, borrowing nobody from another model. It was judged by ear on
        # 2026-08-16 that real timbres and detuned copies could not be told apart,
        # so removing it costs nothing musically, while it is one of the sources of
        # roughness: laying the same note from different recordings on top of each
        # other always leaves a slight pitch difference.
        # v44 (review #6, 2026-08-17): **the lead's own (model, spk) may not be
        # borrowed**. After v42 moved the tenor lead to male8 spk7, the "other
        # person" on the list became the same person as the lead, so a member was a
        # bit-for-bit copy of the lead (compared identical in the npz) plus
        # detuning, which brought the interference noise just condemned back in its
        # worst form, while the start-up banner still reported it as a DIFFERENT
        # singer.
        _avail = ([ms for ms in MEM_SPK.get(_nm, [])
                   if (_nm, *ms) in XBANKS and ms != (_mdl, _sp)]
                  if a.mem_real else [])
        for _k in range(PP - 1):
            _dly = int(_mrng.uniform(0, 2 * a.spread_ms * 1e-3) * SR)
            _spk = _avail[_k] if _k < len(_avail) else None
            if _spk is not None:
                _real += 1
            MEMS[_vi].append({
                # A borrowed bank really is another person; without one it shares
                # BANKS[nm], with no re-rendering, and pretends through detuning
                # and staggering. count=False keeps it out of the forensic
                # counts.
                "p": VoicePlay(_nm, count=False,
                               bank=(XBANKS[(_nm, *_spk)] if _spk is not None
                                     else None)),
                "spk": _spk,
                # Static detuning: this is how far apart the members of a real
                # ensemble are (F3).
                "det": 2.0 ** (float(_mrng.normal(0.0, a.spread_cents)) / 1200.0),
                "hum": (_mrng.uniform(0.06, 0.16), _mrng.uniform(0, 6.28),
                        _mrng.uniform(0.15, 0.3), _mrng.uniform(0, 6.28)),
                # v43 test (2026-08-16): `--vib-sync 1` makes the members share
                # **one vibrato phase** with the lead. The hypothesis: when N
                # copies of the same note each have a random phase, their
                # instantaneous frequencies are offset from one another and, on the
                # high harmonics, fall into the 15-300 Hz difference band, which is
                # psychoacoustic roughness. That is the thing that is audible but
                # invisible to a spectrum, because it is not a new component but
                # interference between existing ones. A shared phase means they
                # move together and there is no relative frequency difference.
                # This random number is drawn on both paths regardless, so the
                # random sequence does not shift; otherwise det, hum and pan would
                # all move with it and this would not be a single variable.
                "vib": (lambda _p: VIBPH[_vi] if a.vib_sync else _p)(
                    _mrng.uniform(0, 6.28)),
                "pan": float(np.clip(
                    PAN_F[_vi] + _mrng.uniform(-a.spread_pan, a.spread_pan),
                    -1.0, 1.0)),
                # The staggered attack is a delay line. It also reads the same
                # loop at different phases, which does most of the decorrelation:
                # with detuning alone, two identical loops only beat against each
                # other and never become two people.
                "dly": _dly,
                "tail": np.zeros(_dly, dtype="float32"),
            })
    print(f"[choir] per-part {PP} → {len(VOICES) * PP} voices "
          f"({_real} of them a DIFFERENT singer, the rest detuned copies; "
          f"detune sd {a.spread_cents:.0f}c, onset 0-{2*a.spread_ms:.0f}ms, "
          f"pan ±{a.spread_pan:.2f}, gain 1/√{PP}={float(MEMG):.3f})",
          flush=True)


def _mdelay(mm, w):
    """A member's fixed delay line, giving the staggered attack and decorrelating
    the loop phase."""
    if mm["dly"] == 0:
        return w
    buf = np.concatenate((mm["tail"], w))
    mm["tail"] = buf[len(w):]
    return buf[:len(w)]
RVIR, RVT = None, None
if a.wet > 0:
    import reverb as rv
    from scipy.signal import fftconvolve as _fftc
    RVIR = rv.make_ir(2.4, trim_db=35.0)
    RVT = [np.zeros(len(RVIR) - 1)]
    print(f"[wet] IR {len(RVIR)/SR:.1f}s, send {a.wet}", flush=True)
dmp = ({"mic": [], "out": [], "mok": [], "f0": [], "face": [], "mon": [],
        "note": [], "vw": [], "val": [], "wd": [], "sp": [], "vn": []}
       if a.dump else None)     # v14.2: the full set of forensic channels (face, mouth, gate, detection, note)
#                                v17 adds vw, the five layer weights per hop,
#                                stored only when vowels is on
DMPMAX = max(1, int(a.dump_max_min * 60 / a.hop))   # v44 ceiling, in hops


def _write_dump():
    """Write the dump, shared by the live shutdown and --file. **Live it must be
    called before the stream is closed**, inside the stream context: CoreAudio's
    FinishStoppingStream can hang, a known playbook issue, and it hangs inside
    __exit__, so a dump written after that is lost to the shell's 8 s SIGKILL along
    with the whole recording (review #13, 2026-08-17; respond2 learned the same
    lesson on 2026-08-04, see the comment in its finally block).
    The length is taken from a snapshot of mic, which is safe while the worker is
    still appending, and all channels are trimmed to match."""
    if dmp is None or not dmp["mic"]:
        return
    k = len(dmp["mic"])
    sf.write(a.dump + "_mic.wav", np.concatenate(dmp["mic"][:k]), SR)
    sf.write(a.dump + "_out.wav", np.concatenate(dmp["out"][:k]), SR)
    np.savez(a.dump + "_st.npz", mok=np.array(dmp["mok"][:k]),
             f0=np.array(dmp["f0"][:k]), face=np.array(dmp["face"][:k]),
             mon=np.array(dmp["mon"][:k]), note=np.array(dmp["note"][:k]),
             val=np.array(dmp["val"][:k]), wd=np.array(dmp["wd"][:k]),
             sp=np.array(dmp["sp"][:k]), vn=np.array(dmp["vn"][:k]),
             hop=a.hop,              # R5#2: the offline forensic workflow has to be runnable
             **({"vw": np.array(dmp["vw"][:k])} if a.vowels else {}))
    print(f"\ndump: {a.dump}_mic/_out.wav + _st.npz", flush=True)
in_q, out_q = [], []
qlock = threading.Lock()


LM = None
if a.predict and os.path.exists(a.predict):
    import json as _json
    _lm = _json.load(open(a.predict))
    LM = {"tri": {tuple(map(int, k.split(","))): v
                  for k, v in _lm["tri"].items()},
          "bi": {int(k): v for k, v in _lm["bi"].items()},
          "uni": _lm["uni"]}
    print(f"[habit] melody_lm loaded ({len(LM['tri'])} trigrams)", flush=True)


def lm_top3(iv1, iv2):
    """The last two intervals to the model's top three next intervals, with
    backoff."""
    sc = {}
    t = LM["tri"].get((iv1, iv2))
    b = LM["bi"].get(iv2)
    for v in range(-12, 13):
        sc[v] = ((t.get(str(v), 0.0) * 100 if t else 0.0)
                 + (b.get(str(v), 0.0) * 10 if b else 0.0)
                 + LM["uni"].get(str(v), 0.0) * 0.01)
    return sorted(sc, key=sc.get, reverse=True)[:3]


_tick = [np.zeros(hopN * 2, dtype="float32")]


_OLV = [-120.0] * 8          # v33: the output level over the last 8 hops; the acoustic delay is
                             # about 5 hops


def process_hop(chunk):
    """The complete pipeline for one hop, from detection through the decision and
    the render to the wet layer, producing output samples.
    Shared by the live worker and the --file bench, so conclusions verified offline
    hold live."""
    _tick[0] = np.concatenate([_tick[0][hopN:], chunk])
    f0 = f0_autocorr(_tick[0][-hopN * 2:])
    f0_raw = f0                          # R4A#2: the forensics record the detection **before** the gate
    _lvl = 20 * np.log10(float(np.sqrt((chunk ** 2).mean())) + 1e-12)
    if f0 > 0 and _lvl < a.min_level:
        f0 = 0.0                         # v33 absolute level gate: room noise does not count as singing
    if f0 > 0 and a.bleed < 0:
        # v33 bleed gate: the parts' output about 164 ms ago plus the coupling is
        # how much of them is in the microphone now.
        if _lvl < _OLV[-5] + a.bleed + a.bleed_margin:
            f0 = 0.0
    mok = M["ok"]
    if a.mic_gate:
        # v43 microphone gate: authority moves from the camera to the microphone
        # (2026-08-18). The floor is **the maximum** output level over the last 8
        # hops, rather than the single point _OLV[-5] of the v33 bleed gate: F26
        # measured the feedback peak at +279 ms, about 8 hops at 35 ms each, and a
        # single point gambles on the delay estimate while the window maximum only
        # ever blocks more, never less. Opening and closing both count hops rather
        # than wall-clock time, so the --file bench reaches the same conclusions
        # however fast it runs; the promise at the top of this file that offline
        # verification holds live rests on that.
        _fl = a.mic_open
        if a.bleed < 0:
            _fl = max(_fl, max(_OLV) + a.bleed + a.bleed_margin)
        if _lvl > _fl:
            M["marm"] = M.get("marm", 0) + 1
            if M["marm"] >= a.mic_frames:
                M["mrun"] = 0
        else:
            M["marm"] = 0
            M["mrun"] = M.get("mrun", 10 ** 6) + 1
        mok = M.get("mrun", 10 ** 6) * a.hop < a.mic_hold
        if a.gate_log and mok != M.get("eff"):
            print(f"[gate] mic {'open' if mok else 'close'}"
                  f"  lvl {_lvl:.1f}  floor {_fl:.1f}", flush=True)
    elif a.mouth and not a.file:
        # R3#2 watchdog: if the camera worker's heartbeat stops, because cap.read
        # is blocking, it is still starting, or it has died, fail open and run bare
        # with sound rather than silence. The original illness was 2-5 s of silence
        # at the start and total silence for a session if the camera stalled.
        # Review: `M.get("hb", 0.0)` made **the start always look like a dead
        # camera**, so the gate ran bare for 3-6 seconds, during camera selection
        # and mediapipe initialisation, and it would sing in the gap before going
        # on stage. It now treats "no heartbeat ever" as "not ready yet, keep the
        # gate closed", and only enters the watchdog logic once one has arrived.
        if "hb" not in M:
            mok = False
            if not st.get("bootwarn"):
                st["bootwarn"] = True
                print("[mouth] warming up: gate closed until camera is live",
                      flush=True)
            alive = True
        else:
            alive = time.time() - M["hb"] < 0.85   # R5#1: aligned with the mouth
        #   grace period of 0.8 s, which narrows the silent gap between the
        #   watchdog and the gate to about 50 ms
        if "hb" in M and not alive:
            mok = True
            if a.vowels:
                # Review A3: a dead camera means the mouth data is stale, so the
                # vowel field returns to the middle layer, which the per-hop
                # interpolation in render slides to smoothly, rather than freezing
                # on the last mouth shape.
                M["vw"] = _VW_MID
            if not st.get("camwarn"):
                st["camwarn"] = True
                # Review #12, 2026-08-17: this path used to run bare in silence,
                # printing one small line without setting M["dead"], so the status
                # bar never showed GATE DEAD and the shell's large warning bar
                # could not catch it, while a stalled USB camera is **the most
                # likely** camera failure. It now uses the same format as the crash
                # path, the except in _mouth_worker, so both are equally visible.
                M["dead"] = "camera heartbeat lost"
                print("⚠ [mouth] GATE DEAD (camera heartbeat lost) = running "
                      "ungated (feedback guard off)", flush=True)
        elif st.get("camwarn"):
            st["camwarn"] = False
            if M.get("dead") == "camera heartbeat lost":
                del M["dead"]                # the heartbeat is back, so it clears;
                                             # the crash path does not clear
            print("[mouth] camera back = gate online again", flush=True)
    M["eff"] = mok                       # R4A#2: the gate value in force, for the
                                         # status bar and the view
    if a.vowels and _VF_W is not None:
        _ABUF[0] = np.concatenate([_ABUF[0], chunk])[-_FN:]   # appended every hop
    if not mok:
        # Reviews A4 and A6 condemned v9's "admit a new pitch": a bass unison,
        # which is the singer's own note, and a pad chord tone, the thirds and
        # fifths they sing most often, were both judged not novel, which instead
        # produced a stumble every 0.5 s and made the reaction time depend on
        # whether a chord happened to be in the way. It returns to the v2 meaning:
        # a closed mouth never counts as you, and the intermittency is fixed at the
        # camera end instead, through faster polling, thresholds and the freeze
        # fix.
        f0 = 0.0
    if a.vowels and _VF_W is not None:
        # v19 fused field: the mouth shape, from the camera EMA, plus the current
        # MFCC give probabilities that serve as the layer-mixing weights.
        # **Where this sits matters**: after the gate. With the mouth closed, f0 is
        # already zero and nothing updates, or bleed from the parts would change
        # the vowel by itself, which is what killed v16.2; this time it is blocked
        # structurally.
        # While not singing, the weights stay on the last vowel, which is better
        # than chasing rubbish, and the per-hop interpolation in render smooths it.
        # With the camera gone (camwarn) it likewise does not update, so the middle
        # layer set by the watchdog holds.
        if f0 > 0 and M.get("rv") is not None and not st.get("camwarn"):
            _P = np.abs(np.fft.rfft(_ABUF[0] * _FWIN)) ** 2
            _c = _FDCT @ np.log(_P @ _FFB.T + 1e-10)
            _x = (np.concatenate([M["rv"], _c]) - _VF_M) / _VF_S
            _l = _VF_W @ _x + _VF_B
            _e = np.exp(_l - _l.max())
            _vw = _sub(_e / _e.sum())
            if a.vw_tau > 0 and M.get("vw") is not None:
                # v20: smooth the weights themselves (--vw-tau), which is the
                # direct control for a wandering timbre
                _k = 1.0 - float(np.exp(-a.hop / a.vw_tau))
                _vw = (M["vw"] + _k * (_vw - M["vw"])).astype("float32")
            M["vw"] = _vw
    if f0 > 0:
        mf = 69 + 12 * np.log2(f0 / 440.0)
        m = int(round(mf))
        # v26 removed v24's pitch hysteresis dead zone **entirely**. Three fresh
        # reviews each condemned it: (1) the mechanism that made it work was
        # accidental, since clamped frames were dropped by the plausibility gate
        # downstream, so the candidate votes were never cleared and votes could
        # later accumulate onto a note grazed half a second earlier; (2) above
        # hys 0.5 the escape condition and the plausibility gate no longer
        # intersect, so **the note just left becomes unreachable and it sticks for
        # good**, measured at 5 s without moving; (3) at hys 0.55 it silently
        # dropped notes, three per pass; (4) it pushed note changes past 0.3 s and
        # took the glide trigger rate from 46.6% to 68.9%, which runs straight into
        # "anything auto-tuned is simply out".
        # In its place: (1) the young condition on need and the pre-empt below,
        # requiring more votes for a note just changed to; and (2) the clamp in
        # this section, which is **the dead zone done correctly**: a clamped frame
        # says outright that they are still on the original note, taking the
        # m == st["note"] branch below and clearing the candidate votes, rather
        # than relying on the plausibility gate to discard the frame. The dead zone
        # is also capped at 0.5 semitones, so the escape threshold is at most 1.0,
        # the centre of the neighbouring note, which is always reachable and makes
        # **sticking and silent dropping structurally impossible** (the cure for S2
        # and S3).
        # Extra votes on a young note stop chatter at the rate of vibrato, but
        # their back-and-forth holds 200-400 ms on each side, which is sustained
        # evidence that votes cannot stop, so this pitch memory is needed.
        clamp = (a.deadzone > 0 and st["note"] is not None
                 and m != st["note"]
                 and abs(mf - st["note"]) < 0.5 + a.deadzone)
        if clamp:
            m = st["note"]
        # v11 rejection of notes already sounding applies only to **entering after
        # a real pause**. The v10 version, which blocked anything more than 6
        # semitones outside a sounding note at any time, was convicted by dump
        # forensics on 2026-08-12: entering on a leap of a fifth or an octave onto
        # a pad or fast-layer note still ringing meant being swallowed, and the
        # break lasted exactly as long as the pad (a measured 2.45 s break against a
        # pad lifetime of 2.4 s).
        # A leap is music, not feedback, so the rejection narrows to the only real
        # scenario in which feedback is captured: they have stopped for 0.5 s or
        # more, the mouth is still open inside the grace period, and the parts'
        # reverb is still sounding. In that state any detection sitting on a
        # sounding note is treated as suspect, near or far.
        # This block is **not dead code**: the mf and m computed above are inputs
        # to the state machine downstream, and only the rejection logic was removed
        # (R3). Deleting the whole block on the strength of the comment gives a
        # NameError.
        # The R3 judgement: the layer that rejected notes already sounding was
        # **removed entirely**. Measured, polyphonic feedback makes the
        # autocorrelation lock onto a phantom note at the chord's common period,
        # for example 38.5 detected while 48, 64 and 67 are sounding, which is more
        # than 0.4 semitones from any of them. So with a real arrangement this
        # layer almost never fires, making it false protection, and its only
        # non-redundant window is the 0.45 s between 0.35 and 0.8 s after the mouth
        # closes, while it brought a fail-open pumping regression with it (R3#1).
        # The real defence is the mouth gate, and a dead camera runs bare honestly,
        # through the watchdog below.
        pass
    if f0 > 0:
        st["quiet"] = 0
        # R4B plausibility gate: a detection more than 0.35 from the semitone grid
        # is a gliding frame in a note transition and does not vote. This cures
        # false neighbouring notes, where 4 of 6 whole-tone changes produced a
        # 70-100 ms false intermediate note, and false octave commits. kt and cents
        # are not contaminated by transitions, and vibrato peaks of plus or minus
        # 60 cents no longer graze the edge.
        plaus = abs(mf - m) <= 0.35 or clamp
        if plaus and not clamp:
            # A clamped frame means their pitch is between two grid points, and
            # neither cents nor the key takes that contamination. Review B1 noted
            # "the dead zone freezes cents"; here it is deliberate, since a
            # transition frame never votes anyway.
            kt.push(mf)
            # Chord lock: their cents deviation from the quantised note, through
            # an EMA, drives the playback rate.
            st["cents"] += 0.25 * ((mf - m) * 100 - st["cents"])
            st["lastm"] = m
        if not plaus:
            pass                         # a transition frame: it neither votes nor clears votes
        elif m == st["note"]:
            st["cand"], st["cc"] = None, 0
        elif m == st["cand"]:
            st["cc"] += 1
            tno = st.get("t", 0.0)
            # R2A#6: the refractory period covers the two-vote path as well, and a
            # vibrato pair, going back and forth between A and B, needs 4 votes.
            need = 2
            if m == st.get("pn") and tno - st["tsw"] < a.rebound:
                need = 4                 # v15: only a very fast back-jump counts as vibrato chatter
            elif st["note"] is not None and abs(m - st["note"]) > 7:
                need = 3                 # R5#3: a leap needs one more vote, which kills the false-octave
                #   transient (R4B measured a false 52 lasting 105 ms) while a real
                #   octave is only about 35 ms slower
            if a.young > 0 and tno - st["tsw"] < a.young:
                need = max(need, 2 + a.young_need)   # v26: extra resistance on a young note
            if st["cc"] >= need and tno - st["tsw"] > 0.10:
                if st["note"] is not None:
                    st["hist"] = (st["hist"] + [m - st["note"]])[-4:]
                st["porta_ok"] = tno - st["tsw"] >= 0.3   # v15
                st["pn"] = st["note"]
                st["note"] = m
                st["tsw"] = tno
                st["cand"], st["cc"] = None, 0
        else:
            st["cand"], st["cc"] = m, 1
            breath = (not a.file and a.mouth
                      and time.time() - M["t_on"] < 1.0)
            hab = False
            if (LM is not None and st["note"] is not None
                    and len(st["hist"]) >= 2):
                hab = (m - st["note"]) in lm_top3(st["hist"][-2],
                                                  st["hist"][-1])
            cool = st.get("t", 0.0) - st["tsw"] > 0.10
            # Review B4: a step is strong evidence in itself and is no longer tied
            # to being in key. Before KeyTracker warms up, root is 0, so every song
            # not in C was wrongly blocked. The in-key test is kept only for
            # entering on a breath, and it passes while the key is undecided.
            keyok = (kt.h.sum() < kt.MIN
                     or (m - kt.root()) % 12 in MAJ)
            rebound = (m == st.get("pn")
                       and st.get("t", 0.0) - st["tsw"] < a.rebound)
            young = (a.young > 0 and st["note"] is not None
                     and st.get("t", 0.0) - st["tsw"] < a.young)
            if (a.fast and cool and not rebound   # v15: block only a very fast back-jump
                    and not young                 # v26: a young note does not get the one-vote channel
                    and ((st["note"] is not None
                          and abs(m - st["note"]) <= 2)
                         # Review A3: hab used to **bypass the false-octave
                         # protection**. In 293 of melody_lm's contexts, 50% of the
                         # top three contain a leap over 7 semitones, so a
                         # single-frame false octave of 105 ms that met the habit
                         # model committed in 35 ms. The pre-empt is now limited to
                         # leaps within a fifth, and anything larger returns to the
                         # slow path and pays R5#3's three votes.
                         or (hab and abs(m - st["note"]) <= 7)
                         or (breath and st["note"] is None and keyok))):
                # v6 pre-empt: commit on one vote, about 35 ms, with a 150 ms
                # refractory period against a vibrato storm.
                if st["note"] is not None:
                    st["hist"] = (st["hist"] + [m - st["note"]])[-4:]
                st["porta_ok"] = (st.get("t", 0.0) - st["tsw"]) >= 0.3  # v15
                st["pn"] = st["note"]
                st["note"] = m
                st["tsw"] = st.get("t", 0.0)   # review A3/B11
                st["cand"], st["cc"] = None, 0
    else:
        st["quiet"] += 1
        st["cents"] *= 0.9               # Review A13: the tuning deviation decays
        #                                  to zero in silence, so the previous
        #                                  note's cents do not hang on the pad
        if st["quiet"] >= max(1, int(0.25 / a.hop)):
            st["note"] = None            # about 0.25 s of silence releases it
            st["cand"], st["cc"] = None, 0   # review B7: clear any leftover candidate too
    root = kt.root()
    n = st["note"]
    for vi, (p, (_nm, _md, _sp, _g, _rng, sh, th)) in enumerate(
            zip(players, VOICES)):
        if n is None:
            p.set_note(None)
        elif a.vl >= 2 and vi > 0:
            # v36 choral arrangement (--vl 2): the older version gave the three
            # upper parts **one shared candidate set and had each take the nearest
            # note to its own previous one**, so they ratcheted upwards together
            # and locked onto the highest candidate. Measured, the upper three sang
            # the same note 95% of the time, at +19 semitones, so four parts were
            # three in unison.
            # In the new version each part works separately: (1) the candidates are
            # every octave of a chord tone **within its own range**; (2) the score
            # is voice motion (|delta|) plus a pull towards the centre of the range
            # plus a unison penalty plus a crossing penalty; (3) they are assigned
            # from low to high, with the notes already settled acting as a floor
            # that must not be crossed.
            lo_, hi_ = _rng
            ctr = (lo_ + hi_) / 2.0
            offs = (0, dia_step(n, root, 2), dia_step(n, root, 4))
            cands = [n + o + 12 * k for o in offs for k in range(-2, 4)]
            cands = [c for c in cands if lo_ <= c <= hi_]
            if not cands:
                cands = [int(np.clip(n + sh, lo_, hi_))]
            prev = p.cur[0] if p.cur is not None else ctr
            floor_ = st.get("vlfloor", None)

            def _score(c):
                v = abs(c - prev) + 0.6 * abs(c - ctr)
                if floor_ is not None:
                    if c == floor_:
                        v += 8.0          # a unison must cost something
                    elif c < floor_:
                        v += 14.0         # voice crossing
                return v
            pick = min(cands, key=_score)
            st["vlfloor"] = pick
            p.set_note(pick)
        elif a.vl and vi > 0:
            # v4 voice leading: the bass always takes the root, and an upper part
            # takes whichever candidate a third, a fifth or an octave away is
            # nearest its own previous note, which gives small steps. Landing on a
            # unison doubles the note, which choirs do anyway.
            third = dia_step(n, root, 2)
            fifth = dia_step(n, root, 4)
            cands = [n + third, n + fifth, n + 12,
                     n + 12 + third, n + 12 + fifth]
            prev = (p.cur[0] if p.cur is not None else
                    n + sh + (dia_third(n + sh, root) if th else 0))
            p.set_note(min(cands, key=lambda c: abs(c - prev)))
        else:
            t = n + sh + (dia_third(n + sh, root) if th else 0)
            p.set_note(t)
        if vi == 0:
            # Review #5, 2026-08-17: the floor records the note the bass
            # **actually sounds**, that is, p.cur after set_note, and not the
            # pre-fold n+sh. set_note folds any target below the bank's lower limit
            # up by an octave, which happens for n <= 43 once the bass is at -8.
            # Recording the pre-fold value under-reports the floor by 12 semitones,
            # so the tenor's unison (+8) and crossing (+14) penalties stop working
            # entirely for the lowest notes and the tenor can be assigned below the
            # bass's real pitch, a regression that was impossible while the shift
            # was 0.
            st["vlfloor"] = (p.cur[0]
                             if (n is not None and p.cur is not None) else None)
    if PP > 1:
        # Members **follow the lead of their own part** and never choose a note
        # themselves. Voice leading scores against p.cur as the previous note, with
        # a unison penalty of 8 and a crossing penalty of 14, so running it once
        # per member would let four members ratchet onto four different notes and
        # one part would scatter into four. In a real choir a division is also one
        # line sung by several people, not each person choosing.
        # Use p.tgt to test whether a note is present: p.cur still holds the
        # previous note after a release.
        for vi, p in enumerate(players):
            m_ = p.cur[0] if (p.cur is not None and p.tgt > 0) else None
            for mm in MEMS[vi]:
                mm["p"].set_note(m_)      # the same note makes set_note return at once, without retriggering
    if pads:
        tmono = st.get("t", 0.0)
        if n is not None and st["pada"] != n and \
                tmono - st["padt"] >= a.pad:
            # v6 slow layer: the anchor note changes the chord, with the dwell
            # limit acting as the harmonic rhythm; the third and fifth follow the
            # key.
            st["pada"], st["padt"] = n, tmono
            r_ = kt.root()
            # Review: the pads are built from the number of parts, which may be 4
            # from v31, while a chord has only 3 notes, so zip **truncates
            # silently** and the last part's pad is mute throughout. The chord is
            # expanded to the number of pads, with the fourth note taking the root
            # an octave up.
            chord = [n - 12, n + dia_step(n, r_, 2), n + dia_step(n, r_, 4),
                     n][:len(pads)]
            for pp, cnote in zip(pads, chord):
                pp.set_note(cnote)
        elif n is None and st["quiet"] >= max(1, int(a.pad_hold / a.hop)):
            st["pada"] = None
            for pp in pads:
                pp.set_note(None)
    rate = 2 ** (a.lock * st["cents"] / 1200.0)
    # Review B: the forensic weights are sampled **before** the render, so what is
    # recorded is the field this hop actually used.
    vw_h = (np.asarray(M.get("vw", _VW_MID)).copy()
            if a.vowels and dmp is not None else None)
    y = np.zeros((hopN, NCH), dtype="float32")    # NCH=2 without --out-map, which is the old shape
    tnow = st["t"] = st.get("t", 0.0) + a.hop     # file mode also runs on the fake clock
    for vi, p in enumerate(players):
        if a.human > 0:
            f1, p1, f2, p2 = HUM[vi]
            c = a.human * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                           + 0.4 * np.sin(6.28 * f2 * tnow + p2))
            r = rate * 2 ** (c / 1200.0)
        else:
            r = rate
        if a.vib > 0:
            # v37 vibrato. The review measured that the parts reproduce only 2.2%
            # of the singer's pitch energy in the 3-8 Hz band, that is, **they have
            # no vibrato at all**, which is the single strongest fingerprint of a
            # corrected sound, stronger than the moment of a note change.
            # It fades in only once the note is older than --vib-delay, so a note
            # change itself is clean and only a sustained note comes alive, which
            # is what a real singer does. It runs per sample, borrowing v25's rate
            # array, so 5.5 Hz is not cut into steps by a 35 ms hop.
            age = tnow - st["tsw"]
            if age > a.vib_delay:
                dep = a.vib * min(1.0, (age - a.vib_delay) / 0.3)
                ph = VIBPH[vi]
                t_ = tnow + np.arange(hopN, dtype="float64") / SR
                r = r * 2 ** (dep * np.sin(6.28 * a.vib_rate * t_ + ph)
                              / 1200.0)
        w = p.render(hopN, r) * MIXG[VOICES[vi][0]] * MEMG
        if OMAP is not None:
            y[:, OMAP[vi]] += w              # routing: a whole part goes to its own channel, unpanned
        else:
            gl, gr = _pan(PAN_F[vi])
            y[:, 0] += w * gl
            y[:, 1] += w * gr
        for mm in MEMS[vi]:
            # Members take the same note and the same bank as the lead, but each
            # has its own detuning, drift, vibrato phase, position and attack delay.
            # This is deliberately written out again rather than factoring the
            # lead's section into a function: factoring it would touch the certified
            # path, and the byte-identical guarantee at PP=1 would have to be
            # proved again.
            rm = rate * mm["det"]
            if a.human > 0:
                f1, p1, f2, p2 = mm["hum"]
                cm = a.human * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                                + 0.4 * np.sin(6.28 * f2 * tnow + p2))
                rm = rm * 2 ** (cm / 1200.0)
            if a.vib > 0:
                age_m = tnow - st["tsw"]
                if age_m > a.vib_delay:
                    dep_m = a.vib * min(1.0, (age_m - a.vib_delay) / 0.3)
                    t_m = tnow + np.arange(hopN, dtype="float64") / SR
                    rm = rm * 2 ** (dep_m * np.sin(6.28 * a.vib_rate * t_m
                                                   + mm["vib"]) / 1200.0)
            wm = _mdelay(mm, mm["p"].render(hopN, rm)
                         * MIXG[VOICES[vi][0]] * MEMG)
            if OMAP is not None:
                y[:, OMAP[vi]] += wm         # members share the lead's channel
            else:
                glm, grm = _pan(mm["pan"])
                y[:, 0] += wm * glm
                y[:, 1] += wm * grm
    for vi, pp in enumerate(pads):
        f1, p1, f2, p2 = HUM[len(VOICES) + vi]
        c = (a.human or 0) * (0.6 * np.sin(6.28 * f1 * tnow + p1)
                              + 0.4 * np.sin(6.28 * f2 * tnow + p2))
        w = (pp.render(hopN, rate * 2 ** (c / 1200.0)) * a.pad_gain
             * MIXG[VOICES[vi][0]])
        if OMAP is not None:
            y[:, OMAP[vi]] += w              # the pad shares its part's channel too
        else:
            gl, gr = _pan(PAN_P[vi])
            y[:, 0] += w * gl
            y[:, 1] += w * gr
    if RVIR is not None:
        # The reverb runs on a mono bus, as a diffuse field, and is fed back into
        # every channel, so the space wraps around the positions.
        # Under OMAP it uses sum, since each sample sits on one channel only and
        # sum is the complete mix; the old path keeps mean and stays
        # byte-identical.
        _bus = y.sum(axis=1) if OMAP is not None else y.mean(axis=1)
        wf = _fftc(_bus.astype(float), RVIR)
        wet_ = wf[:hopN]
        tl = RVT[0]
        wet_ += tl[:hopN]
        nt = wf[hopN:]
        rest = tl[hopN:]
        nt[:len(rest)] += rest
        RVT[0] = nt
        y = (y + a.wet * wet_[:, None]).astype("float32")
    y = np.clip(y * a.gain, -1, 1).astype("float32")
    if OMAP is not None:
        # A routing regression found live on 2026-08-18, when feedback rose until
        # the power had to be cut: the rms over the array is diluted by the number
        # of channels, reading 3 dB lower on four channels than in stereo, so the
        # feedback floor reads low and the microphone gate is more easily forced
        # open by feedback. Converting back to a stereo-equivalent scale, the total
        # energy divided by two channels, keeps the -13.5 bleed calibration in the
        # same terms. The calibration itself was measured with the old speaker
        # placement, so k must be swept again now the four speakers are spread out.
        _OLV.append(20 * np.log10(
            float(np.sqrt((y ** 2).sum() / (y.shape[0] * 2))) + 1e-12))
    else:
        _OLV.append(20 * np.log10(float(np.sqrt((y ** 2).mean())) + 1e-12))
    del _OLV[0]
    if dmp is not None and len(dmp["mic"]) < DMPMAX:
        dmp["mic"].append(chunk.copy())
        dmp["out"].append(y.copy())
        dmp["mok"].append(1.0 if mok else 0.0)   # R4A#2: the gate value in force
        dmp["face"].append(1.0 if M.get("face") else 0.0)
        dmp["mon"].append(1.0 if M["on"] else 0.0)
        dmp["val"].append(float(M.get("val", -1.0)))  # v27 opening value
        dmp["wd"].append(float(M.get("wd", -1.0)))    # v28 mouth width
        dmp["sp"].append(float(M.get("sp", -1.0)))    # v34 probability of singing
        dmp["f0"].append(f0_raw)                 # R4A#2: the detection before the gate
        dmp["note"].append(-1 if st["note"] is None else st["note"])
        dmp["vn"].append([p.cur[0] if p.cur else -1 for p in players])
        if a.vowels:
            dmp["vw"].append(vw_h)
    elif dmp is not None and not st.get("dumpfull"):
        st["dumpfull"] = True                    # v44: full, so recording stops and says so once
        print(f"⚠ dump buffer full (--dump-max-min {a.dump_max_min:g}) = "
              f"later audio not recorded (performance unaffected)", flush=True)
    return y


def worker():
    res = np.zeros(0, dtype="float32")
    while not st["die"]:
        with qlock:
            pend = np.concatenate(in_q) if in_q else None
            in_q.clear()
        if pend is None:
            time.sleep(0.002)
            continue
        res = np.concatenate([res, pend])
        while len(res) >= hopN:
            chunk, res = res[:hopN], res[hopN:]
            try:
                y = process_hop(chunk)
            except Exception as e:
                # Review #11, 2026-08-17: this used to be unguarded, so one
                # exception in process_hop killed the only thread producing sound
                # while the callback carried on. The result was silence for the
                # whole session with the xrun count frozen and every indicator
                # green, which is the worst way a live instrument can die. Now a
                # failed hop inserts a block of silence and carries on, and the line
                # raises the large warning bar; the shell's warn pattern matches the
                # leading marker.
                st["hopfail"] = st.get("hopfail", 0) + 1
                if st["hopfail"] <= 3 or st["hopfail"] % 200 == 0:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠ [engine] process_hop failed #{st['hopfail']} "
                          f"({e!r}) — inserting silence for this hop",
                          flush=True)
                y = np.zeros((hopN, NCH), dtype="float32")
            with qlock:
                out_q.append(y)
                while sum(len(q) for q in out_q) > a.maxlag * hopN:
                    out_q.pop(0)                 # the backlog ceiling is --maxlag blocks;
                    #   Honest note (review A10): dropping the oldest punches a
                    #   hole in a sustained note, a 35 ms jump, which is the price
                    #   of not letting latency settle; A9's fade covers the click.
                    # v25 (review S2), the latency ratchet: a backlog caused by one
                    #   stall in the worker is **never caught up**, because the
                    #   callback only ever takes exactly what it needs to fill, so
                    #   in the worst case +105 ms becomes permanent while the xrun
                    #   count sees nothing at all, since every callback was filled.
                    #   The ceiling is now a flag, defaulting to 3, the old
                    #   behaviour, a dropped block is recorded, and the next block
                    #   fades in; it used to be a hard join.
                    st["drop"] = st.get("drop", 0) + 1
                    UF[0] = True


if a.file:
    x, sr = sf.read(a.file[0], dtype="float32", always_2d=True)
    assert sr == SR, (a.file[0], sr)
    x = x[:, 0]
    outs = [process_hop(np.ascontiguousarray(x[i:i + hopN]))
            for i in range(0, len(x) - hopN + 1, hopN)]
    yy = np.concatenate(outs)
    sf.write(a.file[1], yy, SR)
    _write_dump()                                    # review A15: never lose the dump silently
    print(f"{len(x)/SR:.1f}s -> {a.file[1]}  rms "
          f"{float(np.sqrt((yy**2).mean())):.4f}  "
          f"glide {st.get('porta_n', 0)}", flush=True)
    raise SystemExit

OB = [np.zeros((0, NCH), dtype="float32")]
UF = [False]                              # the previous callback underran
LB = [np.zeros(NCH, dtype="float32")]     # the last output sample, for the underrun fade-out


def cb(indata, outdata, frames, tinfo, status):
    if status:
        st["xrun"] += 1
    with qlock:
        in_q.append(indata[:, 0].copy())
        grabbed = []
        need = frames - len(OB[0])
        while out_q and need > 0:
            b = out_q.pop(0)
            grabbed.append(b)
            need -= len(b)
    # Review A8: the concatenate moves outside the lock. Allocating memory while a
    # real-time thread holds a lock inverts priorities.
    if grabbed:
        OB[0] = np.concatenate([OB[0]] + grabbed) if len(OB[0]) else (
            grabbed[0] if len(grabbed) == 1 else np.concatenate(grabbed))
    buf = OB[0]
    if len(buf) >= frames:
        o = buf[:frames]
        if UF[0]:
            # Review A9: fade in the first block after recovering from an underrun,
            # or it is another step discontinuity.
            o = o.copy()
            k = min(96, frames)
            o[:k] *= np.linspace(0, 1, k, dtype="float32")[:, None]
            UF[0] = False
        outdata[:, :NCH] = o
        LB[0] = np.asarray(o[-1]).copy()
        OB[0] = buf[frames:]
    else:
        # Review A9: an underrun no longer cuts hard to zero; it fades out briefly
        # from the last sample.
        outdata.fill(0)
        nb_ = len(buf)
        if nb_:
            outdata[:nb_, :NCH] = buf
            last = np.asarray(buf[-1])
        else:
            last = LB[0]
        k = min(96, frames - nb_)
        if k > 0:
            outdata[nb_:nb_ + k, :NCH] = (last[None, :]
                                          * np.linspace(1, 0, k,
                                                        dtype="float32")[:, None])
        UF[0] = True
        LB[0] = np.zeros(NCH, dtype="float32")   # R2A#5: a continuous underrun means silence,
        OB[0] = np.zeros((0, NCH), dtype="float32")   # not an 86 Hz pulse train


import sounddevice as sd  # noqa: E402

_WT = threading.Thread(target=worker, daemon=True)
_WT.start()
try:
    VIEW = a.view and a.mouth and not a.file
    FB64 = a.frame_b64 > 0 and a.mouth and not a.file
    if VIEW or FB64:
        import cv2 as _cv                 # imshow has to run on the main thread on macOS
    if FB64:
        import base64 as _b64
    with sd.Stream(samplerate=SR, blocksize=a.sblock, channels=(1, NCH),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency="low", callback=cb):
        # ready is printed **after** the stream opens (review, 2026-08-17): it is
        # what respond_shell uses to retire the old engine on a switch and to show
        # SING in the UI. It used to be printed before the stream opened, so the UI
        # said the choir was following before the audio device was even in hand, and
        # a switch killed the old engine too early.
        print(f"ready (sample bank: detection window {a.hop*1000:.0f}ms x2, "
              f"note change fast ~{a.hop*1000:.0f}ms / slow ~{a.hop*2000:.0f}ms, "
              f"lock {a.lock}; Ctrl-C to stop)", flush=True)
        _ts = 0.0
        _tf = 0.0
        try:
            while True:
                time.sleep(0.1 if (VIEW or FB64) else 2)   # R5: a 30 Hz view measured
                #   +12 xruns per minute, since imshow competes with the real-time
                #   thread; 10 fps is enough to monitor and the xrun count returns
                #   to baseline
                if VIEW and M.get("frame") is not None:
                    _cv.imshow("mouth gate", M["frame"])
                    _cv.waitKey(1)
                if (FB64 and M.get("frame") is not None
                        and time.time() - _tf >= 1.0 / a.frame_b64):
                    # The image channel to respond_shell: one line, FRAME <b64>,
                    # which the shell special-cases and keeps out of the log. At
                    # quality 60 and 480 wide that is about 20 KB per frame, so
                    # 5 fps is about 100 KB/s and the pipe does not notice.
                    _tf = time.time()
                    okj, _jb = _cv.imencode(".jpg", M["frame"],
                                            [int(_cv.IMWRITE_JPEG_QUALITY), 60])
                    if okj:
                        print("FRAME " + _b64.b64encode(_jb).decode(), flush=True)
                if time.time() - _ts < 2:
                    continue
                _ts = time.time()
                # Review #11, 2026-08-17: liveness detection for the worker. It is
                # the only thread producing sound, and if it dies the callback goes
                # on filling underrun fade-outs, giving silence with every indicator
                # green.
                if not _WT.is_alive() and not st.get("wdead"):
                    st["wdead"] = True
                    print("⚠ [engine] audio worker thread died — OUTPUT IS "
                          "SILENT (restart the engine)", flush=True)
                nt = st["note"]
                print(f"note {nt if nt is not None else '—'}  "
                      f"cents {st['cents']:+5.1f}  "
                      f"mouth {'open' if M.get('eff', M['ok']) else 'closed'}  "
                      + f"xrun {st['xrun']}"
                      # v25: the backlog in blocks and the number of dropped blocks
                      # are the instrument for the latency ratchet, which used to be
                      # invisible
                      + f"  backlog {sum(len(q) for q in out_q) // hopN}"
                      + (f"/dropped {st['drop']}" if st.get("drop") else "")
                      # v40: why is the gate open? sp is the model probability and
                      # mo the jaw movement, which casts the veto.
                      # Without those two numbers, a false trigger can only be
                      # guessed at, which cost a whole day on 2026-08-15.
                      + (f"  sp {M.get('sp', 0.0):.2f} mo {M.get('mo', 0.0):.4f}"
                         if a.sing_gate and M.get("sp") is not None else "")
                      # v16/v17: extra fields are appended **at the end of the
                      # line**, so the existing field format is unchanged to the
                      # character, which respond_shell's regex depends on. The field
                      # is the five layer weights as percentages.
                      + (("  field " + "/".join(
                          f"{nm_}{int(round(w * 100))}"
                          for nm_, w in zip(_SEL_NAMES, M.get("vw", _VW_MID))))
                         if a.vowels else "")
                      # Review: warnings are appended **at the end of the line**.
                      # Inserted in the middle they broke the shell's bank_st regex,
                      # so when the gate died the warning and the only live readout
                      # disappeared together.
                      + ("  ⚠GATE DEAD" if M.get("dead") else "")
                      + ("  ⚠WORKER DEAD" if st.get("wdead") else ""),
                      flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            # Review #13, 2026-08-17: the dump is written **before the stream is
            # closed**, inside the stream context. The old code placed it after the
            # with block's __exit__, so when closing the CoreAudio stream hung, a
            # known playbook issue, the shell's 8 s SIGKILL took the whole recording
            # with it — and the entire method of "the ear is the only instrument,
            # read the dump afterwards" hangs on that file.
            st["die"] = True
            time.sleep(0.2)              # let the worker finish the hop in hand
            _write_dump()
except KeyboardInterrupt:
    pass                                 # Ctrl-C again while closing: the dump is
                                         # already written
print("bye")
