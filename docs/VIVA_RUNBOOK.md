# RUNBOOK — the page taken to the performance

Written for the viva of 2026-08-26. This is an operating document, not a record.
Every number in it comes from the code or from a measurement, and the source is
named in brackets. Where it disagrees with any other document, this file and
`git log` win.

---

## 0. In one line

**Open the application on stage. Do not run the command line.**

**Normal launch:** double-click `app/start-respond-app.command` (an alias sits on
the desktop). It changes directory to the repository, opens the application with
the right interpreter, and **runs the clear-down of section 1 first**: it lists
anything left running, waits for Enter, sends `kill -INT`, and only opens a new
instance once `pgrep` comes back empty; if something will not close it prints a
ready-made `kill -9 <pid>`.

**Leave the terminal window open.** The start-up lines of section 3 are printed
there, and the rescue Ctrl-C is pressed there.

> **Do not turn it into a `.app` icon.** With no terminal window there is nowhere
> to press Ctrl-C and no way to see `[gate] sing model loaded`; and the
> microphone and camera permissions belong to Terminal.app, so a freshly built
> `.app` is an unknown program to macOS and will raise a permission dialogue on
> the day.

**Equivalent launch, only if the launcher is broken:**

```
cd <repo>
python app/respond_shell.py          # the vcclient-dev environment
```

The frozen configuration is written into the application's `_spawn_bank` and
`_spawn_solo`. Commit `34e137ed` showed that the argv the application sends,
fed to the offline bench, matches the rehearsal command line with a max
absolute difference of 0.0, that is, rehearsing and performing are the same
instrument. The command lines in section 2 are for when the application will not
start.

---

## 1. Eight-step check before performing

Allow five minutes. Step 7 is also done once while installing.

| # | Do | Pass condition |
|---|---|---|
| 1 | **Clear down.** `pgrep -fl "bank_live\|solo_min\|respond_shell"` | **Empty output.** If anything is left, `kill -INT <pid>`, let it finish writing its dump, and check again |
| 2 | **Open the application, press Start, watch the start-up lines** | **All** the lines of section 3 appear, above all `[gate] sing model loaded` |
| 3 | **The gate is alive.** After Start, watch the readings along the top right of the window: **make sound for three seconds, then be silent for three** | The row reads `pitch … · gate open · xrun 0`. **`gate` is `open` while sounding and turns `closed` after three seconds of silence.** Since 2026-08-18 the gate is driven by **the microphone, not the camera** (the frozen configuration carries `--mouth 0`, so the camera is retired entirely and the mediapipe thread never starts) — so **do not mime at the camera, actually sing**. The old lip-ring view no longer exists and `--view 1` cannot bring it back. The engine's status column is still called `mouth` and the window calls it `gate`; they are the same thing |
| 4 | **PA level, fifteen seconds of silence** | The choir must **fall silent**. Continued sound means feedback or a false gate open: pull the PA level down on the spot (section 5) |
| 5 | **Rehearse a twenty-second restart.** Ctrl-C, wait for it to finish printing, Start again, confirm the lines of section 3 | **Time it**, so the recovery time is known |
| 6 | **Fill in the routing fields** (only needed to send parts to separate units): in the map row of **the current mode**, set every cell — four cells `B·T·A·S` in bank mode, `D·U·L(·S)` in solo and respond modes | **An incomplete row silently falls back to the ordinary left-right split**: the code treats an empty field as "no `--out-map` at all", with **no error, no warning line and no indicator** (`app/respond_shell.py:427-432` bank, `:537-542` solo, `:597-598` respond). This step can only be confirmed by eye. After filling it in, **press Start again or switch mode**, since the engine reads the map only at start-up |
| 7 | **Three things about the screen.** (a) **No sleep** [once, at install]: System Settings, Lock Screen, set both the screen saver and the display-off timer to **Never**. (b) **Full screen** [every launch]: after Start press **Ctrl-Cmd-F**; the window defaults to 760x640 (`app/respond_shell.py:933-935`). (c) **Turn the screen towards the audience** [once, at install] | Only the application is visible, no desktop and no menu bar; **still lit after ten minutes away**. The laptop screen is the mode indicator: the audience changes mode from the caption card and confirms on this screen which mode they are in, so it must not sleep or be covered |
| 8 | **The physical key is connected** (Respond mode only): choose Respond, press Start, look at the capsule at the bottom of the window | The capsule reads **`BUTTON · end phrase`**. If it reads `SPACE` the **Bluetooth key has not connected** — and it fails **silently**, with no error or warning. The consequence: the audience presses and nothing happens, every phrase has to wait for the 1.2 s silence to end it, and the application looks frozen. Check the key has charge and Bluetooth is on, reopen and wait for `connected (BLE key)`. Do not read the engine's `listening on UDP (app space key)` as "no physical key": the Bluetooth key uses that path by design, and that line only says which port the engine listens on |

> **Full screen pushes the terminal to another desktop.** The rescue of step 5 is
> pressed in the terminal, so full screen costs a Cmd-Tab and a second or two;
> include that in the timing. To avoid it, zoom instead of going full screen
> (Option-click the green button): the window still fills the display and the
> terminal stays on the same desktop.

> **No second screen** (decided 2026-08-20). Neither a large display nor an iPad.
> An iPad cannot run the engine, which is Python and subprocesses on macOS; as a
> second display it does not take touch, so the audience taps and nothing
> happens and the mode still has to come from the keyboard; and any second
> screen makes the Mac compress video continuously, the same mechanism as the
> incident of 2026-08-17 where a streaming video tab in the background pushed
> `dropped` to 14 and closing it returned it to zero. The actual CPU cost of a
> second screen has **not been measured**.

---

## 1.5 Output devices

**Each application has its own output menu and neither follows the system
default.**

| When | Respond app `out` | Loop deck playback |
|---|---|---|
| **While the performer plays** | `viva-out` | `speakers4` |
| **Otherwise, audience playing on their own** | `mulit-set` (four parts routed) | off |

The three devices are built in Audio MIDI Setup; **confirm all three still exist
when installing**:

- `viva-out` — **multi-output**: external headphones, RØDE AI-Micro and BlackHole
  2ch. **The built-in laptop speakers are excluded**, being the feedback source
- `speakers4` — **multi-output**: external headphones and RØDE AI-Micro
- `mulit-set` — **aggregate**: external headphones and RØDE, whose four outputs
  are **concatenated** (1 and 2 the RØDE, 3 and 4 the headphones)

**Why the loop deck must not use `viva-out`:** its recording source is hard-wired
to BlackHole (the `--in-dev` default at `app/loop_deck.py:798`, not overridden by
the launcher). An output that also contains BlackHole feeds the bed back into its
own recording source and it smears with every pass. The code only blocks a device
literally named BlackHole (`:513-518`); a combined device with "bh" in its name
**warns but is not blocked** (`:562-566`).

**Why the loop deck must not use `mulit-set`:** an aggregate device concatenates
its outputs, so two channels reach only outputs 1 and 2 and **only two speakers
sound**. The symptom is "it records and loops correctly, but very quietly". Only
a multi-output device copies the same audio to every unit. (Found by measurement
on 2026-08-24.)

**What is given up in this room:** `viva-out` is a multi-output device and
therefore **has only 2 channels** (`loop_deck.py --list` reads out 2). So **while
the performer plays, the four parts are mixed to left and right**; four-way
routing exists only in the other periods, through `mulit-set`. The two cannot be
had at once: one needs a four-channel aggregate, the other needs a multi-output
that duplicates the signal.

**The loop deck has no pass-through** (`play_cb` plays only completed layers,
`:153-215`): it makes no sound while being sung into and starts looping once a
full round is recorded. **Everything the audience hears on the first round comes
from the respond app**, so the respond app must always have a path straight to
the speakers.

**To launch:** `app/start-loop-deck.command`, which also leaves a terminal
window. On 2026-08-24 the tempo field once **would not accept input** and
recovered by itself; the cause was not found. If it recurs, do not fight the
interface, launch with arguments instead:

```
$PY app/loop_deck.py --out-dev speakers4 --bpm 90 --beats 8
```

---

## 2. The two frozen configurations, verbatim

Shared: the working directory is `harmony/` for bank and `server/` for solo, and
the interpreter is the 6x venv, not conda:

```
PY=<repo>/260724_ddsp_svc_6x/venv/bin/python
```

### A. bank_live, four parts (application mode `bank`)

```
cd .../voice-changer/harmony
$PY bank_live.py --tenor 1 --vl 2 --trim bass=+7,tenor=+6 --vib 25 \
  --deadzone 0.35 --maxlag 2 --sing-gate 1 --min-level -50 --attack 0.06 \
  --per-part 4 --mem-real 0 --pad 0 --view 0 --frame-b64 5 \
  --dump scratchpad/app_bank_$(date +%m%d_%H%M%S)
```

Where each flag comes from. Every default is "off", so a missing flag means
falling back to the behaviour from before v26.

| Flag | Why |
|---|---|
| `--sing-gate 1` | the default 0 disables the gate entirely |
| `--vl 2` | at the default 1 the upper three parts sing the same note 79-100% of the time, so four parts are really three in unison. At 2, unison falls to 0% and a full triad occupies 43-50%, and each part is pulled out of the model's distortion region |
| `--tenor 1` | one more part from the same bass voice (v31) |
| `--trim bass=+7,tenor=+6` | settled 2026-08-16 (tenor was +3), after the verdict that bass and tenor could be louder. **It must be measured A-weighted**: an octave down, the bass reads **13.3 dB** lower A-weighted than on flat RMS, so RMS judges it loud enough while the ear cannot hear it — which is exactly why `--balance` was abandoned. Equal loudness is bass +5 and tenor +4; 2 dB was added to each to bring them forward |
| `--vib 25` | the default 0 means no vibrato at all, the strongest fingerprint of a corrected sound. The 3-8 Hz share rises from 11.6% to 19.9% (the singer's own is 22.7%) |
| `--min-level -50` | the default -100 disables the level gate. With the mouth closed at -54 dBFS, 43% of frames still report a pitch |
| `--deadzone 0.35` | the default 0 flickers between notes 32% of the time. At 0.35: 19% flicker, and 0 of 41 real notes missed |
| `--maxlag 2` | the ratchet limit on output latency (default 3) |
| `--attack 0.06` | onset |
| `--view 0` | F25: imshow consumes real-time budget on the main thread, measured at +12 xrun per minute. **Keep `--view` at 0**; this is the flag that was convicted |
| `--frame-b64 5` | **turned back on 2026-08-17.** It was set to 0 on 08-16 over a noise it did not cause (the culprit was `--mem-real`); F25 convicted imshow, and frame-b64 was never measured on its own. Live on 08-17, singing with the view on, `dropped` climbed to 14 — isolated, the culprit was a streaming video tab in the background, and with the tab closed there were zero drops in a minute, so the view itself is free. The lesson: **run no other heavy program on the performing machine** (browser video, downloads, Spotlight indexing) |
| `--mem-real 0` | **the cause of the intermittent noise of 2026-08-16.** At the default 1, a part may borrow a real singer from another model. The same note from two different recordings laid on top of each other always differs slightly in pitch: audible, and **invisible to all five instruments** (dropped, xrun, high-frequency energy, inharmonicity, roughness). Musically it costs nothing: in blind listening the same day, real timbres and detuned copies could not be told apart |
| `--pad 0` | the default 1.5 leaves the slow chord pad on, and it walks out a bass line nobody sang. Heard in a four-part isolation on 08-16 and ruled off, confirmed later the same day as permanent rather than temporary. Measured: the pad contributes +0.8 dB while singing and +5.2 dB while not. Consequently the `--pad-hold` question no longer exists, since with no slow layer there is no slow release |
| **no `--vowels`** | final verdict of 08-13: back to a single vowel, multiple vowels do not go on stage. Reversing this means editing `_spawn_bank` |

**Ranges (settled 2026-08-16): `VOICES` has bass at `-8` and tenor at `+6`, and
the tenor lead voice is `reflow-male8 spk7`.** Bass `-8` (median C3, 131 Hz) was
judged live under clean conditions once the noise was gone and the levels were
settled, against `-5` (D#3, 155 Hz), and it is also the optimum from the scan
(86.4% inside the core). It had been backed off to `-5` at one point to avoid bad
notes in a borrowed bank, but with `--mem-real 0` that bank is never loaded.

How the values moved: bass `0 → -8 → -5`, tenor `+7 → +6`. A dump on 08-15
measured the bass median at **G#3, exactly the singer's own pitch** (207.7 Hz),
with the tenor only two semitones above, so two male parts sat together in the
middle with nobody in the bass resonance region — which is what "the male
resonance is not coming through" meant. Scanning every transposition, **-8 is the
optimum** (86.4% inside the core, 0.4% outside the bank). `-12`, tried first, also
passed by ear but put only 54.8% inside the core with **8.9% outside the bank and
folded by an octave**, while low-frequency energy was nearly identical (13.2%
against 13.8%). Alto and soprano are unchanged, both 100% inside the core.

**`--per-part 4`, settled 2026-08-16:** three cells were compared blind at equal
loudness for 25 s each — A four voices, B sixteen with detuned copies, C sixteen
including real timbres — and the verdict was that **B and C could not really be
told apart, and both sounded more like a group of people**. So sixteen voices
yes, real timbres no gain. C could not win structurally: `MEM_SPK` has only one
surviving singer per part (of twelve auditioned, four survived), so `--per-part 4`
is a lead plus one real singer plus two detuned copies, and only 4 of the 16
voices are real. There is no CPU risk (F25: median 2.7 ms against a 35 ms budget
on the offline bench, zero overruns).

### B. solo_min, neural real time (application mode `Live · neural`)

```
cd .../voice-changer/server
$PY solo_min.py \
  --model .../sop_out_8k/paraphernalia_data_00003000 \
  --mode fixed --interval 12 --interval2 12 \
  --model3 .../model/paraphernalia_data_new25_2k --interval3 -12 \
  --model4 .../model/paraphernalia_data_new25_2k --interval4 -12 \
  --choir-gain 0.5 --gate-floor 0.01 --blocksize 960 --cushion-ms 20 \
  --mouth-gate 1 --gate-fail open --dry-delay-ms 0 --you-gain 0.6
```

| Flag | Why |
|---|---|
| `--blocksize 960` (20 ms) | two **different** models running at once thrash the cache. By ear: 480 crackles, **960 is clean**, 1920 is clean but the latency was rejected. The window is one flag step wide |
| soprano `+12` | +24 is a chipmunk, not a soprano |
| bass `-12`, **not 0** | `new25_2k` is the singer's own voice, so 0 semitones overlaps his dry signal exactly and cannot be heard as a separate part |
| `--choir-gain 0.5` | two groups of two identical settings each double themselves; 0.5 brings it back to one. **In duo mode the verdict was that the neural voice is much louder than the sampled one — adjust in the interface, do not hard-code it** |
| `--dry-delay-ms 0` | the 40 ms alignment exists to line the dry signal up with the converted harmony, which assumes headphones. **Over speakers the singer is in the room**, so the pass-through should align with their real voice, otherwise it is a slap-back echo |
| `--you-gain 0.6` | the pass-through has to stay: without it "you cannot hear the harmony, only two dry voices". Use `--no-dry` only with headphones or a split PA feed |

---

## 3. The start-up lines to watch for

**If this line does not appear there is no gate** — the engine does not stop, it
sings on:

```
[gate] sing model loaded (N params, mouth/jaw only, <same-session|leave-one-session-out> acc 0.XXX) ← sing_gate_model*.npz
```

Any of these means **the gate did not really load**:

```
⚠ [gate] no sing_gate_model.npz, falling back to a blendshape geometry threshold
⚠ [gate] the singing model failed its check, falling back to the geometry gate
⚠ [mouth] GATE DEAD (...) = running ungated
```

In bank_live, `GATE DEAD` is **hard-coded to run on**: the `except` in
`_mouth_worker` sets `M["ok"]=True` and no flag can make it fail to silence.
Only solo_min has `--gate-fail`, where `open` runs on and `close` mutes. Running
ungated is a silent failure; it bit once on 2026-08-14, when the qualification
check was written as `idx.shape != W.size`, comparing a tuple to an int and
therefore always true, and only that warning line revealed it. **The status line
carries `⚠GATE DEAD` at its end for as long as it lasts**, which is the only
persistent indicator.

> **Two reinforcements from 2026-08-17.** (1) A **lost camera heartbeat**, the
> most likely hardware failure, used to print small text and run ungated in
> silence; it now prints `⚠ [mouth] GATE DEAD (camera heartbeat lost)` and hangs
> off the status line, clearing itself when the heartbeat returns. (2) A new
> `⚠WORKER DEAD` warning and status-line marker makes a dead audio thread
> visible — the failure where the room goes silent while every indicator stays
> green. If it appears, Ctrl-C and restart; there is no other remedy. The
> application's warning bar lights for all three.

Also expected: `[mouth] camera index N (brightness ...)`, one `[bank] ... cache
hit (N notes)` per part, and one `[mix]` line.

> **`[mem-norm]` no longer appears, since 2026-08-17**: with `--mem-real 0` the
> borrowed banks are neither rendered nor loaded (previously about 28 MB was
> loaded and never played), and that line disappeared with them. **Its absence is
> correct and is not a start-up failure.**

> **A cold cache** (after changing a model, range or gain in `VOICES`, or moving
> to another machine): `cache hit` becomes `rendering k/N` progress lines and
> then `rendered`, and rendering everything takes several minutes. Since
> 2026-08-17 the application's switch watchdog only calls it stuck after 30 s
> with no output, so a cold-cache switch is no longer killed by mistake — but
> **warm the cache with the command of section 2A before performing anyway**.
> Force a re-render with `--rebuild`. A bad cache (printing `⚠ [bank] ... cache
> unreadable/incomplete → re-rendering` at start-up) re-renders itself and needs
> no manual clearing.

**The status line**, one every two seconds, is what step 3 of the check watches:

```
note 62  cents +2.3  mouth open  xrun 0  backlog 0  sp 0.87 mo 0.0031
```

`sp` is the model's probability that the singer is singing and `mo` is the jaw
movement, which holds a veto. `dropped N` appears **only when it is above zero**;
F25 calls it the only instrument that can see dropped audio at all, since neither
`xrun` nor `--dump` can see it structurally.

> **All five instruments were blind to that noise (2026-08-16).** `dropped`,
> `xrun`, high-frequency energy, inharmonic energy and a purpose-built roughness
> measure all reported clean on both the noisy and the clean take, `dropped` zero
> in both. The reason is that it **is not an added component**: the same note from
> two recordings laid together is only a few close lines in the spectrum, and the
> ear hears the interference. **On stage, do not talk yourself out of it because
> the instruments are clean — if the ear hears it, it is there.**

---

## 4. Three warnings for long runs

1. **Dumps consume disk at a measured 16 MB per minute.** `--dump` writes mono
   mic plus stereo out as PCM_16, that is 3 channels x 44100 x 2 bytes =
   265 KB/s, checked against a 226.6 s file of 60.6 MB. About **960 MB an hour**.
   The application writes one on every Start, with the path hard-coded in
   `_spawn_bank`. **Clear `harmony/scratchpad/app_bank_*` before performing.**
   > Since 2026-08-17 the dump records **only the first 30 minutes**
   > (`--dump-max-min`, protecting a resident memory cost of about 32 MB per
   > minute); when full it prints `⚠ dump buffer full` and the performance
   > continues.
2. **KeyTracker does not decay, so the key locks after a long run.** The
   histogram `KeyTracker.h` in `harmony/bank_live.py` only accumulates; once past
   `MIN=120` it sets the key from the statistics of the entire session.
   **Restart the engine between pieces** and do not let it run much beyond
   fifteen minutes.
3. **Shutdown, changed 2026-08-17: after Ctrl-C the dump is written first and the
   stream closed afterwards**, the write having moved into the stream context's
   `finally`. So if closing the stream hangs and the shell's 8-second SIGKILL
   takes it, **the dump is already safe**, and the few seconds between Ctrl-C and
   the dump line after a long run **are not a hang**. A `kill -9` *before*
   shutdown still loses the whole dump, which never gets the chance to be
   written.

---

## 5. The gate as it stands, and the fallbacks

**As it stands: the gate is unreliable and is left on** (`--sing-gate 1`), on the
grounds that an imperfect gate beats none — off means the choir sings while the
performer does not, which is feedback.

Both routes were measured and neither supports a binary decision about whether
the singer is sounding:

- **The camera route (condemned 2026-08-15).** The `cross-session acc 0.934` the
  engine printed was mislabelled; the model on disk was cross-validated on time
  blocks within one session, and the true cross-session figure is **0.673**. In
  the conditions of that day the model **triggered falsely on a closed mouth
  20-35%** of the time, and no cell of the `sing-open` by `bs-frames` table
  rescued it. The root cause is session drift: the same person in the same
  posture had a resting jawOpen of 0.031 on 08-13 and 0.017 on 08-15. **It cannot
  exclude speech** (63-97% across every session and threshold). And **with a hand
  over the mouth `sp` saturates at 1.00**, a linear model extrapolating with great
  confidence outside its training distribution.
- **The envelope route (insufficient discrimination, 2026-08-16, FINDINGS F26).**
  The feedback path exists and is measurable (a delay peak at **279 ms**, median
  separation 6.76x), but **a residual is not the singer sounding**: with the
  coupling calibrated, the microphone with the singer silent still sits **+2.7 dB
  median** above the predicted feedback, because it holds breath, breathy
  consonants and movement at the same order as the feedback. At the best
  threshold it mistakes feedback for singing **35-37%** of the time, no better
  than the camera.

### Fallbacks on stage, in order of cost

1. **The PA level is the master control.** Step 4 of the check measures exactly
   this. Both a false gate open and feedback scale with the PA level, and
   **pulling the PA down is faster than changing any flag and needs no
   keyboard.**
2. **Do not cover the mouth with a hand.** `sp` saturates at 1.00 and the gate
   opens wide. This failure is entirely avoidable by operation.
3. **Expect a false open while speaking to the audience** (speech cannot be
   excluded, 63-97%). Either pull the PA down before speaking or accept that the
   choir speaks along — **said in advance, this is a design boundary rather than
   an accident**.
4. **`⚠GATE DEAD` means the gate is already dead and the system is running
   ungated.** Only the PA level helps; the engine will not stop itself.
5. **Last resort: Ctrl-C and restart** (rehearsed in step 5, so the time it takes
   is known).

### How to say it in the thesis or aloud

This is **an honest negative result, not a defect**: two entirely different
sensing dimensions, visual mouth shape and acoustic energy, measured 20-35% and
35-37% false opens respectively, and they fail for different reasons — the camera
on session drift and out-of-distribution extrapolation, the envelope because the
residual contains every sound that is not feedback. The shared conclusion is that
**the question "is this person sounding" cannot be answered by observing from
outside the body**; answering it needs a different dimension, a contact one — a
throat microphone, EGG, bone conduction — that measures phonation itself. Which
returns directly to the body-as-interface argument of the thesis.
