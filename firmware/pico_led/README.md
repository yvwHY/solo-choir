# pico_led — wire-light firmware (two units, two WS2812B COB strips each)

Plan: `docs/plans/2026-07-23-pico-led-wire-light.md` · Spec: `docs/specs/2026-07-19-wire-light-design.md`

## Flashing
1. A Pico W (**not the OLED one**) with MicroPython v1.28, the same build as
   260716_pico_oled. On a blank board, hold BOOTSEL and drag the uf2 onto the
   drive that appears.
2. `mpremote cp main.py :main.py`, or save as main.py from Thonny.
3. Unplug USB, apply external power, and it runs on boot (the demo breathing
   pulse).

## Bench wiring (settled after task 1)
```
5V rail --+--> 1N4007 --* strip 5V node (about 4.3V)   required since 2026-07-26, see the threshold curve
          |   (banded end towards the strip)  +--> strip A 5V
          |                                   +--> strip B 5V
          |                                   +--> 100uF across this node and GND (the bulk electrolytic moves after the diode)
          +--> Pico VSYS through a 1N4007 (about 4.4V; blocks back-feed from USB), a separate diode from the one above
GP1 --330R--> strip A DIN (the arrow-in end)   the 330R is required; it prevents back-feed from killing the pin, which is how GP0 died
GP2 --330R--> strip B DIN
Common ground (Pico GND to supply negative to strip GND; all three are needed)
```
**The 5V rail itself stays at 5.0V**, because the PAM8403 needs a full 5V; only
the strip branch is dropped. One 1N4007 serves both strips (rated 1A, and two
strips at low brightness draw about 0.3A).

## Walk test (measuring addressable pixel density; done, 160/m)
```python
import neopixel, time
from machine import Pin
np_ = neopixel.NeoPixel(Pin(1), 400)
for i in range(400):
    np_.fill((0,0,0)); np_[i] = (80,40,10); np_.write(); time.sleep_ms(30)
```
Count by eye how many lit points step through per metre, then set `N` in
`main.py` accordingly.

## Measurements (bench, 2026-07-23, TENMA 72-2690 supply)
| Item | Value | Note |
|---|---|---|
| Addressable px/m | **160/m** (spacing measured at about 60 mm per 10, and all 800 on the reel reach the end) | 1:1 with the LED density |
| Data ladder stops at | ~~direct connection works~~ -> **a 1N4007 dropping the strip VCC (the third step) is required** | reversed on 2026-07-26, see the threshold curve below |
| Standby current | **0.417 A / 5 m, about 83 mA/m** with nothing lit | the bulk of the wearable budget: a 1 m run costs about 83 mA on standby alone |
| Current at MAX_LEVEL 0.2 (fire-flow demo, 160 px) | total reading 0.4x A, so a few tens of mA net | |
| Current at 0.45 (the v5.1 brightness, fire-flow demo, 160 px) | total swings 0.5-0.65 A, so about 0.08-0.23 A net, rising and falling with the flow | includes 0.417 A standby for the whole 5 m reel; after cutting, standby scales at 83 mA/m |
| Settled MAX_LEVEL | 0.45 (settled by eye on the bench; to be revisited after the wearable current is checked) | |
| MT3608 drop | (measure at unit integration) | under 0.1 V passes |

## Measured threshold curve (2026-07-26, perfboard, MT3608 supply, two runs of 122 px each with JST tails)

| Strip VDD | Threshold `0.7 x VDD` | Result |
|---|---|---|
| 4.825 V | 3.38 V | both clean |
| **4.9 V** | **3.43 V** | **one starts corrupting** |
| 5.0 V | 3.50 V | both corrupt |
| **4.3 V (after the 1N4007)** | **3.01 V** | **both clean with the 5.0 V rail held; this is the working point** |

## Strip self-coupling (2026-07-27, unit 1 perfboard)

Reproducible, dark to lit to dark at the same moment:

| Strip state | span (L) | span (R) |
|---|---|---|
| All dark | 1,004 | 928 |
| **All lit** | **5,041** | **5,628** |
| All dark again | 952 | 1,012 |

The strip's PWM current pulses couple back into the ADC through the shared
ground. A single fixed threshold **latches**: the strip lights, the noise floor
crosses the threshold, that reads as sound, the strip stays lit, and it does not
go dark when the sound stops. It did not appear in breadboard task 2 because
that had one strip, one tap and the strip drawing straight from a low-impedance
rail; **the 1N4007 added on 2026-07-26 to fix data corruption put impedance in
the supply path, which is what turned the pulses into voltage swings** — fixing
A caused B.

Three solutions were tried; the third was adopted:

| Solution | Result |
|---|---|
| **Blanked sampling** (go dark for 1 ms every N frames, then measure) | the noise floor falls from 3,203 to 729, which works, but `write` takes 3.7 ms for 122 pixels and 7.4 ms for both, so the real dark period is about 5 ms at 20 Hz. The flicker is clearly visible and it was rejected |
| **Hysteresis** (2000 dark, 6000 lit) | solves the latching, but oscillates and dulls the response when the signal sits between the two thresholds, at middling levels |
| **Coupling compensation** (adopted) | `span -= COUPLING * env`. The coupling is proportional to brightness and `env` is the brightness, so nothing is subtracted while dark, keeping full sensitivity, and at full brightness it exactly cancels. **One low threshold at 2000, no hysteresis, no oscillation, and independent of the Mac's output level** |

The real cure is still hardware: the bulk electrolytic belongs **after** the
1N4007, on the strip side, raised to 470 uF if necessary, and `COUPLING` must be
measured again after any such change.

## Lessons paid for (bench, 2026-07-23 onwards)
- **A dry joint on the GND pin of the header — the most deceptive fault of the
  evening (2026-07-27)**: the symptom was that the Pico did not run once USB was
  unplugged, the lights froze, and yet VSYS measured a healthy 4.56 V. Four
  disguises: (1) with no current there is no drop, so the off-load voltage looks
  perfect; (2) **the continuity buzzer sounds** at its 50 ohm threshold, while a
  few tens of ohms is already enough to stop the Pico booting; (3) with USB
  plugged in, ground returns through USB to the Mac and everything works, so it
  only appears once unplugged; (4) the Pico W has no power LED, since the
  on-board LED hangs off the WiFi chip, so there is nothing to see.
  **The method: run two jumper wires, VSYS and GND, straight from the supply to
  the Pico, bypassing the whole board.** That clears every joint at once; then
  add them back and bisect. It is an order of magnitude faster than measuring
  point by point.
- **A 3.5 mm plug not fully seated kills one channel**: the symptom is "one
  strip is completely dead, the other is fine", which reads as a broken board.
  It is the first thing to rule out when plugging and unplugging often.
- **The Mac's system volume sets the signal level directly**: with the same test
  tone, span is only 1,500-3,200 at volume 44 and 9,000-10,000 at volume 65. Work
  at 65. (The noise floor of about 1,000 with the strip dark **does not follow
  the volume**, because it is interference and not signal.)
- **`mpremote` interrupts `main.py` the moment it connects**: the raw REPL stops
  the firmware and does not restart it, so reading the ADC over the link and
  watching how the lights behave cannot be done at the same time. To watch the
  lights, do not connect; to read numbers, accept that the lights are dead.
- **Driving a 5 V strip from 3.3 V data was always under the threshold
  (reversed 2026-07-26)**: the WS2812 high threshold is `0.7 x VDD`, which is
  **3.5 V** at 5.0 V, and the Pico can only deliver **3.3 V**. The 2026-07-23
  pass on a direct connection was standing on the edge of a cliff: the TENMA was
  set to 5 V, delivered slightly less, and landed just under the failure point.
  Switching to the MT3608, measured at 5.044 V, and adding JST tails failed on
  the spot, with the symptom being **random, scattered, shifting colours**; even
  a static pure red looks like a running animation. The fix is a **1N4007 in
  series with the strip's 5 V feed**, keeping the rail at 5.0 V for the PAM,
  which turns a margin of -0.2 V into +0.35 V. Diagnostic method: send a single
  flat colour and judge from that, because under a static image data corruption
  looks like a deliberate effect.
- **GP0 is dead and abandoned for good**: when the strip's supply collapsed
  under current limiting (CC at 2.76 V), GP0's 3.3 V back-fed into the input of
  a now low-voltage IC, the Pico dropped off and the pin was destroyed. Hence
  `DATA_PINS=(1,2)` and **330 ohms in series on every DATA line**, which is
  exactly what that prevents.
- **This COB reel lights up randomly at full brightness when powered with no
  valid data** (3.8 A, 19 W, and it heats up while still coiled), so the firmware
  must refresh continuously from boot, which main.py already does. For a manual
  single shot on the bench, start the refresh loop first and power the strip
  second.
- **The strip has no memory**: the image is lost on a power cut or a collapse
  and must be resent.
- **Performance is a hard requirement**: rendering with plain Python floats
  manages **5.5 fps**, which is the real cause of the stiff, stepped look. An
  integer heat field, a LUT, `@micropython.native` and pushing only the real
  strip length give **60 fps at 160 px**. Low brightness also needs a gamma of
  1.5; at 2.0 the dark end is crushed to two or three code values.
- When measuring the whole reel, set the TENMA current limit to 2 A or more
  (0.417 A standby plus the lit current; a 0.5 A limit on the first day started
  a collapse cascade).

## Tier 2 ADC tap (from task 2, per channel)
```
PAM input header (before the PAM; never touch the BTL output pins - the rule since 2026-07-13)
   +-- C 100nF-1uF in series --*-- GP26/GP27
                               +-- R1 100k -> 3V3
                               +-- R2 100k -> GND    (R1 = R2; any matched pair from 47k to 220k works)
```
**Settled on the task 2 bench (late on 2026-07-23/24)**: NOISE_FLOOR = **500**
(noise floor about 250, music about 900-1400); **the minimum-of-three
consecutive windows filter is necessary**, because the strip's current pulses
couple back into the ADC through the shared ground and produce flare-ups in
silence, and a minimum of two is not enough; **a bulk electrolytic of 100-470 uF
at the strip's 5 V feed is required**, absorbing the pulses at the source, and
power must be off when connecting or disconnecting, since a live inrush restarts
the Pico. A 10 uF electrolytic coupling capacitor works, negative towards the
source. Acceptance: dark when paused, responds as soon as playback starts,
follows quiet passages, and no flare-ups over two minutes of silence — all
passed, and running live mode from boot verified. Colouration A/B is left to
task 4, when the PAM is really in the path.
**Unit 1 perfboard measurements (2026-07-26)**: noise floor span **about 1000**
(the breadboard bench was about 250) and a 440 Hz test tone **about 17900**, the
same on both channels, so a signal-to-noise ratio of 18x. `NOISE_FLOOR` rose
from 500 to **2000**, and live mode passed acceptance: lit during playback, dark
when it stops.
- **Known defect (to be fixed with a new capacitor)**: the divider midpoint is
  pulled to **0.52 V** where it should be 1.65 V. Working back from the current
  balance, there is a leakage path of about 22 kohm from the midpoint to ground
  that appears only when powered, which is leakage in the 10 uF electrolytic
  (specified under 3 uA, measured at about 23 uA). The consequences are a raised
  noise floor and only 0.52 V of downward headroom; at the present level of
  0.90 Vpp it just avoids clipping.
- **Diagnostic lessons**: (1) the midpoint voltage names the fault at once:
  3.3 V means the lower leg is open, 0 V means the upper leg is open, and 1.65 V
  is correct; (2) **the continuity buzzer cannot measure 100k**, since anything
  over 50 ohms reads open, so use the resistance range or a healthy circuit will
  be called broken; (3) dry joints are usually cold joints that look neat but
  never took solder, which is how the lower 100k was lost.

**Unit power (task 4 build)**: the unit's 5 V rail (TP4056 to MT3608) feeds both
strips directly along with the bulk capacitor; the Pico's VSYS goes through a
1N4007; nothing draws from USB during a performance (the grounding rule of
2026-07-12). If there is not enough headroom, lower MAX_LEVEL.

## Acceptance (the gate for each task in the plan)
- [x] T1 demo lit: warm white, a comet travelling away from the Pico, breathing
      that feels alive (judged by eye, perfboard, N=122, 2026-07-26)
- [x] T2 live: **responds on opening the mouth, the movement feels alive, and
      the resting glow is stable in silence — all three passed** (2026-07-26,
      USB microphone into the Mac, 3.5 mm out, sung live). Colouration A/B waits
      for the PAM.
  - Note when testing through the direct path: turn the microphone input down
    before singing, or feedback pins the envelope at its ceiling (peak 1.000),
    the fire flow runs at maximum speed and no dynamics are visible.
- [ ] T3 four cut runs terminated: 4/4 walk test passed (lengths: both runs of
      unit A measured **122 px, about 763 mm**, using the tape method of
      2026-07-26, changing colour every 100 px and counting whole colour blocks
      plus the fraction at the end, which is faster than a walk test; the other
      two runs are _ / _ mm)
- [ ] T4 unit A (the Mac 3.5 mm side) worn and sung with, both strips independent
- [ ] T5 both units worn and lit as a whole instrument (the final gate)
