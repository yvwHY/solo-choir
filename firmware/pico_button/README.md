# pico_button — the physical "your turn" button for the answering mode

One button on the body, over UDP, to the Mac. Control plane; it never touches
audio.

## Why it exists (measured 2026-08-01)

Phrase-end detection in the answering mode v2 is energy-only, and neither
weakness can be tuned away:

| Weakness | Number |
|---|---|
| Detection needs a full silence before it holds | 0.35 s, which is a floor on the latency |
| The choir returning to the microphone reads as still singing | measured p90 **0.016** against a gate of 0.02, that is **about 2 dB of margin** |

The second crosses the line as soon as the room gets louder, and the response
then never triggers. A physical button is immune to both, at about 6 ms.

**Division of labour**: the button ends the phrase, which is the ambiguous half.
The start is still energy detection, which has never missed, since the singer
opens above 0.05. If the button is not pressed, energy detection ends the phrase
as before, so the system cannot lock up.

## Wiring

| Pico W | To |
|---|---|
| **GP15** | one leg of the button |
| **GND** | the other leg |

Internal pull-up, so pressed is grounded and reads 0. **No resistor or other
part is needed.** The on-board LED is the status light: steady on when
connected, out for a moment on a press as visual feedback.

Which Pico: the one from the OLED line, free since the OLED face line was
dropped on 2026-07-23. Do not use either wire-light Pico; they run WS2812 at
60 fps, are timing-sensitive, and adding WiFi interrupts risks dropped frames.

## Flashing

```
# Copy main.py to the Pico (MicroPython v1.28, the same environment as the OLED one)
# It runs on boot; the serial port prints the STA or AP mode and the IP
```

## Network (dual-mode boot, which also closes the item left open in STATE)

1. If `STA_SSID` is filled in, try that router first, waiting 6 s.
2. If that fails, or it is empty, raise an access point **`SoloChoir`**
   (192.168.4.1) and join it from the Mac.

Both modes use **UDP broadcast on port 8766**, so the Mac never needs the other
side's IP and a change of venue needs no reconfiguration.

Filling in STA is easier during development, since the Mac keeps its network;
AP is simplest for a performance, since it does not depend on the venue.

## On the Mac

```
python harmony/tap_listen.py --selftest   # no hardware needed: send, receive and de-duplication (passing)
python harmony/tap_listen.py --monitor    # with the Pico attached, watch TAP and the connection state
```

In code, `TapListener().take()` consumes one press and `.alive` reports whether
the heartbeat is present, which is what is checked before going on stage.

## Reliability

- Every press sends **three** redundant datagrams 10 ms apart with a rising
  sequence number, which the Mac de-duplicates, so packet loss is tolerated
- A heartbeat every 2 s, so `.alive` can drive a stage connection indicator
- 200 ms debounce in the firmware
- If the port on the Mac is taken, it degrades to "there is no button" and
  blocks nothing

## Still to verify (needs the hardware in hand)

- [ ] Button physically on GP15/GND, flashed, TAP visible under `--monitor`
- [ ] Real WiFi latency (the OLED line measured a 6 ms ping with no loss over
      60 s; this button has not been measured on its own)
- [ ] Physical placement of the button (suggested on the chest post, within
      natural reach of the thumb; a foot switch is not advised, given movement
      and the audience)
