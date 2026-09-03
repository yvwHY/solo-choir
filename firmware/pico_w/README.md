# Pico W firmware — Solo Choir wearable (control plane)

The Pico W is the **on-body control plane only — it never carries audio.**
Audio path stays: `Mac (Beatrice engine) → PAM8403 amp → DAEX / bone conduction`.
The Pico W just sends control events (button → command) to the Mac and shows
status on an LED. See `docs/HARDWARE_ROADMAP.md` §8.

## What `pico_w_solo_choir.py` does
- Connects to WiFi (LED blinks while connecting, solid when up).
- Reads a momentary button / footswitch on GP15.
- On each press, toggles Harmonize and POSTs the control dict to the Mac.
- The payload uses the **same control contract as the UI** (`harmonize`,
  `you`, `gain`, …), so the Mac side treats Pico and UI identically.

## Not done yet (honest gaps)
- **No Mac endpoint exists.** `http://<mac-ip>:8765/control` is a placeholder.
  A small HTTP listener must be added to `app/bridge.py` later that feeds the
  received dict into `set_control`. Until then this is a structural skeleton.
- WiFi+HTTP is chosen for easy debugging. For a real wearable, BLE is lower
  power and router-free — same structure (read button → send command).

## Wiring (control side only)
| Pico W pin | To |
|---|---|
| GP15 | button → other side to GND |
| onboard LED | status (no wiring) |
| 3V3 / GND | as needed for the button pull-up reference |

The PAM8403 + DAEX wiring is the **audio** side and is independent of the Pico W
(driven from the Mac's audio output, not from the Pico).

## Flash / run
1. Install MicroPython for Pico W (UF2 from micropython.org) — hold BOOTSEL,
   drag the UF2 onto the RPI-RP2 drive.
2. Copy `pico_w_solo_choir.py` to the board as `main.py` (e.g. with `mpremote`
   or Thonny) so it runs on boot.
3. Set `WIFI_SSID` / `WIFI_PASS` / `MAC_URL` before flashing.
4. `urequests` is in `micropython-lib`; install via `mpremote mip install urequests`.
