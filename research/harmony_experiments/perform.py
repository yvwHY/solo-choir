"""perform.py - performance mode: pre-rendered parts as a backing track, the singer takes the lead, both in one space.

The live half of the rehearsal pre-render was condemned on 2026-08-01 because of
the score tracker (37.9% hit rate). But that tracker existed only so the machine
could follow the singer, and IN PERFORMANCE that requirement can simply be
dropped: pre-rendered harmony is a backing track and the singer sings to it, which
is how every singer with a backing track works. With no tracking there is no
37.9%. The cost is that it stops being a live instrument and becomes singing with
a backing track, which is an artistic decision rather than a technical limit. The
offline half was re-auditioned on 08-02 and judged improved once the ceiling was
replaced by the target recipe.

**Routing (a hard rule, not an option): send and return.**
  The dry voice takes the direct path, microphone to PA; **the computer outputs
  wet only.**
Two reasons, both in the graveyard:
  - Sending his dry voice through the computer and back out plays him a delayed
    copy of himself, which is the delayed auditory feedback interference condemned
    on 2026-07-16.
  - A live microphone and live speakers at the same time is feedback. The answering
    mode destroys that structurally by never singing and playing at once, and
    performance mode has no such protection.
A late wet signal does not matter: **reverb already has a 20 ms pre-delay**, so
latency on the wet path is heard as a longer pre-delay. The whole latency budget
therefore goes to the dry path, which is direct and zero, and the computer needs
no low latency at all. `--dry-out` lets the dry voice leave the computer too, for
trying it alone on headphones with no PA; off by default.

The backing track and his voice go into the SAME send, so they are in one space.
The impulse response and the send level default to the pair settled on 08-03.

Run (DDSP venv, although it only uses numpy, scipy and sounddevice):
  python perform.py out/perf_stems_dry.wav --in-name "USB PnP" --list-devices
  python perform.py out/perf_stems_dry.wav --in-name "USB PnP" --wait-tap
"""
import argparse
import signal as _sig
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import reverb as rv  # noqa: E402
from pitch import SR  # noqa: E402
from tap_listen import TapListener  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backing", nargs="?", help="a dry mix of the part stems, with no reverb added yet")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--reverb", type=float, default=0.20, help="send level")
    ap.add_argument("--reverb-s", type=float, default=1.8)
    ap.add_argument("--reverb-trim", type=float, default=35.0)
    ap.add_argument("--backing-gain", type=float, default=1.0)
    ap.add_argument("--mic-send", type=float, default=1.0,
                    help="how much of his voice goes to the reverb, relative to the backing track")
    ap.add_argument("--dry-out", type=float, default=0.0,
                    help="dry level out of the computer. **The default 0 means the dry voice "
                         "takes the direct path**, see the file header; raise it only when "
                         "trying it alone on headphones with no PA")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--io-latency", default="high")
    ap.add_argument("--wait-tap", action="store_true",
                    help="wait for the physical key before playing, rather than starting immediately")
    ap.add_argument("--no-tap", action="store_true")
    a = ap.parse_args()

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return
    if not a.backing:
        ap.error("a backing wav is required (dry stems)")

    bak, sr = sf.read(a.backing, dtype="float64", always_2d=True)
    bak = np.ascontiguousarray(bak[:, 0])
    assert sr == SR, (a.backing, sr)
    ir = rv.make_ir(a.reverb_s, trim_db=a.reverb_trim)
    conv = rv.ConvLive(ir, 1024)

    tap = None if a.no_tap else TapListener()
    if tap is not None and tap.err:
        print(f"[tap] bind {tap.port} failed ({tap.err}), treating it as no physical key")
        tap = None

    lock = threading.Lock()
    st = {"pos": 0, "run": not a.wait_tap, "iov": 0, "oun": 0,
          "peak": 0.0, "clip": 0, "mic": 0.0}
    dump = np.zeros(int((len(bak) / SR + 30) * SR), dtype=np.float32)
    st["n"] = 0

    def cb(indata, outdata, frames, t, status):
        if status:
            if status.input_overflow:
                st["iov"] += 1
            if status.output_underflow:
                st["oun"] += 1
        mic = indata[:, 0].astype(np.float64)
        with lock:
            run, p = st["run"], st["pos"]
        b = np.zeros(frames)
        if run:
            s = bak[p:p + frames]
            b[:len(s)] = s * a.backing_gain
            with lock:
                st["pos"] = p + len(s)
        # backing track and voice into the same send, so they share one space
        w = conv(b + a.mic_send * mic)[:frames]
        y = b + a.reverb * w + a.dry_out * mic
        y = y * a.gain
        pk = float(np.abs(y).max()) if frames else 0.0
        st["peak"] = max(st["peak"], pk)
        st["mic"] = float(np.sqrt(np.mean(mic ** 2)))
        if pk > 1.0:
            st["clip"] += 1
        outdata[:, 0] = np.clip(y, -1, 1)
        if outdata.shape[1] > 1:
            outdata[:, 1] = outdata[:, 0]
        n = st["n"]
        if n + frames <= len(dump):
            dump[n:n + frames] = outdata[:, 0] + mic      # what went out plus what he sang
            st["n"] = n + frames

    print(f"perform. backing {len(bak)/SR:.1f}s | space T60 {a.reverb_s} trimmed "
          f"{a.reverb_trim:.0f}dB send {a.reverb} | dry "
          f"{'direct path, not from the computer' if a.dry_out == 0 else f'from the computer at {a.dry_out}'}"
          f" | physical key {'off' if tap is None else 'on'}"
          f" | {'waiting for the key' if a.wait_tap else 'starting now'} -- Ctrl-C stops.",
          flush=True)
    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, 2),
                   device=(a.in_name, a.out_name), callback=cb,
                   latency=a.io_latency):
        t0 = time.perf_counter()
        try:
            while True:
                if tap is not None and tap.take():
                    with lock:
                        if not st["run"]:
                            st["run"] = True
                            print("  (key: start)", flush=True)
                        else:                       # pressed again: back to the top
                            st["pos"] = 0
                            print("  (key: back to the top)", flush=True)
                with lock:
                    p, run = st["pos"], st["run"]
                if run and p >= len(bak):
                    break
                if time.perf_counter() - t0 > 2:
                    t0 = time.perf_counter()
                    print(f"  {p/SR:6.1f}/{len(bak)/SR:.1f}s｜mic rms "
                          f"{st['mic']:.3f}｜peak {st['peak']:.2f}"
                          f"｜io {st['iov']}/{st['oun']}"
                          f"{' | clipping ' + str(st['clip']) if st['clip'] else ''}",
                          flush=True)
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass

    n = st["n"]
    if n > SR:
        p = HERE / "out" / f"perform_{time.strftime('%y%m%d_%H%M%S')}.wav"
        p.parent.mkdir(exist_ok=True)
        sf.write(str(p), dump[:n], SR, subtype="PCM_16")
        print(f"\nsession dump: {p}（{n/SR:.1f}s）")
    print(f"peak {st['peak']:.2f} | clipped blocks {st['clip']}"
          f"｜io overflow/underflow {st['iov']}/{st['oun']}")
    if st["clip"]:
        print(f"clipping detected -> --gain {0.89/max(st['peak'],1e-9):.2f}")


if __name__ == "__main__":
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)     # a lesson learned the hard way
    main()
