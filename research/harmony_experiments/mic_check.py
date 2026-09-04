"""mic_check.py - a thirty-second check on the recording chain before singing.

Why this exists: one session was recorded at only 29 dB SNR, where the same USB
microphone had given 42 dB a week earlier, and all three voices sang the
microphone's noise floor along with everything else - the "whispering breath"
quality. Nobody knew the material was already spoilt, so a whole round of listening
and measurement was built on it. This exists so that does not happen twice.

There is no software gain on the input (spike_stream6.py:486 takes indata
directly), so the only control is the macOS input volume, under Settings, Sound,
Input.

Run (DDSP venv):  python mic_check.py            # 12 seconds by default
                  python mic_check.py --sec 20
How: be quiet for the first 3 seconds, to measure the noise floor, then sing a phrase normally.
"""
import argparse
import numpy as np
import sounddevice as sd

SR = 44100
HOP = 512
QUIET_S = 3.0                 # length of the silent stretch at the start
TRAIN_REF_RMS = 0.0566        # direct_mouth.TRAIN_REF_RMS, the model's training reference level
SNR_GOOD = 40.0               # the good session measured 42 dB
SNR_MIN = 35.0


def db(x):
    return 20 * np.log10(max(float(x), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--sec", type=float, default=12.0)
    ap.add_argument("--file", default="",
                    help="judge an existing wav instead of recording, which also checks this tool offline")
    a = ap.parse_args()

    if a.file:
        import soundfile as sf
        x, _ = sf.read(a.file)
        if x.ndim > 1:
            x = x.mean(1)
        x = x.astype(np.float64)
        print(f"judging {a.file} ({len(x) / SR:.1f}s)")
    else:
        print(f"recording {a.sec:.0f} seconds - be quiet for the first {QUIET_S:.0f}, then sing a phrase normally")
        buf = sd.rec(int(a.sec * SR), samplerate=SR, channels=1,
                     dtype="float32", device=a.in_name)
        for s in range(int(a.sec), 0, -1):
            sd.sleep(1000)
            print(f"  {s}…", end="", flush=True)
        sd.wait()
        print()
        x = buf[:, 0].astype(np.float64)
    n = len(x) // HOP
    r = np.sqrt((x[:n * HOP].reshape(n, HOP) ** 2).mean(1))
    nq = int(QUIET_S * SR / HOP)

    if a.file:
        # an existing file has no agreed silent stretch: use whole-file percentiles, the same measure as the original diagnosis
        floor, voice = np.percentile(r, 10), np.percentile(r, 90)
    else:
        floor = np.percentile(r[:nq], 90)    # the top of the silent stretch, which is the real floor
        voice = np.percentile(r[nq:], 90)    # p90 of the singing
    snr = db(voice) - db(floor)
    peak = np.abs(x[nq * HOP:]).max()

    print(f"\n  noise floor {db(floor):6.1f} dBFS")
    print(f"  singing     {db(voice):6.1f} dBFS   (peak {db(peak):.1f})")
    print(f"  SNR    {snr:6.1f} dB")
    print(f"  against the training reference {db(TRAIN_REF_RMS):.1f} dBFS: "
          f"{db(voice) - db(TRAIN_REF_RMS):+.1f} dB")

    print()
    if peak > 0.95:
        print("  FAIL clipping - turn the input volume down")
    elif snr < SNR_MIN:
        print(f"  FAIL SNR {snr:.0f}dB is too low (the spoilt session was 29 dB). "
              "Turn the macOS input volume up and measure again;\n"
              "    if it is already at maximum, try the distance, the cable or another USB port.")
    elif snr < SNR_GOOD:
        print(f"  MARGINAL SNR {snr:.0f}dB is usable but below the good session (42 dB). "
              "Raise it if there is room.")
    else:
        print(f"  PASS SNR {snr:.0f}dB - ready to sing")

    if abs(db(voice) - db(TRAIN_REF_RMS)) > 6 and snr >= SNR_MIN:
        print("  MARGINAL the singing level is more than 6 dB from the training distribution, where the model degrades")


if __name__ == "__main__":
    main()
