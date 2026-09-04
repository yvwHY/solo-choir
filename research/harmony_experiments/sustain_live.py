"""sustain_live.py - a sustain-triggered synchronous live prototype (v0).

Hold a steady note - swinging less than 30 cents over about 0.26 s - and the parts
(soprano an octave up, bass an octave down) render a 1.2 s chunk in batch and join
in. Keep holding and they refill; stop and they fade over 150 ms. The voice is your
current vowel, a palindromic loop filling the chunk so units and volume vary
naturally rather than freezing one frame; f0 is your current pitch times a fixed
ratio, with desynchronised vibrato; and everything passes through the enhancer, the
same audio path as the answering mode.

Why this can live where streaming WORLD resynthesis died (G28): a sustained note is
quasi-stationary and needs no frame-by-frame context; the dead timbre was already
solved by the enhancer; and units pass straight through, so the parts sing whatever
vowel you sing. The verdict has to be live.

Run (DDSP venv):
  python sustain_live.py                      # USB PnP in, AI-Micro out
  python sustain_live.py --in-name X --out-name Y --gain 0.6
Ctrl-C to stop.
"""
import argparse
import threading
import time

import numpy as np
import sounddevice as sd

import direct_mouth as dm
import respond2 as R
from phrase_render import PhraseRenderer
from prosody_ab import B1

SR, BLK, HOP = 44100, 1024, 512
TICK_S = 0.033
STAB_TICKS = 8            # about a 0.26 s stability window
STAB_CENTS = 60.0         # max minus min within the window
CHUNK_S = 1.2             # seconds rendered per chunk
REFILL_AT = 0.45          # under 0.45 s left to play and still steady: refill
RELEASE_S = 0.15
MIN_HZ, MAX_HZ = 70.0, 500.0
SOP = (f"{R.DDSP}/exp/combsub-m4-sop3/model_10000.pt", 2.0, 4.9, 0.75)
BAS = (B1, 0.5, 4.6, 0.5)


def rt_f0(x):
    """Sliding-window live f0: parselmouth autocorrelation over the last 0.35 s, returning the median of the voiced tail."""
    import parselmouth
    snd = parselmouth.Sound(np.ascontiguousarray(x), SR)
    p = snd.to_pitch_ac(time_step=0.02, pitch_floor=MIN_HZ,
                        pitch_ceiling=MAX_HZ)
    v = p.selected_array["frequency"]
    v = v[v > 0]
    if len(v) < 3:
        return 0.0
    return float(np.median(v[-5:]))


def loop_audio(x, n_samp, xf=int(0.03 * SR)):
    """Loop a vowel fragment palindromically out to n_samp, with a 30 ms crossfade at the splice, so units and volume stay alive."""
    seg = x
    outs = [seg]
    fwd = False
    total = len(seg)
    while total < n_samp:
        nxt = seg[::-1] if not fwd else seg
        fwd = not fwd
        a, b = outs[-1], np.copy(nxt)
        k = min(xf, len(a), len(b))
        r = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
        b[:k] = a[-k:] * (1 - r) + b[:k] * r
        outs[-1] = a[:-k]
        outs.append(b)
        total += len(b) - k
    y = np.concatenate(outs)
    return y[:n_samp]


class Angel:
    def __init__(self, path, ratio, vib_hz, vib_phase):
        self.r = PhraseRenderer(R.DDSP, path, enhance=True)
        self.ratio, self.vib_hz, self.vib_phase = ratio, vib_hz, vib_phase

    def chunk(self, vowel, f0_hz, secs, t0):
        torch = self.r.torch
        n = int(secs * SR) // HOP
        xa = loop_audio(vowel, n * HOP)
        g = dm.TRAIN_REF_RMS / (np.sqrt((xa ** 2).mean()) + 1e-9)
        g = float(np.clip(g, 0.25, 16.0))
        xa = xa * g
        t = t0 + np.arange(n) * HOP / SR
        f0 = (f0_hz * self.ratio) * 2 ** (
            0.12 * np.sin(2 * np.pi * self.vib_hz * t + self.vib_phase) / 12)
        with torch.no_grad():
            au = torch.from_numpy(xa).float()[None].to(self.r.device)
            units = self.r.encoder.encode(au, SR, HOP)
            vol = self.r.vol_ex.extract(xa)
            m = min(units.size(1), n, len(vol))
            mask = self.r._vol_mask(np.asarray(vol[:m]))[:m]
            f0_t = torch.from_numpy(f0[:m]).float().to(self.r.device)[None, :, None]
            vol_t = torch.from_numpy(np.asarray(vol[:m])).float().to(
                self.r.device)[None, :, None]
            out, _, _ = self.r.model(units[:, :m], f0_t, vol_t, spk_id=self.r.spk)
            mk = torch.from_numpy(np.repeat(mask, HOP)).float().to(
                self.r.device)[None, :]
            tt = min(out.size(1), mk.size(1))
            o, esr = self.r.enhancer.enhance(out[:, :tt] * mk[:, :tt], SR,
                                             f0_t, HOP, adaptive_key=0)
            wav = o.squeeze().cpu().numpy().astype(np.float64)
        return wav / g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--out-name", default="AI-Micro")
    ap.add_argument("--gain", type=float, default=0.6)
    ap.add_argument("--attack-ms", type=float, default=80.0)
    a = ap.parse_args()

    print("loading both voices and the enhancer...", flush=True)
    angels = [Angel(*SOP), Angel(*BAS)]
    for an in angels:                      # warm up MPS
        an.chunk(np.random.randn(int(0.4 * SR)) * 0.03, 200.0, 0.3, 0.0)
    print("ready", flush=True)

    ring = np.zeros(SR * 2)
    play = {"buf": np.zeros(0), "pos": 0, "fade": False}
    lock = threading.Lock()
    st = {"mode": "idle", "f0": [], "t": 0.0, "die": False, "n_join": 0}

    def cb(indata, outdata, frames, tinfo, status):
        ring[:-frames] = ring[frames:]
        ring[-frames:] = indata[:, 0]
        outdata[:] = 0
        with lock:
            b, p = play["buf"], play["pos"]
            s = b[p:p + frames]
            if len(s):
                if play["fade"]:
                    k = min(len(s), int(RELEASE_S * SR))
                    ramp = np.linspace(1, 0, k)
                    s = np.copy(s)
                    s[:k] *= ramp
                    s[k:] = 0
                    play["buf"] = np.zeros(0)
                    play["pos"] = 0
                    play["fade"] = False
                else:
                    play["pos"] = p + len(s)
                outdata[:len(s), 0] = s * a.gain
                if outdata.shape[1] > 1:
                    outdata[:len(s), 1] = outdata[:len(s), 0]

    def render_into(f0_hz, t0, attack=False):
        vowel = np.copy(ring[-int(0.4 * SR):])
        ws = [an.chunk(vowel, f0_hz, CHUNK_S, t0) for an in angels]
        n = min(len(w) for w in ws)
        mix = sum(w[:n] for w in ws) / len(ws)
        if attack:
            k = int(a.attack_ms / 1000 * SR)
            mix[:k] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
        with lock:
            b, p = play["buf"], play["pos"]
            keep = b[p:]
            k = min(int(0.05 * SR), len(keep), n)   # 50 ms crossfade between chunks
            if k:
                r = np.linspace(0, 1, k)
                mix[:k] = keep[-k:] * (1 - r) + mix[:k] * r
                keep = keep[:-k]
            play["buf"] = np.concatenate([keep, mix])
            play["pos"] = 0
            play["fade"] = False

    print(f"stream open: in={a.in_name!r} out={a.out_name!r} (Ctrl-C to stop)", flush=True)
    with sd.Stream(samplerate=SR, blocksize=BLK, channels=(1, 2),
                   device=(a.in_name, a.out_name), callback=cb):
        t_start = time.time()
        try:
            while True:
                time.sleep(TICK_S)
                now = time.time() - t_start
                f0 = rt_f0(ring[-int(0.35 * SR):])
                st["f0"].append(f0)
                st["f0"] = st["f0"][-STAB_TICKS:]
                w = [f for f in st["f0"] if f > 0]
                voiced = f0 > 0
                stable = (len(w) == STAB_TICKS and
                          1200 * np.log2(max(w) / min(w)) < STAB_CENTS)
                if st["mode"] == "idle" and stable:
                    st["mode"] = "sustain"
                    st["n_join"] += 1
                    tgt = float(np.median(w))
                    print(f"join #{st['n_join']}  {tgt:.0f} Hz", flush=True)
                    threading.Thread(target=render_into,
                                     args=(tgt, now, True), daemon=True).start()
                elif st["mode"] == "sustain":
                    if not voiced and all(f <= 0 for f in st["f0"][-4:]):
                        st["mode"] = "idle"
                        print("stop", flush=True)
                        with lock:
                            play["fade"] = True
                    else:
                        with lock:
                            left = (len(play["buf"]) - play["pos"]) / SR
                        if voiced and left < REFILL_AT:
                            tgt = float(np.median(w)) if w else 0
                            if tgt:
                                threading.Thread(target=render_into,
                                                 args=(tgt, now, False),
                                                 daemon=True).start()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
