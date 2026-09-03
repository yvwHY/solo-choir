"""sustain_live.py — 長音觸發同步 live 原型 v0（08-05；G28 Instead 案②）

你唱到穩定長音（~0.26s 內擺動 <±30c）→ 天使（S=+8ve Soprano-3、
B=−8ve Bass-1）batch 渲染 1.2s 塊進場跟唱；你還在撐就續塊、你收音
它們 150ms 淡出。嘴＝你當下的母音（回文循環鋪滿塊長＝units/vol 都有
自然微變化，不是凍結單幀）、f0＝你當下音高×固定倍率＋去同步顫音、
全過 enhancer＝應答式終判同款音路。

為什麼能活（vs G28 流式 WORLD 之死）：長音＝準穩態＝不需要逐幀上下文；
音色死感已由 enhancer 解（08-05 §F）；units 直通＝你唱什麼母音天使唱
什麼母音。判決＝Harry live（他 08-05：「即時的東西要聽 live」）。

Run (DDSP venv):
  python sustain_live.py                      # USB PnP 進、AI-Micro 出
  python sustain_live.py --in-name X --out-name Y --gain 0.6
Ctrl-C 結束。
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
STAB_TICKS = 8            # ~0.26s 穩定窗
STAB_CENTS = 60.0         # 窗內 max-min
CHUNK_S = 1.2             # 每塊渲染秒數
REFILL_AT = 0.45          # 播剩 <0.45s 且仍穩定 → 續塊
RELEASE_S = 0.15
MIN_HZ, MAX_HZ = 70.0, 500.0
SOP = (f"{R.DDSP}/exp/combsub-m4-sop3/model_10000.pt", 2.0, 4.9, 0.75)
BAS = (B1, 0.5, 4.6, 0.5)


def rt_f0(x):
    """滑窗即時 f0：parselmouth ac 於最後 0.35s，回傳末端 voiced 中位。"""
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
    """母音片段回文循環鋪到 n_samp（接縫 30ms 交叉淡化）＝units/vol 有活味。"""
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

    print("載入兩張嘴＋enhancer …", flush=True)
    angels = [Angel(*SOP), Angel(*BAS)]
    for an in angels:                      # MPS 預熱
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
            k = min(int(0.05 * SR), len(keep), n)   # 塊間 50ms 交叉淡化
            if k:
                r = np.linspace(0, 1, k)
                mix[:k] = keep[-k:] * (1 - r) + mix[:k] * r
                keep = keep[:-k]
            play["buf"] = np.concatenate([keep, mix])
            play["pos"] = 0
            play["fade"] = False

    print(f"開流：in={a.in_name!r} out={a.out_name!r}（Ctrl-C 結束）", flush=True)
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
                    print(f"▶ 進場 #{st['n_join']}  {tgt:.0f} Hz", flush=True)
                    threading.Thread(target=render_into,
                                     args=(tgt, now, True), daemon=True).start()
                elif st["mode"] == "sustain":
                    if not voiced and all(f <= 0 for f in st["f0"][-4:]):
                        st["mode"] = "idle"
                        print("■ 收", flush=True)
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
