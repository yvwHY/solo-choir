"""loopback_lab.py — 自我迴環三格梯：讓管線把 Harry 自己的旋律唱回來。

動機（2026-07-30 盲聽 blind_retrain_0730 判「差不多」）：2.7× 資料重訓後
autotune 感沒變 → **嘴（DDSP 轉換器）無罪釋放**，嫌疑移到「縫」＝腦的音符線
被渲染成 f0 曲線那一層（量化平直目標＋規則 porta＋人工 vibrato）。

三格梯把「腦選什麼音」這個變因整個拿掉——三格唱的都是**他自己的旋律**：
  格1 = 原聲（take 的一段，什麼都沒過）
  格2 = 直驅嘴 + 他自己的真實 f0 原樣注入（shift 模式、位移恆 0）＝只測轉換
  格3 = 直驅嘴 + 同一條旋律的量化音符線經 porta/vibrato 規則畫成 f0＝縫＋嘴
同段、同 ckpt、同位準。判讀：2≈1 而 3 autotune → 縫定罪；2 已 autotune → 共犯。

用法（vcclient-dev env，於 harmony/ 下）：
  python loopback_lab.py prep   # 切段 + 寫 seg_notes.json（notes=heard=他的線）
  # ...中間跑兩次 direct_mouth.py（指令由 prep 印出）...
  python loopback_lab.py pack   # 位準對齊 → stim_1..3.wav + key.json + 統計
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "loopback_0730")
TAKE = os.path.join(HERE, "..", "..", "..", "260722_harmony_brain", "data", "take.wav")
NOTES = os.path.join(HERE, "out", "angel_v2_world4keyed_notes.json")
SR, HOP = 44100, 512
T0, T1 = 131, 270          # tick 區間：起於休止後的 onset、止於下一個休止（26.1s）
DDSP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/"
        "260724_ddsp_svc")
CKPT = f"{DDSP}/exp/combsub-harry/model_30000.pt"


def prep():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(NOTES))
    heard, ss = d["heard"], int(d["step_samps"])
    a, b = T0 * ss, T1 * ss
    x, sr = sf.read(TAKE, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    assert sr == SR, sr
    seg = np.ascontiguousarray(x[a:b])
    sf.write(os.path.join(OUT, "seg_take.wav"), seg, SR, subtype="PCM_16")
    # notes = heard = 他的 lead notes（管線既有轉譜）：shift 模式位移恆 0，
    # target 模式則把同一條線量化渲染 → 兩格的旋律內容完全同源。
    line = heard[T0:T1]
    json.dump({"bpm": d["bpm"], "sr": SR, "step_samps": ss,
               "notes": line, "heard": line},
              open(os.path.join(OUT, "seg_notes.json"), "w"), indent=1)
    v = [m for m in line if m is not None]
    print(f"seg: ticks {T0}-{T1}  {a/SR:.2f}-{b/SR:.2f}s  {len(seg)/SR:.2f}s  "
          f"voiced {len(v)}/{len(line)} ticks  midi {min(v)}-{max(v)}")
    for tag, mode in (("lb2_ownf0", "shift"), ("lb3_rulef0", "target")):
        print(f"\npython direct_mouth.py --notes out/loopback_0730/seg_notes.json "
              f"--take out/loopback_0730/seg_take.wav "
              f"--tag loopback_0730/{tag} --model {CKPT} "
              f"--f0-mode {mode} --run")


def _load(p):
    y, sr = sf.read(p, dtype="float64")
    if y.ndim > 1:
        y = y[:, 0]
    assert sr == SR, (p, sr)
    return np.ascontiguousarray(y)


def _stats(y):
    """voiced hop 的音準統計：對最近半音的偏差（cents）＋顫音深度。"""
    import pyworld
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    v = f0 > 0
    midi = 69 + 12 * np.log2(np.maximum(f0, 1) / 440.0)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    sus = v & (dm < 0.35)                      # 持續段（排除換音過渡）
    # 顫音深度＝持續段 cents 對 5-hop 移動平均的擺動（去掉慢漂）
    ma = np.convolve(dev, np.ones(5) / 5, mode="same")
    rip = (dev - ma)[sus]
    return dict(voiced=float(v.mean()),
                cents_sd=float(np.std(dev[sus])) if sus.any() else 0.0,
                vib_cents=float(np.std(rip) * np.sqrt(2)) if sus.any() else 0.0,
                f0_med=float(np.median(f0[v])) if v.any() else 0.0)


def pack():
    cells = [("1_take_raw", os.path.join(OUT, "seg_take.wav"),
              "原聲：他自己唱的，未經任何轉換"),
             ("2_ownf0", os.path.join(OUT, "lb2_ownf0_angel.wav"),
              "直驅嘴＋他的真實 f0 原樣注入（shift 模式位移恆 0）＝只測轉換"),
             ("3_rulef0", os.path.join(OUT, "lb3_rulef0_angel.wav"),
              "直驅嘴＋同旋律量化音符線經 porta 33ms/leap-snap 4/vib 5Hz 0.12st "
              "渲染的 f0＝縫＋嘴")]
    ys = [_load(p) for _, p, _ in cells]
    n = min(len(y) for y in ys)
    ys = [y[:n] for y in ys]
    rms = [float(np.sqrt((y ** 2).mean())) for y in ys]
    tgt = float(np.median(rms))
    ys = [y * (tgt / r) for y, r in zip(ys, rms)]
    g = 0.9 / max(float(np.abs(y).max()) for y in ys)   # 全體同一增益，保住相對位準
    key = {}
    for i, ((name, src, desc), y) in enumerate(zip(cells, ys), 1):
        y = y * g
        sf.write(os.path.join(OUT, f"stim_{i}.wav"), y, SR, subtype="PCM_16")
        st = _stats(y)
        key[f"stim_{i}"] = dict(cell=name, source=os.path.basename(src), desc=desc,
                                dur_s=round(n / SR, 3),
                                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                                peak=round(float(np.abs(y).max()), 3), **
                                {k: round(v, 3) for k, v in st.items()})
        print(f"stim_{i} {name:11s} rms {key[f'stim_{i}']['rms']:.5f} "
              f"cents_sd {st['cents_sd']:6.2f}  vib {st['vib_cents']:5.2f}c  "
              f"voiced {st['voiced']:.2f}  f0med {st['f0_med']:.1f}Hz")
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote 3 stimuli + key.json to {OUT}/")


def _take_uv():
    """take 段的 unvoiced 幀（吸氣/擦音）遮罩＝評分的參考真值。"""
    import pyworld
    y = _load(os.path.join(OUT, "seg_take.wav"))
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    return f0 > 0


def _cpp(y, hops, n=2048):
    """指定幀的倒頻譜峰突出度（CPP）＝諧波柱強度；越低越噪音狀。

    先試過頻譜平坦度，但它被頻譜傾斜汙染（gate 後噪音支路的學習包絡本身很有
    色彩，平坦度反而下降）＝會給出反向的錯誤結論。CPP 只看週期性造成的倒頻譜
    峰，對包絡傾斜免疫，才是「有沒有諧波柱」的直答。真靜音幀排除（無意義）。
    """
    q0, q1 = int(SR / 800), int(SR / 65)
    w = np.hanning(n)
    idx = np.arange(q0, q1)
    A = np.vstack([idx, np.ones_like(idx)]).T
    vals = []
    for k in hops:
        s = k * HOP - n // 2
        if s < 0 or s + n > len(y):
            continue
        seg = y[s:s + n] * w
        if np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        lp = np.log(np.abs(np.fft.rfft(seg)) ** 2 + 1e-12)
        lc = np.log(np.abs(np.fft.irfft(lp, n=n))[q0:q1] + 1e-12)
        b = np.linalg.lstsq(A, lc, rcond=None)[0]
        vals.append(float(np.max(lc - A @ b)))
    return float(np.mean(vals)) if vals else 0.0


def _score(y, ref_v):
    """gate 驗收三數：整體 voiced%、吸氣段被唱出來的比率、voiced 段音準迴歸。"""
    import pyworld
    f0, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
    v = f0 > 0
    n = min(len(v), len(ref_v))
    v, r = v[:n], ref_v[:n]
    midi = 69 + 12 * np.log2(np.maximum(f0[:n], 1) / 440.0)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    sus = v & r & (dm < 0.35)                       # 只看 take 也 voiced 的持續段
    return dict(voiced=round(float(v.mean()), 3),
                breath_voiced=round(float(v[~r].mean()), 3),
                cents_sd=round(float(np.std(dev[sus])), 2) if sus.any() else 0.0,
                uv_cpp=round(_cpp(y, np.flatnonzero(~r)), 3),
                v_cpp=round(_cpp(y, np.flatnonzero(r)), 3))


def packg():
    """gate 前後對照：沿用 stim_1 的 rms 當共同位準，stim_1..3 不動。"""
    ref_v = _take_uv()
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2g_ownf0_gated", "lb2g_ownf0_angel.wav", "stim_2g.wav"),
            ("3g_rulef0_gated", "lb3g_rulef0_angel.wav", "stim_3g.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':17s} {'voiced%':>8s} {'breath_v%':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        print(f"{name:17s} {st['voiced']*100:8.1f} {st['breath_voiced']*100:10.1f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="同 " + name[0] + " 格，唯一差異＝--respect-unvoiced 開"
                     "（他的無聲幀注入 f0=0）",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2g.wav / stim_3g.wav + key.json to {OUT}/")


def packh():
    """native-uv + 合議 + 遲滯版（2h/3h）：只寫 2h/3h，既有 stim 全不覆寫。

    多兩個診斷欄：
      interior_v%   距離最近 take-voiced 幀 ≥2 hop 的吸氣幀被唱比率——排除
                    邊界，harvest 在輸出上的分析窗（>23ms）本來就會把相鄰
                    voiced 抹過來，那部分不是「呼吸被唱」而是量測滲透
      cut_on_v%     gate 誤切到 take-voiced 幀的比率（存疑判無聲的代價上限）
    """
    import direct_mouth as dm
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    vm = dm.voicing_mask(take, len(ref_v)) > 0
    print(f"gate: {100 * (1 - vm.mean()):.1f}% hops uv | "
          f"cut_on_voiced {100 * (ref_v & ~vm).sum() / ref_v.sum():.1f}%")
    vidx = np.flatnonzero(ref_v)
    interior = np.array([np.min(np.abs(vidx - k)) >= 2 if len(vidx) else True
                         for k in range(len(ref_v))]) & ~ref_v
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2g_gated", "stim_2g.wav", None),
            ("3g_gated", "stim_3g.wav", None),
            ("2h_native_uv", "lb2h_ownf0_angel.wav", "stim_2h.wav"),
            ("3h_native_uv", "lb3h_rulef0_angel.wav", "stim_3h.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':15s} {'voiced%':>8s} {'breath_v%':>10s} {'interior_v%':>12s} "
          f"{'cents_sd':>9s} {'uv_CPP':>7s} {'v_CPP':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        import pyworld
        f0o, _ = pyworld.harvest(y, SR, frame_period=1000.0 * HOP / SR)
        m = min(len(f0o), len(interior))
        iv = float((f0o[:m] > 0)[interior[:m]].mean())
        print(f"{name:15s} {st['voiced']*100:8.1f} {st['breath_voiced']*100:10.1f} "
              f"{iv*100:12.1f} {st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} "
              f"{st['v_cpp']:6.3f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="native-uv 注入（抽取即 (f0,uv) 成對）＋多特徵合議 uv"
                     "＋信心遲滯邊界",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                interior_voiced=round(iv, 3), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2h.wav / stim_3h.wav + key.json to {OUT}/")


def _lsd(y, ref, hops, n=2048):
    """吸氣段對原聲的對數頻譜距離（dB）；透傳應趨近 0。"""
    w = np.hanning(n)
    vals = []
    for k in hops:
        s = k * HOP - n // 2
        if s < 0 or s + n > min(len(y), len(ref)):
            continue
        a = 20 * np.log10(np.abs(np.fft.rfft(y[s:s + n] * w)) + 1e-9)
        b = 20 * np.log10(np.abs(np.fft.rfft(ref[s:s + n] * w)) + 1e-9)
        vals.append(np.sqrt(np.mean((a - b) ** 2)))
    return float(np.mean(vals)) if vals else 0.0


def packp():
    """源透傳版（2p/3p）：uv 段走原聲，只寫 2p/3p，既有 stim 全不覆寫。"""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2_ownf0", "stim_2.wav", None),
            ("3_rulef0", "stim_3.wav", None),
            ("2h_sentinel", "stim_2h.wav", None),
            ("3h_sentinel", "stim_3h.wav", None),
            ("2p_passthru", "lb2p_ownf0_angel.wav", "stim_2p.wav"),
            ("3p_passthru", "lb3p_rulef0_angel.wav", "stim_3p.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':14s} {'breath_v%':>10s} {'breathLSD_dB':>13s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        # LSD 要在同位準下比：把原聲縮到與本格 voiced rms 相同
        m = min(len(y), len(take))
        vm_s = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vm_s] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vm_s] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:14s} {st['breath_voiced']*100:10.1f} {lsd:13.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="uv 段＝他的原聲透傳（15ms 升餘弦交叉淡化、位準對齊到轉換"
                     "輸出），voiced 段＝轉換結果",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2p.wav / stim_3p.wav + key.json to {OUT}/")


def packq():
    """錨定版（2q/3q）：修掉「12 秒後吸不了氣」的樂句中間開洞。只寫 2q/3q。"""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2p_passthru", "stim_2p.wav", None),
            ("3p_passthru", "stim_3p.wav", None),
            ("2q_anchored", "lb2q_ownf0_angel.wav", "stim_2q.wav"),
            ("3q_anchored", "lb3q_rulef0_angel.wav", "stim_3q.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="源透傳＋gate 錨定於 harvest 無聲處±25ms（不在樂句中間開洞）"
                     "＋分窗透傳增益＋局部自適應合議門檻",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2q.wav / stim_3q.wav + key.json to {OUT}/")


def packr():
    """輸入 AGC 版（2r/3r）：轉換前把位準拉到訓練參考、轉換後還原。只寫 2r/3r。"""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2q_anchored", "stim_2q.wav", None),
            ("3q_anchored", "stim_3q.wav", None),
            ("2r_agc", "lb2r_ownf0_angel.wav", "stim_2r.wav"),
            ("3r_agc", "lb3r_rulef0_angel.wav", "stim_3r.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="q 版設定＋轉換前輸入 AGC 拉到訓練參考 rms 0.0566、"
                     "轉換後乘回 1/gain 還原位準結構",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_2r.wav / stim_3r.wav + key.json to {OUT}/")


def packe():
    """表現層版（3e）：r 版設定＋f0-mode express（規則合成微偏置/漂移/不規則
    顫音/進音彎，參數抄真人統計）。只寫 3e——② 不需要表現層（f0 已是真人）。"""
    ref_v = _take_uv()
    take = _load(os.path.join(OUT, "seg_take.wav"))
    tgt = float(np.sqrt((_load(os.path.join(OUT, "stim_1.wav")) ** 2).mean()))
    uvh = np.flatnonzero(~ref_v)
    rows = [("1_take_raw", "stim_1.wav", None),
            ("2r_agc", "stim_2r.wav", None),
            ("3r_agc", "stim_3r.wav", None),
            ("3e_express", "lb3e_exprf0_angel.wav", "stim_3e.wav")]
    key = json.load(open(os.path.join(OUT, "key.json")))
    print(f"{'cell':13s} {'breath_v%':>9s} {'breathLSD':>10s} {'cents_sd':>9s} "
          f"{'uv_CPP':>7s} {'v_CPP':>6s} {'crest':>6s}")
    for name, src, dst in rows:
        y = _load(os.path.join(OUT, src))
        if dst:
            y = y * (tgt / float(np.sqrt((y ** 2).mean())))
            assert np.abs(y).max() < 1.0, (dst, np.abs(y).max())
            sf.write(os.path.join(OUT, dst), y, SR, subtype="PCM_16")
        st = _score(y, ref_v)
        m = min(len(y), len(take))
        vs = np.repeat(ref_v.astype(float), HOP)[:m] > 0.5
        sc = (np.sqrt((y[:m][vs] ** 2).mean())
              / max(1e-12, np.sqrt((take[:m][vs] ** 2).mean())))
        lsd = _lsd(y, take * sc, uvh)
        crest = float(np.abs(y).max() / np.sqrt((y ** 2).mean()))
        print(f"{name:13s} {st['breath_voiced']*100:9.1f} {lsd:10.2f} "
              f"{st['cents_sd']:9.2f} {st['uv_cpp']:7.3f} {st['v_cpp']:6.3f} "
              f"{crest:6.1f}")
        if dst:
            key[dst.replace(".wav", "")] = dict(
                cell=name, source=src,
                desc="r 版設定＋f0-mode express：target 骨架＋規則表現層"
                     "（每音偏置/慢漂移/快抖/不規則顫音/進音彎，參數抄真人統計，"
                     "seed 20260731）",
                dur_s=round(len(y) / SR, 3),
                rms=round(float(np.sqrt((y ** 2).mean())), 5),
                peak=round(float(np.abs(y).max()), 3),
                breath_lsd_db=round(lsd, 2), crest=round(crest, 2), **st)
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\nwrote stim_3e.wav + key.json to {OUT}/")


if __name__ == "__main__":
    {"prep": prep, "pack": pack, "packg": packg, "packh": packh,
     "packp": packp, "packq": packq, "packr": packr, "packe": packe}[
        sys.argv[1] if len(sys.argv) > 1 else "prep"]()
