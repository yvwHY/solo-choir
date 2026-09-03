"""WORLD mouth for the v2 brain (port of harmony_brain world_replay.py).

Same offline batch WORLD that reached Harry's quality bar on 07-22 -- the
angel is synthesized from HIS OWN voice (formants intact, voice-ownership),
f0 replaced by the v2 brain's line. Streaming WORLD stays dead (G26/G28);
this is offline render only.

One brain pass now yields BOTH f0 philosophies from the same note line
(the 2x2 pitch-axis pair, worklog 07-24 §B):
  {tag}_target_*  -- clean per-note target f0 + portamento + synthetic vibrato
  {tag}_shift_*   -- Harry's own f0 transposed by the per-tick interval
                     (brain note - his sounding note): wobble preserved,
                     the WORLD-timbre twin of Beatrice's shift-style path
plus {tag}_notes.json so other mouths (beatrice_mouth.py) reuse the line.

Run (retraining venv):  venv/bin/python world_mouth.py <take.wav> [accent] [tag]

"""

import sys
import wave
from pathlib import Path

import numpy as np
import pyworld

from live import VoiceToTokens, token_to_midi
from brain_v2 import BrainV2
from pitch import SR, PitchTracker

HERE = Path(__file__).parent
FRAME_MS = 5.0
VIB_HZ, VIB_SEMI = 5.0, 0.12
PORTA = 0.15  # one-pole per 5 ms frame ~ 30 ms glide


def main():
    path = Path(sys.argv[1])
    accent = sys.argv[2] if len(sys.argv) > 2 else "cpdl"
    bpm = 80.0  # 16th @ 80 bpm = the v2 brain tick (0.1875 s)
    tag = sys.argv[3] if len(sys.argv) > 3 else f"angel_v2_world_{accent}"
    legato_gap = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    reg_lo = float(sys.argv[5]) if len(sys.argv) > 5 else 9.0
    ap_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
    key_arg = sys.argv[7] if len(sys.argv) > 7 else "0"
    anticipate = int(sys.argv[8]) if len(sys.argv) > 8 else 0

    with wave.open(str(path)) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    # key 歸一化（任務6，2026-07-26）：腦只懂 C 大調音級；take 實測是 A 大調
    # （音階覆蓋 84% vs C 的 65%）＝此前所有 render 腦都在非母語環境工作。
    # "auto"＝keydet 音階隸屬度偵測 → 輸入 f0 移進 C 框架、腦輸出移回真實調。
    if key_arg == "auto":
        from keydet import KeyDetector
        from pitch import yin_f0
        kd = KeyDetector()
        for ts in range(0, len(mic) - 2048, 1024):   # 與 keydet 驗證同路徑
            f = yin_f0(mic[ts:ts + 2048].astype(np.float32), SR)
            kd.push(float(f) if f else None)
        k = kd.key()
        root = k["root"] if k else 0
        k_shift = (0 - root) % 12
        if k_shift > 6:
            k_shift -= 12
        print(f"key detect: scale root {k['name'] if k else '?'} -> normalize {k_shift:+d} st")
    else:
        k_shift = int(key_arg)

    # 1) brain pass: same tick loop as the live pipeline
    brain, v2t, tracker = BrainV2(accent=accent), VoiceToTokens(), PitchTracker()
    step_samps = int((60.0 / bpm) * 0.25 * SR)
    notes, heard, prev = [], [], None
    legato = {"gap": 0, "note": None}
    n_ant = n_agree = 0            # 預感命中統計（任務5）
    for tick_start in range(0, len(mic) - 1024, step_samps):
        tracker.push(mic[tick_start : tick_start + step_samps])
        f_in = tracker.latest                 # rounded MIDI（非 Hz）
        if f_in is not None and k_shift:
            f_in = int(f_in) + k_shift        # 進腦前移到 C 框架（MIDI 域＝整數加法）
        # 預感模式：先用腦對「他下一顆音」的預報決定天使的音（與他同時落地），
        # 真實 token 隨後照常餵進去；預報與反應式選擇一致率一併統計。
        ant = brain.step_anticipate(f_in is not None) if anticipate else None
        s = v2t.token(f_in)
        h = v2t.prev  # his sounding note (brain space), for the shift-mode interval
        tok = brain.step(s)
        if ant is not None:
            n_ant += 1
            n_agree += int(ant == tok)
            tok = ant
        m = token_to_midi(tok, prev)
        prev = m
        if m is None and legato["note"] is not None and legato["gap"] < legato_gap:
            legato["gap"] += 1
            m = legato["note"]
        else:
            legato["gap"] = 0
            legato["note"] = m
        notes.append(None if m is None else m - (v2t.shift or 0) - k_shift)
        heard.append(None if h is None else h - (v2t.shift or 0) - k_shift)

    # register fold: keep the angel within [-reg_lo, +9] semitones of your
    # own median pitch, so it sings near you instead of in the basement
    ref = float(np.median(tracker.register)) if tracker.register else 55.0
    for i, n in enumerate(notes):
        if n is None:
            continue
        while n < ref - reg_lo:
            n += 12
        while n > ref + 9:
            n -= 12
        notes[i] = n

    # 2) WORLD analysis of the mic (once)
    print("WORLD analysis...")
    f0_mic, t = pyworld.harvest(mic, SR, frame_period=FRAME_MS)
    sp = pyworld.cheaptrick(mic, f0_mic, t, SR)
    ap = pyworld.d4c(mic, f0_mic, t, SR)

    # Borrow every voiced frame EXCEPT clear breath (high aperiodicity) and
    # harvest octave errors. Keep onset/transition frames: replacing them
    # kills the natural attacks and turns the voice electronic (A2 lesson).
    voiced = f0_mic > 0
    apm = ap.mean(axis=1)
    midi_f0 = np.where(voiced, 69 + 12 * np.log2(np.maximum(f0_mic, 1) / 440), np.nan)
    med = np.copy(midi_f0)
    for k in np.flatnonzero(voiced):
        win = midi_f0[max(0, k - 4) : k + 5]
        med[k] = np.nanmedian(win)
    octave_err = voiced & (np.abs(midi_f0 - med) > 6)
    good = voiced & (apm < 0.8) & ~octave_err
    if not good.any():
        good = voiced
    if not good.any():
        sys.exit("no voiced frames in take")
    first_good = np.flatnonzero(good)[0]
    idx = np.zeros(len(f0_mic), dtype=int)
    last = first_good
    for k in range(len(f0_mic)):
        if good[k]:
            last = k
        idx[k] = last
    sp2, ap2 = sp[idx], np.clip(ap[idx] * ap_scale, 0.0, 1.0)

    # 3a) target-mode f0: per-frame target from tick notes + portamento + vibrato
    f0_target = np.zeros(len(f0_mic))
    cur = 0.0
    for k in range(len(f0_mic)):
        tick = min(int(t[k] * SR / step_samps), len(notes) - 1)
        note = notes[tick]
        target = 0.0 if note is None else 440 * 2 ** ((note - 69) / 12)
        if target > 0:
            cur = target if cur == 0 else cur + (target - cur) * PORTA
            vib = 2 ** (VIB_SEMI * np.sin(2 * np.pi * VIB_HZ * t[k]) / 12)
            f0_target[k] = cur * vib
        else:
            cur = 0.0

    # 3b) shift-mode f0: his own contour transposed by the per-tick interval
    # (note - heard); wobble/vibrato/transitions are HIS, no synthetic ones.
    # Carried shift through legato gaps; silent when he is silent -- that is
    # what a shift-style converter does. Octave-error frames take the local
    # median so a tracker blip doesn't masquerade as a shift artifact.
    shifts, last = [], None
    for n, h in zip(notes, heard):
        if n is None:
            shifts.append(None)
        else:
            if h is not None:
                last = n - h
            shifts.append(last)
    f0_base = np.where(octave_err, 440 * 2 ** ((med - 69) / 12), f0_mic)
    f0_shift = np.zeros(len(f0_mic))
    for k in range(len(f0_mic)):
        tick = min(int(t[k] * SR / step_samps), len(shifts) - 1)
        s_tick = shifts[tick]
        if s_tick is not None and f0_base[k] > 0:
            f0_shift[k] = f0_base[k] * 2 ** (s_tick / 12.0)

    def synth_write(f0_new, suffix):
        print(f"WORLD synthesis ({suffix})...")
        angel = pyworld.synthesize(f0_new, sp2, ap2, SR, frame_period=FRAME_MS)
        angel = angel[: len(mic)] if len(angel) >= len(mic) else np.pad(angel, (0, len(mic) - len(angel)))
        # singer-style amplitude envelope: 40 ms attack / 150 ms release at the
        # 5 ms frame rate, then upsampled -- hard 15 ms ramps read as "suddenly
        # switched on/off" (A2/B2 lesson)
        tgt = (f0_new > 0).astype(float)
        a_up, a_dn = 1 - np.exp(-FRAME_MS / 40.0), 1 - np.exp(-FRAME_MS / 150.0)
        g, env = 0.0, np.empty(len(tgt))
        for k in range(len(tgt)):
            g += (tgt[k] - g) * (a_up if tgt[k] > g else a_dn)
            env[k] = g
        hop = int(SR * FRAME_MS / 1000)
        gate = np.repeat(env, hop)[: len(angel)]
        gate = np.pad(gate, (0, len(angel) - len(gate)), constant_values=gate[-1] if len(gate) else 0)
        angel *= gate
        for name, sig in [(f"{tag}_{suffix}_angel", angel), (f"{tag}_{suffix}_mix", 0.5 * mic + 0.8 * angel)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(str(HERE / "out" / f"{name}.wav"), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
        print(f"wrote out/{tag}_{suffix}_angel.wav, out/{tag}_{suffix}_mix.wav")

    synth_write(f0_target, "target")
    synth_write(f0_shift, "shift")

    import json
    json.dump({"bpm": bpm, "sr": SR, "step_samps": step_samps,
               "notes": notes, "heard": heard, "shifts": shifts},
              open(HERE / "out" / f"{tag}_notes.json", "w"))
    active = [n for n in notes if n is not None]
    print(f"ticks {len(notes)}, active {len(active)/max(1,len(notes)):.0%}, "
          f"notes MIDI {min(active) if active else '-'}..{max(active) if active else '-'}")
    if n_ant:
        print(f"anticipation: {n_agree}/{n_ant} ticks agreed with the reactive "
              f"choice ({n_agree/n_ant:.0%}); angel lands with him, not a tick late")


if __name__ == "__main__":
    main()
