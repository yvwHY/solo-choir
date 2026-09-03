"""perform.py — 演出模式：預渲染的天使當伴奏帶，他唱 lead，兩者同一個空間

08-03 Harry：viva 的 open call 可能要一小段現場演出。排練預渲染的 **live 半場
08-01 判死於譜追蹤器（命中 37.9%）**，但那個追蹤器存在的理由是「機器要跟著
他」——**演出裡這個要求可以直接拿掉**：預渲染的和聲就是一條 backing track，
他跟著它唱。每個帶伴奏帶的歌手都是這樣工作的。沒有追蹤，就沒有 37.9%。
（代價是它從「即時樂器」變成「有伴奏帶的演唱」＝藝術決定，不是技術限制。）
離線那半場 08-02 重審已耳測過（ceiling 換 target 後 Harry 裁「變好了」）。

**路由（這條是硬規則，不是選項）：送出/回送。**
  乾聲走直路（麥克風 → PA），**電腦只出濕聲**。
理由有兩條，都在墓園裡：
  - 把他的乾聲從電腦繞一圈再放出來＝播他自己的延遲副本＝**G「--voice-delay」
    07-16 判死的 DAF 干擾**。
  - 麥克風與喇叭同時活著＝回授。應答式是靠「唱/播不同時」在結構上消滅它的，
    演出模式沒有那個保護。
而濕聲晚出來不要緊：**殘響本來就有 20ms predelay**，濕聲延遲聽感上只是
predelay 變長。所以延遲預算全部給乾聲（直路＝0），電腦這端不需要低延遲。
`--dry-out` 可以讓乾聲也從電腦出（沒有 PA、戴耳機自己試的時候用），預設關。

伴奏帶與他的人聲**送進同一個 send**＝同一個空間（08-03 Harry：「有空間感，
自己也進去」）。IR 與送出量預設就是那天定案的那組。

Run（DDSP venv；其實只用到 numpy/scipy/sounddevice）:
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
    ap.add_argument("backing", nargs="?", help="乾的天使 stems 混音（不要先加殘響）")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--reverb", type=float, default=0.20, help="送出量（08-03 定案）")
    ap.add_argument("--reverb-s", type=float, default=1.8)
    ap.add_argument("--reverb-trim", type=float, default=35.0)
    ap.add_argument("--backing-gain", type=float, default=1.0)
    ap.add_argument("--mic-send", type=float, default=1.0,
                    help="他的人聲送進殘響的量（相對伴奏帶）")
    ap.add_argument("--dry-out", type=float, default=0.0,
                    help="從電腦輸出的乾聲量。**預設 0＝乾聲走直路**（見檔頭）；"
                         "沒有 PA、戴耳機自己試的時候才調大")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--io-latency", default="high")
    ap.add_argument("--wait-tap", action="store_true",
                    help="等實體鍵才開始播（不然開場就跑）")
    ap.add_argument("--no-tap", action="store_true")
    a = ap.parse_args()

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return
    if not a.backing:
        ap.error("要給伴奏帶 wav（乾的 stems）")

    bak, sr = sf.read(a.backing, dtype="float64", always_2d=True)
    bak = np.ascontiguousarray(bak[:, 0])
    assert sr == SR, (a.backing, sr)
    ir = rv.make_ir(a.reverb_s, trim_db=a.reverb_trim)
    conv = rv.ConvLive(ir, 1024)

    tap = None if a.no_tap else TapListener()
    if tap is not None and tap.err:
        print(f"[tap] bind {tap.port} 失敗（{tap.err}）→ 當作沒有實體鍵")
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
        # 伴奏帶與人聲進同一個 send＝同一個空間
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
            dump[n:n + frames] = outdata[:, 0] + mic      # 出去的＋他唱的
            st["n"] = n + frames

    print(f"perform. 伴奏帶 {len(bak)/SR:.1f}s｜空間 T60 {a.reverb_s} 截 "
          f"{a.reverb_trim:.0f}dB 送 {a.reverb}｜乾聲"
          f"{'走直路（電腦不出）' if a.dry_out == 0 else f'從電腦出 {a.dry_out}'}"
          f"｜實體鍵 {'關' if tap is None else '開'}"
          f"｜{'等按鍵開始' if a.wait_tap else '立刻開始'} -- Ctrl-C stops.",
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
                            print("  （按鍵：開始）", flush=True)
                        else:                       # 再按＝從頭
                            st["pos"] = 0
                            print("  （按鍵：回到開頭）", flush=True)
                with lock:
                    p, run = st["pos"], st["run"]
                if run and p >= len(bak):
                    break
                if time.perf_counter() - t0 > 2:
                    t0 = time.perf_counter()
                    print(f"  {p/SR:6.1f}/{len(bak)/SR:.1f}s｜mic rms "
                          f"{st['mic']:.3f}｜peak {st['peak']:.2f}"
                          f"｜io {st['iov']}/{st['oun']}"
                          f"{'｜⚠ 削峰 ' + str(st['clip']) if st['clip'] else ''}",
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
    print(f"peak {st['peak']:.2f}｜削峰塊數 {st['clip']}"
          f"｜io overflow/underflow {st['iov']}/{st['oun']}")
    if st["clip"]:
        print(f"⚠ 有削峰 → --gain {0.89/max(st['peak'],1e-9):.2f}")


if __name__ == "__main__":
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)     # 07-31 §H 血訓
    main()
