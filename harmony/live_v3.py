"""live_v3.py — v3 雙天使 live host（spec 開工順序 #4 的第一塊）

mic → EarV3（YIN tracker ＋ v3_joint 腦，每 tick 取樣 upper+lower）
    → 兩張直驅 DDSP 嘴（upper→combsub-girl、lower→combsub-harry，
      各自 out ring；world_live 的 bounded 窗＋SOLA＋splice 機械全繼承）
    → 喇叭（LAG 落後人聲；兩天使 0.6/0.6 混音）

與離線 render_v3 同一顆腦類（BrainV3，含 indep/stab 旋鈕），live 加滑動窗
（預設 768 tokens＝48s 記憶，實測每 tick 兩 forward 38ms；1536→81ms 也在
187.5ms 預算內）。key 假設同 world_live：C 大調（keydet 第一句鎖定屬後續，
「天使＝調性錨」07-27 裁決的 live 版）。

Run live（DDSP venv）:
  260724_ddsp_svc/venv/bin/python live_v3.py [--in-name X] [--lag 0.6]
File check（無音訊裝置，端到端驗證）:
  .../python live_v3.py --file take.wav out/live_v3_check.wav
"""
import argparse
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from keydet import KeyDetector  # noqa: E402
from live import HOLD, REST, VoiceToTokens, token_to_midi  # noqa: E402
from pitch import SR, PitchTracker, yin_f0  # noqa: E402
from render_v3 import GIRL_RANGE, HARRY_RANGE, BrainV3, fold  # noqa: E402
from rehearse_ab import ScoreTracker  # noqa: E402
from world_live import (TICK_SAMPS, DDSPMouth, StreamMouth, ExpressiveF0,  # noqa: E402
                        FRAME_HOP, HOP_FRAMES)

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)

KNEE_RMS = 0.019   # 軟發聲退化膝點（worklog 07-30：失真 14.3dB@rms .0195 → 18.3dB@.0055）


def mic_level_stats(x):
    """否證命題一的儀器（worklog 07-30 §E）：輸入 voiced rms 分布 vs 膝點。
    voiced 判定同 live 前端（yin_f0；rms<0.005 視為無聲不計，故 0.005–0.019
    的危險帶不會被儀器自己吃掉）。只印數字，裁決留給人。"""
    v = []
    for s in range(0, len(x) - 2048, 1024):
        seg = x[s:s + 2048]
        if yin_f0(seg.astype(np.float32)) is not None:
            v.append(float(np.sqrt(np.mean(seg * seg))))
    if not v:
        print("mic level (proposition 1): no voiced frames")
        return
    v = np.array(v)
    print(f"mic level (proposition 1): voiced {len(v)} frames | rms p10 "
          f"{np.percentile(v, 10):.4f} "
          f"p25 {np.percentile(v, 25):.4f} p50 {np.percentile(v, 50):.4f} | "
          f"below knee {KNEE_RMS}: {float((v < KNEE_RMS).mean()):.0%}", flush=True)


class EarV3:
    """tick audio in → (upper, lower) mic-space 絕對音（或 None）。
    world_live.Ear 的 v3 版：一份 lead token 流、兩聲部輸出、
    各自 legato 補洞、八度摺疊進各自嘴的實測音域。

    key="auto"＝第一句鎖定（07-27「天使＝調性錨」的 live 版）：開口的頭
    幾秒 keydet 聽出大調音階、鎖死 k_shift，之後不跟飄；鎖定前天使安靜
    （＝demo 弧線的獨唱開場）。key=<int>＝固定移調（0＝舊 C 大調行為）。"""

    LOCK_MIN, LOCK_CONF, LOCK_MAX = 130, 0.03, 300  # 有聲窗數（~3s / ~7s 上限）

    def __init__(self, indep=0.0, stab=1, window=768, legato_gap=1, key="auto",
                 anticipate=False, ant_conf=0.25, rehearse=None, reh_rate=1.0):
        self.brain = BrainV3("joint", indep=indep, stab=stab, window=window)
        self.lead_hist = []          # 每 tick 的 lead sounding pitch（排練檔錄製用）
        self.reh = None              # 排練模式（緊檔 v1）：腦旁路、天使唱排練的線
        if rehearse:
            self.reh = json.load(open(rehearse))
            self.reh_tracker = ScoreTracker(self.reh["lead"], "v3", rate0=reh_rate)
            print(f"[rehearse] {Path(rehearse).name}: score {len(self.reh['lead'])} "
                  f"ticks, r̂={reh_rate} (v3 forward filter)", flush=True)
        self.v2t = VoiceToTokens()
        self.tracker = PitchTracker()
        self.prev = {"upper": None, "lower": None}
        self.legato = {v: {"gap": 0, "note": None} for v in self.prev}
        self.legato_gap = legato_gap
        self.rng = {"upper": GIRL_RANGE, "lower": HARRY_RANGE}
        self.ant = anticipate
        self.ant_conf = ant_conf
        self.pending = None          # 預感：上一 tick 已定好的 (lead預測,)
        self.n_ant = self.n_agree = 0
        self.lead_prev = None        # 他的 sounding pitch（命中率統計用）
        if key == "auto":
            self.k_shift, self.kd = None, KeyDetector()
        else:
            self.k_shift, self.kd = int(key), None

    def _keylock(self, chunk):
        for t in range(0, len(chunk) - 2048, 1024):
            self.kd.push(yin_f0(chunk[t:t + 2048].astype(np.float32), SR))
        n, k = self.kd.hist.sum(), self.kd.key()
        if not k:
            return
        if n >= self.LOCK_MIN and k["conf"] >= self.LOCK_CONF:
            self.k_shift = (0 - k["root"]) % 12
            if self.k_shift > 6:
                self.k_shift -= 12
            print(f"[keydet] locked {k['name']} -> shift {self.k_shift:+d} st "
                  f"(conf {k['conf']:.2f})", flush=True)
        elif n >= self.LOCK_MAX:
            # 超時且信心不足＝histogram 是平的（真歌 keydet 不可靠，07-30 死案；
            # 07-31 Harry 場上 conf 0.01 被硬鎖 +4 st＝整場和聲錯調）——拒鎖
            # 垃圾，回退 shift 0 並大聲警告。正解仍是手動填整數 key。
            self.k_shift = 0
            print(f"[keydet] ⚠ conf {k['conf']:.2f} < {self.LOCK_CONF} → refusing to "
                  f"lock {k['name']}, falling back to shift 0 (= treat as C "
                  f"major). For a real song stop and set key to an integer "
                  f"(07-30 rule)", flush=True)

    def _to_notes(self, toks):
        """(upper_tok, lower_tok) → mic-space 絕對音對（legato 補洞＋摺疊）。"""
        out = {}
        for v, tok in zip(("upper", "lower"), toks):
            m = token_to_midi(tok, self.prev[v])
            self.prev[v] = m
            lg = self.legato[v]
            if m is None and lg["note"] is not None and lg["gap"] < self.legato_gap:
                lg["gap"] += 1
                m = lg["note"]
            else:
                lg["gap"] = 0
                lg["note"] = m
            if m is None:
                out[v] = None
                continue
            out[v] = fold([m - (self.v2t.shift or 0) - self.k_shift],
                          *self.rng[v])[0]
        return out["upper"], out["lower"]

    def tail_tick(self):
        """他唱完之後，讓天使自己再走一格（08-04）。

        Harry 08-03 實戴觀察：「只有在長音才會顯得自主」——`indep` 的兩個觸發
        點（他長音／他換音）都掛在**他的**音符事件上，所以應答式的短樂句裡這
        個旋鈕大半時間沒有觸發機會。回應的尾巴是繞開這個限制的地方：那段時間
        他本來就在硬靜音裡不出聲，天使可以自己走。

        lead 餵 HOLD＝當作他把最後那個音延長著，腦於是在同一個和聲上繼續移動
        （而不是重新開一句）。不碰 tracker、不碰 v2t——他的音符線與八度鎖定
        完全不受影響。"""
        self.lead_hist.append(self.lead_prev)
        return self._to_notes(self.brain.step(HOLD))

    def tick(self, chunk):
        """處理一個完成的 tick，回傳要 append 進 notes 的新音對列表。
        反應式＝[本 tick]；預感式＝[（首次含本 tick）, 下一 tick 的預定音]
        ——下一 tick 的音在音訊到來前就已存在＝天使與他同時落地。"""
        self.tracker.push(chunk)
        if self.k_shift is None:            # 第一句：只聽調，天使還不進場
            self._keylock(chunk)
            if self.k_shift is None:
                self.lead_hist.append(None)
                return [(None, None)]
        f_in = self.tracker.latest
        if f_in is not None and self.k_shift:
            f_in = int(f_in) + self.k_shift
        s = self.v2t.token(f_in)
        if self.reh is not None:            # 排練模式：對位取代腦
            self.lead_prev = token_to_midi(s, self.lead_prev)
            self.lead_hist.append(self.lead_prev)
            self.reh_tracker.observe(self.lead_prev)
            j = int(round(self.reh_tracker.p + self.reh_tracker.rate))
            if 0 <= j < len(self.reh["lead"]):
                return [(self.reh["upper"][j], self.reh["lower"][j])]
            return [(None, None)]           # 譜走完＝天使收
        new = []
        if self.ant and self.pending is not None:
            fore = self.pending[0]          # 本 tick 的音上一 tick 已定，只對答案
            self.brain.commit_lead(s)
            # 命中＝音高層一致（他 HOLD 時真 token 是 HOLD、預測被迫出音高
            # token，字面比對必不中——比「sounding pitch」才誠實）
            if s != REST:
                self.n_ant += 1
                self.n_agree += int(token_to_midi(fore, self.lead_prev)
                                    == token_to_midi(s, self.lead_prev))
            self.lead_prev = token_to_midi(s, self.lead_prev)
        else:
            new.append(self._to_notes(self.brain.step(s)))
            self.lead_prev = token_to_midi(s, self.lead_prev)
        if self.ant:
            nxt = self.brain.step_anticipate(voiced_hint=s != REST,
                                             conf=self.ant_conf)
            if nxt is not None:
                fore, toks = nxt
                self.pending = (fore,)
                new.append(self._to_notes(toks))
            else:
                self.pending = None
        self.lead_hist.append(self.lead_prev)
        return new


def _warmup_pair(fn, label):
    """跑 fn() 兩遍，量首次（付 MPS kernel 編譯成本）與熱身後耗時。"""
    t0 = time.perf_counter(); fn(); cold = time.perf_counter() - t0
    t0 = time.perf_counter(); fn(); warm = time.perf_counter() - t0
    print(f"[warmup] {label} first hop {cold * 1000:.0f}ms -> "
          f"post-warmup {warm * 1000:.0f}ms", flush=True)


def warmup(ear, mouths):
    """啟動階段、音訊流開始前空跑腦一次＋兩嘴各一 hop（07-29 worklog G「MPS
    冷啟 239ms」）：把首次 forward 的 kernel 編譯成本移出真實首個 hop。"""
    import torch
    brain = ear.brain
    window = brain.window or 768
    x = torch.zeros((1, window), dtype=torch.long, device=brain.device)
    p = torch.zeros((1, window), dtype=torch.long, device=brain.device)
    with torch.no_grad():
        _warmup_pair(lambda: brain.model(x, p, brain.src), "brain")
    for v, m in mouths.items():
        n = max(2, round(HOP_FRAMES * FRAME_HOP / m.BLOCK))
        au = m.torch.zeros((1, n * m.BLOCK), device=m.device)
        z = m.torch.zeros((1, n, 1), device=m.device)
        with m.torch.no_grad():
            _warmup_pair(lambda: m.model(m.encoder.encode(au, SR, m.BLOCK)[:, :n],
                                         z, z, spk_id=m.spk), f"{v} mouth")


def run(a):
    out_map = None
    if a.out_map:
        out_map = [int(t) for t in str(a.out_map).split(",")]
        if len(out_map) != 2 or min(out_map) < 0:
            raise SystemExit("--out-map 需要兩個非負整數（upper,lower），如 '0,1'")
    if a.dry_ch is not None and out_map is None:
        raise SystemExit("--dry-ch 需要配 --out-map")
    pre = None
    if a.pre_stems:
        if not a.rehearse:
            raise SystemExit("--pre-stems 需配 --rehearse")
        import soundfile as _sf
        pre = {}
        for v, p in zip(("upper", "lower"), a.pre_stems.split(",")):
            s, psr = _sf.read(p, dtype="float64", always_2d=True)
            assert psr == SR, (p, psr)
            pre[v] = np.ascontiguousarray(s[:, 0])
        print(f"[pre-stems] offline stems, aligned playback (no mouth, ahead "
              f"{a.pre_ahead} ticks): {a.pre_stems}", flush=True)
    cap = int((a.minutes * 60 + 5) * SR)
    mic_ring = np.zeros(cap, dtype=np.float64)
    rings = {"upper": np.zeros(cap, dtype=np.float64),
             "lower": np.zeros(cap, dtype=np.float64)}
    # 表現層（07-31 F21）：per-voice seed 不同＝兩天使去同步；--expr-gain 0＝關
    ex = (lambda seed: ExpressiveF0(seed, a.expr_gain)) if a.expr_gain > 0 \
        else (lambda seed: None)
    bf = int(round(a.bound_ms / 5.0)) if a.bound_ms else None
    MouthCls = StreamMouth if a.mouth == "stream" else DDSPMouth
    mouths = {} if pre is not None else \
        {"upper": MouthCls(rings["upper"], a.ddsp_repo, a.girl_model,
                           expr=ex(20260731), uv_gate=bool(a.uv_gate),
                           free_run=bool(a.free_run), bound_f=bf),
         "lower": MouthCls(rings["lower"], a.ddsp_repo, a.harry_model,
                           expr=ex(20260732), uv_gate=bool(a.uv_gate),
                           free_run=bool(a.free_run), bound_f=bf)}
    ear = EarV3(indep=a.indep, stab=a.stab, window=a.window, key=a.key,
                anticipate=a.anticipate, ant_conf=a.ant_conf,
                rehearse=a.rehearse, reh_rate=a.reh_rate)
    if a.rehearse:
        rk = ear.reh.get("k_shift")
        if a.key == "auto":
            print("⚠ rehearse mode with --key auto: if the live lock differs from the "
                  "rehearsal file the whole alignment shifts — the demo rule is "
                  "manual --key (07-30 checklist)")
        elif rk is not None and rk != ear.k_shift:
            print(f"⚠ rehearsal file k_shift {rk:+d} != this run {ear.k_shift:+d}, the "
              f"whole alignment will shift")
    warmup(ear, mouths)
    notes = {"upper": [], "lower": []}
    hop_samps = int(round(a.hop_ms / 1000.0 * SR))
    st = {"in": 0, "ticks": 0, "hop_next": hop_samps,
          "under": 0, "done": False}
    lag = int(a.lag * SR)
    lock = threading.Lock()
    t_tick = []

    # pre-stems 對位播放狀態：單一 stem 讀點（兩聲部同時間軸）＋ resync 計數
    ps = {"cur": None, "jumps": 0, "quiet": 0}
    XF = int(0.010 * SR)

    def do_pre(k):
        """tick k 收完（mic 前緣 (k+1)·TICK）：把追蹤器譜位的 stem 寫進 live
        第 k+ahead 格＝音訊先於拍點就位。連續譜位＝續讀無縫；偏差>門檻才
        跳針重對（10ms xfade）；他靜默 ≥2 tick → 寫零＝天使跟著收。"""
        tr = ear.reh_tracker
        b0 = (k + a.pre_ahead) * TICK_SAMPS
        if b0 < 0 or b0 + TICK_SAMPS > cap:
            return
        ps["quiet"] = ps["quiet"] + 1 if ear.lead_prev is None else 0
        # live 第 k+ahead 格起點的預測譜位（float tick）→ stem 樣本讀點
        tgt = (tr.p + (a.pre_ahead - 1) * tr.rate) * TICK_SAMPS
        n_stem = min(len(pre["upper"]), len(pre["lower"]))
        if ps["quiet"] >= 2 or not (0 <= tgt < n_stem - TICK_SAMPS):
            for v in ("upper", "lower"):
                rings[v][b0:b0 + TICK_SAMPS] = 0.0
            ps["cur"] = None
            return
        # 跳針時點的音樂性（v2）：漂移超標後不立刻跳，等目標譜位落在換音處
        # （lead 換音＝take 裡歌手重新起音＝剪接點天然隱形）才跳；漂移積到
        # 4× 門檻＝硬上限，換音等不到也跳（速度失配材料如 pair06 ratio 1.21
        # 的 1:1 續讀漂移是結構性的，60ms 軟門檻單獨用＝每兩 tick 跳一次針）
        thr = a.pre_resync_ms * SR / 1000.0
        drift = None if ps["cur"] is None else abs(ps["cur"] - tgt)
        lead = ear.reh["lead"]
        jt = int(tgt // TICK_SAMPS)
        onset = 0 < jt < len(lead) and lead[jt] != lead[jt - 1]
        jump = ps["cur"] is None or (drift > thr and (onset or drift > 4 * thr))
        c0 = int(round(tgt if jump else ps["cur"]))
        if c0 + TICK_SAMPS > n_stem:   # 續讀游標越過 stem 尾（延後跳針的漂移）
            jump, c0 = True, int(round(tgt))
        ramp = np.linspace(0.0, 1.0, XF)
        for v in ("upper", "lower"):
            seg = pre[v][c0:c0 + TICK_SAMPS].copy()
            if jump:
                if ps["cur"] is not None and int(round(ps["cur"])) + XF <= n_stem:
                    tail = pre[v][int(round(ps["cur"])):int(round(ps["cur"])) + XF]
                    seg[:XF] = seg[:XF] * ramp + tail * (1.0 - ramp)
                else:
                    seg[:XF] *= ramp          # 靜默後進場＝淡入
            rings[v][b0:b0 + TICK_SAMPS] = seg
        if jump and ps["cur"] is not None:
            ps["jumps"] += 1
        ps["cur"] = c0 + TICK_SAMPS

    def do_ticks(n_in):
        while (st["ticks"] + 1) * TICK_SAMPS <= n_in:
            k = st["ticks"]
            t0 = time.perf_counter()
            new = ear.tick(mic_ring[k * TICK_SAMPS: (k + 1) * TICK_SAMPS])
            t_tick.append(time.perf_counter() - t0)
            for up, lo in new:
                notes["upper"].append(up)
                notes["lower"].append(lo)
            if pre is not None:
                do_pre(k)
            st["ticks"] += 1

    def do_hops(t_end, closing=False):
        if pre is not None:      # stems 模式：嘴不在，音訊由 do_pre 直寫 ring
            return
        shared = {}          # 本輪 units 快取（兩張嘴同 content，見 DDSPMouth.hop）
        for v in ("upper", "lower"):
            mouths[v].hop(mic_ring, t_end, notes[v] if notes[v] else [None],
                          len(notes[v]), closing=closing, shared=shared)

    # ticks 與 hops 各自一條線（2026-07-31 §G 斷供案，Regime B）：原本同一條
    # worker 依序跑「全部待辦 tick → 一個 hop」，腦 tick p95 在桌面負載下會飆
    # 242–351ms（>187.5 預算；anticipate 兩次 forward 再翻倍）——嘴明明 25ms
    # 就能出一 hop，卻排在腦後面餓死（實測 starved 21.9s/44s、margin -16s）。
    # 拆線後嘴不再等腦。安全性：notes 只增不減（tick 線 append、hop 線快照
    # len 後索引，GIL 下安全）；mouth 的 lim 本來就以 n_ticks 上限自我節制，
    # 腦落後時嘴只是少建一點、下一 hop 補上（與 file 模式語意一致）；
    # st["hop_next"] 只有 hop 線碰。MPS 兩線併發由 command queue 序列化。
    def tick_worker():
        while not st["done"]:
            with lock:
                n_in = st["in"]
            do_ticks(n_in)
            time.sleep(0.005)

    def hop_worker():
        while not st["done"]:
            with lock:
                n_in = st["in"]
            if n_in >= st["hop_next"]:
                t_end = st["hop_next"]
                do_hops(t_end)
                st["hop_next"] = t_end + hop_samps
            else:
                time.sleep(0.005)

    def report():
        tt = np.array(t_tick) * 1000 if t_tick else np.zeros(1)
        for v, m in mouths.items():
            th = np.array(m.t_hop) * 1000 if m.t_hop else np.zeros(1)
            te = np.array(m.t_enc) * 1000 if m.t_enc else np.zeros(1)
            tf = np.array(m.t_fwd) * 1000 if m.t_fwd else np.zeros(1)
            print(f"{v}: hops {len(th)} p50 {np.percentile(th, 50):.0f} "
                  f"p95 {np.percentile(th, 95):.0f} ms "
                  f"(enc p50 {np.percentile(te, 50):.0f} / "
                  f"fwd p50 {np.percentile(tf, 50):.0f})", flush=True)
        print(f"brain tick: p50 {np.percentile(tt, 50):.0f} "
              f"p95 {np.percentile(tt, 95):.0f} ms vs tick {TICK_SAMPS / SR * 1000:.1f}")
        if ear.n_ant:
            print(f"anticipation: {ear.n_agree}/{ear.n_ant} predictions hit "
                  f"({ear.n_agree / ear.n_ant:.0%}), angel lands with him")
        if ear.reh is not None:
            tr = ear.reh_tracker
            print(f"rehearse: score pos {tr.p}/{len(ear.reh['lead'])} "
                  f"r̂={tr.rate:.2f} MAP jumps x{tr.jumps}", flush=True)
        if pre is not None:
            print(f"pre-stems: resync x{ps['jumps']}"
                  f" (ahead {a.pre_ahead} ticks, threshold {a.pre_resync_ms:.0f}ms)",
                  flush=True)

    if a.file:
        import soundfile as _sf                  # stdlib wave 不吃 float wav（pairs 是 float32）
        raw, in_sr = _sf.read(a.file[0], dtype="float64", always_2d=True)
        assert in_sr == SR, (a.file[0], in_sr)
        mic = np.ascontiguousarray(raw[:, 0])
        mic_ring[: len(mic)] = mic
        if a.notes_json:      # 重放既有音符線＝跳過腦（單一變因 A/B 用）
            d = json.load(open(a.notes_json))
            notes["upper"], notes["lower"] = d["upper"], d["lower"]
            print(f"notes replay: {a.notes_json} (brain off, harmony composition locked)")
        else:
            do_ticks(len(mic))
        for t_end in range(hop_samps, len(mic) + hop_samps, hop_samps):
            do_hops(min(t_end, len(mic)), closing=t_end >= len(mic))
        report()
        angels = 0.6 * rings["upper"][: len(mic)] + 0.6 * rings["lower"][: len(mic)]
        for name, sig in [(a.file[1], angels),
                          (a.file[1].replace(".wav", "_mix.wav"), 0.5 * mic + angels)]:
            peak = max(1e-9, np.abs(sig).max())
            with wave.open(name, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((sig / peak * 0.9 * 32767).astype(np.int16).tobytes())
            print("wrote", name)
        if not a.notes_json:      # 音符線落檔＝下次可 --notes-json 原樣重放
            nj = a.file[1].replace(".wav", "_notes.json")
            d = dict(notes)
            if not a.anticipate and a.rehearse is None:
                # 反應式＝notes 與 lead 逐 tick 對齊 → 可當排練檔（--rehearse）
                d["lead"], d["k_shift"] = ear.lead_hist, ear.k_shift
            json.dump(d, open(nj, "w"))
            print("wrote", nj + (" (includes the lead line = usable as a rehearsal file)"
                                 if "lead" in d else ""))
        mic_level_stats(mic)
        return

    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return

    def cb(indata, outdata, frames, t, status):
        if status:                 # PortAudio 旗標＝輸入鏈波波的直接證據
            if status.input_overflow:
                st["cb_iov"] = st.get("cb_iov", 0) + 1
            if status.output_underflow:
                st["cb_oun"] = st.get("cb_oun", 0) + 1
        n = st["in"]
        mic_ring[n: n + frames] = indata[:, 0]
        pos = n + frames - lag
        outdata[:] = 0
        if pos > 0:
            lo = max(0, pos - frames)
            if out_map is None:                      # 舊路徑：兩天使混合、雙聲道同內容
                seg = (0.6 * rings["upper"][lo:pos]
                       + 0.6 * rings["lower"][lo:pos]) * a.gain
                outdata[:, 0] = np.pad(seg, (frames - len(seg), 0))
                if outdata.shape[1] > 1:
                    outdata[:, 1] = outdata[:, 0]
            else:                                    # per-voice 路由（F11 物理源分離）
                for v, c in zip(("upper", "lower"), out_map):
                    seg = rings[v][lo:pos] * (0.6 * a.gain)
                    outdata[:, c] += np.pad(seg, (frames - len(seg), 0))
            # starved = a mouth owes audio AT THE PLAYHEAD, so ask the note
            # of the tick pos sits in -- not the newest notes at the mic
            # head, one lag ahead: that gate charged the angel's entry as
            # starved while pos was still in the pre-entry silence where
            # the ring is legitimately zero (07-28 §K), and charged every
            # phrase ending the same way (the "slow climb" after it).
            tk = pos // TICK_SAMPS
            if any(pos > m.frontier and tk < len(notes[v])
                   and notes[v][tk] is not None for v, m in mouths.items()):
                st["under"] += frames
        if a.dry_ch is not None:                     # 乾聲直通（live，不延遲）
            if a.dry_ch < 0:
                outdata += indata[:, :1]
            else:
                outdata[:, a.dry_ch] += indata[:, 0]
        with lock:
            st["in"] = n + frames

    threading.Thread(target=tick_worker, daemon=True).start()
    threading.Thread(target=hop_worker, daemon=True).start()
    dev = (a.in_name, a.out_name)
    n_out = 2 if out_map is None else 1 + max(
        out_map + ([a.dry_ch] if a.dry_ch is not None and a.dry_ch >= 0 else []))
    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, n_out),
                   device=dev, callback=cb, latency=a.io_latency):
        key_txt = "key auto-lock（第一句聽調，天使隨後進場）" if a.key == "auto" \
            else f"key shift {a.key} st"
        print(f"live v3. lag {a.lag * 1000:.0f} ms, {key_txt}, "
              f"indep {a.indep} stab {a.stab} -- Ctrl-C stops.", flush=True)
        try:
            while st["in"] < cap - SR * 10:
                time.sleep(2)
                cur = {v: next((x for x in reversed(notes[v]) if x is not None), None)
                       for v in notes}
                if any(m.phrase is not None for m in mouths.values()):
                    margin = (min(m.frontier for m in mouths.values())
                              - (st["in"] - lag)) / SR
                    mtxt = f"{margin * 1000:+5.0f} ms"
                else:
                    mtxt = "  rest"      # frontier 合法凍結，margin 無意義
                bt = np.array(t_tick[-32:]) * 1000 if t_tick else np.zeros(1)
                hp = max(((np.percentile(np.array(m.t_hop[-32:]) * 1000, 95)
                           if m.t_hop else 0.0) for m in mouths.values()),
                         default=0.0)
                print(f"in {st['in'] / SR:6.1f}s | ticks {st['ticks']} "
                      f"| U {cur['upper']} L {cur['lower']} "
                      f"| margin {mtxt} "
                      f"| starved {st['under'] / SR:.1f}s "
                      f"| brain {np.percentile(bt, 95):.0f}ms hop {hp:.0f}ms"
                      f" | io {st.get('cb_iov', 0)}/{st.get('cb_oun', 0)}",
                      flush=True)
        except KeyboardInterrupt:
            pass
    st["done"] = True
    report()
    n = st["in"]
    if n > SR:                    # 實戴 session 收官：mic 落檔＋命題一數字
        stamp = time.strftime("%y%m%d_%H%M%S")
        path = HERE / "out" / f"live_mic_{stamp}.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((np.clip(mic_ring[:n], -1, 1)
                           * 32767).astype(np.int16).tobytes())
        print(f"mic dump: {path}", flush=True)
        if a.dump_out:            # 輸出 ring 落檔＝波波聲/接縫診斷素材
            sig = 0.6 * rings["upper"][:n] + 0.6 * rings["lower"][:n]
            op = HERE / "out" / f"live_out_{stamp}.wav"
            with wave.open(str(op), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((np.clip(sig, -1, 1) * 32767)
                              .astype(np.int16).tobytes())
            print(f"out dump: {op}", flush=True)
        mic_level_stats(mic_ring[:n])


if __name__ == "__main__":
    import signal as _sig
    # 背景 shell 起的父行程（如 nohup 的 lab server）SIGINT 是 SIG_IGN 且會
    # 遺傳——Stop 按了全空包（07-31 Harry 回報；§H 血訓變體）。強制還原。
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--notes-json", default=None,
                    help="file mode 重放既有音符線（前次 file 跑完自動落的 "
                         "*_notes.json）＝腦不跑、和音組成鎖定，單一變因 A/B 用")
    ap.add_argument("--rehearse", default=None,
                    help="排練模式（緊檔 v1，spec 2026-07-29）：載入排練檔"
                         "（反應式 file 跑完落的含 lead 的 *_notes.json），"
                         "ScoreTracker v3 對位你的 lead、天使唱排練好的線，"
                         "腦旁路；脫稿時天使＝排練錨。key 必須與排練檔一致")
    ap.add_argument("--reh-rate", type=float, default=1.0,
                    help="排練對位的速度先驗 r̂（譜tick/實tick；live 無外部速度"
                         "估計器前的固定值，07-30 消融：固定優於線上自估）")
    ap.add_argument("--pre-stems", default=None,
                    help="排練預渲染 stems 'upper.wav,lower.wav'（prerender_stems"
                         ".py 產物，時間軸＝排練 take）：天使改為對位播放離線 "
                         "express 品質音訊、嘴不載＝離線天花板上 live（08-01 "
                         "Harry「優化天使不是優化 live_v3」）。需配 --rehearse")
    ap.add_argument("--pre-ahead", type=int, default=1,
                    help="預寫提前量（tick）：tick k 收完把譜位 p̂ 的 stem 寫進 "
                         "live 第 k+N 格＝音訊先於拍點就位；1＝下一格，lag 只需"
                         "蓋 tick 對齊抖動（~0.05s 可用）")
    ap.add_argument("--pre-resync-ms", type=float, default=60.0,
                    help="stem 讀點與追蹤器譜位偏差超過此值才跳針重對（10ms "
                         "xfade）；以下＝連續續讀（無縫）")
    ap.add_argument("--indep", type=float, default=0.15)
    ap.add_argument("--stab", type=int, default=2)
    ap.add_argument("--window", type=int, default=768,
                    help="腦滑動窗 tokens（768=48s/38ms，1536=96s/81ms）")
    ap.add_argument("--key", default="auto",
                    help="auto＝第一句 keydet 鎖定；整數＝固定移調（0=C 大調舊行為）")
    ap.add_argument("--anticipate", action="store_true",
                    help="預感模式：預測他下一顆音、天使與他同時落地（配 --lag 0.45）")
    ap.add_argument("--ant-conf", type=float, default=0.25,
                    help="預感信心門檻：低於此機率不搶拍、假設續唱（0=關）")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--lag", type=float, default=0.6,
                    help="播放落後人聲秒數；要蓋 tick 187.5+hop 150+腦38-81+嘴2×25ms")
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--mouth", choices=["stream", "splice"], default="stream",
                    help="stream＝真串流（相位續接、每樣本合成一次、零接縫；"
                         "07-31 深夜）；splice＝舊 bounded 重渲染＋SOLA 貼尾")
    ap.add_argument("--bound-ms", type=float, default=None,
                    help="嘴重渲染窗（ms，預設 700）。hop 成本 ∝ 窗長；縮窗＝"
                         "省算力但分歧回捲最深只能修到窗內（決策延遲 ~200-400ms"
                         "，低於 400 開始有修不到的風險）")
    ap.add_argument("--hop-ms", type=float, default=150.0,
                    help="嘴渲染節奏（ms）。lag 地板≈guard 50＋本值＋渲染時間；"
                         "75＝衝 lag 0.3–0.35（MPS 佔用翻倍，starved 遙測驗證）")
    ap.add_argument("--free-run", type=int, default=1,
                    help="嘴腦解耦（07-31）：嘴不等 tick 決定、臨時維持現音、"
                         "腦的決定到了回捲彎入＝lag 可壓向 0.35–0.45；"
                         "0＝舊行為（渲染前緣被音符決定硬鎖）")
    ap.add_argument("--io-latency", choices=["high", "low"], default="high",
                    help="sd.Stream latency：high＝驅動大緩衝（抗 GIL/負載尖峰的"
                         "輸入丟失＝波波嫌疑），low＝舊行為")
    ap.add_argument("--dump-out", action="store_true",
                    help="live 收官時把天使輸出 ring 落檔（波波聲/接縫診斷）")
    ap.add_argument("--uv-gate", type=int, default=0,
                    help="呼吸/子音段天使壓低（v2c 防抖版）。預設 0＝關——"
                         "07-31 深夜 Harry A/B：遮罩抖動＝波波聲主兇，關閉後乾淨"
                         "且無呼吸投訴（表現層/串流嘴已改善其前提）；實戴場若"
                         "呼吸被唱回歸再開 1")
    ap.add_argument("--expr-gain", type=float, default=1.0,
                    help="表現層倍率（07-31 F21 規則版：偏置/漂移/不規則顫音/"
                         "進音彎，兩天使異 seed 去同步）；0＝退回固定顫音舊行為")
    ap.add_argument("--out-map", default=None,
                    help="per-voice 出力通道 'upper,lower'（如 '0,1'＝upper→ch0、"
                         "lower→ch1，rig 的 unit 分開收＝F11 物理源分離）。同 "
                         "solo_min 慣例：單一多通道裝置（AI-Micro/Aggregate），"
                         "絕不開兩條 stream（F7）。預設＝現行立體聲混合")
    ap.add_argument("--dry-ch", type=int, default=None,
                    help="配 --out-map：乾聲（你的 mic 直通）進哪個通道"
                         "（-1＝全部；省略＝不進任何通道＝07-29 rig 慣例，"
                         "燈只看天使）")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--ddsp-repo", default=DDSP)
    ap.add_argument("--girl-model", default=f"{DDSP}/exp/combsub-girl/model_30000.pt")
    # 07-31 Harry 拍板 live 換用 260730 重訓嘴（40.8min、07-30 盲聽平手；
    # 「換換看」實戴順裁）。舊嘴＝--harry-model {DDSP}/exp/combsub-harry/
    # model_30000.pt；離線 direct_mouth 預設不動＝歷史 cell 可重現。
    ap.add_argument("--harry-model",
                    default=f"{DDSP}/exp/combsub-harry-260730/model_30000.pt")
    a = ap.parse_args()
    run(a)
