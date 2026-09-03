"""render_v3.py — 三聲部 render 推論器（brain v3 spec 開工順序 #3 的前半）

同一個 take、同一個直驅嘴，三個 cell 只換腦：
  v1     brain_v2 獨立跑兩次（trio v1 配方，互不聞）
  joint  v3_joint 單模型逐 tick 生 (upper, lower)，lower 看得到同 tick 的 upper
  serial brain_v2 出 upper → v3_serial 以 (lead, upper) 為條件生 lower

輸出（out/）：{tag}_upper_notes.json / {tag}_lower_notes.json（direct_mouth 格式）
＋ {tag}_stats.json（互撞/黏線統計，歸因用不判生死）。--run 再代跑直驅嘴×2
（upper→combsub-girl、lower→combsub-harry）並落 {tag}_stem.wav（雙天使）與
{tag}_trio.wav（take 0.5＋天使各 0.6）。

盲聽包最後用 blind_pack.py 餵三個 stem（同配方混音＋隨機化）。

Run (brain venv):
  .../260722_harmony_brain/venv/bin/python render_v3.py --mode joint --run
  --baseline 印 CPDL/chorale 語料真實分佈的同組統計（spec 終點2 的基準）。
"""
import argparse, json, subprocess, sys, wave
from pathlib import Path

import numpy as np
import torch

from live import HOLD, PITCH_OFFSET, REST, VoiceToTokens, token_to_midi
from brain_v2 import BrainV2
from pitch import SR, PitchTracker, yin_f0
from keydet import KeyDetector
from train_v3 import CORPUS, SRC, HarmonyTransformerV3

HERE = Path(__file__).parent
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
TEMPERATURE, TOP_K = 0.9, 8          # 同 brain_v2，A/B 只換腦不換取樣配方
BPM = 80.0                            # v2/v3 共用 tick（16 分音符 @80 = 187.5ms）
GIRL_RANGE = (52, 79)                 # E3–G5，girl 單元庫實測音域（07-26）
HARRY_RANGE = (43, 70)                # harry 單元庫實測音域
DISSONANT = {1, 2, 6, 10, 11}         # 互撞判定的音程類（07-26 D3 同口徑）
STICKY_RUN = 4                        # 黏線＝同音連走 ≥4 tick（0.75s）


class BrainV3:
    """v3 checkpoint 的逐 tick 取樣器。joint: step(lead)->(upper,lower)；
    serial: step_serial(lead, upper)->lower。context 全長餵（take << CTX）。

    indep＝獨立度旋鈕（推論期個性層，2026-07-28 Harry「同行太多、想要微微
    分開」回饋）：他換音的瞬間以機率 indep 掛留（boost HOLD，晚一兩拍才
    解決）；他唱長音時以機率 indep 走經過音（壓 HOLD）。兩天使各自擲骰，
    音高選擇仍是模型的——只偏節奏行為，不越權寫和聲。"""

    INDEP_BIAS = 3.0

    def __init__(self, mode, accent="cpdl", indep=0.0, stab=1, window=None):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(HERE / "checkpoints" / f"v3_{mode}" / "best.pt",
                        map_location=self.device, weights_only=True)
        self.model = HarmonyTransformerV3(ck["vocab"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.src = torch.tensor([SRC[accent]], device=self.device)
        self.ctx, self.phases, self.tick = [], [], 0
        self.indep = indep
        self.stab = stab                 # 最短音長（tick）：onset 後強制 HOLD
        self.window = window             # live 滑動窗（tokens）；None=全 context
        self.last = {0: REST, 1: REST}   # 每聲部上一 tick 的 token
        self.age = {0: 999, 1: 999}      # 每聲部現任音符已唱的 tick 數

    @torch.no_grad()
    def _sample(self, mask_rest, hold_bias=0.0):
        start = 0
        if self.window and len(self.ctx) > self.window:
            # 切點必須是 3 的倍數：voice embedding 用 idx%3，切歪＝聲部相位全錯
            start = (len(self.ctx) - self.window + 2) // 3 * 3
        x = torch.tensor([self.ctx[start:]], device=self.device)
        p = torch.tensor([self.phases[start:]], device=self.device)
        logits = self.model(x, p, self.src)[0, -1] / TEMPERATURE
        if mask_rest:
            logits[REST] = float("-inf")  # 他唱的時候天使唱（同 v2 規則）
        if hold_bias:
            logits[HOLD] = logits[HOLD] + hold_bias
        k = torch.topk(logits, TOP_K)
        return int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])

    def _push(self, tok, phase):
        self.ctx.append(tok)
        self.phases.append(phase)

    def _indep_bias(self, lead_tok, voice):
        """掛留/經過音的擲骰；天使自己在 REST 時不介入（HOLD 續 REST＝閉嘴）。"""
        if not self.indep or self.last[voice] == REST:
            return 0.0
        if lead_tok >= PITCH_OFFSET and float(torch.rand(1)) < self.indep:
            return +self.INDEP_BIAS      # 他換音，我先不動（掛留）
        if lead_tok == HOLD and float(torch.rand(1)) < self.indep:
            return -self.INDEP_BIAS      # 他長音，我動（經過音）
        return 0.0

    def _voices_for(self, lead_tok, phase):
        out = []
        for voice in (0, 1):
            # 穩定旋鈕（2026-07-28「音符太短」回饋）：onset 後 stab tick 內
            # 強制 HOLD 續唱，模型只在承諾期滿後才重新決定。他停唱時不強制。
            if (self.stab > 1 and self.age[voice] < self.stab
                    and lead_tok != REST and self.last[voice] != REST):
                tok = HOLD
            else:
                tok = self._sample(lead_tok != REST,
                                   self._indep_bias(lead_tok, voice))
            self._push(tok, phase)
            if tok >= PITCH_OFFSET:
                self.age[voice] = 1
            elif tok == HOLD:
                self.age[voice] += 1
            else:
                self.age[voice] = 999
            self.last[voice] = tok
            out.append(tok)
        return out[0], out[1]

    def step(self, lead_tok):
        phase = self.tick % 16
        self.tick += 1
        self._push(lead_tok, phase)
        return self._voices_for(lead_tok, phase)

    @torch.no_grad()
    def step_anticipate(self, voiced_hint=True, conf=0.0):
        """預感步（brain_v2 配方的 v3 版，2026-07-28 live「拉住歌者」回饋後
        重啟——每音滑音的病根已被按需滑音修掉，預測錯改為瞬跳修正）：
        不等下一 tick 的 lead，先 argmax 預測他的音（voiced_hint＝只預測
        音高不預測發聲，REST/HOLD 遮掉），據以取樣兩天使＝與他同時落地。
        隨後真實 token 到手時呼叫 commit_lead() 寫回（天使的選擇保留）。"""
        if not self.ctx:
            return None
        phase = self.tick % 16
        self.tick += 1
        start = 0
        if self.window and len(self.ctx) > self.window:
            start = (len(self.ctx) - self.window + 2) // 3 * 3
        x = torch.tensor([self.ctx[start:]], device=self.device)
        p = torch.tensor([self.phases[start:]], device=self.device)
        lg = self.model(x, p, self.src)[0, -1].clone()
        if voiced_hint:
            lg[REST] = float("-inf")
            lg[HOLD] = float("-inf")
        fore = int(torch.argmax(lg))
        # 低信心不提前（修法候選①）：沒把握就假設他續唱（HOLD），天使延續
        # 現在的音——初次聽的旋律命中率只有 ~10-16%（G29 基線），亂猜傷和聲
        if conf and float(torch.softmax(lg, -1)[fore]) < conf:
            fore = HOLD
        self._push(fore, phase)
        return fore, self._voices_for(fore, phase)

    def commit_lead(self, lead_tok):
        """把上一個預感 tick 的預測 lead 換成真實 token（context 誠實化）。"""
        self.ctx[-3] = lead_tok

    def step_serial(self, lead_tok, upper_tok):
        phase = self.tick % 16
        self.tick += 1
        self._push(lead_tok, phase)
        self._push(upper_tok, phase)
        lo = self._sample(lead_tok != REST)
        self._push(lo, phase)
        return lo


def fold(notes, lo, hi):
    out = []
    for n in notes:
        if n is not None:
            while n < lo:
                n += 12
            while n > hi:
                n -= 12
        out.append(n)
    return out


def line_stats(a, b):
    """兩條 sounding MIDI 線的互撞/黏線統計（None＝無聲）。
    全部以音程類（mod 12）計——八度疊唱聽感上仍是黏同一條線（v1 的
    girl 嘴 +12 之後 same-note 永遠測不到，改口徑才跨 cell 可比）。"""
    both = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not both:
        return {"ticks_both": 0}
    unison = [(x - y) % 12 == 0 for x, y in both]
    diss = [abs(x - y) % 12 in DISSONANT for x, y in both]
    runs, r = [], 0
    for u in unison + [False]:          # 尾端 flush
        if u:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    sticky = sum(x for x in runs if x >= STICKY_RUN)
    return {"ticks_both": len(both),
            "unison": sum(unison) / len(both),
            "dissonant": sum(diss) / len(both),
            "sticky": sticky / len(both),
            "longest_unison_run": max(runs, default=0)}


def corpus_baseline():
    """CPDL＋chorale 真實 upper/lower 分佈（spec 終點2：先學真實分佈的基準）。"""
    split = json.loads((CORPUS / "corpus_split_v3.json").read_text())
    blocks = {"chorale": np.load(CORPUS / "chorale_v3.npz"),
              "cpdl": np.load(CORPUS / "cpdl_v3.npz")}
    agg = {"n": 0, "unison": 0.0, "dissonant": 0.0, "sticky": 0.0}
    for source, wkey in split["train"] + split["val"]:
        tok = blocks[source][f"tok_{wkey}"]  # (T,3) [upper, lead, lower]
        lines = []
        for ch in (0, 2):
            prev, line = None, []
            for t in tok[:, ch]:
                m = token_to_midi(int(t), prev)
                prev = m
                line.append(m)
            lines.append(line)
        s = line_stats(*lines)
        n = s.get("ticks_both", 0)
        if not n:
            continue
        agg["n"] += n
        for k in ("unison", "dissonant", "sticky"):
            agg[k] += s[k] * n
    for k in ("unison", "dissonant", "sticky"):
        agg[k] /= max(1, agg["n"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["v1", "joint", "serial"])
    ap.add_argument("--take", default=str(HERE / "../../../260722_harmony_brain/data/take.wav"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--indep", type=float, default=0.0,
                    help="獨立度：掛留/經過音的每次機率（joint 專用）")
    ap.add_argument("--stab", type=int, default=1,
                    help="穩定度：最短音長 tick 數（1=現行；joint 專用）")
    ap.add_argument("--f0-mode", choices=["target", "shift", "texture"], default="target",
                    help="嘴的 f0 來源：target＝合成目標線；shift＝他的真 f0 移調；"
                         "texture＝target 骨架＋真 f0 微紋理蓋印")
    ap.add_argument("--legato-gap", type=int, default=1)
    ap.add_argument("--run", action="store_true", help="代跑直驅嘴×2＋trio 混音")
    ap.add_argument("--baseline", action="store_true", help="只印語料真實分佈統計")
    a = ap.parse_args()
    if a.baseline:
        print(json.dumps(corpus_baseline(), indent=1))
        return
    if not a.mode:
        ap.error("--mode required (or --baseline)")
    tag = a.tag or f"trio_{a.mode}"
    torch.manual_seed(a.seed)

    with wave.open(a.take) as w:
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0
    mic = (raw.reshape(-1, nch)[:, 0] if nch > 1 else raw).astype(np.float64)

    # key 歸一化（同 world_mouth "auto"）：腦只懂 C 框架，take 實測 A 大調
    kd = KeyDetector()
    for ts in range(0, len(mic) - 2048, 1024):
        f = yin_f0(mic[ts:ts + 2048].astype(np.float32), SR)
        kd.push(float(f) if f else None)
    k = kd.key()
    root = k["root"] if k else 0
    k_shift = (0 - root) % 12
    if k_shift > 6:
        k_shift -= 12
    print(f"key detect: root {k['name'] if k else '?'} -> normalize {k_shift:+d} st")

    # 腦 pass：與 live 相同的 tick 迴圈；lead token 流一份餵所有腦
    if a.mode == "v1":
        brains = [BrainV2(accent="cpdl"), BrainV2(accent="cpdl")]
    elif a.mode == "joint":
        v3 = BrainV3("joint", indep=a.indep, stab=a.stab)
    else:
        v2, v3 = BrainV2(accent="cpdl"), BrainV3("serial")
    v2t, tracker = VoiceToTokens(), PitchTracker()
    step_samps = int((60.0 / BPM) * 0.25 * SR)
    lines = {"upper": [], "lower": []}
    prev = {"upper": None, "lower": None}
    legato = {v: {"gap": 0, "note": None} for v in lines}
    lead = []                            # 他的 sounding 線，供 lead-vs-angel 統計
    for tick_start in range(0, len(mic) - 1024, step_samps):
        tracker.push(mic[tick_start: tick_start + step_samps])
        f_in = tracker.latest
        lead.append(None if f_in is None else int(f_in))
        if f_in is not None and k_shift:
            f_in = int(f_in) + k_shift
        s = v2t.token(f_in)
        if a.mode == "v1":
            toks = {"upper": brains[0].step(s), "lower": brains[1].step(s)}
        elif a.mode == "joint":
            up, lo = v3.step(s)
            toks = {"upper": up, "lower": lo}
        else:
            up = v2.step(s)
            toks = {"upper": up, "lower": v3.step_serial(s, up)}
        for v in lines:
            m = token_to_midi(toks[v], prev[v])
            prev[v] = m
            lg = legato[v]
            if m is None and lg["note"] is not None and lg["gap"] < a.legato_gap:
                lg["gap"] += 1
                m = lg["note"]
            else:
                lg["gap"] = 0
                lg["note"] = m
            lines[v].append(None if m is None else m - (v2t.shift or 0) - k_shift)

    # 音域處理（spec 終點4：八度摺疊，接受摺疊瞬間包夾對調）
    # v1 cell 忠於 trio v1 配方：兩線摺到他的聲區、girl 嘴 -k 12 升八度
    ref = float(np.median(tracker.register)) if tracker.register else 55.0
    if a.mode == "v1":
        girl_key = 12
        sound = {v: fold(lines[v], ref - 9, ref + 9) for v in lines}
        notes = dict(sound)
        sound["upper"] = [None if n is None else n + 12 for n in sound["upper"]]
    else:
        girl_key = 0
        notes = sound = {"upper": fold(lines["upper"], *GIRL_RANGE),
                         "lower": fold(lines["lower"], *HARRY_RANGE)}

    for v in lines:
        path = HERE / "out" / f"{tag}_{v}_notes.json"
        json.dump({"bpm": BPM, "sr": SR, "step_samps": step_samps,
                   "notes": notes[v], "heard": lead}, open(path, "w"))
    stats = {"mode": a.mode, "seed": a.seed, "ticks": len(lines["upper"]),
             "upper_vs_lower": line_stats(sound["upper"], sound["lower"]),
             "lead_vs_upper": line_stats(lead, sound["upper"]),
             "lead_vs_lower": line_stats(lead, sound["lower"]),
             "inversion": _inversion(lead, sound),
             "motion": {v: _motion(lead, sound[v]) for v in sound}}
    json.dump(stats, open(HERE / "out" / f"{tag}_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

    if a.run:
        # vibrato 去同步：兩天使不同速率/相位＋起振延遲（機械齊振＝autotune 感來源之一）
        vib = {"upper": ("5.3", "0.0"), "lower": ("4.6", "0.5")}
        fmin = {"upper": "140", "lower": "80"}   # shift 模式曲線層摺疊地板
        for v, model, key in [("upper", "combsub-girl/model_30000.pt", girl_key),
                              ("lower", "combsub-harry/model_30000.pt", 0)]:
            r = subprocess.run([sys.executable, str(HERE / "direct_mouth.py"),
                                "--notes", str(HERE / "out" / f"{tag}_{v}_notes.json"),
                                "--take", a.take, "--tag", f"{tag}_{v}",
                                "--model", f"{DDSP}/exp/{model}",
                                "--vib-hz", vib[v][0], "--vib-phase", vib[v][1],
                                "--vib-onset-ms", "250",
                                "--f0-mode", a.f0_mode,
                                "--f0-min", fmin[v] if a.f0_mode == "shift" else "65",
                                "--key", str(key), "--run"])
            if r.returncode:
                sys.exit(r.returncode)
        import soundfile as sf
        up, _ = sf.read(HERE / "out" / f"{tag}_upper_angel.wav", dtype="float64")
        lo, _ = sf.read(HERE / "out" / f"{tag}_lower_angel.wav", dtype="float64")
        n = min(len(mic), len(up), len(lo))
        stem = 0.5 * up[:n] + 0.5 * lo[:n]
        stem = stem / (np.max(np.abs(stem)) + 1e-12) * 0.9
        sf.write(HERE / "out" / f"{tag}_stem.wav", stem, SR)
        trio = 0.5 * mic[:n] + 0.6 * up[:n] + 0.6 * lo[:n]
        trio = trio / (np.max(np.abs(trio)) + 1e-12) * 0.9
        sf.write(HERE / "out" / f"{tag}_trio.wav", trio, SR)
        print(f"wrote out/{tag}_stem.wav, out/{tag}_trio.wav")


def _motion(lead, line):
    """獨立度歸因：他換音時天使不動（掛留）率、他長音時天使動（經過音）率。"""
    sus = mov = lead_chg = lead_hold = 0
    for t in range(1, len(lead)):
        if None in (lead[t], lead[t - 1], line[t], line[t - 1]):
            continue
        if lead[t] != lead[t - 1]:
            lead_chg += 1
            sus += line[t] == line[t - 1]
        else:
            lead_hold += 1
            mov += line[t] != line[t - 1]
    return {"hold_on_lead_change": sus / max(1, lead_chg),
            "move_on_lead_hold": mov / max(1, lead_hold)}


def _inversion(lead, sound):
    """包夾結構破壞率：upper 低於 lead / lower 高於 lead 的 tick 比例。"""
    pairs_u = [(l, x) for l, x in zip(lead, sound["upper"]) if l is not None and x is not None]
    pairs_l = [(l, x) for l, x in zip(lead, sound["lower"]) if l is not None and x is not None]
    return {"upper_below_lead": sum(x < l for l, x in pairs_u) / max(1, len(pairs_u)),
            "lower_above_lead": sum(x > l for l, x in pairs_l) / max(1, len(pairs_l))}


if __name__ == "__main__":
    main()
