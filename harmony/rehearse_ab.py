"""rehearse_ab.py — 排練模式機制 A/B 離線裁決鷹架（spec 2026-07-29 §機制岔路）

吃「排練 take＋測試 take」兩個 wav，各機制產出對「他下一顆音」的預測，
印命中率表。**這支只裁機制、不出聲音**——腦側量測，嘴/DDSP 完全不碰。

機制（皆取最小可行版，不造大系統）：
  baseline  冷腦無 prior＝現行 live_v3 --anticipate（§L 11% 基線的離線復現）
  A 譜追蹤  排練 take 的 tick 音高線當譜，單調指標對位，預測＝譜上下一音
  B1 預熱   排練 take 的 token 流先跑進 BrainV3 的滑動窗（窗 768 tok=48s，
            塞不下整首＝spec 已預告的限制，測試 take 一長就被擠出去）
  B2 偏置   位置對齊後給排練音的 logit 加 λ。**離線的位置對齊直接用 A 的
            追蹤結果＝這就是選項 C 的形狀**（B2 單獨上線還需要自己的定位器）

口徑（同 worklog 07-28 §L 音高層，逐字對齊 live_v3.EarV3.tick 的統計）：
他每個非 REST tick 都對答案，比 sounding pitch 不比字面 token（他 HOLD 時
預測被迫出音高 token，字面比對必不中＝不公）。另出 onset 欄＝只算他真的
換音那些 tick（G29 「換音音高、遮 HOLD/REST」11–16% 的口徑）。

Run（DDSP venv，同 live_v3）:
  .../260724_ddsp_svc/venv/bin/python rehearse_ab.py --rehearse a.wav --test b.wav
同 take 冒煙（明知灌水，只驗鷹架）:
  ... --rehearse .../260722_harmony_brain/data/take.wav --test <同一個檔>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from keydet import KeyDetector  # noqa: E402
from live import (HOLD, PITCH_HI, PITCH_LO, PITCH_OFFSET, REST,  # noqa: E402
                  VoiceToTokens, token_to_midi)
from pitch import SR, PitchTracker, yin_f0  # noqa: E402
from render_v3 import BrainV3  # noqa: E402

TICK_SAMPS = int((60.0 / 80.0) * 0.25 * SR)      # 8268 @44.1k，同 world_live 腦脈衝
LOCK_MIN, LOCK_CONF, LOCK_MAX = 130, 0.03, 300   # 同 EarV3 的第一句鎖定門檻


# ---------------------------------------------------------------- 前端（EarV3 複刻）

def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x.astype(np.float64)


def lead_stream(path, key="auto", max_ticks=None, octave=None):
    """wav → lead token 流。EarV3 前端逐字複刻：YIN tracker → 第一句 keydet
    鎖定 k_shift（不跟飄）→ VoiceToTokens。鎖定前的 tick 不進流（＝EarV3 裡
    天使還沒進場那段）。回傳 (tokens, k_shift, octave)。

    octave 非 None＝直接把 VoiceToTokens 的 auto-octave 釘住，跳過它自己的
    warmup（live.py:69-77 用「這個 take 開頭 6 個音的中位數」決定八度，同一首
    歌的兩個 take 會各自決定，實測會差 12 半音＝A/B 的譜與 prior 整體錯一個
    八度）。main 的 --shared-octave 用它讓 test take 沿用 rehearse take 的。"""
    x = load(path)
    tracker, v2t = PitchTracker(), VoiceToTokens()
    if octave is not None:
        v2t.shift = int(octave)
    kd = KeyDetector() if key == "auto" else None
    k_shift = None if key == "auto" else int(key)
    toks = []
    for b in range(0, len(x) - TICK_SAMPS + 1, TICK_SAMPS):
        chunk = x[b:b + TICK_SAMPS]
        tracker.push(chunk)
        if k_shift is None:
            for t in range(0, len(chunk) - 2048, 1024):
                kd.push(yin_f0(chunk[t:t + 2048].astype(np.float32), SR))
            n, k = kd.hist.sum(), kd.key()
            if k and (n >= LOCK_MAX or (n >= LOCK_MIN and k["conf"] >= LOCK_CONF)):
                k_shift = (0 - k["root"]) % 12
                if k_shift > 6:
                    k_shift -= 12
                print(f"[keydet] {Path(path).name}: locked {k['name']} -> "
                      f"shift {k_shift:+d} st (conf {k['conf']:.2f})", flush=True)
            if k_shift is None:
                continue
        f_in = tracker.latest
        if f_in is not None and k_shift:
            f_in = int(f_in) + k_shift
        toks.append(v2t.token(f_in))
        if max_ticks and len(toks) >= max_ticks:
            break
    return toks, k_shift, v2t.shift


def sounding(toks):
    """token 流 → 每 tick 的 sounding pitch（HOLD 延續、REST=None）。"""
    out, prev = [], None
    for t in toks:
        prev = token_to_midi(t, prev)
        out.append(prev)
    return out


def midi_to_tok(m):
    if m is None or not (PITCH_LO <= m <= PITCH_HI):
        return None
    return int(m) - PITCH_LO + PITCH_OFFSET


# ---------------------------------------------------------------- 計分（§L 口徑）

class Hits:
    def __init__(self):
        self.n = self.hit = self.n_on = self.hit_on = 0
        self.prev = None                      # 他上一 tick 的 sounding pitch

    def add(self, pred_midi, actual_tok):
        act = token_to_midi(actual_tok, self.prev)
        if actual_tok != REST:
            ok = int(pred_midi is not None and pred_midi == act)
            self.n += 1
            self.hit += ok
            if actual_tok >= PITCH_OFFSET:    # 他真的換音的 tick（G29 口徑）
                self.n_on += 1
                self.hit_on += ok
        self.prev = act

    def add_tok(self, fore, actual_tok):
        self.add(token_to_midi(fore, self.prev), actual_tok)

    def skip(self, actual_tok):
        """沒有預測可對（第一 tick）——只推進 prev，同 live_v3 的 else 分支。"""
        self.prev = token_to_midi(actual_tok, self.prev)

    def row(self):
        p = f"{self.hit}/{self.n}" if self.n else "-"
        q = f"{self.hit_on}/{self.n_on}" if self.n_on else "-"
        return (f"{p:>10} {self.hit / self.n:6.1%}" if self.n else f"{p:>10}       ",
                f"{q:>10} {self.hit_on / self.n_on:6.1%}" if self.n_on else f"{q:>10}       ")


# ---------------------------------------------------------------- A：譜追蹤

class ScoreTracker:
    """最小可行對位（不造大系統）：排練 take 的 tick 音高線當譜，指標單調
    向前，每 tick 在 [p+1-BACK, p+1+FWD] 窗內挑與他當下 sounding pitch 最
    合的位置，同分取最靠近 p+1（＝等速前進的先驗）。跟丟＝窗內沒有 0 成本
    候選時仍照最小成本前進（v1 不做跟丟策略，spec 已標記為 A 的風險欄）。

    v2（2026-07-30 加，預設；`--track v1` 回到上面的原始行為）修三件實測出來的
    毛病，全部 additive：
      速度先驗   原版同分 tie-break 是 |j-(p+1)|＝偏好「每 tick 正好前進 1 格」。
                但他第二遍普遍唱得比第一遍快（實測 12 對的 譜/測試 tick 比
                0.99–1.27，均值 1.12），譜側每 tick 該前進 r̂ 格；分節素材同分
                候選又多，於是每 tick 少走一點、單調累積成永久落後（實測中位
                偏差可達 192 tick ≈ 36s，12 對裡 10 對 behind>70%）。改成錨在
                p+r̂、tie-break |j-(p+r̂)|，r̂ 由實際前進量線上 EMA 估。
      丟失重進入 局部成本滑動平均連續超門檻＝宣告跟丟，拿最近 BUF tick 的觀測
                對「全譜」做一次相關搜尋重新進入（允許往回跳＝分節重複段）。
      窗放寬     BACK 2→8：原版一旦超前就退不回來。
    """

    BACK, FWD = 2, 6                      # v1 窗（保留原值，--track v1 用）
    BACK_V2, FWD_V2 = 8, 10               # v2 窗：超前必須退得回來
    R_LO, R_HI, R_A = 0.6, 2.0, 0.05      # r̂ 的界與 EMA 係數
    LOST_WIN, LOST_TH, LOST_HOLD = 12, 2.6, 40   # 跟丟：12 tick 均成本 >2.6 半音
    BUF = 24                              # 重進入用的觀測緩衝（≈4.5s）

    R_WIN = 48                            # r̂ 用窗內斜率估，不用逐 tick EMA

    # v3（HMM forward filter）參數。07-31 12 對 27 組格點：面平坦（mean12
    # 31.7–34.4）＝增益來自機制非調參。本組＝hi6 最高（41.6）且 same-take
    # 迴歸不破（99.5/100；τ≥1.2 會讓多假設在自相似段坐錯等價段落、HOLD 延音
    # 對錯 → same-take 掉到 95.6）。
    V3_SIGMA = 1.0    # emission 軟度（半音）：exp(-cost/σ)
    V3_TAU = 0.8      # transition 對 r̂ 的離散度（tick）
    V3_EPS = 1e-3     # 每 tick 均勻漏失＝機率式重進入（取代 v2 的觸發式跳躍）
    V3_DMAX = 4       # 單 tick 最大前進格數
    V3_JUMP = 4       # MAP 位移超過此值計一次「跳躍」（純診斷，對齊 v2 報表）

    def __init__(self, score, mode="v2", rate0=1.0, adapt=False):
        self.score = score
        self.mode = mode
        self.adapt = adapt
        self.p = -1
        self.last = next((m for m in score if m is not None), None)
        self.rate0 = min(max(rate0, self.R_LO), self.R_HI)
        self.rate = self.rate0
        self.obs, self.hist, self.ptrs = [], [], []
        self.tick = self.last_jump = 0
        self.jumps = 0
        if mode == "v3":
            n = len(score)
            self._s = np.array([np.nan if m is None else float(m)
                                for m in score])
            # 起點先驗：與 v1/v2 同假設（測試 take 從素材頭開始）但留斜率餘裕
            self.alpha = np.exp(-np.arange(n) / 10.0)
            self.alpha /= self.alpha.sum()
            d = np.arange(self.V3_DMAX + 1)
            w = np.exp(-((d - self.rate) ** 2) / (2 * self.V3_TAU ** 2))
            self._tw = w / w.sum()
            self.p = 0

    @staticmethod
    def _cost(a, b):
        if a is None and b is None:
            return 0
        if a is None or b is None:
            return 10
        return min(abs(a - b), 9)

    def observe(self, m):
        if self.mode == "v1":
            return self._observe_v1(m)
        if self.mode == "v3":
            return self._observe_v3(m)
        return self._observe_v2(m)

    def _observe_v3(self, m):
        """離散貝氏 forward filter（07-30 §D2「多假設粒子」的精確版）：狀態＝
        譜位置，逐 tick 全後驗更新。分節重複段的歧義以多峰後驗持有，等證據
        自然消歧——取代 v2 單指標被迫「改主意」的觸發式重進入（5/12 對 ≥3 次）。
        狀態空間離散且小（譜長 ~300–830 格），forward 遞推是精確推斷，不需要
        粒子近似；純 numpy 無隨機＝決定性。速度先驗進 transition kernel
        （d∈0..DMAX，權重集中在 r̂），V3_EPS 均勻漏失＝常駐的機率式重進入。"""
        n = len(self._s)
        prop = np.zeros(n)
        for d, w in enumerate(self._tw):               # transition：前進 d 格
            if d == 0:
                prop += w * self.alpha
            else:
                prop[d:] += w * self.alpha[:-d]
        prop = (1.0 - self.V3_EPS) * prop / max(prop.sum(), 1e-300) \
            + self.V3_EPS / n
        if m is None:                                   # emission：同 _cost 表
            cost = np.where(np.isnan(self._s), 0.0, 10.0)
        else:
            cost = np.where(np.isnan(self._s), 10.0,
                            np.minimum(np.abs(self._s - m), 9.0))
        self.alpha = prop * np.exp(-cost / self.V3_SIGMA)
        s = self.alpha.sum()
        if s < 1e-300:                                  # 全滅保護：退回傳遞先驗
            self.alpha = prop
            s = self.alpha.sum()
        self.alpha /= s
        prev = self.p
        self.p = int(np.argmax(self.alpha))
        if abs(self.p - prev) > self.V3_JUMP:
            self.jumps += 1
        if self.score[self.p] is not None:
            self.last = self.score[self.p]

    def _observe_v1(self, m):
        lo = max(0, self.p + 1 - self.BACK)
        hi = min(len(self.score), self.p + 1 + self.FWD + 1)
        if lo >= hi:
            self.p = len(self.score) - 1
            return
        self.p = min(range(lo, hi),
                     key=lambda j: (self._cost(self.score[j], m),
                                    abs(j - (self.p + 1))))
        if self.score[self.p] is not None:
            self.last = self.score[self.p]

    def _observe_v2(self, m):
        self.tick += 1
        self.obs.append(m)
        if len(self.obs) > self.BUF:
            self.obs.pop(0)
        anchor = self.p + self.rate                    # 速度先驗：該落在這裡
        c = int(round(anchor))
        lo = max(0, c - self.BACK_V2)
        hi = min(len(self.score), c + self.FWD_V2 + 1)
        if lo >= hi:
            self.p = len(self.score) - 1
            return
        prev = self.p
        self.p = min(range(lo, hi),
                     key=lambda j: (self._cost(self.score[j], m),
                                    abs(j - anchor)))
        if self.score[self.p] is not None:
            self.last = self.score[self.p]
        # r̂ 預設「不線上調」：從指標自己的前進量估速度是自我實現的——它量到的
        # 正是它該修的那個落後，估出來只會更低（逐 tick EMA 收在 0.60–0.75、
        # 窗內斜率也一樣，真值 0.99–1.27）。實測消融：固定 r̂=tick 比 A/天花板
        # 0.574，改成線上自適應反而掉到 0.537。--rate-adapt 可開回自適應。
        if self.adapt:
            self.ptrs.append(self.p)
            if len(self.ptrs) > self.R_WIN:
                self.ptrs.pop(0)
            if len(self.ptrs) == self.R_WIN:
                slope = (self.ptrs[-1] - self.ptrs[0]) / (self.R_WIN - 1)
                self.rate = min(max(0.5 * self.rate0 + 0.5 * slope,
                                    self.R_LO), self.R_HI)
        if m is not None and self.score[self.p] is not None:
            self.hist.append(self._cost(self.score[self.p], m))
            if len(self.hist) > self.LOST_WIN:
                self.hist.pop(0)
            if (len(self.hist) == self.LOST_WIN
                    and sum(self.hist) / self.LOST_WIN > self.LOST_TH
                    and self.tick - self.last_jump >= self.LOST_HOLD):
                self._reenter()

    def _reenter(self):
        """跟丟＝拿最近 BUF tick 的觀測對全譜重新定位。允許往回跳（分節重複段
        本來就可能回到前面那一遍），這是 v1 單調指標做不到的。"""
        n = len(self.score)
        if sum(o is not None for o in self.obs) < 6:
            return
        best, bs = None, None
        for s in range(n):
            tot = cnt = 0
            for k, o in enumerate(self.obs):
                if o is None:
                    continue
                j = int(round(s + k * self.rate))
                if j >= n:
                    break
                tot += self._cost(self.score[j], o)
                cnt += 1
            if cnt >= 6:
                v = tot / cnt
                if best is None or v < best:
                    best, bs = v, s
        if bs is None:
            return
        self.p = min(max(int(round(bs + (len(self.obs) - 1) * self.rate)), 0), n - 1)
        if self.score[self.p] is not None:
            self.last = self.score[self.p]
        self.hist.clear()
        self.last_jump = self.tick
        self.jumps += 1

    def predict(self):
        """譜上的下一顆音。譜說休止時延用最後一顆實音（＝天使掛著不換），
        因為計分只算他有聲的 tick，回 None 等於自動判錯、不誠實。
        v2 走 p+r̂＝下一個 test tick 對應的譜位置（v1 固定 p+1）。"""
        j = self.p + 1 if self.mode == "v1" else int(round(self.p + self.rate))
        if 0 <= j < len(self.score) and self.score[j] is not None:
            return self.score[j]
        return self.last


def _rate0(score, test_toks, rate0=None):
    """r̂ 的起始值。離線裁決工具知道兩條流的長度，用 譜/測試 tick 比當先驗是
    誠實的（線上版得改成從排練 take 的速度或前幾秒自估）。"""
    if rate0 is not None:
        return float(rate0)
    return len(score) / max(len(test_toks), 1)


def run_A(test_toks, score, mode="v2", rate0=None, adapt=False):
    hits, tr, pending = Hits(), ScoreTracker(
        score, mode, _rate0(score, test_toks, rate0), adapt), None
    for s in test_toks:
        if pending is not None:
            hits.add(pending, s)
        else:
            hits.skip(s)
        tr.observe(hits.prev)
        pending = tr.predict()
    hits.tracker = tr                     # 讓 main 能報 r̂ 與重進入次數
    return hits


# ---------------------------------------------------------------- B：腦吃 prior

@torch.no_grad()
def anticipate(brain, voiced_hint, conf, bias_tok=None, lam=0.0):
    """BrainV3.step_anticipate 的偏置版。lam=0 且 bias_tok=None 時邏輯與原
    方法逐行等價（render_v3.py 零改動，這裡只複刻最小的那段加 λ）。"""
    if not brain.ctx:
        return None
    phase = brain.tick % 16
    brain.tick += 1
    start = 0
    if brain.window and len(brain.ctx) > brain.window:
        start = (len(brain.ctx) - brain.window + 2) // 3 * 3
    x = torch.tensor([brain.ctx[start:]], device=brain.device)
    p = torch.tensor([brain.phases[start:]], device=brain.device)
    lg = brain.model(x, p, brain.src)[0, -1].clone()
    if voiced_hint:
        lg[REST] = float("-inf")
        lg[HOLD] = float("-inf")
    if bias_tok is not None and lam:
        lg[bias_tok] = lg[bias_tok] + lam
    fore = int(torch.argmax(lg))
    if conf and float(torch.softmax(lg, -1)[fore]) < conf:
        fore = HOLD
    brain._push(fore, phase)
    return fore, brain._voices_for(fore, phase)


def brain_run(test_toks, brain, ant_conf, tracker=None, lam=0.0):
    """live_v3.EarV3.tick 預感分支的離線版：預測 → 下一 tick 對答案 →
    commit_lead 把真 token 寫回 context。tracker 非 None＝B2 的位置對齊。"""
    hits, pending = Hits(), None
    for s in test_toks:
        if pending is not None:
            hits.add_tok(pending, s)
            brain.commit_lead(s)
        else:
            brain.step(s)
            hits.skip(s)
        bias_tok = None
        if tracker is not None:
            tracker.observe(hits.prev)
            bias_tok = midi_to_tok(tracker.predict())
        nxt = anticipate(brain, s != REST, ant_conf, bias_tok, lam)
        pending = nxt[0] if nxt is not None else None
    return hits


def fresh_brain(a, preheat=None):
    torch.manual_seed(0)                      # 天使取樣可重現＝跨機制可比
    b = BrainV3("joint", indep=a.indep, stab=a.stab, window=a.window)
    if preheat:
        for s in preheat:
            b.step(s)
        b.tick = 0    # 測試 take 回到與 baseline 同一條 phase 格線＝單一變因
    return b


# ---------------------------------------------------------------- main

def main(a):
    same = Path(a.rehearse).resolve() == Path(a.test).resolve()
    reh, _, oct_r = lead_stream(a.rehearse, a.key, a.max_ticks)
    if same:
        test, oct_t = reh, oct_r
    else:
        test, _, oct_t = lead_stream(a.test, a.key, a.max_ticks,
                                     octave=oct_r if a.shared_octave else None)
    if oct_r != oct_t:
        print(f"⚠ auto-octave 不一致：rehearse {oct_r:+d} / test {oct_t:+d} 半音。"
              f"譜與 prior 會整體差 {oct_t - oct_r:+d} 半音，A/B 系數字無裁決效力"
              f"——加 --shared-octave 讓 test 沿用 rehearse 的八度。", flush=True)
    score = sounding(reh)
    print(f"rehearse {Path(a.rehearse).name}: {len(reh)} ticks "
          f"({len(reh) * TICK_SAMPS / SR:.1f}s) | "
          f"test {Path(a.test).name}: {len(test)} ticks "
          f"({len(test) * TICK_SAMPS / SR:.1f}s)", flush=True)

    warn = "SAME-TAKE, INFLATED" if same else ""
    rows = []
    if "base" in a.mechs:
        rows.append(("baseline 冷腦（無 prior）", brain_run(test, fresh_brain(a), a.ant_conf),
                     "§L/G29 基線 ~11%" if not same else warn))
    if "A" in a.mechs:
        hA = run_A(test, score, a.track, adapt=a.rate_adapt)
        note = "同 take 應 ~100%，否則追蹤器有 bug" if same else "無腦，純對位"
        if a.track != "v1":
            note += f"；r̂={hA.tracker.rate:.2f} 重進入×{hA.tracker.jumps}"
        rows.append(("A 譜追蹤", hA, note))
    if "B1" in a.mechs:
        rows.append(("B1 context 預熱", brain_run(test, fresh_brain(a, reh), a.ant_conf),
                     f"窗 {a.window} tok={a.window / 3 * TICK_SAMPS / SR:.0f}s"
                     + (f"；{warn}" if same else "")))
    if "B2" in a.mechs:
        for lam in a.lams:
            rows.append((f"B2 logit 偏置 λ={lam:g}",
                         brain_run(test, fresh_brain(a), a.ant_conf,
                                   ScoreTracker(score, a.track,
                                                _rate0(score, test),
                                                a.rate_adapt), lam),
                         "位置對齊借 A＝選項 C 形狀" + (f"；{warn}" if same else "")))

    print(f"\n口徑同 worklog 07-28 §L 音高層：預測音 vs 測試 take 下一 tick 實際"
          f" sounding pitch，他 REST 的 tick 不計。\n")
    print(f"{'機制':<26}{'voiced 命中':>20}{'onset 命中':>20}   備註")
    print("-" * 96)
    for name, h, note in rows:
        v, o = h.row()
        print(f"{name:<24}{v:>18}{o:>18}   {note}")
    if same:
        print(f"\n⚠ {warn}：排練與測試是同一個檔，B 系機制在對自己的答案，"
              f"數字無裁決效力。公平材料＝同一首歌的兩個 take。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", required=True, help="排練 take wav（＝譜/prior 來源）")
    ap.add_argument("--test", required=True, help="測試 take wav（＝對答案的那條）")
    ap.add_argument("--mechs", default="base,A,B1,B2", help="要跑的機制（逗號分隔）")
    ap.add_argument("--lams", default="1,3,6", help="B2 的 λ 掃描值")
    ap.add_argument("--window", type=int, default=768, help="腦滑動窗 tokens，同 live_v3")
    ap.add_argument("--indep", type=float, default=0.15)
    ap.add_argument("--stab", type=int, default=2)
    ap.add_argument("--ant-conf", type=float, default=0.25, help="同 live_v3 預感信心門檻")
    ap.add_argument("--key", default="auto", help="auto＝各 take 自己第一句鎖定；整數＝固定移調")
    ap.add_argument("--track", default="v2", choices=["v1", "v2", "v3"],
                    help="A/B2 的譜追蹤器：v2（預設）＝速度先驗＋丟失重進入＋寬窗；"
                         "v1＝原始；v3＝貝氏 forward filter（多假設，重複段歧義以"
                         "多峰後驗持有）")
    ap.add_argument("--rate-adapt", action="store_true",
                    help="r̂ 改成線上自適應（窗內斜率）。實測比固定 tick 比更差，"
                         "因為指標自己的前進量正是它該修的落後")
    ap.add_argument("--shared-octave", action="store_true",
                    help="test take 沿用 rehearse take 的 auto-octave（live.py:69-77 "
                         "各 take 自行決定，同曲兩 take 可能差 12 半音而讓 A/B 全錯）")
    ap.add_argument("--max-ticks", type=int, default=None, help="只跑前 N tick（冒煙用）")
    a = ap.parse_args()
    a.mechs = set(a.mechs.split(","))
    a.lams = [float(x) for x in a.lams.split(",")]
    main(a)
