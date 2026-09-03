"""respond2.py — 應答式 v2：唱一句 → 回應播「離線品質和音＋他剛唱的那句」

Harry 2026-08-01 UX：「使用者 input 唱一句，output 是天使回應 render 好的和音
＋使用者剛剛唱的那一句」。

為什麼這個形狀活著（同日排練預渲染線判死的兩個殺手全繞開）：
  - **無對位問題**：和音是對他剛唱完的那句作的，音永遠對（排練模式死於追蹤
    命中 37.9%）。
  - **無同時性壓力**：回應本來就在他之後＝call-response 語法，不是延遲。
  - **句間空檔足夠跑離線正典渲染**（phrase_render 常駐模型，RTF 見該檔）。
  - 他的聲音在回應裡＝hero loop（聽見自己變成合唱團）的應答形。

⚠ 紅線（G「--voice-delay」07-16 判死：播他自己的延遲副本＝DAF 干擾、耳測
全滅）：**回應只在他不唱時播；他一開口就淡出讓路**。live 模式硬規則，不是
選項。差別在 voice-delay 是邊唱邊疊（DAF 前提成立），本設計是空檔應答。

live 模式的硬靜音（08-01 展場前端定案：空氣麥克風 FIFINE K669B）：**斷句成
立的那一刻起輸入就死透，直到回應播完＋殘響尾巴**。不是「偵測到回授才閃避」
而是結構上唱/播不同時活著——一次消滅迴授、把喇叭當成他在唱的誤判、以及回應
滲進下一句三個問題。代價是這段窗內麥克風聽不見他，所以**實體鍵是他在靜音期
唯一的發話管道**（`tap_listen`）：聽的時候按＝不等 gap 立刻收句、靜音期按＝
中止／作廢這次回應馬上復聽。收不到鍵就退化成純能量判定，不擋路。

句尾偵測與 file 模式共用同一套判準（`find_phrases` / `PhraseGate` 互為鏡像）：
10ms RMS 窗過 gate＝有聲、靜默 ≥gap 斷句、短於 min-phrase 的島丟掉。gate 在
live 開場**現量**（安靜段底噪 p95 與唱歌段 p75 在 log 域取中點），不沿用喉麥
素材的 0.02——參數繼承要驗證（08-01 血訓連兩次）。
腦＝BrainV3 跨句持續（記得這首歌）、indep 0＋reh_polish（08-01 §G 組成層）。
key 必須手動整數（08-01 §H 血訓：真歌 keydet 不可用、bench 口徑不可沾染）。

Run（DDSP venv）:
  python respond2.py --file take.wav out/resp2.wav --key 3      # 離線
  python respond2.py --live --key 3 --in-name K669B             # 現場
  python respond2.py --live --list-devices                      # 找裝置名
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import direct_mouth as dm  # noqa: E402
import reverb as rv  # noqa: E402
from live_v3 import EarV3  # noqa: E402
from phrase_render import PhraseRenderer  # noqa: E402
from pitch import SR  # noqa: E402
from reh_polish import polish  # noqa: E402
from tap_listen import SerialTapListener, TapListener  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
# 斷句 gate 不能用 pitch.py 的 0.005（那是「有沒有音高」的口徑）：換氣噪音
# 過得了 0.005 → 全曲只剩 2 個 ≥0.6s 空隙、斷出 34.6s 的「一句」（08-01 實測）。
# 唱歌 rms p50 0.15、換氣落在 0.01 附近 → 0.02 分得開；配 gap 0.35 得中位
# 2.4s 的樂句（他的換氣實測 0.3–0.5s，0.6s 是舊 respond 對別份素材的值）。
RMS_GATE = 0.02
# serial 開起來後等這麼久要收到一行 TAP/HB，否則當「不是那顆按鈕」退回 UDP
# （韌體 HB 每 2s 一封，所以真的按鈕通常 <2s 就成立）。
TAP_WAIT = 5.0
VOICES = {"upper": (f"{DDSP}/exp/combsub-girl/model_30000.pt", 20260731),
          "lower": (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", 20260805)}
# 08-05 §F Harry 終判過的配方：lower＝Bass-1（M4Singer 真男低音）＋兩聲部
# 過 NSF-HiFiGAN enhancer（autotune 案結論：病在低音域 CombSub 裸輸出的
# 死平諧波，enhancer 治好；f0/骨架無罪）。舊 lower（他自己的模型）：
#   "lower": (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt", 20260732)
# 08-05 §G S 聲部（--sop 守衛；0＝雙天使終判配方逐 byte 不動）：
# M4Singer Soprano-3、val 谷底 10k。線＝他的 mic 音高 +12（旋律在頂＝
# 合唱慣例；S3 p10=67、他中位 54+12=66＝域下緣，先聽再調）。
SOP = (f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 20260806)
# 顫音去同步（render_v3.py:339-340 原值）：兩天使同相位同速率＝機械齊振＝
# autotune 感來源之一（07-28）。--f0-mode target 用。
VIB = {"upper": (5.3, 0.0), "lower": (4.6, 0.5), "sop": (4.9, 0.75)}


def find_phrases(x, gap_s=0.6, min_s=0.5, win=441):
    """(start, end) 樣本區間。舊 respond.py:50-68 的判準（真人驗證過）：
    10ms RMS 窗過 gate＝有聲、靜默 ≥gap_s 才斷句、短於 min_s 的島丟掉。"""
    n = len(x) // win
    loud = np.array([np.sqrt(np.mean(x[i * win:(i + 1) * win] ** 2)) > RMS_GATE
                     for i in range(n)])
    gap_w, out, i = int(gap_s * SR / win), [], 0
    while i < n:
        if not loud[i]:
            i += 1
            continue
        j = i
        run = 0
        while j < n and run < gap_w:
            j += 1
            run = run + 1 if j < n and not loud[j] else 0
        end = j - run
        if (end - i) * win >= min_s * SR:
            out.append((i * win, min(len(x), end * win)))
        i = j
    return out


def phrase_notes(ear, seg):
    """一句音訊 → 該句每 tick 的 (lead, upper, lower)。腦跨句持續＝記得這首
    歌；只餵有聲段，靜默不進腦的上下文。"""
    up, lo = [], []
    n0 = len(ear.lead_hist)
    for k in range(len(seg) // TICK_SAMPS):
        for u, l in ear.tick(seg[k * TICK_SAMPS:(k + 1) * TICK_SAMPS]):
            up.append(u)
            lo.append(l)
    return ear.lead_hist[n0:], up, lo


class PhraseGate:
    """find_phrases 的串流版：push(chunk) → 收滿一句就回傳該句音訊（去掉尾端
    靜默），否則 None。判準逐項與 find_phrases 相同，參數也走同一組 CLI，
    兩邊不可各自漂移（08-01 血訓：參數繼承要驗證）。"""

    WIN = 441                                    # 10ms @44.1k，同 find_phrases

    def __init__(self, gate, gap_s, min_s):
        self.gate = gate
        self.min_w = int(min_s * SR / self.WIN)
        self.lv = []                             # 窗 rms 歷史（顯示／開場校準用）
        self.set_gap(gap_s)
        self.reset()

    def set_gap(self, gap_s):
        """斷句所需的靜默秒數。實體鍵連上時會被拉長成安全網（見 live()）。"""
        self.gap_s = gap_s
        self.gap_w = int(gap_s * SR / self.WIN)

    def reset(self):
        self.res = np.zeros(0)                   # 不滿一窗的餘數
        self.buf = []                            # 這一句（含尾端靜默）的窗
        self.run = 0                             # 連續靜默窗數
        self.on = False
        self.taken = 0                           # 已交給 Prefetch 的窗數

    def committed(self):
        """尚未交出去、且**保證會留在這一句裡**的樣本（管線化用，08-03）。

        兩個保證，缺一個管線化就會拿到不屬於這句的音訊：
          ① 尾端裁切：被裁掉的永遠是「**現在**的尾端靜默 run」（push 再多裁
             最後那個有聲窗＝run+1，force_end 裁 run）。所以裁切起點恆 ≥
             `len(buf) - run - 1`：現在就已經在這個界之前的窗，之後不管是靜
             默續長（裁切起點不動）還是他又出聲（裁切起點往後跑），都不會被
             裁掉。**不能用 gap_w 當保留量**——實體鍵連上時 gap_w 是 6.0s
             (--gap-tap)＝短於 6.5s 的句子一個樣本都交不出去、管線化實質關閉
             （08-12 審查 A5/A4/A2；自動斷句的 6.0s 安全網不受影響）。
          ② 這一句一定會被交出去：長度過 min_w 後，push 與 force_end 兩條路
             的 end 都 ≥ min_w＝不會走「太短當誤按、整段作廢」那條。"""
        if not self.on:
            return np.zeros(0)
        end = len(self.buf) - self.run - 1
        if end < max(1, self.min_w) or end <= self.taken:
            return np.zeros(0)
        out = np.concatenate(self.buf[self.taken:end])
        self.taken = end
        return out

    def force_end(self):
        """實體鍵：不等 gap，現在就收句（尾端靜默丟掉）。太短＝當誤按，
        手上這句作廢繼續聽（按鈕不擋路原則：按錯不會讓系統卡死）。"""
        if not self.on:
            return None
        end = len(self.buf) - self.run
        seg = (np.concatenate(self.buf[:end])
               if end >= max(1, self.min_w) else None)
        self.on, self.buf, self.run, self.taken = False, [], 0, 0
        return seg

    def push(self, x):
        x = np.concatenate([self.res, x])
        n = len(x) // self.WIN
        self.res = x[n * self.WIN:]
        for i in range(n):
            w = x[i * self.WIN:(i + 1) * self.WIN]
            self.lv.append(float(np.sqrt(np.mean(w ** 2))))
            loud = self.lv[-1] > self.gate
            if not self.on:
                if loud:
                    self.on, self.buf, self.run = True, [w], 0
                continue
            self.buf.append(w)
            self.run = 0 if loud else self.run + 1
            if self.run < self.gap_w:
                continue
            # find_phrases 的 end＝j-run 會少算最後一個有聲窗（10ms）。這裡
            # 刻意複製那個 -1：兩模式必須切出同一句界，不然 file 的耳測結論
            # 對 live 不成立。要改就兩邊一起改。
            end = len(self.buf) - self.run - 1
            seg = (np.concatenate(self.buf[:end])
                   if end >= max(1, self.min_w) else None)
            self.on, self.buf, self.run, self.taken = False, [], 0, 0
            if seg is not None:
                self.res = x[(i + 1) * self.WIN:]    # 未處理的窗留給下次
                return seg
        return None


class KeyTracker:
    """逐句累積 raw-f0 音級分布 → 旋轉角度（08-04，`--key auto`）。

    **為什麼現在可以自動，而 07-30／07-31 判死過**：那兩次判死的理由不是估不
    準，是**錯了就毀掉整場**（07-31 實測 conf 0.01 被硬鎖 +4 半音、整場錯調）。
    08-04 把 polish 的參照換成他真實的音高之後，角度錯掉的代價從 37.7% 掉到
    17.9%（對調 17.3%）——一錯全毀變成幾乎不痛，自動估計才划算。腦本來就能
    處理所有調（和聲平移不變、語料正規化到 C/Am＝旋轉進去再旋轉回來），缺的
    一直只是角度。

    量的是 harvest 的 raw f0（管線化本來就算了，零額外成本），**刻意不經
    VoiceToTokens**——那層會把音吸附進音階，拿它量調是循環論證（08-01 §H 血訓：
    「lead 100% 在 C 調內」正是這樣來的）。
    """

    MAJOR = (0, 2, 4, 5, 7, 9, 11)
    MIN_FRAMES = 150                  # ~1.7s 發聲才開始判

    def __init__(self):
        self.h = np.zeros(12)

    def push(self, f0m):
        v = np.asarray(f0m)
        v = v[v > 0]
        if len(v):
            np.add.at(self.h, np.round(69 + 12 * np.log2(v / 440.0)).astype(int) % 12, 1.0)

    def best(self):
        """(key_shift, 涵蓋率, 領先第二名幾 pp)；資料不足回 None。"""
        t = self.h.sum()
        if t < self.MIN_FRAMES:
            return None
        cov = np.array([sum(self.h[(r + d) % 12] for d in self.MAJOR) / t
                        for r in range(12)])
        o = np.argsort(cov)[::-1]
        root = int(o[0])
        sh = (0 - root) % 12
        return (sh - 12 if sh > 6 else sh), float(cov[root]), float(cov[o[0]] - cov[o[1]])


class Prefetch:
    """管線化（08-03）：他還在唱的時候，就把「先算＝逐 byte 相同」的部分算掉。

    08-03 拆帳（5.16s 句、單聲部）：harvest 0.59s／voicing 逐幀 0.41s／hubert
    encode 0.54s／model forward 只有 0.06s。錢在前處理，不在模型。其中：
      **可以先算**（分段算與整段算逐 byte 相同）
        ① 腦（EarV3）：本來就是逐 tick 串流，樣本與順序一樣＝結果一樣。
        ② voicing 逐幀特徵：每幀只看 ±1024 樣本（dm.voicing_frames）。
      **不可以先算**（全段非因果，切了就是另一張嘴）
        harvest（路徑搜尋跨全段）、hubert encode（transformer 全注意力）、
        _local_ref（±3s 窗）。這些留在句尾整段算——寧可多等 0.6s 也不動已經
        耳測過的聲音。同理不切模型 forward，所以沒有接縫、沒有相位跳。

    只吃 PhraseGate.committed() 交出來的樣本＝保證屬於這句（見該方法）。
    背景執行緒跑，主迴圈照常收音與斷句（不能因為在算而變鈍）。
    """

    def __init__(self, ear):
        self.ear = ear
        self.q = queue.Queue()
        self.done = threading.Event()
        self.out = None
        self.fed = 0                      # 主執行緒餵進去的樣本數（統計用）
        self._reset()
        threading.Thread(target=self._run, daemon=True).start()

    def _reset(self):
        self.x = np.zeros(0)              # 這句到目前為止的樣本
        self.n0 = len(self.ear.lead_hist)  # 腦的起點（phrase_notes 的 n0）
        self.up, self.lo = [], []
        self.n_tick = 0                   # 已餵進腦的 tick 數
        self.fr = [np.zeros(0)] * 3       # per / flat / rms
        self.n_fr = 0                     # 已算好的 voicing 幀數

    def feed(self, x):
        self.fed += len(x)
        self.q.put(("feed", x))

    def abandon(self):
        """這句不算數（按鍵作廢等）＝丟掉手上的半成品。腦已經聽過的部分留著
        ——他確實唱了那幾個 tick，跨句持續的上下文本來就該記得。"""
        self._sync(("reset", None))
        self.fed = 0

    def resync(self):
        """重新快照腦的起點（n0 等）。用在 make_response 的尾巴迴圈之後：
        tail_tick 在 finish() 之後仍會 append ear.lead_hist，n0 不跟上的話
        下一句的 lead 會以上一句的 n_tail 個尾音開頭（08-17 review #20）。"""
        self._sync(("reset", None))
        self.fed = 0

    def finish(self, seg):
        """句尾：補完剩下的部分，回傳 (notes, shared)。"""
        self._sync(("finish", seg))
        d, sh = self.out
        self._sync(("reset", None))
        self.fed = 0
        return d, sh

    def _sync(self, msg):
        self.done.clear()
        self.q.put(msg)
        self.done.wait()

    def _run(self):
        while True:
            kind, payload = self.q.get()
            try:
                if kind == "feed":
                    self._feed(payload)
                    continue
                if kind == "finish":
                    self.out = self._finish(payload)
                else:
                    self._reset()
            except Exception as e:                 # 管線化不擋路：退回整句重算
                print(f"⚠ prefetch failed ({e!r}) → recomputing this phrase",
                          flush=True)
                self._reset()
                self.out = (None, {})
            self.done.set()

    def _feed(self, x):
        self.x = np.concatenate([self.x, x])
        self._brain(len(self.x) // TICK_SAMPS)
        self._frames(self.x, dm.frames_ready(len(self.x)))

    def _brain(self, k):
        while self.n_tick < k:
            s = self.x[self.n_tick * TICK_SAMPS:(self.n_tick + 1) * TICK_SAMPS]
            for u, l in self.ear.tick(s):
                self.up.append(u)
                self.lo.append(l)
            self.n_tick += 1

    def _frames(self, x, k1):
        if k1 <= self.n_fr:
            return
        new = dm.voicing_frames(x, self.n_fr, k1)
        self.fr = [np.concatenate([a, b]) for a, b in zip(self.fr, new)]
        self.n_fr = k1

    def _finish(self, seg):
        n = len(self.x)
        if n > len(seg) or not np.array_equal(seg[:n], self.x):
            # committed() 的兩個保證擋掉了這件事；真的發生就是我寫錯了。
            print(f"⚠ prefetch prefix mismatch ({n} vs {len(seg)}) → dropping it,"
                      f" recomputing", flush=True)
            self._reset()
        self.x = seg
        self._brain(len(seg) // TICK_SAMPS)
        n_hops = len(seg) // dm.HOP + 1
        self._frames(seg, n_hops)                  # 句尾窗不足的幀留 0＝原語意
        f0m = dm.harvest_f0(seg)
        sh = {"n": len(seg), "f0m": f0m,
              "agc": dm.input_agc(seg, f0m=f0m),
              "vm": dm.voicing_mask(seg, n_hops, f0m=f0m,
                                    frames=tuple(f[:n_hops] for f in self.fr))}
        d = {"lead": self.ear.lead_hist[self.n0:],
             "upper": self.up, "lower": self.lo}
        return d, sh


def tail_carrier(seg, vm, n_tail, fade=0.45):
    """尾巴的載體音訊：他已經不出聲了，但嘴需要內容（units/f0/volume 都從音訊
    來）。拿他這句**最後一段有聲**的音訊循環當載體＝天使拉長同一個母音，這在
    合唱收尾裡本來就是標準寫法。

    只借「內容」，音高由音符線決定（f0 是注入的），所以循環的是音色與氣息，
    不是旋律。等功率交叉淡化藏接縫，後 fade 比例淡出。"""
    v = np.repeat(np.asarray(vm, dtype=float), dm.HOP)[:len(seg)] > 0.5
    idx = np.flatnonzero(v)
    if not len(idx):
        return np.zeros(n_tail)
    end = idx[-1] + 1
    src = seg[max(0, end - int(0.5 * SR)):end]        # 最後 0.5s 有聲
    if len(src) < int(0.05 * SR):
        return np.zeros(n_tail)
    xf = int(0.03 * SR)
    out, w = [], np.linspace(0, 1, xf)
    cur = src.copy()
    while sum(len(c) for c in out) < n_tail + len(src):
        if out:
            out[-1][-xf:] = out[-1][-xf:] * (1 - w) + cur[:xf] * w
            out.append(cur[xf:].copy())
        else:
            out.append(cur.copy())
    y = np.concatenate(out)[:n_tail]
    k = int(len(y) * (1 - fade))
    if k < len(y):
        y[k:] *= np.cos(np.linspace(0, np.pi / 2, len(y) - k)) ** 2
    return y


def make_response(seg, ear, rend, a, pre=None):
    """他的一句 → 要播回去的音訊（他那句 dry ＋兩天使）與渲染秒數。
    file / live 兩模式共用＝回應的配方只有一份。

    pre＝Prefetch（live 管線化）：腦的音符線與 x-only 特徵已經在他唱的時候
    算好，這裡只剩 harvest／encode／forward。沒有就照舊整句從頭算。"""
    d, sh = (None, {}) if pre is None else pre.finish(seg)
    if d is None:
        lead, up, lo = phrase_notes(ear, seg)
        d = {"lead": lead, "upper": up, "lower": lo}
    n_tail = int(a.tail_s * SR / TICK_SAMPS)
    if n_tail:
        # 尾巴：他不出聲了，天使自己走（08-04 §②）。lead 維持他最後那個音。
        last = next((p for p in reversed(d["lead"]) if p is not None), None)
        for _ in range(n_tail):
            u, l = ear.tail_tick()
            d["lead"].append(last)
            d["upper"].append(u)
            d["lower"].append(l)
        if pre is not None:
            # 08-17 review #20：tail_tick 每格都 append ear.lead_hist，而
            # finish() 的 reset 在尾巴迴圈**之前**就拍了 n0——不重拍的話從
            # 第二句起 d["lead"] 開頭多出上一句的 n_tail 個尾音、up/lo 卻是
            # 新句的 ⇒ 恰好 sop（lead 派生、最頂最顯耳的線）整句慢 tail_s
            # 秒，且只在悠長旋鈕（--tail-s）開著時發作、無任何錯誤可查。
            pre.resync()
    if "sop" in rend:
        # S 線＝他的 mic 音高 +12（lead 是腦空間：−off 換回 mic 空間，§B 血訓
        # ——跨空間的量先問座標系）。在 polish 改寫 d["lead"] 之前取。
        # shift 仍是 None＝auto-octave 還沒鎖到（校準跳過/失敗、或第一句太短：
        # VoiceToTokens 要 6 個有聲 tick）。當 0＝不移＝S 線就是 mic 音高 +12，
        # 是這裡唯一說得通的預設；舊碼在這裡 TypeError 當場死，連 session dump
        # 都一起沒了（08-12 審查 S2/S4）。
        off = (ear.v2t.shift or 0) + (ear.k_shift or 0)
        d["sop"] = [None if p is None else int(p) - off + 12
                    for p in d["lead"]]
    if a.polish:
        # polish 的參照換成**他真實唱的音高**（08-04，Harry「不是應該不定調嗎」）。
        # 原本比的是 lead_hist＝被 VoiceToTokens 吸附進 C 大調、再移調過的線
        # ＝系統的內部表徵，不是他唱的東西。那條線與天使自洽，所以「不和諧率」
        # 量起來永遠漂亮（錯調 10.1% vs 對調 10.6%＝看不出差別），而耳朵聽到的
        # 是天使對著**真實音高**撞。改參照之後（pair06 8 句，對真實音高計）：
        #     對調 key+3、舊參照        17.3%
        #     錯調 key 0、舊參照        37.7%   ← 耳朵聽到的就是這個
        #     錯調 key 0、**新參照**    17.9%   ← 幾乎追平對調
        # ＝**天使不需要被告知調，只需要不跟他實際唱的音撞**。--key 從「一錯
        # 全毀」降成值 2.4pp 的提示（音階那項；完全不管音階是 20.3%）。
        if "f0m" not in sh:
            sh["n"], sh["f0m"] = len(seg), dm.harvest_f0(seg)
        if getattr(a, "keytrack", None) is not None:
            a.keytrack.push(sh["f0m"])       # 用同一份 harvest，零額外成本
        per = TICK_SAMPS / dm.HOP
        ref = []
        for k in range(len(d["lead"])):
            w = sh["f0m"][int(k * per):int((k + 1) * per)]
            w = w[w > 0]
            ref.append(None if len(w) < 3 else
                       int(round(float(np.median(69 + 12 * np.log2(w / 440.0))))))
        d["lead"] = ref + [None] * max(0, len(d["lead"]) - len(ref))
        root = (-(ear.k_shift or 0)) % 12
        polish(d, "upper", "lower", root)
        polish(d, "lower", "upper", root)
    t0 = time.perf_counter()
    # sh＝兩嘴共用「只看 x」的那一整層（harvest/agc/voicing_mask/units/vol）。
    # 管線化時它已經帶著 prefetch 算好的 f0m/agc/vm 進來。
    if a.f0_mode == "target":     # blind_v3 配方：骨架＋固定顫音去同步
        kw = {v: dict(expr_gain=0.0, vib_semi=a.vib_semi, vib_onset_ms=250.0,
                      vib_hz=VIB[v][0], vib_phase=VIB[v][1]) for v in rend}
    else:                         # express：顫音等紋理全由表現層生
        kw = {v: dict(expr_gain=a.expr, tune_seed=20260802,
                      tune_lock=a.tune_lock) for v in rend}
    for v in rend:                # 回應混音裡乾聲已在＝stems 不再透傳（08-04 體檢⑤）
        kw[v]["uv_dry"] = a.uv_dry
    x = seg
    if a.aah:
        # 單純「啊」（08-11 Harry「即時版改單純a、不唱詞」）：天使的材料不再
        # 是他的詞句——tail_carrier 從尾巴擴到整句：他這句最後一段有聲的母音
        # 循環當全句載體＝天使用他此刻的音色唱「啊」，音高仍走腦的和聲線。
        # shared 不能沿用（那是對他原句算的 harvest/agc/vm；載體是另一段音訊
        # 而且長度相同＝守門擋不下來，必須在這裡清掉）。
        sh = {}
        vm0 = dm.voicing_mask(seg, len(seg) // dm.HOP + 1)
        n_all = len(seg) + n_tail * TICK_SAMPS
        x = tail_carrier(seg, vm0, n_all,
                         fade=(0.45 * n_tail * TICK_SAMPS / n_all)
                         if n_tail else 0.0)
    elif n_tail:
        # prefetch 先算的 f0m/agc/vm 是**原句長度**的，加了尾巴就不適用了
        # （PhraseRenderer 的守門會擋下來，那個守門是對的）。丟掉重算——腦的
        # 音符線仍然是先算好的，那才是 prefetch 省下的大頭（RTF 0.31 vs 0.08）。
        sh = {}
        vm0 = dm.voicing_mask(seg, len(seg) // dm.HOP + 1)
        x = np.concatenate([seg, tail_carrier(seg, vm0,
                                              n_tail * TICK_SAMPS)])
    ang = {v: rend[v].render(x, d[v], TICK_SAMPS, shared=sh, **kw[v])
           for v in rend}
    dt = time.perf_counter() - t0
    n = min(len(x), *(len(ang[v]) for v in rend))
    seg = np.concatenate([seg, np.zeros(n - len(seg))]) if n > len(seg) else seg
    # 這一行的加法順序不要動：--reverb 0 必須與加殘響之前逐 byte 相同，
    # 拆成 dry+angels 會換掉浮點結合順序（差在 1e-17，但那就不是同一份檔案了）。
    gu = a.angel_gain * 10 ** (a.upper_db / 20.0)
    gl = a.angel_gain * 10 ** (a.lower_db / 20.0)
    gs = a.angel_gain * 10 ** (getattr(a, "sop_db", 0.0) / 20.0)
    resp = (a.dry_gain * seg[:n] + gu * ang["upper"][:n]
            + gl * ang["lower"][:n])
    if "sop" in ang:              # 加在尾端＝--sop 0 的位元路徑不動
        resp = resp + gs * ang["sop"][:n]
    if a.reverb > 0:
        angels = gu * ang["upper"][:n] + gl * ang["lower"][:n]
        if "sop" in ang:
            angels = angels + gs * ang["sop"][:n]
        # send/return：乾聲位準完全不動，只把濕的加上去。--reverb-all＝連他那句
        # 一起送進空間（黏成一個合唱團）；預設只送天使（他在前、天使在空間裡）。
        w = rv.wet(resp if a.reverb_all else angels, a.ir)
        resp = np.concatenate([resp, np.zeros(len(w) - len(resp))]) + a.reverb * w
    parts = None
    if getattr(a, "out_map", None):
        # 分路（live --out-map）：每 stem 各自帶殘響（卷積線性＝總和與混音版
        # 同一份聲音，只是定位到自己的聲道）。順序＝[dry, upper, lower]，
        # 之後補的聲部往後加。混音 resp 照算不動＝dump/file 口徑不變。
        parts = [a.dry_gain * seg[:n], gu * ang["upper"][:n],
                 gl * ang["lower"][:n]]
        if "sop" in ang:
            parts.append(gs * ang["sop"][:n])
        if a.reverb > 0:
            parts = [q if (i == 0 and not a.reverb_all) else
                     np.concatenate([q, np.zeros(len(w) - len(q))])
                     + a.reverb * rv.wet(q, a.ir)
                     for i, q in enumerate(parts)]
        L = max(len(q) for q in parts)
        parts = [np.concatenate([q, np.zeros(L - len(q))]) for q in parts]
    return resp, dt, parts


def octave_from_calib(x):
    """校準唱段 → auto-octave 的位移（08-03）。

    原本的估計器只吃**最初 6 個有聲 tick（1.1 秒）**，而且那 1.1 秒是他開唱的
    第一口——換氣、起音垃圾、低音預備音都在那裡。file 模式鎖錯可以重跑，
    **live 上台鎖錯就是整場都錯**（08-02 probe 同素材鎖出 +0 與 +12 兩種）。

    校準的第二段本來就是「用演出音量唱 5 秒」＝27 個 tick 的**刻意演唱**，而且
    發生在任何回應之前。同一個估計器、多 4.5 倍的資料、更乾淨的取樣時機。
    這不是換演算法（式子逐字相同：中位數瞄準 65、只走八度），只是換餵它的東西。

    位移量化成整八度，所以估計誤差要大到跨過 6 個半音才會換一格＝對中位數的
    小差異免疫（用 harvest 而不是走腦的 tick，就是靠這個量化吸收掉差別）。"""
    if x is None or len(x) < SR:
        return None
    f0 = dm.harvest_f0(np.ascontiguousarray(x))
    v = f0[f0 > 0]
    if len(v) < 40:                       # 不到 ~0.5s 的有聲＝沒唱，別亂鎖
        return None
    med = float(np.median(69 + 12 * np.log2(v / 440.0)))
    return int(12 * round((65 - med) / 12)), med


def build_ear_and_mouths(a):
    ear = EarV3(indep=a.indep, stab=a.stab, key=a.key)
    if a.octave is not None:
        # auto-octave（live.py:69-77）用最初 6 個有聲 tick 的中位數決定，之後
        # 整個行程不再變：file 鎖錯可重跑，live 上台鎖錯＝天使整晚差一個八度。
        # 它也讓 A/B 不乾淨——key 一動，中位數跟著動、跨過取整邊界就換一格
        # （08-02 實測 --key -2 鎖 +12、--key 5 鎖 +0）。釘住＝單一變因。
        ear.v2t.shift = a.octave
        print(f"[auto-octave] pinned to {a.octave:+d} semitones (--octave)")
    rend = {v: PhraseRenderer(DDSP, m, expr_seed=sd, enhance=True)
            for v, (m, sd) in VOICES.items()}
    if getattr(a, "sop", 0):
        rend["sop"] = PhraseRenderer(DDSP, SOP[0], expr_seed=SOP[1],
                                     enhance=True)
    for r in rend.values():                       # MPS 冷啟預熱（07-29 §G）
        r.render(np.zeros(SR), [60], TICK_SAMPS)
    return ear, rend


def calibrate(drain, a, beep=None):
    """開場兩段量測 → 斷句門檻。0.02 是喉麥素材量出來的（respond2 註解），
    換空氣麥克風、換場地、換增益就不成立——**參數繼承要驗證**（08-01 血訓
    連兩次）。所以不猜，開場現量：底噪 p95 與唱歌 p75 在 log 域取中點。

    順帶回答「輸入 SNR 撐不撐得住」（08-01 待錄那格）：分離度就印在這裡。

    beep＝有聲提示（08-02 §I：第一次真人實測，他把行程丟到背景就**看不到
    「現在該唱了」**，第二段量測期間沒人唱、守門正確地退回預設。展場站著唱
    同樣看不到 console → 提示必須進到喇叭裡）。暗號：
      低音 ×1＝安靜段開始／高音 ×2＝該唱了／上行 ×3＝校準完成／長低音＝失敗。
    """
    raw = {}

    def measure(secs, label, cue=None, keep=None):
        print(label, flush=True)
        if beep is not None and cue is not None:
            beep(*cue)
            drain()                            # 提示音自己進了麥克風＝丟掉
        lv, res, t0, buf = [], np.zeros(0), time.perf_counter(), []
        while time.perf_counter() - t0 < secs:
            x = np.concatenate([res, drain()])
            if keep is not None:
                # 只留這輪新收到的：res 是上一輪的餘數，上一輪已經 append 過
                # ——連 res 一起 append 會把 0–10ms 的碎塊重複塞進去（實測膨脹
                # +55%，而 octave/key 都是對這份音訊 harvest 的）。08-12 審查 A1。
                buf.append(x[len(res):])
            n = len(x) // PhraseGate.WIN
            res = x[n * PhraseGate.WIN:]
            lv += [float(np.sqrt(np.mean(x[i * PhraseGate.WIN:
                                           (i + 1) * PhraseGate.WIN] ** 2)))
                   for i in range(n)]
            time.sleep(0.01)
        if keep is not None:
            raw[keep] = np.concatenate(buf) if buf else np.zeros(0)
        return np.array(lv) if len(lv) else np.zeros(1)

    drain()                                    # 丟掉開場的雜訊
    q = measure(a.calib_quiet, f"1) Calibration: stay silent for "
                               f"{a.calib_quiet:.0f} s (measuring the noise "
                               f"floor)… [beep x1 = quiet starts]",
                cue=(440.0, 200, 1))
    s = measure(a.calib_sing, f"2) Calibration: sing for "
                              f"{a.calib_sing:.0f} s at normal performance "
                              f"volume (measuring your level)… [beep x2 = sing]",
                cue=(880.0, 160, 2), keep="sing")
    # 底噪地板：數位全零的輸入（介面靜音／線沒插）會讓 gate 算成 0＝什麼都
    # 算有聲、永遠斷不了句。真麥克風不會是 0，但沒插線會。地板本身要**遠低於
    # 任何真訊號**：1e-4（−80 dBFS）會在低位準下把好好的一場判成失敗（08-12
    # 審查 S2），1e-6 一樣擋得住全零輸入。
    n95 = max(float(np.percentile(q, 95)), 1e-6)
    s75 = float(np.percentile(s, 75))
    if s75 <= n95:                             # 沒唱／麥克風沒開
        # 退路不能用絕對值 0.02：真的低位準時它比歌聲還高 30–50 dB＝gate.on
        # 永遠不成立＝連實體鍵也救不回來（force_end 在 not on 時回 None）。
        # 改由**實測底噪**推導：n95×3（+9.5 dB）＝至少還在同一個量級。
        g = float(n95 * 3.0) * a.gate_boost
        print(f"⚠ calibration failed: singing level {s75:.4f} <= noise floor "
              f"{n95:.4f} (mic not picking anything up?) → using the "
              f"noise-derived gate {g:.4f} (n95x3). Splitting will be "
              f"touchy; rerun calibration or set --gate before the show",
              flush=True)
        if beep is not None:
            beep(220.0, 700, 1)                # 失敗＝一聲長低音，跟成功分得開
        return g, raw.get("sing")
    g = float(np.sqrt(n95 * s75)) * a.gate_boost   # log 域中點＝兩邊等距；
    # 08-19 Harry「音量門檻都拉更高」→ --gate-boost（預設 1.0＝原行為）
    db = 20 * np.log10(s75 / n95)
    s10 = float(np.percentile(s, 10))          # 唱段裡的低窗＝換氣與字間空隙
    # gate 印 %.6g 不是 %.4f（08-17 review #16）：respond_shell 重用的就是
    # 這行刮下來的字串——安靜鏈路的 gate 可能是 1e-05 級，%.4f 會印成
    # 0.0000，重生引擎拿到 --gate 0.0＝lv>0 恆真＝樂句永遠不結束。
    print(f"   noise floor p95 {n95:.4f} (p50 {np.median(q):.4f}) | "
          f"singing p75 {s75:.4f} (p50 {np.median(s):.4f}) | "
          f"singing p10 {s10:.4f} (breaths/gaps) | "
          f"separation {db:.1f} dB"
          f"\n   → gate {g:.6g} "
          f"({20 * np.log10(g / n95):.1f} dB above the noise floor, "
          f"{20 * np.log10(s75 / g):.1f} dB below the singing level"
          f"{'' if a.gate_boost == 1.0 else f'; boost x{a.gate_boost:g}'})",
          flush=True)
    # 真正要擋掉的是換氣，不是房間底噪（0.02 那個手調值就是對著換氣 ~0.01
    # 調的）。安靜段量不到換氣，所以往高的一邊錯：gate 偏高＝句子被切碎
    # （回應照樣成立），gate 偏低＝斷不出句、整首變成一句（08-01 實測 34.6s
    # 的「一句」）。後者才是致命的。
    if g < s10:
        print(f"   ⚠ gate {g:.4f} is below the quiet windows inside the singing "
              f"{s10:.4f} = breaths clear the threshold and phrases may not "
              f"split. If phrases run long, raise --gate by hand.",
              flush=True)
    if db < 12:
        print("   ⚠ separation <12 dB: breaths and the noise floor crowd the "
              "threshold = splitting will be unreliable. Fix mic distance / "
              "gain / room noise first.", flush=True)
    if beep is not None:
        beep(660.0, 110, 3)                    # 上行三短音＝校準完成，可以開唱
    return g, raw.get("sing")


def live(a):
    """現場：聽 → 斷句 → 硬靜音 → 渲染 → 播回應 → 解除靜音。"""
    import sounddevice as sd
    if a.list_devices:
        print(sd.query_devices())
        return

    ear, rend = build_ear_and_mouths(a)
    pre = None if a.no_pipeline else Prefetch(ear)

    def udp_tap():
        """退回 UDP（app space 鍵／無線鍵，同協定）；開不了就當沒這顆鍵。"""
        t = TapListener()
        if t.err:
            print(f"⚠ [tap] could not open the UDP port ({t.err}) → treating this as "
                  f"no physical key (an orphan engine may still hold the port)",
                  flush=True)
            return None
        return t

    tap = None if a.no_tap else SerialTapListener()
    if tap is not None and tap.err:            # 沒插
        print(f"[tap] could not open serial ({tap.err}) → listening on UDP "
              f"instead (app space key)", flush=True)
        tap = udp_tap()
    elif tap is not None and not tap.wait_proto(TAP_WAIT):
        # 開得起來 ≠ 是那顆按鈕：glob 抓到的可能是印表機／別的板子／REPL 裡的
        # Pico。舊碼只看「開失敗」⇒ 那些情況下 UDP 永遠不開＝space 靜默失效
        # （08-12 審查 A3/A1）。等不到一行 TAP/HB 就關掉退回 UDP。
        print(f"⚠ [tap] no TAP/HB from serial {tap.dev} within {TAP_WAIT:g}s"
              " = not the button → closing it, listening on UDP "
              "(app space key)", flush=True)
        tap.close()
        tap = udp_tap()
    cap, lock = [], threading.Lock()
    cap_n = 0
    st = {"muted": False, "play": None, "ppos": 0, "tail": 0, "n": 0,
          "iov": 0, "oun": 0, "phr": 0, "wait": [], "fade": False,
          "abort": False}
    sess = np.zeros(int(a.max_min * 60 * SR), dtype=np.float32)
    # --out-map 'D,U,L'＝分路（F7：同一台多聲道裝置）。None＝舊行為，
    # 下面所有分路碼都掛在 omap 上＝預設路徑逐行不動。
    omap = None
    if a.out_map:
        omap = [int(x) for x in str(a.out_map).split(",")]
        if len(omap) not in (3, 4) or min(omap) < 0:
            raise SystemExit(f"--out-map 要 3–4 個非負聲道"
                             f"（dry,upper,lower[,sop]；缺第 4 個＝sop 跟"
                             f" upper 同道），拿到 {a.out_map!r}")
    n_out = 2 if omap is None else max(max(omap) + 1, 2)

    def cb(indata, outdata, frames, t, status):
        if status:                       # PortAudio 旗標＝輸入鏈波波的直接證據
            if status.input_overflow:
                st["iov"] += 1
            if status.output_underflow:
                st["oun"] += 1
        outdata[:] = 0
        with lock:
            muted = st["muted"]
            if not muted:                # 硬靜音＝這段音訊根本不進斷句器
                cap.append(indata[:, 0].astype(np.float64))
            p = st["play"]
            if p is not None:
                k = st["ppos"]
                s = p[k:k + frames]
                if p.ndim == 2:          # 分路回應：每 stem 已在自己的聲道
                    outdata[:len(s), :s.shape[1]] = s
                elif omap is not None:   # 分路模式下的單聲道訊號（嗶）＝全聲道
                    outdata[:len(s), :] = s[:, None]
                else:
                    outdata[:len(s), 0] = s
                st["ppos"] = k + len(s)
                if st["fade"]:               # 實體鍵中止：淡出一塊再收，免爆音
                    outdata[:, 0] *= np.linspace(1, 0, frames)
                    if omap is not None:     # 分路：其餘聲道一起淡
                        outdata[:, 1:] *= np.linspace(1, 0, frames)[:, None]
                    st["play"], st["fade"] = None, False
                    st["tail"] = int(0.05 * SR)   # 他要唱了，尾巴只留擋殘響
                elif st["ppos"] >= len(p):   # 播完＝再靜音一段（殘響／喇叭衰減）
                    # max(1,…)：--mute-tail 0 會讓 tail 恆為 0＝下面那條
                    # `elif st["tail"] > 0` 永遠不成立＝muted 再也解不開、
                    # 第一句之後整場死鎖（08-12 審查 S5）。1 個樣本＝下一塊
                    # callback 就解除。
                    st["play"] = None
                    st["tail"] = max(1, int(a.mute_tail * SR))
            elif st["tail"] > 0:
                st["tail"] = max(0, st["tail"] - frames)
                if st["tail"] == 0:
                    st["muted"] = False
            if omap is None and outdata.shape[1] > 1:
                outdata[:, 1] = outdata[:, 0]
            n = st["n"]                  # session 落檔：靜音期他的輸入寫 0
            if n + frames <= len(sess):
                mix = outdata[:, 0] if omap is None else outdata.sum(axis=1)
                sess[n:n + frames] = (indata[:, 0] * (0.0 if muted else 1.0)
                                      + mix)
                st["n"] = n + frames
            else:
                st["sess_full"] = True   # 滿了就停寫；主迴圈印一次警告

    with sd.Stream(samplerate=SR, blocksize=1024, channels=(1, n_out),
                   device=(a.in_name, a.out_name), callback=cb,
                   latency=a.io_latency):
        def drain():
            with lock:
                b, cap[:] = list(cap), []
            return np.concatenate(b) if b else np.zeros(0)

        def beep(hz, ms, n, gap_s=0.10):
            """從喇叭出去的提示音（走既有的 play 路徑，不另開輸出）。
            08-02 §I：唯一一次真人實測敗在「提示只有文字」——他看不到 console，
            第二段校準期間沒人唱。展場站著唱同樣看不到。"""
            t = np.arange(int(ms / 1000.0 * SR)) / SR
            one = 0.25 * np.sin(2 * np.pi * hz * t) * np.hanning(len(t))
            sig = np.tile(np.concatenate([one, np.zeros(int(gap_s * SR))]), n)
            with lock:
                st["play"], st["ppos"] = sig * a.gain, 0
            time.sleep(len(sig) / SR + a.mute_tail + 0.2)   # 等它播完＋殘響

        if a.gate is not None and a.gate <= 1e-6:
            # 08-17 review #16：gate 0（或掉到數位靜音地板）＝所有窗都算有聲
            # ＝樂句永遠不結束、只剩實體鍵能收句。這種值只會來自壞掉的重用
            # 鏈（安靜校準印 0.0000 被刮回來）——拒收、當場重校。
            print(f"⚠ --gate {a.gate:g} is at/below the digital-silence floor "
                  f"= phrases would never end → ignoring it, recalibrating",
                  flush=True)
            a.gate = None
        if a.gate is not None:
            g, sing = a.gate, None
        else:
            g, sing = calibrate(drain, a, beep)
        if a.keytrack is not None and sing is not None and len(sing) > SR:
            # 校準的唱段本來就是他用演出音量唱的 5 秒，且在第一句之前——拿它
            # 種下調的估計，第一句就不會在 0 上踩空（同 auto-octave 的理由）。
            a.keytrack.push(dm.harvest_f0(np.ascontiguousarray(sing)))
            r = a.keytrack.best()
            if r is not None:
                ear.k_shift = r[0]
                print(f"[key] calibration segment seeded --key {r[0]:+d}"
                      f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                      flush=True)
        elif a.keytrack is not None and a.key_seed is not None:
            # 08-17 review #16：帶 --gate 重生（跳過校準）時，shell 把上一顆
            # 引擎鎖到的調用 --key-seed 種回來——否則 auto 從 0 起跑，第一句
            # 就在錯的調上作和聲。之後仍逐句 rotation 修正。
            ear.k_shift = a.key_seed
            print(f"[key] seeded --key {a.key_seed:+d} from the previous "
                  f"engine (--key-seed)", flush=True)
        if a.octave is None:              # 顯式 --octave 永遠優先
            r = octave_from_calib(sing)
            if r is not None:
                ear.v2t.shift, med = r
                print(f"[auto-octave] locked {r[0]:+d} semitones from the calibration "
                      f"segment (median MIDI {med:.1f}, {len(sing)/SR:.1f}s)",
                      flush=True)
            else:
                # 08-17 review #16：這行原本只說 too short——帶 --gate 跳過
                # 校準時 sing 是 None，也走到這裡，訊息要誠實涵蓋那種情況
                # （app 的切換路徑會補 --octave，這行只在裸 CLI 帶 --gate
                # 時出現）。
                print("[auto-octave] no usable calibration segment (too "
                      "short, or skipped because --gate was given) → falling "
                      "back to \"first 6 voiced ticks\" (a wrong lock ruins "
                      "the whole set; --octave pins it)", flush=True)
        drain()          # 收官提示音自己也進了麥克風＝別被當成他的第一句
        gate = PhraseGate(g, a.gap, a.min_phrase)
        # 實體鍵連上＝句子由他決定，能量斷句退居安全網（gap 拉長，只防「按鈕
        # 沒按到」卡死）。否則 0.35s 的 gap 會在他句中換氣時搶先切句，按鈕根本
        # 沒機會等他——「唱多久、中間要不要停」才真的回到他手上（08-02 Harry）。
        def sync_gap():
            want = a.gap_tap if (tap is not None and tap.alive) else a.gap
            if want != gate.gap_s:
                gate.set_gap(want)
                print(f"[tap] {'connected' if want == a.gap_tap else 'disconnected'}"
                      f" → phrase gap {want}s"
                      f" ({'phrases end on the button, energy split is only a safety net' if want == a.gap_tap else 'back to pure energy detection'})",
                      flush=True)

        sync_gap()
        t_txt = "off" if tap is None else (
            "connected" if tap.alive else "not connected")
        # 顯示誠實：印生效中的 gap（實體鍵連上時是 --gap-tap，不是 --gap）
        how = (f"button ends phrases (auto after {gate.gap_s}s = safety net)"
               if tap is not None and tap.alive
               else f"sing, then {gate.gap_s}s of silence")
        print(f"respond2 live. key {a.key} st, gate {g:.4f} gap {gate.gap_s}s, "
              f"response = dry {a.dry_gain} + angel {a.angel_gain}"
              f" (upper {a.upper_db:+.1f} lower {a.lower_db:+.1f} dB), "
              f"gain {a.gain}"
              f" | key {t_txt} | {how} → muted render → response -- Ctrl-C stops.",
              flush=True)
        t_say = time.perf_counter()
        try:
            while True:
                with lock:
                    muted, blocks, cap[:] = st["muted"], list(cap), []
                if tap is not None and tap.dead:   # serial 讀取執行緒死了
                    print("⚠ [tap] serial read interrupted (cable unplugged?) → "
                          "listening on UDP (app space key)", flush=True)
                    tap.close()
                    tap = udp_tap()
                tapped = tap.take() if tap is not None else False
                if tap is not None and not gate.on:   # 句中不換 gap
                    sync_gap()
                if muted:
                    if tapped:               # 靜音期他唯一的發話管道＝中止回應
                        with lock:
                            hot = st["play"] is not None
                            if hot:
                                st["fade"] = True
                        # play 是 None＝回應已經播完、正在 mute_tail 尾巴裡
                        # （渲染期間主迴圈根本不在這裡輪詢）＝沒有東西可中止。
                        # 舊碼在這裡 latch st["abort"]，而那個旗標活到**下一句**
                        # 渲染完才被消費 ⇒ 天使唱完他按鍵表示「換我了」，代價是
                        # 下一句的回應被整個丟掉（08-12 三份審查都抓到）。忽略。
                        print(f"  (key: {'response aborted' if hot else 'response already finished, ignored'})",
                              flush=True)
                    time.sleep(0.01)
                    continue
                if not blocks and not tapped:
                    if time.perf_counter() - t_say > 2:
                        t_say = time.perf_counter()
                        lv = np.array(gate.lv[-400:]) if gate.lv else np.zeros(1)
                        print(f"listening {cap_n / SR:6.1f}s | window rms p50 "
                              f"{np.median(lv):.3f} p95 {np.percentile(lv, 95):.3f}"
                              f" (gate {g:.4f}) | phrase {st['phr']}"
                              f"{'' if tap is None else ' | key ' + ('ON' if tap.alive else 'off')}"
                              f" | io {st['iov']}/{st['oun']}", flush=True)
                        if st.get("sess_full") and not st.get("sess_full_said"):
                            st["sess_full_said"] = True
                            print(f"⚠ dump buffer full (--max-min {a.max_min:g})"
                                  " = later audio no longer enters the session "
                                  "dump (responses carry on)", flush=True)
                    time.sleep(0.01)
                    continue
                if len(blocks):
                    chunk = np.concatenate(blocks)
                    cap_n += len(chunk)
                    seg = gate.push(chunk)
                else:
                    seg = None
                by_tap = False
                if tapped and seg is None:   # 按鍵＝不等 gap，現在就收句
                    seg, by_tap = gate.force_end(), True
                    print(f"  (key: {'phrase ended' if seg is not None else 'not a full phrase yet, ignored'})",
                          flush=True)
                    if seg is None and pre is not None and pre.fed:
                        pre.abandon()
                if seg is None:
                    if pre is not None:      # 他還在唱＝把已確定的部分先算掉
                        c = gate.committed()
                        if len(c):
                            pre.feed(c)
                    continue
                with lock:               # 斷句成立＝立刻死透，渲染期也不聽
                    # abort 一律在這裡歸零：它只該對「這一次渲染」有效，殘留
                    # 下來就是上面那個吃掉整句回應的閂鎖。
                    st["muted"], st["abort"] = True, False
                print("  captured → rendering…", flush=True)   # UI 靠這行知道硬靜音開始
                t_det = time.perf_counter()
                pre_s = 0.0 if pre is None else pre.fed / SR
                try:
                    resp, dt, parts = make_response(seg, ear, rend, a, pre=pre)
                except Exception as e:   # noqa: BLE001 — 壞一句不賠整場
                    # 舊碼只接 KeyboardInterrupt ⇒ 任何一句炸掉都會穿出去、
                    # 連 session dump 都跳過（08-12 審查 A6/S2/S4）。
                    print(f"  ⚠ this phrase failed to render ({e!r}) → dropping it, "
                          f"back to listening", flush=True)
                    if pre is not None:
                        pre.abandon()
                    gate.reset()
                    with lock:
                        st["muted"], st["abort"] = False, False
                    continue
                busy = time.perf_counter() - t_det     # 腦＋polish＋渲染
                if omap is None:
                    peak = float(np.abs(resp).max()) if len(resp) else 0.0
                    out = np.clip(np.concatenate(
                        [np.zeros(int(a.pause * SR)), resp]) * a.gain, -1, 1)
                else:                    # 分路：每 stem 進自己的聲道，clip 逐道
                    pz = int(a.pause * SR)
                    out = np.zeros((pz + len(resp), n_out))
                    om = omap + [omap[1]] * (len(parts) - len(omap))
                    for q, ch in zip(parts, om):   # 第 4 stem（sop）預設跟 upper
                        out[pz:pz + len(q), ch] += q * a.gain
                    peak = (float(np.abs(out).max()) / a.gain) if a.gain else 0.0
                    np.clip(out, -1, 1, out=out)
                st["phr"] += 1
                # 「他停唱到回應開播」要從他最後一個有聲樣本起算：先付 gap 秒
                # 才判得出斷句，再付腦＋渲染，最後才是 pause。只報 pause+渲染
                # 會少算 gap 與腦（08-02 probe：靜音窗實測比它長 0.2–0.6s）。
                # 顯示誠實：按鍵收句根本沒付 gap，能量收句付的是生效中的
                # gate.gap_s（實體鍵連上時＝--gap-tap）——不是固定印 --gap。
                g_paid = 0.0 if by_tap else gate.gap_s
                st["wait"].append(g_paid + busy + a.pause)
                print(f"  phrase {st['phr']} {len(seg) / SR:4.1f}s (pre "
                      f"{pre_s:4.1f}s) → brain+render "
                      f"{busy:4.2f}s (mouth {dt:4.2f}s, RTF "
                      f"{dt / (len(seg) / SR):.2f}) → response {len(resp) / SR:4.1f}s "
                      f"peak {peak * a.gain:.2f} | **wait "
                      f"{g_paid + busy + a.pause:4.2f}s** (gap {g_paid:.2f}"
                      f"{' = key' if by_tap else ''}) | mute window "
                      f"{busy + a.pause + len(resp) / SR + a.mute_tail:4.2f}s",
                      flush=True)
                if peak * a.gain > 1.0:
                    # 削峰不能只靠 np.clip 默默吃掉。回應的峰值直接跟他的輸入
                    # 位準綁在一起（天使的位準是還原成他的），所以沒有一個對所有
                    # 麥克風都對的預設增益——只能量到就喊。08-03 實測：喉麥那份
                    # 素材配 dry 1.1 峰值 p50 1.35／max 2.07，K669B 會低一個量級。
                    print(f"  ⚠ clipping: peak {peak * a.gain:.2f} > 1.0, this phrase got cut"
                          f" (suggest --gain {0.89 / peak:.2f})", flush=True)
                if a.keytrack is not None:
                    r = a.keytrack.best()
                    if r is not None and r[0] != (ear.k_shift or 0) and r[2] >= a.key_margin:
                        # 行首不縮排（08-17）：respond_shell 的 key pattern 是
                        # ^\[key\]，縮排的 rotation 行 UI 抓不到＝KEY tile 凍
                        # 在種子值、切換時的 --key-seed 也帶不到最新值。
                        print(f"[key] rotation {ear.k_shift:+d} → {r[0]:+d}"
                              f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                              flush=True)
                        ear.k_shift = r[0]
                gate.reset()             # 靜音期的輸入不存在＝斷句器狀態一併歸零
                with lock:
                    if st["abort"]:      # 渲染中按了鍵＝他要接著唱，這句作廢
                        st["abort"], st["muted"] = False, False
                        print("  (key: response discarded, back to listening)", flush=True)
                    elif len(resp):
                        st["play"], st["ppos"] = out.astype(np.float64), 0
                    else:                # 這句腦全判休止＝沒東西播，直接回去聽
                        st["muted"] = False
        except KeyboardInterrupt:
            pass
        finally:
            # dump 與統計在關流「之前」落檔——08-04 白天：Ctrl-C 後 PortAudio
            # FinishStoppingStream 在 CoreAudio 裡掛死一次，dump 差點跟行程一起走
            # （靠 lldb 注入救回）。stream 還開著時 sf.write 安全：n 先快照，
            # callback 只往 n 之後寫。**finally**：Ctrl-C 以外的例外也要落檔，
            # 不然一個沒接到的錯就賠掉整場錄音（08-12 審查 A6/S2）。
            n = st["n"]
            if n > SR:
                stamp = time.strftime("%y%m%d_%H%M%S")
                p = HERE / "out" / f"resp2_live_{stamp}.wav"
                p.parent.mkdir(exist_ok=True)
                sf.write(str(p), sess[:n], SR, subtype="PCM_16")
                print(f"\nsession dump: {p} ({n / SR:.1f}s, what was heard from him + "
                      f"what was played, same track; input is 0 through the "
                      f"mute = proof the mute is hard)", flush=True)
            w = np.array(st["wait"]) if st["wait"] else np.zeros(1)
            print(f"total {st['phr']} phrases | wait p50 {np.median(w):.2f}s "
                  f"p95 {np.percentile(w, 95):.2f}s"
                  f" (pipelining {'off' if a.no_pipeline else 'on'}; gap as "
                  f"actually paid)"
                  f" | io overflow/underflow {st['iov']}/{st['oun']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--live", action="store_true",
                    help="現場模式：麥克風進、喇叭出，播回應期間輸入硬靜音")
    ap.add_argument("--in-name", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-map", default=None,
                    help="live 分路：'D,U,L'＝dry/上天使/下天使的 0-based 輸出"
                         "聲道（F7：同一台多聲道裝置；之後補的聲部往後加）。"
                         "無＝混音出雙聲道（舊行為）。file 模式不理它")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="回應輸出音量（live）")
    ap.add_argument("--mute-tail", type=float, default=0.25,
                    help="回應播完後續靜音秒數（吃掉殘響與喇叭衰減，避免自己"
                         "的尾巴被當成他開口）")
    ap.add_argument("--io-latency", default="high",
                    help="sd.Stream latency：high＝驅動大緩衝（抗 GIL 尖峰）")
    ap.add_argument("--max-min", type=float, default=10.0,
                    help="session 落檔上限（分鐘）")
    ap.add_argument("--key-margin", type=float, default=0.015,
                    help="--key auto：領先第二名這麼多才換角度（避免逐句抖動）")
    ap.add_argument("--key-seed", type=int, default=None,
                    help="--key auto 的初始調移（半音）。respond_shell 在帶 "
                         "--gate 重生（跳過校準）時把上一顆引擎鎖到的調種回"
                         "來；之後仍逐句 rotation 修正。無校準又無種子＝從 0 "
                         "起跑（舊行為）")
    ap.add_argument("--key",
                    help="整數移調，或 **auto**（逐句用 raw-f0 音級分布重估）。必填（08-01 血訓：真歌不可 auto，也不可沿用 "
                         "bench 的 0 口徑——先用 raw-f0 音級分布定調）")
    ap.add_argument("--gap", type=float, default=0.35,
                    help="斷句靜默秒數（他換氣實測 0.3–0.5s）；實體鍵沒連上"
                         "時用這個")
    ap.add_argument("--gap-tap", type=float, default=6.0,
                    help="實體鍵連線時的斷句 gap（安全網）：句子由按鈕決定，"
                         "樂句間休止不會被搶切；只防按鈕沒按到卡死"
                         "（6s＝08-05 Harry 定，涵蓋他刻意的句間休止）")
    ap.add_argument("--gate", type=float, default=None,
                    help="斷句有聲門檻 rms（唱 p50 ~0.15、換氣 ~0.01）。live "
                         f"不填＝開場自動校準；file 不填＝用 {RMS_GATE}")
    ap.add_argument("--gate-boost", type=float, default=1.0,
                    help="校準算出的門檻再乘這個倍數（>1＝更難觸發）。預設 "
                         "1.0＝08-19 前的行為，逐位元不變。2.0＝+6 dB。"
                         "偏高的失敗是句子被切碎（回應照樣成立），偏低的失敗"
                         "是斷不出句、整首變一句——所以往高的一邊錯比較安全")
    ap.add_argument("--calib-quiet", type=float, default=3.0,
                    help="校準第一段：量底噪的秒數（live）")
    ap.add_argument("--calib-sing", type=float, default=5.0,
                    help="校準第二段：量唱歌位準的秒數（live）")
    ap.add_argument("--no-tap", action="store_true",
                    help="不收實體鍵（tap_listen USB serial，08-04 展場路線）")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="關掉管線化（live）：句尾才從頭算腦與 voicing 特徵。"
                         "管線化只先算「分段算＝整段算逐 byte 相同」的部分，"
                         "聲音不變；這個旗標是出事時的退路")
    ap.add_argument("--min-phrase", type=float, default=0.5)
    ap.add_argument("--pause", type=float, default=0.35,
                    help="他唱完到回應開播的間隔（呼吸感）")
    ap.add_argument("--dry-gain", type=float, default=0.944,
                    help="回應裡他那句原聲的比例。**08-03 深夜實戴手調拍板**"
                         "（『先以目前數據為主』）：dry 0.944＋天使 0.75（上聲部"
                         "再 +1 dB）＝兩天使合計領先他約 +0.7 dB。⚠ 之前離線 A/B"
                         "選的 1.1 已作廢——那輪沒盲（標籤寫著『你領先 3 dB』）、"
                         "素材用了 pair06、且在桌上判＝踩 07-25 in-context 規矩")
    ap.add_argument("--f0-mode", choices=["express", "target"], default="target",
                    help="嘴的 f0 配方。**target＝預設**＝blind_v3（07-28）那條："
                         "純骨架＋固定顫音、兩天使去同步（5.3/4.6Hz 相位錯開）"
                         "——08-02 Harry 兩份都裁「還行」，express 版裁「怪」。"
                         "express＝07-31 表現層，**獨唱驗過、雙聲部會拆掉和聲**")
    ap.add_argument("--vib-semi", type=float, default=0.12,
                    help="target 模式的顫音深度（半音；0.12＝±12c，"
                         "direct_mouth.VIB_SEMI 原值）")
    ap.add_argument("--tune-lock", type=float, default=0.0,
                    help="合唱團對音（08-02）：兩天使共用表現層的慢成分（每音"
                         "偏置＋漂移＝音高中心），顫音/抖動仍各自獨立。0＝舊"
                         "行為、1＝音高中心完全一致。人味不減、只把音準對回來")
    ap.add_argument("--expr", type=float, default=1.0,
                    help="表現層強度（expressive_cents，07-31 加的）。1.0＝現行"
                         "；0＝純骨架。08-02 量到兩天使的音程誤差 SD 40.7c、"
                         "61.6%% 的時間 >20c＝和聲根本不在同一個音高上（每音偏置 "
                         "bias_sd 22c，兩把嘴 seed 不同＝各偏各的）")
    ap.add_argument("--octave", type=int, default=None,
                    help="釘住 auto-octave 的半音位移（不填＝沿用最初 6 個有聲 "
                         "tick 自動鎖）。A/B 要單一變因、或上台不想賭第一句"
                         "聽錯八度時填")
    ap.add_argument("--angel-gain", type=float, default=0.75,
                    help="兩個天使各自的比例（08-03 實戴手調；見 --dry-gain）")
    ap.add_argument("--uv-dry", type=float, default=0.0,
                    help="stems 無聲段透傳他原聲的比例。回應混音裡乾聲本來就在，"
                         "再透傳＝同一口氣疊三份（08-04 體檢：換氣/子音 +7.7 dB）"
                         "→ 預設 0＝uv 段 stems 靜音。1.0＝舊行為")
    ap.add_argument("--stab", type=int, default=1,
                    help="腦的最短音長（tick；187.5ms/格）。08-04 體檢：他換音→"
                         "天使跟進中位 562ms＝『錯開』的主因之一，stab 2 的地板"
                         "就佔 375ms → 預設降 1。2＝舊行為（08-02 耳測的那個）")
    ap.add_argument("--aah", action="store_true",
                    help="天使只唱「啊」不搬詞（08-11）：材料＝他這句尾端有聲"
                         "母音的整句循環載體；音高仍走腦的和聲線。關＝原路徑"
                         "逐 byte 不動")
    ap.add_argument("--tail-s", type=float, default=0.0,
                    help="他唱完之後天使再自己唱幾秒（08-04）。0＝關。"
                         "載體借他這句最後一段有聲音訊循環（嘴需要內容），"
                         "音高由腦繼續生成＝他不出聲時天使仍在移動")
    ap.add_argument("--upper-db", type=float, default=1.0,
                    help="上聲部（girl 嘴）相對 --angel-gain 的修正，dB"
                         "（08-03 Harry：「女聲可以 +1」）")
    ap.add_argument("--lower-db", type=float, default=1.5,
                    help="下聲部（Bass-1）相對 --angel-gain 的修正，dB"
                         "（08-05 Harry 聽 satb_4part 定 +1.5；原 0.0）")
    ap.add_argument("--sop", type=int, default=1,
                    help="S 聲部（Soprano-3 唱他的 mic 音高+12，08-05 §G）。"
                         "0＝雙天使終判配方（08-05 Aenh）逐 byte 不動")
    ap.add_argument("--sop-db", type=float, default=0.0,
                    help="S 聲部相對 --angel-gain 的修正，dB")
    ap.add_argument("--reverb", type=float, default=0.20,
                    help="殘響送出量（0＝關，整條路徑不執行＝逐 byte 等於沒有）。"
                         "08-03 耳測定 0.20；合成 IR 見 reverb.py")
    ap.add_argument("--reverb-s", type=float, default=1.8,
                    help="殘響 T60 秒。**08-03 耳測：1.8/2.2/2.5/2.8 四格「差不多」"
                         "＝長度在應答式裡不是可聞旋鈕**（IR 能量正規化，拉長只是"
                         "把同一份能量攤薄），所以取最短的——尾巴直接換他的靜音窗。"
                         "真正改變濕度的是 --reverb 那個")
    ap.add_argument("--reverb-trim", type=float, default=35.0,
                    help="尾巴掉到 −這麼多 dB 就截斷（0＝不截）。回應多長他就"
                         "多久不能出聲，所以尾巴是直接付出去的互動成本")
    ap.add_argument("--reverb-angels-only", dest="reverb_all",
                    action="store_false",
                    help="只有天使進空間、他的聲音留在前面。**預設是全部一起進**"
                         "（08-03 Harry：「有空間感，自己也進去」＝黏成一個合唱團）")
    ap.add_argument("--no-polish", dest="polish", action="store_false")
    ap.add_argument("--indep", type=float, default=0.15,
                    help="天使獨立度（08-03 實戴試過 0/0.15/0.3/1.0 未終判；"
                         "0.15＝暫用值。⚠ 應答式短句裡觸發點少——它掛在他的"
                         "音符事件上，樂句形狀才是上限，見 08-04 worklog）")
    a = ap.parse_args()
    if a.list_devices:
        return live(a)                  # live() 開頭印裝置表就返回
    if a.live == bool(a.file):
        ap.error("--file 與 --live 二選一")
    if a.key is None:
        ap.error("--key 必填（真歌先用 raw-f0 音級分布定調）")
    a.ir = (rv.make_ir(a.reverb_s, trim_db=a.reverb_trim)
            if a.reverb > 0 else None)
    a.keytrack = None
    if str(a.key).lower() == "auto":
        a.keytrack, a.key = KeyTracker(), "0"     # 從 0 起跑，第一句之後開始修
    if a.live:
        return live(a)

    x, sr = sf.read(a.file[0], dtype="float64", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    assert sr == SR, (a.file[0], sr)
    if a.gate is None:                  # file 模式沒有現場可量＝維持既有門檻
        a.gate = RMS_GATE
    globals()['RMS_GATE'] = a.gate
    ph = find_phrases(x, a.gap, a.min_phrase)
    print(f"split: {len(ph)} phrases / {len(x)/SR:.1f}s"
          f" ({', '.join(f'{(b-s)/SR:.1f}s' for s, b in ph[:8])}"
          f"{' …' if len(ph) > 8 else ''})", flush=True)

    ear, rend = build_ear_and_mouths(a)

    pieces, t_r = [], []
    for i, (s, e) in enumerate(ph):
        seg = x[s:e]
        resp, dt, _ = make_response(seg, ear, rend, a)
        t_r.append(dt)
        pieces += [seg, np.zeros(int(a.pause * SR)), resp,
                   np.zeros(int(a.pause * SR))]
        print(f"  phrase {i+1} {len(seg)/SR:4.1f}s → render {t_r[-1]:4.2f}s "
              f"(RTF {t_r[-1]/(len(seg)/SR):.2f})", flush=True)
        if a.keytrack is not None:
            r = a.keytrack.best()
            if r is not None and r[0] != (ear.k_shift or 0) and r[2] >= a.key_margin:
                print(f"    [key] rotation {ear.k_shift:+d} → {r[0]:+d}"
                      f" (covers {r[1]*100:.0f}%, leads by {r[2]*100:.1f}pp)",
                      flush=True)
                ear.k_shift = r[0]

    out = np.concatenate(pieces)
    sf.write(a.file[1], out / (np.abs(out).max() + 1e-12) * 0.9, SR,
             subtype="PCM_16")
    tr = np.array(t_r)
    print(f"\nrender: p50 {np.median(tr):.2f}s p95 {np.percentile(tr, 95):.2f}s "
          f"| his stop → response start ≈ gap {a.gap} + brain + mouth + "
          f"pause {a.pause}"
          f" (>={a.gap + np.median(tr) + a.pause:.2f}s; brain is not inside the "
          f"render seconds, see the closing line of --live)")
    print(f"wrote {a.file[1]}  ({len(out)/SR:.1f}s, [his phrase][response]x{len(ph)})")


if __name__ == "__main__":
    import signal as _sig
    # 背景 shell 起的父行程 SIGINT 是 SIG_IGN 且會遺傳＝Ctrl-C/Stop 全空包
    # （07-31 §H 血訓，live_v3.py:529 同解）。強制還原。
    _sig.signal(_sig.SIGINT, _sig.default_int_handler)
    main()
