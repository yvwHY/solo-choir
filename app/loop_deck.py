#!/usr/bin/env python3
"""loop_deck.py — Solo Choir 現場疊層底床（插件；主 app 一行不改）

訊號路徑
--------
    主 app ──▶ 多輸出裝置（喇叭 ＋ BlackHole 2ch）
                                  │
                                  ▼
                          loop_deck 錄下來 ──▶ 喇叭（只送 loop 層）

主 app 完全不知道這支程式存在。它只是把輸出送到一個「同時含 BlackHole」
的裝置。loop_deck 從 BlackHole 收到「主 app 剛送出去的東西」，錄成循環
層，再從自己的輸出送去喇叭。

為什麼不做直通（passthrough）
    直通＝主 app 輸出只送 BlackHole、由本程式轉送。那樣本程式一掛＝全場
    沒聲音。改成「多輸出裝置」後，本程式掛掉只是底床沒了，主 app 的聲音
    照樣從喇叭出來。救場成本差很多。

按鍵（打在本程式的終端機視窗）
    空白   第一次＝開始錄／再按＝停止並定下 loop 長度、立刻開始循環
           之後＝疊下一層（從當前相位開始錄，繞滿一圈自動加入）
    u      撤銷最後一層
    c      全部清空（回到沒有底床）
    m      靜音／解除（底床瞬間消失，救場用）
    q      離開

回授安全
    輸入是 BlackHole＝主 app 的輸出，不是麥克風。疊層不會把房間聲收進來，
    也不會無限回授。前提：**不要把本程式的輸出裝置也設成含 BlackHole 的
    裝置**，那會讓自己錄自己。啟動時會擋掉這個設定。
"""
import argparse
import queue
import sys
import termios
import threading
import time
import tty

import numpy as np
import sounddevice as sd

SR_DEFAULT = 48000
MAX_LOOP_S = 90.0          # 第一層最長；超過就自動停錄


class LoopDeck:
    def __init__(self, sr, ch, block, max_loop_s, lat_ms, gain):
        self.sr, self.ch, self.block = sr, ch, block
        self.gain = float(gain)
        self.lat = int(round(lat_ms * 1e-3 * sr))   # 輸入延遲補償（樣本）
        self.cap = int(max_loop_s * sr)

        self.L = 0                                  # loop 長度（樣本）；0＝還沒定
        self.mix = None                             # (L, ch) 已 commit 的層總和
        self.layers = []                            # 每層一份 (L, ch)，撤銷用
        self.layer_on = []                          # 每層開／關（關＝不出聲但留著）
        self.beats = 0                              # 一圈幾拍；0＝沒有拍的概念
        self.bpm = 0.0                              # 速度；0＝自由長度（按停決定）
        self.clk = 0                                # 自由運行的節拍時鐘（樣本）
        self.armed = False                          # 已按錄、正在等下一個第 1 拍
        self.cin_beats = 4                          # 預備幾拍（0＝按了就錄）
        self.cin = 0                                # 預備還剩幾個樣本
        self.cin_all = 0                            # 這次預備一共幾個樣本
        self.p = 0                                  # 播放相位

        self.rec = False                            # 正在錄
        self.first = True                           # 下一次錄是不是第一層
        self.pend = np.zeros((self.cap, ch), np.float32)
        self.pend_n = 0                             # 第一層已寫入長度
        self.wpos = 0                               # 第 2 層起的寫入位置
        self.wdone = 0                              # 第 2 層起已寫入樣本數

        self.muted = False
        self.in_n = 0                               # 錄音 callback 進來幾次
        self.xruns = 0
        self.click_hi = self._mk_click(1800.0)      # 第 1 拍
        self.click_lo = self._mk_click(1100.0)      # 其他拍
        self.click_cur = self.click_lo
        self.click_i = -1                           # 敲聲播到第幾個樣本；-1＝沒在敲
        self.lock = threading.Lock()
        self.msg = queue.Queue()                    # callback 不列印，丟訊息出去

    # ---- 預備拍的「噠」 -------------------------------------------------
    def _mk_click(self, f):
        """30 毫秒的短音。只在預備拍出現，錄音時不敲。"""
        n = int(self.sr * 0.03)
        t = np.arange(n, dtype=np.float32) / self.sr
        w = np.sin(2 * np.pi * f * t) * np.exp(-t * 70.0) * 0.30
        return np.repeat(w.astype(np.float32)[:, None], self.ch, axis=1)

    def beat_len(self):
        """一拍幾個樣本；沒設速度＝0。"""
        return int(round(self.sr * 60.0 / self.bpm)) if self.bpm > 0 else 0

    def _emit_click(self, outdata, frames):
        """上一塊沒播完的敲聲，接在這一塊開頭播完。"""
        if self.click_i < 0:
            return
        c = self.click_cur
        n = min(frames, c.shape[0] - self.click_i)
        if n <= 0:
            self.click_i = -1
            return
        outdata[:n] += c[self.click_i:self.click_i + n]
        self.click_i += n
        if self.click_i >= c.shape[0]:
            self.click_i = -1

    def _start_click(self, outdata, frames, off, accent):
        c = self.click_hi if accent else self.click_lo
        self.click_cur = c
        n = min(frames - off, c.shape[0])
        outdata[off:off + n] += c[:n]
        self.click_i = n if n < c.shape[0] else -1

    # ---- 音訊 callback（不配置記憶體、不列印） -------------------------
    def rec_cb(self, indata, frames, tinfo, status):
        if status:
            self.xruns += 1
        self.in_n += 1                              # 心跳：證明錄音裝置真的在送資料
        if not self.rec:
            return
        if self.first:
            n = min(frames, self.cap - self.pend_n)
            if n > 0:
                self.pend[self.pend_n:self.pend_n + n] = indata[:n]
                self.pend_n += n
            if self.pend_n >= self.cap:             # 撞到上限＝自動停
                self.rec = False
                self.msg.put(("auto_stop", None))
            return
        # 第 2 層起：長度已知，寫進環狀位置，繞滿一圈就收工
        L = self.L
        n = min(frames, L - self.wdone)
        if n <= 0:
            return
        w = self.wpos
        end = w + n
        if end <= L:
            self.pend[w:end] = indata[:n]
        else:
            k = L - w
            self.pend[w:L] = indata[:k]
            self.pend[0:end - L] = indata[k:n]
        self.wpos = end % L
        self.wdone += n
        if self.wdone >= L:
            self.rec = False
            self.msg.put(("commit", None))

    def play_cb(self, outdata, frames, tinfo, status):
        if status:
            self.xruns += 1
        outdata[:] = 0.0
        self._emit_click(outdata, frames)

        if self.cin > 0:                         # ── 預備拍：時鐘還沒開始走
            bl = self.beat_len()
            done = self.cin_all - self.cin
            if bl:
                k = (done + bl - 1) // bl
                off = k * bl - done
                if off < frames:
                    self._start_click(outdata, frames, off, k == 0)
            self.cin -= frames
            if self.cin <= 0:                    # 數完＝從這一刻起是第 1 拍
                self.cin = 0
                L = self.L
                if not L:                        # 主執行緒剛好取消掉了＝不開錄
                    return
                self.clk = 0
                self.p = 0
                self.wpos = self.lat % L
                self.wdone = 0
                self.rec = True
                self.msg.put(("rec_bar1", None))
            return

        base = self.clk
        self.clk = base + frames                 # 時鐘永遠在走＝節拍器不必等錄音
        L = self.L
        if self.armed and L:
            # 等第 1 拍：時鐘在這條 callback 裡走，所以繞回起點的那一刻在這裡
            # 判最準。以前是在錄音 callback 裡判「圈內位置 < 兩個 block」——
            # 那是另一條執行緒、另一顆時鐘，窗口只有幾毫秒，錯過就再等一整圈，
            # 甚至圈圈都錯過＝按了永遠不開錄。
            bl = self.beat_len()
            if bl and self.cin_beats and (L - base % L) <= min(self.cin_beats * bl, L):
                pos = base % L                   # 起點前幾拍先敲，讓人接得上
                k = (pos + bl - 1) // bl
                off = k * bl - pos
                if off < frames:
                    self._start_click(outdata, frames, off, False)
            if (base % L) + frames >= L:
                self.armed = False
                self.wpos = self.lat % L
                self.wdone = 0
                self.rec = True
                self.msg.put(("rec_layer", None))
        if L and self.bpm > 0:
            self.p = base % L                    # 有速度時播放位置綁在時鐘上，不會漂
        if self.mix is None or self.muted or L == 0:
            return
        p = self.p
        end = p + frames
        if end <= L:
            outdata[:] += self.mix[p:end]
        else:
            k = L - p
            outdata[:k] += self.mix[p:L]
            outdata[k:] += self.mix[0:end - L]
        if self.bpm <= 0:
            self.p = end % L

    # ---- 控制（主執行緒） ----------------------------------------------
    def target_L(self):
        """設了速度和拍數時，一圈該有多長（樣本）。沒設就回 0＝自由長度。"""
        if self.bpm > 0 and self.beats > 0:
            return int(round(self.sr * 60.0 / self.bpm * self.beats))
        return 0

    def toggle(self):
        if self.cin > 0:                            # 預備數到一半反悔
            self.cin = self.cin_all = 0
            self.rec = False                        # 音訊那邊剛好搶開錄也一併收掉
            self.L = 0                              # 一圈長度還沒定案，退回待命
            self.first = True
            self.msg.put(("cancel_arm", None))
            return
        if self.armed:                              # 還沒開始錄就反悔
            self.armed = False
            self.msg.put(("cancel_arm", None))
            return
        if self.rec:
            if self.first:
                self._commit_first()
            else:
                self.rec = False                    # 提早喊停＝丟掉未錄滿的一圈
                self.msg.put(("cancel", None))
            return

        tl = self.target_L()
        if tl and not self.layers:
            # 節拍器模式、還一軌都沒有：一圈長度算得出來，而且還沒有東西在播＝
            # 沒有既有拍點要對。所以不等，按下去那一刻就是第 1 拍（時鐘歸零），
            # 之後照樣錄滿一圈自動收。等下一圈只在「已經有底床」時才有意義。
            with self.lock:
                self.L = tl
                self.first = False
                self.clk = 0
                self.p = 0
                bl = self.beat_len()
                if self.cin_beats and bl:
                    self.cin_all = self.cin = self.cin_beats * bl
                    self.msg.put(("countin", self.cin_beats))
                else:
                    self.wpos = self.lat % tl
                    self.wdone = 0
                    self.rec = True
                    self.msg.put(("rec_bar1", None))
            return

        if self.first:
            self.pend_n = 0
            self.rec = True
            self.msg.put(("rec_first", None))
        elif self.bpm > 0:
            if self.L == 0:
                return
            self.armed = True
            self.msg.put(("armed", None))
        else:
            if self.L == 0:
                return
            self.wpos = (self.p + self.lat) % self.L   # 對齊播放相位，補輸入延遲
            self.wdone = 0
            self.rec = True
            self.msg.put(("rec_layer", None))

    def _rebuild_locked(self):
        """從還開著的層重算 mix。全部關掉＝出零，不是 None（None 會被當成
        「還沒錄」，UI 會退回待命）。"""
        if not self.layers:
            self.mix = None
            return
        acc = None
        for lay, on in zip(self.layers, self.layer_on):
            if not on:
                continue
            acc = lay.copy() if acc is None else acc + lay
        self.mix = (acc * self.gain) if acc is not None \
            else np.zeros((self.L, self.ch), np.float32)

    def _commit_first(self):
        self.rec = False
        n = self.pend_n
        if n < int(0.25 * self.sr):
            self.msg.put(("too_short", None))
            return
        n = max(0, n - self.lat)
        with self.lock:
            layer = self.pend[self.lat:self.lat + n].copy() if self.lat else self.pend[:n].copy()
            self.L = layer.shape[0]
            self.layers = [layer]
            self.layer_on = [True]
            self.p = 0
            self.first = False
            self._rebuild_locked()
        self.msg.put(("first_done", self.L / self.sr))

    def commit_layer(self):
        with self.lock:
            self.layers.append(self.pend[:self.L].copy())
            self.layer_on.append(True)
            self._rebuild_locked()
        self.msg.put(("layer_done", len(self.layers)))

    def toggle_layer(self, i):
        with self.lock:
            if 0 <= i < len(self.layers):
                self.layer_on[i] = not self.layer_on[i]
                self._rebuild_locked()
                return self.layer_on[i]
        return None

    def drop(self, i):
        """刪掉指定的一層。刪光＝回到沒錄過的狀態。"""
        with self.lock:
            if not (0 <= i < len(self.layers)):
                return len(self.layers)
            self.layers.pop(i)
            self.layer_on.pop(i)
            if not self.layers:
                self.clear_locked()
                return 0
            self._rebuild_locked()
            return len(self.layers)

    def undo(self):
        return self.drop(len(self.layers) - 1)

    def clear_locked(self):
        self.layers = []
        self.layer_on = []
        self.mix = None
        self.L = 0
        self.p = 0
        self.first = True
        self.rec = False
        self.armed = False
        self.cin = self.cin_all = 0

    def clear(self):
        with self.lock:
            self.clear_locked()


def _resolve(name_or_idx, kind):
    """裝置名（部分字串）或編號 → 編號。"""
    if name_or_idx is None:
        return None
    try:
        return int(name_or_idx)
    except ValueError:
        pass
    want = str(name_or_idx).lower()
    hits = []
    for i, d in enumerate(sd.query_devices()):
        chans = d["max_input_channels"] if kind == "in" else d["max_output_channels"]
        if chans > 0 and want in d["name"].lower():
            hits.append(i)
    if not hits:
        sys.exit(f"找不到{'輸入' if kind == 'in' else '輸出'}裝置：{name_or_idx}")
    return hits[0]



# ── 訊息處理（兩種介面共用）────────────────────────────────────────────
_hb = {"n": -1, "t": 0.0, "warned": False}     # 錄音心跳：上次看到的計數與時間

def pump(deck, max_loop_s, say):
    """把 callback 丟出來的訊息消化掉。say(text) 決定顯示到哪。"""
    # 錄音裝置停止送資料的話，畫面會停在「錄音中」不動。與其讓人乾等，直接講。
    now = time.monotonic()
    if deck.in_n != _hb["n"]:
        _hb.update(n=deck.in_n, t=now, warned=False)
    elif now - _hb["t"] > 2.0 and not _hb["warned"]:
        _hb["warned"] = True
        say("⚠ 錄音裝置 2 秒沒送資料進來（麥克風權限？BlackHole 沒在跑？）")
    while not deck.msg.empty():
        kind, val = deck.msg.get()
        if kind == "rec_first":
            say("● 錄第一層…（再按一次停）")
        elif kind == "countin":
            say(f"預備 {val} 拍…（數完自己開始錄）")
        elif kind == "rec_bar1":
            say("● 錄第 1 層…（從這一刻算第 1 拍，錄滿一圈自動收）")
        elif kind == "rec_layer":
            say(f"● 疊第 {len(deck.layers) + 1} 層…（錄滿一圈自動加入）")
        elif kind == "first_done":
            say(f"✓ loop 長度 {val:.2f}s，開始循環")
        elif kind == "commit":
            deck.commit_layer()
        elif kind == "layer_done":
            say(f"✓ 第 {val} 層加入")
        elif kind == "armed":
            say("⏳ 等下一個第 1 拍…")
        elif kind == "cancel_arm":
            say("✗ 不錄了")
        elif kind == "cancel":
            say("✗ 這一層取消（沒錄滿一圈）")
        elif kind == "too_short":
            say("✗ 太短（<0.25s），沒收")
        elif kind == "auto_stop":
            say(f"⚠ 撞到 {max_loop_s:.0f}s 上限，自動停錄")
            deck._commit_first()


def _act(deck, c):
    """一個按鍵 → 一個動作。回傳要顯示的話，或 None（沒話說），或 "QUIT"。"""
    if c == " ":
        deck.toggle(); return None
    if c == "u":
        n = deck.undo(); return f"↩ 撤銷，剩 {n} 層" if n else "↩ 撤銷，已清空"
    if c == "c":
        deck.clear(); return "⌫ 清空"
    if c == "m":
        deck.muted = not deck.muted
        return "🔇 靜音" if deck.muted else "🔊 解除靜音"
    if c in ("q", "\x03"):
        return "QUIT"
    return None


# ── 介面一：視窗（預設）────────────────────────────────────────────────
def run_ui(deck, a, i_name, o_name, rs, ps):
    import tkinter as tk

    BG, PANEL, FG, DIM = "#14161a", "#1c2027", "#e8eaed", "#7a8290"
    IDLE, REC, PLAY, MUTE = "#5a6472", "#e5484d", "#3dd68c", "#e8a33d"

    root = tk.Tk()
    root.title("疊層底床")
    root.configure(bg=BG)
    root.geometry("470x600")
    root.minsize(430, 520)
    if not a.no_top:
        root.attributes("-topmost", True)

    # ── 狀態區 ──
    head = tk.Frame(root, bg=BG)
    head.pack(pady=(16, 0))
    lamp = tk.Label(head, text="", font=("Helvetica", 40, "bold"), bg=BG, fg=IDLE)
    lamp.pack(side="left")
    # 每繞回起點閃一下＝看得到「一圈開始了」，不必心裡數
    flash = tk.Label(head, text="●", font=("Helvetica", 26), bg=BG, fg=BG)
    flash.pack(side="left", padx=(10, 0))

    sub = tk.Label(root, text="", font=("Helvetica", 14), bg=BG, fg=DIM)
    sub.pack()

    cv = tk.Canvas(root, height=26, bg="#20242b", highlightthickness=0)
    cv.pack(fill="x", padx=24, pady=(14, 2))
    bar = cv.create_rectangle(0, 0, 0, 26, fill=IDLE, width=0)
    grid_ids = []

    # 節拍燈：一拍一顆，第 1 拍是重拍（換顏色）。錄第一軌時就能跟著它唱。
    dots = tk.Canvas(root, height=26, bg=BG, highlightthickness=0)
    dots.pack(fill="x", padx=24, pady=(6, 0))
    dot_ids = []

    beatrow = tk.Frame(root, bg=BG)
    beatrow.pack(pady=(4, 6))
    tk.Label(beatrow, text="速度", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    pv = tk.StringVar(value=f"{deck.bpm:.0f}")
    ent_bpm = tk.Entry(beatrow, textvariable=pv, width=4, justify="center",
                       font=("Helvetica", 13))
    ent_bpm.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="× 一圈", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    bv = tk.StringVar(value=str(deck.beats))
    ent = tk.Entry(beatrow, textvariable=bv, width=4, justify="center",
                   font=("Helvetica", 13))
    ent.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="拍", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    tk.Label(beatrow, text="　預備", font=("Helvetica", 12), bg=BG,
             fg=DIM).pack(side="left")
    cinv = tk.StringVar(value=str(deck.cin_beats))
    ent_cin = tk.Entry(beatrow, textvariable=cinv, width=3, justify="center",
                       font=("Helvetica", 13))
    ent_cin.pack(side="left", padx=(4, 2))
    tk.Label(beatrow, text="拍", font=("Helvetica", 12), bg=BG, fg=DIM).pack(side="left")
    beatlbl = tk.Label(beatrow, text="", font=("Helvetica", 11), bg=BG, fg=DIM)
    beatlbl.pack(side="left", padx=(8, 0))

    # ── 軌道列表 ──
    tk.Label(root, text="軌", font=("Helvetica", 12), bg=BG, fg=DIM,
             anchor="w").pack(fill="x", padx=24, pady=(6, 2))
    tracks = tk.Frame(root, bg=PANEL)
    tracks.pack(fill="both", expand=True, padx=24)

    # ── 按鈕 ──
    big = tk.Button(root, text="錄 / 停　(空白鍵)", height=2,
                    highlightbackground=BG, command=lambda: press(" "))
    big.pack(fill="x", padx=24, pady=(10, 4))
    btns = tk.Frame(root, bg=BG)
    btns.pack(pady=(2, 6))
    for text, key in (("撤銷 (u)", "u"), ("清空 (c)", "c"), ("靜音 (m)", "m")):
        tk.Button(btns, text=text, width=9, highlightbackground=BG,
                  command=lambda k=key: press(k)).pack(side="left", padx=4)

    # ── 輸出裝置（現場可換）──
    # 聚合裝置的成員被拔掉之後還會留在清單上但沒有輸出通道，所以只列真的能
    # 出聲的；BlackHole 直接排除（底床送回去＝自己錄自己）。
    outs = [(i, d["name"]) for i, d in enumerate(sd.query_devices())
            if d["max_output_channels"] > 0
            and "blackhole" not in d["name"].lower()]
    olabels = [f"{i}｜{nm}" for i, nm in outs]
    cur = next((l for l, (i, nm) in zip(olabels, outs) if nm == o_name),
               olabels[0] if olabels else o_name)

    devrow = tk.Frame(root, bg=BG)
    devrow.pack(side="bottom", pady=(2, 6))
    tk.Label(devrow, text=f"錄：{i_name}　→　播：", font=("Helvetica", 10),
             bg=BG, fg=DIM).pack(side="left")
    ovar = tk.StringVar(value=cur)
    om = tk.OptionMenu(devrow, ovar, *(olabels or [cur]),
                       command=lambda v: switch_out(v))
    om.configure(font=("Helvetica", 10), highlightthickness=0)
    om.pack(side="left")

    stream = {"ps": ps, "label": cur}

    def switch_out(label):
        """換喇叭：先停舊的再開新的。開不起來就把舊的接回去，不會變成沒聲音。"""
        if label == stream["label"]:
            return
        idx = int(label.split("｜")[0])
        old = stream["ps"]
        try:
            old.stop()
        except Exception:
            pass
        try:
            new = sd.OutputStream(device=idx, channels=a.ch, samplerate=a.sr,
                                  blocksize=a.block, dtype="float32",
                                  callback=deck.play_cb)
            new.start()
        except Exception as e:
            try:
                old.start()
            except Exception:
                pass
            ovar.set(stream["label"])
            say(f"⛔ 換不過去（{e.__class__.__name__}），維持原來的")
            return
        try:
            old.close()
        except Exception:
            pass
        stream["ps"], stream["label"] = new, label
        nm = label.split("｜")[1]
        if "bh" in nm.lower():                  # 名字裡有 bh＝底床會繞回 BlackHole
            say(f"⚠ 「{nm}」會把底床送回 BlackHole＝下一軌會錄到自己")
        else:
            say(f"🔊 改從「{nm}」出聲")

    msg = {"t": ""}
    # 畫布寬度在剛開視窗時還是 1（版面還沒算完），所以格線和節拍燈都要記住
    # 「上次是用多寬畫的」，寬度一變就重畫，不然會全部擠在最左邊看不到。
    ui = {"sig": None, "last_p": 0, "flash": 0, "gw": 0, "dw": 0}

    def say(t):
        msg["t"] = t

    def press(c):
        """按鈕按下來的動作。一定執行，並把游標帶離拍數欄位——不然按完
        按鈕，空白鍵還是會被那個欄位吃掉。"""
        root.focus_set()
        r = _act(deck, c)
        if r == "QUIT":
            close()
        elif r:
            say(r)

    def on_key(c):
        """鍵盤按下來的動作。只有這條路要讓拍數欄位優先：游標在欄位裡時，
        空白是打字不是錄音。滑鼠按鈕不走這裡，所以不受影響。"""
        if root.focus_get() in (ent, ent_bpm, ent_cin) \
                and c in (" ", "u", "c", "m", "q"):
            return
        press(c)

    def close():
        try:
            rs.stop(); rs.close()
            stream["ps"].stop(); stream["ps"].close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    for k in (" ", "u", "c", "m", "q"):
        root.bind(f"<KeyPress-{'space' if k == ' ' else k}>", lambda e, k=k: on_key(k))
    # 在拍數欄位按 Enter＝填好了，游標交還出去，空白鍵立刻恢復錄音
    for e_ in (ent, ent_bpm, ent_cin):
        e_.bind("<Return>", lambda e: root.focus_set())
        e_.bind("<Escape>", lambda e: root.focus_set())

    def rebuild_tracks():
        for w in tracks.winfo_children():
            w.destroy()
        if not deck.layers:
            tk.Label(tracks, text="還沒有軌", font=("Helvetica", 12),
                     bg=PANEL, fg=DIM).pack(pady=14)
            return
        for i in range(len(deck.layers)):
            on = deck.layer_on[i]
            row = tk.Frame(tracks, bg=PANEL)
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=f"{i + 1}", font=("Helvetica", 13, "bold"), width=2,
                     bg=PANEL, fg=(FG if on else DIM)).pack(side="left")
            tk.Label(row, text=f"{deck.layers[i].shape[0] / deck.sr:.2f}s",
                     font=("Helvetica", 12), bg=PANEL,
                     fg=(FG if on else DIM)).pack(side="left", padx=(6, 0))
            tk.Button(row, text="刪", width=3, highlightbackground=PANEL,
                      command=lambda i=i: (deck.drop(i), say(f"刪掉第 {i + 1} 軌"))
                      ).pack(side="right", padx=2)
            tk.Button(row, text=("開" if on else "關"), width=3,
                      highlightbackground=PANEL,
                      command=lambda i=i: (deck.toggle_layer(i), say(""))
                      ).pack(side="right", padx=2)

    def draw_grid(n, w):
        for gid in grid_ids:
            cv.delete(gid)
        grid_ids.clear()
        for k in range(1, max(n, 0)):
            x = w * k / n
            grid_ids.append(cv.create_line(x, 0, x, 26, fill="#3a4150", width=1))

    def tick():
        pump(deck, a.max_loop_s, say)
        sr = deck.sr

        try:
            deck.beats = max(0, min(64, int(bv.get() or 0)))
        except ValueError:
            deck.beats = 0
        try:
            deck.cin_beats = max(0, min(16, int(cinv.get() or 0)))
        except ValueError:
            deck.cin_beats = 0
        if not deck.layers:            # 已經有軌之後速度就鎖住，不然格子會亂掉
            try:
                deck.bpm = max(0.0, min(300.0, float(pv.get() or 0)))
            except ValueError:
                deck.bpm = 0.0
            ent_bpm.configure(state="normal")
        elif str(ent_bpm.cget("state")) != "readonly":
            ent_bpm.configure(state="readonly")

        if deck.cin > 0:
            bl = deck.beat_len() or 1
            col, txt = MUTE, f"預備 {-(-deck.cin // bl)}"
            done, total = (deck.cin_all - deck.cin) / sr, (deck.cin_all or 1) / sr
            sub.configure(text="數完就開始錄　（再按一次不錄）")
        elif deck.armed:
            col, txt = MUTE, "⏳ 等第 1 拍"
            total = deck.L / sr if deck.L else 1
            done = (deck.clk % deck.L) / sr if deck.L else 0
            sub.configure(text=f"下一圈開頭就開始錄　（再按一次不錄）")
        elif deck.rec:
            col, txt = REC, "● 錄音中"
            if deck.first:
                done, total = deck.pend_n / sr, a.max_loop_s
                sub.configure(text=f"第一軌　{done:.1f}s　（再按空白停）")
            else:
                done, total = deck.wdone / sr, deck.L / sr
                lab = f"疊第 {len(deck.layers) + 1} 軌" if deck.layers else "錄第 1 軌"
                sub.configure(text=f"{lab}　剩 {total - done:.1f}s")
        elif deck.muted:
            col, txt = MUTE, "🔇 靜音"
            done, total = (deck.p / sr, deck.L / sr) if deck.L else (0, 1)
            sub.configure(text="底床暫時消失　再按 m 回來")
        elif deck.L:
            col, txt = PLAY, "▶ 循環中"
            done, total = deck.p / sr, deck.L / sr
            if deck.beats:
                b = int(done / total * deck.beats) + 1
                sub.configure(text=f"第 {b} / {deck.beats} 拍　·　{done:.1f} / {total:.1f}s")
            else:
                sub.configure(text=f"{done:.1f} / {total:.1f}s")
        elif deck.bpm > 0 and deck.beats > 0:
            # 還沒錄任何東西，但節拍器已經在跑＝跟著它唱，按錄就對得上
            col, txt = IDLE, "♩ 節拍器"
            tl = deck.target_L() or 1
            done, total = (deck.clk % tl) / sr, tl / sr
            sub.configure(text=f"{deck.bpm:.0f} BPM　一圈 {total:.2f}s　按錄跟著唱")
        else:
            col, txt = IDLE, "待命"
            done, total = 0, 1
            sub.configure(text="填速度＋拍數，或直接按空白自由錄")

        lamp.configure(text=txt, fg=col)

        # 繞回起點＝閃一下
        if deck.L and deck.p < ui["last_p"]:
            ui["flash"] = 4
        ui["last_p"] = deck.p
        if ui["flash"] > 0:
            flash.configure(fg=col)
            ui["flash"] -= 1
        else:
            flash.configure(fg=BG)

        w = max(cv.winfo_width(), 1)
        cv.coords(bar, 0, 0, w * min(done / total if total else 0, 1.0), 26)
        cv.itemconfigure(bar, fill=col)
        if len(grid_ids) != max(deck.beats - 1, 0) or ui["gw"] != w:
            draw_grid(deck.beats, w)
            ui["gw"] = w
        for gid in grid_ids:
            cv.tag_raise(gid)

        sig = (len(deck.layers), tuple(deck.layer_on))
        if sig != ui["sig"]:
            ui["sig"] = sig
            rebuild_tracks()

        # ── 節拍燈 ──
        n = deck.beats
        span = deck.L or deck.target_L()
        dw = max(dots.winfo_width(), 1)
        if len(dot_ids) != n or ui["dw"] != dw:
            for did in dot_ids:
                dots.delete(did)
            dot_ids.clear()
            ui["dw"] = dw
            for k in range(n):
                x = dw * (k + 0.5) / n
                dot_ids.append(dots.create_oval(x - 7, 5, x + 7, 19,
                                                fill=BG, outline="#3a4150", width=2))
        if n and span:
            if deck.cin > 0:
                bl = deck.beat_len() or 1
                cur = ((deck.cin_all - deck.cin) // bl) % n
            else:
                pos = (deck.clk % span) if deck.bpm > 0 else deck.p
                cur = int(pos / span * n) % n
            for k, did in enumerate(dot_ids):
                if k == cur:
                    dots.itemconfigure(did, fill=(FG if k == 0 else col),
                                       outline=(FG if k == 0 else col))
                else:
                    dots.itemconfigure(did, fill=BG, outline="#3a4150")

        on_n = sum(1 for x in deck.layer_on if x)
        if deck.layers:
            beatlbl.configure(text=f"{on_n}/{len(deck.layers)} 軌出聲　·　速度已鎖")
        else:
            beatlbl.configure(text="0＝自由長度")

        root.after(50, tick)

    rebuild_tracks()
    tick()
    root.mainloop()
    print(f"[loop_deck] 收場（xruns {deck.xruns}）")


# ── 介面二：終端機按鍵（--no-ui）──────────────────────────────────────
def run_keys(deck, a, rs, ps):
    import select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            pump(deck, a.max_loop_s, print)
            r, _, _ = select.select([fd], [], [], 0.05)
            if not r:
                continue
            out = _act(deck, sys.stdin.read(1))
            if out == "QUIT":
                break
            if out:
                print(out)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        rs.stop(); rs.close()
        ps.stop(); ps.close()
        print(f"\n[loop_deck] 收場（xruns {deck.xruns}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dev", default="BlackHole", help="錄音來源（預設 BlackHole）")
    ap.add_argument("--out-dev", default=None, help="底床送去哪（預設系統輸出）")
    ap.add_argument("--sr", type=int, default=SR_DEFAULT)
    ap.add_argument("--ch", type=int, default=2)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--max-loop-s", type=float, default=MAX_LOOP_S)
    ap.add_argument("--lat-ms", type=float, default=0.0,
                    help="輸入延遲補償；疊層聽起來慢半拍就往上加")
    ap.add_argument("--gain", type=float, default=1.0, help="底床整體音量")
    ap.add_argument("--list", action="store_true", help="只列裝置就離開")
    ap.add_argument("--bpm", type=float, default=0.0,
                    help="速度；設了就有節拍器，第一軌長度也自動算（0＝自由長度）")
    ap.add_argument("--beats", type=int, default=0, help="一圈幾拍")
    ap.add_argument("--no-ui", action="store_true", help="不開視窗，用終端機按鍵")
    ap.add_argument("--no-top", action="store_true", help="視窗不要一直蓋在最上面")
    a = ap.parse_args()

    if a.list:
        for i, d in enumerate(sd.query_devices()):
            print(f"{i:2d} | {d['name']:32s} | in {d['max_input_channels']} | out {d['max_output_channels']}")
        return

    i_dev = _resolve(a.in_dev, "in")
    o_dev = _resolve(a.out_dev, "out") if a.out_dev else sd.default.device[1]
    i_name = sd.query_devices(i_dev)["name"]
    o_name = sd.query_devices(o_dev)["name"]

    if "blackhole" in o_name.lower():
        sys.exit(f"⛔ 輸出裝置是 {o_name}＝會自己錄自己。換一個輸出裝置。")

    # 終端機模式沒有鍵盤就沒有意義（視窗模式自己有鍵盤，不受影響）。
    # 在背景／管線裡跑會在 tcgetattr 炸出 traceback，先擋下來講人話。
    if a.no_ui and not sys.stdin.isatty():
        sys.exit("⛔ 沒有鍵盤可用（不是終端機視窗）。\n"
                 "   請用「開啟疊層底床.command」開，或在終端機裡直接跑。")

    deck = LoopDeck(a.sr, a.ch, a.block, a.max_loop_s, a.lat_ms, a.gain)
    deck.bpm = max(0.0, a.bpm)
    deck.beats = max(0, a.beats)

    print(f"[loop_deck] 錄： {i_name}")
    print(f"[loop_deck] 播： {o_name}")
    print(f"[loop_deck] {a.sr}Hz {a.ch}ch block={a.block} 延遲補償={a.lat_ms}ms 上限={a.max_loop_s:.0f}s")
    if deck.target_L():
        print(f"[loop_deck] 節拍器 {deck.bpm:.0f} BPM × {deck.beats} 拍 "
              f"＝ 一圈 {deck.target_L() / a.sr:.2f}s")
    print("[loop_deck] 空白=錄／停　u=撤銷　c=清空　m=靜音　q=離開")

    rs = sd.InputStream(device=i_dev, channels=a.ch, samplerate=a.sr,
                        blocksize=a.block, dtype="float32", callback=deck.rec_cb)
    ps = sd.OutputStream(device=o_dev, channels=a.ch, samplerate=a.sr,
                         blocksize=a.block, dtype="float32", callback=deck.play_cb)
    rs.start()
    ps.start()
    print("[loop_deck] running")

    if a.no_ui:
        run_keys(deck, a, rs, ps)
    else:
        run_ui(deck, a, i_name, o_name, rs, ps)


if __name__ == "__main__":
    main()
