"""BrainV2: the second-chorus-partner model behind the v1 Brain interface.

step(sop_token) -> alto token, same contract solo_host's Conductor expects;
phase comes from an internal tick counter (wall-clock 16th cycle -- the
same assumption peek_probe_v2 measured, lower bound without a click), and
the context is a sliding window (default 2048 = 3.2 min of memory, 48 ms
per forward in vcclient-dev -- the live-safe config; offline render uses
the same so the ear hears what live would do).

accent: 'chorale' | 'pop909' | 'cpdl' -- the source-embedding conditioning.
"""

from pathlib import Path

import torch

from live import HOLD, REST
from train_v2 import SRC, HarmonyTransformerV2

TEMPERATURE, TOP_K = 0.9, 8
WINDOW = 2048


class BrainV2:
    def __init__(self, accent="cpdl", window=WINDOW):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        ck = torch.load(Path(__file__).parent / "checkpoints" / "v2" / "best.pt",
                        map_location=self.device, weights_only=True)
        self.model = HarmonyTransformerV2(ck["vocab"]).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.ctx, self.phases = [], []
        self.window = window
        self.src = torch.tensor([SRC[accent]], device=self.device)
        self.tick = 0
        with torch.no_grad():  # warm the kernels so tick 1 isn't slow
            self.model(torch.zeros(1, 8, dtype=torch.long, device=self.device),
                       torch.zeros(1, 8, dtype=torch.long, device=self.device),
                       self.src)

    @torch.no_grad()
    def logits_next(self):
        x = torch.tensor([self.ctx[-self.window:]], device=self.device)
        p = torch.tensor([self.phases[-self.window:]], device=self.device)
        return self.model(x, p, self.src)[0, -1]

    @torch.no_grad()
    def step_anticipate(self, voiced_hint=True):
        """預感步（任務5，2026-07-26）：不聽這一 tick 他唱什麼，先用腦自己對
        「他下一顆音」的預報當輸入，提前決定天使的音——天使與他同時落地，
        而非慢一個 tick（187ms）。狀態不留：context 在回傳前復原，真實 token
        隨後由 step() 正式寫入（預感錯了不污染記憶）。

        定價（peek_probe_v2 實測他的真嗓）：同樂句第二遍 pitch-masked top1
        ≈40% / top3 ≈70%（G29 的 Bach 基線 11–16%＝亂猜）。G29 的 timing 半
        仍差，故本法只提前決定「音高」，發聲時機仍由真實 onset 觸發。"""
        if not self.ctx:
            return None
        n0 = len(self.ctx)
        # 只預測「音高」，不預測「有沒有在唱」：發聲狀態用他當下的真實狀態
        # （voiced_hint，live 也拿得到，零預測風險）。不遮 REST/HOLD 的話腦
        # 會一路押 HOLD（G29），天使發聲率從 92% 掉到 42%＝直接閉嘴。
        lg = self.logits_next().clone()
        if voiced_hint:
            lg[REST] = float("-inf")
            lg[HOLD] = float("-inf")
        fore = int(torch.argmax(lg))                      # 他下一顆音的預報
        phase = self.tick % 16
        self.ctx.append(fore)
        self.phases.append(phase)
        logits = self.logits_next() / TEMPERATURE
        if voiced_hint:
            logits[REST] = float("-inf")
        k = torch.topk(logits, TOP_K)
        alto = int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])
        del self.ctx[n0:]
        del self.phases[n0:]
        return alto

    @torch.no_grad()
    def step(self, sop_token):
        phase = self.tick % 16
        self.tick += 1
        self.ctx.append(sop_token)
        self.phases.append(phase)
        logits = self.logits_next() / TEMPERATURE
        if sop_token != REST:
            logits[REST] = float("-inf")  # while you sing, the angel sings
        k = torch.topk(logits, TOP_K)
        alto = int(k.indices[torch.multinomial(torch.softmax(k.values, -1), 1)])
        self.ctx.append(alto)
        self.phases.append(phase)
        return alto
