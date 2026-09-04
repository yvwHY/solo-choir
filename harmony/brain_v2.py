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
        """Anticipation step (task 5, 2026-07-26). Rather than listening to what
        the singer sings on this tick, the model's own forecast of their next
        note is fed in as the input, so the parts decide early and land together
        with the singer instead of one tick, 187 ms, behind. No state is kept:
        the context is restored before returning, and the real token is written
        by step() afterwards, so a wrong anticipation does not pollute the
        memory.

        The price, measured with peek_probe_v2 on this singer's own voice: on a
        second pass through the same phrase, pitch-masked top-1 is about 40% and
        top-3 about 70%, against the Bach baseline of 11-16% in G29, which is
        guessing. The timing half of G29 is still poor, so this method decides
        only the pitch early; the moment of sounding is still triggered by a real
        onset."""
        if not self.ctx:
            return None
        n0 = len(self.ctx)
        # Predict pitch only, never whether the singer is sounding. The
        # sounding state comes from their real current state (voiced_hint, which
        # live has too, at no predictive risk). Without masking REST and HOLD
        # the model bets on HOLD throughout (G29) and the parts' sounding rate
        # falls from 92% to 42%, which is silence.
        lg = self.logits_next().clone()
        if voiced_hint:
            lg[REST] = float("-inf")
            lg[HOLD] = float("-inf")
        fore = int(torch.argmax(lg))                      # the forecast of the singer's next note
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
