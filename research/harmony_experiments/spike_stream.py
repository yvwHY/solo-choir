"""spike_stream.py — DDSP 滑窗串流 spike（08-05；G28 案③、G33b 後的下一注）

問題：即時線 batch 觸發陣亡（延遲地板）。這支驗「連續滑窗轉換」的兩個
生死數字：①端到端延遲體感（Harry live 裁）②串流質感有沒有 G28 死味。

架構＝抄 DDSP-SVC gui.py 的即時核心（rolling window＋SOLA 對齊＋
crossfade 接縫），差異：headless（PySimpleGUI 缺＋授權麻煩）、device
改 MPS（原版只認 cuda/cpu）、f0 用 parselmouth（rmvpe 權重不在）。
單聲部 identity 先行（--pitch 可移調）；過了才蓋和聲層。

Run (DDSP venv):
  python spike_stream.py                          # 他的模型 identity
  python spike_stream.py --pitch -12 --model exp/combsub-m4-bass1/model_11000.pt
  python spike_stream.py --sustain 250            # v15 樂句橋接（E 版 live 化）
Ctrl-C 結束；每塊印 infer ms（>block ms＝會斷音）。
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
                   "SoloChoirCode/260724_ddsp_svc")
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, \
    Units_Encoder  # noqa: E402
from ddsp.core import upsample  # noqa: E402
from enhancer import Enhancer  # noqa: E402

SR = 44100
DDSP = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
        "SoloChoirCode/260724_ddsp_svc")
MAJ = [0, 2, 4, 5, 7, 9, 11]


def dia_from_note(note, steps, root):
    """已定的音符 → diatonic steps 的半音偏移。"""
    rel = int(note) - root
    oc, pc = divmod(rel, 12)
    di = min(range(7), key=lambda i: min((MAJ[i] - pc) % 12,
                                         (pc - MAJ[i]) % 12))
    to, ti = divmod(di + steps, 7)
    return float(root + (oc + to) * 12 + MAJ[ti] - int(note))


class NoteTracker:
    """遲滯音符追蹤（v11 長音穩定核心）：偏離 >0.65 半音且持續 ≥3 幀
    （~35ms）才換音＝顫音/微偏跨界不再抖串。逐幀因果＝跨窗一致。"""

    def __init__(self):
        self.cur = None
        self.pend = 0
        self.hist = []
        self.streak = 0

    def feed(self, f0_hz):
        """回 (notes, streaks)：streak＝同一音符連續幀數（0＝剛換音/無聲）
        ＝v12 units 平滑的門（穩定才釘、換音立即跟）。"""
        out = np.zeros(len(f0_hz))
        stk = np.zeros(len(f0_hz), dtype=int)
        for h, f in enumerate(f0_hz):
            if f > 0:
                m = 69 + 12 * np.log2(f / 440.0)
                self.hist = (self.hist + [m])[-5:]
                mm = float(np.median(self.hist))
                if self.cur is None:
                    self.cur = round(mm)
                    self.streak = 0
                elif abs(mm - self.cur) > 0.65:
                    self.pend += 1
                    if self.pend >= 3:
                        self.cur, self.pend = round(mm), 0
                        self.streak = 0
                else:
                    self.pend = 0
                    self.streak += 1
            else:
                self.streak = 0
            out[h] = self.cur if self.cur is not None else 0.0
            stk[h] = self.streak
        return out, stk


def dia_offsets(f0c, steps, root):
    """凍結 f0（Hz per-hop）→ 每幀半音偏移（diatonic steps 內的調內音程）。
    量化只決定音程、曲線不動（D 版哲學＝08-05 驗證過的美學）。
    決定性：只依賴凍結 fc 的局部（±2 幀中位）＝跨窗一致、不需另建快取。"""
    v = f0c > 0
    m = np.where(v, 69 + 12 * np.log2(np.maximum(f0c, 1.0) / 440.0), 0.0)
    # 局部中位（5 幀）壓顫音抖動，避免音界抖串
    k = 2
    mm = np.copy(m)
    for h in range(len(m)):
        w = m[max(0, h - k):h + k + 1]
        w = w[w > 0]
        if len(w):
            mm[h] = np.median(w)
    off = np.zeros(len(m))
    for h in range(len(m)):
        if mm[h] <= 0:
            off[h] = off[h - 1] if h else 0.0
            continue
        note = int(round(mm[h]))
        rel = note - root
        oc, pc = divmod(rel, 12)
        di = min(range(7), key=lambda i: min((MAJ[i] - pc) % 12,
                                             (pc - MAJ[i]) % 12))
        to, ti = divmod(di + steps, 7)
        target = root + (oc + to) * 12 + MAJ[ti]
        off[h] = float(target - note)
    return off


def phase_vocoder(a, b, fade_out, fade_in):
    """gui.py:15 原樣：接縫相位連續化（跳針的主治醫）。"""
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(
        deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    return (a * (fade_out ** 2) + b * (fade_in ** 2)
            + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n)


class Svc:
    """gui.py SvcDDSP 精簡＋多聲部：encoder/f0/vol/enhancer 共用、N 張嘴。
    voices: [(model_path, semitones, gain), ...]"""

    def __init__(self, voices, device="mps", enhance=True):
        self.device = device
        self.voices = []
        for path, semi, gain in voices:
            model, args = load_model(path, device=device)
            self.voices.append((model, semi, gain))
        self.args = args                      # 同管線＝共用 data 參數
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            f"{DDSP}/{self.args.data.encoder_ckpt}",
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size, device=device)
        self.enhancer = (Enhancer(self.args.enhancer.type,
                                  f"{DDSP}/{self.args.enhancer.ckpt}",
                                  device=device) if enhance else None)
        self.vol_ex = Volume_Extractor(self.args.data.block_size)
        self.spk = torch.LongTensor([[1]]).to(device)

    def prep(self, audio, threhold=-60.0, want_uv=False):
        """回 (f0_np, vol_t, mask[, uv])：f0 留 numpy 給呼叫端上凍結網格。
        want_uv＝一併回傳 parselmouth 真 voicing（f0==0 幀；v16 橋接判準：
        喉麥的子音音量掉不到 -60dB 門檻下，聲帶判定才是 E 版 vm 的正身）。
        插值段復刻 vocoder.py:142-146＝f0 與 uv_interp=True 逐 byte 同。"""
        hop = self.args.data.block_size
        pe = F0_Extractor("parselmouth", SR, hop, 65.0, 800.0)
        if want_uv:
            f0_np = pe.extract(audio, uv_interp=False, device=self.device)
            uv = f0_np == 0
            if len(f0_np[~uv]) > 0:
                f0_np[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0],
                                      f0_np[~uv])
            f0_np[f0_np < 65.0] = 65.0
        else:
            f0_np = pe.extract(audio, uv_interp=True, device=self.device)
        vol = self.vol_ex.extract(audio)
        mask = (vol > 10 ** (threhold / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n:n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device)[None, :, None]
        mask = upsample(mask, hop).squeeze(-1)
        vol_t = torch.from_numpy(vol).float().to(self.device)[None, :, None]
        if want_uv:
            return f0_np, vol_t, mask, uv
        return f0_np, vol_t, mask

    def infer(self, audio, pitch_adjust=0.0, threhold=-60.0,
              units_override=None, feats=None, phases=None, ratios=None,
              enh_tail=0):
        """每聲部各自輸出：共同 units/f0/vol、各自移調＋增益。
        feats＝(f0_np, vol_t, mask) 外部給＝不重算（凍結網格用）。
        phases＝每聲部絕對相位（弧度）＝comb 源跨窗同相（v6 跳針正解）。
        ratios＝每聲部逐幀半音偏移陣列（diatonic）；None＝用固定 semi。"""
        hop = self.args.data.block_size
        f0_np, vol_t, mask = feats if feats is not None \
            else self.prep(audio, threhold)
        f0 = torch.from_numpy(f0_np).float().to(self.device)[None, :, None]
        units = units_override
        if units is None:
            units = self.encode(audio)
        n = min(units.size(1), f0.size(1), vol_t.size(1))
        outs, fvs = [], []
        with torch.no_grad():
            for vi, (model, semi, gain) in enumerate(self.voices):
                if ratios is not None and ratios[vi] is not None:
                    r = torch.from_numpy(
                        ratios[vi][:n]).float().to(self.device)[None, :, None]
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + r) / 12.0)
                else:
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + semi) / 12.0)
                ip = (None if phases is None else torch.tensor(
                    [[[phases[vi]]]], dtype=torch.float32,
                    device=self.device))
                out, _, _ = model(units[:, :n], fv, vol_t[:, :n],
                                  spk_id=self.spk, initial_phase=ip)
                out *= mask[:, :n * hop]
                outs.append(out)
                fvs.append(fv)
            if self.enhancer is not None:
                # 三聲部合批一發 hifigan（worklog 08-05 §M 刀位；逐聲部呼叫
                # ＝3 次 kernel 起跳成本）。44.1k/512＋adaptive_key=0 時
                # enhance() 全路徑 batch 透明（無 resample、f0 走 squeeze(-1)）。
                ob, fb = torch.cat(outs, 0), torch.cat(fvs, 0)
                if enh_tail and enh_tail < ob.size(-1):
                    # v23：只 enhance step 實際取用的尾段＋邊距（−28ms 級）。
                    # NSF 相位從呼叫起點積分＝尾段版相位平移，但每窗相位本
                    # 來就重啟、SOLA＋phase_vocoder 是縫的主治醫——是否可
                    # 聞由 replay 接縫指標＋耳測裁，不再紙上判死。
                    tl = (enh_tail // hop + 1) * hop
                    eb, esr = self.enhancer.enhance(
                        ob[:, -tl:], self.args.data.sampling_rate,
                        fb[:, -tl // hop:], hop, adaptive_key=0)
                    assert esr == SR
                    L = min(tl, eb.size(-1), ob.size(-1))
                    ob = ob.clone()
                    ob[:, -L:] = eb.reshape(eb.size(0), -1)[:, -L:]
                else:
                    eb, esr = self.enhancer.enhance(
                        ob, self.args.data.sampling_rate, fb, hop,
                        adaptive_key=0)
                    assert esr == SR
                    ob = eb.reshape(eb.size(0), -1)[:, :ob.size(-1)] \
                        if eb.dim() > 2 else eb
                outs = [ob[vi:vi + 1] for vi in range(len(self.voices))]
        return [out.squeeze() * gain for out, (_, _, gain)
                in zip(outs, self.voices)]
        # 每聲部分開回＝接縫各自對齊（混音 SOLA 對不齊兩個週期）

    def encode(self, audio):
        au = torch.from_numpy(audio).float()[None].to(self.device)
        return self.units_encoder.encode(au, SR, self.args.data.block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=f"{DDSP}/exp/combsub-harry-260730/model_30000.pt",
                    help="--solo 模式用的模型")
    ap.add_argument("--solo", action="store_true",
                    help="單聲部 identity（質感/延遲檢驗用）；預設＝和聲"
                         "（S=Soprano-3 +12、B=Bass-1 −12 跟著唱）")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--no-enhance", dest="enh", action="store_false")
    ap.add_argument("--in-name", default="USB PnP")
    ap.add_argument("--out-name", default="AI-Micro")
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--thr", type=float, default=-60.0,
                    help="音量門檻 dB（正式配方 -60；gui.py 的 -45 對喉麥"
                         "太高＝輸出被切成斷斷續續）")
    ap.add_argument("--key", type=int, default=0,
                    help="大調主音 pitch class（0=C…11=B）；diatonic 映射用")
    ap.add_argument("--no-brain", dest="brain", action="store_false",
                    help="關天使腦＝退回 v12 固定 diatonic 映射")
    ap.add_argument("--octave", type=int, default=12,
                    help="腦空間位移（respond2 同款；16:00 場＝12）")
    ap.add_argument("--block", type=float, default=0.10,
                    help="每塊秒數＝延遲主項；infer ms 必須小於它"
                         "（bench：1.0s 視窗 infer p95 39ms＝0.10 有 2.5x 餘裕）")
    ap.add_argument("--crossfade", type=float, default=0.04)
    ap.add_argument("--extra", type=float, default=1.0,
                    help="前貼上下文秒數（質感 vs 算力）")
    ap.add_argument("--sustain", type=float, default=0.0,
                    help="v15 樂句橋接 ms：≤此長度的 uv 洞天使撐母音唱過去"
                         "（E 版 sustain live 化；0＝關＝舊路徑）")
    ap.add_argument("--release", type=float, default=120.0,
                    help="超過 sustain 的真休止：淡出收尾 ms（取代硬 gate 剁）")
    ap.add_argument("--floor-db", type=float, default=0.0,
                    help="v21 音量地板 dBFS（0＝關＝舊路徑）：vol 低於此值的"
                         "幀強制視同 uv＝gate 可關、腦收休止。治喉麥氣音/摩擦"
                         "被 parselmouth 判有聲（diag4：gate 開啟幀 61% 是"
                         "mic 峰值<歌聲 1/10 的垃圾、f0 中位 562Hz＝怪音原料；"
                         "−36 實測殺 100% voiced 垃圾、誤傷 0%）")
    ap.add_argument("--dump", default="",
                    help="退場時寫 <dump>_{mic,angel}.wav＋_feats.npz"
                         "（逐幀 f0/uv/gate/橋/音量/音程；診斷用）")
    ap.add_argument("--enc-win", type=float, default=0.0,
                    help="v26 hubert 編碼窗秒數（0＝關＝與渲染窗同）：encode"
                         "吃更長的「過去」音訊＝左上下文買咬字、零延遲代價。"
                         "diag5 劑量反應：0.7s→0.524、1.5s→0.694、3s→0.788"
                         "（canon 0.813；成本 15→40ms）。建議 3.0。")
    ap.add_argument("--lookahead", type=int, default=0,
                    help="v25 前瞻幀數（路線圖①）：渲染/取用窗整體回移 N 幀"
                         "＝該區 hubert/EMA/音符多 N 幀右上下文（咬字準），"
                         "代價＝輸出 +N×11.6ms 延遲。0＝關＝原行為。"
                         "enh-tail 的省幅拿來買這個＝延遲不變咬字升級。")
    ap.add_argument("--rehearse", default="",
                    help="v24 排練譜模式（路線圖⑦論文級）：notes.json"
                         "（lead/upper/lower，mic 空間絕對音，tick=186ms 網格"
                         "＝scratchpad/make_score.py 從 dump 自產）。"
                         "ScoreTracker v3 對位他的 lead、天使唱譜線＝和聲由譜"
                         "保證、腦（含 token 吸附層）整層旁路。")
    ap.add_argument("--reh-rate", type=float, default=1.0,
                    help="排練對位速度先驗 r̂（live_v3 同名旋鈕；固定優於自估）")
    ap.add_argument("--enh-tail", action="store_true",
                    help="v23 enhancer 只算 SOLA 取用尾段＋0.1s 邊距"
                         "（−28ms 級）；接縫品質由 replay 指標＋耳測裁")
    ap.add_argument("--dump-stems", action="store_true",
                    help="dump 加每聲部 stem 波形（歸因診斷用；檔案大三倍）")
    ap.add_argument("--ema-alpha", type=float, default=0.12,
                    help="v12 units EMA 平滑係數（音符穩定時）；調高＝母音"
                         "跟得快/咬字銳、調低＝更釘住。0.12＝原值")
    ap.add_argument("--ema-streak", type=int, default=6,
                    help="幾幀穩定才開始 EMA 釘住（6≈70ms＝原值）")
    ap.add_argument("--agc", action="store_true",
                    help="v22 串流 AGC（direct_mouth input_agc 的因果版）：把"
                         "voiced 位準拉向 TRAIN_REF_RMS 0.0566 再餵模型、輸出"
                         "除回＝位準結構不變。治「位準掉出訓練分布＝轉換退化」"
                         "（咬字糊/沒力主嫌；07-30 量測 14.3→18.3dB 單調惡化）。"
                         "預設關＝舊路徑。")
    ap.add_argument("--replay", default="",
                    help="v22 離線重播：讀 wav 逐 block 走同一條 step/brain"
                         "路徑（不開音訊裝置；腦每 block 同步 drain＝與 live"
                         "同樣落後一塊）。配 --dump 產出可跟現場 dump 對照。"
                         "迭代品質不用重唱的地基。")
    a = ap.parse_args()

    print("loading models…", flush=True)
    ear = kt = None
    REH = None
    if a.rehearse:
        # v24：譜模式＝腦旁路。三張嘴同 brain 配置；音符線來自譜追蹤。
        import json
        from rehearse_ab import ScoreTracker
        d = json.load(open(a.rehearse))
        REH = {"upper": d["upper"], "lower": d["lower"],
               "tracker": ScoreTracker(d["lead"], mode="v3",
                                       rate0=a.reh_rate)}
        print(f"[rehearse] score {len(d['lead'])} ticks", flush=True)
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.8),
                  (f"{DDSP}/exp/combsub-girl/model_30000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]
    elif a.solo:
        voices = [(a.model, 0.0, 1.0)]
        dia_steps = [None]
        vmode = [None]
    elif a.brain:
        # v13 天使腦：EarV3 tick 流（本來就是 live 機器）＝girl 唱上聲部線、
        # Bass-1 唱下聲部線（indep/stab/legato 全部繼承）、S=+8ve 頂旋律。
        # 腦輸出換算成「相對他音高的音程」進快取＝音程恆定（D 哲學）。
        import respond2 as R2
        from live_v3 import EarV3
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.8),
                  (f"{DDSP}/exp/combsub-girl/model_30000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]        # sop 固定 +12；girl/bass 走腦
        ear = EarV3(indep=0.15, stab=1, key=0)
        ear.v2t.shift = a.octave
        ear.k_shift = 0        # keylock 交給 KeyTracker 外部維護（v14 直饋）
        kt = R2.KeyTracker()
    else:
        voices = [(f"{DDSP}/exp/combsub-m4-sop3/model_10000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/combsub-harry-260730/model_30000.pt",
                   -3.0, 0.9),
                  (f"{DDSP}/exp/combsub-m4-bass1/model_11000.pt", -12.0, 1.0)]
        dia_steps = [7, -2, -7]
        vmode = [None, None, None]
    svc = Svc(voices, enhance=a.enh)
    HOP = svc.args.data.block_size
    blk = max(1, round(a.block * SR / HOP)) * HOP     # 512 對齊＝網格不漂
    cf = int(a.crossfade * SR)
    sola_search = int(0.01 * SR)
    last_delay = int(0.02 * SR)
    LA = a.lookahead * HOP              # v25 前瞻（樣本）；0＝關
    input_frame = ((max(int(a.extra * SR),
                        blk + cf + sola_search + 2 * last_delay + LA)
                    // HOP + 1) * HOP)
    nb = blk // HOP
    buf = np.zeros(input_frame, dtype="float32")
    ENC_WIN = (max(int(a.enc_win * SR) // HOP * HOP, input_frame)
               if a.enc_win else 0)
    buf3 = np.zeros(ENC_WIN, dtype="float32") if ENC_WIN else None
    sola_bufs = [torch.zeros(cf, device=svc.device) for _ in svc.voices]
    fade_in = torch.sin(np.pi * torch.arange(0, 1, 1 / cf,
                                             device=svc.device) / 2) ** 2
    fade_out = 1 - fade_in
    stats = {"n": 0, "ms": [], "late": 0}
    st = {"uc": None, "fc": None, "offc": None, "tb": np.zeros(0),
          "cur": {"up": 12.0, "lo": -12.0}, "ktn": 0,
          "hole": 10 ** 9, "env": 0.0, "nbh": 0, "nbf": 0,
          "f0q": [], "hf0": 0.0, "ufifo": [], "inbr": False,
          "ivp": {"up": None, "lo": None}, "klock": False,
          "agc_g": 1.0, "agc_ema": None}
    accs = [0.0] * len(svc.voices)      # 每聲部絕對相位累加器（弧度）
    tracker = NoteTracker()             # 遲滯音符（v11 長音穩定）
    from collections import defaultdict
    dmp = defaultdict(list) if a.dump else None
    sus_f = int(round(a.sustain / 1000.0 * SR / HOP))   # 橋接上限（幀）
    rel_f = max(1, int(round(a.release / 1000.0 * SR / HOP)))
    ENV_DK = float(np.exp(-(HOP / SR) / 0.15))   # 音量慣性 τ≈150ms

    def bridge_feed(vol_np, uv, f0_np):
        """v18 樂句橋接（因果逐幀；演化史見 worklog §O/§P）。回 (gate, vol,
        bridged, f0_hold)，只餵新到幀＝結果可凍進網格（跨窗一致）。
        入洞＝聲帶判無聲 且 音量掉到包絡 1/4 以下（−12dB；擋滑窗尾端假
        uv＝v16 亂源）；洞的**持續**只看 uv——v17 的坑：長休止時包絡衰到
        底噪、相對判準失效＝gate 重開＝bass 拿內插假 f0 唱低頻嗡（review
        F1）。hold 哲學（v17「三個都亂」的教訓）：洞口幀已被子音污染
        （窗尾 f0 亂猜、units 髒）＝不能 hold 洞口——f0 取洞口前 ~45ms 的
        安全幀（4 幀 FIFO 最舊值＝免費的事後 lookahead），音量取包絡
        （起音瞬跟、釋放 τ≈150ms）。短洞（≤sus_f）gate 撐 1；真休止撐滿
        後 rel_f 幀餘弦淡出收尾。"""
        g = np.zeros(len(vol_np))
        v = np.copy(vol_np)
        b = np.zeros(len(vol_np), dtype=bool)
        vf0 = np.copy(f0_np)
        for j, x in enumerate(vol_np):
            st["env"] = max(float(x), st["env"] * ENV_DK)
            if not (uv[j] and (st["hole"] > 0 or x < st["env"] * 0.25)):
                st["hole"] = 0
                g[j] = 1.0
                st["f0q"] = (st["f0q"] + [float(f0_np[j])])[-4:]
            else:
                st["hole"] += 1
                if st["hole"] == 1:
                    st["hf0"] = st["f0q"][0] if st["f0q"] else float(f0_np[j])
                if st["hole"] <= sus_f:
                    g[j], v[j], b[j] = 1.0, st["env"], True
                    vf0[j] = st["hf0"]
                    st["nbf"] += 1
                    if st["hole"] == 1:
                        st["nbh"] += 1
                elif st["hole"] <= sus_f + rel_f:
                    r = (st["hole"] - sus_f) / rel_f
                    g[j] = float(np.cos(r * np.pi / 2) ** 2)
                    v[j], b[j] = st["env"], True
                    vf0[j] = st["hf0"]
        return g, v, b, vf0

    # 預熱（同視窗大小＋同 ratios 路徑＝MPS kernel 編好才開流）
    nfrm = input_frame // HOP + 1
    svc.infer(buf.astype("float64") + 1e-6, a.pitch, a.thr,
              ratios=[np.zeros(nfrm) if (s is not None or vm) else None
                      for s, vm in zip(dia_steps, vmode)])
    print(f"ready  block {blk/SR*1000:.0f}ms / window {input_frame/SR:.2f}s",
          flush=True)

    import threading
    in_q, out_q = [], []
    qlock = threading.Lock()
    ev = threading.Event()
    st2 = {"under": 0, "flags": 0, "die": False}

    def cb(indata, outdata, frames, tinfo, status):
        # v9：callback 只搬記憶體（μs 級）。推論在 worker——110ms 的 MPS
        # 工作放這裡是 v1–v8「斷斷續續」的真兇（CoreAudio 死線是硬的，
        # 平均達標沒用；status 旗標之前還被忽略＝假安心）。
        if status:
            st2["flags"] += 1
        with qlock:
            in_q.append(indata[:, 0].copy())
            have = sum(len(q) for q in out_q)
            outdata[:] = 0
            if have >= frames:
                need = frames
                col = []
                while need > 0:
                    q = out_q[0]
                    take = min(need, len(q))
                    col.append(q[:take])
                    if take == len(q):
                        out_q.pop(0)
                    else:
                        out_q[0] = q[take:]
                    need -= take
                y = np.concatenate(col)
                outdata[:len(y), 0] = y
                if outdata.shape[1] > 1:
                    outdata[:len(y), 1] = y
            else:
                st2["under"] += 1
        ev.set()

    def worker():
        t_pending = np.zeros(0, dtype="float32")
        while not st2["die"]:
            ev.wait(0.5)
            ev.clear()
            with qlock:
                if in_q:
                    t_pending = np.concatenate([t_pending] + in_q)
                    in_q.clear()
            while len(t_pending) >= blk:
                chunk, t_pending = t_pending[:blk], t_pending[blk:]
                step(chunk)

    bq, block = [], threading.Lock()
    TICK_FRAMES = 16                    # 16×512/44100 ≈ 186ms ≈ 原 tick 節奏

    def ear_note_tick(f_in):
        """v14 音符直饋＝ear.tick 減去 tracker.push/keylock（那是 +55ms 的
        全部成本；k_shift 由 KeyTracker 外部維護、f_in 來自凍結 f0）。
        反應式路徑逐行對齊 live_v3.EarV3.tick。
        v24 rehearse：譜追蹤取代腦——mic 空間絕對音直進直出，token/吸附/
        k_shift 整層不進場；他脫稿時天使＝譜錨（iv 對他實唱算）。"""
        if REH is not None:
            tr = REH["tracker"]
            tr.observe(None if f_in is None else int(f_in))
            if f_in is not None:
                p = tr.p
                for part, line in (("up", REH["upper"]),
                                   ("lo", REH["lower"])):
                    note = line[p] if 0 <= p < len(line) else None
                    if note is not None:
                        st["cur"][part] = float(int(note) - int(f_in))
            return
        from live_v3 import token_to_midi
        real = f_in                 # 他真唱的音（吸附/摺疊前；v18 音程參照）
        if f_in is not None and ear.k_shift:
            f_in = int(f_in) + ear.k_shift
        s = ear.v2t.token(f_in)
        toks = ear.brain.step(s)
        u, l = ear._to_notes(toks)
        ear.lead_prev = token_to_midi(s, ear.lead_prev)
        ear.lead_hist.append(ear.lead_prev)
        lead = ear.lead_prev
        offsp = ear.v2t.shift + (ear.k_shift or 0)
        if lead is not None:
            # v18（review F6）：參照＝他真唱的音。舊版 lead−offsp 是內部
            # 吸附/摺疊後表徵，C 大調吸附時差 1 半音（實測 40% tick 觸發
            # ＝bass 唱高半音）——respond2.py:419 同款血訓，串流版重踩。
            # v20：真唱參照要再減 token() 同款吸附差 d——腦全程在吸附空間
            # 寫和聲，iv 對「未吸附的真唱」算＝腦內音程 −d：bass 想同度變
            # −1 貼臉、girl 五度變三全音（diag1：P(bass=−1|吸附)=24% vs
            # 未吸附 4%；修正模擬 −1 佔比 11.3%→4.2%，girl 4/6→5/7）。
            if sus_f and real is not None:
                d_snap = 1 if (real + offsp) % 12 in (1, 3, 6, 8, 10) else 0
                mic = real - d_snap
            else:
                mic = lead - offsp
            for part, note in (("up", u), ("lo", l)):
                if note is None:
                    continue
                iv = float(note - mic)
                if not sus_f:
                    st["cur"][part] = iv
                    continue
                # v19 音程遲滯（diag1 實錘：換音 7 次/s、駐留 p50 12ms＝
                # 和聲線在抖＝「沒有和聲感」主因）：新音程連續兩 tick
                # 一致才換＝駐留下限 ~370ms，一次性的骰子不上線。
                if iv == st["cur"][part]:
                    st["ivp"][part] = None
                elif st["ivp"][part] == iv:
                    st["cur"][part] = iv
                    st["ivp"][part] = None
                else:
                    st["ivp"][part] = iv

    brain_acc = []

    def brain_drain():
        """把 bq 裡累積的凍結 f0 幀吃成 tick（live 由執行緒呼叫、
        replay 每 step 後同步呼叫＝同一份邏輯、腦落後一個 block 一致）。"""
        with block:
            if bq:
                brain_acc.extend(bq)
                bq.clear()
        while len(brain_acc) >= TICK_FRAMES:
            w = np.array(brain_acc[:TICK_FRAMES])
            del brain_acc[:TICK_FRAMES]
            v = w[w > 0]
            f_in = (int(round(float(np.median(
                69 + 12 * np.log2(v / 440.0)))))
                if len(v) >= TICK_FRAMES // 2 else None)
            try:
                ear_note_tick(f_in)
            except Exception as e:  # noqa: BLE001
                print("brain err:", e, flush=True)

    def brain_worker():
        # 腦執行緒吃凍結 f0 幀（不再碰音訊）＝成本趨近零。
        while not st2["die"]:
            time.sleep(0.02)
            brain_drain()

    def step(chunk):
        t0 = time.perf_counter()
        buf[:-blk] = buf[blk:]
        buf[-blk:] = chunk
        if ENC_WIN:
            buf3[:-blk] = buf3[blk:]
            buf3[-blk:] = chunk
        try:
            # f0 凍結網格 v11：到幀即凍（尾端估計 p95 4.3c 夠好）＋遲滯音符
            # →音程/穩定 streak 也進快取＝渲染與相位帳同一份精確值。
            xb = buf.astype("float64")
            if a.agc:
                xb = xb * st["agc_g"]      # v22：模型看拉平位準（輸出會除回）
            if sus_f:
                f0f, vol_t, mask, uvf = svc.prep(xb, a.thr, want_uv=True)
                if a.floor_db < 0:
                    # v21：地板下幀＝uv（下游 bridge_feed 的 gate 與腦 feed
                    # 的休止判定全走 uvf，單點生效）。門檻對原始位準定義，
                    # AGC 開著時 vol 已被 ×g，門檻同乘。
                    uvf = uvf | (vol_t[0, :, 0].cpu().numpy()
                                 < 10 ** (a.floor_db / 20.0) * st["agc_g"])
            else:
                f0f, vol_t, mask = svc.prep(xb, a.thr)
            if a.agc:
                # 增益更新（因果、一 block 延遲生效）：新幀活動段的原始位準
                # EMA（τ≈2s）→ g＝REF/ema，夾 [0.25,16]（input_agc 同界）。
                vn = vol_t[0, -nb:, 0].cpu().numpy() / st["agc_g"]
                act = vn > 10 ** (a.thr / 20.0)
                if act.any():
                    al = 1 - float(np.exp(-(blk / SR) / 2.0))
                    m = float(np.median(vn[act]))
                    st["agc_ema"] = m if st["agc_ema"] is None else \
                        (1 - al) * st["agc_ema"] + al * m
                    st["agc_g"] = float(np.clip(
                        0.0566 / max(st["agc_ema"], 1e-6), 0.25, 16.0))
            fc = st["fc"]
            if fc is None or len(fc) != len(f0f):
                if sus_f:
                    g, v, b, vf0 = bridge_feed(
                        vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                    st["gt"], st["vc"], st["br"] = g, v, b
                    st["fc"] = vf0      # 橋接幀凍安全 f0；有聲幀＝f0f 原值
                else:
                    st["fc"] = f0f
                notes, stks = tracker.feed(f0f)
                st["stk"] = stks
                st["offc"] = []
                for vi, s in enumerate(dia_steps):
                    if s is not None:
                        st["offc"].append(np.array(
                            [0.0 if n <= 0 else dia_from_note(n, s, a.key)
                             for n in notes]))
                    elif vmode[vi]:
                        st["offc"].append(
                            np.full(len(f0f), st["cur"][vmode[vi]]))
                    else:
                        st["offc"].append(None)
            else:
                # 絕對相位推進：離開視窗的 nb 幀 × 快取裡渲染用過的音程
                for vi, (_, semi, _) in enumerate(svc.voices):
                    if st["offc"][vi] is not None:
                        adv = float((fc[:nb] * 2 ** (
                            (a.pitch + st["offc"][vi][:nb]) / 12.0)).sum())
                    else:
                        adv = float(fc[:nb].sum()) \
                            * 2 ** ((a.pitch + semi) / 12.0)
                    accs[vi] = (accs[vi] + 2 * np.pi * adv * HOP / SR) \
                        % (2 * np.pi)
                if sus_f:
                    if st.get("gt") is None:    # 自癒（init 例外後不永久卡死）
                        g, v, b, vf0 = bridge_feed(
                            vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                        st["gt"], st["vc"], st["br"] = g, v, b
                        st["fc"] = vf0
                    else:
                        g, v, b, vf0 = bridge_feed(
                            vol_t[0, -nb:, 0].cpu().numpy(),
                            uvf[-nb:], f0f[-nb:])
                        st["gt"] = np.concatenate([st["gt"][nb:], g])
                        st["vc"] = np.concatenate([st["vc"][nb:], v])
                        st["br"] = np.concatenate([st["br"][nb:], b])
                        st["fc"] = np.concatenate([fc[nb:], vf0])
                else:
                    st["fc"] = np.concatenate([fc[nb:], f0f[-nb:]])
                if ear is not None or REH is not None:
                    with block:
                        # v18（review F7）：uv 幀送 0＝腦收得到休止符
                        # （舊版餵內插 f0＝REST 0/484 tick、token 流離開
                        # 訓練分佈）。brain_worker 的 w>0 篩選現成處理。
                        bq.extend((np.where(uvf[-nb:], 0.0, f0f[-nb:])
                                   if sus_f else f0f[-nb:]).tolist())
                notes, stks = tracker.feed(f0f[-nb:])
                st["stk"] = np.concatenate([st["stk"][nb:], stks])
                for vi, s in enumerate(dia_steps):
                    if s is not None:
                        new = np.array(
                            [0.0 if n <= 0 else dia_from_note(n, s, a.key)
                             for n in notes])
                    elif vmode[vi]:
                        tgt = st["cur"][vmode[vi]]
                        new = np.full(nb, tgt)
                        pv = float(st["offc"][vi][-1])
                        if sus_f and pv != tgt:
                            # v18（review F8）：換音 ~35ms 滑進新音程，
                            # 不整塊硬跳（離線 target_f0 的 porta 精神）。
                            k3 = min(nb, 3)
                            new[:k3] = np.linspace(pv, tgt, k3 + 2)[1:k3 + 1]
                    else:
                        continue
                    st["offc"][vi] = np.concatenate(
                        [st["offc"][vi][nb:], new])
                if kt is not None:
                    st["ktn"] += 1
                    if st["ktn"] % 10 == 0:        # ~2s 推一次 key 追蹤
                        kt.push(st["fc"])
                        rb = kt.best()
                        if rb is not None and rb[2] >= 0.015 and \
                                not (sus_f and st["klock"]):
                            # v19：k_shift 一鎖不再動（respond2 同款紀律；
                            # 持續重估＝一飄全部音程平移＝腦線抖動子嫌）
                            ear.k_shift = rb[0]
                            st["klock"] = True
            # units 凍結快取（v2）＋ v12 長音平滑：他唱「嗚」時窗尾 hubert
            # 逐幀亂猜元音＝輸出母音遊走。音符穩定（streak≥6≈70ms）→ units
            # 走 EMA α=0.12（τ≈90ms）釘住；換音/子音（streak 歸零）→ α=1
            # 立即跟上＝起音不糊。平滑在絕對網格上因果進行＝跨窗一致。
            if ENC_WIN:
                # v26：encode 吃 3s 過去（左上下文＝免費咬字），只取渲染窗
                # 對應的尾端幀＝下游 uc/usm/infer 完全不感知差異。
                xbe = buf3.astype("float64")
                if a.agc:
                    xbe = xbe * st["agc_g"]
                fresh = svc.encode(xbe)[:, -(input_frame // HOP + 1):]
            else:
                fresh = svc.encode(xb)
            uc = st["uc"]
            if uc is None or uc.size(1) != fresh.size(1):
                st["uc"] = fresh
                st["usm"] = fresh.clone()
            else:
                keep = uc[:, nb:]
                m = 2 * nb
                st["uc"] = torch.cat(
                    [keep[:, :fresh.size(1) - m], fresh[:, -m:]], 1)
                us = st["usm"][:, nb:]
                # v23：EMA 逐幀迴圈在 CPU numpy 做（256 維×nb 幀＝CPU 零成
                # 本；原本每幀 2-3 次 MPS kernel 發射＝step 隱形成本大宗）。
                # 一次下載、算完一次上傳，數值同 float32 lerp。
                fr = fresh[:, -nb:].detach().cpu().numpy()
                prev = us[:, -1].detach().cpu().numpy()
                news = np.empty_like(fr)
                for j in range(nb):
                    if sus_f and st["br"][-nb + j]:
                        if not st["inbr"]:
                            # v18 洞口回捲：洞口幀已被子音污染，凍的母音
                            # 取洞前 ~45ms 的 FIFO 最舊值（同 f0 hold 哲學）
                            if st["ufifo"]:
                                prev = st["ufifo"][0]
                            st["inbr"] = True
                        al = 0.0
                    else:
                        st["inbr"] = False
                        al = a.ema_alpha \
                            if st["stk"][-nb + j] >= a.ema_streak else 1.0
                    prev = (1 - al) * prev + al * fr[:, j]
                    if sus_f and not st["inbr"]:
                        st["ufifo"] = (st["ufifo"] + [prev])[-4:]
                    news[:, j] = prev
                st["usm"] = torch.cat(
                    [us, torch.from_numpy(news).to(us)], 1)
            if sus_f:
                # v18：橋接 gate 直接取代原 mask——喉麥底噪 −55dB 高於
                # −60 門檻＝原 mask 恆開（review F1：20 分鐘 3 個洞），
                # 取 max 等於沒 gate、真休止關不掉＝bass 低頻嗡。橋接
                # gate 自含「有聲＝開」。音量走凍結快取（洞內包絡 hold）。
                gu = upsample(torch.from_numpy(st["gt"]).float().to(
                    svc.device)[None, :, None], HOP).squeeze(-1)
                k2 = min(mask.size(1), gu.size(1))
                mask = gu[:, :k2]
                vol_t = torch.from_numpy(st["vc"]).float().to(
                    svc.device)[None, :, None]
            aus = svc.infer(xb, a.pitch, a.thr,
                            units_override=st["usm"],
                            feats=(st["fc"], vol_t, mask), phases=accs,
                            ratios=st["offc"],
                            enh_tail=(blk + cf + sola_search + last_delay
                                      + LA + int(0.1 * SR))
                            if a.enh_tail else 0)
            y = None
            stems = [] if (dmp is not None and a.dump_stems) else None
            for vi, au in enumerate(aus):    # 每聲部各自 SOLA＋接縫，拼完才混
                # 全特徵凍結後重疊區 corr 0.96（儀器實證）＝縫只剩相位位移
                # ＝SOLA 的本職；內容一致 → shift 穩定不再抖。
                tw = au[-blk - cf - sola_search - last_delay - LA:
                        -last_delay - LA if last_delay + LA else None]
                ci = tw[None, None, :cf + sola_search]
                num = F.conv1d(ci, sola_bufs[vi][None, None, :])
                den = torch.sqrt(F.conv1d(ci ** 2, torch.ones(
                    1, 1, cf, device=svc.device)) + 1e-8)
                shift = int(torch.argmax(num[0, 0] / den[0, 0]))
                tw = tw[shift: shift + blk + cf].clone()
                tw[:cf] = phase_vocoder(sola_bufs[vi], tw[:cf],
                                        fade_out, fade_in)
                sola_bufs[vi] = tw[-cf:]
                w = tw[:-cf].cpu().numpy()
                if stems is not None:
                    stems.append(w.copy())
                y = w if y is None else y[:len(w)] + w[:len(y)]
            if a.agc:
                y = y / st["agc_g"]        # 位準結構還原（input_agc 同哲學）
            y = np.clip(y * a.gain, -1.0, 1.0).astype("float32")
            with qlock:
                out_q.append(y)
            if dmp is not None:
                dmp["mic"].append(chunk.copy())
                dmp["out"].append(y.copy())
                if stems is not None:
                    for vi, w2 in enumerate(stems):
                        dmp[f"stem{vi}"].append(w2)
                dmp["f0"].append(st["fc"][-nb:].copy())
                dmp["f0raw"].append(f0f[-nb:].copy())
                dmp["stk"].append(st["stk"][-nb:].copy())
                dmp["vol"].append(
                    vol_t[0, -nb:, 0].detach().cpu().numpy().copy())
                if sus_f and st.get("gt") is not None:
                    dmp["uv"].append(uvf[-nb:].astype(float).copy())
                    for kk in ("gt", "vc", "br"):
                        dmp[kk].append(
                            np.asarray(st[kk][-nb:], dtype=float).copy())
                for vi in range(len(svc.voices)):
                    if st["offc"][vi] is not None:
                        dmp[f"off{vi}"].append(st["offc"][vi][-nb:].copy())
                if a.agc:
                    dmp["agc"].append(np.full(nb, st["agc_g"]))
        except Exception as e:  # noqa: BLE001
            print("infer err:", e, flush=True)
        ms = (time.perf_counter() - t0) * 1000
        stats["n"] += 1
        stats["ms"] = stats["ms"][-40:] + [ms]
        if ms > blk / SR * 1000:
            stats["late"] += 1
        if stats["n"] % 20 == 0:
            m = np.array(stats["ms"][-20:])
            print(f"infer p50 {np.median(m):.0f}ms p95 "
                  f"{np.percentile(m, 95):.0f}ms / block {blk/SR*1000:.0f}ms"
                  f" | late {stats['late']} under {st2['under']}"
                  f" flags {st2['flags']}"
                  + (f" bridge {st['nbh']}holes/{st['nbf']}frames" if sus_f else ""),
                  flush=True)

    def flush_dump():
        """dump 落檔（退場＋每 10s 週期性；行程被硬殺最多丟 10s）。"""
        import soundfile as sf
        sf.write(a.dump + "_mic.wav", np.concatenate(list(dmp["mic"])), SR)
        sf.write(a.dump + "_angel.wav", np.concatenate(list(dmp["out"])), SR)
        np.savez(a.dump + "_feats.npz",
                 **{k: np.concatenate(list(v)) for k, v in list(dmp.items())
                    if k not in ("mic", "out") and len(v)})

    if a.replay:
        import soundfile as sf
        x, sr_in = sf.read(a.replay, dtype="float32", always_2d=True)
        assert sr_in == SR, f"replay 檔要 {SR}Hz，拿到 {sr_in}"
        x = x[:, 0]
        nblk = (len(x) - blk) // blk + 1
        print(f"replay {a.replay}  {len(x)/SR:.0f}s / {nblk} blocks",
              flush=True)
        t0 = time.perf_counter()
        for i in range(nblk):
            step(x[i * blk:(i + 1) * blk])
            with qlock:
                out_q.clear()          # 無播放端，別讓佇列吃記憶體
            if ear is not None or REH is not None:
                brain_drain()
        el = time.perf_counter() - t0
        print(f"replay done {el:.0f}s（RTF {el/(nblk*blk/SR):.2f}）",
              flush=True)
        if dmp is not None and dmp["mic"]:
            flush_dump()
            print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz", flush=True)
        return

    print(f"stream on in={a.in_name!r} out={a.out_name!r} pitch {a.pitch:+.0f}"
          f" enhancer {'on' if a.enh else 'off'} (Ctrl-C to stop)", flush=True)
    wt = threading.Thread(target=worker, daemon=True)
    wt.start()
    if ear is not None or REH is not None:
        threading.Thread(target=brain_worker, daemon=True).start()

    import signal

    def _term(*_):
        raise KeyboardInterrupt      # nohup 背景行程會擋 SIGINT；TERM 也走 dump 路

    signal.signal(signal.SIGTERM, _term)
    with sd.Stream(samplerate=SR, blocksize=blk, channels=(1, 2),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency="high", callback=cb):
        try:
            tick_s = 0
            while True:
                time.sleep(1)
                tick_s += 1
                if dmp is not None and tick_s % 10 == 0 and dmp["mic"]:
                    flush_dump()
        except KeyboardInterrupt:
            st2["die"] = True
            if dmp is not None and dmp["mic"]:
                time.sleep(0.3)          # 等 worker 收尾，避免寫到半筆
                flush_dump()
                print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz",
                      flush=True)
            print("\nbye")


if __name__ == "__main__":
    main()
