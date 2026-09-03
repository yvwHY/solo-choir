"""spike_stream6.py — reflow 嘴版滑窗串流（08-08；spike_stream v26 的 6.x 移植）

spike_stream（CombSub＋enhancer）的全套串流機制（滑窗/凍結網格/SOLA/
腦直饋 v19/v20/橋接/floor/enc-win）原封保留，只換嘴：
  - 三張 reflow 嘴（Rectified-Flow＋NSF-HiFiGAN；08-07/08 訓的
    reflow-{harry,girl,bass1,sop3}）＝擬真天花板 3.0-3.5 級
  - 無 enhancer（reflow 內含 vocoder）；無 initial_phase 相位帳
    （NSF 相位不可控＝接縫全靠 SOLA＋phase_vocoder，gui_reflow 同款）
  - ODE 取樣 --step（2-4 步即飽和，08-07 掃描）

Run (6x venv！):
  260724_ddsp_svc_6x/venv/bin/python spike_stream6.py --sustain 250 \
      --block 0.17 --extra 0.7 --floor-db -36 --ema-alpha 0.3 --enc-win 3
Ctrl-C 結束；每塊印 infer ms（>block ms＝會斷音）。
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP6X as _DDSP6X  # noqa: E402
sys.path.insert(0, str(_DDSP6X))
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from ddsp.vocoder import F0_Extractor, Volume_Extractor, \
    Units_Encoder  # noqa: E402
from ddsp.core import upsample  # noqa: E402
from reflow.vocoder import load_model_vocoder  # noqa: E402

SR = 44100
DDSP = str(_DDSP6X)
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

    NGRID_BANK = 16384      # 凍結噪聲庫長度（幀）≈190s 循環

    def __init__(self, voices, device="mps", step=4, t_start=0.7,
                 f0_extractor="parselmouth", noise_grid=False):
        self.device = device
        self.voc_tail = 0                    # v32：只 vocode 尾段幾幀；0＝關
        self.f0_extractor = f0_extractor
        self.step = step                     # ODE 取樣步數（2-4 即飽和）
        self.t_start = t_start               # ODE 起點（見 --t-start）
        self.voices = []
        self.vocoder = None
        import os
        cwd = os.getcwd()
        os.chdir(DDSP)          # config.yaml 裡 vocoder.ckpt 是相對路徑
        try:
            self.spks = []      # v35 per-voice spk_id（多人模型；預設 1）
            for v in voices:
                path, semi, gain = v[:3]
                spk = v[3] if len(v) > 3 else 1
                model, vocoder, args = load_model_vocoder(path, device=device)
                if self.vocoder is None:
                    self.vocoder = vocoder   # 同一顆 NSF-HiFiGAN，共用
                self.voices.append((model, semi, gain))
                self.spks.append(
                    torch.LongTensor([[spk]]).to(device))
        finally:
            os.chdir(cwd)
        # vocoder 惰性載入（forward 才 torch.load）存的是相對路徑＝絕對化
        vp = getattr(self.vocoder, "vocoder", self.vocoder)
        if hasattr(vp, "model_path") and not os.path.isabs(vp.model_path):
            vp.model_path = os.path.join(DDSP, vp.model_path)
        self.args = args                      # 同管線＝共用 data 參數
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            f"{DDSP}/{self.args.data.encoder_ckpt}",
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size, device=device)
        self.enhancer = None                 # reflow 內含 vocoder＝不需要
        self.vol_ex = Volume_Extractor(
            self.args.data.block_size,
            int(getattr(self.args.data, "volume_smooth_size", 1024)))
        self.spk = torch.LongTensor([[1]]).to(device)
        # v33 凍結噪聲網格：reflow 起始噪聲改按「絕對幀位」查表（每聲部一
        # 份、固定種子），重疊視窗重渲同一幀＝同一份噪聲 → 消掉每窗獨立
        # randn 的音高路徑抖動（08-10 量測：live std 31c vs 正典 9c）。
        # None＝關＝randn 舊行為（infer 不帶 frame0 時亦同）。
        self.ngrid = None
        if noise_grid:
            g = torch.Generator().manual_seed(20260810)
            M = self.vocoder.dimension      # Vocoder.__init__ 存的是 int 屬性
            self.ngrid = [torch.randn(self.NGRID_BANK, M, generator=g)
                          .to(device) for _ in self.voices]
        # v29：F0_Extractor 原本每個 block 重建一次（便宜的 parselmouth 沒
        # 人發現）；rmvpe 是 181MB 的網路，必須只建一次。權重路徑是相對的
        # ＝建構時 cwd 要在 DDSP。
        if self.f0_extractor == "rmvpe":
            # vendor 的 RMVPE 用 torch.load 不帶 map_location，ckpt 存自
            # CUDA 機器＝在 mac 上直接炸。預先建好塞進 F0_KERNEL 快取
            # （F0_Extractor 只在 key 不存在時才自己建），避免改 vendor repo。
            from ddsp.vocoder import F0_KERNEL
            if "rmvpe" not in F0_KERNEL:
                from encoder.rmvpe import RMVPE
                # 08-08：pretrain/rmvpe/model.pt 是舊版架構（無 unet.tf.*），
                # vendor 的 E2E0 要 tf 那 50 層 → load_state_dict(strict=False)
                # 會靜默留下隨機權重＝f0 是垃圾。寧可大聲掛掉。
                _sd = torch.load(f"{DDSP}/pretrain/rmvpe/model.pt",
                                 map_location="cpu")
                if not any(".tf." in k for k in _sd):
                    raise SystemExit(
                        "pretrain/rmvpe/model.pt 與此 vendor 的 RMVPE 架構"
                        "不符（缺 unet.tf.*，50 層會是隨機權重）。"
                        "要用 rmvpe 得換成對應版本的權重；"
                        "現階段請用 --f0 parselmouth 或 fcpe。")
                _tl = torch.load

                def _shim(*ar, **kw):
                    # 兩處不合：ckpt 存自 CUDA（要 map_location），且這份
                    # 權重是 RVC 格式的 raw state_dict、vendor 卻取 ckpt['model']
                    o = _tl(*ar, **{**kw, "map_location": "cpu"})
                    return o if isinstance(o, dict) and "model" in o \
                        else {"model": o}

                torch.load = _shim
                try:
                    F0_KERNEL["rmvpe"] = RMVPE(
                        f"{DDSP}/pretrain/rmvpe/model.pt", hop_length=160)
                finally:
                    torch.load = _tl
        if self.f0_extractor == "fcpe":
            # vendor 寫死 'cuda' if available else 'cpu'＝在 mac 上落到 CPU
            # （RTF 2.13＝不可即時）。同樣預先塞快取，改推到 MPS。
            from ddsp.vocoder import F0_KERNEL
            if "fcpe" not in F0_KERNEL:
                from torchfcpe import spawn_bundled_infer_model
                F0_KERNEL["fcpe"] = spawn_bundled_infer_model(device=device)
        cwd = os.getcwd()
        os.chdir(DDSP)
        try:
            self.pe = F0_Extractor(self.f0_extractor, SR,
                                   self.args.data.block_size, 65.0, 800.0)
            if self.f0_extractor == "fcpe":
                self.pe.device_fcpe = device
        finally:
            os.chdir(cwd)

    def prep(self, audio, threhold=-60.0, want_uv=False):
        """回 (f0_np, vol_t, mask[, uv])：f0 留 numpy 給呼叫端上凍結網格。
        want_uv＝一併回傳 parselmouth 真 voicing（f0==0 幀；v16 橋接判準：
        喉麥的子音音量掉不到 -60dB 門檻下，聲帶判定才是 E 版 vm 的正身）。
        插值段復刻 vocoder.py:142-146＝f0 與 uv_interp=True 逐 byte 同。"""
        hop = self.args.data.block_size
        pe = self.pe                       # v29：建一次（原本每 block 重建）
        # 神經式抽取器（fcpe/crepe/rmvpe）的重取樣核是 float32；parselmouth
        # 維持吃 float64＝預設路徑逐位元不變。
        au = audio if self.f0_extractor in ("parselmouth", "dio", "harvest") \
            else audio.astype("float32")
        if want_uv:
            f0_np = pe.extract(au, uv_interp=False, device=self.device)
            uv = f0_np == 0
            if len(f0_np[~uv]) > 0:
                f0_np[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0],
                                      f0_np[~uv])
            f0_np[f0_np < 65.0] = 65.0
        else:
            f0_np = pe.extract(au, uv_interp=True, device=self.device)
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
              enh_tail=0, frame0=None):
        """每聲部各自輸出：共同 units/f0/vol、各自移調＋增益。
        feats＝(f0_np, vol_t, mask) 外部給＝不重算（凍結網格用）。
        phases＝每聲部絕對相位（弧度）＝comb 源跨窗同相（v6 跳針正解）。
        ratios＝每聲部逐幀半音偏移陣列（diatonic）；None＝用固定 semi。
        frame0＝本視窗第一幀的絕對幀位（v33 凍結噪聲網格查表用）；
        None 或 ngrid 未建＝randn 舊行為。"""
        hop = self.args.data.block_size
        f0_np, vol_t, mask = feats if feats is not None \
            else self.prep(audio, threhold)
        f0 = torch.from_numpy(f0_np).float().to(self.device)[None, :, None]
        units = units_override
        if units is None:
            units = self.encode(audio)
        n = min(units.size(1), f0.size(1), vol_t.size(1))
        # reflow 嘴：phases/enh_tail 參數收下但不用（無相位帳、無 enhancer）
        ws = []
        with torch.no_grad():
            for vi, (model, semi, gain) in enumerate(self.voices):
                if ratios is not None and ratios[vi] is not None:
                    r = torch.from_numpy(
                        ratios[vi][:n]).float().to(self.device)[None, :, None]
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + r) / 12.0)
                else:
                    fv = f0[:, :n] * 2 ** ((pitch_adjust + semi) / 12.0)
                # v32：mel 與 vocoder 拆開呼叫。voc_tail=0 時等價於原本的
                # return_wav=True（模型內部就是 vocoder.infer(mel, f0[-T:])）
                # ＝Regime A 可證；>0 時只 vocode 尾段＋邊距。
                # 視窗 0.71s 但 SOLA 只取尾巴 0.31s＝59% 的 vocoder 工作被丟。
                # 成本 ≈ 3.8ms + 0.29ms×幀（實測，scratchpad/bench_vocoder_tail）。
                # 非逐位元：NSF 靠對 f0 積分產生激發相位，起點一換相位就換
                # （docstring 的「NSF 相位不可控」）＝Regime B，耳測裁。
                nz = None
                if self.ngrid is not None and frame0 is not None:
                    # 給 8 幀餘裕（mel 幀數可能比 n 多），reflow 端裁到實際 T
                    idx = (frame0 + torch.arange(n + 8)) % self.NGRID_BANK
                    # [T,M] → [1,1,M,T]（reflow 的 x 形狀）
                    nz = self.ngrid[vi][idx].t()[None, None]
                mel = model(units[:, :n], fv, vol_t[:, :n],
                            spk_id=self.spks[vi], vocoder=self.vocoder,
                            infer=True, return_wav=False,
                            infer_step=self.step, method="euler",
                            t_start=self.t_start, use_tqdm=False,
                            init_noise=nz)
                f0m = fv[:, -mel.size(1):]
                if self.voc_tail:
                    kf = min(mel.size(1), self.voc_tail)
                    out = self.vocoder.infer(
                        mel[:, -kf:], f0m[:, -kf:]).reshape(1, -1)
                    # 前面補零回整窗長＝下游一切索引（mask、SOLA 的負向切片）
                    # 完全不用改；補的區段本來就會被尾段切片丟掉。
                    pad = mel.size(1) * hop - out.size(1)
                    if pad > 0:
                        out = F.pad(out, (pad, 0))
                else:
                    out = self.vocoder.infer(mel, f0m).reshape(1, -1)
                k = min(out.size(1), mask.size(1), n * hop)
                out = out[:, :k] * mask[:, :k]
                ws.append(out.squeeze() * gain)
        return ws
        # 每聲部分開回＝接縫各自對齊（混音 SOLA 對不齊兩個週期）

    def encode(self, audio):
        au = torch.from_numpy(audio).float()[None].to(self.device)
        return self.units_encoder.encode(au, SR, self.args.data.block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default=f"{DDSP}/exp/reflow-harry-run1/model_14000.pt",
                    help="--solo 模式用的模型")
    ap.add_argument("--solo", action="store_true",
                    help="單聲部 identity（質感/延遲檢驗用）；預設＝和聲"
                         "（S=Soprano-3 +12、B=Bass-1 −12 跟著唱）")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--step", type=int, default=4,
                    help="reflow ODE 取樣步數（08-07 掃描：2 步 3.27、"
                         "4 步飽和 3.30；步數↑＝算力↑）")
    ap.add_argument("--t-start", type=float, default=0.7,
                    help="reflow ODE 起點（0-1；預設 0.7＝08-08 之前寫死的值）"
                         "。reflow.py:80 起始 mel＝t_start×DDSP mel＋"
                         "(1−t_start)×高斯雜訊，dt=(1−t_start)/step。"
                         "調高＝注入雜訊少、更貼 DDSP 底；調低＝更靠生成端。"
                         "三張嘴 config 都是 0.0＋50 步（離線正典的工作點）"
                         "＝串流與正典在這一軸上本來就不同。")
    ap.add_argument("--f0", default="parselmouth",
                    choices=["parselmouth", "rmvpe", "fcpe", "harvest",
                             "crepe", "dio"],
                    help="聲帶／音高偵測器（預設 parselmouth＝原行為）。"
                         "08-08 診斷：整套 sustain/bridge/decay 都在替不可靠"
                         "的 voicing 擦屁股——diag12 有 23 次橋接是「他全音量"
                         "在唱、parselmouth 連續 174ms 判無聲」。rmvpe 權重"
                         "已在 pretrain/rmvpe，對歌聲遠比自相關法穩。")
    ap.add_argument("--seed", type=int, default=None,
                    help="固定 reflow 取樣雜訊（reflow.py:80 的 randn 沒設種子"
                         "＝同輸入同旗標兩次跑不一樣）。離線 A/B 不設種子就"
                         "沒有解析度：實測 null 介入也能給出 ±6% 的差。")
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
    ap.add_argument("--sop-db", type=float, default=0.0,
                    help="S 聲部增益 dB（-99＝靜音）：08-08 診斷 sop3 20000 步"
                         "唱歌時高低頻比 0.016＝girl/bass 的 3 倍（氣音"
                         "「竊竊私語」源頭；訓練不足＝710 clips 只跑 20000 步）。"
                         "續訓期間可用 -99 先關掉聽三缺一。")
    ap.add_argument("--bridge-decay", type=float, default=0.0,
                    help="v27 橋接衰減 τ ms（0＝關＝舊行為）：撐音期 gate 走"
                         "exp(-t/τ)＝子音不斷音但不拖低音量氣音尾巴"
                         "（治「每段結束後吸口水」；建議 80）")
    ap.add_argument("--bridge-hold", type=float, default=0.0,
                    help="v29 撐音凍結門檻（相對包絡的比例；0＝關＝舊行為）："
                         "洞內音量仍高於 env×此值時，衰減／撐音預算的時鐘"
                         "不前進＝gate 維持不掉。治「唱到一半被切」——診斷"
                         "（08-08 diag12）：23 次橋接有 13 次洞內音量已回到"
                         "包絡 1/4 以上，但 v17 的洞持續只看 uv、出不去，"
                         "gate 一路衰減到零。不重開 gate（那會放內插假 f0 "
                         "進來＝v17 要防的 bass 低頻嗡），只是不再往下掉。"
                         "建議 0.5。")
    ap.add_argument("--units-mature", type=int, default=0,
                    help="v36 units 轉換幀成熟刷新：最近 M 幀裡 α==1 的轉換"
                         "幀每窗用本窗 fresh（更多右上下文）覆寫；釘住幀不碰"
                         "（0＝關＝舊行為）。建議 21-42（1-2 塊）。")
    ap.add_argument("--aah", action="store_true",
                    help="v37 天使唱「啊」不搬詞（08-11 Harry）：units 換 "
                         "take17 母音循環（絕對幀網格對齊），f0/音量/和聲比"
                         "照舊即時跟他；encode＋units 凍結鏈旁路＝RTF 還降。"
                         "關＝逐位元舊行為。")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="v37 慢層織體（第 0 階 H）：up/lo 天使改唱樂句級"
                         "和弦墊——釘在「換和弦時的錨音＋音程」的絕對音高，"
                         "不跟他的旋律線；和弦至少駐留這麼多秒才准換。"
                         "sop 維持 +12 跟旋律（影子聲部）。0＝關＝舊行為。")
    ap.add_argument("--swell", type=float, default=0.0,
                    help="v37 軟起音（悠遠層）：音量包絡上升限速 τ（ms，"
                         "合唱式 swell；下降不動＝收尾自然）。絕對幀網格"
                         "快取＝跨窗一致。0＝關＝舊行為。建議 120–200。")
    ap.add_argument("--wet", type=float, default=0.0,
                    help="v37 悠遠層：天使進空間（reverb.py 合成 IR、串流 "
                         "overlap-add 卷積）送出量。0＝關＝舊行為。"
                         "建議 0.3–0.4（respond2 耳測 0.20 是近距語境）。")
    ap.add_argument("--seam-amp", action="store_true",
                    help="v34 接縫改純振幅淡接（預設關＝phase_vocoder）。"
                         "搭 --sola-cont 用；詳接縫段註解。")
    ap.add_argument("--f0-mature", type=int, default=0,
                    help="v34 延遲凍結：最近 M 幀的 f0 每窗用最新估計刷新"
                         "（0＝關＝到達即凍＝舊行為）。窗尾估計缺右上下文，"
                         "換音段偏 37.6c；搭 --lookahead M（幀晚 M 幀播出）"
                         "播出時估計已成熟。代價＝LA×11.6ms 延遲。")
    ap.add_argument("--sola-cont", type=float, default=0.0,
                    help="v34 SOLA shift 連續性偏置 λ（0＝關＝舊行為）。"
                         "score×(1−λ·|s−prev|/search)＝選離上一塊最近的夠好"
                         "峰。治「音不穩」：shift 在相隔一週期的相關峰間亂跳"
                         "（p90 7ms），換音段時基抖動×斜率＝陡段偏差 50-74c。"
                         "建議 0.2-0.4；太大會鎖死在舊 shift、內容真變時縫錯。")
    ap.add_argument("--noise-grid", action="store_true",
                    help="v33 凍結噪聲網格（預設關＝舊行為）。reflow 起始噪聲"
                         "改按絕對幀位查固定種子噪聲庫（每聲部一份、190s 循環）"
                         "＝重疊視窗重渲同一幀用同一份噪聲。治「音不穩」：live "
                         "每窗獨立 randn ＝同一長音每 244ms 換一條音高路徑"
                         "（08-10 量測：live 對命令 f0 偏差 std 31c/12%>30c，"
                         "離線正典同嘴只有 9c/1%）。Regime B（改變 live 輸出）"
                         "；預設關時 randn 路徑逐位元不動＝Regime A 可證。")
    ap.add_argument("--voc-tail", type=int, default=0,
                    help="v32 只 vocode 尾段＋此邊距（幀；0＝關＝舊行為）。"
                         "視窗 0.71s 但 SOLA 只取尾巴 0.31s ＝ 59% 的 vocoder "
                         "工作被丟掉，而 vocoder 是全系統最貴的一段（整塊 "
                         "55%）。台架估三張嘴共省 ~20ms/塊（−13%）。**非逐位"
                         "元**：NSF 的激發相位是對 f0 積分來的，起點一換相位"
                         "就換（邊距掃描證實差值不隨邊距下降）＝Regime B，要"
                         "耳測。但相鄰塊的相位 origin 本來就不一致（SOLA＋"
                         "phase_vocoder 的本職就是縫這個），不是新的問題類別。"
                         "建議 4；聽接縫有沒有變粗糙。")
    ap.add_argument("--drop", type=int, nargs="*", default=[],
                    help="拿掉第 N 個聲部（0=sop 1=girl/中 2=bass；空＝不掉＝"
                         "舊行為）。跟 --sop-db 的差別：那個只把增益歸零、嘴"
                         "照樣推論；這個是真的不載、不算。實測每張嘴 ~41ms"
                         "（vocoder 佔其中 28ms）＝延遲最大的單一槓桿。")
    ap.add_argument("--prof", action="store_true",
                    help="分段計時（0＝關＝舊行為）：把每塊的 ms 拆成 prep／"
                         "bridge+brain／encode／units／infer／sola 六段，退場"
                         "時印預算表。MPS 是非同步佇列，所以每段都插 "
                         "mps.synchronize() 才量得到真值——代價是本身會讓總時"
                         "間略增，**--prof 的絕對 RTF 不可跟非 prof 跑比**，"
                         "只看分項佔比。")
    ap.add_argument("--bridge-units", type=float, default=0.0,
                    help="v30 橋接期 units 解凍門檻（相對包絡的比例；0＝關＝"
                         "舊行為）：洞內音量仍高於 env×此值時，units 不凍結、"
                         "照常走 EMA/streak＝子音跟得上。治「ㄔ／t 音發不出"
                         "來」——診斷（08-08）：--bridge-hold 把子音落在 units "
                         "凍結區的比例從 20.1% 推到 29.3%，且讓它們更大聲。"
                         "與 --bridge-hold 對 gate 的處理對稱（那個管 gate 不"
                         "掉，這個管 units 不凍）。順帶把 v18 洞口回捲的觸發"
                         "點從假洞口（uv 誤判）移到真正的音量下墜。建議 0.5。")
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

    print("載入模型…", flush=True)
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
        print(f"[rehearse] 譜 {len(d['lead'])} tick", flush=True)
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.8),
                  # v35（08-11）：girl → M4Singer Alto-6（三人合訓 40k、
                  # spk2；replay 台架 vs girl＝霧乾淨 8-10dB、音準/咬字持平
                  # ＋Harry 三嗓耳選 Alto-6）。girl 20000 留檔可回退。
                  (f"{DDSP}/exp/reflow-alto3/model_40000.pt", 12.0, 0.85, 2),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
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
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.8),
                  # v35（08-11）：girl → M4Singer Alto-6（三人合訓 40k、
                  # spk2；replay 台架 vs girl＝霧乾淨 8-10dB、音準/咬字持平
                  # ＋Harry 三嗓耳選 Alto-6）。girl 20000 留檔可回退。
                  (f"{DDSP}/exp/reflow-alto3/model_40000.pt", 12.0, 0.85, 2),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
        dia_steps = [None, None, None]
        vmode = [None, "up", "lo"]        # sop 固定 +12；girl/bass 走腦
        ear = EarV3(indep=0.15, stab=1, key=0)
        ear.v2t.shift = a.octave
        ear.k_shift = 0        # keylock 交給 KeyTracker 外部維護（v14 直饋）
        kt = R2.KeyTracker()
    else:
        voices = [(f"{DDSP}/exp/reflow-sop3/model_20000.pt", 12.0, 0.85),
                  (f"{DDSP}/exp/reflow-harry-run1/model_14000.pt",
                   -3.0, 0.9),
                  (f"{DDSP}/exp/reflow-bass1/model_32000.pt", -12.0, 1.0)]
        dia_steps = [7, -2, -7]
        vmode = [None, None, None]
    if a.sop_db and voices:
        p0, s0, g0 = voices[0]
        voices[0] = (p0, s0, 0.0 if a.sop_db <= -90 else
                     g0 * 10 ** (a.sop_db / 20.0))
    if a.drop:
        # v31：真的把嘴拿掉（--sop-db 只把增益歸零＝照樣推論、省不到時間）。
        # 三段（voices/dia_steps/vmode）必須同步過濾＝索引對齊。
        keep = [i for i in range(len(voices)) if i not in a.drop]
        voices = [voices[i] for i in keep]
        dia_steps = [dia_steps[i] for i in keep]
        vmode = [vmode[i] for i in keep]
        print(f"drop {sorted(a.drop)} → 剩 {len(voices)} 聲部", flush=True)
    if a.seed is not None:
        torch.manual_seed(a.seed)
        if hasattr(torch, "mps"):
            torch.mps.manual_seed(a.seed)
    svc = Svc(voices, step=a.step, t_start=a.t_start, f0_extractor=a.f0,
              noise_grid=a.noise_grid)
    HOP = svc.args.data.block_size
    blk = max(1, round(a.block * SR / HOP)) * HOP     # 512 對齊＝網格不漂
    cf = int(a.crossfade * SR)
    sola_search = int(0.01 * SR)
    last_delay = int(0.02 * SR)
    LA = a.lookahead * HOP              # v25 前瞻（樣本）；0＝關
    input_frame = ((max(int(a.extra * SR),
                        blk + cf + sola_search + 2 * last_delay + LA)
                    // HOP + 1) * HOP)
    if a.voc_tail:
        # SOLA 真正碰到的尾段（樣本→幀，進位）＋使用者給的邊距
        used = blk + cf + sola_search + last_delay + LA
        svc.voc_tail = -(-used // HOP) + a.voc_tail
        print(f"voc-tail: vocode 尾段 {svc.voc_tail}/{input_frame // HOP} 幀"
              f"（邊距 {a.voc_tail}）", flush=True)
    nb = blk // HOP
    buf = np.zeros(input_frame, dtype="float32")
    ENC_WIN = (max(int(a.enc_win * SR) // HOP * HOP, input_frame)
               if a.enc_win else 0)
    buf3 = np.zeros(ENC_WIN, dtype="float32") if ENC_WIN else None
    VOW = None
    if a.aah:
        # v37 母音庫（同 score_sing/score_live）：take17 最長有聲段中段
        # ~0.9s 的 units 循環＝天使的「啊」。建一次，之後逐窗按絕對幀位查。
        import soundfile as _sf
        mic0, _ = _sf.read("scratchpad/take17.wav", dtype="float64",
                           always_2d=True)
        mic0 = mic0[:, 0]
        _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
        runs, _i = [], 0
        while _i < len(uv0):
            if not uv0[_i]:
                _j = _i
                while _j < len(uv0) and not uv0[_j]:
                    _j += 1
                runs.append((_i, _j))
                _i = _j
            else:
                _i += 1
        _a0, _b0 = max(runs, key=lambda r: r[1] - r[0])
        _mid = (_a0 + _b0) // 2
        with torch.no_grad():
            VOW = svc.encode(
                mic0[max(0, (_mid - 40) * HOP):(_mid + 40) * HOP])[:, 8:-8]
        print(f"[啊] 母音庫 {VOW.size(1)} 幀（take17）＝天使不搬詞",
              flush=True)
    sola_bufs = [torch.zeros(cf, device=svc.device) for _ in svc.voices]
    prev_shift = [0] * len(svc.voices)   # v34 --sola-cont 用（上一塊的 shift）
    RVIR, RVT = None, None
    if a.wet > 0:
        # v37 悠遠層：合成 IR（reverb.py，殺過金屬感的那顆）＋串流
        # overlap-add。只濕天使（y＝純天使混音，乾聲本來就不走這裡）。
        import reverb as rv
        from scipy.signal import fftconvolve as _fftc
        RVIR = rv.make_ir(2.4, trim_db=35.0)
        RVT = [np.zeros(len(RVIR) - 1)]
        print(f"[濕] IR {len(RVIR)/SR:.1f}s、send {a.wet}", flush=True)
    fade_in = torch.sin(np.pi * torch.arange(0, 1, 1 / cf,
                                             device=svc.device) / 2) ** 2
    fade_out = 1 - fade_in
    stats = {"n": 0, "ms": [], "late": 0}
    PROF = a.prof
    prof = {}       # 純 dict：defaultdict 的 import 在下面（:535）
    if PROF:
        _sync = "mps" in str(svc.device)    # device 是字串，不是 torch.device

        def pt(key, t):
            """分段計時。MPS 是非同步佇列——不 synchronize 只會量到「把工作
            排進去」的時間（幾十 μs），真正的計算成本全部堆到下一個同步點。"""
            if _sync:
                torch.mps.synchronize()
            now = time.perf_counter()
            prof[key] = prof.get(key, 0.0) + (now - t) * 1000.0
            return now
    else:
        def pt(key, t):
            return t
    st = {"uc": None, "fc": None, "offc": None, "tb": np.zeros(0),
          "cur": {"up": 12.0, "lo": -12.0}, "ktn": 0,
          "hole": 10 ** 9, "dk": 10 ** 9, "env": 0.0, "nbh": 0, "nbf": 0,
          "f0q": [], "hf0": 0.0, "ufifo": [], "inbr": False,
          "ivp": {"up": None, "lo": None}, "klock": False,
          "agc_g": 1.0, "agc_ema": None,
          "padt": {"up": 0.0, "lo": 0.0}, "pada": None,   # v37 --pad
          "abs": 0}     # v33：串流至今總幀數（凍結噪聲網格的絕對幀位）
    accs = [0.0] * len(svc.voices)      # 每聲部絕對相位累加器（弧度）
    tracker = NoteTracker()             # 遲滯音符（v11 長音穩定）
    from collections import defaultdict
    dmp = defaultdict(list) if a.dump else None
    sus_f = int(round(a.sustain / 1000.0 * SR / HOP))   # 橋接上限（幀）
    rel_f = max(1, int(round(a.release / 1000.0 * SR / HOP)))
    bdk = a.bridge_decay / 1000.0            # v27 橋接衰減 τ（秒）；0＝關
    bhold = a.bridge_hold                    # v29 撐音凍結門檻；0＝關
    buh = a.bridge_units                     # v30 units 解凍門檻；0＝關
    ENV_DK = float(np.exp(-(HOP / SR) / 0.15))   # 音量慣性 τ≈150ms

    def bridge_feed(vol_np, uv, f0_np):
        """v18 樂句橋接（因果逐幀；演化史見 worklog §O/§P）。回 (gate, vol,
        bridged, f0_hold, unfreeze)，只餵新到幀＝結果可凍進網格（跨窗一致）。
        unfreeze＝v30：橋接幀裡「音量還在」＝他其實還在唱（parselmouth 判錯
        聲帶）→ 下游 units 不該凍。
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
        h = np.zeros(len(vol_np), dtype=bool)
        vf0 = np.copy(f0_np)
        for j, x in enumerate(vol_np):
            st["env"] = max(float(x), st["env"] * ENV_DK)
            if not (uv[j] and (st["hole"] > 0 or x < st["env"] * 0.25)):
                st["hole"] = 0
                st["dk"] = 0
                g[j] = 1.0
                st["f0q"] = (st["f0q"] + [float(f0_np[j])])[-4:]
            else:
                st["hole"] += 1
                if st["hole"] == 1:
                    st["hf0"] = st["f0q"][0] if st["f0q"] else float(f0_np[j])
                    st["dk"] = 0
                # v29：音量還在＝他還在唱（parselmouth 判錯聲帶）→ 衰減時鐘
                # 不前進。bhold=0 時 dk 每幀都加＝與舊行為逐幀相同。
                if not (bhold > 0 and x > st["env"] * bhold):
                    st["dk"] += 1
                # v30：units 解凍判準（獨立門檻＝可與 gate 的 bhold 分開 A/B）
                hv = buh > 0 and x > st["env"] * buh
                if st["dk"] <= sus_f:
                    # v27（08-08 Harry「每段結束後像吸口水」）：橋接期 gate
                    # 指數衰減，不再整段撐 1.0。診斷：橋接/淡出段高低頻比
                    # 0.040 vs 唱歌 0.011＝3.6 倍——凍結母音在低音量下
                    # reflow 吐氣音（諧波掉 5.5×、噪聲只掉 1.5×）。
                    # τ 內前 ~50ms 仍近全開＝子音不斷音的原意保住。
                    gd = (1.0 if bdk <= 0 else
                          float(np.exp(-(st["dk"] - 1) * (HOP / SR) / bdk)))
                    g[j], v[j], b[j] = gd, st["env"] * gd, True
                    h[j] = hv
                    vf0[j] = st["hf0"]
                    st["nbf"] += 1
                    if st["hole"] == 1:
                        st["nbh"] += 1
                elif st["dk"] <= sus_f + rel_f:
                    r = (st["dk"] - sus_f) / rel_f
                    g[j] = float(np.cos(r * np.pi / 2) ** 2)
                    v[j], b[j] = st["env"], True
                    h[j] = hv
                    vf0[j] = st["hf0"]
        return g, v, b, vf0, h

    # 預熱（同視窗大小＋同 ratios 路徑＝MPS kernel 編好才開流）
    nfrm = input_frame // HOP + 1
    svc.infer(buf.astype("float64") + 1e-6, a.pitch, a.thr,
              ratios=[np.zeros(nfrm) if (s is not None or vm) else None
                      for s, vm in zip(dia_steps, vmode)])
    print(f"ready  block {blk/SR*1000:.0f}ms / 視窗 {input_frame/SR:.2f}s",
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
                    if a.pad > 0 and (time.time() - st["padt"][part]
                                      < a.pad):
                        pass         # v37 和聲節奏限制：駐留未滿不換（下
                        #               tick 再試＝ivp 保留）
                    else:
                        st["cur"][part] = iv
                        st["ivp"][part] = None
                        if a.pad > 0:
                            st["padt"][part] = time.time()
                            st["pada"] = float(mic)   # 墊的錨＝換和弦時他的音
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
            tp = pt("prep", t0)
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
            # v33：絕對幀位。首窗＝整窗都是新幀；之後每塊 +nb
            st["abs"] = len(f0f) if (fc is None or len(fc) != len(f0f)) \
                else st["abs"] + nb
            if fc is None or len(fc) != len(f0f):
                if sus_f:
                    g, v, b, vf0, h = bridge_feed(
                        vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                    st["gt"], st["vc"], st["br"], st["bu"] = g, v, b, h
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
                        g, v, b, vf0, h = bridge_feed(
                            vol_t[0, :, 0].cpu().numpy(), uvf, f0f)
                        st["gt"], st["vc"], st["br"], st["bu"] = g, v, b, h
                        st["fc"] = vf0
                    else:
                        g, v, b, vf0, h = bridge_feed(
                            vol_t[0, -nb:, 0].cpu().numpy(),
                            uvf[-nb:], f0f[-nb:])
                        st["gt"] = np.concatenate([st["gt"][nb:], g])
                        st["vc"] = np.concatenate([st["vc"][nb:], v])
                        st["br"] = np.concatenate([st["br"][nb:], b])
                        st["bu"] = np.concatenate([st["bu"][nb:], h])
                        st["fc"] = np.concatenate([fc[nb:], vf0])
                else:
                    st["fc"] = np.concatenate([fc[nb:], f0f[-nb:]])
                # v34 --f0-mature：延遲凍結。最近 M+nb 幀（尚未/剛播出）每窗
                # 用最新估計刷新——窗尾 parselmouth 缺右上下文，換音段偏
                # 37.6c（08-10 重播 vs 全上下文重測）；多一窗上下文＝估計成
                # 熟。橋接幀不刷（凍結哲學保留＝hf0 是刻意的）。搭配
                # --lookahead M 使用：幀晚 M 幀播出＝播出時估計已成熟；
                # 不搭 LA 時只改善「當作左上下文被再引用」的部分。
                # 0＝關＝逐位元舊行為。
                if a.f0_mature > 0:
                    M = min(a.f0_mature + nb, len(f0f))
                    keep = st["br"][-M:] if sus_f else \
                        np.zeros(M, dtype=bool)
                    st["fc"][-M:] = np.where(keep, st["fc"][-M:], f0f[-M:])
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
                        if a.pad > 0 and st["pada"] is not None:
                            # v37 --pad：天使釘在「錨音＋音程」的絕對音高＝
                            # 和弦墊——offc 逐幀抵銷他的 f0 移動（melisma/
                            # 顫音不過去）；無聲幀退回相對音程。跳變（換和
                            # 弦/進出無聲）每幀 ≤0.8 半音滑入 ≈200ms porta。
                            fw = f0f[-nb:]
                            mw = 69 + 12 * np.log2(
                                np.maximum(fw, 1.0) / 440.0)
                            new = np.where(fw > 0,
                                           st["pada"] + tgt - mw, tgt)
                            pv = float(st["offc"][vi][-1])
                            for j3 in range(nb):
                                pv += float(np.clip(new[j3] - pv, -0.8, 0.8))
                                new[j3] = pv
                        else:
                            new = np.full(nb, tgt)
                            pv = float(st["offc"][vi][-1])
                            if sus_f and pv != tgt:
                                # v18（review F8）：換音 ~35ms 滑進新音程，
                                # 不整塊硬跳（離線 target_f0 的 porta 精神）。
                                k3 = min(nb, 3)
                                new[:k3] = np.linspace(pv, tgt,
                                                       k3 + 2)[1:k3 + 1]
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
            tp = pt("bridge+brain", tp)
            # units 凍結快取（v2）＋ v12 長音平滑：他唱「嗚」時窗尾 hubert
            # 逐幀亂猜元音＝輸出母音遊走。音符穩定（streak≥6≈70ms）→ units
            # 走 EMA α=0.12（τ≈90ms）釘住；換音/子音（streak 歸零）→ α=1
            # 立即跟上＝起音不糊。平滑在絕對網格上因果進行＝跨窗一致。
            if VOW is not None:
                # v37 --aah：材料換 take17 母音循環（絕對幀網格＝跨窗一致），
                # encode／EMA 凍結／units-mature 全旁路——母音庫零抖動，
                # 那條鏈防的 hubert 逐幀亂猜不存在；hubert 不跑＝省最大宗。
                # f0（st["fc"]）/音量/mask/和聲比（offc）照舊＝即時跟他。
                Lw = input_frame // HOP + 1
                idx = (st["abs"] - Lw + np.arange(Lw)) % VOW.size(1)
                st["usm"] = VOW[:, torch.from_numpy(idx).to(VOW.device)]
                fresh = None
            elif ENC_WIN:
                # v26：encode 吃 3s 過去（左上下文＝免費咬字），只取渲染窗
                # 對應的尾端幀＝下游 uc/usm/infer 完全不感知差異。
                xbe = buf3.astype("float64")
                if a.agc:
                    xbe = xbe * st["agc_g"]
                fresh = svc.encode(xbe)[:, -(input_frame // HOP + 1):]
            else:
                fresh = svc.encode(xb)
            tp = pt("encode", tp)
            uc = st["uc"]
            if fresh is None:
                pass                                 # --aah：usm 已就緒
            elif uc is None or uc.size(1) != fresh.size(1):
                st["uc"] = fresh
                st["usm"] = fresh.clone()
                st["ua1"] = np.ones(fresh.size(1), dtype=bool)  # v36
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
                ua1_new = np.zeros(nb, dtype=bool)
                for j in range(nb):
                    # v30：橋接幀但音量還在（st["bu"]）＝他還在唱 → 不凍，讓
                    # units 照常跟（治 ㄔ／t 發不出來）。副作用是好的：v18 的
                    # 洞口回捲改由真正的音量下墜觸發，不再從假洞口回捲。
                    if sus_f and st["br"][-nb + j] and not st["bu"][-nb + j]:
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
                    ua1_new[j] = (al == 1.0)
                st["ua1"] = np.concatenate([st["ua1"][nb:], ua1_new])
                st["usm"] = torch.cat(
                    [us, torch.from_numpy(news).to(us)], 1)
                # v36 --units-mature：轉換幀成熟刷新。診斷（08-11 台架）：
                # live 轉換幀 units-cos 0.768 vs 正典 0.899＝缺口全在串流，
                # 而 enc-win/t_start/ema 旋鈕全平（±0.01）。機制＝窗尾
                # hubert 缺右上下文、轉換處最糊、進 usm 後永凍。刷新規則：
                # 最近 M 幀（本塊新幀之前）裡「α==1 的轉換幀」用本窗 fresh
                # 的成熟估計（該幀現有 ~M×11.6ms 右上下文）覆寫；α<1 的
                # 釘住幀不碰＝v12 長音保護原封。0＝關＝逐位元舊行為。
                if a.units_mature > 0:
                    L = st["usm"].size(1)
                    lo = max(0, L - a.units_mature - nb)
                    zone = np.zeros(L, dtype=bool)
                    zone[lo:L - nb] = st["ua1"][lo:L - nb]
                    if zone.any():
                        zi = torch.from_numpy(np.where(zone)[0]).to(
                            st["usm"].device)
                        st["usm"][:, zi] = fresh[:, zi]
            if sus_f:
                # v18：橋接 gate 直接取代原 mask——喉麥底噪 −55dB 高於
                # −60 門檻＝原 mask 恆開（review F1：20 分鐘 3 個洞），
                # 取 max 等於沒 gate、真休止關不掉＝bass 低頻嗡。橋接
                # gate 自含「有聲＝開」。音量走凍結快取（洞內包絡 hold）。
                gu = upsample(torch.from_numpy(st["gt"]).float().to(
                    svc.device)[None, :, None], HOP).squeeze(-1)
                k2 = min(mask.size(1), gu.size(1))
                mask = gu[:, :k2]
                vsrc = st["vc"]
                if a.swell > 0:
                    # v37 --swell：上升限速（一階不對稱；下降瞬時＝收尾
                    # 自然）。新幀算一次進快取＝跨窗一致（凍結網格哲學）。
                    vs = st.get("vsw")
                    if vs is None or len(vs) != len(vsrc):
                        st["vsw"] = np.asarray(vsrc, dtype=float).copy()
                    else:
                        head = vs[nb:]
                        p3 = float(head[-1]) if len(head) else float(vsrc[0])
                        au_ = 1 - float(np.exp(-(HOP / SR)
                                               / (a.swell / 1000.0)))
                        seg3 = np.empty(nb)
                        for j3 in range(nb):
                            t3 = float(vsrc[-nb + j3])
                            p3 += (au_ if t3 > p3 else 1.0) * (t3 - p3)
                            seg3[j3] = p3
                        st["vsw"] = np.concatenate([head, seg3])
                    vsrc = st["vsw"]
                vol_t = torch.from_numpy(vsrc).float().to(
                    svc.device)[None, :, None]
            tp = pt("units", tp)
            aus = svc.infer(xb, a.pitch, a.thr,
                            units_override=st["usm"],
                            feats=(st["fc"], vol_t, mask), phases=accs,
                            ratios=st["offc"],
                            enh_tail=(blk + cf + sola_search + last_delay
                                      + LA + int(0.1 * SR))
                            if a.enh_tail else 0,
                            frame0=st["abs"] - len(st["fc"]))
            tp = pt("infer", tp)
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
                score = num[0, 0] / den[0, 0]
                # v34 --sola-cont：shift 連續性偏置。診斷（08-10 重播）：
                # argmax 在相隔一個基頻週期的相關峰間亂跳（|Δshift| p90
                # 7ms、不挑段落）；平段跳＝週期對齊聽不出，換音段跳＝
                # 偏差=斜率×Δt（陡段 50-74c＝「音不穩」主因）。偏置＝
                # 離上一塊 shift 近的峰加權，選「最近的夠好峰」不選全域
                # 最大。0＝關＝逐位元舊行為。
                if a.sola_cont > 0:
                    dist = torch.abs(torch.arange(
                        score.numel(), device=score.device, dtype=score.dtype)
                        - prev_shift[vi])
                    score = score * (1 - a.sola_cont * dist / sola_search)
                shift = int(torch.argmax(score))
                prev_shift[vi] = shift
                if dmp is not None:      # v33 儀器：接縫時基抖動的追凶證據
                    dmp[f"shift{vi}"].append(np.array([shift]))
                tw = tw[shift: shift + blk + cf].clone()
                # v34 --seam-amp：純振幅淡接（跳過 phase_vocoder）。診斷：
                # crossfade 長度對音準劣化有單調劑量反應（0.02/0.04/0.08 →
                # 陡段 60/73/82c）＝相位內插在彎瞬時頻率；shift 已由
                # --sola-cont 對齊時，振幅淡接不彎音高。預設關＝舊行為。
                if a.seam_amp:
                    tw[:cf] = sola_bufs[vi] * fade_out + tw[:cf] * fade_in
                else:
                    tw[:cf] = phase_vocoder(sola_bufs[vi], tw[:cf],
                                            fade_out, fade_in)
                sola_bufs[vi] = tw[-cf:]
                w = tw[:-cf].cpu().numpy()
                if stems is not None:
                    stems.append(w.copy())
                y = w if y is None else y[:len(w)] + w[:len(y)]
            tp = pt("sola+mix", tp)
            if a.agc:
                y = y / st["agc_g"]        # 位準結構還原（input_agc 同哲學）
            if RVIR is not None:
                # v37 串流卷積：本塊全響應，塊內出頭、尾巴累進 RVT
                wf = _fftc(y.astype(float), RVIR)
                wet_ = wf[:len(y)]
                tl = RVT[0]
                wet_ += tl[:len(y)]
                nt = wf[len(y):]
                rest = tl[len(y):]
                nt[:len(rest)] += rest
                RVT[0] = nt
                y = y + a.wet * wet_
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
                    for kk in ("gt", "vc", "br", "bu"):
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
                  f" ｜ 慢 {stats['late']} 缺塊 {st2['under']}"
                  f" 旗標 {st2['flags']}"
                  + (f" 橋 {st['nbh']}洞/{st['nbf']}幀" if sus_f else ""),
                  flush=True)

    def prof_report():
        """--prof 預算表。注意：synchronize 本身有成本＝合計會高於同設定的
        非 prof 跑，只讀分項佔比，不要拿絕對值跟 RTF 比。"""
        n = max(stats["n"], 1)
        tot = sum(prof.values())
        out = [f"\n分段預算（--prof；{stats['n']} 塊、block {blk/SR*1000:.0f}ms）",
               f"{'段':<13}{'ms/塊':>9}{'佔比':>9}"]
        for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
            out.append(f"{k:<13}{v/n:9.1f}{100*v/max(tot, 1e-9):8.1f}%")
        out.append(f"{'合計':<12}{tot/n:9.1f}{100.0:8.1f}%")
        return "\n".join(out)

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
        if PROF:
            print(prof_report(), flush=True)
        if dmp is not None and dmp["mic"]:
            flush_dump()
            print(f"dump: {a.dump}_mic/_angel.wav + _feats.npz", flush=True)
        return

    print(f"開流 in={a.in_name!r} out={a.out_name!r} pitch {a.pitch:+.0f}"
          f" t_start {a.t_start} step {a.step}"
          f" seed {a.seed if a.seed is not None else '隨機'}"
          f"（Ctrl-C 結束）", flush=True)
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
