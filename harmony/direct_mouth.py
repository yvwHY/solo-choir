"""direct_mouth.py — 直驅嘴：他的原聲進 DDSP，f0 換成腦的目標線（不經 WORLD）

動機（2026-07-26 Harry 全版耳測：每一版都被判「autotune」）：
現行鏈路是 **腦 → WORLD 合成天使 → DDSP 轉換**。F19 已消融證明 autotune 感
不是 f0 曲線配方（H1/H2/H3 humanize 全敗）而是「合成載體」——但**從未檢驗的
是：那個合成載體其實是鏈路中間的 WORLD**。DDSP 只是忠實轉換一個已經帶
WORLD buzz 的訊號。

本嘴把 WORLD 整段拔掉：
  輸入音訊 = 他的原始 take（content units＝他的咬字、音素、氣息全部真實）
  f0        = 腦的目標線（F16：天使一律按目標放音高）
  輸出      = 他的音色、唱他的字、在天使的音高上

實作路徑＝零改動 DDSP repo：main.py:177-199 有 f0 cache（檔名含輸入 md5 與
參數），先把目標 f0 寫進 cache 檔，main.py 就會載入它而不做 pitch extraction。

用法（vcclient-dev env，於 harmony/ 下）：
  python direct_mouth.py --notes out/angel_v2_world4keyed_notes.json \
      --take ../../../260722_harmony_brain/data/take.wav --tag direct30k
  （會印出接著要跑的 DDSP 指令；--run 直接代跑）

2026-07-30 起預設開 voicing gate（`--respect-unvoiced`，見 voicing_mask）：他的
無聲幀注入 f0=0，不再把吸氣/擦音唱出來。舊行為＝`--no-respect-unvoiced`。
"""
import argparse, hashlib, json, math, os, subprocess, sys
import numpy as np
import soundfile as sf

import pathlib as _pl  # noqa: E402
sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from config import DDSP as _DDSP  # noqa: E402
DDSP = str(_DDSP)
HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
HOP = 512            # DDSP combsub config: block_size 512 @ 44100
PORTA = 0.15         # 同 world_mouth 的一極滑音（每 hop）
VIB_HZ, VIB_SEMI = 5.0, 0.12


def load_notes(path):
    d = json.load(open(path))
    if "notes" in d:
        return d["notes"], int(d["step_samps"]), d.get("heard")
    return d["targets"], int(round(d["tick_sec"] * SR)), None


def target_f0(notes, tick, n_hops, porta_ms=33.0, leap_snap=4,
              vib_hz=VIB_HZ, vib_phase=0.0, vib_onset_ms=0.0, vib_semi=VIB_SEMI):
    """腦的音符線 → 每 hop 的目標 f0（Hz），無聲處用前值延續（DDSP 的
    uv_interp 語意：f0 曲線不能有 0 洞，音量遮罩另外處理）。

    滑音按需（2026-07-28 耳測回饋「三 cell 女聲全在滑」後改）：只有樂句內
    ≤leap_snap 半音的級進才滑（一極，時間常數 porta_ms＝world_mouth 的
    33ms；原 PORTA=0.15/hop 在 11.6ms hop 上是 world 的 2.3 倍滑程），
    大跳與跨 rest 的換音直接落點——跨 rest 滑音會從前樂句尾音 scoop 進
    新樂句頭，最大實測 10 半音。

    vib_hz/vib_phase（cycles）/vib_onset_ms：多聲部 vibrato 去同步＋起振
    延遲（2026-07-28 autotune 回饋：兩天使同相位同速率正弦＝機械齊振；
    真歌手長音先直後顫）。預設值＝原行為。"""
    hop_ms = 1000.0 * HOP / SR
    alpha = 1.0 - math.exp(-hop_ms / porta_ms) if porta_ms > 0 else 1.0
    f0 = np.zeros(n_hops)
    cur, prev_m, gap, age = 0.0, None, False, 0
    for k in range(n_hops):
        t = min(int(k * HOP / tick), len(notes) - 1)
        m = notes[t]
        if m is None:
            f0[k] = cur          # 保持最後音高（uv 區段由 volume mask 靜音）
            gap = gap or prev_m is not None
            continue
        tgt = 440.0 * 2 ** ((m - 69) / 12.0)
        if m != prev_m or gap:
            if (cur == 0.0 or prev_m is None or gap
                    or (leap_snap and abs(m - prev_m) > leap_snap)):
                cur = tgt        # 落點：樂句頭、跨 rest、大跳
            prev_m, age = m, 0
        gap = False
        age += 1
        cur = cur + (tgt - cur) * alpha
        depth = vib_semi * (min(1.0, age * hop_ms / vib_onset_ms)
                            if vib_onset_ms > 0 else 1.0)
        vib = 2 ** (depth * math.sin(
            2 * math.pi * (vib_hz * k * HOP / SR + vib_phase)) / 12)
        f0[k] = cur * vib
    if f0[0] == 0:
        nz = np.flatnonzero(f0)
        if len(nz):
            f0[:nz[0]] = f0[nz[0]]
    return f0


def _runs(mask, val):
    """連續同值區間 [a, b)。"""
    i, out = 0, []
    while i < len(mask):
        if mask[i] == val:
            j = i
            while j < len(mask) and mask[j] == val:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def harvest_f0(x):
    """pyworld.harvest 的單一入口。**每句最貴的一項**（08-03 實測 RTF 0.114），
    而 input_agc 與 voicing_conf 各自呼叫一次、兩張嘴又各跑一遍＝同一段音訊
    被 harvest 四次。兩者要的都是同一件事（哪些幀有基頻），所以算一次傳進去。
    傳 f0m 進去是純快取＝結果逐 byte 相同，不是近似。"""
    import pyworld
    f0m, _ = pyworld.harvest(np.ascontiguousarray(x), SR,
                             frame_period=1000.0 * HOP / SR)
    return f0m


def extract_f0_uv(x, n_hops, floor_hz=0.0):
    """他的真實音高 → (f0, voiced)，**unvoiced 幀從頭到尾是 0**。

    2026-07-30 架構改：舊版先把 unvoiced 段用前值填滿、再由 gate 回頭打洞
    ＝「先毀後補」，兩份真相會不一致。這裡改成單一真相源——harvest 的
    voiced/unvoiced 成對帶著走，八度中位修復**只在同一段 voiced run 內**取
    鄰居中位數（不跨 unvoiced 借值、不內插），unvoiced 幀永遠不被填。
    """
    import pyworld
    x = np.ascontiguousarray(x)
    f0m, _ = pyworld.harvest(x, SR, frame_period=1000.0 * HOP / SR)
    v = f0m > 0
    midi = 69 + 12 * np.log2(np.maximum(f0m, 1) / 440.0)
    out = np.zeros(len(f0m))
    for a, b in _runs(v, True):                    # 八度修復侷限在 run 內
        seg = midi[a:b]
        med = np.array([np.median(seg[max(0, i - 4):i + 5])
                        for i in range(len(seg))])
        fixed = np.where(np.abs(seg - med) > 6, med, seg)
        out[a:b] = 440.0 * 2 ** ((fixed - 69) / 12.0)
    if floor_hz > 0:                     # 曲線層八度摺疊：他的句尾下沉＋負移調
        low = (out > 0) & (out < floor_hz)  # 會掉出嘴的音域（實測 ~5% hop <65Hz）
        while low.any():
            out[low] *= 2.0
            low = (out > 0) & (out < floor_hz)
    f0 = np.zeros(n_hops)
    voiced = np.zeros(n_hops, dtype=bool)
    n = min(n_hops, len(out))
    f0[:n], voiced[:n] = out[:n], v[:n]
    return f0, voiced


def shift_f0(notes, heard, tick, n_hops, x, floor_hz=0.0):
    """shift 模式（world_mouth 3b 的直驅移植，2026-07-28 autotune 追根）：
    不合成 f0——抽他自己的真實音高曲線，逐 tick 按（天使音−他的音）音程
    移調。微偏音/飄動/真 vibrato/換音過渡全是他的，源頭是真人不是合成器。
    無聲段 f0 恆為 0（2026-07-30 起，見 extract_f0_uv）。
    porta/vibrato 參數不適用本模式。"""
    f0_base, _ = extract_f0_uv(x, n_hops, floor_hz)
    shifts, last = [], None
    for n, h in zip(notes, heard):
        if n is None:
            shifts.append(None)
        else:
            if h is not None:
                last = n - h
            shifts.append(last)
    out = np.zeros(n_hops)
    for k in range(n_hops):
        s = shifts[min(int(k * HOP / tick), len(shifts) - 1)]
        if s is not None and f0_base[k] > 0:
            out[k] = f0_base[k] * 2 ** (s / 12.0)   # 其餘留 0＝native uv
    return out


PER_K_LO, PER_K_HI = 0.45, 0.80
LOCAL_HALF = int(round(3.0 * SR / HOP))      # 局部參考半窗＝3 秒


def _ramp(v, lo, hi):
    return np.clip((v - lo) / (np.asarray(hi) - np.asarray(lo) + 1e-12), 0.0, 1.0)


def _local_ref(vals, mask, half, q):
    """vals 在 mask 為真的幀上、±half 窗內的第 q 百分位；無樣本則退回全段值。"""
    idx = np.flatnonzero(mask)
    glob = np.percentile(vals[idx], q) if len(idx) else 0.0
    out = np.full(len(vals), glob, dtype=float)
    if not len(idx):
        return out
    for k in range(len(vals)):
        w = idx[(idx >= k - half) & (idx <= k + half)]
        if len(w) >= 8:
            out[k] = np.percentile(vals[w], q)
    return out


def voicing_frames(x, k0, k1, win=2048):
    """hop 區間 [k0, k1) 的逐幀 voicing 特徵 (per, flat, rms)。

    抽出來是為了管線化（08-03）：每一幀只看 [k·HOP−win/2, k·HOP+win/2) 這
    2048 個樣本＝**純局部**，音訊一到窗滿就能算，而且與整句一次算的結果逐
    byte 相同（同一段樣本、同一個式子）。所以他還在唱的時候就可以先算掉。
    窗超出 x 的幀留 0——與原本整段迴圈同語意（句尾那幾幀本來就是 0）。"""
    q0, q1 = int(SR / 800), int(SR / 65)
    w = np.hanning(win)
    per, flat, rms = (np.zeros(k1 - k0) for _ in range(3))
    for k in range(k0, k1):
        s = k * HOP - win // 2
        if s < 0 or s + win > len(x):
            continue
        seg = x[s:s + win] * w
        e = float(np.dot(seg, seg))
        rms[k - k0] = np.sqrt((seg ** 2).mean())
        if e < 1e-12:
            continue
        per[k - k0] = (np.correlate(seg, seg, "full")[win - 1:] / e)[q0:q1].max()
        p = np.abs(np.fft.rfft(seg)) ** 2 + 1e-12
        flat[k - k0] = np.exp(np.mean(np.log(p))) / np.mean(p)
    return per, flat, rms


def frames_ready(n_samps, win=2048):
    """已有 n_samps 個樣本時，window 已經填滿的 hop 數（＝可先算的幀）。"""
    return max(0, (n_samps - win // 2) // HOP + 1)


def voicing_conf(x, n_hops, win=2048, f0m=None, frames=None):
    """逐幀 voicing 信心 ∈[0,1]：多特徵合議，**存疑判無聲**。

    只信 harvest 不夠——它把他的吸氣/擦音judge成 voiced 的比例不低（本段
    take-voiced 幀的週期性 p5 只有 0.35＝那批就是誤判）。三個特徵合議：
      per  自相關週期性峰（65–800Hz lag）——最強判別器（voiced p10 0.72 /
           uv p90 0.62），主導票
      flat 頻譜平坦度（噪音狀↑）
      db   幀能量（相對 voiced 段 p90）
    conf = harvest ? min(per_score, max(flat_score, db_score)) : 0
    ——min＝最弱環節決定（存疑判無聲），max＝低能量但乾淨的尾音仍可由
    flat 救回，不讓安靜的真唱被誤殺。閾值以三格梯這段的標記統計調。
    """
    per, flat, rms = (voicing_frames(x, 0, n_hops, win) if frames is None
                      else frames)
    db = 20 * np.log10(rms + 1e-9)
    if f0m is None:
        f0m = harvest_f0(x)
    hv = np.zeros(n_hops, dtype=bool)
    n = min(n_hops, len(f0m))
    hv[:n] = f0m[:n] > 0
    # 參考值取「局部」而非全段：他一條 take 內位準可以掉 5–13dB（2026-07-30
    # Harry「12 秒後變成吸不了氣又要唱歌」的根因）——安靜段的 SNR 較差，週期性
    # 與能量兩個特徵一起下修，用全段固定門檻會把後半的**真唱**判成無聲，透傳
    # 於是拿原聲蓋掉他正在唱的地方＝嗆到感。局部參考讓門檻跟著段落走。
    ref_db = _local_ref(db, hv, LOCAL_HALF, 90)
    ref_per = _local_ref(per, hv, LOCAL_HALF, 50)
    conf = np.minimum(_ramp(per, PER_K_LO * ref_per, PER_K_HI * ref_per),
                      np.maximum(_ramp(-flat, -0.010, -0.002),
                                 _ramp(db, ref_db - 26, ref_db - 14)))
    return np.where(hv, conf, 0.0), hv, db - ref_db


def voicing_mask(x, n_hops, min_uv_ms=40.0, guard_ms=0.0, hi=0.60, lo=0.35,
                 anchor_ms=25.0, weak_db=-12.0, f0m=None, frames=None):
    """voicing 信心 → 注入用的硬 gate（2026-07-30 Harry 耳測：「格2/格3 都在
    吸氣與西類氣音處有明顯 autotune」）。

    為什麼是硬的：CombSubSuperFast 的 forward 只吃 units/f0/volume
    (`ddsp/vocoder.py:735`)，harmonic:noise 混合比是網路自己預測的
    (`split_map` :729)，**沒有 per-frame ap 入口**——真正的連續軟 gate 要改
    DDSP repo，故不做。退而求其次的漸變是免費的：`ddsp/core.py:66` 的
    upsample 是線性內插，f0 在該 block 內線性過渡＝每個 gate 邊界天生有
    11.6ms 的激振淡入淡出，不需要（也不該）自己拉 f0 斜坡——拉長會變成聽得見
    的下滑音。

    unvoiced 幀注入的是 **1200Hz 哨兵而非 0**（`--uv-f0`）。硬歸零聲學上最好
    （吸氣段 CPP 1.838 ≈ 原聲 1.855），但 combtooth 是
    sinc(sr·x/(f0+1e-3))：f0→0 時相位 x 幾乎凍結、引數趨近 0/0，只要某次
    ramp 的相位環繞剛好落在 near-zero，sinc 就回到 1＝**全刻度爆音**（實測
    peak 1.0000、crest 31.7，26s 內兩次）。哨兵讓相位維持每樣本前進一單位、
    引數永遠良態（crest 5.8），而 1200Hz 遠高於他的音域（110–300），模型的
    諧波濾波器渲染不出可信的音高；實測吸氣段寬頻 CPP 800–4000Hz 1.583 vs 原聲
    1.581＝沒有偷渡哨兵嗡聲。內部表徵仍以 (f0, uv) 成對為真相，哨兵只在最後
    注入那一步替換，屬 vocoder 條件化的權宜。

    信心用施密特遲滯轉成狀態（hi 進 voiced / lo 進 unvoiced），邊界因此由實際
    聲學決定，取代舊版固定 guard 內縮（那正是殘留的主因：邊界幀被強制點回
    voiced）。min_uv_ms 濾掉長音中的偶發掉點（gate 顆粒感來源）。
    """
    hop_ms = 1000.0 * HOP / SR
    conf, hv, rel_db = voicing_conf(x, n_hops, f0m=f0m, frames=frames)
    v = np.zeros(n_hops, dtype=bool)
    state = True
    for k in range(n_hops):
        if state and conf[k] < lo:
            state = False
        elif not state and conf[k] > hi:
            state = True
        v[k] = state
    # 錨定（2026-07-30 Harry「12 秒後像吸不了氣又要唱歌」的根因修法）：gate 只
    # 准落在他**真的沒發聲**的地方（harvest uv）及其 anchor_ms 鄰域。原本合議
    # 可以在連續樂句中間自行判無聲——實測 14s 窗他真實 uv 只有 1.7%（一路唱），
    # gate 卻標了 14.5%，透傳於是把 151ms 的原聲塞進他根本沒換氣的樂句＝嗆到
    # 感。透傳把過度 gate 的代價從「少一點諧波」變成「原聲蓋掉天使」，成本不
    # 對稱翻轉，故寧可漏掉一口呼吸也不可在樂句中間開洞。
    if anchor_ms >= 0:
        allow = ~hv
        # 沿「連續低信心鏈」由真無聲處向外長（2026-07-30 第三輪：Harry 說某顆音
        # 有金屬感）。起音前的極弱音（rms<0.005）harvest 照樣報 voiced，但吐出
        # 亂跳 2–8 半音/hop 的垃圾 f0，combtooth 拿它渲染 4–8 hop＝樂句頭一段
        # 非諧波刮擦（實測該處輸出非諧波能量 75–99% vs 原聲 1–6%）。這段垃圾
        # 從 harvest-uv 往後延伸 35–90ms，固定 25ms 膨脹蓋不到而被錨回 voiced。
        # 改成「與真無聲相連的整條 conf<lo 鏈都可 gate」：樂句中間孤立的低信心
        # 幀仍動不了（不會再開洞），起音前的垃圾鏈則整段收乾淨。
        # 鏈只准穿過「弱音」幀：正常音量的幀就算信心低也不讓鏈長過去，
        # 否則會在樂句裡挖洞（實測不設限時最長挖 81ms）。
        low = (conf < lo) & (rel_db < weak_db)
        for a, b in _runs(low, True):
            if allow[a:b].any() or (a > 0 and allow[a - 1]) or \
                    (b < n_hops and allow[b]):
                allow[a:b] = True
        d = int(round(anchor_ms / hop_ms))
        if d:
            grown = allow.copy()
            for a, b in _runs(allow, True):
                grown[max(0, a - d):min(n_hops, b + d)] = True
            allow = grown
        v = v | ~allow
    for a, b in _runs(v, False):                   # 填回過短的 unvoiced 空隙
        if (b - a) * hop_ms < min_uv_ms:
            v[a:b] = True
    g = max(0, int(round(guard_ms / hop_ms)))
    if g:
        keep = v.copy()
        for a, b in _runs(v, False):
            keep[a:min(b, a + g)] = True
            keep[max(a, b - g):b] = True
        v = keep
    return v.astype(float)


TRAIN_REF_RMS = 0.0566      # 260724_ddsp_svc/data/train/volume 上半部幀中位數


def input_agc(x, ref_rms=TRAIN_REF_RMS, win_s=2.0, lo=0.25, hi=16.0, f0m=None):
    """轉換前把 voiced 段位準拉到訓練參考位準（轉換後再除回去）。

    2026-07-30 Harry：透傳修好呼吸後，「12 秒後還是像吸不了氣／擠著唱」。
    根因不在 gate 也不在呼吸，在 **voiced 段的轉換本身**：他這條 take 的
    voiced rms 中位 0.0134，低於訓練 clips 的 p10（0.0188），後段掉到 0.005
    ＝訓練中位的 1/8——DDSP 的 volume 是模型的條件輸入之一
    (`ddsp/vocoder.py:735` volume_frames)，位準掉出訓練分布，轉換就退化。

    證據：轉換輸出對原聲的頻譜距離隨他的位準單調惡化（14.3dB@rms .0195 →
    18.3dB@rms .0055），而**原聲自己**的 CPP 前後段幾乎不變（2.42→2.34）
    ＝不是他唱壞了，是模型吃到不熟悉的位準。

    分窗量測 voiced rms、線性內插成平滑增益（避免呼吸泵動），回傳
    (x_agc, gain)；輸出端乘回 1/gain，位準結構完全還原。
    """
    f0 = harvest_f0(x) if f0m is None else f0m
    v = np.repeat(f0 > 0, HOP)
    v = (np.concatenate([v, np.zeros(len(x) - len(v), dtype=bool)])
         if len(v) < len(x) else v[:len(x)])
    W = int(win_s * SR)
    cs, gs = [], []
    for s in range(0, len(x), W):
        sl = slice(s, min(len(x), s + W))
        m = v[sl]
        if m.sum() < 0.1 * SR:
            continue
        r = np.sqrt((x[sl][m] ** 2).mean())
        if r > 1e-9:
            cs.append(0.5 * (s + min(len(x), s + W)))
            gs.append(ref_rms / r)
    if not gs:
        return x.copy(), np.ones(len(x))
    g = (np.interp(np.arange(len(x)), cs, gs) if len(gs) >= 2
         else np.full(len(x), gs[0]))
    g = np.clip(g, lo, hi)
    peak = float(np.abs(x * g).max())
    if peak > 0.95:                     # 整條同比例壓回，增益曲線形狀不變
        g = g * (0.95 / peak)
    return x * g, g


def uv_passthrough(ang, x, vm, xfade_ms=15.0, dry_gain=1.0):
    """unvoiced 段直接用他的原聲，不用合成的（2026-07-30 Harry 耳測第三輪：
    1200Hz 哨兵「反而變成刺耳尖刺刮金屬的聲音」——諧波支路在 1200Hz 基頻上
    渲染泛音柱，儀器的 CPP 800–4000Hz 沒抓到、耳朵抓到了）。

    直驅鏈路本來就吃他的原始 take，呼吸/氣音**根本不需要合成**：voiced 段用
    轉換結果，uv 段用原聲，邊界用 gate 的遲滯判定＋升餘弦交叉淡化。這把「無聲
    段該長什麼樣」整個問題從模型手上拿走。

    位準：以 voiced 段的 rms 比值把原聲縮放到轉換輸出的量級——保住他錄音裡
    原本的「呼吸相對於歌聲」的比例，而不是硬把呼吸拉到某個絕對值。
    誠實面：upper 天使是 girl 音色，透傳的呼吸是 Harry 的音色；呼吸的聲者
    辨識度低，先假設可接受——這正是要耳測的點。

    dry_gain（08-04）＝透傳的原聲比例。**stems 單獨播（live_v3/perform）維持
    1.0**；respond2 的回應混音裡他的乾聲本來就在，兩個 stems 再各透傳一份＝
    同一口氣疊三份、實測換氣/子音比純乾聲多 +7.7 dB → respond2 傳 0＝uv 段
    stems 靜音（升餘弦斜坡照舊，哨兵輸出一樣被壓掉，07-30 的雷不會回來）。
    """
    n = min(len(ang), len(x))
    ang, x = ang[:n], x[:n]
    g = np.repeat(np.asarray(vm, dtype=float), HOP)
    g = (np.concatenate([g, np.full(n - len(g), g[-1] if len(g) else 1.0)])
         if len(g) < n else g[:n])
    half = max(1, int(SR * xfade_ms / 2000.0))
    k = np.hanning(2 * half + 1)
    g = np.convolve(g, k / k.sum(), mode="same")     # 二元遮罩 → 升餘弦斜坡
    v = np.repeat(np.asarray(vm, dtype=float), HOP)
    v = (np.concatenate([v, np.ones(n - len(v))]) if len(v) < n else v[:n]) > 0.5
    # 分窗增益：他一條 take 內 voiced rms 實測從 0.037 掉到 0.0075（~13dB），
    # 全段一個 gain 會讓後段透傳的呼吸相對走位。逐 2s 量、線性內插成平滑曲線。
    W = int(2.0 * SR)
    cs, gs = [], []
    for s0 in range(0, n, W):
        sl = slice(s0, min(n, s0 + W))
        m = v[sl]
        if m.sum() < 0.1 * SR:
            continue
        den = np.sqrt((x[sl][m] ** 2).mean())
        if den > 1e-12:
            cs.append(0.5 * (s0 + min(n, s0 + W)))
            gs.append(np.sqrt((ang[sl][m] ** 2).mean()) / den)
    if len(gs) >= 2:
        s = np.clip(np.interp(np.arange(n), cs, gs), 0.25, 4.0)
    else:
        s = np.full(n, gs[0] if gs else 1.0)
    return g * ang + dry_gain * (1.0 - g) * s * x, float(np.median(s))


# 表現層參數（2026-07-31 量測 seg_take.wav 26.1s、19 條長 sustain run 的統計；
# 校準腳本見 worklog 07-31。sustain cents_sd 32.57 分解成四個成分＋進音彎）：
#   bias   每音偏置 sd 22c（median |bias| 20c，range −37..+34）——最大貢獻源
#   drift  音內慢漂移 sd 13c（≤2.5Hz 帶）
#   jit    快抖 sd 6c（9–20Hz 帶佔殘差能量 19%，內含 harvest 量測噪音故取保守值）
#   vib    深度 20c 峰值（p25 16.5/p75 29.4）、速率 5.9Hz、速率抖動 ~0.9Hz
#          （零交越量到 1.71 但含噪，取半）、起振延遲 ~0ms（量測中位）
#   bend   進音彎 |22c| 中位、59% 由下勾入、35ms 中位安定（p75 75ms）
EXPR = dict(bias_sd=22.0, bias_clip=35.0, drift_sd=13.0, drift_fc=1.5,
            jit_sd=6.0, jit_fc=10.0, vib_hz=5.9, vib_walk_sd=0.9,
            vib_depth=20.0, vib_depth_sd=6.0, vib_depth_lo=8.0,
            vib_depth_hi=32.0, bend_mag=22.0, bend_down=0.59,
            bend_tau_lo_ms=15.0, bend_tau_hi_ms=60.0,
            tune_fc=0.25)   # 合唱團共用音準參考的變化速率（tune_lock 用）


def _lp_noise(rng, n, fc, sd):
    """一極低通白噪 → 目標 sd 的緩變曲線（決定論：吃外部 rng）。"""
    alpha = 1.0 - math.exp(-2 * math.pi * fc * HOP / SR)
    y = np.empty(n)
    acc = 0.0
    for i, w in enumerate(rng.standard_normal(n)):
        acc += (w - acc) * alpha
        y[i] = acc
    s = y.std()
    return y * (sd / s) if s > 1e-9 else y


def expressive_cents(notes, tick, n_hops, leap_snap=4, seed=20260731, p=EXPR,
                     tune_seed=None, tune_lock=0.0):
    """規則版表現層（2026-07-31，兇手 (b)「f0 比真人直一倍」的藥）：
    對 target 骨架的 cents 偏差曲線＝每音偏置＋慢漂移＋快抖＋不規則顫音
    ＋進音彎。參數抄真人統計（見 EXPR 註解），固定 seed＝render 可重現。

    與 F19 三敗案（H1/H2/H3 humanize）的差異：當年病灶 (a) 的金屬感蓋台、
    contour 差異聽不見；(a) 收官後 (b) 才可單獨聽見（07-30 三格梯 ② 過盲聽
    ＝f0 紋理確為槓桿）。進音彎只加在骨架瞬間落點處（樂句頭/跨 rest/大跳）
    ——級進已有 porta 彎，不疊加。

    tune_lock（2026-08-02，additive；0＝舊行為 byte-identical）＝合唱團對音。
    這一層本來每個聲部各自亂偏，兩天使的**音高中心**因此對不上：實測音程誤差
    SD 40c、>20c（聽得出不準）佔 61.6% 的時間——Harry 判「怪」，而 08-01
    「修完調性仍有殘餘怪感」也吻合。真實合唱團不是這樣：歌手彼此聽、音準
    中心一致，各自不同的只有顫音與抖動。所以把這層拆兩半：

      慢的（每音偏置 bias＋漂移 drift＝音高中心）→ 依 tune_lock 走共用 rng
      快的（jitter、顫音速率/相位、進音彎＝紋理）→ 永遠各聲部獨立

    兩聲部傳同一個 tune_seed、tune_lock=1 → 慢層完全共用＝音高中心零誤差，
    但顫音仍去同步（07-28「機械齊振＝autotune 感來源之一」的教訓保住）。
    能量守恆：sqrt(L)·共用 + sqrt(1-L)·自有，總 SD 不隨 L 變，只有**聲部間**
    的誤差變。共用成分抽自時間域慢曲線、在每個音的起點取值——兩聲部換音點
    不同也能對上（歌手是照時間對音，不是照小節對音）。

    ⚠ rng 抽取順序刻意不動：共用成分一律抽自獨立的 trng，所以 tune_lock=0
    時每一次 rng 呼叫的順序與數量都與舊版相同＝Regime A 可驗。"""
    rng = np.random.default_rng(seed)
    trng = np.random.default_rng(seed if tune_seed is None else tune_seed)
    hop_s = HOP / SR
    # 走一遍 target_f0 的音符節奏：每 hop 的音符實例 id＋瞬間落點 onset 清單
    note_id = np.full(n_hops, -1)
    snaps = []
    nid, prev_m, gap = -1, None, False
    for k in range(n_hops):
        m = notes[min(int(k * HOP / tick), len(notes) - 1)]
        if m is None:
            gap = gap or prev_m is not None
            note_id[k] = nid
            continue
        if m != prev_m or gap:
            if prev_m is None or gap or (leap_snap and abs(m - prev_m) > leap_snap):
                snaps.append(k)
            nid += 1
            prev_m = m
        gap = False
        note_id[k] = nid
    n_notes = nid + 1
    # 每音抽樣（固定順序＝決定論）
    bias = np.clip(rng.normal(0.0, p["bias_sd"], n_notes),
                   -p["bias_clip"], p["bias_clip"])
    depth = np.clip(rng.normal(p["vib_depth"], p["vib_depth_sd"], n_notes),
                    p["vib_depth_lo"], p["vib_depth_hi"])
    # 連續成分
    drift = _lp_noise(rng, n_hops, p["drift_fc"], p["drift_sd"])
    common = 0.0
    if tune_lock:                              # 合唱團對音：慢層改走共用 rng
        L = tune_lock
        # 共用成分必須是**連續**曲線，不能逐音取樣：兩聲部換音點不同，採同一
        # 條曲線的不同時刻＝等於各抽各的（08-02 實測逐音版慢成分只從 27.7 掉到
        # 14.8c＝沒修到）。這裡改成整條共用，個人的每音偏置隨 L 淡出——
        # **人味住在「動」（漂移/抖動/顫音/進音彎），不住在「一個固定的偏差」**；
        # 獨唱時常數偏置像人，兩個聲部各偏各的就只是不準。
        common = L * _lp_noise(trng, n_hops, p["tune_fc"], p["bias_sd"])
        sdrift = _lp_noise(trng, n_hops, p["drift_fc"], p["drift_sd"])
        bias = (1.0 - L) * bias
        drift = (1.0 - L) * drift + L * sdrift
    jit = _lp_noise(rng, n_hops, p["jit_fc"], p["jit_sd"])
    rate = p["vib_hz"] + _lp_noise(rng, n_hops, 0.8, p["vib_walk_sd"])
    vib = np.sin(2 * math.pi * np.cumsum(rate) * hop_s)
    dep = np.where(note_id >= 0, depth[np.clip(note_id, 0, None)], 0.0)
    a = 1.0 - math.exp(-hop_s / 0.03)          # 換音時深度 30ms 平滑過渡
    for i in range(1, n_hops):
        dep[i] = dep[i - 1] + (dep[i] - dep[i - 1]) * a
    out = np.where(note_id >= 0, bias[np.clip(note_id, 0, None)], 0.0) \
        + drift + jit + dep * vib + common
    # 進音彎：瞬間落點處由下（59%）或上勾入，指數安定
    for k0 in snaps:
        sgn = -1.0 if rng.random() < p["bend_down"] else 1.0
        mag = min(60.0, rng.lognormal(math.log(p["bend_mag"]), 0.5))
        tau = rng.uniform(p["bend_tau_lo_ms"], p["bend_tau_hi_ms"]) / 1000.0
        span = min(n_hops - k0, int(5 * tau / hop_s) + 1)
        t = np.arange(span) * hop_s
        out[k0:k0 + span] += sgn * mag * np.exp(-t / tau)
    return out


def texture_cents(x, n_hops):
    """他的真實 f0 → 微音準紋理（對最近半音的偏差，cents，天然 ±50 有界）。
    2026-07-28 診斷梯裁決（B 微 autotune／C 變重）後的合成解：target 骨架
    的穩定＋他的真人質感，不取 shift 的不穩——不穩來源＝過渡段，故他快速
    換音的 hop 紋理歸零（>35c/hop），只蓋印 sustain 的微偏/飄動/真 vibrato。"""
    import pyworld
    x = np.ascontiguousarray(x)
    f0, _ = pyworld.harvest(x, SR, frame_period=1000.0 * HOP / SR)
    midi = 69 + 12 * np.log2(np.maximum(f0, 1) / 440)
    dev = (midi - np.round(midi)) * 100.0
    dm = np.abs(np.diff(midi, prepend=midi[:1]))
    ok = (f0 > 0) & (dm < 0.35)
    res = np.where(ok, dev, 0.0)
    res = np.convolve(res, np.ones(3) / 3, mode="same")  # 抽取抖動的 3-hop 平滑
    out = np.zeros(n_hops)
    n = min(n_hops, len(res))
    out[:n] = res[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--tag", default="direct30k")
    ap.add_argument("--model", default=f"{DDSP}/exp/combsub-harry/model_30000.pt")
    ap.add_argument("--key", default="0")
    ap.add_argument("--pe", default="parselmouth")
    ap.add_argument("--f0-min", default="65")
    ap.add_argument("--f0-max", default="800")
    ap.add_argument("--porta-ms", type=float, default=33.0,
                    help="級進滑音時間常數 ms（0=全落點）")
    ap.add_argument("--leap-snap", type=int, default=4,
                    help="超過此半音數的換音直接落點（0=全滑）")
    ap.add_argument("--vib-hz", type=float, default=VIB_HZ)
    ap.add_argument("--vib-phase", type=float, default=0.0,
                    help="vibrato 相位偏移（cycles，0–1）")
    ap.add_argument("--vib-onset-ms", type=float, default=0.0,
                    help="vibrato 起振延遲（每音先直後顫；0=關）")
    ap.add_argument("--f0-mode", choices=["target", "shift", "texture", "express"],
                    default="target",
                    help="shift＝他的真實 f0 逐 tick 移調（需 notes json 含 heard）；"
                         "texture＝target 骨架＋他的微音準紋理蓋印；"
                         "express＝target 骨架＋規則合成表現層（參數抄真人統計，"
                         "不需他的 f0，live 可移植）")
    ap.add_argument("--texture-gain", type=float, default=1.0)
    ap.add_argument("--expr-gain", type=float, default=1.0,
                    help="express 表現層整體倍率（0=退回純骨架）")
    ap.add_argument("--expr-seed", type=int, default=20260731,
                    help="express 表現層 RNG seed（固定＝render 可重現）")
    ap.add_argument("--respect-unvoiced", dest="respect_unvoiced",
                    action="store_true", default=True,
                    help="他的無聲幀（吸氣/擦音）注入 f0=0，只走 DDSP 噪音支路（預設開）")
    ap.add_argument("--no-respect-unvoiced", dest="respect_unvoiced",
                    action="store_false", help="關閉 voicing gate＝ 07-30 前的舊行為（對照用）")
    ap.add_argument("--uv-min-ms", type=float, default=40.0,
                    help="短於此的 unvoiced 空隙不 gate（去顆粒）")
    ap.add_argument("--uv-guard-ms", type=float, default=0.0,
                    help="gated 區間兩端內縮（0＝交給遲滯決定邊界）")
    ap.add_argument("--uv-hi", type=float, default=0.60,
                    help="voicing 信心遲滯上閾（進 voiced）")
    ap.add_argument("--uv-lo", type=float, default=0.35,
                    help="voicing 信心遲滯下閾（進 unvoiced）")
    ap.add_argument("--uv-anchor-ms", type=float, default=25.0,
                    help="gate 只准落在 harvest 判無聲處及此鄰域內"
                         "（-1＝不錨定，合議可在樂句中間自行開洞）")
    ap.add_argument("--uv-f0", type=float, default=1200.0,
                    help="unvoiced 幀注入的哨兵 f0（純防爆用，輸出會被透傳蓋掉；"
                         "0＝硬歸零，會偶發全刻度爆音）")
    ap.add_argument("--uv-passthrough", dest="uv_passthrough",
                    action="store_true", default=True,
                    help="unvoiced 段用他的原聲透傳，不用合成的（預設開）")
    ap.add_argument("--no-uv-passthrough", dest="uv_passthrough",
                    action="store_false",
                    help="關閉透傳＝無聲段仍由 DDSP 合成（對照用）")
    ap.add_argument("--uv-xfade-ms", type=float, default=15.0,
                    help="透傳邊界的升餘弦交叉淡化長度")
    ap.add_argument("--input-agc", dest="input_agc", action="store_true",
                    default=True,
                    help="轉換前把輸入位準拉到訓練參考、轉換後還原（預設開）")
    ap.add_argument("--no-input-agc", dest="input_agc", action="store_false",
                    help="關閉輸入 AGC＝模型直接吃他的原始位準（對照用）")
    ap.add_argument("--agc-ref", type=float, default=TRAIN_REF_RMS,
                    help="AGC 目標 rms（預設＝訓練 clips 實測參考位準）")
    ap.add_argument("--run", action="store_true", help="直接代跑 DDSP")
    a = ap.parse_args()

    # 1) take → 單聲道 44.1k 供 DDSP 讀（md5 要對得上，故落一份實體檔）
    x, sr = sf.read(a.take, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    in_path = os.path.join(HERE, "out", f"{a.tag}_in.wav")
    # AGC 只作用在「餵給 DDSP 的那份」；f0 抽取／voicing 判定／透傳一律用原始
    # x，輸出再乘回 1/agc_g，位準結構完全不動。
    if a.input_agc:
        x_in, agc_g = input_agc(x, a.agc_ref)
        print(f"input AGC: gain {agc_g.min():.2f}–{agc_g.max():.2f}× "
              f"(ref rms {a.agc_ref:.4f}, peak {np.abs(x_in).max():.3f})")
    else:
        x_in, agc_g = x, None
    sf.write(in_path, x_in, SR, subtype="PCM_16")

    # 2) 目標 f0 → 寫進 DDSP 的 f0 cache（檔名協定見 main.py:178）
    # cache 檔名協定（main.py:165-178）：hop_size 是浮點運算結果 → "512.0"，
    # md5 取輸入檔 bytes。名字打錯 → main.py 靜靜地自己 extract 他的原音高，
    # 還會存成正確名字污染後續執行（2026-07-26 踩過）。
    notes, tick, heard = load_notes(a.notes)
    md5 = hashlib.md5(open(in_path, "rb").read()).hexdigest()
    hop_name = float(HOP)
    cache = os.path.join(DDSP, "cache",
                         f"{a.pe}_{hop_name}_{a.f0_min}_{a.f0_max}_{md5}.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    n_hops = int(np.load(cache).shape[0]) if os.path.exists(cache) \
        else len(x) // HOP + 1
    if a.f0_mode == "shift":
        if heard is None:
            sys.exit("--f0-mode shift 需要 notes json 內含 heard（render_v3 新版才寫）")
        f0 = shift_f0(notes, heard, tick, n_hops, x, float(a.f0_min))
    elif a.f0_mode == "texture":
        skel = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                         vib_semi=0.0)   # 真 vibrato 在紋理裡，人工的關掉
        f0 = skel * 2 ** (a.texture_gain * texture_cents(x, n_hops) / 1200.0)
    elif a.f0_mode == "express":
        skel = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                         vib_semi=0.0)   # vibrato 由表現層生（不規則版）
        cents = expressive_cents(notes, tick, n_hops, a.leap_snap, a.expr_seed)
        f0 = skel * 2 ** (a.expr_gain * cents / 1200.0)
        print(f"express layer: cents sd {cents.std():.1f} "
              f"(gain {a.expr_gain}, seed {a.expr_seed})")
    else:
        f0 = target_f0(notes, tick, n_hops, a.porta_ms, a.leap_snap,
                       a.vib_hz, a.vib_phase, a.vib_onset_ms)
    vm = None
    if a.respect_unvoiced:
        vm = voicing_mask(x, n_hops, a.uv_min_ms, a.uv_guard_ms,
                          a.uv_hi, a.uv_lo, a.uv_anchor_ms)
        f0 = f0 * vm
        print(f"voicing gate: {100 * (1 - vm.mean()):.1f}% hops uv "
              f"(hi/lo {a.uv_hi}/{a.uv_lo}, min_uv {a.uv_min_ms:.0f}ms, "
              f"guard {a.uv_guard_ms:.0f}ms)")
    # native uv（extract_f0_uv 的 0）與 gate 的 0 一律換成哨兵：任何 f0=0 都會
    # 讓 combtooth 的 sinc 引數趨近 0/0 而偶發全刻度爆音（見 voicing_mask）。
    n_uv = int((f0 <= 0).sum())
    f0 = np.where(f0 > 0, f0, a.uv_f0)
    print(f"uv sentinel: {n_uv} hops ({100.0 * n_uv / n_hops:.1f}%) → {a.uv_f0:.0f}Hz")
    np.save(cache, f0.astype(np.float32), allow_pickle=False)
    print(f"f0 cache written: {os.path.basename(cache)}  "
          f"hops={n_hops} range={f0[f0>0].min():.0f}-{f0.max():.0f}Hz")

    out_path = os.path.join(HERE, "out", f"{a.tag}_angel.wav")
    cmd = [f"{DDSP}/venv/bin/python", f"{DDSP}/main.py",
           "-m", a.model, "-i", in_path, "-o", out_path,
           "-k", a.key, "-pe", a.pe, "-e", "false", "-d", "mps",
           "-fmin", a.f0_min, "-fmax", a.f0_max]
    print("\n" + " ".join(cmd) + "\n")
    if a.run:
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
        r = subprocess.run(cmd, cwd=DDSP, env=env)
        if r.returncode:
            sys.exit(r.returncode)
        ang, _ = sf.read(out_path, dtype="float64")
        if ang.ndim > 1:
            ang = ang[:, 0]
        if agc_g is not None:                    # 還原 AGC：位準結構回到原樣
            m = min(len(ang), len(agc_g))
            ang = ang[:m] / agc_g[:m]
        if a.uv_passthrough and vm is not None:
            ang, s = uv_passthrough(ang, x, vm, a.uv_xfade_ms)
            sf.write(out_path, ang, SR)          # 透傳後才是這張嘴的最終輸出
            print(f"uv passthrough: 原聲補回 {100 * (1 - vm.mean()):.1f}% hops "
                  f"(xfade {a.uv_xfade_ms:.0f}ms, take gain {s:.3f})")
        n = min(len(x), len(ang))
        mix = 0.5 * x[:n] + 0.8 * ang[:n]
        mix = mix / (np.max(np.abs(mix)) + 1e-12) * 0.9
        sf.write(os.path.join(HERE, "out", f"{a.tag}_mix.wav"), mix, SR)
        print(f"wrote out/{a.tag}_angel.wav / out/{a.tag}_mix.wav")


if __name__ == "__main__":
    main()
