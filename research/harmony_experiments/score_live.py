"""score_live — 天使照譜唱、你當指揮（08-11 v3）

Harry 對 --rehearse 的三刀（worklog 08-11）＝本檔需求規格：
  「天使不知道我在哪一段」→ ScoreTracker v3 的 HMM 先驗改均勻＝全局定位，
    從任何一段開口它自己收斂到你的位置；收斂前天使不出聲（乾淨進場）。
  「表現卡在我身上」→ 天使的聲音材料**不再來自麥克風**：譜給 f0/音量、
    母音 units 循環（score_sing 管線）＝獨立歌手；你只給**位置與速度**。
  「雜亂」→ 特徵確定性＝跨窗一致性天生成立（不需要凍結網格/橋接/EMA），
    你不順天使照樣圓潤——只是等你。

  你唱 lead（真嗓直接進房間，不過機器）；天使：t1→sop3、t2→Alto-6、
  t4→bass1＝四部齊響。你停 >2.5s 天使淡出等你，開口再回來。

v3：樂句邊界進場＋速度鎖＝領導權合約「位置是他的、速度是譜的」。
  v2 死結（句內 PLL 追他、他聽天使調自己＝互相跟隨：conf 震盪、rate 飄）
  的根治：句內天使恆定譜速前進（無 PLL、無 rate EMA）；只在樂句邊界對位
  ——你唱進某句開頭（進場窗內）天使才進場，唱完該句就待命等下一句。
  停 >2.5s 淡出棄句，回來（或跳段）一樣從你所在的句首接。

v3.1：嘴部門控（mediapipe 讀嘴＝回授騙不了的第二感測器）。probe 實測
  開口度（唇距/臉高）分離 34×、活動量尾巴反而重疊 → 用開口度＋遲滯＋
  0.8s 寬限。接三點：進場多一 AND｜lastv 只在嘴開時更新（回授騙不了
  淡出）｜嘴閉時 observe(None)（天使回授不進 HMM）。鏡頭掛＝退回純
  音訊 v3（fail-open）；--mouth 0 關閉。

v3.2：聲部獨立織體支援（真譜實測 lead 休止 tick 的 91% 和聲仍有音，
  按 lead 句尾待命會斬斷和聲線）。三改：
  ① 待命點改 tutti 休止（四部全停；真譜 8 段）＝和聲唱完自己的線；
  ② 段太長（最長 212s）純速度鎖會積差 → 句首再同步：你每開新句，
    位置向你做一次有界修正（≤2 tick、每塊 ≤4 幀滑入）；偏 >8 tick
    ＝你跳段了 → 回待命重進場。仍是「位置他的、只在句首對」。
  ③ 棄句看譜：lead 譜上休止＝你的沉默是寫好的，天使照唱；譜上該唱
    而你停 >2.5s 才淡出棄句（每次譜面回到該唱，計時器重給 2.5s）。

v4（現 --wait；當時預設＝順序模式）：合唱團直覺——背譜按順序唱，不做全局定位。
  HMM 那套完整保留在 --explore（排練：從任何一段開口它自己找你）。
  ・段＝tutti 休止切的 8 段，永遠知道下一段是哪段（待命時終端打
    段號+Enter 跳段、r 回頭＝指揮用講的）。
  ・進場 cue：lead 段＝你開唱（音高要對得上句首±4 半音，含低八度）；
    無 lead 段（純和聲 intro）＝呼吸 cue（開口 ≥0.3s）或哼聲；
    無 lead 段之間自動接續（合唱團自己數短休止，不用重 cue）。
  ・段內＝速度鎖；句首對表：安靜 ≥0.5s 後開唱＝句首 onset，位置
    做一次有界修正（≤2 tick 滑入）。嘴＝呼吸 cue＋回授免疫，非監視。
  ・棄段後 cue 從最近句首接（不用從段頭重唱）。

v7（預設＝團員模式）：依賴反轉——團自主唱，你加入它（08-11 第一性拆解，
  成分一「音樂大於你、不靠你」）。v1–v4 的共同前提「天使繞著你轉」翻掉：
  ・lead 也是天使（t0→alto3-40k，同 score_sing）＝五部自主唱；開演 cue
    （呼吸/發聲）之後段自動接續到曲終，**你停團不停**——棄句/淡出等人刪除，
    音樂只在譜上寫休止的地方休止。
  ・你唱哪條線，那條線讓給你：你的音對到五線中最近者（八度折疊、要贏
    第二名 ≥2 半音、連 3 塊同判才換位）→ 該天使淡出讓位；你停 ~1.5s
    天使把線接回來唱。誤判 fail-soft＝沒人讓位（你跟天使 double）。
  ・分類掛嘴部門控後面（嘴閉不算你的音）＝天使自己的 lead 從喇叭繞回
    mic 不會騙到讓位（G34 教訓：能量域判不了回授，只有嘴判得了）。
  ・v4 伴唱模式保留在 --wait（A/B 對照）；--explore 行為不變（排練）。

v7.1：天使改預渲 stems 播放（live 首輪實測：旋律線在滑窗+SOLA live 渲染
  下貼譜 49%／亂 10%（正典 80%／1%）＝G37 換音接縫病×旋律 1–2 tick 換音；
  和聲線長音倖免（70%／0%）。v4 之前 lead 是他唱＝此病從未曝光）。
  團的唱法是譜定的＝live 根本不用渲染：開場把四部渲成正典 stems
  （score_sing 品質＝過耳的那個；磁碟快取，key=譜+聲部配置，第二次免渲），
  live 只做播放＋讓位淡出＋段行進；slew/跳段＝播放頭跳針＋20ms crossfade。
  順帶：渲染 RTF 歸零（xrun 消失）、快取命中時模型都不用載＝秒開。
  同思想前例：prerender_stems.py（08-01 Harry「優化天使、不是優化 live」）。

跑: venv/bin/python score_live.py --score scratchpad/xnn_all5.json \
      --line2 scratchpad/xnn_t34.json [--in-name "USB PnP"] [--out-name ...]
"""
import argparse
import hashlib
import json
import os
import threading
import time

import numpy as np
import parselmouth
import soundfile as sf
import torch

import spike_stream6 as S

SR, HOP, TICK = 44100, 512, 16

ap = argparse.ArgumentParser()
ap.add_argument("--score", required=True, help="lead/upper/lower json（t0/t1/t2）")
ap.add_argument("--line2", default="", help="第二份 json（t3/t4 用 lead/upper）")
ap.add_argument("--in-name", default="USB PnP")
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")
ap.add_argument("--block", type=float, default=0.24)
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--min-rest", type=int, default=2,
                help="樂句邊界的最短休止（tick；1 tick≈186ms）")
ap.add_argument("--entry-win", type=int, default=6,
                help="句首進場窗（tick）：定位落在句首這麼多 tick 內才進場")
ap.add_argument("--explore", action="store_true",
                help="探索模式（HMM 全局定位＝v3.2 行為）；預設＝順序模式")
ap.add_argument("--wait", action="store_true",
                help="v4 伴唱模式（lead 歸你、天使等 cue、你停就淡出）"
                     "＝v7 的 A/B 對照")
ap.add_argument("--voice", type=float, default=0.0,
                help="你的聲音直通輸出的 gain（0=關；喇叭回授風險自負）")
ap.add_argument("--latency", default="low",
                help="stream latency：low/high 或秒數（08-11 實測 high 在"
                     "本機 in 839ms/out 723ms＝房間晚 ~1.9s，改預設 low）")
ap.add_argument("--sblock", type=int, default=512,
                help="stream 塊大小（樣本）；處理粒度仍是 --block。"
                     "08-11 實測 latency 由 stream 塊主宰＝縮小這個才有感")
ap.add_argument("--mouth", type=int, default=1, help="嘴部門控（0=關）")
ap.add_argument("--cam", type=int, default=-1, help="鏡頭 index（-1=自動挑最亮）")
ap.add_argument("--dump", default="")
a = ap.parse_args()

d = json.load(open(a.score))
lead, t1, t2 = d["lead"], d["upper"], d["lower"]
t4 = json.load(open(a.line2))["upper"] if a.line2 else None

WAIT = a.wait or a.explore               # 舊行為：天使等你、你停團停

PARTS = [("t1", f"{S.DDSP}/exp/reflow-sop3/model_20000.pt", 1, 0.7, t1),
         ("t2", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt", 2, 0.85, t2)]
if t4:
    PARTS.append(("t4", f"{S.DDSP}/exp/reflow-bass1/model_32000.pt", 1, 1.0, t4))
LIDX = -1                                # lead 天使在 PARTS 的位置（v7 才有）
if not WAIT:
    # v7：lead 也是天使（alto3-40k；spk 3 避開 t2 的 spk 2），你唱它才讓位
    PARTS.insert(0, ("t0", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt",
                     3, 1.0, lead))
    LIDX = 0

# ── 譜 → 幀網格特徵 ──
NT = len(lead)
NF = NT * TICK


def line_feats(line):
    f0 = np.zeros(NF)
    vol = np.zeros(NF)
    for t, n in enumerate(line):
        if n is None:
            continue
        f0[t * TICK:(t + 1) * TICK] = 440.0 * 2 ** ((n - 69) / 12.0)
        vol[t * TICK:(t + 1) * TICK] = 0.06
    nz = f0 > 0
    if nz.any():
        f0 = np.interp(np.arange(NF), np.where(nz)[0], f0[nz])
    f0 = np.convolve(f0, np.ones(3) / 3, "same")
    vol = np.convolve(vol, np.ones(7) / 7, "same")
    return f0, vol


FEATS = [line_feats(l) for *_, l in PARTS]

# ── v7.1 stems：正典預渲＋磁碟快取（live 只播放，渲染 RTF 歸零）──
_ck = hashlib.md5((repr([(nm_, p_, sp_, g_) for nm_, p_, sp_, g_, _l in PARTS])
                   + repr([l for *_x, l in PARTS])
                   + a.vowel_src).encode()).hexdigest()[:10]
_spaths = [f"scratchpad/stems_{_ck}_{nm_}.wav" for nm_, *_x in PARTS]
STEMS = []
if all(os.path.exists(p_) for p_ in _spaths):
    print(f"stems 快取命中（{_ck}）＝不載模型、秒開", flush=True)
    for p_ in _spaths:
        w_, _ = sf.read(p_, dtype="float32")
        STEMS.append(w_)
else:
    print("預渲 stems（首次較久；之後快取秒開）…", flush=True)
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    UU = None
    for _pi, ((nm_, p_, sp_, g_, _l), (f0L_, volL_)) in enumerate(
            zip(PARTS, FEATS)):
        svc = S.Svc([(p_, 0.0, g_, sp_)], step=2, t_start=0.85)
        if UU is None:
            # 母音 units（同 score_sing）：take17 最長無聲…有聲段中段循環
            _f, _v, _m, uv0 = svc.prep(mic0[:60 * SR], -60.0, want_uv=True)
            runs, i = [], 0
            while i < len(uv0):
                if not uv0[i]:
                    j = i
                    while j < len(uv0) and not uv0[j]:
                        j += 1
                    runs.append((i, j))
                    i = j
                else:
                    i += 1
            a0, b0 = max(runs, key=lambda r: r[1] - r[0])
            mid = (a0 + b0) // 2
            with torch.no_grad():
                UU = svc.encode(
                    mic0[max(0, (mid - 40) * HOP):(mid + 40) * HOP])[:, 8:-8]
            NBANK = UU.size(1)
        outs = []
        CHF = 800                        # 每塊幀（~9.3s）＝score_sing 正典路徑
        torch.manual_seed(1234)
        t0_ = time.time()
        with torch.no_grad():
            for fo in range(0, NF, CHF):
                n_ = min(CHF, NF - fo)
                if n_ <= 1:
                    break
                vw_ = volL_[fo:fo + n_]
                vol_t = torch.from_numpy(vw_).float().to(
                    svc.device)[None, :, None]
                mask = S.upsample(
                    torch.from_numpy((vw_ > 0.005).astype(float)).float()
                    .to(svc.device)[None, :, None], HOP).squeeze(-1)
                un = UU[:, (fo + np.arange(n_)) % NBANK]
                au = svc.infer(np.zeros(n_ * HOP), 0.0, -60.0,
                               units_override=un,
                               feats=(f0L_[fo:fo + n_], vol_t, mask),
                               ratios=[None])[0]
                outs.append(au.cpu().numpy())
        w_ = np.concatenate(outs).astype("float32")
        sf.write(_spaths[_pi], w_, SR)
        STEMS.append(w_)
        print(f"  {nm_} 渲好（{time.time() - t0_:.0f}s）→ {_spaths[_pi]}",
              flush=True)
        del svc
STEMS = [np.pad(w_, (0, max(0, NF * HOP - len(w_))))[:NF * HOP]
         .astype("float32") for w_ in STEMS]

# ── 追蹤器：v3 ＋ 均勻先驗＝從任何一段進場 ──
from rehearse_ab import ScoreTracker  # noqa: E402
tr = ScoreTracker([None if n is None else n - 12 for n in lead], mode="v3")
# ↑ 追蹤用 lead 降八度＝他的音域；天使唱譜面絕對音高不受影響
tr.alpha = np.ones(NT) / NT              # 全局定位：起點不設先驗

# ── 樂句表：lead 休止 ≥ min-rest tick ＝邊界 ──
PHR = []                                 # (start, end) tick，end 不含
_i = 0
while _i < NT and lead[_i] is None:
    _i += 1
while _i < NT:
    _j = _i
    while _j < NT:
        if lead[_j] is not None:
            _j += 1
            continue
        _k = _j
        while _k < NT and lead[_k] is None:
            _k += 1
        if _k - _j >= a.min_rest:
            break
        _j = _k                          # 短休止＝句內換氣，不切
    PHR.append((_i, _j))
    _i = _j
    while _i < NT and lead[_i] is None:
        _i += 1
PIDX = np.full(NT, -1, dtype=int)        # tick → 樂句 index（休止＝-1）
for _pi, (_s, _e) in enumerate(PHR):
    PIDX[_s:_e] = _pi

# ── 段表：tutti 休止（lead＋全和聲部同停 ≥1 tick）＝天使的待命點 ──
_all = [lead] + [l for *_, l in PARTS]
SEG = []                                 # (start, end) tick
_i = 0
_rest = lambda t: all(ln[t] is None for ln in _all)  # noqa: E731
while _i < NT and _rest(_i):
    _i += 1
while _i < NT:
    _j = _i
    while _j < NT and not _rest(_j):
        _j += 1
    SEG.append((_i, _j))
    _i = _j
    while _i < NT and _rest(_i):
        _i += 1
SIDX = np.full(NT, -1, dtype=int)        # tick → 段 index
for _si, (_s, _e) in enumerate(SEG):
    SIDX[_s:_e] = _si
LFIRST = []                              # 各段 lead 首音 (tick, note)；無＝None
for _s, _e in SEG:
    _lt = next((t for t in range(_s, _e) if lead[t] is not None), None)
    LFIRST.append(None if _lt is None else (_lt, lead[_lt]))


def _match(n, ln):
    """他唱的 MIDI 音 vs 譜（原調或低八度），差 ≤4 半音算對。"""
    return ln is not None and min(abs(n - ln), abs(n - ln + 12)) <= 4


def _duck_vote(n, t):
    """v7 讓位判別：他唱的音對到哪條聲部線。八度折疊（±12）、最近者
    要 ≤2 半音且贏第二名 ≥2 半音，否則棄權（-1＝沒人讓位＝double）。"""
    ds = [1e9 if ln[t] is None else
          min(abs(n - ln[t] - o) for o in (-12, 0, 12))
          for *_x, ln in PARTS]
    w = int(np.argmin(ds))
    d2 = sorted(ds)
    return w if d2[0] <= 2 and d2[1] - d2[0] >= 2 else -1
print(f"譜 {NT} tick（{NF*HOP/SR:.0f}s）；樂句 {len(PHR)} 句／"
      f"段 {len(SEG)}；天使 {len(PARTS)} 部（stems {_ck}）",
      flush=True)

# ── 狀態 ──
blk = int(a.block * SR)
NB = blk // HOP                          # 每塊幀數
CF = int(0.02 * SR)                      # 播放頭跳針 crossfade（20ms）
st = {"spos": None, "lastv": time.time(), "conf": 0.0, "okc": 0,
      "until": 0.0, "seg": -1, "sync_ph": -1, "slew": 0.0, "exp": True,
      "jmpc": 0, "ending": False, "nxt": 0, "sq": 0, "vc": 0, "vt0": 0.0,
      "wstart": time.time(), "rsm": None, "xrun": 0, "skip": 0, "blkc": 0,
      "tickbuf": np.zeros(0), "die": False, "fade": 0.0,
      "duck": -1, "dvc": 0, "dcand": -1, "pend": None}
vfade = [1.0] * len(PARTS)               # 每聲部讓位淡出（1=天使唱）


def rearm():
    """回待命：清播放狀態，下一次進場從乾淨狀態起。"""
    st["spos"] = None
    st["ending"] = False
    st["sync_ph"] = -1
    st["slew"] = 0.0
    st["jmpc"] = 0
    st["rsm"] = None
    st["pend"] = None
    st["wstart"] = time.time()
    for vi_ in range(len(vfade)):
        vfade[vi_] = 1.0
    st["duck"] = -1
    st["dvc"] = 0
    st["dcand"] = -1
    with qlock:
        out_q.clear()                    # 沖掉輸出積壓＝延遲不跨段沉澱


# ── 嘴部門控（v3.1）：開口度＋遲滯＋0.8s 寬限；掛了就 fail-open ──
M = {"ok": a.mouth == 0, "on": False, "last_open": 0.0, "val": -1.0,
     "t_on": 0.0}


def _mouth_worker():
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path="scratchpad/face_landmarker.task"),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        lmk = vision.FaceLandmarker.create_from_options(opts)
        if a.cam >= 0:
            cap = cv2.VideoCapture(a.cam)
            assert cap.isOpened(), f"鏡頭 {a.cam} 打不開"
        else:
            # 挑最亮的鏡頭（index 0/2 常是虛擬相機＝全黑；同 mouth_probe）
            best = None
            for ci in range(3):
                c = cv2.VideoCapture(ci)
                if not c.isOpened():
                    continue
                time.sleep(0.4)
                okf, fr = c.read()
                b = float(fr.mean()) if okf else -1
                c.release()
                if best is None or b > best[1]:
                    best = (ci, b)
            assert best and best[1] > 10, f"找不到有畫面的鏡頭 {best}"
            print(f"[嘴] 鏡頭 index {best[0]}（亮度 {best[1]:.0f}）",
                  flush=True)
            cap = cv2.VideoCapture(best[0])
        t0 = time.time()
        last_face = time.time()
        last_det = 0.0
        while not st["die"]:
            okf, frame = cap.read()
            if not okf:
                time.sleep(0.01)
                continue
            now = time.time()
            if now - last_det < 0.05:        # landmark ≤20fps，省 CPU 給渲染
                continue
            last_det = now
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int((now - t0) * 1000))
            if res.face_landmarks:
                last_face = now
                f = res.face_landmarks[0]
                # 開口度＝內唇 13/14 距 ÷ 臉高（額 10–下巴 152）；probe：
                # 唱 p50 0.099 / p10 0.0054，閉嘴 p50 0.0029
                val = abs(f[13].y - f[14].y) / (abs(f[10].y - f[152].y)
                                                + 1e-9)
                M["val"] = val
                nv = val > (0.008 if M["on"] else 0.02)        # 遲滯
                if nv and not M["on"]:
                    M["t_on"] = now      # 開口瞬間＝呼吸 cue 的時間戳
                M["on"] = nv
                if M["on"]:
                    M["last_open"] = now
            # 0.8s 寬限（子音閉合/掉幀不斷線）；臉丟失 >1s＝fail-open
            M["ok"] = (now - M["last_open"] < 0.8) or (now - last_face > 1.0)
        cap.release()
    except Exception as e:
        M["ok"] = True
        print(f"[嘴] 門控失效（{e}）＝退回純音訊 v3", flush=True)


if a.mouth:
    threading.Thread(target=_mouth_worker, daemon=True).start()


def _stdin_jump():
    """跳段＝指揮用講的：終端打段號+Enter，r＝回頭從段 1。"""
    import sys
    for line in sys.stdin:
        w = line.strip().lower()
        if w == "r":
            k = 0
        elif w.isdigit() and 1 <= int(w) <= len(SEG):
            k = int(w) - 1
        else:
            continue
        rearm()
        st["nxt"] = k
        print(f"\n[跳段] 待命段 {k + 1}/{len(SEG)}，cue 進場", flush=True)


threading.Thread(target=_stdin_jump, daemon=True).start()
dmp = {"mic": [], "out": [], "pos": [], "mouth": []} if a.dump else None


def tick_note(x):
    """0.25s 音訊 → sounding MIDI note 或 None。"""
    try:
        p = parselmouth.Sound(x, SR).to_pitch(
            time_step=0.05, pitch_floor=65, pitch_ceiling=800)
        f = p.selected_array["frequency"]
        f = f[f > 0]
        if len(f) < 2:
            return None
        return int(round(69 + 12 * np.log2(np.median(f) / 440.0)))
    except Exception:
        return None


in_q, out_q = [], []
qlock = threading.Lock()


def worker():
    while not st["die"]:
        with qlock:
            pend = np.concatenate(in_q) if in_q else None
            in_q.clear()
        if pend is None:
            time.sleep(0.005)
            continue
        # 逐塊處理（一次進多塊時不能只渲一塊＝輸出斷流）
        chunks = [pend[i:i + blk] for i in range(0, len(pend) - blk + 1, blk)]
        rem = len(pend) - len(chunks) * blk
        if rem > 0:
            with qlock:
                in_q.insert(0, pend[-rem:])
        for chunk in chunks:
            _process(chunk)


def _process(pend):
        st["tickbuf"] = np.concatenate([st["tickbuf"], pend])[-4 * blk:]
        # 每塊做 tick 追蹤（塊 244ms ≈ 1.3 tick；粒度夠）
        note = tick_note(st["tickbuf"][-int(0.25 * SR):])
        mo = M["ok"]                     # 嘴部門控：閉嘴＝音高視為回授/雜訊
        pv_sq = st["sq"]                 # 本塊之前已安靜幾塊（onset 判定用）
        if note is not None and mo:
            if st["vc"] == 0:
                st["vt0"] = time.time()  # 這串發聲的起點
            st["vc"] += 1
            st["sq"] = 0
            st["lastv"] = time.time()
        else:
            st["sq"] += 1
            st["vc"] = 0
        if a.explore:
            # ── 探索模式：HMM 全局定位（v3.2）──
            tr.observe(note if mo else None)
            conf = float(np.max(tr.alpha)) if hasattr(tr, "alpha") else 1.0
            st["conf"] = conf
            p = tr.p
            st["okc"] = st["okc"] + 1 if conf > 0.22 else 0
            loc = (mo and p is not None and 0 <= p < NT and st["okc"] >= 3
                   and PIDX[p] >= 0 and p - PHR[PIDX[p]][0] < a.entry_win)
            # ↑ 定位有效且他正站在某句開頭（進場窗內）
            if st["spos"] is None and loc:
                ph = int(PIDX[p])
                sg = int(SIDX[p])
                st["spos"] = float(p * TICK)
                st["sync_ph"] = ph
                st["seg"] = sg
                st["until"] = float(min(NF - 2, (SEG[sg][1] + 1) * TICK))
                st["exp"] = True
                st["lastv"] = time.time()
                print(f"\n[進場] 段 {sg + 1}/{len(SEG)} "
                      f"句 {ph + 1}/{len(PHR)} tick {p}"
                      f"（conf {conf:.2f}）", flush=True)
            elif st["spos"] is not None and loc and PIDX[p] != st["sync_ph"]:
                # 句首再同步：一次有界修正，非連續 PLL
                err = p * TICK - st["spos"]
                if abs(err) > 8 * TICK:
                    st["jmpc"] += 1
                    if st["jmpc"] >= 3:  # 連 3 塊都在別處＝他真的跳段
                        rearm()
                        print(f"\n[跳段] 偏 {err / TICK:+.0f} tick，"
                              f"重新進場", flush=True)
                else:
                    st["jmpc"] = 0
                    st["slew"] = float(np.clip(err, -2 * TICK, 2 * TICK))
                    st["sync_ph"] = int(PIDX[p])
        else:
            # ── 順序模式：按順序走（v7 團員／--wait v4 伴唱）──
            if st["spos"] is None:
                k = st["nxt"]
                if k < len(SEG):
                    if not WAIT:
                        # v7：呼吸或發聲＝開演 cue；團從段首自主唱
                        at = SEG[k][0]
                        go = ((M["on"] and M["t_on"] > st["wstart"]
                               and time.time() - M["t_on"] > 0.3)
                              or st["vc"] >= 2)
                        adv = 0.0
                    elif st["rsm"] is None and LFIRST[k] is None:
                        # 無 lead 段＝指揮 cue：開口 ≥0.3s（呼吸）或哼聲
                        at = SEG[k][0]
                        go = ((M["on"] and M["t_on"] > st["wstart"]
                               and time.time() - M["t_on"] > 0.3)
                              or st["vc"] >= 2)
                        adv = 0.0
                    else:
                        # lead 段＝你開唱就是 cue（音高要對得上句首）
                        at = st["rsm"] if st["rsm"] is not None else (
                            LFIRST[k][0] if LFIRST[k] else SEG[k][0])
                        go = (st["vc"] >= 2 and note is not None
                              and _match(note, lead[at]))
                        adv = min((time.time() - st["vt0"]) * SR / HOP,
                                  4.0 * TICK)   # 錨點補回偵測延遲
                    if go:
                        st["spos"] = float(at * TICK + adv)
                        st["seg"] = k
                        st["until"] = float(min(NF - 2,
                                                (SEG[k][1] + 1) * TICK))
                        st["exp"] = True
                        st["lastv"] = time.time()
                        st["rsm"] = None
                        print(f"\n[進場] 段 {k + 1}/{len(SEG)} tick {at}",
                              flush=True)
            elif (st["vc"] == 1 and pv_sq >= 2 and note is not None
                  and (WAIT or st["duck"] == LIDX)):
                # 句首對表：安靜 ≥0.5s 後開唱＝句首 onset → 一次有界修正
                cands = [s_ for s_, _e2 in PHR
                         if abs(s_ * TICK - st["spos"]) <= 4 * TICK]
                if cands:
                    s_ = min(cands,
                             key=lambda x: abs(x * TICK - st["spos"]))
                    if _match(note, lead[s_]):
                        st["slew"] = float(np.clip(
                            s_ * TICK - st["spos"], -2 * TICK, 2 * TICK))
        # ── v7 讓位：你唱哪條線，那條線交給你；你停 ~1.5s 天使接回 ──
        if not WAIT and st["spos"] is not None:
            if note is not None and mo:
                v = _duck_vote(note, min(int(st["spos"] / TICK), NT - 1))
                if v >= 0 and v == st["duck"]:
                    st["dvc"] = 0        # 在位者續任
                elif v >= 0:
                    st["dvc"] = st["dvc"] + 1 if v == st["dcand"] else 1
                    st["dcand"] = v
                    if st["dvc"] >= 3:   # 連 ~0.7s 同判才換位（防瞬時誤判）
                        st["duck"] = v
                        st["dvc"] = 0
                        print(f"\n[讓位] {PARTS[v][0]} 交給你", flush=True)
            elif st["sq"] >= 6 and st["duck"] >= 0:
                print(f"\n[接手] {PARTS[st['duck']][0]} 天使接回",
                      flush=True)
                st["duck"] = -1
                st["dvc"] = 0
        # 棄句看譜：譜上該唱而你停 >2.5s 才棄；譜上休止＝你的沉默是寫好的
        # （v7 團員模式無此事：你停團不停，音樂只在譜面休止處休止）
        exp = (st["spos"] is not None
               and lead[min(int(st["spos"] / TICK), NT - 1)] is not None)
        if exp and not st["exp"]:
            st["lastv"] = time.time()    # 譜面回到該唱：計時器重給 2.5s
        st["exp"] = exp
        want = 1.0 if (st["spos"] is not None
                       and (not WAIT or not exp
                            or time.time() - st["lastv"] < 2.5)
                       ) else 0.0
        st["fade"] += np.clip(want - st["fade"], -0.15, 0.15)
        if st["spos"] is not None and want == 0.0 and st["fade"] < 0.01:
            rs = None
            if not a.explore:
                st["nxt"] = st["seg"]    # 棄段後 cue 從最近句首接，不重唱
                sg0 = SEG[st["seg"]][0]
                cands = [s_ for s_, _e2 in PHR
                         if sg0 <= s_ and s_ * TICK <= st["spos"]]
                rs = max(cands) if cands else None
            rearm()
            st["rsm"] = rs
            print(f"\n[淡出] 你停了，cue 我從句首接", flush=True)
        if st["spos"] is not None:
            sl = float(np.clip(st["slew"], -4, 4))   # 句首修正每塊 ≤4 幀滑入
            st["slew"] -= sl
            st["spos"] += NB + sl        # 速度鎖：段內恆定譜速，無 PLL
            if st["spos"] >= st["until"]:
                # 過段尾：本塊照渲，渲完套收尾淡出再待命
                st["spos"] = min(st["spos"], float(NF - 2))
                st["ending"] = True
        st["blkc"] += 1
        if st["spos"] is None or st["fade"] < 0.01:
            y = np.zeros(blk, dtype="float32")
        else:
            # ── v7.1 播放：stems 切片＋讓位淡出（渲染已離線，RTF≈0）──
            end_s = int(st["spos"]) * HOP
            lo_s = end_s - blk
            y = np.zeros(blk, dtype="float32")
            for vi in range(len(PARTS)):
                w = np.zeros(blk, dtype="float32")
                seg_ = STEMS[vi][max(0, lo_s):max(0, end_s)]
                if len(seg_):
                    w[blk - len(seg_):] = seg_   # 譜頭前不足補零
                # 讓位淡出：0.3/塊（~0.8s 全程）、塊內線性＝無拉鏈聲
                vt_ = 0.0 if vi == st["duck"] else 1.0
                vf0 = vfade[vi]
                vfade[vi] = vf0 + float(np.clip(vt_ - vf0, -0.3, 0.3))
                if vf0 != 1.0 or vfade[vi] != 1.0:
                    w = w * np.linspace(vf0, vfade[vi], blk)
                y += w
            if st["pend"] is not None and lo_s != st["pend"]:
                # 播放頭跳針（slew/進場跳位）＝20ms crossfade 防爆音
                cont = np.zeros(CF, dtype="float32")
                for vi in range(len(PARTS)):
                    s2 = STEMS[vi][max(0, st["pend"]):max(0, st["pend"]) + CF]
                    if len(s2):
                        cont[:len(s2)] += s2 * vfade[vi]
                r_ = np.linspace(0, 1, CF, dtype="float32")
                y[:CF] = cont * (1 - r_) + y[:CF] * r_
            st["pend"] = end_s
            y = (np.clip(y * a.gain * st["fade"], -1, 1)).astype("float32")
            if st["ending"]:
                nx = st["seg"] + 1
                if nx < len(SEG) and (not WAIT or (
                        not a.explore and LFIRST[nx] is None)):
                    # v7 全段自動接續（團自主唱到曲終）；
                    # --wait 只接無 lead 段＝合唱團自己數短休止，不重 cue
                    st["ending"] = False
                    st["seg"] = nx
                    st["spos"] = float(SEG[nx][0] * TICK)
                    st["until"] = float(min(NF - 2,
                                            (SEG[nx][1] + 1) * TICK))
                    print(f"\n[接續] 段 {nx + 1}/{len(SEG)}（和聲自走）",
                          flush=True)
                else:
                    y *= np.linspace(1, 0, len(y), dtype="float32")
                    lastsg = st["seg"]
                    rearm()
                    st["nxt"] = nx
                    if nx >= len(SEG):
                        print(f"\n[曲終] 段 {lastsg + 1}/{len(SEG)} 唱完"
                              f"（r+Enter 回頭）", flush=True)
                    else:
                        print(f"\n[待命] 段 {lastsg + 1}/{len(SEG)} 唱完，"
                              f"cue 段 {nx + 1} 進場", flush=True)
        if a.voice > 0:
            # 你的聲音直通（不吃 fade：天使沉默時你照樣出聲）
            y = np.clip(y + pend[:len(y)] * a.voice, -1, 1).astype("float32")
        with qlock:
            out_q.append(y)
            # 積壓上限 2 塊：卡頓＝跳拍趕上（合唱團的絆倒），不當延遲沉澱
            # （worker 輸入時鐘驅動＝積壓永不自排；08-11 buf 軌跡實證）
            while sum(len(q_) for q_ in out_q) > 2 * blk:
                out_q.pop(0)
                st["skip"] += 1
        if dmp is not None:
            dmp["mic"].append(pend.copy())
            dmp["out"].append(y.copy())
            dmp["pos"].append(st["spos"] if st["spos"] is not None else -1)
            dmp["mouth"].append((M["val"], 1.0 if mo else 0.0))


OB = [np.zeros(0, dtype="float32")]      # 輸出 ring：worker 塊 → 小塊串流


def cb(indata, outdata, frames, tinfo, status):
    if status:
        st["xrun"] += 1                  # under/overflow 計數＝低延遲的代價表
    with qlock:
        in_q.append(indata[:, 0].copy())
        buf = OB[0]
        while out_q and len(buf) < frames:
            buf = np.concatenate([buf, out_q.pop(0)])
    if len(buf) >= frames:
        outdata[:, 0] = buf[:frames]
        outdata[:, 1] = buf[:frames]
        OB[0] = buf[frames:]
    else:
        outdata.fill(0)
        if len(buf):
            outdata[:len(buf), 0] = buf
            outdata[:len(buf), 1] = buf
        OB[0] = np.zeros(0, dtype="float32")


import sounddevice as sd  # noqa: E402

threading.Thread(target=worker, daemon=True).start()
_mode = ("探索模式：唱 lead，任何一段都行" if a.explore else
         "伴唱模式：cue 段 1（開口或哼聲）" if a.wait else
         "團員模式：呼吸/發聲＝開演，團自主唱到曲終；唱哪條線哪條讓給你")
print(f"開流 in={a.in_name!r} out={a.out_name!r}"
      f"（{_mode}；跳段＝段號+Enter、r＝回頭；Ctrl-C 結束）", flush=True)
st["wstart"] = time.time()               # 呼吸 cue 只認開流之後的開口
try:
    try:
        _lat = float(a.latency)
    except ValueError:
        _lat = a.latency
    with sd.Stream(samplerate=SR, blocksize=a.sblock, channels=(1, 2),
                   device=(a.in_name, a.out_name), dtype="float32",
                   latency=_lat, callback=cb):
        while True:
            time.sleep(2)
            if st["spos"] is not None:
                _t = min(int(st["spos"] / TICK), NT - 1)
                _ph = int(PIDX[_t])
                with qlock:
                    _bl = sum(len(q_) for q_ in out_q) + len(OB[0])
                print(f"pos {st['spos']/TICK:6.0f}/{NT} tick  "
                      f"段 {st['seg'] + 1}/{len(SEG)} "
                      f"句 {_ph + 1 if _ph >= 0 else '—'}/{len(PHR)}  "
                      f"conf {st['conf']:.2f}  fade {st['fade']:.1f}  "
                      f"你 {PARTS[st['duck']][0] if st['duck'] >= 0 else '—'}  "
                      f"嘴 {'開' if M['ok'] else '閉'}  "
                      f"buf {_bl / SR * 1000:3.0f}ms  "
                      f"xrun {st['xrun']}  skip {st['skip']}", flush=True)
except KeyboardInterrupt:
    st["die"] = True
    if dmp is not None and dmp["mic"]:
        sf.write(a.dump + "_mic.wav", np.concatenate(dmp["mic"]), SR)
        sf.write(a.dump + "_angel.wav", np.concatenate(dmp["out"]), SR)
        np.save(a.dump + "_pos.npy", np.array(dmp["pos"]))
        np.save(a.dump + "_mouth.npy", np.array(dmp["mouth"]))
        print(f"\ndump: {a.dump}_mic/_angel.wav + _pos/_mouth.npy",
              flush=True)
    print("bye")
