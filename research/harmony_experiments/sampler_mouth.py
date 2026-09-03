"""sampler_mouth.py — 選項⓪ 取樣嘴（unit-sampling mouth）

長音不合成：天使的每顆音直接播放真實錄音的 sustain 單元。
v1（--plain）：靜態單元＋等功率硬接 → Harry 耳判「像鋼琴，沒有唱的過程」。
v2（預設）：歌唱手勢層——全部仍是她的真實錄音，只用時變 varispeed 彎音：
  - 起音 scoop：樂句頭從 -60c 滑入
  - legato 彎音過渡：換音時上一顆單元先滑進新音高（≤4 st），
    過渡完才在穩定段內交叉淡接到新單元（接縫藏在音事件之後）
  - 句尾音高微降＋自然收音
  - 呼吸跟隨：天使增益包絡跟隨 take 的 RMS（跟他一起做樂句）

管線：
  build   母音錄音 → 穩定長音單元庫（index JSON，只存 offset）
  render  notes.json（world_mouth）或 .targets.json（solo_host）
          → {tag}_angel.wav ＋ {tag}_mix.wav（0.5*take + 0.8*stem 同 world_mouth 配方）

用法（vcclient-dev env，於 harmony/ 下執行）：
  python sampler_mouth.py build --wavs A.wav B.wav --out out/units_girl.json
  python sampler_mouth.py render --units out/units_girl.json \
      --notes out/angel_v2_world2_notes.json --take ../../260722_harmony_brain/data/take.wav \
      --transpose 12 --tag sampler_girl_v2

server/ 零改動；單元選取固定 seed 輪替（跨聲部抽不同 take＝天然去相關）。
"""
import argparse, json, math, os
import numpy as np
import soundfile as sf
import soxr

from pitch import yin_f0

SR = 44100
HOP = 512
FRAME = 2048
XF = int(0.030 * SR)          # v1 事件接縫
MIN_UNIT_SEC = 0.35
STAB_ST = 0.5
RMS_GATE = 0.01

# v2 手勢參數
XF2 = int(0.120 * SR)         # legato 換單元交叉淡接（藏在穩定段內）
POST_MAX = int(0.180 * SR)    # 過渡後舊單元多唱多久才換人
SCOOP_C = -60.0               # 起音滑入起點（cents）
SCOOP_TAU = 0.040
GLIDE_TAU = 0.050             # legato 彎音時間常數
BEND_MAX_ST = 4               # 超過此音程改重新起音（真人大跳也會重出聲）
CUT_XF = int(0.040 * SR)      # 大跳時的短接縫
END_DROOP_C = -30.0
END_FADE = int(0.160 * SR)


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def load_mono(path):
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return x


def eq_pow_fades(n):
    t = np.linspace(0, np.pi / 2, n)
    return np.sin(t) ** 2, np.cos(t) ** 2  # in, out


def exp_toward(n, c0, target, tau):
    """指數趨近曲線（解析式 one-pole）：c(t)=target+(c0-target)e^(-t/tau)"""
    t = np.arange(n) / SR
    return target + (c0 - target) * np.exp(-t / tau)


# ---------------------------------------------------------------- build

def build(args):
    units, sources, breaths = [], [], []
    for path in args.wavs:
        x = load_mono(path)
        src_i = len(sources)
        sources.append(os.path.abspath(path))
        n_frames = (len(x) - FRAME) // HOP
        midi = np.full(n_frames, np.nan)
        rmsv = np.zeros(n_frames)
        for i in range(n_frames):
            fr = x[i * HOP:i * HOP + FRAME]
            rmsv[i] = np.sqrt(np.mean(fr ** 2))
            if rmsv[i] < RMS_GATE:
                continue
            f = yin_f0(fr.astype(np.float32), SR)
            if f:
                midi[i] = 69 + 12 * math.log2(f / 440.0)
        i = 0
        while i < n_frames:
            if np.isnan(midi[i]):
                i += 1
                continue
            j = i + 1
            while j < n_frames and not np.isnan(midi[j]) and abs(midi[j] - midi[j - 1]) <= 0.7:
                j += 1
            seg = midi[i:j]
            med = float(np.median(seg))
            ok = np.abs(seg - med) <= STAB_ST
            best_a = best_b = 0
            a = None
            for k, v in enumerate(list(ok) + [False]):
                if v and a is None:
                    a = k
                elif not v and a is not None:
                    if k - a > best_b - best_a:
                        best_a, best_b = a, k
                    a = None
            if (best_b - best_a) * HOP / SR >= MIN_UNIT_SEC:
                s0 = (i + best_a) * HOP
                s1 = (i + best_b) * HOP + FRAME
                sub = midi[i + best_a:i + best_b]
                m = float(np.median(sub))
                units.append({
                    "src": src_i, "s0": int(s0), "s1": int(s1),
                    "midi": m, "note": int(round(m)),
                    "rms": float(np.sqrt(np.mean(x[s0:s1] ** 2))),
                    "dur": round((s1 - s0) / SR, 3),
                })
            i = j
        # 換氣單元（任務3）：無聲、高於底噪、0.12–0.8s、緊鄰有聲段
        voiced_fr = ~np.isnan(midi)
        floor = float(np.percentile(rmsv, 20))
        cand = (~voiced_fr) & (rmsv > max(2 * floor, 0.0025)) & (rmsv < RMS_GATE * 1.5)
        ctx = int(2 * SR / HOP)
        i2 = 0
        while i2 < n_frames:
            if not cand[i2]:
                i2 += 1
                continue
            j2 = i2
            while j2 < n_frames and cand[j2]:
                j2 += 1
            dur = (j2 - i2) * HOP / SR
            if 0.12 <= dur <= 0.8 and \
                    voiced_fr[max(0, i2 - ctx):min(n_frames, j2 + ctx)].any():
                breaths.append({"src": src_i, "s0": int(i2 * HOP),
                                "s1": int(j2 * HOP + FRAME),
                                "rms": float(np.mean(rmsv[i2:j2])),
                                "dur": round(dur, 3)})
            i2 = j2
    out = {"sr": SR, "sources": sources, "units": units, "breaths": breaths}
    with open(args.out, "w") as f:
        json.dump(out, f)
    notes = sorted({u["note"] for u in units})
    durs = [u["dur"] for u in units]
    print(f"units={len(units)}  notes={notes}")
    print(f"dur median={np.median(durs):.2f}s max={max(durs):.2f}s  total={sum(durs)/60:.1f}min")
    print(f"breaths={len(breaths)}")


# ---------------------------------------------------------------- render 共用

def parse_notes(path):
    d = json.load(open(path))
    if "notes" in d:                       # world_mouth 格式
        return d["notes"], int(d["step_samps"])
    return d["targets"], int(round(d["tick_sec"] * SR))  # solo_host 格式


def pick_unit(lib_by_note, target, rng, last_id, need_len=None, cluster=None):
    """最近音單元選取。v5 消融加兩個約束（量測歸因 worklog 07-26 §B）：
    cluster＝樂句內音色一致（girl 版主犯：相鄰單元音色跳動 0.125）；
    need_len＝偏好不用循環的夠長單元（harry 版主犯：55% 事件需循環）。
    約束都是軟的——池子空了就逐層退讓。"""
    best_d, cands = None, []
    for note, us in lib_by_note.items():
        d = abs(note - target)
        if best_d is None or d < best_d:
            best_d, cands = d, list(us)
        elif d == best_d:
            cands += us
    pool = cands
    if cluster is not None:
        same = [u for u in pool if u.get("cl") == cluster]
        pool = same or pool
    if need_len:
        lng = [u for u in pool if (u["s1"] - u["s0"]) >= need_len]
        pool = lng or sorted(pool, key=lambda u: u["s1"] - u["s0"])[-3:]
    pool2 = [u for u in pool if u["_id"] != last_id] or pool
    return pool2[rng.integers(len(pool2))], best_d


def timbre_clusters(lib, srcs, k=5):
    """MFCC 均值向量 → k-means 音色分群（母音無標註的代理）。"""
    import librosa
    from scipy.cluster.vq import kmeans2
    feats = []
    for u in lib["units"]:
        x = srcs[u["src"]][u["s0"]:u["s1"]][:SR * 2]
        mf = librosa.feature.mfcc(y=x.astype(np.float32), sr=SR, n_mfcc=13)[1:]
        v = mf.mean(axis=1)
        feats.append(v / (np.linalg.norm(v) + 1e-12))
    feats = np.array(feats)
    _, labels = kmeans2(feats, k, minit="++", seed=0)
    for u, l in zip(lib["units"], labels):
        u["cl"] = int(l)


def fit_duration(y, n_out):
    """單元音訊貼合 n_out 樣本：不足→中段循環（等功率接縫），過長→截尾。"""
    if len(y) >= n_out:
        return y[:n_out].copy()
    a, b = min(int(0.05 * SR), len(y) // 4), len(y) - min(int(0.05 * SR), len(y) // 4)
    core = y[a:b]
    out = y.copy()
    fi, fo = eq_pow_fades(XF)
    while len(out) < n_out:
        seg = core.copy()
        n = min(XF, len(out), len(seg))
        mixed = out[-n:] * fo[-n:] + seg[:n] * fi[:n]
        out = np.concatenate([out[:-n], mixed, seg[n:]])
    return out[:n_out]


def make_events(notes, tick):
    events, i = [], 0
    while i < len(notes):
        if notes[i] is None:
            i += 1
            continue
        j = i
        while j < len(notes) and notes[j] == notes[i]:
            j += 1
        events.append({"midi": notes[i], "t0": i * tick, "n": (j - i) * tick,
                       "legato": j < len(notes) and notes[j] is not None})
        i = j
    return events


def smooth_env(x, hop, win_s):
    """零相位 RMS 包絡（對稱窗，離線不需因果）。回傳 hop 網格上的包絡。"""
    m = len(x) // hop
    rms = np.sqrt(np.mean(x[:m * hop].reshape(m, hop) ** 2, axis=1))
    w = np.hanning(max(3, int(win_s * SR / hop) | 1))
    return np.convolve(rms, w / w.sum(), mode="same")


def take_gain(take, n, floor=0.12, win_s=0.020):
    """咬字蓋印：take 的零相位快包絡（20ms 窗）→ 天使增益曲線。

    保留音節瞬態與子音斷點——她的母音跟著他的字走；零相位＝無滯後
    （v3 的因果追隨器有 +23ms 群延遲，音節帶相關被相位差吃掉）。
    floor 讓天使在他換氣時仍以低音量延音。"""
    hop = 256
    e = smooth_env(take, hop, win_s)
    env = np.interp(np.arange(n), np.arange(len(e)) * hop, e,
                    left=e[0] if len(e) else 0, right=e[-1] if len(e) else 0)
    ref = np.percentile(env, 95) + 1e-12
    return floor + (1 - floor) * np.clip(env / ref, 0, 1)


def flatten_texture(y, strength=1.0, win_s=0.050):
    """熨平載體：單元除以自身包絡（50ms 零相位）的 strength 次方。
    strength=1 全熨（v4：蓋印相關最高但被耳判「音色皮」——表情全死）；
    0.35 部分熨＝保留載體自然生命，讓出部分頭寸給咬字蓋印。"""
    if strength <= 0:
        return y
    hop = 256
    e = smooth_env(y, hop, win_s)
    env = np.interp(np.arange(len(y)), np.arange(len(e)) * hop, e,
                    left=e[0] if len(e) else 1, right=e[-1] if len(e) else 1)
    med = np.median(env[env > 1e-6]) + 1e-12
    factor = np.clip(med / np.clip(env, 1e-6, None), 0.25, 4.0) ** strength
    return y * factor


# ---------------------------------------------------------------- render v1

def render_plain(events, lib_by_note, srcs, med_rms, tick, n_ticks, rng):
    stem = np.zeros(n_ticks * tick + XF)
    fi, fo = eq_pow_fades(XF)
    shifts, used, last_id = [], set(), -1
    for ev in events:
        u, _ = pick_unit(lib_by_note, ev["midi"], rng, last_id)
        last_id = u["_id"]
        used.add(u["_id"])
        y = srcs[u["src"]][u["s0"]:u["s1"]]
        ratio = midi_to_hz(ev["midi"]) / midi_to_hz(u["midi"])
        shifts.append(1200 * math.log2(ratio))
        if abs(ratio - 1) > 1e-4:
            y = soxr.resample(y, SR, SR / ratio)
        y = fit_duration(y, ev["n"] + (XF if ev["legato"] else 0))
        y = y * (med_rms / (u["rms"] + 1e-12))
        y[:XF] *= fi
        y[-XF:] *= fo
        stem[ev["t0"]:ev["t0"] + len(y)] += y
    return stem[:n_ticks * tick], shifts, used, {"bends": 0, "cuts": 0, "phrases": 0}


# ---------------------------------------------------------------- render v2

def render_gesture(events, lib_by_note, srcs, med_rms, tick, n_ticks, rng, opts=None):
    """歌唱手勢層：scoop 起音、legato 彎音過渡、句尾收音。全 varispeed，零合成。"""
    opts = opts or {}
    stem = np.zeros(n_ticks * tick + XF2 + POST_MAX)
    shifts, used, last_id = [], set(), -1
    n_bend = n_cut = 0

    phrases, cur = [], []
    for ev in events:
        cur.append(ev)
        if not ev["legato"]:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    for ph in phrases:
        # 每顆音的進場點：bend 過渡時新單元晚 POST 進場（接縫藏在音事件後的穩定段）
        entries, bend_ok = [ph[0]["t0"]], [False]
        for i in range(1, len(ph)):
            iv = ph[i]["midi"] - ph[i - 1]["midi"]
            ok = abs(iv) <= BEND_MAX_ST
            post = min(POST_MAX, int(0.4 * ph[i]["n"])) if ok else 0
            entries.append(ph[i]["t0"] + post)
            bend_ok.append(ok)
        end = ph[-1]["t0"] + ph[-1]["n"]

        ph_cl = None
        for i, ev in enumerate(ph):
            need = int((ev["n"] + POST_MAX) * 1.1) if opts.get("prefer_long") else None
            u, _ = pick_unit(lib_by_note, ev["midi"], rng, last_id,
                             need_len=need, cluster=ph_cl)
            if i == 0 and opts.get("same_timbre"):
                ph_cl = u.get("cl")
            last_id = u["_id"]
            used.add(u["_id"])
            base = midi_to_hz(ev["midi"]) / midi_to_hz(u["midi"])
            shifts.append(1200 * math.log2(base))

            # 本段時間範圍（含頭部交叉淡接）
            xfc = 0 if i == 0 else (XF2 if bend_ok[i] else CUT_XF)
            xfc = min(xfc, entries[i] - (entries[i - 1] if i else 0)) if i else 0
            seg_s = entries[i] - xfc
            seg_e = (entries[i + 1] if i + 1 < len(ph) else end)
            if i + 1 < len(ph):
                seg_e += 0  # 尾部延伸由下一顆的 xfc 覆蓋：本段唱到對方進場點
                seg_e = entries[i + 1]
            L = seg_e - seg_s
            if L <= 0:
                continue

            # cents 曲線
            c = np.zeros(L)
            if i == 0:
                c += exp_toward(L, SCOOP_C, 0.0, SCOOP_TAU)
            elif bend_ok[i]:
                # 接棒曲線：舊單元滑到入場點時還差 iv*exp(-POST/tau) 沒滑完，
                # 新單元從同一位置出發同 tau 收斂 → 疊影期兩層音高一致。
                # （量測：過渡區 std 33c vs 穩定區 7c——「偏不穩」住在這裡）
                iv_c = (ev["midi"] - ph[i - 1]["midi"]) * 100.0
                post = entries[i] - ev["t0"]
                c0 = -iv_c * math.exp(-post / SR / GLIDE_TAU)
                c += exp_toward(L, c0, 0.0, GLIDE_TAU)
            if i + 1 < len(ph) and bend_ok[i + 1]:
                b = ph[i + 1]["t0"] - seg_s          # 過渡開始（本段座標）
                if 0 < b < L:
                    iv_c = (ph[i + 1]["midi"] - ev["midi"]) * 100.0
                    c[b:] = c[b:] - c[b] + exp_toward(L - b, c[b], iv_c, GLIDE_TAU)
                n_bend += 1
            elif i + 1 < len(ph):
                n_cut += 1
            if i == len(ph) - 1:
                d = min(END_FADE, L)
                c[L - d:] += exp_toward(d, 0.0, END_DROOP_C, END_FADE / SR / 3)

            # 時變 varispeed 讀取（真實錄音，只彎不合成）
            ratio = base * 2 ** (c / 1200.0)
            pos = np.cumsum(ratio)
            need = int(pos[-1]) + FRAME
            raw = srcs[u["src"]][u["s0"]:u["s1"]]
            raw = fit_duration(raw, need)
            y = np.interp(pos, np.arange(len(raw)), raw)
            y = flatten_texture(y, opts.get("flatten", 1.0))
            y *= med_rms / (np.sqrt(np.mean(y ** 2)) + 1e-12)

            # 振幅窗
            if i == 0:
                a = min(int(0.060 * SR), L)
                y[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
            elif xfc > 0:
                y[:xfc] *= eq_pow_fades(xfc)[0]
            if i == len(ph) - 1:
                d = min(END_FADE, L)
                y[L - d:] *= np.cos(np.linspace(0, np.pi / 2, d)) ** 2
            else:
                nxt_xfc = XF2 if bend_ok[i + 1] else CUT_XF
                d = min(nxt_xfc, L)
                y[L - d:] *= eq_pow_fades(d)[1]

            stem[seg_s:seg_s + L] += y

    return stem[:n_ticks * tick], shifts, used, \
        {"bends": n_bend, "cuts": n_cut, "phrases": len(phrases)}


def render_runs(events, lib_by_note, srcs, med_rms, tick, n_ticks, rng, opts=None):
    """v7 run 制：同一顆單元一路彎到底（真 legato、零接縫），只在大跳或
    彎太遠（>3 st）時才換單元。動機（worklog 07-26 §B 續）：v6 逐項排除後
    剩餘的客觀差距＝接縫數量本身——每音換單元＝34 個交叉淡接擾動；
    參照（DDSP）是單一連續源。她的單元中位 4.4s，多數樂句一顆就夠。"""
    opts = opts or {}
    stem = np.zeros(n_ticks * tick + XF2 + POST_MAX)
    shifts, used, last_id = [], set(), -1
    n_bend = n_seam = 0
    MAX_OFF_ST = 3          # 同單元最遠可彎（半音）

    phrases, cur = [], []
    for ev in events:
        cur.append(ev)
        if not ev["legato"]:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    phrase_bounds, _pe = [], 0   # (樂句起點, 前句終點)——換氣插入用
    for ph in phrases:
        phrase_bounds.append((ph[0]["t0"], _pe))
        _pe = ph[-1]["t0"] + ph[-1]["n"]

    for ph in phrases:
        # 切 runs：可用同一顆單元連續彎的音串
        runs, cur_r = [], [ph[0]]
        for k in range(1, len(ph)):
            iv = abs(ph[k]["midi"] - ph[k - 1]["midi"])
            off = abs(ph[k]["midi"] - cur_r[0]["midi"])
            if iv <= BEND_MAX_ST and off <= MAX_OFF_ST:
                cur_r.append(ph[k])
            else:
                runs.append(cur_r)
                cur_r = [ph[k]]
        runs.append(cur_r)

        entries, bend_ok = [runs[0][0]["t0"]], [False]
        for r in range(1, len(runs)):
            iv = abs(runs[r][0]["midi"] - runs[r - 1][-1]["midi"])
            ok = iv <= BEND_MAX_ST
            post = min(POST_MAX, int(0.4 * runs[r][0]["n"])) if ok else 0
            entries.append(runs[r][0]["t0"] + post)
            bend_ok.append(ok)
        ph_end = ph[-1]["t0"] + ph[-1]["n"]

        ph_cl = None
        for r, run in enumerate(runs):
            root = run[0]["midi"]
            need = (int(sum(ev["n"] for ev in run) * 1.25) + POST_MAX + XF2) \
                if opts.get("prefer_long") else None
            u, _ = pick_unit(lib_by_note, root, rng, last_id,
                             need_len=need, cluster=ph_cl)
            if r == 0 and opts.get("same_timbre"):
                ph_cl = u.get("cl")
            last_id = u["_id"]
            used.add(u["_id"])
            base = midi_to_hz(root) / midi_to_hz(u["midi"])
            shifts.append(1200 * math.log2(base))
            n_bend += len(run) - 1
            if r:
                n_seam += 1

            xfc = 0 if r == 0 else (XF2 if bend_ok[r] else CUT_XF)
            if r:
                xfc = max(0, min(xfc, entries[r] - entries[r - 1]))
            seg_s = entries[r] - xfc
            seg_e = entries[r + 1] if r + 1 < len(runs) else ph_end
            L = seg_e - seg_s
            if L <= 0:
                continue

            # cents 曲線（相對 root）：逐音 one-pole 級進；末端彎向下一 run
            offs = [(ev["midi"] - root) * 100.0 for ev in run]
            if r == 0:
                val = offs[0] + SCOOP_C
            elif bend_ok[r]:
                iv_c = (run[0]["midi"] - runs[r - 1][-1]["midi"]) * 100.0
                post = entries[r] - run[0]["t0"]
                val = offs[0] - iv_c * math.exp(-post / SR / GLIDE_TAU)
            else:
                val = offs[0]
            tail_b = L
            if r + 1 < len(runs) and bend_ok[r + 1]:
                tail_b = max(0, min(L, runs[r + 1][0]["t0"] - seg_s))
            c = np.empty(L)
            pos_c = 0
            for k, ev in enumerate(run):
                b_next = tail_b if k == len(run) - 1 \
                    else max(pos_c, min(tail_b, run[k + 1]["t0"] - seg_s))
                if b_next > pos_c:
                    c[pos_c:b_next] = exp_toward(b_next - pos_c, val, offs[k], GLIDE_TAU)
                    val = c[b_next - 1]
                    pos_c = b_next
            if tail_b < L:  # 彎向下一 run 的接棒橋
                nxt = (runs[r + 1][0]["midi"] - root) * 100.0
                c[tail_b:] = exp_toward(L - tail_b, val, nxt, GLIDE_TAU)
            elif pos_c < L:
                c[pos_c:] = val
            if r == len(runs) - 1:
                d = min(END_FADE, L)
                c[L - d:] += exp_toward(d, 0.0, END_DROOP_C, END_FADE / SR / 3)

            ratio = base * 2 ** (c / 1200.0)
            pos = np.cumsum(ratio)
            raw = srcs[u["src"]][u["s0"]:u["s1"]]
            raw = fit_duration(raw, int(pos[-1]) + FRAME)
            y = np.interp(pos, np.arange(len(raw)), raw)
            y = flatten_texture(y, opts.get("flatten", 1.0))
            y *= med_rms / (np.sqrt(np.mean(y ** 2)) + 1e-12)

            if r == 0:
                a = min(int(0.060 * SR), L)
                y[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
            elif xfc > 0:
                y[:xfc] *= eq_pow_fades(xfc)[0]
            if r == len(runs) - 1:
                d = min(END_FADE, L)
                y[L - d:] *= np.cos(np.linspace(0, np.pi / 2, d)) ** 2
            else:
                d = min(XF2 if bend_ok[r + 1] else CUT_XF, L)
                y[L - d:] *= eq_pow_fades(d)[1]

            stem[seg_s:seg_s + L] += y

    return stem[:n_ticks * tick], shifts, used, \
        {"bends": n_bend, "cuts": n_seam, "phrases": len(phrases),
         "phrase_bounds": phrase_bounds}


def add_breaths(stem, breaths, srcs, phrase_bounds, med_rms, rng):
    """樂句開頭前插入真實換氣聲。必須在咬字蓋印之後呼叫——take 在句間是
    靜音，蓋印的 floor 會把先插的換氣壓死。"""
    GAP_MIN = int(0.06 * SR)
    n_ins = 0
    for start, prev_end in phrase_bounds:
        gap = start - prev_end
        cands = [b for b in breaths
                 if (b["s1"] - b["s0"]) + GAP_MIN <= gap]
        if not cands:
            continue
        b = cands[rng.integers(len(cands))]
        y = srcs[b["src"]][b["s0"]:b["s1"]].copy()
        y *= (med_rms * 0.30) / (b["rms"] + 1e-12)
        f = min(int(0.020 * SR), len(y) // 4)
        y[:f] *= np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)
        p0 = start - int(0.025 * SR) - len(y)
        if p0 >= 0:
            stem[p0:p0 + len(y)] += y
            n_ins += 1
    return stem, n_ins


# ---------------------------------------------------------------- render 入口

def render(args):
    lib = json.load(open(args.units))
    srcs = [load_mono(p) for p in lib["sources"]]
    for k, u in enumerate(lib["units"]):
        u["_id"] = k
    lib_by_note = {}
    for u in lib["units"]:
        lib_by_note.setdefault(u["note"], []).append(u)
    med_rms = float(np.median([u["rms"] for u in lib["units"]]))

    notes, tick = parse_notes(args.notes)
    if args.transpose:
        notes = [None if m is None else m + args.transpose for m in notes]
    events = make_events(notes, tick)
    rng = np.random.default_rng(args.seed)

    if args.plain:
        stem, shifts, used, st = render_plain(
            events, lib_by_note, srcs, med_rms, tick, len(notes), rng)
    else:
        if not args.no_same_timbre:
            timbre_clusters(lib, srcs)
        opts = {"same_timbre": not args.no_same_timbre,
                "prefer_long": not args.no_prefer_long,
                "flatten": args.flatten}
        fn = render_gesture if args.per_note else render_runs
        stem, shifts, used, st = fn(
            events, lib_by_note, srcs, med_rms, tick, len(notes), rng, opts)

    take = load_mono(args.take)
    if not args.no_follow and not args.plain:
        stem *= take_gain(take, len(stem))       # 咬字蓋印
    if not args.plain and not args.no_breath and lib.get("breaths"):
        stem, nb = add_breaths(stem, lib["breaths"], srcs,
                               st.get("phrase_bounds", []), med_rms,
                               np.random.default_rng(args.seed + 1))
        print(f"breaths inserted: {nb}/{st.get('phrases', 0)} phrases")

    peak = np.max(np.abs(stem)) + 1e-12
    if peak > 1.0:
        stem = stem / peak * 0.95
    out_dir = os.path.dirname(args.notes) or "."
    sf.write(os.path.join(out_dir, f"{args.tag}_angel.wav"), stem, SR)

    n = min(len(take), len(stem))
    mix = 0.5 * take[:n] + 0.8 * stem[:n]        # 同 world_mouth 配方
    mix = mix / (np.max(np.abs(mix)) + 1e-12) * 0.9
    sf.write(os.path.join(out_dir, f"{args.tag}_mix.wav"), mix, SR)

    sh = np.abs(np.array(shifts))
    print(f"events={len(events)}  phrases={st['phrases']}  "
          f"bends={st['bends']} cuts={st['cuts']}  units used={len(used)}/{len(lib['units'])}")
    print(f"static shift |cents|: median={np.median(sh):.0f} p90={np.percentile(sh,90):.0f} "
          f"max={sh.max():.0f}  (>50c: {int((sh>50).sum())}/{len(sh)})")
    print(f"wrote {args.tag}_angel.wav / {args.tag}_mix.wav in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--wavs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--units", required=True)
    r.add_argument("--notes", required=True)
    r.add_argument("--take", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--transpose", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--plain", action="store_true", help="v1 行為（無手勢層）")
    r.add_argument("--no-follow", action="store_true", help="關閉咬字蓋印")
    r.add_argument("--flatten", type=float, default=0.35,
                   help="載體熨平強度 0–1（1=v4 全熨/音色皮；0=保留全部原始表情）")
    r.add_argument("--no-same-timbre", action="store_true", help="關閉樂句內音色一致約束")
    r.add_argument("--no-prefer-long", action="store_true", help="關閉夠長單元偏好")
    r.add_argument("--per-note", action="store_true", help="v5/v6 行為（每音換單元），A/B 用")
    r.add_argument("--no-breath", action="store_true", help="關閉換氣單元插入")
    args = ap.parse_args()
    {"build": build, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
