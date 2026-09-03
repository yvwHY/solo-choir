"""score_align.py — 把他唱的那一句對到譜上（08-03，MIDI 和聲的對位側）

`score_notes.py` 產出的是**譜自己的時間軸**。要拿它當天使的音符線，必須先回答
兩件事：他剛唱的那句是譜的哪一段、以及他的節奏怎麼映射到譜的節奏。

**為什麼這裡有機會，而 08-01 的追蹤器死了**（命中 37.9%、素材天花板 49.4%
→ 排練預渲染的 live 半場判死）：那個追蹤器必須**因果且即時**——天使要跟他
同時唱，所以得邊唱邊猜位置，猜錯當場就是錯音。應答式完全不同：

  - 對位發生在**他唱完之後**，整句在手 → 可以看未來，DTW 不是線上演算法
  - 譜是**確定的**，不像「他上一次唱的同一首」會自己漂（那正是 49.4% 天花板
    的來源：兩遍唱得不一樣，對得再準也對不到）
  - 樂句邊界是**換氣**，通常就是譜的樂句邊界＝搜尋起點本來就對得差不多

所以這裡用 **subsequence DTW**（在譜上自由起訖，因為他唱的是整首的一小段），
配一個跨句游標（上一句停在哪，下一句就從那裡附近找＝應答式天然的單調前進）。

移調不猜死：對每個候選半音位移各跑一次，取代價最低的（他實際唱的調不見得等
於 score_notes 自動移調的結果）。

**信心不足就退回腦**：對不上的時候寧可讓 BrainV3 即興，也不要把錯的和聲貼上
去。這條退路是這個功能能不能上台的前提，不是保險。

Run（自我驗證，同 take 兩遍當「譜 vs 現唱」）:
  .../python score_align.py --selftest
"""
import numpy as np

REST = 2.0          # 一邊有聲一邊休止的代價（半音為單位）
CAP = 6.0           # 音高距離上限：對不上的離群音不該主導整條路徑
BIG = 1e9


def _cost(a, b):
    """兩個 sounding MIDI（None＝休止）之間的距離。"""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return REST
    return min(abs(a - b), CAP)


def dtw(sung, score, free_ends=True, skip=0.5):
    """subsequence DTW：sung 整條必須用完，score 可以只用中間一段。

    回傳 (cost_per_tick, j_of_i)——j_of_i[i] 是他第 i 個 tick 對到譜的第幾格。
    步進允許 (1,1)/(1,0)/(0,1)＝他可以唱快唱慢、也可以跳過譜上的音，但**非對角
    步要付 skip 的代價**。沒有這個懲罰會塌掉：第一版實測有一句把 23 個 tick 全
    黏到譜上 2 格（代價 0.00＝它找到一段休止就把整句塞進去），命中率照樣印 100%
    ——路徑退化不會被代價抓到，只會被步進形狀抓到。"""
    n, m = len(sung), len(score)
    if n == 0 or m == 0:
        return BIG, []
    D = np.full((n + 1, m + 1), BIG)
    D[0, 0] = 0.0
    if free_ends:
        D[0, :] = 0.0                     # 譜上任何一格都可以是起點
    ptr = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        ci = sung[i - 1]
        for j in range(1, m + 1):
            c = _cost(ci, score[j - 1])
            d, k = D[i - 1, j - 1], 0     # 斜：一個 tick 對一格
            if D[i - 1, j] + skip < d:
                d, k = D[i - 1, j] + skip, 1   # 直下：他拖長（譜不動）
            if D[i, j - 1] + skip < d:
                d, k = D[i, j - 1] + skip, 2   # 右移：他唱快（跳過譜的格）
            D[i, j] = d + c
            ptr[i, j] = k
    j_end = int(np.argmin(D[n, 1:])) + 1 if free_ends else m
    total = D[n, j_end]
    j_of_i = [0] * n
    i, j = n, j_end
    while i > 0:
        j_of_i[i - 1] = j - 1
        k = ptr[i, j]
        if k == 0:
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
        if j < 1:                          # 撞到左界：剩下的都黏在第一格
            while i > 0:
                j_of_i[i - 1] = 0
                i -= 1
            break
    return total / max(1, n), j_of_i


def align(sung, score, cursor=0, back=8, ahead=None, offsets=range(-12, 13)):
    """他這一句 → 譜上的對應位置。

    cursor＝上一句對到哪。**應答式天生單調前進**（他從頭唱到尾），所以窗只往
    回開 back 格（預設 8＝1.5s，容忍上一句尾巴估過頭），不是隨便往回找——第一版
    back=40 實測讓第 3 句跳回第 2 句之前、第 6 句跳回第 5 句之前，重複段一多就
    會被拉走。窗也不能不開——長曲子每句都掃全譜既慢又更容易被遠處的重複段
    拉走（重複段歧義正是 08-01 追蹤器的已知痛點）。

    回傳 dict：j_of_i / cost / offset / start / end。cost 是每 tick 的平均代價，
    也是「信心」——大於門檻就該退回腦。"""
    lo = max(0, cursor - back)
    hi = len(score) if ahead is None else min(len(score), cursor + ahead)
    win = score[lo:hi]
    best = None
    for o in offsets:
        sh = [None if p is None else p + o for p in win]
        c, path = dtw(sung, sh)
        if best is None or c < best[0]:
            best = (c, path, o)
    c, path, o = best
    j = [p + lo for p in path]
    return {"j_of_i": j, "cost": c, "offset": o,
            "start": j[0] if j else lo, "end": j[-1] if j else lo}


def map_parts(a, score_parts, keys=("upper", "lower")):
    """把對位結果套到其他聲部：他第 i 個 tick 唱的時候，天使唱譜上同一格的音。
    移調要跟著 lead 一起走，否則和聲關係會歪。"""
    j = a["j_of_i"]
    return {k: [None if score_parts[k][x] is None else score_parts[k][x] + a["offset"]
                for x in j] for k in keys}


def confidence(sung, score, a):
    """0–1 的信心，用來決定要不要退回腦。

    **代價（cost）不能拿來當信心**——合成驗收實測：對不好的那批代價中位 0.15，
    比對得好的 0.31 還低。因為失敗的典型長相是「塌到休止段」，而休止對休止的
    代價是 0：路徑越退化，代價越漂亮。這是我自己第一版寫錯的地方，留著當警告。

    改看三件在失敗時會一起壞掉的事：
      agree  他唱的音與譜上對應格差 ≤1 半音的比例（只算兩邊都有聲的）
      onrest 他唱到的音落在譜的休止上的比例（塌陷的直接指紋）
      slope  譜前進的格數 ÷ 他的 tick 數，正常 ≈1（塌陷會遠小於 1）
    """
    j = a["j_of_i"]
    o = a["offset"]
    vv = miss = onrest = 0
    for i, p in enumerate(sung):
        if p is None:
            continue
        q = score[j[i]]
        if q is None:
            onrest += 1
            continue
        vv += 1
        miss += abs(p - (q + o)) > 1.0
    n_v = vv + onrest
    agree = (vv - miss) / max(1, vv)
    onrest_r = onrest / max(1, n_v)
    slope = (j[-1] - j[0] + 1) / max(1, len(j))
    conf = agree * (1 - onrest_r) * min(1.0, slope / 0.6)
    return {"conf": conf, "agree": agree, "onrest": onrest_r, "slope": slope}


def hit_rate(sung, score_lead, a, tol=1.0):
    """對位後，他實際唱的音與譜上對應格的音差 ≤tol 半音的比例。

    ⚠ 讀法警告：DTW 最小化的就是這個量，所以它**天生偏高**，不能單獨當成
    「對位成功」的證據。它的用途是跟 oracle（下面）比——差距才是資訊。"""
    ok = tot = 0
    for i, p in enumerate(sung):
        if p is None:
            continue
        q = score_lead[a["j_of_i"][i]]
        tot += 1
        ok += q is not None and abs(p - (q + a["offset"])) <= tol
    return ok / max(1, tot), tot


def oracle_rate(sung, score_lead, offset, lo=0, hi=None, tol=1.0):
    """天花板：不管時序、每個唱到的音都去譜的窗裡找最好的一格。
    對不上不是對位演算法的錯、而是素材本身就沒有那個音時，這個數字會低。"""
    hi = len(score_lead) if hi is None else hi
    cand = [q + offset for q in score_lead[lo:hi] if q is not None]
    if not cand:
        return 0.0
    cand = np.array(cand)
    hit = [np.min(np.abs(cand - p)) <= tol for p in sung if p is not None]
    return float(np.mean(hit)) if hit else 0.0


def _selftest():
    """同一首歌的兩遍（pair06 take1/take2）＝「譜 vs 他現唱」的真實難度：
    take1 的音符線當譜，take2 當他現在唱的。這正是 08-01 即時追蹤器跑的同一份
    素材（當時 A 37.9%／素材天花板 49.4%），所以兩個數字可以直接對照——差別
    只在**離線整句 DTW vs 即時因果追蹤**。"""
    import sys
    import time
    from pathlib import Path
    import soundfile as sf
    import torch
    H = Path(__file__).parent
    sys.path.insert(0, str(H))
    import respond2 as R
    from live_v3 import EarV3
    from pitch import SR

    base = ("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
            "SoloChoirCode/260730_recording/pairs")

    def lead_of(path):
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        x = np.ascontiguousarray(x[:, 0])
        assert sr == SR
        torch.manual_seed(1234)
        ear = EarV3(indep=0.0, stab=2, key=3)
        ear.v2t.shift = 0
        out = []
        for s, e in R.find_phrases(x, 0.35, 0.5):
            lead, _, _ = R.phrase_notes(ear, x[s:e])
            out.append(lead)
        return out

    print("抽音符線……", flush=True)
    score_ph = lead_of(f"{base}/pair06_take1.wav")     # 當「譜」
    sung_ph = lead_of(f"{base}/pair06_take2.wav")      # 當「他現在唱的」
    score = [p for ph in score_ph for p in ph]         # 譜是連續的一整首
    print(f"譜 {len(score)} ticks（take1 {len(score_ph)} 句）｜"
          f"現唱 {len(sung_ph)} 句（take2）")

    cursor, hits, oras, costs, t0 = 0, [], [], [], time.perf_counter()
    for i, sung in enumerate(sung_ph):
        a = align(sung, score, cursor=cursor,
                  ahead=cursor + 4 * len(sung) + 60)
        h, n = hit_rate(sung, score, a)
        lo = max(0, a["start"] - 10)
        o = oracle_rate(sung, score, a["offset"], lo, a["end"] + 10)
        cursor = a["end"]
        hits.append(h * n)
        oras.append(o * n)
        costs.append(a["cost"])
        print(f"  句{i+1:2d} {len(sung):3d} ticks → 譜 [{a['start']:3d},{a['end']:3d}]"
              f" 位移 {a['offset']:+3d}｜代價 {a['cost']:.2f}｜命中 {h*100:3.0f}%"
              f"（該窗天花板 {o*100:3.0f}%）")
    ntot = sum(len([p for p in s if p is not None]) for s in sung_ph)
    print(f"\n合計：命中 {100*sum(hits)/ntot:.1f}%｜素材天花板 "
          f"{100*sum(oras)/ntot:.1f}%｜每句 {1000*(time.perf_counter()-t0)/len(sung_ph):.0f}ms")
    print("參考（08-01 即時因果追蹤器、同素材）：A 37.9%／素材天花板 49.4%")
    print("⚠ 三個讀法警告，缺一個就會把這些數字讀成假的：")
    print("  ① 命中率是 DTW 自己最小化的量＝天生偏高，不能單獨當成功證據。")
    print("  ② **天花板口徑與 08-01 不同**：這裡的 oracle 是「窗內任一格」，"
          "比 08-01 的逐 tick 口徑寬鬆得多 → 兩個天花板數字不可直接相比，"
          "可比的只有『同素材下離線整句 vs 即時因果』這個對照本身。")
    print("  ③ 真正沒被量到的是**節奏**：命中率只看音高對不對，"
          "天使的和聲落在哪個時間點要耳朵判。")


def _synth(trials=200, seed=20260803):
    """合成驗收：**有標準答案**的對位測試。

    take1/take2 那種「同一首唱兩遍」驗不了對位器——兩遍是兩次不同的即興，
    根本沒有正確答案，量到的東西混了「對位準不準」與「兩遍本來就不一樣」。
    這裡反過來做：從譜上切一段，**已知**地把它扭曲成「他唱的樣子」，再看對位
    器能不能把每個 tick 送回它原本的位置。

    扭曲的四件事都照真人來：唱快唱慢（局部 0.7–1.4×）、音準抖（±0.5 半音、
    偶爾整個音抓錯 1 半音）、漏字/換氣（隨機變休止）、調不對（整體位移）。
    """
    import glob
    import sys
    from pathlib import Path
    H = Path(__file__).parent
    sys.path.insert(0, str(H))
    rng = np.random.default_rng(seed)
    fs = sorted(glob.glob("/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/"
                          "SoloChoirCode/260723_corpus/cpdl_partsongs/*.mid"))
    import score_notes as SN
    from score_notes import TICK_S
    scores = []
    for f in fs:
        if len(scores) >= 40:
            break
        try:
            out, _ = SN.build(f)
        except Exception:
            continue
        if sum(p is not None for p in out["lead"]) > 60:
            scores.append(out)
    print(f"譜庫 {len(scores)} 首（CPDL，真編曲）")

    errs, exact, costs, confs, harm = [], [], [], [], []
    for t in range(trials):
        full = scores[rng.integers(len(scores))]
        sc = full["lead"]
        a0 = int(rng.integers(0, max(1, len(sc) - 40)))
        b0 = min(len(sc), a0 + int(rng.integers(12, 40)))
        off = int(rng.integers(-5, 6))
        sung, truth = [], []
        j = a0
        while j < b0:
            rep = 1 + (rng.random() < 0.25) - (rng.random() < 0.25)   # 0/1/2
            for _ in range(max(0, rep)):
                p = sc[j]
                if p is not None:
                    p = p + off + int(rng.random() < 0.10) * rng.choice([-1, 1])
                    if rng.random() < 0.08:
                        p = None                      # 換氣／漏掉
                sung.append(p)
                truth.append(j)
            j += 1
        if len(sung) < 8:
            continue
        a = align(sung, sc, cursor=a0, ahead=a0 + 4 * len(sung) + 60)
        e = np.abs(np.array(a["j_of_i"]) - np.array(truth))
        errs.append(np.median(e))
        exact.append(float(np.mean(e <= 1)))
        costs.append(a["cost"])
        confs.append(confidence(sung, sc, a))
        # **真正該量的**：聽眾聽到的是和聲，不是索引。對到重複段時位置雖然
        # 「錯」，但那裡的和聲往往一樣 → 位置誤差高估了實際傷害。
        got = map_parts(a, full)
        agree = []
        for i, jt in enumerate(truth):
            for k in ("upper", "lower"):
                t_, g_ = full[k][jt], got[k][i]
                if t_ is None and g_ is None:
                    agree.append(1.0)
                elif t_ is None or g_ is None:
                    agree.append(0.0)
                else:
                    agree.append(float((t_ + a["offset"] - g_) % 12 == 0))
        harm.append(float(np.mean(agree)) if agree else 0.0)
    errs, exact, costs = np.array(errs), np.array(exact), np.array(costs)
    print(f"{len(errs)} 次試驗：位置誤差中位 **{np.median(errs):.1f} tick**"
          f"（{np.median(errs)*TICK_S*1000:.0f}ms）｜p90 {np.percentile(errs,90):.1f}")
    print(f"落在 ±1 tick 內的比例：中位 {100*np.median(exact):.0f}%"
          f"｜整體 {100*exact.mean():.0f}%")
    print(f"代價（信心）中位 {np.median(costs):.2f}  p90 {np.percentile(costs,90):.2f}")
    good = exact > 0.7
    print(f"對得好的（±1 tick 佔 >70%）佔 {100*good.mean():.0f}%")
    print(f"  代價 cost      好 {np.median(costs[good]):.2f} / "
          f"壞 {np.median(costs[~good]):.2f}  ← **反向，不可當信心**")
    cf = np.array(confs)
    for k in ("conf", "agree", "onrest", "slope"):
        g, b = np.median([c[k] for c, m in zip(cf, good) if m]), \
               np.median([c[k] for c, m in zip(cf, good) if not m])
        print(f"  {k:8s}      好 {g:.2f} / 壞 {b:.2f}"
              f"{'   ← 可用' if k == 'conf' and g > b + 0.15 else ''}")
    thr = 0.5
    keep = np.array([c["conf"] >= thr for c in cf])
    if keep.any():
        harm = np.array(harm)
    print(f"\n**和聲一致率（音程類 mod 12，天使實際唱的 vs 標準答案）**："
          f"中位 {100*np.median(harm):.0f}%｜平均 {100*harm.mean():.0f}%"
          f"｜對得好的那批 {100*harm[good].mean():.0f}%"
          f"｜位置對錯的那批 {100*harm[~good].mean():.0f}%")
    print("  → 位置誤差高估了實際傷害：對到重複段時和聲往往一樣。")
    print(f"  門檻 conf≥{thr}：留下 {100*keep.mean():.0f}% 的句子，"
              f"其中對得好的佔 {100*good[keep].mean():.0f}%"
              f"（不設門檻是 {100*good.mean():.0f}%）")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="同一首兩遍（沒有標準答案，只看形狀）")
    ap.add_argument("--synth", action="store_true",
                    help="合成驗收：有標準答案，量的是對位器本身（跑 vcclient-dev）")
    a = ap.parse_args()
    if a.synth:
        _synth()
    elif a.selftest:
        _selftest()
