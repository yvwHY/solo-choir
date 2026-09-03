"""score_notes.py — 阿卡貝拉 MIDI → 三聲部 tick 網格（08-03，譜這一側）

Harry 08-03：「輸入阿卡貝拉 midi 生成聲部和音這個功能還是要有」。腦（BrainV3）
即興出來的和聲之外，再開一條「和聲由編好的譜決定」的路。

**下游完全不用重做**：外部音符線進系統的格式早就存在（`prerender_stems.py`
吃的排練檔）＝`{"lead": [...], "upper": [...], "lower": [...]}`，每 tick 一個
sounding MIDI 音高或 None。兩張嘴、target f0 配方、polish、殘響、混音平衡都已
經耳測過，本檔只負責把譜變成那個 dict。

本檔**只做譜這一側**（譜自己的時間軸）。把它對到他唱的那一句、換成他的節奏，
是下一段（DTW）——那才是有風險的部分。刻意分開，因為譜這側可以拿 CPDL 那
4016 個檔大量驗證，對位那側只能拿他的錄音驗。

進門那層要猜的事（現成編曲不由你決定格式）：哪些軌是人聲、哪個是旋律、哪兩個
給天使、要不要移調。**全部可以用旗標覆寫**——自己編的譜預設就會對，撿來的譜
猜錯就指定。一條路徑，不是兩套。

⚠ 這支跑 **vcclient-dev**（pretty_midi 在那裡，DDSP venv 沒有），不是 DDSP venv。
不衝突：它是離線步驟、產物是 json，渲染那側照樣走 DDSP venv。

Run（vcclient-dev）:
  python score_notes.py song.mid --out out/song_notes.json      # 猜
  python score_notes.py song.mid --melody 0 --upper 1 --lower 3 --transpose -5
  python score_notes.py --survey '../../260723_corpus/cpdl_sample/*.mid'
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from pitch import SR  # noqa: E402
from render_v3 import GIRL_RANGE, HARRY_RANGE, fold  # noqa: E402
from world_live import TICK_SAMPS  # noqa: E402

TICK_S = TICK_SAMPS / SR                      # 0.1875s，與腦的 tick 同一格
# 他的舒適音域（voicing_mask 註解記的 110–300 Hz ＝ A2–D4；取稍寬）。lead 移調
# 的目標——譜的旋律不會剛好落在他的音域，而他唱不上去就整條線都不用談。
LEAD_RANGE = (45, 64)                         # A2–E4


def load_parts(path, max_poly=0.15):
    """MIDI → 單音聲部清單（每個 [(start, end, pitch)]，依中位音高由高到低）。

    撿來的譜什麼都有：鋼琴縮譜、四部擠在一軌、空軌。過濾規則：
      - 音符數太少的軌丟掉（標題軌、節拍軌）
      - **和弦比例超過 max_poly 的軌丟掉**＝那是縮譜或多部擠一軌，不是聲部線
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    parts = []
    for inst in pm.instruments:
        if inst.is_drum or len(inst.notes) < 8:
            continue
        ns = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        # 重疊率：起音在前一顆結束之前＝同時發聲
        ov = sum(1 for a, b in zip(ns, ns[1:]) if b.start < a.end - 1e-3)
        if ov / max(1, len(ns) - 1) > max_poly:
            continue
        parts.append({
            "name": inst.name.strip() or f"track{len(parts)}",
            "notes": [(n.start, n.end, n.pitch) for n in ns],
            "med": float(np.median([n.pitch for n in ns])),
        })
    parts.sort(key=lambda p: -p["med"])        # 高→低＝S A T B
    return parts, pm.get_end_time()


def pick(parts, melody=None, upper=None, lower=None):
    """挑 (lead, upper, lower) 的索引。

    預設：**旋律＝最高聲部**（撿來的編曲裡人唱的幾乎都是它），天使取它上下
    最近的兩條＝07-27 v3 spec 定的「上下包夾」配置。旋律在最高聲部時上面沒有
    東西，就往下再借一條、由八度摺疊把它送到 lead 之上（摺疊瞬間包夾對調是
    spec 已接受的代價）。"""
    n = len(parts)
    if n < 3:
        raise ValueError(f"只找到 {n} 條單音聲部，至少要 3 條（--melody/--upper/"
                         f"--lower 可手動指定，或這份譜是縮譜）")
    li = 0 if melody is None else melody
    if upper is not None and lower is not None:
        return li, upper, lower
    rest = [i for i in range(n) if i != li]
    above = [i for i in rest if parts[i]["med"] > parts[li]["med"]]
    below = [i for i in rest if parts[i]["med"] <= parts[li]["med"]]
    ui = (above[-1] if above else below[0]) if upper is None else upper
    lo = [i for i in below if i != ui]
    li_ = (lo[0] if lo else [i for i in rest if i != ui][0]) if lower is None else lower
    return li, ui, li_


def to_ticks(notes, n_ticks, tick_s=TICK_S):
    """(start, end, pitch) → 每 tick 一個 sounding 音高（None＝休止）。
    一格內有多顆音就取**佔比最長的那顆**（tick 0.1875s 比十六分音符長，
    快速走句本來就會被吃掉——這是 tick 網格的既有代價，不是這裡新引入的）。"""
    out = []
    for k in range(n_ticks):
        t0, t1 = k * tick_s, (k + 1) * tick_s
        best, best_ov = None, 0.0
        for s, e, p in notes:
            if e <= t0:
                continue
            if s >= t1:
                break
            ov = min(e, t1) - max(s, t0)
            if ov > best_ov:
                best, best_ov = p, ov
        out.append(best)
    return out


def build(path, melody=None, upper=None, lower=None, transpose=None):
    parts, dur = load_parts(path)
    li, ui, lo = pick(parts, melody, upper, lower)
    n_ticks = int(np.ceil(dur / TICK_S))
    raw = {k: to_ticks(parts[i]["notes"], n_ticks)
           for k, i in (("lead", li), ("upper", ui), ("lower", lo))}
    if transpose is None:
        # 自動移調：把旋律的中位音高送進他的舒適音域，整份譜同幅度移動
        # （三條線一起移＝和聲關係不變）。
        med = float(np.median([p for p in raw["lead"] if p is not None]))
        tgt = 0.5 * (LEAD_RANGE[0] + LEAD_RANGE[1])
        transpose = int(round((tgt - med) / 12.0)) * 12      # 只走八度，不改調
    out = {k: [None if p is None else p + transpose for p in v]
           for k, v in raw.items()}
    # 天使摺進各自嘴的實測音域（render_v3 同一支 fold，同樣的代價與理由）
    out["upper"] = fold(out["upper"], *GIRL_RANGE)
    out["lower"] = fold(out["lower"], *HARRY_RANGE)
    info = {"parts": [p["name"] for p in parts],
            "med": [round(p["med"], 1) for p in parts],
            "picked": {"lead": li, "upper": ui, "lower": lo},
            "transpose": transpose, "ticks": n_ticks,
            "dur_s": round(dur, 1)}
    return out, info


def survey(pattern, limit=400):
    """拿 CPDL 那批當測試素材：進門那層到底吃得下多少現成編曲。"""
    fs = sorted(glob.glob(pattern))[:limit]
    ok = bad_parse = few = 0
    npart = []
    for f in fs:
        try:
            parts, _ = load_parts(f)
        except Exception:
            bad_parse += 1
            continue
        npart.append(len(parts))
        if len(parts) < 3:
            few += 1
        else:
            ok += 1
    npart = np.array(npart) if npart else np.zeros(1)
    print(f"{len(fs)} 個檔：可用 {ok}（{100*ok/max(1,len(fs)):.0f}%）"
          f"｜解析失敗 {bad_parse}｜單音聲部 <3 條 {few}")
    print(f"單音聲部數 中位 {np.median(npart):.0f}  "
          f"分布 {np.bincount(npart.astype(int))[:9].tolist()}（index＝條數）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mid", nargs="?")
    ap.add_argument("--out", default=None, help="寫成排練檔格式的 notes json")
    ap.add_argument("--melody", type=int, default=None,
                    help="旋律是第幾條聲部（0＝最高，猜錯時指定）")
    ap.add_argument("--upper", type=int, default=None)
    ap.add_argument("--lower", type=int, default=None)
    ap.add_argument("--transpose", type=int, default=None,
                    help="半音；不給＝自動把旋律送進他的音域（只走八度）")
    ap.add_argument("--survey", default=None, help="glob，批次檢查可用率")
    a = ap.parse_args()
    if a.survey:
        return survey(a.survey)
    if not a.mid:
        ap.error("要給一個 .mid，或用 --survey")
    try:
        out, info = build(a.mid, a.melody, a.upper, a.lower, a.transpose)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"聲部（高→低）: {list(zip(info['parts'], info['med']))}")
    print(f"挑中 lead={info['picked']['lead']} upper={info['picked']['upper']} "
          f"lower={info['picked']['lower']}｜移調 {info['transpose']:+d} 半音"
          f"｜{info['dur_s']}s = {info['ticks']} ticks")
    for k in ("lead", "upper", "lower"):
        v = [p for p in out[k] if p is not None]
        print(f"  {k:6s} 有聲 {100*len(v)/len(out[k]):3.0f}%  "
              f"音域 {min(v) if v else '-'}–{max(v) if v else '-'}  前 12 {out[k][:12]}")
    if a.out:
        json.dump({**out, "k_shift": 0, "step_samps": TICK_SAMPS},
                  open(a.out, "w"))
        print(f"wrote {a.out}（排練檔格式，prerender_stems / respond2 可直接吃）")


if __name__ == "__main__":
    main()
