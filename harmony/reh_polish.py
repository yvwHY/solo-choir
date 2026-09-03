"""reh_polish.py — 排練檔譜面後處理：持續縱向不和諧就地修協和（08-01）

預渲染範式的紅利：排練檔是離線固定的譜 → 和音組成可以在渲染前用規則
清理，不必動腦。規則刻意保守：

- 只修「對 lead **持續** ≥MIN_RUN tick 的不和諧」（≥375ms 的掛著撞）；
  單 tick 的經過音/掛留是語彙，不動。
- 修法＝同一段天使音整段改到最近的候選音：**調內**（root 決定，見下）、與該
  段 lead 眾數音協和、盡量也不與另一天使相撞、|移動| ≤5 半音；找不到就放棄。

⚠ **呼叫端必須保證 d["lead"] 與 d["upper"]/d["lower"] 在同一個音高空間**
（08-04 抓到的 bug：respond2 把腦空間的 lead 跟麥克風空間的天使拿來比，
偏移＝key+octave；只有在偏移剛好是 12 的倍數時才隱形——`--key 0` 時正確，
一填真實的調就整個比錯）。root 同理：舊版寫死絕對 C 大調，`--key` 一動就把
天使吸附到錯的音階。root＝主音的音級（mic 空間；`--key N` ⇒ root=(-N)%12）。
- 統計印 before/after（mod-12 口徑，同 render_v3）。

Run:
  python reh_polish.py out/pair06_reh0_notes.json out/pair06_reh0p_notes.json
"""
import json
import sys
from collections import Counter

MAJ = {0, 2, 4, 5, 7, 9, 11}
DISS = {1, 2, 6, 10, 11}
MIN_RUN = 2


def stats(d):
    n = diss = n2 = d2 = 0
    for L, U, Lo in zip(d["lead"], d["upper"], d["lower"]):
        if L is not None:
            for a in (U, Lo):
                if a is not None:
                    n += 1
                    diss += abs(int(a) - int(L)) % 12 in DISS
        if U is not None and Lo is not None:
            n2 += 1
            d2 += abs(int(U) - int(Lo)) % 12 in DISS
    return 100.0 * diss / max(n, 1), 100.0 * d2 / max(n2, 1)


def runs(ns):
    """(start, end, note) 的等值段，None 段略過。"""
    out, i = [], 0
    while i < len(ns):
        if ns[i] is None:
            i += 1
            continue
        j = i
        while j < len(ns) and ns[j] == ns[i]:
            j += 1
        out.append((i, j, int(ns[i])))
        i = j
    return out


def polish(d, voice, other, root=0):
    ns, lead, ot = d[voice], d["lead"], d[other]
    fixed = 0
    for a, b, m in runs(ns):
        clash = [t for t in range(a, b) if lead[t] is not None
                 and abs(m - int(lead[t])) % 12 in DISS]
        if len(clash) < MIN_RUN:
            continue
        lm = Counter(int(lead[t]) for t in clash).most_common(1)[0][0]
        oth = [int(ot[t]) for t in range(a, b) if ot[t] is not None]
        om = Counter(oth).most_common(1)[0][0] if oth else None

        def ok(c, strict):
            if (c - root) % 12 not in MAJ or abs(c - lm) % 12 in DISS:
                return False
            return not (strict and om is not None and abs(c - om) % 12 in DISS)

        for strict in (True, False):
            cand = next((m + s * k for k in range(1, 6) for s in (-1, 1)
                         if ok(m + s * k, strict)), None)
            if cand is not None:
                break
        if cand is None:
            continue
        for t in range(a, b):
            ns[t] = cand
        fixed += 1
    return fixed


def main(src, dst):
    d = json.load(open(src))
    # live_v3 落檔的 lead 在腦空間、天使在 mic 空間（08-04 bug 的離線同形狀）。
    # v2t.shift 恆為 12 的倍數 → mod-12 口徑下兩空間只差 k_shift，polish/stats
    # 全是 mod-12 比較 → 減 k_shift 即同空間。只換算比較用的視圖；落檔的 lead
    # 保持腦空間（ScoreTracker 等下游吃的就是腦空間）。
    k = int(d.get("k_shift") or 0)
    root = (-k) % 12
    v = dict(d, lead=[None if x is None else int(x) - k
                      for x in d.get("lead", [])]) if k else d
    print(f"space: k_shift {k:+d} → root {root}（lead 已換算到 mic 空間比較）"
          if k else "space: k_shift 0＝兩空間 mod-12 重合，root 0")
    b = stats(v)
    fu = polish(v, "upper", "lower", root)
    fl = polish(v, "lower", "upper", root)
    a = stats(v)
    json.dump(d, open(dst, "w"))
    print(f"polish: upper 修 {fu} 段, lower 修 {fl} 段")
    print(f"天使vs lead 不和諧 {b[0]:.1f}% → {a[0]:.1f}% | "
          f"upper vs lower {b[1]:.1f}% → {a[1]:.1f}%")
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
