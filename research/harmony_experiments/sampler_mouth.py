"""sampler_mouth.py - option 0, a unit-sampling voice.

Sustained notes are not synthesised: every note a part sings plays a sustain unit
from a real recording.

v1 (--plain): static units with an equal-power butt joint. Judged by ear as
"like a piano, with none of the process of singing".
v2 (the default): a layer of singing gesture. Everything is still her real
recording; only time-varying varispeed bends it.
  - a scoop into the attack: the head of a phrase slides in from -60 cents
  - legato bending across a change: the previous unit slides to the new pitch
    first (up to 4 semitones), and only then does a crossfade to the new unit
    happen inside the stable part, so the splice hides behind the note event
  - a slight fall in pitch at the end of a phrase, and a natural release
  - breath following: the parts' gain envelope follows the take's RMS, so they
    phrase with the singer

Pipeline:
  build   vowel recordings to a library of stable sustain units (a JSON index
          holding only offsets)
  render  notes.json (world_mouth) or .targets.json (solo_host) to
          {tag}_angel.wav and {tag}_mix.wav (0.5 take + 0.8 stem, the world_mouth recipe)

Usage (vcclient-dev environment, run from harmony/):
  python sampler_mouth.py build --wavs A.wav B.wav --out out/units_girl.json
  python sampler_mouth.py render --units out/units_girl.json \
      --notes out/angel_v2_world2_notes.json --take ../../260722_harmony_brain/data/take.wav \
      --transpose 12 --tag sampler_girl_v2

Nothing in server/ changes. Unit selection rotates on a fixed seed, so different
parts draw different takes, which decorrelates them for free.
"""
import argparse, json, math, os
import numpy as np
import soundfile as sf
import soxr

from pitch import yin_f0

SR = 44100
HOP = 512
FRAME = 2048
XF = int(0.030 * SR)          # v1 event splice
MIN_UNIT_SEC = 0.35
STAB_ST = 0.5
RMS_GATE = 0.01

# v2 gesture parameters
XF2 = int(0.120 * SR)         # legato crossfade between units, hidden inside the stable part
POST_MAX = int(0.180 * SR)    # how long the old unit sings on after the transition before handing over
SCOOP_C = -60.0               # where the attack slides in from, in cents
SCOOP_TAU = 0.040
GLIDE_TAU = 0.050             # time constant of the legato bend
BEND_MAX_ST = 4               # beyond this interval, re-attack instead; a real singer re-articulates a large leap too
CUT_XF = int(0.040 * SR)      # the short splice used on a large leap
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
    """Exponential approach, the analytic one-pole: c(t) = target + (c0 - target) e^(-t/tau)"""
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
        # breath units: silent, above the noise floor, 0.12 to 0.8 s, adjacent to a voiced stretch
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


# ---------------------------------------------------------------- shared by render

def parse_notes(path):
    d = json.load(open(path))
    if "notes" in d:                       # world_mouth format
        return d["notes"], int(d["step_samps"])
    return d["targets"], int(round(d["tick_sec"] * SR))  # solo_host format


def pick_unit(lib_by_note, target, rng, last_id, need_len=None, cluster=None):
    """Nearest-pitch unit selection. The v5 ablation added two constraints, both
    attributed by measurement: cluster keeps the timbre consistent within a phrase
    (the main offender in the girl version, where adjacent units jumped 0.125 in
    timbre), and need_len prefers units long enough not to loop (the main offender
    in the other version, where 55% of events needed looping). Both constraints are
    soft and give way layer by layer once the pool empties."""
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
    """Mean MFCC vector to a k-means timbre cluster, a proxy for the unlabelled vowel."""
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
    """Fit a unit's audio to n_out samples: loop the middle if it is short, with an equal-power splice; truncate if it is long."""
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
    """Zero-phase RMS envelope with a symmetric window; offline needs no causality. Returns the envelope on the hop grid."""
    m = len(x) // hop
    rms = np.sqrt(np.mean(x[:m * hop].reshape(m, hop) ** 2, axis=1))
    w = np.hanning(max(3, int(win_s * SR / hop) | 1))
    return np.convolve(rms, w / w.sum(), mode="same")


def take_gain(take, n, floor=0.12, win_s=0.020):
    """Articulation stamp: a zero-phase fast envelope of the take (20 ms window) becomes the parts' gain curve.

    This keeps the syllable transients and the consonant breaks, so her vowels
    follow his words. Zero phase means no lag: the causal follower of v3 had 23 ms
    of group delay, and the phase difference ate the syllable-band correlation.
    The floor lets the parts sustain quietly while he breathes."""
    hop = 256
    e = smooth_env(take, hop, win_s)
    env = np.interp(np.arange(n), np.arange(len(e)) * hop, e,
                    left=e[0] if len(e) else 0, right=e[-1] if len(e) else 0)
    ref = np.percentile(env, 95) + 1e-12
    return floor + (1 - floor) * np.clip(env / ref, 0, 1)


def flatten_texture(y, strength=1.0, win_s=0.050):
    """Flatten the carrier: divide a unit by its own envelope (50 ms, zero phase)
    raised to the power `strength`. At 1 it is fully flattened (v4: the highest
    stamp correlation, but judged by ear to be "a skin of timbre", with all
    expression gone); 0.35 flattens it partly, keeping the carrier's natural life
    and giving some of the budget back to the articulation stamp."""
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
    """The layer of singing gesture: a scoop into the attack, legato bending across changes, and a release. All varispeed, nothing synthesised."""
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
        # where each note enters: on a bend the new unit enters late, so the splice hides in the stable part after the note event
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

            # the time range of this segment, including the crossfade at its head
            xfc = 0 if i == 0 else (XF2 if bend_ok[i] else CUT_XF)
            xfc = min(xfc, entries[i] - (entries[i - 1] if i else 0)) if i else 0
            seg_s = entries[i] - xfc
            seg_e = (entries[i + 1] if i + 1 < len(ph) else end)
            if i + 1 < len(ph):
                seg_e += 0  # the tail extension is covered by the next note's crossfade: this one sings to the other's entry
                seg_e = entries[i + 1]
            L = seg_e - seg_s
            if L <= 0:
                continue

            # the cents curve
            c = np.zeros(L)
            if i == 0:
                c += exp_toward(L, SCOOP_C, 0.0, SCOOP_TAU)
            elif bend_ok[i]:
                # handover curve: when the old unit reaches the entry point it still
                # has iv * exp(-POST/tau) left to slide, and the new unit starts from
                # the same place with the same tau, so both layers agree in pitch
                # while they overlap. Measured: 33 cents of deviation across a
                # transition against 7 in a stable stretch, which is where the
                # unsteadiness lived.
                iv_c = (ev["midi"] - ph[i - 1]["midi"]) * 100.0
                post = entries[i] - ev["t0"]
                c0 = -iv_c * math.exp(-post / SR / GLIDE_TAU)
                c += exp_toward(L, c0, 0.0, GLIDE_TAU)
            if i + 1 < len(ph) and bend_ok[i + 1]:
                b = ph[i + 1]["t0"] - seg_s          # where the transition starts, in this segment's coordinates
                if 0 < b < L:
                    iv_c = (ph[i + 1]["midi"] - ev["midi"]) * 100.0
                    c[b:] = c[b:] - c[b] + exp_toward(L - b, c[b], iv_c, GLIDE_TAU)
                n_bend += 1
            elif i + 1 < len(ph):
                n_cut += 1
            if i == len(ph) - 1:
                d = min(END_FADE, L)
                c[L - d:] += exp_toward(d, 0.0, END_DROOP_C, END_FADE / SR / 3)

            # time-varying varispeed read: a real recording, bent but never synthesised
            ratio = base * 2 ** (c / 1200.0)
            pos = np.cumsum(ratio)
            need = int(pos[-1]) + FRAME
            raw = srcs[u["src"]][u["s0"]:u["s1"]]
            raw = fit_duration(raw, need)
            y = np.interp(pos, np.arange(len(raw)), raw)
            y = flatten_texture(y, opts.get("flatten", 1.0))
            y *= med_rms / (np.sqrt(np.mean(y ** 2)) + 1e-12)

            # amplitude window
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
    """v7 runs: one unit is bent all the way through - real legato with no splices -
    and only replaced on a large leap or a bend of more than 3 semitones. After v6
    ruled out everything else, the remaining objective gap was the NUMBER of splices
    itself: changing unit per note means 34 crossfade disturbances, where the
    reference carrier is one continuous source. Her units have a median length of
    4.4 s, so one is usually enough for a whole phrase."""
    opts = opts or {}
    stem = np.zeros(n_ticks * tick + XF2 + POST_MAX)
    shifts, used, last_id = [], set(), -1
    n_bend = n_seam = 0
    MAX_OFF_ST = 3          # the furthest one unit is bent, in semitones

    phrases, cur = [], []
    for ev in events:
        cur.append(ev)
        if not ev["legato"]:
            phrases.append(cur)
            cur = []
    if cur:
        phrases.append(cur)

    phrase_bounds, _pe = [], 0   # (phrase start, previous phrase end), for inserting breaths
    for ph in phrases:
        phrase_bounds.append((ph[0]["t0"], _pe))
        _pe = ph[-1]["t0"] + ph[-1]["n"]

    for ph in phrases:
        # cut into runs: strings of notes one unit can bend through continuously
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

            # the cents curve relative to the root: a one-pole step per note, bending towards the next run at the end
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
            if tail_b < L:  # the handover bridge into the next run
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
    """Insert a real breath before the start of a phrase. This has to be called
    AFTER the articulation stamp: the take is silent between phrases, and the
    stamp's floor would crush a breath inserted first."""
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


# ---------------------------------------------------------------- render entry point

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
        stem *= take_gain(take, len(stem))       # the articulation stamp
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
    mix = 0.5 * take[:n] + 0.8 * stem[:n]        # the world_mouth recipe
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
    r.add_argument("--plain", action="store_true", help="v1 behaviour, no gesture layer")
    r.add_argument("--no-follow", action="store_true", help="turn the articulation stamp off")
    r.add_argument("--flatten", type=float, default=0.35,
                   help="carrier flattening strength 0 to 1 (1 = v4, fully flattened, a skin of timbre; 0 keeps all the original expression)")
    r.add_argument("--no-same-timbre", action="store_true", help="drop the within-phrase timbre constraint")
    r.add_argument("--no-prefer-long", action="store_true", help="drop the preference for units long enough not to loop")
    r.add_argument("--per-note", action="store_true", help="v5/v6 behaviour, a new unit per note, for A/B")
    r.add_argument("--no-breath", action="store_true", help="do not insert breath units")
    args = ap.parse_args()
    {"build": build, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
