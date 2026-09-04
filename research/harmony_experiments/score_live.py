"""score_live - the parts sing from a score and the singer conducts (v7.1).

An abandoned line, kept as the evidence behind the score-following entries in
GRAVEYARD. Superseded by the answering mode and the sampled choir.

The problem: the parts have no idea where in the piece the singer is. Five
versions circled one question, who leads.

  v3   The parts follow the singer within a phrase, tracked by an HMM over the
       score with a uniform prior so they converge from anywhere. This
       deadlocked: the parts chase the singer while the singer tunes to the
       parts, and confidence oscillates. The fix made leadership explicit -
       position belongs to the singer, tempo belongs to the score - so the parts
       advance at score tempo within a phrase and realign only at boundaries.
  v3.1 A camera gate on the mouth, because feedback cannot fool the lips, and it
       fails open to audio-only if the camera dies.
  v3.2 Parts hold their own line: standby moved to a rest in every part rather
       than the end of the lead's phrase, because in a real score 91% of the
       ticks where the lead rests still have harmony sounding.
  v4   Choir intuition: sing in order from memory, no global search. The HMM
       stayed available for rehearsal under --explore.
  v7   Dependency inverted. The choir sings by itself and the singer joins it:
       whichever line they sing is yielded to them and taken back about 1.5 s
       after they stop. The classification sits behind the mouth gate, so the
       choir's own lead returning through the microphone cannot claim a line.
  v7.1 The parts play pre-rendered stems rather than rendering live. Rendering
       live through a sliding window with SOLA overlap held the melody line to
       the score only 49% of the time against 80% offline, because every note
       change lands on a splice; sustained harmony lines were largely spared.
       Since the choir's part is fixed by the score, nothing needs rendering
       live: the four parts are rendered once at start-up, cached to disk, and
       the live loop only plays, yields and advances.

Run: venv/bin/python score_live.py --score scratchpad/xnn_all5.json \
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
ap.add_argument("--line2", default="", help="a second json; t3/t4 take its lead/upper")
ap.add_argument("--in-name", default="USB PnP")
ap.add_argument("--out-name", default="MacBook Pro的揚聲器")  # zh-Hant macOS speaker name
ap.add_argument("--block", type=float, default=0.24)
ap.add_argument("--vowel-src", default="scratchpad/take17.wav")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--min-rest", type=int, default=2,
                help="shortest rest counting as a phrase boundary, in ticks of about 186 ms")
ap.add_argument("--entry-win", type=int, default=6,
                help="entry window at the head of a phrase, in ticks")
ap.add_argument("--explore", action="store_true",
                help="explore mode: global HMM location, the v3.2 behaviour; default is sequential")
ap.add_argument("--wait", action="store_true",
                help="v4 accompaniment mode: the lead is the singer's, the parts "
                     "wait for a cue and fade when they stop; the A/B against v7")
ap.add_argument("--voice", type=float, default=0.0,
                help="gain of the dry pass-through, 0 to disable; feedback over speakers is your own risk")
ap.add_argument("--latency", default="low",
                help="stream latency: low, high, or seconds. Measured here, high "
                     "gives 839 ms in and 723 ms out, about 1.9 s late in the room")
ap.add_argument("--sblock", type=int, default=512,
                help="stream block size in samples; processing granularity is still "
                     "--block. Latency is dominated by the stream block, so this is "
                     "the one that matters")
ap.add_argument("--mouth", type=int, default=1, help="camera mouth gate, 0 to disable")
ap.add_argument("--cam", type=int, default=-1, help="camera index, -1 picks the brightest")
ap.add_argument("--dump", default="")
a = ap.parse_args()

d = json.load(open(a.score))
lead, t1, t2 = d["lead"], d["upper"], d["lower"]
t4 = json.load(open(a.line2))["upper"] if a.line2 else None

WAIT = a.wait or a.explore               # the older behaviour: the parts wait for the singer and stop when they stop

PARTS = [("t1", f"{S.DDSP}/exp/reflow-sop3/model_20000.pt", 1, 0.7, t1),
         ("t2", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt", 2, 0.85, t2)]
if t4:
    PARTS.append(("t4", f"{S.DDSP}/exp/reflow-bass1/model_32000.pt", 1, 1.0, t4))
LIDX = -1                                # index of the lead part in PARTS (v7 only)
if not WAIT:
    # v7: the lead is a part too (alto3-40k, speaker 3 to avoid t2's speaker 2),
    # and yields only when the singer sings that line
    PARTS.insert(0, ("t0", f"{S.DDSP}/exp/reflow-alto3/model_40000.pt",
                     3, 1.0, lead))
    LIDX = 0

# -- score to a frame grid of features --
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

# -- v7.1 stems: rendered once at reference quality and cached to disk, so the
# live loop only plays and the render cost falls to zero --
_ck = hashlib.md5((repr([(nm_, p_, sp_, g_) for nm_, p_, sp_, g_, _l in PARTS])
                   + repr([l for *_x, l in PARTS])
                   + a.vowel_src).encode()).hexdigest()[:10]
_spaths = [f"scratchpad/stems_{_ck}_{nm_}.wav" for nm_, *_x in PARTS]
STEMS = []
if all(os.path.exists(p_) for p_ in _spaths):
    print(f"stems cache hit ({_ck}): no models loaded, opens instantly", flush=True)
    for p_ in _spaths:
        w_, _ = sf.read(p_, dtype="float32")
        STEMS.append(w_)
else:
    print("pre-rendering stems (slow the first time, instant from cache after)...", flush=True)
    mic0, _ = sf.read(a.vowel_src, dtype="float64", always_2d=True)
    mic0 = mic0[:, 0]
    UU = None
    for _pi, ((nm_, p_, sp_, g_, _l), (f0L_, volL_)) in enumerate(
            zip(PARTS, FEATS)):
        svc = S.Svc([(p_, 0.0, g_, sp_)], step=2, t_start=0.85)
        if UU is None:
            # vowel units, as in score_sing: loop the middle of the longest voiced stretch of take17
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
        CHF = 800                        # frames per chunk, about 9.3 s, the score_sing reference path
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
        print(f"  {nm_} rendered ({time.time() - t0_:.0f}s) -> {_spaths[_pi]}",
              flush=True)
        del svc
STEMS = [np.pad(w_, (0, max(0, NF * HOP - len(w_))))[:NF * HOP]
         .astype("float32") for w_ in STEMS]

# -- tracker: v3 with a uniform prior, so it can enter from anywhere --
from rehearse_ab import ScoreTracker  # noqa: E402
tr = ScoreTracker([None if n is None else n - 12 for n in lead], mode="v3")
# the tracker drops the lead an octave into the singer's range; the parts sing
# the score's absolute pitch and are unaffected
tr.alpha = np.ones(NT) / NT              # global location: no prior on the starting point

# -- phrase table: a lead rest of at least min-rest ticks is a boundary --
PHR = []                                 # (start, end) in ticks, end exclusive
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
        _j = _k                          # a short rest is a breath within the phrase, not a cut
    PHR.append((_i, _j))
    _i = _j
    while _i < NT and lead[_i] is None:
        _i += 1
PIDX = np.full(NT, -1, dtype=int)        # tick to phrase index, -1 for a rest
for _pi, (_s, _e) in enumerate(PHR):
    PIDX[_s:_e] = _pi

# -- section table: a tutti rest, lead and every harmony part silent for at
# least one tick, is where the parts stand by --
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
SIDX = np.full(NT, -1, dtype=int)        # tick to section index
for _si, (_s, _e) in enumerate(SEG):
    SIDX[_s:_e] = _si
LFIRST = []                              # first lead note per section as (tick, note); None if there is none
for _s, _e in SEG:
    _lt = next((t for t in range(_s, _e) if lead[t] is not None), None)
    LFIRST.append(None if _lt is None else (_lt, lead[_lt]))


def _match(n, ln):
    """The MIDI note sung against the score, at pitch or an octave down; within 4 semitones counts."""
    return ln is not None and min(abs(n - ln), abs(n - ln + 12)) <= 4


def _duck_vote(n, t):
    """v7 yielding: which part line the singer is on. Octaves are folded, the
    nearest must be within 2 semitones and beat the runner-up by 2, otherwise it
    abstains (-1 means nobody yields and the singer doubles a part)."""
    ds = [1e9 if ln[t] is None else
          min(abs(n - ln[t] - o) for o in (-12, 0, 12))
          for *_x, ln in PARTS]
    w = int(np.argmin(ds))
    d2 = sorted(ds)
    return w if d2[0] <= 2 and d2[1] - d2[0] >= 2 else -1
print(f"score {NT} ticks ({NF*HOP/SR:.0f}s); {len(PHR)} phrases, "
      f"{len(SEG)} sections; {len(PARTS)} parts (stems {_ck})",
      flush=True)

# -- state --
blk = int(a.block * SR)
NB = blk // HOP                          # frames per block
CF = int(0.02 * SR)                      # 20 ms crossfade when the play head jumps
st = {"spos": None, "lastv": time.time(), "conf": 0.0, "okc": 0,
      "until": 0.0, "seg": -1, "sync_ph": -1, "slew": 0.0, "exp": True,
      "jmpc": 0, "ending": False, "nxt": 0, "sq": 0, "vc": 0, "vt0": 0.0,
      "wstart": time.time(), "rsm": None, "xrun": 0, "skip": 0, "blkc": 0,
      "tickbuf": np.zeros(0), "die": False, "fade": 0.0,
      "duck": -1, "dvc": 0, "dcand": -1, "pend": None}
vfade = [1.0] * len(PARTS)               # per-part yield fade (1 means the part is singing)


def rearm():
    """Return to standby: clear playback state so the next entry starts clean."""
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
        out_q.clear()                    # flush the output backlog so latency does not carry across sections


# -- mouth gate (v3.1): mouth opening, hysteresis and a 0.8 s grace period;
# fails open if the camera dies --
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
            assert cap.isOpened(), f"cannot open camera {a.cam}"
        else:
            # pick the brightest camera; indices 0 and 2 are often virtual cameras and come back black
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
            assert best and best[1] > 10, f"no camera with a picture {best}"
            print(f"[mouth] camera index {best[0]} (brightness {best[1]:.0f})",
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
            if now - last_det < 0.05:        # landmarks at 20 fps or less, leaving CPU for rendering
                continue
            last_det = now
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, int((now - t0) * 1000))
            if res.face_landmarks:
                last_face = now
                f = res.face_landmarks[0]
                # opening = distance between the inner lips (13 to 14) over face
                # height (forehead 10 to chin 152). Probed: singing p50 0.099,
                # p10 0.0054; mouth closed p50 0.0029
                val = abs(f[13].y - f[14].y) / (abs(f[10].y - f[152].y)
                                                + 1e-9)
                M["val"] = val
                nv = val > (0.008 if M["on"] else 0.02)        # hysteresis
                if nv and not M["on"]:
                    M["t_on"] = now      # the moment it opens is the timestamp of a breath cue
                M["on"] = nv
                if M["on"]:
                    M["last_open"] = now
            # 0.8 s grace, so a consonant closure or a dropped frame does not break
            # the line; a face lost for over 1 s fails open
            M["ok"] = (now - M["last_open"] < 0.8) or (now - last_face > 1.0)
        cap.release()
    except Exception as e:
        M["ok"] = True
        print(f"[mouth] gate failed ({e}), falling back to audio-only v3", flush=True)


if a.mouth:
    threading.Thread(target=_mouth_worker, daemon=True).start()


def _stdin_jump():
    """Jumping section is the conductor speaking: type a section number and Enter, or r to go back to section 1."""
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
        print(f"\n[jump] standing by at section {k + 1}/{len(SEG)}, cue to enter", flush=True)


threading.Thread(target=_stdin_jump, daemon=True).start()
dmp = {"mic": [], "out": [], "pos": [], "mouth": []} if a.dump else None


def tick_note(x):
    """0.25 s of audio to a sounding MIDI note, or None."""
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
        # process block by block; when several arrive at once, rendering only one breaks the output stream
        chunks = [pend[i:i + blk] for i in range(0, len(pend) - blk + 1, blk)]
        rem = len(pend) - len(chunks) * blk
        if rem > 0:
            with qlock:
                in_q.insert(0, pend[-rem:])
        for chunk in chunks:
            _process(chunk)


def _process(pend):
        st["tickbuf"] = np.concatenate([st["tickbuf"], pend])[-4 * blk:]
        # tick tracking once per block; a 244 ms block is about 1.3 ticks, fine enough
        note = tick_note(st["tickbuf"][-int(0.25 * SR):])
        mo = M["ok"]                     # mouth gate: closed means any pitch is feedback or noise
        pv_sq = st["sq"]                 # blocks of silence before this one, for onset detection
        if note is not None and mo:
            if st["vc"] == 0:
                st["vt0"] = time.time()  # the start of this run of phonation
            st["vc"] += 1
            st["sq"] = 0
            st["lastv"] = time.time()
        else:
            st["sq"] += 1
            st["vc"] = 0
        if a.explore:
            # -- explore mode: global HMM location (v3.2) --
            tr.observe(note if mo else None)
            conf = float(np.max(tr.alpha)) if hasattr(tr, "alpha") else 1.0
            st["conf"] = conf
            p = tr.p
            st["okc"] = st["okc"] + 1 if conf > 0.22 else 0
            loc = (mo and p is not None and 0 <= p < NT and st["okc"] >= 3
                   and PIDX[p] >= 0 and p - PHR[PIDX[p]][0] < a.entry_win)
            # location is valid and the singer is standing at the head of a phrase
            if st["spos"] is None and loc:
                ph = int(PIDX[p])
                sg = int(SIDX[p])
                st["spos"] = float(p * TICK)
                st["sync_ph"] = ph
                st["seg"] = sg
                st["until"] = float(min(NF - 2, (SEG[sg][1] + 1) * TICK))
                st["exp"] = True
                st["lastv"] = time.time()
                print(f"\n[enter] section {sg + 1}/{len(SEG)} "
                      f"phrase {ph + 1}/{len(PHR)} tick {p}"
                      f"（conf {conf:.2f}）", flush=True)
            elif st["spos"] is not None and loc and PIDX[p] != st["sync_ph"]:
                # realign at the phrase head: one bounded correction, not a continuous PLL
                err = p * TICK - st["spos"]
                if abs(err) > 8 * TICK:
                    st["jmpc"] += 1
                    if st["jmpc"] >= 3:  # three blocks elsewhere means they really did jump
                        rearm()
                        print(f"\n[jump] off by {err / TICK:+.0f} ticks, "
                              f"re-entering", flush=True)
                else:
                    st["jmpc"] = 0
                    st["slew"] = float(np.clip(err, -2 * TICK, 2 * TICK))
                    st["sync_ph"] = int(PIDX[p])
        else:
            # -- sequential mode: in order (v7 choir member, --wait v4 accompaniment) --
            if st["spos"] is None:
                k = st["nxt"]
                if k < len(SEG):
                    if not WAIT:
                        # v7: a breath or a note is the cue to begin, and the choir sings on from the head of the section
                        at = SEG[k][0]
                        go = ((M["on"] and M["t_on"] > st["wstart"]
                               and time.time() - M["t_on"] > 0.3)
                              or st["vc"] >= 2)
                        adv = 0.0
                    elif st["rsm"] is None and LFIRST[k] is None:
                        # a section with no lead takes a conductor cue: mouth open for 0.3 s, or a hum
                        at = SEG[k][0]
                        go = ((M["on"] and M["t_on"] > st["wstart"]
                               and time.time() - M["t_on"] > 0.3)
                              or st["vc"] >= 2)
                        adv = 0.0
                    else:
                        # in a lead section the singer starting is the cue, and the pitch must match the phrase head
                        at = st["rsm"] if st["rsm"] is not None else (
                            LFIRST[k][0] if LFIRST[k] else SEG[k][0])
                        go = (st["vc"] >= 2 and note is not None
                              and _match(note, lead[at]))
                        adv = min((time.time() - st["vt0"]) * SR / HOP,
                                  4.0 * TICK)   # anchor back to compensate detection latency
                    if go:
                        st["spos"] = float(at * TICK + adv)
                        st["seg"] = k
                        st["until"] = float(min(NF - 2,
                                                (SEG[k][1] + 1) * TICK))
                        st["exp"] = True
                        st["lastv"] = time.time()
                        st["rsm"] = None
                        print(f"\n[enter] section {k + 1}/{len(SEG)} tick {at}",
                              flush=True)
            elif (st["vc"] == 1 and pv_sq >= 2 and note is not None
                  and (WAIT or st["duck"] == LIDX)):
                # realign at the phrase head: singing after 0.5 s of silence is a phrase onset
                cands = [s_ for s_, _e2 in PHR
                         if abs(s_ * TICK - st["spos"]) <= 4 * TICK]
                if cands:
                    s_ = min(cands,
                             key=lambda x: abs(x * TICK - st["spos"]))
                    if _match(note, lead[s_]):
                        st["slew"] = float(np.clip(
                            s_ * TICK - st["spos"], -2 * TICK, 2 * TICK))
        # -- v7 yielding: the line the singer sings is handed over, and taken back about 1.5 s after they stop --
        if not WAIT and st["spos"] is not None:
            if note is not None and mo:
                v = _duck_vote(note, min(int(st["spos"] / TICK), NT - 1))
                if v >= 0 and v == st["duck"]:
                    st["dvc"] = 0        # the incumbent stays
                elif v >= 0:
                    st["dvc"] = st["dvc"] + 1 if v == st["dcand"] else 1
                    st["dcand"] = v
                    if st["dvc"] >= 3:   # about 0.7 s of agreement before switching, against momentary errors
                        st["duck"] = v
                        st["dvc"] = 0
                        print(f"\n[yield] {PARTS[v][0]} is yours", flush=True)
            elif st["sq"] >= 6 and st["duck"] >= 0:
                print(f"\n[resume] {PARTS[st['duck']][0]} taken back",
                      flush=True)
                st["duck"] = -1
                st["dvc"] = 0
        # abandoning a phrase follows the score: only if the score says sing and
        # the singer stops for over 2.5 s. A rest in the score means their silence
        # is written. In v7 choir mode this never happens.
        exp = (st["spos"] is not None
               and lead[min(int(st["spos"] / TICK), NT - 1)] is not None)
        if exp and not st["exp"]:
            st["lastv"] = time.time()    # the score says sing again: reset the 2.5 s timer
        st["exp"] = exp
        want = 1.0 if (st["spos"] is not None
                       and (not WAIT or not exp
                            or time.time() - st["lastv"] < 2.5)
                       ) else 0.0
        st["fade"] += np.clip(want - st["fade"], -0.15, 0.15)
        if st["spos"] is not None and want == 0.0 and st["fade"] < 0.01:
            rs = None
            if not a.explore:
                st["nxt"] = st["seg"]    # after abandoning, cue from the nearest phrase head rather than repeating
                sg0 = SEG[st["seg"]][0]
                cands = [s_ for s_, _e2 in PHR
                         if sg0 <= s_ and s_ * TICK <= st["spos"]]
                rs = max(cands) if cands else None
            rearm()
            st["rsm"] = rs
            print(f"\n[fade] you stopped; cue me from a phrase head", flush=True)
        if st["spos"] is not None:
            sl = float(np.clip(st["slew"], -4, 4))   # a phrase-head correction slews in at 4 frames per block
            st["slew"] -= sl
            st["spos"] += NB + sl        # tempo lock: constant score tempo within a section, no PLL
            if st["spos"] >= st["until"]:
                # past the end of the section: render this block, then fade out and stand by
                st["spos"] = min(st["spos"], float(NF - 2))
                st["ending"] = True
        st["blkc"] += 1
        if st["spos"] is None or st["fade"] < 0.01:
            y = np.zeros(blk, dtype="float32")
        else:
            # -- v7.1 playback: slice the stems and apply the yield fade; rendering is offline --
            end_s = int(st["spos"]) * HOP
            lo_s = end_s - blk
            y = np.zeros(blk, dtype="float32")
            for vi in range(len(PARTS)):
                w = np.zeros(blk, dtype="float32")
                seg_ = STEMS[vi][max(0, lo_s):max(0, end_s)]
                if len(seg_):
                    w[blk - len(seg_):] = seg_   # pad with zeros before the head of the score
                # yield fade: 0.3 per block, about 0.8 s in all, linear within the block so there is no zipper noise
                vt_ = 0.0 if vi == st["duck"] else 1.0
                vf0 = vfade[vi]
                vfade[vi] = vf0 + float(np.clip(vt_ - vf0, -0.3, 0.3))
                if vf0 != 1.0 or vfade[vi] != 1.0:
                    w = w * np.linspace(vf0, vfade[vi], blk)
                y += w
            if st["pend"] is not None and lo_s != st["pend"]:
                # a jump of the play head (slew or entry) gets a 20 ms crossfade against clicks
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
                    # v7 runs section to section to the end of the piece;
                    # --wait continues only through sections with no lead, so the
                    # choir counts short rests itself and needs no new cue
                    st["ending"] = False
                    st["seg"] = nx
                    st["spos"] = float(SEG[nx][0] * TICK)
                    st["until"] = float(min(NF - 2,
                                            (SEG[nx][1] + 1) * TICK))
                    print(f"\n[continue] section {nx + 1}/{len(SEG)} (harmony carries on)",
                          flush=True)
                else:
                    y *= np.linspace(1, 0, len(y), dtype="float32")
                    lastsg = st["seg"]
                    rearm()
                    st["nxt"] = nx
                    if nx >= len(SEG):
                        print(f"\n[end] section {lastsg + 1}/{len(SEG)} finished"
                              f" (r and Enter to go back)", flush=True)
                    else:
                        print(f"\n[standby] section {lastsg + 1}/{len(SEG)} finished, "
                              f"cue section {nx + 1} to enter", flush=True)
        if a.voice > 0:
            # the dry pass-through ignores the fade, so the singer still sounds when the parts are silent
            y = np.clip(y + pend[:len(y)] * a.voice, -1, 1).astype("float32")
        with qlock:
            out_q.append(y)
            # backlog capped at two blocks: a stall skips ahead to catch up, the way
            # a choir stumbles, rather than settling into latency. The worker is
            # driven by the input clock, so a backlog never clears itself.
            while sum(len(q_) for q_ in out_q) > 2 * blk:
                out_q.pop(0)
                st["skip"] += 1
        if dmp is not None:
            dmp["mic"].append(pend.copy())
            dmp["out"].append(y.copy())
            dmp["pos"].append(st["spos"] if st["spos"] is not None else -1)
            dmp["mouth"].append((M["val"], 1.0 if mo else 0.0))


OB = [np.zeros(0, dtype="float32")]      # output ring: worker blocks out to small stream blocks


def cb(indata, outdata, frames, tinfo, status):
    if status:
        st["xrun"] += 1                  # under/overflow count, the price of low latency
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
_mode = ("explore mode: sing the lead, from any section" if a.explore else
         "accompaniment mode: cue section 1 with a breath or a hum" if a.wait else
         "choir mode: a breath or a note begins it and the choir sings to the end; the line you sing is yielded to you")
print(f"stream open in={a.in_name!r} out={a.out_name!r}"
      f" ({_mode}; jump with a section number and Enter, r to go back; Ctrl-C to stop)", flush=True)
st["wstart"] = time.time()               # a breath cue only counts after the stream opens
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
                      f"sec {st['seg'] + 1}/{len(SEG)} "
                      f"phr {_ph + 1 if _ph >= 0 else '-'}/{len(PHR)}  "
                      f"conf {st['conf']:.2f}  fade {st['fade']:.1f}  "
                      f"you {PARTS[st['duck']][0] if st['duck'] >= 0 else '-'}  "
                      f"mouth {'open' if M['ok'] else 'closed'}  "
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
