"""Beatrice feeding/pitch-transition experiment harness (2026-07-05 diagnosis).
Usage:  conda run -n vcclient-dev python eval/feed_ab.py OUT.wav MODE [MODEL_DIR] [CTX_HOPS] [CHUNK_HOPS] [XF_MS]
Modes:  v0 stream-baseline | v2 ctx-feed | v3 gain1.7 | v4 dynamic-2 | v5 iv0 | v6/v7 glide .25/.12
        v8 dual-stream | v9/v10 splice | v11 glide+debounce | d1/d2 scrape demo | v12 chunked+ctx dynamic | v14 +SOLA
Findings so far (user-scored): v0=9, v7/v11=7, v6=6, v8/v9/v10~3(blur), v12=10Hz tremor, v4/v5=1.
See memory pitch-transition-scrape-diagnosis. CTX/CHUNK knobs (v12 only): hops of 10ms, default 8/10.

V0 = Solo Choir streaming (bare 10ms hops)
V2 = vcclient-style: per 100ms chunk, re-feed 80ms context, discard warm-up output
V3 = V0 with 1.7x input gain (normalized after)
"""
import sys, os
import numpy as np, soundfile as sf, soxr

SRV = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer/server"
sys.path.insert(0, SRV); os.chdir(SRV)
from voice_changer.SoloChoir import SoloChoirHarmonizer
from beatrice_converter import BeatriceSoloChoir

FEMALE = sys.argv[3] if len(sys.argv)>3 else "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260620/result/girl/paraphernalia_data_00002500"
MEL    = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer/recordings/studio_mel_20260626_001508.wav"
OUT    = sys.argv[1]
HOP=160; OUTHOP=240; CHUNK=1600; CTX=1280   # @16k: 100ms chunk, 80ms context

x48, sr = sf.read(MEL)
if x48.ndim>1: x48=x48[:,0]
x16 = soxr.resample(x48.astype(np.float32), sr, 16000).astype(np.float32)
n = (len(x16)//HOP)*HOP; x16=x16[:n]

def conv(iv=None):
    h = SoloChoirHarmonizer(enabled=False) if iv is None else SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    return BeatriceSoloChoir(FEMALE, engine_dir=None, harmonizer=h, target_speaker=0)

def render_stream(x, gain=1.0, iv=None):
    c=conv(iv); out=[]
    for i in range(0, len(x), HOP):
        out.append(c.process_frame(x[i:i+HOP]*gain))
    return np.concatenate(out)

def render_ctx(x):
    """per 100ms chunk: re-feed the previous 80ms as warm-up (discard), keep the chunk."""
    c=conv(); kept=[]
    for s in range(0, len(x), CHUNK):
        blk = x[s:s+CHUNK]
        if len(blk) < HOP: break
        ctx = x[max(0, s-CTX):s]
        seq = np.concatenate([ctx, blk])
        seq = seq[:(len(seq)//HOP)*HOP]
        outs = [c.process_frame(seq[i:i+HOP]) for i in range(0, len(seq), HOP)]
        y = np.concatenate(outs)
        keep = (len(blk)//HOP)*OUTHOP          # output samples belonging to the chunk (tail)
        kept.append(y[-keep:])
    return np.concatenate(kept)

class GlideSB:
    """Wrap the beatrice engine: pitch-shift changes GLIDE in small steps instead of jumping.
    step in semitones per tick (tick = one 10ms hop)."""
    def __init__(self, sb, step=0.25):
        self._sb=sb; self._cur=0.0; self._tgt=0.0; self._step=float(step)
    def set_pitch_shift_semitone(self, v): self._tgt=float(v)   # converter sets the TARGET
    def tick(self):
        if self._cur != self._tgt:
            d=self._tgt-self._cur
            d=max(-self._step, min(self._step, d))
            self._cur=self._tgt if abs(self._tgt-self._cur)<=self._step else self._cur+d
            self._sb.set_pitch_shift_semitone(self._cur)
    def __getattr__(self,n): return getattr(self._sb,n)

def render_glide(x, iv, step):
    c=conv(iv); c.sb=GlideSB(c.sb, step); out=[]
    for i in range(0, len(x), HOP):
        c.sb.tick()
        out.append(c.process_frame(x[i:i+HOP]))
    return np.concatenate(out)

def render_xfade(x, iv, xf_hops=4, settle_hops=2):
    """Dual-stream: both converters (harmonizer OFF) run every hop; shift changes are applied to the
    SILENT standby stream, allowed to settle, then a short crossfade swaps streams. Scrape hidden."""
    from beatrice_converter import estimate_f0_autocorr
    cA=conv(None); cB=conv(None)              # pure conversion, shift driven externally
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192, dtype=np.float32)
    streams=[cA,cB]; shifts=[0.0,0.0]; act=0   # active index
    pending=None                                # (standby_settle_countdown)
    fade=None                                   # remaining fade hops (fading act->standby)
    out=[]
    for i in range(0, len(x), HOP):
        seg=x[i:i+HOP]
        # external f0 -> desired shift (same math as the converter)
        f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        f0=estimate_f0_autocorr(f0buf,16000)
        sh=harm.compute_shift(np.array([f0],dtype=np.float32))
        desired=float(sh) if sh is not None else shifts[act]
        stb=1-act
        if desired!=shifts[act] and pending is None and fade is None:
            streams[stb].sb.set_pitch_shift_semitone(desired); shifts[stb]=desired
            pending=settle_hops
        yA=streams[act].process_frame(seg); yB=streams[stb].process_frame(seg)
        if pending is not None:
            pending-=1
            if pending<=0: pending=None; fade=xf_hops
            out.append(yA)
        elif fade is not None:
            k=(xf_hops-fade+1)/xf_hops          # 0->1 across the fade
            g=np.linspace((xf_hops-fade)/xf_hops, k, len(yA)).astype(np.float32)
            out.append(yA*np.sqrt(1-g)+yB*np.sqrt(g))   # equal-power
            fade-=1
            if fade<=0: fade=None; act=stb
        else:
            out.append(yA)
    return np.concatenate(out)

def render_splice(x, iv, xf_hops=2):
    """OFFLINE: render the whole clip once per DISTINCT shift value (constant shift = stable),
    then splice along the shift timeline with short crossfades."""
    from beatrice_converter import estimate_f0_autocorr
    # pass 1: shift timeline (same external math as the live path)
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192,dtype=np.float32); tl=[]
    for i in range(0,len(x),HOP):
        seg=x[i:i+HOP]; f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        sh=harm.compute_shift(np.array([estimate_f0_autocorr(f0buf,16000)],dtype=np.float32))
        tl.append(float(sh) if sh is not None else (tl[-1] if tl else 0.0))
    first=next((v for v in tl if v!=0.0), 0.0)
    tl=[v if v!=0.0 else first for v in tl] if first!=0.0 else tl   # backfill leading None-era zeros
    distinct=sorted(set(tl))
    print("distinct shifts:", distinct)
    # pass 2: one STABLE full render per distinct shift (pitch_offset set once -> constant)
    R={}
    for v in distinct:
        c=conv(None); c.pitch_offset=float(v)
        R[v]=np.concatenate([c.process_frame(x[i:i+HOP]) for i in range(0,len(x),HOP)])
    # pass 3: splice with short crossfades at changes
    out=np.zeros(len(tl)*OUTHOP, dtype=np.float32); cur=tl[0]; fade=0; prev=cur
    for k,v in enumerate(tl):
        a,b=k*OUTHOP,(k+1)*OUTHOP
        if v!=cur: prev=cur; cur=v; fade=xf_hops
        if fade>0:
            g=np.linspace(1-(fade)/xf_hops, 1-(fade-1)/xf_hops, OUTHOP).astype(np.float32)
            out[a:b]=R[prev][a:b]*np.sqrt(1-g)+R[cur][a:b]*np.sqrt(g); fade-=1
        else:
            out[a:b]=R[cur][a:b]
    return out

def render_splice2(x, iv, win_ms=30, fade_ms=5):
    """V10: per-shift stable renders, spliced at the quietest point near each transition
    (energy-valley splice) with a tiny fade — phase incoherence masked by silence."""
    from beatrice_converter import estimate_f0_autocorr
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192,dtype=np.float32); tl=[]
    for i in range(0,len(x),HOP):
        seg=x[i:i+HOP]; f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        sh=harm.compute_shift(np.array([estimate_f0_autocorr(f0buf,16000)],dtype=np.float32))
        tl.append(float(sh) if sh is not None else (tl[-1] if tl else 0.0))
    first=next((v for v in tl if v!=0.0), 0.0)
    tl=[v if v!=0.0 else first for v in tl] if first!=0.0 else tl
    R={}
    for v in sorted(set(tl)):
        c=conv(None); c.pitch_offset=float(v)
        R[v]=np.concatenate([c.process_frame(x[i:i+HOP]) for i in range(0,len(x),HOP)])
    N=len(tl)*OUTHOP; win=int(win_ms/1000*24000); F=int(fade_ms/1000*24000)
    # transitions in output samples
    trans=[(k*OUTHOP, tl[k-1], tl[k]) for k in range(1,len(tl)) if tl[k]!=tl[k-1]]
    print(f"transitions: {len(trans)}, distinct: {sorted(set(tl))}")
    # smoothed joint energy envelope for valley search
    env=np.abs(R[tl[0]]).copy()
    for v in R: env=np.minimum(env, np.convolve(np.abs(R[v]), np.ones(48)/48, mode="same")) if False else env
    out=np.zeros(N,dtype=np.float32); pos=0; cur=tl[0]
    for (t, prev, nxt) in trans:
        lo,hi=max(pos+F,t-win),min(N-F,t+win)
        if hi<=lo: sp=t
        else:
            e=np.convolve(np.abs(R[prev][lo:hi])+np.abs(R[nxt][lo:hi]), np.ones(48)/48, mode="same")
            sp=lo+int(np.argmin(e))
        out[pos:sp]=R[cur][pos:sp]
        g=np.linspace(0,1,F).astype(np.float32)
        out[sp:sp+F]=R[cur][sp:sp+F]*np.sqrt(1-g)+R[nxt][sp:sp+F]*np.sqrt(g)
        pos=sp+F; cur=nxt
    out[pos:]=R[cur][pos:]
    return out

def render_glide_hold(x, iv, step=0.12, hold_hops=8):
    """V11: glide + debounce — a NEW shift target must persist >= hold_hops before being accepted.
    Kills timeline flapping; only genuine note changes glide."""
    from beatrice_converter import estimate_f0_autocorr
    c=conv(None); c.sb=GlideSB(c.sb, step)
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192,dtype=np.float32)
    accepted=None; cand=None; cnt=0; out=[]; nacc=0
    for i in range(0,len(x),HOP):
        seg=x[i:i+HOP]
        f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        sh=harm.compute_shift(np.array([estimate_f0_autocorr(f0buf,16000)],dtype=np.float32))
        prop=float(sh) if sh is not None else accepted
        if prop is not None:
            if accepted is None: accepted=prop; c.sb.set_pitch_shift_semitone(accepted); nacc+=1
            elif prop!=accepted:
                if prop==cand: cnt+=1
                else: cand=prop; cnt=1
                if cnt>=hold_hops: accepted=prop; c.sb.set_pitch_shift_semitone(accepted); cand=None; cnt=0; nacc+=1
            else: cand=None; cnt=0
        c.sb.tick()
        out.append(c.process_frame(seg))
    print(f"accepted transitions: {nacc-1}")
    return np.concatenate(out)

def render_scrape_demo(x, toggle_hops):
    """D2: constant -3 vs toggling -3/-4 every toggle_hops — isolates THE scrape sound."""
    c=conv(None); out=[]; cur=-3.0
    for k,i in enumerate(range(0,len(x),HOP)):
        if toggle_hops and k%toggle_hops==0 and k>0:
            cur = -4.0 if cur==-3.0 else -3.0
        c.pitch_offset=cur
        out.append(c.process_frame(x[i:i+HOP]))
    return np.concatenate(out)

def render_v12(x, iv, ctx_hops=8, chunk_hops=10, xf=240):
    """V12 = the vcclient trick under DYNAMIC harmony: per-chunk shift, the set-call scrape lands in
    the DISCARDED context warm-up; 10ms seam crossfade between kept chunks."""
    from beatrice_converter import estimate_f0_autocorr
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192,dtype=np.float32); tl=[]
    for i in range(0,len(x),HOP):
        seg=x[i:i+HOP]; f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        sh=harm.compute_shift(np.array([estimate_f0_autocorr(f0buf,16000)],dtype=np.float32))
        tl.append(float(sh) if sh is not None else (tl[-1] if tl else 0.0))
    first=next((v for v in tl if v!=0.0),0.0); tl=[v if v!=0.0 else first for v in tl] if first!=0.0 else tl
    c=conv(None); pieces=[]; prev_tail=None; nhops=len(tl)
    for s0 in range(0, nhops, chunk_hops):
        hops=list(range(max(0,s0-ctx_hops), min(nhops, s0+chunk_hops)))
        # chunk shift = majority over the KEPT hops
        kept=[k for k in hops if k>=s0]
        vals=[tl[k] for k in kept]; shift=max(set(vals), key=vals.count)
        c.pitch_offset=float(shift)                     # scrape fires on the FIRST (discarded) ctx hop
        ys=[c.process_frame(x[k*HOP:(k+1)*HOP]) for k in hops]
        y=np.concatenate(ys); keep=y[-len(kept)*OUTHOP:]
        if prev_tail is not None:
            g=np.linspace(0,1,xf).astype(np.float32)
            keep=keep.copy(); keep[:xf]=prev_tail*np.sqrt(1-g)+keep[:xf]*np.sqrt(g)
        pieces.append(keep[:-xf]); prev_tail=keep[-xf:]
    pieces.append(prev_tail)
    return np.concatenate(pieces)

def render_v14(x, iv, ctx_hops=32, chunk_hops=10, xf=240, sola=288):
    """V14 = FULL vcclient recipe on dynamic harmony: chunk + context + SOLA(12ms) + 50ms crossfade.
    Seam phase aligned by cross-correlation search before fading."""
    from beatrice_converter import estimate_f0_autocorr
    harm=SoloChoirHarmonizer(enabled=True, key_root=0, minor=False, interval_steps=iv)
    f0buf=np.zeros(8192,dtype=np.float32); tl=[]
    for i in range(0,len(x),HOP):
        seg=x[i:i+HOP]; f0buf=np.roll(f0buf,-len(seg)); f0buf[-len(seg):]=seg
        sh=harm.compute_shift(np.array([estimate_f0_autocorr(f0buf,16000)],dtype=np.float32))
        tl.append(float(sh) if sh is not None else (tl[-1] if tl else 0.0))
    first=next((v for v in tl if v!=0.0),0.0); tl=[v if v!=0.0 else first for v in tl] if first!=0.0 else tl
    c=conv(None); pieces=[]; prev_tail=None; nhops=len(tl)
    for s0 in range(0, nhops, chunk_hops):
        hops=list(range(max(0,s0-ctx_hops), min(nhops, s0+chunk_hops)))
        kept=[k for k in hops if k>=s0]
        vals=[tl[k] for k in kept]; shift=max(set(vals), key=vals.count)
        c.pitch_offset=float(shift)
        y=np.concatenate([c.process_frame(x[k*HOP:(k+1)*HOP]) for k in hops])
        keep_n=len(kept)*OUTHOP
        ext=y[-(keep_n+sola+xf):] if len(y)>=keep_n+sola+xf else y[-keep_n:]
        if prev_tail is None:
            pieces.append(ext[sola:-xf] if len(ext)>sola+xf else ext); prev_tail=ext[-xf:]
            continue
        best,bc=0,-1e18
        for o in range(0, min(sola,len(ext)-xf)+1, 2):     # SOLA search, 2-sample grid
            seg=ext[o:o+xf]; cc=float(np.dot(prev_tail,seg))/(float(np.linalg.norm(seg))+1e-9)
            if cc>bc: bc,best=cc,o
        g=np.linspace(0,1,xf).astype(np.float32)
        pieces.append(prev_tail*np.sqrt(1-g)+ext[best:best+xf]*np.sqrt(g))
        rest=ext[best+xf:]
        if len(rest)>xf: pieces.append(rest[:-xf]); prev_tail=rest[-xf:]
        else: prev_tail=rest if len(rest)==xf else np.pad(rest,(0,xf-len(rest)))
    pieces.append(prev_tail)
    return np.concatenate(pieces)

def norm(y): 
    p=float(np.max(np.abs(y)))+1e-9; return (y/p*0.5).astype(np.float32)

mode = sys.argv[2]
if   mode=="v0": y = render_stream(x16)
elif mode=="v2": y = render_ctx(x16)
elif mode=="v3": y = render_stream(x16, gain=1.7)
elif mode=="v4": y = render_stream(x16, iv=-2)   # dynamic diatonic shift, 3rd below (the live harmony path)
elif mode=="v5": y = render_stream(x16, iv=0)    # scale-snap only (micro dynamic shifts)
elif mode=="v6": y = render_glide(x16, -2, 0.25)  # dynamic -2 with GLIDE (25c per 10ms)
elif mode=="v7": y = render_glide(x16, -2, 0.12)  # slower glide (12c per 10ms)
elif mode=="v8": y = render_xfade(x16, -2)        # dual-stream crossfade at transitions
elif mode=="v9": y = render_splice(x16, -2)       # OFFLINE: per-shift stable renders spliced
elif mode=="v10": y = render_splice2(x16, -2)     # OFFLINE: energy-valley splice + 5ms fade
elif mode=="v11": y = render_glide_hold(x16, -2, 0.12, 8)   # glide + 80ms debounce
elif mode=="v15": y = render_glide_hold(x16, -2, 0.06, 8)   # finer glide 6c/10ms + debounce
elif mode=="v16": y = render_glide_hold(x16, -2, 0.03, 8)   # finest 3c/10ms + debounce
elif mode=="d1": y = render_scrape_demo(x16, 0)     # DEMO: constant -3 (no scrapes)
elif mode=="d2": y = render_scrape_demo(x16, 40)    # DEMO: toggle -3/-4 every 400ms (hear the scrape)
elif mode=="v14":
    _ctx=int(sys.argv[4]) if len(sys.argv)>4 else 32
    _chk=int(sys.argv[5]) if len(sys.argv)>5 else 10
    print(f"v14 knobs: ctx={_ctx*10}ms chunk={_chk*10}ms sola=12ms xf=50ms")
    y = render_v14(x16, -2, ctx_hops=_ctx, chunk_hops=_chk)
elif mode=="v12":
    _ctx=int(sys.argv[4]) if len(sys.argv)>4 else 8
    _chk=int(sys.argv[5]) if len(sys.argv)>5 else 10
    _xf=int(float(sys.argv[6])/1000*24000) if len(sys.argv)>6 else 240
    print(f"v12 knobs: ctx={_ctx*10}ms chunk={_chk*10}ms xf={_xf/24:.0f}ms")
    y = render_v12(x16, -2, ctx_hops=_ctx, chunk_hops=_chk, xf=_xf)
sf.write(OUT, norm(y), 24000)
print(f"{mode}: wrote {OUT} dur={len(y)/24000:.1f}s")
