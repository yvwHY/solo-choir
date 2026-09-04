"""vowel_render_probe - can the model render the rounded back vowels at all?

Checking the new bank: two vowels (sources at 479/700 and 317/818) both came out as
a third vowel after rendering, with F1 pushed up to about 750. Two possibilities:
the stretches I picked happened to be poor, or the model - reflow, trained on his own
voice - simply cannot produce rounded back vowels. This renders one note from each of
several candidate stretches to tell them apart: if all eight collapse, it is the
model's character, and those two layers are honestly cut rather than wasting more
listening on them.

Run: cd harmony && ../../../260724_ddsp_svc_6x/venv/bin/python \
      scratchpad/vowel_render_probe.py
"""
import json

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

_a = {}
exec(compile(open("scratchpad/layer_vowel_audit.py").read()
             .split("\ndef main()")[0], "lva", "exec"), _a)
lpc_formants, nearest_vowel = _a["lpc_formants"], _a["nearest_vowel"]
SR, HOP = 44100, 512
MIDI = 48
MODEL = "reflow-bass1/model_32000.pt"


def meas(x):
    x16 = resample_poly(x, 16000, SR)
    n = 1600
    f = [lpc_formants(x16[i:i + n]) for i in range(0, max(1, len(x16) - n), n)]
    f = [q for q in f if len(q) >= 2]
    if not f:
        return float("nan"), float("nan")
    return (float(np.median([q[0] for q in f])),
            float(np.median([q[1] for q in f])))


def main():
    import torch
    import spike_stream6 as S
    cands = json.load(open("scratchpad/vowel_source_cands.json"))
    svc = S.Svc([(f"{S.DDSP}/exp/{MODEL}", 0.0, 1.0, 1)], step=2,
                t_start=0.85)
    NF = int(2.0 * SR / HOP)
    f0 = np.full(NF, 440.0 * 2 ** ((MIDI - 69) / 12.0))
    vol = np.full(NF, 0.06)
    vol[:4] = np.linspace(0, 0.06, 4)
    vol_t = torch.from_numpy(vol).float().to(svc.device)[None, :, None]
    mask = torch.ones(1, NF * HOP, device=svc.device)
    for v in ("喔", "嗚", "啊", "咿"):        # the last two are the control; they are known to survive
        print(f"\n{v} (source -> rendered)")
        for r in cands[v][:4]:
            x, fs = sf.read(r["clip"], always_2d=True)
            mid = (r["t0"] + r["t1"]) / 2
            a0 = max(0, int((mid - 0.5) * fs))
            seg = x[a0:a0 + int(1.0 * fs), 0].astype("float64")
            if fs != SR:
                seg = resample_poly(seg, SR, fs)
            with torch.no_grad():
                UU = svc.encode(seg)[:, 8:-8]
                nb = UU.size(1)
                torch.manual_seed(1234 + MIDI)
                au = svc.infer(np.zeros(NF * HOP), 0.0, -60.0,
                               units_override=UU[:, np.arange(NF) % nb],
                               feats=(f0, vol_t, mask),
                               ratios=[None])[0].cpu().numpy()
            F1, F2 = meas(au[int(0.5 * SR):int(1.8 * SR)].astype("float64"))
            v1, v2, _d = nearest_vowel(F1, F2)
            print(f"  {r['clip'].split('/')[-1]} {r['t0']:.1f}s  "
                  f"source {r['F1']:.0f}/{r['F2']:.0f} -> rendered {F1:.0f}/{F2:.0f}"
                  f"  = {v1} (second {v2}) {'ok' if v1 == v else 'no'}")


if __name__ == "__main__":
    main()
