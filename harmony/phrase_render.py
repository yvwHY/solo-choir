"""phrase_render.py — an in-process single-phrase renderer: offline quality with
the model resident (2026-08-01)

The engine of the answering mode v2. The offline reference path
(direct_mouth calling DDSP's main.py as a subprocess) has the right quality, but
pays a model load per phrase: a 5 s phrase measured **6.69 s**, which cannot
deliver a response about a second after the phrase ends. This file moves the
same recipe into a resident process, where the model, encoder and volume
extractor are loaded once and each phrase pays only for the inference itself.

The recipe matches direct_mouth plus main.py item by item; one difference and it
is not the same voice:
  1. input_agc (applied only to the copy fed to the model; f0, voicing and
     pass-through always use the original x)
  2. f0 = the express skeleton times expressive_cents (the F21 expression layer)
  3. voicing_mask, then f0 zeroed, then a 1200 Hz unvoiced sentinel (a hard 0
     gives a 0/0 blow-up)
  5. **the volume mask** (main.py:208-213: a -60 dB threshold, 4 frames of pad
     either side, and a 9-frame max dilation) — this exists only in main.py and
     is invisible from direct_mouth; omitting it leaves extra noise-floor
     passages
  7. divide by agc_g to restore the level, then uv_passthrough returns the
     singer's own unvoiced sections

Verification: `--verify` runs the same audio through this file and through the
direct_mouth subprocess and compares against the self-noise floor, since MPS
rendering is not reproducible (see the worklog for 2026-08-01 section D).
"""
import numpy as np

import direct_mouth as dm

SR, BLOCK = dm.SR, dm.HOP


class PhraseRenderer:
    """A resident voice at offline quality. render() may be called repeatedly;
    the model loads once."""

    def __init__(self, repo, model_path, device="mps", expr_seed=20260731,
                 spk_id=1, enhance=False):
        import sys
        import torch
        sys.path.insert(0, str(repo))
        from ddsp.vocoder import Units_Encoder, Volume_Extractor, load_model
        self.torch, self.device = torch, device
        self.model, margs = load_model(str(model_path), device=device)
        # The clue that closed the auto-tune case on 2026-08-05 ("the enhanced
        # file is actually fine"): the flat, lifeless harmonics of the bare
        # CombSub output in the low range are the source of the corrected sound,
        # and they disappear once the NSF-HiFiGAN enhancer reshapes it. This is
        # a guard parameter: False is the byte-identical old path, leaving live
        # and respond2 unchanged.
        self.enhancer = None
        if enhance:
            from enhancer import Enhancer
            self.enhancer = Enhancer(margs.enhancer.type,
                                     f"{repo}/{margs.enhancer.ckpt}",
                                     device=device)
        self.encoder = Units_Encoder(
            margs.data.encoder, str(repo) + "/" + margs.data.encoder_ckpt,
            margs.data.encoder_sample_rate, margs.data.encoder_hop_size,
            device=device)
        self.vol_ex = Volume_Extractor(BLOCK)
        self.spk = torch.LongTensor([[spk_id]]).to(device)
        self.seed = expr_seed
        assert int(margs.data.block_size) == BLOCK
        assert int(margs.data.sampling_rate) == SR

    def _vol_mask(self, vol, thresh_db=-60.0):
        """The volume mask of main.py:208-213, a threshold plus a 9-frame max
        dilation, aligned line for line."""
        m = (vol > 10 ** (thresh_db / 20)).astype(float)
        m = np.pad(m, (4, 4), constant_values=(m[0], m[-1]))
        return np.array([m[n:n + 9].max() for n in range(len(m) - 8)])

    def render(self, x, notes, tick, expr_gain=1.0, porta_ms=33.0, leap_snap=4,
               shared=None, tune_seed=None, tune_lock=0.0, uv_dry=1.0,
               vib_hz=dm.VIB_HZ, vib_phase=0.0, vib_onset_ms=0.0, vib_semi=0.0,
               prosody_cents=None, sustain_ms=0.0):
        """x is the singer's own audio for this phrase (float64 mono at
        44.1 kHz), notes is the target note per tick (None for a rest), and tick
        is the samples per tick. Returns audio of the same length.

        shared is a cross-part cache dict. Both voices take the same audio, so
        everything that looks only at x is identical: harvest, input_agc,
        voicing_mask, HuBERT units and volume. What is cached is **the result of
        the same function on the same input**, so it is byte-identical rather
        than approximate.
        Measured on 2026-08-03 with a 5.16 s phrase: two parts fell from 4.07 s
        to 1.75 s, following the same convention as DDSPMouth.hop."""
        torch = self.torch
        n_hops = len(x) // BLOCK + 1
        sh = shared if shared is not None else {}
        if sh.get("n") not in (None, len(x)):     # a cache not cleared between phrases would return another phrase's voice
            raise ValueError(f"the shared cache holds a phrase of {sh['n']} samples; this one has {len(x)}")
        sh["n"] = len(x)
        # The model called the whole span a rest, which happens in the answering
        # mode, so the parts stay silent. This must return early, because
        # expressive_cents indexes an empty array on a span with no notes and
        # raises (direct_mouth.py:467).
        if not any(notes[min(int(k * BLOCK / tick), len(notes) - 1)] is not None
                   for k in range(n_hops)):
            return np.zeros(len(x))
        if "f0m" not in sh:
            sh["f0m"] = dm.harvest_f0(x)
        if "agc" not in sh:
            sh["agc"] = dm.input_agc(x, f0m=sh["f0m"])
        x_in, agc_g = sh["agc"]

        # vib_semi > 0 is the blind_v3 recipe of 2026-07-28: vibrato is part of
        # the skeleton, at a fixed rate and desynchronised between the parts,
        # rather than generated by the expression layer. The default of 0 leaves
        # vibrato to expressive_cents, which is the old behaviour. Judged on
        # 2026-08-02: the harmony on the blind_v3 path was right.
        skel = dm.target_f0(notes, tick, n_hops, porta_ms, leap_snap,
                            vib_hz=vib_hz, vib_phase=vib_phase,
                            vib_onset_ms=vib_onset_ms, vib_semi=vib_semi)
        cents = dm.expressive_cents(notes, tick, n_hops, leap_snap, self.seed,
                                    tune_seed=tune_seed, tune_lock=tune_lock)
        f0 = skel * 2 ** (expr_gain * cents / 1200.0)
        if prosody_cents is not None:
            # Prosody transplant, 2026-08-05, after "anything auto-tuned is
            # simply out". The slow component of the singer's real f0, in cents
            # per hop, is laid over the skeleton. Both parts share one curve, so
            # the interval between them is constant, which makes the failure that
            # killed express on 2026-08-02, each part drifting on its own,
            # structurally impossible. None skips this line entirely, which is
            # the byte-identical old behaviour.
            pc = np.asarray(prosody_cents, dtype=np.float64)
            pc = (pc[:n_hops] if len(pc) >= n_hops
                  else np.pad(pc, (0, n_hops - len(pc))))
            f0 = f0 * 2 ** (pc / 1200.0)
        if "vm" not in sh:
            sh["vm"] = dm.voicing_mask(x, n_hops, f0m=sh["f0m"])
        vm = sh["vm"]
        bridges = []
        if sustain_ms > 0:
            # 2026-08-05 section O, E: the parts hold a vowel across short gaps,
            # of the order of a consonant or a breath, as a singer does, and stop
            # only at a real rest. The singer's frame-by-frame voicing is no
            # longer copied onto them.
            # The cached vm is left alone, since other parts and modes share it;
            # a bridged copy is used here.
            vm = np.copy(vm)
            gmax = max(1, int(sustain_ms / 1000.0 * SR / BLOCK))
            h = 0
            while h < len(vm):
                if vm[h] <= 0:
                    j = h
                    while j < len(vm) and vm[j] <= 0:
                        j += 1
                    if 0 < h and j < len(vm) and (j - h) <= gmax:
                        vm[h:j] = 1.0
                        bridges.append((h, j))
                    h = j
                else:
                    h += 1
        f0 = np.where(f0 * vm > 0, f0 * vm, 1200.0)   # unvoiced sentinel; a hard 0 blows up

        with torch.no_grad():
            if sh.get("units") is None:
                au = torch.from_numpy(x_in).float().unsqueeze(0).to(self.device)
                sh["units"] = self.encoder.encode(au, SR, BLOCK)
                sh["vol"] = self.vol_ex.extract(x_in)
            units, vol_all = sh["units"], sh["vol"]
            if bridges:
                # Bridged spans freeze the vowel, that is, the units, of the last
                # voiced frame and interpolate the volume across the gap.
                # Otherwise the synthesiser sings the singer's consonant frames
                # against a sustained f0 and produces something strange. This
                # works on a copy and leaves the cache alone.
                units = units.clone()
                vol_all = np.copy(vol_all)
                U = units.size(1)
                for (bi, bj) in bridges:
                    li = min(bi - 1, U - 1)
                    jj = min(bj, U)
                    if li < 0 or bi >= U:
                        continue
                    units[0, bi:jj, :] = units[0, li, :]
                    if bi < len(vol_all):
                        vj = min(bj, len(vol_all) - 1)
                        vol_all[bi:jj] = np.linspace(
                            vol_all[li], vol_all[vj], jj - bi)
            n = min(units.size(1), n_hops)
            vol = vol_all[:n]
            mask = self._vol_mask(vol)[:n]
            f0_t = torch.from_numpy(f0[:n]).float().to(self.device)[None, :, None]
            vol_t = torch.from_numpy(vol).float().to(self.device)[None, :, None]
            out, _, _ = self.model(units[:, :n], f0_t, vol_t, spk_id=self.spk)
            if self.enhancer is not None:
                # Aligned with main.py:259-267: multiply by the mask first, then
                # enhance, before dividing by agc_g, so the enhancer sees the same
                # level as in main.py. The output is the same 44.1 kHz at 512.
                mk = torch.from_numpy(
                    np.repeat(mask, BLOCK)).float().to(self.device)[None, :]
                t = min(out.size(1), mk.size(1))
                o, esr = self.enhancer.enhance(
                    out[:, :t] * mk[:, :t], SR, f0_t, BLOCK, adaptive_key=0)
                assert esr == SR
                wav = o.squeeze().cpu().numpy().astype(np.float64)
            else:
                wav = out.squeeze().cpu().numpy().astype(np.float64)
        if self.enhancer is None:
            m_up = np.repeat(mask, BLOCK)[: len(wav)]
            wav[: len(m_up)] *= m_up

        k = min(len(wav), len(agc_g))
        wav = wav[:k] / agc_g[:k]                      # restore the level structure
        ang, _ = dm.uv_passthrough(wav, x[:k], vm, dry_gain=uv_dry)
        return ang


def _verify(repo, model, take, notes_json, tag="pv"):
    """This file against the direct_mouth subprocess: the same audio and the
    same note line."""
    import json
    import os
    import subprocess
    import sys
    import time

    import soundfile as sf
    x, sr = sf.read(take, dtype="float64", always_2d=True)
    x, = (x[:, 0],)
    assert sr == SR
    d = json.load(open(notes_json))
    notes, tick = (d["notes"], int(d["step_samps"])) if "notes" in d else \
        (d["upper"], int(round(dm.SR * 0.1875)))

    t0 = time.perf_counter()
    r = PhraseRenderer(repo, model)
    t_load = time.perf_counter() - t0
    r.render(np.zeros(SR), [60], tick)                 # warm-up, so the MPS kernels compile
    t0 = time.perf_counter()
    mine = r.render(x, notes, tick)
    t_render = time.perf_counter() - t0
    print(f"in-process: load {t_load:.2f}s (once) | render {t_render:.2f}s "
          f"for {len(x)/SR:.1f}s of audio = RTF {t_render/(len(x)/SR):.2f}")

    nj = f"out/{tag}_notes.json"
    json.dump({"notes": notes, "step_samps": tick}, open(nj, "w"))
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "direct_mouth.py", "--notes", nj,
                    "--take", take, "--tag", tag, "--model", model,
                    "--key", "0", "--f0-mode", "express", "--run"],
                   check=True, capture_output=True)
    print(f"subprocess: {time.perf_counter() - t0:.2f}s (paid on every phrase)")
    ref, _ = sf.read(f"out/{tag}_angel.wav", dtype="float64")
    n = min(len(mine), len(ref))
    d_ = np.abs(mine[:n] - ref[:n])
    print(f"comparison: max|diff| {d_.max():.4f} mean {d_.mean():.5f} "
          f"corr {np.corrcoef(mine[:n], ref[:n])[0,1]:.4f}")
    sf.write(f"out/{tag}_inproc.wav", mine, SR)
    print(f"wrote out/{tag}_inproc.wav (listen against out/{tag}_angel.wav)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--repo", default=dm.DDSP)
    ap.add_argument("--model", default=f"{dm.DDSP}/exp/combsub-girl/model_30000.pt")
    ap.add_argument("--take", default="out/seg5s.wav")
    ap.add_argument("--notes", default="out/pair06_A_upper_notes.json")
    a = ap.parse_args()
    if a.verify:
        _verify(a.repo, a.model, a.take, a.notes)
