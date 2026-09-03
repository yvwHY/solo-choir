"""phrase_render.py — 行程內單句渲染器：離線正典品質、模型常駐（08-01）

應答式 v2 的引擎。離線正典（direct_mouth → DDSP main.py 子行程）品質對，但
每句付一次模型載入＝5 秒句實測 **6.69s**，做不到「句尾後 ~1s 回應」。本檔把
同一條配方搬進常駐行程：模型/encoder/vol_extractor 只載一次，之後每句只付
真正的推論。

配方逐項對齊 direct_mouth + main.py（差一項就不是同一張嘴）：
  1. input_agc（只作用於餵模型那份；f0/voicing/透傳一律用原始 x）
  2. f0 = express 骨架 × expressive_cents（F21 表現層）
  3. voicing_mask → f0 歸零 → uv 哨兵 1200Hz（硬 0 會 0/0 爆音）
  4. units = encoder.encode(x_agc)、vol = vol_ex.extract(x_agc)
  5. **volume mask**（main.py:208-213：-60dB 門檻、前後補 4、9 幀 max 膨脹）
     ——這項只存在於 main.py，direct_mouth 看不到，漏了就多出底噪段
  6. model(units, f0, vol, spk_id) → × mask
  7. ÷ agc_g 還原位準 → uv_passthrough 補回他的原聲無聲段

驗證：`--verify` 對同一段音訊跑「本檔 vs direct_mouth 子行程」，比 self-noise
地板（MPS 渲染不可重現，見 worklog 08-01 §D）。
"""
import numpy as np

import direct_mouth as dm

SR, BLOCK = dm.SR, dm.HOP


class PhraseRenderer:
    """一張常駐的離線品質嘴。render() 可重複呼叫，模型只載一次。"""

    def __init__(self, repo, model_path, device="mps", expr_seed=20260731,
                 spk_id=1, enhance=False):
        import sys
        import torch
        sys.path.insert(0, str(repo))
        from ddsp.vocoder import Units_Encoder, Volume_Extractor, load_model
        self.torch, self.device = torch, device
        self.model, margs = load_model(str(model_path), device=device)
        # 08-05 autotune 案終判線索（Harry：「enh 檔其實還好」）：低音域
        # CombSub 裸輸出的死平諧波＝autotune 感來源；NSF-HiFiGAN enhancer
        # 重整後消失。守衛參數：False＝逐 byte 舊路徑（live/respond2 不變）。
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
        """main.py:208-213 的 volume mask（門檻＋9 幀 max 膨脹），逐行對齊。"""
        m = (vol > 10 ** (thresh_db / 20)).astype(float)
        m = np.pad(m, (4, 4), constant_values=(m[0], m[-1]))
        return np.array([m[n:n + 9].max() for n in range(len(m) - 8)])

    def render(self, x, notes, tick, expr_gain=1.0, porta_ms=33.0, leap_snap=4,
               shared=None, tune_seed=None, tune_lock=0.0, uv_dry=1.0,
               vib_hz=dm.VIB_HZ, vib_phase=0.0, vib_onset_ms=0.0, vib_semi=0.0,
               prosody_cents=None, sustain_ms=0.0):
        """x＝他這一句的原聲（float64 mono @44.1k）、notes＝該句每 tick 的
        目標音（None＝休止）、tick＝每 tick 樣本數。回傳同長度天使音訊。

        shared＝跨聲部快取 dict（兩張嘴吃同一段音訊＝所有「只看 x」的東西都
        相同：harvest、input_agc、voicing_mask、hubert units、volume。快取的
        是**同一個函式對同一份輸入的結果**＝逐 byte 相同，不是近似。
        08-03 實測 5.16s 句：兩聲部 4.07s → 1.75s（同 DDSPMouth.hop 的慣例）。"""
        torch = self.torch
        n_hops = len(x) // BLOCK + 1
        sh = shared if shared is not None else {}
        if sh.get("n") not in (None, len(x)):     # 換句沒清快取＝拿到別句的臉
            raise ValueError(f"shared 快取是 {sh['n']} 樣本的句子，這句 {len(x)}")
        sh["n"] = len(x)
        # 這一段內腦全判休止（應答式會遇到）＝天使不出聲。必須早退：
        # expressive_cents 對零音符段會空陣列索引爆（direct_mouth.py:467）。
        if not any(notes[min(int(k * BLOCK / tick), len(notes) - 1)] is not None
                   for k in range(n_hops)):
            return np.zeros(len(x))
        if "f0m" not in sh:
            sh["f0m"] = dm.harvest_f0(x)
        if "agc" not in sh:
            sh["agc"] = dm.input_agc(x, f0m=sh["f0m"])
        x_in, agc_g = sh["agc"]

        # vib_semi>0＝blind_v3（07-28）的配方：顫音是骨架的一部分（固定速率、
        # 兩聲部去同步），不是表現層生的。預設 0＝顫音交給 expressive_cents
        # ＝舊行為。08-02 Harry 裁決：blind_v3 那條路的和音「對了」。
        skel = dm.target_f0(notes, tick, n_hops, porta_ms, leap_snap,
                            vib_hz=vib_hz, vib_phase=vib_phase,
                            vib_onset_ms=vib_onset_ms, vib_semi=vib_semi)
        cents = dm.expressive_cents(notes, tick, n_hops, leap_snap, self.seed,
                                    tune_seed=tune_seed, tune_lock=tune_lock)
        f0 = skel * 2 ** (expr_gain * cents / 1200.0)
        if prosody_cents is not None:
            # 08-05 prosody 移植（Harry:「有 auto tune 就是不行」）：他真實 f0
            # 的慢成分（cents、per-hop），疊到骨架上。兩聲部共用同一條曲線＝
            # 音程恆定——08-02 express 的死因（雙聲部各偏各的）在構造上不可能
            # 重現。None＝完全不走這行＝逐 byte 舊行為。
            pc = np.asarray(prosody_cents, dtype=np.float64)
            pc = (pc[:n_hops] if len(pc) >= n_hops
                  else np.pad(pc, (0, n_hops - len(pc))))
            f0 = f0 * 2 ** (pc / 1200.0)
        if "vm" not in sh:
            sh["vm"] = dm.voicing_mask(x, n_hops, f0m=sh["f0m"])
        vm = sh["vm"]
        bridges = []
        if sustain_ms > 0:
            # 08-05 §O E：天使像歌手一樣把母音撐過短空隙（子音/換氣量級），
            # 只在真休止收——他的逐幀 voicing 不再原樣複製給天使。
            # 快取的 vm 不動（其他聲部/模式共用），這裡用橋接副本。
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
        f0 = np.where(f0 * vm > 0, f0 * vm, 1200.0)   # uv 哨兵（硬 0 會爆音）

        with torch.no_grad():
            if sh.get("units") is None:
                au = torch.from_numpy(x_in).float().unsqueeze(0).to(self.device)
                sh["units"] = self.encoder.encode(au, SR, BLOCK)
                sh["vol"] = self.vol_ex.extract(x_in)
            units, vol_all = sh["units"], sh["vol"]
            if bridges:
                # 橋接段：凍住最後一個有聲幀的母音（units）、音量兩端插值
                # ——不然合成器會拿他的子音幀配持續 f0 唱出怪聲。副本，不動快取。
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
                # 對齊 main.py:259-267：mask 先乘、再 enhance（在 ÷agc_g 前
                # ＝enhancer 看到的位準與 main.py 相同）。輸出同 44.1k/512。
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
        wav = wav[:k] / agc_g[:k]                      # 位準結構還原
        ang, _ = dm.uv_passthrough(wav, x[:k], vm, dry_gain=uv_dry)
        return ang


def _verify(repo, model, take, notes_json, tag="pv"):
    """本檔 vs direct_mouth 子行程：同一段音訊、同一份音符線。"""
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
    r.render(np.zeros(SR), [60], tick)                 # 熱身（MPS kernel 編譯）
    t0 = time.perf_counter()
    mine = r.render(x, notes, tick)
    t_render = time.perf_counter() - t0
    print(f"in-process: 載入 {t_load:.2f}s（一次）｜渲染 {t_render:.2f}s "
          f"／{len(x)/SR:.1f}s 音訊 = RTF {t_render/(len(x)/SR):.2f}")

    nj = f"out/{tag}_notes.json"
    json.dump({"notes": notes, "step_samps": tick}, open(nj, "w"))
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "direct_mouth.py", "--notes", nj,
                    "--take", take, "--tag", tag, "--model", model,
                    "--key", "0", "--f0-mode", "express", "--run"],
                   check=True, capture_output=True)
    print(f"subprocess: {time.perf_counter() - t0:.2f}s（每句都要付）")
    ref, _ = sf.read(f"out/{tag}_angel.wav", dtype="float64")
    n = min(len(mine), len(ref))
    d_ = np.abs(mine[:n] - ref[:n])
    print(f"對照: max|diff| {d_.max():.4f} mean {d_.mean():.5f} "
          f"corr {np.corrcoef(mine[:n], ref[:n])[0,1]:.4f}")
    sf.write(f"out/{tag}_inproc.wav", mine, SR)
    print(f"wrote out/{tag}_inproc.wav（耳測對照 vs out/{tag}_angel.wav）")


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
