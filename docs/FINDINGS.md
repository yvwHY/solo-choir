# FINDINGS — Validated Recipes & Hard-Won Facts

**Audience: AI assistants (Claude, Codex, any model) and humans.**
Positive counterpart of `GRAVEYARD.md`: conclusions that were tested and *held*, with the numbers. Treat these as ground truth until an entry is explicitly re-tested and overturned (then update here and register the old belief in GRAVEYARD).

## Engine (Beatrice) facts

### F1. Speaker/embedding surface
- There is **no raw-embedding API**. The `v20rc0.pyi` stub is stale — `get_embedding`/`set_embedding` do not exist; only discrete `set_target_speaker(int)`. Continuous timbre morphing (Vocoflex-style) is impossible through the public API.
- `num_speakers` is engine-reported per model: tenor = 1, trained female = 1, **satb2 = 2 (spk0 female / spk1 male)**, JVS = 100. `codebook_size` = 512 for all.
- To audition speakers: `server/sweep_speakers.py` (renders every `target_speaker` of a model on a dry clip + montage + manifest.csv — pick by ear).
- Beatrice is source-dependent (see GRAVEYARD G13): male→female speaker = neutral timbre. T/B (male→male) is fine. Female timbre emerges when output pitch ≥ ~A4 (use TUNE/octave). Vocoflex is **Dreamtonics** (not Voicemod) — cite correctly.
- No open-source engine matches Beatrice's niche (CPU real-time + zero-shot) as of the 2026-06 survey. Best-licensed singing dataset: M4Singer (MIT, SATB-labelled). Avoid community RVC checkpoints (IP).

### F2. Pitch changes must GLIDE, never jump (live)
The root fix for harmony "scrape" (GRAVEYARD G1–G4):
- One stateful stream per voice; change pitch inside it.
- **Glide 3–6 ¢ per 10 ms hop + 80 ms debounce = 8/10 by ear** (25 ¢/10 ms = 6/10; 12 ¢ = 7/10). Decompiled vcclient 2.2.2 does exactly this.
- Ceiling: the rc0 binding's ~5–10 ¢ pitch quantisation deadzone caps quality at ~8/10; 9+ needs a feature-domain custom binding (post-viva).
- Render-side twin (commit `cc91deb9`): shift = **score-to-score** (voice target note − transcribed melody note, constant per note), never per-frame tracked pitch; grid-pin the melody with slew-limited 2 st/s, ±80 ms note-boundary exclusion. Within-note std improved 41–76 ¢ → 13–14 ¢.

### F3. Choir decorrelation recipe (one voice → "many people")
Validated 3-0 in listening tests and cross-confirmed by Dreamtonics Choir Voices' control set:
- Requires **both pitch and timing decorrelation per part** + independent vibrato + stereo pan. Timbre variety is NOT required (GRAVEYARD G9).
- Numbers: real choir inter-singer F0 spread 0–50 ¢ (mean ~20 ¢); onset spread sweet spot **~25 ¢ pitch + ~20 ms onset std** (>40 ms degrades); ~7 well-decorrelated parts ≈ 10 real singers (diminishing returns beyond).
- Offline WORLD (pyworld) pipeline: cancel original vibrato (160 ms smoothing) → add independent vibrato (5–6.5 Hz, ~40 ¢, random phase) + slow random-walk micro-pitch + fixed detune spread → independent onset delays → equal-power stereo pan → sum. Prototype: `choir_decorr2.py` (scratchpad, 2026-07-04).

### F4. Keep live voices warm
Every live voice runs inference every block; `on/off` gates only the mix sum (GRAVEYARD G5). Output is byte-identical for any fixed voice set; spurious zero-fill 27 % → 0.1 %. Offline render may still skip off voices.

### F5. kNN-VC female render (studio "female" button)
- kNN-VC (MIT, arXiv 2305.18975) beats Beatrice's neutral female for A/T/B; **Sop uses the alto reference pool** (melody sits in male register — register match, not high-note synthesis; GRAVEYARD G12). `_VOICE_REF = {Sop: alto, Alto: alto, Tenor: tenor, Bass: bass}` in `server/studio_api.py`.
- Cost on CPU: first render ~19.5 min/clip (builds matching sets), then cached (`knn_assets/cache/*.pt`) → ~7 s/voice, ~14 s full SATB. Commit `2a56555e`.
- Genuinely convincing *live* female (HOLLY+ class) requires GPU inference — offline only on this project. Useful no-retrain lever from F0-AutoVC: renormalise source log-F0 into the female register.

### F6. Latency tuning (solo_min, live)
Measured mouth-to-ear on the speaker_set aggregate device:
| cushion | blocksize | latency | result |
|---|---|---|---|
| 10 ms | 480 | low | ~80 ms |
| 5 ms | 240 | 0.005 | ~56 ms |
| 3 ms | 240 | 0.004 | **~46 ms — clean sweet spot** |
| 1 ms | 128 | 0.004 | ~42 ms, edgy, underruns |
Floor ≈ output device ~23 ms + engine hop 10 ms → ~30 ms is unreachable; that's fine — the research framing is *liveness/simultaneity*, and ~46 ms passes. RTF ≈ 0.14 per converted voice (4 voices ≈ 0.57) → live ceiling ~3–4 converted parts on CPU; full/fancier SATB goes through offline render.

### F7. Coexistence & device facts
- Studio app + live app can run simultaneously sharing mic and output — **only if sample rates match** (48 kHz). Mismatch on one device → PortAudio AUHAL `-10851`. (Fixed: studio count-in 44.1 k → 48 k.)
- `input overflow` in a duplex stream = two independent clocks (USB mic + separate output) drifting; bigger buffers only delay it. Real fix: macOS Aggregate Device with drift correction, or one USB interface. (Re-demonstrated live 2026-07-16 evening: aggregate output + default-device input → 6 overflows; input switched onto the same aggregate → 0 for the rest of the session.)
- **Aggregate Devices (verified live 2026-07-16):** a 3-sub-device/6-output-channel aggregate (Mac speakers + 外接耳機 + AI-Micro, AI-Micro = clock master, others drift-corrected) runs solo_min 4-voice `--out-map` clean — 0 underrun/overflow once in AND out are both on the aggregate. Two gotchas: ① an `--out-map` needing more channels than the default device MUST hit the device at launch (`--out-name`), else PortAudio -9998 kills the stream; ② CLI name-matching fails on Chinese device names but works on ASCII — name aggregates in English.
- Live model default is JVS_SATB_VOICES with a speaker-index guard (a tenor-model + JVS-speaker mismatch crashed the whole choir; commit `891577e5`).

### F23. Two different Beatrice models CAN run live together — the cure is block size, not architecture (2026-08-14)

- **The old rule was half a rule.** `solo_min` carried "two different models may cache-thrash → 滋滋波波" as a flat prohibition, which is what pushed the whole SATB question toward multi-speaker joint training. It never named the threshold. There is one, and it is one flag wide.
- **Measured by ear (Harry, live, 2 × bt-sop @+12 + 2 × tenor @0):** `--blocksize 480` (10 ms, the default) = 滋滋波波; **`--blocksize 960` (20 ms) = clean**, latency acceptable; `1920` (40 ms) = clean but latency rejected. **960 is the working line.**
- **Isolated first, then fixed.** Three controls ran before touching block size, each changing exactly one variable: one model / one interval = clean; one model / **two** intervals (+12 and 0) = clean; the *other* model alone (tenor, 4 voices) = clean. Only "two models at once" reproduced it — so neither the intervals, the voice count, nor either mouth on its own is the cause.
- **Mechanism (consistent with F12):** at 10 ms the callback alternates between two weight sets ~100×/s; at 20 ms it halves the switch rate and doubles the locality per switch. Same family as F12 — the audio callback is the budget, and *what you make it do twice* is the cost.
- **Consequence:** live SATB does **not** need one joint multi-speaker model. Soprano = the B-line self-trained Soprano-3 (`bt 3000`) at **+12**; bass = the June tenor model at **0**; the singer's own dry voice is the third part. Zero further GPU.
- **Rule:** a "these two things can't coexist" note in the code is not a finding until someone has swept the parameter that mediates it. Sweep it before you accept the architecture cost of avoiding it.

### F24. Perceived voice type is set by output register, not by the model or the speaker index (2026-08-14)

- **Same mouth, three registers, three verdicts (Harry, live):** bt-sop at **+24** = "chipmunk, not a soprano"; at **+12** = a soprano; at **0** = "偏中性" — neutral, not male. One model, one speaker, nothing else changed.
- This is the same boundary G13 states from the other side ("push output pitch ≥ ~A4 and female timbre emerges") and the same one the 2026-08-12 `render_bt` lesson found offline (rendering Soprano-3 at identity makes her sing the male range and the timbre reads as "not transferred"). **Register is the first-class variable; pick it before judging any mouth.**
- **Practical consequence:** you cannot get a bass out of a soprano model by transposing down — you get neutral. A low part needs a different mouth, which is why F23 (two models live) is what unlocks SATB rather than transposition.
- **Rule:** never A/B two mouths at a register where neither of them works. Loudness-align *and* register-align, or the comparison measures the register.

### F25. bank_live's glitch source is the mouth-monitor window, and both of its instruments are blind to it (2026-08-15)

- **Symptom:** Harry live, `--per-part 4` — "有一個雜音，一直都有、跟我唱什麼無關、開口才有"。
- **Two instruments said "clean" and both were structurally incapable of showing it.** `xrun` counted 8 in 226 s; the `--dump` out track had no clicks, no clipping (peak −7.9 dBFS), no broadband noise (spectral flatness ≤ 0.011 where white noise = 1.0). Neither is evidence: the engine **drops** a computed block when backlog exceeds `--maxlag` (`bank_live.py:1849`, 35 ms hole in a sustained note), the dump is written *inside* `process_hop` at `:1819` — **before** the drop — and the code's own v25 comment states "xrun 完全照不到（callback 都填滿了）". The only honest instrument is the `dropped` counter, which prints in the status line **only when >0**.
- **It is not compute.** Instrumented per-hop timing over 3143 hops of the offline bench, 35 ms budget: `--per-part 4` median **2.7 ms** (7.7 % of budget), p99.9 17.7 ms, max 26.5 ms — **zero** hops over budget, v38 and v39b alike. The "16 voices are too heavy" hypothesis is measured-false.
- **It is the `imshow` monitor window.** 90 s idle (no singing, gate closed, `--per-part 4`), single variable: `--view 1` → **dropped 8** (reproduced exactly twice); `--view 0` → **dropped 0, backlog 0**. Same family as the earlier "imshow 吃 RT ＝ +12 xrun/分鐘" note, but this time it is the *drop* path, not xrun.
- **Operational consequence (viva):** the show config must carry `--view 0`. The window is still needed for the pre-show gate self-check (唇圈跟嘴動) — so that check runs with `--view 1` and the engine is **restarted** into `--view 0` before the performance. The two cannot be had at once.
- **Method lesson:** an instrument that cannot see the defect returns "clean", and "clean" reads identical to "no defect". Before trusting a null result, ask where in the data path the instrument taps — here the dump tap was 30 lines upstream of the bug.

### F26. bank_live 有可量到的回授路徑（279 ms）——但耦合不是常數，判別力仍不足 (2026-08-16)

問的是門控的原題：**麥克風收到的能量，能不能用引擎剛播出去的東西解釋？** 能＝喇叭繞回來、不能＝有外來聲源＝他在發聲。素材 `260811_bt/ear/dump_0815/`（226 s，`noise_mic.wav` / `noise_out.wav` 同一 callback 兩軌＋`noise_st.npz`）。腳本 `harmony/scratchpad/bank_fb_envcorr.py`。

- **這不是 G34 重跑，而且理由可驗證。** G34 死於結構性論據：串流架構的迴路是 `angel(t)≈G·mic(t)`（即時轉換載體），整個迴路的包絡自相似 → 能量比恆定 → 延遲掃 0–697 ms **單調上升、無峰**。bank_live 沒有那條因果邊：`y` 是音庫取樣播放（`bank_live.py:1864` `p.render(hopN, r)`）＋pad＋殘響，只透過離散的 `note` 決策與 mic 耦合。**前提不成立 → 值得重測，且結果不同。**
- **回授路徑確實存在（G34 拿不到的東西）：** 去 3 s 趨勢後掃延遲，**峰在 +279 ms、r=+0.249**，對照 d=0 的 +0.125 與掃描邊緣 +0.070；負延遲側（mic 領先 out，物理上不可能是回授）是乾淨的**負**相關 −0.24。因果不對稱成立，且 279 ms 對得上系統 p50 ~217 ms＋輸出緩衝＋空氣路徑。
- **去趨勢是必要步驟，不是修飾。** 未去趨勢時曲線從 d=0 的 0.302 一路爬到 1393 ms 的 0.372 還沒轉頭——兩軌共有的數十秒尺度慢趨勢把每個延遲的 r 都墊高，正是 G34 說的那種平凡效果。峰是被慢趨勢蓋住的。
- **耦合 k 本身沒問題——這條假設被既有標定素材推翻了。** 一度看似死穴：三種**無標籤**估法發散 10 dB（10 % 分位迴歸 **−20.0 dB** / f0 靜默候選 **−10.0 dB** / 加 150 ms 保護帶＋50 ms 短窗的乾淨幀 17.0 s **−14.8 dB**），乾淨幀自己內部四分位還跨 **13 dB**。但 08-10 的標定素材（`scratchpad/CALIB_*`，55 s，前後掃頻夾一段 30 s 純回授；`fb_calib_ana.py`）顯示**耦合其實平坦**：80 Hz–12 kHz 的 k(f) 只跨 **3.1 dB**（−23.2 … −20.1），寬頻 **−17.5 dB**——正好落在上面兩個估計之間 ⇒ 三者其實一致。**所以「頻率相關耦合」的解釋是錯的，分頻帶標定救不了這條。**（只移植 k 不移植延遲：CALIB 的 547.6 ms 是「播放檔時間軸→mic」，與 dump 的「產生時間軸→mic」定義不同。）
- **也不是量測 SNR 不足。** 乾淨幀的預測回授相對 mic 底噪（−56.2 dBFS）SNR 中位 **+11.7 dB**，只有 **8.0 %** 的幀低於 +6 dB。
- **真正的死穴是殘差的語意：殘差 ≠ 他在發聲。** 用標定 k 算，**他不出聲時 mic 仍比預測回授高 +2.7 dB 中位**（若 mic 只有回授，該值應該貼近 0）。也就是說，麥克風裡有一份**與回授同量級的非回授能量**——呼吸、氣音、字間殘響、身體與衣服的動作聲、環境。包絡法能誠實回答的是「有沒有非回授的聲音」，而**門控要問的是「他有沒有在發聲」**。36.6 % 的誤觸不是量測不準，是**定義上的**：那些幀裡確實有非回授能量。
- **分離度比 G34 好得多，仍不足以當門控。** 中位分離 **+8.3 dB ＝ 6.76×**（G34 最佳 1.68×，判準 >2×）——所以「換架構就破前提」這個判斷是對的。但分布重疊：最佳門檻 +4 dB 時**抓到在唱 89.4 %、誤把回授當成他在發聲 36.6 %**，重疊率 47.2 %。對照已判死的鏡頭門控閉嘴誤觸 20–35 %——**這條並沒有比較好**。
- **而且 6.76× 是樂觀偏差。** 兩堆標籤用 `npz` 的 `f0`（門控前偵測）切，刻意避開 G35 血訓的「用音量切標籤」；但音高偵測對音量仍有依賴（弱訊號測不到音高），所以兩堆天生在 mic 能量上已分開一些。**真實分離度只會更差，不會更好。**
- **結論：** 回授路徑可量、可估延遲，但**單一寬頻能量比撐不起「他有沒有在發聲」的二元判斷**。門控不可靠這個結論不變，viva 照原計畫走備案。
- **不必再錄標定素材了。** G34/G35 更正開的處方（另錄 30–60 s 乾淨素材）**已經被 08-10 的 `CALIB_*` 滿足**，而且它證明的是問題不在標定：k 平坦、可標、三法一致。要改善的話得換的是**維度**（能量以外、能分辨「人聲發聲」與「呼吸／動作聲」的東西），不是把同一個能量比標得更準。
- **方法教訓（兩條）：** ① 換架構會使墓園裡的死因失效，但**只有結構性死因值得逐條複查前提**——G34 的「迴路自相似」依賴 `out∝mic`，那條因果邊在 bank_live 沒了，所以該重測、也確實測出不同結果；前提沒變的死因不要碰。② **先問殘差的語意再問殘差的精度**。這裡花在「k 準不準」上的功夫全是白工：把 k 標到完美，殘差裡仍然裝著呼吸與動作聲，而那不是門控要的答案。同 F25 的「照不到缺陷的儀器會回報乾淨」，這次是**照得很準、但照的不是那個東西**。

### F12. Live-control work must fit inside one audio callback (2026-07-22)

- **Any UI control press that takes longer than the callback period starves the audio thread and the dropout is audible as a click.** solo_min's `set_control` re-applied every voice's config on *any* key — measured **12.2 ms mean (7.8–15.3 ms)** against a **10 ms** callback → a click on every press, including toggles that cannot change voice config at all. Fixed by (a) a cheap-key fast path (`dry/no_dry/gain/naive/on1..4` — verified none of them is read by `configure_voice`) and (b) caching each voice's applied speaker so `set_target_speaker` only fires on a real change: **12.2 ms → 0.0–0.1 ms** (`e00340b6`).
- **`set_target_speaker` is the expensive half** (~1.5 ms per voice); the rest of `configure_voice` is attribute assignment. It also causes an inherent discontinuity — it swaps the embedding mid-stream.
- **Rule:** anything on the control path gets timed against the callback period before it ships. `print(f"took {ms}")` around the bridge call is enough to find it.
- **Diagnostic note:** the symptom logged as `input overflow`, NOT `output underflow` — the starved callback loses input samples. Don't read `input overflow` as automatically meaning F7's clock drift.

### F13. Granular pitch-shifter baseline — measured behaviour (2026-07-22)

For the RQ1 A/B (`--naive`, `server/voice_changer/naive_shift.py`, Tone.PitchShift parameters: two taps half a 100 ms window apart, equal-power crossfade):
- **Onset latency 4–50 ms, mean 10–33 ms depending on shift amount** (−12: 25.8 · −5: 9.6 · +4: 33.0 · +7: 25.3 · +12: 16.7 ms). NOT the window length — the two taps are half a window apart so one is always reading recent audio. Same order as the engine path (~46 ms low-latency mode), and it drifts with grain phase (part of the granular character).
- Pitch accuracy verified within FFT bin resolution at ±12/±7/±5/0 semitones.
- **A cold shifter outputs RMS 0.0000 on its first hop** (warm: 0.2986) — it must be fed every hop even when unused, or switching into it clicks.
- **Crossfade on switch:** equal-power over 25 ms takes the worst-case single-sample jump from 2.0 to **0.0037**.
- **Both branches must emit the same block size.** The converter emits exactly OUT_HOP (240 @24k) per input hop; a soxr stream does not (237 then 240) — the mismatch truncates the Beatrice block during a crossfade and drifts the output cushion.

### F15. torchcrepe tiny ≠ automatic ear upgrade — parity with the iterated YIN on real stems (2026-07-23)

07-22 §H had "耳升級 torchcrepe tiny" as a prerequisite for designs ②/③, extrapolated from YIN's 45–68% *live* heard rate. Measured offline on real-voice stems (`harmony/ear.py` A/B harness, harvest as reference):
- **Overall heard% — parity:** take.wav YIN 85.9 vs CREPE 87.3; respond_take.wav 71.9 vs 72.2. The live 45–68% figure was a *live-conditions* number; the seven smoothing iterations (median/octave-fix/fry-reject, `pitch.py`) already closed the estimator gap.
- **Sustain metric (what design ② needs) — exact parity:** in-note continuity 95.7 vs 96.0% (take), 50.0 vs 49.7% (respond_take); 0.4 s-stable-window trigger success 13/13 both, 2/4 both — **the same two notes fail for both ears**, so the bottleneck is the shared RMS gate/smoothing layer, not the f0 estimator.
- Cost: CREPE tiny ~3–10 ms per 23.2 ms hop on CPU (fits, but not free); `--device mps` **crashes on torch 2.0.1** (`torch.load` can't restore mps-tagged storage in torchcrepe's model load).
- **Rule:** the validated YIN `PitchTracker` stays the default ear; `harmony/ear.py` (CrepeTracker, same push/latest contract) is kept as an option (`sustain.py --ear crepe`) and as the A/B harness. Don't "upgrade" the ear without this harness showing a gap on the metric the consumer actually needs.

### F16. Shift-style f0 is the main culprit; Beatrice timbre is largely rehabilitated under clean pitch (2026-07-24 blind 2×2)

> **直驅載體重測成立 2026-07-28（worklog 07-28 §E/§F）**：shift 模式移植到直驅嘴（pyworld 抽真 f0＋逐 tick 移調）後 Harry A/B：「用 shift 不對，shift 會讓音不穩，比 target 表現差，且 autotune 還有」→ target 續任，本條在直驅鏈上維持有效。連帶：f0 線三嫌疑（PORTA/vibrato 同步/量化源頭）全數排除或處置後 autotune 仍在＝嫌疑收斂到 DDSP 轉換器本身（診斷梯 `harmony/out/diag_autotune/`）。

Blind 2×2 on the same take, same v2/cpdl brain line for 3 of 4 cells, one mix recipe, Beatrice stems latency-trimmed (worklog 07-24 §B design, §C build, §D verdict; commit `a28cab3d`; stimuli `harmony/out/blind_2x2_0724/`). Harry's ranking: **world_target ≥ beatrice_target > world_shift > beatrice_shift**.
- **f0 philosophy is the dominant axis:** target-f0 beats shift-f0 under BOTH timbres. The "machine-voice" verdict that made the solo_min brain line dormant (07-23 §U) was primarily the shift-style pitch path, not the Beatrice sound per se.
- **Timbre is secondary:** WORLD ≥ Beatrice within both columns, but the target-f0 pair was near-equal to Harry's ear.
- **Beatrice as a mouth is viable IF fed clean-pitch audio:** WORLD target-f0 resynthesis → Beatrice at shift 0 preserves pitch to a per-frame median of 4 cents (measured at 50 ms engine lag, `harmony/beatrice_mouth.py`). "Control what it hears" substitutes for the absolute-f0 control the binary lacks.
- **Rule:** any future angel mouth must place pitch by TARGET (re-synthesize/pin to the note), never by transposing his live contour; his wobble transposed = wrong-sounding even in his own WORLD grain. The 28c shift-architecture floor (07-23 §R) is now also an aesthetic ceiling — don't spend more effort polishing shift-style paths for the angel.

### F22. Response latency is front-end cost, not model cost — and the safe pipeline line is "identical by construction" (2026-08-03)

Cost breakdown of one `make_response` phrase render (5.16 s real phrase, one voice, `harmony/phrase_render.py`): harvest 0.590 s (RTF 0.114) · voicing per-frame loop 0.406 s (0.079) · hubert encode 0.543 s (0.105) · `_local_ref` ×2 0.031 s · **model forward 0.059 s = 2.5 %** · uv_passthrough 0.030 s. **The vocoder is not the bill.** Any latency work on this path must start from that shape.
- **Free 2× first: everything that only reads `x` is voice-independent.** `input_agc` and `voicing_conf` each called `pyworld.harvest`, and both mouths ran both — the same audio was harvested **four times per phrase**. `harvest_f0()` single entry + `f0m=` cache params + shared dict `{f0m, agc, vm, units, vol}` (`8d1bea36`): 16.7 s of material 7.92 s → 3.88 s (RTF 0.47 → **0.23**). Model inputs bit-identical (`max|diff| 0.0e+00` on the agc gain curve and the voicing mask), output diff inside the MPS self-noise band.
- **Pipeline rule: only pre-compute what is bit-identical when computed in pieces.** Safe = the brain (`EarV3` is already a per-tick stream) and the voicing per-frame features (each frame reads only ±1024 samples → `dm.voicing_frames`). Not safe = harvest (global path search), hubert encode (full attention), `_local_ref` (±3 s window), model forward (seams + phase discontinuity). Cutting encode/forward too would buy another ~0.6 s and would produce **a different mouth** — the one Harry ear-passed on 08-02. Waiting 0.6 s longer is the cheaper price.
- **Result** (`3f24b098`, synthetic probe with human-scale 5.5–9.8 s phrases): "he stops singing → response starts" p50 **4.47 s → 2.16 s**, p95 5.42 → 2.48, io overflow/underflow 0/0, 100 % of each phrase pre-computed, and the wait stops scaling with phrase length (slope ~0.13 s/s = the harvest+encode remainder). Verified: brain note lines identical 14/14 phrases; voicing mask and agc `max|diff| 0.0e+00`; **response waveform pipeline-on vs -off `max|diff| 0.0000` (14/14) = Regime A byte-identical.**
- **Instrument rule (new):** a latency saving measured by feeding a file as fast as possible is **fake** — with no elapsed "while he sings" time the background work simply lands at the phrase end anyway (first attempt measured "0.03–0.19 s saved" and nearly killed the feature). Feed at 1× and stop the clock during the muted window. Same family as the 08-02 under-reported latency and the cold-start brain RTF. **To measure something whose nature is time, let time actually pass.**

### F19. The "autotune feel" lives in the synthesis CARRIER — contour scripting and gesture realism both exonerated (2026-07-24 evening)

Three-step ablation chain, all on the same take/brain line:
1. **Parametric humanize (3 variants: onset-ramped vibrato/drift, his-deviation transplant ×0.45, hybrid)** — Harry: "都差不多". Contour PARAMETERS are not the lever.
2. **R1 corpus gesture transplant** (`harmony/gesture_lab.py`: 769 real gesture units mined from the 0619 corpus, interval/duration-matched, onset-anchored warp — zero scripted curves) — autotune feel persisted in both mouths (`blind_gesture_0724`, Harry: "其實都還有；單論音質 2,4>1,3" = DDSP column > WORLD column again). Contour REALISM is not the lever either.
3. **Identity resynthesis blind** (`blind_carrier_0724`, solo, no mix): his own take, own f0, own content, only the carrier changes. Harry: **original > WORLD-identity > DDSP-identity**, WORLD's signature artifact named precisely ("同音高的蜂鳴器在背景" = WORLD's harmonic-excitation buzz). Even a bit-faithful pitch performance sounds "autotuned" through these carriers.
- **Rules:** (a) stop tuning f0 recipes for naturalness — invest in the carrier (more/better training data for the DDSP decoder, reflow-class models, enhancer chain; for WORLD: aperiodicity experiments against the buzz). (b) Carrier rankings INVERT between solo and in-mix listening (DDSP smoothness wins in the angel mix, loses solo on consonants/breaths) — always evaluate mouths in the mix context they will live in.
- **Live-only artifact isolated:** DDSP live's "pitch sliding around" = splice-phase warble (combsub restarts its phase cumsum per window; fixed-position crossfade every 150 ms warbles — offline renders have no chunking, hence clean). Fix = SOLA alignment before the splice (upstream's realtime approach), landed in `DDSPMouth._sola`.

### F20. Carrier-upgrade blind: neither more-old-data nor a bigger architecture beats the 14.9-min combsub — the bottleneck is FRESH data (2026-07-25)

3-cell blind `harmony/out/blind_carrier_upgrade_0725/` (same take/brain line/recipe as F18), Harry: **ddsp30k(控制) > reflow4k > ddspmix30k，差距非常小**.
- **Data axis (G30):** mixing old-session tenor (27.6 min total) LOWERED val loss (1.04–1.07 vs ~1.09) yet came LAST by ear → F9's "old+new mix is worst" holds under DDSP too; **val loss is a weak proxy** (second confirmation).
- **Architecture axis (G31):** 6.x RectifiedFlow at its val valley (4k) still ≤ combsub by ear, overfits by 4k on 14.9 min, and CPU RTF ≈ 2.05 (23× combsub) kills live.
- **Rule:** the carrier ceiling on this dataset is a DATA ceiling in the strict sense — the only moves that can raise it are (a) recording a fresh same-condition session (then retrain combsub; only then re-audition bigger architectures on the enlarged set), or (b) sidestepping synthesis for sustains entirely (母音取樣嘴 option ⓪, worklog 07-25 §A — un-buried, not yet attempted).
- All three cells sit at the same perceived level ("差距非常小") → current 14.9-min combsub stays the reigning mouth; no retraining without new data.

### F21. Rule-based expressive layer closes the autotune case: contour IS a lever once artifact (a) is silenced — F19 rule (a) amended (2026-07-31)

Harry passed `stim_3e` by ear ("autotune 感差不多沒了") — the rule-f0 cell with `direct_mouth.py --f0-mode express` (r-round recipe otherwise unchanged). Instrument: sustain cents_sd 12.95 → **26.81** (his real f0 = 32.5), breath_voiced 20.5 → 13.9% as a side effect.
- **Recipe (all measured from his own take, `EXPR` in direct_mouth.py):** per-note bias sd 22c clip ±35 (**the biggest contributor — absent from every earlier humanize attempt**), slow drift sd 13c (≤2.5 Hz), fast jitter 6c (~10 Hz), vibrato 5.9 Hz ± 0.9 walk / depth 20c per-note-jittered / no onset delay, onset bends |22c| median 59% from below with 15–60 ms settle **applied only where the skeleton lands instantly** (phrase heads / post-rest / leaps — steps already have porta, don't double-bend). Deterministic seeded RNG; `--expr-gain` scales the whole layer.
- **Gain is NOT more-is-better:** gain 1.2 bought +1.5 cents_sd but pushed breath_voiced to 22% — default stays 1.0.
- **F19 reconciliation:** F19 step-1's "contour parameters are not the lever" was measured while artifact (a) (voiced-forced breaths, metallic onsets) masked everything; with (a) closed (worklog 07-30 §H–I) the three-rung ladder showed cell ② (real f0) passing blind — contour texture became audible and fixable. F19's carrier rules stay valid for the carrier axis; its rule (a) "stop tuning f0 recipes" no longer applies as stated.
- **Usage rule (AMENDED 2026-08-02 — read the amendment before applying):** ~~offline/demo direct-mouth renders default to `--f0-mode express` from now on~~ (checklist item, CLI default left at `target` so historical cells stay reproducible).

**⚠ F21 AMENDMENT (2026-08-02): express is validated for ONE voice only. It breaks harmony between two.**
Every cell that passed above was a **solo** loopback (`stim_3e` = his voice, one mouth). The layer's biggest contributor — per-note bias sd 22c — is a *constant pitch offset per note*, and each voice draws its own. Measured on two angels singing together (0802 take, 2007 two-voice hops): **interval error SD 40.7c, |err| >20c for 61.6% of the time, >50c for 20.7%, max 139.9c.** A 22c bias makes a soloist sound human; two soloists with independent biases are simply out of tune with each other.
- **Harry's ear, 2026-08-02:** express two-angel renders = 「怪」 / 「還是很怪」; the same brain + same take + same key through the **07-28 blind_v3 recipe** (`render_v3.py --mode joint`, i.e. plain `target_f0` skeleton + fixed vibrato desynced per voice: 5.3/4.6 Hz, phase 0/0.5, onset 250 ms, `vib_semi` 0.12) = 「這個對了」. Re-rendered through respond2 with `--f0-mode target` = 「兩個都還行」.
- **Why target is in tune by construction:** both voices sit exactly on the skeleton, so their pitch *centres* are identical; the only per-voice difference is vibrato, which is zero-mean (and desyncing it is what killed the 07-28 「機械齊振」 autotune feel).
- **Rule now:** **two-or-more voices → `--f0-mode target` with per-voice vibrato desync.** Solo → express stays valid (F21 as measured). `respond2.py` default flipped to `target` (`aaf53eb1`+).
- **Blast radius — re-審 required:** every two-angel ear verdict from 2026-07-31 onward ran express. That includes the 08-01 rehearsal-prerender line judged dead (「都很糟、不順」) and the 08-01 「修完調性仍有殘餘怪感」 residue attributed to material. Those attributions are unsafe until re-heard on the target recipe.
- **Not a dead end for express:** what it fixed (「f0 比真人直一倍」) is real. The route back is to re-add only the components that do **not** move a voice's pitch centre (jitter, onset bends, vibrato-rate walk) on top of the target skeleton, or to share the slow components across voices — mechanism already built as `expressive_cents(tune_seed, tune_lock)` (`37b82bfe`): tune_lock 1.0 cuts inter-voice pitch-centre SD 25.8c → **3.4c** while per-voice expressive SD stays 28.6c (= the 28.9c F21 baseline). Untested by ear against target; that is the next experiment, not a shipped answer.

### F17. Streaming target-f0 WORLD is live-viable — and the streaming pipeline BEATS batch by ear (2026-07-24)

`harmony/world_stream.py` (commit `dce76b43`), strict-causal chunk simulation on the real take, same brain line as the batch mouth. Blind 3-way (`out/blind_stream_0724`), Harry: **stream_exact ≥ stream_bounded > batch** — both streaming variants outrank the batch render; the G28 quality wall was the naive pipeline, not streaming per se.
- **G28 counters that worked:** per-frame dio+stonemask over a 1.2 s window (dio 7 ms/s vs harvest 103 ms/s; block-constant f0 was the killer), sp/ap on a 0.4 s tail window, frames analyzed once then frozen (natural evolution kept).
- **Zero-seam streaming is provable, not tuned:** WORLD synthesis is deterministic, so whole-phrase resynthesis re-emits the already-played prefix bit-exactly (runtime check max diff 0.0) — provided the last ~28 ms of any synthesis is held back (measured end-boundary effect; HOLD 40 ms ≥ it).
- **Live numbers on the take:** analysis+bounded-synth p95 74 ms per 150 ms hop (~50% of one core); structural lag = hop 150 + guard 50 + hold 40 = 240 ms + compute ≈ 310 ms, inside the angel's 150–250(+) ms lookahead budget with knob room (hop 100 → ~260 ms).
- **Bounded live variant validated by ear:** resynth last 1 s + 30 ms equal-power splice ranked ≈ exact → live does NOT need the C++ incremental synthesizer to start.
- DDSP-SVC checkpoint-10k f0 fidelity through the same target-f0 input: median 4 cents, RTF ~0.09 — the swap-in singing mouth stays compatible with this host design (mouth input = audio ring + per-tick absolute note).

### F18. The trained singing mouth WINS: DDSP-SVC on 15 min of own voice beats every other mouth (2026-07-24 final blind)

Same-line 4-mouth blind (`harmony/out/blind_mouth_final_0724/`), Harry: **ddsp_enh = ddsp > world_stream ≥ beatrice** (4=1≥2>3). One 58-minute MPS training run (CombSubSuperFast, hubertsoft encoder, parselmouth f0) on the F9 clean set (225 clips / 14.9 min) — details `SoloChoirCode/260724_ddsp_svc/` + worklog 07-24 §G.
- **What it is architecturally:** SVC-family = (content, f0) separated inputs — target-式 pitch placement by construction (F16 rule satisfied natively), content follows his vowels live, timbre = trained (his own here = voice-ownership kept).
- **Enhancer is a no-op to his ear** (4=1) → live path can run the raw DDSP output; nsf-hifigan stays optional.
- **Numbers:** pitch fidelity median 4 cents; RTF ~0.09 (58 s in ~5 s, MPS) — cheaper than one Beatrice pass (0.14).
- **Val plateau = data ceiling:** test_loss flat at ~1.09 from step 2k while train loss kept falling → stopped at 30k; more quality wants more data, not more steps. Girl set (26 min) = the "not-me" female angel candidate via the same pipeline.
- **Live route:** DDSP-SVC 5.0 ships a realtime chunked path (`flask_api.py`/gui, SOLA crossfade) — the remaining delta for the angel is injecting the brain's target-f0 curve instead of extracted+shifted f0. `world_live.py`'s mouth slot was built for exactly this swap.
- MPS gotchas patched in the local clone: `.double()` phase accumulators in ddsp/vocoder.py (4 sites) and nsf_hifigan/models.py (3 sites) → float32 (phase error ~0.006 rad over 58 s, inaudible); fairseq made lazy (hubertsoft path avoids it); requirements' `gin`/`wave` are PyPI impostors (real: `gin-config`); py3.10 venv (3.13 can't build fairseq/numpy pins).

### F8. Wiring & transducers
- All transducers Dayton BCE-1 (4 Ω, 1 W/2 W max), one PAM8403 stereo amp. **L(ch0) → bone conduction, R(ch1) → exciter.** Two bone conductors: series (8 Ω) only — parallel (2 Ω) burns the amp.
- Output streams must open **2 channels** (`channels=(1,2)`); a mono stream gets duplicated to both channels by the USB card, turning any per-channel gate into dead code (commit `87616338`).
- **Test A (bone-conduction + throat-mic self-stacking) PASSED by ear 2026-06-27** — no feedback ringing while singing. Test B failed (GRAVEYARD G19) → cup-shaped spatial source.
- **DAEX9-4SM exciter drive levels (measured 2026-07-07, cavity 1.5×5.8×11 cm ≈ 96 cm³, solo_min master gain `3c287180`):** **+10 dB = working point** — normal speech/singing loudness, no distortion, no heating. **+15 dB = thermal limit** — exciter heats up and starts distorting slightly (real coil power limit on the 0.5 W part, not digital clipping). Usable headroom is therefore ~5 dB; louder must come from CAVITY efficiency (resonance/coupling), not gain — sustained +15 shortens exciter life. Cavity v2 target: +3–5 dB perceived from acoustics so the working point stays ≤ +10.

### F21. Wire-light: WS2812 logic threshold and LED self-coupling (2026-07-26/27, unit1 perfboard)
- **3.3 V data into a 5 V strip is BELOW spec, always was.** WS2812 needs `0.7 × VDD` = **3.5 V** at 5.0 V rail; the Pico only sources 3.3 V. Measured breakdown curve on 2×122 px with JST pigtails: **4.825 V both strips clean / 4.9 V one strip corrupts / 5.0 V both corrupt**. The 07-23 breadboard PASS was luck (TENMA's actual output sat just under the cliff); swapping to MT3608 (5.044 V) + adding JST tails pushed it over. **Fix = 1N4007 in the strip's 5 V feed only** (≈4.3 V → threshold 3.01 V, margin +0.35 V); the rail itself stays 5.0 V for the PAM. Symptom to recognise: **randomly scattered wrong-colour pixels that keep changing — a static solid-colour frame looks like it is animating.** Diagnose by sending one pure colour, never a moving effect.
- **LED self-coupling closes a lock-up loop.** Same board, same instant: strips black → ADC span **~1,000**; strips full-on → **~5,300** (repeatable, returns on black). The strips' PWM current pulses couple back through the shared ground into the audio tap, so any fixed threshold self-locks: lit → noise clears threshold → "sound detected" → stays lit, never releases after the music stops. Didn't appear on the breadboard because that was 1 strip / 1 tap / strip fed straight off a low-impedance rail — **the 1N4007 added for the threshold fix is what turned the current pulses into voltage ripple** (fixing A caused B).
- Three fixes tried, third adopted: **blanking-sample** (kill strips ~1 ms, then read) drops noise 3,203→729 but `write` of 122 px costs 3.7 ms ×2 strips → real blank ~5 ms at 20 Hz = **visible flicker, rejected by ear/eye**; **hysteresis** (2,000 wake / 6,000 hold) breaks the lock but **oscillates at mid volume** because the signal sits between the two thresholds; **coupling compensation** ✅ — `span -= COUPLING × env`, since the coupling is proportional to brightness and `env` *is* brightness. Strips dark → no subtraction (full sensitivity), strips lit → cancels its own contribution. Single low threshold 2,000, no oscillation, independent of Mac volume. Root fix remains hardware (bulk cap must sit on the strip side of the 1N4007; 470 µF if needed — re-measure `COUPLING` after).
- **Mac system volume sets signal amplitude, not noise**: same test tone reads span 1,500–3,200 at volume 44 vs 9,000–10,000 at 65 (working point = 65). The dark-strip noise floor ~1,000 does *not* track volume — it is interference, not signal. Practical consequence: an absolute `NOISE_FLOOR` is only meaningful once the coupling term is removed.

### F11. Physical source separation de-fuses the choir (2026-07-14)
- The G9 fusion ("choir sounds like one person" — correlation, not timbre) is broken by **physically separate sources**: same studio stems, one speaker vs two spaced 1–2 m (sop+alto / tenor+bass) → the split version has a clear "several singers in the room" quality the single source lacks. Pilot by ear via `eval/spatial_pilot.py` (pure routing, zero processing — G11 buried in-path *widening*, which stays buried).
- **Live-verified same day**: solo_min `--split-lr` (flag-guarded; V1+V2→ch0, V3+V4→ch1, dry You both sides; default path untouched, commit `7ed16478`) on BCE(L) + DAEX(R): routing correct per-voice, clean (overflow only at device switches, underrun 0), multi-person feel confirmed by ear. RTF unchanged — same 4 inference passes, this is output assignment only.
- Rule pair: **spatial separation = the missing dimension of F3's decorrelation** (pan's physical extreme); **never** re-add signal-domain widening to the live path (G11).
- Exhibition scale-up (W5): 4 discrete sources need a 4-out USB interface (ONE clock — F7; no aggregates) + `channels=(1,4)` per-voice routing; source hardware DAEX25+ply panel vs powered speakers = by-ear A/B; DAEX9 too weak for a room (0.5 W, F8 thermal). Prior-art anchor: Janet Cardiff, *The Forty Part Motet* (2001) — one speaker per singer; Solo Choir's differentiator = the singers are live, from one body.

### F14. Print tolerance & support — numbers validated on the MINI (2026-07-22)
All measured on real prints of `260721_prints` and re-verified in CAD; details in `prints/260722_prints/README.md`, deaths in **G25**.
- **配合餘隙（PLA, 0.2 mm 層高, 0.4 nozzle）：**
  - **滑配（滑蓋插軌槽）**：單邊 **0.13 mm ＝ 塞不進去**（軌槽 2.240 / 蓋板 2.000，實印失敗）；單邊 **0.37 mm ＝ 順**（蓋板減到 1.500）。軌槽印窄＋板印厚，兩邊各吃 0.1–0.2。
  - **⌀3 黃銅棒穿孔**：**直孔** 0.05 單邊可行（`pivot_mount` 實證）；**彎通道** 0.05 單邊**不可行**（G25），本批放到 0.15 試。**印後一律先 ream。**
  - **熱縮不是主因**：PLA 線性收縮 0.3–0.5% → ⌀3.1 只縮 **0.012 mm**。小孔偏小的真正來源是圓的多邊形化、內凹轉角擠出堆積、水平孔頂 bridging 垂邊（0.1–0.3 mm 且不對稱）。
  - **C 形喉口是繞過孔徑問題的正解**：從側面卡入就不吃垂邊也不吃軸向對位（`band_clamp` 髮箍座、`vu1_panel` 夾、`door_clip` 都是這招）。
- **PrusaSlicer 支撐的兩個陷阱：**
  - **`support_material = 1` 不代表會有支撐。** 若 `support_material_buildplate_only = 1` 而懸空面在**模型上方**，產生量是 **0**、且不會有任何警告。實例：P4 蓋的托架（z≈13）整盤 0 段支撐，鉤唇印在空氣裡。**驗法：`grep -c 'TYPE:Support material'`，再逐層光柵化比對「這層有、下層下方沒有」的格子。**
  - **organic 支撐會把樹幹順著通到床面的垂直孔往上長**（`door_col` ⌀3.3 軌孔內實測 467/469 條走線、貫穿 34 mm，拆不出來）。**grid 不會**（孔內 0），snug 仍殘留。**凡是有貫通孔的件，切片後掃一次孔內半徑範圍。**
- **USB／機器端**：Prusa MINI 讀到檔尾就跑自己的收尾流程並顯示 **Print finished**——所以「印一半但機器說完成」＝**檔案被截斷**，不是切片問題。FAT32 碟上的 macOS `._` 影子檔（4 KB）會出現在檔案清單裡，選到它按下去等於什麼都不印。**用 `cp -X` 複製並對 byte 數。**

## Training data hygiene

### F9. Voice-model training rules (post-viva)
- Clean set ready: `260618/test/man/0619_clean_clips/` = 226 clips / ~14.9 min (via `tools/preprocess_voice.py`).
- Train on 0619 material ONLY. Do **not** mix in tenor.wav (old session — old+new mix was verified worst). NEVER train on `held-out.wav` (it is the eval clip for `eval/ablation_eval.py`).

## Research framing facts

### F10. Supervision & viva positioning (2026-06-23 meeting)
- Supervisor: **Daniel Berio**. Matthew Yee-King = external advisor (not supervisor).
- Before viva (2026-08-26): do NOT switch the real-time engine, do NOT self-train chasing quality. DO add a **baseline comparison layer** (plain pitch-shift+EQ vocoder vs Beatrice vs optional third method) + a small listening test (naturalness / artificiality / intonation). Framing: research comparison first, performance support second; target a conference paper.
- Project bar: PhD/Media-Lab portfolio piece — Track A (software ear-training, 4 phases, committed) + Track B (hardware embodied) must serve ONE research question: "body as interface for harmony: learn inside the body / perform outside it". Two half-projects is the failure mode.
