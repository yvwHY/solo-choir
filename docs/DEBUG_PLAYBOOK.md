# DEBUG PLAYBOOK — Live-Audio Symptom → Known Root Cause

**Audience: AI assistants (Claude, Codex, any model).**
Check this table BEFORE touching buffers, threads, or the engine. Most live-audio symptoms in this project already have a diagnosed root cause; the historical failure mode is theorising about buffers when the cause was elsewhere (see GRAVEYARD G5, G7).

| Symptom | Root cause (diagnosed) | Fix / where |
|---|---|---|
| Voice toggle sounds choppy / 斷斷續續 | Cold converter emits ~35 ms of zeros on resume — NOT buffers/underruns | Keep every live voice warm, gate at mix (FINDINGS F4) |
| Scrape / instability when harmony pitch changes | Every `set_pitch_shift_semitone` *change* scrapes engine state | Glide 3–6 ¢/10 ms + 80 ms debounce (FINDINGS F2) |
| ~10 Hz tremolo on converted output | Chunk + context re-feed interrupting the stateful stream | Don't re-feed; one continuous stream (GRAVEYARD G2) |
| Comb-filter / blur when combining voices | Mixing two engine streams — phase free-runs per stream | Never crossfade engine streams (GRAVEYARD G1) |
| `input overflow` messages | USB mic + output = two independent clocks drifting in one duplex stream | Aggregate Device w/ drift correction, or one USB interface; bigger buffers only delay it (FINDINGS F7) |
| PortAudio `-10851 Invalid Property Value` | Two streams on the same device at different sample rates | Everything at 48 kHz (FINDINGS F7) |
| Speaker(s) cut out intermittently, session log CLEAN (0 underrun/overflow) | NOT software — self-powered voice-unit LiPo low: loud passages spike current → MT3608 sags → PAM8403 brownout (recovers when quiet). MT3608 idle drain is mA-class → a battery left on the rail dies in days | Charge/swap the LiPo (TP4056 red LED = confirmed low); add a master switch to the rail; never charge while playing (TP4056 has no load-sharing) — diagnosed 2026-07-19 |
| bank_live: recurring 雜音/glitch, **`xrun` low and the `--dump` sounds clean** | Worker backlog > `--maxlag` ⇒ engine *drops* a computed block (`bank_live.py:1849`), punching a 35 ms hole in sustained notes. **Both instruments are structurally blind to it**: the dump is written inside `process_hop` (`:1819`) *before* the drop, and the code's own comment says "xrun 完全照不到". Read `dropped` in the status line instead — it only prints when >0 | The RT thief is the **mouth-monitor window**: `--view 1` → dropped 8 per 90 s idle (reproduced twice, no singing, gate closed); `--view 0` → **0**. Show config must run `--view 0`; use `--view 1` only for the pre-show gate self-check, then restart — measured 2026-08-15 (FINDINGS F25) |
| Feedback / "reverb on You" through speakers | Acoustic loop speaker→mic (worse with dry-You passthrough) | Headphones or Record→Play; monitoring artefact, not a bug |
| All voices turn male / timbre collapses after device switch or restart | Control state lost the per-voice `model` field → rebuild fell back to tenor | `app/bridge.py set_control` fix; check control-state completeness across restarts (GRAVEYARD G7) |
| Whole choir crashes on model/speaker change | Speaker index out of range for the loaded model (e.g. JVS speaker on tenor model) | Speaker-index guard; check `num_speakers` per model (FINDINGS F1) |
| Fixed-rhythm tick in output | Resampler artefact | QQ resampler + output cushion (commit `b3ccdfe0`) |
| Exciter/bone-conduction channel toggle does nothing | Output stream opened mono → USB card duplicates to both channels, per-channel gate = dead code | Open `channels=(1,2)` (FINDINGS F8, commit `87616338`) |
| Choir sounds like one person, not many | Time-aligned copies fuse — correlation problem, not timbre | Decorrelation recipe (FINDINGS F3); more speakers won't help (GRAVEYARD G9) |
| Detune of a few cents has no effect | rc0 binding pitch quantisation deadzone ~5–10 ¢ | Do micro-pitch offline in signal domain (GRAVEYARD G8) |
| Transcription "wrong" on legato same-pitch syllables or x.5-semitone notes | True input ambiguity — heuristic ceiling reached | Editor drag-fix is the designed answer; do not add heuristics (GRAVEYARD G15) |
| MIDI device list empty / missing a keyboard plugged in AFTER app launch (↻ doesn't help) | CoreMIDI snapshots the device list at the process's first MIDI call; rtmidi on a background thread never receives hot-plug notifications | Plug the keyboard BEFORE launching the app, then restart the app if you forgot (diagnosed 2026-07-07) |
| AUDIO device plugged after launch missing from the selector (e.g. `throat`); or stream open crashes `PaErrorCode -9986` after replugging | Same snapshot class: PortAudio/sounddevice caches the device list at init; a fresh process sees the device fine | Plug ALL audio hardware BEFORE launching; restart the app if you forgot. Do NOT hot-refresh via sd._terminate() — it kills the open stream (2026-07-07; -9986 re-hit 2026-07-14) |
| Crackle/爆裂音 riding ON the voice (more voice → more crackle), present in the interface's RAW input recording, gain-independent, swapping the output path doesn't help | Audio interface wedged in a bad internal state (AI-Micro after hot-plug churn) — NOT clocks, NOT the throat mic/cable, NOT the amp/output chain; verified by recording the raw input digitally (710 impulsive events → 11 after reset) | Fully unplug→replug the interface (power-cycle reset), then restart the app (snapshot rule above). Diagnose by recording the interface input directly — bypasses the whole app. **Show-day SOP: power up in fixed order (all hardware plugged → interface powered → app); on ANY unexplained audio anomaly, power-cycle the interface + restart the app FIRST (~30 s, collapses the whole wedged-state class) before theorising** (2026-07-14) |
| Dry You sounds thin / "high-pass filtered"; converted voices gritty/unstable | Throat mic not coupled to the neck (a contact mic in air only picks up thin airborne bleed) and/or interface input gain far too low (AI-Micro ships low — peaks were 0.03 FS, −29 dBFS) | Strap the mic snug; set input gain in RØDE Central so performance-volume peaks land ~0.2–0.5 FS. When swapping interfaces, verify input LEVEL height, not just signal presence (2026-07-14) |
| Ctrl-C 後行程不退、輸出凍在最後一行「聽……」、session dump 沒落檔 | SIGINT 已收到、主迴圈已退，卡死在 `with sd.Stream` 的 `__exit__`：PortAudio `FinishStoppingStream` 掛在 CoreAudio 裡（`sample <pid>` 可證），60s+ 不自解；再補 Ctrl-C 沒用（訊號進不了 C，只會積成 pending KeyboardInterrupt 咬下一段 Python 碼） | respond2 已改為 dump＋統計在關流**前**落檔（08-04）＝掛死不再吃資料，直接 `kill -9`。若在別的腳本遇到且資料還在記憶體：lldb attach → `PyGILState_Ensure` → `PyRun_SimpleString` 注入 `sys._current_frames()` 從 frame `f_locals` 拿陣列寫檔（純 numpy 陣列不進 `gc.get_objects()`，別走 gc）；注入若被 pending KeyboardInterrupt 炸掉（-1），它已被吃掉，重跑一次即成（2026-08-04） |

## Method (when the symptom is NOT in the table)
1. **Measure before theorising.** Record stems (Record→Play writes per-voice stems to `recordings/`) and inspect the actual samples; count underruns/overflows from engine output. The zero-gap diagnosis succeeded because someone looked at the waveform instead of guessing.
2. Reproduce offline if possible (`--render` runs the live callback deterministically) — a symptom that survives offline is code, one that doesn't is device/clock.
3. Change ONE variable, re-listen/re-measure, keep notes. Then register the diagnosis here (and in GRAVEYARD if a theory died).

## 斷斷續續／規律 glitch（即時串流）— 2026-08-05
**症狀**：輸出從第一聲就斷斷續續或規律跳針，內容層怎麼修都在。
**診斷**：重活（模型推論等 >數十 ms）跑在音訊 callback 執行緒上＝
CoreAudio 死線抖動掉塊。「平均時間 < block 時間」不算數；且 sounddevice
callback 的 status 旗標若被忽略＝掉塊完全隱形（假「超時 0」）。
**修法**：callback 只搬記憶體（in→queue、queue→out，μs 級），推論放
worker 執行緒；status 旗標與輸出缺塊入遙測。實例：spike_stream.py v9
（v1–v8 七版內容層手術都沒治，v9 一刀收）。
