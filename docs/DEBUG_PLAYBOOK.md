# DEBUG PLAYBOOK — Live-Audio Symptom → Known Root Cause

**Audience: AI assistants (Claude, Codex, any model).**
Check this table BEFORE touching buffers, threads, or the engine. Most live-audio symptoms in this project already have a diagnosed root cause; the historical failure mode is theorising about buffers when the cause was elsewhere (see GRAVEYARD G5, G7).

| Symptom | Root cause (diagnosed) | Fix / where |
|---|---|---|
| Voice toggle sounds choppy or broken up | Cold converter emits ~35 ms of zeros on resume — NOT buffers/underruns | Keep every live voice warm, gate at mix (FINDINGS F4) |
| Scrape / instability when harmony pitch changes | Every `set_pitch_shift_semitone` *change* scrapes engine state | Glide 3–6 ¢/10 ms + 80 ms debounce (FINDINGS F2) |
| ~10 Hz tremolo on converted output | Chunk + context re-feed interrupting the stateful stream | Don't re-feed; one continuous stream (GRAVEYARD G2) |
| Comb-filter / blur when combining voices | Mixing two engine streams — phase free-runs per stream | Never crossfade engine streams (GRAVEYARD G1) |
| `input overflow` messages | USB mic + output = two independent clocks drifting in one duplex stream | Aggregate Device w/ drift correction, or one USB interface; bigger buffers only delay it (FINDINGS F7) |
| PortAudio `-10851 Invalid Property Value` | Two streams on the same device at different sample rates | Everything at 48 kHz (FINDINGS F7) |
| Speaker(s) cut out intermittently, session log CLEAN (0 underrun/overflow) | NOT software — self-powered voice-unit LiPo low: loud passages spike current → MT3608 sags → PAM8403 brownout (recovers when quiet). MT3608 idle drain is mA-class → a battery left on the rail dies in days | Charge/swap the LiPo (TP4056 red LED = confirmed low); add a master switch to the rail; never charge while playing (TP4056 has no load-sharing) — diagnosed 2026-07-19 |
| bank_live: recurring noise or glitches, **`xrun` low and the `--dump` sounds clean** | Worker backlog > `--maxlag` ⇒ engine *drops* a computed block (`bank_live.py:1849`), punching a 35 ms hole in sustained notes. **Both instruments are structurally blind to it**: the dump is written inside `process_hop` (`:1819`) *before* the drop, and the code's own comment says the xrun counter sees nothing at all. Read `dropped` in the status line instead — it only prints when >0 | The RT thief is the **mouth-monitor window**: `--view 1` → dropped 8 per 90 s idle (reproduced twice, no singing, gate closed); `--view 0` → **0**. Show config must run `--view 0`; use `--view 1` only for the pre-show gate self-check, then restart — measured 2026-08-15 (FINDINGS F25) |
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
| Crackle riding ON the voice (more voice → more crackle), present in the interface's RAW input recording, gain-independent, swapping the output path doesn't help | Audio interface wedged in a bad internal state (AI-Micro after hot-plug churn) — NOT clocks, NOT the throat mic/cable, NOT the amp/output chain; verified by recording the raw input digitally (710 impulsive events → 11 after reset) | Fully unplug→replug the interface (power-cycle reset), then restart the app (snapshot rule above). Diagnose by recording the interface input directly — bypasses the whole app. **Show-day SOP: power up in fixed order (all hardware plugged → interface powered → app); on ANY unexplained audio anomaly, power-cycle the interface + restart the app FIRST (~30 s, collapses the whole wedged-state class) before theorising** (2026-07-14) |
| Dry You sounds thin / "high-pass filtered"; converted voices gritty/unstable | Throat mic not coupled to the neck (a contact mic in air only picks up thin airborne bleed) and/or interface input gain far too low (AI-Micro ships low — peaks were 0.03 FS, −29 dBFS) | Strap the mic snug; set input gain in RØDE Central so performance-volume peaks land ~0.2–0.5 FS. When swapping interfaces, verify input LEVEL height, not just signal presence (2026-07-14) |
| After Ctrl-C the process will not exit, the output is frozen on the last "listening" line, and no session dump is written | SIGINT was received and the main loop has exited, but it is stuck in `__exit__` of `with sd.Stream`: PortAudio's `FinishStoppingStream` hangs inside CoreAudio (`sample <pid>` proves it) and does not clear itself after 60 s or more. A second Ctrl-C does nothing, since the signal cannot reach the C code and only accumulates as a pending KeyboardInterrupt that will bite the next piece of Python | respond2 now writes the dump and the statistics **before** closing the stream (2026-08-04), so a hang no longer costs data and `kill -9` is safe. If another script hits this while the data is still in memory: attach lldb, call `PyGILState_Ensure`, then `PyRun_SimpleString` to inject code that walks `sys._current_frames()` and writes the arrays out of the frame `f_locals` (plain numpy arrays never appear in `gc.get_objects()`, so do not go through gc). If the injection is destroyed by the pending KeyboardInterrupt and returns -1, that interrupt has now been consumed and a second attempt succeeds (2026-08-04) |

## Method (when the symptom is NOT in the table)
1. **Measure before theorising.** Record stems (Record→Play writes per-voice stems to `recordings/`) and inspect the actual samples; count underruns/overflows from engine output. The zero-gap diagnosis succeeded because someone looked at the waveform instead of guessing.
2. Reproduce offline if possible (`--render` runs the live callback deterministically) — a symptom that survives offline is code, one that doesn't is device/clock.
3. Change ONE variable, re-listen/re-measure, keep notes. Then register the diagnosis here (and in GRAVEYARD if a theory died).

## Broken-up or regularly glitching output (live streaming) — 2026-08-05
**Symptom:** the output is broken up or skips regularly from the very first
sound, and no amount of work at the content layer removes it.
**Diagnosis:** heavy work, such as model inference taking tens of milliseconds or
more, is running on the audio callback thread, so jitter against the CoreAudio
deadline drops blocks. "The average time is below the block time" does not
count; and if the status flag of the sounddevice callback is ignored, dropped
blocks are completely invisible and the overrun count falsely reads zero.
**Fix:** the callback only moves memory, input to queue and queue to output, at
microsecond scale, while inference runs on a worker thread; the status flag and
missing output blocks go into the telemetry. Example: spike_stream.py v9, after
seven versions of surgery at the content layer, v1 to v8, had cured nothing.
