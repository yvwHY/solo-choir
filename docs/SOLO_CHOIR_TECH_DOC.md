# Solo Choir — Technical Documentation & AI-Assisted Development Process

*MA Computational Arts thesis project — Yu-Ting Liao, Goldsmiths. Draft documentation; figures are values measured during development on a MacBook Pro 16" (M2 Pro), macOS.*

---

## 1. What Solo Choir is

Solo Choir is a real-time instrument that turns one singer into a choir. A person sings; the system re-synthesises their voice and adds diatonically-correct harmony voices, in real time, low latency, on CPU. The thesis framing is *"the body as an interface"* — a **choir of one**, where a single performer's live voice drives multiple voices around them. The longer-term wearable form delivers the result into the body via bone-conduction.

This document describes the **software system** (the desktop application built on top of a validated real-time voice engine) and the **process** by which it was built, including the use of AI tools.

---

## 2. System architecture

Three layers, with a deliberate separation between the **audio path** and the **control/telemetry path**:

```
        ┌─────────────────────────────────────────────┐
        │  Desktop app (pywebview, native macOS window) │
        │   app/shell.py  →  serves ui/…_v10.html       │
        │   ┌─────────────────────────────────────┐     │
        │   │  UI (three.js / HTML / JS)           │     │
        │   │  heads-as-controls, readouts, score  │     │
        │   └───────────────▲──────────┬──────────┘     │
        │        telemetry  │          │ control         │
        │   ┌───────────────┴──────────▼──────────┐     │
        │   │  app/bridge.py (pywebview js_api)    │     │
        │   └───────────────▲──────────┬──────────┘     │
        └───────────────────┼──────────┼────────────────┘
              stdout (telemetry JSON)   │  stdin (control JSON)
                            │           ▼
        ┌───────────────────┴──────────────────────────┐
        │  Engine subprocess (server/, validated DSP)    │
        │   beatrice_solo_choir_live.py (harness)        │
        │   beatrice_converter.py + SoloChoir.py         │
        │   mic ──► convert (Beatrice) ──► harmony ──►out │
        └───────────────────────────────────────────────┘
```

Key point: **audio never travels through the bridge.** The engine subprocess owns the live audio (mic → conversion → output). The bridge only carries *telemetry* (engine → UI) and *control* (UI → engine). This keeps the UI free to be a rich WebGL surface without ever being in the audio-critical path.

---

## 3. Components

### 3.1 Engine (`server/`) — the validated DSP, treated as read-only
- A fork of `w-okada/voice-changer` with the **Beatrice v2** engine ported into an editable Python harness, running a personally-trained tenor model (`paraphernalia_00002000`).
- Runs in real time on CPU (no GPU). Neural voice conversion re-synthesises the voice at the target pitch, so **formants are preserved naturally** — this is why quality is far above a generic Tone.js / phase-vocoder pitch shifter.
- Harmony logic (`SoloChoir.py`): estimate f0 → MIDI → snap to the chosen key/scale (major/minor) → shift by *N* diatonic scale-steps. The engine is **monophonic per inference pass**.
- A long-standing buzz artefact (a ~100 Hz comb) was traced to a *stateless* per-block resampler restarting its anti-alias filter every 10 ms; the fix was a **stateful `soxr.ResampleStream`** per stream.

### 3.2 Bridge (`app/bridge.py`)
- pywebview `js_api`. Two directions:
  - **Telemetry (engine → UI):** reads the engine subprocess's stdout (JSON lines, ~30–40 Hz) and pushes it into the page via `window.onTelemetry({…})`.
  - **Control (UI → engine):** writes control JSON to the subprocess's stdin whenever a UI control changes.
- **Device handling:** enumerates real audio devices, picks sensible defaults (USB mic in; never the built-in speaker out, to avoid feedback), and **restarts the engine subprocess** when a device is changed (carrying current control state across). Devices can be **re-enumerated on demand** (refresh button / on dropdown open) so hot-plugged headphones appear without restarting the app.

### 3.3 UI (`ui/solo_choir_ui_v10.html`)
- A single self-contained three.js + HTML/JS surface. Cool, light "sapient" aesthetic.
- Five **glass heads** in a row (Bass / Tenor / **You** / Alto / Sop) loaded from a `.glb` head model via `GLTFLoader`, with a glass shader; falls back to a procedural head if loading fails. Heads are **interactive controls** — tapping a head toggles that voice.
- Singer-facing feedback: live note + Hz readout, a **tuning (cents) needle**, an **input-level meter**, and a **LIVE latency pill** showing real measured milliseconds.
- Transport (Record / Play) centred beneath the *You* head; key / scale / harmonize / ensemble / spread controls.

### 3.4 Desktop app shell (`app/shell.py`)
- Wraps the UI in a **native macOS window** (pywebview → WKWebView), serving the HTML from a tiny local static server so the `.glb` and assets load without browser/CORS friction.
- Starts/stops the engine subprocess with the window lifecycle.
- This makes Solo Choir a standalone desktop app — **not a website** — while reusing the three.js UI.

---

## 4. Data contracts

**Telemetry (engine → UI):** `{ f0, midi, cents, level, rtf, latency_ms, voiced }`

**Control (UI → engine):** `{ convert, key, scale, intervals, harmonize, pitch, gain, gate, you }`

The contracts were fixed early and kept stable, so the UI's JS never had to change when the data source moved from a mock generator to the real engine.

---

## 5. Technical problems solved (development log)

| Problem | Root cause | Solution |
|---|---|---|
| Buzz / comb artefact | Stateless per-block resampler restarting its filter every 10 ms | Stateful `soxr.ResampleStream` |
| UI controls did nothing | Engine ran on fixed start-up CLI flags; UI control wasn't reaching it | stdin control-reader thread in the harness; bridge writes control JSON to stdin; audio loop reads params per block |
| "You" sounded like the tenor model, not the singer | *You* was a 2nd conversion pass at unison | *You* changed to a **raw input passthrough**, delay-aligned to the harmony, so it is the singer's real timbre |
| True simultaneous harmony | Engine is monophonic per pass | A **second inference pass** (You + 1 harmony) summed at output — the first real multi-voice step; still real-time (measured RTF ≈ 0.28–0.36 with 2 passes) |
| Perceptible latency | Audio buffering (block size + in/out buffers), not the UI/bridge | Smaller block size + explicit numeric latency setting (~250 ms → ~95 ms mouth-to-ear), exposed as adjustable flags; real latency/RTF surfaced to the UI |
| `input overflow` crackle | **USB mic and output device are two independent clocks**; one duplex stream drifts | Larger buffer mitigates; the clean fix is a macOS **Aggregate Device** (drift correction) or a single audio interface |
| "Reverb / echo" on the *You* channel | Acoustic real voice + the ~150 ms-delayed speaker playback = slap-back (a monitoring artefact, **not** a code bug) | Disappears on headphones; addressed at the demo level by Record→Play |
| Feedback when played out loud | Live mic + speakers form an acoustic loop (worse with the raw *You* passthrough) | **Record→Play** for loud presentation (mic off during playback → zero feedback); headphones / bone-conduction for live use |

---

## 6. Key design decisions & rationale

- **"You" is the singer's raw voice, not a converted timbre.** This gives a real duet-like timbre contrast (your voice + a synthesised harmony) instead of two tenor clones, is conceptually truer ("You" is actually you), is cheaper (no inference pass for *You*), and sets up the multi-timbre direction (Phase 2) where the *other* heads can be different models/timbres.
- **The "You" toggle is a context switch, not a frill.** In the wearable / bone-conduction case the singer's *real acoustic voice* already provides the unison, so *You* can be off and the engine only adds harmony. In a closed-headphone / standalone / recording context, *You* is synthesised so the captured/heard output contains everything.
- **Record→Play for loud demos.** Live, real-time harmony always works on headphones (no feedback). Playing *out loud* through speakers with a live mic causes feedback, so the audience-facing presentation records a take (engine still runs in real time) and plays it back with the mic off. This does **not** sacrifice real-time; it is a clean way to let a room hear it.
- **Multi-voice is staged.** Two simultaneous voices (You + one harmony) are done now and stay real-time. Full SATB and ensemble *audio* thickness multiply the inference count and are **Phase 2** (they need optimisation to stay real-time).

---

## 7. Verification & guardrail methodology

A consistent discipline was used because the audio engine was already validated and must not regress:

- **Protect the DSP.** Changes were confined to plumbing (telemetry tap, control input, output mix gate, stream settings). The core DSP maths (f0 estimation, conversion, resampling, diatonic computation) was never altered.
- **Byte-identical regression as proof.** Each engine-touching change was verified by running the same audio through the engine with the change inactive/at defaults and confirming **bit-identical output** to the previous version (e.g. *you-only == unison*, *harmony-only == old harmony*, max-abs-diff 0.0). `git diff` was used to show additions-only.
- **One step, then stop for human review.** Work proceeded in small increments, each ending with a pause for the human to verify by ear/eye before continuing.
- **Honest mock-vs-real labelling.** Placeholder data (mock telemetry, the demo melody) was clearly marked and removed deliberately when the real signal was wired in.
- **Commit checkpoints.** Working states were committed so any step could be rolled back cleanly.

---

## 8. AI-assisted development process

This project was built with substantial use of AI tools. Documenting that honestly is part of the methodology.

### 8.1 Tools and roles
- **Claude (chat):** a design and advisory collaborator — conceptual clarification, architecture options, references to comparable systems, music/DSP explanation, UI prototyping (iterating a three.js interface design across many versions), and writing specifications and this documentation.
- **Claude Code (agentic, on the machine):** the implementer — wrote and edited the application code (`app/`, `ui/`), added the engine's read-only telemetry tap and live-control plumbing, ran its own regression/verification scripts, and committed work.

### 8.2 Division of labour
The human (the author) **directed and judged**; the AI **advised and implemented** under that direction and within explicit guardrails:
- *Human-owned:* the concept, the musical and aesthetic decisions, what "good" sounds/looks like, and all by-ear / by-eye verification (singing into the mic, watching the UI, judging timbre, latency, feedback).
- *AI-assisted:* surfacing options and trade-offs, drafting prototypes and specs, implementing and self-verifying code, and explaining unfamiliar territory (e.g. why formant-preserving conversion beats naive pitch-shifting; why simultaneous harmony requires multiple inference passes; why cross-clock devices overflow).

### 8.3 Workflow loop
A repeating loop was used:
1. **Decide / design** in chat (concept, architecture, UI direction).
2. **Specify** — a written build spec and an interactive UI prototype acted as the hand-off artefact.
3. **Implement** in Claude Code, one bounded step at a time.
4. **Verify** — the author ran the app, sang, and listened/looked; the AI self-verified the data path with offline tests.
5. **Refine** based on what the author heard/saw, and repeat.

### 8.4 Guardrails that made AI assistance safe
- Protect the validated DSP; only add plumbing; prove non-regression byte-for-byte.
- Small steps with a human review gate between them.
- Commit checkpoints so any AI change is reversible.
- Clear scope boundaries (Phase 1 vs Phase 2) to stop feature creep.

### 8.5 Honest reflection — where AI fell short
AI assistance was not infallible, and **human testing repeatedly caught its mistakes**, which is itself part of the method:
- An over-corrected "smooth, featureless" head rendered as a glowing **egg** rather than a head; the author's visual judgement redirected it to a real head mesh (with the AI then explaining *why* — a face is recognised by its features).
- A latency optimisation set the audio buffer **too small**, causing cross-clock **input overflow** (crackle); caught by ear, then fixed.
- Several UI controls appeared wired but were **not reaching the engine**; the author discovered this by testing (toggling Convert/Harmonize and hearing no change), which surfaced a whole integration gap.
- An apparent "reverb" turned out to be a **monitoring artefact**, not a bug — distinguished by the author switching to headphones.

The throughline: AI accelerated implementation and broadened the option space, but **direction, taste, musical judgement, and real-world verification stayed with the human**, and that verification was essential precisely because the AI's confident output was sometimes wrong.

---

## 9. Current state & roadmap

**Working now:** standalone pywebview desktop app; live real-time conversion + diatonic harmony; the singer's raw voice + one synthesised harmony summed simultaneously; live UI (note, Hz, cents, input level, real latency); per-voice and key/scale/harmonize controls actually driving the engine; device selection with hot-plug refresh; Record→Play path for clean loud presentation.

**Phase 2 (planned):**
- True multi-voice — *N* inference passes mixed (full SATB), with optimisation to stay real-time.
- Multiple timbres — different voice models, or multiple speaker embeddings within Beatrice, per part.
- Ensemble / dispersion *audio* (currently a visual-only control).
- Score mode — Record → Basic Pitch transcription → part playback.
- Wearable / bone-conduction hardware: intraoral microphone, bone-conduction output (e.g. Dayton BCE-1), amplifier (PAM8403), microcontroller I/O (Pico W).

---

## 10. Build, run, licensing

- **Run (development):** launch the desktop shell with the project's Python environment; it opens the native window and starts the engine.
- **Standalone packaging:** intended as a macOS `.app` (py2app / pyinstaller) wrapping the pywebview UI.
- **Audio setup note:** to avoid `input overflow`, use input and output on one clock domain — a macOS **Aggregate Device** (with drift correction) or a single USB audio interface.
- **Licensing:**
  - The Beatrice engine binary and the trained voice model are **license-restricted (academic/personal use, no redistribution)** and are kept **out of version control**; they are referenced by path and backed up separately.
  - The head model used by the early interface (`app/assets/head.glb`, "CCO_ Male_base_mesh_standing" by iamsunroy, Sketchfab) is under the **Sketchfab Standard** licence, which does not allow redistribution of the file, so it is not in the repository; source and download instructions are in `app/assets/CREDITS.md`.

---

*This is a living document — keep it updated as Phase 2 progresses, and treat Section 8 as the honest process account for the thesis and viva.*
