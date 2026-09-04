# UI build spec

Written 2026-06-29, when the interface design was frozen and handed to
implementation. It is kept because it records the interface decisions that are
still in force, and because it shows what the project expected at the end of
June. **It is not a description of the finished instrument** — see the box at
the end for what actually shipped, and [`PROCESS.md`](PROCESS.md) for why.

The frozen design was `solo_choir_ui_v10.html`: a cool light theme, five glass
heads in SATB order, centred Record and Play, record then transcribe then play
back part by part, Ensemble and Spread controls, and three feedback readings for
input level, latency and tuning.

---

## 0. Core principle

**Do not touch the engine.** The conversion engine, the stateful
`soxr.ResampleStream` and the diatonic harmony logic were already verified
working. The interface work adds a control, telemetry and packaging layer
*around* the engine and changes no audio DSP. Everything that moves in the
prototype is fake and has to be replaced one item at a time by real engine data.

## 1. Form: a desktop application, not a web page

`pywebview` wraps the HTML and three.js in a native macOS window. The interface
is reused unchanged, because WKWebView supports WebGL. The Python engine and the
interface share a process: the GUI owns the main thread and the audio engine
runs on a background thread.

Alternatives considered and rejected: Electron (heaviest), Tauri (light but
needs Rust plus a Python sidecar), PyQt/QML (equivalent to giving up the WebGL
heads).

Licensing: the real-time conversion engine's binaries may not be redistributed.
Packaging for local use and for academic presentation is allowed; an application
containing that binary must not be published.

## 2. Bridge

Either of two bridges, both supported by pywebview:

- **js_api** — the simplest, single process. Python exposes control functions
  with `window.expose()`; the engine pushes telemetry with
  `window.evaluate_js("onTelemetry({...})")`.
- **A localhost WebSocket** on port 8770 — decoupled and easier to extend. The
  engine pushes telemetry at 30-60 fps and the interface sends control.

Telemetry, engine to interface:

```
{ f0: Hz, midi: float, cents: float, level: 0..1 (RMS), rtf: float, latency_ms: float, voiced: bool }
```

Control, interface to engine:

```
{ convert: bool, key: 0..11, scale: "major"|"minor",
  intervals: whichever of [-7,-2,2,4] are enabled, harmonize: bool,
  pitch: semitones, gain: dB, gate: dB }
```

## 3. The 3D heads

`GLTFLoader` loads a `.glb`, takes the geometry of the first mesh in the scene,
centres and scales it with `normalizeHeadGeo()` (orientation correctable through
`MODEL_ROT_Y` and `MODEL_TILT`), and applies the existing glass shader: fresnel,
additive, fading out from the bottom up.

There is a fallback: if `GLTFLoader` is missing, the load fails, or nine seconds
pass, the interface reverts to a procedural head and nothing breaks. A badge in
the bottom right corner shows which one is in use, green for the model and
orange for the fallback. The scene is only built after loading, and event
handling is safe against an empty pick list.

Notes carried forward: use a local file for `MODEL_URL` so there is no CORS
question; reduce the polygon count or add LOD before putting five detailed heads
in a WKWebView; the inner lattice is off and the wireframe is fainter on the
model version, because a wireframe on a dense mesh reads as noise.

## 5. Interface control to engine parameter

| Control | Engine |
|---|---|
| Convert | start and stop the conversion engine |
| Key and Scale | `--key`, major or minor |
| Each head (Bass -7, Tenor -2, Alto +2, Sop +4) | which interval passes are enabled; You is the source |
| Harmonize | the harmony master switch |
| Pitch slider | transpose, formant preserving |
| Output gain, Noise gate | engine parameters |

## 6. Order of work

1. The pywebview shell, confirming WebGL works in a native window.
2. The bridge of section 2, touching no DSP.
3. Live signal into the readings, the heads, Convert, key, scale and intervals.
4. Feedback: input level, latency, tuning.
5. Record, transcribe, display, play.
6. The 3D heads, one before all five.
7. Packaging.

---

## What actually shipped

The answering mode did not exist when this was written, and the interface it
describes is the pre-answering one. The parts of this document still in force
are the pywebview shell, the js_api bridge, the telemetry and control shapes,
and the rule that the engine owns all audio while the bridge carries only
telemetry and control. What changed:

- The engine runs as a **subprocess**, not a background thread, so a crash
  cannot take the window with it, and audio never crosses the bridge.
- **Record, transcribe and play back** was replaced by the answering mode: sing
  a phrase, and the response is rendered at offline quality and played back with
  it. There is no transcription step.
- **Ensemble and Spread** never became audio controls. Spatial separation is
  done by routing parts to separate speakers.
- The three feedback readings survived; the score display did not.
