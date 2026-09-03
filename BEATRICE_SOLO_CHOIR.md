# Solo Choir — Beatrice v2 port (final state)

Phase 1 of the "Solo Choir" instrument, ported onto the performer's **trained Beatrice v2
tenor**. The performer sings a melody; the app produces one additional **tenor** voice whose
pitch is a **diatonically-correct interval** relative to the sung melody, computed live and
applied as a per-chunk pitch shift on the Beatrice engine. Output = the harmony line (the dry
melody is the performer's own acoustic voice) → a 2-part texture.

## What works (validated on M2 Pro, CPU)

- **Trained tenor timbre** — loads the user's `paraphernalia_data_00002000` model.
- **Diatonically-correct harmony** — e.g. C major, a third below: C→A, D→B, E→C, F→D, G→E,
  A→F (shifts alternate −3/−4 semitones with scale degree). Key (root + major/minor) and a
  signed scale-step interval are configurable.
- **Low latency** — engine RTF ≈ 0.12 (≈1.2 ms per 10 ms frame); full per-frame pipeline
  (f0 + harmony + convert) ≈ 1.2 ms/frame.
- **Clean — at parity with the original VCClient** on the same mic and model (verified by
  back-to-back A/B). No buzz beyond the mic's own floor.

Demo recording: [`server/demo/beatrice_solo_choir_demo.wav`](server/demo/beatrice_solo_choir_demo.wav)
(stereo: L = dry voice / melody, R = tenor harmony).

## Architecture

Per 10 ms frame (160 samples @ 16 kHz), entirely in the editable layer:

```
mic frame (160 @16k)
  -> external f0 detector (autocorrelation on a short rolling window)
  -> SoloChoir.py: f0 -> MIDI -> snap to key -> diatonic shift -> transpose (semitones)
  -> SimpleBeatrice.set_pitch_shift_semitone(transpose)
  -> SimpleBeatrice.convert(frame) -> tenor output (240 @24k)
```

- **`server/voice_changer/SoloChoir.py`** — engine-agnostic music logic (f0→MIDI, snap to key,
  signed diatonic scale-step shift with octave carry, transpose = target − input) plus held-note
  hysteresis (anti-warble) and hold-through-unvoiced. Tests: `server/voice_changer/test_solo_choir.py`.
  This module is shared verbatim with the earlier RVC prototype — the music logic is engine-agnostic.
- **`server/beatrice_converter.py`** — `BeatriceSoloChoir`: drives the Beatrice rc.0 engine
  (`SimpleBeatrice`), runs the external f0 detector, feeds `SoloChoir`, applies the transpose via
  the engine's per-call `set_pitch_shift_semitone`. Optional output envelope tuning (default off).
- **`server/beatrice_solo_choir_live.py`** — standalone realtime app: sounddevice duplex stream
  @ 48 kHz, **stateful `soxr.ResampleStream`** for 48↔16 / 24↔48, ring-buffered output,
  device selection by name, optional `--record` for the stereo demo.

Note: rc.0 hides the engine's internal f0 (only the unified `SimpleBeatrice.convert` is exposed),
so the input f0 is estimated externally — matching the original "drive Beatrice from outside" plan.
On the older beta.1 engine the internal `quantized_pitch` is exposed and could be read directly
(see `memory`/`beatrice_converter` history), but the user's tenor is rc.0.

## Fix history (what each problem turned out to be)

1. **No sound at all (RVC baseline)** — fp16 RVC ONNX models emit NaN on the CPU provider
   (RandomNormalLike + fp16 overflow); converted offline to fp32. (RVC prototype only.)
2. **Latency ~5 s on RVC** — RVC fp32-on-CPU is RTF > 1; Beatrice is RTF ≈ 0.12, so the port
   solved latency outright. (Decision: don't chase RVC GPU.)
3. **Sustained-note warble ("oo-ee")** — input f0 jitter flipped the snapped note across a
   scale boundary; fixed with held-note **hysteresis** in `SoloChoir.py`.
4. **Constant high-freq buzz ("滋滋") live, but VCClient clean** — ROOT CAUSE: the harness
   resampled **statelessly per callback block** (`resample_poly`), restarting the filter every
   10 ms → a ~100 Hz boundary comb heard as sizzle. Fixed by **stateful `soxr.ResampleStream`**
   (continuous across blocks). This reached VCClient parity.
   - Ruled out along the way: the model (clean studio `sample.wav` converts clean), the convert
     config, the input block size (engine requires exactly 160), and the mic (VCClient is clean
     on the same mic). The earlier "buy a better mic" conclusion was **wrong** — it was the
     resampling.
5. **Audio device index churn** — indices change on headphone plug/unplug; fixed by selecting
   devices by name (`--in-name` / `--out-name`).

## How to run

Environment: conda env `vcclient-dev` (Python 3.10). Engine path via `BEATRICE_ENGINE_DIR`
(default points at the extracted rc.0 engine; see license note).

```bash
cd server
# list devices
PYTHONPATH=. python beatrice_solo_choir_live.py --list-devices

# live: USB mic -> speakers, C major, a third below
PYTHONPATH=. python beatrice_solo_choir_live.py \
  --in-name "USB PnP" --out-name "揚聲器" \
  --key C --minor 0 --interval -2

# record a stereo demo (L=dry voice, R=harmony) for 12s
PYTHONPATH=. python beatrice_solo_choir_live.py \
  --in-name "USB PnP" --out-name "揚聲器" --interval -2 \
  --record demo/beatrice_solo_choir_demo.wav --rec-seconds 12
```

Intervals: `-2` third below (default), `+2` third above, `-5` sixth below, `+5` sixth above.
Key: `--key C..B`, `--minor 1` for natural minor.

Tested device setup on this machine: input `USB PnP Audio Device`, output `MacBook Pro的揚聲器`
(or `外接耳機` headphones). With headphones there's no feedback regardless of mic. Devices are
selected **by name** because their numeric indices shuffle when headphones are plugged/unplugged.

## License caveat (important)

The Beatrice v2 engine (`v20rc0.abi3.so` + the `beatrice` package) and the trained model `.bin`
files are **NOT committed to git**. The Beatrice API license prohibits commercial use and
redistribution. The engine was extracted from the user's own installed VCClient
(`…/VC/dist/main` → self-extracted bundle) into `_beatrice_inspect/beatrice_engine_rc0/`
(outside the repo). Point `BEATRICE_ENGINE_DIR` at a local copy; the model dir is passed via
`--model` (default points at the user's `paraphernalia_data_00002000`). Keep both out of any
published repository.

## Not done / future

- Multi-voice (SATB): run the engine N× per frame with different diatonic transposes and mix
  (Beatrice's low RTF leaves ample headroom on CPU).
- Score-driven mode; a real UI; bone-conduction / haptic output (Phase 2 backlog).
- A quieter mic would lower the residual floor further, but it is already at VCClient parity.
