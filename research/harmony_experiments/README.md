# harmony_experiments

Probes, labs, corpus tooling and abandoned engine lines. **Nothing here runs in the
piece**, and most of it points at paths that only existed on the machine it was
written on. It is kept as the evidence behind the entries in
[`../../docs/FINDINGS.md`](../../docs/FINDINGS.md) and
[`../../docs/GRAVEYARD.md`](../../docs/GRAVEYARD.md), which is where the conclusions
live; the narrative is in [`../../docs/PROCESS.md`](../../docs/PROCESS.md).

The index below is so a reader does not have to open forty files to find the two
they want.

## Abandoned engine lines

| File | What it was, and what happened |
|---|---|
| `score_live.py` | The parts sing from a score while the singer conducts. Seven versions circling one question, who leads; ended with the parts playing pre-rendered stems. Superseded by the answering mode. |
| `score_align.py` | Aligning a sung phrase to a score with subsequence DTW. Contains the lesson that the alignment cost is INVERTED as a confidence measure, because a collapsed path is cheap. |
| `score_notes.py` | An a cappella MIDI file to a three-part tick grid: the score side, validated in bulk against 4016 public-domain scores. |
| `spike_stream.py` | Sliding-window streaming conversion with SOLA. The most heavily annotated file here; its version history is the whole latency-and-quality argument. |
| `sustain.py`, `sustain_live.py` | Sustain-triggered parts: hold a steady note and they join in on a batch-rendered chunk. |
| `perform.py` | Performance mode with the parts as a backing track. Documents the routing rule: dry voice direct to the PA, wet only from the computer. |
| `prerender_stems.py` | A rehearsal file to offline-quality stems, so the offline ceiling reaches the live output. |
| `world_mouth.py`, `world_stream.py`, `beatrice_mouth.py`, `sampler_mouth.py` | Four carriers that were tried: WORLD resynthesis, its strict-causal streaming version, timbre-only conversion, and unit sampling. `sampler_mouth` is option zero, playing real recorded sustains rather than synthesising, and its gesture layer is the most developed attempt at making that sing. |
| `solo_host.py` | The harmony model driving the proven engine in-process. |
| `lab.py`, `lab_ui.html` | A single-file web console for driving the live engine, with its stdout streamed to the browser. |

## Probes and measurement

| File | What it establishes |
|---|---|
| `mic_check.py` | Thirty seconds of recording-chain check before singing, after a whole round of listening was built on material with 13 dB less SNR than usual. |
| `key_probe.py` | The key of a real take from the UNSNAPPED f0. Written after the model's own representation snapped notes into C and made "the lead is in C" circular. |
| `angel_audit.py` | An eight-axis scorecard for the parts, every axis measured against something audible rather than an internal representation. |
| `clap_probe.py` | Whether a clap can be separated from singing by attack shape. The honest standard: not clap against average singing, but against the most clap-like moment in all the singing material. |
| `bank_gate_forensics.py` | Invariant checks on the gate state machine against a dump. |
| `bare_test.py`, `a_enh.py`, `loopback_lab.py` | The auto-tune investigation: a ladder from the dry take through the offline reference and the bare carrier to the full recipe, so the rung that introduces the artefact is the one to blame. `bare_test` also documents how the earlier material was withdrawn once cross-correlation showed the chosen segments were responses, not solo phrases. |
| `humanize_lab.py`, `gesture_lab.py`, `prosody_ab.py` | Three attempts at removing the auto-tuned quality from the pitch line: parametric humanising, transplanting real gestures from the corpus, and transplanting his own deviation curve. |
| `peek_probe.py`, `peek_probe_v2.py` | Whether the model can predict his NEXT note before he sings it. This is the measurement behind G29. |
| `rehearse_bench.py` | Score-tracker scoring over twelve pairs of double takes. |
| `ear.py` | A streaming pitch tracker on torchcrepe, measured against the hand-written one it replaced. |
| `blind_pack.py` | Packs stems for blind listening: one mix recipe for every cell, shuffled, with the key written separately. |
| `live_probe.py`, `resp2_probe.py` | Synthetic probes that run the whole live chain with a virtual microphone and nobody in the room. `resp2_probe` has to pause its play head while a response plays, or the phrase lengths it measures are fiction. |
| `vs_male3_trial.py` | Listening samples from a trial model; withdrawn with the same material as `prosody_ab`. |

## Corpus tooling

`cpdl_scrape.py`, `cpdl_pairs.py`, `cpdl_triples.py`, `chorale_triples.py`,
`corpus_probe_cpdl.py`, `corpus_parse_cpdl_mids.py`, `corpus_probe_pop909.py`,
`silver_line.py`, `tokenize_v2.py`, `corpus_split.py`, `corpus_split_v3.py` — how
the training corpus for the harmony model was assembled from public-domain choral
scores and a pop dataset, tokenised, and split by whole song so that adjacent
windows cannot leak across the split.

## scratchpad/

One-off analyses, mostly from the gate and vowel work.

| Group | Files |
|---|---|
| The camera gate | `mouth_gate_probe.py`, `mouth_blendshape_probe.py`, `sing_gate_train.py`, `sing_gate_train2.py`, `sing_gate_fit3.py`, `lip_ring_calib.py` — from hand-coded geometry, through blendshapes, to a model self-calibrated per session with a movement veto. `sing_gate_fit3` carries the finding that speech cannot be excluded by mouth shape at all. |
| The vowel layers | `layer_vowel_audit.py`, `vowel_source_scan.py`, `vowel_layer_rebuild.py`, `vowel_render_select.py`, `vowel_render_probe.py`, `vowel_acoustic_probe.py`, `vowel_fusion_probe.py`, `vowel_ml_probe.py`, `vowel_ml_train.py`, `vowel_fusion_train.py`, `vowel_record_ritual.py` — the five layers turned out to be labelled by brightness rather than by vowel. The recurring lesson: **rendering changes the vowel**, so material has to be checked after rendering, not at the source. |
| Other | `bank_fb_envcorr.py` (envelope-domain feedback detection, the measurement behind F26), `bench_vocoder_tail.py` (how much of the vocoder's work SOLA throws away), `melody_lm.py` (an interval model of his melodic habits) |

## A note on the Chinese that remains

A few strings are deliberately not translated because they are matched literally
rather than read: macOS audio device names on a Chinese-language system, the five
vowel labels that key `bank_live.VOWEL_LAYERS` and enter its cache key, and the
prompts spoken to the singer by macOS `say`. Each is annotated in English where it
appears.
