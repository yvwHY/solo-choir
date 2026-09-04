# PROCESS — how Solo Choir was built

A condensed account of the development, June to August 2026. It replaces 53 daily
worklogs and a rolling status file, which were working notes in Chinese and are
kept in the development repository rather than here.

What did and did not work is recorded in two companion files:
[`FINDINGS.md`](FINDINGS.md) holds the measured numbers, and
[`GRAVEYARD.md`](GRAVEYARD.md) holds every approach that was tried and abandoned,
with the evidence that killed it.

---

## The method

Three rules governed the work and explain most of the decisions below.

**The ear decides; measurement guards against regression.** Every substantive
change was judged by listening, usually blind and level-matched. Objective
metrics were used to stop engineering from going backwards and to compare
versions inside one layer, never to overrule a listening verdict. Several
entries in GRAVEYARD are cases where a metric improved and the sound did not
(G30, G31, G33, G35).

**A dead end is recorded rather than forgotten.** Each abandoned approach has an
entry giving what was tried, the evidence that killed it, and the conditions
under which it could be reopened. Two of them were later reopened on those
conditions (G29 in part, G34 in full) once the premise had changed.

**Behaviour is not changed by accident.** Where a new feature touches a path
that has already passed by ear, it is added behind a flag whose default
reproduces the previous behaviour bit for bit, and that equivalence is verified
by re-rendering and comparing. This is why so many options in the code read
"0 = off = the old behaviour".

---

## Phase 1 — Voice material and the first engine (18 June – 8 July)

The project began with recording. The singer's own voice, a friend's voice
recorded with consent, and published singing corpora were prepared into training
clips, and singing-voice conversion models were trained on them locally.

The first working instrument ran on a real-time conversion engine driven from a
desktop application: microphone in, conversion, diatonic harmony, output. The
architecture separated the audio path from the interface at this point and has
kept that separation since: the engine runs as a subprocess and owns all live
audio, while the bridge to the interface carries only telemetry and control.

One long-standing buzz was traced during this phase to a stateless per-block
resampler restarting its anti-alias filter every 10 ms, and fixed with a
stateful resampling stream.

## Phase 2 — The wearable frame (9 – 21 July)

A parallel track built the worn version: printed plates on a hand-bent brass
tube frame across the head, neck and chest, with bone-conduction drivers at the
skull and exciters on panels. This produced five work orders of CAD and print
iterations, an integrated driver box, and a wire-light that runs an audio
envelope along the cable towards the sounding body.

Two results from this phase shaped the final form. The bone-conduction drivers
held, because they only have to rest against bone. The exciters did not, because
an exciter has to drive a panel that is stiff, light and still able to follow the
body, and a plate hung from a frame is none of those.

The clearance failure recorded as G25 also belongs here: narrowing every brass
channel to a single uniform figure worked for straight holes and failed for
curved ones, because the old, looser figure had been absorbing the error of a
hand-bent rod.

## Phase 3 — A harmony model, and what a carrier is (22 – 28 July)

The harmony model was written as a causal language model over interleaved parts,
trained on public-domain chorale scores, producing a note per part per tick from
the singer's detected note.

This phase established the distinction the rest of the project rests on. A
**converted carrier** takes the live voice as its input, so the words and the
legato are inherited for free. A **concatenative carrier** takes a list of notes,
so everything has to be built. Option zero, a sampled voice built from harvested
units, failed four rounds of listening (G32) for two structural reasons: there
is no recorded material for the transition between two notes, and there is
nowhere for the words to come from. A syllable-band envelope correlation of
0.878, close to the converted carrier's 0.951, was still heard as "a skin of
timbre".

Three further approaches died here: a granular dual-tap pitch shifter (G26),
external control of the conversion engine as a live pitch channel (G27), and
naive streaming resynthesis (G28). G26 produced the lesson that passing on a
synthetic test signal is not passing on a real voice, after which every voice
pipeline was accepted against real recordings and an offline replay loop.

Anticipation was also measured and abandoned in this phase (G29): predicting the
singer's next note from a model trained on Bach sopranos scored 0% on timing and
at the level of guessing on pitch. The counterpoint half, reacting to what has
already been sung, was never broken and remains in service.

## Phase 4 — Streaming, and the auto-tuned quality (29 July – 6 August)

Attempts to make the parts sound at the same time as the singer ran into two
walls at once: latency and a quality the listening verdicts consistently called
auto-tuned.

The latency wall was architectural. Sustained-note triggering with block-batch
rendering (G33b) reached about a second end to end, of which roughly half is a
floor set by the detection window and the render, so repairing it fully would
still not pass.

The auto-tune investigation is the longest single thread in the project. It was
first assumed to be the f0 curve; three rounds of humanising the curve failed.
Ablation then placed the artefact in the synthesis carrier rather than the pitch
line, since even a bit-faithful copy of the singer's own pitch still sounded
corrected. Two changes removed most of it: a neural vocoder enhancer, which
repairs the flat harmonics of the bare low register, and desynchronising the
vibrato between parts, since two parts on the same phase and rate at once are
mechanically synchronised in a way no ensemble is. A rule-based expression layer
was measured against 19 sustained runs of real singing and reproduced its
components — a per-note offset, a slow drift, a fast tremor, an irregular
vibrato and a scoop into the note.

That expression layer then produced its own problem: with each part deviating
independently, the interval between them had a standard deviation of 40 cents
and exceeded 20 cents, audibly out of tune, 61.6% of the time. The fix separates
the layer in two — the slow component, which is the pitch centre, is shared
between parts, while the vibrato and the jitter stay independent. A choir tunes
to itself and differs in its movement, not in its centre.

## Phase 5 — The answering mode (1 – 6 August)

The shape that survived came from turning the constraint into the form. The
singer sings a phrase; when it ends, the system renders the harmony at offline
quality and plays it back together with the phrase just sung.

This removes both walls at once. There is no alignment problem, because the
harmony is made for a phrase that has already finished. There is no simultaneity
pressure, because a response after a phrase is call-and-response grammar rather
than latency. The gap between phrases is long enough to run the reference
render.

The engineering that followed was mostly about honesty and safety. The input is
hard-muted from the moment a phrase ends until the response and its reverb have
finished, which removes feedback, the speakers being mistaken for the singer,
and the response bleeding into the next phrase, in one structural move rather
than three detectors. Because the microphone cannot hear the singer during that
window, a physical button became their only channel: pressed while listening it
ends the phrase at once, and pressed while muted it cancels the response.

Phrase-end thresholds are measured on the spot at the start of each session
rather than inherited, after a threshold measured with a throat microphone was
carried over to an air microphone and produced a single 34.6-second "phrase".

## Phase 6 — The sampled choir (11 – 16 August)

A second engine generalised the principle "the score is fixed, so do not render"
into "the pitch space is fixed, so do not render live". Since the parts sing one
vowel in chromatic harmony, the set of possible outputs is finite: at start-up
each part's whole range is pre-rendered semitone by semitone into loopable banks
at reference quality and cached, and the live loop only detects a note,
quantises it, triggers three or four parts and crossfades between them. With no
machine learning in the loop, the latency falls to about 36 ms of detection plus
audio I/O, at the quality of the offline render.

Two problems in this engine were traced to interference rather than to any added
component. Sixteen voices are four detuned copies of each part; a pair of
eight-singer models trained to replace those copies with real timbres could not
be told apart from them in blind comparison, so the copies shipped. And laying
the same note from two different recordings on top of each other always leaves a
slight pitch difference, which was the intermittent noise reported live —
audible, and invisible to all five instruments that were pointed at it, because
it is not an added component but interference between existing ones.

The bass was moved down eight semitones in this phase after a dump showed it
sitting at the same median pitch as the singer, so two male parts two semitones
apart were one thickened line rather than two, with nobody in the bass resonance
region at all.

## Phase 7 — Gating, and what a camera can and cannot see (10 – 18 August)

Running bare, the system feeds back: the parts coming from the speakers are
perfectly periodic, so the detector takes them for the singer and it sings on by
itself. Three approaches to separating the two were tried.

Discriminating in the energy domain from a single microphone failed structurally
under the streaming architecture (G34), because the loop's envelope is
self-similar and any energy ratio tends to a constant. A cepstral feature meant
to separate one note from a three-part chord turned out to be a proxy for level
(G35). Both entries carry the same lesson, recorded after the fact: ask how the
ground truth was cut before trusting a discriminator measured against it.

A camera gate worked, because feedback cannot fool the lips. A model was trained
on mouth and jaw blendshapes and reached 93.4% accuracy across sessions with no
false triggers on a closed mouth. Its honest boundary is that a picture cannot
separate singing from speech, since the mouth is open either way.

On 18 August the gate authority moved back to the microphone, once the sampled
choir's output was no longer a function of its input and the energy argument no
longer applied. That gate mistakes feedback for singing about a third of the
time by definition rather than by measurement error, and it was accepted because
raising the threshold brought false opens to a level the ear accepts. The camera
was retired entirely.

## Phase 8 — Freezing (16 – 24 August)

The last phase changed almost nothing about how the instrument sounds. Two full
reviews of the performing code produced 45 findings between them, of which the
severe ones were structural rather than musical: an engine thread that could die
and leave silence with every indicator green, a dump written after a stream close
that could hang, a latency backlog that was never caught up, and a gate that
could never close.

The performance configuration was then frozen flag by flag and written down with
the reason for each value, so that rehearsing from the command line and
performing from the application are the same instrument.
