# Solo Choir — Research Vision & Unifying Roadmap

*A Wearable Instrument for **Learning** and **Extending** the Singing Body.*
*Triple-duty doc: (1) viva argument backbone · (2) successor to HARDWARE_ROADMAP framing · (3) draft spine for a PhD / MIT Media Lab statement. Living document — last synthesised 2026-06-23.*

---

## 1. One-line thesis

**The body as the interface for harmony.** One singer becomes their own choir — and the choir is *learned in the body* and *performed from the body*, not mediated through a screen or a pair of speakers.

Two modes of one idea:
- **Learn in-body** — an independent harmony target is delivered into the head by bone conduction (open ear), so you hear your own voice *and* the target and train the interval by feel.
- **Perform from-body** — all harmony parts sound *at once* from one body, transcending the human voice's one-note-at-a-time limit.

## 2. The research question

> *Does delivering harmony **into / from the body** change how a singer **learns** and **performs** vocal harmony, compared with screen- or speaker-mediated harmony?*

This is the spine. Both tracks below exist to answer it; neither is a standalone demo.

## 3. Why this is a contribution (not just a vocal effect)

- **Simultaneity = literal body-extension.** A human voice is monophonic — one pitch at a time. Solo Choir makes one body a *simultaneous* SATB chord. This is the "choir of one": not an effect, but a capability the unaided body does not have. *("transcending the one-note-at-a-time limit of the human voice — the body made into a simultaneous chord.")*
- **Open-ear bone conduction is pedagogically motivated, not decorative.** To tune, a singer must hear their own real voice; closed headphones occlude it (the occlusion effect — which is why studio singers take one cup off). Bone conduction gives the *target* without blocking *self-hearing*. This makes bone conduction a **falsifiable claim**, not a gimmick (see §6 evaluation).
- **Real-time-on-CPU + the live-vs-render tradeoff** is itself an honest engineering framing worth writing up, not a limitation to hide (§7).

## 4. The two tracks (both, unified)

### Track A — Software: the harmony ear-trainer *(carries research credibility)*
- Built on the existing engine: **record → SATB stems** already exist; add **solo / mute (+ loop)** playback so a learner can isolate one part.
- **Sing-along pitch feedback using the engine's existing `f0 / midi / cents` telemetry** → "target A4, you are 30 cents flat." This upgrades a practice track into a *feedback* ear-trainer, reusing what's already there.
- The **recorded stem is a frozen, independent target** → it solves the circularity of live auto-derived harmony (which merely shadows the singer) **without needing MIDI** for the learning use.
- Two practice sub-modes: **match your target part** (easier) vs **hold against the other parts** (the real choir skill).

### Track B — Hardware: embodied delivery *(carries the novel artifact)*
- **Learn-mode = bone conduction, open ear.** Independent target into the head; own voice stays audible. (For learning, pitch fidelity is what matters; bone conduction handles the singing f0 range fine even if timbre is dull.)
- **Perform-mode = external radiation (DAEX mouth-rig).** The choir physically emanates from the singer's own mouth/body, co-located and blended with the live voice → the choir reads as coming from one person.
  - **Correction on record:** the DAEX-mouth is an **embodiment / spatial-source** device, **not** a formant filter. Genuine mouth-formant-filtering needs a talkbox tube *into* the mouth, and software (Beatrice) already owns timbre controllably — so timbre stays a software job. Test the mouth-rig for *"does it read as coming from me / blend with my voice,"* not for timbre change.
- On-body, **"You" is OFF** (the real acoustic voice is the unison) → fewer inference passes → more real-time headroom for the *added* harmony.

## 5. What this is NOT (scope discipline)

- **Not** an engine swap before the viva (RAVE / DDSP / ACE-Step: no pitch control, GPU + weeks of training, we are CPU/Mac).
- **Not** a self-trained-model quality chase (25 min / 10k steps is structurally too little — write it up as a finding).
- **Not** a "body as formant filter" claim (retracted — see §4 Track B).
- For Media Lab/PhD, **coherence > feature count**: each track nails its ONE core thing rather than both half-finished.

## 6. Evaluation (this is the part that makes it research)

Per supervisor advice (Yee-King, 2026-06-23) — now core, not polish:

1. **Learning comparison study:** *bone conduction (open ear) vs closed headphones vs room speaker* — which lets a singer learn / hold a harmony part fastest and most accurately? Quantifiable via the engine's `cents` error and time-held-on-pitch.
2. **Synthesis-quality baseline ladder:** ① plain pitch-shift(+EQ) vocoder baseline → ② Beatrice → (optional ③ a pre-trained RAVE/DDSP via Neutone). Produce clean A/B(/C) clips from the render pipeline; explain *why* the baseline is not good enough.
3. **Listening-test metrics:** *sounds like different people · sounds artificial · sounds in tune.*

## 7. Honest technical framing — live vs render

- Each converted part = +1 inference pass; on CPU **live ≈ ≤ 3 parts**, **full simultaneous SATB = offline render**. This is why Record→Play / render exists.
- The wearable claws some of this back: **"You" off on-body** → only the harmony passes run.
- Don't overclaim "full SATB live." The live-vs-render gap is a *legitimate framing of the instrument*, and a real engineering contribution to document.

## 8. Positioning

Imogen Heap (gloves / vocal performance), Holly Herndon (*Spawn*), the vocoder & Auto-Tune-as-instrument lineage, Jacob Collier's harmoniser, talkbox (and why we are *not* that), *Singing Knit* (biosignal-augmented vocal performance, MPI), Rebecca Horn (body-extension sculpture).

## 9. Future work = the PhD vision

- **Biosignal / EMG-driven harmony** — the body *conducts* the choir (laryngeal EMG, breath, jaw), instead of fixed diatonic intervals or MIDI chords. (Precedent: *Singing Knit*; EMG biofeedback for singers.)
- **Robust bone conduction / intraoral** embodied delivery; full wearable form (the medical-brace / Rebecca-Horn aesthetic from the proposal).
- Multi-voice on-body, simultaneous, in real time.

## 10. Sequencing (two deadlines)

| Window | Goal |
|---|---|
| **Now → viva (8/26)** | Track A ear-trainer solid · Track B hardware proof-of-concept working · **a pilot study** → a solid floor. |
| **Viva → application (Media Lab ~Dec)** | Fuller evaluation study · a strong **demo video** (learn-mode + perform-mode) · research-contribution statement · positioning/citations · **future-work = PhD proposal** · hardware polish. |

## 11. Application deliverables (what every step should feed)

1. A **demo video** — one segment learn-mode, one segment perform-mode.
2. A one-paragraph **research-contribution statement** (= §1–§3).
3. An **evaluation result** (= §6 study data).
4. **Positioning / citations** (= §8).
5. **Future-work / PhD vision** (= §9).

---

*This doc supersedes the framing in `HARDWARE_ROADMAP.md`: bone conduction returns to the core (learn-mode), DAEX is reframed as an embodied perform-mode source (not formant filter), and the haptic-motor path remains the viva-safe fallback if decoupling/feedback cannot be tamed.*
