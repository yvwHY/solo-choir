# VOICE CREDITS — whose voices sing inside this

Created 2026-08-19. **This file records only the models that really sound on the
shipping path**: the ones that will be heard at the viva, in the exhibition and in
the portfolio.
Training recipes and experimental records are in `TRAINING_NOTES.md`. This file
answers one question: **whose voice is this, and on what basis is it used.**

> The rule comes from `TRAINING_NOTES.md` section 8: "Record donor consent for any
> non-own voices."
> Where the submitted document's Ethical Considerations section says "credited in
> the repository", this is the file it means.

---

## 1. A friend, recorded in person (the only human voice not from a corpus)

| Field | Content |
|---|---|
| **Model** | `260724_ddsp_svc/exp/combsub-girl/model_30000.pt` (DDSP-SVC CombSub, 30000 steps) |
| **Where it is used** | **The upper part of the Respond mode** (`harmony/respond2.py:66`, `VOICES["upper"]`). It is mixed into the response that is played out (`respond2.py:502`) and is not a spare part |
| **Whose voice** | A friend, recorded in person. **Kept anonymous at their own request, and referred to publicly only as "a friend"** (confirmed 2026-08-19) |
| **Recording chain** | `260620/test/women/girl_clip1-4.m4a` (iOS Voice Memos, created 2026-06-19, 110 s) to `260618/girl_clip 5,7,8,11.wav` to the training set `260724_ddsp_svc/data_girl/train/audio/`, 245 clips |
| **Basis of use** | **Their own consent**, confirmed obtained on 2026-08-10 |
| **Where the consent record is held** | **A message conversation on the author's phone** (confirmed 2026-08-19). No personal-data file is kept in this repository; the conversation can be shown on request |
| **Scope** | Academic use: the MA work, the viva, the exhibition and the portfolio. Commercial use is not covered and would have to be asked separately |
| **Model file** | git-ignored and not redistributed with the repository (hard rule 3) |

**This item is complete as of 2026-08-19**: consent given, the record can be
shown, the person is anonymous publicly, and the credit is here.
Two things still to remember: (1) this is the only voice identifiable to an
individual, and under item 12 of the department's ethics checklist, visual or
vocal methods, it counts as personal data; (2) the consent record exists only in
a phone conversation, so **changing phone or clearing the messages would lose
it**, and a screenshot should be kept on the computer as well.

---

## 2. Voices trained on a published singing corpus

All from **M4Singer** (Zhang et al., NeurIPS 2022 Datasets and Benchmarks Track).
Licensed **CC BY-NC-SA**: non-commercial, attribution required, share-alike. The
licence was verified on 2026-08-04.

| Mode | Part | Model | Original singer (corpus index) |
|---|---|---|---|
| Respond | lower | `exp/combsub-m4-bass1/model_11000.pt` | Bass-1 |
| Respond | upper | `exp/combsub-m4-sop3/model_10000.pt` | Soprano-3 |
| Live (sampled choir) | bass | `exp/reflow-bass1/model_32000.pt` | Bass-1 |
| Live | alto | `exp/reflow-alto3/model_40000.pt` spk2 | Alto-5, 6 and 7 trained jointly; Alto-6 ships |
| Live | sop | `exp/reflow-sop3/model_20000.pt` | Soprano |
| Live | tenor lead voice | `exp/reflow-male8/…` spk7 | Tenor-5 (adopted in v42, 2026-08-16) |

**Trained but not shipped:** `reflow-male8` and `reflow-fem8`, eight singers each
from M4Singer, trained locally to 40000 steps. Blind listening on 2026-08-16
found that real timbres added nothing, so the frozen configuration sets
`--mem-real 0` and does not load them. Apart from the tenor lead voice in the
table above, no other singer from these two models sounds in performance.

---

## 3. The author's own voice

| Mode | Where | Model |
|---|---|---|
| Live, neural | lower part | `model/VC/dist/model_dir/1/model/paraphernalia_data_new25_2k`(F2-C5 material recorded 2026-06-20, about 21 minutes, 2000 steps) |
| Respond and Live | dry pass-through | no model; the raw microphone signal |

---

## 4. Models shipped with the Beatrice engine

| Where | Model | Source and licence |
|---|---|---|
| Live, neural, upper part | `paraphernalia_data_satb2` | A Beatrice-family model. The underlying corpus is the **JVS corpus and JVS-MuSiC** (Shinnosuke Takamichi et al.), which is **non-commercial and academic use only**. **Still to confirm:** whether satb2 ships with the engine or is derived from local training; this cell should be completed once known (`TRAINING_NOTES.md` section 8 records only the JVS base model) |

The engine binaries and the model files are all git-ignored and are not
redistributed with the repository.

---

## Outstanding items

- [x] The friend's name: **kept anonymous, referred to publicly as "a friend"** (2026-08-19)
- [x] Where the consent record is held: **a message conversation on the author's phone** (2026-08-19)
- [ ] The exact origin of satb2, whether shipped with the engine or derived from local training. This is a matter of citation accuracy, not of personal-data compliance
