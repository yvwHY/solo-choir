# Models — what the instrument loads, and why none of it is here

This directory is empty in git. Everything the instrument loads at run time is
listed below with its source and licence. Point [`../config.py`](../config.py) at
a copy either by placing it here under the names in the first column, or by
setting the matching environment variable.

```bash
export SOLO_CHOIR_MODELS=/Volumes/Work/solo-choir-models
```

## Why nothing is committed

| Reason | Applies to |
|---|---|
| Too large for git (207 MB each, GitHub's file limit is 100 MB) | the five `reflow-*` sampled-choir models |
| Non-commercial licence, redistribution not granted | the Beatrice engine build and its JVS-derived models |
| Non-commercial share-alike corpus licence | every model trained on M4Singer |
| Personal data — an identifiable individual's voice | `combsub-girl` |

The one class of weights that **is** committed is `harmony/checkpoints/` — the
harmony-brain transformers. They were trained by this project on public-domain
chorale scores and contain no voice data.

## Layout `config.py` expects

```
models/
  ddsp-svc/                 DDSP-SVC checkout + venv; renders the Respond voices
    exp/combsub-m4-bass1/model_11000.pt
    exp/combsub-m4-sop3/model_10000.pt
    exp/combsub-girl/model_30000.pt
  ddsp-svc-6x/              DDSP-SVC 6x checkout + venv; renders the sampled choir
    exp/reflow-bass1/model_32000.pt
    exp/reflow-alto3/model_40000.pt
    exp/reflow-sop3/model_20000.pt
    exp/reflow-male8/model_40000.pt
    exp/reflow-fem8/model_40000.pt
  beatrice_engine_rc0/      Beatrice v2 engine build
  beatrice/                 paraphernalia_* voice model directories
    paraphernalia_data_new25_2k
    paraphernalia_data_satb2
    paraphernalia_data_00003000
  data/                     optional: training corpora and vowel source clips
```

Each directory has its own environment variable if you keep it elsewhere:
`SOLO_CHOIR_DDSP`, `SOLO_CHOIR_DDSP6X`, `SOLO_CHOIR_BEATRICE_ENGINE`,
`SOLO_CHOIR_BEATRICE_MODELS`, `SOLO_CHOIR_DATA`.

## The models, one by one

### Trained on M4Singer (CC BY-NC-SA)

M4Singer, Zhang et al., *NeurIPS 2022 Datasets and Benchmarks*. Non-commercial,
attribution, share-alike. These models are derivative works of that corpus and
carry the same restriction.

| Model | Mode | Part | Original singer |
|---|---|---|---|
| `exp/combsub-m4-bass1/model_11000.pt` | Respond | lower | Bass-1 |
| `exp/combsub-m4-sop3/model_10000.pt` | Respond | upper | Soprano-3 |
| `exp/reflow-bass1/model_32000.pt` | Live | bass | Bass-1 |
| `exp/reflow-alto3/model_40000.pt` spk2 | Live | alto | Alto-6 |
| `exp/reflow-sop3/model_20000.pt` | Live | soprano | Soprano |
| `exp/reflow-male8/…` spk7 | Live | tenor lead | Tenor-5 |

`reflow-male8` and `reflow-fem8` were trained on eight singers each. The frozen
performance configuration loads only the tenor voice above; the rest are not
heard (`--mem-real 0`).

### The author's own voice

| Model | Mode | Part |
|---|---|---|
| `paraphernalia_data_new25_2k` | Live · neural | lower |

Trained on 21 minutes of the author singing F2–C5, recorded 2026-06-20.

### A friend's voice — consent on file

| Model | Mode | Part |
|---|---|---|
| `exp/combsub-girl/model_30000.pt` | Respond | upper |

Recorded in person by the singer, who consented to academic use (MA project,
viva, exhibition, portfolio) and asked to remain anonymous. Commercial use is
not covered. Full record in [`../docs/VOICE_CREDITS.md`](../docs/VOICE_CREDITS.md).
This is the reason the model file is not published anywhere.

### Beatrice engine and its models

| Item | Source |
|---|---|
| `beatrice_engine_rc0` | Beatrice v2 engine build, ported into an editable Python harness |
| `paraphernalia_data_satb2` | Beatrice-family model; base corpus JVS + JVS-MuSiC (Takamichi et al.), academic / non-commercial |

Whether `satb2` is a shipped model or a derivative trained here is still being
confirmed; see the open item at the end of `docs/VOICE_CREDITS.md`.

### Derived data, not shipped and not needed

`harmony/scratchpad/bank_*` — the sampled-choir render cache. It is rebuilt from
the `reflow-*` models on first run, keyed by a hash of the render settings, so
there is nothing to copy and nothing to keep in sync.
