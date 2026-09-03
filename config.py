"""config — every path that points outside this repository, in one place.

The instrument needs three things that are NOT in this repository, because
they are either too large for git or not ours to redistribute:

  * two DDSP-SVC checkouts (with their virtualenvs and trained voice models)
  * the Beatrice engine build
  * the Beatrice `paraphernalia_*` voice models

See `models/README.md` for what each one is, where it comes from, and under
what licence. Point this file at them either by putting them in `models/`
with the names below, or by setting the matching environment variable.

    export SOLO_CHOIR_MODELS=/Volumes/Work/solo-choir-models
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


# Where the non-redistributable material lives.
MODELS = _env("SOLO_CHOIR_MODELS", REPO / "models")

# DDSP-SVC checkouts. `DDSP` renders the Respond voices (combsub models);
# `DDSP6X` is the 6x fork that renders the sampled choir (reflow models).
DDSP = _env("SOLO_CHOIR_DDSP", MODELS / "ddsp-svc")
DDSP6X = _env("SOLO_CHOIR_DDSP6X", MODELS / "ddsp-svc-6x")

# Beatrice: the engine build and the directory holding paraphernalia_* models.
BEATRICE_ENGINE = _env("SOLO_CHOIR_BEATRICE_ENGINE", MODELS / "beatrice_engine_rc0")
BEATRICE_MODELS = _env("SOLO_CHOIR_BEATRICE_MODELS", MODELS / "beatrice")

# Interpreters. The UI shell and the audio engines live in different
# environments on purpose (see docs/SOLO_CHOIR_TECH_DOC.md); the shell spawns
# the engines as subprocesses, so both interpreters have to be nameable.
PY_APP = _env("SOLO_CHOIR_PY_APP", Path("/opt/anaconda3/envs/vcclient-dev/bin/python"))
PY_ENGINE = _env("SOLO_CHOIR_PY_ENGINE", DDSP6X / "venv" / "bin" / "python")
PY_RESPOND = _env("SOLO_CHOIR_PY_RESPOND", DDSP / "venv" / "bin" / "python")


def bootstrap() -> None:
    """Put the repository root on sys.path so `import config` works from
    inside app/, harmony/ and server/."""
    import sys
    root = str(REPO)
    if root not in sys.path:
        sys.path.insert(0, root)


# --- source material (training corpora, vowel clips) ------------------------
# Also not in git: singing corpora and the recorded clips the vowel layers are
# cut from. Only the training scripts and the optional `--vowels` layer bank
# need these; the frozen performance config does not.
DATA = _env("SOLO_CHOIR_DATA", MODELS / "data")
VOWEL_CLIPS = _env("SOLO_CHOIR_VOWEL_CLIPS", DATA / "0619_clean_clips")
TOKENS_V2 = _env("SOLO_CHOIR_TOKENS_V2", DATA / "tokens_v2")
TOKENS_V3 = _env("SOLO_CHOIR_TOKENS_V3", DATA / "tokens_v3")
CHORALE_NPZ = _env("SOLO_CHOIR_CHORALE_NPZ", DATA / "chorale_tokens.npz")
