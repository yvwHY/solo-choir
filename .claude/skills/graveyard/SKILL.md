---
name: graveyard
description: This skill should be used BEFORE attempting any new approach to live audio streaming/mixing, pitch shifting, ensemble/choir sound, female timbre, transcription, per-voice timbre, or wearable hardware in this repo — and AFTER any approach fails or is abandoned ("這條路死了", "行不通", "we tried X and it didn't work", "放棄這個做法"). Routes to the dead-ends registry so disproven approaches are never re-run and new deaths are always recorded.
---

# Graveyard — check before building, register when dead

The knowledge lives in `docs/GRAVEYARD.md` (dead ends, with root causes) and `docs/FINDINGS.md` (validated recipes, with numbers). This skill is only the trigger.

## Before building
1. Read `docs/GRAVEYARD.md` and search it for the approach you (or the user) are about to try.
2. If it matches an entry: do NOT re-run it. Tell the user which entry it hits, why it died, and what the recorded *Instead* path is. The only legitimate way past an entry is to state explicitly which premise has changed (new API, new model, GPU) and cite evidence.
3. Check `FINDINGS.md` before re-deriving any number (glide rates, latency settings, decorrelation parameters, RTF budgets).

## When an approach dies
Append a GRAVEYARD entry **before moving on**, in the existing format:
- **Tried:** what, concretely.
- **Died because:** the root cause, not "didn't sound good".
- **Evidence:** date, commit hash, worklog, or measurement.
- **Instead:** the surviving path.

If a *belief* was overturned (not just an attempt), also update the corresponding FINDINGS entry. Never delete entries; supersede with a status note.
