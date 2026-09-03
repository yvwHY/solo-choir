---
name: live-audio-debug
description: This skill should be used when any live-audio symptom appears in this repo — choppy/斷斷續續 output, scrape or instability on pitch changes, tremolo, comb/blur, input overflow, PortAudio errors (-10851), feedback/reverb, timbre collapse or all-voices-male after a device switch, crashes on model/speaker change, ticks, dead hardware channel toggles, "choir sounds like one person", or transcription that "looks wrong". Routes to the diagnosed symptom table before anyone touches buffers.
---

# Live-audio-debug — look it up before theorising

The symptom → root-cause table lives in `docs/DEBUG_PLAYBOOK.md`. This skill is only the trigger.

1. Read the playbook and match the symptom. Most live-audio symptoms here already have a *diagnosed* root cause — the project's recurring failure mode is inventing buffer/underrun theories for problems that were engine-state or control-state bugs.
2. If matched: apply/verify the recorded fix; do not open a new investigation.
3. If not matched: follow the playbook's method — **measure before theorising** (record stems and inspect samples; count underruns; reproduce offline via `--render`), change one variable at a time, then register the new diagnosis in the playbook (and in `GRAVEYARD.md` if a theory died along the way).
