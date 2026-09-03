import glob, os
import pretty_midi

def note_name(p): return pretty_midi.note_number_to_name(int(p))

for f in sorted(glob.glob("cpdl_sample/*.mid") + glob.glob("cpdl_sample/*.MID")):
    print("="*70)
    print(os.path.basename(f))
    try:
        pm = pretty_midi.PrettyMIDI(f)
    except Exception as e:
        print("  PARSE ERROR:", e); continue
    print(f"  tracks: {len(pm.instruments)}  duration: {pm.get_end_time():.1f}s")
    for i, inst in enumerate(pm.instruments):
        if not inst.notes:
            print(f"  [{i}] name='{inst.name}' EMPTY"); continue
        ps = [n.pitch for n in inst.notes]
        # polyphony check: overlapping notes within track
        evts = sorted((n.start, n.end) for n in inst.notes)
        overlap = sum(1 for a, b in zip(evts, evts[1:]) if b[0] < a[1] - 1e-3)
        print(f"  [{i}] name='{inst.name}' prog={inst.program} notes={len(ps)} "
              f"range={note_name(min(ps))}-{note_name(max(ps))} overlaps={overlap}")
