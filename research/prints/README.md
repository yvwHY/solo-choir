# prints — the 3D-print batches

Five batches, printed on a Prusa MINI in PLA between 19 July and 3 August 2026,
for the worn frame and the table rig. Each directory holds the STL files that
were sent to the printer, the DXF for the laser-cut door panel, and the packing
script that arranges parts onto build plates.

The per-batch working notes were long and in Chinese; this file replaces them
with what the batches established. Measured tolerances are in
[`../../docs/FINDINGS.md`](../../docs/FINDINGS.md) (F14), the abandoned
approaches are in [`../../docs/GRAVEYARD.md`](../../docs/GRAVEYARD.md) (G25), and
the hardware narrative is in
[`../../docs/HARDWARE_ROADMAP.md`](../../docs/HARDWARE_ROADMAP.md).

**Not included:** the sliced G-code and the packed build-plate STLs. Both are
derived from the part STLs — `pack.py` rebuilds the plates and the slicer
rebuilds the G-code from them with the profiles in `slicer_profiles/`:

| Profile | Used for |
|---|---|
| `miniis_profile.ini` | the default, organic supports |
| `miniis_grid.ini` | anything with a vertical through hole, where organic supports grow up the hole |
| `miniis_shell_gridbrim.ini` | the electronics shell: grid supports plus a 5 mm brim |
| `miniis_lid_onmodel.ini` | the lid, whose bracket overhang is above the model, so support has to stand on the model |

---

## The part numbers

The frame parts kept the same numbering across every batch.

| Part | Body | What it is |
|---|---|---|
| WO-1 | `pivot_mount` | the collar's own-axis pivot seat |
| WO-2 | `band_clamp` | the headband clamp, at the crown where the two centre columns meet the band |
| WO-3 | `door_col_A` / `door_col_R`, `door_clip_*`, `door_panel_flat` | the door system, A the hinge side and R the latch side |
| WO-4 | `wo4_cup` / `wo4_cup_R` | the bone-conduction exciter cups, left and right |
| WO-5 | `wo5_shell_v10` + `wo5_lid_v10` + `jhook_west` | the electronics box and its hanging hooks |
| VU-1 | `vu1_panel` | the flat panel voice unit, clipped to the side rail |
| VU-2 | `vu2_cone`, `vu2_pyramid` | the table-standing voice units |

## The batches

| Batch | Date | What it was |
|---|---|---|
| `260719_prints` | 19 Jul | The full frame reprinted from final geometry: consolidated cable exit, brass channels opened up, M3 holes enlarged |
| `260721_prints` | 21 Jul | After the headband clamp was rebuilt with a C-shaped throat, the door columns rebuilt around grub screws, and every brass channel narrowed again |
| `260722_prints` | 22 Jul | After three failures that only appear on a real print (below); brass channels widened by 0.1, the door clip and its groove widened, support settings fixed |
| `260731_prints` | 31 Jul | A single-piece square-cone voice unit, the seventh iteration of that evening |
| `260801_mesh_coupon` | 1-3 Aug | Test squares for a printed mesh fabric with a rigid plate for the exciter, four pattern families |

## What the batches established

**A hole that a hand-bent rod has to pass through cannot be dimensioned like a
hole.** Batch 260719 opened the curved brass channels to 4.3 mm and the straight
holes to 3.8 mm so a rod would pass, and the fit came out about 1 mm loose.
Batch 260721 narrowed everything to a uniform 0.05 mm per side, which worked for
straight holes and failed for curved ones: the centre-column channel wanders
2.374 mm sideways over 20 mm of height, and no hand-bent rod matches that to
0.05 mm. Batch 260722 went to 0.15 mm per side on the curved channels. This is
G25, and the eventual answer was not a wider hole but a different shape — a
C-shaped throat that the rod clips into from the side, which is what the headband
seat, the panel clip and the door clip all use.

**Clamping a rod by squeezing a printed part does not work; pushing a screw
against it does.** Both the exciter cups and the door columns started as split
clamps with a screw pulling two halves together, and thin PLA either failed to
grip or split. Both were rebuilt around an M3 grub screw bearing directly on the
brass, 4.1 mm of thread tapped straight into PLA.

**Three failures that only a real print reveals** (all found in batch 260721 and
fixed in 260722):

- *The lid bracket printed in mid air.* The profile had `support_material = 1`
  and `support_material_buildplate_only = 1` together, and the overhang was above
  the model, so zero support was generated with no warning.
- *The door column's rail hole was packed solid with support.* Organic supports
  grow trunks up any vertical hole that reaches the bed: 467 of 469 toolpaths
  inside the 3.3 mm hole, 34 mm deep, impossible to remove. Grid supports leave
  none.
- *The sliding lid would not go in.* The rail slot measured 2.240 mm against a
  2.000 mm lid, that is 0.130 mm per side, which is nothing for FDM once the slot
  prints narrow and the lid prints thick. The lid was thinned to 1.500, giving
  0.370 mm per side.

**Support style follows the hole, not the batch.** Grid wins for a straight
vertical hole and organic wins for a curved one; a rule that one is always safer
is wrong.

**A print job that ends early can be a truncated file.** Three jobs failed on one
USB stick before the cause was found: the MINI runs its end-of-print routine and
reports "Print finished" as soon as it reaches the end of the file. The stick was
FAT32 and full of macOS `._` shadow files. Copy with `cp -X` and compare byte
counts.

**The single-piece cone came from giving up on ballast.** Seven iterations in one
evening went from a bayonet mount through a weighted cup to a plain truncated
pyramid: 65 mm square at the base, 104 mm at the mouth, 90 mm tall, walls 2 mm,
with a locating ring for the exciter and a 12 mm cable hole in the side. It is
glued rather than weighted, because the table is the ballast. Zero assembly and
no fit tolerances. Standing on its base it plays into the room; laid on a face it
plays into the table, which is the controllable version of the observation that
resting a unit on the table makes it much louder.

**The mesh coupons are a form study, not a design.** Four families were printed
flat, each 150 x 150 mm with a rigid 55 mm plate at the centre for the exciter
and a stiffness gradient towards the edge:

| Version | Family | Result |
|---|---|---|
| v1 | triangular mesh with a sealing film | comparison only |
| v2 | rotating squares, cut slits | printed but would not open; the cause was never isolated between the 0.6 mm slits fusing and the rigid centre plate locking a single-degree-of-freedom mechanism |
| v3 | re-entrant honeycomb, ribs | opens, but only slightly, and stiffer towards the centre |
| v4 | staggered kirigami, straight slits | wider slits with no fusing risk and a lower open area than v2, so it reads as more solid at rest |
| v5 | v3 with the cell scaled 1.5x and the sheet thinned to 0.6 | estimated at 0.23 of v3's stiffness; an estimate, not a measurement |

The four re-entrant variants (v3 and v5, each with and without the centre plate)
were four near-identical scripts and are now one, `gen_coupon_reentrant.py`,
selected by argument.

In-plane bending stiffness goes as thickness times rib width cubed over rib
length cubed. v3 only varied the rib width, whose floor is 0.8 mm, two extrusion
widths; the rib length is the same cubed term and has no floor, which is the
knob v3 missed. Below 0.4 mm sheet thickness the honest next step is TPU, which
on this printer means a 95A filament at 15-20 mm/s and a print time in hours
rather than minutes.

## Rerunning the generators

The mesh coupons are generated, not modelled, in their own virtual environment:

```
python3 -m venv meshenv
./meshenv/bin/pip install shapely trimesh mapbox_earcut numpy matplotlib
./meshenv/bin/python gen_coupon_reentrant.py v3      # or v3_noplate, v5, v5_noplate
```

`pack.py` in each frame batch arranges the parts onto plates: `python3 pack.py all`.
