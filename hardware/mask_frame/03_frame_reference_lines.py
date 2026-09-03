"""
Solo Choir — mask frame, STEP 3: frame reference lines in front of the face.

Run inside Blender AFTER STEP 1 (needs 'Head'); STEP 2 markers optional. Pure
on-screen reference for the hand-bent wire+fibre frame. Draws the front-of-face
RIGID island — the part that holds the DAEX cup off the mouth — as tube curves
standing a fixed STANDOFF in front of the face surface:

  - rim          : a closed loop around the face perimeter (the mask's outer edge)
  - centre profile : forehead -> chin down the midline (vertical cup support)
  - mouth cross    : cheek -> cheek at mouth height (horizontal cup support)

The rim is the outer frame; the centre + cross are struts that hold the DAEX cup
at their crossing (the mouth). This island stays mechanically SEPARATE from the
throat-mic / bone-conduction mounts (decoupling — see HARDWARE_ROADMAP.md §"one
shell, two isolated islands"): the bone transducers ride an elastic band, not this
frame.

The curves are plain movable objects in a 'Frame' collection — reshape freely.
"""
import bpy, math, mathutils

FRAME_COLL = "Frame"
STANDOFF   = 18.0    # mm the wire sits IN FRONT of the face surface
TUBE       = 2.0     # mm wire radius (visual)

head = bpy.data.objects.get("Head")
if head is None:
    raise RuntimeError("No 'Head' — run 01_import_head.py first.")

# --- fresh Frame collection (re-run safe) ---
old = bpy.data.collections.get(FRAME_COLL)
if old:
    for o in list(old.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(old)
coll = bpy.data.collections.new(FRAME_COLL)
bpy.context.scene.collection.children.link(coll)

# --- geometry: neck + head region (same method as STEP 2) ---
co = [v.co.copy() for v in head.data.vertices]
zmin = min(c.z for c in co); zmax = max(c.z for c in co); H = zmax - zmin
neck_z, neck_w = zmax, 1e9
Nb = 24
for i in range(Nb):
    z0 = zmin + H * i / Nb; z1 = zmin + H * (i + 1) / Nb
    band = [c for c in co if z0 <= c.z < z1]
    if not band or (z0 + z1) / 2 < zmin + 0.45 * H:
        continue
    w = max(c.x for c in band) - min(c.x for c in band)
    if w < neck_w:
        neck_w, neck_z = w, (z0 + z1) / 2
head_h = zmax - neck_z

# --- surface samplers ---
def front_at(x, z, xw=16.0, zw=12.0):
    """Frontmost (min-Y) vertex near (x, z) on the face; None if nothing nearby."""
    b = [c for c in co if abs(c.x - x) <= xw and abs(c.z - z) <= zw]
    return min(b, key=lambda c: c.y) if b else None

def curve_from(name, pts, rgba, closed=False):
    cu = bpy.data.curves.new(name, type='CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = TUBE; cu.bevel_resolution = 3
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    sp.use_cyclic_u = closed
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    ob = bpy.data.objects.new(name, cu)
    m = bpy.data.materials.get(name + "_mat") or bpy.data.materials.new(name + "_mat")
    m.diffuse_color = rgba; cu.materials.append(m)
    coll.objects.link(ob)
    return ob

# heights: chin just above the neck, forehead high on the head, mouth ~lower face
chin_z     = neck_z + 0.05 * head_h
forehead_z = neck_z + 0.85 * head_h
mouth_z    = neck_z + 0.28 * head_h

# --- rim: closed loop around the face perimeter (the mask's outer edge) ---
# Sweep an ellipse in the face plane (X across, Z up) and snap each point to the
# face surface + STANDOFF, giving an oval rim that hugs the face front-sides.
z_mid = (chin_z + forehead_z) / 2
a = 62.0                              # rim half-width  (sits inside the ~72 mm cheek edge)
b = (forehead_z - chin_z) / 2 * 0.95  # rim half-height
rim = []
segs = 40
for i in range(segs):
    th = 2 * math.pi * i / segs
    x = a * math.cos(th); z = z_mid + b * math.sin(th)
    p = front_at(x, z, xw=22.0, zw=18.0)
    if p:
        rim.append((x, p.y - STANDOFF, z))   # clean oval in X-Z, at the face depth
if len(rim) >= 3:
    curve_from("Frame_rim", rim, (1.0, 0.55, 0.0, 1), closed=True)

# --- centre profile: midline forehead -> chin, pushed STANDOFF forward (-Y) ---
prof = []
steps = 12
for i in range(steps + 1):
    z = chin_z + (forehead_z - chin_z) * i / steps
    p = front_at(0.0, z)
    if p:
        prof.append((0.0, p.y - STANDOFF, z))
if len(prof) >= 2:
    curve_from("Frame_centre", prof, (1.0, 0.85, 0.10, 1))

# --- mouth cross: cheek -> cheek at mouth height, pushed forward ---
cross = []
face_half = 70.0     # sample this far each side of the midline
steps = 10
for i in range(steps + 1):
    x = -face_half + 2 * face_half * i / steps
    p = front_at(x, mouth_z)
    if p:
        cross.append((p.x, p.y - STANDOFF, mouth_z))
if len(cross) >= 2:
    curve_from("Frame_mouthcross", cross, (1.0, 0.85, 0.10, 1))

print("=== STEP 3 done ===")
print(f"  standoff={STANDOFF:.0f} mm  tube r={TUBE:.0f} mm")
print(f"  chin z={chin_z:.0f}  mouth z={mouth_z:.0f}  forehead z={forehead_z:.0f}")
print(f"  rim pts={len(rim)} (a={a:.0f} b={b:.0f})  centre pts={len(prof)}  cross pts={len(cross)}")
print("  Frame_rim + Frame_centre + Frame_mouthcross in the 'Frame' collection (movable).")
