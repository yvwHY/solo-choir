"""
Solo Choir — mask frame, STEP 2: mark where the hardware sits on the head.

Run inside Blender AFTER STEP 1 (needs the 'Head' object). Pure on-screen
reference: the mask itself is hand-built wire + fibre, so this just drops
labelled, movable markers on the bust surface for the transducers + throat mic —
a spatial guide for bending wire and mounting parts.

Positions are anatomical DEFAULTS on a faceless mannequin (derived from the
neck/head geometry, not hard-coded heights). Nudge each marker to taste; they are
plain movable objects in a 'Components' collection.

Components (see docs/HARDWARE_ROADMAP.md):
  - DAEX cup  — perform-mode radiator, stands OFF the mouth (front)
  - Bone L/R  — learn-mode BCE-1 bone-conduction, on the cheek in front of each ear
  - Throat mic — on the larynx (front of the neck)
"""
import bpy, mathutils

COMP_COLL = "Components"

head = bpy.data.objects.get("Head")
if head is None:
    raise RuntimeError("No 'Head' — run 01_import_head.py first.")

# --- fresh Components collection (re-run safe) ---
old = bpy.data.collections.get(COMP_COLL)
if old:
    for o in list(old.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.collections.remove(old)
coll = bpy.data.collections.new(COMP_COLL)
bpy.context.scene.collection.children.link(coll)

# --- geometry: find the neck (narrowest slice in the upper head) ---
co = [v.co.copy() for v in head.data.vertices]
zmin = min(c.z for c in co); zmax = max(c.z for c in co); H = zmax - zmin
N = 24
neck_z, neck_w = zmax, 1e9
for i in range(N):
    z0 = zmin + H * i / N; z1 = zmin + H * (i + 1) / N
    band = [c for c in co if z0 <= c.z < z1]
    if not band or (z0 + z1) / 2 < zmin + 0.45 * H:   # only look in the upper half
        continue
    w = max(c.x for c in band) - min(c.x for c in band)
    if w < neck_w:
        neck_w, neck_z = w, (z0 + z1) / 2
head_h = zmax - neck_z   # crown-above-neck height

# --- surface pickers: nearest actual vertex in a thin Z-band ---
def band_at(zc, half=14.0):
    b = [c for c in co if abs(c.z - zc) <= half]
    return b or co

def front_pt(zc):                       # frontmost point of the face (-Y), near mid-line
    b = [c for c in band_at(zc) if abs(c.x) < 25]
    return min(b or band_at(zc), key=lambda c: c.y)

def side_pt(zc, sign):                   # outer point on one side (+X right / -X left)
    b = band_at(zc)
    return (max if sign > 0 else min)(b, key=lambda c: c.x)

# --- material helper (viewport colour so markers read against the wireframe head) ---
def mat(name, rgba):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = rgba
    return m

# --- one marker = a small sphere on the surface (+ outward offset) with a text label ---
def marker(name, pos, rgba, label):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=8, location=pos)
    s = bpy.context.active_object
    s.name = name
    s.data.materials.append(mat(name + "_mat", rgba))
    for c in list(s.users_collection):
        c.objects.unlink(s)
    coll.objects.link(s)
    # text label, standing upright, facing the front (-Y)
    tc = bpy.data.curves.new(name + "_lbl", type='FONT'); tc.body = label; tc.size = 12
    t = bpy.data.objects.new(name + "_lbl", tc)
    t.location = (pos[0] + 10, pos[1], pos[2] + 12)
    t.rotation_euler = (mathutils.Matrix.Rotation(1.5708, 3, 'X')).to_euler()
    t.data.materials.append(mat(name + "_mat", rgba))
    coll.objects.link(t)
    return s

# --- place the components ---
mouth_z  = neck_z + 0.28 * head_h       # lower face
cheek_z  = neck_z + 0.60 * head_h       # cheekbone / ear height
throat_z = neck_z - 0.04 * H            # larynx, just below the neck's narrowest

p = front_pt(mouth_z);  marker("DAEX_cup",  (0.0, p.y - 25, mouth_z),  (1.0, 0.45, 0.05, 1), "DAEX cup")
p = side_pt(cheek_z, -1); marker("Bone_L",  (p.x - 5, p.y, cheek_z),   (0.10, 0.75, 0.95, 1), "Bone L")
p = side_pt(cheek_z, +1); marker("Bone_R",  (p.x + 5, p.y, cheek_z),   (0.10, 0.75, 0.95, 1), "Bone R")
p = front_pt(throat_z); marker("Throat_mic",(0.0, p.y - 8, throat_z),  (0.20, 0.90, 0.30, 1), "Throat mic")

print("=== STEP 2 done ===")
print(f"  neck z={neck_z:.0f} (width {neck_w:.0f})  head height above neck={head_h:.0f} mm")
print(f"  mouth z={mouth_z:.0f}  cheek/ear z={cheek_z:.0f}  throat z={throat_z:.0f}")
print("  Placed: DAEX_cup (mouth), Bone_L / Bone_R (cheeks), Throat_mic (larynx).")
print("  These are movable DEFAULTS in the 'Components' collection — drag to adjust.")
