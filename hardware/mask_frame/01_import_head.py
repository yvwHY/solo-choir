"""
Solo Choir — mask frame, STEP 1: import head.glb as a scaled reference.

Run inside Blender: Scripting tab → Open (this file) → ▶ Run.
Builds nothing yet — just brings the head in, scaled to your real face,
shown as wireframe so the frame can be built in front of it later.

Units: 1 Blender unit = 1 mm (build in mm, export STL → slicers read mm).
"""
import bpy, math, mathutils

# ---------------- PARAMETERS (tweak + re-run) ----------------
HEAD_GLB      = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260615/voice-changer/app/assets/head.glb"
FACE_WIDTH_MM = 145.0   # cheekbone-to-cheekbone; scales the whole head
HEAD_YAW_DEG  = 0.0     # spin face about vertical; set 180 if the face points AWAY from the marker
HEAD_TOP_FRAC = 0.45    # use the top 45% of the bust as "the head" when measuring width
                        # (the glb is a head+neck+shoulders bust — don't scale by shoulder width)
# -------------------------------------------------------------

# --- clear scene ---
# Use bpy.data.objects.remove (NOT select_all + delete): the previous run marks
# the Head hide_select=True, and a hidden/unselectable object survives
# select_all → delete. That stale Head then shadows the freshly imported one
# (get("Head") returns the leftover) and every re-run compounds on old geometry.
# remove() ignores selection/visibility, so this reliably wipes everything.
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
for m in list(bpy.data.meshes):
    if m.users == 0:
        bpy.data.meshes.remove(m)

# --- import + bake the mannequin bust to upright, real-ish units ---
# head.glb is a single Sketchfab mesh 'Cube.001_Material.002_0' (a smooth,
# faceless mannequin bust) parented under empties. The parent 'Cube.001' carries
# a 100x scale + the rotation that stands the bust upright in WORLD space (Z up,
# ~925 mm tall, ~15 m off the origin) — the mesh's own local coords are tiny and
# lying down. So bake the WORLD matrix into the vertices, unparent, reset the
# object to identity, then delete the leftover empties. Matrix math, not
# transform_apply/parent_clear (those need a 3D-view context and no-op over the
# MCP socket, leaving a dirty object transform).
bpy.ops.import_scene.gltf(filepath=HEAD_GLB)
meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    raise RuntimeError("head.glb: no mesh found")
head = meshes[0]
mw = head.matrix_world.copy()
head.parent = None
head.data.transform(mw)                 # world transform -> vertices (upright, ~925 mm tall)
head.matrix_basis = mathutils.Matrix()  # object back to identity at the origin
head.name = "Head"
for o in list(bpy.context.scene.objects):
    if o is not head:
        bpy.data.objects.remove(o, do_unlink=True)

# All bakes below use head.data.transform(matrix) rather than
# bpy.ops.object.transform_apply — the operator needs a 3D-view context and
# silently no-ops over the MCP socket, leaving a non-identity scale/offset on the
# object (looks right via matrix_world, but the object transform is dirty and
# bites later when building/measuring/exporting against it).

# --- scale by HEAD-region width (not shoulders) so face width = FACE_WIDTH_MM ---
co = [v.co.copy() for v in head.data.vertices]
zmin, zmax = min(c.z for c in co), max(c.z for c in co)
thresh = zmin + (1.0 - HEAD_TOP_FRAC) * (zmax - zmin)
head_xs = [c.x for c in co if c.z >= thresh]
head_w = max(head_xs) - min(head_xs)
head.data.transform(mathutils.Matrix.Scale(FACE_WIDTH_MM / head_w, 4))

# --- optional yaw about vertical (Z) ---
if HEAD_YAW_DEG:
    head.data.transform(mathutils.Matrix.Rotation(math.radians(HEAD_YAW_DEG), 4, 'Z'))

# --- center left-right (X) + front-back (Y); sit the base on the floor (Z=0) ---
# Baking matrix_world left the bust ~0.6 m below the origin (an artefact of the
# glb's world offset). Drop min-Z to 0 so it stands on the grid like a real bust.
co = [v.co.copy() for v in head.data.vertices]
xs = [c.x for c in co]; ys = [c.y for c in co]; zs = [c.z for c in co]
head.data.transform(mathutils.Matrix.Translation(
    (-(max(xs) + min(xs)) / 2, -(max(ys) + min(ys)) / 2, -min(zs))))

# --- show as wireframe reference, lock it so you don't grab it while building ---
head.display_type = 'WIRE'
head.hide_select = True

# --- FRONT marker: a cone on the -Y side, tip pointing INTO the face (+Y) ---
co = [v.co.copy() for v in head.data.vertices]
ys = [c.y for c in co]; zs = [c.z for c in co]
zc = (max(zs) + min(zs)) / 2
bpy.ops.mesh.primitive_cone_add(radius1=6, depth=18, location=(0, min(ys) - 25, zc))
marker = bpy.context.active_object
marker.name = "FRONT_marker"
marker.rotation_euler[0] = math.radians(-90)   # tip toward +Y (the face)

# --- report ---
d = head.dimensions
print("=== STEP 1 done ===")
print(f"  bust W×D×H = {d.x:.0f} × {d.y:.0f} × {d.z:.0f} mm   (face width set to {FACE_WIDTH_MM:.0f} mm)")
print("  Head is WIREFRAME. A FRONT_marker cone sits in front, tip pointing at the face.")
print("  CHECK 1: does the FACE point toward the cone tip? if the BACK of the head")
print("           faces it, set HEAD_YAW_DEG = 180 and re-run.")
print("  CHECK 2: does the head look about life-size / the right proportions?")
