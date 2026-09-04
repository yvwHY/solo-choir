# Mesh coupon v4: staggered straight-cut kirigami with a rigid plate.
# It keeps v2's language, printing nearly solid and opening into lace under
# tension, with an order of magnitude fewer slits:
#   v2 rotating squares: 650 short slits, about 8 m of internal wall.
#   v4: about 140 long ones.
# The slits can therefore widen to 0.8 mm, two extrusion widths, which the slicer
# always leaves open, and the open area still comes out lower than v2's, so it
# reads as more solid.
# The gradient is bridge length. The slit period is fixed and the slit length
# varies with radius, so slit centres never move and the pattern is not
# distorted: bridges are 6.0 mm near the plate, stiff, and 2.0 mm at the edge.
# Writes mesh_coupon_v4_kirigami.stl and preview_v4_kirigami.png.
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- parameters (mm) ----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # radius of the solid plate (55 mm across)
SHEET_T = 0.8           # sheet thickness, four layers at 0.2
PLATE_T = 1.6           # total plate thickness

CUT_P = 20.0            # slit period in x; fixed, so slit centres stay put and
                        # the gradient does not distort the pattern
ROW_P = 8.0             # row pitch, which is the strip width and sets the
                        # scale of the buckling
SLIT_W = 0.8            # slit width: two extrusion widths, always left open by a
                        # 0.4 nozzle, where v2's 0.6 was borderline
BRIDGE_NEAR, BRIDGE_FAR = 6.0, 2.0   # bridge length, near the plate to the edge
R_TAPER = 70.0          # bridges are at their shortest, and softest, beyond this

def bridge_len(x, y):
    r = np.hypot(x - CX, y - CY)
    t = np.clip((r - R_PLATE) / (R_TAPER - R_PLATE), 0, 1)
    return BRIDGE_NEAR + t * (BRIDGE_FAR - BRIDGE_NEAR)

# Row j's slits run along x, odd and even rows offset by half a period, so under
# tension the bridges twist and the strips buckle out of plane into lace.
slits = []
n_rows = int(np.floor((SIZE - ROW_P) / ROW_P)) + 1
n_cuts = int(np.ceil(SIZE / CUT_P)) + 2

for j in range(n_rows):
    y = ROW_P / 2 + j * ROW_P
    off = (j % 2) * 0.5
    for i in range(-1, n_cuts):
        xc = (i + 0.5 + off) * CUT_P
        L = CUT_P - bridge_len(xc, y)
        seg = LineString([(xc - L / 2, y), (xc + L / 2, y)])
        # A round cap relieves stress at the slit end so PLA does not tear from
        # there; v2 used a flat cap, whose right angle concentrates stress.
        slits.append(seg.buffer(SLIT_W / 2, cap_style=1, quad_segs=8))

frame = box(0, 0, SIZE, SIZE)
plate = Point(CX, CY).buffer(R_PLATE, quad_segs=96)
sheet_poly = frame.difference(unary_union(slits)).union(plate).intersection(frame)

def extrude(poly, h, z0):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    out = []
    for g in geoms:
        if g.area < 1.0:
            continue
        m = trimesh.creation.extrude_polygon(g, height=h)
        m.apply_translation([0, 0, z0])
        out.append(m)
    return out

meshes = []
meshes += extrude(sheet_poly, SHEET_T, 0)
meshes += extrude(plate, PLATE_T - SHEET_T, SHEET_T)
combined = trimesh.util.concatenate(meshes)
combined.export("mesh_coupon_v4_kirigami.stl")
vol = combined.volume / 1000.0
open_ratio = 1 - sheet_poly.area / (SIZE * SIZE)
print(f"STL out: mesh_coupon_v4_kirigami.stl  triangles={len(combined.faces)}")
print(f"rows={n_rows}  slits={len(slits)}  slit length {CUT_P-BRIDGE_NEAR:.0f} near the plate to {CUT_P-BRIDGE_FAR:.0f} at the edge")
print(f"open area={open_ratio*100:.1f}% (the v2 slit sheet is about 11%)")
print(f"volume={vol:.1f} cm³  PLA≈{vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# ---- preview ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

fig, ax = plt.subplots(figsize=(9, 9))
def draw(poly, fc):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        if g.area < 1.0:
            continue
        ax.add_patch(MplPoly(np.array(g.exterior.coords), fc=fc, ec="none"))
        for hole in g.interiors:
            ax.add_patch(MplPoly(np.array(hole.coords), fc="white", ec="none"))
draw(sheet_poly, "#5a6e62")
draw(plate, "#2f3d35")
ax.set_xlim(-5, SIZE + 5); ax.set_ylim(-5, SIZE + 5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("mesh coupon v4 staggered kirigami — rest state (stretch to open)")
fig.savefig("preview_v4_kirigami.png", dpi=150, bbox_inches="tight")
print("preview: preview_v4_kirigami.png")
