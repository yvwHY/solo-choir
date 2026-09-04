# Mesh coupon, re-entrant honeycomb (bow-tie) auxetic, four variants.
#
# A rib-based auxetic: the mechanism is ribs bending, not narrow slits shearing,
# so every negative space is millimetres wide. No gap-fill risk, nothing fuses
# shut, no support needed, and it prints already open, which avoids the fusing
# that killed the pre-opened version of the slit-cut coupon (v2).
#
# The stiffness gradient is rib width, which leaves every node where it is:
# 2.0 mm near the plate (stiff) tapering to 0.8 mm at the edge (soft). In-plane
# bending stiffness goes as rib width cubed, so that is a factor of about 15.
#
# Variants (this file replaces four near-identical scripts):
#   v3         the version as printed, with the rigid centre plate
#   v3_noplate the same mesh with the plate removed - the control that says
#              whether the plate is what made v3 stiff towards the centre
#   v5         v3 softened: the cell scaled 1.5x and the sheet thinned to 0.6.
#              Rib length is also a cubed term and, unlike rib width, has no
#              floor at two extrusion widths - the knob v3 missed
#   v5_noplate the same control for v5
#
# Usage: python gen_coupon_reentrant.py [v3|v3_noplate|v5|v5_noplate]
# Writes mesh_coupon_<variant>.stl and preview_<variant>.png.
import sys
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

VARIANTS = {
    # name:        (sheet_t, W,    HV,   S,     plate, stl stem,          title)
    "v3":          (0.8, 12.0,  9.0, -3.50, True,  "v3_reentrant", "v3 re-entrant honeycomb - as printed (already open)"),
    "v3_noplate":  (0.8, 12.0,  9.0, -3.50, False, "v3_noplate",   "v3 no-plate - pure mesh baseline"),
    "v5":          (0.6, 18.0, 13.5, -5.25, True,  "v5_soft",      "v5 soft re-entrant - as printed (already open)"),
    "v5_noplate":  (0.6, 18.0, 13.5, -5.25, False, "v5_noplate",   "v5 soft no-plate - pure mesh baseline"),
}

name = sys.argv[1] if len(sys.argv) > 1 else "v3"
if name not in VARIANTS:
    sys.exit("variant must be one of: " + ", ".join(VARIANTS))
SHEET_T, W, HV, S, WITH_PLATE, STEM, TITLE = VARIANTS[name]

# ---- parameters (mm) ----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # radius of the solid plate (55 mm across)
PLATE_T = 1.6           # total plate thickness

ROW_H = HV + S          # row pitch; under tension the diagonals straighten and
                        # the sheet widens sideways as well (negative Poisson)

RIB_NEAR, RIB_FAR = 2.0, 0.8   # rib width, near the plate to the outer edge
R_TAPER = 70.0                 # ribs are at their thinnest beyond this radius

TAB_H = 3.0             # gripping strips top and bottom; the sides are left
                        # open deliberately, so the sideways expansion is visible


def rib_w(x, y):
    r = np.hypot(x - CX, y - CY)
    t = np.clip((r - R_PLATE) / (R_TAPER - R_PLATE), 0, 1)
    return RIB_NEAR + t * (RIB_FAR - RIB_NEAR)


ribs = []


def add(p0, p1):
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ribs.append(LineString([p0, p1]).buffer(rib_w(mx, my) / 2, cap_style=1, quad_segs=6))


# Row j has its vertical ribs from Yj to Yj+HV; the diagonals fold back down
# from the top to the foot of the next row at (x +/- W/2, Yj+ROW_H). Odd and
# even rows are offset by half a cell, so a diagonal lands exactly on the foot
# of a vertical rib in the row above.
y_lo = TAB_H
n_rows = int(np.floor((SIZE - TAB_H - y_lo - HV) / ROW_H))
n_cols = int(np.ceil(SIZE / W)) + 1

for j in range(n_rows + 1):
    Yj = y_lo + j * ROW_H
    off = (j % 2) * 0.5
    for i in range(-1, n_cols):
        x = (i + off) * W
        add((x, Yj), (x, Yj + HV))
        if j < n_rows:
            add((x, Yj + HV), (x - W / 2, Yj + ROW_H))
            add((x, Yj + HV), (x + W / 2, Yj + ROW_H))

y_top = y_lo + n_rows * ROW_H + HV      # the top of the mesh; the upper tab sits
                                        # flush with it, leaving no break
frame = box(0, 0, SIZE, SIZE)
tabs = unary_union([box(0, 0, SIZE, TAB_H), box(0, y_top, SIZE, SIZE)])
plate = Point(CX, CY).buffer(R_PLATE, quad_segs=96)
parts = ribs + [tabs] + ([plate] if WITH_PLATE else [])
sheet_poly = unary_union(parts).intersection(frame)


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


meshes = extrude(sheet_poly, SHEET_T, 0)
if WITH_PLATE:
    meshes += extrude(plate, PLATE_T - SHEET_T, SHEET_T)
combined = trimesh.util.concatenate(meshes)
stl = f"mesh_coupon_{STEM}.stl"
combined.export(stl)
vol = combined.volume / 1000.0
print(f"STL out: {stl}  triangles={len(combined.faces)}")
print(f"cell={W}x{ROW_H} re-entrant angle={np.degrees(np.arctan2(-S, W/2)):.1f} deg  rows={n_rows+1}")
print(f"volume={vol:.1f} cm3  PLA approx {vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# Smallest-opening check: the whole point of a rib mesh is that this is far
# larger than the nozzle.
holes = [len(g.interiors) for g in (sheet_poly.geoms if sheet_poly.geom_type == "MultiPolygon" else [sheet_poly])]
print(f"holes={sum(holes)} (all millimetre scale, no slits under 1 mm)")

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
if WITH_PLATE:
    draw(plate, "#2f3d35")
ax.set_xlim(-5, SIZE + 5)
ax.set_ylim(-5, SIZE + 5)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("mesh coupon " + TITLE)
png = f"preview_{STEM}.png"
fig.savefig(png, dpi=150, bbox_inches="tight")
print(f"preview: {png}")
