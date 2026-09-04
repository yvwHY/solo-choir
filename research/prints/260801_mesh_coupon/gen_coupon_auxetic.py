# Mesh coupon v2: an auxetic rotating-squares slit sheet with a rigid plate.
# It prints nearly solid, with 0.6 mm slits, closed and with nothing overlapping;
# pulled, the squares rotate about their corner hinges and the lace appears.
# The gradient is hinge length: 2.4 mm near the plate, stiff, tapering to 0.8 mm
# at the edge, soft, so the outside opens first and the plate area does not move.
# Writes mesh_coupon_v2_auxetic.stl and preview_auxetic.png.
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- parameters (mm) ----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # radius of the solid plate (55 mm across)
PITCH = 8.0             # square cell pitch
SLIT_W = 0.6            # slit width; the smallest gap a 0.4 nozzle reliably leaves
HINGE_NEAR, HINGE_FAR = 2.4, 0.8   # hinge length, near the plate to the outer edge
R_TAPER1 = 70.0         # hinges are at their shortest beyond this radius
SHEET_T = 0.8           # sheet thickness, four layers at 0.2
PLATE_T = 1.6           # total plate thickness

def hinge_len(r):
    t = np.clip((r - R_PLATE) / (R_TAPER1 - R_PLATE), 0, 1)
    return HINGE_NEAR + t * (HINGE_FAR - HINGE_NEAR)

# Square (i,j) rotates in direction sigma = +1 when (i+j) is even, which decides
# which end of each slit keeps its hinge.
# For a horizontal pair, sigma = +1 puts the hinge at the bottom; for a vertical
# pair, sigma = +1 puts it at the right.
N = int(round(SIZE / PITCH))
slits = []
for i in range(N):
    for j in range(N):
        sigma = 1 if (i + j) % 2 == 0 else -1
        # the edge shared with the neighbour to the right
        if i + 1 < N:
            x = (i + 1) * PITCH
            y0, y1 = j * PITCH, (j + 1) * PITCH
            mid = np.array([x, (y0 + y1) / 2])
            hl = hinge_len(np.hypot(mid[0] - CX, mid[1] - CY))
            if sigma > 0:   # hinge at the bottom, so the slit runs down from the top
                seg = LineString([(x, y1), (x, y0 + hl)])
            else:           # hinge at the top
                seg = LineString([(x, y0), (x, y1 - hl)])
            slits.append(seg.buffer(SLIT_W / 2, cap_style=2))
        # the edge shared with the neighbour above
        if j + 1 < N:
            y = (j + 1) * PITCH
            x0, x1 = i * PITCH, (i + 1) * PITCH
            mid = np.array([(x0 + x1) / 2, y])
            hl = hinge_len(np.hypot(mid[0] - CX, mid[1] - CY))
            if sigma > 0:   # hinge at the right, so the slit runs right from the left
                seg = LineString([(x0, y), (x1 - hl, y)])
            else:           # hinge at the left
                seg = LineString([(x0 + hl, y), (x1, y)])
            slits.append(seg.buffer(SLIT_W / 2, cap_style=2))

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
combined.export("mesh_coupon_v2_auxetic.stl")
vol = combined.volume / 1000.0
print(f"STL out: mesh_coupon_v2_auxetic.stl  triangles={len(combined.faces)}")
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
ax.set_title("mesh coupon v2 auxetic slit sheet — rest state (stretch to open)")
fig.savefig("preview_auxetic.png", dpi=150, bbox_inches="tight")
print("preview: preview_auxetic.png")
