# Mesh coupon v1: a test square for the gradient between printed mesh fabric and
# a rigid plate.
# A proof of concept: a solid centre plate where the exciter is glued, a
# transition band where the cells are filled with a thin membrane, then open
# mesh, all printed flat in one go.
# Writes mesh_coupon_v1.stl and preview.png.
# Run with the meshenv interpreter (shapely and trimesh); see the README.
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- parameters (all mm) ----
SIZE = 150.0            # outline of the square, inside the Prusa MINI's 180 bed
CX = CY = SIZE / 2
R_PLATE = 27.5          # plate radius, 55 mm across: room for a 35 mm exciter
                        # and its adhesive ring
R_MEMBRANE = 62.0       # outer edge of the membrane band: every cell filled at
                        # the plate edge, none here, decided cell by cell
LATTICE_T = 0.8         # mesh thickness, four layers at 0.2
MEMBRANE_T = 0.4        # membrane thickness, two layers
PLATE_T = 1.6           # total plate thickness, 0.8 more on top of the mesh
SPACING = 8.0           # line spacing; three directions give a triangular grid
SEG = 3.0               # segment length at which the width is resampled
HW_NEAR, HW_FAR = 0.70, 0.40   # half rib width: 1.4 mm near the plate, 0.8 at the edge
R_TAPER0, R_TAPER1 = 35.0, 100.0
RIM_W = 1.5             # rim width, to close the edge and avoid cut-off rib stubs

def half_width(r):
    t = np.clip((r - R_TAPER0) / (R_TAPER1 - R_TAPER0), 0, 1)
    return HW_NEAR + t * (HW_FAR - HW_NEAR)

# ---- three directions of grid lines, buffered segment by segment ----
diag = SIZE * 1.5
strut_polys = []
for ang in (0, 60, 120):
    th = np.radians(ang)
    d = np.array([np.cos(th), np.sin(th)])      # along the line
    n = np.array([-np.sin(th), np.cos(th)])     # normal, the direction lines are spaced along
    k0 = int(np.floor(-diag / SPACING))
    k1 = int(np.ceil(diag / SPACING))
    for k in range(k0, k1 + 1):
        base = np.array([CX, CY]) + n * (k * SPACING)
        ts = np.arange(-diag, diag + SEG, SEG)
        for i in range(len(ts) - 1):
            p0 = base + d * ts[i]
            p1 = base + d * ts[i + 1]
            mid = (p0 + p1) / 2
            r = np.hypot(mid[0] - CX, mid[1] - CY)
            if r > SIZE:            # skip segments well outside the square
                continue
            strut_polys.append(LineString([p0, p1]).buffer(float(half_width(r)), cap_style=2))

sq = box(0, 0, SIZE, SIZE)
plate = Point(CX, CY).buffer(R_PLATE, quad_segs=96)
rim = sq.boundary.buffer(RIM_W)
base_poly = unary_union(strut_polys + [plate, rim]).intersection(sq)

# ---- the membrane band: each cell is filled with a probability that falls from
# 1 at the plate edge to 0 at R_MEMBRANE; pseudo-random but deterministic, so the
# same parameters give the same sheet ----
def cell_hash(x, y):
    return (np.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1.0

fill_zone = Point(CX, CY).buffer(R_MEMBRANE, quad_segs=96).difference(
    Point(CX, CY).buffer(R_PLATE, quad_segs=96))
holes = fill_zone.difference(base_poly).intersection(sq)
kept = []
for g in (holes.geoms if holes.geom_type == "MultiPolygon" else [holes]):
    if g.area < 1.0:
        continue
    c = g.centroid
    r = np.hypot(c.x - CX, c.y - CY)
    prob = np.clip((R_MEMBRANE - r) / (R_MEMBRANE - R_PLATE), 0, 1)
    if cell_hash(c.x, c.y) < prob:
        kept.append(g)
membrane_poly = unary_union(kept) if kept else Point(CX, CY).buffer(0)

def extrude(poly, h, z0):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    out = []
    for g in geoms:
        if g.area < 1.0:            # drop slivers
            continue
        m = trimesh.creation.extrude_polygon(g, height=h)
        m.apply_translation([0, 0, z0])
        out.append(m)
    return out

meshes = []
meshes += extrude(base_poly, LATTICE_T, 0)                    # mesh, plate underside and rim, z 0-0.8
meshes += extrude(membrane_poly, MEMBRANE_T, 0)               # membrane, z 0-0.4
meshes += extrude(plate.intersection(sq), PLATE_T - LATTICE_T, LATTICE_T)  # plate top, z 0.8-1.6
combined = trimesh.util.concatenate(meshes)
combined.export("mesh_coupon_v1.stl")

vol = combined.volume / 1000.0  # cm³
print(f"STL out: mesh_coupon_v1.stl  triangles={len(combined.faces)}")
print(f"volume={vol:.1f} cm³  PLA≈{vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# ---- preview ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

fig, ax = plt.subplots(figsize=(9, 9))
def draw(poly, fc, alpha):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        if g.area < 1.0:
            continue
        ax.add_patch(MplPoly(np.array(g.exterior.coords), fc=fc, ec="none", alpha=alpha))
        for hole in g.interiors:
            ax.add_patch(MplPoly(np.array(hole.coords), fc="white", ec="none"))
draw(base_poly, "#5a6e62", 1.0)       # the mesh, mid tone
draw(membrane_poly, "#b8c4bc", 1.0)   # the membrane, light; drawn over the mesh to be visible
draw(plate, "#2f3d35", 1.0)           # the plate, dark
ax.set_xlim(-5, SIZE + 5); ax.set_ylim(-5, SIZE + 5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("mesh coupon v1 — top view (dark=plate 1.6 / mid=lattice 0.8 / light=membrane 0.4)")
fig.savefig("preview.png", dpi=150, bbox_inches="tight")
print("preview: preview.png")
