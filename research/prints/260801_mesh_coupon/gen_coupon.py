# mesh coupon v1 — 網布×剛性板 漸變實驗片
# 概念驗證件：中央實心板（exciter 貼位）→ 膜填充過渡帶 → 開放網格，一次平印
# 產出：mesh_coupon_v1.stl + preview.png
# 跑法：scratchpad meshenv 的 python（shapely+trimesh），見 README
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- 參數（全部 mm）----
SIZE = 150.0            # 片外形（Prusa Mini 床 180 內）
CX = CY = SIZE / 2
R_PLATE = 27.5          # 板半徑（⌀55：35mm exciter + 468MP 膠圈有餘裕）
R_MEMBRANE = 62.0       # 膜漸變帶外緣（板緣 100% 封膜 → 此半徑 0%，逐孔機率決定）
LATTICE_T = 0.8         # 網格厚（4 層 @0.2）
MEMBRANE_T = 0.4        # 膜厚（2 層）
PLATE_T = 1.6           # 板總厚（上層再加 0.8）
SPACING = 8.0           # 格線間距（三方向 → 三角格）
SEG = 3.0               # 變寬取樣段長
HW_NEAR, HW_FAR = 0.70, 0.40   # 桿半寬：板附近 1.4mm → 外緣 0.8mm
R_TAPER0, R_TAPER1 = 35.0, 100.0
RIM_W = 1.5             # 外框寬（收邊，避免斷桿毛邊）

def half_width(r):
    t = np.clip((r - R_TAPER0) / (R_TAPER1 - R_TAPER0), 0, 1)
    return HW_NEAR + t * (HW_FAR - HW_NEAR)

# ---- 三方向格線 → 逐段變寬 buffer ----
diag = SIZE * 1.5
strut_polys = []
for ang in (0, 60, 120):
    th = np.radians(ang)
    d = np.array([np.cos(th), np.sin(th)])      # 線方向
    n = np.array([-np.sin(th), np.cos(th)])     # 法向（排線方向）
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
            if r > SIZE:            # 遠離片外的段直接跳過
                continue
            strut_polys.append(LineString([p0, p1]).buffer(float(half_width(r)), cap_style=2))

sq = box(0, 0, SIZE, SIZE)
plate = Point(CX, CY).buffer(R_PLATE, quad_segs=96)
rim = sq.boundary.buffer(RIM_W)
base_poly = unary_union(strut_polys + [plate, rim]).intersection(sq)

# ---- 膜填充帶：逐孔機率漸變（板緣 100% 封膜 → R_MEMBRANE 處 0%），決定性偽隨機 ----
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
        if g.area < 1.0:            # 濾掉碎片
            continue
        m = trimesh.creation.extrude_polygon(g, height=h)
        m.apply_translation([0, 0, z0])
        out.append(m)
    return out

meshes = []
meshes += extrude(base_poly, LATTICE_T, 0)                    # 網格＋板下層＋外框 z0–0.8
meshes += extrude(membrane_poly, MEMBRANE_T, 0)               # 膜 z0–0.4
meshes += extrude(plate.intersection(sq), PLATE_T - LATTICE_T, LATTICE_T)  # 板上層 z0.8–1.6
combined = trimesh.util.concatenate(meshes)
combined.export("mesh_coupon_v1.stl")

vol = combined.volume / 1000.0  # cm³
print(f"STL out: mesh_coupon_v1.stl  triangles={len(combined.faces)}")
print(f"volume={vol:.1f} cm³  PLA≈{vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# ---- 預覽圖 ----
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
draw(base_poly, "#5a6e62", 1.0)       # 網＝中
draw(membrane_poly, "#b8c4bc", 1.0)   # 膜＝淺（畫在網之上才看得見）
draw(plate, "#2f3d35", 1.0)           # 板＝深
ax.set_xlim(-5, SIZE + 5); ax.set_ylim(-5, SIZE + 5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("mesh coupon v1 — top view (dark=plate 1.6 / mid=lattice 0.8 / light=membrane 0.4)")
fig.savefig("preview.png", dpi=150, bbox_inches="tight")
print("preview: preview.png")
