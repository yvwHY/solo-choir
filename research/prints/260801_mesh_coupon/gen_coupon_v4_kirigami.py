# mesh coupon v4 — 交錯直縫 kirigami（staggered straight-cut）×剛性板
# 保留 v2 的「印時＝近實面、張力下浮現蕾絲」語言，但縫少一個量級：
#   v2 rotating-squares 650 條短縫（~8 m 內壁）→ v4 ~140 條長縫
# 縫因此可以放寬到 0.8（＝2 走線，slicer 必定留空），開孔率反而比 v2 更低＝更像實面
# 漸變＝橋接段長度（縫週期固定、縫長隨半徑變 → 縫中心不動）：板附近橋長 6.0（硬）→ 外緣 2.0（軟）
# 產出：mesh_coupon_v4_kirigami.stl + preview_v4_kirigami.png
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- 參數（mm）----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # 實心板半徑（⌀55）
SHEET_T = 0.8           # 片厚（4 層 @0.2）
PLATE_T = 1.6           # 板總厚

CUT_P = 20.0            # 縫的 x 週期（固定 → 縫中心不動，漸變才不會扭曲圖案）
ROW_P = 8.0             # 行距（＝條帶寬，決定挫曲尺度）
SLIT_W = 0.8            # 縫寬：2 條走線，0.4 噴嘴必留空（v2 的 0.6 是邊界值）
BRIDGE_NEAR, BRIDGE_FAR = 6.0, 2.0   # 橋接段長漸變（近板→外緣）
R_TAPER = 70.0          # 此半徑外橋最短（最軟）

def bridge_len(x, y):
    r = np.hypot(x - CX, y - CY)
    t = np.clip((r - R_PLATE) / (R_TAPER - R_PLATE), 0, 1)
    return BRIDGE_NEAR + t * (BRIDGE_FAR - BRIDGE_NEAR)

# 行 j 的縫沿 x 排列，奇偶行錯開半週期 → 拉伸時橋接段扭轉、條帶面外挫曲成蕾絲
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
        # round cap＝縫端洩壓孔，PLA 才不會從縫端撕開（v2 用 flat cap，直角應力集中）
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
print(f"rows={n_rows}  縫數={len(slits)}  縫長 {CUT_P-BRIDGE_NEAR:.0f}(近板)→{CUT_P-BRIDGE_FAR:.0f}(外緣)")
print(f"開孔率={open_ratio*100:.1f}%（v2 切縫片約 11%）")
print(f"volume={vol:.1f} cm³  PLA≈{vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# ---- 預覽 ----
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
