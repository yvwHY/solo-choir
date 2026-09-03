# mesh coupon v2 — auxetic 切縫片（rotating-squares slit sheet）×剛性板
# 印時＝近實面＋0.6mm 細縫（閉合態，零重疊問題）；拉開時方塊繞角鉸旋轉 → 蕾絲感浮現
# 漸變＝鉸鏈長度：板附近 2.4（硬）→ 外緣 0.8（軟）→ 拉伸時外緣先開、板區不動
# 產出：mesh_coupon_v2_auxetic.stl + preview_auxetic.png
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- 參數（mm）----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # 實心板半徑（⌀55）
PITCH = 8.0             # 方塊格距
SLIT_W = 0.6            # 縫寬（0.4 噴嘴可printable 的最小可靠空隙）
HINGE_NEAR, HINGE_FAR = 2.4, 0.8   # 鉸鏈長漸變（近板→外緣）
R_TAPER1 = 70.0         # 此半徑外鉸鏈最細
SHEET_T = 0.8           # 片厚（4 層 @0.2）
PLATE_T = 1.6           # 板總厚

def hinge_len(r):
    t = np.clip((r - R_PLATE) / (R_TAPER1 - R_PLATE), 0, 1)
    return HINGE_NEAR + t * (HINGE_FAR - HINGE_NEAR)

# 方塊 (i,j) 旋轉方向 σ=+1 if (i+j) even（決定每條縫的鉸鏈留哪一端）
# 水平相鄰對：σ1=+1 → 鉸在下端；垂直相鄰對：σ1=+1 → 鉸在右端
N = int(round(SIZE / PITCH))
slits = []
for i in range(N):
    for j in range(N):
        sigma = 1 if (i + j) % 2 == 0 else -1
        # 與右鄰的共邊（x = (i+1)*PITCH，y 從 j*P 到 (j+1)*P）
        if i + 1 < N:
            x = (i + 1) * PITCH
            y0, y1 = j * PITCH, (j + 1) * PITCH
            mid = np.array([x, (y0 + y1) / 2])
            hl = hinge_len(np.hypot(mid[0] - CX, mid[1] - CY))
            if sigma > 0:   # 鉸在下端 → 縫從上端往下留 hl
                seg = LineString([(x, y1), (x, y0 + hl)])
            else:           # 鉸在上端
                seg = LineString([(x, y0), (x, y1 - hl)])
            slits.append(seg.buffer(SLIT_W / 2, cap_style=2))
        # 與上鄰的共邊（y = (j+1)*PITCH，x 從 i*P 到 (i+1)*P）
        if j + 1 < N:
            y = (j + 1) * PITCH
            x0, x1 = i * PITCH, (i + 1) * PITCH
            mid = np.array([(x0 + x1) / 2, y])
            hl = hinge_len(np.hypot(mid[0] - CX, mid[1] - CY))
            if sigma > 0:   # 鉸在右端 → 縫從左端往右留 hl
                seg = LineString([(x0, y), (x1 - hl, y)])
            else:           # 鉸在左端
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
ax.set_title("mesh coupon v2 auxetic slit sheet — rest state (stretch to open)")
fig.savefig("preview_auxetic.png", dpi=150, bbox_inches="tight")
print("preview: preview_auxetic.png")
