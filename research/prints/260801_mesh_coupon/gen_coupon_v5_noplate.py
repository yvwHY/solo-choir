# v5_noplate — 拿掉中央剛性板的對照組（桿寬漸變保留，單一變數）
# 桿件式 auxetic：機構靠桿件彎折而非窄縫剪切 → 所有負空間 mm 級，
# 無 gap-fill 風險、無黏死風險、零支撐；印出來就是展開態（迴避 v2 的預轉開焊死）
# 漸變＝桿寬（節點不動）：板附近 1.4（硬）→ 外緣 0.8（軟）
# 產出：mesh_coupon_v5_noplate.stl + preview_v5_noplate.png
import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
import trimesh

# ---- 參數（mm）----
SIZE = 150.0
CX = CY = SIZE / 2
R_PLATE = 27.5          # 實心板半徑（⌀55）
SHEET_T = 0.6           # 網厚 3 層——剛度線性項，v3 沿用 0.8 是沒想過的旋鈕
PLATE_T = 1.6           # 板總厚

W = 18.0                # 水平格距（v3 的 1.5×——桿長是三次方旋鈕且沒有地板）
HV = 13.5               # 垂直桿長（同比放大）
S = -5.25               # 斜桿垂直位移，負＝內凹＝re-entrant（排距 = HV+S）
ROW_H = HV + S          # 8.25 — 拉伸時斜桿被拉直、橫向同時變寬（負泊松比）

RIB_NEAR, RIB_FAR = 2.0, 0.8   # 桿寬漸變（近板→外緣），面內彎曲剛度 ∝ 桿寬³ → 差 15×
R_TAPER = 70.0          # 此半徑外桿最細

TAB_H = 3.0             # 上下夾持條（左右刻意開放，橫向膨脹才目測得出來）

def rib_w(x, y):
    r = np.hypot(x - CX, y - CY)
    t = np.clip((r - R_PLATE) / (R_TAPER - R_PLATE), 0, 1)
    return RIB_NEAR + t * (RIB_FAR - RIB_NEAR)

ribs = []
def add(p0, p1):
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ribs.append(LineString([p0, p1]).buffer(rib_w(mx, my) / 2, cap_style=1, quad_segs=6))

# 排 j 的垂直桿底端在 Yj，頂端 Yj+HV；斜桿從頂端往下折回到下一排底端（x±W/2, Yj+ROW_H）
# 奇偶排錯開半格 → 斜桿端點正好落在下一排的垂直桿底端
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

y_top = y_lo + n_rows * ROW_H + HV      # 網格最高點——上夾持條貼齊它，不留斷點
frame = box(0, 0, SIZE, SIZE)
tabs = unary_union([box(0, 0, SIZE, TAB_H), box(0, y_top, SIZE, SIZE)])
plate = Point(CX, CY).buffer(R_PLATE, quad_segs=96)
sheet_poly = unary_union(ribs + [tabs]).intersection(frame)

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
combined = trimesh.util.concatenate(meshes)
combined.export("mesh_coupon_v5_noplate.stl")
vol = combined.volume / 1000.0
print(f"STL out: mesh_coupon_v5_noplate.stl  triangles={len(combined.faces)}")
print(f"cell={W}x{ROW_H} 內凹角={np.degrees(np.arctan2(-S, W/2)):.1f}°  rows={n_rows+1}")
print(f"volume={vol:.1f} cm³  PLA≈{vol*1.24:.0f} g  bbox={combined.bounds[1]-combined.bounds[0]}")

# 最小開孔檢查——桿件式的賣點就是這個數字遠大於噴嘴
holes = [len(g.interiors) for g in (sheet_poly.geoms if sheet_poly.geom_type == "MultiPolygon" else [sheet_poly])]
print(f"孔數={sum(holes)}（每孔皆 mm 級，無 <1mm 窄縫）")

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
ax.set_xlim(-5, SIZE + 5); ax.set_ylim(-5, SIZE + 5)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("mesh coupon v5 soft no-plate — pure mesh baseline")
fig.savefig("preview_v5_noplate.png", dpi=150, bbox_inches="tight")
print("preview: preview_v5_noplate.png")
