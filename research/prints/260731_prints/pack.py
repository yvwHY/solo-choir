#!/usr/bin/env python3
"""260731 batch: single-part plates, so no XY packing (PrusaSlicer auto-centres).
Only job: flip base_lid 180 deg about X (print plate-face-down, skirt up) and
drop everything to z=0. Outputs *_print.stl into plates/.
"""
import struct, math, os

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SRC, "plates")
os.makedirs(OUT, exist_ok=True)

def read_stl(fn):
    d = open(os.path.join(SRC, fn), "rb").read()
    n = struct.unpack('<I', d[80:84])[0]
    tris, off = [], 84
    for _ in range(n):
        tris.append(list(struct.unpack('<12f', d[off:off+48])))
        off += 50
    return tris

def write_stl(fn, tris):
    with open(os.path.join(OUT, fn), "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<12f', *t))
            f.write(b"\0\0")

def flip_x(tris):
    out = []
    for t in tris:
        nt = t[:]
        nt[1], nt[2] = -t[1], -t[2]
        for k in range(3):
            nt[4+k*3] = -t[4+k*3]
            nt[5+k*3] = -t[5+k*3]
        # 翻面後三角形繞向反轉,交換頂點 1/2 保持法向一致
        v1 = nt[3:6][:]; v2 = nt[6:9][:]
        nt[3:6], nt[6:9] = v2, v1
        out.append(nt)
    return out

def drop_z(tris):
    zmin = min(t[5+k*3] for t in tris for k in range(3))
    for t in tris:
        for k in range(3):
            t[5+k*3] -= zmin
    return tris

for src, dst, flip in [
    ("vu2_pyramid.stl", "P1_pyramid.stl", False),
]:
    tris = read_stl(src)
    if flip:
        tris = flip_x(tris)
    tris = drop_z(tris)
    zmax = max(t[5+k*3] for t in tris for k in range(3))
    write_stl(dst, tris)
    print(f"{dst}: {len(tris)} tris, height {zmax:.1f}mm")
