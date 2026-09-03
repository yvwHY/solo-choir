#!/usr/bin/env python3
"""Pack binary STLs onto Prusa MINI plates (180x180) -> combined binary STLs.

Adapted from 260719_prints/gcode/pack.py. New in this batch:
  - rotate_x(): door_col_A/R are modelled tilted 30 deg in CAD; +30 deg about X
    stands the barrel axis vertical (260711 README A-table: "d8 barrel vertical").
PrusaSlicer auto-centres the loaded object group on the bed, so we lay out around origin.
"""
import struct, math, sys, os

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SRC, "plates")

def read_stl(fn):
    d = open(os.path.join(SRC, fn), "rb").read()
    n = struct.unpack('<I', d[80:84])[0]
    tris = []
    off = 84
    for i in range(n):
        tris.append(list(struct.unpack('<12f', d[off:off+48])))
        off += 50
    return tris

def bbox_xy(tris):
    xs=[]; ys=[]
    for t in tris:
        for k in range(3):
            xs.append(t[3+k*3]); ys.append(t[4+k*3])
    return min(xs),max(xs),min(ys),max(ys)

def transform(tris, angle_deg=0.0, dx=0.0, dy=0.0, dz=0.0):
    """Rotate about Z (origin) then translate. Rotates normals too."""
    a = math.radians(angle_deg); c=math.cos(a); s=math.sin(a)
    out=[]
    for t in tris:
        nt=t[:]
        nx,ny=t[0],t[1]
        nt[0]=nx*c-ny*s; nt[1]=nx*s+ny*c
        for k in range(3):
            x=t[3+k*3]; y=t[4+k*3]
            nt[3+k*3]=x*c-y*s+dx
            nt[4+k*3]=x*s+y*c+dy
            nt[5+k*3]=t[5+k*3]+dz
        out.append(nt)
    return out

def rotate_x(tris, angle_deg):
    """Rotate about the X axis (origin). y' = y c - z s ; z' = y s + z c."""
    a=math.radians(angle_deg); c=math.cos(a); s=math.sin(a)
    out=[]
    for t in tris:
        nt=t[:]
        ny,nz=t[1],t[2]
        nt[1]=ny*c-nz*s; nt[2]=ny*s+nz*c
        for k in range(3):
            y=t[4+k*3]; z=t[5+k*3]
            nt[4+k*3]=y*c-z*s
            nt[5+k*3]=y*s+z*c
        out.append(nt)
    return out

def drop_to_bed(tris):
    zmin=min(t[5+k*3] for t in tris for k in range(3))
    return transform(tris,0.0,0,0,-zmin)

def recenter(tris, cx=0.0, cy=0.0):
    x0,x1,y0,y1=bbox_xy(tris)
    return transform(tris, 0.0, cx-(x0+x1)/2, cy-(y0+y1)/2)

def write_stl(all_tris, out):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'wb') as f:
        f.write(b'\0'*80)
        f.write(struct.pack('<I', len(all_tris)))
        for t in all_tris:
            f.write(struct.pack('<12f', *t))
            f.write(b'\0\0')
    print(f"  wrote {out}  ({len(all_tris)} tris)")

GAP = 4.0
MAXW = 172.0

def pack(parts):
    items=[]
    for name,tris in parts:
        x0,x1,y0,y1=bbox_xy(tris)
        w=x1-x0; h=y1-y0
        t0=drop_to_bed(transform(tris,0.0,-x0,-y0))
        items.append([name,t0,w,h])
    for it in items:
        if it[3]>it[2]:
            it[1]=transform(it[1],90.0,0,0)
            x0,x1,y0,y1=bbox_xy(it[1])
            it[1]=transform(it[1],0.0,-x0,-y0)
            it[2],it[3]=it[3],it[2]
    items.sort(key=lambda it:-it[3])
    placed=[]
    cursor_x=0.0; shelf_y=0.0; shelf_h=0.0
    for name,tris,w,h in items:
        if cursor_x+w > MAXW and cursor_x>0:
            shelf_y+=shelf_h+GAP; cursor_x=0.0; shelf_h=0.0
        placed.append(transform(tris,0.0,cursor_x,shelf_y))
        cursor_x+=w+GAP
        shelf_h=max(shelf_h,h)
        print(f"    {name:20} {w:6.1f}x{h:5.1f} @ shelf_y={shelf_y:.1f}")
    total=[t for grp in placed for t in grp]
    total=recenter(total)
    x0,x1,y0,y1=bbox_xy(total)
    print(f"  group bbox: {x1-x0:.1f} x {y1-y0:.1f}")
    if x1-x0>MAXW or y1-y0>MAXW:
        print("  !!! GROUP EXCEEDS PLATE")
    return total

def load(name):
    """Load a part STL, applying its print orientation."""
    t = read_stl(name + ".stl")
    if name in ("door_col_A", "door_col_R"):
        t = rotate_x(t, 30.0)          # barrel axis -> vertical
    return drop_to_bed(t)

if __name__=="__main__":
    plate = sys.argv[1] if len(sys.argv)>1 else "all"
    if plate in ("P1","all"):
        # wo5_shell_v10 and jhook_west are ONE physical part (they touch, min dist 0.0mm):
        # the east hook is part of the shell body, the west hook is a separate body.
        # They must keep their relative position -> drop/recentre the pair as a group.
        print("P1 shell + west hook:")
        grp = read_stl("wo5_shell_v10.stl") + read_stl("jhook_west.stl")
        grp = recenter(drop_to_bed(grp))
        x0,x1,y0,y1 = bbox_xy(grp)
        print(f"  group bbox {x1-x0:.1f} x {y1-y0:.1f}")
        write_stl(grp, f"{OUT}/plate_P1_shell.stl")
    if plate in ("P2","all"):
        print("P2 frame9:")
        names=["band_clamp","door_col_A","door_col_R","pivot_mount",
               "door_clip_A1","door_clip_A2","door_clip_R","wo4_cup","wo4_cup_R"]
        write_stl(pack([(n,load(n)) for n in names]), f"{OUT}/plate_P2_frame9.stl")
    if plate in ("P3","all"):
        print("P3 door_panel @45deg:")
        t=recenter(transform(load("door_panel_flat"),45.0,0,0))
        x0,x1,y0,y1=bbox_xy(t)
        print(f"  bbox {x1-x0:.1f} x {y1-y0:.1f}")
        write_stl(t, f"{OUT}/plate_P3_door_panel.stl")
    if plate in ("P4","all"):
        print("P4 lid:")
        write_stl(recenter(load("wo5_lid_v10")), f"{OUT}/plate_P4_lid.stl")
    if plate in ("P5","all"):
        print("P5 voice-unit panels x3:")
        parts=[(f"vu1_panel_{i+1}", load("vu1_panel")) for i in range(3)]
        write_stl(pack(parts), f"{OUT}/plate_P5_vu1x3.stl")
