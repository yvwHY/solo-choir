#!/usr/bin/env python3
"""Pack binary STLs onto a plate (shelf packing, 90deg rot allowed) -> one combined binary STL.
Also supports free Z-rotation of a single part (door_panel diagonal fit).
PrusaSlicer auto-centers the loaded object group on the bed, so we lay out around origin."""
import struct, math, sys

SRC = "/Users/liaoyu-ting/Documents/GS/CA_Term2/FinalProject/SoloChoirCode/260719_prints"

def read_stl(fn):
    d = open(f"{SRC}/{fn}", "rb").read()
    n = struct.unpack('<I', d[80:84])[0]
    tris = []  # each: (nx,ny,nz, x1,y1,z1, x2,y2,z2, x3,y3,z3)
    off = 84
    for i in range(n):
        vals = struct.unpack('<12f', d[off:off+48])
        tris.append(list(vals))
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
        # normal (index 0,1,2)
        nx,ny=t[0],t[1]
        nt[0]=nx*c-ny*s; nt[1]=nx*s+ny*c
        for k in range(3):
            x=t[3+k*3]; y=t[4+k*3]
            nt[3+k*3]=x*c-y*s+dx
            nt[4+k*3]=x*s+y*c+dy
            nt[5+k*3]=t[5+k*3]+dz
        out.append(nt)
    return out

def drop_to_bed(tris):
    """Shift so min Z = 0 (matches PrusaSlicer ensure-on-bed for a single part)."""
    zmin=min(t[5+k*3] for t in tris for k in range(3))
    return transform(tris,0.0,0,0,-zmin)

def recenter(tris, cx=90.0, cy=90.0):
    x0,x1,y0,y1=bbox_xy(tris)
    return transform(tris, 0.0, cx-(x0+x1)/2, cy-(y0+y1)/2)

def write_stl(all_tris, out):
    with open(out,'wb') as f:
        f.write(b'\0'*80)
        f.write(struct.pack('<I', len(all_tris)))
        for t in all_tris:
            f.write(struct.pack('<12f', *t))
            f.write(b'\0\0')
    print(f"  wrote {out}  ({len(all_tris)} tris)")

# ---- shelf packing ----
GAP = 4.0        # mm between parts
MAXW = 172.0     # usable (leave skirt room on 180 bed)

def pack(parts):
    """parts: list of (name, tris). Returns combined recentred tris.
    Each part placed at its native XY footprint; may rotate 90deg to pack tighter."""
    items=[]
    for name,tris in parts:
        x0,x1,y0,y1=bbox_xy(tris)
        w=x1-x0; h=y1-y0
        # normalise part to origin corner (XY) and drop to bed (Z)
        t0=drop_to_bed(transform(tris,0.0,-x0,-y0))
        items.append([name,t0,w,h])
    # sort by height desc for shelf packing; allow 90deg rot so longer side = width
    for it in items:
        # orient so width>=height (lay long axis along X)
        if it[3]>it[2]:
            it[1]=transform(it[1],90.0,0,0)  # rotate; then re-normalise
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
        print(f"    {name:24} {w:6.1f}x{h:5.1f} @ shelf_y={shelf_y:.1f}")
    total=[t for grp in placed for t in grp]
    total=recenter(total)
    x0,x1,y0,y1=bbox_xy(total)
    print(f"  group bbox: {x1-x0:.1f} x {y1-y0:.1f}")
    if x1-x0>MAXW or y1-y0>MAXW:
        print("  !!! GROUP EXCEEDS PLATE");
    return total

if __name__=="__main__":
    plate=sys.argv[1]
    OUT="/private/tmp/claude-501/-Users-liaoyu-ting-Documents-GS-CA-Term2-FinalProject-SoloChoirCode-260615-voice-changer/1b57875e-31aa-491e-918f-be83e9f81908/scratchpad"
    if plate=="frame9":
        names=["band_clamp","bce_mount_L","bce_mount_R","door_clip_A1","door_clip_A2",
               "door_clip_R","door_col_A","door_col_R","pivot_mount"]
        parts=[(n,read_stl(n+".stl")) for n in names]
        write_stl(pack(parts), f"{OUT}/plate_frame9.stl")
    elif plate=="door_panel":
        t=read_stl("door_panel.stl")
        t=recenter(transform(t,45.0,0,0))
        x0,x1,y0,y1=bbox_xy(t)
        print(f"  door_panel@45deg bbox: {x1-x0:.1f} x {y1-y0:.1f}")
        write_stl(t, f"{OUT}/plate_door_panel.stl")
