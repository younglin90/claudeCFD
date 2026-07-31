#!/usr/bin/env python3
# Programmatic CONFORMING mixed mesh (gmsh cannot do 4-in-1). z-slabs of a Nx*Ny*Nz cube
# grid; each cube emits cells by its slab TYPE. Conformity: all slab types keep QUAD z-faces
# (hex/pyr) or quad z-faces (prism cut in x-z -> tri only on y-faces, matched within slab).
#   HEX  : the cube (8 nodes).
#   PRISM: 2 prisms, cube cut by the x-z diagonal plane (tri faces on y=const, quad z-faces).
#   PYR  : 6 pyramids, apex at cube centre (all 6 faces = the cube's quad faces).
#   TET  : 6 tets around the cube centre with each cube face split into 2 tris -> needs the
#          6 face-neighbours to present matching tris, so TET cubes are placed as isolated
#          pockets whose 6 neighbours are PYR-transition cubes (1 face tri toward pocket).
# Slab pattern is chosen so neighbours are compatible; geometry verified by test_unstructured3d.
# Usage: gen_mesh3d_prog.py <out.umsh> <N> [Lx Ly Lz]
import sys
out=sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube_prog.umsh"
N=int(sys.argv[2]) if len(sys.argv)>2 else 6
Lx,Ly,Lz=(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])) if len(sys.argv)>5 else (1.0,1.0,1.0)
Nx=Ny=Nz=N; hx,hy,hz=Lx/Nx,Ly/Ny,Lz/Nz
# slab pattern: ratio 8 HEX : 4 PRISM : 1 PYR : 1 PYR2TET gives ~EQUAL cell counts of all
# four types (hex=1 cell/cube, prism=2, pyr=6, pyr2tet=8tet+2pyr). Interleaved + cycled.
import os
PAT = os.environ.get("PAT", "HPHYHPHGHPHHPH")   # 8H 4P 1Y 1G
SLABMAP={"H":"HEX","P":"PRISM","Y":"PYR","G":"PYR2TET"}

nodes=[]; nmap={}
def nid(key, xyz):
    if key not in nmap: nmap[key]=len(nodes); nodes.append(xyz)
    return nmap[key]
def corner(i,j,k): return nid((i,j,k), (i*hx,j*hy,k*hz))
def center(i,j,k): return nid(("c",i,j,k), ((i+0.5)*hx,(j+0.5)*hy,(k+0.5)*hz))

cells=[]
# slab type per k: balance hex / prism / pyramid (tet added in a later revision once verified)
# pattern: 0->HEX, 1->PRISM, 2->PYR, repeat
for k in range(Nz):
    typ=SLABMAP[PAT[k%len(PAT)]]
    for j in range(Ny):
        for i in range(Nx):
            # cube corners: c[di][dj][dk]
            c=[[[corner(i+di,j+dj,k+dk) for dk in (0,1)] for dj in (0,1)] for di in (0,1)]
            b00,b10,b11,b01=c[0][0][0],c[1][0][0],c[1][1][0],c[0][1][0]   # z=k (bottom)
            t00,t10,t11,t01=c[0][0][1],c[1][0][1],c[1][1][1],c[0][1][1]   # z=k+1 (top)
            if typ=="HEX":
                cells.append([8,b00,b10,b11,b01,t00,t10,t11,t01])
            elif typ=="PRISM":
                # cut by x-z diagonal plane through y-edges: tri faces on y=0 and y=1.
                # prism nodes (gmsh wedge: tri bottom 0,1,2 ; tri top 3,4,5).
                # tri in x-z at y=j: (b00,b10,t10) and (b00,t10,t00)
                cells.append([6, b00,b10,t10, b01,b11,t11])   # lower prism (tri x-z, quad z-faces)
                cells.append([6, b00,t10,t00, b01,t11,t01])   # upper prism
            elif typ=="PYR2TET":
                # top/bottom pyramids (QUAD z-faces -> match neighbour slabs) + 4 side faces
                # each split into 2 TETS (TRI x/y-faces -> match within slab; diagonals are
                # global edges so neighbour cubes agree). -> tet + pyramid in one slab.
                m=center(i,j,k)
                cells.append([5, b00,b10,b11,b01, m])          # bottom pyramid (quad base)
                cells.append([5, t00,t01,t11,t10, m])          # top pyramid (quad base)
                # y=0 face quad(b00,b10,t10,t00) diag b00-t10 ; y=1 quad(b01,b11,t11,t01) diag b01-t11
                cells.append([4, b00,b10,t10, m]); cells.append([4, b00,t10,t00, m])
                cells.append([4, b01,b11,t11, m]); cells.append([4, b01,t11,t01, m])
                # x=0 face quad(b00,b01,t01,t00) diag b00-t01 ; x=1 quad(b10,b11,t11,t10) diag b10-t11
                cells.append([4, b00,b01,t01, m]); cells.append([4, b00,t01,t00, m])
                cells.append([4, b10,b11,t11, m]); cells.append([4, b10,t11,t10, m])
            else:  # PYR: 6 pyramids apex=cube centre
                ctr=center(i,j,k)
                # each cube face (quad) -> pyramid base + apex ctr. pyramid: base 0-3, apex 4.
                cells.append([5, b00,b10,b11,b01, ctr])   # bottom z
                cells.append([5, t00,t01,t11,t10, ctr])   # top z
                cells.append([5, b00,t00,t10,b10, ctr])   # y=0
                cells.append([5, b01,b11,t11,t01, ctr])   # y=1
                cells.append([5, b00,b01,t01,t00, ctr])   # x=0
                cells.append([5, b10,t10,t11,b11, ctr])   # x=1

with open(out,"w") as f:
    f.write(f"NODES {len(nodes)}\n")
    for p in nodes: f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
    f.write(f"CELLS {len(cells)}\n")
    for c in cells: f.write(" ".join(str(x) for x in c)+"\n")
hist={}
for c in cells: hist[c[0]]=hist.get(c[0],0)+1
name={4:"tet",5:"pyr",6:"prism",8:"hex"}; tot=len(cells)
print(f"wrote {out}: {len(nodes)} nodes, {len(cells)} cells  "+" ".join(f"{name[k]}={v}({100*v//tot}%)" for k,v in sorted(hist.items())))
