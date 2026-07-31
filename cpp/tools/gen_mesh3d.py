#!/usr/bin/env python3
# gen_mesh3d.py — generate a genuine UNSTRUCTURED 3D mesh (mixed tetra/hexa/prism/pyramid)
# with gmsh, export to a simple ASCII ".umsh" the C++ build_unstructured_3d reads.
#
# Usage: python3 gen_mesh3d.py <out.umsh> <h> <mode> [Lx Ly Lz]
#   mode: tet   = all-tetrahedra (Delaunay)            -> genuine unstructured
#         mixed = hex-dominant recombine (3D)          -> tetra+hexa+prism+pyramid mix
# .umsh format:
#   NODES <Nn>
#   x y z                       (Nn lines)
#   CELLS <Nc>
#   <nv> v0 v1 ... v(nv-1)      (Nc lines; nv: 4=tet,5=pyr,6=prism,8=hex, 0-based nodes)
import sys, gmsh

out  = sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube.umsh"
h    = float(sys.argv[2]) if len(sys.argv)>2 else 0.15
mode = sys.argv[3] if len(sys.argv)>3 else "mixed"
Lx,Ly,Lz = (float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])) if len(sys.argv)>6 else (1.0,1.0,1.0)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("dom")
gmsh.model.occ.addBox(0,0,0, Lx,Ly,Lz)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin", h)
gmsh.option.setNumber("Mesh.MeshSizeMax", h)
if mode == "mixed":
    gmsh.option.setNumber("Mesh.RecombineAll", 1)        # recombine 2D -> quads
    gmsh.option.setNumber("Mesh.Recombine3DAll", 1)      # 3D recombine -> hex/prism/pyramid + leftover tet
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.model.mesh.generate(3)

# nodes (remap gmsh tags -> contiguous 0-based)
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
ncoords = ncoords.reshape(-1,3)
tag2idx = {int(t):i for i,t in enumerate(ntags)}

# 3D elements: gmsh type 4=tet(4n) 5=hex(8n) 6=prism(6n) 7=pyramid(5n)
NV = {4:4, 5:8, 6:6, 7:5}
cells=[]; hist={}
for et in (4,5,6,7):
    etags, enodes = gmsh.model.mesh.getElementsByType(et)
    if len(etags)==0: continue
    nv = NV[et]; enodes = enodes.reshape(-1,nv)
    for row in enodes:
        cells.append([nv]+[tag2idx[int(t)] for t in row])
    hist[nv]=hist.get(nv,0)+len(etags)
gmsh.finalize()

with open(out,"w") as f:
    f.write(f"NODES {len(ncoords)}\n")
    for p in ncoords: f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
    f.write(f"CELLS {len(cells)}\n")
    for c in cells: f.write(" ".join(str(x) for x in c)+"\n")
name={4:"tet",5:"pyr",6:"prism",8:"hex"}
print(f"wrote {out}: {len(ncoords)} nodes, {len(cells)} cells  "+
      " ".join(f"{name[k]}={v}" for k,v in sorted(hist.items())))
