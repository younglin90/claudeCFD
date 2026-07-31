#!/usr/bin/env python3
# Reliable MIXED unstructured 3D: a conforming mixed (tri+quad) 2D mesh extruded in z.
#   quad column -> hexahedron, tri column -> prism (wedge).  Vertical faces are all quads
#   (from 2D edges) so the 3D mesh is conforming. Optional tet cap region -> + pyramids/tets.
# Usage: python3 gen_mesh3d_extrude.py <out.umsh> <h> <nz> [Lx Ly Lz]
import sys, gmsh
out = sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube_ext.umsh"
h   = float(sys.argv[2]) if len(sys.argv)>2 else 0.14
nz  = int(sys.argv[3]) if len(sys.argv)>3 else 7
Lx,Ly,Lz = (float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])) if len(sys.argv)>6 else (1.0,1.0,1.0)

gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("ext")
s = gmsh.model.occ.addRectangle(0,0,0, Lx,Ly)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin", h); gmsh.option.setNumber("Mesh.MeshSizeMax", h)
# partial recombine -> conforming mixed tri+quad in 2D (RecombineOptimizeTopology + threshold)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)   # simple -> leaves some triangles -> MIXED
gmsh.model.mesh.generate(2)
# extrude in z with nz layers, recombine columns (quad->hex, tri->prism)
gmsh.model.mesh.setRecombine(2, s) if False else None
ext = gmsh.model.occ.extrude([(2,s)], 0,0,Lz, numElements=[nz], recombine=True)
# (occ.extrude after meshing won't carry the 2D mesh; do extrude on geometry then mesh 3D)
gmsh.finalize()

# --- redo cleanly: geometry extrude THEN mesh ---
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("ext2")
s = gmsh.model.occ.addRectangle(0,0,0, Lx,Ly)
gmsh.model.occ.synchronize()
gmsh.model.occ.extrude([(2,s)], 0,0,Lz, numElements=[nz], recombine=True)
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin", h); gmsh.option.setNumber("Mesh.MeshSizeMax", h)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)   # mixed tri+quad base
gmsh.model.mesh.generate(3)

ntags, ncoords, _ = gmsh.model.mesh.getNodes(); ncoords=ncoords.reshape(-1,3)
tag2idx={int(t):i for i,t in enumerate(ntags)}
NV={4:4,5:8,6:6,7:5}; cells=[]; hist={}
for et in (4,5,6,7):
    etags,enodes=gmsh.model.mesh.getElementsByType(et)
    if len(etags)==0: continue
    nv=NV[et]; enodes=enodes.reshape(-1,nv)
    for row in enodes: cells.append([nv]+[tag2idx[int(t)] for t in row])
    hist[nv]=hist.get(nv,0)+len(etags)
gmsh.finalize()
with open(out,"w") as f:
    f.write(f"NODES {len(ncoords)}\n")
    for p in ncoords: f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
    f.write(f"CELLS {len(cells)}\n")
    for c in cells: f.write(" ".join(str(x) for x in c)+"\n")
name={4:"tet",5:"pyr",6:"prism",8:"hex"}
print(f"wrote {out}: {len(ncoords)} nodes, {len(cells)} cells  "+" ".join(f"{name[k]}={v}" for k,v in sorted(hist.items())))
