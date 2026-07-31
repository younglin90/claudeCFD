#!/usr/bin/env python3
# Balanced all-type unstructured mesh. Stack NS z-slabs; alternate slab type:
#   even slab -> transfinite+recombine = HEX ;  odd slab -> tet (Delaunay).
# Each hex/tet interface forces gmsh to insert PYRAMIDS (quad base) + the tet slabs that
# touch quad side-walls also yield PRISMS. Many interfaces => balanced tet/hex + plenty of
# pyramid/prism (not hex-dominant). Tune NS (more slabs => more transition cells).
# Usage: python3 gen_mesh3d_full.py <out.umsh> <h> <NS> [Lx Ly Lz]
import sys, gmsh
out=sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube_full.umsh"
h  =float(sys.argv[2]) if len(sys.argv)>2 else 0.09
NS =int(sys.argv[3]) if len(sys.argv)>3 else 5
Lx,Ly,Lz=(float(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6])) if len(sys.argv)>6 else (1.0,1.0,1.0)

gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("full")
dz=Lz/NS; boxes=[]
for s in range(NS): boxes.append((3, gmsh.model.occ.addBox(0,0,s*dz, Lx,Ly,dz)))
gmsh.model.occ.fragment(boxes, [])              # conform all shared interfaces
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin",h); gmsh.option.setNumber("Mesh.MeshSizeMax",h)
import os
gmsh.option.setNumber("Mesh.RecombineAll", float(os.environ.get("RECOMB","0")))
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
vols=gmsh.model.getEntities(3)
hexvols=[]
for (d,t) in vols:
    z=gmsh.model.occ.getCenterOfMass(d,t)[2]; s=int(z/dz - 1e-9)
    if s%2==0: hexvols.append((d,t))          # even slab -> hex
if hexvols:
    try: gmsh.model.mesh.setTransfiniteAutomatic(hexvols, recombine=True)
    except Exception as e: print("tf warn:",e)
gmsh.model.mesh.generate(3)

ntags,ncoords,_=gmsh.model.mesh.getNodes(); ncoords=ncoords.reshape(-1,3)
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
name={4:"tet",5:"pyr",6:"prism",8:"hex"}; tot=sum(hist.values())
print(f"wrote {out}: {len(ncoords)} nodes, {len(cells)} cells  "+
      " ".join(f"{name[k]}={v}({100*v//tot}%)" for k,v in sorted(hist.items())))
