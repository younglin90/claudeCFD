#!/usr/bin/env python3
# 4-type mesh attempt: tet interior + BoundaryLayer (PRISM) near walls + RecombineAll (HEX
# where possible) + PYRAMID at quad/tet transitions. Usage: gen_mesh3d_bl.py <out> <h> [L..]
import sys, gmsh
out=sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube_bl.umsh"
h=float(sys.argv[2]) if len(sys.argv)>2 else 0.09
Lx,Ly,Lz=(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])) if len(sys.argv)>5 else (1.0,1.0,1.0)
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("bl")
gmsh.model.occ.addBox(0,0,0,Lx,Ly,Lz); gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin",h); gmsh.option.setNumber("Mesh.MeshSizeMax",h)
# boundary layer (prisms) off all boundary surfaces
surfs=[t for (d,t) in gmsh.model.getEntities(2)]
try:
    f=gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(f,"SurfacesList",surfs)
    gmsh.model.mesh.field.setNumber(f,"Size",h*0.45)
    gmsh.model.mesh.field.setNumber(f,"Ratio",1.3)
    gmsh.model.mesh.field.setNumber(f,"Quads",1)
    gmsh.model.mesh.field.setNumber(f,"Thickness",h*1.5)
    gmsh.model.mesh.field.setAsBoundaryLayer(f)
except Exception as e: print("BL warn:",e)
gmsh.option.setNumber("Mesh.RecombineAll",1); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",0)
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
with open(out,"w") as f2:
    f2.write(f"NODES {len(ncoords)}\n")
    for p in ncoords: f2.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
    f2.write(f"CELLS {len(cells)}\n")
    for c in cells: f2.write(" ".join(str(x) for x in c)+"\n")
name={4:"tet",5:"pyr",6:"prism",8:"hex"}; tot=sum(hist.values()) or 1
print(f"wrote {out}: {len(ncoords)} nodes, {len(cells)} cells  "+
      " ".join(f"{name[k]}={v}({100*v//tot}%)" for k,v in sorted(hist.items())))
