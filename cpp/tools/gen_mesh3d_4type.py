#!/usr/bin/env python3
# True 4-type: extrude a mixed(tri+quad) 2D bottom slab -> HEX+PRISM, cap surface stays
# mixed; top slab tet-meshed sharing the cap -> TET + PYRAMID (quad cap cells). One mesh.
# Usage: gen_mesh3d_4type.py <out.umsh> <h> [Lx Ly Lz] [zc_frac]
import sys, gmsh
out=sys.argv[1] if len(sys.argv)>1 else "/tmp/mbq/cube_4t.umsh"
h=float(sys.argv[2]) if len(sys.argv)>2 else 0.1
Lx,Ly,Lz=(float(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5])) if len(sys.argv)>5 else (1.0,1.0,1.0)
zcf=float(sys.argv[6]) if len(sys.argv)>6 else 0.45
zc=Lz*zcf
gmsh.initialize(); gmsh.option.setNumber("General.Terminal",0)
gmsh.model.add("4t"); occ=gmsh.model.occ
rect=occ.addRectangle(0,0,0,Lx,Ly); occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMin",h); gmsh.option.setNumber("Mesh.MeshSizeMax",h)
gmsh.option.setNumber("Mesh.RecombineAll",1); gmsh.option.setNumber("Mesh.RecombinationAlgorithm",0)
gmsh.model.mesh.setRecombine(2,rect)
nl=max(1,int(round(zc/h)))
ext=occ.extrude([(2,rect)],0,0,zc,numElements=[nl],recombine=True)
occ.synchronize()
volb=[t for (d,t) in ext if d==3][0]
cap=[t for (d,t) in ext if d==2 and abs(occ.getCenterOfMass(2,t)[2]-zc)<1e-6][0]
top=occ.addBox(0,0,zc,Lx,Ly,Lz-zc); occ.synchronize()
occ.fragment([(3,top)],[(3,volb)]); occ.synchronize()
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
name={4:"tet",5:"pyr",6:"prism",8:"hex"}; tot=sum(hist.values()) or 1
print(f"wrote {out}: {len(ncoords)} nodes, {len(cells)} cells  "+
      " ".join(f"{name[k]}={v}({100*v//tot}%)" for k,v in sorted(hist.items())))
