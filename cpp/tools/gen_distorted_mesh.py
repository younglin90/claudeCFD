#!/usr/bin/env python3
# gen_distorted_mesh.py — read a base .umsh, perturb INTERIOR nodes by alpha*h_local
# (random) to inject controlled SKEWNESS + NON-ORTHOGONALITY while keeping the domain
# boundary fixed (topology unchanged -> still conforming). For MMS accuracy-vs-distortion.
#
# Usage: python3 gen_distorted_mesh.py <in.umsh> <out.umsh> <alpha> [seed]
#   alpha = perturbation amplitude as fraction of the local min incident edge (0..~0.4).
import sys, numpy as np

inp   = sys.argv[1]
out   = sys.argv[2]
alpha = float(sys.argv[3]) if len(sys.argv)>3 else 0.25
seed  = int(sys.argv[4])   if len(sys.argv)>4 else 12345
rng = np.random.default_rng(seed)

# ---- read .umsh ----
with open(inp) as f:
    toks = f.read().split()
i = 0
assert toks[i]=="NODES"; nn=int(toks[i+1]); i+=2
xyz = np.array(toks[i:i+3*nn], dtype=float).reshape(nn,3); i+=3*nn
assert toks[i]=="CELLS"; nc=int(toks[i+1]); i+=2
cells=[]
for _ in range(nc):
    nv=int(toks[i]); i+=1
    cells.append([int(toks[i+k]) for k in range(nv)]); i+=nv

# ---- local edge scale per node (min incident edge length) ----
hmin = np.full(nn, np.inf)
for c in cells:
    for a in range(len(c)):
        for b in range(a+1,len(c)):
            d = np.linalg.norm(xyz[c[a]]-xyz[c[b]])
            if d<hmin[c[a]]: hmin[c[a]]=d
            if d<hmin[c[b]]: hmin[c[b]]=d
hmin[~np.isfinite(hmin)] = np.median(hmin[np.isfinite(hmin)])

# ---- boundary nodes (on bbox faces) stay fixed ----
lo = xyz.min(axis=0); hi = xyz.max(axis=0)
tol = 1e-9 + 1e-6*(hi-lo)
on_bnd = np.zeros(nn, dtype=bool)
for d in range(3):
    on_bnd |= (np.abs(xyz[:,d]-lo[d])<tol[d]) | (np.abs(xyz[:,d]-hi[d])<tol[d])

# ---- perturb interior nodes by uniform cube [-alpha*h, alpha*h]^3 ----
disp = (rng.random((nn,3))*2.0-1.0) * (alpha*hmin)[:,None]
disp[on_bnd] = 0.0
xyz2 = xyz + disp

# ---- write ----
with open(out,"w") as f:
    f.write(f"NODES {nn}\n")
    for p in xyz2: f.write(f"{p[0]:.10g} {p[1]:.10g} {p[2]:.10g}\n")
    f.write(f"CELLS {nc}\n")
    for c in cells: f.write(f"{len(c)} "+" ".join(str(v) for v in c)+"\n")
print(f"wrote {out}: {nn} nodes ({on_bnd.sum()} bnd fixed, {nn-on_bnd.sum()} interior perturbed), "
      f"{nc} cells, alpha={alpha}, median h={np.median(hmin):.4g}")
