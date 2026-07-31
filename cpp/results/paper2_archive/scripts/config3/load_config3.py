# -*- coding: utf-8 -*-
"""Load the paper2 configuration-3 dumps (S1, S2, S3) and build the symmetry diagnostics.

Configuration 3 has no analytic solution, so the error measure is the one the bench itself
uses: the residual of the exact diagonal (y = x) reflection symmetry that the initial
condition and the true solution both possess. On triangulate_box(N,N) the reflection is
purely index arithmetic, cell 2*(j*N+i)+s maps to 2*(i*N+j)+(1-s).
"""
import numpy as np, os, re

from _paths import RESULTS as R
from _paths import CACHE_DIR as S
CACHE = os.path.join(S, "cfg3_cache.npz")
N = 280

SRC = {k: os.path.join(R, "paper2_3_data", "config3", k, "config3.txt.vtk")
       for k in ("s1", "s2", "s3")}


def read_vtk(path):
    tok = open(path, "r").read().split("\n")
    i = 0
    pts = cells = None
    fields = {}
    name = None
    while i < len(tok):
        ln = tok[i].strip()
        if ln.startswith("SCALARS"):
            name = ln.split()[1]; i += 1; continue
        if ln.startswith("POINTS"):
            n = int(ln.split()[1]); buf = []; i += 1
            while len(buf) < 3 * n:
                buf.extend(tok[i].split()); i += 1
            pts = np.array(buf[:3 * n], dtype=float).reshape(n, 3); continue
        if ln.startswith("CELLS"):
            nc, ntot = int(ln.split()[1]), int(ln.split()[2]); buf = []; i += 1
            while len(buf) < ntot:
                buf.extend(tok[i].split()); i += 1
            a = np.array(buf[:ntot], dtype=np.int64)
            cells = a.reshape(nc, a[0] + 1)[:, 1:]; continue
        if ln.startswith("LOOKUP_TABLE"):
            buf = []; i += 1
            while i < len(tok) and len(buf) < len(cells):
                s_ = tok[i].split()
                if s_ and not s_[0][0].isalpha():
                    buf.extend(s_)
                i += 1
            fields[name] = np.array(buf[:len(cells)], dtype=float); continue
        i += 1
    return pts, cells, fields


out = {}
for k, p in SRC.items():
    pts, cells, f = read_vtk(p)
    print(f"{k}: points={len(pts)} cells={cells.shape} fields={sorted(f)}")
    for fn, v in f.items():
        out[f"{k}_{fn}"] = v
        print(f"     {fn:10s} [{v.min():.5g}, {v.max():.5g}]")
    if "pts" not in out:
        out["pts"], out["cells"] = pts[:, :2], cells

pts, cells = out["pts"], out["cells"]
NC = len(cells)
assert NC == 2 * N * N, f"expected {2*N*N} cells, got {NC}"
out["cc"] = pts[cells].mean(axis=1)

# reflection permutation, straight from the bench's own indexing
idx = np.arange(NC)
q, s = idx // 2, idx % 2
i_, j_ = q % N, q // N
refl = 2 * (i_ * N + j_) + (1 - s)
out["refl"] = refl

cc = out["cc"]
print(f"\nreflection check: max |x_i - y_refl(i)| = "
      f"{np.abs(cc[:,0]-cc[refl,1]).max():.3e}  "
      f"max |y_i - x_refl(i)| = {np.abs(cc[:,1]-cc[refl,0]).max():.3e}")

print("\n=== symmetry residual, reproducing the bench metric ===")
for k in ("s1", "s2", "s3"):
    rho = out[f"{k}_rho"] if f"{k}_rho" in out else out[f"{k}_density"]
    d = rho - rho[refl]
    sym = np.sqrt((d ** 2).sum() / (rho ** 2).sum())
    out[f"{k}_symres"] = np.abs(d)
    print(f"  {k}: sym_rms = {sym:.3e}   max|d rho| = {np.abs(d).max():.4f}   "
          f"cells with |d rho| > 0.01: {(np.abs(d) > 0.01).sum()}")

np.savez_compressed(CACHE, **out)
print("\ncached ->", CACHE)
