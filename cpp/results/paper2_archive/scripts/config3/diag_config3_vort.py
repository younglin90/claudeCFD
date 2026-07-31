# -*- coding: utf-8 -*-
"""Is the symmetry residual actually concentrated in the vortices, or only near shocks?
   Stratify it by the local vorticity and by a shock indicator, computed on a uniform
   resample of the unstructured field."""
import numpy as np, os
import matplotlib.tri as mtri

from _paths import CACHE_DIR as S
d = np.load(os.path.join(S, "cfg3_cache.npz"))
pts, cells, cc, refl = d["pts"], d["cells"], d["cc"], d["refl"]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)

def to_nodal(cv):
    acc = np.zeros(len(pts)); cnt = np.zeros(len(pts))
    for j in range(cells.shape[1]):
        np.add.at(acc, cells[:, j], cv); np.add.at(cnt, cells[:, j], 1.0)
    return acc / np.maximum(cnt, 1)

M = 561                       # ~2 cells per grid spacing, odd so the diagonal is resolved
g = np.linspace(0.0, 1.0, M)
X, Y = np.meshgrid(g, g)
h = g[1] - g[0]

grids = {}
for k in ("s1", "s2"):
    for fld in ("rho", "u", "v"):
        f = mtri.LinearTriInterpolator(tri, to_nodal(d[f"{k}_{fld}"]))
        Z = np.asarray(f(X, Y)); grids[f"{k}_{fld}"] = np.where(np.isfinite(Z), Z, 0.0)
    du_dy, du_dx = np.gradient(grids[f"{k}_u"], h, h)
    dv_dy, dv_dx = np.gradient(grids[f"{k}_v"], h, h)
    grids[f"{k}_vort"] = dv_dx - du_dy
    grids[f"{k}_divu"] = du_dx + dv_dy               # negative = compression = shock
    grids[f"{k}_gradrho"] = np.hypot(*np.gradient(grids[f"{k}_rho"], h, h))
    print(f"{k}: |vorticity| max={np.abs(grids[f'{k}_vort']).max():.1f}  "
          f"div min={grids[f'{k}_divu'].min():.1f}")

# symmetry residual on the same grid (the reflection is simply the transpose here)
for k in ("s1", "s2"):
    R = grids[f"{k}_rho"]
    grids[f"{k}_res"] = np.abs(R - R.T)

print("\n=== residual stratified by |vorticity| ===")
for k in ("s1", "s2"):
    w = np.abs(grids[f"{k}_vort"]); r = grids[f"{k}_res"] ** 2
    qs = np.percentile(w, [50, 90, 99])
    bins = [(0, qs[0], "|w| below median"), (qs[0], qs[1], "median to p90"),
            (qs[1], qs[2], "p90 to p99"), (qs[2], 1e9, "top 1% |w|")]
    print(f"  {k}  (|w| median={qs[0]:.1f}, p90={qs[1]:.1f}, p99={qs[2]:.1f})")
    for lo, hi, lab in bins:
        m = (w >= lo) & (w < hi)
        print(f"     {lab:20s} n={m.sum():7d} ({100*m.mean():5.1f}% of area)  "
              f"carries {100*r[m].sum()/r.sum():5.1f}% of the squared residual")

print("\n=== residual stratified by compression (shock indicator, div u < 0) ===")
for k in ("s1", "s2"):
    dv = grids[f"{k}_divu"]; r = grids[f"{k}_res"] ** 2
    q = np.percentile(dv, [1, 5])
    for lab, m in (("strong compression (div u < p1)", dv < q[0]),
                   ("moderate (p1 to p5)", (dv >= q[0]) & (dv < q[1])),
                   ("rest", dv >= q[1])):
        print(f"  {k} {lab:34s} n={m.sum():7d} carries {100*r[m].sum()/r.sum():5.1f}%")

print("\n=== joint: high vorticity AND weak compression = a genuine roll ===")
for k in ("s1", "s2"):
    w = np.abs(grids[f"{k}_vort"]); dv = grids[f"{k}_divu"]; r = grids[f"{k}_res"] ** 2
    wq = np.percentile(w, 99); dq = np.percentile(dv, 5)
    roll = (w >= wq) & (dv >= dq)
    shock = dv < dq
    print(f"  {k}: rolls  n={roll.sum():6d} ({100*roll.mean():4.1f}% of area) "
          f"carry {100*r[roll].sum()/r.sum():5.1f}%   |   "
          f"shocks n={shock.sum():6d} ({100*shock.mean():4.1f}%) carry {100*r[shock].sum()/r.sum():5.1f}%")

np.savez_compressed(os.path.join(S, "cfg3_grid.npz"), g=g, **grids)
print("\ncached grid -> cfg3_grid.npz")
