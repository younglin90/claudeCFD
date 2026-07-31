# -*- coding: utf-8 -*-
"""Load the paper2 Mach-3 forward-step dumps and resample onto a uniform grid."""
import numpy as np, os
import matplotlib.tri as mtri

R = os.path.join(RESULTS, "paper2_3_data", "mach3")
import os
from _paths import RESULTS, CACHE_DIR
R = RESULTS
S = CACHE_DIR
def read_vtk(path):
    tok = open(path, "r").read().split("\n")
    i = 0; pts = cells = None; fields = {}; name = None
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


raw = {}
for k in ("s1", "s2"):
    pts, cells, f = read_vtk(os.path.join(R, k, "m3.txt.vtk"))
    print(f"{k}: points={len(pts)} cells={cells.shape} fields={sorted(f)} "
          f"rho=[{f['rho'].min():.4f},{f['rho'].max():.4f}] p=[{f['p'].min():.4f},{f['p'].max():.4f}]")
    raw[k] = f
    if "pts" not in raw:
        raw["pts"], raw["cells"] = pts[:, :2], cells

pts, cells = raw["pts"], raw["cells"]
x0, x1 = pts[:, 0].min(), pts[:, 0].max()
y0, y1 = pts[:, 1].min(), pts[:, 1].max()
print(f"\ndomain x [{x0:.2f}, {x1:.2f}]  y [{y0:.2f}, {y1:.2f}]  aspect {(x1-x0)/(y1-y0):.2f}")

tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)

def to_nodal(cv):
    acc = np.zeros(len(pts)); cnt = np.zeros(len(pts))
    for j in range(cells.shape[1]):
        np.add.at(acc, cells[:, j], cv); np.add.at(cnt, cells[:, j], 1.0)
    return acc / np.maximum(cnt, 1)

NY = 421
NX = int(round(NY * (x1 - x0) / (y1 - y0)))
gx = np.linspace(x0, x1, NX); gy = np.linspace(y0, y1, NY)
X, Y = np.meshgrid(gx, gy)
out = {"gx": gx, "gy": gy}
for k in ("s1", "s2"):
    for fld in ("rho", "u", "v", "p"):
        f = mtri.LinearTriInterpolator(tri, to_nodal(raw[k][fld]))
        Z = np.asarray(f(X, Y)); out[f"{k}_{fld}"] = np.where(np.isfinite(Z), Z, np.nan)
    du_dy, du_dx = np.gradient(np.nan_to_num(out[f"{k}_u"]), gy[1]-gy[0], gx[1]-gx[0])
    dv_dy, dv_dx = np.gradient(np.nan_to_num(out[f"{k}_v"]), gy[1]-gy[0], gx[1]-gx[0])
    out[f"{k}_vort"] = dv_dx - du_dy
print(f"grid {NX} x {NY}")

D = np.abs(out["s1_rho"] - out["s2_rho"])
print(f"\n|rho_S1 - rho_S2|: max={np.nanmax(D):.4f}  mean={np.nanmean(D):.4e}  "
      f"rms={np.sqrt(np.nanmean(D**2)):.4e}")
w = np.abs(out["s1_vort"])
for lab, m in (("|w| below median", w < np.percentile(w, 50)),
               ("median to p90", (w >= np.percentile(w, 50)) & (w < np.percentile(w, 90))),
               ("p90 to p99", (w >= np.percentile(w, 90)) & (w < np.percentile(w, 99))),
               ("top 1% |w|", w >= np.percentile(w, 99))):
    print(f"   {lab:18s} mean={np.nanmean(D[m]):.3e}   "
          f"share of squared diff = {100*np.nansum(D[m]**2)/np.nansum(D**2):5.1f}%")

np.savez_compressed(os.path.join(S, "m3_grid.npz"), **out)
print("\ncached -> m3_grid.npz")
