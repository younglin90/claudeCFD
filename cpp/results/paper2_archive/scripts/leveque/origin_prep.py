# -*- coding: utf-8 -*-
"""Prepare Origin-ready arrays.

Origin cannot contour 160 000 unstructured triangles directly, so the two fields are
resampled onto a uniform grid with the same linear interpolant used for the matplotlib
figure. The disagreement cells stay as a scatter of centroids, unresampled.
"""
import numpy as np, os
import matplotlib.tri as mtri

from _paths import CACHE_DIR as S
d = np.load(os.path.join(S, "lev_cache.npz"))
pts, cells, cc = d["pts"], d["cells"], d["cc"]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)

def to_nodal(cv):
    acc = np.zeros(len(pts)); cnt = np.zeros(len(pts))
    for j in range(cells.shape[1]):
        np.add.at(acc, cells[:, j], cv); np.add.at(cnt, cells[:, j], 1.0)
    return acc / np.maximum(cnt, 1)

M = 401                                   # 2x the 200x200 mesh, enough for smooth contours
gx = np.linspace(0.0, 1.0, M)
gy = np.linspace(0.0, 1.0, M)
X, Y = np.meshgrid(gx, gy)

out = {"gx": gx, "gy": gy}
for k in ("s1", "s2"):
    f = mtri.LinearTriInterpolator(tri, to_nodal(d[k + "_g"]))
    Z = np.asarray(f(X, Y))
    Z = np.where(np.isfinite(Z), Z, 0.0)
    out[k] = Z
    print(f"{k}: grid {Z.shape}  range [{Z.min():.4f}, {Z.max():.4f}]")

diff = d["s2_g"] - d["s1_g"]
THR = 0.02
m = np.abs(diff) > THR
out["hl_x"] = cc[m, 0]; out["hl_y"] = cc[m, 1]
print(f"disagreement cells above {THR}: {m.sum()} of {len(diff)} "
      f"({100*m.sum()/len(diff):.2f}%)")

np.savez_compressed(os.path.join(S, "origin_lev.npz"), **out)
print("wrote origin_lev.npz")
