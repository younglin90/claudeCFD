# -*- coding: utf-8 -*-
"""Read the paper2 LeVeque-Zalesak dumps (N=200, 160000 triangles) and cache them.

S1 and S2 come from paper2_3_data/leveque/{s1,s2}: same mesh, same 11819 steps, recipes
differing in exactly one flag (THINCQQ_GAUSS). S3 is read as well since it costs nothing
and shares the same mesh.
"""
import numpy as np, os

import os
from _paths import RESULTS, CACHE_DIR
R = RESULTS
S = CACHE_DIR
CACHE = os.path.join(CACHE_DIR, "lev_cache.npz")
VTK = {"s1": os.path.join(R, "paper2_3_data", "leveque", "s1", "lev_bvd.txt.vtk"),
       "s2": os.path.join(R, "paper2_3_data", "leveque", "s2", "lev_bvd.txt.vtk"),
       "s3": os.path.join(R, "paper2_3_data", "leveque", "s3", "lev_bvd.txt.vtk")}


def read_vtk(path):
    """These dumps carry TWO cell fields, 'g' and 'bvd_cand' (BVD_CANDFLAG=1 was set),
       so the scalar blocks must be keyed by name -- reading the last one silently
       returns the candidate index instead of the solution."""
    tok = open(path, "r").read().split("\n")
    i = 0
    pts = cells = None
    fields = {}
    name = None
    while i < len(tok):
        ln = tok[i].strip()
        if ln.startswith("SCALARS"):
            name = ln.split()[1]
            i += 1
            continue
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
                s = tok[i].split()
                if s and not s[0][0].isalpha():
                    buf.extend(s)
                i += 1
            fields[name] = np.array(buf[:len(cells)], dtype=float); continue
        i += 1
    return pts, cells, fields


out = {}
for k, p in VTK.items():
    pts, cells, fields = read_vtk(p)
    vals = fields["g"]
    print(f"{k:6s} points={len(pts)} cells={cells.shape} fields={sorted(fields)} "
          f"g=[{vals.min():.4f},{vals.max():.4f}]")
    out[k + "_g"] = vals
    if "bvd_cand" in fields:
        out[k + "_cand"] = fields["bvd_cand"]
    if "pts" not in out:
        out["pts"], out["cells"] = pts[:, :2], cells
    else:
        assert np.array_equal(out["cells"], cells), f"{k}: different connectivity"

pts, cells = out["pts"], out["cells"]
cc = pts[cells].mean(axis=1)
out["cc"] = cc

def exact(xy):
    x, y = xy[:, 0], xy[:, 1]
    r0 = 0.15
    r1 = np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2) / r0
    in_slot = (np.abs(x - 0.5) < 0.025) & (y < 0.85)
    slot = np.where((r1 <= 1.0) & ~in_slot, 1.0, 0.0)
    r2 = np.sqrt((x - 0.5) ** 2 + (y - 0.25) ** 2) / r0
    cone = np.where(r2 <= 1.0, 1.0 - r2, 0.0)
    r3 = np.sqrt((x - 0.25) ** 2 + (y - 0.5) ** 2) / r0
    hump = np.where(r3 <= 1.0, 0.25 * (1.0 + np.cos(np.pi * r3)), 0.0)
    return slot + cone + hump


out["exact_g"] = exact(cc)
np.savez_compressed(CACHE, **out)
print("cached ->", CACHE)

# reproduce the published metrics as a parsing check (cell areas are uniform here)
print("\n=== L1 against the exact profile (bench prints the same quantity) ===")
for k, ref in (("s1", 3.1816e-03), ("s2", 3.2393e-03), ("s3", 3.3225e-03)):
    L1 = np.abs(out[k + "_g"] - out["exact_g"]).mean()
    print(f"  {k:6s} L1={L1:.4e}   run.log={ref:.4e}   ratio={L1/ref:.4f}")

d = out["s2_g"] - out["s1_g"]
print(f"\nS2 - S1 : max|d|={np.abs(d).max():.4e}  rms={np.sqrt((d**2).mean()):.4e}")
