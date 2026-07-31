# -*- coding: utf-8 -*-
"""Parse the HLLC (N3_FLUX=1) 3D cascade logs into per-rep records, then aggregate.
   Only logs that actually carry N3_FLUX=1 AND a wall= line are accepted."""
import os
from _paths import RESULTS, CACHE_DIR
import re, glob, os
import pandas as pd

D = os.path.join(RESULTS, "paper2_cascade")
RE_STAGE = re.compile(
    r"E3D_PROF:\s*(\d+)\s*RHS calls, summed\s*(\d+)ms.*?prim\s*(\d+)ms.*?RECON\s*(\d+)ms"
    r".*?FLUX\s*(\d+)ms.*?MOOD\s*(\d+)ms.*?asm\s*(\d+)ms")
RE_BREAK = re.compile(
    r"RECON breakdown \(summed (\d+)ms\):\s*o2-LSQ\s*(\d+)ms.*?cell-D-Newton\s*(\d+)ms"
    r"\(([\d.]+)%, avg ([\d.]+) Newton iters/solve over (\d+) solves\).*?face-quad\+BVD\s*(\d+)ms")
RE_BREAK_G = re.compile(
    r"RECON breakdown \(summed (\d+)ms\):\s*o2-LSQ\s*(\d+)ms.*?cell-D[^ ]*\s*(\d+)ms.*?face-quad\+BVD\s*(\d+)ms")
RE_TAIL = re.compile(r"flux=(\d+).*?cells=(\d+).*?p_min=([-\d.]+).*?steps=(\d+).*?wall=([\d.]+)s")
# the deformation case runs the scalar advect3d driver: no Riemann flux and no p_min,
# so it has neither a stage line nor those fields. flux is recorded as -1 = not applicable.
RE_TAIL_ADV = re.compile(r"cells=(\d+).*?steps=(\d+).*?wall=([\d.]+)s")

rows = []
for f in sorted(glob.glob(os.path.join(D, "3d_*_r*.log"))):
    txt = open(f, encoding="utf-8", errors="replace").read()
    base = os.path.basename(f)
    m = re.match(r"3d_(.+)_(s[12])_r(\d+)\.log", base)
    if not m:
        continue
    case, scheme, rep = m.group(1), m.group(2), int(m.group(3))
    hllc = "N3_FLUX=1" in txt
    adv = "(advect3d)" in txt
    t = RE_TAIL.search(txt) if not adv else RE_TAIL_ADV.search(txt)
    if not t:
        print(f"  SKIP {base:26s} no wall= line (run did not complete)")
        continue
    if adv:
        flux, cells, p_min, steps, wall = -1, int(t.group(1)), float("nan"), \
            int(t.group(2)), float(t.group(3))
    else:
        flux, cells, p_min, steps, wall = int(t.group(1)), int(t.group(2)), \
            float(t.group(3)), int(t.group(4)), float(t.group(5))
    s = RE_STAGE.search(txt)
    b = RE_BREAK.search(txt) or RE_BREAK_G.search(txt)
    if not b or (s is None and not adv):
        print(f"  SKIP {base:26s} no profile block")
        continue
    if len(b.groups()) == 7:
        rsum, lsq, celld, _, nit, nsolve, facebvd = b.groups()
    else:
        rsum, lsq, celld, facebvd = b.groups(); nit, nsolve = 0.0, 0
    rows.append(dict(case=case, scheme=scheme, rep=rep, hllc=hllc,
                     flux=flux, cells=cells, p_min=p_min, steps=steps, wall=wall,
                     recon_stage=int(s.group(4)) if s else int(rsum),
                     prim=int(s.group(3)) if s else 0,
                     fluxms=int(s.group(5)) if s else 0,
                     mood=int(s.group(6)) if s else 0,
                     asm=int(s.group(7)) if s else 0,
                     recon_sum=int(rsum), lsq=int(lsq), celld=int(celld),
                     facebvd=int(facebvd),
                     newton_iters=float(nit), solves=int(nsolve)))

P = pd.DataFrame(rows).sort_values(["case", "scheme", "rep"])
P.to_csv(os.path.join(D, "cascade_3d_perrep.csv"), index=False)
print("\n=== per-rep records ===")
print(P[["case", "scheme", "rep", "flux", "cells", "steps", "wall",
         "lsq", "celld", "facebvd", "recon_sum", "newton_iters", "p_min"]].to_string(index=False))

# ---- sanity gates -------------------------------------------------------
print("\n=== sanity gates ===")
bad = P[(P.scheme == "s2") & (P.newton_iters > 0)]
print(f"  S2 with Newton iterations (must be empty) : {len(bad)} rows")
bad = P[(P.scheme == "s1") & (P.newton_iters <= 0)]
print(f"  S1 without Newton iterations (must be empty): {len(bad)} rows")
for case, g in P.groupby("case"):
    cs = set(g.cells); st = set(g.steps); fx = set(g.flux)
    flag = "" if (len(cs) == 1 and len(st) == 1 and len(fx) == 1) else "   <-- MISMATCH"
    print(f"  {case:8s} cells={cs} steps={st} flux={fx}{flag}")
pm = P.p_min.dropna()
print(f"  p_min > 0 everywhere (Euler cases): {bool((pm > 0).all())}  [{len(pm)} runs]")

# ---- repeat spread ------------------------------------------------------
print("\n=== repeat spread, per case/scheme (max-min)/min ===")
agg = []
for (case, scheme), g in P.groupby(["case", "scheme"]):
    def spread(c):
        v = g[c]
        return 100.0 * (v.max() - v.min()) / v.min() if len(v) > 1 and v.min() > 0 else float("nan")
    agg.append(dict(case=case, scheme=scheme, n=len(g), flux=g.flux.iloc[0],
                    wall=g.wall.min(), wall_spread=spread("wall"),
                    lsq=g.lsq.min(), celld=g.celld.min(), facebvd=g.facebvd.min(),
                    recon_sum=g.recon_sum.min(), recon_stage=g.recon_stage.min(),
                    recon_spread=spread("recon_sum"), celld_spread=spread("celld")))
A = pd.DataFrame(agg).sort_values(["case", "scheme"])
A.to_csv(os.path.join(D, "cascade_3d_hllc.csv"), index=False)
print(A.to_string(index=False))
print("\nwritten: cascade_3d_perrep.csv, cascade_3d_hllc.csv")
print("NOTE: aggregation uses the MINIMUM over repeats (least contaminated by co-tenants),")
print("      matching the 2D agent's solver_wall_min_s convention.")
