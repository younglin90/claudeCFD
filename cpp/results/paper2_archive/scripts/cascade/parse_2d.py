# -*- coding: utf-8 -*-
"""Re-aggregate the 2D cascade from the per-rep logs with the same minimum-over-repeats
   convention used for 3D, so the two dimensions are treated identically."""
import os
from _paths import RESULTS, CACHE_DIR
import re, os, glob
import pandas as pd

D = os.path.join(RESULTS, "paper2_cascade")
RE_PROF = re.compile(
    r"\[CHENG3_PROF\]\s*recon_calls=(\d+)\s+MUSCL=([\d.]+)s\s+THINC=([\d.]+)s\s*"
    r"\(geom=([\d.]+)s face=([\d.]+)s\)\s+BVD_sel=([\d.]+)s")
RE_WALL = re.compile(r"\[WALL\]\s+(\S+)\s+wall=([\d.]+)s")

# which [WALL] label carries the scheme under test in each bench
TARGET = {"leveque": "BVD", "shockmixing": None, "shockvortex": "shockvortex_2d",
          "mach3": "BVD", "config3": "T-MLP-u", "doublemach": "BVD"}

rows = []
for rep in (1, 2, 3):
    for f in sorted(glob.glob(os.path.join(D, f"rep{rep}", "2d_*.log"))):
        base = os.path.basename(f)
        if "FROZEN" in base:                       # the frozen HLLC_PVRS mach3 attempt
            print(f"  EXCLUDE {base} (frozen run)")
            continue
        m = re.match(r"2d_(.+)_(s[12])\.log", base)
        if not m:
            continue
        case, scheme = m.group(1), m.group(2)
        txt = open(f, encoding="utf-8", errors="replace").read()
        p = RE_PROF.search(txt)
        if not p:
            print(f"  SKIP {base} rep{rep}: no CHENG3_PROF")
            continue
        calls, muscl, thinc, geom, face, bvdsel = (int(p.group(1)),) + tuple(
            float(p.group(i)) for i in range(2, 7))
        walls = dict((a, float(b)) for a, b in RE_WALL.findall(txt))
        tgt = TARGET[case]
        wall = walls.get(tgt) if tgt else None
        if wall is None:                            # bench prints no per-scheme wall
            wall = walls.get("TOTAL")
        rows.append(dict(case=case, scheme=scheme, rep=rep, recon_calls=calls,
                         muscl=muscl, thinc=thinc, geom=geom, face=face, bvd_sel=bvdsel,
                         recon_sum=muscl + thinc + bvdsel, wall=wall))

P = pd.DataFrame(rows).sort_values(["case", "scheme", "rep"])
P.to_csv(os.path.join(D, "cascade_2d_perrep.csv"), index=False)
print("\n=== per-rep records ===")
print(P.to_string(index=False))

print("\n=== gate: recon_calls identical everywhere (attribution check) ===")
print(f"  distinct recon_calls values: {sorted(set(P.recon_calls))}")
print("  (a single value means the profile counts only the cheng3 path, once per RK stage,")
print("   so it is attributable to the scheme under test even in benches that run several)")

print("\n=== gate: profile sum must not exceed the scheme wall ===")
ok = True
for _, r in P.iterrows():
    if r.wall is not None and r.recon_sum > r.wall:
        print(f"  VIOLATION {r.case} {r.scheme} rep{r.rep}: recon {r.recon_sum:.1f} > wall {r.wall:.1f}")
        ok = False
print(f"  clean: {ok}")

agg = []
for (case, scheme), g in P.groupby(["case", "scheme"]):
    def spread(c):
        v = g[c].dropna()
        return 100.0 * (v.max() - v.min()) / v.min() if len(v) > 1 and v.min() > 0 else float("nan")
    agg.append(dict(case=case, scheme=scheme, n=len(g),
                    muscl=g.muscl.min(), geom=g.geom.min(), face=g.face.min(),
                    thinc=g.thinc.min(), bvd_sel=g.bvd_sel.min(),
                    recon_sum=g.recon_sum.min(), wall=g.wall.min(),
                    wall_spread=spread("wall"), recon_spread=spread("recon_sum"),
                    geom_spread=spread("geom")))
A = pd.DataFrame(agg).sort_values(["case", "scheme"])
A.to_csv(os.path.join(D, "cascade_2d_min.csv"), index=False)
print("\n=== minimum over repeats, with spread ===")
print(A.to_string(index=False))
print("\nwritten: cascade_2d_perrep.csv, cascade_2d_min.csv")
