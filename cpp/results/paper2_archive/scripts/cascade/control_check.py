# -*- coding: utf-8 -*-
"""The shared-work bucket does identical work in S1 and S2, so its ratio must be 1.0.
   Whatever it isn't is measurement noise -- an internal control for the cascade."""
import os
from _paths import RESULTS, CACHE_DIR
import pandas as pd

C = os.path.join(RESULTS, "paper2_cascade")
d2 = pd.read_csv(C + r"\cascade_2d.csv")
d3 = pd.read_csv(C + r"\cascade_3d.csv")

print("=== 2D : shared-work control (ideal 1.000) ===")
print(f"  {'case':13s} {'control':>8s} {'accel':>8s} {'solver':>8s} {'corrected':>10s} {'rep spread':>11s}")
for c in ["leveque", "shockmixing", "shockvortex", "mach3", "config3", "doublemach"]:
    a = d2[(d2.case == c) & (d2.scheme == "s1")].iloc[0]
    b = d2[(d2.case == c) & (d2.scheme == "s2")].iloc[0]
    ctl = (a.muscl_s + a.bvd_sel_s) / (b.muscl_s + b.bvd_sel_s)
    acc = (a.geom_s + a.face_s) / (b.geom_s + b.face_s)
    sol = a.solver_wall_s / b.solver_wall_s
    print(f"  {c:13s} {ctl:8.3f} {acc:8.3f} {sol:8.3f} {sol/ctl:10.3f} {a.solver_wall_spread_pct:10.1f}%")

print("\n=== 3D : shared-work control = LSQ bucket ===")
print(f"  {'case':13s} {'control':>8s} {'accel':>8s} {'solver':>8s} {'corrected':>10s}")
for c in ["deform", "sphere", "2cyl"]:
    a = d3[(d3.case == c) & (d3.scheme == "s1")].iloc[0]
    b = d3[(d3.case == c) & (d3.scheme == "s2")].iloc[0]
    ctl = a.lsq / b.lsq
    acc = (a.celld + a.facebvd) / (b.celld + b.facebvd)
    sol = a.wall / b.wall
    print(f"  {c:13s} {ctl:8.3f} {acc:8.3f} {sol:8.3f} {sol/ctl:10.3f}")
