#!/usr/bin/env python3
"""Measure the alpha drift the Y-transport path produces, A/B, per case.

For each case: run denner1d_dump with and without ACID_YADV=1, then report
  - alpha range in each run (the alpha model keeps a homogeneous mixture exactly uniform)
  - max |alpha_Y - alpha_alpha| over the grid
This isolates the ONE physical difference between the two closures: alpha is a material
invariant in the alpha model, but a state function alpha(p,T,Y) in the Y model.
"""
import os, subprocess, sys

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
DUMP = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
CASES = ["01", "02", "04", "05", "07", "13", "14", "15", "24", "25",
         "26", "27", "28", "30", "31", "33", "34", "35", "36"]


def run(case, yadv):
    env = dict(os.environ)
    env["DENNER_ACID"] = "1"
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    r = subprocess.run([DUMP, case], capture_output=True, text=True, env=env, cwd=W)
    rows = []
    for ln in r.stdout.splitlines()[1:]:
        f = ln.split(",")
        if len(f) >= 5:
            rows.append([float(v) for v in f[:5]])
    return rows


print(f"{'case':>5} {'alpha_min(a)':>13} {'alpha_max(a)':>13} "
      f"{'alpha_min(Y)':>13} {'alpha_max(Y)':>13} {'max|dalpha|':>12} {'max|drho|/rho':>13}")
for c in CASES:
    a = run(c, False)
    y = run(c, True)
    if not a or not y or len(a) != len(y):
        print(f"{c:>5}  dump mismatch a={len(a)} y={len(y)}")
        continue
    amin = min(r[1] for r in a); amax = max(r[1] for r in a)
    ymin = min(r[1] for r in y); ymax = max(r[1] for r in y)
    dmax = max(abs(r1[1] - r2[1]) for r1, r2 in zip(a, y))
    rmax = max(abs(r1[4] - r2[4]) / max(abs(r1[4]), 1e-300) for r1, r2 in zip(a, y))
    print(f"{c:>5} {amin:13.6g} {amax:13.6g} {ymin:13.6g} {ymax:13.6g} {dmax:12.4g} {rmax:13.4g}")
