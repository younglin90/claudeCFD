#!/usr/bin/env python3
"""Where is the shock front at t_end on cases 24/33/34, alpha path vs Y path?"""
import os, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"


def dump(case, yadv):
    env = dict(os.environ, DENNER_ACID="1")
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    exe = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
    out = subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=W).stdout
    return [[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]]


for case in ("24", "33", "34"):
    for nm, y in (("alpha", False), ("Y    ", True)):
        rows = dump(case, y)
        p0 = 1.0e5
        shocked = [r for r in rows if r[2] > 2.0 * p0]
        front = shocked[-1][0] if shocked else float("nan")
        print(f"case{case} {nm}: p(first)={rows[0][2]:.4e}  p(last)={rows[-1][2]:.4e}  "
              f"u(last)={rows[-1][3]:.2f}  last shocked x={front:.3f}  "
              f"ref front x=0.800  n_shocked={len(shocked)}/{len(rows)}")
    print()
