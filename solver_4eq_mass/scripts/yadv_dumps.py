#!/usr/bin/env python3
"""Regenerate the ON dumps via subprocess capture (the shell `>` redirect drops output
through the agent harness), then byte-compare ON vs the pre-change alpha baseline."""
import os, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
DUMP = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
for c in ["01", "02", "13", "14", "15", "24", "25", "33", "34"]:
    env = dict(os.environ, DENNER_ACID="1", ACID_YADV="1")
    out = subprocess.run([DUMP, c], capture_output=True, text=True, env=env, cwd=W).stdout
    p = f"/tmp/yadv_on_case{c}.txt"
    open(p, "w").write(out)
    base = f"/tmp/yadv_base/case{c}.txt"
    if os.path.exists(base):
        b = open(base).read()
        same = (b == out)
        print(f"case{c}: {len(out)} bytes, ON vs OFF-baseline "
              f"{'BYTE-IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            nb = sum(1 for x, y in zip(b.splitlines(), out.splitlines()) if x != y)
            print(f"          {nb} of {len(out.splitlines())} rows differ")
    else:
        print(f"case{c}: {len(out)} bytes (no baseline)")
print("--- case01 ON, first and last data row ---")
rows = open("/tmp/yadv_on_case01.txt").read().splitlines()
print(rows[0]); print(rows[1]); print(rows[-1])
