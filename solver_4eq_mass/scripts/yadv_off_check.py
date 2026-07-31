#!/usr/bin/env python3
"""Post-rebuild check: the ACID_YADV-unset dumps must still match the pre-change baseline."""
import os, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
DUMP = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
env = dict(os.environ, DENNER_ACID="1")
env.pop("ACID_YADV", None)
for c in ["01", "02", "14", "25"]:
    out = subprocess.run([DUMP, c], capture_output=True, text=True, env=env, cwd=W).stdout
    base = open(f"/tmp/yadv_base/case{c}.txt").read()
    print(f"case{c}: OFF vs pre-change baseline "
          f"{'BYTE-IDENTICAL' if out == base else 'DIFFERS'} ({len(out)} bytes)")
