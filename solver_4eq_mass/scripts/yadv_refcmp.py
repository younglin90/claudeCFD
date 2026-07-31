#!/usr/bin/env python3
"""Which build reproduces the paper state? Compare, for the alpha path only:
   (a) the pre-change baseline dumps captured from this workspace's ORIGINAL object files,
   (b) this workspace's dumps after a full recompile,
   (c) the reference workspace solver_denner's EXISTING binary (run-only, never rebuilt).
"""
import os, subprocess, difflib

MINE = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
REF = "/home/younglin90/work/claude_code/claudeCFD/solver_denner"
env = dict(os.environ, DENNER_ACID="1")
env.pop("ACID_YADV", None)


def dump(root, case):
    exe = os.path.join(root, "build-cpp/cpp/denner_1d/denner1d_dump")
    if not os.path.exists(exe):
        return None
    return subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=root).stdout


for c in ["01", "02", "14", "25"]:
    base = open(f"/tmp/yadv_base/case{c}.txt").read()
    mine = dump(MINE, c)
    ref = dump(REF, c)
    print(f"--- case{c} ---")
    print(f"  baseline(orig objs) vs mine(full recompile): "
          f"{'SAME' if base == mine else 'DIFF'}")
    if ref is None:
        print("  solver_denner binary: NOT PRESENT")
        continue
    print(f"  solver_denner(paper)  vs baseline           : {'SAME' if ref == base else 'DIFF'}")
    print(f"  solver_denner(paper)  vs mine               : {'SAME' if ref == mine else 'DIFF'}")
    if ref != mine:
        bl = base.splitlines(); ml = mine.splitlines(); rl = ref.splitlines()
        n = sum(1 for a, b in zip(rl, ml) if a != b)
        print(f"    rows differing (ref vs mine): {n}/{len(ml)}")
        for a, b in zip(rl, ml):
            if a != b:
                print(f"    ref : {a}")
                print(f"    mine: {b}")
                break
