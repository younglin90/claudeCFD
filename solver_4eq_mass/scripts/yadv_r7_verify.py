#!/usr/bin/env python3
"""Final dump-level verification on the fully recompiled build.

  (1) ACID_YADV unset  vs  the reference workspace solver_denner's published binary
      -> must be BYTE-IDENTICAL (this, not the stale-object copy in build-cpp, is the
         real "unchanged default path" check).
  (2) ACID_YADV=1      vs  ACID_YADV unset, same binary
      -> case01 must be byte-identical (machine-exact pressure equilibrium preserved).
"""
import os, subprocess

MINE = "/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-7/solver_4eq_mass"  # round-7: worktree, not main tree
REF = "/home/younglin90/work/claude_code/claudeCFD/solver_denner"
CASES = ["01", "02", "13", "14", "15", "24", "25", "33", "34"]


def dump(root, case, yadv=False):
    env = dict(os.environ, DENNER_ACID="1")
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    exe = os.path.join(root, "build-cpp/cpp/denner_1d/denner1d_dump")
    return subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=root).stdout


print("(1) ACID_YADV unset vs solver_denner published binary")
for c in CASES:
    m, r = dump(MINE, c), dump(REF, c)
    print(f"    case{c}: {'BYTE-IDENTICAL' if m == r else 'DIFFERS'}")

print("(2) ACID_YADV=1 vs ACID_YADV unset (same binary)")
for c in CASES:
    off, on = dump(MINE, c), dump(MINE, c, yadv=True)
    if off == on:
        print(f"    case{c}: BYTE-IDENTICAL")
    else:
        ol, nl = off.splitlines(), on.splitlines()
        nd = sum(1 for a, b in zip(ol, nl) if a != b)
        da = max(abs(float(a.split(",")[1]) - float(b.split(",")[1]))
                 for a, b in zip(ol[1:], nl[1:]))
        print(f"    case{c}: differs, {nd}/{len(nl)-1} rows, max|d alpha|={da:.4g}")
