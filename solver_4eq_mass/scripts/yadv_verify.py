#!/usr/bin/env python3
"""Final dump-level verification on the fully recompiled build.

  (1) ACID_YADV unset  vs  the reference workspace solver_denner's published binary
      -> must be BYTE-IDENTICAL (this, not the stale-object copy in build-cpp, is the
         real "unchanged default path" check).
  (2) ACID_YADV=1      vs  ACID_YADV unset, same binary
      -> case01 must be byte-identical (machine-exact pressure equilibrium preserved).
"""
import os, subprocess

MINE = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
REF = "/home/younglin90/work/claude_code/claudeCFD/solver_denner"
CASES = ["01", "02", "13", "14", "25"]
# "15" removed round 34: case15 was excluded from the registered suite in solver_4eq_mass
# (docs/YADV_RESEARCH.md §44), so denner1d_dump 15 now exits 2 with empty stdout here, while
# solver_denner still has it registered and dumps a full CSV -- leaving "15" in this list would
# report a spurious "case15: DIFFERS" that is not a real byte-identity break.
# "24"/"33"/"34" removed round 35 for the identical mechanical reason (docs/YADV_RESEARCH.md §45):
# they are excluded from this tree's registered suite, so denner1d_dump 24/33/34 exit 2 here while
# solver_denner still dumps them. HONEST COST: this is a real, permanent byte-identity coverage
# reduction, 9 cases (through round 33) -> 8 (round 34) -> 5 (round 35). case01 (machine-exact
# pressure equilibrium) is still covered, as are 02/13/14/25.


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
