#!/usr/bin/env python3
"""Task A diagnosis: where does the alpha-space-THINC Y path lose case02?

case02 is a pure gas_a|gas_b contact advected at u=1 with UNIFORM (p,T), so Y-transport
and alpha-transport are the SAME PDE and every difference is discretisation-only.
Prints the interface band of the alpha path and the Y path against the exact reference.
"""
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
    rows = [ln.split(",") for ln in out.strip().splitlines()[1:]]
    return [[float(v) for v in r] for r in rows]


off = dump("02", False)
on = dump("02", True)

# reference contact position: rho_ref switches
ra = off[0][4]
rb = off[-1][4]
print(f"gas_a rho = {ra:.6g}   gas_b rho = {rb:.6g}   ratio = {ra/rb:.6g}")

# locate the reference jump
jref = max(range(1, len(off)), key=lambda i: abs(off[i][7] - off[i - 1][7]))
print(f"reference contact between cells {jref-1}/{jref} at x ~ {off[jref][0]:.4f}")

lo, hi = max(0, jref - 8), min(len(off), jref + 9)
print()
print("| i | x | alpha OFF | alpha ON | rho OFF | rho ON | rho_ref |")
print("|---|---|---|---|---|---|---|")
for i in range(lo, hi):
    print(f"| {i} | {off[i][0]:.4f} | {off[i][1]:.6g} | {on[i][1]:.6g} | "
          f"{off[i][4]:.6g} | {on[i][4]:.6g} | {off[i][7]:.6g} |")


def band(rows, col=1):
    """cells strictly between the pure states -> width of the smeared interface"""
    return [i for i, r in enumerate(rows) if 1e-9 < r[col] < 1 - 1e-9]


for nm, rows in (("alpha path", off), ("Y path (alpha-THINC)", on)):
    b = band(rows)
    print(f"{nm}: mixed-alpha cells = {len(b)}"
          + (f", indices {b[0]}..{b[-1]}" if b else ""))


def l1rho(rows):
    num = sum(abs(r[4] - r[7]) for r in rows)
    den = sum(abs(r[7]) for r in rows)
    return num / den


print(f"l1_rho  alpha={l1rho(off):.6g}   Y={l1rho(on):.6g}")

# mass-fraction cell value implied by each path's alpha, at uniform (p,T)
print()
print("Y implied by each path's alpha (uniform p,T -> Y = a*ra/(a*ra+(1-a)*rb)):")
for i in range(lo, hi):
    aoff, aon = off[i][1], on[i][1]
    yoff = aoff * ra / (aoff * ra + (1 - aoff) * rb)
    yon = aon * ra / (aon * ra + (1 - aon) * rb)
    print(f"  i={i} alpha {aoff:.6g}/{aon:.6g}   Y {yoff:.6g}/{yon:.6g}")
