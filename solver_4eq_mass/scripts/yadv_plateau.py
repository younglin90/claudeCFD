#!/usr/bin/env python3
"""TASK B, part 2 -- measure the solver's post-shock plateau on cases 24/33/34 and compare it
against BOTH closures computed by scripts/yadv_hugoniot.cpp.

Cases 24/33/34: domain [0,1], N=800, shock starts at x=0.1 and the reference front is at
x=0.8 at t_end. The post-shock plateau is sampled well away from both the left boundary and
the front.  Nothing here touches cases.cpp / validation.cpp.
"""
import os, re, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
SAMPLE = (0.25, 0.60)   # plateau window in x


def dump(case, yadv):
    env = dict(os.environ, DENNER_ACID="1")
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    exe = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
    out = subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=W).stdout
    return [[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]]


def plateau(rows):
    sel = [r for r in rows if SAMPLE[0] <= r[0] <= SAMPLE[1]]
    n = len(sel)
    cols = {"alpha": 1, "p": 2, "u": 3, "rho": 4}
    out = {}
    for k, c in cols.items():
        v = [r[c] for r in sel]
        out[k] = (sum(v) / n, min(v), max(v))
    return out, n


# ---- parse the two-closure table produced by the standalone Hugoniot program -----------------
hug = subprocess.run(["/tmp/yadv_hugoniot"], capture_output=True, text=True).stdout
closures = {}
cur = None
for ln in hug.splitlines():
    m = re.match(r"=+ case(\d+)", ln)
    if m:
        cur = m.group(1)
        closures[cur] = {}
        continue
    if cur:
        m = re.match(r"(p_post \[Pa\]|rho_post|u_post \[m/s\]|T_post \[K\]|alpha_post|Y_post)\s+"
                     r"([-\d.e+]+)\s+([-\d.e+]+)", ln)
        if m:
            key = m.group(1).split()[0]
            closures[cur][key] = (float(m.group(2)), float(m.group(3)))

KEYMAP = {"p": "p_post", "rho": "rho_post", "u": "u_post", "alpha": "alpha_post"}

for case in ("24", "33", "34"):
    pa, n = plateau(dump(case, False))
    py, _ = plateau(dump(case, True))
    cl = closures[case]
    print(f"\n=== case{case}  post-shock plateau, x in [{SAMPLE[0]},{SAMPLE[1]}]  ({n} cells) ===")
    print("| qty | (A) alpha-held ref | (B) Y-held ref | solver alpha-path | solver Y-path |"
          " Y-path vs (B) | Y-path vs (A) |")
    print("|---|---|---|---|---|---|---|")
    for q in ("p", "rho", "u", "alpha"):
        A, Bv = cl[KEYMAP[q]]
        sa = pa[q][0]
        sy = py[q][0]
        eB = (sy - Bv) / abs(Bv) if Bv else float("nan")
        eA = (sy - A) / abs(A) if A else float("nan")
        print(f"| {q} | {A:.6g} | {Bv:.6g} | {sa:.6g} | {sy:.6g} | {eB:+.2%} | {eA:+.2%} |")
    # spread inside the plateau window -> is it really a plateau?
    sp = {q: (py[q][2] - py[q][1]) / (abs(py[q][0]) + 1e-300) for q in ("p", "rho", "u", "alpha")}
    print("  Y-path plateau relative spread (max-min)/mean: "
          + "  ".join(f"{q}={sp[q]:.2e}" for q in sp))
    spa = {q: (pa[q][2] - pa[q][1]) / (abs(pa[q][0]) + 1e-300) for q in ("p", "rho", "u", "alpha")}
    print("  alpha-path plateau relative spread            : "
          + "  ".join(f"{q}={spa[q]:.2e}" for q in spa))
