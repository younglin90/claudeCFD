#!/usr/bin/env python3
"""TASK B, decisive check -- is the solver's own leading shock a valid Rankine-Hugoniot jump?

Takes the UNDISTURBED pre-shock state and the plateau immediately behind the leading front
directly out of the dump, infers the shock speed from MASS conservation, and then reports the
MOMENTUM and ENERGY residuals.  This is closure-agnostic: any valid weak solution of the
Euler system must satisfy all three, whatever colour function is transported.

Reference check built in: the alpha path must come out ~0, which validates the algebra.

NASG, b=0 and eta=0 for both phases here:
   rho_k = (p+pinf_k)/(kv_k T (g_k-1)),   h_k = g_k kv_k T
"""
import os, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
GA, KVA, PIA, ETA_A = 1.4, 720.25, 0.0, 0.0
GW, KVW, PIW, ETA_W = 4.1, 474.2, 4.4e8, 0.0


def T_of(p, rho, al):
    return (al * (p + PIA) / (KVA * (GA - 1.0))
            + (1.0 - al) * (p + PIW) / (KVW * (GW - 1.0))) / rho


def h_of(p, rho, al):
    T = T_of(p, rho, al)
    ra = (p + PIA) / (KVA * T * (GA - 1.0))
    rb = (p + PIW) / (KVW * T * (GW - 1.0))
    ha = GA * KVA * T + ETA_A
    hb = GW * KVW * T + ETA_W
    Y = al * ra / rho
    return Y * ha + (1.0 - Y) * hb, T, Y


def dump(case, yadv):
    env = dict(os.environ, DENNER_ACID="1")
    if yadv:
        env["ACID_YADV"] = "1"
    else:
        env.pop("ACID_YADV", None)
    exe = os.path.join(W, "build-cpp/cpp/denner_1d/denner1d_dump")
    out = subprocess.run([exe, case], capture_output=True, text=True, env=env, cwd=W).stdout
    return [[float(v) for v in ln.split(",")] for ln in out.strip().splitlines()[1:]]


print("Rankine-Hugoniot residuals across the solver's own LEADING shock")
print("(pre-shock stationary; Vs inferred from MASS conservation)\n")
print("| case | path | p_post | rho_post | u_post | Vs(mass) | momentum resid (rel) "
      "| energy resid (rel) |")
print("|---|---|---|---|---|---|---|---|")

for case in ("24", "33", "34"):
    for nm, y in (("alpha", False), ("Y", True)):
        rows = dump(case, y)
        p0 = 1.0e5
        undis = [r for r in rows if r[2] < 1.5 * p0]
        if not undis:
            print(f"| {case} | {nm} | -- shock has LEFT the domain, no undisturbed state -- |")
            continue
        i_front = rows.index(undis[0])          # first undisturbed cell
        pre = undis[len(undis) // 2]
        # plateau behind the front: back off 40 cells to clear the numerical shock structure
        post = rows[max(i_front - 60, 0)]
        rho0, p_0, u0, al0 = pre[4], pre[2], pre[3], pre[1]
        rho1, p_1, u1, al1 = post[4], post[2], post[3], post[1]
        h0, T0, Y0 = h_of(p_0, rho0, al0)
        h1, T1, Y1 = h_of(p_1, rho1, al1)
        Vs = rho1 * (u1 - u0) / (rho1 - rho0) + u0
        mom = (p_1 - p_0) - rho0 * (Vs - u0) * (u1 - u0)
        w0, w1 = Vs - u0, Vs - u1
        ene = (h1 + 0.5 * w1 * w1) - (h0 + 0.5 * w0 * w0)
        print(f"| {case} | {nm} | {p_1:.5e} | {rho1:.2f} | {u1:.1f} | {Vs:.1f} "
              f"| {mom:+.3e} ({mom/max(abs(p_1),1):+.2e}) "
              f"| {ene:+.3e} ({ene/max(abs(h1),1):+.2e}) |")
