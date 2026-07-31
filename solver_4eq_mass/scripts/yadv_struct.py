#!/usr/bin/env python3
"""Coarse wave-structure scan of the Y-path solution on cases 24/33/34."""
import os, subprocess

W = "/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass"
GA, KVA, PIA = 1.4, 720.25, 0.0
GW, KVW, PIW = 4.1, 474.2, 4.4e8


def T_of(p, rho, al):
    return (al * (p + PIA) / (KVA * (GA - 1.0))
            + (1.0 - al) * (p + PIW) / (KVW * (GW - 1.0))) / rho


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
    rows = dump(case, True)
    print(f"\n=== case{case} Y-path profile (every 40th cell) ===")
    print("|   x   |   alpha    |     p      |    u     |   rho    |    T     |     Y      |")
    print("|---|---|---|---|---|---|---|")
    for r in rows[::40] + [rows[-1]]:
        x, al, p, u, rho = r[0], r[1], r[2], r[3], r[4]
        T = T_of(p, rho, al)
        ra = (p + PIA) / (KVA * T * (GA - 1.0))
        Y = al * ra / rho
        print(f"| {x:.3f} | {al:.6f} | {p:.5e} | {u:8.2f} | {rho:8.2f} | {T:8.1f} | {Y:.6e} |")
