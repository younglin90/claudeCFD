"""S1 — uniform flow byte-exact preservation.

Goal: verify the v2 R1 baseline does NOT generate spurious oscillations
on a uniform background state (W = const, periodic BC).  Conservative
fluxes should cancel exactly; cons_to_prim Newton round-off limits the
final precision.

Pass:  ‖W_after − W_before‖∞ ≤ 1e-12 (relative to scale of each field)
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX_v2 import solve, IdealEOS, SGEOS, NASGEOS


def _run(label, eos1, eos2, W_const, dx, t_end, cfl=0.4, dt_fixed=None,
         bc=('periodic', 'periodic'), max_steps=200):
    N = W_const[0].shape[0]
    res = solve(eos1, eos2, tuple(c.copy() for c in W_const), dx, t_end,
                cfl=cfl, dt_fixed=dt_fixed,
                bc_l=bc[0], bc_r=bc[1],
                max_steps=max_steps)
    W = res['W_final']
    diffs = []
    names = ['α', 'T1', 'T2', 'u', 'p']
    for k, n in enumerate(names):
        ref = W_const[k]
        scale = max(float(np.max(np.abs(ref))), 1.0)
        d = float(np.max(np.abs(W[k] - ref))) / scale
        diffs.append((n, d))
    print(f"  [{label}] n_steps={res['n_steps']}, t={res['t']:.3e}")
    for n, d in diffs:
        print(f"    Δ{n}/scale = {d:.3e}")
    return diffs


def main():
    print("S1 uniform-flow byte-exact preservation (v2 R1)")
    print("-" * 64)
    fails = []

    # Case A: ideal-ideal (air-air), u = 0
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N = 16; dx = 1.0 / N
    W = (np.full(N, 0.3), np.full(N, 300.0), np.full(N, 300.0),
         np.full(N, 0.0),  np.full(N, 1e5))
    diffs = _run('A: ideal-ideal, u=0', eos1, eos2, W, dx,
                 t_end=1e-3, cfl=0.4, max_steps=2000)
    for n, d in diffs:
        if d > 1e-12:
            fails.append(('A', n, d))

    # Case B: ideal-SG (air-water), u = 1.0 m/s (Galilean)
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = SGEOS(gamma=4.1, pinf=4.4e8, kv=474.2)
    N = 16; dx = 1.0 / N
    W = (np.full(N, 0.5), np.full(N, 300.0), np.full(N, 300.0),
         np.full(N, 1.0),  np.full(N, 1e5))
    diffs = _run('B: ideal-SG, u=1', eos1, eos2, W, dx,
                 t_end=2e-5, cfl=0.4, max_steps=2000)
    for n, d in diffs:
        if d > 1e-12:
            fails.append(('B', n, d))

    # Case C: ideal-NASG (air-water Le Métayer), u = 0
    eos1 = IdealEOS(gamma=1.4, kv=717.5)
    eos2 = NASGEOS()
    N = 16; dx = 1.0 / N
    W = (np.full(N, 0.5), np.full(N, 300.0), np.full(N, 300.0),
         np.full(N, 0.0),  np.full(N, 1e5))
    diffs = _run('C: ideal-NASG, u=0', eos1, eos2, W, dx,
                 t_end=2e-5, cfl=0.4, max_steps=2000)
    for n, d in diffs:
        if d > 1e-12:
            fails.append(('C', n, d))

    print("-" * 64)
    if fails:
        print(f"S1 FAIL — {len(fails)} field(s) over 1e-12 threshold:")
        for c, n, d in fails:
            print(f"  case {c} field {n}: {d:.3e}")
        return 1
    print("S1 PASS — uniform-flow byte-exact (≤1e-12).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
