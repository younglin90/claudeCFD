"""Smoke test 2 — 1D Euler Sod shock tube.

Initial state (Sod 1978):
  Left  (x < 0.5):  ρ=1.0,    u=0,  p=1.0
  Right (x ≥ 0.5):  ρ=0.125,  u=0,  p=0.1
  γ = 1.4

Solve to t=0.2, transmissive BCs both ends.

PASS criteria (sanity):
  - simulation finishes finite
  - max(ρ) ≤ 1.05 and min(ρ) ≥ 0.10  (no large overshoot/undershoot)
  - p_max ≤ 1.05 and p_min ≥ 0.05
  - First-order recon gives the well-known smeared profile (no NaN).
  - Minmod TVD recon is sharper at the shock front (max gradient larger).
"""
from __future__ import annotations
import os, sys
import numpy as np

from _pkgshim import setup_paths
setup_paths()

from mesh import build_structured_1d
from equations import Euler1D
from boundary import BoundaryCondition
from solver import solve


def _run(reconstruction):
    N = 200; L = 1.0
    mesh = build_structured_1d(N, L=L, periodic=False)
    eq = Euler1D(gamma=1.4)

    x = mesh.cell_centers[:, 0]
    rho = np.where(x < 0.5, 1.0,   0.125)
    u   = np.zeros(N)
    p   = np.where(x < 0.5, 1.0,   0.1)
    W0  = np.stack([rho, u, p], axis=0)
    U0  = eq.prim_to_cons(W0)

    bc = {
        'left':  BoundaryCondition('transmissive'),
        'right': BoundaryCondition('transmissive'),
    }
    res = solve(mesh, eq, U0,
                reconstruction=reconstruction,
                flux='hllc',
                integrator='ssp_rk2',
                bc=bc,
                cfl=0.4,
                t_end=0.2,
                max_steps=100_000)
    U_final = res['U_final']
    W_final = eq.cons_to_prim(U_final)
    return x, W_final, res['n_steps']


def main():
    print("Smoke test — 1D Euler Sod shock tube (γ=1.4, t=0.2)")
    print("-" * 70)
    fails = []
    profiles = {}
    for recon in ['first_order', 'minmod_tvd_1d']:
        x, W, n = _run(recon)
        rho_min = float(np.min(W[0])); rho_max = float(np.max(W[0]))
        p_min   = float(np.min(W[2])); p_max   = float(np.max(W[2]))
        # Sharpness proxy — max |∂x ρ| via finite differences
        sharpness = float(np.max(np.abs(np.diff(W[0]))))
        finite = bool(np.all(np.isfinite(W)))
        print(f"  [{recon:14s}] n_steps={n:4d}  "
              f"ρ∈[{rho_min:.4f}, {rho_max:.4f}]  "
              f"p∈[{p_min:.4f}, {p_max:.4f}]  "
              f"max|Δρ|={sharpness:.3f}")
        if not finite:
            fails.append((recon, 'NaN'))
        elif rho_max > 1.05 or rho_min < 0.10:
            fails.append((recon, f'rho∈[{rho_min:.3f}, {rho_max:.3f}]'))
        elif p_max > 1.05 or p_min < 0.05:
            fails.append((recon, f'p∈[{p_min:.3f}, {p_max:.3f}]'))
        profiles[recon] = (x, W, sharpness)

    # Sharpness ordering: minmod TVD should be at least as sharp as 1st order
    if 'first_order' in profiles and 'minmod_tvd_1d' in profiles:
        sh_fo = profiles['first_order'][2]
        sh_tv = profiles['minmod_tvd_1d'][2]
        print(f"  sharpness ratio (TVD / 1st-order) = {sh_tv / max(sh_fo, 1e-30):.2f}")
        if sh_tv < 0.9 * sh_fo:
            fails.append(('minmod_tvd_1d',
                          f'TVD less sharp than 1st-order ({sh_tv:.3f} < {sh_fo:.3f})'))

    print("-" * 70)
    if fails:
        print(f"FAIL — {len(fails)} sanity violations:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — Sod profile within physical bounds; TVD ≥ 1st-order sharpness.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
