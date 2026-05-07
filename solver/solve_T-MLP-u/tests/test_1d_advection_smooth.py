"""Smoke test 1 — 1D linear advection of a smooth (Gaussian) profile,
periodic domain, single round-trip.

Goal: verify mesh → equation → reconstruction → flux → integrator chain
produces sensible results with two reconstructions:
  - first_order  (large diffusion, expected)
  - minmod_tvd_1d (sharper)

PASS criteria (very loose; structural rather than accuracy):
  - simulation finishes finite
  - mass ∫u dx is conserved (drift < 1e-10 relative)
  - shape correlation with the exact solution > 0.5  (first_order)
  - shape correlation > 0.85                          (minmod_tvd_1d)
"""
from __future__ import annotations
import os
import sys
import numpy as np

from _pkgshim import setup_paths
setup_paths()

from mesh import build_structured_1d
from equations import Advection
from solver import solve


def _run(reconstruction, integrator='ssp_rk2'):
    N = 200
    L = 1.0
    mesh = build_structured_1d(N, L=L, periodic=True)
    eq = Advection(velocity=[1.0])

    x = mesh.cell_centers[:, 0]
    sigma = 0.06
    U0 = np.exp(-((x - 0.3) ** 2) / (2 * sigma ** 2))[None, :]   # (1, N)

    res = solve(mesh, eq, U0,
                reconstruction=reconstruction,
                flux='upwind',
                integrator=integrator,
                cfl=0.5,
                t_end=1.0,    # exactly one period for a=1
                max_steps=100_000)
    U_final = res['U_final'][0]
    return x, U0[0], U_final, res['n_steps']


def main():
    print("Smoke test — 1D advection of a smooth Gaussian, single round-trip")
    print("-" * 70)

    fails = []
    for recon, corr_min in [('first_order', 0.50),
                            ('minmod_tvd_1d', 0.85)]:
        x, U0, U_f, n = _run(recon)
        # Mass conservation
        m0 = float(np.sum(U0));  mf = float(np.sum(U_f))
        m_drift = abs(mf - m0) / max(abs(m0), 1e-30)
        # Shape correlation against the *exact* solution (= U0 after 1 period)
        corr = float(np.corrcoef(U_f, U0)[0, 1])
        max_overshoot = float(np.max(U_f) - np.max(U0))
        max_undershoot = float(np.min(U_f) - np.min(U0))
        print(f"  [{recon:14s}] n_steps={n:4d}  mass drift={m_drift:.2e}  "
              f"corr={corr:+.3f}  Δmax={max_overshoot:+.3e}  Δmin={max_undershoot:+.3e}")

        if not np.isfinite(U_f).all():
            fails.append((recon, 'NaN'))
        elif m_drift > 1e-10:
            fails.append((recon, f'mass drift {m_drift:.2e}'))
        elif corr < corr_min:
            fails.append((recon, f'corr {corr:.3f} < {corr_min}'))

    print("-" * 70)
    if fails:
        print(f"FAIL — {len(fails)} cases did not satisfy the threshold:")
        for r, why in fails:
            print(f"  {r}: {why}")
        return 1
    print("PASS — both reconstructions finite, conservative, well-correlated.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
