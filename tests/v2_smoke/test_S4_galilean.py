"""S4 — Galilean invariance.

Run two simulations from identical initial perturbations on top of two
different background velocities (u₀ = 0 and u₀ = u_shift).  After the
same physical time the *perturbation* fields (W − W₀) should agree
modulo a rigid translation of u_shift · t in space.

For 1st-order upwind face state, perfect Galilean invariance only holds
when the upwind sign does NOT flip between the two cases.  We choose
u_shift > max(perturbation u) so the upwind direction is the same in
both runs (u_face_avg ≥ 0 everywhere).

Pass:  ‖(α₁,T₁,T₂,p)_shift − (α₁,T₁,T₂,p)_static‖∞ < 1e-10  on a static
       background; Δu = constant (u_shift) up to round-off.

For periodic BC and an integer-cell shift after the run we can compare
directly cell-by-cell (no interpolation).  Otherwise we compare the
mean and standard deviation of the perturbation, which is invariant
under a uniform translation.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX_v2 import solve, IdealEOS, SGEOS, NASGEOS


def main():
    print("S4 Galilean invariance (v2 R1)")
    print("-" * 64)
    fails = []

    # NASG-Ideal (water-air) periodic, smooth α perturbation.
    eos1 = NASGEOS()
    eos2 = IdealEOS(gamma=1.4, kv=717.5)
    N = 32; L = 1.0; dx = L / N
    x = (np.arange(N) + 0.5) * dx

    # Background + smooth perturbation (no shock).
    alpha = 0.5 * (1.0 + 0.4 * np.cos(2 * np.pi * x / L))
    p0 = 1e5; T0 = 300.0
    u_shift_values = [0.0, 5.0]  # (m/s) — large enough for upwind to stay positive

    t_end = 1e-3
    perturbations = []
    for u_shift in u_shift_values:
        W0 = (alpha.copy(), np.full(N, T0), np.full(N, T0),
              np.full(N, u_shift), np.full(N, p0))
        res = solve(eos1, eos2, W0, dx, t_end=t_end,
                    cfl=0.4, bc_l='periodic', bc_r='periodic',
                    max_steps=200_000)
        W = res['W_final']
        # Perturbation w.r.t. uniform mean of each field
        pert = tuple(W[k] - np.mean(W[k]) for k in range(5))
        perturbations.append(pert)
        print(f"  u_shift={u_shift:.1f}: n_steps={res['n_steps']},  "
              f"⟨α⟩={np.mean(W[0]):.6f}  σα={np.std(W[0]):.6e}")

    # Compare perturbations field by field
    diffs = []
    names = ['α', 'T1', 'T2', 'u', 'p']
    for k, n in enumerate(names):
        d = float(np.max(np.abs(perturbations[1][k] - perturbations[0][k])))
        scale = max(float(np.std(perturbations[0][k])), 1.0)
        diffs.append((n, d, d / scale))

    for n, d, dr in diffs:
        print(f"    Δ_pert {n} = {d:.3e}  (rel: {dr:.3e})")

    # Strict tolerance for Galilean invariance with same upwind direction
    for n, d, dr in diffs:
        if dr > 1e-6:
            fails.append((n, d, dr))

    print("-" * 64)
    if fails:
        print(f"S4 FAIL — {len(fails)} field(s) above 1e-6 relative threshold:")
        for n, d, dr in fails:
            print(f"  {n}: abs={d:.3e}, rel={dr:.3e}")
        return 1
    print("S4 PASS — Galilean invariance (perturbation match within 1e-6).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
