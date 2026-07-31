"""S5 — Mass conservation (per-phase).

Conservative form should preserve ∫α₁ρ₁ dx and ∫α₂ρ₂ dx exactly modulo
boundary fluxes.  For periodic BC the integrals are time-invariant.

Pass:  |Σ(α₁ρ₁) − Σ(α₁ρ₁)_init| / Σ(α₁ρ₁)_init < 1e-10
       same for α₂ρ₂.

Tested on (i) smooth α perturbation, (ii) α-jump, both with NASG-Ideal.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from solver.five_eq_IMEX_v2 import solve, IdealEOS, SGEOS, NASGEOS
from solver.five_eq_IMEX_v2.state import prim_to_cons


def _mass_integrals(W, eos1, eos2, dx):
    U, _ = prim_to_cons(W, eos1, eos2)
    return float(np.sum(U[0]) * dx), float(np.sum(U[1]) * dx)


def main():
    print("S5 Mass conservation (v2 R1)")
    print("-" * 64)
    fails = []

    eos1 = NASGEOS()
    eos2 = IdealEOS(gamma=1.4, kv=717.5)

    # ── Case A: smooth α profile, periodic, u₀ = 1
    N = 32; L = 1.0; dx = L / N
    x = (np.arange(N) + 0.5) * dx
    alpha = 0.5 * (1.0 + 0.4 * np.cos(2 * np.pi * x / L))
    W0 = (alpha.copy(),
          np.full(N, 300.0), np.full(N, 300.0),
          np.full(N, 1.0),    np.full(N, 1e5))
    m1_0, m2_0 = _mass_integrals(W0, eos1, eos2, dx)
    res = solve(eos1, eos2, W0, dx, t_end=1e-3, cfl=0.4,
                bc_l='periodic', bc_r='periodic', max_steps=200_000)
    m1_f, m2_f = _mass_integrals(res['W_final'], eos1, eos2, dx)
    err1 = abs(m1_f - m1_0) / abs(m1_0)
    err2 = abs(m2_f - m2_0) / abs(m2_0)
    print(f"  [A: smooth-α, u=1]  n={res['n_steps']}, "
          f"|Δ∫α₁ρ₁|/∫α₁ρ₁₀={err1:.3e}, |Δ∫α₂ρ₂|/∫α₂ρ₂₀={err2:.3e}")
    if err1 > 1e-10 or err2 > 1e-10:
        fails.append(('A', err1, err2))

    # ── Case B: α-jump, periodic, u₀ = 1.0 (02-A-like)
    N = 10; L = 1.0; dx = L / N
    alpha = np.empty(N)
    alpha[:N // 2] = 1.0 - 1e-3
    alpha[N // 2:] = 1e-3
    W0 = (alpha,
          np.full(N, 300.0), np.full(N, 300.0),
          np.full(N, 1.0),    np.full(N, 1e5))
    m1_0, m2_0 = _mass_integrals(W0, eos1, eos2, dx)
    res = solve(eos1, eos2, W0, dx, t_end=1e-3, cfl=0.4,
                bc_l='periodic', bc_r='periodic', max_steps=200_000)
    m1_f, m2_f = _mass_integrals(res['W_final'], eos1, eos2, dx)
    err1 = abs(m1_f - m1_0) / abs(m1_0)
    err2 = abs(m2_f - m2_0) / abs(m2_0)
    print(f"  [B: α-jump, u=1]    n={res['n_steps']}, "
          f"|Δ∫α₁ρ₁|/∫α₁ρ₁₀={err1:.3e}, |Δ∫α₂ρ₂|/∫α₂ρ₂₀={err2:.3e}")
    if err1 > 1e-10 or err2 > 1e-10:
        fails.append(('B', err1, err2))

    print("-" * 64)
    if fails:
        print(f"S5 FAIL — {len(fails)} case(s) above threshold:")
        for c, e1, e2 in fails:
            print(f"  case {c}: phase1 drift={e1:.3e}, phase2 drift={e2:.3e}")
        return 1
    print("S5 PASS — per-phase mass conserved (drift < 1e-10).")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
