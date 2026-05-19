"""Nesterov-Spectral-Picard (NSP) — novel solver.

Combines:
  - Nesterov momentum (ML-style accelerated gradient flow)
  - FFT-based spectral predictor (SCMK PC) as "gradient" direction
  - LBM as black-box residual oracle

Algorithm:
    y_k = f_k + beta_k (f_k - f_{k-1})            # Nesterov lookahead
    R_y = y_k - L(y_k)                             # residual at lookahead
    delta = T · S^{-1} · M · R_y                   # FFT PC (predicts Newton)
    f_{k+1} = y_k - delta                          # update
    f_{k+1} = L^K(f_{k+1})                         # post-LBM neq cleanup

    beta_k adaptive: increases if R decreasing, restarts if R increases

No FGMRES inner loop. No JVP. ~30 lines core.
Novelty: Nesterov momentum on LBM residual flow with spectral PC has zero
LBM literature precedent. ML-style optimizer adapted to CFD steady-state.
"""

import time
import numpy as np
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_nsp(case, max_outer=300, tol=1e-7,
               kinetic_substeps=15, beta_max=0.95, verbose=True):
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    beta = 0.0
    res_prev = np.inf

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  nsp {k:3d} | res {res_norm:.3e} | beta {beta:.3f} | lbe {lbe_calls:5d}")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Adaptive Nesterov beta
        if res_norm > res_prev:
            beta = 0.0                  # restart (residual grew)
        else:
            beta = min(beta_max, beta + 0.15)

        # Nesterov lookahead
        y = f + beta * (f - f_prev)

        # Residual at lookahead
        R_y = y - case.lbe_step(y); lbe_calls += 1

        # Spectral PC predictor (Newton direction approx)
        delta = apply_spectral_schur(case, R_y, S_inv)

        # Update
        f_new = y - delta

        # Post-LBM kinetic substeps
        for _ in range(kinetic_substeps):
            f_new = case.lbe_step(f_new)
        lbe_calls += kinetic_substeps

        # NaN safeguard
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            beta = 0.0

        f_prev = f
        f = f_new
        res_prev = res_norm

    return f, history
