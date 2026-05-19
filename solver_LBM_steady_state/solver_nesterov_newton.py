"""Nesterov-accelerated Newton-Krylov (NN-K).

Combines:
  - Nesterov momentum on f-state
  - SCMK Newton-Krylov as core step
  - Adaptive beta restart on residual increase

Algorithm:
    y_k = f_k + beta (f_k - f_{k-1})
    R_y = y_k - L(y_k)
    FGMRES(J·δf = -R_y, FFT-PC, maxiter=1)
    f_{k+1} = y_k + δf
    f_{k+1} = L^K(f_{k+1})

Novelty: Nesterov momentum on Newton-Krylov iterates for LBM steady-state.
No LBM literature precedent. Standard ML trick applied to CFD root-finding.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_nn(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
              kinetic_substeps=15, beta_max=0.9, verbose=True):
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
            print(f"  nn {k:3d} | res {res_norm:.3e} | beta {beta:.3f} | lbe {lbe_calls:5d}")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Adaptive Nesterov beta
        if res_norm > res_prev:
            beta = beta * 0.5             # half-restart instead of full
        else:
            beta = min(beta_max, beta + 0.15)

        # Nesterov lookahead
        y = f + beta * (f - f_prev)
        R_y = y - case.lbe_step(y); lbe_calls += 1

        # Newton-Krylov inner on lookahead
        norm_y = case._fast_norm(y)
        probe = [0]
        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, y, R_y, norm_f_cached=norm_y).ravel()
        def precond(r_flat):
            return apply_spectral_schur(case, r_flat.reshape(case.shape),
                                          S_inv).ravel()
        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        df, _ = gmres(Aop, -R_y.ravel(), M=Mop, rtol=krylov_tol,
                       atol=krylov_tol * np.linalg.norm(R_y) * 1e-3,
                       maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df)):
            if verbose: print("  GMRES NaN, abort")
            break

        f_new = y + df.reshape(case.shape)
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
