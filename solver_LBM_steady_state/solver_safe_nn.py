"""Safe-NN — Residual-monotone Nesterov + Newton-Krylov.

Difference from NN:
    y_k = f_k + β(f_k - f_{k-1})
    R(y_k) evaluated
    if ||R(y_k)|| > (1+ε) ||R(f_k)|| :
        β ← 0.5 β, y_k := f_k         (reject lookahead)
    Newton-Krylov step on accepted y_k

Prevents Cavity Re=400 NaN.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_safe_nn(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                   kinetic_substeps=15, beta_max=0.7, eps_accept=0.05, verbose=True):
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    beta = 0.0
    res_prev = np.inf
    reject_count = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res, lbe_calls, wall))
        if verbose:
            print(f"  snn {k:3d} | res {res:.3e} | beta {beta:.3f} | rej {reject_count} | lbe {lbe_calls:5d}")
        if res < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # Tentative β
        if res > res_prev:
            beta = beta * 0.5
        else:
            beta = min(beta_max, beta + 0.15)

        # Nesterov lookahead with residual-safe acceptance
        if beta > 0.3:
            y = f + beta * (f - f_prev)
            R_y = y - case.lbe_step(y); lbe_calls += 1
            norm_R_y = case._fast_norm(R_y)
            norm_R = case._fast_norm(R)
            if norm_R_y > (1.0 + eps_accept) * norm_R or not np.all(np.isfinite(R_y)):
                y = f.copy()
                R_y = R
                beta = beta * 0.5
                reject_count += 1
        else:
            y = f
            R_y = R

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
        # Adaptive K: fewer substeps when near convergence AND res decreasing
        if res < 1e-4 and res < res_prev:
            K_eff = max(5, kinetic_substeps // 2)
        else:
            K_eff = kinetic_substeps
        for _ in range(K_eff):
            f_new = case.lbe_step(f_new)
        lbe_calls += K_eff
        if not np.all(np.isfinite(f_new)):
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            beta = 0.0

        f_prev = f
        f = f_new
        res_prev = res

    return f, history
