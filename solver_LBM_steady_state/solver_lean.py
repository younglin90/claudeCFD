"""LEAN solver : minimum-component SCMK.

Components retained (essential):
    - JFNK outer (matrix-free)
    - FGMRES inner (maxiter=1, restart=2*K_max)
    - Spectral PC (FFT, adaptive Tikhonov)
    - K_kinetic LBE post-step

Components removed:
    - Hybrid fallback (phase B)
    - Line search
    - Stagnation detection
    - Composite alpha logic

Tradeoff: simpler code, fewer safeguards. Should work for periodic + wall cases
where SCMK already does well. May blow up on stiff cases.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_lean(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                kinetic_substeps=15, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  lean {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        # FGMRES inner with JVP matvec, FFT-PC
        norm_f = case._fast_norm(f)
        probe = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            probe[0] += 1
            return case.jvp(w, f, R, norm_f_cached=norm_f).ravel()

        def precond(r_flat):
            R_ = r_flat.reshape(case.shape)
            return apply_spectral_schur(case, R_, S_inv).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R.ravel()
        df_flat, _ = gmres(Aop, rhs, M=Mop, rtol=krylov_tol,
                            atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
                            maxiter=1, restart=2 * krylov_max)
        lbe_calls += probe[0]

        if not np.all(np.isfinite(df_flat)):
            if verbose: print("  GMRES NaN, abort")
            break

        df = df_flat.reshape(case.shape)
        # Accept α=1 + post-LBM; NaN safeguard
        f_trial = f + df
        for _ in range(kinetic_substeps):
            f_trial = case.lbe_step(f_trial)
        lbe_calls += kinetic_substeps
        # check finiteness; if NaN/inf, fall back to baseline (no Newton)
        if not np.all(np.isfinite(f_trial)):
            f_trial = f
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
        f = f_trial

    return f, history
