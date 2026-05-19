"""Phase-3 SCMK-LBM : LBE-smoother + spectral-PC Newton-Krylov.

Algorithm (two-phase outer iteration, §3.1 kinetic-aware decomposition) :

    for outer k :
        # ---- SMOOTHER PHASE ----
        # K_smooth LBE substeps absorb wall / mean / kinetic null-space modes
        for _ in K_smooth :
            f <- L(f)

        # ---- NEWTON PHASE ----
        # FGMRES on remaining high-k macro residual, spectral PC dominant
        R = f - L(f)
        if  ||R|| < tol  :  break
        solve  J df = -R  via FGMRES(M = T S_U^AP^{-1} M, FFT)
        composite line search :  f_trial = L^K_post (f + alpha df)
        accept best of  { Newton+post-smooth,  pure smoother }
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur


def solve_scmk_mg(
    case,
    S_inv,
    max_outer=200,
    tol=1e-9,
    krylov_max=10,
    krylov_tol=1e-3,
    K_smooth=20,
    K_post=15,
    line_search_max=5,
    k_low_cutoff=None,
    verbose=True,
):
    f = case.initial_field()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        # ---------------- pre-smoother (fine LBE substeps) ----------------
        for _ in range(K_smooth):
            f = case.lbe_step(f)
        lbe_calls += K_smooth

        # current residual
        R_f = case.residual(f)
        lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  outer {k:3d} | post-smooth res {res_norm:.3e} | lbe {lbe_calls:6d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose:
                print(f"  CONVERGED at outer {k}")
            break

        # ---------------- Newton + spectral PC ----------------
        norm_f = case._fast_norm(f)
        probe_count = [0]

        def matvec(v_flat):
            w = v_flat.reshape(case.shape)
            Jw = case.jvp(w, f, R_f, norm_f_cached=norm_f)
            probe_count[0] += 1
            return Jw.ravel()

        def precond(r_flat):
            R = r_flat.reshape(case.shape)
            return apply_spectral_schur(case, R, S_inv, k_low_cutoff=k_low_cutoff).ravel()

        Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
        rhs = -R_f.ravel()
        df_flat, info = gmres(
            Aop, rhs, M=Mop,
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(rhs) * 1e-3,
            maxiter=2,
            restart=krylov_max,
        )
        lbe_calls += probe_count[0]

        if not np.all(np.isfinite(df_flat)):
            print("  GMRES NaN, fall back to pure smoother")
            continue

        df = df_flat.reshape(case.shape)

        # ---------------- composite line search w/ post-smooth ----------------
        alpha = 1.0
        accepted = False
        for _ in range(line_search_max):
            f_trial = f + alpha * df
            for _ in range(K_post):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += K_post + 1
            R_trial = f_trial - case.lbe_step(f_trial)
            r_trial = case._fast_norm(R_trial) / np.sqrt(n_full)
            if r_trial < res_norm:
                f = f_trial
                accepted = True
                break
            alpha *= 0.5
        # if not accepted, keep current f (smoother already advanced it)

    return f, history
