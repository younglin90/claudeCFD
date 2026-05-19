"""Spectral-Anderson-Newton (SAN) hybrid solver.

Novel combination: Anderson type-II + SCMK Newton-Krylov interleaved.

Algorithm:
    for k = 1, 2, ... :
        IF k % 3 != 0 :  # Anderson step (cheap)
            g = L^K(f)            # baseline iter as fixed-point map
            F = g - f
            Anderson combine last m residuals to get f_new
        ELSE :  # Newton step every 3rd iter (precise)
            R = f - L(f)
            FGMRES with FFT-PC : J δf = -R
            f = f + δf, then L^K
            reset Anderson history

Intuition:
- Anderson is very fast on smooth periodic (single dominant mode)
- Newton is robust on stiff walls
- Interleaving captures both regimes

Novelty: Anderson + Newton-Krylov combination for LBM steady-state is not
in literature. Most Anderson-LBM papers use only Anderson; SCMK-style use
only Newton.
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_san(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
               kinetic_substeps=15, anderson_m=5, newton_every=3, verbose=True):
    f = case.initial_field()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    F_hist = []
    G_hist = []
    X_hist = []

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            tag = "N" if k % newton_every == 0 else "A"
            print(f"  san[{tag}] {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d}")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        if k % newton_every == 0:
            # ── Newton-Krylov step ──
            norm_f = case._fast_norm(f)
            probe = [0]
            def matvec(v_flat):
                w = v_flat.reshape(case.shape)
                probe[0] += 1
                return case.jvp(w, f, R, norm_f_cached=norm_f).ravel()
            def precond(r_flat):
                return apply_spectral_schur(case, r_flat.reshape(case.shape),
                                              S_inv).ravel()
            Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
            Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
            df, _ = gmres(Aop, -R.ravel(), M=Mop, rtol=krylov_tol,
                            atol=krylov_tol * np.linalg.norm(R) * 1e-3,
                            maxiter=1, restart=2 * krylov_max)
            lbe_calls += probe[0]
            if not np.all(np.isfinite(df)):
                break
            f_trial = f + df.reshape(case.shape)
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
            if not np.all(np.isfinite(f_trial)):
                # fallback
                f_trial = f
                for _ in range(kinetic_substeps):
                    f_trial = case.lbe_step(f_trial)
                lbe_calls += kinetic_substeps
            f = f_trial
            # reset Anderson history after Newton (state jump invalidates linear extrap)
            F_hist, G_hist, X_hist = [], [], []
        else:
            # ── Anderson type-II step ──
            g = f
            for _ in range(kinetic_substeps):
                g = case.lbe_step(g)
            lbe_calls += kinetic_substeps
            F_new = g - f
            F_hist.append(F_new); G_hist.append(g); X_hist.append(f.copy())
            if len(F_hist) > anderson_m + 1:
                F_hist.pop(0); G_hist.pop(0); X_hist.pop(0)
            n_m = len(F_hist) - 1
            if n_m < 1:
                f = g
            else:
                dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(n_m)],
                                axis=-1).reshape(-1, n_m)
                dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(n_m)],
                                axis=-1).reshape(-1, n_m)
                try:
                    gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                    f_new = g.ravel() - dG @ gamma
                    f_new = f_new.reshape(case.shape)
                    if not np.all(np.isfinite(f_new)):
                        f = g
                    else:
                        f = f_new
                except Exception:
                    f = g

    return f, history
