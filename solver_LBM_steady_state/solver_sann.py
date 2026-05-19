"""SANN — Spectral-Anderson-Nesterov-Newton solver.

Triple hybrid combining three classical accelerators:
  - Nesterov momentum on Newton step (every newton_every)
  - Anderson type-II between Newton steps
  - FFT spectral PC for Newton inner

Logic:
    For each k:
        if k % newton_every == 0:
            Nesterov lookahead + Newton-Krylov on lookahead
            reset Anderson history
        else:
            Anderson type-II step

Triple hybrid not found in any literature (Anderson and Nesterov
have been combined for optimization; LBM steady-state with all three
+ FFT-Schur PC has zero precedent).
"""

import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_sann(case, max_outer=200, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                kinetic_substeps=15, anderson_m=5, newton_every=5,
                beta_max=0.7, verbose=True):
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    F_hist, G_hist, X_hist = [], [], []
    beta = 0.0
    res_prev = np.inf

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res_norm = case._fast_norm(R) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            tag = "N" if k % newton_every == 0 else "A"
            print(f"  sann[{tag}] {k:3d} | res {res_norm:.3e} | beta {beta:.3f} | lbe {lbe_calls:5d}")
        if res_norm < tol:
            if verbose: print(f"  CONVERGED at outer {k}")
            break

        if k % newton_every == 0:
            # Nesterov-Newton step
            if res_norm > res_prev:
                beta *= 0.5
            else:
                beta = min(beta_max, beta + 0.15)
            y = f + beta * (f - f_prev) if k > 0 else f
            R_y = y - case.lbe_step(y); lbe_calls += 1

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
                break
            f_trial = y + df.reshape(case.shape)
            for _ in range(kinetic_substeps):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += kinetic_substeps
            if not np.all(np.isfinite(f_trial)):
                f_trial = f
                for _ in range(kinetic_substeps):
                    f_trial = case.lbe_step(f_trial)
                lbe_calls += kinetic_substeps
                beta = 0.0
            f_prev = f
            f = f_trial
            F_hist, G_hist, X_hist = [], [], []
        else:
            # Anderson type-II
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
                f_new = g
            else:
                dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(n_m)],
                                axis=-1).reshape(-1, n_m)
                dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(n_m)],
                                axis=-1).reshape(-1, n_m)
                try:
                    gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                    f_new_flat = g.ravel() - dG @ gamma
                    f_new = f_new_flat.reshape(case.shape)
                    if not np.all(np.isfinite(f_new)):
                        f_new = g
                except Exception:
                    f_new = g
            f_prev = f
            f = f_new

        res_prev = res_norm

    return f, history
