"""SCMK-LBM Phase-7 : direct macro-Newton + Anderson acceleration on outer iteration.

Idea : drop FGMRES inner entirely. Each outer iter applies Phase-4 spectral Schur
       as direct macro Newton step, then composite line search and kinetic relax.
       Anderson acceleration combines last m outer iterates to break stagnation.

Per outer cost is much lower than FGMRES variant:
       1 R_f eval + 1 PC apply + (K_post + line_search) LBE.

Anderson acceleration (Walker-Ni type II):
       g(f) = f + delta_phase4(f)               (nonlinear fixed-point map)
       Keep history of (f_i, F_i = g(f_i) - f_i).
       f_{k+1} = sum_i gamma_i ( f_i + alpha F_i )  with least-squares solve for gamma.
"""

import time
import numpy as np

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_scmk_direct(case, max_outer=80, tol=1e-7, kinetic_substeps=15,
                      line_search_max=4, anderson_m=5, anderson_beta=1.0,
                      verbose=True):
    f = case.initial_field()
    n_full = case.dof
    N = case.N
    S_inv = build_spectral_schur(N, omega=case.omega, mode="ap")

    # Anderson history :  F_i = g(f_i) - f_i,   X_i = f_i
    F_hist = []   # residuals  g(f) - f
    X_hist = []   # iterates   f

    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_outer):
        R_f = case.residual(f)
        lbe_calls += 1
        res_norm = case._fast_norm(R_f) / np.sqrt(n_full)
        wall = time.perf_counter() - t0
        history.append((k, res_norm, lbe_calls, wall))
        if verbose:
            print(f"  outer {k:3d} | res {res_norm:.3e} | lbe {lbe_calls:5d} | wall {wall:.2f}s")
        if res_norm < tol:
            if verbose:
                print(f"  CONVERGED at outer {k}")
            break

        # ------- Phase-4 direct macro Newton step -------
        df = apply_spectral_schur(case, -R_f, S_inv)   # df = T S^{-1} M (-R)
        # composite line search
        alpha = 1.0
        f_trial = None
        accepted = False
        for _ in range(line_search_max):
            ft = f + alpha * df
            for _ in range(kinetic_substeps):
                ft = case.lbe_step(ft)
            lbe_calls += kinetic_substeps + 1
            Rt = ft - case.lbe_step(ft)
            rt = case._fast_norm(Rt) / np.sqrt(n_full)
            if rt < res_norm:
                f_trial = ft
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            for _ in range(kinetic_substeps):
                f = case.lbe_step(f)
            lbe_calls += kinetic_substeps
            continue
        g_f = f_trial

        # ------- Anderson combination -------
        F_new = g_f - f
        X_hist.append(f.copy())
        F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0)
            X_hist.pop(0)

        m = len(F_hist) - 1
        if m >= 1:
            # Build delta-F matrix : dF[:, i] = F_{i+1} - F_i, similarly dX
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(m)], axis=-1).reshape(-1, m)
            dX = np.stack([X_hist[i+1] - X_hist[i] for i in range(m)], axis=-1).reshape(-1, m)
            # Solve dF gamma = F_new in least squares
            gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            # Anderson step
            f_and = g_f - (dX + anderson_beta * dF) @ gamma
            f_and = f_and.reshape(case.shape)
            # safeguard : accept Anderson only if residual decreases
            R_and = f_and - case.lbe_step(f_and)
            lbe_calls += 1
            r_and = case._fast_norm(R_and) / np.sqrt(n_full)
            r_g = case._fast_norm(g_f - case.lbe_step(g_f)) / np.sqrt(n_full)
            lbe_calls += 1
            if r_and < r_g:
                f = f_and
            else:
                f = g_f
        else:
            f = g_f

    return f, history
