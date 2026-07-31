"""Safe-NN-SCMK v2 — Anderson-augmented variant for M6 improvement loop.

Replaces M6's Nesterov lookahead with Type-II Anderson (depth m=5) while
keeping AP-Schur FFT preconditioner, JFNK Newton-Krylov correction,
residual-monotone safeguard, K-annealed post-relaxation, mass projection.

The Anderson lookahead is rank-revealing on low-rank Jacobians (smooth
single-mode cases), giving fast convergence on Kolmogorov / channel where
single-step Nesterov stalls.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_safenn_v2(case, max_outer=400, tol=1e-7,
                     krylov_max=10, krylov_tol=1e-3,
                     kinetic_substeps=15, kinetic_substeps_min=8,
                     adaptive_k_threshold=1e-5,
                     anderson_m=5, anderson_beta=1.0,
                     safeguard_ratio=1.05,
                     line_search_max=4,
                     vchg_tol=1e-6, vchg_check_outer=50,
                     internal_polish_max=20000,
                     internal_polish_check=100,
                     verbose=False):
    """Safe-NN-SCMK v2.

    Algorithm per outer:
      1. R = f - L(f)                                # residual
      2. y = Anderson_lookahead(history of f, F=g(f)-f)   # rank-revealing
      3. R_y = y - L(y)                              # safeguard residual
         if ||R_y|| > safeguard_ratio*||R|| -> reject, y=f, R_y=R
      4. JFNK on y: solve  J·df = -R_y  via GMRES with AP-Schur PC
      5. f_trial = y + alpha·df, K_eff Picard polish
      6. accept f_trial if monotone, else baseline LBE
      7. internal vchg termination check every vchg_check_outer
    """
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    has_mask = hasattr(case, 'chi')

    F_hist, X_hist, G_hist = [], [], []
    f_snap = f.copy()
    last_check = 0

    for k in range(max_outer):
        R = case.residual(f); lbe_calls += 1
        res = case._fast_norm(R) / math.sqrt(n_full)
        history.append((k, res, lbe_calls, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 20 == 0):
            print(f"  v2 k={k:3d} | res {res:.3e} | lbe {lbe_calls}")
        if not np.isfinite(res):
            break
        if res < tol:
            break
        # internal vchg termination (paper-faithful)
        if k - last_check >= vchg_check_outer and k > 0:
            rho, ux, uy = (case.macro(f) if hasattr(case, "macro") else
                            (f.sum(axis=0), None, None))
            if ux is not None:
                _, uxp, uyp = (case.macro(f_snap) if hasattr(case, "macro") else
                                (f_snap.sum(axis=0), None, None))
                num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
                den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
                vchg = num / den
                if vchg < vchg_tol and res < 1e-5:
                    break
                f_snap = f.copy(); last_check = k

        # ---- Anderson lookahead (replaces Nesterov) ----
        g_f = case.lbe_step(f); lbe_calls += 1
        F_new = g_f - f
        X_hist.append(f.copy()); G_hist.append(g_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                y = g_f.ravel() - dG @ gamma
                y = y.reshape(case.shape)
                if anderson_beta < 1.0:
                    y = (1.0 - anderson_beta) * f + anderson_beta * y
                if not np.all(np.isfinite(y)):
                    y = g_f
                    F_hist.clear(); G_hist.clear(); X_hist.clear()
            except np.linalg.LinAlgError:
                y = g_f
        else:
            y = g_f
        if has_mask:
            y = y * case.chi[None, :, :]
        R_y = case.residual(y); lbe_calls += 1
        res_y = case._fast_norm(R_y) / math.sqrt(n_full)
        if not (np.isfinite(res_y) and res_y < safeguard_ratio * res):
            # Anderson reject -> fall back to plain g_f (Picard step)
            y = g_f; R_y = g_f - case.lbe_step(g_f); lbe_calls += 1
            F_hist.clear(); G_hist.clear(); X_hist.clear()

        # ---- Adaptive JFNK skip: pure Anderson + minimal polish ----
        if len(history) >= 2 and history[-1][1] < 0.7 * history[-2][1] and res_y < res:
            f_new = y
            for _ in range(kinetic_substeps_min):
                f_new = case.lbe_step(f_new); lbe_calls += 1
            if has_mask:
                f_new = f_new * case.chi[None, :, :]
            f_prev = f; f = f_new
            continue

        # ---- JFNK on Anderson lookahead y ----
        norm_y = case._fast_norm(y)
        probe = [0]
        def matvec(v_flat):
            probe[0] += 1
            return case.jvp(v_flat.reshape(case.shape), y, R_y,
                              norm_f_cached=norm_y).ravel()
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

        # ---- Adaptive K_eff for kinetic polish (Round 2 stronger) ----
        if res < adaptive_k_threshold:
            K_eff = kinetic_substeps_min
        elif res < 3e-5 and len(history) >= 2 and res < history[-2][1]:
            K_eff = max(5, kinetic_substeps // 2)
        else:
            K_eff = kinetic_substeps

        accepted = False
        alpha = 1.0
        f_new = None
        for _ in range(line_search_max):
            f_trial = y + alpha * df.reshape(case.shape)
            for _ in range(K_eff):
                f_trial = case.lbe_step(f_trial)
            lbe_calls += K_eff
            if not np.all(np.isfinite(f_trial)):
                alpha *= 0.5; continue
            R_trial = f_trial - case.lbe_step(f_trial); lbe_calls += 1
            r_trial = case._fast_norm(R_trial) / math.sqrt(n_full)
            if np.isfinite(r_trial) and r_trial <= max(res, tol):
                f_new = f_trial; accepted = True; break
            alpha *= 0.5

        if not accepted or f_new is None:
            f_new = f
            for _ in range(kinetic_substeps):
                f_new = case.lbe_step(f_new)
            lbe_calls += kinetic_substeps
            F_hist.clear(); G_hist.clear(); X_hist.clear()
        if has_mask:
            f_new = f_new * case.chi[None, :, :]
        f_prev = f
        f = f_new

    # ---- Internal final polish: Picard until vchg over 100 LBE steps < vchg_tol ----
    # Eliminates external picard_tail overhead. Counts toward LBE.
    prev_snap = f.copy()
    for step in range(1, internal_polish_max + 1):
        f = case.lbe_step(f); lbe_calls += 1
        if step % internal_polish_check == 0:
            if hasattr(case, "macro"):
                _, ux, uy = case.macro(f)
                _, uxp, uyp = case.macro(prev_snap)
            else:
                from lbm_core import moments
                _, ux, uy = moments(f); _, uxp, uyp = moments(prev_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            v = num / den
            history.append((max_outer + step, v, lbe_calls, time.perf_counter() - t0))
            if np.isfinite(v) and v < vchg_tol:
                break
            prev_snap = f.copy()
    return f, history
