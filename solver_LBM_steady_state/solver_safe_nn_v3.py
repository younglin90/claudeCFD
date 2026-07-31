"""Safe-NN-SCMK v3: lean Anderson + rare safeguard + final polish.

Per-outer cost is 1 LBE (Anderson lookahead only, no per-outer safeguard).
Residual-monotone safeguard is checked rarely (every N outer); reset history
on regression. JFNK + K-polish only when residual stagnates (auto-detect).
Final 100-LBE Picard polish + vchg check terminates with paper criterion.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_safenn_v3(case, max_outer=2000, tol=1e-7,
                     anderson_m=5, anderson_beta=1.0,
                     safeguard_every=20, safeguard_ratio=1.3,
                     stagnation_ratio=0.95, stagnation_window=10,
                     stagnation_jfnk_K_polish=15, krylov_max=10, krylov_tol=1e-3,
                     final_polish_max=20000, final_polish_check=100,
                     vchg_tol=1e-6, verbose=False):
    """Lean Anderson with auto-stagnation JFNK rescue.

    Algorithm:
      Phase A (Anderson):
        per outer = 1 LBE (g_f = L(f)) + Anderson lstsq
        every safeguard_every outer: residual test, reset history if regression
        if residual stagnated over stagnation_window outer (ratio > stagnation_ratio):
          trigger Phase B once
      Phase B (JFNK rescue):
        1 outer of JFNK + K-polish + AP-Schur PC, then back to Phase A
      Phase C (final polish):
        Picard until vchg over 100 LBE steps < vchg_tol
    """
    f_prev = case.initial_field()
    f = f_prev.copy()
    n_full = case.dof
    history = []
    t0 = time.perf_counter()
    lbe = 0
    has_mask = hasattr(case, 'chi')

    F_hist, X_hist, G_hist = [], [], []
    res_window = []
    S_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    last_safeguard_res = float('inf')

    for k in range(max_outer):
        # ---- Phase A: pure Anderson step (1 LBE) ----
        g_f = case.lbe_step(f); lbe += 1
        if not np.all(np.isfinite(g_f)):
            for _ in range(20):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); X_hist.clear(); G_hist.clear()
            continue
        F_new = g_f - f
        rn = float(np.sqrt((F_new * F_new).mean()))
        history.append((k, rn, lbe, time.perf_counter() - t0))
        res_window.append(rn)
        if len(res_window) > stagnation_window:
            res_window.pop(0)
        if verbose and (k < 3 or k % 50 == 0):
            print(f"  v3 k={k:4d} | res={rn:.3e} | lbe={lbe}", flush=True)
        if not np.isfinite(rn):
            break
        if rn < tol:
            f = g_f; break

        X_hist.append(f.copy()); G_hist.append(g_f.copy()); F_hist.append(F_new.copy())
        if len(F_hist) > anderson_m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)
        meff = len(F_hist) - 1
        if meff >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(meff)], axis=-1).reshape(-1, meff)
            try:
                gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
                f_new = g_f.ravel() - dG @ gamma
                f_new = f_new.reshape(case.shape)
                if anderson_beta < 1.0:
                    f_new = (1.0 - anderson_beta) * f + anderson_beta * f_new
            except np.linalg.LinAlgError:
                f_new = g_f
            if not np.all(np.isfinite(f_new)):
                f_new = g_f
                F_hist.clear(); X_hist.clear(); G_hist.clear()
        else:
            f_new = g_f
        if has_mask:
            f_new = f_new * case.chi[None, :, :]

        # ---- Periodic safeguard check ----
        if k > 0 and k % safeguard_every == 0:
            R_test = f_new - case.lbe_step(f_new); lbe += 1
            r_test = float(np.sqrt((R_test * R_test).mean()))
            if np.isfinite(r_test) and r_test > safeguard_ratio * last_safeguard_res:
                # regression: reset history, take plain g_f
                f_new = g_f
                F_hist.clear(); X_hist.clear(); G_hist.clear()
            last_safeguard_res = r_test

        # ---- Stagnation detection -> Phase B (JFNK rescue) ----
        if (len(res_window) == stagnation_window and
                res_window[-1] > stagnation_ratio * res_window[0] and
                rn > tol * 10):
            R_y = f_new - case.lbe_step(f_new); lbe += 1
            res_y = float(np.sqrt((R_y * R_y).mean()))
            if np.isfinite(res_y) and res_y > tol:
                norm_y = float(np.sqrt(np.sum(f_new * f_new)))
                probe = [0]
                def matvec(v_flat):
                    probe[0] += 1
                    return case.jvp(v_flat.reshape(case.shape), f_new, R_y,
                                      norm_f_cached=norm_y).ravel()
                def precond(r_flat):
                    return apply_spectral_schur(case, r_flat.reshape(case.shape),
                                                  S_inv).ravel()
                Aop = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
                Mop = LinearOperator((n_full, n_full), matvec=precond, dtype=np.float64)
                df, _ = gmres(Aop, -R_y.ravel(), M=Mop, rtol=krylov_tol,
                                atol=krylov_tol * np.linalg.norm(R_y) * 1e-3,
                                maxiter=1, restart=2 * krylov_max)
                lbe += probe[0]
                if np.all(np.isfinite(df)):
                    f_trial = f_new + df.reshape(case.shape)
                    for _ in range(stagnation_jfnk_K_polish):
                        f_trial = case.lbe_step(f_trial); lbe += 1
                    R_trial = f_trial - case.lbe_step(f_trial); lbe += 1
                    r_trial = float(np.sqrt((R_trial * R_trial).mean()))
                    if np.isfinite(r_trial) and r_trial < res_y:
                        f_new = f_trial
                        F_hist.clear(); X_hist.clear(); G_hist.clear()
                        res_window.clear()

        f_prev = f
        f = f_new

    # ---- Phase C: final polish ----
    prev_snap = f.copy()
    for step in range(1, final_polish_max + 1):
        f = case.lbe_step(f); lbe += 1
        if step % final_polish_check == 0:
            if hasattr(case, "macro"):
                _, ux, uy = case.macro(f)
                _, uxp, uyp = case.macro(prev_snap)
            else:
                from lbm_core import moments
                _, ux, uy = moments(f); _, uxp, uyp = moments(prev_snap)
            num = float(np.sqrt(np.sum((ux - uxp) ** 2 + (uy - uyp) ** 2)))
            den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1e-30)
            v = num / den
            history.append((max_outer + step, v, lbe, time.perf_counter() - t0))
            if np.isfinite(v) and v < vchg_tol:
                break
            prev_snap = f.copy()
    return f, history
