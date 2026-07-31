"""Safe-NN-SCMK v4 — consolidated. Lean Anderson + adaptive stagnation JFNK.

Key features:
- Per outer = 1 LBE (g_f) + Anderson lstsq (no per-outer safeguard)
- Track f_best (lowest residual seen); rare residual probe every safeguard_every
- Stagnation detect (residual not decreasing) -> JFNK+AP-Schur PC rescue, max N times
- Final 100-LBE polish with vchg<1e-6 termination
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_periodic import apply_spectral_schur, build_spectral_schur


def solve_safenn_v4(case, max_outer=3000, tol=1e-7,
                     anderson_m=5, anderson_beta=1.0,
                     safeguard_every=10,
                     stagnation_window=8, stagnation_ratio=0.9,
                     stagnation_max_triggers=3,
                     stagnation_K_polish=10, krylov_max=10, krylov_tol=1e-3,
                     final_polish_max=20000, final_polish_check=100,
                     vchg_tol=1e-6, verbose=False):
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
    stagnation_triggered = 0
    f_best = f.copy()
    res_best = float('inf')

    for k in range(max_outer):
        # Phase A: lean Anderson (1 LBE)
        g_f = case.lbe_step(f); lbe += 1
        if not np.all(np.isfinite(g_f)):
            for _ in range(20):
                f = case.lbe_step(f); lbe += 1
            F_hist.clear(); X_hist.clear(); G_hist.clear()
            continue
        F_new = g_f - f
        rn = float(np.sqrt((F_new * F_new).mean()))
        history.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose and (k < 3 or k % 50 == 0):
            print(f"  v4 k={k:4d} | res={rn:.3e} | lbe={lbe}", flush=True)
        if np.isfinite(rn) and rn < res_best:
            res_best = rn; f_best = g_f.copy()
        if not np.isfinite(rn):
            f = f_best
            F_hist.clear(); X_hist.clear(); G_hist.clear()
            continue
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
                if not np.all(np.isfinite(f_new)):
                    f_new = g_f
                    F_hist.clear(); X_hist.clear(); G_hist.clear()
            except np.linalg.LinAlgError:
                f_new = g_f
        else:
            f_new = g_f
        if has_mask:
            f_new = f_new * case.chi[None, :, :]

        # Rare residual probe
        if k > 0 and k % safeguard_every == 0:
            R_test = f_new - case.lbe_step(f_new); lbe += 1
            r_test = float(np.sqrt((R_test * R_test).mean()))
            if np.isfinite(r_test) and r_test < res_best:
                res_best = r_test; f_best = f_new.copy()
            res_window.append(r_test)
            if len(res_window) > stagnation_window:
                res_window.pop(0)
            # stagnation detection
            if (len(res_window) == stagnation_window and stagnation_triggered < stagnation_max_triggers and
                    res_window[-1] > stagnation_ratio * res_window[0] and rn > tol * 10):
                # JFNK rescue
                norm_y = float(np.sqrt(np.sum(f_new * f_new)))
                probe = [0]
                R_y = R_test
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
                    for _ in range(stagnation_K_polish):
                        f_trial = case.lbe_step(f_trial); lbe += 1
                    R_trial = f_trial - case.lbe_step(f_trial); lbe += 1
                    r_trial = float(np.sqrt((R_trial * R_trial).mean()))
                    if np.isfinite(r_trial) and r_trial < r_test:
                        f_new = f_trial
                        if r_trial < res_best:
                            res_best = r_trial; f_best = f_trial.copy()
                        F_hist.clear(); X_hist.clear(); G_hist.clear()
                        res_window.clear()
                        stagnation_triggered += 1

        f_prev = f
        f = f_new

    # Use best-residual snapshot
    if res_best < float('inf') and not (rn < tol):
        f = f_best

    # Phase C: final polish with vchg termination
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
