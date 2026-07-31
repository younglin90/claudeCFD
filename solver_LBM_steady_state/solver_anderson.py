"""Production-oriented Anderson-accelerated baseline LBM.

Walker-Ni style Type-II Anderson acceleration is applied only to the native
fixed-point map ``g(f) = L(f)``.  The solver keeps the comparison honest by
checking every accelerated candidate with the native residual and falling back
to the Picard candidate when the candidate is unstable.
"""

from __future__ import annotations

import time

import numpy as np


def _case_residual_norm(case, residual):
    chi = getattr(case, "chi", None)
    if chi is None:
        return float(np.sqrt(np.mean(residual * residual)))
    fluid = chi > 0.0
    if not np.any(fluid):
        return float(np.sqrt(np.mean(residual * residual)))
    return float(np.sqrt(np.mean(residual[:, fluid] * residual[:, fluid])))


def _macro_l2_residual_from_fields(case, f, g):
    try:
        if hasattr(case, "macro"):
            rho_f, ux_f, uy_f = case.macro(f)
            rho_g, ux_g, uy_g = case.macro(g)
        else:
            from lbm_core import moments
            rho_f, ux_f, uy_f = moments(f)
            rho_g, ux_g, uy_g = moments(g)
        dp = (rho_g - rho_f) / 3.0
        dux = ux_g - ux_f
        duy = uy_g - uy_f
        chi = getattr(case, "chi", None)
        if chi is not None:
            fluid = chi > 0.0
            if np.any(fluid):
                dp = dp[fluid]
                dux = dux[fluid]
                duy = duy[fluid]
        return float(np.sqrt(np.sum(dp * dp) + np.sum(dux * dux) + np.sum(duy * duy)))
    except Exception:
        return _case_residual_norm(case, g - f)


def _state_is_admissible(case, f, rho_floor=1e-10, speed_ceiling=0.5):
    if not np.all(np.isfinite(f)):
        return False
    try:
        if hasattr(case, "macro"):
            rho, ux, uy = case.macro(f)
        else:
            from lbm_core import moments
            rho, ux, uy = moments(f)
        chi = getattr(case, "chi", None)
        fluid = (chi > 0.0) if chi is not None else np.ones_like(rho, dtype=bool)
        if np.any(rho[fluid] <= rho_floor):
            return False
        speed2 = ux[fluid] * ux[fluid] + uy[fluid] * uy[fluid]
        if speed2.size and float(np.max(speed2)) > speed_ceiling * speed_ceiling:
            return False
    except Exception:
        return False
    return True


def _regularized_lstsq(dF, rhs, reg=1e-12, cond_restart=1e12):
    gram = dF.T @ dF
    rhs_g = dF.T @ rhs
    scale = max(float(np.trace(gram)) / max(gram.shape[0], 1), 1.0)
    gram_reg = gram + (reg * scale) * np.eye(gram.shape[0], dtype=gram.dtype)
    try:
        cond = float(np.linalg.cond(gram_reg))
        if not np.isfinite(cond) or cond > cond_restart:
            gamma, *_ = np.linalg.lstsq(dF, rhs, rcond=1e-10)
            return gamma, cond, True
        return np.linalg.solve(gram_reg, rhs_g), cond, False
    except np.linalg.LinAlgError:
        gamma, *_ = np.linalg.lstsq(dF, rhs, rcond=1e-10)
        return gamma, np.inf, True


def solve_anderson(case, max_iter=200, tol=1e-7, m=10, beta=1.0,
                   safeguard=True, verbose=True, check_every=1,
                   regularization=1e-12, max_backtracks=6,
                   monotone_factor=0.995, restart_on_reject=True,
                   plateau_window=50, plateau_eps=0.05):
    """Solve a steady LBM fixed point with safeguarded Anderson acceleration.

    Parameters retain the historical API.  Additional defaults provide
    production-style conditioning and safeguard behavior without case-specific
    tuning.
    """
    f = case.initial_field()

    F_hist = []
    G_hist = []
    history = []
    res_hist = []
    stats = {
        "accepted_anderson": 0,
        "accepted_picard": 0,
        "rejected": 0,
        "backtracks": 0,
        "history_restarts": 0,
        "ill_conditioned_ls": 0,
        "lbe_calls": 0,
    }
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_iter):
        g_f = case.lbe_step(f)
        lbe_calls += 1
        F_new = g_f - f
        rn = _macro_l2_residual_from_fields(case, f, g_f)
        wall = time.perf_counter() - t0
        if k % check_every == 0 or rn < tol:
            history.append((k, rn, lbe_calls, wall))
            res_hist.append(rn)
            if verbose and (k % 50 == 0 or rn < tol):
                print(f"  iter {k:5d} | res {rn:.3e} | lbe {lbe_calls:6d} | wall {wall:.2f}s")
        plateaued = False
        if len(res_hist) >= plateau_window:
            tail = res_hist[-plateau_window:]
            half = max(plateau_window // 2, 1)
            old = float(np.median(tail[:half]))
            new = float(np.median(tail[half:]))
            if np.isfinite(old) and old > 0 and np.isfinite(new):
                plateaued = (old - new) / old <= plateau_eps
        if not np.isfinite(rn) or plateaued:
            break

        G_hist.append(g_f)
        F_hist.append(F_new)
        if len(F_hist) > m + 1:
            F_hist.pop(0)
            G_hist.pop(0)

        n_m = len(F_hist) - 1
        if n_m < 1:
            f = g_f
            stats["accepted_picard"] += 1
            continue

        dF = np.stack([F_hist[i + 1] - F_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
        dG = np.stack([G_hist[i + 1] - G_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
        gamma, _cond, used_fallback = _regularized_lstsq(
            dF, F_new.ravel(), reg=regularization
        )
        if used_fallback:
            stats["ill_conditioned_ls"] += 1

        candidate = (g_f.ravel() - dG @ gamma).reshape(case.shape)
        if beta < 1.0:
            candidate = (1.0 - beta) * g_f + beta * candidate

        if not safeguard:
            f = candidate
            stats["accepted_anderson"] += 1
            continue

        accepted = False
        alpha = 1.0
        for bt in range(max_backtracks + 1):
            f_trial = g_f + alpha * (candidate - g_f)
            if not _state_is_admissible(case, f_trial):
                alpha *= 0.5
                stats["backtracks"] += 1
                continue
            R_test = f_trial - case.lbe_step(f_trial)
            lbe_calls += 1
            r_test = _macro_l2_residual_from_fields(case, f_trial, f_trial - R_test)
            if np.isfinite(r_test) and r_test <= monotone_factor * rn:
                f = f_trial
                accepted = True
                stats["accepted_anderson"] += 1
                stats["backtracks"] += bt
                break
            alpha *= 0.5
            stats["backtracks"] += 1

        if not accepted:
            f = g_f
            stats["accepted_picard"] += 1
            stats["rejected"] += 1
            if restart_on_reject:
                F_hist.clear()
                G_hist.clear()
                stats["history_restarts"] += 1

    stats["lbe_calls"] = lbe_calls
    solve_anderson.last_stats = stats
    return f, history


solve_anderson.last_stats = {}
