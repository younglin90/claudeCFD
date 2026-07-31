"""Baseline LBM time-marching to steady state."""

import time
import numpy as np


def reference_residual_norm(case, residual):
    chi = getattr(case, "chi", None)
    if chi is None:
        return float(np.sqrt(np.mean(residual * residual)))
    fluid = chi > 0.0
    if not np.any(fluid):
        return float(np.sqrt(np.mean(residual * residual)))
    return float(np.sqrt(np.mean(residual[:, fluid] * residual[:, fluid])))


def macro_l2_residual_from_fields(case, f, g):
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


def solve_baseline(case, max_steps=300000, tol=1e-7, check_every=50, verbose=True,
                    plateau_window=50, plateau_eps=0.05):
    """Native LBM fixed-point iteration : f^{n+1} = L(f^n).

    Stops on a non-finite residual or once the residual has plateaued: over
    the last ``plateau_window`` checkpoints, the relative improvement between
    the older and newer half-medians is at most ``plateau_eps``. ``tol`` is
    accepted for API compatibility but no longer used as an early-exit
    condition.
    """
    f = case.initial_field()
    inplace_step = getattr(case, "lbe_step_inplace", None)
    f_work = np.empty_like(f) if callable(inplace_step) else None
    history = []  # list of (step, res, lbe_calls, wall_time)
    res_hist = []
    t0 = time.perf_counter()
    lbe_calls = 0

    # Record the initial native residual so LBE-call plots start from the same
    # physical point as accelerated solvers. This residual evaluation costs one
    # additional LBE application and is counted explicitly.
    R0 = f - case.lbe_step(f)
    lbe_calls += 1
    history.append((0, macro_l2_residual_from_fields(case, f, f - R0), lbe_calls, time.perf_counter() - t0))
    res_hist.append(history[-1][1])
    if not np.isfinite(history[-1][1]):
        return f, history

    for step in range(1, max_steps + 1):
        if callable(inplace_step):
            f_new = inplace_step(f, f_work)
        else:
            f_new = case.lbe_step(f)
        lbe_calls += 1

        if step % check_every == 0:
            # ||R_f|| = ||f - L(f)|| using the freshly computed f_new
            R = f_new - case.lbe_step(f_new)
            lbe_calls += 1
            res = macro_l2_residual_from_fields(case, f_new, f_new - R)
            wall = time.perf_counter() - t0
            history.append((step, res, lbe_calls, wall))
            res_hist.append(res)
            if verbose and (step % 1000 == 0 or step == check_every):
                print(f"  step {step:7d} | res {res:.3e} | wall {wall:.2f}s")
            plateaued = False
            if len(res_hist) >= plateau_window:
                tail = res_hist[-plateau_window:]
                half = max(plateau_window // 2, 1)
                old = float(np.median(tail[:half]))
                new = float(np.median(tail[half:]))
                if np.isfinite(old) and old > 0 and np.isfinite(new):
                    plateaued = (old - new) / old <= plateau_eps
            if not np.isfinite(res) or plateaued:
                f = f_new
                if verbose:
                    print(f"  STOPPED at step {step} | res {res:.3e} | plateaued={plateaued}")
                break
        if callable(inplace_step):
            f, f_work = f_new, f
        else:
            f = f_new

    return f, history
