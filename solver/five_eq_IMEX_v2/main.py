"""v2 R1 explicit FVM solver — entry point.

`solve(eos1, eos2, W0, dx, t_end, …)` integrates the 1-D 5-equation
Allaire–Massoni model with forward Euler + 1st-order upwind face state +
self-consistent (EOS) face flux.

Public API:
    solve(eos1, eos2, W0, dx, t_end,
          cfl=0.4, bc_l='periodic', bc_r='periodic',
          max_steps=200_000, dt_fixed=None,
          history_every=0)

Returns a result dict with keys:
    W_final     — 5-tuple of (N,) arrays, primitive state at t_end
    t           — final simulated time (≤ t_end)
    n_steps     — number of completed time steps
    history     — list of (t, W) snapshots if history_every > 0 else []
    info_last   — dict from the last `euler_step` call (rho_min, p_min, …)

No free parameters except CFL (numerical stability) — dt is *uniquely*
determined as CFL · dx / max(|u|+c) per step (or `dt_fixed` if provided
for the 02-A regression test, which uses dt = 0.01 by spec).
"""
from __future__ import annotations
import numpy as np

from .flux_hllc import cell_max_wave_speed
from .time_euler import euler_step as _step


__all__ = ['solve']


def solve(eos1, eos2, W0, dx, t_end, *,
          cfl=0.4, bc_l='periodic', bc_r='periodic',
          max_steps=200_000, dt_fixed=None,
          history_every=0):
    """Time-integrate the 5-equation system with forward Euler + 1st-order upwind."""
    W = tuple(np.asarray(c, dtype=float).copy() for c in W0)
    t = 0.0
    history = []
    if history_every > 0:
        history.append((t, tuple(c.copy() for c in W)))

    info_last = {}
    n_completed = 0
    for n in range(max_steps):
        if t >= t_end:
            break
        # Time-step selection
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            wmax = float(np.max(cell_max_wave_speed(W, eos1, eos2)))
            if not np.isfinite(wmax) or wmax <= 0.0:
                raise FloatingPointError(f"non-positive max wave speed at t={t}")
            dt = cfl * dx / wmax
        if t + dt > t_end:
            dt = t_end - t
        if dt <= 0:
            break

        W, info_last = _step(W, dt, dx, eos1, eos2, bc_l, bc_r)
        t += dt
        n_completed = n + 1

        if history_every > 0 and (n_completed % history_every == 0):
            history.append((t, tuple(c.copy() for c in W)))
        if not np.isfinite(W[4]).all():
            raise FloatingPointError(f"NaN at step {n_completed}, t={t}")

    return dict(
        W_final=W,
        t=t,
        n_steps=n_completed,
        history=history,
        info_last=info_last,
    )
