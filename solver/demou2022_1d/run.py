# solver/demou2022_1d/run.py
"""
Main entry point for the Demou 2022 1D solver (Mode A: RK3 + Helmholtz).

Usage
-----
from solver.demou2022_1d.run import run

result = run(case_params)

case_params keys
----------------
Required:
    ph1, ph2      : phase EOS dicts  {'gamma','pinf','b','kv','eta'}
    x_cells       : (N,) cell-centre positions
    alpha1_init   : (N,) initial volume fraction of phase 1
    T_init        : (N,) initial temperature  [K]
    u_init        : (N,) initial velocity     [m/s]
    p_init        : (N,) initial pressure     [Pa]
    t_end         : simulation end time       [s]

Optional:
    CFL           : CFL number (default 0.4)
    bc_left       : 'periodic' | 'transmissive' | 'wall'  (default 'transmissive')
    bc_right      : same
    output_times  : list of times for snapshots
    verbose       : print progress every N steps (default False)
    dt_fixed      : fixed time step (overrides CFL)

Returns
-------
dict with keys:
    alpha1_final, T_final, u_final, p_final, rho_final
    x_cells, t_final, n_steps, snapshots
"""

import numpy as np

from . import eos, source_terms, timestepping, rk3


def run(case_params: dict) -> dict:
    ph1 = case_params['ph1']
    ph2 = case_params['ph2']

    x_cells = np.asarray(case_params['x_cells'], dtype=float)
    alpha1  = np.asarray(case_params['alpha1_init'], dtype=float).copy()
    T       = np.asarray(case_params['T_init'],      dtype=float).copy()
    u       = np.asarray(case_params['u_init'],      dtype=float).copy()
    p       = np.asarray(case_params['p_init'],      dtype=float).copy()
    t_end   = float(case_params['t_end'])

    N  = len(x_cells)
    dx = float(x_cells[1] - x_cells[0]) if N > 1 else 1.0

    CFL          = float(case_params.get('CFL', 0.4))
    bc_l         = case_params.get('bc_left',  'transmissive')
    bc_r         = case_params.get('bc_right', 'transmissive')
    output_times = sorted(case_params.get('output_times', []))
    verbose      = bool(case_params.get('verbose', False))
    dt_fixed     = case_params.get('dt_fixed', None)

    # Initial mixture density
    pp1 = eos.phase_props(p, T, ph1)
    pp2 = eos.phase_props(p, T, ph2)
    rho = alpha1 * pp1['rho'] + (1.0 - alpha1) * pp2['rho']

    t       = 0.0
    n_steps = 0
    snapshots = []
    out_idx   = 0

    # Save t=0 snapshot if requested
    if output_times and abs(output_times[0]) < 1e-14:
        snapshots.append(_snap(t, alpha1, T, u, p, rho, x_cells))
        out_idx += 1

    while t < t_end - 1e-14 * t_end:
        # Time step
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            c2_mix = source_terms.mixture_sound_speed_sq(alpha1, pp1, pp2, T)
            c_mix  = np.sqrt(np.maximum(c2_mix, 0.0))
            dt = timestepping.compute_dt(u, c_mix, dx, CFL)

        dt = min(dt, t_end - t)
        if out_idx < len(output_times):
            dt = min(dt, output_times[out_idx] - t)
        if dt <= 0.0:
            break

        # RK3 step
        alpha1, T, u, p = rk3.step(alpha1, T, u, p, ph1, ph2, dx, dt, bc_l, bc_r)

        # Update derived quantities for next CFL estimate
        pp1 = eos.phase_props(p, T, ph1)
        pp2 = eos.phase_props(p, T, ph2)
        rho = alpha1 * pp1['rho'] + (1.0 - alpha1) * pp2['rho']

        t       += dt
        n_steps += 1

        if verbose and n_steps % 200 == 0:
            print(f"  step={n_steps:6d}  t={t:.4e}  dt={dt:.3e}  "
                  f"p=[{p.min():.4e},{p.max():.4e}]  "
                  f"u=[{u.min():.4e},{u.max():.4e}]")

        # Snapshots
        while out_idx < len(output_times) and t >= output_times[out_idx] - 1e-14:
            snapshots.append(_snap(t, alpha1, T, u, p, rho, x_cells))
            out_idx += 1

    if verbose:
        print(f"  Done: {n_steps} steps, t_final={t:.6e}")

    return {
        'alpha1_final': alpha1,
        'T_final':      T,
        'u_final':      u,
        'p_final':      p,
        'rho_final':    rho,
        'x_cells':      x_cells,
        't_final':      t,
        'n_steps':      n_steps,
        'snapshots':    snapshots,
    }


def _snap(t, alpha1, T, u, p, rho, x_cells):
    return {
        't':      t,
        'alpha1': alpha1.copy(),
        'T':      T.copy(),
        'u':      u.copy(),
        'p':      p.copy(),
        'rho':    rho.copy(),
        'x':      x_cells.copy(),
    }
