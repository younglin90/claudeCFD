"""
Validation: 1D Sharp Interface Advection (Air/Water NASG)

Initial condition:
  psi = 1.0  (water) for x in [0.4, 0.6]
  psi = 0.0  (air)   elsewhere
Periodic BC, u0=1 m/s, p0=1e5 Pa, T0=293.15 K.

Tests whether the solver maintains pressure equilibrium (no spurious
pressure waves) when advecting a sharp interface at CFL > 1.

PASS criterion: L2(p) < 1e-4 at each flow-through.

Usage:
    python validation/1D/test_sharp_advection.py [--N 50] [--CFL 2.0] [--max-ft 5]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from solver.denner_1d.main import run

# ---------------------------------------------------------------------------
# EOS parameters
# ---------------------------------------------------------------------------
PH_AIR = {
    'gamma': 1.4,
    'pinf':  0.0,
    'b':     0.0,
    'kv':    717.5,
    'eta':   0.0,
}
PH_WATER = {
    'gamma': 1.19,
    'pinf':  7.028e8,
    'b':     6.61e-4,
    'kv':    3610.0,
    'eta':   -1.177788e6,
}

# ---------------------------------------------------------------------------
# Domain & initial conditions
# ---------------------------------------------------------------------------
def make_case(N, t_end, CFL=2.0, dt_fixed=None, verbose=True,
              output_times=None):
    """Build case_params for denner_1d.run() with sharp interface IC."""
    L = 1.0
    x_cells = np.linspace(L / (2 * N), L - L / (2 * N), N)

    p0, T0, u0 = 1.0e5, 293.15, 1.0

    # Sharp step: water in [0.4, 0.6], air elsewhere
    psi_init = np.where((x_cells >= 0.4) & (x_cells <= 0.6), 1.0, 0.0)
    # Clip to avoid pure-phase EOS singularities
    psi_init = np.clip(psi_init, 1e-8, 1.0 - 1e-8)

    p_init = np.full(N, p0)
    u_init = np.full(N, u0)
    T_init = np.full(N, T0)

    return {
        'ph1':             PH_WATER,
        'ph2':             PH_AIR,
        'x_cells':         x_cells,
        'psi_init':        psi_init,
        'p_init':          p_init,
        'u_init':          u_init,
        'T_init':          T_init,
        't_end':           t_end,
        'CFL':             CFL,
        'bc_left':         'periodic',
        'bc_right':        'periodic',
        'verbose':         verbose,
        'dt_fixed':        dt_fixed,
        'max_picard_iter': 5,    # sharp interface: fewer picard iters (less divergence risk)
        'picard_tol':      1e-4, # relaxed tolerance for sharp interface
        'output_times':    output_times if output_times is not None else [],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                            'results', '1D', 'sharp_advection')


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_fields(x_cells, state, psi_exact, t, ft_idx,
                p0=1e5, u0=1.0, T0=293.15, ph1=None, ph2=None, CFL=2.0):
    """Plot p, u, T, rho, alpha (with exact), Y vs x."""
    from solver.denner_1d.eos.base import compute_phase_props
    _ensure_dir(RESULTS_DIR)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(
        f't = {t:.3f} s  ({ft_idx} flow-through)  CFL={CFL}', fontsize=12)

    # Pressure
    p_err = state['p'] / p0 - 1.0
    axes[0].plot(x_cells, state['p'] / p0, 'b-o', ms=3)
    axes[0].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('p / p0')
    axes[0].set_title(f'Pressure  |max-err|={np.max(np.abs(p_err)):.2e}')

    # Velocity
    u_err = state['u'] / u0 - 1.0
    axes[1].plot(x_cells, state['u'] / u0, 'r-o', ms=3)
    axes[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('u / u0')
    axes[1].set_title(f'Velocity  |max-err|={np.max(np.abs(u_err)):.2e}')

    # Temperature
    T_err = state['T'] / T0 - 1.0
    axes[2].plot(x_cells, state['T'] / T0, 'm-o', ms=3)
    axes[2].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[2].set_xlabel('x [m]')
    axes[2].set_ylabel('T / T0')
    axes[2].set_title(f'Temperature  |max-err|={np.max(np.abs(T_err)):.2e}')

    # Density
    if 'rho' in state:
        axes[3].plot(x_cells, state['rho'], 'brown', marker='o', ms=3)
        axes[3].set_xlabel('x [m]')
        axes[3].set_ylabel('ρ [kg/m³]')
        axes[3].set_title(f'Density  max={np.max(state["rho"]):.2e}')

    # Volume fraction: numerical vs exact
    axes[4].plot(x_cells, psi_exact, 'k--', lw=1.5, label='exact')
    axes[4].plot(x_cells, state['psi'], 'g-o', ms=3, label='numerical')
    axes[4].set_xlabel('x [m]')
    axes[4].set_ylabel('α (water vol. frac.)')
    axes[4].set_title('Volume fraction α')
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].legend(fontsize=8)

    # Mass fraction Y_water
    if ph1 is not None and ph2 is not None and 'rho' in state:
        rho_mix = state['rho']
        pr1 = compute_phase_props(state['p'], state['T'], ph1)
        rho1 = pr1['rho']
        Y_water = state['psi'] * rho1 / np.maximum(rho_mix, 1e-30)
        axes[5].plot(x_cells, Y_water,       'c-o', ms=3, label='Y_water')
        axes[5].plot(x_cells, 1.0 - Y_water, 'k--o', ms=3, label='Y_air')
        axes[5].set_xlabel('x [m]')
        axes[5].set_ylabel('Y (mass fraction)')
        axes[5].set_title('Mass fraction Y')
        axes[5].set_ylim(-0.05, 1.05)
        axes[5].legend(fontsize=8)
    else:
        axes[5].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f'fields_ft{ft_idx:02d}.png')
    plt.savefig(fname, dpi=100)
    plt.close(fig)
    print(f'  [plot] Saved {fname}')


def plot_l2_history(ft_list, l2_list, pass_tol=1e-4, CFL=2.0):
    _ensure_dir(RESULTS_DIR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(ft_list, l2_list, 'b-o', label='L2(p)')
    ax.axhline(pass_tol, color='r', ls='--', label=f'tol={pass_tol:.0e}')
    ax.set_xlabel('Flow-throughs')
    ax.set_ylabel('L2(p) = √Σ(p/p0-1)²/N')
    ax.set_title(f'Pressure equilibrium error  CFL={CFL}')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    fname = os.path.join(RESULTS_DIR, 'L2p_history.png')
    plt.savefig(fname, dpi=100)
    plt.close(fig)
    print(f'  [plot] Saved {fname}')


# ---------------------------------------------------------------------------
# Exact sharp interface at time t (periodic, L=1, u0=1)
# ---------------------------------------------------------------------------
def sharp_exact(x_cells, t, L=1.0, u0=1.0,
                x_lo=0.4, x_hi=0.6):
    """Exact solution: sharp block shifted by u0*t (mod L), periodic."""
    shift = (u0 * t) % L
    x_shifted = (x_cells - shift) % L
    psi = np.where((x_shifted >= x_lo) & (x_shifted <= x_hi), 1.0, 0.0)
    return np.clip(psi, 1e-8, 1.0 - 1e-8)


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------
def l2_p(state, p0=1e5):
    N = len(state['p'])
    return float(np.sqrt(np.sum((state['p'] / p0 - 1.0) ** 2) / N))


def run_validation(N=50, CFL=2.0, max_ft=5, dt_fixed=None,
                   pass_tol=1e-4, verbose=False):
    """
    Run max_ft flow-throughs as a single continuous simulation.
    Domain L=1m, u=1m/s → 1 flow-through = 1 second.
    Returns True if all flow-throughs PASS L2(p) < pass_tol.
    """
    L, u0, p0, T0 = 1.0, 1.0, 1e5, 293.15
    ft_time = L / u0  # 1 s

    output_times = [ft_time * ft for ft in range(1, max_ft + 1)]
    params = make_case(N, max_ft * ft_time, CFL=CFL, dt_fixed=dt_fixed,
                       verbose=verbose, output_times=output_times)
    result = run(params)

    x_cells = params['x_cells']
    ft_list, l2_list = [], []
    all_pass = True

    print(f'\n{"="*60}')
    print(f'Sharp Interface Advection  N={N}  CFL={CFL}  max_ft={max_ft}')
    print(f'{"="*60}')

    if result.get('diverged'):
        print(f'  *** DIVERGED: {result["diverge_reason"]} ***')
        all_pass = False

    for ft in range(1, max_ft + 1):
        t_target = ft * ft_time

        # Find closest snapshot to this flow-through time
        snap = None
        best_dt = float('inf')
        for s in result['snapshots']:
            dt_snap = abs(s['t'] - t_target)
            if dt_snap < best_dt:
                best_dt = dt_snap
                snap = s

        # Skip if no snapshot close enough (diverged before this ft)
        if snap is None or best_dt > 0.5 * ft_time:
            print(f'  ft={ft:2d}  --- no snapshot (simulation ended early) ---  FAIL')
            all_pass = False
            continue

        state = {k: snap[k] for k in ('p', 'u', 'T', 'psi', 'rho') if k in snap}

        # Exact sharp interface at t_target
        psi_exact = sharp_exact(x_cells, t_target, L=L, u0=u0)

        # Check if state is physical before plotting
        if not (np.all(np.isfinite(state['p'])) and np.all(np.isfinite(state['u']))):
            print(f'  ft={ft:2d}  --- non-finite values in state ---  FAIL')
            all_pass = False
            continue

        # Interface sharpness: compare numerical vs exact alpha
        psi_num   = state['psi']
        psi_err   = np.max(np.abs(psi_num - psi_exact))

        l2 = l2_p(state, p0)
        ft_list.append(ft)
        l2_list.append(l2)

        ok = l2 < pass_tol
        status = 'PASS' if ok else 'FAIL'
        print(f'  ft={ft:2d}  t={snap["t"]:.2f}s  '
              f'L2(p)={l2:.3e}  |Δα|_max={psi_err:.3e}  {status}')

        plot_fields(x_cells, state, psi_exact, t_target, ft,
                    p0=p0, u0=u0, T0=T0,
                    ph1=params['ph1'], ph2=params['ph2'], CFL=CFL)

        if not ok:
            all_pass = False
            print(f'  *** FAIL at ft={ft}. ***')

    if ft_list:
        plot_l2_history(ft_list, l2_list, pass_tol=pass_tol, CFL=CFL)

    overall = 'PASS' if all_pass else 'FAIL'
    print(f'\nResult: {overall}  (ran {len(ft_list)} flow-throughs)\n')
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',        type=int,   default=50,
                        help='Number of cells (default 50)')
    parser.add_argument('--CFL',      type=float, default=2.0,
                        help='CFL number (default 2.0)')
    parser.add_argument('--max-ft',   type=int,   default=5,
                        help='Max flow-throughs (default 5)')
    parser.add_argument('--dt-fixed', type=float, default=None,
                        help='Fixed dt override')
    parser.add_argument('--tol',      type=float, default=1e-4,
                        help='L2(p) PASS tolerance (default 1e-4)')
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()

    ok = run_validation(
        N=args.N,
        CFL=args.CFL,
        max_ft=args.max_ft,
        dt_fixed=args.dt_fixed,
        pass_tol=args.tol,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
