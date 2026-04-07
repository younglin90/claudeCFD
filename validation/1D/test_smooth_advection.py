"""
Validation: 1D Smooth Interface Advection — Case C (Air/Water NASG)
Ref: 1D_smooth_interface-advection.md §C

Runs incrementally: 1 flow-through → 2 → ... → max_flow_through.
Plots p, u, ψ fields and L2(p) error history after each flow-through.
PASS criterion: L2(p) < 1e-4 at each checked time.

Usage:
    python validation/1D/test_smooth_advection.py [--N 50] [--max-ft 5] [--dt-fixed 0.01]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from solver.denner_1d.main import run

# ---------------------------------------------------------------------------
# EOS parameters
# ---------------------------------------------------------------------------
PH_AIR = {
    'gamma': 1.4,
    'pinf':  0.0,
    'b':     0.0,
    'kv':    717.5,   # c_v  [J/(kg·K)]
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
def make_case(N, t_end, dt_fixed=None, verbose=True, output_path=None,
              prev_state=None, output_times=None):
    """Build case_params for denner_1d.run().

    If prev_state is given (from a prior run), use it as initial condition
    (enables multi-segment flow-through runs).
    """
    L = 1.0
    x_cells = np.linspace(L / (2 * N), L - L / (2 * N), N)
    dx = L / N

    # Initial condition (smooth tanh interface)
    x_c, r_c, k = 0.5, 0.25, 15.0
    p0, T0, u0 = 1.0e5, 293.15, 1.0

    # Phase densities at (p0, T0)
    from solver.denner_1d.eos.base import compute_phase_props
    pr_air   = compute_phase_props(np.array([p0]), np.array([T0]), PH_AIR)
    pr_water = compute_phase_props(np.array([p0]), np.array([T0]), PH_WATER)
    rho_air   = float(pr_air['rho'][0])
    rho_water = float(pr_water['rho'][0])

    # w_Air from spec: choose so that rhoY_air = rho_air*(1-psi)
    # psi = alpha_water = volume fraction of water
    r = np.abs(x_cells - x_c)
    psi_init = 0.5 * (1.0 + np.tanh(k * (r - r_c)))   # water fraction
    psi_init = np.clip(psi_init, 1e-8, 1.0 - 1e-8)

    if prev_state is not None:
        p_init   = prev_state['p'].copy()
        u_init   = prev_state['u'].copy()
        T_init   = prev_state['T'].copy()
        psi_init = prev_state['psi'].copy()
    else:
        p_init = np.full(N, p0)
        u_init = np.full(N, u0)
        T_init = np.full(N, T0)

    return {
        'ph1':          PH_WATER,   # phase 1 = water (psi = water volume fraction)
        'ph2':          PH_AIR,     # phase 2 = air
        'x_cells':      x_cells,
        'psi_init':     psi_init,
        'p_init':       p_init,
        'u_init':       u_init,
        'T_init':       T_init,
        't_end':        t_end,
        'CFL':          1.0,
        'bc_left':      'periodic',
        'bc_right':     'periodic',
        'verbose':      verbose,
        'dt_fixed':     dt_fixed,
        'max_picard_iter': 20,
        'picard_tol':   1e-6,
        'output_path':  output_path,
        'output_times': output_times if output_times is not None else [],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..',
                            'results', '1D', 'smooth_advection')

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_fields(x_cells, state, t, ft_idx, p0=1e5, u0=1.0, T0=293.15,
                ph1=None, ph2=None):
    """Plot p, u, T, rho, Y (mass fraction), alpha (volume fraction) vs x."""
    from solver.denner_1d.eos.base import compute_phase_props
    _ensure_dir(RESULTS_DIR)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(f't = {t:.3f} s  ({ft_idx} flow-through)', fontsize=12)

    # Pressure
    p_err = (state['p'] / p0 - 1.0)
    axes[0].plot(x_cells, state['p'] / p0, 'b-o', ms=3)
    axes[0].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('p / p0')
    axes[0].set_title(f'Pressure  |max-err|={np.max(np.abs(p_err)):.2e}')

    # Velocity
    axes[1].plot(x_cells, state['u'] / u0, 'r-o', ms=3)
    axes[1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('u / u0')
    u_err = (state['u'] / u0 - 1.0)
    axes[1].set_title(f'Velocity  |max-err|={np.max(np.abs(u_err)):.2e}')

    # Temperature
    T_err = (state['T'] / T0 - 1.0)
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

    # Volume fraction alpha (= psi = water VOF)
    axes[4].plot(x_cells, state['psi'], 'g-o', ms=3)
    axes[4].set_xlabel('x [m]')
    axes[4].set_ylabel('α (water vol. frac.)')
    axes[4].set_title('Volume fraction α')
    axes[4].set_ylim(-0.05, 1.05)

    # Mass fraction Y_water = psi*rho1 / rho_mix
    if ph1 is not None and ph2 is not None and 'rho' in state:
        rho_mix = state['rho']
        pr1 = compute_phase_props(state['p'], state['T'], ph1)
        rho1 = pr1['rho']
        Y_water = state['psi'] * rho1 / np.maximum(rho_mix, 1e-30)
        axes[5].plot(x_cells, Y_water, 'c-o', ms=3, label='Y_water')
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


def plot_l2_history(ft_list, l2_list, pass_tol=1e-4):
    """Plot L2(p) error vs flow-through index."""
    _ensure_dir(RESULTS_DIR)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(ft_list, l2_list, 'b-o', label='L2(p)')
    ax.axhline(pass_tol, color='r', ls='--', label=f'tol={pass_tol:.0e}')
    ax.set_xlabel('Flow-throughs')
    ax.set_ylabel('L2(p) = √Σ(p/p0-1)²/N')
    ax.set_title('Pressure equilibrium error vs time')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    fname = os.path.join(RESULTS_DIR, 'L2p_history.png')
    plt.savefig(fname, dpi=100)
    plt.close(fig)
    print(f'  [plot] Saved {fname}')


# ---------------------------------------------------------------------------
# Main test loop
# ---------------------------------------------------------------------------
def l2_p(state, p0=1e5):
    N = len(state['p'])
    return float(np.sqrt(np.sum((state['p'] / p0 - 1.0)**2) / N))


def run_incremental(N=50, max_ft=5, dt_fixed=None, pass_tol=1e-4, verbose=False):
    """
    Run max_ft flow-throughs as a SINGLE continuous simulation with periodic output.
    Domain L=1m, u=1m/s → 1 flow-through = 1 second.
    Returns True if all PASS, False otherwise.

    Note: uses a single run() call with output_times=[1, 2, ..., max_ft] to avoid
    restart-induced perturbations that accumulate when splitting into segments.
    """
    L, u0, p0, T0 = 1.0, 1.0, 1e5, 293.15
    ft_time = L / u0        # 1 s per flow-through

    output_times = [ft_time * ft for ft in range(1, max_ft + 1)]
    params = make_case(N, max_ft * ft_time, dt_fixed=dt_fixed, verbose=verbose,
                       output_times=output_times)
    result = run(params)

    # Extract snapshots at each flow-through time
    x_cells = params['x_cells']
    ft_list, l2_list = [], []
    all_pass = True

    print(f'\n{"="*60}')
    print(f'Smooth Advection Validation  N={N}  max_ft={max_ft}')
    print(f'{"="*60}')

    for ft in range(1, max_ft + 1):
        t_target = ft * ft_time
        # Find snapshot closest to t_target
        snap = None
        for s in result['snapshots']:
            if abs(s['t'] - t_target) < 0.5 * ft_time:
                snap = s
        if snap is None:
            snap = result['snapshots'][-1]

        # snap stores fields directly (p, u, T, psi, rho)
        state = {k: snap[k] for k in ('p', 'u', 'T', 'psi', 'rho') if k in snap}
        l2 = l2_p(state, p0)
        ft_list.append(ft)
        l2_list.append(l2)

        ok = l2 < pass_tol
        status = 'PASS' if ok else 'FAIL'
        print(f'  ft={ft:2d}  t={snap["t"]:.2f}s  L2(p)={l2:.3e}  {status}')

        plot_fields(x_cells, state, t_target, ft, p0=p0, u0=u0, T0=T0,
                    ph1=params['ph1'], ph2=params['ph2'])

        if not ok:
            all_pass = False
            print(f'  *** FAIL at ft={ft}. ***')

    plot_l2_history(ft_list, l2_list, pass_tol=pass_tol)

    overall = 'PASS' if all_pass else 'FAIL'
    print(f'\nResult: {overall}  (ran {len(ft_list)} flow-throughs)\n')
    return all_pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N',        type=int,   default=50,
                        help='Number of cells (default 50)')
    parser.add_argument('--max-ft',   type=int,   default=5,
                        help='Max flow-throughs (default 5)')
    parser.add_argument('--dt-fixed', type=float, default=None,
                        help='Fixed time step (default: CFL-based)')
    parser.add_argument('--tol',      type=float, default=1e-4,
                        help='L2(p) PASS tolerance (default 1e-4)')
    parser.add_argument('--verbose',  action='store_true',
                        help='Print per-step info')
    args = parser.parse_args()

    ok = run_incremental(
        N=args.N,
        max_ft=args.max_ft,
        dt_fixed=args.dt_fixed,
        pass_tol=args.tol,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
