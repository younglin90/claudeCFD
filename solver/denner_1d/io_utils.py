# solver/denner_1d/io_utils.py
# Ref: DENNER_SCHEME.md § 15
#
# Snapshot saving and result formatting utilities.

import numpy as np


def make_snapshot(t, state, x_cells):
    """
    Package the current solver state into a snapshot dict.

    Parameters
    ----------
    t       : float      simulation time [s]
    state   : dict       solver state (p, u, T, psi, rho, E_total, u_face)
    x_cells : ndarray    cell-centre positions

    Returns
    -------
    snap : dict
    """
    snap = {'t': float(t), 'x': x_cells.copy()}
    for key in ('p', 'u', 'T', 'psi', 'rho', 'E_total'):
        if key in state:
            snap[key] = state[key].copy()
    return snap


def save_snapshots_npz(path, snapshots):
    """
    Save a list of snapshots to a .npz file.

    Parameters
    ----------
    path      : str or Path
    snapshots : list of dict (from make_snapshot)
    """
    out = {}
    for i, snap in enumerate(snapshots):
        prefix = f"snap{i:05d}_"
        for key, val in snap.items():
            out[prefix + key] = np.asarray(val)
    np.savez_compressed(str(path), **out)


def print_step_info(step_num, t, dt, info, state):
    """
    Print per-step diagnostic information.

    Parameters
    ----------
    step_num : int
    t        : float   current time after step
    dt       : float   time step used
    info     : dict    solver info (converged, picard_iters, residuals)
    state    : dict    new state
    """
    conv = "OK" if info.get('converged', False) else "NO"
    nit  = info.get('picard_iters', '?')
    p_min, p_max = state['p'].min(), state['p'].max()
    u_min, u_max = state['u'].min(), state['u'].max()
    psi_min, psi_max = state['psi'].min(), state['psi'].max()
    print(
        f"  step={step_num:5d}  t={t:.4e}s  dt={dt:.3e}s  "
        f"picard={nit}({conv})  "
        f"p=[{p_min:.3e},{p_max:.3e}]  "
        f"u=[{u_min:.3e},{u_max:.3e}]  "
        f"psi=[{psi_min:.4f},{psi_max:.4f}]"
    )
