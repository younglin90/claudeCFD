# solver/denner_1d/grid.py
# Ref: DENNER_SCHEME.md § 15 (file structure)
#
# Simple 1D uniform grid helpers.

import numpy as np


def make_uniform_grid(x_left, x_right, N):
    """
    Create a 1D uniform cell-centred grid.

    Parameters
    ----------
    x_left  : float   left domain boundary [m]
    x_right : float   right domain boundary [m]
    N       : int     number of interior cells

    Returns
    -------
    x_cells : ndarray (N,)   cell-centre positions [m]
    x_faces : ndarray (N+1,) face positions [m]
    dx      : float          uniform cell size [m]
    """
    dx      = (x_right - x_left) / N
    x_faces = np.linspace(x_left, x_right, N + 1)
    x_cells = 0.5 * (x_faces[:-1] + x_faces[1:])
    return x_cells, x_faces, dx


def smooth_vof_profile(x_cells, x_lo, x_hi, dx_interface=None):
    """
    Generate a smooth tanh VOF profile for a phase-1 region [x_lo, x_hi].

    α₁(x) = 0.5 * [tanh((x - x_lo)/δ) - tanh((x - x_hi)/δ)]

    Parameters
    ----------
    x_cells      : ndarray (N,)  cell-centre positions
    x_lo, x_hi   : float         physical extent of phase 1 region
    dx_interface  : float or None  interface half-thickness δ.
                                   Defaults to 0.5 * min cell spacing.

    Returns
    -------
    psi : ndarray (N,)   volume fraction, α₁ ∈ [0, 1]
    """
    if dx_interface is None:
        dx = np.min(np.diff(x_cells)) if len(x_cells) > 1 else abs(x_hi - x_lo) * 0.1
        dx_interface = 0.5 * dx

    delta = max(dx_interface, 1e-15)
    psi = 0.5 * (np.tanh((x_cells - x_lo) / delta)
                 - np.tanh((x_cells - x_hi) / delta))
    return np.clip(psi, 0.0, 1.0)
