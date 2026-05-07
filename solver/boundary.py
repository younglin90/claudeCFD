"""
Boundary condition utilities for the 1D all-speed solver (solver_1d.py).

Supported BC types:
    'periodic'      - wrap-around (left ghost = last cell, right ghost = first cell)
    'transmissive'  - zero-gradient extrapolation (copy)
    'wall'          - reflect normal velocity

Usage:
    U_ext = apply_bc_1d(U_cells, bc_left='periodic', bc_right='periodic')
    Returns array of shape (N+2, n_vars) with ghost cells prepended/appended.
"""

from __future__ import annotations
import numpy as np


def apply_bc_1d(
    U_cells: np.ndarray,
    bc_left: str = 'periodic',
    bc_right: str = 'periodic',
) -> np.ndarray:
    """
    Construct extended state array with one ghost cell on each side.

    Parameters
    ----------
    U_cells : np.ndarray, shape (N, n_vars)
        Interior cell conservative states.
    bc_left : str
        Left boundary condition type.
    bc_right : str
        Right boundary condition type.

    Returns
    -------
    U_ext : np.ndarray, shape (N+2, n_vars)
        Extended array: U_ext[0] = left ghost, U_ext[1:-1] = interior,
        U_ext[-1] = right ghost.
    """
    N, n_vars = U_cells.shape
    U_ext = np.empty((N + 2, n_vars), dtype=float)
    U_ext[1:-1] = U_cells

    # momentum index: layout is [rhoY_1,...,rhoY_Ns, rho*u, rho*E]
    # so rho*u is at index n_vars - 2
    mom_idx = n_vars - 2

    # ---- left ghost ----
    if bc_left == 'periodic':
        U_ext[0] = U_cells[-1]
    elif bc_left == 'transmissive':
        U_ext[0] = U_cells[0]
    elif bc_left == 'wall':
        U_ext[0] = U_cells[0].copy()
        U_ext[0, mom_idx] = -U_cells[0, mom_idx]   # reflect rho*u
    else:
        # default: transmissive
        U_ext[0] = U_cells[0]

    # ---- right ghost ----
    if bc_right == 'periodic':
        U_ext[-1] = U_cells[0]
    elif bc_right == 'transmissive':
        U_ext[-1] = U_cells[-1]
    elif bc_right == 'wall':
        U_ext[-1] = U_cells[-1].copy()
        U_ext[-1, mom_idx] = -U_cells[-1, mom_idx]  # reflect rho*u
    else:
        # default: transmissive
        U_ext[-1] = U_cells[-1]

    return U_ext
