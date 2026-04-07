# solver/demou2022_1d/boundary.py
"""
Ghost-cell boundary conditions.

Supported: 'periodic', 'transmissive', 'wall'

All functions operate on 1-D cell arrays and return an extended
array with n_ghost ghost cells appended on each side.
"""

import numpy as np


def apply_ghost(arr: np.ndarray, bc_l: str, bc_r: str, n_ghost: int = 2) -> np.ndarray:
    """
    Extend cell array with ghost cells.

    Parameters
    ----------
    arr     : (N,) cell-centred values
    bc_l    : left  BC type  ('periodic' | 'transmissive' | 'wall')
    bc_r    : right BC type
    n_ghost : number of ghost cells on each side (≥ 2 for van Leer)

    Returns
    -------
    ext : (N + 2*n_ghost,) extended array
    """
    N = len(arr)
    ext = np.empty(N + 2 * n_ghost, dtype=arr.dtype)
    ext[n_ghost:n_ghost + N] = arr

    for g in range(n_ghost):
        ig = n_ghost - 1 - g   # ghost index from left boundary outward

        # Left ghost
        if bc_l == 'periodic':
            ext[ig] = arr[N - 1 - g]
        elif bc_l == 'wall':
            ext[ig] = arr[g]   # zero-gradient (wall normal velocity handled separately)
        else:  # transmissive
            ext[ig] = arr[g]

        # Right ghost
        rg = n_ghost + N + g   # ghost index from right boundary outward
        if bc_r == 'periodic':
            ext[rg] = arr[g]
        elif bc_r == 'wall':
            ext[rg] = arr[N - 1 - g]
        else:  # transmissive
            ext[rg] = arr[N - 1 - g]

    return ext


def apply_ghost_velocity(u: np.ndarray, bc_l: str, bc_r: str, n_ghost: int = 2) -> np.ndarray:
    """
    Ghost extension for velocity with wall reflection.
    """
    ext = apply_ghost(u, bc_l, bc_r, n_ghost)
    N = len(u)
    for g in range(n_ghost):
        if bc_l == 'wall':
            ext[n_ghost - 1 - g] = -u[g]
        if bc_r == 'wall':
            ext[n_ghost + N + g] = -u[N - 1 - g]
    return ext
