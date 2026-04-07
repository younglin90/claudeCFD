# solver/denner_1d/boundary.py
# Ref: DENNER_SCHEME.md § 13
#
# Ghost cell handling for 1D solver.
# Supported BC types: 'transmissive', 'periodic', 'wall'

import numpy as np


def apply_ghost(arr, bc_l, bc_r, n_ghost=2):
    """
    Extend a cell-centred array with ghost cells.

    Parameters
    ----------
    arr     : ndarray (N,)    interior cell values
    bc_l    : str             left BC  ('transmissive'|'periodic'|'wall')
    bc_r    : str             right BC ('transmissive'|'periodic'|'wall')
    n_ghost : int             number of ghost layers on each side

    Returns
    -------
    arr_ext : ndarray (N + 2*n_ghost,)
    """
    N = len(arr)
    arr_ext = np.empty(N + 2 * n_ghost, dtype=arr.dtype)
    arr_ext[n_ghost:n_ghost + N] = arr

    # Left ghost cells (indices 0..n_ghost-1)
    if bc_l == 'periodic':
        for g in range(n_ghost):
            arr_ext[n_ghost - 1 - g] = arr[N - 1 - g]
    elif bc_l in ('transmissive', 'wall'):
        for g in range(n_ghost):
            arr_ext[n_ghost - 1 - g] = arr[0]
    else:
        raise ValueError(f"Unknown BC type: {bc_l!r}")

    # Right ghost cells (indices n_ghost+N .. N+2*n_ghost-1)
    if bc_r == 'periodic':
        for g in range(n_ghost):
            arr_ext[n_ghost + N + g] = arr[g]
    elif bc_r in ('transmissive', 'wall'):
        for g in range(n_ghost):
            arr_ext[n_ghost + N + g] = arr[N - 1]
    else:
        raise ValueError(f"Unknown BC type: {bc_r!r}")

    return arr_ext


def apply_ghost_velocity(u, bc_l, bc_r, n_ghost=2):
    """
    Extend velocity array with ghost cells.
    Wall BC: ghost = -interior (no-slip / zero normal velocity).
    All other BCs: same as apply_ghost.

    Parameters
    ----------
    u       : ndarray (N,)
    bc_l    : str
    bc_r    : str
    n_ghost : int

    Returns
    -------
    u_ext : ndarray (N + 2*n_ghost,)
    """
    N = len(u)
    u_ext = np.empty(N + 2 * n_ghost, dtype=u.dtype)
    u_ext[n_ghost:n_ghost + N] = u

    # Left
    if bc_l == 'periodic':
        for g in range(n_ghost):
            u_ext[n_ghost - 1 - g] = u[N - 1 - g]
    elif bc_l == 'wall':
        for g in range(n_ghost):
            u_ext[n_ghost - 1 - g] = -u[g]
    elif bc_l == 'transmissive':
        for g in range(n_ghost):
            u_ext[n_ghost - 1 - g] = u[0]
    else:
        raise ValueError(f"Unknown BC type: {bc_l!r}")

    # Right
    if bc_r == 'periodic':
        for g in range(n_ghost):
            u_ext[n_ghost + N + g] = u[g]
    elif bc_r == 'wall':
        for g in range(n_ghost):
            u_ext[n_ghost + N + g] = -u[N - 1 - g]
    elif bc_r == 'transmissive':
        for g in range(n_ghost):
            u_ext[n_ghost + N + g] = u[N - 1]
    else:
        raise ValueError(f"Unknown BC type: {bc_r!r}")

    return u_ext


def get_face_bc_flags(bc_l, bc_r):
    """
    Return a tuple (left_is_periodic, right_is_periodic) for use in assembly.
    """
    return (bc_l == 'periodic'), (bc_r == 'periodic')
