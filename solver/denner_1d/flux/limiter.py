# solver/denner_1d/flux/limiter.py
# Ref: DENNER_SCHEME.md § 4.2 (consistent transport chain)
#
# Van Leer and Minmod TVD face reconstruction.
# Convention: face f (0..N) is between interior cell f-1 and f.
# phi_ext has shape (N + 2*n_ghost,).

import numpy as np


def _van_leer_limiter(r):
    """
    Van Leer limiter: psi = (r + |r|) / (1 + |r|).
    Accepts arrays.
    """
    r_abs = np.abs(r)
    return (r + r_abs) / (1.0 + r_abs + 1e-300)


def _minmod_limiter(r):
    """
    Minmod limiter: psi = max(0, min(1, r)).
    """
    return np.maximum(0.0, np.minimum(1.0, r))


def van_leer_face(phi_ext, u_face, n_ghost=2):
    """
    Compute face values using van Leer limiter (upwind-biased).

    Parameters
    ----------
    phi_ext : ndarray (N + 2*n_ghost,)   extended array with ghost cells
    u_face  : ndarray (N+1,)             face velocities (sign gives upwind direction)
    n_ghost : int

    Returns
    -------
    phi_face : ndarray (N+1,)
    """
    ng = n_ghost
    N = len(u_face) - 1
    phi_face = np.empty(N + 1)

    for f in range(N + 1):
        uf = u_face[f]
        # In extended array: left of face = ng + f - 1, right = ng + f
        iL = ng + f - 1
        iR = ng + f

        if uf >= 0.0:
            # Upwind is left cell: reconstruct from left
            phi_C  = phi_ext[iL]
            phi_D  = phi_ext[iL - 1]  # one cell further upwind
            phi_A  = phi_ext[iR]      # one cell downwind (acceptor)
            diff   = phi_C - phi_D
            scale  = max(abs(phi_C), abs(phi_D), abs(phi_A), 1e-30)
            if abs(diff) < 1e-12 * scale:
                # Flat region: first-order upwind
                phi_face[f] = phi_C
            else:
                r = (phi_A - phi_C) / diff
                psi = _van_leer_limiter(r)
                phi_face[f] = phi_C + 0.5 * psi * diff
        else:
            # Upwind is right cell: reconstruct from right
            phi_C  = phi_ext[iR]
            phi_D  = phi_ext[iR + 1]  # one cell further upwind
            phi_A  = phi_ext[iL]      # acceptor
            diff   = phi_C - phi_D
            scale  = max(abs(phi_C), abs(phi_D), abs(phi_A), 1e-30)
            if abs(diff) < 1e-12 * scale:
                phi_face[f] = phi_C
            else:
                r = (phi_A - phi_C) / diff
                psi = _van_leer_limiter(r)
                phi_face[f] = phi_C + 0.5 * psi * diff

    return phi_face


def minmod_face(phi_ext, u_face, n_ghost=2):
    """
    Compute face values using Minmod limiter (upwind-biased).

    Parameters
    ----------
    phi_ext : ndarray (N + 2*n_ghost,)
    u_face  : ndarray (N+1,)
    n_ghost : int

    Returns
    -------
    phi_face : ndarray (N+1,)
    """
    ng = n_ghost
    N = len(u_face) - 1
    phi_face = np.empty(N + 1)

    for f in range(N + 1):
        uf = u_face[f]
        iL = ng + f - 1
        iR = ng + f

        if uf >= 0.0:
            phi_C  = phi_ext[iL]
            phi_D  = phi_ext[iL - 1]
            phi_A  = phi_ext[iR]
            diff   = phi_C - phi_D
            scale  = max(abs(phi_C), abs(phi_D), abs(phi_A), 1e-30)
            if abs(diff) < 1e-12 * scale:
                phi_face[f] = phi_C
            else:
                r = (phi_A - phi_C) / diff
                psi = _minmod_limiter(r)
                phi_face[f] = phi_C + 0.5 * psi * diff
        else:
            phi_C  = phi_ext[iR]
            phi_D  = phi_ext[iR + 1]
            phi_A  = phi_ext[iL]
            diff   = phi_C - phi_D
            scale  = max(abs(phi_C), abs(phi_D), abs(phi_A), 1e-30)
            if abs(diff) < 1e-12 * scale:
                phi_face[f] = phi_C
            else:
                r = (phi_A - phi_C) / diff
                psi = _minmod_limiter(r)
                phi_face[f] = phi_C + 0.5 * psi * diff

    return phi_face
