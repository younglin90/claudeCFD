# solver/denner_1d/interface/cicsam.py
# Ref: DENNER_SCHEME.md § 5, Eq.(18)
#
# CICSAM (Compressive Interface Capturing Scheme for Arbitrary Meshes)
# 1D implementation = Hyper-C scheme.
# Note: compression term is NOT added [Zanutto2022].

import numpy as np


def cicsam_face(psi_ext, u_face, dt, dx, n_ghost=2):
    """
    Compute face values of psi using CICSAM Hyper-C.

    Parameters
    ----------
    psi_ext  : ndarray (N + 2*n_ghost,)
               VOF field with ghost cells already applied.
    u_face   : ndarray (N+1,)
               Face velocities at interior faces i+1/2, i=0..N-1.
               Face f is between cells (f-1) and f in interior indexing
               (i.e., between psi_ext[n_ghost+f-1] and psi_ext[n_ghost+f]).
    dt       : float  time step [s]
    dx       : float  cell size [m]
    n_ghost  : int    number of ghost cells on each side

    Returns
    -------
    psi_face : ndarray (N+1,)
               CICSAM face values, clipped to [0, 1].
    """
    ng = n_ghost
    N = len(u_face) - 1  # number of interior cells
    psi_face = np.zeros(N + 1)

    for f in range(N + 1):
        uf = u_face[f]
        # Face f sits between interior cells f-1 and f.
        # In extended array: left cell = ng + f - 1, right cell = ng + f.

        if uf >= 0.0:
            # Flow goes right: donor = left cell, acceptor = right cell
            # upupwind = donor - 1
            i_D   = ng + f - 1   # donor
            i_A   = ng + f       # acceptor
            i_UU  = ng + f - 2   # upupwind (one cell further upstream)
        else:
            # Flow goes left: donor = right cell, acceptor = left cell
            # upupwind = donor + 1
            i_D   = ng + f       # donor
            i_A   = ng + f - 1   # acceptor
            i_UU  = ng + f + 1   # upupwind

        psi_D  = psi_ext[i_D]
        psi_A  = psi_ext[i_A]
        psi_UU = psi_ext[i_UU]

        # Normalised donor value [Eq. 18]
        denom = psi_A - psi_UU
        if abs(denom) < 1e-10:
            # Uniform region: upwind (donor value)
            psi_face[f] = psi_D
            continue

        psi_tilde_D = (psi_D - psi_UU) / denom

        # Local Courant number
        Co_f = abs(uf) * dt / dx

        # Hyper-C face normalised value [Eq. 18]
        if 0.0 <= psi_tilde_D <= 1.0:
            psi_tilde_f = min(psi_tilde_D / max(Co_f, 1e-10), 1.0)
        else:
            psi_tilde_f = psi_tilde_D

        # Un-normalise
        psi_face[f] = psi_UU + psi_tilde_f * denom

    # Clip to physical range
    psi_face = np.clip(psi_face, 0.0, 1.0)
    return psi_face
