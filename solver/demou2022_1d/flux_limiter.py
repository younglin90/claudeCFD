# solver/demou2022_1d/flux_limiter.py
"""
van Leer flux limiter for scalar advection (T, u, p).

DENNER_SCHEME.md §7.

Face value (u_{i+1/2} > 0 upwind from left):
    r = (φ_i - φ_{i-1}) / (φ_{i+1} - φ_i + ε)
    Ψ(r) = (r + |r|) / (1 + |r|)
    φ̃_{i+1/2} = φ_i + 0.5 Ψ(r) (φ_{i+1} - φ_i)

The ghost-extended array (with n_ghost ≥ 2) is required as input.
"""

import numpy as np

_EPS = 1e-12


def van_leer_face(phi_ext: np.ndarray, u_face: np.ndarray, n_ghost: int = 2) -> np.ndarray:
    """
    Compute face values using van Leer limiter.

    Parameters
    ----------
    phi_ext : (N + 2*n_ghost,) ghost-extended cell values
    u_face  : (N + 1,) face velocities (interior + boundary faces)
    n_ghost : ghost cell count on each side

    Returns
    -------
    phi_face : (N + 1,) face-interpolated values
    """
    N = len(u_face) - 1
    phi_face = np.empty(N + 1)

    for f in range(N + 1):
        # Cell indices in extended array
        # Face f sits between cells (f-1) and f in 0-indexed interior numbering.
        # In extended array: left cell = f + n_ghost - 1, right cell = f + n_ghost
        L = f + n_ghost - 1   # left cell index in ext
        R = f + n_ghost       # right cell index in ext

        uf = u_face[f]
        if uf >= 0.0:
            # Upwind from left: D=L, UU=L-1, A=R
            denom = phi_ext[R] - phi_ext[L]
            r = (phi_ext[L] - phi_ext[L - 1]) / (denom + _EPS * np.sign(denom + 1e-20))
        else:
            # Upwind from right: D=R, UU=R+1, A=L
            denom = phi_ext[L] - phi_ext[R]
            r = (phi_ext[R + 1] - phi_ext[R]) / (denom + _EPS * np.sign(denom + 1e-20))

        psi = (r + abs(r)) / (1.0 + abs(r))

        if uf >= 0.0:
            phi_face[f] = phi_ext[L] + 0.5 * psi * (phi_ext[R] - phi_ext[L])
        else:
            phi_face[f] = phi_ext[R] + 0.5 * psi * (phi_ext[L] - phi_ext[R])

    return phi_face
