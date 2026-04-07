# solver/demou2022_1d/cicsam.py
"""
CICSAM in 1D = Hyper-C scheme for volume fraction α₁ advection.

DENNER_SCHEME.md §6, Zanutto2022.

In 1D, the weighting factor w_f = 1 (purely compressive).
Face Courant number Co_f < 0.5 required for stability.

Notation:
    u_{i+1/2} > 0 → donor D = i, acceptor A = i+1, upwind-upwind UU = i-1
    u_{i+1/2} < 0 → donor D = i+1, acceptor A = i, upwind-upwind UU = i+2
"""

import numpy as np

_EPS = 1e-12


def cicsam_face(alpha_ext: np.ndarray, u_face: np.ndarray,
                dt: float, dx: float, n_ghost: int = 2) -> np.ndarray:
    """
    Compute CICSAM (Hyper-C) face values for α₁.

    Parameters
    ----------
    alpha_ext : (N + 2*n_ghost,) ghost-extended volume fraction
    u_face    : (N + 1,) face velocities
    dt        : time step
    dx        : cell width
    n_ghost   : ghost cell count on each side

    Returns
    -------
    alpha_face : (N + 1,) face values of α₁
    """
    N = len(u_face) - 1
    alpha_face = np.empty(N + 1)

    for f in range(N + 1):
        uf = u_face[f]
        Co_f = abs(uf) * dt / dx   # face Courant number

        L = f + n_ghost - 1
        R = f + n_ghost

        if uf >= 0.0:
            D, A, UU = L, R, L - 1
        else:
            D, A, UU = R, L, R + 1

        alpha_D  = alpha_ext[D]
        alpha_A  = alpha_ext[A]
        alpha_UU = alpha_ext[UU]

        denom = alpha_A - alpha_UU
        if abs(denom) < _EPS:
            # Uniform region — use donor value
            alpha_face[f] = alpha_D
            continue

        # Normalised donor value
        alpha_D_tilde = (alpha_D - alpha_UU) / denom

        # Hyper-C NVD formula
        if 0.0 <= alpha_D_tilde <= 1.0:
            alpha_f_tilde = min(alpha_D_tilde / max(Co_f, _EPS), 1.0)
        else:
            alpha_f_tilde = alpha_D_tilde

        # De-normalise
        alpha_face[f] = alpha_UU + alpha_f_tilde * denom

    # Clip to [0, 1]
    return np.clip(alpha_face, 0.0, 1.0)
