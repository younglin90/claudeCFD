# solver/denner_1d/flux/mwi.py
# Ref: DENNER_SCHEME.md § 4.4-4.5, Eq.(17) and (17')
#
# MWI (Momentum-Weighted Interpolation) face velocity.
# ACID (Acoustically Conservative Interface Discretisation) face density.

import numpy as np
from ..boundary import apply_ghost, apply_ghost_velocity


def acid_face_density(rho, c_mix, psi, bc_l, bc_r, n_ghost=2):
    """
    Compute ACID face density at all N+1 faces (including boundary faces).

    At interfaces: ACID formula [Eq. 17']
    At smooth regions: harmonic mean

    Parameters
    ----------
    rho    : ndarray (N,)   mixture density
    c_mix  : ndarray (N,)   mixture sound speed
    psi    : ndarray (N,)   volume fraction (for interface detection)
    bc_l   : str
    bc_r   : str
    n_ghost: int

    Returns
    -------
    rho_face : ndarray (N+1,)
    """
    # Extend arrays with ghost cells
    rho_ext   = apply_ghost(rho,   bc_l, bc_r, n_ghost)
    c_ext     = apply_ghost(c_mix, bc_l, bc_r, n_ghost)
    psi_ext   = apply_ghost(psi,   bc_l, bc_r, n_ghost)

    N = len(rho)
    ng = n_ghost
    rho_face = np.empty(N + 1)

    for f in range(N + 1):
        # Left cell: ng + f - 1, Right cell: ng + f  (in extended array)
        iL = ng + f - 1
        iR = ng + f

        rho_L = rho_ext[iL]
        rho_R = rho_ext[iR]
        c_L   = c_ext[iL]
        c_R   = c_ext[iR]

        # Interface detection
        is_interface = abs(psi_ext[iR] - psi_ext[iL]) > 0.01

        if is_interface:
            # ACID formula [Eq. 17']
            num = rho_L * rho_R * (rho_L * c_L + rho_R * c_R)
            den = rho_L * rho_L * c_L + rho_R * rho_R * c_R + 1e-300
            rho_face[f] = num / den
        else:
            # Harmonic mean
            rho_face[f] = 2.0 * rho_L * rho_R / (rho_L + rho_R + 1e-300)

    return rho_face


def mwi_face_coeff(rho_face, dt):
    """
    Compute MWI coefficient d_hat_f = dt / rho_face [Eq. 17].

    Parameters
    ----------
    rho_face : ndarray (N+1,)
    dt       : float

    Returns
    -------
    d_hat : ndarray (N+1,)
    """
    return dt / (rho_face + 1e-300)


def mwi_face_velocity_components(u, p, rho_face, d_hat, dx,
                                  u_face_old, rho_face_old, dt,
                                  bc_l, bc_r, n_ghost=2,
                                  include_transient=True):
    """
    Decompose MWI face velocity into components for implicit assembly.

    The MWI face velocity is:
        u_face = u_arith_f - d_hat_f * (p_{i+1} - p_i) / dx
                 + transient_correction_f      [Eq. 17]

    For implicit assembly we return:
      - u_arith_f   : arithmetic mean of cell u at face (N+1,)
      - d_hat_f     : MWI pressure weight (N+1,)
      - correction  : transient MWI correction (N+1,) — goes to RHS

    Parameters
    ----------
    u              : ndarray (N,)   cell-centre velocities (Picard / current)
    p              : ndarray (N,)   cell-centre pressures  (Picard / current)
    rho_face       : ndarray (N+1,) ACID face density (current step)
    d_hat          : ndarray (N+1,) = dt / rho_face
    dx             : float
    u_face_old     : ndarray (N+1,) face velocity from previous time step
    rho_face_old   : ndarray (N+1,) ACID face density from previous time step
    dt             : float
    bc_l, bc_r     : str
    n_ghost        : int
    include_transient : bool   False for first time step (no previous u_face)

    Returns
    -------
    u_arith_f   : ndarray (N+1,)
    d_hat_f     : ndarray (N+1,)   (same as input d_hat, returned for clarity)
    correction  : ndarray (N+1,)   known correction for RHS
    u_face_full : ndarray (N+1,)   fully evaluated face velocity (for diagnostics)
    """
    N = len(u)
    ng = n_ghost

    u_ext = apply_ghost_velocity(u, bc_l, bc_r, ng)
    p_ext = apply_ghost(p, bc_l, bc_r, ng)

    u_arith_f = np.empty(N + 1)
    dp_f      = np.empty(N + 1)

    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        u_arith_f[f] = 0.5 * (u_ext[iL] + u_ext[iR])
        dp_f[f]      = (p_ext[iR] - p_ext[iL]) / dx

    # Transient correction [Eq. 17]: d_hat * rho_face_old/dt * (u_face_old - u_arith_old)
    if include_transient and (u_face_old is not None):
        u_old_ext = apply_ghost_velocity(u, bc_l, bc_r, ng)  # use current u as old (Picard)
        u_arith_old = np.empty(N + 1)
        for f in range(N + 1):
            iL = ng + f - 1
            iR = ng + f
            u_arith_old[f] = 0.5 * (u_old_ext[iL] + u_old_ext[iR])
        correction = d_hat * (rho_face_old / dt) * (u_face_old - u_arith_old)
    else:
        correction = np.zeros(N + 1)

    # Full face velocity (for diagnostics / deferred use)
    u_face_full = u_arith_f - d_hat * dp_f + correction

    return u_arith_f, d_hat, correction, u_face_full
