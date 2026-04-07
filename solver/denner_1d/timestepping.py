# solver/denner_1d/timestepping.py
# Ref: DENNER_SCHEME.md § 12, Eq.(32)
#
# Time step control for the implicit solver.
# Since the scheme is fully implicit, the acoustic CFL constraint is removed.
# Only convective CFL is enforced.

import numpy as np


def compute_dt(u, dx, CFL, u_min=1e-300):
    """
    Compute time step from convective CFL condition.

    Implicit solver → no acoustic CFL constraint.
    Phase 1 (inviscid): only convective constraint.

    dt = CFL * dx / max|u|

    Parameters
    ----------
    u   : ndarray (N,)  cell-centre velocities
    dx  : float         uniform cell size [m]
    CFL : float         target CFL number (typically 0.5)
    u_min : float       velocity floor to prevent division by zero

    Returns
    -------
    dt : float
    """
    u_max = max(np.max(np.abs(u)), u_min)
    return CFL * dx / u_max


def compute_dt_acoustic(u, c_mix, dx, CFL):
    """
    Compute time step including acoustic CFL (for explicit reference only).
    Not used for the implicit solver.

    dt = CFL * dx / max(|u| + c)

    Parameters
    ----------
    u     : ndarray (N,)   velocities
    c_mix : ndarray (N,)   mixture sound speeds
    dx    : float
    CFL   : float

    Returns
    -------
    dt : float
    """
    lambda_max = np.max(np.abs(u) + c_mix)
    lambda_max = max(lambda_max, 1e-300)
    return CFL * dx / lambda_max
