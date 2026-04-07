# solver/denner_1d/vof_cn.py
# Ref: DENNER_SCHEME.md § 7.2, Eq.(20)
#
# VOF explicit update using CICSAM face values.
# Auto sub-stepping: if Co = |u|·dt/dx > 1, the step is split into
# N_sub = ceil(Co) sub-steps each with dt_sub = dt/N_sub (Co ≤ 1 per sub-step).
# The (p,u,T) implicit step still uses the full dt.

import numpy as np
from .boundary import apply_ghost, apply_ghost_velocity
from .interface.cicsam import cicsam_face


def _vof_substep(psi, u_face, psi_ext, dt_sub, dx, n_ghost):
    """
    Single explicit VOF sub-step (Co ≤ 1 guaranteed by caller).

    ψ^{n+1} = ψ^n - dt/dx * [ψ̃_{i+1/2}·u_{i+1/2} - ψ̃_{i-1/2}·u_{i-1/2}]
                    + dt * ψ_i * [u_{i+1/2} - u_{i-1/2}] / dx
    """
    psi_face = cicsam_face(psi_ext, u_face, dt_sub, dx, n_ghost)

    flux_R = psi_face[1:]   * u_face[1:]
    flux_L = psi_face[:-1]  * u_face[:-1]
    flux_div = (flux_R - flux_L) / dx

    du_dx = (u_face[1:] - u_face[:-1]) / dx
    source = psi * du_dx

    psi_new = psi - dt_sub * flux_div + dt_sub * source
    psi_new = np.clip(psi_new, 1e-8, 1.0 - 1e-8)

    return psi_new, psi_face


def vof_step(psi, u, dx, dt, bc_l, bc_r, n_ghost=2):
    """
    VOF update using CICSAM face values, with automatic sub-stepping.

    When Co = max|u|·dt/dx > 1 (e.g. large-CFL implicit solve), the step
    is split into N_sub = ceil(Co) sub-steps so that each sub-step satisfies
    Co_sub ≤ 1.  The average psi_face across sub-steps is returned for use
    in the (p,u,T) assembly mass-flux consistency check.

    Parameters
    ----------
    psi    : ndarray (N,)   VOF field at time n
    u      : ndarray (N,)   cell velocities
    dx     : float
    dt     : float          full time step for this Mode-A iteration
    bc_l   : str
    bc_r   : str
    n_ghost: int

    Returns
    -------
    psi_new  : ndarray (N,)   updated VOF, clipped to [1e-8, 1-1e-8]
    psi_face : ndarray (N+1,) time-averaged CICSAM face VOF (for assembly reuse)
    u_face   : ndarray (N+1,) face velocities (arithmetic mean, constant in time)
    """
    N  = len(psi)
    ng = n_ghost

    # Face velocities (arithmetic mean, fixed throughout sub-steps)
    u_ext = apply_ghost_velocity(u, bc_l, bc_r, ng)
    u_face = np.empty(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        u_face[f] = 0.5 * (u_ext[iL] + u_ext[iR])

    # Number of sub-steps: ceil(Co) so that each sub-step has Co_sub ≤ 1
    u_face_fin = u_face[np.isfinite(u_face)]
    u_max = max(float(np.max(np.abs(u_face_fin))) if len(u_face_fin) > 0 else 0.0, 1e-300)
    Co = u_max * dt / dx
    if not np.isfinite(Co) or Co < 0:
        Co = 1.0
    N_sub = max(1, int(np.ceil(Co)))
    dt_sub = dt / N_sub

    psi_cur = psi.copy()
    psi_face_accum = np.zeros(N + 1)

    for _ in range(N_sub):
        psi_ext = apply_ghost(psi_cur, bc_l, bc_r, ng)
        psi_cur, psi_face_sub = _vof_substep(psi_cur, u_face, psi_ext, dt_sub, dx, ng)
        psi_face_accum += psi_face_sub

    # Time-averaged psi_face for mass-flux consistency with the full dt
    psi_face_avg = psi_face_accum / N_sub

    return psi_cur, psi_face_avg, u_face
