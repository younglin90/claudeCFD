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
from .eos.base import compute_phase_props


def _compute_K(psi, ph1, ph2, p, T):
    """
    Denner 2018 compressibility factor K for the compressible VOF equation.

    K = (Z_b - Z_a) / (Z_a/(1-psi+eps) + Z_b/(psi+eps))

    where Z_k = rho_k * c_k^2 (acoustic impedance squared).

    Phase 1 (ph1) is treated as phase 'a' (psi = volume fraction of ph1).
    Phase 2 (ph2) is treated as phase 'b'.
    """
    eps = 1e-10
    props_a = compute_phase_props(p, T, ph1)
    props_b = compute_phase_props(p, T, ph2)
    rho_a = props_a['rho'];  c_a = props_a['c']
    rho_b = props_b['rho'];  c_b = props_b['c']
    Z_a = rho_a * c_a ** 2
    Z_b = rho_b * c_b ** 2
    denom = Z_a / np.maximum(1.0 - psi, eps) + Z_b / np.maximum(psi, eps)
    return (Z_b - Z_a) / np.maximum(denom, eps)


def _vof_substep(psi, u_face, psi_ext, dt_sub, dx, n_ghost,
                 ph1=None, ph2=None, p=None, T=None):
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

    # Compressible VOF: ∂ψ/∂t + ∇·(uψ) - (ψ + K)*∇·u = 0
    # where K = Denner 2018 compressibility factor (zero for incompressible flow).
    if ph1 is not None and ph2 is not None and p is not None and T is not None:
        K = _compute_K(psi, ph1, ph2, p, T)
        source = (psi + K) * du_dx
    else:
        source = psi * du_dx  # incompressible fallback

    psi_new = psi - dt_sub * flux_div + dt_sub * source
    # Clip to [psi_min, 1-psi_min] with psi_min=0.01.
    # Rationale: values below 1% are sub-grid; allowing psi→0 or psi→1 creates
    # a 900:1 density ratio at a single cell face which makes the Picard
    # iteration ill-conditioned and causes exponential error growth.
    # 0.01 clip limits the density ratio to ~89:1, which is numerically stable.
    psi_new = np.clip(psi_new, 0.0, 1.0)  # physical bounds only; 0.01 clip removed

    return psi_new, psi_face


def psi_to_Y(psi, rho1, rho2):
    """Volume fraction ψ → mass fraction Y = ρ₁ψ / (ρ₁ψ + ρ₂(1-ψ))."""
    rho_mix = psi * rho1 + (1.0 - psi) * rho2
    return rho1 * psi / np.maximum(rho_mix, 1e-300)


def Y_to_psi(Y, rho1, rho2):
    """Mass fraction Y → volume fraction ψ = Y·ρ₂ / (ρ₁(1-Y) + Y·ρ₂)."""
    denom = rho1 * (1.0 - Y) + Y * rho2
    return Y * rho2 / np.maximum(denom, 1e-300)


def mass_fraction_step(psi, u, dx, dt, bc_l, bc_r, rho1, rho2, n_ghost=2):
    """
    Mass fraction transport: advect Y = ρ₁ψ/ρ_mix using CICSAM, then recover ψ.
    ∂(ρY)/∂t + ∇·(ρuY) = 0.
    At constant (p,T): equivalent to advecting Y with flow velocity.
    Returns psi_new, psi_face, u_face (same interface as vof_step).
    """
    N = len(psi)
    ng = n_ghost

    # Convert ψ → Y
    Y = psi_to_Y(psi, rho1, rho2)

    # Face velocities
    u_ext = apply_ghost_velocity(u, bc_l, bc_r, ng)
    u_face = np.empty(N + 1)
    for f in range(N + 1):
        u_face[f] = 0.5 * (u_ext[ng + f - 1] + u_ext[ng + f])

    # Sub-stepping
    u_max = max(float(np.max(np.abs(u_face))), 1e-300)
    Co = u_max * dt / dx
    N_sub = max(1, int(np.ceil(Co)))
    dt_sub = dt / N_sub

    Y_cur = Y.copy()
    psi_face_accum = np.zeros(N + 1)

    for _ in range(N_sub):
        Y_ext = apply_ghost(Y_cur, bc_l, bc_r, ng)
        Y_face = cicsam_face(Y_ext, u_face, dt_sub, dx, ng)

        flux_R = Y_face[1:] * u_face[1:]
        flux_L = Y_face[:-1] * u_face[:-1]
        Y_cur = Y_cur - dt_sub * (flux_R - flux_L) / dx
        Y_cur = np.clip(Y_cur, 0.0, 1.0)

        # Convert Y_face → psi_face for mass-flux consistency
        psi_face_accum += Y_to_psi(Y_face, rho1, rho2)

    # Convert Y → ψ
    psi_new = Y_to_psi(Y_cur, rho1, rho2)
    psi_new = np.clip(psi_new, 0.0, 1.0)
    psi_face_avg = psi_face_accum / N_sub

    return psi_new, psi_face_avg, u_face


def vof_step(psi, u, dx, dt, bc_l, bc_r, n_ghost=2,
             ph1=None, ph2=None, p=None, T=None):
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
    ph1, ph2 : dict or None   EOS parameters (for compressible VOF K factor)
    p, T     : ndarray (N,) or None  pressure and temperature (for K computation)

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
        psi_cur, psi_face_sub = _vof_substep(
            psi_cur, u_face, psi_ext, dt_sub, dx, ng,
            ph1=ph1, ph2=ph2, p=p, T=T)
        psi_face_accum += psi_face_sub

    # Time-averaged psi_face for mass-flux consistency with the full dt
    psi_face_avg = psi_face_accum / N_sub

    return psi_cur, psi_face_avg, u_face
