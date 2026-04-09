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


def compute_compression_coefficients(phi, u_face, dx, dt_sub, bc_l, bc_r, n_ghost):
    """
    Compute frozen Zalesak FCT limiter C_k and interface normal n_hat.

    Returns
    -------
    C_k      : ndarray (N+1,)  per-face limiter [0,1]
    n_hat    : ndarray (N+1,)  sign(grad phi) at faces
    raw_flux : ndarray (N+1,)  raw compression flux (before limiting)
    """
    N = len(phi)
    ng = n_ghost
    is_per = (bc_l == 'periodic')
    phi_ext = apply_ghost(phi, bc_l, bc_r, ng)

    # --- Raw compression fluxes at faces ---
    grad_phi = np.zeros(N + 1)
    for f in range(N + 1):
        grad_phi[f] = (phi_ext[ng + f] - phi_ext[ng + f - 1]) / dx
    n_hat = np.sign(grad_phi)

    raw_flux = np.zeros(N + 1)
    for f in range(N + 1):
        iL = ng + f - 1
        iR = ng + f
        if n_hat[f] * abs(u_face[f]) >= 0:
            phi_f = phi_ext[iL]
        else:
            phi_f = phi_ext[iR]
        raw_flux[f] = abs(u_face[f]) * phi_f * (1.0 - phi_f) * n_hat[f]

    # --- Zalesak FCT limiting ---
    P_plus = np.zeros(N)
    P_minus = np.zeros(N)

    for i in range(N):
        contrib_R = -raw_flux[i + 1] / dx * dt_sub
        if contrib_R > 0:
            P_plus[i] += contrib_R
        else:
            P_minus[i] -= contrib_R
        contrib_L = raw_flux[i] / dx * dt_sub
        if contrib_L > 0:
            P_plus[i] += contrib_L
        else:
            P_minus[i] -= contrib_L

    Q_plus = np.maximum(1.0 - phi, 0.0)
    Q_minus = np.maximum(phi, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        R_plus = np.where(P_plus > 1e-300, np.minimum(Q_plus / P_plus, 1.0), 1.0)
        R_minus = np.where(P_minus > 1e-300, np.minimum(Q_minus / P_minus, 1.0), 1.0)

    C_k = np.zeros(N + 1)
    for f in range(N + 1):
        iL_cell = f - 1
        iR_cell = f
        if is_per:
            if iL_cell < 0:
                iL_cell = N - 1
            if iR_cell >= N:
                iR_cell = 0
        else:
            if iL_cell < 0:
                iL_cell = 0
            if iR_cell >= N:
                iR_cell = N - 1

        if raw_flux[f] > 0:
            C_k[f] = min(R_minus[iL_cell], R_plus[iR_cell])
        elif raw_flux[f] < 0:
            C_k[f] = min(R_plus[iL_cell], R_minus[iR_cell])
        else:
            C_k[f] = 1.0

    return C_k, n_hat, raw_flux


def _compression_flux_bounded(phi, u_face, dx, dt_sub, bc_l, bc_r, n_ghost):
    """
    Conservative, bounded compression flux using Zalesak FCT limiting.

    Raw flux: F = |u_face| · φ(1-φ) · sign(∇φ).
    Zalesak limiting ensures φ stays in [0, 1] while conserving ∫φ.

    Without limiting, compression pushes cells already at φ=1 above 1.0,
    and np.clip then destroys mass (∫φ drops each step).
    """
    C_k, n_hat, raw_flux = compute_compression_coefficients(
        phi, u_face, dx, dt_sub, bc_l, bc_r, n_ghost)

    limited_flux = C_k * raw_flux

    return (limited_flux[1:] - limited_flux[:-1]) / dx


def _vof_substep(psi, u_face, psi_ext, dt_sub, dx, n_ghost,
                 ph1=None, ph2=None, p=None, T=None,
                 use_compress=False, bc_l='periodic', bc_r='periodic'):
    """
    Single explicit VOF sub-step (Co ≤ 1 guaranteed by caller).
    """
    psi_face = cicsam_face(psi_ext, u_face, dt_sub, dx, n_ghost)

    flux_R = psi_face[1:]   * u_face[1:]
    flux_L = psi_face[:-1]  * u_face[:-1]
    flux_div = (flux_R - flux_L) / dx

    du_dx = (u_face[1:] - u_face[:-1]) / dx

    # Compressible VOF: ∂ψ/∂t + ∇·(uψ) - (ψ + K)*∇·u = 0
    if ph1 is not None and ph2 is not None and p is not None and T is not None:
        K = _compute_K(psi, ph1, ph2, p, T)
        source = (psi + K) * du_dx
    else:
        source = psi * du_dx

    psi_new = psi - dt_sub * flux_div + dt_sub * source

    # Anti-diffusion compression with Zalesak FCT limiting (conservative + bounded)
    # Applied to POST-ADVECTION field so the limiter sees the correct bounds.
    if use_compress:
        psi_new -= dt_sub * _compression_flux_bounded(
            psi_new, u_face, dx, dt_sub, bc_l, bc_r, n_ghost)

    psi_new = np.clip(psi_new, 0.0, 1.0)
    return psi_new, psi_face


def psi_to_Y(psi, rho1, rho2):
    """Volume fraction ψ → mass fraction Y = ρ₁ψ / (ρ₁ψ + ρ₂(1-ψ))."""
    rho_mix = psi * rho1 + (1.0 - psi) * rho2
    return rho1 * psi / np.maximum(rho_mix, 1e-300)


def Y_to_psi(Y, rho1, rho2):
    """Mass fraction Y → volume fraction ψ = Y·ρ₂ / (ρ₁(1-Y) + Y·ρ₂)."""
    denom = rho1 * (1.0 - Y) + Y * rho2
    return Y * rho2 / np.maximum(denom, 1e-300)


def mass_fraction_step(psi, u, dx, dt, bc_l, bc_r, rho1, rho2, n_ghost=2,
                       use_compress=False, return_Y=False):
    """
    Mass fraction transport: advect Y = ρ₁ψ/ρ_mix using CICSAM, then recover ψ.
    ∂(ρY)/∂t + ∇·(ρuY) = 0.
    At constant (p,T): equivalent to advecting Y with flow velocity.

    Parameters
    ----------
    return_Y : bool
        If True, return (Y_new, Y_face_avg, u_face) skipping ψ conversion.
        If False (default), return (psi_new, psi_face_avg, u_face).

    Returns
    -------
    (psi_new or Y_new), face_avg, u_face
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
        if use_compress:
            Y_cur -= dt_sub * _compression_flux_bounded(
                Y_cur, u_face, dx, dt_sub, bc_l, bc_r, ng)
        Y_cur = np.clip(Y_cur, 0.0, 1.0)

        psi_face_accum += Y_to_psi(Y_face, rho1, rho2)

    Y_cur = np.clip(Y_cur, 0.0, 1.0)

    if return_Y:
        Y_face_avg = psi_face_accum / N_sub  # averaged psi_face; caller can ignore if using Y
        return Y_cur, Y_face_avg, u_face

    # Convert Y → ψ
    psi_new = Y_to_psi(Y_cur, rho1, rho2)
    psi_new = np.clip(psi_new, 0.0, 1.0)
    psi_face_avg = psi_face_accum / N_sub

    return psi_new, psi_face_avg, u_face


def vof_step_multi(phi_arr, u, dx, dt, bc_l, bc_r, n_ghost=2, use_compress=False):
    """
    Multi-species VOF explicit step. Advects each species independently.

    Parameters
    ----------
    phi_arr : ndarray (N_s-1, N) — independent species fractions
    u       : ndarray (N,)       — cell velocities

    Returns
    -------
    phi_new      : ndarray (N_s-1, N)
    phi_face_arr : ndarray (N_s-1, N+1)
    u_face       : ndarray (N+1,)
    """
    N_s_m1 = phi_arr.shape[0]
    N      = phi_arr.shape[1]
    phi_new      = np.empty_like(phi_arr)
    phi_face_arr = np.empty((N_s_m1, N + 1))
    u_face_out   = None
    for k in range(N_s_m1):
        pn, pf, uf = vof_step(phi_arr[k], u, dx, dt, bc_l, bc_r,
                               n_ghost=n_ghost, use_compress=use_compress)
        phi_new[k]      = pn
        phi_face_arr[k] = pf
        u_face_out      = uf
    return phi_new, phi_face_arr, u_face_out


def vof_step(psi, u, dx, dt, bc_l, bc_r, n_ghost=2,
             ph1=None, ph2=None, p=None, T=None,
             use_compress=False):
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
            ph1=ph1, ph2=ph2, p=p, T=T,
            use_compress=use_compress, bc_l=bc_l, bc_r=bc_r)
        psi_face_accum += psi_face_sub

    # Time-averaged psi_face for mass-flux consistency with the full dt
    psi_face_avg = psi_face_accum / N_sub

    return psi_cur, psi_face_avg, u_face
