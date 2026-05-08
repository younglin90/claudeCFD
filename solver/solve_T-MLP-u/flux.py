"""Numerical face fluxes — work for any equation that exposes
`physical_flux`, `prim_to_cons`, `wave_speeds_lr`, `max_wave_speed`.

  upwind_advection — exact for linear scalar advection (a·n upwind side)
  llf              — Local Lax-Friedrichs (Rusanov), universal
  hllc_1d          — HLLC for 1D Euler (Toro 1994)

Free parameters: 0 (Davis wave speeds, fixed formula).
"""
from __future__ import annotations
import numpy as np


_EPS = 1e-30


def upwind_advection(eq, W_L, W_R, normal, points=None, face_velocity=None):
    """Pure-upwind flux for the linear advection equation.

    Velocity sampling priority:
      1. If `face_velocity` is supplied — use it directly (one vector per
         face, e.g. ½(a(x_o)+a(x_n)) cell-centre central average).
      2. Else if eq is variable-velocity and `points` provided — sample
         a(x_GP) analytically at the Gauss-quadrature point (default).
      3. Else fall back to constant `eq.velocity`.

    φ_f selection follows the sign of u_f = a·n (upwind upstream cell).
    """
    if face_velocity is not None:
        a = face_velocity
        a_dot_n = np.einsum('...i,...i->...', a, normal)
    elif getattr(eq, 'is_variable_velocity', False) and points is not None:
        a = eq.velocity_at(points)
        a_dot_n = np.einsum('...i,...i->...', a, normal)
    else:
        a_dot_n = np.einsum('i,...i->...', eq.velocity, normal)
    upwind_left = a_dot_n >= 0
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    return np.where(upwind_left, a_dot_n * U_L, a_dot_n * U_R)


def central(eq, W_L, W_R, normal, points=None):
    """Pure central flux:  F = ½ (F_L + F_R), no dissipation.

    Textbook central differencing — well-known to be unconditionally
    unstable for advection without limiting and oscillation-prone at
    discontinuities.  Provided here as a reference comparison only.
    """
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    try:
        F_L = eq.physical_flux(U_L, normal, points=points)
        F_R = eq.physical_flux(U_R, normal, points=points)
    except TypeError:
        F_L = eq.physical_flux(U_L, normal)
        F_R = eq.physical_flux(U_R, normal)
    return 0.5 * (F_L + F_R)


def llf(eq, W_L, W_R, normal, points=None):
    """Local Lax-Friedrichs (Rusanov)."""
    U_L = eq.prim_to_cons(W_L)
    U_R = eq.prim_to_cons(W_R)
    # Forward `points` only when the equation accepts it.
    try:
        F_L = eq.physical_flux(U_L, normal, points=points)
        F_R = eq.physical_flux(U_R, normal, points=points)
        lam = np.maximum(eq.max_wave_speed(U_L, normal, points=points),
                         eq.max_wave_speed(U_R, normal, points=points))
    except TypeError:
        F_L = eq.physical_flux(U_L, normal)
        F_R = eq.physical_flux(U_R, normal)
        lam = np.maximum(eq.max_wave_speed(U_L, normal),
                         eq.max_wave_speed(U_R, normal))
    return 0.5 * (F_L + F_R) - 0.5 * lam * (U_R - U_L)


def hllc_1d(eq, W_L, W_R, normal=None, points=None):
    """HLLC for 1D Euler (Toro et al. 1994), normal-aware.

    The Riemann problem is solved in the face-aligned frame (velocity
    component along +normal), then the resulting flux is rotated back
    to the original frame.  In 1D rotation = scalar multiply by n_x for
    the momentum component; mass and energy fluxes are scalars
    invariant under the rotation (already F·n).

    Wave speeds (Davis):  S_L = min(u_n_L − c_L, u_n_R − c_R)
                          S_R = max(u_n_L + c_L, u_n_R + c_R)
    """
    n_x = np.asarray(normal)[..., 0] if normal is not None else 1.0
    rho_L, u_L_orig, p_L = W_L[0], W_L[1], W_L[2]
    rho_R, u_R_orig, p_R = W_R[0], W_R[1], W_R[2]
    rho_L = np.maximum(rho_L, _EPS); rho_R = np.maximum(rho_R, _EPS)

    # Project velocity onto face normal (face-aligned frame).
    u_L = u_L_orig * n_x
    u_R = u_R_orig * n_x

    c_L = np.sqrt(np.maximum(eq.gamma * p_L / rho_L, _EPS))
    c_R = np.sqrt(np.maximum(eq.gamma * p_R / rho_R, _EPS))

    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    SL_uL = S_L - u_L
    SR_uR = S_R - u_R
    den = rho_L * SL_uL - rho_R * SR_uR
    den = np.where(np.abs(den) > _EPS, den, np.sign(den) * _EPS + _EPS)
    S_star = (p_R - p_L + rho_L * u_L * SL_uL - rho_R * u_R * SR_uR) / den

    # Total energy (frame-invariant scalar)
    E_L = p_L / ((eq.gamma - 1.0) * rho_L) + 0.5 * u_L_orig ** 2
    E_R = p_R / ((eq.gamma - 1.0) * rho_R) + 0.5 * u_R_orig ** 2

    # Conservative state in face-aligned frame: (ρ, ρ u_n, ρE)
    Uf_L = np.stack([rho_L, rho_L * u_L, rho_L * E_L], axis=0)
    Uf_R = np.stack([rho_R, rho_R * u_R, rho_R * E_R], axis=0)
    # Face-aligned physical flux: (ρ u_n, ρ u_n² + p, (ρE + p) u_n)
    Ff_L = np.stack([rho_L * u_L,
                     rho_L * u_L * u_L + p_L,
                     (rho_L * E_L + p_L) * u_L], axis=0)
    Ff_R = np.stack([rho_R * u_R,
                     rho_R * u_R * u_R + p_R,
                     (rho_R * E_R + p_R) * u_R], axis=0)

    factor_L = SL_uL / (S_L - S_star)
    factor_R = SR_uR / (S_R - S_star)
    rho_star_L = rho_L * factor_L
    rho_star_R = rho_R * factor_R
    rhoE_star_L = rho_L * factor_L * (E_L + (S_star - u_L) *
                                       (S_star + p_L / (rho_L * SL_uL)))
    rhoE_star_R = rho_R * factor_R * (E_R + (S_star - u_R) *
                                       (S_star + p_R / (rho_R * SR_uR)))
    Ufstar_L = np.stack([rho_star_L, rho_star_L * S_star, rhoE_star_L], axis=0)
    Ufstar_R = np.stack([rho_star_R, rho_star_R * S_star, rhoE_star_R], axis=0)

    Ff = np.where(
        S_L >= 0.0, Ff_L,
        np.where(
            S_star >= 0.0, Ff_L + S_L * (Ufstar_L - Uf_L),
            np.where(
                S_R >= 0.0, Ff_R + S_R * (Ufstar_R - Uf_R),
                Ff_R,
            )
        )
    )
    # Rotate back: only the momentum component depends on n_x.
    F = np.empty_like(Ff)
    F[0] = Ff[0]
    F[1] = Ff[1] * n_x
    F[2] = Ff[2]
    return F


# ─── Registry helper ───────────────────────────────────────────────────────
def get_flux(name: str):
    table = {
        'upwind':         upwind_advection,
        'upwind_advection': upwind_advection,
        'central':        central,
        'llf':            llf,
        'rusanov':        llf,
        'hllc':           hllc_1d,
        'hllc_1d':        hllc_1d,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"unknown flux '{name}'; available: {list(table)}")
    return table[name]
