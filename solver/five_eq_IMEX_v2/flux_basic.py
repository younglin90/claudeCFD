"""Conservative face flux for v2 R3 (active) — Five-equation Allaire–Massoni
with SLAU2-flavour all-Mach dissipation.

R3 design (one change vs R2 final):
  - Face primitive built by `face_upwind.face_upwind_state` (R2a face state).
  - Central face flux F_c built from EOS-consistent face primitive (PE-preserving
    on uniform-(p, u) states).
  - SLAU2 χ(M̂) = (1−M̂)² scales an LF-style upwind dissipation in U:
        F = F_c − ½ · (1 − χ(M̂)) · |λ|_face · (U_R − U_L)
    where |λ|_face = c_face + |u_face|, M̂ = |u_face|/c_face.

Properties:
  - χ → 1 at low Mach  ⇒ F = F_c, fully PE-preserving central flux.
  - χ → 0 at high Mach ⇒ F = F_c − full LF dissipation, advection-stable.
  - Free parameters: 0 (χ definition fixed; |λ| = c+|u|).

Note (R5 attempt and rollback, 2026-04-28):
  R5 tried wave-decomposed dissipation — |u| on mass and |u|+c on
  momentum/energy — to preserve contact discontinuities (intended
  HLLC/SLAU2 essence).  The result was a severe regression: in a
  Rusanov-form blend the small mass dissipation removed the numerical
  viscosity that stabilises forward-Euler + central-velocity advection,
  triggering von-Neumann instability.  Wave-decomposed dissipation is
  only consistent inside a true Riemann solver framework (HLLC) where
  each wave's contribution lives in its own characteristic field.
  Reverted to single |λ| = c + |u| for all components (R3 default).

Reference:
  Shima & Kitamura 2011 — SLAU/SLAU2 family, χ(M̂)=(1−M̂)² all-Mach scaling.
  Deng 2025 — SLAU2 in 5-equation FVM context.
"""
from __future__ import annotations
import numpy as np


__all__ = ['compute_face_flux', 'cell_max_wave_speed', 'cell_sound_speed']

_EPS = 1e-30


def _phase_face_state(face, eos1, eos2):
    """Evaluate EOS at the face primitive — used for the central flux."""
    alpha = face['alpha']
    T1    = face['T1']
    T2    = face['T2']
    u     = face['u']
    p     = face['p']

    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1   = eos1.energy(rho1, p)
    e2   = eos2.energy(rho2, p)

    beta = 1.0 - alpha
    rho  = alpha * rho1 + beta * rho2
    rhoe = alpha * rho1 * e1 + beta * rho2 * e2
    rhoE = rhoe + 0.5 * rho * u * u
    return rho1, rho2, e1, e2, rho, rhoe, rhoE


def _sound_speed_face(face, eos1, eos2, rho1, rho2, e1, e2, rho):
    """Frozen mixture sound speed at the face: c_mix² = (αρ₁c₁² + βρ₂c₂²)/ρ."""
    p = face['p']
    alpha = face['alpha']
    beta  = 1.0 - alpha
    c1_sq = np.maximum(eos1.sound_speed_sq(rho1, e1, p), _EPS)
    c2_sq = np.maximum(eos2.sound_speed_sq(rho2, e2, p), _EPS)
    c_sq = (alpha * rho1 * c1_sq + beta * rho2 * c2_sq) / np.maximum(rho, _EPS)
    return np.sqrt(np.maximum(c_sq, _EPS))


def compute_face_flux(face, eos1, eos2, *, U_L=None, U_R=None):
    """R3 face flux = central(PE-preserving) + (1-χ(M̂))·LF dissipation.

    Parameters
    ----------
    face : dict
        Output of `face_upwind.face_upwind_state` (R2a-style face primitive).
    eos1, eos2 : EOS objects
    U_L, U_R : 5-tuple of (Nf,) arrays, optional
        Cell conservative state on the left / right side of every face.
        When omitted (or set to None) the LF dissipation is skipped — the
        result is the pure R2a central flux.  When supplied, the SLAU2
        χ(M̂)-scaled LF dissipation is applied.

    Returns
    -------
    F : (5, Nf) ndarray
    rho_face : (Nf,) ndarray  (mixture density at the face — diagnostic only)
    """
    rho1, rho2, e1, e2, rho, rhoe, rhoE = _phase_face_state(face, eos1, eos2)

    alpha = face['alpha']
    u     = face['u']
    p     = face['p']
    beta  = 1.0 - alpha

    # Central (PE-preserving) flux
    F = np.empty((5, alpha.shape[0]), dtype=float)
    F[0] = alpha * rho1 * u
    F[1] = beta  * rho2 * u
    F[2] = rho * u * u + p
    F[3] = (rhoE + p) * u
    F[4] = alpha * u

    if U_L is None or U_R is None:
        return F, rho

    # SLAU2-flavour all-Mach LF dissipation (R3)
    c_face = _sound_speed_face(face, eos1, eos2, rho1, rho2, e1, e2, rho)
    M_hat  = np.abs(u) / np.maximum(c_face, _EPS)
    # χ(M̂) = (1 − M̂)² for M̂ < 1, 0 for M̂ ≥ 1
    chi = np.where(M_hat < 1.0, (1.0 - M_hat) ** 2, 0.0)
    lam = c_face + np.abs(u)
    diss_scale = 0.5 * (1.0 - chi) * lam               # (Nf,)

    # F_diss[k] = diss_scale · (U_R[k] − U_L[k])
    for k in range(5):
        F[k] -= diss_scale * (U_R[k] - U_L[k])

    return F, rho


def cell_sound_speed(W, eos1, eos2):
    """Per-cell mixture sound speed (frozen)."""
    alpha, T1, T2, u, p = W
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1   = eos1.energy(rho1, p)
    e2   = eos2.energy(rho2, p)
    c1_sq = np.maximum(eos1.sound_speed_sq(rho1, e1, p), _EPS)
    c2_sq = np.maximum(eos2.sound_speed_sq(rho2, e2, p), _EPS)
    beta = 1.0 - alpha
    rho  = alpha * rho1 + beta * rho2
    c_sq = (alpha * rho1 * c1_sq + beta * rho2 * c2_sq) / np.maximum(rho, _EPS)
    return np.sqrt(np.maximum(c_sq, _EPS))


def cell_max_wave_speed(W, eos1, eos2):
    """Per-cell |u| + c_mix for CFL.  Frozen mixture sound speed."""
    return np.abs(W[3]) + cell_sound_speed(W, eos1, eos2)
