"""HLLC Riemann face flux for v2 R6.

5-equation Allaire-Massoni model with the Toro 1994 HLLC Riemann solver
(extended to two-phase per Saurel-Petitpas-Berry 2009 §3.3).

The contact-preservation property is *exact*: S* (the contact wave speed)
equals the mass-weighted average velocity at the face.  In a PE state
(u uniform, p uniform, α-jump) the formula reduces to:
  S_L = u₀ − c_L,  S_R = u₀ + c_R,  S* = u₀
  ρ_K* = ρ_K (no change),  p* = p_K (no change)
  U_K* = U_K with α-component shifted by transport at S*
This gives F = F_upwind exactly — no spurious smearing of the contact.

Wave speed estimates — Davis (no free parameter):
  S_L = min(u_L − c_L, u_R − c_R)
  S_R = max(u_L + c_L, u_R + c_R)

HLLC face flux (per Toro et al. 1994):
  F = F_L                        if 0 ≤ S_L
      F_L + S_L · (U_L* − U_L)   if S_L ≤ 0 ≤ S*
      F_R + S_R · (U_R* − U_R)   if S* ≤ 0 ≤ S_R
      F_R                        if S_R ≤ 0

For the 5-equation model the per-side star state is (Saurel-Petitpas-Berry 2009):
  factor_K = (S_K − u_K) / (S_K − S*)
  ρ_K*       = ρ_K · factor_K
  α_K*       = α_K                                (volume fraction passive at contact)
  (αρ_k)_K*  = (αρ_k)_K · factor_K                (per-species)
  (ρu)_K*    = ρ_K* · S*
  (ρE)_K*    = ρ_K · factor_K · (E_K + (S* − u_K)·(S* + p_K / (ρ_K(S_K − u_K))))

Free parameters: 0 (Davis estimate fixed; HLLC closure fixed).
"""
from __future__ import annotations
import numpy as np

from .boundary import extend_W


__all__ = ['compute_face_flux_hllc', 'cell_max_wave_speed', 'cell_sound_speed']


_EPS = 1e-30


def _phase_state(W, eos1, eos2):
    """Build (ρ_k, e_k, c_mix) from primitive W = (α, T1, T2, u, p)."""
    alpha, T1, T2, u, p = W
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1   = eos1.energy(rho1, p)
    e2   = eos2.energy(rho2, p)
    c1_sq = np.maximum(eos1.sound_speed_sq(rho1, e1, p), _EPS)
    c2_sq = np.maximum(eos2.sound_speed_sq(rho2, e2, p), _EPS)
    beta = 1.0 - alpha
    rho  = alpha * rho1 + beta * rho2
    rhoe = alpha * rho1 * e1 + beta * rho2 * e2
    rhoE = rhoe + 0.5 * rho * u * u
    c_mix_sq = (alpha * rho1 * c1_sq + beta * rho2 * c2_sq) / np.maximum(rho, _EPS)
    c_mix = np.sqrt(np.maximum(c_mix_sq, _EPS))
    return dict(rho1=rho1, rho2=rho2, e1=e1, e2=e2,
                rho=rho, rhoe=rhoe, rhoE=rhoE, c=c_mix)


def cell_sound_speed(W, eos1, eos2):
    """Per-cell mixture sound speed (frozen)."""
    return _phase_state(W, eos1, eos2)['c']


def cell_max_wave_speed(W, eos1, eos2):
    """Per-cell |u| + c_mix (frozen) for CFL."""
    return np.abs(W[3]) + cell_sound_speed(W, eos1, eos2)


def _conservative_flux(W, ph):
    """F(W) for cells (or face-cell sides) — per-cell conservative flux."""
    alpha = W[0]
    u, p  = W[3], W[4]
    beta  = 1.0 - alpha
    F = np.empty((5, alpha.shape[0]), dtype=float)
    F[0] = alpha * ph['rho1'] * u
    F[1] = beta  * ph['rho2'] * u
    F[2] = ph['rho'] * u * u + p
    F[3] = (ph['rhoE'] + p) * u
    F[4] = alpha * u
    return F


def compute_face_flux_hllc(W_ext, eos1, eos2):
    """HLLC face flux on every interior face.

    Parameters
    ----------
    W_ext : 5-tuple of (N+2,) arrays
        Ghost-extended primitive (α, T1, T2, u, p) with ng = 1.
    eos1, eos2 : EOS objects

    Returns
    -------
    F : (5, Nf) ndarray   (Nf = N + 1 interior faces)
    """
    # Slice into left / right of every face
    W_L = tuple(W_ext[k][:-1] for k in range(5))
    W_R = tuple(W_ext[k][1:]  for k in range(5))

    ph_L = _phase_state(W_L, eos1, eos2)
    ph_R = _phase_state(W_R, eos1, eos2)
    rho_L, rho_R = ph_L['rho'], ph_R['rho']
    rhoE_L, rhoE_R = ph_L['rhoE'], ph_R['rhoE']
    u_L, u_R = W_L[3], W_R[3]
    p_L, p_R = W_L[4], W_R[4]
    c_L, c_R = ph_L['c'], ph_R['c']

    # Davis wave speed estimates
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    # Contact wave speed S* (HLLC eq 10.37 in Toro)
    SL_uL = S_L - u_L                         # ≤ 0 in subsonic case
    SR_uR = S_R - u_R                         # ≥ 0
    num_star = (p_R - p_L
                + rho_L * u_L * SL_uL
                - rho_R * u_R * SR_uR)
    den_star = rho_L * SL_uL - rho_R * SR_uR
    # Guard the (rare) degenerate denominator
    S_star = num_star / np.where(np.abs(den_star) > _EPS,
                                 den_star,
                                 np.sign(den_star) * _EPS + _EPS)

    # Conservative U_L, U_R (5 components)
    alpha_L = W_L[0]; alpha_R = W_R[0]
    beta_L  = 1.0 - alpha_L; beta_R = 1.0 - alpha_R
    U_L = (alpha_L * ph_L['rho1'],
           beta_L  * ph_L['rho2'],
           rho_L * u_L,
           rhoE_L,
           alpha_L)
    U_R = (alpha_R * ph_R['rho1'],
           beta_R  * ph_R['rho2'],
           rho_R * u_R,
           rhoE_R,
           alpha_R)

    # Star states U*_K (Toro 10.39 generalised to 5-eq)
    factor_L = SL_uL / (S_L - S_star)
    factor_R = SR_uR / (S_R - S_star)
    # E_K (specific total energy)
    E_L = rhoE_L / np.maximum(rho_L, _EPS)
    E_R = rhoE_R / np.maximum(rho_R, _EPS)
    # rhoE_star_K = ρ_K · factor_K · (E_K + (S*−u_K)(S* + p_K/(ρ_K SL_uL)))
    rhoE_star_L = rho_L * factor_L * (
        E_L + (S_star - u_L) * (S_star + p_L / (rho_L * SL_uL))
    )
    rhoE_star_R = rho_R * factor_R * (
        E_R + (S_star - u_R) * (S_star + p_R / (rho_R * SR_uR))
    )

    U_star_L = (U_L[0] * factor_L,            # α₁ρ₁ * factor (per-species mass)
                U_L[1] * factor_L,            # α₂ρ₂ * factor
                rho_L * factor_L * S_star,    # ρu  = ρ* · S*
                rhoE_star_L,
                alpha_L)                      # α  passive (Saurel 2009)
    U_star_R = (U_R[0] * factor_R,
                U_R[1] * factor_R,
                rho_R * factor_R * S_star,
                rhoE_star_R,
                alpha_R)

    # Conservative flux on each side
    F_L = _conservative_flux(W_L, ph_L)
    F_R = _conservative_flux(W_R, ph_R)

    # HLLC sampling at x/t = 0:
    #   region_LL  : 0 <= S_L          → F_L
    #   region_LS  : S_L <= 0 <= S*    → F_L + S_L (U_L* - U_L)
    #   region_SR  : S* <= 0 <= S_R    → F_R + S_R (U_R* - U_R)
    #   region_RR  : S_R <= 0          → F_R
    F = np.empty_like(F_L)
    region_LL = S_L >= 0.0
    region_RR = S_R <= 0.0
    region_LS = (~region_LL) & (S_star >= 0.0)
    region_SR = (~region_RR) & (S_star <  0.0)
    for k in range(5):
        F[k] = np.where(
            region_LL, F_L[k],
            np.where(
                region_LS, F_L[k] + S_L * (U_star_L[k] - U_L[k]),
                np.where(
                    region_SR, F_R[k] + S_R * (U_star_R[k] - U_R[k]),
                    F_R[k]
                )
            )
        )
    return F
