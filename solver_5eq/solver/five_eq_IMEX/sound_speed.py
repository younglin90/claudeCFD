"""Phase + mixture sound speed from EOS (p, T) derivatives.

Per-phase formula (user spec §"phase sound speed"):

    Theta_k = (p / rho_k^2 * rho_p_k − e_p_k) / (e_T_k − p / rho_k^2 * rho_T_k)
    K_k     = rho_p_k + rho_T_k * Theta_k
    c_k^2   = 1 / K_k

Mixture options (user spec §"mixture sound speed"):
  - 'frozen'  : 1 / c_α^2  = α₁/c₁² + α₂/c₂²              (frozen alpha)
  - 'kapila'  : 1 / (ρ c²) = α₁/(ρ₁ c₁²) + α₂/(ρ₂ c₂²)   (pressure-eq, mass-frozen)

Phase 3 default = 'kapila' (matches Phase 9 D_K closure consistency).
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def phase_sound_speed_sq(eos, rho, T):
    """Per-phase c_k² from analytic (p,T) derivatives — robust to mild EOS drift."""
    p = eos.pressure_from_rhoT(rho, T)
    rho_p = eos.drhodp_T(rho, T)   # >0
    rho_T = eos.drhodT_p(rho, T)   # <0 typically
    e_p   = eos.dedp_T(rho, T)
    e_T   = eos.dedT_p(rho, T)
    pr2 = p / np.maximum(rho ** 2, _EPS)
    num = pr2 * rho_p - e_p
    den = e_T - pr2 * rho_T
    Theta = num / np.where(np.abs(den) > _EPS, den, _EPS)
    K = rho_p + rho_T * Theta
    return 1.0 / np.maximum(K, _EPS)


def mixture_sound_speed_sq(alpha1, rho1, c1_sq, rho2, c2_sq, *, kind='kapila'):
    """Return mixture c² for (α₁, ρ_k, c_k²)."""
    a2 = 1.0 - alpha1
    if kind == 'frozen':
        inv = alpha1 / np.maximum(c1_sq, _EPS) + a2 / np.maximum(c2_sq, _EPS)
        return 1.0 / np.maximum(inv, _EPS)
    elif kind == 'kapila':
        rho = alpha1 * rho1 + a2 * rho2
        inv_rho_c2 = (alpha1 / np.maximum(rho1 * c1_sq, _EPS)
                      + a2 / np.maximum(rho2 * c2_sq, _EPS))
        return 1.0 / np.maximum(rho * inv_rho_c2, _EPS)
    else:
        raise ValueError(f"Unknown mixture kind='{kind}' (use 'frozen' or 'kapila').")
