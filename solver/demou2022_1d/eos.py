# solver/demou2022_1d/eos.py
"""
NASG (Noble-Abel Stiffened Gas) EOS — per phase.

Parameters per phase (dict with keys):
    gamma  : γ    — polytropic index
    pinf   : p∞   — stiffness pressure [Pa]
    b      : b    — co-volume [m³/kg]
    kv     : κᵥ   — specific heat at constant volume [J/(kg·K)]
    eta    : η    — reference energy [J/kg]

Ideal gas limit: pinf=0, b=0, eta=0.

Key relations (DENNER_SCHEME.md §2):
    ρ(p,T)  = (p+p∞) / [κᵥ T (γ-1) + b(p+p∞)]           Eq.(8c)
    c²      = γ(p+p∞) / [ρ(1-bρ)]                         Eq.(9)
    Γ       = (γ-1) / (1-bρ)                               Eq.(9)
    h       = κₚ T + b p + η       (κₚ = γ κᵥ)            Eq.(9)
    φ       = ∂ρ/∂T|p  = -ρ Γ κₚ / c²                     Eq.(7a)
    ζ       = ∂ρ/∂p|T  = 1/c² + Γ² κₚ T / c⁴              Eq.(7b)
    χ       = ∂p/∂ρ|E  = c² - Γ h                          Eq.(7c)
"""

import numpy as np


def _safe(arr, floor=1e-300):
    return np.where(np.abs(arr) < floor, floor, arr)


def density(p, T, ph):
    """ρ(p,T) — Eq.(8c)."""
    num = p + ph['pinf']
    den = ph['kv'] * T * (ph['gamma'] - 1.0) + ph['b'] * num
    return num / _safe(den)


def sound_speed_sq(p, rho, ph):
    """c² = γ(p+p∞) / [ρ(1-bρ)] — Eq.(9)."""
    bfac = np.maximum(1.0 - ph['b'] * rho, 1e-10)
    return ph['gamma'] * (p + ph['pinf']) / _safe(rho * bfac)


def gruneisen(rho, ph):
    """Γ = (γ-1)/(1-bρ) — Eq.(9)."""
    bfac = np.maximum(1.0 - ph['b'] * rho, 1e-10)
    return (ph['gamma'] - 1.0) / bfac


def enthalpy(T, p, ph):
    """h = κₚ T + b p + η — Eq.(9)."""
    kp = ph['gamma'] * ph['kv']
    return kp * T + ph['b'] * p + ph['eta']


def dphi_dT(rho, c2, Gamma, ph):
    """φ = ∂ρ/∂T|p = -ρ Γ κₚ / c² — Eq.(7a)."""
    kp = ph['gamma'] * ph['kv']
    return -rho * Gamma * kp / _safe(c2)


def dphi_dp(c2, Gamma, kp_val, T):
    """ζ = ∂ρ/∂p|T = 1/c² + Γ² κₚ T / c⁴ — Eq.(7b)."""
    c2s = _safe(c2)
    return 1.0 / c2s + Gamma**2 * kp_val * T / _safe(c2s**2)


def dp_drho_e(c2, Gamma, h):
    """χ = ∂p/∂ρ|E = c² - Γ h — Eq.(7c)."""
    return c2 - Gamma * h


def phase_props(p, T, ph):
    """
    Compute all per-phase thermodynamic quantities.

    Returns dict with keys:
        rho, c2, c, Gamma, h, kp, phi, zeta, chi, Cp
    """
    rho   = density(p, T, ph)
    c2    = sound_speed_sq(p, rho, ph)
    Gamma = gruneisen(rho, ph)
    h     = enthalpy(T, p, ph)
    kp    = ph['gamma'] * ph['kv']

    phi  = dphi_dT(rho, c2, Gamma, ph)
    zeta = dphi_dp(c2, Gamma, kp, T)
    chi  = dp_drho_e(c2, Gamma, h)

    return {
        'rho':   rho,
        'c2':    c2,
        'c':     np.sqrt(np.maximum(c2, 0.0)),
        'Gamma': Gamma,
        'h':     h,
        'kp':    kp,
        'phi':   phi,
        'zeta':  zeta,
        'chi':   chi,
    }
