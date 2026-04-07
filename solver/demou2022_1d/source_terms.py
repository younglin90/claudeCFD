# solver/demou2022_1d/source_terms.py
"""
Mixture sound speed and source-term coefficients S^(2), S^(3).

Ref: DENNER_SCHEME.md §1.3, Demou2022 Eqs.(3)-(7).

Given per-phase props (from eos.phase_props) and volume fraction α₁:

    Mixture sound speed c_mix  — Eq.(3)
    S_α^(3), S_T^(3)          — Eq.(4), divergence-coupling coefficients
    S_α^(2), S_T^(2), S_p^(2) — Eq.(5), dissipation-source coefficients
"""

import numpy as np


def _safe(x, floor=1e-300):
    return np.where(np.abs(x) < floor, np.sign(x + 1e-310) * floor, x)


def mixture_sound_speed_sq(alpha1, p1, p2, T):
    """
    Mixture sound speed squared — Demou2022 Eq.(3).

    c_mix⁻² = ρ [α₁/(ρ₁c₁²) + α₂/(ρ₂c₂²)]
              + ρ T Cp1 Cp2/(Cp1+Cp2) * (Γ₂/(ρ₂c₂²) - Γ₁/(ρ₁c₁²))²

    Parameters
    ----------
    alpha1 : array  — volume fraction of phase 1
    p1, p2 : dicts  — per-phase props from eos.phase_props
    T      : array  — temperature

    Returns
    -------
    c2_mix : array
    """
    alpha2 = 1.0 - alpha1

    rho1, c21, G1 = p1['rho'], p1['c2'], p1['Gamma']
    rho2, c22, G2 = p2['rho'], p2['c2'], p2['Gamma']

    rho1c21 = _safe(rho1 * c21)
    rho2c22 = _safe(rho2 * c22)

    rho = alpha1 * rho1 + alpha2 * rho2

    # Extensive Cp per phase: C_{pk} = α_k ρ_k κ_{p,k}
    Cp1 = alpha1 * rho1 * p1['kp']
    Cp2 = alpha2 * rho2 * p2['kp']
    Cp_sum = _safe(Cp1 + Cp2)

    # Acoustic part
    inv_c2 = rho * (alpha1 / rho1c21 + alpha2 / rho2c22)

    # Thermal correction
    diff = G2 / rho2c22 - G1 / rho1c21
    inv_c2 = inv_c2 + rho * T * Cp1 * Cp2 / Cp_sum * diff**2

    return 1.0 / _safe(inv_c2)


def source_coefficients(alpha1, p1, p2, T, c2_mix):
    """
    Compute S^(2) and S^(3) source-term coefficients.

    Ref: Demou2022 Eqs.(4)-(6).

    Returns
    -------
    Sa3 : S_α^(3)    — Eq.(4a), divergence term for α₁
    ST3 : S_T^(3)    — Eq.(4b), divergence term for T
    Sa2 : S_α^(2)    — Eq.(5), dissipation/conduction source for α₁
    ST2 : S_T^(2)    — Eq.(5), dissipation/conduction source for T
    Sp2 : S_p^(2)    — Eq.(5), dissipation/conduction source for p
    """
    alpha2 = 1.0 - alpha1

    rho1, c21, G1, kp1, phi1, zeta1 = (
        p1['rho'], p1['c2'], p1['Gamma'], p1['kp'], p1['phi'], p1['zeta'])
    rho2, c22, G2, kp2, phi2, zeta2 = (
        p2['rho'], p2['c2'], p2['Gamma'], p2['kp'], p2['phi'], p2['zeta'])

    rho1c21 = _safe(rho1 * c21)
    rho2c22 = _safe(rho2 * c22)
    rho = alpha1 * rho1 + alpha2 * rho2

    Cp1 = alpha1 * rho1 * kp1
    Cp2 = alpha2 * rho2 * kp2
    Cp_sum = _safe(Cp1 + Cp2)

    diff_G = G2 / rho2c22 - G1 / rho1c21

    # ── S_α^(3)  Eq.(4a) ──────────────────────────────────────────
    # First term: ρc² α₁α₂ (1/ρ₂c₂² - 1/ρ₁c₁²)
    Sa3 = (rho * c2_mix * alpha1 * alpha2
           * (1.0 / rho2c22 - 1.0 / rho1c21))

    # Second term: T Cp1 Cp2/(Cp1+Cp2) * diff_G * (α₁Γ₂/ρ₂c₂² + α₂Γ₁/ρ₁c₁²)
    bracket = alpha1 * G2 / rho2c22 + alpha2 * G1 / rho1c21
    Sa3 = Sa3 + T * Cp1 * Cp2 / Cp_sum * diff_G * bracket

    # ── S_T^(3)  Eq.(4b) ──────────────────────────────────────────
    # ρc²T/(Cp1+Cp2) * (Cp1 Γ₁/ρ₁c₁² + Cp2 Γ₂/ρ₂c₂²)
    ST3 = (rho * c2_mix * T / Cp_sum
           * (Cp1 * G1 / rho1c21 + Cp2 * G2 / rho2c22))

    # ── Auxiliary scalars Eqs.(6a-d) ──────────────────────────────
    phi_rho  = alpha1 * phi1 * rho2 + alpha2 * phi2 * rho1        # (φρ)_T
    zeta_rho = alpha1 * zeta1 * rho2 + alpha2 * zeta2 * rho1      # (ζρ)_T
    phi_zeta = alpha1 * alpha2 * (phi1 * zeta2 - phi2 * zeta1)    # (φζ)_T

    D_bar = ((rho1 * c21 / _safe(G1) - rho2 * c22 / _safe(G2)) * phi_zeta
             + (alpha1 / _safe(G1) + alpha2 / _safe(G2)) * phi_rho)
    D_bar = np.where(np.abs(D_bar) < 1e-12, np.sign(D_bar + 1e-15) * 1e-12, D_bar)

    # ── S^(2) coefficients  Eq.(5) ────────────────────────────────
    Sa2 =  phi_zeta / D_bar
    ST2 = -zeta_rho / D_bar
    Sp2 =  phi_rho  / D_bar

    return Sa3, ST3, Sa2, ST2, Sp2
