"""α-source closure D_1 for the volume-fraction equation:

    ∂α₁/∂t + ∂(α₁ u)/∂x = (α₁ + D₁) ∂u/∂x

Allaire-Massoni  : D₁ = 0
Kapila / Murrone-Guillard:
    D₁ = α₁ α₂ (ρ₂ c₂² − ρ₁ c₁²) / (α₂ ρ₁ c₁² + α₁ ρ₂ c₂²)

The closure can be evaluated either at the cell anchor state W^* or at an
ACID/EOS-consistent face state.  The latter is used by the explicit Kapila
source path to keep D₁ on the same thermodynamic path as the alpha flux.
"""
from __future__ import annotations
import numpy as np

from .sound_speed import phase_sound_speed_sq

_EPS = 1e-30


def D_K_kapila(W, eos1, eos2, face=None):
    """Cell-centered D_K from W = (α, T1, T2, u, p) — frozen at anchor."""
    a1, T1, T2, _, p = W
    a2 = 1.0 - a1
    rho1 = eos1.density(p, T1)
    rho2 = eos2.density(p, T2)
    c1_sq = phase_sound_speed_sq(eos1, rho1, T1)
    c2_sq = phase_sound_speed_sq(eos2, rho2, T2)
    rho1c2 = np.maximum(rho1 * c1_sq, _EPS)
    rho2c2 = np.maximum(rho2 * c2_sq, _EPS)
    num = a1 * a2 * (rho2c2 - rho1c2)
    den = np.maximum(a2 * rho1c2 + a1 * rho2c2, _EPS)
    D_K = num / den
    # Smear out outside two-phase region — pure phase has zero D_K.
    return np.where(a1 * a2 > 1e-12, D_K, 0.0)


def D_K_kapila_face(face):
    """Face-centered Kapila D_K from an ACID/EOS-consistent face state.

    The volume-fraction source is a non-conservative product.  When Kapila
    closure is enabled, using the same face thermodynamic path as the alpha
    flux reduces a cell/face path mismatch in `(alpha + D_K) div(u)`.
    """
    a1 = face['alpha']
    a2 = 1.0 - a1
    rho1c2 = np.maximum(face['rho1'] * face['c1_sq'], _EPS)
    rho2c2 = np.maximum(face['rho2'] * face['c2_sq'], _EPS)
    num = a1 * a2 * (rho2c2 - rho1c2)
    den = np.maximum(a2 * rho1c2 + a1 * rho2c2, _EPS)
    D_K = num / den
    return np.where(a1 * a2 > 1e-12, D_K, 0.0)
