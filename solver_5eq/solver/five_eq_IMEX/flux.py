"""Advective (explicit) face fluxes — Phase 6+7 active.

A single, consistent face velocity drives F_q1, F_q2, F_α, F_ρ; momentum
advection uses ρ_face·u_f² (no pressure — F_I handles ∇p).  Energy uses
APEC χ_k, χ_a (`energy_flux.total_energy_flux`) when `energy_form='apec'`.

Phase-mass fluxes use:
    F_q1 = α_face · ρ1_face · u_face
    F_q2 = (1−α_face) · ρ2_face · u_face

with α_face from the upwind selector built inside `face_state.face_state` and
ρ_k_face from EOS(p_face, T_k_face) (ACID-style).  This makes the new IMEX
solver robust against large-density-ratio interfaces — the small-α cell only
sees its own phase mass advected (≈ 0) instead of receiving bulk mass from
the high-density neighbour.

The `u_face` returned by `face_state` is the *central* average of cell-centre
velocity (Phase 3 default).  Phase 5 (SLAU2) will replace it with the all-Mach
χ(M)-corrected face velocity.
"""
from __future__ import annotations
import numpy as np

from .energy_flux import total_energy_flux


def advective_fluxes(face, eos1, eos2, *, energy_form='apec',
                     energy_alpha_pure_tol=1.0e-12):
    """Compute advective face fluxes for the explicit operator F_E.

    Parameters
    ----------
    face : dict
        Face state from `face_state.face_state(...)`.  Must contain keys
        `alpha, u, rho1, rho2, e1, e2, rho`, and (for APEC) `T1, T2`.
    energy_form : 'apec' (default) | 'allaire'
    energy_alpha_pure_tol : float
        Face alpha tolerance used by APEC's pure-phase energy branch.

    Returns
    -------
    dict of (N+1,) arrays:
        F_a1r1, F_a2r2, F_alpha, F_rho, F_ru, F_rE
    """
    u_f   = face['u']
    a_f   = face['alpha']
    rho1f = face['rho1']
    rho2f = face['rho2']
    rho_f = face['rho']

    F_a1r1 = a_f * rho1f * u_f
    F_a2r2 = (1.0 - a_f) * rho2f * u_f
    F_alpha = a_f * u_f
    F_rho = F_a1r1 + F_a2r2
    F_ru  = rho_f * u_f * u_f                 # ρ u² (no p — implicit handles ∇p)

    F_rE = total_energy_flux(face, eos1, eos2,
                             F_a1r1, F_a2r2, F_alpha, F_rho,
                             energy_form=energy_form,
                             alpha_pure_tol=energy_alpha_pure_tol)
    return dict(F_a1r1=F_a1r1, F_a2r2=F_a2r2, F_alpha=F_alpha,
                F_rho=F_rho, F_ru=F_ru, F_rE=F_rE)


def divergence(face_flux, dx):
    """Discrete divergence ∂F/∂x ≈ (F_{i+1/2} − F_{i−1/2})/Δx."""
    return {k: (F[1:] - F[:-1]) / dx for k, F in face_flux.items()}
