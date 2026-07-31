"""Pressure-equilibrium preservation diagnostic (ChatGPT §3 우선순위 3).

Three residuals must be machine-ε on a uniform-(p, u, T_k) state with α-jump:

    R_q1 = F_q1 − ρ_{1,f} · F_α
    R_q2 = F_q2 − ρ_{2,f} · (u_f − F_α)
    R_E  = Δ(ρe) − χ_1·Δq_1 − χ_2·Δq_2 − χ_α·Δα

The first two are *face-level* identities — when they fail the mass / α
flux assemblies are using inconsistent face velocities or densities.

The third is the *cell update* differential identity — when it fails the
APEC χ-coefficients are not consistent with the discrete update or the
primitive recovery itself drifts.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def face_consistency(face, F_q1, F_q2, F_alpha):
    """R_q1, R_q2 face-level residuals."""
    rho1_f = face['rho1']
    rho2_f = face['rho2']
    u_f = face['u']
    R_q1 = F_q1 - rho1_f * F_alpha
    R_q2 = F_q2 - rho2_f * (u_f * face['alpha'] / face['alpha'] - F_alpha) \
            if False else F_q2 - rho2_f * ((1.0 - face['alpha']) * u_f)
    # simpler: F_q2 = (1-α_f) ρ_2_f u_f, F_α = α_f u_f, so consistency is
    #   F_q2 = ρ_2_f · (u_f − F_α)
    R_q2 = F_q2 - rho2_f * (u_f - F_alpha)
    return R_q1, R_q2


def update_residual(W_n, W_new, eos1, eos2):
    """R_E = Δ(ρe) − χ_1·Δq_1 − χ_2·Δq_2 − χ_α·Δα at *cell* centres.

    χ_k, χ_α evaluated at the average state (W_n + W_new)/2 (mid-state).
    """
    a_n, T1_n, T2_n, u_n, p_n = W_n
    a_e, T1_e, T2_e, u_e, p_e = W_new

    rho1_n = eos1.density(p_n, T1_n); rho1_e = eos1.density(p_e, T1_e)
    rho2_n = eos2.density(p_n, T2_n); rho2_e = eos2.density(p_e, T2_e)
    e1_n = eos1.energy(rho1_n, p_n);  e1_e = eos1.energy(rho1_e, p_e)
    e2_n = eos2.energy(rho2_n, p_n);  e2_e = eos2.energy(rho2_e, p_e)

    rhoe_n = a_n * rho1_n * e1_n + (1.0 - a_n) * rho2_n * e2_n
    rhoe_e = a_e * rho1_e * e1_e + (1.0 - a_e) * rho2_e * e2_e
    d_rhoe = rhoe_e - rhoe_n
    d_q1 = a_e * rho1_e - a_n * rho1_n
    d_q2 = (1.0 - a_e) * rho2_e - (1.0 - a_n) * rho2_n
    d_a = a_e - a_n

    # Mid-state χ coefficients
    a_m = 0.5 * (a_n + a_e)
    T1_m = 0.5 * (T1_n + T1_e); T2_m = 0.5 * (T2_n + T2_e)
    p_m = 0.5 * (p_n + p_e)
    rho1_m = eos1.density(p_m, T1_m); rho2_m = eos2.density(p_m, T2_m)
    e1_m = eos1.energy(rho1_m, p_m); e2_m = eos2.energy(rho2_m, p_m)
    rho1_T = eos1.drhodT_p(rho1_m, T1_m); rho2_T = eos2.drhodT_p(rho2_m, T2_m)
    e1_T = eos1.dedT_p(rho1_m, T1_m); e2_T = eos2.dedT_p(rho2_m, T2_m)
    chi1 = e1_m + rho1_m * e1_T / np.where(np.abs(rho1_T) > _EPS, rho1_T, _EPS)
    chi2 = e2_m + rho2_m * e2_T / np.where(np.abs(rho2_T) > _EPS, rho2_T, _EPS)
    chia = (- rho1_m ** 2 * e1_T / np.where(np.abs(rho1_T) > _EPS, rho1_T, _EPS)
            + rho2_m ** 2 * e2_T / np.where(np.abs(rho2_T) > _EPS, rho2_T, _EPS))
    R_E = d_rhoe - chi1 * d_q1 - chi2 * d_q2 - chia * d_a
    return R_E
