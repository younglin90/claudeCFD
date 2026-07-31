"""APEC-type internal-energy face flux — *secant-consistent* form (v3).

ChatGPT v2 진단 §C 후속:

  F_rho_e^APEC = χ̄_1·F_q1 + χ̄_2·F_q2 + χ̄_α·F_α

where (χ̄_1, χ̄_2, χ̄_α) is a **secant** coefficient triple constructed from
the L/R *face* states such that the discrete identity

    g(q1_R, q2_R, α_R; p_f) − g(q1_L, q2_L, α_L; p_f)
        = χ̄_1·(q1_R − q1_L) + χ̄_2·(q2_R − q2_L) + χ̄_α·(α_R − α_L)

is satisfied **byte-exactly** for the function

    g(q1, q2, α; p) = q1·e_1(q1/α, p) + q2·e_2(q2/(1−α), p)

(per-cell ρe contribution at fixed p).  This eliminates the (face-flux ↔
mid-state χ) inconsistency that left R_E ≠ 0 at step 0 in v2 and seeded
the spectral PE drift.

Construction:
  - Take the L/R face states (a_L, ρ_k_L, ...) from `face_state` (which
    already lives on the EOS surface at p_f, T_k_L/R).
  - Numerically split the increment of g into three additive contributions
    by accumulating along the segment L → R in three sub-steps:
      step (a): only q1 changes (α, q2 fixed at L)
      step (b): only q2 changes (α at L, q1 at R)
      step (c): only α changes (q1, q2 at R)
    Each sub-step is one variable, so its contribution divided by the
    increment gives the corresponding χ̄ secant coefficient.
  - Fallback (when increment is below floor):  collapse to the *face*
    differential coefficients χ_k = e_k_f + ρ_k_f·e_T/ρ_T (the v2 form).

Pressure work p·u remains in F_I (implicit) — F_K = ½ u_f² · F_rho only.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def _secant_chi(face, eos1, eos2, *, increment_floor=1e-12):
    """Return (χ̄_1, χ̄_2, χ̄_α) such that

        Δg(q1, q2, α; p_f) = χ̄_1 Δq1 + χ̄_2 Δq2 + χ̄_α Δα   (byte-exact)

    where g = q1·e_1(q1/α, p_f) + q2·e_2(q2/(1−α), p_f) and the L→R increment
    is the path used by `face_state`.

    Each χ̄ is computed by isolating one variable along the L→R segment:
      step (a): change q1 from q1_L to q1_R (α=α_L, q2=q2_L, p=p_f)
      step (b): change q2 from q2_L to q2_R (α=α_L, q1=q1_R)
      step (c): change α from α_L to α_R (q1=q1_R, q2=q2_R, p=p_f)
    The integrated path-sum equals g_R − g_L exactly by construction.

    When an increment Δ is below `increment_floor` (face has no jump in that
    variable), the corresponding χ̄ is set from the face mid-state
    differential — this branch never affects the flux because Δ = 0 makes
    the contribution vanish either way.
    """
    p_f   = face['p']
    a_L   = face['a_L'];   a_R   = face['a_R']
    rho1_L = face['rho1_L']; rho2_L = face['rho2_L']
    rho1_R = face['rho1_R']; rho2_R = face['rho2_R']

    q1_L = a_L * rho1_L; q1_R = a_R * rho1_R
    q2_L = (1.0 - a_L) * rho2_L; q2_R = (1.0 - a_R) * rho2_R
    da = a_R - a_L
    dq1 = q1_R - q1_L
    dq2 = q2_R - q2_L

    # Step (a): vary q1 only (α at L, q2 at L, p_f)
    # ρ_1 = q1 / α_L → e_1(ρ_1, p_f).  Endpoints: q1=q1_L (= a_L ρ1_L) and q1=q1_R.
    # We also need ρ_1 at the q1_R / α_L combination:
    rho1_aL_qR = q1_R / np.maximum(a_L, 1e-12)
    e1_at_aL_qL = eos1.energy(rho1_L, p_f)                    # = e1_L
    e1_at_aL_qR = eos1.energy(rho1_aL_qR, p_f)
    g_a_L = q1_L * e1_at_aL_qL + q2_L * eos2.energy(rho2_L, p_f)
    g_a_R = q1_R * e1_at_aL_qR + q2_L * eos2.energy(rho2_L, p_f)
    # χ̄_1 = (g_a_R − g_a_L) / dq1
    safe_dq1 = np.where(np.abs(dq1) > increment_floor, dq1, 1.0)
    chi1_bar = (g_a_R - g_a_L) / safe_dq1
    # Fallback when no increment: face differential
    rho1_T_f = eos1.drhodT_p(face['rho1'], face['T1'])
    e1_T_f   = eos1.dedT_p(face['rho1'], face['T1'])
    chi1_diff = face['e1'] + face['rho1'] * e1_T_f / np.where(
        np.abs(rho1_T_f) > 1e-30, rho1_T_f, 1e-30)
    chi1_bar = np.where(np.abs(dq1) > increment_floor, chi1_bar, chi1_diff)

    # Step (b): vary q2 only (α=a_L, q1=q1_R, p_f)
    # ρ_2 = q2 / (1−α_L)
    inv_b = 1.0 / np.maximum(1.0 - a_L, 1e-12)
    rho2_at_aL_qL = q2_L * inv_b
    rho2_at_aL_qR = q2_R * inv_b
    e2_at_aL_qL = eos2.energy(rho2_at_aL_qL, p_f)
    e2_at_aL_qR = eos2.energy(rho2_at_aL_qR, p_f)
    g_b_L = q1_R * e1_at_aL_qR + q2_L * e2_at_aL_qL
    g_b_R = q1_R * e1_at_aL_qR + q2_R * e2_at_aL_qR
    safe_dq2 = np.where(np.abs(dq2) > increment_floor, dq2, 1.0)
    chi2_bar = (g_b_R - g_b_L) / safe_dq2
    rho2_T_f = eos2.drhodT_p(face['rho2'], face['T2'])
    e2_T_f   = eos2.dedT_p(face['rho2'], face['T2'])
    chi2_diff = face['e2'] + face['rho2'] * e2_T_f / np.where(
        np.abs(rho2_T_f) > 1e-30, rho2_T_f, 1e-30)
    chi2_bar = np.where(np.abs(dq2) > increment_floor, chi2_bar, chi2_diff)

    # Step (c): vary α only (q1=q1_R, q2=q2_R, p_f)
    # ρ_1 = q1_R / α; ρ_2 = q2_R / (1−α).  Endpoints α=a_L and α=a_R.
    rho1_aL_qR = q1_R / np.maximum(a_L, 1e-12)
    rho2_aL_qR = q2_R / np.maximum(1.0 - a_L, 1e-12)
    rho1_aR_qR = q1_R / np.maximum(a_R, 1e-12)
    rho2_aR_qR = q2_R / np.maximum(1.0 - a_R, 1e-12)
    g_c_L = (q1_R * eos1.energy(rho1_aL_qR, p_f)
             + q2_R * eos2.energy(rho2_aL_qR, p_f))
    g_c_R = (q1_R * eos1.energy(rho1_aR_qR, p_f)
             + q2_R * eos2.energy(rho2_aR_qR, p_f))
    safe_da = np.where(np.abs(da) > increment_floor, da, 1.0)
    chia_bar = (g_c_R - g_c_L) / safe_da
    chia_diff = (- face['rho1'] ** 2 * e1_T_f / np.where(
                    np.abs(rho1_T_f) > 1e-30, rho1_T_f, 1e-30)
                 + face['rho2'] ** 2 * e2_T_f / np.where(
                    np.abs(rho2_T_f) > 1e-30, rho2_T_f, 1e-30))
    chia_bar = np.where(np.abs(da) > increment_floor, chia_bar, chia_diff)

    return chi1_bar, chi2_bar, chia_bar


def apec_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho, *,
                     fallback_eps=1e-3, mode='differential'):
    """Return F_rho_e^APEC (length-(N+1) array).

    Parameters
    ----------
    mode : 'secant' (default, v3) | 'differential' (v2 face mid-state form)
        'secant' uses path-consistent χ̄_k, χ̄_α from L/R face states so the
        discrete update identity is byte-exact.  'differential' falls back
        to the face-differential coefficients (e_k_f + ρ_k_f · e_T/ρ_T).
    """
    if mode == 'secant':
        chi1, chi2, chia = _secant_chi(face, eos1, eos2)
    elif mode == 'differential':
        rho1_f = face['rho1']; rho2_f = face['rho2']
        e1_f   = face['e1'];   e2_f   = face['e2']
        T1_f   = face['T1'];   T2_f   = face['T2']
        rho1_T = eos1.drhodT_p(rho1_f, T1_f)
        rho2_T = eos2.drhodT_p(rho2_f, T2_f)
        e1_T   = eos1.dedT_p(rho1_f, T1_f)
        e2_T   = eos2.dedT_p(rho2_f, T2_f)
        floor1 = max(fallback_eps * float(np.max(np.abs(rho1_T))), 1e-30)
        floor2 = max(fallback_eps * float(np.max(np.abs(rho2_T))), 1e-30)
        rho1_T_safe = np.where(np.abs(rho1_T) > floor1, rho1_T,
                                np.sign(rho1_T + 1e-300) * floor1)
        rho2_T_safe = np.where(np.abs(rho2_T) > floor2, rho2_T,
                                np.sign(rho2_T + 1e-300) * floor2)
        chi1 = e1_f + rho1_f * e1_T / rho1_T_safe
        chi2 = e2_f + rho2_f * e2_T / rho2_T_safe
        chia = (- rho1_f ** 2 * e1_T / rho1_T_safe
                + rho2_f ** 2 * e2_T / rho2_T_safe)
        bad1 = np.abs(rho1_T) <= floor1
        bad2 = np.abs(rho2_T) <= floor2
        chi1 = np.where(bad1, e1_f, chi1)
        chi2 = np.where(bad2, e2_f, chi2)
        chia = np.where(bad1 | bad2, 0.0, chia)
    else:
        raise ValueError(f"Unknown APEC mode='{mode}'.")

    return chi1 * F_q1 + chi2 * F_q2 + chia * F_alpha


def total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho, *,
                      energy_form='apec', alpha_pure_tol=1.0e-12):
    """Total explicit energy flux F_rE = F_rho_e + ½ u_f² · F_rho.

    `energy_form='apec' | 'differential'` (default 'apec') uses APEC χ_k, χ_a
    differential form (e_k_f + ρ_k_f·e_T/ρ_T).
    `energy_form='secant'` uses path-consistent secant χ̄.
    `energy_form='allaire'` uses the simpler e_up·F_q baseline.
    """
    u_f = face['u']
    e1_f = face['e1']
    e2_f = face['e2']
    if energy_form in ('apec', 'differential'):
        F_rho_e = apec_energy_flux(face, eos1, eos2,
                                   F_q1, F_q2, F_alpha, F_rho,
                                   mode='differential')
    elif energy_form == 'secant':
        F_rho_e = apec_energy_flux(face, eos1, eos2,
                                   F_q1, F_q2, F_alpha, F_rho,
                                   mode='secant')
    elif energy_form == 'allaire':
        F_rho_e = e1_f * F_q1 + e2_f * F_q2
    else:
        raise ValueError(f"Unknown energy_form='{energy_form}'.")

    # APEC is an interface correction. On pure faces it must collapse to the
    # active single-phase internal-energy flux and must not use chi_alpha.
    if alpha_pure_tol > 0.0:
        a_f = face['alpha']
        pure1 = a_f >= 1.0 - alpha_pure_tol
        pure2 = a_f <= alpha_pure_tol
        F_rho_e = np.where(pure1, e1_f * F_q1, F_rho_e)
        F_rho_e = np.where(pure2, e2_f * F_q2, F_rho_e)
    F_K = 0.5 * u_f ** 2 * F_rho
    return F_rho_e + F_K
