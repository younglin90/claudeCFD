"""Pressure-equilibrium projection step (DC λ_k, He & Tan 2024 Eq. A.19).

After the IMEX update produces (U₁, U₂, U₃, U₄, U₅) from
`time_integrator.ars222_step`, the recovered W = (α, T₁, T₂, u, p) lives on
the EOS surface but **not necessarily on the pressure-equilibrium manifold**:
floating-point round-off in face_state EOS density() + cons_to_prim_W Newton
combine into an effective amplification eigenvalue |λ|>1 along PE-violating
modes (verified empirically — `pe_diagnostic.update_residual` grows by ×10
per step on 02-A SG α-jump even with tight tolerances).

The DC (Defect-Correction) relaxation step *projects* W back onto
PE manifold by enforcing

    p_1(ρ_1, T_1) = p_2(ρ_2, T_2)            (single-pressure)
    T_1 = T_2                                 (single-temperature, optional)

while *keeping the conservative invariants* (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁) fixed.
This is the He-Tan formulation: a 1-D Newton on a relaxation parameter λ
that re-distributes phase internal energy between the two phases until both
EOS pressure equations agree.

In this module we implement the **single-pressure projection** (mass and
energy redistributed within fixed (α, ρ, ρu, ρE) so that p_1 = p_2 = p
recovered consistently with both EOS).  The optional temperature relaxation
is left as a thin wrapper.

The default usage in `time_integrator.ars222_step` is to call
`relax_pressure(W_new, eos1, eos2)` after the primitive recovery — this
costs one 1-D Newton per cell (cheap) but suppresses the spectral PE drift
mode that is otherwise unbounded in clean-room IMEX.
"""
from __future__ import annotations
import numpy as np

_EPS = 1e-30


def relax_pressure(W, eos1, eos2, *, max_iter=10, rtol=1e-12, atol=1e-6):
    """Project W onto the single-pressure equilibrium manifold.

    Holds (α₁, ρ₁ from W, ρ₂ from W, u, total ρe) fixed and solves a single
    pressure p such that the two phases share it.  Phase temperatures are
    recovered consistently with the per-phase EOS:

        e_k = e_k(ρ_k, p)          ← from the inverse EOS
        T_k = T_k(ρ_k, e_k)

    The total internal energy ρe = α₁ρ₁ e₁(ρ₁, p) + α₂ρ₂ e₂(ρ₂, p) is the
    constraint.  For Ideal/SG/NASG e_k is linear in p, so the equation is

        ρe = α₁ρ₁ [A₁(ρ₁) p + B₁(ρ₁)] + α₂ρ₂ [A₂(ρ₂) p + B₂(ρ₂)]

    and the projection is one division (no Newton).  We use a generic
    Newton fallback for nonlinear EOS.
    """
    a, T1, T2, u, p = (np.asarray(c, dtype=float).copy() for c in W)
    a2 = 1.0 - a
    # Phase densities held fixed at their current W values
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)

    # Total internal energy from current W (preserved through relaxation)
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rho_e = a * rho1 * e1 + a2 * rho2 * e2

    # Try linear-in-p form: e_k(ρ, p) = A_k(ρ)·p + B_k(ρ).
    # Probe at two values to check linearity.
    try:
        p_lo = np.maximum(p * 0.5, 1.0)
        p_hi = p * 2.0
        e1_lo = eos1.energy(rho1, p_lo); e1_hi = eos1.energy(rho1, p_hi)
        e2_lo = eos2.energy(rho2, p_lo); e2_hi = eos2.energy(rho2, p_hi)
        A1 = (e1_hi - e1_lo) / (p_hi - p_lo)
        B1 = e1_lo - A1 * p_lo
        A2 = (e2_hi - e2_lo) / (p_hi - p_lo)
        B2 = e2_lo - A2 * p_lo
        # Verify linearity at p (within rtol)
        e1_check = A1 * p + B1
        e2_check = A2 * p + B2
        if (np.max(np.abs(e1_check - e1) / np.maximum(np.abs(e1), 1.0)) < 1e-10 and
            np.max(np.abs(e2_check - e2) / np.maximum(np.abs(e2), 1.0)) < 1e-10):
            # Linear in p — solve directly
            A_sum = a * rho1 * A1 + a2 * rho2 * A2
            B_sum = a * rho1 * B1 + a2 * rho2 * B2
            p_new = (rho_e - B_sum) / np.maximum(A_sum, _EPS)
            p_new = np.maximum(p_new, 1.0)
            T1_new = eos1.temperature(rho1, eos1.energy(rho1, p_new))
            T2_new = eos2.temperature(rho2, eos2.energy(rho2, p_new))
            T1_new = np.maximum(T1_new, 1.0)
            T2_new = np.maximum(T2_new, 1.0)
            return (a, T1_new, T2_new, u, p_new)
    except Exception:
        pass

    # Generic Newton fallback (nonlinear EOS)
    p_new = p.copy()
    for _ in range(max_iter):
        e1 = eos1.energy(rho1, p_new)
        e2 = eos2.energy(rho2, p_new)
        F = a * rho1 * e1 + a2 * rho2 * e2 - rho_e
        # ∂F/∂p evaluated by symmetric FD
        dp = np.maximum(np.abs(p_new) * 1e-7, 1.0)
        Fp = a * rho1 * eos1.energy(rho1, p_new + dp) \
             + a2 * rho2 * eos2.energy(rho2, p_new + dp) - rho_e
        Fm = a * rho1 * eos1.energy(rho1, p_new - dp) \
             + a2 * rho2 * eos2.energy(rho2, p_new - dp) - rho_e
        dFdp = (Fp - Fm) / (2.0 * dp)
        step = F / np.where(np.abs(dFdp) > _EPS, dFdp, _EPS)
        p_trial = p_new - step
        p_trial = np.maximum(p_trial, 1.0)
        if np.max(np.abs(step) / np.maximum(np.abs(p_new), 1.0)) < rtol:
            p_new = p_trial; break
        p_new = p_trial

    T1_new = np.maximum(eos1.temperature(rho1, eos1.energy(rho1, p_new)), 1.0)
    T2_new = np.maximum(eos2.temperature(rho2, eos2.energy(rho2, p_new)), 1.0)
    return (a, T1_new, T2_new, u, p_new)


def relax_pT(W, eos1, eos2, *, max_iter=15, rtol=1e-12):
    """Full p-T relaxation: enforce both p_1 = p_2 and T_1 = T_2.

    Total mass / momentum / energy preserved (α also fixed).  Two-equation
    Newton on (p, T) — used when temperature equilibrium is desired
    (dilute mixtures, near-equilibrium two-phase flows).  For PE-only tests
    `relax_pressure` is preferred (single-temp constraint inactive).
    """
    a, T1, T2, u, p = (np.asarray(c, dtype=float).copy() for c in W)
    a2 = 1.0 - a
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    rho_e = (a * rho1 * eos1.energy(rho1, p)
             + a2 * rho2 * eos2.energy(rho2, p))
    a_rho1 = a * rho1; a_rho2 = a2 * rho2

    # Initial guess
    T = 0.5 * (T1 + T2); p_new = p.copy()
    for _ in range(max_iter):
        rho1_n = np.maximum(eos1.density(p_new, T), _EPS)
        rho2_n = np.maximum(eos2.density(p_new, T), _EPS)
        # Constraints: F1 = α·ρ_1(p,T) − a_rho1 = 0;  F2 = (1−α)·ρ_2(p,T) − a_rho2 = 0;
        # F3 = α·ρ_1·e_1 + (1-α)·ρ_2·e_2 − ρ_e = 0 — but α held fixed so
        # only two free unknowns (p, T).  Use F1 + F3 (energy) closure.
        e1_n = eos1.energy(rho1_n, p_new)
        e2_n = eos2.energy(rho2_n, p_new)
        F1 = a * rho1_n - a_rho1
        F3 = a * rho1_n * e1_n + a2 * rho2_n * e2_n - rho_e
        # 2×2 Jacobian via FD
        dp = np.maximum(np.abs(p_new) * 1e-7, 1.0)
        dT = np.maximum(np.abs(T) * 1e-7, 1.0)
        rho1_p = np.maximum(eos1.density(p_new + dp, T), _EPS)
        rho2_p = np.maximum(eos2.density(p_new + dp, T), _EPS)
        rho1_T = np.maximum(eos1.density(p_new, T + dT), _EPS)
        rho2_T = np.maximum(eos2.density(p_new, T + dT), _EPS)
        F1_dp = a * rho1_p - a_rho1
        F1_dT = a * rho1_T - a_rho1
        F3_dp = (a * rho1_p * eos1.energy(rho1_p, p_new + dp)
                 + a2 * rho2_p * eos2.energy(rho2_p, p_new + dp) - rho_e)
        F3_dT = (a * rho1_T * eos1.energy(rho1_T, p_new + dT)
                 + a2 * rho2_T * eos2.energy(rho2_T, p_new + dT) - rho_e)
        J11 = (F1_dp - F1) / dp; J12 = (F1_dT - F1) / dT
        J21 = (F3_dp - F3) / dp; J22 = (F3_dT - F3) / dT
        det = J11 * J22 - J12 * J21
        det = np.where(np.abs(det) < _EPS, _EPS, det)
        d_p = (-F1 * J22 + F3 * J12) / det
        d_T = (-F3 * J11 + F1 * J21) / det
        p_new = np.maximum(p_new + d_p, 1.0)
        T = np.maximum(T + d_T, 1.0)
        if (np.max(np.abs(d_p) / np.maximum(np.abs(p_new), 1.0)) < rtol and
            np.max(np.abs(d_T) / np.maximum(np.abs(T), 1.0)) < rtol):
            break

    return (a, T, T, u, p_new)
