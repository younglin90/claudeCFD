"""K-phase Kapila 5-eq solver (explicit, SSP-RK3 + HLLC).

Minimal K≥3 extension of the 2-phase IMEX solver, focused on:
  - Multi-material shock tubes (Phase 5-1: 3-gas, Phase 5-2: UNDEX)
  - Explicit time integration (no acoustic implicit for now)
  - HLLC flux with generalized mixture EOS

Governing equations (K phases, single velocity + single pressure):
  ∂(α_k ρ_k)/∂t + ∂(α_k ρ_k u)/∂x = 0,  k=1..K
  ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
  ∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
  ∂α_k/∂t + u ∂α_k/∂x = 0,              k=1..K-1  (α_K = 1 - Σ)

State vector (2K+2): (α_1ρ_1, ..., α_Kρ_K, ρu, ρE, α_1, ..., α_{K-1})

Mixture closure: `mixture_pressure_solve_K` (linear for Ideal/SG/NASG/MG/JWL).
"""
import numpy as np
from .eos_general import (to_eos, mixture_pressure_solve_K,
                           mixture_sound_speed_K, _linear_mixture_pressure_K)

_EPS = 1e-12


def _tvd_slope(u):
    """van Leer TVD limiter on array u. Returns slopes of length N (interior)."""
    N = len(u)
    du = u[1:] - u[:-1]  # length N-1
    # Interior slopes via minmod-van Leer
    r_l = du[:-1]  # N-2
    r_r = du[1:]   # N-2
    eps = 1e-30
    phi = np.where(r_l * r_r > 0,
                   2.0 * r_l * r_r / (r_l + r_r + eps),
                   0.0)
    slopes = np.zeros(N)
    slopes[1:-1] = phi  # interior
    return slopes


def _ghost(arr, bc_l, bc_r):
    if bc_l == 'periodic':
        return np.concatenate([arr[-1:], arr, arr[:1]])
    return np.concatenate([arr[:1], arr, arr[-1:]])


def cons_to_prim_K(ar_list, ru, rE, a_list, eos_list):
    """K-phase conservative → primitive."""
    K = len(eos_list)
    rho = sum(ar_list[k] for k in range(K))
    u = ru / np.maximum(rho, _EPS)
    rho_e = rE - 0.5 * ru * u
    _af = 1e-8
    rhos = [np.maximum(ar_list[k] / np.maximum(a_list[k], _af), _EPS) for k in range(K)]
    p = mixture_pressure_solve_K(a_list, rhos, rho_e, eos_list)
    p = np.maximum(p, 1.0)

    # Phase e, T, c²
    es = [eos_list[k].energy(rhos[k], p) for k in range(K)]
    Ts = [np.maximum(eos_list[k].temperature(rhos[k], es[k]), 1.0) for k in range(K)]
    # Admissibility guard
    for k in range(K):
        try:
            adm = eos_list[k].is_admissible(rhos[k], p, Ts[k])
            if not np.all(adm):
                rho_eos = eos_list[k].density(p, Ts[k])
                # Only recover where majority phase protects (a_k small)
                # Always for simplicity:
                rhos[k] = np.where(adm, rhos[k], np.maximum(rho_eos, _EPS))
                es[k] = eos_list[k].energy(rhos[k], p)
        except (AttributeError, NotImplementedError):
            pass
    c_sq_mix, c_sqs = mixture_sound_speed_K(a_list, rhos, es, p, eos_list)
    c_mix = np.sqrt(np.maximum(c_sq_mix, _EPS))
    # Majority-phase T
    T_out = np.zeros_like(p)
    a_max = np.zeros_like(p)
    for k in range(K):
        mask = a_list[k] > a_max
        T_out = np.where(mask, Ts[k], T_out)
        a_max = np.where(mask, a_list[k], a_max)
    return p, u, T_out, rhos, es, c_mix


def prim_to_cons_K(rhos, u, p, a_list, eos_list):
    """K-phase primitive → conservative."""
    K = len(eos_list)
    ar_list = [a_list[k] * rhos[k] for k in range(K)]
    rho = sum(ar_list)
    ru = rho * u
    es = [eos_list[k].energy(rhos[k], p) for k in range(K)]
    rho_e = sum(ar_list[k] * es[k] for k in range(K))
    rE = rho_e + 0.5 * rho * u ** 2
    return ar_list, ru, rE


def _apec_slau2_flux_K(stateL, stateR, eos_list):
    """APEC + SLAU2 flux for K-phase Kapila (PE-preserving, all-Mach).

    Key features (K=2 solve_IMEX parity):
    1. SLAU2 pressure-free u_face → PE preservation at low-Mach
         χ = (1 - M̂)²
         u_face = V_avg - (χ / (ρ_avg · c_avg)) · (p_R - p_L)
    2. APEC energy flux: F_rE = Σ ε_k F_{ar_k} + ½ u_face² F_rho + p̄·ū
       where ε_k = e_k(ρ_k, p_up) is upwind phase internal energy
       → p-equilibrium exactly preserved (no ρe-Π cancellation)
    3. Momentum flux F_ru uses upwind + Rusanov dissipation (shock capture)

    Returns: F_ar, F_ru, F_rE, F_a, u_face
    """
    ar_L, ru_L, rE_L, a_L, pL, uL, rhoL, cL = stateL
    ar_R, ru_R, rE_R, a_R, pR, uR, rhoR, cR = stateR
    K = len(eos_list)

    # --- SLAU2 pressure-free face velocity (all-Mach, Deng 2025) ---
    c_avg = 0.5 * (cL + cR)
    u_rms = np.sqrt(0.5 * (uL ** 2 + uR ** 2))
    M_hat = np.minimum(1.0, u_rms / np.maximum(c_avg, _EPS))
    chi = (1.0 - M_hat) ** 2
    rho_avg = 0.5 * (rhoL + rhoR)
    # Roe-averaged material velocity
    V_avg = (rhoL * uL + rhoR * uR) / np.maximum(rhoL + rhoR, _EPS)
    u_face = V_avg - (chi / np.maximum(rho_avg * c_avg, _EPS)) * (pR - pL)

    # --- Upwind face values ---
    upw = (u_face >= 0.0).astype(float)
    ar_up = [upw * ar_L[k] + (1.0 - upw) * ar_R[k] for k in range(K)]
    a_up = [upw * a_L[k] + (1.0 - upw) * a_R[k] for k in range(K - 1)]
    ru_up = upw * ru_L + (1.0 - upw) * ru_R
    p_up = upw * pL + (1.0 - upw) * pR
    rho_up = sum(ar_up)

    # Phase internal energy at upwind state (for APEC)
    # ρ_k from a_up[k] and ar_up[k]
    if K - 1 < K:
        a_up_full = list(a_up) + [np.maximum(1.0 - sum(a_up), 1e-12)]
    else:
        a_up_full = a_up
    e_up = []
    for k in range(K):
        rho_k_up = ar_up[k] / np.maximum(a_up_full[k], 1e-8)
        rho_k_up = np.maximum(rho_k_up, _EPS)
        e_up.append(eos_list[k].energy(rho_k_up, np.maximum(p_up, 1.0)))

    # --- Central face pressure p̄ (symmetric, for pressure coupling) ---
    p_bar = 0.5 * (pL + pR)

    # --- Mass / alpha flux: upwind (no Rusanov — preserves contact) ---
    F_ar = [ar_up[k] * u_face for k in range(K)]
    F_a = [a_up[k] * u_face for k in range(K - 1)]

    # --- Momentum flux: upwind advective + central pressure
    #     (No Rusanov on ru — Rusanov at contacts breaks PE) ---
    F_ru = ru_up * u_face + p_bar

    # --- APEC energy flux: F_rE = Σ ε_k F_{ar_k} + ½ u_face² F_rho + p̄·ū ---
    F_rho = sum(F_ar)
    F_rE = (sum(e_up[k] * F_ar[k] for k in range(K))
            + 0.5 * u_face ** 2 * F_rho
            + p_bar * u_face)

    return F_ar, F_ru, F_rE, F_a, u_face


def _apec_slau2_flux_K_imex(stateL, stateR, eos_list):
    """APEC + SLAU2 advective-only flux for K-phase Kapila IMEX transport step.

    IMEX-compatible: pressure excluded for IM1 acoustic step.
    Pressure terms (p_bar in F_ru, p_bar*u_face in F_rE) are intentionally
    omitted here because the Peluchon IM1 acoustic step handles all pressure
    work implicitly. Including pressure in the advective flux would cause
    double-counting of acoustic effects (splitting error amplification).

    Differences from _apec_slau2_flux_K (full flux):
      - F_ru = ru_up * u_face          (NO + p_bar)
      - F_rE = APEC advective only     (NO + p_bar * u_face pressure work)

    Returns: F_ar, F_ru, F_rE, F_a, u_face
    """
    ar_L, ru_L, rE_L, a_L, pL, uL, rhoL, cL = stateL
    ar_R, ru_R, rE_R, a_R, pR, uR, rhoR, cR = stateR
    K = len(eos_list)

    # --- SLAU2 pressure-free face velocity (all-Mach, Deng 2025) ---
    c_avg = 0.5 * (cL + cR)
    u_rms = np.sqrt(0.5 * (uL ** 2 + uR ** 2))
    M_hat = np.minimum(1.0, u_rms / np.maximum(c_avg, _EPS))
    chi = (1.0 - M_hat) ** 2
    rho_avg = 0.5 * (rhoL + rhoR)
    V_avg = (rhoL * uL + rhoR * uR) / np.maximum(rhoL + rhoR, _EPS)
    u_face = V_avg - (chi / np.maximum(rho_avg * c_avg, _EPS)) * (pR - pL)

    # --- Upwind face values ---
    upw = (u_face >= 0.0).astype(float)
    ar_up = [upw * ar_L[k] + (1.0 - upw) * ar_R[k] for k in range(K)]
    a_up = [upw * a_L[k] + (1.0 - upw) * a_R[k] for k in range(K - 1)]
    ru_up = upw * ru_L + (1.0 - upw) * ru_R
    p_up = upw * pL + (1.0 - upw) * pR

    # Phase internal energy at upwind state (for APEC)
    if K - 1 < K:
        a_up_full = list(a_up) + [np.maximum(1.0 - sum(a_up), 1e-12)]
    else:
        a_up_full = a_up
    e_up = []
    for k in range(K):
        rho_k_up = ar_up[k] / np.maximum(a_up_full[k], 1e-8)
        rho_k_up = np.maximum(rho_k_up, _EPS)
        e_up.append(eos_list[k].energy(rho_k_up, np.maximum(p_up, 1.0)))

    # --- Mass / alpha flux: upwind ---
    F_ar = [ar_up[k] * u_face for k in range(K)]
    F_a = [a_up[k] * u_face for k in range(K - 1)]

    # --- Momentum flux: advective only (NO pressure term — IM1 handles it) ---
    F_ru = ru_up * u_face

    # --- APEC energy flux: advective only (NO pressure work — IM1 handles it) ---
    #     F_rE = Σ ε_k F_{ar_k} + ½ u_face² F_rho
    F_rho = sum(F_ar)
    F_rE = (sum(e_up[k] * F_ar[k] for k in range(K))
            + 0.5 * u_face ** 2 * F_rho)

    return F_ar, F_ru, F_rE, F_a, u_face


# Keep Rusanov as fallback
def _rusanov_flux_K(stateL, stateR, eos_list):
    """Legacy Rusanov; kept for debugging. Use _apec_slau2_flux_K instead."""
    ar_L, ru_L, rE_L, a_L, pL, uL, rhoL, cL = stateL
    ar_R, ru_R, rE_R, a_R, pR, uR, rhoR, cR = stateR
    K = len(eos_list)
    S_max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)
    F_ar_L = [ar_L[k] * uL for k in range(K)]
    F_ar_R = [ar_R[k] * uR for k in range(K)]
    F_ru_L = ru_L * uL + pL; F_ru_R = ru_R * uR + pR
    F_rE_L = (rE_L + pL) * uL; F_rE_R = (rE_R + pR) * uR
    F_a_L = [a_L[k] * uL for k in range(K - 1)]
    F_a_R = [a_R[k] * uR for k in range(K - 1)]
    F_ar = [0.5 * (F_ar_L[k] + F_ar_R[k]) - 0.5 * S_max * (ar_R[k] - ar_L[k])
            for k in range(K)]
    F_ru = 0.5 * (F_ru_L + F_ru_R) - 0.5 * S_max * (ru_R - ru_L)
    F_rE = 0.5 * (F_rE_L + F_rE_R) - 0.5 * S_max * (rE_R - rE_L)
    F_a = [0.5 * (F_a_L[k] + F_a_R[k]) - 0.5 * S_max * (a_R[k] - a_L[k])
           for k in range(K - 1)]
    u_face = (rhoL * uL + rhoR * uR) / np.maximum(rhoL + rhoR, _EPS)
    return F_ar, F_ru, F_rE, F_a, u_face


# Alias for compat: primary flux is now APEC+SLAU2
_hllc_flux_K = _apec_slau2_flux_K


def _mmacm_ex_K(ar_L_f, ar_R_f, aLs, aRs, rhoLs, rhoRs, u_face, eos_list,
                 eps_intf=1e-3, thinc_beta=2.0):
    """MMACM-Ex interface sharpening for K phases (Zhao et al. 2025 generalization).

    Applies H_k·pure_downwind correction to ALL K species, maintaining
    Σ α_k = 1 and mass conservation. Follows Zhao 2025 Eq. 30-32.

    For each interface pair (k, l) where α_k ~ 0 and α_l ~ 1 at cell i:
        G_α_k = H_k · (α_k_down · u_face - F_α_k_base)

    Simplification for K=2: same as existing MMACM-Ex.
    For K>2: apply pairwise between each active phase and its dominant neighbor.

    Returns: F_ar_corrections (K arrays at faces)
    """
    K = len(eos_list)
    n_faces = len(u_face)
    # Identify interface faces: where α differs notably between L and R
    # H_k_face = max(α_k_L, α_k_R) · min(1-α_k_L, 1-α_k_R) · 2  → peak at α=0.5
    F_ar_corr = [np.zeros_like(u_face) for _ in range(K)]
    upw = (u_face >= 0.0)
    for k in range(K):
        # Interface indicator (non-zero only at diffuse interfaces)
        a_k_max = np.maximum(aLs[k], aRs[k])
        a_k_min = np.minimum(aLs[k], aRs[k])
        # H_k = tanh(β·(a_max - a_min)) — detects sharp transitions
        H_k = np.tanh(thinc_beta * np.maximum(a_k_max - a_k_min, 0.0))
        # Pure downwind α_k at face
        a_k_down = np.where(upw, aRs[k], aLs[k])
        rho_k_down = np.where(upw, rhoRs[k], rhoLs[k])
        # Base upwind flux
        ar_k_base = np.where(upw, ar_L_f[k], ar_R_f[k])
        # Correction: downwind (α_k ρ_k)·u minus upwind base
        ar_k_down = a_k_down * rho_k_down
        F_ar_corr[k] = H_k * (ar_k_down * u_face - ar_k_base * u_face)
    # Conservation: sum of corrections should be zero (mass)
    # Normalize by subtracting mean if needed
    total = sum(F_ar_corr)
    # Redistribute to keep Σ=0 (each phase proportional to its mass)
    rho_face_down = np.where(upw, sum(rhoRs[k]*aRs[k] for k in range(K)),
                              sum(rhoLs[k]*aLs[k] for k in range(K)))
    for k in range(K):
        # Subtract the total weighted by this phase's fraction
        mass_frac_k = (np.where(upw, aRs[k]*rhoRs[k], aLs[k]*rhoLs[k])
                       / np.maximum(rho_face_down, _EPS))
        F_ar_corr[k] = F_ar_corr[k] - mass_frac_k * total
    return F_ar_corr


def rhs_K(ar_list, ru, rE, a_list, eos_list, dx, bc_l, bc_r,
          use_mmacm_ex=False):
    """Advective RHS for K-phase Kapila using HLLC flux."""
    K = len(eos_list)
    N = len(ru)
    p, u, T, rhos, es, c_mix = cons_to_prim_K(ar_list, ru, rE, a_list, eos_list)

    # TVD reconstruct all primitives + conservatives
    def _recon(q):
        q_ext = _ghost(q, bc_l, bc_r)
        # Simple slope limiting using 3-cell stencil
        s = _tvd_slope(q_ext)  # length N+2
        # Face values: L at face i+1/2 from cell i, R from cell i+1
        qL = q_ext[:-1] + 0.5 * s[:-1]  # length N+1 (at faces 0..N)
        qR = q_ext[1:] - 0.5 * s[1:]    # length N+1
        return qL, qR

    # Reconstruct rhos, u, p, a (primitive)
    rhoLs = []; rhoRs = []
    for k in range(K):
        rL, rR = _recon(rhos[k])
        rhoLs.append(np.maximum(rL, _EPS))
        rhoRs.append(np.maximum(rR, _EPS))
    uL, uR = _recon(u)
    pL_f, pR_f = _recon(p)
    pL_f = np.maximum(pL_f, 1.0); pR_f = np.maximum(pR_f, 1.0)
    aLs = []; aRs = []
    for k in range(K):
        aL, aR = _recon(a_list[k])
        aLs.append(np.clip(aL, 0.0, 1.0))
        aRs.append(np.clip(aR, 0.0, 1.0))
    # Normalize alpha
    sL = sum(aLs); sR = sum(aRs)
    for k in range(K):
        aLs[k] = aLs[k] / np.maximum(sL, _EPS)
        aRs[k] = aRs[k] / np.maximum(sR, _EPS)

    # Build face states
    ar_L_f = [aLs[k] * rhoLs[k] for k in range(K)]
    ar_R_f = [aRs[k] * rhoRs[k] for k in range(K)]
    rho_L_f = sum(ar_L_f); rho_R_f = sum(ar_R_f)
    ru_L_f = rho_L_f * uL; ru_R_f = rho_R_f * uR
    e1_L = [eos_list[k].energy(rhoLs[k], pL_f) for k in range(K)]
    e1_R = [eos_list[k].energy(rhoRs[k], pR_f) for k in range(K)]
    rho_e_L = sum(ar_L_f[k] * e1_L[k] for k in range(K))
    rho_e_R = sum(ar_R_f[k] * e1_R[k] for k in range(K))
    rE_L_f = rho_e_L + 0.5 * rho_L_f * uL ** 2
    rE_R_f = rho_e_R + 0.5 * rho_R_f * uR ** 2

    # Face sound speed (Wood mixture) — use max of L/R
    c_L_ext = _ghost(c_mix, bc_l, bc_r)
    c_L = c_L_ext[:-1]; c_R = c_L_ext[1:]

    stateL = (ar_L_f, ru_L_f, rE_L_f, aLs, pL_f, uL, rho_L_f, c_L)
    stateR = (ar_R_f, ru_R_f, rE_R_f, aRs, pR_f, uR, rho_R_f, c_R)
    F_ar, F_ru, F_rE, F_a, u_face = _hllc_flux_K(stateL, stateR, eos_list)

    # MMACM-Ex K-phase interface sharpening (optional)
    if use_mmacm_ex:
        F_ar_corr = _mmacm_ex_K(ar_L_f, ar_R_f, aLs, aRs, rhoLs, rhoRs,
                                 u_face, eos_list)
        for k in range(K):
            F_ar[k] = F_ar[k] + F_ar_corr[k]

    inv_dx = 1.0 / dx
    d_ar = [-(F_ar[k][1:N + 1] - F_ar[k][0:N]) * inv_dx for k in range(K)]
    d_ru = -(F_ru[1:N + 1] - F_ru[0:N]) * inv_dx
    d_rE = -(F_rE[1:N + 1] - F_rE[0:N]) * inv_dx
    # α transport: ∂α_k/∂t + u·∂α_k/∂x = 0 in non-conservative form
    # Upwind: F_a_k is a_k·u → same as conservation, divergence form
    # For α equation: ∂α_k/∂t = -∂(α_k·u)/∂x + α_k·∂u/∂x (Kapila form)
    # Simplification: pure advection of α_k
    du_dx = (u_face[1:N + 1] - u_face[0:N]) * inv_dx
    d_a = []
    for k in range(K - 1):
        # ∂α_k/∂t = -∂(α_k u)/∂x + α_k · ∂u/∂x  (Allaire-Massoni form)
        d_a.append(-(F_a[k][1:N + 1] - F_a[k][0:N]) * inv_dx + a_list[k] * du_dx)
    return d_ar, d_ru, d_rE, d_a


def rhs_K_imex(ar_list, ru, rE, a_list, eos_list, dx, bc_l, bc_r,
               use_mmacm_ex=False):
    """Advective-only RHS for K-phase Kapila IMEX transport step.

    Identical to rhs_K except the flux function is replaced with
    _apec_slau2_flux_K_imex which excludes pressure terms.
    This is the correct transport step for Peluchon IM1 Strang splitting:
      A(dt/2) [IM1 acoustic] -> T(dt) [this advective RHS] -> A(dt/2) [IM1 acoustic]

    Pressure work (p*u terms in momentum and energy) must NOT appear here
    because the IM1 acoustic step adds them implicitly. Including them here
    would cause double-counting of acoustic physics.
    """
    K = len(eos_list)
    N = len(ru)
    p, u, T, rhos, es, c_mix = cons_to_prim_K(ar_list, ru, rE, a_list, eos_list)

    def _recon(q):
        q_ext = _ghost(q, bc_l, bc_r)
        s = _tvd_slope(q_ext)
        qL = q_ext[:-1] + 0.5 * s[:-1]
        qR = q_ext[1:] - 0.5 * s[1:]
        return qL, qR

    rhoLs = []; rhoRs = []
    for k in range(K):
        rL, rR = _recon(rhos[k])
        rhoLs.append(np.maximum(rL, _EPS))
        rhoRs.append(np.maximum(rR, _EPS))
    uL, uR = _recon(u)
    pL_f, pR_f = _recon(p)
    pL_f = np.maximum(pL_f, 1.0); pR_f = np.maximum(pR_f, 1.0)
    aLs = []; aRs = []
    for k in range(K):
        aL, aR = _recon(a_list[k])
        aLs.append(np.clip(aL, 0.0, 1.0))
        aRs.append(np.clip(aR, 0.0, 1.0))
    sL = sum(aLs); sR = sum(aRs)
    for k in range(K):
        aLs[k] = aLs[k] / np.maximum(sL, _EPS)
        aRs[k] = aRs[k] / np.maximum(sR, _EPS)

    ar_L_f = [aLs[k] * rhoLs[k] for k in range(K)]
    ar_R_f = [aRs[k] * rhoRs[k] for k in range(K)]
    rho_L_f = sum(ar_L_f); rho_R_f = sum(ar_R_f)
    ru_L_f = rho_L_f * uL; ru_R_f = rho_R_f * uR
    e1_L = [eos_list[k].energy(rhoLs[k], pL_f) for k in range(K)]
    e1_R = [eos_list[k].energy(rhoRs[k], pR_f) for k in range(K)]
    rho_e_L = sum(ar_L_f[k] * e1_L[k] for k in range(K))
    rho_e_R = sum(ar_R_f[k] * e1_R[k] for k in range(K))
    rE_L_f = rho_e_L + 0.5 * rho_L_f * uL ** 2
    rE_R_f = rho_e_R + 0.5 * rho_R_f * uR ** 2

    c_L_ext = _ghost(c_mix, bc_l, bc_r)
    c_L = c_L_ext[:-1]; c_R = c_L_ext[1:]

    stateL = (ar_L_f, ru_L_f, rE_L_f, aLs, pL_f, uL, rho_L_f, c_L)
    stateR = (ar_R_f, ru_R_f, rE_R_f, aRs, pR_f, uR, rho_R_f, c_R)

    # Use IMEX-compatible flux (pressure excluded)
    F_ar, F_ru, F_rE, F_a, u_face = _apec_slau2_flux_K_imex(stateL, stateR, eos_list)

    if use_mmacm_ex:
        F_ar_corr = _mmacm_ex_K(ar_L_f, ar_R_f, aLs, aRs, rhoLs, rhoRs,
                                 u_face, eos_list)
        for k in range(K):
            F_ar[k] = F_ar[k] + F_ar_corr[k]

    inv_dx = 1.0 / dx
    d_ar = [-(F_ar[k][1:N + 1] - F_ar[k][0:N]) * inv_dx for k in range(K)]
    d_ru = -(F_ru[1:N + 1] - F_ru[0:N]) * inv_dx
    d_rE = -(F_rE[1:N + 1] - F_rE[0:N]) * inv_dx
    du_dx = (u_face[1:N + 1] - u_face[0:N]) * inv_dx
    d_a = []
    for k in range(K - 1):
        d_a.append(-(F_a[k][1:N + 1] - F_a[k][0:N]) * inv_dx + a_list[k] * du_dx)
    return d_ar, d_ru, d_rE, d_a


def _peluchon_acoustic_im1_K(ar_list_star, ru_star, rE_star, a_list_new,
                              eos_list, dx, dt, bc_l, bc_r):
    """K-phase Peluchon IM1 acoustic step (generalization of K=2 version).

    Block-tridiagonal system on (u, p) → Thomas O(N). Identical structure
    to K=2 (u, p are scalar regardless of K). Mixture density / sound speed
    computed via EOS-agnostic utilities.

    Returns: (ar_list_new, ru_new, rE_new)
    """
    K = len(eos_list)
    N = len(ru_star)
    rho_star = sum(ar_list_star)
    u_star = ru_star / np.maximum(rho_star, _EPS)
    rho_e = rE_star - 0.5 * rho_star * u_star ** 2

    _af = 1e-8
    rhos = [np.maximum(ar_list_star[k] / np.maximum(a_list_new[k], _af), _EPS)
            for k in range(K)]
    # Mixture pressure
    p_star = mixture_pressure_solve_K(a_list_new, rhos, rho_e, eos_list)
    p_star = np.maximum(p_star, 1.0)

    # Phase e and c² via EOS
    es = [eos_list[k].energy(rhos[k], p_star) for k in range(K)]
    c_sq_mix, _ = mixture_sound_speed_K(a_list_new, rhos, es, p_star, eos_list)
    c_mix = np.sqrt(np.maximum(c_sq_mix, _EPS))
    a_imp = rho_star * c_mix  # impedance ρc

    # Face impedance average
    if bc_l == 'periodic':
        a_ext = np.concatenate([a_imp[-1:], a_imp, a_imp[:1]])
    else:
        a_ext = np.concatenate([a_imp[:1], a_imp, a_imp[-1:]])
    aLf = a_ext[0:N + 1]
    aRf = a_ext[1:N + 2]
    S_face = aLf + aRf
    S_safe = np.maximum(S_face, _EPS)
    am_S = aLf / S_safe
    ap_S = aRf / S_safe
    amap_S = aLf * aRf / S_safe
    inv_S = 1.0 / S_safe
    vartheta = 1.0 / np.maximum(rho_star, _EPS)
    a_sq = a_imp ** 2
    sigma = dt / dx

    # Build block-tridiag (identical structure to K=2)
    lower = np.zeros((N, 2, 2))
    diag = np.zeros((N, 2, 2))
    upper = np.zeros((N, 2, 2))
    rhs_vec = np.zeros((N, 2))
    for i in range(N):
        vi = vartheta[i]; ai2 = a_sq[i]
        fL = i; fR = i + 1
        lower[i, 0, 0] = sigma * vi * (-amap_S[fL])
        lower[i, 0, 1] = sigma * vi * (-ap_S[fL])
        lower[i, 1, 0] = sigma * vi * ai2 * (-am_S[fL])
        lower[i, 1, 1] = sigma * vi * ai2 * (-inv_S[fL])
        diag[i, 0, 0] = 1.0 + sigma * vi * (amap_S[fL] + amap_S[fR])
        diag[i, 0, 1] = sigma * vi * (-am_S[fL] + ap_S[fR])
        diag[i, 1, 0] = sigma * vi * ai2 * (-ap_S[fL] + am_S[fR])
        diag[i, 1, 1] = 1.0 + sigma * vi * ai2 * (inv_S[fL] + inv_S[fR])
        upper[i, 0, 0] = sigma * vi * (-amap_S[fR])
        upper[i, 0, 1] = sigma * vi * (am_S[fR])
        upper[i, 1, 0] = sigma * vi * ai2 * (ap_S[fR])
        upper[i, 1, 1] = sigma * vi * ai2 * (-inv_S[fR])
        rhs_vec[i, 0] = u_star[i]
        rhs_vec[i, 1] = p_star[i]

    if bc_l == 'transmissive':
        diag[0] += lower[0]; lower[0] = 0.0
    if bc_r == 'transmissive':
        diag[N - 1] += upper[N - 1]; upper[N - 1] = 0.0

    # Reuse the K=2 block tridiag solvers (works for any 2×2 blocks)
    from .explicit_mmacm_ex import _block_tridiag_solve, _block_tridiag_periodic
    if bc_l == 'periodic' and bc_r == 'periodic':
        u_new, p_new = _block_tridiag_periodic(lower, diag, upper, rhs_vec)
    else:
        u_new, p_new = _block_tridiag_solve(lower, diag, upper, rhs_vec)

    # Face (ū, p̄) via impedance-weighted upwind
    if bc_l == 'periodic':
        u_ext2 = np.concatenate([u_new[-1:], u_new, u_new[:1]])
        p_ext2 = np.concatenate([p_new[-1:], p_new, p_new[:1]])
    else:
        u_ext2 = np.concatenate([u_new[:1], u_new, u_new[-1:]])
        p_ext2 = np.concatenate([p_new[:1], p_new, p_new[-1:]])
    u_bar = (aLf * u_ext2[0:N + 1] + aRf * u_ext2[1:N + 2]
             - (p_ext2[1:N + 2] - p_ext2[0:N + 1])) / S_safe
    p_bar = (aRf * p_ext2[0:N + 1] + aLf * p_ext2[1:N + 2]
             - aLf * aRf * (u_ext2[1:N + 2] - u_ext2[0:N + 1])) / S_safe

    # Conservative update: mass unchanged, ru += -dt·∂p̄/∂x, rE += -dt·∂(p̄ū)/∂x
    ar_list_new = [ar.copy() for ar in ar_list_star]  # mass preserved
    ru_new = ru_star - sigma * (p_bar[1:N + 1] - p_bar[0:N])
    F_rE_face = p_bar * u_bar
    rE_new = rE_star - sigma * (F_rE_face[1:N + 1] - F_rE_face[0:N])
    return ar_list_new, ru_new, rE_new


def solve_kapila_K(eos_list, ar_list_0, ru_0, rE_0, a_list_0,
                    dx, t_end, cfl=0.3,
                    bc_l='transmissive', bc_r='transmissive',
                    max_steps=10000, print_interval=100,
                    use_mmacm_ex=False,
                    acoustic_method='explicit'):
    """K-phase Kapila explicit SSP-RK3 solver."""
    K = len(eos_list)
    # Ensure all EOS are EOS objects
    eos_list = [to_eos(e) for e in eos_list]

    ar = [np.array(x, dtype=float).copy() for x in ar_list_0]
    ru = np.array(ru_0, dtype=float).copy()
    rE = np.array(rE_0, dtype=float).copy()
    a = [np.array(x, dtype=float).copy() for x in a_list_0]
    # Ensure K-1 alphas (last one = 1 - sum)
    if len(a) == K:
        a = a[:K - 1]

    N = len(ru)
    t = 0.0
    step = 0

    def _recon_last_alpha(a_kmin1):
        """Append α_K = 1 - sum."""
        aK = np.maximum(1.0 - sum(a_kmin1), 1e-12)
        return a_kmin1 + [aK]

    def _rhs_call(ar, ru, rE, a_kmin1):
        a_full = _recon_last_alpha(a_kmin1)
        if acoustic_method == 'im1':
            # IMEX transport step: pressure excluded (IM1 acoustic handles it)
            return rhs_K_imex(ar, ru, rE, a_full, eos_list, dx, bc_l, bc_r,
                               use_mmacm_ex=use_mmacm_ex)
        else:
            return rhs_K(ar, ru, rE, a_full, eos_list, dx, bc_l, bc_r,
                         use_mmacm_ex=use_mmacm_ex)

    while t < t_end and step < max_steps:
        # Primitive for dt
        a_full = _recon_last_alpha(a)
        p, u, T, rhos, es, c_mix = cons_to_prim_K(ar, ru, rE, a_full, eos_list)
        # Always use acoustic CFL for stability (even for IM1 path) —
        # SLAU2 pressure-velocity coupling in rhs_K_imex requires
        # acoustic resolution to prevent splitting error amplification.
        max_speed = np.max(np.abs(u) + c_mix)
        dt = cfl * dx / max(max_speed, _EPS)
        dt = min(dt, t_end - t)
        if dt <= 0 or not np.isfinite(dt):
            print(f'  dt invalid ({dt}), terminating')
            break
        # State finite check
        if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rE))
                and all(np.all(np.isfinite(a)) for a in ar)):
            print(f'  state became non-finite at step {step}')
            break

        # IM1 acoustic half-step (Strang splitting) — before transport
        if acoustic_method == 'im1':
            a_full = _recon_last_alpha(a)
            ar, ru, rE = _peluchon_acoustic_im1_K(
                ar, ru, rE, a_full, eos_list, dx, dt / 2.0, bc_l, bc_r)

        # SSP-RK3 stage 1
        d_ar, d_ru, d_rE, d_a = _rhs_call(ar, ru, rE, a)
        ar_1 = [np.maximum(ar[k] + dt * d_ar[k], _EPS) for k in range(K)]
        ru_1 = ru + dt * d_ru
        rE_1 = rE + dt * d_rE
        a_1 = [np.clip(a[k] + dt * d_a[k], _EPS, 1.0 - _EPS) for k in range(K - 1)]

        # stage 2
        d_ar, d_ru, d_rE, d_a = _rhs_call(ar_1, ru_1, rE_1, a_1)
        ar_2 = [np.maximum(0.75 * ar[k] + 0.25 * (ar_1[k] + dt * d_ar[k]), _EPS) for k in range(K)]
        ru_2 = 0.75 * ru + 0.25 * (ru_1 + dt * d_ru)
        rE_2 = 0.75 * rE + 0.25 * (rE_1 + dt * d_rE)
        a_2 = [np.clip(0.75 * a[k] + 0.25 * (a_1[k] + dt * d_a[k]), _EPS, 1.0 - _EPS) for k in range(K - 1)]

        # stage 3
        d_ar, d_ru, d_rE, d_a = _rhs_call(ar_2, ru_2, rE_2, a_2)
        ar = [np.maximum((1. / 3) * ar[k] + (2. / 3) * (ar_2[k] + dt * d_ar[k]), _EPS) for k in range(K)]
        ru = (1. / 3) * ru + (2. / 3) * (ru_2 + dt * d_ru)
        rE = (1. / 3) * rE + (2. / 3) * (rE_2 + dt * d_rE)
        a = [np.clip((1. / 3) * a[k] + (2. / 3) * (a_2[k] + dt * d_a[k]), _EPS, 1.0 - _EPS) for k in range(K - 1)]

        # IM1 acoustic half-step (Strang splitting) — after transport
        if acoustic_method == 'im1':
            a_full = _recon_last_alpha(a)
            ar, ru, rE = _peluchon_acoustic_im1_K(
                ar, ru, rE, a_full, eos_list, dx, dt / 2.0, bc_l, bc_r)

        t += dt
        step += 1
        if step % print_interval == 0:
            print(f'  step={step:4d}  t={t:.3e}  dt={dt:.3e}  '
                  f'p=[{p.min():.2e},{p.max():.2e}]  u_max={np.max(np.abs(u)):.2f}')

    print(f'Done: {step} steps, t={t:.4e}')
    a_full = _recon_last_alpha(a)
    return t, ar, ru, rE, a_full
