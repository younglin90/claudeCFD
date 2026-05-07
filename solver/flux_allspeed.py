"""
Numerical flux for 1D multi-component compressible Euler equations.

State layout (solver_1d.py convention):
    U = [rhoY_1, ..., rhoY_Ns, rho*u, rho*E]

    rho   = sum_i U[i]  (mixture density)
    rho*u = U[Ns]       (momentum)
    rho*E = U[Ns+1]     (total energy)

Flux implementations:

1. hllc_flux:
   HLLC (Harten-Lax-van Leer-Contact) flux — Toro (1994).
   Wave speed estimates use Davis bounds (min/max of left/right
   characteristic speeds).  The contact wave speed S* is computed
   from the exact HLLC formula.

   Ref: Toro, E.F. (1994) "Restoration of the contact surface in the
        HLL-Riemann solver." Shock Waves 4, 25-34.

2. ausm_plus_up_flux:
   Delegates to hllc_flux (kept for API compatibility).

Conservative state per cell: [rhoY_1,...,rhoY_Ns, rho*u, rho*E]
Ns = number of species, n_vars = Ns + 2.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

# ============================================================================
# EOS dispatch helpers
# ============================================================================

def _get_pressure(eos, rho_i: float, T: float) -> float:
    try:
        return float(eos.pressure(rho_i, T))
    except TypeError:
        return float(eos.pressure(rho_i))


def _get_internal_energy(eos, rho_i: float, T: float) -> float:
    try:
        return float(eos.internal_energy(rho_i, T))
    except TypeError:
        return float(eos.internal_energy(T))


def _get_sound_speed(eos, rho_i: float, T: float) -> float:
    return float(eos.sound_speed(rho_i, T))


# ============================================================================
# Pure-species density from (T, p)
# ============================================================================

def _rho_i_pure_from_T_p(eos, T: float, p: float) -> float:
    """
    Compute pure-species density rho_i_pure(T, p).
    Ideal Gas: rho = p / (R_s * T)
    NASG:      rho = (p + p_inf) / [(gamma-1)*c_v*T + b*(p+p_inf)]
    """
    from .eos.ideal import IdealGasEOS
    from .eos.nasg import NASGEOS
    if isinstance(eos, IdealGasEOS):
        return p / (eos.R_s * T)
    elif isinstance(eos, NASGEOS):
        num = p + eos.p_inf
        den = (eos.gamma - 1.0) * eos.c_v * T + eos.b * num
        return num / max(den, 1e-300)
    else:
        raise TypeError(f"Unknown EOS type for density inversion: {type(eos)}")


# ============================================================================
# Primitive variable recovery  cons_to_prim_allspeed
# ============================================================================

def cons_to_prim_allspeed(
    U: np.ndarray,
    eos_list: list,
    T_guess: float = 300.0,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Convert conservative state U to primitive (rho, u, p, T, rhoYi).

    Layout: U = [rhoY_1, ..., rhoY_Ns, rho*u, rho*E]

    For pure-species cells (dominant species Yi > 99.9%), uses a direct
    analytical inversion without iteration (fast path).

    For mixed cells, iterates on (T, p) using the isobaric constraint:
        sum_i Yi / rho_i_pure(T, p) = 1/rho

    Returns
    -------
    (rho, u, p, T, rhoYi)
    """
    from .eos.ideal import IdealGasEOS
    from .eos.nasg import NASGEOS

    Ns = len(eos_list)
    rhoYi = U[:Ns].copy()
    rho_u = U[Ns]
    rho_E = U[Ns + 1]

    rho = np.sum(rhoYi)
    rho = max(rho, 1e-300)
    rhoYi = np.clip(rhoYi, 0.0, rho)
    rhoYi[-1] = max(0.0, rho - np.sum(rhoYi[:-1]))

    u = rho_u / rho
    E = rho_E / rho
    e = max(E - 0.5 * u * u, 1e-20)

    Yi = rhoYi / rho
    i_dom = int(np.argmax(Yi))
    Yi_dom = Yi[i_dom]
    eos_dom = eos_list[i_dom]

    # ---- Fast path for essentially pure cells (Yi_dom > 99.99%) ----
    # Threshold must be strict: NASG water energy (988 kJ/kg) >> air energy (215 kJ/kg),
    # so even 0.01% water contamination in an "air" cell inflates the fast-path T/p.
    if Yi_dom > 0.9999:
        rho_d = rho  # pure-species: rho_mix = rho_i_pure

        if isinstance(eos_dom, IdealGasEOS):
            # e = c_v * T  (no pressure-dependent offset for ideal gas)
            T = e / eos_dom.c_v
            T = max(T, 1.0)
            p = eos_dom.R_s * rho_d * T

        elif isinstance(eos_dom, NASGEOS):
            # Fix 2 (CRITICAL): Use pressure_from_rho_e to avoid catastrophic
            # cancellation that occurs when computing T first and then p via the
            # large stiffness pressure p_inf (Abgrall & Karni 2001 pressure
            # equilibrium preservation).
            #
            # Direct path: e → p (no large-number cancellation) → T
            # p = (gamma-1)*rho*(e-q)/(1-b*rho) - 2*p_inf
            # T = (p+p_inf)*(1-b*rho) / ((gamma-1)*c_v*rho)
            rho_safe = max(rho_d, 1e-300)
            p = eos_dom.pressure_from_rho_e(rho_safe, e)
            p = max(p, 1.0)
            T = eos_dom.temperature_from_rho_p(rho_safe, p)
            T = max(T, 1.0)
        else:
            T = T_guess
            p = _get_pressure(eos_dom, rho_d, T)
            p = max(p, 0.0)

        return rho, u, p, T, rhoYi

    # ---- General (mixed) case ----
    # Get initial p from dominant species
    if Yi_dom > 1e-10:
        rho_dom_pure = rhoYi[i_dom] / max(Yi_dom, 1e-30)
        p_init = _get_pressure(eos_dom, rho_dom_pure, T_guess)
        p_init = max(p_init, 1.0)
    else:
        p_init = rho * max(T_guess, 1.0) * 287.0

    def _T_from_e_p(p_v, T_prev):
        cv_m = 0.0
        off_m = 0.0
        for i, eos in enumerate(eos_list):
            yi = Yi[i]
            if yi < 1e-30:
                continue
            cv_m += yi * eos.c_v
            if isinstance(eos, NASGEOS):
                rho_ip = max(_rho_i_pure_from_T_p(eos, T_prev, p_v), 1e-300)
                off_m += yi * (eos.q + eos.p_inf * (1.0 - eos.b * rho_ip)
                                / ((eos.gamma - 1.0) * rho_ip))
        cv_m = max(cv_m, 1e-300)
        return max((e - off_m) / cv_m, 1.0)

    def _isobar_res(p_v, T_v):
        inv_r = sum(Yi[i] / max(_rho_i_pure_from_T_p(eos_list[i], T_v, p_v), 1e-300)
                    for i in range(Ns) if Yi[i] > 1e-30)
        return inv_r - 1.0 / rho

    # ---- Crude energy-based T estimate to detect unphysical state ----
    # If the mixed-cell rhoE is inconsistent (e.g. after double-flux energy
    # correction), T_from_energy can be far below T_guess.  In that case fall
    # back to fixing T = T_guess and solving only the isobaric constraint for p.
    cv_m_crude = sum(Yi[i] * eos_list[i].c_v for i in range(Ns) if Yi[i] > 1e-30)
    cv_m_crude = max(cv_m_crude, 1e-300)
    T_from_energy = max(e / cv_m_crude, 1.0)
    T_ref = max(T_guess, 1.0)

    tol = 1e-7 / max(rho, 1e-300)

    if T_from_energy < 0.1 * T_ref or T_from_energy > 10.0 * T_ref:
        # Fallback path: energy is unphysical — fix T = T_guess, solve only
        # the isobaric residual R = Σ Yᵢ/ρᵢ_pure(T,p) - 1/ρ for p.
        T = T_ref
        p = p_init
        for _ in range(100):
            R = _isobar_res(p, T)
            if abs(R) < tol:
                break
            dpf = max(abs(p) * 1e-4, 100.0)
            dRdp = (_isobar_res(p + dpf, T) - R) / dpf
            if abs(dRdp) > 1e-50:
                p = max(p - R / dRdp, 1.0)
            else:
                break
        p = max(p, 0.0)
        return rho, u, p, T, rhoYi

    # Normal path: energy is physically consistent — iterate (T, p) jointly.
    p = p_init
    T = T_ref
    for _ in range(50):  # Fix 3: increased from 15 to 50 for better convergence
        T = _T_from_e_p(p, T)
        R = _isobar_res(p, T)
        if abs(R) < tol:
            break
        dpf = max(abs(p) * 1e-4, 100.0)
        dRdp = (_isobar_res(p + dpf, T) - R) / dpf
        if abs(dRdp) > 1e-50:
            p = max(p - R / dRdp, 1.0)
        else:
            break

    T = max(T, 1.0)
    p = max(p, 0.0)
    return rho, u, p, T, rhoYi


# ============================================================================
# Mixture sound speed (Wood's formula, using pure-species densities)
# ============================================================================

def _mixture_sound_speed(
    rho: float,
    Yi: np.ndarray,
    rhoYi: np.ndarray,
    T: float,
    p: float,
    eos_list: list,
) -> float:
    """Wood's formula: 1/(rho*a^2) = sum_i Yi / (rho_i_pure * a_i^2)."""
    inv_rhoa2 = 0.0
    for i, eos in enumerate(eos_list):
        yi = Yi[i]
        if yi < 1e-30:
            continue
        rho_i = max(_rho_i_pure_from_T_p(eos, T, max(p, 1.0)), 1e-300)
        try:
            a_i = _get_sound_speed(eos, rho_i, T)
        except Exception:
            continue
        if a_i > 0.0:
            inv_rhoa2 += yi / (rho_i * a_i * a_i)

    if inv_rhoa2 > 0.0:
        a2 = 1.0 / (rho * inv_rhoa2)
    else:
        a2 = max(
            (_get_sound_speed(eos_list[i], max(_rho_i_pure_from_T_p(
                eos_list[i], T, max(p, 1.0)), 1e-300), T) ** 2
             for i in range(len(eos_list)) if Yi[i] > 1e-30),
            default=1.0
        )
    return np.sqrt(max(a2, 1e-6))


# ============================================================================
# Physical flux at a single cell
# ============================================================================

def physical_flux_allspeed(
    U: np.ndarray,
    eos_list: list,
    T_guess: float = 300.0,
) -> np.ndarray:
    """Physical Euler flux F(U) = [rhoYi*u, rho*u^2+p, (rhoE+p)*u]."""
    Ns = len(eos_list)
    rho, u, p, T, rhoYi = cons_to_prim_allspeed(U, eos_list, T_guess)
    F = np.empty(Ns + 2, dtype=float)
    for i in range(Ns):
        F[i] = rhoYi[i] * u
    F[Ns] = rho * u * u + p
    F[Ns + 1] = (U[Ns + 1] + p) * u
    return F


# ============================================================================
# APEC εᵢ helper
# ============================================================================

def _compute_epsilon_i_at_state(
    rho: float,
    Yi: np.ndarray,
    T: float,
    p: float,
    eos_list: list,
) -> np.ndarray:
    """
    Compute εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p} for all species at given state.

    Formula:
        εᵢ = (∂ρe/∂ρᵢ)_T - (ρ·cv_mix / (∂p/∂T)_mix) · (∂p/∂ρᵢ)_T

    Uses pure-species density ρᵢ_pure(T, p) for each species partial derivative.

    Ref: CLAUDE.md § APEC Flux
    """
    from .eos.ideal import IdealGasEOS
    from .eos.nasg import NASGEOS

    Ns = len(eos_list)

    # Compute pure-species densities at (T, p)
    rho_i_pure = np.empty(Ns)
    for i, eos in enumerate(eos_list):
        rho_i_pure[i] = max(_rho_i_pure_from_T_p(eos, T, max(p, 1.0)), 1e-300)

    # cv_mix = sum_i Yi * cv_i
    cv_mix = 0.0
    for i, eos in enumerate(eos_list):
        cv_mix += Yi[i] * eos.c_v
    cv_mix = max(cv_mix, 1e-300)
    rho_cv_mix = rho * cv_mix

    # (dp/dT)_mix = sum_i Yi * (dp/dT)_i(rho_i_pure)
    dp_dT_mix = 0.0
    for i, eos in enumerate(eos_list):
        if isinstance(eos, IdealGasEOS):
            dp_dT_mix += Yi[i] * eos.dp_dT(rho_i_pure[i])
        elif isinstance(eos, NASGEOS):
            dp_dT_mix += Yi[i] * eos.dp_dT(rho_i_pure[i])
        else:
            # fallback: use numerical derivative
            dp_dT_mix += Yi[i] * eos.dp_dT(rho_i_pure[i])
    dp_dT_mix = max(abs(dp_dT_mix), 1e-300) * (1.0 if dp_dT_mix >= 0.0 else -1.0)

    # εᵢ for each species
    eps = np.empty(Ns)
    for i, eos in enumerate(eos_list):
        if isinstance(eos, IdealGasEOS):
            # (∂ρe/∂ρᵢ)_T = e_i = cv_i * T
            drhoE_drho_i_T = eos.c_v * T
            # (∂p/∂ρᵢ)_T = R_s * T  (rho_i_pure passed for API uniformity)
            dp_drho_i_T = eos.dp_drho(rho_i_pure[i], T)
        elif isinstance(eos, NASGEOS):
            # (∂ρe/∂ρᵢ)_T = cv_i*T + q - p_inf*b/(gamma-1)
            drhoE_drho_i_T = eos.drho_e_drho_i_T(rho_i_pure[i], T)
            # (∂p/∂ρᵢ)_T using pure-species density
            dp_drho_i_T = eos.dp_drho(rho_i_pure[i], T)
        else:
            drhoE_drho_i_T = eos.c_v * T
            dp_drho_i_T = 0.0

        eps[i] = drhoE_drho_i_T - (rho_cv_mix / dp_dT_mix) * dp_drho_i_T

    return eps


# ============================================================================
# HLLC flux (Toro 1994) — primary interface flux
# ============================================================================

def hllc_flux(
    UL: np.ndarray,
    UR: np.ndarray,
    eos_list: list,
    T_guess_L: float = 300.0,
    T_guess_R: float = 300.0,
) -> np.ndarray:
    """
    HLLC flux (Toro 1994) for multi-species compressible flow.

    State layout: U = [rhoY_1, ..., rhoY_Ns, rho*u, rho*E]

    Wave speeds are estimated with Davis bounds (min/max eigenvalues).
    The contact wave S* is computed from the exact HLLC formula.
    Star-state densities preserve species mass fractions (Yi) across
    the contact discontinuity, consistent with the 5-equation
    multi-component model.

    Ref: Toro, E.F. (1994) "Restoration of the contact surface in the
         HLL-Riemann solver." Shock Waves 4, 25-34.

    Parameters
    ----------
    UL, UR : shape (Ns+2,)
    eos_list : list of Ns EOS objects
    T_guess_L, T_guess_R : initial temperature guesses for cons_to_prim

    Returns
    -------
    F : shape (Ns+2,)
    """
    Ns = len(eos_list)

    # --- Primitive variables ---
    rhoL, uL, pL, TL, rhoYiL = cons_to_prim_allspeed(UL, eos_list, T_guess_L)
    rhoR, uR, pR, TR, rhoYiR = cons_to_prim_allspeed(UR, eos_list, T_guess_R)
    YiL = rhoYiL / max(rhoL, 1e-300)
    YiR = rhoYiR / max(rhoR, 1e-300)

    # --- Sound speeds ---
    aL = _mixture_sound_speed(rhoL, YiL, rhoYiL, TL, pL, eos_list)
    aR = _mixture_sound_speed(rhoR, YiR, rhoYiR, TR, pR, eos_list)

    # --- Wave speed estimates (Davis bounds) ---
    SL = min(uL - aL, uR - aR)
    SR = max(uL + aL, uR + aR)

    # --- Contact wave speed ---
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    if abs(denom) < 1e-300:
        S_star = 0.5 * (uL + uR)
    else:
        S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom

    # --- Physical fluxes ---
    FL = np.empty(Ns + 2)
    for i in range(Ns):
        FL[i] = rhoYiL[i] * uL
    FL[Ns] = rhoL * uL * uL + pL
    FL[Ns + 1] = (UL[Ns + 1] + pL) * uL

    FR = np.empty(Ns + 2)
    for i in range(Ns):
        FR[i] = rhoYiR[i] * uR
    FR[Ns] = rhoR * uR * uR + pR
    FR[Ns + 1] = (UR[Ns + 1] + pR) * uR

    # --- HLLC flux selection ---
    if SL >= 0.0:
        return FL
    elif SR <= 0.0:
        return FR
    elif S_star >= 0.0:
        # Left star state
        coeff_L = rhoL * (SL - uL) / (SL - S_star)
        U_starL = np.empty(Ns + 2)
        for i in range(Ns):
            U_starL[i] = coeff_L * YiL[i]   # rho* * Yi (contact preserves Yi)
        U_starL[Ns] = coeff_L * S_star       # rho* * S*
        E_starL = UL[Ns + 1] / rhoL + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL)))
        U_starL[Ns + 1] = coeff_L * E_starL
        return FL + SL * (U_starL - UL)
    else:
        # Right star state
        coeff_R = rhoR * (SR - uR) / (SR - S_star)
        U_starR = np.empty(Ns + 2)
        for i in range(Ns):
            U_starR[i] = coeff_R * YiR[i]
        U_starR[Ns] = coeff_R * S_star
        E_starR = UR[Ns + 1] / rhoR + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR)))
        U_starR[Ns + 1] = coeff_R * E_starR
        return FR + SR * (U_starR - UR)


# ============================================================================
# Double-flux HLLC (Abgrall & Karni 2001) — energy flux per EOS side
# ============================================================================

def hllc_flux_double_energy(
    UL: np.ndarray,
    UR: np.ndarray,
    eos_list: list,
    T_guess_L: float = 300.0,
    T_guess_R: float = 300.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Double-flux HLLC (Abgrall & Karni 2001) for pressure-equilibrium preservation.

    Species and momentum fluxes are identical (standard HLLC).
    Energy flux is computed separately for the left and right cells:
      - F_left  : energy flux seen by the left  cell (uses left  EOS / state)
      - F_right : energy flux seen by the right cell (uses right EOS / state)

    This prevents spurious pressure oscillations at material interfaces
    where EOS parameters differ (e.g. water/air with large p_inf contrast).

    Ref: Abgrall, R. & Karni, S. (2001) "Computations of Compressible
         Multifluids." J. Comput. Phys. 169, 594-623.

    Parameters
    ----------
    UL, UR : shape (Ns+2,)
    eos_list : list of Ns EOS objects
    T_guess_L, T_guess_R : initial temperature guesses

    Returns
    -------
    F_left  : shape (Ns+2,) — flux used to update the LEFT  cell (right face)
    F_right : shape (Ns+2,) — flux used to update the RIGHT cell (left  face)
    p_int   : float — interface pressure (used for non-conservative momentum form)
    """
    Ns = len(eos_list)

    # --- Primitive variables ---
    rhoL, uL, pL, TL, rhoYiL = cons_to_prim_allspeed(UL, eos_list, T_guess_L)
    rhoR, uR, pR, TR, rhoYiR = cons_to_prim_allspeed(UR, eos_list, T_guess_R)
    YiL = rhoYiL / max(rhoL, 1e-300)
    YiR = rhoYiR / max(rhoR, 1e-300)

    # --- Sound speeds and wave speed bounds ---
    aL = _mixture_sound_speed(rhoL, YiL, rhoYiL, TL, pL, eos_list)
    aR = _mixture_sound_speed(rhoR, YiR, rhoYiR, TR, pR, eos_list)
    SL = min(uL - aL, uR - aR)
    SR = max(uL + aL, uR + aR)

    # --- Contact wave speed ---
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    if abs(denom) < 1e-300:
        S_star = 0.5 * (uL + uR)
    else:
        S_star = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom

    # --- Physical fluxes (species + momentum) ---
    FL = np.empty(Ns + 2)
    for i in range(Ns):
        FL[i] = rhoYiL[i] * uL
    FL[Ns] = rhoL * uL * uL + pL
    FL[Ns + 1] = (UL[Ns + 1] + pL) * uL

    FR = np.empty(Ns + 2)
    for i in range(Ns):
        FR[i] = rhoYiR[i] * uR
    FR[Ns] = rhoR * uR * uR + pR
    FR[Ns + 1] = (UR[Ns + 1] + pR) * uR

    # --- HLLC star-state energies ---
    if SL >= 0.0:
        # Supersonic right-moving: each cell uses its own rhoE
        rhoE_L_int = UL[Ns + 1]   # left cell uses its own rhoE
        rhoE_R_int = UR[Ns + 1]   # right cell uses its own rhoE
        p_int = pL
        u_int = uL
        F_species_mom = FL.copy()
    elif SR <= 0.0:
        # Supersonic left-moving: each cell uses its own rhoE
        rhoE_L_int = UL[Ns + 1]   # left cell uses its own rhoE
        rhoE_R_int = UR[Ns + 1]   # right cell uses its own rhoE
        p_int = pR
        u_int = uR
        F_species_mom = FR.copy()
    elif S_star >= 0.0:
        # Left star region: double-flux uses each cell's own rhoE
        coeff_L = rhoL * (SL - uL) / (SL - S_star)
        E_starL = UL[Ns + 1] / rhoL + (S_star - uL) * (S_star + pL / (rhoL * (SL - uL)))
        rhoE_L_int = UL[Ns + 1]   # left cell uses its own rhoE (Abgrall & Karni)
        rhoE_R_int = UR[Ns + 1]   # right cell uses its own rhoE (Abgrall & Karni)
        p_int = pL + rhoL * (SL - uL) * (S_star - uL)
        u_int = S_star
        # Star state for species/momentum flux uses standard HLLC star energy
        U_starL = np.empty(Ns + 2)
        for i in range(Ns):
            U_starL[i] = coeff_L * YiL[i]
        U_starL[Ns] = coeff_L * S_star
        U_starL[Ns + 1] = coeff_L * E_starL  # standard HLLC star energy for species/mom flux
        F_species_mom = FL + SL * (U_starL - UL)
    else:
        # Right star region: double-flux uses each cell's own rhoE
        coeff_R = rhoR * (SR - uR) / (SR - S_star)
        E_starR = UR[Ns + 1] / rhoR + (S_star - uR) * (S_star + pR / (rhoR * (SR - uR)))
        rhoE_L_int = UL[Ns + 1]   # left cell uses its own rhoE (Abgrall & Karni)
        rhoE_R_int = UR[Ns + 1]   # right cell uses its own rhoE (Abgrall & Karni)
        p_int = pR + rhoR * (SR - uR) * (S_star - uR)
        u_int = S_star
        # Star state for species/momentum flux uses standard HLLC star energy
        U_starR = np.empty(Ns + 2)
        for i in range(Ns):
            U_starR[i] = coeff_R * YiR[i]
        U_starR[Ns] = coeff_R * S_star
        U_starR[Ns + 1] = coeff_R * E_starR  # standard HLLC star energy for species/mom flux
        F_species_mom = FR + SR * (U_starR - UR)

    # --- Double-energy fluxes ---
    # Each side uses its own (rhoE_int + p_int) * u_int expression.
    # For subsonic regimes the rhoE_int is the same HLLC star value, but
    # the pressure p_int is computed from the respective side's EOS so that
    # the energy flux is consistent with that side's thermodynamics.
    F_E_for_left  = (rhoE_L_int + p_int) * u_int
    F_E_for_right = (rhoE_R_int + p_int) * u_int

    F_left  = F_species_mom.copy()
    F_right = F_species_mom.copy()
    F_left[Ns + 1]  = F_E_for_left
    F_right[Ns + 1] = F_E_for_right

    return F_left, F_right, p_int


# ============================================================================
# API compatibility: ausm_plus_up_flux delegates to hllc_flux
# ============================================================================

def ausm_plus_up_flux(
    UL: np.ndarray,
    UR: np.ndarray,
    eos_list: list,
    T_guess_L: float = 300.0,
    T_guess_R: float = 300.0,
    **kwargs,
) -> np.ndarray:
    """AUSM+up API wrapper — delegates to hllc_flux for multi-material flows."""
    return hllc_flux(UL, UR, eos_list, T_guess_L, T_guess_R)
