# Ref: CLAUDE.md § Flux 기법: APEC
# Ref: docs/APEC_flux.md — APEC Half-point 값, APEC 수치 플럭스 전체, Appendix A
"""
APEC (Approximate Pressure-Equilibrium-preserving with Conservation) flux
with LLF (Local Lax-Friedrichs) upwind dissipation for stability.

Ref: Terashima, Ly, Ihme, J. Comput. Phys. 524 (2025) 113701

The APEC flux resolves the Abgrall problem by adding a PE-correction term
to the internal energy half-point value, reducing PE error by up to 4x
compared to the standard FC-NPE split-form.

Implementation strategy (following Appendix A of docs/APEC_flux.md):
  - F_rho, F_rhou, F_rhoYi : LLF (Local Lax-Friedrichs) upwind base flux
  - F_rhoE : APEC energy correction applied on top of LLF base flux

LLF base flux:
    F_LLF = 0.5*(FL + FR) - 0.5*lambda_max*(UR - UL)
    lambda_max = max(|u_L| + c_L, |u_R| + c_R)

APEC energy correction (Appendix A, docs/APEC_flux.md):
    F_rhoE(APEC) = (F_rhoE_L + F_rhoE_R)/2
                 + 0.5 * sum_i (eps_i - u^2/2)|_L * (F_rhoYi_half - F_rhoYi_L)
                 + 0.5 * u_L                       * (F_rhou_half   - F_rhou_L)
                 - 0.5 * sum_i (eps_i - u^2/2)|_R * (F_rhoYi_R     - F_rhoYi_half)
                 - 0.5 * u_R                       * (F_rhou_R      - F_rhou_half)

where F_rhoYi_half and F_rhou_half are the LLF interface fluxes.

Conservative state vector (1D):
    U = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
"""

from __future__ import annotations

import numpy as np
from math import sqrt
from typing import List

from .eos.ideal import IdealGasEOS
from .eos.nasg import NASGEOS
from .eos.srk import SRKEOS
from .utils import (
    cons_to_prim,
    cell_epsilon_i,
    mixture_sound_speed,
)

EOSType = IdealGasEOS | NASGEOS | SRKEOS


def _compute_sound_speed(
    W: np.ndarray,
    rho: float,
    eos_list: List[EOSType],
) -> float:
    """
    Compute mixture speed of sound from primitive state W and mixture density rho.

    Uses Wood's formula via mixture_sound_speed().
    Falls back to gamma_mix * p / rho approximation if Wood's formula fails.

    Parameters
    ----------
    W : primitive state [p, u, T, Y_1, ..., Y_{N-1}]
    rho : mixture density
    eos_list : list of N EOS objects

    Returns
    -------
    c : float, sound speed >= 1e-10
    """
    N = len(eos_list)
    p = W[0]
    T = W[2]
    Y = np.empty(N)
    Y[:N-1] = W[3:3+N-1]
    Y[N-1] = max(0.0, 1.0 - float(np.sum(Y[:N-1])))

    try:
        c = mixture_sound_speed(rho, Y, T, p, eos_list,
                                rho_cv_mix=0.0, dp_dT_mix=0.0)
        if c > 0.0 and np.isfinite(c):
            return c
    except Exception:
        pass

    # Fallback: gamma_mix * p / rho approximation
    c2 = 0.0
    for i, eos in enumerate(eos_list):
        gamma_i = getattr(eos, 'gamma', 1.4)
        Yi = Y[i]
        c2 += Yi * gamma_i * p / max(rho, 1e-30)
    return max(sqrt(max(c2, 0.0)), 1e-10)


def apec_flux(
    UL: np.ndarray,
    UR: np.ndarray,
    eos_list: List[EOSType],
    T_guess_L: float = 300.0,
    T_guess_R: float = 300.0,
) -> np.ndarray:
    """
    Compute APEC interface flux with LLF dissipation for stability.

    Strategy (docs/APEC_flux.md Appendix A):
      - Base flux for mass, momentum, species: LLF upwind
      - Energy flux: APEC correction applied on LLF base

    Ref: docs/APEC_flux.md — APEC Half-point 값 (핵심 수식)
    Ref: docs/APEC_flux.md — APEC 수치 플럭스 전체 (1D split-form)
    Ref: docs/APEC_flux.md — upwind 기법으로 확장 시 (Appendix A)

    Parameters
    ----------
    UL : np.ndarray, shape (2 + N_species,)
        Conservative state on the left of the interface.
    UR : np.ndarray, shape (2 + N_species,)
        Conservative state on the right of the interface.
    eos_list : list of N EOS objects (one per species).
    T_guess_L, T_guess_R : float
        Temperature initial guesses for primitive recovery.

    Returns
    -------
    F : np.ndarray, shape (2 + N_species,)
        APEC-LLF numerical flux [F_rho, F_rhou, F_rhoE, F_rhoY_1, ..., F_rhoY_{N-1}]
    """
    N = len(eos_list)

    # ------------------------------------------------------------------
    # 1. Extract primitive variables from conservative states
    # ------------------------------------------------------------------
    WL = cons_to_prim(UL, eos_list, T_guess=T_guess_L)
    WR = cons_to_prim(UR, eos_list, T_guess=T_guess_R)

    pL, uL, TL = WL[0], WL[1], WL[2]
    pR, uR, TR = WR[0], WR[1], WR[2]
    YL = np.empty(N)
    YR = np.empty(N)
    YL[:N-1] = WL[3:3+N-1]
    YR[:N-1] = WR[3:3+N-1]
    YL[N-1] = 1.0 - np.sum(YL[:N-1])
    YR[N-1] = 1.0 - np.sum(YR[:N-1])
    YL = np.clip(YL, 0.0, 1.0)
    YR = np.clip(YR, 0.0, 1.0)

    rhoL = UL[0]
    rhoR = UR[0]

    # ------------------------------------------------------------------
    # 2. Compute rho*Y_i (partial densities) for each species
    # ------------------------------------------------------------------
    rhoYiL = np.empty(N)
    rhoYiR = np.empty(N)
    rhoYiL[:N-1] = UL[3:3+N-1]
    rhoYiR[:N-1] = UR[3:3+N-1]
    rhoYiL[N-1] = rhoL - np.sum(rhoYiL[:N-1])
    rhoYiR[N-1] = rhoR - np.sum(rhoYiR[:N-1])
    rhoYiL = np.clip(rhoYiL, 0.0, rhoL)
    rhoYiR = np.clip(rhoYiR, 0.0, rhoR)

    # ------------------------------------------------------------------
    # 3. Physical fluxes at left and right cells
    #    FL = [rho*u, rho*u^2+p, (rho*E+p)*u, rho*Yi*u]
    # ------------------------------------------------------------------
    FL = np.empty(2 + N)
    FL[0] = rhoL * uL
    FL[1] = rhoL * uL ** 2 + pL
    FL[2] = (UL[2] + pL) * uL
    for k in range(N - 1):
        FL[3 + k] = rhoYiL[k] * uL

    FR = np.empty(2 + N)
    FR[0] = rhoR * uR
    FR[1] = rhoR * uR ** 2 + pR
    FR[2] = (UR[2] + pR) * uR
    for k in range(N - 1):
        FR[3 + k] = rhoYiR[k] * uR

    # ------------------------------------------------------------------
    # 4. Compute wave speeds for LLF dissipation
    #    lambda_max = max(|u_L| + c_L, |u_R| + c_R)
    # ------------------------------------------------------------------
    c_L = _compute_sound_speed(WL, rhoL, eos_list)
    c_R = _compute_sound_speed(WR, rhoR, eos_list)
    lambda_max = max(abs(uL) + c_L, abs(uR) + c_R)

    # ------------------------------------------------------------------
    # 5. LLF base flux for all components
    #    F_LLF = 0.5*(FL + FR) - 0.5*lambda_max*(UR - UL)
    # ------------------------------------------------------------------
    F_LLF = 0.5 * (FL + FR) - 0.5 * lambda_max * (UR - UL)

    # ------------------------------------------------------------------
    # 6. APEC energy correction (Appendix A, docs/APEC_flux.md)
    #
    # The LLF base flux is used for mass/momentum/species.
    # The energy flux gets the APEC correction:
    #
    #   F_rhoE(APEC) = (F_rhoE_L + F_rhoE_R)/2
    #     + 0.5*sum_i(eps_i - u^2/2)|_L * (F_rhoYi_half - F_rhoYi_L)
    #     + 0.5*u_L * (F_rhou_half - F_rhou_L)
    #     - 0.5*sum_i(eps_i - u^2/2)|_R * (F_rhoYi_R - F_rhoYi_half)
    #     - 0.5*u_R * (F_rhou_R - F_rhou_half)
    #
    # where F_rhoYi_half and F_rhou_half are the LLF interface fluxes.
    # ------------------------------------------------------------------

    # Compute εᵢ at left and right cells
    # Ref: docs/APEC_flux.md εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}
    eps_L = cell_epsilon_i(rhoL, YL, TL, pL, eos_list)
    eps_R = cell_epsilon_i(rhoR, YR, TR, pR, eos_list)

    # LLF interface flux for species (F_rhoYi_half) and momentum (F_rhou_half)
    F_rhoYi_half = F_LLF[3:3+N-1]   # shape (N-1,), indices 3 to 3+N-2
    F_rhou_half  = F_LLF[1]

    # Compute APEC energy correction terms
    # Left contribution: sum_i (eps_i - u^2/2)|_L * (F_rhoYi_half_i - F_rhoYi_L_i)
    u2half_L = 0.5 * uL ** 2
    u2half_R = 0.5 * uR ** 2

    # Build species flux arrays (N-1 components only, excluding last species)
    # Note: APEC correction uses N-1 independent species (last is determined)
    delta_rhoYi_left  = np.empty(N - 1)
    delta_rhoYi_right = np.empty(N - 1)
    for k in range(N - 1):
        delta_rhoYi_left[k]  = F_rhoYi_half[k] - FL[3 + k]
        delta_rhoYi_right[k] = FR[3 + k] - F_rhoYi_half[k]

    # Species correction using (eps_i - u^2/2) for the N-1 tracked species
    species_corr_left  = np.dot(eps_L[:N-1] - u2half_L, delta_rhoYi_left)
    species_corr_right = np.dot(eps_R[:N-1] - u2half_R, delta_rhoYi_right)

    # Momentum correction: u * (F_rhou_half - F_rhou_cell)
    mom_corr_left  = uL * (F_rhou_half - FL[1])
    mom_corr_right = uR * (FR[1] - F_rhou_half)

    # APEC energy flux
    F_rhoE_apec = (0.5 * (FL[2] + FR[2])
                   + 0.5 * species_corr_left
                   + 0.5 * mom_corr_left
                   - 0.5 * species_corr_right
                   - 0.5 * mom_corr_right)

    # ------------------------------------------------------------------
    # 7. Assemble final flux vector
    #    Mass, momentum, species: LLF base flux
    #    Energy: APEC-corrected
    # ------------------------------------------------------------------
    F = np.empty(2 + N)
    F[0] = F_LLF[0]       # F_rho  (LLF)
    F[1] = F_LLF[1]       # F_rhou (LLF)
    F[2] = F_rhoE_apec    # F_rhoE (APEC-corrected)
    for k in range(N - 1):
        F[3 + k] = F_LLF[3 + k]  # F_rhoYi (LLF)

    return F


def physical_flux(U: np.ndarray, eos_list: List[EOSType], T_guess: float = 300.0) -> np.ndarray:
    """
    Compute the physical (Euler) flux at a single cell:
        F = [rho*u, rho*u^2 + p, (rho*E + p)*u, rho*Y_k*u]

    Used for boundary conditions and Jacobian computation.

    Parameters
    ----------
    U : np.ndarray, shape (3 + N_species - 1,)
    eos_list : list of N EOS objects.
    T_guess : float

    Returns
    -------
    F : np.ndarray, shape (3 + N_species - 1,)
    """
    N = len(eos_list)
    W = cons_to_prim(U, eos_list, T_guess=T_guess)
    p = W[0]
    u = W[1]

    rho = U[0]
    rho_u = U[1]
    rho_E = U[2]

    F = np.empty(3 + N - 1)
    F[0] = rho * u
    F[1] = rho * u ** 2 + p
    F[2] = (rho_E + p) * u
    for k in range(N - 1):
        F[3 + k] = U[3 + k] * u

    return F
