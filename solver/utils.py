# Ref: CLAUDE.md § Primitive & Conservative Variables, § 혼합 물성치
# Ref: docs/APEC_flux.md εᵢ 계산
"""
Utility functions for conservative <-> primitive variable conversion
and mixture thermodynamic properties.

Conservative variables (1D):
    U = [rho, rho*u, rho*E, rho*Y_1, rho*Y_2, ..., rho*Y_{N-1}]

Primitive variables (1D):
    W = [p, u, T, Y_1, Y_2, ..., Y_{N-1}]

Mixture rules:
    1/rho = sum_i (Y_i / rho_i)      (volume-fraction mixing)
    e     = sum_i (Y_i * e_i(T, p))  (mass-fraction mixing)
    E     = e + u^2/2                 (specific total energy)

N species total: Y_N = 1 - sum_{k=1}^{N-1} Y_k
"""

from __future__ import annotations

import numpy as np
from typing import List, Sequence
from scipy.optimize import brentq

from .eos.ideal import IdealGasEOS
from .eos.nasg import NASGEOS
from .eos.srk import SRKEOS


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
EOSType = IdealGasEOS | NASGEOS | SRKEOS


# ---------------------------------------------------------------------------
# Mixture thermodynamic helpers (per-cell, scalar arguments)
# ---------------------------------------------------------------------------

def mixture_density(Y: np.ndarray, eos_list: List[EOSType], T: float, p: float) -> float:
    """
    Compute mixture density from species mass fractions using volume-fraction mixing rule:
        1/rho = sum_i (Y_i / rho_i(T, p))

    Parameters
    ----------
    Y : array of shape (N,)
        Species mass fractions (must sum to 1).
    eos_list : list of N EOS objects
        One EOS per species.
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].

    Returns
    -------
    rho : float
        Mixture density [kg/m^3].
    """
    inv_rho = 0.0
    for Yi, eos in zip(Y, eos_list):
        rho_i = _rho_from_T_p(eos, T, p)
        inv_rho += Yi / rho_i
    return 1.0 / inv_rho


def mixture_internal_energy(Y: np.ndarray, eos_list: List[EOSType], T: float, p: float) -> float:
    """
    Compute mixture specific internal energy:
        e = sum_i (Y_i * e_i(T, p))

    Parameters
    ----------
    Y : array of shape (N,)
        Species mass fractions.
    eos_list : list of N EOS objects.
    T : float
        Temperature [K].
    p : float
        Pressure [Pa].

    Returns
    -------
    e : float
        Mixture specific internal energy [J/kg].
    """
    e = 0.0
    for Yi, eos in zip(Y, eos_list):
        rho_i = _rho_from_T_p(eos, T, p)
        e += Yi * _internal_energy(eos, rho_i, T)
    return e


def mixture_rho_cv(
    Y: np.ndarray, rho: float, eos_list: List[EOSType], T: float, p: float
) -> float:
    """
    Compute volumetric heat capacity of mixture: rho * c_v_mix.

    rho * c_v_mix = sum_i rho_i * c_v_i
                  = rho * sum_i Y_i * c_v_i

    For SRK, c_v is the real-fluid isochoric heat capacity.
    """
    rho_cv = 0.0
    for Yi, eos in zip(Y, eos_list):
        rho_i = _rho_from_T_p(eos, T, p)
        cv_i = _cv(eos, rho_i, T)
        rho_cv += Yi * rho * cv_i
    return rho_cv


def mixture_dp_dT(
    Y: np.ndarray, rho: float, eos_list: List[EOSType], T: float, p: float
) -> float:
    """
    Compute (dp/dT)_rho for the mixture using Dalton's law approximation:
        (dp/dT)_mix = sum_i Y_i * rho * (dp/dT)_i / rho_i

    For a mixture of species each occupying their own specific volume at T, p:
        p = p_mix, but each species has its own (dp/dT)_{rho_i}.
    We use the partial-pressure weighted form.
    """
    dp_dT = 0.0
    for Yi, eos in zip(Y, eos_list):
        rho_i = _rho_from_T_p(eos, T, p)
        dp_dT += Yi * rho * _dp_dT(eos, rho_i, T)
    return dp_dT


def compute_epsilon_i(
    eos: EOSType, rho: float, rho_i: float, T: float,
    rho_cv_mix: float, dp_dT_mix: float
) -> float:
    """
    Compute εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p} for species i.

    εᵢ = (∂ρe/∂ρᵢ)_T - (ρ Cᵥ_mix / (∂p/∂T)_mix) * (∂p/∂ρᵢ)_T

    Ref: docs/APEC_flux.md εᵢ 계산
    """
    return eos.epsilon_i(rho, T, rho_cv_mix, dp_dT_mix)


# ---------------------------------------------------------------------------
# Private helpers: dispatch by EOS type
# ---------------------------------------------------------------------------

def _rho_from_T_p(eos: EOSType, T: float, p: float) -> float:
    """Return species density given T and p."""
    if isinstance(eos, IdealGasEOS):
        return p / (eos.R_s * T)
    elif isinstance(eos, NASGEOS):
        # p = (gamma-1)*c_v*rho*T/(1-b*rho) - p_inf
        # Solve for rho: linear in rho/(1-b*rho)
        # rho*(1-b*rho) denominator  =>  rho = (p + p_inf) / ((gamma-1)*c_v*T/(1-b*rho))
        # Use direct inversion:
        # (p + p_inf)(1 - b*rho) = (gamma-1)*c_v*rho*T
        # p + p_inf - b*(p+p_inf)*rho = (gamma-1)*c_v*T*rho
        # p + p_inf = rho * [(gamma-1)*c_v*T + b*(p+p_inf)]
        num = p + eos.p_inf
        den = (eos.gamma - 1.0) * eos.c_v * T + eos.b * (p + eos.p_inf)
        return num / den
    elif isinstance(eos, SRKEOS):
        # Invert SRK EOS using brentq
        def residual(rho_val):
            return eos.pressure(rho_val, T) - p
        try:
            rho = brentq(residual, 1e-6, 5000.0, xtol=1e-10, maxiter=300)
        except ValueError:
            rho = brentq(residual, 1e-6, 20000.0, xtol=1e-10, maxiter=500)
        return rho
    else:
        raise TypeError(f"Unknown EOS type: {type(eos)}")


def _internal_energy(eos: EOSType, rho: float, T: float) -> float:
    """Return species specific internal energy e_i(rho, T)."""
    if isinstance(eos, IdealGasEOS):
        return eos.internal_energy(T)
    elif isinstance(eos, NASGEOS):
        return eos.internal_energy(rho, T)
    elif isinstance(eos, SRKEOS):
        return eos.internal_energy(rho, T)
    else:
        raise TypeError(f"Unknown EOS type: {type(eos)}")


def _pressure_from_rho_T(eos: EOSType, rho: float, T: float) -> float:
    """Return pressure given density and temperature."""
    return eos.pressure(rho, T)


def _cv(eos: EOSType, rho: float, T: float) -> float:
    """Return species isochoric specific heat c_v_i."""
    if isinstance(eos, IdealGasEOS):
        return eos.c_v
    elif isinstance(eos, NASGEOS):
        return eos.c_v
    elif isinstance(eos, SRKEOS):
        return eos.cv_real(rho, T)
    else:
        raise TypeError(f"Unknown EOS type: {type(eos)}")


def _dp_dT(eos: EOSType, rho: float, T: float) -> float:
    """Return (dp/dT)_rho for species i."""
    if isinstance(eos, IdealGasEOS):
        return eos.dp_dT(rho)
    elif isinstance(eos, NASGEOS):
        return eos.dp_dT(rho)
    elif isinstance(eos, SRKEOS):
        return eos.dp_dT_v(rho, T)
    else:
        raise TypeError(f"Unknown EOS type: {type(eos)}")


def _sound_speed(eos: EOSType, rho: float, T: float) -> float:
    """Return speed of sound for species i."""
    return eos.sound_speed(rho, T)


# ---------------------------------------------------------------------------
# Temperature recovery from (rho, rho*Y_i, rho*e)
# ---------------------------------------------------------------------------

def temperature_from_rho_rhoYi_rhoe(
    rho: float,
    rhoYi: np.ndarray,
    rhoe: float,
    eos_list: List[EOSType],
    T_guess: float = 300.0,
) -> float:
    """
    Recover temperature T given:
        rho  = total mixture density
        rhoYi = array of partial densities [rho*Y_1, ..., rho*Y_N]
        rhoe  = rho * e  (specific internal energy * density)

    Solves: sum_i (rhoYi_i / rho) * e_i(rhoYi_i/rho, T) * rho = rhoe
    i.e.:   sum_i rhoYi_i * e_i(rhoYi_i/rho, T) = rhoe

    For IdealGas: e_i = c_v_i * T  => linear, direct solve
    For NASG: e_i = c_v*T + q + p_inf*(1-b*rho_i)/((gamma-1)*rho_i) => linear, direct solve
    For SRK (or mixed with SRK): non-linear, use brentq with dynamic bracketing
    """
    N = len(eos_list)
    Yi = rhoYi / rho

    # Check if all EOS are linear in T (Ideal Gas and/or NASG only)
    has_srk = any(isinstance(eos, SRKEOS) for eos in eos_list)

    if not has_srk:
        # All EOS are IdealGas or NASG — T appears linearly in e_i.
        # e_i(T) = c_v_i * T + offset_i
        # rhoe = rho * sum_i Y_i * e_i(T) = rho * (T * cv_mix + offset_mix)
        # T = (rhoe/rho - offset_mix) / cv_mix
        cv_mix = 0.0
        offset_mix = 0.0
        for i, eos in enumerate(eos_list):
            Yi_i = Yi[i]
            if Yi_i < 1e-30:
                continue
            if isinstance(eos, IdealGasEOS):
                # e_i = c_v * T  =>  offset = 0
                cv_mix += Yi_i * eos.c_v
                # offset_i = 0
            elif isinstance(eos, NASGEOS):
                # e_i = c_v * T + q + p_inf*(1 - b*rho_i)/((gamma-1)*rho_i)
                # Use rhoYi[i] as proxy for species partial density rho_i
                rho_i_proxy = rhoYi[i]
                if rho_i_proxy < 1e-30:
                    rho_i_proxy = 1e-30
                cv_mix += Yi_i * eos.c_v
                offset_i = eos.q + eos.p_inf * (1.0 - eos.b * rho_i_proxy) / (
                    (eos.gamma - 1.0) * rho_i_proxy
                )
                offset_mix += Yi_i * offset_i
        if cv_mix < 1e-30:
            # Degenerate: fallback to T_guess
            return T_guess
        e_specific = rhoe / rho
        T = (e_specific - offset_mix) / cv_mix
        # Guard against unphysical result
        if not np.isfinite(T) or T <= 0.0:
            # Fallback: ignore offsets and use pure cv estimate
            cv_rough = sum(
                Yi[i] * getattr(eos_list[i], 'c_v', 1000.0)
                for i in range(N)
            )
            T = max(1e-30, abs(rhoe / rho) / max(cv_rough, 1e-30))
        return T

    # SRK present — use brentq with dynamic bracketing
    def residual(T_val):
        e_mix = 0.0
        for i, eos in enumerate(eos_list):
            e_i = _internal_energy(eos, rhoYi[i], T_val)
            e_mix += Yi[i] * e_i
        return rho * e_mix - rhoe

    # Estimate T from rough cv
    cv_rough = 0.0
    for i, eos in enumerate(eos_list):
        if isinstance(eos, SRKEOS):
            cv_i = eos.c_v0
        else:
            cv_i = getattr(eos, 'c_v', 1000.0)
        cv_rough += Yi[i] * cv_i
    cv_rough = max(cv_rough, 1e-30)

    # Use |rhoe/rho| as energy magnitude for T estimate; rhoe could be negative
    # for NASG offsets, so use absolute value as a rough scale
    e_specific = rhoe / rho
    T_est = max(1e-10, abs(e_specific) / cv_rough)
    if T_est < 1e-10:
        T_est = max(1e-10, T_guess)

    T_lo = T_est * 1e-3
    T_hi = T_est * 1e3 + 1.0   # ensure T_hi > T_lo even if T_est is tiny
    T_lo = max(T_lo, 1e-12)

    # Expand bracket until root is bracketed
    for _ in range(200):
        try:
            r_lo = residual(T_lo)
            r_hi = residual(T_hi)
            if r_lo * r_hi < 0.0:
                break
        except Exception:
            pass
        T_lo = max(T_lo * 0.1, 1e-12)
        T_hi = T_hi * 10.0

    T = brentq(residual, T_lo, T_hi, xtol=1e-10, rtol=1e-12, maxiter=1000)
    return T


def pressure_from_rho_T(
    rho: float,
    Y: np.ndarray,
    T: float,
    eos_list: List[EOSType],
) -> float:
    """
    Compute mixture pressure from mixture density, species fractions, and temperature.

    Uses Dalton's law / partial-pressure approach:
        p = p_mix(T, rho_i_species)
    where rho_i_species is recovered from volume-fraction mixing rule.

    In practice, for a given (rho, Y, T), each species has its own EOS
    giving p_i(rho_i_species, T). The mixture pressure is uniquely determined
    by requiring mechanical equilibrium (p_i = p for all i), which is satisfied
    by the mixing rule 1/rho = sum Y_i/rho_i at constant T, p.

    This function solves for p such that the volume-fraction mixing rule is satisfied.
    """
    # Fast path: if all EOS are Ideal Gas, p is directly solvable analytically.
    # Mixing rule: 1/rho = sum_i Y_i / rho_i = sum_i Y_i * R_s_i * T / p
    #   => 1/rho = (T/p) * sum_i Y_i * R_s_i
    #   => p = rho * T * sum_i Y_i * R_s_i
    if all(isinstance(eos, IdealGasEOS) for eos in eos_list):
        R_mix = sum(Y[i] * eos_list[i].R_s for i in range(len(eos_list)))
        return rho * R_mix * T

    # For NASG-only or Ideal+NASG mixtures: no SRK, use pressure direct solve
    # The mixing rule 1/rho = sum Y_i/rho_i(T,p) still requires brentq for NASG
    # because rho_i(T,p) depends on p nonlinearly (via 1-b*rho_i denominator).
    # However, for Ideal-only handled above. For NASG, use brentq with dynamic bracket.

    # Estimate pressure from rough ideal-gas approximation as bracket center
    R_mix_approx = 0.0
    for i, eos in enumerate(eos_list):
        if isinstance(eos, IdealGasEOS):
            R_mix_approx += Y[i] * eos.R_s
        elif isinstance(eos, NASGEOS):
            R_mix_approx += Y[i] * (eos.gamma - 1.0) * eos.c_v
        elif isinstance(eos, SRKEOS):
            R_mix_approx += Y[i] * eos.R_s
    R_mix_approx = max(R_mix_approx, 1e-30)
    p_est = rho * R_mix_approx * T
    p_est = max(p_est, 1e-30)

    p_lo = p_est * 1e-4
    p_hi = p_est * 1e4 + 1.0

    def residual(p_val):
        inv_rho = sum(Y[i] / _rho_from_T_p(eos, T, p_val)
                      for i, eos in enumerate(eos_list))
        return 1.0 / inv_rho - rho

    # Expand bracket if needed
    for _ in range(100):
        try:
            r_lo = residual(max(p_lo, 1e-30))
            r_hi = residual(p_hi)
            if r_lo * r_hi < 0.0:
                break
        except Exception:
            pass
        p_lo = max(p_lo * 0.1, 1e-30)
        p_hi = p_hi * 10.0

    p = brentq(residual, max(p_lo, 1e-30), p_hi, xtol=1e-10, rtol=1e-10, maxiter=1000)
    return p


def mixture_sound_speed(
    rho: float,
    Y: np.ndarray,
    T: float,
    p: float,
    eos_list: List[EOSType],
    rho_cv_mix: float,
    dp_dT_mix: float,
) -> float:
    """
    Mixture speed of sound via Wood's formula generalization:

    a^2 = -(1/rho^2) * (dp/dv)_s

    Using thermodynamic identity:
    a^2 = (dp/drho)_T + T*(dp/dT)^2_rho / (rho * c_v)
         (where derivatives are mixture quantities)

    For a mixture:
        (dp/drho)_mix = sum_i Y_i^2 * rho * dp_i/drho_i_T / rho_i^2
                        (approximate from mixing rule)

    We use a simplified but consistent approach:
    Compute (dp/drho)_T from the implicit mixing rule derivative.

    Alternatively, use the individual species sound speeds (Wood's formula):
        1/(rho*a^2) = sum_i Y_i / (rho_i * a_i^2)
    combined with the thermal correction.

    Here we use the per-species a_i and Wood's formula as first approximation.
    """
    # Wood's formula: 1/(rho*a^2) = sum_i (Y_i/(rho_i * a_i^2))
    inv_rhoa2 = 0.0
    for i, eos in enumerate(eos_list):
        rho_i = _rho_from_T_p(eos, T, p)
        a_i = _sound_speed(eos, rho_i, T)
        if a_i > 0.0:
            inv_rhoa2 += Y[i] / (rho_i * a_i ** 2)
    if inv_rhoa2 > 0.0:
        a2 = 1.0 / (rho * inv_rhoa2)
    else:
        a2 = 0.0
    return np.sqrt(max(a2, 0.0))


# ---------------------------------------------------------------------------
# Conservative <-> Primitive variable conversion
# ---------------------------------------------------------------------------

def prim_to_cons(
    W: np.ndarray,
    eos_list: List[EOSType],
) -> np.ndarray:
    """
    Convert primitive variables to conservative variables (1D).

    Primitive: W = [p, u, T, Y_1, Y_2, ..., Y_{N-1}]
    Conservative: U = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]

    The last species fraction Y_N = 1 - sum_{k=1}^{N-1} Y_k is handled internally.

    Parameters
    ----------
    W : np.ndarray, shape (3 + N_species - 1,) = (2 + N_species,)
        [p, u, T, Y_1, ..., Y_{N-1}]
    eos_list : list of N EOS objects.

    Returns
    -------
    U : np.ndarray, shape (3 + N_species - 1,)
        [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
    """
    N = len(eos_list)
    p = W[0]
    u = W[1]
    T = W[2]
    Y = np.empty(N)
    Y[:N-1] = W[3:3+N-1]
    Y[N-1] = 1.0 - np.sum(Y[:N-1])
    Y = np.clip(Y, 0.0, 1.0)

    # Mixture density from volume-fraction mixing rule
    rho = mixture_density(Y, eos_list, T, p)

    # Mixture specific internal energy
    e = mixture_internal_energy(Y, eos_list, T, p)

    # Total specific energy
    E = e + 0.5 * u ** 2

    U = np.empty(2 + N)
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * E
    for k in range(N - 1):
        U[3 + k] = rho * Y[k]

    return U


def cons_to_prim(
    U: np.ndarray,
    eos_list: List[EOSType],
    T_guess: float = 300.0,
) -> np.ndarray:
    """
    Convert conservative variables to primitive variables (1D).

    Conservative: U = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
    Primitive: W = [p, u, T, Y_1, ..., Y_{N-1}]

    Parameters
    ----------
    U : np.ndarray, shape (3 + N_species - 1,)
        [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
    eos_list : list of N EOS objects.
    T_guess : float
        Initial guess for temperature [K].

    Returns
    -------
    W : np.ndarray, shape (3 + N_species - 1,)
        [p, u, T, Y_1, ..., Y_{N-1}]
    """
    N = len(eos_list)

    # Ensure U is a 1-D numpy array (robust to list, 2D slice, etc.)
    U = np.asarray(U, dtype=float).ravel()

    # Validate U shape: must have exactly 2 + N elements
    # U = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
    n_expected = 2 + N
    if U.shape[0] < n_expected:
        raise ValueError(
            f"cons_to_prim: U has {U.shape[0]} elements, expected {n_expected} "
            f"for {N} species (U=[rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{{N-1}}])"
        )

    rho = U[0]
    rho_u = U[1]
    rho_E = U[2]

    if rho <= 0.0:
        raise ValueError(f"cons_to_prim: non-positive density rho={rho}")

    u = rho_u / rho
    E = rho_E / rho
    e = E - 0.5 * u ** 2

    # Defend against non-positive internal energy (can occur during TVD-RK3 stages)
    if e <= 0.0:
        # Use a very small positive value to allow T recovery attempt
        # This prevents brentq from receiving a trivially negative bracket
        e = max(e, 1e-30)

    # Species mass fractions — use explicit element assignment for robustness
    rhoYi = np.empty(N)
    for k in range(N - 1):
        rhoYi[k] = U[3 + k]
    rhoYi[N - 1] = rho - np.sum(rhoYi[:N - 1])
    rhoYi = np.clip(rhoYi, 0.0, rho)
    Y = rhoYi / rho

    # Compute dynamic T_guess from rough cv estimate if T_guess is default 300 K
    # This handles dimensionless problems where T can be << 1 K
    has_srk = any(isinstance(eos, SRKEOS) for eos in eos_list)
    if not has_srk:
        # For Ideal/NASG: T is directly computable; T_guess is not needed.
        # Pass T_guess through for the fallback path only.
        cv_rough = 0.0
        for i, eos in enumerate(eos_list):
            cv_rough += Y[i] * getattr(eos, 'c_v', 1000.0)
        if cv_rough > 1e-30:
            T_guess_dyn = max(1e-30, e / max(cv_rough, 1e-30))
        else:
            T_guess_dyn = T_guess
    else:
        cv_rough = 0.0
        for i, eos in enumerate(eos_list):
            if isinstance(eos, SRKEOS):
                cv_rough += Y[i] * eos.c_v0
            else:
                cv_rough += Y[i] * getattr(eos, 'c_v', 1000.0)
        if cv_rough > 1e-30:
            T_guess_dyn = max(1e-30, e / max(cv_rough, 1e-30))
        else:
            T_guess_dyn = T_guess

    # Recover temperature from (rho, rhoYi, rho*e)
    T = temperature_from_rho_rhoYi_rhoe(rho, rhoYi, rho * e, eos_list, T_guess_dyn)

    # Recover pressure using volume-fraction mixing rule
    p = pressure_from_rho_T(rho, Y, T, eos_list)

    W = np.empty(2 + N)
    W[0] = p
    W[1] = u
    W[2] = T
    W[3:3 + N - 1] = Y[:N - 1]

    return W


def cell_epsilon_i(
    rho: float,
    Y: np.ndarray,
    T: float,
    p: float,
    eos_list: List[EOSType],
) -> np.ndarray:
    """
    Compute εᵢ for all species at a given cell state.

    Returns array of shape (N,) with εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}.

    Ref: docs/APEC_flux.md εᵢ 계산
    """
    N = len(eos_list)
    rho_cv_mix = mixture_rho_cv(Y, rho, eos_list, T, p)
    dp_dT_mix = mixture_dp_dT(Y, rho, eos_list, T, p)

    eps = np.empty(N)
    for i, eos in enumerate(eos_list):
        eps[i] = eos.epsilon_i(rho, T, rho_cv_mix, dp_dT_mix)
    return eps
