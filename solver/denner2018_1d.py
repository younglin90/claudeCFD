"""
solver/denner2018_1d.py

1D Pressure-based compressible two-phase flow solver.

Reference
---------
Denner F., Xiao C.-N., van Wachem B.G.M.
"Pressure-based algorithm for compressible interfacial flows with
 acoustically-conservative interface discretisation."
 J. Comput. Phys. 367 (2018) 192–234.

Primary variables (cell-centred, collocated grid)
-------------------------------------------------
    p   : pressure                           [Pa]
    u   : velocity                           [m/s]
    T   : temperature                        [K]
    psi : volume fraction of phase 1         [-]

Phase EOS — Noble-Abel Stiffened Gas (NASG, b = 0 reduces to SGS)
    Parameters per phase: {'gam': γ, 'pinf': p∞, 'cv': cv, 'q': q, 'b': b}  (b optional, default 0)
    Density  : ρ_k  = (p + p∞_k) / [(γ_k-1) cv_k T + (p+p∞_k) b_k]
    Enthalpy : h_k  = γ_k cv_k T + q_k     (thermodynamically consistent; cp_k = γ_k cv_k)
    Sound sp.: a_k² = γ_k (p + p∞_k) / (ρ_k (1 - b_k ρ_k))

Mixture (equal-T isobaric closure)
    ρ   = ψ ρ₁ + (1-ψ) ρ₂
    χ   = ψ/(ρ₁a₁²) + (1-ψ)/(ρ₂a₂²)    [Wood's formula, χ = 1/(ρc²)]
    K   = (ρ₂a₂² - ρ₁a₁²) / (ρ₁a₁²(1-ψ) + ρ₂a₂²ψ)   [ACID compressibility coeff]

Governing equations (1D)
    ∂ρ/∂t  + ∂(ρu)/∂x                = 0          [continuity — implicit via pressure eq.]
    ∂(ρu)/∂t + ∂(ρu²+p)/∂x          = 0          [momentum]
    χ ∂p/∂t + ∂u/∂x                  = 0          [acoustic pressure equation]
    ∂ψ/∂t + ∂(ψu)/∂x + K ∂u/∂x     = 0          [color fn, ACID compressibility]
    ρ cp,mix DT/Dt                    = Dp/Dt      [temperature / enthalpy]

PISO loop (per time step, Denner §3)
    1. Compute mixture props (ρ, χ, K) from (p^n, T^n, ψ^n)
    2. Advect ψ  (explicit upwind + K compressibility correction)
    3. Momentum predictor u*  (explicit with ∇p^n)
    4. Compute ACID face densities ρ_f*  (Denner §5, Eq. 38)
    5. Assemble & solve pressure Helmholtz → p^{n+1}
    6. Correct cell velocity u^{n+1}
    7. Update temperature T^{n+1}  (explicit from material Dp/Dt)

ACID face density (§5)
    ρ_f* = ψ_f ρ₁(p_f, T_f) + (1-ψ_f) ρ₂(p_f, T_f)
    where ψ_f, p_f, T_f are face-centred values (simple average).
    This ensures acoustic conservation across sharp interfaces.

Time scheme (Appendix A, Eq. A.5)
    BDF2: (3φ^{n+1} - 4φ^n + φ^{n-1}) / (2Δt) = RHS
    First step uses BDF1 (Backward Euler).
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded
from typing import Dict, Any, Optional, List


# ─────────────────────────────────────────────────────────────────
# NASG EOS helpers  (vectorised over cell arrays)
# NASG with b=0 reduces to SGS (Stiffened Gas).
# ─────────────────────────────────────────────────────────────────

def _b(ph: dict) -> float:
    """Covolume b (default 0 for SGS)."""
    return ph.get('b', 0.0)


def _rho(p: np.ndarray, T: np.ndarray, ph: dict) -> np.ndarray:
    """NASG density: ρ_k = (p + p∞) / [(γ-1) cv T + (p+p∞) b]"""
    b   = _b(ph)
    num = p + ph['pinf']
    den = (ph['gam'] - 1.0) * ph['cv'] * np.maximum(T, 1e-10) + num * b
    return num / np.maximum(den, 1e-300)


def _cp(ph: dict) -> float:
    """cp_k = γ_k cv_k  (thermodynamically consistent for NASG)"""
    return ph['gam'] * ph['cv']


def _h(T: np.ndarray, ph: dict) -> np.ndarray:
    """h_k = γ_k cv_k T + q_k  (thermodynamically consistent enthalpy)"""
    return _cp(ph) * T + ph['q']


def _a2(p: np.ndarray, ph: dict, rho: np.ndarray) -> np.ndarray:
    """NASG speed of sound squared: a_k² = γ_k (p + p∞_k) / [ρ_k (1 - b_k ρ_k)]"""
    b    = _b(ph)
    bfac = np.maximum(1.0 - b * rho, 1e-10)
    return ph['gam'] * (p + ph['pinf']) / np.maximum(rho * bfac, 1e-300)


def _T_from_rho_p_single(rho_val: float, p_val: float, ph: dict) -> float:
    """Recover T from ρ_k, p using NASG: T = (p+p∞)(1-bρ) / [(γ-1)cv ρ]"""
    b = _b(ph)
    return (p_val + ph['pinf']) * (1.0 - b * rho_val) / max((ph['gam']-1.0)*ph['cv']*rho_val, 1e-300)


# ─────────────────────────────────────────────────────────────────
# Mixture properties
# ─────────────────────────────────────────────────────────────────

def _props(p: np.ndarray, T: np.ndarray, psi: np.ndarray,
           ph1: dict, ph2: dict):
    """
    Compute all per-phase and mixture properties.

    Returns
    -------
    rho1, h1, a12, rho2, h2, a22 : per-phase values
    rho, rhoh, chi, K             : mixture values
        chi = 1/(rho c²)  [Wood's formula]
        K   = ACID compressibility coefficient
    """
    rho1 = _rho(p, T, ph1)
    h1   = _h(T, ph1)
    a12  = _a2(p, ph1, rho1)

    rho2 = _rho(p, T, ph2)
    h2   = _h(T, ph2)
    a22  = _a2(p, ph2, rho2)

    rho  = psi * rho1 + (1.0 - psi) * rho2
    rhoh = psi * rho1 * h1 + (1.0 - psi) * rho2 * h2

    # Wood's formula: 1/(ρc²) = ψ/(ρ₁a₁²) + (1-ψ)/(ρ₂a₂²)
    chi = (psi / np.maximum(rho1 * a12, 1e-300)
           + (1.0 - psi) / np.maximum(rho2 * a22, 1e-300))

    # ACID compressibility coefficient (Denner Eq. 6)
    denom = rho1 * a12 * (1.0 - psi) + rho2 * a22 * psi
    K = (rho2 * a22 - rho1 * a12) / np.maximum(denom, 1e-300)

    return rho1, h1, a12, rho2, h2, a22, rho, rhoh, chi, K


def _T_from_rho(rho: np.ndarray, p: np.ndarray, psi: np.ndarray,
                ph1: dict, ph2: dict, T_guess: np.ndarray,
                n_iter: int = 10) -> np.ndarray:
    """
    Recover T from (ρ, p, ψ) using Newton iteration.

    Residual: f(T) = ψ ρ₁(p,T) + (1-ψ) ρ₂(p,T) - ρ = 0

    For SGS (b=0), exact solution:
        ρ = G(p,ψ)/T  →  T = G/ρ
        G = ψ(p+p∞₁)/[(γ₁-1)cv₁] + (1-ψ)(p+p∞₂)/[(γ₂-1)cv₂]

    For NASG (b≠0), solve by Newton.
    """
    b1 = _b(ph1)
    b2 = _b(ph2)

    if b1 == 0.0 and b2 == 0.0:
        # Exact SGS formula
        G = (psi * (p + ph1['pinf']) / ((ph1['gam'] - 1.0) * ph1['cv'])
             + (1.0 - psi) * (p + ph2['pinf']) / ((ph2['gam'] - 1.0) * ph2['cv']))
        return G / np.maximum(rho, 1e-300)

    # Newton iteration for NASG
    T = np.maximum(T_guess.copy(), 1e-3)
    for _ in range(n_iter):
        rho1 = _rho(p, T, ph1)
        rho2 = _rho(p, T, ph2)
        f    = psi * rho1 + (1.0 - psi) * rho2 - rho

        # df/dT: ∂ρ_k/∂T = -ρ_k(1-b_k ρ_k) / [(γ_k-1)cv_k T/(p+p∞_k)*T * ...]
        # Simpler: use numerical diff with step h
        h = 1.0
        r1p = _rho(p, T + h, ph1)
        r2p = _rho(p, T + h, ph2)
        dfdt = psi * (r1p - rho1) / h + (1.0 - psi) * (r2p - rho2) / h
        dT = -f / np.where(np.abs(dfdt) > 1e-20, dfdt, 1e-20)
        dT = np.clip(dT, -100.0, 100.0)
        T  = np.maximum(T + dT, 1e-3)
        if np.max(np.abs(dT)) < 1e-6:
            break
    return T


# ─────────────────────────────────────────────────────────────────
# Ghost-cell boundary conditions
# ─────────────────────────────────────────────────────────────────

def _ghost(arr: np.ndarray, bc_l: str, bc_r: str) -> np.ndarray:
    """Extend a 1-D cell array with one ghost cell on each side."""
    ext = np.empty(len(arr) + 2)
    ext[1:-1] = arr
    if bc_l == 'periodic':
        ext[0] = arr[-1]
    else:  # transmissive
        ext[0] = arr[0]
    if bc_r == 'periodic':
        ext[-1] = arr[0]
    else:
        ext[-1] = arr[-1]
    return ext


# ─────────────────────────────────────────────────────────────────
# Color function advection  (Denner Eq. 5)
# ─────────────────────────────────────────────────────────────────

def _advect_psi(psi: np.ndarray, u: np.ndarray, K: np.ndarray,
                dx: float, dt: float, bc_l: str, bc_r: str) -> np.ndarray:
    """
    Explicit first-order upwind advection of ψ with ACID compressibility.

        ∂ψ/∂t + ∂(uψ)/∂x + K ∂u/∂x = 0

    The K ∂u/∂x term corrects for different phase compressibilities
    so that pressure equilibrium is maintained across the interface.
    """
    psi_g = _ghost(psi, bc_l, bc_r)
    u_g   = _ghost(u,   bc_l, bc_r)

    # Face velocities (right face of cell m = face m+1/2), shape (N+1,)
    u_face = 0.5 * (u_g[:-1] + u_g[1:])

    # Upwind ψ at each face
    psi_face = np.where(u_face >= 0.0, psi_g[:-1], psi_g[1:])

    # Divergence of (uψ) and divergence of u at cell centres, shape (N,)
    div_upsi = (u_face[1:] * psi_face[1:] - u_face[:-1] * psi_face[:-1]) / dx
    div_u    = (u_face[1:] - u_face[:-1]) / dx

    return np.clip(psi - dt * (div_upsi + K * div_u), 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# ACID face density  (Denner §5, Eq. 38)
# ─────────────────────────────────────────────────────────────────

def _acid_face_rho(p: np.ndarray, T: np.ndarray, psi: np.ndarray,
                   ph1: dict, ph2: dict, bc_l: str, bc_r: str) -> np.ndarray:
    """
    ACID face density (Denner §5, Eq. 38):

        ρ_f* = ψ_f ρ₁(p_f, T_f) + (1-ψ_f) ρ₂(p_f, T_f)

    All face quantities are simple arithmetic averages of left/right cells.
    Using face pressure and temperature ensures EOS consistency at the interface,
    which is the key acoustic-conservation property of ACID.

    Returns
    -------
    rho_f : shape (N+1,) — face densities at interior+boundary faces
    """
    p_g   = _ghost(p,   bc_l, bc_r)
    T_g   = _ghost(T,   bc_l, bc_r)
    psi_g = _ghost(psi, bc_l, bc_r)

    p_f   = 0.5 * (p_g[:-1]   + p_g[1:])    # shape (N+1,)
    T_f   = 0.5 * (T_g[:-1]   + T_g[1:])
    psi_f = 0.5 * (psi_g[:-1] + psi_g[1:])

    T_f    = np.maximum(T_f, 1e-10)
    rho1_f = _rho(p_f, T_f, ph1)
    rho2_f = _rho(p_f, T_f, ph2)

    return psi_f * rho1_f + (1.0 - psi_f) * rho2_f


# ─────────────────────────────────────────────────────────────────
# Momentum predictor
# ─────────────────────────────────────────────────────────────────

def _mom_predictor(u: np.ndarray, u_prev: Optional[np.ndarray],
                   p: np.ndarray, rho: np.ndarray,
                   dx: float, dt: float, bdf2: bool,
                   bc_l: str, bc_r: str) -> np.ndarray:
    """
    Momentum predictor u* with explicit convection and old pressure gradient.

    BDF1:   u* = u^n - dt/ρ [conv(u^n) + ∂p^n/∂x]

    BDF2:   u* = (4u^n - u^{n-1})/3 - (2dt/3)/ρ [conv(u^n) + ∂p^n/∂x]

    Convection: first-order upwind  (ρ Du/Dt via conservative form then /ρ)
    Pressure gradient: central differences
    """
    p_g = _ghost(p, bc_l, bc_r)
    u_g = _ghost(u, bc_l, bc_r)

    # Face velocities for convection
    u_face = 0.5 * (u_g[:-1] + u_g[1:])                 # shape (N+1,)

    # Upwind u at each face
    u_up = np.where(u_face >= 0.0, u_g[:-1], u_g[1:])   # shape (N+1,)

    # Convective term: (ρu·u)_{m+1/2} - (ρu·u)_{m-1/2}  divided by ρ·dx
    # Using simple ρ_face = 0.5*(ρ_m + ρ_{m+1}) for convection
    rho_g   = _ghost(rho, bc_l, bc_r)
    rho_face = 0.5 * (rho_g[:-1] + rho_g[1:])
    conv = ((rho_face[1:] * u_face[1:] * u_up[1:]
             - rho_face[:-1] * u_face[:-1] * u_up[:-1]) / dx)  # shape (N,)

    # Cell-centred pressure gradient (central difference)
    dpdx = (p_g[2:] - p_g[:-2]) / (2.0 * dx)   # shape (N,)

    rho_safe = np.maximum(rho, 1e-300)

    if bdf2 and u_prev is not None:
        beta = 2.0 * dt / 3.0
        u_star = (4.0 * u - u_prev) / 3.0 - beta * (conv + dpdx) / rho_safe
    else:
        beta = dt
        u_star = u - beta * (conv + dpdx) / rho_safe

    return u_star


# ─────────────────────────────────────────────────────────────────
# Pressure Helmholtz equation  (tridiagonal, Denner §3.2)
# ─────────────────────────────────────────────────────────────────

def _solve_pressure(chi: np.ndarray, rho_f: np.ndarray,
                    u_star: np.ndarray, p_old: np.ndarray,
                    p_prev: Optional[np.ndarray],
                    dx: float, dt: float, bdf2: bool,
                    bc_l: str, bc_r: str) -> np.ndarray:
    """
    Assemble and solve the 1D pressure Helmholtz equation.

    Derived from acoustic equation  χ ∂p/∂t + ∂u^{n+1}/∂x = 0
    with Rhie-Chow face velocity:
        u_{m+1/2}^{n+1} = ū_{m+1/2}^* - (β/ρ_f) (p^{n+1}_{m+1} - p^{n+1}_m) / dx

    where β = dt (BDF1) or 2dt/3 (BDF2).

    BDF1 system (cell m):
        [χ_m/dt + β/dx² (1/ρ_e + 1/ρ_w)] p_m
         - β/(ρ_e dx²) p_{m+1}  - β/(ρ_w dx²) p_{m-1}
         = χ_m p_m^n / dt  - (ū_e^* - ū_w^*) / dx

    BDF2: replace χ/dt → 3χ/(2dt), RHS → χ(4p^n - p^{n-1})/(2dt) - div(u*)

    BC (transmissive): ghost cell pressure = boundary cell pressure
        → no off-diagonal term at boundary, effectively ∂p/∂n = 0
    BC (periodic): wrap-around connectivity

    Returns
    -------
    p_new : shape (N,)
    """
    N = len(chi)
    beta = (2.0 * dt / 3.0) if bdf2 else dt

    # Face densities: rho_f[m] = ρ at face m-1/2 (left face of cell m)
    # rho_f[m+1] = ρ at face m+1/2 (right face of cell m)
    # rho_f has shape (N+1,); rho_f[0]=left boundary, rho_f[N]=right boundary
    rho_w = rho_f[:-1]   # left faces  of cells 0..N-1
    rho_e = rho_f[1:]    # right faces of cells 0..N-1

    rho_w_safe = np.maximum(rho_w, 1e-300)
    rho_e_safe = np.maximum(rho_e, 1e-300)

    aw = beta / (dx * dx * rho_w_safe)
    ae = beta / (dx * dx * rho_e_safe)

    # Predictor face velocity divergence:  (ū_e^* - ū_w^*) / dx
    u_star_g = _ghost(u_star, bc_l, bc_r)
    u_face_star = 0.5 * (u_star_g[:-1] + u_star_g[1:])   # shape (N+1,)
    div_u_star = (u_face_star[1:] - u_face_star[:-1]) / dx  # shape (N,)

    # Diagonal and RHS
    if bdf2 and p_prev is not None:
        aP_diag = 3.0 * chi / (2.0 * dt) + aw + ae
        rhs = chi * (4.0 * p_old - p_prev) / (2.0 * dt) - div_u_star
    else:
        aP_diag = chi / dt + aw + ae
        rhs = chi * p_old / dt - div_u_star

    # Adjust diagonal for BC (transmissive: no neighbour contribution)
    # For periodic BC, the wrap-around terms are added instead.
    if bc_l == 'transmissive':
        # left ghost = p[0]  → no contribution from left neighbour of cell 0
        aP_diag[0] -= aw[0]   # undo the aw[0] that assumed a left neighbour
    if bc_r == 'transmissive':
        aP_diag[-1] -= ae[-1]

    # Build banded matrix (scipy solve_banded format: ab[0]=upper, ab[1]=diag, ab[2]=lower)
    ab = np.zeros((3, N))
    ab[1, :] = aP_diag
    ab[0, 1:] = -ae[:-1]        # upper diagonal (coefficient of p_{m+1})
    ab[2, :-1] = -aw[1:]        # lower diagonal (coefficient of p_{m-1})

    if bc_l == 'periodic':
        # Cell 0 has a left neighbour = cell N-1 (wrap-around)
        # This makes the system non-tridiagonal; handle with dense solve for small N
        # For simplicity, use a dense system when periodic
        A_dense = np.diag(aP_diag) - np.diag(ae[:-1], 1) - np.diag(aw[1:], -1)
        A_dense[0, -1] = -aw[0]    # periodic wrap-around
        A_dense[-1, 0] = -ae[-1]
        return np.linalg.solve(A_dense, rhs)

    p_new = solve_banded((1, 1), ab, rhs)
    return p_new


# ─────────────────────────────────────────────────────────────────
# Velocity correction
# ─────────────────────────────────────────────────────────────────

def _velocity_correct(u_star: np.ndarray, p_new: np.ndarray, p_old: np.ndarray,
                      rho: np.ndarray, dx: float, dt: float, bdf2: bool,
                      bc_l: str, bc_r: str) -> np.ndarray:
    """
    Correct cell velocity using pressure gradient update.

        u^{n+1} = u* - β/ρ * (∇p^{n+1} - ∇p^n) / 2

    where β = dt (BDF1) or 2dt/3 (BDF2).
    Central-difference gradient uses ghost cells.
    """
    beta = (2.0 * dt / 3.0) if bdf2 else dt
    dp = p_new - p_old
    dp_g = _ghost(dp, bc_l, bc_r)
    grad_dp = (dp_g[2:] - dp_g[:-2]) / (2.0 * dx)
    return u_star - beta * grad_dp / np.maximum(rho, 1e-300)


# ─────────────────────────────────────────────────────────────────
# Temperature update  (material Dp/Dt)
# ─────────────────────────────────────────────────────────────────

def _temperature_update(T: np.ndarray, p_new: np.ndarray, p_old: np.ndarray,
                        u_new: np.ndarray, psi: np.ndarray, rho: np.ndarray,
                        ph1: dict, ph2: dict,
                        dx: float, dt: float, bc_l: str, bc_r: str) -> np.ndarray:
    """
    Explicit temperature update from the enthalpy equation.

        ρ cp_mix DT/Dt = Dp/Dt

    where cp_k = γ_k cv_k  (thermodynamically consistent for NASG),
    so cp_mix = (ψ ρ₁ γ₁ cv₁ + (1-ψ) ρ₂ γ₂ cv₂) / ρ.

    Dp/Dt uses the new pressure (semi-implicit in p).
    Temperature convection is first-order upwind.
    """
    rho1 = _rho(p_old, T, ph1)
    rho2 = _rho(p_old, T, ph2)
    cp1  = _cp(ph1)    # γ₁ cv₁
    cp2  = _cp(ph2)    # γ₂ cv₂
    cp_mix = (psi * rho1 * cp1 + (1.0 - psi) * rho2 * cp2) / np.maximum(rho, 1e-300)

    # Material derivative of p (semi-implicit: use p_new for ∂p/∂t, old for convection)
    p_g   = _ghost(p_old, bc_l, bc_r)
    dpdx  = (p_g[2:] - p_g[:-2]) / (2.0 * dx)
    Dp_Dt = (p_new - p_old) / dt + u_new * dpdx

    # Upwind convection of T
    T_g   = _ghost(T, bc_l, bc_r)
    u_g   = _ghost(u_new, bc_l, bc_r)
    u_f   = 0.5 * (u_g[:-1] + u_g[1:])
    T_up  = np.where(u_f >= 0.0, T_g[:-1], T_g[1:])
    dT_conv = (u_f[1:] * T_up[1:] - u_f[:-1] * T_up[:-1]) / dx

    T_new = T + dt * (Dp_Dt / np.maximum(rho * cp_mix, 1e-300) - dT_conv)
    return np.maximum(T_new, 1e-10)


# ─────────────────────────────────────────────────────────────────
# CFL time step
# ─────────────────────────────────────────────────────────────────

def _cfl_dt(p: np.ndarray, T: np.ndarray, psi: np.ndarray, u: np.ndarray,
            ph1: dict, ph2: dict, dx: float, CFL: float) -> float:
    """
    Compute stable time step from CFL condition using mixture sound speed.
    """
    rho1 = _rho(p, T, ph1)
    rho2 = _rho(p, T, ph2)
    a12  = _a2(p, ph1, rho1)
    a22  = _a2(p, ph2, rho2)
    chi  = (psi / np.maximum(rho1 * a12, 1e-300)
            + (1.0 - psi) / np.maximum(rho2 * a22, 1e-300))
    rho  = psi * rho1 + (1.0 - psi) * rho2
    c2   = 1.0 / np.maximum(rho * chi, 1e-300)
    c    = np.sqrt(np.maximum(c2, 0.0))
    s_max = float(np.max(np.abs(u) + c))
    if s_max <= 0.0:
        s_max = 1.0
    return CFL * dx / s_max


# ─────────────────────────────────────────────────────────────────
# Main 1D solver entry point
# ─────────────────────────────────────────────────────────────────

def run_denner2018_1d(case_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a 1D pressure-based two-phase compressible Euler simulation
    using the Denner–Xiao–van Wachem (2018) PISO + ACID algorithm.

    Parameters
    ----------
    case_params : dict
        Required keys:
            'ph1'     : phase-1 EOS dict  {'gam': γ, 'pinf': p∞, 'cv': cv, 'q': q}
            'ph2'     : phase-2 EOS dict
            'x_cells' : cell-centre positions, shape (N,)
            'p_init'  : initial pressure,        shape (N,)
            'u_init'  : initial velocity,         shape (N,)
            'T_init'  : initial temperature,      shape (N,)
            'psi_init': initial vol. fraction,    shape (N,)
            't_end'   : simulation end time
        Optional keys:
            'CFL'            : CFL number (default 0.5)
            'bc_left'        : 'transmissive' | 'periodic'  (default 'transmissive')
            'bc_right'       : 'transmissive' | 'periodic'  (default 'transmissive')
            'output_times'   : list of times for snapshots
            'verbose'        : print progress every N steps (default False)
            'dt_fixed'       : fixed time step (overrides CFL if given)
            'use_bdf2'       : use BDF2 after first step (default True)

    Returns
    -------
    result : dict
        'p_final'   : pressure,      shape (N,)
        'u_final'   : velocity,      shape (N,)
        'T_final'   : temperature,   shape (N,)
        'psi_final' : vol. fraction, shape (N,)
        'rho_final' : density,       shape (N,)
        'x_cells'   : cell positions
        't_final'   : actual end time
        'n_steps'   : number of time steps
        'snapshots' : list of dicts with 't','p','u','T','psi','rho'
    """
    ph1      = case_params['ph1']
    ph2      = case_params['ph2']
    x_cells  = np.asarray(case_params['x_cells'], dtype=float)
    p        = np.asarray(case_params['p_init'],   dtype=float).copy()
    u        = np.asarray(case_params['u_init'],   dtype=float).copy()
    T        = np.asarray(case_params['T_init'],   dtype=float).copy()
    psi      = np.asarray(case_params['psi_init'], dtype=float).copy()
    t_end    = float(case_params['t_end'])

    N  = len(x_cells)
    dx = float(x_cells[1] - x_cells[0]) if N > 1 else 1.0

    CFL          = float(case_params.get('CFL', 0.5))
    bc_l         = case_params.get('bc_left',  'transmissive')
    bc_r         = case_params.get('bc_right', 'transmissive')
    output_times = sorted(case_params.get('output_times', []))
    verbose      = bool(case_params.get('verbose', False))
    dt_fixed     = case_params.get('dt_fixed', None)
    use_bdf2     = bool(case_params.get('use_bdf2', True))

    # Initial mixture density
    rho = _rho(p, T, ph1) * psi + _rho(p, T, ph2) * (1.0 - psi)

    # BDF2 storage (previous step values)
    p_prev = None
    u_prev = None

    t        = 0.0
    n_steps  = 0
    snapshots: List[Dict] = []
    out_idx  = 0

    # Save snapshot at t=0 if requested
    if output_times and abs(output_times[0]) < 1e-14:
        snapshots.append({'t': 0.0, 'p': p.copy(), 'u': u.copy(),
                          'T': T.copy(), 'psi': psi.copy(), 'rho': rho.copy()})
        out_idx += 1

    while t < t_end - 1e-14 * t_end:
        # ── Compute time step ──────────────────────────────────────────
        if dt_fixed is not None:
            dt = float(dt_fixed)
        else:
            dt = _cfl_dt(p, T, psi, u, ph1, ph2, dx, CFL)

        dt = min(dt, t_end - t)
        if out_idx < len(output_times):
            dt = min(dt, output_times[out_idx] - t)
        if dt <= 0.0:
            break

        bdf2 = use_bdf2 and (n_steps >= 1) and (p_prev is not None)

        # ── Step 1: mixture properties at (p^n, T^n, ψ^n) ────────────
        (rho1_n, h1_n, a12_n,
         rho2_n, h2_n, a22_n,
         rho_n, rhoh_n, chi_n, K_n) = _props(p, T, psi, ph1, ph2)

        # ── Step 2: advect color function ψ ───────────────────────────
        psi_new = _advect_psi(psi, u, K_n, dx, dt, bc_l, bc_r)

        # ── Step 3: momentum predictor u* ─────────────────────────────
        u_star = _mom_predictor(u, u_prev, p, rho_n, dx, dt, bdf2, bc_l, bc_r)

        # ── Step 4: ACID face densities ───────────────────────────────
        rho_f = _acid_face_rho(p, T, psi, ph1, ph2, bc_l, bc_r)   # shape (N+1,)

        # ── Step 5: solve pressure Helmholtz → p^{n+1} ────────────────
        p_new = _solve_pressure(chi_n, rho_f, u_star, p, p_prev,
                                dx, dt, bdf2, bc_l, bc_r)
        p_new = np.maximum(p_new, 1.0)   # pressure floor [Pa]

        # ── Step 6: velocity correction ───────────────────────────────
        u_new = _velocity_correct(u_star, p_new, p, rho_n, dx, dt, bdf2, bc_l, bc_r)

        # ── Step 7: temperature update ────────────────────────────────
        T_new = _temperature_update(T, p_new, p, u_new, psi_new, rho_n,
                                    ph1, ph2, dx, dt, bc_l, bc_r)

        # ── Derived density from SGS EOS ──────────────────────────────
        rho_new = (_rho(p_new, T_new, ph1) * psi_new
                   + _rho(p_new, T_new, ph2) * (1.0 - psi_new))

        # ── BDF2 storage: rotate p_prev ───────────────────────────────
        p_prev = p.copy()
        u_prev = u.copy()

        # ── Advance state ─────────────────────────────────────────────
        p   = p_new
        u   = u_new
        T   = T_new
        psi = psi_new
        rho = rho_new

        t       += dt
        n_steps += 1

        if verbose and n_steps % 100 == 0:
            print(f"  step={n_steps:6d}  t={t:.4e}  dt={dt:.3e}  "
                  f"p_max={p.max():.4e}  p_min={p.min():.4e}")

        # ── Save snapshots ────────────────────────────────────────────
        while out_idx < len(output_times) and t >= output_times[out_idx] - 1e-14:
            snapshots.append({'t': t, 'p': p.copy(), 'u': u.copy(),
                              'T': T.copy(), 'psi': psi.copy(), 'rho': rho.copy()})
            out_idx += 1

    if verbose:
        print(f"  Done: {n_steps} steps, t_final={t:.6e}")

    return {
        'p_final':   p,
        'u_final':   u,
        'T_final':   T,
        'psi_final': psi,
        'rho_final': rho,
        'x_cells':   x_cells,
        't_final':   t,
        'n_steps':   n_steps,
        'snapshots': snapshots,
    }
