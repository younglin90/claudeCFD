"""
ACID_1d.py
──────────────────────────────────────────────────────────────────
Acoustically-Conservative Interface Discretisation (ACID) for
1D multicomponent compressible flow with SRK EOS.

Reference : Denner, Xiao, van Wachem,
            "Pressure-based algorithm for compressible interfacial
             flows with acoustically-conservative interface
             discretisation", JCP 367 (2018) 192-234.

Physical setup : 1D CH4 blob advection in N2, p=5 MPa
                 (Terashima et al. JCP 524, 2025)
EOS            : Soave-Redlich-Kwong (SRK), CH4/N2 mixture
Numerics       : MUSCL-LLF (minmod) + SSP-RK3

ACID core idea (Denner §5)
──────────────────────────
Standard scheme: fluxes see the actual density/enthalpy jump at the
  interface → spurious pressure oscillations (Abgrall problem).

ACID fix: when computing cell P's flux contribution, ALL cells in
  its stencil are assigned cell P's composition (mass fraction Y1_P).
  → consistent thermodynamic properties across the stencil
  → asymmetric fluxes (non-conservative at species level)
  → energy flux dissipation ≈ 0 at uniform-pressure interface

Adaptation to density-based MUSCL-LLF
──────────────────────────────────────
  • Mass (ρ1, ρ2) and momentum (ρu): standard conservative MUSCL-LLF
  • Energy (ρE) only: ACID asymmetric flux

  For cell m's energy RHS:
    dE[m] = -(FE_left[m] - FE_right[m]) / dx

    FE_left[m]  = ACID flux at face m+1/2 using Y1[m], T[m]
    FE_right[m] = ACID flux at face m-1/2 using Y1[m], T[m]
    (different MUSCL face states, same cell-P composition)

Methods
───────
  FC   : fully-conservative standard MUSCL-LLF  (baseline)
  APEC : Terashima 2025 — MUSCL-consistent PE dissipation
  ACID : this file — Denner 2018 adapted to density-based MUSCL
──────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  Physical constants & SRK EOS  (CH4/N2)
# ─────────────────────────────────────────────────────────────
Ru   = 8.314                                      # J/(mol·K)
M    = np.array([16.043e-3, 28.014e-3])           # kg/mol
Tc   = np.array([190.56,    126.19   ])            # K
pc   = np.array([4.599e6,   3.396e6  ])            # Pa
om   = np.array([0.0115,    0.0372   ])            # acentric factor
a_sp = 0.42748 * Ru**2 * Tc**2 / pc
b_sp = 0.08664 * Ru   * Tc    / pc
fom  = 0.480 + 1.574*om - 0.176*om**2
Cv0  = Ru / M * np.array([3.4, 2.5])


def _mix(r1, r2, T):
    r1, r2, T = np.asarray(r1, float), np.asarray(r2, float), np.asarray(T, float)
    rho  = r1 + r2
    C0, C1 = r1 / M[0], r2 / M[1]
    Ctot = np.maximum(C0 + C1, 1e-40)
    X0, X1 = C0 / Ctot, C1 / Ctot
    Mbar = rho / Ctot

    sq0 = np.sqrt(np.maximum(T / Tc[0], 1e-10))
    sq1 = np.sqrt(np.maximum(T / Tc[1], 1e-10))
    al0  = (1.0 + fom[0]*(1.0 - sq0))**2
    al1  = (1.0 + fom[1]*(1.0 - sq1))**2
    da0  = -fom[0]*(1.0 + fom[0]*(1.0 - sq0)) / (sq0*Tc[0] + 1e-300)
    da1  = -fom[1]*(1.0 + fom[1]*(1.0 - sq1)) / (sq1*Tc[1] + 1e-300)

    a01   = np.sqrt(a_sp[0]*a_sp[1])
    al01  = np.sqrt(np.maximum(al0*al1, 1e-300))
    dal01 = a01*(al1*da0 + al0*da1) / (2.0*al01)

    aA  = X0*X0*a_sp[0]*al0 + X1*X1*a_sp[1]*al1 + 2.0*X0*X1*a01*al01
    daA = X0*X0*a_sp[0]*da0 + X1*X1*a_sp[1]*da1 + 2.0*X0*X1*dal01

    b = X0*b_sp[0] + X1*b_sp[1]
    v = Mbar / np.maximum(rho, 1e-30)
    return rho, Mbar, X0, X1, C0, C1, Ctot, b, v, aA, daA


def srk_p(r1, r2, T):
    _, _, _, _, _, _, _, b, v, aA, _ = _mix(r1, r2, T)
    vb  = np.maximum(v - b, 1e-20)
    vvb = np.maximum(v*(v + b), 1e-60)
    return Ru*T/vb - aA/vvb


def srk_rhoe(r1, r2, T):
    _, _, _, _, _, _, Ctot, b, v, aA, daA = _mix(r1, r2, T)
    rhoe0 = r1*Cv0[0]*T + r2*Cv0[1]*T
    arg   = np.where(b > 1e-40, 1.0 + b/np.maximum(v, 1e-30), 1.0)
    dep   = np.where(b > 1e-40,
                     Ctot*(T*daA - aA)/b * np.log(np.maximum(arg, 1e-30)),
                     0.0)
    return rhoe0 + dep


def T_from_p(r1, r2, p_target, T_in=None):
    """Newton iteration: find T such that srk_p(r1, r2, T) = p_target.
    Used by ACID to find the pressure-consistent temperature when composition
    is modified but pressure must remain equal to the original face pressure.
    """
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    p_target = np.asarray(p_target, float)
    T = (np.asarray(T_in, float).copy() if T_in is not None
         else np.full_like(r1, 200.0))
    T = np.clip(T, 10.0, 3000.0)
    h = 1.0
    for _ in range(20):
        f0   = srk_p(r1, r2, T) - p_target
        dfdT = (srk_p(r1, r2, T+h) - srk_p(r1, r2, T-h)) / (2.0*h)
        dT   = np.clip(-f0 / (dfdT + 1e-3), -200.0, 200.0)
        T    = np.clip(T + dT, 10.0, 3000.0)
        if np.max(np.abs(dT)) < 5e-2:
            break
    return T


def T_from_rhoe(r1, r2, rhoe_target, T_in=None):
    r1 = np.asarray(r1, float)
    r2 = np.asarray(r2, float)
    rhoe_target = np.asarray(rhoe_target, float)
    T  = (np.asarray(T_in, float).copy() if T_in is not None
          else np.full_like(r1, 200.0))
    T  = np.clip(T, 10.0, 3000.0)
    h  = 1.0
    for _ in range(25):
        f0   = srk_rhoe(r1, r2, T) - rhoe_target
        fp   = srk_rhoe(r1, r2, T + h)
        fm   = srk_rhoe(r1, r2, T - h)
        dfdT = (fp - fm) / (2.0*h)
        dT   = np.clip(-f0 / (dfdT + 1e-6), -200.0, 200.0)
        T    = np.clip(T + dT, 10.0, 3000.0)
        if np.max(np.abs(dT)) < 1e-3:
            break
    return T


def _dpdT(r1, r2, T, h=1.0):
    return (srk_p(r1, r2, T+h) - srk_p(r1, r2, T-h)) / (2.0*h)


def _dpdr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s == 0 else r2)*f, dmin)
    if s == 0:
        return (srk_p(r1+dr, r2, T) - srk_p(r1-dr, r2, T)) / (2.0*dr)
    return (srk_p(r1, r2+dr, T) - srk_p(r1, r2-dr, T)) / (2.0*dr)


def _drhoedr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s == 0 else r2)*f, dmin)
    if s == 0:
        return (srk_rhoe(r1+dr, r2, T) - srk_rhoe(r1-dr, r2, T)) / (2.0*dr)
    return (srk_rhoe(r1, r2+dr, T) - srk_rhoe(r1, r2-dr, T)) / (2.0*dr)


def srk_Cv(r1, r2, T, h=1.0):
    return ((srk_rhoe(r1, r2, T+h) - srk_rhoe(r1, r2, T-h)) / (2.0*h)
            / np.maximum(r1 + r2, 1e-30))


def epsilon_v(r1, r2, T, s):
    """ε_s = (∂ρe/∂ρ_s)_{p,ρ_{j≠s}}  — needed for APEC."""
    rho  = r1 + r2
    Cv   = srk_Cv(r1, r2, T)
    dpT  = _dpdT(r1, r2, T)
    dprs = _dpdr(r1, r2, T, s)
    drhoers = _drhoedr(r1, r2, T, s)
    return -(rho * Cv / (dpT + 1e-10)) * dprs + drhoers


def srk_c2(r1, r2, T):
    rho   = r1 + r2
    Y0    = r1 / np.maximum(rho, 1e-30)
    dpdrT = Y0*_dpdr(r1, r2, T, 0) + (1.0 - Y0)*_dpdr(r1, r2, T, 1)
    dpT   = _dpdT(r1, r2, T)
    Cv    = srk_Cv(r1, r2, T)
    return np.maximum(dpdrT + T*dpT**2 / (rho**2*Cv + 1e-30), 100.0)


# ─────────────────────────────────────────────────────────────
# 2.  Initial condition
# ─────────────────────────────────────────────────────────────
def initial_condition(x, p_inf=5e6, k=15.0):
    N, xc, rc = len(x), 0.5, 0.25
    r1_inf, r2_inf = 400.0, 100.0
    r = np.abs(x - xc)
    r1 = 0.5*r1_inf*(1.0 - np.tanh(k*(r - rc)))
    r2 = 0.5*r2_inf*(1.0 + np.tanh(k*(r - rc)))

    print("  Solving T from p_inf=5MPa ...", flush=True)
    _, Mb, _, _, _, _, _, *_ = _mix(r1, r2, np.full(N, 300.0))
    T  = np.clip(p_inf * Mb / np.maximum(r1 + r2, 1e-30) / Ru, 50.0, 1000.0)
    for _ in range(60):
        ph, dph = srk_p(r1, r2, T), _dpdT(r1, r2, T)
        dT = np.clip(-(ph - p_inf) / (dph + 1e-3), -100.0, 100.0)
        T  = np.clip(T + dT, 10.0, 2000.0)
        if np.max(np.abs(dT)) < 1e-3:
            break

    p    = srk_p(r1, r2, T)
    rhoe = srk_rhoe(r1, r2, T)
    u    = np.full(N, 100.0)
    return r1, r2, u, rhoe + 0.5*(r1+r2)*u**2, T, p


# ─────────────────────────────────────────────────────────────
# 3.  Numerics helpers
# ─────────────────────────────────────────────────────────────
def muscl_lr(q):
    """Minmod MUSCL left/right states at all faces m+1/2."""
    dR  = np.roll(q, -1) - q
    dL  = q - np.roll(q,  1)
    slp = np.where(dL*dR > 0.0,
                   np.sign(dL)*np.minimum(np.abs(dL), np.abs(dR)),
                   0.0)
    return q + 0.5*slp, np.roll(q - 0.5*slp, -1)


def prim(r1, r2, rhoU, rhoE, T_prev):
    rho  = r1 + r2
    u    = rhoU / np.maximum(rho, 1e-30)
    rhoe = rhoE - 0.5*rho*u**2
    T    = T_from_rhoe(r1, r2, rhoe, T_in=T_prev)
    return u, rhoe, T, srk_p(r1, r2, T)


# ─────────────────────────────────────────────────────────────
# 4.  FC and APEC fluxes (MUSCL-LLF)
# ─────────────────────────────────────────────────────────────
def _fc_apec_fluxes(r1, r2, u, rhoe, p, T, lam_cell, scheme, eps_pair):
    """FC-NPE or APEC fluxes (symmetric, conservative).

    FC  : standard MUSCL-LLF energy flux
    APEC: Terashima 2025 Appendix A (Eq. A.4) — SRK version
          Uses cell-centered fluxes + PE-consistent correction terms.
          Matches the validated formula in apec_1d.py.
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)

    rhoL  = r1L + r2L
    rhoR  = r1R + r2R
    lam   = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # Conservative mass & momentum
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU = 0.5*(rhoL*uL**2 + pL + rhoR*uR**2 + pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    if scheme == 'APEC':
        # ── SRK Appendix A (Eq. A.4) — identical to apec_1d.py ──────
        # Cell-centred fluxes (no upwinding)
        F1_cell  = r1 * u
        F2_cell  = r2 * u
        FU_cell  = rho * u**2 + p
        FE_cell  = (rhoE + p) * u

        eps0, eps1 = eps_pair
        eps0_m1  = np.roll(eps0, -1)
        eps1_m1  = np.roll(eps1, -1)
        u_m1     = np.roll(u,    -1)
        FE_m1    = np.roll(FE_cell, -1)

        # Correction from left cell (m) and right cell (m+1) at face m+1/2
        c0m  = eps0    - 0.5*u**2
        c1m  = eps1    - 0.5*u**2
        c0m1 = eps0_m1 - 0.5*u_m1**2
        c1m1 = eps1_m1 - 0.5*u_m1**2

        tm  = (c0m *(F1 - F1_cell)             + c1m *(F2 - F2_cell)
               + u   *(FU - FU_cell))
        tm1 = (c0m1*(np.roll(F1_cell,-1) - F1) + c1m1*(np.roll(F2_cell,-1) - F2)
               + u_m1*(np.roll(FU_cell,-1) - FU))

        FE = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1
    else:
        # FC-NPE: standard MUSCL-LLF energy flux
        # Each face state uses the temperature of its parent cell:
        #   Left state at face m+1/2  → T[m]      (current cell)
        #   Right state at face m+1/2 → T[m+1]    (adjacent cell, np.roll(T,-1))
        # This is the correct baseline that produces the expected PE error.
        rhoEL = srk_rhoe(r1L, r2L, T)            + 0.5*rhoL*uL**2
        rhoER = srk_rhoe(r1R, r2R, np.roll(T,-1)) + 0.5*rhoR*uR**2
        FE = 0.5*((rhoEL+pL)*uL + (rhoER+pR)*uR) - 0.5*lam*(rhoER - rhoEL)

    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# 5.  ACID energy flux  (Denner 2018 §5, adapted to density-based)
# ─────────────────────────────────────────────────────────────
def _acid_fluxes(r1, r2, u, rhoe, p, T, lam_cell, eps_pair):
    """ACID flux set — NON-conservative energy using cell-m ε for both faces.

    Adaptation of Denner 2018 §5 to a density-based MUSCL-LLF solver.

    Key idea
    ────────
    APEC (Terashima App. A) at face m+1/2:
        FE = 0.5*(FE_m + FE_{m+1}) + 0.5*tm - 0.5*tm1
      where tm uses ε[m] and tm1 uses ε[m+1].

    ACID: cell m computes its energy RHS using ε[m] for BOTH adjacent faces:
      Face m+1/2 (cell m is LEFT):
        FE_left[m] = 0.5*(FE_m+FE_{m+1}) + 0.5*tm - 0.5*tm1_acid
        where tm1_acid uses ε[m] instead of ε[m+1] (ACID modification)
      Face m-1/2 (cell m is RIGHT):
        FE_right[m] = 0.5*(FE_{m-1}+FE_m) + 0.5*tm_mm_acid - 0.5*tm1_mm
        where tm_mm_acid uses ε[m] instead of ε[m-1]

    Cell m energy RHS (NON-conservative):
        dE[m] = -(FE_left[m] - FE_right[m]) / dx

    Away from interface (ε[m] ≈ ε[m±1]): ACID ≈ APEC.
    At interface: cell m sees consistent thermodynamics at BOTH adjacent faces.

    Returns
    ───────
    F1, F2, FU : conservative fluxes at face m+1/2
    FE_left    : ACID energy flux at face m+1/2 (cell m's view)
    FE_right   : ACID energy flux at face m-1/2 (cell m's view)
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)
    rhoL = r1L + r2L
    rhoR = r1R + r2R
    lam  = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # ── Conservative mass & momentum ─────────────────────────────
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU = 0.5*(rhoL*uL**2+pL + rhoR*uR**2+pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    # ── Cell-centred fluxes ──────────────────────────────────────
    F1_cell = r1 * u
    F2_cell = r2 * u
    FU_cell = rho * u**2 + p
    FE_cell = (rhoE + p) * u
    FE_m1   = np.roll(FE_cell, -1)
    FE_mm1  = np.roll(FE_cell,  1)

    eps0, eps1 = eps_pair
    u_m1  = np.roll(u,  -1)   # u[m+1]
    u_mm1 = np.roll(u,   1)   # u[m-1]

    # ── ACID face m+1/2: use ε[m] for both tm and tm1 ────────────
    # tm (left-cell correction, cell m is left): same as APEC
    c0m = eps0 - 0.5*u**2
    c1m = eps1 - 0.5*u**2
    tm = c0m*(F1 - F1_cell) + c1m*(F2 - F2_cell) + u*(FU - FU_cell)

    # tm1_acid (right-cell correction, use ε[m] but velocity u[m+1])
    c0m_right = eps0 - 0.5*u_m1**2   # ε[m] with u[m+1] for kinetic consistency
    c1m_right = eps1 - 0.5*u_m1**2
    tm1_acid = (c0m_right*(np.roll(F1_cell,-1) - F1)
                + c1m_right*(np.roll(F2_cell,-1) - F2)
                + u_m1*(np.roll(FU_cell,-1) - FU))

    FE_left = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1_acid

    # ── ACID face m-1/2: use ε[m] for both tm and tm1 ────────────
    F1_mm = np.roll(F1, 1)     # face flux at m-1/2
    F2_mm = np.roll(F2, 1)
    FU_mm = np.roll(FU, 1)
    F1_cell_mm1 = np.roll(F1_cell, 1)   # F1_cell[m-1]
    F2_cell_mm1 = np.roll(F2_cell, 1)
    FU_cell_mm1 = np.roll(FU_cell, 1)

    # tm_mm_acid: left-cell (m-1) correction at face m-1/2 using ε[m]
    c0m_left_mm = eps0 - 0.5*u_mm1**2   # ε[m] with u[m-1] for kinetic consistency
    c1m_left_mm = eps1 - 0.5*u_mm1**2
    tm_mm_acid = (c0m_left_mm*(F1_mm - F1_cell_mm1)
                  + c1m_left_mm*(F2_mm - F2_cell_mm1)
                  + u_mm1*(FU_mm - FU_cell_mm1))

    # tm1_mm (right-cell correction, cell m is right): use ε[m] (same as ACID)
    tm1_mm = c0m*(F1_cell - F1_mm) + c1m*(F2_cell - F2_mm) + u*(FU_cell - FU_mm)

    FE_right = 0.5*(FE_mm1 + FE_cell) + 0.5*tm_mm_acid - 0.5*tm1_mm

    return F1, F2, FU, FE_left, FE_right


# ─────────────────────────────────────────────────────────────
# 6.  RHS
# ─────────────────────────────────────────────────────────────
def rhs(U, scheme, dx, T_prev):
    r1, r2, rhoU, rhoE = U
    u, rhoe, T, p = prim(r1, r2, rhoU, rhoE, T_prev)
    lam_c = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))

    eps_pair = (epsilon_v(r1, r2, T, 0), epsilon_v(r1, r2, T, 1)) \
               if scheme in ('APEC', 'ACID') else None

    if scheme == 'ACID':
        # Non-conservative energy: cell m uses ε[m] at BOTH adjacent faces
        F1, F2, FU, FE_left, FE_right = _acid_fluxes(
            r1, r2, u, rhoe, p, T, lam_c, eps_pair)
        d1 = -(F1      - np.roll(F1, 1)) / dx  # conservative
        d2 = -(F2      - np.roll(F2, 1)) / dx  # conservative
        dU = -(FU      - np.roll(FU, 1)) / dx  # conservative
        dE = -(FE_left - FE_right       ) / dx  # non-conservative
    else:
        F1, F2, FU, FE = _fc_apec_fluxes(r1, r2, u, rhoe, p, T,
                                          lam_c, scheme, eps_pair)
        d1 = -(F1 - np.roll(F1, 1)) / dx
        d2 = -(F2 - np.roll(F2, 1)) / dx
        dU = -(FU - np.roll(FU, 1)) / dx
        dE = -(FE - np.roll(FE, 1)) / dx

    return [d1, d2, dU, dE], T, p


# ─────────────────────────────────────────────────────────────
# 7.  SSP-RK3 time integration
# ─────────────────────────────────────────────────────────────
def _clip(U):
    return [np.maximum(U[0], 0.0), np.maximum(U[1], 0.0), U[2], U[3]]


def rkstep(U, scheme, dx, dt, T_prev):
    k1, T1, p1 = rhs(U, scheme, dx, T_prev)
    U1 = _clip([U[q] + dt*k1[q] for q in range(4)])
    k2, T2, p2 = rhs(U1, scheme, dx, T1)
    U2 = _clip([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])
    k3, T3, p3 = rhs(U2, scheme, dx, T2)
    return _clip([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)]), T3, p3


# ─────────────────────────────────────────────────────────────
# 8.  Diagnostics
# ─────────────────────────────────────────────────────────────
def pe_err(p, p0=5e6):
    return float(np.max(np.abs(p - p0)) / p0)


def energy_err(rhoE, rhoE0):
    return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-30))


# ─────────────────────────────────────────────────────────────
# 9.  Main runner
# ─────────────────────────────────────────────────────────────
def run(scheme, N=101, t_end=0.07, CFL=0.4, p_inf=5e6, k=15.0,
        verbose=True):
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1 - dx/2, N)
    if verbose:
        print(f"\n{'='*55}")
        print(f"Scheme: {scheme}  N={N}  t_end={t_end:.4f}  CFL={CFL}  k={k}")

    r1, r2, u, rhoE, T, p = initial_condition(x, p_inf, k=k)
    U    = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
    rhoE0 = rhoE.copy()

    t_hist, pe_hist, en_hist = [0.0], [pe_err(p, p_inf)], [0.0]
    t, step, diverged = 0.0, 0, False

    while t < t_end - 1e-14:
        lam = float(np.max(
            np.abs(U[2] / np.maximum(U[0]+U[1], 1e-30))
            + np.sqrt(srk_c2(U[0], U[1], T))
        ))
        dt = min(CFL*dx/(lam + 1e-10), t_end - t)

        try:
            U, T, p = rkstep(U, scheme, dx, dt, T)
        except Exception as e:
            if verbose:
                print(f"  Exception at t={t:.5f}: {e}")
            diverged = True
            break

        t += dt; step += 1
        pe_ = pe_err(p, p_inf)
        en_ = energy_err(U[3], rhoE0)
        t_hist.append(t); pe_hist.append(pe_); en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 5.0:
            if verbose:
                print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}")
            diverged = True
            break

    if verbose:
        status = 'Completed' if not diverged else 'Diverged'
        print(f"  --> {status} at t={t:.5f}  ({step} steps)")

    return (x, U, T, p,
            np.array(t_hist), np.array(pe_hist), np.array(en_hist),
            diverged)


# ─────────────────────────────────────────────────────────────
# 10.  Comparison plots & validation
# ─────────────────────────────────────────────────────────────
def compare_pe_time(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """PE vs time for FC / APEC / ACID."""
    print("\n[Test 1] PE vs time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    results = {}
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, eh, div = run(sc, N=N, t_end=t_end,
                                           CFL=CFL, k=k)
        results[sc] = (th, ph, div)
        print(f"  {sc:4s}  PE(t=1step)={ph[1]:.3e}  "
              f"PE(final)={ph[-1]:.3e}  diverged={div}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'ACID': 'tab:green'}
    for sc, (th, ph, div) in results.items():
        ax.semilogy(th*1e3, ph, label=sc, color=colors[sc])
    ax.set_xlabel('t  [ms]')
    ax.set_ylabel('max |Δp/p₀|')
    ax.set_title(f'Pressure-equilibrium error  N={N}  k={k}  CFL={CFL}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'acid_pe_time_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    return results


def compare_resolution(t_end=1e-4, CFL=0.3, k=15.0):
    """PE at t≈1 step for different N (FC / APEC / ACID)."""
    print("\n[Test 2] Resolution study  k=%.0f" % k)
    Ns = [51, 101, 201, 501]
    header = f"{'N':>6}  {'FC PE':>12}  {'APEC PE':>12}  {'ACID PE':>12}"
    print("  " + header)
    print("  " + "-"*len(header))
    rows = []
    for N in Ns:
        dx  = 1.0 / N
        x   = np.linspace(dx/2, 1 - dx/2, N)
        r1, r2, u, rhoE, T, p0 = initial_condition(x, k=k)
        U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
        row = [N]
        for sc in ('FC', 'APEC', 'ACID'):
            lam = float(np.max(np.abs(u) + np.sqrt(srk_c2(r1, r2, T))))
            dt  = CFL * dx / (lam + 1e-10)
            U1, T1, p1 = rkstep(U, sc, dx, dt, T)
            row.append(pe_err(p1))
        rows.append(row)
        fc_, ap_, ac_ = row[1], row[2], row[3]
        print(f"  {N:>6}  {fc_:>12.3e}  {ap_:>12.3e}  {ac_:>12.3e}")
    return rows


def compare_divergence(N=101, t_end=0.06, CFL=0.3, k=15.0):
    """Divergence time comparison."""
    print("\n[Test 3] Divergence time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, th, ph, _, div = run(sc, N=N, t_end=t_end,
                                          CFL=CFL, k=k)
        t_div = th[-1]
        print(f"  {sc:4s}  t_diverge={t_div*1e3:.2f} ms  diverged={div}")


def plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=5e-3):
    """Snapshot profiles: density, pressure, velocity."""
    print(f"\n[Test 4] Profiles at t={t_snap*1e3:.1f} ms  N={N}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'ACID': 'tab:green'}

    for sc in ('FC', 'APEC', 'ACID'):
        x, U, T, p, *_ = run(sc, N=N, t_end=t_snap, CFL=CFL, k=k)
        r1, r2, rhoU, rhoE = U
        rho = r1 + r2
        u   = rhoU / np.maximum(rho, 1e-30)
        axes[0].plot(x, rho, label=sc, color=colors[sc])
        axes[1].plot(x, p*1e-6, label=sc, color=colors[sc])
        axes[2].plot(x, u, label=sc, color=colors[sc])

    titles = ['Density  [kg/m³]', 'Pressure  [MPa]', 'Velocity  [m/s]']
    for ax, ttl in zip(axes, titles):
        ax.set_xlabel('x')
        ax.set_title(ttl)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f't = {t_snap*1e3:.1f} ms,  N={N},  k={k}', fontsize=11)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'acid_profiles_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────
# 11.  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # ── Quick single-step PE comparison across resolutions ────
    compare_resolution(k=15.0)

    # ── PE vs time  (N=101, up to divergence) ────────────────
    compare_pe_time(N=101, t_end=0.055, CFL=0.3, k=15.0)

    # ── Divergence time ───────────────────────────────────────
    compare_divergence(N=101, t_end=0.055, CFL=0.3, k=15.0)

    # ── Snapshot profiles ─────────────────────────────────────
    plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=3e-3)

    print("\nDone.")
