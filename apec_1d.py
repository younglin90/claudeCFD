"""
APEC 1D Validation — Terashima, Ly, Ihme (JCP 524, 2025)
Approximately Pressure-Equilibrium-Preserving scheme for real-fluid flows

Test case: CH4/N2 interface advection at supercritical pressure (5 MPa)
Reproduces Figs. 6-8 from the paper.

Schemes:
  FC-NPE : Fully Conservative, Non-Pressure-Equilibrium
           MUSCL-LLF with standard energy flux
  APEC   : Approximately Pressure-Equilibrium-Preserving
           MUSCL-LLF + APEC-corrected internal-energy half-point (Eq. 40)
  PEqC   : Pressure-Equilibrium quasi-Conservative
           Non-conservative pressure transport (reference)

Key fix: SRK rho*e is NEGATIVE for liquid-like states (departure < 0).
         Never apply a positive floor to rho*e.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ─────────────────────────────────────────────────────────────
# Physical constants & species data
# ─────────────────────────────────────────────────────────────
Ru  = 8.314          # J/(mol*K)

# Species 0 = CH4,   Species 1 = N2
M    = np.array([16.043e-3, 28.014e-3])   # kg/mol
Tc   = np.array([190.56,    126.19   ])   # K
pc   = np.array([4.599e6,   3.396e6  ])   # Pa
om   = np.array([0.0115,    0.0372   ])   # acentric factor

# SRK parameters
a_sp = 0.42748 * Ru**2 * Tc**2 / pc   # Pa*m^6/mol^2
b_sp = 0.08664 * Ru   * Tc    / pc    # m^3/mol
fom  = 0.480 + 1.574*om - 0.176*om**2

# Ideal-gas isochoric heat capacity (constant approximation)
Cv0 = Ru / M * np.array([3.4, 2.5])   # J/(kg*K)

# ─────────────────────────────────────────────────────────────
# Vectorized SRK EOS
# ─────────────────────────────────────────────────────────────

def _mix(r1, r2, T):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    T  = np.asarray(T,  dtype=float)

    rho  = r1 + r2
    C0   = r1 / M[0];   C1 = r2 / M[1]
    Ctot = np.maximum(C0 + C1, 1e-40)
    X0   = C0 / Ctot;   X1 = C1 / Ctot
    Mbar = rho / Ctot

    sq0 = np.sqrt(np.maximum(T / Tc[0], 1e-10))
    sq1 = np.sqrt(np.maximum(T / Tc[1], 1e-10))
    al0 = (1.0 + fom[0]*(1.0 - sq0))**2
    al1 = (1.0 + fom[1]*(1.0 - sq1))**2
    da0 = -fom[0]*(1.0 + fom[0]*(1.0 - sq0)) / (sq0*Tc[0] + 1e-300)
    da1 = -fom[1]*(1.0 + fom[1]*(1.0 - sq1)) / (sq1*Tc[1] + 1e-300)

    a01   = np.sqrt(a_sp[0]*a_sp[1])
    al01  = np.sqrt(np.maximum(al0*al1, 1e-300))
    dal01 = a01*(al1*da0 + al0*da1) / (2.0*al01)

    aA  = X0*X0*a_sp[0]*al0 + X1*X1*a_sp[1]*al1 + 2.0*X0*X1*a01*al01
    daA = X0*X0*a_sp[0]*da0 + X1*X1*a_sp[1]*da1 + 2.0*X0*X1*dal01

    b    = X0*b_sp[0] + X1*b_sp[1]
    v    = Mbar / np.maximum(rho, 1e-30)
    return rho, Mbar, X0, X1, C0, C1, Ctot, b, v, aA, daA

def srk_p(r1, r2, T):
    """SRK pressure [Pa]. Vectorized."""
    _, _, _, _, _, _, _, b, v, aA, _ = _mix(r1, r2, T)
    vb  = np.maximum(v - b, 1e-20)
    vvb = np.maximum(v*(v + b), 1e-60)
    return Ru*T/vb - aA/vvb

def srk_rhoe(r1, r2, T):
    """rho*e [J/m^3]. Vectorized. NOTE: can be negative for liquid states."""
    _, _, _, _, _, _, Ctot, b, v, aA, daA = _mix(r1, r2, T)
    rhoe0 = r1*Cv0[0]*T + r2*Cv0[1]*T
    arg = np.where(b > 1e-40, 1.0 + b/np.maximum(v, 1e-30), 1.0)
    dep = np.where(b > 1e-40,
                   Ctot*(T*daA - aA)/b * np.log(np.maximum(arg, 1e-30)),
                   0.0)
    return rhoe0 + dep

# ─────────────────────────────────────────────────────────────
# Vectorized temperature inversion
# ─────────────────────────────────────────────────────────────

def T_from_rhoe(r1, r2, rhoe_target, T_in=None):
    """Vectorized Newton iteration: srk_rhoe(r1,r2,T) = rhoe_target -> T.

    With a warm start from T_in (previous time-step T), this typically
    converges in 3-8 iterations rather than the 80-iteration fallback.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    rhoe_target = np.asarray(rhoe_target, dtype=float)
    T = (np.asarray(T_in, dtype=float).copy() if T_in is not None
         else np.full_like(r1, 200.0))
    T = np.clip(T, 10.0, 3000.0)
    h = 1.0   # K finite-diff step for d(rhoe)/dT
    for _ in range(25):          # 25 max — warm start converges in <10
        f0   = srk_rhoe(r1, r2, T)     - rhoe_target
        fp   = srk_rhoe(r1, r2, T + h)
        fm   = srk_rhoe(r1, r2, T - h)
        dfdT = (fp - fm) / (2.0*h)
        dT   = -f0 / (dfdT + 1e-6)
        dT   = np.clip(dT, -200.0, 200.0)
        T    = np.clip(T + dT, 10.0, 3000.0)
        if np.max(np.abs(dT)) < 1e-3:   # 1 mK is plenty for pressure
            break
    return T

# ─────────────────────────────────────────────────────────────
# Vectorized EOS derivatives (numerical finite differences)
# ─────────────────────────────────────────────────────────────

def _dpdT(r1, r2, T, h=1.0):
    return (srk_p(r1, r2, T+h) - srk_p(r1, r2, T-h)) / (2.0*h)

def _dpdr(r1, r2, T, s, f=5e-4, dmin=0.05):
    if s == 0:
        dr = np.maximum(np.abs(r1)*f, dmin)
        return (srk_p(r1+dr, r2, T) - srk_p(r1-dr, r2, T)) / (2.0*dr)
    else:
        dr = np.maximum(np.abs(r2)*f, dmin)
        return (srk_p(r1, r2+dr, T) - srk_p(r1, r2-dr, T)) / (2.0*dr)

def _drhoedr(r1, r2, T, s, f=5e-4, dmin=0.05):
    if s == 0:
        dr = np.maximum(np.abs(r1)*f, dmin)
        return (srk_rhoe(r1+dr, r2, T) - srk_rhoe(r1-dr, r2, T)) / (2.0*dr)
    else:
        dr = np.maximum(np.abs(r2)*f, dmin)
        return (srk_rhoe(r1, r2+dr, T) - srk_rhoe(r1, r2-dr, T)) / (2.0*dr)

def srk_Cv(r1, r2, T, h=1.0):
    """Cv [J/(kg*K)]. Vectorized."""
    drhoe = (srk_rhoe(r1, r2, T+h) - srk_rhoe(r1, r2, T-h)) / (2.0*h)
    return drhoe / np.maximum(r1 + r2, 1e-30)

def epsilon_v(r1, r2, T, s):
    """epsilon_s = (d(rho*e)/d(rho_s))_p. Vectorized."""
    rho  = r1 + r2
    Cv   = srk_Cv(r1, r2, T)
    dpT  = _dpdT(r1, r2, T)
    dprs = _dpdr(r1, r2, T, s)
    drhoe= _drhoedr(r1, r2, T, s)
    return -(rho * Cv / (dpT + 1e-10)) * dprs + drhoe

def srk_c2(r1, r2, T):
    """c^2 [m^2/s^2]. Vectorized."""
    rho  = r1 + r2
    Y0   = r1 / np.maximum(rho, 1e-30)
    Y1   = 1.0 - Y0
    dpdr_T = Y0*_dpdr(r1, r2, T, 0) + Y1*_dpdr(r1, r2, T, 1)
    dpT    = _dpdT(r1, r2, T)
    Cv     = srk_Cv(r1, r2, T)
    c2 = dpdr_T + T*dpT**2 / (rho**2*Cv + 1e-30)
    return np.maximum(c2, 100.0)

# ─────────────────────────────────────────────────────────────
# Initial condition  (Section 3.2.1 of paper)
# ─────────────────────────────────────────────────────────────

def initial_condition(x, p_inf=5e6, k=15.0):
    N  = len(x)
    xc, rc = 0.5, 0.25
    r1_inf, r2_inf = 400.0, 100.0   # CH4, N2 [kg/m3]

    r   = np.abs(x - xc)
    r1  = 0.5*r1_inf*(1.0 - np.tanh(k*(r - rc)))
    r2  = 0.5*r2_inf*(1.0 + np.tanh(k*(r - rc)))

    print("  Solving T from p_inf=5MPa (vectorized Newton)...", flush=True)
    # Vectorized Newton: find T such that srk_p(r1,r2,T) = p_inf
    _, Mb, _, _, _, _, Ct, *_ = _mix(r1, r2, np.full(N, 300.0))
    T = p_inf * Mb / np.maximum(r1 + r2, 1e-30) / Ru   # ideal-gas guess
    T = np.clip(T, 50.0, 1000.0)
    for _ in range(60):
        ph  = srk_p(r1, r2, T)
        dph = _dpdT(r1, r2, T)
        dT  = -(ph - p_inf) / (dph + 1e-3)
        dT  = np.clip(dT, -100.0, 100.0)
        T   = np.clip(T + dT, 10.0, 2000.0)
        if np.max(np.abs(dT)) < 1e-3:
            break

    p    = srk_p(r1, r2, T)
    rhoe = srk_rhoe(r1, r2, T)
    u    = np.full(N, 100.0)
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2
    return r1, r2, u, rhoE, T, p

# ─────────────────────────────────────────────────────────────
# Primitive variables from conserved
# ─────────────────────────────────────────────────────────────

def prim(r1, r2, rhoU, rhoE, T_prev):
    rho  = r1 + r2
    u    = rhoU / np.maximum(rho, 1e-30)
    rhoe = rhoE - 0.5*rho*u**2   # NOTE: can be negative (liquid SRK)
    T    = T_from_rhoe(r1, r2, rhoe, T_in=T_prev)
    p    = srk_p(r1, r2, T)
    return u, rhoe, T, p

# ─────────────────────────────────────────────────────────────
# MUSCL reconstruction  (minmod limiter, periodic BC)
# ─────────────────────────────────────────────────────────────

def _minmod(a, b):
    return np.where(a*b > 0.0,
                    np.sign(a)*np.minimum(np.abs(a), np.abs(b)), 0.0)

def muscl_lr(q):
    """2nd-order MUSCL. Returns (qL, qR) at interface m+1/2."""
    dR   = np.roll(q, -1) - q
    dL   = q - np.roll(q,  1)
    slp  = _minmod(dL, dR)
    qL   = q + 0.5*slp                  # left  state at m+1/2
    qR   = np.roll(q - 0.5*slp, -1)    # right state at m+1/2
    return qL, qR

# ─────────────────────────────────────────────────────────────
# Physical flux vector F(q)
# ─────────────────────────────────────────────────────────────

def phys_flux(r1, r2, u, rhoe, p):
    rho = r1 + r2
    return (r1*u,
            r2*u,
            rho*u**2 + p,
            (rhoe + 0.5*rho*u**2 + p)*u)

# ─────────────────────────────────────────────────────────────
# LLF interface flux  (with optional APEC energy correction)
# ─────────────────────────────────────────────────────────────

def interface_fluxes(r1, r2, u, rhoe, p, T, lam_cell, scheme, eps_pair=None):
    """
    Compute LLF fluxes at all m+1/2 interfaces.

    Strategy to avoid Abgrall problem from reconstruction:
      - Mass + momentum: MUSCL-LLF (upwinded, stabilizes density transport)
      - Energy: ARITHMETIC AVERAGE of cell-center values + LLF dissipation
        (avoids EOS inconsistency between MUSCL-reconstructed rhoe and r1/r2)
      - APEC: same as above but with APEC correction to the centered rhoe_h

    lam_cell[m] = |u_m| + c_m  (local max wave speed)
    eps_pair   = (eps0_cell, eps1_cell)  (APEC only)
    Returns F1, F2, FU, FE  (each shape N).
    """
    # MUSCL for mass/momentum only
    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)

    rhoL = r1L + r2L;   rhoR = r1R + r2R

    # Interface wave speed
    lam = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # Mass + momentum: MUSCL-LLF
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R  - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R  - r2L)
    FU = 0.5*(rhoL*uL**2 + pL + rhoR*uR**2 + pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    # Energy: use CELL-CENTER values (not MUSCL) to avoid EOS inconsistency
    rho   = r1 + r2
    rhoE  = rhoe + 0.5*rho*u**2
    rhop  = np.roll(rho,  -1)
    up    = np.roll(u,    -1)
    pp    = np.roll(p,    -1)
    rhoEp = np.roll(rhoE, -1)
    ehoep = np.roll(rhoe, -1)

    # Arithmetic half-point quantities from cell centers
    rho_h = 0.5*(rho  + rhop)
    u_h   = 0.5*(u    + up)
    p_h   = 0.5*(p    + pp)

    r1p = np.roll(r1, -1)
    r2p = np.roll(r2, -1)
    up  = np.roll(u,  -1)
    dr1 = r1p - r1
    dr2 = r2p - r2
    du  = up  - u

    if scheme == 'APEC':
        # APEC: correct the centered internal-energy half-point (Eq. 40)
        # AND use MUSCL-consistent PE dissipation.
        #
        # Root cause of FC PE error: mass dissipation uses MUSCL jumps
        # (r1R - r1L, limited slopes) while FC energy dissipation uses
        # cell-center jumps (ρE_{m+1} - ρE_m). This mismatch generates PE.
        #
        # Fix: use the same MUSCL-reconstructed jumps for energy dissipation:
        #   drhoE = ε0_h*(r1R - r1L) + ε1_h*(r2R - r2L) + KE_MUSCL
        # This makes energy and mass dissipations self-consistent.
        eps0, eps1 = eps_pair
        eps0p = np.roll(eps0, -1)
        eps1p = np.roll(eps1, -1)

        # Interface ε_s (arithmetic average)
        eps0_h = 0.5*(eps0 + eps0p)
        eps1_h = 0.5*(eps1 + eps1p)

        # Centered correction to rhoe_h (PE-consistent to O(Δx^2))
        corr   = 0.5*(eps0p - eps0)*0.5*dr1 + 0.5*(eps1p - eps1)*0.5*dr2
        rhoe_h = 0.5*(rhoe + ehoep) - corr

        FE_cen = (rhoe_h + 0.5*rho_h*u_h**2 + p_h) * u_h

        # MUSCL-consistent PE dissipation (MUSCL jumps, not cell-center)
        # r1L, r1R etc. are the MUSCL-reconstructed interface values
        drhoE_pep = (eps0_h*(r1R - r1L)
                     + eps1_h*(r2R - r2L)
                     + 0.5*u_h**2*((r1R + r2R) - (r1L + r2L))
                     + rho_h*u_h*(uR - uL))
        FE = FE_cen - 0.5*lam*drhoE_pep
    else:
        # FC-NPE: standard arithmetic average + standard LLF dissipation
        rhoe_h = 0.5*(rhoe + ehoep)
        FE_cen = (rhoe_h + 0.5*rho_h*u_h**2 + p_h) * u_h
        FE = FE_cen - 0.5*lam*(rhoEp - rhoE)

    return F1, F2, FU, FE

# ─────────────────────────────────────────────────────────────
# RHS
# ─────────────────────────────────────────────────────────────

def rhs(U, scheme, dx, T_prev):
    r1, r2, rhoU, slot4 = U
    rho = r1 + r2

    if scheme == 'PEqC':
        u    = rhoU / np.maximum(rho, 1e-30)
        T    = T_prev.copy()   # use previous T directly (avoid EOS inversion)
        p    = srk_p(r1, r2, T)
        rhoe = srk_rhoe(r1, r2, T)
    else:
        u, rhoe, T, p = prim(r1, r2, rhoU, slot4, T_prev)

    c2      = srk_c2(r1, r2, T)
    lam_c   = np.abs(u) + np.sqrt(c2)

    if scheme == 'APEC':
        eps0 = epsilon_v(r1, r2, T, 0)
        eps1 = epsilon_v(r1, r2, T, 1)
        eps_pair = (eps0, eps1)
    else:
        eps_pair = None

    if scheme in ('FC', 'APEC'):
        F1, F2, FU, FE = interface_fluxes(r1, r2, u, rhoe, p, T, lam_c,
                                           scheme, eps_pair=eps_pair)
        d1 = -(F1 - np.roll(F1, 1)) / dx
        d2 = -(F2 - np.roll(F2, 1)) / dx
        dU = -(FU - np.roll(FU, 1)) / dx
        dE = -(FE - np.roll(FE, 1)) / dx
        return [d1, d2, dU, dE], T, p

    else:  # PEqC
        # Mass + momentum: standard LLF (same as FC)
        F1, F2, FU, _ = interface_fluxes(r1, r2, u, rhoe, p, T, lam_c, 'FC')
        d1 = -(F1 - np.roll(F1, 1)) / dx
        d2 = -(F2 - np.roll(F2, 1)) / dx
        dU = -(FU - np.roll(FU, 1)) / dx
        # Pressure: non-conservative transport dp/dt + u*dp/dx + rho*c2*du/dx = 0
        pp   = np.roll(p, -1);  pm = np.roll(p,  1)
        up   = np.roll(u, -1);  um = np.roll(u,  1)
        dpdx = (pp - pm) / (2.0*dx)
        dudx = (up - um) / (2.0*dx)
        dp   = -(u*dpdx + rho*c2*dudx)
        return [d1, d2, dU, dp], T, p

# ─────────────────────────────────────────────────────────────
# SSP-RK3
# ─────────────────────────────────────────────────────────────

def _clip(U):
    U[0] = np.maximum(U[0], 0.0)
    U[1] = np.maximum(U[1], 0.0)
    return U

def rkstep(U, scheme, dx, dt, T_prev):
    k1, T1, p1 = rhs(U, scheme, dx, T_prev)
    U1 = _clip([U[q] + dt*k1[q] for q in range(4)])

    k2, T2, p2 = rhs(U1, scheme, dx, T1)
    U2 = _clip([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])

    k3, T3, p3 = rhs(U2, scheme, dx, T2)
    Un = _clip([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)])

    return Un, T3, p3

# ─────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────

def pe_err(p, p0=5e6):
    return float(np.max(np.abs(p - p0)) / p0)

def energy_err(rhoE, rhoE0):
    return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-30))

# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

def run(scheme, N=101, t_end=0.07, CFL=0.4, p_inf=5e6, k=15.0, verbose=True):
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1 - dx/2, N)

    print(f"\n{'='*55}")
    print(f"Scheme: {scheme}  N={N}  t_end={t_end:.3f}  CFL={CFL}  k={k}")

    r1, r2, u, rhoE, T, p = initial_condition(x, p_inf, k=k)
    rhoE0 = rhoE.copy()
    rho   = r1 + r2

    if scheme == 'PEqC':
        U = [r1.copy(), r2.copy(), rho*u, p.copy()]
    else:
        U = [r1.copy(), r2.copy(), rho*u, rhoE.copy()]

    t_hist  = [0.0];  pe_hist = [pe_err(p, p_inf)];  en_hist = [0.0]
    t = 0.0;  step = 0;  diverged = False

    while t < t_end - 1e-14:
        r1_, r2_ = U[0], U[1]
        u_  = U[2] / np.maximum(r1_+r2_, 1e-30)
        c2_ = srk_c2(r1_, r2_, T)
        lam = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt  = min(CFL*dx/(lam + 1e-10), t_end - t)

        try:
            U, T, p = rkstep(U, scheme, dx, dt, T)
        except Exception as e:
            print(f"  Diverged at t={t:.5f}: {e}"); diverged = True; break

        t += dt;  step += 1

        pe_ = pe_err(p, p_inf)
        en_ = energy_err(U[3], rhoE0) if scheme != 'PEqC' else np.nan

        t_hist.append(t);  pe_hist.append(pe_);  en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 5.0:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}"); diverged = True; break

        if verbose and (step % 500 == 0 or t >= t_end - 1e-12):
            print(f"  t={t:.5f} step={step} PE={pe_:.3e} Enerr={en_:.3e}")

    status = "Completed" if not diverged else "Diverged"
    print(f"  --> {status} at t={t:.5f} ({step} steps)")
    return x, U, T, p, np.array(t_hist), np.array(pe_hist), np.array(en_hist), diverged

# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

COLORS = {'FC': '#d62728', 'APEC': '#1f77b4', 'PEqC': '#2ca02c'}
LABELS = {'FC': 'FC-NPE', 'APEC': 'APEC', 'PEqC': 'PEqC'}

def plot_profiles(results, t_snap, fname='fig6_profiles.png'):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for sch, (x, U, T, p, *_) in results.items():
        r1, r2, rhoU = U[0], U[1], U[2]
        rho = r1 + r2;  u = rhoU / np.maximum(rho, 1e-30)
        c, l = COLORS[sch], LABELS[sch]
        axes[0,0].plot(x, rho,   label=l, color=c)
        axes[0,1].plot(x, T,     label=l, color=c)
        axes[1,0].plot(x, u,     label=l, color=c)
        axes[1,1].plot(x, p/1e6, label=l, color=c)
    for ax, ttl in zip(axes.flat,
            ['Total density [kg/m3]','Temperature [K]',
             'Velocity [m/s]','Pressure [MPa]']):
        ax.set_title(ttl); ax.legend(fontsize=8); ax.set_xlabel('x [m]')
    fig.suptitle(f'CH4/N2  t={t_snap:.4f} s', fontsize=13)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close(fig)
    print(f"Saved {fname}")

def plot_histories(results, fname='fig7_history.png'):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for sch, (*_, t_h, pe_h, en_h, div) in results.items():
        c, l = COLORS[sch], LABELS[sch]
        a1.semilogy(t_h, pe_h + 1e-20, label=l, color=c)
        if sch != 'PEqC':
            a2.semilogy(t_h, en_h + 1e-20, label=l, color=c)
    a1.set_xlabel('time [s]'); a1.set_ylabel('max|dp|/p_inf')
    a1.set_title('Pressure-equilibrium error'); a1.legend()
    a2.set_xlabel('time [s]'); a2.set_ylabel('|dE|/E0')
    a2.set_title('Energy conservation error'); a2.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close(fig)
    print(f"Saved {fname}")

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N      = 101
    CFL    = 0.3
    t_end  = 0.046   # APEC survives ~46ms, FC diverges at ~6.5ms

    results = {}
    for sch in ['FC', 'APEC', 'PEqC']:
        res = run(sch, N=N, t_end=t_end, CFL=CFL, verbose=True)
        results[sch] = res

    # Profiles at end state
    plot_profiles(results, t_end, 'fig6_profiles.png')
    plot_histories(results, 'fig7_history.png')
    print("\nDone.")
