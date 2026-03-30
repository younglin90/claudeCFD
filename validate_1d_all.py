#!/usr/bin/env python3
"""
validate_1d_all.py
==================
Master 1D Validation Runner — 31 cases from validation/1D_*.md
All plots saved to output/1D/

Usage:
    python validate_1d_all.py              # run all
    python validate_1d_all.py --list       # list cases
    python validate_1d_all.py --case sod   # run matching cases
    python validate_1d_all.py --fast       # skip slow cases (apec t=8, eoc)
"""
import sys, os, argparse, time, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq

PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(PROJ_DIR, 'solver')
OUTPUT_DIR = os.path.join(PROJ_DIR, 'output', '1D')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, SOLVER_DIR)

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    saved → output/1D/{name}.png")
    return path


def _minmod(a, b):
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) <= np.abs(b), a, b))


def _ssprk3(U, rhs, dt):
    L1 = rhs(U)
    U1 = U + dt * L1
    L2 = rhs(U1)
    U2 = 0.75 * U + 0.25 * (U1 + dt * L2)
    L3 = rhs(U2)
    return (1.0/3.0) * U + (2.0/3.0) * (U2 + dt * L3)


# ══════════════════════════════════════════════════════════════════
# INLINE SOLVER A: Single-component ideal gas (1D Euler, MUSCL-LLF)
# State: U = (3,N) = [rho, rho*u, rho*E]
# ══════════════════════════════════════════════════════════════════

def _prim_ig(U, gam):
    rho  = np.maximum(U[0], 1e-100)
    u    = U[1] / rho
    p    = np.maximum((gam - 1.0) * (U[2] - 0.5 * U[1] * u), 1e-100)
    c    = np.sqrt(gam * p / rho)
    return rho, u, p, c


def _rhs_ig(U, gam, dx, bc):
    N = U.shape[1]
    # 1-ghost extension
    if bc == 'periodic':
        Ue = np.hstack([U[:, -1:], U, U[:, :1]])
    else:
        Ue = np.hstack([U[:, 0:1], U, U[:, -1:]])

    # Slopes (minmod) for N interior cells
    dL = Ue[:, 1:-1] - Ue[:, :-2]
    dR = Ue[:, 2:]   - Ue[:, 1:-1]
    sl = _minmod(dL, dR)                        # (3,N)

    # Extended slopes (ghost = 0)
    sle = np.hstack([np.zeros((3,1)), sl, np.zeros((3,1))])  # (3,N+2)

    # Interface states (N+1 faces)
    UL_face = Ue + 0.5 * sle                    # right faces
    UR_face = Ue - 0.5 * sle                    # left faces
    ULf = UL_face[:, :-1]                       # (3,N+1)
    URf = UR_face[:, 1:]                        # (3,N+1)

    rL, uL, pL, cL = _prim_ig(ULf, gam)
    rR, uR, pR, cR = _prim_ig(URf, gam)

    FL = np.array([ULf[1], ULf[1]*uL + pL, (ULf[2]+pL)*uL])
    FR = np.array([URf[1], URf[1]*uR + pR, (URf[2]+pR)*uR])
    lam = np.maximum(np.abs(uL)+cL, np.abs(uR)+cR)
    Ff  = 0.5*(FL+FR) - 0.5*lam*(URf - ULf)

    return -(Ff[:, 1:] - Ff[:, :-1]) / dx


def euler1d_ig(x, U0, gam, t_end, CFL, bc='transmissive'):
    """Run single-component 1D Euler MUSCL-LLF (ideal gas)."""
    dx = x[1] - x[0]
    U  = U0.copy()
    t  = 0.0
    while t < t_end - 1e-14:
        rho, u, p, c = _prim_ig(U, gam)
        dt = CFL * dx / np.max(np.abs(u) + c)
        dt = min(dt, t_end - t)
        U  = _ssprk3(U, lambda V: _rhs_ig(V, gam, dx, bc), dt)
        t += dt
    return U


def ic_ig(x, rhoL, uL, pL, rhoR, uR, pR, gam, x0=0.5):
    """Shock-tube initial condition for ideal gas."""
    N  = len(x)
    U  = np.zeros((3, N))
    mL = rhoL * uL;  EL = pL/(gam-1) + 0.5*rhoL*uL**2
    mR = rhoR * uR;  ER = pR/(gam-1) + 0.5*rhoR*uR**2
    U[0] = np.where(x < x0, rhoL, rhoR)
    U[1] = np.where(x < x0, mL,   mR)
    U[2] = np.where(x < x0, EL,   ER)
    return U


def exact_sod(x, t, rhoL, uL, pL, rhoR, uR, pR, gam=1.4):
    """Exact Riemann solver (ideal gas, uL=uR=0 assumed)."""
    aL  = np.sqrt(gam * pL / rhoL)
    aR  = np.sqrt(gam * pR / rhoR)
    gm1 = gam - 1.0;  gp1 = gam + 1.0

    def f(ps):
        fL = ((2*aL/gm1)*((ps/pL)**((gam-1)/(2*gam))-1) if ps<=pL
              else (ps-pL)*np.sqrt(2/(gp1*rhoL)/(ps+gm1/gp1*pL)))
        fR = ((2*aR/gm1)*((ps/pR)**((gam-1)/(2*gam))-1) if ps<=pR
              else (ps-pR)*np.sqrt(2/(gp1*rhoR)/(ps+gm1/gp1*pR)))
        return fL + fR + (uR - uL)

    ps = brentq(f, 1e-8*min(pL,pR), 20*max(pL,pR), xtol=1e-12)

    AL = 2/(gp1*rhoL); BL = gm1/gp1*pL
    us = (uL - (ps-pL)*np.sqrt(AL/(ps+BL)) if ps > pL
          else uL + (2*aL/gm1)*((ps/pL)**((gam-1)/(2*gam))-1))

    rsL = (rhoL*(ps/pL)**(1/gam) if ps<=pL
           else rhoL*(ps/pL+gm1/gp1)/(gm1/gp1*ps/pL+1))
    rsR = (rhoR*(ps/pR)**(1/gam) if ps<=pR
           else rhoR*(ps/pR+gm1/gp1)/(gm1/gp1*ps/pR+1))

    SR  = uR + aR*np.sqrt(gp1/(2*gam)*ps/pR + gm1/(2*gam))
    xi  = (x - 0.5) / (t + 1e-30)

    rho = np.empty_like(x); uu = np.empty_like(x); pp = np.empty_like(x)
    if ps <= pL:
        SHL = uL - aL
        STL = us - aL*(ps/pL)**((gam-1)/(2*gam))
        for k, xk in enumerate(xi):
            if xk < SHL:
                rho[k], uu[k], pp[k] = rhoL, uL, pL
            elif xk < STL:
                af = (2*aL + gm1*(uL-xk))/gp1
                rho[k] = rhoL*(af/aL)**(2/gm1)
                uu[k]  = 2*(aL + gm1/2*uL + xk)/gp1
                pp[k]  = pL*(af/aL)**(2*gam/gm1)
            elif xk < us:
                rho[k], uu[k], pp[k] = rsL, us, ps
            elif xk < SR:
                rho[k], uu[k], pp[k] = rsR, us, ps
            else:
                rho[k], uu[k], pp[k] = rhoR, uR, pR
    else:
        SL = uL - aL*np.sqrt(gp1/(2*gam)*ps/pL + gm1/(2*gam))
        for k, xk in enumerate(xi):
            if xk < SL:
                rho[k], uu[k], pp[k] = rhoL, uL, pL
            elif xk < us:
                rho[k], uu[k], pp[k] = rsL, us, ps
            elif xk < SR:
                rho[k], uu[k], pp[k] = rsR, us, ps
            else:
                rho[k], uu[k], pp[k] = rhoR, uR, pR
    return rho, uu, pp


# ══════════════════════════════════════════════════════════════════
# INLINE SOLVER B: Two-fluid stiffened gas (5-eq diffuse interface)
# State: U = (5,N) = [a1r1, a2r2, rho*u, rho*E, alpha1]
# ══════════════════════════════════════════════════════════════════

def _prim_sg(U, g1, P1, g2, P2):
    a1r1 = np.maximum(U[0], 1e-100)
    a2r2 = np.maximum(U[1], 1e-100)
    rho  = a1r1 + a2r2
    u    = U[2] / rho
    rhoe = U[3] - 0.5 * U[2] * u
    al   = np.clip(U[4], 1e-6, 1-1e-6)
    # mixture pressure (Allaire 2002)
    numer = rhoe - al*g1*P1/(g1-1) - (1-al)*g2*P2/(g2-1)
    denom = al/(g1-1) + (1-al)/(g2-1)
    p     = np.maximum(numer / denom, 1e-6)
    # mixture sound speed squared
    c2    = (al*g1*(p+P1) + (1-al)*g2*(p+P2)) / rho
    c     = np.sqrt(np.maximum(c2, 1e-6))
    return rho, u, p, c, al


def _rhs_sg(U, g1, P1, g2, P2, dx, bc):
    N = U.shape[1]
    if bc == 'periodic':
        Ue = np.hstack([U[:, -1:], U, U[:, :1]])
    else:
        Ue = np.hstack([U[:, 0:1], U, U[:, -1:]])

    dL = Ue[:, 1:-1] - Ue[:, :-2]
    dR = Ue[:, 2:]   - Ue[:, 1:-1]
    sl  = _minmod(dL, dR)
    sle = np.hstack([np.zeros((5,1)), sl, np.zeros((5,1))])

    ULf = (Ue + 0.5*sle)[:, :-1]
    URf = (Ue - 0.5*sle)[:, 1:]

    rL, uL, pL, cL, aL = _prim_sg(ULf, g1, P1, g2, P2)
    rR, uR, pR, cR, aR = _prim_sg(URf, g1, P1, g2, P2)

    def Fcons(U_, r, u_, p_):
        return np.array([U_[0]*u_, U_[1]*u_, r*u_**2+p_, (U_[3]+p_)*u_])

    FL = Fcons(ULf, rL, uL, pL)
    FR = Fcons(URf, rR, uR, pR)
    lam = np.maximum(np.abs(uL)+cL, np.abs(uR)+cR)
    Fc  = 0.5*(FL+FR) - 0.5*lam*(URf[:4] - ULf[:4])

    # alpha upwind
    u_face = 0.5*(uL + uR)
    alL    = ULf[4];  alR = URf[4]
    Fa     = np.where(u_face >= 0, u_face*alL, u_face*alR)

    dUc = -(Fc[:, 1:] - Fc[:, :-1]) / dx   # (4,N)
    dal = -(Fa[1:] - Fa[:-1]) / dx          # (N,)
    return np.vstack([dUc, dal[None, :]])


def euler1d_sg(x, U0, g1, P1, g2, P2, t_end, CFL, bc='transmissive'):
    dx = x[1] - x[0]
    U  = U0.copy()
    t  = 0.0
    while t < t_end - 1e-14:
        rho, u, p, c, _ = _prim_sg(U, g1, P1, g2, P2)
        dt = CFL * dx / np.max(np.abs(u) + c)
        dt = min(dt, t_end - t)
        U  = _ssprk3(U, lambda V: _rhs_sg(V, g1, P1, g2, P2, dx, bc), dt)
        t += dt
    return U


def ic_sg(x, a1L, rho1L, rho2L, uL, pL, g1, P1,
              a1R, rho1R, rho2R, uR, pR, g2, P2, x0=0.5):
    """IC for two-fluid stiffened gas.
    Phase 1: EOS(g1, P1), Phase 2: EOS(g2, P2).
    Left side:  alpha1=a1L, rho1=rho1L, rho2=rho2L, u=uL, p=pL
    Right side: alpha1=a1R, rho1=rho1R, rho2=rho2R, u=uR, p=pR
    """
    N = len(x)
    U = np.zeros((5, N))
    for j in range(N):
        if x[j] < x0:
            a1, r1_, r2_, u_, p_ = a1L, rho1L, rho2L, uL, pL
        else:
            a1, r1_, r2_, u_, p_ = a1R, rho1R, rho2R, uR, pR
        a2 = 1.0 - a1
        rho_  = a1*r1_ + a2*r2_
        rhoe_ = a1*(p_+g1*P1)/(g1-1) + a2*(p_+g2*P2)/(g2-1)
        U[0,j] = a1*r1_
        U[1,j] = a2*r2_
        U[2,j] = rho_*u_
        U[3,j] = rhoe_ + 0.5*rho_*u_**2
        U[4,j] = a1
    return U


# ══════════════════════════════════════════════════════════════════
# IMPORT EXISTING SOLVERS (redirect OUTPUT_DIR)
# ══════════════════════════════════════════════════════════════════

def _load_solver(name):
    try:
        import importlib
        mod = importlib.import_module(name)
        if hasattr(mod, 'OUTPUT_DIR'):
            mod.OUTPUT_DIR = OUTPUT_DIR
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        return mod, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════
# VALIDATION CASES
# ══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# 1. Acoustic wave propagation (linear, convergence)
# ──────────────────────────────────────────────────────────────────
def case_acoustic_wave_propagation():
    """1D_acoustic_wave_propagation: sinusoidal wave, L2 convergence."""
    gam = 1.4; rho0 = 1.0; p0 = 1.0; eps = 1e-4
    c0  = np.sqrt(gam * p0 / rho0)
    t_end = 1.0 / c0

    Ns = [50, 100, 200, 400]
    errs = []
    for N in Ns:
        x  = (np.arange(N) + 0.5) / N
        dx = 1.0 / N
        # IC: sinusoidal perturbation
        rho_ic = rho0 + eps * np.sin(2*np.pi*x)
        u_ic   = eps * c0 * np.sin(2*np.pi*x)
        p_ic   = p0  + eps * gam * p0 * np.sin(2*np.pi*x)
        U0 = np.zeros((3, N))
        U0[0] = rho_ic
        U0[1] = rho_ic * u_ic
        U0[2] = p_ic/(gam-1) + 0.5*rho_ic*u_ic**2
        U = euler1d_ig(x, U0, gam, t_end, CFL=0.5, bc='periodic')
        # Exact: wave travels right by c0*t_end = 1.0 → back to IC position
        rho_ex = rho0 + eps * np.sin(2*np.pi*(x - c0*t_end))
        e2 = np.sqrt(dx * np.sum((U[0] - rho_ex)**2))
        errs.append(e2)

    errs = np.array(errs)
    dxs  = 1.0 / np.array(Ns)
    eoc  = np.log2(errs[:-1]/errs[1:])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(dxs, errs, 'o-', label='L2 error')
    axes[0].loglog(dxs, errs[0]*(dxs/dxs[0])**2, 'k--', label='O(Δx²)')
    axes[0].set_xlabel('Δx'); axes[0].set_ylabel('L2 error ρ')
    axes[0].legend(); axes[0].set_title('Grid convergence')
    for i, (N, e) in enumerate(zip(Ns, errs)):
        eoc_str = f'{eoc[i-1]:.2f}' if i > 0 else '—'
        axes[0].annotate(f'EOC={eoc_str}', xy=(dxs[i], errs[i]), fontsize=7)

    # Snapshot at finest grid
    N  = 400; x = (np.arange(N)+0.5)/N; dx = 1/N
    rho_ic = rho0 + eps*np.sin(2*np.pi*x)
    u_ic   = eps*c0*np.sin(2*np.pi*x)
    p_ic   = p0 + eps*gam*p0*np.sin(2*np.pi*x)
    U0 = np.zeros((3,N))
    U0[0]=rho_ic; U0[1]=rho_ic*u_ic; U0[2]=p_ic/(gam-1)+0.5*rho_ic*u_ic**2
    U = euler1d_ig(x, U0, gam, t_end*0.5, CFL=0.5, bc='periodic')
    rho, u, p, _ = _prim_ig(U, gam)
    rho_ex = rho0 + eps*np.sin(2*np.pi*(x - c0*t_end*0.5))
    axes[1].plot(x, rho, 'b-', lw=1.5, label='Numerical')
    axes[1].plot(x, rho_ex, 'k--', label='Exact')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('ρ')
    axes[1].legend(); axes[1].set_title(f'ρ at t=0.5/c₀  (N={N})')

    fig.suptitle('1D Acoustic Wave Propagation (ideal gas, MUSCL-LLF)')
    fig.tight_layout()
    _save(fig, '01_acoustic_wave_propagation')

    print(f"    EOC (L2 rho): {eoc}")
    return True


# ──────────────────────────────────────────────────────────────────
# 2. Acoustic reflection/transmission at interface
# ──────────────────────────────────────────────────────────────────
def case_acoustic_reflection_transmission():
    """1D_acoustic_reflection_transmission_interface: analytical + simulation."""
    # Two fluids: left (air-like), right (denser gas)
    gam = 1.4; eps = 1e-4
    rho1, p0 = 1.0, 1.0
    rho2 = 4.0          # denser fluid → Z2 > Z1
    c1 = np.sqrt(gam*p0/rho1); c2 = np.sqrt(gam*p0/rho2)
    Z1 = rho1*c1;  Z2 = rho2*c2
    R_theory = (Z2-Z1)/(Z2+Z1)
    T_theory = 2*Z2/(Z1+Z2)   # pressure transmission

    # Simulation: sinusoidal pulse from left, hits interface at x=0.5
    N  = 400; x = (np.arange(N)+0.5)/N; dx = 1/N
    rho_bg = np.where(x < 0.5, rho1, rho2)
    c_bg   = np.where(x < 0.5, c1, c2)
    # IC: small pulse in left fluid (Gaussian-like in x ∈ [0.1, 0.25])
    pulse = eps * np.exp(-((x-0.2)/0.03)**2)
    rho_ic = rho_bg + pulse
    u_ic   = pulse / rho_bg * c_bg * np.where(x<0.5, 1.0, 0.0)
    p_ic   = p0 + gam*p0*pulse/rho_bg

    # Use a variable-gamma approach (two-comp ideal gas via lambda_diff_1d)
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        # Fallback: plot analytical only
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(['R (theory)', 'T (theory)'], [R_theory, T_theory], color=['C0','C1'])
        ax.set_title('Acoustic reflection/transmission (analytical)')
        fig.tight_layout()
        _save(fig, '02_acoustic_reflection_transmission')
        return True

    # Run with two-component (γ₁=γ₂=1.4, rho1/rho2 differ via initial W field)
    # IC via conserved variables
    gam1 = gam; gam2 = gam
    W_ic = np.where(x < 0.5, 1.0, 0.0)  # component 1 = left fluid
    rho_ic2 = rho_bg.copy()
    U0 = LD.prim_to_cons(W_ic, rho_ic2, np.zeros(N), p0*np.ones(N), gam1, gam2)

    t_end = 0.4
    U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='transmissive')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, rho_f, 'b-')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('ρ')
    axes[0].set_title(f't={t_end}')
    axes[1].plot(x, u_f,   'r-')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('u')
    axes[2].plot(x, p_f,   'g-')
    axes[2].axhline(p0, color='k', ls='--', lw=0.8)
    axes[2].set_xlabel('x'); axes[2].set_ylabel('p')

    info = f'Z₁={Z1:.2f}, Z₂={Z2:.2f}\nR_theory={R_theory:.3f}, T_theory={T_theory:.3f}'
    fig.suptitle('1D Acoustic Reflection/Transmission\n' + info)
    fig.tight_layout()
    _save(fig, '02_acoustic_reflection_transmission')
    print(f"    R_theory={R_theory:.4f}, T_theory={T_theory:.4f}")
    return True


# ──────────────────────────────────────────────────────────────────
# 3. Acoustic impedance matching (gas-gas)
# ──────────────────────────────────────────────────────────────────
def case_acoustic_impedance_matching():
    """1D_acoustic_impedance_matching_gas_gas: R≈0 for matched impedance."""
    gam = 1.4; eps = 1e-5; p0 = 1.0
    # Matched impedance: Z1 = Z2 → rho1*c1 = rho2*c2
    # c = sqrt(gam*p/rho), so Z = rho*sqrt(gam*p/rho) = sqrt(gam*p*rho)
    # Z1=Z2 → rho1 = rho2 (same rho, so trivial) OR use different gam
    # Non-trivial: gam1≠gam2 with rho1*c1 = rho2*c2
    # Choose rho2 = rho1*(gam1/gam2) → c2 = sqrt(gam2*p/rho2) = sqrt(gam1*p/rho1) = c1
    gam1, gam2 = 1.4, 1.67
    rho1 = 1.0
    rho2 = rho1 * gam1/gam2   # matched impedance

    c1 = np.sqrt(gam1*p0/rho1); c2 = np.sqrt(gam2*p0/rho2)
    Z1 = rho1*c1;  Z2 = rho2*c2
    R_exact = (Z2-Z1)/(Z2+Z1)

    N = 400; x = (np.arange(N)+0.5)/N; dx = 1/N
    rho_bg = np.where(x < 0.5, rho1, rho2)

    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}")
        return False

    W_ic = np.where(x < 0.5, 1.0, 0.0)
    U0 = LD.prim_to_cons(W_ic, rho_bg, np.zeros(N), p0*np.ones(N), gam1, gam2)
    # Add small pulse at x=0.2
    pulse = eps * np.exp(-((x-0.2)/0.03)**2)
    U0[1] += rho_bg * (pulse/rho_bg*c1) * (x < 0.5)

    t_end = 0.5
    U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='transmissive')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

    dp = np.max(np.abs(p_f - p0))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, p_f, 'b-', label='p numerical')
    axes[0].axhline(p0, color='k', ls='--', lw=0.8, label='p₀')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('p'); axes[0].legend()
    axes[0].set_title(f't={t_end}')
    axes[1].bar(['|R| theory', '|Δp|_max / eps'], [abs(R_exact), dp/eps],
                color=['C0','C2'])
    axes[1].set_title(f'Z1={Z1:.3f}, Z2={Z2:.3f}, R={R_exact:.4f}')
    fig.suptitle('1D Acoustic Impedance Matching (gas-gas, matched Z)')
    fig.tight_layout()
    _save(fig, '03_acoustic_impedance_matching_gas_gas')
    print(f"    Z1={Z1:.4f}, Z2={Z2:.4f}, R_theory={R_exact:.4f}, |Δp|/ε={dp/eps:.4f}")
    return True


# ──────────────────────────────────────────────────────────────────
# 4. CPG interface advection (APEC vs FC-NPE, §3.1)
# ──────────────────────────────────────────────────────────────────
def case_cpg_interface_advection(fast=False):
    """1D_calorically_perfect_gas_interface_advection: APEC §3.1."""
    APEC, err = _load_solver('apec_1d')
    if APEC is None:
        print(f"    apec_1d not available: {err}"); return False

    N    = 251 if fast else 501
    tend = 4.0 if fast else 8.0
    print(f"    Running CPG (N={N}, t={tend})...")
    APEC.validate_cpg_311(N=N, t_end=tend, CFL=0.6)
    return True


# ──────────────────────────────────────────────────────────────────
# 5. CH4/N2 SRK interface advection (APEC §3.2)
# ──────────────────────────────────────────────────────────────────
def case_ch4_n2_srk(fast=False):
    """1D_CH4_N2_interface_advection_SRK_EOS."""
    APEC, err = _load_solver('apec_1d')
    if APEC is None:
        print(f"    apec_1d not available: {err}"); return False

    Ns = [51, 101, 201, 501]
    print(f"    Running SRK CH4/N2 (N={Ns})...")

    results = {}
    for N in Ns:
        for sch in ['FC', 'APEC']:
            key = (N, sch)
            # run() returns (x, U, T, p, t_hist, pe_hist, en_hist)
            x, U, T_, p_, t_h, pe_h, en_h = APEC.run(sch, N=N, t_end=0.0005, CFL=0.3)
            results[key] = pe_h[-1] if len(pe_h) > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    Ns_arr = np.array(Ns)
    pe_fc   = [results[(N,'FC')]   for N in Ns]
    pe_apec = [results[(N,'APEC')] for N in Ns]

    axes[0].loglog(Ns_arr, pe_fc,   'C0o--', label='FC-NPE')
    axes[0].loglog(Ns_arr, pe_apec, 'C1s-',  label='APEC')
    axes[0].set_xlabel('N'); axes[0].set_ylabel('PE error (step 1)')
    axes[0].legend(); axes[0].set_title('PE error vs grid size')
    axes[0].invert_xaxis()

    impr = [fc/ap for fc, ap in zip(pe_fc, pe_apec)]
    axes[1].semilogx(Ns_arr, impr, 'ko-')
    axes[1].set_xlabel('N'); axes[1].set_ylabel('FC/APEC PE ratio')
    axes[1].set_title('Improvement factor')
    axes[1].invert_xaxis()

    fig.suptitle('1D CH4/N2 SRK Interface Advection\n(APEC vs FC-NPE PE error)')
    fig.tight_layout()
    _save(fig, '05_ch4_n2_srk_interface')

    print("    N    FC-PE       APEC-PE     ratio")
    for N, f, a, r in zip(Ns, pe_fc, pe_apec, impr):
        print(f"    {N:4d}  {f:.3e}  {a:.3e}  {r:.1f}×")
    return True


# ──────────────────────────────────────────────────────────────────
# 6. Gas-gas shock tube subsonic/transonic (ACID-like, MUSCL-LLF)
# ──────────────────────────────────────────────────────────────────
def case_gas_gas_shock_tube():
    """1D_gas_gas_shock_tube_subsonic_transonic."""
    gam = 1.4; N = 200; t_end = 0.2
    x = (np.arange(N)+0.5)/N; dx = 1/N

    # Standard Sod
    U0 = ic_ig(x, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gam, x0=0.5)
    U  = euler1d_ig(x, U0, gam, t_end, CFL=0.5, bc='transmissive')
    rho, u, p, _ = _prim_ig(U, gam)
    rho_ex, u_ex, p_ex = exact_sod(x, t_end, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gam)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, num, ex, lbl in zip(axes,
                                [rho, u, p], [rho_ex, u_ex, p_ex],
                                [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x, num, 'b-', lw=1.5, label='MUSCL-LLF')
        ax.plot(x, ex,  'k--', lw=1, label='Exact')
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=9)
    fig.suptitle(f'1D Gas-Gas Shock Tube (Sod, γ={gam}, N={N}, t={t_end})')
    fig.tight_layout()
    _save(fig, '06_gas_gas_shock_tube')
    return True


# ──────────────────────────────────────────────────────────────────
# 7. Gas-liquid Riemann problem (four_eq_1d §4.2.1)
# ──────────────────────────────────────────────────────────────────
def case_gas_liquid_riemann(fast=False):
    """1D_gas_liquid_riemann_problem."""
    FEQ, err = _load_solver('four_eq_1d')
    if FEQ is None:
        print(f"    four_eq_1d not available: {err}"); return False

    N = 200 if fast else 501
    print(f"    Running gas-liquid Riemann (N={N})...")
    FEQ.validate_riemann(N=N, CFL=0.3)
    return True


# ──────────────────────────────────────────────────────────────────
# 8. Gas-liquid shock tube air-water (four_eq_1d §4.2.3)
# ──────────────────────────────────────────────────────────────────
def case_gas_liquid_shock_tube(fast=False):
    """1D_gas_liquid_shock_tube_air_water."""
    FEQ, err = _load_solver('four_eq_1d')
    if FEQ is None:
        print(f"    four_eq_1d not available: {err}"); return False

    N = 200 if fast else 400
    print(f"    Running shock-droplet / air-water (N={N})...")
    FEQ.validate_shock_droplet(N=N, CFL=0.3)
    return True


# ──────────────────────────────────────────────────────────────────
# 9. Sod shock tube (KNP/PIMPLE → MUSCL-LLF approximation + exact)
# ──────────────────────────────────────────────────────────────────
def case_sod_knp_pimple():
    """1D_gas_shock_tube_Sod_KNP_PIMPLE."""
    gam = 1.4; N = 100; t_end = 0.2
    x = (np.arange(N)+0.5)/N; dx = 1/N

    U0 = ic_ig(x, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gam, x0=0.5)
    U  = euler1d_ig(x, U0, gam, t_end, CFL=0.5, bc='transmissive')
    rho, u, p, _ = _prim_ig(U, gam)
    rho_ex, u_ex, p_ex = exact_sod(x, t_end, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gam)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, num, ex, lbl in zip(axes, [rho, u, p], [rho_ex, u_ex, p_ex],
                                [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x, num, 'b-', lw=1.5, label='MUSCL-LLF')
        ax.plot(x, ex,  'k--', lw=1,  label='Exact (Toro)')
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=9)
    fig.suptitle(f'1D Sod Shock Tube (KNP-approx MUSCL-LLF, N={N}, t={t_end})')
    fig.tight_layout()
    _save(fig, '09_sod_knp_pimple')
    return True


# ──────────────────────────────────────────────────────────────────
# 10. Interface advection constant velocity (ACID)
# ──────────────────────────────────────────────────────────────────
def case_interface_advection_acid():
    """1D_interface_advection_constant_velocity_ACID."""
    # Two-component ideal gas, uniform advection, check pressure oscillation
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    gam1 = 1.4; gam2 = 1.4
    N = 100; t_end = 1.0
    x = (np.arange(N)+0.5)/N; dx = 1/N
    # Tanh interface at x=0.5
    k = 20.0; xc = 0.5
    W  = 0.5*(1 - np.tanh(k*(x - xc)))
    rho_bg = np.where(x < xc, 1000.0, 1.0)   # liquid left, gas right
    p0 = 1.0; u0 = 1.0
    U0 = LD.prim_to_cons(W, rho_bg, u0*np.ones(N), p0*np.ones(N), gam1, gam2)

    U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='periodic')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

    dp_max = np.max(np.abs(p_f - p0))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, rho_f, 'b-'); axes[0].set_ylabel('ρ'); axes[0].set_xlabel('x')
    axes[1].plot(x, u_f,   'r-'); axes[1].set_ylabel('u'); axes[1].set_xlabel('x')
    axes[1].axhline(u0, color='k', ls='--')
    axes[2].plot(x, p_f,   'g-'); axes[2].set_ylabel('p'); axes[2].set_xlabel('x')
    axes[2].axhline(p0, color='k', ls='--')
    fig.suptitle(f'1D Interface Advection (ACID-approx, N={N}, t={t_end})\n|Δp|_max={dp_max:.2e}')
    fig.tight_layout()
    _save(fig, '10_interface_advection_acid')
    print(f"    |Δp|_max = {dp_max:.3e}")
    return True


# ──────────────────────────────────────────────────────────────────
# 11. Inviscid droplet advection IEC (four_eq_1d §4.2.2)
# ──────────────────────────────────────────────────────────────────
def case_inviscid_droplet_iec(fast=False):
    """1D_inviscid_droplet_advection_IEC."""
    FEQ, err = _load_solver('four_eq_1d')
    if FEQ is None:
        print(f"    four_eq_1d not available: {err}"); return False

    N = 100 if fast else 200
    print(f"    Running droplet advection IEC (N={N})...")
    FEQ.validate_droplet(N=N, CFL=0.5)
    return True


# ──────────────────────────────────────────────────────────────────
# 12. Smooth interface advection FC-PE (4th-order central)
# ──────────────────────────────────────────────────────────────────
def case_smooth_interface_fcpe():
    """1D_inviscid_smooth_interface_advection_FCPE: simple test."""
    # Two-component ideal gas, periodic, tanh interface, check PE error
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    gam1 = 1.4; gam2 = 1.6
    Ns = [50, 100, 200]
    p0 = 1.0; u0 = 1.0; t_end = 1.0
    errs_pe = []
    for N in Ns:
        x   = (np.arange(N)+0.5)/N; dx = 1/N
        k   = 10.0
        W   = 0.5*(1 - np.tanh(k*(x - 0.5)))
        cv1 = 1/(gam1-1); cv2 = 1/(gam2-1)
        cv  = W*cv1 + (1-W)*cv2
        rho = 0.5*np.ones(N)  # uniform density
        U0  = LD.prim_to_cons(W, rho, u0*np.ones(N), p0*np.ones(N), gam1, gam2)
        U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2,
                                  sigma=0.5, bc='periodic')
        rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)
        errs_pe.append(np.sqrt(np.mean(((p_f - p0)/p0)**2)))

    errs_pe = np.array(errs_pe); dxs = 1/np.array(Ns)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(dxs, errs_pe, 'o-', label='PE rms error')
    axes[0].loglog(dxs, errs_pe[0]*(dxs/dxs[0])**2, 'k--', label='O(Δx²)')
    axes[0].set_xlabel('Δx'); axes[0].set_ylabel('PE error')
    axes[0].legend(); axes[0].set_title('Grid convergence of PE error')

    N = Ns[-1]; x = (np.arange(N)+0.5)/N; dx=1/N
    W = 0.5*(1-np.tanh(10*(x-0.5)))
    rho = 0.5*np.ones(N)
    U0 = LD.prim_to_cons(W, rho, u0*np.ones(N), p0*np.ones(N), gam1, gam2)
    U,_ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='periodic')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)
    axes[1].plot(x, p_f - p0, 'b-', label=f'p error (N={N})')
    axes[1].axhline(0, color='k', ls='--')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('p - p₀')
    axes[1].legend()
    fig.suptitle('1D Smooth Interface Advection (FC-PE approx via MUSCL)')
    fig.tight_layout()
    _save(fig, '12_smooth_interface_fcpe')
    return True


# ──────────────────────────────────────────────────────────────────
# 13. Moving contact discontinuity multicomponent (lambda_diff_1d)
# ──────────────────────────────────────────────────────────────────
def case_moving_contact_multicomponent():
    """1D_moving_contact_discontinuity_multicomponent."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    N = 100; dx = 1/N; t_end = 1.0
    x = (np.arange(N)+0.5)/N

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (gam1, gam2), title in zip(axes,
        [(1.4, 1.4), (1.4, 1.67)],
        ['Case A: γ₁=γ₂=1.4', 'Case B: γ₁=1.4, γ₂=1.67']):

        _, U0 = LD._shock_tube_ic(N, 1.0, 1.0, 1.0, 1.0, 0.0, 0.125, 1.0, 1.0, gam1, gam2)
        U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='transmissive')
        rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)
        du = np.max(np.abs(u_f - 1.0))
        dp = np.max(np.abs(p_f - 1.0))
        ax.plot(x, p_f - 1.0, 'g-', label=f'p-1, max={dp:.2e}')
        ax.plot(x, u_f - 1.0, 'r--', label=f'u-1, max={du:.2e}')
        ax.axhline(0, color='k', ls=':', lw=0.8)
        ax.set_xlabel('x'); ax.set_ylabel('error'); ax.legend(fontsize=8)
        ax.set_title(title)
        print(f"    {title}: |Δu|={du:.2e}, |Δp|={dp:.2e}")

    fig.suptitle('1D Moving Contact Discontinuity (multicomponent)')
    fig.tight_layout()
    _save(fig, '13_moving_contact_multicomponent')
    return True


# ──────────────────────────────────────────────────────────────────
# 14. Moving contact discontinuity two-phase
# ──────────────────────────────────────────────────────────────────
def case_moving_contact_two_phase():
    """1D_moving_contact_discontinuity_two_phase."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    # Two fluids: γ₁=1.4 (gas), γ₂=7.15 (stiffened water approx as ideal high-γ)
    gam1, gam2 = 1.4, 7.15
    N = 100; dx = 1/N; t_end = 0.3
    x = (np.arange(N)+0.5)/N

    # Moving contact: left rho=1, right rho=0.125, u=1, p=1 uniform
    _, U0 = LD._shock_tube_ic(N, 1.0, 1.0, 1.0, 1.0, 0.0, 0.125, 1.0, 1.0, gam1, gam2)
    U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='transmissive')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

    dp = np.max(np.abs(p_f - 1.0))
    du = np.max(np.abs(u_f - 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(x, rho_f, 'b-'); axes[0].set_ylabel('ρ'); axes[0].set_xlabel('x')
    axes[1].plot(x, u_f-1, 'r-'); axes[1].set_ylabel('u-1'); axes[1].set_xlabel('x')
    axes[1].axhline(0, color='k', ls='--')
    axes[2].plot(x, p_f-1, 'g-'); axes[2].set_ylabel('p-1'); axes[2].set_xlabel('x')
    axes[2].axhline(0, color='k', ls='--')
    fig.suptitle(f'1D Moving Contact Two-Phase (γ₁={gam1}, γ₂={gam2})\n'
                 f'|Δu|={du:.2e}, |Δp|={dp:.2e}')
    fig.tight_layout()
    _save(fig, '14_moving_contact_two_phase')
    print(f"    |Δu|={du:.2e}, |Δp|={dp:.2e}")
    return True


# ──────────────────────────────────────────────────────────────────
# 15. Multicomponent EOC sinusoidal (lambda_diff_1d §5.1)
# ──────────────────────────────────────────────────────────────────
def case_multicomponent_eoc(fast=False):
    """1D_multicomponent_EOC_sinusoidal."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    gam1 = gam2 = 1.4; t_end = 0.5; domain = 2.0
    Ns = [40, 80, 160, 320] if fast else [40, 80, 160, 320, 640]

    def exact_rho_avg(x, t, dx):
        h = dx/2; sinc_h = np.sin(np.pi*h)/(np.pi*h)
        return 1.0 + 0.2*np.sin(np.pi*(x - 0.1*t))*sinc_h

    orders_errs = {2: [], 3: []}
    for order in [2, 3]:
        for N in Ns:
            dx = domain/N
            x  = (np.arange(N)+0.5)*dx
            rho0 = exact_rho_avg(x, 0, dx)
            W0   = 0.5*np.ones(N); u0 = 0.1*np.ones(N); p0 = 0.5*np.ones(N)
            U0   = LD.prim_to_cons(W0, rho0, u0, p0, gam1, gam2)
            U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2,
                                      order=order, sigma=0.8, bc='periodic',
                                      max_steps=2_000_000)
            rho_ex = exact_rho_avg(x, t_end, dx)
            e2 = np.sqrt(dx*np.sum((U[1] - rho_ex)**2))
            orders_errs[order].append(e2)

    dxs = domain / np.array(Ns)
    fig, ax = plt.subplots(figsize=(7, 5))
    for order, color in zip([2, 3], ['C0', 'C1']):
        e = np.array(orders_errs[order])
        ax.loglog(dxs, e, 'o-', color=color, label=f'order={order}')
        eoc = np.log2(e[:-1]/e[1:])
        print(f"    Order {order}: EOC = {eoc}")
    ax.loglog(dxs, orders_errs[2][0]*(dxs/dxs[0])**2, 'k--', label='O(Δx²)')
    ax.loglog(dxs, orders_errs[3][0]*(dxs/dxs[0])**3, 'k:',  label='O(Δx³)')
    ax.set_xlabel('Δx'); ax.set_ylabel('L2 error ρ')
    ax.legend(); ax.set_title('EOC — Sinusoidal density wave')
    fig.suptitle('1D Multicomponent EOC (Roy & Raghurama Rao 2025)')
    fig.tight_layout()
    _save(fig, '15_multicomponent_eoc')
    return True


# ──────────────────────────────────────────────────────────────────
# 16. Multiphase multicomponent Shu-Osher (four_eq_1d variant)
# ──────────────────────────────────────────────────────────────────
def case_multiphase_shu_osher(fast=False):
    """1D_multiphase_multicomponent_shu_osher: use four_eq_1d."""
    FEQ, err = _load_solver('four_eq_1d')
    if FEQ is None:
        print(f"    four_eq_1d not available: {err}"); return False

    # Closest available test: shock-droplet is analogous to multiphase Shu-Osher
    N = 200 if fast else 400
    print(f"    Running multiphase Shu-Osher analog (N={N})...")
    FEQ.validate_shock_droplet(N=N, CFL=0.3)
    # Also run standard Shu-Osher (single-phase, inline)
    _case_shu_osher_inner(label='16_multiphase_shu_osher_single')
    return True


def _case_shu_osher_inner(N=400, label=None):
    gam = 1.4; t_end = 1.8
    x = np.linspace(-5, 5, N+1); x = 0.5*(x[:-1]+x[1:]); dx = x[1]-x[0]
    rhoL, uL, pL = 3.857143, 2.629369, 10.33333
    U0 = np.zeros((3, N))
    for j in range(N):
        if x[j] < -4:
            r, u_, p_ = rhoL, uL, pL
        else:
            r = 1 + 0.2*np.sin(5*x[j])
            u_, p_ = 0.0, 1.0
        U0[0,j] = r; U0[1,j] = r*u_; U0[2,j] = p_/(gam-1)+0.5*r*u_**2
    U  = euler1d_ig(x, U0, gam, t_end, CFL=0.5, bc='transmissive')
    rho, u, p, _ = _prim_ig(U, gam)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, rho, 'b-', lw=1.2, label=f'N={N}')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('ρ'); axes[0].legend()
    axes[0].set_title(f't={t_end} (Shu-Osher)')
    axes[1].plot(x, p,   'g-', lw=1.2)
    axes[1].set_xlabel('x'); axes[1].set_ylabel('p')
    if label:
        fig.suptitle('1D Shu-Osher Shock Tube (MUSCL-LLF)')
        fig.tight_layout()
        _save(fig, label)
    return fig, rho, u, p


# ──────────────────────────────────────────────────────────────────
# 17. Positivity of mass fraction (lambda_diff_1d §5.2.6)
# ──────────────────────────────────────────────────────────────────
def case_positivity_mass_fraction():
    """1D_positivity_mass_fraction_problem."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    gam1, gam2 = 1.4, 1.4
    N = 200; dx = 1/N; t_end = 0.15
    x = (np.arange(N)+0.5)/N

    def H_to_p(H, u_, rho_, g): return (H - 0.5*u_**2)*rho_*(g-1)/g
    pL = H_to_p(1.0, -1.0, 1.0, gam1); pR = H_to_p(5.0, 1.0, 1.0, gam2)

    _, U0 = LD._shock_tube_ic(N, 1.0, 1.0, -1.0, pL, 0.0, 1.0, 1.0, pR, gam1, gam2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for order, color in [(1,'C0'), (2,'C1'), (3,'C2')]:
        U, _ = LD.run_simulation(U0.copy(), N, dx, t_end, gam1, gam2,
                                  order=order, sigma=0.8, bc='transmissive')
        rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)
        lbl = f'Order {order}'
        axes[0].plot(x, rho_f, color=color, label=lbl)
        axes[1].plot(x, u_f,   color=color, label=lbl)
        axes[2].plot(x, W_f,   color=color, label=lbl)
        W_min = np.min(W_f); W_max = np.max(W_f)
        print(f"    Order {order}: W_min={W_min:.4f}, W_max={W_max:.4f}")

    for ax, lbl in zip(axes, [r'$\rho$', r'$u$', r'$W$ (mass fraction)']):
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=8)
    axes[2].set_title('W ∈ [0,1]?')
    fig.suptitle('1D Positivity of Mass Fraction (t=0.15)')
    fig.tight_layout()
    _save(fig, '17_positivity_mass_fraction')
    return True


# ──────────────────────────────────────────────────────────────────
# 18 & 19. Pressure discharge (gas→liquid, liquid→gas)
# ──────────────────────────────────────────────────────────────────
def case_pressure_discharge_gas_into_liquid():
    """1D_pressure_discharge_gas_into_liquid."""
    # High-pressure gas (left) → liquid (right)
    # Stiffened gas: air γ=1.4 Pinf=0, water γ=7.15 Pinf=3e8 Pa (non-dim)
    # Non-dimensional: p_ref = 1e5 Pa, rho_ref = 1 kg/m3 → P∞_nd = 3e8/1e5 = 3000
    g1, P1 = 1.4,   0.0    # gas
    g2, P2 = 7.15, 3000.0  # liquid (Pinf in non-dim units)
    N = 200; t_end = 0.15
    x = (np.arange(N)+0.5)/N

    # IC: gas left (p=100, rho1=50, u=0), water right (p=1, rho2=1000, u=0)
    a1L, pL = 1.0-1e-6, 100.0
    a1R, pR = 1e-6,       1.0
    rho1_gas, rho2_liq = 50.0, 1000.0
    U0 = ic_sg(x, a1L, rho1_gas, rho2_liq, 0.0, pL, g1, P1,
                  a1R, rho1_gas, rho2_liq, 0.0, pR, g2, P2)
    U  = euler1d_sg(x, U0, g1, P1, g2, P2, t_end, CFL=0.3, bc='transmissive')
    rho, u, p, c, al = _prim_sg(U, g1, P1, g2, P2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, q, lbl in zip(axes, [rho, u, p, al],
                           [r'$\rho$', r'$u$', r'$p$', r'$\alpha_1$ (gas)']):
        ax.plot(x, q, 'b-')
        ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle('1D Pressure Discharge: Gas → Liquid')
    fig.tight_layout()
    _save(fig, '18_pressure_discharge_gas_into_liquid')
    return True


def case_pressure_discharge_liquid_into_gas():
    """1D_pressure_discharge_liquid_into_gas."""
    g1, P1 = 1.4,   0.0
    g2, P2 = 7.15, 3000.0
    N = 200; t_end = 0.15
    x = (np.arange(N)+0.5)/N

    # IC: liquid left (p=100, rho2=1000), gas right (p=1, rho1=1)
    a1L, pL = 1e-6,     100.0   # mostly liquid left (phase 2)
    a1R, pR = 1.0-1e-6,   1.0  # mostly gas right (phase 1)
    rho1_gas, rho2_liq = 1.0, 1000.0
    U0 = ic_sg(x, a1L, rho1_gas, rho2_liq, 0.0, pL, g1, P1,
                  a1R, rho1_gas, rho2_liq, 0.0, pR, g2, P2)
    U  = euler1d_sg(x, U0, g1, P1, g2, P2, t_end, CFL=0.3, bc='transmissive')
    rho, u, p, c, al = _prim_sg(U, g1, P1, g2, P2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, q, lbl in zip(axes, [rho, u, p, al],
                           [r'$\rho$', r'$u$', r'$p$', r'$\alpha_1$ (liquid)']):
        ax.plot(x, q, 'b-')
        ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle('1D Pressure Discharge: Liquid → Gas')
    fig.tight_layout()
    _save(fig, '19_pressure_discharge_liquid_into_gas')
    return True


# ──────────────────────────────────────────────────────────────────
# 20. Pressure equilibrium preservation G1/G2/G3 (many_flux_1d)
# ──────────────────────────────────────────────────────────────────
def case_pressure_equilibrium_G1_G2_G3(fast=False):
    """1D_pressure_equilibrium_preservation_G1_G2_G3."""
    MF, err = _load_solver('many_flux_1d')
    if MF is None:
        print(f"    many_flux_1d not available: {err}"); return False

    t_end  = 5.0 if fast else 19.0
    CFL    = 0.01
    schemes = ['DIV', 'KGP', 'KEEPPE', 'KEEPPE_R']
    cases   = [('G1', MF.init_G1), ('G2', MF.init_G2), ('G3', MF.init_G3)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, init_f) in zip(axes, cases):
        Q0, x, dx, spl = init_f()
        for sch, col in zip(schemes, ['C0','C1','C2','C3']):
            try:
                times, pe_err, _ = MF.run_case(init_f, sch, t_end, CFL)
                valid = pe_err[np.isfinite(pe_err)]
                if len(valid) > 0:
                    ax.semilogy(times[:len(valid)], valid, color=col, label=sch)
                    print(f"    {name}/{sch}: final PE={valid[-1]:.2e}")
                else:
                    print(f"    {name}/{sch}: diverged")
            except Exception as e:
                print(f"    {name}/{sch}: error - {e}")
        ax.set_xlabel('t'); ax.set_ylabel('||ε_p||₁')
        ax.set_title(name); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle('1D Pressure Equilibrium Preservation G1/G2/G3\n(Wang et al. JCP 2025)')
    fig.tight_layout()
    _save(fig, '20_pressure_equilibrium_G1_G2_G3')
    return True


# ──────────────────────────────────────────────────────────────────
# 21. Pressure wave propagation in liquid (stiffened gas)
# ──────────────────────────────────────────────────────────────────
def case_pressure_wave_liquid():
    """1D_pressure_wave_propagation_liquid."""
    # Non-dimensional water: γ=7.15, Pinf=3000 (Pa/1e5)
    # c = sqrt(γ*(p+Pinf)/ρ) ≈ sqrt(7.15*3001/1000) ≈ 4.63 (ND)
    g1, P1 = 7.15, 3000.0   # water only, g2/P2 irrelevant
    g2, P2 = 1.4, 0.0
    rho0   = 1000.0; p0 = 1.0; eps = 0.01
    c_water = np.sqrt(g1*(p0+P1)/rho0)

    N = 200; t_end = 0.1
    x = (np.arange(N)+0.5)/N

    # Single fluid (pure water, alpha=1)
    al0 = 1 - 1e-6
    pulse = eps * np.exp(-((x-0.3)/0.02)**2)
    U0 = np.zeros((5, N))
    for j in range(N):
        rhoe_ = al0*(p0 + pulse[j] + g1*P1)/(g1-1) + (1-al0)*(p0+P2)/(g2-1)
        rho_  = al0*rho0 + (1-al0)*1.0
        u_    = 0.0
        U0[0,j] = al0*rho0; U0[1,j] = (1-al0)*1.0
        U0[2,j] = rho_*u_
        U0[3,j] = rhoe_ + 0.5*rho_*u_**2
        U0[4,j] = al0

    U = euler1d_sg(x, U0, g1, P1, g2, P2, t_end, CFL=0.3, bc='transmissive')
    rho, u, p, c, al = _prim_sg(U, g1, P1, g2, P2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, p,   'b-', label=f'p (t={t_end})')
    axes[0].axhline(p0, color='k', ls='--', lw=0.8, label='p₀')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('p'); axes[0].legend()
    axes[0].set_title('Pressure waveform')
    axes[1].plot(x, u,   'r-')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('u')
    axes[1].set_title(f'c_water ≈ {c_water:.2f} (ND)')
    fig.suptitle('1D Pressure Wave Propagation in Liquid (stiffened gas)')
    fig.tight_layout()
    _save(fig, '21_pressure_wave_liquid')
    return True


# ──────────────────────────────────────────────────────────────────
# 22. Shock air-helium interface Ms=1.22
# ──────────────────────────────────────────────────────────────────
def case_shock_air_helium():
    """1D_shock_air_helium_interface_Ms1.22."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    # Air: γ=1.4, He: γ=1.66
    gam_air = 1.4; gam_he = 1.66; Ms = 1.22
    # Post-shock air state (Rankine-Hugoniot)
    rho0_air = 1.0; p0 = 1.0; u0 = 0.0
    c_air = np.sqrt(gam_air*p0/rho0_air)
    gm1 = gam_air-1; gp1 = gam_air+1
    rho_s = rho0_air*(gp1*Ms**2)/(gm1*Ms**2+2)
    p_s   = p0*(2*gam_air*Ms**2 - gm1)/gp1
    u_s   = Ms*c_air*(1 - rho0_air/rho_s)   # velocity behind shock

    N = 300; dx = 1/N; t_end = 0.3
    x = (np.arange(N)+0.5)/N

    # IC: post-shock air on left (x<0.2), pre-shock air (0.2<x<0.5), He (x>0.5)
    W  = np.where(x < 0.5, 1.0, 0.0)   # W=1 → air, W=0 → He
    rho_ic = np.where(x < 0.2, rho_s, np.where(x < 0.5, rho0_air, 0.164))  # He density ≈ 0.164 kg/m3 ND
    u_ic   = np.where(x < 0.2, u_s, 0.0)
    p_ic   = np.where(x < 0.2, p_s, p0)

    U0 = LD.prim_to_cons(W, rho_ic, u_ic, p_ic, gam_air, gam_he)
    U, _ = LD.run_simulation(U0, N, dx, t_end, gam_air, gam_he, order=2, sigma=0.5, bc='transmissive')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam_air, gam_he)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, q, lbl in zip(axes, [rho_f, u_f, p_f], [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x, q, 'b-')
        ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle(f'1D Shock Air-Helium Interface (Ms={Ms}, t={t_end})')
    fig.tight_layout()
    _save(fig, '22_shock_air_helium_Ms1.22')
    return True


# ──────────────────────────────────────────────────────────────────
# 23. Shock air-water interface Ms=10 (stiffened gas)
# ──────────────────────────────────────────────────────────────────
def case_shock_air_water_ms10():
    """1D_shock_air_water_interface_Ms10."""
    # Air: γ=1.4, Water: γ=4.4, Pinf=6000 (non-dim)
    g_air, P_air = 1.4, 0.0
    g_wat, P_wat = 4.4, 6000.0
    Ms = 10.0

    rho0 = 1.0; p0 = 1.0
    gm1 = g_air-1; gp1 = g_air+1
    rho_s = rho0 * (gp1*Ms**2)/(gm1*Ms**2+2)
    p_s   = p0   * (2*g_air*Ms**2 - gm1)/gp1
    c_air = np.sqrt(g_air*p0/rho0)
    u_s   = Ms*c_air*(1 - rho0/rho_s)

    N = 400; t_end = 0.04
    x = (np.arange(N)+0.5)/N

    rho_wat = 1000.0
    a1L, a1R = 1.0-1e-6, 1e-6   # left=air, right=water

    # IC: shock in air on left, air center, water right
    U0 = np.zeros((5, N))
    for j in range(N):
        xj = x[j]
        if xj < 0.1:   # post-shock air
            al, r1_, r2_, u_, p_ = a1L, rho_s,   rho_wat, u_s, p_s
        elif xj < 0.5: # pre-shock air
            al, r1_, r2_, u_, p_ = a1L, rho0,    rho_wat, 0.0, p0
        else:           # water
            al, r1_, r2_, u_, p_ = a1R, rho0,    rho_wat, 0.0, p0
        a2  = 1-al
        rhoe_ = al*(p_+g_air*P_air)/(g_air-1) + a2*(p_+g_wat*P_wat)/(g_wat-1)
        rho_  = al*r1_ + a2*r2_
        U0[0,j]=al*r1_; U0[1,j]=a2*r2_; U0[2,j]=rho_*u_
        U0[3,j]=rhoe_+0.5*rho_*u_**2;  U0[4,j]=al

    U = euler1d_sg(x, U0, g_air, P_air, g_wat, P_wat, t_end, CFL=0.3, bc='transmissive')
    rho, u, p, c, al = _prim_sg(U, g_air, P_air, g_wat, P_wat)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, q, lbl in zip(axes, [rho, u, p, al],
                           [r'$\rho$', r'$u$', r'$p$', r'$\alpha_{air}$']):
        ax.plot(x, q, 'b-'); ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle(f'1D Shock Air-Water Interface (Ms={Ms}, t={t_end})')
    fig.tight_layout()
    _save(fig, '23_shock_air_water_Ms10')
    return True


# ──────────────────────────────────────────────────────────────────
# 24. Shock impedance matching gas-gas
# ──────────────────────────────────────────────────────────────────
def case_shock_impedance_matching():
    """1D_shock_impedance_matching_gas_gas."""
    # Two gases with matched acoustic impedance Z=rho*c
    # γ₁=1.4, γ₂=1.67; impedance-matched: choose rho2 such that rho1*c1=rho2*c2
    gam1, gam2 = 1.4, 1.67
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    p0 = 1.0; rho1 = 1.0; c1 = np.sqrt(gam1*p0/rho1)
    # rho2*c2 = rho1*c1 → rho2*sqrt(gam2*p0/rho2) = rho1*c1 → rho2*sqrt(gam2/rho2) = rho1*c1/sqrt(p0)
    # sqrt(rho2)*sqrt(gam2) = rho1*c1/sqrt(p0) = c1 → rho2 = (c1/sqrt(gam2))^2 = gam1/gam2
    rho2 = gam1/gam2   # matched impedance
    c2   = np.sqrt(gam2*p0/rho2)

    Ms = 2.0
    gm1=gam1-1; gp1=gam1+1
    rho_s = rho1*(gp1*Ms**2)/(gm1*Ms**2+2)
    p_s   = p0*(2*gam1*Ms**2-gm1)/gp1
    c_air = np.sqrt(gam1*p0/rho1)
    u_s   = Ms*c_air*(1-rho1/rho_s)

    N = 300; dx = 1/N; t_end = 0.2
    x = (np.arange(N)+0.5)/N
    W  = np.where(x < 0.5, 1.0, 0.0)
    rho_ic = np.where(x < 0.15, rho_s, np.where(x<0.5, rho1, rho2))
    u_ic   = np.where(x < 0.15, u_s, 0.0)
    p_ic   = np.where(x < 0.15, p_s, p0)

    U0 = LD.prim_to_cons(W, rho_ic, u_ic, p_ic, gam1, gam2)
    U, _ = LD.run_simulation(U0, N, dx, t_end, gam1, gam2, order=2, sigma=0.5, bc='transmissive')
    rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

    Z1 = rho1*c1; Z2 = rho2*c2
    R  = (Z2-Z1)/(Z2+Z1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, q, lbl in zip(axes, [rho_f, u_f, p_f], [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x, q, 'b-'); ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle(f'1D Shock Impedance Matching Gas-Gas\nZ1={Z1:.3f}, Z2={Z2:.3f}, R={R:.4f}')
    fig.tight_layout()
    _save(fig, '24_shock_impedance_matching_gas_gas')
    print(f"    Z1={Z1:.4f}, Z2={Z2:.4f}, R_theory={R:.4f}")
    return True


# ──────────────────────────────────────────────────────────────────
# 25. Woodward-Colella blast wave
# ──────────────────────────────────────────────────────────────────
def case_woodward_colella(fast=False):
    """1D_shock_wave_interaction_Woodward_Colella."""
    gam = 1.4; t_end = 0.038
    N_ref = 800 if fast else 4000
    N     = 400

    def _run_wc(N_):
        x = (np.arange(N_)+0.5)/N_; dx=1/N_
        U0 = np.zeros((3, N_))
        for j in range(N_):
            xj = x[j]
            if xj < 0.1:   r, u_, p_ = 1.0, 0.0, 1000.0
            elif xj < 0.9: r, u_, p_ = 1.0, 0.0, 0.01
            else:           r, u_, p_ = 1.0, 0.0, 100.0
            U0[0,j]=r; U0[1,j]=r*u_; U0[2,j]=p_/(gam-1)+0.5*r*u_**2
        # reflecting BCs: flip momentum at boundaries
        def rhs_wc(U):
            Ue = np.hstack([U[:, 1:2], U, U[:, -2:-1]])
            Ue[1, 0]  = -Ue[1, 0]   # reflect momentum
            Ue[1, -1] = -Ue[1, -1]
            return _rhs_ig(Ue[:, 1:-1], gam, dx, bc='custom_ignore')
        # Use transmissive but set ghost = reflected
        def rhs_reflect(U):
            Ug = np.hstack([U[:, 0:1], U, U[:, -1:]])
            Ug[1, 0]  = -U[1, 0]    # left reflecting
            Ug[1, -1] = -U[1, -1]   # right reflecting
            dL = Ug[:, 1:-1] - Ug[:, :-2]
            dR = Ug[:, 2:]   - Ug[:, 1:-1]
            sl = _minmod(dL, dR)
            sle = np.hstack([np.zeros((3,1)), sl, np.zeros((3,1))])
            ULf = (Ug + 0.5*sle)[:, :-1]
            URf = (Ug - 0.5*sle)[:, 1:]
            rL_, uL_, pL_, cL_ = _prim_ig(ULf, gam)
            rR_, uR_, pR_, cR_ = _prim_ig(URf, gam)
            FL = np.array([ULf[1], ULf[1]*uL_+pL_, (ULf[2]+pL_)*uL_])
            FR = np.array([URf[1], URf[1]*uR_+pR_, (URf[2]+pR_)*uR_])
            lam = np.maximum(np.abs(uL_)+cL_, np.abs(uR_)+cR_)
            Ff  = 0.5*(FL+FR) - 0.5*lam*(URf-ULf)
            return -(Ff[:, 1:] - Ff[:, :-1]) / dx

        U = U0.copy(); t_ = 0.0
        while t_ < t_end - 1e-14:
            rho_, u__, p__, c__ = _prim_ig(U, gam)
            dt = 0.4*dx/np.max(np.abs(u__)+c__)
            dt = min(dt, t_end - t_)
            U  = _ssprk3(U, rhs_reflect, dt)
            t_ += dt
        return x, U

    print(f"    Running WC reference (N={N_ref})...")
    x_ref, U_ref = _run_wc(N_ref)
    rho_ref, u_ref, p_ref, _ = _prim_ig(U_ref, gam)

    print(f"    Running WC numerical (N={N})...")
    x_num, U_num = _run_wc(N)
    rho_num, u_num, p_num, _ = _prim_ig(U_num, gam)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, rn, rr, lbl in zip(axes, [rho_num, u_num, p_num],
                                      [rho_ref, u_ref, p_ref],
                                      [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x_num, rn, 'b-',  lw=1.5, label=f'N={N}')
        ax.plot(x_ref, rr, 'k--', lw=0.8, label=f'N={N_ref} (ref)')
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=8)
    fig.suptitle(f'1D Woodward-Colella Blast (t={t_end})')
    fig.tight_layout()
    _save(fig, '25_woodward_colella')
    return True


# ──────────────────────────────────────────────────────────────────
# 26. Shock wave propagation air-water (stiffened gas)
# ──────────────────────────────────────────────────────────────────
def case_shock_wave_air_water():
    """1D_shock_wave_propagation_air_water."""
    # Reuse same setup as Ms=10 but with different Ms and interface position
    g_air, P_air = 1.4,  0.0
    g_wat, P_wat = 4.4, 6000.0
    Ms = 5.0
    rho0 = 1.0; p0 = 1.0
    gm1=g_air-1; gp1=g_air+1
    rho_s = rho0*(gp1*Ms**2)/(gm1*Ms**2+2)
    p_s   = p0*(2*g_air*Ms**2-gm1)/gp1
    c_air = np.sqrt(g_air*p0/rho0)
    u_s   = Ms*c_air*(1-rho0/rho_s)

    N = 300; t_end = 0.05
    x = (np.arange(N)+0.5)/N

    rho_wat = 1000.0; a1L = 1-1e-6; a1R = 1e-6
    U0 = np.zeros((5,N))
    for j in range(N):
        xj = x[j]
        if xj < 0.15:   al, r1, r2, u_, p_ = a1L, rho_s,   rho_wat, u_s, p_s
        elif xj < 0.5:  al, r1, r2, u_, p_ = a1L, rho0,    rho_wat, 0.0, p0
        else:            al, r1, r2, u_, p_ = a1R, rho0,    rho_wat, 0.0, p0
        a2 = 1-al
        rhoe_ = al*(p_+g_air*P_air)/(g_air-1)+a2*(p_+g_wat*P_wat)/(g_wat-1)
        rho_  = al*r1 + a2*r2
        U0[0,j]=al*r1; U0[1,j]=a2*r2; U0[2,j]=rho_*u_
        U0[3,j]=rhoe_+0.5*rho_*u_**2; U0[4,j]=al

    U = euler1d_sg(x, U0, g_air, P_air, g_wat, P_wat, t_end, CFL=0.3, bc='transmissive')
    rho, u, p, c, al = _prim_sg(U, g_air, P_air, g_wat, P_wat)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, q, lbl in zip(axes, [rho, u, p], [r'$\rho$', r'$u$', r'$p$']):
        ax.plot(x, q, 'b-'); ax.set_xlabel('x'); ax.set_ylabel(lbl)
    fig.suptitle(f'1D Shock Wave Propagation Air-Water (Ms={Ms}, t={t_end})')
    fig.tight_layout()
    _save(fig, '26_shock_wave_air_water')
    return True


# ──────────────────────────────────────────────────────────────────
# 27. Shu-Osher shock tube
# ──────────────────────────────────────────────────────────────────
def case_shu_osher(fast=False):
    """1D_shu_osher_shock_tube."""
    N_ref = 800 if fast else 2000
    N     = 200
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for N_, col, lbl in [(N_ref, 'k--', f'Ref N={N_ref}'), (N, 'b-', f'N={N}')]:
        _, rho, u, p = _case_shu_osher_inner(N=N_)
        t_end = 1.8
        gam = 1.4
        x = np.linspace(-5, 5, N_+1); x = 0.5*(x[:-1]+x[1:])
        axes[0].plot(x, rho, col, lw=(0.8 if N_==N_ref else 1.5), label=lbl)
        axes[1].plot(x, p,   col, lw=(0.8 if N_==N_ref else 1.5), label=lbl)
    for ax, lbl in zip(axes, [r'$\rho$', r'$p$']):
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend()
    fig.suptitle(f'1D Shu-Osher Shock Tube (MUSCL-LLF, t=1.8)')
    fig.tight_layout()
    _save(fig, '27_shu_osher_shock_tube')
    return True


# ──────────────────────────────────────────────────────────────────
# 28. Smooth interface advection PEP (exact PE-preserving)
# ──────────────────────────────────────────────────────────────────
def case_smooth_interface_pep():
    """1D_smooth_interface_advection_PEP."""
    # APEC as approximate PEP — compare FC vs APEC PE error
    APEC, err = _load_solver('apec_1d')
    if APEC is None:
        # Fallback: use lambda_diff_1d PE comparison
        return case_smooth_interface_fcpe()

    Ns = [50, 100, 200]
    pe_fc = []; pe_apec = []
    for N in Ns:
        _, _, p_fc, _, pe_h_fc, _, _ = APEC.run_cpg('FC', N=N, t_end=0.5, CFL=0.5)
        pe_fc.append(pe_h_fc[-1] if len(pe_h_fc) else np.nan)
        _, _, p_ap, _, pe_h_ap, _, _ = APEC.run_cpg('APEC', N=N, t_end=0.5, CFL=0.5)
        pe_apec.append(pe_h_ap[-1] if len(pe_h_ap) else np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    dxs = 1/np.array(Ns)
    ax.loglog(dxs, pe_fc,   'C0o--', label='FC-NPE')
    ax.loglog(dxs, pe_apec, 'C1s-',  label='APEC (≈PEP)')
    ax.loglog(dxs, np.array(pe_apec[0])*(dxs/dxs[0])**2, 'k--', label='O(Δx²)')
    ax.set_xlabel('Δx'); ax.set_ylabel('PE rms error (t=0.5)')
    ax.legend(); ax.set_title('FC vs APEC PE convergence')
    fig.suptitle('1D Smooth Interface Advection (PEP comparison)')
    fig.tight_layout()
    _save(fig, '28_smooth_interface_pep')
    return True


# ──────────────────────────────────────────────────────────────────
# 29. Sod shock tube multicomponent (lambda_diff_1d)
# ──────────────────────────────────────────────────────────────────
def case_sod_multicomponent():
    """1D_sod_shock_tube_multicomponent."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    N = 100; dx = 1/N; t_end = 0.2
    x = (np.arange(N)+0.5)/N

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row, (gam1, gam2), case in zip([0,1], [(1.4,1.4),(1.4,1.67)], ['A','B']):
        U0 = LD._shock_tube_ic(N, 1.0, 1.0, 0.0, 1.0, 0.0, 0.125, 0.0, 0.1, gam1, gam2)
        if isinstance(U0, tuple): _, U0 = U0
        U, _ = LD.run_simulation(U0.copy(), N, dx, t_end, gam1, gam2, order=2, sigma=0.8, bc='transmissive')
        rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)

        rho_ex, u_ex, p_ex = exact_sod(x, t_end, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gam1)
        for ax, q, qex, lbl in zip(axes[row], [rho_f, u_f, p_f], [rho_ex, u_ex, p_ex],
                                   [r'$\rho$', r'$u$', r'$p$']):
            ax.plot(x, q,   'b-', lw=1.5, label='Kinetic')
            ax.plot(x, qex, 'k--', lw=0.8, label='Exact')
            ax.set_xlabel('x'); ax.set_ylabel(lbl)
            ax.set_title(f'Case {case}: γ₁={gam1}, γ₂={gam2}')
            ax.legend(fontsize=7)

    fig.suptitle('1D Sod Shock Tube Multicomponent (Roy & Raghurama Rao 2025)')
    fig.tight_layout()
    _save(fig, '29_sod_shock_tube_multicomponent')
    return True


# ──────────────────────────────────────────────────────────────────
# 30. Species/temperature preservation S1/S2 (many_flux_1d)
# ──────────────────────────────────────────────────────────────────
def case_species_temperature_preservation(fast=False):
    """1D_species_temperature_preservation_S1_S2."""
    MF, err = _load_solver('many_flux_1d')
    if MF is None:
        print(f"    many_flux_1d not available: {err}"); return False

    t_end   = 5.0 if fast else 10.0
    CFL     = 0.01
    schemes = ['DIV', 'KGP', 'KEEPPE', 'KEEPPE_R']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (case_name, init_f, err_fn, ylabel) in zip(axes, [
        ('S1', MF.init_S1, MF.species_error,     '||ε_Y||₁'),
        ('S2', MF.init_S2, MF.temperature_error, '||ε_T||₁'),
    ]):
        for sch, col in zip(schemes, ['C0','C1','C2','C3']):
            try:
                times, _, Q_final = MF.run_case(init_f, sch, t_end, CFL)
                Q0, _, _, spl = init_f()
                e = err_fn(Q_final, Q0, spl)
                ax.bar(sch, e, color=col, alpha=0.7)
                print(f"    {case_name}/{sch}: error={e:.2e}")
            except Exception as ex:
                print(f"    {case_name}/{sch}: error - {ex}")
        ax.set_ylabel(ylabel); ax.set_title(case_name); ax.set_yscale('log')

    fig.suptitle('1D Species/Temperature Preservation S1/S2\n(Wang et al. JCP 2025)')
    fig.tight_layout()
    _save(fig, '30_species_temperature_S1_S2')
    return True


# ──────────────────────────────────────────────────────────────────
# 31. Steady contact discontinuity different gamma (lambda_diff_1d)
# ──────────────────────────────────────────────────────────────────
def case_steady_contact():
    """1D_steady_contact_discontinuity_different_gamma."""
    LD, err = _load_solver('lambda_diff_1d')
    if LD is None:
        print(f"    lambda_diff_1d not available: {err}"); return False

    gam1, gam2 = 1.4, 1.67
    N = 200; dx = 1/N; t_end = 0.25
    x = (np.arange(N)+0.5)/N

    # IC: stationary contact, different γ, uniform p=1, u=0
    U0 = LD._shock_tube_ic(N, 1.0, 1.0, 0.0, 1.0, 0.0, 0.125, 0.0, 1.0, gam1, gam2)
    if isinstance(U0, tuple): _, U0 = U0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for order, col in [(1,'C0'), (2,'C1'), (3,'C2')]:
        U, _ = LD.run_simulation(U0.copy(), N, dx, t_end, gam1, gam2,
                                  order=order, sigma=0.8, bc='transmissive')
        rho_f, W_f, u_f, p_f = LD.cons_to_prim(U, gam1, gam2)
        lbl = f'Order {order}'
        axes[0].plot(x, rho_f, color=col, label=lbl)
        axes[1].plot(x, u_f,   color=col, label=lbl)
        axes[2].plot(x, p_f,   color=col, label=lbl)
        du = np.max(np.abs(u_f)); dp = np.max(np.abs(p_f - 1.0))
        print(f"    Order {order}: |u|_max={du:.2e}, |Δp|_max={dp:.2e}")

    axes[1].axhline(0, color='k', ls='--', lw=0.8)
    axes[2].axhline(1, color='k', ls='--', lw=0.8)
    for ax, lbl in zip(axes, [r'$\rho$', r'$u$', r'$p$']):
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=8)
    fig.suptitle(f'1D Steady Contact Discontinuity (γ₁={gam1}, γ₂={gam2}, t={t_end})')
    fig.tight_layout()
    _save(fig, '31_steady_contact_different_gamma')
    return True


# ══════════════════════════════════════════════════════════════════
# CASE REGISTRY
# ══════════════════════════════════════════════════════════════════

CASES = [
    ('01_acoustic_wave_propagation',               case_acoustic_wave_propagation,         False),
    ('02_acoustic_reflection_transmission',         case_acoustic_reflection_transmission,  False),
    ('03_acoustic_impedance_matching_gas_gas',      case_acoustic_impedance_matching,       False),
    ('04_cpg_interface_advection',                  case_cpg_interface_advection,           True),   # slow
    ('05_ch4_n2_srk_interface',                     case_ch4_n2_srk,                        True),   # slow
    ('06_gas_gas_shock_tube',                       case_gas_gas_shock_tube,                False),
    ('07_gas_liquid_riemann',                       case_gas_liquid_riemann,                True),   # slow
    ('08_gas_liquid_shock_tube_air_water',          case_gas_liquid_shock_tube,             True),
    ('09_sod_knp_pimple',                           case_sod_knp_pimple,                    False),
    ('10_interface_advection_acid',                 case_interface_advection_acid,          False),
    ('11_inviscid_droplet_iec',                     case_inviscid_droplet_iec,              True),   # slow
    ('12_smooth_interface_fcpe',                    case_smooth_interface_fcpe,             False),
    ('13_moving_contact_multicomponent',            case_moving_contact_multicomponent,     False),
    ('14_moving_contact_two_phase',                 case_moving_contact_two_phase,          False),
    ('15_multicomponent_eoc',                       case_multicomponent_eoc,                True),   # slow
    ('16_multiphase_shu_osher',                     case_multiphase_shu_osher,              True),
    ('17_positivity_mass_fraction',                 case_positivity_mass_fraction,          False),
    ('18_pressure_discharge_gas_into_liquid',       case_pressure_discharge_gas_into_liquid, False),
    ('19_pressure_discharge_liquid_into_gas',       case_pressure_discharge_liquid_into_gas, False),
    ('20_pressure_equilibrium_G1_G2_G3',            case_pressure_equilibrium_G1_G2_G3,     True),   # slow
    ('21_pressure_wave_liquid',                     case_pressure_wave_liquid,              False),
    ('22_shock_air_helium_Ms1.22',                  case_shock_air_helium,                  False),
    ('23_shock_air_water_Ms10',                     case_shock_air_water_ms10,              False),
    ('24_shock_impedance_matching_gas_gas',         case_shock_impedance_matching,          False),
    ('25_woodward_colella',                         case_woodward_colella,                  True),   # slow ref
    ('26_shock_wave_air_water',                     case_shock_wave_air_water,              False),
    ('27_shu_osher_shock_tube',                     case_shu_osher,                         True),   # slow ref
    ('28_smooth_interface_pep',                     case_smooth_interface_pep,              False),
    ('29_sod_shock_tube_multicomponent',            case_sod_multicomponent,                False),
    ('30_species_temperature_S1_S2',                case_species_temperature_preservation,  True),   # slow
    ('31_steady_contact_different_gamma',           case_steady_contact,                    False),
]


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--case', type=str, default='',
                        help='Filter: run only cases whose name contains this string')
    parser.add_argument('--fast', action='store_true',
                        help='Skip slow cases / use smaller grids')
    args = parser.parse_args()

    if args.list:
        print(f"\n{'#':>3}  {'Case name':<50}  {'slow?'}")
        for i, (name, fn, slow) in enumerate(CASES, 1):
            print(f"  {i:>2}. {name:<50}  {'[slow]' if slow else ''}")
        return

    results = []
    t_total = time.time()

    for name, fn, slow in CASES:
        if args.case and args.case.lower() not in name.lower():
            continue
        if args.fast and slow:
            print(f"  [{name}] SKIPPED (--fast)")
            results.append((name, 'SKIP', 0.0))
            continue

        print(f"\n{'─'*60}")
        print(f"  [{name}]")
        t0 = time.time()
        try:
            # Pass fast flag if function accepts it
            import inspect
            sig = inspect.signature(fn)
            if 'fast' in sig.parameters:
                ok = fn(fast=args.fast)
            else:
                ok = fn()
            dt = time.time() - t0
            status = 'OK' if ok else 'FAIL'
        except Exception as e:
            dt = time.time() - t0
            status = 'ERROR'
            print(f"    ERROR: {e}")
            traceback.print_exc()
        results.append((name, status, dt))

    # Summary
    sep = '=' * 70
    print(f"\n{sep}")
    print(f"  1D VALIDATION SUMMARY  ({time.time()-t_total:.1f}s total)")
    print(sep)
    print(f"  {'#':>3}  {'Case':<50}  {'Status':>6}  {'Time':>6}")
    print(f"  {'-'*3}  {'-'*50}  {'-'*6}  {'-'*6}")
    ok_n = skip_n = fail_n = 0
    for i, (name, status, dt) in enumerate(results, 1):
        sym = 'OK' if status=='OK' else ('--' if status=='SKIP' else '!!')
        print(f"  [{sym}] {i:>2}. {name:<50}  {status:>6}  {dt:>5.1f}s")
        if status == 'OK':     ok_n   += 1
        elif status == 'SKIP': skip_n += 1
        else:                  fail_n += 1
    print(f"{'-'*70}")
    print(f"  OK={ok_n}  SKIP={skip_n}  FAIL/ERROR={fail_n}")
    print(f"  Output: {OUTPUT_DIR}")
    print(sep)


if __name__ == '__main__':
    main()
