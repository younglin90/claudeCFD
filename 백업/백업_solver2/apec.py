import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Physical constants & species data
# ─────────────────────────────────────────────────────────────
Ru  = 8.314          # J/(mol*K)
M   = np.array([16.043e-3, 28.014e-3])   # kg/mol
Tc  = np.array([190.56,    126.19   ])   # K
pc  = np.array([4.599e6,   3.396e6  ])   # Pa
om  = np.array([0.0115,    0.0372   ])   # acentric factor

a_sp = 0.42748 * Ru**2 * Tc**2 / pc
b_sp = 0.08664 * Ru   * Tc    / pc
fom  = 0.480 + 1.574*om - 0.176*om**2
Cv0 = Ru / M * np.array([3.4, 2.5])

# ─────────────────────────────────────────────────────────────
# Vectorized SRK EOS (From Original Code)
# ─────────────────────────────────────────────────────────────
def _mix(r1, r2, T):
    r1, r2, T = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float), np.asarray(T, dtype=float)
    rho  = r1 + r2
    C0, C1 = r1 / M[0], r2 / M[1]
    Ctot = np.maximum(C0 + C1, 1e-40)
    X0, X1 = C0 / Ctot, C1 / Ctot
    Mbar = rho / Ctot

    sq0, sq1 = np.sqrt(np.maximum(T / Tc[0], 1e-10)), np.sqrt(np.maximum(T / Tc[1], 1e-10))
    al0, al1 = (1.0 + fom[0]*(1.0 - sq0))**2, (1.0 + fom[1]*(1.0 - sq1))**2
    da0 = -fom[0]*(1.0 + fom[0]*(1.0 - sq0)) / (sq0*Tc[0] + 1e-300)
    da1 = -fom[1]*(1.0 + fom[1]*(1.0 - sq1)) / (sq1*Tc[1] + 1e-300)

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
    arg = np.where(b > 1e-40, 1.0 + b/np.maximum(v, 1e-30), 1.0)
    dep = np.where(b > 1e-40, Ctot*(T*daA - aA)/b * np.log(np.maximum(arg, 1e-30)), 0.0)
    return rhoe0 + dep

def T_from_rhoe(r1, r2, rhoe_target, T_in=None):
    r1, r2, rhoe_target = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float), np.asarray(rhoe_target, dtype=float)
    T = (np.asarray(T_in, dtype=float).copy() if T_in is not None else np.full_like(r1, 200.0))
    T = np.clip(T, 10.0, 3000.0)
    h = 1.0
    for _ in range(25):
        f0   = srk_rhoe(r1, r2, T) - rhoe_target
        fp   = srk_rhoe(r1, r2, T + h)
        fm   = srk_rhoe(r1, r2, T - h)
        dfdT = (fp - fm) / (2.0*h)
        dT   = -f0 / (dfdT + 1e-6)
        dT   = np.clip(dT, -200.0, 200.0)
        T    = np.clip(T + dT, 10.0, 3000.0)
        if np.max(np.abs(dT)) < 1e-3: break
    return T

def _dpdT(r1, r2, T, h=1.0): return (srk_p(r1, r2, T+h) - srk_p(r1, r2, T-h)) / (2.0*h)
def _dpdr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
    return (srk_p(r1+dr, r2, T) - srk_p(r1-dr, r2, T)) / (2.0*dr) if s==0 else \
           (srk_p(r1, r2+dr, T) - srk_p(r1, r2-dr, T)) / (2.0*dr)
def _drhoedr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
    return (srk_rhoe(r1+dr, r2, T) - srk_rhoe(r1-dr, r2, T)) / (2.0*dr) if s==0 else \
           (srk_rhoe(r1, r2+dr, T) - srk_rhoe(r1, r2-dr, T)) / (2.0*dr)
def srk_Cv(r1, r2, T, h=1.0): return ((srk_rhoe(r1, r2, T+h) - srk_rhoe(r1, r2, T-h)) / (2.0*h)) / np.maximum(r1 + r2, 1e-30)

def epsilon_v(r1, r2, T, s):
    rho = r1 + r2
    Cv  = srk_Cv(r1, r2, T)
    dpT = _dpdT(r1, r2, T)
    dprs = _dpdr(r1, r2, T, s)
    drhoe= _drhoedr(r1, r2, T, s)
    return -(rho * Cv / (dpT + 1e-10)) * dprs + drhoe

def srk_c2(r1, r2, T):
    rho, Y0 = r1 + r2, r1 / np.maximum(r1 + r2, 1e-30)
    dpdr_T = Y0*_dpdr(r1, r2, T, 0) + (1.0 - Y0)*_dpdr(r1, r2, T, 1)
    dpT, Cv = _dpdT(r1, r2, T), srk_Cv(r1, r2, T)
    return np.maximum(dpdr_T + T*dpT**2 / (rho**2*Cv + 1e-30), 100.0)

# ─────────────────────────────────────────────────────────────
# CPG EOS  (Validation §3.1: Terashima JCP 2025)
# Species 0: γ=1.4,  M=28 g/mol  (N2-like)
# Species 1: γ=1.66, M=4  g/mol  (He-like)
# ─────────────────────────────────────────────────────────────
_GAM = np.array([1.4,  1.66])
_MW  = np.array([28.0,  4.0])

def _cpg_inv_gm1(r1, r2):
    """1/(γ̄-1) = Σ X_i/(γ_i-1)  (mole-fraction weighted)"""
    n1, n2 = r1 / _MW[0], r2 / _MW[1]
    ntot   = np.maximum(n1 + n2, 1e-60)
    return (n1/(_GAM[0]-1.0) + n2/(_GAM[1]-1.0)) / ntot

def cpg_rhoe_from_p(r1, r2, p):
    """ρe = p/(γ̄-1)"""
    return p * _cpg_inv_gm1(r1, r2)

def cpg_p_from_rhoe(r1, r2, rhoe):
    """p = ρe*(γ̄-1)"""
    inv_gm1 = _cpg_inv_gm1(r1, r2)
    return rhoe / np.maximum(inv_gm1, 1e-60)

def cpg_c2(r1, r2, p):
    """c² = γ̄ p / ρ"""
    inv_gm1 = _cpg_inv_gm1(r1, r2)
    gbar = 1.0 + 1.0 / np.maximum(inv_gm1, 1e-60)
    return gbar * p / np.maximum(r1 + r2, 1e-60)

def cpg_T(r1, r2, rhoe):
    """CPG 혼합 온도 (Ru=1 무차원): T = ρe / Σ_i ρY_i·cv_i
    cv_i = 1 / (M_i·(γ_i-1))"""
    cv1 = 1.0 / (_MW[0] * (_GAM[0] - 1.0))
    cv2 = 1.0 / (_MW[1] * (_GAM[1] - 1.0))
    return rhoe / np.maximum(r1*cv1 + r2*cv2, 1e-60)

def cpg_epsilon(r1, r2, p):
    """ε_i = (∂ρe/∂ρ_i)_{ρ_{j≠i},p}  (Eq. 59)"""
    rho  = r1 + r2
    n1, n2 = r1 / _MW[0], r2 / _MW[1]
    ntot = np.maximum(n1 + n2, 1e-60)
    Mbar = rho / ntot
    B    = n1/(_GAM[0]-1.0) + n2/(_GAM[1]-1.0)
    pref = p * Mbar**2 / np.maximum(rho**2, 1e-60)
    eps0 = pref / _MW[0] * (ntot/(_GAM[0]-1.0) - B)
    eps1 = pref / _MW[1] * (ntot/(_GAM[1]-1.0) - B)
    return eps0, eps1

# ─────────────────────────────────────────────────────────────
# Initial condition (Section 3.2.1)
# ─────────────────────────────────────────────────────────────
def initial_condition(x, p_inf=5e6, k=15.0):
    N, xc, rc = len(x), 0.5, 0.25
    r1_inf, r2_inf = 400.0, 100.0
    r = np.abs(x - xc)
    r1 = 0.5*r1_inf*(1.0 - np.tanh(k*(r - rc)))
    r2 = 0.5*r2_inf*(1.0 + np.tanh(k*(r - rc)))

    print("  Solving T from p_inf=5MPa (vectorized Newton)...", flush=True)
    _, Mb, _, _, _, _, _, *_ = _mix(r1, r2, np.full(N, 300.0))
    T = np.clip(p_inf * Mb / np.maximum(r1 + r2, 1e-30) / Ru, 50.0, 1000.0)
    for _ in range(60):
        ph, dph = srk_p(r1, r2, T), _dpdT(r1, r2, T)
        dT = np.clip(-(ph - p_inf) / (dph + 1e-3), -100.0, 100.0)
        T = np.clip(T + dT, 10.0, 2000.0)
        if np.max(np.abs(dT)) < 1e-3: break

    p = srk_p(r1, r2, T)
    rhoe = srk_rhoe(r1, r2, T)
    u = np.full(N, 100.0)
    return r1, r2, u, rhoe + 0.5*(r1+r2)*u**2, T, p

def prim(r1, r2, rhoU, rhoE, T_prev):
    rho = r1 + r2
    u = rhoU / np.maximum(rho, 1e-30)
    rhoe = rhoE - 0.5*rho*u**2
    T = T_from_rhoe(r1, r2, rhoe, T_in=T_prev)
    return u, rhoe, T, srk_p(r1, r2, T)

def muscl_lr(q):
    dR, dL = np.roll(q, -1) - q, q - np.roll(q, 1)
    slp = np.where(dL*dR > 0.0, np.sign(dL)*np.minimum(np.abs(dL), np.abs(dR)), 0.0)
    return q + 0.5*slp, np.roll(q - 0.5*slp, -1)

# ─────────────────────────────────────────────────────────────
# Interface Fluxes - Rigorous implementation of Paper's Appendix A
# ─────────────────────────────────────────────────────────────
def interface_fluxes(r1, r2, u, rhoe, p, T, lam_cell, scheme, eps_pair=None, eos='SRK'):
    rho, rhoE = r1 + r2, rhoe + 0.5*(r1+r2)*u**2

    # 1. Standard MUSCL variables
    r1L, r1R = muscl_lr(r1)
    r2L, r2R = muscl_lr(r2)
    uL,  uR  = muscl_lr(u)
    pL,  pR  = muscl_lr(p)
    if eos == 'CPG':
        rhoEL = cpg_rhoe_from_p(r1L, r2L, pL) + 0.5*(r1L+r2L)*uL**2
        rhoER = cpg_rhoe_from_p(r1R, r2R, pR) + 0.5*(r1R+r2R)*uR**2
    else:
        rhoEL = srk_rhoe(r1L, r2L, T) + 0.5*(r1L+r2L)*uL**2
        rhoER = srk_rhoe(r1R, r2R, T) + 0.5*(r1R+r2R)*uR**2

    rhoL, rhoR = r1L + r2L, r1R + r2R
    lam = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # 2. MUSCL-LLF Fluxes (Base Upwind Scheme)
    F1_int = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2_int = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU_int = 0.5*(rhoL*uL**2 + pL + rhoR*uR**2 + pR) - 0.5*lam*(rhoR*uR - rhoL*uL)
    FE_int_upwind = 0.5*((rhoEL+pL)*uL + (rhoER+pR)*uR) - 0.5*lam*(rhoER - rhoEL)

    if scheme == 'APEC':
        eps0, eps1 = eps_pair
        if eos == 'CPG':
            # ── CPG: MUSCL-일관 PE 소산 (Appendix A 핵심 아이디어) ──
            # 에너지 소산항에 질량·운동량과 동일한 MUSCL 점프를 사용하여
            # 인터페이스 PE 오차를 줄인다.
            # drhoE = Σ_i ε_{i,h}·Δ(ρY_i) + ½u_h²·Δρ + ρ_h·u_h·Δu
            eps0_h = 0.5*(eps0 + np.roll(eps0, -1))
            eps1_h = 0.5*(eps1 + np.roll(eps1, -1))
            u_h_   = 0.5*(uL  + uR)
            rho_h_ = 0.5*(rhoL + rhoR)
            drhoE  = (eps0_h*(r1R - r1L) + eps1_h*(r2R - r2L)
                    + 0.5*u_h_**2 * ((r1R + r2R) - (r1L + r2L))
                    + rho_h_ * u_h_ * (uR - uL))
            FE_cen = 0.5*((rhoEL + pL)*uL + (rhoER + pR)*uR)
            FE_int = FE_cen - 0.5*lam*drhoE
        else:
            # ── SRK: Appendix A (Eq. A.4) ───────────────────────────
            F1_cell = r1 * u;  F2_cell = r2 * u
            FU_cell = rho * u**2 + p;  FE_cell = (rhoE + p) * u
            F1_m1, F2_m1 = np.roll(F1_cell,-1), np.roll(F2_cell,-1)
            FU_m1, FE_m1 = np.roll(FU_cell,-1), np.roll(FE_cell,-1)
            u_m1   = np.roll(u,   -1)
            eps0_m1= np.roll(eps0,-1); eps1_m1= np.roll(eps1,-1)
            c0m,  c1m  = eps0    - 0.5*u**2,    eps1    - 0.5*u**2
            c0m1, c1m1 = eps0_m1 - 0.5*u_m1**2, eps1_m1 - 0.5*u_m1**2
            tm  = c0m *(F1_int-F1_cell) + c1m *(F2_int-F2_cell) + u   *(FU_int-FU_cell)
            tm1 = c0m1*(F1_m1 -F1_int ) + c1m1*(F2_m1 -F2_int ) + u_m1*(FU_m1 -FU_int )
            FE_int = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1
    else:
        # FC-NPE: 표준 MUSCL-LLF 에너지 플럭스
        FE_int = FE_int_upwind

    return F1_int, F2_int, FU_int, FE_int

# ─────────────────────────────────────────────────────────────
# RHS Logic
# ─────────────────────────────────────────────────────────────
def rhs(U, scheme, dx, T_prev, flux='UPWIND'):
    r1, r2, rhoU, rhoE = U
    u, rhoe, T, p = prim(r1, r2, rhoU, rhoE, T_prev)

    eps_pair = (epsilon_v(r1, r2, T, 0), epsilon_v(r1, r2, T, 1)) if scheme in ('APEC', 'PEqC') else None

    if flux == 'KEEP':
        # Paper main scheme: KEEP split-form (Eq. 35-40), no numerical dissipation
        F1, F2, FU, FE = interface_fluxes_keep(r1, r2, u, rhoe, p, scheme, eps_pair=eps_pair)
    else:
        # Appendix A: MUSCL-LLF upwind
        lam_c = np.abs(u) + np.sqrt(srk_c2(r1, r2, T))
        F1, F2, FU, FE = interface_fluxes(r1, r2, u, rhoe, p, T, lam_c, scheme, eps_pair=eps_pair)

    d1 = -(F1 - np.roll(F1, 1)) / dx
    d2 = -(F2 - np.roll(F2, 1)) / dx
    dU = -(FU - np.roll(FU, 1)) / dx
    dE = -(FE - np.roll(FE, 1)) / dx

    if scheme == 'PEqC':
        rhoeu_m12 = 0.5*(rhoe + np.roll(rhoe, -1)) * 0.5*(u + np.roll(u, -1))
        rho1u_m12 = 0.5*(r1 + np.roll(r1, -1)) * 0.5*(u + np.roll(u, -1))
        rho2u_m12 = 0.5*(r2 + np.roll(r2, -1)) * 0.5*(u + np.roll(u, -1))
        d_rhoeu_dx = (rhoeu_m12 - np.roll(rhoeu_m12, 1)) / dx
        d_rho1u_dx = (rho1u_m12 - np.roll(rho1u_m12, 1)) / dx
        d_rho2u_dx = (rho2u_m12 - np.roll(rho2u_m12, 1)) / dx
        eps0, eps1 = eps_pair
        source_E = d_rhoeu_dx - (eps0 * d_rho1u_dx + eps1 * d_rho2u_dx)
        dE = -(FE - np.roll(FE, 1))/dx + source_E

    return [d1, d2, dU, dE], T, p

# ─────────────────────────────────────────────────────────────
# SSP-RK3 & Main Runner
# ─────────────────────────────────────────────────────────────
def _clip(U): return [np.maximum(U[0], 0.0), np.maximum(U[1], 0.0), U[2], U[3]]

def rkstep(U, scheme, dx, dt, T_prev, flux='UPWIND'):
    k1, T1, p1 = rhs(U, scheme, dx, T_prev, flux=flux)
    U1 = _clip([U[q] + dt*k1[q] for q in range(4)])
    k2, T2, p2 = rhs(U1, scheme, dx, T1, flux=flux)
    U2 = _clip([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])
    k3, T3, p3 = rhs(U2, scheme, dx, T2, flux=flux)
    return _clip([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)]), T3, p3

def pe_err(p, p0=5e6): return float(np.max(np.abs(p - p0)) / p0)
def energy_err(rhoE, rhoE0): return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-30))

def run(scheme, N=101, t_end=0.07, CFL=0.4, p_inf=5e6, k=15.0, flux='KEEP'):
    """SRK CH4/N2 인터페이스 이송 시뮬레이션.
    flux='KEEP'  : 논문 Eq.35-40 split-form (수치 점성 없음, 논문 메인 결과)
    flux='UPWIND': Appendix A MUSCL-LLF (수치 점성 있음)
    """
    dx = 1.0 / N
    x = np.linspace(dx/2, 1 - dx/2, N)
    print(f"\n{'='*55}\nScheme: {scheme}  flux={flux}  N={N}  t_end={t_end:.3f}  CFL={CFL}  k={k}")

    r1, r2, u, rhoE, T, p = initial_condition(x, p_inf, k=k)
    U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
    rhoE0 = rhoE.copy()

    t_hist, pe_hist, en_hist = [0.0], [pe_err(p, p_inf)], [0.0]
    t, step, diverged = 0.0, 0, False

    while t < t_end - 1e-14:
        lam = float(np.max(np.abs(U[2] / np.maximum(U[0]+U[1], 1e-30)) + np.sqrt(srk_c2(U[0], U[1], T))))
        dt  = min(CFL*dx/(lam + 1e-10), t_end - t)

        try:
            U, T, p = rkstep(U, scheme, dx, dt, T, flux=flux)
        except Exception as e:
            print(f"  Diverged at t={t:.5f}: {e}"); diverged = True; break

        t += dt; step += 1
        pe_ = pe_err(p, p_inf)
        en_ = energy_err(U[3], rhoE0) if scheme != 'PEqC' else np.nan
        t_hist.append(t); pe_hist.append(pe_); en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 5.0:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}"); diverged = True; break

    print(f"  --> {'Completed' if not diverged else 'Diverged'} at t={t:.5f} ({step} steps)")
    return x, U, T, p, np.array(t_hist), np.array(pe_hist), np.array(en_hist), diverged

# ─────────────────────────────────────────────────────────────
# KEEP split-form interface fluxes (Eq. 35-40)
# ─────────────────────────────────────────────────────────────
def interface_fluxes_keep(r1, r2, u, rhoe, p, scheme, eps_pair=None):
    """KEEP arithmetic-average fluxes (no numerical dissipation).

    Mass / momentum: Eq. 35-37 — half-point arithmetic means.
    Total energy   : Eq. 38-40 — internal energy half-point (FC or APEC
                     corrected) plus KEP-consistent KE and pressure-work terms.

    APEC correction (Eq. 40):
        ρe_{m+1/2} = (ρe_m + ρe_{m+1})/2
                   - Σ_i  Δε_i/2 · Δ(ρY_i)/2
    where Δ denotes the (m+1) − m jump.  This reduces the PE-error
    coefficient from 1/3 (FC-NPE) to 1/12 (APEC).
    """
    rho   = r1 + r2
    r1p   = np.roll(r1,   -1)
    r2p   = np.roll(r2,   -1)
    up    = np.roll(u,    -1)
    pp    = np.roll(p,    -1)
    rhoep = np.roll(rhoe, -1)
    rhop  = r1p + r2p

    u_h   = 0.5*(u   + up)     # half-point velocity
    rho_h = 0.5*(rho + rhop)   # half-point density
    p_h   = 0.5*(p   + pp)     # half-point pressure

    # ── Mass fluxes (Eq. 35) ─────────────────────────────────
    F1 = 0.5*(r1 + r1p) * u_h
    F2 = 0.5*(r2 + r2p) * u_h

    # ── Momentum flux — split / KEP form (Eq. 36-37) ────────
    FU = rho_h * u * up + p_h

    # ── KE + pressure-work terms (Eq. 38-39, 공통) ──────────
    F_KE = 0.5 * rho_h * u * up * u_h    # Eq. 38: KE transport (KEP)
    F_pu = 0.5 * (p * up + pp * u)        # Eq. 39: pressure work (symmetric)

    # ── 내부에너지 반점 및 총에너지 플럭스 ──────────────────
    if scheme == 'APEC':
        # Eq. 40: ρe_h = (ρe_m + ρe_{m+1})/2 - Σ_i Δε_i/2·Δ(ρY_i)/2
        # → PE 오차 계수 1/3 → 1/12 감소
        eps0, eps1 = eps_pair
        eps0p = np.roll(eps0, -1)
        eps1p = np.roll(eps1, -1)
        corr   = (0.5*(eps0p - eps0) * 0.5*(r1p - r1)
                + 0.5*(eps1p - eps1) * 0.5*(r2p - r2))
        rhoe_h = 0.5*(rhoe + rhoep) - corr
        FE = rhoe_h * u_h + F_KE + F_pu

    else:  # FC-NPE or PEqC (PEqC는 rhs_cpg에서 소스항으로 처리)
        # 단순 산술 평균: PE 오차 계수 1/3
        rhoe_h = 0.5*(rhoe + rhoep)
        FE = rhoe_h * u_h + F_KE + F_pu

    return F1, F2, FU, FE

# ─────────────────────────────────────────────────────────────
# CPG: primitive variables, rhs, runner
# ─────────────────────────────────────────────────────────────
def cpg_prim(r1, r2, rhoU, rhoE):
    rho  = r1 + r2
    u    = rhoU / np.maximum(rho, 1e-30)
    rhoe = rhoE - 0.5*rho*u**2
    p    = cpg_p_from_rhoe(r1, r2, rhoe)
    return u, rhoe, p

def rhs_cpg(U, scheme, dx):
    r1, r2, rhoU, rhoE = U
    u, rhoe, p = cpg_prim(r1, r2, rhoU, rhoE)
    eps_pair = cpg_epsilon(r1, r2, p) if scheme in ('APEC', 'PEqC') else None

    # KEEP 플럭스: PEqC는 FC 플럭스 기반 + 소스항 보정
    flux_scheme = 'FC' if scheme == 'PEqC' else scheme
    F1, F2, FU, FE = interface_fluxes_keep(r1, r2, u, rhoe, p,
                                            flux_scheme, eps_pair=eps_pair)

    d1 = -(F1 - np.roll(F1, 1)) / dx
    d2 = -(F2 - np.roll(F2, 1)) / dx
    dU = -(FU - np.roll(FU, 1)) / dx
    dE = -(FE - np.roll(FE, 1)) / dx

    if scheme == 'PEqC':
        # Eq. 54-56: 준보존 에너지 방정식
        # dρE/dt = -∂(F_KE + F_pu)/∂x - Σ_i ε_i · ∂F_{ρY_i}/∂x
        # 소스항: div(ρe_h·u_h) - Σ_i ε_i·div(F_i) 를 추가하면
        # dE = -div(FE_FC) + src = -div(F_KE+F_pu) - Σ ε_i·div(F_i)
        eps0, eps1 = eps_pair
        u_h     = 0.5*(u + np.roll(u, -1))
        rhoeu_h = 0.5*(rhoe + np.roll(rhoe, -1)) * u_h   # ρe_h · u_h
        src = ((rhoeu_h - np.roll(rhoeu_h, 1))
               - eps0*(F1 - np.roll(F1, 1))
               - eps1*(F2 - np.roll(F2, 1))) / dx
        dE = -(FE - np.roll(FE, 1))/dx + src

    return [d1, d2, dU, dE], p

def rhs_cpg_upwind(U, scheme, dx):
    """업윈드(MUSCL-LLF) 기반 CPG RHS — FC 또는 APEC (MUSCL-일관 PE 소산)."""
    r1, r2, rhoU, rhoE = U
    u, rhoe, p = cpg_prim(r1, r2, rhoU, rhoE)
    lam_c    = np.abs(u) + np.sqrt(cpg_c2(r1, r2, p))
    eps_pair = cpg_epsilon(r1, r2, p) if scheme == 'APEC' else None
    F1, F2, FU, FE = interface_fluxes(r1, r2, u, rhoe, p, None, lam_c,
                                       scheme, eps_pair=eps_pair, eos='CPG')
    d1 = -(F1 - np.roll(F1, 1)) / dx
    d2 = -(F2 - np.roll(F2, 1)) / dx
    dU = -(FU - np.roll(FU, 1)) / dx
    dE = -(FE - np.roll(FE, 1)) / dx
    return [d1, d2, dU, dE], p

def _clip_cpg(U):
    return [np.maximum(U[0], 1e-10), np.maximum(U[1], 1e-10), U[2], U[3]]

def rkstep_cpg(U, scheme, dx, dt, rhs_fn=None):
    """SSP-RK3. rhs_fn: rhs_cpg(KEEP, 기본) 또는 rhs_cpg_upwind."""
    if rhs_fn is None:
        rhs_fn = rhs_cpg
    k1, p1 = rhs_fn(U, scheme, dx)
    U1 = _clip_cpg([U[q] + dt*k1[q] for q in range(4)])
    k2, p2 = rhs_fn(U1, scheme, dx)
    U2 = _clip_cpg([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])
    k3, p3 = rhs_fn(U2, scheme, dx)
    return _clip_cpg([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)]), p3

def cpg_initial(x, p0=0.9, k=20.0):
    xc, rc, w1, w2 = 0.5, 0.25, 0.6, 0.2
    r = np.abs(x - xc)
    r1 = 0.5*w1*(1.0 - np.tanh(k*(r - rc)))
    r2 = 0.5*w2*(1.0 + np.tanh(k*(r - rc)))
    u  = np.ones_like(x)
    rhoe = cpg_rhoe_from_p(r1, r2, np.full_like(x, p0))
    rhoE = rhoe + 0.5*(r1+r2)*u**2
    return r1, r2, u, rhoE

def pe_err_cpg(p, p0=0.9):
    return float(np.sqrt(np.mean(((p - p0)/p0)**2)))

def run_cpg(scheme, N=501, t_end=8.0, CFL=0.6, p0=0.9, k=20.0, flux='KEEP'):
    """flux='KEEP'(기본) 또는 flux='UPWIND'(MUSCL-LLF)."""
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1 - dx/2, N)
    rhs_fn = rhs_cpg if flux == 'KEEP' else rhs_cpg_upwind
    print(f"\n{'='*55}\n[CPG-{flux}] Scheme: {scheme}  N={N}  t_end={t_end:.1f}  CFL={CFL}")

    r1, r2, u, rhoE = cpg_initial(x, p0, k)
    U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
    rhoE0 = rhoE.copy()

    _, _, p = cpg_prim(r1, r2, (r1+r2)*u, rhoE)
    t_hist, pe_hist, en_hist = [0.0], [pe_err_cpg(p, p0)], [0.0]
    t, step, diverged = 0.0, 0, False

    while t < t_end - 1e-14:
        u_now = U[2] / np.maximum(U[0]+U[1], 1e-30)
        _, _, p_now = cpg_prim(U[0], U[1], U[2], U[3])
        lam = float(np.max(np.abs(u_now) + np.sqrt(cpg_c2(U[0], U[1], p_now))))
        dt  = min(CFL*dx/(lam + 1e-10), t_end - t)

        try:
            U, p = rkstep_cpg(U, scheme, dx, dt, rhs_fn=rhs_fn)
        except Exception as e:
            print(f"  Diverged at t={t:.4f}: {e}"); diverged = True; break

        t += dt; step += 1
        pe_ = pe_err_cpg(p, p0)
        en_ = float(abs(np.sum(U[3]) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-30))
        t_hist.append(t); pe_hist.append(pe_); en_hist.append(en_)

        if not np.isfinite(pe_) or pe_ > 50.0:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.4f}"); diverged = True; break

        if step % 2000 == 0:
            print(f"  t={t:.3f}  PE={pe_:.2e}  E_err={en_:.2e}", flush=True)

    print(f"  --> {'Done' if not diverged else 'Diverged'} at t={t:.4f} ({step} steps)")
    return x, U, None, p, np.array(t_hist), np.array(pe_hist), np.array(en_hist), diverged

# ─────────────────────────────────────────────────────────────
# Validation §3.1: CPG smooth interface advection
# ─────────────────────────────────────────────────────────────
def _cpg_pe_error_spatial(r1, r2, p, dx, p0=0.9):
    """Leading-order PE error at each cell (finite-difference estimate)."""
    eps0, eps1 = cpg_epsilon(r1, r2, p)
    deps0 = (np.roll(eps0, -1) - np.roll(eps0, 1)) / (2*dx)
    deps1 = (np.roll(eps1, -1) - np.roll(eps1, 1)) / (2*dx)
    d2r1  = (np.roll(r1, -1) - 2*r1 + np.roll(r1, 1)) / dx**2
    d2r2  = (np.roll(r2, -1) - 2*r2 + np.roll(r2, 1)) / dx**2
    dr1   = (np.roll(r1, -1) - np.roll(r1, 1)) / (2*dx)
    dr2   = (np.roll(r2, -1) - np.roll(r2, 1)) / (2*dx)
    d2eps0 = (np.roll(eps0, -1) - 2*eps0 + np.roll(eps0, 1)) / dx**2
    d2eps1 = (np.roll(eps1, -1) - 2*eps1 + np.roll(eps1, 1)) / dx**2
    e_apec = ((1/12)*(deps0*d2r1 - d2eps0*dr1) +
              (1/12)*(deps1*d2r2 - d2eps1*dr2)) * dx**2
    e_fc   = ((1/3)*(deps0*d2r1) + (1/6)*(d2eps0*dr1) +
              (1/3)*(deps1*d2r2) + (1/6)*(d2eps1*dr2)) * dx**2
    return e_apec, e_fc

def validate_cpg_311(N=501, t_end=8.0, CFL=0.6):
    """Reproduce Figs 1-5 from §3.1, plus Fig 6: KEEP vs Upwind 비교."""
    schemes = ['FC', 'APEC', 'PEqC']
    results = {}
    for sch in schemes:
        results[sch] = run_cpg(sch, N=N, t_end=t_end, CFL=CFL, flux='KEEP')

    dx = 1.0 / N
    x  = results['FC'][0]

    # ── Fig 1: spatial distributions at t=t_end ──────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = {'FC': 'FC-NPE', 'APEC': 'APEC', 'PEqC': 'PEqC'}
    colors = {'FC': 'C0', 'APEC': 'C1', 'PEqC': 'C2'}
    for sch, res in results.items():
        x_, U, p, *_ = res
        r1, r2, rhoU, rhoE = U
        u_ = rhoU / np.maximum(r1+r2, 1e-30)
        ls = '--' if sch == 'FC' else '-'
        axes[0,0].plot(x_, r1, ls, color=colors[sch], label=labels[sch])
        axes[0,1].plot(x_, r2, ls, color=colors[sch], label=labels[sch])
        axes[1,0].plot(x_, u_,  ls, color=colors[sch], label=labels[sch])
        axes[1,1].plot(x_, p,   ls, color=colors[sch], label=labels[sch])
    for ax, lbl in zip(axes.flat, [r'$\rho_1$', r'$\rho_2$', r'$u$', r'$p$']):
        ax.set_xlabel('x'); ax.set_ylabel(lbl); ax.legend(fontsize=8)
    axes[1,1].axhline(0.9, color='k', lw=0.8, ls=':')
    fig.suptitle(f'Fig 1: Spatial distributions at t={t_end} (N={N})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig1_spatial.png'), dpi=150)
    plt.close(fig)
    print(f"Saved cpg_fig1_spatial.png")

    # ── Fig 2: PE error & energy conservation time history ────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for sch, res in results.items():
        x_, U, p, t_h, pe_h, en_h, div = res
        ls = '--' if sch == 'FC' else '-'
        ax1.semilogy(t_h, pe_h, ls, color=colors[sch], label=labels[sch])
        ax2.semilogy(t_h, np.maximum(en_h, 1e-20), ls, color=colors[sch], label=labels[sch])
    ax1.set_xlabel('t'); ax1.set_ylabel(r'$E_{PE}$'); ax1.legend(); ax1.set_title('PE error')
    ax2.set_xlabel('t'); ax2.set_ylabel(r'$E_{cons}$'); ax2.legend(); ax2.set_title('Energy conservation')
    fig.suptitle('Fig 2: Error time history')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig2_timehist.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig2_timehist.png")

    # ── Fig 3: PE error spatial at t=0 ───────────────────────
    r1_0, r2_0, u_0, rhoE_0 = cpg_initial(x, p0=0.9, k=20.0)
    p_0 = cpg_p_from_rhoe(r1_0, r2_0, rhoE_0 - 0.5*(r1_0+r2_0)*u_0**2)
    e_apec0, e_fc0 = _cpg_pe_error_spatial(r1_0, r2_0, p_0, dx)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, e_fc0,   'C0--', label='FC-NPE')
    ax.plot(x, e_apec0, 'C1-',  label='APEC')
    ax.set_xlabel('x'); ax.set_ylabel(r'$e_{PE}(x)$')
    ax.legend(); ax.set_title('Fig 3: Leading-order PE error spatial (t=0)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig3_pe_spatial.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig3_pe_spatial.png")

    # ── Fig 4: PE error norm time history ────────────────────
    #   ||e|| = rms of leading-order error (evolves as solution advects)
    #   Here we use the simulated PE rms as proxy (same as E_PE above)
    fig, ax = plt.subplots(figsize=(7, 4))
    for sch, res in results.items():
        x_, U, p, t_h, pe_h, en_h, div = res
        ls = '--' if sch == 'FC' else '-'
        ax.semilogy(t_h, pe_h, ls, color=colors[sch], label=labels[sch])
    ax.set_xlabel('t'); ax.set_ylabel(r'$\|e\|$')
    ax.legend(); ax.set_title('Fig 4: PE error norm vs time')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig4_pe_norm.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig4_pe_norm.png")

    # ── Fig 5: grid convergence at t=20 ──────────────────────
    print("\nRunning grid convergence (t=20, N=251 and N=501)...")
    conv_results = {}
    for Nc in [251, 501]:
        conv_results[Nc] = run_cpg('APEC', N=Nc, t_end=20.0, CFL=CFL)
        _, Uc, pc, *_ = conv_results[Nc]
        pe_rms = float(np.sqrt(np.mean(((pc - 0.9)/0.9)**2)))
        print(f"  N={Nc}: PE_rms={pe_rms:.4e}")

    dxs = [1/Nc for Nc in [251, 501]]
    pe_rms_vals = []
    for Nc in [251, 501]:
        _, Uc, pc, *_ = conv_results[Nc]
        pe_rms_vals.append(float(np.sqrt(np.mean(((pc - 0.9)/0.9)**2))))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dxs, pe_rms_vals, 'C1o-', label='APEC')
    dx_ref = np.array(dxs)
    ax.loglog(dx_ref, pe_rms_vals[0]*(dx_ref/dxs[0])**2, 'k--', label=r'$O(\Delta x^2)$')
    ax.set_xlabel(r'$\Delta x$'); ax.set_ylabel('PE rms error at t=20')
    ax.legend(); ax.set_title('Fig 5: Grid convergence')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig5_convergence.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig5_convergence.png")

    # ── Fig 6: KEEP vs Upwind PE 오차 비교 ───────────────────
    print("\nRunning upwind comparison (FC vs APEC, KEEP vs MUSCL-LLF)...")
    upwind_schemes = ['FC', 'APEC']
    upwind_results = {}
    for sch in upwind_schemes:
        upwind_results[sch] = run_cpg(sch, N=N, t_end=t_end, CFL=CFL, flux='UPWIND')

    fig, ax = plt.subplots(figsize=(8, 4))
    # KEEP 결과
    for sch in ['FC', 'APEC']:
        res = results[sch]
        x_, U, p_, t_h, pe_h, en_h, div = res
        ls = '--' if sch == 'FC' else '-'
        ax.semilogy(t_h, pe_h, ls, color='C0' if sch=='FC' else 'C1',
                    lw=2, label=f'KEEP-{sch}')
    # Upwind 결과
    for sch in upwind_schemes:
        res = upwind_results[sch]
        x_, U, p_, t_h, pe_h, en_h, div = res
        ls = '--' if sch == 'FC' else '-'
        ax.semilogy(t_h, pe_h, ls, color='C3' if sch=='FC' else 'C4',
                    lw=1.5, alpha=0.8, label=f'UPWIND-{sch}')
    ax.set_xlabel('t'); ax.set_ylabel(r'$E_{PE}$')
    ax.legend(ncol=2, fontsize=9)
    ax.set_title(f'Fig 6: KEEP vs Upwind PE error (N={N})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig6_keep_vs_upwind.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig6_keep_vs_upwind.png")

    # ── 업윈드 PE 개선율 출력 ────────────────────────────────
    print("\n=== PE improvement summary (t=8) ===")
    pe_fc_keep  = results['FC'][4][-1]
    pe_ap_keep  = results['APEC'][4][-1]
    pe_fc_upw   = upwind_results['FC'][4][-1]
    pe_ap_upw   = upwind_results['APEC'][4][-1]
    print(f"  KEEP   FC   PE = {pe_fc_keep:.3e}")
    print(f"  KEEP   APEC PE = {pe_ap_keep:.3e}  (improvement {pe_fc_keep/pe_ap_keep:.1f}x)")
    print(f"  UPWIND FC   PE = {pe_fc_upw:.3e}")
    print(f"  UPWIND APEC PE = {pe_ap_upw:.3e}  (improvement {pe_fc_upw/pe_ap_upw:.1f}x)")

    # ── Fig 7: p, u, T, Y1 공간 분포 비교 (t=t_end) ─────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_p, ax_u, ax_T, ax_Y = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # 초기 조건 (t=0) 참조선
    r1_0, r2_0, u_0, rhoE_0 = cpg_initial(x, p0=0.9, k=20.0)
    rhoe_0 = rhoE_0 - 0.5*(r1_0+r2_0)*u_0**2
    p_0    = cpg_p_from_rhoe(r1_0, r2_0, rhoe_0)
    T_0    = cpg_T(r1_0, r2_0, rhoe_0)
    Y1_0   = r1_0 / np.maximum(r1_0+r2_0, 1e-30)
    for ax, q, lbl in zip([ax_p, ax_u, ax_T, ax_Y],
                          [p_0, u_0, T_0, Y1_0],
                          ['p', 'u', 'T', 'Y1']):
        ax.plot(x, q, 'k:', lw=1.2, label='IC (t=0)')

    # KEEP 결과 (FC, APEC, PEqC)
    keep_styles = {'FC': ('C0', '--', 'KEEP-FC'),
                   'APEC': ('C1', '-',  'KEEP-APEC'),
                   'PEqC': ('C2', '-.',  'KEEP-PEqC')}
    for sch, (col, ls, lbl) in keep_styles.items():
        x_, U, p_, t_h, pe_h, en_h, div = results[sch]
        r1_, r2_, rhoU_, rhoE_ = U
        rho_ = r1_ + r2_
        u_   = rhoU_ / np.maximum(rho_, 1e-30)
        rhoe_= rhoE_ - 0.5*rho_*u_**2
        T_   = cpg_T(r1_, r2_, rhoe_)
        Y1_  = r1_ / np.maximum(rho_, 1e-30)
        ax_p.plot(x_, p_,  ls, color=col, lw=1.5, label=lbl)
        ax_u.plot(x_, u_,  ls, color=col, lw=1.5, label=lbl)
        ax_T.plot(x_, T_,  ls, color=col, lw=1.5, label=lbl)
        ax_Y.plot(x_, Y1_, ls, color=col, lw=1.5, label=lbl)

    # Upwind 결과 (FC, APEC)
    upw_styles = {'FC': ('C3', '--', 'UPWIND-FC'),
                  'APEC': ('C4', '-',  'UPWIND-APEC')}
    for sch, (col, ls, lbl) in upw_styles.items():
        x_, U, p_, t_h, pe_h, en_h, div = upwind_results[sch]
        r1_, r2_, rhoU_, rhoE_ = U
        rho_ = r1_ + r2_
        u_   = rhoU_ / np.maximum(rho_, 1e-30)
        rhoe_= rhoE_ - 0.5*rho_*u_**2
        T_   = cpg_T(r1_, r2_, rhoe_)
        Y1_  = r1_ / np.maximum(rho_, 1e-30)
        ax_p.plot(x_, p_,  ls, color=col, lw=1.2, alpha=0.8, label=lbl)
        ax_u.plot(x_, u_,  ls, color=col, lw=1.2, alpha=0.8, label=lbl)
        ax_T.plot(x_, T_,  ls, color=col, lw=1.2, alpha=0.8, label=lbl)
        ax_Y.plot(x_, Y1_, ls, color=col, lw=1.2, alpha=0.8, label=lbl)

    ax_p.axhline(0.9, color='k', lw=0.8, ls=':')
    for ax, lbl in zip([ax_p, ax_u, ax_T, ax_Y],
                       [r'$p$', r'$u$', r'$T$ (non-dim)', r'$Y_1$']):
        ax.set_xlabel('x'); ax.set_ylabel(lbl)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f'Fig 7: Spatial profiles at t={t_end}  (N={N})', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'cpg_fig7_profiles.png'), dpi=150)
    plt.close(fig)
    print("Saved cpg_fig7_profiles.png")

    print("\n=== Validation §3.1 complete. Outputs in solver/output/ ===")

if __name__ == '__main__':
    validate_cpg_311(N=501, t_end=8.0, CFL=0.6)

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


