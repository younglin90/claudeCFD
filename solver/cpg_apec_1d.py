"""
CPG-APEC 1D Validation
======================
Terashima, Ly, Ihme (JCP 524, 2025) — Section 3, Calorically Perfect Gas case.

지배방정식 : 1D 압축성 오일러 (Eq. 1-3)
공간 이산화 :
  KEEP split-form flux (Eq. 35-40)  → FC, APEC, Fujiwara, PEqC
  LLF upwind flux (Appendix A)      → FC_LLF, APEC_A
비교 방법  : FC-NPE, APEC, Fujiwara [15], PEqC (Eq. 54),
             FC_LLF (LLF 기준), APEC_A (Appendix A, Eq. A.4)
검증 케이스 : 1D smooth interface advection (Fig. 1-5)
              x∈[0,1], N=501, 주기 경계, CFL=0.6, t_end=8.0
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Species properties (Section 3)
# Component 0 : γ1=1.4,  M1=28 g/mol  (N2-like)
# Component 1 : γ2=1.66, M2=4  g/mol  (He-like)
# ─────────────────────────────────────────────────────────────
GAM  = np.array([1.4,  1.66])   # [-]
MW   = np.array([28.0, 4.0 ])   # g/mol (consistent units for mixing)

N_SPEC = 2


# ─────────────────────────────────────────────────────────────
# Calorically Perfect Gas EOS helpers
# ─────────────────────────────────────────────────────────────

def _mixing(r1, r2):
    """
    Returns n_tot (total molar concentration in consistent units),
    Mbar (mean molar mass), gam_bar (effective gamma).
    All inputs/outputs are arrays.

    n_tot = r1/M1 + r2/M2   [consistent molar density]
    Mbar  = rho / n_tot
    1/(gbar-1) = sum_i X_i/(gam_i-1)  where X_i = mole fraction
    """
    rho   = r1 + r2
    n1    = r1 / MW[0]
    n2    = r2 / MW[1]
    ntot  = n1 + n2
    Mbar  = rho / np.maximum(ntot, 1e-60)
    X1    = n1 / np.maximum(ntot, 1e-60)
    X2    = n2 / np.maximum(ntot, 1e-60)
    inv_gm1 = X1 / (GAM[0]-1.0) + X2 / (GAM[1]-1.0)   # 1/(gbar-1)
    gbar  = 1.0 + 1.0 / np.maximum(inv_gm1, 1e-60)
    return ntot, Mbar, gbar, inv_gm1


def cpg_rhoe(r1, r2, p):
    """ρe = p/(γ̄-1)  (Eq. 57)."""
    _, _, _, inv_gm1 = _mixing(r1, r2)
    return p * inv_gm1


def cpg_p(r1, r2, rhoe):
    """p = ρe*(γ̄-1)."""
    _, _, gbar, _ = _mixing(r1, r2)
    return rhoe * (gbar - 1.0)


def cpg_c2(r1, r2, p):
    """c² = γ̄ p / ρ."""
    _, _, gbar, _ = _mixing(r1, r2)
    rho = r1 + r2
    return gbar * p / np.maximum(rho, 1e-60)


def epsilon_cpg(r1, r2, p):
    """
    ε_i = (∂ρe/∂ρ_i)_{ρ_{j≠i}, p}   for i = 0, 1.  (Eq. 59)

    ε_i = p * Mbar²/rho² * (1/M_i) * [1/(γ_i-1) * n_tot - Σ_k n_k/(γ_k-1)]

    Returns (eps0, eps1) each of shape matching r1.
    """
    rho  = r1 + r2
    n1   = r1 / MW[0];  n2 = r2 / MW[1]
    ntot = n1 + n2
    Mbar = rho / np.maximum(ntot, 1e-60)

    B = n1/(GAM[0]-1.0) + n2/(GAM[1]-1.0)   # Σ_k n_k/(γ_k-1)
    pref = p * Mbar**2 / rho**2              # p * Mbar²/ρ²

    eps0 = pref / MW[0] * (ntot/(GAM[0]-1.0) - B)
    eps1 = pref / MW[1] * (ntot/(GAM[1]-1.0) - B)
    return eps0, eps1


# ─────────────────────────────────────────────────────────────
# Initial condition  (Eq. 62)
# ─────────────────────────────────────────────────────────────

def initial_condition(x, xc=0.5, rc=0.25, w1=0.6, w2=0.2, k=20.0,
                      u0=1.0, p0=0.9):
    r  = np.abs(x - xc)
    r1 = 0.5 * w1 * (1.0 - np.tanh(k * (r - rc)))   # ρY_1
    r2 = 0.5 * w2 * (1.0 + np.tanh(k * (r - rc)))   # ρY_2
    rho  = r1 + r2
    rhoe = cpg_rhoe(r1, r2, np.full_like(x, p0))
    u    = np.full_like(x, u0)
    rhoE = rhoe + 0.5 * rho * u**2
    p    = np.full_like(x, p0)
    return r1, r2, u, rhoE, p


# ─────────────────────────────────────────────────────────────
# Primitive variables from conservative
# ─────────────────────────────────────────────────────────────

def prim_from_cons(r1, r2, rhoU, rhoE):
    rho  = r1 + r2
    u    = rhoU / np.maximum(rho, 1e-60)
    rhoe = rhoE - 0.5 * rho * u**2
    p    = cpg_p(r1, r2, rhoe)
    return u, rhoe, p


# ─────────────────────────────────────────────────────────────
# KEEP split-form interface fluxes  (Eq. 35-39)
# Notation: _h = half-point (m+1/2), p = m+1 (via np.roll)
# ─────────────────────────────────────────────────────────────

def _roll(q):
    """Periodic roll to get cell m+1 values."""
    return np.roll(q, -1)


def keep_fluxes_mass_mom(r1, r2, u, p):
    """
    Mass (Eq. 35) and momentum (Eq. 36-37) fluxes — common to all methods.

    F_{ρY_i} = ((ρY_i)_m + (ρY_i)_{m+1})/2  *  (u_m + u_{m+1})/2
    F_{ρuu}  = ρ_h * u_h²
    F_p      = p_h
    """
    rho = r1 + r2
    r1p, r2p, up, rhop, pp = _roll(r1), _roll(r2), _roll(u), _roll(rho), _roll(p)

    u_h   = 0.5 * (u  + up)
    rho_h = 0.5 * (rho + rhop)
    p_h   = 0.5 * (p   + pp)

    F1 = 0.5 * (r1  + r1p) * u_h          # Eq. 35
    F2 = 0.5 * (r2  + r2p) * u_h          # Eq. 35
    FU = rho_h * u_h**2                    # Eq. 36 (convective part)
    # pressure handled separately in total momentum flux
    return F1, F2, FU, p_h, u_h, rho_h


def keep_energy_flux(r1, r2, u, p, rhoe, scheme, eps0=None, eps1=None):
    """
    Total energy flux = F_ρeu + F_KE + F_pu.

    F_KE  = ρ_h * (u_m * u_{m+1})/2 * u_h          (Eq. 38)
    F_pu  = (p_m * u_{m+1} + p_{m+1} * u_m) / 2    (Eq. 39)
    F_ρeu depends on scheme.
    """
    rho   = r1 + r2
    r1p   = _roll(r1);  r2p   = _roll(r2)
    up    = _roll(u);   pp    = _roll(p)
    rhoep = _roll(rhoe)
    rhop  = _roll(rho)

    u_h   = 0.5 * (u + up)
    rho_h = 0.5 * (rho + rhop)

    # KE flux  (Eq. 38)
    F_KE = rho_h * (u * up) / 2.0 * u_h

    # pressure-velocity flux  (Eq. 39)
    F_pu = 0.5 * (p * up + pp * u)

    # internal energy flux — scheme-dependent
    if scheme == 'FC':
        # FC-NPE: simple arithmetic average
        rhoe_h = 0.5 * (rhoe + rhoep)
        F_rhoe = rhoe_h * u_h

    elif scheme == 'APEC':
        # APEC: Eq. 40
        eps0p  = _roll(eps0)
        eps1p  = _roll(eps1)
        dr1    = r1p - r1
        dr2    = r2p - r2
        deps0  = eps0p - eps0
        deps1  = eps1p - eps1
        corr   = 0.5 * deps0 * 0.5 * dr1 + 0.5 * deps1 * 0.5 * dr2
        rhoe_h = 0.5 * (rhoe + rhoep) - corr
        F_rhoe = rhoe_h * u_h

    elif scheme == 'Fujiwara':
        # Fujiwara half-point ρe (Eq. 61): simple arithmetic
        rhoe_h = 0.5 * (rhoe + rhoep)
        F_rhoe = rhoe_h * u_h

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return F_rhoe + F_KE + F_pu


def fujiwara_fluxes(r1, r2, u, p):
    """
    Fujiwara's half-point (Eq. 60-61) — fully consistent fluxes.

    ρY_i|_{m+1/2} = 1/2 * [ wL*ρY_i_m + wR*ρY_i_{m+1} ]
    where  wL = ntot_{m+1}/ntot_m,  wR = ntot_m/ntot_{m+1}

    ρe|_{m+1/2}   = (ρe_m + ρe_{m+1})/2                  (Eq. 61)

    NOTE: total density half-point ρ_h is derived consistently from
    Fujiwara mass half-points (r1_h + r2_h) so that the momentum
    and KE fluxes are fully consistent with the mass fluxes.
    """
    rho  = r1 + r2
    n1   = r1 / MW[0];  n2 = r2 / MW[1]
    ntot = np.maximum(n1 + n2, 1e-60)

    r1p   = _roll(r1);  r2p = _roll(r2)
    rhop  = _roll(rho)
    n1p   = r1p / MW[0];  n2p = r2p / MW[1]
    ntotp = np.maximum(n1p + n2p, 1e-60)

    up = _roll(u);  pp = _roll(p)
    u_h = 0.5 * (u + up)
    p_h = 0.5 * (p + pp)

    # Fujiwara weights: wL = ntot_{m+1}/ntot_m
    wL = ntotp / ntot
    wR = ntot  / ntotp

    # species half-points (Eq. 60)
    r1_h = 0.5 * (wL * r1 + wR * r1p)
    r2_h = 0.5 * (wL * r2 + wR * r2p)
    rho_h = r1_h + r2_h   # consistent with mass fluxes

    # mass fluxes
    F1 = r1_h * u_h
    F2 = r2_h * u_h

    # momentum flux — use Fujiwara-consistent ρ_h
    FU = rho_h * u_h**2 + p_h

    # KE flux (Eq. 38) — use Fujiwara-consistent ρ_h
    F_KE = rho_h * (u * up) / 2.0 * u_h

    # pressure-velocity flux (Eq. 39)
    F_pu = 0.5 * (p * up + pp * u)

    # internal energy flux (Eq. 61): simple arithmetic
    rhoe  = cpg_rhoe(r1,  r2,  p)
    rhoep = cpg_rhoe(r1p, r2p, pp)
    rhoe_h = 0.5 * (rhoe + rhoep)
    F_rhoe = rhoe_h * u_h

    FE = F_rhoe + F_KE + F_pu
    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# Appendix A — HLLC 기반 플럭스  (FC_HLLC 및 APEC_A)
# ─────────────────────────────────────────────────────────────

def hllc_fluxes(r1, r2, u, p, rhoe, scheme, eps0=None, eps1=None):
    """
    HLLC (Harten-Lax-van Leer-Contact) 기반 플럭스.

    scheme='FC_HLLC' : 표준 HLLC — 모든 변수에 HLLC 플럭스 적용
    scheme='APEC_A'  : Appendix A (Eq. A.4)
                       질량·운동량은 HLLC, 에너지는 APEC correction

    파속 추정 (Einfeldt):
        S_L = u_L - c_L,  S_R = u_R + c_R
        S*  = (p_R - p_L + ρ_L u_L(S_L-u_L) - ρ_R u_R(S_R-u_R))
              / (ρ_L(S_L-u_L) - ρ_R(S_R-u_R))

    HLLC star state:
        ρ_i* = ρ_i * (S_k - u_k)/(S_k - S*)
        ρu*  = ρ_k* * S*
        ρE*  = ρ_k*(E_k + (S*-u_k)*(S* + p_k/(ρ_k(S_k-u_k)))) * (S_k-u_k)/(S_k-S*)

    Eq. A.4 (APEC_A energy flux):
    F_{ρE}|_{m+1/2} =
        0.5*(F_E|_m + F_E|_{m+1})
      + 0.5*Σ_i(ε_i - u²/2)|_m    * (F_{ρY_i}|_{m+1/2} - F_{ρY_i}|_m)
      + 0.5*u|_m                   * (F_{ρu}|_{m+1/2}   - F_{ρu}|_m)
      - 0.5*Σ_i(ε_i - u²/2)|_{m+1} * (F_{ρY_i}|_{m+1}  - F_{ρY_i}|_{m+1/2})
      - 0.5*u|_{m+1}               * (F_{ρu}|_{m+1}     - F_{ρu}|_{m+1/2})
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5 * rho * u**2

    # 우측 셀 (m+1)
    r1R   = _roll(r1);   r2R  = _roll(r2)
    uR    = _roll(u);    pR   = _roll(p)
    rhoR  = r1R + r2R
    rhoeR = _roll(rhoe)
    rhoER = rhoeR + 0.5 * rhoR * uR**2

    # 좌측 = 셀 m
    r1L, r2L, uL, pL, rhoL, rhoeL, rhoEL = r1, r2, u, p, rho, rhoe, rhoE

    # ── 음속 ──────────────────────────────────────────────────
    cL = np.sqrt(np.maximum(cpg_c2(r1L, r2L, pL), 0.0))
    cR = np.sqrt(np.maximum(cpg_c2(r1R, r2R, pR), 0.0))

    # ── 파속 추정 ─────────────────────────────────────────────
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    denom  = rhoL*(SL - uL) - rhoR*(SR - uR)
    denom  = np.where(np.abs(denom) > 1e-14, denom, 1e-14)
    S_star = (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / denom

    # ── 셀 중심 물리 플럭스 ───────────────────────────────────
    F1_L = r1L*uL;           F1_R = r1R*uR
    F2_L = r2L*uL;           F2_R = r2R*uR
    FU_L = rhoL*uL**2 + pL;  FU_R = rhoR*uR**2 + pR
    FE_L = (rhoEL + pL)*uL;  FE_R = (rhoER + pR)*uR

    # ── HLLC star states ──────────────────────────────────────
    # 공통 팩터 (S_k - u_k)/(S_k - S*)
    facL_d = np.where(np.abs(SL - S_star) > 1e-14, SL - S_star, 1e-14)
    facR_d = np.where(np.abs(SR - S_star) > 1e-14, SR - S_star, 1e-14)
    facL = (SL - uL) / facL_d
    facR = (SR - uR) / facR_d

    r1L_s  = r1L  * facL;        r1R_s  = r1R  * facR
    r2L_s  = r2L  * facL;        r2R_s  = r2R  * facR
    rhoUL_s = rhoL * facL * S_star;  rhoUR_s = rhoR * facR * S_star

    EL = rhoEL / np.maximum(rhoL, 1e-60)
    ER = rhoER / np.maximum(rhoR, 1e-60)
    rhoEL_s = rhoL * facL * (EL + (S_star - uL)*(S_star + pL/(rhoL*(SL - uL) + 1e-60)))
    rhoER_s = rhoR * facR * (ER + (S_star - uR)*(S_star + pR/(rhoR*(SR - uR) + 1e-60)))

    # ── HLLC interface fluxes: F_k + S_k*(Q_k* - Q_k) ───────
    F1_Ls = F1_L + SL*(r1L_s  - r1L);   F1_Rs = F1_R + SR*(r1R_s  - r1R)
    F2_Ls = F2_L + SL*(r2L_s  - r2L);   F2_Rs = F2_R + SR*(r2R_s  - r2R)
    FU_Ls = FU_L + SL*(rhoUL_s - rhoL*uL); FU_Rs = FU_R + SR*(rhoUR_s - rhoR*uR)
    FE_Ls = FE_L + SL*(rhoEL_s - rhoEL); FE_Rs = FE_R + SR*(rhoER_s - rhoER)

    # ── 파속에 따른 상태 선택 ─────────────────────────────────
    def _sel(fL, fLs, fRs, fR):
        return np.where(SL >= 0,  fL,
               np.where(S_star >= 0, fLs,
               np.where(SR >= 0,  fRs, fR)))

    F1 = _sel(F1_L, F1_Ls, F1_Rs, F1_R)
    F2 = _sel(F2_L, F2_Ls, F2_Rs, F2_R)
    FU = _sel(FU_L, FU_Ls, FU_Rs, FU_R)

    if scheme == 'FC_HLLC':
        FE = _sel(FE_L, FE_Ls, FE_Rs, FE_R)

    else:   # APEC_A — Eq. A.4
        eps0R = _roll(eps0);  eps1R = _roll(eps1)

        # h_i = (ε_i - u²/2)  at left(m) and right(m+1)
        h0L = eps0  - 0.5*uL**2;   h0R = eps0R - 0.5*uR**2
        h1L = eps1  - 0.5*uL**2;   h1R = eps1R - 0.5*uR**2

        # 좌측 셀(m)  기여 항 (Eq. A.2)
        corr_L = h0L*(F1 - F1_L) + h1L*(F2 - F2_L) + uL*(FU - FU_L)

        # 우측 셀(m+1) 기여 항 (Eq. A.3)
        corr_R = h0R*(F1_R - F1) + h1R*(F2_R - F2) + uR*(FU_R - FU)

        FE = 0.5*(FE_L + FE_R) + 0.5*corr_L - 0.5*corr_R

    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# Roe 기반 플럭스  (FC_Roe 및 APEC_A_Roe)
# ─────────────────────────────────────────────────────────────

def roe_fluxes(r1, r2, u, p, rhoe, scheme, eps0=None, eps1=None):
    """
    Roe scheme with Harten entropy fix.
    scheme='FC_Roe'    : 표준 Roe 에너지 플럭스
    scheme='APEC_A_Roe': Roe 질량·운동량 + Eq. A.4 에너지 보정
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1R, r2R = _roll(r1), _roll(r2)
    uR, pR   = _roll(u),  _roll(p)
    rhoR     = r1R + r2R
    rhoeR    = _roll(rhoe)
    rhoER    = rhoeR + 0.5*rhoR*uR**2

    r1L, r2L, uL, pL, rhoL, rhoeL, rhoEL = r1, r2, u, p, rho, rhoe, rhoE

    HL = (rhoEL + pL) / np.maximum(rhoL, 1e-60)
    HR = (rhoER + pR) / np.maximum(rhoR, 1e-60)
    Y1L = r1L / np.maximum(rhoL, 1e-60)
    Y2L = r2L / np.maximum(rhoL, 1e-60)
    Y1R = r1R / np.maximum(rhoR, 1e-60)

    # Roe 평균
    sqL   = np.sqrt(np.maximum(rhoL, 1e-60))
    sqR   = np.sqrt(np.maximum(rhoR, 1e-60))
    denom = sqL + sqR

    u_r   = (sqL*uL + sqR*uR) / denom
    H_r   = (sqL*HL + sqR*HR) / denom
    Y1_r  = (sqL*Y1L + sqR*Y1R) / denom
    Y2_r  = 1.0 - Y1_r
    rho_r = sqL * sqR   # sqrt(ρL*ρR)

    # Roe-평균 γ 및 음속
    n1_r     = Y1_r / MW[0];  n2_r = Y2_r / MW[1]
    ntot_r   = np.maximum(n1_r + n2_r, 1e-60)
    X1_r     = n1_r / ntot_r;  X2_r = n2_r / ntot_r
    inv_gm1_r = X1_r/(GAM[0]-1.0) + X2_r/(GAM[1]-1.0)
    gam_r    = 1.0 + 1.0/np.maximum(inv_gm1_r, 1e-60)
    c2_r     = np.maximum((gam_r - 1.0)*(H_r - 0.5*u_r**2), 1e-10)
    c_r      = np.sqrt(c2_r)

    # Harten entropy fix
    delta = 0.1 * c_r
    def abs_fix(lam):
        return np.where(np.abs(lam) >= delta, np.abs(lam),
                        (lam**2 + delta**2) / (2.0*delta))

    lam1 = abs_fix(u_r - c_r)
    lam2 = abs_fix(u_r)
    lam4 = abs_fix(u_r + c_r)

    # 파동 강도
    dp = pR - pL;  du = uR - uL
    dr1 = r1R - r1L;  dr2 = r2R - r2L

    alpha1 = (dp - rho_r*c_r*du) / (2.0*c2_r)
    alpha4 = (dp + rho_r*c_r*du) / (2.0*c2_r)
    alpha2 = dr1 - Y1_r*(dp/c2_r)   # 성분1 접촉파
    alpha3 = dr2 - Y2_r*(dp/c2_r)   # 성분2 접촉파

    # Roe 소산
    diss_r1 = lam1*alpha1*Y1_r + lam2*alpha2              + lam4*alpha4*Y1_r
    diss_r2 = lam1*alpha1*Y2_r + lam2*alpha3              + lam4*alpha4*Y2_r
    diss_rU = (lam1*alpha1*(u_r - c_r)
             + lam2*(alpha2 + alpha3)*u_r
             + lam4*alpha4*(u_r + c_r))

    # 셀 중심 물리 플럭스
    F1_L = r1L*uL;           F1_R = r1R*uR
    F2_L = r2L*uL;           F2_R = r2R*uR
    FU_L = rhoL*uL**2 + pL;  FU_R = rhoR*uR**2 + pR
    FE_L = (rhoEL + pL)*uL;  FE_R = (rhoER + pR)*uR

    F1 = 0.5*(F1_L + F1_R) - 0.5*diss_r1
    F2 = 0.5*(F2_L + F2_R) - 0.5*diss_r2
    FU = 0.5*(FU_L + FU_R) - 0.5*diss_rU

    if scheme == 'FC_Roe':
        # 에너지 Roe 소산 — Roe 상태에서 ε_i 필요
        p_r    = rho_r * c2_r / np.maximum(gam_r, 1e-10)
        eps0_r, eps1_r = epsilon_cpg(Y1_r*rho_r, Y2_r*rho_r, p_r)
        diss_rE = (lam1*alpha1*(H_r - u_r*c_r)
                 + lam2*alpha2*(eps0_r + 0.5*u_r**2)
                 + lam2*alpha3*(eps1_r + 0.5*u_r**2)
                 + lam4*alpha4*(H_r + u_r*c_r))
        FE = 0.5*(FE_L + FE_R) - 0.5*diss_rE

    else:   # APEC_A_Roe — Eq. A.4
        eps0R = _roll(eps0);  eps1R = _roll(eps1)
        h0L = eps0  - 0.5*uL**2;  h0R = eps0R - 0.5*uR**2
        h1L = eps1  - 0.5*uL**2;  h1R = eps1R - 0.5*uR**2
        corr_L = h0L*(F1 - F1_L) + h1L*(F2 - F2_L) + uL*(FU - FU_L)
        corr_R = h0R*(F1_R - F1) + h1R*(F2_R - F2) + uR*(FU_R - FU)
        FE = 0.5*(FE_L + FE_R) + 0.5*corr_L - 0.5*corr_R

    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# SLAU 기반 플럭스  (FC_SLAU 및 APEC_A_SLAU)
# ─────────────────────────────────────────────────────────────

def slau_fluxes(r1, r2, u, p, rhoe, scheme, eps0=None, eps1=None):
    """
    SLAU (Shima & Kitamura 2011) — Simple Low-dissipation AUSM.
    scheme='FC_SLAU'    : 표준 SLAU 에너지 플럭스
    scheme='APEC_A_SLAU': SLAU 질량·운동량 + Eq. A.4 에너지 보정

    질량 플럭스 (Eq. 12):
        ṁ = 0.5*(ρL*(uL+|uL|) + ρR*(uR-|uR|)) - χ*(pR-pL)/c_{1/2}
        χ = (1 - min(V̄/c_{1/2}, 1))²,  V̄ = 0.5*(|uL|+|uR|)

    압력: p_{1/2} = 0.5*(pL + pR)   [SLAU 근사]
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    r1R, r2R = _roll(r1), _roll(r2)
    uR, pR   = _roll(u),  _roll(p)
    rhoR     = r1R + r2R
    rhoeR    = _roll(rhoe)
    rhoER    = rhoeR + 0.5*rhoR*uR**2

    r1L, r2L, uL, pL, rhoL, rhoeL, rhoEL = r1, r2, u, p, rho, rhoe, rhoE

    cL  = np.sqrt(np.maximum(cpg_c2(r1L, r2L, pL), 1e-10))
    cR  = np.sqrt(np.maximum(cpg_c2(r1R, r2R, pR), 1e-10))
    c_h = 0.5*(cL + cR)

    # 저속 센서 χ
    V_bar = 0.5*(np.abs(uL) + np.abs(uR))
    chi   = (1.0 - np.minimum(V_bar / np.maximum(c_h, 1e-10), 1.0))**2

    # SLAU 질량 플럭스
    m_dot = (0.5*(rhoL*(uL + np.abs(uL)) + rhoR*(uR - np.abs(uR)))
             - chi * (pR - pL) / np.maximum(c_h, 1e-10))

    m_pos = np.maximum(m_dot, 0.0)
    m_neg = np.minimum(m_dot, 0.0)

    # 성분 플럭스 (질량 플럭스 부호로 풍상)
    F1 = m_pos*(r1L/np.maximum(rhoL, 1e-60)) + m_neg*(r1R/np.maximum(rhoR, 1e-60))
    F2 = m_pos*(r2L/np.maximum(rhoL, 1e-60)) + m_neg*(r2R/np.maximum(rhoR, 1e-60))

    # 운동량 플럭스 (대류 + 압력)
    u_up = np.where(m_dot >= 0, uL, uR)
    p_h  = 0.5*(pL + pR)
    FU   = m_dot*u_up + p_h

    # 셀 중심 물리 플럭스
    F1_L = r1L*uL;           F1_R = r1R*uR
    F2_L = r2L*uL;           F2_R = r2R*uR
    FU_L = rhoL*uL**2 + pL;  FU_R = rhoR*uR**2 + pR
    FE_L = (rhoEL + pL)*uL;  FE_R = (rhoER + pR)*uR

    if scheme == 'FC_SLAU':
        HL = (rhoEL + pL) / np.maximum(rhoL, 1e-60)
        HR = (rhoER + pR) / np.maximum(rhoR, 1e-60)
        FE = np.where(m_dot >= 0, m_dot*HL, m_dot*HR)

    else:   # APEC_A_SLAU — Eq. A.4
        eps0R = _roll(eps0);  eps1R = _roll(eps1)
        h0L = eps0  - 0.5*uL**2;  h0R = eps0R - 0.5*uR**2
        h1L = eps1  - 0.5*uL**2;  h1R = eps1R - 0.5*uR**2
        corr_L = h0L*(F1 - F1_L) + h1L*(F2 - F2_L) + uL*(FU - FU_L)
        corr_R = h0R*(F1_R - F1) + h1R*(F2_R - F2) + uR*(FU_R - FU)
        FE = 0.5*(FE_L + FE_R) + 0.5*corr_L - 0.5*corr_R

    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# Chandrasekhar EC + ES 스킴  (2-종 CPG 확장)
# ─────────────────────────────────────────────────────────────

def _log_mean(aL, aR):
    """
    수치적으로 안정한 로그 평균:  (aR - aL) / (ln aR - ln aL).
    aL ≈ aR 이면 Taylor 급수 사용.
    """
    xi   = np.maximum(aR, 1e-300) / np.maximum(aL, 1e-300)
    f    = (xi - 1.0) / (xi + 1.0)
    u2   = f * f
    small = u2 < 1e-6
    F_small = 1.0 + u2 / 3.0 + u2**2 / 5.0 + u2**3 / 7.0
    F_large = np.log(np.maximum(xi, 1e-300)) / (
                  2.0 * np.where(np.abs(f) > 1e-15, f, 1e-15))
    F = np.where(small, F_small, F_large)
    return (aL + aR) / (2.0 * F)


def ec_chn_fluxes(r1, r2, u, p, rhoe, eps0, eps1):
    """
    Chandrasekhar-inspired EC flux (2-종 CPG 확장).

    질량 플럭스 : 로그평균 밀도 × 산술평균 성분비 × 산술평균 속도
    운동량 플럭스: ρ_ln · u_h² + p̂  (p̂ = ρ_ln / (2β_avg), β = ρ/(2p))
    에너지 플럭스: APEC Eq.40 보정 ρe (PE 일관) + KEEP KE + KEEP pu
                  → 단성분 Chandrasekhar와 달리 혼합 γ 변화를 APEC 방식으로 처리
    소산 항 없음 (순수 EC-like 중심 플럭스).
    """
    rho  = r1 + r2
    r1R  = _roll(r1);  r2R  = _roll(r2)
    uR   = _roll(u);   pR   = _roll(p)
    rhoeR = _roll(rhoe)
    rhoR = r1R + r2R

    # ── 로그평균 밀도 ─────────────────────────────────────────
    rho_ln = _log_mean(rho, rhoR)

    # ── Chandrasekhar 계면 압력 : p̂ = ρ_ln/(2β_avg) ─────────
    betaL   = rho  / (2.0 * np.maximum(p,  1e-300))
    betaR_  = rhoR / (2.0 * np.maximum(pR, 1e-300))
    beta_avg = 0.5 * (betaL + betaR_)
    p_hat    = rho_ln / (2.0 * beta_avg)

    u_h = 0.5 * (u + uR)

    # ── 성분비 (산술평균) ─────────────────────────────────────
    Y1_avg = 0.5 * (r1  / np.maximum(rho,  1e-300)
                  + r1R / np.maximum(rhoR, 1e-300))
    Y2_avg = 1.0 - Y1_avg

    # ── 질량 플럭스 ───────────────────────────────────────────
    F1 = rho_ln * Y1_avg * u_h
    F2 = rho_ln * Y2_avg * u_h

    # ── 운동량 플럭스 ─────────────────────────────────────────
    FU = rho_ln * u_h**2 + p_hat

    # ── 에너지 플럭스 (APEC Eq.40 보정 ρe) ───────────────────
    eps0p = _roll(eps0);  eps1p = _roll(eps1)
    dr1   = r1R - r1;    dr2   = r2R - r2
    corr  = 0.5 * (eps0p - eps0) * 0.5 * dr1 + 0.5 * (eps1p - eps1) * 0.5 * dr2
    rhoe_h = 0.5 * (rhoe + rhoeR) - corr

    F_KE = rho_ln * (u * uR) / 2.0 * u_h   # KEEP KE (로그평균 ρ)
    F_pu = 0.5 * (p * uR + pR * u)          # KEEP pu
    FE   = rhoe_h * u_h + F_KE + F_pu

    return F1, F2, FU, FE


def _roe_diss_4eq(r1, r2, u, p, rhoe, eps0, eps1):
    """
    Roe 소산 벡터 (4방정식 2-종 CPG):
      d = F_cen - F_Roe = 0.5 |A_Roe| (q_R - q_L)

    기존 roe_fluxes('FC_Roe')를 재사용하여 추출.
    """
    rho  = r1 + r2
    rhoE = rhoe + 0.5 * rho * u**2

    r1R  = _roll(r1);  r2R  = _roll(r2)
    uR   = _roll(u);   pR   = _roll(p)
    rhoR = r1R + r2R
    rhoeR = _roll(rhoe)
    rhoER = rhoeR + 0.5 * rhoR * uR**2

    # 산술평균 중심 플럭스
    F1_cen = 0.5 * (r1 * u    + r1R * uR)
    F2_cen = 0.5 * (r2 * u    + r2R * uR)
    FU_cen = 0.5 * (rho * u**2 + p   + rhoR * uR**2 + pR)
    FE_cen = 0.5 * ((rhoE  + p)  * u  + (rhoER + pR) * uR)

    # Roe 플럭스 (기존 함수 재사용)
    F1_roe, F2_roe, FU_roe, FE_roe = roe_fluxes(
        r1, r2, u, p, rhoe, 'FC_Roe', eps0, eps1)

    return (F1_cen - F1_roe,
            F2_cen - F2_roe,
            FU_cen - FU_roe,
            FE_cen - FE_roe)


def es_chn_roe_fluxes(r1, r2, u, p, rhoe, eps0, eps1):
    """
    ES_CHN_Roe : EC (Chandrasekhar) + Roe 소산 (Eq.64).

    f*_ES = f*_EC_CHN - (F_cen - F_Roe)
          = f*_EC_CHN - 0.5 |A_Roe| (q_R - q_L)
    """
    F1, F2, FU, FE = ec_chn_fluxes(r1, r2, u, p, rhoe, eps0, eps1)
    d1, d2, dU, dE = _roe_diss_4eq(r1, r2, u, p, rhoe, eps0, eps1)
    return F1 - d1, F2 - d2, FU - dU, FE - dE


def es_chn_apec_fluxes(r1, r2, u, p, rhoe, eps0, eps1):
    """
    ES_CHN_APEC : EC mass/mom + APEC energy + Roe 소산.

    질량·운동량: Chandrasekhar EC (로그평균 ρ, p_hat)
    에너지     : APEC Eq.40 centered + Roe 에너지 소산
    → FC_Roe 대비: 중심 플럭스를 EC로 개선
    → APEC_A_Roe 대비: 질량/운동량 중심 플럭스도 EC로 개선
    """
    F1, F2, FU, _ = ec_chn_fluxes(r1, r2, u, p, rhoe, eps0, eps1)
    # APEC 에너지 중심 플럭스
    FE_apec = keep_energy_flux(r1, r2, u, p, rhoe, 'APEC', eps0, eps1)
    # Roe 소산
    d1, d2, dU, dE = _roe_diss_4eq(r1, r2, u, p, rhoe, eps0, eps1)
    return F1 - d1, F2 - d2, FU - dU, FE_apec - dE


# ─────────────────────────────────────────────────────────────
# KEEPPE-Q2 and KEEPPE-R-RA  (2L-th order split-form fluxes)
# ─────────────────────────────────────────────────────────────

# 정규화 조건: 2 × Σ_{l=1}^{L} a_{L,l} × l = 1
# L=1: a_{1,1} = 1/2
# L=4: a = [4/5, -1/5, 4/105, -1/280]  → 2×(4/5×1 - 1/5×2 + 4/105×3 - 1/280×4) = 1 ✓
_KEEPPE_COEFS = {
    1: [0.5],
    4: [4/5, -1/5, 4/105, -1/280],
}


def keeppe_q2_fluxes(r1, r2, u, p, rhoe, L=4):
    """
    KEEPPE-Q2: 2L-th order quadratic-split fluxes (arithmetic means).

    Interface m+1/2 flux (이중합 구조):
      F = 2 × Σ_{l=1}^{L} a_{L,l} × Σ_{k=0}^{l-1} f(m-k, m-k+l)

    k-번째 쌍:  Left = 셀 m-k  (np.roll(q, k))
               Right = 셀 m-k+l (np.roll(q, -(l-k)))

    At L=1 (2nd order, a=0.5): 표준 KEEP/FC 2차 플럭스와 동일.
    At L=4 (8th order): a = [4/5, -1/5, 4/105, -1/280].
    """
    a = _KEEPPE_COEFS[L]
    F1      = np.zeros_like(r1)
    F2      = np.zeros_like(r2)
    FU_conv = np.zeros_like(u)
    FG      = np.zeros_like(p)    # Ĝ: 압력 (고차합)
    F_KE    = np.zeros_like(u)
    F_pu    = np.zeros_like(u)
    F_rhoe  = np.zeros_like(rhoe)

    for l in range(1, L + 1):
        al = a[l - 1]
        for k in range(l):          # k = 0, 1, ..., l-1
            # 셀 m-k  ↔  셀 m+(l-k)
            r1L = np.roll(r1,    k);   r1R = np.roll(r1,   -(l - k))
            r2L = np.roll(r2,    k);   r2R = np.roll(r2,   -(l - k))
            uL  = np.roll(u,     k);   uR  = np.roll(u,    -(l - k))
            pL  = np.roll(p,     k);   pR  = np.roll(p,    -(l - k))
            rhoeL = np.roll(rhoe, k);  rhoeR = np.roll(rhoe, -(l - k))
            rhoL  = r1L + r2L;         rhoR  = r1R + r2R

            rho_h  = 0.5 * (rhoL  + rhoR)
            u_h    = 0.5 * (uL    + uR)
            r1_h   = 0.5 * (r1L   + r1R)
            r2_h   = 0.5 * (r2L   + r2R)
            rhoe_h = 0.5 * (rhoeL + rhoeR)
            p_h    = 0.5 * (pL    + pR)

            c = 2.0 * al
            F1      += c * r1_h  * u_h
            F2      += c * r2_h  * u_h
            FU_conv += c * rho_h * u_h**2
            FG      += c * p_h
            F_KE    += c * rho_h * (uL * uR) / 2.0 * u_h
            F_pu    += c * 0.5 * (pL * uR + pR * uL)
            F_rhoe  += c * rhoe_h * u_h

    FU = FU_conv + FG
    FE = F_rhoe + F_KE + F_pu
    return F1, F2, FU, FE


def keeppe_rra_fluxes(r1, r2, u, p, rhoe, L=4):
    """
    KEEPPE-R-RA: 2L-th order with geometric-mean ρ and ρe (Roe-type).

    같은 이중합 구조, 단 ρ와 ρe를 기하평균으로 대체:
      ρ_geo^{(l,k)}   = √(ρ_{m-k}  × ρ_{m+(l-k)})
      ρe_geo^{(l,k)}  = √(ρe_{m-k} × ρe_{m+(l-k)})
    """
    a = _KEEPPE_COEFS[L]
    F1      = np.zeros_like(r1)
    F2      = np.zeros_like(r2)
    FU_conv = np.zeros_like(u)
    FG      = np.zeros_like(p)
    F_KE    = np.zeros_like(u)
    F_pu    = np.zeros_like(u)
    F_rhoe  = np.zeros_like(rhoe)

    for l in range(1, L + 1):
        al = a[l - 1]
        for k in range(l):
            r1L = np.roll(r1,    k);   r1R = np.roll(r1,   -(l - k))
            r2L = np.roll(r2,    k);   r2R = np.roll(r2,   -(l - k))
            uL  = np.roll(u,     k);   uR  = np.roll(u,    -(l - k))
            pL  = np.roll(p,     k);   pR  = np.roll(p,    -(l - k))
            rhoeL = np.roll(rhoe, k);  rhoeR = np.roll(rhoe, -(l - k))
            rhoL  = r1L + r2L;         rhoR  = r1R + r2R

            rho_geo  = np.sqrt(np.maximum(rhoL  * rhoR,  0.0))
            rhoe_geo = np.sqrt(np.maximum(rhoeL * rhoeR, 0.0))
            u_h  = 0.5 * (uL + uR)
            p_h  = 0.5 * (pL + pR)
            Y1_h = 0.5 * (r1L / np.maximum(rhoL, 1e-60)
                         + r1R / np.maximum(rhoR, 1e-60))
            Y2_h = 0.5 * (r2L / np.maximum(rhoL, 1e-60)
                         + r2R / np.maximum(rhoR, 1e-60))

            c = 2.0 * al
            F1      += c * rho_geo  * u_h * Y1_h
            F2      += c * rho_geo  * u_h * Y2_h
            FU_conv += c * rho_geo  * u_h**2
            FG      += c * p_h
            F_KE    += c * rho_geo  * (uL * uR) / 2.0 * u_h
            F_pu    += c * 0.5 * (pL * uR + pR * uL)
            F_rhoe  += c * rhoe_geo * u_h

    FU = FU_conv + FG
    FE = F_rhoe + F_KE + F_pu
    return F1, F2, FU, FE


# ─────────────────────────────────────────────────────────────
# RHS — returns [dρ1, dρ2, dρu, dρE] / dt
# ─────────────────────────────────────────────────────────────

def rhs(U, scheme, dx):
    r1, r2, rhoU, rhoE = U
    u, rhoe, p = prim_from_cons(r1, r2, rhoU, rhoE)

    eps0 = eps1 = None
    if scheme in ('APEC', 'PEqC', 'APEC_A', 'APEC_A_KEEP',
                  'APEC_A_Roe', 'APEC_A_SLAU',
                  'APEC_HYB_HLLC', 'APEC_HYB_Roe', 'APEC_HYB_SLAU',
                  'EC_CHN', 'ES_CHN_Roe', 'ES_CHN_APEC'):
        eps0, eps1 = epsilon_cpg(r1, r2, p)

    # ── Chandrasekhar EC / ES 스킴 ────────────────────────────
    if scheme in ('EC_CHN', 'ES_CHN_Roe', 'ES_CHN_APEC'):
        if scheme == 'EC_CHN':
            F1, F2, FU, FE = ec_chn_fluxes(r1, r2, u, p, rhoe, eps0, eps1)
        elif scheme == 'ES_CHN_Roe':
            F1, F2, FU, FE = es_chn_roe_fluxes(r1, r2, u, p, rhoe, eps0, eps1)
        else:  # ES_CHN_APEC
            F1, F2, FU, FE = es_chn_apec_fluxes(r1, r2, u, p, rhoe, eps0, eps1)
        def div(f): return (f - np.roll(f, 1)) / dx
        d1=-div(F1); d2=-div(F2); dU=-div(FU); dE=-div(FE)
        return [d1, d2, dU, dE], p

    # ── 하이브리드: 연속·운동량·성분 = 풍상, 에너지 = APEC Eq.40 (KEEP split-form) ──
    if scheme in ('APEC_HYB_HLLC', 'APEC_HYB_Roe', 'APEC_HYB_SLAU'):
        base = scheme.split('_')[-1]   # 'HLLC', 'Roe', 'SLAU'
        if base == 'HLLC':
            F1, F2, FU, _ = hllc_fluxes(r1, r2, u, p, rhoe, 'FC_HLLC')
        elif base == 'Roe':
            F1, F2, FU, _ = roe_fluxes(r1, r2, u, p, rhoe, 'FC_Roe')
        else:  # SLAU
            F1, F2, FU, _ = slau_fluxes(r1, r2, u, p, rhoe, 'FC_SLAU')
        FE = keep_energy_flux(r1, r2, u, p, rhoe, 'APEC', eps0, eps1)
        def div(f): return (f - np.roll(f, 1)) / dx
        d1=-div(F1); d2=-div(F2); dU=-div(FU); dE=-div(FE)
        return [d1, d2, dU, dE], p

    if scheme in ('FC_Roe', 'APEC_A_Roe'):
        F1, F2, FU, FE = roe_fluxes(r1, r2, u, p, rhoe, scheme, eps0, eps1)
        def div(f): return (f - np.roll(f, 1)) / dx
        d1=-div(F1); d2=-div(F2); dU=-div(FU); dE=-div(FE)
        return [d1, d2, dU, dE], p

    if scheme in ('FC_SLAU', 'APEC_A_SLAU'):
        F1, F2, FU, FE = slau_fluxes(r1, r2, u, p, rhoe, scheme, eps0, eps1)
        def div(f): return (f - np.roll(f, 1)) / dx
        d1=-div(F1); d2=-div(F2); dU=-div(FU); dE=-div(FE)
        return [d1, d2, dU, dE], p

    if scheme in ('FC_HLLC', 'APEC_A'):
        F1, F2, FU, FE = hllc_fluxes(r1, r2, u, p, rhoe, scheme, eps0, eps1)
        def div(f):
            return (f - np.roll(f, 1)) / dx
        d1 = -div(F1);  d2 = -div(F2);  dU = -div(FU);  dE = -div(FE)
        return [d1, d2, dU, dE], p

    if scheme == 'APEC_A_KEEP':
        # Eq. A.4 에너지 플럭스, 단 F_{ρY_i}|_{m+1/2}, F_{ρu}|_{m+1/2}는 KEEP 플럭스 사용
        rho  = r1 + r2
        rhoE = rhoe + 0.5 * rho * u**2
        r1p, r2p    = _roll(r1), _roll(r2)
        up,  pp     = _roll(u),  _roll(p)
        rhoep       = _roll(rhoe)
        rhop        = r1p + r2p
        rhoEp       = rhoep + 0.5 * rhop * up**2
        eps0p, eps1p = _roll(eps0), _roll(eps1)

        # 물리 플럭스 at 셀 중심 m 및 m+1
        F1_m  = r1  * u;           F1_p  = r1p * up
        F2_m  = r2  * u;           F2_p  = r2p * up
        FU_m  = rho * u**2  + p;   FU_p  = rhop * up**2 + pp
        FE_m  = (rhoE  + p)  * u
        FE_p  = (rhoEp + pp) * up

        # KEEP 계면 플럭스 (질량·운동량)
        F1_half, F2_half, FU_conv, p_h, u_h, rho_h = keep_fluxes_mass_mom(r1, r2, u, p)
        FU_half = FU_conv + p_h   # 압력항 포함 운동량 플럭스

        # h_i = (ε_i − u²/2) at m and m+1
        h0_m = eps0  - 0.5 * u**2;   h0_p = eps0p - 0.5 * up**2
        h1_m = eps1  - 0.5 * u**2;   h1_p = eps1p - 0.5 * up**2

        # Eq. A.4
        corr_m = h0_m*(F1_half - F1_m) + h1_m*(F2_half - F2_m) + u *(FU_half - FU_m)
        corr_p = h0_p*(F1_p    - F1_half) + h1_p*(F2_p  - F2_half) + up*(FU_p  - FU_half)
        FE = 0.5*(FE_m + FE_p) + 0.5*corr_m - 0.5*corr_p

        F1, F2, FU = F1_half, F2_half, FU_half

        def div(f):
            return (f - np.roll(f, 1)) / dx
        d1 = -div(F1);  d2 = -div(F2);  dU = -div(FU);  dE = -div(FE)
        return [d1, d2, dU, dE], p

    if scheme in ('KEEPPE_Q2', 'KEEPPE_RRA'):
        if scheme == 'KEEPPE_Q2':
            F1, F2, FU, FE = keeppe_q2_fluxes(r1, r2, u, p, rhoe)
        else:
            F1, F2, FU, FE = keeppe_rra_fluxes(r1, r2, u, p, rhoe)
        def div(f): return (f - np.roll(f, 1)) / dx
        d1=-div(F1); d2=-div(F2); dU=-div(FU); dE=-div(FE)
        return [d1, d2, dU, dE], p

    if scheme == 'Fujiwara':
        F1, F2, FU, FE = fujiwara_fluxes(r1, r2, u, p)
    elif scheme == 'PEqC':
        F1, F2, FU_conv, p_h, u_h, rho_h = keep_fluxes_mass_mom(r1, r2, u, p)
        FU = FU_conv + p_h

        # PEqC energy: dρE/dt = -(div F_KE) - (div F_pu) - Σ_i ε_i*(div F_i)
        # F_KE (Eq. 38): ρ_h * (u_m * u_{m+1})/2 * u_h
        up  = _roll(u);  pp = _roll(p)
        F_KE = rho_h * (u * up) / 2.0 * u_h   # rho_h already computed above
        F_pu = 0.5 * (p * up + pp * u)         # Eq. 39

        # species flux for PEqC correction (Eq. 56)
        r1p, r2p = _roll(r1), _roll(r2)
        F1_pe = 0.5 * (r1 + r1p) * u_h
        F2_pe = 0.5 * (r2 + r2p) * u_h

        def div(f):
            return (f - np.roll(f, 1)) / dx

        dE = -(div(F_KE) + div(F_pu) + eps0 * div(F1_pe) + eps1 * div(F2_pe))

        d1 = -div(F1);  d2 = -div(F2);  dU = -div(FU)
        return [d1, d2, dU, dE], p

    else:
        # FC, APEC
        F1, F2, FU_conv, p_h, u_h, rho_h = keep_fluxes_mass_mom(r1, r2, u, p)
        FU = FU_conv + p_h
        FE = keep_energy_flux(r1, r2, u, p, rhoe, scheme, eps0, eps1)

    def div(f):
        return (f - np.roll(f, 1)) / dx

    d1 = -div(F1);  d2 = -div(F2);  dU = -div(FU);  dE = -div(FE)
    return [d1, d2, dU, dE], p


# ─────────────────────────────────────────────────────────────
# SSP-RK3
# ─────────────────────────────────────────────────────────────

def _clip_pos(U):
    """Species densities must stay non-negative."""
    U[0] = np.maximum(U[0], 0.0)
    U[1] = np.maximum(U[1], 0.0)
    return U


def rkstep(U, scheme, dx, dt):
    k1, p1 = rhs(U, scheme, dx)
    U1 = _clip_pos([U[q] + dt * k1[q] for q in range(4)])

    k2, p2 = rhs(U1, scheme, dx)
    U2 = _clip_pos([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)])

    k3, p3 = rhs(U2, scheme, dx)
    Un = _clip_pos([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)])

    return Un, p3


# ─────────────────────────────────────────────────────────────
# Error metrics
# ─────────────────────────────────────────────────────────────

def pe_err_L2(p, p0=0.9):
    """L2 norm of pressure deviation (normalised)."""
    return float(np.sqrt(np.mean((p - p0)**2)) / p0)


def energy_err(rhoE, rhoE0):
    return float(abs(np.sum(rhoE) - np.sum(rhoE0)) / (abs(np.sum(rhoE0)) + 1e-60))


# ─────────────────────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────────────────────

def run(scheme, N=501, t_end=8.0, CFL=0.6, verbose=True):
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1.0 - dx/2, N)

    print(f"\n{'='*55}")
    print(f"  scheme={scheme}  N={N}  t_end={t_end}  CFL={CFL}")

    r1, r2, u, rhoE, p = initial_condition(x)
    rhoE0 = rhoE.copy()
    p0    = 0.9

    U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]

    u0      = 1.0
    t_hist  = [0.0]
    pe_hist = [pe_err_L2(p, p0)]
    en_hist = [0.0]
    ue_hist = [0.0]   # ||u - u0||_L2 / u0
    t = 0.0;  step = 0;  diverged = False

    while t < t_end - 1e-12:
        r1_, r2_ = U[0], U[1]
        u_  = U[2] / np.maximum(r1_+r2_, 1e-60)
        _, _, p_ = prim_from_cons(r1_, r2_, U[2], U[3])
        c2  = cpg_c2(r1_, r2_, p_)
        lam = float(np.max(np.abs(u_) + np.sqrt(np.maximum(c2, 0.0))))
        dt  = min(CFL * dx / (lam + 1e-10), t_end - t)

        try:
            U, p_ = rkstep(U, scheme, dx, dt)
        except Exception as e:
            print(f"  Exception at t={t:.4f}: {e}")
            diverged = True; break

        t += dt;  step += 1

        pe_ = pe_err_L2(p_, p0)
        en_ = energy_err(U[3], rhoE0)
        u_cur = U[2] / np.maximum(U[0]+U[1], 1e-60)
        ue_ = float(np.sqrt(np.mean((u_cur - u0)**2))) / u0

        t_hist.append(t)
        pe_hist.append(pe_)
        en_hist.append(en_)
        ue_hist.append(ue_)

        if not np.isfinite(pe_) or pe_ > 10.0:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.4f}")
            diverged = True; break

        if verbose and (step % 2000 == 0 or t >= t_end - 1e-11):
            print(f"  t={t:.3f}  step={step}  PE={pe_:.3e}  Enerr={en_:.3e}")

    status = "Completed" if not diverged else "DIVERGED"
    print(f"  --> {status} at t={t:.4f} ({step} steps)")

    r1f, r2f = U[0], U[1]
    uf, rhoef, pf = prim_from_cons(r1f, r2f, U[2], U[3])
    return (x, r1f, r2f, uf, pf,
            np.array(t_hist), np.array(pe_hist), np.array(en_hist),
            np.array(ue_hist), diverged)


# ─────────────────────────────────────────────────────────────
# PE conservation error distribution  (Eq. 63-64, Fig. 3)
# ─────────────────────────────────────────────────────────────

def pe_error_distribution(x, r1, r2, p):
    """
    Analytical PE error from Eq. 63 (APEC) and Eq. 64 (FC-NPE) at t=0.

    e_APEC   = Σ_i { (1/12) * d(ε_i)/dx * d²(ρY_i)/dx²
                    - (1/12) * d²(ε_i)/dx² * d(ρY_i)/dx } * Δx²

    e_FC     = Σ_i { (1/3)  * d(ε_i)/dx * d²(ρY_i)/dx²
                    + (1/6) * d²(ε_i)/dx² * d(ρY_i)/dx } * Δx²
    """
    dx = x[1] - x[0]
    eps0, eps1 = epsilon_cpg(r1, r2, p)

    def d1(q):   # 1st derivative (central)
        return (np.roll(q, -1) - np.roll(q, 1)) / (2*dx)

    def d2(q):   # 2nd derivative (central)
        return (np.roll(q, -1) - 2*q + np.roll(q, 1)) / dx**2

    # ε derivatives
    deps0_1 = d1(eps0);  deps1_1 = d1(eps1)
    deps0_2 = d2(eps0);  deps1_2 = d2(eps1)

    # ρY_i derivatives
    dr1_1 = d1(r1);  dr2_1 = d1(r2)
    dr1_2 = d2(r1);  dr2_2 = d2(r2)

    def sum_terms(c1, c2):
        return (c1 * deps0_1 * dr1_2 + c2 * deps0_2 * dr1_1
              + c1 * deps1_1 * dr2_2 + c2 * deps1_2 * dr2_1) * dx**2

    e_apec = sum_terms(+1.0/12.0, -1.0/12.0)
    e_fc   = sum_terms(+1.0/3.0,  +1.0/6.0)
    return e_apec, e_fc


# ─────────────────────────────────────────────────────────────
# PE residual f_PE (for grid convergence, Fig. 5)
# ─────────────────────────────────────────────────────────────

def compute_fpe_norm(N):
    """
    Compute ||f_PE||_L2 for APEC at t=0 on a grid of N cells.

    f_PE|_m = Σ_i ε_i|_m * (F_{ρY_i}|_{m+1/2} - F_{ρY_i}|_{m-1/2})
            - (F_{ρeu}|_{m+1/2} - F_{ρeu}|_{m-1/2})

    The F_{ρY_i} and F_{ρeu} are the APEC half-point fluxes.
    """
    dx = 1.0 / N
    x  = np.linspace(dx/2, 1.0 - dx/2, N)
    r1, r2, u, rhoE, p = initial_condition(x)
    rhoe = cpg_rhoe(r1, r2, p)
    eps0, eps1 = epsilon_cpg(r1, r2, p)

    # APEC mass fluxes
    r1p   = _roll(r1);  r2p = _roll(r2)
    up    = _roll(u)
    u_h   = 0.5 * (u + up)
    F1    = 0.5 * (r1 + r1p) * u_h
    F2    = 0.5 * (r2 + r2p) * u_h

    # APEC ρe flux (Eq. 40)
    rhoep = _roll(rhoe)
    eps0p = _roll(eps0);  eps1p = _roll(eps1)
    dr1   = r1p - r1;    dr2   = r2p - r2
    corr  = 0.5*(eps0p - eps0)*0.5*dr1 + 0.5*(eps1p - eps1)*0.5*dr2
    rhoe_h = 0.5*(rhoe + rhoep) - corr
    Frhoe  = rhoe_h * u_h

    def div(f):
        return (f - np.roll(f, 1)) / dx

    f_pe = (eps0 * div(F1) + eps1 * div(F2)) - div(Frhoe)
    return float(np.sqrt(np.mean(f_pe**2)))


# ─────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────

COLORS = {
    'FC'      : '#d62728',   # red
    'APEC'    : '#1f77b4',   # blue
    'Fujiwara': '#2ca02c',   # green
    'PEqC'    : '#ff7f0e',   # orange
    'FC_HLLC'    : '#9467bd',   # purple
    'APEC_A'     : '#8c564b',   # brown
    'APEC_A_KEEP': '#e377c2',   # pink
    'FC_Roe'      : '#17becf',   # cyan
    'APEC_A_Roe'  : '#bcbd22',   # yellow-green
    'FC_SLAU'     : '#7f7f7f',   # gray
    'APEC_A_SLAU' : '#e377c2',   # pink
    'APEC_HYB_HLLC': '#d62728',  # dark red
    'APEC_HYB_Roe' : '#ff7f0e',  # orange
    'APEC_HYB_SLAU': '#9467bd',  # violet
    'KEEPPE_Q2'    : '#1a9850',  # dark green
    'KEEPPE_RRA'   : '#d73027',  # dark red-orange
    'EC_CHN'       : '#4393c3',  # blue (EC, no dissipation)
    'ES_CHN_Roe'   : '#d6604d',  # red-orange
    'ES_CHN_APEC'  : '#4dac26',  # green
}
LABELS = {
    'FC'      : 'FC-NPE (KEEP)',
    'APEC'    : 'APEC (KEEP, Sec.2.8)',
    'Fujiwara': 'Fujiwara',
    'PEqC'    : 'PEqC',
    'FC_HLLC'     : 'FC-NPE (HLLC)',
    'APEC_A'      : 'APEC-A (HLLC)',
    'APEC_A_KEEP' : 'APEC-A (KEEP)',
    'FC_Roe'      : 'FC-NPE (Roe)',
    'APEC_A_Roe'  : 'APEC-A (Roe)',
    'FC_SLAU'     : 'FC-NPE (SLAU)',
    'APEC_A_SLAU' : 'APEC-A (SLAU)',
    'APEC_HYB_HLLC': 'HYB: HLLC mass/mom + APEC energy',
    'APEC_HYB_Roe' : 'HYB: Roe  mass/mom + APEC energy',
    'APEC_HYB_SLAU': 'HYB: SLAU mass/mom + APEC energy',
    'KEEPPE_Q2'    : 'KEEPPE-Q2  (8th order, arith.)',
    'KEEPPE_RRA'   : 'KEEPPE-R-RA (8th order, geom.)',
    'EC_CHN'       : 'EC_CHN (Chandrasekhar, no diss.)',
    'ES_CHN_Roe'   : 'ES_CHN_Roe (EC + Roe diss.)',
    'ES_CHN_APEC'  : 'ES_CHN_APEC (EC mass/mom + APEC E)',
}
LS = {
    'FC'      : '--',
    'APEC'    : '-',
    'Fujiwara': '-.',
    'PEqC'    : ':',
    'FC_HLLC'      : '--',
    'APEC_A'       : '-',
    'APEC_A_KEEP'  : ':',
    'FC_Roe'       : '--',
    'APEC_A_Roe'   : '-',
    'FC_SLAU'      : '--',
    'APEC_A_SLAU'  : '-',
    'APEC_HYB_HLLC': '-.',
    'APEC_HYB_Roe' : '-.',
    'APEC_HYB_SLAU': '-.',
    'KEEPPE_Q2'    : '-',
    'KEEPPE_RRA'   : '--',
    'EC_CHN'       : '-',
    'ES_CHN_Roe'   : '--',
    'ES_CHN_APEC'  : '-.',
}


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Fig. 1 : profiles at t=8.0 ─────────────────────────────

def plot_fig1(results, t_snap=8.0):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flat

    for sch, res in results.items():
        x, r1, r2, u, p = res[:5]
        c, l, ls = COLORS[sch], LABELS[sch], LS[sch]
        axes[0].plot(x, r1, color=c, label=l, ls=ls)
        axes[1].plot(x, r2, color=c, label=l, ls=ls)
        axes[2].plot(x, u,  color=c, label=l, ls=ls)
        axes[3].plot(x, p,  color=c, label=l, ls=ls)

    titles = [r'$\rho_1$ [kg/m³]', r'$\rho_2$ [kg/m³]',
              r'$u$ [m/s]',        r'$p$ [-]']
    for ax, t in zip(axes, titles):
        ax.set_title(t); ax.legend(fontsize=8); ax.set_xlabel('x')
    fig.suptitle(f'1D interface advection  t = {t_snap:.1f}', fontsize=13)
    plt.tight_layout()
    savefig(fig, 'fig1_profiles.png')


# ── Fig. 2 : energy conservation & PE error history ────────

def plot_fig2(results):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    for sch, res in results.items():
        t_h, pe_h, en_h = res[5], res[6], res[7]
        c, l, ls = COLORS[sch], LABELS[sch], LS[sch]
        a1.semilogy(t_h, np.maximum(pe_h, 1e-16), color=c, label=l, ls=ls)
        a2.semilogy(t_h, np.maximum(en_h, 1e-16), color=c, label=l, ls=ls)

    a1.set_xlabel('t');  a1.set_ylabel(r'$\|p - p_0\|_{L_2} / p_0$')
    a1.set_title('Pressure-equilibrium error');  a1.legend()
    a2.set_xlabel('t');  a2.set_ylabel(r'$|\sum \rho E - \sum \rho E_0| / |\sum \rho E_0|$')
    a2.set_title('Total energy conservation error');  a2.legend()
    plt.tight_layout()
    savefig(fig, 'fig2_error_history.png')


# ── Fig. 3 : PE error distribution at t=0 ──────────────────

def plot_fig3(x, r1, r2, p):
    e_apec, e_fc = pe_error_distribution(x, r1, r2, p)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, e_fc,   color=COLORS['FC'],   label='FC-NPE  (Eq. 64)', ls='--')
    ax.plot(x, e_apec, color=COLORS['APEC'], label='APEC    (Eq. 63)', ls='-')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('x');  ax.set_ylabel(r'$e_{PE}(x)$')
    ax.set_title('PE conservation error distribution at $t=0$')
    ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig3_pe_error_dist.png')


# ── Fig. 4 : PE error norm time history ─────────────────────

def plot_fig4(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sch in ('FC', 'APEC'):
        if sch not in results:
            continue
        t_h, pe_h = results[sch][5], results[sch][6]
        ax.semilogy(t_h, np.maximum(pe_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t');  ax.set_ylabel(r'$\|p - p_0\|_{L_2} / p_0$')
    ax.set_title('PE error norm time history (N=501)')
    ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig4_pe_norm_history.png')


# ── Fig. 5 : grid convergence of f_PE ───────────────────────

def plot_fig5():
    Ns  = [51, 101, 201, 501]
    dxs = [1.0/N for N in Ns]
    norms = [compute_fpe_norm(N) for N in Ns]
    print("\n  Grid convergence (APEC f_PE):")
    for N, dx, nm in zip(Ns, dxs, norms):
        print(f"    N={N:4d}  dx={dx:.5f}  ||f_PE||={nm:.4e}")

    # reference 2nd-order line
    ref = norms[0] * (np.array(dxs) / dxs[0])**2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dxs, norms, 'o-', color=COLORS['APEC'], label='APEC')
    ax.loglog(dxs, ref,   'k--', label=r'$O(\Delta x^2)$')
    ax.set_xlabel(r'$\Delta x$');  ax.set_ylabel(r'$\|f_{PE}\|_{L_2}$')
    ax.set_title(r'Grid convergence of $f_{PE}$ for APEC')
    ax.legend()
    plt.tight_layout()
    savefig(fig, 'fig5_grid_convergence.png')


# ─────────────────────────────────────────────────────────────
# Fig. 6 : APEC (Sec.2.8) vs APEC_A (App.A) 비교
# ─────────────────────────────────────────────────────────────

def plot_fig6_apec_vs_apeca(results):
    """
    APEC (Sec.2.8) vs APEC_A (HLLC / Roe / SLAU) vs PEqC 비교.
    2x2 layout: (a) PE error, (b) Energy error, (c) Pressure profile, (d) Velocity profile
    Exact solution: p = 0.9, u = 1.0 everywhere.
    """
    SCHEMES_CMP = ('FC', 'APEC', 'PEqC', 'Fujiwara',
                   'KEEPPE_Q2', 'KEEPPE_RRA')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── (a) PE 오차 시간 이력 ────────────────────────────────
    ax = axes[0, 0]
    for sch in SCHEMES_CMP:
        if sch not in results:
            continue
        t_h, pe_h = results[sch][5], results[sch][6]
        ax.semilogy(t_h, np.maximum(pe_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t');  ax.set_ylabel(r'$\|p - p_0\|_{L_2} / p_0$')
    ax.set_title('(a) PE error history');  ax.legend(fontsize=7)

    # ── (b) 속도 오차 시간 이력 ──────────────────────────────
    ax = axes[0, 1]
    for sch in SCHEMES_CMP:
        if sch not in results:
            continue
        t_h, ue_h = results[sch][5], results[sch][8]
        ax.semilogy(t_h, np.maximum(ue_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t');  ax.set_ylabel(r'$\|u - u_0\|_{L_2} / u_0$')
    ax.set_title('(b) Velocity error history');  ax.legend(fontsize=7)

    # ── (c) 에너지 보존 오차 ─────────────────────────────────
    ax = axes[0, 2]
    for sch in SCHEMES_CMP:
        if sch not in results:
            continue
        t_h, en_h = results[sch][5], results[sch][7]
        ax.semilogy(t_h, np.maximum(en_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t');  ax.set_ylabel('Energy conservation error')
    ax.set_title('(c) Energy conservation');  ax.legend(fontsize=7)

    # ── (d) t=8.0 압력 프로파일 (exact: p=0.9) ───────────────
    ax = axes[1, 0]
    ax.axhline(0.9, color='k', lw=2, ls='-', label='Exact (p=0.9)', zorder=5)
    for sch in SCHEMES_CMP:
        if sch not in results:
            continue
        x, _, _, _, p_prof = results[sch][:5]
        ax.plot(x, p_prof, color=COLORS[sch], label=LABELS[sch], ls=LS[sch], lw=1)
    ax.set_xlabel('x');  ax.set_ylabel('p')
    ax.set_title('(d) Pressure profile at t=8.0  [exact: p=0.9]')
    ax.legend(fontsize=7)

    # ── (e) t=8.0 속도 프로파일 (exact: u=1.0) ───────────────
    ax = axes[1, 1]
    ax.axhline(1.0, color='k', lw=2, ls='-', label='Exact (u=1.0)', zorder=5)
    for sch in SCHEMES_CMP:
        if sch not in results:
            continue
        x, _, _, u_prof, _ = results[sch][:5]
        ax.plot(x, u_prof, color=COLORS[sch], label=LABELS[sch], ls=LS[sch], lw=1)
    ax.set_xlabel('x');  ax.set_ylabel('u')
    ax.set_title('(e) Velocity profile at t=8.0  [exact: u=1.0]')
    ax.legend(fontsize=7)

    # ── (f) 비어있음 (여백) ───────────────────────────────────
    axes[1, 2].axis('off')

    fig.suptitle('KEEPPE-Q2 & KEEPPE-R-RA vs FC / APEC / PEqC / Fujiwara  [N=501, CFL=0.6, 8th order]',
                 fontsize=12)
    plt.tight_layout()
    savefig(fig, 'fig6_apec_vs_apeca.png')


# ─────────────────────────────────────────────────────────────
# Fig. 7 : Chandrasekhar EC / ES vs 기존 스킴 비교
# ─────────────────────────────────────────────────────────────

def plot_fig7_es_comparison(results):
    """
    EC_CHN / ES_CHN_Roe / ES_CHN_APEC 를
    기존 FC / APEC / FC_Roe / APEC_A_Roe 와 비교.
    (a) PE 오차 이력  (b) 에너지 보존  (c) 압력 프로파일  (d) 속도 프로파일
    """
    SCHEMES_CMP = [
        'FC',          # 기준 (최악)
        'APEC',        # 논문 방법
        'FC_Roe',      # Roe 기반 FC
        'APEC_A_Roe',  # Roe 기반 APEC_A
        'EC_CHN',      # Chandrasekhar EC (소산 없음)
        'ES_CHN_Roe',  # EC + Roe 소산
        'ES_CHN_APEC', # EC mass/mom + APEC 에너지 + Roe 소산
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) PE 오차 이력
    ax = axes[0, 0]
    for sch in SCHEMES_CMP:
        if sch not in results: continue
        t_h, pe_h = results[sch][5], results[sch][6]
        ax.semilogy(t_h, np.maximum(pe_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t'); ax.set_ylabel(r'$\|p - p_0\|_{L_2} / p_0$')
    ax.set_title('(a) PE error history'); ax.legend(fontsize=7)

    # (b) 에너지 보존
    ax = axes[0, 1]
    for sch in SCHEMES_CMP:
        if sch not in results: continue
        t_h, en_h = results[sch][5], results[sch][7]
        ax.semilogy(t_h, np.maximum(en_h, 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t'); ax.set_ylabel('Energy conservation error')
    ax.set_title('(b) Energy conservation'); ax.legend(fontsize=7)

    # (c) 압력 프로파일 (exact: p=0.9)
    ax = axes[1, 0]
    ax.axhline(0.9, color='k', lw=2, ls='-', label='Exact (p=0.9)', zorder=5)
    for sch in SCHEMES_CMP:
        if sch not in results: continue
        x, _, _, _, p_prof = results[sch][:5]
        ax.plot(x, p_prof, color=COLORS[sch], label=LABELS[sch],
                ls=LS[sch], lw=1)
    ax.set_xlabel('x'); ax.set_ylabel('p')
    ax.set_title('(c) Pressure profile at t=8.0  [exact: p=0.9]')
    ax.legend(fontsize=7)

    # (d) PE 초기 1스텝 분포 — 격자별 비교
    ax = axes[1, 1]
    # PE error at t=0+ (첫 번째 스텝 이후)
    for sch in SCHEMES_CMP:
        if sch not in results: continue
        t_h, pe_h = results[sch][5], results[sch][6]
        ax.semilogy(t_h[:min(500, len(t_h))],
                    np.maximum(pe_h[:min(500, len(pe_h))], 1e-16),
                    color=COLORS[sch], label=LABELS[sch], ls=LS[sch])
    ax.set_xlabel('t (초기 구간)'); ax.set_ylabel(r'$\|p - p_0\|_{L_2} / p_0$')
    ax.set_title('(d) PE error — 초기 구간 확대'); ax.legend(fontsize=7)

    fig.suptitle('Chandrasekhar EC/ES vs FC/APEC/Roe  [N=501, CFL=0.6, t_end=8.0]',
                 fontsize=12)
    plt.tight_layout()
    savefig(fig, 'fig7_es_comparison.png')

    # ── 초기 PE 테이블 출력 ───────────────────────────────────
    print("\n  PE error (1st step) summary:")
    print("  {:20s}  {:>12s}  {:>12s}".format("scheme", "PE@t~0", "PE@t=8"))
    for sch in SCHEMES_CMP:
        if sch not in results: continue
        pe_h = results[sch][6]
        pe_init = pe_h[1] if len(pe_h) > 1 else float('nan')
        pe_end  = pe_h[-1]
        print("  {:20s}  {:>12.3e}  {:>12.3e}".format(sch, pe_init, pe_end))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N     = 501
    T_END = 8.0
    CFL   = 0.6

    # ── 논문 원래 비교 (KEEP 기반) ───────────────────────────
    SCHEMES_PAPER = ['FC', 'APEC', 'Fujiwara', 'PEqC']
    results = {}
    for sch in SCHEMES_PAPER:
        res = run(sch, N=N, t_end=T_END, CFL=CFL, verbose=True)
        results[sch] = res

    # ── Appendix A 추가 비교 (HLLC / Roe / SLAU) ────────────
    for sch in ['FC_HLLC', 'APEC_A', 'APEC_A_KEEP',
                'FC_Roe', 'APEC_A_Roe', 'FC_SLAU', 'APEC_A_SLAU',
                'APEC_HYB_HLLC', 'APEC_HYB_Roe', 'APEC_HYB_SLAU']:
        res = run(sch, N=N, t_end=T_END, CFL=CFL, verbose=True)
        results[sch] = res

    # ── KEEPPE 고차 스킴 (8th order) ─────────────────────────
    for sch in ['KEEPPE_Q2', 'KEEPPE_RRA']:
        res = run(sch, N=N, t_end=T_END, CFL=CFL, verbose=True)
        results[sch] = res

    # ── Chandrasekhar EC / ES 스킴 ───────────────────────────
    for sch in ['EC_CHN', 'ES_CHN_Roe', 'ES_CHN_APEC']:
        res = run(sch, N=N, t_end=T_END, CFL=CFL, verbose=True)
        results[sch] = res

    # ── Figures 1-5 (논문 재현) ──────────────────────────────
    print("\nGenerating figures...")
    plot_fig1(results, t_snap=T_END)
    plot_fig2(results)

    dx0 = 1.0 / N
    x0  = np.linspace(dx0/2, 1.0 - dx0/2, N)
    r1_0, r2_0, _, _, p0 = initial_condition(x0)
    plot_fig3(x0, r1_0, r2_0, p0)
    plot_fig4(results)
    plot_fig5()

    # ── Figure 6 : APEC vs APEC_A 비교 ──────────────────────
    plot_fig6_apec_vs_apeca(results)

    # ── Figure 7 : EC / ES_CHN 비교 (핵심) ──────────────────
    plot_fig7_es_comparison(results)

    print("\nDone. All figures saved to ./output/")
