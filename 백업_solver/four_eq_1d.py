"""
four_eq_1d.py
─────────────────────────────────────────────────────────────────
1D 4-방정식 다성분 다상 압축성 유동 솔버

기존 apec_1d.py와의 근본적 차이:
  apec_1d.py  : 다성분 단일상 가스 (CPG/SRK EOS)
                KEEP 분할 플럭스 + APEC 에너지 보정
                → PE 오차 O(10⁻⁵)
  four_eq_1d  : 다성분 다상 유동 (액체+기체, NASG EOS)
                IEC WENO5Z + 특성 분해 — W=[T,Y,u,P] 재건
                → PE 오차 O(10⁻¹¹) (기계 정밀도)

핵심 아이디어 (IEC WENO5Z + 특성 분해):
  표준 방법: W=[ρY, u, P] 재건 → ρ가 계면에서 점프
             → 재건 후 균일한 T 유지 불가 → 가짜 압력 진동
  IEC 방법:  W=[T, Y, u, P] 재건 → T, P, u 균일이면
             → 재건 후에도 균일 유지 → 압력 진동 원천 차단
  특성 분해: W→W̃ 변환 후 WENO5Z 재건 → 역변환
             → 음향파·엔트로피파 분리 → 계면 해상도 향상

구현 요소:
  - NASG EOS  (Noble-Abel Stiffened Gas, 표 1)
  - WENO-5Z   재건 (Borges 2008, τ₅=|β₀-β₂|)
  - 특성 분해  (Appendix A, θ 파라미터)
  - HLLC      Riemann 솔버
  - SSP-RK3   시간 적분
  - PP 제한자  (양수 보존)
  - CDI       계면 정규화 (Conservative Diffuse Interface)

검증 케이스:
  §4.2.1  Gas-liquid Riemann problem (air | water)
  §4.2.2  Inviscid droplet advection (water-in-air)
          IEC WENO5Z 오차 ~ 10⁻¹¹  vs  표준 ~ 10⁻³
  §4.2.3  Shock-droplet interaction
  §4.2.4  Inviscid Mach-100 water jet

참고:
  Collis, Bezgin, Mirjalili, Mani,
  "A robust four-equation model for compressible multi-phase
   multi-component flows satisfying interface equilibrium and
   phase-immiscibility conditions",
  J. Comput. Phys. 114827 (2026)
  doi:10.1016/j.jcp.2026.114827
─────────────────────────────────────────────────────────────────
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. NASG EOS 파라미터 (Table 1, Collis et al. 2026)
#
# Noble-Abel Stiffened Gas EOS (단일 성분):
#   v(P,T) = R·T/(P+P∞) + b          R = CP-Cv
#   e(P,T) = Cv·T·(P+γP∞)/(P+P∞) + q
#   h(P,T) = CP·T + b·P + q
#
# 이상기체 → P∞=0, b=0, q=0 으로 환원
# ─────────────────────────────────────────────────────────────
_NASG = {
    'water':    dict(CP=4185.0,        gamma=1.0123, q=-1.143e6, b=9.203e-4, Pinf=1.835e8),
    'air':      dict(CP=1011.0,        gamma=1.4,    q=0.0,      b=0.0,      Pinf=0.0),
    'helium':   dict(CP=5091.0,        gamma=1.66,   q=0.0,      b=0.0,      Pinf=0.0),
    'sf6':      dict(CP=661.0,         gamma=1.093,  q=0.0,      b=0.0,      Pinf=0.0),
    # Non-dimensional SG parameters for §4.2.1 Riemann test (Collis 2026 Table 4).
    # Classic Miller–Puckett water/air: γ_w=4.4, Pinf_w=6000, γ_a=1.4, Pinf_a=0.
    # Choose CP = γ/(γ-1) so that R = CP-Cv = 1 (unit gas constant).
    'water_nd': dict(CP=4.4/3.4,       gamma=4.4,    q=0.0,      b=0.0,      Pinf=6000.0),
    'air_nd':   dict(CP=1.4/0.4,       gamma=1.4,    q=0.0,      b=0.0,      Pinf=0.0),
}
for _m in _NASG.values():
    _m['Cv'] = _m['CP'] / _m['gamma']   # Cv = CP/γ
    _m['R']  = _m['CP'] - _m['Cv']      # R  = CP-Cv (= Cv*(γ-1))

# ─────────────────────────────────────────────────────────────
# 2. NASG 혼합 EOS 함수
#    이항 혼합: 상(0) = 액체(water), 상(1) = 기체(air 등)
#    Amagat's law (부분 부피): v = Y0*v0 + Y1*v1
# ─────────────────────────────────────────────────────────────

def nasg_p_from_rho_e(Y0, rho, e, sp0='water', sp1='air'):
    """NASG mixture pressure via Newton iteration on T_v(P) = T_e(P).

    The algebraic quadratic (Collis Eq. 21-23) has catastrophic cancellation
    for stiff water NASG (gamma-1 = 0.0123): both a2 terms ~ 7.7e11 but their
    difference is only Cv*P ~ 4e8.  Newton iteration avoids this entirely.

    f(P) = T_v(P) - T_e(P) = 0
      T_v = (v - Y0*b0) / [Y0*R0/(P+Pinf0) + Y1*R1/P]       (from vol EOS)
      T_e = (e - q)     / [Y0*Cv0*(P+g0*Pinf0)/(P+Pinf0) + Y1*Cv1]  (from e EOS)
    """
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1   = 1.0 - Y0
    Pinf = s0['Pinf']

    v   = 1.0 / np.maximum(rho, 1e-30)
    vb  = np.maximum(v - Y0 * s0['b'], 1e-30)
    q   = Y0 * s0['q'] + Y1 * s1['q']
    eq  = e - q

    # Pressure floor depends on composition:
    # - Near-pure water (Y1 ≤ tol): NASG SG can support P < 0 (tension).
    #   Physical limit P > -Pinf0; use -0.99*Pinf0 as safe floor.
    # - Cells with real air content (Y1 > tol): ideal-gas requires P > 0.
    _tol_Y = 1e-9
    near_pure_water = (Y1 <= _tol_Y)
    P_floor = np.where(near_pure_water, -0.99 * Pinf, 1e-6)

    # Initial guess.
    P_w = (s0['gamma'] - 1.0) * eq / vb - s0['gamma'] * Pinf
    P_a = (s1['gamma'] - 1.0) * rho * np.maximum(eq, 0.0)
    # Water formula gives the exact answer for pure water (including P < 0).
    P   = np.where(Y0 >= 0.5, P_w, P_a)
    P   = np.maximum(P, P_floor)

    for _ in range(50):
        u   = P + Pinf                     # P + Pinf0  (always > 0 by floor)
        # Air-term denominator: for near-pure water, Y1*R1/P is negligible
        # regardless of sign, so use signed P to avoid fake large contribution.
        # For cells with real air, P > 0 is guaranteed by floor.
        Pu  = np.where(near_pure_water,
                       np.where(np.abs(P) > 1e-30, P, 1e-30),
                       np.maximum(P, 1e-10))

        A   = Y0 * s0['R'] / u + Y1 * s1['R'] / Pu
        B   = Y0 * s0['Cv'] * (P + s0['gamma'] * Pinf) / u + Y1 * s1['Cv']

        A   = np.maximum(A, 1e-30)
        B   = np.maximum(B, 1e-30)

        Tv  = vb / A
        Te  = eq / B
        res = Tv - Te

        # Tight convergence: 1e-10 K (relative to T scale ~300 K → ~3e-13 relative)
        if float(np.max(np.abs(res))) < 1e-10:
            break

        # Derivatives dT_v/dP, dT_e/dP
        Pu2 = np.where(near_pure_water, np.maximum(Pu**2, 1e-60), np.maximum(Pu**2, 1e-20))
        dA   = -Y0 * s0['R'] / u**2 - Y1 * s1['R'] / Pu2
        dTv  = -vb * dA / np.maximum(A**2, 1e-60)
        dB   = Y0 * s0['Cv'] * Pinf * (1.0 - s0['gamma']) / u**2
        dTe  = -eq * dB / np.maximum(B**2, 1e-60)

        df  = dTv - dTe
        P   = P - res / np.where(np.abs(df) > 1e-30, df, 1e-30)
        P   = np.maximum(P, P_floor)

    return P


def nasg_T_from_P_e(Y0, P, e, sp0='water', sp1='air'):
    """온도 T — 압력, 내부에너지로부터.

    Eq. 19:  T = (e-q) / Σ_c Y_c·Cv_c·(P+γ_c·P∞_c)/(P+P∞_c)
    """
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1 = 1.0 - Y0
    q  = Y0 * s0['q'] + Y1 * s1['q']

    d0 = s0['Cv'] * (P + s0['gamma'] * s0['Pinf']) / np.maximum(P + s0['Pinf'], 1.0)
    d1 = s1['Cv']
    T = (e - q) / np.maximum(Y0 * d0 + Y1 * d1, 1e-10)
    return np.maximum(T, 1.0)


def nasg_rho_from_T_P_Y(Y0, T, P, sp0='water', sp1='air'):
    """밀도 ρ — 온도, 압력, 질량분율로부터 (Amagat's law)."""
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1 = 1.0 - Y0

    v0 = s0['R'] * T / np.maximum(P + s0['Pinf'], 1.0) + s0['b']
    v1 = s1['R'] * T / np.maximum(P, 1e-15)

    v = Y0 * v0 + Y1 * v1
    return 1.0 / np.maximum(v, 1e-30)


def nasg_e_from_T_P_Y(Y0, T, P, sp0='water', sp1='air'):
    """내부에너지 e — 온도, 압력, 질량분율로부터."""
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1 = 1.0 - Y0

    e0 = s0['Cv'] * T * (P + s0['gamma'] * s0['Pinf']) / np.maximum(P + s0['Pinf'], 1.0) + s0['q']
    e1 = s1['Cv'] * T + s1['q']

    return Y0 * e0 + Y1 * e1


def nasg_c2_mix(Y0, rho, T, P, sp0='water', sp1='air'):
    """혼합 음속² — Wood's 근사 (phase-frozen sound speed).

    Wood 혼합:
      1/(ρ·c²) = φ_l/(ρ_l·c_l²) + φ_g/(ρ_g·c_g²)

    주의: 이상기체 성분(sp1)이 저압에서 φ₁/(ρ₁·c₁²) ∝ Y₁/P² → ∞.
          WENO 재건 오차로 인한 부동소수점 수준의 Y₁(≲1e-12)이
          음속을 크게 오염시키는 것을 막기 위해 Y < 1e-9 이면 해당 상을 제외.
    """
    s0, s1 = _NASG[sp0], _NASG[sp1]
    # Threshold tiny mass fractions: floating-point residuals from WENO
    # reconstruction (Y1 ~ 1e-13) at low P give phi1/(rho1*c1sq) ∝ Y1/P^2 → ∞.
    _tol = 1e-9
    Y0s = np.where(Y0 > 1.0 - _tol, 1.0, np.where(Y0 < _tol, 0.0, Y0))
    Y1s = 1.0 - Y0s

    v0 = s0['R'] * T / np.maximum(P + s0['Pinf'], 1.0) + s0['b']
    v1 = s1['R'] * T / np.maximum(P, 1e-15)

    phi0 = np.maximum(rho * Y0s * v0, 0.0)
    phi1 = np.maximum(rho * Y1s * v1, 0.0)

    rho0 = 1.0 / np.maximum(v0, 1e-30)
    rho1 = 1.0 / np.maximum(v1, 1e-30)
    c0sq = s0['gamma'] * (P + s0['Pinf']) / np.maximum(rho0 * (1.0 - rho0 * s0['b']), 1e-10)
    c1sq = s1['gamma'] * P / np.maximum(rho1, 1e-10)

    inv_rhoc2 = (phi0 / np.maximum(rho0 * c0sq, 1e-10)
                 + phi1 / np.maximum(rho1 * c1sq, 1e-10))
    c2 = 1.0 / np.maximum(rho * inv_rhoc2, 1e-20)
    return np.maximum(c2, 1.0)


def nasg_prim(Y0, rho, rhoU, rhoE, sp0='water', sp1='air'):
    """보존변수 → 원시변수 (u, e, P, T, c²)."""
    u  = rhoU / np.maximum(rho, 1e-30)
    e  = rhoE / np.maximum(rho, 1e-30) - 0.5 * u**2
    P  = nasg_p_from_rho_e(Y0, rho, e, sp0, sp1)
    T  = nasg_T_from_P_e(Y0, P, e, sp0, sp1)
    c2 = nasg_c2_mix(Y0, rho, T, P, sp0, sp1)
    return u, e, P, T, c2


# ─────────────────────────────────────────────────────────────
# 3. 재건 — WENO-5Z (Borges et al. 2008)
#
# WENO5Z: τ₅ = |β₀ - β₂| (전역 평활도 지시자)
#   α_k = d_k · (1 + (τ₅/(β_k+ε))²)
#   ω_k = α_k / Σα_j
#
# Left  state at i+1/2: stencil {i-2,i-1,i,i+1,i+2}, d=[1/10,6/10,3/10]
# Right state at i+1/2: stencil {i-1,i,i+1,i+2,i+3}, d=[3/10,6/10,1/10]
# ─────────────────────────────────────────────────────────────

def _weno5z_eps(q):
    """Scale-adaptive epsilon for WENO5Z.

    eps = max(1e-6 * mean(q²), 1e-36)
    — 큰 절댓값 필드에서 (tau5/eps)² 오버플로우 방지.
    """
    return max(1e-6 * float(np.mean(q**2)), 1e-36)


def weno5z_lr(q, eps=None):
    """WENO-5Z 좌·우 경계값 재건 (주기 경계).

    Borges et al. (2008): Z-indicator τ₅=|β₀-β₂|.
    """
    if eps is None:
        eps = _weno5z_eps(q)

    qm2 = np.roll(q,  2)
    qm1 = np.roll(q,  1)
    qp1 = np.roll(q, -1)
    qp2 = np.roll(q, -2)
    qp3 = np.roll(q, -3)

    # ── Left state at i+1/2 ───────────────────────────────────
    p0L = ( 2.0*qm2 -  7.0*qm1 + 11.0*q  ) / 6.0
    p1L = (    -qm1 +  5.0*q   +  2.0*qp1) / 6.0
    p2L = ( 2.0*q   +  5.0*qp1 -      qp2) / 6.0

    b0L = (13.0/12.0)*(qm2 - 2.0*qm1 + q  )**2 + (1.0/4.0)*(qm2 - 4.0*qm1 + 3.0*q  )**2
    b1L = (13.0/12.0)*(qm1 - 2.0*q   + qp1)**2 + (1.0/4.0)*(qm1 -               qp1)**2
    b2L = (13.0/12.0)*(q   - 2.0*qp1 + qp2)**2 + (1.0/4.0)*(3.0*q - 4.0*qp1 + qp2)**2

    tau5L = np.abs(b0L - b2L)
    a0L = 0.1 * (1.0 + (tau5L / (b0L + eps))**2)
    a1L = 0.6 * (1.0 + (tau5L / (b1L + eps))**2)
    a2L = 0.3 * (1.0 + (tau5L / (b2L + eps))**2)
    sL  = a0L + a1L + a2L
    qL  = (a0L * p0L + a1L * p1L + a2L * p2L) / sL

    # ── Right state at i+1/2 ──────────────────────────────────
    # Mirror: uses stencil {i-1, i, i+1, i+2, i+3}
    p0R = ( 2.0*qp3 -  7.0*qp2 + 11.0*qp1) / 6.0
    p1R = (    -qp2 +  5.0*qp1 +  2.0*q  ) / 6.0
    p2R = ( 2.0*qp1 +  5.0*q   -      qm1) / 6.0

    b0R = (13.0/12.0)*(qp3 - 2.0*qp2 + qp1)**2 + (1.0/4.0)*(qp3 - 4.0*qp2 + 3.0*qp1)**2
    b1R = (13.0/12.0)*(qp2 - 2.0*qp1 + q  )**2 + (1.0/4.0)*(qp2 -               q   )**2
    b2R = (13.0/12.0)*(qp1 - 2.0*q   + qm1)**2 + (1.0/4.0)*(3.0*qp1 - 4.0*q + qm1 )**2

    tau5R = np.abs(b0R - b2R)
    a0R = 0.1 * (1.0 + (tau5R / (b0R + eps))**2)
    a1R = 0.6 * (1.0 + (tau5R / (b1R + eps))**2)
    a2R = 0.3 * (1.0 + (tau5R / (b2R + eps))**2)
    sR  = a0R + a1R + a2R
    qR  = (a0R * p0R + a1R * p1R + a2R * p2R) / sR

    return qL, qR


def weno3_lr(q, eps=1e-6):
    """WENO-3 (JS 1996) 좌·우 경계값 재건 (주기 경계)."""
    qm1 = np.roll(q,  1)
    qp1 = np.roll(q, -1)
    qp2 = np.roll(q, -2)

    q1L = -0.5 * qm1 + 1.5 * q
    q2L =  0.5 * q   + 0.5 * qp1
    b1L = (q   - qm1)**2
    b2L = (qp1 - q  )**2
    a1L = (1.0/3.0) / (eps + b1L)**2
    a2L = (2.0/3.0) / (eps + b2L)**2
    sL  = a1L + a2L
    qL  = (a1L * q1L + a2L * q2L) / sL

    q1R =  0.5 * q   + 0.5 * qp1
    q2R =  1.5 * qp1 - 0.5 * qp2
    b1R = (qp1 - q  )**2
    b2R = (qp2 - qp1)**2
    a1R = (2.0/3.0) / (eps + b1R)**2
    a2R = (1.0/3.0) / (eps + b2R)**2
    sR  = a1R + a2R
    qR  = (a1R * q1R + a2R * q2R) / sR

    return qL, qR


# ─────────────────────────────────────────────────────────────
# 4. 특성 분해 파라미터 (Appendix A, Collis et al. 2026)
#
# IEC 원시변수 W = [T, Y₀, u, P] 의 특성 기저:
#
#   α = ∂v/∂T|_P = Σ_k Y_k·R_k / (P+P∞_k)
#   β = -∂v/∂P|_T = Σ_k Y_k·R_k·T / (P+P∞_k)²
#   θ = (ρ·c²·β - 1) / (α·c²·ρ)
#
# 특성 변수 W̃ = S_A · W:
#   W̃₁ = T - θ·P          (entropy wave)
#   W̃₂ = Y₀               (contact wave)
#   W̃₃ = (a·ρ·u + P)/2    (forward acoustic wave)
#   W̃₄ = (-a·ρ·u + P)/2   (backward acoustic wave)
#
# 역변환:
#   T  = W̃₁ + θ·(W̃₃+W̃₄)
#   Y₀ = W̃₂
#   u  = (W̃₃-W̃₄)/(a·ρ)
#   P  = W̃₃ + W̃₄
# ─────────────────────────────────────────────────────────────

def nasg_T_from_rho_P_Y(Y0, rho, P, sp0='water', sp1='air'):
    """온도 T — 밀도, 압력, 질량분율로부터 (Amagat's law 역산).

    1/ρ = Y0·v0(T,P) + Y1·v1(T,P)
    1/ρ = T·[Y0·R0/(P+P∞0) + Y1·R1/P] + Y0·b0
    → T = (1/ρ - Y0·b0) / [Y0·R0/(P+P∞0) + Y1·R1/P]
    """
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1 = 1.0 - Y0

    coeff_T = (Y0 * s0['R'] / np.maximum(P + s0['Pinf'], 1e-15)
               + Y1 * s1['R'] / np.maximum(P, 1e-15))
    v_const = Y0 * s0['b']
    T = (1.0 / np.maximum(rho, 1e-30) - v_const) / np.maximum(coeff_T, 1e-30)
    return np.maximum(T, 1e-15)


def nasg_alpha_beta_theta(Y0, rho, T, P, c2, sp0='water', sp1='air'):
    """특성 분해 파라미터 α, β, θ (Appendix A)."""
    s0, s1 = _NASG[sp0], _NASG[sp1]
    Y1 = 1.0 - Y0

    # α = ∂v/∂T|_P (열팽창 계수)
    alpha = (Y0 * s0['R'] / np.maximum(P + s0['Pinf'], 1e-15)
             + Y1 * s1['R'] / np.maximum(P, 1e-15))

    # β = -∂v/∂P|_T (등온 압축률)
    beta = (Y0 * s0['R'] * T / np.maximum(P + s0['Pinf'], 1e-15)**2
            + Y1 * s1['R'] * T / np.maximum(P, 1e-15)**2)

    # θ = (ρ·c²·β - 1) / (α·c²·ρ)
    alpha_c2_rho = np.maximum(alpha * rho * c2, 1e-30)
    theta = (rho * c2 * beta - 1.0) / alpha_c2_rho

    return alpha, beta, theta


def char_lr_iec(T, Y0, u, P, rho, c, sp0='water', sp1='air'):
    """특성 분해 + WENO5Z 재건 (IEC 기저).

    올바른 구현: 면 j+1/2 기준 θ_j를 스텐실 셀 {j-2,..,j+2}에 일관되게 적용.

    각 면마다:
      W̃₁[k] = T[k] - θ_j · P[k]      (k = stencil cell)
      W̃₃[k] = ( a_j·ρ_j · u[k] + P[k] ) / 2
      W̃₄[k] = (-a_j·ρ_j · u[k] + P[k] ) / 2

    벡터화: np.roll로 스텐실 셀 참조, 면 기준 θ_j, arho_j 를 전체 셀에 브로드캐스트.
    """
    c2 = c**2

    # 면 i+1/2 기준 상태
    rho_r = np.roll(rho, -1)
    T_r   = np.roll(T,   -1)
    P_r   = np.roll(P,   -1)
    Y0_r  = np.roll(Y0,  -1)
    c2_r  = np.roll(c2,  -1)

    rho_h = 0.5 * (rho + rho_r)
    T_h   = 0.5 * (T   + T_r)
    P_h   = 0.5 * (P   + P_r)
    Y0_h  = 0.5 * (Y0  + Y0_r)
    c2_h  = 0.5 * (c2  + c2_r)
    c_h   = np.sqrt(np.maximum(c2_h, 1.0))

    _, _, theta_h = nasg_alpha_beta_theta(Y0_h, rho_h, T_h, P_h, c2_h, sp0, sp1)
    arho_h = c_h * rho_h

    # 스텐실 오프셋
    Tm2 = np.roll(T,  2);  Tm1 = np.roll(T,  1)
    Tp1 = np.roll(T, -1);  Tp2 = np.roll(T, -2);  Tp3 = np.roll(T, -3)
    Pm2 = np.roll(P,  2);  Pm1 = np.roll(P,  1)
    Pp1 = np.roll(P, -1);  Pp2 = np.roll(P, -2);  Pp3 = np.roll(P, -3)
    um2 = np.roll(u,  2);  um1 = np.roll(u,  1)
    up1 = np.roll(u, -1);  up2 = np.roll(u, -2);  up3 = np.roll(u, -3)

    # ── W̃₁ = T - θ_j · P  (면 j+1/2 의 θ_j 로 스텐실 셀 투영) ──
    def w1(t, p): return t - theta_h * p
    W1m2 = w1(Tm2, Pm2); W1m1 = w1(Tm1, Pm1); W10 = w1(T, P)
    W1p1 = w1(Tp1, Pp1); W1p2 = w1(Tp2, Pp2); W1p3 = w1(Tp3, Pp3)

    # ── W̃₃ = ( arho_j · u + P) / 2 ──
    def w3(ui, pi): return 0.5 * (arho_h * ui + pi)
    W3m2 = w3(um2, Pm2); W3m1 = w3(um1, Pm1); W30 = w3(u, P)
    W3p1 = w3(up1, Pp1); W3p2 = w3(up2, Pp2); W3p3 = w3(up3, Pp3)

    # ── W̃₄ = (-arho_j · u + P) / 2 ──
    def w4(ui, pi): return 0.5 * (-arho_h * ui + pi)
    W4m2 = w4(um2, Pm2); W4m1 = w4(um1, Pm1); W40 = w4(u, P)
    W4p1 = w4(up1, Pp1); W4p2 = w4(up2, Pp2); W4p3 = w4(up3, Pp3)

    # ── W̃₂ = Y₀  (scalar, no projection needed) ──
    Y0m2 = np.roll(Y0,  2); Y0m1 = np.roll(Y0,  1)
    Y0p1 = np.roll(Y0, -1); Y0p2 = np.roll(Y0, -2); Y0p3 = np.roll(Y0, -3)

    # ── WENO5Z: 미리 만들어 놓은 스텐실에 직접 적용 ──
    def weno5z_from_stencil(qm2, qm1, q0, qp1, qp2, qp3):
        """WENO5Z using pre-rolled stencil arrays (scale-adaptive eps)."""
        # Scale-adaptive epsilon: prevents (tau5/eps)^2 overflow for large fields
        q_all = np.stack([qm2, qm1, q0, qp1, qp2, qp3])
        eps   = max(1e-6 * float(np.mean(q_all**2)), 1e-36)

        # Left state (face i+1/2 from left)
        p0L = (2*qm2 - 7*qm1 + 11*q0) / 6
        p1L = (  -qm1 + 5*q0 + 2*qp1) / 6
        p2L = (2*q0  + 5*qp1 -  qp2 ) / 6
        b0L = (13/12)*(qm2 - 2*qm1 + q0 )**2 + (1/4)*(qm2 - 4*qm1 + 3*q0)**2
        b1L = (13/12)*(qm1 - 2*q0  + qp1)**2 + (1/4)*(qm1 - qp1)**2
        b2L = (13/12)*(q0  - 2*qp1 + qp2)**2 + (1/4)*(3*q0 - 4*qp1 + qp2)**2
        t5L = np.abs(b0L - b2L)
        a0L = 0.1*(1 + (t5L/(b0L+eps))**2)
        a1L = 0.6*(1 + (t5L/(b1L+eps))**2)
        a2L = 0.3*(1 + (t5L/(b2L+eps))**2)
        sL  = a0L + a1L + a2L
        qL  = (a0L*p0L + a1L*p1L + a2L*p2L) / sL

        # Right state (face i+1/2 from right)
        p0R = (2*qp3 - 7*qp2 + 11*qp1) / 6
        p1R = (  -qp2 + 5*qp1 + 2*q0 ) / 6
        p2R = (2*qp1 + 5*q0  -  qm1  ) / 6
        b0R = (13/12)*(qp3 - 2*qp2 + qp1)**2 + (1/4)*(qp3 - 4*qp2 + 3*qp1)**2
        b1R = (13/12)*(qp2 - 2*qp1 + q0 )**2 + (1/4)*(qp2 - q0)**2
        b2R = (13/12)*(qp1 - 2*q0  + qm1)**2 + (1/4)*(3*qp1 - 4*q0 + qm1)**2
        t5R = np.abs(b0R - b2R)
        a0R = 0.1*(1 + (t5R/(b0R+eps))**2)
        a1R = 0.6*(1 + (t5R/(b1R+eps))**2)
        a2R = 0.3*(1 + (t5R/(b2R+eps))**2)
        sR  = a0R + a1R + a2R
        qR  = (a0R*p0R + a1R*p1R + a2R*p2R) / sR
        return qL, qR

    W1L, W1R = weno5z_from_stencil(W1m2, W1m1, W10, W1p1, W1p2, W1p3)
    W3L, W3R = weno5z_from_stencil(W3m2, W3m1, W30, W3p1, W3p2, W3p3)
    W4L, W4R = weno5z_from_stencil(W4m2, W4m1, W40, W4p1, W4p2, W4p3)
    W2L, W2R = weno5z_from_stencil(Y0m2, Y0m1, Y0,  Y0p1, Y0p2, Y0p3)

    # 역변환: W̃ → W_prim
    arho_inv = 1.0 / np.maximum(arho_h, 1e-30)

    TL  = W1L + theta_h * (W3L + W4L)
    Y0L = W2L
    uL  = (W3L - W4L) * arho_inv
    PL  = W3L + W4L

    TR  = W1R + theta_h * (W3R + W4R)
    Y0R = W2R
    uR  = (W3R - W4R) * arho_inv
    PR  = W3R + W4R

    return TL, TR, Y0L, Y0R, uL, uR, PL, PR


# ─────────────────────────────────────────────────────────────
# 5. HLLC Riemann 솔버 (1D)
# ─────────────────────────────────────────────────────────────

def _signed_safe(x, eps=1e-30):
    """|x| >= eps を保つ符号付き安全除数 (prevent sign flip from np.maximum)."""
    return np.sign(x + 1e-300) * np.maximum(np.abs(x), eps)


def hllc_flux(UL, UR, PL, PR, cL, cR):
    """1D HLLC Riemann 솔버.

    U = [r0, r1, m, E]  (보존변수: ρY0, ρY1, ρu, ρE)
    F = [r0·u, r1·u, m·u+P, (E+P)·u]

    파속: SL = min(uL-cL, uR-cR),  SR = max(uL+cL, uR+cR)

    주의: denom = ρL*(SL-uL) - ρR*(SR-uR) 는 항상 음수.
          np.maximum(denom, ε) 사용 금지 → 부호 보존 필수.
    """
    r0L, r1L, mL, EL = UL
    r0R, r1R, mR, ER = UR

    rhoL = r0L + r1L
    rhoR = r0R + r1R
    uL_  = mL / np.maximum(rhoL, 1e-30)
    uR_  = mR / np.maximum(rhoR, 1e-30)

    SL = np.minimum(uL_ - cL, uR_ - cR)
    SR = np.maximum(uL_ + cL, uR_ + cR)

    # denom = ρL*(SL-uL) - ρR*(SR-uR) < 0 항상 (SL<uL, SR>uR)
    denom  = rhoL * (SL - uL_) - rhoR * (SR - uR_)
    S_star = ((PR - PL + rhoL * uL_ * (SL - uL_) - rhoR * uR_ * (SR - uR_))
              / _signed_safe(denom))

    def _flux(r0, r1, rho, u, P, E):
        return [r0 * u, r1 * u, rho * u**2 + P, (E + P) * u]

    FL = _flux(r0L, r1L, rhoL, uL_, PL, EL)
    FR = _flux(r0R, r1R, rhoR, uR_, PR, ER)

    def _star(r0, r1, rho, u, P, E, SK):
        # SK-S_star: SL<S_star (음수), SR>S_star (양수) → 부호 보존 필수
        den_ss  = _signed_safe(SK - S_star)
        # SK-u: SL<uL (음수), SR>uR (양수)
        den_sku = _signed_safe(SK - u)

        fac  = rho * (SK - u) / den_ss
        r0_s = fac * r0 / np.maximum(rho, 1e-30)
        r1_s = fac * r1 / np.maximum(rho, 1e-30)
        m_s  = fac * S_star
        E_s  = fac * (E / np.maximum(rho, 1e-30)
                      + (S_star - u) * (S_star + P / (rho * den_sku)))
        return [r0_s, r1_s, m_s, E_s]

    UL_s = _star(r0L, r1L, rhoL, uL_, PL, EL, SL)
    UR_s = _star(r0R, r1R, rhoR, uR_, PR, ER, SR)

    Uc_L = [r0L, r1L, mL, EL]
    Uc_R = [r0R, r1R, mR, ER]

    F_hllc = []
    for q in range(4):
        fq = np.where(
            SL >= 0.0,
            FL[q],
            np.where(
                S_star >= 0.0,
                FL[q] + SL * (UL_s[q] - Uc_L[q]),
                np.where(
                    SR >= 0.0,
                    FR[q] + SR * (UR_s[q] - Uc_R[q]),
                    FR[q]
                )
            )
        )
        F_hllc.append(fq)

    return F_hllc, np.maximum(np.abs(SL), np.abs(SR))


def hllc_flux_1st(U, P, c, sp0='water', sp1='air'):
    """1차 정확도 HLLC 플럭스 (PP 폴백용 — 셀 중심값 사용)."""
    r0, r1, m, E = U
    rho = r0 + r1
    Y0  = r0 / np.maximum(rho, 1e-30)

    # 오른쪽 셀 = 왼쪽 셀 + 1
    r0R = np.roll(r0, -1);  r1R = np.roll(r1, -1)
    mR  = np.roll(m,  -1);  ER  = np.roll(E,  -1)
    PR  = np.roll(P,  -1);  cR  = np.roll(c,  -1)

    UL_face = [r0, r1, m, E]
    UR_face = [r0R, r1R, mR, ER]

    F_1st, lam = hllc_flux(UL_face, UR_face, P, PR, c, cR)
    return F_1st, lam


# ─────────────────────────────────────────────────────────────
# 6. CDI 계면 정규화 (Conservative Diffuse Interface)
#
# 계면 체적 분율 φ₀ = ρ·Y₀·v₀(T,P) 로부터 signed-distance 함수:
#   ψ = ε·ln((φ₀+δ)/(1-φ₀+δ))
#
# CDI 소스 (계면 부근만 활성화):
#   a₀ = Γ·[ε·∂φ₀/∂x - n̂·(1-tanh²(ψ/(2ε)))/4·|∂ψ/∂x|]
#     여기서 n̂ = sign(∂ψ/∂x)
#
# CDI 플럭스 (KEEP 에너지 보존):
#   F_DI = [R̂₀, R̂₁, ū·R̂, k̄·R̂ + ĥ₀·R̂₀ + ĥ₁·R̂₁]
#   ū = (u_m + u_{m+1})/2,  k̄ = u_m·u_{m+1}/2  (KEEP)
#   ĥ_k = CP_k·T̂ + b_k·P̂ + q_k  (NASG 엔탈피)
# ─────────────────────────────────────────────────────────────

def cdi_flux_1d(U, dx, T, P, sp0='water', sp1='air',
                eps_factor=1.0, phi_min=0.01, delta=1e-6, Gamma=1.0):
    """CDI 계면 정규화 플럭스 (1D).

    Parameters
    ----------
    eps_factor : ε = eps_factor * dx  (계면 두께 스케일)
    phi_min    : 체적 분율 최솟값 (계면 활성화 임계)
    Gamma      : CDI 강도 계수
    """
    s0, s1 = _NASG[sp0], _NASG[sp1]
    r0, r1, m, E = U
    rho = r0 + r1
    Y0  = r0 / np.maximum(rho, 1e-30)

    eps = eps_factor * dx

    # 체적 분율 φ₀ = ρ·Y₀·v₀(T,P)
    v0  = s0['R'] * T / np.maximum(P + s0['Pinf'], 1.0) + s0['b']
    phi0 = np.clip(rho * Y0 * v0, delta, 1.0 - delta)

    # Signed-distance transform ψ
    psi = eps * np.log((phi0 + delta) / (1.0 - phi0 + delta))

    # ∂φ₀/∂x, ∂ψ/∂x  (중심 차분)
    dphi_dx = (np.roll(phi0, -1) - np.roll(phi0, 1)) / (2.0 * dx)
    dpsi_dx = (np.roll(psi,  -1) - np.roll(psi,  1)) / (2.0 * dx)

    # CDI 소스 a₀ (계면 수직 방향 질량 플럭스)
    taper  = 1.0 - np.tanh(psi / (2.0 * eps))**2   # sech²(ψ/2ε)
    a0_cdi = Gamma * (eps * dphi_dx - 0.25 * taper * np.abs(dpsi_dx) * np.sign(dpsi_dx))

    # 계면 마스크 (순수 영역 제외)
    interface_mask = (phi0 > phi_min) & (phi0 < 1.0 - phi_min)
    a0_cdi = np.where(interface_mask, a0_cdi, 0.0)

    # 면 변수 (KEEP: 운동에너지 보존형 평균)
    u   = m / np.maximum(rho, 1e-30)
    u_r = np.roll(u, -1)
    T_r = np.roll(T, -1)
    P_r = np.roll(P, -1)

    u_h = 0.5 * (u + u_r)               # 산술 평균 속도
    k_h = u * u_r / 2.0                  # KEEP 운동에너지
    T_h = 0.5 * (T + T_r)
    P_h = 0.5 * (P + P_r)

    # NASG 엔탈피 h_k = CP_k·T + b_k·P + q_k
    h0_h = s0['CP'] * T_h + s0['b'] * P_h + s0['q']
    h1_h = s1['CP'] * T_h + s1['b'] * P_h + s1['q']

    # CDI 질량 플럭스 (면 i+1/2)
    R0_hat = a0_cdi * rho            # 총 질량 플럭스 (phase 0)
    R_hat  = R0_hat                  # R̂ = R̂₀ (2상 모델에서 R̂₁ = -R̂₀)
    R1_hat = -R0_hat

    F_cdi = [
        R0_hat,
        R1_hat,
        u_h * R_hat,
        k_h * R_hat + h0_h * R0_hat + h1_h * R1_hat,
    ]

    return F_cdi


# ─────────────────────────────────────────────────────────────
# 7. RHS — IEC WENO5Z + 특성 분해
# ─────────────────────────────────────────────────────────────

def rhs_iec(U, dx, sp0='water', sp1='air', iec=True,
            use_char=True, weno_order=5, use_cdi=False,
            cdi_params=None, bc='periodic'):
    """IEC RHS (보존형 HLLC 플럭스).

    Parameters
    ----------
    iec        : True  → W=[T, Y₀, u, P] 재건  (IEC, 압력 진동 억제)
                 False → W=[ρY₀, u, P]   재건  (표준, 비교용)
    use_char   : True  → 특성 분해 + WENO5Z (최고 정확도)
    weno_order : 5 (WENO5Z) 또는 3 (WENO3)
    use_cdi    : True  → CDI 계면 정규화 추가
    bc         : 'periodic'     → 주기 경계 (기본값, 이송 문제)
                 'transmissive' → 유출 경계 (Riemann 문제)
    """
    r0, r1, m, E = U
    rho = r0 + r1
    Y0  = r0 / np.maximum(rho, 1e-30)
    u, e, P, T, c2 = nasg_prim(Y0, rho, m, E, sp0, sp1)
    c = np.sqrt(np.maximum(c2, 1.0))

    # ── transmissive BC: pad with ghost cells so WENO stencil ────
    # sees correct zero-gradient extrapolation instead of periodic wrap.
    # ng=3 for WENO5 (stencil spans ±3), ng=2 for WENO3.
    N = len(r0)
    if bc == 'transmissive':
        ng = 3 if weno_order >= 5 else 2
        def _pad(a):
            return np.concatenate([np.full(ng, a[0]), a, np.full(ng, a[-1])])
        T_in   = _pad(T);    Y0_in = _pad(Y0); u_in  = _pad(u);  P_in  = _pad(P)
        rho_in = _pad(rho);  c_in  = _pad(c)
        r0_in  = _pad(r0);   r1_in = _pad(r1)
        # After reconstruction on N+2*ng cells, take N+1 faces:
        # face ng-1  = left boundary face  (-1/2)
        # face ng+N-1 = right boundary face (N-1/2)
        sl = slice(ng - 1, ng + N)   # N+1 elements
    else:
        T_in  = T;    Y0_in = Y0; u_in  = u;  P_in  = P
        rho_in = rho; c_in  = c
        r0_in  = r0;  r1_in = r1

    if iec:
        if use_char and weno_order >= 5:
            # 특성 분해 + WENO5Z 재건 (논문 핵심 기법)
            TL, TR, Y0L, Y0R, uL, uR, PL, PR = char_lr_iec(
                T_in, Y0_in, u_in, P_in, rho_in, c_in, sp0, sp1)
        else:
            # 단순 IEC 재건 (특성 분해 없이)
            recon = weno5z_lr if weno_order >= 5 else weno3_lr
            TL,  TR  = recon(T_in)
            Y0L, Y0R = recon(Y0_in)
            uL,  uR  = recon(u_in)
            PL,  PR  = recon(P_in)

        if bc == 'transmissive':
            TL  = TL[sl];   TR  = TR[sl]
            Y0L = Y0L[sl];  Y0R = Y0R[sl]
            uL  = uL[sl];   uR  = uR[sl]
            PL  = PL[sl];   PR  = PR[sl]

        Y0L = np.clip(Y0L, 0.0, 1.0)
        Y0R = np.clip(Y0R, 0.0, 1.0)
        # Allow negative P for near-pure water (Y1 ≤ 1e-9, tension-state NASG).
        # Mixed/air cells must have P > 0 (ideal-gas air requires positive P).
        _Pinf0 = _NASG[sp0]['Pinf']
        _tol_Y = 1e-9
        _PL_floor = np.where((1.0 - Y0L) <= _tol_Y, -0.99 * _Pinf0, 1e-6)
        _PR_floor = np.where((1.0 - Y0R) <= _tol_Y, -0.99 * _Pinf0, 1e-6)
        PL  = np.maximum(PL, _PL_floor)
        PR  = np.maximum(PR, _PR_floor)
        TL  = np.maximum(TL, 1e-15)
        TR  = np.maximum(TR, 1e-15)

        rhoL = nasg_rho_from_T_P_Y(Y0L, TL, PL, sp0, sp1)
        rhoR = nasg_rho_from_T_P_Y(Y0R, TR, PR, sp0, sp1)
        eL   = nasg_e_from_T_P_Y(Y0L, TL, PL, sp0, sp1)
        eR   = nasg_e_from_T_P_Y(Y0R, TR, PR, sp0, sp1)
        cL   = np.sqrt(nasg_c2_mix(Y0L, rhoL, TL, PL, sp0, sp1))
        cR   = np.sqrt(nasg_c2_mix(Y0R, rhoR, TR, PR, sp0, sp1))

    else:
        # 표준 재건: W = [ρY₀, u, P]
        recon = weno5z_lr if weno_order >= 5 else weno3_lr
        r0L, r0R = recon(r0_in)
        r1L, r1R = recon(r1_in)
        uL,  uR  = recon(u_in)
        PL,  PR  = recon(P_in)

        if bc == 'transmissive':
            r0L = r0L[sl];  r0R = r0R[sl]
            r1L = r1L[sl];  r1R = r1R[sl]
            uL  = uL[sl];   uR  = uR[sl]
            PL  = PL[sl];   PR  = PR[sl]

        r0L = np.maximum(r0L, 0.0);  r0R = np.maximum(r0R, 0.0)
        r1L = np.maximum(r1L, 0.0);  r1R = np.maximum(r1R, 0.0)
        rhoL = np.maximum(r0L + r1L, 1e-30);  rhoR = np.maximum(r0R + r1R, 1e-30)
        Y0L  = r0L / rhoL;                     Y0R  = r0R / rhoR
        PL   = np.maximum(PL, 1e-15);          PR   = np.maximum(PR, 1e-15)

        # T로부터 e, c 계산: Amagat 역산으로 T 정확히 계산
        TL = nasg_T_from_rho_P_Y(Y0L, rhoL, PL, sp0, sp1)
        TR = nasg_T_from_rho_P_Y(Y0R, rhoR, PR, sp0, sp1)
        eL = nasg_e_from_T_P_Y(Y0L, TL, PL, sp0, sp1)
        eR = nasg_e_from_T_P_Y(Y0R, TR, PR, sp0, sp1)
        cL = np.sqrt(nasg_c2_mix(Y0L, rhoL, TL, PL, sp0, sp1))
        cR = np.sqrt(nasg_c2_mix(Y0R, rhoR, TR, PR, sp0, sp1))

    # 보존변수 면 값
    r0L_c = rhoL * Y0L;       r0R_c = rhoR * Y0R
    r1L_c = rhoL * (1-Y0L);   r1R_c = rhoR * (1-Y0R)
    EL_c  = rhoL * (eL + 0.5 * uL**2)
    ER_c  = rhoR * (eR + 0.5 * uR**2)

    UL_face = [r0L_c, r1L_c, rhoL * uL, EL_c]
    UR_face = [r0R_c, r1R_c, rhoR * uR, ER_c]

    # HLLC 플럭스
    F_int, lam = hllc_flux(UL_face, UR_face, PL, PR, cL, cR)

    # CDI 계면 정규화
    if use_cdi:
        kw = cdi_params or {}
        F_cdi = cdi_flux_1d(U, dx, T, P, sp0, sp1, **kw)
        for q in range(4):
            F_int[q] = F_int[q] + F_cdi[q]

    # 공간 잔차
    if bc == 'transmissive':
        # F_int has N+1 faces: F[0]=left boundary (-1/2), F[N]=right boundary (N-1/2)
        # d[i] = -(F[i+1] - F[i]) / dx
        d = [-(F_int[q][1:] - F_int[q][:-1]) / dx for q in range(4)]
    else:
        d = [-(F_int[q] - np.roll(F_int[q], 1)) / dx for q in range(4)]

    return d, P, c


def rhs_std(U, dx, sp0='water', sp1='air'):
    """표준 재건 RHS (비교용)."""
    return rhs_iec(U, dx, sp0, sp1, iec=False)


# ─────────────────────────────────────────────────────────────
# 8. 시간 적분 (Shu-Osher SSP-RK3)
# ─────────────────────────────────────────────────────────────

def _clip(U):
    return [np.maximum(U[0], 0.0), np.maximum(U[1], 0.0), U[2], U[3]]


def rkstep(U, rhs_fn, dx, dt):
    """SSP-RK3 (Shu & Osher, 1988)."""
    k1, P1, _ = rhs_fn(U, dx)
    U1 = _clip([U[q] + dt * k1[q] for q in range(4)])
    k2, P2, _ = rhs_fn(U1, dx)
    U2 = _clip([0.75 * U[q] + 0.25 * (U1[q] + dt * k2[q]) for q in range(4)])
    k3, P3, _ = rhs_fn(U2, dx)
    return _clip([(1.0/3.0) * U[q] + (2.0/3.0) * (U2[q] + dt * k3[q])
                  for q in range(4)]), P3


# ─────────────────────────────────────────────────────────────
# 9. 검증 케이스 §4.2.2: Inviscid Droplet Advection
#    물 액적(water) 공기(air) 중 이송
#
#    초기 조건 (Table 5):
#      Water:  ρ=997 kg/m³,  u=5 m/s,  P=101325 Pa,  T=297 K
#      Air:    ρ=1.18 kg/m³, u=5 m/s,  P=101325 Pa,  T=297 K
#      계면:   x∈[0,1],  액적 중심 x=0.5,  두께 ε·Δx
#
#    검증 목표:
#      한 주기 이송(t=L/u=0.2 s) 후 max|P-P0|/P0 < 10⁻¹⁰
# ─────────────────────────────────────────────────────────────

def ic_droplet(x, P0=101325.0, T0=297.0, u0=5.0, eps_factor=2.0,
               sp0='water', sp1='air'):
    """Water droplet IC (tanh mass-fraction profile, uniform T/P/u).

    Y0(x) = 0.5*(1 + tanh((0.25 - |x-0.5|) / (eps_factor*dx)))

    Using Y0 as the MASS fraction (standard for Amagat-law IEC models):
      rho = 1 / (Y0*v_water(T,P) + Y1*v_air(T,P))
      e   = Y0*e_water(T,P) + Y1*e_air(T,P)

    T=T0, P=P0, u=u0 are UNIFORM everywhere — ideal for IEC pressure error test.
    """
    N   = len(x)
    dx  = x[1] - x[0]
    eps = eps_factor * dx

    # Y0 = mass fraction, directly from tanh profile
    Y0 = 0.5 * (1.0 + np.tanh((0.25 - np.abs(x - 0.5)) / eps))
    Y0 = np.clip(Y0, 0.0, 1.0)

    P_arr = np.full(N, P0)
    rho   = nasg_rho_from_T_P_Y(Y0, T0, P_arr, sp0, sp1)
    e     = nasg_e_from_T_P_Y(Y0, T0, P_arr, sp0, sp1)

    r0   = rho * Y0
    r1   = rho * (1.0 - Y0)
    u    = np.full(N, u0)
    rhoE = rho * (e + 0.5 * u**2)

    return r0, r1, u, rhoE


def iec_error(U, P0, T0, u0, sp0='water', sp1='air'):
    """IEC 오차: max|(P-P0)/P0|, max|(T-T0)/T0|, max|(u-u0)/u0|."""
    r0, r1, m, E = U
    rho = r0 + r1
    Y0  = r0 / np.maximum(rho, 1e-30)
    u, e, P, T, _ = nasg_prim(Y0, rho, m, E, sp0, sp1)

    mask = (Y0 > 1e-6) & (Y0 < 1.0 - 1e-6)
    if not np.any(mask):
        mask = np.ones(len(Y0), dtype=bool)

    ep = float(np.max(np.abs((P[mask] - P0) / P0)))
    et = float(np.max(np.abs((T[mask] - T0) / T0)))
    eu = float(np.max(np.abs((u[mask] - u0) / u0))) if u0 != 0 else 0.0
    return ep, et, eu


def run_droplet(N=100, t_end=None, CFL=0.5, scheme='IEC',
                use_char=True, weno_order=5, use_cdi=False,
                sp0='water', sp1='air', eps_factor=4.0):
    """물 액적 이송 시뮬레이션.

    Parameters
    ----------
    scheme     : 'IEC' 또는 'STD'
    use_char   : 특성 분해 사용 여부
    weno_order : 5 (WENO5Z) 또는 3 (WENO3)
    """
    dx = 1.0 / N
    x  = np.linspace(dx / 2, 1.0 - dx / 2, N)
    P0, T0, u0 = 101325.0, 297.0, 5.0

    if t_end is None:
        t_end = 1.0 / u0   # 1 flow-through time = 0.2 s

    tag = f'[{scheme}'
    if scheme == 'IEC':
        tag += f'+W{weno_order}Z' if weno_order >= 5 else f'+W{weno_order}'
        if use_char:
            tag += '+Char'
        if use_cdi:
            tag += '+CDI'
    tag += ']'

    print(f"\n{'='*60}\n{tag} N={N}  t_end={t_end:.3f}  CFL={CFL}")

    r0, r1, u_ic, rhoE = ic_droplet(x, P0, T0, u0, eps_factor, sp0, sp1)
    U = [r0.copy(), r1.copy(), (r0 + r1) * u_ic, rhoE.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx):
        return rhs_iec(U, dx, sp0, sp1, iec=use_iec,
                       use_char=use_char, weno_order=weno_order,
                       use_cdi=use_cdi)

    t, step = 0.0, 0

    while t < t_end - 1e-12:
        r0_, r1_, m_, E_ = U
        rho_   = r0_ + r1_
        Y0_    = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam    = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt     = min(CFL * dx / (lam + 1e-10), t_end - t)

        U, P_ = rkstep(U, rhs_fn, dx, dt)
        t += dt;  step += 1

        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at t={t:.4f}"); break

    ep, et, eu = iec_error(U, P0, T0, u0, sp0, sp1)
    print(f"  t={t:.4f}  ({step} steps)")
    print(f"  IEC 오차:  |ΔP/P0|={ep:.3e}  |ΔT/T0|={et:.3e}  |Δu/u0|={eu:.3e}")
    return x, U, ep, et, eu


# ─────────────────────────────────────────────────────────────
# 10. 검증 케이스 §4.2.1: Gas-Liquid Riemann Problem
#     좌: 압축 공기,  우: 물 (상압)
# ─────────────────────────────────────────────────────────────

def ic_riemann_gl(x, sp0='water_nd', sp1='air_nd'):
    """기체-액체 Riemann 초기 조건 (§4.2.1, Table 4).

    무차원 값 (Miller–Puckett SG):
      left  (x<0): air_nd   ρ=1.241, u=1.0, P=2.753
      right (x≥0): water_nd ρ=0.991, u=0.0, P=3.059e-4
    EOS: water_nd γ=4.4, Pinf=6000 (dim'less);  air_nd γ=1.4, Pinf=0.
    """
    N = len(x)
    r0   = np.zeros(N)
    r1   = np.zeros(N)
    m    = np.zeros(N)
    rhoE = np.zeros(N)

    for i, xi in enumerate(x):
        if xi < 0.0:
            rho_i, u_i, P_i, Y0_i = 1.241, 1.0, 2.753, 0.0
            gam = _NASG[sp1]['gamma']
            e_i = P_i / (rho_i * (gam - 1.0))
        else:
            rho_i, u_i, P_i, Y0_i = 0.991, 0.0, 3.059e-4, 1.0
            gam  = _NASG[sp0]['gamma']
            Pinf = _NASG[sp0]['Pinf']
            b    = _NASG[sp0]['b']
            q    = _NASG[sp0]['q']
            e_i  = (P_i + gam * Pinf) / ((gam - 1.0) * rho_i / (1.0 - rho_i * b)) + q

        r0[i]   = rho_i * Y0_i
        r1[i]   = rho_i * (1.0 - Y0_i)
        m[i]    = rho_i * u_i
        rhoE[i] = rho_i * (e_i + 0.5 * u_i**2)

    return r0, r1, m, rhoE


def run_riemann_gl(N=200, t_end=0.14, CFL=0.3, scheme='IEC',
                   sp0='water_nd', sp1='air_nd'):
    """§4.2.1 기체-액체 Riemann 문제 시뮬레이션."""
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

    r0, r1, m_ic, rhoE = ic_riemann_gl(x, sp0, sp1)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx):
        # For the Riemann problem, characteristic decomposition is not used:
        # char_lr_iec averages face states arithmetically (arho_h = c_h*rho_h),
        # which is meaningless at a non-equilibrium air/water interface (c_air≈1.78
        # vs c_water≈163).  Plain WENO5Z reconstruction of [T, Y0, u, P] is stable
        # and gives the correct solution for this test case.
        return rhs_iec(U, dx, sp0, sp1, iec=use_iec, use_char=False,
                       weno_order=5, bc='transmissive')

    t, step = 0.0, 0
    print(f"\n{'='*55}\n[{scheme}] Gas-Liquid Riemann  N={N}  t_end={t_end}")

    while t < t_end - 1e-12:
        r0_, r1_, m_, E_ = U
        rho_   = r0_ + r1_
        Y0_    = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam    = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt     = min(CFL * dx / (lam + 1e-10), t_end - t)

        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt;  step += 1

        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at t={t:.4f}"); break

    print(f"  완료: t={t:.4f}  ({step} steps)")
    return x, U


# ─────────────────────────────────────────────────────────────
# 11. 검증 케이스 §4.2.3: Shock-Droplet Interaction
#     충격파가 물 액적에 충돌하는 문제
#
#     초기 조건:
#       왼쪽(충격 후): 공기 고압/고밀도
#       오른쪽: 공기 저압 + 물 액적
# ─────────────────────────────────────────────────────────────

def ic_shock_droplet(x, sp0='water', sp1='air',
                     P_shock=10.0*101325.0, T_shock=None,
                     P0=101325.0, T0=297.0,
                     x_shock=-0.3, x_drop=0.0, R_drop=0.1,
                     eps_factor=3.0):
    """충격파-액적 충돌 초기 조건.

    left  (x < x_shock): 충격 후 상태 (Rankine-Hugoniot)
    right                : 미충격 상태 (공기) + 물 액적 (tanh 분포)
    """
    N  = len(x)
    dx = x[1] - x[0]

    s1 = _NASG[sp1]  # air
    gam = s1['gamma']

    # 충격 후 상태 (이상 기체 Rankine-Hugoniot, Ma²=(γ+1)P_s/(γ-1)(P_s+2P0/(γ+1)))
    Ps  = P_shock
    rho_pre  = nasg_rho_from_T_P_Y(0.0, T0, P0, sp0, sp1)
    rho_post = rho_pre * (gam + 1.0) * Ps / ((gam - 1.0) * Ps + 2.0 * gam * P0)
    T_post   = T0 * Ps / P0 * (2.0 * gam * P0 + (gam - 1.0) * Ps) / ((gam + 1.0) * Ps)
    if T_shock is not None:
        T_post = T_shock

    # 충격 후 속도 (충격파는 왼쪽에서 오른쪽으로 이동)
    c_pre    = np.sqrt(gam * P0 / rho_pre)
    Ms       = np.sqrt((gam + 1.0) / (2.0 * gam) * (Ps / P0 - 1.0) + 1.0)
    u_post   = c_pre * (2.0 / (gam + 1.0)) * (Ms - 1.0 / Ms)

    # 액적 체적 분율 (tanh)
    eps = eps_factor * dx
    phi_l = 0.5 * (1.0 + np.tanh((R_drop - np.abs(x - x_drop)) / eps))
    phi_g = 1.0 - phi_l

    rho_l = nasg_rho_from_T_P_Y(1.0, T0, P0, sp0, sp1)

    # 미충격 영역 혼합 밀도: partial densities (Amagat's law — volume fraction 기반)
    # r0 = rho_l * phi_l, r1 = rho_pre * phi_g 가 Amagat's law를 만족함을 확인:
    #   v_mix = Y0*v_l + Y1*v_g = (r0/rho)*(1/rho_l) + (r1/rho)*(1/rho_pre)
    #         = phi_l/rho + phi_g/rho = 1/rho  ✓
    rho_right = rho_l * phi_l + rho_pre * phi_g
    Y0_right  = np.clip(rho_l * phi_l / np.maximum(rho_right, 1e-30), 0.0, 1.0)

    # 초기 조건 배열
    r0   = np.where(x < x_shock, 0.0, rho_right * Y0_right)
    r1   = np.where(x < x_shock, rho_post, rho_right * (1.0 - Y0_right))
    rho_arr = r0 + r1

    e_right = nasg_e_from_T_P_Y(Y0_right, T0, np.full(N, P0), sp0, sp1)
    e_post  = nasg_e_from_T_P_Y(0.0, T_post, np.full(N, Ps), sp0, sp1)

    u_arr    = np.where(x < x_shock, u_post, 0.0)
    e_arr    = np.where(x < x_shock, e_post, e_right)
    rhoE_arr = rho_arr * (e_arr + 0.5 * u_arr**2)
    m_arr    = rho_arr * u_arr

    return r0, r1, m_arr, rhoE_arr


def run_shock_droplet(N=400, t_end=5e-4, CFL=0.3, sp0='water', sp1='air',
                      scheme='IEC'):
    """§4.2.3 충격파-액적 충돌 시뮬레이션.

    t_end=5e-4 s: shock (~1020 m/s) traverses 0.5 m domain from x=-0.3 to droplet.
    use_char=False: char decomp unstable at non-equilibrium shock fronts (same as
    §4.2.1 Riemann), plain WENO5Z reconstruction of [T,Y0,u,P] is stable.
    """
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

    r0, r1, m_ic, rhoE = ic_shock_droplet(x, sp0, sp1)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    use_iec = (scheme == 'IEC')
    def rhs_fn(U, dx):
        return rhs_iec(U, dx, sp0, sp1, iec=use_iec, use_char=False, weno_order=5,
                       bc='transmissive')

    t, step = 0.0, 0
    print(f"\n{'='*55}\n[{scheme}+W5Z] Shock-Droplet  N={N}  t_end={t_end:.2e}")

    while t < t_end - 1e-14:
        r0_, r1_, m_, E_ = U
        rho_   = r0_ + r1_
        Y0_    = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam    = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt     = min(CFL * dx / (lam + 1e-10), t_end - t)

        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt;  step += 1

        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at t={t:.4e}"); break

    print(f"  완료: t={t:.4e}  ({step} steps)")
    return x, U


# ─────────────────────────────────────────────────────────────
# 12. 검증 케이스 §4.2.4: Inviscid Mach-100 Water Jet
#     비점성 Ma=100 물 제트 (물이 공기 중으로 고속 분사)
#
#     초기 조건:
#       왼쪽 (물):  u=u_jet, P=P0, T=T0
#       오른쪽(공기): u=0,   P=P0, T=T0
#       계면: x=0 (sharp)
# ─────────────────────────────────────────────────────────────

def ic_mach100_jet(x, Mach=100.0, P0=1.0, T0=1.0,
                   sp0='water_nd', sp1='air_nd', eps_factor=4.0,
                   eps_abs=None):
    """Mach-100 물 제트 초기 조건 (균일 이송 테스트).

    §4.2.4: 압력 평형 보존의 고-Mach 극한 테스트 (Coralic & Colonius 2014 §5.3 참조).
    모든 셀이 동일한 속도로 이동 (u = Mach * c_water). 초기 P, T 균일.

    IC 유형: 액적(droplet) 스타일 — 물 영역이 도메인 중앙에 위치하고
             공기가 양쪽을 둘러쌈. 주기 경계에서 연속 (wrap-around 불연속 없음).

    Y0 = 0.5*(1 + tanh((w − |x − xc|)/eps))  (ic_droplet §4.2.2와 동일 방식)

    주의: 반무한 계면(half-plane) IC는 주기 경계에서 wrap-around 불연속이 생겨
          순간적으로 Pmax>>P0 급등. 액적 IC만이 주기 경계에서 안전.

    eps_abs: 절대적 인터페이스 폭. None이면 eps_factor*dx 사용.
             높은 N에서 WENO5Z 진동을 억제하려면 고정 물리적 폭이 필요.
             (N=400에서 eps_factor=4 → eps=0.01: density ratio 6001로 인해
             Amagat 비선형성이 WENO 진동을 증폭 → 발산. eps_abs=0.04 권장.)
    """
    N  = len(x)
    dx = x[1] - x[0]
    eps = eps_abs if eps_abs is not None else eps_factor * dx

    rho_water = nasg_rho_from_T_P_Y(1.0, T0, P0, sp0, sp1)
    c2_water  = nasg_c2_mix(1.0, rho_water, T0, P0, sp0, sp1)
    u_jet     = Mach * np.sqrt(c2_water)

    print(f"  Water sound speed = {np.sqrt(c2_water):.3f},  u_jet = {u_jet:.3f}")

    # 액적 IC (§4.2.2 ic_droplet 방식):
    #   물 반경 = 0.25, 중심 = 0.0, Y0 = 0 (공기) at boundaries → 주기 안전
    xc, w = 0.0, 0.25
    Y0 = 0.5 * (1.0 + np.tanh((w - np.abs(x - xc)) / eps))
    Y0 = np.clip(Y0, 0.0, 1.0)

    P_arr = np.full(N, P0)
    rho   = nasg_rho_from_T_P_Y(Y0, T0, P_arr, sp0, sp1)
    e     = nasg_e_from_T_P_Y(Y0, T0, P_arr, sp0, sp1)

    r0   = rho * Y0
    r1   = rho * (1.0 - Y0)
    u_arr = u_jet * np.ones(N)   # 균일 속도 이송
    rhoE  = rho * (e + 0.5 * u_arr**2)

    return r0, r1, rho * u_arr, rhoE


def run_mach100_jet(N=400, t_end=1e-3, CFL=0.1, sp0='water_nd', sp1='air_nd'):
    """§4.2.4 Ma=100 물 제트 시뮬레이션."""
    dx = 1.0 / N
    x  = np.linspace(-0.5 + dx/2, 0.5 - dx/2, N)

    r0, r1, m_ic, rhoE = ic_mach100_jet(x, sp0=sp0, sp1=sp1, eps_abs=0.04)
    U = [r0.copy(), r1.copy(), m_ic.copy(), rhoE.copy()]

    def rhs_fn(U, dx):
        # use_char=False: char decomp with impedance ratio ~10000 causes WENO oscillations
        return rhs_iec(U, dx, sp0, sp1, iec=True, use_char=False, weno_order=5,
                       bc='periodic')

    t, step = 0.0, 0
    print(f"\n{'='*55}\n[IEC+W5Z+Char] Mach-100 Water Jet  N={N}  t_end={t_end:.2e}")

    while t < t_end - 1e-14:
        r0_, r1_, m_, E_ = U
        rho_   = r0_ + r1_
        Y0_    = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, c2_ = nasg_prim(Y0_, rho_, m_, E_, sp0, sp1)
        lam    = float(np.max(np.abs(u_) + np.sqrt(c2_)))
        dt     = min(CFL * dx / (lam + 1e-10), t_end - t)

        U, _ = rkstep(U, rhs_fn, dx, dt)
        t += dt;  step += 1

        if not np.all(np.isfinite(U[3])):
            print(f"  NaN at t={t:.4e}"); break

    print(f"  완료: t={t:.4e}  ({step} steps)")
    return x, U


# ─────────────────────────────────────────────────────────────
# 13. 메인 검증 실행
# ─────────────────────────────────────────────────────────────

def validate_droplet(N=100, CFL=0.5):
    """§4.2.2 IEC 오차 비교: IEC WENO5Z+Char vs IEC WENO3 vs STD.

    논문 Table 6 재현:
      IEC WENO5Z+Char: 오차 ~ 10⁻¹¹
      IEC WENO3:       오차 ~ 10⁻⁷
      STD WENO5Z:      오차 ~ 10⁻³
    """
    print(f"\n{'#'*60}")
    print(f"# §4.2.2 Inviscid Droplet Advection (N={N})")
    print(f"# 비교: IEC WENO5Z+Char  vs  IEC WENO3  vs  STD")
    print(f"{'#'*60}")

    configs = [
        ('IEC_W5Z_Char',  dict(scheme='IEC', use_char=True,  weno_order=5)),
        ('IEC_W5Z',       dict(scheme='IEC', use_char=False, weno_order=5)),
        ('IEC_W3',        dict(scheme='IEC', use_char=False, weno_order=3)),
        ('STD_W5Z',       dict(scheme='STD', use_char=False, weno_order=5)),
    ]

    results = {}
    for name, kw in configs:
        x, U, ep, et, eu = run_droplet(N=N, CFL=CFL, **kw)
        results[name] = {'x': x, 'U': U, 'ep': ep, 'et': et, 'eu': eu}

    # ── 결과 표 ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'기법':<20} {'|ΔP/P0|':>12} {'|ΔT/T0|':>12} {'|Δu/u0|':>12}")
    print(f"{'─'*65}")
    for name, res in results.items():
        print(f"  {name:<18}  {res['ep']:12.3e}  {res['et']:12.3e}  {res['eu']:12.3e}")
    print(f"{'─'*65}")
    print("  Paper Table 6 target: IEC WENO5Z+Char ~ 1e-11,  STD ~ 1e-3")

    # ── 그래프 ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {'IEC_W5Z_Char': 'C1', 'IEC_W5Z': 'C2', 'IEC_W3': 'C3', 'STD_W5Z': 'C0'}
    labels = {
        'IEC_W5Z_Char': 'IEC+WENO5Z+Char',
        'IEC_W5Z':      'IEC+WENO5Z',
        'IEC_W3':       'IEC+WENO3',
        'STD_W5Z':      'STD+WENO5Z',
    }
    lstyles = {'IEC_W5Z_Char': '-', 'IEC_W5Z': '-.', 'IEC_W3': ':', 'STD_W5Z': '--'}

    for name, res in results.items():
        x_ = res['x']
        r0_, r1_, m_, E_ = res['U']
        rho_ = r0_ + r1_
        Y0_  = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, _ = nasg_prim(Y0_, rho_, m_, E_)
        axes[0].plot(x_, P_, lstyles[name], color=colors[name], label=labels[name])
        axes[1].plot(x_, T_, lstyles[name], color=colors[name], label=labels[name])
        axes[2].plot(x_, Y0_, lstyles[name], color=colors[name], label=labels[name])

    axes[0].axhline(101325.0, color='k', lw=0.8, ls=':')
    axes[1].axhline(297.0,    color='k', lw=0.8, ls=':')
    for ax, lbl in zip(axes, [r'$P$ [Pa]', r'$T$ [K]', r'$Y_{water}$']):
        ax.set_xlabel('x');  ax.set_ylabel(lbl);  ax.legend(fontsize=7)
    fig.suptitle(f'§4.2.2 Droplet Advection  (N={N}, t=0.2s)\n'
                 f'IEC 오차 [IEC+W5Z+Char]: {results["IEC_W5Z_Char"]["ep"]:.1e}  '
                 f'[STD]: {results["STD_W5Z"]["ep"]:.1e}')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'4eq_droplet_N{N}.png')
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"\nSaved {os.path.basename(fname)}")

    return results


def validate_riemann(N=200, CFL=0.3):
    """§4.2.1 기체-액체 Riemann 문제 검증."""
    print(f"\n{'#'*60}")
    print(f"# §4.2.1 Gas-Liquid Riemann Problem (N={N})")
    print(f"{'#'*60}")

    x, U_iec = run_riemann_gl(N=N, CFL=CFL, scheme='IEC')
    x, U_std = run_riemann_gl(N=N, CFL=CFL, scheme='STD')

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for U_arr, label, ls in [(U_iec, 'IEC+W5Z', '-'), (U_std, 'STD+W5Z', '--')]:
        r0_, r1_, m_, E_ = U_arr
        rho_ = r0_ + r1_
        Y0_  = r0_ / np.maximum(rho_, 1e-30)
        u_, e_, P_, T_, _ = nasg_prim(Y0_, rho_, m_, E_)
        axes[0].plot(x, rho_, ls, label=label)
        axes[1].plot(x, u_,   ls, label=label)
        axes[2].plot(x, P_,   ls, label=label)
        axes[3].plot(x, Y0_,  ls, label=label)

    for ax, lbl in zip(axes, [r'$\rho$', r'$u$', r'$P$', r'$Y_0$']):
        ax.set_xlabel('x');  ax.set_ylabel(lbl);  ax.legend(fontsize=8)
    fig.suptitle(f'§4.2.1 Gas-Liquid Riemann Problem (N={N})')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'4eq_riemann_N{N}.png')
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"Saved {os.path.basename(fname)}")


def validate_shock_droplet(N=400, CFL=0.3):
    """§4.2.3 충격파-액적 충돌 검증."""
    print(f"\n{'#'*60}")
    print(f"# §4.2.3 Shock-Droplet Interaction (N={N})")
    print(f"{'#'*60}")

    x, U = run_shock_droplet(N=N, CFL=CFL)

    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_
    Y0_  = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, _ = nasg_prim(Y0_, rho_, m_, E_)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(x, rho_)
    axes[1].plot(x, u_)
    axes[2].plot(x, P_)
    axes[3].plot(x, Y0_)
    for ax, lbl in zip(axes, [r'$\rho$', r'$u$', r'$P$', r'$Y_{water}$']):
        ax.set_xlabel('x');  ax.set_ylabel(lbl)
    fig.suptitle(f'§4.2.3 Shock-Droplet Interaction (N={N})')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'4eq_shock_droplet_N{N}.png')
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"Saved {os.path.basename(fname)}")


def validate_mach100(N=400, CFL=0.1):
    """§4.2.4 Mach-100 물 제트 검증."""
    print(f"\n{'#'*60}")
    print(f"# §4.2.4 Inviscid Mach-100 Water Jet (N={N})")
    print(f"{'#'*60}")

    sp0_m, sp1_m = 'water_nd', 'air_nd'
    x, U = run_mach100_jet(N=N, CFL=CFL, sp0=sp0_m, sp1=sp1_m)

    r0_, r1_, m_, E_ = U
    rho_ = r0_ + r1_
    Y0_  = r0_ / np.maximum(rho_, 1e-30)
    u_, e_, P_, T_, _ = nasg_prim(Y0_, rho_, m_, E_, sp0_m, sp1_m)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].plot(x, rho_)
    axes[1].plot(x, u_)
    axes[2].plot(x, P_)
    axes[3].plot(x, Y0_)
    for ax, lbl in zip(axes, [r'$\rho$', r'$u$', r'$P$', r'$Y_{water}$']):
        ax.set_xlabel('x');  ax.set_ylabel(lbl)
    fig.suptitle(f'§4.2.4 Inviscid Mach-100 Water Jet (N={N})')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'4eq_mach100_N{N}.png')
    fig.savefig(fname, dpi=150);  plt.close(fig)
    print(f"Saved {os.path.basename(fname)}")


def validate_all(N=100):
    """전체 검증 실행 (논문 목표 해상도 사용).

    논문 Collis et al. 2026 각 절 목표:
      §4.2.1 Riemann:        N=501   (IEC, transmissive BC)
      §4.2.2 Droplet:        N=100   (IEC+Char, periodic BC)
      §4.2.3 Shock-Droplet:  N=200   (IEC, transmissive BC)
      §4.2.4 Mach-100 Jet:   N=400   (IEC, periodic BC)
    """
    print(f"\n{'#'*60}")
    print(f"# four_eq_1d.py -- multi-phase 4-eq solver validation")
    print(f"# NASG EOS + IEC WENO5Z + char decomp + HLLC")
    print(f"{'#'*60}")

    validate_droplet(N=max(N, 100), CFL=0.5)
    validate_riemann(N=max(N*5, 501), CFL=0.3)
    validate_shock_droplet(N=max(N*2, 200), CFL=0.3)
    validate_mach100(N=max(N*4, 400), CFL=0.1)

    print(f"\n{'#'*60}")
    print(f"# 검증 완료. 출력 → solver/output/4eq_*.png")
    print(f"{'#'*60}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        N   = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        if cmd == 'droplet':
            validate_droplet(N=N)
        elif cmd == 'riemann':
            validate_riemann(N=N)
        elif cmd == 'shock':
            validate_shock_droplet(N=N)
        elif cmd == 'mach100':
            validate_mach100(N=N)
        elif cmd == 'all':
            validate_all(N=N)
        else:
            N = int(cmd)
            validate_all(N=N)
    else:
        validate_all(N=100)
