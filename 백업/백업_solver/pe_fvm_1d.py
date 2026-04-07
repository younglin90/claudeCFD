"""
PE-consistent FVM 스킴 (1D, CPG 혼합물)
========================================
Wang et al. JCP 2025의 KEEPPE 아이디어를 FVM + general EOS + 3D 비정렬
확장에 적합한 형태로 재정식화.

핵심 아이디어 (Terashima et al. JCP 2025 + Wang et al. JCP 2025):
  - FVM 패러다임: 각 셀 경계(face)에서 플럭스 계산 후 발산 형태로 업데이트
  - PE-consistent 에너지 플럭스: ρe_f = 0.5*(ρe_L + ρe_R) - 0.5*Σ_i ε_i_f*(ρY_i_R - ρY_i_L)
    * ε_i = (∂ρe/∂ρY_i)_{p, ρY_{j≠i}}: 압력 고정 내부에너지 민감도
    * CPG에서 해석적으로: ε_i = (p/ρ) * (W_mix/W_i) * (1/(γ_i-1) - 1/(γ_mix-1))
  - 종 플럭스: avg(ρY_i)*avg(u)  [Q2와 동등, Wang Q2 스킴]
  - 3D 비정렬 확장: face-normal 속도 u_n = u⃗·n̂_f 로 스칼라 플럭스 구성.
    각 face에서 (QL, QR, n̂_f)를 입력받아 수직 방향 플럭스를 계산.
    현재 1D 구현에서는 n̂_f = [1,0,0] (오른쪽 방향)이 고정.

지원 스킴:
  'DIV_FVM' : 단순 FVM (ρE 단순 평균) — 비교 기준
  'PE_FVM'  : APEC 에너지 보정 포함 — 메인 스킴

인터페이스 호환:
  many_flux_1d.py의 run_case와 완전히 동일한 시그니처.
"""

import numpy as np
import sys
import os

# ─────────────────────────────────────────────
# 상수 (many_flux_1d.py에서 재사용)
# ─────────────────────────────────────────────
Ru = 8.314  # J/(mol·K)

# 종 물성 (SI 단위)
SPECIES = {
    'H2':  {'W': 0.002016,  'gamma': 1.4},
    'N2':  {'W': 0.028014,  'gamma': 1.4},
    'H2O': {'W': 0.018015,  'gamma': 1.33},
    'O2':  {'W': 0.031999,  'gamma': 1.4},
}


# ─────────────────────────────────────────────
# EOS — 열량완전기체 혼합물 (many_flux_1d.py에서 재사용)
# ─────────────────────────────────────────────

def mixture_props(Ys, species_list):
    """
    혼합물 열역학 물성 계산.

    Parameters
    ----------
    Ys : ndarray, shape (..., Ns)  질량분율
    species_list : list[str]

    Returns
    -------
    W     : 평균 분자량 (kg/mol)
    gamma : 혼합 비열비
    Cv    : 비정적비열 (J/kg/K)
    """
    Ws = np.array([SPECIES[s]['W']     for s in species_list])  # (Ns,)
    gs = np.array([SPECIES[s]['gamma'] for s in species_list])  # (Ns,)

    # 1/W_mix = Σ Y_α / W_α
    inv_W = np.dot(Ys, 1.0 / Ws)    # (...)
    W = 1.0 / inv_W

    # 몰분율 X_α = (Y_α/W_α) * W_mix
    Xs = (Ys / Ws) * W[..., None]

    # 1/(γ_mix - 1) = Σ X_α / (γ_α - 1)
    inv_gm1 = np.dot(Xs, 1.0 / (gs - 1.0))
    gm1   = 1.0 / inv_gm1
    gamma = gm1 + 1.0

    Cv = (Ru / W) / gm1
    return W, gamma, Cv


def pressure(rho, rhoE, rhou, Ys, species_list):
    """
    보존변수 → 압력  (CPG: p = ρe(γ-1))

    Parameters
    ----------
    rho, rhoE, rhou : ndarray, shape (...)
    Ys              : ndarray, shape (..., Ns)
    """
    u   = rhou / rho
    e   = rhoE / rho - 0.5 * u**2
    _, gamma, _ = mixture_props(Ys, species_list)
    return rho * e * (gamma - 1.0)


def temperature(rho, p, Ys, species_list):
    """압력·밀도 → 온도  (이상기체: T = p*W/(Ru*ρ))"""
    W, _, _ = mixture_props(Ys, species_list)
    return p * W / (Ru * rho)


# ─────────────────────────────────────────────
# PE-FVM 전용 EOS 헬퍼
# ─────────────────────────────────────────────

def rhoe_from_p(rho, p, Ys, species_list):
    """
    압력으로부터 EOS-consistent ρe 계산.

    CPG:  ρe = p / (γ_mix - 1)

    3D 비정렬 확장:
      face의 (rho_f, p_f, Ys_f)에 동일 공식 적용.

    Parameters
    ----------
    rho  : ndarray, shape (...)  (현재 사용 안 함; 범용 EOS 확장 시 필요)
    p    : ndarray, shape (...)
    Ys   : ndarray, shape (..., Ns)

    Returns
    -------
    rhoe : ndarray, shape (...)
    """
    _, gamma, _ = mixture_props(Ys, species_list)
    return p / (gamma - 1.0)


def eps_species(rho, p, Ys, species_list):
    """
    압력-평형 보존 에너지 민감도 ε_i 계산.

    정의:
      ε_i = (∂ρe / ∂(ρY_i))_{p, ρY_{j≠i}}

    CPG 해석해 (유도):
      ρe = p/(γ_mix-1)
      γ_mix는 Ys에 의존하므로 ∂(ρe)/∂(ρY_i)를 고정 p에서 계산.

      ρe = p * inv_gm1  (inv_gm1 = 1/(γ_mix-1))
      ∂ρe/∂(ρY_i) = p * ∂inv_gm1/∂(ρY_i)

      inv_gm1 = Σ_α X_α/(γ_α-1)
               = Σ_α [Y_α*W_mix/W_α] / (γ_α-1)
               = W_mix * Σ_α Y_α/[W_α*(γ_α-1)]

      ∂(ρY_i): ρ, ρY_{j≠i} 고정 → Y_i = (ρY_i)/ρ 증가, W_mix 변화.

      단계별 전개:
        W_mix = 1 / (Σ_α Y_α/W_α)
        inv_gm1 = W_mix * Σ_α (Y_α/W_α)/(γ_α-1)
                = (Σ_α Y_α/[W_α(γ_α-1)]) / (Σ_β Y_β/W_β)

      ∂(ρe)/∂(ρY_i)
        = p * ∂inv_gm1/∂Y_i * (1/ρ)
        = (p/ρ) * [
            1/(W_i*(γ_i-1)) * (1/inv_W_mix)
          - (1/W_i) * Σ_α Y_α/[W_α(γ_α-1)] / inv_W_mix**2
          ]
        = (p/ρ) * W_mix/W_i * [1/(γ_i-1) - 1/(γ_mix-1)]

    최종 공식:
      ε_i = (p/ρ) * (W_mix/W_i) * (1/(γ_i-1) - 1/(γ_mix-1))

    3D 비정렬 확장:
      face 중심 물성 (rho_f, p_f, Ys_f)에 동일 공식 적용.
      shape (..., Ns) 출력.

    Parameters
    ----------
    rho  : ndarray, shape (...)
    p    : ndarray, shape (...)
    Ys   : ndarray, shape (..., Ns)

    Returns
    -------
    eps  : ndarray, shape (..., Ns)
    """
    Ns   = len(species_list)
    Ws   = np.array([SPECIES[s]['W']     for s in species_list])  # (Ns,)
    gs   = np.array([SPECIES[s]['gamma'] for s in species_list])  # (Ns,)

    W_mix, gamma_mix, _ = mixture_props(Ys, species_list)

    # broadcast shape (..., Ns)
    p_over_rho = (p / rho)[..., None]          # (..., 1)
    W_mix_bc   = W_mix[..., None]              # (..., 1)
    gm1_mix    = (gamma_mix - 1.0)[..., None]  # (..., 1)

    # 1/(γ_i-1) - 1/(γ_mix-1)
    diff_inv_gm1 = 1.0 / (gs - 1.0) - 1.0 / gm1_mix  # (..., Ns)

    eps = p_over_rho * (W_mix_bc / Ws) * diff_inv_gm1  # (..., Ns)
    return eps


# ─────────────────────────────────────────────
# FVM 인터페이스 플럭스
# ─────────────────────────────────────────────

def interface_flux(QL, QR, species_list, scheme):
    """
    셀 경계 LLF(Rusanov) 플럭스 계산 (1D, face-normal = +x 방향).

    두 스킴 모두 LLF(Rusanov) 기반으로 수치 소산을 포함:
      DIV_FVM : 표준 LLF — 에너지 소산에 표준 δ(ρE) 사용
      PE_FVM  : APEC-LLF — 에너지 소산에 PE-일관 δ(ρE)_pep 사용
                (Terashima et al. JCP 2025 / 이 코드의 CLAUDE.md 참조)

    APEC 에너지 소산 공식:
      δ(ρE)_pep = Σ_i ε_i_f * δ(ρY_i) + ½*u_f²*δρ + ρ_f*u_f*δu
      이것은 p=일정 조건에서 δ(ρE)의 1차 근사이므로
      PE 보존을 해치는 가짜 소산을 방지한다.

    3D 비정렬 확장 시:
      - face_normal = n̂_f (unit vector, shape (Nf, 3)) 추가
      - u_n = u⃗ · n̂_f (법선 속도), u_t = u⃗ - u_n*n̂_f (접선 속도)
      - F_mass = ρ_f * u_n_f
      - F_mom  = F_mass * u_f_vec + p_f * n̂_f   (벡터, shape (Nf,3))
      - F_E    = (ρE_f + p_f) * u_n_f / ρ_f * ρ_f  = (ρe_f+½ρ|u|²+p_f)*u_n_f
      - F_Yi   = avg(ρY_i) * u_n_f
      - LLF 소산: -0.5*λ_max*(Q_R - Q_L) in each component

    Parameters
    ----------
    QL, QR       : ndarray, shape (Nf, 3+Ns)
    species_list : list[str]
    scheme       : 'DIV_FVM' | 'PE_FVM'

    Returns
    -------
    F : ndarray, shape (Nf, 3+Ns)
    """
    Ns = len(species_list)
    Nf = QL.shape[0]

    # ── 원시변수 계산 ──────────────────────────────────────────────
    rho_L  = QL[:, 0];  rho_R  = QR[:, 0]
    rhou_L = QL[:, 1];  rhou_R = QR[:, 1]
    rhoE_L = QL[:, 2];  rhoE_R = QR[:, 2]
    Ys_L   = QL[:, 3:3+Ns] / rho_L[:, None]
    Ys_R   = QR[:, 3:3+Ns] / rho_R[:, None]
    u_L    = rhou_L / rho_L;  u_R = rhou_R / rho_R
    p_L    = pressure(rho_L, rhoE_L, rhou_L, Ys_L, species_list)
    p_R    = pressure(rho_R, rhoE_R, rhou_R, Ys_R, species_list)
    e_L    = rhoE_L / rho_L - 0.5 * u_L**2
    e_R    = rhoE_R / rho_R - 0.5 * u_R**2
    rhoe_L = rho_L * e_L;  rhoe_R = rho_R * e_R

    # ── 음속 및 LLF 파속 ──────────────────────────────────────────
    _, gamma_L, _ = mixture_props(Ys_L, species_list)
    _, gamma_R, _ = mixture_props(Ys_R, species_list)
    c_L = np.sqrt(np.maximum(gamma_L * p_L / rho_L, 0.0))
    c_R = np.sqrt(np.maximum(gamma_R * p_R / rho_R, 0.0))
    # LLF(Rusanov) 파속: 각 face의 최대 고유치
    lam = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)  # (Nf,)

    # ── face 평균값 ──────────────────────────────────────────────
    rho_f = 0.5 * (rho_L + rho_R)
    u_f   = 0.5 * (u_L + u_R)
    p_f   = 0.5 * (p_L + p_R)
    Ys_f  = 0.5 * (Ys_L + Ys_R)

    # ── 물리 플럭스 (중앙값) ────────────────────────────────────────
    # F_L = [ρu, ρu²+p, (ρE+p)u]_L,  F_R = 오른쪽
    FC_L = rho_L * u_L;           FC_R = rho_R * u_R
    FM_L = rho_L*u_L**2 + p_L;    FM_R = rho_R*u_R**2 + p_R
    FE_L = (rhoE_L + p_L) * u_L;  FE_R = (rhoE_R + p_R) * u_R

    # ── 질량 플럭스 (LLF) ─────────────────────────────────────────
    # 3D 확장: F_mass = 0.5*(ρ_L*u_nL + ρ_R*u_nR) - 0.5*λ*(ρ_R - ρ_L)
    F_mass = 0.5*(FC_L + FC_R) - 0.5*lam*(rho_R - rho_L)

    # ── 운동량 플럭스 (LLF) ───────────────────────────────────────
    # 3D 확장: F_mom_vec = 0.5*(FM_L + FM_R)*n̂ - 0.5*λ*(ρu_R - ρu_L)
    F_mom = 0.5*(FM_L + FM_R) - 0.5*lam*(rhou_R - rhou_L)

    # ── 에너지 플럭스 ─────────────────────────────────────────────
    if scheme == 'DIV_FVM':
        # 표준 LLF: 에너지 소산에 표준 δ(ρE) 사용
        F_E = 0.5*(FE_L + FE_R) - 0.5*lam*(rhoE_R - rhoE_L)

    elif scheme == 'PE_FVM':
        # APEC-LLF: 에너지 소산에 PE-일관 δ(ρE)_pep 사용
        #
        # 아이디어 (Terashima et al. APEC):
        #   표준 δ(ρE) = ρE_R - ρE_L은 p 변화를 포함하여 PE 오류 유발.
        #   대신, p=const 조건에서의 에너지 변화로 근사:
        #     δ(ρE)_pep = Σ_i ε_i*δ(ρY_i) + ½*u_f²*δρ + ρ_f*u_f*δu
        #
        # 3D 비정렬 확장:
        #   ε_i, δ(ρY_i)는 스칼라 → face-normal과 무관하게 동일 공식.
        #   δu = u_nR - u_nL  (법선 속도 차이만 사용)
        eps_f  = eps_species(rho_f, p_f, Ys_f, species_list)  # (Nf, Ns)
        drhoY  = QR[:, 3:3+Ns] - QL[:, 3:3+Ns]               # (Nf, Ns)
        drho   = rho_R - rho_L                                 # (Nf,)
        du     = u_R - u_L                                     # (Nf,)

        # PE-일관 에너지 점프: ρe 변화 + KE 변화 (선형화)
        drhoe_pep = np.sum(eps_f * drhoY, axis=1)              # Σ ε_i δ(ρY_i)
        drhoKE    = 0.5 * u_f**2 * drho + rho_f * u_f * du    # δ(½ρu²) 선형화
        drhoE_pep = drhoe_pep + drhoKE                         # (Nf,)

        F_E = 0.5*(FE_L + FE_R) - 0.5*lam*drhoE_pep

    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # ── 종 플럭스: Y_f * F_mass ────────────────────────────────────
    # F_{ρY_i} = Y_i_f * F_mass (LLF 포함)
    #
    # 근거: F_mass = 0.5*(ρ_L*u_L + ρ_R*u_R) - 0.5*λ*(ρ_R - ρ_L)
    #       F_{ρY_i} = Y_i_f * F_mass
    #       → 균일 Y_i (S1): F_{ρY_i} = Y_i * F_mass
    #                         d(ρY_i)/dt = Y_i * d(ρ)/dt
    #                         → d(Y_i)/dt = 0  (기계 정밀도)
    #
    # 3D 비정렬 확장:
    #   F_Yi = Y_i_f * F_mass  (F_mass는 법선 방향 스칼라 플럭스)
    Y_f = 0.5 * (Ys_L + Ys_R)   # (Nf, Ns) — face 평균 질량분율

    F = np.zeros((Nf, 3 + Ns))
    F[:, 0] = F_mass
    F[:, 1] = F_mom
    F[:, 2] = F_E

    for s in range(Ns - 1):
        F[:, 3+s] = Y_f[:, s] * F_mass

    # 마지막 종: 질량 일관성 (Σ Y_i = 1)
    F[:, 3+Ns-1] = F_mass - np.sum(F[:, 3:3+Ns-1], axis=1)

    return F


# ─────────────────────────────────────────────
# FVM RHS
# ─────────────────────────────────────────────

def rhs_fvm(Q, species_list, scheme, dx):
    """
    FVM 공간 이산화: dQ/dt = -1/dx * (F_{m+1/2} - F_{m-1/2})

    주기 경계조건 적용.

    Parameters
    ----------
    Q            : ndarray, shape (N, 3+Ns)
    species_list : list[str]
    scheme       : str  'DIV_FVM' | 'PE_FVM'
    dx           : float  셀 크기

    Returns
    -------
    dQdt : ndarray, shape (N, 3+Ns)

    3D 비정렬 확장 노트:
      - Q : shape (Ncells, 3+Ns)
      - face connectivity: (Nfaces, 2) 정수 배열로 좌우 셀 인덱스 관리
      - QL = Q[face_left_idx],  QR = Q[face_right_idx]
      - F = interface_flux(QL, QR, species_list, scheme, face_normals)
      - 각 셀에 대해 플럭스 산란(scatter): dQdt[cell_i] -= F[f]*face_area[f] / vol[i]
        (나가는 face는 +, 들어오는 face는 -)
    """
    N  = Q.shape[0]
    Ns = len(species_list)

    # face 인덱스: face m+1/2 는 셀 m (L)과 셀 m+1 (R) 사이
    # 주기 경계: face N-1/2 는 셀 N-1 (L)과 셀 0 (R)
    idx_L = np.arange(N)
    idx_R = (idx_L + 1) % N

    QL = Q[idx_L]  # (N, 3+Ns)
    QR = Q[idx_R]  # (N, 3+Ns)

    # face 플럭스 F_{m+1/2}
    F_right = interface_flux(QL, QR, species_list, scheme)  # (N, 3+Ns)

    # F_{m-1/2} = F_right rolled by +1
    F_left = np.roll(F_right, 1, axis=0)  # F_{m-1/2}[m] = F_{m-1/2}

    # RHS: -1/dx * (F_{m+1/2} - F_{m-1/2})
    dQdt = -(F_right - F_left) / dx

    return dQdt


# ─────────────────────────────────────────────
# RHS 팩토리 (many_flux_1d.make_rhs와 동일 인터페이스)
# ─────────────────────────────────────────────

def make_rhs_fvm(scheme, species_list, dx):
    """
    FVM RHS 함수 생성기.

    Parameters
    ----------
    scheme       : str   'DIV_FVM' | 'PE_FVM'
    species_list : list[str]
    dx           : float

    Returns
    -------
    rhs : callable  rhs(Q) → dQdt
    """
    def rhs(Q):
        return rhs_fvm(Q, species_list, scheme, dx)
    return rhs


# ─────────────────────────────────────────────
# 시간 적분 (many_flux_1d.py에서 재사용)
# ─────────────────────────────────────────────

def rk4_step(Q, rhs, dt):
    """고전 4차 Runge-Kutta 시간 적분."""
    k1 = rhs(Q)
    k2 = rhs(Q + 0.5 * dt * k1)
    k3 = rhs(Q + 0.5 * dt * k2)
    k4 = rhs(Q + dt * k3)
    return Q + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def spectral_filter(Q):
    """
    2/3 규칙 스펙트럼 디앨리어싱 필터.
    |k| > N/3 인 고파수 모드를 0으로 만들어 비선형 앨리어싱 억제.
    모든 보존변수에 동시 적용하여 Σ ρY_s = ρ 일관성 유지.
    """
    N = Q.shape[0]
    cutoff = N // 3
    Q_f = np.zeros_like(Q)
    for col in range(Q.shape[1]):
        f_hat = np.fft.rfft(Q[:, col])
        f_hat[cutoff:] = 0.0
        Q_f[:, col] = np.fft.irfft(f_hat, N)
    return Q_f


def cfl_dt(Q, dx, CFL, species_list):
    """CFL 조건에 따른 시간 스텝 계산."""
    rho  = Q[:, 0]
    rhou = Q[:, 1]
    rhoE = Q[:, 2]
    Ns   = Q.shape[1] - 3
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    u    = rhou / rho
    p    = pressure(rho, rhoE, rhou, Ys, species_list)
    _, gamma, _ = mixture_props(Ys, species_list)
    c    = np.sqrt(np.abs(gamma * p / rho))
    lam  = np.max(np.abs(u) + c)
    return CFL * dx / lam


# ─────────────────────────────────────────────
# 초기조건 (many_flux_1d.py에서 재사용)
# ─────────────────────────────────────────────

def init_G1(N=61):
    """G1: H2/N2 균일 γ"""
    species = ['H2', 'N2']
    L  = 1.0
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho  = 1.0 + np.exp(np.sin(2*np.pi*x))
    u    = np.ones(N)
    p    = np.ones(N)

    Y_H2 = (np.e - np.exp(np.sin(2*np.pi*x))) / (np.e - np.exp(-1.0))
    Y_N2 = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e   = p / (rho * (gamma - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    Q[:, 3] = rho * Y_H2
    Q[:, 4] = rho * Y_N2
    return Q, x, dx, species


def init_G2(N=61):
    """G2: H2/H2O 변화 γ"""
    species = ['H2', 'H2O']
    L  = 1.0
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho  = 1.0 + np.exp(np.sin(2*np.pi*x))
    u    = np.ones(N)
    p    = np.ones(N)

    Y_H2  = (np.e - np.exp(np.sin(2*np.pi*x))) / (np.e - np.exp(-1.0))
    Y_H2O = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_H2O], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e   = p / (rho * (gamma - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    Q[:, 3] = rho * Y_H2
    Q[:, 4] = rho * Y_H2O
    return Q, x, dx, species


def init_S1(N=41):
    """S1: 균일 질량분율 보존 (1D, H2/N2)"""
    species = ['H2', 'N2']
    L  = 1.0
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho  = 2.0 + np.sin(2*np.pi*x)
    u    = 1.0 + 0.1 * np.sin(2*np.pi*x)
    p    = 10.0 * np.ones(N)
    Y_H2 = 0.5 * np.ones(N)
    Y_N2 = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e   = p / (rho * (gamma - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    Q[:, 3] = rho * Y_H2
    Q[:, 4] = rho * Y_N2
    return Q, x, dx, species


def init_S2(N=41):
    """S2: 온도 평형 보존 (1D, H2/O2/N2 3성분)"""
    species = ['H2', 'O2', 'N2']
    L  = 1.0
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 2.0 + np.sin(2*np.pi*x)
    u   = np.ones(N)
    p   = np.ones(N)
    T   = np.ones(N)

    Y_O2 = 0.1 * np.ones(N)

    W_mix = p / (Ru * rho * T)

    WH2 = SPECIES['H2']['W']
    WO2 = SPECIES['O2']['W']
    WN2 = SPECIES['N2']['W']

    inv_W   = 1.0 / W_mix
    A_coef  = 1.0/WH2 - 1.0/WN2
    B_coef  = Y_O2/WO2 + (1.0 - Y_O2)/WN2
    Y_H2    = (inv_W - B_coef) / A_coef
    Y_H2    = np.clip(Y_H2, 0.01, 0.89)
    Y_N2    = 1.0 - Y_H2 - Y_O2

    Ys = np.stack([Y_H2, Y_O2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e   = p / (rho * (gamma - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3+3))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    Q[:, 3] = rho * Y_H2
    Q[:, 4] = rho * Y_O2
    Q[:, 5] = rho * Y_N2
    return Q, x, dx, species


# ─────────────────────────────────────────────
# 오차 측정 (many_flux_1d.py에서 재사용)
# ─────────────────────────────────────────────

def pressure_error(Q, Q0, species_list):
    """||ε_p||₁ = (1/N) Σ |p(t) - p(0)|"""
    Ns   = len(species_list)
    rho  = Q[:, 0];  rhou = Q[:, 1];  rhoE = Q[:, 2]
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    p    = pressure(rho, rhoE, rhou, Ys, species_list)

    rho0  = Q0[:, 0];  rhou0 = Q0[:, 1];  rhoE0 = Q0[:, 2]
    Ys0   = Q0[:, 3:3+Ns] / rho0[:, None]
    p0    = pressure(rho0, rhoE0, rhou0, Ys0, species_list)

    return np.mean(np.abs(p - p0))


def species_error(Q, Q0, species_list, idx=0):
    """||ε_Y||₁"""
    Y  = Q[:, 3+idx]  / Q[:, 0]
    Y0 = Q0[:, 3+idx] / Q0[:, 0]
    return np.mean(np.abs(Y - Y0))


def temperature_error(Q, Q0, species_list):
    """||ε_T||₁"""
    Ns   = len(species_list)
    rho  = Q[:, 0];  rhou = Q[:, 1];  rhoE = Q[:, 2]
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    p    = pressure(rho, rhoE, rhou, Ys, species_list)
    T    = temperature(rho, p, Ys, species_list)

    rho0  = Q0[:, 0];  rhou0 = Q0[:, 1];  rhoE0 = Q0[:, 2]
    Ys0   = Q0[:, 3:3+Ns] / rho0[:, None]
    p0    = pressure(rho0, rhoE0, rhou0, Ys0, species_list)
    T0    = temperature(rho0, p0, Ys0, species_list)

    return np.mean(np.abs(T - T0))


# ─────────────────────────────────────────────
# 시뮬레이션 실행 (many_flux_1d.run_case와 동일 인터페이스)
# ─────────────────────────────────────────────

def run_case(init_func, scheme_name, t_end, CFL=0.01, species_scheme=None, N=None,
             use_filter=False):
    """
    단일 케이스 실행. 시간별 ||ε_p||₁ 기록.

    many_flux_1d.run_case와 완전히 동일한 시그니처.
    species_scheme 파라미터는 인터페이스 호환을 위해 유지하나 현재 미사용
    (PE_FVM은 항상 Q2 스타일 종 플럭스 사용).

    Parameters
    ----------
    init_func    : callable  () → (Q0, x, dx, species_list)
    scheme_name  : str  'DIV_FVM' | 'PE_FVM'
    t_end        : float
    CFL          : float
    species_scheme : str | None  (인터페이스 호환, 무시됨)
    N            : int | None  격자 수 오버라이드
    use_filter   : bool  True이면 매 스텝 스펙트럼 필터 적용

    Returns
    -------
    times    : ndarray
    pe_errors: ndarray
    Q        : ndarray  최종 상태
    """
    if N is not None:
        Q0, x, dx, species_list = init_func(N)
    else:
        Q0, x, dx, species_list = init_func()

    rhs = make_rhs_fvm(scheme_name, species_list, dx)

    Q = Q0.copy()
    t = 0.0
    times     = [0.0]
    pe_errors = [pressure_error(Q, Q0, species_list)]

    while t < t_end:
        dt = cfl_dt(Q, dx, CFL, species_list)
        if t + dt > t_end:
            dt = t_end - t

        Q = rk4_step(Q, rhs, dt)

        if use_filter:
            Q = spectral_filter(Q)

        t += dt

        # 발산 감지
        if not np.all(np.isfinite(Q)) or np.any(Q[:, 0] < 0):
            print(f"  [{scheme_name}] 발산: t={t:.4f}")
            times.append(t)
            pe_errors.append(np.nan)
            break

        times.append(t)
        pe_errors.append(pressure_error(Q, Q0, species_list))

    return np.array(times), np.array(pe_errors), Q
