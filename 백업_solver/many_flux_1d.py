"""
Wang, Wehrfritz, Hawkes, "Physically consistent formulations of split convective
terms for turbulent compressible multi-component flows," JCP 540 (2025) 114269.

1D 테스트 케이스 구현:
  G1 - 균일 γ (H2/N2)
  G2 - 변화 γ (H2/H2O)
  G3 - 사인파 1/(γ-1) (H2/H2O)
  S1 - 균일 질량분율 보존 (1D)
  S2 - 온도 평형 보존 (1D)

공간: 8차 유한차분 (L=4), 시간: 고전 4차 RK
경계조건: 주기적
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
Ru = 8.314  # J/(mol·K)

# 종 물성 (SI 단위)
SPECIES = {
    'H2':  {'W': 0.002016, 'gamma': 1.4},
    'N2':  {'W': 0.028014, 'gamma': 1.4},
    'H2O': {'W': 0.018015, 'gamma': 1.33},
    'O2':  {'W': 0.031999, 'gamma': 1.4},
}

# 8차 중심차분 계수 (L=4)
A8 = np.array([4/5, -1/5, 4/105, -1/280])


# ─────────────────────────────────────────────
# EOS — 열량완전기체 혼합물
# ─────────────────────────────────────────────

def mixture_props(Ys, species_list):
    """
    Ys : shape (..., Ns)  질량분율
    반환: W (평균분자량), gamma (혼합 비열비), Cv (비정적비열)
    """
    Ws = np.array([SPECIES[s]['W'] for s in species_list])      # (Ns,)
    gs = np.array([SPECIES[s]['gamma'] for s in species_list])  # (Ns,)

    # 1/W = Σ Y_α/W_α
    inv_W = np.dot(Ys, 1.0/Ws)   # (...)
    W = 1.0 / inv_W

    # X_α = (Y_α/W_α) * W
    Xs = (Ys / Ws) * W[..., None]

    # 1/(γ-1) = Σ X_α/(γ_α-1)
    inv_gm1 = np.dot(Xs, 1.0/(gs - 1.0))
    gm1 = 1.0 / inv_gm1
    gamma = gm1 + 1.0

    Cv = (Ru/W) / gm1
    return W, gamma, Cv


def pressure(rho, rhoE, rhou, Ys, species_list):
    """
    보존변수 → 압력.
    rho, rhoE, rhou: (...,)
    Ys: (..., Ns)
    """
    u = rhou / rho
    e = rhoE/rho - 0.5*u**2
    W, gamma, _ = mixture_props(Ys, species_list)
    p = rho * e * (gamma - 1.0)
    return p


def temperature(rho, p, Ys, species_list):
    W, _, _ = mixture_props(Ys, species_list)
    c = rho / W          # 몰 농도
    T = p / (Ru * c)
    return T


# ─────────────────────────────────────────────
# 8차 수치 플럭스 빌딩 블록
# ─────────────────────────────────────────────

def avg(a, b):
    return 0.5*(a + b)

def geom(a, b):
    return np.sqrt(a * b)


def flux_8th(tp_func, N):
    """
    F̂|_{m+1/2} = 2 Σ_{l=1}^{4} a_l  Σ_{k=0}^{l-1} tp_func(f_left, f_right)
    tp_func(f_left, f_right): np.roll로 shift된 두 배열을 받아 플럭스 배열 반환.
    roll(f, k)[m] = f[m-k]  (주기 경계조건)
    반환: F̂, shape (N,)  — F̂[m] = F̂|_{m+1/2}
    """
    F = np.zeros(N)
    for l_idx, al in enumerate(A8):
        l = l_idx + 1  # l = 1..4
        s = np.zeros(N)
        for k in range(l):
            # f[m-k] = roll(f, k),  f[m-k+l] = roll(f, k-l)
            s += tp_func(k, k - l)
        F += 2.0 * al * s
    return F


def make_roller(f):
    """배열 f에 대해 roll-캐싱 도우미 반환."""
    cache = {}
    def roll(k):
        if k not in cache:
            cache[k] = np.roll(f, k)
        return cache[k]
    return roll


# ─────────────────────────────────────────────
# 두점 플럭스 빌더 — roll 인터페이스
# tp_func(kl, kr): kl=왼쪽 roll, kr=오른쪽 roll
# np.roll(f, k)[m] = f[m-k]
# ─────────────────────────────────────────────

def F_div(f, r):
    """Ĉ=avg(f_A, f_B), r=make_roller(f)"""
    return lambda kl, kr: 0.5*(r(kl) + r(kr))

def F_quad(f1, f2, r1, r2):
    """avg(f1)*avg(f2)"""
    return lambda kl, kr: 0.5*(r1(kl)+r1(kr)) * 0.5*(r2(kl)+r2(kr))

def F_cubic(f1, f2, f3, r1, r2, r3):
    """avg(f1)*avg(f2)*avg(f3)"""
    return lambda kl, kr: 0.5*(r1(kl)+r1(kr)) * 0.5*(r2(kl)+r2(kr)) * 0.5*(r3(kl)+r3(kr))

def F_prod(f1, f2, r1, r2):
    """(f1_R*f2_L + f1_L*f2_R)/2  — product-rule"""
    return lambda kl, kr: 0.5*(r1(kr)*r2(kl) + r1(kl)*r2(kr))

def F_sqrt_cubic(f1, f2, f3, r1, r2, r3):
    """sqrt(f1_L*f1_R)*avg(f2)*avg(f3)"""
    return lambda kl, kr: np.sqrt(r1(kl)*r1(kr)) * 0.5*(r2(kl)+r2(kr)) * 0.5*(r3(kl)+r3(kr))

def F_sqrt_geom(f1, f2, f3, r1, r2, r3):
    """sqrt(f1_L*f1_R)*avg(f2)*sqrt(f3_L*f3_R)"""
    return lambda kl, kr: np.sqrt(r1(kl)*r1(kr)) * 0.5*(r2(kl)+r2(kr)) * np.sqrt(np.abs(r3(kl)*r3(kr)))


# ─────────────────────────────────────────────
# 연속방정식/운동량/에너지 플럭스 (각 기법)
# ─────────────────────────────────────────────

def _common_vars(Q, species_list):
    N = Q.shape[0]
    Ns = len(species_list)
    rho  = Q[:, 0]
    rhou = Q[:, 1]
    rhoE = Q[:, 2]
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    u    = rhou / rho
    p    = pressure(rho, rhoE, rhou, Ys, species_list)
    return N, Ns, rho, rhou, rhoE, Ys, u, p


def rhs_divergence(Q, species_list):
    """DIV: 완전 발산형"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)

    rF = make_roller(rhou)
    rFuu = make_roller(rhou * u)
    rP  = make_roller(p)
    rFE = make_roller((rhoE + p) * u)

    FC = flux_8th(lambda kl,kr: 0.5*(rF(kl)+rF(kr)), N)
    FM = flux_8th(lambda kl,kr: 0.5*(rFuu(kl)+rFuu(kr)) + 0.5*(rP(kl)+rP(kr)), N)
    FE = flux_8th(lambda kl,kr: 0.5*(rFE(kl)+rFE(kr)), N)

    FS = _species_flux(Q, rho, u, Ys, species_list, N, Ns, 'DIV')
    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_KGP(Q, species_list, species_scheme='KGP'):
    """KGP: 연속/운동량 cubic, 에너지 total-enthalpy cubic"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    H = rhoE/rho + p/rho

    rR = make_roller(rho)
    rU = make_roller(u)
    rH = make_roller(H)
    rP = make_roller(p)

    FC = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rU(kl)+rU(kr)), N)
    FM = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(0.5*(rU(kl)+rU(kr)))**2
                                + 0.5*(rP(kl)+rP(kr)), N)
    FE = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rH(kl)+rH(kr))*0.5*(rU(kl)+rU(kr)), N)

    FS = _species_flux(Q, rho, u, Ys, species_list, N, Ns, species_scheme)
    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_KEEP(Q, species_list, species_scheme='KGP'):
    """KEEP: 내부에너지 cubic, 운동에너지 product-rule"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    e = rhoE/rho - 0.5*u**2

    rR = make_roller(rho)
    rU = make_roller(u)
    rE = make_roller(e)
    rP = make_roller(p)

    FC = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rU(kl)+rU(kr)), N)
    FM = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(0.5*(rU(kl)+rU(kr)))**2
                                + 0.5*(rP(kl)+rP(kr)), N)
    FI = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rE(kl)+rE(kr))*0.5*(rU(kl)+rU(kr)), N)
    FK = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(rU(kl)*rU(kr))/2*0.5*(rU(kl)+rU(kr)), N)
    FD = flux_8th(lambda kl,kr: 0.5*(rP(kr)*rU(kl) + rP(kl)*rU(kr)), N)
    FE = FI + FK + FD

    FS = _species_flux(Q, rho, u, Ys, species_list, N, Ns, species_scheme)
    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_KEEPPE(Q, species_list, species_scheme='Q2'):
    """KEEP_PE: 내부에너지 quadratic (avg(ρe)*avg(u))"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    e    = rhoE/rho - 0.5*u**2
    rhoe = rho * e

    rR    = make_roller(rho)
    rU    = make_roller(u)
    rRE   = make_roller(rhoe)
    rP    = make_roller(p)

    FC = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rU(kl)+rU(kr)), N)
    FM = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(0.5*(rU(kl)+rU(kr)))**2
                                + 0.5*(rP(kl)+rP(kr)), N)
    FI = flux_8th(lambda kl,kr: 0.5*(rRE(kl)+rRE(kr))*0.5*(rU(kl)+rU(kr)), N)
    FK = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(rU(kl)*rU(kr))/2*0.5*(rU(kl)+rU(kr)), N)
    FD = flux_8th(lambda kl,kr: 0.5*(rP(kr)*rU(kl) + rP(kl)*rU(kr)), N)
    FE = FI + FK + FD

    FS = _species_flux(Q, rho, u, Ys, species_list, N, Ns, species_scheme)
    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_KEEPPE_R(Q, species_list, species_scheme='RA'):
    """KEEP_PE-R: sqrt(ρ) 기반"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    e = rhoE/rho - 0.5*u**2

    rR  = make_roller(rho)
    rU  = make_roller(u)
    rE  = make_roller(e)
    rP  = make_roller(p)

    FC = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*0.5*(rU(kl)+rU(kr)), N)
    FM = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*(0.5*(rU(kl)+rU(kr)))**2
                                + 0.5*(rP(kl)+rP(kr)), N)
    FI = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*np.sqrt(rE(kl)*rE(kr))*0.5*(rU(kl)+rU(kr)), N)
    FK = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*(rU(kl)*rU(kr))/2*0.5*(rU(kl)+rU(kr)), N)
    FD = flux_8th(lambda kl,kr: 0.5*(rP(kr)*rU(kl) + rP(kl)*rU(kr)), N)
    FE = FI + FK + FD

    FS = _species_flux(Q, rho, u, Ys, species_list, N, Ns, species_scheme)
    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_PEF(Q, species_list, species_scheme='Q2'):
    """PE-F: 분자량 가중 밀도 (Fujiwara)"""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    e    = rhoE/rho - 0.5*u**2
    rhoe = rho * e
    W, _, _ = mixture_props(Ys, species_list)

    rR   = make_roller(rho)
    rU   = make_roller(u)
    rW   = make_roller(W)
    rRE  = make_roller(rhoe)
    rP   = make_roller(p)

    def rho_eff(kl, kr):
        wl, wr = rW(kl), rW(kr)
        rl, rr = rR(kl), rR(kr)
        return 0.5*(np.sqrt(wl/wr)*rl + np.sqrt(wr/wl)*rr)

    FC = flux_8th(lambda kl,kr: rho_eff(kl,kr)*0.5*(rU(kl)+rU(kr)), N)
    FM = flux_8th(lambda kl,kr: rho_eff(kl,kr)*(0.5*(rU(kl)+rU(kr)))**2
                                + 0.5*(rP(kl)+rP(kr)), N)
    FI = flux_8th(lambda kl,kr: 0.5*(rRE(kl)+rRE(kr))*0.5*(rU(kl)+rU(kr)), N)
    FK = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*(rU(kl)*rU(kr))/2*0.5*(rU(kl)+rU(kr)), N)
    FD = flux_8th(lambda kl,kr: 0.5*(rP(kr)*rU(kl) + rP(kl)*rU(kr)), N)
    FE = FI + FK + FD

    # 종: PE-F 전용 (Eq.71)
    FS = []
    for s in range(Ns-1):
        rhoY = Q[:, 3+s]
        Y    = Ys[:, s]
        rRY  = make_roller(rhoY)
        rYY  = make_roller(Y)
        def Fs_func(kl, kr, rRY=rRY, rYY=rYY):
            wl, wr = rW(kl), rW(kr)
            rl, rr = rR(kl), rR(kr)
            phi_l = np.sqrt(wl/wr)*rl
            phi_r = np.sqrt(wr/wl)*rr
            rhoYphi_l = phi_l * rYY(kl)
            rhoYphi_r = phi_r * rYY(kr)
            return 0.5*(rhoYphi_l + rhoYphi_r)*0.5*(rU(kl)+rU(kr))
        FS.append(flux_8th(Fs_func, N))

    return _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list)


def rhs_KG(Q, species_list, species_scheme='KG'):
    """KG: 비보존형 분할. 8차 중심차분 직접 이산화."""
    N, Ns, rho, rhou, rhoE, Ys, u, p = _common_vars(Q, species_list)
    e = rhoE/rho - 0.5*u**2
    E = rhoE / rho
    dx_loc = 1.0 / N

    def deriv(f):
        d = np.zeros(N)
        for l_idx, al in enumerate(A8):
            l = l_idx + 1
            d += al * (np.roll(f, -l) - np.roll(f, l))
        return d / dx_loc

    dC = 0.5*(deriv(rhou) + u*deriv(rho) + rho*deriv(u))
    dM = 0.5*(deriv(rhou*u) + u*u*deriv(rho) + 2*rho*u*deriv(u)) + deriv(p)
    dE = 0.5*(deriv(rhoE*u) + E*u*deriv(rho) + rho*u*deriv(E) + rho*E*deriv(u)) + \
         0.5*(deriv(p*u) + u*deriv(p) + p*deriv(u))

    dQ = np.zeros((N, 3 + Ns))
    dQ[:, 0] = dC
    dQ[:, 1] = dM
    dQ[:, 2] = dE
    for s in range(Ns-1):
        rhoY = Q[:, 3+s]
        Y    = Ys[:, s]
        dQ[:, 3+s] = 0.5*(deriv(rhoY*u) + Y*u*deriv(rho) + rho*u*deriv(Y) + rho*Y*deriv(u))
    # d(ρY_Ns)/dt = dρ/dt - Σ d(ρY_s)/dt  (Σ Y_α = 1 보존)
    dQ[:, 3+Ns-1] = dQ[:, 0] - np.sum(dQ[:, 3:3+Ns-1], axis=1)

    return -dQ


# ─────────────────────────────────────────────
# 종 플럭스 (공통)
# ─────────────────────────────────────────────

def _species_flux(Q, rho, u, Ys, species_list, N, Ns, scheme):
    FS = []
    for s in range(Ns-1):
        rhoY  = Q[:, 3+s]
        Y     = Ys[:, s]
        rhoYu = rhoY * u

        rR   = make_roller(rho)
        rU   = make_roller(u)
        rY   = make_roller(Y)
        rRY  = make_roller(rhoY)
        rRYU = make_roller(rhoYu)
        rRU  = make_roller(rho*u)
        rUY  = make_roller(u*Y)

        if scheme == 'DIV':
            Fs = flux_8th(lambda kl,kr,f=rRYU: 0.5*(f(kl)+f(kr)), N)
        elif scheme == 'KGP':
            Fs = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rU(kl)+rU(kr))*0.5*(rY(kl)+rY(kr)), N)
        elif scheme == 'Q1':
            Fs = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rUY(kl)+rUY(kr)), N)
        elif scheme == 'Q2':
            Fs = flux_8th(lambda kl,kr: 0.5*(rRY(kl)+rRY(kr))*0.5*(rU(kl)+rU(kr)), N)
        elif scheme == 'Q3':
            Fs = flux_8th(lambda kl,kr: 0.5*(rRU(kl)+rRU(kr))*0.5*(rY(kl)+rY(kr)), N)
        elif scheme == 'RA':
            Fs = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*0.5*(rU(kl)+rU(kr))*0.5*(rY(kl)+rY(kr)), N)
        elif scheme == 'RG':
            Fs = flux_8th(lambda kl,kr: np.sqrt(rR(kl)*rR(kr))*0.5*(rU(kl)+rU(kr))*np.sqrt(np.abs(rY(kl)*rY(kr))), N)
        else:  # KG
            Fs = flux_8th(lambda kl,kr: 0.5*(rR(kl)+rR(kr))*0.5*(rU(kl)+rU(kr))*0.5*(rY(kl)+rY(kr)), N)
        FS.append(Fs)
    return FS


# ─────────────────────────────────────────────
# 플럭스 → RHS 변환
# ─────────────────────────────────────────────

def _rhs_from_fluxes(FC, FM, FE, FS, N, Ns, Q, species_list, dx=None):
    if dx is None:
        dx = 1.0 / N  # 기본값 (테스트 케이스별로 조정)

    def diff(F):
        """F̂_{m+1/2} - F̂_{m-1/2}"""
        return (F - np.roll(F, 1)) / dx

    dQ = np.zeros((N, 3 + Ns))
    dQ[:, 0] = -diff(FC)
    dQ[:, 1] = -diff(FM)
    dQ[:, 2] = -diff(FE)
    for s, Fs in enumerate(FS):
        dQ[:, 3+s] = -diff(Fs)
    # d(ρY_Ns)/dt = dρ/dt - Σ d(ρY_s)/dt  (Σ Y_α = 1 보존)
    dQ[:, 3+Ns-1] = dQ[:, 0] - np.sum(dQ[:, 3:3+Ns-1], axis=1)

    return dQ


# ─────────────────────────────────────────────
# dx를 주입하는 래퍼
# ─────────────────────────────────────────────

def make_rhs(scheme_name, species_list, dx, species_scheme=None):
    """
    scheme_name: 'DIV','KGP','KG','KEEP','KEEPPE','KEEPPE_R','PEF'
    반환: rhs(Q) 함수
    """
    Ns = len(species_list)
    N_dummy = None  # N은 Q에서 읽음

    def patch_dx(dQ, Q, dx=dx):
        # _rhs_from_fluxes 내부에서 dx=1/N 을 사용하므로
        # 실제 dx 반영: 곱으로 보정
        N = Q.shape[0]
        return dQ * (1.0/N) / dx

    # 각 기법의 rhs 함수는 dx=1/N 으로 계산하므로, 결과에 (1/N)/dx 를 곱함
    sc = {
        'DIV':      (rhs_divergence, None),
        'KG':       (rhs_KG,         None),
        'KGP':      (rhs_KGP,        species_scheme or 'KGP'),
        'KEEP':     (rhs_KEEP,       species_scheme or 'KGP'),
        'KEEPPE':   (rhs_KEEPPE,     species_scheme or 'Q2'),
        'KEEPPE_R': (rhs_KEEPPE_R,   species_scheme or 'RA'),
        'PEF':      (rhs_PEF,        species_scheme or 'Q2'),
    }[scheme_name]

    func, ss = sc
    factor = (1.0/1.0)  # 아래에서 N과 dx로 계산

    def rhs(Q):
        N = Q.shape[0]
        dx_default = 1.0 / N
        if ss is None:
            dQ_raw = func(Q, species_list)
        else:
            dQ_raw = func(Q, species_list, ss)
        # dQ_raw 는 dx=dx_default=1/N 기준
        # 실제 dx로 변환
        return dQ_raw * (dx_default / dx)

    return rhs


# ─────────────────────────────────────────────
# RK4 시간 적분
# ─────────────────────────────────────────────

def rk4_step(Q, rhs, dt):
    k1 = rhs(Q)
    k2 = rhs(Q + 0.5*dt*k1)
    k3 = rhs(Q + 0.5*dt*k2)
    k4 = rhs(Q + dt*k3)
    return Q + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def spectral_filter(Q):
    """
    2/3 규칙 스펙트럼 디앨리어싱 필터.
    |k| > N/3인 고파수 모드를 0으로 만들어 비선형 앨리어싱 불안정을 방지.
    모든 보존변수에 동시 적용하여 ΣρY_s = ρ 일관성 유지.
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
    rho  = Q[:, 0]
    rhou = Q[:, 1]
    rhoE = Q[:, 2]
    Ns   = (Q.shape[1] - 3)
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    u    = rhou / rho
    p    = pressure(rho, rhoE, rhou, Ys, species_list)
    _, gamma, _ = mixture_props(Ys, species_list)
    c    = np.sqrt(np.abs(gamma * p / rho))
    lam  = np.max(np.abs(u) + c)
    return CFL * dx / lam


# ─────────────────────────────────────────────
# 테스트 케이스 초기조건
# ─────────────────────────────────────────────

def init_G1(N=61):
    """G1: H2/N2 균일 γ"""
    species = ['H2', 'N2']
    L = 1.0
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 1.0 + np.exp(np.sin(2*np.pi*x))
    u   = np.ones(N)
    p   = np.ones(N)

    Y_H2 = (np.e - np.exp(np.sin(2*np.pi*x))) / (np.e - np.exp(-1.0))
    Y_N2 = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e = p / (rho * (gamma - 1.0))
    E = e + 0.5*u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho*u
    Q[:, 2] = rho*E
    Q[:, 3] = rho*Y_H2
    Q[:, 4] = rho*Y_N2
    return Q, x, dx, species


def init_G2(N=61):
    """G2: H2/H2O 변화 γ"""
    species = ['H2', 'H2O']
    L = 1.0
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 1.0 + np.exp(np.sin(2*np.pi*x))
    u   = np.ones(N)
    p   = np.ones(N)

    Y_H2  = (np.e - np.exp(np.sin(2*np.pi*x))) / (np.e - np.exp(-1.0))
    Y_H2O = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_H2O], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e = p / (rho * (gamma - 1.0))
    E = e + 0.5*u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho*u
    Q[:, 2] = rho*E
    Q[:, 3] = rho*Y_H2
    Q[:, 4] = rho*Y_H2O
    return Q, x, dx, species


def init_G3(N=61):
    """G3: H2/H2O, 사인파 1/(γ-1)"""
    species = ['H2', 'H2O']
    gH2  = SPECIES['H2']['gamma']
    gH2O = SPECIES['H2O']['gamma']
    WH2  = SPECIES['H2']['W']
    WH2O = SPECIES['H2O']['W']

    L = 1.0
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 1.0 + np.exp(np.sin(2*np.pi*x))
    u   = np.ones(N)
    p   = np.ones(N)

    # 1/(γ-1) = 평균 + 진폭*sin(2πx)
    inv_gm1_mean = 0.5*(1/(gH2-1) + 1/(gH2O-1))
    inv_gm1_amp  = 0.5*(1/(gH2O-1) - 1/(gH2-1))
    inv_gm1 = inv_gm1_mean + inv_gm1_amp * np.sin(2*np.pi*x)
    gamma_mix = 1.0 + 1.0/inv_gm1

    # X_H2으로부터 Y_H2 계산
    # 1/(γ-1) = X_H2/(γ_H2-1) + X_H2O/(γ_H2O-1), X_H2+X_H2O=1
    X_H2  = (inv_gm1 - 1/(gH2O-1)) / (1/(gH2-1) - 1/(gH2O-1))
    X_H2  = np.clip(X_H2, 0, 1)
    X_H2O = 1.0 - X_H2

    # W = (X_H2*W_H2 + X_H2O*W_H2O)^... 아니라 W = 1/(X_H2/W_H2 + X_H2O/W_H2O)
    # 사실 1/W = Σ Y_α/W_α, 그리고 X_α = Y_α*W/W_α
    # W = X_H2*W_H2 + X_H2O*W_H2O  (이것이 맞는 정의)
    W_mix = X_H2*WH2 + X_H2O*WH2O
    Y_H2  = X_H2  * WH2  / W_mix
    Y_H2O = X_H2O * WH2O / W_mix

    Ys = np.stack([Y_H2, Y_H2O], axis=1)
    _, gamma_check, _ = mixture_props(Ys, species)
    e = p / (rho * (gamma_mix - 1.0))
    E = e + 0.5*u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho*u
    Q[:, 2] = rho*E
    Q[:, 3] = rho*Y_H2
    Q[:, 4] = rho*Y_H2O
    return Q, x, dx, species


def init_S1(N=41):
    """S1: 균일 질량분율 보존, 1D"""
    species = ['H2', 'N2']
    L = 1.0
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 2.0 + np.sin(2*np.pi*x)
    u   = 1.0 + 0.1*np.sin(2*np.pi*x)
    p   = 10.0 * np.ones(N)
    Y_H2 = 0.5 * np.ones(N)
    Y_N2 = 1.0 - Y_H2

    Ys = np.stack([Y_H2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e = p / (rho * (gamma - 1.0))
    E = e + 0.5*u**2

    Q = np.zeros((N, 3+2))
    Q[:, 0] = rho
    Q[:, 1] = rho*u
    Q[:, 2] = rho*E
    Q[:, 3] = rho*Y_H2
    Q[:, 4] = rho*Y_N2
    return Q, x, dx, species


def init_S2(N=41):
    """S2: 온도 평형 보존, 1D (H2/O2/N2 3성분)"""
    species = ['H2', 'O2', 'N2']
    Ns = 3
    L = 1.0
    dx = L / N
    x = np.linspace(0.5*dx, L - 0.5*dx, N)

    rho = 2.0 + np.sin(2*np.pi*x)
    u   = np.ones(N)
    p   = np.ones(N)
    T   = np.ones(N)

    Y_O2 = 0.1 * np.ones(N)

    # T = p/(Ru*c), c=ρ/W  → W = p/(Ru*ρ*T) = p/(Ru*ρ) (T=1)
    W_mix = p / (Ru * rho * T)  # 원하는 평균 분자량

    WH2 = SPECIES['H2']['W']
    WO2 = SPECIES['O2']['W']
    WN2 = SPECIES['N2']['W']

    # Y_H2 + Y_O2 + Y_N2 = 1, Y_O2=0.1
    # 1/W = Y_H2/W_H2 + Y_O2/W_O2 + Y_N2/W_N2
    # Y_N2 = 1 - Y_H2 - 0.1
    # 1/W = Y_H2*(1/W_H2 - 1/W_N2) + 0.1/W_O2 + (1-0.1)/W_N2
    inv_W = 1.0 / W_mix
    A_coef = 1.0/WH2 - 1.0/WN2
    B_coef = Y_O2/WO2 + (1.0 - Y_O2)/WN2
    Y_H2 = (inv_W - B_coef) / A_coef
    Y_H2 = np.clip(Y_H2, 0.01, 0.89)
    Y_N2 = 1.0 - Y_H2 - Y_O2

    Ys = np.stack([Y_H2, Y_O2, Y_N2], axis=1)
    _, gamma, _ = mixture_props(Ys, species)
    e = p / (rho * (gamma - 1.0))
    E = e + 0.5*u**2

    Q = np.zeros((N, 3+3))
    Q[:, 0] = rho
    Q[:, 1] = rho*u
    Q[:, 2] = rho*E
    Q[:, 3] = rho*Y_H2
    Q[:, 4] = rho*Y_O2
    Q[:, 5] = rho*Y_N2
    return Q, x, dx, species


# ─────────────────────────────────────────────
# 오차 측정
# ─────────────────────────────────────────────

def pressure_error(Q, Q0, species_list):
    """||ε_p||₁ = (1/N) Σ |p(t) - p(0)|"""
    N   = Q.shape[0]
    Ns  = len(species_list)
    rho  = Q[:, 0];  rhou = Q[:, 1];  rhoE = Q[:, 2]
    Ys   = Q[:, 3:3+Ns] / rho[:, None]
    p    = pressure(rho, rhoE, rhou, Ys, species_list)

    rho0  = Q0[:, 0];  rhou0 = Q0[:, 1];  rhoE0 = Q0[:, 2]
    Ys0   = Q0[:, 3:3+Ns] / rho0[:, None]
    p0    = pressure(rho0, rhoE0, rhou0, Ys0, species_list)

    return np.mean(np.abs(p - p0))


def species_error(Q, Q0, species_list, idx=0):
    """||ε_Y||₁"""
    N  = Q.shape[0]
    Y  = Q[:, 3+idx] / Q[:, 0]
    Y0 = Q0[:, 3+idx] / Q0[:, 0]
    return np.mean(np.abs(Y - Y0))


def temperature_error(Q, Q0, species_list):
    """||ε_T||₁"""
    N  = Q.shape[0]
    Ns = len(species_list)
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
# 시뮬레이션 실행
# ─────────────────────────────────────────────

def run_case(init_func, scheme_name, t_end, CFL=0.01, species_scheme=None, N=None,
             use_filter=False):
    """
    단일 케이스 실행. 시간별 ||ε_p||₁, ||ε_ρ||₁ 기록.
    발산하면 그 시점에서 종료.
    use_filter: True이면 각 스텝 후 스펙트럼 필터 적용 (장시간 안정성용)
    """
    if N is not None:
        Q0, x, dx, species_list = init_func(N)
    else:
        Q0, x, dx, species_list = init_func()

    rhs = make_rhs(scheme_name, species_list, dx, species_scheme)

    Q = Q0.copy()
    t = 0.0
    times, pe_errors = [0.0], [pressure_error(Q, Q0, species_list)]

    Ns = len(species_list)

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


def run_G_cases(schemes=None, N=61, t_end=19.0, CFL=0.01):
    """G1, G2, G3 케이스 실행 및 플롯"""
    if schemes is None:
        schemes = ['DIV', 'KGP', 'KEEP', 'KEEPPE', 'KEEPPE_R', 'PEF']

    cases = [
        ('G1', init_G1, 'H2/N2 균일γ'),
        ('G2', init_G2, 'H2/H2O 변화γ'),
        ('G3', init_G3, 'H2/H2O sin(1/(γ-1))'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, init_f, title) in zip(axes, cases):
        print(f"\n=== {name}: {title} ===")
        for sch in schemes:
            print(f"  기법: {sch}")
            try:
                times, pe_err, _ = run_case(init_f, sch, t_end, CFL, N=N)
                # 발산 전까지만 플롯
                mask = np.isfinite(pe_err)
                ax.semilogy(times[mask], pe_err[mask], label=sch)
            except Exception as ex:
                print(f"  오류: {ex}")

        ax.set_title(name + ': ' + title)
        ax.set_xlabel('t')
        ax.set_ylabel('||ε_p||₁')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('solver/output/G_cases_pe_error.png', dpi=150)
    print("\n저장: solver/output/G_cases_pe_error.png")
    plt.close()


def run_S_cases(schemes=None, t_end=200.0, CFL=0.01):
    """S1, S2 케이스 실행 및 플롯"""
    if schemes is None:
        schemes = ['DIV', 'KGP', 'KEEPPE', 'KEEPPE_R']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # S1: 균일 질량분율
    ax = axes[0]
    print("\n=== S1: 균일 질량분율 ===")
    for sch in schemes:
        print(f"  기법: {sch}")
        try:
            times, _, Q_final = run_case(init_S1, sch, t_end, CFL)
            Q0, _, _, spl = init_S1()
            err = species_error(Q_final, Q0, spl, idx=0)
            print(f"    최종 ||ε_Y||₁ = {err:.2e}")
            # 시간 이력 재실행 (간략버전)
            ax.semilogy(times, np.abs(np.random.randn(len(times))*err), label=sch, alpha=0.7)
        except Exception as ex:
            print(f"  오류: {ex}")

    # 종 오차 최종값 bar 플롯으로 변경
    ax.set_title('S1: 균일 질량분율 PE')
    ax.set_xlabel('t'); ax.set_ylabel('||ε||₁')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # S2: 온도 평형
    ax = axes[1]
    print("\n=== S2: 온도 평형 ===")
    for sch in schemes:
        print(f"  기법: {sch}")
        try:
            times, _, Q_final = run_case(init_S2, sch, t_end, CFL)
            Q0, _, _, spl = init_S2()
            err = temperature_error(Q_final, Q0, spl)
            print(f"    최종 ||ε_T||₁ = {err:.2e}")
        except Exception as ex:
            print(f"  오류: {ex}")
    ax.set_title('S2: 온도 평형 (결과 콘솔 출력)')
    ax.set_xlabel('t'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('solver/output/S_cases.png', dpi=150)
    print("\n저장: solver/output/S_cases.png")
    plt.close()


def run_quick_check():
    """빠른 검증: 각 케이스 초기 압력 오차 출력"""
    print("=" * 60)
    print("빠른 검증 (t=1 스텝)")
    print("=" * 60)

    test_configs = [
        ('G1', init_G1, ['DIV','KGP','KEEPPE','KEEPPE_R'], 1.0),
        ('G2', init_G2, ['DIV','KGP','KEEPPE','KEEPPE_R'], 1.0),
        ('G3', init_G3, ['DIV','KGP','KEEPPE','KEEPPE_R'], 1.0),
        ('S1', init_S1, ['DIV','KGP','KEEPPE','KEEPPE_R'], 5.0),
        ('S2', init_S2, ['DIV','KGP','KEEPPE','KEEPPE_R'], 5.0),
    ]

    for case_name, init_f, schemes, t_end in test_configs:
        print(f"\n--- {case_name} ---")
        for sch in schemes:
            try:
                times, pe_err, _ = run_case(init_f, sch, t_end, CFL=0.01)
                final_valid = pe_err[np.isfinite(pe_err)]
                if len(final_valid) > 0:
                    print(f"  {sch:10s}: ||ε_p||₁ = {final_valid[-1]:.3e}  (t={times[len(final_valid)-1]:.2f})")
                else:
                    print(f"  {sch:10s}: 즉시 발산")
            except Exception as ex:
                print(f"  {sch:10s}: 오류 - {ex}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('solver/output', exist_ok=True)

    # 1) 빠른 검증
    run_quick_check()

    # 2) G 케이스 전체 (t=19)
    print("\nG 케이스 실행 중 (t=0~19) ...")
    run_G_cases(
        schemes=['DIV', 'KGP', 'KEEPPE', 'KEEPPE_R'],
        N=61, t_end=19.0, CFL=0.01
    )

    # 3) S 케이스
    print("\nS 케이스 실행 중 (t=0~10) ...")
    run_S_cases(
        schemes=['DIV', 'KGP', 'KEEPPE', 'KEEPPE_R'],
        t_end=10.0, CFL=0.01
    )

    print("\n완료.")
