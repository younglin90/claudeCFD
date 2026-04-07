"""
cpg_euler_1d.py
================
범용 1D 다성분 Calorically Perfect Gas (CPG) Euler FVM 솔버.

지원:
  - Ns >= 1 성분 CPG 혼합물 (성분별 gamma_i, W_i 지정)
  - 2차 MUSCL + minmod 재구성
  - LLF (Rusanov) Riemann solver
  - 경계조건: transmissive / periodic / wall / wall_left
  - SSP-RK3 시간 적분

보존변수:
  Q[m] = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{Ns-1}]
  마지막 종: rho*Y_Ns = rho - sum(rho*Y_s for s < Ns)
"""
import numpy as np

Ru = 8.314  # J/(mol·K)

# ─────────────────────────────────────────────
# 성분 물성 테이블
# ─────────────────────────────────────────────
SPECIES_LIB = {
    'H2':    {'gamma': 1.40, 'W': 2.016e-3},
    'N2':    {'gamma': 1.40, 'W': 28.014e-3},
    'H2O':   {'gamma': 1.33, 'W': 18.015e-3},
    'O2':    {'gamma': 1.40, 'W': 31.999e-3},
    'He':    {'gamma': 1.67, 'W': 4.003e-3},
    'Air':   {'gamma': 1.40, 'W': 29.0e-3},
    'G14':   {'gamma': 1.40, 'W': 1.0},   # 일반 gamma=1.4 기체
    'G167':  {'gamma': 1.67, 'W': 1.0},   # 일반 gamma=1.67 기체
    'G16':   {'gamma': 1.60, 'W': 1.0},   # 일반 gamma=1.6 기체
}


def make_gammas_Ws(names_or_gammas, Ws=None):
    """
    성분 물성 배열 반환.
    입력:
      names_or_gammas : list[str]  (SPECIES_LIB 이름)
                      or list[float] (gamma 직접 지정 시 Ws도 필요)
    반환: gs (Ns,), ws (Ns,)
    """
    if isinstance(names_or_gammas[0], str):
        gs = np.array([SPECIES_LIB[n]['gamma'] for n in names_or_gammas])
        ws = np.array([SPECIES_LIB[n]['W']     for n in names_or_gammas])
    else:
        gs = np.asarray(names_or_gammas, float)
        ws = np.ones_like(gs) if Ws is None else np.asarray(Ws, float)
    return gs, ws


# ─────────────────────────────────────────────
# EOS 헬퍼
# ─────────────────────────────────────────────

def gamma_mix_vec(Ys, gs, ws):
    """
    혼합 gamma 계산.
    Ys : (..., Ns)  질량분율
    gs, ws : (Ns,)
    반환: (...,) 혼합 gamma
    """
    inv_W  = np.dot(Ys, 1.0 / ws)            # (...,)
    W_mix  = 1.0 / inv_W
    Xs     = Ys / ws * W_mix[..., None]       # 몰분율 (..., Ns)
    inv_gm1 = np.dot(Xs, 1.0 / (gs - 1.0))   # (...,)
    return 1.0 + 1.0 / inv_gm1


def _get_rhoY_full(Q, Ns):
    """보존변수에서 Ns 종 rho*Y 행렬 복원."""
    rho  = Q[..., 0]
    rhoY = Q[..., 3:3+Ns]
    if rhoY.shape[-1] == Ns - 1:
        last = rho - np.sum(rhoY, axis=-1, keepdims=True)
        rhoY = np.concatenate([rhoY, last], axis=-1)
    return rhoY


def pressure_from_Q(Q, gs, ws):
    """
    보존변수 Q -> 압력.
    Q : (..., 3+Ns)  (Ns = len(gs))
    """
    Ns   = len(gs)
    rho  = Q[..., 0]
    rhou = Q[..., 1]
    rhoE = Q[..., 2]
    rhoY = _get_rhoY_full(Q, Ns)
    Ys   = rhoY / rho[..., None]
    u    = rhou / rho
    e    = rhoE / rho - 0.5 * u**2
    gm   = gamma_mix_vec(Ys, gs, ws)
    return rho * e * (gm - 1.0)


def sound_speed_from_Q(Q, gs, ws):
    """Q -> 음속."""
    Ns   = len(gs)
    rho  = Q[..., 0]
    rhoY = _get_rhoY_full(Q, Ns)
    Ys   = rhoY / rho[..., None]
    gm   = gamma_mix_vec(Ys, gs, ws)
    p    = pressure_from_Q(Q, gs, ws)
    return np.sqrt(np.maximum(gm * p / rho, 0.0))


def temperature_from_Q(Q, gs, ws):
    """Q -> 온도 (이상기체: T = p*W_mix/(Ru*rho))."""
    Ns   = len(gs)
    rho  = Q[..., 0]
    rhoY = _get_rhoY_full(Q, Ns)
    Ys   = rhoY / rho[..., None]
    inv_W = np.dot(Ys, 1.0 / ws)
    W_mix = 1.0 / inv_W
    p    = pressure_from_Q(Q, gs, ws)
    return p * W_mix / (Ru * rho)


# ─────────────────────────────────────────────
# 물리 플럭스
# ─────────────────────────────────────────────

def physical_flux(Q, gs, ws):
    """
    F(Q) = [rho*u, rho*u^2+p, (rhoE+p)*u, rho*Y_s*u] for s < Ns
    Q : (N, 3+Ns)
    """
    Ns   = len(gs)
    rho  = Q[:, 0]
    rhou = Q[:, 1]
    rhoE = Q[:, 2]
    rhoY = _get_rhoY_full(Q, Ns)
    u    = rhou / rho
    p    = pressure_from_Q(Q, gs, ws)

    F = np.zeros_like(Q)
    F[:, 0] = rhou
    F[:, 1] = rhou * u + p
    F[:, 2] = (rhoE + p) * u
    for s in range(Ns - 1):
        F[:, 3+s] = rhoY[:, s] * u
    return F


# ─────────────────────────────────────────────
# MUSCL 재구성 + LLF 플럭스
# ─────────────────────────────────────────────

def minmod(a, b):
    return np.where(a * b > 0.0,
                    np.where(np.abs(a) < np.abs(b), a, b),
                    0.0)


def _get_ghosts(Q, bc):
    """
    ghost 셀 2개씩 추가.
    bc = 'periodic' | 'transmissive' | 'wall' | 'wall_left'
    반환: Q_ext (N+4, Nv)
    """
    N, Nv = Q.shape
    Q_ext = np.zeros((N + 4, Nv))
    Q_ext[2:N+2] = Q

    def reflect(q):
        r = q.copy()
        r[1] = -r[1]
        return r

    if bc == 'periodic':
        Q_ext[0]    = Q[-2]
        Q_ext[1]    = Q[-1]
        Q_ext[N+2]  = Q[0]
        Q_ext[N+3]  = Q[1]
    elif bc == 'transmissive':
        Q_ext[0]    = Q[0]
        Q_ext[1]    = Q[0]
        Q_ext[N+2]  = Q[-1]
        Q_ext[N+3]  = Q[-1]
    elif bc == 'wall':
        Q_ext[0]    = reflect(Q[1])
        Q_ext[1]    = reflect(Q[0])
        Q_ext[N+2]  = reflect(Q[-1])
        Q_ext[N+3]  = reflect(Q[-2])
    elif bc == 'wall_left':
        # 좌측 wall, 우측 transmissive
        Q_ext[0]    = reflect(Q[1])
        Q_ext[1]    = reflect(Q[0])
        Q_ext[N+2]  = Q[-1]
        Q_ext[N+3]  = Q[-1]
    else:
        raise ValueError(f"Unknown bc: {bc}")
    return Q_ext


def muscl_faces(Q, bc='transmissive'):
    """
    MUSCL + minmod.
    반환: QL (N+1, Nv), QR (N+1, Nv)
      face f=0 : left boundary
      face f=N : right boundary
    """
    N, Nv = Q.shape
    Qe = _get_ghosts(Q, bc)   # (N+4, Nv)
    # 셀 m (0-based) -> Qe[m+2]
    # face f: between cell f-1 and cell f  (f=0..N)
    # QL[f] = Qe[f+1] + 0.5*minmod(Qe[f+1]-Qe[f], Qe[f+2]-Qe[f+1])
    # QR[f] = Qe[f+2] - 0.5*minmod(Qe[f+2]-Qe[f+1], Qe[f+3]-Qe[f+2])
    QL = np.zeros((N+1, Nv))
    QR = np.zeros((N+1, Nv))
    for f in range(N+1):
        iL = f + 1   # Qe index: cell (f-1)
        iR = f + 2   # Qe index: cell (f)
        dL = minmod(Qe[iL] - Qe[iL-1], Qe[iL+1] - Qe[iL])
        dR = minmod(Qe[iR] - Qe[iR-1], Qe[iR+1] - Qe[iR])
        QL[f] = Qe[iL] + 0.5 * dL
        QR[f] = Qe[iR] - 0.5 * dR
    return QL, QR


def llf_face_flux(QL, QR, gs, ws):
    """
    LLF (Rusanov) face 플럭스.
    QL, QR : (Nf, Nv)
    반환: F (Nf, Nv)
    """
    uL  = QL[:, 1] / QL[:, 0]
    uR  = QR[:, 1] / QR[:, 0]
    cL  = sound_speed_from_Q(QL, gs, ws)
    cR  = sound_speed_from_Q(QR, gs, ws)
    lam = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    FL  = physical_flux(QL, gs, ws)
    FR  = physical_flux(QR, gs, ws)
    return 0.5 * (FL + FR) - 0.5 * lam[:, None] * (QR - QL)


# ─────────────────────────────────────────────
# RHS
# ─────────────────────────────────────────────

def rhs_euler(Q, gs, ws, dx, bc='transmissive'):
    """FVM RHS: dQ/dt = -1/dx * (F_{m+1/2} - F_{m-1/2})"""
    QL, QR = muscl_faces(Q, bc)
    F      = llf_face_flux(QL, QR, gs, ws)
    return -(F[1:] - F[:-1]) / dx


def ssprk3(Q, rhs_fn, dt):
    """SSP-RK3."""
    k1 = rhs_fn(Q)
    Q1 = Q + dt * k1
    k2 = rhs_fn(Q1)
    Q2 = 0.75 * Q + 0.25 * (Q1 + dt * k2)
    k3 = rhs_fn(Q2)
    return Q / 3.0 + 2.0 / 3.0 * (Q2 + dt * k3)


def cfl_dt(Q, gs, ws, dx, CFL):
    u   = Q[:, 1] / Q[:, 0]
    c   = sound_speed_from_Q(Q, gs, ws)
    lam = np.max(np.abs(u) + c)
    if lam < 1e-30:
        return 1e10
    return CFL * dx / lam


# ─────────────────────────────────────────────
# 시뮬레이션 실행
# ─────────────────────────────────────────────

def run_euler(Q0, gs, ws, dx, CFL, T_end, bc='transmissive',
              save_interval=None, max_steps=int(5e6)):
    """
    시뮬레이션 실행.
    save_interval : 저장 간격 (None이면 처음/끝만 저장)
    반환: Q_final (N, Nv),  snapshots [(t, Q), ...]
    """
    Q = Q0.copy()
    t = 0.0
    snapshots = [(0.0, Q0.copy())]
    t_next_save = save_interval if save_interval else T_end

    def rhs_fn(q):
        return rhs_euler(q, gs, ws, dx, bc)

    for step in range(max_steps):
        if t >= T_end:
            break
        dt = min(cfl_dt(Q, gs, ws, dx, CFL), T_end - t)
        if dt <= 1e-15:
            break
        Q  = ssprk3(Q, rhs_fn, dt)
        t += dt
        if save_interval and t >= t_next_save - 1e-12:
            snapshots.append((t, Q.copy()))
            t_next_save += save_interval

    if len(snapshots) == 0 or abs(snapshots[-1][0] - t) > 1e-12:
        snapshots.append((t, Q.copy()))
    return Q, snapshots


# ─────────────────────────────────────────────
# 초기조건 헬퍼
# ─────────────────────────────────────────────

def make_Q(N, L, rho_fn, u_fn, p_fn, Ys_fn, gs, ws):
    """
    초기 Q 배열 생성.
    rho_fn, u_fn, p_fn, Ys_fn : callable(x) -> array(N,) or array(N,Ns)
    """
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)
    Ns = len(gs)

    rho = np.asarray(rho_fn(x))
    u   = np.asarray(u_fn(x))
    p   = np.asarray(p_fn(x))
    Ys  = np.asarray(Ys_fn(x))    # (N, Ns)
    if Ys.ndim == 1:
        Ys = np.stack([Ys, 1.0 - Ys], axis=1)

    gm  = gamma_mix_vec(Ys, gs, ws)
    e   = p / (rho * (gm - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3 + Ns))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    for s in range(Ns - 1):
        Q[:, 3+s] = rho * Ys[:, s]
    return Q, x, dx


def riemann_IC(N, L, xd,
               rho_L, u_L, p_L, Y1_L,
               rho_R, u_R, p_R, Y1_R,
               gs, ws):
    """
    Riemann 문제 초기조건 (2성분).
    xd : 불연속 위치
    """
    dx = L / N
    x  = np.linspace(0.5*dx, L - 0.5*dx, N)
    Ns = len(gs)

    rho = np.where(x < xd, rho_L, rho_R)
    u   = np.where(x < xd, u_L,   u_R)
    p   = np.where(x < xd, p_L,   p_R)
    Y1  = np.where(x < xd, Y1_L,  Y1_R)

    Ys  = np.stack([Y1, 1.0 - Y1], axis=1)
    gm  = gamma_mix_vec(Ys, gs, ws)
    e   = p / (rho * (gm - 1.0))
    E   = e + 0.5 * u**2

    Q = np.zeros((N, 3 + Ns))
    Q[:, 0] = rho
    Q[:, 1] = rho * u
    Q[:, 2] = rho * E
    for s in range(Ns - 1):
        Q[:, 3+s] = rho * Ys[:, s]
    return Q, x, dx
