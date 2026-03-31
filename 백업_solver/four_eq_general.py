"""
four_eq_general.py
══════════════════════════════════════════════════════════════════
범용 4방정식 다성분 다상 압축성 유동 솔버 (고도화 버전)

four_eq_1d.py 대비 개선 사항:
  1. 범용 EOS 프레임워크
       EOSSingle (추상), NASGSingleEOS, SRKSingleEOS
       MixtureEOS (Amagat 2성분 혼합)
  2. Face-based 비정렬 격자 (UnstructuredMesh)
       - 1D 선형 격자: mesh_1d(N)  [특수 케이스]
       - 2D 삼각/사각 격자: mesh_2d_rect(Nx,Ny)
       - 3D 확장 구조 유지
  3. IEC 재건: W = [T, Y₀, u·n, u_t1, u_t2, P]
       - 정렬 1D: WENO5Z (5차, four_eq_1d.py 동일 구현 → 수치 동등성)
       - 비정렬:  MUSCL + Venkatakrishnan 제한자 (2차)
  4. 3D HLLC Riemann 솔버 (face-normal 회전)
  5. SSP-RK3

참고: Collis et al. J. Comput. Phys. 114827 (2026)
══════════════════════════════════════════════════════════════════
"""
import os, warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# §1. EOS 프레임워크
# ══════════════════════════════════════════════════════════════════

class EOSSingle(ABC):
    """단일 성분 EOS 추상 기반 클래스."""

    @abstractmethod
    def rho_from_T_P(self, T, P):
        """(T, P) → 밀도 ρ"""

    @abstractmethod
    def e_from_T_P(self, T, P):
        """(T, P) → 비내부에너지 e"""

    @abstractmethod
    def c2(self, rho, T, P):
        """(ρ, T, P) → 음속² c²"""


class NASGSingleEOS(EOSSingle):
    """Noble-Abel Stiffened Gas EOS (단일 성분).

    v(T,P) = R·T/(P+P∞) + b
    e(T,P) = Cv·T·(P+γP∞)/(P+P∞) + q
    c²     = γ·(P+P∞)/ρ / (1 − ρ·b)²
    """

    def __init__(self, CP, gamma, Pinf=0.0, b=0.0, q=0.0):
        self.CP = CP; self.gamma = gamma
        self.Pinf = Pinf; self.b = b; self.q = q
        self.Cv = CP / gamma
        self.R  = CP - self.Cv

    def rho_from_T_P(self, T, P):
        v = self.R * T / np.maximum(P + self.Pinf, 1e-30) + self.b
        return 1.0 / np.maximum(v, 1e-30)

    def e_from_T_P(self, T, P):
        u = np.maximum(P + self.Pinf, 1e-30)
        return self.Cv * T * (P + self.gamma * self.Pinf) / u + self.q

    def c2(self, rho, T, P):
        u = np.maximum(P + self.Pinf, 1e-30)
        return self.gamma * u / np.maximum(rho * (1.0 - rho * self.b), 1e-30)


class SRKSingleEOS(EOSSingle):
    """Soave-Redlich-Kwong EOS (단일 성분, 실기체).

    P = R·T/(v−b) − a·α(T)/(v(v+b))
    α(T) = [1 + κ·(1−√(T/Tc))]²
    κ = 0.480 + 1.574·ω − 0.176·ω²
    a = 0.42748·R²·Tc²/Pc,  b = 0.08664·R·Tc/Pc
    """

    def __init__(self, R, Tc, Pc, omega, Cv=None):
        self.R  = R; self.Tc = Tc; self.Pc = Pc; self.omega = omega
        self.Cv = Cv if Cv is not None else 2.5 * R
        self.a  = 0.42748 * R**2 * Tc**2 / Pc
        self.b  = 0.08664 * R * Tc / Pc
        self.kappa = 0.480 + 1.574 * omega - 0.176 * omega**2

    def _alpha(self, T):
        return (1.0 + self.kappa * (1.0 - np.sqrt(np.maximum(T, 1e-6) / self.Tc)))**2

    def _dalpha_dT(self, T):
        sqr = np.sqrt(np.maximum(T, 1e-6) / self.Tc)
        return -self.kappa / self.Tc * (1.0 + self.kappa * (1.0 - sqr)) / sqr

    def _v_from_T_P(self, T, P):
        """Newton 반복으로 SRK 입방방정식 풀기."""
        alpha = self._alpha(T)
        a_ = self.a * alpha; b_ = self.b; R = self.R
        v  = np.maximum(R * T / np.maximum(P, 1e-10) + b_, b_ * 1.001)
        for _ in range(60):
            u   = np.maximum(v - b_, 1e-30)
            w   = v * (v + b_)
            f   = P - R * T / u + a_ / w
            df  = R * T / u**2 - a_ * (2*v + b_) / np.maximum(w**2, 1e-60)
            dv  = f / np.where(np.abs(df) > 1e-30, df, 1e-30)
            v   = np.maximum(v - dv, b_ * 1.001)
            if float(np.max(np.abs(dv))) < 1e-12:
                break
        return v

    def rho_from_T_P(self, T, P):
        return 1.0 / np.maximum(self._v_from_T_P(T, P), 1e-30)

    def e_from_T_P(self, T, P):
        v     = self._v_from_T_P(T, P)
        alpha = self._alpha(T)
        da_dT = self.a * self._dalpha_dT(T)
        ln_vb = np.log(np.maximum(v / (v + self.b), 1e-30))
        e_dep = (self.a * alpha - T * da_dT) / self.b * (-ln_vb)
        return self.Cv * T + e_dep

    def c2(self, rho, T, P):
        v    = 1.0 / np.maximum(rho, 1e-30)
        a_   = self.a * self._alpha(T)
        da   = self.a * self._dalpha_dT(T)
        d2a  = -self.kappa**2 / (2 * T) * self.a / self.Tc  # approximate
        b_   = self.b
        u    = np.maximum(v - b_, 1e-30)
        w    = v * (v + b_)
        dPdv = -self.R * T / u**2 + a_ * (2*v + b_) / np.maximum(w**2, 1e-60)
        dPdT = self.R / u - da / w
        cv_d = -T * d2a / b_ * np.log(np.maximum(v/(v+b_), 1e-30))
        cv   = np.maximum(self.Cv + cv_d, self.R * 0.5)
        return np.maximum(-v**2 * (dPdv - dPdT**2 / np.maximum(rho**2 * cv, 1e-30)), 1e-4)


# ─────────────────────────────────────────────────────────────────
# 2성분 Amagat 혼합 EOS
# ─────────────────────────────────────────────────────────────────

class MixtureEOS:
    """2성분 Amagat 법칙 혼합 EOS.

    v_mix = Y0·v0(T,P) + Y1·v1(T,P)
    e_mix = Y0·e0(T,P) + Y1·e1(T,P)
    """

    def __init__(self, eos0: EOSSingle, eos1: EOSSingle):
        self.eos0 = eos0
        self.eos1 = eos1
        self._is_nasg = (isinstance(eos0, NASGSingleEOS) and
                         isinstance(eos1, NASGSingleEOS))

    def rho_from_T_P_Y(self, Y0, T, P):
        Y1 = 1.0 - Y0
        v0 = 1.0 / np.maximum(self.eos0.rho_from_T_P(T, P), 1e-30)
        v1 = 1.0 / np.maximum(self.eos1.rho_from_T_P(T, P), 1e-30)
        return 1.0 / np.maximum(Y0*v0 + Y1*v1, 1e-30)

    def e_from_T_P_Y(self, Y0, T, P):
        Y1 = 1.0 - Y0
        return Y0*self.eos0.e_from_T_P(T, P) + Y1*self.eos1.e_from_T_P(T, P)

    def p_from_rho_e_Y(self, Y0, rho, e):
        """(Y0, ρ, e) → P via Newton 반복.

        NASG 혼합의 경우 four_eq_1d.py와 동일한 안정적 Newton 구현 사용.
        일반 EOS의 경우 2D Newton (T, P 동시 해결).
        """
        if self._is_nasg:
            return self._p_nasg(Y0, rho, e)
        return self._p_general(Y0, rho, e)

    def _p_nasg(self, Y0, rho, e):
        """NASG Newton — four_eq_1d.py의 nasg_p_from_rho_e와 동일."""
        s0, s1 = self.eos0, self.eos1
        Y1 = 1.0 - Y0
        Pinf = s0.Pinf
        v   = 1.0 / np.maximum(rho, 1e-30)
        vb  = np.maximum(v - Y0*s0.b, 1e-30)
        q   = Y0*s0.q + Y1*s1.q
        eq  = e - q
        _tol_Y = 1e-9
        nw  = (Y1 <= _tol_Y)
        P_fl = np.where(nw, -0.99*Pinf, 1e-6)
        P_w  = (s0.gamma-1)*eq/vb - s0.gamma*Pinf
        P_a  = (s1.gamma-1)*rho*np.maximum(eq, 0.0)
        P    = np.where(Y0 >= 0.5, P_w, P_a)
        P    = np.maximum(P, P_fl)
        for _ in range(50):
            u   = P + Pinf
            Pu  = np.where(nw, np.where(np.abs(P)>1e-30,P,1e-30), np.maximum(P,1e-10))
            A   = Y0*s0.R/u + Y1*s1.R/Pu
            B   = Y0*s0.Cv*(P+s0.gamma*Pinf)/u + Y1*s1.Cv
            A   = np.maximum(A, 1e-30); B = np.maximum(B, 1e-30)
            Tv  = vb/A; Te = eq/B; res = Tv-Te
            if float(np.max(np.abs(res))) < 1e-10:
                break
            Pu2 = np.where(nw, np.maximum(Pu**2,1e-60), np.maximum(Pu**2,1e-20))
            dA  = -Y0*s0.R/u**2 - Y1*s1.R/Pu2
            dTv = -vb*dA/np.maximum(A**2,1e-60)
            dB  = Y0*s0.Cv*Pinf*(1-s0.gamma)/u**2
            dTe = -eq*dB/np.maximum(B**2,1e-60)
            df  = dTv-dTe
            P   = P - res/np.where(np.abs(df)>1e-30, df, 1e-30)
            P   = np.maximum(P, P_fl)
        return P

    def _p_general(self, Y0, rho, e):
        """일반 EOS: 2D Newton (T, P)."""
        T = np.full_like(rho, 300.0)
        P = np.full_like(rho, 1e5)
        for _ in range(100):
            rc = self.rho_from_T_P_Y(Y0, T, P)
            ec = self.e_from_T_P_Y(Y0, T, P)
            f1 = rc - rho; f2 = ec - e
            if (float(np.max(np.abs(f1/np.maximum(rho,1e-10)))) < 1e-8 and
                    float(np.max(np.abs(f2/np.maximum(np.abs(e),1e-10)))) < 1e-8):
                break
            dT = np.maximum(np.abs(T)*1e-6, 1e-8)
            dP = np.maximum(np.abs(P)*1e-6, 1e-4)
            dr_dT = (self.rho_from_T_P_Y(Y0,T+dT,P)-rc)/dT
            dr_dP = (self.rho_from_T_P_Y(Y0,T,P+dP)-rc)/dP
            de_dT = (self.e_from_T_P_Y(Y0,T+dT,P)-ec)/dT
            de_dP = (self.e_from_T_P_Y(Y0,T,P+dP)-ec)/dP
            det   = dr_dT*de_dP - dr_dP*de_dT
            det   = np.where(np.abs(det)>1e-60, det, 1e-60)
            T = np.maximum(T - (de_dP*f1 - dr_dP*f2)/det, 1.0)
            P = np.maximum(P - (dr_dT*f2 - de_dT*f1)/det, 1e-6)
        return P

    def T_from_P_e_Y(self, Y0, P, e):
        """(Y0, P, e) → T."""
        s0, s1 = self.eos0, self.eos1
        if self._is_nasg:
            Y1 = 1.0-Y0; q = Y0*s0.q+Y1*s1.q; Pinf = s0.Pinf
            d0 = s0.Cv*(P+s0.gamma*Pinf)/np.maximum(P+Pinf, 1.0)
            d1 = s1.Cv
            T  = (e-q)/np.maximum(Y0*d0+Y1*d1, 1e-10)
            return np.maximum(T, 1e-6)
        T = np.full_like(P, 300.0)
        for _ in range(50):
            ec = self.e_from_T_P_Y(Y0, T, P)
            dT = np.maximum(np.abs(T)*1e-6, 1e-8)
            de = (self.e_from_T_P_Y(Y0, T+dT, P)-ec)/dT
            T  = np.maximum(T - (ec-e)/np.where(np.abs(de)>1e-20,de,1e-20), 1e-6)
        return T

    def T_from_rho_P_Y(self, Y0, rho, P):
        """Amagat 역산: (Y0, ρ, P) → T."""
        s0, s1 = self.eos0, self.eos1
        if self._is_nasg:
            Y1 = 1.0-Y0
            v  = 1.0/np.maximum(rho, 1e-30)
            vb = np.maximum(v - Y0*s0.b, 1e-30)
            A  = (Y0*s0.R/np.maximum(P+s0.Pinf,1e-30)
                 +Y1*s1.R/np.maximum(P, 1e-30))
            return np.maximum(vb/np.maximum(A, 1e-30), 1e-6)
        # general: Newton
        T = np.full_like(rho, 300.0)
        for _ in range(50):
            rc = self.rho_from_T_P_Y(Y0, T, P)
            dT = np.maximum(np.abs(T)*1e-6, 1e-8)
            dr = (self.rho_from_T_P_Y(Y0, T+dT, P)-rc)/dT
            T  = np.maximum(T-(rc-rho)/np.where(np.abs(dr)>1e-20,dr,1e-20), 1e-6)
        return T

    def c2_mix(self, Y0, rho, T, P):
        """Wood 혼합 음속² (phase-frozen)."""
        _tol = 1e-9
        Y0s = np.where(Y0>1-_tol, 1.0, np.where(Y0<_tol, 0.0, Y0))
        Y1s = 1.0 - Y0s
        rho0 = self.eos0.rho_from_T_P(T, P)
        rho1 = self.eos1.rho_from_T_P(T, P)
        v0   = 1.0/np.maximum(rho0, 1e-30)
        v1   = 1.0/np.maximum(rho1, 1e-30)
        phi0 = np.maximum(rho*Y0s*v0, 0.0)
        phi1 = np.maximum(rho*Y1s*v1, 0.0)
        c0sq = self.eos0.c2(rho0, T, P)
        c1sq = self.eos1.c2(rho1, T, P)
        inv  = (phi0/np.maximum(rho0*c0sq,1e-10)
               +phi1/np.maximum(rho1*c1sq,1e-10))
        return np.maximum(1.0/np.maximum(rho*inv, 1e-20), 1e-4)

    def cons_to_prim(self, r0, r1, rho_ux, rho_uy, rho_uz, rhoE):
        """보존변수 → 원시변수."""
        rho = r0 + r1
        Y0  = r0 / np.maximum(rho, 1e-30)
        ux  = rho_ux / np.maximum(rho, 1e-30)
        uy  = rho_uy / np.maximum(rho, 1e-30)
        uz  = rho_uz / np.maximum(rho, 1e-30)
        e   = rhoE / np.maximum(rho, 1e-30) - 0.5*(ux**2+uy**2+uz**2)
        P   = self.p_from_rho_e_Y(Y0, rho, e)
        T   = self.T_from_P_e_Y(Y0, P, e)
        c2  = self.c2_mix(Y0, rho, T, P)
        return Y0, ux, uy, uz, e, P, T, c2


# ─────────────────────────────────────────────────────────────────
# 사전 정의 EOS 팩토리
# ─────────────────────────────────────────────────────────────────
_NASG_PARAMS = {
    'water':    dict(CP=4185.0,      gamma=1.0123, q=-1.143e6, b=9.203e-4, Pinf=1.835e8),
    'air':      dict(CP=1011.0,      gamma=1.4,    q=0.0,      b=0.0,      Pinf=0.0),
    'helium':   dict(CP=5091.0,      gamma=1.66,   q=0.0,      b=0.0,      Pinf=0.0),
    'water_nd': dict(CP=4.4/3.4,     gamma=4.4,    q=0.0,      b=0.0,      Pinf=6000.0),
    'air_nd':   dict(CP=1.4/0.4,     gamma=1.4,    q=0.0,      b=0.0,      Pinf=0.0),
}

def make_nasg_mixture(sp0: str, sp1: str) -> MixtureEOS:
    def _mk(nm):
        p = _NASG_PARAMS[nm]
        return NASGSingleEOS(**p)
    return MixtureEOS(_mk(sp0), _mk(sp1))

def make_srk_water_air() -> MixtureEOS:
    """SRK EOS 기반 water/air 혼합 (실기체)."""
    Ru = 8.314
    e0 = SRKSingleEOS(R=Ru/0.018015, Tc=647.1, Pc=22.064e6, omega=0.345,
                       Cv=(Ru/0.018015)*3.0)
    e1 = SRKSingleEOS(R=Ru/0.02897,  Tc=132.5, Pc=3.77e6,   omega=0.036,
                       Cv=(Ru/0.02897)*2.5)
    return MixtureEOS(e0, e1)


# ══════════════════════════════════════════════════════════════════
# §2. 비정렬 격자 (Face-based Unstructured Mesh)
# ══════════════════════════════════════════════════════════════════

@dataclass
class UnstructuredMesh:
    """Face-based 비정렬 격자.

    face_cells[f] = [cL, cR]:
      cL = 법선벡터 n이 나오는 방향의 셀 (left)
      cR = 법선벡터 n이 향하는 방향의 셀 (right)
      cL = -1: 왼쪽 경계면
      cR = -1: 오른쪽 경계면

    bc_type[f]: 'I'=내부, 'P'=주기, 'T'=transmissive
    bc_pair[f]: 주기면의 paired face (-1이면 비주기)
    """
    n_cells:      int
    n_faces:      int
    ndim:         int           # 공간 차원 (1, 2, 3)
    cell_centers: np.ndarray    # (n_cells, 3)
    cell_volumes: np.ndarray    # (n_cells,)
    face_centers: np.ndarray    # (n_faces, 3)
    face_normals: np.ndarray    # (n_faces, 3) — 단위 법선 (L→R)
    face_areas:   np.ndarray    # (n_faces,)
    face_cells:   np.ndarray    # (n_faces, 2) int
    bc_type:      object        # (n_faces,) dtype=object
    bc_pair:      np.ndarray    # (n_faces,) int
    # 1D 구조 격자 전용
    is_structured_1d: bool = False
    dx_1d:            float = 0.0


def mesh_1d(N: int, bc: str = 'periodic',
            x_lo: float = -0.5, x_hi: float = 0.5) -> UnstructuredMesh:
    """균일 1D 격자 (N 셀, N+1 면)."""
    dx    = (x_hi - x_lo) / N
    xc    = np.linspace(x_lo + dx/2, x_hi - dx/2, N)
    xf    = np.linspace(x_lo, x_hi, N+1)

    cc = np.zeros((N, 3));   cc[:, 0] = xc
    fc = np.zeros((N+1, 3)); fc[:, 0] = xf
    fn = np.zeros((N+1, 3)); fn[:, 0] = 1.0   # +x 방향

    # face_cells: f번 면의 [cL, cR]
    # 내부 면 f (1 ≤ f ≤ N-1): cL=f-1, cR=f
    # 왼쪽 경계 f=0:   cL=-1, cR=0
    # 오른쪽 경계 f=N: cL=N-1, cR=-1
    fcs   = np.zeros((N+1, 2), dtype=int)
    fcs[:, 0] = np.arange(-1, N)    # cL
    fcs[:, 1] = np.arange(0, N+1)  # cR
    fcs[N, 1] = -1

    bt   = np.full(N+1, 'I', dtype=object)
    bp   = np.full(N+1, -1, dtype=int)
    bstr = bc[0].upper()
    bt[0] = bstr; bt[N] = bstr
    if bstr == 'P':
        bp[0] = N; bp[N] = 0

    return UnstructuredMesh(
        n_cells=N, n_faces=N+1, ndim=1,
        cell_centers=cc, cell_volumes=np.full(N, dx),
        face_centers=fc, face_normals=fn,
        face_areas=np.ones(N+1), face_cells=fcs,
        bc_type=bt, bc_pair=bp,
        is_structured_1d=True, dx_1d=dx,
    )


def mesh_2d_rect(Nx: int, Ny: int,
                 x_lo=0.0, x_hi=1.0, y_lo=0.0, y_hi=1.0,
                 bc_x='periodic', bc_y='transmissive') -> UnstructuredMesh:
    """균일 2D 사각 격자 (Nx×Ny 셀).

    각 셀은 동·서·남·북 4면을 가짐.
    전체 면 수 = Nx*(Ny+1) [수평] + (Nx+1)*Ny [수직].
    """
    dx = (x_hi - x_lo) / Nx
    dy = (y_hi - y_lo) / Ny
    N  = Nx * Ny

    # 셀 중심
    xc = np.linspace(x_lo+dx/2, x_hi-dx/2, Nx)
    yc = np.linspace(y_lo+dy/2, y_hi-dy/2, Ny)
    XX, YY = np.meshgrid(xc, yc)   # (Ny, Nx)
    cc = np.zeros((N, 3))
    cc[:, 0] = XX.ravel()
    cc[:, 1] = YY.ravel()
    vol = np.full(N, dx*dy)

    def cell_id(i, j):  # i: x-index, j: y-index
        return j * Nx + i

    faces_list = []   # (cL, cR, nx, ny, nz, area, xf, yf)

    # ── 수직면 (法線: +x) i=0..Nx, j=0..Ny-1 ────────────────────
    for j in range(Ny):
        for i in range(Nx+1):
            xf_  = x_lo + i*dx
            yf_  = y_lo + (j+0.5)*dy
            cL   = cell_id(i-1, j) if i > 0   else -1
            cR   = cell_id(i,   j) if i < Nx  else -1
            bc_f = 'I'
            if i == 0 or i == Nx:
                bc_f = bc_x[0].upper()
            faces_list.append((cL, cR, 1.0, 0.0, 0.0, dy, xf_, yf_, bc_f))

    # ── 수평면 (法線: +y) j=0..Ny, i=0..Nx-1 ────────────────────
    for j in range(Ny+1):
        for i in range(Nx):
            xf_  = x_lo + (i+0.5)*dx
            yf_  = y_lo + j*dy
            cL   = cell_id(i, j-1) if j > 0   else -1
            cR   = cell_id(i, j)   if j < Ny  else -1
            bc_f = 'I'
            if j == 0 or j == Ny:
                bc_f = bc_y[0].upper()
            faces_list.append((cL, cR, 0.0, 1.0, 0.0, dx, xf_, yf_, bc_f))

    Nf = len(faces_list)
    fcs_arr = np.array([(r[0], r[1]) for r in faces_list], dtype=int)
    fn_arr  = np.array([(r[2], r[3], r[4]) for r in faces_list])
    fa_arr  = np.array([r[5] for r in faces_list])
    fc_arr  = np.zeros((Nf, 3))
    fc_arr[:, 0] = [r[6] for r in faces_list]
    fc_arr[:, 1] = [r[7] for r in faces_list]
    bt_arr  = np.array([r[8] for r in faces_list], dtype=object)
    bp_arr  = np.full(Nf, -1, dtype=int)

    # 주기 면 연결 (x 방향)
    if bc_x[0].upper() == 'P':
        left_faces  = [k for k,(c0,c1,*_) in enumerate(faces_list) if c0==-1]
        right_faces = [k for k,(c0,c1,*_) in enumerate(faces_list) if c1==-1
                       and faces_list[k][2]!=0.0]
        for lf, rf in zip(sorted(left_faces), sorted(right_faces)):
            bp_arr[lf] = rf; bp_arr[rf] = lf

    return UnstructuredMesh(
        n_cells=N, n_faces=Nf, ndim=2,
        cell_centers=cc, cell_volumes=vol,
        face_centers=fc_arr, face_normals=fn_arr,
        face_areas=fa_arr, face_cells=fcs_arr,
        bc_type=bt_arr, bc_pair=bp_arr,
        is_structured_1d=False, dx_1d=0.0,
    )


# ══════════════════════════════════════════════════════════════════
# §3. WENO5Z 재건 (정렬 1D — four_eq_1d.py와 동일)
# ══════════════════════════════════════════════════════════════════

def _weps(q):
    return max(1e-6 * float(np.mean(q**2)), 1e-36)

def weno5z_lr(q, eps=None):
    """WENO5Z Borges(2008) 좌·우 경계값 재건 (주기 경계 wrap)."""
    if eps is None: eps = _weps(q)
    qm2=np.roll(q,2); qm1=np.roll(q,1); qp1=np.roll(q,-1); qp2=np.roll(q,-2); qp3=np.roll(q,-3)
    p0L=( 2*qm2- 7*qm1+11*q  )/6; p1L=(-qm1+ 5*q  + 2*qp1)/6; p2L=(2*q  + 5*qp1-  qp2)/6
    b0L=(13/12)*(qm2-2*qm1+q  )**2+(1/4)*(qm2-4*qm1+3*q )**2
    b1L=(13/12)*(qm1-2*q  +qp1)**2+(1/4)*(qm1-        qp1)**2
    b2L=(13/12)*(q  -2*qp1+qp2)**2+(1/4)*(3*q -4*qp1+qp2)**2
    t5L=np.abs(b0L-b2L)
    a0L=0.1*(1+(t5L/(b0L+eps))**2); a1L=0.6*(1+(t5L/(b1L+eps))**2); a2L=0.3*(1+(t5L/(b2L+eps))**2)
    s=a0L+a1L+a2L; qL=(a0L*p0L+a1L*p1L+a2L*p2L)/s
    p0R=(11*qp1- 7*qp2+2*qp3)/6; p1R=( 2*q  + 5*qp1-  qp2)/6; p2R=(-qm1+ 5*q  + 2*qp1)/6
    b0R=(13/12)*(qp3-2*qp2+qp1)**2+(1/4)*(qp3-4*qp2+3*qp1)**2
    b1R=(13/12)*(qp2-2*qp1+q  )**2+(1/4)*(qp2-        q   )**2
    b2R=(13/12)*(qp1-2*q  +qm1)**2+(1/4)*(3*qp1-4*q+qm1   )**2
    t5R=np.abs(b0R-b2R)
    a0R=0.1*(1+(t5R/(b0R+eps))**2); a1R=0.6*(1+(t5R/(b1R+eps))**2); a2R=0.3*(1+(t5R/(b2R+eps))**2)
    s=a0R+a1R+a2R; qR=(a0R*p0R+a1R*p1R+a2R*p2R)/s
    return qL, qR

def weno3_lr(q, eps=1e-36):
    qm1=np.roll(q,1); qp1=np.roll(q,-1); qp2=np.roll(q,-2)
    q1L=.5*q+.5*qp1; q2L=1.5*q-.5*qm1; b1L=(q-qp1)**2; b2L=(q-qm1)**2
    a1L=(2/3)/(eps+b1L)**2; a2L=(1/3)/(eps+b2L)**2
    qL=(a1L*q1L+a2L*q2L)/(a1L+a2L)
    q1R=.5*q+.5*qp1; q2R=1.5*qp1-.5*qp2; b1R=(q-qp1)**2; b2R=(qp1-qp2)**2
    a1R=(2/3)/(eps+b1R)**2; a2R=(1/3)/(eps+b2R)**2
    qR=(a1R*q1R+a2R*q2R)/(a1R+a2R)
    return qL, qR

def _pad_t(a, ng):
    return np.concatenate([np.full(ng, a[0]), a, np.full(ng, a[-1])])

def weno5z_transmissive(q, ng=3):
    """Transmissive BC용 WENO5Z: zero-gradient 패딩 후 재건."""
    q_ = _pad_t(q, ng); sl = slice(ng-1, ng+len(q))
    qL_, qR_ = weno5z_lr(q_)
    return qL_[sl], qR_[sl]

def weno3_transmissive(q, ng=2):
    q_ = _pad_t(q, ng); sl = slice(ng-1, ng+len(q))
    qL_, qR_ = weno3_lr(q_)
    return qL_[sl], qR_[sl]


# ══════════════════════════════════════════════════════════════════
# §4. MUSCL 재건 (비정렬 격자용, 2차)
# ══════════════════════════════════════════════════════════════════

def muscl_face_states_1d(phi, dx, bc='periodic', K=1.5):
    """1D MUSCL + Venkatakrishnan 제한자.

    Returns phi_L[f], phi_R[f] at each face (N+1 faces).
    phi_L[f] = reconstructed from left cell (face right-extrapolation)
    phi_R[f] = reconstructed from right cell (face left-extrapolation)
    """
    N = len(phi)
    eps2 = (K * dx)**3

    # Neighbor values for BC
    if bc == 'periodic':
        phip = np.roll(phi, -1); phim = np.roll(phi, 1)
    else:
        phip = np.empty(N); phip[:-1]=phi[1:]; phip[-1]=phi[-1]
        phim = np.empty(N); phim[1:]=phi[:-1]; phim[0]=phi[0]

    phi_max = np.maximum(phi, np.maximum(phip, phim))
    phi_min = np.minimum(phi, np.minimum(phip, phim))

    # Green-Gauss gradient
    grad = (phip - phim) / (2*dx)

    # Face extrapolation distance = dx/2
    dR = grad * dx/2    # right face delta
    dL = -grad * dx/2   # left  face delta

    def vk_lim(delta_f, delta_ext):
        de = delta_ext
        df = delta_f
        num = (de**2 + eps2)*df + 2*df**2*de
        den = de**2 + 2*df**2 + de*df + eps2
        return np.where(np.abs(df) < 1e-30, 1.0,
                        num / np.where(np.abs(den)>1e-30, den, 1e-30))

    # Right-face limiter (phi_i → face i+1/2)
    de_R = np.where(dR > 1e-30, phi_max - phi, phi_min - phi)
    psi_R = vk_lim(dR, de_R)
    psi_R = np.clip(psi_R, 0.0, 1.0)

    # Left-face limiter (phi_i → face i-1/2)
    de_L = np.where(dL > 1e-30, phi_max - phi, phi_min - phi)
    psi_L = vk_lim(dL, de_L)
    psi_L = np.clip(psi_L, 0.0, 1.0)

    # Face values: N+1 faces
    # face f=0: left boundary — from cell 0
    # face f (1..N-1): phi_L[f]=phi[f-1]+psi_R[f-1]*dR[f-1], phi_R[f]=phi[f]+psi_L[f]*dL[f]
    # face f=N: right boundary — from cell N-1
    phi_L_face = np.empty(N+1)  # value from left cell
    phi_R_face = np.empty(N+1)  # value from right cell

    # Interior faces (1..N-1): L from cell f-1, R from cell f
    phi_L_face[1:N] = phi[:N-1] + psi_R[:N-1] * dR[:N-1]  # right extrapolation of left cell
    phi_R_face[1:N] = phi[1:]   + psi_L[1:]   * dL[1:]     # left  extrapolation of right cell

    # Boundary faces
    if bc == 'periodic':
        # face 0: L=cell N-1 (wrapped), R=cell 0
        phi_L_face[0] = phi[N-1] + psi_R[N-1] * dR[N-1]
        phi_R_face[0] = phi[0]   + psi_L[0]   * dL[0]
        # face N: L=cell N-1, R=cell 0 (wrapped)
        phi_L_face[N] = phi[N-1] + psi_R[N-1] * dR[N-1]
        phi_R_face[N] = phi[0]   + psi_L[0]   * dL[0]
    else:  # transmissive
        phi_L_face[0] = phi[0]
        phi_R_face[0] = phi[0]
        phi_L_face[N] = phi[N-1]
        phi_R_face[N] = phi[N-1]

    return phi_L_face, phi_R_face


# ══════════════════════════════════════════════════════════════════
# §5. IEC Face 상태 재건
# ══════════════════════════════════════════════════════════════════

def _weno_to_n1(qL, qR):
    """WENO5Z 주기 결과 N → N+1 면 배열 변환.

    WENO5Z(q) → qL[i], qR[i] = face i+1/2의 좌/우 상태 (크기 N)
    mesh_1d face f (1..N) = face (f-1)+1/2 → qL[f-1], qR[f-1]
    face 0 (주기 왼쪽 경계) = face N-1+1/2 → qL[N-1], qR[N-1]
    """
    return (np.concatenate([[qL[-1]], qL]),
            np.concatenate([[qR[-1]], qR]))


def _make_iec_faces(mesh, eos, U, weno_order, bc, use_muscl=False):
    """IEC 재건 핵심: W=[T, Y0, ux, uy, uz, P] → face 상태.

    U: (6, n_cells) = [r0, r1, rho_ux, rho_uy, rho_uz, rhoE]
    Returns WL, WR: (9, n_faces) = [T, Y0, ux, uy, uz, P, rho, e, c]
    """
    r0, r1, rho_ux, rho_uy, rho_uz, rhoE = U
    rho = r0 + r1
    Y0, ux, uy, uz, e, P, T, c2 = eos.cons_to_prim(r0, r1, rho_ux, rho_uy, rho_uz, rhoE)
    c = np.sqrt(np.maximum(c2, 0.0))

    if bc == 'periodic' and not use_muscl:
        recon = weno5z_lr if weno_order >= 5 else weno3_lr
        TL,  TR  = _weno_to_n1(*recon(T))
        Y0L, Y0R = _weno_to_n1(*recon(Y0))
        uxL, uxR = _weno_to_n1(*recon(ux))
        uyL, uyR = _weno_to_n1(*recon(uy))
        uzL, uzR = _weno_to_n1(*recon(uz))
        PL,  PR  = _weno_to_n1(*recon(P))

    elif bc == 'transmissive' and not use_muscl:
        _rn = weno5z_transmissive if weno_order >= 5 else weno3_transmissive
        TL,  TR  = _rn(T)
        Y0L, Y0R = _rn(Y0)
        uxL, uxR = _rn(ux)
        uyL, uyR = _rn(uy)
        uzL, uzR = _rn(uz)
        PL,  PR  = _rn(P)

    else:  # MUSCL
        dx_  = mesh.dx_1d if mesh.is_structured_1d else float(
            np.mean(np.linalg.norm(np.diff(mesh.cell_centers, axis=0), axis=1)))
        _mr = lambda q: muscl_face_states_1d(q, dx_, bc)
        TL,  TR  = _mr(T)
        Y0L, Y0R = _mr(Y0)
        uxL, uxR = _mr(ux)
        uyL, uyR = _mr(uy)
        uzL, uzR = _mr(uz)
        PL,  PR  = _mr(P)

    # 클리핑
    Y0L = np.clip(Y0L, 0.0, 1.0); Y0R = np.clip(Y0R, 0.0, 1.0)
    s0 = eos.eos0
    _Pinf = s0.Pinf if isinstance(s0, NASGSingleEOS) else 0.0
    PL = np.maximum(PL, np.where((1-Y0L)<=1e-9, -0.99*_Pinf, 1e-6))
    PR = np.maximum(PR, np.where((1-Y0R)<=1e-9, -0.99*_Pinf, 1e-6))
    TL = np.maximum(TL, 1e-15); TR = np.maximum(TR, 1e-15)

    rhoL = eos.rho_from_T_P_Y(Y0L, TL, PL)
    rhoR = eos.rho_from_T_P_Y(Y0R, TR, PR)
    eL   = eos.e_from_T_P_Y(Y0L, TL, PL)
    eR   = eos.e_from_T_P_Y(Y0R, TR, PR)
    cL   = np.sqrt(eos.c2_mix(Y0L, rhoL, TL, PL))
    cR   = np.sqrt(eos.c2_mix(Y0R, rhoR, TR, PR))

    WL = np.array([TL, Y0L, uxL, uyL, uzL, PL, rhoL, eL, cL])
    WR = np.array([TR, Y0R, uxR, uyR, uzR, PR, rhoR, eR, cR])
    return WL, WR


def _make_std_faces(mesh, eos, U, weno_order, bc, use_muscl=False):
    """표준 재건: W = [ρY₀, u, P]."""
    r0, r1, rho_ux, rho_uy, rho_uz, rhoE = U
    rho = r0+r1
    Y0  = r0/np.maximum(rho, 1e-30)
    ux  = rho_ux/np.maximum(rho, 1e-30)
    uy  = rho_uy/np.maximum(rho, 1e-30)
    uz  = rho_uz/np.maximum(rho, 1e-30)
    e   = rhoE/np.maximum(rho, 1e-30) - 0.5*(ux**2+uy**2+uz**2)
    P   = eos.p_from_rho_e_Y(Y0, rho, e)

    if bc == 'periodic' and not use_muscl:
        recon = weno5z_lr if weno_order >= 5 else weno3_lr
        r0L,r0R=_weno_to_n1(*recon(r0)); r1L,r1R=_weno_to_n1(*recon(r1))
        uxL,uxR=_weno_to_n1(*recon(ux)); uyL,uyR=_weno_to_n1(*recon(uy))
        uzL,uzR=_weno_to_n1(*recon(uz)); PL,PR=_weno_to_n1(*recon(P))
    elif bc == 'transmissive' and not use_muscl:
        _rn = weno5z_transmissive if weno_order >= 5 else weno3_transmissive
        r0L,r0R=_rn(r0); r1L,r1R=_rn(r1)
        uxL,uxR=_rn(ux); uyL,uyR=_rn(uy); uzL,uzR=_rn(uz); PL,PR=_rn(P)
    else:
        dx_ = mesh.dx_1d if mesh.is_structured_1d else 0.01
        _mr = lambda q: muscl_face_states_1d(q, dx_, bc)
        r0L,r0R=_mr(r0); r1L,r1R=_mr(r1)
        uxL,uxR=_mr(ux); uyL,uyR=_mr(uy); uzL,uzR=_mr(uz); PL,PR=_mr(P)

    r0L=np.maximum(r0L,0.); r0R=np.maximum(r0R,0.)
    r1L=np.maximum(r1L,0.); r1R=np.maximum(r1R,0.)
    rhoL=np.maximum(r0L+r1L,1e-30); rhoR=np.maximum(r0R+r1R,1e-30)
    Y0L=r0L/rhoL; Y0R=r0R/rhoR
    PL=np.maximum(PL,1e-15); PR=np.maximum(PR,1e-15)
    TL=eos.T_from_rho_P_Y(Y0L, rhoL, PL)
    TR=eos.T_from_rho_P_Y(Y0R, rhoR, PR)
    eL=eos.e_from_T_P_Y(Y0L, TL, PL); eR=eos.e_from_T_P_Y(Y0R, TR, PR)
    cL=np.sqrt(eos.c2_mix(Y0L, rhoL, TL, PL))
    cR=np.sqrt(eos.c2_mix(Y0R, rhoR, TR, PR))

    WL = np.array([TL, Y0L, uxL, uyL, uzL, PL, rhoL, eL, cL])
    WR = np.array([TR, Y0R, uxR, uyR, uzR, PR, rhoR, eR, cR])
    return WL, WR


# ══════════════════════════════════════════════════════════════════
# §6. 3D HLLC Riemann 솔버
# ══════════════════════════════════════════════════════════════════

def hllc_flux_3d(WL, WR, nx, ny, nz):
    """3D HLLC 플럭스 (face-normal 방향).

    WL, WR: (9, n_faces) = [T, Y0, ux, uy, uz, P, rho, e, c]
    nx, ny, nz: (n_faces,) 단위 법선벡터 성분
    Returns:
      F:   (6, n_faces) = [Fr0, Fr1, Frux, Fruy, Fruz, FrhoE]
      lam: (n_faces,)   최대 파속
    """
    TL, Y0L, uxL, uyL, uzL, PL, rhoL, eL, cL = WL
    TR, Y0R, uxR, uyR, uzR, PR, rhoR, eR, cR  = WR

    # 법선방향 속도
    u_nL = uxL*nx + uyL*ny + uzL*nz
    u_nR = uxR*nx + uyR*ny + uzR*nz

    # Einfeldt 파속
    SL = np.minimum(u_nL - cL, u_nR - cR)
    SR = np.maximum(u_nL + cL, u_nR + cR)

    # Contact 파속
    den = rhoL*(SL - u_nL) - rhoR*(SR - u_nR)
    Ss  = (PR - PL + rhoL*u_nL*(SL - u_nL) - rhoR*u_nR*(SR - u_nR)) / \
           np.where(np.abs(den) > 1e-30, den, 1e-30)

    EL  = rhoL*(eL + 0.5*(uxL**2+uyL**2+uzL**2))
    ER  = rhoR*(eR + 0.5*(uxR**2+uyR**2+uzR**2))
    Y1L = 1.0 - Y0L; Y1R = 1.0 - Y0R

    def _phys_flux(rho, Y0, ux, uy, uz, P, E, u_n):
        Y1 = 1.0 - Y0
        return np.array([
            rho*Y0*u_n, rho*Y1*u_n,
            rho*ux*u_n + P*nx, rho*uy*u_n + P*ny, rho*uz*u_n + P*nz,
            (E+P)*u_n
        ])

    FL = _phys_flux(rhoL, Y0L, uxL, uyL, uzL, PL, EL, u_nL)
    FR = _phys_flux(rhoR, Y0R, uxR, uyR, uzR, PR, ER, u_nR)

    def _star_U(rho, Y0, ux, uy, uz, P, E, u_n, S):
        fac = rho * (S - u_n) / np.where(np.abs(S-Ss) > 1e-30, S-Ss, 1e-30)
        # 접선방향 속도 보존, 법선방향 → Ss
        ux_s = ux + (Ss - u_n)*nx
        uy_s = uy + (Ss - u_n)*ny
        uz_s = uz + (Ss - u_n)*nz
        # S-u_n 부호 보존 (SL<u_n<SR 이므로 SL-u_n<0, SR-u_n>0)
        S_un = np.where(np.abs(S - u_n) > 1e-20,
                        S - u_n,
                        np.sign(S - u_n + 1e-100) * 1e-20)
        E_s  = fac*(E/np.maximum(rho,1e-30)
                    + (Ss-u_n)*(Ss + P/(np.maximum(rho,1e-30)*S_un)))
        Y1   = 1.0 - Y0
        return np.array([fac*Y0, fac*Y1, fac*ux_s, fac*uy_s, fac*uz_s, E_s])

    UL  = np.array([rhoL*Y0L, rhoL*Y1L, rhoL*uxL, rhoL*uyL, rhoL*uzL, EL])
    UR  = np.array([rhoR*Y0R, rhoR*Y1R, rhoR*uxR, rhoR*uyR, rhoR*uzR, ER])
    ULs = _star_U(rhoL, Y0L, uxL, uyL, uzL, PL, EL, u_nL, SL)
    URs = _star_U(rhoR, Y0R, uxR, uyR, uzR, PR, ER, u_nR, SR)

    F = np.where(SL[np.newaxis,:] >= 0, FL,
        np.where(Ss[np.newaxis,:]  >= 0, FL + SL[np.newaxis,:]*(ULs - UL),
        np.where(SR[np.newaxis,:]  >= 0, FR + SR[np.newaxis,:]*(URs - UR),
                                          FR)))

    lam = np.maximum(np.abs(SL), np.abs(SR))
    return F, lam


# ══════════════════════════════════════════════════════════════════
# §7. RHS (공간 잔차)
# ══════════════════════════════════════════════════════════════════

def compute_rhs(U, mesh: UnstructuredMesh, eos: MixtureEOS,
                iec=True, weno_order=5, bc='periodic',
                use_muscl=False):
    """dU/dt = -∇·F (보존형 FVM 잔차).

    U: (6, n_cells) = [r0, r1, rhoUx, rhoUy, rhoUz, rhoE]
    """
    # 면 상태 재건
    if iec:
        WL, WR = _make_iec_faces(mesh, eos, U, weno_order, bc, use_muscl)
    else:
        WL, WR = _make_std_faces(mesh, eos, U, weno_order, bc, use_muscl)

    # 면 법선 (3, n_faces)
    n  = mesh.face_normals.T
    nx, ny, nz = n[0], n[1], n[2]

    # HLLC 플럭스 (6, n_faces)
    F, lam = hllc_flux_3d(WL, WR, nx, ny, nz)
    F_A    = F * mesh.face_areas[np.newaxis, :]   # F·A

    dU = np.zeros_like(U)

    if mesh.is_structured_1d:
        # ── 1D 최적화: 직접 차분 ─────────────────────────────────
        # F_A[:, f] = 면 f의 플럭스 (f=0은 왼쪽경계, f=N은 오른쪽경계)
        # dU[:, i] = -(F_A[:, i+1] - F_A[:, i]) / dx
        dU = -(F_A[:, 1:] - F_A[:, :-1]) / mesh.dx_1d
    else:
        # ── 일반 비정렬: 면 루프 ─────────────────────────────────
        fc = mesh.face_cells
        for f in range(mesh.n_faces):
            cL, cR = fc[f, 0], fc[f, 1]
            bt = mesh.bc_type[f]
            if cL >= 0:
                dU[:, cL] -= F_A[:, f] / mesh.cell_volumes[cL]
            if cR >= 0:
                dU[:, cR] += F_A[:, f] / mesh.cell_volumes[cR]

    return dU, float(np.max(lam))


# ══════════════════════════════════════════════════════════════════
# §8. SSP-RK3
# ══════════════════════════════════════════════════════════════════

def _clip_U(U):
    """r0, r1 음수 클리핑."""
    U2 = U.copy()
    U2[0] = np.maximum(U2[0], 0.0)
    U2[1] = np.maximum(U2[1], 0.0)
    return U2

def ssprk3_step(U, rhs_fn):
    """Shu-Osher SSP-RK3."""
    k1, lam = rhs_fn(U)
    U1 = _clip_U(U + k1)
    k2, _  = rhs_fn(U1)
    U2 = _clip_U(0.75*U + 0.25*(U1 + k2))
    k3, _  = rhs_fn(U2)
    return _clip_U((1/3)*U + (2/3)*(U2 + k3)), lam


# ══════════════════════════════════════════════════════════════════
# §9. 초기조건 함수
# ══════════════════════════════════════════════════════════════════

def ic_droplet(mesh, eos, P0=101325.0, T0=297.0, u0=5.0, eps_factor=2.0):
    """§4.2.2 물 액적 이송 초기조건.

    Y0 = 0.5*(1 + tanh((0.25 - |x-0.5|) / (eps_factor*dx)))
    T=T0, P=P0, u=u0 (균일)
    """
    x   = mesh.cell_centers[:, 0]
    dx  = mesh.dx_1d if mesh.is_structured_1d else float(
          np.mean(np.diff(np.sort(x))))
    eps = eps_factor * dx
    Y0  = 0.5*(1.0 + np.tanh((0.25 - np.abs(x - 0.5)) / eps))
    Y0  = np.clip(Y0, 0.0, 1.0)
    P_  = np.full(mesh.n_cells, P0)
    rho = eos.rho_from_T_P_Y(Y0, T0, P_)
    e   = eos.e_from_T_P_Y(Y0, T0, P_)
    U = np.zeros((6, mesh.n_cells))
    U[0] = rho * Y0
    U[1] = rho * (1.0 - Y0)
    U[2] = rho * u0
    U[5] = rho * (e + 0.5*u0**2)
    return U


def ic_riemann(mesh, eos,
               state_L=(997.0, 0.0, 101325.0, 'water'),
               state_R=(1.18,  0.0, 101325.0, 'air')):
    """§4.2.1 Gas-Liquid Riemann 초기조건.

    state: (rho, u, P, phase) — 왼쪽/오른쪽 순수 상태.
    """
    x    = mesh.cell_centers[:, 0]
    xmid = 0.5*(x.min() + x.max())
    N    = mesh.n_cells

    def _rho_e(state):
        rho0, u0_, P0_, phase = state
        # 순수 상 (Y0=1 for water, Y0=0 for air)
        Y0_s = 1.0 if phase == 'water' or phase == 'water_nd' else 0.0
        Y0_  = np.full(N, Y0_s)
        T_   = eos.T_from_rho_P_Y(Y0_, np.full(N, rho0), np.full(N, P0_))
        e_   = eos.e_from_T_P_Y(Y0_, T_, np.full(N, P0_))
        return rho0, u0_, P0_, Y0_s, e_

    rhoL, uL, PL, Y0L, eL = _rho_e(state_L)
    rhoR, uR, PR, Y0R, eR = _rho_e(state_R)

    left  = x <= xmid
    right = ~left

    r0  = np.where(left, rhoL*Y0L,  rhoR*Y0R)
    r1  = np.where(left, rhoL*(1-Y0L), rhoR*(1-Y0R))
    rho = r0 + r1
    u_  = np.where(left, uL, uR)
    e_  = np.where(left, eL, eR)

    U    = np.zeros((6, N))
    U[0] = r0
    U[1] = r1
    U[2] = rho * u_
    U[5] = rho * (e_ + 0.5*u_**2)
    return U


def ic_shock_droplet(mesh, eos, sp0='water', sp1='air'):
    """§4.2.3 충격파-액적 상호작용 초기조건.

    구성:
      왼쪽 1/3: 충격파 후 상태 (air, P=1e6 Pa)
      중앙 1/3: 물 액적 (water, P=1e5 Pa)
      오른쪽 1/3: 주변 공기 (air, P=1e5 Pa)
    four_eq_1d.py의 ic_shock_droplet과 동일 파라미터.
    """
    x    = mesh.cell_centers[:, 0]
    xmin, xmax = x.min(), x.max()
    N    = mesh.n_cells

    # 공기 NASG 파라미터
    s0_air = eos.eos1
    s0_wat = eos.eos0
    Cp_air = s0_air.CP if isinstance(s0_air, NASGSingleEOS) else 1011.0
    g_air  = s0_air.gamma if isinstance(s0_air, NASGSingleEOS) else 1.4

    # Rankine-Hugoniot for M=3 shock in air (P_pre=1e5)
    P_pre  = 1e5; rho_pre = 1.18; T_pre = P_pre / (287.0 * rho_pre)
    Ms     = 3.0
    g      = g_air
    P_post = P_pre * (2*g*Ms**2 - (g-1)) / (g+1)
    rho_post = rho_pre * ((g+1)*Ms**2) / ((g-1)*Ms**2 + 2)
    T_post   = P_post / (287.0 * rho_post)
    u_post   = 1020.0  # m/s (approximate piston velocity)

    dx  = mesh.dx_1d if mesh.is_structured_1d else float(np.mean(np.diff(np.sort(x))))
    eps = 4.0 * dx
    xc_drop = 0.0; w_drop = 0.15

    # Water droplet indicator
    phi_wat = 0.5*(1 + np.tanh((w_drop - np.abs(x - xc_drop)) / eps))
    phi_wat = np.clip(phi_wat, 0.0, 1.0)

    # Shock region: x < -0.3
    phi_shock = 0.5*(1 - np.tanh((x - (-0.3)) / eps))
    phi_shock = np.clip(phi_shock, 0.0, 1.0)

    # Y0 (water mass fraction)
    T_water = 293.0; P_water = P_pre
    Y0  = phi_wat * (1.0 - phi_shock)
    Y0  = np.clip(Y0, 0.0, 1.0)

    # P, T profiles
    P_  = (phi_shock * P_post + (1-phi_shock) * P_pre)
    T_  = np.where(phi_shock > 0.5, T_post,
           np.where(phi_wat > 0.5, T_water, T_pre))
    T_  = np.maximum(T_, 1.0)

    rho = eos.rho_from_T_P_Y(Y0, T_, P_)
    e_  = eos.e_from_T_P_Y(Y0, T_, P_)
    u_  = phi_shock * u_post

    U    = np.zeros((6, N))
    U[0] = rho * Y0
    U[1] = rho * (1.0 - Y0)
    U[2] = rho * u_
    U[5] = rho * (e_ + 0.5*u_**2)
    return U


def ic_mach100_jet(mesh, eos, Mach=100.0, P0=1.0, T0=1.0, eps_abs=0.04):
    """§4.2.4 Mach-100 물 제트 (비차원 water_nd/air_nd).

    eps_abs: 고정 물리적 계면 폭 (four_eq_1d.py의 eps_abs=0.04와 동일).
    """
    x   = mesh.cell_centers[:, 0]
    N   = mesh.n_cells

    P_  = np.full(N, P0)
    rho_w = eos.rho_from_T_P_Y(np.ones(1), np.array([T0]), np.array([P0]))[0]
    T_w_  = np.array([T0]); Y0_w_ = np.ones(1)
    c2_w  = eos.c2_mix(Y0_w_, np.array([rho_w]), T_w_, np.array([P0]))[0]
    u_jet = Mach * float(np.sqrt(c2_w))

    xc, w = 0.0, 0.25
    Y0   = 0.5*(1.0 + np.tanh((w - np.abs(x - xc)) / eps_abs))
    Y0   = np.clip(Y0, 0.0, 1.0)
    rho  = eos.rho_from_T_P_Y(Y0, T0, P_)
    e_   = eos.e_from_T_P_Y(Y0, T0, P_)

    U    = np.zeros((6, N))
    U[0] = rho * Y0
    U[1] = rho * (1.0 - Y0)
    U[2] = rho * u_jet           # 균일 속도
    U[5] = rho * (e_ + 0.5*u_jet**2)
    return U


# ══════════════════════════════════════════════════════════════════
# §10. 범용 시뮬레이션 루프
# ══════════════════════════════════════════════════════════════════

def run_simulation(U0, mesh, eos, t_end, CFL=0.3, iec=True, weno_order=5,
                   bc='periodic', use_muscl=False, print_interval=100):
    """SSP-RK3 시뮬레이션 루프."""
    U = U0.copy()
    t, step = 0.0, 0
    vol = mesh.cell_volumes  # (N,)

    def rhs_fn(U_):
        dU, lam = compute_rhs(U_, mesh, eos, iec=iec, weno_order=weno_order,
                               bc=bc, use_muscl=use_muscl)
        dx_eff  = float(np.min(vol)) if not mesh.is_structured_1d else mesh.dx_1d
        dt      = min(CFL * dx_eff / (lam + 1e-10), t_end - t)
        return dU * dt, lam

    while t < t_end - 1e-14:
        r0,r1,mx,my,mz,E = U
        rho = r0+r1; Y0 = r0/np.maximum(rho,1e-30)
        Y0_, ux,uy,uz,e_,P_,T_,c2_ = eos.cons_to_prim(r0,r1,mx,my,mz,E)
        lam = float(np.max(np.abs(ux)+np.sqrt(c2_)))
        dx_ = mesh.dx_1d if mesh.is_structured_1d else float(np.min(mesh.cell_volumes))
        dt  = min(CFL * dx_ / (lam+1e-10), t_end - t)

        def _rhs(U_):
            dU, _ = compute_rhs(U_, mesh, eos, iec=iec, weno_order=weno_order,
                                 bc=bc, use_muscl=use_muscl)
            return dU * dt, 0.0

        U, _ = ssprk3_step(U, _rhs)
        t += dt; step += 1

        if not np.all(np.isfinite(U[5])):
            print(f"  NaN/Inf at step={step} t={t:.4e}")
            break
        if print_interval > 0 and step % print_interval == 0:
            r0f,r1f,mxf,_,_,Ef = U
            rhof = r0f+r1f; Y0f = r0f/np.maximum(rhof,1e-30)
            Y0_f,uxf,_,_,_,Pf,_,c2f = eos.cons_to_prim(r0f,r1f,mxf,
                                         np.zeros_like(mxf),np.zeros_like(mxf),Ef)
            print(f"  step={step} t={t:.4e} Pmin={Pf.min():.3e} Pmax={Pf.max():.3e}")

    return U, t, step


# ══════════════════════════════════════════════════════════════════
# §11. 검증 함수 (기존 four_eq_1d.py 결과와 비교)
# ══════════════════════════════════════════════════════════════════

def validate_droplet(N=100, CFL=0.5, t_end=0.2, compare=True):
    """§4.2.2 물 액적 이송 — IEC WENO5Z+MUSCL vs STD 비교.

    목표: IEC PE < 1e-10 (four_eq_1d.py: 8.2e-12)
    """
    print(f"\n{'='*60}")
    print(f"§4.2.2 Droplet Advection  N={N}  t_end={t_end}")
    print(f"{'='*60}")

    eos  = make_nasg_mixture('water', 'air')
    P0, T0, u0 = 101325.0, 297.0, 5.0

    results = {}
    configs = [
        ('IEC_W5Z',    dict(iec=True,  weno_order=5, use_muscl=False)),
        ('IEC_MUSCL',  dict(iec=True,  weno_order=5, use_muscl=True)),
        ('STD_W5Z',    dict(iec=False, weno_order=5, use_muscl=False)),
    ]

    for label, cfg in configs:
        mesh  = mesh_1d(N, bc='periodic', x_lo=0.0, x_hi=1.0)
        U0_   = ic_droplet(mesh, eos, P0=P0, T0=T0, u0=u0, eps_factor=4.0)
        U_f, t_f, steps = run_simulation(
            U0_, mesh, eos, t_end=t_end, CFL=CFL,
            bc='periodic', print_interval=0, **cfg)

        r0f,r1f,mxf,_,_,Ef = U_f
        rhof = r0f+r1f; Y0f = r0f/np.maximum(rhof,1e-30)
        _,uxf,_,_,ef_,Pf,Tf,_ = eos.cons_to_prim(r0f,r1f,mxf,
                                   np.zeros(N),np.zeros(N),Ef)
        pe = float(np.max(np.abs(Pf/P0 - 1.0)))
        te = float(np.max(np.abs(Tf/T0 - 1.0)))
        ue = float(np.max(np.abs(uxf/u0 - 1.0)))
        mark = 'OK' if pe<1e-3 else 'NG'
        print(f"  {label:<14} |dP/P0|={pe:.3e}  |dT/T0|={te:.3e}  "
              f"|du/u0|={ue:.3e}  [{mark}]")
        results[label] = dict(pe=pe, te=te, ue=ue, P=Pf, x=mesh.cell_centers[:,0])

    if compare:
        print(f"\n  [기준 four_eq_1d.py] IEC W5Z+Char: ~8e-12, STD: ~1e-6")
    return results


def validate_riemann(N=501, CFL=0.3, t_end=0.14):
    """§4.2.1 Gas-Liquid Riemann 문제.

    목표: IEC 완료, STD NaN (four_eq_1d.py와 동일)
    """
    print(f"\n{'='*60}")
    print(f"§4.2.1 Gas-Liquid Riemann  N={N}  t_end={t_end}")
    print(f"{'='*60}")

    eos = make_nasg_mixture('water_nd', 'air_nd')

    # Table 4 (Collis 2026): water_nd | air_nd Riemann
    # Left: pure water_nd, Right: pure air_nd
    rho_w_nd = eos.rho_from_T_P_Y(np.ones(1), np.array([1.0]), np.array([1.0]))[0]
    rho_a_nd = eos.rho_from_T_P_Y(np.zeros(1), np.array([1.0]), np.array([1.0]))[0]

    for label, iec_f in [('IEC', True), ('STD', False)]:
        mesh = mesh_1d(N, bc='transmissive', x_lo=-0.5, x_hi=0.5)
        # IC: pure water_nd left, pure air_nd right, P=1, T=1, u=0
        x   = mesh.cell_centers[:, 0]
        Y0_ = np.where(x < 0.0, 1.0 - 1e-12, 1e-12)
        P_  = np.ones(N); T_  = np.ones(N)
        rho = eos.rho_from_T_P_Y(Y0_, T_, P_)
        e_  = eos.e_from_T_P_Y(Y0_, T_, P_)
        U0_ = np.zeros((6, N))
        U0_[0] = rho * Y0_; U0_[1] = rho*(1-Y0_); U0_[5] = rho*e_

        U_f, t_f, steps = run_simulation(
            U0_, mesh, eos, t_end=t_end, CFL=CFL, iec=iec_f,
            weno_order=5, bc='transmissive', print_interval=0)

        r0f,r1f,mxf,_,_,Ef = U_f
        rhof=r0f+r1f; Y0f=r0f/np.maximum(rhof,1e-30)
        ok = np.all(np.isfinite(Ef))
        mark = 'OK' if (iec_f and ok) or (not iec_f and not ok) else 'NOTE'
        print(f"  [{label}] t={t_f:.4f}  steps={steps}  finite={ok}  [{mark}]")

    print(f"  [기준] IEC=완료, STD=NaN")


def validate_shock_droplet(N=200, CFL=0.3, t_end=5e-4):
    """§4.2.3 충격파-액적 상호작용.

    목표: IEC N=200 완료 (four_eq_1d.py와 동일)
    """
    print(f"\n{'='*60}")
    print(f"§4.2.3 Shock-Droplet  N={N}  t_end={t_end}")
    print(f"{'='*60}")

    eos = make_nasg_mixture('water', 'air')
    mesh = mesh_1d(N, bc='transmissive', x_lo=-0.5, x_hi=0.5)
    U0_  = ic_shock_droplet(mesh, eos)

    U_f, t_f, steps = run_simulation(
        U0_, mesh, eos, t_end=t_end, CFL=CFL, iec=True,
        weno_order=5, bc='transmissive', print_interval=0)

    r0f,r1f,mxf,_,_,Ef = U_f
    rhof=r0f+r1f; Y0f=r0f/np.maximum(rhof,1e-30)
    _,uxf,_,_,_,Pf,_,_ = eos.cons_to_prim(r0f,r1f,mxf,
                           np.zeros(N),np.zeros(N),Ef)
    ok = np.all(np.isfinite(Ef))
    mark = 'OK' if ok else 'NaN'
    print(f"  [IEC] t={t_f:.4e}  steps={steps}  Pmax={Pf.max():.3e}  "
          f"rhomax={rhof.max():.1f}  [{mark}]")
    print(f"  [기준 four_eq_1d.py] IEC N=200 완료")
    return t_f, ok


def validate_mach100(N=400, CFL=0.1, t_end=1e-3):
    """§4.2.4 Mach-100 물 제트 — IEC vs STD PE 비교.

    목표: IEC PE < 1e-7 (four_eq_1d.py: 2.2e-8 at N=400)
    """
    print(f"\n{'='*60}")
    print(f"§4.2.4 Mach-100 Water Jet  N={N}  t_end={t_end}")
    print(f"{'='*60}")

    eos = make_nasg_mixture('water_nd', 'air_nd')
    results = {}
    for label, iec_f in [('IEC', True), ('STD', False)]:
        mesh = mesh_1d(N, bc='periodic')
        U0_  = ic_mach100_jet(mesh, eos, eps_abs=0.04)

        U_f, t_f, steps = run_simulation(
            U0_, mesh, eos, t_end=t_end, CFL=CFL, iec=iec_f,
            weno_order=5, bc='periodic', print_interval=0)

        r0f,r1f,mxf,_,_,Ef = U_f
        rhof=r0f+r1f; Y0f=r0f/np.maximum(rhof,1e-30)
        _,uxf,_,_,_,Pf,_,_ = eos.cons_to_prim(r0f,r1f,mxf,
                               np.zeros(N),np.zeros(N),Ef)
        pe = float(np.max(np.abs(Pf - 1.0))) if np.all(np.isfinite(Ef)) else float('inf')
        mark = 'OK' if pe<1e-5 else ('NOTE' if pe<1.0 else 'NG')
        print(f"  [{label}] t={t_f:.3e}  PE={pe:.3e}  [{mark}]")
        results[label] = pe

    iec_pe = results['IEC']; std_pe = results['STD']
    ratio  = std_pe/iec_pe if iec_pe>0 and np.isfinite(iec_pe) else float('inf')
    print(f"\n  IEC/STD 비율: {ratio:.1f}×  [기준 four_eq_1d.py: ~53×, IEC=2.2e-8]")
    return results


def validate_all_compare(N_base=100):
    """전체 검증 + four_eq_1d.py 비교 출력."""
    print("\n" + "█"*60)
    print("  four_eq_general.py — 전체 검증 (four_eq_1d.py 비교)")
    print("█"*60)
    validate_droplet(N=N_base, CFL=0.5)
    validate_riemann(N=max(N_base*5, 501), CFL=0.3)
    validate_shock_droplet(N=max(N_base*2, 200), CFL=0.3)
    validate_mach100(N=max(N_base*4, 400), CFL=0.1)
    print("\n" + "█"*60)
    print("  검증 완료")
    print("█"*60)


# ══════════════════════════════════════════════════════════════════
# §12. 2D 데모 (비정렬 격자 시연)
# ══════════════════════════════════════════════════════════════════

def demo_2d_advection(Nx=50, Ny=50, t_end=0.1):
    """2D 액적 이송 시연 (비정렬 격자 기능 확인).

    비정렬 격자에서 MUSCL 재건 + HLLC 플럭스 동작 확인.
    """
    print(f"\n{'='*60}")
    print(f"2D Droplet Advection Demo  {Nx}×{Ny}  t_end={t_end}")
    print(f"{'='*60}")

    eos  = make_nasg_mixture('water_nd', 'air_nd')
    mesh = mesh_2d_rect(Nx, Ny, x_lo=0.0, x_hi=1.0,
                         y_lo=0.0, y_hi=1.0,
                         bc_x='periodic', bc_y='transmissive')

    N    = mesh.n_cells
    x    = mesh.cell_centers[:, 0]
    y    = mesh.cell_centers[:, 1]

    # 원형 액적: 중심 (0.5, 0.5), 반경 0.2
    r2   = (x-0.5)**2 + (y-0.5)**2
    Y0_  = np.exp(-r2 / (2*0.05**2))
    Y0_  = np.clip(Y0_, 0.0, 1.0)
    P_   = np.ones(N); T_   = np.ones(N)
    rho  = eos.rho_from_T_P_Y(Y0_, T_, P_)
    e_   = eos.e_from_T_P_Y(Y0_, T_, P_)
    u0_  = 1.0  # 균일 속도

    U0   = np.zeros((6, N))
    U0[0] = rho*Y0_; U0[1] = rho*(1-Y0_); U0[2] = rho*u0_; U0[5] = rho*(e_+0.5*u0_**2)

    dx_eff = 1.0/Nx
    t, step = 0.0, 0
    U = U0.copy()

    while t < t_end - 1e-14:
        r0,r1,mx,_,_,E = U
        rho_=r0+r1; Y0f=r0/np.maximum(rho_,1e-30)
        _,ux,_,_,_,P,_,c2 = eos.cons_to_prim(r0,r1,mx,np.zeros(N),np.zeros(N),E)
        lam = float(np.max(np.abs(ux)+np.sqrt(c2)))
        dt  = min(0.3*dx_eff/(lam+1e-10), t_end-t)

        def _rhs(U_):
            dU, _ = compute_rhs(U_, mesh, eos, iec=True, weno_order=5,
                                 bc='periodic', use_muscl=True)
            return dU*dt, 0.0

        U, _ = ssprk3_step(U, _rhs)
        t += dt; step += 1
        if not np.all(np.isfinite(U[5])):
            print(f"  NaN at step={step}"); break

    print(f"  완료: t={t:.4f}  steps={step}  finite={np.all(np.isfinite(U[5]))}")

    # 결과 저장
    r0f,r1f,mxf,_,_,Ef = U
    rhof=r0f+r1f; Y0f_=r0f/np.maximum(rhof,1e-30)
    _,uxf,_,_,_,Pf,_,_ = eos.cons_to_prim(r0f,r1f,mxf,np.zeros(N),np.zeros(N),Ef)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    Z0 = Y0f_.reshape(Ny, Nx)
    ZP = Pf.reshape(Ny, Nx)
    axes[0].contourf(Z0, 20, cmap='Blues'); axes[0].set_title('Y₀ (water)')
    axes[1].contourf(ZP, 20, cmap='RdBu'); axes[1].set_title('P (non-dim)')
    for ax in axes: ax.set_aspect('equal')
    plt.suptitle(f'2D Demo  {Nx}×{Ny}  t={t:.3f}')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'gen_2d_demo_{Nx}x{Ny}.png')
    plt.savefig(fname, dpi=100); plt.close()
    print(f"  Saved: {fname}")
    return U


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    N   = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if   cmd == 'droplet':   validate_droplet(N=N)
    elif cmd == 'riemann':   validate_riemann(N=N)
    elif cmd == 'shock':     validate_shock_droplet(N=N)
    elif cmd == 'mach100':   validate_mach100(N=N)
    elif cmd == '2d':        demo_2d_advection(Nx=N, Ny=N)
    elif cmd == 'srk':
        print("SRK EOS 테스트...")
        eos_srk = make_srk_water_air()
        T_test = np.array([300.0])
        P_test = np.array([1e5])
        Y0_test = np.array([0.5])
        rho_srk = eos_srk.rho_from_T_P_Y(Y0_test, T_test, P_test)
        print(f"  SRK rho at T=300K, P=1bar, Y0=0.5: {rho_srk[0]:.4f} kg/m3")
    else:
        validate_all_compare(N_base=N)
