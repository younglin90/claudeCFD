"""
gacid_fvm.py
────────────────────────────────────────────────────────────────────────
G-ACID: Generalised ACID for Face-based FVM / General EOS / nD flows.

Extension of Denner et al. (JCP 2018) ACID to:
  • Abstract EOS interface — plug in any equation of state
  • Face-based FVM mesh (1-D periodic now; extendable to 3-D unstructured)
  • Velocity vector formulation — 1-D scalar or 3-D [ux, uy, uz]
  • Non-conservative GACID energy flux using cell-own ε at both faces
  • Conservative mass + momentum fluxes (standard Rusanov/LLF)
  • SSP-RK3 time integration

GACID energy flux in FVM framework (3-D generalisation)
────────────────────────────────────────────────────────
For face f between owner O and neighbour N, unit normal n̂ (O→N):

  Kinetic energy:  kin_s = ½ ‖u_s‖²  (s = O or N, full 3-D velocity)
  Normal velocity: un_s  = u_s · n̂

  LLF fluxes (conservative):
    F_sp [f]  = Rusanov flux of partial density ρ_p at face f
    F_Un [f]  = Rusanov flux of normal momentum (ρ un) + pressure
    F_Ut [f]  = Rusanov flux of tangential momentum ρ u_t (each component t)

  Cell-centred normal fluxes:
    f_sp_O   = ρ_p[O] · un_O
    f_Un_O   = ρ[O] · un_O² + p[O]
    f_Ut_O   = ρ[O] · u_t[O] · un_O       (per tangential direction t)
    f_sp_N, f_Un_N, f_Ut_N  (same for N)

  GACID correction from owner O's perspective (uses ε[O] at both sides):
    c_p  = ε_p[O] − ½ ‖u_N‖²    (ε[O] + neighbour kinetic for consistency)

    tm_O = Σ_p c_p·(F_sp − f_sp_O) + un_O·(F_Un − f_Un_O) + Σ_t u_t[O]·(F_Ut − f_Ut_O)
    tm1  = Σ_p c_p·(f_sp_N − F_sp) + un_N·(f_Un_N − F_Un) + Σ_t u_t[N]·(f_Ut_N − F_Ut)

    FE_owner_view = ½(FE_cell_O + FE_cell_N) + ½ tm_O − ½ tm1

  Non-conservative RHS for cell c:
    dρE_c/dt = −Σ_{faces of c} [sign_f · FE_c_view(f) · area_f] / vol_c
    where sign_f = +1 (face points outward from c) or −1 (face points inward).

  In 1-D (n̂=[1], un=u, no tangential): formula recovers ACID_1d.py exactly.

Authors : derived from ACID_1d.py (this project) + Denner 2018.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1.  Abstract EOS interface
# ─────────────────────────────────────────────────────────────

class EOS(ABC):
    """Abstract equation of state for n_species partial-density formulation.

    All arrays are shape (N,) for N computational cells.
    `densities` is a list of n_species partial-density arrays [ρ_1, ρ_2, ...].
    """

    @abstractmethod
    def pressure(self, densities, T):
        """p(ρ_1,...,ρ_n, T)."""

    @abstractmethod
    def internal_energy(self, densities, T):
        """ρe(ρ_1,...,ρ_n, T)  — volumetric internal energy."""

    @abstractmethod
    def T_from_rhoe(self, densities, rhoe, T_init=None):
        """Invert ρe(ρ, T) = rhoe_target for T (Newton iteration)."""

    @abstractmethod
    def T_from_p(self, densities, p_target, T_init=None):
        """Invert p(ρ, T) = p_target for T (Newton iteration)."""

    @abstractmethod
    def epsilon(self, densities, T, s):
        """ε_s = (∂ρe/∂ρ_s)_{p, ρ_{j≠s}}  — PE-consistency coefficient."""

    @abstractmethod
    def sound_speed_sq(self, densities, T):
        """c² = speed of sound squared (used for CFL estimate)."""


# ─────────────────────────────────────────────────────────────
# 2.  SRK EOS  (Soave-Redlich-Kwong, binary CH4/N2 mixture)
# ─────────────────────────────────────────────────────────────

class SRK_CH4_N2(EOS):
    """SRK EOS for CH4 (species 0) / N2 (species 1) at supercritical pressures."""

    Ru = 8.314
    _M  = np.array([16.043e-3, 28.014e-3])
    _Tc = np.array([190.56,    126.19   ])
    _pc = np.array([4.599e6,   3.396e6  ])
    _om = np.array([0.0115,    0.0372   ])

    def __init__(self):
        R = self.Ru
        self.a_sp = 0.42748 * R**2 * self._Tc**2 / self._pc
        self.b_sp = 0.08664 * R   * self._Tc    / self._pc
        self.fom  = 0.480 + 1.574*self._om - 0.176*self._om**2
        self.Cv0  = R / self._M * np.array([3.4, 2.5])

    def _mix(self, r1, r2, T):
        r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
        T  = np.asarray(T,  float)
        rho  = r1 + r2
        C0   = r1 / self._M[0];  C1 = r2 / self._M[1]
        Ctot = np.maximum(C0 + C1, 1e-40)
        X0   = C0 / Ctot;        X1 = C1 / Ctot
        Mbar = rho / Ctot

        sq0  = np.sqrt(np.maximum(T / self._Tc[0], 1e-10))
        sq1  = np.sqrt(np.maximum(T / self._Tc[1], 1e-10))
        al0  = (1.0 + self.fom[0]*(1.0 - sq0))**2
        al1  = (1.0 + self.fom[1]*(1.0 - sq1))**2
        da0  = -self.fom[0]*(1.0 + self.fom[0]*(1.0 - sq0)) / (sq0*self._Tc[0] + 1e-300)
        da1  = -self.fom[1]*(1.0 + self.fom[1]*(1.0 - sq1)) / (sq1*self._Tc[1] + 1e-300)

        a01   = np.sqrt(self.a_sp[0]*self.a_sp[1])
        al01  = np.sqrt(np.maximum(al0*al1, 1e-300))
        dal01 = a01*(al1*da0 + al0*da1) / (2.0*al01)

        aA  = X0*X0*self.a_sp[0]*al0 + X1*X1*self.a_sp[1]*al1 + 2.0*X0*X1*a01*al01
        daA = X0*X0*self.a_sp[0]*da0 + X1*X1*self.a_sp[1]*da1 + 2.0*X0*X1*dal01

        b   = X0*self.b_sp[0] + X1*self.b_sp[1]
        v   = Mbar / np.maximum(rho, 1e-30)
        return rho, Mbar, X0, X1, C0, C1, Ctot, b, v, aA, daA

    def pressure(self, densities, T):
        r1, r2 = densities
        _, _, _, _, _, _, _, b, v, aA, _ = self._mix(r1, r2, T)
        vb  = np.maximum(v - b, 1e-20)
        vvb = np.maximum(v*(v + b), 1e-60)
        return self.Ru*T/vb - aA/vvb

    def internal_energy(self, densities, T):
        r1, r2 = densities
        _, _, _, _, _, _, Ctot, b, v, aA, daA = self._mix(r1, r2, T)
        rhoe0 = r1*self.Cv0[0]*T + r2*self.Cv0[1]*T
        arg   = np.where(b > 1e-40, 1.0 + b/np.maximum(v, 1e-30), 1.0)
        dep   = np.where(b > 1e-40,
                         Ctot*(T*daA - aA)/b * np.log(np.maximum(arg, 1e-30)),
                         0.0)
        return rhoe0 + dep

    # ── Finite-difference derivatives ─────────────────────────
    def _dpdT(self, r1, r2, T, h=1.0):
        return (self.pressure([r1,r2], T+h) - self.pressure([r1,r2], T-h)) / (2.0*h)

    def _dpdr(self, r1, r2, T, s, f=5e-4, dmin=0.05):
        dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
        if s == 0:
            return (self.pressure([r1+dr,r2],T) - self.pressure([r1-dr,r2],T)) / (2.0*dr)
        return (self.pressure([r1,r2+dr],T) - self.pressure([r1,r2-dr],T)) / (2.0*dr)

    def _drhoedr(self, r1, r2, T, s, f=5e-4, dmin=0.05):
        dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
        if s == 0:
            return (self.internal_energy([r1+dr,r2],T)
                    - self.internal_energy([r1-dr,r2],T)) / (2.0*dr)
        return (self.internal_energy([r1,r2+dr],T)
                - self.internal_energy([r1,r2-dr],T)) / (2.0*dr)

    def _Cv(self, r1, r2, T, h=1.0):
        return ((self.internal_energy([r1,r2], T+h)
                 - self.internal_energy([r1,r2], T-h)) / (2.0*h)
                / np.maximum(r1 + r2, 1e-30))

    def epsilon(self, densities, T, s):
        r1, r2 = densities
        rho  = r1 + r2
        Cv   = self._Cv(r1, r2, T)
        dpT  = self._dpdT(r1, r2, T)
        dprs = self._dpdr(r1, r2, T, s)
        drhoe_rs = self._drhoedr(r1, r2, T, s)
        return -(rho * Cv / (dpT + 1e-10)) * dprs + drhoe_rs

    def sound_speed_sq(self, densities, T):
        r1, r2 = densities
        rho   = r1 + r2
        Y0    = r1 / np.maximum(rho, 1e-30)
        dpdrT = Y0*self._dpdr(r1,r2,T,0) + (1.0-Y0)*self._dpdr(r1,r2,T,1)
        dpT   = self._dpdT(r1, r2, T)
        Cv    = self._Cv(r1, r2, T)
        return np.maximum(dpdrT + T*dpT**2 / (rho**2*Cv + 1e-30), 100.0)

    def T_from_rhoe(self, densities, rhoe, T_init=None):
        r1, r2 = densities
        r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
        rhoe = np.asarray(rhoe, float)
        T = (np.asarray(T_init, float).copy() if T_init is not None
             else np.full_like(r1, 200.0))
        T = np.clip(T, 10.0, 3000.0)
        h = 1.0
        for _ in range(25):
            f0   = self.internal_energy([r1,r2], T) - rhoe
            dfdT = (self.internal_energy([r1,r2], T+h)
                    - self.internal_energy([r1,r2], T-h)) / (2.0*h)
            dT   = np.clip(-f0 / (dfdT + 1e-6), -200.0, 200.0)
            T    = np.clip(T + dT, 10.0, 3000.0)
            if np.max(np.abs(dT)) < 1e-3:
                break
        return T

    def T_from_p(self, densities, p_target, T_init=None):
        r1, r2 = densities
        r1 = np.asarray(r1, float); r2 = np.asarray(r2, float)
        p_target = np.asarray(p_target, float)
        T = (np.asarray(T_init, float).copy() if T_init is not None
             else np.full_like(r1, 200.0))
        T = np.clip(T, 10.0, 3000.0)
        h = 1.0
        for _ in range(20):
            f0   = self.pressure([r1,r2], T) - p_target
            dfdT = (self.pressure([r1,r2], T+h)
                    - self.pressure([r1,r2], T-h)) / (2.0*h)
            dT   = np.clip(-f0 / (dfdT + 1e-3), -200.0, 200.0)
            T    = np.clip(T + dT, 10.0, 3000.0)
            if np.max(np.abs(dT)) < 5e-2:
                break
        return T


# ─────────────────────────────────────────────────────────────
# 3.  Face-based FVM mesh (1-D periodic)
# ─────────────────────────────────────────────────────────────

class Mesh1D:
    """1-D periodic FVM mesh.

    Cells  : 0 … N-1, volume = dx each.
    Faces  : f=0…N-1, face f connects owner=f to neighbour=(f+1)%N.
    Normal : +x  (unit vector [1,0,0]).
    Area   : 1.0 per face.

    For 3-D extension, replace this class with a mesh that stores
    face_owner, face_neighbor, face_normal (shape n_faces×3), face_area,
    and cell_volume as 1-D arrays of length n_cells.
    """

    def __init__(self, N, L=1.0):
        self.N           = N
        self.L           = L
        self.dx          = L / N
        self.x           = np.linspace(self.dx/2, L - self.dx/2, N)
        self.cell_volume = np.full(N, self.dx)
        self.n_faces     = N
        # face f: owner = f, neighbour = (f+1)%N
        self.face_owner    = np.arange(N, dtype=int)
        self.face_neighbor = (np.arange(N) + 1) % N
        # For 3-D: face_normal would be shape (n_faces, 3)
        # Here all normals are +x  (stored as scalar 1.0 for efficiency)
        self.face_normal_x = np.ones(N)   # normal dot x = +1 for all faces


# ─────────────────────────────────────────────────────────────
# 4.  MUSCL reconstruction helpers
# ─────────────────────────────────────────────────────────────

def _muscl_lr(q):
    """Minmod MUSCL: left/right states at all faces of a 1-D periodic mesh.

    Returns qL[f], qR[f] = left and right states at face f
    (face f connects cell f to cell (f+1)%N).

    For 3-D: apply this per-direction gradient on each face independently,
    or use a Green-Gauss gradient reconstruction with a Barth-Jespersen limiter.
    """
    dR  = np.roll(q, -1) - q          # q[m+1] - q[m]
    dL  = q - np.roll(q,  1)          # q[m]   - q[m-1]
    slp = np.where(dL*dR > 0.0,
                   np.sign(dL)*np.minimum(np.abs(dL), np.abs(dR)),
                   0.0)
    return q + 0.5*slp, np.roll(q - 0.5*slp, -1)


# ─────────────────────────────────────────────────────────────
# 5.  GACID flux kernel  (1-D; see docstring for 3-D notes)
# ─────────────────────────────────────────────────────────────

def _gacid_fluxes_1d(densities, u, T, p, rhoe, eos, lam_cell, scheme):
    """Compute all numerical fluxes for a 1-D periodic FVM mesh.

    Parameters
    ----------
    densities : [r1, r2]  partial densities  shape (N,)
    u         : velocity  shape (N,)
    T, p, rhoe: temperature, pressure, volumetric internal energy  shape (N,)
    eos       : EOS instance
    lam_cell  : cell-level wave speed |u| + c  shape (N,)
    scheme    : 'FC' | 'APEC' | 'GACID'

    Returns
    -------
    F_mass : list [F1, F2]  conservative mass fluxes at faces  shape (N,)
    F_mom  : conservative momentum flux at faces                shape (N,)
    FE_conservative : energy flux for FC/APEC (conservative)    shape (N,) or None
    FE_left  : GACID energy flux at face f from cell f's view   shape (N,) or None
    FE_right : GACID energy flux at face f-1 from cell f's view shape (N,) or None

    For FC/APEC: FE_left = FE_conservative, FE_right = None.
    For GACID  : FE_conservative = None, FE_left and FE_right are non-conservative.

    ── 3-D extension notes ──────────────────────────────────────────────────────
    In 3-D at face f with unit normal n̂:

      • Replace scalar u with normal component un = u · n̂.
      • LLF wave speed: λ_f = max(|un_O|+c_O, |un_N|+c_N).
      • F_mass[p] stays the same formula with u → un.
      • F_mom (normal + tangential):
          F_Un = Rusanov(ρun, un, p)                (normal momentum)
          F_Ut = 0.5*(ρut_L*unL + ρut_R*unR) - 0.5λ*(ρut_R - ρut_L)  (per tangential t)
      • Energy correction generalises to:
          tm_3d = Σ_p c_p*(F_sp − ρ_p[O]*un_O)
                + un_O*(F_Un − (ρ_O*un_O²+p_O))
                + Σ_t u_t[O]*(F_Ut − ρ_O*u_t[O]*un_O)
        where c_p = ε_p[O] − ½‖u_N‖²  (ACID: ε[O], neighbour kinetic energy).
      • FE_cell = (ρE+p)*un  (enthalpy flux normal to face).
      • This module can be called in a face loop for unstructured 3-D meshes.
    ─────────────────────────────────────────────────────────────────────────────
    """
    r1, r2 = densities
    rho  = r1 + r2
    rhoE = rhoe + 0.5*rho*u**2

    # ── MUSCL reconstruction ──────────────────────────────────
    r1L, r1R = _muscl_lr(r1)
    r2L, r2R = _muscl_lr(r2)
    uL,  uR  = _muscl_lr(u)
    pL,  pR  = _muscl_lr(p)
    rhoL = r1L + r2L
    rhoR = r1R + r2R

    # ── Rusanov (LLF) wave speed ──────────────────────────────
    lam = np.maximum(lam_cell, np.roll(lam_cell, -1))

    # ── Conservative mass & momentum fluxes ──────────────────
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)
    FU = 0.5*(rhoL*uL**2+pL + rhoR*uR**2+pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    # ── Energy flux ───────────────────────────────────────────
    if scheme == 'FC':
        # Standard MUSCL-LLF: left state from T[m], right state from T[m+1]
        rhoEL = eos.internal_energy([r1L, r2L], T) + 0.5*rhoL*uL**2
        rhoER = eos.internal_energy([r1R, r2R], np.roll(T, -1)) + 0.5*rhoR*uR**2
        FE = 0.5*((rhoEL+pL)*uL + (rhoER+pR)*uR) - 0.5*lam*(rhoER - rhoEL)
        return [F1, F2], FU, FE, None, None

    # ── Cell-centred fluxes (used by both APEC and GACID) ────
    F1_cell = r1 * u
    F2_cell = r2 * u
    FU_cell = rho * u**2 + p
    FE_cell = (rhoE + p) * u
    FE_m1   = np.roll(FE_cell, -1)   # cell (m+1) centred energy flux
    FE_mm1  = np.roll(FE_cell,  1)   # cell (m-1) centred energy flux

    # ── ε coefficients ────────────────────────────────────────
    eps0 = eos.epsilon([r1, r2], T, 0)
    eps1 = eos.epsilon([r1, r2], T, 1)

    u_m1  = np.roll(u, -1)   # u[m+1]
    u_mm1 = np.roll(u,  1)   # u[m-1]

    if scheme == 'APEC':
        # Terashima 2025 Appendix A Eq. A.4 — symmetric APEC
        eps0_m1 = np.roll(eps0, -1)
        eps1_m1 = np.roll(eps1, -1)

        c0m  = eps0    - 0.5*u**2
        c1m  = eps1    - 0.5*u**2
        c0m1 = eps0_m1 - 0.5*u_m1**2
        c1m1 = eps1_m1 - 0.5*u_m1**2

        tm  = (c0m *(F1 - F1_cell)              + c1m *(F2 - F2_cell)
               + u   *(FU - FU_cell))
        tm1 = (c0m1*(np.roll(F1_cell,-1) - F1)  + c1m1*(np.roll(F2_cell,-1) - F2)
               + u_m1*(np.roll(FU_cell,-1) - FU))

        FE = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1
        return [F1, F2], FU, FE, None, None

    # ── GACID: non-conservative, cell m uses ε[m] at BOTH faces ──
    # ── Face m+1/2 from cell m (left cell, owner view) ───────
    c0m = eps0 - 0.5*u**2       # ε[m], cell m velocity
    c1m = eps1 - 0.5*u**2

    # tm: left-cell correction using ε[m] and u[m]
    tm = c0m*(F1 - F1_cell) + c1m*(F2 - F2_cell) + u*(FU - FU_cell)

    # tm1_acid: right-cell correction using ε[m] but u[m+1] (kinetic consistency)
    # In 3-D: replace ½u_m1² with ½‖u_N‖² (full velocity norm of neighbour)
    c0m_right = eps0 - 0.5*u_m1**2
    c1m_right = eps1 - 0.5*u_m1**2
    tm1_acid  = (c0m_right*(np.roll(F1_cell,-1) - F1)
                 + c1m_right*(np.roll(F2_cell,-1) - F2)
                 + u_m1*(np.roll(FU_cell,-1) - FU))

    FE_left = 0.5*(FE_cell + FE_m1) + 0.5*tm - 0.5*tm1_acid

    # ── Face m-1/2 from cell m (right cell, neighbour view) ──
    F1_mm       = np.roll(F1, 1)           # face flux at m-1/2
    F2_mm       = np.roll(F2, 1)
    FU_mm       = np.roll(FU, 1)
    F1_cell_mm1 = np.roll(F1_cell, 1)      # F1_cell[m-1]
    F2_cell_mm1 = np.roll(F2_cell, 1)
    FU_cell_mm1 = np.roll(FU_cell, 1)

    # tm_mm_acid: left-cell correction at m-1/2 using ε[m] and u[m-1]
    # (cell m-1 is the left cell, but we substitute ε[m]; kinetic uses u[m-1])
    c0m_left = eps0 - 0.5*u_mm1**2
    c1m_left = eps1 - 0.5*u_mm1**2
    tm_mm_acid = (c0m_left*(F1_mm - F1_cell_mm1)
                  + c1m_left*(F2_mm - F2_cell_mm1)
                  + u_mm1*(FU_mm - FU_cell_mm1))

    # tm1_mm: right-cell correction at m-1/2: cell m is right, use ε[m] with u[m]
    tm1_mm = c0m*(F1_cell - F1_mm) + c1m*(F2_cell - F2_mm) + u*(FU_cell - FU_mm)

    FE_right = 0.5*(FE_mm1 + FE_cell) + 0.5*tm_mm_acid - 0.5*tm1_mm

    return [F1, F2], FU, None, FE_left, FE_right


# ─────────────────────────────────────────────────────────────
# 6.  FVM Solver (1-D, periodic, SSP-RK3)
# ─────────────────────────────────────────────────────────────

class GACIDSolver1D:
    """FVM solver for 1-D multicomponent compressible flow using GACID.

    State vector per cell: [r1, r2, ρu, ρE]  (conservative)
    Schemes: 'FC', 'APEC', 'GACID'

    Design for 3-D extension
    ─────────────────────────
    Replace `mesh` with a 3-D Mesh object whose face arrays have shape (n_faces,).
    In `_rhs`, loop over faces (or vectorise per-direction for structured grids).
    For each face f:
      • Gather owner/neighbour states.
      • Apply per-face MUSCL reconstruction (Green-Gauss gradient + limiter).
      • Call `_gacid_fluxes_3d(...)` with face normal n̂_f and face area A_f.
      • Accumulate: dρE_owner -= FE_owner_view * A_f / vol_owner
                    dρE_neighbor += FE_neighbor_view * A_f / vol_neighbor
    """

    def __init__(self, eos, mesh):
        self.eos  = eos
        self.mesh = mesh

    # ── Primitive variables from conservative state ───────────
    def _prim(self, densities, rhoU, rhoE, T_prev):
        r1, r2 = densities
        rho  = r1 + r2
        u    = rhoU / np.maximum(rho, 1e-30)
        rhoe = rhoE - 0.5*rho*u**2
        T    = self.eos.T_from_rhoe([r1, r2], rhoe, T_init=T_prev)
        p    = self.eos.pressure([r1, r2], T)
        return u, rhoe, T, p

    # ── RHS: returns dU/dt and current T, p ──────────────────
    def _rhs(self, state, T_prev, scheme):
        r1, r2, rhoU, rhoE = state
        u, rhoe, T, p = self._prim([r1, r2], rhoU, rhoE, T_prev)

        lam_c = np.abs(u) + np.sqrt(self.eos.sound_speed_sq([r1, r2], T))

        F_mass, F_mom, FE_con, FE_left, FE_right = _gacid_fluxes_1d(
            [r1, r2], u, T, p, rhoe, self.eos, lam_c, scheme)

        F1, F2 = F_mass
        dx = self.mesh.dx

        d1 = -(F1      - np.roll(F1, 1)) / dx   # conservative
        d2 = -(F2      - np.roll(F2, 1)) / dx   # conservative
        dU = -(F_mom   - np.roll(F_mom, 1)) / dx  # conservative

        if scheme == 'GACID':
            # Non-conservative energy: each cell uses its own energy flux views
            dE = -(FE_left - FE_right) / dx
        else:
            dE = -(FE_con - np.roll(FE_con, 1)) / dx

        return [d1, d2, dU, dE], T, p

    # ── SSP-RK3 step ─────────────────────────────────────────
    @staticmethod
    def _clip(state):
        return [np.maximum(state[0], 0.0), np.maximum(state[1], 0.0),
                state[2], state[3]]

    def _step(self, state, T_prev, dt, scheme):
        def stage(s, T):
            d, T_, p_ = self._rhs(s, T, scheme)
            return d, T_, p_

        k1, T1, p1 = stage(state, T_prev)
        S1 = self._clip([state[i] + dt*k1[i] for i in range(4)])

        k2, T2, p2 = stage(S1, T1)
        S2 = self._clip([0.75*state[i] + 0.25*(S1[i] + dt*k2[i]) for i in range(4)])

        k3, T3, p3 = stage(S2, T2)
        Snew = self._clip([(1/3)*state[i] + (2/3)*(S2[i] + dt*k3[i]) for i in range(4)])
        return Snew, T3, p3

    # ── Main runner ───────────────────────────────────────────
    def run(self, scheme, t_end=0.07, CFL=0.3, p_inf=5e6, k=15.0, verbose=True):
        mesh = self.mesh
        eos  = self.eos
        dx   = mesh.dx

        if verbose:
            print(f"\n{'='*55}")
            print(f"Scheme: {scheme}  N={mesh.N}  t_end={t_end:.4f}  CFL={CFL}  k={k}")

        r1, r2, u, rhoE, T, p = _initial_condition_1d(mesh.x, eos, p_inf, k)
        state = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
        rhoE0 = rhoE.copy()

        t_hist, pe_hist, en_hist = [0.0], [_pe_err(p, p_inf)], [0.0]
        t, step, diverged = 0.0, 0, False

        while t < t_end - 1e-14:
            r1_, r2_ = state[0], state[1]
            rhoU_    = state[2]
            rho_     = r1_ + r2_
            u_       = rhoU_ / np.maximum(rho_, 1e-30)
            lam      = float(np.max(np.abs(u_) + np.sqrt(eos.sound_speed_sq([r1_,r2_], T))))
            dt       = min(CFL*dx/(lam + 1e-10), t_end - t)

            try:
                state, T, p = self._step(state, T, dt, scheme)
            except Exception as e:
                if verbose:
                    print(f"  Exception at t={t:.5f}: {e}")
                diverged = True
                break

            t += dt; step += 1
            pe_ = _pe_err(p, p_inf)
            en_ = float(abs(np.sum(state[3]) - np.sum(rhoE0))
                        / (abs(np.sum(rhoE0)) + 1e-30))
            t_hist.append(t); pe_hist.append(pe_); en_hist.append(en_)

            if not np.isfinite(pe_) or pe_ > 5.0:
                if verbose:
                    print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}")
                diverged = True
                break

        if verbose:
            status = 'Completed' if not diverged else 'Diverged'
            print(f"  --> {status} at t={t:.5f}  ({step} steps)")

        return (mesh.x, state, T, p,
                np.array(t_hist), np.array(pe_hist), np.array(en_hist),
                diverged)


# ─────────────────────────────────────────────────────────────
# 7.  Initial condition + diagnostics
# ─────────────────────────────────────────────────────────────

def _initial_condition_1d(x, eos, p_inf=5e6, k=15.0):
    """CH4 blob in N2: tanh profile, uniform p=p_inf, uniform u=100 m/s."""
    N, xc, rc = len(x), 0.5, 0.25
    r1_inf, r2_inf = 400.0, 100.0
    r = np.abs(x - xc)
    r1 = 0.5*r1_inf*(1.0 - np.tanh(k*(r - rc)))
    r2 = 0.5*r2_inf*(1.0 + np.tanh(k*(r - rc)))

    print("  Solving T from p_inf=5MPa ...", flush=True)
    T = np.full(N, 200.0)
    for _ in range(60):
        ph   = eos.pressure([r1, r2], T)
        dpdT = (eos.pressure([r1, r2], T+1.0)
                - eos.pressure([r1, r2], T-1.0)) / 2.0
        dT   = np.clip(-(ph - p_inf) / (dpdT + 1e-3), -100.0, 100.0)
        T    = np.clip(T + dT, 10.0, 2000.0)
        if np.max(np.abs(dT)) < 1e-3:
            break

    p    = eos.pressure([r1, r2], T)
    rhoe = eos.internal_energy([r1, r2], T)
    u    = np.full(N, 100.0)
    return r1, r2, u, rhoe + 0.5*(r1+r2)*u**2, T, p


def _pe_err(p, p0=5e6):
    return float(np.max(np.abs(p - p0)) / p0)


# ─────────────────────────────────────────────────────────────
# 8.  Validation tests (match ACID_1d.py results)
# ─────────────────────────────────────────────────────────────

def run_scheme(scheme, N=101, t_end=0.07, CFL=0.3, k=15.0):
    """Convenience wrapper: creates SRK EOS + Mesh1D and runs one scheme."""
    eos    = SRK_CH4_N2()
    mesh   = Mesh1D(N)
    solver = GACIDSolver1D(eos, mesh)
    return solver.run(scheme, t_end=t_end, CFL=CFL, k=k)


def compare_resolution(CFL=0.3, k=15.0):
    """1-step PE vs resolution — should match ACID_1d.py reference values:
    N=101: FC≈7.6e-3  APEC≈1.3e-3  GACID≈2.7e-4
    """
    print("\n[Test 2] Resolution study  k=%.0f" % k)
    Ns = [51, 101, 201, 501]
    header = f"{'N':>6}  {'FC PE':>12}  {'APEC PE':>12}  {'GACID PE':>12}"
    print("  " + header)
    print("  " + "-"*len(header))
    eos = SRK_CH4_N2()
    rows = []
    for N in Ns:
        mesh = Mesh1D(N)
        dx   = mesh.dx
        r1, r2, u, rhoE, T, p0 = _initial_condition_1d(mesh.x, eos, k=k)
        state = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
        solver = GACIDSolver1D(eos, mesh)
        row = [N]
        for sc in ('FC', 'APEC', 'GACID'):
            lam = float(np.max(np.abs(u) + np.sqrt(eos.sound_speed_sq([r1,r2], T))))
            dt  = CFL * dx / (lam + 1e-10)
            s1, T1, p1 = solver._step(state, T, dt, sc)
            row.append(_pe_err(p1))
        rows.append(row)
        fc_, ap_, gc_ = row[1], row[2], row[3]
        print(f"  {N:>6}  {fc_:>12.3e}  {ap_:>12.3e}  {gc_:>12.3e}")
    return rows


def compare_pe_time(N=101, t_end=0.055, CFL=0.3, k=15.0):
    """PE vs time for FC / APEC / GACID — should match ACID_1d.py results."""
    print("\n[Test 1] PE vs time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    results = {}
    labels  = {'FC': 'FC', 'APEC': 'APEC', 'GACID': 'GACID (G-ACID)'}
    for sc in ('FC', 'APEC', 'GACID'):
        x, state, T, p, th, ph, eh, div = run_scheme(sc, N=N, t_end=t_end,
                                                      CFL=CFL, k=k)
        results[sc] = (th, ph, div)
        print(f"  {labels[sc]:18s}  PE(t=1step)={ph[1]:.3e}  "
              f"PE(final)={ph[-1]:.3e}  diverged={div}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'GACID': 'tab:green'}
    for sc, (th, ph, div) in results.items():
        ax.semilogy(th*1e3, ph, label=labels[sc], color=colors[sc])
    ax.set_xlabel('t  [ms]')
    ax.set_ylabel('max |Δp/p₀|')
    ax.set_title(f'G-ACID pressure-equilibrium error  N={N}  k={k}  CFL={CFL}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'gacid_pe_time_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")
    return results


def compare_divergence(N=101, t_end=0.055, CFL=0.3, k=15.0):
    """Divergence time comparison — should match ACID_1d.py:
    FC: 55ms (no diverge), APEC: ~50ms (diverge), GACID: 55ms (no diverge).
    """
    print("\n[Test 3] Divergence time  N=%d  CFL=%.2f  k=%.0f" % (N, CFL, k))
    for sc in ('FC', 'APEC', 'GACID'):
        _, _, _, _, th, ph, _, div = run_scheme(sc, N=N, t_end=t_end,
                                                CFL=CFL, k=k)
        t_div = th[-1]
        print(f"  {sc:6s}  t_diverge={t_div*1e3:.2f} ms  diverged={div}")


def plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=3e-3):
    """Snapshot profiles: density, pressure, velocity."""
    print(f"\n[Test 4] Profiles at t={t_snap*1e3:.1f} ms  N={N}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {'FC': 'tab:blue', 'APEC': 'tab:orange', 'GACID': 'tab:green'}

    for sc in ('FC', 'APEC', 'GACID'):
        x, state, T, p, *_ = run_scheme(sc, N=N, t_end=t_snap, CFL=CFL, k=k)
        r1, r2, rhoU, rhoE = state
        rho = r1 + r2
        u_  = rhoU / np.maximum(rho, 1e-30)
        axes[0].plot(x, rho, label=sc, color=colors[sc])
        axes[1].plot(x, p/1e6, color=colors[sc])
        axes[2].plot(x, u_, color=colors[sc])

    for ax, lbl in zip(axes, ['ρ [kg/m³]', 'p [MPa]', 'u [m/s]']):
        ax.set_xlabel('x [m]')
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_title(f'G-ACID profiles  t={t_snap*1e3:.1f}ms  N={N}')
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f'gacid_profiles_N{N}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ─────────────────────────────────────────────────────────────
# 9.  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    compare_resolution(CFL=0.3, k=15.0)
    compare_pe_time(N=101, t_end=0.055, CFL=0.3, k=15.0)
    compare_divergence(N=101, t_end=0.055, CFL=0.3, k=15.0)
    plot_profiles(N=101, CFL=0.3, k=15.0, t_snap=3e-3)
    print("\nDone.")
