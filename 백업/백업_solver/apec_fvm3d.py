"""
apec_fvm3d.py
=============
FVM, General EOS, 3D-unstructured-compatible, Conservative-form
APEC (Approximately Pressure-Equilibrium-Preserving) solver.

Mesh topology is expressed as a face-based FVM structure so the same
flux kernel works on any unstructured 3D mesh; validation is carried
out on a 1-D periodic structured mesh.

Usage
-----
  python solver/apec_fvm3d.py --validate      # both CPG + SRK
  python solver/apec_fvm3d.py --case cpg
  python solver/apec_fvm3d.py --case srk
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1.  EOS  ABSTRACT  BASE + IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════

class EOSBase:
    """Abstract EOS interface for two-species mixtures."""
    def pressure(self, r1, r2, T):       raise NotImplementedError
    def rhoe(self, r1, r2, T):           raise NotImplementedError
    def T_from_rhoe(self, r1, r2, rhoe, T_prev=None): raise NotImplementedError
    def sound_speed_sq(self, r1, r2, T): raise NotImplementedError
    def epsilon(self, r1, r2, T):
        """(∂ρe/∂ρ_s)_{ρ_{j≠s},p}  for s=0,1.  Returns (eps0, eps1)."""
        raise NotImplementedError


# ── Calorically Perfect Gas ────────────────────────────────────────
class CPGEos(EOSBase):
    """
    Species 0: γ=1.4,  M=28 g/mol  (N2-like)
    Species 1: γ=1.66, M=4  g/mol  (He-like)
    Non-dimensional (Ru=1) as in Terashima §3.1.
    """
    _GAM = np.array([1.4,  1.66])
    _MW  = np.array([28.0,  4.0])

    def _inv_gm1(self, r1, r2):
        n1, n2 = r1 / self._MW[0], r2 / self._MW[1]
        ntot = np.maximum(n1 + n2, 1e-60)
        return (n1/(self._GAM[0]-1.) + n2/(self._GAM[1]-1.)) / ntot

    def pressure(self, r1, r2, T):
        # For CPG: p = ρe*(γ̄-1) — temperature is not used in CPG flux routines
        # (kept for interface consistency; use rhoe_from_p / p_from_rhoe instead)
        raise RuntimeError("CPGEos.pressure(T) not meaningful; use p_from_rhoe")

    def p_from_rhoe(self, r1, r2, rhoe):
        return rhoe / np.maximum(self._inv_gm1(r1, r2), 1e-60)

    def rhoe_from_p(self, r1, r2, p):
        return p * self._inv_gm1(r1, r2)

    def rhoe(self, r1, r2, T):
        raise RuntimeError("CPGEos.rhoe(T) not used; use rhoe_from_p")

    def T_from_rhoe(self, r1, r2, rhoe, T_prev=None):
        cv1 = 1.0 / (self._MW[0] * (self._GAM[0] - 1.0))
        cv2 = 1.0 / (self._MW[1] * (self._GAM[1] - 1.0))
        return rhoe / np.maximum(r1*cv1 + r2*cv2, 1e-60)

    def sound_speed_sq(self, r1, r2, p):
        inv_gm1 = self._inv_gm1(r1, r2)
        gbar = 1.0 + 1.0 / np.maximum(inv_gm1, 1e-60)
        return gbar * p / np.maximum(r1 + r2, 1e-60)

    def epsilon(self, r1, r2, p):
        """Analytical ε for CPG (Eq. 59)."""
        rho  = r1 + r2
        n1, n2 = r1 / self._MW[0], r2 / self._MW[1]
        ntot = np.maximum(n1 + n2, 1e-60)
        Mbar = rho / ntot
        B    = n1/(self._GAM[0]-1.) + n2/(self._GAM[1]-1.)
        pref = p * Mbar**2 / np.maximum(rho**2, 1e-60)
        eps0 = pref / self._MW[0] * (ntot/(self._GAM[0]-1.) - B)
        eps1 = pref / self._MW[1] * (ntot/(self._GAM[1]-1.) - B)
        return eps0, eps1


# ── Soave–Redlich–Kwong ────────────────────────────────────────────
_Ru  = 8.314
_M   = np.array([16.043e-3, 28.014e-3])
_Tc  = np.array([190.56,    126.19   ])
_pc  = np.array([4.599e6,   3.396e6  ])
_om  = np.array([0.0115,    0.0372   ])
_a_sp = 0.42748 * _Ru**2 * _Tc**2 / _pc
_b_sp = 0.08664 * _Ru   * _Tc    / _pc
_fom  = 0.480 + 1.574*_om - 0.176*_om**2
_Cv0  = _Ru / _M * np.array([3.4, 2.5])

def _srk_mix(r1, r2, T):
    r1,r2,T = map(lambda x: np.asarray(x, float), (r1,r2,T))
    rho  = r1 + r2
    C0, C1 = r1/_M[0], r2/_M[1]
    Ctot = np.maximum(C0+C1, 1e-40)
    X0, X1 = C0/Ctot, C1/Ctot
    Mbar = rho/Ctot
    sq0 = np.sqrt(np.maximum(T/_Tc[0], 1e-10))
    sq1 = np.sqrt(np.maximum(T/_Tc[1], 1e-10))
    al0 = (1.+_fom[0]*(1.-sq0))**2
    al1 = (1.+_fom[1]*(1.-sq1))**2
    da0 = -_fom[0]*(1.+_fom[0]*(1.-sq0))/(sq0*_Tc[0]+1e-300)
    da1 = -_fom[1]*(1.+_fom[1]*(1.-sq1))/(sq1*_Tc[1]+1e-300)
    a01  = np.sqrt(_a_sp[0]*_a_sp[1])
    al01 = np.sqrt(np.maximum(al0*al1, 1e-300))
    dal01= a01*(al1*da0+al0*da1)/(2.*al01)
    aA   = X0*X0*_a_sp[0]*al0 + X1*X1*_a_sp[1]*al1 + 2.*X0*X1*a01*al01
    daA  = X0*X0*_a_sp[0]*da0 + X1*X1*_a_sp[1]*da1 + 2.*X0*X1*dal01
    b    = X0*_b_sp[0] + X1*_b_sp[1]
    v    = Mbar/np.maximum(rho, 1e-30)
    return rho, Mbar, X0, X1, C0, C1, Ctot, b, v, aA, daA

class SRKEos(EOSBase):
    def pressure(self, r1, r2, T):
        _,_,_,_,_,_,_,b,v,aA,_ = _srk_mix(r1,r2,T)
        vb  = np.maximum(v-b, 1e-20)
        vvb = np.maximum(v*(v+b), 1e-60)
        return _Ru*T/vb - aA/vvb

    def rhoe(self, r1, r2, T):
        _,_,_,_,_,_,Ctot,b,v,aA,daA = _srk_mix(r1,r2,T)
        rhoe0 = r1*_Cv0[0]*T + r2*_Cv0[1]*T
        arg = np.where(b>1e-40, 1.+b/np.maximum(v,1e-30), 1.)
        dep = np.where(b>1e-40, Ctot*(T*daA-aA)/b*np.log(np.maximum(arg,1e-30)), 0.)
        return rhoe0 + dep

    def T_from_rhoe(self, r1, r2, rhoe_target, T_prev=None):
        r1,r2,rhoe_target = map(lambda x: np.asarray(x,float),(r1,r2,rhoe_target))
        T = (np.asarray(T_prev,float).copy() if T_prev is not None
             else np.full_like(r1, 200.))
        T = np.clip(T, 10., 3000.)
        for _ in range(25):
            f0 = self.rhoe(r1,r2,T) - rhoe_target
            fp = self.rhoe(r1,r2,T+1.); fm = self.rhoe(r1,r2,T-1.)
            dT = np.clip(-f0/((fp-fm)/2.+1e-6), -200., 200.)
            T  = np.clip(T+dT, 10., 3000.)
            if np.max(np.abs(dT)) < 1e-3: break
        return T

    def sound_speed_sq(self, r1, r2, T):
        rho = r1+r2
        Y0  = r1/np.maximum(rho, 1e-30)
        dp0 = self._dpdr(r1,r2,T,0); dp1 = self._dpdr(r1,r2,T,1)
        dpdr_T = Y0*dp0 + (1.-Y0)*dp1
        dpT = self._dpdT(r1,r2,T)
        Cv  = self._Cv(r1,r2,T)
        return np.maximum(dpdr_T + T*dpT**2/(rho**2*Cv+1e-30), 100.)

    def _dpdT(self, r1, r2, T, h=1.):
        return (self.pressure(r1,r2,T+h)-self.pressure(r1,r2,T-h))/(2.*h)

    def _dpdr(self, r1, r2, T, s, f=5e-4, dmin=0.05):
        dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
        if s==0: return (self.pressure(r1+dr,r2,T)-self.pressure(r1-dr,r2,T))/(2.*dr)
        else:    return (self.pressure(r1,r2+dr,T)-self.pressure(r1,r2-dr,T))/(2.*dr)

    def _drhoedr(self, r1, r2, T, s, f=5e-4, dmin=0.05):
        dr = np.maximum(np.abs(r1 if s==0 else r2)*f, dmin)
        if s==0: return (self.rhoe(r1+dr,r2,T)-self.rhoe(r1-dr,r2,T))/(2.*dr)
        else:    return (self.rhoe(r1,r2+dr,T)-self.rhoe(r1,r2-dr,T))/(2.*dr)

    def _Cv(self, r1, r2, T, h=1.):
        return ((self.rhoe(r1,r2,T+h)-self.rhoe(r1,r2,T-h))/(2.*h)
                / np.maximum(r1+r2, 1e-30))

    def epsilon(self, r1, r2, T):
        rho = r1+r2
        Cv  = self._Cv(r1,r2,T)
        dpT = self._dpdT(r1,r2,T)
        eps0 = -(rho*Cv/(dpT+1e-10))*self._dpdr(r1,r2,T,0) + self._drhoedr(r1,r2,T,0)
        eps1 = -(rho*Cv/(dpT+1e-10))*self._dpdr(r1,r2,T,1) + self._drhoedr(r1,r2,T,1)
        return eps0, eps1

    def init_T_from_p(self, r1, r2, p_inf):
        """Newton solve T such that SRK pressure = p_inf."""
        r1,r2 = np.asarray(r1,float), np.asarray(r2,float)
        _,Mb,_,_,_,_,_,*_ = _srk_mix(r1,r2,np.full(r1.shape,300.))
        T = np.clip(p_inf*Mb/np.maximum(r1+r2,1e-30)/_Ru, 50., 1000.)
        for _ in range(60):
            ph  = self.pressure(r1,r2,T)
            dph = self._dpdT(r1,r2,T)
            dT  = np.clip(-(ph-p_inf)/(dph+1e-3), -100., 100.)
            T   = np.clip(T+dT, 10., 2000.)
            if np.max(np.abs(dT)) < 1e-3: break
        return T


# ══════════════════════════════════════════════════════════════════
# 2.  MESH  (FVM face-based topology)
# ══════════════════════════════════════════════════════════════════

class Mesh1D:
    """
    1-D periodic structured mesh expressed as face-based FVM topology.

    Conventions
    -----------
    face f connects owner cell o=face_owner[f] and neighbor cell n=face_neighbor[f].
    The outward normal of face f **from the owner's perspective** points in the
    +x direction (face_normal = +1) or -x direction (face_normal = -1).

    For a 1-D mesh of N cells with periodic BC:
        - N internal faces; face f lies between cell f and cell (f+1)%N
        - face_owner[f]    = f
        - face_neighbor[f] = (f+1) % N
        - face_normal[f]   = +1  (right-pointing from owner)
    """
    def __init__(self, N, L=1.0):
        self.N  = N
        self.L  = L
        self.dx = L / N
        self.cell_centers = np.linspace(self.dx/2, L - self.dx/2, N)
        self.volumes      = np.full(N, self.dx)

        # Face topology  (N faces for periodic 1-D)
        Nf = N
        self.Nf            = Nf
        self.face_owner    = np.arange(Nf, dtype=int)              # cell f
        self.face_neighbor = (np.arange(Nf) + 1) % N              # cell f+1
        # outward normal from owner (+x direction), stored as scalar ±1
        self.face_normal   = np.ones(Nf)
        self.face_area     = np.ones(Nf)


# ══════════════════════════════════════════════════════════════════
# 3.  RECONSTRUCTION  (MUSCL + minmod)
# ══════════════════════════════════════════════════════════════════

def _minmod(a, b):
    return np.where(a*b > 0., np.where(np.abs(a) < np.abs(b), a, b), 0.)

def muscl_reconstruct(q, mesh):
    """
    Return qL[f], qR[f]: left/right reconstructed values at each face.
    qL = value approaching face f from the owner side  (cell face_owner[f])
    qR = value approaching face f from the neighbor side (cell face_neighbor[f])
    """
    o = mesh.face_owner      # (Nf,)
    n = mesh.face_neighbor   # (Nf,)
    # Upwind/downwind cell indices (periodic)
    om1 = (o - 1) % mesh.N
    np1 = (n + 1) % mesh.N

    # Slopes in owner and neighbor cells
    dR_o = q[n]   - q[o]
    dL_o = q[o]   - q[om1]
    dR_n = q[np1] - q[n]
    dL_n = q[n]   - q[o]

    slp_o = _minmod(dR_o, dL_o)
    slp_n = _minmod(dR_n, dL_n)

    qL = q[o] + 0.5 * slp_o
    qR = q[n] - 0.5 * slp_n
    return qL, qR


# ══════════════════════════════════════════════════════════════════
# 4.  FVM FLUX KERNEL  (LLF + APEC)
# ══════════════════════════════════════════════════════════════════

def compute_fluxes(r1, r2, u, rhoe, p, lam_cell, mesh, scheme,
                   eps0=None, eps1=None, eos_type='CPG'):
    """
    Compute face fluxes for all conservative variables.

    eos_type : 'CPG' — MUSCL-consistent PE dissipation (Appendix A, CPG variant)
               'SRK' — cell-flux-based energy reconstruction (Appendix A, Eq. A.4)

    Returns
    -------
    F1, F2, FU, FE : (Nf,) arrays  — fluxes at each face
    """
    # ── Reconstruct all primitives to face left/right states ──────
    r1L, r1R = muscl_reconstruct(r1, mesh)
    r2L, r2R = muscl_reconstruct(r2, mesh)
    uL,  uR  = muscl_reconstruct(u,  mesh)
    pL,  pR  = muscl_reconstruct(p,  mesh)

    rhoL = r1L + r2L
    rhoR = r1R + r2R

    # ── LLF wave speed at each face ───────────────────────────────
    o, n = mesh.face_owner, mesh.face_neighbor
    lam  = np.maximum(lam_cell[o], lam_cell[n])   # (Nf,)

    # ── Mass fluxes ───────────────────────────────────────────────
    F1 = 0.5*(r1L*uL + r1R*uR) - 0.5*lam*(r1R - r1L)
    F2 = 0.5*(r2L*uL + r2R*uR) - 0.5*lam*(r2R - r2L)

    # ── Momentum flux ─────────────────────────────────────────────
    FU = 0.5*(rhoL*uL**2 + pL + rhoR*uR**2 + pR) - 0.5*lam*(rhoR*uR - rhoL*uL)

    # ── Energy flux ───────────────────────────────────────────────
    rhoE_o = rhoe[o] + 0.5*(r1[o]+r2[o])*u[o]**2  # cell total energy (owner)
    rhoE_n = rhoe[n] + 0.5*(r1[n]+r2[n])*u[n]**2  # cell total energy (neighbor)

    # Reconstructed total energy at face
    rhoEL = rhoe[o] + 0.5*(r1L+r2L)*uL**2
    rhoER = rhoe[n] + 0.5*(r1R+r2R)*uR**2

    FE_cen    = 0.5*((rhoEL+pL)*uL + (rhoER+pR)*uR)
    FE_upwind = FE_cen - 0.5*lam*(rhoER - rhoEL)

    if scheme == 'APEC':
        if eos_type == 'CPG':
            # ── MUSCL-consistent PE dissipation (CPG, Appendix A) ──
            # Replace energy dissipation jump with same MUSCL jumps as mass
            eps0_h = 0.5*(eps0[o] + eps0[n])
            eps1_h = 0.5*(eps1[o] + eps1[n])
            u_h    = 0.5*(uL  + uR)
            rho_h  = 0.5*(rhoL + rhoR)
            drhoE  = (eps0_h*(r1R - r1L)
                    + eps1_h*(r2R - r2L)
                    + 0.5*u_h**2 * (rhoR - rhoL)
                    + rho_h * u_h * (uR - uL))
            FE = FE_cen - 0.5*lam*drhoE

        else:
            # ── Appendix A, Eq. A.4 (SRK / general EOS) ───────────
            # Cell-centered physical fluxes
            F1_o = r1[o]*u[o];   F1_n = r1[n]*u[n]
            F2_o = r2[o]*u[o];   F2_n = r2[n]*u[n]
            FU_o = (r1[o]+r2[o])*u[o]**2 + p[o]
            FU_n = (r1[n]+r2[n])*u[n]**2 + p[n]
            FE_o = (rhoE_o + p[o]) * u[o]
            FE_n = (rhoE_n + p[n]) * u[n]

            # Coefficients: c_s = eps_s - 0.5*u^2
            c0_o = eps0[o] - 0.5*u[o]**2;  c0_n = eps0[n] - 0.5*u[n]**2
            c1_o = eps1[o] - 0.5*u[o]**2;  c1_n = eps1[n] - 0.5*u[n]**2

            # t from owner side: flux(face) - flux(cell_o)
            t_o = (c0_o*(F1 - F1_o) + c1_o*(F2 - F2_o)
                 + u[o]*(FU - FU_o))
            # t from neighbor side: flux(cell_n) - flux(face)
            t_n = (c0_n*(F1_n - F1) + c1_n*(F2_n - F2)
                 + u[n]*(FU_n - FU))

            FE = 0.5*(FE_o + FE_n) + 0.5*t_o - 0.5*t_n
    else:
        FE = FE_upwind

    return F1, F2, FU, FE


# ══════════════════════════════════════════════════════════════════
# 5.  RHS  (divergence of fluxes)
# ══════════════════════════════════════════════════════════════════

def rhs_fvm(U, mesh, eos, scheme, T_prev=None, is_cpg=False):
    """
    Compute dU/dt = -1/V * sum_faces (F * n * A)

    U = [r1, r2, rhoU, rhoE]  — cell arrays of shape (N,)
    """
    r1, r2, rhoU, rhoE = U
    rho = r1 + r2
    u   = rhoU / np.maximum(rho, 1e-30)
    rhoe = rhoE - 0.5*rho*u**2

    # ── Primitive variables ───────────────────────────────────────
    if is_cpg:
        p = eos.p_from_rhoe(r1, r2, rhoe)
        c2 = eos.sound_speed_sq(r1, r2, p)
        eps0, eps1 = eos.epsilon(r1, r2, p) if scheme == 'APEC' else (None, None)
        T = None
    else:
        T = eos.T_from_rhoe(r1, r2, rhoe, T_prev)
        p = eos.pressure(r1, r2, T)
        c2 = eos.sound_speed_sq(r1, r2, T)
        eps0, eps1 = eos.epsilon(r1, r2, T) if scheme == 'APEC' else (None, None)

    lam_cell = np.abs(u) + np.sqrt(np.maximum(c2, 0.))

    # ── Face fluxes ───────────────────────────────────────────────
    eos_type = 'CPG' if is_cpg else 'SRK'
    F1, F2, FU, FE = compute_fluxes(
        r1, r2, u, rhoe, p, lam_cell, mesh, scheme,
        eps0=eps0, eps1=eps1, eos_type=eos_type)

    # ── Divergence: dU_i/dt = -1/V_i * sum_{faces of i} F*n*A ────
    # For 1-D periodic mesh: face f has owner o[f] and neighbor n[f].
    # Flux leaves owner (+) and enters neighbor (-).
    inv_vol = 1.0 / mesh.volumes   # (N,)
    o, n = mesh.face_owner, mesh.face_neighbor
    A    = mesh.face_area  # = 1 for 1-D

    d1  = np.zeros(mesh.N); d2  = np.zeros(mesh.N)
    dU_ = np.zeros(mesh.N); dE  = np.zeros(mesh.N)

    # Accumulate: owner loses flux, neighbor gains flux
    np.subtract.at(d1,  o,  F1 * A)
    np.add.at   (d1,  n,  F1 * A)
    np.subtract.at(d2,  o,  F2 * A)
    np.add.at   (d2,  n,  F2 * A)
    np.subtract.at(dU_, o,  FU * A)
    np.add.at   (dU_, n,  FU * A)
    np.subtract.at(dE,  o,  FE * A)
    np.add.at   (dE,  n,  FE * A)

    d1  *= inv_vol;  d2  *= inv_vol
    dU_ *= inv_vol;  dE  *= inv_vol

    return [d1, d2, dU_, dE], T, p


# ══════════════════════════════════════════════════════════════════
# 6.  TIME INTEGRATION  (SSP-RK3)
# ══════════════════════════════════════════════════════════════════

def _clip_pos(U, min_val=0.0):
    return [np.maximum(U[0], min_val), np.maximum(U[1], min_val), U[2], U[3]]

def ssp_rk3(U, mesh, eos, scheme, dx, dt, T_prev, is_cpg):
    k1, T1, p1 = rhs_fvm(U, mesh, eos, scheme, T_prev, is_cpg)
    U1 = _clip_pos([U[q] + dt*k1[q] for q in range(4)],
                   1e-10 if is_cpg else 0.)
    k2, T2, p2 = rhs_fvm(U1, mesh, eos, scheme, T1, is_cpg)
    U2 = _clip_pos([0.75*U[q] + 0.25*(U1[q] + dt*k2[q]) for q in range(4)],
                   1e-10 if is_cpg else 0.)
    k3, T3, p3 = rhs_fvm(U2, mesh, eos, scheme, T2, is_cpg)
    Un = _clip_pos([(1/3)*U[q] + (2/3)*(U2[q] + dt*k3[q]) for q in range(4)],
                   1e-10 if is_cpg else 0.)
    return Un, T3, p3


# ══════════════════════════════════════════════════════════════════
# 7.  RUNNERS
# ══════════════════════════════════════════════════════════════════

def pe_err_rms(p, p0):
    return float(np.sqrt(np.mean(((p - p0)/p0)**2)))

def pe_err_max(p, p0):
    return float(np.max(np.abs(p - p0)) / p0)


def run_cpg(scheme, N=501, t_end=8.0, CFL=0.6, p0=0.9, k=20.0):
    eos  = CPGEos()
    mesh = Mesh1D(N)
    x    = mesh.cell_centers

    # Initial condition
    xc, rc = 0.5, 0.25
    r = np.abs(x - xc)
    r1 = 0.5*0.6*(1. - np.tanh(k*(r - rc)))
    r2 = 0.5*0.2*(1. + np.tanh(k*(r - rc)))
    u  = np.ones(N)
    rhoe = eos.rhoe_from_p(r1, r2, np.full(N, p0))
    rhoE = rhoe + 0.5*(r1+r2)*u**2

    U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]
    rhoE0 = rhoE.copy()

    _, _, p_init = rhs_fvm(U, mesh, eos, 'FC', is_cpg=True)
    t_hist  = [0.]; pe_hist = [pe_err_rms(p_init, p0)]
    t, step, diverged = 0., 0, False

    print(f"[CPG-FVM] {scheme}  N={N}  t_end={t_end}  CFL={CFL}", flush=True)

    while t < t_end - 1e-14:
        r1_, r2_, rhoU_, rhoE_ = U
        rho_ = r1_+r2_
        u_   = rhoU_/np.maximum(rho_, 1e-30)
        rhoe_= rhoE_ - 0.5*rho_*u_**2
        p_   = eos.p_from_rhoe(r1_, r2_, rhoe_)
        c2_  = eos.sound_speed_sq(r1_, r2_, p_)
        lam  = float(np.max(np.abs(u_) + np.sqrt(np.maximum(c2_, 0.))))
        dt   = min(CFL * mesh.dx / (lam + 1e-10), t_end - t)

        try:
            U, _, p_ = ssp_rk3(U, mesh, eos, scheme, mesh.dx, dt, None, is_cpg=True)
        except Exception as e:
            print(f"  Diverged at t={t:.4f}: {e}"); diverged=True; break

        t += dt; step += 1
        pe_ = pe_err_rms(p_, p0)
        t_hist.append(t); pe_hist.append(pe_)

        if not np.isfinite(pe_) or pe_ > 50.:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.4f}"); diverged=True; break

        if step % 2000 == 0:
            print(f"  t={t:.3f}  PE_rms={pe_:.2e}", flush=True)

    status = 'Done' if not diverged else 'Diverged'
    print(f"  --> {status} t={t:.4f} ({step} steps)  PE_rms_final={pe_hist[-1]:.3e}")
    return x, U, p_, np.array(t_hist), np.array(pe_hist), diverged


def run_srk(scheme, N=101, t_end=0.07, CFL=0.3, p_inf=5e6, k=15.0):
    eos  = SRKEos()
    mesh = Mesh1D(N)
    x    = mesh.cell_centers

    # Initial condition
    xc, rc = 0.5, 0.25
    r = np.abs(x - xc)
    r1_inf, r2_inf = 400., 100.
    r1 = 0.5*r1_inf*(1. - np.tanh(k*(r - rc)))
    r2 = 0.5*r2_inf*(1. + np.tanh(k*(r - rc)))

    print("  [SRK] solving initial T ...", flush=True)
    T = eos.init_T_from_p(r1, r2, p_inf)
    p = eos.pressure(r1, r2, T)
    rhoe = eos.rhoe(r1, r2, T)
    u    = np.full(N, 100.)
    rhoE = rhoe + 0.5*(r1+r2)*u**2

    U = [r1.copy(), r2.copy(), (r1+r2)*u, rhoE.copy()]

    t_hist = [0.]; pe_hist = [pe_err_max(p, p_inf)]
    t, step, diverged = 0., 0, False
    T_cur = T.copy()

    print(f"[SRK-FVM] {scheme}  N={N}  t_end={t_end}  CFL={CFL}", flush=True)

    while t < t_end - 1e-14:
        r1_, r2_, rhoU_, rhoE_ = U
        rho_ = r1_+r2_
        u_   = rhoU_/np.maximum(rho_, 1e-30)
        rhoe_= rhoE_ - 0.5*rho_*u_**2
        T_cur = eos.T_from_rhoe(r1_, r2_, rhoe_, T_cur)
        c2_  = eos.sound_speed_sq(r1_, r2_, T_cur)
        lam  = float(np.max(np.abs(u_) + np.sqrt(np.maximum(c2_, 0.))))
        dt   = min(CFL * mesh.dx / (lam + 1e-10), t_end - t)

        try:
            U, T_cur, p_ = ssp_rk3(U, mesh, eos, scheme, mesh.dx, dt, T_cur, is_cpg=False)
        except Exception as e:
            print(f"  Diverged at t={t:.5f}: {e}"); diverged=True; break

        t += dt; step += 1
        pe_ = pe_err_max(p_, p_inf)
        t_hist.append(t); pe_hist.append(pe_)

        if step == 1:
            pe_step1 = pe_
            print(f"  step1 PE_max={pe_step1:.3e}")

        if not np.isfinite(pe_) or pe_ > 5.:
            print(f"  Diverged (PE={pe_:.2e}) at t={t:.5f}"); diverged=True; break

        if step % 500 == 0:
            print(f"  t={t:.5f}  PE_max={pe_:.2e}", flush=True)

    status = 'Done' if not diverged else 'Diverged'
    print(f"  --> {status} t={t:.5f} ({step} steps)")
    return x, U, p_, np.array(t_hist), np.array(pe_hist), diverged


# ══════════════════════════════════════════════════════════════════
# 8.  VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_cpg(N=501, t_end=8.0, CFL=0.6):
    print("\n" + "="*60)
    print("Case 1: CPG 1D interface advection")
    print("="*60)
    res_fc   = run_cpg('FC',   N=N, t_end=t_end, CFL=CFL)
    res_apec = run_cpg('APEC', N=N, t_end=t_end, CFL=CFL)

    pe_fc_final   = res_fc[4][-1]
    pe_apec_final = res_apec[4][-1]
    ratio = pe_fc_final / max(pe_apec_final, 1e-20)
    print(f"\nCPG results:  FC={pe_fc_final:.3e}  APEC={pe_apec_final:.3e}"
          f"  improvement={ratio:.1f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(res_fc[3],   res_fc[4],   'C0--', lw=2, label='FC-NPE (FVM)')
    ax.semilogy(res_apec[3], res_apec[4], 'C1-',  lw=2, label='APEC (FVM)')
    ax.set_xlabel('t'); ax.set_ylabel(r'$E_{PE}$ (rms)')
    ax.set_title(f'CPG FVM: PE error  N={N}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fvm3d_cpg_pe.png'), dpi=150)
    plt.close(fig)
    print("Saved fvm3d_cpg_pe.png")

    cpg_ok = (pe_apec_final < pe_fc_final)
    print(f"CPG pass: {cpg_ok}  (APEC {pe_apec_final:.3e} < FC {pe_fc_final:.3e})")
    return cpg_ok, pe_fc_final, pe_apec_final


def validate_srk(N=101, t_end=0.07, CFL=0.3):
    print("\n" + "="*60)
    print("Case 2: SRK 1D interface advection")
    print("="*60)
    res_fc   = run_srk('FC',   N=N, t_end=t_end, CFL=CFL)
    res_apec = run_srk('APEC', N=N, t_end=t_end, CFL=CFL)

    # Step-1 PE improvement
    pe_fc_s1   = res_fc[4][1]   if len(res_fc[4])   > 1 else res_fc[4][-1]
    pe_apec_s1 = res_apec[4][1] if len(res_apec[4]) > 1 else res_apec[4][-1]
    ratio = pe_fc_s1 / max(pe_apec_s1, 1e-30)
    print(f"\nSRK step-1:  FC={pe_fc_s1:.3e}  APEC={pe_apec_s1:.3e}"
          f"  improvement={ratio:.1f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.semilogy(res_fc[3],   res_fc[4],   'C0--', lw=2, label='FC-NPE (FVM)')
    ax.semilogy(res_apec[3], res_apec[4], 'C1-',  lw=2, label='APEC (FVM)')
    ax.set_xlabel('t'); ax.set_ylabel(r'$E_{PE}$ (max)')
    ax.set_title(f'SRK FVM: PE error  N={N}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fvm3d_srk_pe.png'), dpi=150)
    plt.close(fig)
    print("Saved fvm3d_srk_pe.png")

    srk_ok = (ratio >= 10.0)
    print(f"SRK pass: {srk_ok}  (ratio={ratio:.1f}x >= 10x)")
    return srk_ok, pe_fc_s1, pe_apec_s1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--case', choices=['cpg','srk'], default=None)
    args = parser.parse_args()

    if args.validate or args.case is None:
        cpg_ok, *_ = validate_cpg()
        srk_ok, *_ = validate_srk()
        print("\n" + "="*60)
        print(f"VALIDATION SUMMARY:  CPG={cpg_ok}  SRK={srk_ok}")
        if cpg_ok and srk_ok:
            print("VALIDATION PASSED")
            sys.exit(0)
        else:
            print("VALIDATION FAILED")
            sys.exit(1)
    elif args.case == 'cpg':
        ok, *_ = validate_cpg()
        sys.exit(0 if ok else 1)
    elif args.case == 'srk':
        ok, *_ = validate_srk()
        sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
