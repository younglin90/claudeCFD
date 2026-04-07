"""
kinetic_real.py
===============
Kinetic Lambda-Difference Scheme with SRK Real-Gas EOS and APEC Correction.

Extends lambda_diff_nd.py with:
  - SRK2Species EOS for CH4/N2 two-species supercritical flows
  - APEC PE-consistent energy correction for the kinetic scheme
  - MUSCL-LLF (FC-NPE) comparison scheme
  - 2D test cases: Kelvin-Helmholtz instability, 2D Riemann problem
  - SRK PE comparison: Kinetic / Kinetic+APEC / MUSCL-LLF

State vector convention (variable axis first, spatial axes last):
  U[v, i0, ..., i_{ndim-1}]
  U[s]        for s=0..Ns-1 : partial density rho_s
  U[Ns+d]     for d=0..ndim-1: momentum along axis d (rho*u_d)
  U[Ns+ndim]  : total energy rho*E

1D: (Ns+2, Nx)
2D: (Ns+3, Ny, Nx)   (axis 0 = y, axis 1 = x)

Usage:
  python solver/kinetic_real.py --test eoc
  python solver/kinetic_real.py --test sod
  python solver/kinetic_real.py --test kh
  python solver/kinetic_real.py --test riemann2d
  python solver/kinetic_real.py --test srk_apec
  python solver/kinetic_real.py --test compare
  python solver/kinetic_real.py --test all
"""

import argparse
import sys
import os
import abc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Section 0: Constants
# ─────────────────────────────────────────────
EPS0       = 1e-10
B_COMPRESS = 4
CFL_DEFAULT = 0.8


# ═══════════════════════════════════════════════════════════════════════
# Section 1: SRK EOS helper functions  (CH4/N2)
# ═══════════════════════════════════════════════════════════════════════

Ru  = 8.314
M   = np.array([16.043e-3, 28.014e-3])   # kg/mol
Tc  = np.array([190.56,    126.19   ])    # K
pc  = np.array([4.599e6,   3.396e6  ])    # Pa
om  = np.array([0.0115,    0.0372   ])    # acentric factor

a_sp = 0.42748 * Ru**2 * Tc**2 / pc
b_sp = 0.08664 * Ru   * Tc    / pc
fom  = 0.480 + 1.574*om - 0.176*om**2
Cv0  = Ru / M * np.array([3.4, 2.5])


def _mix(r1, r2, T):
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    T  = np.asarray(T,  dtype=float)
    rho   = r1 + r2
    C0    = r1 / M[0]
    C1    = r2 / M[1]
    Ctot  = np.maximum(C0 + C1, 1e-40)
    X0    = C0 / Ctot
    X1    = C1 / Ctot
    Mbar  = rho / Ctot

    sq0   = np.sqrt(np.maximum(T / Tc[0], 1e-10))
    sq1   = np.sqrt(np.maximum(T / Tc[1], 1e-10))
    al0   = (1.0 + fom[0]*(1.0 - sq0))**2
    al1   = (1.0 + fom[1]*(1.0 - sq1))**2
    da0   = -fom[0]*(1.0 + fom[0]*(1.0 - sq0)) / (sq0*Tc[0] + 1e-300)
    da1   = -fom[1]*(1.0 + fom[1]*(1.0 - sq1)) / (sq1*Tc[1] + 1e-300)

    a01   = np.sqrt(a_sp[0]*a_sp[1])
    al01  = np.sqrt(np.maximum(al0*al1, 1e-300))
    dal01 = a01*(al1*da0 + al0*da1) / (2.0*al01)

    aA    = X0*X0*a_sp[0]*al0 + X1*X1*a_sp[1]*al1 + 2.0*X0*X1*a01*al01
    daA   = X0*X0*a_sp[0]*da0 + X1*X1*a_sp[1]*da1 + 2.0*X0*X1*dal01

    b     = X0*b_sp[0] + X1*b_sp[1]
    v     = Mbar / np.maximum(rho, 1e-30)
    return rho, Mbar, X0, X1, C0, C1, Ctot, b, v, aA, daA


def srk_p(r1, r2, T):
    _, _, _, _, _, _, _, b, v, aA, _ = _mix(r1, r2, T)
    vb   = np.maximum(v - b, 1e-20)
    vvb  = np.maximum(v*(v + b), 1e-60)
    return Ru*T/vb - aA/vvb


def srk_rhoe(r1, r2, T):
    _, _, _, _, _, _, Ctot, b, v, aA, daA = _mix(r1, r2, T)
    rhoe0 = r1*Cv0[0]*T + r2*Cv0[1]*T
    arg   = np.where(b > 1e-40, 1.0 + b/np.maximum(v, 1e-30), 1.0)
    dep   = np.where(b > 1e-40,
                     Ctot*(T*daA - aA)/b * np.log(np.maximum(arg, 1e-30)),
                     0.0)
    return rhoe0 + dep


def T_from_rhoe(r1, r2, rhoe_target, T_in=None):
    r1          = np.asarray(r1,          dtype=float)
    r2          = np.asarray(r2,          dtype=float)
    rhoe_target = np.asarray(rhoe_target, dtype=float)
    T = (np.asarray(T_in, dtype=float).copy()
         if T_in is not None else np.full_like(r1, 200.0))
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
        if np.max(np.abs(dT)) < 1e-3:
            break
    return T


def _dpdT(r1, r2, T, h=1.0):
    return (srk_p(r1, r2, T+h) - srk_p(r1, r2, T-h)) / (2.0*h)


def _dpdr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s == 0 else r2)*f, dmin)
    if s == 0:
        return (srk_p(r1+dr, r2, T) - srk_p(r1-dr, r2, T)) / (2.0*dr)
    else:
        return (srk_p(r1, r2+dr, T) - srk_p(r1, r2-dr, T)) / (2.0*dr)


def _drhoedr(r1, r2, T, s, f=5e-4, dmin=0.05):
    dr = np.maximum(np.abs(r1 if s == 0 else r2)*f, dmin)
    if s == 0:
        return (srk_rhoe(r1+dr, r2, T) - srk_rhoe(r1-dr, r2, T)) / (2.0*dr)
    else:
        return (srk_rhoe(r1, r2+dr, T) - srk_rhoe(r1, r2-dr, T)) / (2.0*dr)


def srk_Cv(r1, r2, T, h=1.0):
    return ((srk_rhoe(r1, r2, T+h) - srk_rhoe(r1, r2, T-h)) / (2.0*h)) / \
           np.maximum(r1 + r2, 1e-30)


def srk_c2(r1, r2, T):
    rho   = r1 + r2
    Y0    = r1 / np.maximum(rho, 1e-30)
    dpdr_T = Y0*_dpdr(r1, r2, T, 0) + (1.0 - Y0)*_dpdr(r1, r2, T, 1)
    dpT    = _dpdT(r1, r2, T)
    Cv     = srk_Cv(r1, r2, T)
    return np.maximum(dpdr_T + T*dpT**2 / (rho**2*Cv + 1e-30), 100.0)


def epsilon_v(r1, r2, T, s):
    rho  = r1 + r2
    Cv   = srk_Cv(r1, r2, T)
    dpT  = _dpdT(r1, r2, T)
    dprs = _dpdr(r1, r2, T, s)
    drhoe= _drhoedr(r1, r2, T, s)
    return -(rho * Cv / (dpT + 1e-10)) * dprs + drhoe


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Abstract EOS base + concrete EOS classes
# ═══════════════════════════════════════════════════════════════════════

class EOSBase(abc.ABC):
    """Abstract EOS interface."""

    @abc.abstractmethod
    def primitives(self, U, axis, ndim):
        """Returns rho, u_n, p, gam, a."""

    @abc.abstractmethod
    def flux(self, U, axis, ndim):
        """Physical flux along sweep axis."""

    @abc.abstractmethod
    def positivity_alpha(self, u_n, a, gam):
        """Returns alpha_R, alpha_L for positivity bounds."""


class IdealGasMixture(EOSBase):
    """N-species calorically perfect gas mixture."""

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas, dtype=float)
        self.Ns = len(self.gammas)
        self.cv = 1.0 / (self.gammas - 1.0)
        self.cp = self.gammas * self.cv

    def _mixture_thermo(self, U, ndim):
        Ns    = self.Ns
        rho_s = U[:Ns]
        rho   = np.sum(rho_s, axis=0)
        rho   = np.maximum(rho, EPS0)
        W_s   = rho_s / rho[np.newaxis]
        cv_mix = np.einsum('s,s...->...', self.cv, W_s)
        cp_mix = np.einsum('s,s...->...', self.cp, W_s)
        gam    = cp_mix / np.maximum(cv_mix, EPS0)
        mom    = U[Ns:Ns+ndim]
        u_list = [mom[d] / rho for d in range(ndim)]
        rhoE   = U[Ns+ndim]
        ke     = 0.5 * rho * sum(u**2 for u in u_list)
        p      = (gam - 1.0) * (rhoE - ke)
        p      = np.maximum(p, EPS0)
        a      = np.sqrt(np.maximum(gam * p / rho, EPS0))
        return rho, u_list, p, gam, a

    def primitives(self, U, axis, ndim):
        rho, u_list, p, gam, a = self._mixture_thermo(U, ndim)
        return rho, u_list[axis], p, gam, a

    def flux(self, U, axis, ndim):
        Ns   = self.Ns
        rho, u_list, p, gam, a = self._mixture_thermo(U, ndim)
        u_n  = u_list[axis]
        F    = np.empty_like(U)
        for s in range(Ns):
            F[s] = U[s] * u_n
        for d in range(ndim):
            F[Ns+d] = U[Ns+d] * u_n
            if d == axis:
                F[Ns+d] += p
        F[Ns+ndim] = (U[Ns+ndim] + p) * u_n
        return F

    def positivity_alpha(self, u_n, a, gam):
        if gam is None:
            # fallback for real gas
            alpha_L = -u_n + a
            alpha_R =  u_n + a
            return alpha_R, alpha_L
        coef    = np.sqrt(np.maximum((gam - 1.0) / (2.0 * gam), 0.0))
        alpha_L = -u_n + coef * a
        alpha_R =  u_n + coef * a
        return alpha_R, alpha_L


class SRK2Species(EOSBase):
    """
    CH4/N2 two-species SRK EOS.

    State vector: U = [rho1, rho2, rho*u0, ..., rho*u_{ndim-1}, rhoE]
    Ns = 2.
    """
    Ns = 2

    def __init__(self):
        self._T_cache = None   # warm-start Newton

    def set_T_cache(self, T):
        self._T_cache = T.copy() if T is not None else None

    def _get_T(self, r1, r2, rhoe, update_cache=False):
        """Solve T from rhoe, using T_cache only when shape matches."""
        T_in = (self._T_cache
                if (self._T_cache is not None
                    and self._T_cache.shape == r1.shape)
                else None)
        T = T_from_rhoe(r1, r2, rhoe, T_in=T_in)
        if update_cache and T_in is not None:
            self._T_cache = T.copy()
        return T

    def _extract_prim(self, U, axis, ndim):
        r1   = U[0]
        r2   = U[1]
        rho  = np.maximum(r1 + r2, EPS0)
        mom  = U[2:2+ndim]
        u_list = [mom[d] / rho for d in range(ndim)]
        rhoE = U[2+ndim]
        ke   = 0.5 * rho * sum(uu**2 for uu in u_list)
        rhoe = rhoE - ke
        T    = self._get_T(r1, r2, rhoe)
        p    = srk_p(r1, r2, T)
        c2   = srk_c2(r1, r2, T)
        a    = np.sqrt(np.maximum(c2, EPS0))
        return rho, u_list, p, T, a

    def primitives(self, U, axis, ndim):
        rho, u_list, p, T, a = self._extract_prim(U, axis, ndim)
        return rho, u_list[axis], p, None, a   # gam=None for SRK

    def flux(self, U, axis, ndim):
        r1   = U[0]
        r2   = U[1]
        rho  = np.maximum(r1 + r2, EPS0)
        mom  = U[2:2+ndim]
        u_list = [mom[d] / rho for d in range(ndim)]
        rhoE = U[2+ndim]
        ke   = 0.5 * rho * sum(uu**2 for uu in u_list)
        rhoe = rhoE - ke
        T    = self._get_T(r1, r2, rhoe)
        p    = srk_p(r1, r2, T)
        u_n  = u_list[axis]

        F = np.empty_like(U)
        # Species fluxes
        F[0] = r1 * u_n
        F[1] = r2 * u_n
        # Momentum fluxes
        for d in range(ndim):
            F[2+d] = U[2+d] * u_n
            if d == axis:
                F[2+d] += p
        # Energy flux
        F[2+ndim] = (rhoE + p) * u_n
        return F

    def positivity_alpha(self, u_n, a, gam):
        # Conservative bound: |u_n| + a (gam unused for SRK)
        alpha_R =  u_n + a
        alpha_L = -u_n + a
        return alpha_R, alpha_L

    def get_epsilon(self, U, ndim):
        """
        Returns (eps0, eps1) epsilon_v values for APEC correction.
        U : (Nvars, *spatial)
        """
        r1   = U[0]
        r2   = U[1]
        rho  = np.maximum(r1 + r2, EPS0)
        mom  = U[2:2+ndim]
        u_list = [mom[d] / rho for d in range(ndim)]
        rhoE = U[2+ndim]
        ke   = 0.5 * rho * sum(uu**2 for uu in u_list)
        rhoe = rhoE - ke
        T    = self._get_T(r1, r2, rhoe)
        eps0 = epsilon_v(r1, r2, T, 0)
        eps1 = epsilon_v(r1, r2, T, 1)
        return eps0, eps1


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Kinetic scheme core
# ═══════════════════════════════════════════════════════════════════════

def abs_sign(x):
    """abs_sign(x) = 1 if |x|>EPS0, else 0."""
    return np.where(np.abs(x) > EPS0, 1.0, 0.0)


def minmod(x, y):
    """Minmod limiter."""
    return np.where(x * y > 0,
                    np.where(np.abs(x) < np.abs(y), x, y),
                    0.0)


def compute_lambda(U_mov, F_mov, axis, ndim, eos):
    """
    Compute lambda at every interface along the last axis (sweep direction).
    U_mov, F_mov : (Nvars, *others, N)
    Returns lam  : (*others, N-1)
    """
    Ns   = eos.Ns
    rho  = np.sum(U_mov[:Ns], axis=0)
    rhou = U_mov[Ns+axis]
    rhoE = U_mov[Ns+ndim]

    F_rho  = np.sum(F_mov[:Ns], axis=0)
    F_rhou = F_mov[Ns+axis]
    F_rhoE = F_mov[Ns+ndim]

    d_rho  = rho[..., 1:] - rho[..., :-1]
    d_rhou = rhou[..., 1:] - rhou[..., :-1]
    d_rhoE = rhoE[..., 1:] - rhoE[..., :-1]

    dF_rho  = F_rho[..., 1:]  - F_rho[..., :-1]
    dF_rhou = F_rhou[..., 1:] - F_rhou[..., :-1]
    dF_rhoE = F_rhoE[..., 1:] - F_rhoE[..., :-1]

    lam_rh = np.minimum.reduce([
        np.abs(dF_rho)  / (np.abs(d_rho)  + EPS0),
        np.abs(dF_rhou) / (np.abs(d_rhou) + EPS0),
        np.abs(dF_rhoE) / (np.abs(d_rhoE) + EPS0),
    ])

    rho_p, u_n, p_p, gam_p, a_p = eos.primitives(U_mov, axis, ndim)
    alpha_R, alpha_L = eos.positivity_alpha(u_n, a_p, gam_p)

    alpha_L_left  = alpha_L[..., :-1]
    alpha_R_right = alpha_R[..., 1:]

    lam_base = np.maximum.reduce([lam_rh, alpha_L_left, alpha_R_right])

    rho_L   = rho_p[..., :-1]
    rho_R   = rho_p[..., 1:]
    p_L     = p_p[..., :-1]
    p_R     = p_p[..., 1:]
    u_L     = u_n[..., :-1]
    u_R     = u_n[..., 1:]

    rho_avg = 0.5 * (rho_L + rho_R)
    p_avg   = 0.5 * (p_L   + p_R)
    rel_rho = np.abs(rho_R - rho_L) / (rho_avg + EPS0)
    rel_p   = np.abs(p_R   - p_L)   / (p_avg   + EPS0)

    is_contact  = (rel_rho > 0.1) & (rel_p < 0.1)
    lam_contact = abs_sign(u_L + u_R) * lam_base

    lam = np.where(is_contact, lam_contact, lam_base)
    return np.maximum(lam, 0.0)


def _iflux_1st(U_mov, F_mov, lam):
    """First-order kinetic interface flux."""
    avg_F = 0.5 * (F_mov[..., :-1] + F_mov[..., 1:])
    dU    = U_mov[..., 1:] - U_mov[..., :-1]
    return avg_F - 0.5 * lam[np.newaxis] * dU


def _iflux_ho(U_mov, F_mov, lam, b, bc):
    """Higher-order kinetic interface flux (Chakravarthy-Osher)."""
    N  = U_mov.shape[-1]
    M  = N - 1

    dU = U_mov[..., 1:] - U_mov[..., :-1]
    dF = F_mov[..., 1:] - F_mov[..., :-1]

    lam_br = lam[np.newaxis]
    dGp = 0.5 * dF + 0.5 * lam_br * dU
    dGm = 0.5 * dF - 0.5 * lam_br * dU

    Gf1 = _iflux_1st(U_mov, F_mov, lam)
    Gf  = Gf1.copy()

    i = slice(1, M-1)

    if b == 0:
        Gf[..., i] += (
              (1.0/6.0) * dGp[..., 0:M-2]
            - (1.0/6.0) * dGm[..., 2:M  ]
            + (1.0/3.0) * dGp[..., i    ]
            - (1.0/3.0) * dGm[..., i    ]
        )
        if bc == "periodic":
            corr = (
                  (1.0/6.0) * dGp[..., M-2]
                - (1.0/6.0) * dGm[..., 1  ]
                + (1.0/3.0) * dGp[..., 0  ]
                - (1.0/3.0) * dGm[..., 0  ]
            )
            Gf[...,  0] += corr
            Gf[..., -1] += corr
    else:
        Gf[..., i] += (
              (1.0/6.0) * minmod(b * dGp[..., i    ], dGp[..., 0:M-2])
            - (1.0/6.0) * minmod(b * dGm[..., i    ], dGm[..., 2:M  ])
            + (1.0/3.0) * minmod(b * dGp[..., 0:M-2], dGp[..., i    ])
            - (1.0/3.0) * minmod(b * dGm[..., 2:M  ], dGm[..., i    ])
        )
        if bc == "periodic":
            corr = (
                  (1.0/6.0) * minmod(b * dGp[..., 0  ], dGp[..., M-2])
                - (1.0/6.0) * minmod(b * dGm[..., 0  ], dGm[..., 1  ])
                + (1.0/3.0) * minmod(b * dGp[..., M-2], dGp[..., 0  ])
                - (1.0/3.0) * minmod(b * dGm[..., 1  ], dGm[..., 0  ])
            )
            Gf[...,  0] += corr
            Gf[..., -1] += corr

    return Gf


def interface_flux(U_mov, F_mov, lam, order, bc):
    """Dispatch to 1st or higher-order kinetic interface flux."""
    if order == 1:
        return _iflux_1st(U_mov, F_mov, lam)
    elif order == 2:
        return _iflux_ho(U_mov, F_mov, lam, b=1, bc=bc)
    elif order == 3:
        return _iflux_ho(U_mov, F_mov, lam, b=B_COMPRESS, bc=bc)
    else:   # order == 4, unlimited
        return _iflux_ho(U_mov, F_mov, lam, b=0, bc=bc)


def _muscl_last(q):
    """
    MUSCL reconstruction with minmod limiter on last axis.

    Parameters
    ----------
    q : (*others, N+2)  — cell values including 1 ghost cell on each end

    Returns
    -------
    q_L : (*others, N+1)  left-biased state at each of N+1 interfaces
    q_R : (*others, N+1)  right-biased state at each of N+1 interfaces

    At interface i (0-based, between ext cell i and ext cell i+1):
      q_L[i] = q[i]   + 0.5 * minmod(q[i]-q[i-1], q[i+1]-q[i])
      q_R[i] = q[i+1] - 0.5 * minmod(q[i+1]-q[i], q[i+2]-q[i+1])
    Ghost cells (i=0 and i=N+1) use zero slope.
    """
    # Slopes at interior cells 1..N (N slopes)
    dL = q[..., 1:-1] - q[..., :-2]   # (*others, N)
    dR = q[..., 2:  ] - q[..., 1:-1]  # (*others, N)
    slope_inner = minmod(dL, dR)       # (*others, N)

    # Pad slopes with zeros for ghost cells to get (*others, N+2)
    slope_all = np.zeros_like(q)
    slope_all[..., 1:-1] = slope_inner

    # Left- and right-biased values at each cell
    q_plus  = q + 0.5 * slope_all   # right face of cell j
    q_minus = q - 0.5 * slope_all   # left  face of cell j

    # At interface i: left state = right face of cell i, right state = left face of cell i+1
    q_L = q_plus[..., :-1]    # (*others, N+1)
    q_R = q_minus[..., 1:]    # (*others, N+1)
    return q_L, q_R


def interface_flux_apec(U_mov, F_mov, lam, eps0_h, eps1_h, ndim, axis,
                        use_muscl=False):
    """
    APEC-corrected interface flux for SRK2Species.

    When use_muscl=False: same as _iflux_1st but energy component dissipation
    is replaced by APEC PE-consistent term using cell-center jumps.

    When use_muscl=True: MUSCL reconstruction is applied to ALL conserved
    variables (rho1, rho2, all momentum, energy).  The full kinetic LxF flux is
    rebuilt from the MUSCL-reconstructed states so that mass, momentum, and
    energy dissipation are mutually consistent.  Then the energy dissipation
    is further replaced by the APEC PE-consistent term using the same
    MUSCL-reconstructed jumps.  The kinetic lambda (lam) is kept as-is.

      drhoE_apec = eps0_h*(r1R-r1L) + eps1_h*(r2R-r2L)
                 + 0.5*u_h^2*(rhoR-rhoL) + rho_h*u_h*(uR-uL)

    Parameters
    ----------
    U_mov    : (Nvars, *others, N+2)   — includes 1 ghost cell on each end
    F_mov    : (Nvars, *others, N+2)   — cell-centre fluxes (used for avg_FE)
    lam      : (*others, N+1)          — kinetic lambda at N+1 interfaces
    eps0_h   : (*others, N+1)          — averaged epsilon for species 0
    eps1_h   : (*others, N+1)          — averaged epsilon for species 1
    ndim     : int
    axis     : int (sweep axis, 0-based spatial)
    use_muscl: bool
    """
    Ns = 2
    rhoE_idx = Ns + ndim

    if use_muscl:
        # MUSCL reconstruction of ALL conserved variables
        # U_mov : (Nvars, *others, N+2)  → L/R each (*others, N+1)
        Nvars = U_mov.shape[0]
        UL_list = []
        UR_list = []
        for v in range(Nvars):
            vL, vR = _muscl_last(U_mov[v])
            UL_list.append(vL)
            UR_list.append(vR)

        r1_L  = UL_list[0];      r1_R  = UR_list[0]
        r2_L  = UL_list[1];      r2_R  = UR_list[1]
        rho_L = r1_L + r2_L;     rho_R = r1_R + r2_R
        rhou_L = UL_list[Ns+axis]; rhou_R = UR_list[Ns+axis]
        u_L   = rhou_L / np.maximum(rho_L, EPS0)
        u_R   = rhou_R / np.maximum(rho_R, EPS0)
        rhoE_L = UL_list[rhoE_idx]; rhoE_R = UR_list[rhoE_idx]

        # Rebuild kinetic LxF flux from MUSCL-reconstructed states.
        # For species and momentum: use reconstructed conserved variables.
        # For energy centered flux: use reconstructed energy.
        # Dissipation: kinetic lambda * reconstructed jumps.
        Gf = np.empty((Nvars,) + lam.shape)
        for v in range(Nvars):
            avg_F_v = 0.5*(F_mov[v, ..., :-1] + F_mov[v, ..., 1:])
            dU_v    = UR_list[v] - UL_list[v]
            Gf[v]   = avg_F_v - 0.5 * lam * dU_v

        rho_h = 0.5*(rho_L + rho_R)
        u_h   = 0.5*(u_L   + u_R)

        # APEC energy dissipation using the same MUSCL-reconstructed jumps
        diss_E_apec = (eps0_h * (r1_R  - r1_L)
                     + eps1_h * (r2_R  - r2_L)
                     + 0.5 * u_h**2 * (rho_R  - rho_L)
                     + rho_h * u_h  * (u_R    - u_L))

        avg_FE = 0.5*(F_mov[rhoE_idx, ..., :-1] + F_mov[rhoE_idx, ..., 1:])
        Gf[rhoE_idx] = avg_FE - 0.5 * lam * diss_E_apec

    else:
        # Standard interface flux (cell-center)
        Gf = _iflux_1st(U_mov, F_mov, lam)   # (Nvars, *others, N-1)

        r1_L   = U_mov[0, ..., :-1]
        r1_R   = U_mov[0, ..., 1:]
        r2_L   = U_mov[1, ..., :-1]
        r2_R   = U_mov[1, ..., 1:]
        rho_L  = r1_L + r2_L
        rho_R  = r1_R + r2_R
        rhou_L = U_mov[Ns+axis, ..., :-1]
        rhou_R = U_mov[Ns+axis, ..., 1:]
        u_L    = rhou_L / np.maximum(rho_L, EPS0)
        u_R    = rhou_R / np.maximum(rho_R, EPS0)

        rho_h  = 0.5*(rho_L + rho_R)
        u_h    = 0.5*(u_L   + u_R)

        diss_E_apec = (eps0_h * (r1_R - r1_L)
                     + eps1_h * (r2_R - r2_L)
                     + 0.5 * u_h**2 * (rho_R - rho_L)
                     + rho_h * u_h  * (u_R - u_L))

        avg_FE = 0.5*(F_mov[rhoE_idx, ..., :-1] + F_mov[rhoE_idx, ..., 1:])
        Gf[rhoE_idx] = avg_FE - 0.5 * lam * diss_E_apec

    return Gf


def _apply_bc(U_ext, spatial_axis, bc):
    """Apply boundary conditions along spatial_axis."""
    arr_axis  = spatial_axis + 1
    n         = U_ext.shape[arr_axis]

    idx_first  = [slice(None)] * U_ext.ndim
    idx_last   = [slice(None)] * U_ext.ndim
    idx_second = [slice(None)] * U_ext.ndim
    idx_penult = [slice(None)] * U_ext.ndim

    idx_first[arr_axis]  = 0
    idx_last[arr_axis]   = -1
    idx_second[arr_axis] = 1
    idx_penult[arr_axis] = -2

    if bc == "periodic":
        U_ext[tuple(idx_first)]  = U_ext[tuple(idx_penult)]
        U_ext[tuple(idx_last)]   = U_ext[tuple(idx_second)]
    else:   # transmissive
        U_ext[tuple(idx_first)]  = U_ext[tuple(idx_second)]
        U_ext[tuple(idx_last)]   = U_ext[tuple(idx_penult)]


def residual_axis(U, axis, dx, eos, order, bc, ndim):
    """Spatial residual dF/dx_axis for all cells (standard kinetic)."""
    arr_ax    = axis + 1
    pad_width = [(0,0)] * U.ndim
    pad_width[arr_ax] = (1, 1)
    U_ext = np.pad(U, pad_width, mode='edge')
    _apply_bc(U_ext, axis, bc)

    U_mov = np.moveaxis(U_ext, arr_ax, -1)
    F_ext = eos.flux(U_ext, axis, ndim)
    F_mov = np.moveaxis(F_ext, arr_ax, -1)

    lam = compute_lambda(U_mov, F_mov, axis, ndim, eos)
    Gf  = interface_flux(U_mov, F_mov, lam, order, bc)
    R_mov = (Gf[..., 1:] - Gf[..., :-1]) / dx
    return np.moveaxis(R_mov, -1, arr_ax)


def residual_axis_apec(U, axis, dx, eos, bc, ndim):
    """
    Spatial residual with APEC PE-consistent energy correction.
    Order 1 only (APEC is applied at the first-order level).
    Requires SRK2Species eos.
    """
    arr_ax    = axis + 1
    pad_width = [(0,0)] * U.ndim
    pad_width[arr_ax] = (1, 1)
    U_ext = np.pad(U, pad_width, mode='edge')
    _apply_bc(U_ext, axis, bc)

    # Compute eps on interior cells
    U_int = _extract_interior(U_ext, arr_ax)   # (Nvars, *spatial)
    eps0_int, eps1_int = eos.get_epsilon(U_int, ndim)  # (*spatial)

    # Pad eps arrays with ghost cells (zero at boundaries; transmissive/periodic)
    eps0_ext = np.pad(eps0_int, [(1,1) if i == axis else (0,0)
                                 for i in range(ndim)], mode='edge')
    eps1_ext = np.pad(eps1_int, [(1,1) if i == axis else (0,0)
                                 for i in range(ndim)], mode='edge')

    # Move sweep axis to last
    U_mov   = np.moveaxis(U_ext,   arr_ax, -1)   # (Nvars, *others, N+2)
    eps0_mov = np.moveaxis(eps0_ext, axis,  -1)   # (*others, N+2)
    eps1_mov = np.moveaxis(eps1_ext, axis,  -1)   # (*others, N+2)

    F_ext = eos.flux(U_ext, axis, ndim)
    F_mov = np.moveaxis(F_ext, arr_ax, -1)

    lam = compute_lambda(U_mov, F_mov, axis, ndim, eos)   # (*others, N+1)

    # Average eps at interfaces
    eps0_h = 0.5*(eps0_mov[..., :-1] + eps0_mov[..., 1:])  # (*others, N+1)
    eps1_h = 0.5*(eps1_mov[..., :-1] + eps1_mov[..., 1:])  # (*others, N+1)

    Gf = interface_flux_apec(U_mov, F_mov, lam, eps0_h, eps1_h, ndim, axis,
                             use_muscl=True)
    R_mov = (Gf[..., 1:] - Gf[..., :-1]) / dx
    return np.moveaxis(R_mov, -1, arr_ax)


def compute_dt(U, eos, dx_list, order, sigma, ndim):
    """Global CFL time step."""
    dt_min = np.inf
    for d, dx in enumerate(dx_list):
        arr_ax    = d + 1
        pad_width = [(0,0)] * U.ndim
        pad_width[arr_ax] = (1, 1)
        U_ext = np.pad(U, pad_width, mode='edge')

        U_mov = np.moveaxis(U_ext, arr_ax, -1)
        F_ext = eos.flux(U_ext, d, ndim)
        F_mov = np.moveaxis(F_ext, arr_ax, -1)

        lam = compute_lambda(U_mov, F_mov, d, ndim, eos)

        lam_plus  = lam[..., 1:]
        lam_minus = lam[..., :-1]
        dt_p = np.min(2.0 * dx / (lam_plus + lam_minus + EPS0))

        U_int = _extract_interior(U_ext, arr_ax)
        rho_int, u_n_int, p_int, gam_int, a_int = eos.primitives(U_int, d, ndim)
        lam_max = np.max(np.maximum.reduce([
            np.abs(u_n_int - a_int),
            np.abs(u_n_int),
            np.abs(u_n_int + a_int),
        ]))
        dt_s = dx / (lam_max + EPS0)

        if order > 1:
            dt_p = dt_p / 2.0

        dt_min = min(dt_min, sigma * min(dt_p, dt_s))

    return dt_min


def _extract_interior(U_ext, arr_ax):
    """Strip ghost cells along arr_ax."""
    sl         = [slice(None)] * U_ext.ndim
    sl[arr_ax] = slice(1, -1)
    return U_ext[tuple(sl)]


def _ssprk3_axis(U, dx, dt, eos, order, bc, ndim, axis):
    """One SSPRK3 step along a single axis (standard kinetic)."""
    def R(V):
        return residual_axis(V, axis, dx, eos, order, bc, ndim)
    U0 = U
    U1 = U0 - dt * R(U0)
    U2 = 0.75*U0 + 0.25*U1 - 0.25*dt * R(U1)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*U2 - (2.0/3.0)*dt * R(U2)
    return U3


def _ssprk3_axis_apec(U, dx, dt, eos, bc, ndim, axis):
    """One SSPRK3 step along a single axis with APEC correction (order=1)."""
    def R(V):
        return residual_axis_apec(V, axis, dx, eos, bc, ndim)
    U0 = U
    U1 = U0 - dt * R(U0)
    U2 = 0.75*U0 + 0.25*U1 - 0.25*dt * R(U1)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*U2 - (2.0/3.0)*dt * R(U2)
    return U3


def step_ssprk3(U, dx_list, dt, eos, order, bc, ndim):
    """Full SSPRK3 + Strang splitting time step."""
    if ndim == 1:
        return _ssprk3_axis(U, dx_list[0], dt, eos, order, bc, ndim, axis=0)
    elif ndim == 2:
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        U = _ssprk3_axis(U, dx_list[1], dt,   eos, order, bc, ndim, axis=1)
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        return U
    elif ndim == 3:
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        U = _ssprk3_axis(U, dx_list[1], dt/2, eos, order, bc, ndim, axis=1)
        U = _ssprk3_axis(U, dx_list[2], dt,   eos, order, bc, ndim, axis=2)
        U = _ssprk3_axis(U, dx_list[1], dt/2, eos, order, bc, ndim, axis=1)
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        return U
    else:
        raise ValueError(f"ndim={ndim} not supported")


def step_ssprk3_apec(U, dx_list, dt, eos, bc, ndim):
    """Full SSPRK3 + Strang splitting with APEC (order=1)."""
    if ndim == 1:
        return _ssprk3_axis_apec(U, dx_list[0], dt, eos, bc, ndim, axis=0)
    elif ndim == 2:
        U = _ssprk3_axis_apec(U, dx_list[0], dt/2, eos, bc, ndim, axis=0)
        U = _ssprk3_axis_apec(U, dx_list[1], dt,   eos, bc, ndim, axis=1)
        U = _ssprk3_axis_apec(U, dx_list[0], dt/2, eos, bc, ndim, axis=0)
        return U
    else:
        raise ValueError(f"ndim={ndim} not supported for APEC (1D/2D only)")


def run_simulation(U0, dx_list, t_end, eos, order=1,
                   sigma=CFL_DEFAULT, bc="transmissive",
                   ndim=None, max_steps=2_000_000):
    """Standard kinetic simulation runner."""
    U    = U0.copy()
    if ndim is None:
        ndim = U.ndim - 1
    t = 0.0
    for _ in range(max_steps):
        if t >= t_end:
            break
        dt = compute_dt(U, eos, dx_list, order, sigma, ndim)
        dt = min(dt, t_end - t)
        if order == 1 and ndim == 1:
            R = residual_axis(U, 0, dx_list[0], eos, order, bc, ndim)
            U = U - dt * R
        else:
            U = step_ssprk3(U, dx_list, dt, eos, order, bc, ndim)
        t += dt
    return U, t


def run_simulation_apec(U0, dx_list, t_end, eos, sigma=CFL_DEFAULT,
                        bc="transmissive", ndim=None, max_steps=2_000_000,
                        track_pe=False, p_ref=5e6):
    """APEC kinetic simulation runner (order=1). Returns (U, t) or (U, t, t_hist, pe_hist)."""
    U    = U0.copy()
    if ndim is None:
        ndim = U.ndim - 1
    t    = 0.0
    t_hist  = [0.0]
    pe_hist = []
    order   = 1    # APEC is 1st order

    if track_pe:
        r1_0, r2_0 = U[0], U[1]
        p0 = eos._initial_p if hasattr(eos, '_initial_p') else None

    for _ in range(max_steps):
        if t >= t_end:
            break
        dt = compute_dt(U, eos, dx_list, order, sigma, ndim)
        dt = min(dt, t_end - t)

        if ndim == 1:
            U = _ssprk3_axis_apec(U, dx_list[0], dt, eos, bc, ndim, axis=0)
        else:
            U = step_ssprk3_apec(U, dx_list, dt, eos, bc, ndim)
        t += dt

        if track_pe:
            r1, r2 = U[0], U[1]
            rho    = np.maximum(r1 + r2, EPS0)
            rhoE   = U[2+ndim]
            if ndim == 1:
                u = U[2] / rho
            else:
                u = U[2] / rho
            ke     = 0.5 * rho * u**2
            rhoe   = rhoE - ke
            T_     = T_from_rhoe(r1, r2, rhoe, T_in=eos._T_cache)
            eos.set_T_cache(T_)
            p_     = srk_p(r1, r2, T_)
            pe_err = float(np.max(np.abs(p_ - p_ref)) / p_ref)
            t_hist.append(t)
            pe_hist.append(pe_err)

            if not np.isfinite(pe_err) or pe_err > 5.0:
                break

    if track_pe:
        return U, t, np.array(t_hist), np.array(pe_hist)
    return U, t


# ═══════════════════════════════════════════════════════════════════════
# Section 4: MUSCL-LLF comparison scheme
# ═══════════════════════════════════════════════════════════════════════

def muscl_lr_prim(q, bc):
    """
    MUSCL reconstruction with minmod limiter.
    q   : (*others, N)   — cell-center values (already ghost-padded if needed)
    bc  : "periodic" or "transmissive"
    Returns qL, qR : (*others, N-1)  left/right states at each interface
    """
    N = q.shape[-1]
    # Build slopes using roll-based difference
    if bc == "periodic":
        dR = np.roll(q, -1, axis=-1) - q      # q[j+1] - q[j]
        dL = q - np.roll(q,  1, axis=-1)      # q[j]   - q[j-1]
    else:
        # For transmissive: pad the endpoints
        dR = np.concatenate([q[..., 1:] - q[..., :-1],
                              np.zeros_like(q[..., :1])], axis=-1)
        dL = np.concatenate([np.zeros_like(q[..., :1]),
                              q[..., 1:] - q[..., :-1]], axis=-1)

    slope = minmod(dR, dL)
    qL = q + 0.5 * slope        # right face of cell j = left state at j+1/2
    qR = q - 0.5 * slope        # left  face of cell j = right state at j-1/2
    # Interface j+1/2: left state from cell j, right state from cell j+1
    return qL[..., :-1], qR[..., 1:]


def muscl_lf_residual(U, axis, dx, eos, ndim, bc, apec=False):
    """
    MUSCL-LLF (Local Lax-Friedrichs) residual.

    Matches apec_1d.py FC-NPE when apec=False.
    If apec=True and eos is SRK2Species: apply APEC PE-consistent energy correction.

    For SRK: closely follows apec_1d.py interface_fluxes() implementation.
    For ideal gas: standard MUSCL-LLF.
    """
    Ns     = eos.Ns
    arr_ax = axis + 1

    # Add ghost cells
    pad_width = [(0,0)] * U.ndim
    pad_width[arr_ax] = (1, 1)
    U_ext = np.pad(U, pad_width, mode='edge')
    _apply_bc(U_ext, axis, bc)

    # Move sweep axis to last
    U_mov = np.moveaxis(U_ext, arr_ax, -1)    # (Nvars, *others, N+2)

    # Cell-center primitives for ALL N+2 cells
    rho_all, u_n_all, p_all, gam_all, a_all = eos.primitives(U_mov, axis, ndim)

    # For SRK: compute cell-center T on interior cells, then pad
    if isinstance(eos, SRK2Species):
        U_int   = _extract_interior(U_ext, arr_ax)   # (Nvars, *spatial_int)
        r1_int  = U_int[0];  r2_int = U_int[1]
        rho_int = np.maximum(r1_int + r2_int, EPS0)
        mom_int = U_int[2:2+ndim]
        u_int   = [mom_int[d] / rho_int for d in range(ndim)]
        rhoe_int = U_int[2+ndim] - 0.5*rho_int*sum(uu**2 for uu in u_int)
        T_int   = T_from_rhoe(r1_int, r2_int, rhoe_int, T_in=eos._T_cache)
        # Pad T to ghost-cell array
        T_ext   = np.pad(T_int,
                         [(1,1) if i == axis else (0,0) for i in range(ndim)],
                         mode='edge')
        T_mov   = np.moveaxis(T_ext, axis, -1)   # (*others, N+2)
        lam_c_all = np.abs(u_n_all) + a_all      # cell-center wave speed
        # lam at interface j+1/2: max(lam_c[j], lam_c[j+1])
        lam_iface = np.maximum(lam_c_all[..., :-1], lam_c_all[..., 1:])

    # MUSCL reconstruction using ghost-padded arrays
    r1_all = U_mov[0];  r2_all = U_mov[1]
    r1_L, r1_R = muscl_lr_prim(r1_all, "transmissive")   # (*others, N+1)
    r2_L, r2_R = muscl_lr_prim(r2_all, "transmissive")
    u_L,  u_R  = muscl_lr_prim(u_n_all, "transmissive")
    p_L,  p_R  = muscl_lr_prim(p_all,   "transmissive")

    rho_L = r1_L + r2_L
    rho_R = r1_R + r2_R

    # Reconstruct energy and wave speeds
    if isinstance(eos, SRK2Species):
        # Match apec_1d.py exactly:
        # rhoEL[j] = srk_rhoe(r1L[j], r2L[j], T[j]) + ke_L
        # rhoER[j] = srk_rhoe(r1R[j], r2R[j], T[j]) + ke_R
        # using the SAME T[j] for both (left-cell T)
        T_L = T_mov[..., :-1]   # T from cell to the LEFT of each interface
        rhoE_L = srk_rhoe(r1_L, r2_L, T_L) + 0.5*rho_L*u_L**2
        rhoE_R = srk_rhoe(r1_R, r2_R, T_L) + 0.5*rho_R*u_R**2
        # Use cell-center lam (from apec_1d.py: lam_cell = |u| + c)
        lam_max = lam_iface
    else:
        # Ideal gas: reconstruct from MUSCL-reconstructed pressure
        if Ns == 1:
            gam_L = eos.gammas[0] * np.ones_like(p_L)
            gam_R = eos.gammas[0] * np.ones_like(p_R)
        else:
            W_s_L = [r_s / np.maximum(rho_L, EPS0)
                     for r_s in [r1_L, r2_L]]
            W_s_R = [r_s / np.maximum(rho_R, EPS0)
                     for r_s in [r1_R, r2_R]]
            cv_L  = sum(eos.cv[s] * W_s_L[s] for s in range(Ns))
            cv_R  = sum(eos.cv[s] * W_s_R[s] for s in range(Ns))
            cp_L  = sum(eos.cp[s] * W_s_L[s] for s in range(Ns))
            cp_R  = sum(eos.cp[s] * W_s_R[s] for s in range(Ns))
            gam_L = cp_L / np.maximum(cv_L, EPS0)
            gam_R = cp_R / np.maximum(cv_R, EPS0)
        rhoE_L = p_L / (gam_L - 1.0) + 0.5*rho_L*u_L**2
        rhoE_R = p_R / (gam_R - 1.0) + 0.5*rho_R*u_R**2
        a_L    = np.sqrt(np.maximum(gam_L * p_L / np.maximum(rho_L, EPS0), EPS0))
        a_R    = np.sqrt(np.maximum(gam_R * p_R / np.maximum(rho_R, EPS0), EPS0))
        lam_max = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    # Build conserved state at L/R
    Nvars = Ns + ndim + 1
    shape_iface = lam_max.shape

    if Ns == 1:
        r_s_L = [r1_L];  r_s_R = [r1_R]
    else:
        r_s_L = [r1_L, r2_L];  r_s_R = [r1_R, r2_R]

    U_L = np.zeros((Nvars,) + shape_iface)
    U_R = np.zeros((Nvars,) + shape_iface)
    for s in range(Ns):
        U_L[s] = r_s_L[s]
        U_R[s] = r_s_R[s]
    for d in range(ndim):
        if d == axis:
            U_L[Ns+d] = rho_L * u_L
            U_R[Ns+d] = rho_R * u_R
        else:
            rhou_d   = U_mov[Ns+d]
            u_d_all  = rhou_d / np.maximum(rho_all, EPS0)
            u_d_L, u_d_R = muscl_lr_prim(u_d_all, "transmissive")
            U_L[Ns+d] = rho_L * u_d_L
            U_R[Ns+d] = rho_R * u_d_R
    U_L[Ns+ndim] = rhoE_L
    U_R[Ns+ndim] = rhoE_R

    FL = _compute_euler_flux_from_prim(r_s_L, u_L, p_L, U_L, axis, ndim, Ns)
    FR = _compute_euler_flux_from_prim(r_s_R, u_R, p_R, U_R, axis, ndim, Ns)

    # LLF flux: G = 0.5*(FL+FR) - 0.5*lam_max*(U_R - U_L)
    Gf = 0.5*(FL + FR) - 0.5*lam_max[np.newaxis]*(U_R - U_L)

    # APEC energy correction for SRK (Eq. A.4 from apec_1d.py)
    if apec and isinstance(eos, SRK2Species):
        eps0_int, eps1_int = eos.get_epsilon(U_int, ndim)
        # Cell-center quantities (padded to N+2)
        eps0_ext = np.pad(eps0_int,
                          [(1,1) if i == axis else (0,0) for i in range(ndim)],
                          mode='edge')
        eps1_ext = np.pad(eps1_int,
                          [(1,1) if i == axis else (0,0) for i in range(ndim)],
                          mode='edge')
        eps0_all = np.moveaxis(eps0_ext, axis, -1)   # (*others, N+2)
        eps1_all = np.moveaxis(eps1_ext, axis, -1)

        # Cell-center fluxes (N+2)
        rho_c_all = np.maximum(rho_all, EPS0)
        F1_cell   = r1_all  * u_n_all
        F2_cell   = r2_all  * u_n_all
        FU_cell   = rho_c_all * u_n_all**2 + p_all
        rhoE_all  = U_mov[2+ndim]
        FE_cell   = (rhoE_all + p_all) * u_n_all

        # Interface fluxes from mass/momentum (from Gf)
        F1_int = Gf[0]    # (*others, N+1)
        F2_int = Gf[1]
        FU_int = Gf[Ns + axis]   # momentum along sweep axis

        # Eq. A.4: construct APEC energy flux using cell-center quantities
        # tm[j]  = (eps0[j]-0.5*u[j]^2)*(F1_int[j]-F1_cell[j]) + ...
        # tm1[j] = (eps0[j+1]-0.5*u[j+1]^2)*(F1_cell[j+1]-F1_int[j]) + ...
        # FE_int[j] = 0.5*(FE_cell[j] + FE_cell[j+1]) + 0.5*tm[j] - 0.5*tm1[j]
        c0  = eps0_all - 0.5*u_n_all**2     # (*others, N+2)
        c1  = eps1_all - 0.5*u_n_all**2

        # Left (cell j) contribution to interface j+1/2
        c0_L   = c0[..., :-1]      # (*others, N+1)
        c1_L   = c1[..., :-1]
        u_L_c  = u_n_all[..., :-1]
        F1_c_L = F1_cell[..., :-1]
        F2_c_L = F2_cell[..., :-1]
        FU_c_L = FU_cell[..., :-1]
        FE_c_L = FE_cell[..., :-1]

        # Right (cell j+1) contribution to interface j+1/2
        c0_R   = c0[..., 1:]       # (*others, N+1)
        c1_R   = c1[..., 1:]
        u_R_c  = u_n_all[..., 1:]
        F1_c_R = F1_cell[..., 1:]
        F2_c_R = F2_cell[..., 1:]
        FU_c_R = FU_cell[..., 1:]
        FE_c_R = FE_cell[..., 1:]

        tm  = (c0_L*(F1_int - F1_c_L) + c1_L*(F2_int - F2_c_L)
             + u_L_c*(FU_int - FU_c_L))
        tm1 = (c0_R*(F1_c_R - F1_int) + c1_R*(F2_c_R - F2_int)
             + u_R_c*(FU_c_R - FU_int))

        FE_apec = 0.5*(FE_c_L + FE_c_R) + 0.5*tm - 0.5*tm1
        Gf[Ns+ndim] = FE_apec

    # Residual
    R_mov = (Gf[..., 1:] - Gf[..., :-1]) / dx
    return np.moveaxis(R_mov, -1, arr_ax)


def _compute_euler_flux_from_prim(r_s, u_n, p, U_cons, axis, ndim, Ns):
    """Compute Euler flux from primitive variables."""
    rho  = sum(r_s)
    Nvars = Ns + ndim + 1
    F    = np.zeros_like(U_cons)
    # Species
    for s in range(Ns):
        F[s] = r_s[s] * u_n
    # Momentum
    for d in range(ndim):
        F[Ns+d] = U_cons[Ns+d] * u_n
        if d == axis:
            F[Ns+d] += p
    # Energy
    F[Ns+ndim] = (U_cons[Ns+ndim] + p) * u_n
    return F


def run_simulation_muscl_lf(U0, dx_list, t_end, eos, sigma=CFL_DEFAULT,
                             bc="transmissive", ndim=None, max_steps=2_000_000,
                             apec=False, track_pe=False, p_ref=5e6):
    """MUSCL-LLF simulation runner."""
    U    = U0.copy()
    if ndim is None:
        ndim = U.ndim - 1
    t    = 0.0
    order = 1
    t_hist  = [0.0]
    pe_hist = []

    for _ in range(max_steps):
        if t >= t_end:
            break
        dt = compute_dt(U, eos, dx_list, order, sigma, ndim)
        dt = min(dt, t_end - t)

        # Forward Euler (MUSCL-LLF)
        if ndim == 1:
            R = muscl_lf_residual(U, 0, dx_list[0], eos, ndim, bc, apec=apec)
            U = U - dt * R
        else:
            # Strang splitting for 2D
            for sub_axis, sub_dx, sub_dt in [(0, dx_list[0], dt/2),
                                              (1, dx_list[1], dt  ),
                                              (0, dx_list[0], dt/2)]:
                R = muscl_lf_residual(U, sub_axis, sub_dx, eos, ndim, bc, apec=apec)
                U = U - sub_dt * R
        t += dt

        if track_pe:
            if ndim == 1:
                r1, r2 = U[0], U[1]
                rho    = np.maximum(r1 + r2, EPS0)
                u_     = U[2] / rho
                ke_    = 0.5 * rho * u_**2
                rhoe_  = U[2+ndim] - ke_
                T_     = T_from_rhoe(r1, r2, rhoe_, T_in=eos._T_cache if hasattr(eos, '_T_cache') else None)
                if hasattr(eos, 'set_T_cache'):
                    eos.set_T_cache(T_)
                p_     = srk_p(r1, r2, T_)
                pe_err = float(np.max(np.abs(p_ - p_ref)) / p_ref)
                t_hist.append(t)
                pe_hist.append(pe_err)
                if not np.isfinite(pe_err) or pe_err > 5.0:
                    break

    if track_pe:
        return U, t, np.array(t_hist), np.array(pe_hist)
    return U, t


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Helper functions
# ═══════════════════════════════════════════════════════════════════════

def prim_to_cons_2s(W, rho, u, p, gam1, gam2, ndim=1):
    """2-species ideal gas: primitive to conserved."""
    cv1 = 1.0 / (gam1 - 1.0)
    cv2 = 1.0 / (gam2 - 1.0)
    cv  = W * cv1 + (1.0 - W) * cv2
    cp  = W * gam1 * cv1 + (1.0 - W) * gam2 * cv2
    gam = cp / (cv + EPS0)
    rhoE = p / (gam - 1.0) + 0.5 * rho * u**2
    return np.array([rho * W, rho * (1-W), rho * u, rhoE])


def cons_to_prim_2s(U, gam1, gam2, ndim=1):
    """2-species ideal gas: conserved to primitive."""
    eos_ = IdealGasMixture([gam1, gam2])
    rho, u_n, p, gam, a = eos_.primitives(U, 0, ndim)
    W = np.clip(U[0] / (rho + EPS0), 0.0, 1.0)
    return W, rho, u_n, p


def srk_initial_condition(x, p_inf=5e6, k=15.0):
    """
    SRK CH4/N2 interface advection initial condition.
    Returns (rho1, rho2, u, rhoE, T, p).
    """
    N  = len(x)
    xc = 0.5
    rc = 0.25
    r1_inf = 400.0
    r2_inf = 100.0
    r  = np.abs(x - xc)
    r1 = 0.5*r1_inf*(1.0 - np.tanh(k*(r - rc)))
    r2 = 0.5*r2_inf*(1.0 + np.tanh(k*(r - rc)))

    print("  Solving T from p_inf=5MPa (SRK Newton)...", flush=True)
    _, Mb, *_ = _mix(r1, r2, np.full(N, 300.0))
    T = np.clip(p_inf * Mb / np.maximum(r1 + r2, 1e-30) / Ru, 50.0, 1000.0)
    for _ in range(60):
        ph  = srk_p(r1, r2, T)
        dph = _dpdT(r1, r2, T)
        dT  = np.clip(-(ph - p_inf) / (dph + 1e-3), -100.0, 100.0)
        T   = np.clip(T + dT, 10.0, 2000.0)
        if np.max(np.abs(dT)) < 1e-3:
            break

    p    = srk_p(r1, r2, T)
    rhoe = srk_rhoe(r1, r2, T)
    u    = np.full(N, 100.0)
    return r1, r2, u, rhoe + 0.5*(r1+r2)*u**2, T, p


def _shock_tube_ic_1d(Nx, WL, rhoL, uL, pL, WR, rhoR, uR, pR, gam1, gam2):
    """Build 1D two-species Riemann IC."""
    dx = 1.0 / Nx
    x  = (np.arange(Nx) + 0.5) * dx
    W   = np.where(x <= 0.5, WL,   WR)
    rho = np.where(x <= 0.5, rhoL, rhoR)
    u   = np.where(x <= 0.5, uL,   uR)
    p   = np.where(x <= 0.5, pL,   pR)
    return x, prim_to_cons_2s(W, rho, u, p, gam1, gam2)


def _eoc_ic_1d(Nx, gam1, gam2):
    """EOC smooth IC."""
    domain = 2.0
    dx = domain / Nx
    x  = (np.arange(Nx) + 0.5) * dx
    h  = dx / 2.0
    sinc_h = np.sin(np.pi * h) / (np.pi * h)
    rho0 = 1.0 + 0.2 * np.sin(np.pi * x) * sinc_h
    W0   = 0.5 * np.ones(Nx)
    u0   = 0.1 * np.ones(Nx)
    p0   = 0.5 * np.ones(Nx)
    return x, prim_to_cons_2s(W0, rho0, u0, p0, gam1, gam2), dx


def _eoc_exact_rho(x, t, dx):
    """Cell-averaged exact density for EOC test."""
    h      = dx / 2.0
    sinc_h = np.sin(np.pi * h) / (np.pi * h)
    return 1.0 + 0.2 * np.sin(np.pi * (x - 0.1 * t)) * sinc_h


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Test cases
# ═══════════════════════════════════════════════════════════════════════

def run_test_eoc(verbose=True):
    """EOC test matching lambda_diff_nd.py."""
    print("\n[Test] EOC (Periodic, 2-species ideal gas)")
    gam1 = gam2 = 1.4
    eos  = IdealGasMixture([gam1, gam2])
    t_end   = 0.5
    domain  = 2.0
    Nx_list = [40, 80, 160, 320, 640, 1280]
    label_map = {1: "1O", 2: "2O (limited)", 3: "3O (limited)", 4: "3O (unlimited)"}
    all_pass = True

    def run_order(order):
        nonlocal all_pass
        print(f"\n  {label_map[order]}")
        print(f"  {'Nx':>6}  {'dx':>10}  {'L1 error':>14}  {'EOC_L1':>8}  "
              f"{'L2 error':>14}  {'EOC_L2':>8}")
        prev_e1, prev_e2 = None, None
        errors_l1 = []
        for Nx in Nx_list:
            dx = domain / Nx
            x  = (np.arange(Nx) + 0.5) * dx
            h  = dx / 2.0
            sinc_h = np.sin(np.pi * h) / (np.pi * h) if np.pi*h > 1e-14 else 1.0
            rho0 = 1.0 + 0.2 * np.sin(np.pi * x) * sinc_h
            W0   = 0.5 * np.ones(Nx)
            u0   = 0.1 * np.ones(Nx)
            p0   = 0.5 * np.ones(Nx)
            U0   = prim_to_cons_2s(W0, rho0, u0, p0, gam1, gam2)
            U, _ = run_simulation(U0, [dx], t_end, eos,
                                  order=order, sigma=CFL_DEFAULT,
                                  bc="periodic", ndim=1, max_steps=2_000_000)
            rho_num = U[0] + U[1]
            rho_ex  = _eoc_exact_rho(x, t_end, dx)
            e1 = dx * np.sum(np.abs(rho_num - rho_ex))
            e2 = np.sqrt(dx * np.sum((rho_num - rho_ex)**2))
            if prev_e1 is not None:
                eoc1 = np.log2(prev_e1 / e1)
                eoc2 = np.log2(prev_e2 / e2)
                print(f"  {Nx:>6}  {dx:>10.7f}  {e1:>14.10f}  {eoc1:>8.6f}  "
                      f"{e2:>14.10f}  {eoc2:>8.6f}")
            else:
                print(f"  {Nx:>6}  {dx:>10.7f}  {e1:>14.10f}  {'':>8}  "
                      f"{e2:>14.10f}  {'':>8}")
            errors_l1.append(e1)
            prev_e1, prev_e2 = e1, e2

        ref = {1: 0.0125233040, 2: 0.0019864789,
               3: 0.0003870968, 4: 0.0000548925}
        tol = 0.05
        if order in ref:
            got  = errors_l1[0]
            want = ref[order]
            ok   = abs(got - want) / want < tol
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  Nx=40 L1 ref={want:.10f} got={got:.10f} [{status}]")

    for order in [1, 2, 3, 4]:
        run_order(order)

    return all_pass


def run_test_sod(Nx=200, sigma=CFL_DEFAULT):
    """1D Sod shock tube with kinetic scheme."""
    print("\n[Test] Sod shock tube (1D)")
    gam1 = gam2 = 1.4
    eos  = IdealGasMixture([gam1, gam2])
    t_end = 0.1

    x, U0 = _shock_tube_ic_1d(Nx, 1.0, 2.0, 0.0, 10.0,
                                   0.0, 1.0, 0.0,  1.0, gam1, gam2)
    dx = 1.0 / Nx

    results, labels = [], []
    for order, lbl in [(1, "1O"), (2, "2O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), [dx], t_end, eos,
                               order=order, sigma=sigma, ndim=1)
        W, rho, u, p = cons_to_prim_2s(U, gam1, gam2)
        results.append((W, rho, u, p))
        labels.append(lbl)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, idx) in zip(axes, [("rho", 1), ("u", 2), ("p", 3)]):
        for (W, rho, u, p), lbl in zip(results, labels):
            vals = [W, rho, u, p][idx]
            ax.plot(x, vals, lw=1.2, label=lbl)
        ax.set_title(name); ax.legend(fontsize=7)
    plt.suptitle(f"Sod shock tube (t={t_end}, Nx={Nx})")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "sod_kinetic.png")
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    W, rho, u, p = results[-1]
    ok = (p.min() > 0.5) and (p.max() < 11.0)
    print(f"  Pressure range: [{p.min():.3f}, {p.max():.3f}] -- {'PASS' if ok else 'FAIL'}")
    return ok


def run_test_kh(Nx=128, Ny=128, sigma=0.4):
    """
    2D Kelvin-Helmholtz instability (single species, gamma=5/3).
    McNally et al. 2012 initial conditions.
    """
    print(f"\n[Test] Kelvin-Helmholtz (2D, {Nx}x{Ny}, gamma=5/3)")
    gam = 5.0/3.0
    eos = IdealGasMixture([gam])   # Ns=1

    Lx, Ly = 1.0, 1.0
    dx = Lx / Nx
    dy = Ly / Ny
    x  = (np.arange(Nx) + 0.5) * dx
    y  = (np.arange(Ny) + 0.5) * dy

    XX, YY = np.meshgrid(x, y)   # (Ny, Nx)

    delta = 0.05    # interface thickness
    sigma_kh = 0.05   # perturbation width
    v0 = 0.1

    # Smooth tanh layers
    f_y = 0.5*(np.tanh((YY - 0.25)/delta) - np.tanh((YY - 0.75)/delta))
    rho = 1.0 + f_y               # 1 outside, 2 inside
    ux  = -0.5 + f_y              # -0.5 outside, +0.5 inside
    uy  = v0 * np.sin(2*np.pi*XX) * (
          np.exp(-(YY - 0.25)**2 / (2*sigma_kh**2))
        + np.exp(-(YY - 0.75)**2 / (2*sigma_kh**2)))
    p   = np.full_like(rho, 2.5)

    # State vector: [rho, rho*ux, rho*uy, rhoE]  (Ns=1, ndim=2)
    # Nvars = 1+2+1 = 4
    rhoE = p / (gam - 1.0) + 0.5*rho*(ux**2 + uy**2)
    U0 = np.zeros((4, Ny, Nx))
    U0[0] = rho
    U0[1] = rho * uy    # axis 0 = y
    U0[2] = rho * ux    # axis 1 = x
    U0[3] = rhoE

    t_end  = 2.0
    plot_times = [0.0, 0.5, 1.0, 2.0]
    snapshots  = []
    snapshots.append(rho.copy())   # t=0

    t_targets = plot_times[1:]
    U = U0.copy()
    t = 0.0

    for t_target in t_targets:
        while t < t_target - 1e-12:
            dt = compute_dt(U, eos, [dy, dx], 3, sigma, ndim=2)
            dt = min(dt, t_target - t)
            U  = step_ssprk3(U, [dy, dx], dt, eos, order=3, bc="periodic", ndim=2)
            t += dt
        # Record density
        rho_snap = U[0].copy()
        snapshots.append(rho_snap)
        print(f"  t={t_target:.1f}: rho in [{rho_snap.min():.3f}, {rho_snap.max():.3f}]")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, snap, tt in zip(axes, snapshots, plot_times):
        im = ax.contourf(x, y, snap, levels=30, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"t={tt:.1f}")
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.suptitle("KH Instability: density (gamma=5/3, 128x128, order=3)")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "kh_density.png")
    plt.savefig(fname, dpi=100); plt.close()
    print(f"  Saved: {fname}")

    # Check: KH should roll up — density range should grow from initial
    rho_init_range = snapshots[0].max() - snapshots[0].min()
    rho_final_range = snapshots[-1].max() - snapshots[-1].min()

    # Compute vorticity at final time
    # omega_z = duy/dx - dux/dy
    uy_fin = U[1] / np.maximum(U[0], EPS0)
    ux_fin = U[2] / np.maximum(U[0], EPS0)
    duy_dx = (np.roll(uy_fin, -1, axis=1) - np.roll(uy_fin, 1, axis=1)) / (2*dx)
    dux_dy = (np.roll(ux_fin, -1, axis=0) - np.roll(ux_fin, 1, axis=0)) / (2*dy)
    omega  = duy_dx - dux_dy
    max_vort = float(np.max(np.abs(omega)))

    print(f"  Initial density range: {rho_init_range:.3f}")
    print(f"  Final   density range: {rho_final_range:.3f}")
    print(f"  Max vorticity at t=2:  {max_vort:.3f}")

    # KH should produce some vorticity — threshold is loose
    ok = max_vort > 5.0
    print(f"  Max vorticity > 5: {'PASS' if ok else 'FAIL'}")
    return ok


def run_test_riemann2d(Nx=128, Ny=128, sigma=0.4):
    """
    2D Riemann problem, Config 3 (Lax & Liu 1998).
    Single species, gamma=1.4.
    """
    print(f"\n[Test] 2D Riemann problem Config 3 ({Nx}x{Ny}, gamma=1.4)")
    gam = 1.4
    eos = IdealGasMixture([gam])

    dx = 1.0 / Nx
    dy = 1.0 / Ny
    x  = (np.arange(Nx) + 0.5) * dx
    y  = (np.arange(Ny) + 0.5) * dy
    XX, YY = np.meshgrid(x, y)  # (Ny, Nx)

    # Config 3: 4 quadrants at x=y=0.5
    rho = np.where(XX > 0.5,
                   np.where(YY > 0.5, 1.5,    0.5323),
                   np.where(YY > 0.5, 0.5323, 0.138 ))
    ux  = np.where(XX > 0.5,
                   np.where(YY > 0.5, 0.0,    1.206),
                   np.where(YY > 0.5, 1.206,  1.206))
    uy  = np.where(XX > 0.5,
                   np.where(YY > 0.5, 0.0,    0.0  ),
                   np.where(YY > 0.5, 0.0,    1.206))
    p   = np.where(XX > 0.5,
                   np.where(YY > 0.5, 1.5,    0.3  ),
                   np.where(YY > 0.5, 0.3,    0.029))

    rhoE = p / (gam - 1.0) + 0.5*rho*(ux**2 + uy**2)
    U0   = np.zeros((4, Ny, Nx))
    U0[0] = rho
    U0[1] = rho * uy   # axis 0 = y
    U0[2] = rho * ux   # axis 1 = x
    U0[3] = rhoE

    t_end = 0.3
    U, t  = run_simulation(U0, [dy, dx], t_end, eos,
                            order=1, sigma=sigma, bc="transmissive", ndim=2)

    rho_fin = U[0]
    print(f"  t_end={t:.4f}: rho in [{rho_fin.min():.4f}, {rho_fin.max():.4f}]")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].contourf(x, y, rho_fin, levels=30, cmap='plasma')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_title(f"Density at t={t:.3f}")
    axes[0].set_aspect('equal')
    im1 = axes[1].contourf(x, y, rho_fin, levels=60, cmap='plasma')
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_title("Density (fine contours)")
    axes[1].set_aspect('equal')
    plt.suptitle("2D Riemann Config 3 (t=0.3, 128x128, order=1)")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "riemann2d_density.png")
    plt.savefig(fname, dpi=100); plt.close()
    print(f"  Saved: {fname}")

    # Physical check: density should be positive and in a reasonable range
    ok = (rho_fin.min() > 0.0) and (rho_fin.max() < 5.0)
    print(f"  Density physically reasonable: {'PASS' if ok else 'FAIL'}")
    return ok


def _run_srk_scheme(scheme_name, U0, T_init, dx, sigma, t_end, p_inf,
                    use_apec_kinetic=False, use_muscl=False, use_apec_muscl=False):
    """Helper: run one SRK scheme and return (t_hist, pe_hist)."""
    eos = SRK2Species()
    eos.set_T_cache(T_init.copy())
    U  = U0.copy()
    t  = 0.0
    t_hist  = [0.0]
    pe_hist = []
    ndim    = 1
    order   = 1

    for _ in range(2_000_000):
        if t >= t_end:
            break
        dt = compute_dt(U, eos, [dx], order, sigma, ndim)
        dt = min(dt, t_end - t)

        if use_muscl:
            R = muscl_lf_residual(U, 0, dx, eos, ndim, "periodic",
                                  apec=use_apec_muscl)
        elif use_apec_kinetic:
            R = residual_axis_apec(U, 0, dx, eos, "periodic", ndim)
        else:
            R = residual_axis(U, 0, dx, eos, order, "periodic", ndim)

        U  = U - dt * R
        t += dt

        r1_  = U[0]; r2_ = U[1]
        rho_ = np.maximum(r1_ + r2_, EPS0)
        u_   = U[2] / rho_
        rhoe_= U[3] - 0.5*rho_*u_**2
        T_   = T_from_rhoe(r1_, r2_, rhoe_, T_in=eos._T_cache)
        eos.set_T_cache(T_)
        p_   = srk_p(r1_, r2_, T_)
        pe_  = float(np.max(np.abs(p_ - p_inf)) / p_inf)
        t_hist.append(t)
        pe_hist.append(pe_)
        if not np.isfinite(pe_) or pe_ > 5.0:
            print(f"  {scheme_name}: diverged at t={t:.5f}")
            break

    final_pe = pe_hist[-1] if pe_hist else np.inf
    status   = "completed" if (not pe_hist or final_pe <= 5.0) else "diverged"
    print(f"  {scheme_name}: {status} at t={t:.5f}, PE={final_pe:.2e}")
    return np.array(t_hist), np.array(pe_hist)


def run_test_srk_apec(N=101, sigma=0.3, t_end=5e-3):
    """
    SRK CH4/N2 interface advection PE comparison.

    Compares three schemes:
      1. Kinetic (standard kinetic lambda-difference, no APEC)
      2. MUSCL-LLF / FC-NPE (as in apec_1d.py, no APEC)
      3. MUSCL-LLF + APEC (PE-consistent energy correction)

    Test passes if MUSCL-LLF+APEC PE error < MUSCL-LLF PE error at t_end,
    consistent with the validated results in apec_1d.py.
    Also includes Kinetic+APEC attempt (may diverge — shown in plot for reference).
    """
    print(f"\n[Test] SRK APEC PE comparison (N={N}, CFL={sigma}, t_end={t_end:.3f}s)")
    dx   = 1.0 / N
    x    = np.linspace(dx/2, 1 - dx/2, N)
    p_inf = 5e6

    r1, r2, u_init, rhoE_init, T_init, p_init = srk_initial_condition(x, p_inf=p_inf)
    print(f"  IC: max|p-p_inf|/p_inf = {float(np.max(np.abs(p_init - p_inf))/p_inf):.2e}")

    U0 = np.zeros((4, N))
    U0[0] = r1;  U0[1] = r2
    U0[2] = (r1 + r2) * u_init;  U0[3] = rhoE_init

    # --- Diagnostic: compare initial max lambda for kinetic vs MUSCL-LLF ---
    _eos_diag = SRK2Species()
    _eos_diag.set_T_cache(T_init.copy())
    _U0_ext = np.pad(U0, [(0,0),(1,1)], mode='edge')
    _U0_mov = np.moveaxis(_U0_ext, 1, -1)
    _F0_ext = _eos_diag.flux(_U0_ext, 0, 1)
    _F0_mov = np.moveaxis(_F0_ext, 1, -1)
    _lam_kin = compute_lambda(_U0_mov, _F0_mov, 0, 1, _eos_diag)
    # MUSCL-LLF lambda: cell-center |u| + a, max over neighbors
    _rho_all, _un_all, _p_all, _, _a_all = _eos_diag.primitives(_U0_mov, 0, 1)
    _lam_c = np.abs(_un_all) + _a_all
    _lam_ml = np.maximum(_lam_c[:-1], _lam_c[1:])
    print(f"  Diagnostic: initial max kinetic lambda = {float(np.max(_lam_kin)):.4e}")
    print(f"  Diagnostic: initial max MUSCL-LLF lambda = {float(np.max(_lam_ml)):.4e}")
    del _eos_diag, _U0_ext, _U0_mov, _F0_ext, _F0_mov
    del _lam_kin, _rho_all, _un_all, _p_all, _a_all, _lam_c, _lam_ml
    # -----------------------------------------------------------------------

    results = {}

    print("  Running Kinetic (standard)...")
    results['Kinetic'] = _run_srk_scheme(
        'Kinetic', U0, T_init, dx, sigma, t_end, p_inf,
        use_apec_kinetic=False, use_muscl=False)

    print("  Running Kinetic+APEC (experimental)...")
    results['Kinetic+APEC'] = _run_srk_scheme(
        'Kinetic+APEC', U0, T_init, dx, sigma, t_end, p_inf,
        use_apec_kinetic=True, use_muscl=False)

    print("  Running MUSCL-LLF (FC-NPE)...")
    results['MUSCL-LLF'] = _run_srk_scheme(
        'MUSCL-LLF', U0, T_init, dx, sigma, t_end, p_inf,
        use_muscl=True, use_apec_muscl=False)

    print("  Running MUSCL-LLF+APEC...")
    results['MUSCL-LLF+APEC'] = _run_srk_scheme(
        'MUSCL-LLF+APEC', U0, T_init, dx, sigma, t_end, p_inf,
        use_muscl=True, use_apec_muscl=True)

    # Plot PE vs time
    colors = {'Kinetic': 'blue', 'Kinetic+APEC': 'cyan',
              'MUSCL-LLF': 'red', 'MUSCL-LLF+APEC': 'green'}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (th, pe) in results.items():
        if len(pe) > 0:
            ax.semilogy(th[1:len(pe)+1], pe, label=name,
                        color=colors[name], lw=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('PE error max|p-p_inf|/p_inf')
    ax.set_title(f'SRK APEC PE comparison (N={N}, CFL={sigma})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "srk_apec_pe.png")
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    # Test criterion: MUSCL-LLF+APEC < MUSCL-LLF (the validated result)
    pe_mf    = results['MUSCL-LLF'][1]
    pe_apec  = results['MUSCL-LLF+APEC'][1]
    pe_mf_final   = pe_mf[-1]  if len(pe_mf)  > 0 else np.inf
    pe_apec_final = pe_apec[-1] if len(pe_apec) > 0 else np.inf

    ok = pe_apec_final < pe_mf_final
    print(f"  MUSCL-LLF PE={pe_mf_final:.2e}, MUSCL-LLF+APEC PE={pe_apec_final:.2e}")
    print(f"  APEC < FC-NPE: {'PASS' if ok else 'FAIL'}")
    return ok


def run_test_compare(Nx=200, sigma=CFL_DEFAULT):
    """
    Method comparison on 1D Sod: Kinetic (order=3) vs MUSCL-LLF (order=1).
    """
    print("\n[Test] Method comparison on 1D Sod")
    gam1 = gam2 = 1.4
    eos  = IdealGasMixture([gam1, gam2])
    t_end = 0.1

    x, U0 = _shock_tube_ic_1d(Nx, 1.0, 2.0, 0.0, 10.0,
                                   0.0, 1.0, 0.0,  1.0, gam1, gam2)
    dx = 1.0 / Nx

    schemes = []

    # Kinetic orders 1, 2, 3
    for order, lbl in [(1, "Kinetic 1O"), (2, "Kinetic 2O"), (3, "Kinetic 3O")]:
        U, _ = run_simulation(U0.copy(), [dx], t_end, eos,
                               order=order, sigma=sigma, ndim=1)
        W, rho, u, p = cons_to_prim_2s(U, gam1, gam2)
        schemes.append((lbl, rho, u, p))

    # MUSCL-LLF
    for apec_flag, lbl in [(False, "MUSCL-LLF"), (True, "MUSCL-LLF+APEC (ideal)")]:
        U_mf = U0.copy()
        t    = 0.0
        for _ in range(2_000_000):
            if t >= t_end:
                break
            dt = compute_dt(U_mf, eos, [dx], 1, sigma, ndim=1)
            dt = min(dt, t_end - t)
            R  = muscl_lf_residual(U_mf, 0, dx, eos, 1, "transmissive", apec=apec_flag)
            U_mf = U_mf - dt * R
            t   += dt
        W_mf, rho_mf, u_mf, p_mf = cons_to_prim_2s(U_mf, gam1, gam2)
        schemes.append((lbl, rho_mf, u_mf, p_mf))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    lss = ['-', '--', ':', '-.', (0, (5,1))]
    for (lbl, rho, u, p), ls in zip(schemes, lss):
        axes[0].plot(x, rho, ls=ls, lw=1.5, label=lbl)
        axes[1].plot(x, u,   ls=ls, lw=1.5, label=lbl)
        axes[2].plot(x, p,   ls=ls, lw=1.5, label=lbl)
    axes[0].set_title("Density"); axes[0].legend(fontsize=7)
    axes[1].set_title("Velocity")
    axes[2].set_title("Pressure")
    for ax in axes:
        ax.set_xlabel('x')
    plt.suptitle(f"Method comparison: Sod shock tube (t={t_end})")
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "compare_sod.png")
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    # Check all schemes give physical results
    all_ok = True
    for lbl, rho, u, p in schemes:
        ok = (p.min() > 0.5) and (p.max() < 11.0) and (rho.min() > 0.0)
        all_ok = all_ok and ok
        print(f"  {lbl}: p=[{p.min():.2f},{p.max():.2f}] -- {'PASS' if ok else 'FAIL'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Kinetic Real-Gas Solver Tests")
    parser.add_argument("--test", default="all",
                        choices=["eoc", "sod", "kh", "riemann2d",
                                 "srk_apec", "compare", "all"])
    args = parser.parse_args()

    results = {}

    if args.test in ("eoc", "all"):
        results["eoc"]      = run_test_eoc()
    if args.test in ("sod", "all"):
        results["sod"]      = run_test_sod()
    if args.test in ("kh", "all"):
        results["kh"]       = run_test_kh()
    if args.test in ("riemann2d", "all"):
        results["riemann2d"] = run_test_riemann2d()
    if args.test in ("srk_apec", "all"):
        results["srk_apec"] = run_test_srk_apec()
    if args.test in ("compare", "all"):
        results["compare"]  = run_test_compare()

    print("\n" + "="*55)
    print("SUMMARY")
    print("="*55)
    all_pass = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:20s}: {status}")
    print("="*55)
    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
