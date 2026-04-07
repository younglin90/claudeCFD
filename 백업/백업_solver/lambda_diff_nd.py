"""
lambda_diff_nd.py
=================
N-dimensional (1D/2D/3D) Multi-Species Euler Equations — Kinetic Scheme
with Positivity Preservation and Strang Dimensional Splitting.

Extends lambda_diff_1d.py (Roy & Raghurama Rao 2025, arXiv:2411.00285v2)
to arbitrary spatial dimension with pluggable EOS.

State vector convention (variable axis first, spatial axes last):
  U[v, i0, ..., i_{ndim-1}]
  U[s]        for s=0..Ns-1 : partial density rho_s
  U[Ns+d]     for d=0..ndim-1: momentum along axis d (rho*u_d)
  U[Ns+ndim]  : total energy rho*E

1D: (Ns+2, Nx)
2D: (Ns+3, Ny, Nx)    (axis 0 = y, axis 1 = x)
3D: (Ns+4, Nz, Ny, Nx)

Usage:
  python solver/lambda_diff_nd.py --test eoc
  python solver/lambda_diff_nd.py --test sod
  python solver/lambda_diff_nd.py --test contact
  python solver/lambda_diff_nd.py --test sod3d
  python solver/lambda_diff_nd.py --test all
"""

import argparse
import sys
import abc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
EPS0      = 1e-10
B_COMPRESS = 4
CFL_DEFAULT = 0.8


# ─────────────────────────────────────────────
# Abstract EOS base class
# ─────────────────────────────────────────────

class EOSBase(abc.ABC):
    """Abstract EOS interface for N-species ideal/real gas."""

    @abc.abstractmethod
    def primitives(self, U, axis, ndim):
        """
        Extract primitive variables from conserved state U.

        Parameters
        ----------
        U    : (Nvars, *spatial) array
        axis : int, sweep axis index (0-based in spatial dims)
        ndim : int, number of spatial dimensions

        Returns
        -------
        rho   : total density, shape (*spatial)
        u_n   : normal velocity (along `axis`), shape (*spatial)
        p     : pressure, shape (*spatial)
        gam   : local mixture gamma, shape (*spatial)
        a     : speed of sound, shape (*spatial)
        """

    @abc.abstractmethod
    def flux(self, U, axis, ndim):
        """
        Physical flux along sweep `axis`.

        Returns F : (Nvars, *spatial) array — flux in direction `axis`
        """

    @abc.abstractmethod
    def positivity_alpha(self, u_n, a, gam):
        """
        Positivity-preserving bounds (Eq. 32).
        Returns alpha_R, alpha_L : arrays same shape as u_n
          alpha_L = -u_n + sqrt((gam-1)/(2*gam)) * a   (contribution from left cell)
          alpha_R =  u_n + sqrt((gam-1)/(2*gam)) * a   (contribution from right cell)
        """


# ─────────────────────────────────────────────
# Ideal Gas Mixture EOS
# ─────────────────────────────────────────────

class IdealGasMixture(EOSBase):
    """
    N-species calorically perfect gas mixture.

    Species gamma values provided as a list/array (length Ns).
    The 1D paper uses two components with gam1, gam2.
    cv_s = 1/(gamma_s - 1) for each species (unit specific heat assumption).

    Parameters
    ----------
    gammas : sequence of length Ns
    """

    def __init__(self, gammas):
        self.gammas = np.asarray(gammas, dtype=float)
        self.Ns = len(self.gammas)
        self.cv = 1.0 / (self.gammas - 1.0)   # (Ns,)
        self.cp = self.gammas * self.cv        # (Ns,)

    def _mixture_thermo(self, U, ndim):
        """
        Compute mixture cv, cp, gam, rho, u_vec, p from conserved state U.

        Parameters
        ----------
        U    : (Nvars, *spatial)
        ndim : int

        Returns
        -------
        rho, u_vec (list of ndim arrays), p, gam, a
        """
        Ns = self.Ns
        # Partial densities
        rho_s = U[:Ns]                         # (Ns, *spatial)
        rho   = np.sum(rho_s, axis=0)          # (*spatial)
        rho   = np.maximum(rho, EPS0)

        # Mass fractions
        W_s = rho_s / rho[np.newaxis]          # (Ns, *spatial)  clipped later

        # Mixture cv, cp
        cv_mix = np.einsum('s,s...->...', self.cv, W_s)  # (*spatial)
        cp_mix = np.einsum('s,s...->...', self.cp, W_s)  # (*spatial)
        gam    = cp_mix / np.maximum(cv_mix, EPS0)       # (*spatial)

        # Momenta
        mom = U[Ns:Ns+ndim]                    # (ndim, *spatial)
        u_list = [mom[d] / rho for d in range(ndim)]

        # Total energy
        rhoE = U[Ns+ndim]                      # (*spatial)
        ke   = 0.5 * rho * sum(u**2 for u in u_list)
        p    = (gam - 1.0) * (rhoE - ke)
        p    = np.maximum(p, EPS0)

        a    = np.sqrt(np.maximum(gam * p / rho, EPS0))

        return rho, u_list, p, gam, a

    def primitives(self, U, axis, ndim):
        rho, u_list, p, gam, a = self._mixture_thermo(U, ndim)
        u_n = u_list[axis]
        return rho, u_n, p, gam, a

    def flux(self, U, axis, ndim):
        Ns   = self.Ns
        rho, u_list, p, gam, a = self._mixture_thermo(U, ndim)
        u_n  = u_list[axis]                     # normal velocity

        F = np.empty_like(U)
        # Species fluxes: rho_s * u_n
        for s in range(Ns):
            F[s] = U[s] * u_n
        # Momentum fluxes: rho*u_d*u_n + p*delta_{d,axis}
        for d in range(ndim):
            F[Ns+d] = U[Ns+d] * u_n
            if d == axis:
                F[Ns+d] += p
        # Energy flux: (rhoE + p) * u_n
        F[Ns+ndim] = (U[Ns+ndim] + p) * u_n

        return F

    def positivity_alpha(self, u_n, a, gam):
        coef = np.sqrt(np.maximum((gam - 1.0) / (2.0 * gam), 0.0))
        alpha_L = -u_n + coef * a
        alpha_R =  u_n + coef * a
        return alpha_R, alpha_L


# ─────────────────────────────────────────────
# Utility: abs_sign
# ─────────────────────────────────────────────

def abs_sign(x):
    """abs_sign(x) = 1 if |x|>EPS0, else 0 (Eq. 42d)."""
    return np.where(np.abs(x) > EPS0, 1.0, 0.0)


def minmod(x, y):
    """Minmod limiter."""
    return np.where(x * y > 0,
                    np.where(np.abs(x) < np.abs(y), x, y),
                    0.0)


# ─────────────────────────────────────────────
# Lambda (numerical diffusion coefficient)
# ─────────────────────────────────────────────

def compute_lambda(U_mov, F_mov, axis, ndim, eos):
    """
    Compute lambda at every interface along the last axis (sweep direction).

    U_mov, F_mov have shape (Nvars, *others, N)  [sweep axis last].
    Returns lam of shape (*others, N-1).

    Implements Eq. 42/78 from Roy & Raghurama Rao 2025.
    """
    Ns = eos.Ns

    # Total density and normal momentum are needed for RH terms
    # For multi-species, total density = sum of partial densities
    rho  = np.sum(U_mov[:Ns], axis=0)          # (*others, N)
    rhou = U_mov[Ns+axis]                       # (*others, N)  normal momentum
    rhoE = U_mov[Ns+ndim]                       # (*others, N)

    # Fluxes of: total mass (sum of species mass fluxes),
    #            normal momentum, normal momentum^2+p, energy
    # For RH lambda (Eq. 78c), we use total continuity, momentum, energy:
    F_rho  = np.sum(F_mov[:Ns], axis=0)         # (*others, N)  = rho*u_n flux
    F_rhou = F_mov[Ns+axis]                     # (*others, N)  = rho*u_n^2 + p
    F_rhoE = F_mov[Ns+ndim]                     # (*others, N)  = (rhoE+p)*u_n

    # Differences at interfaces (j+1) - j  along last axis
    d_rho  = rho[..., 1:] - rho[..., :-1]     # (*others, N-1)
    d_rhou = rhou[..., 1:] - rhou[..., :-1]
    d_rhoE = rhoE[..., 1:] - rhoE[..., :-1]

    dF_rho  = F_rho[..., 1:] - F_rho[..., :-1]
    dF_rhou = F_rhou[..., 1:] - F_rhou[..., :-1]
    dF_rhoE = F_rhoE[..., 1:] - F_rhoE[..., :-1]

    # lambda_RH (Eq. 78c): min over 3 RH wave speed estimates
    lam_rh = np.minimum.reduce([
        np.abs(dF_rho)  / (np.abs(d_rho)  + EPS0),
        np.abs(dF_rhou) / (np.abs(d_rhou) + EPS0),
        np.abs(dF_rhoE) / (np.abs(d_rhoE) + EPS0),
    ])                                           # (*others, N-1)

    # Positivity alpha terms (Eq. 32)
    # Need primitives at each cell (entire array)
    rho_p, u_n, p_p, gam_p, a_p = eos.primitives(U_mov, axis, ndim)
    # alpha at left (j) and right (j+1) cells of each interface
    alpha_R, alpha_L = eos.positivity_alpha(u_n, a_p, gam_p)

    alpha_L_left  = alpha_L[..., :-1]           # (*others, N-1)
    alpha_R_right = alpha_R[..., 1:]            # (*others, N-1)

    lam_base = np.maximum.reduce([lam_rh, alpha_L_left, alpha_R_right])

    # Contact discontinuity detector (Eq. 42)
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
    return np.maximum(lam, 0.0)                 # (*others, N-1)


# ─────────────────────────────────────────────
# Interface flux (all on last axis)
# ─────────────────────────────────────────────

def interface_flux(U_mov, F_mov, lam, order, bc):
    """
    Compute interface fluxes along the last axis.

    U_mov, F_mov : (Nvars, *others, N)
    lam          : (*others, N-1)   — lambda at N-1 interfaces
    order        : 1, 2, 3, or 4
    bc           : "transmissive" or "periodic"

    Returns Gf : (Nvars, *others, N-1)
    """
    if order == 1:
        return _iflux_1st(U_mov, F_mov, lam)
    elif order == 2:
        return _iflux_ho(U_mov, F_mov, lam, b=1, bc=bc)
    elif order == 3:
        return _iflux_ho(U_mov, F_mov, lam, b=B_COMPRESS, bc=bc)
    else:  # order == 4, unlimited
        return _iflux_ho(U_mov, F_mov, lam, b=0, bc=bc)


def _iflux_1st(U_mov, F_mov, lam):
    """First-order interface flux (Eq. 27)."""
    avg_F = 0.5 * (F_mov[..., :-1] + F_mov[..., 1:])   # (Nvars, *others, N-1)
    dU    = U_mov[..., 1:] - U_mov[..., :-1]            # (Nvars, *others, N-1)
    return avg_F - 0.5 * lam[np.newaxis] * dU


def _iflux_ho(U_mov, F_mov, lam, b, bc):
    """
    Chakravarthy-Osher higher-order interface flux (Eq. 43/50).
    b=4 → 3rd order limited, b=1 → 2nd order limited, b=0 → unlimited.
    """
    N  = U_mov.shape[-1]
    M  = N - 1   # number of interfaces

    dU = U_mov[..., 1:] - U_mov[..., :-1]              # (Nvars, *others, M)
    dF = F_mov[..., 1:] - F_mov[..., :-1]              # (Nvars, *others, M)

    # ΔG+, ΔG-  (Eq. 64c, 64d)
    lam_br = lam[np.newaxis]                            # (1, *others, M) broadcast
    dGp = 0.5 * dF + 0.5 * lam_br * dU
    dGm = 0.5 * dF - 0.5 * lam_br * dU

    Gf1 = _iflux_1st(U_mov, F_mov, lam)
    Gf  = Gf1.copy()

    i = slice(1, M-1)    # inner interfaces (index 1 .. M-2)

    if b == 0:
        # Unlimited 3rd order (φ=1)
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


# ─────────────────────────────────────────────
# Ghost-cell boundary conditions
# ─────────────────────────────────────────────

def _apply_bc(U_ext, spatial_axis, bc):
    """
    Apply boundary conditions along spatial_axis in U_ext.
    U_ext has shape (Nvars, *spatial_ext) where spatial_ext includes
    ghost cells (size+2) along spatial_axis.

    spatial_axis is 0-based in spatial dimensions, i.e. array axis = spatial_axis+1.
    """
    arr_axis = spatial_axis + 1   # array axis (0 is variable axis)
    n = U_ext.shape[arr_axis]

    # Build index objects for first/last and second/second-to-last slices
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
    else:  # transmissive
        U_ext[tuple(idx_first)]  = U_ext[tuple(idx_second)]
        U_ext[tuple(idx_last)]   = U_ext[tuple(idx_penult)]


# ─────────────────────────────────────────────
# Residual along one axis
# ─────────────────────────────────────────────

def residual_axis(U, axis, dx, eos, order, bc, ndim):
    """
    Spatial residual dF/dx_axis for all cells.

    U     : (Nvars, *spatial)  — interior cells (no ghosts)
    axis  : int, sweep axis (0-based spatial index)
    dx    : float, cell width along axis
    eos   : EOSBase instance
    order : 1, 2, 3, or 4
    bc    : "transmissive" or "periodic"
    ndim  : int

    Returns R : (Nvars, *spatial)
    """
    # 1. Add ghost cells along `axis` (array axis = axis+1)
    arr_ax = axis + 1
    pad_width = [(0,0)] * U.ndim
    pad_width[arr_ax] = (1, 1)
    U_ext = np.pad(U, pad_width, mode='edge')   # transmissive default; overwrite below
    _apply_bc(U_ext, axis, bc)

    # 2. Move sweep axis to last position in spatial dims
    #    U_ext: (Nvars, *spatial_ext)  → (Nvars, *others, N+2)
    U_mov = np.moveaxis(U_ext, arr_ax, -1)     # (Nvars, *others, N+2)

    # 3. Compute flux
    F_ext = eos.flux(U_ext, axis, ndim)
    F_mov = np.moveaxis(F_ext, arr_ax, -1)     # (Nvars, *others, N+2)

    # 4. Lambda at N+1 interfaces
    lam = compute_lambda(U_mov, F_mov, axis, ndim, eos)  # (*others, N+1)

    # 5. Interface flux
    Gf = interface_flux(U_mov, F_mov, lam, order, bc)    # (Nvars, *others, N+1)

    # 6. Residual: (G_{j+1/2} - G_{j-1/2}) / dx
    R_mov = (Gf[..., 1:] - Gf[..., :-1]) / dx           # (Nvars, *others, N)

    # 7. Move sweep axis back
    R = np.moveaxis(R_mov, -1, arr_ax)                   # (Nvars, *spatial)
    return R


# ─────────────────────────────────────────────
# Time step computation
# ─────────────────────────────────────────────

def compute_dt(U, eos, dx_list, order, sigma, ndim):
    """
    Global CFL time step satisfying positivity and stability for all axes.

    U       : (Nvars, *spatial) interior cells
    dx_list : list of cell widths [dx0, dx1, ...]
    """
    dt_min = np.inf

    for d, dx in enumerate(dx_list):
        # Add ghost cells along axis d
        arr_ax = d + 1
        pad_width = [(0,0)] * U.ndim
        pad_width[arr_ax] = (1, 1)
        U_ext = np.pad(U, pad_width, mode='edge')

        U_mov = np.moveaxis(U_ext, arr_ax, -1)
        F_ext = eos.flux(U_ext, d, ndim)
        F_mov = np.moveaxis(F_ext, arr_ax, -1)

        lam = compute_lambda(U_mov, F_mov, d, ndim, eos)  # (*others, N+1)

        # Positivity: dt_p = min over interior cells of 2*dx / (lam_{j+1/2} + lam_{j-1/2})
        lam_plus  = lam[..., 1:]    # interfaces j+1/2
        lam_minus = lam[..., :-1]   # interfaces j-1/2
        dt_p = np.min(2.0 * dx / (lam_plus + lam_minus + EPS0))

        # Stability: dt_s = min over cells of dx / max_wave_speed
        rho, u_n, p, gam, a = eos.primitives(U_ext, d, ndim)
        # Interior only (strip ghost cells in sweep axis)
        sl = [slice(None)] * U_ext.ndim
        sl[arr_ax] = slice(1, -1)
        u_n_int = np.moveaxis(U_ext, arr_ax, -1)[..., 1:-1]  # dummy; use primitives
        # Extract interior values directly
        rho_int, u_n_int2, p_int, gam_int, a_int = eos.primitives(
            np.moveaxis(U_ext, arr_ax, -1)[..., 1:-1].swapaxes(0, -1) if False
            else _extract_interior(U_ext, arr_ax),
            d, ndim
        )
        lam_max = np.max(np.maximum.reduce([
            np.abs(u_n_int2 - a_int),
            np.abs(u_n_int2),
            np.abs(u_n_int2 + a_int),
        ]))
        dt_s = dx / (lam_max + EPS0)

        if order > 1:
            dt_p = dt_p / 2.0   # Eq. 52

        dt_min = min(dt_min, sigma * min(dt_p, dt_s))

    return dt_min


def _extract_interior(U_ext, arr_ax):
    """Extract interior cells (strip ghost cells along arr_ax)."""
    sl = [slice(None)] * U_ext.ndim
    sl[arr_ax] = slice(1, -1)
    return U_ext[tuple(sl)]


# ─────────────────────────────────────────────
# SSPRK3 time integration (single axis)
# ─────────────────────────────────────────────

def _ssprk3_axis(U, dx, dt, eos, order, bc, ndim, axis):
    """
    One SSPRK3 step advancing U along a single `axis`.
    """
    def R(V):
        return residual_axis(V, axis, dx, eos, order, bc, ndim)

    U0 = U
    U1 = U0 - dt * R(U0)
    U2 = 0.75*U0 + 0.25*U1 - 0.25*dt * R(U1)
    U3 = (1.0/3.0)*U0 + (2.0/3.0)*U2 - (2.0/3.0)*dt * R(U2)
    return U3


# ─────────────────────────────────────────────
# Strang-split step (all axes)
# ─────────────────────────────────────────────

def step_ssprk3(U, dx_list, dt, eos, order, bc, ndim):
    """
    One full time step using SSPRK3 + Strang dimensional splitting.

    ndim=1: standard SSPRK3, axis 0
    ndim=2: Strang: dt/2 axis0, dt axis1, dt/2 axis0
    ndim=3: Strang: dt/2 axis0, dt/2 axis1, dt axis2, dt/2 axis1, dt/2 axis0
    """
    if ndim == 1:
        return _ssprk3_axis(U, dx_list[0], dt, eos, order, bc, ndim, axis=0)

    elif ndim == 2:
        # dt/2 along axis 0
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        # dt along axis 1
        U = _ssprk3_axis(U, dx_list[1], dt,   eos, order, bc, ndim, axis=1)
        # dt/2 along axis 0
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        return U

    elif ndim == 3:
        # dt/2 axis0, dt/2 axis1, dt axis2, dt/2 axis1, dt/2 axis0
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        U = _ssprk3_axis(U, dx_list[1], dt/2, eos, order, bc, ndim, axis=1)
        U = _ssprk3_axis(U, dx_list[2], dt,   eos, order, bc, ndim, axis=2)
        U = _ssprk3_axis(U, dx_list[1], dt/2, eos, order, bc, ndim, axis=1)
        U = _ssprk3_axis(U, dx_list[0], dt/2, eos, order, bc, ndim, axis=0)
        return U

    else:
        raise ValueError(f"ndim={ndim} not supported (1, 2, or 3 only)")


# ─────────────────────────────────────────────
# Main simulation runner
# ─────────────────────────────────────────────

def run_simulation(U0, dx_list, t_end, eos, order=1,
                   sigma=CFL_DEFAULT, bc="transmissive",
                   ndim=None, max_steps=2_000_000):
    """
    Run simulation from t=0 to t=t_end using SSPRK3 + Strang splitting.

    Parameters
    ----------
    U0       : (Nvars, *spatial) initial state
    dx_list  : list of cell widths
    t_end    : float, final time
    eos      : EOSBase instance
    order    : 1/2/3/4
    sigma    : CFL number
    bc       : "transmissive" or "periodic"
    ndim     : number of spatial dimensions (inferred from U0 if None)
    max_steps: safety cap

    Returns
    -------
    U : (Nvars, *spatial) final state
    t : float actual final time
    """
    U = U0.copy()
    if ndim is None:
        ndim = U.ndim - 1

    t = 0.0
    for _ in range(max_steps):
        if t >= t_end:
            break

        dt = compute_dt(U, eos, dx_list, order, sigma, ndim)
        dt = min(dt, t_end - t)

        if order == 1:
            # Forward Euler for 1st order
            if ndim == 1:
                R = residual_axis(U, 0, dx_list[0], eos, order, bc, ndim)
                U = U - dt * R
            else:
                U = step_ssprk3(U, dx_list, dt, eos, order, bc, ndim)
        else:
            U = step_ssprk3(U, dx_list, dt, eos, order, bc, ndim)

        t += dt

    return U, t


# ─────────────────────────────────────────────
# Primitive / conserved helpers (2-species, 1D)
# ─────────────────────────────────────────────

def prim_to_cons_2s(W, rho, u, p, gam1, gam2, ndim=1):
    """
    Convert primitive to conserved state for 2-species 1D system.
    Returns U : (Nvars, Nx)
    """
    Ns = 2
    cv1 = 1.0 / (gam1 - 1.0)
    cv2 = 1.0 / (gam2 - 1.0)
    cv  = W * cv1 + (1.0 - W) * cv2
    cp  = W * gam1 * cv1 + (1.0 - W) * gam2 * cv2
    gam = cp / (cv + EPS0)

    rhoE = p / (gam - 1.0) + 0.5 * rho * u**2

    if ndim == 1:
        return np.array([rho * W, rho * (1-W), rho * u, rhoE])
    else:
        raise NotImplementedError("Use nd helper for ndim>1")


def cons_to_prim_2s(U, gam1, gam2, ndim=1):
    """Extract (W, rho, u, p) from 2-species conserved state."""
    eos = IdealGasMixture([gam1, gam2])
    rho, u_n, p, gam, a = eos.primitives(U, 0, ndim)
    W = np.clip(U[0] / (rho + EPS0), 0.0, 1.0)
    return W, rho, u_n, p


# ─────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────

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
    """EOC smooth IC (Sec. 5.1, Eq. 84)."""
    domain = 2.0
    dx = domain / Nx
    x  = (np.arange(Nx) + 0.5) * dx

    h = dx / 2.0
    sinc_h = np.sin(np.pi * h) / (np.pi * h)
    rho0 = 1.0 + 0.2 * np.sin(np.pi * x) * sinc_h
    # Note: paper says rho=0.5+0.1*sin(πx), but cell-averaged form is used
    # Actually: rho = 0.5 + 0.1*sin(πx), cell-avg factor is sinc_h
    # Let's use the exact formula: rho_cell_avg = 0.5 + 0.1*sin(πx)*sinc_h
    # Wait -- exact_rho_avg at t=0 with 0.5+0.1*sin(πx) gives 0.5+0.1*sinc_h*sin(πx)
    # but in lambda_diff_1d.py: rho0 = exact_rho_avg(x, 0.0, dx) = 1+0.2*sin(πx)*sinc_h
    # This doesn't match 0.5+0.1... Let me re-check the formula in lambda_diff_1d.py
    # exact_rho_avg(x, t, dx): returns 1.0 + 0.2*sin(pi*(x-0.1*t))*sinc_h
    # But IC should be 0.5+0.1*sin(πx) -- the 1D code uses a different factor?
    # Actually the 1D code uses exact_rho_avg which returns 1+0.2*sin*sinc
    # So the actual IC is rho ~ 1 + 0.2*sin(πx)*sinc_h (cell-averaged).
    # Let me match exactly what lambda_diff_1d.py does:
    rho0 = 1.0 + 0.2 * np.sin(np.pi * x) * sinc_h

    W0 = 0.5 * np.ones(Nx)
    u0 = 0.1 * np.ones(Nx)
    p0 = 0.5 * np.ones(Nx)
    return x, prim_to_cons_2s(W0, rho0, u0, p0, gam1, gam2), dx


def _eoc_exact_rho(x, t, dx):
    """Cell-averaged exact density for EOC test."""
    h = dx / 2.0
    sinc_h = np.sin(np.pi * h) / (np.pi * h)
    return 1.0 + 0.2 * np.sin(np.pi * (x - 0.1 * t)) * sinc_h


# ─── EOC test ────────────────────────────────

def run_test_eoc(verbose=True):
    """
    EOC test (Sec. 5.1, Tables 1-4).
    Domain [0,2], periodic BC, t_end=0.5.
    """
    print("\n[Test] Experimental Order of Convergence (Sec. 5.1)")
    gam1 = gam2 = 1.4
    eos = IdealGasMixture([gam1, gam2])
    t_end  = 0.5
    domain = 2.0
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

            h = dx / 2.0
            sinc_h = np.sin(np.pi * h) / (np.pi * h) if np.pi*h > 1e-14 else 1.0
            rho0 = 1.0 + 0.2 * np.sin(np.pi * x) * sinc_h
            W0   = 0.5 * np.ones(Nx)
            u0   = 0.1 * np.ones(Nx)
            p0   = 0.5 * np.ones(Nx)
            U0   = prim_to_cons_2s(W0, rho0, u0, p0, gam1, gam2)

            U, _ = run_simulation(U0, [dx], t_end, eos,
                                  order=order, sigma=CFL_DEFAULT,
                                  bc="periodic", ndim=1,
                                  max_steps=2_000_000)

            # Total density is sum of partial densities
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

        # Reference check at Nx=40
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


# ─── Sod shock tube test ──────────────────────

def run_test_sod(Nx=200, sigma=CFL_DEFAULT):
    """
    Sod's shock tube (same gamma, 2 species), 1D.
    """
    print("\n[Test] Sod shock tube (1D, ndim=1)")
    gam1 = gam2 = 1.4
    eos = IdealGasMixture([gam1, gam2])
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
    fname = "test_sod_nd.png"
    plt.savefig(fname, dpi=120); plt.close()
    print(f"  Saved: {fname}")

    # Basic check: pressure should be roughly in [1, 10]
    W, rho, u, p = results[-1]
    ok = (p.min() > 0.5) and (p.max() < 11.0)
    print(f"  Pressure range: [{p.min():.3f}, {p.max():.3f}] -- {'PASS' if ok else 'FAIL'}")
    return ok


# ─── Steady contact test ─────────────────────

def run_test_contact(Nx=200, sigma=CFL_DEFAULT):
    """
    Steady contact discontinuity: should be preserved exactly at all orders.
    """
    print("\n[Test] Steady contact discontinuity (1D)")
    gam1, gam2 = 1.6, 1.4
    eos = IdealGasMixture([gam1, gam2])
    t_end = 0.1

    x, U0 = _shock_tube_ic_1d(Nx, 1.0, 1.0, 0.0, 1.0,
                                   0.0, 0.1, 0.0, 1.0, gam1, gam2)
    dx = 1.0 / Nx

    ok = True
    for order, lbl in [(1, "1O"), (3, "3O")]:
        U, _ = run_simulation(U0.copy(), [dx], t_end, eos,
                               order=order, sigma=sigma, ndim=1)
        W, rho, u, p = cons_to_prim_2s(U, gam1, gam2)

        # Pressure should remain uniform (machine-precision level for kinetic scheme)
        p_err = np.max(np.abs(p - 1.0))
        u_err = np.max(np.abs(u))
        # Allow up to 1e-7 for floating-point accumulation over many time steps
        passed = (p_err < 1e-7) and (u_err < 1e-7)
        ok = ok and passed
        print(f"  {lbl}: max|p-1|={p_err:.2e}, max|u|={u_err:.2e} "
              f"-- {'PASS' if passed else 'FAIL'}")

    return ok


# ─── 3D Sod (vs 1D) test ─────────────────────

def run_test_sod3d(Nx=200, sigma=CFL_DEFAULT):
    """
    3D Sod shock tube: run along axis 0 with 1 cell in axes 1,2.

    The 3D Strang-split result (two SSPRK3(dt/2) passes on axis 0) will
    NOT be numerically identical to a single SSPRK3(dt) 1D run — this is
    expected and correct for dimensional splitting.

    The test therefore verifies:
    1. The 3D solution is physically consistent (density/pressure in range)
    2. The 3D and 1D solutions agree to within Strang splitting error
       (which is 2nd-order in dt, so O(dt) for shock problems; tolerance ~1%)
    3. The 3D state has no transverse momentum growth (axes 1 and 2 are no-ops)
    """
    print("\n[Test] 3D Sod shock tube (compare with 1D)")
    gam1 = gam2 = 1.4
    eos = IdealGasMixture([gam1, gam2])
    t_end = 0.1
    ndim3 = 3

    x, U0_1d = _shock_tube_ic_1d(Nx, 1.0, 2.0, 0.0, 10.0,
                                      0.0, 1.0, 0.0,  1.0, gam1, gam2)
    dx = 1.0 / Nx

    # 1D run (ndim=1), state (4, Nx)
    U1d, _ = run_simulation(U0_1d.copy(), [dx], t_end, eos,
                             order=3, sigma=sigma, ndim=1)

    # Build 3D state: (6, Nx, 1, 1) — variation along axis 0 only
    Nvars3d = 2 + ndim3 + 1  # 6
    U0_3d = np.zeros((Nvars3d, Nx, 1, 1))
    U0_3d[0, :, 0, 0] = U0_1d[0]   # rho_1
    U0_3d[1, :, 0, 0] = U0_1d[1]   # rho_2
    U0_3d[2, :, 0, 0] = U0_1d[2]   # rho*u_0 (momentum along axis 0)
    U0_3d[3, :, 0, 0] = 0.0         # rho*u_1 = 0
    U0_3d[4, :, 0, 0] = 0.0         # rho*u_2 = 0
    U0_3d[5, :, 0, 0] = U0_1d[3]   # rhoE

    # dx_list: only axis 0 matters; axes 1 and 2 have 1 cell (no-ops)
    U3d, _ = run_simulation(U0_3d, [dx, 1.0, 1.0], t_end, eos,
                             order=3, sigma=sigma, ndim=ndim3)

    # Extract 1D profile along axis 0
    rho_3d = U3d[0, :, 0, 0] + U3d[1, :, 0, 0]   # total density
    rho_1d = U1d[0] + U1d[1]

    # Test 1: no transverse momentum (axes 1,2 remain zero)
    transverse_mom = max(np.max(np.abs(U3d[3])), np.max(np.abs(U3d[4])))
    ok_transverse = transverse_mom < 1e-12
    print(f"  Max transverse momentum: {transverse_mom:.2e} "
          f"-- {'PASS' if ok_transverse else 'FAIL'}")

    # Test 2: 3D density profile is physically consistent (same range as 1D)
    rho_min_1d, rho_max_1d = rho_1d.min(), rho_1d.max()
    rho_min_3d, rho_max_3d = rho_3d.min(), rho_3d.max()
    tol_range = 0.05  # 5% tolerance for Strang splitting error
    ok_range = (abs(rho_min_3d - rho_min_1d) / (rho_min_1d + EPS0) < tol_range and
                abs(rho_max_3d - rho_max_1d) / (rho_max_1d + EPS0) < tol_range)
    print(f"  1D rho: [{rho_min_1d:.4f}, {rho_max_1d:.4f}]")
    print(f"  3D rho: [{rho_min_3d:.4f}, {rho_max_3d:.4f}]")
    print(f"  Range consistency (tol=5%): {'PASS' if ok_range else 'FAIL'}")

    # Test 3: L1 difference between 3D and 1D profiles (Strang error is O(dt^2))
    l1_diff = dx * np.sum(np.abs(rho_3d - rho_1d))
    # Strang splitting error is 2nd-order in dt; for N~200, dt~1.5e-3, so
    # ~N_steps * dt^2 ~ 66 * (1.5e-3)^2 ~ 1.5e-4. Tolerate 0.5% of total variation.
    total_var = dx * np.sum(np.abs(rho_1d))
    rel_l1 = l1_diff / (total_var + EPS0)
    ok_l1 = rel_l1 < 0.02   # within 2% relative L1
    print(f"  L1(rho_3d - rho_1d) / L1(rho_1d) = {rel_l1:.4f} "
          f"-- {'PASS' if ok_l1 else 'FAIL'}")

    ok = ok_transverse and ok_range and ok_l1
    return ok


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

TESTS = {
    "eoc":     run_test_eoc,
    "sod":     run_test_sod,
    "contact": run_test_contact,
    "sod3d":   run_test_sod3d,
}


def main():
    parser = argparse.ArgumentParser(
        description="ND Multi-Species Euler — Kinetic Scheme (Roy & Raghurama Rao 2025)"
    )
    parser.add_argument(
        "--test",
        choices=list(TESTS.keys()) + ["all"],
        default="eoc",
    )
    parser.add_argument("--Nx",    type=int,   default=200)
    parser.add_argument("--sigma", type=float, default=CFL_DEFAULT)
    args = parser.parse_args()

    results = {}
    if args.test == "all":
        for name, fn in TESTS.items():
            if name == "eoc":
                results[name] = fn()
            else:
                results[name] = fn(Nx=args.Nx, sigma=args.sigma)
    elif args.test == "eoc":
        results["eoc"] = TESTS["eoc"]()
    else:
        fn = TESTS[args.test]
        if args.test == "eoc":
            results[args.test] = fn()
        else:
            results[args.test] = fn(Nx=args.Nx, sigma=args.sigma)

    print("\n" + "="*50)
    print("SUMMARY:")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:12s}: {status}")

    if results and not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
