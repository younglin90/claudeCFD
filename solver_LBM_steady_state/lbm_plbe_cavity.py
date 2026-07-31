"""Guo-Zhao-Shi preconditioned LBE cavity model.

This implements the PLBE model from Phys. Rev. E 70, 066706 (2004):

    f_i^eq = w_i rho [1 + 3 c_i.u
        + (9/(2 gamma)) (c_i.u)^2 - (3/(2 gamma)) u.u]

with viscosity relation nu = gamma * cs^2 * (tau - 1/2).

The cavity boundary uses nonequilibrium extrapolation, matching the paper more
closely than the bounce-back cavity case in ``lbm_core.py``.
"""

from __future__ import annotations

import numpy as np

from lbm_core import CX, CY, CX_INT, CY_INT, W


def equilibrium_plbe(rho, ux, uy, gamma):
    feq = np.empty((9,) + rho.shape, dtype=np.float64)
    u2 = ux * ux + uy * uy
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu / gamma - 1.5 * u2 / gamma)
    return feq


def moments(f):
    rho = f.sum(axis=0)
    rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
    uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
    return rho, ux, uy


def stream(f):
    fnew = np.empty_like(f)
    for i in range(9):
        fnew[i] = np.roll(np.roll(f[i], CY_INT[i], axis=0), CX_INT[i], axis=1)
    return fnew


def _apply_wall_from_neighbor(f, gamma, side, u_wall):
    rho, ux, uy = moments(f)
    if side == "left":
        b = (slice(None), slice(None), 0)
        n = (slice(None), slice(None), 1)
        unknown = (1, 5, 8)
        rho_b = rho[:, 1]
        ux_b = np.full_like(rho_b, u_wall[0])
        uy_b = np.full_like(rho_b, u_wall[1])
        rho_n, ux_n, uy_n = rho[:, 1], ux[:, 1], uy[:, 1]
    elif side == "right":
        b = (slice(None), slice(None), -1)
        n = (slice(None), slice(None), -2)
        unknown = (3, 6, 7)
        rho_b = rho[:, -2]
        ux_b = np.full_like(rho_b, u_wall[0])
        uy_b = np.full_like(rho_b, u_wall[1])
        rho_n, ux_n, uy_n = rho[:, -2], ux[:, -2], uy[:, -2]
    elif side == "bottom":
        b = (slice(None), 0, slice(None))
        n = (slice(None), 1, slice(None))
        unknown = (2, 5, 6)
        rho_b = rho[1, :]
        ux_b = np.full_like(rho_b, u_wall[0])
        uy_b = np.full_like(rho_b, u_wall[1])
        rho_n, ux_n, uy_n = rho[1, :], ux[1, :], uy[1, :]
    elif side == "top":
        b = (slice(None), -1, slice(None))
        n = (slice(None), -2, slice(None))
        unknown = (4, 7, 8)
        rho_b = rho[-2, :]
        ux_b = np.full_like(rho_b, u_wall[0])
        uy_b = np.full_like(rho_b, u_wall[1])
        rho_n, ux_n, uy_n = rho[-2, :], ux[-2, :], uy[-2, :]
    else:
        raise ValueError(side)

    feq_b = equilibrium_plbe(rho_b, ux_b, uy_b, gamma)
    feq_n = equilibrium_plbe(rho_n, ux_n, uy_n, gamma)
    for i in unknown:
        if side == "left":
            f[i, :, 0] = feq_b[i] + (f[i, :, 1] - feq_n[i])
        elif side == "right":
            f[i, :, -1] = feq_b[i] + (f[i, :, -2] - feq_n[i])
        elif side == "bottom":
            f[i, 0, :] = feq_b[i] + (f[i, 1, :] - feq_n[i])
        else:
            f[i, -1, :] = feq_b[i] + (f[i, -2, :] - feq_n[i])


def apply_neq_extrapolation_bc(f, gamma, u_wall):
    # Apply fixed walls first, moving lid last so the lid owns the top corners.
    _apply_wall_from_neighbor(f, gamma, "left", (0.0, 0.0))
    _apply_wall_from_neighbor(f, gamma, "right", (0.0, 0.0))
    _apply_wall_from_neighbor(f, gamma, "bottom", (0.0, 0.0))
    _apply_wall_from_neighbor(f, gamma, "top", (u_wall, 0.0))
    return f


class PLBECavity:
    """Preconditioned LBE lid-driven cavity with NEQ extrapolation walls."""

    def __init__(self, N, Re, U_wall=0.1, gamma=0.25):
        self.N = N
        self.Re = Re
        self.U_wall = U_wall
        self.gamma = gamma
        self.nu = U_wall * (N - 1) / Re
        self.tau = 0.5 + 3.0 * self.nu / gamma
        self.omega = 1.0 / self.tau
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N

    def initial_field(self):
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.zeros((self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        ux[-1, :] = self.U_wall
        f = equilibrium_plbe(rho, ux, uy, self.gamma)
        return apply_neq_extrapolation_bc(f, self.gamma, self.U_wall)

    def lbe_step(self, f):
        rho, ux, uy = moments(f)
        feq = equilibrium_plbe(rho, ux, uy, self.gamma)
        fstar = f - self.omega * (f - feq)
        fnew = stream(fstar)
        return apply_neq_extrapolation_bc(fnew, self.gamma, self.U_wall)

    def residual(self, f):
        return f - self.lbe_step(f)

    def macro(self, f):
        return moments(f)

    def project(self, f):
        rho = f.sum(axis=0)
        rhoux = (f * CX[:, None, None]).sum(axis=0)
        rhouy = (f * CY[:, None, None]).sum(axis=0)
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    def _fast_norm(self, x):
        xr = x.ravel()
        return float(np.sqrt(xr @ xr))

    def _fd_eps(self, norm_f_cached, w):
        norm_w = self._fast_norm(w)
        if norm_w < 1e-30:
            return 1e-8
        return 1e-7 * (norm_f_cached + 1.0) / norm_w

    def jvp(self, w, f_base, r_base, norm_f_cached=None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        eps = self._fd_eps(norm_f_cached, w)
        return (self.residual(f_base + eps * w) - r_base) / eps
