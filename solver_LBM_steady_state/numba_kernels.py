"""Numba-parallel njit kernels for every Case class.

call ``enable_numba_kernels()`` once at startup to monkey-patch all 6 Case
classes (LBMCavity, ChannelCase, CouetteCase, KolmogorovCase, VoxelCase,
PLBECavity) so their ``lbe_step`` runs as parallel njit code instead of pure
NumPy. This makes wall-time comparisons across cases fair and gives every
solver wrapper the same kinetic kernel cost.

Algorithm semantics are byte-for-byte equivalent to the original ``.lbe_step``
methods up to floating-point reduction order.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array(
    [4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0],
    dtype=np.float64,
)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


# ----------------------------------------------------------------------
# Standard BGK equilibrium and Guo forcing
# ----------------------------------------------------------------------
@njit(cache=True, inline="always")
def _feq_i(i, rho, ux, uy):
    cu = 3.0 * (CX[i] * ux + CY[i] * uy)
    u2 = 1.5 * (ux * ux + uy * uy)
    return W[i] * rho * (1.0 + cu + 0.5 * cu * cu - u2)


@njit(cache=True, inline="always")
def _guo_S(i, rho, ux, uy, Fx, Fy, omega):
    coeff = 1.0 - 0.5 * omega
    cu = CX[i] * ux + CY[i] * uy
    e_dot_F = CX[i] * Fx + CY[i] * Fy
    eu_dot_F = (CX[i] - ux) * Fx + (CY[i] - uy) * Fy
    return coeff * W[i] * (3.0 * e_dot_F + 9.0 * cu * e_dot_F - 3.0 * (ux * Fx + uy * Fy)) * 0.0 + coeff * W[i] * (
        3.0 * eu_dot_F + 9.0 * cu * e_dot_F
    )


@njit(cache=True)
def _moments(f, rho, ux, uy):
    ny, nx = rho.shape
    for y in prange(ny):
        for x in range(nx):
            r = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[i, y, x]
                r += fi
                mx += CX[i] * fi
                my += CY[i] * fi
            rho[y, x] = r
            if r > 1e-12:
                ux[y, x] = mx / r
                uy[y, x] = my / r
            else:
                ux[y, x] = 0.0
                uy[y, x] = 0.0


# ----------------------------------------------------------------------
# Kolmogorov (fully periodic + Guo force)
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _kolmo_step(f, out, Fx, Fy, omega):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _moments(f, rho, ux, uy)
    # collide with Guo velocity shift, then store collided in tmp `out`
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            uxs = ux[y, x] + 0.5 * Fx[y, x] / r
            uys = uy[y, x] + 0.5 * Fy[y, x] / r
            for i in range(9):
                feq = _feq_i(i, r, uxs, uys)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * Fx[y, x] + CY[i] * Fy[y, x]
                eu_F = (CX[i] - ux[y, x]) * Fx[y, x] + (CY[i] - uy[y, x]) * Fy[y, x]
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    # periodic stream from out -> f (reuse f as scratch input via copy)
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny
                xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]


def kolmogorov_step(self, f):
    out = np.empty_like(f)
    _kolmo_step(f, out, self.Fx, self.Fy, self.omega)
    return out


# ----------------------------------------------------------------------
# Channel (periodic-x, bb walls at j=0 and j=N-1)
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _channel_step(f, out, Fx, Fy, omega):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            uxs = ux[y, x] + 0.5 * Fx[y, x] / r
            uys = uy[y, x] + 0.5 * Fy[y, x] / r
            for i in range(9):
                feq = _feq_i(i, r, uxs, uys)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * Fx[y, x] + CY[i] * Fy[y, x]
                eu_F = (CX[i] - ux[y, x]) * Fx[y, x] + (CY[i] - uy[y, x]) * Fy[y, x]
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    f_in = out.copy()
    # stream periodic-y and periodic-x then apply bb on top/bottom
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny
                xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]
    # bottom j=0: incoming 2,5,6 from bb of 4,7,8 at same cell
    for x in prange(nx):
        out[2, 0, x] = out[4, 0, x]
        out[5, 0, x] = out[7, 0, x]
        out[6, 0, x] = out[8, 0, x]
    for x in prange(nx):
        out[4, ny - 1, x] = out[2, ny - 1, x]
        out[7, ny - 1, x] = out[5, ny - 1, x]
        out[8, ny - 1, x] = out[6, ny - 1, x]


def channel_step(self, f):
    out = np.empty_like(f)
    _channel_step(f, out, self.Fx, self.Fy, self.omega)
    return out


# ----------------------------------------------------------------------
# Couette (periodic-x, bb bottom, moving lid top)
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _couette_step(f, out, omega, U_wall):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            for i in range(9):
                feq = _feq_i(i, r, ux[y, x], uy[y, x])
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq)
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny
                xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]
    for x in prange(nx):
        out[2, 0, x] = out[4, 0, x]
        out[5, 0, x] = out[7, 0, x]
        out[6, 0, x] = out[8, 0, x]
        # top moving lid - ladd
        rho_top = (
            out[0, ny - 1, x] + out[1, ny - 1, x] + out[3, ny - 1, x]
            + 2.0 * (out[2, ny - 1, x] + out[5, ny - 1, x] + out[6, ny - 1, x])
        )
        out[4, ny - 1, x] = out[2, ny - 1, x]
        out[7, ny - 1, x] = out[5, ny - 1, x] - 6.0 * W[5] * rho_top * U_wall
        out[8, ny - 1, x] = out[6, ny - 1, x] + 6.0 * W[6] * rho_top * U_wall


def couette_step(self, f):
    out = np.empty_like(f)
    _couette_step(f, out, self.omega, self.U_wall)
    return out


# ----------------------------------------------------------------------
# LBMCavity (all 4 walls bb, top moving lid)
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _cavity_step(f, out, omega, U_wall):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            for i in range(9):
                feq = _feq_i(i, r, ux[y, x], uy[y, x])
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq)
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            rho_wall = 0.0
            for i in range(9):
                rho_wall += f_in[i, y, x]
            for i in range(9):
                ys = (y - CY[i]) % ny
                xs = (x - CX[i]) % nx
                if 0 <= y - CY[i] < ny and 0 <= x - CX[i] < nx:
                    out[i, y, x] = f_in[i, ys, xs]
                else:
                    val = f_in[OPP[i], y, x]
                    if y == ny - 1 and y - CY[i] >= ny:
                        if i == 7:
                            val -= 6.0 * W[7] * rho_wall * U_wall
                        elif i == 8:
                            val += 6.0 * W[8] * rho_wall * U_wall
                    out[i, y, x] = val


def cavity_step(self, f):
    out = np.empty_like(f)
    _cavity_step(f, out, self.omega, self.U_wall)
    return out


# ----------------------------------------------------------------------
# Voxel mask (Guo forcing + half-way bb at fluid-solid links)
# ----------------------------------------------------------------------
@njit(cache=True, parallel=True)
def _voxel_step(f, out, chi, omega, Fx, Fy):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _moments(f, rho, ux, uy)
    fstar = np.empty_like(f)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            if r < 1e-12:
                r = 1.0
            uxs = ux[y, x] + 0.5 * Fx[y, x] / r
            uys = uy[y, x] + 0.5 * Fy[y, x] / r
            for i in range(9):
                feq = _feq_i(i, r, uxs, uys)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * Fx[y, x] + CY[i] * Fy[y, x]
                eu_F = (CX[i] - ux[y, x]) * Fx[y, x] + (CY[i] - uy[y, x]) * Fy[y, x]
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                fstar[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    for y in prange(ny):
        for x in range(nx):
            ci = (chi[y, x] == 1.0)
            for i in range(9):
                ys = (y - CY[i]) % ny
                xs = (x - CX[i]) % nx
                src_fluid = ci and (chi[ys, xs] == 1.0)
                if src_fluid:
                    out[i, y, x] = fstar[i, ys, xs] * chi[y, x]
                else:
                    out[i, y, x] = fstar[OPP[i], y, x] * chi[y, x]


def voxel_step(self, f):
    out = np.empty_like(f)
    _voxel_step(f, out, self.chi, self.omega, self.Fx, self.Fy)
    return out


# ----------------------------------------------------------------------
# PLBE (gamma equilibrium + NEQ-extrapolation walls + lid)
#   Re-export the existing njit step from verify_plbe_re1000.
# ----------------------------------------------------------------------
def plbe_step(self, f):
    from verify_plbe_re1000 import _plbe_step_numba as _plbe_jit
    ny, nx = f.shape[1], f.shape[2]
    out = np.empty_like(f)
    rho = np.empty((ny, nx), dtype=np.float64)
    ux = np.empty((ny, nx), dtype=np.float64)
    uy = np.empty((ny, nx), dtype=np.float64)
    _plbe_jit(f, out, rho, ux, uy, self.gamma, self.omega, self.U_wall)
    return out


# ----------------------------------------------------------------------
# Monkey-patch each Case class
# ----------------------------------------------------------------------
_PATCHED = False


def enable_numba_kernels(verbose: bool = False) -> None:
    """Replace each Case.lbe_step with a parallel njit kernel."""
    global _PATCHED
    if _PATCHED:
        return
    from lbm_periodic import KolmogorovCase
    from lbm_channel import ChannelCase
    from lbm_couette import CouetteCase
    from lbm_core import LBMCavity
    from lbm_voxel import VoxelCase
    from lbm_plbe_cavity import PLBECavity

    KolmogorovCase.lbe_step = kolmogorov_step
    ChannelCase.lbe_step = channel_step
    CouetteCase.lbe_step = couette_step
    LBMCavity.lbe_step = cavity_step
    VoxelCase.lbe_step = voxel_step
    PLBECavity.lbe_step = plbe_step
    _PATCHED = True
    if verbose:
        print("[numba_kernels] patched lbe_step on 6 Case classes", flush=True)
