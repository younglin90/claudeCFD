"""Allocation-light D2Q9 kernels for large 2D revision calculations.

The public case APIs match the existing LBM case classes closely enough for the
Newton-Krylov solvers, while the native LBE step itself is compiled and streams
directly into a caller-provided output array.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from lbm_periodic import CX, CY, W


CX_N = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0], dtype=np.float64)
CY_N = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=np.float64)
W_N = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float64)
CX_I = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY_I = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)


@njit(cache=True, fastmath=True)
def _fill_equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, out: np.ndarray) -> None:
    ny, nx = rho.shape
    for j in range(ny):
        for k in range(nx):
            r = rho[j, k]
            u = ux[j, k]
            v = uy[j, k]
            u2 = 1.5 * (u * u + v * v)
            for q in range(9):
                cu = 3.0 * (CX_N[q] * u + CY_N[q] * v)
                out[q, j, k] = W_N[q] * r * (1.0 + cu + 0.5 * cu * cu - u2)


@njit(cache=True, fastmath=True, parallel=True)
def _periodic_forced_step(f, out, omega, fx_y, force_is_y_profile):
    n = f.shape[1]
    coeff = 1.0 - 0.5 * omega
    for j in prange(n):
        for k in range(n):
            f0 = f[0, j, k]
            f1 = f[1, j, k]
            f2 = f[2, j, k]
            f3 = f[3, j, k]
            f4 = f[4, j, k]
            f5 = f[5, j, k]
            f6 = f[6, j, k]
            f7 = f[7, j, k]
            f8 = f[8, j, k]
            rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            rhoux = f1 + f5 + f8 - f3 - f6 - f7
            rhouy = f2 + f5 + f6 - f4 - f7 - f8
            ux = rhoux / rho
            uy = rhouy / rho
            fx = fx_y[j] if force_is_y_profile else fx_y[0]
            ux_eq = ux + 0.5 * fx / rho
            uy_eq = uy
            u2 = 1.5 * (ux_eq * ux_eq + uy_eq * uy_eq)
            vals = (f0, f1, f2, f3, f4, f5, f6, f7, f8)
            for q in range(9):
                cu_eq = 3.0 * (CX_N[q] * ux_eq + CY_N[q] * uy_eq)
                feq = W_N[q] * rho * (1.0 + cu_eq + 0.5 * cu_eq * cu_eq - u2)
                cu = CX_N[q] * ux + CY_N[q] * uy
                term = ((CX_N[q] - ux) + 3.0 * cu * CX_N[q]) * fx
                source = coeff * W_N[q] * 3.0 * term
                post = vals[q] - omega * (vals[q] - feq) + source
                jj = (j + CY_I[q]) % n
                kk = (k + CX_I[q]) % n
                out[q, jj, kk] = post


@njit(cache=True, fastmath=True)
def _apply_channel_bc(out):
    n = out.shape[1]
    for k in range(n):
        out[2, 0, k] = out[4, 0, k]
        out[5, 0, k] = out[7, 0, k]
        out[6, 0, k] = out[8, 0, k]
        out[4, n - 1, k] = out[2, n - 1, k]
        out[7, n - 1, k] = out[5, n - 1, k]
        out[8, n - 1, k] = out[6, n - 1, k]


@njit(cache=True, fastmath=True, parallel=True)
def _cavity_step(f, out, omega, u_wall):
    n = f.shape[1]
    for j in prange(n):
        for k in range(n):
            f0 = f[0, j, k]
            f1 = f[1, j, k]
            f2 = f[2, j, k]
            f3 = f[3, j, k]
            f4 = f[4, j, k]
            f5 = f[5, j, k]
            f6 = f[6, j, k]
            f7 = f[7, j, k]
            f8 = f[8, j, k]
            rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
            rhoux = f1 + f5 + f8 - f3 - f6 - f7
            rhouy = f2 + f5 + f6 - f4 - f7 - f8
            ux = rhoux / rho
            uy = rhouy / rho
            u2 = 1.5 * (ux * ux + uy * uy)
            vals = (f0, f1, f2, f3, f4, f5, f6, f7, f8)
            for q in range(9):
                cu = 3.0 * (CX_N[q] * ux + CY_N[q] * uy)
                feq = W_N[q] * rho * (1.0 + cu + 0.5 * cu * cu - u2)
                post = vals[q] - omega * (vals[q] - feq)
                jj = (j + CY_I[q]) % n
                kk = (k + CX_I[q]) % n
                out[q, jj, kk] = post

    for j in range(n):
        out[1, j, 0] = out[3, j, 0]
        out[5, j, 0] = out[7, j, 0]
        out[8, j, 0] = out[6, j, 0]
        out[3, j, n - 1] = out[1, j, n - 1]
        out[6, j, n - 1] = out[8, j, n - 1]
        out[7, j, n - 1] = out[5, j, n - 1]

    for k in range(n):
        out[2, 0, k] = out[4, 0, k]
        out[5, 0, k] = out[7, 0, k]
        out[6, 0, k] = out[8, 0, k]

        rho_top = (
            out[0, n - 1, k]
            + out[1, n - 1, k]
            + out[3, n - 1, k]
            + 2.0 * (out[2, n - 1, k] + out[5, n - 1, k] + out[6, n - 1, k])
        )
        out[4, n - 1, k] = out[2, n - 1, k]
        out[7, n - 1, k] = out[5, n - 1, k] - (1.0 / 6.0) * rho_top * u_wall
        out[8, n - 1, k] = out[6, n - 1, k] + (1.0 / 6.0) * rho_top * u_wall


@njit(cache=True, fastmath=True)
def _norm_diff(a, b):
    acc = 0.0
    size = a.size
    flat_a = a.ravel()
    flat_b = b.ravel()
    for i in range(size):
        d = flat_a[i] - flat_b[i]
        acc += d * d
    return np.sqrt(acc / size)


def _equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    out = np.empty((9,) + rho.shape, dtype=np.float64)
    _fill_equilibrium(rho, ux, uy, out)
    return out


class _OptimizedBase:
    def _fast_norm(self, x):
        xr = x.ravel()
        return float(np.sqrt(xr @ xr))

    def macro(self, f):
        rho = f.sum(axis=0)
        ux = (f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho
        uy = (f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho
        return rho, ux, uy

    def project(self, f):
        rho = f.sum(axis=0)
        rhoux = f[1] + f[5] + f[8] - f[3] - f[6] - f[7]
        rhouy = f[2] + f[5] + f[6] - f[4] - f[7] - f[8]
        return np.stack([rho, rhoux, rhouy], axis=0)

    def lift(self, dU):
        drho, drhoux, drhouy = dU[0], dU[1], dU[2]
        df = np.empty((9,) + drho.shape, dtype=np.float64)
        for i in range(9):
            df[i] = W[i] * (drho + 3.0 * CX[i] * drhoux + 3.0 * CY[i] * drhouy)
        return df

    def residual(self, f):
        return f - self.lbe_step(f)

    def res_norm(self, f):
        return self._fast_norm(self.residual(f)) / np.sqrt(self.dof)

    def jvp(self, w, f_base, R_base, norm_f_cached=None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1e-30:
            return np.zeros_like(R_base)
        eps = 1e-7 * (norm_f_cached + 1.0) / norm_w
        return (self.residual(f_base + eps * w) - R_base) / eps


class OptimizedKolmogorovCase(_OptimizedBase):
    def __init__(self, N, nu, F0=1e-5, kf=1):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.kf = kf
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N
        y = np.arange(N, dtype=np.float64)
        self.k_lat = 2.0 * np.pi * kf / N
        self.Fx_y = F0 * np.sin(self.k_lat * y)
        self.U_amp = F0 / (nu * self.k_lat * self.k_lat)
        self.Re = self.U_amp * N / nu

    def lbe_step_into(self, f, out):
        _periodic_forced_step(f, out, self.omega, self.Fx_y, True)
        return out

    def lbe_step(self, f):
        out = np.empty_like(f)
        return self.lbe_step_into(f, out)

    def initial_field(self):
        z = np.zeros((self.N, self.N), dtype=np.float64)
        return _equilibrium(np.ones((self.N, self.N), dtype=np.float64), z, z)

    def analytical_ux(self):
        y = np.arange(self.N, dtype=np.float64).reshape(self.N, 1)
        return self.U_amp * np.sin(self.k_lat * y) * np.ones((self.N, self.N))


class OptimizedChannelCase(_OptimizedBase):
    def __init__(self, N, nu, F0=1e-5):
        self.N = N
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.F0 = F0
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N
        self.Fx_y = np.array([F0], dtype=np.float64)
        L = N - 1.0
        self.L_eff = L
        self.U_max = F0 * L * L / (8.0 * nu)
        self.Re = self.U_max * L / nu

    def lbe_step_into(self, f, out):
        _periodic_forced_step(f, out, self.omega, self.Fx_y, False)
        _apply_channel_bc(out)
        return out

    def lbe_step(self, f):
        out = np.empty_like(f)
        return self.lbe_step_into(f, out)

    def initial_field(self):
        z = np.zeros((self.N, self.N), dtype=np.float64)
        return _equilibrium(np.ones((self.N, self.N), dtype=np.float64), z, z)

    def analytical_ux(self):
        y = np.arange(self.N, dtype=np.float64)
        prof = self.F0 / (2.0 * self.nu) * y * (self.L_eff - y)
        return prof.reshape(self.N, 1) * np.ones((self.N, self.N))


class OptimizedCavityCase(_OptimizedBase):
    def __init__(self, N, Re, U_wall=0.1):
        self.N = N
        self.Re = Re
        self.U_wall = U_wall
        self.nu = U_wall * (N - 1) / Re
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.shape = (9, N, N)
        self.dof = 9 * N * N
        self.macro_dof = 3 * N * N

    def lbe_step_into(self, f, out):
        _cavity_step(f, out, self.omega, self.U_wall)
        return out

    def lbe_step(self, f):
        out = np.empty_like(f)
        return self.lbe_step_into(f, out)

    def initial_field(self):
        rho = np.ones((self.N, self.N), dtype=np.float64)
        ux = np.zeros((self.N, self.N), dtype=np.float64)
        uy = np.zeros((self.N, self.N), dtype=np.float64)
        ux[-1, :] = self.U_wall
        return _equilibrium(rho, ux, uy)


def solve_baseline_fast(case, max_steps=300000, tol=1e-7, check_every=500, verbose=True):
    """Picard iteration with ping-pong LBE buffers and compact history."""
    import time

    f = case.initial_field()
    buf = np.empty_like(f)
    probe = np.empty_like(f)
    history = []
    t0 = time.perf_counter()
    lbe_calls = 0
    for step in range(1, max_steps + 1):
        case.lbe_step_into(f, buf)
        f, buf = buf, f
        lbe_calls += 1
        if step % check_every == 0:
            case.lbe_step_into(f, probe)
            lbe_calls += 1
            res = float(_norm_diff(f, probe))
            wall = time.perf_counter() - t0
            history.append((step, res, lbe_calls, wall))
            if verbose and (step == check_every or step % (10 * check_every) == 0):
                print(f"  step {step:8d} | res {res:.3e} | wall {wall:.1f}s", flush=True)
            if not np.isfinite(res):
                break
            if res < tol:
                break
    return f, history
