"""Paper-faithful re-implementation of three baseline accelerators.

1. preconditioned_lbm   = Guo-Zhao-Shi PLBE EDF + standard Picard (PRE 70.066706, 2004).
   - Modifies equilibrium only: feq = w_i rho [1 + 3 cu + 4.5 cu^2 / gamma - 1.5 u^2 / gamma]
   - Viscosity reset:           nu = gamma cs^2 (tau - 0.5)  -> tau = 0.5 + 3 nu / gamma
   - Algorithm:                 same explicit Picard (no Newton, no Schur, no Anderson).

2. inexact_newton_lbe   = Huang-Yang-Cai 2017 simplified NKS with local Nonlinear
   Elimination (NE) preconditioner and optional Reynolds continuation
   (single-domain version of the published 2-level ASPIN scheme).
   - Per outer: K_NE Picard sweeps (local nonlinear elimination)
     -> matrix-free JFNK update via case.jvp + GMRES
     -> K_smooth Picard polish.

3. dual_time_mg_lbm     = Jia-Luo 2026 dual-time stepping with a 2-level V-cycle
   over the steady residual.
   - Pseudo-time outer iteration on R(f) = f - L(f)
   - Per outer: K_pre fine LBE smoothing
                -> restrict residual to coarse grid (avg pool 2)
                -> K_coarse coarse Picard smoothing on residual
                -> prolongate correction back (bilinear)
                -> K_post fine LBE smoothing
"""

from __future__ import annotations

import math
import time
import types
from typing import Tuple

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

from numba_kernels import CX, CY, W, OPP


def _case_res_norm(case, r):
    chi = getattr(case, "chi", None)
    if chi is None:
        return float(case._fast_norm(r) / math.sqrt(case.dof))
    fluid = chi > 0.0
    if not np.any(fluid):
        return float(case._fast_norm(r) / math.sqrt(case.dof))
    return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))


def _macro_l2_res_norm(case, f, r, residual_sign: int = 1):
    """Macro L2 norm for residual r = sign * (f - L(f))."""
    try:
        g = f - float(residual_sign) * r
        if hasattr(case, "macro"):
            rho_f, ux_f, uy_f = case.macro(f)
            rho_g, ux_g, uy_g = case.macro(g)
        else:
            from lbm_core import moments
            rho_f, ux_f, uy_f = moments(f)
            rho_g, ux_g, uy_g = moments(g)
        dp = (rho_g - rho_f) / 3.0
        dux = ux_g - ux_f
        duy = uy_g - uy_f
        chi = getattr(case, "chi", None)
        if chi is not None:
            fluid = chi > 0.0
            if np.any(fluid):
                dp = dp[fluid]
                dux = dux[fluid]
                duy = duy[fluid]
        return float(np.sqrt(np.sum(dp * dp) + np.sum(dux * dux) + np.sum(duy * duy)))
    except Exception:
        return _case_res_norm(case, r)


def _bad_cell_mask(case, r, rho1=0.35, rho2=0.85):
    per_cell = np.max(np.abs(r), axis=0)
    chi = getattr(case, "chi", None)
    fluid = (chi > 0.0) if chi is not None else np.ones_like(per_cell, dtype=bool)
    if not np.any(fluid):
        return np.zeros_like(per_cell, dtype=bool), False
    threshold = rho1 * max(float(np.max(per_cell[fluid])), 1e-300)
    bad = (per_cell >= threshold) & fluid
    # NE is most useful when a localized subset dominates the residual.
    localized = int(np.count_nonzero(bad)) < rho2 * int(np.count_nonzero(fluid))
    return bad, localized


def _dilate_mask_periodic(mask, radius=2):
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def _admissible_field(case, f):
    if not np.all(np.isfinite(f)):
        return False
    try:
        if hasattr(case, "macro"):
            rho, ux, uy = case.macro(f)
        else:
            from lbm_core import moments
            rho, ux, uy = moments(f)
        chi = getattr(case, "chi", None)
        fluid = (chi > 0.0) if chi is not None else np.ones_like(rho, dtype=bool)
        if np.any(rho[fluid] <= 1e-10):
            return False
        speed2 = ux[fluid] * ux[fluid] + uy[fluid] * uy[fluid]
        if speed2.size and float(np.max(speed2)) > 0.5 * 0.5:
            return False
    except Exception:
        return False
    return True


def _build_block_preconditioner(case, omega_floor=0.2):
    """Matrix-free point-block/RAS substitute for Newton GMRES.

    The Huang-Cai implementation uses analytic Jacobian blocks and RAS.  This
    repo keeps the native residual oracle matrix-free, so this preconditioner is
    a conservative local collision-block inverse approximation.  It is exposed
    and documented as a substitute, not as the PETSc sparse RAS factorization.
    """
    omega = float(getattr(case, "omega", 1.0))
    scale = 1.0 / max(abs(omega), omega_floor)
    chi = getattr(case, "chi", None)

    def apply(v_flat):
        z = scale * v_flat.reshape(case.shape)
        if chi is not None:
            z = z * chi[None, :, :]
        return z.ravel()

    return apply


# ---------------------------------------------------------------------------
# PLBE equilibrium (gamma-preconditioned)
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _feq_plbe(i, rho, ux, uy, gamma):
    cu = CX[i] * ux + CY[i] * uy
    u2 = ux * ux + uy * uy
    return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu / gamma - 1.5 * u2 / gamma)


@njit(cache=True, parallel=True)
def _moments(f, rho, ux, uy):
    ny, nx = rho.shape
    for y in prange(ny):
        for x in range(nx):
            r = 0.0; mx = 0.0; my = 0.0
            for i in range(9):
                fi = f[i, y, x]
                r += fi; mx += CX[i] * fi; my += CY[i] * fi
            rho[y, x] = r
            if r > 1e-12:
                ux[y, x] = mx / r; uy[y, x] = my / r
            else:
                ux[y, x] = 0.0; uy[y, x] = 0.0


# --- Kolmogorov PLBE step (periodic + Guo) ---
@njit(cache=True, parallel=True)
def _plbe_kolmo(f, out, Fx, Fy, omega, gamma):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            fxe = Fx[y, x] / gamma
            fye = Fy[y, x] / gamma
            uxs = ux[y, x] + 0.5 * fxe / r
            uys = uy[y, x] + 0.5 * fye / r
            for i in range(9):
                feq = _feq_plbe(i, r, uxs, uys, gamma)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * fxe + CY[i] * fye
                eu_F = (CX[i] - ux[y, x]) * fxe + (CY[i] - uy[y, x]) * fye
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny; xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]


# --- Channel PLBE step (periodic-x + bb walls) ---
@njit(cache=True, parallel=True)
def _plbe_channel(f, out, Fx, Fy, omega, gamma):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            fxe = Fx[y, x] / gamma
            fye = Fy[y, x] / gamma
            uxs = ux[y, x] + 0.5 * fxe / r
            uys = uy[y, x] + 0.5 * fye / r
            for i in range(9):
                feq = _feq_plbe(i, r, uxs, uys, gamma)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * fxe + CY[i] * fye
                eu_F = (CX[i] - ux[y, x]) * fxe + (CY[i] - uy[y, x]) * fye
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny; xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]
    for x in prange(nx):
        out[2, 0, x] = out[4, 0, x]; out[5, 0, x] = out[7, 0, x]; out[6, 0, x] = out[8, 0, x]
        out[4, ny - 1, x] = out[2, ny - 1, x]; out[7, ny - 1, x] = out[5, ny - 1, x]
        out[8, ny - 1, x] = out[6, ny - 1, x]


# --- Couette PLBE step ---
@njit(cache=True, parallel=True)
def _plbe_couette(f, out, omega, U_wall, gamma):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            for i in range(9):
                feq = _feq_plbe(i, r, ux[y, x], uy[y, x], gamma)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq)
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny; xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]
    for x in prange(nx):
        out[2, 0, x] = out[4, 0, x]; out[5, 0, x] = out[7, 0, x]; out[6, 0, x] = out[8, 0, x]
        rho_top = (out[0, ny - 1, x] + out[1, ny - 1, x] + out[3, ny - 1, x]
                   + 2.0 * (out[2, ny - 1, x] + out[5, ny - 1, x] + out[6, ny - 1, x]))
        out[4, ny - 1, x] = out[2, ny - 1, x]
        out[7, ny - 1, x] = out[5, ny - 1, x] - 6.0 * W[5] * rho_top * U_wall
        out[8, ny - 1, x] = out[6, ny - 1, x] + 6.0 * W[6] * rho_top * U_wall


@njit(cache=True)
def _apply_plbe_horizontal_neq(out, gamma, U_top):
    ny, nx = out.shape[1], out.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(out, rho, ux, uy)
    for x in range(nx):
        rb = rho[1, x]
        rn = rho[1, x]
        for i in (2, 5, 6):
            out[i, 0, x] = _feq_plbe(i, rb, 0.0, 0.0, gamma) + (
                out[i, 1, x] - _feq_plbe(i, rn, ux[1, x], uy[1, x], gamma)
            )
        rb = rho[ny - 2, x]
        rn = rho[ny - 2, x]
        for i in (4, 7, 8):
            out[i, ny - 1, x] = _feq_plbe(i, rb, U_top, 0.0, gamma) + (
                out[i, ny - 2, x] - _feq_plbe(i, rn, ux[ny - 2, x], uy[ny - 2, x], gamma)
            )


@njit(cache=True)
def _plbe_channel_neq(f, out, Fx, Fy, omega, gamma):
    _plbe_channel(f, out, Fx, Fy, omega, gamma)
    _apply_plbe_horizontal_neq(out, gamma, 0.0)


@njit(cache=True)
def _plbe_couette_neq(f, out, omega, U_wall, gamma):
    _plbe_couette(f, out, omega, U_wall, gamma)
    _apply_plbe_horizontal_neq(out, gamma, U_wall)


# --- Cavity PLBE step (4 walls bb + lid) ---
@njit(cache=True, parallel=True)
def _plbe_cavity(f, out, omega, U_wall, gamma):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(f, rho, ux, uy)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            for i in range(9):
                feq = _feq_plbe(i, r, ux[y, x], uy[y, x], gamma)
                out[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq)
    f_in = out.copy()
    for y in prange(ny):
        for x in range(nx):
            for i in range(9):
                ys = (y - CY[i]) % ny; xs = (x - CX[i]) % nx
                out[i, y, x] = f_in[i, ys, xs]
    for y in prange(ny):
        out[1, y, 0] = out[3, y, 0]; out[5, y, 0] = out[7, y, 0]; out[8, y, 0] = out[6, y, 0]
        out[3, y, nx - 1] = out[1, y, nx - 1]; out[6, y, nx - 1] = out[8, y, nx - 1]
        out[7, y, nx - 1] = out[5, y, nx - 1]
    for x in prange(nx):
        out[2, 0, x] = out[4, 0, x]; out[5, 0, x] = out[7, 0, x]; out[6, 0, x] = out[8, 0, x]
        rho_top = (out[0, ny - 1, x] + out[1, ny - 1, x] + out[3, ny - 1, x]
                   + 2.0 * (out[2, ny - 1, x] + out[5, ny - 1, x] + out[6, ny - 1, x]))
        out[4, ny - 1, x] = out[2, ny - 1, x]
        out[7, ny - 1, x] = out[5, ny - 1, x] - 6.0 * W[5] * rho_top * U_wall
        out[8, ny - 1, x] = out[6, ny - 1, x] + 6.0 * W[6] * rho_top * U_wall


# --- Voxel PLBE step (mask + Guo) ---
@njit(cache=True, parallel=True)
def _plbe_voxel(f, out, chi, omega, Fx, Fy, gamma):
    ny, nx = f.shape[1], f.shape[2]
    rho = np.empty((ny, nx)); ux = np.empty((ny, nx)); uy = np.empty((ny, nx))
    _moments(f, rho, ux, uy)
    fstar = np.empty_like(f)
    for y in prange(ny):
        for x in range(nx):
            r = rho[y, x]
            if r < 1e-12: r = 1.0
            fxe = Fx[y, x] / gamma
            fye = Fy[y, x] / gamma
            uxs = ux[y, x] + 0.5 * fxe / r
            uys = uy[y, x] + 0.5 * fye / r
            for i in range(9):
                feq = _feq_plbe(i, r, uxs, uys, gamma)
                cu = CX[i] * ux[y, x] + CY[i] * uy[y, x]
                e_dot_F = CX[i] * fxe + CY[i] * fye
                eu_F = (CX[i] - ux[y, x]) * fxe + (CY[i] - uy[y, x]) * fye
                S = (1.0 - 0.5 * omega) * W[i] * (3.0 * eu_F + 9.0 * cu * e_dot_F)
                fstar[i, y, x] = f[i, y, x] - omega * (f[i, y, x] - feq) + S
    for y in prange(ny):
        for x in range(nx):
            ci = (chi[y, x] == 1.0)
            for i in range(9):
                ys = (y - CY[i]) % ny; xs = (x - CX[i]) % nx
                src_fluid = ci and (chi[ys, xs] == 1.0)
                if src_fluid:
                    out[i, y, x] = fstar[i, ys, xs] * chi[y, x]
                else:
                    out[i, y, x] = fstar[OPP[i], y, x] * chi[y, x]


# ---------------------------------------------------------------------------
# PreconditionedCaseWrapper
# ---------------------------------------------------------------------------
def _select_plbe_step(case):
    cname = type(case).__name__
    if cname == "KolmogorovCase":
        return lambda c, f, out, gamma: _plbe_kolmo(f, out, c.Fx, c.Fy, c.omega, gamma)
    if cname == "ChannelCase":
        return lambda c, f, out, gamma: _plbe_channel_neq(f, out, c.Fx, c.Fy, c.omega, gamma)
    if cname == "CouetteCase":
        return lambda c, f, out, gamma: _plbe_couette_neq(f, out, c.omega, c.U_wall, gamma)
    if cname == "LBMCavity":
        return None
    if cname == "VoxelCase":
        return lambda c, f, out, gamma: _plbe_voxel(f, out, c.chi, c.omega, c.Fx, c.Fy, gamma)
    if cname == "NoForceChannelCase":
        return None
    if cname == "NoForceMaskedCase":
        return None
    if cname == "NoForceCylinderWakeRectCase":
        return None
    if cname == "NoForceCylinderWakeOpenCase":
        return None
    if cname == "NoForcePoiseuilleRectCase":
        return None
    if cname == "NoForceTJunctionRectCase":
        return None
    if cname == "PLBECavity":
        # PLBECavity already uses PLBE EDF — return its existing step
        from numba_kernels import plbe_step
        return None  # signal: use case.lbe_step as-is
    raise ValueError(f"no PLBE step for {cname}")


def wrap_as_preconditioned(case, gamma: float = 0.5):
    """Build a wrapper case that uses PLBE EDF with given gamma.

    Re-derives omega from nu and gamma per PRE 70: tau = 0.5 + 3 nu / gamma.
    """
    cname = type(case).__name__
    if cname == "PLBECavity":
        return case  # already paper-correct preconditioned
    if cname == "LBMCavity":
        from lbm_plbe_cavity import PLBECavity
        return PLBECavity(N=case.N, Re=case.Re, U_wall=case.U_wall, gamma=gamma)
    step_fn = _select_plbe_step(case)
    # nu retrieval
    nu = getattr(case, "nu", None)
    if nu is None:
        # LBMCavity stores nu = U_wall*(N-1)/Re
        nu = case.U_wall * (case.N - 1) / case.Re
    plbe_omega = 1.0 / (0.5 + 3.0 * nu / gamma)

    wrap = types.SimpleNamespace()
    # mirror attributes
    for attr in ("N", "nu", "Re", "U_wall", "shape", "dof", "macro_dof", "F0", "kf",
                 "Fx", "Fy", "chi", "k_lat", "U_amp", "Nx", "Ny", "D", "U_in", "x0", "y0"):
        if hasattr(case, attr):
            setattr(wrap, attr, getattr(case, attr))
    wrap.gamma = gamma
    wrap.omega = plbe_omega
    wrap.base = case

    if step_fn is not None:
        def lbe_step(self, f):
            out = np.empty_like(f)
            step_fn(self, f, out, gamma)
            return out
    else:
        def lbe_step(self, f):
            return self.base.lbe_step(f)

    def residual(self, f):
        return f - self.lbe_step(f)

    def initial_field(self):
        return self.base.initial_field()

    def macro(self, f):
        if hasattr(self.base, "macro"):
            return self.base.macro(f)
        from lbm_core import moments as _m
        return _m(f)

    def project(self, f):
        return self.base.project(f) if hasattr(self.base, "project") else f

    def lift(self, dU):
        return self.base.lift(dU) if hasattr(self.base, "lift") else dU

    def _fast_norm(self, x):
        return float(np.sqrt(np.sum(x * x)))

    def jvp(self, w, f_base, r_base, norm_f_cached=None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1e-30:
            return np.zeros_like(w)
        eps = 1e-7 * (1.0 + norm_f_cached) / norm_w
        return (self.residual(f_base + eps * w) - r_base) / eps

    wrap.lbe_step = types.MethodType(lbe_step, wrap)
    wrap.residual = types.MethodType(residual, wrap)
    wrap.initial_field = types.MethodType(initial_field, wrap)
    wrap.macro = types.MethodType(macro, wrap)
    wrap.project = types.MethodType(project, wrap)
    wrap.lift = types.MethodType(lift, wrap)
    wrap._fast_norm = types.MethodType(_fast_norm, wrap)
    wrap.jvp = types.MethodType(jvp, wrap)
    return wrap


# ---------------------------------------------------------------------------
# Solver 1: paper-faithful preconditioned LBM = PLBE EDF + Picard
# ---------------------------------------------------------------------------
def solve_preconditioned_lbm(case, max_steps=300000, tol=1e-7, check_every=200,
                              gamma=0.5, verbose=False,
                              plateau_window=50, plateau_eps=0.05):
    """PRE 70.066706 PLBE + standard Picard."""
    pcase = wrap_as_preconditioned(case, gamma=gamma)
    f = pcase.initial_field()
    hist = []
    res_hist = []
    t0 = time.perf_counter()
    lbe = 0

    # Record the initial PLBE residual so residual-vs-LBE-call plots do not
    # appear to start only after the first sparse checkpoint.
    R0 = f - pcase.lbe_step(f); lbe += 1
    res0 = _macro_l2_res_norm(pcase, f, R0)
    hist.append((0, res0, lbe, time.perf_counter() - t0))
    if not np.isfinite(res0):
        solve_preconditioned_lbm.last_stats = {
            "gamma": gamma,
            "omega": float(getattr(pcase, "omega", float("nan"))),
            "force_scaled_by_inverse_gamma": True,
            "converged": False,
            "lbe_calls": lbe,
            "final_residual": res0,
        }
        return f, hist

    for step in range(1, max_steps + 1):
        f_new = pcase.lbe_step(f)
        lbe += 1
        if step % check_every == 0:
            R = f_new - pcase.lbe_step(f_new); lbe += 1
            res = _macro_l2_res_norm(pcase, f_new, R)
            hist.append((step, res, lbe, time.perf_counter() - t0))
            res_hist.append(res)
            if verbose:
                print(f"  PLBE picard {step:7d} | res {res:.3e}", flush=True)
            plateaued = False
            if len(res_hist) >= plateau_window:
                tail = res_hist[-plateau_window:]
                half = max(plateau_window // 2, 1)
                old = float(np.median(tail[:half]))
                new = float(np.median(tail[half:]))
                if np.isfinite(old) and old > 0 and np.isfinite(new):
                    plateaued = (old - new) / old <= plateau_eps
            if not np.isfinite(res) or plateaued:
                f = f_new; break
        f = f_new
    if not hist:
        R = f - pcase.lbe_step(f); lbe += 1
        hist.append((max_steps, _macro_l2_res_norm(pcase, f, R), lbe, time.perf_counter() - t0))
    solve_preconditioned_lbm.last_stats = {
        "gamma": gamma,
        "omega": float(getattr(pcase, "omega", float("nan"))),
        "force_scaled_by_inverse_gamma": True,
        "lbe_calls": lbe,
        "operator": "PLBE",
        "wall_boundary_note": "NEQ extrapolation for channel/couette/cavity, periodic bounce-back mask for voxel obstacles",
    }
    return f, hist


solve_preconditioned_lbm.last_stats = {}


# ---------------------------------------------------------------------------
# Solver 2: Huang-Cai NE-preconditioned inexact Newton-Krylov (simplified)
# ---------------------------------------------------------------------------
def solve_inexact_newton_ne(case, max_outer=200, tol=1e-7,
                             krylov_max=10, krylov_tol=1e-3,
                             K_ne=20, K_smooth=10, line_search_max=4,
                             reynolds_continuation=False, verbose=False,
                             plateau_window=50, plateau_eps=0.05):
    """JFNK with local Nonlinear Elimination preconditioner.

    Paper-faithful spirit of Huang-Yang-Cai (2017) ASPIN-style scheme,
    simplified to single-domain (no Schwarz partitioning). Reynolds
    continuation is enabled only for cavity cases.
    """
    f = case.initial_field()
    n_full = case.dof
    hist = []
    stats = {
        "gmres_probes": 0,
        "preconditioner_applications": 0,
        "gmres_failures": 0,
        "line_search_rejects": 0,
        "ne_attempts": 0,
        "ne_accepts": 0,
        "picard_fallbacks": 0,
        "lbe_calls": 0,
    }
    t0 = time.perf_counter()
    lbe = 0

    # Reynolds continuation: build a sequence of cases [Re/4, Re/2, Re] and
    # warm-start each. Only applies to LBMCavity / PLBECavity.
    cases_seq = [case]
    if reynolds_continuation and hasattr(case, "Re") and case.Re > 200:
        # crude continuation: only one half-Re predecessor
        try:
            half = type(case)(N=case.N, Re=case.Re / 2, U_wall=case.U_wall)
            cases_seq = [half, case]
        except Exception:
            cases_seq = [case]

    for stage_idx, stage_case in enumerate(cases_seq):
        prev_rn = None
        res_hist = []
        for k in range(max_outer):
            r = stage_case.residual(f); lbe += 1
            rn = _macro_l2_res_norm(stage_case, f, r)
            hist.append((stage_idx * max_outer + k, rn, lbe, time.perf_counter() - t0))
            res_hist.append(rn)
            if verbose and (k < 3 or k % 20 == 0):
                print(f"  NKS stage={stage_idx} k={k:3d} | res {rn:.3e} | lbe {lbe}")
            if not np.isfinite(rn):
                break
            plateaued = False
            if len(res_hist) >= plateau_window:
                tail = res_hist[-plateau_window:]
                half = max(plateau_window // 2, 1)
                old = float(np.median(tail[:half]))
                new = float(np.median(tail[half:]))
                if np.isfinite(old) and old > 0 and np.isfinite(new):
                    plateaued = (old - new) / old <= plateau_eps
            if plateaued:
                break

            # ---- Bad-cell nonlinear elimination preconditioner ----
            slow = prev_rn is not None and rn / max(prev_rn, 1e-300) > 0.75
            bad, localized = _bad_cell_mask(stage_case, r)
            if K_ne > 0 and (slow or localized):
                stats["ne_attempts"] += 1
                patch = _dilate_mask_periodic(bad, radius=2)
                f_ne = f.copy()
                for _ in range(K_ne):
                    f_step = stage_case.lbe_step(f_ne); lbe += 1
                    f_ne[:, patch] = f_step[:, patch]
                r_ne = stage_case.residual(f_ne); lbe += 1
                rn_ne = _macro_l2_res_norm(stage_case, f_ne, r_ne)
                if np.isfinite(rn_ne) and rn_ne < rn and _admissible_field(stage_case, f_ne):
                    f = f_ne
                    r = r_ne
                    rn = rn_ne
                    stats["ne_accepts"] += 1

            # ---- Matrix-free JFNK update ----
            norm_f = stage_case._fast_norm(f)
            probes = [0]
            def matvec(v):
                probes[0] += 1
                return stage_case.jvp(v.reshape(stage_case.shape), f, r,
                                       norm_f_cached=norm_f).ravel()
            op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
            precond_apply = _build_block_preconditioner(stage_case)
            def psolve(v):
                stats["preconditioner_applications"] += 1
                return precond_apply(v)
            mop = LinearOperator((n_full, n_full), matvec=psolve, dtype=np.float64)
            eta = min(krylov_tol, max(1e-7, 0.5 * math.sqrt(max(rn, 1e-300))))
            df, info = gmres(op, -r.ravel(),
                              M=mop,
                              rtol=eta,
                              atol=eta * np.linalg.norm(r) * 1e-3,
                              maxiter=1, restart=2 * krylov_max)
            lbe += probes[0]
            stats["gmres_probes"] += probes[0]
            if info < 0 or not np.all(np.isfinite(df)):
                stats["gmres_failures"] += 1
                break

            # ---- Line search ----
            alpha = 1.0; accepted = False
            for _ in range(line_search_max):
                f_trial = f + alpha * df.reshape(stage_case.shape)
                if not _admissible_field(stage_case, f_trial):
                    alpha *= 0.5
                    stats["line_search_rejects"] += 1
                    continue
                # smooth with K_smooth picard
                for _ in range(K_smooth):
                    f_trial = stage_case.lbe_step(f_trial); lbe += 1
                r_trial = stage_case.residual(f_trial); lbe += 1
                rt = _macro_l2_res_norm(stage_case, f_trial, r_trial)
                if np.isfinite(rt) and rt <= (1.0 - 1e-4 * alpha) * rn:
                    f = f_trial; accepted = True; break
                alpha *= 0.5
                stats["line_search_rejects"] += 1
            if not accepted:
                for _ in range(K_smooth):
                    f = stage_case.lbe_step(f); lbe += 1
                stats["picard_fallbacks"] += 1
            prev_rn = rn
    stats["lbe_calls"] = lbe
    solve_inexact_newton_ne.last_stats = stats
    return f, hist


solve_inexact_newton_ne.last_stats = {}


# ---------------------------------------------------------------------------
# Solver 3: dual-time stepping with 2-level V-cycle MG (Jia-Luo 2026 spirit)
# ---------------------------------------------------------------------------
def _restrict_avg(f):
    """Average-pool by 2 along (y, x): (9, N, N) -> (9, N/2, N/2)."""
    ny, nx = f.shape[1], f.shape[2]
    ny2 = ny // 2
    nx2 = nx // 2
    if 2 * ny2 != ny or 2 * nx2 != nx:
        f = f[:, : 2 * ny2, : 2 * nx2]
    return 0.25 * (
        f[:, 0::2, 0::2]
        + f[:, 1::2, 0::2]
        + f[:, 0::2, 1::2]
        + f[:, 1::2, 1::2]
    )


def _restrict_bilinear(f):
    """Boundary-safe bilinear/full-weighting restriction for residual fields."""
    q, ny, nx = f.shape
    ncy = ny // 2
    ncx = nx // 2
    out = np.zeros((q, ncy, ncx), dtype=np.float64)
    weights = np.array(
        [[1.0, 3.0, 3.0, 1.0],
         [3.0, 9.0, 9.0, 3.0],
         [3.0, 9.0, 9.0, 3.0],
         [1.0, 3.0, 3.0, 1.0]],
        dtype=np.float64,
    ) / 64.0
    for jc in range(ncy):
        for ic in range(ncx):
            j0 = 2 * jc - 1
            i0 = 2 * ic - 1
            acc = np.zeros(q, dtype=np.float64)
            for dj in range(4):
                jf = min(max(j0 + dj, 0), ny - 1)
                for di in range(4):
                    if_ = min(max(i0 + di, 0), nx - 1)
                    acc += weights[dj, di] * f[:, jf, if_]
            out[:, jc, ic] = acc
    return out


def _prolongate_bilinear(c, target_shape):
    """Cell-centered bilinear prolongation of coarse correction to fine grid.

    Handles any (target_nf, source_nc) including odd nf where 2*nc may
    exceed nf (truncate to nf) or fall short (replicate edge)."""
    q, ny, nx = target_shape
    nc_y, nc_x = c.shape[1], c.shape[2]
    out = np.zeros(target_shape, dtype=np.float64)
    for y in range(ny):
        yc = (y + 0.5) / 2.0 - 0.5
        y0 = int(math.floor(yc))
        ty = yc - y0
        y0 = min(max(y0, 0), nc_y - 1)
        y1 = min(y0 + 1, nc_y - 1)
        for x in range(nx):
            xc = (x + 0.5) / 2.0 - 0.5
            x0 = int(math.floor(xc))
            tx = xc - x0
            x0 = min(max(x0, 0), nc_x - 1)
            x1 = min(x0 + 1, nc_x - 1)
            out[:, y, x] = (
                (1.0 - ty) * (1.0 - tx) * c[:, y0, x0]
                + ty * (1.0 - tx) * c[:, y1, x0]
                + (1.0 - ty) * tx * c[:, y0, x1]
                + ty * tx * c[:, y1, x1]
            )
    return out


def _coarse_case(case):
    """Build coarse-grid analog of `case` (N -> N/2)."""
    cname = type(case).__name__
    n2 = case.N // 2
    if n2 < 8:
        return None
    try:
        if cname == "KolmogorovCase":
            from lbm_periodic import KolmogorovCase
            coarse = KolmogorovCase(N=n2, nu=case.nu, F0=case.F0, kf=case.kf)
        elif cname == "ChannelCase":
            from lbm_channel import ChannelCase
            coarse = ChannelCase(N=n2, nu=case.nu, F0=case.F0)
        elif cname == "CouetteCase":
            from lbm_couette import CouetteCase
            coarse = CouetteCase(N=n2, nu=case.nu, U_wall=case.U_wall)
        elif cname == "LBMCavity":
            from lbm_core import LBMCavity
            coarse = LBMCavity(N=n2 if n2 % 2 == 1 else n2 + 1, Re=case.Re, U_wall=case.U_wall)
        elif cname == "PLBECavity":
            from lbm_plbe_cavity import PLBECavity
            coarse = PLBECavity(N=n2 if n2 % 2 == 1 else n2 + 1, Re=case.Re,
                                U_wall=case.U_wall, gamma=case.gamma)
        elif cname == "VoxelCase":
            from lbm_voxel import VoxelCase
            chi_c = case.chi[: 2 * (case.chi.shape[0] // 2): 2,
                              : 2 * (case.chi.shape[1] // 2): 2].copy()
            coarse = VoxelCase(chi_c, nu=case.nu, F0=case.F0, kf=getattr(case, "kf", 0))
        elif cname == "NoForceMaskedCase":
            from no_force_suite.no_force_lb_core import NoForceMaskedCase
            if getattr(case, "chi", None) is None:
                return None
            chi_c = case.chi[: 2 * (case.chi.shape[0] // 2): 2,
                              : 2 * (case.chi.shape[1] // 2): 2].copy()
            coarse = NoForceMaskedCase(chi_c, nu=case.nu, U_in=case.U_in)
        elif cname == "NoForceCylinderWakeRectCase":
            chi = getattr(case, "chi", None)
            if chi is None:
                return None
            chi_c = chi[: 2 * (chi.shape[0] // 2): 2,
                        : 2 * (chi.shape[1] // 2): 2].copy()
            coarse = type(case)(
                Nx=chi_c.shape[1],
                Ny=chi_c.shape[0],
                D=case.D / 2.0,
                Re=case.Re,
                U_in=case.U_in,
                x0=case.x0 / 2.0,
                y0=case.y0 / 2.0,
            )
            coarse.chi = chi_c
            coarse.fluid_fraction = float(chi_c.mean())
        elif cname == "NoForceChannelCase":
            from no_force_suite.no_force_lb_core import NoForceChannelCase
            coarse = NoForceChannelCase(n2, nu=case.nu, U_in=case.U_in)
        elif cname == "NoForcePoiseuilleRectCase":
            from no_force_suite.no_force_lb_core import NoForcePoiseuilleRectCase
            coarse = NoForcePoiseuilleRectCase(
                Ny=max(8, case.Ny // 2),
                Nx=max(16, case.Nx // 2),
                nu=case.nu,
                U_in=case.U_in,
                initial_profile=getattr(case, "initial_profile", "poiseuille"),
            )
        elif cname == "NoForceTJunctionRectCase":
            from no_force_suite.no_force_lb_core import NoForceTJunctionRectCase
            chi = getattr(case, "chi", None)
            if chi is None:
                return None
            chi_c = chi[: 2 * (chi.shape[0] // 2): 2,
                        : 2 * (chi.shape[1] // 2): 2].copy()
            coarse = NoForceTJunctionRectCase(chi_c, nu=case.nu, U_in=case.U_in)
        else:
            return None
        # The DTS-MG paper keeps the same dimensionless relaxation frequency
        # across levels where feasible.  The coarse operator is a correction
        # equation, so preserving omega is more important than matching a
        # standalone physical coarse Reynolds number.
        if hasattr(case, "omega"):
            coarse.omega = case.omega
        return coarse
    except Exception:
        return None
    return None


def _match_shape(a, shape):
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=np.float64)
    yy = min(a.shape[1], shape[1])
    xx = min(a.shape[2], shape[2])
    out[:, :yy, :xx] = a[:, :yy, :xx]
    if yy < shape[1]:
        out[:, yy:, :xx] = out[:, yy - 1:yy, :xx]
    if xx < shape[2]:
        out[:, :, xx:] = out[:, :, xx - 1:xx]
    return out


def _smooth_residual_equation(case, f, defect, sweeps, lambda_weight, stats):
    if defect is None:
        defect = np.zeros_like(f)
    for _ in range(sweeps):
        R = case.residual(f)
        stats["lbe_calls"] += 1
        f_next = f - lambda_weight * (R - defect)
        chi = getattr(case, "chi", None)
        if chi is not None:
            f_next *= chi[None, :, :]
        if not np.all(np.isfinite(f_next)):
            stats["unstable_smooths"] += 1
            return f
        f = f_next
    return f


def _fas_cycle(case, f, defect, level, max_levels, n_pre, n_post, n_coarse,
               mu, lambda_weight, stats):
    coarse = _coarse_case(case)
    if level >= max_levels - 1 or coarse is None:
        return _smooth_residual_equation(case, f, defect, n_coarse, lambda_weight, stats)

    f_pre = _smooth_residual_equation(case, f, defect, n_pre, lambda_weight, stats)
    R_pre = case.residual(f_pre)
    stats["lbe_calls"] += 1

    f_c0 = _match_shape(_restrict_avg(f_pre), coarse.shape)
    restricted_defect_residual = _restrict_bilinear(R_pre if defect is None else (R_pre - defect))
    restricted_defect_residual = _match_shape(restricted_defect_residual, coarse.shape)
    R_c0 = coarse.residual(f_c0)
    stats["lbe_calls"] += 1
    defect_c = R_c0 - restricted_defect_residual

    f_c = f_c0.copy()
    for _ in range(mu):
        f_c = _fas_cycle(
            coarse, f_c, defect_c, level + 1, max_levels,
            n_pre, n_post, n_coarse, mu, lambda_weight, stats
        )
    correction = _prolongate_bilinear(f_c - f_c0, case.shape)
    f_corr = f_pre + correction
    chi = getattr(case, "chi", None)
    if chi is not None:
        f_corr *= chi[None, :, :]
    return _smooth_residual_equation(case, f_corr, defect, n_post, lambda_weight, stats)


def solve_dual_time_mg(case, max_outer=500, tol=1e-7,
                        K_pre=20, K_coarse=30, K_post=20,
                        check_every=5, verbose=False,
                        max_levels=6, cycle="V", lambda_weight=0.7,
                        max_backtracks=5,
                        plateau_window=50, plateau_eps=0.05):
    """Residual-driven FAS dual-time MG baseline.

    Outer pseudo-time iteration. Each cycle:
        weighted-Jacobi smoothing on R(f)=D
        FAS coarse defect correction
        recursive V/W-cycle
    """
    f = case.initial_field()
    hist = []
    res_hist = []
    stats = {
        "lbe_calls": 0,
        "cycles": 0,
        "accepted_cycles": 0,
        "cycle_backtracks": 0,
        "cycle_rejects": 0,
        "picard_fallbacks": 0,
        "unstable_smooths": 0,
        "final_lambda": lambda_weight,
    }
    t0 = time.perf_counter()
    mu = 2 if str(cycle).upper().startswith("W") else 1
    lambda_current = float(lambda_weight)
    lambda_min = 0.1
    for k in range(max_outer):
        R = case.residual(f)
        stats["lbe_calls"] += 1
        rn = _macro_l2_res_norm(case, f, R)
        hist.append((k, rn, stats["lbe_calls"], time.perf_counter() - t0))
        res_hist.append(rn)
        if verbose and (k < 3 or k % 10 == 0):
            print(f"  DT-MG k={k:3d} | res {rn:.3e} | lbe {stats['lbe_calls']}", flush=True)
        plateaued = False
        if len(res_hist) >= plateau_window:
            tail = res_hist[-plateau_window:]
            half = max(plateau_window // 2, 1)
            old = float(np.median(tail[:half]))
            new = float(np.median(tail[half:]))
            if np.isfinite(old) and old > 0 and np.isfinite(new):
                plateaued = (old - new) / old <= plateau_eps
        if not np.isfinite(rn) or plateaued:
            break
        f_old = f
        f_cycle = _fas_cycle(
            case, f, None, 0, max_levels,
            K_pre, K_post, K_coarse, mu, lambda_current, stats
        )
        stats["cycles"] += 1
        accepted = False
        alpha = 1.0
        for bt in range(max_backtracks + 1):
            f_trial = f_old + alpha * (f_cycle - f_old)
            chi = getattr(case, "chi", None)
            if chi is not None:
                f_trial *= chi[None, :, :]
            R_trial = case.residual(f_trial)
            stats["lbe_calls"] += 1
            rt = _macro_l2_res_norm(case, f_trial, R_trial)
            if np.isfinite(rt) and rt <= rn:
                f = f_trial
                stats["accepted_cycles"] += 1
                stats["cycle_backtracks"] += bt
                if bt == 0:
                    lambda_current = min(float(lambda_weight), lambda_current * 1.05)
                accepted = True
                break
            alpha *= 0.5
            stats["cycle_backtracks"] += 1
        if not accepted:
            stats["cycle_rejects"] += 1
            lambda_current = max(lambda_min, 0.5 * lambda_current)
            f_picard = case.lbe_step(f_old)
            stats["lbe_calls"] += 1
            R_picard = case.residual(f_picard)
            stats["lbe_calls"] += 1
            rp = _macro_l2_res_norm(case, f_picard, R_picard)
            if np.isfinite(rp) and rp <= rn:
                f = f_picard
                stats["picard_fallbacks"] += 1
            else:
                f = f_old
    solve_dual_time_mg.last_stats = stats
    stats["final_lambda"] = lambda_current
    return f, hist


solve_dual_time_mg.last_stats = {}
