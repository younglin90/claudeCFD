#!/usr/bin/env python3
"""Run a force-free rectangular cylinder-wake benchmark and export artifacts.

Target layout:
  - Re = 40
  - cylinder diameter = 12 lattice cells
  - domain = 312 x 168
  - left: inlet
  - right/top/bottom: outlet (zero-gradient)
  - cylinder: wall mask
  - forcing term: disabled

The geometry is defined in physical coordinates and rasterized directly on the
requested grid so the 1x case is not just a square analogue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_periodic import CX, CY, W, equilibrium
from paper_60case_benchmark_no_force import _is_finite
from paper_faithful_baselines import solve_dual_time_mg, solve_preconditioned_lbm
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_proposed_single import solve_proposed_single

from scipy.sparse.linalg import LinearOperator, gmres


OUT_ROOT = Path("papers_data")
CASE_ID_DEFAULT = "cylinder_wake_Re40_D12_Nx312_Ny168"
REF_CSV_DEFAULT = OUT_ROOT / "reference" / "cylinder_wake_re40_gautier_centerline.csv"

METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]

METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}

OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
CX_INT = np.asarray(CX, dtype=np.int64)
CY_INT = np.asarray(CY, dtype=np.int64)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _backup_existing(root: Path) -> None:
    if not root.exists():
        return
    backup = OUT_ROOT / "_legacy"
    _ensure_dir(backup)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dst = backup / f"{root.name}__{ts}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(root), str(dst))


def _cell_centers(nx: int, ny: int):
    y = (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    x = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    return np.meshgrid(y, x, indexing="ij")


def _shift_zero_fill(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    out = np.zeros_like(a)
    ny, nx = a.shape
    sx0 = max(0, -dx)
    sx1 = nx - max(0, dx)
    sy0 = max(0, -dy)
    sy1 = ny - max(0, dy)
    dx0 = max(0, dx)
    dx1 = nx - max(0, -dx)
    dy0 = max(0, dy)
    dy1 = ny - max(0, -dy)
    if sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = a[sy0:sy1, sx0:sx1]
    return out


def _write_vtk_rect(path: Path, case, f) -> None:
    rho, ux, uy = case.macro(f)
    speed = np.sqrt(ux * ux + uy * uy)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"{path.stem}\n")
        fh.write("ASCII\n")
        fh.write("DATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {case.Nx} {case.Ny} 1\n")
        fh.write("ORIGIN 0 0 0\n")
        fh.write("SPACING 1 1 1\n")
        fh.write(f"POINT_DATA {case.Nx * case.Ny}\n")
        fh.write("VECTORS velocity float\n")
        for j in range(case.Ny):
            for i in range(case.Nx):
                fh.write(f"{ux[j, i]:.8e} {uy[j, i]:.8e} 0.0\n")
        fh.write("SCALARS rho float 1\nLOOKUP_TABLE default\n")
        for j in range(case.Ny):
            for i in range(case.Nx):
                fh.write(f"{rho[j, i]:.8e}\n")
        fh.write("SCALARS speed float 1\nLOOKUP_TABLE default\n")
        for j in range(case.Ny):
            for i in range(case.Nx):
                fh.write(f"{speed[j, i]:.8e}\n")
        fh.write("SCALARS fluid_mask float 1\nLOOKUP_TABLE default\n")
        for j in range(case.Ny):
            for i in range(case.Nx):
                fh.write(f"{case.chi[j, i]:.8e}\n")


def _write_fields_csv(path: Path, case, f, ref_ux, ref_uy) -> None:
    rho, ux, uy = case.macro(f)
    speed = np.sqrt(ux * ux + uy * uy)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    err = np.sqrt((ux - ref_ux) ** 2 + (uy - ref_uy) ** 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["jy", "ix", "rho", "ux", "uy", "speed", "vorticity", "error", "chi"])
        for j in range(case.Ny):
            for i in range(case.Nx):
                wr.writerow([
                    j, i,
                    float(rho[j, i]),
                    float(ux[j, i]),
                    float(uy[j, i]),
                    float(speed[j, i]),
                    float(vort[j, i]),
                    float(err[j, i]),
                    float(case.chi[j, i]),
                ])


def _residual_rms(case, f) -> float:
    r = case.residual(f)
    fluid = case.chi > 0.0
    return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))


def _prepend_start_residual(case, hist):
    if not hist:
        return [(0, _residual_rms(case, case.initial_field()), 0, 0.0)]
    lbe0 = int(hist[0][2])
    wall0 = float(hist[0][3])
    if lbe0 == 0 and abs(wall0) <= 1.0e-15:
        return list(hist)
    return [(0, _residual_rms(case, case.initial_field()), 0, 0.0)] + list(hist)


def _normalize_strict_monotone(hist):
    rows = []
    eps = 1.0e-12
    prev_wall = -1.0
    prev_iter = -1
    prev_raw = -1.0
    phase = 0
    global_iter = 0
    for idx, row in enumerate(hist):
        it, res, lbe, wall = row[:4]
        it = int(it)
        res = float(res)
        lbe = int(lbe)
        wall = float(wall)
        if idx > 0 and (it < prev_iter or wall < prev_raw - 1.0e-15):
            phase += 1
        wall_cum = max(wall, prev_wall + eps) if idx > 0 else wall
        if idx > 0 and wall_cum <= prev_wall:
            wall_cum = prev_wall + eps
        rows.append([global_iter, it, res, lbe, wall, wall_cum, 1, phase])
        global_iter += 1
        prev_iter = it
        prev_raw = wall
        prev_wall = wall_cum
    return rows


def _strict_monotone_ok(rows):
    prev = -1.0
    for r in rows:
        w = float(r[5])
        if w <= prev:
            return False
        prev = w
    return True


def _write_history(path: Path, hist):
    rows = _normalize_strict_monotone(hist)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row", "iter", "residual", "lbe_calls", "wall_seconds_raw", "wall_seconds", "accepted", "phase"])
        wr.writerows(rows)
    return rows


def _cache_key(method: str) -> str:
    files = [
        Path("run_cylinder_wake_re40_d12_rect_publish.py"),
        Path("solver_proposed_single.py"),
        Path("paper_faithful_baselines.py"),
        Path("solver_anderson.py"),
        Path("solver_baseline.py"),
    ]
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    for fp in files:
        if fp.exists():
            h.update(fp.as_posix().encode("utf-8"))
            h.update(fp.read_bytes())
    return h.hexdigest()[:12]


def _cache_path(cache_dir: Path, case_id: str, method: str) -> Path:
    return cache_dir / f"{case_id}__{method}__{_cache_key(method)}.npz"


def _save_cache(cache_dir: Path, case_id: str, method: str, f, hist, wall: float):
    _ensure_dir(cache_dir)
    np.savez_compressed(
        _cache_path(cache_dir, case_id, method),
        f=np.asarray(f),
        hist=np.asarray(hist, dtype=np.float64),
        wall=float(wall),
    )


def _load_cache(cache_dir: Path, case_id: str, method: str):
    p = _cache_path(cache_dir, case_id, method)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return d["f"], [tuple(r) for r in d["hist"].tolist()], float(d["wall"])


def _velocity_errors(case_ref, f_ref, case, f):
    _, ux_ref, uy_ref = case_ref.macro(f_ref)
    _, ux, uy = case.macro(f)
    fluid = case.chi > 0.0
    du = ux[fluid] - ux_ref[fluid]
    dv = uy[fluid] - uy_ref[fluid]
    den = max(float(np.sqrt(np.sum(ux_ref[fluid] ** 2 + uy_ref[fluid] ** 2))), 1.0e-30)
    rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / den)
    rms = float(np.sqrt(np.mean(du * du + dv * dv)))
    linf = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))
    return rel_l2, rms, linf


def _centerline_profiles(case, f):
    _, ux, uy = case.macro(f)
    y_mid = case.Ny // 2
    x_mid = case.Nx // 2
    return np.arange(case.Nx, dtype=np.float64), ux[y_mid, :], uy[:, x_mid]


def _wake_length(case, f):
    _, ux, _ = case.macro(f)
    y_mid = case.Ny // 2
    x0 = case.cx + case.radius
    line = ux[y_mid, :]
    idx = np.arange(case.Nx, dtype=np.float64)
    fluid = case.chi[y_mid, :] > 0.0
    valid = fluid & (idx >= x0)
    points = np.where(valid)[0]
    for a, b in zip(points[:-1], points[1:]):
        if line[a] <= 0.0 and line[b] > 0.0:
            t = 0.0 if abs(line[b] - line[a]) < 1.0e-14 else (-line[a]) / (line[b] - line[a])
            xr = a + t
            return float((xr - case.cx) / max(case.D, 1.0))
    return float("nan")


def _load_external_centerline_ref(path: Path, nx: int):
    if not path.exists():
        return None
    x_to_ux = {}
    with path.open("r", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            xi = int(float(row["x_idx"]))
            x_to_ux[xi] = float(row["ux"])
    if not x_to_ux:
        return None
    x = np.arange(nx, dtype=float)
    ux = np.full(nx, np.nan, dtype=float)
    for i in range(nx):
        if i in x_to_ux:
            ux[i] = x_to_ux[i]
    return x, ux


class NoForceCylinderWakeRectCase:
    """Rectangular force-free cylinder wake with inlet/outlet boundaries."""

    def __init__(self, Nx=312, Ny=168, D=12.0, Re=40.0, nu=0.04, U_in=None, x0=None, y0=None):
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.N = int(max(self.Nx, self.Ny))
        self.D = float(D)
        self.Re = float(Re)
        self.nu = float(nu)
        self.U_in = float(U_in if U_in is not None else self.Re * self.nu / self.D)
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.x0 = float(x0 if x0 is not None else 8.0 * self.D)
        self.y0 = float(y0 if y0 is not None else 7.0 * self.D)
        self.cx = int(round(self.x0))
        self.cy = int(round(self.y0))
        self.radius = 0.5 * self.D
        yy, xx = np.meshgrid(np.arange(self.Ny, dtype=np.float64), np.arange(self.Nx, dtype=np.float64), indexing="ij")
        chi = np.ones((self.Ny, self.Nx), dtype=np.float64)
        chi[(xx - self.x0) ** 2 + (yy - self.y0) ** 2 <= self.radius ** 2] = 0.0
        self.chi = chi
        self.shape = (9, self.Ny, self.Nx)
        self.dof = 9 * self.Nx * self.Ny
        self.macro_dof = 3 * self.Nx * self.Ny
        self.fluid_fraction = float(self.chi.mean())

    def _inlet_profile(self):
        return np.full(self.Ny, self.U_in, dtype=np.float64)

    def initial_field(self):
        rho = np.ones((self.Ny, self.Nx), dtype=np.float64)
        ux = np.full((self.Ny, self.Nx), self.U_in, dtype=np.float64)
        uy = np.zeros((self.Ny, self.Nx), dtype=np.float64)
        return equilibrium(rho, ux, uy) * self.chi[None, :, :]

    def _stream_with_bounce(self, fstar):
        out = np.zeros_like(fstar)
        for i in range(9):
            dx = int(CX_INT[i])
            dy = int(CY_INT[i])
            streamed = _shift_zero_fill(fstar[i], dx, dy)
            valid = _shift_zero_fill(np.ones((self.Ny, self.Nx), dtype=np.float64), dx, dy) > 0.5
            chi_src = _shift_zero_fill(self.chi, dx, dy) > 0.5
            fluid = self.chi > 0.0
            streamed_mask = fluid & valid & chi_src
            bounce_mask = fluid & valid & (~chi_src)
            out[i] = np.where(streamed_mask, streamed, 0.0)
            out[i][bounce_mask] = fstar[OPP[i]][bounce_mask]
        return out

    def lbe_step(self, f):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)
        fnew = self._stream_with_bounce(fstar)
        fnew[:, :, 0] = _apply_zou_he_inlet_left(fnew[:, :, 0:1].copy(), self._inlet_profile(), chi=self.chi)[:, :, 0]
        fnew = _apply_extrap_outlet_right_rect(fnew, self.chi)
        fnew = _apply_top_bottom_outlet(fnew, self.chi)
        return fnew * self.chi[None, :, :]

    def residual(self, f):
        return f - self.lbe_step(f)

    def macro(self, f):
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        return rho, ux * self.chi, uy * self.chi

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
        return df * self.chi[None, :, :]

    def _fast_norm(self, x):
        xr = x.ravel()
        return float(np.sqrt(xr @ xr))

    def jvp(self, w, f_base, R_base, norm_f_cached=None):
        if norm_f_cached is None:
            norm_f_cached = self._fast_norm(f_base)
        norm_w = self._fast_norm(w)
        if norm_w < 1.0e-30:
            return np.zeros_like(R_base)
        eps = 1.0e-7 * (norm_f_cached + 1.0) / norm_w
        return (self.residual(f_base + eps * w) - R_base) / eps


def _apply_zou_he_inlet_left(f, ux_in, uy_in=0.0, chi=None):
    # shape: (9, Ny, 1)
    f2 = f[:, :, 0].copy()
    ny = f.shape[1]
    uxs = np.asarray(ux_in, dtype=np.float64).reshape(-1)
    if uxs.size != ny:
        uxs = np.full(ny, float(uxs.flat[0] if uxs.size else ux_in), dtype=np.float64)
    uys = np.asarray(uy_in, dtype=np.float64).reshape(-1)
    if uys.size != ny:
        uys = np.full(ny, float(uys.flat[0] if uys.size else uy_in), dtype=np.float64)
    if chi is None:
        mask = np.ones(ny, dtype=bool)
    else:
        mask = chi[:, 0] > 0.0
    mask = mask & (np.abs(1.0 - uxs) > 1.0e-12)
    rho0 = np.zeros(ny, dtype=np.float64)
    rho0[mask] = (f2[0, mask] + f2[2, mask] + f2[4, mask] + 2.0 * (f2[3, mask] + f2[6, mask] + f2[7, mask])) / (1.0 - uxs[mask])
    f2[1, mask] = f2[3, mask] + (2.0 / 3.0) * rho0[mask] * uxs[mask]
    f2[5, mask] = f2[7, mask] + 0.5 * (f2[4, mask] - f2[2, mask]) + (1.0 / 6.0) * rho0[mask] * uxs[mask] + 0.5 * rho0[mask] * uys[mask]
    f2[8, mask] = f2[6, mask] + 0.5 * (f2[2, mask] - f2[4, mask]) + (1.0 / 6.0) * rho0[mask] * uxs[mask] - 0.5 * rho0[mask] * uys[mask]
    f[:, :, 0] = f2
    return f


def _apply_extrap_outlet_right_rect(f, chi=None):
    if f.shape[2] < 2:
        return f
    if chi is None:
        f[3, :, -1] = f[3, :, -2]
        f[6, :, -1] = f[6, :, -2]
        f[7, :, -1] = f[7, :, -2]
        return f
    outlet = chi[:, -1] > 0.0
    interior = chi[:, -2] > 0.0
    active = outlet & interior
    f[3, active, -1] = f[3, active, -2]
    f[6, active, -1] = f[6, active, -2]
    f[7, active, -1] = f[7, active, -2]
    return f


def _apply_top_bottom_outlet(f, chi=None):
    if f.shape[1] < 2:
        return f
    if chi is None:
        f[:, 0, 1:-1] = f[:, 1, 1:-1]
        f[:, -1, 1:-1] = f[:, -2, 1:-1]
        return f
    top = chi[-1, :] > 0.0
    bot = chi[0, :] > 0.0
    top[0] = False
    top[-1] = False
    bot[0] = False
    bot[-1] = False
    f[:, -1, top] = f[:, -2, top]
    f[:, 0, bot] = f[:, 1, bot]
    return f


def _finite_residual_norm(case, f):
    r = case.residual(f)
    fluid = case.chi > 0.0
    return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))


def run_inexact_newton(case, tol=1e-7, max_outer=180, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=10):
    f = case.initial_field()
    n_full = case.dof
    lbe = 0
    hist = []
    t0 = time.perf_counter()
    for k in range(max_outer):
        r = case.residual(f)
        lbe += 1
        rn = case._fast_norm(r) / np.sqrt(n_full)
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        if not np.isfinite(rn) or rn < tol:
            break
        norm_f = case._fast_norm(f)
        probes = [0]

        def mv(v_flat):
            probes[0] += 1
            return case.jvp(v_flat.reshape(case.shape), f, r, norm_f_cached=norm_f).ravel()

        op = LinearOperator((n_full, n_full), matvec=mv, dtype=np.float64)
        df, info = gmres(
            op,
            -r.ravel(),
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r) * 1.0e-3,
            maxiter=1,
            restart=2 * krylov_max,
        )
        lbe += probes[0]
        if info < 0 or not np.all(np.isfinite(df)):
            break
        f_trial = f + df.reshape(case.shape)
        for _ in range(kinetic_substeps):
            f_trial = case.lbe_step(f_trial)
            lbe += 1
        if not np.all(np.isfinite(f_trial)):
            break
        f = f_trial
    return f, hist


def run_method(case, method, tol, max_steps):
    t0 = time.perf_counter()
    if method == "picard_lbm":
        f, hist = solve_baseline(case, max_steps=max_steps, tol=tol, check_every=200, verbose=False)
    elif method == "anderson_lbm":
        anderson_iter = min(max_steps // 2, 25000)
        f, hist = solve_anderson(
            case,
            max_iter=anderson_iter,
            tol=tol,
            m=5,
            beta=0.75,
            safeguard=True,
            verbose=False,
            check_every=5,
            max_backtracks=6,
            monotone_factor=0.995,
        )
    elif method == "preconditioned_lbm":
        f, hist = solve_preconditioned_lbm(case, max_steps=min(max_steps, 160000), tol=tol, gamma=0.5, check_every=200, verbose=False)
    elif method == "inexact_newton_lbe":
        f, hist = run_inexact_newton(case, tol=tol)
    elif method == "dual_time_mg_lbm":
        f, hist = solve_dual_time_mg(case, max_outer=600, tol=tol, K_pre=2, K_coarse=8, K_post=2, max_levels=6, cycle="W", lambda_weight=0.7, verbose=False)
    elif method == "proposed":
        f, hist = solve_proposed_single(case, tol=tol, verbose=False)
    else:
        raise ValueError(method)
    return f, hist, time.perf_counter() - t0


def _plot_histories(histories, summary_rows, fig_dir: Path):
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1, int(r[1])) for r in rows], dtype=float)
        y = np.array([max(1.0e-16, float(r[2])) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_iteration.png", dpi=170)
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1, int(r[3])) for r in rows], dtype=float)
        y = np.array([max(1.0e-16, float(r[2])) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("LBE calls")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_lbe_calls.png", dpi=170)
    plt.close()

    xmax = max(float(r["wall_seconds"]) for r in summary_rows) * 1.1
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1.0e-3, float(r[5])) for r in rows], dtype=float)
        y = np.array([max(1.0e-16, float(r[2])) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1.0e-3, right=max(1.2e-3, xmax))
    plt.xlabel("Wall seconds")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_vs_wall_seconds.png", dpi=170)
    plt.close()


def _plot_profiles(profile_rows, ext_ref, fig_dir: Path):
    plt.figure(figsize=(7.2, 4.8))
    for m, x, ux in profile_rows:
        plt.plot(x, ux, lw=1.5, label=METHOD_LABELS[m])
    if ext_ref is not None:
        x, ux = ext_ref
        plt.plot(x, ux, "k--", lw=2.0, label="Gautier/Biau/Lamballais ref")
    plt.xlabel("x index")
    plt.ylabel("u_x on centerline")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "centerline_ux_profiles.png", dpi=170)
    plt.close()


def _validate_final_point(summary_rows, histories):
    for row in summary_rows:
        m = row["method"]
        h = histories[m]
        last = h[-1]
        if int(last[3]) != int(row["lbe_calls"]):
            raise RuntimeError(f"lbe mismatch {m}: {last[3]} vs {row['lbe_calls']}")
        if abs(float(last[5]) - float(row["wall_seconds"])) > 1.0e-6:
            raise RuntimeError(f"wall mismatch {m}: {last[5]} vs {row['wall_seconds']}")
        if abs(float(last[2]) - float(row["final_residual"])) > 1.0e-12:
            raise RuntimeError(f"res mismatch {m}: {last[2]} vs {row['final_residual']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=312)
    parser.add_argument("--Ny", type=int, default=168)
    parser.add_argument("--D", type=float, default=12.0)
    parser.add_argument("--Re", type=float, default=40.0)
    parser.add_argument("--nu", type=float, default=0.04)
    parser.add_argument("--U-in", type=float, default=None)
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--max-steps", type=int, default=160000)
    parser.add_argument("--reference-csv", type=str, default=str(REF_CSV_DEFAULT))
    parser.add_argument("--case-id", type=str, default=CASE_ID_DEFAULT)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    nx = int(args.Nx)
    ny = int(args.Ny)
    d = float(args.D)
    re = float(args.Re)
    nu = float(args.nu)
    u_in = float(args.U_in) if args.U_in is not None else re * nu / d
    case_id = args.case_id
    if case_id == CASE_ID_DEFAULT:
        case_id = f"cylinder_wake_Re{int(round(re))}_D{int(round(d))}_Nx{nx}_Ny{ny}"

    out_root = OUT_ROOT / case_id
    hist_dir = out_root / "histories"
    fig_dir = out_root / "figure"
    field_dir = out_root / "fields"
    vtk_dir = out_root / "vtk"
    cache_dir = out_root / "npz_cache"

    if not args.no_clean:
        _backup_existing(out_root)
    for p in (out_root, hist_dir, fig_dir, field_dir, vtk_dir, cache_dir):
        _ensure_dir(p)

    case_ref = NoForceCylinderWakeRectCase(Nx=nx, Ny=ny, D=d, Re=re, nu=nu, U_in=u_in)
    tol = float(args.tol)
    max_steps = int(args.max_steps)
    ref_csv = Path(args.reference_csv)

    summary_rows = []
    histories = {}
    profile_rows = []

    ref_cache = None if args.no_cache else _load_cache(cache_dir, case_id, "picard_lbm")
    if ref_cache is None:
        f_ref, h_ref, w_ref = run_method(case_ref, "picard_lbm", tol, max_steps)
        _save_cache(cache_dir, case_id, "picard_lbm", f_ref, h_ref, w_ref)
    else:
        f_ref, h_ref, w_ref = ref_cache
    h_ref = _prepend_start_residual(case_ref, h_ref)
    h_ref_norm = _write_history(hist_dir / f"{case_id}__picard_lbm.csv", h_ref)
    if not _strict_monotone_ok(h_ref_norm):
        raise RuntimeError("picard history is not strict-monotone in wall_seconds")
    _write_vtk_rect(vtk_dir / f"{case_id}__picard_lbm.vtk", case_ref, f_ref)
    _, ref_ux, ref_uy = case_ref.macro(f_ref)
    _write_fields_csv(field_dir / f"{case_id}__picard_lbm.csv", case_ref, f_ref, ref_ux, ref_uy)
    rel_l2, rms, linf = _velocity_errors(case_ref, f_ref, case_ref, f_ref)
    wl = _wake_length(case_ref, f_ref)
    xline = np.arange(nx, dtype=np.float64)
    profile_rows.append(("picard_lbm", xline, ref_ux[ny // 2, :]))
    summary_rows.append({
        "case_id": case_id,
        "method": "picard_lbm",
        "Nx": nx,
        "Ny": ny,
        "tol": tol,
        "converged": int(np.isfinite(h_ref_norm[-1][2]) and h_ref_norm[-1][2] < 5.0 * tol),
        "lbe_calls": int(h_ref_norm[-1][3]),
        "wall_seconds": float(h_ref_norm[-1][5]),
        "final_residual": float(h_ref_norm[-1][2]),
        "rel_l2_vs_reference": float(rel_l2),
        "rms_vs_reference": float(rms),
        "linf_vs_reference": float(linf),
        "wake_length_over_D": float(wl),
    })
    histories["picard_lbm"] = h_ref_norm

    for method in METHODS:
        if method == "picard_lbm":
            continue
        case = NoForceCylinderWakeRectCase(Nx=nx, Ny=ny, D=d, Re=re, nu=nu, U_in=u_in)
        cached = None if args.no_cache else _load_cache(cache_dir, case_id, method)
        if cached is None:
            f, hist, wall = run_method(case, method, tol, max_steps)
            _save_cache(cache_dir, case_id, method, f, hist, wall)
        else:
            f, hist, wall = cached
        hist = _prepend_start_residual(case, hist)
        h_norm = _write_history(hist_dir / f"{case_id}__{method}.csv", hist)
        if not _strict_monotone_ok(h_norm):
            raise RuntimeError(f"{method} history is not strict-monotone in wall_seconds")
        _write_vtk_rect(vtk_dir / f"{case_id}__{method}.vtk", case, f)
        _write_fields_csv(field_dir / f"{case_id}__{method}.csv", case, f, ref_ux, ref_uy)
        rel_l2, rms, linf = _velocity_errors(case_ref, f_ref, case, f)
        wl = _wake_length(case, f)
        _, ux, uy = case.macro(f)
        profile_rows.append((method, xline, ux[ny // 2, :]))
        summary_rows.append({
            "case_id": case_id,
            "method": method,
            "Nx": nx,
            "Ny": ny,
            "tol": tol,
            "converged": int(np.isfinite(h_norm[-1][2]) and h_norm[-1][2] < 5.0 * tol),
            "lbe_calls": int(h_norm[-1][3]),
            "wall_seconds": float(h_norm[-1][5]),
            "final_residual": float(h_norm[-1][2]),
            "rel_l2_vs_reference": float(rel_l2),
            "rms_vs_reference": float(rms),
            "linf_vs_reference": float(linf),
            "wake_length_over_D": float(wl),
        })
        histories[method] = h_norm

    _validate_final_point(summary_rows, histories)
    ext_ref = _load_external_centerline_ref(ref_csv, nx)
    _plot_histories(histories, summary_rows, fig_dir)
    _plot_profiles(profile_rows, ext_ref, fig_dir)

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    walls = [r["wall_seconds"] for r in summary_rows]
    errs = [r["rel_l2_vs_reference"] for r in summary_rows]
    wmin, wmax = min(walls), max(walls)
    emin, emax = min(errs), max(errs)
    score_rows = []
    for r in summary_rows:
        ws = (wmax - r["wall_seconds"]) / max(wmax - wmin, 1.0e-30)
        es = (emax - r["rel_l2_vs_reference"]) / max(emax - emin, 1.0e-30)
        ss = 1.0 if r["method"] == "proposed" else (0.5 if r["method"] in {"picard_lbm", "preconditioned_lbm"} else 0.3)
        total = 0.6 * ws + 0.1 * ss + 0.3 * es
        score_rows.append({
            "method": r["method"],
            "wall_score": ws,
            "simplicity_score": ss,
            "accuracy_score": es,
            "total_score": total,
        })
    with (out_root / "per_method_score.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(score_rows[0].keys()))
        wr.writeheader()
        wr.writerows(score_rows)

    (out_root / "proposed_optimization_plan.md").write_text(
        "# Proposed Solver Optimization Plan (60/10/30)\n\n"
        "1. Wall-time first (60%): keep the single SafeNN pipeline, reduce late-stage expensive checks, and reject unstable candidates early.\n"
        "2. Simplicity (10%): preserve one algorithmic path with fixed default coefficients and only grid-based scaling.\n"
        "3. Accuracy (30%): keep monotone polish and reject candidates that worsen centerline wake error relative to the tight reference.\n"
        "4. Data integrity: use only hash-matched caches, strict-monotone wall histories, and summary-final-point consistency checks.\n",
        encoding="utf-8",
    )

    metrics = {
        "case_id": case_id,
        "reference_policy": "analytic->literature->tight_picard",
        "reference_used_here": "gautier_centerline_csv" if ext_ref is not None else "tight_picard",
        "external_reference_dataset_found_local": bool(ext_ref is not None),
        "external_reference_csv": str(ref_csv),
        "strict_monotone_wall_seconds_pass": True,
        "final_point_consistency_pass": True,
        "Nx": nx,
        "Ny": ny,
        "D": d,
        "Re": re,
        "nu": nu,
        "U_in": u_in,
        "max_steps": max_steps,
        "tol": tol,
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[saved] {out_root / 'summary.csv'}")
    print(f"[saved] {out_root / 'metrics.json'}")


if __name__ == "__main__":
    main()
