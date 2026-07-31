"""Verify paper-style PLBE cavity Re=1000 against stored local references."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

from lbm_plbe_cavity import PLBECavity
from solver_unified_safe_nn import _residual_norm


OUT = Path("paper_revision_data") / "plbe_re1000"
BOUNCEBACK_REF_VTK = Path("paper_revision_data") / "bench60" / "vtk" / "cavity_re1000_n129__picard_lbm.vtk"
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0])


@njit(cache=True)
def _feq_one(i, rho, ux, uy, gamma):
    cu = CX[i] * ux + CY[i] * uy
    u2 = ux * ux + uy * uy
    return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu / gamma - 1.5 * u2 / gamma)


@njit(cache=True, parallel=True)
def _macro_numba(f, rho, ux, uy):
    n = f.shape[1]
    for y in prange(n):
        for x in range(n):
            r = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[i, y, x]
                r += fi
                mx += fi * CX[i]
                my += fi * CY[i]
            rho[y, x] = r
            if r > 1.0e-12:
                ux[y, x] = mx / r
                uy[y, x] = my / r
            else:
                ux[y, x] = 0.0
                uy[y, x] = 0.0


@njit(cache=True)
def _apply_side_bc(f, rho, ux, uy, gamma, side, u_wall):
    n = f.shape[1]
    if side == 0:  # left
        xb = 0
        xn = 1
        for y in range(n):
            rb = rho[y, xn]
            rn = rho[y, xn]
            uxb = 0.0
            uyb = 0.0
            uxn = ux[y, xn]
            uyn = uy[y, xn]
            for i in (1, 5, 8):
                f[i, y, xb] = _feq_one(i, rb, uxb, uyb, gamma) + (f[i, y, xn] - _feq_one(i, rn, uxn, uyn, gamma))
    elif side == 1:  # right
        xb = n - 1
        xn = n - 2
        for y in range(n):
            rb = rho[y, xn]
            rn = rho[y, xn]
            uxb = 0.0
            uyb = 0.0
            uxn = ux[y, xn]
            uyn = uy[y, xn]
            for i in (3, 6, 7):
                f[i, y, xb] = _feq_one(i, rb, uxb, uyb, gamma) + (f[i, y, xn] - _feq_one(i, rn, uxn, uyn, gamma))
    elif side == 2:  # bottom
        yb = 0
        yn = 1
        for x in range(n):
            rb = rho[yn, x]
            rn = rho[yn, x]
            uxb = 0.0
            uyb = 0.0
            uxn = ux[yn, x]
            uyn = uy[yn, x]
            for i in (2, 5, 6):
                f[i, yb, x] = _feq_one(i, rb, uxb, uyb, gamma) + (f[i, yn, x] - _feq_one(i, rn, uxn, uyn, gamma))
    else:  # top
        yb = n - 1
        yn = n - 2
        for x in range(n):
            rb = rho[yn, x]
            rn = rho[yn, x]
            uxb = u_wall
            uyb = 0.0
            uxn = ux[yn, x]
            uyn = uy[yn, x]
            for i in (4, 7, 8):
                f[i, yb, x] = _feq_one(i, rb, uxb, uyb, gamma) + (f[i, yn, x] - _feq_one(i, rn, uxn, uyn, gamma))


@njit(cache=True)
def _apply_bc_numba(f, rho, ux, uy, gamma, u_wall):
    _macro_numba(f, rho, ux, uy)
    _apply_side_bc(f, rho, ux, uy, gamma, 0, u_wall)
    _apply_side_bc(f, rho, ux, uy, gamma, 1, u_wall)
    _apply_side_bc(f, rho, ux, uy, gamma, 2, u_wall)
    _apply_side_bc(f, rho, ux, uy, gamma, 3, u_wall)


@njit(cache=True, parallel=True)
def _plbe_step_numba(f, out, rho, ux, uy, gamma, omega, u_wall):
    n = f.shape[1]
    _macro_numba(f, rho, ux, uy)
    for y in prange(n):
        for x in range(n):
            r = rho[y, x]
            ux0 = ux[y, x]
            uy0 = uy[y, x]
            for i in range(9):
                fs = f[i, y, x] - omega * (f[i, y, x] - _feq_one(i, r, ux0, uy0, gamma))
                yd = (y + CY[i]) % n
                xd = (x + CX[i]) % n
                out[i, yd, xd] = fs
    _apply_bc_numba(out, rho, ux, uy, gamma, u_wall)


@njit(cache=True)
def _velocity_change_numba(f, prev, rho, ux, uy, rho_prev, ux_prev, uy_prev):
    n = f.shape[1]
    _macro_numba(f, rho, ux, uy)
    _macro_numba(prev, rho_prev, ux_prev, uy_prev)
    num = 0.0
    den = 0.0
    for y in range(n):
        for x in range(n):
            dux = ux[y, x] - ux_prev[y, x]
            duy = uy[y, x] - uy_prev[y, x]
            num += dux * dux + duy * duy
            den += ux[y, x] * ux[y, x] + uy[y, x] * uy[y, x]
    if den < 1.0e-60:
        den = 1.0e-60
    return np.sqrt(num) / np.sqrt(den)


@njit(cache=True, parallel=True)
def _initial_plbe_numba(f, rho, ux, uy, gamma, u_wall):
    n = f.shape[1]
    for y in prange(n):
        for x in range(n):
            rho0 = 1.0
            ux0 = 0.0
            uy0 = 0.0
            if y == n - 1:
                ux0 = u_wall
            for i in range(9):
                f[i, y, x] = _feq_one(i, rho0, ux0, uy0, gamma)
    _apply_bc_numba(f, rho, ux, uy, gamma, u_wall)


def velocity_change_norm(case, f, f_prev):
    _, ux, uy = case.macro(f)
    _, ux_prev, uy_prev = case.macro(f_prev)
    num = np.sqrt(np.sum((ux - ux_prev) ** 2 + (uy - uy_prev) ** 2))
    den = max(float(np.sqrt(np.sum(ux * ux + uy * uy))), 1.0e-30)
    return float(num / den)


def run_plbe(case, max_steps=50000, tol=1.0e-6, check_every=100, verbose=False):
    n = case.N
    f = np.empty((9, n, n), dtype=np.float64)
    out = np.empty_like(f)
    rho = np.empty((n, n), dtype=np.float64)
    ux = np.empty((n, n), dtype=np.float64)
    uy = np.empty((n, n), dtype=np.float64)
    rho_prev = np.empty((n, n), dtype=np.float64)
    ux_prev = np.empty((n, n), dtype=np.float64)
    uy_prev = np.empty((n, n), dtype=np.float64)
    _initial_plbe_numba(f, rho, ux, uy, case.gamma, case.U_wall)
    prev = f.copy()
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    for step in range(1, max_steps + 1):
        _plbe_step_numba(f, out, rho, ux, uy, case.gamma, case.omega, case.U_wall)
        f, out = out, f
        lbe += 1
        if not np.all(np.isfinite(f)):
            hist.append((step, float("nan"), lbe, time.perf_counter() - t0))
            break
        if step % check_every == 0:
            change = _velocity_change_numba(f, prev, rho, ux, uy, rho_prev, ux_prev, uy_prev)
            hist.append((step, change, lbe, time.perf_counter() - t0))
            if verbose:
                print(f"  PLBE step {step:7d} change={change:.3e} lbe={lbe:7d}")
            if change < tol:
                break
            prev = f.copy()
    return f, hist


def read_vtk_velocity(path: Path, n: int):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("VECTORS velocity"):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"missing VECTORS velocity in {path}")
    data = np.loadtxt(lines[start : start + n * n], dtype=np.float64)
    ux = data[:, 0].reshape(n, n)
    uy = data[:, 1].reshape(n, n)
    return ux, uy


def velocity_error_arrays(ux_ref, uy_ref, ux, uy):
    dux = ux - ux_ref
    duy = uy - uy_ref
    num = float(np.sqrt(np.sum(dux * dux + duy * duy)))
    den = max(float(np.sqrt(np.sum(ux_ref * ux_ref + uy_ref * uy_ref))), 1.0e-30)
    mag = np.sqrt(dux * dux + duy * duy)
    return {
        "rel_l2": num / den,
        "linf": float(np.max(mag)),
        "rms": float(np.sqrt(np.mean(dux * dux + duy * duy))),
    }


def row_for_gamma(gamma, ux_ref, uy_ref, baseline_lbe, max_steps=50000, verbose=False):
    case = PLBECavity(N=129, Re=1000, U_wall=0.1, gamma=gamma)
    f, hist = run_plbe(case, max_steps=max_steps, tol=1.0e-6, check_every=100, verbose=verbose)
    final_change = float(hist[-1][1]) if hist else float("nan")
    lbe = int(hist[-1][2]) if hist else 0
    _, native_res = _residual_norm(case, f) if np.all(np.isfinite(f)) else (None, float("nan"))
    if np.all(np.isfinite(f)):
        _, ux, uy = case.macro(f)
        err = velocity_error_arrays(ux_ref, uy_ref, ux, uy)
    else:
        err = {"rel_l2": float("inf"), "linf": float("inf"), "rms": float("inf")}
    return {
        "method": "paper_plbe_neq",
        "gamma": gamma,
        "tau": case.tau,
        "omega": case.omega,
        "lbe_calls": lbe,
        "velocity_change_100": final_change,
        "native_residual": float(native_res),
        "speedup_vs_picard": baseline_lbe / max(lbe, 1),
        "rel_l2_vs_bounceback_picard": err["rel_l2"],
        "linf_vs_bounceback_picard": err["linf"],
        "rms_vs_bounceback_picard": err["rms"],
        "finite": bool(np.all(np.isfinite(f))),
        "paper_converged": bool(np.isfinite(final_change) and final_change < 1.0e-6),
    }


def write_rows(rows):
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: (not r["paper_converged"], r["rel_l2_vs_bounceback_picard"], r["lbe_calls"]))
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows_sorted[0].keys()))
        wr.writeheader()
        wr.writerows(rows_sorted)
    metrics = {
        "baseline_lbe": 17535,
        "variant_count": len(rows_sorted),
        "best": rows_sorted[0],
        "paper_converged_count": sum(int(r["paper_converged"]) for r in rows_sorted),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ux_ref, uy_ref = read_vtk_velocity(BOUNCEBACK_REF_VTK, 129)
    baseline_lbe = 17535
    rows = []
    gammas = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    for gamma in gammas:
        row = row_for_gamma(gamma, ux_ref, uy_ref, baseline_lbe, max_steps=50000, verbose=False)
        rows.append(row)
        write_rows(rows)
        print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda r: (not r["paper_converged"], r["rel_l2_vs_bounceback_picard"], r["lbe_calls"]))
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
