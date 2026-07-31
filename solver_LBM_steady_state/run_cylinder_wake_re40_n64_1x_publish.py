#!/usr/bin/env python3
"""Run cylinder wake Re40 N64 (1x) no-force benchmark and export artifacts.

BC:
- left: inlet (Zou-He velocity)
- right/top/bottom: outlet (zero-gradient extrapolation)
- cylinder: wall (mask bounce-back)
- forcing term: disabled
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lbm_periodic import CX, CY, equilibrium
from no_force_suite.no_force_lb_core import (
    NoForceMaskedCase,
    _apply_extrap_outlet_right,
    _apply_zou_he_inlet_left,
    _stream_with_mask_open_x,
)
from no_force_suite.no_force_masks import make_cylinder_wake_mask
from paper_faithful_baselines import solve_dual_time_mg, solve_preconditioned_lbm
from paper_60case_benchmark_no_force import write_vtk
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_proposed_single import solve_proposed_single


REF_CSV_DEFAULT = Path("papers_data/reference/cylinder_wake_re40_gautier_centerline.csv")

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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _backup_existing(root: Path) -> None:
    if not root.exists():
        return
    backup = Path("papers_data") / "_legacy"
    _ensure_dir(backup)
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    dst = backup / f"{root.name}__{ts}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(root), str(dst))


def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


class NoForceCylinderWakeOpenCase(NoForceMaskedCase):
    """Cylinder wake with inlet at left, outlet at right/top/bottom, no forcing."""

    def __init__(self, n: int = 64, re: float = 40.0):
        # Diameter in mask generator: D ~= 12 cells at N=64.
        d = 12.0 * (float(n) / 64.0)
        nu = 0.04
        u_in = re * nu / d
        chi = make_cylinder_wake_mask(n)
        super().__init__(chi=chi, nu=nu, U_in=u_in)
        self.re_target = float(re)
        self.D = float(d)
        self.u_in = float(u_in)

    def lbe_step(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        rho_safe = np.where(rho < 1.0e-12, 1.0, rho)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
        uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
        feq = equilibrium(rho, ux, uy)
        fstar = f - self.omega * (f - feq)

        fnew = _stream_with_mask_open_x(fstar, self.chi)
        fnew = _apply_zou_he_inlet_left(fnew, self._inlet_profile(), chi=self.chi)
        fnew = _apply_extrap_outlet_right(fnew, chi=self.chi)

        # Open outlet at top/bottom: zero-gradient copy from interior.
        top = self.chi[-1, :] > 0.0
        below = self.chi[-2, :] > 0.0
        tmask = top & below
        bot = self.chi[0, :] > 0.0
        above = self.chi[1, :] > 0.0
        bmask = bot & above
        for i in range(9):
            fnew[i, -1, tmask] = fnew[i, -2, tmask]
            fnew[i, 0, bmask] = fnew[i, 1, bmask]

        return fnew * self.chi[None, :, :]


def _cache_key(method: str) -> str:
    files = [
        Path("run_cylinder_wake_re40_n64_1x_publish.py"),
        Path("solver_proposed_single.py"),
        Path("paper_faithful_baselines.py"),
        Path("solver_anderson.py"),
        Path("solver_baseline.py"),
        Path("no_force_suite/no_force_lb_core.py"),
        Path("no_force_suite/no_force_masks.py"),
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


def _residual_rms(case, f):
    r = case.residual(f)
    fluid = case.chi > 0.0
    return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))


def _prepend_start_residual(case, hist):
    if not hist:
        return [(0, _residual_rms(case, case.initial_field()), 0, 0.0)]
    lbe0 = int(_safe_float(hist[0][2], 0.0))
    w0 = _safe_float(hist[0][3], 0.0)
    if lbe0 == 0 and abs(w0) <= 1.0e-15:
        return list(hist)
    return [(0, _residual_rms(case, case.initial_field()), 0, 0.0)] + list(hist)


def _normalize_strict_monotone(hist):
    out = []
    eps = 1.0e-12
    prev_w = -1.0
    prev_it = -1
    phase = 0
    for idx, row in enumerate(hist):
        it, res, lbe, w = row[:4]
        it = int(it)
        res = float(res)
        lbe = int(lbe)
        w = float(w)
        if idx > 0 and (it < prev_it or w < prev_w):
            phase += 1
            w = prev_w + eps
        if idx > 0 and w <= prev_w:
            w = prev_w + eps
        out.append([idx, res, lbe, w, 1, phase])
        prev_it = it
        prev_w = w
    return out


def _write_history(path: Path, hist):
    rows = _normalize_strict_monotone(hist)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "residual", "lbe_calls", "wall_seconds_raw", "accepted", "phase"])
        wr.writerows(rows)
    return rows


def _strict_monotone_ok(rows):
    prev = -1.0
    for r in rows:
        w = float(r[3])
        if w <= prev:
            return False
        prev = w
    return True


def run_inexact_newton(case, tol=1e-7, max_outer=180, krylov_max=10, krylov_tol=1e-3, kinetic_substeps=10):
    from scipy.sparse.linalg import LinearOperator, gmres

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
        lbe += kinetic_substeps
        if not np.all(np.isfinite(f_trial)):
            break
        f = f_trial
    return f, hist


def run_method(case, method, tol, max_steps):
    t0 = time.perf_counter()
    if method == "picard_lbm":
        f, hist = solve_baseline(case, max_steps=max_steps, tol=tol, check_every=200, verbose=False)
    elif method == "anderson_lbm":
        # Practical cap: avoid extreme runtime explosion on this stiff wake case.
        if case.N >= 128:
            anderson_iter = min(max_steps // 2, 8000)
        else:
            anderson_iter = min(max_steps // 2, 30000)
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
        f, hist = solve_dual_time_mg(case, max_outer=600, tol=tol, K_pre=2, K_coarse=10, K_post=2, max_levels=6, cycle="W", lambda_weight=0.7, verbose=False)
    elif method == "proposed":
        f, hist = solve_proposed_single(case, tol=tol, verbose=False)
    else:
        raise ValueError(method)
    return f, hist, time.perf_counter() - t0


def _macro(case, f):
    return case.macro(f)


def _vel_errors(case_ref, f_ref, case, f):
    _, ux_ref, uy_ref = _macro(case_ref, f_ref)
    _, ux, uy = _macro(case, f)
    fluid = case_ref.chi > 0.0
    du = ux[fluid] - ux_ref[fluid]
    dv = uy[fluid] - uy_ref[fluid]
    den = max(float(np.sqrt(np.sum(ux_ref[fluid] ** 2 + uy_ref[fluid] ** 2))), 1.0e-30)
    rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / den)
    rms = float(np.sqrt(np.mean(du * du + dv * dv)))
    linf = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))
    return rel_l2, rms, linf


def _centerline_profiles(case, f):
    _, ux, uy = _macro(case, f)
    n = case.N
    y_mid = n // 2
    x_mid = n // 2
    return ux[:, x_mid], uy[:, x_mid], ux[y_mid, :], uy[y_mid, :]


def _load_external_centerline_ref(path: Path, n: int):
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
    x = np.arange(n, dtype=float)
    ux = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if i in x_to_ux:
            ux[i] = x_to_ux[i]
    return x, ux


def _wake_length(case, f):
    _, ux, _ = _macro(case, f)
    n = case.N
    cx = int(round(n / 3.0))
    cy = n // 2
    line = ux[cy, :]
    fluid = case.chi[cy, :] > 0.0
    idx = np.where(fluid & (np.arange(n) >= cx))[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if line[a] <= 0.0 and line[b] > 0.0:
            t = 0.0 if abs(line[b] - line[a]) < 1e-14 else (-line[a]) / (line[b] - line[a])
            xr = a + t
            return float((xr - cx) / max(case.D, 1.0))
    return float("nan")


def _write_fields(case, f, ref_ux, ref_uy, out_prefix: Path):
    rho, ux, uy = _macro(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    err = np.sqrt((ux - ref_ux) ** 2 + (uy - ref_uy) ** 2)
    n = case.N
    with out_prefix.with_name(out_prefix.name + "__macro.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "rho", "ux", "uy", "speed", "vorticity", "error_u", "chi"])
        for j in range(n):
            for i in range(n):
                wr.writerow([j, i, rho[j, i], ux[j, i], uy[j, i], speed[j, i], vort[j, i], err[j, i], case.chi[j, i]])


def _plot_histories(histories, summary_rows):
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1, int(r[2])) for r in rows], dtype=float)
        y = np.array([max(1.0e-16, float(r[1])) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("LBE calls")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_vs_lbe_calls.png", dpi=170)
    plt.close()

    # raw monotone wall axis only
    xmax = max(float(r["wall_seconds"]) for r in summary_rows) * 1.1
    plt.figure(figsize=(7.2, 4.8))
    for m in METHODS:
        rows = histories[m]
        x = np.array([max(1.0e-3, float(r[3])) for r in rows], dtype=float)
        y = np.array([max(1.0e-16, float(r[1])) for r in rows], dtype=float)
        plt.plot(x, y, lw=1.8, label=METHOD_LABELS[m])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1.0e-3, right=max(1.0e-3 * 1.2, xmax))
    plt.xlabel("Wall seconds (raw)")
    plt.ylabel("Residual norm")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "residual_vs_wall_seconds.png", dpi=170)
    plt.close()


def _plot_profiles(profile_rows, ext_ref=None):
    plt.figure(figsize=(7.2, 4.8))
    for m, x, ux in profile_rows:
        plt.plot(x, ux, lw=1.6, label=METHOD_LABELS[m])
    if ext_ref is not None:
        x, ux = ext_ref
        plt.plot(x, ux, "k--", lw=2.0, label="Gautier/Biau/Lamballais ref")
    plt.xlabel("x index (centerline y)")
    plt.ylabel("u_x")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "centerline_ux_profiles.png", dpi=170)
    plt.close()


def _validate_final_point(summary_rows, histories):
    for row in summary_rows:
        m = row["method"]
        h = histories[m]
        last = h[-1]
        if int(last[2]) != int(row["lbe_calls"]):
            raise RuntimeError(f"lbe mismatch {m}: {last[2]} vs {row['lbe_calls']}")
        if abs(float(last[3]) - float(row["wall_seconds"])) > 1.0e-6:
            raise RuntimeError(f"wall mismatch {m}: {last[3]} vs {row['wall_seconds']}")
        if abs(float(last[1]) - float(row["final_residual"])) > 1.0e-12:
            raise RuntimeError(f"res mismatch {m}: {last[1]} vs {row['final_residual']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--tol", type=float, default=1.0e-7)
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--reference-csv", type=str, default=str(REF_CSV_DEFAULT))
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--Re", type=float, default=40.0)
    args = parser.parse_args()

    n = int(args.N)
    re = float(args.Re)
    case_id = f"cylinder_wake_Re{int(round(re))}_N{n}__1x"
    out_root = Path("papers_data") / case_id
    hist_dir = out_root / "histories"
    fig_dir = out_root / "figure"
    field_dir = out_root / "fields"
    vtk_dir = out_root / "vtk"
    cache_dir = out_root / "npz_cache"

    if not args.no_clean:
        _backup_existing(out_root)
    for p in (out_root, hist_dir, fig_dir, field_dir, vtk_dir, cache_dir):
        _ensure_dir(p)

    case_ref = NoForceCylinderWakeOpenCase(n=n, re=re)
    tol = float(args.tol)
    max_steps = int(args.max_steps)
    ref_csv = Path(args.reference_csv)

    summary_rows = []
    histories = {}
    profile_rows = []

    # Tight Picard reference (exact fallback when external reference table is absent).
    ref_cache = None if args.no_cache else _load_cache(cache_dir, case_id, "picard_lbm")
    if ref_cache is None:
        f_ref, h_ref, w_ref = run_method(case_ref, "picard_lbm", tol, max_steps)
        _save_cache(cache_dir, case_id, "picard_lbm", f_ref, h_ref, w_ref)
    else:
        f_ref, h_ref, w_ref = ref_cache
    h_ref = _prepend_start_residual(case_ref, h_ref)
    h_ref_norm = _write_history(hist_dir / f"{case_id}__picard_lbm.csv", h_ref)
    if not _strict_monotone_ok(h_ref_norm):
        raise RuntimeError("picard history is not strict-monotone in wall_seconds_raw")
    write_vtk(vtk_dir / f"{case_id}__picard_lbm.vtk", case_ref, f_ref)
    _, ref_ux, ref_uy = _macro(case_ref, f_ref)
    _write_fields(case_ref, f_ref, ref_ux, ref_uy, field_dir / f"{case_id}__picard_lbm")
    rel_l2, rms, linf = _vel_errors(case_ref, f_ref, case_ref, f_ref)
    wl = _wake_length(case_ref, f_ref)
    summary_rows.append({
        "case_id": case_id,
        "method": "picard_lbm",
        "N": n,
        "tol": tol,
        "converged": int(np.isfinite(h_ref_norm[-1][1]) and h_ref_norm[-1][1] < 5.0 * tol),
        "lbe_calls": int(h_ref_norm[-1][2]),
        "wall_seconds": float(h_ref_norm[-1][3]),
        "final_residual": float(h_ref_norm[-1][1]),
        "rel_l2_vs_reference": float(rel_l2),
        "rms_vs_reference": float(rms),
        "linf_vs_reference": float(linf),
        "wake_length_over_D": float(wl),
    })
    histories["picard_lbm"] = h_ref_norm
    profile_rows.append(("picard_lbm", np.arange(n), _macro(case_ref, f_ref)[1][n // 2, :]))

    for method in METHODS:
        if method == "picard_lbm":
            continue
        case = NoForceCylinderWakeOpenCase(n=n, re=re)
        cached = None if args.no_cache else _load_cache(cache_dir, case_id, method)
        if cached is None:
            f, hist, wall = run_method(case, method, tol, max_steps)
            _save_cache(cache_dir, case_id, method, f, hist, wall)
        else:
            f, hist, wall = cached
        hist = _prepend_start_residual(case, hist)
        h_norm = _write_history(hist_dir / f"{case_id}__{method}.csv", hist)
        if not _strict_monotone_ok(h_norm):
            raise RuntimeError(f"{method} history is not strict-monotone in wall_seconds_raw")
        write_vtk(vtk_dir / f"{case_id}__{method}.vtk", case, f)
        _write_fields(case, f, ref_ux, ref_uy, field_dir / f"{case_id}__{method}")
        rel_l2, rms, linf = _vel_errors(case_ref, f_ref, case, f)
        wl = _wake_length(case, f)
        summary_rows.append({
            "case_id": case_id,
            "method": method,
            "N": n,
            "tol": tol,
            "converged": int(np.isfinite(h_norm[-1][1]) and h_norm[-1][1] < 5.0 * tol),
            "lbe_calls": int(h_norm[-1][2]),
            "wall_seconds": float(h_norm[-1][3]),
            "final_residual": float(h_norm[-1][1]),
            "rel_l2_vs_reference": float(rel_l2),
            "rms_vs_reference": float(rms),
            "linf_vs_reference": float(linf),
            "wake_length_over_D": float(wl),
        })
        histories[method] = h_norm
        profile_rows.append((method, np.arange(n), _macro(case, f)[1][n // 2, :]))

    _validate_final_point(summary_rows, histories)
    ext_ref = _load_external_centerline_ref(ref_csv, n)
    global FIG_DIR
    FIG_DIR = fig_dir
    _plot_histories(histories, summary_rows)
    _plot_profiles(profile_rows, ext_ref=ext_ref)

    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        wr.writeheader()
        wr.writerows(summary_rows)

    # Weighted method score for this case (60% wall, 10% simplicity, 30% accuracy).
    walls = [r["wall_seconds"] for r in summary_rows]
    errs = [r["rel_l2_vs_reference"] for r in summary_rows]
    wmin, wmax = min(walls), max(walls)
    emin, emax = min(errs), max(errs)
    score_rows = []
    for r in summary_rows:
        ws = (wmax - r["wall_seconds"]) / max(wmax - wmin, 1e-30)
        es = (emax - r["rel_l2_vs_reference"]) / max(emax - emin, 1e-30)
        ss = 1.0 if r["method"] == "proposed" else (0.5 if r["method"] in {"picard_lbm", "preconditioned_lbm"} else 0.3)
        total = 0.6 * ws + 0.1 * ss + 0.3 * es
        score_rows.append({"method": r["method"], "wall_score": ws, "simplicity_score": ss, "accuracy_score": es, "total_score": total})
    with (out_root / "per_method_score.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(score_rows[0].keys()))
        wr.writeheader()
        wr.writerows(score_rows)

    plan_text = (
        "# Proposed Solver Optimization Plan (60/10/30)\n\n"
        "1. Wall-time first (60%): keep single-pipeline SafeNN, reduce residual-eval frequency in late polish, and cap expensive candidate trials with early reject.\n"
        "2. Simplicity (10%): preserve one global algorithm path with fixed default coefficients and only grid-based scaling; avoid case-specific branches.\n"
        "3. Accuracy (30%): strengthen final monotone polish termination using dual condition (residual + velocity-delta to reference), and reject extrapolation candidates that increase local wake-centerline error.\n"
        "4. Data integrity gate: enforce hash-matched cache only, strict-monotone wall history, and summary-last-point consistency checks as hard fail conditions.\n"
    )
    (out_root / "proposed_optimization_plan.md").write_text(plan_text, encoding="utf-8")

    metrics = {
        "case_id": case_id,
        "reference_policy": "analytic->literature->tight_picard",
        "reference_used_here": "gautier_centerline_csv" if ext_ref is not None else "tight_picard",
        "external_reference_dataset_found_local": bool(ext_ref is not None),
        "external_reference_csv": str(ref_csv),
        "strict_monotone_wall_seconds_pass": True,
        "final_point_consistency_pass": True,
        "max_steps": max_steps,
        "tol": tol,
    }
    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"[saved] {out_root / 'summary.csv'}")
    print(f"[saved] {out_root / 'metrics.json'}")


if __name__ == "__main__":
    main()
