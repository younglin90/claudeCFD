"""Paper-grade 60-run benchmark harness for 2D steady LBM acceleration.

Methods are repo-local implementations for the paper comparison families:
  1. picard_lbm: native fixed-point LBM
  2. anderson_lbm: Walker-Ni type Anderson acceleration on LBM map
  3. preconditioned_lbm: Guo-Zhao-Shi-style PLBE EDF + Picard
  4. inexact_newton_lbe: NE-preconditioned matrix-free JFNK
  5. dual_time_mg_lbm: 2-level dual-time V-cycle proxy
  6. proposed: single public SafeNN-Final dispatcher

The script writes paper-ready machine-readable artifacts:
  paper_revision_data/bench60/
    summary.csv, summary.json, metrics.json
    histories/*.csv
    vtk/*.vtk

The final stdout line is a JSON metrics object for codex-autoresearch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "24")
os.environ.setdefault("OMP_NUM_THREADS", "24")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from lbm_channel import ChannelCase
from lbm_core import LBMCavity, moments as cavity_moments
from lbm_couette import CouetteCase
from lbm_periodic import apply_spectral_schur, build_spectral_schur, KolmogorovCase
from lbm_voxel import VoxelCase, build_cylinder_mask
from paper_extra_benchmarks import backward_step_mask, cylinder_wake_mask, solve_baseline_generic, t_junction_mask
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_scmk import solve_baseline_periodic, solve_scmk, solve_scmk_core
from solver_scmk_direct import solve_scmk_direct
from solver_scmk_mg import solve_scmk_mg
from paper_faithful_baselines import (
    solve_dual_time_mg,
    solve_inexact_newton_ne,
    solve_preconditioned_lbm,
)
from solver_proposed_single import solve_proposed_single


OUT = Path("paper_revision_data") / "bench60"
HIST_DIR = OUT / "histories"
VTK_DIR = OUT / "vtk"


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    return cavity_moments(f)


def res_norm(case, f) -> float:
    r = case.residual(f)
    chi = getattr(case, "chi", None)
    if chi is not None:
        fluid = chi > 0
        return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))
    return float(case._fast_norm(r) / math.sqrt(case.dof))


def velocity_error(case_ref, f_ref, case, f, fluid_mask=None):
    _, ux_ref, uy_ref = macro_of(case_ref, f_ref)
    _, ux, uy = macro_of(case, f)
    if fluid_mask is None:
        fluid_mask = np.ones_like(ux_ref, dtype=bool)
    du = ux[fluid_mask] - ux_ref[fluid_mask]
    dv = uy[fluid_mask] - uy_ref[fluid_mask]
    ref = np.sqrt(ux_ref[fluid_mask] ** 2 + uy_ref[fluid_mask] ** 2)
    den = max(float(np.sqrt(np.sum(ref * ref))), 1e-30)
    npoints = int(du.size) if du.size else 1
    return {
        "vel_abs_l2": float(np.sqrt(np.sum(du * du + dv * dv))),
        "vel_abs_linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv))) if du.size else 0.0),
        "vel_abs_rms": float(np.sqrt(np.mean(du * du + dv * dv)) if npoints else 0.0),
        "rel_l2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
        "rms": float(np.sqrt(np.mean(du * du + dv * dv))),
    }


def write_history_csv(path: Path, hist):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter_or_step", "residual", "lbe_calls", "wall_seconds"])
        for row in hist:
            wr.writerow([row[0], row[1], row[2], row[3] if len(row) > 3 else ""])


def write_vtk(path: Path, case, f):
    path.parent.mkdir(parents=True, exist_ok=True)
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    ny, nx = rho.shape
    chi = getattr(case, "chi", np.ones((ny, nx), dtype=np.float64))
    arrays = {
        "rho": rho,
        "ux": ux,
        "uy": uy,
        "speed": speed,
        "fluid_mask": chi,
    }
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write(f"{path.stem}\n")
        fh.write("ASCII\n")
        fh.write("DATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {nx} {ny} 1\n")
        fh.write("ORIGIN 0 0 0\n")
        fh.write("SPACING 1 1 1\n")
        fh.write(f"POINT_DATA {nx*ny}\n")
        fh.write("VECTORS velocity float\n")
        for j in range(ny):
            for i in range(nx):
                fh.write(f"{ux[j, i]:.9e} {uy[j, i]:.9e} 0.0\n")
        for name, arr in arrays.items():
            fh.write(f"SCALARS {name} float 1\n")
            fh.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    fh.write(f"{arr[j, i]:.9e}\n")


MULTI_CYLINDER_GEOMETRY_VERSION = "physical_v1"
MULTI_CYLINDER_RADIUS = 1.0 / 12.0
MULTI_CYLINDER_CENTERS = (
    (0.1875, 0.140625),
    (0.40625, 0.171875),
    (0.265625, 0.453125),
    (0.6875, 0.5),
    (0.765625, 0.203125),
    (0.796875, 0.609375),
)


def make_multi_cylinder_mask(n=32):
    """Rasterize one fixed physical six-cylinder geometry on an n x n grid.

    Earlier scripts sampled integer cylinder centers independently for each
    grid size, so N=32 and N=64 were statistically similar but not a true grid
    refinement of the same geometry.  This version defines the obstacles in
    unit-square physical coordinates and samples them at lattice cell centers.
    Therefore N=32/64/96/128 all represent the same geometry with only the
    rasterization resolution changed.
    """
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    chi = np.ones((n, n), dtype=np.float64)
    r2_limit = MULTI_CYLINDER_RADIUS * MULTI_CYLINDER_RADIUS
    for cx, cy in MULTI_CYLINDER_CENTERS:
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        chi[r2 < r2_limit] = 0.0
    return chi


def make_t_junction_case(n=64):
    chi = t_junction_mask(n)
    case = VoxelCase(chi, nu=0.05, F0=0.0, kf=0)
    case.Fx = np.zeros((n, n), dtype=np.float64)
    case.Fy = np.zeros((n, n), dtype=np.float64)
    fluid = case.chi > 0
    case.Fx[fluid] = 8.0e-6
    case.Fy[: n // 2, n // 2 - 5 : n // 2 + 6] = -8.0e-6
    case.Fx *= case.chi
    case.Fy *= case.chi
    return case


def case_factory(case_id):
    if case_id == "kolmogorov_n32":
        return "Kolmogorov flow N=32", 1e-7, lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1)
    if case_id == "channel_n32":
        return "Plane Poiseuille channel N=32", 1e-7, lambda: ChannelCase(N=32, nu=0.05, F0=1e-5)
    if case_id == "couette_n32":
        return "Couette flow N=32", 1e-7, lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05)
    if case_id == "cavity_re100_n33":
        return "Lid-driven cavity Re=100 N=33", 5e-7, lambda: LBMCavity(N=33, Re=100, U_wall=0.1)
    if case_id == "cavity_re400_n49":
        return "Lid-driven cavity Re=400 N=49", 5e-7, lambda: LBMCavity(N=49, Re=400, U_wall=0.1)
    if case_id == "cavity_re1000_n129":
        return "Lid-driven cavity Re=1000 N=129", 5e-7, lambda: LBMCavity(N=129, Re=1000, U_wall=0.1)
    if case_id == "multi_cylinder_n32":
        chi = make_multi_cylinder_mask(32)
        return "Multi-cylinder voxel mask N=32", 1e-7, lambda: VoxelCase(chi, nu=0.05, F0=2e-4, kf=1)
    if case_id == "backward_step_n64":
        chi = backward_step_mask(64)
        return "Backward-facing step mask N=64", 1e-7, lambda: VoxelCase(chi, nu=0.05, F0=1.5e-5, kf=0)
    if case_id == "cylinder_wake_n64":
        chi = cylinder_wake_mask(64)
        return "Cylinder wake analogue N=64", 1e-7, lambda: VoxelCase(chi, nu=0.04, F0=1.0e-5, kf=0)
    if case_id == "t_junction_n64":
        return "T-junction mask N=64", 1e-7, make_t_junction_case
    raise ValueError(case_id)


CASE_IDS = [
    "kolmogorov_n32",
    "channel_n32",
    "couette_n32",
    "cavity_re100_n33",
    "cavity_re400_n49",
    "cavity_re1000_n129",
    "multi_cylinder_n32",
    "backward_step_n64",
    "cylinder_wake_n64",
    "t_junction_n64",
]


METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]


def baseline_runner_for(case):
    if isinstance(case, LBMCavity):
        return lambda c, max_steps, tol, check_every, verbose: solve_baseline(
            c, max_steps=max_steps, tol=tol, check_every=check_every, verbose=verbose
        )
    if isinstance(case, VoxelCase):
        return solve_baseline_generic
    return solve_baseline_periodic


def solve_inexact_newton_identity(case, max_outer=140, tol=1e-7, krylov_max=8, krylov_tol=1e-3, kinetic_substeps=10, verbose=False):
    f = case.initial_field()
    n_full = case.dof
    hist = []
    t0 = time.perf_counter()
    lbe = 0
    for k in range(max_outer):
        r = case.residual(f)
        lbe += 1
        rn = case._fast_norm(r) / math.sqrt(n_full)
        hist.append((k, rn, lbe, time.perf_counter() - t0))
        if verbose and (k < 5 or k % 20 == 0 or rn < tol):
            print(f"  newton {k:4d} | res {rn:.3e} | lbe {lbe:7d}", flush=True)
        if not np.isfinite(rn) or rn < tol:
            break
        norm_f = case._fast_norm(f)
        probes = [0]

        def matvec(v_flat):
            probes[0] += 1
            return case.jvp(v_flat.reshape(case.shape), f, r, norm_f_cached=norm_f).ravel()

        op = LinearOperator((n_full, n_full), matvec=matvec, dtype=np.float64)
        df, info = gmres(
            op,
            -r.ravel(),
            rtol=krylov_tol,
            atol=krylov_tol * np.linalg.norm(r) * 1e-3,
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


def run_method(method, case, tol, max_steps, verbose=False):
    t0 = time.perf_counter()
    if method == "picard_lbm":
        runner = baseline_runner_for(case)
        f, hist = runner(case, max_steps=max_steps, tol=tol, check_every=500 if case.N >= 64 else 200, verbose=verbose)
    elif method == "anderson_lbm":
        f, hist = solve_anderson(
            case,
            max_iter=max_steps // 2,
            tol=tol,
            m=5,
            beta=0.8 if isinstance(case, (LBMCavity, VoxelCase)) else 1.0,
            safeguard=True,
            verbose=verbose,
            check_every=5,
        )
    elif method == "preconditioned_lbm":
        # PLBE Picard is a faithful explicit reference, but some stiff masked
        # or high-Re cases can run to the historical Picard max_steps for many
        # minutes.  Use one uniform, grid-scaled computational budget; if it
        # fails to converge within that budget, report it as nonconverged
        # rather than silently making the comparison infeasible.
        plbe_budget = min(max_steps, 40000 if case.N < 64 else 120000)
        f, hist = solve_preconditioned_lbm(
            case,
            max_steps=plbe_budget,
            tol=tol,
            gamma=0.5,
            check_every=500 if case.N >= 64 else 200,
            verbose=verbose,
        )
    elif method == "inexact_newton_lbe":
        f, hist = solve_inexact_newton_ne(
            case,
            max_outer=200,
            tol=tol,
            krylov_max=10,
            krylov_tol=1e-3,
            K_ne=20,
            K_smooth=10,
            line_search_max=4,
            reynolds_continuation=False,
            verbose=verbose,
        )
    elif method == "dual_time_mg_lbm":
        f, hist = solve_dual_time_mg(
            case,
            max_outer=500,
            tol=tol,
            K_pre=2,
            K_coarse=10,
            K_post=2,
            max_levels=6,
            cycle="W",
            lambda_weight=0.7,
            verbose=verbose,
        )
    elif method == "proposed":
        f, hist = solve_proposed_single(
            case,
            tol=tol,
            verbose=verbose,
        )
    else:
        raise ValueError(method)
    return f, hist, time.perf_counter() - t0


def max_steps_for(case_id):
    if case_id == "cavity_re1000_n129":
        return 900000
    if "cavity" in case_id:
        return 250000
    if case_id in {"backward_step_n64", "cylinder_wake_n64"}:
        return 90000
    return 70000


def run_case(case_id, methods, write_fields=True, verbose=False):
    label, tol, factory = case_factory(case_id)
    print(f"[case] {label}", flush=True)
    case_ref = factory()
    f_ref, hist_ref, wall_ref = run_method("picard_lbm", case_ref, tol, max_steps_for(case_id), verbose=verbose)
    ref_res = hist_ref[-1][1]
    fluid = getattr(case_ref, "chi", np.ones((case_ref.N, case_ref.N), dtype=np.float64)) > 0
    results = []
    ref_row = None
    for method in methods:
        if method == "picard_lbm":
            case = case_ref
            f = f_ref
            hist = hist_ref
            wall = wall_ref
        else:
            case = factory()
            try:
                f, hist, wall = run_method(method, case, tol, max_steps_for(case_id), verbose=verbose)
            except Exception as exc:
                print(f"  {method} crashed: {exc}", flush=True)
                f = case.initial_field()
                hist = [(0, float("nan"), 0, 0.0)]
                wall = 0.0
        final_res = float(hist[-1][1])
        lbe = int(hist[-1][2])
        err = velocity_error(case_ref, f_ref, case, f, fluid_mask=fluid)
        converged = bool(np.isfinite(final_res) and final_res < 5.0 * tol)
        row = {
            "case_id": case_id,
            "case_label": label,
            "method": method,
            "tol": tol,
            "N": case.N,
            "baseline_lbe": int(hist_ref[-1][2]),
            "baseline_wall": float(wall_ref),
            "baseline_residual": float(ref_res),
            "lbe_calls": lbe,
            "wall_seconds": float(wall),
            "final_residual": final_res,
            "converged": converged,
            "lbe_speedup_vs_picard": float(hist_ref[-1][2] / max(lbe, 1)) if lbe > 0 else 0.0,
            "wall_speedup_vs_picard": float(wall_ref / max(wall, 1e-12)) if wall > 0 else 0.0,
            "rel_l2_vs_picard": err["rel_l2"],
            "linf_vs_picard": err["linf"],
            "rms_vs_picard": err["rms"],
        }
        results.append(row)
        if method == "picard_lbm":
            ref_row = row
        write_history_csv(HIST_DIR / f"{case_id}__{method}.csv", hist)
        if write_fields:
            write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, f)
        print(
            f"  {method:22s} lbe={lbe:7d} x={row['lbe_speedup_vs_picard']:7.2f} "
            f"res={final_res:.3e} relL2={err['rel_l2']:.3e} conv={converged}",
            flush=True,
        )
    return results


def score_results(rows):
    ordered_cases = []
    for row in rows:
        if row["case_id"] not in ordered_cases:
            ordered_cases.append(row["case_id"])
    by_case = {case_id: [r for r in rows if r["case_id"] == case_id] for case_id in ordered_cases}
    pass_count = 0
    win_lbe = 0
    win_wall = 0
    win_accuracy = 0
    nonfinite = 0
    proposed_rows = []
    for case_id, case_rows in by_case.items():
        prop = next((r for r in case_rows if r["method"] == "proposed"), None)
        if prop is None:
            continue
        proposed_rows.append(prop)
        competitors = [r for r in case_rows if r["method"] != "proposed"]
        if not np.isfinite(prop["final_residual"]) or not np.isfinite(prop["rel_l2_vs_picard"]):
            nonfinite += 1
            continue
        eligible = [
            r
            for r in competitors
            if r["converged"] and np.isfinite(r["rel_l2_vs_picard"]) and r["rel_l2_vs_picard"] <= 0.05
        ]
        if not eligible:
            eligible = [r for r in competitors if r["method"] == "picard_lbm"]
        best_comp_lbe = min(r["lbe_calls"] if r["lbe_calls"] > 0 else 10**18 for r in eligible)
        best_comp_wall = min(r["wall_seconds"] if r["wall_seconds"] > 0 else 10**18 for r in eligible)
        finite_comp_errs = [r["rel_l2_vs_picard"] for r in eligible if np.isfinite(r["rel_l2_vs_picard"])]
        best_comp_err = min(finite_comp_errs) if finite_comp_errs else float("inf")
        lbe_ok = prop["lbe_calls"] <= best_comp_lbe
        wall_ok = prop["wall_seconds"] <= best_comp_wall
        acc_ok = prop["rel_l2_vs_picard"] <= max(best_comp_err, 1e-12) + 0.05
        conv_ok = prop["converged"]
        if lbe_ok:
            win_lbe += 1
        if wall_ok:
            win_wall += 1
        if acc_ok:
            win_accuracy += 1
        if conv_ok and lbe_ok and acc_ok and prop["rel_l2_vs_picard"] <= 0.05:
            pass_count += 1
    worst_rel_l2 = max((r["rel_l2_vs_picard"] for r in proposed_rows if np.isfinite(r["rel_l2_vs_picard"])), default=float("inf"))
    proposed_converged = sum(1 for r in proposed_rows if r["converged"])
    proposed_mean_speedup = float(np.mean([r["lbe_speedup_vs_picard"] for r in proposed_rows]))
    score = (
        100.0 * pass_count
        + 10.0 * win_lbe
        + 4.0 * win_accuracy
        + 2.0 * win_wall
        + min(100.0, proposed_mean_speedup)
        - 40.0 * nonfinite
        - 30.0 * max(0, worst_rel_l2 - 0.05)
    )
    return {
        "score": float(score),
        "pass_count_10": int(pass_count),
        "proposed_converged_count": int(proposed_converged),
        "proposed_wins_lbe_count": int(win_lbe),
        "proposed_wins_wall_count": int(win_wall),
        "proposed_wins_accuracy_count": int(win_accuracy),
        "nonfinite_count": int(nonfinite),
        "worst_rel_l2": float(worst_rel_l2),
        "proposed_mean_lbe_speedup": proposed_mean_speedup,
        "all_pass": int(pass_count == len(ordered_cases) and nonfinite == 0),
    }


def write_summary(rows, metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT / "summary.csv"
    fields = [
        "case_id",
        "case_label",
        "method",
        "tol",
        "N",
        "baseline_lbe",
        "baseline_wall",
        "baseline_residual",
        "lbe_calls",
        "wall_seconds",
        "final_residual",
        "converged",
        "lbe_speedup_vs_picard",
        "wall_speedup_vs_picard",
        "rel_l2_vs_picard",
        "linf_vs_picard",
        "rms_vs_picard",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row[k] for k in fields})
    (OUT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASE_IDS), help="comma-separated case ids")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method ids")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    VTK_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = [x.strip() for x in args.cases.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    rows = []
    started = time.perf_counter()
    for case_id in case_ids:
        rows.extend(run_case(case_id, methods, write_fields=not args.no_vtk, verbose=args.verbose))
    metrics = score_results(rows)
    metrics["elapsed_wall_seconds"] = time.perf_counter() - started
    metrics["case_count"] = len(case_ids)
    metrics["method_count"] = len(methods)
    write_summary(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}", flush=True)
    print(f"[saved] {OUT / 'summary.json'}", flush=True)
    print(f"[saved] {OUT / 'metrics.json'}", flush=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
