"""Reference-solver-only production smoke verification.

This script intentionally excludes every proposed/SafeNN solver.  It exists to
verify that the comparison solvers can run, report exact accounting metadata,
and write reproducible artifacts before any paper-claim benchmark is resumed.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
import argparse
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from lbm_channel import ChannelCase
from lbm_core import LBMCavity
from lbm_periodic import KolmogorovCase
from lbm_couette import CouetteCase
from lbm_voxel import VoxelCase, build_cylinder_mask
from paper_extra_benchmarks import solve_baseline_generic
from paper_faithful_baselines import (
    solve_dual_time_mg,
    solve_inexact_newton_ne,
    solve_preconditioned_lbm,
)
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_scmk import solve_baseline_periodic


OUT = Path("paper_revision_data/reference_solver_production")
HIST_DIR = OUT / "histories"

MULTI_CYLINDER_RADIUS = 1.0 / 12.0
MULTI_CYLINDER_CENTERS = (
    (0.1875, 0.140625),
    (0.40625, 0.171875),
    (0.265625, 0.453125),
    (0.6875, 0.5),
    (0.765625, 0.203125),
    (0.796875, 0.609375),
)


def make_multi_cylinder_mask(n):
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    chi = np.ones((n, n), dtype=np.float64)
    r2_limit = MULTI_CYLINDER_RADIUS * MULTI_CYLINDER_RADIUS
    for cx, cy in MULTI_CYLINDER_CENTERS:
        chi[(xx - cx) ** 2 + (yy - cy) ** 2 < r2_limit] = 0.0
    return chi


def residual_norm(case, f):
    r = case.residual(f)
    chi = getattr(case, "chi", None)
    if chi is not None:
        fluid = chi > 0.0
        return float(np.sqrt(np.mean(r[:, fluid] * r[:, fluid])))
    return float(case._fast_norm(r) / math.sqrt(case.dof))


def write_history(path, hist):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter_or_step", "residual", "lbe_calls", "wall_seconds"])
        for row in hist:
            wr.writerow([row[0], row[1], row[2], row[3] if len(row) > 3 else ""])


def baseline(case, max_steps, tol):
    if isinstance(case, LBMCavity):
        return solve_baseline(case, max_steps=max_steps, tol=tol, check_every=20, verbose=False)
    if isinstance(case, VoxelCase):
        return solve_baseline_generic(case, max_steps=max_steps, tol=tol, check_every=20, verbose=False)
    return solve_baseline_periodic(case, max_steps=max_steps, tol=tol, check_every=20, verbose=False)


def cases():
    return {
        "kolmogorov_n16": lambda: KolmogorovCase(N=16, nu=0.05, F0=2e-4, kf=1),
        "channel_n16": lambda: ChannelCase(N=16, nu=0.05, F0=1e-5),
        "cavity_re100_n17": lambda: LBMCavity(N=17, Re=100, U_wall=0.1),
        "voxel_cylinder_n16": lambda: VoxelCase(build_cylinder_mask(16, 8, 8, 3), nu=0.05, F0=1e-5, kf=0),
    }


def group_a_cases():
    return {
        "kolmogorov_n32": lambda: KolmogorovCase(N=32, nu=0.05, F0=2e-4, kf=1),
        "channel_n32": lambda: ChannelCase(N=32, nu=0.05, F0=1e-5),
        "couette_n32": lambda: CouetteCase(N=32, nu=0.05, U_wall=0.05),
        "cavity_re100_n33": lambda: LBMCavity(N=33, Re=100, U_wall=0.1),
        "cavity_re400_n49": lambda: LBMCavity(N=49, Re=400, U_wall=0.1),
        "multi_cylinder_n32": lambda: VoxelCase(make_multi_cylinder_mask(32), nu=0.05, F0=2e-4, kf=1),
    }


def run_method(method, case, tol, budget):
    max_steps = budget
    max_outer = max(8, budget // 80)
    if method == "picard_lbm":
        return baseline(case, max_steps=max_steps, tol=tol)
    if method == "anderson_lbm":
        return solve_anderson(case, max_iter=max_outer * 4, tol=tol, m=5, beta=0.8,
                              safeguard=True, verbose=False, check_every=5)
    if method == "preconditioned_lbm":
        return solve_preconditioned_lbm(case, max_steps=max_steps, tol=tol, check_every=20,
                                        gamma=0.5, verbose=False)
    if method == "inexact_newton_lbe":
        return solve_inexact_newton_ne(case, max_outer=max_outer, tol=tol, krylov_max=4,
                                       krylov_tol=1e-3, K_ne=4, K_smooth=2,
                                       line_search_max=4, verbose=False)
    if method == "dual_time_mg_lbm":
        return solve_dual_time_mg(case, max_outer=max_outer, tol=tol, K_pre=2,
                                  K_coarse=10, K_post=2, max_levels=4,
                                  cycle="W", lambda_weight=0.7, verbose=False)
    raise ValueError(method)


def method_stats(method):
    if method == "anderson_lbm":
        return getattr(solve_anderson, "last_stats", {})
    if method == "preconditioned_lbm":
        return getattr(solve_preconditioned_lbm, "last_stats", {})
    if method == "inexact_newton_lbe":
        return getattr(solve_inexact_newton_ne, "last_stats", {})
    if method == "dual_time_mg_lbm":
        return getattr(solve_dual_time_mg, "last_stats", {})
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "groupA"], default="smoke")
    ap.add_argument("--budget", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    methods = [
        "picard_lbm",
        "anderson_lbm",
        "preconditioned_lbm",
        "inexact_newton_lbe",
        "dual_time_mg_lbm",
    ]
    rows = []
    tol = 1e-7
    case_map = cases() if args.suite == "smoke" else group_a_cases()
    budget = args.budget if args.budget is not None else (600 if args.suite == "smoke" else 3000)
    for case_id, factory in case_map.items():
        for method in methods:
            case = factory()
            t0 = time.perf_counter()
            try:
                f, hist = run_method(method, case, tol, budget)
                wall = time.perf_counter() - t0
                native_res = residual_norm(case, f)
                method_res = hist[-1][1] if hist else native_res
                status = "ok" if np.isfinite(method_res) and np.isfinite(native_res) else "nonfinite"
                err = ""
            except Exception as exc:
                hist = []
                wall = time.perf_counter() - t0
                native_res = float("nan")
                method_res = float("nan")
                status = "crashed"
                err = repr(exc)
            if hist:
                write_history(HIST_DIR / f"{case_id}__{method}.csv", hist)
                lbe = hist[-1][2]
            else:
                lbe = 0
            rows.append({
                "suite": args.suite,
                "case": case_id,
                "method": method,
                "status": status,
                "method_operator_residual": method_res,
                "native_bgk_residual": native_res,
                "lbe_calls": lbe,
                "wall_seconds": wall,
                "method_stats_json": json.dumps(method_stats(method), sort_keys=True),
                "error": err,
            })

    csv_path = OUT / f"reference_only_{args.suite}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    json_path = OUT / f"reference_only_{args.suite}.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    crashed = [r for r in rows if r["status"] != "ok"]
    print(json.dumps({
        "suite": args.suite,
        "cases": len(case_map),
        "methods": len(methods),
        "runs": len(rows),
        "crashed": len(crashed),
        "csv": str(csv_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
