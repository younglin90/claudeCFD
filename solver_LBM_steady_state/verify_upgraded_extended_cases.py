"""Extended verification for the current public solve_scmk dispatcher.

This script checks whether the upgraded solver remains strong outside the
five-case verify_metric.py target.  It intentionally excludes 3D cases.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "32")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

try:
    import numba

    numba.set_num_threads(32)
except Exception:
    numba = None

from lbm_core import LBMCavity
from lbm_periodic import KolmogorovCase, build_spectral_schur
from lbm_channel import ChannelCase
from lbm_couette import CouetteCase
from lbm_voxel import VoxelCase
from lbm_optimized_2d import (
    OptimizedCavityCase,
    OptimizedChannelCase,
    OptimizedKolmogorovCase,
    solve_baseline_fast,
)
from paper_extra_benchmarks import (
    backward_step_mask,
    cylinder_wake_mask,
    solve_baseline_generic,
    t_junction_mask,
)
from solver_baseline import solve_baseline
from solver_scmk import solve_baseline_periodic, solve_scmk


OUT = Path("paper_revision_data")
JSON_OUT = OUT / "upgraded_extended_verification_2026-05-23.json"
MD_OUT = OUT / "upgraded_extended_verification_2026-05-23.md"


def residual_norm(case, f) -> float:
    r = case.residual(f)
    return float(case._fast_norm(r) / math.sqrt(case.dof))


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    from lbm_core import moments

    return moments(f)


def velocity_metrics(ref_case, f_ref, test_case, f_test, fluid_mask=None):
    _, ux_ref, uy_ref = macro_of(ref_case, f_ref)
    _, ux, uy = macro_of(test_case, f_test)
    if fluid_mask is None:
        fluid_mask = np.ones_like(ux_ref, dtype=bool)
    du = ux[fluid_mask] - ux_ref[fluid_mask]
    dv = uy[fluid_mask] - uy_ref[fluid_mask]
    ref = np.sqrt(ux_ref[fluid_mask] ** 2 + uy_ref[fluid_mask] ** 2)
    den = max(float(np.sqrt(np.sum(ref * ref))), 1e-30)
    rel_l2 = float(np.sqrt(np.sum(du * du + dv * dv)) / den)
    linf = float(max(np.max(np.abs(du)), np.max(np.abs(dv))))
    return {"rel_l2": rel_l2, "linf": linf}


def run_solver(case, tol, max_outer, kinetic_substeps, krylov_max=10):
    s_inv = build_spectral_schur(case.N, omega=case.omega, mode="ap")
    t0 = time.perf_counter()
    f, hist = solve_scmk(
        case,
        s_inv,
        max_outer=max_outer,
        tol=tol,
        krylov_max=krylov_max,
        krylov_tol=1e-3,
        line_search_max=5,
        kinetic_substeps=kinetic_substeps,
        verbose=False,
    )
    wall = time.perf_counter() - t0
    return f, hist, wall


def classify(speedup, rel_l2, residual, tol):
    if not (np.isfinite(speedup) and np.isfinite(rel_l2) and np.isfinite(residual)):
        return "fail_nonfinite"
    if residual >= 5.0 * tol:
        return "fail_residual"
    if rel_l2 > 0.10:
        return "weak_accuracy"
    if speedup < 1.5:
        return "weak_speedup"
    if speedup < 5.0 or rel_l2 > 0.03:
        return "borderline"
    return "good"


def scaling_forcing(n, kind):
    target_u = 0.05
    nu = 0.05
    if kind == "kolmogorov":
        k_lat = 2.0 * math.pi / n
        return nu, target_u * nu * k_lat * k_lat
    if kind == "channel":
        return nu, 8.0 * nu * target_u / ((n - 1.0) ** 2)
    raise ValueError(kind)


def run_periodic_scaling(kind, n, tol=1e-9):
    nu, f0 = scaling_forcing(n, kind)
    label = f"{kind.capitalize()} N={n}"
    print(f"[case] {label}", flush=True)
    if kind == "kolmogorov":
        baseline_case = OptimizedKolmogorovCase(N=n, nu=nu, F0=f0, kf=1)
        solver_case = KolmogorovCase(N=n, nu=nu, F0=f0, kf=1)
    elif kind == "channel":
        baseline_case = OptimizedChannelCase(N=n, nu=nu, F0=f0)
        solver_case = ChannelCase(N=n, nu=nu, F0=f0)
    else:
        raise ValueError(kind)

    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(
        baseline_case,
        max_steps=900000,
        tol=tol,
        check_every=1000 if n <= 128 else 2000,
        verbose=False,
    )
    wall_b = time.perf_counter() - t0
    f_s, h_s, wall_s = run_solver(
        solver_case,
        tol=tol,
        max_outer=220 if n <= 128 else 300,
        kinetic_substeps=15,
        krylov_max=10,
    )
    vm = velocity_metrics(baseline_case, f_b, solver_case, f_s)
    speedup = float(h_b[-1][2] / max(h_s[-1][2], 1))
    return {
        "group": "scaling",
        "label": label,
        "N": n,
        "tol": tol,
        "baseline_lbe": int(h_b[-1][2]),
        "solver_lbe": int(h_s[-1][2]),
        "baseline_residual": float(h_b[-1][1]),
        "solver_residual": float(h_s[-1][1]),
        "baseline_wall": float(wall_b),
        "solver_wall": float(wall_s),
        "lbe_speedup": speedup,
        "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
        "velocity_metrics": vm,
        "status": classify(speedup, vm["rel_l2"], float(h_s[-1][1]), tol),
    }


def run_cavity(re, n, tol=5e-7):
    label = f"Cavity Re={re} N={n}"
    print(f"[case] {label}", flush=True)
    baseline_case = OptimizedCavityCase(N=n, Re=re, U_wall=0.1)
    solver_case = LBMCavity(N=n, Re=re, U_wall=0.1)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_fast(
        baseline_case,
        max_steps=500000 if re <= 400 else 900000,
        tol=tol,
        check_every=500 if re <= 400 else 1000,
        verbose=False,
    )
    wall_b = time.perf_counter() - t0
    f_s, h_s, wall_s = run_solver(
        solver_case,
        tol=tol,
        max_outer=360 if re <= 400 else 480,
        kinetic_substeps=20 if re >= 1000 else 15,
        krylov_max=10,
    )
    vm = velocity_metrics(baseline_case, f_b, solver_case, f_s)
    speedup = float(h_b[-1][2] / max(h_s[-1][2], 1))
    return {
        "group": "cavity",
        "label": label,
        "N": n,
        "Re": re,
        "nu": float(solver_case.nu),
        "omega": float(solver_case.omega),
        "tol": tol,
        "baseline_lbe": int(h_b[-1][2]),
        "solver_lbe": int(h_s[-1][2]),
        "baseline_residual": float(h_b[-1][1]),
        "solver_residual": float(h_s[-1][1]),
        "baseline_wall": float(wall_b),
        "solver_wall": float(wall_s),
        "lbe_speedup": speedup,
        "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
        "velocity_metrics": vm,
        "status": classify(speedup, vm["rel_l2"], float(h_s[-1][1]), tol),
    }


def make_mask_case(name):
    n = 64
    if name == "backward_step":
        chi = backward_step_mask(n)
        return "Backward-facing step N=64", VoxelCase(chi, nu=0.05, F0=1.5e-5, kf=0)
    if name == "cylinder_wake":
        chi = cylinder_wake_mask(n)
        return "Cylinder wake analogue N=64", VoxelCase(chi, nu=0.04, F0=1.0e-5, kf=0)
    if name == "t_junction":
        chi = t_junction_mask(n)
        case = VoxelCase(chi, nu=0.05, F0=0.0, kf=0)
        case.Fx = np.zeros((n, n), dtype=np.float64)
        case.Fy = np.zeros((n, n), dtype=np.float64)
        fluid = case.chi > 0
        case.Fx[fluid] = 8.0e-6
        case.Fy[: n // 2, n // 2 - 5 : n // 2 + 6] = -8.0e-6
        case.Fx *= case.chi
        case.Fy *= case.chi
        return "T-junction N=64", case
    raise ValueError(name)


def run_mask(name, tol=1e-7):
    label, case_b = make_mask_case(name)
    _, case_s = make_mask_case(name)
    print(f"[case] {label}", flush=True)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_generic(
        case_b, max_steps=70000, tol=tol, check_every=500, verbose=False
    )
    wall_b = time.perf_counter() - t0
    f_s, h_s, wall_s = run_solver(
        case_s, tol=tol, max_outer=260, kinetic_substeps=15, krylov_max=10
    )
    vm = velocity_metrics(case_b, f_b, case_s, f_s, fluid_mask=case_b.chi > 0)
    speedup = float(h_b[-1][2] / max(h_s[-1][2], 1))
    return {
        "group": "extra_masks",
        "label": label,
        "N": case_b.N,
        "tol": tol,
        "fluid_fraction": float(case_b.fluid_fraction),
        "baseline_lbe": int(h_b[-1][2]),
        "solver_lbe": int(h_s[-1][2]),
        "baseline_residual": float(h_b[-1][1]),
        "solver_residual": float(h_s[-1][1]),
        "baseline_wall": float(wall_b),
        "solver_wall": float(wall_s),
        "lbe_speedup": speedup,
        "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
        "velocity_metrics": vm,
        "status": classify(speedup, vm["rel_l2"], float(h_s[-1][1]), tol),
    }


def run_couette_64(tol=1e-7):
    label = "Couette N=64"
    print(f"[case] {label}", flush=True)
    case_b = CouetteCase(N=64, nu=0.05, U_wall=0.05)
    case_s = CouetteCase(N=64, nu=0.05, U_wall=0.05)
    t0 = time.perf_counter()
    f_b, h_b = solve_baseline_periodic(
        case_b, max_steps=100000, tol=tol, check_every=500, verbose=False
    )
    wall_b = time.perf_counter() - t0
    f_s, h_s, wall_s = run_solver(
        case_s, tol=tol, max_outer=220, kinetic_substeps=15, krylov_max=10
    )
    vm = velocity_metrics(case_b, f_b, case_s, f_s)
    speedup = float(h_b[-1][2] / max(h_s[-1][2], 1))
    return {
        "group": "wall_flow",
        "label": label,
        "N": 64,
        "tol": tol,
        "baseline_lbe": int(h_b[-1][2]),
        "solver_lbe": int(h_s[-1][2]),
        "baseline_residual": float(h_b[-1][1]),
        "solver_residual": float(h_s[-1][1]),
        "baseline_wall": float(wall_b),
        "solver_wall": float(wall_s),
        "lbe_speedup": speedup,
        "wall_speedup": float(wall_b / max(wall_s, 1e-12)),
        "velocity_metrics": vm,
        "status": classify(speedup, vm["rel_l2"], float(h_s[-1][1]), tol),
    }


def case_plan(mode):
    cases = [
        ("fn", run_periodic_scaling, ("kolmogorov", 64)),
        ("fn", run_periodic_scaling, ("channel", 64)),
        ("fn", run_couette_64, ()),
        ("fn", run_cavity, (400, 65)),
        ("fn", run_mask, ("backward_step",)),
        ("fn", run_mask, ("cylinder_wake",)),
        ("fn", run_mask, ("t_junction",)),
    ]
    if mode in {"balanced", "full"}:
        cases[2:2] = [
            ("fn", run_periodic_scaling, ("kolmogorov", 128)),
            ("fn", run_periodic_scaling, ("channel", 128)),
        ]
    if mode == "full":
        cases[4:4] = [
            ("fn", run_periodic_scaling, ("kolmogorov", 256)),
            ("fn", run_periodic_scaling, ("channel", 256)),
        ]
        cases.append(("fn", run_cavity, (1000, 129)))
    return cases


def write_md(payload):
    rows = payload["results"]
    lines = [
        "# Upgraded Solver Extended Verification",
        "",
        "This check uses the current public `solve_scmk` dispatcher. 3D cases are excluded. Thread caps: NUMBA/OMP 32, BLAS 1.",
        "",
        "| Group | Case | Picard LBE | Solver LBE | LBE x | wall x | Solver residual | rel L2 | Linf | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        vm = r["velocity_metrics"]
        lines.append(
            f"| {r['group']} | {r['label']} | {r['baseline_lbe']} | {r['solver_lbe']} | "
            f"{r['lbe_speedup']:.2f} | {r['wall_speedup']:.2f} | {r['solver_residual']:.3e} | "
            f"{vm['rel_l2']:.3e} | {vm['linf']:.3e} | {r['status']} |"
        )
    status_counts = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    lines += [
        "",
        "## Status counts",
        "",
        json.dumps(status_counts, indent=2),
        "",
        "## Interpretation",
        "",
        "- `good`: residual converged, rel L2 <= 3%, and LBE speedup >= 5x.",
        "- `borderline`: converged but either speedup < 5x or rel L2 > 3%.",
        "- `weak_accuracy`: converged residual but rel L2 > 10%; should not be used as a strong accuracy claim.",
        "- `weak_speedup`: converged but speedup < 1.5x.",
        "- `fail_residual`: final residual is outside 5x tolerance.",
        "- `fail_nonfinite`: non-finite residual or velocity metric occurred.",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["quick", "balanced", "full"],
        default="balanced",
        help="quick excludes N=128/256 and Re=1000; full includes N=256 and Re=1000.",
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    results = []
    started = time.perf_counter()
    for _, fn, fn_args in case_plan(args.mode):
        results.append(fn(*fn_args))
        last = results[-1]
        print(
            f"  -> {last['status']}: {last['lbe_speedup']:.2f}x, "
            f"relL2={last['velocity_metrics']['rel_l2']:.3e}, "
            f"res={last['solver_residual']:.3e}",
            flush=True,
        )
    payload = {
        "mode": args.mode,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_wall": time.perf_counter() - started,
        "results": results,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_md(payload)
    print(f"[saved] {JSON_OUT}", flush=True)
    print(f"[saved] {MD_OUT}", flush=True)
    print(json.dumps({r["label"]: r["status"] for r in results}, indent=2))


if __name__ == "__main__":
    main()
