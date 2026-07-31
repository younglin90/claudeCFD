"""Strict 1x/2x/3x scaling comparison for the single proposed LBM solver.

This extends ``verify_fixed10_strict.py`` from the base 10 validation cases to
three mesh levels.  The pass gate is unchanged: at every case/mesh level the
proposed solver must converge, use no more LBE calls than the best converged
fixed method, be no slower in wall-clock time, and be at least as accurate as
the best converged non-Picard fixed method against the Picard reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

try:
    import numba

    numba.set_num_threads(24)
except Exception:
    numba = None

from lbm_channel import ChannelCase
from lbm_core import LBMCavity
from lbm_couette import CouetteCase
from lbm_periodic import KolmogorovCase
from lbm_voxel import VoxelCase
from paper_60case_benchmark import (
    CASE_IDS as BASE_CASE_IDS,
    METHODS,
    make_multi_cylinder_mask,
    run_method,
    velocity_error,
)
from paper_extra_benchmarks import backward_step_mask, cylinder_wake_mask, t_junction_mask
from solver_proposed_single import solve_proposed_single


OUT = Path("paper_revision_data") / "fixed30_scaling_strict"
CACHE = OUT / "npz_cache"
HIST = OUT / "histories"
FIXED_METHODS = [m for m in METHODS if m != "proposed"]
FORBIDDEN_TOKENS = (
    "analytical_ux",
    "_analytic_equilibrium",
    "target_converged",
    "target-deflated",
    "target deflated",
    "solve_anderson(",
    "low_amplitude_unmasked",
)


BASE_META = {
    "kolmogorov_n32": ("Kolmogorov flow", 32, 1e-7),
    "channel_n32": ("Plane Poiseuille channel", 32, 1e-7),
    "couette_n32": ("Couette flow", 32, 1e-7),
    "cavity_re100_n33": ("Lid-driven cavity Re=100", 33, 5e-7),
    "cavity_re400_n49": ("Lid-driven cavity Re=400", 49, 5e-7),
    "cavity_re1000_n129": ("Lid-driven cavity Re=1000", 129, 5e-7),
    "multi_cylinder_n32": ("Multi-cylinder voxel mask", 32, 1e-7),
    "backward_step_n64": ("Backward-facing step mask", 64, 1e-7),
    "cylinder_wake_n64": ("Cylinder wake analogue", 64, 1e-7),
    "t_junction_n64": ("T-junction mask", 64, 1e-7),
}


def scaled_n(base_n: int, level: int, odd_refinement: bool = False) -> int:
    if level == 1:
        return base_n
    if odd_refinement:
        return level * (base_n - 1) + 1
    return level * base_n


def _force_scale(level: int) -> float:
    """Diffusive refinement scaling for fixed physical forcing/Re.

    With a fixed physical domain and dx ~ 1/level, keeping lattice viscosity
    fixed implies U_lattice ~ dx and body-force acceleration ~ dx^3.  This
    prevents the 2x/3x cases from becoming higher-Re/Mach stress tests.
    """
    return 1.0 / float(level ** 3)


def _velocity_scale(level: int) -> float:
    """Velocity scaling for wall-driven fixed-Re refinement."""
    return 1.0 / float(level)


def _tol_scale(level: int) -> float:
    """Residual tolerance scaling for fixed-physics refinement.

    Absolute native residuals shrink when lattice velocities are scaled down.
    Scaling tol with the lattice velocity avoids classifying the high-level
    zero/rest state as converged while keeping comparable relative accuracy.
    """
    return _velocity_scale(level)


def make_t_junction_case(n: int, level: int = 1) -> VoxelCase:
    case = VoxelCase(t_junction_mask(n), nu=0.05, F0=0.0, kf=0)
    case.Fx = np.zeros((n, n), dtype=np.float64)
    case.Fy = np.zeros((n, n), dtype=np.float64)
    fluid = case.chi > 0
    half_width = 5.5 / 64.0
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    vertical = (yy <= 0.5) & (np.abs(xx - 0.5) <= half_width)
    scale = _force_scale(level)
    case.Fx[fluid] = 8.0e-6 * scale
    case.Fy[vertical & fluid] = -8.0e-6 * scale
    case.Fx *= case.chi
    case.Fy *= case.chi
    return case


def case_factory_scaled(base_id: str, level: int):
    if base_id not in BASE_META:
        raise ValueError(base_id)
    label0, base_n, tol = BASE_META[base_id]
    tol = tol * _tol_scale(level)
    if base_id == "kolmogorov_n32":
        n = scaled_n(base_n, level)
        f0 = 2e-4 * _force_scale(level)
        factory = lambda n=n, f0=f0: KolmogorovCase(N=n, nu=0.05, F0=f0, kf=1)
    elif base_id == "channel_n32":
        n = scaled_n(base_n, level)
        f0 = 1e-5 * _force_scale(level)
        factory = lambda n=n, f0=f0: ChannelCase(N=n, nu=0.05, F0=f0)
    elif base_id == "couette_n32":
        n = scaled_n(base_n, level)
        u_wall = 0.05 * _velocity_scale(level)
        factory = lambda n=n, u_wall=u_wall: CouetteCase(N=n, nu=0.05, U_wall=u_wall)
    elif base_id == "cavity_re100_n33":
        n = scaled_n(base_n, level, odd_refinement=True)
        u_wall = 0.1 * _velocity_scale(level)
        factory = lambda n=n, u_wall=u_wall: LBMCavity(N=n, Re=100, U_wall=u_wall)
    elif base_id == "cavity_re400_n49":
        n = scaled_n(base_n, level, odd_refinement=True)
        u_wall = 0.1 * _velocity_scale(level)
        factory = lambda n=n, u_wall=u_wall: LBMCavity(N=n, Re=400, U_wall=u_wall)
    elif base_id == "cavity_re1000_n129":
        n = scaled_n(base_n, level, odd_refinement=True)
        u_wall = 0.1 * _velocity_scale(level)
        factory = lambda n=n, u_wall=u_wall: LBMCavity(N=n, Re=1000, U_wall=u_wall)
    elif base_id == "multi_cylinder_n32":
        n = scaled_n(base_n, level)
        f0 = 2e-4 * _force_scale(level)
        factory = lambda n=n, f0=f0: VoxelCase(make_multi_cylinder_mask(n), nu=0.05, F0=f0, kf=1)
    elif base_id == "backward_step_n64":
        n = scaled_n(base_n, level)
        f0 = 1.5e-5 * _force_scale(level)
        factory = lambda n=n, f0=f0: VoxelCase(backward_step_mask(n), nu=0.05, F0=f0, kf=0)
    elif base_id == "cylinder_wake_n64":
        n = scaled_n(base_n, level)
        f0 = 1.0e-5 * _force_scale(level)
        factory = lambda n=n, f0=f0: VoxelCase(cylinder_wake_mask(n), nu=0.04, F0=f0, kf=0)
    elif base_id == "t_junction_n64":
        n = scaled_n(base_n, level)
        factory = lambda n=n, level=level: make_t_junction_case(n, level=level)
    else:
        raise ValueError(base_id)
    case_id = f"{base_id}__{level}x"
    label = f"{label0} N={n} ({level}x)"
    return case_id, label, tol, factory


def max_steps_for_scaled(base_id: str, level: int) -> int:
    base = 900000 if base_id == "cavity_re1000_n129" else 250000 if "cavity" in base_id else 90000 if base_id in {"backward_step_n64", "cylinder_wake_n64"} else 70000
    scale = level * level
    cap = 1600000 if base_id == "cavity_re1000_n129" else 600000 if "cavity" in base_id else 300000
    return int(min(cap, base * scale))


def _hist_to_list(hist):
    return [[int(a), float(b), int(c), float(d)] for a, b, c, d in hist]


def _write_history(case_id: str, method: str, hist):
    HIST.mkdir(parents=True, exist_ok=True)
    path = HIST / f"{case_id}__{method}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iter", "residual", "lbe_calls", "wall_seconds"])
        for row in hist:
            wr.writerow(row)


def _cache_key(method: str) -> str:
    files = [
        Path("verify_fixed30_scaling_strict.py"),
        Path("paper_60case_benchmark.py"),
        Path("paper_extra_benchmarks.py"),
    ]
    if method == "proposed":
        files += [
            Path("solver_proposed_single.py"),
            Path("solver_safe_nn.py"),
            Path("solver_unified_safe_nn.py"),
        ]
    else:
        files += [
            Path("paper_faithful_baselines.py"),
            Path("solver_anderson.py"),
            Path("solver_baseline.py"),
        ]
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    for path in files:
        if path.exists():
            h.update(path.as_posix().encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _cache_path(case_id: str, method: str) -> Path:
    return CACHE / f"{case_id}__{method}__{_cache_key(method)}.npz"


def _load_cached(case_id: str, method: str):
    path = _cache_path(case_id, method)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    f = data["f"]
    hist = [tuple(row) for row in data["hist"].tolist()]
    wall = float(data["wall"])
    return f, hist, wall


def _save_cached(case_id: str, method: str, f, hist, wall: float):
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _cache_path(case_id, method),
        f=np.asarray(f),
        hist=np.asarray(_hist_to_list(hist), dtype=np.float64),
        wall=float(wall),
    )


def run_one(case_id: str, base_id: str, level: int, method: str, tol: float, factory, use_cache=True):
    cached = _load_cached(case_id, method) if use_cache else None
    case = factory()
    if cached is not None:
        f, hist, wall = cached
        return case, f, hist, wall

    t0 = time.perf_counter()
    if method == "proposed":
        f, hist = solve_proposed_single(case, tol=tol, verbose=False)
        wall = time.perf_counter() - t0
    else:
        f, hist, wall = run_method(method, case, tol, max_steps_for_scaled(base_id, level), verbose=False)
    _save_cached(case_id, method, f, hist, wall)
    _write_history(case_id, method, hist)
    return case, f, hist, wall


def row_for(base_id, level, case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall):
    final_res = float(hist[-1][1]) if hist else float("inf")
    lbe = int(hist[-1][2]) if hist else 0
    fluid = getattr(ref_case, "chi", np.ones((ref_case.N, ref_case.N), dtype=np.float64)) > 0
    err = velocity_error(ref_case, ref_f, case, f, fluid_mask=fluid)
    converged = bool(np.isfinite(final_res) and final_res < 5.0 * tol)
    return {
        "base_case_id": base_id,
        "scaling_level": int(level),
        "case_id": case_id,
        "case_label": label,
        "method": method,
        "tol": float(tol),
        "N": int(case.N),
        "lbe_calls": int(lbe),
        "wall_seconds": float(wall),
        "final_residual": float(final_res),
        "converged": int(converged),
        "rel_l2_vs_picard": float(err["rel_l2"]),
        "linf_vs_picard": float(err["linf"]),
        "rms_vs_picard": float(err["rms"]),
    }


def anti_cheat_pass() -> tuple[bool, list[str]]:
    text = Path("solver_proposed_single.py").read_text(encoding="utf-8")
    hits = [token for token in FORBIDDEN_TOKENS if token in text]
    return (len(hits) == 0), hits


def score(rows, selected_case_ids):
    by_case = {case_id: [r for r in rows if r["case_id"] == case_id] for case_id in selected_case_ids}
    case_results = []
    for case_id, case_rows in by_case.items():
        prop = next((r for r in case_rows if r["method"] == "proposed"), None)
        if prop is None:
            continue
        fixed = [r for r in case_rows if r["method"] != "proposed"]
        eligible = [
            r for r in fixed
            if r["converged"] and np.isfinite(r["final_residual"]) and r["lbe_calls"] > 0 and r["wall_seconds"] > 0
        ]
        if not eligible:
            eligible = fixed
        acc_eligible = [r for r in eligible if r["method"] != "picard_lbm" and np.isfinite(r["rel_l2_vs_picard"])]
        if not acc_eligible:
            acc_eligible = [r for r in eligible if np.isfinite(r["rel_l2_vs_picard"])]
        best_lbe = min((r["lbe_calls"] for r in eligible if r["lbe_calls"] > 0), default=10**18)
        best_wall = min((r["wall_seconds"] for r in eligible if r["wall_seconds"] > 0), default=float("inf"))
        best_acc = min((r["rel_l2_vs_picard"] for r in acc_eligible), default=float("inf"))
        lbe_win = bool(prop["lbe_calls"] <= best_lbe)
        wall_win = bool(prop["wall_seconds"] <= best_wall)
        acc_win = bool(prop["rel_l2_vs_picard"] <= best_acc * 1.001 + 1e-12)
        conv = bool(prop["converged"])
        case_pass = bool(conv and lbe_win and wall_win and acc_win)
        case_results.append({
            "case_id": case_id,
            "base_case_id": prop["base_case_id"],
            "scaling_level": prop["scaling_level"],
            "case_pass": int(case_pass),
            "converged": int(conv),
            "lbe_win": int(lbe_win),
            "wall_win": int(wall_win),
            "accuracy_win": int(acc_win),
            "proposed_lbe": prop["lbe_calls"],
            "best_fixed_lbe": int(best_lbe) if best_lbe < 10**18 else None,
            "proposed_wall": prop["wall_seconds"],
            "best_fixed_wall": best_wall,
            "proposed_rel_l2": prop["rel_l2_vs_picard"],
            "best_fixed_rel_l2": best_acc,
        })
    anti_ok, anti_hits = anti_cheat_pass()
    pass_count = sum(c["case_pass"] for c in case_results)
    lbe_wins = sum(c["lbe_win"] for c in case_results)
    wall_wins = sum(c["wall_win"] for c in case_results)
    acc_wins = sum(c["accuracy_win"] for c in case_results)
    convs = sum(c["converged"] for c in case_results)
    speedups = [
        c["best_fixed_lbe"] / max(c["proposed_lbe"], 1)
        for c in case_results
        if c["best_fixed_lbe"] is not None and c["proposed_lbe"] > 0
    ]
    score_value = (
        100.0 * pass_count
        + 10.0 * lbe_wins
        + 10.0 * wall_wins
        + 10.0 * acc_wins
        + 5.0 * convs
        + float(np.mean(speedups) if speedups else 0.0)
        - (0 if anti_ok else 200.0)
    )
    return {
        "score": float(score_value),
        "all_pass": int(bool(case_results) and pass_count == len(case_results) and anti_ok),
        "case_count": len(case_results),
        "pass_count": int(pass_count),
        "converged_count": int(convs),
        "lbe_win_count": int(lbe_wins),
        "wall_win_count": int(wall_wins),
        "accuracy_win_count": int(acc_wins),
        "mean_lbe_speedup_vs_best_fixed": float(np.mean(speedups) if speedups else 0.0),
        "anti_cheat_pass": int(anti_ok),
        "anti_cheat_hits": anti_hits,
        "case_results": case_results,
    }


def write_outputs(rows, metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_case_id",
        "scaling_level",
        "case_id",
        "case_label",
        "method",
        "tol",
        "N",
        "lbe_calls",
        "wall_seconds",
        "final_residual",
        "converged",
        "rel_l2_vs_picard",
        "linf_vs_picard",
        "rms_vs_picard",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row[k] for k in fields})
    (OUT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_csv_arg(value: str, allowed):
    if value == "all":
        return list(allowed)
    parsed = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in parsed if x not in allowed]
    if bad:
        raise ValueError(f"unknown values: {bad}")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,3", help="comma-separated scaling levels, or all")
    parser.add_argument("--base-cases", default="all", help="comma-separated base case ids, or all")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method ids")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="run only fixed/reference methods; do not auto-append proposed",
    )
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()] if args.levels != "all" else [1, 2, 3]
    for level in levels:
        if level not in {1, 2, 3}:
            raise ValueError("levels must be 1, 2, or 3")
    base_ids = parse_csv_arg(args.base_cases, BASE_CASE_IDS)
    methods = parse_csv_arg(args.methods, METHODS)
    if "picard_lbm" not in methods:
        methods = ["picard_lbm"] + methods
    if args.reference_only:
        methods = [m for m in methods if m != "proposed"]
    elif "proposed" not in methods:
        methods = methods + ["proposed"]

    started = time.perf_counter()
    rows = []
    selected_case_ids = []
    for base_id in base_ids:
        for level in levels:
            case_id, label, tol, factory = case_factory_scaled(base_id, level)
            selected_case_ids.append(case_id)
            print(f"[case] {case_id}: {label}", flush=True)
            ref_case, ref_f, ref_hist, ref_wall = run_one(
                case_id, base_id, level, "picard_lbm", tol, factory, use_cache=not args.no_cache
            )
            for method in methods:
                if method == "picard_lbm":
                    case, f, hist, wall = ref_case, ref_f, ref_hist, ref_wall
                else:
                    try:
                        case, f, hist, wall = run_one(
                            case_id, base_id, level, method, tol, factory, use_cache=not args.no_cache
                        )
                    except Exception as exc:
                        print(f"  {method} crashed: {exc}", flush=True)
                        case = factory()
                        f = case.initial_field()
                        hist = [(0, float("inf"), 0, 0.0)]
                        wall = 0.0
                row = row_for(base_id, level, case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall)
                rows.append(row)
                print(
                    f"  {method:22s} lbe={row['lbe_calls']:8d} wall={row['wall_seconds']:8.3f} "
                    f"res={row['final_residual']:.3e} rel={row['rel_l2_vs_picard']:.3e} conv={row['converged']}",
                    flush=True,
                )
    metrics = score(rows, selected_case_ids)
    metrics["elapsed_wall_seconds"] = float(time.perf_counter() - started)
    metrics["levels"] = levels
    metrics["base_cases"] = base_ids
    metrics["physics_scaling"] = {
        "force_driven": "F_lattice(level) = F_lattice(1) / level^3",
        "wall_driven": "U_wall(level) = U_wall(1) / level",
        "residual_tolerance": "tol(level) = tol(1) / level",
        "cavity_viscosity": "LBMCavity uses nu = U_wall * (N - 1) / Re, so odd refinement plus U_wall/level keeps Re and nu fixed.",
        "initial_condition": "Native case initial fields are regenerated after applying scaled forcing/wall velocity; wall-driven initial lid rows therefore scale with U_wall.",
    }
    write_outputs(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}", flush=True)
    print(f"[saved] {OUT / 'metrics.json'}", flush=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
