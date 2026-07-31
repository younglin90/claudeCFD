"""No-force-only 9-case benchmark: fixed-point + reference accelerators.

This mirrors ``paper_60case_benchmark.py`` but uses force-free inlet/outlet
cases from ``no_force_suite.no_force_cases``. It keeps baseline/anderson/
preconditioned/inexact-Newton/dual-time MG/proposed all on the same
single-case, single-parameter protocol.
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
from scipy.sparse.linalg import LinearOperator, gmres

from no_force_suite.no_force_cases import SUPPORTED_CASES as NO_FORCE_CASES, make_case as make_no_force_case
from paper_faithful_baselines import solve_dual_time_mg, solve_inexact_newton_ne, solve_preconditioned_lbm
from solver_anderson import solve_anderson
from solver_baseline import solve_baseline
from solver_proposed_single import solve_proposed_single
from lbm_core import moments as cavity_moments

OUT = Path("paper_revision_data") / "bench60_no_force"
HIST_DIR = OUT / "histories"
VTK_DIR = OUT / "vtk"
CACHE_DIR = OUT / "npz_cache"

CASE_IDS = tuple(NO_FORCE_CASES.keys())
METHODS = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]


def _fast_abs_mean(residual):
    return float(np.sqrt(np.mean(residual * residual)))


def macro_of(case, f):
    if hasattr(case, "macro"):
        return case.macro(f)
    if hasattr(case, "N") and hasattr(case, "dof"):
        return cavity_moments(f)
    raise RuntimeError("case has no macro method")


def _residual_norm(case, f):
    r = case.residual(f)
    return float(_fast_abs_mean(r))


def velocity_error(case_ref, f_ref, case, f, fluid_mask=None):
    _, ux_ref, uy_ref = macro_of(case_ref, f_ref)
    _, ux, uy = macro_of(case, f)
    if fluid_mask is None:
        fluid_mask = np.ones_like(ux_ref, dtype=bool)
    du = ux[fluid_mask] - ux_ref[fluid_mask]
    dv = uy[fluid_mask] - uy_ref[fluid_mask]
    ref = np.sqrt(ux_ref[fluid_mask] ** 2 + uy_ref[fluid_mask] ** 2)
    den = max(float(np.sqrt(np.sum(ref * ref))), 1.0e-30)
    return {
        "rel_l2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv)))),
        "rms": float(np.sqrt(np.mean(du * du + dv * dv))),
    }


def write_history_csv(path: Path, hist):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([
            "iter",
            "residual",
            "lbe_calls",
            "wall_seconds_raw",
            "wall_seconds",
            "wall_seconds_monotone",
            "accepted",
            "phase",
        ])
        for row in _normalize_history_wall_axes(hist):
            out = list(row)
            if len(out) >= 5:
                raw_wall = out[3]
                monotone_wall = out[4]
                out = out[:4] + [raw_wall, monotone_wall] + out[5:]
            wr.writerow(out)


def _normalize_history_wall_axes(hist):
    """Return rows with both raw and monotonic cumulative wall times.

    Some methods internally restart or switch sub-phases and emit iteration histories
    where `iter`/`wall_seconds` restart from zero. For transparency, we preserve
    raw wall-time values and also emit a normalized monotonic wall-time axis.
    """
    rows = [list(r) for r in hist if len(r) >= 4]
    if not rows:
        return rows

    spike_ratio = 2.0
    sanitized = []
    phase = 0
    segment_offset = 0.0
    first_raw = float(rows[0][3]) if _is_finite(rows[0][3]) else 0.0
    segment_base = first_raw
    prev_iter = int(rows[0][0]) if _is_finite(rows[0][0]) else -1
    prev_residual = float(rows[0][1]) if _is_finite(rows[0][1]) else float("inf")
    prev_wall_raw = first_raw
    prev_wall_cum = 0.0
    global_iter = 0

    first = [
        int(global_iter),
        float(rows[0][1]),
        int(rows[0][2]) if _is_finite(rows[0][2]) else 0,
        float(rows[0][3]) if _is_finite(rows[0][3]) else float("nan"),
        0.0,
        1,
        phase,
    ]
    sanitized.append(first)
    global_iter += 1

    eps_time = 1.0e-12

    for row in rows[1:]:
        it, res, lbe, wall_local = row[:4]
        it_v = int(it) if _is_finite(it) else it
        wall_v = float(wall_local) if _is_finite(wall_local) else wall_local
        res_v = float(res) if _is_finite(res) else float("nan")
        if not _is_finite(wall_v):
            continue
        is_reset = (
            _is_finite(prev_iter)
            and _is_finite(it_v)
            and (it_v < prev_iter or wall_v < prev_wall_raw - 1e-15)
        )
        if is_reset:
            # Start a new segment strictly after previous cumulative time so
            # phase boundaries cannot produce duplicate wall_seconds x-values.
            segment_offset = prev_wall_cum + eps_time
            segment_base = wall_v
            phase += 1

        wall_norm = wall_v - segment_base
        wall_norm = wall_norm if np.isfinite(wall_norm) else 0.0
        wall_norm = max(wall_norm, 0.0)
        wall_cum = segment_offset + wall_norm
        if wall_cum <= prev_wall_cum:
            wall_cum = prev_wall_cum + eps_time

        accepted = 1
        if _is_finite(prev_residual) and _is_finite(res_v) and prev_residual > 0.0:
            if res_v / prev_residual > spike_ratio:
                accepted = 0

        sanitized.append([
            int(global_iter),
            res_v,
            int(lbe) if _is_finite(lbe) else 0,
            float(wall_v),
            float(wall_cum),
            accepted,
            phase,
        ])
        global_iter += 1

        prev_iter = it_v
        prev_residual = res_v
        prev_wall_raw = wall_v
        prev_wall_cum = wall_cum

    return sanitized


def _is_finite(v):
    try:
        return float(v) == float(v)
    except Exception:
        return False


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


def run_inexact_newton(case, max_outer=160, tol=1e-7, krylov_max=10, krylov_tol=1e-3,
                     kinetic_substeps=8, verbose=False):
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
        if not np.isfinite(rn):
            break
        if rn < tol:
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


def run_method(method, case, tol, max_steps, verbose=False):
    t0 = time.perf_counter()
    if method == "picard_lbm":
        f, hist = solve_baseline(case, max_steps=max_steps, tol=tol, check_every=200 if case.N >= 64 else 100, verbose=verbose)
    elif method == "anderson_lbm":
        f, hist = solve_anderson(
            case,
            max_iter=max_steps // 2,
            tol=tol,
            m=5,
            beta=0.75,
            safeguard=True,
            verbose=verbose,
            check_every=5,
            max_backtracks=6,
            monotone_factor=0.995,
        )
    elif method == "preconditioned_lbm":
        budget = min(max_steps, 100000 if case.N < 64 else 160000)
        f, hist = solve_preconditioned_lbm(
            case,
            max_steps=budget,
            tol=tol,
            gamma=0.5,
            check_every=500 if case.N >= 64 else 200,
            verbose=verbose,
        )
    elif method == "inexact_newton_lbe":
        f, hist = run_inexact_newton(case, max_outer=180, tol=tol, krylov_max=10, krylov_tol=1e-3,
                                   kinetic_substeps=10, verbose=verbose)
    elif method == "dual_time_mg_lbm":
        f, hist = solve_dual_time_mg(
            case,
            max_outer=600,
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
        f, hist = solve_proposed_single(case, tol=tol, verbose=verbose)
    else:
        raise ValueError(method)
    return f, hist, time.perf_counter() - t0


def max_steps_for(case_id):
    if case_id == "cavity_re1000_n129":
        return 900000
    if "cavity" in case_id:
        return 250000
    if case_id in {"backward_step_n64", "cylinder_wake_n64", "t_junction_n64"}:
        return 100000
    return 70000


def _cache_key(method: str) -> str:
    paths = [
        Path("paper_60case_benchmark_no_force.py"),
        Path("paper_faithful_baselines.py"),
        Path("solver_anderson.py"),
        Path("solver_baseline.py"),
        Path("solver_unified_safe_nn.py"),
        Path("solver_proposed_single.py"),
        Path("no_force_suite/no_force_cases.py"),
        Path("no_force_suite/no_force_lb_core.py"),
    ]
    if method == "proposed":
        paths += [Path("solver_safe_nn.py")]
    h = hashlib.sha256()
    h.update(method.encode("utf-8"))
    for p in paths:
        if p.exists():
            h.update(p.as_posix().encode("utf-8"))
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _cache_path(case_id: str, method: str) -> Path:
    return CACHE_DIR / f"{case_id}__{method}__{_cache_key(method)}.npz"


def _load_cached(case_id: str, method: str):
    path = _cache_path(case_id, method)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return data["f"], [tuple(row) for row in data["hist"].tolist()], float(data["wall"])


def _save_cached(case_id: str, method: str, f, hist, wall: float):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_path(case_id, method)
    np.savez_compressed(
        tmp,
        f=np.asarray(f),
        hist=np.asarray(hist, dtype=np.float64),
        wall=float(wall),
    )


def run_one(case_id, method, use_cache=True):
    label, n, tol, _ = NO_FORCE_CASES[case_id]
    case = make_no_force_case(case_id)
    cached = _load_cached(case_id, method) if use_cache else None
    if cached is not None:
        f, hist, wall = cached
        return case, case_id, label, tol, f, hist, wall

    f, hist, wall = run_method(method, case, tol, max_steps_for(case_id), verbose=False)
    _save_cached(case_id, method, f, hist, wall)
    return case, case_id, label, tol, f, hist, wall


def row_for(base_id, case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall):
    final_res = float(hist[-1][1]) if hist else float("inf")
    lbe = int(hist[-1][2]) if hist else 0
    fluid = getattr(ref_case, "chi", np.ones((ref_case.N, ref_case.N), dtype=np.float64)) > 0
    err = velocity_error(ref_case, ref_f, case, f, fluid_mask=fluid)
    converged = bool(np.isfinite(final_res) and final_res < 5.0 * tol)
    return {
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


def score(rows):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)

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

        acc_eligible = [r for r in eligible if np.isfinite(r["rel_l2_vs_picard"])]
        if not acc_eligible:
            acc_eligible = [r for r in fixed if np.isfinite(r["rel_l2_vs_picard"])]

        best_lbe = min((r["lbe_calls"] for r in eligible if r["lbe_calls"] > 0), default=10**18)
        best_wall = min((r["wall_seconds"] for r in eligible if r["wall_seconds"] > 0), default=float("inf"))
        best_acc = min((r["rel_l2_vs_picard"] for r in acc_eligible), default=float("inf"))

        lbe_win = bool(prop["lbe_calls"] <= best_lbe)
        wall_win = bool(prop["wall_seconds"] <= best_wall)
        acc_win = bool(prop["rel_l2_vs_picard"] <= best_acc * 1.001 + 1e-12)
        conv = bool(prop["converged"])
        case_results.append(
            {
                "case_id": case_id,
                "case_pass": int(conv and lbe_win and wall_win and acc_win),
                "converged": int(conv),
                "lbe_win": int(lbe_win),
                "wall_win": int(wall_win),
                "acc_win": int(acc_win),
                "proposed_lbe": prop["lbe_calls"],
                "best_fixed_lbe": int(best_lbe) if best_lbe < 10**18 else None,
                "proposed_wall": prop["wall_seconds"],
                "best_fixed_wall": best_wall,
                "proposed_rel_l2": prop["rel_l2_vs_picard"],
                "best_fixed_rel_l2": best_acc,
            }
        )

    pass_count = sum(c["case_pass"] for c in case_results)
    lbe_wins = sum(c["lbe_win"] for c in case_results)
    wall_wins = sum(c["wall_win"] for c in case_results)
    acc_wins = sum(c["acc_win"] for c in case_results)
    convs = sum(c["converged"] for c in case_results)
    speedups = [
        c["best_fixed_lbe"] / c["proposed_lbe"] for c in case_results if c["best_fixed_lbe"] is not None and c["proposed_lbe"] > 0
    ]
    return {
        "all_pass": int(bool(case_results and pass_count == len(case_results))),
        "case_count": len(case_results),
        "pass_count": int(pass_count),
        "converged_count": int(convs),
        "lbe_win_count": int(lbe_wins),
        "wall_win_count": int(wall_wins),
        "accuracy_win_count": int(acc_wins),
        "mean_lbe_speedup_vs_best_fixed": float(np.mean(speedups) if speedups else 0.0),
        "case_results": case_results,
    }


def write_outputs(rows, metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASE_IDS), help="comma-separated no-force case ids")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method ids")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    case_ids = [x.strip() for x in args.cases.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if "picard_lbm" not in methods:
        methods = ["picard_lbm"] + methods
    if "proposed" not in methods:
        methods = methods + ["proposed"]

    rows = []
    started = time.perf_counter()
    for case_id in case_ids:
        if case_id not in NO_FORCE_CASES:
            raise ValueError(f"unknown case_id: {case_id}")
        label, _, _, _ = NO_FORCE_CASES[case_id]

        print(f"[case] {case_id}: {label}")
        # reference from picard
        ref_case, _, _, ref_tol, ref_f, ref_hist, ref_wall = run_one(case_id, "picard_lbm", use_cache=not args.no_cache)
        for method in methods:
            try:
                if method == "picard_lbm":
                    case, _, _, tol, f, hist, wall = ref_case, case_id, label, ref_tol, ref_f, ref_hist, ref_wall
                else:
                    case, _, _, tol, f, hist, wall = run_one(case_id, method, use_cache=not args.no_cache)
            except Exception as exc:
                print(f"  {method} crashed: {exc}")
                case = make_no_force_case(case_id)
                f = case.initial_field()
                hist = [(0, float("inf"), 0, 0.0)]
                wall = 0.0

            row = row_for(case_id, case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall)
            rows.append(row)
            write_history_csv(HIST_DIR / f"{case_id}__{method}.csv", hist)
            if not args.no_vtk:
                try:
                    write_vtk(VTK_DIR / f"{case_id}__{method}.vtk", case, f)
                except Exception:
                    pass

            print(
                f"  {method:22s} lbe={row['lbe_calls']:8d} res={row['final_residual']:.3e} "
                f"relL2={row['rel_l2_vs_picard']:.3e} conv={row['converged']}"
            )

    metrics = score(rows)
    metrics.update({
        "elapsed_wall_seconds": float(time.perf_counter() - started),
        "case_count": len(case_ids),
        "method_count": len(methods),
        "methods": methods,
    })
    write_outputs(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}")
    print(f"[saved] {OUT / 'metrics.json'}")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
