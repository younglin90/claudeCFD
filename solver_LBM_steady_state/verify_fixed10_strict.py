"""Strict 10-case comparison for the current single proposed LBM solver.

This harness keeps all baseline/comparison methods fixed by importing their
existing implementations from ``paper_60case_benchmark.py``.  The only
mutable research surface is ``solver_proposed_single.solve_proposed_single``.

The pass gate is intentionally strict and mechanical:
  * proposed must converge;
  * proposed must use no more LBE calls than the best converged fixed method;
  * proposed wall time must be no slower than the best converged fixed method;
  * proposed accuracy must be no worse than the best converged non-reference
    fixed method, measured against the Picard reference solution.

The Picard run is the reference solution for per-case velocity differences,
so Picard itself is excluded from the accuracy competitor set.
"""

from __future__ import annotations

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

from paper_60case_benchmark import (
    CASE_IDS,
    METHODS,
    case_factory,
    max_steps_for,
    run_method,
    velocity_error,
)
from solver_proposed_single import solve_proposed_single


OUT = Path("paper_revision_data") / "fixed10_strict"
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


def _final_residual(case, f) -> float:
    r = case.residual(f)
    val = case._fast_norm(r) / math.sqrt(case.dof)
    return float(val) if np.isfinite(val) else float("inf")


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


def _cache_path(case_id: str, method: str) -> Path:
    return CACHE / f"{case_id}__{method}__{_cache_key(method)}.npz"


def _cache_key(method: str) -> str:
    """Version cache by solver/config sources so stale npz files are bypassed.

    This keeps old artifacts on disk for auditability, but changing the
    proposed solver, faithful baseline dispatcher, or verifier scoring creates
    a new cache namespace automatically.
    """
    files = [
        Path("verify_fixed10_strict.py"),
        Path("paper_60case_benchmark.py"),
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


def run_one(case_id: str, method: str, tol: float, factory, use_cache=True):
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
        f, hist, wall = run_method(method, case, tol, max_steps_for(case_id), verbose=False)
    _save_cached(case_id, method, f, hist, wall)
    _write_history(case_id, method, hist)
    return case, f, hist, wall


def row_for(case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall):
    final_res = float(hist[-1][1]) if hist else _final_residual(case, f)
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


def anti_cheat_pass() -> tuple[bool, list[str]]:
    text = Path("solver_proposed_single.py").read_text(encoding="utf-8")
    hits = [token for token in FORBIDDEN_TOKENS if token in text]
    return (len(hits) == 0), hits


def score(rows):
    by_case = {case_id: [r for r in rows if r["case_id"] == case_id] for case_id in CASE_IDS}
    case_results = []
    for case_id, case_rows in by_case.items():
        prop = next(r for r in case_rows if r["method"] == "proposed")
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
        "all_pass": int(pass_count == len(CASE_IDS) and anti_ok),
        "case_count": len(CASE_IDS),
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
    started = time.perf_counter()
    rows = []
    for case_id in CASE_IDS:
        label, tol, factory = case_factory(case_id)
        print(f"[case] {case_id}: {label}", flush=True)
        ref_case, ref_f, ref_hist, ref_wall = run_one(case_id, "picard_lbm", tol, factory)
        for method in FIXED_METHODS:
            if method == "picard_lbm":
                case, f, hist, wall = ref_case, ref_f, ref_hist, ref_wall
            else:
                try:
                    case, f, hist, wall = run_one(case_id, method, tol, factory)
                except Exception as exc:
                    print(f"  {method} crashed: {exc}", flush=True)
                    case = factory()
                    f = case.initial_field()
                    hist = [(0, float("inf"), 0, 0.0)]
                    wall = 0.0
            row = row_for(case_id, label, tol, ref_case, ref_f, method, case, f, hist, wall)
            rows.append(row)
            print(
                f"  {method:22s} lbe={row['lbe_calls']:8d} wall={row['wall_seconds']:8.3f} "
                f"res={row['final_residual']:.3e} rel={row['rel_l2_vs_picard']:.3e} conv={row['converged']}",
                flush=True,
            )
        try:
            case, f, hist, wall = run_one(case_id, "proposed", tol, factory)
        except Exception as exc:
            print(f"  proposed crashed: {exc}", flush=True)
            case = factory()
            f = case.initial_field()
            hist = [(0, float("inf"), 0, 0.0)]
            wall = 0.0
        row = row_for(case_id, label, tol, ref_case, ref_f, "proposed", case, f, hist, wall)
        rows.append(row)
        print(
            f"  {'proposed':22s} lbe={row['lbe_calls']:8d} wall={row['wall_seconds']:8.3f} "
            f"res={row['final_residual']:.3e} rel={row['rel_l2_vs_picard']:.3e} conv={row['converged']}",
            flush=True,
        )
    metrics = score(rows)
    metrics["elapsed_wall_seconds"] = float(time.perf_counter() - started)
    write_outputs(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}", flush=True)
    print(f"[saved] {OUT / 'metrics.json'}", flush=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
