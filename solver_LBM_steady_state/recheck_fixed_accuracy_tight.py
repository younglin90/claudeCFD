"""Recheck fixed-method accuracy at tighter stopping targets.

Purpose:
  Decide whether the fixed/reference methods can actually reach the same
  field accuracy level as the proposed method without a substantial cost
  increase.

This script does not modify any solver.  It compares default-tolerance and
strict-tolerance fixed-method runs against a tighter Picard reference, then
reports LBE/time/accuracy changes.
"""

from __future__ import annotations

import csv
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

from paper_60case_benchmark import (
    case_factory,
    max_steps_for,
    run_method,
    baseline_runner_for,
    velocity_error,
)
from solver_proposed_single import solve_proposed_single


OUT = Path("paper_revision_data") / "fixed_accuracy_tight_recheck"
CASES = [
    "kolmogorov_n32",
    "channel_n32",
    "couette_n32",
    "multi_cylinder_n32",
    "backward_step_n64",
    "cylinder_wake_n64",
    "t_junction_n64",
]
FIXED = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
]


def tight_reference_tol(case_id: str, default_tol: float) -> float:
    if case_id in {"kolmogorov_n32", "channel_n32", "couette_n32", "multi_cylinder_n32"}:
        return 1.0e-10
    return min(1.0e-8, default_tol * 0.1)


def run_tight_picard_reference(case_id: str, default_tol: float, factory):
    case = factory()
    runner = baseline_runner_for(case)
    tol_ref = tight_reference_tol(case_id, default_tol)
    max_steps = max_steps_for(case_id)
    if case_id in {"cylinder_wake_n64"}:
        max_steps *= 2
    else:
        max_steps *= 3
    t0 = time.perf_counter()
    f, hist = runner(
        case,
        max_steps=max_steps,
        tol=tol_ref,
        check_every=500 if case.N >= 64 else 200,
        verbose=False,
    )
    return case, f, hist, time.perf_counter() - t0, tol_ref


def run_fixed(case_id: str, method: str, tol: float, factory):
    case = factory()
    f, hist, wall = run_method(method, case, tol, max_steps_for(case_id) * 2, verbose=False)
    return case, f, hist, wall


def run_proposed(tol: float, factory):
    case = factory()
    t0 = time.perf_counter()
    f, hist = solve_proposed_single(case, tol=tol, verbose=False)
    return case, f, hist, time.perf_counter() - t0


def final_res(hist):
    return float(hist[-1][1]) if hist else float("inf")


def final_lbe(hist):
    return int(hist[-1][2]) if hist else 0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in CASES:
        label, tol, factory = case_factory(case_id)
        print(f"[case] {case_id}", flush=True)
        ref_case, ref_f, ref_hist, ref_wall, ref_tol = run_tight_picard_reference(case_id, tol, factory)
        print(
            f"  tight reference tol={ref_tol:.1e} lbe={final_lbe(ref_hist)} "
            f"res={final_res(ref_hist):.3e}",
            flush=True,
        )
        prop_case, prop_f, prop_hist, prop_wall = run_proposed(tol, factory)
        prop_err = velocity_error(ref_case, ref_f, prop_case, prop_f)
        rows.append({
            "case_id": case_id,
            "case_label": label,
            "method": "proposed",
            "run_tol": tol,
            "reference_tol": ref_tol,
            "lbe_calls": final_lbe(prop_hist),
            "wall_seconds": prop_wall,
            "final_residual": final_res(prop_hist),
            "rel_l2_vs_tight_ref": prop_err["rel_l2"],
            "linf_vs_tight_ref": prop_err["linf"],
            "reference_lbe": final_lbe(ref_hist),
            "reference_wall": ref_wall,
        })
        print(
            f"  proposed              lbe={final_lbe(prop_hist):7d} "
            f"rel={prop_err['rel_l2']:.3e}",
            flush=True,
        )
        for method in FIXED:
            for mode, run_tol in [("default", tol), ("tight", ref_tol)]:
                try:
                    c, f, hist, wall = run_fixed(case_id, method, run_tol, factory)
                    err = velocity_error(ref_case, ref_f, c, f)
                    row = {
                        "case_id": case_id,
                        "case_label": label,
                        "method": f"{method}__{mode}",
                        "run_tol": run_tol,
                        "reference_tol": ref_tol,
                        "lbe_calls": final_lbe(hist),
                        "wall_seconds": wall,
                        "final_residual": final_res(hist),
                        "rel_l2_vs_tight_ref": err["rel_l2"],
                        "linf_vs_tight_ref": err["linf"],
                        "reference_lbe": final_lbe(ref_hist),
                        "reference_wall": ref_wall,
                    }
                except Exception as exc:
                    row = {
                        "case_id": case_id,
                        "case_label": label,
                        "method": f"{method}__{mode}",
                        "run_tol": run_tol,
                        "reference_tol": ref_tol,
                        "lbe_calls": 0,
                        "wall_seconds": 0.0,
                        "final_residual": float("inf"),
                        "rel_l2_vs_tight_ref": float("inf"),
                        "linf_vs_tight_ref": float("inf"),
                        "reference_lbe": final_lbe(ref_hist),
                        "reference_wall": ref_wall,
                        "error": str(exc),
                    }
                rows.append(row)
                print(
                    f"  {row['method']:28s} lbe={row['lbe_calls']:7d} "
                    f"res={row['final_residual']:.3e} rel={row['rel_l2_vs_tight_ref']:.3e}",
                    flush=True,
                )
    fields = [
        "case_id",
        "case_label",
        "method",
        "run_tol",
        "reference_tol",
        "lbe_calls",
        "wall_seconds",
        "final_residual",
        "rel_l2_vs_tight_ref",
        "linf_vs_tight_ref",
        "reference_lbe",
        "reference_wall",
    ]
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k) for k in fields})
    (OUT / "summary.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[saved] {OUT / 'summary.csv'}", flush=True)
    print(json.dumps({"rows": len(rows), "out": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
