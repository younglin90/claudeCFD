"""Single representative case: 6 methods on cavity_re400 N=49.

Goal: verify the patched 'proposed' = solve_unified_safe_nn is actually used,
and rank against 5 baselines on one case with Ghia reference.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from v2_run_all import (
    _patched_run_method,
    case_macro,
    ghia_score,
    make_case,
    picard_tail,
)
from solver_unified_safe_nn import _residual_norm


CASE_ID = "cavity_re400"
N = 49
TOL = 1e-8
BUDGET = 200000

METHODS = [
    ("picard_lbm", "Baseline Picard"),
    ("anderson_lbm", "Anderson"),
    ("preconditioned_lbm", "Preconditioned LBM"),
    ("inexact_newton_lbe", "Inexact Newton"),
    ("dual_time_mg_lbm", "Dual-time MG"),
    ("proposed", "Safe-NN (proposed)"),
]


def run_one(method):
    case = make_case(CASE_ID, N)[0]
    t0 = time.perf_counter()
    f, hist, _ = _patched_run_method(method, case, TOL, BUDGET, verbose=False)
    accel_lbe = int(hist[-1][2]) if hist else 0
    # paper-faithful tail
    f, tail_hist, tail_change = picard_tail(case, f, max_steps=BUDGET)
    tail_lbe = int(tail_hist[-1][2]) if tail_hist else 0
    wall = time.perf_counter() - t0
    _, nres = _residual_norm(case, f)
    gm, _ = ghia_score(case, f, 400)
    return {
        "method": method,
        "accel_lbe": accel_lbe,
        "tail_lbe": tail_lbe,
        "total_lbe": accel_lbe + tail_lbe,
        "wall_s": float(wall),
        "native_residual": float(nres),
        "tail_velocity_change": float(tail_change),
        "ghia_u_rms": float(gm["ghia_u_rms"]),
        "ghia_v_rms": float(gm["ghia_v_rms"]),
        "ghia_u_linf": float(gm["ghia_u_linf"]),
        "ghia_v_linf": float(gm["ghia_v_linf"]),
        "converged": bool(np.isfinite(tail_change) and tail_change < 1e-6),
    }


def main():
    rows = []
    for m, label in METHODS:
        print(f"=== {label} ({m}) ===", flush=True)
        r = run_one(m)
        r["label"] = label
        rows.append(r)
        print(json.dumps(r, indent=2), flush=True)
    out = Path("paper_revision_data/v2_final/single_compare_cavity_re400_n49.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # report
    print("\n\n=== Ranking (cavity_re400 N=49, all methods + paper tail) ===")
    print(f"{'method':22s} | {'total LBE':>10s} | {'wall (s)':>8s} | "
          f"{'native res':>11s} | {'vel-chg':>9s} | {'Ghia u_rms':>11s} | conv")
    base_lbe = next(r["total_lbe"] for r in rows if r["method"] == "picard_lbm")
    base_wall = next(r["wall_s"] for r in rows if r["method"] == "picard_lbm")
    for r in rows:
        sp_l = f"{base_lbe / r['total_lbe']:.2f}x" if r["total_lbe"] else "-"
        sp_w = f"{base_wall / r['wall_s']:.2f}x" if r["wall_s"] else "-"
        print(f"{r['label']:22s} | {r['total_lbe']:>10d} | {r['wall_s']:>8.1f} | "
              f"{r['native_residual']:>11.2e} | {r['tail_velocity_change']:>9.2e} | "
              f"{r['ghia_u_rms']:>11.3e} | {'Y' if r['converged'] else 'N'}  "
              f"speed L={sp_l} W={sp_w}")


if __name__ == "__main__":
    main()
