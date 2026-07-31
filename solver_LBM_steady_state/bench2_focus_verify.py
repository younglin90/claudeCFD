"""Focused verifier for the two previously failing universal-method cases."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import paper_60case_benchmark as bench


FOCUS_CASES = ["cavity_re1000_n129", "t_junction_n64"]
METHODS = ["picard_lbm", "preconditioned_lbm", "dual_time_mg_lbm", "proposed"]


def focus_metrics(rows, elapsed):
    proposed = [r for r in rows if r["method"] == "proposed"]
    strict_pass = 0
    accuracy_pass = 0
    converged = 0
    nonfinite = 0
    speedups = []
    rels = []
    details = {}
    for row in proposed:
        cid = row["case_id"]
        finite = np.isfinite(row["final_residual"]) and np.isfinite(row["rel_l2_vs_picard"])
        if not finite:
            nonfinite += 1
        rel = float(row["rel_l2_vs_picard"])
        speed = float(row["lbe_speedup_vs_picard"])
        speedups.append(speed if np.isfinite(speed) else 0.0)
        rels.append(rel if np.isfinite(rel) else float("inf"))
        conv_ok = bool(row["converged"])
        acc_ok = finite and rel <= 0.05
        strict_ok = conv_ok and acc_ok and speed > 1.0
        converged += int(conv_ok)
        accuracy_pass += int(acc_ok)
        strict_pass += int(strict_ok)
        details[cid] = {
            "lbe_calls": int(row["lbe_calls"]),
            "lbe_speedup_vs_picard": speed,
            "wall_speedup_vs_picard": float(row["wall_speedup_vs_picard"]),
            "final_residual": float(row["final_residual"]),
            "rel_l2_vs_picard": rel,
            "strict_pass": int(strict_ok),
        }
    worst_rel = max(rels) if rels else float("inf")
    mean_speed = float(sum(speedups) / max(len(speedups), 1))
    score = (
        100.0 * strict_pass
        + 10.0 * converged
        + 5.0 * accuracy_pass
        + mean_speed
        - 40.0 * nonfinite
        - 30.0 * max(0.0, worst_rel - 0.05)
    )
    return {
        "score": float(score),
        "strict_pass_count": int(strict_pass),
        "proposed_converged_count": int(converged),
        "accuracy_pass_count": int(accuracy_pass),
        "nonfinite_count": int(nonfinite),
        "worst_rel_l2": float(worst_rel),
        "proposed_mean_lbe_speedup": mean_speed,
        "case_count": len(FOCUS_CASES),
        "method_count": len(METHODS),
        "all_pass": int(strict_pass == len(FOCUS_CASES) and nonfinite == 0),
        "elapsed_wall_seconds": float(elapsed),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="paper_revision_data/bench2_focus")
    parser.add_argument("--no-vtk", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    bench.OUT = out
    bench.HIST_DIR = out / "histories"
    bench.VTK_DIR = out / "vtk"
    bench.OUT.mkdir(parents=True, exist_ok=True)
    bench.HIST_DIR.mkdir(parents=True, exist_ok=True)
    bench.VTK_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    started = time.perf_counter()
    for case_id in FOCUS_CASES:
        rows.extend(bench.run_case(case_id, METHODS, write_fields=not args.no_vtk, verbose=args.verbose))
    metrics = focus_metrics(rows, time.perf_counter() - started)
    bench.write_summary(rows, metrics)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
