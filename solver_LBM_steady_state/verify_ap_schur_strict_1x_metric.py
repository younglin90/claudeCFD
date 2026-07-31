"""Score 1x proposed AP-Schur runs by strict convergence wall time.

The final stdout line is JSON for codex-autoresearch metrics_json parsing.
Lower ``loss`` is better.  A wall-time win is counted only when the proposed
run itself satisfies the benchmark's strict ``converged`` flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "paper_revision_data" / "_coord_autoresearch_strict_convergence_1x_current"
OLD_ANALYSIS = ROOT / "paper_revision_data" / "_coord_round105_min_tjunction_3x_resume" / "proposed_vs_fixed_all_levels_analysis.csv"


def _read_float(row, key, default=float("nan")):
    try:
        value = row.get(key, "")
        if value in ("", "nan", "NaN", None):
            return default
        return float(value)
    except Exception:
        return default


def _run_screen(out_dir: Path):
    env = os.environ.copy()
    env.setdefault("NUMBA_NUM_THREADS", "4")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("SAFE_NN_ENABLE_AP_SCHUR", "1")
    env.setdefault("SAFE_NN_UNIFORM_PROPOSED", "1")
    env.setdefault("SAFE_NN_CAVITY_PLATEAU_MAX_STEPS", "1000000")
    cmd = [
        "python3",
        "run_ap_schur_proposed_only.py",
        "--levels",
        "1",
        "--base-cases",
        "all",
        "--out-dir",
        str(out_dir),
        "--overwrite",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def compute_metrics(out_dir: Path):
    summary_path = out_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    old_rows = [r for r in csv.DictReader(OLD_ANALYSIS.open(newline="")) if r.get("level") == "1x"]
    new_rows = list(csv.DictReader(summary_path.open(newline="")))
    old_by_base = {r["base_case"]: r for r in old_rows}
    new_by_base = {r["base_case_id"]: r for r in new_rows}

    rows = []
    strict_converged = 0
    wall_wins = 0
    accuracy_regressions = 0
    speedups = []
    ap_accepts = 0
    ap_trials = 0
    for base, old in old_by_base.items():
        new = new_by_base.get(base)
        if new is None:
            continue
        final_wall = _read_float(new, "wall_seconds")
        new_wall = final_wall
        fixed_wall = _read_float(old, "best_fixed_wall")
        converged = int(float(new.get("converged", 0) or 0))
        strict_converged += converged
        speedup = fixed_wall / new_wall if math.isfinite(fixed_wall) and new_wall > 0.0 else float("nan")
        wall_win = int(converged and math.isfinite(speedup) and speedup > 1.0)
        wall_wins += wall_win
        if wall_win:
            speedups.append(speedup)

        old_acc = _read_float(old, "proposed_acc")
        new_acc = _read_float(new, "rel_l2_vs_ref")
        accuracy_regression = 0
        if math.isfinite(old_acc) and math.isfinite(new_acc):
            accuracy_regression = int(new_acc > max(1.05 * old_acc, old_acc + 1.0e-12))
            accuracy_regressions += accuracy_regression

        ap_accepts += int(float(new.get("ap_schur_accepts", 0) or 0))
        ap_trials += int(float(new.get("ap_schur_trials", 0) or 0))
        rows.append(
            {
                "base_case": base,
                "proposed_wall": new_wall,
                "proposed_final_wall": final_wall,
                "best_fixed_wall": fixed_wall,
                "speedup_to_convergence": speedup,
                "converged": converged,
                "wall_win_to_convergence": wall_win,
                "accuracy_regression": accuracy_regression,
                "final_residual": _read_float(new, "final_residual"),
                "tol": _read_float(new, "tol"),
                "ap_accepts": int(float(new.get("ap_schur_accepts", 0) or 0)),
                "ap_trials": int(float(new.get("ap_schur_trials", 0) or 0)),
            }
        )

    case_count = max(len(rows), 1)
    nonconverged = case_count - strict_converged
    nonwins = case_count - wall_wins
    mean_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    # Lower is better.  Strict convergence dominates wall-time wins, and any
    # accuracy regression is a hard paper-risk penalty.
    loss = (
        1000.0
        + 80.0 * nonconverged
        + 40.0 * nonwins
        + 200.0 * accuracy_regressions
        - 10.0 * min(mean_speedup, 5.0)
    )
    metrics = {
        "loss": float(loss),
        "case_count": int(case_count),
        "strict_converged_count": int(strict_converged),
        "wall_wins_to_convergence_vs_fixed": int(wall_wins),
        "accuracy_regressions": int(accuracy_regressions),
        "mean_speedup_to_convergence_vs_fixed": float(mean_speedup),
        "nonconverged_count": int(nonconverged),
        "nonwin_count": int(nonwins),
        "ap_schur_accepts": int(ap_accepts),
        "ap_schur_trials": int(ap_trials),
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--run", action="store_true", help="run/update the proposed-only 1x screen before scoring")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.run:
        _run_screen(out_dir)
    metrics, rows = compute_metrics(out_dir)
    compare_path = out_dir / "strict_convergence_metric_rows.json"
    compare_path.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
