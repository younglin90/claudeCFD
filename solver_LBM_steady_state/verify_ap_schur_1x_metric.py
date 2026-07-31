"""Compute the 1x AP-Schur proposed screening metric.

The final stdout line is JSON for codex-autoresearch metrics_json parsing.
Lower ``loss`` is better.  The score compares the latest proposed-only 1x run
against the previous fixed-reference analysis without recomputing fixed methods.
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
DEFAULT_OUT = ROOT / "paper_revision_data" / "_coord_round106_ap_schur_proposed_1x_quick"
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
    env.setdefault("SAFE_NN_ENABLE_AP_SCHUR", "1")
    env.setdefault("SAFE_NN_RELATIVE_MACRO_MAX_LBE", "300000")
    env.setdefault("SAFE_NN_RELATIVE_MACRO_MIN_LBE", "20000")
    env.setdefault("SAFE_NN_CAVITY_PLATEAU_MAX_STEPS", "250000")
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
    wall_wins = 0
    converged = 0
    accuracy_regressions = 0
    ap_accepts = 0
    ap_trials = 0
    speedups = []
    old_proposed_speedups = []
    for base, old in old_by_base.items():
        new = new_by_base.get(base)
        if new is None:
            continue
        new_wall = _read_float(new, "wall_seconds")
        fixed_wall = _read_float(old, "best_fixed_wall")
        old_wall = _read_float(old, "proposed_wall")
        if math.isfinite(new_wall) and math.isfinite(fixed_wall) and new_wall > 0.0:
            speedup = fixed_wall / new_wall
            speedups.append(speedup)
            if speedup > 1.0:
                wall_wins += 1
        else:
            speedup = float("nan")
        if math.isfinite(old_wall) and math.isfinite(new_wall) and new_wall > 0.0:
            old_proposed_speedups.append(old_wall / new_wall)
        if int(float(new.get("converged", 0) or 0)):
            converged += 1
        old_acc = _read_float(old, "proposed_acc")
        new_acc = _read_float(new, "rel_l2_vs_ref")
        if math.isfinite(old_acc) and math.isfinite(new_acc):
            if new_acc > max(1.05 * old_acc, old_acc + 1.0e-12):
                accuracy_regressions += 1
        ap_accepts += int(float(new.get("ap_schur_accepts", 0) or 0))
        ap_trials += int(float(new.get("ap_schur_trials", 0) or 0))
        rows.append(
            {
                "base_case": base,
                "new_wall": new_wall,
                "best_fixed_wall": fixed_wall,
                "fixed_wall_speedup": speedup,
                "new_converged": int(float(new.get("converged", 0) or 0)),
                "ap_accepts": int(float(new.get("ap_schur_accepts", 0) or 0)),
                "ap_trials": int(float(new.get("ap_schur_trials", 0) or 0)),
            }
        )

    n = max(len(rows), 1)
    mean_speedup = sum(speedups) / len(speedups) if speedups else 0.0
    mean_old_prop_speedup = sum(old_proposed_speedups) / len(old_proposed_speedups) if old_proposed_speedups else 0.0
    nonconverged = n - converged

    # Lower is better.  Wall wins and convergence dominate; accuracy regressions
    # are a hard penalty because they are reviewer-visible.
    loss = (
        100.0
        - 8.0 * wall_wins
        - 3.0 * converged
        - 5.0 * min(mean_speedup, 3.0)
        + 25.0 * accuracy_regressions
        + 2.0 * nonconverged
    )
    metrics = {
        "loss": float(loss),
        "case_count": int(n),
        "wall_wins_vs_fixed": int(wall_wins),
        "converged_count": int(converged),
        "accuracy_regressions": int(accuracy_regressions),
        "mean_wall_speedup_vs_fixed": float(mean_speedup),
        "mean_wall_speedup_vs_previous_proposed": float(mean_old_prop_speedup),
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
    compare_path = out_dir / "autoresearch_metric_rows.json"
    compare_path.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
