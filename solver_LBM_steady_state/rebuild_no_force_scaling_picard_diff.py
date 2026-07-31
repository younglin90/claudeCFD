"""Rebuild no-force scaling summary from cached/no-vtk artifacts.

This script does not re-run solvers. It rehydrates cached fields and histories
to generate an updated summary table that includes Picard-referenced metrics
such as velocity/rho absolute errors.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from paper_60case_benchmark_no_force_scaling import (
    CASE_IDS,
    METHODS,
    HIST_DIR,
    OUT,
    case_factory_scaled,
    load_existing_rows,
    row_for,
    score,
    write_outputs,
    _load_cached,
    _read_float,
)


def read_history(path: Path):
    hist = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                hist.append((
                    int(row["iter_or_step"]),
                    float(row["residual"]),
                    int(float(row["lbe_calls"])),
                    float(row["wall_seconds"]),
                ))
            except Exception:
                continue
    return hist


def parse_filename(stem: str):
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    base_case_id = parts[0]
    level_token = parts[1]
    method = "__".join(parts[2:])
    if not level_token.endswith("x"):
        return None
    try:
        level = int(level_token[:-1])
    except Exception:
        return None
    return base_case_id, level, method


def discover_pairs():
    files = sorted(HIST_DIR.glob("*__*.csv"))
    for p in files:
        info = parse_filename(p.stem)
        if info is None:
            continue
        base_case_id, level, method = info
        if base_case_id not in CASE_IDS:
            continue
        if method not in METHODS:
            continue
        yield p, base_case_id, level, method


def rebuild():
    rows = []
    for p, base_case_id, level, method in discover_pairs():
        case_id = f"{base_case_id}__{level}x"
        _, _, tol, factory = case_factory_scaled(base_case_id, level)
        ref_case = factory()
        ref = _load_cached(case_id, "picard_lbm")
        if ref is None:
            continue
        ref_f, ref_hist, ref_wall = ref
        method_data = _load_cached(case_id, method)
        if method_data is None:
            continue
        f, hist, wall = method_data
        if not hist:
            hist = read_history(p)
            if hist:
                wall = hist[-1][3]
                case = factory()
            else:
                wall = 0.0
        case = factory()
        row = row_for(base_case_id, case_id, "", tol, ref_case, ref_f, method, case, f, hist, wall)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--levels", default="2,3")
    parser.add_argument("--base-cases", default="all")
    args = parser.parse_args()

    rows = rebuild()
    # Keep only requested subsets when explicitly passed.
    if args.base_cases != "all":
        bases = set(args.base_cases.split(","))
        rows = [r for r in rows if r["base_case_id"] in bases]
    if args.methods != "all":
        methods = set(args.methods.split(","))
        rows = [r for r in rows if r["method"] in methods]
    if args.levels != "all":
        levels = set(int(x) for x in args.levels.split(",") if x.strip())
        rows = [r for r in rows if int(r["scaling_level"]) in levels]

    metrics = score(rows)
    write_outputs(rows, metrics)
    print(f"[saved] {OUT / 'summary.csv'}")
    print(f"[saved] {OUT / 'summary.json'}")
    print(f"[saved] {OUT / 'metrics.json'}")


if __name__ == "__main__":
    main()
