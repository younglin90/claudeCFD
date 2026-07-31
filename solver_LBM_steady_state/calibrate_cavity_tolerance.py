#!/usr/bin/env python3
"""Calibrate cavity Picard residual tolerances against Ghia/tight-reference gates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from paper_60case_benchmark_no_force_scaling import (
    _cavity_centerline_delta,
    _cavity_physical_tolerances,
    _ghia_centerline_error,
    case_factory_scaled,
)
from paper_60case_benchmark_no_force import macro_of


def _candidate_tolerances(base_tol: float, level: int) -> list[float]:
    vals = [base_tol, 1.0e-7, 5.0e-8, 1.0e-8]
    if level >= 3:
        vals.append(5.0e-9)
    vals = [float(v) for v in vals if v > 0.0]
    return sorted(set(vals), reverse=True)


def _capture_metrics(base_case_id: str, case, f, f_ref) -> dict:
    err = _ghia_centerline_error(case, f, case, f)
    du, dv = _cavity_centerline_delta(case, f, f_ref)
    rho_ref, ux_ref, uy_ref = macro_of(case, f_ref)
    rho, ux, uy = macro_of(case, f)
    den = max(float(np.sqrt(np.sum(ux_ref * ux_ref + uy_ref * uy_ref))), 1.0e-30)
    rel = float(np.sqrt(np.sum((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2)) / den)
    tt = _cavity_physical_tolerances(base_case_id)
    ghia_lit = err["u_rms"] <= tt["ghia_literature_rms"] and err["v_rms"] <= tt["ghia_literature_rms"]
    ghia_method = err["u_rms"] <= tt["ghia_method_rms"] and err["v_rms"] <= tt["ghia_method_rms"]
    tight = du <= tt["centerline_delta_rms"] and dv <= tt["centerline_delta_rms"] and rel <= tt["field_rel_l2"]
    return {
        "ghia_u_centerline_rms": float(err["u_rms"]),
        "ghia_v_centerline_rms": float(err["v_rms"]),
        "ghia_linf": float(err["linf"]),
        "cavity_centerline_delta_u_rms": float(du),
        "cavity_centerline_delta_v_rms": float(dv),
        "cavity_field_rel_l2_vs_deep_ref": float(rel),
        "ghia_literature_gate_pass": int(ghia_lit),
        "ghia_method_gate_pass": int(ghia_method),
        "tight_ref_gate_pass": int(tight),
        "physical_converged": int(ghia_lit and ghia_method and tight),
    }


def calibrate_case(base_case_id: str, level: int, out_dir: Path, max_steps: int, check_every: int) -> list[dict]:
    case_id, _label, tol, factory = case_factory_scaled(base_case_id, level)
    case = factory()
    targets = _candidate_tolerances(tol, level)
    pending = set(targets)
    captured: dict[float, tuple[int, float, float, np.ndarray]] = {}
    f = case.initial_field()
    t0 = time.perf_counter()
    lbe = 0
    rn = float(case.res_norm(f))
    lbe += 1
    best_f = np.array(f, copy=True)
    best_r = rn
    for target in list(pending):
        if rn <= target:
            captured[target] = (0, rn, time.perf_counter() - t0, np.array(f, copy=True))
            pending.remove(target)

    for step in range(1, int(max_steps) + 1):
        f = case.lbe_step(f)
        lbe += 1
        if step % check_every != 0 and pending:
            continue
        rn = float(case.res_norm(f))
        lbe += 1
        if np.isfinite(rn) and rn < best_r:
            best_r = rn
            best_f = np.array(f, copy=True)
        for target in list(pending):
            if rn <= target:
                captured[target] = (step, rn, time.perf_counter() - t0, np.array(f, copy=True))
                pending.remove(target)
        if not pending and step >= check_every:
            break

    deep_step = step
    deep_wall = time.perf_counter() - t0
    deep_f = np.array(best_f, copy=True)
    rows = []
    for target in targets:
        if target in captured:
            step_i, res_i, wall_i, f_i = captured[target]
            status = "captured"
        else:
            step_i, res_i, wall_i, f_i = deep_step, best_r, deep_wall, best_f
            status = "not_reached"
        row = {
            "base_case_id": base_case_id,
            "case_id": case_id,
            "level": int(level),
            "N": int(case.N),
            "Re": int(getattr(case, "Re", 0)),
            "U_wall": float(getattr(case, "U_wall", 0.0)),
            "nu": float(getattr(case, "nu", 0.0)),
            "Re_eff": float(getattr(case, "U_wall", 0.0) * (case.N - 1) / getattr(case, "nu", 1.0)),
            "target_tol": float(target),
            "status": status,
            "step": int(step_i),
            "lbe_calls_est": int(step_i + 1),
            "wall_seconds": float(wall_i),
            "final_residual": float(res_i),
        }
        row.update(_capture_metrics(base_case_id, case, f_i, deep_f))
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"{case_id}__picard_tolerance_calibration.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cases", default="cavity_re100_n33")
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--out-dir", default="paper_revision_data/cavity_tolerance_calibration")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--check-every", type=int, default=1000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    all_rows = []
    for base in [x.strip() for x in args.base_cases.split(",") if x.strip()]:
        for level in [int(x) for x in args.levels.split(",") if x.strip()]:
            rows = calibrate_case(base, level, out_dir, args.max_steps, args.check_every)
            all_rows.extend(rows)
            print(json.dumps({"case_id": rows[0]["case_id"], "rows": rows}, sort_keys=True))
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        wr.writeheader()
        wr.writerows(all_rows)


if __name__ == "__main__":
    main()
