"""Mass-aware strict 1x proposed-method metric for autoresearch.

The solver convergence criterion remains owned by the benchmark runner.  This
wrapper adds paper-facing gates for mass/flux conservation and reference-error
regression without changing the macroscopic L2 + plateau stop logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from audit_mass_conservation_1x import audit as audit_mass
from verify_ap_schur_strict_1x_metric import compute_metrics, _run_screen, OLD_ANALYSIS


def _read_float(row, key, default=float("nan")):
    try:
        value = row.get(key, "")
        if value in ("", "nan", "NaN", None):
            return default
        return float(value)
    except Exception:
        return default


def _write_mass_csv(path: Path, rows):
    fieldnames = [
        "base_case_id",
        "case_id",
        "status",
        "mass_kind",
        "mass_initial",
        "mass_final",
        "mass_rel_drift",
        "rho_min",
        "rho_mean",
        "rho_max",
        "speed_mean",
        "speed_max",
        "inflow",
        "outflow",
        "net_flux",
        "flux_closure_rel",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mass_metrics(out_dir: Path):
    rows = audit_mass(out_dir)
    _write_mass_csv(out_dir / "mass_audit.csv", rows)
    fail_count = 0
    rho_nonpositive = 0
    max_closed_drift = 0.0
    max_open_flux_closure = 0.0
    for row in rows:
        if row.get("status") != "ok":
            fail_count += 1
            continue
        rho_min = _read_float(row, "rho_min")
        if math.isfinite(rho_min) and rho_min <= 0.0:
            rho_nonpositive += 1
            fail_count += 1
        kind = row.get("mass_kind")
        if kind == "closed_or_periodic":
            drift = _read_float(row, "mass_rel_drift")
            if math.isfinite(drift):
                max_closed_drift = max(max_closed_drift, drift)
                if drift > 1.0e-7:
                    fail_count += 1
        elif kind == "open":
            closure = _read_float(row, "flux_closure_rel")
            if math.isfinite(closure):
                max_open_flux_closure = max(max_open_flux_closure, closure)
                if closure > 1.0e-4:
                    fail_count += 1
            else:
                fail_count += 1
    return {
        "mass_conservation_fail_count": int(fail_count),
        "rho_nonpositive_count": int(rho_nonpositive),
        "max_closed_mass_rel_drift": float(max_closed_drift),
        "max_open_flux_closure_rel": float(max_open_flux_closure),
    }


def _reference_metrics(out_dir: Path):
    summary = list(csv.DictReader((out_dir / "summary.csv").open(newline="", encoding="utf-8")))
    old = {
        row["base_case"]: row
        for row in csv.DictReader(OLD_ANALYSIS.open(newline="", encoding="utf-8"))
        if row.get("level") == "1x"
    }
    regression_count = 0
    nan_count = 0
    max_ratio = 0.0
    rows = []
    for row in summary:
        base = row.get("base_case_id", "")
        new_acc = _read_float(row, "rel_l2_vs_ref")
        best_acc = _read_float(old.get(base, {}), "best_fixed_acc")
        proposed_old = _read_float(old.get(base, {}), "proposed_acc")
        if not math.isfinite(new_acc):
            nan_count += 1
            if math.isfinite(best_acc):
                regression_count += 1
        elif math.isfinite(best_acc):
            limit = max(1.01 * best_acc, best_acc + 1.0e-10)
            if new_acc > limit:
                regression_count += 1
            if best_acc > 0.0:
                max_ratio = max(max_ratio, new_acc / best_acc)
        rows.append(
            {
                "base_case": base,
                "new_rel_l2_vs_ref": new_acc,
                "best_fixed_acc": best_acc,
                "old_proposed_acc": proposed_old,
            }
        )
    (out_dir / "reference_error_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {
        "reference_error_regressions_vs_best_fixed": int(regression_count),
        "nan_reference_error_count": int(nan_count),
        "max_reference_error_ratio_vs_best_fixed": float(max_ratio),
    }


def compute_massaware_metrics(out_dir: Path):
    base_metrics, rows = compute_metrics(out_dir)
    mass_metrics = _mass_metrics(out_dir)
    ref_metrics = _reference_metrics(out_dir)
    metrics = dict(base_metrics)
    metrics.update(mass_metrics)
    metrics.update(ref_metrics)
    metrics["loss"] = float(
        base_metrics["loss"]
        + 150.0 * mass_metrics["mass_conservation_fail_count"]
        + 200.0 * ref_metrics["reference_error_regressions_vs_best_fixed"]
        + 100.0 * ref_metrics["nan_reference_error_count"]
    )
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.run:
        _run_screen(out_dir)
    metrics, rows = compute_massaware_metrics(out_dir)
    (out_dir / "massaware_metric_rows.json").write_text(
        json.dumps({"metrics": metrics, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
