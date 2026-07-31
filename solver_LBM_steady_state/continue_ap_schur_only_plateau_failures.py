"""Continue AP-Schur-only proposed runs that failed only the plateau gate.

This script is intentionally post-processing plus continuation only.  It loads
the final field and history already saved by ``run_ap_schur_proposed_only.py``,
runs the same relative macro-L2 plateau tail with a larger LBE budget, and
updates the existing proposed-only result folder in place.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_60case_benchmark_no_force import write_history_csv
from paper_60case_benchmark_no_force_scaling import (
    _case_grid_shape,
    _f_rms_residual_value,
    _macro_l2_residual_components,
    _macro_l2_residual_value,
    _relative_macro_l2_convergence_tail,
    write_diagnostic_csv,
)
from run_ap_schur_proposed_only import (
    _accuracy_against_available_reference,
    _history_final_row,
    _strict_convergence_flags,
    case_factory_scaled,
)


def _history_from_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                rows.append(
                    (
                        int(float(row.get("iter", len(rows)))),
                        float(row["residual"]),
                        int(float(row["lbe_calls"])),
                        float(row["wall_seconds"]),
                    )
                )
            except Exception:
                continue
    return rows


def _continuation_budget(case_id: str, current_lbe: int, default_extra: int) -> int:
    if "backward_step" in case_id:
        return current_lbe + max(default_extra, 1_200_000)
    if "cavity" in case_id:
        return current_lbe + max(default_extra, 600_000)
    return current_lbe + default_extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extra-lbe", type=int, default=600_000)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "summary.csv"
    rows = list(csv.DictReader(summary_path.open(newline="", encoding="utf-8")))
    selected = {x.strip() for x in args.only.split(",") if x.strip()}

    updated = []
    for row in rows:
        case_id = row["case_id"]
        if selected and case_id not in selected:
            continue
        if int(float(row.get("converged", "0"))) != 0:
            continue
        if int(float(row.get("residual_converged", "0"))) != 1:
            continue
        if int(float(row.get("plateau_converged", "0"))) != 0:
            continue

        base_id = row["base_case_id"]
        level = int(float(row["scaling_level"]))
        case_id_expected, label, tol, factory = case_factory_scaled(base_id, level)
        if case_id_expected != case_id:
            raise RuntimeError(f"case mismatch: {case_id_expected} != {case_id}")
        case = factory()

        npz_path = out_dir / "npz" / f"{case_id}__proposed.npz"
        hist_path = out_dir / "histories" / f"{case_id}__proposed.csv"
        f = np.load(npz_path, allow_pickle=False)["f"]
        hist = _history_from_csv(hist_path)
        if not hist:
            hist = [(0, float(_macro_l2_residual_value(case, f)), int(float(row["lbe_calls"])), 0.0)]

        current_lbe = int(hist[-1][2])
        max_lbe = _continuation_budget(case_id, current_lbe, int(args.extra_lbe))
        t0 = time.perf_counter()
        print(
            f"[continue] {case_id} level={level} current_lbe={current_lbe} "
            f"max_lbe={max_lbe} old_res={float(row['final_residual']):.3e}",
            flush=True,
        )

        f_new, hist_new = _relative_macro_l2_convergence_tail(
            "proposed", case, f, hist, t0, max_steps=max_lbe
        )
        wall_added = time.perf_counter() - t0
        old_wall = float(row.get("wall_seconds", 0.0))
        hist_new = _history_final_row(case, f_new, hist_new, old_wall + wall_added)
        final_res, res_p, res_ux, res_uy, res_uz = _macro_l2_residual_components(case, f_new)
        final_f_rms = _f_rms_residual_value(case, f_new)
        ref_source, err = _accuracy_against_available_reference(base_id, level, case, f_new)
        converged, residual_converged, plateau_converged, mode, rel_stats = _strict_convergence_flags(
            base_id, hist_new, float(final_res), float(tol)
        )
        ny, nx = _case_grid_shape(case)

        row.update(
            {
                "case_label": label,
                "method_variant": "uniform_ap_schur_only_continued",
                "tol": str(float(tol)),
                "Ny": str(int(ny)),
                "Nx": str(int(nx)),
                "lbe_calls": str(int(hist_new[-1][2])),
                "wall_seconds": str(float(old_wall + wall_added)),
                "final_residual": str(float(final_res)),
                "final_macro_l2_pressure": str(float(res_p)),
                "final_macro_l2_ux": str(float(res_ux)),
                "final_macro_l2_uy": str(float(res_uy)),
                "final_macro_l2_uz": str(float(res_uz)),
                "final_f_rms_residual": str(float(final_f_rms)),
                "initial_macro_l2_residual": str(
                    float(rel_stats.get("initial_macro_l2_residual", _macro_l2_residual_value(case, case.initial_field())))
                ),
                "relative_macro_l2_residual": str(
                    float(rel_stats.get("relative_macro_l2_residual", final_res / max(_macro_l2_residual_value(case, case.initial_field()), 1.0e-300)))
                ),
                "plateau_improvement": str(float(rel_stats.get("plateau_improvement", float("nan")))),
                "macro_change": str(float(rel_stats.get("macro_change", float("nan")))),
                "relative_plateau": str(int(rel_stats.get("relative_plateau", 0))),
                "macro_change_pass": str(int(rel_stats.get("macro_change_pass", 0))),
                "relative_floor_pass": str(int(rel_stats.get("relative_floor_pass", 0))),
                "min_lbe_pass": str(int(rel_stats.get("min_lbe_pass", 0))),
                "converged": str(int(converged)),
                "residual_converged": str(int(residual_converged)),
                "plateau_converged": str(int(plateau_converged)),
                "convergence_mode": mode,
                "reference_source": ref_source,
                "rel_l2_vs_ref": str(float(err["rel_l2"])),
                "linf_vs_ref": str(float(err["linf"])),
                "rms_vs_ref": str(float(err["rms"])),
                "vel_abs_l2_vs_ref": str(float(err["vel_abs_l2"])),
                "vel_abs_linf_vs_ref": str(float(err["vel_abs_linf"])),
                "vel_abs_rms_vs_ref": str(float(err["vel_abs_rms"])),
            }
        )

        write_history_csv(hist_path, hist_new)
        write_diagnostic_csv(out_dir / "diagnostics" / f"{case_id}__proposed__diagnostics.csv", hist_new)
        np.savez_compressed(npz_path, f=f_new)
        updated.append(
            {
                "case_id": case_id,
                "lbe_calls": int(hist_new[-1][2]),
                "wall_seconds": float(row["wall_seconds"]),
                "final_residual": float(final_res),
                "plateau_improvement": float(row["plateau_improvement"]),
                "converged": int(row["converged"]),
            }
        )
        print(
            f"  -> lbe={row['lbe_calls']} wall={float(row['wall_seconds']):.3f}s "
            f"res={float(final_res):.3e} plateau={float(row['plateau_improvement']):.3e} "
            f"conv={row['converged']}",
            flush=True,
        )

    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pd.DataFrame(updated).to_csv(out_dir / "plateau_continuation_updates.csv", index=False)
    print(f"[saved] {summary_path}")
    print(pd.DataFrame(updated).to_string(index=False))


if __name__ == "__main__":
    main()
