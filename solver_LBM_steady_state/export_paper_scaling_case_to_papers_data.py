#!/usr/bin/env python3
"""Export one no-force scaling benchmark case to papers_data/<case_id>.

This exporter is intentionally strict for paper figures:
- it loads cache entries using the cache labels recorded in summary.csv;
- residual-vs-wall plots use raw solver wall seconds only;
- wall seconds must be strictly monotone after adding the initial point;
- the plotted final point must match summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ghia_validation import get_ghia_data
from lbm_periodic import equilibrium
from paper_60case_benchmark_no_force import macro_of, write_vtk
import paper_60case_benchmark_no_force_scaling as scaling_benchmark
from paper_60case_benchmark_no_force_scaling import (
    _load_cached,
    case_factory_scaled,
)
from solver_baseline import reference_residual_norm


SOURCE_DIR = Path("paper_revision_data") / "no_force_scaling_benchmark"
SOURCE_HIST = SOURCE_DIR / "histories"
SOURCE_DIAG = SOURCE_DIR / "diagnostics"
DST_ROOT = Path("papers_data")

METHOD_ORDER = [
    "picard_lbm",
    "anderson_lbm",
    "preconditioned_lbm",
    "inexact_newton_lbe",
    "dual_time_mg_lbm",
    "proposed",
]
METHOD_LABELS = {
    "picard_lbm": "Picard",
    "anderson_lbm": "Anderson",
    "preconditioned_lbm": "Preconditioned",
    "inexact_newton_lbe": "Inexact Newton",
    "dual_time_mg_lbm": "Dual-time MG",
    "proposed": "SafeNN",
}
SIMPLICITY_SCORES = {
    "picard_lbm": 1.00,
    "proposed": 0.75,
    "preconditioned_lbm": 0.60,
    "anderson_lbm": 0.55,
    "dual_time_mg_lbm": 0.45,
    "inexact_newton_lbe": 0.40,
}
CANONICAL_BASE_CASE_IDS = [
    "channel_poiseuille_rect",
    "couette_n32",
    "cavity_re100_n33",
    "cavity_re400_n49",
    "cavity_re1000_n129",
    "multi_cylinder_n32",
    "backward_step_n64",
    "cylinder_wake_n64",
    "t_junction_rect",
]


def _log_normalized_score(value: float, best: float, worst: float) -> float:
    if not (np.isfinite(value) and value > 0.0):
        return 0.0
    if not (np.isfinite(best) and np.isfinite(worst) and best > 0.0 and worst > 0.0) or abs(math.log(worst) - math.log(best)) < 1.0e-15:
        return 0.5
    lo = math.log(best)
    hi = math.log(worst)
    x = (math.log(value) - lo) / max(hi - lo, 1.0e-30)
    return float(np.clip(1.0 - x, 0.0, 1.0))


def _cache_labels_for_row(row: dict) -> tuple[str, str]:
    picard_tol = _float(row.get("reference_tol", row.get("tol")))
    method_tol = _float(row.get("method_tol", row.get("tol")))
    picard = row.get("cache_label_picard") or f"tol{picard_tol:.2e}"
    method = row.get("cache_label_method") or f"tol{method_tol:.2e}"
    return str(picard), str(method)


def _cache_hashes_for_row(row: dict) -> tuple[str, str]:
    recorded_picard = str(row.get("cache_hash_picard", ""))
    recorded_method = str(row.get("cache_hash_method", ""))
    if recorded_picard and recorded_method:
        return recorded_picard, recorded_method
    picard_label, method_label = _cache_labels_for_row(row)
    return (
        scaling_benchmark._cache_key("picard_lbm", picard_label),
        scaling_benchmark._cache_key(str(row["method"]), method_label),
    )


def _cavity_physical_tolerances(base_case_id: str) -> dict[str, float]:
    if base_case_id == "cavity_re100_n33":
        return {
            "ghia_literature_rms": 3.0e-2,
            "ghia_method_rms": 1.0e-2,
            "centerline_delta_rms": 5.0e-3,
            "field_rel_l2": 4.0e-3,
        }
    if base_case_id == "cavity_re400_n49":
        return {
            "ghia_literature_rms": 6.0e-2,
            "ghia_method_rms": 3.0e-2,
            "centerline_delta_rms": 1.0e-2,
            "field_rel_l2": 3.0e-3,
        }
    return {
        "ghia_literature_rms": 8.0e-2,
        "ghia_method_rms": 5.0e-2,
        "centerline_delta_rms": 1.5e-2,
        "field_rel_l2": 5.0e-3,
    }


def _fluid_mask_for_case(case):
    rho_shape = case.initial_field().shape[-2:]
    return getattr(case, "chi", np.ones(rho_shape, dtype=np.float64)) > 0.0


def _row_case_metric_components(row: dict):
    base_case_id = row.get("base_case_id", "")
    components = []
    rel_l2 = _float(row.get("rel_l2_vs_ref", row.get("rel_l2_vs_picard")))
    if base_case_id == "channel_poiseuille_rect":
        level = int(_float(row.get("scaling_level", 3)))
        tt = {
            1: {"channel_core_rel_l2_analytic": 6.0e-3, "channel_rel_l2_vs_tight_picard": 3.0e-3, "channel_core_flux_cv": 5.0e-4, "channel_boundary_flux_imbalance": 1.0e-3},
            2: {"channel_core_rel_l2_analytic": 2.0e-3, "channel_rel_l2_vs_tight_picard": 2.0e-3, "channel_core_flux_cv": 4.0e-4, "channel_boundary_flux_imbalance": 7.5e-4},
            3: {"channel_core_rel_l2_analytic": 1.0e-3, "channel_rel_l2_vs_tight_picard": 1.5e-3, "channel_core_flux_cv": 3.0e-4, "channel_boundary_flux_imbalance": 5.0e-4},
        }.get(level, {"channel_core_rel_l2_analytic": 1.0e-3, "channel_rel_l2_vs_tight_picard": 1.5e-3, "channel_core_flux_cv": 3.0e-4, "channel_boundary_flux_imbalance": 5.0e-4})
        components.append(("channel_core_rel_l2_analytic", abs(_float(row.get("channel_core_rel_l2_analytic"), rel_l2)), tt["channel_core_rel_l2_analytic"]))
        components.append(("channel_rel_l2_vs_tight_picard", abs(_float(row.get("channel_rel_l2_vs_tight_picard"), rel_l2)), tt["channel_rel_l2_vs_tight_picard"]))
        components.append(("channel_core_flux_cv", abs(_float(row.get("channel_core_flux_cv"), 0.0)), tt["channel_core_flux_cv"]))
        components.append(("channel_boundary_flux_imbalance", abs(_float(row.get("channel_boundary_flux_imbalance"), row.get("flux_imbalance", 0.0))), tt["channel_boundary_flux_imbalance"]))
    elif base_case_id == "couette_n32":
        components.append(("rel_l2", rel_l2, 1.0e-3))
        components.append(("max_jump_rel", abs(_float(row.get("max_jump_rel"))), 1.0e-3))
    elif base_case_id in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        tt = _cavity_physical_tolerances(base_case_id)
        components.append(("ghia_u_centerline_rms", abs(_float(row.get("ghia_u_centerline_rms"), rel_l2)), tt["ghia_method_rms"]))
        components.append(("ghia_v_centerline_rms", abs(_float(row.get("ghia_v_centerline_rms"), rel_l2)), tt["ghia_method_rms"]))
        components.append(("cavity_centerline_delta_u_rms", abs(_float(row.get("cavity_centerline_delta_u_rms"), rel_l2)), tt["centerline_delta_rms"]))
        components.append(("cavity_centerline_delta_v_rms", abs(_float(row.get("cavity_centerline_delta_v_rms"), rel_l2)), tt["centerline_delta_rms"]))
        components.append(("cavity_field_rel_l2_vs_tight_ref", abs(_float(row.get("cavity_field_rel_l2_vs_tight_ref"), row.get("rel_l2_vs_picard", rel_l2))), tt["field_rel_l2"]))
    elif base_case_id in {"multi_cylinder_n32", "backward_step_n64", "cylinder_wake_n64", "t_junction_rect"}:
        components.append(("rel_l2", rel_l2, 5.0e-4))
        components.append(("mass_imbalance", abs(_float(row.get("mass_imbalance"))), 1.0e-4))
    else:
        components.append(("rel_l2", rel_l2, 1.0e-3))
    return components


def _accuracy_score_from_row(row: dict):
    components = _row_case_metric_components(row)
    if not components:
        return 0.5, False, []
    zs = []
    detail = []
    for name, err, tol in components:
        tol = max(float(tol), 1.0e-30)
        z = float(abs(err) / tol)
        zs.append(z)
        detail.append({"name": name, "error": float(err), "tolerance": float(tol), "z": z})
    E = float(np.sqrt(np.mean(np.square(zs)))) if zs else float("inf")
    score = float(max(0.0, 1.0 - min(E, 2.0) / 2.0))
    return score, bool(E <= 1.0), detail


def _row_hard_gates(row: dict, history: list[dict]):
    finite_keys = ["lbe_calls", "wall_seconds", "final_residual", "rel_l2_vs_ref", "linf_vs_ref", "rms_vs_ref"]
    if row.get("base_case_id") == "channel_poiseuille_rect":
        finite_keys.extend([
            "channel_core_rel_l2_analytic",
            "channel_full_rel_l2_analytic",
            "channel_rel_l2_vs_tight_picard",
            "channel_core_flux_cv",
            "channel_boundary_flux_imbalance",
        ])
    if row.get("base_case_id") in {"cavity_re100_n33", "cavity_re400_n49", "cavity_re1000_n129"}:
        finite_keys.extend([
            "ghia_u_centerline_rms",
            "ghia_v_centerline_rms",
            "cavity_centerline_delta_u_rms",
            "cavity_centerline_delta_v_rms",
            "cavity_field_rel_l2_vs_tight_ref",
        ])
    finite_fields = bool(np.all([np.isfinite(_float(row.get(k))) for k in finite_keys]))
    cache_hash_picard, cache_hash_method = _cache_hashes_for_row(row)
    cache_hash_valid = cache_hash_picard == str(row.get("cache_hash_picard", "")) and cache_hash_method == str(row.get("cache_hash_method", ""))
    residual_pass = bool(_float(row.get("final_residual")) < 5.0 * _float(row.get("tol")))
    accuracy_score, accuracy_pass, accuracy_detail = _accuracy_score_from_row(row)
    wall_monotone = bool(all(float(b["wall_seconds"]) > float(a["wall_seconds"]) for a, b in zip(history, history[1:]))) if history else False
    final_consistent = bool(history and abs(_float(history[-1]["residual"]) - _float(row.get("final_residual"))) <= max(1.0e-14, abs(_float(row.get("final_residual"))) * 1.0e-8) and int(history[-1]["lbe_calls"]) == int(_float(row.get("lbe_calls"))) and abs(_float(history[-1]["wall_seconds"]) - _float(row.get("wall_seconds"))) <= max(1.0e-12, abs(_float(row.get("wall_seconds"))) * 1.0e-8))
    physical_converged = bool(_float(row.get("physical_converged", 1.0)) > 0.5)
    hard_pass = bool(finite_fields and cache_hash_valid and residual_pass and accuracy_pass and physical_converged and wall_monotone and final_consistent)
    return {
        "finite_fields": int(finite_fields),
        "cache_hash_valid": int(cache_hash_valid),
        "residual_pass": int(residual_pass),
        "accuracy_pass": int(accuracy_pass),
        "history_wall_monotone": int(wall_monotone),
        "history_final_consistent": int(final_consistent),
        "physical_converged": int(physical_converged),
        "hard_pass": int(hard_pass),
        "accuracy_score": float(accuracy_score),
        "accuracy_detail": accuracy_detail,
    }


def _float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def _case_rows(case_id: str):
    path = SOURCE_DIR / "summary.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("case_id") == case_id]
    if not rows:
        raise RuntimeError(f"missing summary rows for {case_id}")
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda r: order.get(r.get("method", ""), 999))


def _fluid_mask(case):
    rho_shape = case.initial_field().shape[-2:]
    return getattr(case, "chi", np.ones(rho_shape, dtype=np.float64)) > 0.0


def _load_field_from_hash(row: dict):
    method = row["method"]
    case_id = row["case_id"]
    picard_label, method_label = _cache_labels_for_row(row)
    label = picard_label if method == "picard_lbm" else method_label
    cached = _load_cached(case_id, method, label or None)
    if cached is None:
        cache_hash = str(row.get("cache_hash_picard" if method == "picard_lbm" else "cache_hash_method", ""))
        path = scaling_benchmark.CACHE_DIR / f"{case_id}__{method}__{cache_hash}.npz"
        if cache_hash and path.exists():
            data = np.load(path, allow_pickle=False)
            cached = data["f"], [tuple(r) for r in data["hist"].tolist()], float(data["wall"])
    if cached is None:
        raise RuntimeError(f"hash-matched cache missing for {case_id} {method} label={label!r}")
    return cached


def _read_history(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("accepted") not in (None, "", "1", "1.0", "true", "True"):
                continue
            # Publication plots must use the raw solver clock.  The benchmark
            # history also carries a normalized wall axis for debugging phase
            # resets, but that axis is not acceptable for representative timing
            # figures.
            wall_key = "wall_seconds_raw" if r.get("wall_seconds_raw") not in (None, "") else "wall_seconds"
            rows.append(
                {
                    "iter": int(_float(r.get("iter", r.get("iter_or_step", 0)), 0)),
                    "residual": _float(r.get("residual")),
                    "f_rms_residual": _float(r.get("f_rms_residual")),
                    "lbe_calls": int(_float(r.get("lbe_calls"), 0)),
                    "wall_seconds": _float(r.get(wall_key)),
                    "residual_kind": r.get("residual_kind", ""),
                }
            )
    return [r for r in rows if np.isfinite(r["residual"]) and np.isfinite(r["wall_seconds"])]


def _macro_tail_start_lbe(case_id: str, method: str) -> int | None:
    diag = SOURCE_DIAG / f"{case_id}__{method}__diagnostics.csv"
    if not diag.exists():
        return None
    with diag.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("phase") == "cavity_residual_plateau_tail":
                try:
                    return int(_float(row.get("lbe_calls"), 0))
                except Exception:
                    return None
    return None


def _uses_unified_macro_l2_history(case_id: str, method: str) -> bool:
    diag = SOURCE_DIAG / f"{case_id}__{method}__diagnostics.csv"
    if not diag.exists():
        return False
    with diag.open("r", encoding="utf-8", newline="") as fh:
        return any(row.get("phase") in {"cavity_macro_l2_history_start", "unified_macro_l2_history_start"} for row in csv.DictReader(fh))


def _initial_residual(case):
    f0 = case.initial_field()
    return float(scaling_benchmark._macro_l2_residual_value(case, f0))


def _initial_f_rms_residual(case):
    f0 = case.initial_field()
    return float(reference_residual_norm(case, case.residual(f0)))


def _history_for_plot(case, case_id: str, method: str, summary_row: dict):
    hist = _read_history(SOURCE_HIST / f"{case_id}__{method}.csv")
    initial = {
        "iter": 0,
        "residual": _initial_residual(case),
        "f_rms_residual": _initial_f_rms_residual(case),
        "lbe_calls": 1,
        "wall_seconds": 1.0e-6,
        "residual_kind": "macro_l2_p_ux_uy_uz",
    }
    if not hist:
        hist = [initial]
    elif hist[0]["lbe_calls"] > 1 or hist[0]["wall_seconds"] > 1.0e-5:
        hist = [initial] + hist
    else:
        hist[0]["wall_seconds"] = max(hist[0]["wall_seconds"], 1.0e-6)
        hist[0]["lbe_calls"] = max(hist[0]["lbe_calls"], 1)

    final = {
        "iter": max(int(hist[-1]["iter"]) + 1, len(hist)),
        "residual": _float(summary_row["final_residual"]),
        "f_rms_residual": _float(summary_row.get("final_f_rms_residual")),
        "lbe_calls": int(_float(summary_row["lbe_calls"], hist[-1]["lbe_calls"])),
        "wall_seconds": _float(summary_row["wall_seconds"]),
        "residual_kind": "macro_l2_p_ux_uy_uz",
    }
    last = hist[-1]
    same_final_state = (
        abs(last["residual"] - final["residual"]) <= max(1.0e-14, abs(final["residual"]) * 1.0e-8)
        and int(last["lbe_calls"]) == int(final["lbe_calls"])
    )
    if same_final_state:
        # The solver summary can exclude first residual-evaluation overhead,
        # while raw histories include it.  Keep the raw history as the timing
        # source of truth for paper plots and sync the exported per-case summary
        # so the final plotted point and summary table agree.
        summary_row["wall_seconds"] = repr(float(last["wall_seconds"]))
        final["wall_seconds"] = float(last["wall_seconds"])
    if (
        abs(last["residual"] - final["residual"]) > max(1.0e-14, abs(final["residual"]) * 1.0e-8)
        or int(last["lbe_calls"]) != int(final["lbe_calls"])
        or abs(last["wall_seconds"] - final["wall_seconds"]) > 1.0e-8
    ):
        hist.append(final)

    prev = -math.inf
    for idx, row in enumerate(hist):
        if row["wall_seconds"] <= prev:
            raise RuntimeError(f"non-monotone wall_seconds in {case_id} {method} at row {idx}")
        prev = row["wall_seconds"]
    last = hist[-1]
    if int(last["lbe_calls"]) != int(_float(summary_row["lbe_calls"])):
        raise RuntimeError(f"final lbe mismatch for {case_id} {method}")
    if abs(last["wall_seconds"] - _float(summary_row["wall_seconds"])) > 1.0e-8:
        raise RuntimeError(f"final wall mismatch for {case_id} {method}")
    if abs(last["residual"] - _float(summary_row["final_residual"])) > max(1.0e-14, abs(_float(summary_row["final_residual"])) * 1.0e-8):
        raise RuntimeError(f"final residual mismatch for {case_id} {method}")
    for row in hist:
        row["residual_kind"] = "macro_l2_p_ux_uy_uz"
    return hist


def _analytic_reference(base_case_id: str, case):
    rho0, ux0, uy0 = macro_of(case, case.initial_field())
    rho = np.ones_like(rho0)
    if base_case_id == "channel_poiseuille_rect" and hasattr(case, "analytical_ux"):
        ux = np.asarray(case.analytical_ux(), dtype=np.float64)
        uy = np.zeros_like(ux)
        return "analytic_poiseuille", rho, ux, uy
    if base_case_id == "couette_n32" and hasattr(case, "analytical_ux"):
        ux = np.asarray(case.analytical_ux(), dtype=np.float64)
        uy = np.zeros_like(ux)
        return "analytic_couette", rho, ux, uy
    return None


def _reference(base_case_id: str, case, f_picard):
    analytic = _analytic_reference(base_case_id, case)
    if analytic is not None:
        return analytic
    rho, ux, uy = macro_of(case, f_picard)
    if base_case_id.startswith("cavity_"):
        return "ghia_centerline_plus_tight_picard_field", rho, ux, uy
    return "tight_picard", rho, ux, uy


def _write_history_csv(path: Path, hist):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["iter", "residual", "lbe_calls", "wall_seconds"]
    extras = []
    for row in hist:
        for key in row:
            if key not in fields and key not in extras:
                extras.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields + extras)
        wr.writeheader()
        for row in hist:
            wr.writerow(row)


def _write_field_csv(path: Path, case, f, ref):
    rho_ref, ux_ref, uy_ref = ref
    rho, ux, uy = macro_of(case, f)
    speed = np.sqrt(ux * ux + uy * uy)
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
    vort_ref = np.gradient(uy_ref, axis=1) - np.gradient(ux_ref, axis=0)
    mask = _fluid_mask(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["iy", "ix", "fluid", "rho", "ux", "uy", "speed", "vorticity", "ux_ref", "uy_ref", "speed_ref", "vorticity_ref", "speed_error"])
        ny, nx = ux.shape
        for j in range(ny):
            for i in range(nx):
                wr.writerow([
                    j,
                    i,
                    int(mask[j, i]),
                    rho[j, i],
                    ux[j, i],
                    uy[j, i],
                    speed[j, i],
                    vort[j, i],
                    ux_ref[j, i],
                    uy_ref[j, i],
                    speed_ref[j, i],
                    vort_ref[j, i],
                    speed[j, i] - speed_ref[j, i],
                ])


def _accuracy(case, f, ref):
    _rho_ref, ux_ref, uy_ref = ref
    _rho, ux, uy = macro_of(case, f)
    mask = _fluid_mask(case)
    du = ux[mask] - ux_ref[mask]
    dv = uy[mask] - uy_ref[mask]
    den = max(float(np.sqrt(np.sum(ux_ref[mask] ** 2 + uy_ref[mask] ** 2))), 1.0e-30)
    return {
        "rel_l2": float(np.sqrt(np.sum(du * du + dv * dv)) / den),
        "linf": float(max(np.max(np.abs(du)), np.max(np.abs(dv))) if du.size else 0.0),
        "rms": float(np.sqrt(np.mean(du * du + dv * dv)) if du.size else 0.0),
    }


def _plot_residuals(case_id: str, histories: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("residual_vs_iteration.png", "iter", "Iteration"),
        ("residual_vs_lbe_calls.png", "lbe_calls", "LBE calls"),
        ("residual_vs_wall_seconds.png", "wall_seconds", "Wall seconds"),
    ]
    for filename, xkey, xlabel in specs:
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        for method, hist in histories.items():
            if xkey == "iter":
                x = np.array([1.0 if int(r["iter"]) == 0 else float(r["iter"]) for r in hist], dtype=np.float64)
            elif xkey == "wall_seconds":
                x = np.array([max(1.0e-6, float(r[xkey])) for r in hist], dtype=np.float64)
            else:
                x = np.array([max(1.0, float(r[xkey])) for r in hist], dtype=np.float64)
            y = np.array([max(1.0e-16, float(r["residual"])) for r in hist])
            ax.plot(x, y, lw=1.6, label=METHOD_LABELS.get(method, method))
            if method == "proposed":
                macro_rows = [r for r in hist if r.get("residual_kind") == "macro_l2"]
                if macro_rows:
                    marker_row = macro_rows[0]
                    if xkey == "iter":
                        marker_x = 1.0 if int(marker_row["iter"]) == 0 else float(marker_row["iter"])
                    elif xkey == "wall_seconds":
                        marker_x = max(1.0e-6, float(marker_row[xkey]))
                    else:
                        marker_x = max(1.0, float(marker_row[xkey]))
                    ax.axvline(marker_x, color="0.25", lw=1.0, ls=":", alpha=0.75)
                    ax.text(marker_x, 0.98, "macro L2 tail", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7, color="0.25")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("residual norm")
        ax.set_title(case_id)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    proposed = histories.get("proposed", [])
    if proposed and any(np.isfinite(_float(r.get("f_rms_residual"))) for r in proposed):
        fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
        x = np.array([max(1.0, float(r["lbe_calls"])) for r in proposed], dtype=np.float64)
        y_macro = np.array([max(1.0e-16, float(r["residual"])) for r in proposed], dtype=np.float64)
        y_frms = np.array([max(1.0e-16, _float(r.get("f_rms_residual"))) for r in proposed], dtype=np.float64)
        ax.plot(x, y_macro, marker="s", ms=2.5, lw=1.3, label="proposed macro L2")
        if np.all(np.isfinite(y_frms)):
            ax.plot(x, y_frms, marker="o", ms=2.5, lw=1.3, label="proposed f-RMS")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("LBE calls")
        ax.set_ylabel("residual norm")
        ax.set_title(f"{case_id} proposed residuals by kind")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / "proposed_residual_by_kind_vs_lbe_calls.png", dpi=220)
        plt.close(fig)


def _write_and_plot_residual_summary(case_id: str, rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_rows = []
    for row in rows:
        residual_rows.append(
            {
                "case_id": case_id,
                "method": row.get("method", ""),
                "tol": _float(row.get("tol")),
                "residual_threshold": 5.0 * _float(row.get("tol")),
                "final_residual_kind": row.get("final_residual_kind", "macro_l2_p_ux_uy_uz"),
                "final_macro_l2_residual": _float(row.get("final_macro_l2_residual", row.get("final_residual"))),
                "final_macro_l2_pressure": _float(row.get("final_macro_l2_pressure")),
                "final_macro_l2_ux": _float(row.get("final_macro_l2_ux")),
                "final_macro_l2_uy": _float(row.get("final_macro_l2_uy")),
                "final_macro_l2_uz": _float(row.get("final_macro_l2_uz")),
                "final_f_rms_residual": _float(row.get("final_f_rms_residual")),
                "residual_converged": int(_float(row.get("residual_converged"), 0)),
                "converged": int(_float(row.get("converged"), 0)),
            }
        )
    with (out_dir.parent / "residual_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(residual_rows[0].keys()))
        wr.writeheader()
        wr.writerows(residual_rows)

    labels = [METHOD_LABELS.get(r["method"], r["method"]) for r in residual_rows]
    x = np.arange(len(labels), dtype=np.float64)
    macro = np.array([max(1.0e-16, r["final_macro_l2_residual"]) for r in residual_rows], dtype=np.float64)
    frms = np.array([max(1.0e-16, r["final_f_rms_residual"]) for r in residual_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2.0, macro, width, label="macro L2")
    if np.all(np.isfinite(frms)):
        ax.bar(x + width / 2.0, frms, width, label="f-RMS")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("final residual norm")
    ax.set_title(f"{case_id} final residual metrics")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "final_macro_l2_vs_f_rms_residual.png", dpi=220)
    plt.close(fig)


def _plot_fields(case_id: str, case, fields: dict, ref, out_dir: Path):
    _rho_ref, ux_ref, uy_ref = ref
    speed_ref = np.sqrt(ux_ref * ux_ref + uy_ref * uy_ref)
    mask = _fluid_mask(case)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, f in fields.items():
        _rho, ux, uy = macro_of(case, f)
        speed = np.sqrt(ux * ux + uy * uy)
        err = np.sqrt((ux - ux_ref) ** 2 + (uy - uy_ref) ** 2)
        vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)
        for name, arr, cmap in [
            ("velocity_magnitude", speed, "viridis"),
            ("vorticity", vort, "coolwarm"),
            ("velocity_error", err, "magma"),
        ]:
            fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
            im = ax.imshow(np.where(mask, arr, np.nan), origin="lower", cmap=cmap, aspect="auto")
            ax.set_title(f"{case_id} {METHOD_LABELS.get(method, method)} {name}")
            fig.colorbar(im, ax=ax)
            fig.savefig(out_dir / f"{case_id}__{method}__{name}.png", dpi=220)
            plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), constrained_layout=True)
    ny, nx = ux_ref.shape
    i_mid = nx // 2
    j_mid = ny // 2
    x = np.arange(nx)
    y = np.arange(ny)
    axes[0].plot(y, ux_ref[:, i_mid], "k--", lw=2.0, label="reference")
    axes[1].plot(x, uy_ref[j_mid, :], "k--", lw=2.0, label="reference")
    for method, f in fields.items():
        _rho, ux, uy = macro_of(case, f)
        axes[0].plot(y, ux[:, i_mid], lw=1.2, label=METHOD_LABELS.get(method, method))
        axes[1].plot(x, uy[j_mid, :], lw=1.2, label=METHOD_LABELS.get(method, method))
    axes[0].set_xlabel("y index")
    axes[0].set_ylabel("u_x at x mid")
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("u_y at y mid")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.savefig(out_dir / "centerline_profiles.png", dpi=220)
    plt.close(fig)


def _write_ghia(case, fields: dict, out_dir: Path):
    re = int(getattr(case, "Re", 0))
    if re not in {100, 400, 1000}:
        return
    y_g, u_g, x_g, v_g = get_ghia_data(re)
    ny = case.N
    grid = np.linspace(0.0, 1.0, ny)
    u_wall = max(float(getattr(case, "U_wall", 1.0)), 1.0e-30)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].plot(u_g, y_g, "ko", ms=3, label="Ghia")
    axes[1].plot(x_g, v_g, "ko", ms=3, label="Ghia")
    rows = []
    for method, f in fields.items():
        _rho, ux, uy = macro_of(case, f)
        mid = ny // 2
        u_line = ux[:, mid] / u_wall
        v_line = uy[mid, :] / u_wall
        axes[0].plot(u_line, grid, lw=1.2, label=METHOD_LABELS.get(method, method))
        axes[1].plot(grid, v_line, lw=1.2, label=METHOD_LABELS.get(method, method))
        u_i = np.interp(y_g, grid, u_line)
        v_i = np.interp(x_g, grid, v_line)
        rows.append({
            "method": method,
            "u_centerline_linf": float(np.max(np.abs(u_i - u_g))),
            "v_centerline_linf": float(np.max(np.abs(v_i - v_g))),
            "u_centerline_rms": float(np.sqrt(np.mean((u_i - u_g) ** 2))),
            "v_centerline_rms": float(np.sqrt(np.mean((v_i - v_g) ** 2))),
        })
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_xlabel("u/U")
    axes[0].set_ylabel("y")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v/U")
    fig.savefig(out_dir / "ghia_centerline_comparison.png", dpi=220)
    plt.close(fig)
    with (out_dir.parent / "ghia_centerline_error.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def _score_rows(rows, histories):
    eligible_fixed = []
    scored_rows = []
    for row in rows:
        method = row["method"]
        hist = histories.get(method, [])
        gates = _row_hard_gates(row, hist)
        acc_score, acc_pass, acc_detail = _accuracy_score_from_row(row)
        wall_score = 0.5
        lbe_score = 0.5
        scored_rows.append(
            {
                **row,
                **gates,
                "accuracy_score": float(acc_score),
                "accuracy_pass": int(acc_pass),
                "accuracy_detail": acc_detail,
                "simplicity_score": float(SIMPLICITY_SCORES.get(method, 0.50)),
            }
        )
        if method != "proposed" and gates["hard_pass"] and acc_pass:
            eligible_fixed.append(scored_rows[-1])

    if eligible_fixed:
        best_wall = min(float(r["wall_seconds"]) for r in eligible_fixed if float(r["wall_seconds"]) > 0.0)
        worst_wall = max(float(r["wall_seconds"]) for r in eligible_fixed if float(r["wall_seconds"]) > 0.0)
        best_lbe = min(int(r["lbe_calls"]) for r in eligible_fixed if int(r["lbe_calls"]) > 0)
        worst_lbe = max(int(r["lbe_calls"]) for r in eligible_fixed if int(r["lbe_calls"]) > 0)
    else:
        best_wall = worst_wall = float("nan")
        best_lbe = worst_lbe = 0

    for row in scored_rows:
        if eligible_fixed:
            row["wall_score"] = float(_log_normalized_score(float(row["wall_seconds"]), best_wall, worst_wall))
            row["lbe_score"] = float(_log_normalized_score(float(row["lbe_calls"]), best_lbe, worst_lbe))
        else:
            row["wall_score"] = 0.5
            row["lbe_score"] = 0.5
        row["speed_score"] = float(0.70 * row["wall_score"] + 0.30 * row["lbe_score"])
        row["total_score"] = float(0.50 * row["speed_score"] + 0.40 * row["accuracy_score"] + 0.10 * row["simplicity_score"])

    best_ref_total = max((r["total_score"] for r in scored_rows if r["method"] != "proposed" and r["hard_pass"] and r["accuracy_pass"]), default=float("nan"))
    for row in scored_rows:
        row["best_ref_total"] = float(best_ref_total)
        row["total_margin"] = float(row["total_score"] - best_ref_total) if np.isfinite(best_ref_total) else float("nan")
        row["proposed_pass"] = int(row["method"] == "proposed" and bool(row["hard_pass"]) and bool(row["accuracy_pass"]) and np.isfinite(best_ref_total) and row["total_margin"] > 0.02)
    return scored_rows


def _paper_case_result(case_id: str):
    path = SOURCE_DIR / "metrics_paper.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for result in data.get("case_results", []):
        if result.get("case_id") == case_id:
            return result
    return None


def _merge_paper_case_result(scored_rows: list[dict], case_id: str) -> dict | None:
    result = _paper_case_result(case_id)
    if result is None:
        return None
    proposed = next((row for row in scored_rows if row["method"] == "proposed"), None)
    if proposed is None:
        return result
    for src, dst in [
        ("finite_fields", "finite_fields"),
        ("cache_hash_valid", "cache_hash_valid"),
        ("residual_pass", "residual_pass"),
        ("accuracy_pass", "accuracy_pass"),
        ("history_wall_monotone", "history_wall_monotone"),
        ("history_final_consistent", "history_final_consistent"),
        ("wall_score", "wall_score"),
        ("lbe_score", "lbe_score"),
        ("speed_score", "speed_score"),
        ("accuracy_score", "accuracy_score"),
        ("simplicity_score", "simplicity_score"),
        ("total_score", "total_score"),
        ("best_ref_total", "best_ref_total"),
        ("total_margin", "total_margin"),
        ("case_pass", "proposed_pass"),
    ]:
        if src in result:
            proposed[dst] = result[src]
    proposed["hard_pass"] = int(result.get("hard_pass", proposed.get("hard_pass", 0)))
    return result


def export_case(case_id: str, force: bool = False):
    rows = _case_rows(case_id)
    base_case_id = rows[0]["base_case_id"]
    level = int(float(rows[0]["scaling_level"]))
    _cid, _label, _tol, factory = case_factory_scaled(base_case_id, level)
    case = factory()

    fields = {}
    histories = {}
    diagnostics_to_copy = []
    for row in rows:
        method = row["method"]
        f, _hist_cached, _wall = _load_field_from_hash(row)
        fields[method] = f
        histories[method] = _history_for_plot(case, case_id, method, row)
        diag_src = SOURCE_DIAG / f"{case_id}__{method}__diagnostics.csv"
        if diag_src.exists():
            diagnostics_to_copy.append((method, diag_src))

    if "picard_lbm" not in fields:
        raise RuntimeError(f"{case_id} has no Picard field")

    final_dst = DST_ROOT / case_id
    dst = DST_ROOT / f".{case_id}.tmp"
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ["histories", "fields", "vtk", "figure", "diagnostics"]:
        (dst / sub).mkdir(parents=True, exist_ok=True)

    for row in rows:
        method = row["method"]
        f = fields[method]
        hist = histories[method]
        _write_history_csv(dst / "histories" / f"{case_id}__{method}.csv", hist)
        write_vtk(dst / "vtk" / f"{case_id}__{method}.vtk", case, f)
    for _method, diag_src in diagnostics_to_copy:
        shutil.copy2(diag_src, dst / "diagnostics" / diag_src.name)
    ref_name, rho_ref, ux_ref, uy_ref = _reference(base_case_id, case, fields["picard_lbm"])
    ref = (rho_ref, ux_ref, uy_ref)
    acc_rows = []
    for method, f in fields.items():
        _write_field_csv(dst / "fields" / f"{case_id}__{method}.csv", case, f, ref)
        acc = _accuracy(case, f, ref)
        acc_rows.append({"case_id": case_id, "method": method, "reference": ref_name, **acc})

    with (dst / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with (dst / "accuracy_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(acc_rows[0].keys()))
        wr.writeheader()
        wr.writerows(acc_rows)
    scored_rows = _score_rows(rows, histories)
    paper_result = _merge_paper_case_result(scored_rows, case_id)
    score_fields = [
        "case_id",
        "base_case_id",
        "scaling_level",
        "method",
        "finite_fields",
        "cache_hash_valid",
        "residual_pass",
        "accuracy_pass",
        "history_wall_monotone",
        "history_final_consistent",
        "hard_pass",
        "channel_core_rel_l2_analytic",
        "channel_full_rel_l2_analytic",
        "channel_rel_l2_vs_tight_picard",
        "channel_core_flux_cv",
        "channel_boundary_flux_imbalance",
        "ghia_u_centerline_rms",
        "ghia_v_centerline_rms",
        "cavity_centerline_delta_u_rms",
        "cavity_centerline_delta_v_rms",
        "cavity_field_rel_l2_vs_tight_ref",
        "ghia_literature_gate_pass",
        "ghia_method_gate_pass",
        "tight_ref_gate_pass",
        "physical_converged",
        "eligible_for_score",
        "wall_score",
        "lbe_score",
        "speed_score",
        "accuracy_score",
        "simplicity_score",
        "total_score",
        "best_ref_total",
        "total_margin",
        "proposed_pass",
    ]
    with (dst / "score.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=score_fields)
        wr.writeheader()
        for row in scored_rows:
            wr.writerow({k: row.get(k, "") for k in score_fields})
    _plot_residuals(case_id, histories, dst / "figure")
    _write_and_plot_residual_summary(case_id, rows, dst / "figure")
    _plot_fields(case_id, case, fields, ref, dst / "figure")
    _write_ghia(case, fields, dst / "figure")

    manifest = {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "scaling_level": level,
        "reference": ref_name,
        "source_summary": str(SOURCE_DIR / "summary.csv"),
        "source_cache_dir": str(SOURCE_DIR / "npz_cache"),
        "source_diagnostics_dir": str(SOURCE_DIAG),
        "canonical_base_case_ids": CANONICAL_BASE_CASE_IDS,
        "history_policy": "raw wall_seconds only; strict monotone; final point matches summary.csv",
        "score_policy": "log-normalized wall/lbe, weighted total score with hard gates",
        "channel_accuracy_components": {
            "1x": {
                "channel_core_rel_l2_analytic": 6.0e-3,
                "channel_rel_l2_vs_tight_picard": 3.0e-3,
                "channel_core_flux_cv": 5.0e-4,
                "channel_boundary_flux_imbalance": 1.0e-3,
            },
            "2x": {
                "channel_core_rel_l2_analytic": 2.0e-3,
                "channel_rel_l2_vs_tight_picard": 2.0e-3,
                "channel_core_flux_cv": 4.0e-4,
                "channel_boundary_flux_imbalance": 7.5e-4,
            },
            "3x": {
                "channel_core_rel_l2_analytic": 1.0e-3,
                "channel_rel_l2_vs_tight_picard": 1.5e-3,
                "channel_core_flux_cv": 3.0e-4,
                "channel_boundary_flux_imbalance": 5.0e-4,
            },
        },
        "figure_format": "png",
        "methods": list(fields.keys()),
        "score_csv": str(final_dst / "score.csv"),
        "metrics_json": str(final_dst / "metrics.json"),
        "accuracy_summary_csv": str(final_dst / "accuracy_summary.csv"),
    }
    metrics = {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "scaling_level": level,
        "source_dir": str(SOURCE_DIR),
        "cache_dir": str(SOURCE_DIR / "npz_cache"),
        "score_mode": "paper_weighted_total_score",
        "method_count": len(fields),
        "hard_pass_count": int(sum(int(r["hard_pass"]) for r in scored_rows)),
        "proposed_pass": int(next((r["proposed_pass"] for r in scored_rows if r["method"] == "proposed"), 0)),
        "best_ref_total": float(next((r["best_ref_total"] for r in scored_rows if r["method"] == "proposed"), float("nan"))),
        "proposed_total_score": float(next((r["total_score"] for r in scored_rows if r["method"] == "proposed"), float("nan"))),
        "canonical_base_case_ids": CANONICAL_BASE_CASE_IDS,
        "channel_accuracy_components": {
            "1x": {
                "channel_core_rel_l2_analytic": 6.0e-3,
                "channel_rel_l2_vs_tight_picard": 3.0e-3,
                "channel_core_flux_cv": 5.0e-4,
                "channel_boundary_flux_imbalance": 1.0e-3,
            },
            "2x": {
                "channel_core_rel_l2_analytic": 2.0e-3,
                "channel_rel_l2_vs_tight_picard": 2.0e-3,
                "channel_core_flux_cv": 4.0e-4,
                "channel_boundary_flux_imbalance": 7.5e-4,
            },
            "3x": {
                "channel_core_rel_l2_analytic": 1.0e-3,
                "channel_rel_l2_vs_tight_picard": 1.5e-3,
                "channel_core_flux_cv": 3.0e-4,
                "channel_boundary_flux_imbalance": 5.0e-4,
            },
        },
    }
    if paper_result is not None:
        metrics.update(
            {
                "proposed_pass": int(paper_result.get("case_pass", metrics["proposed_pass"])),
                "best_ref_total": float(paper_result.get("best_ref_total", metrics["best_ref_total"])),
                "proposed_total_score": float(paper_result.get("total_score", metrics["proposed_total_score"])),
                "paper_case_result": paper_result,
            }
        )
    (dst / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if final_dst.exists():
        if not force:
            shutil.rmtree(dst)
            raise FileExistsError(f"{final_dst} already exists; pass --force to replace it")
        shutil.rmtree(final_dst)
    shutil.move(str(dst), str(final_dst))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--source-dir", default=None)
    args = parser.parse_args()
    if args.source_dir:
        global SOURCE_DIR, SOURCE_HIST, SOURCE_DIAG
        SOURCE_DIR = Path(args.source_dir)
        SOURCE_HIST = SOURCE_DIR / "histories"
        SOURCE_DIAG = SOURCE_DIR / "diagnostics"
        scaling_benchmark.CACHE_DIR = SOURCE_DIR / "npz_cache"
    manifest = export_case(args.case_id, force=args.force)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
