#!/usr/bin/env python3
"""Emit a numeric JSON metric for paper-level EHD cone-jet validation gaps.

The final stdout line is intentionally numeric-only because codex-autoresearch
metrics_json rows reject nested objects and arrays.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _newest_existing(paths: list[Path]) -> Path:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return paths[0]
    return max(existing, key=lambda path: path.stat().st_mtime)


def _candido_log_dir_is_partial(path: Path) -> bool:
    """Detect a Candido CTest run that was interrupted after truncating CSVs."""
    if not path.exists():
        return True
    candido_csvs = list(path.glob("candido_*.csv"))
    if not candido_csvs:
        return True
    zero_byte = sum(1 for csv_path in candido_csvs if csv_path.stat().st_size == 0)
    nonzero = len(candido_csvs) - zero_byte
    return zero_byte > 0 and zero_byte >= max(2, nonzero)


def _select_logs_dir(root: Path) -> Path:
    """Prefer fresh CTest logs unless they are an interrupted partial run."""
    marker = "candido_guo_morphology_error3d.csv"
    candidates = [
        root / "build" / "benchmark_logs" / marker,
        root / "benchmark_logs" / marker,
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return candidates[0].parent
    complete = [
        path for path in existing if not _candido_log_dir_is_partial(path.parent)
    ]
    if complete:
        return max(complete, key=lambda path: path.stat().st_mtime).parent
    return _newest_existing(candidates).parent


def _finite_fields(row: dict[str, str], keys: list[str]) -> bool:
    return all(math.isfinite(_as_float(row, key)) for key in keys)


def _all_positive(rows: list[dict[str, str]], key: str) -> bool:
    return bool(rows) and all(_as_float(row, key) > 0.0 for row in rows)


def _has_complete_morphology_window(rows: list[dict[str, str]]) -> bool:
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row.get("case", ""), []).append(row)
    for case_rows in by_case.values():
        if len(case_rows) >= 5 and all(
            row.get("status") != "OUT_OF_WINDOW_NOT_VALIDATED" for row in case_rows
        ):
            return True
    return False


def _has_quantified_morphology_error(rows: list[dict[str, str]]) -> bool:
    """Require a real simulation-vs-reference morphology error, not just a time-window row."""
    if not rows or not all("computed_relative_error_percent" in row for row in rows):
        return False
    finite_errors = [
        abs(_as_float(row, "computed_relative_error_percent"))
        for row in rows
        if row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
    ]
    return len(finite_errors) >= 3 and all(math.isfinite(value) for value in finite_errors)


def _morphology_error_within_bar(rows: list[dict[str, str]], threshold_percent: float) -> bool:
    errors = [
        abs(_as_float(row, "computed_relative_error_percent"))
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(errors) >= 2 and max(errors) <= threshold_percent


def _morphology_connected_proxy_within_bar(
    rows: list[dict[str, str]], threshold_percent: float
) -> bool:
    errors = [
        abs(_as_float(row, "connected_proxy_error_percent"))
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(errors) >= 2 and max(errors) <= threshold_percent


def _morphology_ray_alpha05_quantified(rows: list[dict[str, str]]) -> bool:
    values = [
        _as_float(row, "ray_alpha05_silhouette_error_percent")
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(values) >= 2 and all(math.isfinite(value) for value in values)


def _morphology_ray_alpha05_within_bar(
    rows: list[dict[str, str]], threshold_percent: float
) -> bool:
    values = [
        abs(_as_float(row, "ray_alpha05_silhouette_error_percent"))
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(values) >= 2 and max(values) <= threshold_percent


def _morphology_outer_envelope_alpha05_quantified(rows: list[dict[str, str]]) -> bool:
    values = [
        _as_float(row, "outer_envelope_alpha05_silhouette_error_percent")
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(values) >= 2 and all(math.isfinite(value) for value in values)


def _morphology_outer_envelope_alpha05_within_bar(
    rows: list[dict[str, str]], threshold_percent: float
) -> bool:
    values = [
        abs(_as_float(row, "outer_envelope_alpha05_silhouette_error_percent"))
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and _as_float(row, "reference_time_ms") > 0.0
    ]
    return len(values) >= 2 and max(values) <= threshold_percent


def _morphology_tip_sync_within_bar(rows: list[dict[str, str]], threshold_percent: float) -> bool:
    if not rows:
        return False
    for row in rows:
        if row.get("case") != "long_window_ca025":
            continue
        errors = [
            abs(_as_float(row, "sync_0_4_error_percent")),
            abs(_as_float(row, "sync_0_7_error_percent")),
        ]
        return all(math.isfinite(value) for value in errors) and max(errors) <= threshold_percent
    return False


def _morphology_phase_lag_quantified(rows: list[dict[str, str]]) -> bool:
    if len(rows) < 2:
        return False
    seen_times = set()
    for row in rows:
        if row.get("case") != "long_window_ca025":
            continue
        t = round(_as_float(row, "reference_time_ms"), 3)
        values = [
            _as_float(row, "fixed_error_percent"),
            _as_float(row, "best_volume_error_percent"),
            _as_float(row, "time_lag_ms"),
            _as_float(row, "local_slope_di3_per_ms"),
            _as_float(row, "phase_explained_fraction"),
        ]
        if row.get("status", "") and all(math.isfinite(value) for value in values):
            seen_times.add(t)
    return 0.4 in seen_times and 0.7 in seen_times


def _initial_time_origin_identified(
    rows: list[dict[str, str]], logs: Path, threshold_percent: float
) -> bool:
    if not rows:
        return False
    t0_rows = [
        row
        for row in rows
        if row.get("case") == "long_window_ca025"
        and abs(_as_float(row, "reference_time_ms")) <= 1e-12
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
    ]
    if not t0_rows:
        return False
    reference_volume = _as_float(t0_rows[0], "digitized_experimental_volume_di3")
    history = _read_rows(logs / "candido_morphology_timeseries3d.csv")
    errors = [
        abs(100.0 * (_as_float(row, "morphology_volume_di3") - reference_volume) /
            max(abs(reference_volume), 1e-30))
        for row in history
        if row.get("case") == "long_window_ca025"
    ]
    return bool(errors) and min(errors) <= threshold_percent


def _late_morphology_digitized(rows: list[dict[str, str]]) -> bool:
    digitized_times = {
        round(_as_float(row, "reference_time_ms"), 3)
        for row in rows
        if row.get("case") == "long_window_ca025"
        and row.get("status") == "DIGITIZED_EXTERNAL_COMPARISON"
        and math.isfinite(_as_float(row, "computed_relative_error_percent"))
    }
    return 0.8 in digitized_times and 0.9 in digitized_times


def _late_morphology_blocker_documented(rows: list[dict[str, str]]) -> bool:
    if len(rows) < 2:
        return False
    blocked_times = {
        round(_as_float(row, "reference_time_ms"), 3)
        for row in rows
        if row.get("status") == "BLOCKED_DIGITIZED_GEOMETRY_REQUIRED"
        and row.get("required_input", "")
    }
    return 0.8 in blocked_times and 0.9 in blocked_times


def _late_morphology_source_audit_quantified(rows: list[dict[str, str]]) -> bool:
    required = {
        "candido_public_aip_fig3_image",
        "candido_local_pdf_fig3_render",
        "candido_paper_text_error_row",
        "guo2018_source_paper_figures",
        "candido_data_availability_statement",
        "candido_external_dataset_schema",
        "candido_author_github_interisofoamehd",
    }
    by_source = {row.get("source_id", ""): row for row in rows}
    if not required.issubset(by_source):
        return False
    for source_id in required:
        row = by_source[source_id]
        if not row.get("required_input", ""):
            return False
        if row.get("has_0_8_contour") != "0" or row.get("has_0_9_contour") != "0":
            return False
        if not row.get("status", ""):
            return False
    return True


def _refinement_radius_convergent(rows: list[dict[str, str]]) -> bool:
    """Coarse paper-readiness guard: the jet-radius observable must not oscillate wildly."""
    if len(rows) < 3:
        return False
    ordered = sorted(rows, key=lambda row: _as_float(row, "nx"))
    radii = [_as_float(row, "final_midplane_jet_radius") for row in ordered]
    if any((not math.isfinite(r)) or r <= 0.0 for r in radii):
        return False
    coarse, mid, fine = radii[:3]
    denom = max(abs(fine), 1e-30)
    coarse_to_mid = abs(mid - coarse) / denom
    mid_to_fine = abs(fine - mid) / denom
    return mid_to_fine < coarse_to_mid and mid_to_fine <= 0.35


def _quality_row_passes(rows: list[dict[str, str]], observable: str) -> bool:
    return any(
        row.get("observable") == observable and row.get("status") == "PASS_CONVERGING"
        for row in rows
    )


def _current_scaling_order_of_magnitude(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    validation_rows = [row for row in rows if row.get("case") == "long_window_ca025"]
    if not validation_rows:
        return False
    ratios = [_as_float(row, "convective_ratio") for row in validation_rows]
    return all(math.isfinite(ratio) and 0.1 <= ratio <= 10.0 for ratio in ratios)


def _current_voltage_sensitivity_ok(rows: list[dict[str, str]]) -> bool:
    return any(
        row.get("status") == "APPROXIMATE_WEAK_AVERAGE_VOLTAGE_SENSITIVITY"
        and _as_float(row, "tail_mean_current_ratio") <= 2.0
        for row in rows
    )


def _long_window_mass_budget_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {"long_window_ca025", "long_window_ca042"}
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        residual = _as_float(row, "relative_mass_budget_residual")
        boundary_inflow = _as_float(row, "cumulative_boundary_liquid_inflow")
        boundary_outflow = _as_float(row, "cumulative_boundary_liquid_outflow")
        if (
            row.get("status") == "OPEN_BOUNDARY_BUDGET_CLOSED"
            and math.isfinite(residual)
            and residual <= 1e-10
            and math.isfinite(boundary_inflow)
            and math.isfinite(boundary_outflow)
            and (boundary_inflow > 0.0 or boundary_outflow > 0.0)
        ):
            seen.add(case)
    return seen == required_cases


def _long_window_charge_budget_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {"long_window_ca025", "long_window_ca042"}
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        values = [
            _as_float(row, "initial_integrated_charge"),
            _as_float(row, "final_integrated_charge"),
            _as_float(row, "cumulative_boundary_charge_flux"),
            _as_float(row, "charge_budget_expected_final"),
            _as_float(row, "charge_budget_residual"),
            _as_float(row, "cumulative_charge_clamp_correction_l1"),
            _as_float(row, "max_charge_clamped_cells"),
        ]
        if (
            row.get("status", "").startswith("CHARGE_BUDGET_QUANTIFIED")
            and all(math.isfinite(value) for value in values)
        ):
            seen.add(case)
    return seen == required_cases


def _charge_subcycling_diagnostic_quantified(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        values = [
            _as_float(row, "subcycles"),
            _as_float(row, "baseline_relative_charge_budget_residual"),
            _as_float(row, "subcycled_relative_charge_budget_residual"),
            _as_float(row, "clamp_correction_ratio"),
            _as_float(row, "current_ratio"),
        ]
        return row.get("status", "") and all(math.isfinite(value) for value in values)
    return False


def _charge_conservative_bounding_diagnostic_quantified(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        values = [
            _as_float(row, "baseline_relative_charge_budget_residual"),
            _as_float(row, "bounded_relative_charge_budget_residual"),
            _as_float(row, "residual_ratio"),
            _as_float(row, "bounded_max_redistribution_residual"),
            _as_float(row, "current_ratio"),
        ]
        return row.get("status", "") and all(math.isfinite(value) for value in values)
    return False


def _charge_limit_sensitivity_quantified(rows: list[dict[str, str]]) -> bool:
    if len(rows) < 3:
        return False
    limits = {_as_float(row, "charge_limit_base") for row in rows}
    if not {5.0, 50.0, 500.0}.issubset(limits):
        return False
    for row in rows:
        values = [
            _as_float(row, "relative_charge_budget_residual"),
            _as_float(row, "cumulative_charge_clamp_correction_l1"),
            _as_float(row, "max_convective_current"),
            _as_float(row, "max_velocity"),
        ]
        if not all(math.isfinite(value) for value in values):
            return False
    return True


def _charge_reference_gap_documented(rows: list[dict[str, str]]) -> bool:
    required = {
        "bulk_charge_conservation",
        "current_boundary_treatment",
        "voltage_sensitivity",
    }
    documented = {
        row.get("item", "")
        for row in rows
        if row.get("missing_requirement", "") and row.get("status", "")
    }
    return required.issubset(documented)


def _reduced_collector_current_fixture_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "ca_independent_boundary_reduced_collector_0_75mm",
        "ca_independent_boundary_inlet_alpha_reduced_collector_0_75mm",
    }
    cases = {row.get("case", "") for row in rows}
    if not required_cases.issubset(cases):
        return False
    numeric_keys = [
        "midplane_y_over_Di",
        "low_tail_max_tip_y",
        "high_tail_max_tip_y",
        "low_alpha_mass_drift",
        "high_alpha_mass_drift",
        "low_max_div",
        "high_max_div",
    ]
    return all(
        row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in numeric_keys)
        for row in rows
    )


def _paper_charge_boundary_current_candidate_quantified(
    rows: list[dict[str, str]]
) -> bool:
    sources = {row.get("external_source", "") for row in rows}
    required_sources = {
        "Candido_Fig8b_text_average_current_not_influenced_by_voltage",
        "Candido_Fig8b_current_ie=int_S_qe_U_dot_n_dS;alpha05_liquid_jet_cross_section",
    }
    if not required_sources.issubset(sources):
        return False
    numeric_keys = [
        "low_peak_convective_current",
        "high_peak_convective_current",
        "peak_current_ratio",
        "low_mean_tail_convective_current",
        "high_mean_tail_convective_current",
        "tail_mean_current_ratio",
    ]
    return all(
        row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in numeric_keys)
        for row in rows
    )


def _fig8b_current_blocker_documented(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required_numeric = [
        "paper_midplane_y_over_Di",
        "reduced_midplane_y_over_Di",
        "paper_low_midplane_developed_samples",
        "paper_high_midplane_developed_samples",
        "reduced_low_midplane_developed_samples",
        "reduced_high_midplane_developed_samples",
        "best_tail_ratio",
        "best_peak_ratio",
    ]
    return any(
        row.get("case") == "coarse_smoke_fig8b_current"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        for row in rows
    )


def _current_pareto_tradeoff_documented(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "baseline_long_window",
        "paper_charge_boundary",
        "paper_inlet_velocity",
    }
    cases = {row.get("case", "") for row in rows}
    if not required_cases.issubset(cases):
        return False
    numeric_keys = [
        "low_max_morphology_error_0_4_0_7_percent",
        "high_max_radial_asymmetry",
        "all_phase_tail_current_ratio",
        "all_phase_peak_current_ratio",
        "axial_alpha05_convective_ratio",
        "axial_alpha05_total_ratio",
        "low_fixed_midplane_developed_samples",
        "high_fixed_midplane_developed_samples",
    ]
    return all(
        row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in numeric_keys)
        for row in rows
        if row.get("case", "") in required_cases
    )


def _open_boundary_current_diagnostic_quantified(rows: list[dict[str, str]]) -> bool:
    required_numeric = [
        "low_boundary_liquid_outflow",
        "high_boundary_liquid_outflow",
        "all_phase_tail_current_ratio",
        "axial_alpha05_convective_ratio",
        "low_fixed_midplane_developed_samples",
        "high_fixed_midplane_developed_samples",
        "low_max_morphology_error_0_4_0_7_percent",
        "high_max_radial_asymmetry",
    ]
    return any(
        row.get("case") == "paper_inlet_velocity_open_atmosphere"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        for row in rows
    )


def _paper_current_development_tradeoff_quantified(rows: list[dict[str, str]]) -> bool:
    required_numeric = [
        "stable_low_alpha_mass_drift",
        "stable_high_alpha_mass_drift",
        "extended_low_alpha_mass_drift",
        "extended_high_alpha_mass_drift",
        "extended_low_max_div",
        "extended_high_max_div",
        "stable_low_midplane_developed_samples",
        "stable_high_midplane_developed_samples",
        "extended_low_midplane_developed_samples",
        "extended_high_midplane_developed_samples",
        "extended_low_max_midplane_alpha05_area_di2",
        "extended_high_max_midplane_alpha05_area_di2",
        "extended_low_max_tip_y",
        "extended_high_max_tip_y",
    ]
    return any(
        row.get("case") == "paper_inlet_velocity_open_atmosphere_extended90"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        for row in rows
    )


def _preconditioned_current_plane_diagnostic_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_numeric = [
        "preconditioned_tip_y_over_Di",
        "preconditioned_radius_Di",
        "preconditioned_width_Di",
        "low_fixed_midplane_developed_samples",
        "high_fixed_midplane_developed_samples",
        "low_max_fixed_midplane_alpha05_area_di2",
        "high_max_fixed_midplane_alpha05_area_di2",
        "fixed_mean_current_ratio",
        "fixed_peak_current_ratio",
        "all_phase_tail_current_ratio",
        "all_phase_peak_current_ratio",
        "low_alpha_mass_drift",
        "high_alpha_mass_drift",
        "low_max_div",
        "high_max_div",
    ]
    return any(
        row.get("case") == "paper_preconditioned_current_plane"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        for row in rows
    )


def _moving_collector_boundary_diagnostic_quantified(rows: list[dict[str, str]]) -> bool:
    required_numeric = [
        "collector_speed_m_per_s",
        "collector_speed_dimensionless",
        "all_phase_tail_current_ratio",
        "all_phase_peak_current_ratio",
        "low_fixed_midplane_developed_samples",
        "high_fixed_midplane_developed_samples",
        "low_axial_developed_samples",
        "high_axial_developed_samples",
        "axial_alpha05_convective_ratio",
        "axial_alpha05_total_ratio",
        "low_alpha_mass_drift",
        "high_alpha_mass_drift",
        "low_max_div",
        "high_max_div",
    ]
    return any(
        row.get("case") == "paper_inlet_velocity_open_atmosphere_moving_collector"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        for row in rows
    )


def _poisson_face_current_observable_quantified(rows: list[dict[str, str]]) -> bool:
    required_numeric = [
        "low_ca_e",
        "high_ca_e",
        "peak_current_ratio",
        "tail_mean_current_ratio",
    ]
    current_columns = [
        key
        for key in (rows[0].keys() if rows else [])
        if key.startswith("low_peak_")
        or key.startswith("high_peak_")
        or key.startswith("low_mean_tail_")
        or key.startswith("high_mean_tail_")
    ]
    return any(
        row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        and current_columns
        and all(math.isfinite(_as_float(row, key)) for key in current_columns)
        for row in rows
    )


def _poisson_face_axial_current_window_quantified(rows: list[dict[str, str]]) -> bool:
    required_numeric = [
        "low_tail_samples",
        "high_tail_samples",
        "low_developed_samples",
        "high_developed_samples",
        "low_max_area_di2",
        "high_max_area_di2",
        "low_mean_developed_area_di2",
        "high_mean_developed_area_di2",
        "low_mean_developed_y_over_Di",
        "high_mean_developed_y_over_Di",
        "low_mean_developed_current",
        "high_mean_developed_current",
    ]
    return any(
        row.get("case") == "face_consistent_electric_poisson_face_alpha05_total"
        and row.get("status", "")
        and all(math.isfinite(_as_float(row, key)) for key in required_numeric)
        and (math.isfinite(_as_float(row, "developed_current_ratio"))
             or math.isinf(_as_float(row, "developed_current_ratio")))
        for row in rows
    )


def _poisson_face_candidate_axial_windows_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "paper_charge_boundary_poisson_face_alpha05_total",
        "paper_inlet_velocity_poisson_face_alpha05_total",
        "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_total",
        "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_total",
    }
    required_numeric = [
        "low_tail_samples",
        "high_tail_samples",
        "low_developed_samples",
        "high_developed_samples",
        "low_max_area_di2",
        "high_max_area_di2",
        "low_mean_developed_current",
        "high_mean_developed_current",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        ratio = _as_float(row, "developed_current_ratio")
        if not (math.isfinite(ratio) or math.isinf(ratio)):
            continue
        seen.add(case)
    return seen == required_cases


def _poisson_face_candidate_axial_convective_windows_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_cases = {
        "paper_charge_boundary_poisson_face_alpha05_convective",
        "paper_inlet_velocity_poisson_face_alpha05_convective",
        "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05_convective",
        "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05_convective",
    }
    required_numeric = [
        "low_tail_samples",
        "high_tail_samples",
        "low_developed_samples",
        "high_developed_samples",
        "low_max_area_di2",
        "high_max_area_di2",
        "low_mean_developed_current",
        "high_mean_developed_current",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        ratio = _as_float(row, "developed_current_ratio")
        if not (math.isfinite(ratio) or math.isinf(ratio)):
            continue
        seen.add(case)
    return seen == required_cases


def _poisson_face_convective_factorization_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_cases = {
        "paper_charge_boundary_poisson_face_alpha05",
        "paper_inlet_velocity_poisson_face_alpha05",
        "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
        "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
    }
    required_numeric = [
        "low_developed_samples",
        "high_developed_samples",
        "low_mean_area_di2",
        "high_mean_area_di2",
        "low_mean_signed_current",
        "high_mean_signed_current",
        "low_mean_abs_upwind_charge",
        "high_mean_abs_upwind_charge",
        "low_mean_abs_face_flux",
        "high_mean_abs_face_flux",
        "low_mean_abs_convective_flux",
        "high_mean_abs_convective_flux",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _poisson_face_velocity_projection_factorization_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_cases = {
        "paper_charge_boundary_poisson_face_alpha05",
        "paper_inlet_velocity_poisson_face_alpha05",
        "paper_inlet_velocity_open_atmosphere_poisson_face_alpha05",
        "paper_inlet_velocity_open_atmosphere_moving_collector_poisson_face_alpha05",
    }
    required_numeric = [
        "low_developed_samples",
        "high_developed_samples",
        "low_projected_current",
        "high_projected_current",
        "low_raw_velocity_current",
        "high_raw_velocity_current",
        "low_projected_abs_face_flux",
        "high_projected_abs_face_flux",
        "low_raw_velocity_abs_face_flux",
        "high_raw_velocity_abs_face_flux",
        "low_projected_to_raw_current",
        "high_projected_to_raw_current",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _momentum_source_factorization_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "paper_charge_boundary_alpha05",
        "paper_inlet_velocity_alpha05",
        "paper_inlet_velocity_open_atmosphere_alpha05",
        "paper_inlet_velocity_open_atmosphere_moving_collector_alpha05",
    }
    required_numeric = [
        "low_developed_samples",
        "high_developed_samples",
        "low_mean_abs_uy",
        "high_mean_abs_uy",
        "low_mean_abs_electric_source",
        "high_mean_abs_electric_source",
        "low_mean_abs_surface_source",
        "high_mean_abs_surface_source",
        "low_mean_abs_source",
        "high_mean_abs_source",
        "low_mean_abs_acceleration",
        "high_mean_abs_acceleration",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("dominant_factor", "") or not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _electric_drive_source_tradeoff_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "ca_independent_drive_relaxation_limited_alpha05",
        "ca_independent_drive_boundary_advected_alpha05",
        "unit_maxwell_drive_boundary_advected_alpha05",
    }
    required_numeric = [
        "low_developed_samples",
        "high_developed_samples",
        "velocity_ratio",
        "electric_source_ratio",
        "surface_source_ratio",
        "source_ratio",
        "acceleration_ratio",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("dominant_factor", "") or not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _boundary_current_sensitivity_quantified(rows: list[dict[str, str]]) -> bool:
    required_cases = {
        "long_window",
        "combined_charge_bounding_subcycled",
        "ca_independent_drive_boundary_advected",
        "paper_charge_boundary",
        "paper_inlet_velocity_open_atmosphere_moving_collector",
        "unit_maxwell_drive_boundary_advected",
    }
    required_numeric = [
        "low_total_cumulative_conductive_flux",
        "high_total_cumulative_conductive_flux",
        "total_cumulative_ratio",
        "low_nozzle_cumulative_flux",
        "high_nozzle_cumulative_flux",
        "nozzle_ratio",
        "low_collector_cumulative_flux",
        "high_collector_cumulative_flux",
        "collector_ratio",
        "low_lateral_cumulative_flux",
        "high_lateral_cumulative_flux",
        "lateral_ratio",
        "low_nozzle_fraction",
        "high_nozzle_fraction",
        "low_lateral_fraction",
        "high_lateral_fraction",
        "nozzle_peak_ratio",
        "lateral_peak_ratio",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("dominant_high_patch", "") or not row.get("status", ""):
            continue
        if not all(math.isfinite(_as_float(row, key)) for key in required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _interface_charge_transport_diagnostic_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_numeric = [
        "baseline_low_developed_samples",
        "baseline_high_developed_samples",
        "candidate_low_developed_samples",
        "candidate_high_developed_samples",
        "baseline_axial_alpha05_current_ratio",
        "candidate_axial_alpha05_current_ratio",
        "candidate_charge_ratio",
        "candidate_velocity_ratio",
        "candidate_low_redistribution_deficit_l1",
        "candidate_high_redistribution_deficit_l1",
        "candidate_low_weighted_cells",
        "candidate_high_weighted_cells",
        "candidate_low_weighted_capacity",
        "candidate_high_weighted_capacity",
        "candidate_low_relative_charge_budget_residual",
        "candidate_high_relative_charge_budget_residual",
        "candidate_low_alpha_mass_drift",
        "candidate_high_alpha_mass_drift",
        "candidate_low_max_div",
        "candidate_high_max_div",
        "candidate_low_morphology_error_percent",
        "candidate_high_max_radial_asymmetry",
    ]
    for row in rows:
        if row.get("case") != "interface_localized_charge_transport":
            continue
        if not row.get("status", ""):
            return False
        return all(math.isfinite(_as_float(row, key)) for key in required_numeric)
    return False


def _post_charge_potential_refresh_diagnostic_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_numeric = [
        "baseline_low_developed_samples",
        "baseline_high_developed_samples",
        "candidate_low_developed_samples",
        "candidate_high_developed_samples",
        "baseline_axial_alpha05_current_ratio",
        "candidate_axial_alpha05_current_ratio",
        "candidate_charge_ratio",
        "candidate_velocity_ratio",
        "candidate_electric_source_ratio",
        "candidate_low_post_charge_potential_residual",
        "candidate_high_post_charge_potential_residual",
        "candidate_low_post_charge_relative_gauss_residual",
        "candidate_high_post_charge_relative_gauss_residual",
        "candidate_low_alpha_mass_drift",
        "candidate_high_alpha_mass_drift",
        "candidate_low_max_div",
        "candidate_high_max_div",
        "candidate_low_morphology_error_percent",
        "candidate_high_max_radial_asymmetry",
    ]
    for row in rows:
        if row.get("case") != "post_charge_potential_refresh":
            continue
        if not row.get("status", ""):
            return False
        return all(math.isfinite(_as_float(row, key)) for key in required_numeric)
    return False


def _conductivity_potential_charge_closure_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_numeric = [
        "baseline_low_developed_samples",
        "baseline_high_developed_samples",
        "candidate_low_developed_samples",
        "candidate_high_developed_samples",
        "baseline_axial_alpha05_current_ratio",
        "candidate_axial_alpha05_current_ratio",
        "candidate_charge_ratio",
        "candidate_velocity_ratio",
        "candidate_electric_source_ratio",
        "candidate_low_conductivity_potential_residual",
        "candidate_high_conductivity_potential_residual",
        "candidate_low_closure_clamp_l1",
        "candidate_high_closure_clamp_l1",
        "candidate_low_relative_charge_budget_residual",
        "candidate_high_relative_charge_budget_residual",
        "candidate_low_alpha_mass_drift",
        "candidate_high_alpha_mass_drift",
        "candidate_low_max_div",
        "candidate_high_max_div",
        "candidate_low_morphology_error_percent",
        "candidate_high_max_radial_asymmetry",
    ]
    for row in rows:
        if row.get("case") != "conductivity_potential_charge_closure":
            continue
        if not row.get("status", ""):
            return False
        return all(math.isfinite(_as_float(row, key)) for key in required_numeric)
    return False


def _conservative_surface_charge_closure_quantified(
    rows: list[dict[str, str]]
) -> bool:
    always_required_numeric = [
        "baseline_low_developed_samples",
        "baseline_high_developed_samples",
        "candidate_low_developed_samples",
        "candidate_high_developed_samples",
        "candidate_low_implicit_ohmic_residual",
        "candidate_high_implicit_ohmic_residual",
        "candidate_low_charge_clamp_l1",
        "candidate_high_charge_clamp_l1",
        "candidate_low_redistribution_residual",
        "candidate_high_redistribution_residual",
        "candidate_low_redistribution_deficit_l1",
        "candidate_high_redistribution_deficit_l1",
        "candidate_low_relative_charge_budget_residual",
        "candidate_high_relative_charge_budget_residual",
        "candidate_low_alpha_mass_drift",
        "candidate_high_alpha_mass_drift",
        "candidate_low_max_div",
        "candidate_high_max_div",
        "candidate_low_morphology_error_percent",
        "candidate_high_max_radial_asymmetry",
    ]
    ratio_numeric = [
        "baseline_axial_alpha05_current_ratio",
        "candidate_axial_alpha05_current_ratio",
        "candidate_charge_ratio",
        "candidate_velocity_ratio",
        "candidate_electric_source_ratio",
    ]
    for row in rows:
        if row.get("case") != "conservative_surface_charge_closure":
            continue
        status = row.get("status", "")
        if not status:
            return False
        if not _finite_fields(row, always_required_numeric):
            return False
        if status.startswith("BLOCKED_"):
            return True
        return _finite_fields(row, ratio_numeric)
    return False


def _electrode_surface_current_boundary_isolation_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_cases = {
        "paper_charge_boundary",
        "nozzle_allowed_boundary",
        "collector_only_boundary",
        "implicit_filtered_paper_charge_boundary",
        "implicit_filtered_collector_only_boundary",
    }
    required_numeric = [
        "low_developed_samples",
        "high_developed_samples",
        "axial_alpha05_current_ratio",
        "charge_ratio",
        "velocity_ratio",
        "low_total_cumulative_conductive_flux",
        "high_total_cumulative_conductive_flux",
        "total_cumulative_ratio",
        "low_nozzle_cumulative_flux",
        "high_nozzle_cumulative_flux",
        "nozzle_ratio",
        "low_collector_cumulative_flux",
        "high_collector_cumulative_flux",
        "collector_ratio",
        "low_lateral_cumulative_flux",
        "high_lateral_cumulative_flux",
        "lateral_ratio",
        "nozzle_peak_ratio",
        "collector_peak_ratio",
        "lateral_peak_ratio",
        "low_total_vs_paper_ratio",
        "high_total_vs_paper_ratio",
        "low_nozzle_vs_paper_ratio",
        "high_nozzle_vs_paper_ratio",
        "low_collector_vs_paper_ratio",
        "high_collector_vs_paper_ratio",
        "max_option_effect_deviation",
        "low_relative_charge_budget_residual",
        "high_relative_charge_budget_residual",
        "low_alpha_mass_drift",
        "high_alpha_mass_drift",
        "low_max_div",
        "high_max_div",
    ]
    seen: set[str] = set()
    for row in rows:
        case = row.get("case", "")
        if case not in required_cases:
            continue
        if not row.get("dominant_high_patch", "") or not row.get("status", ""):
            continue
        if not _finite_fields(row, required_numeric):
            continue
        seen.add(case)
    return seen == required_cases


def _interfacial_ohmic_charge_source_quantified(
    rows: list[dict[str, str]]
) -> bool:
    required_numeric = [
        "baseline_low_developed_samples",
        "baseline_high_developed_samples",
        "candidate_low_developed_samples",
        "candidate_high_developed_samples",
        "baseline_axial_alpha05_current_ratio",
        "candidate_axial_alpha05_current_ratio",
        "candidate_charge_ratio",
        "candidate_velocity_ratio",
        "candidate_electric_source_ratio",
        "candidate_low_source_cells",
        "candidate_high_source_cells",
        "candidate_low_max_source_density",
        "candidate_high_max_source_density",
        "candidate_low_applied_source_charge",
        "candidate_high_applied_source_charge",
        "candidate_low_source_clamp_l1",
        "candidate_high_source_clamp_l1",
        "candidate_low_relative_charge_budget_residual",
        "candidate_high_relative_charge_budget_residual",
        "candidate_low_alpha_mass_drift",
        "candidate_high_alpha_mass_drift",
        "candidate_low_max_div",
        "candidate_high_max_div",
        "candidate_low_post_source_potential_residual",
        "candidate_high_post_source_potential_residual",
        "candidate_low_post_source_gauss_residual",
        "candidate_high_post_source_gauss_residual",
    ]
    return any(
        row.get("case") == "interfacial_ohmic_charge_source"
        and row.get("status", "")
        and _finite_fields(row, required_numeric)
        and _as_float(row, "candidate_low_source_cells") > 0.0
        and _as_float(row, "candidate_high_source_cells") > 0.0
        and _as_float(row, "candidate_low_max_source_density") > 0.0
        and _as_float(row, "candidate_high_max_source_density") > 0.0
        for row in rows
    )


def _whipping_diagnostic_quantified(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if row.get("case") != "long_window_ca042":
            continue
        values = [
            _as_float(row, "max_radial_asymmetry"),
            _as_float(row, "onset_time_ms"),
            _as_float(row, "onset_tip_y_over_Di"),
            _as_float(row, "onset_location_error_percent"),
            _as_float(row, "max_asymmetry_location_error_percent"),
            _as_float(row, "wave_peak_y_over_Di"),
            _as_float(row, "wave_peak_location_error_percent"),
            _as_float(row, "wave_speed_Di_per_sh"),
            _as_float(row, "early_to_peak_wave_speed_Di_per_sh"),
        ]
        return all(math.isfinite(value) for value in values)
    return False


def _whipping_positive_translation(rows: list[dict[str, str]]) -> bool:
    return any(
        row.get("case") == "long_window_ca042"
        and row.get("status") != "DOWNGRADED_NO_POSITIVE_WAVE_TRANSLATION"
        and _as_float(row, "early_to_peak_wave_speed_Di_per_sh") > 0.0
        for row in rows
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    logs = _select_logs_dir(root)
    docs = root / "docs" / "electrospray"
    papers = root / "papers" / "library" / "md"

    smoke_rows = _read_rows(logs / "candido_cone_jet_smoke3d.csv")
    morphology_rows = _read_rows(logs / "candido_guo_morphology_error3d.csv")
    boundary_rows = _read_rows(logs / "candido_boundary_conditions3d.csv")
    refinement_rows = _read_rows(logs / "candido_refinement_sweep3d.csv")
    refinement_quality_rows = _read_rows(logs / "candido_refinement_quality3d.csv")
    late_blocker_rows = _read_rows(logs / "candido_late_morphology_blocker3d.csv")
    late_source_audit_rows = _read_rows(
        logs / "candido_late_morphology_source_audit3d.csv"
    )
    tip_sync_rows = _read_rows(logs / "candido_morphology_tip_sync_diagnostic3d.csv")
    phase_lag_rows = _read_rows(logs / "candido_morphology_phase_lag_diagnostic3d.csv")
    current_scaling_validation_rows = _read_rows(
        logs / "candido_current_scaling_validation3d.csv"
    )
    current_voltage_sensitivity_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity3d.csv"
    )
    combined_current_voltage_sensitivity_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity_combined_charge3d.csv"
    )
    long_window_mass_budget_rows = _read_rows(
        logs / "candido_long_window_mass_budget3d.csv"
    )
    long_window_charge_budget_rows = _read_rows(
        logs / "candido_long_window_charge_budget3d.csv"
    )
    charge_subcycling_rows = _read_rows(
        logs / "candido_charge_subcycling_diagnostic3d.csv"
    )
    charge_conservative_bounding_rows = _read_rows(
        logs / "candido_charge_conservative_bounding_diagnostic3d.csv"
    )
    charge_combined_bounding_subcycling_rows = _read_rows(
        logs / "candido_charge_combined_bounding_subcycling3d.csv"
    )
    charge_limit_sensitivity_rows = _read_rows(
        logs / "candido_charge_limit_sensitivity3d.csv"
    )
    charge_reference_gap_rows = _read_rows(
        logs / "candido_charge_model_reference_gap3d.csv"
    )
    reduced_collector_current_rows = _read_rows(
        logs / "candido_reduced_collector_current_fixture3d.csv"
    )
    paper_charge_boundary_current_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity_paper_charge_boundary3d.csv"
    )
    paper_inlet_velocity_current_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity_paper_inlet_velocity3d.csv"
    )
    fig8b_current_blocker_rows = _read_rows(
        logs / "candido_fig8b_current_blocker3d.csv"
    )
    current_pareto_rows = _read_rows(
        logs / "candido_current_morphology_whip_pareto3d.csv"
    )
    open_boundary_current_rows = _read_rows(
        logs / "candido_open_boundary_current_diagnostic3d.csv"
    )
    moving_collector_boundary_rows = _read_rows(
        logs / "candido_moving_collector_boundary_diagnostic3d.csv"
    )
    paper_current_development_rows = _read_rows(
        logs / "candido_paper_current_development_tradeoff3d.csv"
    )
    preconditioned_current_plane_rows = _read_rows(
        logs / "candido_preconditioned_current_plane_diagnostic3d.csv"
    )
    poisson_face_total_current_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity_poisson_face_total3d.csv"
    )
    poisson_face_alpha05_total_current_rows = _read_rows(
        logs / "candido_current_voltage_sensitivity_poisson_face_alpha05_total3d.csv"
    )
    axial_developed_current_rows = _read_rows(
        logs / "candido_axial_developed_jet_current_window3d.csv"
    )
    poisson_face_convective_factorization_rows = _read_rows(
        logs / "candido_poisson_face_convective_factorization3d.csv"
    )
    poisson_face_velocity_projection_factorization_rows = _read_rows(
        logs / "candido_poisson_face_velocity_projection_factorization3d.csv"
    )
    momentum_source_factorization_rows = _read_rows(
        logs / "candido_momentum_source_factorization3d.csv"
    )
    boundary_current_sensitivity_rows = _read_rows(
        logs / "candido_boundary_current_sensitivity3d.csv"
    )
    interface_charge_transport_rows = _read_rows(
        logs / "candido_interface_charge_transport_diagnostic3d.csv"
    )
    post_charge_potential_refresh_rows = _read_rows(
        logs / "candido_post_charge_potential_refresh_diagnostic3d.csv"
    )
    conductivity_potential_charge_closure_rows = _read_rows(
        logs / "candido_conductivity_potential_charge_closure3d.csv"
    )
    conservative_surface_charge_closure_rows = _read_rows(
        logs / "candido_conservative_surface_charge_closure3d.csv"
    )
    electrode_surface_current_boundary_rows = _read_rows(
        logs / "candido_electrode_surface_current_boundary_isolation3d.csv"
    )
    interfacial_ohmic_charge_source_rows = _read_rows(
        logs / "candido_interfacial_ohmic_charge_source3d.csv"
    )
    whipping_rows = _read_rows(logs / "candido_whipping_diagnostic3d.csv")

    checks: dict[str, bool] = {
        "paper_md_library_ready": sum(1 for _ in papers.glob("*.md")) >= 11,
        "candido_smoke_csv_ready": len(smoke_rows) >= 2,
        "ca_e_validation_case_present": any(
            abs(_as_float(row, "computed_ca_e") - 0.25) < 1e-9 for row in smoke_rows
        ),
        "mass_conservation_ok": bool(smoke_rows)
        and max(abs(_as_float(row, "alpha_mass_drift")) for row in smoke_rows) <= 1e-12,
        "continuity_ok": bool(smoke_rows)
        and max(abs(_as_float(row, "max_div")) for row in smoke_rows) <= 1e-10,
        "electric_force_active": bool(smoke_rows)
        and max(_as_float(row, "max_electric_force") for row in smoke_rows) > 0.0,
        "csf_force_active": bool(smoke_rows)
        and max(_as_float(row, "max_csf_force") for row in smoke_rows) > 0.0,
        "charge_current_active": bool(smoke_rows)
        and max(_as_float(row, "max_conductive_current") for row in smoke_rows) > 0.0,
        "tip_motion_resolved": bool(smoke_rows)
        and max(abs(_as_float(row, "tip_displacement")) for row in smoke_rows) > 1e-8,
        "morphology_timeseries_present": (logs / "candido_morphology_timeseries3d.csv").exists(),
        "morphology_observable_audit_present": (
            logs / "candido_morphology_observable_audit3d.csv"
        ).exists(),
        "candido_refinement_sweep_present": (logs / "candido_refinement_sweep3d.csv").exists(),
        "candido_refinement_quality_present": (
            logs / "candido_refinement_quality3d.csv"
        ).exists(),
        "paper_current_scaling_present": (logs / "candido_current_scaling3d.csv").exists(),
        "paper_current_scaling_quantified": bool(current_scaling_validation_rows),
        "paper_current_scaling_order_of_magnitude": _current_scaling_order_of_magnitude(
            current_scaling_validation_rows
        ),
        "paper_current_voltage_sensitivity_ok": _current_voltage_sensitivity_ok(
            current_voltage_sensitivity_rows
        ),
        "combined_charge_current_voltage_sensitivity_ok": _current_voltage_sensitivity_ok(
            combined_current_voltage_sensitivity_rows
        ),
        "long_window_open_boundary_mass_budget_quantified": (
            _long_window_mass_budget_quantified(long_window_mass_budget_rows)
        ),
        "long_window_charge_budget_quantified": _long_window_charge_budget_quantified(
            long_window_charge_budget_rows
        ),
        "charge_subcycling_diagnostic_quantified": (
            _charge_subcycling_diagnostic_quantified(charge_subcycling_rows)
        ),
        "charge_conservative_bounding_diagnostic_quantified": (
            _charge_conservative_bounding_diagnostic_quantified(
                charge_conservative_bounding_rows
            )
        ),
        "charge_combined_bounding_subcycling_quantified": (
            _charge_conservative_bounding_diagnostic_quantified(
                charge_combined_bounding_subcycling_rows
            )
        ),
        "charge_limit_sensitivity_quantified": _charge_limit_sensitivity_quantified(
            charge_limit_sensitivity_rows
        ),
        "charge_reference_gap_documented": _charge_reference_gap_documented(
            charge_reference_gap_rows
        ),
        "reduced_collector_current_fixture_quantified": (
            _reduced_collector_current_fixture_quantified(reduced_collector_current_rows)
        ),
        "paper_charge_boundary_current_candidate_quantified": (
            _paper_charge_boundary_current_candidate_quantified(
                paper_charge_boundary_current_rows
            )
        ),
        "paper_inlet_velocity_current_candidate_quantified": (
            _paper_charge_boundary_current_candidate_quantified(
                paper_inlet_velocity_current_rows
            )
        ),
        "fig8b_current_blocker_documented": _fig8b_current_blocker_documented(
            fig8b_current_blocker_rows
        ),
        "current_morphology_whip_pareto_documented": (
            _current_pareto_tradeoff_documented(current_pareto_rows)
        ),
        "open_boundary_current_diagnostic_quantified": (
            _open_boundary_current_diagnostic_quantified(open_boundary_current_rows)
        ),
        "paper_current_development_tradeoff_quantified": (
            _paper_current_development_tradeoff_quantified(
                paper_current_development_rows
            )
        ),
        "preconditioned_current_plane_diagnostic_quantified": (
            _preconditioned_current_plane_diagnostic_quantified(
                preconditioned_current_plane_rows
            )
        ),
        "moving_collector_boundary_diagnostic_quantified": (
            _moving_collector_boundary_diagnostic_quantified(
                moving_collector_boundary_rows
            )
        ),
        "poisson_face_total_current_observable_quantified": (
            _poisson_face_current_observable_quantified(
                poisson_face_total_current_rows
            )
        ),
        "poisson_face_alpha05_total_current_observable_quantified": (
            _poisson_face_current_observable_quantified(
                poisson_face_alpha05_total_current_rows
            )
        ),
        "poisson_face_axial_current_window_quantified": (
            _poisson_face_axial_current_window_quantified(
                axial_developed_current_rows
            )
        ),
        "poisson_face_candidate_axial_windows_quantified": (
            _poisson_face_candidate_axial_windows_quantified(
                axial_developed_current_rows
            )
        ),
        "poisson_face_candidate_axial_convective_windows_quantified": (
            _poisson_face_candidate_axial_convective_windows_quantified(
                axial_developed_current_rows
            )
        ),
        "poisson_face_convective_factorization_quantified": (
            _poisson_face_convective_factorization_quantified(
                poisson_face_convective_factorization_rows
            )
        ),
        "poisson_face_velocity_projection_factorization_quantified": (
            _poisson_face_velocity_projection_factorization_quantified(
                poisson_face_velocity_projection_factorization_rows
            )
        ),
        "momentum_source_factorization_quantified": (
            _momentum_source_factorization_quantified(
                momentum_source_factorization_rows
            )
        ),
        "electric_drive_source_tradeoff_quantified": (
            _electric_drive_source_tradeoff_quantified(
                momentum_source_factorization_rows
            )
        ),
        "boundary_current_sensitivity_quantified": (
            _boundary_current_sensitivity_quantified(boundary_current_sensitivity_rows)
        ),
        "interface_charge_transport_diagnostic_quantified": (
            _interface_charge_transport_diagnostic_quantified(
                interface_charge_transport_rows
            )
        ),
        "post_charge_potential_refresh_diagnostic_quantified": (
            _post_charge_potential_refresh_diagnostic_quantified(
                post_charge_potential_refresh_rows
            )
        ),
        "conductivity_potential_charge_closure_quantified": (
            _conductivity_potential_charge_closure_quantified(
                conductivity_potential_charge_closure_rows
            )
        ),
        "conservative_surface_charge_closure_quantified": (
            _conservative_surface_charge_closure_quantified(
                conservative_surface_charge_closure_rows
            )
        ),
        "electrode_surface_current_boundary_isolation_quantified": (
            _electrode_surface_current_boundary_isolation_quantified(
                electrode_surface_current_boundary_rows
            )
        ),
        "interfacial_ohmic_charge_source_quantified": (
            _interfacial_ohmic_charge_source_quantified(
                interfacial_ohmic_charge_source_rows
            )
        ),
        "whipping_diagnostic_quantified": _whipping_diagnostic_quantified(whipping_rows),
        "whipping_positive_wave_translation": _whipping_positive_translation(whipping_rows),
        "conejet_bc_diagnostic_present": (logs / "candido_boundary_conditions3d.csv").exists(),
        "conejet_contact_angle_diagnostic_present": (
            logs / "candido_contact_angle_diagnostic3d.csv"
        ).exists(),
        "paper_validation_matrix_present": (docs / "candido_paper_validation_matrix.json").exists(),
        "morphology_error_vs_guo_present": (logs / "candido_guo_morphology_error3d.csv").exists(),
        "physical_time_progress_present": (logs / "candido_physical_time_progress3d.csv").exists(),
        "jet_radius_current_metrics_present": (logs / "candido_jet_current_metrics3d.csv").exists(),
        "morphology_reference_window_validated": _has_complete_morphology_window(morphology_rows),
        "morphology_error_quantified_vs_external_geometry": _has_quantified_morphology_error(
            morphology_rows
        ),
        "morphology_error_within_10_percent": _morphology_error_within_bar(
            morphology_rows, 10.0
        ),
        "morphology_connected_proxy_within_10_percent": (
            _morphology_connected_proxy_within_bar(morphology_rows, 10.0)
        ),
        "morphology_ray_alpha05_quantified": _morphology_ray_alpha05_quantified(
            morphology_rows
        ),
        "morphology_outer_envelope_alpha05_quantified": (
            _morphology_outer_envelope_alpha05_quantified(morphology_rows)
        ),
        "morphology_outer_envelope_alpha05_within_10_percent": (
            _morphology_outer_envelope_alpha05_within_bar(morphology_rows, 10.0)
        ),
        "morphology_tip_sync_within_10_percent": _morphology_tip_sync_within_bar(
            tip_sync_rows, 10.0
        ),
        "morphology_phase_lag_quantified": _morphology_phase_lag_quantified(
            phase_lag_rows
        ),
        "morphology_initial_time_origin_identified": _initial_time_origin_identified(
            morphology_rows, logs, 10.0
        ),
        "morphology_late_0_8_0_9_digitized": _late_morphology_digitized(morphology_rows),
        "morphology_late_blocker_documented": _late_morphology_blocker_documented(
            late_blocker_rows
        ),
        "morphology_late_source_audit_quantified": (
            _late_morphology_source_audit_quantified(late_source_audit_rows)
        ),
        "boundary_inlet_resolved": bool(boundary_rows)
        and max(_as_float(row, "inlet_candidate_faces", 0.0) for row in boundary_rows) > 0.0,
        "refinement_force_all_levels_active": _all_positive(refinement_rows, "max_electric_force"),
        "refinement_jet_radius_convergent": _refinement_radius_convergent(refinement_rows)
        and _quality_row_passes(refinement_quality_rows, "final_midplane_jet_radius"),
    }

    metrics: dict[str, int] = {name: int(ok) for name, ok in sorted(checks.items())}
    metrics["paper_validation_gap_count"] = sum(1 for ok in checks.values() if not ok)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
