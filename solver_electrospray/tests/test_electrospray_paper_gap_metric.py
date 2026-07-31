#!/usr/bin/env python3
"""Regression checks for the Candido paper-gap metric log selection."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from pathlib import Path


def load_metric_module(repo: Path):
    path = repo / "apps" / "electrospray_paper_gap_metric.py"
    spec = importlib.util.spec_from_file_location("electrospray_paper_gap_metric", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def touch(path: Path, delay: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("case,status\n", encoding="utf-8")
    time.sleep(delay)


def zero(path: Path, delay: float = 0.02) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    time.sleep(delay)


def main() -> int:
    repo = Path(os.environ["FVM_SOURCE_DIR"])
    metric = load_metric_module(repo)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        root_marker = root / "benchmark_logs" / "candido_guo_morphology_error3d.csv"
        build_marker = root / "build" / "benchmark_logs" / "candido_guo_morphology_error3d.csv"

        touch(root_marker)
        touch(build_marker)
        selected = metric._select_logs_dir(root)
        assert selected == build_marker.parent, selected

        touch(root_marker)
        selected = metric._select_logs_dir(root)
        assert selected == root_marker.parent, selected

        zero(build_marker)
        zero(root / "build" / "benchmark_logs" / "candido_current_scaling3d.csv")
        zero(root / "build" / "benchmark_logs" / "candido_current_voltage_sensitivity3d.csv")
        selected = metric._select_logs_dir(root)
        assert selected == root_marker.parent, selected

        touch(build_marker)
        touch(root / "build" / "benchmark_logs" / "candido_current_scaling3d.csv")
        touch(root / "build" / "benchmark_logs" / "candido_current_voltage_sensitivity3d.csv")
        selected = metric._select_logs_dir(root)
        assert selected == build_marker.parent, selected

    blocked_conservative_surface_charge_row = {
        "case": "conservative_surface_charge_closure",
        "status": "BLOCKED_NO_AXIAL_DEVELOPED_JET_WINDOW",
        "baseline_low_developed_samples": "2",
        "baseline_high_developed_samples": "2",
        "candidate_low_developed_samples": "2",
        "candidate_high_developed_samples": "0",
        "baseline_axial_alpha05_current_ratio": "1.8",
        "candidate_axial_alpha05_current_ratio": "inf",
        "candidate_charge_ratio": "inf",
        "candidate_velocity_ratio": "inf",
        "candidate_electric_source_ratio": "inf",
        "candidate_low_implicit_ohmic_residual": "1e-12",
        "candidate_high_implicit_ohmic_residual": "1e-12",
        "candidate_low_charge_clamp_l1": "0",
        "candidate_high_charge_clamp_l1": "0",
        "candidate_low_redistribution_residual": "0",
        "candidate_high_redistribution_residual": "0",
        "candidate_low_redistribution_deficit_l1": "0",
        "candidate_high_redistribution_deficit_l1": "0",
        "candidate_low_relative_charge_budget_residual": "0.02",
        "candidate_high_relative_charge_budget_residual": "0.03",
        "candidate_low_alpha_mass_drift": "1e-14",
        "candidate_high_alpha_mass_drift": "1e-14",
        "candidate_low_max_div": "1e-12",
        "candidate_high_max_div": "1e-12",
        "candidate_low_morphology_error_percent": "33.0",
        "candidate_high_max_radial_asymmetry": "0.01",
    }
    assert metric._conservative_surface_charge_closure_quantified(
        [blocked_conservative_surface_charge_row]
    )

    def electrode_row(case: str) -> dict[str, str]:
        return {
            "case": case,
            "status": "DOWNGRADED_BOUNDARY_CURRENT_NOT_SOLE_LIMITER",
            "dominant_high_patch": "collector",
            "low_developed_samples": "3",
            "high_developed_samples": "3",
            "axial_alpha05_current_ratio": "2.1",
            "charge_ratio": "1.2",
            "velocity_ratio": "1.6",
            "low_total_cumulative_conductive_flux": "1e-12",
            "high_total_cumulative_conductive_flux": "2e-12",
            "total_cumulative_ratio": "2",
            "low_nozzle_cumulative_flux": "1e-13",
            "high_nozzle_cumulative_flux": "2e-13",
            "nozzle_ratio": "2",
            "low_collector_cumulative_flux": "8e-13",
            "high_collector_cumulative_flux": "1.8e-12",
            "collector_ratio": "2.25",
            "low_lateral_cumulative_flux": "1e-13",
            "high_lateral_cumulative_flux": "0",
            "lateral_ratio": "0",
            "nozzle_peak_ratio": "2",
            "collector_peak_ratio": "2.25",
            "lateral_peak_ratio": "0",
            "low_total_vs_paper_ratio": "1",
            "high_total_vs_paper_ratio": "1",
            "low_nozzle_vs_paper_ratio": "1",
            "high_nozzle_vs_paper_ratio": "1",
            "low_collector_vs_paper_ratio": "1",
            "high_collector_vs_paper_ratio": "1",
            "max_option_effect_deviation": "0",
            "low_relative_charge_budget_residual": "1e-14",
            "high_relative_charge_budget_residual": "1e-14",
            "low_alpha_mass_drift": "1e-15",
            "high_alpha_mass_drift": "1e-15",
            "low_max_div": "1e-12",
            "high_max_div": "1e-12",
        }

    assert metric._electrode_surface_current_boundary_isolation_quantified(
        [
            electrode_row("paper_charge_boundary"),
            electrode_row("nozzle_allowed_boundary"),
            electrode_row("collector_only_boundary"),
            electrode_row("implicit_filtered_paper_charge_boundary"),
            electrode_row("implicit_filtered_collector_only_boundary"),
        ]
    )

    interfacial_ohmic_row = {
        "case": "interfacial_ohmic_charge_source",
        "status": "DOWNGRADED_INTERFACIAL_OHMIC_SOURCE_DOES_NOT_REDUCE_CURRENT_SENSITIVITY",
        "baseline_low_developed_samples": "3",
        "baseline_high_developed_samples": "3",
        "candidate_low_developed_samples": "3",
        "candidate_high_developed_samples": "3",
        "baseline_axial_alpha05_current_ratio": "2.1",
        "candidate_axial_alpha05_current_ratio": "2.2",
        "candidate_charge_ratio": "1.3",
        "candidate_velocity_ratio": "1.6",
        "candidate_electric_source_ratio": "1.7",
        "candidate_low_source_cells": "40",
        "candidate_high_source_cells": "40",
        "candidate_low_max_source_density": "1e-5",
        "candidate_high_max_source_density": "2e-5",
        "candidate_low_applied_source_charge": "1e-12",
        "candidate_high_applied_source_charge": "2e-12",
        "candidate_low_source_clamp_l1": "0",
        "candidate_high_source_clamp_l1": "0",
        "candidate_low_relative_charge_budget_residual": "1e-12",
        "candidate_high_relative_charge_budget_residual": "1e-12",
        "candidate_low_alpha_mass_drift": "1e-15",
        "candidate_high_alpha_mass_drift": "1e-15",
        "candidate_low_max_div": "1e-12",
        "candidate_high_max_div": "1e-12",
        "candidate_low_post_source_potential_residual": "1e-12",
        "candidate_high_post_source_potential_residual": "1e-12",
        "candidate_low_post_source_gauss_residual": "0.1",
        "candidate_high_post_source_gauss_residual": "0.2",
    }
    assert metric._interfacial_ohmic_charge_source_quantified(
        [interfacial_ohmic_row]
    )

    print("paper_gap_metric_log_selection=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
