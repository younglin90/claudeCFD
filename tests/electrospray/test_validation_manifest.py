from __future__ import annotations

import pytest

from validation_manifest import (
    VALIDATION_CASES,
    ValidationResult,
    all_required_cases_passed,
    cases_by_dimension,
    manifest_coverage,
    required_case_ids,
    validation_summary,
)


def test_manifest_contains_1d_2d_and_3d_validation_layers() -> None:
    dimensions = {case.dimension for case in VALIDATION_CASES}
    assert "1D" in dimensions
    assert "2D-axisymmetric" in dimensions
    assert "3D" in dimensions


def test_required_case_ids_exclude_optional_application_cases() -> None:
    ids = required_case_ids()
    assert "3d_multi_emitter" in ids
    assert "3d_plume_impingement" not in ids
    assert "3d_microthruster_performance" not in ids


def test_cases_by_dimension_filters_manifest() -> None:
    assert {case.case_id for case in cases_by_dimension("1D")} == {
        "1d_parallel_plate",
        "1d_dielectric_jump",
        "1d_charge_relaxation",
        "1d_maxwell_jump",
    }
    assert {case.case_id for case in cases_by_dimension("2D")} >= {
        "vof_interface_transport",
        "2d_droplet_deformation",
    }


def test_manifest_coverage_counts_required_cases_only() -> None:
    assert manifest_coverage(set()) == pytest.approx(0.0)
    assert manifest_coverage(set(required_case_ids())) == pytest.approx(1.0)
    assert manifest_coverage({"3d_plume_impingement"}) == pytest.approx(0.0)
    assert manifest_coverage({"3d_microthruster_performance"}) == pytest.approx(0.0)


def test_all_required_cases_passed_requires_every_required_result() -> None:
    partial = [ValidationResult("1d_parallel_plate", True)]
    complete = [ValidationResult(case_id, True) for case_id in required_case_ids()]

    assert not all_required_cases_passed(partial)
    assert all_required_cases_passed(complete)


def test_validation_summary_counts_passed_results_and_required_coverage() -> None:
    results = [
        ValidationResult("1d_parallel_plate", True, metric=1.0e-12, tolerance=1.0e-9),
        ValidationResult("1d_dielectric_jump", False, metric=0.1, tolerance=1.0e-6),
        ValidationResult("3d_plume_impingement", True),
    ]
    summary = validation_summary(results)

    assert summary["total_results"] == 3
    assert summary["passed_results"] == 2
    assert summary["required_coverage"] == pytest.approx(1.0 / len(required_case_ids()))
    assert summary["all_required_passed"] is False
