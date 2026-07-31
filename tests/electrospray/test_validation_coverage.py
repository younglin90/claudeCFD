from __future__ import annotations

import pytest

from validation_coverage import covered_manifest_case_ids, executable_manifest_coverage, manifest_case_status
from validation_manifest import ValidationResult, required_case_ids


def test_covered_manifest_case_ids_maps_passing_subcases_to_parent_cases() -> None:
    results = [
        ValidationResult("2d_droplet_axis_extents", True),
        ValidationResult("2d_droplet_point_deformation", False),
        ValidationResult("3d_plume_loss", True),
    ]

    assert covered_manifest_case_ids(results) == {"2d_droplet_deformation", "3d_plume_impingement"}


def test_executable_manifest_coverage_counts_required_parent_cases_only() -> None:
    results = [ValidationResult(case_id, True) for case_id in required_case_ids()]

    assert executable_manifest_coverage(results) == pytest.approx(1.0)


def test_manifest_case_status_marks_optional_plume_when_subcase_passes() -> None:
    results = [ValidationResult("3d_plume_loss", True)]
    status = manifest_case_status(results)

    assert status["3d_plume_impingement"] is True
    assert status["1d_parallel_plate"] is False


def test_manifest_case_status_marks_optional_microthruster_when_subcase_passes() -> None:
    results = [ValidationResult("3d_microthruster_operating_point", True)]
    status = manifest_case_status(results)

    assert status["3d_microthruster_performance"] is True
    assert executable_manifest_coverage(results) == pytest.approx(0.0)


def test_application_component_status_schema_maps_to_microthruster_manifest_case() -> None:
    results = [ValidationResult("3d_application_component_status_schema", True)]
    status = manifest_case_status(results)

    assert covered_manifest_case_ids(results) == {"3d_microthruster_performance"}
    assert status["3d_microthruster_performance"] is True


def test_confined_charge_subcase_maps_to_vof_interface_transport_manifest_case() -> None:
    results = [ValidationResult("1d_confined_charge_leakage_fraction", True)]
    status = manifest_case_status(results)

    assert covered_manifest_case_ids(results) == {"vof_interface_transport"}
    assert status["vof_interface_transport"] is True
