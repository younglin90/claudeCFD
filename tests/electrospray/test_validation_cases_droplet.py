from __future__ import annotations

from validation_cases_droplet import (
    run_all_droplet_cases,
    run_droplet_axis_extent_case,
    run_droplet_deformation_parameter_case,
    run_droplet_point_deformation_case,
    run_droplet_small_deformation_scaling_case,
    run_droplet_surface_charge_trend_case,
    run_taylor_melcher_droplet_reference_case,
    run_taylor_melcher_transient_deformation_case,
)


def test_droplet_deformation_parameter_validation_case_passes() -> None:
    assert run_droplet_deformation_parameter_case().passed


def test_droplet_axis_extent_validation_case_passes() -> None:
    assert run_droplet_axis_extent_case().passed


def test_droplet_point_deformation_validation_case_passes() -> None:
    assert run_droplet_point_deformation_case().passed


def test_droplet_small_deformation_scaling_validation_case_passes() -> None:
    assert run_droplet_small_deformation_scaling_case().passed


def test_droplet_surface_charge_trend_validation_case_passes() -> None:
    assert run_droplet_surface_charge_trend_case().passed


def test_taylor_melcher_droplet_reference_validation_case_passes() -> None:
    assert run_taylor_melcher_droplet_reference_case().passed


def test_taylor_melcher_transient_deformation_validation_case_passes() -> None:
    assert run_taylor_melcher_transient_deformation_case().passed


def test_all_droplet_validation_cases_pass() -> None:
    results = run_all_droplet_cases()

    assert len(results) == 7
    assert all(result.passed for result in results)
