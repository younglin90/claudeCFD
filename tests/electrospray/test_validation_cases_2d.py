from __future__ import annotations

from validation_cases_2d import (
    run_2d_dielectric_strip_case,
    run_2d_parallel_plate_case,
    run_2d_top_bottom_dirichlet_case,
    run_2d_uniform_space_charge_case,
    run_all_2d_electrostatic_cases,
)


def test_2d_parallel_plate_validation_case_passes() -> None:
    result = run_2d_parallel_plate_case()

    assert result.case_id == "2d_parallel_plate"
    assert result.passed


def test_2d_dielectric_strip_validation_case_passes() -> None:
    result = run_2d_dielectric_strip_case()

    assert result.case_id == "2d_dielectric_strip"
    assert result.passed


def test_2d_top_bottom_dirichlet_validation_case_passes() -> None:
    result = run_2d_top_bottom_dirichlet_case()

    assert result.case_id == "2d_top_bottom_dirichlet"
    assert result.passed


def test_2d_uniform_space_charge_validation_case_passes() -> None:
    result = run_2d_uniform_space_charge_case()

    assert result.case_id == "2d_uniform_space_charge_poisson"
    assert result.passed


def test_all_2d_electrostatic_cases_pass() -> None:
    results = run_all_2d_electrostatic_cases()

    assert len(results) == 4
    assert all(result.passed for result in results)
