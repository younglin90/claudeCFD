from __future__ import annotations

from validation_cases_capillary import (
    run_all_capillary_cases,
    run_axisymmetric_laplace_pressure_case,
    run_continuum_surface_force_case,
    run_laplace_pressure_case,
)


def test_laplace_pressure_validation_case_passes() -> None:
    assert run_laplace_pressure_case().passed


def test_axisymmetric_laplace_pressure_validation_case_passes() -> None:
    assert run_axisymmetric_laplace_pressure_case().passed


def test_continuum_surface_force_validation_case_passes() -> None:
    assert run_continuum_surface_force_case().passed


def test_all_capillary_validation_cases_pass() -> None:
    results = run_all_capillary_cases()

    assert len(results) == 3
    assert all(result.passed for result in results)
