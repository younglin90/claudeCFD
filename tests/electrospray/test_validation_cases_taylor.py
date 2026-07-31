from __future__ import annotations

from validation_cases_taylor import (
    run_all_taylor_cone_cases,
    run_taylor_angle_reference_case,
    run_taylor_cone_fit_case,
    run_taylor_cone_field_voltage_balance_case,
    run_taylor_cone_level_set_force_residual_case,
    run_taylor_cone_level_set_case,
    run_taylor_cone_static_balance_case,
    run_taylor_cone_voltage_ramp_balance_case,
)


def test_taylor_angle_reference_validation_case_passes() -> None:
    assert run_taylor_angle_reference_case().passed


def test_taylor_cone_level_set_validation_case_passes() -> None:
    assert run_taylor_cone_level_set_case().passed


def test_taylor_cone_fit_validation_case_passes() -> None:
    assert run_taylor_cone_fit_case().passed


def test_taylor_cone_static_balance_validation_case_passes() -> None:
    assert run_taylor_cone_static_balance_case().passed


def test_taylor_cone_field_voltage_balance_validation_case_passes() -> None:
    assert run_taylor_cone_field_voltage_balance_case().passed


def test_taylor_cone_level_set_force_residual_validation_case_passes() -> None:
    assert run_taylor_cone_level_set_force_residual_case().passed


def test_taylor_cone_voltage_ramp_balance_validation_case_passes() -> None:
    assert run_taylor_cone_voltage_ramp_balance_case().passed


def test_all_taylor_cone_cases_pass() -> None:
    results = run_all_taylor_cone_cases()

    assert len(results) == 7
    assert all(result.passed for result in results)
