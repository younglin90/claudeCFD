from __future__ import annotations

from validation_cases_1d import (
    run_all_1d_cases,
    run_charge_relaxation_case,
    run_dielectric_jump_case,
    run_maxwell_jump_case,
    run_parallel_plate_case,
)
from validation_manifest import validation_summary


def test_parallel_plate_validation_case_passes() -> None:
    result = run_parallel_plate_case()

    assert result.case_id == "1d_parallel_plate"
    assert result.passed


def test_dielectric_jump_validation_case_passes() -> None:
    result = run_dielectric_jump_case()

    assert result.case_id == "1d_dielectric_jump"
    assert result.passed


def test_charge_relaxation_and_maxwell_jump_validation_cases_pass() -> None:
    assert run_charge_relaxation_case().passed
    assert run_maxwell_jump_case().passed


def test_all_1d_validation_cases_report_five_passing_results() -> None:
    results = run_all_1d_cases()
    summary = validation_summary(results)

    assert len(results) == 5
    assert summary["passed_results"] == 5
