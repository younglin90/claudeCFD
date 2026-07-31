from __future__ import annotations

from validation_cases_rayleigh import (
    run_all_rayleigh_cases,
    run_rayleigh_fissility_case,
    run_rayleigh_instability_case,
    run_rayleigh_limit_case,
)


def test_rayleigh_limit_validation_case_passes() -> None:
    assert run_rayleigh_limit_case().passed


def test_rayleigh_fissility_validation_case_passes() -> None:
    assert run_rayleigh_fissility_case().passed


def test_rayleigh_instability_validation_case_passes() -> None:
    assert run_rayleigh_instability_case().passed


def test_all_rayleigh_validation_cases_pass() -> None:
    results = run_all_rayleigh_cases()

    assert len(results) == 3
    assert all(result.passed for result in results)
