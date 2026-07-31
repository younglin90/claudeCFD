from __future__ import annotations

from validation_cases_array import (
    run_all_array_cases,
    run_array_current_sharing_case,
    run_array_pairwise_current_reference_case,
    run_array_pitch_sweep_trend_case,
    run_array_shielding_case,
    run_square_array_geometry_case,
)


def test_array_current_sharing_validation_case_passes() -> None:
    assert run_array_current_sharing_case().passed


def test_array_shielding_validation_case_passes() -> None:
    assert run_array_shielding_case().passed


def test_square_array_geometry_validation_case_passes() -> None:
    assert run_square_array_geometry_case().passed


def test_array_pitch_sweep_trend_validation_case_passes() -> None:
    assert run_array_pitch_sweep_trend_case().passed


def test_array_pairwise_current_reference_validation_case_passes() -> None:
    assert run_array_pairwise_current_reference_case().passed


def test_all_array_validation_cases_pass() -> None:
    results = run_all_array_cases()

    assert len(results) == 5
    assert all(result.passed for result in results)
