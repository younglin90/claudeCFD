from __future__ import annotations

from validation_cases_regime import (
    run_all_regime_map_cases,
    run_regime_map_multi_regime_case,
    run_regime_map_voltage_trend_case,
)


def test_regime_map_multi_regime_validation_case_passes() -> None:
    result = run_regime_map_multi_regime_case()

    assert result.case_id == "2d_regime_map_multi_regime"
    assert result.passed


def test_regime_map_voltage_trend_validation_case_passes() -> None:
    result = run_regime_map_voltage_trend_case()

    assert result.case_id == "2d_regime_map_voltage_trend"
    assert result.passed


def test_all_regime_map_validation_cases_pass() -> None:
    results = run_all_regime_map_cases()

    assert len(results) == 2
    assert all(result.passed for result in results)
