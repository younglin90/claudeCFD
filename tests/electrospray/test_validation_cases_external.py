from __future__ import annotations

from validation_cases_external import (
    run_all_external_benchmark_cases,
    run_das_saintillan_reduced_solver_comparison_case,
    run_external_numeric_benchmark_comparison_case,
    run_huh_wirz_reduced_solver_comparison_case,
)


def test_huh_wirz_reduced_solver_comparison_validation_case_passes() -> None:
    result = run_huh_wirz_reduced_solver_comparison_case()

    assert result.case_id == "external_huh_wirz_conejet_reduced_comparison"
    assert result.passed
    assert result.metric < 0.08


def test_das_saintillan_reduced_solver_comparison_validation_case_passes() -> None:
    result = run_das_saintillan_reduced_solver_comparison_case()

    assert result.case_id == "external_das_saintillan_droplet_reduced_comparison"
    assert result.passed
    assert result.metric < 0.08


def test_external_numeric_benchmark_comparison_validation_case_passes() -> None:
    result = run_external_numeric_benchmark_comparison_case()

    assert result.case_id == "external_numeric_benchmark_comparison"
    assert result.passed
    assert result.metric < 0.08


def test_all_external_benchmark_validation_cases_pass() -> None:
    results = run_all_external_benchmark_cases()

    assert len(results) == 3
    assert all(result.passed for result in results)
