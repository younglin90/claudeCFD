from __future__ import annotations

from validation_manifest import ValidationResult
from validation_report import count_failed_results, format_validation_markdown, format_validation_markdown_with_summary


def test_validation_markdown_report_is_stable_table() -> None:
    results = [
        ValidationResult("case_a", True, metric=1.0e-12, tolerance=1.0e-9),
        ValidationResult("case_b", False, metric=2.0e-2, tolerance=1.0e-3),
    ]

    report = format_validation_markdown(results)

    assert report.splitlines()[0] == "| case_id | status | metric | tolerance |"
    assert "| case_a | PASS | 1.000000e-12 | 1.000000e-09 |" in report
    assert "| case_b | FAIL | 2.000000e-02 | 1.000000e-03 |" in report


def test_count_failed_results_counts_only_failed_cases() -> None:
    results = [ValidationResult("a", True), ValidationResult("b", False), ValidationResult("c", False)]

    assert count_failed_results(results) == 2


def test_validation_markdown_with_summary_prepends_coverage_block() -> None:
    results = [ValidationResult("case_a", True, metric=0.0, tolerance=1.0)]
    summary = {
        "total_results": 1,
        "passed_results": 1,
        "executable_manifest_coverage": 1.0,
        "manifest_case_status": {"case_a_parent": True},
    }

    report = format_validation_markdown_with_summary(results, summary)

    assert report.startswith(
        "- total_results: 1\n"
        "- passed_results: 1\n"
        "- executable_result_case_id_count: unknown\n"
        "- unique_executable_result_case_id_count: unknown\n"
        "- executable_result_case_id_status: unknown\n"
        "- passed_executable_result_case_id_count: unknown\n"
        "- failed_executable_result_case_id_count: unknown\n"
        "- validation_result_accounting_status: unknown\n"
        "- validation_summary_status: unknown\n"
        "- validation_summary_failure_count: unknown\n"
        "- validation_summary_pass_fraction: 0.000000\n"
        "- validation_summary_component_statuses: unknown\n"
        "- validation_summary_component_status_keys: unknown\n"
        "- validation_summary_component_status_count: unknown\n"
        "- validation_summary_component_status_count_status: unknown\n"
        "- validation_summary_component_status_pass_count: unknown\n"
        "- validation_summary_component_status_failure_count: unknown\n"
        "- validation_summary_failed_component_statuses: unknown\n"
        "- validation_summary_component_status_schema_status: unknown\n"
        "- validation_summary_component_health_status: unknown\n"
        "- validation_summary_health_status: unknown\n"
        "- validation_summary_health_inputs: unknown\n"
        "- validation_summary_health_input_statuses: unknown\n"
        "- validation_summary_health_input_count: unknown\n"
        "- validation_summary_health_input_count_status: unknown\n"
        "- validation_summary_failed_health_inputs: unknown\n"
        "- validation_summary_health_input_failure_count: unknown\n"
        "- manifest_case_count: unknown\n"
        "- manifest_case_ids: unknown\n"
        "- required_manifest_case_ids: unknown\n"
        "- optional_manifest_case_ids: unknown\n"
        "- manifest_metadata_status: unknown\n"
        "- manifest_dimension_counts: unknown\n"
        "- covered_manifest_dimension_counts: unknown\n"
        "- manifest_dimension_coverage_status: unknown\n"
        "- required_manifest_case_count: unknown\n"
        "- optional_manifest_case_count: unknown\n"
        "- covered_required_manifest_case_count: unknown\n"
        "- covered_optional_manifest_case_count: unknown\n"
        "- required_manifest_coverage: 0.000000\n"
        "- optional_manifest_coverage: 0.000000\n"
        "- required_manifest_coverage_status: unknown\n"
        "- optional_manifest_coverage_status: unknown\n"
        "- manifest_coverage_rollup_status: unknown\n"
        "- manifest_case_count_status: unknown\n"
        "- manifest_summary_status: unknown\n"
        "- covered_manifest_case_count: unknown\n"
        "- manifest_case_coverage_status: unknown\n"
        "- executable_manifest_coverage: 1.000000\n"
    )
    assert "| case_a_parent | True |" in report
    assert "| case_a | PASS | 0.000000e+00 | 1.000000e+00 |" in report


def test_validation_markdown_summary_health_trace_order_is_stable() -> None:
    results = [ValidationResult("case_a", True, metric=0.0, tolerance=1.0)]
    summary = {
        "total_results": 1,
        "passed_results": 1,
        "executable_manifest_coverage": 1.0,
        "validation_summary_component_health_status": "pass",
        "validation_summary_health_status": "pass",
        "validation_summary_health_inputs": [
            "validation_summary_status",
            "validation_summary_component_health_status",
        ],
        "validation_summary_health_input_statuses": {
            "validation_summary_status": "pass",
            "validation_summary_component_health_status": "pass",
        },
        "validation_summary_health_input_count": 2,
        "validation_summary_health_input_count_status": "pass",
        "validation_summary_failed_health_inputs": [],
        "validation_summary_health_input_failure_count": 0,
    }

    lines = format_validation_markdown_with_summary(results, summary).splitlines()
    health_lines = [line for line in lines if line.startswith("- validation_summary_") and "health" in line]

    assert health_lines == [
        "- validation_summary_component_health_status: pass",
        "- validation_summary_health_status: pass",
        "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']",
        "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass', 'validation_summary_component_health_status': 'pass'}",
        "- validation_summary_health_input_count: 2",
        "- validation_summary_health_input_count_status: pass",
        "- validation_summary_failed_health_inputs: []",
        "- validation_summary_health_input_failure_count: 0",
    ]
