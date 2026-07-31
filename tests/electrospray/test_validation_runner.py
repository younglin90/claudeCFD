from __future__ import annotations

from validation_runner import (
    core_validation_markdown,
    core_validation_summary,
    run_core_validation_suite,
    run_validation_summary_accounting_case,
)


def test_core_validation_suite_runs_all_current_executable_cases() -> None:
    results = run_core_validation_suite()

    assert len(results) == 128
    assert all(result.passed for result in results)


def test_core_validation_suite_case_ids_are_unique_before_summary_aggregation() -> None:
    results = run_core_validation_suite()
    case_ids = [result.case_id for result in results]

    assert len(case_ids) == 128
    assert len(set(case_ids)) == len(case_ids)


def test_core_validation_summary_preserves_suite_case_id_order() -> None:
    results = run_core_validation_suite()
    summary = core_validation_summary()

    assert summary["executable_result_case_ids"] == [result.case_id for result in results]


def test_core_validation_summary_reports_all_executed_cases_passing() -> None:
    summary = core_validation_summary()

    assert summary["total_results"] == 128
    assert summary["passed_results"] == 128
    assert summary["executable_result_case_id_count"] == 128
    assert summary["unique_executable_result_case_id_count"] == 128
    assert summary["executable_result_case_id_status"] == "pass"
    assert len(summary["passed_executable_result_case_ids"]) == 128
    assert summary["passed_executable_result_case_id_count"] == 128
    assert summary["passed_executable_result_case_id_count"] == summary["passed_results"]
    assert summary["passed_executable_result_case_ids"] == summary["executable_result_case_ids"]
    assert summary["failed_executable_result_case_ids"] == []
    assert summary["failed_executable_result_case_id_count"] == 0
    assert len(summary["failed_executable_result_case_ids"]) == summary["failed_executable_result_case_id_count"]
    assert summary["failed_executable_result_case_id_count"] == summary["total_results"] - summary["passed_results"]
    assert summary["validation_result_accounting_status"] == "pass"
    assert summary["validation_summary_status"] == "pass"
    assert summary["validation_summary_failure_count"] == 0
    assert summary["validation_summary_failure_count"] == summary["total_results"] - summary["passed_results"]
    assert summary["validation_summary_pass_fraction"] == 1.0
    assert summary["validation_summary_pass_fraction"] == summary["passed_results"] / summary["total_results"]
    assert summary["validation_summary_component_statuses"] == {
        "result_accounting": "pass",
        "manifest_summary": "pass",
        "failure_count": "pass",
        "pass_fraction": "pass",
    }
    assert summary["validation_summary_component_status_keys"] == [
        "failure_count",
        "manifest_summary",
        "pass_fraction",
        "result_accounting",
    ]
    assert summary["validation_summary_component_status_keys"] == sorted(
        summary["validation_summary_component_statuses"]
    )
    assert summary["validation_summary_component_status_count"] == 4
    assert summary["validation_summary_component_status_count"] == len(summary["validation_summary_component_statuses"])
    assert summary["validation_summary_component_status_count_status"] == "pass"
    assert summary["validation_summary_component_status_pass_count"] == 4
    assert summary["validation_summary_component_status_pass_count"] == sum(
        1 for status in summary["validation_summary_component_statuses"].values() if status == "pass"
    )
    assert summary["validation_summary_component_status_failure_count"] == 0
    assert (
        summary["validation_summary_component_status_pass_count"]
        + summary["validation_summary_component_status_failure_count"]
        == summary["validation_summary_component_status_count"]
    )
    assert summary["validation_summary_component_status_pass_count"] == len(
        summary["validation_summary_component_status_keys"]
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_summary_component_status_pass_count"]
        == summary["validation_summary_component_status_count"]
        and summary["validation_summary_component_status_failure_count"] == 0
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_summary_component_status_schema_status"] == "pass"
        and summary["validation_summary_component_status_failure_count"] == 0
    )
    assert summary["validation_summary_component_health_status"] == "pass"
    assert (summary["validation_summary_component_health_status"] == "pass") is (
        summary["validation_summary_component_status_schema_status"] == "pass"
        and summary["validation_summary_component_status_failure_count"] == 0
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_summary_component_health_status"] == "pass"
    )
    assert summary["validation_summary_health_status"] == "pass"
    assert (summary["validation_summary_health_status"] == "pass") is (
        summary["validation_summary_status"] == "pass"
        and summary["validation_summary_component_health_status"] == "pass"
    )
    assert summary["validation_summary_health_inputs"] == [
        "validation_summary_status",
        "validation_summary_component_health_status",
    ]
    assert all(summary[key] == "pass" for key in summary["validation_summary_health_inputs"])
    assert summary["validation_summary_health_input_statuses"] == {
        key: summary[key] for key in summary["validation_summary_health_inputs"]
    }
    assert all(status == "pass" for status in summary["validation_summary_health_input_statuses"].values())
    assert summary["validation_summary_health_input_count"] == 2
    assert summary["validation_summary_health_input_count"] == len(summary["validation_summary_health_inputs"])
    assert summary["validation_summary_health_input_count"] == len(summary["validation_summary_health_input_statuses"])
    assert summary["validation_summary_health_input_count_status"] == "pass"
    assert summary["validation_summary_failed_health_inputs"] == []
    assert summary["validation_summary_health_input_failure_count"] == 0
    assert summary["validation_summary_health_input_failure_count"] == len(
        summary["validation_summary_failed_health_inputs"]
    )
    assert summary["validation_summary_failed_component_statuses"] == []
    assert summary["validation_summary_component_status_failure_count"] == sum(
        1 for status in summary["validation_summary_component_statuses"].values() if status != "pass"
    )
    assert summary["validation_summary_failed_component_statuses"] == [
        name for name, status in summary["validation_summary_component_statuses"].items() if status != "pass"
    ]
    assert summary["validation_summary_component_status_schema_status"] == "pass"
    assert (summary["validation_summary_component_status_schema_status"] == "pass") is (
        summary["validation_summary_component_status_count_status"] == "pass"
        and summary["validation_summary_component_status_keys"] == sorted(
            summary["validation_summary_component_statuses"]
        )
        and summary["validation_summary_component_status_pass_count"]
        + summary["validation_summary_component_status_failure_count"]
        == summary["validation_summary_component_status_count"]
    )
    assert all(status == "pass" for status in summary["validation_summary_component_statuses"].values())
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_summary_component_status_failure_count"] == 0
        and summary["validation_summary_failed_component_statuses"] == []
    )
    assert (summary["validation_summary_status"] == "pass") is all(
        summary["validation_summary_component_statuses"][key] == "pass"
        for key in summary["validation_summary_component_status_keys"]
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_summary_failure_count"] == 0 and summary["validation_summary_pass_fraction"] == 1.0
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["validation_result_accounting_status"] == "pass" and summary["manifest_summary_status"] == "pass"
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["failed_executable_result_case_id_count"] == 0
        and summary["passed_results"] == summary["total_results"]
        and summary["manifest_summary_status"] == "pass"
    )
    assert (summary["validation_summary_status"] == "pass") is (
        summary["total_results"] - summary["passed_results"] == 0
    )
    assert len(summary["executable_result_case_ids"]) == 128
    assert len(set(summary["executable_result_case_ids"])) == 128
    assert "3d_application_component_status_schema" in summary["executable_result_case_ids"]
    assert len(summary["manifest_case_ids"]) == 11
    assert set(summary["manifest_case_ids"]) == set(summary["manifest_case_status"])
    assert set(summary["manifest_case_dimensions"]) == set(summary["manifest_case_ids"])
    assert summary["manifest_dimension_counts"] == {"1D": 4, "2D": 2, "2D-axisymmetric": 2, "3D": 3}
    assert sum(summary["manifest_dimension_counts"].values()) == summary["manifest_case_count"]
    assert summary["covered_manifest_dimension_counts"] == summary["manifest_dimension_counts"]
    assert sum(summary["covered_manifest_dimension_counts"].values()) == summary["covered_manifest_case_count"]
    assert summary["manifest_dimension_coverage_status"] == "pass"
    assert set(summary["manifest_case_purposes"]) == set(summary["manifest_case_ids"])
    assert summary["manifest_metadata_status"] == "pass"
    assert summary["manifest_case_dimensions"]["2d_cone_jet"] == "2D-axisymmetric"
    assert summary["manifest_case_dimensions"]["3d_multi_emitter"] == "3D"
    assert summary["manifest_case_purposes"]["2d_cone_jet"] == "steady cone-jet observables"


def test_core_validation_summary_health_input_trace_is_self_consistent() -> None:
    summary = core_validation_summary()

    health_inputs = summary["validation_summary_health_inputs"]
    health_input_statuses = summary["validation_summary_health_input_statuses"]

    assert health_inputs == ["validation_summary_status", "validation_summary_component_health_status"]
    assert set(health_input_statuses) == set(health_inputs)
    assert summary["validation_summary_health_input_count"] == len(health_inputs)
    assert summary["validation_summary_health_input_count"] == len(health_input_statuses)
    assert summary["validation_summary_health_input_count_status"] == "pass"
    assert summary["validation_summary_failed_health_inputs"] == [
        name for name, status in health_input_statuses.items() if status != "pass"
    ]
    assert summary["validation_summary_health_input_failure_count"] == len(
        summary["validation_summary_failed_health_inputs"]
    )
    assert (summary["validation_summary_health_status"] == "pass") is (
        summary["validation_summary_health_input_count_status"] == "pass"
        and summary["validation_summary_health_input_failure_count"] == 0
    )


def test_core_validation_summary_exposes_complete_health_trace_keyset() -> None:
    summary = core_validation_summary()

    assert {
        "validation_summary_component_health_status",
        "validation_summary_health_status",
        "validation_summary_health_inputs",
        "validation_summary_health_input_statuses",
        "validation_summary_health_input_count",
        "validation_summary_health_input_count_status",
        "validation_summary_failed_health_inputs",
        "validation_summary_health_input_failure_count",
    }.issubset(summary)
    assert summary["manifest_case_purposes"]["3d_microthruster_performance"] == "microthruster performance metrics"
    assert summary["manifest_case_count"] == 11
    assert len(summary["required_manifest_case_ids"]) == summary["required_manifest_case_count"]
    assert summary["required_manifest_case_ids"] == [
        "1d_parallel_plate",
        "1d_dielectric_jump",
        "1d_charge_relaxation",
        "1d_maxwell_jump",
        "vof_interface_transport",
        "2d_droplet_deformation",
        "2d_taylor_cone",
        "2d_cone_jet",
        "3d_multi_emitter",
    ]
    assert len(summary["optional_manifest_case_ids"]) == summary["optional_manifest_case_count"]
    assert summary["optional_manifest_case_ids"] == ["3d_plume_impingement", "3d_microthruster_performance"]
    assert set(summary["required_manifest_case_ids"]).isdisjoint(summary["optional_manifest_case_ids"])
    assert set(summary["required_manifest_case_ids"]) | set(summary["optional_manifest_case_ids"]) == set(summary["manifest_case_ids"])
    assert summary["manifest_case_count"] == len(summary["manifest_case_status"])
    assert summary["required_manifest_case_count"] == 9
    assert summary["optional_manifest_case_count"] == 2
    assert summary["covered_required_manifest_case_count"] == 9
    assert summary["covered_optional_manifest_case_count"] == 2
    assert (
        summary["covered_required_manifest_case_count"] + summary["covered_optional_manifest_case_count"]
        == summary["covered_manifest_case_count"]
    )
    assert summary["required_manifest_coverage"] == 1.0
    assert summary["required_coverage"] == summary["required_manifest_coverage"]
    assert summary["required_cases"] == summary["required_manifest_case_count"]
    assert summary["all_required_passed"] is True
    assert summary["optional_manifest_coverage"] == 1.0
    assert (
        summary["required_manifest_coverage"]
        == summary["covered_required_manifest_case_count"] / summary["required_manifest_case_count"]
    )
    assert (
        summary["optional_manifest_coverage"]
        == summary["covered_optional_manifest_case_count"] / summary["optional_manifest_case_count"]
    )
    assert summary["required_manifest_coverage_status"] == "pass"
    assert summary["optional_manifest_coverage_status"] == "pass"
    assert summary["manifest_coverage_rollup_status"] == "pass"
    assert (
        summary["required_manifest_coverage_status"] == "pass"
        and summary["optional_manifest_coverage_status"] == "pass"
        and summary["manifest_dimension_coverage_status"] == "pass"
    ) is (summary["manifest_coverage_rollup_status"] == "pass")
    assert summary["manifest_coverage_rollup_status"] == summary["manifest_case_coverage_status"]
    assert (summary["manifest_coverage_rollup_status"] == "pass") is (
        summary["required_manifest_coverage"] == 1.0
        and summary["optional_manifest_coverage"] == 1.0
        and summary["executable_manifest_coverage"] == 1.0
    )
    assert summary["required_manifest_case_count"] + summary["optional_manifest_case_count"] == summary["manifest_case_count"]
    assert summary["manifest_case_count_status"] == "pass"
    assert summary["manifest_summary_status"] == "pass"
    assert summary["manifest_summary_status"] == summary["manifest_case_coverage_status"]
    assert (summary["manifest_summary_status"] == "pass") is all(
        summary[key] == "pass"
        for key in (
            "manifest_metadata_status",
            "manifest_case_count_status",
            "manifest_coverage_rollup_status",
            "manifest_case_coverage_status",
        )
    )
    assert summary["covered_manifest_case_count"] == 11
    assert summary["covered_manifest_case_count"] == sum(1 for covered in summary["manifest_case_status"].values() if covered)
    assert (summary["manifest_case_coverage_status"] == "pass") is (
        summary["covered_manifest_case_count"] == summary["manifest_case_count"]
    )
    assert summary["manifest_case_coverage_status"] == "pass"
    assert summary["executable_manifest_coverage"] == 1.0
    assert summary["executable_manifest_coverage"] == summary["covered_manifest_case_count"] / summary["manifest_case_count"]
    assert all(summary["manifest_case_status"].values())
    assert summary["manifest_case_status"]["1d_parallel_plate"] is True
    assert summary["manifest_case_status"]["3d_plume_impingement"] is True
    assert summary["manifest_case_status"]["3d_microthruster_performance"] is True
    assert summary["reduced_phase_pair_step_diagnostics"]["max_violation"] < 1.0e-15


def test_core_validation_summary_case_ids_match_suite_order() -> None:
    results = run_core_validation_suite()
    summary = core_validation_summary()

    assert summary["executable_result_case_ids"] == [result.case_id for result in results]


def test_validation_summary_accounting_case_passes() -> None:
    result = run_validation_summary_accounting_case()

    assert result.case_id == "validation_summary_accounting"
    assert result.passed


def test_core_validation_markdown_contains_current_executable_cases() -> None:
    report = core_validation_markdown()

    assert "- executable_result_case_id_count: 128" in report
    assert "- unique_executable_result_case_id_count: 128" in report
    assert "- executable_result_case_id_status: pass" in report
    assert "- passed_executable_result_case_id_count: 128" in report
    assert "- failed_executable_result_case_id_count: 0" in report
    assert "- validation_result_accounting_status: pass" in report
    assert "- validation_summary_status: pass" in report
    assert "- validation_summary_failure_count: 0" in report
    assert "- validation_summary_pass_fraction: 1.000000" in report
    assert "- validation_summary_component_statuses: {'result_accounting': 'pass'" in report
    assert "- validation_summary_component_status_keys: ['failure_count', 'manifest_summary', 'pass_fraction', 'result_accounting']" in report
    assert "- validation_summary_component_status_count: 4" in report
    assert "- validation_summary_component_status_count_status: pass" in report
    assert "- validation_summary_component_status_pass_count: 4" in report
    assert "- validation_summary_component_status_failure_count: 0" in report
    assert "- validation_summary_failed_component_statuses: []" in report
    assert "- validation_summary_component_status_schema_status: pass" in report
    assert "- validation_summary_component_health_status: pass" in report
    assert "- validation_summary_health_status: pass" in report
    assert "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']" in report
    assert "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'" in report
    assert "- validation_summary_health_input_count: 2" in report
    assert "- validation_summary_health_input_count_status: pass" in report
    assert "- validation_summary_failed_health_inputs: []" in report
    assert "- validation_summary_health_input_failure_count: 0" in report
    assert "- manifest_case_count: 11" in report
    assert "- manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in report
    assert "- required_manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in report
    assert "- optional_manifest_case_ids: 3d_plume_impingement, 3d_microthruster_performance" in report
    assert "- manifest_metadata_status: pass" in report
    assert "- manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in report
    assert "- covered_manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in report
    assert "- manifest_dimension_coverage_status: pass" in report
    assert "- required_manifest_case_count: 9" in report
    assert "- optional_manifest_case_count: 2" in report
    assert "- covered_required_manifest_case_count: 9" in report
    assert "- covered_optional_manifest_case_count: 2" in report
    assert "- required_manifest_coverage: 1.000000" in report
    assert "- optional_manifest_coverage: 1.000000" in report
    assert "- required_manifest_coverage_status: pass" in report
    assert "- optional_manifest_coverage_status: pass" in report
    assert "- manifest_coverage_rollup_status: pass" in report
    assert "- manifest_case_count_status: pass" in report
    assert "- manifest_summary_status: pass" in report
    assert "- covered_manifest_case_count: 11" in report
    assert "- manifest_case_coverage_status: pass" in report
    assert "- executable_manifest_coverage: 1.000000" in report
    assert "| 1d_parallel_plate | True |" in report
    assert "| 3d_plume_impingement | True |" in report
    assert "| 3d_microthruster_performance | True |" in report
    assert "| 2d_cone_jet | 2D-axisymmetric | steady cone-jet observables |" in report
    assert "| 3d_microthruster_performance | 3D | microthruster performance metrics |" in report
    assert "| reduced_step_diagnostic | value |" in report
    assert "| max_violation |" in report
    assert "| 1d_parallel_plate | PASS |" in report
    assert "| 1d_reduced_phase_pair_step | PASS |" in report
    assert "| 2d_parallel_plate | PASS |" in report
    assert "| 2d_droplet_deformation_parameter | PASS |" in report
    assert "| material_phase_pair_leaky_dielectric | PASS |" in report
    assert "| material_phase_pair_electrical_diagnostics | PASS |" in report
    assert "| material_phase_pair_harmonic_face | PASS |" in report
    assert "| 2d_capillary_axisymmetric_laplace | PASS |" in report
    assert "| 2d_interface_shear_traction | PASS |" in report
    assert "| 2d_interface_phase_pair_jumps | PASS |" in report
    assert "| 2d_taylor_cone_angle | PASS |" in report
    assert "| 2d_cone_jet_current | PASS |" in report
    assert "| 2d_cone_jet_stateful_evolution | PASS |" in report


def test_core_validation_markdown_lists_executable_accounting_before_manifest_summary() -> None:
    report = core_validation_markdown()

    assert report.index("- executable_result_case_id_count: 128") < report.index("- manifest_case_count: 11")
    assert report.index("- validation_summary_status: pass") < report.index("- validation_summary_health_status: pass")
    assert "| 2d_rayleigh_limit_charge | PASS |" in report
    assert "| 3d_multi_emitter_current_sharing | PASS |" in report
    assert "| 3d_plume_panel_impingement | PASS |" in report
    assert "| 3d_plume_surface_loading | PASS |" in report
    assert "| 3d_microthruster_operating_point | PASS |" in report
    assert "| 3d_application_effective_performance | PASS |" in report
    assert "| 3d_application_loss_accounting | PASS |" in report
    assert "| 3d_application_power_accounting | PASS |" in report
    assert "| 3d_application_contamination | PASS |" in report
    assert "| 3d_application_component_status_schema | PASS |" in report
