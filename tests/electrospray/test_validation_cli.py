from __future__ import annotations

import json
from pathlib import Path

import pytest

import validation_cli
from validation_cli import main
from validation_artifacts import validation_summary_health_trace_is_current
from validation_runner import core_validation_summary, run_core_validation_suite

EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS = [
    "application_accounting",
    "application_closure_count",
    "application_effective_performance",
    "application_mass_accounting",
    "application_power_accounting",
    "application_source",
    "contamination_exposure",
    "microthruster_operating_point",
    "plume_current_accounting",
    "plume_surface_loading",
    "plume_thrust_accounting",
]


def test_validation_cli_prints_json_summary(capsys) -> None:
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["total_results"] == 128
    assert payload["passed_results"] == 128
    assert payload["executable_result_case_id_count"] == 128
    assert payload["unique_executable_result_case_id_count"] == 128
    assert payload["executable_result_case_id_status"] == "pass"
    assert len(payload["passed_executable_result_case_ids"]) == 128
    assert payload["passed_executable_result_case_id_count"] == 128
    assert payload["passed_executable_result_case_id_count"] == payload["passed_results"]
    assert payload["passed_executable_result_case_ids"] == payload["executable_result_case_ids"]
    assert payload["failed_executable_result_case_ids"] == []
    assert payload["failed_executable_result_case_id_count"] == 0
    assert len(payload["failed_executable_result_case_ids"]) == payload["failed_executable_result_case_id_count"]
    assert payload["failed_executable_result_case_id_count"] == payload["total_results"] - payload["passed_results"]
    assert payload["validation_result_accounting_status"] == "pass"
    assert payload["validation_summary_status"] == "pass"
    assert payload["validation_summary_failure_count"] == 0
    assert payload["validation_summary_failure_count"] == payload["total_results"] - payload["passed_results"]
    assert payload["validation_summary_pass_fraction"] == 1.0
    assert payload["validation_summary_pass_fraction"] == payload["passed_results"] / payload["total_results"]
    assert payload["validation_summary_component_statuses"] == {
        "result_accounting": "pass",
        "manifest_summary": "pass",
        "failure_count": "pass",
        "pass_fraction": "pass",
    }
    assert payload["validation_summary_component_status_keys"] == [
        "failure_count",
        "manifest_summary",
        "pass_fraction",
        "result_accounting",
    ]
    assert payload["validation_summary_component_status_keys"] == sorted(
        payload["validation_summary_component_statuses"]
    )
    assert payload["validation_summary_component_status_count"] == 4
    assert payload["validation_summary_component_status_count"] == len(payload["validation_summary_component_statuses"])
    assert payload["validation_summary_component_status_count_status"] == "pass"
    assert payload["validation_summary_component_status_pass_count"] == 4
    assert payload["validation_summary_component_status_pass_count"] == sum(
        1 for status in payload["validation_summary_component_statuses"].values() if status == "pass"
    )
    assert payload["validation_summary_component_status_failure_count"] == 0
    assert (
        payload["validation_summary_component_status_pass_count"]
        + payload["validation_summary_component_status_failure_count"]
        == payload["validation_summary_component_status_count"]
    )
    assert payload["validation_summary_component_status_pass_count"] == len(
        payload["validation_summary_component_status_keys"]
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_component_status_pass_count"]
        == payload["validation_summary_component_status_count"]
        and payload["validation_summary_component_status_failure_count"] == 0
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_component_status_schema_status"] == "pass"
        and payload["validation_summary_component_status_failure_count"] == 0
    )
    assert payload["validation_summary_component_health_status"] == "pass"
    assert (payload["validation_summary_component_health_status"] == "pass") is (
        payload["validation_summary_component_status_schema_status"] == "pass"
        and payload["validation_summary_component_status_failure_count"] == 0
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_component_health_status"] == "pass"
    )
    assert payload["validation_summary_health_status"] == "pass"
    assert (payload["validation_summary_health_status"] == "pass") is (
        payload["validation_summary_status"] == "pass"
        and payload["validation_summary_component_health_status"] == "pass"
    )
    assert payload["validation_summary_health_inputs"] == [
        "validation_summary_status",
        "validation_summary_component_health_status",
    ]
    assert all(payload[key] == "pass" for key in payload["validation_summary_health_inputs"])
    assert payload["validation_summary_health_input_statuses"] == {
        key: payload[key] for key in payload["validation_summary_health_inputs"]
    }
    assert all(status == "pass" for status in payload["validation_summary_health_input_statuses"].values())
    assert payload["validation_summary_health_input_count"] == 2
    assert payload["validation_summary_health_input_count"] == len(payload["validation_summary_health_inputs"])
    assert payload["validation_summary_health_input_count"] == len(payload["validation_summary_health_input_statuses"])
    assert payload["validation_summary_health_input_count_status"] == "pass"
    assert payload["validation_summary_failed_health_inputs"] == []
    assert payload["validation_summary_health_input_failure_count"] == 0
    assert payload["validation_summary_health_input_failure_count"] == len(
        payload["validation_summary_failed_health_inputs"]
    )
    assert payload["validation_summary_failed_component_statuses"] == []
    assert payload["validation_summary_component_status_failure_count"] == sum(
        1 for status in payload["validation_summary_component_statuses"].values() if status != "pass"
    )
    assert payload["validation_summary_failed_component_statuses"] == [
        name for name, status in payload["validation_summary_component_statuses"].items() if status != "pass"
    ]
    assert payload["validation_summary_component_status_schema_status"] == "pass"
    assert (payload["validation_summary_component_status_schema_status"] == "pass") is (
        payload["validation_summary_component_status_count_status"] == "pass"
        and payload["validation_summary_component_status_keys"] == sorted(
            payload["validation_summary_component_statuses"]
        )
        and payload["validation_summary_component_status_pass_count"]
        + payload["validation_summary_component_status_failure_count"]
        == payload["validation_summary_component_status_count"]
    )
    assert all(status == "pass" for status in payload["validation_summary_component_statuses"].values())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_component_status_failure_count"] == 0
        and payload["validation_summary_failed_component_statuses"] == []
    )
    assert (payload["validation_summary_status"] == "pass") is all(
        payload["validation_summary_component_statuses"][key] == "pass"
        for key in payload["validation_summary_component_status_keys"]
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_failure_count"] == 0 and payload["validation_summary_pass_fraction"] == 1.0
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_result_accounting_status"] == "pass" and payload["manifest_summary_status"] == "pass"
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["failed_executable_result_case_id_count"] == 0
        and payload["passed_results"] == payload["total_results"]
        and payload["manifest_summary_status"] == "pass"
    )
    assert (payload["validation_summary_status"] == "pass") is (
        payload["total_results"] - payload["passed_results"] == 0
    )
    assert len(payload["executable_result_case_ids"]) == 128
    assert len(set(payload["executable_result_case_ids"])) == 128
    assert "3d_application_component_status_schema" in payload["executable_result_case_ids"]
    assert len(payload["manifest_case_ids"]) == 11
    assert set(payload["manifest_case_ids"]) == set(payload["manifest_case_status"])
    assert set(payload["manifest_case_dimensions"]) == set(payload["manifest_case_ids"])
    assert payload["manifest_dimension_counts"] == {"1D": 4, "2D": 2, "2D-axisymmetric": 2, "3D": 3}
    assert sum(payload["manifest_dimension_counts"].values()) == payload["manifest_case_count"]
    assert payload["covered_manifest_dimension_counts"] == payload["manifest_dimension_counts"]
    assert sum(payload["covered_manifest_dimension_counts"].values()) == payload["covered_manifest_case_count"]
    assert payload["manifest_dimension_coverage_status"] == "pass"
    assert set(payload["manifest_case_purposes"]) == set(payload["manifest_case_ids"])
    assert payload["manifest_metadata_status"] == "pass"
    assert payload["manifest_case_dimensions"]["2d_cone_jet"] == "2D-axisymmetric"
    assert payload["manifest_case_dimensions"]["3d_multi_emitter"] == "3D"
    assert payload["manifest_case_purposes"]["2d_cone_jet"] == "steady cone-jet observables"
    assert payload["manifest_case_purposes"]["3d_microthruster_performance"] == "microthruster performance metrics"
    assert payload["manifest_case_count"] == 11
    assert len(payload["required_manifest_case_ids"]) == payload["required_manifest_case_count"]
    assert payload["required_manifest_case_ids"] == [
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
    assert len(payload["optional_manifest_case_ids"]) == payload["optional_manifest_case_count"]
    assert payload["optional_manifest_case_ids"] == ["3d_plume_impingement", "3d_microthruster_performance"]
    assert set(payload["required_manifest_case_ids"]).isdisjoint(payload["optional_manifest_case_ids"])
    assert set(payload["required_manifest_case_ids"]) | set(payload["optional_manifest_case_ids"]) == set(payload["manifest_case_ids"])
    assert payload["manifest_case_count"] == len(payload["manifest_case_status"])
    assert payload["required_manifest_case_count"] == 9
    assert payload["optional_manifest_case_count"] == 2
    assert payload["covered_required_manifest_case_count"] == 9
    assert payload["covered_optional_manifest_case_count"] == 2
    assert (
        payload["covered_required_manifest_case_count"] + payload["covered_optional_manifest_case_count"]
        == payload["covered_manifest_case_count"]
    )
    assert payload["required_manifest_coverage"] == 1.0
    assert payload["optional_manifest_coverage"] == 1.0
    assert (
        payload["required_manifest_coverage"]
        == payload["covered_required_manifest_case_count"] / payload["required_manifest_case_count"]
    )
    assert (
        payload["optional_manifest_coverage"]
        == payload["covered_optional_manifest_case_count"] / payload["optional_manifest_case_count"]
    )
    assert payload["required_manifest_coverage_status"] == "pass"
    assert payload["optional_manifest_coverage_status"] == "pass"
    assert payload["manifest_coverage_rollup_status"] == "pass"
    assert (
        payload["required_manifest_coverage_status"] == "pass"
        and payload["optional_manifest_coverage_status"] == "pass"
        and payload["manifest_dimension_coverage_status"] == "pass"
    ) is (payload["manifest_coverage_rollup_status"] == "pass")
    assert payload["manifest_coverage_rollup_status"] == payload["manifest_case_coverage_status"]
    assert (payload["manifest_coverage_rollup_status"] == "pass") is (
        payload["required_manifest_coverage"] == 1.0
        and payload["optional_manifest_coverage"] == 1.0
        and payload["executable_manifest_coverage"] == 1.0
    )
    assert payload["required_manifest_case_count"] + payload["optional_manifest_case_count"] == payload["manifest_case_count"]
    assert payload["manifest_case_count_status"] == "pass"
    assert payload["manifest_summary_status"] == "pass"
    assert payload["manifest_summary_status"] == payload["manifest_case_coverage_status"]
    assert (payload["manifest_summary_status"] == "pass") is all(
        payload[key] == "pass"
        for key in (
            "manifest_metadata_status",
            "manifest_case_count_status",
            "manifest_coverage_rollup_status",
            "manifest_case_coverage_status",
        )
    )
    assert payload["covered_manifest_case_count"] == 11
    assert payload["covered_manifest_case_count"] == sum(1 for covered in payload["manifest_case_status"].values() if covered)
    assert (payload["manifest_case_coverage_status"] == "pass") is (
        payload["covered_manifest_case_count"] == payload["manifest_case_count"]
    )
    assert payload["manifest_case_coverage_status"] == "pass"
    assert payload["executable_manifest_coverage"] == payload["covered_manifest_case_count"] / payload["manifest_case_count"]
    assert all(payload["manifest_case_status"].values())
    assert payload["manifest_case_status"]["3d_multi_emitter"] is True


def test_validation_cli_json_summary_health_input_trace_is_self_consistent(capsys) -> None:
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    health_inputs = payload["validation_summary_health_inputs"]
    health_input_statuses = payload["validation_summary_health_input_statuses"]

    assert health_inputs == ["validation_summary_status", "validation_summary_component_health_status"]
    assert set(health_input_statuses) == set(health_inputs)
    assert payload["validation_summary_health_input_count"] == len(health_inputs)
    assert payload["validation_summary_health_input_count"] == len(health_input_statuses)
    assert payload["validation_summary_health_input_count_status"] == "pass"
    assert payload["validation_summary_failed_health_inputs"] == [
        name for name, status in health_input_statuses.items() if status != "pass"
    ]
    assert payload["validation_summary_health_input_failure_count"] == len(
        payload["validation_summary_failed_health_inputs"]
    )
    assert (payload["validation_summary_health_status"] == "pass") is (
        payload["validation_summary_health_input_count_status"] == "pass"
        and payload["validation_summary_health_input_failure_count"] == 0
    )


def test_validation_cli_json_summary_preserves_suite_case_id_order(capsys) -> None:
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["executable_result_case_ids"] == [
        result.case_id for result in run_core_validation_suite()
    ]


def test_validation_cli_json_summary_exposes_complete_health_trace_keyset(capsys) -> None:
    assert main(["--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert {
        "validation_summary_component_health_status",
        "validation_summary_health_status",
        "validation_summary_health_inputs",
        "validation_summary_health_input_statuses",
        "validation_summary_health_input_count",
        "validation_summary_health_input_count_status",
        "validation_summary_failed_health_inputs",
        "validation_summary_health_input_failure_count",
    }.issubset(payload)


def test_validation_cli_prints_markdown_report(capsys) -> None:
    assert main(["--format", "markdown"]) == 0
    output = capsys.readouterr().out

    assert output.startswith("- total_results: 128")
    assert "- executable_result_case_id_count: 128" in output
    assert "- unique_executable_result_case_id_count: 128" in output
    assert "- executable_result_case_id_status: pass" in output
    assert "- passed_executable_result_case_id_count: 128" in output
    assert "- failed_executable_result_case_id_count: 0" in output
    assert "- validation_result_accounting_status: pass" in output
    assert "- validation_summary_status: pass" in output
    assert "- validation_summary_failure_count: 0" in output
    assert "- validation_summary_pass_fraction: 1.000000" in output
    assert "- validation_summary_component_statuses: {'result_accounting': 'pass'" in output
    assert "- validation_summary_component_status_keys: ['failure_count', 'manifest_summary', 'pass_fraction', 'result_accounting']" in output
    assert "- validation_summary_component_status_count: 4" in output
    assert "- validation_summary_component_status_count_status: pass" in output
    assert "- validation_summary_component_status_pass_count: 4" in output
    assert "- validation_summary_component_status_failure_count: 0" in output
    assert "- validation_summary_failed_component_statuses: []" in output
    assert "- validation_summary_component_status_schema_status: pass" in output
    assert "- validation_summary_component_health_status: pass" in output
    assert "- validation_summary_health_status: pass" in output
    assert "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']" in output
    assert "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'" in output
    assert "- validation_summary_health_input_count: 2" in output
    assert "- validation_summary_health_input_count_status: pass" in output
    assert "- validation_summary_failed_health_inputs: []" in output
    assert "- validation_summary_health_input_failure_count: 0" in output
    assert "- manifest_case_count: 11" in output
    assert "- manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in output
    assert "- required_manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in output
    assert "- optional_manifest_case_ids: 3d_plume_impingement, 3d_microthruster_performance" in output
    assert "- manifest_metadata_status: pass" in output
    assert "- manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in output
    assert "- covered_manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in output
    assert "- manifest_dimension_coverage_status: pass" in output
    assert "- required_manifest_case_count: 9" in output
    assert "- optional_manifest_case_count: 2" in output
    assert "- covered_required_manifest_case_count: 9" in output
    assert "- covered_optional_manifest_case_count: 2" in output
    assert "- required_manifest_coverage: 1.000000" in output
    assert "- optional_manifest_coverage: 1.000000" in output
    assert "- required_manifest_coverage_status: pass" in output
    assert "- optional_manifest_coverage_status: pass" in output
    assert "- manifest_coverage_rollup_status: pass" in output
    assert "- manifest_case_count_status: pass" in output
    assert "- manifest_summary_status: pass" in output
    assert "- covered_manifest_case_count: 11" in output
    assert "- manifest_case_coverage_status: pass" in output
    assert "- executable_manifest_coverage: 1.000000" in output
    assert "| 2d_cone_jet | 2D-axisymmetric | steady cone-jet observables |" in output
    assert "| 3d_microthruster_performance | 3D | microthruster performance metrics |" in output
    assert "| 3d_plume_panel_impingement | PASS |" in output
    assert "| 3d_application_effective_performance | PASS |" in output
    assert "| 3d_application_loss_accounting | PASS |" in output
    assert "| 3d_application_power_accounting | PASS |" in output
    assert "| 3d_application_contamination | PASS |" in output
    assert "| 3d_application_component_status_schema | PASS |" in output


def test_validation_cli_markdown_summary_health_input_trace_is_visible(capsys) -> None:
    assert main(["--format", "markdown"]) == 0
    output = capsys.readouterr().out

    assert "- validation_summary_health_status: pass" in output
    assert "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']" in output
    assert "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'" in output
    assert "- validation_summary_health_input_count: 2" in output
    assert "- validation_summary_health_input_count_status: pass" in output
    assert "- validation_summary_failed_health_inputs: []" in output
    assert "- validation_summary_health_input_failure_count: 0" in output


def test_validation_cli_markdown_lists_executable_accounting_before_manifest_summary(capsys) -> None:
    assert main(["--format", "markdown"]) == 0
    output = capsys.readouterr().out

    assert output.index("- executable_result_case_id_count: 128") < output.index("- manifest_case_count: 11")
    assert output.index("- validation_summary_status: pass") < output.index("- validation_summary_health_status: pass")


def test_validation_cli_prints_reduced_step_json_report(capsys) -> None:
    assert main(["--format", "reduced-step-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["case_id"] == "1d_reduced_phase_pair_step"
    assert payload["passed"] is True
    assert payload["diagnostics"]["max_gas_charge_density"] == 0.0


def test_validation_cli_writes_reduced_step_json_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "reduced_phase_pair_step_report.json"

    assert main(["--format", "reduced-step-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["case_id"] == "1d_reduced_phase_pair_step"
    assert payload["passed"] is True
    assert payload["diagnostics"]["max_gas_charge_density"] == 0.0


def test_validation_cli_prints_plume_json_report(capsys) -> None:
    assert main(["--format", "plume-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["impingement_fraction"] == 0.25
    assert payload["deposited_current_fraction"] == 0.25
    assert payload["panel_current_density_balance_error"] == pytest.approx(0.0)
    assert payload["retained_current"] == pytest.approx(9.0e-6)
    assert payload["retained_current_fraction"] == 0.75
    assert payload["current_fraction_balance_error"] == 0.0
    assert payload["plume_current_accounting_status"] == "pass"
    assert payload["retained_thrust_fraction"] == 0.8
    assert payload["thrust_fraction_balance_error"] == 0.0
    assert payload["plume_thrust_accounting_status"] == "pass"
    assert payload["closure_residual_count"] == 3
    assert payload["plume_surface_loading_status"] == "pass"


def test_validation_cli_writes_plume_json_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "plume_impingement_report.json"

    assert main(["--format", "plume-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["plume_current_accounting_status"] == "pass"
    assert payload["plume_thrust_accounting_status"] == "pass"
    assert payload["plume_surface_loading_status"] == "pass"
    assert payload["closure_residual_count"] == 3


def test_validation_cli_prints_microthruster_json_report(capsys) -> None:
    assert main(["--format", "microthruster-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["propellant"] == "validation"
    assert payload["mass_flow_balance_error"] == pytest.approx(0.0)
    assert payload["charge_to_mass_balance_error"] == pytest.approx(0.0)
    assert payload["exhaust_velocity_balance_error"] == pytest.approx(0.0)
    assert payload["ideal_efficiency"] == 1.0
    assert payload["thrust_momentum_balance_error"] == pytest.approx(0.0)
    assert payload["specific_impulse_balance_error"] == pytest.approx(0.0)
    assert payload["power_efficiency_balance_error"] == pytest.approx(0.0)
    assert payload["thrust_to_power_balance_error"] == pytest.approx(0.0)
    assert payload["closure_residual_count"] == 7
    assert payload["microthruster_operating_point_status"] == "pass"


def test_validation_cli_writes_microthruster_json_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "microthruster_operating_point_report.json"

    assert main(["--format", "microthruster-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["mass_flow_balance_error"] == pytest.approx(0.0)
    assert payload["charge_to_mass_balance_error"] == pytest.approx(0.0)
    assert payload["closure_residual_count"] == 7
    assert payload["microthruster_operating_point_status"] == "pass"


def test_validation_cli_prints_application_json_report(capsys) -> None:
    assert main(["--format", "application-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["plume_impingement"]["impingement_fraction"] == 0.25
    assert payload["microthruster_operating_point"]["propellant"] == "validation"
    assert payload["application_component_statuses"] == {
        "application_accounting": "pass",
        "application_closure_count": "pass",
        "application_effective_performance": "pass",
        "application_mass_accounting": "pass",
        "application_power_accounting": "pass",
        "application_source": "pass",
        "contamination_exposure": "pass",
        "microthruster_operating_point": "pass",
        "plume_current_accounting": "pass",
        "plume_surface_loading": "pass",
        "plume_thrust_accounting": "pass",
    }
    assert payload["application_component_status_keys"] == sorted(payload["application_component_statuses"])
    assert payload["application_component_status_expected_count"] == 11
    assert payload["application_component_status_count"] == 11
    assert payload["application_component_status_pass_count"] == 11
    assert payload["application_component_status_failure_count"] == 0
    assert payload["application_failed_component_statuses"] == []
    assert payload["application_component_status_count_status"] == "pass"
    assert payload["application_component_status_schema_status"] == "pass"
    assert payload["plume_current_accounting_status"] == "pass"
    assert payload["plume_thrust_accounting_status"] == "pass"
    assert payload["plume_surface_loading_status"] == "pass"
    assert payload["plume_closure_residual_count"] == 3
    assert payload["microthruster_operating_point_status"] == "pass"
    assert payload["microthruster_closure_residual_count"] == 7
    assert payload["application_total_closure_residual_count"] == 10
    assert payload["application_closure_count_status"] == "pass"
    assert payload["application_source_status"] == "pass"
    assert payload["effective_retained_thrust_fraction_after_impingement"] == pytest.approx(0.8)
    assert payload["retained_thrust_fraction_balance_error"] == pytest.approx(0.0)
    assert payload["effective_thrust_to_power_after_impingement"] == 0.0016
    assert payload["application_effective_performance_status"] == "pass"
    assert payload["panel_current_density_after_impingement"] == pytest.approx(1.5e-5)
    assert payload["panel_mass_flux"] == 3e-08
    assert payload["panel_deposited_charge_to_mass"] == pytest.approx(500.0)
    assert payload["panel_charge_to_mass_balance_error"] == pytest.approx(0.0)
    assert payload["panel_mass_loading"] == 0.000108
    assert payload["panel_charge_loading"] == pytest.approx(5.4e-2)
    assert payload["panel_charge_loading_balance_error"] == pytest.approx(0.0)
    assert payload["time_to_contamination_limit"] == 33333.333333333336
    assert payload["contamination_exposure_margin"] == 9.25925925925926
    assert payload["contamination_exposure_status"] == "pass"
    assert payload["application_accounting_status"] == "pass"
    assert payload["application_validation_status"] == "pass"
    assert payload["deposited_current_after_impingement"] == pytest.approx(3.0e-7)
    assert payload["deposited_electrical_power_after_impingement"] == pytest.approx(3.0e-4)
    assert payload["deposited_kinetic_power_after_impingement"] == pytest.approx(3.0e-4)
    assert payload["deposited_kinetic_power_fraction_after_impingement"] == pytest.approx(0.25)
    assert payload["deposited_power_efficiency_after_impingement"] == pytest.approx(1.0)
    assert payload["deposited_power_fraction_after_impingement"] == pytest.approx(0.25)
    assert payload["deposited_current_fraction"] == 0.25
    assert payload["deposited_mass_flow_fraction"] == 0.25
    assert payload["current_fraction_balance_error"] == 0.0
    assert payload["mass_flow_fraction_balance_error"] == 0.0
    assert payload["application_mass_accounting_status"] == "pass"
    assert payload["power_fraction_balance_error"] == 0.0
    assert payload["kinetic_power_fraction_balance_error"] == 0.0
    assert payload["application_power_accounting_status"] == "pass"
    assert payload["effective_thrust_to_retained_power_after_impingement"] == pytest.approx(0.0021333333333333334)
    assert payload["retained_current_after_impingement"] == 9e-07
    assert payload["retained_current_fraction_after_impingement"] == 0.75
    assert payload["retained_electrical_power_after_impingement"] == 0.0009
    assert payload["retained_kinetic_power_after_impingement"] == 0.0009
    assert payload["retained_kinetic_power_fraction_after_impingement"] == pytest.approx(0.75)
    assert payload["retained_power_efficiency_after_impingement"] == 1.0
    assert payload["retained_power_fraction_after_impingement"] == pytest.approx(0.75)
    assert payload["retained_mass_flow_after_impingement"] == 1.8e-09
    assert payload["retained_mass_flow_fraction_after_impingement"] == 0.75
    assert payload["retained_charge_to_mass_after_impingement"] == 500.0


def test_validation_cli_writes_application_json_component_status_trace_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "application_report.json"

    assert main(["--format", "application-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["application_component_status_count"] == len(payload["application_component_statuses"])
    assert payload["application_component_status_pass_count"] == len(payload["application_component_statuses"])
    assert payload["application_component_status_failure_count"] == 0
    assert payload["application_component_status_schema_status"] == "pass"


def test_validation_cli_application_component_status_trace_is_self_consistent(capsys) -> None:
    assert main(["--format", "application-json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    statuses = payload["application_component_statuses"]
    failed = [name for name, status in statuses.items() if status != "pass"]

    assert payload["application_component_status_count"] == len(statuses)
    assert payload["application_component_status_pass_count"] == sum(1 for status in statuses.values() if status == "pass")
    assert payload["application_component_status_failure_count"] == len(failed)
    assert payload["application_failed_component_statuses"] == failed
    assert payload["application_component_status_count_status"] == "pass"


def test_validation_cli_application_component_status_trace_matches_top_level_fields(capsys) -> None:
    assert main(["--format", "application-json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    statuses = payload["application_component_statuses"]

    assert statuses["application_accounting"] == payload["application_accounting_status"]
    assert statuses["application_closure_count"] == payload["application_closure_count_status"]
    assert statuses["application_effective_performance"] == payload["application_effective_performance_status"]
    assert statuses["application_mass_accounting"] == payload["application_mass_accounting_status"]
    assert statuses["application_power_accounting"] == payload["application_power_accounting_status"]
    assert statuses["application_source"] == payload["application_source_status"]
    assert statuses["contamination_exposure"] == payload["contamination_exposure_status"]
    assert statuses["microthruster_operating_point"] == payload["microthruster_operating_point_status"]
    assert statuses["plume_current_accounting"] == payload["plume_current_accounting_status"]
    assert statuses["plume_surface_loading"] == payload["plume_surface_loading_status"]
    assert statuses["plume_thrust_accounting"] == payload["plume_thrust_accounting_status"]


def test_validation_cli_application_component_status_keys_match_expected_schema(capsys) -> None:
    assert main(["--format", "application-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["application_component_status_keys"] == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert payload["application_component_status_expected_count"] == len(EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS)
    assert sorted(payload["application_component_statuses"]) == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert payload["application_component_status_schema_status"] == "pass"


def test_validation_cli_prints_artifact_status_json(capsys) -> None:
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is True
    assert payload["validation_summary_health_trace"] is True
    assert payload["application_report"] is True


def test_validation_cli_artifact_status_json_keyset_is_stable(capsys) -> None:
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert sorted(payload) == [
        "application_report",
        "das_saintillan_droplet_benchmark_metadata",
        "external_benchmark_readiness_report",
        "field_contours",
        "full_cfd_readiness",
        "huh_wirz_conejet_benchmark_metadata",
        "manuscript_figures",
        "manuscript_tables",
        "microthruster_operating_point_report",
        "plume_impingement_report",
        "reduced_phase_pair_step_report",
        "submission_claim_audit",
        "submission_readiness_matrix",
        "validation_artifacts",
        "validation_summary_health_trace",
    ]
    assert all(payload.values())


def test_validation_cli_artifact_status_json_values_are_booleans(capsys) -> None:
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert all(isinstance(value, bool) for value in payload.values())


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_incomplete_validation_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text(json.dumps({"validation_summary_health_status": "pass"}) + "\n")
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_malformed_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("{not-json\n")
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_non_object_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("[]\n")
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_directory_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").mkdir()
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_directory_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").mkdir()
    (docs / "validation_summary.json").write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_missing_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "validation_report.md").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_artifact_status_json_reports_health_trace_false_for_missing_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "validation_summary.json").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_artifacts"] is False
    assert payload["validation_summary_health_trace"] is False


def test_validation_cli_writes_report_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    text = output_path.read_text()
    assert text.startswith("- total_results: 128")
    assert "- executable_result_case_id_count: 128" in text
    assert "- unique_executable_result_case_id_count: 128" in text
    assert "- executable_result_case_id_status: pass" in text
    assert "- passed_executable_result_case_id_count: 128" in text
    assert "- failed_executable_result_case_id_count: 0" in text
    assert "- validation_result_accounting_status: pass" in text
    assert "- validation_summary_status: pass" in text
    assert "- validation_summary_failure_count: 0" in text
    assert "- validation_summary_pass_fraction: 1.000000" in text
    assert "- validation_summary_component_statuses: {'result_accounting': 'pass'" in text
    assert "- validation_summary_component_status_keys: ['failure_count', 'manifest_summary', 'pass_fraction', 'result_accounting']" in text
    assert "- validation_summary_component_status_count: 4" in text
    assert "- validation_summary_component_status_count_status: pass" in text
    assert "- validation_summary_component_status_pass_count: 4" in text
    assert "- validation_summary_component_status_failure_count: 0" in text
    assert "- validation_summary_failed_component_statuses: []" in text
    assert "- validation_summary_component_status_schema_status: pass" in text
    assert "- validation_summary_component_health_status: pass" in text
    assert "- validation_summary_health_status: pass" in text
    assert "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']" in text
    assert "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'" in text
    assert "- validation_summary_health_input_count: 2" in text
    assert "- validation_summary_health_input_count_status: pass" in text
    assert "- validation_summary_failed_health_inputs: []" in text
    assert "- validation_summary_health_input_failure_count: 0" in text
    assert "- manifest_case_count: 11" in text
    assert "- manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in text
    assert "- required_manifest_case_ids: 1d_parallel_plate, 1d_dielectric_jump" in text
    assert "- optional_manifest_case_ids: 3d_plume_impingement, 3d_microthruster_performance" in text
    assert "- manifest_metadata_status: pass" in text
    assert "- manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in text
    assert "- covered_manifest_dimension_counts: {'1D': 4, '2D': 2, '2D-axisymmetric': 2, '3D': 3}" in text
    assert "- manifest_dimension_coverage_status: pass" in text
    assert "- required_manifest_case_count: 9" in text
    assert "- optional_manifest_case_count: 2" in text
    assert "- covered_required_manifest_case_count: 9" in text
    assert "- covered_optional_manifest_case_count: 2" in text
    assert "- required_manifest_coverage: 1.000000" in text
    assert "- optional_manifest_coverage: 1.000000" in text
    assert "- required_manifest_coverage_status: pass" in text
    assert "- optional_manifest_coverage_status: pass" in text
    assert "- manifest_coverage_rollup_status: pass" in text
    assert "- manifest_case_count_status: pass" in text
    assert "- manifest_summary_status: pass" in text
    assert "- covered_manifest_case_count: 11" in text
    assert "- manifest_case_coverage_status: pass" in text
    assert "| 1d_parallel_plate | PASS |" in text
    assert "| 3d_application_effective_performance | PASS |" in text
    assert "| 3d_application_loss_accounting | PASS |" in text
    assert "| 3d_application_power_accounting | PASS |" in text
    assert "| 3d_application_contamination | PASS |" in text
    assert "| 3d_application_component_status_schema | PASS |" in text


def test_validation_cli_markdown_output_file_preserves_accounting_order(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    text = output_path.read_text()
    assert text.index("- executable_result_case_id_count: 128") < text.index("- manifest_case_count: 11")
    assert text.index("- validation_summary_status: pass") < text.index("- validation_summary_health_status: pass")


def test_validation_cli_write_artifacts_creates_all_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0

    assert capsys.readouterr().out == "validation artifacts written\n"
    assert (tmp_path / "docs" / "electrospray" / "validation_report.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "validation_summary.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "plume_impingement_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "application_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "huh_wirz_conejet_benchmark_metadata.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "das_saintillan_droplet_benchmark_metadata.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "external_benchmark_readiness_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "submission_claim_audit.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "submission_readiness_matrix.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "full_cfd_readiness_report.json").exists()
    assert (tmp_path / "docs" / "electrospray" / "full_cfd_readiness_gates.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "cone_jet_error_budget_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "external_benchmark_numeric_comparison_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "full_cfd_huh_wirz_nonbreakup_comparison_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "full_cfd_huh_wirz_subgrid_breakup_comparison_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "taylor_cone_voltage_ramp_balance_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "coupled_droplet_grid_refinement_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "dielectric_maxwell_droplet_history_table.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "field_contour_manifest.md").exists()
    assert (tmp_path / "docs" / "electrospray" / "figures" / "cone_jet_error_budget.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "figures" / "taylor_cone_voltage_ramp.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "figures" / "coupled_droplet_grid_refinement.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "figures" / "external_benchmark_numeric_comparison.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "contours" / "full_cfd_timestep_contract" / "pressure.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "contours" / "full_cfd_timestep_contract" / "velocity_magnitude.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "contours" / "full_cfd_timestep_contract" / "temperature_isothermal_K.png").exists()
    assert (tmp_path / "docs" / "electrospray" / "contours" / "full_cfd_timestep_contract" / "density.png").exists()
    assert validation_summary_health_trace_is_current(
        tmp_path / "docs" / "electrospray" / "validation_report.md",
        tmp_path / "docs" / "electrospray" / "validation_summary.json",
    )


def test_validation_cli_write_artifacts_creates_regular_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    artifact_dir = tmp_path / "docs" / "electrospray"

    assert all(
        (artifact_dir / name).is_file()
        for name in (
            "validation_report.md",
            "validation_summary.json",
            "reduced_phase_pair_step_report.json",
            "plume_impingement_report.json",
            "microthruster_operating_point_report.json",
            "application_report.json",
            "das_saintillan_droplet_benchmark_metadata.json",
            "external_benchmark_readiness_report.json",
            "huh_wirz_conejet_benchmark_metadata.json",
            "submission_claim_audit.json",
            "submission_readiness_matrix.md",
            "full_cfd_readiness_report.json",
            "full_cfd_readiness_gates.md",
            "cone_jet_error_budget_table.md",
            "external_benchmark_numeric_comparison_table.md",
            "full_cfd_huh_wirz_nonbreakup_comparison_table.md",
            "full_cfd_huh_wirz_subgrid_breakup_comparison_table.md",
            "taylor_cone_voltage_ramp_balance_table.md",
            "coupled_droplet_grid_refinement_table.md",
            "dielectric_maxwell_droplet_history_table.md",
            "field_contour_manifest.md",
        )
    )
    assert all(
        (artifact_dir / "figures" / name).is_file()
        for name in (
            "cone_jet_error_budget.png",
            "taylor_cone_voltage_ramp.png",
            "coupled_droplet_grid_refinement.png",
            "external_benchmark_numeric_comparison.png",
        )
    )


def test_validation_cli_write_artifacts_creates_documented_filenames(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    artifact_dir = tmp_path / "docs" / "electrospray"

    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "application_report.json",
        "cone_jet_error_budget_table.md",
        "contours",
        "coupled_droplet_grid_refinement_table.md",
        "das_saintillan_droplet_benchmark_metadata.json",
        "dielectric_maxwell_droplet_history_table.md",
        "external_benchmark_numeric_comparison_table.md",
        "external_benchmark_readiness_report.json",
        "field_contour_manifest.md",
        "figures",
        "full_cfd_huh_wirz_nonbreakup_comparison_table.md",
        "full_cfd_huh_wirz_subgrid_breakup_comparison_table.md",
        "full_cfd_readiness_gates.md",
        "full_cfd_readiness_report.json",
        "huh_wirz_conejet_benchmark_metadata.json",
        "huh_wirz_same_path_grid_refinement_table.md",
        "microthruster_operating_point_report.json",
        "plume_impingement_report.json",
        "reduced_phase_pair_step_report.json",
        "submission_claim_audit.json",
        "submission_readiness_matrix.md",
        "taylor_cone_voltage_ramp_balance_table.md",
        "validation_report.md",
        "validation_summary.json",
    ]


def test_validation_cli_write_artifacts_uses_docs_electrospray_directory(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    artifact_dir = tmp_path / "docs" / "electrospray"

    assert artifact_dir.is_dir()
    assert {path.parent for path in artifact_dir.iterdir()} == {artifact_dir}


def test_validation_cli_artifact_status_json_reports_written_health_trace_current(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_summary_health_trace"] is True
    assert all(payload.values())


def test_validation_cli_artifact_status_json_reports_reduced_step_false_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["reduced_phase_pair_step_report"] is False


def test_validation_cli_artifact_status_json_reports_reduced_step_false_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    report_path = tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json"
    report_path.unlink()
    report_path.mkdir()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["reduced_phase_pair_step_report"] is False


def test_validation_cli_artifact_status_json_reports_reduced_step_false_for_stale_content(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json").write_text("{}\n")
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["reduced_phase_pair_step_report"] is False


def test_validation_cli_artifact_status_json_reports_plume_false_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "plume_impingement_report.json").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["plume_impingement_report"] is False


def test_validation_cli_artifact_status_json_reports_plume_false_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    report_path = tmp_path / "docs" / "electrospray" / "plume_impingement_report.json"
    report_path.unlink()
    report_path.mkdir()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["plume_impingement_report"] is False


def test_validation_cli_artifact_status_json_reports_plume_false_for_stale_content(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "plume_impingement_report.json").write_text("{}\n")
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["plume_impingement_report"] is False


def test_validation_cli_artifact_status_json_reports_microthruster_false_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["microthruster_operating_point_report"] is False


def test_validation_cli_artifact_status_json_reports_microthruster_false_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    report_path = tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json"
    report_path.unlink()
    report_path.mkdir()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["microthruster_operating_point_report"] is False


def test_validation_cli_artifact_status_json_reports_microthruster_false_for_stale_content(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json").write_text("{}\n")
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["microthruster_operating_point_report"] is False


def test_validation_cli_artifact_status_json_reports_application_false_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "application_report.json").unlink()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["application_report"] is False


def test_validation_cli_artifact_status_json_reports_application_false_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    report_path = tmp_path / "docs" / "electrospray" / "application_report.json"
    report_path.unlink()
    report_path.mkdir()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["application_report"] is False


def test_validation_cli_artifact_status_json_reports_application_false_for_stale_content(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    (tmp_path / "docs" / "electrospray" / "application_report.json").write_text("{}\n")
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["application_report"] is False


def test_validation_cli_written_artifact_status_json_keyset_is_stable(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert sorted(payload) == [
        "application_report",
        "das_saintillan_droplet_benchmark_metadata",
        "external_benchmark_readiness_report",
        "field_contours",
        "full_cfd_readiness",
        "huh_wirz_conejet_benchmark_metadata",
        "manuscript_figures",
        "manuscript_tables",
        "microthruster_operating_point_report",
        "plume_impingement_report",
        "reduced_phase_pair_step_report",
        "submission_claim_audit",
        "submission_readiness_matrix",
        "validation_artifacts",
        "validation_summary_health_trace",
    ]


def test_validation_cli_written_artifact_status_json_reports_all_current(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--format", "artifact-status-json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == {
        "application_report": True,
        "das_saintillan_droplet_benchmark_metadata": True,
        "external_benchmark_readiness_report": True,
        "field_contours": True,
        "full_cfd_readiness": True,
        "huh_wirz_conejet_benchmark_metadata": True,
        "manuscript_figures": True,
        "manuscript_tables": True,
        "microthruster_operating_point_report": True,
        "plume_impingement_report": True,
        "reduced_phase_pair_step_report": True,
        "submission_claim_audit": True,
        "submission_readiness_matrix": True,
        "validation_artifacts": True,
        "validation_summary_health_trace": True,
    }


def test_validation_cli_writes_artifact_status_json_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "artifact_status.json"

    assert main(["--format", "artifact-status-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_trace"] is True
    assert all(payload.values())


def test_validation_cli_writes_artifact_status_json_object_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "artifact_status.json"

    assert main(["--format", "artifact-status-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert isinstance(json.loads(output_path.read_text()), dict)


def test_validation_cli_writes_artifact_status_boolean_values_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "artifact_status.json"

    assert main(["--format", "artifact-status-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert all(isinstance(value, bool) for value in payload.values())


def test_validation_cli_writes_artifact_status_stable_keyset_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "artifact_status.json"

    assert main(["--format", "artifact-status-json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert sorted(payload) == [
        "application_report",
        "das_saintillan_droplet_benchmark_metadata",
        "external_benchmark_readiness_report",
        "field_contours",
        "full_cfd_readiness",
        "huh_wirz_conejet_benchmark_metadata",
        "manuscript_figures",
        "manuscript_tables",
        "microthruster_operating_point_report",
        "plume_impingement_report",
        "reduced_phase_pair_step_report",
        "submission_claim_audit",
        "submission_readiness_matrix",
        "validation_artifacts",
        "validation_summary_health_trace",
    ]


def test_validation_cli_json_output_file_preserves_suite_case_id_order(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["executable_result_case_ids"] == [
        result.case_id for result in run_core_validation_suite()
    ]


def test_validation_cli_writes_json_summary_object_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert isinstance(json.loads(output_path.read_text()), dict)


def test_validation_cli_writes_json_summary_health_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_status"] == "pass"


def test_validation_cli_writes_json_summary_health_status_inputs_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_health_status"] == "pass") is (
        payload["validation_summary_status"] == "pass"
        and payload["validation_summary_component_health_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_health_inputs_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_inputs"] == [
        "validation_summary_status",
        "validation_summary_component_health_status",
    ]


def test_validation_cli_writes_json_summary_health_input_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_statuses"] == {
        key: payload[key] for key in payload["validation_summary_health_inputs"]
    }


def test_validation_cli_writes_json_summary_health_input_status_values_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_statuses"] == {
        "validation_summary_status": "pass",
        "validation_summary_component_health_status": "pass",
    }


def test_validation_cli_writes_json_summary_health_input_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_count"] == len(
        payload["validation_summary_health_inputs"]
    )


def test_validation_cli_writes_json_summary_health_input_count_value_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_count"] == 2


def test_validation_cli_writes_json_summary_health_input_count_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_count_status"] == "pass"


def test_validation_cli_writes_json_summary_health_input_count_status_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_health_input_count_status"] == "pass") is (
        payload["validation_summary_health_input_count"] == len(payload["validation_summary_health_inputs"])
        == len(payload["validation_summary_health_input_statuses"])
    )


def test_validation_cli_writes_json_summary_failed_health_inputs_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_failed_health_inputs"] == []


def test_validation_cli_writes_json_summary_health_input_failure_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_health_input_failure_count"] == len(
        payload["validation_summary_failed_health_inputs"]
    )


def test_validation_cli_writes_json_summary_component_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_statuses"] == {
        "failure_count": "pass",
        "manifest_summary": "pass",
        "pass_fraction": "pass",
        "result_accounting": "pass",
    }


def test_validation_cli_writes_json_summary_result_accounting_component_status_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["validation_summary_component_statuses"]["result_accounting"]
        == payload["validation_result_accounting_status"]
    )


def test_validation_cli_writes_json_summary_pass_fraction_component_status_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_component_statuses"]["pass_fraction"] == "pass") is (
        payload["validation_summary_pass_fraction"] == 1.0
    )


def test_validation_cli_writes_json_summary_failure_count_component_status_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_component_statuses"]["failure_count"] == "pass") is (
        payload["validation_summary_failure_count"] == 0
    )


def test_validation_cli_writes_json_summary_manifest_summary_component_status_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["validation_summary_component_statuses"]["manifest_summary"]
        == payload["manifest_summary_status"]
    )


def test_validation_cli_writes_json_summary_component_status_keys_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_keys"] == sorted(
        payload["validation_summary_component_statuses"]
    )


def test_validation_cli_writes_json_summary_component_status_key_values_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_keys"] == [
        "failure_count",
        "manifest_summary",
        "pass_fraction",
        "result_accounting",
    ]


def test_validation_cli_writes_json_summary_component_status_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_count"] == len(
        payload["validation_summary_component_statuses"]
    )


def test_validation_cli_writes_json_summary_component_status_count_value_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_count"] == 4


def test_validation_cli_writes_json_summary_component_status_pass_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_pass_count"] == sum(
        1 for status in payload["validation_summary_component_statuses"].values() if status == "pass"
    )


def test_validation_cli_writes_json_summary_component_status_pass_count_value_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_pass_count"] == 4


def test_validation_cli_writes_json_summary_component_status_failure_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_failure_count"] == sum(
        1 for status in payload["validation_summary_component_statuses"].values() if status != "pass"
    )


def test_validation_cli_writes_json_summary_component_status_failure_count_value_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_failure_count"] == 0


def test_validation_cli_writes_json_summary_component_status_schema_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_status_schema_status"] == "pass"


def test_validation_cli_writes_json_summary_component_status_schema_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_component_status_schema_status"] == "pass") is (
        payload["validation_summary_component_status_keys"] == sorted(
            payload["validation_summary_component_statuses"]
        )
        and payload["validation_summary_component_status_count"]
        == len(payload["validation_summary_component_statuses"])
        and payload["validation_summary_component_status_pass_count"]
        + payload["validation_summary_component_status_failure_count"]
        == payload["validation_summary_component_status_count"]
    )


def test_validation_cli_writes_json_summary_component_health_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_component_health_status"] == "pass"


def test_validation_cli_writes_json_summary_component_health_status_inputs_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_component_health_status"] == "pass") is (
        payload["validation_summary_component_status_schema_status"] == "pass"
        and payload["validation_summary_component_status_failure_count"] == 0
    )


def test_validation_cli_writes_json_summary_component_status_count_balance_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["validation_summary_component_status_pass_count"]
        + payload["validation_summary_component_status_failure_count"]
        == payload["validation_summary_component_status_count"]
    )


def test_validation_cli_writes_json_summary_failed_component_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_failed_component_statuses"] == [
        name for name, status in payload["validation_summary_component_statuses"].items() if status != "pass"
    ]


def test_validation_cli_writes_json_summary_status_failed_component_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_failed_component_statuses"] == []
    )


def test_validation_cli_writes_json_summary_status_component_health_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_component_health_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_health_status_inputs_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_health_status"] == "pass") is (
        payload["validation_summary_status"] == "pass"
        and payload["validation_summary_component_health_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_status"] == "pass"


def test_validation_cli_writes_json_summary_pass_fraction_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_pass_fraction"] == 1.0


def test_validation_cli_writes_json_summary_pass_fraction_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_pass_fraction"] == payload["passed_results"] / payload["total_results"]


def test_validation_cli_writes_json_summary_fraction_failure_status_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_summary_failure_count"] == 0
        and payload["validation_summary_pass_fraction"] == 1.0
    )


def test_validation_cli_writes_json_summary_result_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["passed_results"] == payload["total_results"]


def test_validation_cli_writes_json_summary_result_accounting_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_result_accounting_status"] == "pass"


def test_validation_cli_writes_json_summary_result_accounting_count_balance_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_result_accounting_status"] == "pass") is (
        payload["passed_executable_result_case_id_count"]
        + payload["failed_executable_result_case_id_count"]
        == payload["total_results"]
    )


def test_validation_cli_writes_json_summary_result_accounting_zero_failure_balance_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["passed_results"] == payload["total_results"]) is (
        payload["failed_executable_result_case_id_count"] == 0
    )


def test_validation_cli_writes_json_summary_manifest_summary_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_summary_status"] == "pass"


def test_validation_cli_writes_json_summary_manifest_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_count"] == len(payload["manifest_case_status"])


def test_validation_cli_writes_json_summary_manifest_case_count_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_count_status"] == "pass"


def test_validation_cli_writes_json_summary_covered_manifest_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["covered_manifest_case_count"] == sum(
        1 for covered in payload["manifest_case_status"].values() if covered
    )


def test_validation_cli_writes_json_summary_manifest_case_coverage_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_coverage_status"] == "pass"


def test_validation_cli_writes_json_summary_manifest_coverage_rollup_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_coverage_rollup_status"] == "pass"


def test_validation_cli_writes_json_summary_manifest_coverage_rollup_matches_case_coverage_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_coverage_rollup_status"] == payload["manifest_case_coverage_status"]


def test_validation_cli_writes_json_summary_executable_manifest_coverage_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["executable_manifest_coverage"] == (
        payload["covered_manifest_case_count"] / payload["manifest_case_count"]
    )


def test_validation_cli_writes_json_summary_required_manifest_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["required_manifest_case_count"] == len(payload["required_manifest_case_ids"])


def test_validation_cli_writes_json_summary_optional_manifest_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["optional_manifest_case_count"] == len(payload["optional_manifest_case_ids"])


def test_validation_cli_writes_json_summary_required_optional_manifest_disjoint_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert set(payload["required_manifest_case_ids"]).isdisjoint(payload["optional_manifest_case_ids"])


def test_validation_cli_writes_json_summary_required_optional_manifest_partition_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert set(payload["required_manifest_case_ids"]) | set(payload["optional_manifest_case_ids"]) == set(
        payload["manifest_case_ids"]
    )


def test_validation_cli_writes_json_summary_required_optional_manifest_count_sum_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["required_manifest_case_count"] + payload["optional_manifest_case_count"]
        == payload["manifest_case_count"]
    )


def test_validation_cli_writes_json_summary_covered_required_optional_manifest_count_sum_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["covered_required_manifest_case_count"] + payload["covered_optional_manifest_case_count"]
        == payload["covered_manifest_case_count"]
    )


def test_validation_cli_writes_json_summary_required_manifest_coverage_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["required_manifest_coverage"] == (
        payload["covered_required_manifest_case_count"] / payload["required_manifest_case_count"]
    )


def test_validation_cli_writes_json_summary_optional_manifest_coverage_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["optional_manifest_coverage"] == (
        payload["covered_optional_manifest_case_count"] / payload["optional_manifest_case_count"]
    )


def test_validation_cli_writes_json_summary_required_manifest_coverage_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["required_manifest_coverage_status"] == "pass"


def test_validation_cli_writes_json_summary_optional_manifest_coverage_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["optional_manifest_coverage_status"] == "pass"


def test_validation_cli_writes_json_summary_manifest_coverage_rollup_inputs_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_coverage_rollup_status"] == "pass") is (
        payload["required_manifest_coverage_status"] == "pass"
        and payload["optional_manifest_coverage_status"] == "pass"
        and payload["manifest_dimension_coverage_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_manifest_dimension_counts_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert sum(payload["manifest_dimension_counts"].values()) == payload["manifest_case_count"]


def test_validation_cli_writes_json_summary_manifest_dimension_count_values_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_dimension_counts"] == {"1D": 4, "2D": 2, "2D-axisymmetric": 2, "3D": 3}


def test_validation_cli_writes_json_summary_covered_manifest_dimension_counts_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert sum(payload["covered_manifest_dimension_counts"].values()) == payload["covered_manifest_case_count"]


def test_validation_cli_writes_json_summary_covered_manifest_dimension_count_values_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["covered_manifest_dimension_counts"] == {"1D": 4, "2D": 2, "2D-axisymmetric": 2, "3D": 3}


def test_validation_cli_writes_json_summary_dimension_counts_covered_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["covered_manifest_dimension_counts"] == payload["manifest_dimension_counts"]


def test_validation_cli_writes_json_summary_manifest_dimension_coverage_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_dimension_coverage_status"] == "pass") is (
        payload["covered_manifest_dimension_counts"] == payload["manifest_dimension_counts"]
    )


def test_validation_cli_writes_json_summary_dimension_status_rollup_input_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_coverage_rollup_status"] == "pass") is (
        payload["manifest_dimension_coverage_status"] == "pass"
        and payload["required_manifest_coverage_status"] == "pass"
        and payload["optional_manifest_coverage_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_manifest_metadata_keys_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert set(payload["manifest_case_dimensions"]) == set(payload["manifest_case_status"])
    assert set(payload["manifest_case_purposes"]) == set(payload["manifest_case_status"])


def test_validation_cli_writes_json_summary_manifest_metadata_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_metadata_status"] == "pass") is (
        set(payload["manifest_case_dimensions"]) == set(payload["manifest_case_status"])
        and set(payload["manifest_case_purposes"]) == set(payload["manifest_case_status"])
    )


def test_validation_cli_writes_json_summary_manifest_summary_status_inputs_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_summary_status"] == "pass") is all(
        payload[key] == "pass"
        for key in (
            "manifest_metadata_status",
            "manifest_case_count_status",
            "manifest_coverage_rollup_status",
            "manifest_case_coverage_status",
        )
    )


def test_validation_cli_writes_json_summary_metadata_status_summary_input_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_summary_status"] == "pass") is (
        payload["manifest_metadata_status"] == "pass"
        and payload["manifest_case_count_status"] == "pass"
        and payload["manifest_coverage_rollup_status"] == "pass"
        and payload["manifest_case_coverage_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_case_count_status_summary_input_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_summary_status"] == "pass") is (
        payload["manifest_case_count_status"] == "pass"
        and payload["manifest_metadata_status"] == "pass"
        and payload["manifest_coverage_rollup_status"] == "pass"
        and payload["manifest_case_coverage_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_coverage_rollup_status_summary_input_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_summary_status"] == "pass") is (
        payload["manifest_coverage_rollup_status"] == "pass"
        and payload["manifest_metadata_status"] == "pass"
        and payload["manifest_case_count_status"] == "pass"
        and payload["manifest_case_coverage_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_case_coverage_status_summary_input_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["manifest_summary_status"] == "pass") is (
        payload["manifest_case_coverage_status"] == "pass"
        and payload["manifest_metadata_status"] == "pass"
        and payload["manifest_case_count_status"] == "pass"
        and payload["manifest_coverage_rollup_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_manifest_case_ids_match_status_keys_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert set(payload["manifest_case_ids"]) == set(payload["manifest_case_status"])


def test_validation_cli_writes_json_summary_cone_jet_manifest_dimension_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_dimensions"]["2d_cone_jet"] == "2D-axisymmetric"


def test_validation_cli_writes_json_summary_multi_emitter_manifest_dimension_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_dimensions"]["3d_multi_emitter"] == "3D"


def test_validation_cli_writes_json_summary_plume_manifest_dimension_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_dimensions"]["3d_plume_impingement"] == "3D"


def test_validation_cli_writes_json_summary_microthruster_manifest_dimension_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_dimensions"]["3d_microthruster_performance"] == "3D"


def test_validation_cli_writes_json_summary_cone_jet_manifest_purpose_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_purposes"]["2d_cone_jet"] == "steady cone-jet observables"


def test_validation_cli_writes_json_summary_plume_manifest_purpose_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_purposes"]["3d_plume_impingement"] == "plume divergence and target impingement"


def test_validation_cli_writes_json_summary_multi_emitter_manifest_purpose_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_purposes"]["3d_multi_emitter"] == "array current sharing and shielding"


def test_validation_cli_writes_json_summary_microthruster_manifest_purpose_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_purposes"]["3d_microthruster_performance"] == "microthruster performance metrics"


def test_validation_cli_writes_json_summary_required_manifest_case_order_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["required_manifest_case_ids"] == [
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


def test_validation_cli_writes_json_summary_optional_manifest_case_order_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["optional_manifest_case_ids"] == [
        "3d_plume_impingement",
        "3d_microthruster_performance",
    ]


def test_validation_cli_writes_json_summary_manifest_case_status_values_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert all(payload["manifest_case_status"].values())


def test_validation_cli_writes_json_summary_multi_emitter_manifest_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_status"]["3d_multi_emitter"] is True


def test_validation_cli_writes_json_summary_plume_manifest_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_status"]["3d_plume_impingement"] is True


def test_validation_cli_writes_json_summary_microthruster_manifest_status_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["manifest_case_status"]["3d_microthruster_performance"] is True


def test_validation_cli_writes_json_summary_3d_manifest_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert {
        case_id: payload["manifest_case_status"][case_id]
        for case_id in payload["manifest_case_ids"]
        if payload["manifest_case_dimensions"][case_id] == "3D"
    } == {
        "3d_multi_emitter": True,
        "3d_plume_impingement": True,
        "3d_microthruster_performance": True,
    }


def test_validation_cli_writes_json_summary_2d_manifest_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert {
        case_id: payload["manifest_case_status"][case_id]
        for case_id in payload["manifest_case_ids"]
        if payload["manifest_case_dimensions"][case_id] in {"2D", "2D-axisymmetric"}
    } == {
        "vof_interface_transport": True,
        "2d_droplet_deformation": True,
        "2d_taylor_cone": True,
        "2d_cone_jet": True,
    }


def test_validation_cli_writes_json_summary_1d_manifest_statuses_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert {
        case_id: payload["manifest_case_status"][case_id]
        for case_id in payload["manifest_case_ids"]
        if payload["manifest_case_dimensions"][case_id] == "1D"
    } == {
        "1d_parallel_plate": True,
        "1d_dielectric_jump": True,
        "1d_charge_relaxation": True,
        "1d_maxwell_jump": True,
    }


def test_validation_cli_writes_json_summary_status_result_manifest_equivalence_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_result_accounting_status"] == "pass"
        and payload["manifest_summary_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_manifest_status_validation_summary_input_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["manifest_summary_status"] == "pass"
        and payload["validation_result_accounting_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_result_accounting_validation_summary_input_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["validation_summary_status"] == "pass") is (
        payload["validation_result_accounting_status"] == "pass"
        and payload["manifest_summary_status"] == "pass"
    )


def test_validation_cli_writes_json_summary_failure_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_failure_count"] == 0


def test_validation_cli_writes_json_summary_failure_count_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["validation_summary_failure_count"] == payload["total_results"] - payload["passed_results"]


def test_validation_cli_writes_json_summary_empty_failed_cases_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["failed_executable_result_case_ids"] == []


def test_validation_cli_writes_json_summary_empty_failure_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["failed_executable_result_case_ids"]
        == []
        and payload["failed_executable_result_case_id_count"] == 0
        and payload["validation_summary_failure_count"] == 0
    )


def test_validation_cli_writes_json_summary_passed_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["passed_executable_result_case_id_count"] == payload["passed_results"]


def test_validation_cli_writes_json_summary_passed_case_count_length_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["passed_executable_result_case_id_count"] == len(
        payload["passed_executable_result_case_ids"]
    )


def test_validation_cli_writes_json_summary_failed_case_count_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["failed_executable_result_case_id_count"] == 0


def test_validation_cli_writes_json_summary_failed_case_count_length_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["failed_executable_result_case_id_count"] == len(
        payload["failed_executable_result_case_ids"]
    )


def test_validation_cli_writes_json_summary_failed_case_count_accounting_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["failed_executable_result_case_id_count"] == payload["total_results"] - payload["passed_results"]


def test_validation_cli_writes_json_summary_passed_case_ids_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["passed_executable_result_case_ids"] == payload["executable_result_case_ids"]


def test_validation_cli_writes_json_summary_passed_case_full_coverage_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (
        payload["passed_executable_result_case_id_count"]
        == payload["executable_result_case_id_count"]
        == len(payload["executable_result_case_ids"])
    )


def test_validation_cli_writes_json_summary_executable_case_count_length_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["executable_result_case_id_count"] == len(payload["executable_result_case_ids"])


def test_validation_cli_writes_json_summary_unique_executable_case_count_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert payload["unique_executable_result_case_id_count"] == len(set(payload["executable_result_case_ids"]))


def test_validation_cli_writes_json_summary_executable_case_id_status_to_output_file(
    tmp_path, capsys
) -> None:
    output_path = tmp_path / "validation_summary.json"

    assert main(["--format", "json", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output_path.read_text())
    assert (payload["executable_result_case_id_status"] == "pass") is (
        payload["executable_result_case_id_count"] == payload["unique_executable_result_case_id_count"]
    )


def test_validation_cli_writes_nonempty_markdown_to_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_report.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert output_path.read_text()


def test_validation_cli_writes_health_status_to_markdown_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_report.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert "- validation_summary_health_status: pass" in output_path.read_text()


def test_validation_cli_writes_failed_health_inputs_to_markdown_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_report.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert "- validation_summary_failed_health_inputs: []" in output_path.read_text()


def test_validation_cli_writes_health_input_failure_count_to_markdown_output_file(tmp_path, capsys) -> None:
    output_path = tmp_path / "validation_report.md"

    assert main(["--format", "markdown", "--output", str(output_path)]) == 0

    assert capsys.readouterr().out == ""
    assert "- validation_summary_health_input_failure_count: 0" in output_path.read_text()


def test_validation_cli_check_all_artifacts_reports_current_after_write_artifacts(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-all-artifacts"]) == 0

    assert capsys.readouterr().out == "all validation artifacts current\n"


def test_validation_cli_check_artifacts_reports_current_after_write_artifacts(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-artifacts"]) == 0

    assert capsys.readouterr().out == "validation artifacts current\n"


def test_validation_cli_check_reduced_step_artifact_reports_current_after_write_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-reduced-step-artifact"]) == 0

    assert capsys.readouterr().out == "reduced step artifact current\n"


def test_validation_cli_check_plume_artifact_reports_current_after_write_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-plume-artifact"]) == 0

    assert capsys.readouterr().out == "plume artifact current\n"


def test_validation_cli_check_microthruster_artifact_reports_current_after_write_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-microthruster-artifact"]) == 0

    assert capsys.readouterr().out == "microthruster artifact current\n"


def test_validation_cli_check_application_artifact_reports_current_after_write_artifacts(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(validation_cli, "_repo_root", lambda: tmp_path)

    assert main(["--write-artifacts"]) == 0
    capsys.readouterr()
    assert main(["--check-application-artifact"]) == 0

    assert capsys.readouterr().out == "application artifact current\n"


def test_validation_cli_check_artifacts_reports_current(capsys) -> None:
    assert main(["--check-artifacts"]) == 0
    assert capsys.readouterr().out == "validation artifacts current\n"


def test_validation_cli_check_reduced_step_artifact_reports_current(capsys) -> None:
    assert main(["--check-reduced-step-artifact"]) == 0
    assert capsys.readouterr().out == "reduced step artifact current\n"


def test_validation_cli_check_all_artifacts_reports_current(capsys) -> None:
    assert main(["--check-all-artifacts"]) == 0
    assert capsys.readouterr().out == "all validation artifacts current\n"


def test_validation_cli_check_all_artifacts_preserves_validation_summary_health_trace(capsys) -> None:
    root = Path(__file__).resolve().parents[2]

    assert main(["--check-all-artifacts"]) == 0
    assert capsys.readouterr().out == "all validation artifacts current\n"

    payload = json.loads((root / "docs" / "electrospray" / "validation_summary.json").read_text())
    report = (root / "docs" / "electrospray" / "validation_report.md").read_text()

    assert payload["validation_summary_health_status"] == "pass"
    assert payload["validation_summary_failed_health_inputs"] == []
    assert payload["validation_summary_health_input_failure_count"] == 0
    assert "- validation_summary_health_status: pass" in report
    assert "- validation_summary_failed_health_inputs: []" in report
    assert "- validation_summary_health_input_failure_count: 0" in report


def test_validation_cli_check_plume_artifact_reports_current(capsys) -> None:
    assert main(["--check-plume-artifact"]) == 0
    assert capsys.readouterr().out == "plume artifact current\n"


def test_validation_cli_check_microthruster_artifact_reports_current(capsys) -> None:
    assert main(["--check-microthruster-artifact"]) == 0
    assert capsys.readouterr().out == "microthruster artifact current\n"


def test_validation_cli_check_application_artifact_reports_current(capsys) -> None:
    assert main(["--check-application-artifact"]) == 0
    assert capsys.readouterr().out == "application artifact current\n"


def test_validation_cli_check_reduced_step_artifact_reports_stale_for_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-reduced-step-artifact"]) == 1
    assert capsys.readouterr().out == "reduced step artifact stale\n"


def test_validation_cli_check_plume_artifact_reports_stale_for_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-plume-artifact"]) == 1
    assert capsys.readouterr().out == "plume artifact stale\n"


def test_validation_cli_check_microthruster_artifact_reports_stale_for_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-microthruster-artifact"]) == 1
    assert capsys.readouterr().out == "microthruster artifact stale\n"


def test_validation_cli_check_application_artifact_reports_stale_for_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-application-artifact"]) == 1
    assert capsys.readouterr().out == "application artifact stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_reduced_step_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_reduced_step_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_reduced_step(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_plume_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_plume_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_plume(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_microthruster_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_microthruster_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_microthruster(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_application_mismatch(tmp_path, monkeypatch, capsys) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_application_directory(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_application(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "application_report.json"
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_validation_artifact_mismatch(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("stale\n")
    (docs / "validation_summary.json").write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_validation_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_summary.json").write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_missing_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_malformed_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("{not-json\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_non_object_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("[]\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_directory_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").mkdir()
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_all_artifacts_reports_stale_for_directory_validation_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").mkdir()
    (docs / "validation_summary.json").write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-all-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_mismatch(tmp_path, monkeypatch, capsys) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("stale\n")
    (docs / "validation_summary.json").write_text("{}\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_missing_validation_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_summary.json").write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_missing_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_malformed_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("{not-json\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_non_object_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").write_text("[]\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_directory_validation_summary(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").write_text("- validation_summary_health_status: pass\n")
    (docs / "validation_summary.json").mkdir()
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_artifacts_reports_stale_for_directory_validation_markdown(
    tmp_path, monkeypatch, capsys
) -> None:
    docs = tmp_path / "docs" / "electrospray"
    docs.mkdir(parents=True)
    (docs / "validation_report.md").mkdir()
    (docs / "validation_summary.json").write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")
    monkeypatch.setattr(validation_cli, "_artifact_paths", lambda: (docs / "validation_report.md", docs / "validation_summary.json"))

    assert main(["--check-artifacts"]) == 1
    assert capsys.readouterr().out == "validation artifacts stale\n"


def test_validation_cli_check_reduced_step_artifact_reports_stale_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-reduced-step-artifact"]) == 1
    assert capsys.readouterr().out == "reduced step artifact stale\n"


def test_validation_cli_check_reduced_step_artifact_reports_stale_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    monkeypatch.setattr(validation_cli, "_reduced_step_artifact_path", lambda: report_path)

    assert main(["--check-reduced-step-artifact"]) == 1
    assert capsys.readouterr().out == "reduced step artifact stale\n"


def test_validation_cli_check_plume_artifact_reports_stale_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-plume-artifact"]) == 1
    assert capsys.readouterr().out == "plume artifact stale\n"


def test_validation_cli_check_plume_artifact_reports_stale_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    monkeypatch.setattr(validation_cli, "_plume_artifact_path", lambda: report_path)

    assert main(["--check-plume-artifact"]) == 1
    assert capsys.readouterr().out == "plume artifact stale\n"


def test_validation_cli_check_microthruster_artifact_reports_stale_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-microthruster-artifact"]) == 1
    assert capsys.readouterr().out == "microthruster artifact stale\n"


def test_validation_cli_check_microthruster_artifact_reports_stale_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    monkeypatch.setattr(validation_cli, "_microthruster_artifact_path", lambda: report_path)

    assert main(["--check-microthruster-artifact"]) == 1
    assert capsys.readouterr().out == "microthruster artifact stale\n"


def test_validation_cli_check_application_artifact_reports_stale_for_directory_path(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.mkdir()
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-application-artifact"]) == 1
    assert capsys.readouterr().out == "application artifact stale\n"


def test_validation_cli_check_application_artifact_reports_stale_when_file_missing(
    tmp_path, monkeypatch, capsys
) -> None:
    report_path = tmp_path / "application_report.json"
    monkeypatch.setattr(validation_cli, "_application_artifact_path", lambda: report_path)

    assert main(["--check-application-artifact"]) == 1
    assert capsys.readouterr().out == "application artifact stale\n"
