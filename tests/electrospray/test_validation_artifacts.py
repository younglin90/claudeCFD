from __future__ import annotations

import json
from pathlib import Path

import pytest

from field_contours import FIELD_CONTOUR_CASES, FIELD_CONTOUR_FIELDS
from validation_artifacts import (
    VALIDATION_SUMMARY_HEALTH_TRACE_MARKDOWN_LINES,
    application_report_artifact_is_current,
    microthruster_report_artifact_is_current,
    plume_report_artifact_is_current,
    reduced_step_report_artifact_is_current,
    validation_artifacts_are_current,
    validation_artifact_status,
    validation_summary_health_trace_is_current,
    write_validation_artifacts,
)
from validation_cases_application import application_report_json
from validation_cases_coupled import reduced_phase_pair_step_report_json
from validation_cases_plume import plume_impingement_report_json
from validation_cases_thruster import microthruster_operating_point_report_json
from validation_runner import core_validation_markdown, core_validation_summary, run_core_validation_suite

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


def test_validation_artifacts_are_current_for_matching_files(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert validation_artifacts_are_current(markdown_path, json_path)


def test_validation_artifacts_are_current_rejects_stale_files(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text("stale\n")
    json_path.write_text("{}\n")

    assert not validation_artifacts_are_current(markdown_path, json_path)


def test_validation_artifacts_are_current_rejects_missing_markdown_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_artifacts_are_current(markdown_path, json_path)


def test_validation_artifacts_are_current_rejects_missing_json_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())

    assert not validation_artifacts_are_current(markdown_path, json_path)


def test_validation_artifacts_are_current_rejects_directory_markdown_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.mkdir()
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_artifacts_are_current(markdown_path, json_path)


def test_validation_artifacts_are_current_rejects_directory_json_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.mkdir()

    assert not validation_artifacts_are_current(markdown_path, json_path)


def test_validation_summary_health_trace_is_current_for_matching_files(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_markdown_line_contract_lists_required_fields() -> None:
    assert VALIDATION_SUMMARY_HEALTH_TRACE_MARKDOWN_LINES == (
        "- validation_summary_health_status: pass",
        "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']",
        "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'",
        "- validation_summary_health_input_count: 2",
        "- validation_summary_health_input_count_status: pass",
        "- validation_summary_failed_health_inputs: []",
        "- validation_summary_health_input_failure_count: 0",
    )


def test_validation_summary_health_trace_rejects_missing_or_stale_files(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)

    markdown_path.write_text("- validation_summary_health_status: fail\n")
    json_path.write_text(json.dumps({"validation_summary_health_status": "fail"}) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_missing_markdown_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_missing_json_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_directory_json_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.mkdir()

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_directory_markdown_path(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.mkdir()
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_malformed_json(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text("{not-json\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_non_object_json(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text("[]\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_failed_health_status(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_status"] = "fail"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_report_missing_health_lines(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text("- validation_summary_health_status: pass\n")
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_missing_failed_input_trace(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps({"validation_summary_health_status": "pass"}) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_non_list_health_inputs(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_inputs"] = {
        "validation_summary_status": "pass",
        "validation_summary_component_health_status": "pass",
    }
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_reordered_health_inputs(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_inputs"] = [
        "validation_summary_component_health_status",
        "validation_summary_status",
    ]
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_extra_health_input(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_inputs"] = [
        "validation_summary_status",
        "validation_summary_component_health_status",
        "manifest_summary_status",
    ]
    payload["validation_summary_health_input_statuses"] = {
        "validation_summary_status": "pass",
        "validation_summary_component_health_status": "pass",
        "manifest_summary_status": "pass",
    }
    payload["validation_summary_health_input_count"] = 3
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_mismatched_input_count(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_count"] = 1
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_mismatched_input_status_keys(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_statuses"] = {"validation_summary_status": "pass"}
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_non_mapping_input_statuses(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_statuses"] = [
        "validation_summary_status",
        "validation_summary_component_health_status",
    ]
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_failed_input_status(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_statuses"] = {
        "validation_summary_status": "pass",
        "validation_summary_component_health_status": "fail",
    }
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_failed_count_status(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_count_status"] = "fail"
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_nonempty_failed_inputs(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_failed_health_inputs"] = ["validation_summary_status"]
    payload["validation_summary_health_input_failure_count"] = 1
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_json_nonzero_failure_count(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    payload = core_validation_summary()
    payload["validation_summary_health_input_failure_count"] = 1
    markdown_path.write_text(core_validation_markdown())
    json_path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_report_missing_count_status_line(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    report = core_validation_markdown().replace(
        "- validation_summary_health_input_count_status: pass\n",
        "",
    )
    markdown_path.write_text(report)
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_validation_summary_health_trace_rejects_report_missing_input_statuses_line(tmp_path) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    report = core_validation_markdown().replace(
        "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass', 'validation_summary_component_health_status': 'pass'}\n",
        "",
    )
    markdown_path.write_text(report)
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


@pytest.mark.parametrize("line", VALIDATION_SUMMARY_HEALTH_TRACE_MARKDOWN_LINES)
def test_validation_summary_health_trace_rejects_each_missing_report_contract_line(tmp_path, line: str) -> None:
    markdown_path = tmp_path / "validation.md"
    json_path = tmp_path / "validation.json"
    report_lines = [
        report_line for report_line in core_validation_markdown().splitlines()
        if line not in report_line
    ]
    report = "\n".join(report_lines) + "\n"
    markdown_path.write_text(report)
    json_path.write_text(json.dumps(core_validation_summary(), sort_keys=True) + "\n")

    assert not validation_summary_health_trace_is_current(markdown_path, json_path)


def test_write_validation_artifacts_creates_current_files(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)
    markdown_path, json_path, reduced_step_path, plume_path, microthruster_path, application_path = artifact_paths[:6]

    assert validation_artifacts_are_current(markdown_path, json_path)
    assert validation_summary_health_trace_is_current(markdown_path, json_path)
    assert reduced_step_report_artifact_is_current(reduced_step_path)
    assert plume_report_artifact_is_current(plume_path)
    assert microthruster_report_artifact_is_current(microthruster_path)
    assert application_report_artifact_is_current(application_path)
    assert all(validation_artifact_status(tmp_path).values())


def test_write_validation_artifacts_returns_regular_files(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert all(path.is_file() for path in artifact_paths)


def test_write_validation_artifacts_returns_expected_file_count(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert len(artifact_paths) == 56


def test_write_validation_artifacts_returns_unique_paths(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert len(set(artifact_paths)) == len(artifact_paths)


def test_write_validation_artifacts_writes_nonempty_files(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert all(path.read_bytes() for path in artifact_paths)


def test_write_validation_artifacts_writes_parseable_json_files(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert all(json.loads(path.read_text()) is not None for path in artifact_paths if path.suffix == ".json")


def test_write_validation_artifacts_writes_json_objects(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert all(isinstance(json.loads(path.read_text()), dict) for path in artifact_paths if path.suffix == ".json")


def test_write_validation_artifacts_returns_expected_suffix_counts(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert sum(1 for path in artifact_paths if path.suffix == ".md") == 12
    assert sum(1 for path in artifact_paths if path.suffix == ".json") == 10
    assert sum(1 for path in artifact_paths if path.suffix == ".png") == 34


def test_write_validation_artifacts_returns_documented_filenames(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert [path.name for path in artifact_paths] == [
        "validation_report.md",
        "validation_summary.json",
        "reduced_phase_pair_step_report.json",
        "plume_impingement_report.json",
        "microthruster_operating_point_report.json",
        "application_report.json",
        "huh_wirz_conejet_benchmark_metadata.json",
        "das_saintillan_droplet_benchmark_metadata.json",
        "external_benchmark_readiness_report.json",
        "cone_jet_error_budget_table.md",
        "external_benchmark_numeric_comparison_table.md",
        "full_cfd_huh_wirz_nonbreakup_comparison_table.md",
        "full_cfd_huh_wirz_subgrid_breakup_comparison_table.md",
        "taylor_cone_voltage_ramp_balance_table.md",
        "coupled_droplet_grid_refinement_table.md",
        "dielectric_maxwell_droplet_history_table.md",
        "huh_wirz_same_path_grid_refinement_table.md",
        "submission_claim_audit.json",
        "submission_readiness_matrix.md",
        "full_cfd_readiness_report.json",
        "full_cfd_readiness_gates.md",
        "cone_jet_error_budget.png",
        "taylor_cone_voltage_ramp.png",
        "coupled_droplet_grid_refinement.png",
        "external_benchmark_numeric_comparison.png",
        "field_contour_manifest.md",
        *[f"{field_name}.png" for _case_id, _description in FIELD_CONTOUR_CASES for field_name, _field_label in FIELD_CONTOUR_FIELDS],
    ]


def test_write_validation_artifacts_returns_paths_under_docs_electrospray(tmp_path) -> None:
    artifact_paths = write_validation_artifacts(tmp_path)

    assert {path.parent for path in artifact_paths} == {
        tmp_path / "docs" / "electrospray",
        tmp_path / "docs" / "electrospray" / "figures",
        *[tmp_path / "docs" / "electrospray" / "contours" / case_id for case_id, _description in FIELD_CONTOUR_CASES],
    }


def test_write_validation_artifacts_reports_health_trace_status_current(tmp_path) -> None:
    write_validation_artifacts(tmp_path)

    assert validation_artifact_status(tmp_path)["validation_summary_health_trace"] is True


def test_validation_artifact_status_reports_expected_key_count(tmp_path) -> None:
    write_validation_artifacts(tmp_path)

    assert len(validation_artifact_status(tmp_path)) == 15


def test_validation_artifact_status_reports_boolean_values(tmp_path) -> None:
    write_validation_artifacts(tmp_path)

    assert all(isinstance(value, bool) for value in validation_artifact_status(tmp_path).values())


def test_reduced_step_report_artifact_is_current_for_matching_file(tmp_path) -> None:
    report_path = tmp_path / "reduced_step.json"
    report_path.write_text(reduced_phase_pair_step_report_json())

    assert reduced_step_report_artifact_is_current(report_path)


def test_reduced_step_report_artifact_is_current_rejects_stale_file(tmp_path) -> None:
    report_path = tmp_path / "reduced_step.json"
    report_path.write_text("{}\n")

    assert not reduced_step_report_artifact_is_current(report_path)


def test_plume_report_artifact_is_current_for_matching_file(tmp_path) -> None:
    report_path = tmp_path / "plume_report.json"
    report_path.write_text(plume_impingement_report_json())

    assert plume_report_artifact_is_current(report_path)


def test_plume_report_artifact_is_current_rejects_stale_file(tmp_path) -> None:
    report_path = tmp_path / "plume_report.json"
    report_path.write_text("{}\n")

    assert not plume_report_artifact_is_current(report_path)


def test_microthruster_report_artifact_is_current_for_matching_file(tmp_path) -> None:
    report_path = tmp_path / "microthruster_report.json"
    report_path.write_text(microthruster_operating_point_report_json())

    assert microthruster_report_artifact_is_current(report_path)


def test_microthruster_report_artifact_is_current_rejects_stale_file(tmp_path) -> None:
    report_path = tmp_path / "microthruster_report.json"
    report_path.write_text("{}\n")

    assert not microthruster_report_artifact_is_current(report_path)


def test_application_report_artifact_is_current_for_matching_file(tmp_path) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.write_text(application_report_json())

    assert application_report_artifact_is_current(report_path)


def test_application_report_artifact_is_current_rejects_stale_file(tmp_path) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.write_text("{}\n")

    assert not application_report_artifact_is_current(report_path)


def test_repository_reduced_step_report_artifact_matches_live_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    assert reduced_step_report_artifact_is_current(root / "docs" / "electrospray" / "reduced_phase_pair_step_report.json")


def test_repository_plume_report_artifact_matches_live_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    assert plume_report_artifact_is_current(root / "docs" / "electrospray" / "plume_impingement_report.json")


def test_repository_microthruster_report_artifact_matches_live_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    assert microthruster_report_artifact_is_current(root / "docs" / "electrospray" / "microthruster_operating_point_report.json")


def test_repository_application_report_artifact_matches_live_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    assert application_report_artifact_is_current(root / "docs" / "electrospray" / "application_report.json")


def test_repository_application_report_artifact_component_status_trace_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "application_report.json").read_text())
    statuses = payload["application_component_statuses"]
    failed = [name for name, status in statuses.items() if status != "pass"]

    assert payload["application_component_status_keys"] == sorted(statuses)
    assert payload["application_component_status_expected_count"] == 11
    assert payload["application_component_status_count"] == len(statuses)
    assert payload["application_component_status_pass_count"] == sum(1 for status in statuses.values() if status == "pass")
    assert payload["application_component_status_failure_count"] == len(failed)
    assert payload["application_failed_component_statuses"] == failed
    assert payload["application_component_status_count_status"] == "pass"
    assert payload["application_component_status_schema_status"] == "pass"


def test_repository_application_report_artifact_component_status_trace_matches_top_level_fields() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "application_report.json").read_text())
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


def test_repository_application_report_artifact_component_status_keys_match_expected_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "application_report.json").read_text())

    assert payload["application_component_status_keys"] == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert payload["application_component_status_expected_count"] == len(EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS)
    assert sorted(payload["application_component_statuses"]) == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert payload["application_component_status_schema_status"] == "pass"


def test_repository_validation_artifacts_match_live_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    assert validation_artifacts_are_current(
        root / "docs" / "electrospray" / "validation_report.md",
        root / "docs" / "electrospray" / "validation_summary.json",
    )


def test_repository_validation_report_artifact_lists_executable_case_id_integrity() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "electrospray" / "validation_report.md").read_text()

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
    assert "| 2d_cone_jet | 2D-axisymmetric | steady cone-jet observables |" in text
    assert "| 3d_microthruster_performance | 3D | microthruster performance metrics |" in text


def test_repository_validation_report_artifact_health_input_trace_is_visible() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "electrospray" / "validation_report.md").read_text()

    assert "- validation_summary_health_status: pass" in text
    assert "- validation_summary_health_inputs: ['validation_summary_status', 'validation_summary_component_health_status']" in text
    assert "- validation_summary_health_input_statuses: {'validation_summary_status': 'pass'" in text
    assert "- validation_summary_health_input_count: 2" in text
    assert "- validation_summary_health_input_count_status: pass" in text
    assert "- validation_summary_failed_health_inputs: []" in text
    assert "- validation_summary_health_input_failure_count: 0" in text


def test_repository_validation_report_artifact_contains_all_health_trace_lines() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs" / "electrospray" / "validation_report.md").read_text()

    assert "- validation_summary_component_health_status: pass" in text
    assert all(line in text for line in VALIDATION_SUMMARY_HEALTH_TRACE_MARKDOWN_LINES)


def test_repository_validation_summary_artifact_lists_executable_case_ids() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "validation_summary.json").read_text())

    assert payload["executable_result_case_id_count"] == payload["total_results"]
    assert payload["unique_executable_result_case_id_count"] == payload["total_results"]
    assert payload["executable_result_case_id_status"] == "pass"
    assert len(payload["passed_executable_result_case_ids"]) == payload["passed_results"]
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
    assert len(payload["executable_result_case_ids"]) == payload["total_results"]
    assert len(set(payload["executable_result_case_ids"])) == payload["total_results"]
    assert "3d_application_component_status_schema" in payload["executable_result_case_ids"]
    assert len(payload["manifest_case_ids"]) == payload["manifest_case_count"]
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


def test_repository_validation_summary_artifact_health_input_trace_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "validation_summary.json").read_text())

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


def test_repository_validation_summary_artifact_exposes_complete_health_trace_keyset() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "validation_summary.json").read_text())

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


def test_repository_validation_summary_artifact_case_ids_match_suite_order() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / "docs" / "electrospray" / "validation_summary.json").read_text())

    assert payload["executable_result_case_ids"] == [result.case_id for result in run_core_validation_suite()]


def test_repository_validation_artifact_status_reports_all_current() -> None:
    root = Path(__file__).resolve().parents[2]

    assert validation_artifact_status(root) == {
        "application_report": True,
        "microthruster_operating_point_report": True,
        "huh_wirz_conejet_benchmark_metadata": True,
        "das_saintillan_droplet_benchmark_metadata": True,
        "external_benchmark_readiness_report": True,
        "submission_claim_audit": True,
        "submission_readiness_matrix": True,
        "full_cfd_readiness": True,
        "plume_impingement_report": True,
        "reduced_phase_pair_step_report": True,
        "validation_artifacts": True,
        "validation_summary_health_trace": True,
        "manuscript_tables": True,
        "manuscript_figures": True,
        "field_contours": True,
    }


def test_validation_artifact_status_reports_missing_files_false(tmp_path) -> None:
    assert validation_artifact_status(tmp_path) == {
        "application_report": False,
        "microthruster_operating_point_report": False,
        "huh_wirz_conejet_benchmark_metadata": False,
        "das_saintillan_droplet_benchmark_metadata": False,
        "external_benchmark_readiness_report": False,
        "submission_claim_audit": False,
        "submission_readiness_matrix": False,
        "full_cfd_readiness": False,
        "plume_impingement_report": False,
        "reduced_phase_pair_step_report": False,
        "validation_artifacts": False,
        "validation_summary_health_trace": False,
        "manuscript_tables": False,
        "manuscript_figures": False,
        "field_contours": False,
    }


def test_validation_artifact_status_reports_health_trace_false_for_incomplete_validation_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_report.md").write_text(core_validation_markdown())
    (output_dir / "validation_summary.json").write_text(json.dumps({"validation_summary_health_status": "pass"}) + "\n")

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_validation_artifact_status_reports_false_for_missing_validation_markdown(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "validation_report.md").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_validation_artifact_status_reports_false_for_missing_validation_summary(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "validation_summary.json").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_validation_artifact_status_reports_health_trace_false_for_malformed_validation_summary(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_report.md").write_text(core_validation_markdown())
    (output_dir / "validation_summary.json").write_text("{not-json\n")

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_validation_artifact_status_reports_health_trace_false_for_non_object_validation_summary(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_report.md").write_text(core_validation_markdown())
    (output_dir / "validation_summary.json").write_text("[]\n")

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_validation_artifact_status_reports_health_trace_false_for_directory_validation_summary(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "validation_report.md").write_text(core_validation_markdown())
    (output_dir / "validation_summary.json").mkdir()

    status = validation_artifact_status(tmp_path)

    assert status["validation_artifacts"] is False
    assert status["validation_summary_health_trace"] is False


def test_reduced_step_report_artifact_rejects_directory_path(tmp_path) -> None:
    report_path = tmp_path / "reduced_phase_pair_step_report.json"
    report_path.mkdir()

    assert not reduced_step_report_artifact_is_current(report_path)


def test_plume_report_artifact_rejects_directory_path(tmp_path) -> None:
    report_path = tmp_path / "plume_impingement_report.json"
    report_path.mkdir()

    assert not plume_report_artifact_is_current(report_path)


def test_microthruster_report_artifact_rejects_directory_path(tmp_path) -> None:
    report_path = tmp_path / "microthruster_operating_point_report.json"
    report_path.mkdir()

    assert not microthruster_report_artifact_is_current(report_path)


def test_application_report_artifact_rejects_directory_path(tmp_path) -> None:
    report_path = tmp_path / "application_report.json"
    report_path.mkdir()

    assert not application_report_artifact_is_current(report_path)


def test_validation_artifact_status_reports_reduced_step_false_for_directory_path(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "reduced_phase_pair_step_report.json").mkdir()

    status = validation_artifact_status(tmp_path)

    assert status["reduced_phase_pair_step_report"] is False


def test_validation_artifact_status_reports_reduced_step_false_when_file_missing(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["reduced_phase_pair_step_report"] is False


def test_validation_artifact_status_reports_reduced_step_false_for_stale_content(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "reduced_phase_pair_step_report.json").write_text("{}\n")

    status = validation_artifact_status(tmp_path)

    assert status["reduced_phase_pair_step_report"] is False


def test_validation_artifact_status_reports_plume_false_for_directory_path(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "plume_impingement_report.json").mkdir()

    status = validation_artifact_status(tmp_path)

    assert status["plume_impingement_report"] is False


def test_validation_artifact_status_reports_plume_false_when_file_missing(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "plume_impingement_report.json").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["plume_impingement_report"] is False


def test_validation_artifact_status_reports_plume_false_for_stale_content(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "plume_impingement_report.json").write_text("{}\n")

    status = validation_artifact_status(tmp_path)

    assert status["plume_impingement_report"] is False


def test_validation_artifact_status_reports_microthruster_false_for_directory_path(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "microthruster_operating_point_report.json").mkdir()

    status = validation_artifact_status(tmp_path)

    assert status["microthruster_operating_point_report"] is False


def test_validation_artifact_status_reports_microthruster_false_when_file_missing(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["microthruster_operating_point_report"] is False


def test_validation_artifact_status_reports_microthruster_false_for_stale_content(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "microthruster_operating_point_report.json").write_text("{}\n")

    status = validation_artifact_status(tmp_path)

    assert status["microthruster_operating_point_report"] is False


def test_validation_artifact_status_reports_application_false_for_directory_path(tmp_path) -> None:
    output_dir = tmp_path / "docs" / "electrospray"
    output_dir.mkdir(parents=True)
    (output_dir / "application_report.json").mkdir()

    status = validation_artifact_status(tmp_path)

    assert status["application_report"] is False


def test_validation_artifact_status_reports_application_false_when_file_missing(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "application_report.json").unlink()

    status = validation_artifact_status(tmp_path)

    assert status["application_report"] is False


def test_validation_artifact_status_reports_application_false_for_stale_content(tmp_path) -> None:
    write_validation_artifacts(tmp_path)
    (tmp_path / "docs" / "electrospray" / "application_report.json").write_text("{}\n")

    status = validation_artifact_status(tmp_path)

    assert status["application_report"] is False
