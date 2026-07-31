import json

import pytest

from benchmark_stepper_observables import (
    AXISYMMETRIC_CONEJET_SOLVER_PATH,
    COMMON_SCHEME_ID,
    COUPLED_DROPLET_SOLVER_PATH,
    FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH,
    FULL_CFD_DROPLET_SOLVER_PATH,
)
from full_cfd_solver import FULL_CFD_SCHEME_ID
from external_benchmarks import (
    das_saintillan_droplet_benchmark_metadata,
    das_saintillan_droplet_benchmark_metadata_json,
    das_saintillan_droplet_external_comparison_ready,
    das_saintillan_full_cfd_droplet_comparison_rows,
    das_saintillan_reduced_solver_comparison_rows,
    external_benchmark_readiness_report_json,
    external_benchmark_readiness_report,
    external_benchmark_comparison_rows,
    external_benchmark_comparison_markdown,
    external_benchmark_metadata_has_numeric_references,
    combined_external_reduced_solver_comparison_passes,
    combined_external_reduced_solver_comparison_rows,
    huh_wirz_conejet_benchmark_metadata,
    huh_wirz_conejet_benchmark_metadata_json,
    huh_wirz_external_comparison_ready,
    huh_wirz_full_cfd_breakup_comparison_rows,
    huh_wirz_full_cfd_nonbreakup_comparison_rows,
    huh_wirz_reduced_solver_comparison_rows,
)


def test_huh_wirz_benchmark_metadata_is_machine_readable_and_source_checked() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()

    assert metadata["source"]["doi"] == "10.1063/5.0120737"
    assert metadata["source"]["arxiv"] == "2111.10383"
    assert [block["case_id"] for block in metadata["benchmark_blocks"]] == [
        "huh_wirz_heptane_moderate_conductivity",
        "huh_wirz_tbp_high_conductivity",
        "huh_wirz_tbp_minimum_flow_cone_to_jet",
    ]


def test_huh_wirz_benchmark_metadata_requires_numeric_reference_values_before_completion() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()

    assert [block["status"] for block in metadata["benchmark_blocks"]] == [
        "partial_numeric_reference",
        "partial_numeric_reference",
        "partial_numeric_reference",
    ]
    assert metadata["benchmark_blocks"][0]["reference_values"][0]["observable"] == "droplet_diameter"
    assert metadata["benchmark_blocks"][0]["reference_values"][0]["value"] == pytest.approx(3.5e-5)
    assert {reference["observable"] for reference in metadata["benchmark_blocks"][1]["reference_values"]} == {
        "charge_to_mass_ratio",
        "droplet_diameter",
        "jet_diameter",
        "total_current",
    }
    assert metadata["benchmark_blocks"][2]["reference_values"][0]["observable"] == "cone_to_jet_length"
    assert metadata["benchmark_blocks"][2]["reference_values"][0]["value"] == pytest.approx(65.0e-6)
    assert "reference_values must contain digitized or tabulated numeric entries" in metadata["completion_gate"]


def test_huh_wirz_tbp_digitized_references_are_machine_readable() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    tbp_block = metadata["benchmark_blocks"][1]

    references = {reference["observable"]: reference for reference in tbp_block["reference_values"]}

    assert references["droplet_diameter"]["value"] == pytest.approx(1.0e-5)
    assert references["jet_diameter"]["value"] == pytest.approx(6.0e-6)
    assert references["total_current"]["value"] == pytest.approx(3.0e-8)
    assert references["charge_to_mass_ratio"]["value"] == pytest.approx(0.65)
    assert all(reference["digitization_method"] == "manual_from_pdf_raster" for reference in references.values())
    assert {reference["source_figure"] for reference in references.values()} == {"Figure 7", "Figure 8a", "Figure 12"}
    assert all(
        reference["independent_variables"]["nondimensional_flow_rate_delta"] == pytest.approx(47.6)
        for reference in references.values()
    )


def test_huh_wirz_heptane_digitized_reference_is_machine_readable() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    heptane = metadata["benchmark_blocks"][0]
    reference = heptane["reference_values"][0]

    assert heptane["case_id"] == "huh_wirz_heptane_moderate_conductivity"
    assert reference["observable"] == "droplet_diameter"
    assert reference["value"] == pytest.approx(3.5e-5)
    assert reference["source_figure"] == "Figure 3b"
    assert reference["independent_variables"]["voltage_V"] == pytest.approx(4000.0)
    assert reference["digitization_method"] == "manual_from_pdf_raster"


def test_huh_wirz_minimum_flow_cone_to_jet_reference_is_machine_readable() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    minimum_flow = metadata["benchmark_blocks"][2]
    reference = minimum_flow["reference_values"][0]

    assert minimum_flow["case_id"] == "huh_wirz_tbp_minimum_flow_cone_to_jet"
    assert reference["observable"] == "cone_to_jet_length"
    assert reference["value"] == pytest.approx(65.0e-6)
    assert reference["source_figure"] == "Figure 8"
    assert reference["digitization_method"] == "text_extracted_reference"
    assert reference["independent_variables"]["nondimensional_flow_rate_delta"] == pytest.approx(18.1)
    assert "5 to 95 percent" in reference["independent_variables"]["definition"]


def test_huh_wirz_benchmark_metadata_lists_digitization_tasks() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    tasks = [
        task
        for block in metadata["benchmark_blocks"]
        for task in block["digitization_tasks"]
    ]

    assert {task["source_figure"] for task in tasks} == {"Figure 3", "Figure 7", "Figure 8", "Figure 12"}
    assert {task["observable"] for task in tasks} >= {
        "charge_to_mass_ratio",
        "cone_to_jet_length",
        "droplet_diameter",
        "jet_diameter",
        "total_current",
    }
    assert all(task["independent_variables"] for task in tasks)


def test_huh_wirz_benchmark_metadata_json_is_stable() -> None:
    payload = json.loads(huh_wirz_conejet_benchmark_metadata_json())

    assert payload == huh_wirz_conejet_benchmark_metadata()


def test_huh_wirz_external_comparison_is_ready_with_digitized_numeric_references() -> None:
    assert huh_wirz_external_comparison_ready() is True


def test_das_saintillan_droplet_metadata_is_machine_readable_and_source_checked() -> None:
    metadata = das_saintillan_droplet_benchmark_metadata()

    assert metadata["source"]["doi"] == "10.1017/jfm.2016.704"
    assert metadata["source"]["arxiv"] == "1605.04036"
    assert [block["case_id"] for block in metadata["benchmark_blocks"]] == [
        "das_saintillan_transient_system_1b",
        "das_saintillan_transient_system_1c",
        "das_saintillan_prolate_system_3",
        "das_saintillan_steady_systems_2a_2b",
    ]


def test_das_saintillan_droplet_metadata_lists_required_digitization_tasks() -> None:
    metadata = das_saintillan_droplet_benchmark_metadata()
    tasks = [
        task
        for block in metadata["benchmark_blocks"]
        for task in block["digitization_tasks"]
    ]

    assert {task["source_figure"] for task in tasks} == {"Figure 4", "Figure 5", "Figure 6", "Figure 7"}
    assert {task["observable"] for task in tasks} >= {
        "deformation_parameter",
        "surface_charge_profile",
    }
    assert any("electric_capillary_number" in block["required_outputs"] for block in metadata["benchmark_blocks"])
    assert all(task["independent_variables"] for task in tasks)


def test_das_saintillan_droplet_metadata_requires_numeric_values_before_completion() -> None:
    metadata = das_saintillan_droplet_benchmark_metadata()

    assert [block["status"] for block in metadata["benchmark_blocks"]] == [
        "partial_numeric_reference",
        "partial_numeric_reference",
        "partial_numeric_reference",
        "partial_numeric_reference",
    ]
    assert all(block["reference_values"] for block in metadata["benchmark_blocks"])
    assert "reference_values must contain digitized or tabulated numeric entries" in metadata["completion_gate"]
    assert das_saintillan_droplet_external_comparison_ready() is True


def test_das_saintillan_digitized_references_cover_transient_steady_and_charge_trends() -> None:
    metadata = das_saintillan_droplet_benchmark_metadata()
    by_case = {block["case_id"]: block["reference_values"] for block in metadata["benchmark_blocks"]}

    assert {reference["observable"] for reference in by_case["das_saintillan_transient_system_1b"]} == {
        "deformation_parameter",
        "surface_charge_endpoint_difference",
    }
    assert {reference["observable"] for reference in by_case["das_saintillan_transient_system_1c"]} == {
        "deformation_parameter",
        "surface_charge_sign_change",
    }
    steady = by_case["das_saintillan_steady_systems_2a_2b"][0]
    assert steady["observable"] == "deformation_parameter"
    assert steady["independent_variables"]["electric_capillary_number"] == pytest.approx(0.5)
    assert steady["value"] == pytest.approx(-0.067)


def test_das_saintillan_system3_text_reference_is_machine_readable() -> None:
    metadata = das_saintillan_droplet_benchmark_metadata()
    system3 = metadata["benchmark_blocks"][2]
    reference = system3["reference_values"][0]

    assert system3["case_id"] == "das_saintillan_prolate_system_3"
    assert reference["observable"] == "deformation_parameter"
    assert reference["value"] == pytest.approx(0.27)
    assert reference["digitization_method"] == "text_extracted_reference"
    assert reference["comparison_role"] == "boundary_element_reference"
    assert reference["source_figure"] == "Figure 6"


def test_das_saintillan_droplet_metadata_json_is_stable() -> None:
    payload = json.loads(das_saintillan_droplet_benchmark_metadata_json())

    assert payload == das_saintillan_droplet_benchmark_metadata()


def test_combined_external_benchmark_readiness_requires_all_suites_and_blocks() -> None:
    report = external_benchmark_readiness_report()

    assert report["ready"] is True
    assert report["suite_status"] == {
        "das_saintillan_droplet": True,
        "huh_wirz_conejet": True,
    }
    assert report["required_suites"] == ["das_saintillan_droplet", "huh_wirz_conejet"]
    assert report["missing_reference_blocks"]["huh_wirz_conejet"] == []
    assert report["missing_reference_blocks"]["das_saintillan_droplet"] == []
    assert "numeric reference_values" in report["completion_gate"]


def test_combined_external_benchmark_readiness_json_is_stable() -> None:
    payload = json.loads(external_benchmark_readiness_report_json())

    assert payload == external_benchmark_readiness_report()


def test_external_benchmark_numeric_reference_checker_requires_values_in_every_block() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    first_block, second_block, third_block = metadata["benchmark_blocks"]
    second_block["reference_values"] = []
    third_block["reference_values"] = [{"observable": "cone_to_jet_length", "value": 65.0e-6, "unit": "m"}]
    first_block["reference_values"] = [{"observable": "droplet_diameter", "value": 1.2e-6, "unit": "m"}]

    assert external_benchmark_metadata_has_numeric_references(metadata) is False

    second_block["reference_values"] = [{"observable": "total_current", "value": 8.0e-8, "unit": "A"}]
    assert external_benchmark_metadata_has_numeric_references(metadata) is True


def test_external_benchmark_comparison_rows_compute_relative_errors() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    first_block, second_block, third_block = metadata["benchmark_blocks"]
    first_block["reference_values"] = [{"observable": "droplet_diameter", "value": 1.2e-6, "unit": "m"}]
    second_block["reference_values"] = [{"observable": "total_current", "value": 8.0e-8, "unit": "A"}]
    third_block["reference_values"] = []

    rows = external_benchmark_comparison_rows(
        metadata,
        predictions={
            ("huh_wirz_heptane_moderate_conductivity", "droplet_diameter"): 1.1e-6,
            ("huh_wirz_tbp_high_conductivity", "total_current"): 8.6e-8,
        },
        tolerances={"droplet_diameter": 0.25, "total_current": 0.20},
    )

    assert [row.observable for row in rows] == ["droplet_diameter", "total_current"]
    assert rows[0].relative_error == pytest.approx(1.0 / 12.0)
    assert rows[1].relative_error == pytest.approx(0.075)
    assert all(row.passed for row in rows)
    assert {row.solver_path for row in rows} == {"unreported"}
    assert {row.scheme_id for row in rows} == {"unreported"}
    assert {row.claim_scope for row in rows} == {"reduced-kernel comparison only"}


def test_external_benchmark_comparison_rows_reject_missing_predictions() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    metadata["benchmark_blocks"][0]["reference_values"] = [{"observable": "droplet_diameter", "value": 1.2e-6}]

    with pytest.raises(ValueError, match="missing prediction"):
        external_benchmark_comparison_rows(metadata, predictions={}, tolerances={"droplet_diameter": 0.25})


def test_external_benchmark_comparison_markdown_is_manuscript_ready() -> None:
    metadata = huh_wirz_conejet_benchmark_metadata()
    metadata["benchmark_blocks"][0]["reference_values"] = [{"observable": "droplet_diameter", "value": 1.2e-6}]
    metadata["benchmark_blocks"][1]["reference_values"] = [{"observable": "total_current", "value": 8.0e-8}]
    metadata["benchmark_blocks"][2]["reference_values"] = []
    rows = external_benchmark_comparison_rows(
        metadata,
        predictions={
            ("huh_wirz_heptane_moderate_conductivity", "droplet_diameter"): 1.1e-6,
            ("huh_wirz_tbp_high_conductivity", "total_current"): 8.6e-8,
        },
        tolerances={"droplet_diameter": 0.25, "total_current": 0.20},
    )

    markdown = external_benchmark_comparison_markdown(rows)

    assert "# External Benchmark Numeric Comparison" in markdown
    assert "huh_wirz_heptane_moderate_conductivity" in markdown
    assert "droplet_diameter" in markdown
    assert "8.333333e-02" in markdown
    assert "7.500000e-02" in markdown
    assert markdown.count("PASS") == 2
    assert "digitized or tabulated external reference values" in markdown
    assert "Solver path" in markdown
    assert "Scheme" in markdown
    assert "Claim scope" in markdown
    assert "reduced-kernel comparison only" in markdown


def test_huh_wirz_reduced_solver_external_comparison_rows_pass() -> None:
    rows = huh_wirz_reduced_solver_comparison_rows()

    assert [(row.case_id, row.observable) for row in rows] == [
        ("huh_wirz_heptane_moderate_conductivity", "droplet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "droplet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "jet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "total_current"),
        ("huh_wirz_tbp_high_conductivity", "charge_to_mass_ratio"),
        ("huh_wirz_tbp_minimum_flow_cone_to_jet", "cone_to_jet_length"),
    ]
    assert all(row.passed for row in rows)
    assert max(row.relative_error for row in rows) < 0.08
    assert {row.solver_path for row in rows} == {AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert {row.scheme_id for row in rows} == {COMMON_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"reduced-kernel comparison only"}


def test_huh_wirz_full_cfd_nonbreakup_external_comparison_rows_pass() -> None:
    rows = huh_wirz_full_cfd_nonbreakup_comparison_rows()

    assert [(row.case_id, row.observable) for row in rows] == [
        ("huh_wirz_tbp_high_conductivity", "jet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "total_current"),
        ("huh_wirz_tbp_high_conductivity", "charge_to_mass_ratio"),
        ("huh_wirz_tbp_minimum_flow_cone_to_jet", "cone_to_jet_length"),
    ]
    assert all(row.passed for row in rows)
    assert {row.solver_path for row in rows} == {FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert {row.scheme_id for row in rows} == {FULL_CFD_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"full-timestep non-breakup cone-jet observables only"}


def test_huh_wirz_full_cfd_subgrid_breakup_external_comparison_rows_pass() -> None:
    rows = huh_wirz_full_cfd_breakup_comparison_rows()

    assert [(row.case_id, row.observable) for row in rows] == [
        ("huh_wirz_heptane_moderate_conductivity", "droplet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "droplet_diameter"),
        ("huh_wirz_tbp_high_conductivity", "charge_to_mass_ratio"),
    ]
    assert all(row.passed for row in rows)
    assert {row.solver_path for row in rows} == {FULL_CFD_AXISYMMETRIC_CONEJET_SOLVER_PATH}
    assert {row.scheme_id for row in rows} == {FULL_CFD_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"full-timestep plus single global charged-breakup subgrid model"}


def test_das_saintillan_reduced_solver_external_comparison_rows_pass() -> None:
    rows = das_saintillan_reduced_solver_comparison_rows()

    assert [(row.case_id, row.observable) for row in rows] == [
        ("das_saintillan_transient_system_1b", "deformation_parameter"),
        ("das_saintillan_transient_system_1b", "surface_charge_endpoint_difference"),
        ("das_saintillan_transient_system_1c", "deformation_parameter"),
        ("das_saintillan_transient_system_1c", "surface_charge_sign_change"),
        ("das_saintillan_prolate_system_3", "deformation_parameter"),
        ("das_saintillan_steady_systems_2a_2b", "deformation_parameter"),
    ]
    assert all(row.passed for row in rows)
    assert max(row.relative_error for row in rows) < 0.08
    assert {row.solver_path for row in rows} == {COUPLED_DROPLET_SOLVER_PATH}
    assert {row.scheme_id for row in rows} == {COMMON_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"reduced-kernel comparison only"}


def test_das_saintillan_full_cfd_droplet_external_comparison_rows_pass() -> None:
    rows = das_saintillan_full_cfd_droplet_comparison_rows()

    assert [(row.case_id, row.observable) for row in rows] == [
        ("das_saintillan_transient_system_1b", "deformation_parameter"),
        ("das_saintillan_transient_system_1b", "surface_charge_endpoint_difference"),
        ("das_saintillan_transient_system_1c", "deformation_parameter"),
        ("das_saintillan_transient_system_1c", "surface_charge_sign_change"),
        ("das_saintillan_prolate_system_3", "deformation_parameter"),
        ("das_saintillan_steady_systems_2a_2b", "deformation_parameter"),
    ]
    assert all(row.passed for row in rows)
    assert max(row.relative_error for row in rows) < 0.07
    assert {row.solver_path for row in rows} == {FULL_CFD_DROPLET_SOLVER_PATH}
    assert {row.scheme_id for row in rows} == {FULL_CFD_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"full-timestep droplet deformation comparison"}


def test_combined_external_reduced_solver_comparison_gate_passes() -> None:
    rows = combined_external_reduced_solver_comparison_rows()

    assert len(rows) == 12
    assert combined_external_reduced_solver_comparison_passes() is True
    assert {row.scheme_id for row in rows} == {COMMON_SCHEME_ID}
    assert {row.claim_scope for row in rows} == {"reduced-kernel comparison only"}
