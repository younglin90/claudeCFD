from __future__ import annotations

import json

import pytest

from validation_cases_application import (
    application_report,
    application_report_json,
    conejet_array_plume_application_report,
    conejet_array_plume_application_report_json,
    run_application_contamination_case,
    run_application_component_status_schema_case,
    run_application_effective_performance_case,
    run_application_loss_accounting_case,
    run_application_power_accounting_case,
)

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


def test_application_report_combines_plume_and_microthruster_metrics() -> None:
    report = application_report()

    assert report["plume_impingement"]["impingement_fraction"] == 0.25
    assert report["microthruster_operating_point"]["ideal_efficiency"] == 1.0
    assert report["application_component_statuses"] == {
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
    assert report["application_component_status_keys"] == sorted(report["application_component_statuses"])
    assert report["application_component_status_expected_count"] == 11
    assert report["application_component_status_count"] == 11
    assert report["application_component_status_pass_count"] == 11
    assert report["application_component_status_failure_count"] == 0
    assert report["application_failed_component_statuses"] == []
    assert report["application_component_status_count_status"] == "pass"
    assert report["application_component_status_schema_status"] == "pass"
    assert report["plume_current_accounting_status"] == "pass"
    assert report["plume_thrust_accounting_status"] == "pass"
    assert report["plume_surface_loading_status"] == "pass"
    assert report["plume_closure_residual_count"] == 3
    assert report["microthruster_operating_point_status"] == "pass"
    assert report["microthruster_closure_residual_count"] == 7
    assert report["application_total_closure_residual_count"] == 10
    assert report["application_closure_count_status"] == "pass"
    assert report["application_source_status"] == "pass"
    assert report["effective_thrust_after_impingement"] == pytest.approx(1.92e-6)
    assert report["effective_retained_thrust_fraction_after_impingement"] == pytest.approx(0.8)
    assert report["retained_thrust_fraction_balance_error"] == pytest.approx(0.0)
    assert report["effective_specific_impulse_after_impingement"] == pytest.approx(800.0 / 9.80665)
    assert report["effective_thrust_to_power_after_impingement"] == pytest.approx(1.6e-3)
    assert report["application_effective_performance_status"] == "pass"
    assert report["effective_thrust_to_retained_power_after_impingement"] == pytest.approx(2.1333333333333334e-3)
    assert report["deposited_current_after_impingement"] == pytest.approx(3.0e-7)
    assert report["deposited_electrical_power_after_impingement"] == pytest.approx(3.0e-4)
    assert report["deposited_kinetic_power_after_impingement"] == pytest.approx(3.0e-4)
    assert report["deposited_kinetic_power_fraction_after_impingement"] == pytest.approx(0.25)
    assert report["deposited_power_efficiency_after_impingement"] == pytest.approx(1.0)
    assert report["deposited_power_fraction_after_impingement"] == pytest.approx(0.25)
    assert report["retained_current_after_impingement"] == pytest.approx(9.0e-7)
    assert report["retained_current_fraction_after_impingement"] == pytest.approx(0.75)
    assert report["retained_electrical_power_after_impingement"] == pytest.approx(9.0e-4)
    assert report["retained_kinetic_power_after_impingement"] == pytest.approx(9.0e-4)
    assert report["retained_kinetic_power_fraction_after_impingement"] == pytest.approx(0.75)
    assert report["retained_power_efficiency_after_impingement"] == pytest.approx(1.0)
    assert report["retained_power_fraction_after_impingement"] == pytest.approx(0.75)
    assert report["deposited_mass_flow"] == pytest.approx(6.0e-10)
    assert report["deposited_current_fraction"] == pytest.approx(0.25)
    assert report["deposited_mass_flow_fraction"] == pytest.approx(0.25)
    assert report["retained_mass_flow_after_impingement"] == pytest.approx(1.8e-9)
    assert report["retained_mass_flow_fraction_after_impingement"] == pytest.approx(0.75)
    assert report["current_fraction_balance_error"] == pytest.approx(0.0)
    assert report["mass_flow_fraction_balance_error"] == pytest.approx(0.0)
    assert report["application_mass_accounting_status"] == "pass"
    assert report["power_fraction_balance_error"] == pytest.approx(0.0)
    assert report["kinetic_power_fraction_balance_error"] == pytest.approx(0.0)
    assert report["application_power_accounting_status"] == "pass"
    assert report["retained_charge_to_mass_after_impingement"] == pytest.approx(500.0)
    assert report["panel_current_density_after_impingement"] == pytest.approx(1.5e-5)
    assert report["panel_mass_flux"] == pytest.approx(3.0e-8)
    assert report["panel_deposited_charge_to_mass"] == pytest.approx(500.0)
    assert report["panel_charge_to_mass_balance_error"] == pytest.approx(0.0)
    assert report["contamination_exposure_time"] == pytest.approx(3600.0)
    assert report["contamination_mass_loading_limit"] == pytest.approx(1.0e-3)
    assert report["panel_mass_loading"] == pytest.approx(1.08e-4)
    assert report["panel_charge_loading"] == pytest.approx(5.4e-2)
    assert report["panel_charge_loading_balance_error"] == pytest.approx(0.0)
    assert report["time_to_contamination_limit"] == pytest.approx(33333.333333333336)
    assert report["contamination_exposure_margin"] == pytest.approx(9.25925925925926)
    assert report["contamination_exposure_status"] == "pass"
    assert report["application_accounting_status"] == "pass"
    assert report["application_validation_status"] == "pass"


def test_application_report_json_is_stable() -> None:
    payload = application_report_json()

    assert payload.endswith("\n")
    assert json.loads(payload) == application_report()
    assert payload == json.dumps(application_report(), sort_keys=True) + "\n"


def test_conejet_array_plume_application_report_uses_current_conejet_source() -> None:
    report = conejet_array_plume_application_report()

    assert report["source_case"] == "huh_wirz_tbp_high_conductivity"
    assert report["source_scheme_id"] == "full_two_phase_ns_vof_ehd_projection_v1"
    assert report["source_solver_path"] == "full_two_phase_ns_axisymmetric_conejet_adapter"
    assert report["single_emitter_current"] == pytest.approx(3.2249840994886336e-8)
    assert report["single_emitter_jet_diameter"] == pytest.approx(6.327633914430798e-6)
    assert report["single_emitter_droplet_diameter"] == pytest.approx(1.0711695407254123e-5)
    assert report["array_total_current_scaling"] == pytest.approx(0.8807815188754272)
    assert report["array_current_uniformity"] == pytest.approx(0.0)
    assert report["plume_tracking_method"] == "deterministic_weighted_lagrangian_conical_tracks"
    assert report["plume_tracking_particle_count"] == 100
    assert report["plume_tracking_hit_count"] == 17
    assert report["plume_tracking_status"] == "pass"
    assert report["plume_tracking_weight_balance_error"] == pytest.approx(0.0)
    assert report["plume_tracking_half_angle"] == pytest.approx(0.2871954066136265)
    assert report["plume_tracking_panel_plane_z"] == pytest.approx(3.0)
    assert report["plume_tracking_panel_width"] == pytest.approx(0.8)
    assert report["plume_tracking_panel_height"] == pytest.approx(0.8)
    assert report["impingement_fraction"] == pytest.approx(0.17)
    assert report["plume_tracking_deposited_current"] == pytest.approx(report["deposited_array_current"])
    assert report["plume_tracking_retained_current"] == pytest.approx(report["retained_array_current"])
    assert report["current_balance_error"] == pytest.approx(0.0)
    assert report["mass_balance_error"] == pytest.approx(0.0)
    assert report["panel_charge_to_mass"] == pytest.approx(620.0)
    assert report["panel_charge_to_mass_error"] == pytest.approx(0.0)
    assert report["conejet_array_plume_application_status"] == "pass"


def test_conejet_array_plume_application_report_json_is_stable() -> None:
    payload = conejet_array_plume_application_report_json()

    assert payload.endswith("\n")
    assert json.loads(payload) == conejet_array_plume_application_report()
    assert payload == json.dumps(conejet_array_plume_application_report(), sort_keys=True) + "\n"


def test_application_component_status_trace_is_self_consistent() -> None:
    report = application_report()
    statuses = report["application_component_statuses"]
    failed = [name for name, status in statuses.items() if status != "pass"]

    assert report["application_component_status_count"] == len(statuses)
    assert report["application_component_status_pass_count"] == sum(1 for status in statuses.values() if status == "pass")
    assert report["application_component_status_failure_count"] == len(failed)
    assert report["application_failed_component_statuses"] == failed
    assert report["application_component_status_count_status"] == "pass"


def test_application_component_status_keys_match_expected_schema() -> None:
    report = application_report()

    assert report["application_component_status_keys"] == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert report["application_component_status_expected_count"] == len(EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS)
    assert sorted(report["application_component_statuses"]) == EXPECTED_APPLICATION_COMPONENT_STATUS_KEYS
    assert report["application_component_status_schema_status"] == "pass"


def test_application_component_status_trace_matches_top_level_status_fields() -> None:
    report = application_report()
    statuses = report["application_component_statuses"]

    assert statuses["application_accounting"] == report["application_accounting_status"]
    assert statuses["application_closure_count"] == report["application_closure_count_status"]
    assert statuses["application_effective_performance"] == report["application_effective_performance_status"]
    assert statuses["application_mass_accounting"] == report["application_mass_accounting_status"]
    assert statuses["application_power_accounting"] == report["application_power_accounting_status"]
    assert statuses["application_source"] == report["application_source_status"]
    assert statuses["contamination_exposure"] == report["contamination_exposure_status"]
    assert statuses["microthruster_operating_point"] == report["microthruster_operating_point_status"]
    assert statuses["plume_current_accounting"] == report["plume_current_accounting_status"]
    assert statuses["plume_surface_loading"] == report["plume_surface_loading_status"]
    assert statuses["plume_thrust_accounting"] == report["plume_thrust_accounting_status"]


def test_application_effective_performance_case_passes() -> None:
    result = run_application_effective_performance_case()

    assert result.case_id == "3d_application_effective_performance"
    assert result.passed


def test_application_loss_accounting_case_passes() -> None:
    result = run_application_loss_accounting_case()

    assert result.case_id == "3d_application_loss_accounting"
    assert result.passed


def test_application_power_accounting_case_passes() -> None:
    result = run_application_power_accounting_case()

    assert result.case_id == "3d_application_power_accounting"
    assert result.passed


def test_application_contamination_case_passes() -> None:
    result = run_application_contamination_case()

    assert result.case_id == "3d_application_contamination"
    assert result.passed


def test_application_component_status_schema_case_passes() -> None:
    result = run_application_component_status_schema_case()

    assert result.case_id == "3d_application_component_status_schema"
    assert result.passed
