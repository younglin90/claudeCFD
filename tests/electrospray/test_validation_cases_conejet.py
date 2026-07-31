from __future__ import annotations

from validation_cases_conejet import (
    run_all_cone_jet_observable_cases,
    run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_case,
    run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_grid_refinement_case,
    run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_iteration_refinement_case,
    run_axisymmetric_cone_jet_advected_combined_momentum_grid_refinement_case,
    run_axisymmetric_cone_jet_advected_force_kinematic_charge_cotransport_case,
    run_axisymmetric_cone_jet_advected_force_kinematic_grid_refinement_case,
    run_axisymmetric_cone_jet_advected_force_kinematic_interface_case,
    run_axisymmetric_cone_jet_combined_momentum_grid_refinement_case,
    run_axisymmetric_cone_jet_combined_momentum_predictor_case,
    run_axisymmetric_cone_jet_force_driven_interface_case,
    run_axisymmetric_cone_jet_force_kinematic_interface_case,
    run_axisymmetric_cone_jet_state_evolution_case,
    run_axisymmetric_cone_jet_grid_refinement_case,
    run_axisymmetric_cone_jet_momentum_advection_predictor_case,
    run_axisymmetric_cone_jet_open_outflow_accounting_case,
    run_axisymmetric_cone_jet_viscous_momentum_predictor_case,
    run_axisymmetric_cone_jet_vof_charge_transport_case,
    run_cone_jet_charge_to_mass_case,
    run_cone_jet_current_case,
    run_cone_jet_diameter_case,
    run_cone_jet_error_budget_table_case,
    run_cone_jet_quantitative_reference_case,
    run_droplet_sauter_mean_case,
)


def test_cone_jet_current_validation_case_passes() -> None:
    result = run_cone_jet_current_case()

    assert result.case_id == "2d_cone_jet_current"
    assert result.passed


def test_cone_jet_diameter_validation_case_passes() -> None:
    result = run_cone_jet_diameter_case()

    assert result.case_id == "2d_cone_jet_diameter"
    assert result.passed


def test_droplet_sauter_mean_validation_case_passes() -> None:
    result = run_droplet_sauter_mean_case()

    assert result.case_id == "2d_cone_jet_sauter_mean"
    assert result.passed


def test_cone_jet_charge_to_mass_validation_case_passes() -> None:
    result = run_cone_jet_charge_to_mass_case()

    assert result.case_id == "2d_cone_jet_charge_to_mass"
    assert result.passed


def test_cone_jet_quantitative_reference_validation_case_passes() -> None:
    result = run_cone_jet_quantitative_reference_case()

    assert result.case_id == "2d_cone_jet_quantitative_reference"
    assert result.passed


def test_cone_jet_error_budget_table_validation_case_passes() -> None:
    result = run_cone_jet_error_budget_table_case()

    assert result.case_id == "2d_cone_jet_error_budget_table"
    assert result.passed


def test_axisymmetric_cone_jet_state_evolution_case_passes() -> None:
    result = run_axisymmetric_cone_jet_state_evolution_case()

    assert result.case_id == "2d_cone_jet_stateful_evolution"
    assert result.passed


def test_axisymmetric_cone_jet_vof_charge_transport_case_passes() -> None:
    result = run_axisymmetric_cone_jet_vof_charge_transport_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_vof_charge_transport"
    assert result.passed


def test_axisymmetric_cone_jet_open_outflow_accounting_case_passes() -> None:
    result = run_axisymmetric_cone_jet_open_outflow_accounting_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_open_outflow_accounting"
    assert result.passed


def test_axisymmetric_cone_jet_grid_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_grid_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_grid_refinement"
    assert result.passed


def test_axisymmetric_cone_jet_viscous_momentum_predictor_case_passes() -> None:
    result = run_axisymmetric_cone_jet_viscous_momentum_predictor_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_viscous_momentum_predictor"
    assert result.passed


def test_axisymmetric_cone_jet_momentum_advection_predictor_case_passes() -> None:
    result = run_axisymmetric_cone_jet_momentum_advection_predictor_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_momentum_advection_predictor"
    assert result.passed


def test_axisymmetric_cone_jet_combined_momentum_predictor_case_passes() -> None:
    result = run_axisymmetric_cone_jet_combined_momentum_predictor_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_combined_momentum_predictor"
    assert result.passed


def test_axisymmetric_cone_jet_combined_momentum_grid_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_combined_momentum_grid_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_combined_momentum_grid_refinement"
    assert result.passed


def test_axisymmetric_cone_jet_force_driven_interface_case_passes() -> None:
    result = run_axisymmetric_cone_jet_force_driven_interface_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_force_driven_interface"
    assert result.passed


def test_axisymmetric_cone_jet_force_kinematic_interface_case_passes() -> None:
    result = run_axisymmetric_cone_jet_force_kinematic_interface_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_force_kinematic_interface"
    assert result.passed


def test_axisymmetric_cone_jet_advected_force_kinematic_interface_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_force_kinematic_interface_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_force_kinematic_interface"
    assert result.passed


def test_axisymmetric_cone_jet_advected_force_kinematic_charge_cotransport_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_force_kinematic_charge_cotransport_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport"
    assert result.passed


def test_axisymmetric_cone_jet_advected_force_kinematic_grid_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_force_kinematic_grid_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement"
    assert result.passed


def test_axisymmetric_cone_jet_advected_combined_momentum_grid_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_combined_momentum_grid_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement"
    assert result.passed


def test_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz"
    assert result.passed


def test_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_grid_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_grid_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement"
    assert result.passed


def test_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_iteration_refinement_case_passes() -> None:
    result = run_axisymmetric_cone_jet_advected_combined_momentum_huh_wirz_iteration_refinement_case()

    assert result.case_id == "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement"
    assert result.passed


def test_all_cone_jet_observable_cases_pass() -> None:
    results = run_all_cone_jet_observable_cases()

    assert len(results) == 23
    assert all(result.passed for result in results)
