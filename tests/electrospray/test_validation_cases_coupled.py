from __future__ import annotations

import json

from validation_cases_coupled import (
    failed_reduced_step_invariants,
    reduced_phase_pair_step_diagnostics,
    reduced_phase_pair_step_report,
    reduced_phase_pair_step_report_json,
    reduced_phase_pair_step_scenario,
    reduced_step_invariant_status,
    run_no_through_wall_projection_2d_case,
    run_all_coupled_cases,
    run_confined_charge_leakage_fraction_case,
    run_coupled_ehd_2d_bounded_domain_multiphysics_case,
    run_coupled_ehd_2d_bounded_domain_grid_refinement_case,
    run_coupled_ehd_2d_capillary_step_case,
    run_coupled_ehd_2d_incompressible_ns_taylor_green_case,
    run_coupled_ehd_2d_level_set_capillary_case,
    run_coupled_ehd_2d_maxwell_stress_force_case,
    run_coupled_ehd_2d_droplet_deformation_evolution_case,
    run_coupled_ehd_2d_droplet_deformation_grid_refinement_case,
    run_coupled_ehd_2d_droplet_deformation_time_history_case,
    run_coupled_ehd_2d_momentum_advection_case,
    run_coupled_ehd_2d_momentum_budget_closure_case,
    run_coupled_ehd_2d_multistep_case,
    run_coupled_ehd_2d_no_through_transport_case,
    run_coupled_ehd_2d_no_through_momentum_advection_case,
    run_coupled_ehd_2d_no_through_wall_projection_case,
    run_coupled_ehd_2d_same_operator_charge_transport_case,
    run_coupled_ehd_2d_pressure_capillary_force_balance_case,
    run_coupled_ehd_2d_pressure_electric_force_balance_case,
    run_coupled_ehd_2d_pressure_maxwell_force_balance_case,
    run_coupled_ehd_2d_projection_case,
    run_coupled_ehd_2d_dielectric_maxwell_droplet_deformation_case,
    run_coupled_ehd_2d_dielectric_maxwell_droplet_grid_refinement_case,
    run_coupled_ehd_2d_dielectric_maxwell_droplet_timestep_refinement_case,
    run_coupled_ehd_2d_dielectric_maxwell_droplet_voltage_scaling_case,
    run_coupled_ehd_2d_refreshed_interface_capillary_case,
    run_coupled_ehd_2d_step_case,
    run_coupled_ehd_2d_top_bottom_electrode_case,
    run_coupled_ehd_2d_two_phase_density_case,
    run_coupled_ehd_2d_two_phase_ns_momentum_grid_refinement_case,
    run_coupled_ehd_2d_two_phase_ns_momentum_kernel_case,
    run_coupled_ehd_2d_variable_density_projection_case,
    run_coupled_ehd_2d_variable_viscosity_case,
    run_coupled_ehd_2d_viscous_damping_case,
    run_coupled_ehd_momentum_step_case,
    run_coupled_ehd_multistep_case,
    run_coupled_ehd_projection_case,
    run_reduced_phase_pair_step_case,
)


def test_reduced_phase_pair_step_validation_case_passes() -> None:
    assert run_reduced_phase_pair_step_case().passed


def test_confined_charge_leakage_fraction_validation_case_passes() -> None:
    result = run_confined_charge_leakage_fraction_case()

    assert result.case_id == "1d_confined_charge_leakage_fraction"
    assert result.passed


def test_coupled_ehd_momentum_step_validation_case_passes() -> None:
    result = run_coupled_ehd_momentum_step_case()

    assert result.case_id == "1d_coupled_ehd_momentum_step"
    assert result.passed


def test_coupled_ehd_projection_validation_case_passes() -> None:
    result = run_coupled_ehd_projection_case()

    assert result.case_id == "1d_coupled_ehd_projection"
    assert result.passed


def test_coupled_ehd_multistep_validation_case_passes() -> None:
    result = run_coupled_ehd_multistep_case()

    assert result.case_id == "1d_coupled_ehd_multistep"
    assert result.passed


def test_coupled_ehd_2d_step_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_step_case()

    assert result.case_id == "2d_coupled_ehd_step"
    assert result.passed


def test_coupled_ehd_2d_capillary_step_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_capillary_step_case()

    assert result.case_id == "2d_coupled_ehd_capillary_step"
    assert result.passed


def test_coupled_ehd_2d_multistep_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_multistep_case()

    assert result.case_id == "2d_coupled_ehd_multistep"
    assert result.passed


def test_coupled_ehd_2d_level_set_capillary_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_level_set_capillary_case()

    assert result.case_id == "2d_coupled_ehd_level_set_capillary"
    assert result.passed


def test_coupled_ehd_2d_viscous_damping_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_viscous_damping_case()

    assert result.case_id == "2d_coupled_ehd_viscous_damping"
    assert result.passed


def test_coupled_ehd_2d_momentum_advection_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_momentum_advection_case()

    assert result.case_id == "2d_coupled_ehd_momentum_advection"
    assert result.passed


def test_coupled_ehd_2d_momentum_budget_closure_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_momentum_budget_closure_case()

    assert result.case_id == "2d_coupled_ehd_momentum_budget_closure"
    assert result.passed


def test_coupled_ehd_2d_incompressible_ns_taylor_green_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_incompressible_ns_taylor_green_case()

    assert result.case_id == "2d_coupled_ehd_incompressible_ns_taylor_green"
    assert result.passed


def test_coupled_ehd_2d_two_phase_ns_momentum_kernel_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_two_phase_ns_momentum_kernel_case()

    assert result.case_id == "2d_coupled_ehd_two_phase_ns_momentum_kernel"
    assert result.passed


def test_coupled_ehd_2d_two_phase_ns_momentum_grid_refinement_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_two_phase_ns_momentum_grid_refinement_case()

    assert result.case_id == "2d_coupled_ehd_two_phase_ns_momentum_grid_refinement"
    assert result.passed


def test_coupled_ehd_2d_two_phase_density_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_two_phase_density_case()

    assert result.case_id == "2d_coupled_ehd_two_phase_density"
    assert result.passed


def test_coupled_ehd_2d_variable_viscosity_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_variable_viscosity_case()

    assert result.case_id == "2d_coupled_ehd_variable_viscosity"
    assert result.passed


def test_coupled_ehd_2d_variable_density_projection_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_variable_density_projection_case()

    assert result.case_id == "2d_coupled_ehd_variable_density_projection"
    assert result.passed


def test_no_through_wall_projection_2d_validation_case_passes() -> None:
    result = run_no_through_wall_projection_2d_case()

    assert result.case_id == "2d_no_through_wall_projection"
    assert result.passed


def test_coupled_ehd_2d_no_through_wall_projection_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_no_through_wall_projection_case()

    assert result.case_id == "2d_coupled_ehd_no_through_wall_projection"
    assert result.passed


def test_coupled_ehd_2d_top_bottom_electrode_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_top_bottom_electrode_case()

    assert result.case_id == "2d_coupled_ehd_top_bottom_electrode"
    assert result.passed


def test_coupled_ehd_2d_no_through_transport_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_no_through_transport_case()

    assert result.case_id == "2d_coupled_ehd_no_through_transport"
    assert result.passed


def test_coupled_ehd_2d_same_operator_charge_transport_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_same_operator_charge_transport_case()

    assert result.case_id == "2d_coupled_ehd_same_operator_charge_transport"
    assert result.passed


def test_coupled_ehd_2d_no_through_momentum_advection_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_no_through_momentum_advection_case()

    assert result.case_id == "2d_coupled_ehd_no_through_momentum_advection"
    assert result.passed


def test_coupled_ehd_2d_bounded_domain_multiphysics_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_bounded_domain_multiphysics_case()

    assert result.case_id == "2d_coupled_ehd_bounded_domain_multiphysics"
    assert result.passed


def test_coupled_ehd_2d_bounded_domain_grid_refinement_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_bounded_domain_grid_refinement_case()

    assert result.case_id == "2d_coupled_ehd_bounded_domain_grid_refinement"
    assert result.passed


def test_coupled_ehd_2d_projection_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_projection_case()

    assert result.case_id == "2d_coupled_ehd_projection"
    assert result.passed


def test_coupled_ehd_2d_pressure_electric_force_balance_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_pressure_electric_force_balance_case()

    assert result.case_id == "2d_coupled_ehd_pressure_electric_force_balance"
    assert result.passed


def test_coupled_ehd_2d_maxwell_stress_force_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_maxwell_stress_force_case()

    assert result.case_id == "2d_coupled_ehd_maxwell_stress_force"
    assert result.passed


def test_coupled_ehd_2d_pressure_maxwell_force_balance_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_pressure_maxwell_force_balance_case()

    assert result.case_id == "2d_coupled_ehd_pressure_maxwell_force_balance"
    assert result.passed


def test_coupled_ehd_2d_pressure_capillary_force_balance_validation_case_passes() -> None:
    result = run_coupled_ehd_2d_pressure_capillary_force_balance_case()

    assert result.case_id == "2d_coupled_ehd_pressure_capillary_force_balance"
    assert result.passed


def test_coupled_ehd_2d_droplet_deformation_evolution_case_passes() -> None:
    result = run_coupled_ehd_2d_droplet_deformation_evolution_case()

    assert result.case_id == "2d_coupled_ehd_droplet_deformation_evolution"
    assert result.passed


def test_coupled_ehd_2d_droplet_deformation_time_history_case_passes() -> None:
    result = run_coupled_ehd_2d_droplet_deformation_time_history_case()

    assert result.case_id == "2d_coupled_ehd_droplet_deformation_time_history"
    assert result.passed


def test_coupled_ehd_2d_dielectric_maxwell_droplet_deformation_case_passes() -> None:
    result = run_coupled_ehd_2d_dielectric_maxwell_droplet_deformation_case()

    assert result.case_id == "2d_coupled_ehd_dielectric_maxwell_droplet_deformation"
    assert result.passed


def test_coupled_ehd_2d_dielectric_maxwell_droplet_voltage_scaling_case_passes() -> None:
    result = run_coupled_ehd_2d_dielectric_maxwell_droplet_voltage_scaling_case()

    assert result.case_id == "2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling"
    assert result.passed


def test_coupled_ehd_2d_dielectric_maxwell_droplet_grid_refinement_case_passes() -> None:
    result = run_coupled_ehd_2d_dielectric_maxwell_droplet_grid_refinement_case()

    assert result.case_id == "2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement"
    assert result.passed


def test_coupled_ehd_2d_dielectric_maxwell_droplet_timestep_refinement_case_passes() -> None:
    result = run_coupled_ehd_2d_dielectric_maxwell_droplet_timestep_refinement_case()

    assert result.case_id == "2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement"
    assert result.passed


def test_coupled_ehd_2d_droplet_deformation_grid_refinement_case_passes() -> None:
    result = run_coupled_ehd_2d_droplet_deformation_grid_refinement_case()

    assert result.case_id == "2d_coupled_ehd_droplet_deformation_grid_refinement"
    assert result.passed


def test_coupled_ehd_2d_refreshed_interface_capillary_case_passes() -> None:
    result = run_coupled_ehd_2d_refreshed_interface_capillary_case()

    assert result.case_id == "2d_coupled_ehd_refreshed_interface_capillary"
    assert result.passed


def test_all_coupled_validation_cases_pass() -> None:
    results = run_all_coupled_cases()

    assert len(results) == 39
    assert all(result.passed for result in results)


def test_reduced_phase_pair_step_diagnostics_are_reportable() -> None:
    diagnostics = reduced_phase_pair_step_diagnostics()

    assert diagnostics["max_violation"] < 1.0e-15
    assert diagnostics["max_gas_charge_density"] == 0.0


def test_reduced_phase_pair_step_report_is_deterministic() -> None:
    report = reduced_phase_pair_step_report()

    assert report["case_id"] == "1d_reduced_phase_pair_step"
    assert report["passed"] is True
    assert report["cell_count"] == 5
    assert report["face_count"] == 6
    assert report["diagnostics"] == reduced_phase_pair_step_diagnostics()
    assert report["failed_invariants"] == []
    assert all(report["invariants"].values())


def test_reduced_phase_pair_step_report_names_all_invariants() -> None:
    report = reduced_phase_pair_step_report()

    assert set(report["invariants"]) == {
        "alpha_bounded",
        "charge_nonnegative",
        "gas_charge_confined",
        "vof_mass_conserved",
    }


def test_reduced_step_invariant_status_flags_failures() -> None:
    diagnostics = {
        "vof_mass_error": 2.0e-3,
        "min_charge_density": -1.0e-3,
        "max_gas_charge_density": 3.0e-3,
        "alpha_bounds_violation": 4.0e-3,
    }
    status = reduced_step_invariant_status(diagnostics, tolerance=1.0e-6)

    assert status == {
        "alpha_bounded": False,
        "charge_nonnegative": False,
        "gas_charge_confined": False,
        "vof_mass_conserved": False,
    }
    assert failed_reduced_step_invariants(diagnostics, tolerance=1.0e-6) == [
        "alpha_bounded",
        "charge_nonnegative",
        "gas_charge_confined",
        "vof_mass_conserved",
    ]


def test_failed_reduced_step_invariants_is_empty_for_valid_diagnostics() -> None:
    assert failed_reduced_step_invariants(reduced_phase_pair_step_diagnostics()) == []


def test_reduced_phase_pair_step_report_json_is_stable() -> None:
    payload = reduced_phase_pair_step_report_json()

    assert payload.endswith("\n")
    assert json.loads(payload) == reduced_phase_pair_step_report()
    assert payload == json.dumps(reduced_phase_pair_step_report(), sort_keys=True) + "\n"


def test_reduced_phase_pair_step_scenario_is_well_formed() -> None:
    state, velocity_faces, dx, dt, phase_pair = reduced_phase_pair_step_scenario()

    assert velocity_faces.shape == (state.size + 1,)
    assert dx > 0.0
    assert dt > 0.0
    assert phase_pair.liquid.conductivity > phase_pair.gas.conductivity
