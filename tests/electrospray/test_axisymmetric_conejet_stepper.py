import pytest
import numpy as np

from axisymmetric_conejet_stepper import (
    advance_axisymmetric_conejet_state,
    force_kinematic_radius_profile,
    force_driven_radius_profile,
    initialize_axisymmetric_conejet_state,
    outflow_jet_diameter,
    run_axisymmetric_conejet_evolution,
)


def test_axisymmetric_conejet_stepper_updates_interface_and_confines_charge() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    next_state, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
    )

    assert diagnostics.max_radius_change > 0.0
    assert diagnostics.final_outflow_jet_diameter < diagnostics.initial_outflow_jet_diameter
    assert diagnostics.interface_kinematic_relative_residual < 0.05
    assert diagnostics.transported_vof_volume_error < 1.0e-12
    assert diagnostics.transported_charge_error < 1.0e-12
    assert diagnostics.transported_charge_ratio_error < 1.0e-12
    assert diagnostics.transported_alpha_bounds_violation < 1.0e-12
    assert diagnostics.electric_pressure_monotonicity_violation < 1.0e-12
    assert diagnostics.outlet_electric_to_capillary_pressure_ratio > 1.0
    assert diagnostics.electric_focusing_correlation > 0.99
    assert diagnostics.axial_pressure_force_balance_residual < 1.0e-12
    assert diagnostics.volume_accounting_residual < 1.0e-24
    assert diagnostics.max_gas_charge_density == pytest.approx(0.0)
    assert diagnostics.min_alpha >= 0.0
    assert diagnostics.max_alpha <= 1.0
    assert diagnostics.pressure_correction_norm > 0.0
    assert diagnostics.max_pressure_balance_residual < 1.0e-12
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm
    assert diagnostics.radial_projection_update_norm > 0.0
    assert diagnostics.axial_projection_update_norm > 0.0
    assert diagnostics.viscous_velocity_update_norm == pytest.approx(0.0)
    assert diagnostics.viscous_step_dt == pytest.approx(0.0)
    assert diagnostics.momentum_advection_update_norm == pytest.approx(0.0)
    assert diagnostics.momentum_advection_step_dt == pytest.approx(0.0)
    assert diagnostics.force_driven_radius_update_norm == pytest.approx(0.0)
    assert diagnostics.max_radial_force_imbalance == pytest.approx(0.0)
    assert diagnostics.force_driven_interface_speed_norm == pytest.approx(0.0)
    assert diagnostics.force_driven_acceleration_norm == pytest.approx(0.0)
    assert diagnostics.force_driven_displacement_cfl_fraction == pytest.approx(0.0)
    assert diagnostics.interface_transport_state_volume_error == pytest.approx(0.0)
    assert diagnostics.interface_transport_state_bounds_violation == pytest.approx(0.0)
    assert diagnostics.interface_transport_state_charge_error == pytest.approx(0.0)
    assert diagnostics.interface_transport_state_charge_ratio_error == pytest.approx(0.0)
    assert diagnostics.interface_transport_state_gas_charge_density == pytest.approx(0.0)
    assert outflow_jet_diameter(next_state) == pytest.approx(diagnostics.final_outflow_jet_diameter)


def test_force_driven_radius_profile_focuses_when_electric_pressure_exceeds_capillary() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    z_faces = np.linspace(0.0, 4.0 * nozzle_radius, 49)
    old_radius = np.ones(48) * nozzle_radius

    new_radius, update_norm, force_imbalance = force_driven_radius_profile(
        old_radius,
        z_faces,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        relaxation_fraction=0.30,
        surface_tension=0.03,
        electric_pressure_scale=2.0e3,
    )

    assert update_norm > 0.0
    assert force_imbalance > 0.0
    assert new_radius[-1] < old_radius[-1]
    assert new_radius[0] <= old_radius[0]
    assert new_radius[-1] >= jet_radius


def test_force_kinematic_radius_profile_reports_acceleration_and_limited_motion() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    z_faces = np.linspace(0.0, 4.0 * nozzle_radius, 49)
    old_radius = np.ones(48) * nozzle_radius

    new_radius, update_norm, force_imbalance, speed_norm, acceleration_norm, cfl_fraction = (
        force_kinematic_radius_profile(
            old_radius,
            z_faces,
            nozzle_radius=nozzle_radius,
            jet_radius=jet_radius,
            relaxation_fraction=0.30,
            surface_tension=0.03,
            electric_pressure_scale=2.0e3,
            density=973.0,
            dt=1.0e-3,
        )
    )

    assert update_norm > 0.0
    assert force_imbalance > 0.0
    assert speed_norm > 0.0
    assert acceleration_norm > 0.0
    assert 0.0 < cfl_fraction <= 1.0
    assert new_radius[-1] < old_radius[-1]
    assert new_radius[-1] >= jet_radius


def test_axisymmetric_conejet_stepper_executes_viscous_momentum_predictor() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    _, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        kinematic_viscosity=1.0e-6,
    )

    assert diagnostics.viscous_velocity_update_norm > 0.0
    assert diagnostics.viscous_step_dt > 0.0
    assert diagnostics.viscous_kinetic_energy_change <= 1.0e-12
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm


def test_axisymmetric_conejet_stepper_executes_momentum_advection_predictor() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    _, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        advect_momentum=True,
    )

    assert diagnostics.momentum_advection_update_norm > 0.0
    assert diagnostics.momentum_advection_step_dt > 0.0
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm


def test_axisymmetric_conejet_stepper_executes_combined_momentum_predictor_path() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    _, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        advect_momentum=True,
        kinematic_viscosity=1.0e-6,
    )

    assert diagnostics.momentum_advection_update_norm > 0.0
    assert diagnostics.viscous_velocity_update_norm > 0.0
    assert diagnostics.momentum_advection_step_dt > 0.0
    assert diagnostics.viscous_step_dt > 0.0
    assert diagnostics.viscous_kinetic_energy_change <= 1.0e-12
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm


def test_axisymmetric_conejet_stepper_executes_force_driven_interface_path() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    _, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radius_update_mode="force",
    )

    assert diagnostics.force_driven_radius_update_norm > 0.0
    assert diagnostics.max_radial_force_imbalance > 0.0
    assert diagnostics.final_outflow_jet_diameter < diagnostics.initial_outflow_jet_diameter
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm


def test_axisymmetric_conejet_stepper_executes_force_kinematic_interface_path() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    _, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radius_update_mode="force_kinematic",
    )

    assert diagnostics.force_driven_radius_update_norm > 0.0
    assert diagnostics.max_radial_force_imbalance > 0.0
    assert diagnostics.force_driven_interface_speed_norm > 0.0
    assert diagnostics.force_driven_acceleration_norm > 0.0
    assert 0.0 < diagnostics.force_driven_displacement_cfl_fraction <= 1.0
    assert diagnostics.final_outflow_jet_diameter < diagnostics.initial_outflow_jet_diameter
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm


def test_axisymmetric_conejet_stepper_advects_interface_state_with_force_kinematics() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    state = initialize_axisymmetric_conejet_state(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radial_cells=48,
        axial_cells=48,
    )

    next_state, diagnostics = advance_axisymmetric_conejet_state(
        state,
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        radius_update_mode="force_kinematic",
        advect_interface=True,
    )

    assert diagnostics.force_driven_radius_update_norm > 0.0
    assert diagnostics.interface_transport_state_volume_error < 1.0e-12
    assert diagnostics.interface_transport_state_bounds_violation < 1.0e-12
    assert diagnostics.interface_transport_state_charge_error < 1.0e-12
    assert diagnostics.interface_transport_state_charge_ratio_error < 1.0e-12
    assert diagnostics.interface_transport_state_gas_charge_density < 1.0e-12
    assert diagnostics.projected_divergence_norm < diagnostics.predictor_divergence_norm
    assert np.linalg.norm(next_state.alpha_liquid - state.alpha_liquid) > 0.0


def test_axisymmetric_conejet_evolution_reaches_huh_wirz_observable_range() -> None:
    nozzle_radius = 55.0e-6
    jet_radius = nozzle_radius / (47.6 * 6.62) ** 0.5
    evolution = run_axisymmetric_conejet_evolution(
        nozzle_radius=nozzle_radius,
        jet_radius=jet_radius,
        density=973.0,
        axial_velocity=1.70e-3,
        charge_to_mass=620.0,
        steps=48,
        radial_cells=96,
        axial_cells=96,
    )

    final_diagnostics = evolution.diagnostics[-1]
    assert evolution.monotone_focusing is True
    assert evolution.jet_diameter_history[0] > evolution.jet_diameter_history[-1]
    assert evolution.jet_diameter_history[-1] == pytest.approx(6.20166475902742e-6)
    assert final_diagnostics.emitted_current == pytest.approx(3.092884917036496e-8)
    assert final_diagnostics.charge_to_mass_ratio == pytest.approx(620.0)
    assert final_diagnostics.axial_mass_flux_relative_variation < 1.0e-12
    assert final_diagnostics.axial_current_relative_variation < 1.0e-12
    assert final_diagnostics.interface_kinematic_relative_residual < 0.11
    assert final_diagnostics.transported_vof_volume_error < 1.0e-12
    assert final_diagnostics.transported_charge_error < 1.0e-12
    assert final_diagnostics.transported_charge_ratio_error < 1.0e-12
    assert final_diagnostics.transported_alpha_bounds_violation < 1.0e-12
    assert final_diagnostics.electric_pressure_monotonicity_violation < 1.0e-12
    assert final_diagnostics.outlet_electric_to_capillary_pressure_ratio > 3.0
    assert final_diagnostics.electric_focusing_correlation > 0.80
    assert final_diagnostics.axial_pressure_force_balance_residual < 1.0e-12
    assert final_diagnostics.volume_accounting_residual < 1.0e-24
    assert final_diagnostics.pressure_correction_norm > 0.0
    assert final_diagnostics.max_pressure_balance_residual < 1.0e-12
    assert final_diagnostics.projected_divergence_norm < final_diagnostics.predictor_divergence_norm
    assert final_diagnostics.radial_projection_update_norm > 0.0
    assert final_diagnostics.axial_projection_update_norm > 0.0
    assert final_diagnostics.viscous_velocity_update_norm == pytest.approx(0.0)
    assert final_diagnostics.momentum_advection_update_norm == pytest.approx(0.0)
