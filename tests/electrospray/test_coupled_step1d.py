from __future__ import annotations

import numpy as np

from coupled_step1d import (
    advance_coupled_ehd_1d_phase_pair,
    advance_reduced_electrospray_1d,
    advance_reduced_electrospray_1d_phase_pair,
    cell_velocity_to_periodic_faces,
    reduced_step_diagnostics,
    solve_coupled_ehd_1d_phase_pair,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair
from state import ElectrosprayState1D, free_charge_loss_fraction, total_free_charge, validate_state_bounds
from vof_transport import vof_mass


def test_reduced_step_preserves_vof_mass_and_state_bounds() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.array([1.0, 1.0, 0.5, 0.0, 0.0]),
        charge_density=np.array([2.0, 0.0, 0.0, 0.0, 0.0]),
        velocity=np.zeros(5),
        pressure=np.ones(5),
    )
    dx = 0.2
    velocity_faces = np.ones(6) * 0.2
    next_state = advance_reduced_electrospray_1d(state, velocity_faces, dx, dt=0.1, epsilon=2.0, conductivity=0.0)

    validate_state_bounds(next_state)
    assert abs(vof_mass(next_state.alpha_liquid, dx) - vof_mass(state.alpha_liquid, dx)) < 1.0e-15


def test_reduced_step_confines_charge_to_liquid_cells() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.array([1.0, 0.5, 0.0, 0.0]),
        charge_density=np.array([1.0, 0.0, 0.0, 0.0]),
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )
    next_state = advance_reduced_electrospray_1d(state, np.ones(5) * 0.4, dx=0.25, dt=0.1, epsilon=1.0, conductivity=0.0)
    assert np.all(next_state.charge_density[next_state.alpha_liquid == 0.0] == 0.0)


def test_reduced_step_relaxes_total_free_charge() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(4),
        charge_density=np.ones(4) * 2.0,
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )
    dx = 0.25
    next_state = advance_reduced_electrospray_1d(
        state,
        velocity_faces=np.zeros(5),
        dx=dx,
        dt=0.5,
        epsilon=1.0,
        conductivity=3.0,
    )
    assert total_free_charge(next_state, dx) < total_free_charge(state, dx)


def test_phase_pair_reduced_step_uses_vof_mixed_relaxation() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(4),
        charge_density=np.ones(4) * 2.0,
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=3.0, conductivity=3.0),
    )
    dx = 0.25

    next_state = advance_reduced_electrospray_1d_phase_pair(
        state,
        velocity_faces=np.zeros(5),
        dx=dx,
        dt=0.5,
        phase_pair=phase_pair,
    )

    validate_state_bounds(next_state)
    assert total_free_charge(next_state, dx) < total_free_charge(state, dx)
    assert free_charge_loss_fraction(state, next_state, dx) > 0.0
    diagnostics = reduced_step_diagnostics(state, next_state, dx)
    assert diagnostics.vof_mass_error < 1.0e-15
    assert diagnostics.free_charge_loss_fraction > 0.0
    assert diagnostics.min_charge_density >= 0.0
    assert diagnostics.max_gas_charge_density == 0.0
    assert diagnostics.alpha_bounds_violation == 0.0
    assert diagnostics.max_violation < 1.0e-15
    assert diagnostics.as_dict()["max_violation"] == diagnostics.max_violation


def test_reduced_step_diagnostics_include_gas_charge_confinement() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(4),
        charge_density=np.ones(4),
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )
    next_state = ElectrosprayState1D(
        alpha_liquid=np.array([1.0, 0.0, 0.0, 1.0]),
        charge_density=np.array([0.5, 2.0e-4, -3.0e-4, 0.5]),
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )

    diagnostics = reduced_step_diagnostics(state, next_state, dx=0.25)

    assert diagnostics.max_gas_charge_density == 3.0e-4
    assert diagnostics.as_dict()["max_gas_charge_density"] == diagnostics.max_gas_charge_density
    assert diagnostics.max_violation >= diagnostics.max_gas_charge_density


def test_cell_velocity_to_periodic_faces_averages_adjacent_cells() -> None:
    velocity = np.array([1.0, 3.0, 5.0])
    faces = cell_velocity_to_periodic_faces(velocity)

    np.testing.assert_allclose(faces, np.array([3.0, 2.0, 4.0, 3.0]))


def test_coupled_ehd_step_updates_velocity_from_electric_body_force() -> None:
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(4),
        charge_density=np.ones(4) * 0.2,
        velocity=np.zeros(4),
        pressure=np.ones(4),
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=0.0 + 1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
    )
    x_faces = np.linspace(0.0, 1.0, 5)

    next_state, diagnostics = advance_coupled_ehd_1d_phase_pair(
        state,
        x_faces=x_faces,
        dt=0.01,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        project_velocity=False,
    )

    validate_state_bounds(next_state)
    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_electric_acceleration > 0.0
    assert diagnostics.velocity_change_norm > 0.0
    assert np.linalg.norm(next_state.velocity) > 0.0
    assert abs(vof_mass(next_state.alpha_liquid, 0.25) - vof_mass(state.alpha_liquid, 0.25)) < 1.0e-5
    assert diagnostics.as_dict()["velocity_change_norm"] == diagnostics.velocity_change_norm


def test_coupled_ehd_step_projection_reduces_velocity_divergence() -> None:
    n = 64
    x_faces = np.linspace(0.0, 1.0, n + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(n),
        charge_density=0.2 + 0.1 * np.sin(2.0 * np.pi * x_centers),
        velocity=np.zeros(n),
        pressure=np.ones(n) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_1d_phase_pair(
        state,
        x_faces=x_faces,
        dt=0.001,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        project_velocity=True,
    )

    validate_state_bounds(next_state)
    assert diagnostics.predictor_divergence_norm > 0.0
    assert diagnostics.projected_divergence_norm < 0.2 * diagnostics.predictor_divergence_norm
    assert diagnostics.as_dict()["predictor_divergence_norm"] == diagnostics.predictor_divergence_norm


def test_coupled_ehd_multistep_solver_preserves_bounds_and_records_diagnostics() -> None:
    n = 32
    x_faces = np.linspace(0.0, 1.0, n + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    state = ElectrosprayState1D(
        alpha_liquid=np.ones(n),
        charge_density=0.2 + 0.05 * np.sin(2.0 * np.pi * x_centers),
        velocity=np.zeros(n),
        pressure=np.ones(n) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
    )

    final_state, diagnostics = solve_coupled_ehd_1d_phase_pair(
        state,
        x_faces=x_faces,
        dt=0.0005,
        steps=3,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        project_velocity=True,
    )

    validate_state_bounds(final_state)
    assert len(diagnostics) == 3
    assert all(item.max_electric_field > 0.0 for item in diagnostics)
    assert all(item.projected_divergence_norm < item.predictor_divergence_norm for item in diagnostics)
    assert abs(vof_mass(final_state.alpha_liquid, 1.0 / n) - vof_mass(state.alpha_liquid, 1.0 / n)) < 1.0e-4
