from __future__ import annotations

import numpy as np
import pytest

from coupled_step2d import (
    advance_coupled_ehd_2d_phase_pair,
    advect_velocity_components_2d_no_through,
    advect_velocity_components_2d_periodic,
    advect_scalar_upwind_2d_no_through,
    advect_scalar_upwind_2d_periodic,
    alpha_gradient_2d,
    capillary_force_density_from_alpha_2d,
    capillary_force_density_from_level_set_2d,
    cell_velocity_to_periodic_faces_2d,
    confine_charge_to_liquid_2d,
    gas_charge_leakage_fraction_2d,
    pressure_gradient_acceleration_2d,
    scalar_area_2d,
    solve_coupled_ehd_2d_phase_pair,
    vof_alpha_level_set_surrogate,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair
from state import ElectrosprayState2D
from interface_geometry import circle_signed_distance, regularized_heaviside


def test_cell_velocity_to_periodic_faces_2d_shapes_and_wraps() -> None:
    ux = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    uy = ux + 10.0

    u_faces, v_faces = cell_velocity_to_periodic_faces_2d(ux, uy)

    assert u_faces.shape == (2, 4)
    assert v_faces.shape == (3, 3)
    np.testing.assert_allclose(u_faces[:, 0], 0.5 * (ux[:, -1] + ux[:, 0]))
    np.testing.assert_allclose(u_faces[:, -1], u_faces[:, 0])
    np.testing.assert_allclose(v_faces[0, :], 0.5 * (uy[-1, :] + uy[0, :]))
    np.testing.assert_allclose(v_faces[-1, :], v_faces[0, :])


def test_advect_scalar_upwind_2d_periodic_conserves_content() -> None:
    scalar = np.array(
        [
            [0.62, 0.66, 0.7, 0.74],
            [0.58, 0.64, 0.72, 0.76],
            [0.6, 0.68, 0.71, 0.73],
        ]
    )
    u_faces = np.ones((3, 5)) * 0.03
    v_faces = np.ones((4, 4)) * -0.02
    dx = 0.25
    dy = 0.2

    updated = advect_scalar_upwind_2d_periodic(scalar, u_faces, v_faces, dx, dy, dt=0.02)

    assert abs(scalar_area_2d(updated, dx, dy) - scalar_area_2d(scalar, dx, dy)) < 1.0e-15
    assert np.min(updated) > 0.0
    assert np.max(updated) < 1.0


def test_advect_scalar_upwind_2d_no_through_conserves_content_and_ignores_wall_flux() -> None:
    scalar = np.array(
        [
            [0.62, 0.66, 0.7, 0.74],
            [0.58, 0.64, 0.72, 0.76],
            [0.6, 0.68, 0.71, 0.73],
        ]
    )
    u_faces = np.ones((3, 5)) * 0.03
    v_faces = np.ones((4, 4)) * -0.02
    u_faces[:, 0] = 10.0
    u_faces[:, -1] = -10.0
    v_faces[0, :] = 8.0
    v_faces[-1, :] = -8.0
    dx = 0.25
    dy = 0.2

    updated = advect_scalar_upwind_2d_no_through(scalar, u_faces, v_faces, dx, dy, dt=0.02)
    updated_without_wall_noise = advect_scalar_upwind_2d_no_through(
        scalar,
        np.ones((3, 5)) * 0.03,
        np.ones((4, 4)) * -0.02,
        dx,
        dy,
        dt=0.02,
    )

    assert abs(scalar_area_2d(updated, dx, dy) - scalar_area_2d(scalar, dx, dy)) < 1.0e-15
    np.testing.assert_allclose(updated, updated_without_wall_noise)
    assert np.min(updated) > 0.0
    assert np.max(updated) < 1.0


def test_advect_velocity_components_2d_periodic_preserves_uniform_velocity() -> None:
    velocity_x = np.ones((4, 5)) * 0.12
    velocity_y = np.ones((4, 5)) * -0.03

    updated_x, updated_y = advect_velocity_components_2d_periodic(
        velocity_x,
        velocity_y,
        dx=0.2,
        dy=0.25,
        dt=0.05,
    )

    np.testing.assert_allclose(updated_x, velocity_x)
    np.testing.assert_allclose(updated_y, velocity_y)


def test_capillary_force_density_from_alpha_2d_uses_csf_gradient() -> None:
    x_faces = np.linspace(0.0, 1.0, 5)
    y_faces = np.linspace(0.0, 1.0, 4)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.2 + 0.3 * xx + 0.4 * yy
    curvature = np.ones_like(alpha) * 2.0

    grad_x, grad_y = alpha_gradient_2d(alpha, dx=x_faces[1] - x_faces[0], dy=y_faces[1] - y_faces[0])
    force_x, force_y = capillary_force_density_from_alpha_2d(
        surface_tension=0.5,
        curvature=curvature,
        alpha_liquid=alpha,
        dx=x_faces[1] - x_faces[0],
        dy=y_faces[1] - y_faces[0],
    )

    np.testing.assert_allclose(grad_x, 0.3)
    np.testing.assert_allclose(grad_y, 0.4)
    np.testing.assert_allclose(force_x, 0.3)
    np.testing.assert_allclose(force_y, 0.4)


def test_capillary_force_density_from_level_set_2d_computes_curvature() -> None:
    x_faces = np.linspace(-1.0, 1.0, 41)
    y_faces = np.linspace(-1.0, 1.0, 41)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    phi = circle_signed_distance(xx, yy, center=(0.0, 0.0), radius=0.5)
    alpha = regularized_heaviside(phi, width=0.1)

    force_x, force_y, curvature = capillary_force_density_from_level_set_2d(
        surface_tension=0.05,
        level_set_phi=phi,
        alpha_liquid=alpha,
        dx=x_faces[1] - x_faces[0],
        dy=y_faces[1] - y_faces[0],
    )

    band = np.abs(phi) < 0.05
    assert abs(float(np.mean(curvature[band])) - 2.0) < 0.08
    assert np.max(np.hypot(force_x, force_y)) > 0.0


def test_pressure_gradient_acceleration_2d_balances_linear_pressure_field() -> None:
    x_faces = np.linspace(0.0, 1.0, 6)
    y_faces = np.linspace(0.0, 0.8, 5)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    pressure = 10.0 + 3.0 * xx - 2.0 * yy
    density = np.ones_like(pressure) * 4.0

    acceleration_x, acceleration_y = pressure_gradient_acceleration_2d(
        pressure,
        density,
        dx=x_faces[1] - x_faces[0],
        dy=y_faces[1] - y_faces[0],
    )

    np.testing.assert_allclose(acceleration_x, -0.75, atol=1.0e-13)
    np.testing.assert_allclose(acceleration_y, 0.5, atol=1.0e-13)


def test_vof_alpha_level_set_surrogate_has_negative_liquid_sign() -> None:
    alpha = np.array([[1.0, 0.5, 0.0]])

    surrogate = vof_alpha_level_set_surrogate(alpha)

    np.testing.assert_allclose(surrogate, np.array([[-0.5, 0.0, 0.5]]))


def test_confine_charge_to_liquid_2d_removes_pure_gas_leakage_conservatively() -> None:
    alpha = np.array([[1.0, 0.5, 0.0], [0.2, 0.0, 0.0]])
    charge = np.array([[1.0, 0.2, 0.4], [0.1, 0.3, 0.0]])
    dx = 0.1
    dy = 0.2

    projected = confine_charge_to_liquid_2d(alpha, charge, dx, dy)

    assert gas_charge_leakage_fraction_2d(alpha, projected, dx, dy) == 0.0
    assert abs(scalar_area_2d(projected, dx, dy) - scalar_area_2d(charge, dx, dy)) < 1.0e-15


def test_advance_coupled_ehd_2d_phase_pair_keeps_transport_invariants() -> None:
    nx = 6
    ny = 4
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.8, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.72 + 0.05 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy / 0.8)
    charge = 0.2 + 0.02 * np.cos(2.0 * np.pi * xx)
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=charge,
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
    )

    assert next_state.shape == state.shape
    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_electric_acceleration > 0.0
    assert diagnostics.max_maxwell_stress_acceleration == 0.0
    assert diagnostics.max_capillary_acceleration == 0.0
    assert diagnostics.density_min == 2.0
    assert diagnostics.density_max == 2.0
    assert diagnostics.velocity_change_norm > 0.0
    assert abs(diagnostics.vof_area_error) < 1.0e-12
    assert diagnostics.gas_charge_leakage_fraction == 0.0
    assert diagnostics.alpha_bounds_violation == 0.0


def test_advance_coupled_ehd_2d_phase_pair_uses_phase_pair_density_field() -> None:
    nx = 6
    ny = 4
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.8, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.25 + 0.5 * xx
    charge = np.ones_like(alpha) * 0.2
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=charge,
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(
            permittivity=2.0,
            conductivity=1.0e-12,
            density=4.0,
            dynamic_viscosity=4.0e-3,
        ),
        gas=LeakyDielectricMaterial(
            permittivity=1.0,
            conductivity=1.0e-12,
            density=1.0,
            dynamic_viscosity=1.0e-3,
        ),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        kinematic_viscosity=0.0,
    )

    assert diagnostics.density_min == pytest.approx(float(np.min(phase_pair.density_field(alpha, 2.0))))
    assert diagnostics.density_max == pytest.approx(float(np.max(phase_pair.density_field(alpha, 2.0))))
    assert diagnostics.kinematic_viscosity_effective == pytest.approx(
        float(np.mean(phase_pair.kinematic_viscosity_field(alpha, 2.0, 0.0)))
    )
    assert diagnostics.max_electric_acceleration > 0.0


def test_advance_coupled_ehd_2d_phase_pair_can_drive_maxwell_stress_force() -> None:
    nx = 8
    ny = 6
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.75, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, _yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.2 + 0.6 * xx
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=np.zeros_like(alpha),
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=4.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        include_maxwell_stress=True,
    )

    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_electric_acceleration == 0.0
    assert diagnostics.max_maxwell_stress_acceleration > 0.0
    assert diagnostics.max_capillary_acceleration == 0.0
    assert diagnostics.velocity_change_norm > 0.0
    assert diagnostics.as_dict()["max_maxwell_stress_acceleration"] == diagnostics.max_maxwell_stress_acceleration
    assert np.max(np.abs(next_state.velocity_x)) > 0.0


def test_advance_coupled_ehd_2d_phase_pair_can_drive_capillary_force_only() -> None:
    nx = 6
    ny = 4
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.8, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.5 + 0.05 * xx + 0.04 * yy
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=np.zeros_like(alpha),
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        surface_tension=0.1,
        curvature_cells=np.ones_like(alpha) * 4.0,
    )

    assert diagnostics.max_electric_field == 0.0
    assert diagnostics.max_electric_acceleration == 0.0
    assert diagnostics.max_capillary_acceleration > 0.0
    assert diagnostics.velocity_change_norm > 0.0
    assert np.max(np.abs(next_state.velocity_x)) > 0.0
    assert np.max(np.abs(next_state.velocity_y)) > 0.0


def test_solve_coupled_ehd_2d_phase_pair_keeps_multistep_state_bounded() -> None:
    nx = 6
    ny = 4
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.8, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.7 + 0.04 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy / 0.8)
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=0.2 + 0.02 * np.cos(2.0 * np.pi * xx),
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    final_state, diagnostics = solve_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=5.0e-5,
        steps=3,
        phase_pair=phase_pair,
        phi_left=1.0,
        phi_right=0.0,
        density=2.0,
        surface_tension=0.05,
        curvature_cells=np.ones_like(alpha) * 2.0,
    )

    assert len(diagnostics) == 3
    assert all(item.max_violation < 1.0e-8 for item in diagnostics)
    assert np.min(final_state.alpha_liquid) >= 0.0
    assert np.max(final_state.alpha_liquid) <= 1.0
    assert np.linalg.norm(final_state.velocity_x) > 0.0


def test_advance_coupled_ehd_2d_phase_pair_damps_velocity_with_viscosity() -> None:
    nx = 16
    ny = 16
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    velocity_x = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    velocity_y = -np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    state = ElectrosprayState2D(
        alpha_liquid=np.ones_like(velocity_x) * 0.8,
        charge_density=np.zeros_like(velocity_x),
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=np.ones_like(velocity_x) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        kinematic_viscosity=0.01,
    )

    assert diagnostics.kinetic_energy_change < 0.0
    assert diagnostics.max_electric_acceleration == 0.0
    assert diagnostics.max_capillary_acceleration == 0.0


def test_advance_coupled_ehd_2d_phase_pair_advects_momentum() -> None:
    nx = 16
    ny = 16
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    velocity_x = 0.08 + 0.02 * np.sin(2.0 * np.pi * yy)
    velocity_y = 0.03 * np.cos(2.0 * np.pi * xx)
    state = ElectrosprayState2D(
        alpha_liquid=np.ones_like(velocity_x) * 0.8,
        charge_density=np.zeros_like(velocity_x),
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        pressure=np.ones_like(velocity_x) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=2.0e-3,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        advect_momentum=True,
    )

    assert diagnostics.momentum_advection_change_norm > 0.0
    assert diagnostics.max_electric_acceleration == 0.0
    assert diagnostics.max_capillary_acceleration == 0.0
    assert diagnostics.as_dict()["momentum_advection_change_norm"] == diagnostics.momentum_advection_change_norm


def test_advect_velocity_components_2d_no_through_uses_wall_fluxes() -> None:
    nx = 8
    ny = 8
    x_centers = (np.arange(nx) + 0.5) / nx
    y_centers = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x_centers, y_centers)
    velocity_x = 0.05 + 0.01 * xx + 0.02 * yy
    velocity_y = -0.02 + 0.03 * xx - 0.01 * yy

    periodic_x, periodic_y = advect_velocity_components_2d_periodic(velocity_x, velocity_y, 1.0 / nx, 1.0 / ny, 1.0e-2)
    wall_x, wall_y = advect_velocity_components_2d_no_through(velocity_x, velocity_y, 1.0 / nx, 1.0 / ny, 1.0e-2)

    assert np.linalg.norm(wall_x - velocity_x) > 0.0
    assert np.linalg.norm(wall_y - velocity_y) > 0.0
    assert np.linalg.norm(wall_x - periodic_x) > 0.0
    assert np.linalg.norm(wall_y - periodic_y) > 0.0


def test_advance_coupled_ehd_2d_phase_pair_projects_velocity_divergence() -> None:
    nx = 16
    ny = 16
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    state = ElectrosprayState2D(
        alpha_liquid=np.ones((ny, nx)) * 0.8,
        charge_density=np.zeros((ny, nx)),
        velocity_x=np.sin(2.0 * np.pi * xx),
        velocity_y=np.cos(2.0 * np.pi * yy),
        pressure=np.ones((ny, nx)) * 1.0e5,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        project_velocity=True,
    )

    assert diagnostics.predictor_divergence_norm > 0.0
    assert diagnostics.projected_divergence_norm < 1.0e-10 * diagnostics.predictor_divergence_norm
    assert diagnostics.pressure_correction_norm > 0.0
    assert diagnostics.projection_velocity_update_norm > 0.0
    assert diagnostics.as_dict()["pressure_correction_norm"] == diagnostics.pressure_correction_norm
    assert diagnostics.as_dict()["projection_velocity_update_norm"] == diagnostics.projection_velocity_update_norm


def test_advance_coupled_ehd_2d_phase_pair_projects_no_through_wall_velocity() -> None:
    nx = 24
    ny = 20
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    state = ElectrosprayState2D(
        alpha_liquid=np.ones((ny, nx)) * 0.8,
        charge_density=np.zeros((ny, nx)),
        velocity_x=np.sin(np.pi * xx) + 0.2 * np.cos(2.0 * np.pi * yy),
        velocity_y=0.8 * np.sin(np.pi * yy) + 0.1 * np.cos(2.0 * np.pi * xx),
        pressure=np.ones((ny, nx)) * 1.0e5,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        project_velocity=True,
        projection_boundary="no_through_wall",
    )

    assert diagnostics.predictor_divergence_norm > 0.0
    assert diagnostics.projected_divergence_norm < 0.6 * diagnostics.predictor_divergence_norm
    assert diagnostics.pressure_correction_norm > 0.0
    assert diagnostics.projection_velocity_update_norm > 0.0
    assert np.max(np.abs(next_state.velocity_x[:, [0, -1]])) < 1.0e-14
    assert np.max(np.abs(next_state.velocity_y[[0, -1], :])) < 1.0e-14


def test_advance_coupled_ehd_2d_phase_pair_accepts_top_bottom_electrodes() -> None:
    nx = 10
    ny = 12
    x_faces = np.linspace(0.0, 0.4, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    alpha = np.ones((ny, nx)) * 0.8
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=np.ones_like(alpha) * 0.2,
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 1.0e5,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=3.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=3.0, conductivity=1.0e-12),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-4,
        phase_pair=phase_pair,
        phi_left=None,
        phi_right=None,
        phi_bottom=2.0,
        phi_top=8.0,
        density=2.0,
    )

    assert diagnostics.max_electric_field > 0.0
    assert diagnostics.max_electric_acceleration > 0.0
    assert diagnostics.velocity_change_norm > 0.0
    assert diagnostics.max_violation <= 1.0e-12


def test_advance_coupled_ehd_2d_phase_pair_accepts_no_through_transport() -> None:
    nx = 12
    ny = 10
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    alpha = 0.5 + 0.2 * np.sin(np.pi * xx) * np.sin(np.pi * yy)
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=0.1 * alpha,
        velocity_x=np.ones_like(alpha) * 0.04,
        velocity_y=np.ones_like(alpha) * -0.03,
        pressure=np.ones_like(alpha) * 1.0e5,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-3,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        advect_momentum=False,
        transport_boundary="no_through_wall",
    )

    assert abs(scalar_area_2d(next_state.alpha_liquid, x_faces[1] - x_faces[0], y_faces[1] - y_faces[0]) - scalar_area_2d(alpha, x_faces[1] - x_faces[0], y_faces[1] - y_faces[0])) < 1.0e-12
    assert diagnostics.alpha_bounds_violation == 0.0
    assert diagnostics.gas_charge_leakage_fraction == 0.0


def test_advance_coupled_ehd_2d_phase_pair_accepts_level_set_curvature() -> None:
    nx = 24
    ny = 24
    x_faces = np.linspace(-1.0, 1.0, nx + 1)
    y_faces = np.linspace(-1.0, 1.0, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, yy = np.meshgrid(x_centers, y_centers)
    phi = circle_signed_distance(xx, yy, center=(0.0, 0.0), radius=0.45)
    alpha = regularized_heaviside(phi, width=0.15)
    state = ElectrosprayState2D(
        alpha_liquid=alpha,
        charge_density=np.zeros_like(alpha),
        velocity_x=np.zeros_like(alpha),
        velocity_y=np.zeros_like(alpha),
        pressure=np.ones_like(alpha) * 10.0,
    )
    phase_pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=2.0, conductivity=1.0e-12),
        gas=LeakyDielectricMaterial(permittivity=1.0, conductivity=1.0e-12),
    )

    _next_state, diagnostics = advance_coupled_ehd_2d_phase_pair(
        state,
        x_faces=x_faces,
        y_faces=y_faces,
        dt=1.0e-5,
        phase_pair=phase_pair,
        phi_left=0.0,
        phi_right=0.0,
        density=2.0,
        surface_tension=0.05,
        level_set_phi=phi,
    )

    assert diagnostics.max_electric_field == 0.0
    assert diagnostics.max_capillary_acceleration > 0.0
    assert diagnostics.velocity_change_norm > 0.0
    assert diagnostics.max_violation < 1.0e-8
