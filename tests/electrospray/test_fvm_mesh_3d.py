from __future__ import annotations

import numpy as np

from fvm_mesh import (
    advect_scalar_limited_linear_fvm,
    divergence_from_face_flux,
    gauss_gradient,
    least_squares_gradient,
    project_face_flux,
    project_velocity_pimple,
    project_velocity_piso,
    rhie_chow_face_flux,
    solve_diffusion,
    structured_cartesian_mesh_3d,
    two_cell_skew_mesh_3d,
)


def test_structured_cartesian_mesh_gauss_gradient_is_exact_for_linear_field() -> None:
    mesh = structured_cartesian_mesh_3d(3, 2, 2)
    phi = mesh.cell_centers[:, 0]

    gradient = gauss_gradient(mesh, phi, dirichlet_boundary_values={"x_min": 0.0, "x_max": 1.0})

    assert np.max(np.abs(gradient[:, 0] - 1.0)) <= 1.0e-12
    assert np.max(np.abs(gradient[:, 1:])) <= 1.0e-12


def test_least_squares_gradient_is_exact_for_linear_field() -> None:
    mesh = structured_cartesian_mesh_3d(3, 2, 2)
    phi = 2.0 * mesh.cell_centers[:, 0] - 3.0 * mesh.cell_centers[:, 1] + 0.5 * mesh.cell_centers[:, 2]

    gradient = least_squares_gradient(mesh, phi)

    active = np.linalg.norm(gradient, axis=1) > 0.0
    assert np.max(np.abs(gradient[active] - np.array([2.0, -3.0, 0.5]))) <= 1.0e-12


def test_structured_cartesian_mesh_diffusion_solves_linear_potential() -> None:
    mesh = structured_cartesian_mesh_3d(3, 2, 2)

    phi = solve_diffusion(mesh, np.ones(mesh.cell_count), dirichlet_boundary_values={"x_min": 1.0, "x_max": 0.0})

    expected = 1.0 - mesh.cell_centers[:, 0]
    assert np.max(np.abs(phi - expected)) <= 1.0e-12


def test_skew_two_cell_mesh_supports_unstructured_diffusion() -> None:
    mesh = two_cell_skew_mesh_3d()

    phi = solve_diffusion(mesh, np.ones(mesh.cell_count), dirichlet_boundary_values={"x_min": 1.0, "x_max": 0.0})

    assert phi.shape == (2,)
    assert 1.0 > phi[0] > phi[1] > 0.0
    assert not np.isclose(phi[0] - phi[1], 0.0)


def test_face_flux_projection_reduces_closed_domain_divergence() -> None:
    mesh = structured_cartesian_mesh_3d(2, 1, 1)
    face_flux = np.zeros(mesh.face_count)
    internal_face = np.flatnonzero(mesh.internal_faces)[0]
    face_flux[internal_face] = 0.1

    projection = project_face_flux(mesh, face_flux, np.ones(mesh.cell_count), dt=0.25)

    assert projection.initial_divergence_norm > 0.0
    assert projection.projected_divergence_norm <= 1.0e-12
    assert projection.divergence_reduction_ratio <= 1.0e-12


def test_limited_linear_advection_is_conservative_and_bounded_on_closed_mesh() -> None:
    mesh = structured_cartesian_mesh_3d(3, 1, 1)
    scalar = np.array([0.2, 0.6, 0.9])
    flux = np.zeros(mesh.face_count)
    flux[np.flatnonzero(mesh.internal_faces)] = np.array([0.05, -0.02])

    updated = advect_scalar_limited_linear_fvm(mesh, scalar, flux, dt=0.1, clip_bounds=(0.0, 1.0))

    assert np.all((updated >= 0.0) & (updated <= 1.0))
    assert np.sum(updated * mesh.cell_volumes) == np.sum(scalar * mesh.cell_volumes)


def test_rhie_chow_face_flux_responds_to_checkerboard_pressure() -> None:
    mesh = structured_cartesian_mesh_3d(4, 1, 1)
    pressure = np.array([1.0, -1.0, 1.0, -1.0])
    velocity = np.zeros((mesh.cell_count, 3))

    flux = rhie_chow_face_flux(
        mesh,
        velocity,
        pressure,
        np.ones(mesh.cell_count),
        dt=0.1,
        no_through_boundary_tags=("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
    )

    internal_flux = flux[mesh.internal_faces]
    boundary_flux = flux[mesh.boundary_faces]
    assert np.linalg.norm(internal_flux) > 0.0
    assert np.max(np.abs(boundary_flux)) == 0.0


def test_piso_projection_reduces_unstructured_velocity_divergence() -> None:
    mesh = two_cell_skew_mesh_3d()
    velocity = np.array([[0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    projection = project_velocity_piso(
        mesh,
        velocity,
        np.ones(mesh.cell_count),
        dt=0.05,
        corrector_count=3,
        no_through_boundary_tags=("x_min", "x_max"),
    )

    assert projection.corrector_count == 3
    assert projection.rhie_chow_active is True
    assert projection.initial_divergence_norm > 0.0
    assert projection.projected_divergence_norm <= projection.initial_divergence_norm
    assert np.linalg.norm(divergence_from_face_flux(mesh, projection.corrected_flux)) == projection.projected_divergence_norm


def test_pimple_projection_reports_outer_and_non_orthogonal_corrections() -> None:
    mesh = two_cell_skew_mesh_3d()
    velocity = np.array([[0.2, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    projection = project_velocity_pimple(
        mesh,
        velocity,
        np.ones(mesh.cell_count),
        dt=0.05,
        outer_corrector_count=2,
        pressure_corrector_count=2,
        non_orthogonal_corrector_count=1,
        no_through_boundary_tags=("x_min", "x_max"),
    )

    assert projection.outer_corrector_count == 2
    assert projection.non_orthogonal_corrector_count == 1
    assert projection.corrector_count == 6
    assert projection.rhie_chow_active is True
    assert projection.projected_divergence_norm <= projection.initial_divergence_norm
