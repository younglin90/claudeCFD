import pytest
import numpy as np

from axisymmetric_projection import (
    axisymmetric_cell_centered_divergence,
    axisymmetric_cell_centered_divergence_open_z,
    project_axisymmetric_velocity,
    project_axisymmetric_velocity_open_z,
)


def test_axisymmetric_divergence_is_zero_for_uniform_axial_flow() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    velocity_r = np.zeros((64, 32))
    velocity_z = np.ones((64, 32)) * 3.0

    divergence = axisymmetric_cell_centered_divergence(velocity_r, velocity_z, r_faces, z_faces)

    assert np.linalg.norm(divergence) < 1.0e-12


def test_axisymmetric_pressure_projection_reduces_divergence() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    rr, zz = np.meshgrid(r_centers, z_centers)
    velocity_r = 0.04 * rr * (1.0 - rr) * np.cos(2.0 * np.pi * zz / z_faces[-1])
    velocity_z = 0.02 * np.sin(2.0 * np.pi * zz / z_faces[-1]) * (1.0 - rr * rr)

    initial = axisymmetric_cell_centered_divergence(velocity_r, velocity_z, r_faces, z_faces)
    projected_r, projected_z, pressure = project_axisymmetric_velocity(
        velocity_r,
        velocity_z,
        r_faces,
        z_faces,
        dt=1.0e-3,
        density=1.0,
        iterations=1200,
    )
    final = axisymmetric_cell_centered_divergence(projected_r, projected_z, r_faces, z_faces)

    assert np.linalg.norm(pressure) > 0.0
    assert np.linalg.norm(final) / np.linalg.norm(initial) < 0.15
    assert np.max(np.abs(projected_r[:, 0])) == 0.0
    assert np.max(np.abs(projected_r[:, -1])) == 0.0


def test_axisymmetric_open_z_divergence_has_zero_gradient_axial_faces() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    velocity_r = np.zeros((64, 32))
    velocity_z = np.ones((64, 32)) * 3.0

    divergence = axisymmetric_cell_centered_divergence_open_z(velocity_r, velocity_z, r_faces, z_faces)

    assert np.linalg.norm(divergence) < 1.0e-12


def test_axisymmetric_open_z_pressure_projection_reduces_divergence() -> None:
    r_faces = np.linspace(0.0, 1.0, 33)
    z_faces = np.linspace(0.0, 2.0, 65)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    rr, zz = np.meshgrid(r_centers, z_centers)
    velocity_r = 0.03 * rr * (1.0 - rr) * np.cos(np.pi * zz / z_faces[-1])
    velocity_z = 0.02 * np.sin(np.pi * zz / z_faces[-1]) * (1.0 - rr * rr)

    initial = axisymmetric_cell_centered_divergence_open_z(velocity_r, velocity_z, r_faces, z_faces)
    projected_r, projected_z, pressure = project_axisymmetric_velocity_open_z(
        velocity_r,
        velocity_z,
        r_faces,
        z_faces,
        dt=1.0e-3,
        density=1.0,
        iterations=1200,
    )
    final = axisymmetric_cell_centered_divergence_open_z(projected_r, projected_z, r_faces, z_faces)

    assert np.linalg.norm(pressure) > 0.0
    assert np.linalg.norm(final) / np.linalg.norm(initial) < 0.25
    assert np.max(np.abs(projected_r[:, 0])) == pytest.approx(0.0)
    assert np.max(np.abs(projected_r[:, -1])) == pytest.approx(0.0)
