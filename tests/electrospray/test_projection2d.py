from __future__ import annotations

import numpy as np

from projection2d import (
    finite_difference_divergence_2d,
    finite_difference_divergence_2d_no_through,
    project_no_through_wall_velocity_2d,
    project_periodic_velocity_2d,
    project_variable_density_periodic_velocity_2d,
    spectral_divergence_2d,
)


def test_project_periodic_velocity_2d_reduces_spectral_divergence() -> None:
    nx = 32
    ny = 32
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    velocity_x = np.sin(2.0 * np.pi * xx)
    velocity_y = np.cos(2.0 * np.pi * yy)
    before = np.linalg.norm(spectral_divergence_2d(velocity_x, velocity_y, dx=x[1] - x[0], dy=y[1] - y[0]))

    projected_x, projected_y, pressure = project_periodic_velocity_2d(
        velocity_x,
        velocity_y,
        dx=x[1] - x[0],
        dy=y[1] - y[0],
        dt=1.0e-3,
        density=2.0,
    )
    after = np.linalg.norm(spectral_divergence_2d(projected_x, projected_y, dx=x[1] - x[0], dy=y[1] - y[0]))

    assert before > 0.0
    assert after < 1.0e-10 * before
    assert abs(float(np.mean(pressure))) < 1.0e-14


def test_project_periodic_velocity_2d_preserves_mean_velocity() -> None:
    velocity_x = np.ones((8, 8)) * 0.2
    velocity_y = np.ones((8, 8)) * -0.1

    projected_x, projected_y, _pressure = project_periodic_velocity_2d(
        velocity_x,
        velocity_y,
        dx=0.1,
        dy=0.1,
        dt=1.0e-3,
        density=1.0,
    )

    np.testing.assert_allclose(projected_x, velocity_x)
    np.testing.assert_allclose(projected_y, velocity_y)


def test_project_variable_density_periodic_velocity_2d_reduces_divergence() -> None:
    nx = 24
    ny = 24
    x = np.linspace(0.0, 1.0, nx, endpoint=False)
    y = np.linspace(0.0, 1.0, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    velocity_x = np.sin(2.0 * np.pi * xx)
    velocity_y = np.cos(2.0 * np.pi * yy)
    density = 1.0 + 0.5 * (1.0 + np.sin(2.0 * np.pi * yy))
    before = np.linalg.norm(finite_difference_divergence_2d(velocity_x, velocity_y, dx=x[1] - x[0], dy=y[1] - y[0]))

    projected_x, projected_y, pressure = project_variable_density_periodic_velocity_2d(
        velocity_x,
        velocity_y,
        density=density,
        dx=x[1] - x[0],
        dy=y[1] - y[0],
        dt=1.0e-3,
        iterations=600,
    )
    after = np.linalg.norm(finite_difference_divergence_2d(projected_x, projected_y, dx=x[1] - x[0], dy=y[1] - y[0]))

    assert before > 0.0
    assert after < 0.25 * before
    assert abs(float(np.mean(pressure))) < 1.0e-12


def test_project_no_through_wall_velocity_2d_reduces_divergence_and_wall_flux() -> None:
    nx = 24
    ny = 20
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y)
    velocity_x = np.sin(np.pi * xx) + 0.2 * np.cos(2.0 * np.pi * yy)
    velocity_y = 0.8 * np.sin(np.pi * yy) + 0.1 * np.cos(2.0 * np.pi * xx)
    before = np.linalg.norm(finite_difference_divergence_2d_no_through(velocity_x, velocity_y, dx=1.0 / nx, dy=1.0 / ny))

    projected_x, projected_y, pressure = project_no_through_wall_velocity_2d(
        velocity_x,
        velocity_y,
        dx=1.0 / nx,
        dy=1.0 / ny,
        dt=1.0e-3,
        density=2.0,
        iterations=1200,
    )
    after = np.linalg.norm(
        finite_difference_divergence_2d_no_through(projected_x, projected_y, dx=1.0 / nx, dy=1.0 / ny)
    )

    assert before > 0.0
    assert after < 0.6 * before
    assert np.max(np.abs(projected_x[:, [0, -1]])) < 1.0e-14
    assert np.max(np.abs(projected_y[[0, -1], :])) < 1.0e-14
    assert abs(float(np.mean(pressure))) < 1.0e-12
