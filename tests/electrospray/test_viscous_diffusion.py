from __future__ import annotations

import numpy as np
import pytest

from viscous_diffusion import (
    explicit_variable_viscous_step_2d,
    explicit_viscous_step_1d,
    explicit_viscous_step_2d,
    kinetic_energy_density,
    periodic_laplacian_1d,
    periodic_laplacian_2d,
)


def test_periodic_laplacian_matches_sine_eigenvalue() -> None:
    n = 128
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = x[1] - x[0]
    mode = np.sin(2.0 * np.pi * x)
    lap = periodic_laplacian_1d(mode, dx)
    exact = -(2.0 * np.pi) ** 2 * mode
    assert np.linalg.norm(lap - exact) / np.linalg.norm(exact) < 3.0e-4


def test_explicit_viscous_step_damps_kinetic_energy() -> None:
    n = 64
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = x[1] - x[0]
    u = np.sin(2.0 * np.pi * x)
    updated = explicit_viscous_step_1d(u, kinematic_viscosity=0.01, dx=dx, dt=0.001)
    assert np.sum(kinetic_energy_density(1.0, updated)) < np.sum(kinetic_energy_density(1.0, u))


def test_periodic_laplacian_2d_matches_sine_eigenvalue() -> None:
    n = 64
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xx, yy = np.meshgrid(x, y)
    mode = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    lap = periodic_laplacian_2d(mode, dx, dy)
    exact = -2.0 * (2.0 * np.pi) ** 2 * mode

    assert np.linalg.norm(lap - exact) / np.linalg.norm(exact) < 9.0e-4


def test_explicit_viscous_step_2d_damps_kinetic_energy() -> None:
    n = 32
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    velocity = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    updated = explicit_viscous_step_2d(velocity, kinematic_viscosity=0.01, dx=x[1] - x[0], dy=y[1] - y[0], dt=0.001)

    assert np.sum(updated * updated) < np.sum(velocity * velocity)


def test_explicit_variable_viscous_step_2d_damps_kinetic_energy() -> None:
    n = 32
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    xx, yy = np.meshgrid(x, y)
    velocity = np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    kinematic_viscosity = 0.005 + 0.003 * (1.0 + np.sin(2.0 * np.pi * yy))

    updated = explicit_variable_viscous_step_2d(
        velocity,
        kinematic_viscosity,
        dx=x[1] - x[0],
        dy=y[1] - y[0],
        dt=0.001,
    )

    assert np.sum(updated * updated) < np.sum(velocity * velocity)


def test_kinetic_energy_density_is_half_rho_u_squared() -> None:
    np.testing.assert_allclose(kinetic_energy_density(2.0, np.array([1.0, 3.0])), np.array([1.0, 9.0]))
