from __future__ import annotations

import numpy as np
import pytest

from capillary import axisymmetric_laplace_pressure_jump, continuum_surface_force, laplace_pressure_jump


def test_laplace_pressure_jump_for_cylinder_and_sphere_curvature() -> None:
    gamma = 0.072
    radius = 1.2e-3
    assert laplace_pressure_jump(gamma, 1.0 / radius) == pytest.approx(gamma / radius)
    assert laplace_pressure_jump(gamma, 2.0 / radius) == pytest.approx(2.0 * gamma / radius)


def test_axisymmetric_laplace_pressure_jump_sums_meridional_and_radial_curvature() -> None:
    gamma = 0.05
    radius = 2.0e-3
    assert axisymmetric_laplace_pressure_jump(gamma, 1.0 / radius, 1.0 / radius) == pytest.approx(2.0 * gamma / radius)


def test_continuum_surface_force_is_gamma_kappa_grad_alpha() -> None:
    curvature = np.array([2.0, -1.0])
    grad_x = np.array([0.5, 1.0])
    grad_y = np.array([0.0, -0.5])
    fx, fy = continuum_surface_force(0.1, curvature, grad_x, grad_y)
    np.testing.assert_allclose(fx, np.array([0.1, -0.1]))
    np.testing.assert_allclose(fy, np.array([0.0, 0.05]))


def test_zero_surface_tension_removes_capillary_force() -> None:
    fx, fy = continuum_surface_force(0.0, np.ones(3), np.ones(3), np.ones(3))
    assert np.all(fx == 0.0)
    assert np.all(fy == 0.0)
