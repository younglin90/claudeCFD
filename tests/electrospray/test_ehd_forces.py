from __future__ import annotations

import numpy as np
import pytest

from ehd_forces import (
    electric_body_force,
    maxwell_normal_traction,
    maxwell_stress_force_density_2d,
    maxwell_stress_2d,
    maxwell_stress_tensor,
    maxwell_traction_vector,
    normal_traction_jump_2d,
)
from electrospray1d import maxwell_normal_pressure_jump


def test_maxwell_stress_tensor_matches_analytic_components() -> None:
    ex = np.array([[2.0, 0.0]])
    ey = np.array([[0.0, 3.0]])
    txx, txy, tyy = maxwell_stress_2d(4.0, ex, ey)

    assert txx[0, 0] == pytest.approx(8.0)
    assert tyy[0, 0] == pytest.approx(-8.0)
    assert txx[0, 1] == pytest.approx(-18.0)
    assert tyy[0, 1] == pytest.approx(18.0)
    assert np.all(txy == 0.0)


def test_maxwell_stress_force_density_2d_matches_linear_permittivity_gradient() -> None:
    x_faces = np.linspace(0.0, 1.0, 6)
    y_faces = np.linspace(0.0, 0.8, 5)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    y_centers = 0.5 * (y_faces[:-1] + y_faces[1:])
    xx, _yy = np.meshgrid(x_centers, y_centers)
    epsilon = 2.0 + 3.0 * xx
    e_x = np.ones_like(epsilon) * 4.0
    e_y = np.zeros_like(epsilon)

    force_x, force_y = maxwell_stress_force_density_2d(
        epsilon,
        e_x,
        e_y,
        dx=x_faces[1] - x_faces[0],
        dy=y_faces[1] - y_faces[0],
    )

    np.testing.assert_allclose(force_x, 0.5 * 3.0 * 4.0**2, atol=1.0e-13)
    np.testing.assert_allclose(force_y, 0.0, atol=1.0e-13)


def test_coulomb_body_force_is_charge_times_electric_field() -> None:
    charge = np.array([1.0, -2.0, 0.5])
    ex = np.array([3.0, 3.0, -1.0])
    ey = np.array([0.0, 2.0, 4.0])
    fx, fy = electric_body_force(charge, ex, ey)

    np.testing.assert_allclose(fx, np.array([3.0, -6.0, -0.5]))
    np.testing.assert_allclose(fy, np.array([0.0, -4.0, 2.0]))


def test_normal_traction_jump_reduces_to_flat_interface_pressure_jump() -> None:
    eps_l = 9.0
    eps_g = 3.0
    e_l = (2.0, 0.0)
    e_g = (eps_l * e_l[0] / eps_g, 0.0)
    jump = normal_traction_jump_2d(eps_l, e_l, eps_g, e_g, normal=(1.0, 0.0))
    expected = maxwell_normal_pressure_jump(eps_l, e_l[0], eps_g, e_g[0])
    assert jump == pytest.approx(expected)


def test_maxwell_stress_tensor_is_symmetric_and_matches_3d_formula() -> None:
    tensor = maxwell_stress_tensor(2.0, np.array([1.0, 2.0, 0.0]))

    np.testing.assert_allclose(tensor, tensor.T)
    np.testing.assert_allclose(tensor, 2.0 * (np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]]) - 2.5 * np.eye(3)))


def test_maxwell_normal_traction_normalizes_normal_vector() -> None:
    traction = maxwell_normal_traction(3.0, np.array([2.0, 0.0, 0.0]), np.array([4.0, 0.0, 0.0]))

    assert traction == pytest.approx(0.5 * 3.0 * 4.0)


def test_maxwell_traction_vector_matches_tensor_times_unit_normal() -> None:
    field = np.array([1.0, 2.0, 0.0])
    normal = np.array([0.0, 3.0, 0.0])
    tensor = maxwell_stress_tensor(2.0, field)

    np.testing.assert_allclose(maxwell_traction_vector(2.0, field, normal), tensor @ np.array([0.0, 1.0, 0.0]))
