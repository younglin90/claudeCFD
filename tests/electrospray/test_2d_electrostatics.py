from __future__ import annotations

import numpy as np
import pytest

from electrospray1d import layered_dielectric_exact
from electrospray2d import solve_laplace_2d


def test_2d_parallel_plate_field_is_uniform_with_insulated_sides() -> None:
    nx = 24
    ny = 10
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.4, ny + 1)
    eps = np.ones((ny, nx)) * 3.0
    sol = solve_laplace_2d(x_faces, y_faces, eps, phi_left=8.0, phi_right=2.0)

    exact_x = 8.0 + (2.0 - 8.0) * sol.x_centers
    assert np.max(np.abs(sol.phi - exact_x[np.newaxis, :])) < 1.0e-12
    assert np.max(np.abs(sol.e_x - 6.0)) < 1.0e-12
    assert np.max(np.abs(sol.e_y)) < 1.0e-12


def test_2d_top_bottom_dirichlet_field_is_uniform() -> None:
    nx = 10
    ny = 24
    x_faces = np.linspace(0.0, 0.4, nx + 1)
    y_faces = np.linspace(0.0, 1.0, ny + 1)
    eps = np.ones((ny, nx)) * 3.0
    sol = solve_laplace_2d(
        x_faces,
        y_faces,
        eps,
        phi_left=None,
        phi_right=None,
        phi_bottom=2.0,
        phi_top=8.0,
    )

    exact_y = 2.0 + (8.0 - 2.0) * sol.y_centers
    assert np.max(np.abs(sol.phi - exact_y[:, np.newaxis])) < 1.0e-12
    assert np.max(np.abs(sol.e_x)) < 1.0e-12
    assert np.max(np.abs(sol.e_y + 6.0)) < 1.0e-12


def test_2d_vertical_dielectric_strip_matches_1d_layer_solution() -> None:
    nx = 40
    ny = 8
    x_faces = np.linspace(0.0, 1.0, nx + 1)
    y_faces = np.linspace(0.0, 0.2, ny + 1)
    x_centers = 0.5 * (x_faces[:-1] + x_faces[1:])
    eps_1d = np.where(x_centers < 0.5, 2.0, 7.0)
    eps = np.tile(eps_1d, (ny, 1))

    sol = solve_laplace_2d(x_faces, y_faces, eps, phi_left=5.0, phi_right=-1.0)
    exact_phi, exact_ex = layered_dielectric_exact(x_faces, eps_1d, 5.0, -1.0)

    assert np.max(np.abs(sol.phi - exact_phi[np.newaxis, :])) < 1.0e-12
    assert np.max(np.abs(sol.e_x - exact_ex[np.newaxis, :])) < 1.0e-12
    assert np.max(np.abs(sol.e_y)) < 1.0e-12

    left_d = eps[:, nx // 2 - 1] * sol.e_x[:, nx // 2 - 1]
    right_d = eps[:, nx // 2] * sol.e_x[:, nx // 2]
    assert np.max(np.abs(left_d - right_d)) < 1.0e-12


def test_2d_uniform_space_charge_poisson_converges_second_order() -> None:
    eps_value = 5.0
    rho = 10.0
    errors = []
    for nx in (16, 32, 64):
        ny = 6
        x_faces = np.linspace(0.0, 1.0, nx + 1)
        y_faces = np.linspace(0.0, 0.3, ny + 1)
        eps = np.ones((ny, nx)) * eps_value
        charge = np.ones((ny, nx)) * rho
        sol = solve_laplace_2d(x_faces, y_faces, eps, phi_left=0.0, phi_right=0.0, charge_density_cells=charge)
        exact = rho / (2.0 * eps_value) * sol.x_centers * (1.0 - sol.x_centers)
        errors.append(np.max(np.abs(sol.phi - exact[np.newaxis, :])))
        assert np.max(np.abs(sol.e_y)) < 1.0e-12
    assert errors[1] / errors[0] == pytest.approx(0.25, rel=0.03)
    assert errors[2] / errors[1] == pytest.approx(0.25, rel=0.03)
