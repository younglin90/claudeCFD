from __future__ import annotations

import numpy as np
import pytest

from electrospray1d import (
    charge_relaxation,
    layered_dielectric_exact,
    maxwell_normal_pressure_jump,
    solve_electrostatic_1d,
    solve_flat_interface_maxwell_balance_1d,
)


def test_parallel_plate_uniform_field_second_order_exact_on_grid() -> None:
    errors = []
    for n in (16, 32, 64):
        faces = np.linspace(0.0, 1.0, n + 1)
        sol = solve_electrostatic_1d(faces, np.ones(n) * 2.5, phi_left=10.0, phi_right=0.0)
        exact_phi = 10.0 * (1.0 - sol.x_centers)
        errors.append(np.linalg.norm(sol.phi - exact_phi) / np.sqrt(n))
        assert np.max(np.abs(sol.e_cell - 10.0)) < 1.0e-12
    assert errors[-1] < 1.0e-12


def test_dielectric_jump_preserves_normal_displacement() -> None:
    n = 80
    faces = np.linspace(0.0, 1.0, n + 1)
    eps = np.where(0.5 * (faces[:-1] + faces[1:]) < 0.5, 2.0, 8.0)
    sol = solve_electrostatic_1d(faces, eps, phi_left=5.0, phi_right=0.0)
    exact_phi, exact_e = layered_dielectric_exact(faces, eps, 5.0, 0.0)

    assert np.linalg.norm(sol.phi - exact_phi, ord=np.inf) < 1.0e-12
    assert np.linalg.norm(sol.e_cell - exact_e, ord=np.inf) < 1.0e-12
    left_d = eps[n // 2 - 1] * sol.e_cell[n // 2 - 1]
    right_d = eps[n // 2] * sol.e_cell[n // 2]
    assert abs(left_d - right_d) < 1.0e-12


def test_uniform_space_charge_poisson_solution_is_quadratic() -> None:
    eps = 4.0
    rho = 12.0
    errors = []
    for n in (20, 40, 80):
        faces = np.linspace(0.0, 1.0, n + 1)
        sol = solve_electrostatic_1d(faces, np.ones(n) * eps, 0.0, 0.0, charge_density_cells=np.ones(n) * rho)
        exact_phi = rho / (2.0 * eps) * sol.x_centers * (1.0 - sol.x_centers)
        errors.append(np.linalg.norm(sol.phi - exact_phi, ord=np.inf))
    assert errors[1] / errors[0] == pytest.approx(0.25, rel=0.02)
    assert errors[2] / errors[1] == pytest.approx(0.25, rel=0.02)


def test_charge_relaxation_matches_epsilon_over_sigma_timescale() -> None:
    epsilon = 4.0e-10
    sigma = 2.0e-8
    tau = epsilon / sigma
    times = np.linspace(0.0, 5.0 * tau, 12)
    rho = charge_relaxation(3.2, times, epsilon=epsilon, sigma=sigma)
    expected = 3.2 * np.exp(-times / tau)
    assert np.max(np.abs(rho - expected)) / 3.2 < 1.0e-14


def test_flat_interface_maxwell_pressure_jump_balances_traction() -> None:
    eps_l = 12.0
    eps_g = 3.0
    e_l = 2.0
    e_g = eps_l * e_l / eps_g
    jump = maxwell_normal_pressure_jump(eps_l, e_l, eps_g, e_g)
    residual = jump - 0.5 * (eps_l * e_l**2 - eps_g * e_g**2)
    assert abs(residual) < 1.0e-14


def test_flat_interface_maxwell_balance_uses_solved_dielectric_fields() -> None:
    n = 80
    faces = np.linspace(0.0, 1.0, n + 1)
    eps = np.where(0.5 * (faces[:-1] + faces[1:]) < 0.5, 12.0, 3.0)

    balance = solve_flat_interface_maxwell_balance_1d(
        faces,
        eps,
        phi_left=8.0,
        phi_right=0.0,
        interface_face=n // 2,
    )

    assert balance.displacement_left == pytest.approx(balance.displacement_right, abs=1.0e-12)
    assert balance.pressure_jump == pytest.approx(balance.maxwell_jump)
    assert abs(balance.residual) < 1.0e-14
