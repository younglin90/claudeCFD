from __future__ import annotations

import numpy as np
import pytest

from axisymmetric_electrostatics import solve_radial_poisson_axisymmetric


def test_radial_axisymmetric_laplace_solution_is_constant() -> None:
    r_faces = np.linspace(0.0, 1.0, 32 + 1)
    rc, phi, e_r = solve_radial_poisson_axisymmetric(r_faces, np.ones(32) * 2.0, phi_outer=3.0)
    assert rc.shape == phi.shape == e_r.shape
    assert np.max(np.abs(phi - 3.0)) < 1.0e-12
    assert np.max(np.abs(e_r)) < 1.0e-12


def test_radial_axisymmetric_uniform_charge_converges_second_order() -> None:
    eps = 4.0
    rho = 8.0
    radius = 1.0
    errors = []
    for n in (24, 48, 96):
        r_faces = np.linspace(0.0, radius, n + 1)
        rc, phi, _e_r = solve_radial_poisson_axisymmetric(
            r_faces,
            np.ones(n) * eps,
            phi_outer=0.0,
            charge_density_cells=np.ones(n) * rho,
        )
        exact = rho * (radius * radius - rc * rc) / (4.0 * eps)
        errors.append(np.max(np.abs(phi - exact)))
    assert errors[1] / errors[0] == pytest.approx(0.25, rel=0.03)
    assert errors[2] / errors[1] == pytest.approx(0.25, rel=0.03)
