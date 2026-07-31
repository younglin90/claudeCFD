from __future__ import annotations

import math

import numpy as np
import pytest

from manufactured_solutions import (
    separable_charge_2d,
    separable_potential_2d,
    sinusoidal_charge_1d,
    sinusoidal_potential_1d,
)


def test_sinusoidal_1d_manufactured_charge_matches_laplacian_relation() -> None:
    x = np.linspace(0.0, 1.0, 9)
    phi = sinusoidal_potential_1d(x, amplitude=2.0, length=1.0)
    rho = sinusoidal_charge_1d(x, epsilon=3.0, amplitude=2.0, length=1.0)
    np.testing.assert_allclose(rho, 3.0 * math.pi**2 * phi)
    assert phi[0] == pytest.approx(0.0)
    assert phi[-1] == pytest.approx(0.0, abs=1.0e-15)


def test_separable_2d_manufactured_charge_matches_laplacian_relation() -> None:
    x = np.array([[0.25, 0.5]])
    y = np.array([[0.0, 0.5]])
    eps = 4.0
    phi = separable_potential_2d(x, y, amplitude=1.5, lx=1.0, ly=2.0)
    rho = separable_charge_2d(x, y, epsilon=eps, amplitude=1.5, lx=1.0, ly=2.0)
    expected_factor = eps * (math.pi**2 + (math.pi / 2.0) ** 2)
    np.testing.assert_allclose(rho, expected_factor * phi)


def test_manufactured_2d_potential_has_dirichlet_zero_on_x_boundaries() -> None:
    y = np.linspace(0.0, 1.0, 5)
    assert np.max(np.abs(separable_potential_2d(np.zeros_like(y), y))) < 1.0e-15
    assert np.max(np.abs(separable_potential_2d(np.ones_like(y), y))) < 1.0e-15
