from __future__ import annotations

import math

import pytest

from droplet_stability import is_rayleigh_unstable, rayleigh_fissility, rayleigh_limit_charge


def test_rayleigh_limit_charge_matches_closed_form() -> None:
    eps = 8.8541878128e-12
    gamma = 0.05
    radius = 10.0e-6
    expected = math.sqrt(64.0 * math.pi**2 * eps * gamma * radius**3)
    assert rayleigh_limit_charge(eps, gamma, radius) == pytest.approx(expected)


def test_rayleigh_fissility_is_quadratic_in_charge_ratio() -> None:
    eps = 2.0
    gamma = 3.0
    radius = 0.5
    q_limit = rayleigh_limit_charge(eps, gamma, radius)
    assert rayleigh_fissility(0.5 * q_limit, eps, gamma, radius) == pytest.approx(0.25)


def test_rayleigh_instability_threshold_uses_absolute_charge() -> None:
    eps = 1.0
    gamma = 1.0
    radius = 1.0
    q_limit = rayleigh_limit_charge(eps, gamma, radius)
    assert not is_rayleigh_unstable(0.999 * q_limit, eps, gamma, radius)
    assert is_rayleigh_unstable(-q_limit, eps, gamma, radius)
