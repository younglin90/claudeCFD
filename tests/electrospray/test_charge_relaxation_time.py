from __future__ import annotations

import numpy as np
import pytest

from charge_relaxation_time import (
    backward_euler_relaxation_step,
    backward_euler_relaxation_step_material,
    backward_euler_relaxation_step_phase_pair,
    exact_relaxation_step,
    exact_relaxation_step_material,
    exact_relaxation_step_phase_pair,
    relaxation_decay_rate,
)
from material_properties import LeakyDielectricMaterial, LeakyDielectricPhasePair


def test_exact_relaxation_step_matches_exponential_solution() -> None:
    charge = np.array([1.0, 2.0])
    out = exact_relaxation_step(charge, dt=0.5, epsilon=2.0, conductivity=4.0)
    np.testing.assert_allclose(out, charge * np.exp(-1.0))


def test_backward_euler_relaxation_is_positive_and_more_diffusive_than_exact() -> None:
    charge = np.array([1.0, 3.0])
    exact = exact_relaxation_step(charge, dt=1.0, epsilon=1.0, conductivity=2.0)
    implicit = backward_euler_relaxation_step(charge, dt=1.0, epsilon=1.0, conductivity=2.0)
    assert np.all(implicit > 0.0)
    assert np.all(implicit > exact)


def test_material_relaxation_steps_match_scalar_property_steps() -> None:
    charge = np.array([0.5, 2.0])
    material = LeakyDielectricMaterial(permittivity=2.0, conductivity=4.0)

    np.testing.assert_allclose(
        exact_relaxation_step_material(charge, dt=0.5, material=material),
        exact_relaxation_step(charge, dt=0.5, epsilon=2.0, conductivity=4.0),
    )
    np.testing.assert_allclose(
        backward_euler_relaxation_step_material(charge, dt=0.5, material=material),
        backward_euler_relaxation_step(charge, dt=0.5, epsilon=2.0, conductivity=4.0),
    )


def test_phase_pair_exact_relaxation_step_uses_mixed_timescale_field() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=3.0, conductivity=3.0),
    )
    charge = np.array([1.0, 2.0, 3.0])
    alpha = np.array([0.0, 0.5, 1.0])

    out = exact_relaxation_step_phase_pair(charge, dt=1.0, phase_pair=pair, alpha_liquid=alpha)

    np.testing.assert_allclose(out, charge * np.exp(-1.0 / pair.relaxation_time_field(alpha)))
    with pytest.raises(ValueError, match="same shape"):
        exact_relaxation_step_phase_pair(charge, dt=1.0, phase_pair=pair, alpha_liquid=np.array([0.0]))


def test_phase_pair_backward_euler_relaxation_step_uses_mixed_timescale_field() -> None:
    pair = LeakyDielectricPhasePair(
        liquid=LeakyDielectricMaterial(permittivity=8.0, conductivity=4.0),
        gas=LeakyDielectricMaterial(permittivity=3.0, conductivity=3.0),
    )
    charge = np.array([1.0, 2.0, 3.0])
    alpha = np.array([0.0, 0.5, 1.0])

    out = backward_euler_relaxation_step_phase_pair(charge, dt=1.0, phase_pair=pair, alpha_liquid=alpha)

    np.testing.assert_allclose(out, charge / (1.0 + 1.0 / pair.relaxation_time_field(alpha)))
    assert np.all(out > exact_relaxation_step_phase_pair(charge, dt=1.0, phase_pair=pair, alpha_liquid=alpha))
    with pytest.raises(ValueError, match="same shape"):
        backward_euler_relaxation_step_phase_pair(charge, dt=1.0, phase_pair=pair, alpha_liquid=np.array([0.0]))


def test_relaxation_decay_rate_recovers_sigma_over_epsilon() -> None:
    old = 5.0
    rate = 3.0
    dt = 0.2
    new = old * np.exp(-rate * dt)
    assert relaxation_decay_rate(old, new, dt) == pytest.approx(rate)


def test_backward_euler_relaxation_recovers_decay_rate_for_small_steps() -> None:
    epsilon = 4.0e-10
    sigma = 2.0e-8
    tau = epsilon / sigma
    dt = 0.01 * tau
    steps = 500
    charge = np.array([3.2])

    for _ in range(steps):
        charge = backward_euler_relaxation_step(charge, dt, epsilon, sigma)

    measured = relaxation_decay_rate(3.2, float(charge[0]), steps * dt)
    assert abs(measured / (sigma / epsilon) - 1.0) < 1.0e-2
