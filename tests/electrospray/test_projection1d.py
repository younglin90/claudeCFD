from __future__ import annotations

import numpy as np

from projection1d import project_periodic_velocity, solve_periodic_pressure_correction


def test_periodic_pressure_correction_has_zero_mean() -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    div = np.sin(2.0 * np.pi * x)
    p = solve_periodic_pressure_correction(div, dx=x[1] - x[0], dt=0.1, density=2.0)
    assert abs(float(np.mean(p))) < 1.0e-14


def test_projection_reduces_periodic_velocity_divergence() -> None:
    x = np.linspace(0.0, 1.0, 128, endpoint=False)
    dx = x[1] - x[0]
    velocity = np.sin(2.0 * np.pi * x)
    corrected, _pressure = project_periodic_velocity(velocity, dx=dx, dt=0.1, density=1.0)
    div_before = np.linalg.norm((np.roll(velocity, -1) - np.roll(velocity, 1)) / (2.0 * dx))
    div_after = np.linalg.norm((np.roll(corrected, -1) - np.roll(corrected, 1)) / (2.0 * dx))
    assert div_after < 0.1 * div_before


def test_projection_preserves_mean_velocity() -> None:
    x = np.linspace(0.0, 1.0, 64, endpoint=False)
    velocity = 0.3 + 0.1 * np.sin(4.0 * np.pi * x)
    corrected, _pressure = project_periodic_velocity(velocity, dx=x[1] - x[0], dt=0.2, density=1.5)
    assert abs(float(np.mean(corrected) - np.mean(velocity))) < 1.0e-14
