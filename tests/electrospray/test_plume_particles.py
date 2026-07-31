from __future__ import annotations

import math

import numpy as np
import pytest

from plume_particles import ballistic_positions, plane_impingement_fraction, plume_half_angle


def test_ballistic_positions_apply_constant_acceleration() -> None:
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    v = np.array([[1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    a = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 0.0]])
    out = ballistic_positions(x0, v, time=3.0, acceleration=a)
    np.testing.assert_allclose(out, x0 + 3.0 * v + 4.5 * a)


def test_plume_half_angle_uses_largest_radial_to_axial_ratio() -> None:
    velocities = np.array([[0.0, 0.0, 10.0], [1.0, 0.0, 10.0], [0.0, 2.0, 10.0]])
    assert plume_half_angle(velocities) == pytest.approx(math.atan2(2.0, 10.0))


def test_plane_impingement_fraction_counts_particles_inside_target_radius() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.5, 0.0, 2.0],
            [1.5, 0.0, 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert plane_impingement_fraction(positions, plane_z=2.0, radius=1.0) == pytest.approx(0.5)
