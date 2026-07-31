from __future__ import annotations

import numpy as np
import pytest

from vof_transport import advect_vof_upwind_1d, vof_mass


def test_vof_upwind_transport_preserves_liquid_volume_periodic() -> None:
    alpha = np.array([1.0, 1.0, 0.5, 0.0, 0.0])
    velocity_faces = np.ones(alpha.size + 1) * 0.25
    dx = 0.2
    dt = 0.1
    updated = advect_vof_upwind_1d(alpha, velocity_faces, dx, dt)

    assert vof_mass(updated, dx) == pytest.approx(vof_mass(alpha, dx), abs=1.0e-15)


def test_vof_upwind_transport_remains_bounded() -> None:
    alpha = np.array([1.0, 0.0, 1.0, 0.0])
    velocity_faces = np.array([0.5, -0.5, 0.5, -0.5, 0.5])
    updated = advect_vof_upwind_1d(alpha, velocity_faces, dx=0.25, dt=0.05)
    assert np.all((0.0 <= updated) & (updated <= 1.0))


def test_vof_mass_supports_nonuniform_cell_volumes() -> None:
    alpha = np.array([1.0, 0.5, 0.0])
    volumes = np.array([0.2, 0.4, 0.8])
    assert vof_mass(alpha, volumes) == pytest.approx(0.4)
