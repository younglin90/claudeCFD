from __future__ import annotations

import numpy as np
import pytest

from plume_tracking import (
    deterministic_conical_tracks_to_plane,
    plume_half_angle_from_jet_and_droplet,
    weighted_rectangular_panel_tracking,
)


def test_deterministic_conical_tracks_preserve_current_weights() -> None:
    emitters = np.array([[-0.5, 0.0], [0.5, 0.0]])
    currents = np.array([2.0, 4.0])

    positions, weights = deterministic_conical_tracks_to_plane(
        emitters,
        plane_z=2.0,
        half_angle=np.arctan(0.25),
        particles_per_emitter=4,
        current_weights=currents,
    )

    assert positions.shape == (8, 3)
    assert weights.shape == (8,)
    assert np.all(positions[:, 2] == pytest.approx(2.0))
    assert weights[:4].sum() == pytest.approx(2.0)
    assert weights[4:].sum() == pytest.approx(4.0)
    assert weights.sum() == pytest.approx(6.0)


def test_weighted_rectangular_panel_tracking_closes_deposited_and_retained_weight() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.6, 0.0, 1.0],
            [0.0, 0.6, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )
    weights = np.array([2.0, 3.0, 5.0, 7.0])

    tracking = weighted_rectangular_panel_tracking(
        positions,
        weights,
        plane_z=1.0,
        width=1.0,
        height=1.0,
    )

    assert tracking.hit_mask.tolist() == [True, False, False, False]
    assert tracking.deposited_weight == pytest.approx(2.0)
    assert tracking.retained_weight == pytest.approx(15.0)
    assert tracking.impingement_fraction == pytest.approx(2.0 / 17.0)
    assert tracking.weight_balance_error == pytest.approx(0.0)


def test_plume_half_angle_from_jet_and_droplet_validates_positive_scales() -> None:
    assert plume_half_angle_from_jet_and_droplet(2.0, 4.0) == pytest.approx(np.arctan(0.25))
    with pytest.raises(ValueError, match="jet_diameter"):
        plume_half_angle_from_jet_and_droplet(0.0, 1.0)
    with pytest.raises(ValueError, match="droplet_diameter"):
        plume_half_angle_from_jet_and_droplet(1.0, 0.0)
