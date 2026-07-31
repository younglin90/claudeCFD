from __future__ import annotations

import numpy as np
import pytest

from emitter_array import (
    current_uniformity,
    pairwise_shielded_currents,
    pairwise_shielding_factors,
    pitch_limited_total_current,
    pitch_shielding_factor,
    shielding_ratio,
    square_array_positions,
    total_current,
)


def test_total_current_sums_emitters() -> None:
    assert total_current([1.0e-6, 2.0e-6, 3.0e-6]) == pytest.approx(6.0e-6)


def test_current_uniformity_is_coefficient_of_variation() -> None:
    currents = np.array([1.0, 2.0, 3.0])
    assert current_uniformity(currents) == pytest.approx(float(np.std(currents) / np.mean(currents)))
    assert current_uniformity([2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_shielding_ratio_compares_array_to_single_onset_voltage() -> None:
    assert shielding_ratio(1320.0, 1200.0) == pytest.approx(1.1)


def test_square_array_positions_are_centered() -> None:
    positions = square_array_positions(2, pitch=0.5)
    assert positions.shape == (4, 2)
    np.testing.assert_allclose(np.mean(positions, axis=0), np.array([0.0, 0.0]))
    assert set(map(tuple, np.round(positions, 8))) == {(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)}


def test_pitch_shielding_factor_decreases_with_pitch() -> None:
    small_pitch_factor = pitch_shielding_factor(pitch=0.5, reference_pitch=1.0)
    large_pitch_factor = pitch_shielding_factor(pitch=2.0, reference_pitch=1.0)

    assert small_pitch_factor > large_pitch_factor > 1.0
    assert pitch_limited_total_current(1.0e-6, 4, small_pitch_factor) < pitch_limited_total_current(1.0e-6, 4, large_pitch_factor)


def test_pairwise_shielded_currents_are_uniform_for_symmetric_square_array() -> None:
    positions = square_array_positions(2, pitch=1.0)
    factors = pairwise_shielding_factors(positions, reference_pitch=1.0, strength=0.05)
    currents = pairwise_shielded_currents(positions, single_emitter_current=1.0e-6, reference_pitch=1.0, strength=0.05)

    assert np.std(factors) == pytest.approx(0.0)
    assert current_uniformity(currents) == pytest.approx(0.0)
    assert total_current(currents) < 4.0e-6
    assert total_current(currents) / 4.0e-6 == pytest.approx(0.8807815188754272)
