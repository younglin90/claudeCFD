from __future__ import annotations

import numpy as np
import pytest

from state import ElectrosprayState1D, free_charge_loss_fraction, total_free_charge, validate_state_bounds


def test_state_container_normalizes_arrays_and_reports_size() -> None:
    state = ElectrosprayState1D([1.0, 0.0], [2.0, 3.0], [0.1, 0.2], [1.0, 1.0])
    assert state.size == 2
    assert isinstance(state.alpha_liquid, np.ndarray)


def test_state_bounds_accept_admissible_state() -> None:
    state = ElectrosprayState1D(np.array([1.0, 0.5]), np.array([0.0, 1.0]), np.zeros(2), np.ones(2))
    validate_state_bounds(state)


def test_state_bounds_reject_unbounded_volume_fraction() -> None:
    state = ElectrosprayState1D(np.array([1.2]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="alpha_liquid"):
        validate_state_bounds(state)


def test_total_free_charge_integrates_over_cell_volume() -> None:
    state = ElectrosprayState1D(np.ones(3), np.array([1.0, 2.0, 3.0]), np.zeros(3), np.ones(3))
    assert total_free_charge(state, np.array([0.1, 0.2, 0.3])) == pytest.approx(1.4)


def test_free_charge_loss_fraction_reports_normalized_decrease() -> None:
    old = ElectrosprayState1D(np.ones(2), np.array([2.0, 2.0]), np.zeros(2), np.ones(2))
    new = ElectrosprayState1D(np.ones(2), np.array([1.0, 1.0]), np.zeros(2), np.ones(2))

    assert free_charge_loss_fraction(old, new, 0.5) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="nonzero"):
        free_charge_loss_fraction(ElectrosprayState1D(np.ones(1), np.zeros(1), np.zeros(1), np.ones(1)), new, 1.0)
