from __future__ import annotations

import math

import numpy as np
import pytest

from error_metrics import convergence_rate, l2_error, linf_error, passes_threshold


def test_l2_error_uses_weighted_rms_norm() -> None:
    numerical = np.array([1.0, 3.0])
    exact = np.array([0.0, 1.0])
    weights = np.array([1.0, 3.0])
    expected = math.sqrt((1.0 * 1.0**2 + 3.0 * 2.0**2) / 4.0)
    assert l2_error(numerical, exact, weights) == pytest.approx(expected)


def test_linf_error_is_max_abs_difference() -> None:
    assert linf_error(np.array([1.0, -2.0, 3.0]), np.array([0.0, 1.0, 1.0])) == pytest.approx(3.0)


def test_convergence_rate_recovers_second_order() -> None:
    assert convergence_rate(4.0e-3, 1.0e-3, refinement=2.0) == pytest.approx(2.0)


def test_passes_threshold_handles_inclusive_and_strict_modes() -> None:
    assert passes_threshold(1.0, 1.0)
    assert not passes_threshold(1.0, 1.0, inclusive=False)
