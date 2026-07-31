from __future__ import annotations

import pytest

from regime_map import classify_cone_jet_regime, is_stable_cone_jet


@pytest.mark.parametrize(
    ("ca_e", "re_e", "oh", "qstar", "expected"),
    [
        (0.05, 0.01, 0.1, 1.0, "dripping"),
        (1.0, 2.0, 0.1, 1.0, "charge_relaxation_limited"),
        (1.0, 0.01, 2.0, 1.0, "viscous_pulsating"),
        (1.0, 0.01, 0.1, 10.0, "high_flow_pulsating"),
        (1.0, 0.01, 0.1, 1.0, "stable_cone_jet"),
    ],
)
def test_regime_classifier_returns_expected_reduced_label(ca_e: float, re_e: float, oh: float, qstar: float, expected: str) -> None:
    assert classify_cone_jet_regime(ca_e, re_e, oh, qstar) == expected


def test_stable_cone_jet_predicate_is_label_exact() -> None:
    assert is_stable_cone_jet("stable_cone_jet")
    assert not is_stable_cone_jet("dripping")
