from __future__ import annotations

import json

import pytest

from validation_cases_thruster import (
    microthruster_operating_point_report,
    microthruster_operating_point_report_json,
    run_all_thruster_cases,
    run_microthruster_operating_point_case,
)


def test_microthruster_operating_point_validation_case_passes() -> None:
    result = run_microthruster_operating_point_case()

    assert result.case_id == "3d_microthruster_operating_point"
    assert result.passed


def test_microthruster_operating_point_report_is_deterministic() -> None:
    report = microthruster_operating_point_report()

    assert report["propellant"] == "validation"
    assert report["mass_flow_rate"] == pytest.approx(2.4e-9)
    assert report["mass_flow_balance_error"] == pytest.approx(0.0)
    assert report["charge_to_mass"] == pytest.approx(500.0)
    assert report["charge_to_mass_balance_error"] == pytest.approx(0.0)
    assert report["exhaust_velocity"] == pytest.approx(1000.0)
    assert report["exhaust_velocity_balance_error"] == pytest.approx(0.0)
    assert report["thrust"] == pytest.approx(2.4e-6)
    assert report["thrust_momentum_balance_error"] == pytest.approx(0.0)
    assert report["specific_impulse_balance_error"] == pytest.approx(0.0)
    assert report["ideal_efficiency"] == pytest.approx(1.0)
    assert report["power_efficiency_balance_error"] == pytest.approx(0.0)
    assert report["thrust_to_power"] == pytest.approx(2.0e-3)
    assert report["thrust_to_power_balance_error"] == pytest.approx(0.0)
    assert report["closure_residual_count"] == 7
    assert report["microthruster_operating_point_status"] == "pass"


def test_microthruster_operating_point_report_json_is_stable() -> None:
    payload = microthruster_operating_point_report_json()

    assert payload.endswith("\n")
    assert json.loads(payload) == microthruster_operating_point_report()
    assert payload == json.dumps(microthruster_operating_point_report(), sort_keys=True) + "\n"


def test_all_thruster_validation_cases_pass() -> None:
    results = run_all_thruster_cases()

    assert len(results) == 1
    assert all(result.passed for result in results)
