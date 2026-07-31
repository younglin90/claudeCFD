from __future__ import annotations

import pytest

from validation_cases_plume import (
    plume_impingement_report,
    plume_impingement_report_json,
    run_all_plume_cases,
    run_plume_half_angle_case,
    run_plume_loss_case,
    run_plume_surface_loading_case,
    run_rectangular_panel_impingement_case,
)


def test_plume_half_angle_validation_case_passes() -> None:
    assert run_plume_half_angle_case().passed


def test_rectangular_panel_impingement_validation_case_passes() -> None:
    assert run_rectangular_panel_impingement_case().passed


def test_plume_loss_validation_case_passes() -> None:
    assert run_plume_loss_case().passed


def test_plume_surface_loading_validation_case_passes() -> None:
    assert run_plume_surface_loading_case().passed


def test_plume_impingement_report_is_deterministic() -> None:
    report = plume_impingement_report()

    assert set(report) == {
        "closure_residual_count",
        "current_fraction_balance_error",
        "deposited_current",
        "deposited_current_fraction",
        "impingement_fraction",
        "panel_current_density",
        "panel_current_density_balance_error",
        "plume_current_accounting_status",
        "plume_surface_loading_status",
        "plume_thrust_accounting_status",
        "retained_current",
        "retained_current_fraction",
        "retained_thrust_fraction",
        "thrust_fraction_balance_error",
        "thrust_loss_fraction",
    }
    assert report["deposited_current"] == pytest.approx(3.0e-6)
    assert report["deposited_current_fraction"] == pytest.approx(0.25)
    assert report["impingement_fraction"] == pytest.approx(0.25)
    assert report["panel_current_density"] == pytest.approx(1.5e-4)
    assert report["panel_current_density_balance_error"] == pytest.approx(0.0)
    assert report["retained_current"] == pytest.approx(9.0e-6)
    assert report["retained_current_fraction"] == pytest.approx(0.75)
    assert report["current_fraction_balance_error"] == pytest.approx(0.0)
    assert report["plume_current_accounting_status"] == "pass"
    assert report["retained_thrust_fraction"] == pytest.approx(0.8)
    assert report["thrust_loss_fraction"] == pytest.approx(0.2)
    assert report["thrust_fraction_balance_error"] == pytest.approx(0.0)
    assert report["plume_thrust_accounting_status"] == "pass"
    assert report["closure_residual_count"] == 3
    assert report["plume_surface_loading_status"] == "pass"


def test_plume_impingement_report_json_is_stable() -> None:
    payload = plume_impingement_report_json()

    assert payload.endswith("\n")
    assert '"panel_current_density"' in payload
    assert payload == __import__("json").dumps(plume_impingement_report(), sort_keys=True) + "\n"


def test_all_plume_validation_cases_pass() -> None:
    results = run_all_plume_cases()

    assert len(results) == 4
    assert all(result.passed for result in results)
