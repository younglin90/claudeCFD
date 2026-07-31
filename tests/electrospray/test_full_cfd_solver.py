from __future__ import annotations

from full_cfd_solver import FULL_CFD_SCHEME_ID, advance_full_two_phase_navier_stokes_electrospray
from validation_cases_full_cfd import (
    full_cfd_timestep_contract_report,
    full_cfd_timestep_contract_scenario,
    run_full_two_phase_navier_stokes_timestep_contract_case,
)


def test_full_two_phase_timestep_uses_one_state_contract_and_all_operators() -> None:
    state, x_faces, y_faces, level_set, config = full_cfd_timestep_contract_scenario()

    next_state, diagnostics = advance_full_two_phase_navier_stokes_electrospray(
        state,
        x_faces,
        y_faces,
        dt=1.0e-4,
        config=config,
        level_set_phi=level_set,
    )

    assert next_state.shape == state.shape
    assert diagnostics.scheme_id == FULL_CFD_SCHEME_ID
    assert diagnostics.one_state_timestep_contract is True
    assert diagnostics.all_required_operators_active is True
    assert diagnostics.density_ratio > 100.0
    assert diagnostics.coupled.density_min == 2.0
    assert diagnostics.coupled.density_max == 1000.0
    assert diagnostics.coupled.kinematic_viscosity_effective > 0.0


def test_full_two_phase_timestep_preserves_bounds_charge_and_projection_gate() -> None:
    report = full_cfd_timestep_contract_report()

    assert report["full_cfd_timestep_contract_status"] == "pass"
    assert report["vof_area_relative_error"] < 1.0e-5
    assert report["gas_charge_leakage_fraction"] <= 1.0e-12
    assert report["alpha_bounds_violation"] <= 1.0e-12
    assert report["divergence_reduction_ratio"] < 0.20
    assert report["pass_metric"] < 0.20


def test_full_two_phase_timestep_contract_validation_case_passes() -> None:
    result = run_full_two_phase_navier_stokes_timestep_contract_case()

    assert result.case_id == "2d_full_ns_electrospray_timestep_contract"
    assert result.passed is True
    assert result.metric is not None and result.metric < result.tolerance

