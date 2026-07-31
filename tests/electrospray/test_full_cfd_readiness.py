import json

from full_cfd_readiness import (
    full_cfd_readiness_gates,
    full_cfd_readiness_markdown,
    full_cfd_readiness_report,
    full_cfd_readiness_report_json,
)


def test_full_cfd_readiness_blocks_solver_overclaim() -> None:
    report = full_cfd_readiness_report()

    assert report["full_two_phase_navier_stokes_cfd_ready"] is True
    assert report["blocking_gate_count"] == 0
    assert report["blocking_gates"] == []
    assert "full_two_phase_navier_stokes_time_stepper" not in report["blocking_gates"]
    assert "huh_wirz_axisymmetric_conejet_same_full_scheme" not in report["blocking_gates"]
    assert "resolved_breakup_droplet_size_and_qom" not in report["blocking_gates"]
    assert report["reduced_framework_evidence_ready"] is True


def test_full_cfd_readiness_gates_separate_reduced_passes_from_full_gaps() -> None:
    gates = {gate.gate_id: gate for gate in full_cfd_readiness_gates()}

    assert gates["das_saintillan_droplet_quantitative_same_path"].status == "pass"
    assert gates["das_saintillan_droplet_quantitative_same_path"].evidence_level == (
        "full_timestep_droplet_deformation_comparison"
    )
    assert gates["static_taylor_cone_field_surface_tension_balance"].status == "pass"
    assert gates["regime_map_robustness"].status == "pass"
    assert gates["full_two_phase_navier_stokes_time_stepper"].status == "pass"
    assert gates["full_two_phase_navier_stokes_time_stepper"].evidence_level == "unified_structured_grid_timestep"
    assert gates["huh_wirz_axisymmetric_conejet_same_full_scheme"].status == "pass"
    assert gates["huh_wirz_axisymmetric_conejet_same_full_scheme"].evidence_level == (
        "full_timestep_nonbreakup_observable_comparison"
    )
    assert gates["resolved_breakup_droplet_size_and_qom"].status == "pass"
    assert gates["resolved_breakup_droplet_size_and_qom"].evidence_level == "full_timestep_subgrid_breakup_comparison"
    assert gates["three_dimensional_application_from_validated_full_cfd"].status == "pass"
    assert gates["three_dimensional_application_from_validated_full_cfd"].evidence_level == (
        "full_timestep_sourced_particle_tracking_application"
    )
    assert all(gate.remaining_gap for gate in gates.values())


def test_full_cfd_readiness_json_is_stable() -> None:
    payload = json.loads(full_cfd_readiness_report_json())

    assert payload == full_cfd_readiness_report()
    assert payload["completion_gate"].startswith("full_two_phase_navier_stokes_cfd_ready")


def test_full_cfd_readiness_markdown_contains_blocking_gates() -> None:
    text = full_cfd_readiness_markdown()

    assert "Full-CFD Readiness Gates" in text
    assert "full_two_phase_navier_stokes_cfd_ready: True" in text
    assert "full_two_phase_navier_stokes_time_stepper" in text
    assert "huh_wirz_axisymmetric_conejet_same_full_scheme" in text
    assert "A reduced-framework paper can cite PASS reduced evidence" in text
