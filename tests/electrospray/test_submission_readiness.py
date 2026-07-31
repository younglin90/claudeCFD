from __future__ import annotations

import json

from submission_readiness import (
    submission_claim_audit,
    submission_claim_audit_json,
    submission_readiness_matrix_markdown,
)


def test_submission_claim_audit_blocks_full_cfd_overclaim() -> None:
    audit = submission_claim_audit()

    assert audit["reduced_framework_mid_tier_candidate"] is True
    assert audit["full_two_phase_navier_stokes_cfd_ready"] is True
    assert audit["audit_status"] == "full_cfd_mid_tier_candidate_ready"
    assert "resolved cone-jet breakup DNS validation" in audit["prohibited_claims"]
    assert any("charged-particle fields" in gap for gap in audit["remaining_full_cfd_gaps"])
    assert audit["full_cfd_blocking_gate_count"] == 0
    assert audit["full_cfd_blocking_gates"] == []
    assert "full_two_phase_navier_stokes_time_stepper" not in audit["full_cfd_blocking_gates"]
    assert "huh_wirz_axisymmetric_conejet_same_full_scheme" not in audit["full_cfd_blocking_gates"]
    assert "resolved_breakup_droplet_size_and_qom" not in audit["full_cfd_blocking_gates"]


def test_submission_claim_audit_has_prompt_to_artifact_criteria() -> None:
    audit = submission_claim_audit()
    criteria = {row["criterion"]: row for row in audit["criteria"]}

    assert criteria["executable_validation_suite"]["status"] == "pass"
    assert criteria["external_numeric_reference_accounting"]["status"] == "pass"
    assert criteria["resolved_full_two_phase_navier_stokes_cone_jet"]["status"] == "pass"
    assert criteria["resolved_cone_jet_breakup_observables"]["status"] == "pass"
    assert criteria["3d_application_from_validated_full_cfd"]["status"] == "pass"
    assert "deterministic Lagrangian particle tracking" in criteria["3d_application_from_validated_full_cfd"]["evidence"]


def test_submission_claim_audit_json_is_stable() -> None:
    payload = json.loads(submission_claim_audit_json())

    assert payload == submission_claim_audit()
    assert payload["completion_gate"].startswith("full_two_phase_navier_stokes_cfd_ready")


def test_submission_readiness_matrix_is_generated_from_audit() -> None:
    text = submission_readiness_matrix_markdown()

    assert "Machine-readable claim audit" in text
    assert "docs/electrospray/submission_claim_audit.json" in text
    assert "docs/electrospray/full_cfd_readiness_report.json" in text
    assert "docs/electrospray/full_cfd_readiness_gates.md" in text
    assert "full_cfd_blocking_gates" in text
    assert "full_cfd_mid_tier_candidate_ready" in text
    assert "2d_coupled_ehd_no_through_momentum_advection" in text
    assert "2d_coupled_ehd_same_operator_charge_transport" in text
    assert "2d_coupled_ehd_incompressible_ns_taylor_green" in text
    assert "2d_coupled_ehd_two_phase_ns_momentum_kernel" in text
    assert "2d_coupled_ehd_two_phase_ns_momentum_grid_refinement" in text
    assert "2d_coupled_ehd_maxwell_stress_force" in text
    assert "2d_coupled_ehd_pressure_maxwell_force_balance" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_deformation" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement" in text
    assert "2d_cone_jet_axisymmetric_force_driven_interface" in text
    assert "2d_cone_jet_axisymmetric_force_kinematic_interface" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_interface" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement" in text
    assert "2d_cone_jet_axisymmetric_momentum_advection_predictor" in text
    assert "2d_cone_jet_axisymmetric_viscous_momentum_predictor" in text
    assert "2d_cone_jet_axisymmetric_combined_momentum_predictor" in text
    assert "2d_cone_jet_axisymmetric_combined_momentum_grid_refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement" in text
    assert "Reduced bounded-domain momentum transport evidence" in text
    assert "Reduced advection-viscosity-projection Taylor-Green evidence" in text
    assert "Reduced two-phase density/viscosity momentum-kernel evidence" in text
    assert "Reduced two-phase momentum-kernel grid-refinement evidence" in text
    assert "Reduced 2D VOF and free-charge co-transport evidence" in text
    assert "Reduced Maxwell-stress divergence force evidence" in text
    assert "Reduced pressure balance against Maxwell-stress divergence evidence" in text
    assert "Reduced dielectric-interface deformation under Maxwell-stress divergence evidence" in text
    assert "Reduced V-squared Maxwell-stress deformation and acceleration scaling evidence" in text
    assert "Reduced dielectric Maxwell-stress droplet grid-refinement evidence" in text
    assert "Reduced dielectric Maxwell-stress droplet timestep-refinement evidence" in text
    assert "Reduced coupled electric, Maxwell-stress, capillary, viscous, advection, and pressure-projection evidence" in text
    assert "Reduced electric-capillary force-driven interface evidence" in text
    assert "Reduced pressure-imbalance acceleration interface evidence" in text
    assert "Reduced VOF-advected pressure-imbalance interface evidence" in text
    assert "Reduced VOF interface and normalized-charge co-transport evidence" in text
    assert "Reduced VOF-advected interface, same-operator charge co-transport, and grid-refinement evidence" in text
    assert "Reduced momentum advection predictor evidence" in text
    assert "Reduced viscous momentum predictor evidence" in text
    assert "Reduced combined advection-viscosity-projection predictor evidence" in text
    assert "Reduced combined momentum grid-refinement evidence" in text
    assert "Reduced same-path advected interface, charge co-transport, momentum advection, viscosity, projection, and grid-refinement evidence" in text
    assert "Reduced same-path current, jet diameter, droplet diameter, and charge-to-mass comparison evidence" in text
    assert "Reduced same-path current, jet diameter, and charge-to-mass grid-refinement evidence" in text
    assert "Reduced same-path current, jet diameter, droplet diameter, and charge-to-mass iteration-refinement evidence" in text
    assert "full_two_phase_navier_stokes_cfd_ready: True" in text
    assert "Submission readiness is now complete for a bounded mid-tier full-CFD solver claim" in text
    assert (
        "bounded-domain no-through momentum advection, incompressible Navier-Stokes Taylor-Green advection-viscosity-projection dynamics, two-phase density/viscosity momentum-kernel dynamics with grid-refinement evidence, Maxwell-stress divergence forcing and pressure balance, charge-free dielectric Maxwell-stress droplet deformation with V-squared scaling, grid-refinement, and timestep-refinement evidence, Maxwell-stress-enabled bounded-domain multiphysics, force-driven, force-kinematic, and advected force-kinematic axisymmetric interface updates with same-operator charge co-transport and grid-refinement evidence, "
        "axisymmetric cone-jet momentum advection/viscous/combined predictors, same-path advected combined-momentum grid-refinement evidence, same-path Huh-Wirz observable evidence, and same-path Huh-Wirz observable grid-refinement evidence plus iteration-refinement evidence"
    ) in text
