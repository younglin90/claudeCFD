from pathlib import Path


def test_submission_readiness_matrix_lists_required_artifacts_and_gaps() -> None:
    text = Path("docs/electrospray/submission_readiness_matrix.md").read_text(encoding="utf-8")

    for artifact in (
        "docs/electrospray/coupled_droplet_grid_refinement_table.md",
        "2d_coupled_ehd_incompressible_ns_taylor_green",
        "2d_coupled_ehd_two_phase_ns_momentum_kernel",
        "2d_coupled_ehd_two_phase_ns_momentum_grid_refinement",
        "2d_coupled_ehd_no_through_momentum_advection",
        "2d_coupled_ehd_maxwell_stress_force",
        "2d_coupled_ehd_pressure_maxwell_force_balance",
        "docs/electrospray/taylor_cone_voltage_ramp_balance_table.md",
        "docs/electrospray/cone_jet_error_budget_table.md",
        "2d_cone_jet_stateful_evolution",
        "2d_cone_jet_axisymmetric_force_driven_interface",
        "2d_cone_jet_axisymmetric_force_kinematic_interface",
        "2d_cone_jet_axisymmetric_advected_force_kinematic_interface",
        "2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement",
        "2d_cone_jet_axisymmetric_momentum_advection_predictor",
        "2d_cone_jet_axisymmetric_viscous_momentum_predictor",
        "2d_cone_jet_axisymmetric_combined_momentum_predictor",
        "2d_cone_jet_axisymmetric_combined_momentum_grid_refinement",
        "2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement",
        "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz",
        "docs/electrospray/figure_manifest.md",
        "docs/electrospray/huh_wirz_conejet_benchmark_metadata.json",
        "docs/electrospray/full_cfd_huh_wirz_nonbreakup_comparison_table.md",
        "docs/electrospray/full_cfd_huh_wirz_subgrid_breakup_comparison_table.md",
        "docs/electrospray/das_saintillan_droplet_benchmark_metadata.json",
        "docs/electrospray/external_benchmark_readiness_report.json",
        "docs/electrospray/full_cfd_readiness_report.json",
        "docs/electrospray/full_cfd_readiness_gates.md",
        "docs/electrospray/sci_manuscript_skeleton.md",
    ):
        assert artifact in text

    for completed_gate in (
        "External quantitative literature comparison",
        "External benchmark comparison plots",
    ):
        assert completed_gate in text
        assert "Reduced-kernel" in text

    assert "3D application from validated full cone-jet output" in text
    assert "Full-output-sourced current sharing and particle-tracking plume-loss accounting" in text


def test_submission_readiness_matrix_prevents_overclaiming_sci_readiness() -> None:
    text = Path("docs/electrospray/submission_readiness_matrix.md").read_text(encoding="utf-8")

    assert "Submission readiness is now complete" in text
    assert "numeric reference values in both cone-jet and droplet external benchmark metadata files" in text
    assert "bounded mid-tier full-CFD solver claim" in text
    assert "Full-CFD readiness gates" in text
    assert "full_cfd_blocking_gates" in text
    assert "full_cfd_mid_tier_candidate_ready" in text
    assert "pressure-projection, pressure-balance, and charge-confinement evidence" in text
    assert "bounded-domain no-through momentum advection" in text
    assert "incompressible Navier-Stokes Taylor-Green advection-viscosity-projection dynamics" in text
    assert "two-phase density/viscosity momentum-kernel dynamics" in text
    assert "two-phase density/viscosity momentum-kernel dynamics with grid-refinement evidence" in text
    assert "Maxwell-stress divergence forcing" in text
    assert "Maxwell-stress divergence forcing and pressure balance" in text
    assert "Maxwell-stress-enabled bounded-domain multiphysics" in text
    assert (
        "force-driven, force-kinematic, and advected force-kinematic axisymmetric interface updates "
        "with same-operator charge co-transport and grid-refinement evidence"
    ) in text
    assert "axisymmetric cone-jet momentum advection/viscous/combined predictors, same-path advected combined-momentum grid-refinement evidence, same-path Huh-Wirz observable evidence, and same-path Huh-Wirz observable grid-refinement evidence" in text
    assert "Reduced electric-capillary force-driven interface evidence" in text
    assert "Reduced pressure-imbalance acceleration interface evidence" in text
    assert "Reduced VOF-advected pressure-imbalance interface evidence" in text
    assert "Reduced VOF-advected interface, same-operator charge co-transport, and grid-refinement evidence" in text
    assert "Reduced momentum advection predictor evidence" in text
    assert "Reduced viscous momentum predictor evidence" in text
    assert "Reduced combined advection-viscosity-projection predictor evidence" in text
    assert "Reduced combined momentum grid-refinement evidence" in text
    assert "Reduced same-path advected interface, charge co-transport, momentum advection, viscosity, projection, and grid-refinement evidence" in text
    assert "Reduced same-path current, jet diameter, droplet diameter, and charge-to-mass comparison evidence" in text
    assert "Reduced same-path current, jet diameter, and charge-to-mass grid-refinement evidence" in text
    assert "Full timestep current, jet diameter, q/m, and cone-to-jet length evidence" in text
    assert "Full timestep jet output plus one global charged-breakup subgrid model" in text
    assert "projection velocity update evidence" in text
    assert "Reduced bounded-domain momentum transport evidence" in text
    assert "Reduced advection-viscosity-projection Taylor-Green evidence" in text
    assert "Reduced two-phase density/viscosity momentum-kernel evidence" in text
    assert "Reduced two-phase momentum-kernel grid-refinement evidence" in text
    assert "Reduced Maxwell-stress divergence force evidence" in text
    assert "Reduced pressure balance against Maxwell-stress divergence evidence" in text
    assert "Reduced coupled electric, Maxwell-stress, capillary, viscous, advection, and pressure-projection evidence" in text
    assert "pressure-projection divergence reduction" in text
    assert "Full-output-sourced current sharing and particle-tracking plume-loss accounting" in text
    assert "full-output-sourced particle-tracking application accounting" in text
    assert "bounded mid-tier full-CFD solver manuscript claim" in text
