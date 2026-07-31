from __future__ import annotations

from pathlib import Path


def test_sci_manuscript_skeleton_contains_required_submission_sections() -> None:
    text = Path("docs/electrospray/sci_manuscript_skeleton.md").read_text()

    for heading in (
        "## Scope Statement",
        "## Candidate Contribution",
        "## Validation Evidence To Report",
        "## Required Figures And Tables",
        "## Claims Allowed Now",
        "## Claims Not Yet Allowed",
        "## Current Acceptance Command",
    ):
        assert heading in text


def test_sci_manuscript_skeleton_keeps_solver_scope_honest() -> None:
    text = Path("docs/electrospray/sci_manuscript_skeleton.md").read_text()

    assert "reduced structured-grid verification framework" in text
    assert "Do not claim a full production two-phase Navier-Stokes electrospray CFD solver." in text
    assert "Taylor-Melcher" in text
    assert "Cone-jet observable error-budget table" in text
    assert "no-through-wall momentum advection" in text
    assert "projection_velocity_update_norm" in text
    assert "2d_coupled_ehd_no_through_momentum_advection" in text
    assert "2d_coupled_ehd_incompressible_ns_taylor_green" in text
    assert "2d_coupled_ehd_two_phase_ns_momentum_kernel" in text
    assert "2d_coupled_ehd_two_phase_ns_momentum_grid_refinement" in text
    assert "incompressible Navier-Stokes Taylor-Green advection-viscosity-projection" in text
    assert "two-phase density/viscosity momentum-kernel" in text
    assert "grid-refinement evidence" in text
    assert "2d_coupled_ehd_same_operator_charge_transport" in text
    assert "2d_coupled_ehd_maxwell_stress_force" in text
    assert "2d_coupled_ehd_pressure_maxwell_force_balance" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_deformation" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement" in text
    assert "2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement" in text
    assert "2d_coupled_ehd_bounded_domain_multiphysics" in text
    assert "Maxwell-stress divergence forcing" in text
    assert "charge-free dielectric Maxwell-stress droplet deformation" in text
    assert "V-squared scaling" in text
    assert "verifying V-squared scaling of deformation and Maxwell-stress acceleration" in text
    assert "checking grid-refinement consistency" in text
    assert "checking timestep-refinement consistency" in text
    assert "Maxwell-stress-enabled bounded-domain multiphysics" in text
    assert "static pressure balance against Maxwell-stress divergence" in text
    assert "same-operator 2D VOF/free-charge co-transport" in text
    assert "force-driven electric-capillary interface update" in text
    assert "2d_cone_jet_axisymmetric_force_driven_interface" in text
    assert "force-kinematic pressure-imbalance interface update" in text
    assert "2d_cone_jet_axisymmetric_force_kinematic_interface" in text
    assert "VOF-advected force-kinematic interface update" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_interface" in text
    assert "explicit same-operator charge co-transport gate" in text
    assert "same-operator charge co-transport" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport" in text
    assert "2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement" in text
    assert "axisymmetric momentum advection predictor" in text
    assert "2d_cone_jet_axisymmetric_momentum_advection_predictor" in text
    assert "axisymmetric viscous momentum predictor" in text
    assert "2d_cone_jet_axisymmetric_viscous_momentum_predictor" in text
    assert "combined advection-viscosity-projection predictor" in text
    assert "2d_cone_jet_axisymmetric_combined_momentum_predictor" in text
    assert "combined momentum grid refinement" in text
    assert "2d_cone_jet_axisymmetric_combined_momentum_grid_refinement" in text
    assert "same-path advected combined-momentum grid refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement" in text
    assert "same-path Huh-Wirz current/jet/droplet/q-m comparison" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz" in text
    assert "same-path Huh-Wirz observable grid refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement" in text
    assert "2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement" in text
    assert "grid- and iteration-refinement evidence" in text
    assert "stateful pseudo-time interface focusing" in text
    assert "axisymmetric pressure-projection diagnostics" in text
    assert "reduce divergence" in text
    assert "pressure-balance residual" in text
    assert "bounded annular VOF" in text
    assert "grid-refinement evidence" in text
    assert "cone-jet-sourced current and charge-to-mass" in text
    assert "charge-conservative reduced EHD validation framework" in text
