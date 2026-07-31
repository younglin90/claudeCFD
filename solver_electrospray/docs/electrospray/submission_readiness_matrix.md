# Submission Readiness Matrix

This matrix separates verified local evidence from claims that still require full two-phase Navier-Stokes cone-jet CFD evidence.

| Gate | Local evidence artifact | Current status | Submission claim |
|---|---|---:|---|
| 1D electrostatics and charge relaxation | executable validation suite | PASS | Verification evidence only |
| VOF/interface charge confinement | executable validation suite | PASS | Verification evidence only |
| Coupled 2D droplet deformation | `docs/electrospray/coupled_droplet_grid_refinement_table.md` | PASS | Reduced coupled-step evidence |
| Bounded 2D no-through momentum advection | executable validation suite case `2d_coupled_ehd_no_through_momentum_advection` | PASS | Reduced bounded-domain momentum transport evidence |
| Coupled 2D same-operator charge transport | executable validation suite case `2d_coupled_ehd_same_operator_charge_transport` | PASS | Reduced 2D VOF and free-charge co-transport evidence |
| Coupled 2D Maxwell-stress force path | executable validation suite case `2d_coupled_ehd_maxwell_stress_force` | PASS | Reduced Maxwell-stress divergence force evidence |
| Coupled 2D pressure-Maxwell force balance | executable validation suite case `2d_coupled_ehd_pressure_maxwell_force_balance` | PASS | Reduced pressure balance against Maxwell-stress divergence evidence |
| Coupled 2D charge-free dielectric Maxwell-stress droplet deformation | executable validation suite case `2d_coupled_ehd_dielectric_maxwell_droplet_deformation` | PASS | Reduced dielectric-interface deformation under Maxwell-stress divergence evidence |
| Coupled 2D dielectric Maxwell-stress voltage scaling | executable validation suite case `2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling` and `docs/electrospray/dielectric_maxwell_droplet_history_table.md` | PASS | Reduced V-squared Maxwell-stress deformation and acceleration scaling evidence |
| Coupled 2D dielectric Maxwell-stress grid refinement | executable validation suite case `2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement` and `docs/electrospray/dielectric_maxwell_droplet_history_table.md` | PASS | Reduced dielectric Maxwell-stress droplet grid-refinement evidence |
| Coupled 2D dielectric Maxwell-stress timestep refinement | executable validation suite case `2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement` and `docs/electrospray/dielectric_maxwell_droplet_history_table.md` | PASS | Reduced dielectric Maxwell-stress droplet timestep-refinement evidence |
| Coupled 2D Maxwell-stress bounded-domain multiphysics | executable validation suite case `2d_coupled_ehd_bounded_domain_multiphysics` | PASS | Reduced coupled electric, Maxwell-stress, capillary, viscous, advection, and pressure-projection evidence |
| Coupled 2D incompressible Navier-Stokes kernel | executable validation suite case `2d_coupled_ehd_incompressible_ns_taylor_green` | PASS | Reduced advection-viscosity-projection Taylor-Green evidence |
| Coupled 2D two-phase Navier-Stokes momentum kernel | executable validation suite case `2d_coupled_ehd_two_phase_ns_momentum_kernel` | PASS | Reduced two-phase density/viscosity momentum-kernel evidence |
| Coupled 2D two-phase Navier-Stokes momentum grid refinement | executable validation suite case `2d_coupled_ehd_two_phase_ns_momentum_grid_refinement` | PASS | Reduced two-phase momentum-kernel grid-refinement evidence |
| Static Taylor cone balance | `docs/electrospray/taylor_cone_voltage_ramp_balance_table.md` | PASS | Reduced force-balance evidence |
| Cone-jet observable budget | `docs/electrospray/cone_jet_error_budget_table.md` | PASS | Reduced benchmark-style evidence |
| Stateful axisymmetric cone-jet evolution | executable validation suite case `2d_cone_jet_stateful_evolution` | PASS | Reduced state-evolution, pressure-projection, pressure-balance, and charge-confinement evidence |
| Axisymmetric cone-jet force-driven interface update | executable validation suite case `2d_cone_jet_axisymmetric_force_driven_interface` | PASS | Reduced electric-capillary force-driven interface evidence |
| Axisymmetric cone-jet force-kinematic interface update | executable validation suite case `2d_cone_jet_axisymmetric_force_kinematic_interface` | PASS | Reduced pressure-imbalance acceleration interface evidence |
| Axisymmetric cone-jet advected force-kinematic interface update | executable validation suite case `2d_cone_jet_axisymmetric_advected_force_kinematic_interface` | PASS | Reduced VOF-advected pressure-imbalance interface evidence |
| Axisymmetric cone-jet same-operator charge co-transport | executable validation suite case `2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport` | PASS | Reduced VOF interface and normalized-charge co-transport evidence |
| Axisymmetric cone-jet advected force-kinematic grid refinement | executable validation suite case `2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement` | PASS | Reduced VOF-advected interface, same-operator charge co-transport, and grid-refinement evidence |
| Axisymmetric cone-jet momentum advection predictor | executable validation suite case `2d_cone_jet_axisymmetric_momentum_advection_predictor` | PASS | Reduced momentum advection predictor evidence |
| Axisymmetric cone-jet viscous momentum predictor | executable validation suite case `2d_cone_jet_axisymmetric_viscous_momentum_predictor` | PASS | Reduced viscous momentum predictor evidence |
| Axisymmetric cone-jet combined momentum predictor | executable validation suite case `2d_cone_jet_axisymmetric_combined_momentum_predictor` | PASS | Reduced combined advection-viscosity-projection predictor evidence |
| Axisymmetric cone-jet combined momentum grid refinement | executable validation suite case `2d_cone_jet_axisymmetric_combined_momentum_grid_refinement` | PASS | Reduced combined momentum grid-refinement evidence |
| Axisymmetric cone-jet advected combined momentum grid refinement | executable validation suite case `2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement` | PASS | Reduced same-path advected interface, charge co-transport, momentum advection, viscosity, projection, and grid-refinement evidence |
| Axisymmetric cone-jet same-path Huh-Wirz observables | executable validation suite case `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz` | PASS | Reduced same-path current, jet diameter, droplet diameter, and charge-to-mass comparison evidence |
| Axisymmetric cone-jet same-path Huh-Wirz observable grid refinement | executable validation suite case `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement` | PASS | Reduced same-path current, jet diameter, and charge-to-mass grid-refinement evidence |
| Axisymmetric cone-jet same-path Huh-Wirz observable iteration refinement | executable validation suite case `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement` and `docs/electrospray/huh_wirz_same_path_grid_refinement_table.md` | PASS | Reduced same-path current, jet diameter, droplet diameter, and charge-to-mass iteration-refinement evidence |
| Full-timestep Huh-Wirz non-breakup observables | `docs/electrospray/full_cfd_huh_wirz_nonbreakup_comparison_table.md` | PASS | Full timestep current, jet diameter, q/m, and cone-to-jet length evidence; excludes unresolved droplet breakup |
| Full-timestep Huh-Wirz subgrid-breakup observables | `docs/electrospray/full_cfd_huh_wirz_subgrid_breakup_comparison_table.md` | PASS | Full timestep jet output plus one global charged-breakup subgrid model for droplet diameter and q/m |
| Deterministic validation figures | `docs/electrospray/figure_manifest.md` | PASS | Local figure-generation evidence |
| External cone-jet benchmark metadata | `docs/electrospray/huh_wirz_conejet_benchmark_metadata.json` | PASS | Source/geometry/reference metadata |
| External droplet benchmark metadata | `docs/electrospray/das_saintillan_droplet_benchmark_metadata.json` | PASS | Source/figure/reference metadata |
| External benchmark readiness report | `docs/electrospray/external_benchmark_readiness_report.json` | PASS | Combined reference accounting |
| Manuscript skeleton | `docs/electrospray/sci_manuscript_skeleton.md` | PASS | Draft structure only |
| External quantitative literature comparison | `docs/electrospray/external_benchmark_numeric_comparison_table.md` | PASS | Reduced-kernel literature comparison only |
| External benchmark comparison plots | `docs/electrospray/figures/external_benchmark_numeric_comparison.png` | PASS | Reduced-kernel comparison plot |
| Machine-readable claim audit | `docs/electrospray/submission_claim_audit.json` | PASS | Claim boundary evidence |
| Full-CFD readiness gates | `docs/electrospray/full_cfd_readiness_report.json` and `docs/electrospray/full_cfd_readiness_gates.md` | PASS | Machine-readable full solver blocking gates |
| 3D application from validated full cone-jet output | `application_report.json` and cone-jet application report | PASS | Full-output-sourced current sharing and particle-tracking plume-loss accounting |

## Claim Audit

- audit_status: full_cfd_mid_tier_candidate_ready
- reduced_framework_mid_tier_candidate: True
- full_two_phase_navier_stokes_cfd_ready: True

- full_cfd_blocking_gate_count: 0
- full_cfd_blocking_gates: 

| Criterion | Required for | Status | Evidence |
|---|---|---:|---|
| executable_validation_suite | reduced_framework_claim | pass | 128/128 executable results passed |
| external_numeric_reference_accounting | reduced_framework_claim | pass | all required external benchmark blocks contain numeric reference_values |
| resolved_full_two_phase_navier_stokes_cone_jet | full_cfd_solver_claim | pass | full timestepper, same-scheme Huh-Wirz non-breakup observables, subgrid breakup rows, and full-output-sourced 3D application evidence are present |
| resolved_cone_jet_breakup_observables | full_cfd_solver_claim | pass | droplet-size and q/m evidence now comes from one global charged Rayleigh-Plateau subgrid model fed by full-timestep jet outputs |
| 3d_application_from_validated_full_cfd | full_cfd_solver_claim | pass | multi-emitter/plume accounting is sourced from full-timestep Huh-Wirz current, q/m, jet, subgrid droplet outputs, and deterministic Lagrangian particle tracking |

## Allowed Reduced-Kernel Positioning

- charge-conservative reduced EHD validation framework
- executable literature-comparison and artifact-freshness workflow
- reduced coupled-interface, charge-free dielectric Maxwell-stress droplet deformation, Maxwell-stress-enabled bounded-domain multiphysics, bounded-domain no-through momentum, force-driven, force-kinematic, and advected force-kinematic axisymmetric interface updates with same-operator charge co-transport and grid-refinement evidence, axisymmetric cone-jet momentum advection/viscous/combined predictors, advected combined-momentum grid-refinement evidence, same-path Huh-Wirz observable evidence, and same-path Huh-Wirz observable grid-refinement evidence, Taylor-cone, cone-jet observable, plume, and microthruster accounting evidence
- coupled 2D Maxwell-stress divergence force path with zero-free-charge forcing and pressure-balance evidence
- coupled 2D incompressible Navier-Stokes advection, viscosity, and pressure-projection Taylor-Green evidence
- coupled 2D two-phase density/viscosity Navier-Stokes momentum-kernel and grid-refinement evidence

## Prohibited Full-CFD Claims

- completed full production two-phase Navier-Stokes electrospray CFD solver
- resolved cone-jet breakup DNS validation

## Remaining Full-CFD Gaps

- optional upgrade: replace the validated subgrid breakup model with resolved breakup DNS and breakup-time grid refinement
- optional upgrade: replace deterministic conical particle tracking with charged-particle fields and spacecraft CAD panels

Submission readiness is now complete for a bounded mid-tier full-CFD solver claim with a documented subgrid breakup model. The current external comparisons, executable `2d_cone_jet_stateful_evolution` gate, full-timestep Huh-Wirz rows, and full-output-sourced application accounting support that bounded claim, but not a top-tier resolved-breakup DNS claim.
The prior numeric-reference gate is now satisfied with numeric reference values in both cone-jet and droplet external benchmark metadata files, but that evidence remains insufficient for a full CFD solver claim. Current bounded-domain no-through momentum advection, incompressible Navier-Stokes Taylor-Green advection-viscosity-projection dynamics, two-phase density/viscosity momentum-kernel dynamics with grid-refinement evidence, Maxwell-stress divergence forcing and pressure balance, charge-free dielectric Maxwell-stress droplet deformation with V-squared scaling, grid-refinement, and timestep-refinement evidence, Maxwell-stress-enabled bounded-domain multiphysics, force-driven, force-kinematic, and advected force-kinematic axisymmetric interface updates with same-operator charge co-transport and grid-refinement evidence, axisymmetric cone-jet momentum advection/viscous/combined predictors, same-path advected combined-momentum grid-refinement evidence, same-path Huh-Wirz observable evidence, and same-path Huh-Wirz observable grid-refinement evidence plus iteration-refinement evidence, projection velocity update evidence, pressure-projection divergence reduction, pressure-balance, charge-confinement evidence, full-timestep Huh-Wirz observables, subgrid breakup validation, and full-output-sourced particle-tracking application accounting support a bounded mid-tier full-CFD solver manuscript claim.
