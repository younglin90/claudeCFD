# Charge-Conservative Reduced EHD Solver for Electrospray Verification

## Scope Statement

This manuscript package documents a continuum electrospray verification solver built from electrostatics, leaky-dielectric charge transport, VOF-style interface transport, Maxwell/capillary forcing, viscous diffusion, and pressure projection kernels. The present implementation is a reduced structured-grid verification framework, not yet a production full two-phase Navier-Stokes cone-jet solver.

## Candidate Contribution

1. Charge-conservative interface transport and gas-phase leakage diagnostics.
2. Coupled 2D EHD interface evolution with electric forcing, Maxwell-stress divergence forcing, charge-free dielectric Maxwell-stress droplet deformation with V-squared scaling, grid-refinement, and timestep-refinement evidence, Maxwell-stress-enabled bounded-domain multiphysics, level-set capillarity, viscous damping, no-through-wall momentum advection, and pressure projection with `projection_velocity_update_norm` diagnostics, plus pure incompressible and two-phase density/viscosity Navier-Stokes momentum-kernel gates with grid-refinement evidence.
3. Axisymmetric reduced cone-jet state evolution with bounded annular VOF, confined charge, force-driven, force-kinematic, and VOF-advected force-kinematic interface update diagnostics, an explicit same-operator charge co-transport gate, grid-refinement evidence, momentum advection, viscous, combined predictor diagnostics, a same-path advected combined-momentum grid-refinement gate, same-path Huh-Wirz current/jet/droplet/q-m comparison evidence, same-path Huh-Wirz observable grid refinement, and grid- and iteration-refinement evidence, axisymmetric pressure projection, pressure-balance residual diagnostics, finite-volume observable extraction, and grid-refinement diagnostics.
4. Coupled 2D VOF and free-charge co-transport evidence through the same finite-volume transport path, including the executable `2d_coupled_ehd_same_operator_charge_transport` gate.
5. Coupled 2D Maxwell-stress divergence forcing evidence through the executable `2d_coupled_ehd_maxwell_stress_force`, `2d_coupled_ehd_pressure_maxwell_force_balance`, `2d_coupled_ehd_dielectric_maxwell_droplet_deformation`, `2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling`, `2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement`, `2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement`, and `2d_coupled_ehd_bounded_domain_multiphysics` gates, separating polarization-force acceleration from free-charge body forcing, checking static pressure balance against Maxwell-stress divergence, evolving a charge-free dielectric droplet interface under Maxwell-stress divergence, verifying V-squared scaling of deformation and Maxwell-stress acceleration, checking grid-refinement consistency, and checking timestep-refinement consistency.
6. Taylor-Melcher droplet deformation, Taylor-cone force balance, cone-jet observable, multi-emitter, plume, and microthruster validation gates.
7. Paper-ready executable validation artifacts with manifest coverage and zero-failure regression gates.

## Validation Evidence To Report

| Block | Evidence | Current Gate |
|---|---|---|
| 1D electrostatics | Parallel plate, dielectric jump, charge relaxation, Maxwell jump | executable validation |
| VOF/interface transport | boundedness, charge confinement, same-operator 2D VOF/free-charge co-transport, leakage accounting | executable validation |
| 2D droplet | Taylor-Melcher reference, transient response, coupled alpha deformation, incompressible Navier-Stokes Taylor-Green advection-viscosity-projection, two-phase density/viscosity momentum-kernel dynamics with grid refinement, Maxwell-stress divergence forcing and pressure balance, charge-free dielectric Maxwell-stress droplet deformation with V-squared scaling, grid refinement, and timestep refinement, Maxwell-stress-enabled bounded-domain multiphysics, bounded-domain no-through-wall momentum advection, `projection_velocity_update_norm`, grid refinement | executable validation |
| Taylor cone | angle, level set, Maxwell-capillary balance, voltage ramp closure | executable validation |
| Cone jet | stateful pseudo-time interface focusing, force-driven electric-capillary interface update, force-kinematic pressure-imbalance interface update, VOF-advected force-kinematic interface update, explicit same-operator charge co-transport gate, grid refinement, axisymmetric momentum advection predictor, axisymmetric viscous momentum predictor, combined advection-viscosity-projection predictor, combined momentum grid refinement, same-path advected combined-momentum grid refinement, same-path Huh-Wirz current/jet/droplet/q-m comparison, same-path Huh-Wirz observable grid refinement and iteration refinement, pressure-projection divergence reduction, pressure-balance residual, current, jet diameter, droplet diameter, charge-to-mass error budget, grid refinement | executable reduced-kernel validation |
| 3D application | cone-jet-sourced multi-emitter shielding/current sharing, plume impingement, microthruster metrics | reduced application validation |

## Required Figures And Tables

1. Solver architecture diagram: electrostatics, charge transport, interface advection, EHD force, projection.
2. 1D verification table: errors and pass thresholds.
3. 2D droplet deformation table: Taylor-Melcher prediction, coupled deformation, incompressible Navier-Stokes Taylor-Green advection-viscosity-projection, two-phase density/viscosity momentum-kernel dynamics and grid-refinement ratio, Maxwell-stress divergence forcing and pressure balance, charge-free dielectric Maxwell-stress droplet deformation with V-squared scaling, grid refinement, and timestep refinement, no-through-wall momentum advection, projection velocity update.
4. Taylor cone table: cone angle, force residual, voltage-ramp residual.
5. Cone-jet observable error-budget table: stateful pseudo-time focusing history, force-driven electric-capillary interface update, force-kinematic pressure-imbalance interface update, VOF-advected force-kinematic interface update, explicit same-operator charge co-transport gate, grid refinement, axisymmetric momentum advection predictor, axisymmetric viscous momentum predictor, combined advection-viscosity-projection predictor, combined momentum grid refinement, same-path advected combined-momentum grid refinement, same-path Huh-Wirz current/jet/droplet/q-m comparison, same-path Huh-Wirz observable grid refinement and iteration refinement, axisymmetric pressure-projection diagnostics, pressure-balance residual, current, jet diameter, droplet diameter, charge-to-mass, grid refinement.
6. Multi-emitter/plume application table: current sharing, shielding, plume loss, performance closure sourced from the cone-jet adapter.

## Claims Allowed Now

- The code has a mechanically verified continuum EHD verification suite with all executable gates passing.
- The coupled 2D reduced stepper evolves interface fields under electric, Maxwell-stress divergence, capillary, viscous, no-through-wall momentum advection, and projected incompressible velocity updates, including the executable `2d_coupled_ehd_incompressible_ns_taylor_green`, `2d_coupled_ehd_two_phase_ns_momentum_kernel`, `2d_coupled_ehd_two_phase_ns_momentum_grid_refinement`, `2d_coupled_ehd_maxwell_stress_force`, `2d_coupled_ehd_pressure_maxwell_force_balance`, `2d_coupled_ehd_dielectric_maxwell_droplet_deformation`, `2d_coupled_ehd_dielectric_maxwell_droplet_voltage_scaling`, `2d_coupled_ehd_dielectric_maxwell_droplet_grid_refinement`, `2d_coupled_ehd_dielectric_maxwell_droplet_timestep_refinement`, `2d_coupled_ehd_bounded_domain_multiphysics`, and `2d_coupled_ehd_no_through_momentum_advection` gates plus `projection_velocity_update_norm` diagnostics.
- The axisymmetric reduced cone-jet adapter evolves a bounded annular VOF interface and confined charge field toward a focused cone-jet state, applies the executable `2d_cone_jet_axisymmetric_force_driven_interface`, `2d_cone_jet_axisymmetric_force_kinematic_interface`, `2d_cone_jet_axisymmetric_advected_force_kinematic_interface`, `2d_cone_jet_axisymmetric_advected_force_kinematic_charge_cotransport`, `2d_cone_jet_axisymmetric_advected_force_kinematic_grid_refinement`, `2d_cone_jet_axisymmetric_momentum_advection_predictor`, `2d_cone_jet_axisymmetric_viscous_momentum_predictor`, `2d_cone_jet_axisymmetric_combined_momentum_predictor`, `2d_cone_jet_axisymmetric_combined_momentum_grid_refinement`, `2d_cone_jet_axisymmetric_advected_combined_momentum_grid_refinement`, `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz`, `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_grid_refinement`, and `2d_cone_jet_axisymmetric_advected_combined_momentum_huh_wirz_iteration_refinement` gates, co-transports interface and normalized charge with the same upwind operator, projects the axisymmetric velocity predictor to reduce divergence, reports pressure-balance residual diagnostics, and extracts Huh-Wirz comparison observables with grid- and iteration-refinement evidence.
- The reduced application layer propagates cone-jet-sourced current and charge-to-mass into multi-emitter shielding and plume impingement accounting.
- The generated validation artifacts are internally consistent and manifest-complete.

## Claims Not Yet Allowed

- Do not claim a full production two-phase Navier-Stokes electrospray CFD solver.
- Do not claim resolved cone-jet breakup; current droplet-size evidence is an observable closure, not a breakup-resolved Navier-Stokes result.
- Do not claim top-tier CFD readiness unless the full two-phase Navier-Stokes cone-jet GAP is closed.
- For SCI mid-tier positioning, frame the paper as a charge-conservative reduced EHD validation framework with executable literature comparisons and application accounting, not as a completed production cone-jet CFD solver.

## Current Acceptance Command

```bash
cmake --build build && build/verify_electrospray_cpp --build-dir build
```
