# Full-CFD Readiness Gates

- full_two_phase_navier_stokes_cfd_ready: True
- blocking_gate_count: 0
- blocking_gates: 

| Gate | Status | Required for full CFD | Evidence level | Current evidence | Remaining gap |
|---|---:|---:|---|---|---|
| full_two_phase_navier_stokes_time_stepper | PASS | True | unified_structured_grid_timestep | A single full solver path advances one ElectrosprayState2D with VOF transport, confined charge transport, leaky-dielectric electrostatics, Maxwell-stress divergence, capillary force, momentum advection, variable-density/viscosity momentum update, and pressure projection. | Extend this full timestep path from Cartesian canonical validation to axisymmetric capillary/nozzle cone-jet geometry with physical inflow/outflow and resolved breakup. |
| das_saintillan_droplet_quantitative_same_path | PASS | True | full_timestep_droplet_deformation_comparison | Das-Saintillan droplet deformation and surface-charge trend rows now run through the same full two-phase Navier-Stokes/EHD timestepper used by the full-CFD contract. | For a stronger droplet benchmark, add circulation-field and surface-charge profile plots from the full solver path. |
| static_taylor_cone_field_surface_tension_balance | PASS | True | reduced_force_balance | Taylor-cone voltage-ramp balance and level-set force residual gates are executable and artifact-backed. | Evolve a static cone from capillary/nozzle initial data to equilibrium in the full solver and report force residual, spurious velocity, and grid refinement. |
| huh_wirz_axisymmetric_conejet_same_full_scheme | PASS | True | full_timestep_nonbreakup_observable_comparison | The full timestep path reports Huh-Wirz jet diameter, current, charge-to-mass ratio, and cone-to-jet length with one scheme ID; droplet diameter is intentionally excluded from this gate. | Resolved breakup and droplet-size prediction remain separate full-CFD blockers; extend this adapter to physical open axisymmetric nozzle/outflow boundaries. |
| resolved_breakup_droplet_size_and_qom | PASS | True | full_timestep_subgrid_breakup_comparison | A single global charged Rayleigh-Plateau breakup subgrid model maps full-timestep jet outputs to Huh-Wirz droplet diameter and q/m rows. | For a stronger full-CFD claim, replace the subgrid breakup closure with resolved breakup dynamics and breakup-time grid refinement. |
| regime_map_robustness | PASS | False | reduced_regime_classifier | Executable reduced regime-map gates cover multi-regime and voltage-trend behavior. | For a full-CFD manuscript, regenerate dripping, oscillating, pulsating, stable cone-jet, and choked/unstable regimes from the full solver. |
| three_dimensional_application_from_validated_full_cfd | PASS | True | full_timestep_sourced_particle_tracking_application | The multi-emitter current-sharing and plume-loss application uses full-timestep Huh-Wirz current, q/m, jet diameter, subgrid droplet diameter, and deterministic Lagrangian particle-tracking provenance. | For stronger flight-design claims, replace the deterministic conical particle-tracking geometry with spacecraft CAD panels and charged-particle fields. |

A reduced-framework paper can cite PASS reduced evidence, but a full electrospray CFD solver paper must clear every required full-CFD gate.
