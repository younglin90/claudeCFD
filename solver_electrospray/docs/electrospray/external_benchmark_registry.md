# External Benchmark Registry

This registry identifies the external literature that must anchor the next manuscript-level quantitative comparison. It is not a claim that the current reduced solver has already reproduced every external dataset.

| Benchmark block | Reference | DOI / source | Observable to compare | Current local artifact |
|---|---|---|---|---|
| Cone-jet emission | Henry Huh and Richard E. Wirz, "Simulation of Electrospray Emission Processes for Highly Conductive Liquids" | DOI: 10.1063/5.0120737; arXiv:2111.10383 | current, droplet diameter, charge-to-mass ratio, cone-jet formation | `docs/electrospray/cone_jet_error_budget_table.md` |
| Droplet leaky-dielectric deformation | Debasish Das and David Saintillan, "A nonlinear small-deformation theory for transient droplet electrohydrodynamics" | DOI: 10.1017/jfm.2016.704; arXiv:1605.04036 | transient small-deformation response, charge relaxation, circulation trend | `docs/electrospray/coupled_droplet_grid_refinement_table.md` |
| 3D leaky-dielectric droplet dynamics | Debasish Das and David Saintillan, "Electrohydrodynamics of viscous drops in strong electric fields: Numerical simulations" | DOI: 10.1017/jfm.2017.560; arXiv:1612.02070 | 3D deformation, charge convection, Quincke-regime trend | `docs/electrospray/submission_readiness_matrix.md` |
| VOF charge leakage control | Qiang Liu, Jie Zhang, and Jian Wu, "Direct numerical simulations of incompressible multiphase electrohydrodynamic flow with single-phase transportation schemes" | arXiv:2207.08152 | gas-phase charge leakage, single-phase charge transport, interface flux correction | executable validation suite |

Before claiming SCI submission readiness, each external block must be converted into a numeric comparison table with explicit operating conditions, dimensional or nondimensional observables, tolerance rationale, and a citation-backed statement of what the local solver can and cannot reproduce.
