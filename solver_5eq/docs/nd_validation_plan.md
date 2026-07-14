# 2-D/3-D Validation Plan for `solver/five_eq_IMEX`

This document records the recommended multidimensional validation order and the
new solver entry points.

Detailed case specifications are recorded in
`docs/multiphase_2d3d_validation_specs.md`.

## Solver Entry Points

- 2-D: `solver.five_eq_IMEX.main_2d.solve_2d`
- 3-D: `solver.five_eq_IMEX.main_3d.solve_3d`

Primitive variables:

- 2-D: `W = (alpha1, T1, T2, ux, uy, p)`
- 3-D: `W = (alpha1, T1, T2, ux, uy, uz, p)`

Conservative variables:

- 2-D: `U = (alpha1*rho1, alpha2*rho2, rho*ux, rho*uy, rhoE, alpha1)`
- 3-D: `U = (alpha1*rho1, alpha2*rho2, rho*ux, rho*uy, rho*uz, rhoE, alpha1)`

Current ND numerical path:

- Time integration: SSPRK(3,3), exposed through `time_integrator="imex_ssp3"` API.
- Material/advection flux: HLLC normal flux.
- Primitive reconstruction: TVD MUSCL with `T-MLP-u` local boundedness option.
- Default limiter: `superbee`.
- Alpha path: directionally bounded compressive TVD path for sharp-interface smoke tests.
- Boundary conditions: periodic, transmissive/outflow/zeroGradient, reflective/wall.
- Sources: optional constant gravity vector.

The validated 1-D IMEX pressure block is not modified. A true multidimensional
implicit acoustic/pressure block should be added later behind `rhs_nd`.

## Recommended 2-D Order

1. `2D_01_material_advection`
   - Pressure-equilibrium material-interface advection.
   - Purpose: verify multidimensional alpha transport, pressure preservation,
     and result/plot plumbing.
   - Driver: `PYTHONPATH=. python3 results/run_2d3d_recommended.py --case 2d`
   - Plot: `results/2D/01_material_advection/diff_vs_exact.png`

2. `2D_02_deformation_reversal`
   - Rider-Kothe / LeVeque-type deformation reversal.
   - Purpose: sharp-interface deformation and reverse-time recovery.

3. `2D_03_pressure_equilibrium_interface_advection`
   - Johnsen-Ham-style pressure-equilibrium material interface.
   - Purpose: suppress pressure/velocity oscillations across density/EOS jumps.

4. `2D_04_shock_bubble_or_shock_cylinder`
   - Haas-Sturtevant / Giordano-Burtschell family.
   - Purpose: shock-interface interaction and shock position.

5. `2D_05_rayleigh_taylor`
   - Purpose: gravity source, large density ratio, long-time stability.

## Recommended 3-D Order

1. `3D_01_sphere_advection`
   - Pressure-equilibrium sphere advection.
   - Driver: `PYTHONPATH=. python3 results/run_2d3d_recommended.py --case 3d`
   - Plot: `results/3D/01_sphere_advection/diff_vs_exact.png`

2. `3D_02_deformation_reversal`
   - Liovic/Rider-Kothe 3-D sphere deformation.
   - Purpose: 3-D interface preservation and mass conservation.

3. `3D_03_rayleigh_taylor`
   - Purpose: gravity source and nonlinear interface growth.

4. `3D_04_shock_sphere`
   - Purpose: compressible shock-interface interaction.

5. `3D_05_dam_break_obstacle`
   - Purpose: air-water large density ratio, gravity, impact pressure trend.

## Plot Rule

Each validation case overwrites:

- `results/2D/{case_name}/diff_vs_exact.png`
- `results/3D/{case_name}/diff_vs_exact.png`

No round-specific plot names.
