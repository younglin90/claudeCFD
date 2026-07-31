# 3D Multiphase Validation Index

Recommended execution order for the multidimensional `solver/five_eq_IMEX` extension.

| Case | File | Purpose |
|---|---|---|
| 3D-01 | `01_MATERIAL_sphere_advection.md` | Pressure-equilibrium sphere advection |
| 3D-02 | `02_INTERFACE_deformation_reversal.md` | 3D interface deformation and recovery |
| 3D-03 | `03_RTI.md` | 3D Rayleigh-Taylor instability |
| 3D-04 | `04_SHOCK_sphere.md` | 3D shock-interface interaction |
| 3D-05 | `05_DAM_break_obstacle.md` | Air-water dam-break impact with gravity |

Common requirements:

- Use the same numerical method across the 3D validation set unless the case explicitly isolates a source term.
- Use second-order or higher reconstruction for primitive variables.
- Use a sharp-interface alpha scheme path for material interfaces.
- Avoid Rusanov as the primary material/advection flux.
- Preserve `0 <= alpha_k <= 1`, `sum(alpha_k)=1`, positive `p`, `rho_k`, and `T_k`.
- Save the final comparison plot as `results/3D/{case_name}/diff_vs_exact.png` and overwrite the file each run.
