# 2D Multiphase Validation Index

Recommended execution order for the multidimensional `solver/five_eq_IMEX` extension.

| Case | File | Purpose |
|---|---|---|
| 2D-01 | `01_MATERIAL_advection.md` | Pressure-equilibrium material-interface advection |
| 2D-02 | `02_INTERFACE_deformation_reversal.md` | Sharp-interface deformation and reverse-time recovery |
| 2D-03 | `03_PE_interface_advection.md` | Large-density/EOS-jump pressure-equilibrium preservation |
| 2D-04 | `04_SHOCK_bubble.md` | Shock-interface interaction and pressure-oscillation control |
| 2D-05 | `05_RTI.md` | Gravity source, hydrostatic balance, Rayleigh-Taylor instability |

Common requirements:

- Use the same numerical method across the 2D validation set unless the case explicitly isolates a source term.
- Use second-order or higher reconstruction for primitive variables.
- Use a sharp-interface alpha scheme path for material interfaces.
- Avoid Rusanov as the primary material/advection flux.
- Preserve `0 <= alpha_k <= 1`, `sum(alpha_k)=1`, positive `p`, `rho_k`, and `T_k`.
- Save the final comparison plot as `results/2D/{case_name}/diff_vs_exact.png` and overwrite the file each run.
