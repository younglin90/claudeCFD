# 2D Multiphase Validation Index

Recommended execution order for the multidimensional `solver/five_eq_IMEX` extension.

| Case | File | Purpose |
|---|---|---|
| 2D-01 | `01_MATERIAL_advection.md` | Pressure-equilibrium material-interface advection |
| 2D-02 | `02_INTERFACE_deformation_reversal.md` | Sharp-interface deformation and reverse-time recovery |
| 2D-03 | `03_PE_interface_advection.md` | Large-density/EOS-jump pressure-equilibrium preservation |
| 2D-04 | `04_SHOCK_bubble.md` | Shock-interface interaction and pressure-oscillation control |
| 2D-05 | `05_RTI.md` | Gravity source, hydrostatic balance, Rayleigh-Taylor instability |
| 2D-06 | `06_LM_PE_interface_advection.md` | Ultra-low-Mach pressure-equilibrium interface advection |
| 2D-07 | `07_LM_liquid_gas_alternating_bands.md` | Low-Mach liquid-gas alternating bands |
| 2D-08 | `08_LM_acoustic_interface_pulse.md` | Low-Mach acoustic pulse through material interface |
| 2D-09 | `09_RM_single_mode_air_SF6.md` | Single-mode Richtmyer-Meshkov instability |
| 2D-10 | `10_SHOCK_bubble_Haas_Sturtevant.md` | Haas-Sturtevant / Quirk-Karni shock-bubble |
| 2D-11 | `11_linear_RTI.md` | Linear Rayleigh-Taylor instability |
| 2D-12 | `12_linear_KH.md` | Linear Kelvin-Helmholtz instability |
| 2D-13 | `13_shock_droplet_aerobreakup.md` | Shock-droplet aerobreakup smoke validation |
| 2D-14 | `14_dam_break_or_wave_impact.md` | Dam-break / wave-impact low-Mach free-surface limit |

Common requirements:

- Use the same numerical method across the 2D validation set unless the case explicitly isolates a source term.
- Use second-order or higher reconstruction for primitive variables.
- Use a sharp-interface alpha scheme path for material interfaces.
- Avoid Rusanov as the primary material/advection flux.
- Preserve `0 <= alpha_k <= 1`, `sum(alpha_k)=1`, positive `p`, `rho_k`, and `T_k`.
- Save the final comparison plot as `results/2D/{case_name}/diff_vs_exact.png` and overwrite the file each run.
