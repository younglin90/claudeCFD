# Figure manifest — 5-equation all-speed IMEX solver (1D Euler evidence)

Proposed paper figure plan. All files live flat in `docs/paper/figures/` (75 PNGs,
copied verbatim from `results/1D/paper_euler_evidence/plots/`). Descriptions marked
**[inspected]** were read from the rendered PNG and describe what is actually visible;
others give the data-grounded content (from the evidence JSON) without claiming a
visual read. Metrics referenced in captions are tabulated in `../results_data_notes.md`.

Legend convention across core plots: solid blue = numerical (`num`), red dashed/dash-dot
= exact/analytic reference.

---

## Main text

### Fig 1 — Pressure-equilibrium preservation (02_A)
Files: `core_02_A.png`, `pressure_equilibrium_preservation.png`
- `core_02_A.png` **[inspected]**: 2×3 panel (rho, u, p top; abs-error bottom). Density
  is a clean top-hat (~1050 in [0.4,0.6], ~0 outside) with num exactly over exact. The u
  panel sits at 1.0 with only ±1e-13 jitter and p at 1e5 with ~1e-7 spread — i.e. the
  advected air/NASG-water interface holds velocity and pressure equilibrium. Bottom row:
  |Δrho| spikes ~5e-11 only at the two interface cells, |Δu|~1e-13, |Δp|~1e-10. Title
  reports `p_rel=2.183e-15, u_linf=7.683e-14`.
- `pressure_equilibrium_preservation.png`: companion summary of the PE-preservation
  metric (not visually inspected — describe as the PE preservation overview panel).

### Fig 2 — Single-fluid acoustic wave trains
Files: `core_04_B.png` (air, 2000 Hz), `core_05_B.png` (NASG-water, 6000 Hz)
- 04_B: sinusoidal acoustic train on a u=1 flow; ~0.6% amplitude damping
  (`p_amp_ratio=0.9940`), phase `lambda_meas/exact = 0.1745/0.17393`.
- 05_B: stiff-liquid acoustics; `p_amp_ratio=0.9947`, `p_corr=0.99992`.
  (Not visually inspected.)

### Fig 3 — Acoustic reflection/transmission at a material interface (07_B, three impedance pairs)
Files: `core_07_B.png`, `all_speed_interface_acoustic_07_B.png`
- `core_07_B.png` **[inspected]**: 3×3 grid, rows = Air-Water / Helium-Air / Argon-Air,
  columns = rho / u / p-P0. Each row shows a density step at the interface and a
  reflected+transmitted acoustic wave pair. num tracks exact closely in all nine panels.
  Air-Water (hardest, largest Z jump) shows the transmitted pressure peak (~16 near
  x≈1.15) slightly clipped with faint base ringing near x≈1.25–1.3; Helium-Air and
  Argon-Air transmitted/reflected peaks (p≈4.6 / 9.3) match almost exactly. Quantifies
  the §1 07_B table (Air-Water L2p=0.090, corr_p=0.986; residual ringing flag §9a).
- `all_speed_interface_acoustic_07_B.png`: the Air-Water subcase at N=400 as the
  interface-acoustic anchor of the all-speed sweep (same numbers as core Air-Water).

### Fig 4 — CFL robustness (07 Air-Water, N=200, CFL 0.2/0.4/0.6)
Files: `cfl_07_AirWater_CFL0p2.png`, `cfl_07_AirWater_CFL0p4.png`,
`cfl_07_AirWater_CFL0p6.png`, `acoustic_cfl_sweep.png`
- Demonstrates CFL-independent accuracy: L2p≈0.143–0.153, corr_p≈0.958–0.965 as CFL
  triples (§3). The subcase FAIL is the N=200 peak-amplitude limit, not instability.
  `acoustic_cfl_sweep.png` is the summary overlay across the three CFLs.
  (Not visually inspected.)

### Fig 5 — Two-phase shock tubes
Files: `core_13_E.png`, `core_14_E.png`, `core_15_E.png`
- `core_13_E.png` **[inspected]**: 5 panels (rho, u, p, Mach M, acoustic impedance Z)
  over x∈[0,2] m. HP-air/LP-water tube: left rarefaction from rho≈11600, a rho plateau
  ≈6300, contact drop at x≈0.63 to the water plateau ≈1000, and a shock at x≈1.85. u
  rises to a ~200 m/s plateau; p falls 1e9→~0.42e9→1e4; M peaks ≈0.65; Z steps
  4.05→~1.9→2.63 with a small num undershoot at the shock. num overlays the
  ideal/NASG Riemann exact across all five panels.
- `core_14_E.png`: HP-water/LP-air tube, close shock+contact pair split
  (`split_gap_ratio=1.005`), density-peak 735 vs 736. (Not visually inspected.)
- `core_15_E.png`: air/water cavitation double-rarefaction; α grows to 0.977, near-vacuum
  p_min=3.07 Pa stays positive (CFL=0.01 limitation, §9e). (Not visually inspected.)

### Fig 6 — Thermal / temperature transport (material-interface advection)
Files: `core_16_T.png`, `core_17_T.png`, `core_18_T.png`
- `core_16_T.png` **[inspected]**: 3×3 panels (α₁, rho, u / p, T_mixture, |ΔT| /
  |Δp|, |Δu|, |Δrho|). α₁ and rho are sharp top-hat blocks in [0.35,0.65]; T_mixture is
  the inverted block (1200 gas → 300 liquid). u sits at 10 and p at 1e5+~3.5e-8, so
  mechanical equilibrium is held. Error panels: |ΔT| and |Δrho| peak (~8e-4 / ~1e-3)
  only at the interface cell; |Δp|~3.5e-8, |Δu|~1.2e-10. Title `p_rel=3.77e-13`.
- `core_17_T.png`: smooth Gaussian-α hot-gas advection, alpha peak retention 0.995.
- `core_18_T.png`: smooth-α mixture thermal-wave advection, `Tmix_linf_ratio=0.00179`,
  rho-peak ratio 0.9996. (17/18 not visually inspected.)

### Fig 7 — Hypersonic (Mach-10) cases
Files: `core_24_H.png`, `core_25_H.png`, `all_speed_hypersonic_25_H.png`
- `core_24_H.png` **[inspected]**: 5 rows (psi_water = 0/0.25/0.5/0.75/1) × 3 columns
  (rho, u, p). Each is a single Mach-10 mixture shock sitting exactly at x≈0.80 with a
  flat post-shock plateau; num overlays the Kapila/Wood path-conservative RH exact.
  Post-shock pressure ranges 1.17e7 (pure air, top) to 7.55e10 (pure water, bottom) —
  ~4 decades — with only a small rho dip just upstream of the shock in the mixed rows,
  no post-shock oscillation.
- `core_25_H.png`: Mach-10 air shock → water interface (reflected/transmitted shocks);
  shock Δcells=0.54, but ~10% pressure checkerboard (`p_osc=0.105`, §9e).
- `all_speed_hypersonic_25_H.png`: the N=200 all-speed hypersonic anchor.
  (25_H panels not visually inspected.)

### Fig 8 — All-speed span (low-Mach anchors)
Files: `all_speed_ultra_low_mach_03_B.png`, `all_speed_low_mach_air_04_B.png`
- 03_B: Δp/p≈1e-5 acoustic pulse (SG water), `osc≈7.7e-8` — extreme low-Mach anchor.
- 04_B (N=200): low-Mach air acoustics, `p_corr=0.9983`. Together with Fig 7 these
  bracket M~1e-5 → Mach-10 in one scheme. (Not visually inspected.)

### Fig 9 — Grid-refinement convergence
File: `grid_refinement_errors.png` **[inspected]**
- log-log plot, representative normalized error vs N (2^7–2^9), six curves. Reading top
  to bottom at the right: 25_H (brown) is highest and **non-monotone** (dips at N=400,
  rises at N=800); 07_B (blue) descends 0.32→0.09; 24_H (purple) is nearly flat
  ~0.03→0.022 (shock-limited, low order ~0.26); 14_E (green) is the steepest, plunging
  from ~0.88 at N=200 through ~0.013 to ~0.003 (crossing its resolution threshold);
  13_E (orange) 0.007→0.0028; 18_T (red) lowest, 8.9e-4→7e-5. Matches §2 orders.

### Fig 10 — Scheme ablation
Files: `ablation_pass_heatmap.png`, `baseline_ablation_metrics.png`
- `ablation_pass_heatmap.png` **[inspected]**: 4 rows (02_A, 07_B, 13_E, 18_T) × 8
  variant columns (alpha_cicsam, alpha_mstacs, hllc_flux, production, superbee_only,
  tmlpu_minmod, tmlpu_vanleer, upwind_primitive), green=PASS/red=FAIL. 02_A is all
  green; 07_B is all red (coarse-N peak-amplitude limit); 13_E green except hllc_flux &
  upwind_primitive; 18_T all red except upwind_primitive. Exactly the §5 matrix.
- `baseline_ablation_metrics.png`: companion per-variant key-metric bar/plot (the
  numeric values behind the heatmap; hllc_flux blow-ups, §5). (Not visually inspected.)

---

## Supplementary / appendix

### Fig S1 — Per-variant profile panels (ablation detail, 32 files)
`{production, superbee_only, tmlpu_minmod, tmlpu_vanleer, upwind_primitive, hllc_flux,
alpha_cicsam, alpha_mstacs}_{02_A, 07_B, 13_E, 18_T}.png`
- One profile plot per (variant × case) cell of the §5 matrix, at the fixed ablation
  resolutions (02_A N=100, 07_B N=200, 13_E N=400, 18_T N=400). Supports the heatmap of
  Fig 10; `hllc_flux_07_B.png` / `hllc_flux_18_T.png` show the blow-ups.
  (Not visually inspected.)

### Fig S2 — Per-resolution grid panels (grid-study detail, 18 files)
`grid_{07_B_N100,07_B_N200,07_B_N400, 13_E_N200,13_E_N400,13_E_N800,
14_E_N200,14_E_N400,14_E_N800, 18_T_N200,18_T_N400,18_T_N550,
24_H_N100,24_H_N200,24_H_N400, 25_H_N200,25_H_N400,25_H_N800}.png`
- Individual solution profiles at each (case × N) point summarized in Fig 9 / §2.
  (Not visually inspected.)

---

## Full file inventory (75 PNGs)
- Core cases (13): `core_01_A, core_02_A, core_04_B, core_05_B, core_07_B, core_13_E,
  core_14_E, core_15_E, core_16_T, core_17_T, core_18_T, core_24_H, core_25_H`
- All-speed (4): `all_speed_{ultra_low_mach_03_B, low_mach_air_04_B,
  interface_acoustic_07_B, hypersonic_25_H}`
- CFL (4): `cfl_07_AirWater_CFL0p2/0p4/0p6`, `acoustic_cfl_sweep`
- Grid (19): `grid_refinement_errors` + 18 `grid_<case>_N<res>`
- Ablation (34): `ablation_pass_heatmap`, `baseline_ablation_metrics` + 32 per-variant
- Extra (1): `pressure_equilibrium_preservation`
