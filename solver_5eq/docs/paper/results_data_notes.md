# Results data notes — 5-equation all-speed IMEX solver (1D Euler evidence)

Structured quantitative notes for the Results section. All numbers copied from
`results/1D/paper_euler_evidence/paper_euler_evidence.json` (evidence, regenerated
`2026-07-14T05:59:58+0900` with the production configuration including fifth-order WENO
reconstruction of the acoustic faces, top-level `pass_count=41 fail_count=29 total=70`)
and from the acceptance-verifier sources under `.codex-loop/`. EOS parameters and
run configurations are quoted with `file:line`. Values are given to 3–4 significant
figures except machine-precision quantities, which are kept verbatim.

**Production configuration** (BASE_ENV, `results/1D/paper_euler_evidence.py:39-53`),
applied to every run unless a variant/sweep overrides it:
`imex_ssp3` time integrator, `adaptive_bvd` alpha scheme, `tmlpu` primitive
reconstruction with `superbee` TVD limiter, `slau2` material flux, fifth-order WENO
reconstruction of the acoustic (p, u) faces on material-clean stencils (Section 3.6),
`regime_auto` pressure closure, `CHARACTERISTIC_RECON=1`, `RUSANOV_FALLBACK=0`,
`UNIFORM_PERIODIC_REMAP=0`, and the core override `FIVE_EQ_CASE24_N=400`.

> Regeneration (2026-07-14, weno5 acoustic faces): 02_A `p_rel_linf=5.821e-16`,
> `u_abs_linf=5.329e-14`; 07_B Air-Water `L2p=6.026e-2 Lip=2.331e-1 corr_p=0.993`;
> Helium-Air `L2p=1.059e-2` PASS; Argon-Air `L2p=2.412e-3` PASS. These are the
> evidence-JSON values used throughout (see §9d).

---

## 1. Per-case core results (13 core entries, all PASS)

Core resolutions actually used in the evidence run: 01_A/02_A/16_T `N=100`,
04_B `N=500`, 05_B/07_B/15_E/24_H/25_H `N=400`, 13_E/14_E `N=800`, 17_T/18_T `N=550`.

### 01_A — PE static interface (drift test), N=100, dt=0.01, t_end=1.0, PASS
- `p_rel = 4.366e-14` (pressure drift, rel. to P0=1e5)
- `u_abs = 1.042e-12` (velocity drift from rest)
- `osc = 2.609e-16` (pressure checkerboard)
- `p_sharp_overshoot = 4.293e-14`; steps=100
- Interpretation: a static (u=0) material interface holds mechanical equilibrium to
  machine precision over 100 steps.

### 02_A — PE advected interface (NASG), N=100, dt=0.01, t_end=1.0, PASS
- `p_rel_linf = 5.821e-16`, `u_abs_linf = 5.329e-14` (evidence JSON, 2026-07-14 regeneration)
- `alpha_range_ratio = 1.000`, `rho_range_ratio = 1.0000000000016`
- `corr_alpha = 1.0000000000000002`, `corr_rho = 0.9999999999999998`
- `alpha_l1_ratio = 3.505e-15`; steps=100
- Interpretation: an air/NASG-water interface advected at u=1 preserves pressure and
  velocity equilibrium to machine precision (the strictest single result in the set).

### 04_B — single-fluid acoustic sinusoid, air 2000 Hz, N=500, CFL=0.4, t_end=2.3e-3, PASS
- `p_scaled_l2 = 0.03366` (amplitude-rescaled profile L2); `p_amp_ratio = 0.9940`
- `p_corr = 0.9988`
- Phase: `lambda_meas = 0.1745` vs `lambda_exact = 0.17393` (wavelength, +0.33%)
- `dp_meas = 4.011` vs `dp_exact = 4.025` (wave pressure amplitude); steps=1003
- Interpretation: injected acoustic wave train advected on a u=1 flow; ~0.6% amplitude
  damping, phase error <0.5%.

### 05_B — single-fluid acoustic sinusoid, water(NASG) 6000 Hz, N=400, CFL=0.4, t_end=5.10e-4, PASS
- `p_scaled_l2 = 0.008806`; `p_amp_ratio = 0.9947`; `p_corr = 0.99992`
- Phase: `lambda_meas = 0.2600` vs `lambda_exact = 0.26122` (-0.47%)
- `dp_meas = 15562` vs `dp_exact = 15642` Pa; steps=794
- Interpretation: stiff-liquid acoustics resolved with ~0.5% amplitude/phase error.

### 07_B — acoustic reflection/transmission at material interface, N=400, CFL=0.4, PASS (0 failures)
Linear d'Alembert exact solution; three subcases (metric key = L2 error normalized by
wave amplitude). Peak-location deltas are in cells (tolerance 3 cells).

| subcase | L2p | Lip | L2u | Liu | corr_p | frac_p | p-peak Δcells | p amp ratio | u amp ratio | p-sym err | steps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Air-Water  | 0.06026 | 0.2331 | 0.02210 | 0.1783 | 0.9933 | 1.000 | 1 (u:1) | 0.9969 | 0.9803 | 0.05534 | 1620 |
| Helium-Air | 0.01059 | 0.06596 | 0.005599 | 0.02719 | 0.9988 | 1.000 | 0 | 0.9745 | 0.9743 | 0.03765 | 1017 |
| Argon-Air  | 0.002412 | 0.01188 | 0.002970 | 0.01583 | 0.9999 | 1.000 | 0 | 0.9920 | 0.9918 | 0.01618 | 469 |

- Symmetry limit 0.38; all three well inside. All peak-amplitude ratios inside the
  0.80–1.13 gas band and the 0.85–1.10 liquid band.
- **Air-Water residual ringing now within the tightened guard:** `p_smooth_local_tv_excess = 0.204`.
  With the fifth-order WENO acoustic faces this is below both the evidence-era guard
  `HF_SMOOTH_LOCAL_TV_EXCESS_LIMIT_07 = 0.80` (`.codex-loop/verify_02_07_acceptance.py:76`)
  and the later-tightened internal guard of 0.30 (the second-order TVD acoustic faces of
  the earlier scheme gave 0.537). Helium-Air `p_smooth_local_tv_excess = 0.05714`,
  Argon-Air `0.01177` are cleaner still.

### 13_E — HP-air / LP-water shock tube (Denner), N=800, CFL=0.30, t_end=6.7e-4, L=2.0, PASS
- `p_osc = 0.003124`, `rho_osc = 0.004761` (checkerboard)
- Contact at `iface_x = 0.6338`; `contact_rho_peak_value = 6217` vs exact_hi 6298
  (`overshoot_ratio = 0.01598`, limit 0.05)
- Shock: `u_shock_x_num = 1.845` vs exact 1.84992 → `Δcells = 1.970` (limit 3)
- Smooth-region L2 (rel.): `p = 0.004066`, `rho = 0.003002`, `u = 0.008479`
- `shock_p_overshoot_ratio = 0.01289`; `p_max = 1.000e9`, `p_min = 1.000e4`; steps=2189

### 14_E — HP-water / LP-air shock tube (Yoo-Sung), N=800, CFL=0.25, t_end=2.29e-4, L=1.0, PASS
- `p_osc = 0.0002489`, `rho_osc = 0.007598`
- Contact: `x_num = 0.83125` vs exact 0.83228 → `Δcells = 0.8220`
- Shock: `x_num = 0.85875` vs exact 0.85965 → `Δcells = 0.7222`
- Density peak `736.5` vs exact_hi `736.1` (`overshoot_ratio = 0.00067`, limit 0.001); plateau
  `linf_ratio = 0.002436` (limit 0.03)
- `split_gap_ratio = 1.005` (band 0.5–1.8); `reflected_pressure_ratio = 1.000`; steps=1790

### 15_E — air/water cavitation (double rarefaction), N=400, CFL=0.01, t_end=9.5e-4, PASS
- `alpha_peak = 0.9771` (gas fraction growth from seed 0.055)
- `rho_min = 22.93`, `p_min = 3.067` Pa (near-vacuum, stays positive)
- `u_min = -100.0`, `u_max = +100.0` (symmetric pull-apart)
- `p_osc = 0.0006779`, `rho_osc = 0.002824`
- Center velocity jump `2.666` vs ref `2.283` (smooth-ok); steps=5772
- **HONESTY FLAG:** CFL reduced to 0.01 (nominal 0.25 is unstable for the Kapila
  `D1·div(u)` source at the initial velocity jump) — documented numerical limitation
  (`.codex-loop/verify_08_26_acceptance.py:2719-2723`).

### 16_T — hot-gas/cold-liquid material-interface advection, N=100, dt=5e-4 (Co=0.5), t_end=0.1, PASS
- `p_rel_linf = 6.112e-14`, `u_abs_linf ≈ 4.90e-13`
- `T1_l1_ratio = 5.087e-13`, `T1_linf_ratio = 2.867e-12`
- `T2_l1_ratio = 3.936e-13`, `T2_linf_ratio = 7.768e-12`
- `Tmix_l1_ratio = 1.946e-8`, `Tmix_linf_ratio = 9.681e-7`
- `alpha_range_ratio = 1.000` (peak retention), `alpha_corr = 0.99999999999997`
- `p_checkerboard = 2.187e-14`, `rho_checkerboard = 0.1010` (sharp-interface localized)
- Active-phase T error ~1e-11; mixture-T error localized to the sharp contact.

### 17_T — smooth-alpha Gaussian hot-gas advection, N=550, dt=1e-4 (Co=0.55), t_end=0.1, PASS
- `p_rel_linf = 4.075e-15`, `u_abs_linf = 1.990e-13`
- `T1_linf_ratio = 5.874e-15`, `T2_linf_ratio = 6.196e-14`
- `Tmix_linf_ratio = 0.004350`, `Tmix_l1_ratio = 1.635e-4`
- `alpha_range_ratio = 0.9951` (alpha peak retention), `case17_peak_ok = True`
- `alpha_smooth_tv_excess_max = 2.426e-4`; `rho_checkerboard = 8.685e-5`

### 18_T — smooth-alpha mixture thermal-wave advection, N=550, dt=1/11000 (Co=0.5), t_end=0.1, PASS
- `p_rel_linf = 4.366e-16`, `u_abs_linf = 1.776e-15`
- `T1_linf_ratio = 2.871e-4`, `T2_linf_ratio = 0.001193`
- `Tmix_linf_ratio = 0.001790`, `Tmix_l1_ratio = 4.274e-4`
- `alpha_range_ratio = 0.9986`; `rho_l1_ratio = 6.981e-5`
- `case18_rho_peak_amp_ratio = 0.9996` (band 0.98–1.02)
- `T_active_hf_max`: T1 `2.809e-4`, T2 `5.747e-4` (limit 8e-4); `rho_checkerboard = 2.839e-5`

### 24_H — Mach-10 homogeneous-mixture shock (Kapila/Wood RH), N=400, CFL=0.10, PASS (5 psi subcases)
Single shock at `x≈0.80` (exact 0.80). Shock-location tolerance 3 cells;
`rho_profile_l2` limit 0.03; `rho_corr` limit 0.99.

| psi_water | p_profile_l2 | rho_profile_l2 | u_profile_l2 | p_corr | rho_corr | p_osc | shock_x | shock Δcells |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.02371 | 0.02172 | 0.03680 | 0.9982 | 0.9978 | 0.01400 | 0.7988 | 0.50 |
| 0.25 | 0.03015 | 0.01985 | 0.03738 | 0.9971 | 0.9963 | 0.01562 | 0.8013 | 0.50 |
| 0.50 | 0.04299 | 0.01833 | 0.04772 | 0.9942 | 0.9924 | 0.01598 | 0.8013 | 0.50 |
| 0.75 | 0.03654 | 0.008273 | 0.04115 | 0.9958 | 0.9935 | 0.01539 | 0.8013 | 0.50 |
| 1.00 | 0.04526 | 0.009474 | 0.03435 | 0.9936 | 0.9971 | 0.02515 | 0.7963 | 1.50 |
- Post-shock pressure spans `1.17e7` (air) to `7.55e10` (water) — 4 pressure decades
  handled without oscillation. `admissible/finite = True` all subcases.

### 25_H — Mach-10 air shock hitting water interface (Denner §7.4.4), N=400, CFL=0.30, PASS
- `p_scaled_l2 = 0.04811`, `rho_scaled_l2 = 0.004469`, `u_scaled_l2 = 0.02400`
- `p_corr = 0.9969`, `rho_corr = 0.99999`, `u_corr = 0.9990`
- Shock `Δcells = 0.5433`; interface `Δcells = 0.4528`;
  reflected shock `Δcells = 0.7074`; transmitted shock `Δcells = 0.5433`
- `p_osc = 0.09952` (≈10% pressure checkerboard, the noisiest core case),
  `interface_instability = 0.1071`, `interface_p_linf = 0.002189`,
  `interface_rho_overshoot = 0.0005773`; admissible=True.

---

## 2. Grid refinement (18 entries → 6 cases × 3 resolutions)

Observed order computed as `p = ln(err_coarse/err_fine) / ln(N_fine/N_coarse)`.
**Primary metric = the "representative normalized error" plotted in
`grid_refinement_errors.png`** (`_representative_metric`,
`results/1D/paper_euler_evidence.py:292-317`). A physically-meaningful alternate
metric is listed for context. PASS/FAIL is the full acceptance verdict at that N.

### 07_B — metric = L2p (Air-Water), interface-acoustic error
| N | L2p | PASS | order (prev→this) |
|---|---|---|---|
| 100 | 0.3090 | FAIL | — |
| 200 | 0.1512 | FAIL | 1.031 |
| 400 | 0.06026 | PASS | 1.328 |
(Lip: 1.100 → 0.5514 → 0.2331.) Converges monotonically; PASS at N=400.

### 13_E — metric = case13_rho_smooth_l2_rel (smooth-region density L2, rel.)
| N | rho_smooth_l2_rel | PASS | order |
|---|---|---|---|
| 200 | 0.007635 | FAIL | — |
| 400 | 0.004878 | PASS | 0.646 |
| 800 | 0.003002 | PASS | 0.700 |
(Alternate p_smooth_l2_rel: 0.01085 → 0.006684 → 0.004066, order ~0.70/0.72.)
Shock-tube smooth error ~0.7 order (shock+contact limit the rate).

### 14_E — metric = case14_rho_plateau085_089_linf_ratio (density-plateau Linf ratio)
| N | rho_plateau_linf_ratio | PASS | order |
|---|---|---|---|
| 200 | 0.8946 | FAIL | — |
| 400 | 0.01666 | FAIL | 5.747 |
| 800 | 0.002436 | PASS | 2.774 |
The N=200 plateau is essentially unresolved (ratio ~0.89); by N=400 the metric collapses
~1.7 decades but the case still FAILs there because the physical two-jump density split
(`case14_rho_peak085_ok`, `two_jump_split=False`) is not yet resolved (the plateau
`linf_ratio=0.01666` is itself already below its 0.025 limit); PASS only at N=800. The
large apparent order (5.7) reflects crossing the resolution threshold, not a true
high-order scheme. (Alternate physical shock-position error `shock_Δcells·dx`: order ≈1.1–1.3.)

### 18_T — metric = rho_l1_ratio (mixture-density L1 error ratio)
| N | rho_l1_ratio | PASS | order |
|---|---|---|---|
| 200 | 8.866e-4 | FAIL | — |
| 400 | 2.531e-4 | FAIL | 1.809 |
| 550 | 6.981e-5 | PASS | 4.045 |
(Alternate T2_linf_ratio: 0.01471 → 0.004468 → 0.001193, order 1.72 then 4.15.)
~1.8 order 200→400; the 400→550 order (~4) is inflated by the short N-interval near the
guard floor — treat as "well below tolerance", not a literal 4th-order claim.

### 24_H — metric = max rho_profile_l2 over the 5 psi subcases
| N | max rho_profile_l2 | PASS | order |
|---|---|---|---|
| 100 | 0.03102 | FAIL | — |
| 200 | 0.02593 | FAIL | 0.259 |
| 400 | 0.02172 | PASS | 0.255 |
Sub-first-order (~0.26) — expected for an L2 error that includes the Mach-10 shock
discontinuity. Convergence is monotone; PASS only at N=400.

### 25_H — metric = p_scaled_l2 (pressure profile L2, amplitude-rescaled)
| N | p_scaled_l2 | PASS | order |
|---|---|---|---|
| 200 | 0.1001 | PASS | — |
| 400 | 0.04811 | PASS | 1.057 |
| 800 | 0.03988 | PASS | 0.271 |
- **Pressure metric now MONOTONE under weno5:** p_scaled_l2 decreases 200→400→800
  (N=800 is best; both observed orders positive). The scaled density and velocity
  metrics remain non-monotone: rho_scaled_l2 (0.06244 → 0.004469 → 0.03096) and
  u_scaled_l2 (0.06869 → 0.02400 → 0.03519) are each smallest at N=400. All three N
  PASS acceptance. Residual non-monotonicity in rho/u tracks interface-interaction
  phasing / checkerboard (`p_osc≈0.10`) rather than smooth truncation error.

---

## 3. CFL sweep (07 Air-Water, N=200, CFL 0.2/0.4/0.6)

| CFL | steps | L2p | Lip | L2u | corr_p | frac_p | p amp ratio | u amp ratio | p-sym err | subcase PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 1620 | 0.1506 | 0.5363 | 0.04205 | 0.9716 | 0.935 | 0.7317 | 0.6848 | 0.04205 | FAIL |
| 0.4 | 810  | 0.1512 | 0.5514 | 0.04210 | 0.9712 | 0.935 | 0.7242 | 0.6845 | 0.04347 | FAIL |
| 0.6 | 540  | 0.1576 | 0.5920 | 0.04254 | 0.9680 | 0.925 | 0.7068 | 0.6816 | 0.08701 | FAIL |

- **Interpretation:** accuracy is essentially CFL-independent: L2p, corr_p and the
  amplitude ratios move by <5% as CFL triples (CFL=0.6 only slightly worse). The FAIL is
  a **resolution** artifact: at N=200 the transmitted peak amplitude is damped to ~0.71–0.73,
  below the 0.85 acceptance band (`peak_amplitude_ok=False`). Symmetry (0.042–0.087) and
  correlation stay strong. So the sweep evidences CFL robustness of the scheme; the
  peak-amplitude bar is only met at N=400 (see §2 07_B). No instability or blow-up at
  any CFL.

---

## 4. All-speed regimes (4 entries, all PASS)

| label | case | config | key metrics |
|---|---|---|---|
| ultra_low_mach | 03_B | N=200, SG-water pulse Δp=1 Pa on 1e5 (M~1e-5) | `p_l2 = 0.02795`, `dp_max = 0.5007` Pa, `osc = 7.71e-8`, steps=215 |
| low_mach_air | 04_B | N=200 | `p_scaled_l2 = 0.03945`, `p_amp_ratio = 0.9833`, `p_corr = 0.9983`, `lambda_meas = 0.17375` vs `0.17393` |
| interface_acoustic | 07_B | N=400 Air-Water | `L2p = 0.06026`, `Lip = 0.2331`, `corr_p = 0.9933` (identical to core 07_B Air-Water) |
| hypersonic | 25_H | N=200 | `p_scaled_l2 = 0.1001`, `rho_scaled_l2 = 0.06244`, `p_corr = 0.9866`, shock `Δcells = 0.5216`, iface `Δcells = 0.02361` |

- 03_B is the extreme low-Mach anchor (Δp/p ≈ 1e-5, `osc` at ~1e-7). Spans from
  M~1e-5 (03_B) to Mach-10 (25_H) within the same production scheme.

---

## 5. Ablation / baseline matrix (32 entries = 8 variants × 4 cases) — Table 3

Variant env deltas (`results/1D/paper_euler_evidence.py:87-102`) over BASE_ENV:
- `production` = BASE_ENV (tmlpu+superbee, adaptive_bvd, slau2)
- `upwind_primitive` = primitive→upwind
- `superbee_only` = primitive→superbee (no THINC/BVD blend)
- `tmlpu_vanleer` = tmlpu + TVD→vanleer
- `tmlpu_minmod` = tmlpu + TVD→minmod
- `alpha_cicsam` = alpha scheme→cicsam
- `alpha_mstacs` = alpha scheme→mstacs
- `hllc_flux` = material flux→hllc_split

**Ablation resolutions are fixed and COARSER than the core/converged runs**
(`BASELINE_TARGETS`, lines 104-113): 02_A `N=100`, 07_B `N=200` (Air-Water only),
13_E `N=400`, 18_T `N=400`. This is a same-N stress test to separate schemes — it is
NOT run at the converged resolution.

PASS/FAIL with the per-cell key metric (02_A: `p_rel_linf`; 07_B: Air-Water `L2p`;
13_E: `case13_p_smooth_l2_rel`; 18_T: `T2_linf_ratio`):

| variant | 02_A (N100) | 07_B (N200) | 13_E (N400) | 18_T (N400) | pass/4 |
|---|---|---|---|---|---|
| production | PASS 5.82e-16 | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004468 | 2 |
| upwind_primitive | PASS 1.89e-15 | FAIL 0.4269 | **FAIL 0.01888** | **PASS 0.01872** | 2 |
| superbee_only | PASS 5.82e-16 | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004468 | 2 |
| tmlpu_vanleer | PASS 2.47e-15 | FAIL 0.1688 | PASS 0.007252 | FAIL 0.004461 | 2 |
| tmlpu_minmod | PASS 1.02e-15 | FAIL 0.1753 | PASS 0.008551 | FAIL 0.004466 | 2 |
| alpha_cicsam | PASS 5.82e-16 | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004626 | 2 |
| alpha_mstacs | PASS 1.60e-15 | FAIL 0.1512 | PASS 0.006684 | **FAIL nan** | 2 |
| hllc_flux | PASS 4.37e-16 | **FAIL 1.20e39** | **FAIL 0.1303** | **FAIL 52543** | 1 |

Per-case PASS counts (out of 8 variants): 02_A **8/8**, 07_B **0/8**, 13_E **6/8**, 18_T **1/8**.

**Reading the matrix (important for honesty):**
- **02_A robust for every variant** — machine-precision PE preservation is scheme-agnostic.
- **07_B fails for all 8 variants at N=200** — including production; this is the coarse-N
  peak-amplitude limit (§3), not a per-scheme defect. Metric *ordering* is the signal:
  production/superbee/cicsam/mstacs share `L2p=0.1512` (best), tmlpu_vanleer 0.1688,
  tmlpu_minmod 0.1753, upwind_primitive 0.4269 (worst dispersive), hllc_flux **blows up**
  (`L2p=1.20e39`).
- **13_E:** all pass except `upwind_primitive` (too diffusive, 0.01888) and `hllc_flux`
  (near blow-up, 0.1303).
- **18_T:** only `upwind_primitive` passes — its extra diffusion satisfies the wiggle
  guard at N=400 while sharper schemes (production/tmlpu/superbee/alpha_cicsam) trip
  `case18_wiggle_ok` at 0.00446–0.00463; `alpha_mstacs` returns `nan` (terminated at
  798 steps); `hllc_flux` blows up (52543).
- **hllc_flux is the clear worst variant** (1/4; catastrophic on 07_B, 13_E, 18_T),
  motivating the SLAU2 production choice. The TMLPU+superbee production limiter gives the
  best 07_B accuracy among stable variants.

---

## 6. EOS parameters actually used (exact, with source)

All EOS built via `solver/five_eq_IMEX/eos_facade.py::make_eos`.

### Ideal-gas phases
| gas | gamma | cv (kv, J/kg/K) | used in | source |
|---|---|---|---|---|
| Air | 1.400 | 717.5 | 01,02,03,04(both),05(gas),07,13,14,15,16-18,24,25 | `verify_02_07:108`, `verify_08_26:2459` |
| Air (04_B, both phases) | 1.400 | 717.5 | 04_B | `verify_01_03_06:324-325` |
| Helium | 1.667 | 3120.0 | 07 Helium-Air | `verify_02_07:109` |
| Argon | 1.660 | 312.0 | 07 Argon-Air | `verify_02_07:110` |

(07_B reference densities/speeds used for impedance: Air rho=1.157 c=347.8;
Helium rho=0.164 c=1008.2; Argon rho=1.748 c=308.2 — `verify_02_07:108-110`.)

### NASG water (primary liquid EOS) — used in 02, 05, 07(Air-Water), 13, 14, 15, 24, 25
`make_eos("nasg", gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6)`
- gamma = 1.187
- P∞ (pinf) = 7.028e8 Pa
- cv (kv) = 3610.0 J/kg/K
- b (covolume) = 6.61e-4 m³/kg
- eta (energy offset) = -1.177788e6 J/kg
- reference: rho=998.0, c=1567.335 m/s
- Sources (identical params in all four): `verify_02_07:111-121` and `299-307`,
  `verify_08_26:73-81` (`_make_water_nasg`) and `2460-2466` (case_13 inline),
  `verify_16_19:126-134`, `verify_01_03_06:45-50`.

### SG water (stiffened-gas) — used ONLY in 03_B
`make_eos("sg", gamma=4.4, pinf=6.0e8, kv=474.2)`
- gamma = 4.4, P∞ = 6.0e8 Pa, cv = 474.2 J/kg/K
- Source: `verify_01_03_06:271`. (Confirmed: SG water is NOT used in 13/14/15/24/25 —
  those use NASG water above.)

---

## 7. Per-case run configuration (Table 2 — replaces all [TBD])

Domain is `[0, L]`, cell-centered `x = (i+0.5)·dx`, `dx = L/N`. Production numerics per
BASE_ENV. "core N" is the resolution the evidence core run actually used.

| case | N | L | dt / CFL | t_end | BC (L / R) | initial condition (source) |
|---|---|---|---|---|---|---|
| 01_A | 100 | 1.0 | dt=0.01 | 1.0 | transmissive / transmissive | static interface, u=0, p=1e5, air/NASG-water block (`verify_01_03_06:212-227`) |
| 02_A | 100 | 1.0 | dt=0.01 (CFL 0.5 cap) | 1.0 | periodic / periodic | α₁=1e-3 in [0.4,0.6) else 1-1e-3; u=1.0; p=1e5; T=300; air/NASG-water (`verify_02_07:308-347`) |
| 03_B | 200 | 1.0 | CFL=0.4 | 3.0e-4 | transmissive / transmissive | Δp=1 Pa pulse (|x-0.5|<0.1) on p=1e5; air + SG-water (`verify_01_03_06:270-287`) |
| 04_B | 500 | 1.0 | CFL=0.4 | 2.3e-3 | inlet (sinusoid) / transmissive | air both phases, rho=1.157, u0=1, f=2000 Hz, du=0.01·u0 (`verify_01_03_06:322-374`) |
| 05_B | 400 | 1.0 | CFL=0.4 | 5.10e-4 | inlet (sinusoid) / transmissive | NASG-water phase2 (α=1e-6), rho=998, u0=1, f=6000 Hz (`verify_01_03_06:332-374`) |
| 07_B | 400 | 1.5 | CFL=0.4 | subcase (1.55e-3 / 1.513e-3 / 2.02e-3) | reflective / transmissive | interface + Gaussian u-pulse U_peak=0.02; α_floor=1e-8 (`verify_02_07:805-868`) |
| 13_E | 800 | 2.0 | CFL=0.30 | 6.7e-4 | transmissive / transmissive | p=1e9 (x<0.5) / 1e4; air(L)/NASG-water(R), T=300 (`verify_08_26:2457-2496`) |
| 14_E | 800 | 1.0 | CFL=0.25 | 2.29e-4 | transmissive / transmissive | p=1e9 (x<0.7) / 1e5; water(L,rho=1000)/air(R,rho=50) (`verify_08_26:2567-2596`) |
| 15_E | 400 | 1.0 | CFL=0.01 | 9.5e-4 | transmissive / transmissive | α_air=0.055, p=1e5, u=∓100 about x=0.5, rho_air=1.3/rho_water=1000 (`verify_08_26:1896-1912`) |
| 16_T | 100 | 1.0 | dt=5e-4 (Co=0.5) | 0.1 | periodic / periodic | α block [0.35,0.65); T1=300,T2=1200; u=10; p=1e5 (`verify_16_19:198-207`) |
| 17_T | 550 | 1.0 | dt=1e-4 (Co=0.55) | 0.1 | periodic / periodic | Gaussian α (xc=0.5,σ=0.08); T1=300,T2=1200; u=10 (`verify_16_19:208-213`) |
| 18_T | 550 | 1.0 | dt=1/11000 (Co=0.5) | 0.1 | periodic / periodic | α=0.5+0.25 sin; T1=300+50 sin, T2=1200+600 cos; u=10 (`verify_16_19:214-218`) |
| 24_H | 400 | 1.0 | CFL=0.10 | 0.7/V_s | transmissive / transmissive | Ms=10 mixture shock, x_shock0=0.1; 5 psi_water∈{0,.25,.5,.75,1} (`verify_08_26:3584-3644`) |
| 25_H | 400 | 1.0 | CFL=0.30 | t_hit+2.42e-4 | transmissive / transmissive | Ms=10 air shock (x0=0.25) → water interface (x=0.5); p_post=1.165e7, u_post=2869.3 (`verify_08_26:4014-4055`) |

Notes: 24_H core N=400 is set by BASE_ENV override (`FIVE_EQ_CASE24_N=400`); the verifier
code default is 800. 13_E/14_E code default N=800 (`verify_08_26:2494`, `:2594`) and the
evidence uses it. 15_E and 24_H CFL are intentionally reduced (documented limitations, §9).

---

## 8. Acceptance thresholds per case family

### 02 PE-advection (`.codex-loop/verify_02_07_acceptance.py:43-47`)
`P_TOL = 1e-10`, `U_TOL = 1e-10`, `MIN_RANGE_RATIO = 0.85`, `MIN_CORR = 0.90`,
`MAX_L1_RATIO = 0.20`.

### 07 acoustic reflection/transmission (`verify_02_07:53-87`)
`MAX_L2 = 0.216`, `MAX_LINF = 0.81` (gas), `MAX_LINF_AIR_WATER = 0.756`,
`MIN_CORR = 0.88`, `MIN_FRAC = 0.76`, `MAX_L1 = 0.648`, peak-amplitude band
0.85–1.10 (Air-Water/liquid) and 0.80–1.13 (gas), wave-symmetry limit 0.38,
HF smooth-local TV-excess limit 0.80, HF sharp TV-excess limit 1.10, peak-location
tol 3 cells.

### 13/14/24 shock tubes (`verify_08_26_acceptance.py:44-53` and inline)
`CASE24_PROFILE_CORR_MIN = 0.91`, `CASE24_RHO_PROFILE_CORR_MIN = 0.99`,
rho_profile_l2 limit 0.03, rho_corr limit 0.99; shock-location tol 3 cells;
13_E contact_rho_peak overshoot_limit 0.05, p_smooth_hf_limit 0.015,
max_reflected_pressure_ratio 1.15; 14_E rho_peak085 overshoot_limit 0.001 /
tv_excess_limit 0.06, plateau linf_limit 0.03, split_gap_ratio band 0.5–1.8,
max_reflected_pressure_ratio 1.1.

### 16/17/18 temperature transport (`verify_16_19_temperature.py:44-79`)
`T_LINF_TOL = 0.25`, `T_MEAN_TOL = 0.05`, `ACTIVE_PHASE_TOL = 0.05`,
`T_MIX_TRANSPORT_L1_TOL = 0.05`, `T_MIX_TRANSPORT_LINF_TOL = 0.25`,
`T_ACTIVE_HF_TOL = 0.01`, `CASE16_UPWIND_CONTACT_TOL = 0.01`,
`CASE17_EXTREMA_ERROR_TOL = 0.08`, `CASE17_RANGE_RATIO_MIN = 0.90`,
`RHO_PEAK_RATIO_MIN = 0.98`, case18 `T_active_hf_max_limit = 8e-4`,
`T_active_tv_excess_max_limit = 2e-3`.

### 04/05 single-fluid acoustics (`verify_01_03_06_acceptance.py:40, 330-343`)
`CASE05_PEAK_RATIO_MIN = 0.98`, amp_floor_ratio 0.10, oscillation limit 1e-3 (04) /
5e-2 (05).

---

## 9. Honesty flags (verbatim, must appear in the paper)

**(a) 07_B Air-Water residual ringing now within the tightened guard.**
Air-Water `p_smooth_local_tv_excess = 0.204`. It PASSES both the evidence-era guard
(`HF_SMOOTH_LOCAL_TV_EXCESS_LIMIT_07 = 0.80`, `verify_02_07:76`) and the later-tightened
internal guard of 0.30. The fifth-order WENO acoustic reconstruction (Section 3.6) removes
the interface-adjacent ringing that the earlier second-order TVD acoustic faces left at
the transmitted-peak base (guard value 0.537 then). Helium-Air (0.05714) and Argon-Air
(0.01177) are cleaner still. This flag is now resolved rather than a shortfall.

**(b) 13_E / 14_E core resolution.** Both core runs use `N=800` (verifier default,
`verify_08_26:2494` and `:2594`), higher than most other cases; report the N.

**(c) Water EOS family per case (as actually run).** NASG water
(γ=1.187, P∞=7.028e8, cv=3610, b=6.61e-4, η=-1.177788e6) is used in 02, 05,
07(Air-Water), 13, 14, 15, 24, 25. Stiffened-gas (SG) water (γ=4.4, P∞=6e8, cv=474.2)
is used ONLY in 03_B. Do not conflate the two.

**(d) Provenance / reproduction.** Evidence JSON regenerated `2026-07-14T05:59:58+0900`
(`pass_count=41, fail_count=29, total=70`) with the production configuration including
fifth-order WENO acoustic reconstruction. The 70 tally counts every grid/CFL/baseline/
all-speed run (many are intentional coarse-N or worst-variant FAILs), NOT the 13 physics
cases; all 13 core cases PASS.

**(e) Additional documented numerical limitations.**
- 15_E CFL reduced to 0.01 (nominal 0.25 unstable for the Kapila `D1·div(u)` source at
  the velocity jump) — `verify_08_26:2719-2723`.
- 24_H CFL reduced to 0.10 (hypersonic mixture shock sensitive to source/flux
  time-centering) — `verify_08_26:3640-3643`.
- 25_H now grid-converges MONOTONICALLY in the scaled-L2 pressure metric under weno5
  (N=800 best, orders +1.06/+0.27; §2); the scaled density and velocity metrics remain
  non-monotone (each smallest at N=400). Carries ~10% pressure checkerboard
  (`p_osc=0.09952`); all three N PASS.
- 14_E grid: N=400 now FAILs (was PASS with the earlier scheme). The plateau
  `linf_ratio=0.01666` is below its 0.025 limit, but `case14_rho_peak085_ok` fails there
  (`two_jump_split=False`, the physical two-jump density split unresolved at N=400);
  PASS only at N=800. Core 14_E uses N=800 so this does not affect the core verdict (§9b).
- Ablation matrix (§5) is a fixed coarse-N stress test; `production` FAILs 07_B (N=200)
  and 18_T (N=400) there purely from resolution, while the converged core 07_B (N=400)
  and 18_T (N=550) both PASS.
