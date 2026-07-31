# Denner Faithfulness Audit — EOS / IC / BC / Reference (iter231)

Audit of the C++ solver vs the Denner ACID paper (Table 1, §7 cases). Cross-checked
`eos.cpp`, `cases.cpp`, `validation.cpp`, the paper PDF, and `denner_acid_faithful.md`.
Denner cases = 01,02,04,05,07,13,24,25. (14,15 are project extensions, not audited.)

## >>> ALGORITHM faithfulness (iter241) — THE missing piece <<<
This audit originally only covered EOS/IC/BC/reference and ASSUMED the algorithm was
faithful. It was NOT. The biggest discrepancy is the energy coupling:

| aspect | Denner (faithful spec §5, Eq.28) | our `acid.cpp` (default 2×2) | impact |
|---|---|---|---|
| unknowns/cell | **3: (u, p, h)** | 2: (u, p) | — |
| system | **3N×3N block-tridiag**, block-Thomas | 2N×2N | — |
| energy | **inside the coupled Newton** (h solved with u,p) | **SEGREGATED** (T updated outside the Newton) | weak cases OK, **strong shocks (24,25) FAIL** |

Why this is exactly why "we did it like Denner but it fails": for weak/acoustic cases
(01,02,04,05,13) the temperature barely moves, so segregating the energy is harmless and
they PASS. For the strong-shock cases (24,25) the shock heating couples tightly to p and
u; the segregated static enthalpy Hstat = rhoH − 0.5·rho·u² goes negative during the
inner iteration (conservative rhoH uses rho_o, the kinetic term uses the EOS rho — they
disagree before convergence) → T floors → rho explodes. Coupling the energy makes
rho/h/T consistent every iteration and removes the inconsistency. FIX IN PROGRESS: a
gated faithful 3×3 (u,p,h) path (ACID_COUPLED), numerical Jacobian, energy residual
Eq.28; the 2×2 path stays the default until the 3×3 passes 01,02,04,05,13 + improves 25.

Other algorithm items to recheck for faithfulness once 3×3 lands: BDF2 (we use 1st-order
Backward Euler -> 07 acoustic dispersion/ringing), Minmod TVD higher-order reconstruction
(we use 1st-order upwind -> 07 amplitude/dispersion, 13/24/25 smearing).

## EOS — status after fix (commit 025e7a9)
- **Air** ✅ now faithful: `{γ=1.4, Π=0, b=0, cv=720.25, η=0}` → ρ0=1.157, a0=347.85 (Table 1).
  (was cv=717.5 → R=287, ρ0=1.1614, a0=347.19.)
- **Water** ✅ now faithful for Denner cases: `denner_water{γ=4.1, Π=4.4e8, b=0, cv=474.2, η=0}`
  → ρ0=998, a0=1344.6 (Table 1). (was NASG `water_liquid_phase` γ=1.187, a0=1615, η=−1.18e6 —
  20% sound-speed error, and η≠0 caused interface-mixing blow-up.)
- EOS FORM: `phase_props` is NASG (covolume b, offset η); reduces to Denner SG when b=η=0. ✓.

## Remaining discrepancies (IC / BC / reference) — TODO

| case | discrepancy vs Denner | priority |
|---|---|---|
| 01 §7.1 | project = air-water static; Denner = **gas-gas** (L ρ1.156 γ1.4 / R ρ0.160 γ1.6), N=500, interface@0.1. Project N=200, interface@0.5. | P1 |
| 02 §7.1 | project = air-water **box** [0.25,0.45] N=200 t=5e-4 **transmissive** (inflow ill-posed); Denner = gas-gas single step@0.1, N=500, t=0.7, **inlet** BC. | P1 (blocks ACID 02) |
| 04 §7.3.1 | ✅ matches (air, f=2000, δu=0.01, N=500). gate amp_ratio≥0.10 too loose vs paper <0.1%. | P3 gate |
| 05 §7.3.1 | EOS now SG-water ✅; f=6000 ✅. reference auto-tracks phase. | ~ok |
| 07 §7.3.2 | project = Gaussian pulse + **reflective wall**; Denner = **sinusoid f=5000, δu=0.02u0, inlet+outflow** (no wall), L=1.5. | P1 |
| 13 §7.5.2 | L=2 ✅, p_L=1e9 air ✅, p_R=1e4 water ✅. **N=400→800**, **t=6.7e-4→8e-4** (now reachable w/ SG a0=1344). | P1 |
| 24 §7.4.1 | ✅ FIXED: reference now uses Denner Eq.57-62 mixture RH (was Wood-speed Kapila/Wood RH = 27x-too-slow Vs, 1000x-too-weak p_I). Faithful: Vs=6426.8, p_I=1.508e10, ρ_I=1857.3, u_I=4698.0; c24.coupled=true. Solver matches the post-shock plateau exactly. ψ=0.5 only (sweep ψ∈{.25,.75} optional). | DONE |
| 25 §7.4.4 | air post-shock numbers ✅ (u=2869.3,p=1.165e7,ρ_air=6.614). **N=400→1000**, **t=2.42e-4→2.78e-4**, Δt=1e-8 fixed. water EOS now SG ✅. | P1 |

## BC summary
transmissive 13/24/25 ✅ (Denner confirmed transmissive for 24/25, no walls).
07 reflective-wall is WRONG (Denner inlet+outflow). 02 transmissive WRONG (Denner inlet).

## Reference summary
All water references now auto-track the SG water (after EOS fix), so 05/07/13/25 analytic
exact will match Denner once the ICs/geometry are aligned. **24 now uses the faithful Denner
Eq.57-62 interface-region RH** (fixed; was Kapila/Wood). 14/15 = project (NASG / self-convergence).

### case24 RH fix detail (cpp/denner_1d/src/cases.cpp::compute_case24_shock)
- Mixture sound speed a_II = sqrt((γ_mix-1)·cp_mix·T_II) (Eq.57), γ_mix from Eq.58 isobaric
  closure, cp_mix density-weighted (Eq.46). NOT mixture_sound_speed (Wood). M_s=u_s/a_II.
- Eq.59 pressure ratio uses the MULTIPLIER (1+Π̂/p_II) (paper-verified; the old code used
  the Wood speed + conservative+equilibrium Hugoniot, an entirely different, unfaithful path).
- Post-shock colour ψ held = pre-shock ψ (0.5, "constant air-water mixture"); T_I from
  (p_I,ρ_I,ψ) via the applied EOS (temperature_for_mixture_density_pressure).

## Gates (align to paper accuracy)
- 25: drop `interface_u_linf≤1.5` (artificially tight, doc-mandated).
- 04: tighten amp_ratio toward paper ±0.1%.
- others: structural/reasonable.

## Fix order
P0 EOS ✅done. Next P1: faithful ICs/geometry/BC per case (02 gas-gas+inlet → 13 N/t → 25
N/t → 07 sinusoid → 24 CFL/ψ), then P2 references (24 RH), P3 gates. Each ACID case
validated at Denner paper accuracy.
