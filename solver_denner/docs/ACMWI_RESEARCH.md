# Acoustically-consistent MWI (ACMWI) — derivation, measurements, negative result

Date: 2026-07-14. Status: **implemented, fully measured, REJECTED — code reverted
byte-identical (19/19, case01 linf 0), doc-only.** Written as a paper-section skeleton per
mandate. Companions: docs/MWI_DT_RESEARCH.md (Bartholomew unified form),
.claude/rules/denner-pitfalls.md (dt-scaling root cause).

## 1. Motivation

The baseline MWI dissipation coefficient is built from the transient-only momentum
coefficient, `a_P = rho_bar*dx/dt`, giving `dhat = dx/(rho_bar*dx/dt + rho_f*dx/dt)` whose
memory-corrected fixed point is `~dt/rho` — it VANISHES with the time step. Small acoustic-
CFL steps therefore under-damp the pressure-velocity coupling at strong shocks (measured:
case25 reflected-shock amp_ratio_p grows 1.24 -> 1.35 as Co drops 0.45 -> 0.1). Bartholomew's
unified form (advective-only `a = rho|u|`) is dt-independent but its memory feedback
`1/(1+Co_material)` -> 1 at low Mach explodes acoustics (measured, MWI_DT_RESEARCH.md).

## 2. Formulation

Replace the a_P SCALE by the local signal speed (parameter-free characteristic of the
momentum system; not a tuned floor):

    a_P   = 0.5*(re_L+re_R) * (|ubar_f| + a_f)      (a_f = 0.5*(ae_L+ae_R))
    d_f   = dx / a_P
    dhat  = d_f / (1 + (rho_f/dt) * d_f)            (saturation form unchanged, harmonic rho_f)
    theta = ubar + clamp(-dhat*(dpf-gpbar)) + (rho_f/dt)*dhat*(theta_o - ubar_o)   (unchanged)

Analytic properties (VERIFIED numerically to 1e-6, scratch script):
1. Fixed point (with memory): `dhat/(1-g) = dx/(rho_bar(|u|+a))` — dt-INDEPENDENT, ratio
   1.000000 across dt = 1e-6..1e-10 for air-shock / water / air-water-interface states.
2. Low-Mach per-step magnitude vs baseline at Co=0.45: x1.3793 = 2/(1+Co) exactly.
3. Memory feedback `g = 1/(1 + (rho_bar/rho_f)*Co_loc)`, `Co_loc = (|u|+a)dt/dx`: 0.6897 at
   Co_loc=0.45 single-phase, 0.0101 at the air-water interface (harmonic/arithmetic ratio
   protects interfaces) — bounded away from 1 AT THE OPERATING COURANT.

Jacobian: exact for both forms via a stored per-face `df_f` (d(d_f)/dRk = -d_f/(R1+R2) holds
for any frozen speed scale K in d_f = dx/(0.5(R1+R2)K)). Wall time unchanged (heavy shocks
13/14/24/25: 7.67 s new vs 7.66 s legacy). Sound-speed clamp effectively dormant under both
forms (new: 10 of 4.3e6 face evals on case25, zero on case13; legacy: zero).

## 3. Measurements (ACID_LEGACY_DHAT = baseline A/B; ACID_CFL_STUDY = sweep-only cfl env,
   both removed with the revert)

A. Full suite, new form default: **18/19** — case15 FAILS; case07 amp_ratio_p 0.9777 (<0.98
   bar; legacy 0.9858). All other cases pass; case01 linf exactly 0; 04/05/35/36 amps
   1.0006/1.0003/0.994/0.999; case14 hf_p improves 0.040->0.026; case25 amp 1.2428->1.2066,
   hf_p 0.842->0.749.

B. case25 Courant sweep (headline claim: flatness) — **NOT FLAT**:

   | Co   | legacy amp_ratio_p (hf_p) | ACMWI amp_ratio_p (hf_p) |
   |------|---------------------------|--------------------------|
   | 0.10 | 1.3508 (1.087)            | 1.3242 (0.996)           |
   | 0.20 | 1.3138 (0.994)            | 1.2769 (0.897)           |
   | 0.45 | 1.2428 (0.842)            | 1.2066 (0.749)           |

   Uniform improvement at every Courant (amp -0.027..-0.037, hf -0.09) but the SLOPE is
   unchanged (span 0.118 vs 0.108): the overshoot remains dt-dependent.

C. case15 failing terms (spec gate, python mirror): central u-jump 6.04 -> 9.68 (limit 8.0),
   jump concentration 0.0345 -> 0.0501 (limit 0.04); p_osc/r_osc stay 0.

## 4. Why it fails (mechanism, consistent with all measurements)

1. **Fixed-point dt-independence does not transfer to a MOVING front.** The dt-independent
   dissipation is the steady fixed point of the memory recursion at a face; a shock front
   transits a face in ~1/Co steps and never reaches it. The per-step filter is
   `dhat = dx/(rho(s_loc + s_max/Co))`, still Co-dependent for Co<1 — hence the measured
   unchanged sweep slope. Any per-step dt-independent filter would need the memory feedback
   -> 1, which is exactly the Bartholomew explosion (measured, MWI_DT_RESEARCH.md).
2. **Local signal collapse re-creates the g->1 wall.** Where the LOCAL signal speed is far
   below the domain max (case15's near-vacuum stagnation core: |u|+a -> EOS floor; case07's
   air side vs water-set dt), Co_loc << Co_global and g -> 1 locally: the memory accumulates
   the correction and over-filters. Measured: case15's expansion core steepens into a
   step-like central jump (9.68 vs limit 8.0 — the exact failure mode the spec gate exists
   to catch); case07's reflected amplitude drops 0.986 -> 0.978 (below the 0.98 bar).
3. Relation to prior forms: Denner Eq.21's own e_P (advection coefficient) and Bartholomew's
   unified d_f fail across Mach for the same two reasons at their extremes; ACMWI moves the
   wall from "all acoustics" to "locally-slow regions", an improvement in kind but not
   enough for a suite that contains a near-vacuum core (15) and a two-medium acoustic
   domain (07).

## 5. Honest residual analysis / outlook

- The uniform (Courant-independent) part of the gain (amp -0.03, hf -0.09 on case25, hf
  improvements on 04/05/07/14) shows the acoustic scale IS the right magnitude for the
  filter where the flow is acoustically active. The failures are confined to faces whose
  local signal speed decouples from the global dt.
- A form whose per-step filter is dt-independent WITHOUT memory accumulation would need
  dhat itself ~ dx/(rho(|u|+a)) unsaturated; that is exactly Denner Eq.21's steady e_P,
  which this project measured to diverge on 04/05 (pitfalls). The saturation is what keeps
  implicit steps stable, and it is also what re-introduces the dt-dependence. This looks
  like a genuine structural trade-off of collocated implicit MWI, not an implementation
  artifact: per-step dt-independence, low-Mach stability, and near-vacuum robustness —
  pick two.
- If revisited: the measured decisive experiments are the Co sweep (Sec. 3B) and the
  case15 core jump statistics (Sec. 3C); any new candidate must move the SWEEP SLOPE, not
  just its level, while keeping g bounded where s_loc -> 0.

Baseline case25 overshoot remains the documented accepted residual (Denner's own Fig. 23
shows the same single-cell feature).

Reproduce: formulas above are complete; the study envs (ACID_LEGACY_DHAT, ACID_CFL_STUDY)
existed only in the measured working tree and are NOT in the committed code.
