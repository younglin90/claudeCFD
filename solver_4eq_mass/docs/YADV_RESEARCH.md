# Mass-fraction transport (`ACID_YADV`) — derivation, implementation, A/B measurements, verdict

**Workspace**: `solver_4eq_mass`, forked from `solver_denner` at the published-paper state.
**Question**: the 4-equation model transports the VOLUME fraction `alpha`. Does transporting the
MASS fraction `Y` instead work as well?
**Short answer**: it is byte-identical where the mixture is pure, measurably *better* on the
two-phase shock tube (case13), measurably *worse* at a moving contact (case02/14), and it
**breaks 4 of 19 cases** — 3 of them because it is a genuinely different closure from the one the
project's reference solutions encode, 1 (case15) because of a real stiffness failure.

All numbers below are measured on this workspace. Nothing was tuned to force a pass.

---

## 1. Derivation

### 1.1 The two colour functions

The Denner mixture places both phases at the SAME `(p, T)` (single-temperature, single-pressure
4-equation closure). With `rho_a = rho_a(p,T)` and `rho_b = rho_b(p,T)` from the NASG EOS:

```
volume fraction :  rho = alpha*rho_a + (1 - alpha)*rho_b            (Eq.37)
mass   fraction :  Y   = alpha*rho_a / rho                          (phase A mass per unit mass)
```

### 1.2 The inverse is explicit — no iteration

Solve `Y = alpha*rho_a / (alpha*rho_a + (1-alpha)*rho_b)` for `alpha`:

```
  alpha*rho_a*(1 - Y) = Y*(1 - alpha)*rho_b
  alpha*[rho_a*(1-Y) + Y*rho_b] = Y*rho_b
=> alpha = Y*rho_b / ( rho_a*(1-Y) + Y*rho_b )
```

Substituting back gives the mixture density implied by `Y`:

```
  rho = rho_a*rho_b / ( rho_a*(1-Y) + Y*rho_b )   <=>   1/rho = Y/rho_a + (1-Y)/rho_b
```

i.e. the specific-volume (mass-fraction) blend, which is **identical** to the volume-fraction
blend evaluated at the corresponding `alpha`. The two descriptions are the same mixture; only the
*transported* variable differs. Both maps are implemented header-inline in
`cpp/denner_1d/include/denner1d/eos.hpp` as `mass_fraction_from_alpha` /
`alpha_from_mass_fraction`, so the solver and the unit test share one definition.

### 1.3 Why the experiment is worth running

With no phase change, `rho*Y` is conserved:

```
  d(rho Y)/dt + d(rho Y u)/dx = 0
```

Subtracting `Y` times continuity leaves `rho*(Y_t + u Y_x) = 0`, so **`Y` is an exact material
invariant**. `alpha` is not: at fixed composition a mixture cell's volume fraction changes under
compression, because `rho_a` and `rho_b` respond differently to `p` and `T`. The K=0
Allaire/Denner `alpha`-equation (Eq.32) is therefore an approximation wherever a MIXED cell is
compressed. The Y-form should be more faithful under shocks (13/24/25/33/34) and identical where
`(p,T)` is uniform (01/02).

### 1.4 Round-trip conditioning (measured, not assumed)

The brief expected `alpha -> Y -> alpha` to round-trip to ~1e-14. **It does not in general.**
Measured with `scripts/yadv_cond.cpp` over `p` in `[1e4, 1e9]`, `T` in `[250, 1200]`, and the
air/water/vapour pairs in both orders, the worst absolute error tracks

```
  |d alpha| ~ eps * kappa,   kappa = max(rho_a/rho_b, rho_b/rho_a)
```

to within a factor of 2 in every one of the 48 sampled states:

| pair | p | T | kappa | worst \|d alpha\| | eps*kappa |
|---|---|---|---|---|---|
| air\|vapor | any | any | 1.55 | 3.3e-16 | 3.4e-16 |
| air\|water | 1e5 | 300 | 9.1e2 | 1.7e-16 | 2.0e-13 |
| water\|air | 1e5 | 300 | 9.1e2 | 1.5e-13 | 2.0e-13 |
| water\|air | 1e4 | 1200 | 1.9e4 | **2.9e-12** | 4.2e-12 |

Mechanism: when one phase is ~1e4x denser, `Y` compresses the whole `alpha` range into a sliver
of the `Y` range, so `(1-Y)` loses relative precision and the inverse magnifies it. This is
conditioning, not a coding defect.

**The pure ends are EXACT.** `alpha = 0` maps to `Y = 0` and back, `alpha = 1` maps to `Y = 1` and
back, bit-for-bit, because the off-phase term is a multiplication by `0.0`. This is the property
that matters for a sharp interface with pure cells — and it is what keeps case01 exact (§4).

The unit test `cpp/denner_1d/tests/denner1d_unit.cpp` asserts exactness at the ends and the
`8*eps*kappa` bound in between; it passes (`denner1d_unit ok`).

---

## 2. Discretisation

### 2.1 Transport

`Y` is advected with the **same stencil** as `alpha`, in `cpp/denner_1d/src/acid.cpp`:

```
  flux  = thf[i+1]*cf[i+1] - thf[i]*cf[i]
  divu  = (thf[i+1] - thf[i])/dx
  cnew  = clamp( c[i] - dt/dx*flux + dt*c[i]*divu, 0, 1 )
```

Same advecting face velocity `thf`, same flux-minus-`c*div(theta)` (non-conservative advective)
form, same clamp to `[0,1]`. For `c = alpha` this is Eq.32 with K=0; for `c = Y` it is
`Y_t + theta Y_x = 0`, which §1.3 shows is exact for a no-phase-change mixture.

### 2.2 THINC + rho-monotonicity BVD guard

`Y` is also a bounded `[0,1]` colour function that is constant in each pure phase, so the
closed-form semi-Lagrangian tanh reconstruction (`beta = 3.5`) and its interface indicator
(straddle / steep / monotone / unsaturated) apply unchanged. **No new tunable constant was
introduced.**

The rho-monotonicity guard is a *density* test, so the candidate face `Y` is first mapped to the
face `alpha` it implies at that same upwind `(p,T)` via §1.2, and the existing
`min(rho_i-1, rho_i) <= rho_implied <= max(...)` test then runs unchanged.

### 2.3 alpha recovery

`alpha` is a DERIVED quantity on this path. Immediately after the transport update — and
immediately *before* the existing ACID old-level `rho_o/h_o` re-evaluation, so both use the same
`(alpha, p_o, T_o)` triple — `alpha` is recovered from the new `Y` at the OLD level:

```
  alpha[i] = clamp( alpha_from_mass_fraction( Y[i], rho_a(p_o[i],T_o[i]), rho_b(p_o[i],T_o[i]) ), 0, 1 )
```

`alpha` is deliberately **not** a Newton unknown in this first attempt (that would add
`d alpha/d p` and `d alpha/d T` rows to the analytic Jacobian). Within a step `alpha` is frozen at
its old-`(p,T)` value, exactly as it is on the `alpha` path. §5 shows this lag is the mechanism
behind the case15 failure.

### 2.4 What changed in the code

| file | change |
|---|---|
| `cpp/denner_1d/include/denner1d/eos.hpp` | `+` inline `mass_fraction_from_alpha`, `alpha_from_mass_fraction`, with the derivation in comments |
| `cpp/denner_1d/src/acid.cpp` | `+` `ACID_YADV` env switch; `+` `Vec Yv` initialised once from the case `alpha` IC at the initial `(p,T)`; `+` `Yv` saved/restored across the adaptive-dt retry loop; transport block reads `const Vec& cvar = yadv ? Yv : s.alpha`; rho-guard maps the candidate through `cand_a`; `+` the recovery loop |
| `cpp/denner_1d/tests/denner1d_unit.cpp` | `+` round-trip / conditioning / mixture-density-identity checks |
| `scripts/yadv_*.{sh,py,cpp}` | measurement harness (baseline capture, A/B sweep, conditioning probe, alpha-drift probe) |

**Untouched, as required**: `cases.cpp` (case definitions and reference solutions) and
`validation.cpp` (gates are p/u/rho based and need no change).

---

## 3. Criterion A — switch OFF is byte-identical

```
denner1d_unit ok
OFF: DENNER1D_CPP_METRIC pass_count=19 total=19
```

**Build-hygiene finding, worth recording.** The `build-cpp/` tree copied into this workspace
contained **stale object files** for `eos.cpp`, `numerics.cpp` and `png.cpp` that did not
correspond to the paper state. A dump baseline captured from that tree therefore was *not* the
published result. The incremental builds during development kept matching it (same stale objects),
so the check looked green while measuring the wrong reference. A full reconfigure + recompile
exposed the discrepancy in the 12-digit dump columns; the JSON validate metrics print 6 digits and
were unaffected, so no A/B number in this document changed.

The check was then redone against the authoritative artefact — the untouched `solver_denner`
workspace's existing binary, run read-only, never rebuilt (`scripts/yadv_verify.py`):

```
(1) ACID_YADV unset  vs  solver_denner published binary
    case01 BYTE-IDENTICAL   case02 BYTE-IDENTICAL   case13 BYTE-IDENTICAL
    case14 BYTE-IDENTICAL   case15 BYTE-IDENTICAL   case24 BYTE-IDENTICAL
    case25 BYTE-IDENTICAL   case33 BYTE-IDENTICAL   case34 BYTE-IDENTICAL
```

Nine of nine, including all four cases the brief named. The default path is bit-preserved against
the published solver, which is a stronger statement than the originally requested one.

**Always rebuild this workspace from a clean configure before trusting a dump comparison.**

---

## 4. Criterion C — case01 pressure equilibrium with the switch ON

```
{"case":"01","N":200,"pass":true,"finite":true,
 "l2_p":0,"l2_u":0,"l2_rho":0,"corr_p":1,"corr_u":1,"corr_rho":1,
 "linf_p":0,"linf_u":0,"linf_rho":0}
```

`linf_p = linf_u = linf_rho = 0` **exactly**. Stronger than required: the whole case01 dump is
**byte-identical between the two paths** (`scripts/yadv_verify.py` step 2, 11036 bytes), while
every other case differs on nearly every row:

```
(2) ACID_YADV=1 vs ACID_YADV unset (same binary)
    case01: BYTE-IDENTICAL
    case02: differs, 192/500 rows, max|d alpha|=0.2945
    case13: differs, 400/400 rows, max|d alpha|=0.0669
    case14: differs, 400/400 rows, max|d alpha|=0.444
    case15: differs, 400/400 rows, max|d alpha|=0.9448
    case24: differs, 800/800 rows, max|d alpha|=0.4998
    case25: differs, 327/400 rows, max|d alpha|=0.04086
    case33: differs, 761/800 rows, max|d alpha|=0.7493
    case34: differs, 800/800 rows, max|d alpha|=0.2499
```

Why: case01 is a static air/water interface with `u = 0` and `alpha` in `{0, 1}` exactly. `u = 0`
makes `thf = 0`, so flux and `divu` vanish and `Y` is unchanged; and by §1.4 the pure ends
round-trip bit-for-bit, so the recovered `alpha` is the identical `{0,1}` field. The machine-exact
interface-equilibrium property survives the change of transported variable.

---

## 5. Criterion B — all 19 cases with `ACID_YADV=1`

`DENNER1D_CPP_METRIC pass_count=15 total=19`

| case | alpha path | Y path | note |
|---|---|---|---|
| 01 PE static interface | PASS | **PASS** | byte-identical |
| 02 PE advection (gas-gas) | PASS | PASS | degraded (§6) |
| 04 acoustic, homogeneous | PASS | PASS | unchanged to print precision |
| 05 acoustic, homogeneous | PASS | PASS | `l2_rho` 4.5e-5 -> 9.3e-5 |
| 07 acoustic + interface | PASS | PASS | `l2_rho` 1.9e-8 -> 1.2e-5 |
| 13 two-phase shock tube | PASS | **PASS, improved** | best result of the experiment (§6) |
| 14 water-air shock tube | PASS | PASS | degraded (§6) |
| 15 cavitation / tension | PASS | **FAIL** | genuine stiffness failure (§7) |
| 24 mixture shock, alpha=0.50 | PASS | **FAIL** | closure conflict (§7) |
| 25 shock-interface | PASS | PASS | essentially neutral (§6) |
| 26 single-phase Mach 10 air | PASS | PASS | marginally better |
| 27 single-phase Mach 10 water | PASS | PASS | neutral |
| 28 single-phase Mach 100 air | PASS | PASS | marginally better |
| 30 shock-contact air/gas | PASS | PASS | `linf_rho` 0.304 -> 0.105 |
| 31 shock-contact air/gas | PASS | PASS | neutral |
| 33 mixture shock, alpha=0.75 | PASS | **FAIL** | closure conflict (§7) |
| 34 mixture shock, alpha=0.25 | PASS | **FAIL** | closure conflict (§7) |
| 35 acoustic helium/air | PASS | PASS | `l2_rho` 1.9e-5 -> 9.4e-5 |
| 36 acoustic argon/air | PASS | PASS | `l2_rho` 5.5e-5 -> 6.4e-5 |

### The one physical difference, measured

`scripts/yadv_alpha_drift.py` reports the `alpha` field produced by each path (dump precision on
the `alpha` column is 6 digits):

| case | alpha range, alpha path | alpha range, Y path | max abs d(alpha) | max rel d(rho) |
|---|---|---|---|---|
| 01 | 0 … 1 | 0 … 1 | 0 | 0 |
| 02 | 0 … 1 | 0 … 1 | 0.295 | 0.593 |
| 13 | 1e-6 … 0.999999 | 4.9e-11 … 0.999999 | 0.067 | 0.059 |
| 14 | 1e-6 … 0.999999 | 1e-6 … 0.999999 | 0.444 | 2.19 |
| **15** | **0.055 (uniform)** | **0.991 … 0.9998** | **0.945** | **0.9998** |
| **24** | **0.5 (uniform)** | **2.3e-4 … 0.5** | **0.4998** | **2.47** |
| 25 | 1e-6 … 0.999999 | 1.4e-9 … 0.999999 | 0.041 | 0.028 |
| 28 | 0.999999 (uniform) | 0.996813 … 0.999999 | 3.2e-3 | 0.032 |
| 30 | 1e-6 … 0.999999 | 1e-6 … 0.999999 | 0.240 | 0.409 |
| **33** | **0.75 (uniform)** | **7.0e-4 … 0.75** | **0.7493** | **5.09** |
| **34** | **0.25 (uniform)** | **7.6e-5 … 0.249233** | **0.2499** | **1.35** |

The pattern is unambiguous: wherever the mixture is a **homogeneous mixed state** (15/24/33/34),
the two closures diverge by O(1) in `alpha` and by up to 5x in density. Wherever cells are pure
(01, and the bulk of 02/13/14/25/30), the difference is confined to the interface band.

---

## 6. Criterion D — A/B on the shock/compression and interface cases

| metric | | 13 | 24 | 25 | 33 | 34 | 02 | 14 |
|---|---|---|---|---|---|---|---|---|
| `l2_p` | alpha | 0.02161 | 0.02894 | 0.04805 | 0.03033 | 0.02665 | 0 | 0.01406 |
| | **Y** | **0.01711** | 1.123 | **0.04779** | 1.574 | 0.7831 | 0 | 0.01453 |
| `l2_u` | alpha | 0.05258 | 0.03718 | 0.02606 | 0.04008 | 0.03275 | 3.6e-15 | 0.1028 |
| | **Y** | **0.04247** | 0.3959 | 0.02634 | 0.4110 | 0.3872 | 6.4e-15 | 0.1301 |
| `l2_rho` | alpha | 0.01707 | 0.02992 | 0.03389 | 0.03104 | 0.02806 | 0.03052 | 0.03819 |
| | **Y** | **0.01645** | 0.5032 | 0.03388 | 0.7607 | 0.4074 | 0.04378 | 0.07636 |
| `corr_rho` | alpha | 0.9990 | 0.9972 | 0.9976 | 0.9970 | 0.9975 | 0.9971 | 0.9929 |
| | **Y** | 0.9990 | 0.4072 | 0.9976 | 0.5081 | 0.4460 | 0.9940 | 0.9715 |
| `linf_p` | alpha | 0.2637 | 0.6367 | 0.5084 | 0.6405 | 0.6133 | 0 | 0.05969 |
| | **Y** | **0.1690** | 2.640 | **0.5007** | 2.551 | 2.034 | 0 | 0.05966 |
| `linf_u` | alpha | 0.6643 | 0.7802 | 0.4642 | 0.8031 | 0.7321 | 9.7e-15 | 0.9902 |
| | **Y** | **0.4376** | 0.6999 | 0.4703 | 0.5680 | 0.8102 | 1.5e-14 | 0.9963 |
| `linf_rho` | alpha | 0.1932 | 0.6421 | 0.3974 | 0.6445 | 0.6238 | 0.6623 | 0.2727 |
| | **Y** | **0.1726** | 0.9083 | **0.3868** | 1.367 | 0.8016 | 0.9559 | 0.5352 |

One line each:

- **13 (water-air shock tube, sharp interface, strong compression) — the Y path WINS.** `linf_p`
  0.264 -> 0.169 (-36%), `linf_u` 0.664 -> 0.438 (-34%), `l2_u` -19%, and the shock-front
  diagnostic `case13_u_shock_delta_cells` goes 1 -> 0 with `peak_delta_u` 373 -> 0 cells. This is
  the predicted effect: the residual 1e-6 off-phase in each compressed region is a *mixed* state,
  and holding `Y` rather than `alpha` through the compression puts the wave in the right place.
- **24 / 33 / 34 (homogeneous mixture Rankine-Hugoniot) — catastrophic, but by construction.**
  `corr_p` collapses to -0.29 / 0.35 / 0.007. See §7: the reference solution *defines* `alpha` as
  held across the shock, which is precisely what the Y closure refuses to do.
- **25 (shock-interface) — neutral to marginally better.** `linf_p` 0.5084 -> 0.5007, `linf_rho`
  0.3974 -> 0.3868, `l2_rho` unchanged at 4 digits. The interface is pure on both sides, so the
  two closures barely differ; the small gain comes from the mixed cells inside the front.
- **02 (pure advection of a gas-gas contact at uniform p) — degraded, and the mechanism is the
  reconstruction, not the transport.** `l2_rho` 0.0305 -> 0.0438, `corr_rho` 0.9971 -> 0.9940,
  `linf_rho` 0.662 -> 0.956. Here `(p,T)` is uniform in each gas, so `rho_a` and `rho_b` are
  constants and Y-transport and alpha-transport are the *same PDE*. The difference is purely
  THINC: in a cell cut by a sharp interface at sub-cell position `xi_c`, the exact averages are
  `alpha_avg = xi_c` but `Y_avg = xi_c*rho_a / (xi_c*rho_a + (1-xi_c)*rho_b)`. The tanh profile is
  fitted assuming the average is volume-like, so fitting it to `Y_avg` places the reconstructed
  interface at the wrong `xi_c` whenever `rho_a != rho_b` — a density-ratio distortion of the
  sub-cell reconstruction, worth ~1 cell of front position here.
- **14 (water-air shock tube, opposite orientation) — degraded.** `l2_rho` 0.0382 -> 0.0764 (2x),
  `corr_rho` 0.9929 -> 0.9715, `linf_rho` 0.273 -> 0.535 (2x); pressure is untouched
  (`linf_p` 0.0597 both). Same THINC distortion as case02, amplified because case14's contact
  carries a 20:1 density ratio *and* sits in a compressing field, so the misplaced sub-cell
  interface is then advected by a shock-driven velocity.

---

## 7. Why the four failures fail

### 7.1 Cases 24 / 33 / 34 — a closure conflict, not a numerical failure

`cpp/denner_1d/src/cases.cpp` builds these references from Denner's mixture Rankine-Hugoniot
(Eqs. 59-62) with the volume fraction **explicitly held constant across the shock**:

```cpp
s.alpha_post = s.alpha_pre;  // psi held (homogeneous mixture)
```

and the comment above it records that letting `alpha` float "giv[es] the wrong post-shock state"
under the paper's closure. That reference is self-consistent with the `alpha` model, in which
`alpha` is a material invariant by construction (§2.1).

The Y closure asserts the opposite: `Y` is the invariant and `alpha = alpha(p,T,Y)` must move.
Measured, it moves a long way — the 50/50 air-water mixture of case24 has `Y ~ 1.16e-3`, and a
~7e7 pressure ratio compresses the gas from `alpha = 0.5` to `alpha = 2.3e-4`, a 2000x collapse
of the gas volume fraction, with a 247% density error against the held-`alpha` reference.

That collapse is *physically reasonable* — real bubbles hit by such a shock do collapse — but it
means **cases 24/33/34 cannot adjudicate between the two models with the current references**.
Deciding the question would require regenerating their Rankine-Hugoniot with the `Y` closure.
This is reported, not worked around; no gate, case or reference was modified.

### 7.2 Case 15 — a genuine failure of the Y path

Case15's reference is NOT a closed-form solution: `reference_state` calls
`computed_reference(c, 800)`, which calls `solve_case` -> `solve_case_acid` **in the same
process**, so `ACID_YADV=1` applies to the N=800 reference too. The comparison is therefore
self-consistent and this failure cannot be blamed on a mismatched closure.

```
{"case":"15","pass":false,"amp_ratio_p":0.330034,"amp_ratio_rho":0.339504,
 "l2_p":0.166533,"l2_rho":0.167608,"linf_p":0.669966,"linf_rho":0.660496}
```

`amp_ratio_p = 0.33` means the N=400 solution carries a third of the N=800 pressure amplitude:
the Y path is **not converging** on this case.

Mechanism. Case15 is a uniform `alpha = 0.055` air-in-water mixture pulled apart at `u = ±100`,
so `p` falls toward cavitation. `Y = 0.055*1.157 / 945 ~ 6.7e-5`. Because `rho_air ~ p/(R T)`,

```
  alpha = Y*rho_b / ( rho_a(p,T)*(1-Y) + Y*rho_b )   ->   1   as p -> 0
```

and measured, it does: `alpha` runs `0.055 -> 0.991…0.9998`, i.e. the mixture becomes essentially
pure gas and the density collapses (`max rel d(rho) = 0.9998`). `d alpha/d p` is enormous in this
regime, and by the §2.3 design decision `alpha` is recovered at the OLD `(p,T)` and then held
frozen through the Newton solve. That makes the `alpha <-> p` coupling an **explicit, lagged
feedback loop across the stiffest state dependence in the problem** — exactly the configuration
that produces grid-dependent amplitude loss. The `alpha` path has no such loop, because there
`alpha` genuinely does not depend on `p`.

The fix implied by this diagnosis is to make `alpha` an implicit function of the Newton unknowns
(add `d alpha/d p`, `d alpha/d T` to the analytic Jacobian). That was explicitly out of scope for
this first attempt and is **not** attempted here.

Note that the round-trip conditioning of §1.4 is *not* the culprit: at case15's state
`kappa ~ 860`, so the round-trip noise on `alpha` is ~2e-13, eleven orders below the 0.055 -> 0.99
excursion.

---

## 8. Verdict

**Transporting `Y` instead of `alpha` does not "work as well" as a drop-in replacement, but the
experiment produced a real positive result and a sharp diagnosis of the negatives.**

What is established:

1. **The switch is safe.** `ACID_YADV` unset reproduces the published `solver_denner` binary
   byte-for-byte (19/19 PASS, 9/9 dumps identical). The A/B is clean.
2. **The pressure-equilibrium property survives.** case01 is byte-identical on both paths;
   `linf_p = linf_u = linf_rho = 0` exactly. The change of transported variable does not damage
   the scheme's defining invariant.
3. **The physical hypothesis is confirmed where it can be tested cleanly.** On case13 — a sharp
   interface, strong compression, pure cells on both sides — the Y path is uniformly better
   (`linf_p` -36%, `linf_u` -34%, shock front exactly placed). That is the predicted benefit of
   transporting a true material invariant.
4. **Two independent defects block adoption**, and they are different in kind:
   - a **reconstruction** defect (cases 02, 14): fitting the THINC tanh to a *mass* average
     misplaces the sub-cell interface by a density-ratio distortion. This is a fixable modelling
     error in the reconstruction step, not in the transport — reconstruct in `alpha` (recovered at
     the face-local `(p,T)`), then convert the face value to `Y`. Not attempted here because it
     deviates from the agreed design; recommended as the next experiment.
   - a **stiffness** defect (case15): `alpha` recovered at the old `(p,T)` and frozen through the
     Newton makes `alpha <-> p` an explicit lagged loop, which diverges from the grid-converged
     answer where `d alpha/d p` is large. Fixing this requires `alpha` inside the Jacobian, which
     was out of scope by design.
5. **Cases 24/33/34 are currently unanswerable.** Their references *define* `alpha` as held across
   the shock, so they test the `alpha` closure by construction. They should be excluded from any
   future Y-path comparison until their Hugoniot is regenerated under the `Y` closure.

Recommendation: keep `ACID_YADV` as an opt-in research path (default OFF, published build
untouched); do **not** promote it. The two next experiments, in order of expected value, are
(a) alpha-space THINC reconstruction with a `Y` flux, which should recover 02/14 while keeping the
case13 win, and (b) `alpha` as an implicit function in the Newton, which is what case15 needs.

---

## 9. Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release   # clean configure -- see the §3 warning
cmake --build build-cpp -j8

# unit tests (includes the Y<->alpha round-trip / conditioning checks)
./build-cpp/cpp/denner_1d/denner1d_unit

# A/B suite  (writes /tmp/yadv_off_val.txt and /tmp/yadv_on_val.txt)
bash scripts/yadv_remeasure.sh       # OFF + ON validate sweeps
python3 scripts/yadv_verify.py       # sections 3 and 4: dump-level byte checks
python3 scripts/yadv_table.py        # the A/B metric table of section 6
python3 scripts/yadv_alpha_drift.py  # the alpha-drift table of section 5

# round-trip conditioning probe of section 1.4
g++ -O2 -Icpp/denner_1d/include scripts/yadv_cond.cpp \
    -o /tmp/yadv_cond build-cpp/cpp/denner_1d/libdenner1d.a -fopenmp && /tmp/yadv_cond
```

Harness note (this workspace, WSL through the agent tooling): inline `for` loops lose variables
and `>` redirects intermittently produce 0-byte files. Every measurement above was taken through a
script file or through Python `subprocess` capture. Never build with `-march=native`.
