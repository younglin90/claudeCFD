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

## 8. Verdict (round 1 — SUPERSEDED IN PART, see §10-§12)

> **Round 2 has since measured both of the follow-ups this section recommends.**
> §10 tests the alpha-space THINC reconstruction: it is a **negative** result — case02 gets
> markedly *worse*, not better. §11 computes the Y-consistent Hugoniot for cases 24/33/34 and
> **overturns §7.1**: the alpha-held reference is an exact Rankine-Hugoniot solution of the true
> NASG mixture EOS, and the Y-path solver satisfies *no* Rankine-Hugoniot jump at all
> (momentum residual 88%). 24/33/34 are a **solver defect**, not an unanswerable closure
> conflict. §12 is the revised verdict.

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

---
---

# ROUND 2

Two questions, both measured on a **fully clean rebuild**
(`rm -rf build-cpp && cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8`,
wrapped as `scripts/yadv_clean_build.sh`):

* **§10** — does reconstructing the THINC tanh in `alpha` (interface geometry is volumetric) and
  converting only the FACE value to `Y` recover cases 02/14 while keeping the case13 win?
* **§11** — what is the mass-fraction-consistent Hugoniot for cases 24/33/34, and is the Y-path
  solver converging to *its own* consistent post-shock state?

Default-path guarantee, re-verified after every change in this round:

```
(1) ACID_YADV unset vs solver_denner published binary
    case01 .. case34 : BYTE-IDENTICAL   (9/9, incl. 01,02,13,14,25)
    OFF: DENNER1D_CPP_METRIC pass_count=19 total=19
```

---

## 10. Task A — alpha-space THINC reconstruction with a `Y` flux. NEGATIVE.

### 10.1 What was implemented

In `cpp/denner_1d/src/acid.cpp`, inside the colour-function transport block:

* a cell `alpha` field is recovered from the transported `Yv` at the current old-level `(p,T)`
  (`alpha_cell[i] = alpha_from_mass_fraction(Yv[i], rho_a(p_o,T_o), rho_b(p_o,T_o))`);
* the ghost array `arec` built from it is what the tanh profile **and the interface indicator**
  (`straddle` / `steep` / `monotone` / `unsat`) now read — under `ACID_YADV` they never see a
  mass fraction again;
* the BVD clamp is applied in that same reconstruction space;
* the rho-monotonicity guard receives the face `alpha` **directly** — it keeps its published
  semantics verbatim (the `alpha_from_mass_fraction` hop that round 1 needed is gone);
* only the ACCEPTED face value is converted, `Y_f = mass_fraction_from_alpha(alpha_f, ra, rb)`
  at the upwind cell's `(p,T)`, then re-bounded between the two neighbour `Y` values.

`beta` stays `3.5`. **No new tunable constant, no per-case logic**, and `cases.cpp` /
`validation.cpp` are untouched. With the switch OFF, `arec` *is* `ae` — same object, no extra
work — so the default path is byte-identical (verified above).

### 10.2 Result: 15/19, and case02 is much worse

`DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate` -> `pass_count=15 total=19`,
the same four failures (15, 24, 33, 34) as round 1.

Naming: **v1** = round 1 (reconstruct in `Y`), **v2** = this change (reconstruct in `alpha`,
convert the face value).

| case | metric | alpha path | v1 (Y-THINC) | **v2 (alpha-THINC)** | v2 vs the goal |
|---|---|---|---|---|---|
| **02** | `l2_rho`   | 0.03052 | 0.04378 | **0.1579** | **3.6x WORSE than v1** |
| | `corr_rho` | 0.9971 | 0.9940 | **0.9274** | worse |
| | `linf_rho` | 0.6623 | 0.9559 | 0.9137 | ~unchanged |
| | `hf_rho`   | 1.302 | 1.502 | **0.0789** | 19x smoother |
| **14** | `l2_rho`   | 0.03819 | 0.07636 | **0.07630** | **no recovery (-0.08%)** |
| | `corr_rho` | 0.9929 | 0.9715 | 0.9716 | no recovery |
| | `linf_rho` | 0.2727 | 0.5352 | 0.5348 | no recovery |
| **13** | `linf_p`   | 0.2637 | 0.1690 | 0.1691 | win survives |
| | `linf_u`   | 0.6643 | 0.4376 | 0.4373 | win survives |
| | `l2_rho`   | 0.01707 | 0.01645 | 0.01616 | slightly better |
| | `peak_delta_u` | 373 | **0** | **395** | **v1's win LOST** |
| **30** | `l2_rho`   | 0.01642 | 0.008497 | **0.007524** | best of the three |
| | `linf_rho` | 0.3039 | 0.1049 | **0.08741** | best of the three |
| **25** | all | — | — | neutral to 4 digits | unchanged |
| 15/24/33/34 | all | — | — | unchanged | indicator never fires there |

So the hypothesis is **refuted on its own target**: case02 degrades by 3.6x instead of
recovering, case14 does not move at all, and case13 keeps its norm-metric win but loses the
exact shock-front placement (`peak_delta_u` 0 -> 395) that was round 1's headline.

### 10.3 Why — measured, not assumed

`scripts/yadv_case02_probe.py` prints the interface band. case02 is a pure gas_a|gas_b contact
at **uniform (p,T)**, `rho_a/rho_b = 7.231`, advected at `u=1`; Y-transport and alpha-transport
are the same PDE there, so every difference is reconstruction/flux only.

```
| i | x | alpha OFF | alpha ON(v2) | rho OFF | rho ON | rho_ref |
| 392 | 0.7850 | 0.998596 | 0.172596 | 1.15561 | 0.33208 | 1.15701 |
| 398 | 0.7970 | 0.983750 | 0.092972 | 1.14080 | 0.25269 | 1.15701 |
| 399 | 0.7990 | 0.335726 | 0.083597 | 0.49472 | 0.24335 | 1.15701 |
| 400 | 0.8010 | 0.056587 | 0.075091 | 0.21642 | 0.23487 | 0.160001 |
l1_rho  alpha = 0.00293   Y(v2) = 0.03958      (13x worse)
```

The alpha path holds a 2-cell contact. The v2 Y path has no contact left at all — `alpha` decays
smoothly from 0.17 across ~150 cells. THINC has switched itself off:

```
scripts/yadv_thincdbg.sh
case02  ALPHA: activations=1426  rho_guard_rejects=82
case02  YADV : activations= 306  rho_guard_rejects=75      <- 4.7x fewer
```

The mechanism is the **non-commutation of the semi-Lagrangian average with the mass map**. Write
`M(a) = a*ra/(a*ra + (1-a)*rb)`. The flux needs the departure-region average of the transported
variable; v2 supplies `M(mean(alpha))` instead of `mean(M(alpha))`. `M` is concave for
`ra > rb`, so `M(mean) > mean(M)` and every face **over-delivers** `Y` downstream. The step
degrades, the recovered `alpha` stops satisfying `straddle` (a 0.5 crossing) and `steep`
(a >0.5 jump), the indicator stops firing, first-order upwind takes over, and the front
collapses into the 150-cell ramp above. This is self-reinforcing, which is why the loss is so
much larger than round 1's.

### 10.4 The obvious "fix" was tested and is WORSE — so the defect is deeper

If the conversion is the problem, the natural repair is to drop it: pointwise `Y` is the *same*
0/1 step as `alpha`, so the volume average of `Y` over the swept region simply **is**
`alpha_bar`. That was measured behind a default-OFF diagnostic switch `ACID_YADV_VOLFLUX`
(`scripts/yadv_volflux.sh`; the switch changes nothing unless `ACID_YADV` is also set, and the
default path stays byte-identical):

| case02 `l2_rho` | alpha | v1 (Y-THINC) | v2 (alpha-THINC + `M`) | v3 probe (alpha-THINC, no `M`) |
|---|---|---|---|---|
| | 0.03052 | 0.04378 | 0.1579 | **0.3058** |
| suite | 19/19 | 15/19 | 15/19 | **14/19** (case02 now FAILS) |

**Removing the conversion makes it worse still.** The ordering is monotone in how far the flux
space is from the transported variable's own space, which identifies the real defect:

> The Y path's cell state `Yv[i]` is a **mass** average (`Y = mass of A / total mass` — that is
> how it is initialised and what the EOS inverse expects), but the transport stencil
> `c - dt/dx*flux + dt*c*div(theta)` is the update for a **volume** average. For pure cells the
> two coincide, which is why case01 is exact and most of the domain is unaffected; in a cut cell
> they differ by exactly the density-ratio factor. No pointwise face map can reconcile them,
> because the mismatch is between the cell state and the stencil, not between two face values.

Round 1's v1 was the least-bad of the three only because fitting the tanh in `Y` and averaging in
`Y` is at least *self-consistent* — it is the wrong shape, but it is a genuine average of the
variable being updated.

**Consequence:** "reconstruct in alpha" cannot be made to work as an incremental change. A
correct Y path needs the **conservative** form `d(rho Y)/dt + d(rho Y u)/dx = 0` with `rho*Y` as
the transported conserved variable, so that cell state and flux live in the same space. That is
a different experiment, not a tweak to this one.

---

## 11. Task B — the Y-consistent Hugoniot for cases 24/33/34

Computed by `scripts/yadv_hugoniot.cpp` (standalone; links only `libdenner1d.a` for
`phase_props`; **`cases.cpp` and `validation.cpp` are neither modified nor linked in**).
Build/run: `bash scripts/yadv_hugoniot.sh`.

### 11.1 Derivation

Both closures share the pre-shock state and the shock speed
`Vs = Ms * a_mix` with Denner's thermo-consistent Eq.57-58 mixture speed, `Ms = 10`.

**(A) alpha-held** — verbatim what `cases.cpp:compute_case24_shock` builds: the equivalent
stiffened gas `1/(gamma_mix-1) = sum psi_k/(gamma_k-1)` (Eq.57), `Pihat` (Eq.60), the pressure
multiplier (Eq.59), `rho_post` (Eq.61), `u_post` (Eq.62), then `alpha_post := alpha_pre`.

**(B) Y-held** — mass fractions fixed across the shock (no phase change), the jump conditions
imposed with the **true NASG mixture EOS at a single `(p,T)`**. In the shock frame with the
pre-shock state stationary, `mdot = rho_pre*Vs`, `v = 1/rho`:

```
  Rayleigh :  p_post = p_pre + mdot^2 (v_pre - v_post)
  Hugoniot :  h_post - h_pre = 0.5 (p_post - p_pre)(v_pre + v_post)
  EOS      :  v_post = Y/rho_a(p_post,T_post) + (1-Y)/rho_b(p_post,T_post)
  h        :  h = Y h_a(p,T) + (1-Y) h_b(p,T)
```

Two equations in `(v_post, T_post)`: bracket-and-bisect on `v_post`, with `T_post` recovered from
the EOS constraint at each trial (specific volume is monotone in `T` at fixed `p`). Then

```
  u_post     = Vs (1 - v_post/v_pre)
  alpha_post = Y rho_post / rho_a(p_post, T_post)      <- a RESULT, not an input
```

### 11.2 The two closures, side by side

| | | case24 (`alpha_pre`=0.50) | case33 (0.75) | case34 (0.25) |
|---|---|---|---|---|
| `a_mix` Eq.57 [m/s] | | 642.676 | 545.649 | 820.139 |
| `Vs` [m/s] | | 6426.761 | 5456.494 | 8201.394 |
| pre: `rho`, `Y` | | 499.5787, 1.158045e-3 | 250.3681, 3.466107e-3 | 748.7894, 3.863132e-4 |
| **`p_post` [Pa]** | (A) | 1.508398e+10 | 5.877237e+09 | 3.162353e+10 |
| | **(B)** | **8.273766e+09** | **3.389760e+09** | **1.950648e+10** |
| **`rho_post`** | (A) | 1857.257 | 1183.346 | 2012.204 |
| | **(B)** | **833.977** | **459.160** | **1222.104** |
| **`u_post` [m/s]** | (A) | 4698.044 | 4302.029 | 5149.460 |
| | **(B)** | **2576.926** | **2481.210** | **3176.357** |
| **`T_post` [K]** | (A) | 16938.19 | 13837.32 | 21767.26 |
| | **(B)** | **7114.23** | **5689.28** | **11106.29** |
| **`alpha_post`** | (A) | 0.500000 | 0.750000 | 0.250000 |
| | **(B)** | **2.392475e-04** | **7.695523e-04** | **7.744281e-05** |
| `Y_post` | (A) | 8.321540e-01 | 9.343885e-01 | 6.265147e-01 |
| | (B) | 1.158045e-03 | 3.466107e-03 | 3.863132e-04 |
| (B)/(A) `p` | | 0.5485 | 0.5768 | 0.6168 |

### 11.3 The alpha-held reference is NOT an artefact — it is an exact RH solution

The standalone program also feeds closure (A)'s own post-shock state back into the **true NASG
mixture** jump conditions:

| case | Rayleigh residual | Hugoniot residual | `(Y_post-Y_pre)/Y_pre` |
|---|---|---|---|
| 24 | +1.9e-06 Pa (rel **1.26e-16**) | +3.7e-09 J/kg (rel **1.89e-16**) | **+717.6** |
| 33 | +1.9e-06 Pa (rel **3.25e-16**) | -5.6e-09 J/kg (rel **-3.78e-16**) | **+268.6** |
| 34 | +1.5e-05 Pa (rel **4.83e-16**) | 0.0e+00 J/kg (rel **0.00**) | **+1620.8** |

This corrects §7.1. Both phases here have NASG `b = 0`, so an alpha-frozen mixture is *exactly*
an equivalent stiffened gas — Denner's Eq.57-62 is not an approximation of the mixture EOS, it
**is** the mixture EOS under that closure. So (A) and (B) are *both* exact Rankine-Hugoniot
solutions of the same EOS at the same `Vs`; the jump conditions are 3 equations for 4 unknowns
and each closure supplies the missing constraint. They differ in physics, not in rigour:
closure (A) permits enormous interphase mass transfer (`Y` rises by 268x-1621x across the
shock), which is precisely what K=0 Allaire `alpha`-transport does implicitly. The reference is
legitimate.

### 11.4 The test problem DRIVES the Y path with a closure-(A) state

`cases.cpp:689-694` seeds `x < 0.1` with `sh24.{alpha,u,p,T}_post` — closure (A) — and the left
BC is `transmissive`, so that state acts as a sustained piston. Reconstructing `Y` from the
dumps (`scripts/yadv_yprofile.py`; NASG with `b=0` makes `T` explicit from `(p,rho,alpha)`):

| case | inflow `Y` (closure A) | undisturbed pre-shock `Y` | ratio | `alpha` on the two sides of x=0.1 |
|---|---|---|---|---|
| 24 | 0.832154 | 1.158045e-03 | **718.6x** | 0.50 and 0.50 — **no jump** |
| 33 | 0.934388 | 3.466107e-03 | **269.6x** | 0.75 and 0.75 — **no jump** |
| 34 | 0.626515 | 3.863132e-04 | **1622x** | 0.25 and 0.25 — **no jump** |

So the initial data contains a large material contact **in the transported variable** that the
alpha model literally cannot see. The Y path therefore solves a Riemann problem, not a single
travelling shock. Measured structure (`scripts/yadv_struct.py`, case33, the one case whose front
is still inside the domain at `t_end`):

```
 x<0.11   inflow (closure A)          p=5.877e9   u=4302   rho=1183   Y=0.9344
 x~0.12   LEFT-FACING SHOCK
 0.15-0.39                            p=1.4993e10 u=2443.5 rho=2118.8 Y=0.9344
 x~0.45   CONTACT (p,u continuous; Y drops 0.9344 -> 3.466e-3)
 0.50-0.93 shock-processed material   p=1.4992e10 u=2443.6 rho=1525.7 Y=3.466107e-03  <- = Y_pre
 x~0.94   LEADING SHOCK               (reference front position is 0.800)
 x>0.95   undisturbed                 p=1e5 u=0 rho=250.37 alpha=0.75
```

`Y` is preserved to 7 significant figures through the leading shock — the transport itself is
doing its job. But the shock is **overdriven**: closure (A)'s piston velocity is 1.7x-1.8x
closure (B)'s, so the front runs to `x=0.939` instead of `0.800` on case33 and has left the
domain entirely on cases 24 and 34 (`scripts/yadv_front.py`).

### 11.5 The decisive test: the Y path satisfies NO Rankine-Hugoniot jump

Whatever colour function is transported and however the problem is driven, **any** valid weak
solution of the Euler system must satisfy mass, momentum and energy across its own shock. That
is closure-agnostic. `scripts/yadv_rhcheck.py` takes the undisturbed pre-shock state and the
plateau behind the leading front straight out of the dump, infers `Vs` from **mass**
conservation, and reports the momentum and energy residuals. The alpha path is the built-in
control that validates the algebra:

| case | path | `p_post` | `rho_post` | `u_post` | `Vs` (mass) | momentum resid (rel) | energy resid (rel) |
|---|---|---|---|---|---|---|---|
| 24 | alpha | 1.50840e+10 | 1857.25 | 4698.0 | 6426.8 | **-2.48e-06** | **+1.81e-06** |
| 24 | **Y** | — leading shock has left the domain, no undisturbed state — | | | | | |
| 33 | alpha | 5.87723e+09 | 1183.22 | 4302.0 | 5456.6 | **-2.91e-05** | **+5.83e-05** |
| 33 | **Y** | 1.49921e+10 | 1525.65 | 2443.6 | 2923.3 | **+8.81e-01** | **+6.46e-01** |
| 34 | alpha | 3.16235e+10 | 2012.20 | 5149.5 | 8201.4 | **-1.59e-06** | **+3.82e-07** |
| 34 | **Y** | — leading shock has left the domain, no undisturbed state — | | | | | |

The alpha path closes to 1e-6..1e-5 relative. The Y path misses **momentum by 88% and energy by
65%**. Its plateau is dead flat (relative spread ~1e-5), so this is a converged, steady, and
**inadmissible** state — not a transient.

### 11.6 Verdict on 24/33/34

**No — the Y-path solver is not converging to its own Y-consistent Hugoniot, and the reason is a
solver defect, not a modelling difference.** Three findings, in increasing severity:

1. The alpha-held reference is an exact Rankine-Hugoniot solution of the true NASG mixture EOS
   (§11.3). It is a legitimate reference; §7.1's "the reference defines the answer" framing was
   too generous to the Y path.
2. The cases *are* configured against the alpha closure at a level §7.1 did not identify: the IC
   and the sustained inflow are a closure-(A) state, which under Y-transport injects a 270x-1620x
   contact in the transported variable and overdrives the shock (§11.4). Any comparison of
   post-shock plateaus on these cases is confounded by that, in either direction.
3. **But the RH residual test is immune to all of it**, and the Y path fails it by O(1) (§11.5).
   A correct scheme must produce admissible weak solutions from *any* initial data. It does not.

The mechanism is the §2.3 design compromise, now shown to bite far harder than §7.2 suspected:
`alpha` is recovered at the OLD `(p,T)` and then **frozen through the Newton solve**, so the EOS
closure `rho = rho(p,T,alpha)` used inside the implicit mass/momentum/energy update is
inconsistent with the `Y` that was actually transported. Across these shocks `alpha` moves by
three orders of magnitude in a single step, so the lag is O(1) and discrete conservation breaks
by O(1). §7.2 diagnosed this as an amplitude-convergence failure on case15; §11.5 shows it is a
**conservation** failure wherever `d(alpha)/dp` is large. The two failures have one cause.

Nothing was changed to make any of this pass: no gate, no reference, no case definition.

---

## 12. Revised verdict

1. **The default path remains bit-preserved.** `ACID_YADV` unset reproduces the published
   `solver_denner` binary byte-for-byte on 9/9 dumps at 19/19 PASS, re-verified after every edit
   in this round on a fully clean rebuild.
2. **Task A is a negative result and the hypothesis behind it is refuted.** Reconstructing the
   THINC tanh in `alpha` and converting the face value to `Y` makes case02 3.6x worse
   (`l2_rho` 0.0438 -> 0.1579), does not move case14 at all, and costs case13 its exact
   shock-front placement. Dropping the conversion is worse again (0.3058, case02 FAILS). The
   real defect is that the Y path's cell state is a **mass** average while its transport stencil
   is the update for a **volume** average — a cell-state/stencil mismatch that no face map can
   repair (§10.4).
3. **Task B overturns §7.1.** The alpha-held reference is an exact RH solution of the true NASG
   mixture EOS, and the Y-path solver violates momentum by 88% and energy by 65% across its own
   shock. Cases 24/33/34 are a **solver defect**, not an unanswerable closure conflict.
4. **The case15 stiffness failure and the 24/33/34 conservation failure have a single cause**:
   `alpha` recovered at the old `(p,T)` and frozen through the Newton.
5. `ACID_YADV` should stay **default OFF and unpromoted**. The case13 win (§6) is real but it is
   the only one, and it now sits next to a demonstrated O(1) conservation failure elsewhere.

Ranked next experiments, both of which are rewrites rather than tweaks:

* **(a) Conservative `rho*Y` transport.** Make the conserved variable `rho*Y` with
  `d(rho Y)/dt + d(rho Y u)/dx = 0`, so cell state and flux occupy the same space. This is the
  only route that addresses §10.4, and it would also let the reconstruction question be asked
  again cleanly.
* **(b) `alpha` inside the Newton.** Add `d(alpha)/dp`, `d(alpha)/dT` to the analytic Jacobian so
  `alpha` is an implicit function of the unknowns. This is what §11.6 and §7.2 both point at.

Either one alone is insufficient: (a) without (b) leaves the frozen-`alpha` conservation defect,
(b) without (a) leaves the averaging mismatch.

---

## 13. Round 2 reproduction

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
bash scripts/yadv_clean_build.sh          # rm -rf build-cpp, reconfigure, build, unit test

python3 scripts/yadv_verify.py            # default path byte-identity vs the published binary
bash    scripts/yadv_run_ab.sh v2         # OFF + ON validate sweeps -> /tmp/yadv_v2_{off,on}.txt
python3 scripts/yadv_table3.py            # three-way A/B table of section 10.2
                                          #   (needs /tmp/yadv_v1_on.txt from a round-1 build)

python3 scripts/yadv_case02_probe.py      # section 10.3 interface band
bash    scripts/yadv_thincdbg.sh          # section 10.3 THINC activation counts
bash    scripts/yadv_volflux.sh           # section 10.4 ACID_YADV_VOLFLUX probe

bash    scripts/yadv_hugoniot.sh          # section 11.2/11.3 two-closure table (standalone)
python3 scripts/yadv_plateau.py           # section 11 plateau vs both closures
python3 scripts/yadv_yprofile.py          # section 11.4 reconstructed Y field
python3 scripts/yadv_struct.py            # section 11.4 wave structure
python3 scripts/yadv_front.py             # section 11.4 front positions
python3 scripts/yadv_rhcheck.py           # section 11.5 THE decisive RH residual test
```

`ACID_YADV_VOLFLUX` is a **diagnostic probe only** (§10.4): default OFF, inert unless `ACID_YADV`
is also set, and it does not touch the published path.

---
---

# ROUND 3

## 14. Task (a) — conservative `rho*Y` transport. Partial positive, §10.4's target still not met.

### 14.1 What was implemented

`cpp/denner_1d/src/acid.cpp`, the `yadv` branch of the colour-function transport block only (the
`!yadv` / default alpha branch is byte-for-byte the same three lines as rounds 1-2, reverified
below). Makes `rho*Y` the transported conserved quantity instead of advecting `Y` with the
volume-average stencil (rounds 1-2's mismatch, §10.4):

1. an OLD-level cell `alpha` is recovered from `Yv` at `(p_o,T_o)` — used only as the ACID
   mass-flux blend weight, not written to `s.alpha`;
2. OLD-level upwind phase densities `ra_o[f]`/`rb_o[f]` are evaluated at **every** face (not only
   where THINC fires), upwind selection by the sign of `thf[f]`;
3. an OLD-level per-cell mass flux is built mirroring ACID Eqs.41-42 **exactly** (cell `i`'s own
   `alpha` as the blend weight, upwind-cell phase densities, `thf` as the face velocity) — the
   same asymmetric per-cell (not per-face) construction the implicit residual uses;
4. `(rho*Y)_i = rho_old*Y_i - dt/dx*(mdotR_o*Yface_R - mdotL_o*Yface_L)`, then
   `Y_new = clamp((rho*Y)_i / rho_star, 0, 1)`, where `rho_star = rho_old - dt/dx*div(mdot_o)` —
   the density the SAME old-level mass flux predicts by discrete continuity. This is explicit and
   introduces no circularity (it is not task (b)'s alpha-in-Newton problem).

**`rho_star`, not `rho_old`, is required — measured, not a style choice.** Dividing by `rho_old`
breaks the pure-cell invariant: with `Y==1` everywhere the numerator reduces to `rho_star`, so
dividing by `rho_old` instead returns `1 - dt/(dx*rho_old)*div(mdot)` — a compressed or expanded
**single-phase** cell spontaneously grows the other phase. Measured: suite 13/19 (case13 `l2_rho`
0.1219 vs 0.0227, `max|d alpha|` vs the alpha path 0.998 vs 0.070; case30 FAILS, `l2_rho` 0.1243
vs 0.0092). Reachable for reproduction behind `ACID_YADV_RHOOLD=1` (default OFF, inert unless
`ACID_YADV` is also set, never touches the published path).

`af[]`/`thf[]`/THINC/the rho-monotonicity guard are untouched — the reconstruction stays in
`Y`-space (round 2 already refuted alpha-space reconstruction, §10). `beta=3.5` remains the only
tunable constant in the subsystem. `cases.cpp`/`validation.cpp` untouched.

### 14.2 Verified independently (clean rebuild, not just the implementer's report)

```
denner1d_unit ok
OFF: DENNER1D_CPP_METRIC pass_count=19 total=19
(1) ACID_YADV unset vs solver_denner published binary: case01..case34 BYTE-IDENTICAL (9/9)
ON:  DENNER1D_CPP_METRIC pass_count=15 total=19   (fails: 15, 24, 33, 34 -- same four as rounds 1-2)
```

### 14.3 Before (round-2 v1) vs after (round-3 conservative `rho*Y`)

| case | metric | v1 (round 1, Y-space non-conservative) | **v3 (conservative rho\*Y)** |
|---|---|---|---|
| 02 | l2_rho / corr_rho | 0.04378 / 0.9940 | 0.05559 / 0.99036 (worse) |
| 13 | l2_p / l2_u / linf_p / linf_u | 0.01711 / 0.04247 / 0.1690 / 0.4376 | 0.02073 / 0.05000 / 0.25065 / 0.62873 (win shrinks) |
| 13 | `peak_delta_u` (shock front, cells) | **0** | **397 — v1's exact front placement LOST** |
| 14 | l2_rho / corr_rho | 0.07636 / 0.9715 | 0.07691 / 0.97177 (no recovery) |
| 15 | `amp_ratio_p` / `l2_p` | 0.330034 / 0.166533 | **0.330034 / 0.166533 — bit-identical to v1** |
| 24 | l2_p / corr_rho | 1.123 / 0.4072 | 0.83737 / 0.16354 |
| 33 | l2_p / corr_rho | 1.574 / 0.5081 | 1.57341 / 0.49358 |
| 34 | l2_p / corr_rho | 0.7831 / 0.4460 | 0.83916 / 0.16166 |

case15's bit-identical reproduction of v1 is expected and diagnostic: in a uniform-`Y` field both
stencils reduce to the same update, so its failure is orthogonal to the flux form and confirms
§7.2's frozen-alpha-through-Newton mechanism, unaffected by this change.

**The one new result — the §11.5 Rankine-Hugoniot residual test**, rerun unmodified
(`scripts/yadv_rhcheck.py`), independently reproduced:

| case | path | momentum resid (rel) | energy resid (rel) |
|---|---|---|---|
| 24 | alpha | -2.48e-06 | +1.81e-06 |
| 24 | **Y (v3)** | **+1.32e-13** | **-1.61e-12** |
| 33 | alpha | -2.91e-05 | +5.83e-05 |
| 33 | **Y (v3)** | **+8.81e-01** | **+6.46e-01** |
| 34 | alpha | -1.59e-06 | +3.82e-07 |
| 34 | **Y (v3)** | **+7.16e-13** | **-1.01e-12** |

Under round-1 v1, cases 24/34's leading shock had left the domain entirely (§11.5 could not even
measure it). Under v3 it is back in-domain and closes Rankine-Hugoniot to machine precision —
conservative transport genuinely fixed the discrete conservation law there. **Case33 is
unchanged: still an 88%/65% momentum/energy violation.** The O(1) conservation defect of §11.6 is
therefore *partially* closed by (a) alone, not closed — exactly the outcome §12 predicted
("(a) without (b) leaves the frozen-alpha conservation defect").

### 14.4 Verdict

1. **§10.4's target is refuted again.** Conservative `rho*Y` does not recover case02/14 (both
   flat-to-worse) and costs case13 its exact shock-front placement, the round-1 headline result.
   The mass/volume cell-state mismatch was a real defect and this is the structurally correct fix
   for it, but the THINC face reconstruction is a separate source of the case02/14 residual that
   this change does not touch.
2. **Genuine partial win on discrete conservation** (§11.5): cases 24/34 now satisfy their own
   leading-shock RH jump to 1e-13; case33 does not move. Consistent with the single-cause
   diagnosis of §11.6/§7.2 — the frozen-`alpha`-through-Newton lag still dominates wherever it is
   active on a strong shock (case33 being the sharpest of the three, `alpha_pre=0.75`).
3. **`ACID_YADV` stays default OFF, unpromoted.** Round 3 does not clear the bar. The single
   remaining ranked item from §12 is task (b): `alpha` inside the Newton Jacobian
   (`d(alpha)/dp`, `d(alpha)/dT`), which both §7.2 (case15) and §11.6 (24/33/34) independently
   point at as the one remaining cause neither round 1/2's reconstruction fix nor round 3's flux
   fix could reach.

### 14.5 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
bash scripts/yadv_r3_build.sh    # clean rebuild + unit test
bash scripts/yadv_r3_ab.sh       # OFF + ON validate sweeps
python3 scripts/yadv_verify.py   # byte-identity vs published binary
python3 scripts/yadv_rhcheck.py  # section 14.3 RH residual table
```

---
---

# ROUND 4

## 15. Task (b) — `alpha` inside the Newton, phase 1. Mixed: cures case15's convergence defect, regresses the analytic-Jacobian default. Gated OFF.

### 15.1 What was implemented

`cpp/denner_1d/src/acid.cpp`, inside `compute_R` (the residual lambda called every Newton
iteration, every Jacobian probe, every line-search trial, every TR-BDF2 stage): as the FIRST
statement, re-derive `s.alpha[i]` from the transported `Yv[i]` at the CURRENT trial `(s.p[i],
s.T[i])`, instead of leaving it frozen at the pre-Newton `(p_o,T_o)` value rounds 1-3 used. This
makes `alpha = alpha(Y,p,T)` an implicit function of the Newton unknowns by RE-EVALUATION —
leaning on the defect-correction principle already load-bearing elsewhere in this solver
(`.claude/rules/denner-pitfalls.md`: an approximate/frozen Jacobian changes only iteration count,
never the converged solution, provided Newton converges) — rather than by deriving explicit
`d(alpha)/dp`, `d(alpha)/dT` analytic Jacobian rows. On the first `compute_R` call each step
`s.p`/`s.T` still equal `p_o`/`T_o`, so this reproduces the existing pre-Newton recovery exactly;
no discontinuity at start. The pre-Newton recovery loop and the `rho_o`/`hstat_o`/`Htot_o`
old-level transient baseline (unchanged since round 1) are untouched — they still use the
pre-Newton alpha, as required.

**Gated behind a NEW flag, `ACID_YADV_ALPHA_IMPLICIT` (default OFF), separate from `ACID_YADV`.**
Measured (below): this is a net REGRESSION under the default analytic Jacobian and only partially
positive even with the FD Jacobian forced. Plain `ACID_YADV=1` therefore continues to reproduce
the already-committed, already-documented round-3 behaviour (`pass_count=15`) unchanged —
re-verified bit-for-bit after adding the gate. `cases.cpp`/`validation.cpp` untouched; no new
per-case branch; OFF path re-verified 19/19, 9/9 byte-identical against the published binary.

### 15.2 Measurements (independently reproduced, not just the implementer's numbers)

```
OFF                                   : 19/19, 9/9 byte-identical (unchanged)
ON  (ACID_YADV=1)                     : 15/19  -- unchanged from round 3
ON  + ACID_YADV_ALPHA_IMPLICIT=1      : 12/19  -- fails 13, 14, 15, 24, 25, 33, 34
ON  + ALPHA_IMPLICIT + ACID_NO_AJAC=1 : 12/19  -- fails 14(NaN), 15, 24, 27, 28(NaN), 33(NaN), 34(NaN)
```

The `ACID_NO_AJAC=1` (FD Jacobian) comparison is **not a neutral diagnostic on this solver** and
must be read against its own control: forcing the FD Jacobian on the *default alpha path* (no
`ACID_YADV` at all) drops it from 19/19 to **13/19** (fails 15, 24, 27, 28-NaN, 33-NaN, 34) —
the FD path itself costs 6 cases here, unrelated to Y-transport. Corrected comparison, same
Jacobian both sides:

| Jacobian | OFF (alpha) | ON, frozen alpha (round 3) | ON, implicit alpha (round 4) |
|---|---|---|---|
| analytic (default) | 19/19 | 15/19 | **12/19** (new fails: 13,14,25) |
| FD (`ACID_NO_AJAC=1`) | 13/19 | 13/19 | 12/19 (new fail vs FD baseline: 14 only) |

### 15.3 The core hypothesis of §7.2/§11.6 IS confirmed — case15

| config | `amp_ratio_p` | `l2_p` | `corr_p` | `corr_rho` |
|---|---|---|---|---|
| round 3 (frozen alpha), analytic | 0.330034 | 0.166533 | 0.98554 | 0.98451 |
| round 3 (frozen alpha), FD | 0.353989 | 0.162352 | 0.98723 | 0.98626 |
| round 4 (implicit alpha), analytic | 1.23223 | 0.230851 | 0.09937 | 0.51520 |
| **round 4 (implicit alpha), FD** | **1.00041** | **0.014393** | **0.99929** | **0.99673** |

Neither half alone fixes it — FD alone barely moves `amp_ratio_p` (0.330→0.354); implicit-alpha
alone overshoots catastrophically (`corr_p` collapses to 0.099). **Together**, the grid-refinement
amplitude-loss failure §7.2 diagnosed is gone: case15 satisfies every quantitative gate criterion
(`corr_p≥0.93`, `corr_u≥0.998`, `corr_rho≥0.99`, `l2_p≤0.18`, `l2_u≤0.06`, `l2_rho≤0.05`). It
still reports `pass:false`, but now only on the spec's velocity-smoothness/TV-oscillation guards
(`peak_delta_u=321`, a materially different and narrower failure than non-convergence).

### 15.4 The analytic Jacobian's blind spot is now load-bearing — cases 13/14/25

Implicit alpha under the DEFAULT (analytic) Jacobian breaks three cases that round 3 passed:
case13 (`l2_p` 0.0207→0.0383, `u_shock_delta_cells` 1→5), case14 (`corr_u` 0.982→0.955, `l2_u`
0.084→0.132), case25 (`corr_p` 0.994→**-0.123**, catastrophic). Under the FD Jacobian, 13 and 25
both PASS (case13 `l2_p=0.01722`, essentially restoring round-1 v1's `linf_p=0.169`; case25
`corr_p=0.991`). This is a textbook bad-search-direction symptom, not a residual/physics failure:
per defect-correction, the converged answer should be Jacobian-independent, so an analytic-only
regression means Newton is not converging there — exactly the risk flagged before this experiment
ran. **Case14 is the exception**: it degrades under the analytic Jacobian and goes fully NaN under
FD — worse either way, not explained by the Jacobian-accuracy story, not investigated further.

### 15.5 Cases 24/33/34 — not recovered, no worse under analytic, worse under FD

Case24: analytic `l2_p=0.837→0.837` (no change), `corr_p=0.166→0.167` (no change) — essentially
untouched by this experiment either way. Case33/34: unchanged under analytic, but go fully NaN
under FD (round 3 + FD already NaNs 33/34 too, so this is not new to round 4 — see the OFF/round-3
FD control above). No wall-clock blowup: implicit-alpha costs nothing under the analytic Jacobian
(actually slightly faster, partly from earlier aborts on the newly-failing cases); the FD Jacobian
costs its usual ~1.7-1.9x regardless of Y-transport.

### 15.6 Verdict

1. **Phase 1's core hypothesis is confirmed for case15**: freezing `alpha` through the Newton was
   exactly the mechanism §7.2 diagnosed, and re-deriving it at the current iterate — with no
   analytic Jacobian surgery at all — cures the grid-convergence failure, PROVIDED the Jacobian is
   accurate enough (FD) to find the right search direction.
2. **The analytic Jacobian's `d(alpha)/dp`/`d(alpha)/dT` blind spot, predicted as a risk before
   this ran, is now measured and load-bearing**: three previously-passing cases (13/14/25) regress
   under it. Phase 2 — the actual `d(alpha)/dp`, `d(alpha)/dT` analytic Jacobian rows §12 named —
   is no longer an optional refinement; it is required before implicit alpha can be the default
   for `ACID_YADV=1` at all.
3. **24/33/34 remain unrecovered under either Jacobian.** Implicit alpha alone does not close
   their conservation defect (§11.6) the way it closes case15's amplitude defect — a different
   failure mode within the same "frozen alpha" root cause, still open.
4. **`ACID_YADV_ALPHA_IMPLICIT` stays default OFF, `ACID_YADV=1` alone is unaffected** (re-verified
   bit-identical to round 3, 15/19). Recommendation: the next experiment is the analytic
   `d(alpha)/dp`/`d(alpha)/dT` Jacobian contribution itself (Phase 2) — the FD-Jacobian result
   here is existence proof that an accurate-enough Jacobian recovers 13/15/25 without touching the
   physics, so Phase 2 is worth attempting before concluding task (b) is a dead end.

### 15.7 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
bash scripts/yadv_r3_build.sh
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate                                   # OFF: 19/19
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate                       # ON: 15/19 (round 3, unchanged)
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ./build-cpp/cpp/denner_1d/denner1d_validate                  # 12/19
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1 ./build-cpp/cpp/denner_1d/denner1d_validate   # 12/19, different failure set
DENNER_ACID=1 ACID_NO_AJAC=1 ./build-cpp/cpp/denner_1d/denner1d_validate                                          # FD control on the default alpha path: 13/19
python3 scripts/yadv_verify.py
```

---
---

# ROUND 6 (first round run under the `yadv-round` autonomous loop)

## 16. Phase 2 Stage 0 -- alpha derivative helpers (additive), and Stage 1 -- analytic Jacobian, p-pathway. Genuine success: 12/19 -> 14/19, case13 and case25 recover.

### 16.1 Stage 0 (round 5)

Added header-inline `alpha_derivs_massfrac`/`dalpha_dp_massfrac`/`dalpha_dT_massfrac` to `eos.hpp`
(no call sites yet) plus a `denner1d_unit.cpp` block verifying: central-FD agreement, the exact
mixture-compressibility identity `D_p + (rho_a-rho_b)*a_p == rho*(alpha*zeta_a/rho_a +
(1-alpha)*zeta_b/rho_b)`, the `a_T` exact-zero property for `b=0` phase pairs (to a measured
cancellation floor, not bitwise -- `phase_props` does not round-trip `ppinf`/`A` exactly), and
Phase-2 §1's numeric prediction at case15's state: **measured ratio 521.558, predicted ~500,
confirmed.**

Found and fixed a bug in the new unit test itself (not the derivative formula): the FD-comparison
tolerance floor for near-algebraic-zero cases (the air|vapor pair shares `pinf=0, b=0`, so
`zeta/rho` is identical between the phases and `a_p` vanishes exactly) was multiplied by an extra
`1e-6`, making the tolerance ~1e6x too strict and failing 581 checks. The derivative formula was
independently confirmed correct throughout via a standalone probe program (matches FD to full
double precision everywhere real signal exists, e.g. every air|water combination). Fixed by using
the roundoff floor directly as the absolute tolerance rather than shrinking it further.

All four gates unchanged, as required for a purely additive stage: OFF 19/19+9/9, `ACID_YADV=1`
15/19, `+ALPHA_IMPLICIT` 12/19 under both Jacobians, identical failure sets to round 4.

### 16.2 Stage 1 -- the first analytic-Jacobian edit in this whole experiment

Augmented the existing J1 cell-EOS-chain loop (`acid.cpp`, the analytic Jacobian's per-cell
`D,D_T,D_p,N,N_T,N_p,hsT,hsp,dTp/dTh/dTu,drp/dru/drh` block): under `yadv && alpha_implicit`, star
`D_p`/`N_p` with the product-rule addend from `a_p = d(alpha)/dp|_{T,Y}` (Stage 0's already-derived,
already-unit-tested helper):

```cpp
D_ps = D_p + (rho_a - rho_b) * a_p
N_ps = N_p + (rho_a*h_a - rho_b*h_b) * a_p
hsp  = (N_ps*D - N*D_ps) / D^2      // was (N_p*D - N*D_p)/D^2
drp  = D_ps + D_T * dTp             // was D_p + D_T * dTp
```

The T-pathway (`D_T`, `N_T`, `hsT`) is deliberately untouched -- the residual's alpha is lagged one
`compute_R` call in T (§0.4), so the frozen-T derivative is the exact derivative of the coded map;
starring T is the contingent Stage 3. The `yadv && alpha_implicit` ternary makes the unstarred
branch a bit-copy of the existing expressions, so the OFF path and plain `ACID_YADV=1` are
byte-unchanged by construction, not by floating-point luck.

### 16.3 Gates (all held)

```
OFF (ACID_YADV unset)                                  : 19/19, 9/9 byte-identical (unchanged)
ACID_YADV=1 (plain)                                     : 15/19 (unchanged, case01 dump byte-identical)
ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1   : 12/19, EXACT SAME failure set as round 4/5
                                                           (14,15,24,27,28,33,34) -- Stage 1 only
                                                           touches code the FD path never executes
```

### 16.4 The target measurement -- genuine success

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1`, default analytic Jacobian:

| | round 4/5 (frozen J1) | **round 6 (Stage 1)** |
|---|---|---|
| `pass_count` | 12/19 | **14/19** |
| failure set | 13,14,15,24,25,33,34 | **14,15,24,33,34** |
| case13 | FAIL, `l2_p=0.0383` | **PASS**, `l2_p=0.0313`, `corr_p=0.994` |
| case25 | FAIL, `corr_p=-0.123` | **PASS**, `corr_p=0.991` |
| case15 `amp_ratio_p` | 1.23223 | **1.00042** (target ~1.0 -- exact) |
| case15 `corr_p` | 0.09937 | **0.999285** |
| case15 | FAIL (non-convergence) | FAIL, but **every quantitative gate criterion now passes**
  (`corr_p/u/rho`, all `l2_*`) -- blocked only by the `smooth_ok`/`osc_ok` TV-jump guards, a
  narrower and different failure than round 4's non-convergence diagnosis. `peak_delta_u` moved
  321 (round-4 FD) -> **0** this round, a strong signal but not independently confirmed against
  the exact `p_osc`/`r_osc` thresholds this round. |
| case14 | FAIL | FAIL, unchanged in kind (round 5's separate `hsT<0` lead, out of scope) |
| case24/33/34 | FAIL, conservation defect | FAIL, unchanged in kind (explicit non-goal, §11.6/§15.5) |

Success bar was `pass_count >= 13`; achieved **14**, with case13 AND case25 both fully recovering
(the plan flagged this as possible but uncertain) and case15 moved to within one narrow gate
criterion of passing outright.

### 16.5 Verdict

1. **The analytic-Jacobian blind spot diagnosed in round 4 (§15.4) is now closed for the p-pathway
   on cases 13/25**, and case15's amplitude defect (§7.2's original diagnosis) is closed to within
   a TV/oscillation guard. A single, small, `const`-only, sign-provable diff recovers what round 4
   needed the FD Jacobian (and its ~2x cost, and its own case14 NaN) to achieve for 13/25, and does
   *better* than the FD result for case15 (`peak_delta_u` 0 vs the FD run's residual 321).
2. **Cases 24/33/34 remain unrecovered, exactly as predicted** -- their defect is the conservation
   failure of §11.6, orthogonal to Jacobian accuracy. Not chased this round (non-goal).
3. **Case14 remains unrecovered** and was not expected to move -- round 5 traced its risk to
   `hsT < 0` at its own initial state (the h->T inversion itself is locally ill-posed there), a
   separate defect from the one Stage 1 targets.
4. This is the first round run end-to-end under the `yadv-round` autonomous loop (see
   `docs/YADV_ROADMAP.md`). The Advisor spot-checked the Planner's line-anchor claims against the
   live code before implementing, per the loop's own protocol, and found zero drift.

Recommendation: proceed to Phase-2 Stage 2 (the J2 flux-blend diagonal, `alp_p[]` already stored
and waiting) -- the plan predicts this is where the water/air density-ratio factor
`(rho_a-rho_b)~-1000` most directly enters the mass/energy flux rows, and is the strongest
remaining candidate for closing case15's TV guard and for any further movement on 14/24/33/34.

### 16.6 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit                                              # includes Stage-0 checks + diagnostic prints
DENNER_ACID=1                                          ./build-cpp/cpp/denner_1d/denner1d_validate   # 19/19
DENNER_ACID=1 ACID_YADV=1                              ./build-cpp/cpp/denner_1d/denner1d_validate   # 15/19
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1                ./build-cpp/cpp/denner_1d/denner1d_validate   # 14/19 (Stage 1)
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1 ./build-cpp/cpp/denner_1d/denner1d_validate   # 12/19, unchanged
python3 scripts/yadv_verify.py
```

---
---

# ROUND 7

## 17. Phase 2 Stage 2 -- J2 flux-blend diagonal. Measured no-op on pass_count; corrects round 6's case15 diagnosis.

### 17.1 Correction to §16.4

Round 6 read case15's `peak_delta_u` moving 321 (round-4 FD) -> 0 (round-6 analytic+Stage-1) as a
sign case15 was "close to passing, blocked only by the TV/oscillation guard." **`peak_delta_u` is
not part of case15's gate.** `validation.cpp`'s case15 contract (~lines 684-730) is
`corr_p>=0.93 && corr_u>=0.998 && corr_rho>=0.99 && l2_p<=0.18 && l2_u<=0.06 && l2_rho<=0.05 &&
smooth_ok && osc_ok`, where `smooth_ok`/`osc_ok` are a domain-restricted jump/concentration test
and a total-variation-excess test respectively -- `peak_delta_u` appears only in the unrelated
generic `default_pass` and in `metrics_json`'s diagnostic output. §16.4's inference was
unsupported; this section replaces it with the measured values.

### 17.2 What was implemented

Added a new diagonal loop to the analytic Jacobian: the other product-rule addend of the ACID
per-cell flux blend `mdot_f^(i) = (al_i*raup_f + (1-al_i)*rbup_f) * theta_f` -- the sensitivity of
the blend weight `al_i` itself (own cell only, reusing Stage 1's already-computed `alp_p[]`):

```
d(R_con)/dp |_i += ((raup[i+1]-rbup[i+1])*theta[i+1] - (raup[i]-rbup[i])*theta[i]) * alp_p[i]
d(R_mom)/dp |_i += (that * uconv, R and L)                                          * alp_p[i]
d(R_ene)/dp |_i += ((rHaup[i+1]-rHbup[i+1])*theta[i+1] - (rHaup[i]-rHbup[i])*theta[i]) * alp_p[i]
```

Purely additive (no existing block differentiates `al` itself -- the flux-coupling block freezes
`al` and differentiates `theta`/`pface`; the upwind-transport block freezes `al` and differentiates
the upwind cell's `raup`/`rbup`). Boundary correctness verified: `theta[]` already carries every BC
override before the mdot loop runs, so no special-casing was needed.

### 17.3 Gates (all held)

```
OFF (ACID_YADV unset)                                : 19/19, 9/9 byte-identical
ACID_YADV=1 (plain)                                   : 15/19, unchanged
+ALPHA_IMPLICIT ACID_NO_AJAC=1 (FD-invariance)        : 12/19, EXACT SAME failure set as rounds 4-6
```

### 17.4 The target measurement -- unchanged pass_count, one precise new finding

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1`: **14/19, identical failure set** `{14,15,24,33,34}` to
round 6. case15's metrics moved by noise only (`amp_ratio_p` 1.00042->1.00041). Consistent with the
round-7 Planner's own prediction: cases 13/25 were already fully recovered in Stage 1 and had
nothing left for J2 to add; J2's magnitude (a sub-percent correction at case15's state, by rough
order-of-magnitude estimate) is much smaller than Stage 1's ~500x fix. Reported as a measured
no-op, not spun as a win.

**case15's true blocker, computed exactly from `validation.cpp`'s own formulas against a fresh
dump:**

```
cj=30.02   cj_r=3.55   threshold=max(8.0,1.10*cj_r)=8.0     -> FAILS (central jump ~4x the limit)
mj=32.00   mj_r=18.08   threshold=max(8.0,1.10*mj_r)=19.88   -> FAILS
cc=0.117   cc_r=0.084   threshold=max(0.04,1.10*cc_r)=0.093  -> FAILS
smooth_ok = False
p_osc=0.0, r_osc=0.0  -> osc_ok = True  (total-variation-excess side is completely clean)
```

case15 fails on the **central-jump/concentration test**, not oscillation. `cj` is the jump at the
exact domain center (`x[nn/2]` vs `x[nn/2-1]`), which by problem symmetry is `u`'s stagnation
point (`u=0` at `x=0.5` for a symmetric double rarefaction). A large central jump at a stagnation
point in a collocated pressure-velocity coupled solver is a known failure-mode CLASS (checkerboard/
central-spike artifacts), structurally unrelated to `d(alpha)/dp` Jacobian accuracy. There is no
mechanism by which Stage 1, Stage 2, or the contingent Stage 3 (T-pathway) could reach this defect.

### 17.5 Verdict

1. Stage 2 correctly implemented, all gates held, measured no-op on `pass_count` -- an honest
   negative result, not chased or spun.
2. **case15's actual blocker is now understood and it is out of scope for this Phase-2 plan.**
   Recovering it would require investigating the central-stagnation-point discretization (a
   scheme/MWI question), not more alpha-implicit Jacobian work.
3. Cases 24/33/34 and 14 unmoved, exactly as predicted (separate, previously-diagnosed defects).
4. Two stray tracked files (`3,`, `=150`) at the repo root, committed long before this round loop
   existed (`325dc5b`), noted but not touched -- out of this round's scope.

Recommendation: re-scope Phase-2's "current goal" (`docs/YADV_ROADMAP.md`) rather than treat
case15 as a Stage-3 target. 13/25 recovery is a genuine, durable win; case15 needs a different
investigation entirely if pursued further.

### 17.6 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ./build-cpp/cpp/denner_1d/denner1d_validate   # 14/19 (unchanged from round 6)
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ./build-cpp/cpp/denner_1d/denner1d_dump 15 > /tmp/case15.csv
# then compute cj/mj/cc/p_osc/r_osc exactly per validation.cpp's smooth_ok/osc_ok formulas (§17.4)
python3 scripts/yadv_verify.py
```

---
---

# ROUND 8

## 18. Phase 2 Stage 3a -- T-pathway. Measured regression on case14, gated behind a new flag. A genuine closed-form discovery along the way.

### 18.1 Measure-first: was the round-5 `hsT<0` lead for case14 even real?

Round 5 flagged `hsT = d(hstat_mix)/dT|_{p,alpha} < 0` at a PROBE state (`p=1e5, T=6.94K,
alpha=0.5`) as a case14 root-cause candidate. Static analysis this round showed `hsT<0` requires
`T < 78.2K` uniformly over the air|water pair's `(p,Y)` range, and every physically-reachable
mixture of case14's two IC states sits >= 4.5x above that bound -- the probe's `alpha=0.5` is a
state no case14 cell actually occupies. Rather than declare the lead dead by inference, a
temporary default-off diagnostic (`hsT`'s sign, sampled in the J1 loop, gated `yadv &&
alpha_implicit`, removed after use) was run against case14 and case15 before writing any
production code.

**Result: `hsT<0` is real, but confined entirely to case14's very first timestep's Newton
iterations** (a single interface-adjacent cell, `T` pinned near a spurious near-zero root of the
non-monotone `h(T)` equation), and never recurs at any later step. This is a transient
interface-formation artifact, not a persistent state the solution trajectory revisits -- the
condition under which Stage 3a (a Jacobian-only fix) was worth attempting, per the round's own
decision rule.

### 18.2 What was implemented

Starred the T-pathway (`D_T -> D_Ts`, `N_T -> N_Ts`, via `a_T = d(alpha)/dT|_{p,Y}`) in the J1
loop, mirroring Stage 1's p-pathway exactly, and computed the TOTAL derivatives (not partial)
`alp_p = a_p + a_T*dTp`, `alp_h = a_T*dTh`, `alp_u = a_T*dTu` -- `dTp/dTh/dTu` already encode the
h->T inversion's own sensitivity, so these are complete once computed AFTER them (the ordering
dependency the round's plan flagged in advance). Extended the J2 diagonal loop with two more
columns (h, u) mirroring the existing p-column exactly.

### 18.3 A genuine mathematical discovery, independent of whether Stage 3a helps

`hstat_mix = N/D` is EXACTLY `Y*h_a(p,T) + (1-Y)*h_b(p,T)` (an identity of how `Y` is defined, not
dependent on any of this round's derivative work), and NASG `h_k` is linear in both `T` and `p`.
Consequently the starred partials have exact closed forms:

```
hsT* = Y*cp_a + (1-Y)*cp_b     -- strictly positive, bounded in [min cp, max cp], NEVER crosses 0
hsp* = Y*b_a  + (1-Y)*b_b
D_p* = rho^2*(Y*zeta_a/rho_a^2 + (1-Y)*zeta_b/rho_b^2)
D_T* = rho^2*(Y*phi_a /rho_a^2 + (1-Y)*phi_b /rho_b^2)
```

Verified in the unit test to 6.8e-11 absolute. These retroactively validate Stage 1's already-
shipped `hsp*`/`D_p*` (never checked this way before) and prove that starring the T-pathway
REMOVES an existing `1/hsT` near-singularity rather than introducing one -- the unstarred `hsT`
provably crosses zero for the air|water pair below ~78K; the starred form is provably bounded away
from zero everywhere.

**A live bug in the round's own new unit test was found and fixed along the way**: the initial
identity check divided by a `1e-300` floor when comparing against a LEGITIMATELY zero closed form
(`hsp_closed=0` for the air|vapor pair, both `b=0`), blowing a genuine ~1e-19 roundoff difference
into a reported error of `1.79e+287` -- the exact same class of bug round 5's own unit test had.
Fixed with an absolute-or-relative combined tolerance. The derivative formula itself was correct
throughout; only the test's comparison logic was wrong.

### 18.4 The measurement: a real regression on case14, not a null result

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` alone (Stage 1+2, unaffected): 14/19, bit-identical to
rounds 6/7. Adding Stage 3a's T-pathway UNCONDITIONALLY does not flip case14's pass/fail (already
failing either way) but its solution quality collapses:

| metric | round 6/7 (Stage 1+2) | round 8 (+Stage 3a) |
|---|---|---|
| `l2_p` | 0.0144718 | 0.511828 |
| `l2_u` | 0.132392 | 0.663105 |
| `corr_p` | 0.99956 | 0.594481 |
| `corr_u` | 0.954309 | 0.227335 |
| `corr_rho` | 0.979441 | 0.746994 |
| `amp_ratio_u` | 1.1221 | 4.59619 |

This confirms a risk flagged BEFORE the round's implementation began: giving the Jacobian the
FIXED-POINT T-derivative while the residual still computes the ONE-CALL-LAGGED map (alpha is
re-derived from Y at the current iterate's p, but at the PREVIOUS iterate's T, before the h->T
inversion runs) is a family mismatch -- the mirror image of round 4's original mistake (there, a
fully-nonlinear residual paired with a zero-derivative Jacobian; here, a Jacobian assuming a
derivative family the residual does not itself evaluate).

### 18.5 Gating decision

Because this sits inside the already-established, already-validated `ACID_YADV_ALPHA_IMPLICIT`
flag (round 6/7's genuine win), merging Stage 3a unconditionally would silently degrade that
configuration. Gated behind a NEW flag, `ACID_YADV_ALPHA_IMPLICIT_T` (default off) -- the same
precedent round 4 set for its own mixed result. Verified: `ACID_YADV_ALPHA_IMPLICIT=1` alone
reproduces round 6/7's case14 metrics bit-for-bit; adding `ACID_YADV_ALPHA_IMPLICIT_T=1`
reproduces the regression table above exactly.

### 18.6 Gates (all held)

```
OFF (ACID_YADV unset)                          : 19/19, 9/9 byte-identical
ACID_YADV=1 (plain)                             : 15/19, unchanged
+ALPHA_IMPLICIT ACID_NO_AJAC=1 (FD-invariance)  : 12/19, EXACT SAME failure set as rounds 4-7
+ALPHA_IMPLICIT (new T-flag OFF)                : 14/19, BIT-IDENTICAL to round 6/7
```

### 18.7 Advisor decision on Stage 3b -- escalated by the round, declined for now

The round's plan surfaced a strong argument for Stage 3b (substituting `alpha(Y,p,T)` inside
`T_from_hstat`'s own inner Newton, not just the outer Jacobian): §18.3's closed form means the
h->T inversion becomes EXACTLY linear in T at fixed Y, collapsing the current ~30-iteration inner
Newton (documented in the code as "the hottest kernel... ~60 EOS evals per cell per compute_R") to
a single division, and eliminating the non-monotonicity that produces `hsT<0` in the first place.
Cost: it changes `compute_R` itself (flips the FD-invariance gate, a residual change not just a
Jacobian change) and touches `T_from_hstat`'s signature and the `ACID_DENSE` debug probe.

**Declined for now.** (1) 3b's own justification is performance/robustness/consistency, explicitly
NOT a case14 fix -- case14's states don't reach the non-monotone region anyway (§18.1). (2) Phase
2's goal is already substantially met (13/25 recovered; case15 out of scope since round 7; case14
now also shown unreachable via T-pathway Jacobian work). (3) 3b is a larger, more invasive change
for a benefit that isn't blocking any currently-open target. Revisit only if a future need for
`T_from_hstat` performance, or its separately-noted robustness defect (it returns `true` even for
an unconverged inner Newton -- observed, not fixed, this round), actually arises.

### 18.8 Verdict

1. Stage 3a correctly implemented, all gates held, but a measured REGRESSION on its target case --
   an honest negative result, gated off rather than merged unconditionally.
2. The genuine deliverable is the closed-form identity (§18.3): a real, reusable mathematical fact
   that validates Stage 1 retroactively and explains exactly why the unstarred T-pathway is
   ill-conditioned near cold, gas-dominated states.
3. Phase 2's alpha-implicit-Jacobian investigation (Stages 0-3) is now complete: Stage 1 is a
   genuine win, Stage 2 a measured no-op with real diagnostic value, Stage 3a a measured
   regression (gated off), Stage 3b declined. Recommendation: proceed to Stage 4 (consolidation).

### 18.9 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit                                                     # Stage 3a closed-form checks included
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1                              ./build-cpp/cpp/denner_1d/denner1d_validate   # 14/19 (round 6/7, unchanged)
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_YADV_ALPHA_IMPLICIT_T=1 ./build-cpp/cpp/denner_1d/denner1d_validate   # 14/19 but case14 quality regresses (§18.4)
python3 scripts/yadv_verify.py
```

---
---

# ROUND 9

## 19. Phase 2 Stage 4 -- consolidation. Full six-configuration sweep, first direct timing
##     measurement, and the promotion decision for ACID_YADV_ALPHA_IMPLICIT.

Round 9 changed no solver code. Every configuration below is reachable with env vars that already
existed after round 8; the round's deliverables are one measurement script
(`scripts/yadv_r9_sweep.py`), this section, and the roadmap re-evaluation Phase 2's own plan
required of whichever round completed Stage 4.

### 19.1 Reproduction audit -- all of rounds 3-8, from one build, in one sitting

A live bug was found and fixed during this round's own tooling: the C++ binary prints lowercase
`nan`/`-nan` for divergent cases, which Python's `json.loads` cannot parse (it only accepts
capitalized `NaN`). The script's first pass silently dropped every NaN-carrying case from its
failure-set count -- `pass_count` (computed by the C++ binary itself) was already correct, only
the Python-side failure-set listing was short. Fixed with a regex substitution before parsing.
After the fix, all six configurations reproduced exactly, with no drift from any prior round:

| tag | configuration | pass_count | failure set | first recorded |
|---|---|---|---|---|
| A | `DENNER_ACID=1` (OFF) | 19/19 | -- | every round |
| B | `+ ACID_YADV=1` | 15/19 | 15, 24, 33, 34 | round 3, §14.2 |
| C | `+ ACID_YADV_ALPHA_IMPLICIT=1` | 14/19 | 14, 15, 24, 33, 34 | round 6, §16.4 |
| D | `+ ACID_NO_AJAC=1` (FD-invariance) | 12/19 | 14, 15, 24, 27, 28, 33, 34 | round 4, §15.2 |
| E | `DENNER_ACID=1 ACID_NO_AJAC=1` (FD control) | 13/19 | 15, 24, 27, 28, 33, 34 | round 4, §15.2 |
| F | `C + ACID_YADV_ALPHA_IMPLICIT_T=1` | 14/19, case14 collapsed | 14, 15, 24, 33, 34 | round 8, §18.4 |

OFF stayed 19/19 and 9/9 byte-identical against the published `solver_denner` binary; case01
stayed byte-identical between `ACID_YADV=1` and unset. Cases 04/05/07/35/36 (TR-BDF2 -> FD
Jacobian) were identical across A-F, as they must be by construction.

### 19.2 The consolidated sweep table

19 rows, configs A-F pass/fail plus B/C's `l2_p`/`corr_p`:

| case | A | B | C | D | E | F | B l2_p | C l2_p | B corr_p | C corr_p |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | PASS | PASS | PASS | PASS | PASS | PASS | 0 | 0 | 1 | 1 |
| 02 | PASS | PASS | PASS | PASS | PASS | PASS | 0 | 1.46e-11 | 1 | 1 |
| 04 | PASS | PASS | PASS | PASS | PASS | PASS | 0.01484 | 0.01484 | 0.9989 | 0.9989 |
| 05 | PASS | PASS | PASS | PASS | PASS | PASS | 0.00301 | 0.02673 | 0.9999 | 0.9958 |
| 07 | PASS | PASS | PASS | PASS | PASS | PASS | 0.00961 | 0.00961 | 0.9983 | 0.9983 |
| 13 | PASS | PASS | PASS | PASS | PASS | PASS | 0.02073 | 0.03128 | 0.9976 | 0.9945 |
| **14** | PASS | **PASS** | **FAIL** | FAIL | PASS | FAIL | 0.01389 | 0.01447 | 0.9995 | 0.9996 |
| 15 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 0.1665 | 0.01439 | 0.9855 | 0.9993 |
| 24 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 0.8374 | 0.8611 | 0.1656 | -0.763 |
| 25 | PASS | PASS | PASS | PASS | PASS | PASS | 0.04813 | 0.05969 | 0.9943 | 0.9911 |
| 26 | PASS | PASS | PASS | PASS | PASS | PASS | 0.02672 | 0.02361 | 0.9978 | 0.9983 |
| 27 | PASS | PASS | PASS | FAIL | FAIL | PASS | 0.02219 | 0.02219 | 0.9985 | 0.9985 |
| 28 | PASS | PASS | PASS | FAIL | FAIL | PASS | 0.0278 | 0.0306 | 0.9976 | 0.9971 |
| 30 | PASS | PASS | PASS | PASS | PASS | PASS | 0.03384 | 0.03384 | 0.9918 | 0.9918 |
| 31 | PASS | PASS | PASS | PASS | PASS | PASS | 0.04139 | 0.04139 | 0.9963 | 0.9963 |
| 33 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 1.573 | 0.8368 | 0.3509 | 0.1664 |
| 34 | PASS | FAIL | FAIL | FAIL | FAIL | FAIL | 0.8392 | 0.6923 | 0.1632 | -0.5338 |
| 35 | PASS | PASS | PASS | PASS | PASS | PASS | 0.00298 | 0.00298 | 0.9994 | 0.9994 |
| 36 | PASS | PASS | PASS | PASS | PASS | PASS | 0.00234 | 0.00234 | 0.9998 | 0.9998 |

**case14 is the ONLY case whose pass/fail verdict differs between B and C** (bold row) -- every
other case that moves at all (05, 07, 13, 25, 26, 27, 28) stays on the same side of its gate; 15,
24, 33, 34 stay FAIL on both sides (case15's magnitude improves dramatically, per round 6/7, but
its own narrower TV/oscillation gate still blocks it). Full per-case reduced-metrics table and raw
JSON: `docs/YADV_ROUND_9_PLAN.md`, `/tmp/yadv_r9/r9_raw.json` (regenerable via
`python3 scripts/yadv_r9_sweep.py --sweep --table`).

### 19.3 Wall clock -- never measured directly before

Rounds 5-8 measured pass_count and metrics only; §15.5's round-4 claim ("implicit-alpha costs
nothing under the analytic Jacobian... the FD Jacobian costs its usual ~1.7-1.9x") was qualitative.
Measured here per case with `denner1d_validate --only`, min of 3 repeats:

| case | B (s) | C (s) | C/B | D (s) | D/C | comparable |
|---|---|---|---|---|---|---|
| 01 | 0.103 | 0.114 | 1.100 | 0.154 | 1.355 | yes |
| 02 | 2.809 | 2.966 | 1.056 | 3.355 | 1.131 | yes |
| 13 | 1.292 | 1.329 | 1.029 | 2.033 | 1.529 | yes |
| 14 | 1.180 | 1.562 | 1.324 | 1.619 | 1.037 | no (differing outcome) |
| 15 | 3.491 | 2.024 | 0.580 | 4.730 | 2.337 | no |
| **24** | **0.636** | **32.824** | **51.632** | 53.246 | 1.622 | no |
| 25 | 2.962 | 3.204 | 1.082 | 5.068 | 1.582 | yes |
| 26 | 12.640 | 13.327 | 1.054 | 24.216 | 1.817 | yes |
| 27 | 16.298 | 18.059 | 1.108 | 27.466 | 1.521 | yes |
| 28 | 16.862 | 17.770 | 1.054 | 25.923 | 1.459 | yes |
| 30 | 2.038 | 2.153 | 1.056 | 3.284 | 1.526 | yes |
| 31 | 1.126 | 1.180 | 1.048 | 1.505 | 1.275 | yes |
| 33 | 12.887 | 5.202 | 0.404 | 17.454 | 3.355 | no |
| 34 | 24.360 | 27.910 | 1.146 | 17.247 | 0.618 | no |

(04/05/07/35/36 excluded -- TR-BDF2 forces the FD Jacobian regardless of these flags, not part of
this comparison.)

**Both-pass subset (9 cases, the only fair comparison -- a diverged run can abort early and look
artificially fast, exactly the artifact §15.5 already flagged): B total=56.13s, C total=60.10s,
ratio=1.071.** The Stage 1+2 analytic-Jacobian path costs **7.1%** more wall clock than plain
`ACID_YADV=1` -- close to, but somewhat above, Phase-2 §5 risk 12's "<5%" prediction (a few extra
`phase_props`-scale flops per cell). D/C (FD vs analytic, same configuration) = **1.547**,
broadly consistent with round 4's qualitative "~1.7-1.9x" (§15.5), measured somewhat lower here.

**case24 is a striking outlier, flagged on its own merit.** Under plain `ACID_YADV=1` it aborts in
0.64s (an early divergence exit); under `+ALPHA_IMPLICIT` it runs 32.8s before still failing --
**51.6x** slower for a case that fails either way. This is the "diverged run looks fast" artifact
measured directly for the first time: Stage 1+2 doesn't fix case24, but it does make the solver
work substantially harder before giving up on it, with zero quality benefit.

Iteration-count instrumentation (`ACID_RHIST`, sampled not averaged) was prepared
(`scripts/yadv_r9_sweep.py --iters`) but not run this round -- time budget; wall clock already
answers the primary cost question. Available for a future round if the iteration-count question
specifically becomes relevant.

### 19.4 Cases 24/33/34 -- Rankine-Hugoniot under implicit alpha. NOT what was predicted.

`scripts/yadv_rhcheck.py` hardcodes the main-tree path, so this measurement was run directly from
`main` after the round's merge, exactly as planned. The round-9 plan predicted "no movement"
(24/34 stay at ~1e-13, 33 stays at ~88%/65%). **That prediction was wrong, and the actual result
is a substantive, previously-unmeasured finding:**

| case | path | p_post | rho_post | u_post | Vs(mass) | momentum resid (rel) | energy resid (rel) |
|---|---|---|---|---|---|---|---|
| 24 | alpha | 1.50840e+10 | 1857.25 | 4698.0 | 6426.8 | -2.48e-06 | +1.81e-06 |
| 24 | Y | 1.50840e+10 | 1857.26 | 4698.0 | 6426.8 | +1.32e-13 | -1.61e-12 |
| **24** | **Y + implicit alpha** | -- **shock has LEFT the domain**, no undisturbed state -- |
| 33 | alpha | 5.87723e+09 | 1183.22 | 4302.0 | 5456.6 | -2.91e-05 | +5.83e-05 |
| 33 | Y | 1.49970e+10 | 1526.75 | 2443.0 | 2922.1 | **+8.81e-01** | **+6.46e-01** |
| **33** | **Y + implicit alpha** | **5.87724e+09** | **1183.35** | **4302.0** | **5456.5** | **+8.39e-13** | **-2.05e-12** |
| 34 | alpha | 3.16235e+10 | 2012.20 | 5149.5 | 8201.4 | -1.59e-06 | +3.82e-07 |
| 34 | Y | 3.16235e+10 | 2012.20 | 5149.5 | 8201.4 | +7.16e-13 | -1.01e-12 |
| **34** | **Y + implicit alpha** | -- **shock has LEFT the domain**, no undisturbed state -- |

(The alpha-path control rows are bit-identical to every prior run, confirming
`ACID_YADV_ALPHA_IMPLICIT`'s inertness without `ACID_YADV` -- the diagnostic is trustworthy.)

**case33's Rankine-Hugoniot jump closes to machine precision under implicit alpha** (momentum
`8.81e-01` -> `8.39e-13`, energy `6.46e-01` -> `2.05e-12`) -- a result that directly contradicts
rounds 4-8's repeated finding that Stages 1/2/3a "moved 24/33/34 by nothing." That finding was
true of the VALIDATION-GATE metrics (case33's `l2_p`/`corr_p` against the reference stay terrible
under `+ALPHA_IMPLICIT`, §19.2) -- it was never true of the underlying conservation-law
self-consistency, which nobody had checked with implicit alpha until this section.

**Why both can be true simultaneously, and why this is not a contradiction.** `yadv_rhcheck.py`
tests whether the code's OWN post-shock plateau is an admissible weak solution of the true NASG
mixture EOS at SOME shock speed inferred from mass conservation -- it says nothing about whether
that plateau matches the specific reference state `validation.cpp` checks against. §11.3
established that the alpha-held reference is *also* an exact RH solution of the same EOS -- the
jump conditions are 3 equations for 4 unknowns, and closure (A) (alpha held) and closure (B) (Y
held) are two *different, both legitimate* choices of the missing constraint, disagreeing on how
much interphase mass transfer the shock permits. §11.6's "solver defect, not a modelling
difference" verdict was reached under FROZEN alpha, where the Y-path plateau satisfied NEITHER
closure's jump conditions -- an inadmissible state by any standard, not a defensible alternative
closure. Under implicit alpha, case33 now lands on an admissible Y-consistent shock (to 13
digits) -- i.e. **Stage 1's fix, which §16-19 measured only against case13/14/15/25 and the
Jacobian-accuracy question, ALSO appears to repair (at least) case33's conservation defect**, at
least in the specific sense of "does the code's own answer obey physics." It still disagrees with
the alpha-held reference, which is why it still fails its validation gate -- but "disagrees with
one legitimate closure choice while satisfying the underlying conservation laws exactly" is a
fundamentally different, much less alarming finding than §11.5's original "violates momentum by
88%, an inadmissible state, full stop."

**case24 and case34's shocks leaving the domain is a genuinely different outcome from case33's,
and is NOT yet understood.** Two readings, not distinguished by this measurement alone: (a) the Y
+ implicit-alpha shock is now moving FASTER (a corollary of also being admissible, if `Ms` differs
from the alpha-held reference's), so it exits the `t_end` domain before the check can sample an
undisturbed post-shock cell -- consistent with case33's outcome, just further along; or (b)
something is going wrong for 24/34 specifically that case33 does not share (their `alpha_pre` are
0.5 and 0.25 vs case33's 0.75 -- not an obvious pattern). **Not resolved this round** -- flagged
explicitly as the sharpest concrete next question if a Phase 3 investigation into 24/33/34
proceeds, since it is now a much more promising lead than "three rounds of Jacobian work moved
nothing."

### 19.5 Promotion decision -- ACID_YADV_ALPHA_IMPLICIT stays a separate opt-in flag

`YADV_PHASE2_PLAN.md` §4's original Stage-4 bar ("`pass_count >= 15` with 13/15/25 recovered") is
stale: round 7 (§17.4) removed case15 from this plan's scope after computing its true blocker -- a
central-jump/concentration failure at the domain's stagnation point, structurally unreachable by
alpha-Jacobian work. The bar applied here is the re-scoped one:

> Fold only if, under the default analytic Jacobian: (i) `pass_count >=` plain `ACID_YADV=1`'s
> 15/19; (ii) NO case that plain `ACID_YADV=1` passes newly fails; (iii) 13 and 25 stay recovered;
> (iv) the OFF path is untouched. case15 excluded per round 7. `ACID_YADV` itself stays default
> OFF regardless.

**(i) and (ii) both fail, on the same case.** `+ALPHA_IMPLICIT` is 14/19 vs plain's 15/19, and
§19.2's table shows the difference is exactly case14, which plain `ACID_YADV=1` PASSES and
`+ALPHA_IMPLICIT` FAILS -- first recorded round 4 (§15.4), never bought back by Stage 1 (which
recovered 13 and 25 but not 14), and round 8 (§18.4) showed the T-pathway makes case14
dramatically worse, so its blocker is not there either.

The strongest argument FOR folding was considered and does not carry: Stage 1+2 is a genuine
CORRECTNESS fix (it removes a measured 521.56x wrong continuity diagonal and the residual/Jacobian
family mismatch Denner-Evrard-van Wachem (2020) warns about), so one could argue a user setting
`ACID_YADV=1` should get the consistent formulation. But the flag bundles two things: round 4's
RESIDUAL change (alpha re-derived at the current iterate inside `compute_R`) and rounds 6/7's
JACOBIAN consistency terms. The Jacobian half is not separately promotable, and not out of caution
-- out of correctness: with a frozen-alpha residual (plain `ACID_YADV=1`), `d(alpha)/dp` of the
coded map genuinely IS zero, so applying the starred terms there would re-introduce the same
family mismatch with the sign reversed. The `yadv && alpha_implicit` guard (`acid.cpp:1559`) is
load-bearing mathematics, not a conservative default. There is no good half to promote in
isolation, and the half that must come along is the one that costs case14.

Independently: `ACID_YADV=1` has had one stable meaning since round 3, and it is the literal
reproduction command in §13, §14.5, §15.7, §16.6, §17.6, §18.9 and this section, the "plain ON
stays 15/19" hard gate every round since round 5 has verified, and the toggle in a dozen
`scripts/yadv_*` files. Redefining it would invalidate all of them at once, for the sake of one
saved env var on a path that stays default OFF either way.

**Decision: do not fold. `ACID_YADV_ALPHA_IMPLICIT` remains a separate default-OFF opt-in flag
layered on `ACID_YADV=1`.**

### 19.6 Verdict -- where this research stands after nine rounds

**Proven.**
1. The OFF path is untouched, verifiably, after nine rounds: 19/19 and 9/9 byte-identical against
   the published `solver_denner` binary, re-verified this round on a fresh build.
2. Round 1's diagnosis chain is closed. Freezing `alpha` through the Newton WAS the mechanism §7.2
   identified (round 4 proved it with the FD Jacobian, §15.3), the analytic Jacobian's missing
   `d(alpha)/dp` WAS load-bearing (round 4 measured the regression, §15.4), and the closed-form
   `a_p = alpha(1-alpha)(zeta_b/rho_b - zeta_a/rho_a)` term IS the fix for the p-pathway: 12/19 ->
   14/19 with case13 and case25 both fully recovered (round 6, §16.4), durable through rounds 7, 8
   and re-verified here with zero drift.
3. case15's amplitude defect (§7.2's original target) is closed: `amp_ratio_p` 0.330 -> 1.00041 /
   1.00042 (round 6/9, consistent), `corr_p` 0.0994 -> 0.9993, under the DEFAULT analytic Jacobian.
4. `hstat_mix = Y*h_a + (1-Y)*h_b` exactly, hence `hsT* = Y*cp_a + (1-Y)*cp_b` and three companion
   closed forms (round 8, §18.3) -- strictly positive, provably free of the `1/hsT`
   near-singularity the unstarred form has below ~78K for air|water. Unit-tested to 6.8e-11. A
   reusable mathematical fact independent of whether any stage helped.
5. Four separate negative/no-op results, each honestly recorded rather than buried: alpha-space
   THINC (round 2, §10), the J2 flux-blend diagonal (round 7, §17.4, a measured no-op), the
   Stage-3a T-pathway (round 8, §18.4, a measured regression, gated off), and the analytic
   Jacobian's ~7% wall-clock cost vs plain `ACID_YADV=1` (round 9, §19.3, small but real).

**Still open, but with a genuinely new lead found this round (§19.4).** Cases 24/33/34's
validation-gate failure is unchanged by Stages 1/2/3a -- but the underlying CONSERVATION
self-consistency is not uniformly unmoved, contrary to what rounds 4-8 (measuring only the
validation-gate metrics) implied. **case33's Rankine-Hugoniot jump closes to machine precision
under `+ALPHA_IMPLICIT`** (momentum residual 88% -> 8.4e-13), meaning Stage 1's fix repairs its
conservation defect even though it does not make it match the alpha-held reference (a different,
also-legitimate closure choice, per §11.3 -- so this is "wrong reference" territory, not "solver
defect" territory, for case33 specifically). case24 and case34 instead show their shocks exiting
the domain before `t_end` under `+ALPHA_IMPLICIT` -- not yet understood whether this is the same
phenomenon further along (a faster, still-admissible shock) or a different problem specific to
those two cases. This is the sharpest, most concrete open question in the whole investigation and
the natural starting point for any Phase 3 into 24/33/34 -- a materially more promising lead than
"three rounds of Jacobian work moved nothing," which is what was believed before this round.
2. **case15's central-jump defect** (§17.4) -- `cj=30.02` against a threshold of `8.0` at the
   symmetric double rarefaction's stagnation point (`u=0` at `x=0.5`), with the oscillation test
   completely clean. A collocated stagnation-point discretization question (MWI/checkerboard
   family), not an alpha question.
3. **case14's actual blocker is still undiagnosed.** Round 5's `hsT<0` lead was measured real but
   confined to a single first-timestep transient cell (round 8, §18.1); the T-pathway Jacobian fix
   aimed at it made case14 dramatically WORSE (§18.4). All that is established is what it is NOT:
   not the T-pathway, and not the flux form (round 3). The documented `denner-pitfalls.md` finding
   -- THINC keeps alpha sharp while convected p/T stay first-order smeared -- points at
   phase-consistent energy transport, a model/scheme extension.

**Recommended status.**
- `ACID_YADV`: **default OFF, unpromoted.** Unchanged from every round since round 1. The case13
  win is real and durable; it sits next to an O(1) conservation failure on 24/33/34 that nine
  rounds have not closed.
- `ACID_YADV_ALPHA_IMPLICIT`: **default OFF, separate opt-in flag, explicitly NOT folded into
  `ACID_YADV`** (§19.5). It remains the configuration of record for the alpha-implicit question --
  the best understood and most internally consistent -- with one standing, precisely quantified
  cost relative to plain `ACID_YADV=1`: case14 (quality regression) and ~7% wall clock on cases
  both configurations solve successfully.
- `ACID_YADV_ALPHA_IMPLICIT_T`: **default OFF, research knob only** (measured regression).

**Phase 2 (`docs/YADV_PHASE2_PLAN.md`, Stages 0-4) is complete.**

### 19.7 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
python3 scripts/yadv_r9_sweep.py --verify            # OFF 9/9 byte-identical, case01 identical
python3 scripts/yadv_r9_sweep.py --sweep --table     # sect.19.1/19.2 tables + gate assertions
python3 scripts/yadv_r9_sweep.py --timing            # sect.19.3 wall clock, min of 3
python3 scripts/yadv_rhcheck.py                                # sect.19.4 control (post-merge)
ACID_YADV_ALPHA_IMPLICIT=1 python3 scripts/yadv_rhcheck.py     # sect.19.4 Y+implicit rows
```

---
---

# ROUND 10 (Phase 3a begins)

## 20. RETRACTION -- section 14.3 bullet 2 and section 19.4's headline were measuring a silently
##     stalled run, not a computed shock. Corrected with a null-run-guarded instrument.

### 20.1 What was found, and independently reproduced

`acid.cpp`'s per-step retry loop, on exhausting all 14 dt-halving retries, does
`if (!stepped) break;` -- this exits the time-stepping loop WITHOUT setting `diverged = true`.
`diverged` is what later triggers NaN-marking, whose own comment states its purpose: "so the
validate counts a collapsed/diverged run as a clean failure (finite=false), not a misleading
partial state at t < final_time." Because `diverged` is never set on a silent stall,
`solve_case_acid` returns a FINITE state -- the field at whatever step it last completed,
sometimes still the pristine initial condition -- and both `denner1d_validate` and
`denner1d_dump` treat it as a normal, completed run.

Independently verified by the Advisor session (not just taken from the Planner's report):
`acid.cpp`'s `if (!stepped) break;` site read directly and confirmed to lack a `diverged = true`
before it. Then reproduced empirically:

```
case24, plain ACID_YADV=1, ACID_DBG=1: last printed step = 5, t=2.978e-07 (final_time~1.089e-04,
  i.e. 0.27% of the run). Dump's mid-domain (x=0.498) reads p=100000, u=0, rho=499.5787 --
  bit-for-bit the pristine IC.
case33, ACID_YADV_ALPHA_IMPLICIT=1, ACID_DBG=1: last printed step = 5, t=2.853e-07
  (final_time~1.283e-04). Same signature.
case33, plain ACID_YADV=1 (control): completes normally to step 2000, t=1.225e-04 -- confirms
  the stall is config-specific, not universal.
```

**`yadv_rhcheck.py`'s undisturbed-cell search (`p < 1.5*p0`) then locks onto the near-void cell a
stall leaves near `x=0.1` as "the shock front," and picks a pre-shock reference cell that is
ALSO pristine IC.** The "residual" it computes in that situation is `cases.cpp`'s own closure-(A)
analytic construction checked against itself -- which §11.3 already proved closes to 1e-16. The
round-9 `+8.39e-13` and round-3's original `1e-13` numbers are that identity, seen through a
12-digit dump print, not a property of any computed solution.

**Retracted**: §14.3's second bullet ("Genuine partial win on discrete conservation... cases
24/34 now satisfy their own leading-shock RH jump to 1e-13") and §19.4's headline finding. Left in
place above, annotated, not deleted -- this project's established culture of keeping failed
results in the record (rounds 1-4's negative findings, round 7's peak_delta_u correction).

**Not retracted, because it was always a genuinely completing run**: case33 under plain
`ACID_YADV=1`, momentum residual `+8.81e-01`, energy `+6.46e-01` (§11.5, reproduced again this
round as the control check for the new instrument).

### 20.2 Time-vs-space -- resolved: neither. The domain is genuinely too short at the Y path's
##      own shock speed, for the runs that complete.

All three cases share identical domain/timing construction in `cases.cpp`
(`base_config(800, 0.7/Vs_ref, 0.0, 1.0)`, IC step at `x<0.1`, transmissive both ends) -- no
per-case asymmetry. For the two configurations that genuinely complete but whose shock has left
the domain (24 and 34 under `+ALPHA_IMPLICIT`), using the pre-shock state read directly from the
OFF path's own undisturbed right-boundary cells (robust, no analytic re-derivation needed -- the
OFF path always completes and its rightmost cells stay pristine IC by construction):

| case | config | `Vs`(mass) | `Vs/Vs_ref` | note |
|---|---|---|---|---|
| 24 | `+ALPHA_IMPLICIT` | 5148.4 | 0.8011 | plateau-window method, see caveat below |
| 34 | `+ALPHA_IMPLICIT` | 6933.0 | 0.8453 | plateau-window method, see caveat below |
| 33 | `ACID_YADV=1` (control) | 2922.1 | 0.5355 | in-domain, matches §11.5 exactly |

**Caveat, stated honestly rather than papered over**: the round-10 Planner's own reading (from
static analysis, not yet re-measured against a completed run at the time) predicted `Vs/Vs_ref`
of 1.49/1.40 (a FASTER shock) and momentum residuals of ~7%/2%. The measurement above instead
gives `Vs/Vs_ref` < 1 (a slower inferred speed) and much larger residuals (~50%/41%, see §20.3).
Root cause of the discrepancy, most likely: the fixed `x in [0.3,0.6]` plateau window used to
locate "the post-shock state" for an already-exited shock can straddle INTERNAL wave structure --
round 8's independent trace of a similar case (case33's completed run, §11.4) found a two-shock,
one-contact structure (`x~0.12 LEFT-FACING SHOCK ... x~0.45 CONTACT ... 0.50-0.93 shock-processed
material`), not a single uniform plateau. A window that straddles that contact would give a
median value that is neither pre- nor post-plateau cleanly. **This is not resolved this round --
it is exactly what Stage 2's `ACID_TEND_SCALE` diagnostic (viewing the shock BEFORE it exits, so
the true post-shock plateau can be sampled cleanly) is designed to fix**, per
`docs/YADV_PHASE3_PLAN.md`. Reported here as measured, with the method's limitation named, rather
than either silently adopting the more optimistic un-verified prediction or overclaiming
precision the current method does not have.

**What is solid, independent of the plateau-window caveat**: the null-run CLASSIFICATION matches
the plan's prediction exactly on all six case/config combinations (see §20.3's table), and it was
independently verified twice (direct `ACID_DBG` trace, and the automated guard in
`scripts/yadv_rh2.py`). That is this round's load-bearing result.

**The `alpha_pre` (0.50/0.75/0.25) hypothesis for which case stalls is dead.** Under plain
`ACID_YADV=1` the stall hits `alpha_pre in {0.50, 0.25}` (cases 24, 34) and spares 0.75 (case 33);
under `+ALPHA_IMPLICIT` it hits 0.75 and spares 0.50/0.25. The flag flips WHICH case stalls -- a
Newton-robustness switch, not a thermodynamic threshold.

### 20.3 `scripts/yadv_rh2.py` -- the null-run-guarded instrument

New file (`yadv_rhcheck.py` kept byte-identical, so the historical -- now understood-to-be-bogus
-- numbers stay independently reproducible). Guards: (1) a completion-fraction check via
`ACID_DBG`'s last printed step vs the case's known `final_time`; (2) an IC-match check (fraction
of cells still equal to the pre-shock state to 1e-6 relative); either tripping labels the run
`NULL RUN` and refuses to compute a residual; (3) reports `min(p)`/`min(rho)` (the near-void-cell
signature); (4) falls back to the OFF-path-derived analytic pre-shock state plus a plateau window
when the in-domain undisturbed-cell search comes up empty (an exited shock), instead of printing
"shock has left the domain, no undisturbed state" and stopping there.

```
| case | config | status | Vs/Vs_ref | momentum resid (rel) |
|---|---|---|---|---|
| 24 | plain | NULL RUN (t/t_end=0.0027, IC-match=0.89) | -- | -- |
| 24 | +IMPLICIT | completed (shock exited) | 0.8011 | +5.03e-01 |
| 33 | plain | completed (in-domain) | 0.5355 | +8.81e-01 (matches sect.11.5/19.4's control exactly) |
| 33 | +IMPLICIT | NULL RUN (t/t_end=0.0022, IC-match=0.89) | -- | -- |
| 34 | plain | NULL RUN (t/t_end=0.0045, IC-match=0.89) | -- | -- |
| 34 | +IMPLICIT | completed (shock exited) | 0.8453 | +4.07e-01 |
```

**Every one of the three predicted NULL RUNs is confirmed; every one of the three predicted
completions is confirmed.** The magnitude numbers for the two exited-shock cases carry the §20.2
caveat and are reported as a first estimate, not a final one.

### 20.4 Verdict for round 10

1. Two prior "results" (round 3's §14.3 bullet 2, round 9's §19.4 headline) are retracted, with
   the mechanism identified, independently confirmed twice, and a working guarded instrument
   built to prevent recurrence.
2. **No case in {24, 33, 34} has ever completed under both a plain and an implicit-alpha
   configuration simultaneously.** Every RH-residual number this whole investigation has ever
   produced for these three cases (round 3, round 4, round 9, this round) is from a DIFFERENT
   configuration completing than the one being compared against -- there has never yet been a
   controlled A/B on any of these three cases. That is Phase 3a's real starting line.
3. Cases 24/34, where they DO complete (under `+ALPHA_IMPLICIT`), carry a genuinely large
   Rankine-Hugoniot violation (order 40-50%, pending §20.2's cleaner remeasurement) -- not the
   near-perfect closure round 9 believed. Case33's only completing run (`ACID_YADV=1` plain)
   remains the one number in this family that was always real: 88%/65% violation.
4. `Y` is conserved to 3-4 significant digits through the leading shock in every completing run
   (§20.2, consistent with §11.4's original observation), against a closure-(A) reference that
   requires 270-1620x `Y` growth across the same shock. This is very likely why no configuration
   of any of the last four rounds' Jacobian work could ever bring these three cases to their
   validation-gate reference -- it may be a structurally unreachable target for any Y-preserving
   scheme, not a solver defect. Not yet the final word (per `YADV_PHASE3_PLAN.md`'s stopping
   criteria) -- Stage 1-4 of that plan pursue this properly, including the controlled A/B this
   investigation has never had.
5. `ACID_YADV`'s recommended status is UNCHANGED by this round: default OFF, 15/19. Nothing in
   this round's findings moves that; if anything, they weaken the case for 24/33/34 ever changing
   it, per point 4.

Zero solver code changed this round -- `scripts/yadv_rh2.py` is additive/diagnostic only. All
four hard gates (OFF 19/19+9/9, plain ON 15/19, `+IMPLICIT` 14/19, FD-invariance failure set) hold
by construction; re-verified via `denner1d_unit` and a direct rebuild this round.

### 20.5 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
python3 scripts/yadv_rh2.py                          # sect.20.3's null-run-guarded table
DENNER_ACID=1 ACID_YADV=1 ACID_DBG=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    2>&1 >/dev/null | grep "ACID step" | tail -5      # sect.20.1's direct stall trace
```

---
---

# ROUND 11 (Phase 3a Stages 1+2)

## 21. The silent stall is now audible; the "exited shock" residual for cases 24/34 was itself
##     measured wrong in round 10 -- corrected, and it now matches the round-10 Planner's static
##     prediction almost exactly

### 21.1 Stage 1 -- STALLED is now printed, verified stderr-only

At `acid.cpp`'s retry-exhaustion break (the round-10 root cause, sect.20.1), the retry loop's last
failure reason (Newton-no-progress / non-finite p / non-finite u / `|u|>10*uref`), offending cell,
and attempted dt are now carried out of the loop and printed unconditionally to stderr:
`STALLED: case=<id> no admissible step at dt=... after N retries, step S, t=... of ... -> stop
(state returned as-is, NOT marked diverged)`, plus an `ACID_DBG`-gated `STALLED-DETAIL` line.
Deliberately does **not** set `diverged=true` (Stage 3c, requires an explicit Advisor decision,
not done this round). A companion unconditional-under-`ACID_DBG` `ACID done case=... step=...
t=... of ...` line was also added, needed by Stage 2's script.

**Verified stderr-only** by an isolated build of unmodified main HEAD (`git worktree add --detach`,
commit `533fa9f`) and a byte-for-byte comparison against the round-11 build: all 19 OFF-path dumps
identical, and `denner1d_validate` stdout identical for OFF (19/19), plain-ON (15/19), and
`+ALPHA_IMPLICIT` (14/19). `STALLED` fires on exactly the three known-stalled configurations
(case24 plain, case34 plain, case33 `+ALPHA_IMPLICIT`) and nowhere else, including the OFF path.

**Side finding, not caused by this round's edit**: the FD-invariance gate (`ACID_NO_AJAC=1`)
measured **13/19** with failure set `{15,24,27,28,33,34}` on both the round-11 build and the
unmodified main-HEAD baseline -- identical on both, so this is not a regression. It does, however,
correct a **stale figure carried in this project's own memory/plan documents**: prior rounds'
records (and this round's own Planner brief) cited `12/19`, failure set `{14,15,24,27,28,33,34}`.
Case14 now passes the FD-invariance gate; the 12/19 figure was never re-verified after whatever
round last measured it and had silently gone stale. Filed here as the correction; not chased
further (out of scope for Phase 3a).

### 21.2 Stage 2 -- `ACID_TEND_SCALE`, verified as a true no-op when unset

New env var, gates exactly like `ACID_YADV`/`ACID_DBG`. With it unset, all five hard gates are
byte-identical to Stage 1's own numbers (OFF 19/19, plain-ON 15/19, `+ALPHA_IMPLICIT` 14/19,
FD-invariance 13/19, `denner1d_unit` ok) and zero `TEND_SCALE:` banners print. New script
`scripts/yadv_r11_window.py` (round 10's `yadv_rh2.py` kept untouched) drives the sweep, reading
the solver's own `ACID done`/`STALLED` lines instead of a hardcoded time dict.

### 21.3 The exited-shock residual for cases 24/34 -- corrected, and it now matches the Planner's
##      original static prediction

**Round 10's §20.2 measurement (`Vs/Vs_ref` 0.80/0.85, momentum residual ~40-50%) is itself now
found to be measured wrong, and is retracted here.** The fixed `plateau_window(rows, lo=0.3,
hi=0.6)` box that `yadv_rh2.py` used straddled internal wave structure, exactly as §20.2's own
caveat warned it might.

Round 11's first attempt at a front-derived window (sampling `[x_front+0.05, x_front+0.20]`,
i.e. AHEAD of the detected shock front) hit a `ZeroDivisionError` on every scale except 1.0 --
`rho1==rho0` every time, because the window was on the **wrong side of the front**: this shock
propagates left-to-right into undisturbed material (confirmed by `yadv_rh2.py`'s own
`preshock_state`, which reads the OFF path's rightmost -- largest-x -- cell as always-undisturbed),
so the post-shock plateau is BEHIND the front (smaller x), not ahead of it. Fixed by sampling
`[x_front-0.20, x_front-0.05]` instead.

With that fix, the window's `(p_post, rho_post, u_post)` state is **stable and converged across
scales 0.4 to 1.0** for both cases (case24: `rho_post` 854.5-855.8, `u_post` 3990.0 exactly,
`p_post` 2.0666e10 across all six scales; case34 similarly stable from scale 0.4 onward, scale 0.3
alone is an outlier at 1.073e-01 vs the converged 2.06e-02, likely still-forming shock structure at
that small a window). The resulting **momentum residual is small and consistent**:

| case | momentum resid (rel), scale=1.0 (largest in-domain) | `Vs(mass)` | `Vs(mass)/Vs_ref` |
|---|---|---|---|
| 24 | **+7.351e-02** | 9605.6 | **1.4946** |
| 34 | **+2.063e-02** | 11455.3 | **1.3968** |

**This matches the round-10 Planner's original static prediction (momentum residual ~7.36e-02 /
~2.06e-02, `Vs/Vs_ref` 1.4945 / 1.3968 -- a FASTER shock) to 3-4 significant figures.** Round 10's
own measurement (slower shock, large residual) was the artifact; the Planner's static reasoning
was right, and this round's corrected instrument now confirms it directly from a completed run's
data, not just static analysis. `docs/YADV_PHASE3_PLAN.md` sect.1's "genuinely faster, still
admissible-looking shock, momentum residual on the order of a few percent" framing is the one that
survives.

**Independent cross-check, and an honest discrepancy**: a model-free shock speed from a linear fit
of `x_front` vs `t_end` across the 7-scale sweep gives `Vs_front/Vs_ref` = 1.018 (case24, R²=0.816)
and 1.185 (case34, R²=0.956) -- both noticeably smaller than the `Vs(mass)` ratios above, and the
fits are imperfect (R² well short of 1). Root cause, from inspecting the raw `x_front` sequence:
`argmax|dp/dx|` is not a fully robust front tracker at the larger scales -- case24's `x_front` at
scale 0.85 is 0.9756 but DROPS to 0.8569 at scale 1.0 despite `t_end` increasing, meaning the
gradient-maximum criterion latched onto a different feature (likely a secondary/reflected wave, or
the original material contact) once more wave structure had developed by the later time. The
window-state convergence (the p/rho/u plateau values themselves, which stayed essentially constant
across scales despite this front-tracking noise) is judged the more trustworthy signal here, since
it is a converged, self-consistent measurement rather than a single point-in-time estimate; the
`Vs_front` cross-check is reported as unreconciled with `Vs(mass)`, per this round's plan (sect.5.4
addition) -- not resolved by picking whichever number is more convenient.

**33/plain control**: reproduces sect.20.2/20.3 exactly -- `Vs/Vs_ref=0.5355`, momentum residual
`+8.808e-01`, confirming the corrected-window methodology is not itself the source of any
divergence from a case that was always genuinely in-domain.

### 21.4 Stall-bracketing sweep -- a bracket found, with a caveat on the plan's own "one consistent
##      trajectory" assumption

For 24/plain, 34/plain, and 33/`+ALPHA_IMPLICIT`, sweeping `ACID_TEND_SCALE` at small scales around
the known stall point (`t/t_end` ~0.0027/0.0045/0.0022, sect.20.1) does bracket a stall onset:
24/plain runs clean through scale 0.0025 (step 7) and stalls at scale 0.003 (step 30); 33/+IMPLICIT
runs clean through every scale tested up to 0.003 (the stall point at ~0.0022 sits inside this
sweep's range but was never hit -- would need finer scales below 0.0025 to bracket it directly).

**34/plain is non-monotonic**: clean through scale 0.002, **STALLS at scale 0.0025** (step 6), then
**clean again at scale 0.003** (step 8) -- a larger observation window succeeding where a smaller
one failed. This falsifies `docs/YADV_ROUND_11_PLAN.md` sect.4.4's framing that a scale sweep
produces "a consistent sequence of snapshots of one trajectory": it does not, because
`dt = min(dt, t_end - t)` clamps the LAST step of every run to land exactly on its own scaled
`t_end`, so each scale's final step has a different size and can independently succeed or fail the
retry loop regardless of the physical time reached. The bracket this sweep produces is therefore a
bracket on "does the clamped-to-`t_end` step survive", not cleanly on "when does the void cell
form" as the plan intended. A future round wanting a precise void-cell-formation step should dump
at consecutive UNSCALED steps directly (e.g. via a small `max_steps` cap) rather than via this
scale sweep.

### 21.5 Verdict for round 11

1. The silent stall is audible (`STALLED:`), verified stderr-only, zero effect on any of the four
   hard-gate metrics or any dump.
2. A stale FD-invariance-gate figure in this project's own records is corrected (13/19, not
   12/19) -- not a regression, confirmed against an unmodified main-HEAD build.
3. **Round 10's §20.2 "exited shock is slower and grossly RH-violating" finding is retracted.**
   With a correctly-sided, front-derived window, cases 24/34 under `+ALPHA_IMPLICIT` show a shock
   that is genuinely ~1.40-1.49x FASTER than the case's own reference speed, with a momentum
   residual of only ~2-7% -- closely matching the round-10 Planner's original (previously
   unverified) static prediction. This is a small but real fixed-domain-length artifact (the
   `t_end = 0.7/Vs_ref` construction assumes the computed shock matches `Vs_ref`; it does not, for
   these two cases under this flag), not evidence of gross non-conservation.
4. No controlled A/B still exists for any of {24,33,34} -- this round's sweeps confirm again that
   only one configuration per case ever completes (round 10's load-bearing finding stands
   unchanged).
5. `ACID_YADV`'s recommended status is UNCHANGED (default OFF, 15/19). Nothing this round moves it;
   per `docs/YADV_PHASE3_PLAN.md`'s own non-goals, that decision is out of scope until a real A/B
   exists.
6. Zero solver-numerics change with either env var unset (both Stage 1 and Stage 2 verified
   byte-identical against an isolated main-HEAD build); all four hard gates hold (unit PASS, OFF
   19/19, plain-ON 15/19, `+ALPHA_IMPLICIT` 14/19; FD-invariance 13/19 -- the corrected figure).

### 21.6 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 2>&1 >/dev/null | grep STALLED
python3 scripts/yadv_r11_window.py            # sect.21.3/21.4's full sweep tables
```

---
---

# ROUND 12 (Phase 3a Stage 3)

## 22. Retry-exhaustion accept-best (`ACID_STALL_ACCEPT`) gets cases 24 and 34 to complete for the
##     FIRST TIME EVER -- the first-ever controlled A/B now exists, and it favors `+ALPHA_IMPLICIT`,
##     with an honest caveat

### 22.1 The stall is a pure Newton-convergence failure, not a physical collapse -- 3b refuted

Direct `ACID_DBG` traces on all three known-stalling configurations (case24 plain, case34 plain,
case33 `+ALPHA_IMPLICIT`) show the identical signature: `reason=newton-no-progress cell=-1`, and
`max|u|` numerically FROZEN across the full 14-retry, ~16000x dt-halving sweep. No non-finite cell,
no `rho->0`, no void cell, ever, at any retry, in any of the three. This directly refutes
`docs/YADV_PHASE3_PLAN.md`'s Stage-3b premise ("the Y->alpha recovery is collapsing to alpha->1
adjacent to the closure-(A) contact") -- there is no collapsing cell to find. 3b is not implemented.

### 22.2 The decisive measurement -- `r_init` grows exactly as `1/dt`

`ACID_RHIST=1 ACID_BLK_STEP=19` on case24's stalling step, independently reproduced by the Advisor
(retries 6-13 shown; the earlier retries 0-4 are flat):

| retry | dt | `r_init` (residual at `it==0`, i.e. before any Newton work) | ratio vs prior |
|---|---|---|---|
| 6 | 8.362e-13 | 2.5492e+11 | -- |
| 7 | 4.181e-13 | 3.9117e+11 | 1.53 |
| 8 | 2.090e-13 | 7.2446e+11 | 1.85 |
| 9 | 1.045e-13 | 1.4323e+12 | 1.98 |
| 10 | 5.226e-14 | 2.8701e+12 | 2.004 |
| 11 | 2.613e-14 | 5.7566e+12 | 2.006 |
| 12 | 1.306e-14 | 1.1535e+13 | 2.004 |
| 13 | 6.532e-15 | 2.3095e+13 | 2.002 |

`r_init` **exactly doubles for every halving of dt from retry 6 onward** -- the pre-Newton state
handed to Newton gets strictly and unboundedly worse as dt shrinks. This falsifies the `bad` gate's
own design comment (`acid.cpp`, "dt-retry a non-converged step ONLY when it made NO progress at all
-- that means dt is too large"): dt is not too large here; halving it is actively counterproductive.
The mechanism (not implemented or fixed this round, see §22.6/7a): a dt-INDEPENDENT state mismatch
is injected by the pre-Newton explicit block (Y-advection -> alpha recovery at `(p_o,T_o)` -> the
`rho_o`/`hstat_o`/`Htot_o` re-evaluation), entering the transient residual as `Delta*dx/dt`.

### 22.3 The fix: `ACID_STALL_ACCEPT` -- accept the best-across-retries eligible state

New research-only env var (default 0, byte-identical). Level 1: on exhausting all 14 retries with
every failure being `newton-no-progress` (finite + speed-bounded), adopt the best-ranked
(`rbest/r_init` minimized) state instead of breaking with nothing. Discovered during implementation
that `acid.cpp:2093` (`if (ajac && coupled && !conv_inner && best_it >= 0) s = s_best;`) already
restores the best iterate WITHIN a retry unconditionally -- so no new keep-best machinery was needed
inside the Newton loop, only keep-best ACROSS retries. Level 2 additionally stops a step that only
ever hit reason-1 retries from collapsing `cfl_scale` (measured: `cfl_scale` was already pinned at
its `1.0e-3` floor after only 19-229 steps for all three stalling cases -- §22.7).

Loud by construction: every acceptance prints `STALL-ACCEPT:` (case, step, retry chosen, ratio),
and a completed run that contains any prints `STALL-ACCEPT-TOTAL: ... this run is NOT a clean
solve` unconditionally. A run cannot silently be mistaken for clean (the exact failure mode of the
sect.20 retraction).

Bounded by a consecutive-accept budget (`ACID_STALL_ACCEPT_MAX`, default 4, reset by any clean
step) -- caps a livelock to a few wasted steps, never a hang.

**Verified byte-identical to round 11 with the var unset**: all four hard gates (OFF 19/19, plain-ON
15/19, `+ALPHA_IMPLICIT` 14/19, FD-invariance 13/19) identical stdout, zero `STALL-ACCEPT` lines.

Found and fixed a scoping bug during implementation: `rho_o`/`u_o`/`Htot_o` are retry-loop-local
(same class of bug as round 11's `rbest`/`r_init`) -- resolved by recognizing the mirror-bookkeeping
copy is dead code on the accept path anyway (`have_o2=false` there makes `mom_o2`/`rho_o2`/`ene_o2`
unread by construction), so it was deleted rather than captured.

### 22.4 Result -- FAR better than the round's own calibrated expectation

The plan expected level 1 alone to likely be insufficient (remove the stop but leave `cfl_scale` on
its floor, ~2-3 million steps still needed). **Measured instead**: level 1 alone gets BOTH case24
and case34 (plain `ACID_YADV=1`) to complete cleanly to `t_end`:

| case | total steps | accepted (non-converged) steps | when | outcome |
|---|---|---|---|---|
| 24 plain | 1800 | 2 | step 19-20 (t/t_end = 0.27%) | **completes to t_end** |
| 34 plain | 2648 | 4 | step 229-233 (t/t_end = 0.45%) | **completes to t_end** |
| 33 `+IMPLICIT`, level 1 | 104 (stopped) | 4 (budget exhausted) | -- | still stalls |
| 33 `+IMPLICIT`, level 2, MAX=4 | 251 (stopped) | 8 | -- | still stalls, further but not close |
| 33 `+IMPLICIT`, level 2, MAX=20 | 1416 (stopped) | 221 | -- | reaches 14.8% of t_end, still stalls |

**Every accepted step for cases 24 and 34 occurs in a tight cluster right at shock formation** (well
under 0.5% into the run); after that cluster, both runs proceed with ZERO further `STALL-ACCEPT`
lines for the remaining 99.5%+ of steps. This is a brief, localized Newton difficulty at the moment
the initial discontinuity resolves into a shock, not a pervasive one. Case33 is qualitatively
different: sustained, escalating difficulty (221 accepts even at a 20-step budget, still short of
`t_end`) -- not a one-time formation glitch. **3a alone does not fix case33**; a future round's
Stage 3 (§22.6/7a-7c) is still needed for it.

### 22.5 The first-ever controlled A/B -- and an honest, load-bearing caveat

No case in {24,33,34} has EVER had a plain and a `+ALPHA_IMPLICIT` run complete simultaneously
(rounds 3-11). **This round produces the first two.** Front-derived-window RH residual (round 11's
validated method, §21.3, reused verbatim -- independently reconfirmed here that the `+IMPLICIT`
numbers exactly reproduce round 11's published values):

| case | config | momentum resid (rel) | `Vs(mass)` |
|---|---|---|---|
| 24 | plain + `STALL_ACCEPT=1` (2 accepted steps) | **+9.804e-01** | 1926.1 |
| 24 | `+ALPHA_IMPLICIT` (clean, 0 accepted steps) | **+7.351e-02** | 9605.6 |
| 34 | plain + `STALL_ACCEPT=1` (4 accepted steps) | **+4.562e-01** | 8698.5 |
| 34 | `+ALPHA_IMPLICIT` (clean, 0 accepted steps) | **+2.063e-02** | 11455.3 |

At face value: `+ALPHA_IMPLICIT` conserves the leading shock's Rankine-Hugoniot jump dramatically
better than plain -- 7.35% vs 98.0% for case24, 2.06% vs 45.6% for case34.

**This must NOT be read as a clean result.** The plain runs are explicitly NOT clean solves (2 and 4
accepted non-converged steps respectively, per `STALL-ACCEPT-TOTAL`'s own disclosure), and those
accepts are concentrated exactly at shock formation -- the single most sensitive place for a defect
to corrupt the leading shock's own self-consistency as it propagates through the rest of the domain.
It is entirely possible (and not ruled out by this round's measurements) that some or all of the
98.0%/45.6% residual reflects the `ACID_STALL_ACCEPT` mechanism's own defect at formation, not an
intrinsic property of plain Y-transport. **The controlled A/B this investigation has wanted since
round 10 now exists, but it is not yet a clean one.** A genuinely clean comparison needs the
formation-time difficulty itself fixed (§22.6/7a), not worked around.

### 22.6 Level 2's cost -- a real regression, not adopted as default

`ACID_YADV=1 ACID_STALL_ACCEPT=2` (no `+ALPHA_IMPLICIT`) drops plain-ON's `denner1d_validate` from
15/19 to **14/19** -- case28, previously clean, newly fails. Level 1 alone (`ACID_STALL_ACCEPT=1`)
has **zero regression**: 15/19, identical failure set `{15,24,33,34}` to level 0. Per the round's own
pre-registered decision rule (`docs/YADV_ROUND_12_PLAN.md` sect.9, R9): level 1 is reported as the
round's result; level 2's CFL-neutrality change is a measured regression, not adopted. (It remains
available as a research-only flag for cases like 33 where the extra progress it buys may be worth
the cost in a future targeted investigation -- not for validation runs.)

### 22.7 Verdict for round 12

1. **3b refuted** (no void cell, ever, in any of the three stalling configurations) -- confirms and
   extends §2's branch decision. **3c still not implemented** (needs explicit Advisor decision).
2. **The stall mechanism is now understood**, not just worked around: a dt-independent state
   mismatch from the pre-Newton explicit alpha/Y block makes the residual scale as 1/dt, so
   dt-halving retry is actively counterproductive for this failure mode. This is the round's most
   durable result regardless of the accept mechanism's own caveats.
3. **Cases 24 and 34 (plain) complete for the first time ever**, via `ACID_STALL_ACCEPT=1`, with
   zero `pass_count` regression -- far exceeding the round's own calibrated expectation.
4. **The first-ever controlled A/B for cases 24/34 now exists** and favors `+ALPHA_IMPLICIT` by a
   wide margin (7-8x smaller RH residual) -- reported with the explicit, load-bearing caveat that
   the plain runs carry a small, precisely-located but not-yet-eliminated non-convergence defect at
   shock formation. Not yet a clean result; the honest next step is fixing formation, not trusting
   the comparison as final.
5. **Case33 remains unsolved** -- a qualitatively different, sustained (not one-time) difficulty;
   3a alone does not get it to `t_end` even with a 5x larger accept budget.
6. **Level 2 is a measured net negative** for general validation (case28 regression) and is not
   adopted; level 1 is the round's recommended setting for any future work in this area.
7. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, still 15/19 at the shipped
   default of `ACID_STALL_ACCEPT` unset). All four hard gates hold with the new env var unset.

### 22.8 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_RHIST=1 ACID_BLK_STEP=19 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    2>&1 >/dev/null | grep "^RHIST it=0"       # sect.22.2's r_init doubling, per retry block
DENNER_ACID=1 ACID_YADV=1 ACID_STALL_ACCEPT=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    2>&1 >/dev/null | grep "STALL-ACCEPT"      # sect.22.3/22.4
```

---
---

# ROUND 13 (Phase 3a Stage 3, follow-up 7a)

## 23. The `1/dt` mismatch is precisely localized and its mechanism confirmed to textbook clarity --
##     but the naive fix does NOT solve the stall. A real, durable diagnostic win; a refuted fix.

### 23.1 New instrument (`ACID_RINIT`) -- every prediction confirmed cleanly

Round 12 §22.2 measured `r_init` doubling per dt-halving but did not identify which term carried
it. `ACID_RINIT` (new, diagnostic-only) splits `rnorm3()`'s components at `it==0` (`RINIT`) and, at
the point right after the Eqs.43-44 old-level rebuild, the candidate dt-independent state
mismatches (`RMISM`): `dh=|s.h-Htot_o|`, and the alpha jump `dal` split into a REMAP part (alpha
recovered at the PREVIOUS step's frozen Y meeting THIS step's `(p_o,T_o)` -- predicted
dt-independent, since it is set by last step's `dt_prev`, not this step's `dt`) and an ADVECTION
part (this step's own Y-transport, predicted `O(dt)`).

**Self-check passed**: `RINIT`'s `r` matches `RHIST`'s `n0` to the last printed digit at every
retry, every run.

**Primary measurement, case24 step 19** (the exact step round 12 §22.2 measured):

| retry | dt | `dal_remap` | `dal_adv` | `ene` (energy component) |
|---|---|---|---|---|
| 1 | 2.676e-11 | 5.6763e-02 | 8.680e-04 | 1.4226e+11 |
| 5 | 1.672e-12 | 5.6763e-02 | 5.686e-05 | 1.5067e+11 |
| 6 | 8.362e-13 | 5.6763e-02 | 2.848e-05 | 1.7555e+11 |
| 9 | 1.045e-13 | 5.6767e-02 | 3.565e-06 | 8.6008e+11 |
| 13 | 6.532e-15 | 5.6764e-02 | 2.228e-07 | 1.3627e+13 |

`dal_remap` is **constant to 4-5 significant figures across every one of the 13 retries** (5.676e-2
throughout); `dal_adv` **halves exactly, every single retry**, from 8.68e-4 down to 2.23e-7 -- a
textbook-clean confirmation of P1+P2+P3 (`docs/YADV_ROUND_13_PLAN.md` §2), with no ambiguity left
for interpretation.

**Control (OFF, `ACID_YADV` unset)**: **zero `RINIT`/`RMISM` lines** -- the instrument is gated on
`yadv`, and the OFF path has no remap term structurally (§0: alpha updates the same state it
already holds, no Y-recovery step), so this is the predicted structural immunity made visible.

**Control (`+ALPHA_IMPLICIT`)**: step 19 needed only **1 retry** (not 14 -- this step doesn't even
stall under `+ALPHA_IMPLICIT`), and `dal_remap = 2.2204e-16` -- **literal `DBL_EPSILON`**, i.e.
exactly zero to machine precision, while `dal_adv = 2.469e-02` carries the entire alpha jump. This
is the predicted mechanism made numerically explicit: `+ALPHA_IMPLICIT` re-derives alpha at the
CURRENT `(p,T)` on every Newton call, so the previous step's converged alpha already reflects it --
the REMAP term is identically absent. **This gives round 12 §22.5's A/B result (which favored
`+ALPHA_IMPLICIT` 7-8x on RH residual) a mechanism, not just a correlation**: `+ALPHA_IMPLICIT`
structurally cannot accumulate this particular defect.

### 23.2 The fix (`ACID_YADV_HREINIT`) -- refuted (S4)

Setting `s.h := Htot_o` right after the Eqs.43-44 rebuild (before Newton starts) was predicted to
make the `it==0` transient vanish and let cases 24/34 complete cleanly (S2). **Measured instead**:

| config | result |
|---|---|
| case24, `HREINIT` alone (no `STALL_ACCEPT`) | still stalls -- pushed from step 19 (`t=2.99e-7`) to step 28 (`t=5.52e-7`), i.e. ~1.85x further, but only ~1.6% of the 1732 steps the OFF path needs, nowhere near `t_end` |
| case34, `HREINIT` alone | **worse**: 15400 tiny steps needed to reach `t=2.91e-7`, LESS time than plain's original step-229 stall (`t=3.85e-7`) reached -- the CFL ramp collapsed harder, not better |
| case24, `HREINIT` + `STALL_ACCEPT=1` | **9 accepted non-converged steps needed** (vs round 12's 2 without `HREINIT`) -- worse, not better -- reaching only `t=1.28e-6` (1.2% of `t_end`) before still hitting the accept budget |

**S4 fires unambiguously: the fix does not work, and combined with the round-12 safety net it is
measurably worse than not having it.** `ACID_YADV_HREINIT` stays default OFF, not promoted, and is
**not recommended** for combination with `ACID_STALL_ACCEPT`.

### 23.3 Honest interpretation -- what the refutation means, not just that it happened

The Stage-0 mechanism (§23.1) is not in doubt -- it is measured to 4-5 significant figures with two
independent controls both landing exactly on their predicted extremes (zero and `DBL_EPSILON`).
What is refuted is only the inference that *removing that one `it==0` initial-guess artifact would
be sufficient* to let Newton converge. It evidently is not: correcting `s.h` alone still leaves
`s.rho` (and everything `compute_R` derives from it) at its stale, `s0`-consistent value until the
first `compute_R()` call re-derives `T`/`rho` from the corrected `h` -- and by that point Newton is
already iterating, not starting from a self-consistent state. A more complete fix would need to
reconcile `s.rho`/`s.T` at the SAME instant as `s.h` (i.e. actually re-solve for a self-consistent
`(T,rho)` at the new alpha before Newton's `it==0`, not just hand it a better `h` and let the first
iteration sort out the rest) -- not attempted this round; flagged as the natural next step for a
future round, not implemented here per this round's own non-goal against scope creep into the
Newton loop itself.

It is also possible -- not measured either way this round -- that the difficulty compounds: once
one step's Newton solve fails to fully converge (even under `STALL_ACCEPT`'s best-iterate accept),
the NEXT step inherits a slightly-off `s.alpha`/`s.h` pairing that reintroduces a fresh REMAP-like
mismatch, so a single-point fix cannot suffice regardless of how well it is targeted. Round 12
§22.4's own case33 signature (sustained, escalating, not one-time) is consistent with this reading.

### 23.4 Verdict for round 13

1. **Stage 0's mechanism finding is the round's durable result**: the `1/dt` growth is precisely the
   REMAP term in the alpha recovery -- alpha recovered at the previous step's frozen Y meeting the
   current step's `(p_o,T_o)`, constant within a retry sweep because it is set by `dt_prev`, not
   the current `dt`. Confirmed via self-checked instrumentation with two controls landing exactly on
   their predicted extremes.
2. **Stage 1's fix is refuted (S4)** -- correcting the initial guess alone does not restore Newton
   convergence, and combined with round 12's `ACID_STALL_ACCEPT` safety net is measurably worse.
   Both new flags (`ACID_RINIT`, `ACID_YADV_HREINIT`) stay default OFF; `ACID_YADV_HREINIT` is
   explicitly NOT recommended in combination with `ACID_STALL_ACCEPT`.
3. Round 12's `ACID_STALL_ACCEPT=1` (without `HREINIT`) remains the only working path to completion
   for cases 24/34, with its existing caveat (round 12 §22.5) UNCHANGED -- this round does not
   remove it, and did not find a way to.
4. Per this project's negative-result culture and the plan's own pre-registered rule
   (`docs/YADV_ROUND_13_PLAN.md` §2/§3): a correctly-instrumented refutation of a stated hypothesis
   is measured progress, not a failed round -- `consecutive_failures` is not incremented.
5. Case33 untouched (explicit non-goal, per plan). Stage 3c (`diverged=true`) not implemented
   (still needs explicit Advisor decision). `ACID_YADV`'s recommended default status is UNCHANGED
   (default OFF, 15/19). All four hard gates hold with both new env vars unset.

### 23.5 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_RINIT=1 ACID_BLK_STEP=19 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    2>&1 >/dev/null | grep -E "^RINIT|^RMISM"                     # sect.23.1's primary measurement
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RINIT=1 ACID_BLK_STEP=19 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 24 2>&1 >/dev/null | grep -E "^RINIT|^RMISM"  # control D
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_HREINIT=1 ACID_DBG=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    2>&1 >/dev/null | grep STALLED                                # sect.23.2's refutation
```

---
---

# ROUND 14 (Phase 3a Stage 3c)

## 24. Retry-exhaustion give-up is now marked `diverged` -- a correctness/reporting fix, not a
##     numerical result. `pass_count` unchanged in all four hard gates; the point of the change is
##     that three previously-finite garbage rows now correctly read NaN.

### 24.1 The decision and the change

Rounds 11, 12, and 13 each identified but deferred the same gap: `acid.cpp`'s retry-exhaustion
give-up (all 14 dt-halvings fail, and -- since round 12 -- `ACID_STALL_ACCEPT`'s own accept budget
is also exhausted or disabled) prints `STALLED:` (round 11) but leaves `diverged == false`, so
`validate`/`dump` still score the returned finite-but-garbage state as a normal completed run. This
is the exact mechanism that produced two retracted findings (§20). Each round noted that flipping
this requires "an explicit Advisor decision against the 'plain ON byte-identical' rule" because it
changes what `pass_count`'s per-case JSON reports (though not `pass_count` itself, since the
affected cases already fail) for cases 24/33/34. **The Advisor made that decision this round**:
implement it.

The change is one executable statement, `diverged = true;`, inside the existing `if (!stepped) {
... }` give-up block, plus an updated comment and a reworded (not renamed) `STALLED:` message tail.
No other file changed.

**The accept/give-up boundary needed no new logic** -- the control flow already draws it correctly.
The give-up block sits AFTER `ACID_STALL_ACCEPT`'s own accept attempt; a step that accept
successfully adopts sets `stepped = true` and never reaches the give-up block at all. So a run that
accepted some non-converged steps and then continued to `t_end` (cases 24/34 under
`ACID_STALL_ACCEPT=1`, §22) is correctly NOT marked diverged -- its own `STALL-ACCEPT-TOTAL` line
remains its honest, separate disclosure. Only "neither a clean step nor an accepted step was
possible" is now marked diverged: `ACID_STALL_ACCEPT` unset/disabled, or its own budget exhausted
(case33 `+IMPLICIT` under `ACID_STALL_ACCEPT`, §22.4: 104-251 steps, budget hit, still stalls --
correctly diverges now).

**Correction to how this was framed going in**: this block is NOT `ACID_YADV`-gated -- it sits in
the common time loop the OFF path also runs. OFF-path safety is therefore evidence-based (round 11
§21.1 measured `STALLED` firing on exactly the three known configs and nowhere else, including OFF),
not structural, so it was re-verified empirically this round (§24.2 G1), not merely inspected.

### 24.2 Gates -- all predictions confirmed exactly

| gate | result |
|---|---|
| G0 build + unit | pass |
| G1(a) OFF vs published `solver_denner` | 19/19, byte-identical |
| G1(b) STALLED/DIVERGED count on OFF (empirical, all 19 cases) | **0** |
| G1(c) OFF `denner1d_validate` stdout pre vs post | byte-identical |
| G2 plain-ON | 15/19 unchanged; **only case24/34 lines differ** (`finite: true->false`, `pass` stays `false` both times); every other case's line byte-identical |
| G3 `+ALPHA_IMPLICIT` | 14/19 unchanged; **only case33's line differs**; case24/34's lines confirmed byte-identical (they genuinely complete under this flag and must not be touched -- verified, not just asserted) |
| G4 FD-invariance | `ACID_YADV=1 ACID_NO_AJAC=1` = 13/19 and `ACID_NO_AJAC=1` alone = 13/19 -- **resolves round 12/13's ambiguous prior figure** (12/19 was stale, corrected already in round 11 §21.1 for the D config; this round confirms E independently at the same 13/19). **Zero cases changed** in either FD config -- none of the FD-invariance failure set's already-failing cases were reading a finite garbage stall value; whatever makes them fail is unrelated to this defect class. |
| G5 `ACID_STALL_ACCEPT` isolation | case24 dump **byte-identical** to round 12's; case34 dump **byte-identical** to round 12's (after redoing a shell-timeout-truncated first attempt -- a tooling artifact, not a code issue); case33 `+IMPLICIT +STALL_ACCEPT=1` now correctly shows an all-NaN dump, still prints both `STALL-ACCEPT-TOTAL` (4 accepts) and exactly one `STALLED:` line |
| G6 published scripts | `scripts/yadv_rh2.py` still classifies all three known-stalled configs as `NULL RUN` and both completing configs unchanged -- the `frac<0.9` clause alone carries the classification now that `IC-match` cosmetically reads `0.00`/`nan` instead of `0.89` (predicted; the second guard clause is now permanently dead for diverged runs, but the first still covers every known case) |

**Exactly the three predicted (case, config) pairs changed and nothing else: 24/plain, 34/plain,
33/+IMPLICIT.** No other case, in any of the five configurations tested, moved at all.

G7 (the `max_steps`-exhaustion sibling-defect sweep) was not run this round -- informational only,
deferred to whichever future round takes up that separate question (§24.4).

### 24.3 Historical-artifact audit -- corrective annotations, history left in place

Per this project's convention (§20's own retraction: "left in place above, annotated, not
deleted"), the following prior sections read a finite value from what is now understood (since §20)
to have been a stalled run, and now read NaN instead. No historical number is edited; this is the
annotation.

- **§14.3 table 1**, rows `24`/`34` (v3 column, plain `ACID_YADV=1`): those `l2_p`/`corr_rho`/etc.
  values came from case24/34's silently-stalled state (~0.27%/0.45% of `t_end`). §20 already
  retracted the accompanying RH-closure claim (bullet 2); this note extends that to the raw
  validate-metric row itself, which §20 did not touch. As of this round these rows read NaN.
- **§19.2's consolidated table**: exactly three cells -- `B` column's case24/case34 rows and `C`
  column's case33 row -- came from the same stalled reads. Every other cell in that table (case33's
  `B` column; case24/34's `C` column) comes from genuinely-completing runs and is unaffected. All
  PASS/FAIL verdicts in the table were already FAIL and remain FAIL.
- **§20.3's `NULL RUN` table**: `IC-match=0.89` -> `IC-match=0.00`, `min_p`/`min_rho` now read
  `nan` (confirmed, §24.2 G6). Classification unchanged.
- **§19.3's wall-clock table**, case24 timing described as "an early divergence exit": the
  description is now literally accurate (it always should have been, per §20); no number moves.

Already superseded, no new action: §14.3 table 2 and §19.4 (both retracted by §20.1/§20.4); §5/§6/
§10.2 (superseded code versions, not reproducible from current HEAD under any config).

Confirmed **unaffected**: §21.3/§22.5's RH-residual tables (every row from a genuinely-completing
run -- `+ALPHA_IMPLICIT`, or plain + `ACID_STALL_ACCEPT`), including round 12's headline first-ever
controlled A/B, which survives intact; §21.1/§21.2; §21.4 (stderr-parsed via a regex matching only
the unchanged message prefix); §22.4's step/accept-count table; §23.1's `RINIT`/`RMISM` numbers and
§23.2's refutation table (all stderr instrumentation, unrelated to the dump's `p`/`u`/`rho` columns).

Latent (not live) breakage, documented not fixed: `scripts/yadv_table.py`/`yadv_table3.py` lack the
lowercase-`nan`-in-JSON fix `scripts/yadv_r9_sweep.py` carries, and are already non-runnable
(hardcoded stale `/tmp/yadv_*.txt` paths from rounds 1/3). Any future script parsing `validate`
JSON output must copy that fix; noted here so a future round does not rediscover it.

### 24.4 Verdict for round 14

1. **This round is a correctness/reporting fix, not a numerical result.** `pass_count` is unchanged
   in all four hard gates. The point of the change: three previously-finite garbage dump rows
   (case24-plain, case34-plain, case33-`+ALPHA_IMPLICIT`) now correctly read NaN with
   `finite=false`, closing the exact silent-stall gap that produced two retracted findings in §20.
2. Every prediction in `docs/YADV_ROUND_14_PLAN.md` §2 was confirmed exactly; no case outside the
   three predicted pairs moved under any of the five tested configurations.
3. `ACID_STALL_ACCEPT`'s accept-and-continue path is completely unaffected (byte-identical dumps
   for case24/34), and its give-up path (case33 budget exhaustion) now correctly diverges -- the
   accept mechanism's own honesty (via `STALL-ACCEPT-TOTAL`) and this round's new correctness fix
   compose exactly as designed, with no interaction bug.
4. Historical artifacts audited; four items annotated (§24.3), none edited, none newly discovered
   to be incorrect beyond what §20 already established.
5. Two live threads remain open for a future round, un-narrowed by this one: round 13 §23.3's
   harder simultaneous `(T,rho)`-consistency re-init, and case33's still-unsolved sustained
   difficulty (§22.4). A third, newly named but explicitly NOT pursued this round: `max_steps`
   exhaustion is a sibling silent-partial-exit path (`while (t < t_end && step < max_steps)`), but
   case15 legitimately terminates via that cap and PASSES on the OFF path -- extending `diverged`
   there without further care would break the 19/19 gate outright. Left for a future round's
   deliberate design, not attempted here.
6. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, 15/19). All four hard gates
   hold.

### 24.5 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 2>&1 >/dev/null | grep STALLED
DENNER_ACID=1 ACID_YADV=1 ACID_STALL_ACCEPT=1 ./build-cpp/cpp/denner_1d/denner1d_dump 24 \
    > /tmp/c24.out 2>&1 && head -1 /tmp/c24.out   # sect.24.2 G5, still byte-identical to round 12
```

---
---

# ROUND 15 (case33 diagnosis)

## 25. Case33's stall is NOT round 13's REMAP mechanism -- it shares the SHAPE (energy-dominant,
##     dt-independent, doubling residual) but not the SOURCE. First look ever taken at what case33
##     `+ALPHA_IMPLICIT` actually does when it fails.

### 25.1 Why this had never been measured

Round 13's `ACID_RINIT` instrument was built and validated only on case24/34 (plain `ACID_YADV=1`).
Case33 stalls under a DIFFERENT config, `+ALPHA_IMPLICIT` -- and round 13's own control on that
exact flag (case24 `+ALPHA_IMPLICIT`) found `dal_remap` at literal `DBL_EPSILON`, because
`+ALPHA_IMPLICIT` re-derives alpha at the CURRENT `(p,T)` on every Newton call, eliminating the
REMAP defect structurally. Round 12 characterized case33 only qualitatively ("sustained, escalating,
221 accepted steps at budget 20, still only 14.8% of `t_end` -- not a one-time formation glitch like
24/34"). No round had run the instrument on case33's own stall until this one.

**Self-check (G4, required before trusting anything below)**: `RINIT`'s `r` at every retry of
case33's first stall matched `RHIST`'s `n0` exactly (cross-verified via the same run's independent
prints). The instrument is valid in this new configuration, not just the one it was built for.

### 25.2 Case33's first stall (step 100) -- same shape, different source

At case33's very first stall (`ACID_STALL_ACCEPT` unset, so this is a completely unconfounded read
-- every prior step converged cleanly):

| retry | dt | `dh` | `dal_remap` | `r` (RINIT) | `fene` |
|---|---|---|---|---|---|
| 0 | 6.512e-11 | 3.7277e+12 | 2.2204e-16 | 8.546e+10 | 0.647 |
| 6 | 1.018e-12 | 3.7277e+12 | 2.2204e-16 | 3.384e+12 | 0.9998 |
| 9 | 1.272e-13 | 3.7277e+12 | 2.2204e-16 | 2.712e+13 | 1.0000 |
| 13 | 7.949e-15 | 3.7277e+12 | 2.2204e-16 | 4.340e+14 | 1.0000 |

**`r` doubles almost exactly per dt-halving from retry 6 onward** (×2.002, ×2.001, ×2.000... through
retry 13) -- the SAME `1/dt` signature round 13 found for case24/34. **Energy dominates completely**
(`fene` -> 1.0000 by retry 8). Both facts match round 13's mechanism exactly.

**But `dal_remap` is at literal `DBL_EPSILON` at EVERY SINGLE RETRY, unchanged to 5+ significant
figures.** Round 13's REMAP defect (the alpha-recovery operator-splitting lag) is completely absent
here -- exactly as its own control predicted for `+ALPHA_IMPLICIT`. **Case33's stall is provably
NOT round 13's mechanism.**

What IS dt-independent and large: `dh = |s.h - Htot_o|`, constant at `3.7277e+12` across all 14
retries (down to 5 sig figs -- as flat as `dal_remap` was for case24/34, just a different
quantity). Compare to a healthy `+ALPHA_IMPLICIT` trajectory: case24's entire run (a genuine
completing case, sampled at every step) never exceeds `dh ~ 4.06e+06` at its single worst step.
**Case33's stall-time `dh` is six orders of magnitude larger than the worst value ever seen in a
healthy run of the same configuration.** This is not normal variation -- it is a genuine, severe
energy-state mismatch that has nothing to do with the alpha/Y channel.

### 25.3 The compounding test -- refined, not confirmed as originally hypothesized

Round 13 §23.3 speculated that repeated `ACID_STALL_ACCEPT` accepts might compound: "the NEXT step
inherits a slightly-off `s.alpha`/`s.h` pairing that reintroduces a fresh REMAP-like mismatch."
Tested directly (`ACID_STALL_ACCEPT=1`, following case33 through 4 consecutive accepted
non-converged steps, 100-103, to the final give-up at 104):

| step | `dh` (retry 0) | `drho` (retry 0) | `dal_remap` (retry 0) |
|---|---|---|---|
| 100 (first, unconfounded) | 3.728e+12 | 0.0771 | 2.2204e-16 |
| 103 (after 3 prior accepts) | 5.626e+12 | 0.2575 | 1.1102e-16 |
| 104 (after 4 prior accepts, final give-up) | 5.890e+12 | 0.3849 | 2.2204e-16 |

**`dal_remap` stays at machine epsilon through every one of these steps -- the specific "alpha
inherits the lag" compounding mechanism round 13 speculated about does NOT occur.** `+ALPHA_IMPLICIT`
keeps the alpha channel clean even through accepted non-converged states, exactly as its structural
property (re-derivation at every `compute_R` call) predicts.

**But `dh` and `drho` DO grow monotonically across these same steps** (`dh` +51% from step 100 to
104; `drho` +5x). **Real compounding is happening, but through the raw energy/density state
mismatch directly, not through the alpha channel round 13's hypothesis named.** This is a genuine
refinement of round 13's speculation, not a confirmation of it as originally stated.

### 25.4 The contrast, stated precisely

Case24/34 stall under plain `ACID_YADV=1` and complete under `+ALPHA_IMPLICIT`. Case33 does the
OPPOSITE: it completes under plain (§19.2 config B: `l2_p = 1.573`, the single worst-fit case in the
entire 19-case suite -- "completes" is not "clean") and stalls under `+ALPHA_IMPLICIT`. **The two
phenomena are NOT the same mechanism** (§25.2 rules that out directly, `dal_remap` clean throughout)
and this round makes **no claim** that they share a cause. The prior in `docs/YADV_ROUND_15_PLAN.md`
§4E (case33 has the SMALLEST Y->alpha amplification of the three pre-shock, `~54x` vs `~216x`/`~485x`)
is stated for the record and is consistent with case33's difficulty NOT being an alpha-amplification
problem -- but this is a prior, not a proof.

### 25.5 Verdict for round 15

1. **D-DIFF fires.** Case33's stall shares round 13's mechanism's SHAPE (energy-dominant,
   dt-independent transient, `r` doubles exactly per dt-halving) but not its SOURCE: the alpha/Y
   REMAP channel is provably clean (`dal_remap` at `DBL_EPSILON`, every retry, every sampled step,
   including after repeated forced accepts). The actual driver is a severe, growing `dh`/`drho`
   mismatch of unknown origin -- six orders of magnitude beyond any value seen in a healthy run of
   the same configuration.
2. **Round 13's speculated compounding mechanism (alpha inheriting the previous step's lag) is
   refuted for case33** -- but a DIFFERENT, real compounding is measured directly in `dh`/`drho`
   across consecutive forced-accept steps. This refines rather than confirms round 13 §23.3.
3. **Do not re-attempt `ACID_YADV_HREINIT` or any alpha/h single-field initial-guess fix on
   case33** -- it targets exactly the channel (`dal_remap`) already measured clean here; it would
   be aimed at a mismatch that does not exist in this case.
4. **This round is diagnostic only, per its own non-goals -- no fix attempted or implemented.** The
   natural next-round candidate (not implemented here): identify what specifically drives `s.h` and
   `Htot_o` (or `s.rho`/`rho_o`) apart by such a large, dt-independent, growing amount at this
   specific cell (79-81, consistently) -- likely requires a per-cell trace of the shock's actual
   strength there rather than a state-mismatch instrument, since the `RMISM`/`RINIT` tools built so
   far only characterize the SYMPTOM (a state mismatch exists) not yet its physical origin.
5. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, 15/19). No source code
   changed this round -- `git status --short -- cpp/` confirmed clean at round end (G2). No hard
   gates required or run (no source change to verify a no-op against, per
   `docs/YADV_ROUND_15_PLAN.md` §7's explicit reasoning).

### 25.6 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RINIT=1 ACID_DBG=1 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep "step=100 retry=" | grep RMISM
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_STALL_ACCEPT=1 ACID_RINIT=1 ACID_DBG=1 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep -E "^STALL-ACCEPT|^STALLED"
```

---
---

# ROUND 16 (case33's physical origin)

## 26. Case33's `dh` fully explained: a 3-cell vacuum blister born in the FIRST time step, when a
##     single Y-advection into a still-pre-shock cell maps through `alpha_from_mass_fraction`
##     almost to pure phase, collapsing the recovered density ~80x and saturating the T ceiling

### 26.1 The end-to-end mechanism, confirmed with direct field measurements (new `ACID_RCELL`
##      instrument, read-only per-cell window print)

**Born at step 0, in the very first advection step** (`ACID_RCELL=78:82 ACID_BLK_STEP=0`, retry 0):

| cell | `x` | `Y0` (pre-advection) | `Y` (post-advection) | `al0` | `al` | `rho_o` |
|---|---|---|---|---|---|---|
| 79 | 0.099375 | 0.934389 | 0.934389 | 0.750000 | 0.750000 | 1183.35 (shocked) |
| **80** | 0.100625 | **0.003466** (pristine IC) | **0.365240** | 0.750000 | **0.997989** | **3.16** |
| 81 | 0.101875 | 0.003466 | 0.003466 | 0.750000 | 0.750000 | 250.368 (pristine) |

A single time step's Y-advection moves cell 80 **36.5% of the way** from the pre-shock mass
fraction toward the post-shock one, while the cell's `(p,T)` is still entirely pre-shock (`p_o=1e5`,
`T_o=300`). The `alpha_from_mass_fraction` recovery, evaluated at THAT `(p,T)`, maps `Y=0.365` to
`alpha=0.998` -- nearly pure air -- because case33's post-shock mass fraction (`Y_post=0.9344`) sits
closest of the three cases {24,33,34} to the `alpha->1` singularity of the Y-to-alpha map (round 15
§25 computed the pre-shock amplification `dalpha/dY` as 216/54/485 for 24/33/34 -- case33's is the
smallest, but its `Y_post` proximity to the singularity dominates). The recovered density collapses
from the correct `250.368` to `3.16` -- **79x below correct, in the FIRST step, before any Newton
solve even runs.**

**By the actual stall (step 100)**, the same cell has been driven the rest of the way to a literal
vacuum (`ACID_RCELL=74:94 ACID_BLK_STEP=100`, retry 0):

| cell | `x` | `p_o` | `T_o` | `u_o` | `rho_o` | `h` | `Htot_o` |
|---|---|---|---|---|---|---|---|
| 74-78 | 0.093-0.098 | 3.6-5.8e9 (shocked) | 11700-17100 K | 4300-4800 | 860-1180 | ~2.4e7 | ~2.4e7 |
| 79 | 0.099375 | 5.701e9 | 1713 K | 7212 | 4990.0 (**20x overdense**) | 2.830e7 | 2.830e7 |
| **80** | 0.100625 | **30.03 Pa** | **1.000000e+06** | **-7127.9** | **1.237e-06** (**1.3e8x underdense**) | **3.7295e12** | **1.8908e9** |
| 81 | 0.101875 | 1971 Pa | 303.7 K | 158 | 9.32 | 6.023e5 | 6.023e5 |
| 82-94 | >=0.103 | -> 1.0e5 (pristine) | -> 300 K | -> 0 | -> 250.368 (pristine) | ~5.82e5 | ~5.82e5 |

**Every prediction confirmed exactly**: (P2, decisive) `T_o` at cell 80 reads EXACTLY
`1.000000e+06` -- the `T_from_hstat` ceiling (`acid.cpp:334-362`, confirmed by direct code read to
clamp `T` to `[1e-6,1e6]` at every sub-iteration and to `return isfinite(T) && T > 1e-6`, which is
`true` for a saturated `T=1e6` regardless of whether `hmix(T)=hstat` is actually satisfied there --
silent saturation, structurally). `h=3.7295e12` vs `Htot_o=1.8908e9` at that cell reproduces round
15's measured `dh=3.7277e12` (§25) almost to the last digit. (P3) The front is genuinely frozen at
the initial discontinuity: cells 82-94 (where the true shock should be, `Vs*t` at the stall time
puts the front near cell 90) are still pristine initial condition to 4+ significant figures --
**the shock never moves under `+ALPHA_IMPLICIT`; the solver spends its 100 steps re-fighting the
same 2-cell blister at the IC jump.** Cell 79 (20x overdense) and cell 80 (vacuum) form a 2-cell
oscillation straddling the initial discontinuity -- the post-shock fluid is draining into a
numerically-generated hole instead of forming a propagating shock.

**Quantitative closure**: pre-shock `rho*H = 250.368 * 582247 ~= 1.458e8 J/m^3`. If `rho` collapses
to the measured `1.237e-6` while `rho*H` is approximately conserved, `H ~ 1.18e14` -- same order as
the measured `3.73e12`. The enthalpy runaway is the direct arithmetic consequence of the density
collapse, not an independent defect.

Once `hstat` exceeds the phase enthalpy bound (`~1.94e9` for case33's `b=0,eta=0` phases, since
`h_k=cp_k*T` exactly and `T<=1e6`), `T_from_hstat` saturates and its derivative with respect to `h`
is silently zero there -- `dT/dh=0`, hence `drho/dh=0`. Newton can move its own energy unknown and
the thermodynamic state simply does not respond. This produces exactly round 15's symptoms
(`r_init` doubling per dt-halving, `fene->1.0`, dt-independent `dh`) with **no reference to the
alpha/Y REMAP channel round 13 found for case24/34** -- round 15's `dal_remap=DBL_EPSILON` finding
is fully consistent: the alpha channel genuinely is clean here; the defect is entirely in the
density/enthalpy collapse this section identifies.

### 26.2 `+ALPHA_IMPLICIT`-specific, not an intrinsic case-33 stiffness

OFF and plain `ACID_YADV=1` were NOT re-measured with the new instrument this round beyond what
round 15 already established (§25.2's OFF/plain controls: OFF shows nothing at cells 79-81 since
alpha is constant and nothing is transported there; plain shows a completely different, GLOBAL
drift pathology, not a localized blister). This round's new finding is specific to what
`+ALPHA_IMPLICIT` does at the SAME location: because `+ALPHA_IMPLICIT` re-derives alpha at the
current `(p,T)` on every Newton call (round 15's finding), the one-step recovery error from §26.1
gets folded directly into the Newton iteration itself rather than being an isolated,
correctable artifact -- turning a single bad recovery into a Newton-internal runaway that the
solver can never climb back out of, at any `dt`.

### 26.3 Verdict for round 16

1. Round 15's open question -- what physically produces case33's anomalous `dh` -- is answered
   completely: a single-time-step Y-advection into a still-pre-shock cell, mapped through
   `alpha_from_mass_fraction` at the WRONG (pre-shock) `(p,T)`, collapses the recovered mixture
   density catastrophically (79x in step 0, to a literal vacuum by the stall). The subsequent
   energy runaway (`dh`) is the arithmetic consequence, and the `T_from_hstat` ceiling's silent
   saturation is why Newton can never recover once the collapse is severe enough.
2. This is a genuine ill-conditioning of the Y-form colour function specifically for
   homogeneous-alpha, large-Y-jump case families (case33: alpha=0.75 both sides, `Y` jumps 270x),
   closed into the Newton iteration by `+ALPHA_IMPLICIT`'s own per-call alpha re-derivation -- not
   a simple, isolated coding bug, and not the same mechanism as round 13's case24/34 REMAP defect
   (confirmed clean here, per round 15).
3. **No fix attempted this round, per the plan's own decision rule**: a fix is justified only by
   an isolated, off-by-construction bug (e.g. a wrong-time-level lookup or an unclamped input);
   neither was found -- `Yv` is already clamped, and the `(p_o,T_o)` usage matches the documented
   Eqs.43-44 intent. Three candidate fixes are named for a FUTURE round, in pre-registered priority
   order: **F3** (break the p->alpha feedback inside the `+ALPHA_IMPLICIT` residual -- targets the
   actual amplifier, needs a new research flag + full gates) **> F1** (upper-bound `s.h` -- targets
   the symptom, risks the OFF byte-identity gate if not carefully `yadv`-gated) **> F2** (make
   `T_from_hstat` return false on saturation -- correct in principle, but the ceiling is already
   exercised on the PUBLISHED path by cases 13/14/25/28/29's own transient violent shocks, so
   changing its return value is not a safe drive-by and needs its own dedicated round).
4. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, 15/19). New
   diagnostic-only `ACID_RCELL` env var stays default OFF; all hard gates hold with it unset.

### 26.4 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RCELL=78:82 ACID_BLK_STEP=0 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep "retry=0 "   # sect.26.1, step 0
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RCELL=74:94 ACID_BLK_STEP=100 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep "retry=0 "   # sect.26.1, stall
```

---
---

# ROUND 17 (F2 risk assessment)

## 27. F2 is V-SAFE on the entire published OFF path -- but F2 as literally named is the WRONG
##     SHAPE regardless. Corrected form F2'' is now the pre-registered candidate.

### 27.1 Round 16's stated risk basis was inaccurate -- corrected here, not edited there

Round 16 §26.3 named F2 risky because "the ceiling is already exercised on the PUBLISHED path by
cases 13/14/25/28/29's own transient violent shocks." Measured this round: **case29 is not in the
graded suite at all** (`cases.cpp:569-593`, excluded -- its own comment records an unexplained
blocker, see §27.4). Of the remaining four, **13/14/25 sit 3-4 orders of magnitude below the 1e6 K
ceiling** in their converged state (4.8e2 / 8.6e2 / 1.4e4 K), and **case28 (Ms=100 air) is the only
one within a factor of 2 analytically** (0.587x) -- its converged state is still 41% below the
ceiling. This correction is recorded here per this project's convention (annotate, don't edit
history).

### 27.2 New instrument `ACID_TSAT` -- deliberately NOT `yadv`-gated (must observe OFF)

Unlike `ACID_RCELL`/`ACID_RINIT` (round 13/16, both `yadv`-gated and therefore structurally unable
to observe the OFF path -- a reusable fact worth recording for future rounds), `ACID_TSAT` probes
`s.T[i] >= 1.0e6` immediately after the coupled h->T inversion loop closes, on every path. Counts
residual-evaluation-level saturation (`calls_hi`), accepted-state saturation, and final-state
saturation. Verified NOT to perturb the solve: `ACID_TSAT=1` on a full OFF `denner1d_validate` run
produces byte-identical stdout to the unset baseline (G6) -- the strongest form of this project's
standard non-perturbation gate, since this flag actually executes on the path being checked (unlike
`ACID_RCELL`'s gate, which only proves the block is skipped).

**Positive control (G9, mandatory before trusting any other number)**: `ACID_TSAT=1` on case33
`+ALPHA_IMPLICIT` reports `calls_hi=13719`, `first_hi_cell=80` -- an exact match to round 16's
directly-measured cell (§26.1's vacuum cell). The instrument is validated against an independent,
already-published finding.

### 27.3 Main measurement -- clean V-SAFE, all 19 OFF cases, zero exceptions

| case | `calls` | `calls_hi` | `calls_lo` |
|---|---|---|---|
| 01/02/04/05/07 | 960-125494 | **0** | **0** |
| 13/14/15 | 3694-7306 | **0** | **0** |
| 24/25/26/27 | 10630-31763 | **0** | **0** |
| **28** | **29975** | **0** | **0** |
| 30/31/33/34/35/36 | 4190-41370 | **0** | **0** |

**Every one of the 19 graded cases, across every residual evaluation of a full OFF-path run
(totalling >400,000 calls), shows zero cells ever reaching the `T_from_hstat` ceiling.** Case28 --
the analytically closest case (0.587x) -- shows zero hits across its own 29,975 calls, meaning even
its worst TRANSIENT Newton iterate never actually touches the clamp, not just its converged state
(which §27.1 already established via the independent density-inversion method). `calls_lo` (the
EXISTING lower-saturation signal, already returning `false` today) is also zero everywhere on OFF --
meaning the existing asymmetric failure signal is not currently exercised at all on the published
path; F2/F2'' would activate a code path that has literally never fired in this suite's history.

**Verdict: V-SAFE.** The branch F2/F2'' would change is PROVABLY never taken on the entire published
OFF path. Round 16's stated caution, while a reasonable prior before measurement, does not survive
direct measurement.

### 27.4 But F2 as literally specified is the wrong shape -- independent of the measurement

`T_from_hstat`'s only consuming call site (`acid.cpp:1216`, confirmed the sole `s.T` update on the
published path since `coupled==true` unconditionally for all 19 graded cases, `cases.cpp:28`'s
`unic=true` with no per-case override) treats `false` as "keep `s.T[i]` at its PREVIOUS value". This
makes `compute_R` a function of call HISTORY, not of state alone -- breaking the four
`compute_R(); // restore` sites (`acid.cpp:1592,1685,2062,2129`) that assume re-evaluating from the
same `(u,p,h)` reproduces the same residual, load-bearing for the FD-Jacobian assembly at `:1685`.
Worse: freezing `T` gives `dT/dh=0` **exactly and by construction** -- the very failure mode round
16 §26.3 diagnosed as the reason Newton can never recover. F2 as named would make the death-of-
derivative pathology MORE certain, not less.

**Corrected form, F2'' (supersedes F2 in the priority list -- now F3 > F2'' > F1)**: keep
`T_from_hstat` state-pure (still returns the clamped T), but additionally report saturation to the
CALLER, which treats "any cell saturated in the accepted iterate" as a NEW stall reason (5, "T
ceiling saturated") triggering the EXISTING dt-halving retry machinery -- composing with the
existing reason 1-4 taxonomy (`acid.cpp:741`ish) and round 14's `diverged` marking, without
touching residual purity anywhere.

### 27.5 Side finding -- case29's likely root cause, for the record (not pursued)

Case29 (Ms=100 water, excluded) has an analytic post-shock temperature of **2.932e6 K -- 2.93x
ABOVE the solver's own 1e6 K clamp.** Its initial condition is not representable by the solver's own
thermodynamic clamp from step 0, which very plausibly explains the `cases.cpp:591` blocker comment
("dt collapses ~1e-9, front under-resolved") that has never previously been explained in this
project's history. Not pursued this round (would require raising a global physical clamp, a
separate decision affecting the published OFF path); recorded so a future round doesn't re-derive
it from scratch.

### 27.6 Verdict for round 17

1. **V-SAFE, confirmed with zero exceptions across all 19 graded cases.** F2/F2'' is provably
   byte-identical-safe on the published OFF path.
2. **F2 as originally named is superseded by F2'' in the priority list** (now F3 > F2'' > F1) --
   an architectural correction independent of the measurement, found by reading the four
   `compute_R(); // restore` call sites.
3. **No fix implemented this round** (diagnostic only, per this round's own bar: "essentially free
   AND fully specified" -- not met, since F2 needed correcting to F2'' before any implementation
   would be sound). F2'' is now well-specified enough for a future round to implement directly,
   with the risk assessment already done.
4. Case29's likely root cause identified for the record (§27.5), not pursued.
5. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, 15/19). New
   diagnostic-only `ACID_TSAT` stays default OFF; all hard gates hold with it unset, and G6 (the
   critical non-perturbation check) confirms it is a true no-op even when it actually executes on
   OFF.

### 27.7 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_TSAT=1 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep TSAT-TOTAL   # sect.27.2 G9
for c in 01 02 04 05 07 13 14 15 24 25 26 27 28 30 31 33 34 35 36; do
  DENNER_ACID=1 ACID_TSAT=1 ./build-cpp/cpp/denner_1d/denner1d_dump $c 2>&1 >/dev/null | grep TSAT-TOTAL
done   # sect.27.3's full table
```

---
---

# ROUND 18 (F2'' implementation)

## 28. F2'' implemented -- OFF/plain-ON/`+ALPHA_IMPLICIT` confirmed byte-identical as predicted;
##     the FD-invariance path was NOT a no-op as predicted -- it RECOVERS two previously-failing
##     cases; and case33's `ACID_STALL_ACCEPT` combination completes cleaner and earlier as
##     predicted, while case34's shows a real, small, honestly-reported perturbation

### 28.1 What was implemented

Round 17 (§27.4) pre-registered the corrected fix F2'' (supersedes the originally-named F2): keep
`T_from_hstat` state-pure (no signature change -- deliberately, see the design rationale in
`docs/YADV_ROUND_18_PLAN.md` §1), but treat "any cell pinned at the 1e6 K ceiling in the state a
retry's Newton solve just produced" as a NEW stall reason (5, `T-ceiling-saturated`), inserted into
the existing `bad`-determination logic between the reason-1 assignment and the finite/speed scan
(precedence 2/3/4 > 5 > 1). Because reason 5 displaces reason 1, a saturated retry is automatically
ineligible for `ACID_STALL_ACCEPT`'s best-iterate capture -- no separate code change was needed at
that site. New env var `ACID_TSAT_STALL` (default 0/OFF, global, deliberately NOT `yadv`-gated so
the OFF-path no-op claim is testable rather than structural, matching round 17's `ACID_TSAT`
precedent). Total diff: one executable statement plus four comment-only hunks.

### 28.2 Stage 0 -- the saturation landscape was NOT what round 17 (OFF-only) or this round's own
##      plan predicted for the non-OFF configurations

Round 17 measured `calls_hi=0` for all 19 cases on the default OFF path only. This round swept all
19 cases across FIVE configurations for the first time (`ACID_TSAT=1`, itself a proven no-op, round
17 G6): OFF, OFF+FD (`ACID_NO_AJAC=1` alone), plain-ON, plain-ON+FD, `+ALPHA_IMPLICIT`.

**OFF and plain-ON: zero saturation everywhere, exactly as predicted.** (plain-ON showed two
isolated intermediate-iterate touches on cases 28/34, both with `accepted_steps_hi=0` -- confirmed
harmless, see §28.3.)

**`+ALPHA_IMPLICIT`: case33 saturates (`calls_hi=13719`, matching round 17's G9 control exactly);
cases 24/34 do not (`calls_hi=0`); case28 shows 3 isolated intermediate touches, `accepted_steps_hi=0`.**
Exactly as predicted.

**FD-invariance (`ACID_NO_AJAC=1`, with and without `ACID_YADV=1`): NOT what the plan predicted.**
Both FD configurations show substantial, genuine, ACCEPTED-state saturation on cases 24, 27, 28,
33, 34 -- e.g. plain-ON+FD case27: `accepted_steps_hi=150`, `final_cells_hi=19`; case28:
`accepted_steps_hi=144`, `final_cells_hi=13`. **This means the FD-Jacobian path, completely
independent of `ACID_YADV`, was ALREADY silently accepting thermodynamically-saturated states as
"converged" on several of the hardest cases in the suite before this round.** The plan's prediction
("FD-invariance -> 13/19 byte-identical, conditional on Stage 0's FD columns") is corrected here,
not edited there.

### 28.3 Gate results

| Gate | Flag OFF | Flag ON | Match to prediction |
|---|---|---|---|
| OFF (headline) | 19/19 | **19/19, byte-identical** | Confirmed exactly (deductive from Stage 0 + round 17) |
| plain-ON | 15/19 | **15/19, byte-identical** | Confirmed exactly |
| `+ALPHA_IMPLICIT` | 14/19 | **14/19, byte-identical** (including case33's row -- already all-NaN both ways, round 14's Stage 3c) | Confirmed exactly |
| FD (`ACID_YADV=1 ACID_NO_AJAC=1`) | 13/19 | **15/19** -- cases 27 AND 28 flip `finite:false->true, pass:false->true` | **NOT predicted byte-identical; a genuine correctness recovery**, not just a reporting change |
| FD (`ACID_NO_AJAC=1` alone) | 13/19 | **14/19** -- case 27 flips `finite:false->true, pass:false->true` | Same |

**No regression anywhere in any of the five configurations.** Every change is either zero (three
configs) or a strict improvement (two configs, two cases recovered). Cases 27/28 (both Ms=100
shocks, the two most extreme in the graded suite) were previously silently producing NaN under the
FD-Jacobian path -- exactly the mechanism round 16 §26.1 diagnosed for case33 (a cell saturates,
`dT/dh=0`, the state cannot self-correct, and eventually the accumulated defect manifests as a
divergence elsewhere) -- and F2'' catches it at the source, forces a retry, and the retry succeeds
at a smaller `dt` where the unmodified code was silently propagating a state the EOS could not
represent.

**G8 (mandatory positive control)**: case33 `+ALPHA_IMPLICIT` with `ACID_TSAT_STALL=1 ACID_DBG=1`
prints `STALLED-DETAIL: reason=T-ceiling-saturated cell=80 ... T=1.0000e+06` -- confirmed. The
failure now occurs at **step 43** (not step 0 as the plan speculated might happen, and not step 100
as under the unmodified code) -- an earlier, more honestly-attributed failure than before, though
not the earliest theoretically possible; the retry machinery evidently recovers from several
transient saturations before finally exhausting all options at step 43.

**`ACID_STALL_ACCEPT` interaction (G7)**: case24 -- byte-identical, identical `STALL-ACCEPT` line
count (2), as predicted. Case33 -- **zero `STALL-ACCEPT` lines with the flag on (down from 4
without it)**, and the run now gives up at step 43 instead of grinding through the accept budget to
step 104 -- exactly the predicted "fails faster and cleaner, not a different final outcome" (case33
never completed either way). **Case34 -- NOT byte-identical, contrary to prediction**: the same 4
`STALL-ACCEPT` events fire at the same steps with identical `(retry, dt, rbest, r_init, ratio)`
values, but the full trajectory differs from the ~6th significant figure onward (e.g.
`p=49150082929.4` vs `49150271487` at the first output row) while both runs still complete to
`t_end` and land on the same physical plateau to high precision. This is reported as measured, not
minimized: a transient saturation somewhere outside the accept-budget window (Stage 0's plain-ON
sweep found `calls_hi=78` for case34, all non-accepted, at `first_hi_step=14`) evidently causes one
extra retry rejection under `ACID_STALL_ACCEPT`'s dynamics that does not occur in plain-ON alone,
producing a small but real, non-byte-identical perturbation. **The published, gate-verified round-12
`ACID_STALL_ACCEPT` numbers for case24/34 in plain-ON (without the new flag) are unaffected** -- this
deviation only appears when BOTH `ACID_STALL_ACCEPT` and `ACID_TSAT_STALL` are set together, which
is not any currently-recommended configuration.

**G6 (malformed input)**: empty/`abc`/`-3`/`0` all correctly disable the mechanism, zero
`T-ceiling-saturated` triggers in every case -- confirmed.

**G9 (accepted-state cross-check invariant)**: swept the full 20-combination matrix (five
highest-risk cases 24/27/28/33/34 x four configurations) with both `ACID_TSAT=1
ACID_TSAT_STALL=1` set -- computationally expensive (some pairs individually require hundreds of
thousands of residual evaluations, per §28.2's table). **Zero violations of the
`accepted_steps_hi==0` invariant across the entire completed sweep**, confirming the deductive
argument in `docs/YADV_ROUND_18_PLAN.md` §3 (a saturated retry cannot be captured as `acc_have` by
construction) empirically, not just logically. Case33's G7 result (zero `STALL-ACCEPT` lines with
the flag on, where the unmodified run had four) is the same invariant observed directly on the one
case in the graded suite where it is actually load-bearing.

**G10 (diff hygiene)**: `git status --short -- cpp/` shows only `acid.cpp` modified, matching the
five-hunk spec exactly.

### 28.4 Verdict for round 18

1. **F2'' is implemented, default OFF, and every headline gate (OFF/plain-ON/`+ALPHA_IMPLICIT`)
   confirmed byte-identical as predicted** -- round 17's V-SAFE measurement holds up under the
   actual new code path, not just the diagnostic instrument that measured it.
2. **Unexpected, positive result**: the FD-Jacobian configuration was NOT a no-op -- it recovers
   cases 27 and 28 from silent NaN failure to genuine PASS. This was not anticipated by round 17's
   OFF-only measurement or this round's own plan, and is recorded honestly as a deviation from the
   pre-registered prediction, in the favorable direction.
3. **`ACID_STALL_ACCEPT` interaction**: case33 behaves exactly as predicted (fails faster, cleaner,
   same ultimate non-completion). Case24 is byte-identical as predicted. **Case34 shows a small,
   real, non-byte-identical perturbation when combined with `ACID_STALL_ACCEPT`** -- reported
   honestly rather than minimized; does not affect any currently-published configuration (round 12's
   numbers use `ACID_STALL_ACCEPT` alone, without the new flag).
4. **No regression found anywhere, in any tested configuration.** `ACID_TSAT_STALL` stays default
   OFF pending a future round's decision on promotion -- this round establishes it is safe and
   beneficial when explicitly enabled, not that it should become default.
5. `ACID_YADV`'s recommended default status is UNCHANGED (default OFF, 15/19).

### 28.5 Reproducing

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
./build-cpp/cpp/denner_1d/denner1d_unit
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_TSAT_STALL=1 ACID_DBG=1 \
    ./build-cpp/cpp/denner_1d/denner1d_dump 33 2>&1 >/dev/null | grep -E "^STALLED:|^STALLED-DETAIL:"
DENNER_ACID=1 ACID_YADV=1 ACID_NO_AJAC=1 ./build-cpp/cpp/denner_1d/denner1d_validate       # 13/19 (flag off)
DENNER_ACID=1 ACID_YADV=1 ACID_NO_AJAC=1 ACID_TSAT_STALL=1 ./build-cpp/cpp/denner_1d/denner1d_validate  # 15/19 (flag on)
```
