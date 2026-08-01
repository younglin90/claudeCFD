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
