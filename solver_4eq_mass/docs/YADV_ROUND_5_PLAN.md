# Round 5 Execution Brief — Phase 2 Stage 0

Produced by Agent(subagent_type="Plan", model=opus) during round 5 of the `yadv-round` loop.
Spot-checked by the Advisor session against the actual code before use (scripts/yadv_r3_build.sh's
hardcoded main-tree path, eos.hpp lines 53-62, acid.cpp J1 loop 1503-1515, unit.cpp insertion
point — all confirmed correct). Saved verbatim below.

---

# Stage-0 Execution Brief — Round 5 (`docs/YADV_PHASE2_PLAN.md` Stage 0)

Verified against the code as it stands in this worktree (`767252e`). All paths absolute.
**Every Phase-2 line anchor I checked is still correct.** Corrections and additions are marked ⚠.

---

## 0. Environment gotchas — read before running anything

**0.1 ⚠ The worktree has NO `build-cpp`, and every `scripts/yadv_*.sh` hardcodes the MAIN tree.**
`/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-5/solver_4eq_mass/scripts/yadv_r3_build.sh` line 4 is
`W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass` — running it from the worktree wipes and rebuilds the **main** tree, not the round's. 24 of the 31 `scripts/yadv_*` files hardcode that path. **Do not run any of them unmodified this round.** Use explicit `cmake -S <worktree>/solver_4eq_mass -B ...` commands (§4.1).

**0.2 ⚠ `rm -rf` is blocked by the round-loop hook** (`solver_4eq_mass/.claude/hooks/block_destructive_bash.py`, `DENY_PATTERNS`) whenever `solver_4eq_mass/.claude/round-loop-active` exists (it does not right now). `bash scripts/yadv_r3_build.sh` would *bypass* that hook (the deny regex matches the Bash command string, not the script body) — another reason not to use it. No `rm -rf` is needed anyway: the worktree build dir does not exist yet, so the first configure is already clean.

**0.3 The byte-identity reference binary is outside the worktree** and is correct as-is:
`/home/younglin90/work/claude_code/claudeCFD/solver_denner/build-cpp/cpp/denner_1d/denner1d_dump` (dated Jul 14). `scripts/yadv_verify.py` compares `MINE` (main tree) vs `REF`; for this round `MINE` must be re-pointed at the worktree build. It is a 3-line constant edit (`MINE = ".../worktrees/yadv-round-5/solver_4eq_mass"`), and it is fine to make a `scripts/yadv_r5_verify.py` copy — Stage 4 asks for `yadv_r5_*` scripts anyway.

---

## 1. Confirmed file:line anchors

### `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-5/solver_4eq_mass/cpp/denner_1d/include/denner1d/eos.hpp`

| what | lines | status |
|---|---|---|
| `PhaseProps` struct (`zeta`, `phi` are members 5,6) | 7-18 | correct |
| `phase_thermo` lean helper | 26-37 | correct |
| alpha<->Y doc comment block | 39-52 | correct |
| `mass_fraction_from_alpha` | 53-57 | correct |
| `alpha_from_mass_fraction` | 58-62 | correct |
| **insertion point for the new helpers** | **after line 62, before line 64 (`Phase air_phase();`)** | |

### `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-5/solver_4eq_mass/cpp/denner_1d/src/acid.cpp` (2040 lines)

| plan's claim | plan's lines | actual | status |
|---|---|---|---|
| `eval_thermo` alpha blends (R1) | 311-326 | 306-327 (loop body); `rho` 314, `hstat` 316, `cp` 317, `s.a` 325, `s.drhodp` 326 | correct (minor) |
| `T_from_hstat` definition (Stage 3b target) | 334-362 | 334-362 | exact |
| implicit-alpha residual block | 1014-1022 | 1014-1022 | exact |
| h->T inversion / `T_from_hstat` call (R2) | 1033 | 1026-1038, call at **1033** | exact |
| `use4` phase test (R5) | 1096 | 1096 | exact |
| MWI clamp bound `af` (R6) | 1117 | 1117 | exact |
| ACID per-cell mass flux `mdotL/mdotR` (R3) | 1172-1177 | loop **1172-1176**, BC overrides 1177-1179 | correct (+-1) |
| coupled energy flux blend (R4) | 1204-1206 | **1204-1207** | correct (+-1) |
| `bool ajac` | -- | 1268 | |
| `if (tr_bdf2) ajac = false;` (§0.2) | 1274 | **1274** | exact |
| `const bool ajblk = getenv("ACID_AJAC_BLK")` | -- | **1278** | |
| `ACID_MNEWTON` `Kmn` (default 2) | -- | 1288 | |
| iter cap `ajac ? 150 : 40` | -- | 1356 | |
| `do_fd_assembly` gate | 1445 | **1445-1446** | exact |
| `if (ajac \|\| ajblk)` analytic block start | 1487 | **1487** | exact |
| **J1 cell EOS chain** | **1503-1515** | **1503-1515** | exact |
| transient diagonal | 1516-1521 | 1516-1521 | exact |
| flux-coupling (frozen transport) | 1544-1562 | 1544-1562 | exact |
| upwind-transport derivatives | 1567-1595 | 1567-1595 | exact |
| frozen-MWI d(theta)/d(rho) | 1600-1632 | 1600-1632 | exact |
| `ajblk` report `rep()` | -- | 1633-1659, **default `ACID_BLK_STEP` = 40** (line 1636) | |
| `MA3=aA; ...` install (skipped under ajblk) | -- | 1661-1664 | |
| `int step = 0` / `++step` | -- | 467 / 2016 | |

### Others
- `cpp/denner_1d/src/eos.cpp`: `phase_props` 25-43; **`out.zeta` line 35, `out.phi` line 36** (plan cited `eos.cpp:36` for the `phi_k/rho_k = -kv(gamma-1)/A` identity -- correct).
- `cpp/denner_1d/tests/denner1d_unit.cpp` (115 lines): existing `(p,T,Y,pair)` grid block **53-83** (`pairs[][2] = {{air,water},{water,air}}` at 55; `p0 in {1e4,1e5,8e6,1e9}` at 57; `T0 in {250,300,360,1200}` at 58). **Insert the new Stage-0 block after line 83, before line 85.**
- `cpp/denner_1d/src/cases.cpp`: `denner_water{4.1, 4.4e8, 0.0, 474.2, 0.0}` at **446**; case table 573-611.
- `include/denner1d/types.hpp:8-14`: `Phase{gamma, pinf, b, kv, eta}`.

---

## 2. Three factual corrections / sharpenings to the Phase-2 plan

These change what Stage 0 must assert and what it should expect to measure. **They are not optional.**

### 2.1 `a_T` is NOT exactly zero in floating point -- the plan's "assert *exact* zero" test WILL FAIL

Phase-2 §0.5 / Stage-0 deliverable 2 bullet 3 says: *"the exactness claim `a_T == 0` for `b_a == b_b == 0` (air|vapor pair) -- assert **exact** zero"*. The claim is true **algebraically** (`phi_k/rho_k = -1/T` for every `b=0` phase), but `phase_props` computes it as `(-ppinf*kv*gm1*inv_A2) / (ppinf/A)` (eos.cpp:30,36), which does not round-trip `ppinf` or `A` exactly.

Measured over the full b=0 grid (air|vapor, air|denner_water, air|helium, both orders; `p in {1e4,1e5,8e6,1e9}`, `T in {6.94,250,300,360,1200}`, `alpha` in 1001 steps):

```
worst |a_T|                                              = 1.39e-17
worst |a_T| / (eps * al*(1-al) * max|phi_k/rho_k|)       = 1.735
```

**Correct assertion (use this, not `== 0.0`):**
```
|a_T| <= 8 * DBL_EPSILON * al*(1-al) * max(|phi_a/rho_a|, |phi_b/rho_b|)
```
(4.6x margin over the measured worst case, same "eps x condition number" idiom the existing round-trip test at unit.cpp:62 already uses.)

**What IS exactly zero and SHOULD be asserted with `==`:** the endpoint case. `al == 0.0` gives `0.0 * 1.0 * x`; `al == 1.0` gives `1.0 * 0.0 * x` -- both exactly `+0.0` for any finite `x`. That is deliverable-2 bullet 4 and it is genuinely exact.

### 2.2 ⚠ Only cases **14 and 15** have a non-zero `a_T` at all -- 17 of 19 are b=0/b=0 pairs

`cases.cpp:573-611`: every case uses `denner_water{..., b=0.0, ...}` (line 446), `air` (b=0), `helium`, `helium_pure`, `argon`, `matched_gas`, `denner_gas2` -- all `b = 0`. Only **case 14** (`air, water`, line 581) and **case 15** (`air, water`, line 582) use `water_liquid_phase()` with `b = 6.61e-4`.

Consequence: **Stage 3 (the T-pathway) can only ever affect cases 14 and 15.** This is a much sharper statement than §0.5's ("only `water_liquid_phase` has b!=0") and should be recorded -- it makes Stage 3's contingency test trivially decidable.

### 2.3 ⚠ The "~400x on a mixed interface cell of 13/25" prediction is state-dependent and **partly wrong for case 25**

Computed exactly from `phase_props`' own formulas (air | denner_water, T=300 unless noted):

| state | `alpha` | `D_p` (frozen, what the Jacobian uses) | `D_p + (ra-rb)*a_p` (correct) | ratio |
|---|---|---|---|---|
| case13/25 pre-shock region, p=1e5 | 0.5 | 6.92e-6 | 2.50e-3 | **361** |
| case13 low-p side, p=1e4 | 0.5 | 6.92e-6 | 2.50e-2 | **3606** |
| p=1e6 | 0.5 | 6.92e-6 | 2.54e-4 | 36.6 |
| p=1e8 | 0.5 | 6.92e-6 | 7.06e-6 | 1.02 |
| case13 HP-air side, p=1e9 | 0.5 | 6.92e-6 | 6.28e-6 | **0.91** (correction *reduces* it) |
| **case25 post-shock, p=1.165e7, T=6114** | 0.5 | 3.40e-7 | 1.25e-6 | **3.69** |
| pure cell, any p, alpha=1e-6 or 1-1e-6 | -- | -- | -- | **~1.00** |

So: the defect is huge in the **low-pressure mixed** cells and essentially absent at `p >~ 1e8` and in pure cells. **Case 25's post-shock mixed cells show only ~3.7x**, not ~400x -- its material interface at `x=0.5` sits at `p=1e5` until the shock arrives, where it *is* ~361x. Record this; do not report the plan's 400x figure for case25 without qualification.

### 2.4 The §1 predictions that ARE confirmed exactly (use these as the unit-test oracle)

Computed from the code's own EOS with the case IC states:

```
case15 IC: T_air = 267.0013 K, T_water = 352.9754 K, T_mix = 0.055*Ta + 0.945*Tw = 348.2468 K
  at p=1e5, T=348.2468, alpha=0.055, air|water(NASG):
    ra = 0.996712        rb = 1004.56       rho = 949.366
    zeta_a/ra = 1.0e-05  zeta_b/rb = 4.77998e-10
    phi_a/ra = -2.87153e-03  phi_b/rb = -9.64788e-04
    a_p = -5.19725e-07 /Pa      a_T = +9.91027e-05 /K
    D_p (frozen)  = 1.00196e-06
    D_p*(correct) = 5.22580e-04     ratio = 521.56          <-- plan said "~500x"  CONFIRMED
    identity (§1): |LHS-RHS|/|RHS| = 2.1e-16
    D_T (frozen)  = -0.916042 ; D_T* = -1.01550 ; ratio = 1.109
    hstat = 314542.22 , hs_al = 40.6756 , hsT = 4284.88 , hsp = 6.82102e-04
    Picard loop gain |hs_al * a_T / hsT| = 9.41e-07          <-- plan estimated "~3e-6"  CONFIRMED

  air|water, p=1e5, T=300, alpha=0.5 (generic mixed interface cell):
    a_p = -2.49989e-06 ; D_p = 6.01254e-06 ; D_p* = 2.63702e-03 ; ratio = 438.59  <-- "~400x" CONFIRMED
    Picard loop gain = 1.157e-04
```

**The §1 algebraic identity is proven, not just numerically checked.** With `kappa_k := zeta_k/rho_k`:
```
LHS = al*Za + (1-al)*Zb + (ra-rb)*al(1-al)(Zb/rb - Za/ra)
    = al^2*Za + (1-al)^2*Zb + al(1-al)(ra*Zb/rb + rb*Za/ra)
RHS = (al*ra + (1-al)*rb)(al*Za/ra + (1-al)*Zb/rb)   -> same expansion.  QED
```
It is purely algebraic in `(Za, Zb)`, so **the identical identity holds with `zeta -> phi`** -- the unit test should assert both forms (the T version is a free extra check, and it is the *only* nontrivial check available for `a_T` on b!=0 pairs).

---

## 3. Exact code to add

### 3.1 `eos.hpp` -- insert after line 62 (i.e. between `alpha_from_mass_fraction` and `Phase air_phase();`)

```cpp
// ---- derivatives of alpha(Y, rho_a(p,T), rho_b(p,T)) at FIXED mass fraction Y ---------------
// From alpha = Y*rb / D with D = ra(1-Y) + Y*rb, the two identities alpha = Y*rb/D and
// 1-alpha = ra(1-Y)/D give
//     d(alpha)/d(ra) = -Y*rb*(1-Y)/D^2 = -alpha(1-alpha)/ra
//     d(alpha)/d(rb) = +Y*ra*(1-Y)/D^2 = +alpha(1-alpha)/rb
// Chaining through rho_k(p,T) with the EXISTING PhaseProps partials zeta = drho/dp|_T and
// phi = drho/dT|_p:
//     a_p = d(alpha)/dp|_{T,Y} = alpha(1-alpha) * ( zeta_b/rho_b - zeta_a/rho_a )
//     a_T = d(alpha)/dT|_{p,Y} = alpha(1-alpha) * ( phi_b /rho_b - phi_a /rho_a )
// zeta_k/rho_k is phase k's ISOTHERMAL compressibility (= 1/p for an ideal gas), -phi_k/rho_k
// its thermal expansivity. a_p < 0 for gas-in-liquid: compress -> gas volume fraction falls.
// The alpha(1-alpha) prefactor vanishes EXACTLY at both pure ends (a multiply by 0.0), so these
// are automatically consistent with the clamp(alpha,0,1) the residual applies -- no epsilon, no
// kink handling, no new constant.
// NOTE on a_T and the NASG covolume b: phi_k/rho_k = -kv_k(gamma_k-1)/A_k with
// A_k = kv_k(gamma_k-1)T + b_k(p+pinf_k) (eos.cpp:27,36), which is exactly -1/T for ANY phase
// with b_k = 0. So a_T is ALGEBRAICALLY zero whenever both phases have b == 0 -- which is 17 of
// this suite's 19 cases (only cases 14 and 15 use water_liquid_phase, b = 6.61e-4). It is NOT
// bitwise zero, because phase_props evaluates phi/rho as (-ppinf*kv*gm1/A^2)/(ppinf/A); the
// residual is <= ~2*eps*alpha(1-alpha)*|phi/rho| (measured worst 1.4e-17 over the suite grid).
//
// Cross-check identity (asserted in denner1d_unit.cpp), exact for BOTH zeta and phi:
//     D_p + (rho_a - rho_b)*a_p  ==  rho * ( alpha*zeta_a/rho_a + (1-alpha)*zeta_b/rho_b )
//     where D_p = alpha*zeta_a + (1-alpha)*zeta_b is the FROZEN-alpha value acid.cpp uses today,
//     and the RHS is the (isothermal) Wood-type mixture compressibility.
struct AlphaDerivs { double a_p; double a_T; };
inline AlphaDerivs alpha_derivs_massfrac(double alpha,
                                         double zeta_a, double phi_a, double rho_a,
                                         double zeta_b, double phi_b, double rho_b) {
    const double w = alpha * (1.0 - alpha);
    AlphaDerivs o;
    o.a_p = w * (zeta_b / rho_b - zeta_a / rho_a);
    o.a_T = w * (phi_b  / rho_b - phi_a  / rho_a);
    return o;
}
inline double dalpha_dp_massfrac(double alpha, double zeta_a, double rho_a,
                                 double zeta_b, double rho_b) {
    return alpha * (1.0 - alpha) * (zeta_b / rho_b - zeta_a / rho_a);
}
inline double dalpha_dT_massfrac(double alpha, double phi_a, double rho_a,
                                 double phi_b, double rho_b) {
    return alpha * (1.0 - alpha) * (phi_b / rho_b - phi_a / rho_a);
}
```

Notes for the implementer:
- Provide **all three** (the struct + the two scalars). Stage 1 wants only `a_p`; Stage 3 wants both; the struct avoids recomputing `alpha*(1-alpha)` twice in the hot J1 loop.
- Argument order in `alpha_derivs_massfrac` is `(zeta, phi, rho)` per phase so a call site can write `pa.zeta, pa.phi, pa.rho`.
- **Do not** clamp `alpha` inside the helpers -- the caller already does `std::clamp(s.alpha[i], 0.0, 1.0)` at acid.cpp:1504 and passing the clamped value is what makes the endpoint-exact property meaningful.
- Zero call sites this stage. `-Wall -Wextra -Wpedantic` is on (CMakeLists) but `inline` functions in a header are not `-Wunused`.

### 3.2 `tests/denner1d_unit.cpp` -- insert after line 83 (after the closing `}` of the round-trip block, before `std::vector<double> a{1.0, 2.0, 3.0};` at line 85)

```cpp
    // --- Phase 2 Stage 0: d(alpha)/dp, d(alpha)/dT at fixed mass fraction Y ----------------
    // Four properties, over the SAME (p,T,pair) grid the round-trip test above uses:
    //   (1) a_p, a_T reproduce a central FD of alpha_from_mass_fraction o phase_props;
    //   (2) the exact algebraic identity  D_p + (ra-rb)*a_p == rho*(al*za/ra + (1-al)*zb/rb),
    //       and its zeta->phi twin, to ~1e-12 relative;
    //   (3) for a b==0 / b==0 phase pair, a_T is zero to within the cancellation floor
    //       (~2*eps*al(1-al)*|phi/rho|). NOT bitwise zero: phase_props evaluates phi/rho as
    //       (-ppinf*kv*gm1/A^2)/(ppinf/A), which does not round-trip ppinf and A exactly.
    //       Measured worst over this grid: 1.39e-17 abs, 1.74*eps relative. See eos.hpp.
    //   (4) alpha in {0,1} => a_p == a_T == +0.0 EXACTLY (a multiply by 0.0), which is what
    //       makes these derivatives consistent with the residual's clamp(alpha,0,1).
    {
        const auto vapor = denner1d::water_vapor_phase();
        const double macheps = 2.220446049250313e-16;
        const denner1d::Phase pairs2[][2] = {{air, water}, {water, air},
                                             {air, vapor}, {vapor, air}};
        double worst_fd_p = 0.0, worst_fd_T = 0.0, worst_id = 0.0, worst_bzero = 0.0;
        for (const auto& pr : pairs2) {
            const bool both_b_zero = (pr[0].b == 0.0 && pr[1].b == 0.0);
            for (const double p0 : {1.0e4, 1.0e5, 8.0e6, 1.0e9}) {
                for (const double T0 : {250.0, 300.0, 360.0, 1200.0}) {
                    const auto pa = denner1d::phase_props(p0, T0, pr[0]);
                    const auto pb = denner1d::phase_props(p0, T0, pr[1]);
                    for (int k = 1; k <= 19; ++k) {
                        const double al = 0.05 * static_cast<double>(k);   // 0.05 .. 0.95
                        const double Y = denner1d::mass_fraction_from_alpha(al, pa.rho, pb.rho);
                        const auto d = denner1d::alpha_derivs_massfrac(
                            al, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);

                        // (1) central FD in p at fixed (Y, T)
                        const double hp = 1.0e-6 * p0;
                        const double ap_fd =
                            (denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0 + hp, T0, pr[0]).rho,
                                 denner1d::phase_props(p0 + hp, T0, pr[1]).rho)
                           - denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0 - hp, T0, pr[0]).rho,
                                 denner1d::phase_props(p0 - hp, T0, pr[1]).rho)) / (2.0 * hp);
                        // central FD in T at fixed (Y, p)
                        const double hT = 1.0e-6 * T0;
                        const double aT_fd =
                            (denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0, T0 + hT, pr[0]).rho,
                                 denner1d::phase_props(p0, T0 + hT, pr[1]).rho)
                           - denner1d::alpha_from_mass_fraction(
                                 Y, denner1d::phase_props(p0, T0 - hT, pr[0]).rho,
                                 denner1d::phase_props(p0, T0 - hT, pr[1]).rho)) / (2.0 * hT);
                        // FD floor: central-difference roundoff is ~eps*alpha/h, so compare
                        // against max(|analytic|, that floor) rather than |analytic| alone.
                        const double fp = std::max(std::abs(d.a_p), macheps * al / hp);
                        const double fT = std::max(std::abs(d.a_T), macheps * al / hT);
                        worst_fd_p = std::max(worst_fd_p, std::abs(ap_fd - d.a_p) / fp);
                        worst_fd_T = std::max(worst_fd_T, std::abs(aT_fd - d.a_T) / fT);
                        check(std::abs(ap_fd - d.a_p) <= 1.0e-6 * fp, "a_p vs central FD");
                        check(std::abs(aT_fd - d.a_T) <= 1.0e-6 * fT, "a_T vs central FD");

                        // (2) the exact mixture-compressibility identity, p and T forms
                        const double rho = al * pa.rho + (1.0 - al) * pb.rho;
                        const double D_p = al * pa.zeta + (1.0 - al) * pb.zeta;
                        const double D_T = al * pa.phi  + (1.0 - al) * pb.phi;
                        const double lhs_p = D_p + (pa.rho - pb.rho) * d.a_p;
                        const double rhs_p = rho * (al * pa.zeta / pa.rho
                                                  + (1.0 - al) * pb.zeta / pb.rho);
                        const double lhs_T = D_T + (pa.rho - pb.rho) * d.a_T;
                        const double rhs_T = rho * (al * pa.phi / pa.rho
                                                  + (1.0 - al) * pb.phi / pb.rho);
                        worst_id = std::max(worst_id, std::abs(lhs_p - rhs_p) / std::abs(rhs_p));
                        worst_id = std::max(worst_id, std::abs(lhs_T - rhs_T) / std::abs(rhs_T));
                        check(std::abs(lhs_p - rhs_p) <= 1.0e-12 * std::abs(rhs_p),
                              "mixture-compressibility identity (zeta)");
                        check(std::abs(lhs_T - rhs_T) <= 1.0e-12 * std::abs(rhs_T),
                              "mixture-compressibility identity (phi)");

                        // (3) b==0 phase pair => a_T zero to the cancellation floor
                        if (both_b_zero) {
                            const double sc = al * (1.0 - al)
                                * std::max(std::abs(pa.phi / pa.rho), std::abs(pb.phi / pb.rho));
                            worst_bzero = std::max(worst_bzero, std::abs(d.a_T) / (macheps * sc));
                            check(std::abs(d.a_T) <= 8.0 * macheps * sc,
                                  "a_T == 0 (to cancellation floor) for b=0 phase pair");
                        }
                    }
                    // (4) endpoints are EXACT zeros (multiply by 0.0)
                    for (const double al : {0.0, 1.0}) {
                        const auto d0 = denner1d::alpha_derivs_massfrac(
                            al, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);
                        check(d0.a_p == 0.0 && d0.a_T == 0.0, "a_p==a_T==0 exactly at alpha in {0,1}");
                        check(denner1d::dalpha_dp_massfrac(al, pa.zeta, pa.rho, pb.zeta, pb.rho) == 0.0,
                              "dalpha_dp_massfrac exact 0 at endpoint");
                        check(denner1d::dalpha_dT_massfrac(al, pa.phi, pa.rho, pb.phi, pb.rho) == 0.0,
                              "dalpha_dT_massfrac exact 0 at endpoint");
                    }
                }
            }
        }
        std::cerr << "  Stage0 derivs: worst FD rel a_p=" << worst_fd_p
                  << " a_T=" << worst_fd_T << " ; worst identity rel=" << worst_id
                  << " ; worst |a_T|/(eps*scale) on b=0 pairs=" << worst_bzero << "\n";

        // --- Stage 0 deliverable 5: verify Phase-2 §1's numeric prediction at case15's state.
        // Printed, plus a LOOSE physical bound (the Wood-type mixture compressibility of a
        // bubbly liquid is orders of magnitude above the volume-blend value). Not a tuned
        // constant: the assertion is ">100x", the prediction is 521.56x.
        {
            const double p15 = 1.0e5;
            // case15 IC (cases.cpp:682-688): T = al*T(rho_air=1.3) + (1-al)*T(rho_water=1000)
            const double T15 = 348.2468430731;      // recompute if the IC ever changes
            const double al15 = 0.055;
            const auto pa = denner1d::phase_props(p15, T15, air);
            const auto pb = denner1d::phase_props(p15, T15, water);
            const auto d = denner1d::alpha_derivs_massfrac(
                al15, pa.zeta, pa.phi, pa.rho, pb.zeta, pb.phi, pb.rho);
            const double D_p = al15 * pa.zeta + (1.0 - al15) * pb.zeta;
            const double D_p_star = D_p + (pa.rho - pb.rho) * d.a_p;
            std::cerr << "  Stage0 case15 state: a_p=" << d.a_p << " a_T=" << d.a_T
                      << " D_p=" << D_p << " D_p*=" << D_p_star
                      << " ratio=" << D_p_star / D_p << " (Phase-2 §1 predicts ~500)\n";
            check(D_p_star / D_p > 100.0, "case15 continuity-diagonal defect exceeds 100x");
        }
    }
```

Implementer notes:
- Needs no new `#include` (`<cmath>`, `<iostream>` already at unit.cpp:6-7). `air` and `water` are already in scope from lines 22 and 27.
- `worst_bzero` is expected to print ~1.74; `worst_id` ~ 1e-16; the FD rels ~ 1e-9 to 1e-10.
- The `T15` literal: recompute it as `0.055*T_air + 0.945*T_water` from `temperature_for_density_pressure`-equivalent algebra if you prefer not to hardcode. `T_air = p/((gamma-1)kv*1.3) = 267.0013`, `T_water` solves `(p+pinf)/1000 = kv(gamma-1)T + b(p+pinf)` -> `352.9754`.
- `alpha` grid deliberately excludes the pure ends for the FD comparison (the relative FD error blows up there); the ends are covered by the exact-zero check instead.

---

## 4. Exact measurement procedure

### 4.1 Build (worktree-local, no `rm -rf`, no `scripts/yadv_r3_build.sh`)

```bash
W=/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-5/solver_4eq_mass
cmake -S "$W" -B "$W/build-cpp" -DCMAKE_BUILD_TYPE=Release
cmake --build "$W/build-cpp" -j8
"$W/build-cpp/cpp/denner_1d/denner1d_unit"        # expect: denner1d_unit ok
```
Do **not** add `-march=native` (`denner-pitfalls.md`: FMA breaks case01's `linf_p == 0`).

### 4.2 The four gate sweeps (record `pass_count` + the exact failure set for each)

```bash
V="$W/build-cpp/cpp/denner_1d/denner1d_validate"
cd "$W"
DENNER_ACID=1                                                       $V   # gate: 19/19
DENNER_ACID=1 ACID_YADV=1                                           $V   # gate: 15/19
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1                $V   # gate: 12/19, fails 13,14,15,24,25,33,34
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1 $V   # gate: 12/19, fails 14,15,24,27,28,33,34
```
`pass_count` is on the `DENNER1D_CPP_METRIC pass_count=<n> total=<m>` line (validation.cpp:847).
Byte-identity: run a `yadv_r5_verify.py` copy of `scripts/yadv_verify.py` with `MINE` re-pointed at `$W` -- gate **9/9 BYTE-IDENTICAL** vs `solver_denner`'s published binary.

**All four must be unchanged.** Stage 0 adds only header-inline functions with no call sites plus test code; if any sweep moves, something non-additive was edited.

(sections 4.3-6 elided from this saved copy for length -- see the round's transcript / final report
for the full ACID_AJAC_BLK reading guide, the Picard loop-gain measurement code, the literature
search results, and the 13-item failure-mode checklist. The Advisor session executing this round
has the full text and will follow it.)

---

## Advisor spot-check (done before implementation began)

Directly verified against the worktree's actual files (not trusted blindly):
- `scripts/yadv_r3_build.sh:4` does hardcode `W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass` -- confirmed, this script must NOT be used from the worktree.
- `eos.hpp:53-62` matches exactly; insertion point after line 62 confirmed correct.
- `acid.cpp:1503-1515` J1 loop structure matches the plan's description exactly (D, D_T, D_p, N,
  N_T, N_p, hsT, hsp, dTh/dTu/dTp, drh/dru/drp).
- `denner1d_unit.cpp` round-trip block closes at line 84; insertion before line 85 confirmed.
