# ROUND 30 PLAN — case15's stagnation-point core jet: mechanism measured, refinement question instrumented — **DIAGNOSTIC-ONLY**

**Target**: `solver_4eq_mass/`, `ACID_YADV=1` mass-fraction transport path, case15 (`15_E air-water cavitation`).
**HEAD**: `08d3c2e` ("ACID_YADV round 29 — case15 gap is a reference artifact; latch reverted (S5)").
**Control state on entry**: `round_counter = 29`, `consecutive_failures = 2` (loop stop condition is `>= 3`).
**Worktree integrity check (done)**: `diff -rq solver_4eq_mass/cpp .claude/worktrees/yadv-round-30/solver_4eq_mass/cpp` → identical, no output. Every code citation below was read live in the worktree at this HEAD; every number below was **measured this round** with the already-built `build-cpp/cpp/denner_1d/denner1d_dump` binary at that HEAD (no code was changed to produce any of them).

**Advisor verification note**: key structural claims checked directly against the code -- `acid.cpp:1786` (`pface[f] = use4 ? ... : 0.5*(pe[gL]+pe[gR]);`, central-mean form, exact match), `acid.cpp:1848` (`Rres[i][0] = trans_m + conv + pres;`, off-by-one from plan's cited `:1849`, immaterial), `denner1d_dump.cpp` (full 28-line file read directly; structurally matches the plan's citation with `const auto c = denner1d::find_case(id);` at line 10, matching exactly). No structural error found.

---

## 0. Executive summary

**This round is DIAGNOSTIC-ONLY. It proposes no fix, and it pre-registers (S-rule S6-e) that no fix may be added mid-round even if the diagnosis suggests one.** The default code footprint is **one ~20-line addition to `cpp/denner_1d/apps/denner1d_dump.cpp`** — a diagnostic cells-override, round 28's never-built Stage-4 idea — and **nothing else**. A second, optional instrument in `acid.cpp` is specified but is gated behind an explicit trigger and is expected NOT to fire.

I measured the core jet directly before designing anything, and the measurements overturn three separate pieces of the inherited framing:

1. **The core jet is a property of a fully CONVERGED discrete fixed point, not a solver stall.** Under config C, case15's Newton converges *quadratically with full steps* (`al=1.000`, no backtracking) at every sampled step (0,2,5,20,30,50,60,80,84), reaching `rnorm3 ≈ 3e-7` from `r_init ≈ 5e6` in 3–5 iterations; the run takes **85 steps, zero retries, zero `STALLED`/`STALL-ACCEPT`/`DIVERGED` lines on stderr**. `acid.cpp`'s own comment "case15 cavitation: NEVER converges" describes the **frozen-alpha path (plain B)**, not config C. So the jet cannot be attributed to non-convergence, and the accepted state's momentum balance is a real, exact statement about the scheme.

2. **The "overheating"/stagnation-point-heating framing that round 29 pre-registered for this round is measurably WRONG.** At the last step, the temperature across the entire core is **uniform to 0.02 K** (`T_o = 349.3479, 349.3622, 349.3496, 349.3652` K at cells 196/197/198/199, spanning a 340× pressure drop from 1133 Pa to 3.34 Pa). There is no thermal anomaly at the stagnation point of any kind. The three papers round 29 queued as the round-30 priority (Noh 1987 artificial heat flux; Petitpas et al. 2007 artificial heat; Bigdelou et al. 2021 overheating ghost-fluid) target a pathology that **does not occur here**, and the plan redirects the literature step accordingly (§5).

3. **The operative mechanism is already documented, with a named remedy, in a paper that is ALREADY IN THIS REPO** — `papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md` **§5 "Density discontinuities"** (lines 2019–2100), the JCP 375 (2018) MWI paper this solver's own Rhie–Chow term is built from. Its Eq. (90) states exactly what I measured: *"the discrete pressure gradient is underpredicted in the heavier phase and overpredicted in the lighter phase, which leads to an artificial acceleration of the flow in the vicinity of the interface. In the case of extremely large density ratios, the large and unphysical force applied to the lighter phase may lead to divergence of the solution algorithm."* case15's core has a **128:1 density ratio and a 137:1 pressure ratio across ONE cell face**, and the light cell is the one that gets the spurious acceleration.

**The mechanism, stated precisely** (§3): the momentum residual's pressure term is `pres_i = pface[i+1] − pface[i]` (`acid.cpp:1845`) with `pface[f] = 0.5*(p_L + p_R)` (`acid.cpp:1786-1788`, central — `use4` is provably false for case15, §2). So `pres_i = 0.5*(p_{i+1} − p_{i−1})`: **the cell's own pressure cancels out exactly**. In the first near-vacuum cell (i=198, `p = 3.243 Pa`, `rho = 0.5577`), the left face is assigned `pface[198] = 0.5*(444.96 + 3.243) = 224.10 Pa` — **69× the cell's own pressure** — because its heavy neighbour (i=197, `p = 444.96`, `rho = 71.13`) dominates the arithmetic mean across a nearly-discontinuous expansion front. The resulting pressure term `pres[198] = −214.62 Pa` and the convection term `conv[198] = +183.07 Pa` are each two orders of magnitude larger than their net `trans_m[198] = +31.55 Pa` (all three **measured**, §3.4, two independent routes agreeing to 0.1%). The accepted momentum balance in the near-vacuum cell is therefore a **near-cancellation of two O(200 Pa) terms conditioned entirely on an interpolated face pressure that is 69× the local pressure**. The MWI (Rhie–Chow) term, which is the scheme's only defence against exactly this, is `|mwi_p| ≈ 0.007 m/s` there against velocities of ~18 m/s — **~2700× too small to act, and NOT clamped** (§3.5), because `dhat ∝ dt` and case15 runs a *material* CFL (`dt = cfl·dx/max|u| = 1.125e-5` exactly, measured, §2).

**What round 30 does with that.** The one question that decides whether this is even worth attacking numerically — *does the artifact converge away under mesh refinement, or is it structural?* — cannot be answered from the N=800 reference, because `computed_reference` (`cases.cpp:423-437`) collapses each N=800 cell pair into one N=400 sample (I prove it is an *exact* pair average, §2.6). I therefore build the one instrument that settles it: an `ACID_DUMP_CELLS` override in `denner1d_dump.cpp` only — no `cases.cpp`, no `validation.cpp`, no `acid.cpp`, no solver arithmetic — with **two bit-level self-tests** that make it impossible for the instrument to be silently wrong (§4.2). Then a five-point refinement census (N = 100…3200) with **pre-registered, falsifiable predictions** (§4.3). Everything else in the round is zero-code measurement using existing instruments (`ACID_RCELL`, `ACID_TEND_SCALE`, `ACID_RHIST`, `ACID_DBG`, `ACID_MBAL`, `ACID_REGIME`).

**Why no fix.** The mechanism points at exactly one lever — the face pressure `pface` — and I derived and *numerically evaluated* the two parameter-free candidates before writing this (§3.7). The density-weighted (momentum-consistent) face pressure, which is Denner's own Eq. (93) weighting and reuses the harmonic `rho_f` the code already computes at `acid.cpp:1739`, gives `pface[198] = 6.68 Pa` instead of `224.10` (a 33× improvement, right direction) — **but on case25's reflected shock the same formula gives `1.82e6 Pa` where central gives `5.88e6 Pa`, a 3.2× under-prediction of the shock face pressure, i.e. a wrong shock speed.** That is verbatim the dead end `.claude/rules/denner-pitfalls.md` records ("Upwinding the face PRESSURE (pface) … is NOT valid — it breaks shock speed/position or diverges. Keep the conservative central pface."). The acoustic-impedance (Riemann-consistent) face pressure fails the same test (§3.7). **There is no bounded candidate on this evidence, and with `consecutive_failures = 2` a third harm would stop the autonomous loop entirely. Proposing one would be exactly the mistake rounds 27 and 29 made.** The correct hand-off is a measured mechanism plus a settled refinement verdict, which is what this round delivers.

---

## 1. Verified code facts (read live at HEAD `08d3c2e`, worktree == main tree)

All paths relative to `solver_4eq_mass/`.

| Fact | Location | Exact content (verified by reading) |
|---|---|---|
| Momentum residual, the three named terms | `cpp/denner_1d/src/acid.cpp:1842-1849` | `trans_m = (bdf_c0[i]*s.rho[i]*s.u[i] - Cold_mom[i]) * VdT;` `conv = mdotR[i]*uconv[i+1] - mdotL[i]*uconv[i];` **`pres = pface[i+1] - pface[i];`** `Rres[i][0] = trans_m + conv + pres;` |
| **Face pressure — central, no shock branch** | `acid.cpp:1786-1788` | `pface[f] = use4 ? (-pe[gL-1] + 7*pe[gL] + 7*pe[gR] - pe[gR+1])/12.0 : 0.5*(pe[gL] + pe[gR]);` |
| `use4` gate (proves `pface` is the plain mean for case15) | `acid.cpp:1754-1759` | `bool use4 = false; if (lowdiss && !face_shock) {...}` — `lowdiss` is **false** for case15 (§2), so `use4 ≡ false` at every face, every step |
| Harmonic face density (already present, used only in the MWI memory term) | `acid.cpp:1739` | `rho_f[f] = 2.0 / (1.0/re[gL] + 1.0/re[gR]);  // harmonic (ACID Eq.22, for MWI)` |
| `dhat` (ACID Eq.21, transient-dominated `a_P`) | `acid.cpp:1767-1769` | `aP = 0.5*(re[gL]+re[gR])*dx/dt_mwi; d_f = dx/max(aP,1e-300); dhat[f] = d_f/(1.0 + (rho_f[f]/dt_mwi)*d_f);` ⇒ `dhat ≈ dt/(2·rho̅)` |
| MWI correction + its sound-speed clamp | `acid.cpp:1777-1785` | `af = 0.5*(ae[gL]+ae[gR]); mwi_p = -dhat[f]*(dpf - gpbar); mwiOK_f[f] = (abs(mwi_p) < af); af_f[f] = af; dpgpf[f] = dpf - gpbar; mwi_p = clamp(mwi_p, -af, af); theta[f] = ubar + mwi_p + (rho_f[f]/dt_mwi)*dhat[f]*(theta_o[f]-ubar_o);` |
| `gpbar` — **arithmetic**, not density-weighted (the gap vs the MWI paper's Eq. 91) | `acid.cpp:1709`, `:1771` | `cell_gradp(gi) = (pe[gi+1]-pe[gi-1])/(2*dx);` `gpbar = 0.5*(cell_gradp(gL) + cell_gradp(gR));` |
| ACID Eqs.41-42 per-cell face mass flux | `acid.cpp:1832-1836` | `mdotR[i] = (al*raup[i+1] + (1-al)*rbup[i+1])*theta[i+1];` (`al = clamp(s.alpha[i],0,1)`) |
| Upwind cell selection at a face | `acid.cpp:1789-1791` | `const bool fromL = theta[f] >= 0.0; const int gU = fromL ? gL : gR;` |
| Mixture sound speed `s.a` (this is **not** the Wood speed) | `acid.cpp:299`, `:301-327` | `a = sqrt((gamma_mix-1)*cp*T)`, `1/(gamma_mix-1) = alpha/(ga-1)+(1-alpha)/(gb-1)` (Denner Eqs.57-58) |
| CFL selector — **material** CFL for case15 | `acid.cpp:444`, `:452-453`, `:953-957`, `:965` | `auto_material = !acoustic_src && p_ratio < 1.01 && umax0 > 0.0;` `penta_solve = ... \|\| (coupled && (auto_material \|\| unic));` `mat_dt = auto_material && (!coupled \|\| penta_solve);` `li = abs(s.u[i]) + (mat_dt ? 0.0 : s.a[i]); dt_full = cfl*dx/lam;` |
| `ACID_RCELL` window dump (reused, zero-code, gated on `yadv`) | `acid.cpp:679-690`, `:1475-1495` | prints `Y0 Y al0 al p_o T_o u_o h Htot_o rho rho_o` per cell per retry, filtered by `ACID_BLK_STEP` |
| `ACID_MBAL` end-of-step block — **the exact template for any new end-of-step instrument** | `acid.cpp:2795-2831` | `const Field s_mbal_backup = s; compute_R(); … ; s = s_mbal_backup;` inside the retry body, after the TR-BDF2 stage loop |
| All face/flux scratch arrays live in the same scope as the MBAL site | `acid.cpp:1589-1599` | `Vec theta, rho_f, dhat, pface, uconv (n+1); Vec raup, rbup, rHaup, rHbup (n+1), mdotL, mdotR (n); Vec af_f(n+1,0.0), dpgpf(n+1,0.0); vector<char> use4_f, mwiOK_f; vector<int> uwc_f; vector<Vec2> Rres(n);` |
| `Yv` is outer-scope and always sized `n` (safe to read on the OFF path) | `acid.cpp:915` | `Vec Yv(n, 0.0);` |
| `ACID_TEND_SCALE` (existing diagnostic sweep knob, the precedent for a dump-only override) | `acid.cpp:890-899`, `:931-939` | prints an explicit "`*_ref` columns and all validate metrics are INVALID for this run" warning |
| **`denner1d_dump` — the whole file, 28 lines** | `cpp/denner_1d/apps/denner1d_dump.cpp:1-28` | `const auto c = find_case(id); const auto s = solve_case(c); const auto r = reference_state(c);` then a 12-sig-digit CSV `x,alpha,p,u,rho,p_ref,u_ref,rho_ref` |
| case15 reference = the SAME solver, same env, N=800 | `cpp/denner_1d/src/cases.cpp:754-756` | `if (c.id == "15") return computed_reference(c, 800);` |
| `computed_reference` — linear interpolation onto the coarse grid | `cases.cpp:423-437` | `hi.config.cells = std::max(c.config.cells, cells); const PrimitiveState fine = solve_case(hi); … out.u[i] = interp(fine.x, fine.u, out.x[i]); …` |
| `interp` is plain linear, `resize_state` puts centres at `x0+(i+0.5)dx` | `cases.cpp:228-236`, `:187-197` | ⇒ N=400 centres fall **exactly midway** between N=800 centres `2i` and `2i+1` ⇒ the reference is an **exact pair average** (§2.6) |
| case15 config / IC | `cases.cpp:493`, `:682-688`, `:21` | `base_config(400, 9.5e-4, 0.0, 1.0)`; `alpha=0.055`, `u=∓100`, `p=1e5`, `T` = alpha-blend; `cfl=0.45`; both BCs `transmissive` |
| case15 gate, all 8 criteria | `cpp/denner_1d/src/validation.cpp:684-729` | `corr_p≥0.93, corr_u≥0.998, corr_rho≥0.99, l2_p≤0.18, l2_u≤0.06, l2_rho≤0.05, smooth_ok, osc_ok`; `smooth_ok = cj ≤ max(8,1.1·cj_r) && mj ≤ max(8,1.1·mj_r) && cc ≤ max(0.04,1.1·cc_r)` |
| The smoothness criteria are spec-mandated | `validation/1D/15_E_Cavitation.md:190-193` | "…one-cell step-like velocity fan을 FAIL 처리한다" |
| `ap_advection` / `dhat_scale` / `ACID_DHK` **no longer exist** | `cpp/denner_1d/include/denner1d/types.hpp:66-69` | "the former ap_advection / dhat_scale knobs were DELETED" — confirmed by grep: **zero** hits for `ACID_DHK` anywhere in `cpp/`. `.claude/rules/denner-pitfalls.md`'s claim that `ACID_DHK` "remains only as a research env knob" is **stale**; there is no dhat lever to sweep. |
| No grid-override env exists anywhere | grep over `cpp/` | `denner1d_validate` takes only `--only`/`--out`; `denner1d_run`/`denner1d_dump` take only a case id |

---

## 2. Baseline facts measured this round (zero code changed)

Command prefix throughout: `cd solver_4eq_mass && DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ./build-cpp/cpp/denner_1d/denner1d_dump 15`.

### 2.1 Config C's regime, exactly

`ACID_REGIME=1` → `REGIME 15 p_ratio=1 -> energy=coupled(3x3) recon=1st-upwind time=BE dt=acoustic-CFL`.

So `use_minmod = false`, `lowdiss = false`, `bdf2 = false` ⇒ `tr_bdf2 = false`, `bdf_c0 ≡ 1`, `use4 ≡ false`, `rec()` never called, `pU = pe[gU]` plain 1st-order upwind, **`pface ≡ 0.5*(p_L+p_R)` at every face**.

**Correction to the `REGIME` label**: it prints `acoustic-CFL` whenever `coupled`, but `mat_dt = auto_material && (!coupled || penta_solve)` and case15 has `auto_material = true` (no acoustic source, `p_ratio = 1 < 1.01`, `umax0 = 100 > 0`) and `penta_solve = true` (`coupled && unic`), so **`mat_dt = true` and the sound speed is excluded from `lam`**. Confirmed numerically: `ACID_DBG`/`ACID_MBAL` give `dt = 1.125000e-05` **constant for all 85 steps**, and `0.45 × 0.0025 / 1.125e-5 = 100.0` exactly `= max|u|`. This matters: `dhat ∝ dt` is set by the far-field `|u| = 100`, not by anything local to the core.

### 2.2 The Newton converges — the jet is a fixed-point property

`ACID_RHIST=1 ACID_BLK_STEP=<k>`:

| step | iteration history (`n0 → rnorm3`, `al`) |
|---|---|
| 2 | `2.80e7→4.19e6`, `4.19e6→1.22e5`, `1.23e5→1.03e2`, `8.64e1→7.75e-5`, `3.21e-2→1.72e-7` — all `al=1.000` |
| 5 | `9.22e6→4.96e5 … 5.35e-4→1.36e-7`, 5 its, all `al=1.000` |
| 20 | `9.80e6→1.19e6 … 4.41e0→4.36e-6`, 4 its, all `al=1.000` |
| 30 | `8.39e6→4.62e5 … 2.36e-1→3.54e-7`, 4 its, all `al=1.000` |
| 50 | `5.81e6→2.02e5 … 2.67e-2→2.42e-7`, 4 its, all `al=1.000` |
| 60 | `5.36e6→1.56e5 … 1.50e-2→2.81e-7`, 4 its, all `al=1.000` |
| 80 | `4.82e6→1.06e5 … 5.74e-3→3.14e-7`, 4 its, all `al=1.000` |
| 84 | `4.70e6→4.29e4 … 3.99e-4→3.64e-7`, 4 its, all `al=1.000` |

Textbook quadratic convergence, **no line-search backtracking at any sampled step**, 12–13 orders of residual reduction. Total stderr from a full config-C case15 run: **empty** (`grep -c "STALL\|diverg\|retry"` → 0). 85 steps, zero retries.

### 2.3 Config C's case15 gate, all 8 criteria (recomputed from the dump, reimplementing `validation.cpp:684-729` and `:18-51,317-359`)

| criterion | limit | config C | verdict |
|---|---|---|---|
| `corr_p` | ≥0.93 | 0.999285 | PASS |
| `corr_u` | ≥0.998 | **0.999344** | PASS |
| `corr_rho` | ≥0.99 | 0.996734 | PASS |
| `l2_p` | ≤0.18 | 0.014393 | PASS |
| `l2_u` | ≤0.06 | **0.017038** | PASS |
| `l2_rho` | ≤0.05 | 0.019664 | PASS |
| `p_osc` / `r_osc` (`osc_ok`) | <0.02 / <0.04 | 0.000000 / 0.000000 | PASS |
| `cj` | ≤ max(8, 1.1·3.5455) = **8.0000** | **30.0178** | **FAIL** |
| `mj` | ≤ max(8, 1.1·18.0773) = **19.8850** | **31.9982** (at i=198) | **FAIL** |
| `cc` | ≤ max(0.04, 1.1·0.08439) = **0.09283** | **0.11746** | **FAIL** |

**Config C fails case15 on `smooth_ok` and nothing else — all three of its sub-criteria, and everything else passes with 3–12× margin.** This confirms and sharpens §39.5: the core jet is the single blocker, and it is a *velocity-smoothness* blocker only.

### 2.4 The core profile and the momentum bookkeeping (final state)

Reproduces round 27 §4.5 / §39.5 digit-for-digit, extended with the exactly-computable pressure term `pres_i = 0.5*(p_{i+1} − p_{i−1})` and the specific force:

| i | x | alpha | p (Pa) | u (m/s) | rho | `pface[i]` | `pres_i` (Pa) | `pres_i/(dx·rho)` (m/s²) |
|---|---|---|---|---|---|---|---|---|
| 195 | 0.48875 | 0.800593 | 1452.6 | −21.653 | 200.1 | 1604.6 | −316.9 | −633.4 |
| 196 | 0.49125 | 0.838563 | 1122.8 | −17.781 | 162.00 | 1287.7 | −503.8 | −1243.9 |
| 197 | 0.49375 | 0.929115 | 444.96 | −13.844 | 71.133 | 783.9 | −559.8 | −3147.7 |
| **198** | 0.49625 | 0.999444 | **3.2432** | **+18.154** | **0.55774** | **224.10** | **−214.62** | **+153 920** |
| **199** | 0.49875 | 0.997313 | **15.716** | **+15.009** | **2.6968** | 9.4796 | **+6.2364** | −925 |
| **200** | 0.50125 | 0.997313 | 15.716 | −15.009 | 2.6968 | 15.716 | **−6.2364** | +925 |
| **201** | 0.50375 | 0.999444 | 3.2432 | −18.154 | 0.55774 | 9.4796 | +214.62 | −153 920 |
| 202 | 0.50625 | 0.929115 | 444.96 | +13.844 | 71.133 | 224.10 | +559.8 | +3147.7 |

The specific pressure force in cell 198 is **49× that in cell 197 and 124× that in cell 196** — a pure `1/rho` amplification of an interpolation error, on a face whose two sides differ by **128× in density** (71.13 vs 0.5577) and **137× in pressure** (444.96 vs 3.2432).

`pface[198] = 224.10 Pa` is **69.1× cell 198's own pressure.**

### 2.5 The onset, and what it is NOT

`ACID_TEND_SCALE` sweep (solver columns only — the `*_ref` columns and all validate metrics are invalid under this knob by the code's own printed warning, and are never used below):

| `TEND_SCALE` | `u[197]` | `u[198]` | `u[199]` | reversed cells (x<0.5) | **min p over the whole domain** |
|---|---|---|---|---|---|
| 0.24 | −67.868 | −54.784 | −29.529 | 0 | 10.13 |
| 0.28 | −56.645 | −41.625 | −6.247 | 0 | 6.599 |
| **0.32** | −47.485 | −33.281 | **+11.782** | **1** | 5.307 |
| 0.36 | −40.770 | −29.595 | +23.963 | 1 | 4.899 |
| 0.50 | −31.200 | −28.284 | +33.261 | 1 | 5.320 |
| 0.80 | −20.471 | −11.028 | +21.914 | 1 | 7.120 |
| 1.00 | −13.844 | **+18.154** | +15.009 | **2** | 3.243 |

* Onset is between `t = 2.66e-4` and `3.04e-4` (≈ step 24–27 of 85), starting as **exactly one cell per side** — the cell adjacent to the symmetry face — and reaching two only in the last few steps.
* **The 1.0 Pa pressure floor is never approached under config C** (global minimum `3.24–10.1 Pa` at all times). The floor is *not* implicated. (It *is* what plain B and OFF sit on — measured below.)
* The velocity field is not an odd–even checkerboard: going outward from the centre the profile is monotone except for **one** inversion (`p[200]=15.72 > p[201]=3.24`).

### 2.6 The N=800 reference is an *exact pair average*, and it is already non-monotone

`out.x[i] = (i+0.5)·(1/400) = (2i+1)·(1/800)`, which is exactly midway between `fine.x[2i] = (2i+0.5)/800` and `fine.x[2i+1] = (2i+1.5)/800`. With `interp` linear (`cases.cpp:228-236`) the reference is therefore **`ref[i] = ½(fine[2i] + fine[2i+1])`** to round-off. A pair average **cannot create non-monotonicity from a monotone field**. And the reference *is* non-monotone:

`p_ref[193..199] = 1567.7, 1314.8, 1102.2, 1027.6, **1141.6**, 591.4, 44.3` — a one-sample local **maximum** at i=197.
`u_ref[195..199] = −17.270, −17.454, **−20.826**, −19.850, −1.773` — a one-sample local **minimum** at i=197.

⇒ **the raw N=800 field is itself non-monotone in both `p` and `u` at the same location.** §39.8's "genuine under-resolution artifact (absent at N=800)" is therefore too strong: a *weaker* version of the same structure survives one refinement. Whether it still contains a **sign reversal** cannot be decided from the averages — by antisymmetry the only constraint available is `u₈₀₀[398] + u₈₀₀[399] = 2·u_ref[199] = −3.5455`, which is consistent both with "no reversal" (e.g. −2.0, −1.5) and with a weak one (e.g. −8.5, +5.0). **This is precisely why Stage 1's cells-override is needed.**

One decisive hint in favour of convergence: over the identical physical window `x ∈ [0.4950, 0.5000]`, N=400's two cells average **+16.58 m/s (inward)** while N=800's four cells average **−10.81 m/s (outward)** — opposite signs.

### 2.7 Configs A and B are degenerate at the core (control)

| config | `p[195..202]` | `rho` | `alpha` |
|---|---|---|---|
| A (OFF) | `1, 1, 1, 1, 1, 1, 1, 1` (at the floor) | 1022 (frozen) | 0.055 (frozen) |
| B (plain `ACID_YADV=1`) | `1, 1, 1, 1, 1, 1, 1, 1` | 0.2007→0.2035 | 0.99981 |
| C | 1452.6 … 444.96, 3.24, 15.72 … | 200.1 … 0.5577 | 0.80→0.999 |

Config A's own `u` at the core is `−8.94, −5.09, −6.08, −0.78, −3.02` — non-monotone too, on a completely floored field. **Config C is the only configuration producing a physically meaningful core at all**, which is why it is the only one on which this diagnosis is meaningful. (Recorded, not acted on: this reinforces §39.5's already-flagged finding about config A's degenerate case15 "pass". No `validation.cpp`/`cases.cpp` change is proposed here either.)

---

## 3. The diagnostic work — mechanism traced to its immediate cause

### 3.1 The discretisation, stated exactly

With `use4 ≡ false` (§2.1), the momentum pressure term at cell `i` is

```
pres_i = pface[i+1] − pface[i] = ½(p_{i}+p_{i+1}) − ½(p_{i−1}+p_{i}) = ½(p_{i+1} − p_{i−1})
```

**The cell's own pressure cancels identically.** This is the standard collocated 2Δx pressure gradient; it is second-order accurate for smooth `p` and is *supposed* to be protected against the associated decoupling by the MWI term in the advecting velocity.

### 3.2 Why that is catastrophic in the first near-vacuum cell

At the vacuum front the profile is nearly a one-cell discontinuity: `p₁₉₇ = 444.96`, `p₁₉₈ = 3.2432` (137:1) with `rho₁₉₇ = 71.13`, `rho₁₉₈ = 0.5577` (128:1). Two independent errors compound:

1. **Interpolation error.** The exact face pressure at `x = 0.4950` between a 445 Pa cell and a 3.2 Pa cell in a strong expansion is close to the *low* side (the star pressure of that local expansion), not the arithmetic mean. The scheme uses `224.10 Pa` — 69× cell 198's own pressure.
2. **`1/rho` amplification.** The same absolute face-pressure error produces `1/rho` times more acceleration. Cell 198's specific pressure force is `+153 920 m/s²` versus `+3 148` in its neighbour — a factor **49**, exactly the density ratio.

This is Bartholomew/Denner/van Wachem §5, Eq. (90) (`papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md:2065`), verbatim: *"the discrete pressure gradient is underpredicted in the heavier phase and overpredicted in the lighter phase, which leads to an artificial acceleration of the flow in the vicinity of the interface."*

### 3.3 Why the same thing hits the innermost cell (the one-cell onset, §2.5)

By symmetry `pface[200] = ½(p₁₉₉+p₂₀₀) = p₁₉₉` exactly, so

```
pres_199 = p₁₉₉ − ½(p₁₉₈ + p₁₉₉) = ½(p₁₉₉ − p₁₉₈)
```

i.e. **the innermost cell's net pressure force is exactly `(p₁₉₈ − p₁₉₉)/(2dx)`, driven entirely by its *outer* neighbour** — pointing inward whenever the outer cell is at higher pressure, which is the whole of the run until the core has fully evacuated. At `TEND_SCALE = 0.2` (before the flip) the neighbour is at `12 870 Pa` and the cell at `19.96 Pa` with `rho = 3.426`: specific force `= (12870−19.96)/(2·0.0025·3.426) = +750 000 m/s²`. Over a `1.125e-5` step that is `+8.4 m/s` of inward velocity per step against a field of ~50 m/s. That is the onset, and it explains why it appears first at exactly `i = N/2 − 1` and only later spreads outward.

### 3.4 The full accepted momentum balance, measured two independent ways

`ACID_RCELL=196:203 ACID_BLK_STEP=84` gives the *exact* old-level quantities the transient uses (`Cold_mom[i] = rho_o[i]·u_o[i]`, `bdf_c0 ≡ 1`, `dt = 5.000000e-06` for the final partial step, `VdT = dx/dt = 500`):

| i | `rho_o` | `u_o` | `rho·u` (new) | `rho_o·u_o` | **`trans_m`** | **`pres`** | **`conv` = −trans−pres** |
|---|---|---|---|---|---|---|---|
| 197 | 73.68908 | −13.85924 | −984.77 | −1021.28 | **+18 255** | −559.76 | **−17 695** |
| **198** | 0.5736214 | +17.54123 | 10.1252 | 10.0621 | **+31.55** | **−214.62** | **+183.07** |
| **199** | 2.608308 | +14.90692 | 40.4728 | 38.8798 | **+796.5** | **+6.236** | **−802.7** |

Cross-check by an entirely different route — evaluating `conv = mdotR·uconv[i+1] − mdotL·uconv[i]` directly from `acid.cpp:1832-1836` with `theta ≈ ubar` (the MWI term is negligible, §3.5), 1st-order upwind `uconv`, and the phase-blend face density (`≈ (1−alpha_i)·rho_water(p_up,T_up) ≈ 0.5538` at both faces of cell 198) gives `conv[198] = 9.182×18.154 + 1.1934×13.844 = +183.2` and `conv[199] = −44.37×18.154 = −805.5`. **Both agree with the residual-closure values to 0.1 % and 0.3 %** — the attribution is solid without any new instrument.

**The headline reading**: in cell 198 the pressure and convection terms are `−214.62` and `+183.07` Pa, each **~6.6× the net** `+31.55` Pa. The accepted balance is a near-cancellation of two large terms, and the pressure term is set by a face pressure that is 69× the local pressure. A 15 % error in `pface[198]` would change the net acceleration by 100 %. **The near-vacuum cell's momentum is conditioned entirely on an interpolation the scheme has no accuracy claim for at this pressure/density ratio.**

Note this is a *near-cancellation*, not "pressure dominates the net" — stated that way deliberately, because the honest form of the statement is the one that survives scrutiny.

### 3.5 The MWI is not the actor here — and specifically NOT case25's mechanism, NOT a clamp saturation

At the critical face `f = 199` (between cells 198 and 199), final state, `dt_mwi = dt = 1.125e-5` (`tr_bdf2 = false`):

```
d_f            = dt/(½(rho₁₉₇+rho₁₉₈))          = 1.125e-5/35.84   = 3.139e-7
rho_f          = 2/(1/71.13 + 1/0.5577)                            = 1.1067
(rho_f/dt)·d_f = 0.03088
dhat[199]      = 3.139e-7/1.03088                                  = 3.045e-7
dpf            = (p₁₉₈−p₁₉₇)/dx                                    = −176 687
gpbar          = ½[(p₁₉₈−p₁₉₆)/2dx + (p₁₉₉−p₁₉₇)/2dx]              = −154 880
mwi_p          = −dhat·(dpf−gpbar)                                 = **+0.00664 m/s**
```

against local velocities of ~18 m/s: **the Rhie–Chow correction is ~2700× too small to influence anything.** And it is **not clamped**: `af = ½(a₁₉₇+a₁₉₈)` where `s.a` is Denner Eqs.57-58's mixture speed (`acid.cpp:299`) — with `T ≈ 349 K` and a water-dominated `cp`, `af = O(10²–10³) m/s`, five orders above `|mwi_p|`. So `mwiOK_f = 1` and the clamp branch never fires here.

This settles the distinction the briefing asked me to re-verify rather than assume:

* **case25's documented defect** (`.claude/rules/denner-pitfalls.md`, "MWI pressure dissipation scales with dt … SMALL time steps UNDER-damp") is a *pressure–velocity decoupling oscillation* that the MWI is supposed to damp and doesn't damp *enough*.
* **case15's core jet** is a *one-sided, non-oscillatory, sign-consistent over-acceleration* driven by the momentum equation's own face-pressure interpolation at a 128:1 density ratio. The MWI is a bystander at 0.4 % of even the *net* acceleration.

They share a family resemblance only in that both live in a collocated pressure-based discretisation. **Round 27's §4.5 conclusion ("a different mechanism from the pitfalls file's `dhat ~ dt` entry") is CONFIRMED — by direct measurement rather than by assertion — but its stated reason ("an under-resolved near-vacuum core in a mixture whose Wood sound speed has collapsed to a few m/s, M≈40") is not the operative one: the code's `s.a` is not the Wood speed, the clamp is inactive by five orders, and `dt` is set by the far-field material CFL, not by any local acoustic scale.**

### 3.6 The "overheating" framing is refuted

`T_o` across the entire core at step 84: `349.3479, 349.3622, 349.3496, 349.3652 K` at i = 196, 197, 198, 199 — **spread 0.02 K over a 340× pressure drop**. There is no thermal anomaly at the stagnation point. Noh-type wall heating, Petitpas-type artificial-heat corrections, and Bigdelou-type overheating ghost-fluid treatments all address a defect that is measurably absent here. The three `*_needed.md` stubs round 29 created (`1987_Noh_…`, `2007_Petitpas_…`, `2021_Bigdelou_…`) should be **annotated with this refutation, not chased** (§5).

### 3.7 The two parameter-free fix candidates, derived and killed *before* proposing anything

**(F-a) Density-weighted / momentum-consistent face pressure.** Denner's own Eq. (93)/(91) weighting (`…MWI_collocated_main.md:2090-2130`), reusing the harmonic `rho_f` the code already computes at `acid.cpp:1739`, would give `pface = rho_f · ½(p_P/rho_P + p_F/rho_F)`:

| face | central `pface` | density-weighted `pface` | comment |
|---|---|---|---|
| case15 i=197\|198 (`p` 445/3.24, `rho` 71.1/0.558) | 224.10 | **6.68** | 33× better; qualitatively right |
| case01 (uniform `p`, any `rho` jump) | `p` | `p` **exactly** | pressure equilibrium preserved |
| case25 reflected shock (`p` 1e5/1.165e7, `rho` 1.157/6.614) | 5.88e6 | **1.82e6** | **3.2× under-prediction ⇒ wrong shock speed** |

The last row is fatal and is *exactly* the dead end already on record: "*Upwinding the face PRESSURE (pface) or the advecting velocity (ubar) at a shock is NOT valid — pressure is not advected; it breaks shock speed/position or diverges. Keep the conservative central pface.*" Blast radius: **every** shock case (13/14/24/25/26/27/28/29/32/33/34) plus the OFF path, which must stay byte-identical to `solver_denner`.

**(F-b) Acoustic-impedance (Riemann-consistent) face pressure**, `p_f = (Z_R p_L + Z_L p_R)/(Z_L+Z_R) − (Z_L Z_R/(Z_L+Z_R))(u_R−u_L)`: at case15's face 198 it evaluates **negative** (`≈ −44 Pa`, needing a floor); at case25's shock it gives `≈5.2e5` against central's `5.88e6` — an 11× error. Also dead.

Note that Denner's own §5.1 remedy is applied **only to the MWI's advecting velocity**, not to the momentum equation's pressure term — and §3.5 shows the MWI term here is 2700× too small for that remedy to matter (recomputing `gpbar` with density weighting changes `mwi_p` from `+0.0066` to `+0.0273 m/s`, still negligible). **The MWI density weighting is a genuine faithfulness gap versus the source paper, but it is measurably NOT the core-jet fix; it is recorded as such, and not proposed.**

**Conclusion: no bounded, parameter-free candidate exists on this evidence. Round 30 proposes none.**

---

## 4. Stages

Execution is strictly sequential. **After every stage that touches code, the full hard-gate battery (§6) is run and READ before any case15-specific number is looked at** — the three-round-running discipline that caught both round 27's and round 29's harm, retained unchanged even though this round proposes no fix.

### Stage 0 — baseline (no code)

1. Clean rebuild.
2. Run G1/G2/G3 (§6) and record the baseline numbers.
3. Reproduce §2.1–2.7 exactly. **Every table in §2 must reproduce digit-for-digit.** Any mismatch → S5-adjacent: stop and report a build/environment discrepancy before touching anything.

### Stage 1 — `ACID_DUMP_CELLS` (the ONLY default code change; `denner1d_dump.cpp` only)

**Exact change spec.** In `cpp/denner_1d/apps/denner1d_dump.cpp`, change `const auto c = ...` to `auto c = ...` and insert immediately after it the `ACID_DUMP_CELLS` override (see §4.2 body — code already applied this round).

**Self-test A (bit-level, mandatory).** `ACID_DUMP_CELLS=800` on case15, config C: because `computed_reference(c,800)` then sets `hi.config.cells = max(800,800) = 800`, the reference solve is bit-identical to the primary solve, and `interp` at exactly-coincident grid points returns `y[j]` exactly. **Therefore `p == p_ref` and `u == u_ref` bit-for-bit in every one of the 800 rows.** A single mismatched row falsifies the instrument.

**Self-test B (cross-validation against the untouched path, mandatory).** Pair-average the `ACID_DUMP_CELLS=800` run's `u`, `p`, `alpha` columns (`½(row[2i]+row[2i+1])`) and compare against the **unset** `denner1d_dump 15` run's `u_ref`, `p_ref`, `alpha_ref` columns. These must agree to **≥ 10 significant digits** (the residual is `interp`'s weight being `0.5 ± 1 ulp`, §2.6). Specific pre-registered values that must come out of this: `u₈₀₀[398]+u₈₀₀[399] = −3.54553723`, `u₈₀₀[396]+u₈₀₀[397] = −39.70007298`, `p₈₀₀[398]+p₈₀₀[399] = 88.6492379648`, `p₈₀₀[396]+p₈₀₀[397] = 1182.8483677`.

Self-test B is the strong one: it validates the new instrument against a code path that this round does not modify, and it simultaneously proves §2.6's exact-pair-average claim.

**Gates**: G1, G2, G3, G4(a), G5 — run and read *before* Stage 2.

### Stage 2 — refinement census (no further code)

For `N ∈ {100, 200, 400, 800, 1600, 3200}`, config C (`DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_DUMP_CELLS=N`), record:

| quantity | definition |
|---|---|
| `cj_N` | `\|u[N/2] − u[N/2−1]\|` = `2·\|u[N/2−1]\|` by antisymmetry |
| `n_rev` | count of cells with `x < 0.5` and `u > 0` |
| `w_rev` | `n_rev · dx` — the **physical** width of the reversed region |
| `n_vac`, `w_vac` | cells with `p < 100 Pa` on `x < 0.5`, and their physical width |
| `p_min`, `rho_min`, and their `x` | global minima |
| core profile | `p, u, rho, alpha` for the 8 cells straddling `x = 0.5` |
| `mj_N`, `cc_N` | the other two `smooth_ok` components, on that N's own grid |
| wall time, step count, `dt` | from `ACID_DBG` (predicted: `dt = 0.45·dx/100`, steps `≈ 9.5e-4/dt`) |

**Pre-registered, falsifiable hypotheses** (written before running):

* **H-R1 (converging artifact — FAVOURED).** `n_rev` stays at 0–2 independent of `N`, so `w_rev ∝ dx → 0`; `cj_N` decreases monotonically with `N`; specifically **`cj_800 < 8`**. Supporting prior evidence: over the fixed physical window `[0.4950, 0.5000]` the N=400 mean velocity is `+16.58` (inward) but the N=800 mean is `−10.81` (outward) — §2.6.
* **H-R2 (structural defect).** `n_rev` grows roughly `∝ N` (fixed physical width), and/or `cj_N` stays `≳ 30` or grows. Then the scheme has a genuine, non-converging near-vacuum stagnation-point defect.
* **H-R3 (mixed).** `w_rev` converges to a *fixed physical* width while `cj_N` plateaus above 8. Report as such; do not force into H-R1 or H-R2.

**Anti-rescue, binding**: no `cj_N`, `mj_N`, `cc_N` or any other number from a run with `ACID_DUMP_CELLS` set may be reported as a case15 gate result. case15's gate is, and remains, N=400 scored against `computed_reference(c,800)`.

### Stage 3 — momentum attribution at the onset step (no code)

Repeat §3.4's `ACID_RCELL` + dump + `ACID_DBG` procedure at three steps: the pre-onset step, the onset step, and the final step 84. Report `trans_m`, `pres`, `conv` for cells 195-204, plus `pface[i]/p_i` and the specific force. Repeat the same three-step table at N=800 via Stage 1's override.

### Stage 4 — `ACID_FMOM` face/cell momentum census — CONDITIONAL, expected NOT to run

Trigger: Stage 3's residual-closure `conv` disagrees with the direct estimate by >5%, or the accepted-state residual `|R|` exceeds `1e-4`. Given §3.4's two independent routes already agreeing to 0.1-0.3%, this is not expected to fire. If it does, full spec in the Planner's own transcript (not reproduced here for brevity; implementing session must re-derive from first principles per this plan's own discipline if the trigger fires, using the MBAL block as the exact template).

---

## 5. Gates and targets

### Hard gates

* **G1 — OFF-path byte identity.** 9/9 byte-identical vs published `solver_denner`; case01 `ACID_YADV=1` ≡ unset.
* **G2 — 7-config sweep.** All configs A-G match `EXPECTED`.
* **G3 — unit tests.** Clean, unchanged.
* **G4 — new-instrument no-op.** `ACID_DUMP_CELLS` unset ⇒ byte-identical `denner1d_dump`/`denner1d_validate` output on cases 01,02,13,14,15,24,25 under OFF/B/C. Self-tests A and B pass.
* **G5 — diff hygiene.** `git diff --stat -- cpp/` shows only `apps/denner1d_dump.cpp` (plus `src/acid.cpp` only if Stage 4 fires). No `cases.cpp`/`validation.cpp`/`types.hpp` change. No new numeric literal on any solution path. Floor untouched.

### Execution order

`clean rebuild -> G1/G2/G3 baseline -> Stage 0 (T1) -> Stage 1 code -> G1/G2/G3/G4(a)/G5 HARM GATE, read in full before any case15 number -> G4(c) self-tests -> Stage 2 census -> Stage 3 attribution -> Stage 4 only if triggered -> literature + writeup -> re-run all gates before commit`.

---

## 6. Pre-registered stop / decision rules

**S1 — diagnostic success (expected).** All hard gates hold; refinement question decided (H-R1/H-R2/H-R3). Merge Stage 1. Write §40 with corrections: (a) config C converges quadratically -- not a stall; (b) T uniform to 0.02K -- overheating refuted; (c) MWI clamp inactive by 5 orders, `mwi_p` 2700x too small; (d) `ACID_DHK` stale in denner-pitfalls.md; (e) §39.8's "absent at N=800" too strong. `consecutive_failures -> 0`.

**S2 — partial (Stage 4 fails its own gate).** Revert Stage 4 in full. Stages 1-3 merge. `consecutive_failures` NOT incremented.

**S3 — structural finding, user escalation (H-R2/H-R3).** Do not attempt a fix. Escalate: (i) accept case15 unreachable under ACID_YADV=1; (ii) authorise a scheme-level pface change with explicit shock-case risk; (iii) spec conversation about case15's mesh. `consecutive_failures` NOT incremented.

**S4 — inconclusive.** Report exactly why. NOT incremented only if a named, specific reason exists.

**S5 — harm.** Any hard gate fails, or any previously-finite case becomes NaN (independent of pass/fail status), or G5 shows a forbidden file touched. Revert the offending stage in full, do not merge it, `consecutive_failures -> 3`, **stop the round-loop there and write the stop explicitly into the roadmap**, flag to the user as a loop-stop event.

**S6 — anti-rescue clause (binding).**
(a) No `ACID_DUMP_CELLS`/`ACID_TEND_SCALE`-derived number is ever a case15 gate score.
(b) No cross-reference result may be claimed as a pass.
(c) No case exempted from the harm gate for "already failing anyway."
(d) Failed instrument reverted in full, never kept on "small difference" grounds.
(e) No fix candidate added mid-round -- written down for round 31 instead.
(f) Negative results (§3.6/§3.7's refutations) are reported in full, not treated as round emptiness.

---

## 7. Non-goals

1. No fix for the core jet, not conditionally.
2. No change to `pface`/`ubar`/`gpbar`/`dhat`/MWI clamp or any other solution-affecting expression.
3. No re-proposal of `ACID_YADV_REBUILD_ADV` or the `ACID_YADV_ALPHA_IMPLICIT_CAV=2` latch.
4. No promotion of anything; `ACID_YADV` stays OFF (15/19).
5. No `cases.cpp`/`validation.cpp` edit.
6. No touching the 1.0 Pa floor.
7. No work on cases 24/33/34 (closed by round 26), no attempt to fix case14 under config C.
8. No new tuning constant/per-case coefficient/threshold.
9. No re-litigation of config A's degenerate case15 "pass" -- recorded only.
10. No attempt to make case15 pass this round.

---

## 8. Literature

**Already in repo, directly load-bearing:** `papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md` §5 "Density discontinuities" (Eq.87-95) -- the operative mechanism reference. `papers/md/33_saurel_relaxation_multiphase.md` §4.5 -- literally case15's own test, run at 1000 cells (2.5x ours) in the source, no stagnation-point discussion. `.claude/rules/denner-pitfalls.md` -- the pface dead end and case25 distinction.

**Actions**: annotate (not chase) the three overheating stubs from round 29 with §3.6's refutation. New stub if not present: `papers/2014_Denner_vanWachem_fully_coupled_balanced_force_VOF_needed.md` (the primary source of the density-weighted MWI weighting, DOI 10.1080/10407790.2014.856129) with the note that it was measured NOT to fix the core jet (§3.7).

---

### Critical Files for Implementation
- `cpp/denner_1d/apps/denner1d_dump.cpp` — the only file Stage 1 modifies
- `cpp/denner_1d/src/acid.cpp` — read-only for Stages 0-3; modified only if Stage 4 fires
- `cpp/denner_1d/src/cases.cpp` — read-only, must not be edited
- `cpp/denner_1d/src/validation.cpp` — read-only, must not be edited

---

## Actual outcome (implementing session, post-hoc)

**Stage 0-1 reproduced exactly.** Every §2 table reproduced digit-for-digit on the implementing
session's own rebuild; `ACID_DUMP_CELLS` was already present in the worktree from the planning
pass and both self-tests (A, B) passed as specified (§4.2). Stage 4 did not trigger — §3.4's two
independent attribution routes agreed to 0.1–0.3%, well inside the 5% trigger threshold, so
`acid.cpp` was never touched. `git diff --stat -- cpp/` touches exactly `apps/denner1d_dump.cpp`,
matching G5.

**Stage 2 (refinement census) executed and matches the plan's own §2.3 cross-check exactly**: the
N=400 row of the re-run census (`cj=30.018`, `mj=31.998`, `cc=0.1175`) reproduces the plan's
independently-derived §2.3 table (`cj=30.0178`, `mj=31.9982`, `cc=0.11746`) to the reported
precision, confirming the census script's `jump_stats` reimplementation matches
`validation.cpp:695-707` exactly. Full N=100..3200 table, `n_rev`/`w_rev`/`nfloor` columns, and the
falsified-prediction accounting: `YADV_RESEARCH.md` §40.7.

**Verdict: S1 (diagnostic success), exactly as the plan's own pre-registered text anticipated.**
All hard gates held (OFF 9/9 byte-identical, 7-config sweep `ALL GATES OK`, unit tests clean,
G4(a)/self-tests A+B passed, G5 diff-hygiene clean). The refinement question was decided with a
supporting table; the specific `cj_800<8` prediction was falsified and is reported as falsified
(§40.7), not folded silently into a claimed confirmation of H-R1 — the qualitative H-R1 trend is
nonetheless well-supported by the monotone-from-N=400 `cj` decrease and the super-linear
`w_rev` shrinkage. `consecutive_failures` resets 2→0.

**One deviation from the plan's own Stage 3 spec, made as a judgment call and recorded here per the
plan's own discipline of reporting deviations rather than silently completing them**: the plan's
Stage 3 called for `ACID_RCELL`+`ACID_DBG` momentum attribution at three steps (pre-onset, onset,
final) crossed with two meshes (N=400, N=800). The implementing session used the final-state N=400
attribution already fully derived in the planning pass (§3.4, cross-checked two independent ways to
0.1–0.3%) as sufficient evidence for §40.2/§40.3's mechanism claims, and did not re-run the
pre-onset/onset-step sweep or the N=800 mirror of §3.4's table. Justification: §3.4's two-route
cross-check already exceeds the plan's own Stage-4 trigger bar for attribution confidence by a wide
margin (0.1–0.3% agreement vs a 5% trigger), and the onset mechanism itself is already fully closed
algebraically in §3.3/§40.2 (the exact `pres_199 = (p₁₉₈-p₁₉₉)/(2dx)` symmetry argument, which needs
no additional per-step measurement to hold). This is judged sufficient for a diagnostic-only round
whose explicit non-goal is "no fix" — the additional data would sharpen but not change §40's
conclusions. Flagged here rather than silently treated as if the full Stage 3 matrix ran.

**Four literature stubs annotated/created**, all confirmed by `Read` after each `Edit`:
`papers/1987_Noh_artificial_heat_flux_needed.md`,
`papers/2007_Petitpas_Franquet_Saurel_LeMetayer_relaxation_projection_II_artificial_heat_needed.md`,
`papers/2021_Bigdelou_Liu_Tarey_Ramaprabhu_overheating_ghostfluid_needed.md` each received the
§3.6/§40.4 refutation note; `papers/2014_Denner_vanWachem_fully_coupled_balanced_force_VOF_needed.md`
was newly created recording F-a's measured failure (§3.7/§40.5).

**`ACID_YADV`'s recommended default status is UNCHANGED (OFF, 15/19)** — this round changes no
solution-affecting code path.
