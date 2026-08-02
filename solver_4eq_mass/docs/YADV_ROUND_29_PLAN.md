# ROUND 29 PLAN — case15 under `B+CAV`: the reported density gap is a **reference** artifact; three separable defects, only one of which any mask work can reach

**Target**: `solver_4eq_mass/`, `ACID_YADV=1` mass-fraction transport path, case15 (`15_E air-water cavitation`).
**HEAD**: `35b3cb8` ("ACID_YADV round 28 — case15 mass collapse solved, accuracy gap remains (S4)"). Every code citation below was read live at this HEAD; the round-29 worktree's `acid.cpp` is `diff`-identical to the main tree's.
**Control state on entry**: `round_counter=28`, `consecutive_failures=1`.

**Advisor verification note**: key structural claims checked directly against the code -- `acid.cpp:622` (`alpha_implicit_cav` presence test, exact match), `acid.cpp:628` (`cav_dbg`, exact match), `acid.cpp:1109` (`for (int retry = 0; retry < 14; ++retry) {`, exact match), `acid.cpp:1585` (`std::vector<char> cav(n, 0);`, exact match), `acid.cpp:2470` (the infeasibility predicate `sbak.p[i] + om * dxk[i][1] <= 1.0`, exact match), `acid.cpp:915` (time-loop start `while (t < t_end && step < c.config.max_steps)`, exact match). No structural error found.

---

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §39)**: Stage 0
reproduced exactly (§2.1's headline table, §2.3's mask shrink sequence `70,62,56,52,48,...` both
digit-for-digit). Stage 1 (level parse + spatial `NFEAS` fields) landed clean, G4-early
byte-identical. M2's required pre-Stage-2 check passed: case14's level-2 shadow count is 0
throughout. **T0's harm gate initially read `pass_count=15/19`, fail set unchanged** -- but a
closer look (not skipped, per the plan's own discipline) found **case33, `finite:true` under
both plain B and level 1, flips to `finite:false` under level 2**. The plan's own S5 rule and its
anti-rescue clause are explicit that this counts as harm regardless of case33's pre-existing
fail status -- **S5 fires**, exactly as pre-registered, and no attempt was made to argue it away.
The latch mechanism (the run-scope hoist + conditional clear) was reverted in full; the harmless
level-parse and `ACID_NFEAS`'s spatial fields were kept as gated-off diagnostics, re-verified
byte-identical post-revert (level 2 now degenerates to level 1's exact behaviour, case33 confirmed
`finite:true` again). `consecutive_failures` incremented to 2 per S5's explicit instruction --
**now 2/3 toward the loop's own stop condition**, both increments on this same case15
mass-collapse-family thread. The round's diagnostic value (§2's reference-artifact correction,
the binding-blocker identification shared with config C) stands unconditionally on Stage 2's
outcome and is the round's real, durable output. All hard gates held.

---

## 0. Executive summary

Round 28 closed case15's mass collapse and left one stated open question: *"why is `B+CAV`'s density field measurably worse than config C's, despite near-identical mass?"*, with one stated (unmeasured) hypothesis: a transition-zone inconsistency between `cav`-active and `cav`-inactive cells.

I measured it directly, before designing anything. The answer is not what round 28 supposed, and it changes what this round can and cannot achieve:

1. **`B+CAV`'s N=400 solution is already config C's solution.** Comparing the two fields cell-by-cell: `l2_rho = 0.001580`, `corr_rho = 0.999995`. The density field `B+CAV` produces on case15's own validation grid is C-grade to 0.16%.

2. **The entire reported `l2_rho`/`corr_rho` gap lives in the N=800 self-convergence *reference*, not in the solution.** case15's reference is `computed_reference(c, 800)` — the *same solver under the same env config* on a 2x mesh (`cases.cpp:751-753`). Scored against config C's own N=800 reference instead of its own, `B+CAV`'s N=400 solution gives `l2_rho = 0.01972`, `corr_rho = 0.996662` — statistically indistinguishable from C's own `0.01966` / `0.996734`, and **both clear their gates**. The `B+CAV` N=800 run develops a spurious dense/hot core plug (`rho_ref = 662.6`, `p_ref = 1.195e4` at `x=0.4963`, where C's reference has `86.7` / `591`) that poisons the reference.

3. **Round 28's own characterization of the remaining gap is incomplete.** It reported "`l2_rho`/`corr_rho` alone; `l2_p`/`corr_p` clear their bars". The full gate (`validation.cpp:684-729`) shows `B+CAV` failing **six** criteria, not two: `l2_rho`, `corr_rho`, plus `smooth_ok` (`cj = 25.906`, `mj = 25.906`, `cc = 0.10293` against limits `8.0`/`8.0`/`0.04`) and `osc_ok` (`p_osc = 0.4476` against `< 0.02`). Round 28 never computed `smooth_ok`/`osc_ok` for `B+CAV`. The briefing premise for this round ("it currently fails on `l2_rho`/`corr_rho` alone") is therefore false and is corrected here.

4. **Round 28's transition-zone hypothesis is CONFIRMED — but it explains the *pressure* failure, not the density one.** The `B+CAV`-minus-C pressure difference is a ~28 kPa spurious rarefaction notch at cells `134-142` / `258-266` (`x ≈ 0.336-0.356` and `0.644-0.664`), which is precisely where the `cav` mask's spatial boundary sits in the last steps (mask size 130 cells centred on `i=199.5` → edges at `i=135` / `i=264`). Config C, which has no boundary, has `p_osc = 0.0000` and a smooth monotone profile there. So the mask edge radiates a pressure artifact; it barely touches `rho` (max `|Δrho| = 2.53` on ~1000), which is why it never showed up in round 28's density-only table.

5. **The mask's non-monotonicity across steps is measured, not conjectured.** `cav[]` is declared *inside* the retry body (`acid.cpp:1585`) and must be re-earned every step. `ACID_NFEAS`'s own per-step counts show the mask **shrinking from 70 cells (step 0) to 40 cells (step 8)** before growing again — at N=400 *and*, digit-for-digit identically, at N=800. At least 30 cells that the exact infeasibility predicate flagged at step 0 are running the frozen (≈521x too stiff, per this file's own Jacobian comment) closure eight steps later. This is the mechanism behind both the boundary notch and the N=800 core plug.

6. **The binding blocker for case15 is none of the above, and no mask work can reach it.** Even scored against a perfect (config-C) reference, `B+CAV` still fails on `smooth_ok` (`cj = 25.906`) and `osc_ok`. And **config C itself — the state any mask fix converges toward — fails case15 on `smooth_ok` alone (`cj = 30.018`, `mj = 31.998`, `cc = 0.11746`)**: round 27 §4.5's still-untouched stagnation-point core-jet (a 4-cell velocity sign reversal at `x=0.5`, measured again here: C's `u` runs `-13.84, +18.15, +15.01, -15.01, -18.15, +13.84` across cells 197-202, while both N=800 references are monotone through zero). **Therefore case15 cannot pass under any variant of the implicit-alpha family until the core-jet defect is solved, and that defect is independent of `cav[]`.**

**What this round should do, given that.** Pre-registered up front, so there is no room to rescue a result later: **this round cannot make case15 pass, and it does not attempt to.** It proposes (a) a spatial extension of the existing `ACID_NFEAS` diagnostic to nail the mask's geometry and confirm/falsify the mechanism, and (b) one narrow, constant-free, blast-radius-bounded candidate — `ACID_YADV_ALPHA_IMPLICIT_CAV=2`, a run-scoped **latch** on the existing mask (once the exact predicate has fired on a cell, that cell stays implicit) — whose *entire* claimed benefit is to remove defects (2) and (4)/(5), i.e. to make `B+CAV` reach config C's case15 quality *without* C's case13/14/25 cost. If it works, case15's status changes from "three defects, one of them unexplained" to "one sharply-defined defect shared with C", which is round 30's target. That is an **S4 (partial)** outcome by construction, not an S1.

---

## 1. Verified code facts (read live at `HEAD = 35b3cb8`)

All paths relative to `solver_4eq_mass/`.

| Fact | Location | Exact content (verified) |
|---|---|---|
| `ACID_YADV_ALPHA_IMPLICIT_CAV` flag + its rationale comment | `cpp/denner_1d/src/acid.cpp:594-622` | `const bool alpha_implicit_cav = std::getenv("ACID_YADV_ALPHA_IMPLICIT_CAV") != nullptr;` — a **presence test**, no level parse |
| `ACID_NFEAS` diagnostic flag | `acid.cpp:623-628` | `const bool cav_dbg = std::getenv("ACID_NFEAS") != nullptr;` |
| Retry loop start (14 retries) | `acid.cpp:1109` | `for (int retry = 0; retry < 14; ++retry) {` — followed immediately by `s = s0; Yv = Yv0;` |
| **`cav[]` / `cav_n` declaration — INSIDE the retry body** | `acid.cpp:1579-1586` | `std::vector<char> cav(n, 0); int cav_n = 0;` — the comment itself says "Reset every retry restart (this declaration is inside the retry body)" |
| Shadow mask (diagnostic-only, keeps `ACID_NFEAS` a true no-op) | `acid.cpp:1587-1591` | `std::vector<char> cav_shadow(n, 0); int cav_n_shadow = 0;` |
| `compute_R`'s implicit-alpha re-derivation (the solution-affecting consumer) | `acid.cpp:1611-1619` | `if (yadv && (alpha_implicit \|\| cav_n > 0)) { … if (!(alpha_implicit \|\| cav[i])) continue; s.alpha[i] = clamp(alpha_from_mass_fraction(Yv[i], phase_props(pu,Tu,A).rho, phase_props(pu,Tu,B).rho),0,1); }` with `pu = max(s.p[i],1.0)`, `Tu = max(s.T[i],1e-6)` |
| Jacobian per-cell `aimp` gate | `acid.cpp:2186` | `const bool aimp = yadv && (alpha_implicit \|\| cav[i]);` |
| Jacobian alpha-sensitivity addend gate | `acid.cpp:2282` | `if (yadv && (alpha_implicit \|\| cav_n > 0)) {` |
| Pressure floor (the literal the predicate reuses) | `acid.cpp:2443` | `s.p[i] = std::max(sbak.p[i] + dpi, 1.0);` (coupled 3x3 path) |
| **Mask setter site** (after the line search closes) | `acid.cpp:2467-2491` | `if (yadv && (alpha_implicit_cav \|\| cav_dbg)) { … const bool trip = sbak.p[i] + om * dxk[i][1] <= 1.0; if (alpha_implicit_cav && !cav[i] && trip) { cav[i]=1; ++cav_n; ++grew; } if (!cav_shadow[i] && trip) { cav_shadow[i]=1; ++cav_n_shadow; ++grew_shadow; } … }` |
| `rbest`/`best_it` reset when the residual function changes | `acid.cpp:2474-2479` | `if (grew && ajac) { rbest = max; best_it = it; }` |
| `NFEAS` print (has **no spatial field**, and no `n`) | `acid.cpp:2481-2490` | `"NFEAS case=%s step=%d retry=%d it=%d cav_n=%d grew=%d shadow_n=%d shadow_grew=%d r_init=%.4e rnow=%.4e\n"` |
| Pre-Newton alpha recovery at the **stale** `(p_o,T_o)` | `acid.cpp:1343-1349` | `s.alpha[i] = clamp(alpha_from_mass_fraction(Yv[i], phase_props(max(p_o[i],1.0),max(T_o[i],1e-6),A).rho, …),0,1);` — **textually the same expression** as `acid.cpp:1615`, evaluated at the same `(p,T)` on Newton iteration 0 |
| `p_o`,`T_o` capture (retry scope) | `acid.cpp:1112` | `const Vec u_o = s.u, p_o = s.p, T_o = s.T;` |
| `eval_thermo` — where `alpha` becomes `rho` | `acid.cpp:301-327` | `rho = al*pa.rho + (1-al)*pb.rho`; `drhodp = al/(Ra T) + (1-al)/(Rb T)` — the frozen-composition compressibility |
| Time-step loop start (run scope boundary) | `acid.cpp:915` | `while (t < t_end && step < c.config.max_steps) {` |
| Legacy 2x2 segregated Newton's own line search — **no mask setter there** | `acid.cpp:2660-2680` | `s.p[i] = std::max(sbak.p[i] + dpi, 1.0);` at `:2676`; reachable only via `ACID_NO_UNIC` |
| `ACID_MBAL` `adv`/`remap` arithmetic | `acid.cpp:2757-2758` | `const double adv = mbal_Mstar - M_prev; const double remap = mbal_Mreb - mbal_Mstar;` |
| case15 reference = **same solver, same env, N=800** | `cpp/denner_1d/src/cases.cpp:751-753` | `if (c.id == "15") return computed_reference(c, 800);` |
| `computed_reference` interpolates `alpha,u,p,T` then `refresh_thermo` | `cases.cpp:423-437` | `hi.config.cells = std::max(c.config.cells, cells); const PrimitiveState fine = solve_case(hi);` — **reads the same env**, so the reference runs whatever config the primary runs |
| case15 config | `cases.cpp:493` | `base_config(400, 9.5e-4, 0.0, 1.0)`; `cfl = 0.45` (`cases.cpp:21`) |
| case15 IC | `cases.cpp:682-688` | `alpha = 0.055`, `u = ∓100`, `p = 1e5`, `T` = alpha-blend |
| case15 gate (all 8 criteria) | `cpp/denner_1d/src/validation.cpp:684-729` | `corr_p≥0.93, corr_u≥0.998, corr_rho≥0.99, l2_p≤0.18, l2_u≤0.06, l2_rho≤0.05, smooth_ok, osc_ok`; `smooth_ok = cj ≤ max(8.0, 1.10 cj_r) && mj ≤ max(8.0, 1.10 mj_r) && cc ≤ max(0.04, 1.10 cc_r)`; `osc_ok = p_osc < 0.02 && r_osc < 0.04` |
| The smoothness criteria are **spec-mandated, not a code invention** | `validation/1D/15_E_Cavitation.md:190-193` | "velocity smoothness in the cavitation core must also match the reference shape … 목적: correlation/L2만으로 통과되는 one-cell step-like velocity fan을 FAIL 처리한다" |
| No grid-override env exists in this solver | (grep) | `ACID_N` / `DENNER_CELLS` / `DENNER_CASE15_REF_N` are **not implemented** here (the last is named in the spec doc but never wired) |

**One structural fact worth flagging for the implementer**: because `acid.cpp:1615` and `acid.cpp:1345` are the *same expression at the same `(p,T)`*, re-deriving `alpha` on a latched cell at Newton iteration 0 of a step is **bit-exactly an identity**. A latch therefore cannot inject anything at a step boundary; it can only change iterations ≥ 1. This is load-bearing for §3.3.

---

## 2. The diagnostic work — where the error actually lives

All numbers below were measured this round with the already-built `HEAD` binary (`build-cpp/cpp/denner_1d/denner1d_dump`, `…/denner1d_validate`) and the existing `scripts/yadv_r27_case15.py` machinery. No code was changed to produce them. Commands in §9.

### 2.1 Round 28's headline table reproduces exactly, and is incomplete

`python3 scripts/yadv_r27_case15.py overlays` reproduces §38.5 digit-for-digit. But the script's `overlays` mode prints only the `l2`/`corr` columns; running the **full** gate (its own `case15_gate`, a faithful reimplementation of `validation.cpp:684-729`) gives:

| config / reference | `l2_p` | `l2_u` | `l2_rho` | `corr_p` | `corr_u` | `corr_rho` | `cj` | `cj_r` | `mj` | `cc` | `cc_r` | `smooth_ok` | `p_osc` | `osc_ok` | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A (OFF)** / A ref | 0.00000 | 0.02801 | 0.04246 | 1.000000 | 0.998621 | 0.995036 | 6.044 | 2.259 | 6.044 | 0.03449 | 0.01246 | **True** | 0.0000 | **True** | **True** |
| **B (plain)** / B ref | 0.16653 | 0.01874 | **0.16761** | 0.985535 | 0.999334 | **0.984514** | 2.306 | 2.290 | 2.306 | 0.01292 | 0.01196 | True | 0.0000 | True | False |
| **C** / C ref | 0.01439 | 0.01704 | 0.01966 | 0.999285 | 0.999344 | 0.996734 | **30.018** | 3.546 | **31.998** | **0.11746** | 0.08439 | **False** | 0.0000 | True | False |
| **B+CAV** / CAV ref | 0.13338 | 0.01275 | **0.06898** | 0.994056 | 0.999635 | **0.957806** | **25.906** | 1.222 | **25.906** | **0.10293** | 0.06547 | **False** | **0.4476** | **False** | False |
| **B+CAV** / **C's** ref | 0.05515 | 0.01452 | **0.01972** | 0.992958 | 0.999525 | **0.996662** | 25.906 | 3.546 | 25.906 | 0.10293 | 0.08439 | False | 0.5196 | False | False |
| **C** / **CAV's** ref | 0.17135 | 0.01512 | **0.06925** | 0.994925 | 0.999485 | **0.957985** | 30.018 | 1.222 | 31.998 | 0.11746 | 0.06547 | False | 0.0000 | True | False |

Three things fall straight out of the last two rows, which are the whole diagnosis:

* **`B+CAV` scored against C's reference → `l2_rho = 0.01972`, `corr_rho = 0.996662`. Both PASS.** Config C's own numbers are `0.01966` / `0.996734`. The difference is in the fifth digit.
* **Config C scored against `B+CAV`'s reference → `l2_rho = 0.06925`, `corr_rho = 0.957985`.** Config C's celebrated density accuracy *evaporates* the moment you score it against `B+CAV`'s reference. The metric is a property of the reference, not of the scheme.
* `p_osc` and `smooth_ok` are the criteria round 28 never looked at, and both fail for `B+CAV`.

### 2.2 The N=400 fields are the same; the N=800 references are not

Direct cell-by-cell comparison (relative-L2 in the gate's own `rel_scale` normalisation):

```
CAV400 vs C400   : rho l2=0.001580 corr=0.999995 | p l2=0.051283 corr=0.993546 | u l2=0.003075 corr=0.999979
CAVref vs Cref   : rho l2=0.051851 corr=0.972084 | p l2=0.149367 corr=0.995661 | u l2=0.002375 corr=0.999987
```

The solutions agree to 0.16% in density; the references disagree by 5.2%, i.e. by more than the whole gate budget. The reference disagreement is localised at the stagnation point (values sampled on the 400-grid, symmetric about `i=199.5`):

| `i` | `x` | `CAVref rho` | `CAVref p` | `Cref rho` | `Cref p` | `CAV400 rho` | `C400 rho` |
|---|---|---|---|---|---|---|---|
| 196 | 0.4913 | 157.32 | 1.084e3 | 150.33 | 1.028e3 | 161.99 | 162.00 |
| 197 | 0.4938 | 124.30 | 8.321e2 | 164.25 | 1.142e3 | 71.71 | 71.13 |
| **198** | 0.4963 | **662.63** | **1.195e4** | 86.67 | 5.914e2 | 0.561 | 0.558 |
| **199** | 0.4988 | **362.76** | **7.613e3** | 7.515 | 4.433e1 | 2.656 | 2.697 |

The `B+CAV` N=800 run **fails to cavitate in a ~0.012 m core** and instead leaves a dense, hot plug: `rho` rises from 124 to 663 as you approach `x=0.5`, and `p` rises from 830 Pa to 12 kPa. Config C's N=800 run expands monotonically to `rho = 7.5`, `p = 44 Pa`. Consequence for the metric: the reference's `rho` range collapses from `[7.5, 949.4]` to `[124.3, 941.4]`, so even the gate's `rel_scale` differs (941.9 vs 817.1) — the two configs are not being scored on the same yardstick.

### 2.3 The `cav[]` mask is not monotone across steps — measured

`ACID_NFEAS=1` with the applying flag on, on `denner1d_dump 15` (which runs *both* the N=400 primary and the N=800 reference in one process; the two runs separate cleanly because `step` resets):

```
run0 (N=400): 85 steps, 619 NFEAS lines, max retry = 0, no stalls, total grew = 7306
  per-step max cav_n: 70 62 56 52 48 44 42 42 40 42 44 46 48 50 52 54 56 58 58 60 62 62 64 66 66 68 …
  … → 130 (step 83), 116 (step 84, the short landing step)
run1 (N=800): 169 steps, 1299 lines, max retry = 0, no stalls, total grew = 22194
  per-step max cav_n: 70 62 56 52 48 44 42 42 40 42 44 46 48 50 52 54 56 58 58 60 62 62 64 66 66 68 …
  … → 220 (step 167), 218 (step 168)
```

Two facts, both load-bearing:

* **The mask shrinks by 30 cells over steps 0→8, at both resolutions.** Cells the exact infeasibility predicate flagged at step 0 are back on the frozen closure at step 8. That is direct evidence of the *re-earn* problem: `cav[]` is retry-scoped (`acid.cpp:1579-1586`), so a cell whose pressure is currently *rising* — precisely the stagnation-point cell being compressed by the core jet — never re-trips `sbak.p[i] + om*dxk[i][1] <= 1.0`, reverts to the ≈521x-too-stiff closure, and is compressed further. A positive feedback.
* **The per-step mask *count* is digit-identical at N=400 and N=800 for the first 40 steps.** A physically-defined region would double its cell count under refinement. This one does not: at step 0 the masked band is 0.175 m wide at N=400 and 0.0875 m wide at N=800. The activation criterion is a per-step pressure-**increment** test, and the increment scales with `dt ∝ dx`; so the discrete operator depends on the mesh in a way that does not vanish under refinement. **That is exactly why the N=800 "self-convergence reference" is not a refinement of the N=400 scheme**, and hence why §2.2's numbers look the way they do.

No retries and no stalls occur in either run (`max retry = 0`, zero non-`NFEAS` stderr lines), so the plug is not a retry/dt-collapse artifact. `ACID_RHIST` at steps 1/5/40/84 confirms clean Newton convergence to `~5e-7` in both runs (the N=400 final step takes 17 iterations with a residual plateau at `1.6e4` during `it=5..12` — the signature of the mask growing mid-solve and resetting `rbest` at `acid.cpp:2474-2479` — then converges).

### 2.4 Round 28's transition-zone hypothesis: CONFIRMED, in pressure

Where do `B+CAV` and C's N=400 fields actually differ? Ranked cell-by-cell:

```
top |p_CAV − p_C| : 28396 Pa @ i=139/260 (x=0.3488/0.6512); 27657 @ 138/261; 24653 @ 137/262;
                    24047 @ 140/259; 20430 @ 136/263; 15964 @ 135/264; 12058 @ 134/265
top |u_CAV − u_C| : 8.45 m/s @ i=198/201; 2.06 @ 199/200; everything else ≤ 0.089
top |rho_CAV−rho_C|: 2.53 kg/m³ @ i=133/266 (on ~1000 kg/m³)
```

and the raw profile shows what it is — a spurious ~30 kPa rarefaction **notch**:

```
CAV p, i=130…150: 9.391e4 9.316e4 9.204e4 9.033e4 8.767e4 8.367e4 7.906e4 7.466e4 7.142e4 7.038e4
                  7.434e4 | 9.514e4 | 9.483e4 9.430e4 9.356e4 9.259e4 9.141e4 9.001e4 8.837e4 …
C   p, i=130…150: 9.993e4 9.990e4 9.986e4 9.981e4 9.973e4 9.963e4 9.949e4 9.932e4 9.908e4 9.878e4
                  9.839e4   9.789e4   9.727e4 9.650e4 9.556e4 9.443e4 9.309e4 9.151e4 8.967e4 …
```

C is smooth and monotone; `B+CAV` dips 30 kPa and jumps back 21 kPa in one cell at `i=141`. The mask covered 130 cells centred on `i=199.5` at step 83 → its edges are at `i=135` and `i=264`. **The notch sits on the mask edge.** The `p`-TV excess confirms it quantitatively:

```
CAV: TV(p)=3.0447e5  TV(p_ref)=2.1033e5  excess=9.4136e4 → p_osc=0.4476
     top excess faces: 141/259 (20646 Pa each), 136/264 (4499), 137/263 (4296), 135/265 (3908)
C:   TV(p)=2.0002e5  TV(p_ref)=2.0037e5  excess=−349     → p_osc=0.0000
     top excess faces: 197/203 (564)  — 36x smaller, and at the core, not at any boundary
```

So round 28's hypothesis was right about the *mechanism* (cav-active/cav-inactive transition zone) and wrong about the *observable*: it costs `osc_ok` and part of `l2_p`, not `l2_rho`. The mixture density at 90 kPa is water-dominated and essentially insensitive to a 30 kPa pressure notch, which is why a density-only table missed it entirely.

*(Inference flagged as such: the mask is assumed centred and contiguous, from the symmetry of the notch about `x=0.5` and from `cav_n/2` matching the notch radius. `ACID_NFEAS` prints no spatial information today. Stage 1 exists to confirm this.)*

### 2.5 The binding blocker: the stagnation-point core jet, shared with config C

Velocity across the centre at N=400 vs the N=800 references:

| `i` | `x` | `CAV400 u` | `C400 u` | `CAVref u` | `Cref u` |
|---|---|---|---|---|---|
| 196 | 0.4913 | −17.78 | −17.78 | −18.05 | −17.45 |
| 197 | 0.4938 | −13.75 | −13.84 | −18.34 | −20.83 |
| 198 | 0.4963 | **+9.70** | **+18.15** | −13.82 | −19.85 |
| 199 | 0.4988 | **+12.95** | **+15.01** | −0.61 | −1.77 |
| 200 | 0.5012 | **−12.95** | **−15.01** | +0.61 | +1.77 |
| 201 | 0.5038 | **−9.70** | **−18.15** | +13.82 | +19.85 |

Both N=400 solutions have a **4-cell velocity sign reversal**: an inward jet at the exact stagnation point of an otherwise outward-diverging flow, piling mass into the two central cells (`C400`: `p = 15.7 Pa` at `i=199/200` against `3.24 Pa` at `i=198/201`; `CAV400`: `4945 Pa` against `3.26 Pa`). Both N=800 references are monotone through zero. This is round 27 §4.5's already-characterised defect ("an under-resolved near-vacuum core at the stagnation point, 4-cell velocity sign reversal") reproduced independently here, and it is exactly the failure mode `validation/1D/15_E_Cavitation.md:193` says the smoothness criteria exist to catch.

Its consequences for this round are decisive:

* `cj = 25.906` (`B+CAV`), `30.018` (C) against a limit of `max(8.0, 1.10·cj_r)`; `cj_r` is 1.222 / 3.546, so the limit is `8.0` in both cases. **Neither config can pass `smooth_ok` on this metric no matter what the reference does.**
* Any mask fix moves `B+CAV` *toward* C, and C's `cj` is *worse* (30.0 vs 25.9). So the ceiling of this candidate family on case15 is "config C's quality" — which is a FAIL.
* That the artifact is **absent at N=800** in both references (monotone `u`, no sign reversal at the sampled points) is evidence it is a genuine under-resolution artifact that converges away, not a structural incompatibility — i.e. it *is* legitimately attackable, but by scheme work at the near-vacuum core, not by anything in the `cav` family. Caveat, stated honestly: the reference is a *linear interpolation* of the N=800 field onto the 400-grid, so it cannot resolve sub-fine-cell structure; §4 Stage 4 proposes the (optional) measurement that would settle it.

### 2.6 Alternatives falsified or excluded, with the measurement that did it

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Round 28's transition zone explains `l2_rho`/`corr_rho` | **Falsified** | `CAV400 vs C400` density `l2 = 0.001580`, `corr = 0.999995`. There is no density discrepancy in the N=400 field to explain. |
| THINC / rho-monotonicity-guard interaction (briefing item 5) | **Excluded, live measurement** | `ACID_THINC_DBG=1` under `B+CAV` prints `THINC case=15 activations=0 rho_guard_rejects=0` for **both** the N=400 and the N=800 run. Structurally necessary: case15's colour function (`Y ≈ 7e-5`) is uniform, so the indicator's `straddle` test (`min(a_{i-1},a_{i+1}) < 0.5 < max`) can never fire. Consistent with `denner-pitfalls.md`'s own documented activation list, which already excludes 15. |
| A retry/stall/dt-collapse produces the N=800 plug | **Excluded** | `max retry = 0` and zero non-`NFEAS` stderr lines in both runs. |
| Jacobian-quality effect specific to partial activation | **Not excluded, but not needed** | The `aimp` gates (`:2186`, `:2282`) are per-cell and are consistent with whatever `cav[]` says on each iteration; the measured artifacts (boundary notch, non-monotone mask) are fully explained by *which* cells are implicit, not by the Jacobian. A latch subsumes this: it makes the Jacobian gate consistent across steps too. |

### 2.7 A note that must be recorded, but must not be acted on

Config A (the OFF path, which "passes" case15 19/19) produces on case15: `nfloor = 400/400`, `p ≡ 1.0 Pa` across the *entire* domain, `alpha ≡ 0.055000` unchanged everywhere, `rho ≈ 1022` uniform, `min rho = 949.3`. Its `l2_p = 0.00000` and `corr_p = 1.000000` **exactly because its N=800 reference is equally collapsed** — the same degeneracy round 28 §38.2 flagged and corrected for `B+F3`. Saurel/Petitpas/Berry (`papers/md/33_saurel_relaxation_multiphase.md` §4.5) run *literally this test* and state the physics: "As gas is present, the pressure cannot become negative. To maintain positive pressure, the gas volume fraction increases and creates a cavitation pocket." The frozen-alpha path cannot do this by construction (`alpha` is pinned at 0.055); the implicit-alpha family does it (`alpha → 0.9994` at the core).

**This is recorded as a finding, not as a licence.** It does not justify touching `validation.cpp` or the spec, and this plan does not propose to. The spec's smoothness criteria are independently justified (`15_E_Cavitation.md:190-193`) and the core jet they catch is a real defect. Round 30, or a user escalation, is the right home for the broader "config A passes case15 degenerately" question.

---

## 3. Candidate: `ACID_YADV_ALPHA_IMPLICIT_CAV=2` — a run-scoped latch on the existing mask

### 3.1 Derivation

The predicate `sbak.p[i] + om*dxk[i][1] <= 1.0` (`acid.cpp:2470`) tests a property of the **cell's thermodynamic state**: this cell's mixture, at its current composition, has a frozen-composition compressibility (`drhodp = al/(Ra T) + (1−al)/(Rb T)`, `acid.cpp:326`) that understates the true PTE compressibility by the factor this file's own Jacobian comment measures ("`D_p 1.00196e-06 → D_p* 5.22580e-04`, a factor 521.56"), by enough that no admissible pressure closes its continuity equation. That property does not expire at the end of a time step.

But the mask *does* expire: `cav[]` is declared inside the retry body (`acid.cpp:1579-1586`) and must be re-earned from scratch at every retry, hence at every step. Three measured consequences:

* **(a)** the first Newton iteration of every step is frozen-alpha even in a persistently cavitating cell (round 28 §38.5's own note);
* **(b)** a cell whose pressure is *rising* — the stagnation-point cell being compressed by the core jet, or any cell in a locally recompressing part of the core — cannot re-trip a test on a *downward* pressure demand, so it silently reverts to the 521x-too-stiff closure for a whole step. Frozen ⇒ stiff ⇒ inbound mass raises `p` sharply ⇒ still won't trip. **Measured**: the mask shrinks 70 → 40 cells over steps 0-8 (§2.3);
* **(c)** because the test is on a per-step *increment*, and increments scale with `dt ∝ dx`, the mask's spatial support is mesh-dependent — measured as an identical *cell count* (not an identical width) at N=400 and N=800 (§2.3). A scheme whose discrete operator depends on `dx` in a non-vanishing way is not consistent under refinement, which is precisely what makes case15's self-convergence reference meaningless for this config.

The minimal change that removes (a), (b) and (c) together — **without inventing a new predicate, a new constant, or a second alpha** — is to make the mask a **latch**: once the exact, already-existing predicate has fired on a cell, that cell stays implicit for the rest of the run.

### 3.2 Exact code-change spec

Four edits, all in `cpp/denner_1d/src/acid.cpp`. No other file changes (except the optional Stage 4, §4).

**(1) Level parse, replacing the presence test at `acid.cpp:622`.**

```cpp
// ACID_YADV_ALPHA_IMPLICIT_CAV level (round 29). "=1" (or any non-numeric/zero value, i.e.
// mere presence -- the pre-round-29 semantics, preserved exactly) keeps the RETRY-SCOPED mask
// round 28 shipped. "=2" LATCHES the mask for the whole run: once the (unchanged, constant-
// free) infeasibility predicate at the line-search site has fired on a cell, that cell keeps
// the implicit-alpha closure. Motivation, all measured in docs/YADV_ROUND_29_PLAN.md sect.2.3:
// the retry-scoped mask SHRINKS 70 -> 40 cells over case15's steps 0-8 (a cell whose pressure
// is RISING cannot re-trip a test on a downward pressure demand, so it reverts to the ~521x-
// too-stiff frozen closure), and its size is mesh-INDEPENDENT (identical counts at N=400 and
// N=800, i.e. half the physical width on the finer mesh), which is what makes case15's own
// N=800 self-convergence reference not a refinement of the N=400 scheme.
const int cav_level = [] {
    const char* e = std::getenv("ACID_YADV_ALPHA_IMPLICIT_CAV");
    if (!e) return 0;
    const int v = std::atoi(e);
    return v > 0 ? v : 1;   // presence with a non-numeric/0 value == the old "on" (level 1)
}();
const bool alpha_implicit_cav = cav_level > 0;
```

**(2) Same treatment for the diagnostic, replacing `acid.cpp:628`**, so the blast-radius census for level 2 can be run *before* level 2 is ever applied:

```cpp
const int nfeas_level = [] {
    const char* e = std::getenv("ACID_NFEAS");
    if (!e) return 0;
    const int v = std::atoi(e);
    return v > 0 ? v : 1;
}();
const bool cav_dbg = nfeas_level > 0;
```

**(3) Hoist the four mask variables out of the retry body and clear them conditionally.**
Move the declarations currently at `acid.cpp:1585-1591` to **just before the time loop** at `acid.cpp:915` (run scope), keeping the existing comment block updated. Then, at the position they used to occupy inside the retry body (i.e. immediately before `auto compute_R = [&]()` at `acid.cpp:1592`), insert:

```cpp
// Round 29: level 1 == the round-28 behaviour EXACTLY -- clearing here at the top of every
// retry body is bit-equivalent to the old in-body `std::vector<char> cav(n, 0); int cav_n = 0;`
// declaration. Level 2 skips the clear, so the mask latches for the whole run. The shadow
// mask follows the HIGHER of the two levels, so ACID_NFEAS=2 can report the level-2 mask
// without the applying flag ever being set (round 28's blast-radius-first discipline).
if (cav_level < 2) { std::fill(cav.begin(), cav.end(), 0); cav_n = 0; }
if (std::max(cav_level, nfeas_level) < 2) {
    std::fill(cav_shadow.begin(), cav_shadow.end(), 0); cav_n_shadow = 0;
}
```

No change is needed at `acid.cpp:1611`, `:1613`, `:2186`, `:2282`, or `:2467-2473` — the consumers and the setter read `cav[]`/`cav_n` and are already correct under either scope.

**(4) Spatial fields on the existing `NFEAS` print (`acid.cpp:2481-2490`), diagnostic-only.**
Inside the already-`cav_dbg`-guarded block, compute and print the geometry of both masks:

```cpp
auto span = [&](const std::vector<char>& m, int cnt, int& lo, int& hi, int& holes) {
    lo = -1; hi = -1; holes = 0;
    for (int i = 0; i < n; ++i) if (m[i]) { if (lo < 0) lo = i; hi = i; }
    if (lo >= 0) holes = (hi - lo + 1) - cnt;
};
int lo=-1, hi=-1, hol=0, slo=-1, shi=-1, shol=0;
span(cav, cav_n, lo, hi, hol);
span(cav_shadow, cav_n_shadow, slo, shi, shol);
```

and append `n=%d lo=%d hi=%d holes=%d slo=%d shi=%d sholes=%d` to the existing format string. `n` disambiguates the N=400 primary from the N=800 reference run inside one `denner1d_dump` invocation; `holes` is the count of cells inside `[lo,hi]` that are *not* masked — the direct test of §2.4's "contiguous, centred" inference and of any spatial hole at the stagnation point.

This is print-only, inside a block that already exists and is already gated. It adds no FP arithmetic to any path and cannot run at all when `ACID_NFEAS` is unset.

### 3.3 Why level 1 remains byte-identical, and why level 2 is inert at step boundaries

* **Level 1**: the conditional clear at the top of the retry body reproduces exactly what a fresh in-body declaration did — same values (`0`), same order, no FP arithmetic. `cav_level == 1` for `ACID_YADV_ALPHA_IMPLICIT_CAV=1` and for any non-numeric presence, so every round-28 measurement remains reproducible verbatim. **G4(a) below verifies this by byte-comparison, not by argument.**
* **Level 2 at step boundaries**: on Newton iteration 0 of every step, `s.p == p_o` and `s.T == T_o` (captured at `acid.cpp:1112`, unmodified by the transport block), and `s.alpha` was just written by `acid.cpp:1343-1349` using the *textually identical expression* at the *identical clamped* `(p_o, T_o)`. So `compute_R`'s re-derivation at `acid.cpp:1615` for a latched cell is a bit-exact identity. **A latch cannot inject a discontinuity at a step boundary; it only changes iterations ≥ 1.** (This holds with `ACID_YADV_F3` and `ACID_YADV_HREINIT` unset, which is the default and the configuration of every gate run.)
* **Legacy 2x2 path** (`ACID_NO_UNIC`): has no mask setter of its own (`acid.cpp:2660-2680`), so the mask stays empty there and level 2 is a no-op, exactly as level 1 is today. No new behaviour is created on a path this suite does not exercise.

### 3.4 Blast-radius analysis — done before proposing, as the rules require

**The bracketing argument survives unchanged**, and it is the strongest safety property this candidate has: the predicate never fires ⇒ identical to plain B (15/19, `{15,24,33,34}`); the predicate fires on every cell ⇒ identical to config C (14/19, `{14,15,24,33,34}`). A latch only moves a config *within* that bracket, toward C. It never touches `Yv`, `s.p`, `s.T`, the Eqs.43-44 old-level rebuild, or the 1.0 Pa floor; it introduces no second alpha; `REBUILD_ADV`'s failure mode (a second, different alpha visible to one consumer and not the others) is structurally impossible here.

**What actually changes, per case**, from round 28 §38.3's own census (`ACID_NFEAS`, plain B, cells the predicate would fire on over the whole run):

```
01=0  02=0  04=0  05=0  07=0  13=0  14=0  15=604  24=9  25=2
26=2  27=1  28=5  30=0  31=0  33=9  34=10  35=0  36=0
```

| case | firing | change under level 2 | risk |
|---|---|---|---|
| 01,02,04,05,07,13,**14**,30,31,35,36 | **0** | **none — byte-identical to plain B** | **zero**. Includes case14, the one case where B's and C's `EXPECTED` fail sets differ, i.e. the only case the CAV family exists to protect. It never fires. |
| 25 | 2 (step 0) | 2 cells stay implicit for the whole run instead of one step | **the one to watch.** case25 passes under B and fails under C. Two latched cells is a long way from C's 400, but this is measured, not argued — the harm gate decides. |
| 26,27,28 | 2,1,5 (step 0) | same, a handful of cells | low. All pass under B; C's own regressions are 13/14/25, not these. |
| 24,33,34 | 9,9,10 (step 0) | same | none that matters: these fail under B *and* C, closed by round 26 as a closure mismatch. Worst case is a metric wobble inside an already-failing case. |
| **15** | 604 | the whole point | this is the target. |

**Pre-registered expectation**: `pass_count = 15/19`, `fail = {15,24,33,34}`, unchanged. Anything else fires S5-early (§6).

**A new blast-radius instrument, available before applying anything**: `ACID_NFEAS=2` with the applying flag *unset* reports the level-2 shadow mask (final latched count, span, holes) on all 19 cases while writing nothing that any consumer reads. This is round 28's "census before application" discipline, applied to the new level. Stage 1 runs it.

### 3.5 What this candidate cannot do — stated before any measurement

`B+CAV` at level 2 converges toward config C on case15. Config C **fails** case15 on `smooth_ok` (`cj = 30.018 > 8.0`). Therefore:

> **A fully successful latch cannot make case15 pass.** Its maximum achievable outcome is: `l2_rho`/`corr_rho`/`l2_p`/`corr_p`/`l2_u`/`corr_u`/`osc_ok` all clear, `smooth_ok` still fails, `pass = false`.

This is written down *now* so that no post-hoc reading of the numbers can present a partial success as a win. It is also the reason this candidate is worth doing at all: it would leave case15 with exactly **one** defect, shared with C, sharply defined, and independent of everything the last three rounds have been working on.

---

## 4. Staging — harm-gate-first, exactly as round 28 did

### Stage 0 — reproduce the diagnosis (no code change)
Run the §9 command block against the existing `HEAD` build. Expected: the §2.1/§2.2/§2.3/§2.4 tables reproduce digit-for-digit. **If any number disagrees, stop and reconcile before writing a line of code** — the whole plan rests on them.

### Stage 1 — diagnostic only: level parse + spatial `NFEAS` (§3.2 edits 1, 2, 4)
Apply edits (1), (2) and (4) *only* — **not** the scope hoist (3). This gives the level parse (a pure no-op at level 1) and the spatial print, with the mask still retry-scoped.

* **G4-early (mandatory, round 28's own bug-catching check)**: with `ACID_NFEAS=1` set, `denner1d_dump 15` and `denner1d_dump 24` stdout must be **byte-identical** to the same commands on the pre-change build. Same for `ACID_YADV_ALPHA_IMPLICIT_CAV=1`. Round 28 caught a silently solution-affecting "diagnostic" flag exactly here; do not skip it.
* **Measurement M1**: `ACID_NFEAS=1` + `ACID_YADV_ALPHA_IMPLICIT_CAV=1` on `dump 15`. Record, per run (`n=400` vs `n=800`) and per step: `cav_n`, `lo`, `hi`, `holes`. **Decision content**: does the mask have interior *holes* (spatial revert at the stagnation point) or is it contiguous with a *shrinking span* (temporal revert)? §2.3's count data already proves a temporal revert; M1 says whether there is also a hole. Either way the latch is the fix; this is confirmation, and it feeds round 30's write-up.
* **Measurement M2 (blast radius for level 2, before level 2 exists)**: apply edit (3) as well *at this point only if* you prefer a single build — otherwise `ACID_NFEAS=2` needs edit (3)'s conditional clear. Cleanest: apply (3) too but keep every *applying* run at level 1, then run `ACID_NFEAS=2` alone (applying flag unset) across all 19 cases and record the final latched shadow count / span per case. **Required check before Stage 2**: case14's latched shadow count must be 0.

### Stage 2 — the applying flag at level 2, **harm gate read FIRST**
Enable `ACID_YADV_ALPHA_IMPLICIT_CAV=2` and, **before looking at a single case15 metric**:

```
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT_CAV=2 ./build-cpp/cpp/denner_1d/denner1d_validate
```

Read **only** `pass_count` and the fail set. Required: `pass_count = 15`, `fail = {15,24,33,34}`. If not → **S5-early**: revert edit (3) (keep Stage 1's diagnostics), write up, done. Do not "investigate whether the regression is real" first; round 27's `REBUILD_ADV` is the precedent.

Only after the harm gate is clean:

* **R1** `python3 scripts/yadv_r27_case15.py overlays` (add a `B+CAV=2` row) plus the full-gate breakdown of §2.1 for `B+CAV=2` against its own reference **and** against C's reference.
* **R2** `CAV2_400 vs C400` and `CAV2ref vs Cref` cross-comparison (§2.2's four numbers).
* **R3** the `p`-notch check: `max |p_CAV2 − p_C|` and its location; `p_osc`.
* **R4** `ACID_NFEAS=1` per-step `cav_n`/`lo`/`hi` under level 2 — must now be **monotone non-decreasing** in `cav_n` by construction; verify.
* **R5** `ACID_MBAL` over the run: `|Σadv+Σremap|/|Σremap|` (round 28's R3 instrument) must not regress from 1.6%.

### Stage 3 — write-up
`docs/YADV_RESEARCH.md` §39, `docs/YADV_ROADMAP.md` control state, this plan committed as `docs/YADV_ROUND_29_PLAN.md`. The write-up must lead with §2's reference-artifact finding and with the correction to round 28 §38.5's "fails on `l2_rho`/`corr_rho` alone", regardless of how Stage 2 turns out.

### Stage 4 — OPTIONAL, only if Stages 0-3 finish cleanly and cheaply
Add an optional cells-override to `cpp/denner_1d/apps/denner1d_dump.cpp` (`argv[2]`, applied to a local copy of the `CaseDefinition`; absent ⇒ textually the current behaviour; `denner1d_validate` and `cases.cpp` untouched). Purpose: measure `cj` on the N=800 grid *at its own resolution* rather than through the reference's linear interpolation, and so answer round 30's gating question — **is the stagnation-point core jet converging under refinement, or structural?** Warn in the commit message that with `cells ≥ 800` the `*_ref` columns become degenerate (`computed_reference` takes `max(cells, 800)`), so no `l2_*`/`corr_*` from such a run may be quoted — round 28's own degeneracy lesson.
*If a reference-resolution override is wanted instead, note that `validation/1D/15_E_Cavitation.md` itself names `DENNER_CASE15_REF_N=800` as the intended knob (never implemented in this solver) — that is the explicit spec-level justification the `cases.cpp` rule requires. **Still: do not do this unless Stage 2 has already landed cleanly.***

---

## 5. Gates and targets

### Hard gates (all must hold before merge; non-negotiable)

| | Gate | Command | Requirement |
|---|---|---|---|
| **G1** | OFF-path byte identity | `python3 scripts/yadv_r9_sweep.py --verify` | `VERIFY OK` — all 9 `VERIFY_CASES` byte-identical to the published `solver_denner` binary, and case01 `ACID_YADV=1` ≡ unset |
| **G2** | 7-config battery | `python3 scripts/yadv_r9_sweep.py --sweep` | `ALL GATES OK` — every config matches `EXPECTED` (A 19/∅, B 15/`{15,24,33,34}`, C 14/`{14,15,24,33,34}`, D, E, F, G) |
| **G3** | unit tests | `./build-cpp/cpp/denner_1d/denner1d_unit` | all pass, numbers unchanged |
| **G4(a)** | new-level no-op | `dump 15`, `dump 24`, `dump 25` under `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT_CAV=1` | **byte-identical** to the pre-change build |
| **G4(b)** | diagnostic no-op | same three cases under `ACID_NFEAS=1` and under `ACID_NFEAS=2` (applying flag unset) | **byte-identical** stdout to the pre-change build |
| **G4(c)** | default no-op | `dump 15` with **no** `ACID_*` set | byte-identical |
| **G5** | diff hygiene | `git diff --stat` | only `acid.cpp` (+ `docs/`, `scripts/`, and Stage-4's `denner1d_dump.cpp` if taken). **No** `cases.cpp`, **no** `validation.cpp`, no new constants, no per-case coefficients, no touching the 1.0 Pa floor |

Add `"ACID_YADV_ALPHA_IMPLICIT_CAV"` — already present — and verify `ACID_NFEAS` is in the purge lists of **both** `scripts/yadv_r9_sweep.py` (`ACID_ENV_VARS`) and `scripts/yadv_r27_case15.py` (they are; no change needed, but confirm before quoting any number).

### Round-specific targets, pre-registered in priority order

| | Target | Threshold | Predicted value under level 2 |
|---|---|---|---|
| **T0** | Harm gate under level 2, read before any case15 metric | `pass_count = 15`, `fail = {15,24,33,34}` | as stated |
| **T1** | `cav_n` monotone non-decreasing per step under level 2 | no decrease anywhere in case15's run | structurally guaranteed; verify anyway |
| **T2** | The mask-boundary pressure notch is removed | `max\|p_CAV2 − p_C\| < 2000 Pa` (from **28 396 Pa**) | ~0 |
| **T3** | `osc_ok` recovers | `p_osc < 0.02` (from **0.4476**) | ~0.00 |
| **T4** | The N=800 reference stops plugging | `min(rho_ref) < 20` (from **124.3**; C's is **7.515**) | ~7-10 |
| **T5** | `l2_rho`/`corr_rho` against its **own** reference | `l2_rho ≤ 0.03`, `corr_rho ≥ 0.99` (from **0.06898** / **0.957806**) | ~0.020 / ~0.997 |
| **T6** | Mass/conservation not regressed | `M ≥ 800`, `nfloor ≤ 5`, `ACID_MBAL` residual `< 5%` | 869-871 / 0 / ~1.6% |
| **T7** | `smooth_ok` — **predicted to STILL FAIL** | `cj` expected in **[26, 31]**, limit `8.0` | fail. **If `cj < 8` this is a surprise and case15 passes → S1.** |

### Execution order (strict)

`Stage 0 reproduce` → `G1` → `G2` (both on the unmodified build, to establish the baseline) → Stage 1 edits → `G4(a)/(b)/(c)` → `M1`, `M2` → Stage 2 edit (3) → **T0 harm gate** → `G1`, `G2`, `G3` → `T1…T7` → Stage 3 write-up → (optional Stage 4).

Rationale for putting G1/G2 *before* Stage 1's edits: they take the longest and a stale baseline invalidates every later comparison. Rationale for T0 before T1-T7: round 27's lesson, applied at the same point round 28 applied it.

---

## 6. Pre-registered stop / decision rules

Decided now, before any Stage-2 number exists. The round's verdict is whichever rule fires first, read in order.

* **S1 — case15 PASSES.** T0 clean **and** `denner1d_validate --only 15` reports `pass:true` under `B+CAV=2`. Would require T7 to be violated (`cj < 8`), which this plan predicts will not happen. Action: keep level 2 gated OFF (promotion is still a separate decision, §7), `consecutive_failures = 0`, escalate the promotion question to the user with the full 19-case table.
* **S2 — Full mechanical success, gate still fails (the predicted outcome).** T0 clean, T1-T6 all hold, T7 fails as predicted. Verdict: **S4-class partial**, matching round 28's own precedent. Keep level 2 as gated-off research infrastructure. `consecutive_failures` **NOT** incremented — the round produced a decisive, verified mechanism plus a measured correction to round 28's framing, i.e. real progress. Round 30's target becomes the stagnation-point core jet, with `B+CAV=2` established as the config in which to attack it (C-grade accuracy without C's case13/14/25 cost).
* **S3 — Partial mechanical success.** T0 clean, but some of T2-T5 hold and others do not (e.g. the notch goes but the N=800 plug stays). Verdict: partial. Report exactly which held and which did not, with the numbers, **without** re-deriving a new mechanism to explain the miss inside this round. Keep level 2 gated off. `consecutive_failures` unchanged. Name the surviving anomaly as round 30's first question.
* **S4 — Neutral / no effect.** T0 clean but T2-T5 essentially unchanged from level 1. Verdict: the latch is not the mechanism; §2's diagnosis stands but its causal attribution to mask timing is **weakened and must be reported as such**. Revert edit (3), keep Stage 1's diagnostics (they are the round's durable output). `consecutive_failures` unchanged (the diagnostic finding is measured progress on its own).
* **S5 — Harm (any fail-set change at T0, or any hard gate G1-G5 failing).** Revert edit (3) **in full, immediately**, keep only Stage 1's diagnostics. `consecutive_failures` **INCREMENTED to 2**. Write it up as a clean negative result, exactly as round 27 did. Do not attempt a second candidate in the same round.
* **S6 — The Stage-0 reproduction fails.** Any §2 number that does not reproduce on the implementer's build: **stop, write up the discrepancy, change no solver code.** The whole plan is downstream of those numbers.

**Anti-rescue clause (binding).**
1. §3.5's statement — a fully successful latch cannot make case15 pass — is fixed. If case15 still fails, that is **S2, not a failure of the round**, and it must not be re-labelled as anything better.
2. Conversely, no metric may be re-scored, re-normalised, or compared against a different config's reference *in the verdict*. The §2.1 cross-reference rows are a **diagnostic instrument**, deliberately reported as such; `B+CAV`'s gate result is and remains its score against **its own** reference. Presenting `l2_rho = 0.01972` (against C's reference) as "`B+CAV` passes the density gate" is explicitly forbidden.
3. If T0 fails, no case15 metric may be read, quoted, or used to argue for keeping the change. S5 fires on the fail set alone.
4. Any `l2_p = 0.00000` / `corr_p = 1.000000` / `l2_*` computed against a floored or plugged reference must be flagged **degenerate** at the point of quoting, per round 28 §38.2's correction and §2.7 above.
5. `consecutive_failures` moves only as S1-S6 specify. It is not a judgement call.

---

## 7. Non-goals — explicitly not attempted this round

* **Making case15 pass.** §3.5: measured to be out of reach of this candidate family. This round's ceiling is config-C-grade accuracy with case14 preserved.
* **The stagnation-point core jet / `cj` / `mj` / `cc` defect** (round 27 §4.5). It is the binding blocker, it is shared with config C, it is independent of `cav[]`, and it needs its own round. This plan characterises it (§2.5) and seeds its literature (§8); it does not attempt it.
* **Touching the 1.0 Pa pressure floor.** Forbidden by rounds 27/28's own non-goals; the reasoning still holds (it is the trigger, not the sink).
* **Promoting `ACID_YADV_ALPHA_IMPLICIT_CAV` (any level) into `ACID_YADV`'s default path**, or changing `ACID_YADV`'s recommended status. Explicit non-goal carried forward from round 28.
* **Adopting config C wholesale.** It costs case14 (the only case where B's and C's `EXPECTED` fail sets differ) and it has the same `cj` defect anyway — measured here at `30.018`, worse than `B+CAV`'s `25.906`.
* **Editing `validation.cpp` or `cases.cpp`**, or the case15 spec, or its gate thresholds. §2.7's finding about config A's degenerate pass is recorded for escalation, not acted on. (The one exception the rules would permit — a spec-named `DENNER_CASE15_REF_N` — is deferred to the optional Stage 4 and is not on the critical path.)
* **Reopening cases 24/33/34.** Closed by round 26 as a closure mismatch. They appear here only as blast-radius rows.
* **Any THINC / rho-monotonicity-guard work.** Excluded on live measurement (§2.6): THINC records `activations=0` on case15 at both resolutions, and cannot activate on a uniform colour function by construction.
* **A new predicate.** The existing predicate at `acid.cpp:2470` is unchanged; only the mask's *lifetime* changes. No new constant is introduced anywhere.

---

## 8. Literature

**Already in the corpus, full text, directly on point — read this round:**

* **`papers/md/33_saurel_relaxation_multiphase.md`** — Saurel, Petitpas & Berry, *"Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures"*, JCP **228** (2009) 1678-1712.
 * **§4.5 "Cavitation test"** (md lines ~1337-1345) is **literally case15**: a 1 m tube of water at atmospheric pressure with `alpha_air = 10⁻²` everywhere, a velocity discontinuity at `x = 0.5` with `u = ∓100 m/s`, solved to `t = 1.85 ms` on 1000 cells. Its stated mechanism — *"As gas is present, the pressure cannot become negative. To maintain positive pressure, the gas volume fraction increases and creates a cavitation pocket. This results in the dynamic appearance of two interfaces that were not present initially"* — is precisely what the implicit-alpha family reproduces (`alpha: 0.055 → 0.9994`) and what the frozen-alpha closure structurally cannot (§2.7). It independently validates round 28's mechanism and this round's §2.5 framing.
 * **Appendix A** (md line ~1681) states the modelling assumption that bears directly on §3.1: *"cavitation is modeled as a mechanical relaxation process, occurring at infinite rate, and not as a mass transfer process."* Infinite-rate relaxation is applied **uniformly to every cell** — which is config C's own per-cell update, and which is what a latched mask converges to. There is no precedent in this literature for a *per-step-re-earned* relaxation region; the retry-scoped mask is the anomaly, and the latch moves toward the standard formulation.
 * Same appendix notes the Wood sound speed's non-monotonic behaviour *"causes computational difficulties"* — a relevant thread for round 30's core-jet work.
* **`papers/md/73.md`** — Maltsev, Skote & Tsoutsanis, *"High-order methods for diffuse-interface models in compressible multi-medium flows: A review"*, Phys. Fluids **34**, 021301 (2022). Its bibliography (ref. 25, md line 4549) is the entry point to the interface-**overheating** literature that round 30 will need.

**Searched and found nothing usable externally.** arXiv and Semantic Scholar queries on the two mechanisms this round implicates ("switching implicit/explicit activation criterion multiphase", "expansion tube cavitation double rarefaction numerical artifact", "overheating material interface removal") returned no relevant hits. The corpus's own Saurel Appendix A gives the stronger and more direct argument, and is used instead.

**`{slug}_needed.md` candidates** — record these for round 30's core-jet thread; none could be fetched this round:

* `bigdelou_2021_overheating_ghostfluid_needed.md` — P. Bigdelou, C. Liu, P. Tarey, P. Ramaprabhu, *"An efficient ghost fluid method to remove overheating from material interfaces in compressible multi-medium flows"*, Computers & Fluids **233**, 105250 (2021). DOI `10.1016/j.compfluid.2021.105250`. *Why: the canonical modern treatment of the overheating class that case15's stagnation-point core jet belongs to.*
* `petitpas_2007_relaxation_projection_II_artificial_heat_needed.md` — F. Petitpas, E. Franquet, R. Saurel, O. Le Métayer, *"A relaxation-projection method for compressible flows. Part II: Artificial heat exchanges for multiphase shocks"*, JCP **225** (2007) 2214-2248. DOI `10.1016/j.jcp.2007.03.014`. *Why: same authors, same model family as the case15 test itself; treats exactly the spurious-heating pathology in a diffuse-interface multiphase setting.*
* `noh_1987_artificial_heat_flux_needed.md` — W. F. Noh, *"Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux"*, JCP **72** (1987) 78-120. DOI `10.1016/0021-9991(87)90074-X`. *Why: the origin of the stagnation-point overheating diagnosis; case15's core jet is its expansion-side analogue.*

---

## 9. Reproducing the diagnosis (exact commands — every §2 number came from these)

```bash
cd /home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
D=./build-cpp/cpp/denner_1d/denner1d_dump

python3 scripts/yadv_r27_case15.py overlays
#   expect  C: l2_rho=0.01966 corr_rho=0.996734 ; B+CAV: l2_rho=0.06898 corr_rho=0.957806

DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT_CAV=1 ACID_NFEAS=1 $D 15 2>&1 >/dev/null \
  | grep '^NFEAS'
#   split into 2 runs where `step` resets: run0 = N=400 (85 steps), run1 = N=800 (169 steps)
#   expect per-step max cav_n: 70 62 56 52 48 44 42 42 40 42 44 ... IDENTICAL in both runs

DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT_CAV=1 ACID_THINC_DBG=1 $D 15 2>&1 >/dev/null
#   expect exactly 2 lines, both "THINC case=15 activations=0 rho_guard_rejects=0"

DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT_CAV=1 \
  ./build-cpp/cpp/denner_1d/denner1d_validate 2>/dev/null | tail -1
#   expect DENNER1D_CPP_METRIC pass_count=15 total=19
```
