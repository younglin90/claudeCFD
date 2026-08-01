# Phase 2 — analytic `d(alpha)/dp`, `d(alpha)/dT` for the implicit-alpha Y path

**Status: PLAN ONLY, not started.** Produced by a Plan (opus) agent after reading
`docs/YADV_RESEARCH.md` rounds 1-4, `.claude/rules/denner-pitfalls.md`, `cpp/denner_1d/src/acid.cpp`,
`eos.hpp`, and four papers gathered this session (see §"Literature" below). Saved by the Advisor
session for review before any implementation begins.

**Goal.** Make `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` converge under the DEFAULT analytic Jacobian
at least as well as round 4's FD-Jacobian result: recover 13/15/25 (and if possible 14), keep
01/02/04/05/07/26/27/28/30/31/35/36, do not regress the OFF path or plain `ACID_YADV=1`. Target
`pass_count >= 15`, with case15's `amp_ratio_p -> ~1.0` (§15.3's FD row) reproduced with the
analytic Jacobian.

## Literature grounding (gathered this session, see `/home/younglin90/work/claude_code/claudeCFD/papers/`)

- `03_denner_2018_coupled_compressible_summary.md` — Denner, *Comput. Fluids* 175 (2018), DOI
  10.1016/j.compfluid.2018.07.005. The general Newton product-rule linearisation template for a
  coefficient*variable term, already the pattern this solver's existing analytic Jacobian follows.
- `02_janodet_2025_coupled_large_density_summary.md` — Janodet, van Wachem, Denner, *JCP* 520
  (2025). Closest structural precedent: brings a previously-external colour/interface function
  into a coupled Newton system (incompressible, constant density jump — no p,T dependence, so the
  formula doesn't transfer, but the convergence STRATEGY — what's Newton-linearised vs
  Picard-lagged — does).
- `85_denner_evrard_vanwachem_2020_unified_thermo_closure_summary.md` — Denner, Evrard, van
  Wachem, *JCP* 409 (2020), arXiv:2002.10482 (new summary written this session). "Barotropic
  substitution": the Jacobian entry must differentiate the SAME nonlinear map the residual
  evaluates, not a separately-linearised surrogate. Directly explains round 4's regression
  (residual re-evaluates alpha fully nonlinearly; Jacobian implicitly assumed `d(alpha)/dp=0` —
  a family mismatch).
- `1_denner.md` — Denner, Xiao, van Wachem, the original ACID paper. Confirms the historical
  baseline: alpha/VOF solved in a separate linear system, fully decoupled from the coupled Newton
  solve — why the existing analytic Jacobian has no `d(alpha)/dp` term at all.

No paywalled/unreachable papers were found this round — everything relevant had an open-access
copy.

---

## 0. Facts established while planning (implementers: do not re-derive)

**0.1 Every one of the 19 cases runs the COUPLED 3x3 `(u,p,h)` path.** `cases.cpp:28` sets
`c.unic = true` for every case, and `acid.cpp:429` `coupled = unic ? true : ...`. The segregated
2x2 analytic Jacobian (`acid.cpp:1773-1811`) and its post-Newton enthalpy->T block are **dead code
on the default path**. Phase 2 touches only the coupled analytic block, `acid.cpp:1487-1665`.

**0.2 The 5 acoustic cases never see the analytic Jacobian.** `bdf2 = unic ? acoustic_src`,
`tr_bdf2 = bdf2 && coupled`, and `acid.cpp:1274` `if (tr_bdf2) ajac = false;`. So 04/05/07/35/36
(inlet_frequency > 0) already run the FD Jacobian. **They must be byte-identical across every
stage** — a free, sharp regression gate.

**0.3 `alpha_i` depends only on cell `i`'s own `(p_i, T_i)`.** Every new Jacobian term is
therefore a **diagonal (`aB[i]`) contribution**. No stencil growth, no change to the
tridiagonal/pentadiagonal structure, no new solver code.

**0.4 The residual's alpha is lagged in `T` by exactly one `compute_R` call.** `acid.cpp:1014-1022`
runs *before* the h->T inversion at 1026-1038, so `alpha_i = A(Y_i, p_i^{current}, T_i^{previous
call})`. Consequence: the exact derivative of the map **as coded** has `d(alpha)/dp = a_p` (at
frozen `T`) and `d(alpha)/dh = d(alpha)/du = 0`. This makes a p-only Jacobian stage *exactly*
consistent with the current residual — no residual edit needed (see §3).

**0.5 `d(alpha)/dT|_Y` is identically ZERO when both phases have NASG covolume `b = 0`.** From
`eos.cpp:36`, `phi_k/rho_k = -kv_k(gamma_k-1)/A_k` with `A_k = kv_k(gamma_k-1)T + b_k(p+pinf_k)`;
for `b=0` this is `-1/T` for *every* phase, so the phase difference vanishes exactly. Only
`water_liquid_phase()` has `b = 6.61e-4`; air and vapour have `b = 0`. So the T-pathway is exactly
zero for gas-gas cases and small (est. `a_T ~ 1.2e-4 /K` at case15's state) for air/water. **This
is why the T-pathway is staged last.**

---

## 1. The derivative chain (worked out, code-ready)

With `alpha = Y*rb / (ra*(1-Y) + Y*rb)`, `D = ra(1-Y) + Y rb`, and the identities `alpha = Y
rb/D`, `1-alpha = ra(1-Y)/D`:

```
  d(alpha)/d(ra) = -Y rb (1-Y) / D^2 = -alpha(1-alpha)/ra
  d(alpha)/d(rb) = +Y ra (1-Y) / D^2 = +alpha(1-alpha)/rb
```

Chaining through `ra = rho_a(p,T)`, `rb = rho_b(p,T)` with the EXISTING `PhaseProps` partials
`zeta = drho/dp|_T`, `phi = drho/dT|_p`:

```
  a_p := d(alpha)/dp|_{T,Y} = alpha(1-alpha) * ( zeta_b/rho_b - zeta_a/rho_a )
  a_T := d(alpha)/dT|_{p,Y} = alpha(1-alpha) * ( phi_b /rho_b - phi_a /rho_a )
```

Properties worth knowing:
- `zeta_k/rho_k` is the isothermal compressibility (`= 1/p` for an ideal gas), `-phi_k/rho_k` the
  thermal expansivity. `a_p < 0` for a gas-in-liquid mixture: compress -> gas volume fraction
  falls. Case15 (`p` falling) -> `alpha` rises, exactly the measured 0.055 -> 0.99.
- The `alpha(1-alpha)` prefactor vanishes at both pure ends, so the derivative is **automatically
  consistent with the `clamp(alpha,0,1)`** in the residual. No kink handling, no epsilon, no new
  constant.
- `a_T == 0` exactly when both phases have `b = 0` (§0.5).

**Cross-check (do this in the unit test).** The total mixture compressibility implied by the
chain must equal the closed form of the specific-volume blend `1/rho = Y/ra + (1-Y)/rb`:

```
  D_p + (ra - rb)*a_p  ==  rho * ( alpha*zeta_a/ra + (1-alpha)*zeta_b/rb )
  where D_p = alpha*zeta_a + (1-alpha)*zeta_b   (the FROZEN-alpha value the code uses today)
```
This is an exact algebraic identity (verified by hand during planning). The RHS is the
(isothermal) Wood-type mixture compressibility; the LHS's second term is precisely what the
current Jacobian is missing.

**Predicted magnitude of the defect (verify, do not assume).** At case15's state (air
`alpha=0.055` in water, `p=1e5`, `T=300`, `rho=945`):

| quantity | value |
|---|---|
| `zeta_a/ra = 1/p` | 1.0e-5 |
| `zeta_b/rb` | ~4.3e-10 |
| `a_p` | ~-5.2e-7 /Pa |
| `D_p` (what the Jacobian uses now) | ~1.05e-6 |
| `D_p + (ra-rb)*a_p` (correct) | ~5.2e-4 |
| **ratio** | **~500x** |

The same computation at a genuinely mixed air/water interface cell (`alpha=0.5`, `p=1e5`) gives
~400x. So the analytic Jacobian's continuity-transient diagonal is wrong by 2-3 orders of
magnitude in exactly the cells that dominate 13/15/25 — the quantitative explanation of §15.4. In
pure cells (`alpha -> 0` or `1`) the term vanishes, which is why 01/02/26/27/28 are untouched.

---

## 2. Where each term enters

### 2.1 Residual sites that read `s.alpha` (complete enumeration, verified by grep)

| # | site | line | needs a Jacobian term? |
|---|---|---|---|
| R1 | `eval_thermo`: `rho`, `hstat`, `cp` blends | 311-326 | YES — augment the existing cell EOS chain |
| R2 | `T_from_hstat(..., s.alpha[i], ...)` h->T inversion | 1033 | YES — same chain as R1, no separate machinery |
| R3 | ACID per-cell mass flux `mdotL/mdotR` (Eqs.41-42) | 1172-1177 | YES — NEW blend-weight term |
| R4 | coupled energy flux blend `ai*rHaup + (1-ai)*rHbup` | 1204-1206 | YES — NEW blend-weight term |
| R5 | `use4` phase test `s.alpha[i] >= 0.5` | 1096 | NO — discrete switch, correctly frozen (existing style) |
| R6 | sound speed `s.a` -> MWI clamp bound `af` | 325, 1117 | NO — clamp/bound frozen (existing style) |
| R7 | `s.drhodp` | 326 | NO — consumed only by the dead segregated Jacobian (§0.1). Non-goal. |
| R8 | pre-Newton transport / recovery / `rho_o` baseline | 661-893 | NO — outside the Newton |

**R2 needs no separate machinery.** The h->T inversion solves `hstat_mix(p, T, alpha) = h -
u^2/2`. Once `alpha = A(p,T)` the composed function is `H(p,T) = hstat_mix(p,T,A(p,T))`, and its
partials are obtained by simply *starring* the four quantities the existing code already builds.
This is the "circular dependency" the round-4 design note flagged; it collapses to two extra
addends.

### 2.2 Jacobian edits

**J1 — the cell EOS chain, `acid.cpp:1503-1515` (AUGMENT existing, no new entries).**

Existing: `D`, `D_T`, `D_p`, `N`, `N_T`, `N_p`, `hsT`, `hsp`, `dTh/dTu/dTp`, `drh/dru/drp`.
Replace the p-partials (Stage 1) and later the T-partials (Stage 3) with starred versions:

```
  ap  = al*(1-al) * (pb.zeta/pb.rho - pa.zeta/pa.rho);      // 0 if !yadv || !alpha_implicit
  aT  = al*(1-al) * (pb.phi /pb.rho - pa.phi /pa.rho);      // Stage 3 only; == 0 when both b==0
  D_p* = D_p + (pa.rho - pb.rho) * ap;
  N_p* = N_p + (pa.rho*pa.h - pb.rho*pb.h) * ap;
  D_T* = D_T + (pa.rho - pb.rho) * aT;                      // Stage 3
  N_T* = N_T + (pa.rho*pa.h - pb.rho*pb.h) * aT;            // Stage 3
  hsp* = (N_p* * D - N * D_p*) / (D*D);
  hsT* = (N_T* * D - N * D_T*) / (D*D);                     // Stage 3 (== hsT at Stage 1)
  dTh = 1/hsT*;  dTu = -u/hsT*;  dTp = -hsp*/hsT*;
  drh = D_T* * dTh;  dru = D_T* * dTu;  drp = D_p* + D_T* * dTp;
```

This single edit propagates automatically to **three** consumers, with no further code:
- the transient diagonal block (1516-1521) — the 500x term of §1 lands here;
- the frozen-MWI `d(theta)/d(rho)` block (1600-1632), which chains through `dru/drp/drh` of the
  neighbours;
- the upwind-transport block (1567-1595), which chains `d(raup)/dp = zeta + phi*dTp[uw]` through
  the starred `dTp`.

Also store per cell, for J2:
```
  alp_p[i] = ap;                                  // Stage 1/2
  alp_p[i] = ap + aT*dTp[i]; alp_h[i] = aT*dTh[i]; alp_u[i] = aT*dTu[i];   // Stage 3 (total derivative)
```

**J2 — the ACID flux blend weight (NEW diagonal addends).** New, non-overlapping with anything
present: the flux-coupling block (1544-1562) freezes `rblL/rblR` entirely, and the
upwind-transport block (1567-1595) differentiates `raup/rbup` at the *upwind* cell column while
holding `al` constant. Both addends of the product rule `d(al*ra + (1-al)*rb)` are needed — this
is exactly Denner (2018) Eq.1-2's coefficient*variable Newton template, both factors linearised.

```
for (int i = 0; i < n; ++i) {
    const double dR = (raup[i+1] - rbup[i+1]) * theta[i+1];
    const double dL = (raup[i]   - rbup[i]  ) * theta[i];
    const double eR = (rHaup[i+1]- rHbup[i+1]) * theta[i+1];
    const double eL = (rHaup[i]  - rHbup[i]  ) * theta[i];
    add(i, i, 1, 1, (dR - dL) * alp_p[i]);                              // R_con d/dp
    add(i, i, 0, 1, (dR*uconv[i+1] - dL*uconv[i]) * alp_p[i]);          // R_mom d/dp
    add(i, i, 2, 1, (eR - eL) * alp_p[i]);                              // R_ene d/dp
    // Stage 3 adds the same three with alp_h[i] -> column 2 and alp_u[i] -> column 0
}
```

Boundary correctness: `theta[]` already carries every BC override (`theta[0]=uin` for inlet, `0`
for reflective, set at 1164-1166), and `mdotL[0]`/`mdotR[n-1]` use exactly those values — so plain
`theta[f]` is the right factor with no special-casing. Under TR-BDF2 these flux rows would need
the `flux_w` scaling, but `tr_bdf2 => ajac=false`, so guard with a comment, not code.

**J3 — nothing else.** No new off-diagonal entries, no change to `block_thomas3`/`block_penta`,
no change to `compute_R` (Stages 1-2).

---

## 3. Barotropic substitution vs Taylor derivative — the decision

**Decision: the Jacobian term must be the EXACT analytic derivative of the identical nonlinear
map `compute_R` evaluates, at the current iterate. Not a Taylor/secant surrogate, and not a
second, differently-linearised substitution.**

Reasoning, addressing paper 85 (Denner/Evrard/van Wachem, JCP 409, 2020) directly:

1. **Round 4 is already the barotropic-substitution design, half-done.** Paper 85's Eq.43
   substitutes the *full* nonlinear EOS at the trial `p^{(n+1)}` with `T` frozen at level `n`.
   Round 4's `compute_R` does precisely that for `alpha`: `alpha_from_mass_fraction(Y,
   rho_a(p^{trial}, T^{(n)}), rho_b(p^{trial}, T^{(n)}))` — full nonlinear rational map in `p`, `T`
   lagged (§0.4). The paper's own prescription is then to obtain the Jacobian entry by
   **differentiating that substitution formula**. That is what §1 does. The two are the *same*
   linearisation family by construction, which is the entire point of the paper's lesson.
2. **Round 4's regression IS the family mismatch the paper warns about.** Today the residual is
   fully nonlinear in `alpha(p,T)` while the Jacobian implicitly asserts `d(alpha)/dp = 0` — a
   zeroth-order/fixed-coefficient family (Denner 2018 Eq.1) paired with a fully-substituted
   residual. §1's measured 400-500x diagonal error is the quantitative form of that mismatch, and
   §15.4's "textbook bad-search-direction symptom" is its signature. The fix is not to weaken the
   residual back to a Taylor form; it is to promote the Jacobian into the residual's family.
3. **A truncated Taylor `alpha^{(n+1)} ~= alpha^{(n)} + a_p*dp` in the RESIDUAL is explicitly
   rejected.** It would be a *different* residual (round 4's key win — case15's `amp_ratio_p 0.33
   -> 1.00` under FD — comes from `alpha`'s genuinely nonlinear excursion 0.055 -> 0.99 as `p`
   falls, which a linear extrapolation cannot represent; it would also break the `[0,1]` bound).
   The Taylor form appears only inside the Jacobian, where it is *by definition* the correct
   object.
4. **The FD experiment is the existence proof and the verification oracle.** `ACID_NO_AJAC=1`
   differentiates `compute_R` exactly, and §15.3/§15.4 show it recovers 13/15/25. §1's analytic
   terms must therefore *reproduce the FD blocks*, which is directly checkable with the existing
   `ACID_AJAC_BLK` diagnostic. If they do not, the derivation or the sign is wrong — a falsifiable
   gate at every stage.
5. **No damping constant, ever.** `a_p` is enormous in the cavitation regime, and the temptation
   will be to scale it. The project bans tuned constants (`dhat_scale` precedent in
   `denner-pitfalls.md`). If a stage produces overshoot, use the existing structural globalisation
   (backtracking line search, keep-best + stall-break, penta solve, cfl retry) or revert the stage
   — do not add a knob.

**One caveat handled by staging, not by fudging.** Because of the one-call `T` lag (§0.4),
`compute_R` is not strictly a pure function of `(u,p,h,Y)` — it carries `s.T` as hidden state. The
exact derivative of the map *as coded* is the frozen-`T` one (`a_p` only, `a_T` pathway = 0); the
derivative of the *fixed point* is the total one. **Stage 1/2 implement the former (exactly
consistent, zero residual edits); Stage 3 implements the latter and is contingent on
measurement.** Since `a_T == 0` for every `b=0` phase pair (§0.5), the two coincide exactly for
the gas-gas cases and differ by an estimated ~13% of one Jacobian entry for air/water. This also
predicts the alpha<->T Picard loop gain `|hs_al * a_T / hsT|` is negligible (rough estimate at
case15's state: ~3e-6) — **to be measured in Stage 0, not assumed**, because if it is not
negligible it is a separate root cause for case14 and changes the plan.

---

## 4. Staging — Worker briefs

Every stage: clean rebuild -> unit test -> OFF-path byte-identity gate -> plain-ON byte-identity
gate -> full ON+IMPLICIT sweep -> report to Advisor -> Advisor verifies the diff and re-runs the
sweep independently before the next stage starts. All new code lives inside `if (yadv &&
alpha_implicit)` so the OFF path and plain `ACID_YADV=1` are structurally unreachable — but still
verify them empirically, per project rigor.

### Standard measurement protocol (identical every stage)

```bash
bash scripts/yadv_r3_build.sh                 # rm -rf build-cpp, reconfigure, build, unit test
DENNER_ACID=1                                            $V   # gate: 19/19
python3 scripts/yadv_verify.py                                # gate: 9/9 BYTE-IDENTICAL vs published binary
DENNER_ACID=1 ACID_YADV=1                                $V   # gate: 15/19, dumps bit-identical to previous build
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1     $V   # THE measurement
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1 $V   # INVARIANT gate, see below
```

**The FD-invariance gate (new, and the sharpest one available).** Stages 0-2 touch *only* the
analytic-Jacobian block. The FD path never reads that block, so `ACID_NO_AJAC=1` results must be
**bit-identical to the previous build** (12/19, same failure set 14/15/24/27/28/33/34). Any change
there means the Worker accidentally perturbed `compute_R` or shared state. Stage 3 (if it edits
`compute_R`) must declare this gate flipped in advance.

**Acoustic byte-identity gate.** Cases 04/05/07/35/36 run TR-BDF2 -> FD Jacobian (§0.2), so their
ON+IMPLICIT dumps must be byte-identical across Stages 0-2.

---

### Stage 0 — derivation, unit test, and diagnosis. No functional change.

Deliverables:
1. Add to `eos.hpp`, header-inline beside `alpha_from_mass_fraction`, with the §1 derivation in
   comments: `dalpha_dp_massfrac(alpha, zeta_a, rho_a, zeta_b, rho_b)` and
   `dalpha_dT_massfrac(alpha, phi_a, rho_a, phi_b, rho_b)` (or one small struct returning both).
   Header-only, zero call sites this stage.
2. Extend `tests/denner1d_unit.cpp` (reuse the existing `(p,T,Y,pair)` grid from the round-trip
   test):
   - `a_p`, `a_T` vs central FD of `alpha_from_mass_fraction` composed with `phase_props`, rel.
     tol ~1e-6;
   - the §1 identity `D_p + (ra-rb)*a_p == rho*(alpha*zeta_a/ra + (1-alpha)*zeta_b/rb)` to ~1e-12
     rel;
   - the exactness claim `a_T == 0` for `b_a == b_b == 0` (air|vapor pair) — assert *exact* zero;
   - endpoint behaviour `alpha in {0,1} => a_p == a_T == 0` exactly.
3. Measurement only, no code path change: with `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1
   ACID_AJAC_BLK=1 ACID_BLK_STEP=<a few>` on cases **13, 15, 25** (and 14 for reference), record
   the per-block analytic-vs-FD `rel` table. Record the **before** numbers the later stages must
   move.
4. Measure the Picard loop gain `g_i = |hs_al * a_T / hsT|` and `max_i g_i` per case (a temporary
   `stderr` print behind a default-off env var is acceptable; it must be removed or left inert
   before commit).
5. Verify the §1 numeric prediction (`~500x` on case15's continuity diagonal; `~400x` at a mixed
   interface cell of 13/25).

Advisor gate: unit test green; OFF 19/19 + 9/9; plain-ON 15/19 bit-identical; ON+IMPLICIT still
exactly 12/19 with the same failure set (nothing functional changed). If the ajblk table does
**not** show the continuity diagonal as the dominant discrepancy on 13/15/25, **stop and
re-plan** — the §1 diagnosis would be wrong.

### Stage 1 — the p-pathway in the cell EOS chain (J1, `D_p*`/`N_p*` only).

Smallest possible functional increment: ~8 lines inside the existing per-cell loop at 1503-1515,
plus filling `alp_p[]` (unused this stage). `a_T` computed but multiplied by nothing.

Expected: **case15 recovers** (this is where the 500x diagonal lives); 13/25 improve, possibly
recover; 24/33/34 unmoved (they are a conservation defect, §15.5); 14 unknown.
Success bar: `pass_count >= 13` and case15's `amp_ratio_p` moving from 1.232 toward ~1.0 with
`corr_p` back above 0.93 under the **analytic** Jacobian. Post-stage ajblk on 13/15/25 must show
the continuity/energy diagonal `rel` dropping by orders of magnitude.
Revert trigger: `pass_count < 12`, or any of 01/02/26/27/28/30/31 breaks, or the FD-invariance
gate trips.

### Stage 2 — the p-pathway in the ACID flux blend weight (J2).

~10 lines, one new diagonal loop. Reads `alp_p[]` from Stage 1.

Expected: this is where the `(ra - rb) ~ -1000` water/air factor multiplies the mass and energy
fluxes — the strongest candidate for **13/25** if Stage 1 left them short, and the closest
structural analogue of Denner 2018's Table 3 case G (full-Newton advection, "best performance")
vs case D (Newton transient only, "2x").
Success bar: `pass_count >= 15` with 13, 15, 25 all passing; case13's `l2_p` back near the FD
value 0.01722 and `u_shock_delta_cells` back to 1.
If Stage 2 *degrades* relative to Stage 1: that is the Janodet (2025) "Picard for advection,
Newton for transient" outcome — keep Stage 1, record Stage 2 behind a default-off probe flag, and
document. Do not tune.

### Stage 3 — the T-pathway (CONTINGENT).

Run only if (a) Stage 0 measured a non-negligible `a_T`/loop gain in a still-failing case, or (b)
Stage 2 leaves case14 (or another air/water case) broken.
Two sub-parts, in order, each separately measured:
- **3a (Jacobian only):** star `D_T`/`N_T`, set `alp_p = ap + aT*dTp`, `alp_h = aT*dTh`, `alp_u =
  aT*dTu`, add the J2 columns 2 and 0. No `compute_R` edit. Note this is now the *fixed-point*
  derivative while the residual is still one-sweep lagged — a deliberate, declared, measurable
  mismatch.
- **3b (residual, only if 3a is inconclusive):** remove the alpha<->T lag by substituting
  `alpha(Y,p,T)` **inside** `T_from_hstat`'s inner Newton (pass `Y` instead of a fixed `al`, use
  `hsT*` as `dfdT`), making `compute_R` a true function of `(u,p,h,Y)`. **This flips the
  FD-invariance gate** and changes the ON+IMPLICIT residual, so it must be A/B'd against 3a with
  both Jacobians. Highest-risk stage; do not start it without an Advisor decision.

### Stage 4 — consolidation and documentation.

- Full sweep tables (all 19 cases, both Jacobians), wall-clock and mean inner-iteration counts vs
  round 4.
- Append `## 16. ROUND 5` to `docs/YADV_RESEARCH.md` in the established format: what was
  implemented, independently reproduced measurements, before/after tables, the §11.5 RH residual
  check (`scripts/yadv_rhcheck.py`) on 24/33/34, verdict, reproduction commands.
- A `scripts/yadv_r5_*.sh` pair mirroring `yadv_r3_build.sh` / `yadv_r3_ab.sh`.
- **Advisor-only decision, not the Worker's:** whether `ACID_YADV_ALPHA_IMPLICIT` folds into
  `ACID_YADV` as the default. Requires `pass_count >= 15` with 13/15/25 recovered and no new
  failures. `ACID_YADV` itself stays default OFF regardless.

---

## 5. Risks, per stage, and the measurement that catches each

| # | risk | stage | detection |
|---|---|---|---|
| 1 | Sign error in `a_p`/`a_T` (the `(ra-rb)` and `(ra ha - rb hb)` factors are easy to flip) | 0,1 | Stage 0 unit test vs central FD; ajblk `rel` must *drop*, not rise. A sign flip typically *doubles* the error — visible immediately. |
| 2 | Wrong which-phase-is-A convention (`alpha` is phase A's volume fraction; `Y` is phase A's mass fraction) | 0 | Unit test over **both** phase orders (air\|water and water\|air), as the existing round-trip test already does. |
| 3 | Double counting: adding a blend-weight term where the upwind-transport block already contributes | 2 | They are distinct product-rule addends (own-cell `alpha` column vs upwind-cell `rho_k` column); when `uw == i` both land in `aB[i]` legitimately. ajblk vs FD is the arbiter — double counting shows as a ~2x *over*shoot on `dcon/dp`. |
| 4 | Boundary faces (inlet `uin`, reflective `theta=0`) mis-weighted | 2 | Use `theta[f]` only (it already carries every override). Cases 01 (reflective-ish/static) and 07/30/31 (inlet) are the detectors; a boundary bug shows as a first-step blow-up. |
| 5 | `T_from_hstat` circular dependency mishandled | 1,3 | Stage 1 avoids it by construction (frozen-`T` derivative = exact derivative of the coded map). Stage 0 item 4 measures the loop gain; if `max g_i >= 0.5` anywhere, escalate to 3b before 3a. |
| 6 | `ACID_MNEWTON` stale-Jacobian caching interaction | all | None by inspection: `do_fd_assembly` is gated on `(!ajac \|\| ajblk)` (1445), so caching applies only to the FD path; the analytic block reassembles every iteration (1487). But: any `ACID_NO_AJAC` A/B must pin `ACID_MNEWTON=1` to be apples-to-apples. |
| 7 | TR-BDF2 forces FD -> acoustic cases silently unaffected, masking a bug | all | Treat as a gate, not a risk: 04/05/07/35/36 dumps **must** be byte-identical Stages 0-2. If they move, shared state was perturbed. |
| 8 | Improved Jacobian changes the dt-retry / `cfl_ramp` history, so a case passes or fails for dt reasons rather than Jacobian reasons | 1,2 | Cross-check any newly passing/failing case with `ACID_NO_CFLRAMP=1`; log per-case step counts and `dt` histories alongside metrics. |
| 9 | keep-best / stall-break (`rbest`, `best_it`, `ACID_STALLWIN=5`) masks non-convergence, so a metric improves while Newton still stalls | 1,2 | Log `conv_inner`, mean inner iterations, and the accepted-`al` histogram per case. A genuine Jacobian fix must *reduce* iterations and *raise* `conv_inner` rate — if metrics improve while iterations rise, the gain is accidental. |
| 10 | The new diagonal is so large it makes the continuity row ill-conditioned / the block solve near-singular | 1 | `ACID_LDBG` linear-residual ratio; `ACID_DBG` NaN probe. Mitigate structurally (line search, penta) — **never** with a damping constant. |
| 11 | Cases 24/33/34 stay broken and someone is tempted to "fix" them here | all | Explicit non-goal (§6). Report `scripts/yadv_rhcheck.py` unchanged and move on. |
| 12 | Performance regression (extra `phase_props` in the Jacobian loop) | 1,2 | `pa`/`pb` are already computed in that loop — the added cost is a handful of flops per cell. If wall clock moves >5%, something else changed. |
| 13 | OpenMP nondeterminism from a new parallel loop | 1,2 | Keep both new blocks **serial** (the existing analytic assembly is serial). Byte-identity gates catch it. |
| 14 | A stage "passes" only because a different case that was already failing flipped | all | Always compare the **failure set**, never just `pass_count`. Round 4's 12/19 vs 12/19 with different sets is the cautionary precedent (§15.2). |

---

## 6. Explicit non-goals

1. **No edits to `cases.cpp` or `validation.cpp`.** No gate, reference solution, case definition,
   or tolerance is touched. (Unbroken since round 1.)
2. **No new tunable constants**, no damping/relaxation factors, no per-case branches, no new env
   knobs beyond what a stage strictly needs (and any diagnostic knob must be default-OFF and inert
   unless `ACID_YADV` is set).
3. **The OFF path and plain `ACID_YADV=1` stay byte-identical at every stage** — 19/19 and 9/9 vs
   the published `solver_denner` binary; 15/19 and dump-identical for plain ON. All new code
   inside `if (yadv && alpha_implicit)`.
4. **Cases 24/33/34 are out of scope.** Round 4 showed implicit-alpha-by-re-evaluation moves them
   neither way (§15.5); their residual defect is the conservation failure of §11.6/§14.3, a
   different failure mode. Report their numbers, do not chase them.
5. **No change to the residual in Stages 0-2.** `compute_R` is edited only in the contingent Stage
   3b, and only after an explicit Advisor decision.
6. **No work on the segregated 2x2 analytic Jacobian or `s.drhodp`** — dead code under the default
   `unic` scheme (§0.1). If someone later revives `ACID_NO_UNIC`, that path needs the same
   treatment; note it, do not do it.
7. **No THINC / reconstruction / flux-form changes.** Rounds 2 and 3 closed those questions
   (§10.4, §14.4); the case02/14 reconstruction residual is a separate, documented defect.
8. **No promotion of `ACID_YADV` to default**, under any outcome.

---

## Critical files for implementation

- `cpp/denner_1d/src/acid.cpp` (J1 at lines 1503-1515; J2 as a new diagonal loop after 1562; the
  implicit-alpha residual block at 1014-1022 and the flux sites at 1172-1177 / 1204-1206 for
  reference; Stage 3b would touch `T_from_hstat` at 334-362)
- `cpp/denner_1d/include/denner1d/eos.hpp` (the two new inline derivative helpers, beside
  `alpha_from_mass_fraction` at 53-62)
- `cpp/denner_1d/tests/denner1d_unit.cpp` (Stage 0 derivative + identity + exact-zero tests,
  reusing the existing `(p,T,Y,pair)` grid)
- `docs/YADV_RESEARCH.md` (append `## 16. ROUND 5`)
- `scripts/yadv_r3_build.sh` + `yadv_r3_ab.sh` + `yadv_verify.py` (the measurement harness to
  clone as `yadv_r5_*`)

**Summary of the four things that make this plan different from round 4:** (1) the derivative is
a two-line closed form, `a_p = alpha(1-alpha)(zeta_b/rb - zeta_a/ra)`, automatically
clamp-consistent; (2) every new term is **diagonal**, so no solver structure changes; (3) `a_T` is
provably **exactly zero** for `b=0` phase pairs, which lets the risky T-pathway be deferred to a
contingent last stage; (4) the FD path and the 5 TR-BDF2 acoustic cases give two byte-identity
invariants that make each stage falsifiable before the 19-case sweep is even read.
