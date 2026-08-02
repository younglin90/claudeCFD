# Round 23 Plan — What RECON does that RESYNC does not: separating `dal_remap` removal from the state write on case24

**Thread**: round 22 §32.6 pt.3's explicit open question -- "`dal_remap` collapses under BOTH
mechanisms, yet case24 outcomes differ (20x vs 2.6x) -- suggesting the state-level write RECON
performs carries additional case24-specific benefit beyond `dal_remap` removal alone."

**Advisor verification note**: key structural claims checked directly against the code --
`acid.cpp:926-927` (`s0` snapshot boundary), `:1184-1190` (alpha recovery site -- the birth site
of round 16's collapse), `:1197-1205` (Eqs.43-44 rebuild), and `cases.cpp:148`
(`s.alpha_post = s.alpha_pre; // psi held`) -- all confirmed exactly as the plan states. This last
fact establishes case24's exact invariant `alpha(x,t) == 0.5` for all x,t, a tuning-constant-free
error norm this round's diagnostics (`ACID_ADRIFT`) will use.

Non-goals: case13's Jacobian-sensitivity (reference only), `rho_star`/`theta_o`, case33 advection
channel, `max_steps`, case29.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` sect.33)**: Stage 0
(zero code, using existing `ACID_RINIT`/`ACID_RECON`/`ACID_RESYNC` instrumentation) confirmed P1
(step-0 identity), P2 (RECON does real O(1) work from step 1), and P4' (drho separation -- plain B
loses ~500 of a cell's 499.58 true mass at steps 0-2, B+RECON suppresses this to single digits by
step 4). Stage 1 implemented ONLY `ACID_PROJ_UNTIL` (the dose-response knob; the plan's 5.1
relative-dp meter extension and 5.3 ACID_ADRIFT trace were not needed and not implemented -- scope
note in sect.33.5). **P6' (roundoff-null control) decisively excludes H-B**: `ACID_PROJ_UNTIL=1`
reproduces plain B's stall EXACTLY (step 19, identical rbest/r_init). **But P6 (monotone
dose-response) is FALSIFIED**: N=2 gives step 6 (worse than N=1), N=10-100 give no stall at all,
N=200 gives step 501 (further than N=400's 399) -- non-monotone. The two pre-registered tests
disagree on H-A vs H-B, an outcome the plan's S1/S2/S3 table didn't anticipate as a combination.
**Unplanned discovery**: N=50's "no stall" was checked against the actual validate gate and found
`pass=false` with a severely wrong solution (shock frozen mid-domain) -- "completes without
STALLED" and "correct answer" are different properties never previously distinguished because
B+RECON's own always-applied run never completes either. **Verdict: S4 (partial attribution),
declared explicitly** -- H-B is ruled out, H-A's mechanism is real and measured, but the practical
dose-response relationship is more complex than predicted (likely a withdrawal-point compounding
effect, not characterized further). Section 8 (third projection) NOT attempted -- its own gate
("only if S1 or S2 fires") is not met by S4. consecutive_failures NOT incremented.

---

## 0. Executive summary — the derived answer this round's job is to confirm or refute

`dal_remap` is a DIAGNOSTIC of the defect, not the defect itself. The real defect is a `(p,T)`-lag:
Y-transport delivers post-shock mass fraction into a cell BEFORE Newton has raised that cell's
`(p,T)`; alpha recovery (`1184-1190`) evaluates `alpha_from_mass_fraction` at the LAGGING `(p_o,T_o)`
where `dalpha/dY` is huge for case24 (~859 at the pre-shock state), driving alpha toward 1; the
Eqs.43-44 rebuild (`1197-1205`) then deletes ~99% of the cell's mass at that alpha -- round 16
§26.1's vacuum blister, arithmetically reproduced for case24 in §2 below.

- **RECON removes the lag itself**: moves `(p,T,alpha)` onto the conserved-state manifold, so next
  step's recovery evaluates at a CONSISTENT `(p_o,T_o)`. Collapse never starts. Preventive.
- **RESYNC removes only the diagnostic**: moves Y onto the lagging state so
  `alpha_prev == alpha_from_mass_fraction(Y, rho_a(p_o,T_o), rho_b(p_o,T_o))` holds TRIVIALLY
  (tautological -- Yv0 is defined as the pre-image of s0.alpha under the exact map dal_remap
  measures) -- but `(p_o,T_o)` themselves are bit-unchanged, so the lag and its dalpha/dY
  amplification on the NEXT advection increment are fully intact.

Predicted split: RESYNC's 2.6x (19->50) is round 13's `1/dt`/`r_init` channel alone; RECON's
remaining ~8x (50->399) is the §26 collapse-prevention channel, which only a state write reaches.

**Second derived claim, sharper and independently falsifiable**: RECON's repair is PREVENTIVE not
CURATIVE. §2.4 shows the closed-form inversion applied to an ALREADY-collapsed cell returns the
collapsed state back (that state genuinely is a near-pure-air PTE state). So RECON must delay
failure by roughly how long it's applied, predicting a monotone near-affine dose-response --
simultaneously the test that bounds the competing hypothesis (H-B: Newton trajectory sensitivity)
out, via a roundoff-null control.

---

## 1. Code facts verified (post-round-22 line numbers, `acid.cpp` unless noted)

| Line(s) | What is there |
|---|---|
| `334-362` | `T_from_hstat` -- clamps T to [1e-6,1e6]; returns true even when saturated (round 16's silent-saturation finding) |
| `743-755` | `Vec Yv` IC init -- the expression RESYNC re-applies every step |
| `796-810` | dt finalised |
| `812-867` | RECON block. Gate `817`: `yadv && (yrecon||recon_dbg)`. Exact skip `828-830`. Closed form `831-834`. Writes `839-848`: `s.p,s.T,s.alpha` (then `eval_thermo` `851`, `s.h` for touched cells `853-854`) |
| `869-921` | RESYNC block. Mutual-exclusion notice `878-885`. Gate `886`: `else if (yadv && (yresync||resync_dbg))`. Single write `902`: `Yv[i]=Ynew` only -- no `s.*` field touched |
| `926-927` | `const Field s0 = s; const Vec Yv0 = Yv;` -- both projections run strictly before this (verified) |
| `956` | `p_o, T_o` captured from the projected `s0` |
| `1138-1145` | `al_o[i]` -- alpha implied by pre-advection Yv at `(p_o,T_o)`, mass-flux blend weight |
| `1174-1183` | conservative rho*Y update, `rho_star` divide, `Yv=anew` |
| `1184-1190` | **alpha recovery site** -- `alpha_from_mass_fraction(Yv[i], rho_a(p_o,T_o), rho_b(p_o,T_o))` -- verified, the §26.1 birth site |
| `1197-1205` | **Eqs.43-44 rebuild** -- `rho_o, hstat_o, Htot_o` from NEW alpha at OLD (p,T) -- verified |
| `1213-1241` | `RMISM` (`ACID_RINIT`): prints `dh`, `drho=abs(s.rho[i]-rho_o[i])`, `dal`, `dal_remap`, `dal_adv`, argmax cells. Unset `ACID_BLK_STEP` -> every step every retry |
| `1248-1263` | `RCELL`: per-cell window (`Y0,Y,al0,al,p_o,T_o,u_o,h,Htot_o,rho,rho_o`) |
| `2202` | `if (it==0) r_init = n0;` |
| `2478-2479` | `bad` gate -- `ajac`-gated (reason 1 never fires under FD Jacobian) |
| `2507-2510` | F2''/reason-5 T-ceiling scan |
| `2633-2676` | `STALLED`/`STALLED-DETAIL` (`ACID_DBG`-gated) |
| `eos.hpp` | `mass_fraction_from_alpha`, `alpha_from_mass_fraction`, `pT_from_v_e_massfrac` (round 21) |
| `cases.cpp:148` | `s.alpha_post = s.alpha_pre; // psi held (homogeneous mixture)` -- VERIFIED |
| `cases.cpp:689-694` | case 24/33/34 IC: alpha uniform at t=0 (0.5/0.75/0.25 resp.) |
| `scripts/yadv_r9_sweep.py:34-38` | `ACID_ENV_VARS` purge list -- any new flag this round must be added |

Flag-combination facts (load-bearing for Stage 0):
1. `ACID_RECON=1` with `ACID_YADV_RESYNC=1` (RECON var unset): RECON meter measures the state
   BEFORE this step's resync -- exactly the departure RESYNC leaves in the state.
2. `ACID_RESYNC=1` with `ACID_YADV_RECON=1` (RESYNC var unset): meter runs, writes nothing;
   `dM_total` is a would-be accumulator only in this configuration -- only `worst_dY` is
   interpretable.

---

## 2. The derivation (pre-registered before measurement)

### 2.1 case24's exact invariant

`cases.cpp:148` + IC seeding => `alpha(x,t) == 0.5` for ALL x,t in the exact solution. Any
departure is pure numerical error, no threshold/constant needed (same for 33: 0.75, 34: 0.25).
Corollary: `Y = alpha*rho_a/(alpha*rho_a+(1-alpha)*rho_b)` must JUMP across the shock since
rho_a/rho_b jump -- the entire {24,33,34} difficulty lives in this alpha-held-but-Y-jumps gap.

### 2.2 Hand-derived case24 state (verify numerically in Stage 0.0, not asserted blindly)

Air (gamma=1.4,cv=717.5,Pi=b=eta=0), Denner water (gamma=4.1,Pi=4.4e8,b=eta=0,cv=474.2).
Pre-shock: p=1e5, T~300K, alpha=0.5, rho_a=1.1613, rho_b=998.0, rho=499.58, Y~1.16e-3.
Post-shock: p~1.5e10, T~1.7e4K, alpha=0.5, rho~1857.5, Y~0.833.

Amplification: dalpha/dY|_{Y->0} = rho_b/rho_a = 859 at pre-shock (p,T). A cell receiving Y=0.36
while still at pre-shock (p,T) recovers alpha~0.998, and the Eqs.43-44 rebuild sets
rho_o ~ 0.998*1.16+0.002*998 ~ 3.2 -- **99.4% of the cell's true mass (499.58) deleted silently.**
Round 16 §26.1's mechanism, reproduced arithmetically for case24.

### 2.3 What RECON does at that cell -- the preventive repair

Applying `pT_from_v_e_massfrac` to the cell's TRUE conserved state (rho=499.58, e=5.827e5, Y=0.36)
returns p*~3.5e7, T*~407K, alpha*~0.60 -- and rho implied at that state ~499.4, matching the true
mass. Recovering alpha at THIS (p_o,T_o) on the next step gives alpha~0.60, not 0.998 -- the
collapse never starts.

### 2.4 Why the repair is preventive, not curative

Applying the SAME inversion to an ALREADY-collapsed cell (rho~3.22, e~4.816e5, same Y=0.36) returns
p*~1.06e5, T*~320K, alpha*~0.998 -- essentially unchanged, because that state genuinely IS a
near-pure-air state at ~1 bar. RECON cannot undo a collapse, only prevent one -- explains the delay
(not fix) and predicts a monotone dose-response (P6), which is also the H-B bound.

### 2.5 Why RESYNC gets none of this

RESYNC writes only Yv, at the pre-advection point, as the exact inverse of the recovery at
`1184-1190` evaluated at the SAME (p,T) -- so `dal_remap` collapsing to ~1e-16 under RESYNC is a
TAUTOLOGY, not a repair. `(p_o,T_o)` at `956` are bit-unchanged from the un-projected run; the lag
and its dalpha/dY amplifier on the next step's advection increment are fully intact. The advected
mass that created the inconsistency is discarded instead -- appears as case14's measured 16.1%
drift (round 22 §32.5), same lag on the conservation ledger instead of the alpha ledger.

### 2.6 Pre-emptive kill of the most obvious "third projection"

A pressure-preserving (Abgrall/Shyue-canonical) update at the §2.3 cell (hold p=1e5, solve for T
from 1/rho=Y/rho_a(p,T)+(1-Y)/rho_b(p,T)) gives T~1.93K -- the conserved state is NOT representable
at frozen pressure. **Any third projection that helps case24 must move p**, and therefore confronts
the Abgrall mechanism on 13/14 directly. Derivation-level result, deliverable regardless of the
primary outcome.

---

## 3. The competing hypothesis H-B (trajectory sensitivity), why it must be bounded not assumed away

RECON does NOT exact-skip on case24 (alpha=0.5 everywhere, not {0,1}) -- it perturbs essentially
all 800 cells every step, including a step-0 identity-to-roundoff perturbation (~1e-11, since the
IC is already a PTE state). If a 1e-11 nudge can move the stall from 19 to 399, the stall step is a
chaotic observable and NO mechanism can be attributed from single-realization comparisons -- this
must be TESTED (roundoff-null control, P6'), not assumed against. Round 22 §32.1's case13 finding
(Jacobian-approximation sensitivity affecting which discrete admissible state Newton converges
near) is real, measured precedent for exactly this class of effect -- reference only, not this
round's target.

FD-Jacobian control caveat: `2478` gates reason-1 on `ajac`; under `ACID_NO_AJAC=1` "newton-no-
progress" doesn't exist as a stall reason, so config G cannot discriminate H-A/H-B by "stall step"
the way B can -- run as a supporting failure-mode observation only (§4.7), not the primary test.

---

## 4. Stage 0 -- zero solver code

Build first (`DENNER_ACID=1` on every invocation, per denner-pitfalls.md; capture via Python
subprocess, never shell redirection). Never hardcode a step number.

### 4.0 Desk check of §2.2-2.3/2.6 numbers (throwaway Python, 5 min)
Gate: if the table is wrong, rewrite §2 before interpreting 4.2.

### 4.1 Baseline reproduction (hard gate)
`DENNER_ACID=1 ACID_YADV=1 [unset|+RECON|+RESYNC] ACID_DBG=1 denner1d_dump 24 | grep STALLED`.
Expect 19 / 399 / 50, reasons newton-no-progress / T-ceiling-saturated / T-ceiling-saturated. Any
mismatch -> S5, halt, investigate baseline first.

### 4.2 State-departure meter, all steps (core measurement)
`ACID_RECON=1` under (a) no projection, (b) RECON applied, (c) RESYNC applied (pre-resync
departure per the flag-combo fact above). Track `worst_dal@i` as the locality tracker (worst_dp is
dominated by post-shock cells at p~1.5e10 and won't track the front -- the plan's own caveat).

### 4.3 Mirror meter, all steps
`ACID_RESYNC=1` under (a) none, (b) RECON applied (duality self-check, P9), (c) RESYNC applied
(case24's own never-before-measured phase-mass drift).

### 4.4 `drho` trace (zero new code -- ACID_RINIT already gives this)
`ACID_RINIT=1` (BLK_STEP unset -> every step/retry): `RMISM`'s `drho` IS the mass the Eqs.43-44
rebuild deletes per step, argmax cell included. Direct observable of §2.2's collapse.

### 4.5 Per-cell window trajectory
`ACID_RCELL=<lo>:<hi>` from 4.2's worst_dal@i trace + 4.1's STALLED-DETAIL cell -- never hardcoded.

### 4.6 `r_init` channel separation
`ACID_RINIT=1 ACID_BLK_STEP=<that config's own stall step>` for B / B+RECON / B+RESYNC -- read `r`
across the 14 retries, reproducing round 21 §31.1's methodology.

### 4.7 FD-Jacobian supporting observation (non-comparable, report failure mode + step only)

**Gate to Stage 1**: must answer (i) does RECON's correction ever exceed roundoff (P2), (ii) does
RESYNC leave the state departure intact (P3), (iii) do the drho traces separate (P4'). If 4.2 can't
be interpreted due to the absolute/relative caveat, do Stage 1's meter extension first.

---

## 5. Stage 1 -- minimal new code, read-only/diagnostic, default OFF, zero numeric constants

Nothing promoted this round (round 4/13/18/21/22 precedent).

### 5.1 `ACID_RECON` meter extension: worst RELATIVE dp/dT (~4 lines)
Independently-tracked maxima on `|dp|/s.p[i]` and `|dT|/s.T[i]`, appended (not inserted) to the
existing print format so old parsers still work.

### 5.2 `ACID_PROJ_UNTIL=<N>` -- dose-response / roundoff-null knob (~5 lines)
```cpp
const int proj_until = []{ const char* e = std::getenv("ACID_PROJ_UNTIL");
                           return e ? std::atoi(e) : -1; }();
// inside time loop, before the RECON/RESYNC block:
const bool proj_now = (proj_until < 0) || (step < proj_until);
// RECON write gate: if (yrecon)  -> if (yrecon  && proj_now)
// RESYNC write gate: if (yresync) -> if (yresync && proj_now)
```
Unset -> `proj_until<0` -> `proj_now==true` always -> textually identical to current behaviour;
lives inside `yadv`-gated blocks so OFF path cannot see it. `N=1` = roundoff-null control (P6').

### 5.3 `ACID_ADRIFT` -- per-accepted-step state-extremum trace (~10 lines)
Placed after `t+=dt; ++step;`, NOT yadv-gated (works on OFF control too). One line per accepted
step: `almax@i, almin@i, Tmax@i, rhomin@i`. Gives, for case24/33/34 via §2.1's exact invariant, a
tuning-constant-free physical error norm `max(|almax-alpha0|,|almin-alpha0|)`.

### 5.4 Hygiene
Add `ACID_PROJ_UNTIL`, `ACID_ADRIFT` to `yadv_r9_sweep.py`'s `ACID_ENV_VARS`. `CONFIGS`/`EXPECTED`
must NOT change.

---

## 6. Stage 2 -- measurements and pre-registered predictions

M1 `ADRIFT` on case24 under B/B+RECON/B+RESYNC/OFF. M2 `ADRIFT` on 33,34 under B (budgeted). M3
dose-response `ACID_YADV_RECON=1 ACID_PROJ_UNTIL=N`, N in {1,2,5,10,20,50,100,200,400}. M4 mirror
dose-response for RESYNC, N in {1,5,25,50}. M5 4.2 re-run with 5.1 extension.

**Predictions + falsifiers** (declared before running):
- P1 step-0 identity: RECON's step-0 dp_rel/dal at 1e-12..1e-9. Falsified if dp_rel>1e-6.
- P2 RECON does real O(1) work: worst-relative dp grows from ~1e-11 to O(1) well before step 399.
  Falsified if it never exceeds 1e-6 -> RECON is roundoff noise, H-A dead, go to H-B branch.
- P3 RESYNC leaves departure intact: 4.2(c) same order of magnitude as 4.2(a). Falsified if
  orders of magnitude below -> RESYNC repairs indirectly, §2.5 wrong.
- P4' decisive drho separation: RMISM's drho grows to O(rho_pre)=O(500) under B and B+RESYNC
  before each stall, stays orders smaller under B+RECON for most of its 399 steps. Falsified if
  comparable -> stall-step gap isn't a collapse-rate effect (H-B/S4).
- P5 preventive not curative: failure arrives abruptly (two-sided, report which pattern occurs).
- P6 dose-response monotone near-affine: stall_step(N) ~ N+19 for N<<399, saturating at 399.
  Falsified if non-monotone/erratic -> H-B dominates (S2).
- P6' roundoff-null control (sharpest single test): `ACID_YADV_RECON=1 ACID_PROJ_UNTIL=1` stalls
  within a few steps of plain B's 19. Falsified if far away (>40, or completes) -> case24's stall
  step is chaotically sensitive to a 1e-11 perturbation; prior rounds' case24 numbers are single
  noisy realizations; stall step ceases to be admissible as a case24 metric without an ensemble.
- P7 1/dt channel worth 2.6x, no more: dal_remap~0 under both at each own stall step, stall
  persists regardless. Falsified if dal_remap not ~0 -> a prior G6 result doesn't reproduce.
- P8 case24's own RESYNC phase-mass drift exceeds 1% (non-load-bearing, never measured before).
- P9 the two projections are exact duals: `ACID_RESYNC=1` under B+RECON reports worst_dY at
  roundoff floor (~1e-16). Falsified if O(1e-3) or larger -> not duals, correction needed to §32.3.

**Hard gates**: G1 `--verify` OFF byte-identical. G2 `--sweep` unset unchanged, `EXPECTED` not
edited. G3 `denner1d_unit` clean. G4 new-flag no-op (stdout byte-identical with new flags unset,
and with `ACID_ADRIFT=1` set alone -- stderr only). G5 diff hygiene (no cases.cpp/validation.cpp,
no CONFIGS/EXPECTED, no -march=native).

---

## 7. Stop / decision rules

| Outcome | Trigger | Consequence |
|---|---|---|
| S1 H-A decisive | P2,P3,P4',P6,P6' all hold | Write §33: gain comes from the state write keeping (p_o,T_o) consistent with Y so Eqs.43-44 stops deleting mass; dal_remap removal alone worth 19->50 only. Only then, if gates green and time remains, proceed to §8 (design only). |
| S2 H-B decisive | P6' falsified and/or P6 non-monotone | §33 documents as first-class finding + correction notes to §31.6/§32.4-G5 (their case24 numbers are single noisy realizations) + methodological consequence (stall step not admissible as case24 metric without ensemble). Do NOT attempt a third projection. |
| S3 both, quantified | P2,P3,P4' hold, P6/P6' show bounded (not clean) sensitivity | Report split with numerical H-B bound. §8 permitted, caveated by that bound. |
| S4 partial attribution | P2/P4' ambiguous, or neither S1 nor S2's predicate holds and S3's bound can't be stated numerically | Declare S4 explicitly, name what is/isn't established. consecutive_failures NOT incremented. No third projection. |
| S5 baseline moved | 4.1 doesn't reproduce 19/399/50 | Halt scientific content, investigate+report baseline discrepancy as the deliverable. |
| S6 hard gate failure | G1-G5 | Round does not merge until fixed. |

**Expected outcome, declared honestly in advance**: no promotion, no new default. Deliverable is a
mechanism (plus, in S2, a methodological correction). That is the intended shape.

---

## 8. Secondary goal -- third projection (design only, conditional on S1/S3, do not start otherwise)

1. Pressure-preserving projections structurally impossible for this case family (§2.6, derived).
2. Alpha-only projection (hold p,T, write alpha from Y) reproduces the collapse exactly -- refuted.
3. Any third projection must write p, spatially restricted by a structural constant-free criterion
   (project rule: no tuning coefficients). No such predicate identified -- do not invent one under
   time pressure.
4. Strongest candidate: not a third projection at all -- move the repair UPSTREAM to the recovery
   site itself (`1184-1190`, recover alpha at a (p,T) consistent with the NEW Y). This is what
   round 16 §26.3 named F3, and what `+ALPHA_IMPLICIT` (config C) approximates by brute force.
   **Recommended as round 24's thread** if this round reaches S1/S3, with `C+RECON`=14/19 (harmless
   under C since the in-step closure already follows Y, round 22 §2) as the entry point.
5. Known dead end: promoting via `C+RECON` alone -- C already fails case14, 14/19<15/19.

---

## 9. Wall-clock budget

Case24 runs: 5 min each. M2 (33/34): 5 min each. M3's nine dose-response runs: 5 min each, total
45 min. Anything exceeding budget: report "not evaluated", never chase to a conclusion. Use Python
subprocess.run(capture_output=True) for per-step diagnostic capture, never shell redirection.

---

## 10. Literature

Already in repo (do not duplicate): round 21/22's 10 stubs, papers/library/md's Fujiwara 2023,
Collis 2025, AlahyariBeig & Johnsen 2015, Terashima 2025, Denner 2018 ACID.

New needed stubs (DOIs to verify before commit):
- Hawkins 2024 (arXiv:2408.16872, Anderson-Picard nonlinear preconditioning of Newton) -- closest
  formal statement of H-B; natural citation if S2 fires.
- Zhang/Kumbaro/Ghidaglia 2019 (10.1016/j.jcp.2019.04.007, conservative pressure-based collocated
  two-fluid solver) -- closest architectural sibling; check their step-boundary consistency
  treatment for a possible third answer to the RECON/RESYNC dilemma.

---

## 11. Deliverables

`docs/YADV_RESEARCH.md` §33 (33.1 Stage0+derivation, 33.2 trajectory tables, 33.3 dose-response/H-B
bound, 33.4 verdict against §7 naming the outcome letter, 33.5 correction notes if S2, 33.6
reproducing). `docs/YADV_ROUND_23_PLAN.md` (this file + actual outcome). `docs/YADV_ROADMAP.md`
update. `acid.cpp` diff only (5.1-5.3) + `yadv_r9_sweep.py` ACID_ENV_VARS. 2 paper stubs. Commits
per convention, local merge only, round does not merge if G1-G5 fail.
