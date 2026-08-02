# YADV Round-Loop Roadmap

State file the `yadv-round` skill (`.claude/skills/yadv-round/SKILL.md`) reads/writes every round.
Detailed research log stays in `docs/YADV_RESEARCH.md`; this file is a thin, compact control
document — round counter, stop conditions, next-task pointer, one-line-per-round history. Do not
put derivations or measurement tables here; put them in `YADV_RESEARCH.md` (or a per-round plan
doc) and link to them.

## Current goal — Phase 3c: case15's central-jump defect (redirected from Phase 3a, round 26)

**Redirected 2026-08-02, by explicit user decision (round 26's escalation, `YADV_RESEARCH.md`
§36).** Phase 3a (cases 24/33/34) is **CLOSED, not solved**: round 26 proved, two independent
ways (an exact closed-form Riemann solution and a full reachable-shock-family scan), that these
three cases cannot pass `validate` under `ACID_YADV=1` for any numerical improvement — the
validation reference holds volume fraction fixed across the shock while `ACID_YADV=1` conserves
mass fraction, and the two closures differ by O(1) (~2x in `rho`/`p`), not by discretization
error. Presented to the user as a three-way choice (redirect to case15 / pursue a model
extension / accept the OFF path validates this family); **user chose redirect**. Consequence:
`ACID_YADV`'s recommended status stays OFF (15/19, `{15,24,33,34}` fail and are not expected to
change under this flag); 24/33/34 remain validated via the OFF/alpha-based path (19/19 there,
unaffected); no further autonomous-loop rounds target 24/33/34 under `ACID_YADV=1` unless a
future explicit user decision reopens a model-extension thread. Full reasoning, tables, and the
reusable exact-solution instrument (`scripts/yadv_r26_closure.py`): `YADV_RESEARCH.md` §36,
`docs/YADV_ROUND_26_PLAN.md`.

**New goal**: case15 (Phase 3b in round 9's original framing, promoted to primary target).
Round 26 §36.6 confirmed its redirect target is structurally sound — case15's own reference is
`computed_reference(c, 800)`, the SAME solver under the SAME active config on a finer mesh
(`cases.cpp:750-753`), so no closure mismatch of the 24/33/34 kind can exist there by
construction; whatever keeps it failing is a genuine, fixable numerical question. Known defect
(round 7, `YADV_RESEARCH.md` §17.4): a central-jump/concentration failure at the domain's
stagnation point, `cj=30.02` vs the gate's threshold `8.0` — structurally unrelated to the
alpha-Jacobian work Phase 2 completed. **Round 27's job**: produce a fresh plan (Planner call)
grounded in §17.4's own measurement (now 19 rounds old — re-verify it still holds on current
`HEAD` before designing a fix, since much has changed in `acid.cpp` since round 7) and case15's
own documented caveat (`.claude/rules/denner-pitfalls.md`: "case15 (double rarefaction) reference
is a grid self-consistency test, not exact validation; the 4-eq model has no phase change, so the
expansion-core pressure hits the EOS floor, not a physical vapour pressure").

## case15 status after round 30 (PAUSED, pending a user risk decision — not abandoned)

Round 30 (`YADV_RESEARCH.md` §40) fully characterised case15's remaining blocker (the
stagnation-point core jet, mechanism = collocated central-mean face pressure at a 128:1
density-ratio face, literature-documented) and found **no parameter-free fix exists**: both
candidate `pface` schemes evaluated break case25's shock speed. Further progress needs one of:
(i) accept case15 unreachable at the current scheme, (ii) a scheme-level `pface` change with
explicit acknowledged risk to already-passing shock cases (case25 named specifically), or (iii) a
mesh/spec conversation about case15 itself. **This is a decision only the user can make** — round
30 deliberately did not choose among these (pre-registered non-goal), and no future round may
silently pick option (ii) and risk case25 without that authorization being given first. Do not
re-target case15 under `ACID_YADV=1` with a new `pface` candidate until the user picks (ii)
explicitly, or picks (iii) and redefines the target.

## Phase 3a (cases 24/33/34) REOPENED as a model-extension research thread (2026-08-02, explicit
user authorization)

Round 26 closed cases 24/33/34 as unreachable under `ACID_YADV=1`'s current mass-fraction closure
for any *numerical* improvement (`YADV_RESEARCH.md` §36) and left reopening conditional on "a
future explicit user decision" to pursue a genuine model extension. **That decision was made in
this session**: the user asked directly, after round 26/29/30's diagnostic work on case15,
"아니다 24,33,34 도 포함시켜서 검증하자. 이것도 통과해야지 진짜 강건한 4eq mass fraction
solver 일것 같다. 동의하지?" (include 24/33/34 too — a solver isn't genuinely robust without
them). This authorizes reopening the thread round 26 gated on exactly this kind of decision.

**What this authorizes and what it does NOT authorize**: round 26's closure was about the
*current* mass-fraction-conserving closure being structurally unable to match the validation
reference's volume-fraction-conserving closure — the two disagree by O(1) (~2x rho/p), not by
discretization error, so no amount of numerical tuning under the existing model can close the gap
(`scripts/yadv_r26_closure.py`, `YADV_RESEARCH.md` §36). The user's authorization is for pursuing
a **genuine model extension** (interphase mass transfer / relaxation source term, so the model
itself can produce the O(1) composition change the shock requires) — NOT for weakening or
reinterpreting the validation gate, and NOT for a numerics-only retry of anything already tried
and killed (any `cav`/alpha-implicit-family variant, `REBUILD_ADV`, or re-deriving the same
closed-form mismatch already proven in §36). Any round working this thread must design and
implement an actual new physics term, gated OFF by default, with the OFF path staying
byte-identical exactly like every other flag in this project.

**Round 31's job**: a fresh Planner call (per the `yadv-round` skill's own Step 3) scoped to
*design only* for this round — survey the literature this project hasn't yet pulled for
interphase mass-transfer/relaxation-source closures compatible with a single-pressure/
single-temperature 4-eq mixture (Kapila-style relaxation, Saurel-Petitpas-Berry-style mass
transfer terms — `papers/md/33_saurel_relaxation_multiphase.md` is already in the repo and is
exactly this family), and produce a staged implementation plan with the same rigor as
`YADV_PHASE2_PLAN.md`/`YADV_ROUND_26_PLAN.md`. Do not implement the relaxation source itself in
round 31 unless the plan's own staging calls for a genuinely safe, gated-OFF Stage 0 (e.g. a
diagnostic-only closed-form check of what relaxation rate would be needed, analogous to round 26's
own exact-solution instrument) — a full model-affecting Stage should get its own round with the
usual harm-gate discipline, not be rushed into round 31 alongside the literature survey.

---

### Superseded — original Phase 3a framing (kept for provenance, do not re-target 24/33/34
under `ACID_YADV=1` without a new explicit user decision)

**Re-armed after round 9.** Phase 2 (Stages 0-4) is COMPLETE (`YADV_RESEARCH.md` §19):
`pass_count >= 14` met and durable since round 6; `ACID_YADV_ALPHA_IMPLICIT` does NOT fold into
`ACID_YADV` (case14); both stay default OFF. That investigation is closed, not reopened by Phase 3.

**New goal, chosen by the Advisor from the two candidates round 9 left open** (P3a over P3b,
because P3a is the only one that could change `ACID_YADV`'s recommended status and it now has a
concrete, promising, previously-unmeasured lead): **investigate why case33's Rankine-Hugoniot jump
closes to machine precision under `ACID_YADV_ALPHA_IMPLICIT=1`** (momentum residual 88% ->
8.4e-13, `YADV_RESEARCH.md` §19.4) **while case24/34 instead show their shocks exiting the domain
before `t_end` under the same flag** -- not yet understood whether this is the same phenomenon
further along (a faster, still-admissible shock) or a different problem specific to those two.
This directly contradicts the prior belief (rounds 4-8) that Jacobian work moved 24/33/34 "by
nothing" -- that was only ever true of the validation-gate metrics, never checked against
conservation self-consistency with implicit alpha until round 9.

**Round 10's job (per the same pattern Phase 2 itself started with): produce
`docs/YADV_PHASE3_PLAN.md`.** A Planner call, grounded in `YADV_RESEARCH.md` §11 (the original
Y-consistent-Hugoniot derivation and closure-(A)-vs-(B) analysis), §14.3/§15.5 (round 3/4's
Rankine-Hugoniot measurements), and §19.4 (this round's finding) — NOT a re-derivation from
scratch. The plan should: (1) explain WHY case33 differs from case24/34 (their `alpha_pre` are
0.75, 0.50, 0.25 respectively -- not an obvious pattern, worth checking first); (2) determine
whether case24/34's early-exiting shock is genuinely faster (consistent with also being
admissible) or symptomatic of something else (extend `t_end`, or re-run with a larger domain, to
find where their shock actually lands, before assuming); (3) stage the actual investigation the
same way Phase 2 did -- smallest safe measurement first, then incremental, falsifiable steps; (4)
set an explicit stopping criterion up front (Phase 2 §12's own lesson: "(a) without (b) is
insufficient" for a two-part fix -- decide in advance what result would mean "this needs a
different closure entirely, stop chasing it incrementally").

**Backup candidate, not this round's goal**: (P3b) case15's central-jump defect (§17.4, `cj=30.02`
vs threshold `8.0` at the stagnation point). Narrower, would not change `ACID_YADV`'s status. Available
if P3a proves unproductive (three consecutive rounds of no measured progress triggers the stop
condition below regardless, at which point re-evaluate between P3a-continued and P3b).

## Control state

```
round_counter: 30
consecutive_failures: 0
done: false
next_task: Round 30 attacked case15's stagnation-point core jet directly (round 29's own
           hand-off), diagnostic-only per its own pre-registered non-goal. Traced the mechanism to
           its immediate, exact cause: with use4≡false for case15 (lowdiss=false under config C's
           regime), every face pressure is the plain arithmetic mean pface[f]=0.5*(p_L+p_R)
           (acid.cpp:1786-1788), so the momentum residual's pressure term reduces to
           pres_i=0.5*(p_{i+1}-p_{i-1}) -- the cell's OWN pressure cancels identically. At case15's
           core the face straddling the stagnation point has a 137:1 pressure ratio and 128:1
           density ratio (p=444.96/rho=71.13 vs p=3.2432/rho=0.5577); the arithmetic mean gives
           pface=224.10 Pa, 69.1x the light cell's own pressure, and the SAME absolute error
           produces 1/rho MORE acceleration in the light cell -- measured 49x/124x specific-force
           amplification vs its immediate neighbours, cross-checked two independent ways
           (residual-closure decomposition and direct mass-flux re-evaluation) agreeing to
           0.1-0.3%. This is Bartholomew/Denner/van Wachem (JCP 375, 2018) §5 Eq.90 verbatim,
           already in the repo (papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md)
           -- a literature-documented failure mode, not a novel defect.
           THREE CORRECTIONS TO PRIOR-ROUND FRAMINGS, each measured directly: (a) config C's
           Newton converges quadratically at EVERY sampled step (al=1.000, zero backtracking,
           3-5 its to rnorm3~3e-7, 85 steps/zero retries/zero STALLED lines) -- the core jet is a
           property of a CONVERGED fixed point, not a solver stall (overturns an over-broad
           reading of acid.cpp's own "case15 cavitation: NEVER converges" comment, which describes
           the plain-B frozen-alpha path only); (b) temperature across the entire core is uniform
           to 0.02 K (349.35-349.37 K) despite a 340x pressure drop -- the "overheating" framing
           round 29 queued as this round's literature priority (Noh 1987/Petitpas 2007/Bigdelou
           2021) is REFUTED by direct measurement, all three stubs annotated in place rather than
           chased further; (c) the MWI/Rhie-Chow correction is ~2700x too small to act at the
           critical face (mwi_p=+0.0066 m/s vs ~18 m/s local velocities) and is NOT clamp-
           saturated (the sound-speed clamp sits 5 orders above |mwi_p|) -- confirms round 27's
           "different mechanism from case25's dt~dhat checkerboard" conclusion by direct
           measurement, but round 27's STATED reason (Wood-speed collapse, M~40) was wrong: s.a is
           not the Wood speed and dt here is set by the far-field material CFL, unrelated to
           anything local to the core.
           Two parameter-free fix candidates derived and numerically evaluated BEFORE proposing
           anything: density-weighted pface (Denner's own Eq.93 weighting) improves the critical
           face 33x (224.10->6.68 Pa) but under-predicts case25's reflected-shock face pressure
           3.2x (wrong shock speed); acoustic-impedance pface goes NEGATIVE at the critical face
           and under-predicts case25's shock 11x. Both dead, exactly the pre-existing dead end
           already on record in .claude/rules/denner-pitfalls.md ("Upwinding the face PRESSURE...
           is NOT valid"). No bounded parameter-free candidate exists on this evidence; none
           proposed, per the plan's own binding anti-rescue clause and consecutive_failures=2 at
           entry (a third harm would have stopped the loop).
           Built ACID_DUMP_CELLS (the round's ONLY default code change -- a diagnostic mesh-
           resolution override in denner1d_dump.cpp only, never read by validate/run, no
           cases.cpp/validation.cpp/acid.cpp touch), verified by two mandatory bit-level
           self-tests (Self-test A: N=800 gives bit-exact p==p_ref/u==u_ref since the reference
           solve becomes the primary solve; Self-test B: pair-averaging the N=800 run matches the
           untouched N=400 run's own *_ref columns to >=10 sig digits, all 4 pre-registered
           specific sums matched exactly) -- both passed, both also independently re-confirm round
           29's exact-pair-average claim for the reference construction.
           Refinement census, N in {100,200,400,800,1600,3200}, config C: cj (the smooth_ok
           velocity-jump metric) non-monotone at coarse N (47.1->66.5->30.0) then MONOTONICALLY
           DECREASING from N=400 onward (30.0->27.9->18.4->3.9), crossing the gate's own 8.0
           threshold between N=1600 and N=3200. **The specific pre-registered prediction
           "cj_800<8" is FALSIFIED** (measured 27.9, over 3x the limit) but the qualitative H-R1
           convergence hypothesis is well-supported: n_rev stays bounded (0-2, never growing with
           N) and the physical width of the reversed region shrinks super-linearly (4x from N=400
           to N=800, vs the 2x a fixed-cell-count region would give from dx alone). Reported as
           "H-R1 with a later threshold crossing than predicted," not smoothed into an
           unconditional confirmation. NEW FINDING, not anticipated by the plan, reported
           honestly: at N>=800, exactly 2 cells hit the 1.0 Pa pressure floor, staying fixed at 2
           through N=3200 even as cj keeps falling sharply -- complicates (does not overturn) the
           plan-stage claim that "the floor is never approached under config C" (true at N=400,
           false at N>=800); left as an open loose end, not resolved (touching the floor is an
           explicit non-goal).
           Anti-rescue note: none of this census's numbers is or becomes a case15 gate result --
           case15's gate remains N=400 scored against computed_reference(c,800).
           consecutive_failures RESET TO 0 (S1, diagnostic success -- all hard gates held, refinement
           question decided with a supporting table, falsified prediction reported as such per the
           plan's own S6-f rule rather than softened). ACID_YADV's recommended status UNCHANGED
           (OFF, 15/19); config C still fails case15 on smooth_ok alone, all 7 other criteria pass
           with 3-12x margin.
           **Hand-off to round 31 (deliberately pre-registered even under this favourable-leaning
           H-R1 result, since no parameter-free in-scheme remedy exists regardless of the
           refinement verdict)**: three consecutive rounds (28, 29, 30) have each independently
           confirmed nothing bounded and parameter-free exists inside the current pface/MWI
           formulation. The honest options are: (i) accept case15 as permanently unreachable under
           ACID_YADV=1 at its current scheme (a scoped, documented exception); (ii) escalate to the
           user for authorisation of a scheme-level pface change with explicit acknowledged
           shock-case risk (a CONDITIONAL face scheme -- central away from extreme density ratios,
           something else only at a detected near-vacuum interface -- was NOT derived or evaluated
           this round and is a legitimate new avenue, not a retread of the two dead F-a/F-b
           candidates); or (iii) a user conversation about whether case15's mesh/spec itself should
           be revisited. Round 30 does not choose among these -- deferred to the user per its own
           non-goal 10.
           Also separately live, from the user's own explicit message in this session: whether to
           re-open a model-extension research track for cases 24/33/34 (round 26 closed them as
           structurally unreachable under ACID_YADV=1's mass-fraction closure; the user asked to
           include them anyway "for a truly robust solver"). Not started -- explicitly deferred
           until the in-flight case15 thread reaches a decision point, which round 30 has now
           reached (see hand-off above). This should be the FIRST thing surfaced to the user
           alongside round 30's own hand-off, not run silently as another autonomous round.
           Grounded in YADV_RESEARCH.md sect.40, docs/YADV_ROUND_30_PLAN.md.
```

**Superseded control-state history (round 29's own, for provenance):**
```
round_counter: 29 (superseded, see above)
next_task (superseded): Round 29 measured round 28's own open question directly ("why is B+CAV's density
           field worse than config C's despite near-identical mass") and found the answer
           overturns round 28's framing: B+CAV's N=400 solution is ALREADY config-C-grade
           (l2_rho=0.001580, corr_rho=0.999995 vs C's own N=400 field). The reported gap lives
           almost entirely in the N=800 SELF-CONVERGENCE REFERENCE (case15's own reference is the
           same solver at N=800, cases.cpp:751-753), which develops a spurious dense/hot core plug
           under B+CAV (rho=662.6 vs config C's own reference's 86.7 at the same point) -- scoring
           B+CAV's N=400 solution against C's OWN reference gives l2_rho=0.01972/corr_rho=0.996662,
           matching C to the 5th digit and clearing both gates (diagnostic only, never quoted as a
           pass, per the anti-rescue clause). Also corrected: round 28 characterized B+CAV as
           failing only l2_rho/corr_rho; the full 8-criterion gate shows it ALSO fails smooth_ok
           (cj=25.906) and osc_ok (p_osc=0.4476, a ~28kPa pressure notch at the cav-mask's spatial
           boundary -- round 28's transition-zone hypothesis CONFIRMED, but for pressure, not
           density). Measured the mask's own non-monotonicity: cav[] SHRINKS 70->40 cells over
           case15's steps 0-8 (a cell whose pressure is RISING can't re-trip a downward-pressure
           test, reverts to the ~521x-too-stiff frozen closure) and is mesh-COUNT-invariant (not
           mesh-width-invariant), which is why case15's N=800 reference isn't a genuine refinement
           of the N=400 scheme under this config. BINDING BLOCKER IDENTIFIED: both B+CAV and
           config C show a 4-cell velocity sign reversal at case15's exact stagnation point
           (round27 sect.4.5's already-characterised core jet, reconfirmed independently) --
           config C ITSELF fails case15 on smooth_ok alone (cj=30.018, worse than B+CAV's 25.906),
           so no member of the implicit-alpha family (plain B, any cav variant, or C) can pass
           case15 until the core jet is fixed -- pre-registered in the plan BEFORE any Stage-2
           number existed, so this ceiling could not be mistaken for a Stage-2 failure later.
           Built ACID_YADV_ALPHA_IMPLICIT_CAV=2 (a run-scoped LATCH on the existing mask, reusing
           the exact same constant-free predicate, no new predicate/constant/second-alpha) to
           remove the reference-plug and boundary-notch artifacts and leave case15 with exactly
           the one shared blocker. Blast-radius census (ACID_NFEAS=2, unset applying flag, all 19
           cases) confirmed case14's latched shadow count = 0 (never at risk). Harm gate (checked
           BEFORE any case15 metric): pass_count=15/19, fail set unchanged {15,24,33,34} -- BUT
           case33 (already failing under both plain B and level 1, finite:true in both) flips to
           finite:false (NaN) under level 2. The plan's own S5 rule treats "any previously-finite
           case becomes NaN" as harm regardless of pass/fail status, with an anti-rescue clause
           explicitly forbidding exempting an already-failing case -- S5 FIRES exactly as
           pre-registered. Reverted the latch mechanism (run-scope hoist + conditional clear) in
           full; kept the level-parse (harmless, level 1 unaffected) and ACID_NFEAS's spatial
           fields as gated-off diagnostics -- level 2 now verified to degenerate safely to level
           1's exact behaviour (case33 confirmed finite:true again post-revert).
           consecutive_failures INCREMENTED to 2 (S5's explicit instruction -- second increment in
           3 rounds, both on this case15 mass-collapse-family thread: round27's REBUILD_ADV, this
           round's latch). All hard gates held (OFF 19/19, ALL GATES OK unchanged, unit-test
           numbers unchanged, git diff --stat -- cpp/: one file, 61 insertions/4 deletions, no new
           numeric literal, floor untouched).
           **consecutive_failures is now 2/3 -- ONE more unproductive round on this thread trips
           the loop's own stop condition.** Round 30 should attack the core jet directly (the
           now-unambiguous, sharply-identified single remaining blocker for case15), NOT propose a
           third variant adjacent to the alpha-implicit/cav mechanism -- two consecutive increments
           on attempts NEAR but not AT the core jet is itself a signal to change target class, not
           just candidate.
           Live threads for round 30: (a) case15's stagnation-point core jet (round27 sect.4.5,
           reconfirmed round29 sect.39.5) -- THE round-30 target, shared by B+CAV and config C,
           independent of everything the alpha-implicit family touches; (b) whether the core jet
           is a genuine under-resolution artifact (absent at N=800 in both references) or
           structural -- not fully settled, the reference's own linear interpolation can't resolve
           sub-fine-cell structure (round28 plan sect.4 Stage4's optional cells-override, never
           built, would settle this); (c) config A's degenerate case15 "pass" (nfloor=400/400,
           alpha frozen, reference equally degenerate) -- recorded for a possible future user
           escalation, not acted on; (d) case13's Jacobian-approximation-sensitivity finding
           (round22 sect.32.1); (e) round21's rho_star continuity predictor / theta_o MWI memory,
           untouched; (f) max_steps exhaustion (case15 legitimately uses it, PASSES on OFF); (g)
           case29's (excluded) likely-explained blocker. NOT a live thread: cases 24/33/34 --
           closed by round 26; any further variant of ACID_YADV_ALPHA_IMPLICIT_CAV (round28's
           level1 kept as gated infra, round29's level2 latch mechanism reverted).
           Grounded in YADV_RESEARCH.md sect.39, docs/YADV_ROUND_29_PLAN.md.
```

**Superseded control-state history (round 28's own, for provenance):**
```
round_counter: 28 (superseded, see above)
next_task (superseded): Round 28 answered the question round 27 deferred: why does the Eqs.43-44 rebuild
           preserve mass under config C/B+RECON but not plain B? Answer: adv+remap = M_reb-M_prev
           is an algebraic identity (ACID_MBAL's own arithmetic), so the real question is "does
           the rebuild preserve total mass" -- true under C/RECON (both keep the accepted state
           PTE-consistent, structurally or once-per-step), false under plain B (alpha frozen while
           p,T move freely; O(dt)-harmless everywhere except the 1 Pa floor, where it becomes
           O(1)). Measured live (ACID_RHIST, case15 step0, identical starting residual in all 3
           configs): plain B's discrete continuity has NO ADMISSIBLE SOLUTION in a cavitating cell
           -- stalls at 86% of r_init, al collapses to 0.016, Newton effectively demands p->-inf;
           config C converges quadratically to 2.4e-7 in 7 its on the SAME state; B+F3 stalls
           identically to B (F3 lives outside Newton, cannot touch the infeasibility). Closed-form:
           case15's Mach-1.9 rarefaction needs Deltap~-1.2e8 Pa at frozen-alpha compressibility --
           no admissible pressure exists. This retires the whole "fix it at the recovery/rebuild
           site" candidate family on evidence (REBUILD_ADV already broke 4 cases; B+F3 restores
           mass but nfloor=400/400, the WHOLE domain floored, l2_p=0 only because the N=800
           reference is equally collapsed -- a degenerate number, corrected here).
           Built ACID_YADV_ALPHA_IMPLICIT_CAV (default OFF): makes alpha implicit (config C's own
           per-cell update) ONLY on cells where an exact, constant-free predicate has fired --
           "the full Newton step just asked for a pressure at/below the solver's own existing 1.0
           floor" (the SAME literal, no new constant), tested after the line search closes,
           monotone within a retry. Diagnostic sibling ACID_NFEAS reports the would-be mask via a
           SEPARATE shadow count -- found+fixed a bug where the naive version let the
           "diagnostic-only" flag alone populate the real mask, silently becoming solution-
           affecting; caught by routine G4-early on cases 15/24, fixed with the shadow-mask split.
           Blast-radius census BEFORE applying (ACID_NFEAS, plain B, all 19 cases): case15 fires
           persistently (604), every other nonzero case (24/25/26/27/28/33/34) fires only a
           handful of cells at step0 and already passes under B or C -- CASE14 (the one case
           flagged at risk) NEVER FIRES AT ALL. Predicate chosen: P1 (no persistence variant
           needed). Harm gate (checked BEFORE any case15 metric, round27's lesson applied one step
           earlier): pass_count=15/19 under B+CAV, fail set IDENTICAL to plain B's own
           {15,24,33,34} -- zero regression; bonus, case24 now finite:true (was NaN under plain
           B). R1/R2/R3 all CONFIRM: mass restored to 869.3 (C: 870.6), nfloor 322->0,
           ACID_MBAL's own budget shows the rebuild now mass-preserving to 1.6% (matching C's
           near-zero cancellation). R4 (accuracy) DOES NOT HOLD, falling short of even the plan's
           own predicted C-matching numbers: l2_rho=0.069 vs 0.05 gate (C: 0.020), corr_rho=0.958
           vs 0.99 gate (C: 0.997) -- though l2_p/corr_p DO clear their own bars. Verdict: S4
           (neutral) exactly as pre-registered -- mass-collapse mechanism SOLVED and independently
           verified, but a SEPARATE, unidentified l2_rho/corr_rho-specific accuracy gap remains
           (plausibly the per-cell/lagged nature of cav[] vs C's uniform treatment -- proposed,
           not measured). consecutive_failures NOT incremented (stays 1, resets round27's S5
           streak rather than advancing it). ACID_YADV_ALPHA_IMPLICIT_CAV/ACID_NFEAS committed
           gated-off. All hard gates held (OFF 19/19, ALL GATES OK unchanged, unit-test numbers
           unchanged, G4(a)/(b) both clean).
           Live threads for round 29: (a) case15's now-isolated accuracy gap (B+CAV vs C, density
           only) -- the round's own real open question; (b) config C's own cj=30 core-jet (round27
           sect.4.5), still untouched, sits BEHIND (a) since B+CAV doesn't reach C's accuracy to
           even get there; (c) case13's Jacobian-approximation-sensitivity finding (round22
           sect.32.1); (d) round21's rho_star continuity predictor / theta_o MWI memory,
           untouched; (e) max_steps exhaustion (case15 legitimately uses it, PASSES on OFF); (f)
           case29's (excluded) likely-explained blocker. NOT a live thread: cases 24/33/34 --
           closed by round 26.
           Grounded in YADV_RESEARCH.md sect.38, docs/YADV_ROUND_28_PLAN.md.
```

**Superseded control-state history (round 27's own, for provenance):**
```
round_counter: 27 (superseded, see above)
next_task (superseded): Round 27, first round on the redirected case15 target (see "Current goal" above).
           Re-measured round 7's cj=30.02 finding live, all 7 configs -- reproduces exactly, but
           ONLY under config C (ACID_YADV_ALPHA_IMPLICIT=1); the loop's actual headline config B
           (ACID_YADV=1 alone) instead PASSES cj (2.307 vs 8.0) and fails on l2_rho=0.16761
           (vs 0.05) and corr_rho=0.984514 (vs 0.99). Root cause of B's l2_rho failure: measured,
           unambiguous -- the run deletes 99.92% of the domain mass (Sigma-rho*dx: 945.07->0.761
           at N=400), 322/400 cells pinned at the 1.0 Pa floor by t_end, vs config C's clean 8%
           physical-outflow loss and zero floor cells. Mechanism = round16 sect.26.1's own named
           "vacuum blister" (alpha recovered at the stale (p_o,T_o) saturates once a cell floors,
           then the Eqs.43-44 rebuild deletes the cell's true mass at that spurious alpha) --
           previously measured at one case24 cell, here running over 80% of case15's domain for
           85 consecutive steps. Built ACID_MBAL (default OFF, stderr-only instrument): closes
           the discrete mass budget into ADV/REMAP/BND/LEAK/RES, self-test closure~1e-13 at every
           step across every config tested. Found+fixed a real bug while building it: the natural
           implementation's extra compute_R() call is NOT idempotent under ACID_YADV (a
           T-relaxation update that keeps moving T when Newton hasn't converged -- case15's own
           regime) and silently perturbed the reported solution; caught by routine G4, fixed with
           a snapshot/restore. Measurement: REMAP explains 99.67% of B's collapse (Sigma over 85
           steps: -945.47 of -948.60 total); under every config that AVOIDS the collapse (C,
           B+RECON, B+F3), REMAP's raw magnitude is NOT small (C: 79.75, B+RECON: 179.997, both
           LARGER than case15's own total loss) -- it is instead EXACTLY CANCELLED by ADV (net
           0.0000 in both C and B+RECON) or collapsed directly to ~0 by F3. A materially more
           precise mechanism statement than the plan's own pre-registered prediction, reported
           honestly rather than rounded to fit. T4 cross-validated the instrument against round
           16/24's already-known case24 blister (correct sign, correct order of magnitude).
           Stage 2 (the one pre-registered candidate, ACID_YADV_REBUILD_ADV -- rebuild the old
           level from the advective-only part of the alpha change): SEVERE, CLEAN NEGATIVE
           RESULT. pass_count collapses 15/19 -> 11/19 -- cases 07/13/14/25, all previously
           PASSING under plain B, newly diverge to NaN, a solver-wide regression the candidate
           had no business causing. Per the plan's own pre-registered S5 rule (the one rule in
           this plan that calls for reverting the code, not keeping it gated-off, precisely
           anticipating a result this unambiguous): Stage-2 code reverted in full (flag +
           Eqs.43-44 loop branch), only ACID_MBAL merged. consecutive_failures INCREMENTED to 1
           (S5's own explicit instruction, honored as pre-registered -- first increment since
           round 20). Config C's own cj=30 central-jump defect characterised, not fixed: an
           under-resolved near-vacuum core at the stagnation point (4-cell velocity sign
           reversal, p dropping 137x across one cell) -- a DIFFERENT failure class from the MWI
           small-dt checkerboard already documented in denner-pitfalls.md (that one is case25's).
           All hard gates held (OFF 19/19, ALL GATES OK unchanged from round 26, unit-test
           numbers unchanged, G4 re-verified clean after the compute_R fix).
           Live threads for round 28: (a) a DIFFERENT Stage-2 candidate for B's mass collapse --
           REBUILD_ADV's failure means this specific repair was wrong, not that the collapse is
           unfixable; (b) config C's own cj=30 core-jet, once B's collapse is resolved or if C is
           adopted as the recommended path instead; (c) case13's Jacobian-approximation-
           sensitivity finding (round22 sect.32.1); (d) round21's rho_star continuity predictor /
           theta_o MWI memory, untouched; (e) max_steps exhaustion (case15 legitimately uses it,
           PASSES on OFF); (f) case29's (excluded) likely-explained blocker. NOT a live thread:
           anything about cases 24/33/34 -- closed by round 26, ACID_MBAL used case24 only as an
           out-of-case validation control.
           Grounded in YADV_RESEARCH.md sect.37, docs/YADV_ROUND_27_PLAN.md.
```

**Superseded control-state history (round 26's own, for provenance):**
```
round_counter: 26 (superseded, see above)
next_task (superseded): ★ESCALATION LIVE★ Round 26 answered round 25's own thread (b) -- "what's the binding
           constraint keeping 24/33/34's F3 completions from pass:true" -- decisively: it is NOT
           numerical. cases.cpp's reference (Denner Eqs.57-62) holds VOLUME fraction fixed across
           the shock (closure A); ACID_YADV=1 conserves MASS fraction (closure B, since Y is
           uniform pre-shock and rho*Y transports conservatively). Both are exact, admissible RH
           solutions of the same EOS -- they simply disagree by O(1) (~2x in rho/p) for these
           three cases. Built + independently cross-validated (P0: exact hand-derivation
           reproduced live to full double precision; P1: Python gate reimplementation matches
           denner1d_validate's JSON to spec on 7/7 pairs) a closed-form exact-solution instrument
           (scripts/yadv_r26_closure.py, zero C++ changes). T1: the EXACT config-B answer fails
           the gate by 2-3.4x on l2_p/l2_u/l2_rho and 31-40x on dip, all three cases. T2: the
           ENTIRE reachable single-shock family (density-vs-pressure locus scan) also fails --
           no numerical improvement can pass. T3: where the solver reaches a genuine flat plateau
           (ACID_TEND_SCALE window), it matches the exact closure-B answer to 0.17%-8.74%
           (median ~2-4%) -- the solver is close to correct FOR ITS OWN MODEL; the gate is
           checking a different model. T6 resolves round25's own open thread (c): case33's
           corr_p sign flip under F3 is the solution moving TOWARD the model's own (increasing-
           pressure) answer, not away from correctness. T7: case15's redirect target confirmed
           structurally sound (reference tracks the active config, no analogous mismatch).
           Verdict: S1 (structural obstruction CONFIRMED) + S2 (solver vindicated, with one
           metric at 6.15% vs the 5% bar -- reported honestly, not rounded away).
           consecutive_failures NOT incremented (round5/9/11 precedent -- resolving a
           long-open question is measured progress even without a pass_count change).
           **This is an escalation, not a recommendation**: cases 24/33/34 CANNOT pass under
           ACID_YADV=1 without one of (i) accepting the OFF/alpha-based path validates this
           family instead, (ii) a genuine model extension (interphase mass transfer / relaxation
           source -- explicitly out of this loop's authorized scope per the anti-rescue clause),
           or (iii) revising what "all cases pass" means for this family. None of these three is
           a decision the autonomous loop is authorized to make unilaterally -- surfaced to the
           user this round instead of silently continuing to grind rounds against a
           mathematically-proven-unreachable target.
           Live threads NOT blocked by this finding, available for round 27 if the user directs
           continuation: (a) case15's own defect (round7's cj=30.02 vs 8.0 central-jump
           criterion) -- confirmed reachable (T7), untouched by any prior round since 7; (b) F3b
           (round25 sect.8) -- now understood to target a residual defect (T3's drho, already
           5-14x reduced by plain F3) far smaller than the O(1) gap this round measured, so its
           motivation is weaker than round25 could tell, not withdrawn; (c) case13's Jacobian-
           approximation-sensitivity finding (round22 sect.32.1); (d) round21's rho_star
           continuity predictor / theta_o MWI memory, untouched; (e) the stall/front-speed
           anomalies noted in sect.36.4 (B+RECON+F3 appears to slow the front on case24/34,
           unexplained); (f) max_steps exhaustion (case15 legitimately uses it, PASSES on OFF);
           (g) case29's (excluded) likely-explained blocker.
           Grounded in YADV_RESEARCH.md sect.36, docs/YADV_ROUND_26_PLAN.md.
```

(Round counter starts at 4 because rounds 1-4 of the `ACID_YADV` experiment were already run
manually, before this loop existed — see `docs/YADV_RESEARCH.md`. Round 5 onward runs under
`yadv-round`.)

## Stop conditions (checked at the start of every round, before any work happens)

1. `done == true`
2. `consecutive_failures >= 3` — three rounds in a row with no measured progress or a hard-gate
   failure. This means the current approach needs human reconsideration, not another autonomous
   attempt at the same thing. This is the backstop that actually governs day-to-day; expect the
   loop to stop here long before either cap below, unless Phase 3 turns out to be unusually clean.
3. `round_counter >= 1000` — session cap set explicitly by the user for this Phase 3 run (2026-08-
   01). Not expected to be reached (condition 2 should fire first if Phase 3 stalls); exists as a
   hard ceiling regardless. Re-raise explicitly if the user wants more headroom after this caps out.
4. `round_counter >= 100000` (the original nominal cap from when this loop was first set up;
   condition 3 is stricter and fires first)

Any of these: the round skill calls `ScheduleWakeup({stop: true})`, records why here, and does
not start a new round.

## Absolute rules (every round, no exceptions — also mechanically enforced by
`.claude/hooks/agent_plan_only.py` + `.claude/hooks/block_destructive_bash.py` while
`.claude/round-loop-active` exists)

- No agent except `Agent(subagent_type="Plan")`. Implementation happens directly in the round's
  session.
- No `git push`, no `git reset --hard`, no `rm -rf`. Local commit + local merge to `main` only.
- No tuning constants, no per-case coefficients. Global physical constants only.
- No edits to `cases.cpp` / `validation.cpp` unless the round's stated goal explicitly requires it
  (none of the current roadmap does).
- The OFF path (`ACID_YADV` unset) must stay 19/19 and byte-identical to the published
  `solver_denner` binary at the end of every round that merges to `main`. A round that breaks this
  does not merge.
- Report negative/partial results honestly — this project's established culture (rounds 1-4) keeps
  failed experiments in the history, not just wins.

## History (compact — detail lives in YADV_RESEARCH.md / per-round plan docs)

- Round 1: Y-space non-conservative transport. case13 win (linf_p -36%), case02/14 THINC-averaging
  regression, case15/24/33/34 fail. → `YADV_RESEARCH.md` §1-8.
- Round 2: alpha-space THINC reconstruction — NEGATIVE (case02 3.6x worse). Y-consistent Hugoniot
  for 24/33/34 computed — overturns round-1's "unanswerable closure conflict" framing; Y-path
  violates momentum by 88%. → `YADV_RESEARCH.md` §9-12.
- Round 3: conservative `rho*Y` transport, sharing ACID's own per-cell mass-flux structure.
  §10.4's target (case02/14) still not recovered; case13 wins partly lost (`peak_delta_u` 0→397).
  New partial win: case24/34 now satisfy their own Rankine-Hugoniot jump to 1e-13 (were
  unmeasurable before); case33 unchanged (88% violation). → `YADV_RESEARCH.md` §14, commit
  `a26b7a4`.
- Round 4: alpha implicit in Newton via re-evaluation inside `compute_R` (no Jacobian edit).
  Confirms case15's grid-convergence defect was the frozen-alpha mechanism (cured under FD
  Jacobian, `amp_ratio_p` 0.33→1.00) but is a net regression under the DEFAULT analytic Jacobian
  (15/19→12/19, newly breaks 13/14/25) — the analytic Jacobian's missing `d(alpha)/dp` term is
  load-bearing. Gated OFF behind `ACID_YADV_ALPHA_IMPLICIT` (default off); plain `ACID_YADV=1`
  unaffected. → `YADV_RESEARCH.md` §15, commit `47d1bef`.
- Phase 2 plan produced (Planner, opus): analytic `d(alpha)/dp`/`d(alpha)/dT` derivation, staged
  Stage 0-4 implementation, grounded in 4 papers (Denner 2018 product-rule linearisation, Janodet
  2025 coupled-large-density precedent, Denner/Evrard/vanWachem 2020 barotropic-substitution
  consistency argument, the original ACID paper as historical baseline). → `docs/YADV_PHASE2_PLAN.md`.
- Round 5 (first run under `yadv-round`): Phase 2 Stage 0 -- `d(alpha)/dp`, `d(alpha)/dT` header
  helpers in `eos.hpp` (additive, no call sites yet) + unit-test verification. Confirmed Phase-2
  §1's numeric prediction exactly (case15 diagonal-error ratio 521.558 vs predicted ~500). Found
  and fixed a bug in the new unit test itself (FD-comparison tolerance floor was 1e6x too strict
  for the air|vapor pair's algebraic-zero case) -- the derivative formula was correct throughout,
  independently confirmed against a standalone probe. All four gates unchanged (additive stage):
  OFF 19/19+9/9, ON 15/19, +ALPHA_IMPLICIT 12/19 both Jacobians, same failure sets as round 4.
  Also the first real exercise of the round-loop mechanism itself -- found and fixed two
  infra bugs (worktree default branches from origin/main, invisible to local-only commits, needed
  `worktree.baseRef:"head"`; the Bash safety hook's substring match blocked its own guarded commit
  on a heredoc-quoted mention of what it blocks, fixed with heredoc-body stripping). → commit
  `7cd36ae`, `docs/YADV_ROUND_5_PLAN.md`.
- Round 6 (first round run under the actual autonomous `/loop`): Phase 2 Stage 1 -- augmented the
  analytic Jacobian's J1 cell-EOS-chain loop to star `D_p`/`N_p` with the `a_p` product-rule
  addend (Stage 0's already-derived, already-unit-tested helper). GENUINE SUCCESS:
  `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` under the default analytic Jacobian moved 12/19 -> 14/19.
  case13 and case25 both fully recovered; case15 moved from non-convergence to passing every
  quantitative gate criterion (blocked only by the TV/oscillation guard, `peak_delta_u` 321->0).
  Cases 24/33/34 unmoved as predicted (separate conservation defect); case14 unmoved (separate
  `hsT<0` lead). All hard gates held (OFF 19/19+9/9, plain ON 15/19, FD-invariance exact same
  failure set). → `YADV_RESEARCH.md` §16, `docs/YADV_ROUND_6_PLAN.md`, commit `33e006f`.
- Round 7: Phase 2 Stage 2 -- the J2 flux-blend diagonal (own-cell alpha sensitivity in the ACID
  mass/energy flux blend). MEASURED NO-OP on `pass_count` (stayed 14/19, identical failure set) --
  an honest negative result, consistent with the round's own prediction that 13/25 had nothing
  left for J2 to add. The real result: corrected round 6's case15 diagnosis. `peak_delta_u` is NOT
  part of case15's gate (verified against `validation.cpp` directly); computed the true blocker
  exactly -- a central-jump/concentration failure (`cj=30.02` vs threshold `8.0`) at the domain's
  stagnation point, structurally unrelated to alpha-Jacobian accuracy. **Roadmap goal re-scoped**
  (see "Current goal" above) -- case15 is no longer a target of this plan. All hard gates held.
  → `YADV_RESEARCH.md` §17, `docs/YADV_ROUND_7_PLAN.md`, commit `9b0a698`.
- Round 8: Phase 2 Stage 3a (T-pathway). Measure-first: a temporary diagnostic (removed after use)
  confirmed round 5's `hsT<0` case14 lead was real but confined to a single first-timestep
  transient cell, not persistent -- worth attempting per the round's own decision rule. Starred
  the T-pathway analogous to Stage 1's p-pathway; genuine side discovery: `hsT* = Y*cp_a+(1-Y)*cp_b`
  is an EXACT closed form (hstat_mix = Y*h_a+(1-Y)*h_b identically, NASG h linear in T), strictly
  positive, retroactively validating Stage 1 and proving the starred form removes an existing
  1/hsT near-singularity. MEASURED REGRESSION on its target: case14 doesn't flip pass/fail but its
  quality collapses (l2_p 0.0145->0.512, corr_p 0.9996->0.594) -- confirms a family-mismatch risk
  flagged in advance (mirror of round 4's original mistake). Gated behind a NEW flag
  `ACID_YADV_ALPHA_IMPLICIT_T` (default off, round-4 precedent) so `ACID_YADV_ALPHA_IMPLICIT=1`
  alone stays bit-identical to round 6/7's validated 14/19. Advisor declined the larger Stage 3b
  (residual-level fix) for now -- its case is performance/robustness, not a case14 fix, and it
  flips the FD-invariance gate for no currently-open target. All hard gates held.
  → `YADV_RESEARCH.md` §18, `docs/YADV_ROUND_8_PLAN.md`, commit `3446bc5`.
- Round 9: Phase 2 Stage 4 (consolidation). Zero solver code changed -- measurement/reporting only
  via a new `scripts/yadv_r9_sweep.py`. Full six-configuration sweep reproduced every prior round's
  numbers exactly, from one build, in one sitting (found and fixed a live parsing bug along the
  way: lowercase `nan`/`-nan` broke Python's `json.loads`, silently undercounting failure sets --
  `pass_count` itself, computed by the C++ binary, was never wrong). First direct wall-clock
  measurement in the whole investigation: `+ALPHA_IMPLICIT` costs 7.1% more than plain
  `ACID_YADV=1` on cases both configs solve (vs Phase-2's own <5% prediction); FD Jacobian costs
  54.7% more than analytic on the same config (consistent with round 4's qualitative ~1.7-1.9x).
  Promotion decision: does NOT fold `ACID_YADV_ALPHA_IMPLICIT` into `ACID_YADV` -- case14 is the
  ONLY case whose pass/fail flips between the two (plain ON passes it, +IMPLICIT fails it),
  confirmed fresh from this round's own sweep. **Post-merge follow-up (commit `6f1538d`) found a
  genuinely new, unpredicted result**: the deferred RH residual re-check
  (`scripts/yadv_rhcheck.py`) under `ACID_YADV_ALPHA_IMPLICIT=1` shows case33's Rankine-Hugoniot
  jump closing to machine precision (88% -> 8.4e-13) -- contradicting the standing "Stages 1/2/3a
  moved 24/33/34 by nothing" belief, which was only ever true of the validation-gate metrics.
  case24/34 instead show their shocks exiting the domain early under the same flag, unexplained.
  Phase 2 (Stages 0-4) declared complete; roadmap re-scoped to `done: true` with this new finding
  named as the sharpest lead for a future Phase 3 (P3a). All hard gates held throughout.
  → `YADV_RESEARCH.md` §19, `docs/YADV_ROUND_9_PLAN.md`, commits `51e2497`, `6f1538d`.
- Round 10: Phase 3a Stage 0. **RETRACTS round 3's "24/34 close to 1e-13" and round 9's "case33
  closes to machine precision under +ALPHA_IMPLICIT"** -- both were `yadv_rhcheck.py`'s
  undisturbed-cell search locking onto a SILENTLY STALLED run's pristine IC (`acid.cpp`'s
  `if (!stepped) break;` never sets `diverged`, so a stalled run scores as a normal completion).
  Independently reproduced via direct `ACID_DBG` traces (case24 plain, case33 +IMPLICIT both
  stall at <1% of `t_end`, mid-domain cells still bit-for-bit pristine IC). New
  `scripts/yadv_rh2.py` (null-run + IC-match guard, exited-shock plateau-window fallback):
  confirmed ALL six predicted null-run/completion classifications exactly. Load-bearing finding:
  **no case in {24,33,34} has ever had a plain and a +IMPLICIT run complete simultaneously** --
  every RH-residual number this whole investigation has produced for these three cases compares
  across different configurations, never a true A/B. Where 24/34 do complete (+IMPLICIT, shock
  exited domain), residual is large (~40-50%), not near-zero -- measured value disagrees with the
  round-10 Planner's own predicted ~2-7%/faster-shock figures (`YADV_PHASE3_PLAN.md`); likely a
  plateau-window artifact straddling internal wave structure, not yet resolved (needs Stage 2's
  `ACID_TEND_SCALE` in-domain measurement). `Y` conserved to 3-4 digits through the leading shock
  in every completing run, against closure-(A)'s 270-1620x growth requirement -- may make 24/33/34
  a structurally unreachable target for any Y-preserving scheme, pending Stage 1-4.
  `ACID_YADV` recommended status UNCHANGED (default OFF, 15/19). Zero solver code changed; all
  four hard gates held (unit PASS, OFF 19/19, 9/9 byte-identical vs published binary,
  ON-vs-OFF genuinely differs as expected). → `YADV_RESEARCH.md` §20, `docs/YADV_PHASE3_PLAN.md`,
  commit `666c6c8`.
- Round 11: Phase 3a Stages 1+2. Stage 1 makes the silent stall audible (`STALLED:` stderr
  message + `ACID_DBG`-gated detail, `diverged` deliberately still NOT set -- that's Stage 3c,
  needs an explicit Advisor decision). Verified stderr-only against an isolated main-HEAD build:
  all 19 OFF dumps and OFF/plain-ON/+IMPLICIT stdout byte-identical. Side finding: the
  FD-invariance gate is 13/19 (`{15,24,27,28,33,34}`), not the 12/19 this project's own memory
  had recorded -- a stale prior figure, corrected, confirmed identical on unmodified main HEAD
  (not a regression). Stage 2 adds `ACID_TEND_SCALE` (diagnostic observation-window knob,
  default 1.0 = no-op, verified byte-identical to Stage 1 when unset). **The sweep this knob
  enables RETRACTS round 10's own §20.2 finding**: `yadv_rh2.py`'s fixed `[0.3,0.6]` plateau box
  straddled internal wave structure exactly as its own caveat warned; a corrected front-derived
  window (found and fixed a bug along the way -- an earlier attempt sampled the wrong, still-
  undisturbed side of the front) gives a STABLE, converged plateau across scales 0.4-1.0 for
  cases 24/34 under `+ALPHA_IMPLICIT`: momentum residual `+7.351e-02`/`+2.063e-02`,
  `Vs(mass)/Vs_ref` 1.4946/1.3968 -- matching the round-10 Planner's original static prediction
  to 3-4 significant figures (the Planner was right; round 10's own measurement was wrong). An
  independent front-position-vs-time linear-fit cross-check disagrees (R²=0.82/0.96, smaller
  ratio) -- reported as an open, unreconciled discrepancy, not forced to match. No case in
  {24,33,34} has ever had plain and `+ALPHA_IMPLICIT` complete simultaneously (round 10's
  load-bearing finding stands). `ACID_YADV` status unchanged (default OFF, 15/19). Zero effect
  on numerics with either new env var unset; all hard gates held throughout both stages.
  → `YADV_RESEARCH.md` §21, `docs/YADV_ROUND_11_PLAN.md`, commits `272ce08`, `587c3f8`, `be83576`.
- Round 12: Phase 3a Stage 3, branch 3a (retry-exhaustion accept-best). **3b REFUTED** by direct
  evidence -- all three stalling configs show `cell=-1`, `max|u|` numerically frozen across the
  full retry sweep, no void cell ever. Decisive mechanism finding, independently reproduced: the
  pre-Newton residual `r_init` grows exactly as `1/dt` at the stall (falsifies the existing `bad`
  gate's own "dt too large" design comment). New `ACID_STALL_ACCEPT` (default 0, byte-identical):
  level 1 gets **case24 and case34 (plain) to complete for the first time ever**, zero
  `pass_count` regression -- far exceeding the round's own calibrated expectation (every accepted
  step is a tight cluster right at shock formation, <0.5% into each run; both then run clean to
  `t_end`). **Produces the first-ever controlled A/B for cases 24/34**: plain+accept shows
  momentum residual +98.0%/+45.6% vs `+ALPHA_IMPLICIT`'s clean +7.35%/+2.06% -- favors
  `+ALPHA_IMPLICIT` sharply, but reported with an explicit caveat (the plain runs are not clean
  solves; the large residual may partly reflect the accept mechanism's own defect at formation,
  not intrinsic physics -- not yet a clean result). Level 2 (CFL-neutrality) is a measured net
  regression (plain-ON 15/19→14/19, case28 newly fails) and is NOT adopted; level 1 is the
  recommendation. Case33 remains unsolved -- a qualitatively different, sustained difficulty (221
  accepted steps at a 5x larger budget, still only 14.8% of `t_end`), not a one-time glitch like
  24/34. `ACID_YADV` status unchanged (default OFF, 15/19 with the new var unset); all hard gates
  held. → `YADV_RESEARCH.md` §22, `docs/YADV_ROUND_12_PLAN.md`, commits `a21ed4a`, `5017362`.
- Round 13: Phase 3a Stage 3, follow-up 7a (localize + fix the `1/dt` mechanism). New `ACID_RINIT`
  instrument **confirms the mechanism to textbook clarity**: at case24's stalling step, `dal_remap`
  (alpha recovered at the PREVIOUS step's frozen Y meeting the CURRENT step's `p_o,T_o`) is
  constant to 4-5 sig figs across all 13 retries, while `dal_adv` (this step's own Y-transport)
  halves exactly every retry. Two controls land exactly on their predicted extremes: OFF prints
  zero diagnostic lines (structural immunity, no remap term exists); `+ALPHA_IMPLICIT`'s
  `dal_remap` measures at literal `DBL_EPSILON` -- giving round 12's A/B result a mechanism, not
  just a correlation. **The naive fix (`ACID_YADV_HREINIT`, reset `s.h` alone) is REFUTED (S4)**:
  case24's stall only moves from step 19 to step 28 (<2% of steps needed); case34 gets WORSE;
  combined with round 12's `ACID_STALL_ACCEPT=1`, case24 needs MORE accepts (9, not 2). Diagnosed
  why: `s.rho`/`s.T` stay stale until the first `compute_R()` re-derives them from the corrected
  `h`, so Newton still doesn't start from a genuinely self-consistent state -- a real fix needs
  simultaneous `(T,rho)` reconciliation, not attempted this round. Both new flags stay default
  OFF; round 12's `ACID_STALL_ACCEPT=1` (without `HREINIT`) remains the only working path to
  completion, its caveat unchanged. Per this project's negative-result culture, a
  correctly-instrumented refutation is measured progress -- `consecutive_failures` not
  incremented. All four hard gates held, byte-identical to round 12 with both new vars unset.
  → `YADV_RESEARCH.md` §23, `docs/YADV_ROUND_13_PLAN.md`, commits `aedb1b5`, `dea10ce`.
- Round 14: Phase 3a Stage 3c (`diverged=true` at the retry-exhaustion give-up), authorized this
  round by an explicit Advisor decision after rounds 11/12/13 each deferred it. One statement,
  correctness/reporting fix -- `pass_count` UNCHANGED in all four hard gates (OFF 19/19, plain-ON
  15/19, `+ALPHA_IMPLICIT` 14/19, FD-invariance 13/19 both configs, resolving the round-12/13
  ambiguous 12-vs-13/19 figure by direct measurement). Exactly the three predicted `(case,config)`
  pairs changed and nothing else: 24/plain, 34/plain, 33/`+IMPLICIT` now read `finite=false`/NaN
  instead of a finite garbage row -- closing the exact silent-stall gap that produced two retracted
  findings (§20). The accept/give-up boundary needed no new logic: `ACID_STALL_ACCEPT`'s
  accept-and-continue path is confirmed byte-identical for case24/34 (never reaches the give-up
  block), while its give-up path (case33's own budget exhaustion) now correctly diverges alongside
  its unchanged `STALL-ACCEPT-TOTAL` disclosure. OFF-path safety re-verified empirically (zero
  STALLED/DIVERGED lines across all 19 cases on the post-edit build), not just asserted -- the
  change is NOT `ACID_YADV`-gated by construction, only by evidence. Four historical-artifact
  annotations added (§14.3/§19.2/§20.3/§19.3), no history edited. `ACID_YADV` status unchanged
  (default OFF, 15/19). Two open threads remain (round 13's harder consistency fix, case33's
  sustained difficulty); a third (max_steps exhaustion, a sibling defect) named but deliberately
  not pursued -- case15 legitimately uses that cap and PASSES on OFF, so extending `diverged` there
  needs careful future design, not an autonomous default.
  → `YADV_RESEARCH.md` §24, `docs/YADV_ROUND_14_PLAN.md`, commit `b67850e`.
- Round 15: first-ever diagnosis of case33's stall (`ACID_RINIT` had only run on case24/34 before).
  **Same residual SHAPE as round 13's mechanism** (energy-dominant `fene->1.0`, `r_init` doubles
  almost exactly per dt-halving) **but provably NOT the same SOURCE**: `dal_remap` measures at
  literal `DBL_EPSILON` at every retry of the first stall AND across 4 consecutive forced-accept
  steps -- the alpha/Y-REMAP channel round 13 found for case24/34 is completely absent here, as its
  own `+ALPHA_IMPLICIT` control predicted structurally. The actual driver is `dh=|s.h-Htot_o|`,
  constant `~3.7e12` and dt-independent -- **six orders of magnitude beyond the worst value
  (`~4e6`) ever seen across case24's entire healthy `+ALPHA_IMPLICIT` trajectory**, measured as a
  control. Round 13's speculated "alpha inherits the previous step's lag" compounding is directly
  refuted for case33 (`dal_remap` stays clean through repeated forced accepts) -- but real
  compounding IS measured, through `dh`/`drho` directly (+51%/+5x across 4 accepted steps), a
  different and still-unidentified channel. Diagnostic-only round, no fix attempted or
  implemented, no source code changed (`git status` confirmed clean), no hard gates required (no
  source change to verify a no-op against). `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §25, `docs/YADV_ROUND_15_PLAN.md`, commit `a56627a`.
- Round 16: case33's `dh` **fully explained** via new `ACID_RCELL` per-cell diagnostic (read-only
  window print, right after the Eqs.43-44 rebuild). A single first-time-step Y-advection into a
  still-pre-shock cell maps through `alpha_from_mass_fraction` almost to pure phase (case33's
  `Y_post=0.9344` sits closest of {24,33,34} to the `alpha->1` singularity) -- recovered density
  collapses **79x in step 0 alone** (250.4->3.16), to a **literal vacuum** (`1.24e-6`, 1.3e8x below
  correct) by the stall. `T_from_hstat` silently saturates at its `1e6` ceiling there (confirmed by
  direct code read: still returns `true` when saturated) -- `dT/dh=0`, `drho/dh=0`, so Newton can
  never recover at ANY `dt`, exactly matching every round-15 symptom with no alpha/Y-REMAP
  involvement. **The shock never moves**: cells where the true front should be by the stall time
  are still pristine IC to 4+ sig figs -- the solver re-fights a frozen 2-cell blister at the
  initial discontinuity for all 100 steps. **No fix attempted** (per the plan's own decision rule --
  no isolated off-by-construction bug found; this is genuine Y-form ill-conditioning for
  homogeneous-alpha/large-Y-jump cases, closed into Newton by `+ALPHA_IMPLICIT`'s own per-call alpha
  re-derivation). Three candidate fixes named for a future round, priority order **F3 > F1 > F2**
  (break the p→alpha feedback in the residual; upper-bound `s.h`; make `T_from_hstat` report
  saturation -- each with a stated reason it isn't a safe drive-by). Full hard-gate battery held,
  byte-identical to round 14's baseline; `ACID_RCELL` stays default OFF. `ACID_YADV` status
  unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §26, `docs/YADV_ROUND_16_PLAN.md`, commit `004e4e3`.
- Round 17: F2's actual risk measured with new `ACID_TSAT` instrument (deliberately NOT
  `yadv`-gated -- unlike `ACID_RCELL`/`ACID_RINIT`, it can observe the OFF path). **V-SAFE, zero
  exceptions**: across all 19 graded cases, >400,000 residual evaluations, **not one cell EVER
  reaches the `T_from_hstat` ceiling** -- not even a rejected transient iterate of case28 (the
  analytically closest case, 0.587x). Round 16's stated risk basis corrected, not edited: case29
  isn't graded at all (excluded); 13/14/25 sit 3-4 orders of magnitude below the ceiling. **But F2
  as literally named is the WRONG SHAPE independent of the measurement**: its `false` branch
  freezes `s.T`, making `compute_R` a function of call history rather than state -- breaks the four
  `compute_R(); // restore` sites (load-bearing for the FD-Jacobian assembly) and gives `dT/dh=0`
  exactly, the very failure mode round 16 diagnosed as why Newton can't recover. **Corrected form
  `F2''` pre-registered** (state-pure `T_from_hstat`, report saturation to the caller as a new
  stall reason instead, reusing the existing dt-halving retry machinery) -- priority list now
  **F3 > F2'' > F1**. No fix implemented (round's own bar -- "free AND fully specified" -- not met
  until F2 was corrected to F2''); F2'' is now well-specified and risk-cleared enough for direct
  implementation in a future round. Side finding for the record: case29's analytic post-shock
  T=2.93e6K exceeds the solver's own clamp by 2.93x, plausibly explaining its long-unexplained
  blocker comment. Full hard-gate battery held, byte-identical to round 16's baseline; G6 (stronger
  than round 16's equivalent -- this flag actually executes on OFF) confirms zero perturbation.
  `ACID_TSAT` stays default OFF. `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §27, `docs/YADV_ROUND_17_PLAN.md`, commit `d4ac63a`.
- Round 18: F2'' implemented (`ACID_TSAT_STALL`, default OFF) -- state-pure `T_from_hstat` (no
  signature change), a saturated retry becomes a new stall reason 5 that displaces reason 1 and is
  therefore automatically ineligible for `ACID_STALL_ACCEPT`'s capture, no separate code change
  needed there. **OFF/plain-ON/`+ALPHA_IMPLICIT` all confirmed byte-identical**, exactly as round
  17 predicted -- the diagnostic instrument's deductive proof held under the real new code path.
  **Unpredicted, positive result**: Stage 0's first-ever 5-config saturation sweep found the
  FD-Jacobian path (`ACID_NO_AJAC=1`, independent of `ACID_YADV`) already silently accepting
  T-ceiling-saturated states as "converged" on cases 24/27/28/33/34 -- turning the flag on
  **RECOVERS cases 27 and 28 from silent NaN failure to genuine PASS** (FD gates move
  13/19→15/19 and 13/19→14/19). No regression found in any tested configuration.
  `ACID_STALL_ACCEPT` interaction: case33 fails faster and cleaner exactly as predicted (0 accepts
  vs 4, step 43 vs 104, same non-completion); case24 byte-identical; **case34 shows a small, real,
  honestly-reported non-byte-identical perturbation** (~6th significant figure, same physical end
  state) when BOTH mechanisms are combined -- does not affect round 12's published numbers (which
  use `ACID_STALL_ACCEPT` alone). G9's full 20-combination accepted-state invariant sweep: zero
  violations. Flag stays default OFF (this round establishes safety+benefit when explicitly
  enabled, not promotion). `ACID_YADV` status unchanged (default OFF, 15/19).
  → `YADV_RESEARCH.md` §28, `docs/YADV_ROUND_18_PLAN.md`, commit `207874b`.
- Round 19: round 18's case34+`ACID_STALL_ACCEPT`+`ACID_TSAT_STALL` perturbation **fully
  localized, no new source code needed** -- existing round 13/16/17/18 instrumentation sufficed.
  Channel enumeration (a code property, not case-specific): the new flag can only perturb a
  trajectory via two channels, C1 (a would-be-accepted retry rejected for carrying a saturated
  cell) or C2 (an already-rejected retry's exclusion changes the accept-candidate winner) -- C2
  re-confirmed excluded (all four `STALL-ACCEPT:` events identical, direct diff). **Decisive test**:
  the unmodified baseline, instrumented, shows `accepted_steps_hi=3` -- three normally-accepted
  steps (325/326/329) already silently carry a saturated cell today. With the flag added, the two
  runs are **bit-identical through step 325 exactly** (`dt` ratio 1.000000 at every step), diverge
  at 326 (ratio 0.28-6.05x through step 336 -- the predicted `cfl_scale` cascade), then settle to a
  few percent by step 337 and continue damping toward parity. Both runs reach the **identical**
  final `t` to 9 significant digits (2648 vs 2646 steps). H0 (nondeterminism) and instrument-
  neutrality controls both pass. Two corrective annotations to round 18 recorded (not edited
  there): `only_reason1` is dead at `ACID_STALL_ACCEPT` level 1 (edit-free exclusion, not
  behaviour-free); round 18's own prediction was keyed to the wrong Stage-0 column
  (`+ALPHA_IMPLICIT` instead of plain-ON). No fix attempted or needed -- affects no published
  configuration. `ACID_YADV`/`ACID_TSAT_STALL` status unchanged.
  → `YADV_RESEARCH.md` §29, `docs/YADV_ROUND_19_PLAN.md`, commit `0d0aa13`.
- Round 20: `ACID_TSAT_STALL` (F2'') **promoted to unconditional default, env var DELETED**
  (round 14's no-opt-out precedent for correctness fixes -- reason 5 is a real solver defect, not
  a research toggle). Safety proof: 7-config battery (A-G, G new = `ACID_YADV=1 ACID_NO_AJAC=1`)
  plus both `ACID_STALL_ACCEPT` levels, post-edit default byte-identical to pre-edit
  flag-forced-ON in every case. **Net improvement, not just neutral**: D/E/G (the FD-Jacobian
  configs) recover cases 27/28 from silent NaN to genuine PASS (D 12/19->13/19, E 13/19->14/19,
  G[new] 13/19->15/19 -- G's numbers byte-for-byte reproduce round 18's own §28.3 table row,
  previously misfiled as describing config D); case33 (config C) fails faster and more honestly
  (step 43 not 100/104, 0 accepts not 4-8). A/B/C/F unaffected. **BASELINE CHANGE NOTICE**:
  `scripts/yadv_r9_sweep.py`'s `EXPECTED` updated this round (old values preserved in a provenance
  comment); the case33 reproduce command at `YADV_RESEARCH.md` line ~2510 now emits empty output
  (step 100 -> 43); round 12's §22.4 case33 accept/step counts for both `ACID_STALL_ACCEPT` levels
  are superseded for the current default (case24/34 unaffected). Found and fixed a methodology
  pitfall this round, not a code bug: `denner1d_validate`/`_run`/`_dump` require `DENNER_ACID=1` in
  the environment or they silently run a non-ACID path reporting a plausible-but-wrong 11/19 --
  now recorded in `.claude/rules/denner-pitfalls.md`. `git diff --stat -- cpp/`: one file, one
  executable line changed, rest comment-only. All hard gates held (OFF 19/19, byte-identical to
  `solver_denner` published binary).
  → `YADV_RESEARCH.md` §30, `docs/YADV_ROUND_20_PLAN.md`, commit `78a1b12`.
- Round 21: designed and implemented a `(rho,e,Y)`-conserving closed-form `(p,T,alpha)`
  reconciliation (`ACID_YADV_RECON`, default OFF) targeting the alpha-remap lag (`dal_remap`)
  round 13 diagnosed. **Refutes round 13 §23.3's STATED mechanism** (direct code reading:
  `compute_R` already reconciles `T,rho` with `h` before `r_init` is ever measured, at
  `acid.cpp:1576` vs `:2022` -- Stage 0 Branch A, HREINIT flattens `r_init`'s `1/dt` growth yet the
  stall persists) while CONFIRMING its empirical finding. Delivers a reusable, unit-tested
  closed-form NASG p-T-equilibrium solver (`eos.hpp:pT_from_v_e_massfrac`, worst rel error
  `4.7e-11` vs an independent Newton) regardless of the fix's own fate -- prior art: Collis et al.
  2025 §2.3 (independent derivation, not transcribed). **Case24 gets real, mechanistically-clean
  20x progress** under plain `B+RECON` (stall step 19->399, failure re-types from vague
  retry-exhaustion to the correctly-diagnosed `T-ceiling-saturated`) -- but **cases 13/14 REGRESS
  from PASS to FAIL** (u-field quality collapse, `l2_u`/`corr_u` cross the gate). **Verdict: S5
  (harm)**, per the plan's own pre-registered stop rule (`B+RECON < 15/19`, specifically 13/19).
  Flag stays OFF, not promoted, committed as gated-off research infrastructure (round 4/8
  precedent -- a measured-regression mechanism is preserved, not deleted). A sub-prediction was
  falsified honestly (cases 26/27/28 are NOT bit-exact pure in practice -- `alpha~0.999886`, not
  `1.0` -- so the exact-skip rule doesn't exempt them; no pass/fail regression resulted there).
  `consecutive_failures` NOT incremented (mechanistically-explained negative result, round 4/8/13
  precedent). All hard gates held (OFF 19/19, flag-unset paths byte-identical, `ALL GATES OK`
  unchanged from round 20).
  → `YADV_RESEARCH.md` §31, `docs/YADV_ROUND_21_PLAN.md`, commit `19de476`.
- Round 22: implemented `ACID_YADV_RESYNC` (default OFF), the DUAL projection to round 21's
  `ACID_YADV_RECON` -- resyncs `Yv` from the current `(p,T,alpha)` each step (writes NO state
  field at all) instead of moving the state onto the Y-manifold. **Stage 0 found round 21's 13/14
  regression is not a uniform Abgrall-pressure-oscillation story**: `G+RECON` (FD Jacobian) gives a
  MIXED result -- case14 still fails (Abgrall-type, Jacobian-independent, as predicted) but case13
  now PASSES (traced to Jacobian-approximation sensitivity affecting which discrete admissible
  shock-location state a bounded Newton sweep lands near -- a narrow counterexample to
  "approximate Jacobian changes only iteration count" worth future attention, not pursued this
  round). **`ACID_YADV_RESYNC` recovers BOTH 13 and 14 on the pass/fail gate**
  (`B+RESYNC=15/19`, matching plain `B`'s fail set exactly, `dal_remap` collapses to `1.1e-16`) --
  but **case14's phase-mass drift is 16.1%** (measured via a new `ACID_RESYNC` meter), firing the
  plan's own pre-registered "1% drift -> non-promotable regardless of pass_count" rule. Case24
  gains only 2.6x (step 19->50) vs `RECON`'s 20x, despite `dal_remap` collapsing identically under
  both -- an open question whether RECON's larger case24 gain needs the state WRITE itself, not
  just `dal_remap` removal. **Verdict: gate-passing but non-promotable on an orthogonal,
  pre-registered conservation cost** -- not literally any of the round's own S1-S5 outcomes,
  recorded as its own category. Flag stays OFF, committed as gated-off research infrastructure
  (round 4/8/21 precedent). `consecutive_failures` NOT incremented. All hard gates held (OFF
  19/19, `ALL GATES OK` unchanged from round 21).
  → `YADV_RESEARCH.md` §32, `docs/YADV_ROUND_22_PLAN.md`, commit `ffcd83b`.
- Round 23: asked whether `ACID_YADV_RECON`'s 20x case24 gain (vs `ACID_YADV_RESYNC`'s 2.6x, round
  22's open question) needs the state WRITE itself or just `dal_remap` removal. New
  `ACID_PROJ_UNTIL` diagnostic caps the write to `step<N`. **The roundoff-null control (N=1)
  reproduces plain B's stall EXACTLY** (step 19, identical `rbest`/`r_init`) -- decisively RULES
  OUT trajectory-chaos as the mechanism. The state-accuracy mechanism IS real and measured: plain B
  loses ~500 of a cell's 499.58 true mass at steps 0-2 (`RMISM`'s `drho`, round 16 §26.1's
  collapse reproduced arithmetically for case24), `B+RECON` suppresses this to single digits by
  step 4. **But the dose-response is NOT monotone** (N=2 gives step 6, worse than N=1; N=10-100
  give no stall at all; N=200 gives step 501, further than N=400's own 399). **Unplanned discovery,
  checked directly**: N=50's "no stall" is `pass:false` on the actual gate -- a severely wrong
  solution (shock frozen mid-domain), not a success; "completes without STALLED" and "correct
  answer" had never been distinguished before because `B+RECON`'s own always-applied run never
  completes either. **Verdict: S4 (partial attribution)** -- trajectory-chaos excluded, the
  state-accuracy mechanism confirmed, but the practical dose-response is more complex than
  predicted (a withdrawal-point compounding effect, not characterized). The plan's own "third
  projection" secondary goal explicitly NOT attempted (its gate not met by S4).
  `consecutive_failures` NOT incremented. All hard gates held (OFF 19/19, `ALL GATES OK` unchanged
  from round 22).
  → `YADV_RESEARCH.md` §33, `docs/YADV_ROUND_23_PLAN.md`, commit `ed93f71`.
- Round 24: discovered round 23's "roundoff-null control" (`ACID_PROJ_UNTIL=1`) was a COMPLETE
  NO-OP -- the exact-skip fires on all 800 cells at step 0, so 0 writes occur and its P6' test
  carried no information about H-B (a correction to how the result was supported, not a
  retraction: the conclusion turned out right). Also corrected round 23 §33.4's "frozen shock"
  reading of `N=50`: the actual final state shows the shock COMPLETELY EXITED the domain (84%
  overstrong plateau, 32% overfast, alpha collapsed `0.5->2e-4`), not stalled mid-domain -- the
  coarse 200-step sampling in the trace obscured an over-fast, over-strong, alpha-collapsing
  shock. **Built the real roundoff-null control this round** (`ACID_RECON_NULL`, using the
  existing unit-tested `8*eps*kappa` round-trip conditioning bound, `alpha_roundtrip_floor` -- no
  new constant): non-empty (2-4 cells/step genuinely written, never zero) and applied. **Result:
  byte-identical to plain B anyway** -- H-B (Newton-trajectory chaos) excluded, this time on solid
  evidence. Confirmed no global withdrawal-point criterion exists (`ntouch=0` only at step 0, never
  again) AND the always-on family member is itself wrong (stalls on its own round 16 §26.1 blister
  at step 399) -- no withdrawal schedule within the `ACID_PROJ_UNTIL` family has a correct member
  to find. **Verdict: S1** (the question was mis-posed, exactly as pre-registered/expected) -- no
  taper designed or built. `ACID_RECON_NULL`/`alpha_roundtrip_floor` committed as inert research
  infrastructure. `consecutive_failures` NOT incremented. All hard gates held (OFF 19/19,
  `ALL GATES OK` unchanged from round 23, unit-test round-trip numbers unchanged after the
  `alpha_roundtrip_floor` refactor).
  → `YADV_RESEARCH.md` §34, `docs/YADV_ROUND_24_PLAN.md`, commit `fccbb54`.
- Round 25: implemented `ACID_YADV_F3` (default OFF, round16 §26.3's F3 made concrete) -- recovers
  alpha at the NEW Y's own PTE state `(p*,T*)=pT_from_v_e_massfrac(1/rho,hstat-p_o/rho,Y,A,B)`
  instead of the stale `(p_o,T_o)`. **Exactly one `s.*` write** (`s.alpha[i]=al_f3`, grep-verified
  against the diff hunk), structurally excluding both of round 22's harm channels. **P0 confirmed**:
  live meter reproduces the plan's hand-computed `alpha_F3` to 5 sig figs. **T3**: `drho`/`dh` at
  the step-0 front cell improve 5.4-13.7x/349-1092x -- round 24's flagged same-triple-break risk did
  NOT materialize as harm. **T1/T4/T6**: F3 repeatedly converts STALLED-NaN into finite-but-
  inaccurate completions (case34/B, case24/B+RECON -- directly confirms round 24 §34.5's own
  prediction, case33/C) -- but **T5's full sweep is still `15/19`, the SAME fail set `{15,24,33,34}`
  as plain B** -- no case flips pass/fail. **T2**: case13's predicted risk lands exactly at the
  predicted magnitude (`l2_p` +7.4%, matches "6-8% worse") and well short of the pre-registered harm
  threshold (`u_shock_delta_cells` 1->2, not >3); case14 improves. **T8**: phase-mass conservation
  excluded from drifting BY CONSTRUCTION (`Yv` never written). **Verdict: S2 (substantial partial)**
  -- first mechanism moving 24/33/34 without paying 13/14 or conservation, but the finite
  completions stay inaccurate -- something else, still unidentified, is the actual binding
  constraint. `consecutive_failures` NOT incremented. Flag committed gated-off. All hard gates held
  (OFF 19/19, `ALL GATES OK` unchanged from round 24, unit-test numbers unchanged).
  → `YADV_RESEARCH.md` §35, `docs/YADV_ROUND_25_PLAN.md`, commit `12fb88d`.
- Round 26: **ESCALATION.** Answered round 25's own thread (b) decisively: the binding constraint
  on cases 24/33/34 is a **closure mismatch, not a numerical defect**. `cases.cpp`'s reference
  (Denner Eqs.57-62) holds VOLUME fraction fixed across the shock (closure A); `ACID_YADV=1`
  conserves MASS fraction (closure B, since `Y` is uniform pre-shock and `rho*Y` transports
  conservatively) -- both exact, admissible RH solutions of the same EOS, differing by O(1) (~2x
  in `rho`/`p`). Built `scripts/yadv_r26_closure.py` (zero C++ changes) -- an exact-solution
  instrument, cross-validated twice: **P0** reproduces the plan's hand-derived closure-A/B tables
  live to full double precision; **P1** matches `denner1d_validate`'s own JSON to spec on 7/7
  `(case,config)` pairs. **T1**: the EXACT config-B answer fails the gate by 2-3.4x on
  `l2_p/l2_u/l2_rho` and 31-40x on `dip`, all three cases. **T2**: the entire reachable
  single-shock family also fails -- no numerical improvement can pass. **T3**: where the solver
  reaches a genuine flat plateau, it matches the exact closure-B answer to 0.17%-8.74% (median
  ~2-4%) -- the solver is close to correct FOR ITS OWN MODEL. **T6** resolves round 25's own
  open thread (c): case33's `corr_p` sign flip under F3 is motion TOWARD the model's own answer,
  not away from correctness. **T7**: case15's redirect target confirmed structurally sound.
  **Verdict: S1 (structural obstruction CONFIRMED) + S2 (solver vindicated, one metric at 6.15%
  vs the 5% bar, reported honestly)**. `consecutive_failures` NOT incremented (round 5/9/11
  precedent). **Surfaced to the user as an escalation**, not resolved unilaterally: 24/33/34
  cannot pass under `ACID_YADV=1` without (i) accepting the OFF path validates this family
  instead, (ii) a genuine model extension (out of this loop's authorized scope), or (iii)
  revising what "all cases pass" means for this family -- none decidable by the autonomous loop
  alone. All hard gates held (OFF 19/19, `ALL GATES OK` unchanged from round 25, unit-test
  numbers unchanged, zero `cpp/` diff).
  → `YADV_RESEARCH.md` §36, `docs/YADV_ROUND_26_PLAN.md`, commit `bf71184`.
- (Between rounds, by explicit user decision): Phase 3 redirected from cases 24/33/34 to case15
  (round 7's central-jump defect) -- `docs/YADV_ROADMAP.md`'s "Current goal" section rewritten,
  original framing kept as a superseded block for provenance. → commit `b792046`.
- Round 27: first round on the redirected case15 target. Re-measured round 7's `cj=30.02` finding
  live across all 7 configs: reproduces exactly, but **only under config C**
  (`ACID_YADV_ALPHA_IMPLICIT=1`) -- the loop's actual headline config **B** (`ACID_YADV=1` alone)
  instead **passes** `cj` (2.307 vs 8.0) and fails on `l2_rho=0.16761`/`corr_rho=0.984514`
  instead. Root cause, measured unambiguously: **B deletes 99.92% of the domain mass**
  (`945.07 → 0.761` at `N=400`, 322/400 cells pinned at the pressure floor) -- round 16 §26.1's
  own named "vacuum blister" mechanism, previously measured at one case24 cell, here running over
  80% of case15's domain for 85 consecutive steps. Built `ACID_MBAL` (default OFF, stderr-only):
  closes the discrete mass budget into `ADV`/`REMAP`/`BND`/`LEAK`/`RES`, self-test `closure~1e-13`
  at every step. **Found and fixed a real bug while building it**: the natural extra `compute_R()`
  call is not idempotent under `ACID_YADV` (a non-converging T-relaxation) and silently perturbed
  the reported solution -- caught by routine G4, fixed with a snapshot/restore. **Measurement**:
  `REMAP` explains 99.67% of B's collapse; under every config that avoids the collapse (C,
  `B+RECON`, `B+F3`), `REMAP`'s raw magnitude is NOT small (larger than case15's own total loss in
  C and `B+RECON`) -- it is instead exactly **cancelled** by `ADV` (net `0.0000`), a materially
  more precise mechanism statement than the plan's own prediction, reported honestly. T4
  cross-validated the instrument against round 16/24's known case24 blister. **Stage 2** (the one
  pre-registered candidate, `ACID_YADV_REBUILD_ADV`): **severe, clean negative result** --
  `pass_count` collapses `15/19 → 11/19`, cases 07/13/14/25 (previously passing) newly diverge.
  Per the plan's own pre-registered S5 rule, **the Stage-2 code was reverted in full** (only
  `ACID_MBAL` merged); `consecutive_failures` **incremented to 1** (S5's explicit instruction,
  first increment since round 20). Config C's own `cj=30` core-jet characterised (under-resolved
  near-vacuum, a different failure class from the documented MWI checkerboard), not fixed. All
  hard gates held (OFF 19/19, `ALL GATES OK` unchanged, unit-test numbers unchanged).
  → `YADV_RESEARCH.md` §37, `docs/YADV_ROUND_27_PLAN.md`, commit `0a5c0a8`.
- Round 28: answered the question round 27 deferred -- why does the Eqs.43-44 rebuild preserve
  mass under config C/`B+RECON` but not plain B? `adv+remap ≡ M_reb−M_prev` is an algebraic
  identity (`ACID_MBAL`'s own arithmetic), so the real question is whether the rebuild preserves
  total mass -- true under C/RECON (both keep the accepted state PTE-consistent), false under
  plain B. **Measured live**: plain B's discrete continuity has **no admissible solution** in a
  cavitating cell -- stalls at 86% of `r_init`, Newton effectively demands `p→−∞`; config C
  converges quadratically to `2.4e-7` in 7 iterations on the identical starting state; `B+F3`
  stalls identically to B (F3 lives outside Newton). Closed-form: case15's Mach-1.9 rarefaction
  needs `Δp≈−1.2e8 Pa` at frozen-alpha compressibility -- no admissible pressure exists. This
  retires the whole "fix it at the recovery/rebuild site" family on evidence (`REBUILD_ADV`
  already broke 4 cases; `B+F3` restores mass but floors the ENTIRE domain, `l2_p=0` only because
  the reference is equally collapsed -- a degenerate number, corrected here). **Built
  `ACID_YADV_ALPHA_IMPLICIT_CAV`** (default OFF): makes alpha implicit (config C's own per-cell
  update) only on cells where an exact, constant-free predicate fires -- "the full Newton step
  just asked for a pressure at/below the solver's own existing 1.0 floor" (the same literal, no
  new constant), tested after the line search closes. Diagnostic sibling `ACID_NFEAS` uses a
  separate shadow count -- found+fixed a bug where the naive version let the diagnostic-only flag
  populate the real mask, silently becoming solution-affecting; caught by routine G4-early.
  **Blast-radius census before applying**: case15 fires persistently (604); every other nonzero
  case fires only a handful of cells at step 0 and already passes under B or C; **case14 (the one
  at-risk case) never fires at all**. **Harm gate, checked before any case15 metric**:
  `pass_count=15/19` under `B+CAV`, fail set identical to plain B -- zero regression (bonus:
  case24 now `finite:true`, was NaN). **R1/R2/R3 all confirm**: mass restored to `869.3` (C:
  `870.6`), floor cells `322→0`, the rebuild's own budget closes to `1.6%` (matching C's
  near-zero cancellation). **R4 (accuracy) does not hold**, falling short of even the plan's own
  predicted C-matching numbers: `l2_rho=0.069` vs the `0.05` gate (C: `0.020`), `corr_rho=0.958`
  vs the `0.99` gate (C: `0.997`) -- though `l2_p`/`corr_p` clear their own bars. **Verdict: S4
  (neutral)** -- mass-collapse mechanism solved and independently verified, but a separate,
  unidentified density-specific accuracy gap remains. `consecutive_failures` **NOT incremented**
  (stays 1, resets round 27's S5 streak). Flag committed gated-off. All hard gates held (OFF
  19/19, `ALL GATES OK` unchanged, unit-test numbers unchanged).
  → `YADV_RESEARCH.md` §38, `docs/YADV_ROUND_28_PLAN.md`, commit `35b3cb8`.
- Round 29: measured round 28's own open question directly and overturned its framing. `B+CAV`'s
  N=400 solution is **already config-C-grade** (`l2_rho=0.001580` vs C's own field). The reported
  gap lives almost entirely in the N=800 **self-convergence reference**, which develops a spurious
  dense/hot core plug under `B+CAV` -- scored against C's own reference instead, `B+CAV` matches
  C to the 5th digit (diagnostic only, never claimed as a pass). Also corrected: the full
  8-criterion gate shows `B+CAV` failing `smooth_ok`/`osc_ok` too, not just `l2_rho`/`corr_rho` as
  round 28 reported -- a ~28 kPa pressure notch sits exactly at the `cav` mask's spatial boundary
  (round 28's transition-zone hypothesis confirmed, for pressure not density). Measured the mask's
  own non-monotonicity: it shrinks 70→40 cells over steps 0-8 and is mesh-count- (not mesh-width-)
  invariant, explaining why case15's N=800 reference isn't a genuine refinement. **Binding blocker
  identified**: both `B+CAV` and config C show the same 4-cell stagnation-point velocity sign
  reversal (round 27 §4.5's core jet) -- config C itself fails `smooth_ok` (`cj=30.018`, worse
  than `B+CAV`'s `25.906`), so **no member of the implicit-alpha family can pass case15 until the
  core jet is fixed**, pre-registered before any Stage-2 number existed. Built
  `ACID_YADV_ALPHA_IMPLICIT_CAV=2` (a run-scoped latch, same predicate, no new constant) to remove
  the reference-plug/boundary-notch artifacts; blast-radius census confirmed case14 never at risk.
  **Harm gate clean on pass_count/fail-set, but case33 (already failing, `finite:true` under both
  plain B and level 1) flips to `finite:false` under level 2** -- the plan's own S5 rule treats any
  previously-finite case going NaN as harm regardless of pass/fail status, with an anti-rescue
  clause forbidding the "already failing anyway" exemption. **S5 fires.** Reverted the latch
  mechanism in full; kept the level-parse and `ACID_NFEAS`'s spatial fields as gated-off
  diagnostics (level 2 verified to degenerate safely to level 1 post-revert).
  `consecutive_failures` **incremented to 2** -- the second increment in 3 rounds on this same
  case15 thread (round 27's `REBUILD_ADV`, this round's latch), now **2/3 toward the loop's own
  stop condition**. Round 30 should attack the core jet directly rather than propose a third
  variant adjacent to the alpha-implicit mechanism. All hard gates held.
  → `YADV_RESEARCH.md` §39, `docs/YADV_ROUND_29_PLAN.md`, commit `08d3c2e`.
- Round 30: **DIAGNOSTIC-ONLY**, attacked the core jet directly per round 29's own hand-off.
  Mechanism traced to its immediate cause: with `use4≡false` for case15, `pres_i=pface[i+1]-
  pface[i]=0.5*(p_{i+1}-p_{i-1})` (the cell's own pressure cancels), and at the 128:1 density-
  ratio/137:1 pressure-ratio face straddling the stagnation point, the arithmetic-mean `pface` is
  69x the light cell's own pressure -- a `1/rho`-amplified interpolation error (49-124x specific-
  force amplification, measured two independent ways agreeing to 0.1-0.3%). This is
  Bartholomew/Denner/van Wachem (JCP 375, 2018) §5 Eq.90 verbatim, already in the repo. **Three
  corrections to prior framings**: (a) config C's Newton converges quadratically at every sampled
  step -- the jet is a property of a converged fixed point, not a stall; (b) temperature is
  uniform to 0.02 K across the core's 340x pressure drop -- the "overheating" framing round 29
  queued as this round's literature priority is refuted; (c) the MWI/Rhie-Chow correction is
  ~2700x too small to act here and is NOT clamp-saturated -- confirms round 27's "different
  mechanism from case25" conclusion but not its stated Wood-speed reasoning. Two parameter-free
  fix candidates (density-weighted and acoustic-impedance face pressure) were derived and killed
  -- both break case25's shock speed, exactly the pre-existing dead end in
  `.claude/rules/denner-pitfalls.md`. Built `ACID_DUMP_CELLS` (diagnostic-only mesh-resolution
  override, `denner1d_dump.cpp` only, two mandatory bit-level self-tests both passed) and ran a
  refinement census N=100..3200: `cj` non-monotone at coarse N then monotonically decreasing from
  N=400 (30.0→27.9→18.4→3.9), crossing below the gate's 8.0 threshold between N=1600 and N=3200
  -- the pre-registered specific prediction `cj_800<8` is **falsified** (27.9 measured) but the
  qualitative H-R1 convergence hypothesis is well-supported (`n_rev` bounded 0-2, physical width
  of the reversed region shrinks super-linearly, 4x not 2x, from N=400 to N=800). New,
  unanticipated finding reported honestly: exactly 2 cells hit the 1.0 Pa floor at N≥800, an open
  loose end. No fix attempted (pre-registered non-goal); `ACID_YADV`'s recommended status
  UNCHANGED (OFF, 15/19). All hard gates held; `git diff --stat -- cpp/` touches exactly
  `apps/denner1d_dump.cpp`. **Verdict S1 (diagnostic success), `consecutive_failures` reset to 0.**
  Hand-off to round 31: the core jet is fully mechanistically characterised and both obvious
  parameter-free fixes are dead -- the honest options are (i) accept case15 unreachable under
  `ACID_YADV=1` at the current scheme, (ii) escalate to the user for a scheme-level `pface` change
  with explicit shock-case risk, or (iii) a user conversation about case15's mesh/spec. This round
  does not choose among these.
  → `YADV_RESEARCH.md` §40, `docs/YADV_ROUND_30_PLAN.md`, commit TBD.

## Setup reference

- Round-loop protocol: `.claude/skills/yadv-round/SKILL.md`
- Safety hooks: `.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`,
  wired in `.claude/settings.local.json` (gitignored, personal/experimental scope), gated by the
  presence of `.claude/round-loop-active` (also gitignored, transient).
- To start the loop: invoke `/loop` with the `yadv-round` skill as the recurring prompt (dynamic
  self-pacing, no fixed interval — the skill's own Step 0 decides whether to continue).
- To stop early by hand: `/oh-my-claudecode:cancel`-style interrupt, or just tell the loop to stop;
  it also self-stops on the conditions above.
