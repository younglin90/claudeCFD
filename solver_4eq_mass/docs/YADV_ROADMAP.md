# YADV Round-Loop Roadmap

State file the `yadv-round` skill (`.claude/skills/yadv-round/SKILL.md`) reads/writes every round.
Detailed research log stays in `docs/YADV_RESEARCH.md`; this file is a thin, compact control
document — round counter, stop conditions, next-task pointer, one-line-per-round history. Do not
put derivations or measurement tables here; put them in `YADV_RESEARCH.md` (or a per-round plan
doc) and link to them.

## Current goal — ACHIEVED, loop idle

Phase 2 (`docs/YADV_PHASE2_PLAN.md`, Stages 0-4) is COMPLETE as of round 9. Re-scoped goal
(`pass_count >= 14` under `ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` with the default analytic
Jacobian, cases 13 and 25 durably recovered) was met at round 6 and re-verified at rounds 7, 8 and
9 with zero drift. Stage 4's consolidation, timing measurement and promotion decision are in
`YADV_RESEARCH.md` §19: `ACID_YADV_ALPHA_IMPLICIT` does NOT fold into `ACID_YADV` (it would cost
case14, which plain `ACID_YADV=1` passes); both stay default OFF.

`done: true` — the loop is idle pending a human decision on whether to open a Phase 3. Two named,
scoped candidates, both DIFFERENT defect classes from Phase 2's Jacobian work:

- **(P3a) The cases-24/33/34 conservation defect — NOW WITH A CONCRETE, PROMISING LEAD (round 9,
  §19.4).** Round 3's conservative `rho*Y` transport brought 24/34 to 1e-13 RH closure but all
  three still fail their validation gates. Round 9's post-merge RH re-check under
  `ACID_YADV_ALPHA_IMPLICIT=1` (predicted by the round's own plan to show "no movement" — that
  prediction was WRONG) found: **case33's Rankine-Hugoniot jump closes to machine precision**
  (momentum residual 88% → 8.4e-13) — Stage 1's Jacobian fix repairs its conservation
  self-consistency even though it still disagrees with the alpha-held validation reference (a
  different, also-legitimate closure choice per §11.3, not a "solver defect" in the §11.6 sense).
  case24/34 instead show their shocks exiting the domain before `t_end` under the same flag —
  not yet understood whether this is the same phenomenon further along (a faster, still-admissible
  shock) or a different problem specific to those two. This is a materially better starting point
  than "three rounds of Jacobian work moved nothing," which is what rounds 4-8 (measuring only
  validation-gate metrics, never the RH self-consistency under implicit alpha) implied. Decisive
  instrument already exists and needs no new code: `scripts/yadv_rhcheck.py` /
  `ACID_YADV_ALPHA_IMPLICIT=1 python3 scripts/yadv_rhcheck.py`.
- **(P3b) case15's central-jump defect** (§17.4). `cj=30.02` vs threshold `8.0` at the
  stagnation point, oscillation side clean. A collocated-scheme/MWI question. Narrower and
  better localized than P3a, but it would not change `ACID_YADV`'s status.

To re-arm the loop: set `done: false`, write the chosen goal here (P3a is the Advisor's
recommendation given the round-9 finding above), set `next_task` to a new plan document, and
reset `consecutive_failures: 0`.

## Control state

```
round_counter: 9
consecutive_failures: 0
done: true
next_task: (none -- loop idle, see "Current goal" above for P3a/P3b candidates)
```

(Round counter starts at 4 because rounds 1-4 of the `ACID_YADV` experiment were already run
manually, before this loop existed — see `docs/YADV_RESEARCH.md`. Round 5 onward runs under
`yadv-round`.)

## Stop conditions (checked at the start of every round, before any work happens)

1. `done == true`
2. `consecutive_failures >= 3` — three rounds in a row with no measured progress or a hard-gate
   failure. This means the current approach needs human reconsideration, not another autonomous
   attempt at the same thing.
3. `round_counter >= 100000` (nominal cap; `consecutive_failures` is the real backstop)

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

## Setup reference

- Round-loop protocol: `.claude/skills/yadv-round/SKILL.md`
- Safety hooks: `.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`,
  wired in `.claude/settings.local.json` (gitignored, personal/experimental scope), gated by the
  presence of `.claude/round-loop-active` (also gitignored, transient).
- To start the loop: invoke `/loop` with the `yadv-round` skill as the recurring prompt (dynamic
  self-pacing, no fixed interval — the skill's own Step 0 decides whether to continue).
- To stop early by hand: `/oh-my-claudecode:cancel`-style interrupt, or just tell the loop to stop;
  it also self-stops on the conditions above.
