# YADV Round 12 Plan — Phase 3a Stage 3: retry-exhaustion policy (branch 3a), evidence-selected

Planner output (Agent, subagent_type="Plan", model=opus), round 12. Status: 3a CONFIRMED as the
correct branch; 3b's stated premise REFUTED; but 3a as literally scoped in `YADV_PHASE3_PLAN.md`
is shown below to be INSUFFICIENT for this round's real goal, and the plan widens it by one
precisely-justified step.

Advisor spot-check (round 12 session, before implementation): independently re-`grep`'d every
line-number claim (`bool bad = ...`=2112, `s = s_best`=2093, `if (!stepped) {`=2153,
`stall_reason`/etc block=694-701, env-read anchors `cfl_ramp`=520/`yadv`=550/`alpha_implicit`=565)
against the current worktree HEAD -- all exact. Independently re-ran the decisive `ACID_RHIST=1
ACID_BLK_STEP=19` diagnostic on case24 plain and confirmed the `r_init` doubling sequence for
retries 6-13 (2.5492e11 -> 3.9117e11 -> 7.2446e11 -> 1.4323e12 -> 2.8701e12 -> 5.7566e12 ->
1.1535e13 -> 2.3095e13, successive ratios 1.53/1.85/1.98/2.004/2.006/2.004/2.002) matches the
Planner's reported table exactly -- the core §1.2 finding is real, not fabricated.

---

## 1. What was verified (all re-run in this worktree, no rebuild needed)

### 1.1 The signature reproduces exactly

| config | result |
|---|---|
| case24 plain | `STALLED ... step 19, t=2.986e-07 of 1.089e-04`; `reason=newton-no-progress cell=-1 rbest=2.7939e+13 r_init=2.3095e+13`; `max\|u\|` pinned at 1.027e+04 across all 14 retries |
| case34 plain | `STALLED ... step 229, t=3.850e-07 of 8.535e-05`; `reason=newton-no-progress cell=-1 rbest=3.3695e+13 r_init=1.1896e+13` |
| case33 `+ACID_YADV_ALPHA_IMPLICIT=1` | `STALLED ... step 100, t=2.396e-06 of 1.283e-04`; `reason=newton-no-progress cell=-1 rbest=6.5145e+14 r_init=4.3403e+14` |

`cell=-1` in all three: no non-finite cell, no `|u|>10*uref` cell, ever, at any retry.

### 1.2 The decisive measurement -- `r_init` grows as 1/dt

`ACID_RHIST=1 ACID_BLK_STEP=19` on case24 plain, per-iteration residual history at the stalling
step (retry blocks 6-13 shown; independently reproduced by the Advisor):

| retry | dt | `r_init` (= `n0` at `it=0`) | ratio vs prior retry |
|---|---|---|---|
| 6 | 8.362e-13 | 2.5492e+11 | -- |
| 7 | 4.181e-13 | 3.9117e+11 | 1.53 |
| 8 | 2.090e-13 | 7.2446e+11 | 1.85 |
| 9 | 1.045e-13 | 1.4323e+12 | 1.98 |
| 10 | 5.226e-14 | 2.8701e+12 | 2.004 |
| 11 | 2.613e-14 | 5.7566e+12 | 2.006 |
| 12 | 1.306e-14 | 1.1535e+13 | 2.004 |
| 13 | 6.532e-15 | 2.3095e+13 | 2.002 |

`r_init` is flat for retries 0-4, then **exactly doubles for every halving of dt from retry 6
onward** -- the Newton problem gets strictly and unboundedly worse as dt shrinks. `r_init` is the
residual at `it==0`, i.e. of the state *before any Newton work*. A residual that scales as `1/dt`
at `it==0` means there is a **dt-independent mismatch between the state handed to Newton and the
`_o` conserved levels it is measured against**, entering the transient term as `Delta*dx/dt`. That
mismatch is injected by the pre-Newton explicit block (Y advection -> alpha recovery at `(p_o,T_o)`
-> the `rho_o`/`hstat_o`/`Htot_o` re-evaluation, `acid.cpp:925-955`), and `alpha` is NOT a Newton
unknown -- so Newton must absorb an O(1) state jump entirely through `(u,p,h)`, and the required
correction grows as `1/dt`.

The line search gives up at `al=0.016` at every single iteration in every late retry (the
`|| al < 0.03` escape at `acid.cpp:1858`), accepting a state it knows is worse than `n0`; the
residual then grows monotonically iteration over iteration.

### 1.3 The control: the OFF path is fine

`denner1d_dump 24` (no `ACID_YADV`) completes: `ACID done case=24 step=1732 t=1.089195600e-04 of
1.089195600e-04`, only 10 `RETRY` lines in the whole run, steady `dt~6.36e-08`. So `dt_full ~
6.36e-8` at `cfl_scale=1` for case24, and the stall is entirely YADV-induced.

### 1.4 `cfl_scale` is already pinned at its floor

At case24's stalling step, retry-0 `dt = 5.351e-11` against the OFF-path `dt_full ~ 6.36e-8` is
`cfl_scale ~ 8.4e-4` -- already sitting on the `1.0e-3` floor (`acid.cpp:2141`) after only 19
steps. Same for case34/33. Steps-to-`t_end` at these dt: **~2.0-2.8 million steps** for all three.
**Removing the stop is therefore necessary but not sufficient** -- reaching `t_end` also requires
`cfl_scale` to escape the floor, which requires reason-1 failures to stop driving the ramp down.

---

## 2. Branch decision: 3a confirmed, 3b's premise refuted, 3c not selected

**3b is refuted as stated.** `YADV_PHASE3_PLAN.md` scopes 3b as "if the void cell precedes Newton
failure: the Y->alpha recovery is collapsing to alpha->1". The antecedent is false in all three
configurations: `cell=-1` at every stall, `max|u|` numerically frozen across a 16000x dt sweep. No
void cell, no `rho->0`, no over-speed cell. Do not implement 3b. (§1.2's `1/dt` finding does
implicate the pre-Newton explicit alpha/Y block as the SOURCE of the inconsistency -- that's 3b's
*file*, not 3b's *mechanism*; a future "3b'" targeting the operator-splitting inconsistency is a
real candidate, see §7a below -- not the round-10 formulation.)

**3a is confirmed, with a mechanism, not just a correlation.** The `bad` gate's design intent
(comment at `acid.cpp:2107-2111`, "dt-retry ... ONLY when it made NO progress ... that means dt is
too large") is FALSIFIED for this failure mode: dt is not too large; halving it makes `r_init`
strictly worse from retry 5 onward, and the best achievable progress ratio (1.039, at retry 0, the
LARGEST dt) degrades monotonically with every halving. Retrying is actively counterproductive here.

**3c not selected** (needs Advisor sign-off). Recommendation in §5.

**Correction to 3a's assumed scope**: `s_best` is NOT "tracked but never consumed". `acid.cpp:2093`
already does `if (ajac && coupled && !conv_inner && best_it >= 0) s = s_best;` unconditionally,
before the `bad` evaluation -- confirmed by direct read. So the state `s` reaching the `bad` check
is ALREADY the best iterate of that retry. 3a needs NO new keep-best machinery inside the Newton
loop -- only keep-best ACROSS retries, plus the decision to stop rejecting it. Smaller, lower-risk
change than the original Phase3Plan assumed. `stall_reason==1` after the cell scan is precisely the
eligibility predicate 3a needs ("failed on newton-no-progress AND finite AND speed-bounded"), for
free from round 11's instrumentation.

---

## 3. The change

One policy: **`newton-no-progress` is not a timestep problem, so stop treating it as one.**

New env `ACID_STALL_ACCEPT` (RESEARCH-ONLY, default 0 = current behaviour exactly):
- `0`/unset -- byte-identical to today.
- `1` -- Stage 3a-i: on retry exhaustion, adopt the best-across-retries eligible state instead of
  breaking.
- `2` -- 3a-i plus 3a-ii: a step that succeeded after only reason-1 retries does not collapse
  `cfl_scale`.

New env `ACID_STALL_ACCEPT_MAX` (default 4) -- consecutive-accept budget.

### 3.0 Hard requirement: byte-identity of the default path

Every added statement sits inside `if (stall_accept_lvl > 0)`. No new FP arithmetic executes when
the env is unset.

### 3.1 Declarations -- anchor `acid.cpp:520` region (env reads)

```cpp
const int stall_accept_lvl = []{ const char* e = std::getenv("ACID_STALL_ACCEPT");
                                 return e ? std::max(0, std::atoi(e)) : 0; }();
const int stall_accept_max = []{ const char* e = std::getenv("ACID_STALL_ACCEPT_MAX");
                                 return e ? std::max(0, std::atoi(e)) : 4; }();
long n_stall_accept = 0;
int  stall_accept_run = 0;
```

### 3.2 Per-step candidate state -- anchor after `stall_rinit` decl (`acid.cpp:701`)

```cpp
bool   acc_have  = false;
Field  acc_s;
Vec    acc_Yv;
double acc_dt    = 0.0, acc_ratio = 0.0, acc_rbest = 0.0, acc_rinit = 0.0;
int    acc_retry = -1;
bool   only_reason1 = true;
```

### 3.3 Candidate capture -- anchor between the `!bad` block's closing `}` (after `break;`, i.e.
right after the block ending at `acid.cpp:2144`) and `if (dbg) { ... RETRY ... }`

```cpp
if (stall_accept_lvl > 0) {
    if (stall_reason != 1) only_reason1 = false;
    if (stall_reason == 1 && r_init > 0.0 && std::isfinite(rbest)) {
        const double ratio = rbest / r_init;
        if (!acc_have || ratio < acc_ratio) {
            acc_have = true; acc_s = s; acc_Yv = Yv;
            acc_dt = dt; acc_retry = retry;
            acc_ratio = ratio; acc_rbest = rbest; acc_rinit = r_init;
        }
    }
}
```

### 3.4 Level-2 CFL neutrality -- anchor the `if (cfl_ramp) { ... }` block inside `if (!bad) { ... }`
(`acid.cpp:2139-2142`)

```cpp
            if (cfl_ramp) {
                const bool r1_only = (stall_accept_lvl >= 2 && retry > 0 && only_reason1);
                if (retry == 0)   cfl_scale = std::min(1.0, cfl_scale * 1.5);
                else if (!r1_only) cfl_scale = std::max(1.0e-3, cfl_scale * std::pow(0.5, retry));
            }
```

Also add (guarded) `stall_accept_run = 0;` in the same `!bad` block -- a clean step resets the
consecutive budget.

### 3.5 The accept -- anchor immediately before `if (!stepped) {` (`acid.cpp:2153`)

```cpp
if (!stepped && stall_accept_lvl > 0 && acc_have && stall_accept_run < stall_accept_max) {
    s = acc_s; Yv = acc_Yv; dt = acc_dt;
    for (int i = 0; i < n; ++i) {
        mom_o2[i] = rho_o[i] * u_o[i];
        rho_o2[i] = rho_o[i];
        ene_o2[i] = rho_o[i] * Htot_o[i];
    }
    have_o2 = false;   // do NOT build a BDF2 level on a non-converged state
    dt_prev = dt;
    if (cfl_ramp) {
        if (acc_retry == 0) cfl_scale = std::min(1.0, cfl_scale * 1.5);
        else if (stall_accept_lvl < 2)
            cfl_scale = std::max(1.0e-3, cfl_scale * std::pow(0.5, acc_retry));
    }
    ++n_stall_accept; ++stall_accept_run;
    std::fprintf(stderr,
        "STALL-ACCEPT: case=%s step %d t=%.3e -> accepting non-converged retry %d dt=%.3e "
        "(rbest=%.4e r_init=%.4e ratio=%.4f) run=%d/%d total=%ld\n",
        c.id.c_str(), step, t, acc_retry, acc_dt, acc_rbest, acc_rinit, acc_ratio,
        stall_accept_run, stall_accept_max, n_stall_accept);
    stepped = true;
}
```

Falls through to the existing `theta_o`/`t += dt; ++step;` code unchanged.

### 3.6 End-of-run report -- anchor immediately before the `if (dbg)` "ACID done" print

```cpp
if (n_stall_accept > 0)
    std::fprintf(stderr,
        "STALL-ACCEPT-TOTAL: case=%s accepted %ld non-converged step(s) "
        "(ACID_STALL_ACCEPT=%d max_run=%d) -- this run is NOT a clean solve\n",
        c.id.c_str(), n_stall_accept, stall_accept_lvl, stall_accept_max);
```

Unconditional (not `dbg`-gated), matching round 11's `STALLED:` precedent.

---

## 4. Acceptance criterion, argued

No absolute residual threshold is derivable (residual magnitudes span 2.05e+11 to 6.51e+14 across
cases, no case-independent scale). The dimensionless ratio `rbest/r_init` IS used, but only for
RANKING candidates, not as an accept/reject gate -- inventing a numeric threshold from n=1 properly-
measured data point (1.039, case24 retry 0) would be exactly the kind of unjustified constant this
project avoids. **Decision: implement with NO ratio gate, print the ratio on every acceptance, and
pre-register the read**: if all three cases' accepted ratios are O(1) (<2), no-gate is confirmed; if
any is >=10 that's a qualitatively different (diverging, not stalling) regime needing a future gate.

Safety instead comes from: (1) per-retry eligibility already guarantees finite + speed-bounded: (2)
the next step's own finite/`10*uref` checks catch a bad accept immediately; (3) the pre-existing
CFL-collapse divergence guard aborts loudly on a genuine blow-up; (4) the NEW consecutive-accept
budget (`stall_accept_max`, default 4, not asserted as optimal -- swept in §6.3 to convert it from
invented to measured) bounds any livelock to a few wasted steps, never a hang; (5) loud, unconditional
stderr reporting (`STALL-ACCEPT` / `STALL-ACCEPT-TOTAL`) so a run containing accepted-unconverged
steps can never again be mistaken for clean (the exact failure mode of the sect.20 retraction).

Literature check (5 searches, `papers/*.md` dedup): "accept the non-converged iterate" is not
documented industry-standard practice (standard is fail-the-run on cutback exhaustion), but no
danger specific to hyperbolic conservation-law solvers was found that would make bounded,
loudly-reported acceptance worse than a clean stall. The closest documented analogue is
solution-limited time stepping (Ceze & Fidkowski, JCP 2009) -- conceptually the same move, and this
solver already implements the ingredients (update clamps, line-search escape). The specific failure
mode (dt-independent state jump from a lagged/explicit sub-step, O(1/dt) transient residual) is the
textbook operator-splitting/sequential-implicit inconsistency -- well-established, not novel.
**No new literature needed** (round 11's precedent holds); no papers needed.

---

## 5. Interaction with `diverged` (Stage 3c) -- recommendation, not implemented this round

3a makes 3c MORE important, not less. The accept path is loud and bounded (no new silent-bad-data
risk). But the GIVE-UP path (budget exhausted) is UNCHANGED -- still prints `STALLED:`, still
returns a finite partial state with `diverged==false` that `validate`/`dump` score as normal. That
is exactly round 10's bug, still standing. Recommended round-13 scope (Advisor decision required,
touches `validation.cpp`/result plumbing, moves `pass_count`): (1) `diverged=true` at the
budget-exhausted break; (2) surface `n_stall_accept` in the result struct so a completed-but-rough
run is machine-distinguishable from a clean one without failing it.

---

## 6. Success criteria

**Primary, tied to the round's real goal**: does at least one of {case24 plain, case34 plain, case33
`+ALPHA_IMPLICIT`} reach `t_end` under `ACID_STALL_ACCEPT`? If yes for any, the first-ever
controlled A/B for that case becomes possible in a later round.

Calibrated expectation: level 1 alone likely insufficient (removes the stop, leaves `cfl_scale` on
its floor, ~2-3e6 steps still needed -- expect a "grind", caught by the budget). Level 2 is designed
to escape the floor -- whether it does is the round's real experiment. **A clean negative is a valid
outcome** provided the diagnostic (dt trajectory, did `cfl_scale` climb off the floor) is recorded.

Regression gates (all must hold, byte-identity guaranteed by §3.0):
| # | command | required |
|---|---|---|
| 1 | build clean | -- |
| 2 | OFF `denner1d_validate` | 19/19, stdout byte-identical |
| 3 | plain-ON | 15/19, byte-identical |
| 4 | `+ALPHA_IMPLICIT` | 14/19, byte-identical |
| 5 | FD-invariance | 13/19 (round 11's corrected figure), byte-identical |
| 6 | any run with `ACID_STALL_ACCEPT` unset | zero `STALL-ACCEPT` lines |

Extra (not a hard gate, informational): `ACID_YADV=1 ACID_STALL_ACCEPT=2 denner1d_validate`
`pass_count` -- a drop below 15/19 means level 2 is a targeted crutch, not a general improvement.

Diagnostics to record per configuration x level: final `ACID done` line + wall time; every
`STALL-ACCEPT` line (ratio, acc_retry); `STALL-ACCEPT-TOTAL`; termination mode (t_end / STALLED /
DIVERGED); whether `cfl_scale` climbed off the 1e-3 floor by step 500/1000. Bounded-cost rule: cap
each experimental run at ~10 minutes wall; a run still going is a confirmed grind -- kill and record.

---

## 7. Pre-registered follow-ups (NOT implemented this round)

**7a -- the real 3b.** §1.2 localises the root cause to a dt-independent state jump from the
pre-Newton explicit block. Diagnostic: under `ACID_DBG`, print `rnorm3`'s three components
(momentum / continuity / energy) separately at `it==0` of each retry -- whichever carries the `1/dt`
scaling names the term. Prior: energy (`s.h` entering Newton is the previous step's converged value,
but `rho_o`/`Htot_o` are re-evaluated from the NEW alpha, so they disagree by a dt-independent
amount). If confirmed, fix = consistency re-init of `s.h`/`s.T` against the new alpha before Newton
starts.

**7b -- the line-search escape.** `acid.cpp:1858`'s `al < 0.03` escape accepts a state it just
measured to be WORSE. A `keep-sbak-if-no-reduction` variant is a small, independently-testable fix.

**7c -- the CFL floor.** `max(1.0e-3, ...)` (`acid.cpp:2141`) is what turns "slow" into "never" if
3a-ii proves insufficient.

---

## 8. Non-goals

No 3b (premise refuted). No 3c (Advisor decision required). No `cases.cpp`/`validation.cpp` edits.
No stdout/metric change at any level. No new default behaviour (env defaults to 0, byte-identical).
No tuning constants in the default path (`stall_accept_max=4` and the no-ratio-gate choice are both
research-env-gated and explicitly scheduled for future measurement, not asserted). No changes to the
Newton loop/line search/Jacobian/clamps/advection/recovery -- every edit is in the retry/accept/
report region and env-read region. No re-measurement of exited-shock RH residuals (round 11 answered
that).

---

## 9. Risks and detection (abbreviated -- full table in Planner's response, preserved in commit history)

Livelock (bounded by budget), grind (level 2's purpose; §6.4 caps cost), silent bad completion (new
surface -- mitigated by unconditional STALL-ACCEPT-TOTAL), default-path FP drift (guarded by §3.0,
checked by gates 2-5), `dt`/`Yv` mis-restore (checked by comparing `acc_dt` against the next `ACID
step` line and by symptom -- alpha/rho guard trips), BDF2-on-nonconverged-state (`have_o2=false`
mitigates), `acc_retry != 0` still collapsing cfl_scale at level 1 (expected/acceptable, level 2's
`else if` guard is unconditional on `acc_retry`), level-2 regressing a currently-passing case (R9
gate above).

---

## 10. Implementation sequence

Re-grep every anchor before editing (line numbers above verified against `bb6ff7b`, will shift once
edits land -- apply bottom-up: §3.6 -> §3.5 -> §3.4 -> §3.3 -> §3.2 -> §3.1). Build. Gates 2-5 first
(byte-identity), stop and fix if any moves. Level-1 experiment (3 configs, 10-min cap). Level-2
experiment (3 configs, 10-min cap). R9 gate. Sweep `ACID_STALL_ACCEPT_MAX` if cheap. Write up as
`YADV_RESEARCH.md` §22 including the `r_init ∝ 1/dt` table verbatim (the round's most durable result
regardless of whether 3a completes anything -- first mechanistic explanation of the 24/33/34
failure, falsifies a design premise stated in the code). Update roadmap.
