# Phase 3a — the cases-24/33/34 conservation defect

Produced by Agent(subagent_type="Plan", model=opus) during round 10 of the `yadv-round` loop.
Independently verified by the Advisor session before use: `acid.cpp`'s `if (!stepped) break;`
(silent stall, no `diverged` flag set) confirmed by direct code read; case24 under plain
`ACID_YADV=1` confirmed to stall at step 5 (t=2.978e-07 of final_time≈1.09e-4), with the dump's
mid-domain literally reading the pristine initial condition (`p=1e5, u=0`); case33 under
`ACID_YADV_ALPHA_IMPLICIT=1` confirmed to stall similarly (last visible debug step 5, t=2.853e-07)
while completing normally (step 2000, t=1.225e-04) under plain `ACID_YADV=1`.

## HEADLINE — the round-9 premise was false; two prior results are retracted

**Round 9's exciting finding ("case33's RH jump closes to machine precision under
`ACID_YADV_ALPHA_IMPLICIT=1`") is an artifact of a silently stalled run, not a computed shock. So
is round 3's original "cases 24/34 close to 1e-13 RH residual" claim.**

Mechanism: `acid.cpp`'s per-step retry loop, on exhausting all 14 dt-halving retries, does
`if (!stepped) break;` -- this exits the time-stepping loop WITHOUT setting `diverged = true`.
The `diverged` flag is what triggers NaN-marking (`std::fill(s.p, nan(""))` etc.) later in the
function, whose own comment explains its purpose: "so the validate counts a collapsed/diverged
run as a clean failure (finite=false), not a misleading partial state at t < final_time." Because
`diverged` is never set on a silent stall, this exact misleading-partial-state scenario happens
anyway -- `solve_case_acid` returns a FINITE state (the field at whatever step it last completed,
sometimes the pristine initial condition itself), and `validate`/`denner1d_dump` score/print it as
a normal, completed run.

Measured today (verified independently, see the header of this doc):

| case | config | completes? | last visible step / t | t_end | field at "t_end" |
|---|---|---|---|---|---|
| 24 | `ACID_YADV=1` | **NO -- stalls** | step 5, t=2.978e-07 | ~1.089e-04 | pristine IC + one near-void cell |
| 34 | `ACID_YADV=1` | **NO -- stalls** | (traced similarly) | ~8.535e-05 | pristine IC + one near-void cell |
| 33 | `+ALPHA_IMPLICIT=1` | **NO -- stalls** | step 5, t=2.853e-07 | ~1.283e-04 | pristine IC + one near-void cell |
| 33 | `ACID_YADV=1` | yes | step 2000, t=1.225e-04 | ~1.283e-04 | real, front at x=0.9356 |
| 24 | `+ALPHA_IMPLICIT=1` | yes | (reaches t_end) | | real, shock exited domain |
| 34 | `+ALPHA_IMPLICIT=1` | yes | (reaches t_end) | | real, shock exited domain |
| 24/33/34 | OFF (alpha path) | yes | reaches t_end | | control, fine |

The stalled dumps read the initial condition (`p=1e5, u=0, rho=rho_pre(alpha_pre)`) everywhere
except a 3-5 cell neighbourhood near `x=0.1` containing one near-void cell (extreme low `p`/`rho`,
large `|u|`). `yadv_rhcheck.py`'s undisturbed-cell search (`p < 1.5*p0`) then locks onto that void
cell as "the front," and picks a pre-shock reference cell that is *also* pristine IC -- so the
"residual" it computes is `cases.cpp`'s own closure-(A) analytic construction checked against
itself, which §11.3 already proved closes at machine precision. The round-9 "8.4e-13" and round-3's
original "1e-13" numbers are that identity, seen through a 12-digit dump print. **Retracted**:
`YADV_RESEARCH.md` §14.3 bullet 2 and §19.4's headline. The original text stays in the log,
annotated, per this project's "keep failed experiments in the history" culture -- not deleted.

The one number in that family that was always real: case33 under plain `ACID_YADV=1` (it
completes) -- momentum residual +8.81e-01, energy +6.46e-01, exactly as originally measured.

---

## 1. The time-vs-space question -- resolved, and it's neither

All three cases share identical domain/timing (`cases.cpp`: `base_config(800, 0.7/Vs_ref, 0.0,
1.0)`, IC step at `x<0.1`, transmissive both ends) -- no per-case asymmetry to blame. For the
configurations that genuinely complete, using the analytically-known pre-shock state (no
undisturbed-cell search needed, since it's known in closed form from `cases.cpp`'s own
construction):

| case | config | `Vs`(mass) | `Vs/Vs_ref` | implied front position at t_end | momentum resid | energy resid |
|---|---|---|---|---|---|---|
| 24 | `+ALPHA_IMPLICIT` | 9605.1 | **1.4945** | **1.146** (past x=1) | +7.36e-02 | +5.29e-02 |
| 34 | `+ALPHA_IMPLICIT` | 11455.5 | **1.3968** | **1.078** (past x=1) | +2.06e-02 | +1.51e-02 |
| 33 | `ACID_YADV=1` (control) | 2922.1 | 0.5355 | 0.9356 (in domain) | +8.808e-01 | +6.461e-01 |

The control row reproduces `yadv_rhcheck.py`'s published `+8.81e-01`/`+6.46e-01` exactly --
validates the analytic-pre-state method used for the other two rows.

**24/34 under implicit alpha carry a genuinely 1.40-1.49x faster shock that exits the domain
before t_end.** Extending `t_end` (the roadmap's own Stage-0 suggestion) moves the exit point
further out, not into view -- the domain itself is too short at the current speed, and `cases.cpp`
cannot be edited to lengthen it. The useful lever is a shorter *observation* window (see Stage 2).

**The `alpha_pre` (0.50/0.75/0.25) hypothesis is dead.** It predicts neither threshold nor
continuous behavior: under plain `ACID_YADV=1` the stall hits `alpha_pre in {0.50, 0.25}` and
spares 0.75; under `+ALPHA_IMPLICIT` it hits 0.75 and spares 0.50/0.25. The flag flips WHICH case
stalls -- this is a Newton-robustness switch, not a thermodynamic threshold in `alpha_pre`.

**One further fact worth carrying into every later stage.** In every completing run, `Y` is
conserved through the leading shock to 3-4 significant digits (measured: 24 `1.1583e-3` vs
`1.1580e-3` pre-shock; 34 `3.8853e-4` vs `3.8631e-4`; 33 `3.4661e-3` vs `3.4661e-3` — matches
§11.4's exact-conservation observation). The closure-(A) reference these cases are scored against
*requires* `Y` to grow 270-1620x across the shock (§11.2/§11.4). No Y-preserving scheme can ever
match that reference on these three cases. This is very likely the terminal finding for Phase 3a
(see the stopping criteria below) -- worth stating up front so later stages aren't chasing an
unreachable target without knowing it.

---

## 2. Staged plan

### Stage 0 (this round, round 10) -- honest instruments, zero solver code

New file `scripts/yadv_rh2.py` (root derived from `__file__`, following `yadv_r9_sweep.py`'s
pattern -- `yadv_rhcheck.py` stays untouched so its old, now-understood-to-be-bogus numbers remain
reproducible for the historical record). Guards the old script lacked:

1. **Null-run guard (primary).** Compare the dump against the analytic IC; if >90% of cells match
   to 1e-9 relative, print `NULL RUN (stalled)` and refuse to compute a residual.
2. **Completion guard.** Parse the last `ACID_DBG` step/t line; `t_last/final_time < 0.9` ⇒ also
   `NULL RUN`.
3. **Void guard.** Report `min(p)`, `min(rho)` and their cell; flag `p <= 10 Pa` or
   `rho < 1e-3*rho_pre`.
4. **Analytic pre-state**, not a dump search -- `(p=1e5, u=0, rho_pre(alpha_pre), alpha_pre)` from
   the closure-(A) construction. Makes exited shocks measurable; eliminates the "shock has left
   the domain, no undisturbed state" dead end entirely.
5. **Two independent `Vs` estimates**: mass-conservation-inferred, and front-position-inferred
   (`(x_front-0.1)/t_end`, or the `>1.2857*Vs_ref` lower bound when the shock has exited). Their
   disagreement is itself diagnostic.
6. **A plateau window** that excludes the Y-contact and the domain's outer ~15% (boundary
   reflection contamination, confirmed present in 24/34's completing runs).
7. **stderr capture** of any `DIVERGED`/`RETRY` lines.

Also this stage: `YADV_RESEARCH.md` §20 (the retraction, in the established style -- old sections
annotated not deleted) and the roadmap re-scope below.

**Success:** `yadv_rh2.py` labels exactly {24/plain, 34/plain, 33/+IMPLICIT} `NULL RUN`, and
reports 24/+IMPLICIT and 34/+IMPLICIT's momentum residuals as `+7.36e-02`/`+2.06e-02` (not "shock
left the domain"), and 33/plain as `+8.808e-01` (method's own control, matches `yadv_rhcheck.py`).
Hard gates (OFF 19/19+9/9, plain ON 15/19, `+IMPLICIT` 14/19, FD-invariance failure set) hold --
trivially true since no solver source changes this stage.
**Failure:** any of those labels/numbers disagree with the table above ⇒ the disagreement itself
is the finding; stop and report, don't force a match.
**Revert trigger:** none applicable -- purely additive.

### Stage 1 -- make the silent stall audible (smallest possible solver diff)

At the `if (!stepped) break;` site, add an unconditional stderr line mirroring the existing
`DIVERGED` message style: `STALLED: no admissible step at dt=%.3e after %d retries, step %d,
t=%.3e of %.3e -> abort`. Do **not** set `diverged` yet (would change plain-ON's metric VALUES for
24/34, a bigger decision deferred to Stage 3c). Under the existing `ACID_DBG` flag only, also
report which retry-loop condition triggered the stall and the first offending cell.
**Success:** all four hard gates byte-identical on stdout (only stderr changes); the three stalled
configurations print `STALLED`, no OFF-path case does.
**Failure:** any OFF-path case prints `STALLED` -- a much bigger finding than Phase 3a; stop, do
not proceed to Stage 2, re-plan from there.
**Revert trigger:** any stdout/metric difference anywhere.

### Stage 2 -- a diagnostic observation-time knob (gated, additive, not a physical constant)

One new env var, e.g. `ACID_TEND_SCALE` (default 1.0, unset behavior byte-identical), scaling
`c.config.final_time` inside `solve_case_acid` only -- gated exactly like `ACID_YADV`, not a
per-case coefficient or a physical parameter, purely an observation window for diagnostics. Use
it to: (a) view 24/34's shock at scale 0.5-0.7 so the plateau sits inside the domain, away from
the boundary-reflection contamination already observed in the current (post-exit) samples -- gives
a clean RH residual and an independent front-position `Vs`; (b) view the stalling configurations
at scale 0.001-0.01 with per-step dumps, to see exactly when/where the void cell forms relative to
the Newton failure.
**Success:** a clean (uncontaminated) 24/34 residual is measured; the void-cell formation step is
identified precisely.
**Failure:** the clean residual isn't materially different from the contaminated one -- still a
valid answer (rules out boundary contamination as the driver), record and continue.
**Revert trigger:** `ACID_TEND_SCALE` unset changes any metric anywhere -- revert immediately.

### Stage 3 -- only if Stage 2 points at a specific, single mechanism

Pick ONE, whichever Stage 2's evidence actually supports -- do not implement more than one without
re-measuring between them:
- **3a.** Retry-exhaustion policy: on exhausting retries, accept the best iterate reached instead
  of breaking (the existing keep-best/stall-break precedent for case15, extended). Global, no
  constant.
- **3b.** If the void cell precedes Newton failure: the `Y -> alpha` recovery is collapsing to
  `alpha -> 1` (pure phase, `rho -> 0`) adjacent to the closure-(A) contact -- fix in the
  recovery/limiting step, not the Newton solve.
- **3c.** Set `diverged = true` at the stall site -- correctness improvement (stalled runs become
  honest NaN failures instead of misleading finite ones), but CHANGES plain-ON's reported metric
  values for 24/34 (both already FAIL either way, so pass/fail is unaffected, but the metric
  numbers in every prior round's tables would change). Requires an explicit Advisor decision
  against the "plain ON byte-identical" rule before implementing -- propose, do not assume.
**Success:** the previously-stalled configuration completes to `t_end` with no new failure
anywhere -- the first ever both-configs-complete case among {24,33,34}.
**Failure/revert:** any drop below plain ON 15/19 or `+IMPLICIT` 14/19, or any OFF-path change ⇒
gate behind a new default-off flag (round 4/8 precedent) or revert outright.

### Stage 4 -- adjudication

With at least one case completing under BOTH configs, run the controlled A/B this investigation
has never yet had: does implicit alpha reduce that SAME case's leading-shock RH residual? Write
the verdict.

---

## 3. Explicit stopping criteria (set up front, per the roadmap's own requirement)

Stop Phase 3a and close it with a finding -- do not keep going incrementally past this point --
the moment ANY of these is true:

1. **The closure verdict (the likely outcome, per §1's Y-conservation observation).** A case in
   {24,33,34} completes under both configs and its leading-shock RH residual stays >1e-2 under
   both. Combined with Y being conserved to 3-4 digits in every completing run against a reference
   that requires 270-1620x Y growth (§1): these three cases are unpassable by any Y-preserving
   scheme against a closure-(A) reference driven by a closure-(A) piston. A modelling conclusion,
   not an implementation bug -- the fix would need a Y-consistent inflow BC or a Y-consistent
   reference, both requiring `cases.cpp`/`validation.cpp` edits that are permanently off-limits.
2. **The unreachable-stall verdict.** Two consecutive stages fail to make any stalled
   configuration complete, and Stage 2 shows the void cell forming within the first 1-2 steps.
   Then the Y path structurally cannot represent a 270-1620x Y contact at the initial
   discontinuity -- a model/discretisation redesign question, not a Newton repair.
3. **The scope guard.** Any stage would require a new tunable constant, a per-case coefficient, or
   a `cases.cpp`/`validation.cpp` edit to proceed.
4. The loop's standing `consecutive_failures >= 3`.

In either terminal case (1 or 2): `ACID_YADV`'s status is unchanged (default OFF, 15/19), cases
24/33/34 are documented as structurally out of reach for it, and the two retracted results are
corrected in the permanent record. That is a legitimate, valuable terminal answer -- worth more
than the (wrong) result it replaces.

## Non-goals

No `cases.cpp`/`validation.cpp` edits. No new tunable constants or per-case coefficients. OFF and
plain `ACID_YADV=1` stay byte-identical unless Stage 3c is explicitly escalated and approved by
the Advisor. Stage 3b of Phase 2 (T-pathway residual change) stays declined, unrelated to Phase 3
anyway. case15 (P3b) is not this plan's target. No `git push`, no `rm -rf`, no non-`Plan` agents.
