# YADV Round 13 Plan — name the residual term that carries the `1/dt` growth, then fix the initial guess

Planner output (Agent, subagent_type="Plan", model=opus), round 13. Grounded in
`docs/YADV_RESEARCH.md` §22 (round 12), `docs/YADV_ROUND_12_PLAN.md` §7a, direct code read.

Advisor spot-check: confirmed `rnorm3` lambda at `acid.cpp:1336`, `escal` decl at `:1335` (BEFORE,
not after, the Planner's own re-check note in its response was self-contradictory here but harmless
-- its own fallback "recompute locally" is adopted regardless, avoiding any scope-dependence risk),
`for (int it = 0; it < (ajac ? 150 : 40); ++it) {` at `:1438`, `compute_R();` at `:1439` -- matches
plan's stated anchor exactly. Eqs.43-44 `rho_o[i]`/`hstat_o[i]`/`Htot_o[i]` block confirmed near
`:975-981`.

## 0. Executive summary

Round 12's hypothesis: the pre-Newton block re-derives `alpha` at `(p_o,T_o)` and rebuilds
`rho_o`/`hstat_o`/`Htot_o` from it, but `s.h` entering Newton is still the PREVIOUS step's converged
value -- a dt-independent mismatch that makes `r_init` grow as `1/dt` (round 12 measured this
exactly: doubles per dt-halving from retry 6 onward). The mechanism, derived from reading the code:
at `it==0`, `s.p==p_o`, `s.u==u_o`, `s.h==s0.h` (nothing between retry-restart and the Newton loop
touches them), so `Rene|it=0 = (rho*·s0.h - rho_o·Htot_o)·dx/dt + adv` -- a STATE mismatch scaled by
`1/dt`. The mismatch itself decomposes into a REMAP term (alpha recovered at the PREVIOUS step's
frozen Y meeting THIS step's p,T -- constant across a retry sweep, since it's set by last step's
`dt_prev`, not this step's `dt`) and an ADVECTION term (this step's Y-transport, `O(dt)`, harmless).
REMAP dominates and is `dt`-independent -- exactly what produces the doubling. The OFF path has no
remap term structurally (alpha updates the SAME state it already holds, no recovery-from-Y step).
`+ALPHA_IMPLICIT` also has no remap term (it re-derives alpha at the CURRENT p,T on every Newton
call, so the previous step's converged alpha already reflects it) -- this predicts `+ALPHA_IMPLICIT`
shows NO `1/dt` growth, giving round 12's A/B result a mechanism, not just a correlation.

The fix, if confirmed, cannot change conservation (defect-correction: `compute_R` is the single
source of truth, the fixed point `R=0` is unchanged by the initial guess) -- it can only change
solvability. That is precisely round 12's diagnosed defect (pure Newton-convergence failure, never
a void cell).

## 1. Stage 0 -- diagnostic (implemented this round, streamlined from the Planner's full spec)

New env var `ACID_RINIT` (default OFF, diagnostic only, no FP arithmetic added on any path when
unset). At `it==0` of every Newton solve (optionally restricted to one step via `ACID_BLK_STEP`,
reusing the existing env var), prints two lines:

**RINIT** -- the component split of `rnorm3()` (self-check: must equal `RHIST`'s `n0` on the same
run) plus each component's argmax cell:
```
RINIT case=%s step=%d retry=%d dt=%.6e r=%.6e mom=%.6e con=%.6e ene=%.6e fene=%.4f iene=%d
```

**RMISM** -- the candidate dt-independent state mismatches, decisive measurement:
```
RMISM case=%s step=%d retry=%d dh=%.4e@%d drho=%.4e@%d dal=%.4e@%d dal_remap=%.4e@%d dal_adv=%.4e@%d
```
`dh=|s.h[i]-Htot_o[i]|`, `drho=|s.rho[i]-rho_o[i]|`, `dal=|s.alpha[i]-s0.alpha[i]|` (total alpha
jump), `dal_remap=|f(Yv0[i],rho_A(p_o,T_o),rho_B(p_o,T_o)) - s0.alpha[i]|` (REMAP, predicted
dt-independent), `dal_adv=|s.alpha[i]-f(...)|` (ADVECTION, predicted proportional to dt). When
`!yadv`, `dal_remap:=0` and `dal_adv:=dal` (OFF has no remap term by construction).

## 2. Pre-registered predictions and decision rule

- **P1**: `r`/`mom`/`con`/`ene` double per dt-halving in the stalling regime (retries >=6, matching
  round 12 §22.2 exactly), while a flat regime holds for retries 0-4.
- **P2**: `dal_remap` roughly constant across retries; `dal_adv` halves per halving.
- **P3**: `dal_remap >> dal_adv` at the stalling step.
- **Control (OFF, `ACID_YADV` unset)**: no `1/dt` growth of `r` (structural immunity, §0).
- **Control (`+ALPHA_IMPLICIT`)**: `dal_remap ~ 0`, `r` flat in dt.

**Decision rule**: P1+P2+P3 confirmed on case24 AND the `+ALPHA_IMPLICIT` control shows no `1/dt`
growth -> proceed to Stage 1 (the fix). Otherwise: stop, write up the actual finding (which may
still be informative), do not force a fix around a wrong diagnosis. Round succeeds either way if
the instrument is correctly self-checked and produces an interpretable measurement --
`consecutive_failures` is not incremented by a clean refutation (this project's negative-result
culture, exemplified by §20's own retraction).

## 3. Stage 1 (conditional) -- the consistency re-init

New env var `ACID_YADV_HREINIT` (default OFF, inert unless `ACID_YADV` and `coupled` are also
active). Immediately after the Eqs.43-44 block (`rho_o`/`hstat_o`/`Htot_o` computed), set:
```cpp
if (yadv && hreinit && coupled)
    for (int i = 0; i < n; ++i)
        s.h[i] = std::max(Htot_o[i], 0.5 * u_o[i] * u_o[i] * 1.0001 + 1.0);  // reuse the existing
                                                                              // line-search kinetic
                                                                              // floor, no new constant
```
This changes ONLY the Newton initial guess -- `compute_R` (the fixed point) is unaffected, so no
conservation/RH property can move. `s.T` needs no re-init (already equals `T_o`; with `s.h==Htot_o`
the `T_from_hstat` call converges on its first test and returns `T~T_o` to machine precision).

Gated on `coupled` (h is a Newton unknown only there) and deliberately NOT excluded on
`alpha_implicit` (predicted near-no-op there -- a free falsification test of the mechanism).

**Success metrics (pre-registered)**: S2 (headline) = case24 AND case34 reach `t_end` with ZERO
`STALLED:` lines under `ACID_YADV=1 ACID_YADV_HREINIT=1` alone (no `ACID_STALL_ACCEPT`). S3 =
combined with `ACID_STALL_ACCEPT=1`, accepted-step count drops below round 12's (2 for 24, 4 for
34), ideally to 0. S4 = no improvement (measured negative, flag stays default OFF). S5 = a
`pass_count` regression on plain-ON or `+ALPHA_IMPLICIT` (net cost, flag stays default OFF).

**If S2 or zero-accept S3 holds**: re-run round 11's validated front-derived-window RH-residual
method (verbatim, `scripts/yadv_r11_window.py`'s helpers reused, that file itself untouched) on the
now-CLEAN plain run vs `+ALPHA_IMPLICIT` -- the first genuinely clean controlled A/B, removing round
12 §22.5's caveat.

## 4. Gates (hard, with both new env vars unset)

Same five as round 12 (OFF 19/19, plain-ON 15/19 same failure set, `+ALPHA_IMPLICIT` 14/19,
FD-invariance 13/19, all byte-identical stdout), plus: zero `RINIT`/`RMISM` lines with `ACID_RINIT`
unset; zero effect from `ACID_YADV_HREINIT` unset (byte-identical, same argument); and an explicit
no-op check -- with `ACID_YADV_HREINIT` unset, cases 24/34/33 must still fail EXACTLY the same way
(same `STALLED:`/`STALLED-DETAIL:` content) as round 12's baseline.

## 5. Non-goals

No case33 work (different, sustained difficulty per round 12 -- separate future round; at most one
observational run, no fix attempt). No Stage 3c (`diverged=true`, still needs explicit Advisor
decision). No weakening of `ACID_STALL_ACCEPT` (stays as the safety net regardless of outcome). No
promotion to default for any new flag. No `cases.cpp`/`validation.cpp` edits, no tuning/per-case
constants (the only new numeric expression reuses the existing kinetic floor). No edits to
`scripts/yadv_rh2.py`/`yadv_r11_window.py`/`yadv_verify.py` (published instruments). No changes to
the Newton loop, line search, Jacobian, clamps, advection stencil, or alpha recovery itself --
Stage 1 sets one initial guess; Stage 0 only reads and prints. No re-measurement of exited-shock RH
residuals in general (round 11 answered that); §3's re-run reuses that exact method unchanged.

## 6. Literature

Straightforward operator-splitting/fractional-step initial-guess lag -- standard, not novel (round
11's precedent applies, no new citation needed) -- UNLESS the measurement instead shows the growth
is carried by the flux terms rather than the transient (would be genuinely unexplained, since the
MWI memory coefficient tends to a finite limit as dt->0 and cannot produce 1/dt growth) or by a
BDF2/`Cold_*` inconsistency across retries -- in either of those cases, stop, do not force the Stage
1 fix, and write up the actual finding with a note that a literature search may be warranted next
round.
