# Round 6 Execution Brief — Phase 2 Stage 1

Produced by Agent(subagent_type="Plan", model=opus) during round 6 of the `yadv-round` loop.
Spot-checked by the Advisor session against the actual code before use (`acid.cpp:1487-1515`
matched the brief's transcription exactly, `dalpha_dp_massfrac` confirmed present in `eos.hpp`).

## What was implemented

Augmented the existing J1 cell-EOS-chain loop in `acid.cpp` (the analytic Jacobian's per-cell
`D`,`D_T`,`D_p`,`N`,`N_T`,`N_p`,`hsT`,`hsp`,`dTp/dTh/dTu`,`drp/dru/drh` block, lines ~1503-1521)
to star `D_p`/`N_p` with the product-rule addend from `a_p = d(alpha)/dp|_{T,Y}` (already derived
and unit-tested in round 5's `dalpha_dp_massfrac`), gated by `yadv && alpha_implicit`:

```cpp
const bool aimp = yadv && alpha_implicit;
const double ap = aimp ? dalpha_dp_massfrac(al, pa.zeta, pa.rho, pb.zeta, pb.rho) : 0.0;
const double D_ps = aimp ? D_p + (pa.rho - pb.rho) * ap : D_p;
const double N_ps = aimp ? N_p + (pa.rho * pa.h - pb.rho * pb.h) * ap : N_p;
alp_p[i] = ap;   // stored for Stage 2's J2 flux-blend diagonal
const double hsT = (N_T * D - N * D_T) / (D * D), hsp = (N_ps * D - N * D_ps) / (D * D);
dTh[i] = 1.0 / hsT; dTu[i] = -u / hsT; dTp[i] = -hsp / hsT;
drh[i] = D_T * dTh[i]; dru[i] = D_T * dTu[i]; drp[i] = D_ps + D_T * dTp[i];
```

The T-pathway (`D_T`, `N_T`, `hsT`) is untouched — deliberately, per Phase-2 §0.4's finding that
the residual's alpha is lagged one `compute_R` call in T, so the frozen-T derivative is the exact
derivative of the coded map; starring T is the contingent Stage 3. `ap` ternary-selects a bit-copy
of the unstarred value when `!aimp`, so the OFF path and plain `ACID_YADV=1` are provably
byte-unchanged by inspection, not by floating-point luck.

This is the first round in the whole `ACID_YADV` experiment that edits the analytic Jacobian
itself (all prior rounds edited the residual or the transport stencil).

## Verified gates (all held)

1. OFF (`ACID_YADV` unset): 19/19, 9/9 byte-identical to the published `solver_denner` binary.
2. Plain `ACID_YADV=1`: 15/19, dumps byte-identical (case01 confirmed).
3. FD-invariance (`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_NO_AJAC=1`): 12/19, **exact same
   failure set** as rounds 4/5 (14,15,24,27,28,33,34) — Stage 1 only touches code the FD path
   never executes, confirmed unmoved.
4. Acoustic cases (04/05/07/35/36, forced onto the FD Jacobian by TR-BDF2): not independently
   re-dumped this round, but gate 3 holding is sufficient evidence they are untouched (same code
   path reason).

## The target measurement — genuine success

`ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1` (default analytic Jacobian):

| | round 4/5 baseline | **round 6 (Stage 1)** |
|---|---|---|
| pass_count | 12/19 | **14/19** |
| failure set | 13,14,15,24,25,33,34 | **14,15,24,33,34** |
| case13 | FAIL (l2_p=0.0383) | **PASS** (l2_p=0.0313, corr_p=0.994) |
| case25 | FAIL (corr_p=-0.123) | **PASS** (corr_p=0.991) |
| case15 amp_ratio_p | 1.23223 | **1.00042** (target ~1.0, exact) |
| case15 corr_p | 0.09937 | **0.999285** |
| case15 pass | FAIL (non-convergence) | FAIL, but **every quantitative gate criterion now
  passes** (corr_p/u/rho, all l2_*) — blocked only by the TV/oscillation guard (`osc_ok`), a
  narrower, different failure than round 4's diagnosis |
| case14 | FAIL | FAIL, unchanged in kind (separate `hsT<0` lead from round 5, out of scope) |
| case24/33/34 | FAIL (conservation defect) | FAIL, unchanged in kind (explicit non-goal) |

Success bar from the plan was `pass_count >= 13`; achieved 14, with case13 AND case25 both fully
recovering (not just one) and case15 moved to within one narrow gate criterion of passing.

## Case15's remaining blocker (recorded, not chased this round)

`validation.cpp`'s case15 gate requires `corr_p>=0.93, corr_u>=0.998, corr_rho>=0.99,
l2_p<=0.18, l2_u<=0.06, l2_rho<=0.05, smooth_ok, osc_ok`. All six quantitative thresholds now
pass. `smooth_ok`/`osc_ok` (jump-concentration and total-variation-excess guards, `validation.cpp`
~695-728) were not independently re-measured this round — case15's `peak_delta_u` moved from
round 4's FD-path 321 to **0** this round, which is a strong signal `osc_ok` is close, but this
was not confirmed against the exact `p_osc`/`r_osc` formulas. Left for the next round to close out
or for Stage 2/3 to move incidentally.

## Non-goals honored

No edits to `cases.cpp`/`validation.cpp`. No new tunable constant. `cpp/denner_1d/src/acid.cpp`
was the only file touched. Cases 24/33/34 not chased (explicit non-goal — their defect is
conservation, not Jacobian accuracy, per round 4's §15.5/§11.6 diagnosis).

## Literature

No new papers needed reading for this stage (round 5's search already covered the derivative's
correctness; Stage 1 only consumes it). The Planner surfaced two reservoir-simulation
"volume-balance fully-implicit" papers (Fernandes/Marcondes/Sepehrnoori 2016, DOI
10.1016/j.apm.2015.09.002; Coats 1998, SPE 50990) as a structural precedent for a mixture-total-
compressibility term on a Newton Jacobian diagonal, both unreachable this round (403 / expired
TLS) — recorded at `papers/volume_balance_jacobian_needed.md` for the user to supply if wanted.
