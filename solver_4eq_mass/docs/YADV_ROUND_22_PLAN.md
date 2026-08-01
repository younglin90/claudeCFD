# Round 22 Plan — RECON's 13/14 regression: the Jacobian is *not* the cause, and the fix is the dual projection

**Thread**: Advisor-chosen candidate (b) from round 21's `next_task`. Non-goals, explicitly out of
scope: (a) `rho_star`/`theta_o` dt-lag, (c) case33's advection channel, (d) `max_steps` exhaustion,
(e) case29.

**Advisor verification note (before implementation)**: the plan's most load-bearing structural
claims were checked directly against the code, not taken on trust:
- `acid.cpp:837` (`const Field s0 = s;`) confirmed as the insertion boundary.
- `acid.cpp:713-720` (the `Yv` initialisation expression `ACID_YADV_RESYNC` re-applies) confirmed
  verbatim.
- `acid.cpp:1665-1666` (`for (int it...) { compute_R();`) and `acid.cpp:1821` (`if (ajac || ajblk)`,
  the analytic Jacobian assembly) confirmed: the Jacobian is assembled INSIDE the Newton iteration,
  AFTER `compute_R()` has already run — it cannot be linearized at a stale, pre-RECON alpha, since
  RECON finishes ~830 lines and a full `s0` snapshot before the Newton loop even starts.
- `acid.cpp:1876-1885` confirmed: under plain `ACID_YADV=1` (`aimp = yadv && alpha_implicit` is
  false), `ap ≡ 0`, `aT ≡ 0`, and `D_ps/N_ps/D_Ts/N_Ts` reduce to the unstarred `D_p/N_p/D_T/N_T` —
  the Jacobian's alpha-family (`∂alpha/∂(p,T)=0`) matches the residual's (alpha frozen through
  Newton, `1302` gates the only re-derivation on `alpha_implicit`). No family mismatch exists under
  config B.
- `cases.cpp:669-676` confirmed: case13 IC sets `alpha = 1.0-1e-6 | 1e-6`, case14 sets
  `alpha = 1e-6 | 1.0-1e-6` — NEITHER case ever has a bit-exact `0.0`/`1.0` cell in its IC, so
  RECON's exact-skip cannot exempt them structurally, contrary to what round 21's plan assumed for
  "single-phase-like" cases (extending round 21 §31.5's correction).
- `validation.cpp:300-315` confirmed: case13's pass criterion is the 5-clause
  `case13_python_contract` (`smooth_error_ok && shock_ok && contact_ok && shock_location_ok &&
  hf_ok`), NOT a direct `l2_u`/`corr_u` threshold as round 21 §31.6 implied when quoting those
  fields as "the" regression evidence.

No structural error found in the Jacobian-exoneration argument or the RESYNC design. Proceeding to
Stage 0 (zero-code measurement) before writing any new solver code.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` sect.32)**: Stage 0
confirmed the plan's structural Jacobian argument but found a MIXED empirical result at 4.3
(`G+RECON`: case14 still fails as predicted, but case13 now PASSES) -- the plan's own falsification
condition at 4.2 fired for case13 (crossing criterion is `shock_location_ok`, not
hf/contact as predicted; traced to Jacobian-approximation sensitivity, not a family mismatch).
`RESYNC` was implemented anyway per the plan's own reasoning (a no-state-write projection is safe
regardless of which exact mechanism dominates). **Result: B+RESYNC = 15/19, cases 13 AND 14 both
PASS** (matching plain B's fail set exactly) -- the round's primary success criterion (G4) is met.
`dal_remap` collapses (G6, `1.1e-16`) as designed. Case24 gains only 2.6x (step 19->50) vs RECON's
20x. **But case14's phase-mass drift is 16.1%** (`ACID_RESYNC` meter, G8) -- firing the plan's own
pre-registered 1% non-promotable threshold (sect.6, prediction 7). Verdict: neither S1-S5 as
literally written; recorded as its own category -- gate-passing but non-promotable due to an
orthogonal, pre-registered conservation cost. Flag stays OFF. consecutive_failures NOT incremented.

---

## 0. Executive summary — the round's premise changes before any code is written

The brief offered two options: (A) the Jacobian re-evaluated after RECON is somehow stale, (B)
re-linearize J1/J2 at the post-RECON alpha. **Direct code reading refutes both, a priori** (see the
verification note above and §2 below) — and, more importantly, shows that the *corrective* action
Option B proposes would reproduce round 4/8's exact failure (giving the Jacobian a
`d(alpha)/dp` term for a residual that has none).

What replaces it: RECON's 13/14 regression is read here as a textbook, literature-named failure —
the **Abgrall spurious-pressure-oscillation mechanism**, re-entered through the back door. RECON
re-derives `p` from the *conserved* `(rho, e, Y)`. This is exact (a bit-level no-op in `p`) at any
smeared interface where `p` and `T` are both uniform (explaining round 21's case01 `linf_p=0`
result), and inexact exactly where `T` jumps at constant `p` -- a contact discontinuity. Cases 13
and 14 are air|water shock tubes whose defining feature is a large-`T`-jump contact at uniform `p`.

The designed fix is the **dual projection**, `ACID_YADV_RESYNC`: instead of moving the *state*
`(p,T,alpha)` onto the `Y`-manifold at fixed `(rho,e,Y)` (RECON), move the *auxiliary transported
variable* `Y` onto the state at fixed `(rho,u,e,p,T,alpha)`. It removes `dal_remap` by the same
construction RECON does, but cannot inject a pressure perturbation because it writes no state
field at all. Its implementation is the existing `Yv` initialisation (`acid.cpp:713-719`), hoisted
into the time loop. Its honest cost is phase-mass drift across step boundaries, which must be
measured, not asserted.

---

## 1. Code facts verified by reading (all in `cpp/denner_1d/src/acid.cpp` unless noted)

| Line | What is there |
|---|---|
| `578` | `const bool yadv = getenv("ACID_YADV")` |
| `593` | `const bool alpha_implicit = getenv("ACID_YADV_ALPHA_IMPLICIT")` |
| `610` | `const bool alpha_implicit_t = getenv("ACID_YADV_ALPHA_IMPLICIT_T")` |
| `647` / `674` | `recon_dbg` (`ACID_RECON`) / `yrecon` (`ACID_YADV_RECON`) |
| `711-720` | `Vec Yv(n,0.0)` + the once-only `Y = M(alpha, rho_a(p,T), rho_b(p,T))` init -- RESYNC's expression |
| `736` | time loop opens |
| `773-775` | `dt` finalised |
| `777-832` | round 21 RECON/`ACID_RECON` block |
| `837-838` | `const Field s0 = s; const Vec Yv0 = Yv;` -- RESYNC's insertion point is immediately before this |
| `862-864` | retry loop; `s = s0; Yv = Yv0;` |
| `867` | `p_o, T_o` captured from the (reconciled/resynced) `s0` |
| `1049-1056` | `al_o[i]` = alpha implied by the pre-advection `Yv` at `(p_o,T_o)` -- flux blend weight |
| `1085-1094` | conservative `rho*Y` update, `rho_star` divide, `Yv = anew` |
| `1095-1101` | `s.alpha[i]` recovery from the NEW `Yv` at `(p_o,T_o)` |
| `1105-1116` | Eqs.43-44 rebuild -> `rho_o`, `hstat_o`, `Htot_o` from the NEW alpha at the OLD `(p,T)` |
| `1118-1152` | `RMISM` (`ACID_RINIT`); computes `dal_remap`/`dal_adv` |
| `1283` | `compute_R` lambda opens |
| `1302-1310` | the ONLY in-Newton alpha re-derivation, gated `yadv && alpha_implicit` |
| `1348` | `eval_thermo(s, A, B)` inside `compute_R` |
| `1577` | `bool ajac = getenv("ACID_NO_AJAC") == nullptr;` |
| `1665-1666` | `for (int it...) { compute_R();` -- residual runs first |
| `1821` | `if (ajac \|\| ajblk) {` -- analytic Jacobian assembly, INSIDE the it loop, AFTER `compute_R()` |
| `1841-1896` | J1 loop, reads `s.alpha[i],s.p[i],s.T[i]` at the CURRENT iterate |
| `1876-1885` | `aimp = yadv && alpha_implicit`; `ap`/`aT` are `0.0` unless `aimp` |
| `1968-1985` | J2 flux-blend diagonal, entirely inside `if (yadv && alpha_implicit)` |
| `2113` | `if (it==0) r_init = n0;` |
| `eos.hpp:55-64` | `mass_fraction_from_alpha`/`alpha_from_mass_fraction`, bit-exact at `Y in {0,1}` |
| `cases.cpp:669-676` | case13 IC `alpha=1-1e-6\|1e-6`; case14 IC `alpha=1e-6\|1-1e-6` -- NEVER bit-exact pure |
| `validation.cpp:300-315` | case13's actual gate: 5-clause `case13_python_contract`, not `l2_u`/`corr_u` directly |
| `validation.cpp:670-684` | case14's gate (14 terms) |

---

## 2. The core technical question, answered from the code: Options A and B are both refuted

**Q: after RECON has moved `s.p,s.T,s.alpha`, is the analytic Jacobian still linearized around the
pre-RECON alpha?**

**A: No, structurally impossible.** Three reasons:

1. The Jacobian is assembled inside the Newton iteration (`1821`), AFTER `compute_R()` (`1666`)
   has already run and refreshed `s.alpha`/`s.p`/`s.T` via `eval_thermo` (`1348`). RECON finishes
   at line 832, ~830 lines before the Newton loop starts. No staleness is possible even in
   principle.
2. Under config B, the Jacobian's alpha-family is IDENTICAL to the residual's, and RECON changes
   neither: `aimp` is false, so `ap≡0, aT≡0` (Jacobian assumes `d(alpha)/d(p,T)=0`); the residual
   ALSO freezes alpha through Newton (`1302` gates the only re-derivation on `alpha_implicit`).
   Residual family == Jacobian family, exactly. RECON changes the VALUE alpha is frozen at, which
   both sides read from the same `s.alpha`. No family mismatch.
3. Option B implemented literally under config B — turning on `ap`/`aT` without `alpha_implicit` —
   would manufacture round 8's exact measured failure (case14 `l2_p 0.0145->0.512`). **Forbidden.**

**Consequence**: Option A is a no-op (already true); Option B is prohibited by round 8's own
precedent. This plan does NOT touch J1/J2.

**Corroborating datum already in hand**: round 21 §31.4 G5 measured `C+RECON` at 14/19 unchanged
from `C` — and `C`'s fail set is `{14,15,24,33,34}`, so **case13 already PASSES under `C+RECON`**.
Under `C` the residual re-derives alpha at the current `(p,T)` every eval and the Jacobian knows it
— RECON is harmful when the in-step closure FREEZES alpha (B) and harmless when it FOLLOWS Y (C).
A projection-vs-closure consistency statement, one level above round 4's Jacobian-vs-residual
statement.

---

## 3. The mechanism, derived

### 3.1 Exactness theorem for RECON at uniform (p,T)

NASG mixture at p-T equilibrium, mass fraction Y: `bbar(Y), qbar(Y), cpbar(Y), Ka(Y), Kb(Y)` all
AFFINE in Y (per-phase NASG coefficients are linear, mixture is a Y-weighted blend). At fixed
`(p,T)`, both `v(p,T,Y)` and `e(p,T,Y)` are therefore affine in Y.

**Theorem**: states 1,2 share `(p,T,u)`, differ in Y. Any mass-weighted convex combination of the
CONSERVED variables `(rho, rho*Y, rho*e)` recovers exactly `(p,T)` under `pT_from_v_e_massfrac`.
*Proof*: with mass weight `theta`, `Ybar` and `ebar` are the same `theta`-weighted combination;
affineness in Y makes `v(p,T,Ybar)=vbar`, `e(p,T,Ybar)=ebar` identically; uniqueness of the
admissible root makes this THE answer. QED.

**Corollary 1**: RECON is a no-op in p at any smeared interface with uniform (p,T) -- explains
round 21's case01 `linf_p=0` exactly (not luck).

**Corollary 2**: if T jumps at constant p (a contact discontinuity), the mixed cell's `(vbar,ebar)`
is NOT a PTE state at p, and the inversion returns `p* != p`. RECON writes that into `s.p`,
injecting a genuine pressure perturbation -- the Abgrall (1996) mechanism. Aggravated in
cases 13/14 by: no bit-exact pure cells (so exact-skip never fires), and air|stiffened-water pairs
where a trace liquid mass fraction carries a large Pi and eta into the quadratic's coefficients --
most sensitive exactly in the "nearly pure gas" cells filling most of the domain.

### 3.2 Why the closure pairing matters (unifying round 4, 8, 21)

**Rule**: a once-per-step projection must project onto the manifold the in-step closure already
lives on. Round 4/8 is the Jacobian-level instance (differentiate the alpha family the residual
evaluates). Round 21's `B+RECON` is the state-level instance; `C+RECON` (unharmed) is the matching
counterexample proving the rule.

### 3.3 The designed fix -- `ACID_YADV_RESYNC`, the dual projection

| | writes | conserves | injects dp? | dal_remap after |
|---|---|---|---|---|
| RECON (r21) | p,T,alpha,rho,hstat,h | rho,rho*u,rho*E,Y per cell | YES (Cor.2) | 0 |
| RESYNC (r22) | Yv ONLY | every state field bit-unchanged | NO, by construction | ~eps*kappa floor |

```cpp
// ACID_YADV_RESYNC (round 22, RESEARCH-ONLY, default OFF; inert unless ACID_YADV):
if (yadv && yresync && !yrecon) {
    for (int i = 0; i < n; ++i) {
        const double pu = std::max(s.p[i], 1.0), Tu = std::max(s.T[i], 1e-6);
        const double Ynew = std::clamp(
            mass_fraction_from_alpha(std::clamp(s.alpha[i], 0.0, 1.0),
                                     phase_props(pu, Tu, A).rho,
                                     phase_props(pu, Tu, B).rho), 0.0, 1.0);
        if (std::isfinite(Ynew)) Yv[i] = Ynew;   // fail-safe
    }
}
```
No `eval_thermo`, no `h` refresh, no `s` write of any kind.

**Why it removes the same defect**: after RESYNC, `al_o` (`1049-1056`, evaluated from `Yv` at
`(p_o,T_o)` -- the SAME (p,T) RESYNC just used) reproduces `s0.alpha` to the conditioning floor;
post-advection alpha differs from `s0.alpha` only by this step's O(dt) advection. The Eqs.43-44
rebuild then produces `rho_o ~ s0.rho + O(dt)`, matching round 21's own mechanism argument, applied
to the same site, from the other side.

**Honest cost**: `rho*Y` no longer carried exactly across step boundaries -- must be measured
(phase-mass drift meter, Stage 1a) and reported, not asserted away. Same class of compromise the
published 19/19 OFF path already makes (transports alpha, also not a strict material invariant).

**Two exact-by-construction properties (sharp gates)**:
- Step 0 is a BIT-LEVEL no-op (RESYNC's expression is textually identical to `713-719`'s own IC
  init at step 0 on the same state).
- case01 stays `linf_p=0` trivially (no state field written, ever).

### 3.4 Alternatives considered and rejected (do not revisit)

- Advection-only Eqs.43-44 rebuild: rejected -- would break the existing remap cancellation at
  convergence, reintroducing an apparent mass source of the same size.
- dt-proportional damping of RECON: rejected -- a rate constant is a tuning coefficient (project
  rule), and its splitting error does not vanish under refinement.
- Tolerance-based skip: rejected -- `tol` is a tuning coefficient.
- Conditioning-bound skip (`|d alpha| > 8*eps*kappa`): not a coefficient, but likely ineffective in
  13/14's bulk (genuine O(a_p*dp) response, not roundoff) -- Stage 0.4 confirms/refutes cheaply.
- THINC-indicator-gated RECON: rejected as PRIMARY (indicator marks 1-2 cells; theory predicts
  damage across the whole non-skipped domain) -- kept as a fallback ONLY if Stage 0.5 localises
  the damage to those cells.

---

## 4. Stage 0 -- zero code. Do not write a line until every item below has an answer.

### 4.0 Hygiene fix first (the ONLY Stage-0 edit, in `scripts/`, not `cpp/`)

`scripts/yadv_r9_sweep.py`'s `ACID_ENV_VARS` purge list is missing every research flag added since
round 9. Add `ACID_YADV_RECON, ACID_RECON, ACID_YADV_RESYNC, ACID_RESYNC, ACID_YADV_HREINIT,
ACID_RINIT, ACID_RCELL, ACID_STALL_ACCEPT, ACID_STALL_ACCEPT_MAX, ACID_TSAT, ACID_AJAC_BLK`.
Without this, a stale exported flag silently contaminates every `--sweep`/`--verify` number this
round. `CONFIGS`/`EXPECTED` must NOT change.

### 4.1 Reproduce round 21's headline (control)
`DENNER_ACID=1 ACID_YADV=1 $V` (expect 15/19, fail `{15,24,33,34}`); `+ACID_YADV_RECON=1` (expect
13/19, fail `{13,14,15,24,33,34}`). If either differs, STOP -- baseline moved.

### 4.2 Identify the actual failing criterion for case13/14
Parse the per-case JSON, evaluate case13's 5-clause contract by hand, case14's 14 terms by hand.
Pre-registered expectation: the crossing terms are HF and/or contact-related
(`case13_u_smooth_hf`/`case13_p_smooth_hf`/`case13_contact_rho_overshoot`; for case14 `hf_p`
and/or `linf_p`/`amp_ratio_*`). Falsified if the crossing term is `shock_location_ok` or a plain
smooth-l2 term with hf comfortably inside gate -- would point at shock damage instead of interface
oscillation, and §3 must be rewritten before Stage 2.

### 4.3 The decisive Jacobian falsification (settles Options A/B empirically)
Config G (FD Jacobian) currently PASSES 13/14 (round 20 `EXPECTED["G"]=(15,{15,24,33,34})`).
- `G+RECON` ALSO fails 13/14 -> Jacobian exonerated, Options A/B dead empirically too -> Branch I.
- `G+RECON` keeps 13/14 passing -> Branch J: pair RECON with C instead (already measured 14/19,
  below B's 15/19, remains non-promotable, report honestly).
- Mixed (fails but on different criteria) -> report both, proceed to Branch I with lowered claim.

### 4.4 RECON's actual footprint on 13/14 (read-only)
`ACID_RECON=1` (diagnostic only, applies nothing) on cases 13/14: expect would-touch ~ncell (nearly
all 800 cells), worst-dp cell tracking the CONTACT position, not the shock. Falsified if
would-touch stays in the tens or worst-dp sits at the shock.

### 4.5 Spatial localisation
Dump B vs B+RECON for case13, diff `|u|`/`|p|` vs x. Prediction: damage concentrated at/behind the
contact, high wavenumber content; shock position essentially unmoved.

### 4.6 Limit-cycle test
Per-step worst `|dp/p|` under B+RECON+ACID_RECON on case13: prediction it does NOT decay toward
roundoff -- settles to a roughly constant per-step value (the step operator re-creates what RECON
removes). Falsified if it decays monotonically to ~1e-14 within tens of steps.

### 4.7 Does the consistent pairing keep case24's gain?
Compare case24 stall step under B, B+RECON, C, C+RECON. Never hardcode a step number -- read it
via `ACID_DBG`.

**Gate to Stage 1**: Stage 0 must produce (i) the exact failing criterion for 13/14, (ii) a
Branch I/J verdict, (iii) a localisation verdict. If 4.2 cannot be resolved, STOP and report.

---

## 5. Stage 1 -- the fix and its instrument (new code, all default OFF)

Nothing promoted to default this round (round 4/13/18/21 precedent).

### 1a `ACID_RESYNC` (diagnostic, default OFF, stderr, applies nothing)
Per step, per cell: worst `|Ynew-Yv[i]|` + cell index, count of bitwise-changed cells, phase-A mass
drift this step `dM = sum(rho_i*(Ynew_i-Yv_i)*dx)`, running sum, `M0` captured once after `711-720`,
report `SumdM/M0`. Honours `ACID_BLK_STEP`.

### 1b `ACID_YADV_RESYNC` (research-only, default OFF, inert unless ACID_YADV)
Exactly §3.3's block, placed immediately after the round-21 RECON block, before line 837. Gated
`yadv && yresync`; skipped with a stderr notice if `yrecon` also set (mirrors HREINIT/RECON
exclusion). Diagnostic and fix share one loop. No new numeric constants. No `eval_thermo`, no `h`
write, no other `s.*` write.

### 1c unit tests
Round-trip idempotence (extend existing test), pure-end exactness through the RESYNC expression,
`dal_remap` collapse assertion for a synthetic `(alpha,p_o,T_o) -> Y_resync` round-trip.

---

## 6. Stage 2 -- gates

G0 unit test. G1 `--verify` OFF byte-identical (HARD, round doesn't merge if this fails). G2
`--sweep` unset unchanged, `EXPECTED` NOT edited (HARD). G3 diff hygiene (HARD). G4 `B+RESYNC` full
sweep -- primary success: >=15/19 WITH 13 and 14 passing. G5 case24 stall step vs B/B+RECON. G6
`dal_remap` collapse under B+RESYNC (~1e-11 floor, not exactly 0). G7 case01 `linf_p=0` + step-0
bit-identity on 26/27/28/32/33. G8 phase-mass drift table, all 19 cases. G9 `C+RESYNC` near-identity
check. G10 diagnostic-only no-op. G11 perf sanity (<=2%). G12 case34/33 with declared wall-clock
budget, honest "not evaluated" if exceeded.

**Predictions** (declared before running): unset->unaffected (falsified by any byte diff, HARD);
step0 bit-level no-op (falsified by any step-0 diff -> implementation bug); 13/14 keep passing (the
round's whole point -- falsified -> S4 below); `dal_remap` collapses (near-certain by construction);
case24 gain is genuinely ~even-odds (may reveal RECON's case24 gain came from the STATE projection,
not dal_remap removal -- pre-registered either way, no post-hoc rationalising); case33 unchanged or
near so (any movement = the surprise it is); phase-mass drift <1% on passing cases or RESYNC is
non-promotable regardless of pass_count (decision threshold, not a code coefficient).

---

## 7. Stop / decision rule

| Outcome | Trigger | Consequence |
|---|---|---|
| S1 strong success | B+RESYNC>=15/19 WITH 13,14 passing, case24 >=10x further, drift<1% | Round 23 = promotion eval. Do not promote this round. |
| S2 partial | 13/14 recovered, no regressions, case24 unmoved | Flag OFF, kept as infra. dal_remap removal alone did NOT drive case24's gain -- next thread: case24's T-ceiling-saturated failure, or rho_star/theta_o. |
| S3 null | dal_remap collapse confirmed, everything else bit-neutral, no case moves | Honest negative result, consecutive_failures NOT incremented. Stage-0 findings stand as deliverable. |
| S4 structural | 13/14 regress under B+RESYNC too | Y<->alpha step-boundary consistency is harmful to 13/14 by EVERY available route (state-side, Y-side, residual-side per C). Halt this line pending explicit Advisor/user scope decision (scheme redesign territory). |
| S5 harm | any hard gate fails, or B+RESYNC<15/19 | Flag OFF, regression reported honestly; round does not merge if hard gate fails. |

Hard abort before any code: Stage 0 cannot establish §4.2, or §4.3 self-contradicts.

---

## 8. Literature (needed stubs, DOIs to be verified before commit)

Already in repo: Fujiwara 2023 (pressure-equilibrium-preserving, states the Abgrall/Johnsen
tradeoff verbatim -- read §2/§3 before Stage 2), AlahyariBeig/Johnsen 2015, Terashima 2025, Coppola
2026 PEP/APEC/KEEP, Collis 2025 (already cited by round 21). Round 21's 5 stubs -- do not
duplicate.

New stubs needed: Abgrall 1996 (10.1006/jcph.1996.0085, the mechanism's origin), Shyue 1998
(10.1006/jcph.1998.5930, mass-fraction/stiffened-gas extension), Johnsen & Ham 2012
(10.1016/j.jcp.2012.04.048, the conservation-cost objection RESYNC pays), Saurel & Abgrall 1999
(10.1006/jcph.1999.6187, operator-split relaxation precedent, relevant to why dt-damping was
rejected), Ma/Lv/Ihme 2017 (10.1016/j.jcp.2017.03.022, double-flux alternative, scope note only).

---

## 9. Risks / pitfalls carried forward

`DENNER_ACID=1` required on every invocation. Do the `ACID_ENV_VARS` purge fix FIRST -- every
Stage-0 number before it is untrustworthy. Never combine `ACID_YADV_RESYNC` with
`ACID_YADV_RECON`/`ACID_YADV_HREINIT`. Do not touch `cases.cpp`/`validation.cpp` or
`CONFIGS`/`EXPECTED`. No `-march=native`. No inline WSL for-loops/shell redirect for captures. Never
hardcode `ACID_BLK_STEP`. **Do not add `ap`/`aT` to J1/J2 outside `alpha_implicit`** -- the single
most tempting wrong move this round. Round 20/21 baselines are live. Case34 has a wall-clock
history -- declare budget up front, report "not evaluated" honestly if exceeded.

---

## 10. Deliverables

`docs/YADV_RESEARCH.md` §32 (32.1 Stage 0 + two annotations to §31 -- not edits; 32.2 exactness
theorem/Abgrall reading; 32.3 RESYNC as implemented; 32.4 gates; 32.5 target measurements + drift
table; 32.6 verdict; 32.7 reproducing). `docs/YADV_ROUND_22_PLAN.md` (this file, +actual outcome).
`docs/YADV_ROADMAP.md` update. 5 `papers/*_needed.md` stubs. `scripts/yadv_r9_sweep.py`
`ACID_ENV_VARS` fix only. Commits per convention, local merge only.
