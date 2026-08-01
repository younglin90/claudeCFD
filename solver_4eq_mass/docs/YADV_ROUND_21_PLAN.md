# Round 21 Plan — the simultaneous `(T, ρ, h)` reconciliation for cases 24/34

**Thread**: Advisor-chosen candidate (a), round 13 §23.3's "harder" fix. Non-goals: `max_steps`
exhaustion, case29, case33's own difficulty (case33 appears only as a prediction target).

**Advisor verification note (before Step 5 implementation)**: the plan's most load-bearing
structural claims were checked directly against the code, not taken on trust:
- `acid.cpp:747` (`const Field s0 = s;`) confirmed as the insertion point, immediately after `dt`
  is finalized.
- `acid.cpp:1576` (`compute_R();`, inside the `it` loop) confirmed to run BEFORE `acid.cpp:2022`
  (`if (it == 0) r_init = n0;`) — this is the load-bearing correction to round 13 §23.3's literal
  wording: by the time `r_init` is captured, `compute_R`'s own `T_from_hstat`/`eval_thermo` calls
  have already reconciled `s.T`/`s.rho` with `s.h`. Confirmed exactly as claimed.
- `acid.cpp:1018-1026` (Eqs.43-44 rebuild) confirmed verbatim.
- `eos.hpp:58-64` (`alpha_from_mass_fraction`): confirmed bit-exact at `Y=0`/`Y=1` (`num`/`den`
  reduce to a single term, no cancellation) — the exact-skip rule (§2.4 below) is sound.
- `cases.cpp:446-463`: confirmed only `denner_water` (`pinf=4.4e8`) has nonzero `pinf` among the
  phases actually instantiated; every other phase (`denner_gas2`, `helium`, `matched_gas`,
  `helium_pure`, `argon`) has `pinf=0.0` — the §2.3 root-selection argument (at most one stiffened
  phase per pair ⇒ unique positive root) is grounded in the actual phase table, not assumed.

No structural error found. Proceeding to implementation per the staging below.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §31 for full detail)**:
Stage 0 = Branch A (r_init flattens under HREINIT, stall persists — sect.23.3's mechanism refuted,
proceeded with lowered prior on S1). Stage 1 closed form validated (worst rel_p=4.7e-11). Stage 2/3
gates: G1/G2/G5/G6/G7 pass; **G8 falsified** (cases 26/27/28 are not bit-exact pure in practice —
alpha≈0.999886, not 1.0 — so RECON legitimately acts on them; no pass/fail regression resulted).
**Verdict: S5 (harm)** — case24 gets real 20x progress (step 19→399, failure re-types to the
correctly-diagnosed T-ceiling-saturated) but cases 13/14 regress from PASS to FAIL (u-field quality
collapse). Per the stop rule below, flag stays OFF, not promoted, committed as gated-off research
infrastructure per round 4/8 precedent. consecutive_failures NOT incremented (mechanistically-
explained negative result, round 4/8/13 precedent).

---

## 0. Executive summary, and the one thing this plan changes about the round's premise

Round 13 §23.3 states the reason `ACID_YADV_HREINIT` failed:

> "correcting `s.h` alone still leaves `s.rho` (and everything `compute_R` derives from it) at its
> stale, `s0`-consistent value **until the first `compute_R()` call re-derives `T`/`rho` from the
> corrected `h`** — and by that point Newton is already iterating"

**This premise does not survive contact with the code (verified above).** `r_init` is captured
*after* `compute_R()` has already run once, and `compute_R`'s first two acts in `coupled` mode are
exactly the reconciliation §23.3 asks for (`T_from_hstat` at line ~1224, `eval_thermo` at line
~1258). So "reconcile `(T,ρ)` at the same instant as `h`, before `it==0`" as literally worded would
be a **provable no-op** relative to `HREINIT` on the coupled path.

**Reframing (this round's thesis).** The defect is not an *initial-guess* problem. It is a
**level-n conserved-state discontinuity injected by the alpha remap**:

- The Eqs.43-44 rebuild (`acid.cpp:1018-1026`) computes `rho_o`, `hstat_o`, `Htot_o` from the
  **new** alpha at `(p_o, T_o)` — these become the step's conserved reference state.
- But the level-n conserved state step *n−1* actually accepted used the **lagged** alpha.
- `alpha_new − alpha_prev` splits into `dal_adv` (`O(dt)`, physical) and `dal_remap` (`O(1)`,
  dt-independent, spurious — round 13 measured `dal_remap = 5.676e-2`, constant to 4-5 sig figs
  across all 13 retries).
- The `O(1)` remap piece appears in the transient as `Δ·dx/dt` ⇒ `r_init ∝ 1/dt` ⇒ the retry
  loop's `bad = (… && rbest >= r_init)` gate can never be satisfied by halving `dt` (halving `dt`
  *raises* the bar). That is the stall.

**The fix**: once per step, before the retry loop, re-derive `(p, T, alpha)` from the cell's own
**conserved** state `(ρ, e, Y)` — holding mass, momentum, total energy exactly fixed — so that
`alpha` becomes, by construction, the value the NASG p-T-equilibrium mixture actually implies for
that state. Closed form (a quadratic in `p`), no iteration, cannot saturate the `T_from_hstat`
ceiling, does not touch `compute_R`.

---

## 1. Code facts verified by reading (anchors)

All in `cpp/denner_1d/src/acid.cpp` unless noted.

| Anchor | What is there |
|---|---|
| `703` | time loop opens |
| `705-742` | CFL, `dt` finalized |
| **`747-748`** | `const Field s0 = s; const Vec Yv0 = Yv;` — **insertion point is immediately before this** |
| `772-774` | retry loop: `s = s0; Yv = Yv0;` |
| `777` | `p_o, T_o` captured from (reconciled) `s0` |
| `995-1011` | Y transport + alpha recovery at `(p_o,T_o)` |
| **`1018-1026`** | Eqs.43-44 rebuild → `rho_o`, `hstat_o`, `Htot_o` |
| `1034-1062` | `RMISM` (round 13, `ACID_RINIT`) |
| `1064-1084` | `RCELL` (round 16) |
| `1086-1097` | `ACID_YADV_HREINIT` (round 13) |
| `1115-1155` | `Cold_mom/con/ene` built from `rho_o, u_o, Htot_o` |
| `1193` | `compute_R` lambda opens |
| `~1224-1236` | coupled `h→T` inversion via `T_from_hstat` |
| `~1258` | `eval_thermo(s, A, B)` |
| `1400-1440` | residual: `trans_c`, `trans_e`, `srcp` |
| `1576` | first `compute_R()` of iteration `it` |
| `2022` | `if (it == 0) r_init = n0;` — **after** line 1576 |
| `2299` | `bool bad = (ajac && coupled && !conv_inner && rbest >= r_init);` |
| `2328-2331` | F2''/reason-5 T-ceiling scan, unconditional since round 20 |
| `334-362` | `T_from_hstat` — state-pure (round 17) |
| `eos.cpp:29-45` | NASG `phase_props`: `A = cv·T·(γ−1) + b(p+Π)`, `ρ = (p+Π)/A`, `h = γ·cv·T + b·p + η` |
| `eos.cpp:73-127` | existing `recover_pressure_temperature_from_density_energy` — frozen-alpha 2×2 Newton, used only by `denner1d_unit.cpp` |
| `eos.hpp:53-70` | `mass_fraction_from_alpha`/`alpha_from_mass_fraction`, bit-exact at `Y∈{0,1}` (verified) |
| `acid.cpp:485,498` | cases 24/33/34 have `bdf2=false` ⇒ BE path ⇒ `ajac=true` |
| `cases.cpp:446-463` | at most one phase per pair has `pinf≠0` (verified) |

Two clarifications to prior rounds' wording (annotations, not edits):
1. §23.1's "OFF prints zero diagnostic lines ⇒ structural immunity" is about `dal_remap` only;
   `RMISM` is `yadv`-gated by construction. The Eqs.43-44 rebuild itself is not `yadv`-gated.
2. Cases 24/33/34 have spatially uniform alpha but non-uniform Y (Y is evaluated at local (p,T)
   against a shock IC).

---

## 2. The derived reconciliation

### 2.1 What is held fixed

Hold `(ρ, u, e, Y)` fixed (all directly conserved/transported); re-derive `(p, T, alpha, h_stat, h)`.
Mass, momentum, total energy conserved exactly. `alpha` becomes
`alpha_from_mass_fraction(Y, ρ_a(p*,T*), ρ_b(p*,T*))` by construction ⇒ `dal_remap ≡ 0` at the next
recovery.

**Honest cost**: the reconciled state no longer exactly satisfies step n−1's own residual `R=0`
(same trade `+ALPHA_IMPLICIT` makes inside every residual eval; here made once per step, outside
`compute_R`).

### 2.2 Closed form (exact, no iteration)

Per-phase NASG: `v_k = (γ_k−1)cv_k T/(p+Π_k) + b_k`, `h_k = γ_k cv_k T + b_k p + q_k`.

Mixture at p-T equilibrium, mass fraction Y of phase a:
```
b̄=Y b_a+(1−Y)b_b   q̄=Y q_a+(1−Y)q_b   c̄p=Y γ_a cv_a+(1−Y)γ_b cv_b   c̄v=Y cv_a+(1−Y)cv_b
K_a=Y(γ_a−1)cv_a    K_b=(1−Y)(γ_b−1)cv_b
v(p,T)=b̄+T·S(p), S(p)=K_a/(p+Π_a)+K_b/(p+Π_b)      ... (I)
h(p,T)=c̄p T+b̄p+q̄,  e=h−pv                            ... (II)
```
Given `v_t=1/ρ_prev`, `e_t=h_stat,prev−p_prev v_t`, `W=v_t−b̄`, `E0=e_t−q̄`:
```
T(p) = (E0+W p)/c̄p                                    ... (III)
A2 p² + A1 p + A0 = 0
A2 = W c̄v
A1 = c̄p W(Π_a+Π_b) − W(K_a Π_b+K_b Π_a) − E0(K_a+K_b)
A0 = c̄p W Π_a Π_b − E0(K_a Π_b+K_b Π_a)
```
`T* = (E0+W p*)/c̄p`, `alpha* = alpha_from_mass_fraction(Y, ρ_a(p*,T*), ρ_b(p*,T*))`.

Sanity checks (do as unit tests): ideal single-phase reduces to `p*=(γ−1)ρ(e−q)`; single-NASG-phase
factors correctly; `E0>0` guaranteed for admissible input.

### 2.3 Root selection (unambiguous in this suite)

Since every phase pair here has at most one `Π≠0` (verified §1), `A0 ≤ 0` always ⇒ opposite-sign
roots when `A0<0` (take the unique positive one), or `{0, −A1/A2}` when `A0=0` (take the nonzero
one). Stable eval via `qq = −0.5(A1+copysign(sqrt(disc),A1))`, roots `qq/A2`, `A0/qq`.

Per-cell acceptance (fail-safe — untouched if any check fails): `W>0` ∧ `disc≥0` ∧ `p*≥1` ∧
`T*∈(1e-6,1e6)` ∧ finite. The `T*` bound keeps this F2''-clean by construction — no inner Newton to
saturate, so the reconciliation cannot trip reason 5 itself.

### 2.4 Parameter-free exact skip

```cpp
al_chk = clamp(alpha_from_mass_fraction(Yv[i], rho_a(s.p[i],s.T[i]), rho_b(s.p[i],s.T[i])), 0, 1);
if (al_chk == s.alpha[i]) continue;   // exact FP equality, no tolerance
```
Bit-exact skip for pure cells (Y∈{0,1}, verified §1) and undisturbed cells. Makes the fix
automatically local to the lag region and free elsewhere.

### 2.5 What gets written

Per accepted cell: `s.p[i]=p*; s.T[i]=T*; s.alpha[i]=alpha*;` then one `eval_thermo` call, then for
touched cells only `s.h[i] = s.hstat[i] + 0.5*u[i]^2`. Skipped cells' `h` must NOT be rewritten.

### 2.6 Why this removes the `1/dt` growth

After reconciliation, `alpha_prev = alpha*` exactly ⇒ `dal_remap ≡ 0` at the next recovery ⇒
`rho_o = ρ_prev + O(dt)`, `Htot_o = h_prev + O(dt)` ⇒ `trans_c`, `trans_e` become `O(1)` not
`O(1/dt)` ⇒ `r_init` becomes dt-independent ⇒ the `bad` gate is no longer un-satisfiable by
dt-halving.

### 2.7 Named residual risk (pre-registered)

Once the `O(1)` mismatch is gone, roundoff amplified by `VdT=dx/dt` at very small `dt` still gives
a residual floor `∝1/dt`. Candidate explanation for why `HREINIT` made case34 *worse* (lowering
`r_init` can make the `bad` gate easier to trip). Must be measured (Stage 0), not assumed.

---

## 3. `compute_R` state-purity

Reconciliation runs once per step, between `dt` finalization (line 742) and `s0` snapshot (line
747) — **outside** the retry loop and **outside** `compute_R`. Every retry's `s=s0` restores the
reconciled state identically. Pure function of `(s.p,s.T,s.alpha,s.rho,s.hstat,s.u,Yv,A,B)` — no
call history. `compute_R` itself is not modified; round 17's invariant (approximate Jacobian
changes iteration count only, never the converged answer) is preserved verbatim.

---

## 4. Literature check

**In repo (dedup, no re-download)**: `papers/library/md/newest5/2025_Collis_..._four_equation_thermodynamic_ENO.md` §2.3 — direct prior art: closed mixture pressure/temperature when at most one phase has `pinf≠0`, exactly our hypothesis. Cited as independent confirmation, not a formula source (their equations are page images).

**Needed (to be filed as stubs by implementing session)**:
- `papers/2026_Clayton_McConnell_Solomon_PTE_four_equation_needed.md` — arXiv:2606.27726, existence/uniqueness of PTE solution.
- `papers/2016_LeMetayer_Saurel_NASG_EOS_needed.md` — doi:10.1063/1.4945981, the NASG EOS itself.
- `papers/2011_Flatten_Morin_Munkejord_stiffened_gas_equilibrium_needed.md` — doi:10.1137/100784321, existence/uniqueness for stiffened-gas equilibrium.
- `papers/2017_Chiapolino_Boivin_Saurel_fast_relaxation_needed.md` — fast UV-flash analogue.
- `papers/2021_assessment_nonconservative_four_equation_needed.md` — arXiv:2105.12874, conservative-vs-non-conservative trade assessment.

---

## 5. Staging

Nothing promoted to default this round (round 4/13/18 precedent).

### Stage 0 — zero code, test the round's own premise (can cancel Stages 1-3)

```bash
D=./build-cpp/cpp/denner_1d/denner1d_dump
DENNER_ACID=1 ACID_YADV=1 ACID_RINIT=1 ACID_BLK_STEP=19 $D 24 2>&1 >/dev/null | grep -E "^RINIT|^RMISM"
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_ALPHA_IMPLICIT=1 ACID_RINIT=1 ACID_BLK_STEP=19 $D 24 2>&1 >/dev/null | grep -E "^RINIT|^RMISM"
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_HREINIT=1 ACID_DBG=1 $D 24 2>&1 >/dev/null | grep STALLED
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_HREINIT=1 ACID_RINIT=1 ACID_BLK_STEP=<stall_step> $D 24 2>&1 >/dev/null | grep -E "^RINIT|^RMISM"
```
Branches pre-registered: A (r_init flattens, stall persists → proceed, lowered prior on S1) /
B (r_init flattens somewhat but still ~doubles → roundoff floor, proceed toward S3) /
C (r_init barely moves → code reading wrong, STOP, write no code).

### Stage 1 — closed form as pure function + unit test + read-only `ACID_RECON` diagnostic

`eos.hpp`: `pT_from_v_e_massfrac(v,e,Y,a,b) -> {p,T,ok}`. `denner1d_unit.cpp`: identity round-trip
over a grid, cross-check vs existing `recover_pressure_temperature_from_density_energy`, pure-end
tests, gas-gas degenerate, rejection test. `ACID_RECON` (yadv-gated, default OFF, stderr-only,
applies nothing): per-step lag magnitude report.

Gate: `denner1d_unit` passes; unset ⇒ byte-identical to HEAD on A/B/C.

### Stage 2 — the fix, `ACID_YADV_RECON` (default OFF)

Gated `yadv && yrecon`, at the insertion point. Per cell: exact-skip check, closed-form solve,
fail-safe untouched on rejection, `eval_thermo` once, `h` refresh for touched cells only. No level
knob. Must not compose with `ACID_YADV_HREINIT` (leave HREINIT unset in all Stage 2/3 runs).

### Stage 3 — sweep and gates

G0 unit test. G1 `--verify` (OFF 19/19 byte-identical vs solver_denner). G2 `--sweep` unaffected
(A19/B15/C14/D13/E14/F14/G15, per round 20's `EXPECTED`, flag unset). G3 diff hygiene. G4 new
configs B/C/G+RECON measured. **G5 falsification F-a**: `C+RECON` ≈ `C` (near-identity, since
`+ALPHA_IMPLICIT` already keeps alpha reconciled). **G6 falsification F-b**: `dal_remap→DBL_EPSILON`
scale under B+RECON. G7 case01 `linf_p=0` under B+RECON. G8 cases 26/27/28/32 byte-identical
(pure-cell skip). G9 targets 24/34/33 measured with/without `STALL_ACCEPT`. G10 diagnostic-only
no-op. G11 perf sanity (≤2% overhead expected).

---

## 6. Predictions and falsification criteria

- **OFF/B/C/D/E/F/G, flag unset**: provably unaffected (zero new executable statements run).
  Falsified by any byte difference.
- **Cases 26/27/28/32, case01 under B+RECON**: bit-exact skip ⇒ byte-identical / `linf_p=0`.
  Falsified by any difference — would itself be a real, reportable finding.
- **C+RECON ≈ C** (G5): the strongest mechanism check — `+ALPHA_IMPLICIT` already keeps alpha
  reconciled to Newton tolerance, so RECON should have almost nothing to do there. If it differs
  materially, stop and explain before trusting any other number.
- **Case 24/34**: `dal_remap` collapse to DBL_EPSILON scale is near-certain (follows by
  construction). Whether this actually resolves the stall (vs. hits the §2.7 roundoff floor) is
  the genuine, ~even-odds experiment — case34 more likely to disappoint than case24, mirroring
  round 13's `HREINIT` asymmetry.
- **Case 33**: predicted IRRELEVANT (no help, no harm) — round 15 already measured its
  `dal_remap` at literal DBL_EPSILON; its driver is the ADVECTION channel (round 16 §26.1), not
  REMAP. Sharpest check: step 0 must be bit-unchanged (Yv was just derived from s.alpha at the
  same (p,T)). Any movement at all in case33 is the round's most interesting finding and must not
  be rationalized away.
- **13/14/25 under B+RECON**: must keep passing — round 4's `+ALPHA_IMPLICIT` regression came from
  a residual/Jacobian alpha disagreement; RECON changes neither, so that failure mode should not
  recur. If they regress anyway, that argument is directly refuted and the flag stays off
  regardless of 24/34's outcome.

---

## 7. Stop / decision rule (declared in advance)

| Outcome | Trigger | Consequence |
|---|---|---|
| S1 strong success | 24 AND 34 complete to t_end under B+RECON with STALL_ACCEPT unset, no regression | Round 22 = promotion evaluation (7-config battery, round-20 style). Do not promote this round. |
| S2 partial | stall persists but r_init now dt-dependent, and materially further progress (≥10x t, or fewer accepted steps than round 12's 2/4) | Flag stays OFF; follow-up round targets the remaining dt-inconsistent term (rho_star continuity predictor, theta_o MWI memory). |
| S3 null | dal_remap→0 confirmed, stall unmoved, r_init still grows ∝1/dt | Negative result reported honestly; consecutive_failures NOT incremented (round 13 precedent). One follow-up permitted if the new source is localized by measurement. |
| S4 stop chasing incrementally | dal_remap→0, r_init loses 1/dt growth, stall persists against an O(1) dt-independent residual | Halt the whole (T,ρ,h)/initialization line — the closure conflict (round 10 §20.4 / §11.6) reasserts itself. Needs explicit Advisor/user decision on scope (declare 24/33/34 structurally unreachable, or a closure change). |
| S5 harm | B+RECON < 15/19, or any hard gate fails | Flag stays OFF, regression reported honestly; round does not merge if a hard gate fails. |

Hard abort before any code: Stage 0 Branch C.

---

## 8. Risks / pitfalls carried forward

`DENNER_ACID=1` required on every invocation (round 20 §30.2). Round 20 baselines are live
(`TSAT_STALL` gone; case33's `+IMPLICIT` stall now at step 43). Never hardcode `ACID_BLK_STEP` —
read the stall step first. No inline WSL `for` loops / shell redirection for captures. No
`-march=native`. Do not touch `cases.cpp`/`validation.cpp`. Never set `HREINIT` in Stage 2/3 runs.
Copy `_NAN_RE`-style nan handling in any new script.

---

## 9. Deliverables

`docs/YADV_RESEARCH.md` §31 (31.1 Stage 0 verdict — annotate §23.3, don't edit; 31.2 derivation +
unit-test evidence; 31.3 `ACID_RECON` lag measurement; 31.4 gate results; 31.5 target
measurements; 31.6 verdict against §7; 31.7 reproducing). `docs/YADV_ROADMAP.md` update. 5
`papers/*_needed.md` stubs. Commits per project convention (feat/docs/chore), local merge only.
