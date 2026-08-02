# YADV Round 26 Plan — Is the case 24/33/34 gate *reachable at all* under `ACID_YADV=1`? A reachability-vs-fidelity decomposition of the binding constraint

**Thread**: round 25's own designated next step, roadmap thread **(b)** — "identify the actual binding constraint keeping 24/33/34's now-finite F3 completions from `validate pass:true`". Resolves thread **(c)** (case33's `corr_p` sign flip) as a corollary. Defers thread (a) (F3b) with an explicit, evidence-based reason.

**Nature of the round**: diagnostic-only. **Zero C++ changes.** One new script. Precedent: rounds 10, 15, 19.

**Advisor verification note**: key structural claims checked directly against the code -- `cases.cpp:148` (`s.alpha_post = s.alpha_pre;  // psi held (homogeneous mixture)`, exact match), `acid.cpp:1257-1266` (conservative `rho*Y` update, `Yv = anew;` at line 1266, exact match), `validation.cpp:469-496` (`case24_spec_pass`: `l2_p/u/rho<=0.20`, `corr_p/u/rho>=0.92`, `dip<=0.02`, `hump<=0.01`, `plateau_l2<=0.015`, exact match), `validation.cpp:49` (`den <= 1.0e-300 -> return 1.0` degenerate-correlation branch, exact match, off by one line from the plan's cited 48-50 -- immaterial). No structural error found.

**Actual outcome (appended after execution, see `docs/YADV_RESEARCH.md` §36)**: P0 confirmed live
-- the Python closure-A/closure-B/star-state solver reproduces every number in §3.1/§3.2's hand
tables to FULL double precision (not merely 5 sig figs), an independent cross-check via a
different code path than the hand derivation. P1 confirmed: the Python gate re-implementation
matches `denner1d_validate`'s own JSON to spec (6 sig figs) and the exact `pass` verdict on 7/7
`(case,config)` pairs. **T1/T2 confirm S1 with a wide margin**: the exact config-B Riemann
solution fails the gate by 2-3.4x on `l2_p/l2_u/l2_rho` and 31-40x on `dip` for all three cases
(§3.4's table reproduced live exactly, after fixing a script bug this round found in its own
first draft -- the initial 3-segment profile omitted the left-shock jump between the IC's L state
and the *L star state, which silently gave `corr_p=+1.0` everywhere instead of the predicted
negative value; fixed to the correct 4-segment profile, then §3.4 reproduced exactly). No member
of the entire reachable single-shock family passes either. **T3 leans S2, not cleanly**: of 11
runs whose observation window passed the flatness guard, 10 land within 0.17%-4.83% of the exact
closure-B plateau; one (`p` under plain `B+F3`, case33) reads 6.15%, 1.15 points over the plan's
own pre-registered 5% bar -- reported as measured, not rounded toward the cleaner verdict. **T6
resolves thread (c)** exactly as predicted: case33's `corr_p` sign flip under F3 is motion toward
the exact model's own increasing-pressure answer (`-0.2848`), not a fidelity regression. **T7
confirms case15's redirect target is sound** (reference tracks the active config). Verdict:
**S1 + S2 (with an honest caveat)**. `consecutive_failures` NOT incremented (round 5/9/11
precedent). Per S1's own pre-registered consequence, **this round escalates to the user** rather
than choosing unilaterally among the three ways to reconcile the goal with this finding -- the
anti-rescue clause explicitly forbids the loop from picking one on its own. F3b (§8) stays
deferred, now understood to target a residual defect an order of magnitude smaller than this
round's measured gap. All hard gates held (OFF 19/19, `ALL GATES OK` unchanged, zero `cpp/` diff,
unit-test numbers unchanged).

---

## 0. Executive summary

Rounds 11-25 have all asked *fidelity* questions about cases 24/33/34: is the solver stalling, is `alpha` lagged, is `drho`/`dh` at the recovery site too big, does F3 make the run finite. Round 25 answered the last one *yes* — and the cases still fail. Nobody since round 2 has asked the prior, cheaper **reachability** question:

> **If the solver were perfect — if it produced the *exact* solution of the model `ACID_YADV=1` actually implements — would cases 24/33/34 pass `case24_spec_pass`?**

That question is answerable in closed form, by hand, today, with no new solver code, because:

1. `ACID_YADV=1` transports `rho*Y` conservatively (`acid.cpp:1257-1266`). Therefore across the leading shock the mass fraction `Y` is **structurally** conserved (uniform ahead ⟹ uniform behind). This is not a numerical property, it is what the conservative update *is*.
2. The validation reference for 24/33/34 (`cases.cpp:779-791`, built from `compute_case24_shock`) is an exact step whose post-shock state holds the **volume** fraction: `s.alpha_post = s.alpha_pre` (`cases.cpp:148`, literal assignment). §11.3 already established this forces `Y` to grow **+268x to +1621x** across the shock.
3. So the reference and the model are two *different* closures of the same 3 jump conditions (`YADV_RESEARCH.md` §11.2's closure (A) vs (B)) — both exact RH solutions, differing in physics by O(1).

I computed the exact answer this round (§3, §4, all verified to machine precision). **The exact solution of the config-B model, with the cases' own initial data and transmissive BCs, is a three-wave Riemann fan (left shock | contact | leading shock) whose leading shock is 42-61% overfast and has left the 800-cell domain by `t_end`, and whose gate metrics are:**

| case | `l2_p` | `l2_u` | `l2_rho` | `corr_rho` | `dip` |
|---|---|---|---|---|---|
| threshold | ≤ 0.20 | ≤ 0.20 | ≤ 0.20 | ≥ 0.92 | ≤ 0.02 |
| **24 exact** | **0.6689** (3.3x) | **0.4001** (2.0x) | **0.4337** (2.2x) | **0.5318** | **0.7586** (38x) |
| **33 exact** | **0.6818** (3.4x) | **0.3986** (2.0x) | **0.4267** (2.1x) | **0.5667** | **0.8043** (40x) |
| **34 exact** | **0.6014** (3.0x) | **0.4080** (2.0x) | **0.4043** (2.0x) | **0.4877** | **0.6214** (31x) |

and, more strongly, **no member of the entire admissible single-shock family can pass either**, because the density behind *any* `Y`-conserving shock into these pre-shock states is bounded (case24: `rho ≤ 988.10`) far below what the gate demands (`rho ≥ 1553.68` for `l2_rho`, `rho ≥ 1830.10` for `dip`).

**Hypothesis under test (H-STRUCT)**: the binding constraint on 24/33/34 is not the recovery-site defect, not the stall, not the Jacobian, and not anything F3b could fix — it is that **`ACID_YADV=1` implements a different closure than the one cases 24/33/34 grade against**, and the gap is O(1), not O(numerical error).

**Falsifier (H-NUM)**: if the exact-solution gate scores come out inside or near the thresholds, my §3/§4 arithmetic is wrong, H-STRUCT dies, and the binding constraint is numerical — in which case Stage 3 hands round 27 a measured target (the gap between the solver and the exact model answer).

The round is designed so **both** outcomes are load-bearing, and so that a third, independent question gets answered on the way: **is the solver *correct for the model it implements*?** (Stage 3 re-runs §11.5's O(1) RH-violation measurement on the current code — that measurement is from round 2's long-dead non-conservative implementation and has never been repeated.)

### Why this thread and not the alternatives

- **(a) F3b (same-triple restoration)** — deliberately NOT built. Round 25 recorded its trigger as only partially met, and the middle clause ("T3's residual `drho` is the plausible remaining obstacle") reads *false* after this plan's §4: the residual `drho` after F3 is O(50) on a state whose *model-level* error is O(1000) in `rho` (case24: exact-model plateau 827.3 vs reference 1857.3). Building F3b before knowing whether the target is reachable would repeat rounds 21-25's pattern of paying implementation cost for a constraint that was never the binding one. **If Stage 2 falsifies H-STRUCT, F3b becomes the round-27 candidate with a measured motivation instead of a guessed one.**
- **(c) case33's `corr_p` sign flip** — subsumed, at zero extra cost. §4.4 predicts, *before* looking, that the exact model answer for case 24 and 33 has **negative** `corr_p` (−0.2090, −0.2848), because the model's own solution is an *increasing* pressure step (`pA` behind the left shock → `p* > pA` ahead of it) while the reference is a *decreasing* step. Round 25's observed `+0.351 → −0.817` is then the signature of the solution moving *toward* the model's own answer, i.e. a fidelity **improvement** showing up as a gate **regression**. Directly testable from the existing dumps.
- **(d)-(h)** — narrower, or blocked, and none of them can change the answer to "should we keep working on 24/33/34 at all", which this round settles.

---

## 1. Verified code facts

Every line number below was read directly in this worktree at HEAD `12fb88d` (round 25's commit). Nothing is copied from a prior round's doc.

### 1.1 What `ACID_YADV=1` transports (`cpp/denner_1d/src/acid.cpp`)

| Line(s) | Content (verified verbatim) |
|---|---|
| `1257-1265` | conservative `rho*Y` update: `rY = rho_old*Yv[i] - dt/dx*(mdR_o[i]*af[i+1] - mdL_o[i]*af[i])`, `rho_star = rho_old - dt/dx*(mdR_o[i]-mdL_o[i])`, `anew[i] = clamp(rY/rho_star, 0, 1)` |
| `1266` | `Yv = anew;` — the transported colour function is `Y` |
| `1267-1273` | the stale-`(p_o,T_o)` alpha recovery site: `s.alpha[i] = clamp(alpha_from_mass_fraction(Yv[i], phase_props(pu,Tu,A).rho, phase_props(pu,Tu,B).rho),0,1)` at `pu=max(p_o[i],1)`, `Tu=max(T_o[i],1e-6)` |
| `1285-1316` | round 25's F3 block (`yadv_f3 \|\| f3_dbg`); the only `s.*` write is `s.alpha[i] = al_f3;` at `1302` |
| `1323-1331` | Eqs.43-44 rebuild of `rho_o`/`hstat_o`/`Htot_o` at OLD `(p_o,T_o)` from whatever `s.alpha` is live |
| `1333-1367` | `RMISM` (`ACID_RINIT`) — `dh`, `drho`, `dal`, `dal_remap`, `dal_adv` |
| `1369+` | `RCELL` (`ACID_RCELL`) per-cell window print |
| `746-747` | `const bool f3_dbg = getenv("ACID_F3"); const bool yadv_f3 = getenv("ACID_YADV_F3");` |
| `784-804` | `ACID_TEND_SCALE` — multiplies **this solver's** stop time only; default 1.0 = byte-identical. **Its own comment (`791-794`) warns: with scale ≠ 1 the dump's `*_ref` columns and every `denner1d_validate` metric are meaningless; only the solver columns `p,u,rho` are valid.** This round uses it *only* in that mode. |

**Consequence used by this plan**: `Y` enters the mass/momentum/energy update only through `s.alpha` → `rho_o`/`hstat_o` and the ACID mass flux. Nothing in the update can create or destroy phase mass; the `rho*Y` sweep at `1257-1266` is the sole `Y` evolution and it is a conservative flux-difference form with the *same* face mass flux `mdR_o`/`mdL_o` used for the mixture. Therefore a uniform `Y` ahead of a shock is transported unchanged behind it, up to the flux limiter's own boundedness clamp at `1264`. This is the structural premise of §3.

### 1.2 What cases 24/33/34 are (`cpp/denner_1d/src/cases.cpp`)

| Line(s) | Content |
|---|---|
| `105-151` | `compute_case24_shock(a, b, alpha_air)` — Denner Eq.57-62. `Vs = 10 * sqrt((gamma_mix-1)*cp_mix*T_pre)`; `Pihat` (Eq.60) at `140`; pressure multiplier (Eq.59) at `141-143`; `rho_post` (Eq.61) at `146`; `u_post` (Eq.62) at `147`; **`s.alpha_post = s.alpha_pre;  // psi held (homogeneous mixture)` at `148`** |
| `689-694` | IC: `post = (x < 0.1)`; `alpha[i] = post ? sh24.alpha_post : sh24.alpha_pre` — **`alpha` is uniform, `Y` is not** |
| `779-791` | reference: exact step at `x_shock = 0.8`, post = closure-(A) `(p,u,T,alpha)`, pre = pre-shock state. **Config-independent — it is analytic, not solver-computed.** |
| `195-196` | grid: `dx=(x1-x0)/n`, `x[i] = x0 + (i+0.5)*dx` → for `N=800` on `[0,1]`, `x_i = (i+0.5)/800`, `dx = 1.25e-3` |
| `495-496`, `543-546` | `c24/c33/c34 = base_config(800, 0.7/Vs, 0.0, 1.0)` with `alpha_air = 0.5 / 0.75 / 0.25` |
| `446` | `denner_water{4.1, 4.4e8, 0.0, 474.2, 0.0}` (`gamma, pinf, b, kv, eta`) |
| `750-753` | **case15's reference is `computed_reference(c, 800)`** — the *same solver under the same env* on a finer mesh (`423-437`). So case15 has **no** closure mismatch by construction; its reference tracks whatever config is active. |
| `types.hpp:36-37` | `left_bc = right_bc = "transmissive"` (default, not overridden for 24/33/34) |
| `eos.cpp:11-15` | `air_phase() = {1.4, 0.0, 0.0, 720.25, 0.0}` |

### 1.3 What the gate actually is (`cpp/denner_1d/src/validation.cpp`)

| Line(s) | Content |
|---|---|
| `731-734` | cases `"24" \|\| "33" \|\| "34"` all dispatch to `case24_spec_pass` |
| `469-505` | `case24_spec_pass` = `m.finite && profile_ok && plateau_ok` |
| `493-495` | `profile_ok`: `l2_p,l2_u,l2_rho ≤ 0.20` **and** `corr_p,corr_u,corr_rho ≥ 0.92` |
| `473` | `x_shock = gradient_peak_x(ref.p, ref.x, 0.05, 0.99)` — of the **reference**, so `= 0.8` always |
| `477-492` | plateau window `lo=0.005`, `hi = x_shock - max(10*dx, 0.03) = 0.77`; `jump = max(\|rho_post-rho_pre\|,1)`; `dip = max (rho_post - got.rho[i])/jump`; `hump = max (got.rho[i]-rho_post)/jump`; `plateau_l2 = sqrt(mean (got.rho-ref.rho)^2)/jump` |
| `492` | `plateau_ok`: `dip ≤ 0.02 && hump ≤ 0.01 && plateau_l2 ≤ 0.015` |
| `18-26` | `rel_scale(ref) = max(max(ref)-min(ref), 1.0)` — the normaliser is the **reference's own range** |
| `317-359` | `accumulate`: `l2 = sqrt( mean_i ((a_i-b_i)/scale)^2 )` over **all** cells |
| `49` | `correlation`: Pearson; **`if (den <= 1.0e-300) return 1.0;`** ← a *constant* `got` field scores `corr = 1.0`, a degenerate pass. Load-bearing for case34 in §4.3. |
| `405-412` | `compare()` calls `accumulate` on `p`, `u`, `rho` |
| `752-774` | `metrics_json` emits `pass`, `finite`, `l2_p/u/rho`, `corr_p/u/rho`, `linf_*`, ... at default `ostringstream` precision (**6 significant digits**) — sets the P1 tolerance in §5.1 |

### 1.4 Instruments already available (no new code needed)

- `apps/denner1d_dump.cpp:16-22` — prints `x,alpha,p,u,rho,p_ref,u_ref,rho_ref` at `setprecision(12)`. Everything §6 needs.
- `scripts/yadv_yprofile.py:22-30` — reconstructs `T` and `Y` from `(p,rho,alpha)` (both phases have NASG `b=0`, so `T` is explicit). **Note: it hardcodes an absolute worktree path `W` at line 20 — the new script must derive `ROOT` from `__file__`, per `yadv_r9_sweep.py`'s pattern.**
- `scripts/yadv_rh2.py:52-64` — `last_step_fraction()`: parses `ACID_DBG`'s last `ACID step ... t=` to get the fraction of `t_end` actually reached. Required as a null-run guard (round 10's retracted findings).
- `scripts/yadv_r9_sweep.py:34-40` `ACID_ENV_VARS` (purge list, already contains `ACID_F3`/`ACID_YADV_F3`), `42-51` `CONFIGS`, `61-72` `EXPECTED` / `ALL_CASES`.
- `.claude/rules/denner-pitfalls.md`: **`DENNER_ACID=1` is mandatory** or the binaries silently run a non-ACID path at `pass_count=11/19`.

---

## 2. Derivation

### 2.1 The two closures (this is `YADV_RESEARCH.md` §11, re-verified, not re-derived)

Both phases have NASG `b = 0`, so at a single `(p,T)`:

```
v_k(p,T) = (gamma_k - 1) cv_k T / (p + pinf_k),      h_k(p,T) = gamma_k cv_k T + eta_k
```

For a mixture at p-T equilibrium with **mass** fraction `Y` of phase a (`eos.hpp:88-113` derives exactly this):

```
cpbar = Y*ga*cva + (1-Y)*gb*cvb          cvbar = Y*cva + (1-Y)*cvb
Ka    = Y*(ga-1)*cva                     Kb    = (1-Y)*(gb-1)*cvb        (Ka+Kb = cpbar-cvbar)
S(p)  = Ka/(p+pinf_a) + Kb/(p+pinf_b)
v(p,T) = T*S(p)                          h(p,T) = cpbar*T + qbar         e = h - p*v
```

**Closure (A), alpha-held** (what `cases.cpp:105-151` builds and what the OFF path reproduces exactly, because with `alpha` uniform the colour-function advection leaves it uniform for all time): `alpha_post := alpha_pre`. With `b=0` an alpha-frozen mixture is *exactly* an equivalent stiffened gas, so Eq.59-62 is not an approximation — §11.3 verified its Rayleigh/Hugoniot residuals at `1e-16`.

**Closure (B), Y-held** (what `acid.cpp:1257-1266` enforces): `Y_post = Y_pre`. Because `h` and `v` are both *linear in T at fixed p*, the Hugoniot is **explicit**, not iterative — this is new relative to §11.1's bisection and is what makes §3's tables hand-checkable:

```
h1 - h0 = 0.5 (p1 - p0)(v0 + v1),   v1 = T1 S(p1),   h1 = cpbar T1 + qbar
  =>  T1 = ( cpbar*T0 + 0.5 (p1-p0) v0 ) / ( cpbar - 0.5 (p1-p0) S(p1) )        (Eq. B1)
      v1 = T1 * S(p1)
      mdot^2 = (p1-p0)/(v0-v1),   Vs = mdot*v0,   u1 = Vs (1 - v1/v0)           (Eq. B2)
```

(`qbar = 0` for this phase pair; `eta_air = eta_water = 0`.)

### 2.2 The structural premise

> **P**: In any run with `ACID_YADV=1`, the state immediately behind the leading shock lies on the closure-(B) Hugoniot locus (Eq. B1-B2) emanating from the undisturbed pre-shock state.

Justification: (i) `Y` is uniform `= Y_pre` ahead of the leading shock at all times (undisturbed IC); (ii) `acid.cpp:1257-1266` updates `rho*Y` by a flux difference using the *same* face mass fluxes as the mixture continuity, with the face value `af[]` a convex reconstruction of neighbouring `Y` and the result clamped to `[0,1]` — a uniform `Y` state is therefore preserved exactly; (iii) mass/momentum/energy are the conserved quantities of the residual `compute_R`, so an admissible discrete weak solution satisfies their jump conditions; (iv) the 4-eq closure ties `(p,T)` to a single equilibrium pair. (i)+(ii)+(iii)+(iv) = the state behind the shock is on the (B)-locus.

**P is falsifiable** and is falsified exactly when the solver produces an inadmissible weak solution — which is precisely what §11.5 measured (88% momentum residual) on round-2's *non-conservative* implementation. Stage 3 re-tests P on the current code. This is why the round measures reachability *and* fidelity: they are independent, and P is the bridge.

### 2.3 The initial data is a closure-(A) state — so config B solves a Riemann problem

`cases.cpp:689-694` seeds `x<0.1` with `(p_A_post, u_A_post, T_A_post, alpha_pre)`. Its mass fraction is `Y_L = alpha_pre*rho_a(p_A,T_A)/rho_A`, hundreds of times `Y_pre` (§3.1). `alpha` is identical on both sides of `x=0.1`, so the alpha model sees no discontinuity in its transported variable — but the Y model sees a **material contact**. With `left_bc = "transmissive"` the left state is sustained. Hence config B's exact solution is a **three-wave Riemann fan** at `x=0.1`, not a single travelling shock:

```
 L (inflow, closure-A state, Y_L)  |  left shock  |  *L  |  contact  |  *R  |  leading shock  |  R (pre-shock, Y_pre)
```

with `p` and `u` continuous across the contact and `rho`, `Y`, `T`, `alpha` jumping. Both non-linear waves are **shocks** (verified: `p* > p_L` and `p* > p_R` in all three cases), so the exact solution is **piecewise constant** — four states, three straight wave rays — and the gate can be evaluated on it in closed form.

Star-state solve: `u*_L(p) = u_L - sqrt((p-p_L)(v_L - v_L(p)))` from Eq. B1 at `Y_L`; `u*_R(p) = 0 + sqrt((p-p_R)(v_R - v_R(p)))` from Eq. B1 at `Y_pre`; bisect on `u*_L(p) = u*_R(p)`.

### 2.4 The reachable-set argument (independent of §2.3, weaker assumptions)

Even discarding the specific initial data, premise **P** bounds what any config-B run can put behind its leading shock. Scanning the (B)-locus over all `p1 > p0` gives `rho_max^B`. The gate's `dip ≤ 0.02` requires `min_window(got.rho) ≥ rho_A_post - 0.02*jump`; the gate's `l2_rho ≤ 0.20` requires (since the plateau covers ≥ 80% of the domain whenever the shock is near `x=0.8`, and cells between a too-slow shock and `0.8` carry the *full* normalised error 1.0) `rho_plateau ≥ rho_A_post - 0.2236*jump`. Both are compared against `rho_max^B` in §3.3.

---

## 3. Stage-0 hand-computed tables

All numbers below were computed by hand/derivation **this round, before any code exists**, from `cases.cpp`'s own constants (`air = {1.4, 0, 0, 720.25, 0}`, `denner_water = {4.1, 4.4e8, 0, 474.2, 0}`, `rho_air_ref = 1.1574`, `rho_water_ref = 998.0`, `Ms = 10`). They are the P0 self-check target for Stage 1.

### 3.1 Closure (A) vs closure (B) at the reference shock speed — reproduces §11.2 exactly

| quantity | case24 (`alpha_pre`=0.50) | case33 (0.75) | case34 (0.25) |
|---|---|---|---|
| `a_mix` Eq.57 [m/s] | 642.6761 | 545.6494 | 820.1394 |
| `Vs_ref` [m/s] | 6426.7612 | 5456.4936 | 8201.3944 |
| `t_end = 0.7/Vs_ref` [s] | 1.089196e-04 | 1.282875e-04 | 8.535134e-05 |
| `rho_pre` | 499.57870 | 250.36805 | 748.78935 |
| `T_pre` [K] | 299.9835 | 299.9834 | 299.9836 |
| `Y_pre` | 1.1580449e-03 | 3.4661068e-03 | 3.8631321e-04 |
| **(A)** `p_post` [Pa] | 1.5083982e+10 | 5.8772374e+09 | 3.1623533e+10 |
| **(A)** `rho_post` | 1857.2573 | 1183.3461 | 2012.2044 |
| **(A)** `u_post` [m/s] | 4698.0438 | 4302.0286 | 5149.4596 |
| **(A)** `T_post` [K] | 16938.188 | 13837.320 | 21767.260 |
| **(A)** `Y_post` = `Y_L` | 8.3215405e-01 | 9.3438847e-01 | 6.2651470e-01 |
| **(B)** `p_post` @ `Vs_ref` | 8.2737665e+09 | 3.3897597e+09 | 1.9506482e+10 |
| **(B)** `rho_post` @ `Vs_ref` | 833.9767 | 459.1602 | 1222.1038 |
| **(B)** `u_post` @ `Vs_ref` | 2576.9259 | 2481.2101 | 3176.3574 |
| **(B)** `T_post` @ `Vs_ref` [K] | 7114.232 | 5689.285 | 11106.293 |

Every (A) and (B) entry reproduces `YADV_RESEARCH.md` §11.2 to all shown digits, computed here from Eq. B1-B2's **explicit** form rather than §11.1's bisection — an independent confirmation of round 2's table, obtained by a different method.

### 3.2 The exact config-B Riemann solution (new this round)

Star state from §2.3, verified: `u*` matches from both sides to `0.00e+00` relative; mass/momentum/energy residuals across the leading shock `≤ 2.2e-16` relative; entropy strictly increases across **both** shocks (`ds_right` = 2194 / 3029 / 1736 J/kg·K, `ds_left` = 0.992 / 1.087 / 0.439); the Lax conditions hold for the leading shock (`c_pre` = 20.0 / 23.1 / 23.1 m/s < `S_R` < `u* + c*`).

| quantity | case24 | case33 | case34 |
|---|---|---|---|
| `p*` [Pa] | 2.040993e+10 | 8.133623e+09 | 3.949956e+10 |
| `u*` [m/s] | 4022.796 | 3701.450 | 4534.419 |
| `rho*_L` (behind left shock, `Y=Y_L`) | 2208.396 | 1459.415 | 2227.475 |
| `rho*_R` (behind leading shock, `Y=Y_pre`) | **827.274** | **432.967** | **1227.067** |
| `S_L` [m/s] | +451.2 | +1127.1 | −1214.6 |
| `S_R` [m/s] | 10155.6 | 8776.6 | 11633.5 |
| **`S_R / Vs_ref`** | **1.5802** | **1.6085** | **1.4185** |
| `x_leftshock(t_end)` | 0.1491 | 0.2446 | −0.0037 |
| `x_contact(t_end)` | 0.5382 | 0.5748 | 0.4870 |
| **`x_leadshock(t_end)`** | **1.2061** | **1.2259** | **1.0929** |
| reference front | 0.800 | 0.800 | 0.800 |

**The leading shock is 42-61% overfast and has left the domain in all three cases.** Round 24 §34.5 saw exactly this signature for `ACID_PROJ_UNTIL=50` ("shock COMPLETELY EXITED the domain, 84% overstrong plateau, 32% overfast") and read it as a numerical pathology; §3.2 says it is the model's own answer.

### 3.3 The reachable-set bounds (independent of the initial data)

`rho_max^B` = max of `1/v1(p1)` over the entire admissible (B)-locus (`0 < v1 < v0`, `mdot^2 > 0`). Note it is attained at a *moderate* pressure, not asymptotically — the mixture Hugoniot's density is non-monotone in `p` (shock heating of the near-incompressible water eventually wins). Its `p→∞` asymptote is the mass-weighted `(gbar+1)/(gbar-1)` with `gbar = cpbar/cvbar`, also tabulated as a cross-check.

| case | `rho_pre` | `rho_A_post` | `jump` | **`rho_max^B`** (at `p =`) | `gbar` | asymptotic `rho` | needed for `dip≤0.02` | needed for `l2_rho≤0.20` | shortfall |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 499.579 | 1857.257 | 1357.679 | **988.103** (1.862e7) | 4.095254 | 822.382 | ≥ 1830.104 | ≥ 1553.680 | **1.85x / 1.57x** |
| 33 | 250.368 | 1183.346 | 932.978 | **953.685** (1.241e7) | 4.085811 | 412.639 | ≥ 1164.687 | ≥ 974.732 | **1.22x / 1.02x** |
| 34 | 748.789 | 2012.204 | 1263.415 | **1232.126** (→∞) | 4.098416 | 1232.126 | ≥ 1986.936 | ≥ 1729.705 | **1.61x / 1.40x** |

`dip` is over threshold by **32.0x / 12.3x / 30.9x** even at the most favourable point of the whole locus. `l2_rho`'s lower bound `sqrt(0.8)*(rho_A - rho_max^B)/jump` = **0.5726 / 0.2202 / 0.5523** vs threshold 0.20. Case33's `l2_rho` bound is only 1.10x over — honest caveat: **for case33 the `l2_rho` argument alone is not decisive; its `dip` argument (12.3x) is.**

### 3.4 The gate scores of the exact config-B solution (§3.2's profile, `N=800`, `x_i=(i+0.5)/800`)

| metric | threshold | case24 | case33 | case34 |
|---|---|---|---|---|
| `l2_p` | ≤ 0.20 | **0.6689** | **0.6818** | **0.6014** |
| `l2_u` | ≤ 0.20 | **0.4001** | **0.3986** | **0.4080** |
| `l2_rho` | ≤ 0.20 | **0.4337** | **0.4267** | **0.4043** |
| `corr_p` | ≥ 0.92 | **−0.2090** | **−0.2848** | +1.0000 † |
| `corr_u` | ≥ 0.92 | **+0.2090** | **+0.2848** | +1.0000 † |
| `corr_rho` | ≥ 0.92 | **+0.5318** | **+0.5667** | **+0.4877** |
| `dip` | ≤ 0.02 | **0.7586** | **0.8043** | **0.6214** |
| `hump` | ≤ 0.01 | **0.2586** | **0.2959** | **0.1704** |
| `plateau_l2` | ≤ 0.015 | **0.4562** | **0.4502** | **0.4012** |

† case34's left shock has just exited (`x_L = −0.0037`), so the idealised `p` and `u` fields are *exactly* constant over the grid, `va = 0`, and `validation.cpp:49` returns `1.0` — a **degenerate** pass of the correlation criterion, not a real one. The real solver will not be bit-constant; treat case34's `corr_p`/`corr_u` as unconstrained by this table (its `l2`s, `corr_rho`, `dip`, `hump`, `plateau_l2` all fail regardless).

**Reading**: `corr_p < 0` for cases 24/33 because the model's own solution is an *increasing* pressure step (`p_A` behind the left shock, then `p* > p_A`) while the reference is a *decreasing* step. This is §4.4's prediction for thread (c).

---

## 4. What is new versus what is already in the record

- §11.2's A/B table: **re-verified independently** (explicit Eq. B1 instead of bisection). Not new, but now hand-reproducible.
- §11.4's "the test problem DRIVES the Y path with a closure-(A) state" and "the shock is overdriven, front at 0.939 / left the domain": **known qualitatively**. What is new is the *exact* solution — `p*`, `u*`, `rho*_L`, `rho*_R`, `S_L`, `S_R`, and the resulting **gate scores**. §11.4 never asked what score the exact answer earns.
- §11.6 concluded "solver defect, not a modelling difference", from §11.5's 88% RH violation on round-2's non-conservative code. **This round does not retract that**; it observes that §11.6 answered a *fidelity* question and never asked the *reachability* question. Both can be true. Stage 3 re-measures the fidelity half on the current code.
- Round 10's "may make 24/33/34 a structurally unreachable target for any Y-preserving scheme, pending Stage 1-4" is the closest prior statement. It was a flagged *suspicion* with no quantitative test; rounds 11-25 proceeded as if it were false. This round converts it into a number.

---

## 5. Staging

Zero C++ changes. One new file: **`scripts/yadv_r26_closure.py`** (`ROOT` derived from `__file__`, `base_env()` copied from `yadv_r9_sweep.py`'s pattern including `DENNER_ACID=1` and the `ACID_ENV_VARS` purge). Sub-commands `--stage0 --gatecheck --gate --autopsy --window`.

### Stage 1 — build the instrument and prove it (P0, P1). Independently falsifiable.

1a. **Closure solver** (`--stage0`): implement `closureA()` (Eq.57-62, transcribed from `cases.cpp:105-151`) and `hugoniotB()` (Eq. B1-B2), plus the two-shock star solve of §2.3.
   - **P0**: reproduce §3.1 and §3.2 to **≥ 5 significant digits**. Also assert internally: `|u*_L − u*_R|/|u*_R| < 1e-12`; mass/mom/energy residuals across each shock `< 1e-12`; `ds > 0` across both shocks; Lax holds for the leading shock. **If P0 fails, everything downstream is void — stop and fix before reporting anything.**

1b. **Gate re-implementation** (`--gatecheck`): re-implement `rel_scale` (`validation.cpp:18-26`), `correlation` **including the `den ≤ 1e-300 → return 1.0` branch** (`:49`), `accumulate` (`:317-359`), `gradient_peak_x` (`:445-460`) and `case24_spec_pass` (`:469-505`) in Python, then feed it `denner1d_dump`'s own `(p,u,rho)` and `(p_ref,u_ref,rho_ref)` columns.
   - **P1 (hard prerequisite)**: for every `(case, config)` in `{24,33,34} x {A(OFF), B, B+F3}` plus `{26,27,28} x {A}` (which exercise the same `l2`/`corr` path through `single_shock_pass`), the Python `l2_p/l2_u/l2_rho/corr_p/corr_u/corr_rho` must match `denner1d_validate --only <case>`'s JSON to **6 significant digits** (the JSON's own precision), and the Python `case24_spec_pass` verdict must match the JSON `pass` field in every one of those runs. **P1 failing ⇒ S6: report the instrument failure, publish nothing downstream.**

### Stage 2 — the reachability answer. Independently falsifiable.

2a. `--gate` prints §3.4's table computed by the *validated* Stage-1b gate on the *validated* Stage-1a exact solution, alongside the thresholds and the ratio to threshold.

2b. The **reachable-set scan**: sweep `p1` over the entire admissible (B)-locus (log-spaced, `p0*1.0001` to `1e18`, with the admissibility filter `0 < v1 < v0`, `mdot^2 > 0`), and for each member build the idealised step profile (post-state for `x < 0.1 + Vs(p1)*t_end`, pre-state beyond) and score it. Report the **per-metric minimum over the family** and `rho_max^B` (§3.3). This tests the weaker-assumption version of the claim.

2c. **The one-line answer**: does *any* member of 2b, or the exact solution of 2a, satisfy `case24_spec_pass`?

### Stage 3 — the fidelity answer (is the solver right for its own model?). Independently falsifiable.

The leading shock has left the domain by `t_end` (§3.2), so the jump cannot be measured there. Use `ACID_TEND_SCALE = 0.6` (`acid.cpp:784-804`) — an observation-window knob built for exactly this in round 11 — and honour its own warning: **only the `p,u,rho` solver columns are read; `*_ref` and all `validate` metrics are discarded for these runs.**

Predicted structure at `sigma = 0.6` (from §3.2, hand-computed now):

| case | `x_leftshock` | `x_contact` | `x_leadshock` | safe `*R` window |
|---|---|---|---|---|
| 24 | 0.1295 | 0.3629 | 0.7637 | `[0.45, 0.70]` |
| 33 | 0.1868 | 0.3849 | 0.7756 | `[0.45, 0.70]` |
| 34 | 0.0378 | 0.3322 | 0.6958 | `[0.42, 0.63]` |

Predicted plateau values, `sigma`-independent (self-similar):

| case | `p*` | `u*` | `rho*_L` (`Y=Y_L`) | `rho*_R` (`Y=Y_pre`) | `Y_L` | `Y_pre` |
|---|---|---|---|---|---|---|
| 24 | 2.040993e10 | 4022.796 | 2208.396 | 827.274 | 0.832154 | 1.1580449e-3 |
| 33 | 8.133623e9 | 3701.450 | 1459.415 | 432.967 | 0.934388 | 3.4661068e-3 |
| 34 | 3.949956e10 | 4534.419 | 2227.475 | 1227.067 | 0.626515 | 3.8631321e-4 |

3a. Run `ACID_TEND_SCALE=0.6` under `B`, `B+F3`, `B+RECON+F3`, `C+F3` for 24/33/34. **Null-run guard first** (`yadv_rh2.py:52-64` pattern): reject any run whose `ACID_DBG` last `t` is below `0.99 * 0.6 * t_end`, and any run printing `STALLED:`/`DIVERGED`. Report the rejects explicitly — a rejected run is data, not a gap.

3b. For each surviving run, extract from the dump: leading-front position (steepest `|Δp|` in `x>x_contact`), contact position (steepest `|Δrho|` with `p,u` continuous), the `*R` plateau means of `p,u,rho`, and the reconstructed `Y` in both plateaus (`yadv_yprofile.py:22-30`'s explicit `T`). Compare against the table above; report the relative gap per quantity.

3c. **Premise-P test**: independently of the exact solution, take the measured undisturbed pre-state and the measured `*R` plateau straight out of the dump, infer `Vs` from **mass** conservation, and report the momentum and energy residuals — §11.5's test, re-run on current code. Include the OFF path as the algebra's built-in control (it must close to ~1e-6, as §11.5's alpha rows did).

3d. **Predicted-front cross-check**: `x_leadshock(sigma) = 0.1 + S_R * sigma * t_end` should be linear in `sigma`. Run `sigma ∈ {0.4, 0.5, 0.6, 0.7}` and fit; a converged slope is the measured `S_R`. Compare to §3.2's `S_R`. (This is round 11's own front-position-vs-time method, whose earlier disagreement with the plateau method was left unreconciled — a second chance to reconcile it.)

### Stage 4 — thread (c): case33's `corr_p` sign flip. Zero extra runs.

From §3.4, the exact model answer has `corr_p < 0` for cases 24 and 33. **Prediction**: plain `B+F3`'s case33 pressure profile is closer to an *increasing* step (low on the left, high in the middle/right) than plain `B`'s, and its `p` level in the shock-processed region is closer to `p* = 8.1336e9` than to the reference `5.8772e9`. Test directly on the existing `B` and `B+F3` dumps (both complete finite for case33). Report the measured `p` plateau of each, the sign of `d(p)/dx` across the domain, and whether the observed `+0.351 → −0.817` motion is toward or away from the exact answer's `−0.2848`.

### Stage 5 — the redirect check (cheap, 2 runs). Does case15 have an analogous obstruction?

`cases.cpp:750-753`: case15's reference is `computed_reference(c, 800)` — **the same solver under the same environment** on a finer mesh. So under `ACID_YADV=1` the reference *is* the Y-path's own 800-cell solution, and no closure mismatch can exist by construction. Confirm empirically by dumping case15 under OFF and under B and checking that the `*_ref` columns differ between the two configs (they must, if the reference tracks the config). This one check establishes whether the roadmap's redirect target is structurally sound before recommending it.

---

## 6. Gates and targets, with execution order

### Hard gates

| gate | content | note |
|---|---|---|
| **G0** | `git diff --stat -- cpp/` shows **zero** changed lines; `git status` shows exactly one new file, `scripts/yadv_r26_closure.py` | new this round; the strongest possible no-op proof |
| **G1** | `scripts/yadv_r9_sweep.py --verify`: OFF byte-identical to the published `solver_denner` binary, 9/9 | must still be run |
| **G2** | `--sweep`: configs A-G match `EXPECTED` exactly, `ALL GATES OK` | must still be run |
| **G3** | `denner1d_unit` clean, `pT_from_v_e_massfrac` numbers unchanged | must still be run |
| **G4** | new-flag no-op | vacuous (no new env var); state so explicitly rather than silently skipping |
| **G5** | diff hygiene: one new script, no numeric literal in it that is not either transcribed from `cases.cpp`/`eos.cpp` (with the source line cited in a comment) or a gate threshold transcribed from `validation.cpp` | the script re-implements a gate; every constant must be traceable |

### Round-specific targets, in execution order

1. **P0** (Stage 1a) — exact solver reproduces §3.1/§3.2 to 5 sig figs, all internal consistency asserts pass. **Blocking.**
2. **P1** (Stage 1b) — Python gate reproduces `denner1d_validate`'s JSON to 6 sig figs and its `pass` verdict, on ≥ 6 `(case,config)` pairs. **Blocking.**
3. **T1** (Stage 2a) — §3.4's table, live. Primary reachability result.
4. **T2** (Stage 2b) — reachable-set scan; per-metric minimum over the family; `rho_max^B` vs §3.3's requirements.
5. **T3** (Stage 3a/3b) — measured `p*`, `u*`, `rho*_R`, `rho*_L`, `Y` in both plateaus, front/contact positions vs §3.2's predictions, per config. Primary fidelity result.
6. **T4** (Stage 3c) — premise-P / RH-residual re-test on current code, with the OFF control.
7. **T5** (Stage 3d) — `S_R` from the `sigma`-sweep front fit.
8. **T6** (Stage 4) — thread (c) resolution.
9. **T7** (Stage 5) — case15 redirect soundness.

**Discipline (rounds 23/24's lesson, restated)**: "the exact solution's front left the domain" is not a failure of anything — it is a *prediction*. Do not report a run as agreeing with §3.2 unless the null-run guard passed *and* the plateau window was verified non-empty and flat (report the within-window relative spread of `p`; reject the window if it exceeds 1%).

---

## 7. Pre-registered stop / decision rules

| Outcome | Trigger | Consequence |
|---|---|---|
| **S1 — structural obstruction CONFIRMED** | T1 shows the exact config-B solution failing `case24_spec_pass` on **≥ 2 of {`l2_p`,`l2_u`,`l2_rho`} by ≥ 1.5x AND on `dip` by ≥ 10x**, for all three cases; **and** T2 finds no member of the admissible single-shock family satisfying `case24_spec_pass` | Headline: **cases 24/33/34 are unreachable under `ACID_YADV=1` by any numerical improvement.** Record in `YADV_RESEARCH.md` §36 and the roadmap Control state. Redirect round 27 to case15 (subject to T7). **Escalate to the user**: the "19/19 under `ACID_YADV`" goal cannot be met without a change this loop is forbidden to make. |
| **S2 — S1 + solver VINDICATED** | S1 **and** T3 shows the measured `p*`/`u*`/`rho*_R` within **5%** of §3.2 **and** T4's momentum/energy residuals **< 5%** | Strongest outcome. The solver is *correct for the model it implements*; the gate is measuring a closure choice, not accuracy. Annotate (never edit) §11.6: its "solver defect" verdict was measured on round-2's non-conservative code and does not describe the current one. |
| **S3 — S1 + solver ALSO wrong** | S1 **and** T3 gap **> 20%** on any of `p*`/`rho*_R`/`S_R`, or T4 residual **> 10%** | Both an obstruction and a real numerical defect. 24/33/34 stay unreachable, but the measured gap is a concrete, quantified round-27 target *for its own sake* (it plausibly touches other cases). Report both halves; do not let either hide the other. |
| **S4 — H-STRUCT REFUTED** | T1 shows the exact solution passing, or within **1.2x** of every threshold | §3/§4's arithmetic is wrong. Retract this plan's premise in the round write-up, publish the corrected numbers, and hand round 27 the *numerical* target (the T3 gap) — with **F3b (round 25 §8) as the leading candidate**, now motivated by measurement. |
| **S5 — PARTIAL / case-split** | S1's criteria met for some cases and not others (case33's `l2_rho` bound is only 1.10x — this is the pre-identified candidate) | Report per-case. A case that is reachable in principle stays on the work list; one that is not, does not. No global claim. |
| **S6 — INSTRUMENT FAILURE** | P0 or P1 fails | **Publish nothing downstream.** Report the instrument failure as the round's result (round 5/9/11 precedent: a found-and-fixed measurement bug is measured progress). `consecutive_failures` not incremented if the bug is found and characterised; incremented if the round ends with an unexplained instrument. |
| **S7 — hard gate failure** | G0-G5 | Does not merge until fixed. |

**Anti-rescue clause.** If S1/S2/S3/S5 fires, **the honest record is the deliverable.** Explicitly forbidden as follow-ups, in this round or any future one, without a new explicit user decision:
- editing `cases.cpp` or `validation.cpp` to relax 24/33/34's thresholds, move `x_shock`, change `alpha_post := alpha_pre`, or replace the analytic reference with a solver-computed one;
- adding any per-cell, per-case, or per-regime predicate inside `acid.cpp` that switches the Y path back to alpha transport where the mixture is "homogeneous" — that is a per-case coefficient in disguise, and it is also just the OFF path;
- any damping, blending, or relaxation coefficient introduced to move the config-B answer toward closure (A). Closure (A) implies +268x to +1621x interphase mass transfer (§11.3); reproducing it *requires* unphysical phase change, which is a **model extension** (a fifth equation, or a physically-derived relaxation source), not a numerical fix, and is out of this loop's authorised scope.

**Corrections discovered this round** go into `YADV_RESEARCH.md` §36 as annotations. **Nothing in §1-§35 is edited.**

---

## 8. Non-goals (explicit)

- **F3b / same-triple restoration (round 25 §8)** — NOT built. Reason stated in §0. If S4 fires it becomes round 27's lead candidate; if S1/S2 fires it is moot for 24/33/34 and would need a different justification.
- **Any C++ change at all**, including a new diagnostic. Everything needed is already emitted by `denner1d_dump` (`apps/denner1d_dump.cpp:16-22`) and `ACID_TEND_SCALE`/`ACID_DBG`.
- **Promotion of any flag.** `ACID_YADV`, `ACID_YADV_F3`, `ACID_YADV_RECON`, `ACID_YADV_RESYNC` all stay default OFF; `ACID_YADV`'s recommended status stays OFF at 15/19.
- **case15** — Stage 5 checks only that it has *no analogous structural obstruction*. Its actual defect (round 7's `cj=30.02` vs 8.0 central-jump criterion) is not touched.
- **case29** (excluded from grading), **`max_steps` exhaustion** (thread g), **`rho_star` continuity / `theta_o` MWI memory** (thread e), **case13's Jacobian-approximation sensitivity** (thread d) — all untouched.
- **Reconciling round 11's plateau-method vs front-fit-method disagreement** as a goal in itself; T5 gives it a second data point as a by-product only.
- **Any claim that closure (A) is "wrong"**. The round's claim is narrower and defensible: (A) and (B) are both exact RH solutions of the same EOS, they differ by O(1), `cases.cpp` grades against (A), `acid.cpp` under `ACID_YADV=1` computes (B). Which one is the better *physics* for a non-reacting mixture is a statement for the user and the paper, not for this loop to decide unilaterally.

---

## 9. Literature

### Already in the repo — do not duplicate
`papers/library/md/`: Denner ACID 2018 + 2019 corrigendum, Denner 2018 linearisation, Denner 2020 conservative all-speeds, Xiao/Denner/vanWachem 2017, Bartholomew MWI 2018, Hanimann 2021, Collis et al. 2025 (`newest5/2025_Collis_..._four_equation_thermodynamic_ENO.md`, = arXiv:2504.14063), Coppola 2026, AlahyariBeig & Johnsen 2015, Fujiwara 2023, Terashima 2025, Kitamura 2016, Zhang & Jia 2022, Chamarthi wave-appropriate (`papers/wave_approp_main.md`). Existing stubs (rounds 21-25): Hou & LeFloch 1994, Karni 1994, Abgrall 1996, Shyue 1998, Saurel & Abgrall 1999, Flåtten/Morin/Munkejord 2011, Johnsen & Ham 2012, Le Métayer & Saurel 2016, Chiapolino 2017, Ma/Lv/Ihme 2017, Zhang/Kumbaro/Ghidaglia 2019, "assessment of a non-conservative four-equation system" 2021 (arXiv:2105.12874), Hawkins 2024, Bai et al. 2026 (RFQC), Clayton/McConnell/Solomon 2026 (PTE four-equation).

Two of these are directly on point and should be **read, not just cited**, if the calling session can obtain them: **Hou & LeFloch 1994** (a nonconservative discretisation converges to a shock of the wrong speed/strength — the exact signature of an A-vs-B mismatch, and the control for distinguishing it from a genuine closure difference) and the **2021 assessment of a non-conservative four-equation system** (arXiv:2105.12874, open access).

### New, not covered anywhere in `papers/` (verified by grep for `petitpas|gavrilyuk|00193-006-0065|2006.01630|bacigaluppi`)

The single most on-point precedent for this round's mechanism is that **the mixture shock relations are under-determined and require an extra closure — the volume-fraction (or equivalent thermodynamic) jump — and different closures give different post-shock states.** That is §3.1's A-vs-B table, in the literature, twenty years earlier.

1. **`papers/2007_Saurel_LeMetayer_Massoni_Gavrilyuk_shock_jump_relations_multiphase_stiff_relaxation_needed.md`**
   R. Saurel, O. Le Métayer, J. Massoni, S. Gavrilyuk, *Shock jump relations for multiphase mixtures with stiff mechanical relaxation*, Shock Waves **16** (2007) 209-232. **DOI 10.1007/s00193-006-0065-7**.
   Status: **paywalled (Springer); no OA copy found via `search_arxiv`, `search_semantic`, `search_google_scholar`, or web search.** Report as `_needed`.
   Why: states explicitly that mixture shock relations "necessitate the determination of the volume fraction jump or any other thermodynamic variable jump", and derives phase-Hugoniot closures compatible with the mixture energy equation. This is the literature statement of §2.1's closure (A) vs (B) non-uniqueness, and it is what makes this round's finding a *known modelling fact* rather than a novel claim about our solver.

2. **`papers/2009_Petitpas_Saurel_Franquet_Chinnayya_multiphase_CJ_conditions_needed.md`**
   F. Petitpas, R. Saurel, E. Franquet, A. Chinnayya, *Modelling detonation waves in condensed energetic materials: multiphase CJ conditions and multidimensional computations*, Shock Waves (2009). **DOI 10.1007/s00193-009-0217-7**.
   Status: **paywalled (Springer).** Report as `_needed`.
   Why: the follow-up that establishes appropriate jump relations for Kapila's reduced (mechanical-equilibrium) model in the weak-shock limit — the closest analogue to asking which closure our 4-eq PTE model *should* satisfy at a strong shock.

3. **`papers/2020_Abgrall_Bacigaluppi_Re_multicomponent_multiphase_compressible.md`** — **arXiv:2006.01630, OPEN ACCESS, fetchable now.**
   R. Abgrall, P. Bacigaluppi, B. Re, *On the simulation of multicomponent and multiphase compressible flows*.
   Why: directly contrasts a full non-equilibrium relaxation model against **a four-equation model assuming stiff mechanical *and thermal* equilibrium** — i.e. precisely this solver's PTE closure — and discusses when the two disagree. The natural reference for "is the PTE 4-eq closure the right one at a strong shock in a homogeneous mixture", which is the model-level question S1 hands back to the user.

If Stage 2 fires S1, one further sanity item is worth a search in round 27 (not this round): whether Denner's own §7.4.1 Fig.18 results were produced with volume-fraction transport (they must have been, given Eq.57-62), which would confirm that the 24/33/34 case family is *by construction* a volume-fraction-model validation and was never intended as a mass-fraction-model test.

---

## 10. Reproduce commands (for the implementing session)

```bash
cd <worktree>/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8

python3 scripts/yadv_r26_closure.py --stage0      # P0: must reproduce sect.3.1 + sect.3.2
python3 scripts/yadv_r26_closure.py --gatecheck   # P1: must match denner1d_validate JSON, 6 sig figs
python3 scripts/yadv_r26_closure.py --gate        # T1 + T2
python3 scripts/yadv_r26_closure.py --autopsy     # T6 (thread c) on existing B / B+F3 dumps
python3 scripts/yadv_r26_closure.py --window 0.6  # T3 + T4; also 0.4 0.5 0.7 for T5

# raw spot checks (DENNER_ACID=1 is MANDATORY -- .claude/rules/denner-pitfalls.md)
D=./build-cpp/cpp/denner_1d/denner1d_dump
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_F3=1 ACID_TEND_SCALE=0.6 $D 24 2>/dev/null | awk -F, 'NR>1 && $1>0.45 && $1<0.70'
    # expect p ~ 2.041e10, u ~ 4023, rho ~ 827  (sect.3.2 *R plateau); front near x=0.764
DENNER_ACID=1 ACID_YADV=1 ACID_YADV_F3=1 ACID_TEND_SCALE=0.6 ACID_DBG=1 $D 24 2>&1 >/dev/null | tail -3
    # null-run guard: last t must be >= 0.99 * 0.6 * 1.089196e-04
```

---

### Critical files for implementation
- `solver_4eq_mass/scripts/yadv_r26_closure.py` (new — the only file created)
- `solver_4eq_mass/cpp/denner_1d/src/validation.cpp` (read-only: `:18-26`, `:49`, `:317-359`, `:445-460`, `:469-505`, `:731-734`, `:752-774` — the gate being re-implemented)
- `solver_4eq_mass/cpp/denner_1d/src/cases.cpp` (read-only: `:105-151`, `:195-196`, `:689-694`, `:750-753`, `:779-791` — closure (A), the grid, the IC, the reference)
- `solver_4eq_mass/cpp/denner_1d/src/acid.cpp` (read-only: `:746-747`, `:784-804`, `:1257-1273`, `:1285-1316`, `:1323-1331` — the `rho*Y` update, the recovery site, F3, `ACID_TEND_SCALE`)
- `solver_4eq_mass/scripts/yadv_r9_sweep.py` (`base_env()`/`ACID_ENV_VARS` pattern to copy; `CONFIGS`/`EXPECTED` unchanged this round)
- `solver_4eq_mass/scripts/yadv_rh2.py` (`last_step_fraction()` null-run guard) and `solver_4eq_mass/scripts/yadv_yprofile.py` (`T`/`Y` reconstruction — note its hardcoded path must **not** be copied)
