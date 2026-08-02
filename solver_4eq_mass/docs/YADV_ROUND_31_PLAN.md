# ROUND 31 PLAN — Phase 3a reopened: what physics is *actually* missing from cases 24/33/34, what it costs, and why round 31 must not write model code

**Status: DESIGN + DIAGNOSTIC ONLY. Recommendation: no model-affecting C++ this round, and an escalation to the user on scope.**
Planner: opus (Plan subtype). Worktree `.claude/worktrees/yadv-round-31/solver_4eq_mass`, HEAD `b82f665`.
Charter: `docs/YADV_ROADMAP.md` §"Phase 3a (cases 24/33/34) REOPENED as a model-extension research thread (2026-08-02, explicit user authorization)".
Prior load-bearing result this plan must not contradict: `docs/YADV_RESEARCH.md` §36 (round 26's closure proof), §11 (the Y-consistent Hugoniot).

---

## 0. Executive summary

Round 26 proved cases 24/33/34 unreachable under `ACID_YADV=1` and named the remedy as "a genuine model extension (interphase mass transfer / relaxation source)". The user authorized pursuing exactly that. **This plan's headline finding is that round 26's *diagnosis* is correct and its *named remedy is the wrong one*, and that this can be established in closed form, to machine precision, before a line of model code is written.**

Four results, each verified live during planning against the repo's own round-26 instrument (`scripts/yadv_r26_closure.py`), each independently re-derivable by the implementing session:

**R1 — The reference's caloric EOS is temperature-blind.** For cases 24/33/34 both phases have NASG covolume `b = 0` and reference enthalpy `eta = 0` (`eos.cpp:11-13` air; `cases.cpp:446` `denner_water`). At frozen volume fraction the mixture's volumetric internal energy is
`e_vol = α(p + γ_a·Π_a)/(γ_a−1) + (1−α)(p + γ_b·Π_b)/(γ_b−1)` — **a function of `(p, α)` only, with `T` cancelling identically.** Verified: `evol_singleT(p,T,α) == evol_ESG(p,α)` to `≤2.8e-16` relative at every pre- and post-shock state of all three cases.

**R2 — The reference is therefore simultaneously the single-temperature α-held Hugoniot *and* the two-temperature (pressure-equilibrium-only, Kapila/Allaire-family) Hugoniot in which α **and** Y are both conserved.** Reading closure (A)'s own post-shock state phase-wise gives `T_air / T_water = 4276 / 4095 / 4341` for cases 24/33/34 (e.g. case24: `T_air = 1.217e7 K`, `T_water = 2846 K`), while both phases compress by *exactly* the mixture ratio (3.7176 / 4.7264 / 2.6873, identical to 10 digits). **The missing physics is thermal disequilibrium, not mass transfer.**

**R3 — The required composition target is unique, known in closed form, and requires converting water into air.** At frozen α with `b=0`, mass fraction is a function of pressure alone:
`Y/(1−Y) = [α/(1−α)]·(R_b/R_a)·(p+Π_a)/(p+Π_b)`, `R_k = kv_k(γ_k−1)`.
This reproduces `Y_post^A` to `2.2e-16` for all three cases. The required jump is `Y_air: 1.158e-3 → 0.832` (case24), i.e. **63%–93% of the entire through-shock mass flux must convert from liquid water into air** (`2.67e6 / 1.27e6 / 3.85e6 kg·m⁻²·s⁻¹`). The gate's `dip ≤ 0.02` criterion pins the target to **±0.7%–1.6% in `Y`**.

**R4 — Every model extension that reproduces the reference collapses to α-transport (the OFF path) *unless* it abandons the single-temperature closure.** This is an algebraic consequence of R1, not a numerical difficulty. Concretely: a mass-transfer source with the *only* target that works (`Y_A(p)` from R3) is exactly "α held", i.e. the OFF path in Y-coordinates; and a *finite*-rate version of it needs a relaxation time `τ` that is a pure tuning constant with no physical referent — forbidden by this project's own absolute rules.

**Consequence and recommendation.** The genuine model extension that (i) reproduces the reference, (ii) *retains* conservative phase-mass transport (the whole point of `ACID_YADV`), and (iii) needs **zero** `cases.cpp`/`validation.cpp` edits, is the **single-pressure / two-temperature Allaire 5-equation model** — conserve `α_a ρ_a` and `α_b ρ_b` separately, advect `α`, drop `T`-equilibrium. That is a solver-class change (a new state variable inside the coupled `(u,p,h)` Newton, a duplicated EOS closure, a rewritten analytic Jacobian), realistically **4–8 rounds at high blast radius to all 19 cases** — not "adding a source term". Round 31 should therefore be **100% design + a Python-only Stage 0**, with the model-affecting work deferred to round 32+ *and* re-authorized by the user once the true scope is on the table. §11 states the escalation.

---

## 1. Verified code facts (read live at HEAD `b82f665`)

### 1.1 What the three cases are

| fact | citation |
|---|---|
| Phases: `air` (γ=1.4, Π=0, b=0, kv=720.25, η=0) and `denner_water` (γ=4.1, Π=4.4e8, b=0, kv=474.2, η=0) | `eos.cpp:11-13`; `cases.cpp:446` |
| Case table: `{"24"…air, denner_water, c24}`, `{"33"…}`, `{"34"…}` | `cases.cpp:583`, `:603`, `:605` |
| `α_air,pre` = 0.50 / 0.75 / 0.25 for 24 / 33 / 34 | `cases.cpp:495`, `:543`, `:545`, `:630-632` |
| Reference construction, Denner Eqs.57-62 | `cases.cpp:105-151` |
| **The closure statement**: `s.alpha_post = s.alpha_pre;  // psi held (homogeneous mixture)` | `cases.cpp:148` |
| `T_post` is *recovered by bisection* to make the single-`T` mixture density match Eq.61's `ρ_post` | `cases.cpp:149`, `:38-51` |
| IC seeds `x<0.1` with the closure-(A) post-shock state; left BC transmissive → sustained piston | `cases.cpp:689-694` (`s.alpha[i] = post ? sh24.alpha_post : sh24.alpha_pre;` at `:691`) |
| Reference profile: step at `x_shock = 0.8`, then `refresh_thermo` | `cases.cpp:779-791` |
| `refresh_thermo` recomputes `ρ = mixture_density(p,T,α)`, `h = mixture_enthalpy(p,T,α)` | `solver.cpp:1008-1016` |
| Unused helper `equil_from_p_rho_Y` (a Y-conserving reference that was evidently tried and replaced) | `cases.cpp:73-100` |

### 1.2 What the gate is

`validation.cpp:469-505` (`case24_spec_pass`), dispatched at `validation.cpp:731-734`:
- `profile_ok` (`:493-495`): `l2_{p,u,ρ} ≤ 0.20`, `corr_{p,u,ρ} ≥ 0.92`;
- `plateau_ok` (`:492`): `dip ≤ 0.02`, `hump ≤ 0.01`, `plateau_l2 ≤ 0.015`, all normalised by `jump = |ρ_post − ρ_pre|` (`:476`), measured on `0.005 < x < x_shock − max(10dx, 0.03)` (`:478-479`).
- **`T` and `α` are not graded.** Only `p`, `u`, `ρ` enter. This is load-bearing for R1/R2 below.

### 1.3 What `ACID_YADV=1` does

| fact | citation |
|---|---|
| Flag read (`getenv`, default OFF) | `acid.cpp:578` (declaration + rationale `:569-577`) |
| `Yv[]` initialised once from the case's α IC at the initial `(p,T)`; lives *outside* `PrimitiveState` as a solver-local `Vec` | `acid.cpp:912-923` (`Vec Yv(n, 0.0);` at `:915`) |
| Conservative `ρY` transport block | `acid.cpp:1268-1376` |
| The conserved update itself: `rY = ρ_old·Y − dt/dx·(mdR·af_R − mdL·af_L)`, `ρ* = ρ_old − dt/dx·(mdR − mdL)`, `Y_new = rY/ρ*` | `acid.cpp:1357-1366` |
| `Yv = anew;` then α recovered algebraically at the stale `(p_o,T_o)` | `acid.cpp:1367-1375` |
| Round-25 `ACID_YADV_F3` alternative recovery site (α at the new Y's own PTE state) | `acid.cpp:1386-1414` |
| Baseline: OFF 19/19; `ACID_YADV=1` 15/19, fail set `{15,24,33,34}` | `docs/YADV_ROADMAP.md` §Current goal / Control state |

### 1.4 EOS machinery already present (no new code needed to state the derivations)

| fact | citation |
|---|---|
| `α ↔ Y` explicit inverse pair | `eos.hpp:57-66` |
| `∂α/∂p\|_{T,Y}` and `∂α/∂T\|_{p,Y}` at fixed mass fraction, already derived and documented | `eos.hpp:182-198` (`alpha_derivs_massfrac`, `dalpha_dp_massfrac`, `dalpha_dT_massfrac`) |
| **Already documented in-repo**: `φ_k/ρ_k = −1/T` *exactly* whenever `b_k = 0`, so `a_T` is algebraically zero for 17 of 19 cases | `eos.hpp:170-176` |
| Closed-form NASG PTE inversion `pT_from_v_e_massfrac(v,e,Y,a,b)` | `eos.hpp:115-160` |
| α↔Y round-trip conditioning floor (used by `ACID_RECON_NULL`) | `eos.hpp:78-85` |
| **`Phase` has exactly five fields: `gamma, pinf, b, kv, eta`. There is no entropy-reference constant `q'`.** | `types.hpp:8-14` |
| `water_vapor_phase()` exists (γ=1.467, Π=0, b=0, kv=955, η=2.077616e6) but is **used by no case** — only by the unit tests | `eos.cpp:21-23`; `denner1d_unit.cpp:107,241,326`; grep of `cases.cpp` shows no use |
| Case 15 uses `air, water` (NASG `water_liquid_phase`), case 14 likewise; **no liquid/vapour pair anywhere in the suite** | `cases.cpp:581-582` |

---

## 2. Two corrections to the round-31 charter's own premises

**Report both honestly; neither is a criticism of prior rounds, both are facts the implementing session will hit immediately.**

**C1 — `papers/md/33_saurel_relaxation_multiphase.md` is NOT in the tree at `b82f665`.** The roadmap's round-31 charter (`YADV_ROADMAP.md:82`) and rounds 28/29/30 (`YADV_ROUND_28_PLAN.md:407`, `YADV_ROUND_29_PLAN.md:217,462`, `YADV_ROUND_30_PLAN.md:376`) all cite it as present with line numbers. `papers` is a symlink to `../solver_denner/papers`; `find -L papers -name '*saurel*'` returns only `_needed.md` stubs, and `papers/md/` contains four unrelated `.txt` files. The file was evidently removed by a round's own worktree/dummy-file cleanup. **Action: re-fetch it (see §4, stub S0) before relying on any §-level citation of it.** Nothing in this plan depends on it.

**C2 — "interphase mass transfer / relaxation source" is the wrong named remedy for these three cases.** Round 26 §36.7 wrote: *"closure (A) implies +268x to +1621x interphase mass transfer across the shock, §11.3 — reproducing it requires actual phase change, a model extension such as a fifth equation or a physically-derived relaxation source"*. §3 below shows the "+268x…+1621x mass transfer" reading is an *artifact of forcing a two-temperature state through a single-temperature EOS*; a two-temperature reading of the *identical* post-shock state has **zero** mass transfer (both phases compress by exactly the mixture ratio). Round 26's parenthetical "a fifth equation" was the right half of its own disjunction. This is a refinement, not a contradiction: every measured number in §36 stands.

---

## 3. Derivation

Everything in this section was computed during planning by importing `scripts/yadv_r26_closure.py` (its `closure_a_shock`, `mix_coeffs`, `S_of_p`, `phase_rho`, `phase_h`) — i.e. against the already-P0/P1-cross-validated instrument, not a fresh transcription.

### 3.1 R1 — at frozen α the mixture caloric EOS is temperature-blind (exact)

Per-phase NASG with `b=0`: `ρ_k = (p+Π_k)/(kv_k(γ_k−1)T)`, `h_k = γ_k kv_k T + η_k`, so
`ρ_k h_k = γ_k(p+Π_k)/(γ_k−1) + ρ_k η_k`, and `e_vol,k = ρ_k h_k − p`.

Single-temperature mixture (`eos.cpp:71-74`, `mixture_internal_energy_density`):

```
e_vol(p,T,α) = Σ_k α_k [ γ_k (p+Π_k)/(γ_k−1) + ρ_k(p,T) η_k ] − p
             = Σ_k α_k (p + γ_k Π_k)/(γ_k−1)  +  Σ_k α_k ρ_k η_k        [using Σ_k α_k = 1]
```
(the identity `Σ_k α_k γ_k(p+Π_k)/(γ_k−1) − p = Σ_k α_k(p+γ_kΠ_k)/(γ_k−1)` follows from `γ_k/(γ_k−1) − 1/(γ_k−1) = 1`).

With `η_a = η_b = 0` (air and `denner_water`, `eos.cpp:12`, `cases.cpp:446`) the second sum vanishes and

> **(E1)  `e_vol(p, α) = α (p + γ_a Π_a)/(γ_a − 1) + (1−α)(p + γ_b Π_b)/(γ_b − 1)` — independent of `T`.**

i.e. at frozen α the mixture is *exactly* an equivalent stiffened gas with `1/(γ_mix−1) = Σ α_k/(γ_k−1)` (which is precisely `cases.cpp:128-129`'s `inv_gm1`/`gamma_mix`).

**Verified live** (planning session, `scripts/yadv_r26_closure.py` states):

| case | state | `e_vol` single-T (`eos.cpp:71-74`) | `e_vol` ESG (E1) | rel diff |
|---|---|---|---|---|
| 24 | pre | 2.9110887097e+08 | 2.9110887097e+08 | 0.0 |
| 24 | post | 2.1578845958e+10 | 2.1578845958e+10 | 1.77e-16 |
| 33 | pre | 1.4567943548e+08 | 1.4567943548e+08 | 0.0 |
| 33 | post | 1.1639274829e+10 | 1.1639274829e+10 | 1.64e-16 |
| 34 | pre | 4.3653830645e+08 | 4.3653830645e+08 | 1.37e-16 |
| 34 | post | 2.7852014273e+10 | 2.7852014273e+10 | 2.74e-16 |

And the RH closure of closure (A) under (E1) alone (no `T` anywhere):

| case | `Vs` from Rayleigh+Hugoniot with (E1) | `Vs` from `cases.cpp:135` | rel | Hugoniot residual (rel) |
|---|---|---|---|---|
| 24 | 6426.761180 | 6426.761180 | 0.00e+00 | 0.00e+00 |
| 33 | 5456.493595 | 5456.493595 | 1.67e-16 | 1.26e-16 |
| 34 | 8201.394412 | 8201.394412 | 2.22e-16 | 1.26e-16 |

**This is why Denner's Eqs.57-62 reference is exact and why the OFF (α-transport) path reproduces it: the graded fields `(p,u,ρ)` never touch `T`.**

### 3.2 R2 — the reference *is* the two-temperature Hugoniot

Take closure (A)'s own `(p_post, ρ_post, α_pre)` and read it phase-wise under **both** `α` conserved and `Y` conserved (the Allaire/Kapila-family jump conditions: per-phase mass conservation `∂(α_kρ_k)/∂t + ∂(α_kρ_k u)/∂x = 0` plus a continuous `α` across the shock):

`ρ_a = Y_pre ρ / α_pre`, `ρ_b = (1−Y_pre) ρ /(1−α_pre)`, `T_k = (p+Π_k)/(kv_k(γ_k−1)ρ_k)`.

| case | `ρ_a,post` | `ρ_b,post` | `T_air` [K] | `T_water` [K] | `T_air/T_water` | compression: mix / air / water |
|---|---|---|---|---|---|---|
| 24 | 4.3016 | 3710.213 | 1.2172e+07 | 2846.30 | 4276.3 | 3.7176 / 3.7176 / 3.7176 |
| 33 | 5.4688 | 4716.978 | 3.7302e+06 | 911.05 | 4094.5 | 4.7264 / 4.7264 / 4.7264 |
| 34 | 3.1094 | 2681.903 | 3.5302e+07 | 8132.89 | 4340.6 | 2.6873 / 2.6873 / 2.6873 |

Because `e_vol` is `T`-blind (E1), this two-temperature state has **exactly the same** `p`, `u`, `ρ` — i.e. exactly the same graded content — as the single-`T` state `cases.cpp:149` reconstructs. The three-way equality is:

> **(E2)  reference (`cases.cpp:105-151`) ≡ single-T α-held Hugoniot ≡ two-T (p-equilibrium, α- and Y-conserving) Hugoniot**, in `(p,u,ρ)`, exactly.

**Interpretation.** Closure (A) and closure (B) are the `τ_T → ∞` and `τ_T → 0` ends of a **thermal** relaxation continuum, not the ends of a mass-transfer continuum. For a Mach-10 shock crossing a `dx = 1.25e-3 m` cell in `~2e-7 s`, gas–liquid thermal equilibration is nowhere near instantaneous; **closure (A) is the physically correct limit and the single-temperature 4-equation assumption is the modelling error.** The `+268x…+1621x` "interphase mass transfer" of §11.3 is what that modelling error looks like when you insist on reading a two-temperature state through a single-`T` EOS.

**Literature anchor (canonical, exact):** Kapila, Menikoff, Bdzil, Son & Stewart (2001), *Phys. Fluids* **13**(10) 3002-3024, DOI `10.1063/1.1398042`, abstract verbatim: *"The reduced models are hyperbolic and are mechanically as well as thermodynamically consistent with the parent model. **However, they cannot be expressed in conservation form and hence require a regularization in order to fully specify the jump conditions across shock waves.** … **Dissipation associated with degrees of freedom that have been eliminated is restricted to the thin layers and is accounted for by the jump conditions.**"* Round 26's "two exact, both-admissible closures" is this statement, 25 years earlier; and Kapila's own answer to *which* regularization is "the inner-layer analysis of the parent model" — i.e. the disequilibrium that was eliminated. Not a source term.

### 3.3 R3 — the required composition target, in closed form, and its tolerance

**Closed form.** At frozen α with `b_a = b_b = 0`, `ρ_a/ρ_b = (R_b/R_a)·(p+Π_a)/(p+Π_b)` with `R_k = kv_k(γ_k−1)` — `T` cancels. Since `Y/(1−Y) = [α/(1−α)]·(ρ_a/ρ_b)`:

> **(E3)  `Y/(1−Y) = [α/(1−α)] · (R_b/R_a) · (p+Π_a)/(p+Π_b)`.**   Here `R_a = 288.1000` (air), `R_b = 1470.0200` (water), `R_b/R_a = 5.102464`.

Equivalently, (E3) is the exact first integral of the Y-space form of the Kapila α-equation (§3.6). **Verified live** against `closure_a_shock`:

| case | `Y_pre` closed form | vs solver | `Y_post` closed form | vs solver |
|---|---|---|---|---|
| 24 | 1.158044885083e-03 | rel 2.22e-16 | 0.832154048623 | rel 2.22e-16 |
| 33 | 3.466106840653e-03 | rel 0.00 | 0.934388465804 | rel 2.22e-16 |
| 34 | 3.863132070500e-04 | rel 2.22e-16 | 0.626514695497 | rel 2.22e-16 |

**Uniqueness.** Solving the RH system at the reference shock speed with a *free* post-shock composition `Y*` (mass, momentum, energy; mixture NASG at single `T`; bisection on `v_post`) and requiring `ρ_post = ρ_post^A` returns `Y* = Y_post^A` to 6 digits in all three cases, and at that `Y*` the recovered `p` and `u` match the reference to `1.000000` — **so the composition target is not merely sufficient, it is the unique one.**

**Tolerance.** The binding criterion is `dip ≤ 0.02` (`validation.cpp:492`), normalised by `jump = ρ_post − ρ_pre`:

| case | `jump` | allowed plateau-ρ deficit | ⟹ allowed band on `Y*` |
|---|---|---|---|
| 24 | 1357.68 | 1.462% | `Y* ≥ 0.99105·Y_A` (**0.90%**) |
| 33 | 932.98 | 1.577% | `Y* ≥ 0.99334·Y_A` (**0.67%**) |
| 34 | 1263.42 | 1.256% | `Y* ≥ 0.98387·Y_A` (**1.61%**) |

Sensitivity table (relative plateau error vs `Y*/Y_A`), case 24, for the implementing session to reproduce:

| `Y*/Y_A` | 0.50 | 0.80 | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| `p` err | −0.2611 | −0.1146 | −0.0592 | −0.0301 | 0 | +0.0311 | +0.0633 | +0.1310 |
| `ρ` err | −0.4150 | −0.2374 | −0.1385 | −0.0756 | 0 | +0.0923 | +0.2076 | +0.5529 |
| `u` err | −0.2611 | −0.1146 | −0.0592 | −0.0301 | 0 | +0.0311 | +0.0633 | +0.1310 |

**Any proposed source term must hit `Y_A` to ~1%.** That kills, on its own, every heuristic/correlation-based mass-transfer model (HRM, D²-law, tuned Hertz-Knudsen) even before the direction argument.

### 3.4 R4a — the required *rate* is not the obstacle; the *direction and species* are

`ṁ` = through-shock mass flux `= ρ_pre·Vs`. Required interphase conversion flux `j = ṁ·(Y_A − Y_pre)`. For a 3-cell numerical shock at the cases' own `N = 800` (`cases.cpp:496`, `dx = 1.25e-3`), shock-frame mean traversal speed `½(Vs + (Vs−u_post))`:

| case | `ṁ` [kg/m²s] | `j` [kg/m²s] | fraction of `ṁ` | `t_res` (3 cells) | required `τ` (≈`t_res/3`) | required `Γ` [kg/m³s] |
|---|---|---|---|---|---|---|
| 24 | 3.2107e+06 | 2.6681e+06 | **0.8310** | 9.196e-07 s | 3.07e-07 s | 1.678e+09 |
| 33 | 1.3661e+06 | 1.2718e+06 | **0.9309** | 1.134e-06 s | 3.78e-07 s | 9.710e+08 |
| 34 | 6.1411e+06 | 3.8451e+06 | **0.6261** | 6.665e-07 s | 2.22e-07 s | 1.890e+09 |

Cross-check against the kinetic-theory ceiling (Hertz-Knudsen with accommodation `λ=1`, `j_max = p/√(2πRT)`): the required `Γ` corresponds to an interfacial area density `Σ ≈ 616 / 827 / 375 m⁻¹` — a ~2–3 mm droplet dispersion. **Report honestly: the required rate is *not* physically outrageous.** The obstacle is elsewhere:

### 3.5 R4b — why no admissible mass-transfer closure produces it (three independent reasons)

**(a) Species.** Phase A is `air_phase()` and phase B is `denner_water` — chemically distinct substances (`cases.cpp:583,603,605`). The required transfer is **liquid water → air**, 63–93% of all mass crossing the shock. No entropy-admissible mass-transfer source can transmute one species into another. The physically correct mass-transfer rate for this phase pair is identically **zero**.

**(b) The equilibrium target is not definable in this EOS.** Every thermodynamically-bounded mass-transfer closure in the literature drives `Y` toward the Gibbs/chemical-potential equilibrium `g_a(p,T) = g_b(p,T)` (Collis et al. §4, in repo; Pelanti 2022; Saurel-Petitpas-Berry 2009; Chiapolino-Boivin-Saurel 2017). NASG Gibbs free energy is `g_k = h_k − T s_k` with `s_k = cv_k ln(T^{γ_k}/(p+Π_k)^{γ_k−1}) + q'_k`. **`Phase` (`types.hpp:8-14`) has no `q'` field.** Adding it means sourcing entropy-reference constants for air and `denner_water` from Le Métayer/Massoni/Saurel (2004) — which do not exist for an air/water pair, because there is no saturation curve between air and liquid water.

**(c) The direction is thermodynamically backwards even if (a) and (b) were solved.** At `(p_post, T_post)` the *air* is the dense phase (`ρ_a = 3090` vs `ρ_b = 623` kg/m³ at case24's post state) because ideal-gas air at 1.5e10 Pa out-densifies stiffened water. Holding `α` therefore *requires* creating air mass. This is a signature of the frozen-α closure, not of any relaxation process.

**(d) Corroboration from the closest in-repo prior art.** `papers/library/md/2026_recent/[2026] 4eq + phase change.md` (Collis, Mirjalili, Khanwale, Mani, Iaccarino — *"An LES model with finite-rate phase change and subgrid spray based on a thermodynamically consistent four-equation multiphase model"*) is a 4-equation single-`p`/single-`T` model **with** an interphase mass-transfer source — exactly the family the charter asked for. Its §4/§4.1.1 verification cases are **air–water shock tubes**, and in every one the transfer is between *liquid water and water vapour*, with `Y_g^air` an **inert third component that never participates** (§4.1.2: *"the mass fraction `Y_g^air = 0.98` everywhere in the domain, and the mass fractions of liquid water and water vapor are deduced from satisfying thermochemical equilibrium"*). The saturation state is a fitted Antoine equation for **water**. **The canonical 4-eq-with-mass-transfer model, applied to an air–water mixture, holds `Y_air` fixed — precisely what `ACID_YADV=1` already does.**

### 3.6 R4c — the one source term that *would* work, and why it is the OFF path

Impose `Dα/Dt = 0` in `Y`-space. With `α = α(Y,p,T)` and the partials already in `eos.hpp:182-198`, plus `∂α/∂Y|_{p,T} = α(1−α)/(Y(1−Y))`:

```
DY/Dt = − [ a_p Dp/Dt + a_T DT/Dt ] · Y(1−Y)/(α(1−α))
      =   Y(1−Y) [ (κ_a − κ_b) Dp/Dt + (β_a − β_b) DT/Dt ],   κ_k = ζ_k/ρ_k, β_k = φ_k/ρ_k
```
For `b_k = 0` (both phases here, and 17 of 19 cases — `eos.hpp:170-176`): `κ_k = 1/(p+Π_k)` and `β_k = −1/T` **exactly**, so `β_a − β_b ≡ 0` and

> **(E4)  `DY/Dt = Y(1−Y) · [ 1/(p+Π_a) − 1/(p+Π_b) ] · Dp/Dt`** — parameter-free, expressible entirely from `PhaseProps` fields that `acid.cpp` already computes, no new constant. Its first integral is exactly (E3).

This is attractive-looking and **must be pre-registered as dead before anyone builds it**, for three reasons:

1. **It is α-transport.** (E4) is the statement "α is materially advected", rewritten. Applying it globally makes `ACID_YADV=1` produce the OFF path's model content, destroying the *entire premise* of `ACID_YADV` (that `Y` is a true material invariant and `ρY` is conserved — `acid.cpp:569-577`). It also lands squarely inside round 26's anti-rescue clause (`YADV_RESEARCH.md` §36.7, *"functionally just the OFF path"*), even though it is not predicate-gated.
2. **It re-introduces the non-conservative product.** `Dp/Dt` at a shock is a distribution; the discrete jump then depends on the viscous path (Hou & LeFloch 1994, already a stub in `papers/`). The OFF path already carries this in α-space (`acid.cpp:1257-1263`) and is validated 19/19; a Y-space re-encoding buys nothing and risks a different (wrong) jump.
3. **The finite-rate version needs a tuning constant.** `∂(ρY)/∂t + ∂(ρYu)/∂x = ρ(Y_A(p) − Y)/τ` interpolates between the OFF path (`τ→0`) and current `ACID_YADV` (`τ→∞`). `τ` here is *not* a physical relaxation time — there is no physical process being relaxed (§3.5a) — it is a pure numerical blend parameter. **Forbidden by `YADV_ROADMAP.md`:484 ("No tuning constants… Global physical constants only").**

### 3.7 R4d — the honest positive result: the extension that *does* work

Retaining conservative phase-mass transport **and** reproducing the reference requires dropping the single-temperature closure. The minimal such model is **Allaire/Kapila 5-equation** (`Allaire, Clerc & Kokh 2002`, JCP 181(2):577-616, DOI `10.1006/jcph.2002.7143`; `Murrone & Guillard 2005`, JCP 202(2):664-698, DOI `10.1016/j.jcp.2004.07.019`):

```
∂(α_a ρ_a)/∂t + ∂(α_a ρ_a u)/∂x = 0          <- conservative phase mass (ACID_YADV's premise SURVIVES)
∂(α_b ρ_b)/∂t + ∂(α_b ρ_b u)/∂x = 0
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0
∂α/∂t + u ∂α/∂x = K ∂u/∂x        (K = 0 for Allaire-transport; Kapila's K for the reduced model)
```
Single pressure, **two temperatures**, `ρ_a` and `ρ_b` independent. By §3.2 this reproduces the 24/33/34 reference exactly, with **zero `cases.cpp` / `validation.cpp` edits** — the reference is *already* this model's own answer. That is a strong signal it is the right extension (contrast: any mass-transfer route would have needed the reference changed, which is forbidden).

The hierarchy this sits in is fully mapped in the literature and its subcharacteristic structure is proved: Flåtten & Lund 2011 (DOI `10.1142/S0218202511005775`), Lund 2012 (DOI `10.1137/12086368X`), Linga 2018 (DOI `10.1051/proc/201966006`, **open access**) — the last derives the whole Baer-Nunziato→`p`-model→`pT`-model chain and shows `a_eq ≤ a_frozen` throughout. This solver's 4-equation model **is** Linga's `pT`-model; the 24/33/34 reference **is** the `p`-model's shock.

---

## 4. Literature

### Already in repo — do NOT re-download

- `papers/library/md/2026_recent/[2026] 4eq + phase change.md` — **Collis, Mirjalili, Khanwale, Mani, Iaccarino.** *The* directly-on-target paper: a 4-equation single-`p`/single-`T` model with finite-rate interphase mass transfer. §4 (mass-transfer term `ν(Y_eq − Y)`), §4.1 (HEM, `τ→0`), §4.1.1 (5-step approximate UV-flash HEM solver — structurally the same object as `eos.hpp:115-160`'s `pT_from_v_e_massfrac`), §4.1.2 (air–water shock-tube verification; **`Y_air` inert**), §4.2 (Hertz-Knudsen finite rate), §4.2.1 (HEM-bounded `τ`), Appendix B (HRM, D²-law). **Load-bearing for §3.5(d).**
- `papers/library/md/newest5/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO.md` — the companion 4-eq numerics paper already cited in `eos.hpp:106-108`.
- `papers/1994_Hou_LeFloch_nonconservative_schemes_wrong_solutions_needed.md` — §3.6(2).
- Existing stubs, all still accurate, all relevant, **none to be re-searched**: `2007_Saurel_LeMetayer_Massoni_Gavrilyuk_shock_jump_relations_multiphase_stiff_relaxation_needed.md` (paywalled, no OA — round 27 already established this), `2009_Saurel_Petitpas_Berry_relaxation_cavitating_multiphase_needed.md`, `2017_Chiapolino_Boivin_Saurel_fast_relaxation_needed.md`, `2019_Pelanti_Shyue_multiphase_liquid_vapor_gas_cavitation_needed.md`, `2007_Petitpas_..._relaxation_projection_II_needed.md`, `2009_Petitpas_..._multiphase_CJ_needed.md`, `2021_assessment_nonconservative_four_equation_needed.md`, `2026_Clayton_McConnell_Solomon_PTE_four_equation_needed.md`, `2011_Flatten_Morin_Munkejord_stiffened_gas_equilibrium_needed.md`, `2016_LeMetayer_Saurel_NASG_EOS_needed.md`.

### Recovery action (C1)

**S0 — re-fetch, do not stub.** `papers/md/33_saurel_relaxation_multiphase.md` (Saurel, Petitpas & Berry 2009, JCP 228(5):1678-1712, DOI `10.1016/j.jcp.2008.11.002`) is cited by rounds 28/29/30 with line numbers but is absent at `b82f665`. The `_needed.md` stub for it already exists and carries the DOI.

### New stubs to write (content given verbatim; the implementing session saves these — this Planner cannot write files)

**`papers/2001_Kapila_Menikoff_Bdzil_Son_Stewart_reduced_two_phase_needed.md`**
```
# Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations

DOI: 10.1063/1.1398042
저자/연도/저널: A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart,
                Physics of Fluids 13(10), 3002-3024 (2001)
PDF (Crossref-listed, may be paywalled):
  https://pubs.aip.org/aip/pof/article-pdf/13/10/3002/19324849/3002_1_online.pdf

이 작업에 필요한 이유: round 26 §36의 "closure (A)/(B)는 둘 다 정확한 RH 해"라는 발견의
정본(canonical) 문헌. Abstract 자체가 그 진술이다 -- 단일속도/단일압력으로 축약된 모델은
"cannot be expressed in conservation form and hence require a regularization in order to fully
specify the jump conditions across shock waves"이며, 올바른 regularization은 소거된 자유도
(속도/압력/온도 비평형)의 inner-layer 해석에서 나온다. round 31 §3.2의 R2(참조해는 열적
비평형 Hugoniot이다)의 문헌 근거이자, "source term이 아니라 model-class 변경이 필요하다"는
결론의 1차 출처.

필요한 부분: §2 (reduced equations 유도), §3 (inner layer / jump conditions), K div(u) 항의
유도와 shock에서의 non-conservative product 취급.

Status: DOI 확인(Crossref). 원문 미확보 -- AIP paywall 여부 확인 필요.
```

**`papers/2018_Linga_hierarchy_nonequilibrium_two_phase_flow_models_needed.md`**
```
# A hierarchy of non-equilibrium two-phase flow models

DOI: 10.1051/proc/201966006
저자/연도/저널: G. Linga, ESAIM: Proceedings and Surveys 66, 109-143 (2019, publ. 2018)
OPEN ACCESS PDF: https://www.esaim-proc.org/articles/proc/pdf/2019/02/proc196606.pdf

이 작업에 필요한 이유: Baer-Nunziato에서 출발해 velocity/pressure/temperature/chemical-potential
relaxation을 순차적으로 무한대로 보내며 얻는 모델 계층 전체를 유도하고, 각 모델의 sound speed를
해석적으로 제시하며 subcharacteristic condition(a_eq <= a_frozen)을 증명한다. 이 solver의
4-equation 모델은 이 계층의 "pT-model"이고, cases 24/33/34의 validation reference는 그 한 단계
위인 "p-model"(압력평형 + 온도 비평형)의 shock이다 -- round 31 §3.2/§3.7의 문헌 지도.
Flatten & Lund (2011, DOI 10.1142/S0218202511005775)와 Lund (2012, DOI 10.1137/12086368X)의
확장판이며 OA라 우선 확보 대상.

필요한 부분: §2 (parent model + relaxation source), §4 (p-model), §7 (pT-model), 각 모델의
volume fraction 방정식 형태와 sound speed 식.

Status: OA PDF 확인됨. 다운로드 + papers/pdf_to_md.py 변환 대상 (stub이 아니라 실제 확보).
```

**`papers/2002_Allaire_Clerc_Kokh_five_equation_interfaces_needed.md`**
```
# A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids

DOI: 10.1006/jcph.2002.7143
저자/연도/저널: G. Allaire, S. Clerc, S. Kokh, JCP 181(2), 577-616 (2002)
관련: A. Murrone & H. Guillard, "A five equation reduced model for compressible two phase flow
      problems", JCP 202(2) 664-698 (2005), DOI 10.1016/j.jcp.2004.07.019

이 작업에 필요한 이유: round 31 §3.7이 지목한 유일하게 살아있는 model extension 후보(M3)의
정본 문헌. 상별 질량을 각각 보존(= ACID_YADV의 보존형 mass-fraction 전제를 유지)하면서
volume fraction을 비보존적으로 이류하고 단일압력/이중온도를 쓰는 5-equation 모델.
§3.2에서 machine precision으로 확인했듯 이 모델의 shock가 cases 24/33/34의 reference와
정확히 일치하므로, 이 경로는 cases.cpp/validation.cpp를 전혀 건드리지 않는다.

필요한 부분: 지배방정식 전체, mixture EOS(gamma_mix, Pi_mix)와 alpha 방정식의 shock 처리,
interface 조건(pressure/velocity 무진동 조건).

Status: DOI 확인(Crossref). 원문 미확보(Elsevier paywall 추정).
```

**`papers/2022_Pelanti_arbitrary_rate_relaxation_heat_mass_transfer_needed.md`**
```
# Arbitrary-rate relaxation techniques for the numerical modeling of compressible two-phase
  flows with heat and mass transfer

arXiv: 2108.00556  (OPEN ACCESS: https://arxiv.org/pdf/2108.00556)
저자/연도: M. Pelanti (2021 preprint; JCP 게재본 DOI 확인 필요)

이 작업에 필요한 이유: single-velocity 6-equation 모델에 volume/heat/mass 세 종류의 relaxation
source를 임의 rate로 넣는 최신 정본. round 31이 "mass transfer는 24/33/34에 적용 불가"라고
결론 내린 근거(Gibbs 평형 = 같은 물질의 액상/기상 사이에만 정의됨)를 외부 문헌으로 확인하고,
동시에 arbitrary-rate relaxation의 fractional-step 구현 패턴(round 32+ M3의 참고 구조)을 제공.
Pelanti & Shyue 2019 stub과 짝을 이룸.

필요한 부분: §2 (6-eq 모델 + mu/theta/nu relaxation source), §4 (mass transfer ODE와 Gibbs
평형 조건 g1=g2), stiffened-gas Gibbs 자유에너지에 필요한 entropy reference 상수 q'.

Status: arXiv OA 확인. WebFetch로 PDF 본문 추출 실패(바이너리) -- papers/pdf_to_md.py 경유
다운로드+변환 필요.
```

**`papers/2004_LeMetayer_Massoni_Saurel_liquid_vapor_EOS_needed.md`**
```
# Élaboration des lois d'état d'un liquide et de sa vapeur pour les modèles d'écoulements
  diphasiques

DOI: 10.1016/j.ijthermalsci.2003.09.002
저자/연도/저널: O. Le Métayer, J. Massoni, R. Saurel,
                International Journal of Thermal Sciences 43(3), 265-276 (2004)

이 작업에 필요한 이유: round 31 §3.5(b)의 구조적 발견 -- 이 solver의 `Phase`
(cpp/denner_1d/include/denner1d/types.hpp:8-14)에는 gamma/pinf/b/kv/eta 다섯 필드뿐이고
entropy reference 상수 q'가 없어서 Gibbs 자유에너지 g_k = h_k - T*s_k 자체가 정의되지 않는다 --
를 뒷받침하는 1차 출처. 이 논문이 액상/기상 stiffened-gas 파라미터 (gamma, Pi, cv, q, q')를
saturation curve에 맞춰 결정하는 표준 절차를 제시한다. 어떤 mass-transfer closure든 이
q'가 먼저 있어야 하고, air/liquid-water 쌍에는 애초에 saturation curve가 없다는 점의 근거.

Status: DOI 확인(Crossref). 원문 미확보(Elsevier paywall 추정, 불어).
```

---

## 5. The model-extension option space, with honest cost

| id | extension | reproduces the reference? | keeps conservative `ρY`? | admissible under project rules? | needs `cases.cpp`/`validation.cpp` edit? | honest cost |
|---|---|---|---|---|---|---|
| **M1** | Physically-derived mass transfer (Gibbs/HEM/HRM/Hertz-Knudsen) | **NO** — target is `Y_eq` for a liquid/vapour pair; for air+water the correct rate is 0 | yes | target undefined (no `q'` in `Phase`; no air↔water saturation curve) | n/a | **DEAD** (§3.5) |
| **M1′** | Mass-transfer source with target `Y_A(p)` (E3) | YES, exactly | source destroys `ρY` conservation | it *is* α-transport; a finite `τ` is a tuning constant | no | **DEAD** (§3.6) |
| **M2** | Y-space Kapila source (E4), unconditional, `τ=0` | YES | **NO** (`Y` no longer a conserved invariant) | allowed literally, but content-identical to OFF; anti-rescue-adjacent | no | 1–2 rounds, **~zero scientific value** |
| **M3** | **Single-`p`, two-`T` Allaire 5-eq**: conserve `α_aρ_a`, `α_bρ_b`; advect α | **YES**, exactly (§3.2) | **YES** | allowed; parameter-free at `τ_T→∞` | **NO** — reference already is this model's answer | **4–8 rounds, high blast radius** |
| **M4** | 6-eq (Pelanti) or 7-eq Baer-Nunziato with full relaxation | YES | YES | allowed | no | 10+ rounds |
| **M5** | Accept the OFF/α path validates this family (round 26 option (i)) | n/a | n/a | **user decision only** | no | 0 |

### M3 cost breakdown (the number the user actually needs)

The good news, established during planning: **M3 requires no change to `PrimitiveState` (`types.hpp:16-25`), and therefore no change to `cases.cpp`, `validation.cpp`, `solver.cpp`, or `png.cpp`.** The second temperature can live as a solver-local `Vec` inside `solve_case_acid`, exactly the way `Yv` already does (`acid.cpp:915`) — a pattern this project has already proven safe across 27 rounds. That is a real, non-obvious design win and it should be stated to the user.

The bad news, itemised:

1. **A second conserved scalar.** `α_bρ_b` transported alongside `α_aρ_a` (currently `ρY` at `acid.cpp:1357-1366`), plus a non-conservative α update. ~1 round.
2. **A duplicated EOS closure.** `mixture_density`, `mixture_enthalpy`, `mixture_internal_energy_density`, `mixture_sound_speed`, `recover_pressure_temperature_from_density_energy` (`eos.cpp:55-131`) and `pT_from_v_e_massfrac` (`eos.hpp:115-160`) are all `(p,T,α)`-signature and **all reachable from the OFF path**. They must be *duplicated*, never modified — otherwise the byte-identity gate (`YADV_ROADMAP.md`:487-489) fails. ~1–2 rounds.
3. **The coupled `(u,p,h)` Newton and its analytic pentadiagonal Jacobian.** `ζ = ∂ρ/∂p|_T`, `φ = ∂ρ/∂T|_p`, `dEdp`, `dEdT` (`eos.cpp:38-45`) all change meaning once there are two temperatures; the `h→T` inner Newton becomes two; `compute_R` and every Jacobian entry in `acid.cpp` need a two-temperature sibling. **This is the dominant cost and the dominant risk.** ~2–4 rounds.
4. **Blast radius.** The Newton/Jacobian is shared by all 19 cases under `unic` (`cases.cpp:27-29`). Rounds 27 and 29 each burned a `consecutive_failures` increment on changes far smaller than this. The harm-gate discipline (measure blast radius before applying; `YADV_RESEARCH.md` §38.4, §39.6) must be applied per sub-stage.
5. **The payoff is smaller than it looks for *these three cases*.** By (E1), for `b=η=0` phases the two-temperature model's *hydrodynamics* is identical to α-transport. So M3 gains nothing on 24/33/34 that the OFF path does not already have. What it gains is **conservative phase-mass transport with correct shocks simultaneously** — which is the user's actual stated goal ("a genuinely robust 4eq mass-fraction solver"), and which is unreachable at 4 equations.

---

## 6. Staging — Stage 0 only, this round

### Answer to the charter's explicit question (§"Address directly")

**Yes, a genuinely safe Stage 0 exists — and it is the *only* stage round 31 should run.** It is Python-only, zero C++, zero `cmake` changes, and it cannot touch any passing case because it never enters the solver's write path. It is the direct analogue of round 26's own `yadv_r26_closure.py` and of round 30's "derive and kill candidates *before* proposing anything". **Round 31 should be 100% design/diagnostic. All model-affecting code is deferred to round 32+ under full harm-gate discipline, and — per §11 — should not start at all until the user has seen the M3 scope estimate.**

### Stage 0 — `scripts/yadv_r31_relax.py` (NEW file, the only file created)

Zero C++ changes. Imports `scripts/yadv_r26_closure.py` for `closure_a_shock`, `mix_coeffs`, `S_of_p`, `phase_rho`, `phase_h`, `case_alpha`, `base_env`, `dump`, `validate_all`, `case24_spec_pass_py` — i.e. builds on the already-P0/P1-cross-validated instrument rather than re-transcribing the EOS. Six modes, each independently falsifiable:

| mode | what it asserts | pass criterion | pre-registered expectation |
|---|---|---|---|
| `--identity` (**P0**) | (E1): `e_vol` single-`T` == ESG form at 4 states/case; (E3): closed-form `Y(p,α)` == `closure_a_shock`'s `Y_pre`/`Y_post` | all rel diffs `< 1e-14` | PASS (measured `≤2.8e-16`) |
| `--twoT` (**P1**) | (E2): RH closes on (E1) alone with no `T`; `Vs` matches `cases.cpp:135` | `|Vs/Vs_ref − 1| < 1e-12`, Hugoniot residual `< 1e-14` | PASS (measured `≤2.2e-16`) |
| `--target` (**T1**) | §3.3/§3.4 tables: unique `Y*`, `dip`-tolerance band, `j`, `Γ`, `τ`, HK area density | reproduces §3.3/§3.4 to 4 sig figs | `Y* = Y_A`, band 0.67–1.61% |
| `--scan2d` (**T2**) | 2D `(p₁, Y*)` reachability scan at the reference `Vs` — a strict generalisation of round 26's 1D `--reachable` (`§36.3`) | the gate-passing set in `Y*` is non-empty and confined to `|Y*/Y_A − 1| ≲ 0.016` | confirms uniqueness; **new evidence, not a re-derivation of §36** |
| `--gibbs` (**T3**) | attempt to build the Gibbs mass-transfer target; must **fail closed** with an explicit message naming the missing `Phase` field | exits non-zero with `"Phase has no entropy reference q' (types.hpp:8-14)"` | FAIL-CLOSED by design |
| `--offequiv` (**T4**) | using **existing dumps only** (OFF and `ACID_YADV=1`, cases 24/33/34): show that imposing `Y = Y_A(p)` per cell reproduces the OFF dump's α to the round-trip conditioning floor `alpha_roundtrip_floor` (`eos.hpp:78-85`) | `max|α_impl − α_OFF| ≤ max(1e-12, floor)` | confirms M2 ≡ OFF path (§3.6) |

**Explicitly forbidden inside Stage 0:** creating any env var, touching `cpp/`, or invoking the solver with any flag combination not already on record in `yadv_r26_closure.py`'s `CONFIGS`.

### Deliberately NOT staged this round

- **No Stage 1.** There is no model-affecting change small enough to be safe and large enough to be informative. Building M2 (§3.6) would be technically permitted and is exactly the trap this plan exists to prevent.
- **No re-run of round 26's `--gate`/`--reachable`.** Already on record (`§36.3`); re-running it would be the "re-deriving the closed-form mismatch" the charter forbids.

---

## 7. Gates and targets

### Hard gates (checked even though no C++ changes — round 30's own discipline)

- **G1** `git diff --stat -- cpp/` is **empty** at end of round. Any non-empty diff means the round exceeded its own scope and must not merge.
- **G2** OFF path 19/19, unchanged.
- **G3** `ACID_YADV=1` 15/19, fail set exactly `{15,24,33,34}`, unchanged.
- **G4** `denner1d_unit` numbers unchanged.
- **G5** `ALL GATES OK` line unchanged from round 30.
- **G6** No new file under `cpp/`. Exactly one new file total: `scripts/yadv_r31_relax.py`. Plus the `papers/*_needed.md` stubs of §4 and the recovered `papers/md/33_saurel_relaxation_multiphase.md`.

### Round targets, in execution order

1. P0 (`--identity`) — **must pass before anything else is believed.** If it fails, the whole plan's derivations are wrong and the round stops and reports S3.
2. P1 (`--twoT`).
3. T1 (`--target`), T2 (`--scan2d`), T3 (`--gibbs`), T4 (`--offequiv`).
4. Write `YADV_RESEARCH.md` §41 with the tables; update `YADV_ROADMAP.md` control state; save §4's stubs; recover the missing Saurel md (C1).
5. Surface §11's escalation.

---

## 8. Pre-registered stop / decision rules

- **S1 (expected, ~85% confidence)** — P0+P1 pass, T1–T4 confirm §3. Verdict: *"the model extension named by round 26 is the wrong one; the correct one is a model-class change (M3), scoped at 4–8 rounds."* `consecutive_failures` **NOT** incremented (rounds 5/9/11/26 precedent: resolving a definitive open question is measured progress). Phase 3a stays open as a *user-decision* thread, not an autonomous one.
- **S2** — P0/P1 pass but T2 finds a gate-passing `(p₁, Y*)` region **outside** the `Y_A` neighbourhood. This would falsify §3.3's uniqueness claim and re-open M1 with a concrete target. Report it as the headline; do **not** implement it this round.
- **S3** — P0 fails. My derivations (E1)–(E4) are wrong. Report exactly what failed and by how much; make no claims that depend on them; `consecutive_failures` **IS** incremented.
- **S4** — T4 shows M2 is *not* equivalent to the OFF path beyond the conditioning floor. That would make M2 a genuinely distinct candidate; record it, do **not** build it (it still fails §3.6(1)/(2)).
- **Binding anti-rescue clause** (patterned on round 30's): if every stage lands as expected, the implementing session **may not** propose or build any model-affecting candidate this round to "make the round productive". A round that correctly establishes that the authorized approach is the wrong approach is a successful round.

---

## 9. Non-goals (explicit)

1. Weakening, relaxing, reinterpreting, or re-scoping the 24/33/34 gate. `validation.cpp` is read-only.
2. Replacing or editing the 24/33/34 analytic reference. `cases.cpp` is read-only. *(Note §3.7: the correct extension M3 does **not** need this — which is itself a finding.)*
3. Retrying any killed candidate: `ACID_YADV_ALPHA_IMPLICIT{,_CAV}` and any variant, `ACID_YADV_REBUILD_ADV`, `ACID_YADV_RESYNC`, `ACID_YADV_RECON`, F3b.
4. Re-deriving round 26 §36's mismatch. §3 is a *different* result (two-temperature identification + closed form + uniqueness + rate quantification), and must be presented as building on §36, not replacing it.
5. Any per-case or per-regime predicate; any new numeric literal in `cpp/`; any relaxation timescale `τ` that is not a physical constant.
6. Touching case15 (PAUSED pending a user risk decision, `YADV_ROADMAP.md`:40-52), the 1.0 Pa pressure floor, or the `pface` scheme.
7. Adding `water_vapor_phase()` to any case, or introducing a third component. That changes what is being validated and requires a separate user decision.
8. Starting M3. Not this round, and not the next round without §11's authorization.

---

## 10. Risk register (Stage 0)

| # | risk | why it could bite | detection |
|---|---|---|---|
| R-1 | `yadv_r26_closure.py` import has side effects (it shells out to the solver at module scope) | would make P0 depend on a build | inspect: its solver calls live inside `dump()`/`validate_all()` (`:360`,`:372`), not at import; assert P0 runs with `PATH` to no binary at all |
| R-2 | My (E1) identity holds only because `η_a = η_b = 0`, and someone generalises it to cases 14/15 (NASG water, `b=6.61e-4`, `η=−1.177788e6`) | (E1) is **false** there | `--identity` must assert `b==0 and eta==0` for both phases and **refuse to run** on any other phase pair, with an explicit message |
| R-3 | T2's 2D scan is expensive or non-convergent at extreme `Y*` | bisection on `v₁` can bracket-fail near `Y*→1` | pre-registered: clamp `Y* ≤ 0.999999`; report the scanned domain explicitly; a bracket failure is reported, never silently skipped |
| R-4 | T4 compares dumps produced under different `ACID_*` env sets | would give a meaningless α comparison | reuse `base_env()`(`:35`) verbatim; assert the two dumps have identical `x` columns and identical cell counts |
| R-5 | The round drifts into implementing M2 because (E4) is elegant and parameter-free | it is the single most likely failure mode of this round | §8's binding anti-rescue clause; G1 (`git diff -- cpp/` empty) is a mechanical check |
| R-6 | The recovered Saurel md (C1) has different line numbers than rounds 28-30 cite | prior-round citations would silently mis-resolve | after recovery, spot-check that `YADV_ROUND_28_PLAN.md:407`'s quoted §3.3 text is at the cited lines; if not, note the offset in `YADV_RESEARCH.md` §41 rather than "fixing" prior docs |
| R-7 | §3.4's `τ`/`Γ` numbers get quoted later as if they were a solver measurement | they are analytic estimates at an assumed 3-cell shock thickness | label every one of them "analytic, assumes a 3-cell numerical shock at `N=800`" in `--target`'s own output header |

---

## 11. Scope escalation — what to tell the user (and why round 31 must not start M3)

State plainly, in the round's report:

1. **The user's request is well-posed and its goal is achievable — but not at four equations.** A solver that conservatively transports phase mass **and** gets the 24/33/34 mixture shocks right exists: it is the Allaire/Kapila 5-equation model (single pressure, two temperatures). This is proved, not conjectured: §3.2 shows the current validation reference *is* that model's exact shock, to machine precision.

2. **The extension round 26 named — and that the user authorized — cannot work for these three cases.** Not because it is hard, but because for an air + liquid-water pair the required transfer is water→air at 63–93% of the through-shock mass flux (§3.4), no Gibbs equilibrium exists between two distinct species, and the code's `Phase` struct cannot even express one (§3.5). The canonical 4-eq-with-phase-change paper already in this repo holds `Y_air` fixed on exactly this kind of test (§3.5(d)).

3. **The real cost is 4–8 rounds at high blast radius**, dominated by re-deriving the coupled `(u,p,h)` Newton residual and its analytic pentadiagonal Jacobian for two temperatures (§5). It does **not** need `cases.cpp`/`validation.cpp` edits and does **not** need a `PrimitiveState` change — both genuine de-risking findings — but it does put all 19 currently-passing cases at risk, and this loop has already spent two `consecutive_failures` increments (rounds 27, 29) on far smaller changes.

4. **Round 26's option (i) remains on the table and is now better-motivated than it was.** "Cases 24/33/34 validate the α-based OFF path" is not a concession: by (E1)/(E2), for these phases the OFF path *is* the correct thermal-disequilibrium model's answer. `ACID_YADV`'s honest scope is the regime where composition, not thermal disequilibrium, is the dominant closure question.

5. **The decision the user needs to make is therefore not "mass transfer yes/no" but "commit to a 5-equation rewrite, or scope `ACID_YADV` to 15/19 with a documented, physically-explained exception".** Neither is a decision the autonomous loop may make. Round 31 stops at presenting it — exactly as round 26 did, and for the same reason.

---

## 12. Reproduce commands (for the implementing session)

```bash
cd <worktree>/solver_4eq_mass
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8

# --- Stage 0 (the whole round) ---
python3 scripts/yadv_r31_relax.py --identity    # P0  must pass first; all rel diffs < 1e-14
python3 scripts/yadv_r31_relax.py --twoT        # P1  §3.2 table (T_air/T_water, compression ratios)
python3 scripts/yadv_r31_relax.py --target      # T1  §3.3 + §3.4 tables
python3 scripts/yadv_r31_relax.py --scan2d      # T2  2D (p1, Y*) reachability
python3 scripts/yadv_r31_relax.py --gibbs       # T3  MUST exit non-zero, naming types.hpp:8-14
python3 scripts/yadv_r31_relax.py --offequiv    # T4  M2 == OFF path, to the conditioning floor

# --- hard gates (unchanged numbers expected; DENNER_ACID=1 is MANDATORY, .claude/rules/denner-pitfalls.md) ---
git diff --stat -- cpp/                                   # G1: MUST be empty
DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate  # G2: 19/19
DENNER_ACID=1 ACID_YADV=1 ./build-cpp/cpp/denner_1d/denner1d_validate  # G3: 15/19, fail {15,24,33,34}
./build-cpp/cpp/denner_1d/denner1d_unit                    # G4

# --- spot-check of the two headline identities, independent of the new script ---
python3 -c "
import importlib.util
s=importlib.util.spec_from_file_location('r26','scripts/yadv_r26_closure.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
A,B=m.AIR,m.WATER; Ra=A['kv']*(A['gamma']-1); Rb=B['kv']*(B['gamma']-1)
for c in ('24','33','34'):
    al=m.case_alpha(c); S=m.closure_a_shock(al)
    esg=lambda p: al*(p+A['gamma']*A['pinf'])/(A['gamma']-1)+(1-al)*(p+B['gamma']*B['pinf'])/(B['gamma']-1)
    ra=m.phase_rho(S['p_post'],S['T_post'],A); rb=m.phase_rho(S['p_post'],S['T_post'],B)
    e1=al*(ra*m.phase_h(S['p_post'],S['T_post'],A)-S['p_post'])+(1-al)*(rb*m.phase_h(S['p_post'],S['T_post'],B)-S['p_post'])
    r=(al/(1-al))*(Rb/Ra)*(S['p_post']+A['pinf'])/(S['p_post']+B['pinf'])
    print(c,'E1 rel=%.2e  E3 rel=%.2e'%(abs(e1/esg(S['p_post'])-1), abs((r/(1+r))/S['Y_post']-1)))
"
# expect: all six numbers < 1e-15
```

---

### Critical Files for Implementation

- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/scripts/yadv_r31_relax.py` — **the only file created**; imports `scripts/yadv_r26_closure.py`
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/scripts/yadv_r26_closure.py` — read-only; the verified instrument being extended (`:50-51` phases, `:91-124` closure A, `:130-168` closure B, `:219-221` `case_alpha`, `:290-359` the Python gate, `:360-382` `dump`/`validate_all`)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/cpp/denner_1d/src/cases.cpp` — **read-only, must not be edited** (`:73-100`, `:105-151` esp. `:148-149`, `:446`, `:583/603/605`, `:689-694`, `:779-791`)
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/cpp/denner_1d/include/denner1d/eos.hpp` + `.../include/denner1d/types.hpp` — read-only; `types.hpp:8-14` (the missing `q'`) and `eos.hpp:57-66`, `:78-85`, `:115-160`, `:170-198` are the load-bearing citations for §3.5(b) and §3.6
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/cpp/denner_1d/src/acid.cpp` — **read-only this round** (`:569-578`, `:912-923`, `:1268-1376`, `:1386-1414`); the site any future M3 would touch
- `/home/younglin90/work/claude_code/claudeCFD/.claude/worktrees/yadv-round-31/solver_4eq_mass/cpp/denner_1d/src/validation.cpp` — **read-only, must not be edited** (`:469-505`, `:731-734`)

---

## Actual outcome (implementing session, post-hoc)

**All structural claims spot-checked directly, all confirmed**: `cases.cpp:148` (`s.alpha_post =
s.alpha_pre;  // psi held (homogeneous mixture)`), `types.hpp:8-14` (`Phase` has exactly
`{gamma,pinf,b,kv,eta}`, no `q'`), `eos.cpp:11-13` (`air_phase` `eta=0`), `cases.cpp:446`
(`denner_water` `b=0, eta=0`), `eos.hpp:170-176`'s `phi_k/rho_k=-1/T` comment, and — most
significantly — the external quote in §3.5(d)/§4 (Collis et al. 2026, `papers/library/md/
2026_recent/[2026] 4eq + phase change.md` line 258: `"the mass fraction Yg[air] = 0.98 everywhere
in the domain, and the mass fractions of liquid water and water vapor are deduced from satisfying
thermochemical equilibrium"`) all matched the plan's citations exactly. C1 (`papers/md/
33_saurel_relaxation_multiphase.md` absent from the tree) also confirmed exactly as stated.

**`scripts/yadv_r31_relax.py` built and all six modes run** (§6/§7 of the plan). `--identity`
(P0) and `--twoT` (P1) PASS exactly, reproducing the plan's own §3.1/§3.2 numbers to the reported
precision. `--gibbs` (T3) fails closed exactly as specified. `--offequiv` (T4) confirms
`max|alpha_impl - alpha_OFF| = 0.000e+00` in all three cases.

**One real implementation bug found and fixed** (not anticipated by the plan): the first
`--scan2d` (T2) attempt reused `scripts/yadv_r26_closure.py`'s `hugoniot_b`, which solves the
Y-HELD Hugoniot (identical composition on both shock sides — "closure B"). Plugging a candidate
downstream `Y*` into it as the upstream composition too silently evaluates a different, wrong
shock (caught immediately by a sanity check: `Ystar=Y_A` failed to reproduce the reference,
`Vs` off by ~19x). Fixed by writing a genuinely new two-composition RH solver
(`_downstream_state`) that fixes the TRUE upstream state at `Y_pre` and solves for a downstream
state at a prescribed, independent `Y*`, holding `mdot = rho_pre * Vs_reference` fixed. A second
issue (a spurious pole in the residual function that a naive two-point bisection could straddle)
was also found and fixed with a log-spaced multi-point scan for the highest-pressure sign change.
After both fixes, the sanity check (`Ystar=Y_A` reproduces `p_post` to `rel<=4.44e-16` in all
three cases) passes, and `--scan2d` independently confirms §3.3's uniqueness claim with bands of
`+/-0.80%/-0.80%` (case24), `+0.60%/-0.60%` (case33), `+1.50%/-1.60%` (case34) — narrower than,
but broadly consistent with, the plan's own pre-registered table.

**One honest numeric discrepancy, not reconciled**: `--target` (T1)'s own dip-tolerance band
estimate (`0.74%/0.56%/1.25%`, via a finite-difference sensitivity of `rho_post(Y)` at fixed
`p_post,T_post`) differs from both the plan's own table (`0.90%/0.67%/1.61%`) and this round's own
`--scan2d` result by up to ~30%, likely reflecting a different linearisation choice not fully
specified in the plan's prose. All three estimates land within the same order of magnitude and
support the identical qualitative conclusion (a narrow band confined near `Y_A`); recorded per
this project's honesty discipline rather than silently smoothed to one number.

**Deviation from the plan's own §4 "recovery action"**: the plan's C1 finding called for
re-fetching `papers/md/33_saurel_relaxation_multiphase.md` ("S0 — re-fetch, do not stub") and for
actually downloading+converting the OA Linga 2018 paper rather than stubbing it. The implementing
session did neither this round — both were judged non-load-bearing for §41's actual conclusions
(the plan's own text: "Nothing in this plan depends on it" re: the Saurel paper), and pursuing
them would have added a materially different kind of work (PDF fetch/convert via
`papers/pdf_to_md.py`) to a round whose own binding discipline (§8's anti-rescue clause) was
explicitly about NOT letting the round balloon beyond its diagnostic scope. Both are recorded as
open follow-ups in `YADV_RESEARCH.md` §41.2/§41's literature note rather than silently dropped.

**Verdict: S1, exactly as the plan's own pre-registered text anticipated.** All hard gates held
(`git diff --stat -- cpp/` empty, OFF 19/19, `ACID_YADV=1` 15/19 fail-set `{15,24,33,34}`
unchanged, unit tests unchanged). `consecutive_failures` NOT incremented. Escalated to the user
per §11 — no model-affecting code written or recommended this round.