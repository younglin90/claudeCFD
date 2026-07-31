# GOAL (autonomous, do not stop until reached)

## >>> iter241: OFFICIAL 6/8 -- case25 FIXED by the faithful coupled energy <<<
The user's question ("we did it like Denner, why does it keep failing?") pinpointed
the root cause: we did NOT do Denner exactly. Denner is FULLY-COUPLED -- 3 unknowns/
cell (u,p,h), 3Nx3N block-tridiag, energy IN the Newton (spec §5, Eq.28). Our solver
coupled only (u,p) in a 2x2 system and SEGREGATED the energy. Weak/acoustic cases
don't care (pass); strong shocks need the coupling (the segregated static enthalpy
goes negative -> T floors -> rho explodes).
Implemented the faithful 3x3 coupled (u,p,h) path (Mat3/Vec3 + block_thomas3, energy
residual Eq.28, T-from-h Newton inversion for per-iteration consistency, numerical 3x3
Jacobian, line search on u,p,h). Gated by SolverConfig.coupled / ACID_COUPLED; case25
auto-uses it. VERIFIED (independent of the implementing agent):
  - 2x2 default unchanged: 01,02,04,05,13 still pass.
  - case25 (target Mach-10 air-shock/water-interface): NaN/blowup -> **PASS corr_u=0.998
    corr_p=0.994**.
  - case13 coupled: pass corr_u=0.996.
  - **OFFICIAL DENNER SUITE: 6/8** (01,02,04,05,13,25). 07, 24 remain.

REMAINING (for 8/8):
- **case24** (homogeneous Mach-10 mixture shock): coupled lifted corr_u 0.74->0.90 but
  still FAILS (corr_p=0.74, l2_p=2.86) -- the mixture shock OVER-PRESSURES (~earlier
  noted 5.5x at psi=0.5, over-heating). Beyond the energy coupling: the single-T
  mixture thermodynamics / sound speed at a strong mixture shock, and/or the reference
  (audit: case24 ref uses Kapila/Wood RH, Denner uses Eq.59-62 interface-region RH --
  make the reference faithful to Eq.59-62, then re-judge). Try ACID_COUPLED on 24 +
  inspect the post-shock mixture state vs the Eq.59-62 RH.
- **case07** (air-water acoustic refl/trans): finite, amp_ratio ~0.90 (Denner R/T
  metric OK) but fails peak_delta_p=18 & hf_u=0.237 -- 1st-order Backward-Euler phase
  error + ringing. Needs BDF2 (2nd-order time) and/or Minmod TVD higher-order space.

Speed note: the coupled path is ~15 compute_R/iter (numerical 3x3 Jacobian) + a per-cell
T-from-h Newton -> minutes per shock case. Fine within timeouts; optimise later
(analytic energy Jacobian, frozen Jacobian, or fewer T-inversion iters) if needed.


## >>> iter240: analytic Jacobian FIXED (NJAC == numerical) + case07 finite <<<
Ported the NJAC-revealed correct terms into the analytic Jacobian (fast, no 10x):
  (1) continuity dR/du: use the ACID per-face densities rfR/rfL (not the harmonic
      rho_f) + the u_i diagonal 0.5(rfR-rfL) [does NOT cancel at a shock]. Matched
      NJAC -2.73.
  (2) momentum dR/dp: the MWI (Rhie-Chow) pressure-velocity coupling d theta_f/dp =
      (3/4) dhat_f/dx + the transient d(rho(p) u)/dp.
NJAC now shows analytic == numerical (MB/MA/MC all match at the probed cell). The
buggy-Jacobian ROOT CAUSE of the 07/25 divergence is fixed. case07: NaN -> finite,
corr_u=0.97 corr_p=0.93. Default still 5/5 (01,02,04,05,13).

Remaining (all SEPARATE from the Jacobian):
- **case07** finite but fails the strict gate: with cfl 0.15 (smaller dt -> less
  Backward-Euler dissipation) amp_ratio_p/u recovered to ~0.90 (Denner's key R/T
  metric, was 0.78), but peak_delta_p=18 (>4) and hf_u=0.237 (>0.20) -> phase error
  + ringing from 1st-order BE. NEEDS higher-order time/space (Phase 10). The 1st-
  order BE has an amp-vs-ringing trade-off: high cfl damps (low amp), low cfl rings.
- **case25** CONFIRMED = the segregated energy (ACID_ISOTHERMAL freezes T -> the u-p
  machine is perfectly stable at the violent shock, max|u|~3000 to step 600). Root
  cause: the static enthalpy Hstat = rhoH - 0.5*rho*u^2 goes NEGATIVE during the
  inner iteration (conservative rhoH uses rho_o, kinetic uses the EOS rho_new --
  inconsistent before convergence) -> T floored to 1e-6 -> rho explodes. FIXED the
  NaN (reject non-physical T updates + under-relax 0.5) -> case25 now FINITE and the
  shock builds (maxp~1.2e8), BUT still INACCURATE (du=32033 vs ref 1434, corr_u=0.08,
  corr_p=-0.55; some cells stay T-floored, max|u| creeps, dt shrinks). The band-aid
  bounds it but the segregated energy can't get the violent shock right. NEXT (big):
  fully-COUPLE the enthalpy into the Newton (u,p,h 3x3 block-tridiag) -- the faithful
  Denner formulation -- so mass/momentum/energy stay consistent every iteration.
- **case24** mixture-shock over-pressure (separate energy bug).


## >>> iter239 BREAKTHROUGH: the analytic Jacobian was BUGGY -> use numerical (FD) Jacobian <<<
User's insight was correct. NJAC check (analytic vs finite-difference Jacobian at the worst
cell) revealed multiple WRONG/missing terms in the hand-derived analytic Jacobian:
  MB[0][1] (dRmom/dp) ana=0 vs num=7.55; MB[1][0] (dRcon/du) ana=0 vs num=-2.73;
  MA[0][1] ana=-0.5 vs num=-3.15; MA[1][0] ana=-0.985 vs num=-3.31.
So delta was non-descent -> divergence. FIX: replaced the analytic Jacobian with a NUMERICAL
block-tridiagonal Jacobian (finite differences, stride-5 graph colouring, 10 compute_R per
Newton iter; compute_R was factored out as a reusable lambda). RESULT: **case07 NaN -> PASS-
level (corr_u=0.972, corr_p=0.963)**; 04/13 still pass. The lesson: ALWAYS verify the Jacobian
numerically. case25 (violent Mach-10 IC shock) still NaN -- the exact Jacobian alone isn't
enough there; needs a line search (now easy: compute_R + rnorm lambdas exist) and/or IC
smearing. case24 mixture over-pressure is separate.


## >>> iter238 UPDATE: linear solve PROVEN exact; need a ROBUST NONLINEAR solver <<<
- PROVED block-Thomas linear residual ‖A*dxk-b‖/‖b‖ = 2e-16 (machine eps) for BOTH case25
  (diverging) and case13 (working) -> the linear solver is EXACT; ILU/GMRES/AMG (amgcl) give
  the SAME dxk -> would NOT fix divergence. (amgcl is still right for the future multi-D port.)
- Added the momentum-pressure convective Jacobian term (d(mdot*uconv)/dp via rho_f(p)) -> did
  NOT fix 07/25 (still diverge), 13/04 still pass. So the exact dxk is a NON-DESCENT direction
  even with a more complete J.
- CONCLUSION: need a robust NONLINEAR solver. Plan: (1) backtracking LINE SEARCH (below); (2)
  if it STALLS (no alpha reduces ‖R‖ because dxk is uphill) add **Levenberg-Marquardt damping**:
  solve (J + lambda*diag(J)) dxk = -R with lambda grown until ‖R‖ decreases -> always a descent
  step (lambda->inf is scaled steepest descent). (3) Alternatively/additionally smear the case25
  IC shock over ~3-5 cells (a 1-cell u-jump of 2869 is an unresolved initial residual) or use
  pseudo-transient continuation. The merit function is f=1/2 ‖R‖^2 (R = [Rmom, uref*Rcon]).

## >>> NEXT ACTION (definitive, do this first): LINE SEARCH (+ LM fallback) <<<
EXHAUSTIVE diagnosis (iter237) proved the 07/24/25 blocker is a single thing: the inner
Newton DIVERGES at the violent shock/interface -- BOTH p and u grow ~50%/iter exponentially,
fully dt-INDEPENDENT (halving dt 18 orders doesn't change it). Mechanism: a shock/interface
cell (case25 cell 100 = x=0.25) over-corrects -> u exceeds sqrt(2 h_total) -> e_int<0 ->
T->0 -> rho->inf, OR p collapses/explodes. ALL band-aids failed (T-ceiling, uref step limit,
positivity |u|<sqrt(2h), rho_floor, adaptive-dt-retry, under-relaxation; adaptive-om also BROKE case13). The Jacobian gives
a NON-DESCENT delta. The ONLY fix is a globalised Newton = **backtracking line search**:

RECIPE (cpp/denner_1d/src/acid.cpp, inner `for(it...)` loop):
1. Factor the "eval_thermo + ghosts + flux loop (theta,rho_f,dhat,pface,uconv,raup,rbup,
   rHaup,rHbup) + mdotL/mdotR + residual Rres" into a lambda `compute_R(Field& s)->double`
   that fills the outer flux vars + Rres and RETURNS ‖Rres‖ (e.g. sqrt(sum Rmom^2/pscale_m^2
   + Rcon^2/pscale_c^2), normalised). Declare the flux vars in the it-loop scope.
2. Each iter: normR0 = compute_R(s); assemble J from the (just-filled) flux vars; solve dxk.
3. BACKTRACK: Field sbak=s; for alpha in {1,.5,.25,.125,.0625,.03}: restore s=sbak; apply
   s.u += alpha*om*dxk[i][0] (with positivity clamp), s.p = max(s.p+alpha*om*dxk[i][1],1);
   energy update (T); double nr = compute_R(s); if (nr < normR0) break; (accept first alpha
   that reduces the residual). If none reduces -> keep alpha=smallest (or break the it-loop).
4. Converge when normR0 small (relative). Keep the existing adaptive-dt-retry as a fallback.
This makes every step monotonically reduce ‖R‖ -> no divergence. Expect it to fix 07 and 25
(and likely improve 24's accuracy). Test order after it builds: 04,13 (regression) then 07,25.
If the line search STALLS (no alpha reduces ‖R‖ at the shock), the Jacobian is too incomplete
-> add d R_mom/dp via convection (mdot's p-dependence: drhodp_up*theta + rho_f*dtheta/dp).

## North star
The faithful Denner ACID solver (`DENNER_ACID=1`, cpp/denner_1d/src/acid.cpp) PASSES
**all 8 Denner-paper 1D validation cases at Denner-paper accuracy**, with EOS, ICs, BCs
and references all faithful to the Denner ACID paper (JCP 367, 2018).

## Metric
`DENNER_ACID=1 denner1d_validate` pass_count over the 8 Denner cases = **8/8**.
(Project extensions 14,15 are out of scope.) Paper-accuracy gates (no artificially tight
gates like case25 interface_u<=1.5; no over-loose gates like case04 amp_ratio>=0.10).

## STATUS iter235: 5/8 PASS
PASS: 01(static), 02(gas-gas advection), 04, 05(acoustic), 13(shock tube 1e5 ratio).
REMAINING: 07 (NaN, Gaussian+wall -> needs faithful sinusoid+inlet/outflow),
24 (finite corr_p=0.74, needs accuracy + Denner RH ref + speed/CFL),
25 (NaN, violent Mach-10 IC shock u=2869 -> inner Newton diverges).
Keys that got here: ACID deferred-enthalpy flux (Eq.47), faithful SG EOS, defect-
correction coupled u-p + MWI + ACID face density, material_dt for advection (case02).
NOTE: HLLC baseline now 8/10 (case02 is faithfully t=0.7 which explicit HLLC can't
reach; ACID is the faithful target, this is expected).

### iter236: robustness band-aids INSUFFICIENT for 07/25 -> need line-search
Tried (all committed, safe, don't hurt 13/04/05): T-ceiling 1e6, Newton step-limit to a
FIXED uref (blown-up `a` must not feed back), rho_floor in the momentum Jacobian diagonal,
relative 1e-8 inner-convergence (also restored 04 from the bad 1e-6). 07 & 25 STILL NaN:
the inner Newton genuinely DIVERGES at the interface/shock (cells evacuate, p collapses).
**The proper fix is a LINE SEARCH (Newton globalisation)** + completing the Jacobian:
1. Factor the flux+residual into a lambda compute_R(Field)->(Rres, ‖R‖) [eval_thermo + flux
   loop (mdot,theta,raup,rbup,pface,uconv) + residual].
2. Per inner iter: assemble J from current state, solve delta, then line search
   alpha in {1,0.5,0.25,...}: accept the alpha that reduces ‖R‖; if none, smallest.
3. If the line search stalls (no alpha reduces ‖R‖), delta is not a descent direction ->
   the Jacobian is too incomplete; add the missing cross terms: d R_mom/dp via convection
   (mdot depends on p through rho_f and theta), and the full d(mdot)/dp in continuity.
4. This should fix 07 (acoustic+interface) and 25 (violent Mach-10 IC shock). Then 24.
case24 is a SEPARATE issue (finite, mixture-shock pressure 5.5x too high at psi=0.5 -- the
coupled solve over-pressurises the homogeneous mixture shock; energy flux IS conservative
for psi=0.5, so suspect the coupled p-solve / mixture compressibility at the shock).

### iter237 KEY DIAGNOSTIC: case25 divergence is fully dt-INDEPENDENT
With adaptive-dt-retry + ACID_DBG, halving dt from 2.5e-7 down to 1.4e-12 (18 orders!)
leaves max|u| pinned at ~3.82e5 every time. A correct implicit scheme MUST quiesce as
dt->0 (the transient rho*V/dt dominates the diagonal -> delta->0). It does NOT -> this is
a BUG in the spatial coupled solve, not a CFL/robustness issue. Hypothesis: at the
interface (x=0.5) or shock (x=0.25) a cell's residual is astronomically large (an evacuated
cell gives R_mom = -rho_o*u_o*V/dt ~ huge at small dt; or a near-singular 2x2 block), and
the block-tridiagonal elimination PROPAGATES it to every cell -> global delta blowup,
independent of dt.
NEXT (highest priority, fixes 07 & 25): instrument ONE inner iteration of case25 step 1 at
small dt -- print argmax|R_mom|, argmax|R_con|, argmax|delta|, and the 2x2 block det at
those cells. Determine whether it's (a) a huge residual at the interface/shock cell, (b) a
near-singular eliminated block, or (c) the Jacobian giving a non-descent delta. Then fix
that specific term (likely the momentum/continuity coupling or the ACID flux at the
interface under a strong gradient). The whole 07/25 family hinges on this one bug.

### Remaining (next context)
- **25 NaN at step 1** (PINPOINTED): block_thomas singular at i=102-106 (the shock cells
  x~0.25, u=2869). rho~6.8e-15 -> momentum diagonal B00~0 -> singular. The pre-shock air
  cells EVACUATE (p collapses to the floor, rho->0) because the inner Newton DIVERGES at the
  fully-formed Mach-10 shock IC (u=2869). Tried & did NOT fix: T-ceiling 1e6 (acid.cpp),
  Newton step-limit (|dp|<=50%,|du|<=10a), under-relaxation ACID_URF=0.3/0.5. case13 works
  (1e9 pressure jump) but has u=0 IC and the shock FORMS gradually; case25 starts with a
  pre-formed Mach-10 shock -> the coupled solve can't accept it in one step.
  NEXT IDEAS: (a) more robust convection linearisation (full upwind Jacobian for the
  momentum convection at the shock); (b) a Riemann-informed or sub-cycled first step;
  (c) initialise from a slightly smeared shock; (d) clamp rho floor in the momentum diagonal
  so a transient rho->0 can't make B singular (regularise B00 = max(B00, rho_floor*VdT));
  (e) check whether the MWI/convective-pressure terms destabilise at u>>a (Mach 8+).
- **24 (PINPOINTED bug)**: ACID post-shock p=76 MPa vs ref 13.94 MPa (5.5x too high) at
  similar rho (1019 vs 988) -> post-shock TEMPERATURE is ~5.5x too high = mixture-shock
  ENERGY OVER-HEATING. Sanity: ideal-gas RH gives ~12.6 MPa (matches the ref), so the ref
  is right and the ACID is wrong. case13 (separated air/water, psi=0 or 1) is fine; the bug
  is specific to the HOMOGENEOUS mixture (psi=0.5). The T-from-h round-trips, so it's the
  energy ADVECTION / shock-heating at psi=0.5. NEXT: check the ACID enthalpy flux & pressure
  work for a uniform mixture under a shock (compare a single mixture-shock step's energy
  balance to the RH; the over-heating suggests the pressure-work or enthalpy-advection term
  double-counts for psi in (0,1)).
- **02**: steady inlet BC added (left_bc="inlet", base_velocity=1, inlet_left generalised to
  f=0) -> steps 1-2 now uniform (inflow fixed!). BUT then a velocity blow-up at the slab edge
  (x~0.4675, the 860:1 air/water moving contact): u->1.7e7. The extreme density-ratio moving
  contact is the blocker. **FIX = faithful Denner gas-gas (Denner §7.1, rho 1.156/0.160,
  gamma 1.4/1.6, ratio ~7:1)**: both faithful AND avoids the 860:1 blow-up. Needs a 2nd gas
  phase (gamma=1.6, rho0=0.160 -> cv from R=p0/(rho0 T0)), single-step IC at x=0.1, reference
  = interface advects at u0 to x=0.8, p/u flat. (The project air-water box is NOT Denner's 7.1.)
- **07 NaN**: replace Gaussian+reflective-wall with Denner sinusoid f=5000 + inlet/outflow.

## Subgoal status
- 01 static interface — ✅ PASS (l2=0 exact)
- 04 acoustic air — ✅ PASS (corr 0.989)
- 05 acoustic water — ✅ PASS (corr 0.994)
- 02 interface advection — ❌ (needs faithful gas-gas + inlet BC)
- 07 acoustic refl/trans — ❌ (needs sinusoid + inlet/outflow, no wall)
- 13 air-water shock tube — ❌ (needs N=800,t=8e-4 + shock capture)
- 24 mixture M_s=10 shock — ❌ (needs CFL 0.5, Denner RH ref)
- 25 air-water shock-interface — ❌ (needs N=1000,t=2.78e-4 + shock capture)

## Done
- Faithful EOS (commit 025e7a9): Denner SG water (g=4.1,Pi=4.4e8,b=0,eta=0), air R=288.1.
- HLLC baseline 9/10 intact throughout (ACID is opt-in).
- Coupled defect-correction u-p solver + MWI + mixture EOS + ACID face density + VOF
  + ACID old-level: WORKS for static + acoustic.

## SHOCK-CAPTURE STATUS (iter232) — the remaining hard piece
ACID works for static(01)+acoustic(04,05). ALL shock/moving cases blow up: 02 (step2,
left inflow cell — needs inlet BC), 13/24/25 (strong shocks).
case13 (p_L=1e9 air / p_R=1e4 water, ratio 1e5): steps 1-2 FINITE (max|u|~150-240,
p=1e9), then a cell's |u|+a blows up -> **dt collapses -> Jacobian ~3e118 -> NaN**.
Added (safe, 04/05 unaffected, did NOT fix 13): MWI pressure-correction bounded to local
sound speed; all-Mach convective-pressure diagonal drhodp_i*|theta| in continuity Jacobian.

### Shock-capture TODO (next context)
1. **Find the blow-up source cell**: change the dt-step diagnostic to print argmax(|u|+a)
   and its neighbours each step (the contact/shock cell, NOT i=0 which is just the dt-collapse
   symptom). Currently the NaN print breaks on i=0 (first NaN delta after dt collapse).
2. **Robustness**: (a) limit the Newton update per iteration (|dp| <= f*p, |du| <= few*a);
   (b) positivity clamp p,rho,T after each update; (c) cap dt floor / detect divergence.
3. **First-order upwind for rho,h at interface/shock faces** (Denner §5.4, ξ=0 where
   |alpha_P-alpha_Q|>eps) -- the ACID partial-density upwind is already ~1st order, but verify
   the energy/enthalpy advection is monotone at the shock.
4. **Inner-Newton convergence**: check the it<60 loop actually converges at the strong shock;
   if not, under-relax (ACID_URF) or improve the Jacobian (add convective-pressure OFF-diagonal
   + d(rho_f)/dp full term, and the momentum convection linearisation).
5. **case02 (simplest moving case)**: add a steady inlet BC (Dirichlet u=u0, p extrapolated,
   incoming phase) -- the transmissive-at-inflow is ill-posed. OR set up faithful gas-gas
   (rho 1.156/0.160, gamma 1.4/1.6) which is lower density ratio. Do 02 before the shocks.
6. After robustness: 13 -> 25 -> 24, each at paper accuracy; then 07 (sinusoid+inlet/outflow).

## Plan (faithful, paper §6 ICs + audit fix list docs/denner_faithfulness_audit.md)
P1 ICs/BC/geometry per case, P2 references (24 RH = Eq.59-62), P3 gates -> paper accuracy.
Work order: assess 13/24/25 shock-capture state -> fix the closest -> iterate all to 8/8.
Commit each case as it passes. Do NOT stop to ask; report progress and continue.
