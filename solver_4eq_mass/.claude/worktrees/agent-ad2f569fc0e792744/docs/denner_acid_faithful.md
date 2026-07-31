# Faithful Denner ACID 1D Implementation — Method Spec & Plan

> Source: Denner, Xiao & van Wachem, *JCP* 367 (2018), "Pressure-based algorithm for
> compressible interfacial flows with acoustically-conservative interface discretisation
> (ACID)"; 2019 corrigendum; Xiao 2017 (fully-coupled base); Bartholomew 2018 (MWI).
> Extracted from `papers/library/pdf/`. **Goal: implement THIS exactly, validate against
> the paper's own 1D cases at the paper's own accuracy.** No artificial-tight gates.

## 0. Strategy (user-confirmed 2026-06-22)

- Implement Denner's method **faithfully** (그대로 충실히). No ad-hoc variants.
- Pass **only** the Denner-paper 1D cases that our suite already has:
  **01, 02, 04, 05, 07, 13, 24, 25** (mapping in §7).
- Pass criterion = **Denner-paper accuracy** (1st-order convergence, graphical agreement
  with the exact Riemann/linear-acoustic solution, no spurious oscillations, finite).
  **Drop** the over-strict custom gates (e.g. case25 `interface_u≤1.5`).
- Defer 14 (reversed shock tube) and 15 (cavitation) — not in Denner.
- Build order (easy→hard): interface equilibrium (02,01) → acoustics (04,05,07)
  → shock tube (13) → shock-interface (24,25).
- New clean module inside `cpp/denner_1d/` (constraint: only modify denner_1d). Gated /
  separate solve path during development; existing HLLC stays as the 9/10 fallback until
  the ACID solver passes the 8 cases.

## 1. Governing equations (1D, inviscid) — 4-eq one-fluid + VOF

Primary variables: **u, p, h** (density from EOS). Colour function ψ∈[0,1] advected separately.

```
momentum:    ∂(ρu)/∂t + ∂(ρu·u)/∂x = -∂p/∂x                       (1)
continuity:  ∂ρ/∂t   + ∂(ρu)/∂x   = 0                            (2)
energy:      ∂(ρh)/∂t + ∂(ρu·h)/∂x = ∂p/∂t                        (3)   [enthalpy form, RHS = transient p only]
VOF:         ∂ψ/∂t + ∂(ψu)/∂x - (ψ+K)·∂u/∂x = 0                   (32)
```
- specific total enthalpy: `h = cp·T + u²/2`.
- compressibility factor (material-dependent, VOF only):
  `K = [ρb·ab² - ρa·aa²] / [(1-ψ)·ρb·ab² + ψ·ρa·aa²]`           (11, cleaned)
- interface conditions: u, p continuous; ρ discontinuous (contact).

### EOS — Stiffened Gas (ideal gas = Π=0)
```
ρ  = (p + Π)/(R·T)                                                (4)
a  = sqrt(γ·(p+Π)/ρ)                                              (5)
cp = γ·cp0·(p+Π)/(p+γ·Π),   cp0 = γR/(γ-1)                        (7)   [varies with p for SG]
h  = cp·T + u²/2                                                  (6)
```
Phase props per fluid (a,b): `ρ_k=(p+Π_k)/(R_k T)`, `a_k`, `cp_k`. R_k = a0²/(γ_k·...) — derive from Table 1.

### Fluid table (Table 1):  [γ, Π(Pa), ρ0(kg/m³), a0(m/s)]
```
Air     1.400   0          1.157   347.8
Helium  1.667   0          0.164   1008.2
Argon   1.660   0          1.748   308.2
Water   4.100   4.4e8      998     1344.6
Copper  4.220   3.24e10    8960    3906.4
```
R_k from a0² = γ·(p0+Π)/ρ0 and ρ0=(p0+Π)/(R·T0) at p0=1e5,T0=300 (or as each case sets).

## 2. Discretisation

- **Space**: cell-centred collocated FV. Non-advected vars → central (Eq.15).
  Advected vars → TVD limiter ξ_f (Minmod); **at interface faces (|ψ_P-ψ_Q|>ε) force
  1st-order upwind (ξ_f=0) for ρ and h** (Section 5.4 — fixes TVD stall).
- **Time**: implicit. 1st-order Backward Euler to start; BDF2 later.
  BE: `∂φ/∂t·V ≈ (φ_P - φ_P^old)/Δt·V`.
  BDF2 (constant step): `(3φ - 4φ^old + φ^old2)/(2Δt)·V` (corrigendum-correct).

### Discrete equations (BE form; all transient terms same scheme)
```
continuity→pressure:  (ρ_P^{n+1}-ρ_P^old)/Δt·V + Σ_f ρ̃_f ϑ_f A_f = 0,   ρ_P^{n+1}=(p_P^{n+1}+Π)/(R T)   (23,24)
momentum:   (ρ_P u_P - ρ_P^old u_P^old)/Δt·V + Σ_f ρ̃_f ϑ_f ũ_f A_f = -Σ_f p̄_f n_f A_f                   (27)
energy:     (ρ_P h_P - ρ_P^old h_P^old)/Δt·V + Σ_f ρ̃_f ϑ_f h̃_f A_f = (p_P^{n+1}-p_P^old)/Δt·V             (28)
```
Newton-linearise products (Eq.25/29/30): `(ab)^{n+1} ≈ a^n b^{n+1} + a^{n+1} b^n - a^n b^n`.

### Pressure equation structure (Xiao 2017 Eq.28) — all-Mach
```
(∂/∂t)(p/(RT))·V                                   [compressible transient]
 + Σ_f ρ_f ū_f^n n_f A_f                            [vel coupling]
 - Σ_f ρ_f d̂_f [(∇p^n)_f - (∇p)_f^bar] n_f A_f      [incompr. Laplacian]
 + Σ_f (A_f u_f^C /(R T_f)) p_f^n                   [compressible convective p]
 = Σ_f ρ_f (u_f^C - u_f^{n-1}) n_f A_f
```
coeff ratio (convective/Laplacian) = M_f² → incompressible Poisson at low M, hyperbolic at high M.

## 3. MWI advecting face velocity (ACID Eq.20 / Bartholomew Eq.51,61)
```
ϑ_f = ū_f                                                  [linear-interp face velocity]
    - d̂_f·[ (p_Q-p_P)/Δs - ((1-l)·(∂p/∂x)_P + l·(∂p/∂x)_Q) ]   [pressure 3rd-deriv filter → kills checkerboard]
    + d̂_f·(ρ_f^old/Δt)·( ϑ_f^old - ū_f^old )                    [transient correction → dt-independent]
d̂_f = ρ̄_f · (V_P/e_P + V_Q/e_Q) / ( 2 + (ρ̄_f/Δt)(V_P/e_P + V_Q/e_Q) )      (21)
1/ρ̄_f = (1-l)/ρ_P + l/ρ_Q     [HARMONIC face density — needed for big ratio]              (22)
```
`e_P` = sum of central coeffs of u from advection terms at P (≡ a_P). l=½ on equidistant mesh.

## 4. ACID interface (THE KEY) — Section 5
Core: assign cell P's ψ to its ENTIRE stencil (ψ piecewise-const), discretise single-phase,
re-evaluate ρ,h face values from ψ_P. → asymmetric ρ̃_f, cp_f; symmetric ϑ_f. No Riemann.

### Density (Eqs.37,40-44)
```
ρ = ρa + ψ(ρb-ρa)                       (37)  [linear partial-density blend = isobaric]
ρ̃_f = ρ_U + (ξ_f/L_f)(ρ_D-ρ_U)          (40)
 ρ_U = ρa,U + ψ_P (ρb,U - ρa,U)          (41)  ← ψ_P (the discretised cell), NOT ψ_U
 ρ_D = ρa,D + ψ_P (ρb,D - ρa,D)          (42)  ← ψ_P
old levels: ρ_P^old = ρa,P^old + ψ_P (ρb,P^old - ρa,P^old)                                  (43,44)
```

### Enthalpy (Eqs.45-56) — deferred correction (h is primary, can't overwrite)
```
cp = [ρa cpa + ψ(ρb cpb - ρa cpa)]/ρ     (46)   [density-weighted mixture cp]
target face enthalpy:
ĥ_f = (1/ρ̃_f)[ ρ_U h_U + (ξ_f/L_f)(ρ_D h_D - ρ_U h_U) ]    (47)
 h_U = cp,U T_U + ½u_U²,  cp,U = [ρa,U cpa,U + ψ_P(ρb,U cpb,U - ρa,U cpa,U)]/ρ_U   (48,50, ψ_P)
 (same for D)
deferred correction: δh_f = ĥ_f - h̃_f ; energy advection uses (h̃_f + δh_f).   (52,53)
 → δh_f=0 in bulk (conservation preserved), active only at interface.
```

### Thermo consistency (Eqs.57,58) mixture sound speed
```
1/(γ-1) = (1-ψ)/(γa-1) + ψ/(γb-1)        (58)
a = sqrt( [ρa cpa + ψ(ρb cpb - ρa cpa)] / [(1-ψ)/(γa-1)+ψ/(γb-1)] · T )   (57)   [<0.33% err]
```

### Mixture RH (Eqs.59-62) for shock cases (pre-shock II stationary, M_s=u_s/a_II)
```
p_I/p_II = 1 + (2γ/(γ+1))(M_s²-1)/(1+Π̂/p_II)                       (59)
Π̂ = ((γ-1)/γ)(ρ_II cp,II T_II) - p_II        [=0 ideal]            (60)
ρ_I/ρ_II = [((γ+1)/(γ-1))(p_I+Π̂)/(p_II+Π̂)+1] / [((γ+1)/(γ-1))+(p_I+Π̂)/(p_II+Π̂)]   (61)
u_I = u_s(1 - ρ_II/ρ_I)                                             (62)
```

## 5. Solver loop (Section 6) — fully coupled, no under-relaxation
1D ⇒ 3 unknowns/cell (u,p,h) ⇒ **3N×3N block-tridiagonal** → **block-Thomas** (no PETSc needed).
```
per time step:
  shift old levels
  advect ψ (Eq.34 CN; 1D: bounded upwind/CICSAM-lite) with K term
  outer loop m:
    ACID: ρ,cp,h face/cell values from ψ_P
    inner loop n (barotropic, ρ=ρ(p) at fixed T^m):
       assemble block-tridiag [momentum|continuity-pressure|energy] w/ Newton-lin + MWI ϑ_f
       block-Thomas solve → (u,p,h)^{n+1}
       update ρ from p^{n+1},T^m ; update ϑ_f
       until ‖res‖/‖b‖ < ε                                    (65)
    update ρ from p,T^{m+1}; update T from h^{m+1}; props
    until (65) AND δ^{m+1}<ε                                   (66)
```
Start: BE + 1st-order upwind everywhere (matches paper's shock runs); add Minmod + BDF2 after core works.

## 6. The 8 Denner validation cases — ICs + relaxed (paper-accuracy) pass

| our | Denner § | setup (ICs) | t_end | pass = paper accuracy |
|---|---|---|---|---|
| 02 | 7.1 | L=1,N=500,Co=0.5; u0=1,p0=1e5,T0=300; left ρ=1.156 γ=1.4 / right ρ=0.160 γ=1.6; sharp ψ@x=0.1 | 0.7 | u,p stay flat (≤~0.1% of p0); interface@x≈0.8; ρ step no osc/no blow-up |
| 01 | 7.1 | same but u0=0 (static) | — | u≈0, p flat (equilibrium preserved, machine-ish) |
| 04 | 7.3.1 | air, f=2000, δu0=0.01u0 acoustic inlet | 2.3e-3 | δp=±4.02Pa, δρ=±3.32e-5, λ=0.174 (≤~0.5%) |
| 05 | 7.3.1 | water, f=6000 | 6.5e-4 | δp=±13416Pa, δρ=±7.42e-3, λ=0.224 |
| 07 | 7.3.2 | He-air f=5000 / air-water f=5000 / Ar-air f=2000; δu0=0.02u0 (Eq.69) | — | refl/trans ratios: He-air p_trans/p_refl≈3.40, air-water≈2.00, Ar-air≈-5.92 (≤~few%) |
| 13 | 7.5.2 | L=2,N=800(we use 400),interface@0.5; u0=0,T0=300; air p_L=1e9 / water p_R=1e4 | 8e-4 | match exact stiffened-gas Riemann (wave pos ~few cells, states 1st-order, monotone) |
| 24 | 7.4.1 | air-water mixture ψ∈{.25,.5,.75}, M_s=10 shock; L=1,x_s0=0.1,Co=0.5,BE+1st upwind | 0.7/u_s | shock@correct x; post-shock=RH(59-62); O(Δx); monotone |
| 25 | 7.4.4 | L=1,N=1000,Δt=1e-8; shock x_s0=0.25, interface x_ψ0=0.50, M_s=10; post-shock I u=2869.3 p=1.165e7 ρ_air=6.614; pre II u=0 p=1e5 ρ_air=1.157 | 2.78e-4 | wave positions + states match Riemann to 1st order; no spurious interface osc; finite |

Extra Denner 1D cases (not in our suite, optional validation): §7.5.1 subsonic gas-gas
(L=1,N=400,@0.5; L:u=0 p=2e5 ρ=3.57 γ=1.66 / R:u=0 p=1e5 ρ=1.20 γ=1.4; t=8e-4),
transonic gas-gas (L:u=200 p=5e5 ρ=8.92 γ=1.66 / R same; t=6e-4), §7.4.3 air-helium
shock-interface, §7.4.5 impedance-matched.

## 7. Build phases
- **B1** ✅ this doc.
- **B2** SG EOS (per-phase + mixture ρ,a,cp,h,R), field/BC structs, exact stiffened-gas
  Riemann solver (for shock-tube reference).
- **B3** single-phase coupled u-p-h block-tridiag solver + MWI; validate single-fluid
  acoustic (04,05) + a single-fluid shock tube → core machinery correct.
- **B4** + VOF advection + ACID interface; validate 02,01 (equilibrium) + 13 (air-water tube).
- **B5** validate 07 (refl/trans), 24, 25 at paper accuracy. Relax validation.cpp gates to §6.
- **B6** (later) Minmod + BDF2 for 2nd order where smooth; sophistication.

## CURRENT STATUS (iter230, commit 326c540) — ACID solver WORKS, 4/10
The defect-correction coupled u-p solver + MWI + mixture EOS + ACID face density now
works. **PASS: case01 (static interface, exact l2=0), case04 & case05 (acoustic, corr
0.989/0.995)**. ACID full = 4/10; default HLLC = 9/10 (ACID opt-in, intact).

Key fixes that got here (all in cpp/denner_1d/src/acid.cpp):
- DEFECT-CORRECTION form (mdot single source, RHS=-R, approx Jacobian) — killed the
  velocity checkerboard / instability.
- mixture EOS (alpha-blend rho/cp, mixture drho/dp, actual mixture_sound_speed for CFL).
- ACID per-cell face density rho_f^(i)=alpha_i*rho_a_up+(1-alpha_i)*rho_b_up (Eqs.41-42).
- VOF colour advection (Eq.32, K=0) + ACID old-level density (Eqs.43-44).
- transient-MWI theta_o initialised from the initial velocity.

### BLOCKER for the mixing cases (02, 13, 24, 25): EOS params
case02 blows up at step 2 because the PROJECT water EOS has eta=-1.18e6 (NASG reference
energy) while air has eta=0. Once VOF makes an interface cell fractional, the eta mismatch
injects a spurious temperature at mixing. **Denner Table-1 params have eta=0 AND b=0 (pure
SG), so h_k = gamma_k cv_k T with no offset and mixing is consistent → no spurious T.**

### NEXT (faithful path): use Denner Table-1 params for the ACID validation
The cases (cases.cpp) are set up with PROJECT EOS params (NASG water gamma=1.187, eta!=0).
Faithful Denner validation must use Denner Table-1 (air gamma=1.4 Pi=0 eta=0; water
gamma=4.1 Pi=4.4e8 eta=0, R=1469.8 cv=474.1). Plan:
1. Add a Denner-param case path (denner_sg_phase for air/water) + Denner ICs from §6,
   and reference solutions (exact stiffened-gas Riemann / linear acoustics / RH).
   Keep the project cases for the HLLC baseline; ACID validates against the Denner setup.
2. With eta=0,b=0 the eta-mixing blowup disappears → case02 advection should pass; then
   13 (shock tube), 24/25 (shock-interface).
3. If a residual interface energy error remains, add the ACID deferred-enthalpy correction
   (Eqs.47-52, psi_P-blended upwind enthalpy).
Relax validation.cpp gates to paper accuracy (§6) for the ACID cases.

## CURRENT STATUS (iter228, commit 47b82f8)
Scaffold built (`acid.{hpp,cpp}`, `DENNER_ACID=1`), default HLLC 9/10 intact (opt-in).
Diagnostics:
- **single-phase case04 (air acoustic)**: FINITE but acoustic amplitude 93x too large,
  corr_u=0.028 → **coupled u-p dynamics bug** (not yet correct). p somewhat correlated
  (corr_p=0.66).
- **two-phase case01 (static interface)**: NaN → expected (ACID interface NOT yet
  implemented; without it the density jump blows up — that's exactly what ACID fixes).

### Debug findings (iter229) — assembly needs a clean rewrite
With `scripts/acid_test.sh` (proper file, avoids the WSL inline-bash var-expansion bug):
- **case04 ISOTHERMAL**: blows up, du=29.6 (ref 0.01), corr_u=-0.13.
- **case04 FULL**: du=0.76 (76x too large), corr_u=0.027 (≈noise), corr_p=0.67.
Signature = **velocity checkerboard not suppressed** (u uncorrelated/noise, p smoother).
Targeted patches (transient-MWI old-ubar fix, under-relaxation) did NOT resolve it.
Root: the hand-rolled ABSOLUTE-form 2x2 assembly is fragile — the continuity's velocity
coupling is the central divergence ½(u_{i+1}-u_{i-1}) (odd-even blind to u), and the MWI
pressure-Laplacian coupling is ~1000x weaker than the other continuity terms at this dt,
so it cannot pin the velocity checkerboard. The explicit/implicit split of mdot between
the flux loop and the assembly is also error-prone.

**DECISION: rewrite the coupled solve in DEFECT-CORRECTION (Newton residual) form** next:
- compute the full nonlinear residual R(u,p) [momentum, continuity] using the SAME mdot[f]
  (one source of truth) → RHS = -R.
- assemble the Jacobian (∂R/∂u, ∂R/∂p) consistently → matrix.
- solve A·δ = -R, update (u,p) += δ; iterate to ‖R‖→0.
This guarantees flux/assembly consistency and is far easier to verify (residual→0 ⇔ correct).
Also: unit-test block_thomas on a small saddle system first; add a no-forcing uniform-state
preservation test as the very first gate.

### Debug findings (iter228)
- `ACID_ISOTHERMAL=1` (freeze energy) STILL blows up case04: p→3.75e5, u→579 over ~4800
  steps then dt collapses (9.5e-14). ⇒ **bug is in the coupled u-p solve, NOT energy.**
  It is a SLOW/MILD instability (amplification slightly >1 per step), i.e. anti-dissipation,
  not an immediate divergence. Signs/units of momentum pressure-grad & continuity MWI
  Laplacian were reviewed and look correct.

### Debug plan (next session) — single-phase u-p first
1. Suspect list for the mild u-p instability (test each in isolation on case04 ISOTHERMAL):
   - **lagged outer iteration divergence**: the it<40 loop lags mdot/rho and re-solves with
     NO under-relaxation; may not converge. Try: (a) freeze mdot at the linear-interp ūbar
     (no MWI feedback within the loop) to test; (b) add under-relaxation; (c) check the
     convergence break threshold.
   - **MWI transient term** `(rho_f/dt)*dhat*(theta_o-ūbar_o)`: indexing of uu_o/theta_o,
     and whether it injects energy. Try disabling it.
   - **aP for dhat** omits convection (uses transient-only); fine at low u but verify.
   - **inlet** currently folds ghost→MB instead of a FIXED Dirichlet mdot[0]→RHS — wrong;
     but interior blow-up suggests a deeper issue than the BC.
   - **block-Thomas 2x2**: unit-test it on a known small system (saddle-point) — a bug here
     would corrupt everything.
   - **momentum lacks velocity dissipation**: convection is upwind (has dissipation), but
     verify the central velocity-divergence in continuity isn't decoupling (the MWI Laplacian
     should couple — confirm dhat magnitude is large enough; dt/dx ~ 1.4e-3 may be too weak).
2. Sanity harness: a uniform single-phase state with NO forcing must stay uniform — add a
   tiny standalone driver (set u=u0,p=p0 uniform, transmissive, 100 steps, assert flat).
   This is the cleanest first test (cheaper than case04 acoustic).
3. Once 04/05 match linear theory, add **ACID interface** (§4 Eqs.41-42,47-53) → 01/02/13.
4. Then 07, 24, 25. Relax validation.cpp gates to §6 (paper accuracy).

## Notes
- pdftotext is the reliable source (md/ equations are PNG). Re-extract specific eqs from
  `papers/library/pdf/2018_Denner_Xiao_vanWachem_ACID...pdf` if a formula is ambiguous.
- corrigendum only fixes varying-step BDF2; ACID formulas unchanged.
- 1D block-Thomas replaces the paper's PETSc BiCGSTAB (exact in 1D).
