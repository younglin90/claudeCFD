# Parameter-free suppression of the collocated pressure checkerboard mode in the implicit acoustic block

> Literature study, 2026-07-13. Question: replace the tuned biharmonic face dissipation
> `p_f = (p_L+p_R)/2 − (D/8)(−p_{i−1}+3p_i−3p_{i+1}+p_{i+2})`, `D = 0.02`,
> in the 5-equation IMEX solver's implicit acoustic block with a closure that has **no user-tuned coefficient**.
> Strict rule used throughout: any O(1) constant chosen by an author counts as tuning; constants *derived* from
> the scheme (½, Z-impedance weights, Mach-polynomial splits, momentum-matrix coefficients) do not.
>
> Source pools: local corpus `../papers/md/` (88 files) + `../papers/*.md` summaries (≈50 files); targeted web
> checks for Bartholomew 2018, Cubero–Fueyo, SLAU2, Saade–Lohse–Fuster. Every claim cites a local file or URL.

## 0. Notation — our acoustic block

The block referred to as (18): frozen-coefficient (u,p) Crank–Nicolson solve,

```
ρ_i (u_i^{n+1} − u_i^*)/Δt + ( p̂_{i+1/2} − p̂_{i−1/2} )/Δx           = 0
(p_i^{n+1} − p_i^*)/Δt     + ρ_i c_i² ( û_{i+1/2} − û_{i−1/2} )/Δx    = 0
```

with `φ̂_f = θ φ_f^{n+1} + (1−θ) φ_f^n`, `θ = 1/2` (CN), coefficients `ρ_i, c_i, Z_i = ρ_i c_mix,i` frozen at `t^n`
(cf. `docs/five_eq_IMEX_current_formulation.md` §7.2, §11). With the central closure
`p_f = ½(p_L+p_R)`, `u_f = ½(u_L+u_R)` the composed discrete wave operator is a 2Δx-central product whose kernel
contains the odd–even mode `p_i = (−1)^i` — the checkerboard. Theory reference: Dellacherie, *Checkerboard modes
and wave equation*, Proc. ALGORITMY 2009, pp. 71–80 (cited in `../papers/md/37_paper.md` ref [81] and
`../papers/md/42_hope_collins_2022_lowmach_artificial_diffusion.md` ref [16]). Hope-Collins & di Mare 2022 show
formally that a discrete scheme constraining only `∂_x p` (central gradient) admits odd–even modes; a constraint on
`∂_xx p` is needed (`42_hope_collins_2022...md`, §around Eq. (26)–(34)).

PE ground rule (non-negotiable): at an isolated material interface with uniform (p,u), any face correction
proportional to Δp or Δu across the face is machine-precision safe; corrections carrying ρ-, α-, c- or
h-*differences* that survive at uniform p are NOT.

---

## A. Godunov / acoustic-Riemann face closure  ★ primary candidate

### Formula (adapted to block (18))

Impedances `Z_i = ρ_i c_mix,i` frozen at `t^n` (Kapila mixture c, as already in
`five_eq_IMEX/residual.py::implicit_face_pu`, `imp_dissipation_form='acoustic_riemann'`):

```
û_{i+1/2} = ( Z_L u_L^{n+1} + Z_R u_R^{n+1} − (p_R^{n+1} − p_L^{n+1}) ) / (Z_L + Z_R)
p̂_{i+1/2} = ( Z_R p_L^{n+1} + Z_L p_R^{n+1} − Z_L Z_R (u_R^{n+1} − u_L^{n+1}) ) / (Z_L + Z_R)
```

(and the same closure with n-values for the explicit CN half). Both faces are linear in the n+1 unknowns with
frozen Z → the block stays a 2×2-block tridiagonal linear solve; no Newton. This is exactly Peluchon–Gallice–
Mieussens Eq. (16) with slopes `ā₋ = Z_L`, `ā₊ = Z_R` (`../papers/md/25_peluchon_2017_imex_acoustic_transport.md`,
Eqs. (13)–(16), (36); JCP 339 (2017) 328–355), which is itself the Gallice simple Riemann solver for the Lagrangian
acoustic system; the equal-Z restriction is the Suliciu/Chalons–Girardin–Kokh Lagrange-projection acoustic flux
(CGK16/CGK17 cited therein).

### How the Lagrange-projection school avoids checkerboard in the *implicit* step

They never use a central face closure: **the implicit operator is assembled from the acoustic Riemann flux itself**.
Peluchon's IM1 solves the linear system `(Id + (Δt/Δx) M̃) W^{n+1} = W^n` where the tridiagonal blocks of `M̃`
contain `1/(ā₋+ā₊)` couplings of the pressure jump into the velocity/divergence row (file
`25_peluchon_2017...md`, matrices Ã, B̃, C̃ around Eq. (36)). The Δp term in `û` yields, in the p-equation, a
compact Laplacian `+ ρc²Δt/(Z_L+Z_R) ∂_xx p ≈ (c/2)Δt ∂_xx p` — the checkerboard mode `(−1)^i` produces
`∂_xx p ≠ 0` and is damped every step; dissipation is automatic, no constant anywhere. Tallois–Peluchon–Villedieu
extend the *same* matrix to a modified Crank–Nicolson `(Id + Δt/(2Δx) M)(X† − X^n) = −(Δt/Δx) M X^n`
(`../papers/md/29_tallois_2022_2nd_order_imex_twophase.md`, Eqs. (3.14)–(3.16); Comput. Fluids 244 (2022)) —
i.e. literally our time discretization with the Riemann closure inside. Second order in space is recovered by a
quasi-Newton defect correction: matrix built from the first-order Riemann flux, RHS from the second-order flux
(ibid., Eq. (3.18)) — a directly reusable recipe for keeping our MUSCL faces while the implicit matrix stays
first-order Riemann.

### Verdict: coefficient-free — YES

Weights are Z-derived. The only constant in Peluchon is `k = 1.01` in the positivity-slope enlargement
`ā₋ = k·max(C̄₋(r₀), ρ_l c_l)` (Eq. (27)) — a positivity safety factor derived from exact admissibility conditions
(Prop. 3), not a dissipation knob; the baseline choice `ā₋ = Z_L, ā₊ = Z_R` needs no constant at all.

### PE preservation at material interfaces

Exact to machine precision: at uniform (p,u), `p̂_f = p (Z_R+Z_L)/(Z_L+Z_R) = p`, `û_f = u`, independent of any
ρ, α, c jump — all dissipation is strictly ∝ Δp or Δu. The scheme "preserves contact discontinuities, i.e. the
evolution of constant pressure and velocity profiles" (`25_peluchon_2017...md`, §2.4, Prop. 1 discussion).

### Extreme impedance contrast (air–water, Z-ratio ≈ 3600)

The Z-weighted closure *is* the exact linearized two-material acoustic Riemann solution — transmission/reflection
coefficients at the face are built in, so the closure degrades gracefully rather than averaging impedances.
Peluchon validates on molten-metal/gas with ρ-ratio ~10³ and stiffened gas P∞ = O(10⁸–10⁹) at large acoustic CFL
(ibid., numerical section; see also `../papers/peluchon_alternatives_2024_2025.md` §1.2: "SG EOS 포함 stiff 조건
stable (ρ₁=50, ρ₂=1000, P∞=6e8)"). Asymmetric slopes `r₀ = Z_R/Z_L` are exactly the recommended choice for large
contrast (Eq. (27)–(28)).

### Low-Mach accuracy

Inside the acoustic block, c-scaled dissipation is the point (the block exists to damp acoustics). One caution for
the *material-flux* side: the `Z_L Z_R Δu` term in `p̂_f` is the classical low-Mach over-dissipation (pressure flux
scale ρcΔu). The CGK16/Rieper fix multiplies **only the Δu term of p̂_f** by `θ_f = min(1, max(|u_L|,|u_R|)/c_f)`
(`25_peluchon_2017...md`, Introduction, refs [CGK16, Rie11]) — θ is derived from the local Mach number, so still
no user constant. Do NOT scale the Δp term of `û_f`: that term is the checkerboard damping. This repo has already
observed the failure mode when it is absent — "Block-tridiag null-space … `p̄ = ½(p_L+p_R) − ½a(u_R−u_L)` 에서 u=0,
2Δx p-mode 시 상쇄" (`../papers/peluchon_alternatives_2024_2025.md` §1.3): a p̄-only dissipation leaves the pressure
checkerboard invisible; the Δp term in `û` is the essential ingredient. Independent caution on over-centralized
low-Mach pressure fixes: Jung–Lannabi–Perrier 2025 spurious-mode analysis (listed in
`../papers/40_hybrid_allmach_summary.md`, item P4).

### CN-specific remark

CN is A-stable, not L-stable: the damped checkerboard amplification is `g = (1−x)/(1+x)`, `x ∝ cΔt/Δx`, so
`|g| → 1` (oscillatory, marginally damped) at very large acoustic CFL. With the repo default `be1` the mode is
strongly damped (`g = 1/(1+2x)`-type). If CN is retained at CFL_ac ≫ 1, expect slow (but monotone-in-norm)
checkerboard decay, never growth.

---

## B. Rhie–Chow / momentum-weighted interpolation, dt-consistent form

### Formula (adapted; 1D, applied to `û_f` in block (18))

Denner–van Wachem / ACID form (`../papers/md/[적용해볼것] ACID 2.md`, Eqs. (20)–(21); Denner, Xiao, van Wachem,
JCP 367 (2018) 192–234), 1D reduction with central cell gradients:

```
û_f = ½(u_i+u_{i+1})
      − d̂_f [ (p_{i+1}−p_i)/Δx − ½( (p_{i+1}−p_{i−1})/(2Δx) + (p_{i+2}−p_i)/(2Δx) ) ]
      + d̂_f (ρ_f^{*,n}/Δt) ( û_f^n − ½(u_i^n+u_{i+1}^n) )
    = ½(u_i+u_{i+1}) − d̂_f ( p_{i−1} − 3p_i + 3p_{i+1} − p_{i+2} )/(4Δx) + (transient term)
```

with `d̂_f = (V_P/e_P + V_Q/e_Q) / ( 2 + (ρ_f^*/Δt)(V_P/e_P + V_Q/e_Q) )` where `e_P` is the momentum-equation
central coefficient and `ρ_f^*` is the **harmonic** face density (large-density-ratio form, ACID Eq. (20)–(21);
also `../papers/1_denner.md` §3). In our pure acoustic block there is no advection/viscous coefficient, so
`V/e → ∞` and the coefficient saturates to the closed form

```
d̂_f = Δt / ρ_f^* ,   ρ_f^* = 2 ρ_i ρ_{i+1} / (ρ_i + ρ_{i+1})     →     û_f = ū_f − (Δt / (4 ρ_f^* Δx)) (p_{i−1} − 3p_i + 3p_{i+1} − p_{i+2})
```

Note the bracket is **exactly the same 4-point third difference as our current biharmonic**, but (i) applied to the
*velocity* face (so it enters the p-equation as a `c²Δt² ∂_x⁴`-type filter — the correct Brezzi–Pitkäranta-like
placement), and (ii) with coefficient `Δt/(4ρ_f^*Δx)` *derived* from the momentum matrix instead of `D/8` chosen by
hand. Effective dissipation sits at `O(Δx³ ∂³p/∂x³)` in `û_f`, i.e. `O(Δx⁴/Δt·…)` consistency error — second-order
accuracy preserved; kinetic-energy dissipation of the filter is marginal and `∝ Δx²` (ACID file, line ~236, citing
Ham & Iaccarino).

### Verdict: coefficient-free — YES, with a consistency obligation

No O(1) free constant. But the classical RC closure is Δt- and relaxation-dependent unless the transient/relaxation
consistency terms are carried: Majumdar (1988), Pascau (2011), Cubero & Fueyo (Numer. Heat Transfer B 53, 2008,
"A compact momentum interpolation procedure for unsteady flows and relaxation"), Cubero–Sánchez-Insa–Fueyo
(Comput. Chem. Eng. 62, 2014) remove the dependence; the unified/canonical modern statement is Bartholomew, Denner,
Abdol-Azis, Marquis, van Wachem, JCP 375 (2018) 177–208
(https://www.sciencedirect.com/science/article/pii/S0021999118305539). The third RHS term above (previous-step
`ϑ − ū` correction) is precisely that consistency term — omit it and an effective hidden coefficient reappears
through Δt.

### PE preservation

Safe: every correction term is a linear combination of p-*differences* (vanishes identically at uniform p) plus a
transient term that vanishes in equilibrium. The ρ-weights (harmonic ρ_f^*, 1/ρ_P, 1/ρ_Q gradient weights) multiply
p-differences only, so a ρ/α jump under uniform p produces exactly zero correction.

### Impedance contrast

The harmonic `ρ_f^*` weighting was introduced specifically for large density ratios; validated on air–water-class
interfacial problems (shock–interface, shock–bubble) in the ACID paper (`[적용해볼것] ACID 2.md`, §7; helium-bubble
Haas–Sturtevant case line ~795) and in Denner 2018's coupled all-Mach algorithm
(`../papers/md/03_denner_2018_coupled_compressible.md`, Eq. (12)–(14) — filter acts on `∂³p/∂x³`). Caveat: MWI is a
*filter*, not a Riemann solution; at Z-contrast 3600 the damping rate on the interface-adjacent faces is set by the
light phase (harmonic mean ≈ 2ρ_air), which is robust but not acoustically exact.

### Low-Mach

Excellent by construction — this family was invented for incompressible/low-Mach collocated grids; dissipation
scale `Δt/ρ` is Mach-independent, does not scale with c in the material flux. Inside the acoustic block, at acoustic
CFL ≫ 1 the damping `c²Δt²∂_x⁴` is *stronger* than the Godunov `cΔtΔx∂_x²` per step — no weakness at large Δt.

---

## C. SLAU2 / AUSM pressure flux in the implicit block

### Formulas

SLAU2 pressure flux (Kitamura & Shima, JCP 245 (2013) 62–83; SLAU: Shima & Kitamura, AIAA J. 49 (2011) 1693–1709 —
title literally "Parameter-Free, Simple, Low-Dissipation AUSM-Family Scheme for All Speeds"):

```
p̃_f = ½(p_L+p_R) + ½(P₅⁺(M_L) − P₅⁻(M_R))(p_L−p_R) + sqrt((u_L²+u_R²)/2) · (P₅⁺+P₅⁻−1) · ρ̄ c̄
P₅^±(M) = ¼(M±1)²(2∓M)  (|M|<1),  else ½(1+sign(±M))
```

SLAU2/Deng mass-flux pressure coupling (the checkerboard-relevant term; Deng–Xie–Matar–Boivin, JCP 525 (2025)
106945, arXiv:2502.02570; local `../papers/40_hybrid_allmach_summary.md` §2):

```
ṁ_f = ½ρ_L(V_L+|V|_L^+) + ½ρ_R(V_R−|V|_R^−) − θ_int (χ/c̄)(p_R−p_L),   χ = (1−M̂)²,  M̂ = min(1, |ū|/c̄)
```

Adapted to block (18), the natural transplant is only the Δp-coupling in the face velocity:
`û_f = ½(u_L+u_R) − χ/( (ρ̄ c̄) ) (p_R−p_L)/2` — which is family A's Δp term with arithmetic-mean impedance and a
Mach modulation.

### Precedent inside a semi-implicit / pressure-based operator

Deng et al. 2025 use the SLAU2 mass flux for the explicit advection sub-step and then solve an **elliptic pressure
Helmholtz equation** `(1/ρc²)(p^{n+1}−p^{**})/Δt − Δt ∇·(∇p^{n+1}/ρ) = −∇·u^{**}` for the implicit part
(`40_hybrid_allmach_summary.md` §2.3) — i.e. their implicit stencil is a compact 3-point Laplacian (family F1
structure), *not* an AUSM-dissipated stencil; the AUSM part lives in the explicit material flux. Battisti–Boscheri
likewise use **central** differences for all implicit terms (see family E below). So the literature precedent is:
AUSM/SLAU2 dissipation for the *explicit/material* side; structural (Laplacian/staggered) suppression for the
*implicit* side. This repo already adopted the Deng mass-flux term in the He2024 research solver with large measured
gains (`../papers/peluchon_alternatives_2024_2025.md` §2.3: EB4 2Δx-mode metric 9.74e-4 → 8.22e-6).

### Verdict: coefficient-free — YES (all constants are Mach-polynomial-derived), BUT

### PE preservation — FAILS strictly at moving interfaces

The third term of `p̃_f`, `sqrt(ū²)(P₅⁺+P₅⁻−1)ρ̄c̄`, does **not** vanish at uniform (p,u) when c jumps across the
face (M_L ≠ M_R ⇒ P₅⁺(M_L)+P₅⁻(M_R)−1 ≠ 0): a moving material interface at uniform pressure receives a spurious
face-pressure perturbation proportional to |u|ρ̄c̄ × (Mach-split mismatch). At u = 0 it is PE-exact. Deng et al.
handle it by an interface indicator θ_int that switches the pressure–velocity coupling off at α-jump faces
(`40_hybrid_allmach_summary.md` §2.2) — effective, but it is a detector, and under our strict rule the *pressure
flux* `p̃` is disqualified for the implicit block at interfaces. The mass-flux Δp term alone (∝ Δp) is PE-safe.

### Impedance contrast / low-Mach

Uses single arithmetic `ρ̄, c̄` — at Z-ratio 3600 the face impedance is dominated by water, mis-scaling the damping
on the air side (no Z-weighting). Low-Mach behavior is the family's strength: χ(M̂) removes c-scaled dissipation
from the material flux automatically (`χ→1` low-Mach keeps Δp coupling; `χ→0` supersonic removes it).

---

## D. Staggered / semi-staggered semi-implicit

### What they do

Dumbser & Casulli (Appl. Math. Comput. 272 (2016) 479–497; local
`../papers/md/50_dumbser_casulli_2016_semiimplicit_general_eos.md`): p at cell centers, u (and ρ, momentum) at
faces; implicit pressure gradient `(p_{i+1}−p_i)/Δx_{i+1/2}` couples nearest neighbors directly, pressure system
matrix "symmetric and at least positive semi-definite" (line ~146) — the checkerboard mode is simply not in the
kernel; no dissipation constant exists or is needed. Same structural choice: Boscheri–Pareschi's antecedents
(Tavelli–Dumbser staggered DG, `../papers/md/28_boscheri_2021_imex_allMach_navier_stokes.md` line 25:
"Staggered meshes … permitting to recover by construction the divergence free constraint"), Re et al. semi-implicit
BN (`../papers/md/08_re_2022_semiimplicit_BN.md`), Chiocchetti et al. (`../papers/md/09_chiocchetti_2023_semiimplicit_BN.md`),
and the Basilisk all-Mach VOF line (Fuster & Popinet JCP 374 (2018); Saade–Lohse–Fuster JCP 476 (2023) 111865 —
face-velocity projection; https://www.sciencedirect.com/science/article/pii/S0021999122009159).

### Verdict: coefficient-free — YES (structural)

### Cost of adoption in our collocated 1D FVM

High. Face momentum unknowns; explicit transport requires the average–advect–average-back machinery
(`50_dumbser_casulli_2016...md` lines ~303–323) which is conservative but adds its own averaging diffusion; the
face density definition `ρ_{i+1/2}` interacts with the isolated-interface PE property and with our α-advection and
THINC path (all cell-centered); every 2D/3D extension doubles the variable layout work. Dumbser–Casulli's own
motivation for face-ρ is "a simple explicit discretization … that preserves uniform pressure and velocity flows"
(line ~70) — PE is achievable but must be re-proven for the 5-equation model. Recommendation: fallback option only;
the compact-stencil trick (F1) delivers the staggered kernel property on the collocated layout — Boscheri–Pareschi
say so verbatim ("achieving the same properties typically related to the usage of staggered meshes",
`28_boscheri_2021...md` line ~306).

---

## E. Implicit operators from upwind flux Jacobians

Checked against Battisti–Boscheri (JCP 2025, "A linearly implicit shock capturing scheme for compressible two-phase
flows at all Mach numbers", `../papers/md/30_battisti_2025_linearly_implicit_twophase.md`): they do **not** upwind
the implicit part — "Central finite difference operators on Cartesian grids are adopted for the implicit terms"
(lines ~26, 806); nonlinear convective terms are explicit (upwind FV). Their checkerboard control is the compact
Lagrange stencil (→ F1). The classical density-based route (backward Euler with Roe/HLLC-linearized Jacobians,
`|A|`-dissipation inside the matrix) does damp odd–even modes automatically, but the Roe `|A|(U_R−U_L)` dissipation
acts on *conservative-variable* differences: at a material interface with uniform p, `U_R−U_L` contains
`Δ(αρ), Δ(ρE)` ≠ 0 → PE-violating by our strict rule (this is the reason our low-order flux is
`lo_flux='pe_preserving'` with no Rusanov `(U_R−U_L)` term, `docs/five_eq_IMEX_current_formulation.md` §12.2).
Family A is exactly the PE-safe two-wave member of this class (upwinding restricted to the (u,p) characteristic
variables) — use A, not full-Jacobian upwinding.

Verdict: coefficient-free YES, PE-preservation NO (except its family-A restriction). Not recommended in general
form.

---

## F. Other genuinely coefficient-free devices found

### F1. Compact-product discretization of the implicit wave operator  ★ co-primary candidate

Boscheri & Pareschi (JCP 435 (2021) 110206; local `../papers/md/28_boscheri_2021_imex_allMach_navier_stokes.md`,
Eq. (53) discussion, line ~306): all implicitly-discretized first derivatives are built from degree-2 Lagrange
polynomials on `{i−1, i, i+1}` so that the *product* operator in the pressure wave equation stays on a 3-point
stencil, instead of the naive substitution which "would involve those cells spanning [i−2 … i+2]" (line ~333) —
the wide form is 2Δx-blind, the compact form is not. Battisti–Boscheri give the explicit 1D matrix
(`30_battisti_2025...md`, Eqs. (30)–(31)):

```
Δt² ∂_x( h ∂_x q )|_i ≈ (Δt²/Δx²) [h_{i−1}, h_i, h_{i+1}] · [[3/4, −1, 1/4], [0,0,0], [1/4, −1, 3/4]] · [q_{i−1}, q_i, q_{i+1}]ᵀ
```

Row-sum check with uniform h: weights collapse to `[1, −2, 1]` — the exact compact Laplacian; the checkerboard
`q=(−1)^i` gives `−4(−1)^i ≠ 0` → controlled. Uniform-q gives identically 0 for any h → PE-safe regardless of the
face coefficient (h, ρc², …). Battisti–Boscheri prove the operator "guarantee[s] constant pressure and velocity
preservation across a contact discontinuity" (text after Eq. (31), §3.7).

Adapted to our Schur path (`schur=True`): eliminate `u^{n+1}` from block (18) and discretize the Schur complement as

```
p_i^{n+1} + θ²Δt² ρ_i c_i² · [ (p_{i+1}^{n+1}−p_i^{n+1})/ρ_{i+1/2} − (p_i^{n+1}−p_{i−1}^{n+1})/ρ_{i−1/2} ] / Δx²  =  RHS(p^n, u^*, …)
```

(3-point, θ=1/2 CN, face `1/ρ_{i+1/2}` harmonic), then update `u^{n+1} = u^* − (Δt/ρ_i)(p̂_{i+1/2}−p̂_{i−1/2})/Δx`
with central `p̂_f = ½(p_L+p_R)` — checkerboard cannot persist because the p-equation itself penalizes it; the
momentum row never generates it. This is the same structure as Deng 2025's Helmholtz step
(`40_hybrid_allmach_summary.md` §2.3, §4.2 — "Helmholtz 방정식은 elliptic → 자연스럽게 2Δx mode 감쇠") and as the
repo's own earlier diagnosis ("Helmholtz form으로 바꾸면 null-space 자동 해결", ibid.). Zero constants. The known
trade-off (Boscheri–Pareschi "Remark on the compactness", `28_boscheri_2021...md` line ~333–355): the compact
operator is not the exact discrete substitution of the momentum update, i.e. a small O(Δx²) energy-consistency
slack, proven not to break the low-Mach divergence constraint.

### F2. Flux-weighted (Roe-type) enthalpy averaging in the pressure equation

Boscheri–Dimarco–Tavelli (CMAME 374 (2021) 113602; local `../papers/md/90.md`, Eq. (49) and lines ~1355–1360):
face enthalpy `h̃_{i+1/2} = (h_i q_i/ρ_i + h_{i+1/2} q_{i+1/2}/ρ_{i+1/2}) / (q_i + q_{i+1/2})` (q = mass fluxes),
with arithmetic fallback when `|q_i+q_{i+1}| < δ = 10⁻¹²`, "adopted to avoid the so-called checkerboard effect in
the resulting equation for the pressure [8]". Coefficient-free (δ is a machine tolerance). PE risk: the weights are
flux- and h-based → at a moving interface the face enthalpy is biased toward the upwind phase; harmless for
single-phase Euler, but for our 5-eq energy row `(pu)_f` this mixes phase enthalpies with ρ-dependent weights —
requires case-by-case PE proof. Secondary interest only.

### F3. Fourth-difference with a stability-derived coefficient

Fallback that keeps the current stencil: keep `p_f = ½(p_L+p_R) − (D_f/8)(−p_{LL}+3p_L−3p_R+p_{RR})` but *derive*
`D_f` per-face from the CN amplification analysis of block (18) instead of choosing 0.02 — e.g. requiring the 2Δx
mode amplification `|g(π)| ≤ g_target(ν)` yields `D_f = f(ν_f)`, `ν_f = c_f Δt/Δx`. Family B shows what the answer
looks like when derived from momentum consistency: `D_f^{MWI} = 2 ρ̄_f c̄_f² Δt² /(ρ_f^* Δx²) ×(placement factor)` —
i.e. the "correct" D is CFL-dependent, which is exactly why a fixed 0.02 needs retuning per case. Under the strict
rule this is coefficient-free only if `g_target` is forced to a derived value (e.g. the be1 value); otherwise it is
tuning in disguise. Listed for completeness, ranked last.

### F4. Evidence that "1D central survives by accident" is fragile

Schropff et al. (JCP 547 (2026) 114545; local `../papers/md/37_paper.md`, lines ~1190–1215) use
`p_f^{n+⋆} = ½(p_i+p_{i+1})` in their semi-implicit two-phase scheme and report the checkerboard "was not
encountered", while citing "various authors suggest that the checkerboard problem may not appear in 1D because it
is dissipated by the 1D discretization, but could manifest itself in a multidimensional study" and that pressure
relaxation mitigates it. I.e. the central closure has no defense of its own — consistent with our observed need for
D > 0, and a warning for the 2D/3D extension (`nd_solver.py`).

---

## Ranked shortlist (implementable in five_eq_IMEX)

| # | Candidate | One-line discrete formula (implicit faces of block (18)) | Coeff-free | PE at interface | Air–water Z=3600 |
|---|-----------|----------------------------------------------------------|------------|-----------------|-------------------|
| 1 | **A. Acoustic-Riemann closure as the implicit face state** | `û_f = (Z_L u_L + Z_R u_R − Δp)/(Z_L+Z_R)`, `p̂_f = (Z_R p_L + Z_L p_R − Z_L Z_R Δu)/(Z_L+Z_R)`, Z frozen at tⁿ | yes | exact | exact linear acoustics (best) |
| 2 | **F1. Compact 3-point Schur/Helmholtz operator** | `p_i + θ²Δt²ρ_i c_i²[(p_{i+1}−p_i)/ρ_{i+1/2} − (p_i−p_{i−1})/ρ_{i−1/2}]/Δx² = RHS` | yes | exact (uniform-p annihilated for any face coeff) | good (face 1/ρ harmonic) |
| 3 | **B. dt-consistent MWI on û_f** | `û_f = ū_f − (Δt/(4ρ_f^*Δx))(p_{i−1}−3p_i+3p_{i+1}−p_{i+2}) + d̂_f(ρ_f^*/Δt)(û_f^n−ū_f^n)` | yes (derived d̂) | exact | robust (harmonic ρ*), not acoustically exact |
| 4 | **C. SLAU2 Δp mass-flux term (explicit/material side only)** | `û_f += −χ(M̂)Δp/(2ρ̄c̄)`, `χ=(1−M̂)²` | yes | mass-flux term exact; SLAU2 `p̃` NOT (|u|ρ̄c̄ term) | mediocre (arithmetic ρ̄c̄) |
| 5 | **D. Staggered (u at faces)** | `u_{i+1/2}`: `ρ_{i+1/2}(u^{n+1}−u^*)/Δt + (p_{i+1}^{n+1}−p_i^{n+1})/Δx = 0` | yes (structural) | must re-prove for 5-eq | proven class (Dumbser–Casulli) | 

**Recommendation.** Implement 1 and 2 (they compose: 1 supplies the face closure, 2 fixes the operator structure on
the Schur path; either alone removes D). Candidate 1 is the shortest path: the solver already computes the exact
face states under `imp_dissipation_form='acoustic_riemann'` (`docs/five_eq_IMEX_current_formulation.md` §11.2) —
the change is to make those states the *implicit* closure of block (18) with frozen Z (Peluchon IM1 / Tallois CN,
which is literally our time scheme) and to delete the residual `w = imp_dissipation` post-smoothing pass, which is
the last surviving knob. Keep the CGK/Rieper θ_f = min(1, M_loc) *only* on the `Z_L Z_R Δu` term of `p̂_f` if
low-Mach material-flux over-dissipation appears; never on the Δp term of `û_f` (repo's own 2Δx null-space
post-mortem, `../papers/peluchon_alternatives_2024_2025.md` §1.3). For second-order faces keep the Tallois
defect-correction: first-order Riemann matrix, high-order RHS (`29_tallois_2022...md` Eq. (3.18)).

## Citations (key)

- Peluchon, Gallice, Mieussens, JCP 339 (2017) 328–355 — `../papers/md/25_peluchon_2017_imex_acoustic_transport.md`
- Tallois, Peluchon, Villedieu, Comput. Fluids 244 (2022) — `../papers/md/29_tallois_2022_2nd_order_imex_twophase.md`
- Chalons, Girardin, Kokh (CGK16/17), Lagrange-projection all-regime schemes — cited via Peluchon file, Introduction
- Denner, Xiao, van Wachem, JCP 367 (2018) 192–234 (ACID) — `../papers/md/[적용해볼것] ACID 2.md`, `../papers/1_denner.md`
- Denner, JCP-adjacent linearisation study (2018) — `../papers/md/03_denner_2018_coupled_compressible.md` (MWI Eq. (12)–(14), ∂³p filter)
- Bartholomew, Denner, Abdol-Azis, Marquis, van Wachem, JCP 375 (2018) 177–208 — https://www.sciencedirect.com/science/article/pii/S0021999118305539
- Cubero & Fueyo, Numer. Heat Transfer B 53 (2008); Cubero, Sánchez-Insa, Fueyo, Comput. Chem. Eng. 62 (2014) — https://www.researchgate.net/publication/239394732 , https://www.sciencedirect.com/science/article/abs/pii/S0098135413003694
- Shima & Kitamura, AIAA J. 49 (2011) 1693–1709 (SLAU); Kitamura & Shima, JCP 245 (2013) 62–83 (SLAU2)
- Deng, Xie, Matar, Boivin, JCP 525 (2025) 106945, arXiv:2502.02570 — `../papers/40_hybrid_allmach_summary.md`
- Boscheri & Pareschi, JCP 435 (2021) 110206 — `../papers/md/28_boscheri_2021_imex_allMach_navier_stokes.md`
- Battisti & Boscheri, JCP (2025), linearly implicit two-phase — `../papers/md/30_battisti_2025_linearly_implicit_twophase.md`
- Boscheri, Dimarco, Tavelli, CMAME 374 (2021) 113602 — `../papers/md/90.md`
- Dumbser & Casulli, Appl. Math. Comput. 272 (2016) 479–497 — `../papers/md/50_dumbser_casulli_2016_semiimplicit_general_eos.md`
- Re et al. (2022), Chiocchetti et al. (2023) semi-implicit BN, staggered — `../papers/md/08_re_2022_semiimplicit_BN.md`, `../papers/md/09_chiocchetti_2023_semiimplicit_BN.md`
- Fuster & Popinet, JCP 374 (2018); Saade, Lohse, Fuster, JCP 476 (2023) 111865 — https://www.sciencedirect.com/science/article/pii/S0021999122009159
- Dellacherie, Proc. ALGORITMY 2009, 71–80 — via `../papers/md/37_paper.md` refs, `../papers/md/42_hope_collins_2022_lowmach_artificial_diffusion.md`
- Hope-Collins & di Mare (2022), artificial-diffusion low-Mach analysis — `../papers/md/42_hope_collins_2022_lowmach_artificial_diffusion.md`
- Schropff et al., JCP 547 (2026) 114545 — `../papers/md/37_paper.md`
- Repo-internal precedent: `../papers/peluchon_alternatives_2024_2025.md` (IM1 null-space post-mortem, SLAU2 adoption), `../papers/40_hybrid_allmach_summary.md` (Helmholtz reform plan)
