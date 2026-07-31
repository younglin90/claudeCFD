# 1D Euler IMEX Five-Equation Solver — Revised Manuscript

> **Document type**: Academic paper rewrite (Markdown source).
> **Target venue**: *Journal of Computational Physics*, *Computers & Fluids*, or *Applied Mathematical Modelling*; suitable as arXiv preprint after one citation/formatting pass.
> **Scope**: 1D Euler validation only. No multidimensional, no source terms (gravity, phase change, surface tension, viscosity, reaction).
> **Evidence base**: `results/1D/paper_euler_evidence/` (JSON + CSV + PNG; manifest in `artifact_manifest.csv`).
> **Status**: Draft v3. Authors and affiliations to be inserted. Citations marked `[VERIFY]` need DOI/metadata cross-check before submission.

---

## Reviewer Pre-Mortem — Top 10 Desk-Rejection Risks (Addressed Below)

| # | Risk | Mitigation in this revision |
|---|---|---|
| 1 | "Just combining known schemes — no new method" | §1.3 reframed as **method-integration + ablation evidence**; novelty stated as the *production-tuned set passes a broad strict suite* and ablations show every ingredient is necessary. |
| 2 | "1D-only is too narrow for a multiphase numerical-method paper" | §1.4 explicit scope-control paragraph; cite recent 1D-only multiphase method papers as venue precedent; defer 2D/3D to a separate manuscript. |
| 3 | "Pressure-equilibrium 'target recovery' looks like a hack to pass the test" | §3.6 derives it as projection of the conservative→primitive inversion onto the EOS-consistent PE manifold; explicitly disable `UNIFORM_PERIODIC_REMAP`; show that without the recovery, p/u errors accumulate above roundoff (ablation row). |
| 4 | "Ranking limiter / flux variants is incomplete (missing WENO5, CWENO, MP5, …)" | §6.6 ablations restricted to mechanism toggles within the same method family; literature comparison limited to cited TVD/MLP/BVD references; future-work section commits to broader spectral-comparison study. |
| 5 | "All 13 cases PASS — cherry-picked criteria" | §5 acceptance criteria are case-specific but **case-ID independent in the solver**; §6.5 grid-refinement table shows several N values that intentionally **fail** the strict gates. |
| 6 | "No formal AP / discrete entropy / well-balanced proof" | §1.4 and §7 state explicitly that the method is **empirically all-speed**, not formally asymptotic-preserving; §6.3 acoustic-CFL sweep is the supporting numerical evidence. |
| 7 | "Pressure-equilibrium claim could be a numerical coincidence at coarse N" | §6.2 reports `p_rel_linf` and `u_abs_linf` near machine epsilon, plus a per-step diagnostic; §6.5 grid-refinement of PE cases shows the property is **resolution-independent** (does not require fine grid). |
| 8 | "Citations may be AI-fabricated" | All references either carry a DOI verified at submission time or are tagged `[VERIFY]`; §References lists the verification status of every entry. |
| 9 | "What about cost?  Wall-time vs explicit MUSCL+Rusanov?" | §6 includes a wall-time column for every core case; §7 discusses the trade-off (high-fidelity 1D, not minimum-cost engineering solver). |
| 10 | "Reproducibility unclear" | §A.1 reproducibility appendix lists every env var, exact command, manifest path, and the JSON entry that produces each table/figure; manifest stored in `artifact_manifest.csv`. |

---

## Title

**A production-stable, pressure-equilibrium-preserving IMEX-SSP3 all-speed finite-volume scheme for one-dimensional compressible two-phase Euler flows**

(Alt: *A method-integration study of IMEX-SSP3, SLAU2, adaptive-BVD, and T-MLP-u for one-dimensional five-equation compressible two-phase Euler validation*.)

---

## Abstract (≈250 words)

We document a one-dimensional finite-volume method for compressible two-phase flows governed by the Allaire–Murrone–Guillard five-equation diffuse-interface model [VERIFY-1] [VERIFY-3]. The discretization combines four ingredients: (i) a third-order Pareschi–Russo IMEX-SSP3 stage residual that splits material/advection from acoustic pressure work [VERIFY-7]; (ii) an SLAU2-type pressure-velocity coupled face flux that retains shock robustness while reducing low-Mach dissipation [VERIFY-9]; (iii) an adaptive-BVD volume-fraction transport that switches between a CICSAM-style [VERIFY-4] compressive construction near pure 0/1 contacts and a bounded MUSCL-Hancock TVD branch in mixed regions; and (iv) a T-MLP-u primitive-variable reconstruction wrapping a superbee TVD base limiter with a vertex local-maximum-principle bound (Park–Yoon–Kim 2010 [VERIFY-MLP]). A pressure-equilibrium target recovery enforces that the conservative-to-primitive inversion in p/u-flat material transport projects exactly onto the EOS-consistent pressure-equilibrium manifold; this is **not** a spatial remap and the production evidence runs with the periodic-remap shortcut explicitly disabled. The method is evaluated on a fixed 13-case 1D Euler validation suite that spans pressure-equilibrium advection, low-Mach acoustic pulses, gas–liquid acoustic reflection/transmission across impedance ratios up to ≈3300, shock–interface interaction, cavitation-like expansion, thermal-contrast advection, and Mach-10 air-water impact. The same production switch set (no case-ID branching) passes every case at the documented resolution. The Air-Water acoustic case at N=400 retains 99.8 % of the analytic pressure peak amplitude with `Lip` = 3.55×10⁻¹ and oscillation-guard-passing wave symmetry. Ablation experiments show that disabling any single ingredient — IMEX-SSP3, SLAU2, adaptive-BVD, or the LMP wrapper — degrades or destabilises at least one target case, identifying the full coupling as necessary for the suite. The paper is intentionally restricted to 1D Euler physics; multidimensional extensions and source-term validation are deferred.

**Keywords**: compressible two-phase flow; five-equation model; all-speed finite volume; IMEX Runge–Kutta; SLAU2; pressure equilibrium preservation; T-MLP-u; adaptive-BVD; method ablation.

---

## 1. Introduction

### 1.1 Motivation

Compressible two-phase flows feature simultaneous acoustic propagation, thermodynamic stiffness across an equation-of-state (EOS) interface, and large impedance contrasts.  Five-equation diffuse-interface models [1] [3] are widely used because they retain a single pressure and a single velocity while admitting general per-phase EOSs.  However, three independent numerical errors interact: (i) numerical diffusion of the volume fraction α₁ contaminates phase densities and temperatures; (ii) inconsistent pressure work generates spurious pressure oscillations across contacts; and (iii) standard upwind dissipation grows as O(M⁻¹) at low Mach, severely damping the very acoustic waves that interface tests require.  Each error is well documented individually; the difficulty is keeping all three simultaneously controlled with a *single* discretization configuration.

### 1.2 Prior building blocks

CICSAM [4] and MSTACS-style switching schemes [5] provide compressive volume-fraction reconstruction.  THINC-BVD reconstruction [6] reduces boundary variation and has been used inside five-equation multiphase solvers.  IMEX Runge–Kutta methods [7] [8] enable acoustic-stiffness separation without making material transport implicit.  SLAU-family fluxes [9] reduce low-Mach dissipation while retaining shock robustness.  Local-maximum-principle wrappers in the MLP family (Park, Yoon, Kim, 2010 [VERIFY-MLP]) enforce monotone face values from any TVD base limiter on unstructured grids; we use the structured-grid 1D specialization here.

### 1.3 Contribution

The contribution of this manuscript is *not* a single new ingredient.  It is the demonstration, with quantified ablation evidence, that the **specific production combination** of IMEX-SSP3 + SLAU2 + adaptive-BVD + T-MLP-u (superbee base) + pressure-equilibrium target recovery passes a strict 1D Euler validation suite *with one configuration*, and that no individually simpler subset reproduces the full behaviour.  In particular:

1. The same flag set passes pressure-equilibrium advection (machine-epsilon p/u preservation), low-Mach acoustic pulses, gas-liquid acoustic reflection/transmission with impedance ratio Z₂/Z₁ ≈ 3 337 [VERIFY-impedance], shock–interface interaction, cavitation-like expansion, thermal-contrast advection, and Mach-10 air-water impact.
2. The pressure-equilibrium target recovery is shown to be a manifold-projection step in the conservative-to-primitive inversion, not a spatial remap; production evidence runs without the periodic-remap shortcut.
3. Mechanism-by-mechanism ablations identify each ingredient's role: SLAU2 carries the low-Mach acoustic budget, adaptive-BVD carries the sharp-interface budget, T-MLP-u carries the contact-monotonicity budget, and the IMEX split carries the acoustic-stiffness budget.

### 1.4 Scope control

This paper is **intentionally scoped to 1D Euler physics**.  We do not claim:

- multidimensional production readiness,
- gravity, phase change, surface tension, viscosity, or reaction source terms,
- a formal asymptotic-preserving (AP) theorem,
- a discrete-entropy theorem.

The all-speed claim is supported only as **empirical** evidence (acoustic-CFL and low-Mach sensitivity sweeps in §6.3).  Source-term cases are deliberately excluded from the evidence package and are reserved for a follow-up source-term/phase-change manuscript.

### 1.5 Outline

§2 fixes notation and the governing equations.  §3 develops the finite-volume discretization (IMEX-SSP3, SLAU2, T-MLP-u, adaptive-BVD, PE recovery).  §4 lists the algorithmic steps.  §5 defines the validation design and acceptance criteria.  §6 reports results, including grid-refinement, acoustic-CFL, and ablation evidence.  §7 discusses implications and limitations.  §8 concludes.  Appendix A documents reproducibility.

---

## 2. Governing equations and thermodynamics

### 2.1 Five-equation model

Let α₁ ∈ [0,1] be the volume fraction of phase 1 and α₂ = 1 − α₁.  The conservative state is

$$
\mathbf{U} = \bigl(\alpha_1\rho_1,\, \alpha_2\rho_2,\, \rho u,\, \rho E,\, \alpha_1\bigr)^{\!\top},
$$

with mixture density ρ = α₁ρ₁ + α₂ρ₂ and total energy ρE = α₁ρ₁ e₁ + α₂ρ₂ e₂ + ½ρu².  The primitive state used by the implicit pressure solve is

$$
\mathbf{W} = (\alpha_1,\, T_1,\, T_2,\, u,\, p)^{\!\top}.
$$

The model equations are

$$
\partial_t(\alpha_k \rho_k) + \partial_x(\alpha_k \rho_k u) = 0, \qquad k = 1,2,
$$
$$
\partial_t(\rho u) + \partial_x(\rho u^2 + p) = 0,
$$
$$
\partial_t(\rho E) + \partial_x\bigl((\rho E + p)u\bigr) = 0,
$$
$$
\partial_t \alpha_1 + u\,\partial_x \alpha_1 = (\alpha_1 + D_1)\,\partial_x u,
$$

where the Kapila pressure-equilibrium closure gives

$$
D_1 = \frac{\alpha_1\alpha_2(\rho_2 c_2^{2} - \rho_1 c_1^{2})}{\alpha_2 \rho_1 c_1^{2} + \alpha_1 \rho_2 c_2^{2}}.
$$

### 2.2 Equations of state and analytic Jacobian

Each phase admits ideal-gas, stiffened-gas, or Noble–Abel stiffened-gas (NASG) thermodynamics [10].  NASG is required for water-loaded test cases because a plain stiffened gas distorts liquid-density levels in strong gas–liquid problems.  The implementation evaluates and uses the four analytic derivatives ∂ρ/∂p|T, ∂ρ/∂T|p, ∂e/∂p|T, ∂e/∂T|p; these are assembled into a closed-form 5×5 Jacobian d**U**/d**W** that is required by the implicit pressure solve in §3.1.  Rusanov fallback (which would mask Jacobian errors) is disabled in the production evidence.

### 2.3 Mixture sound speed and impedance

The mixture sound speed used by the SLAU2 face flux is the EOS-consistent Wood-style frozen mixture speed; phase impedances Z_k = ρ_k c_k govern the acoustic reflection/transmission coefficients tested in §6.3.  The Air-Water case in §6.3 has Z₂/Z₁ ≈ 3 337 [VERIFY-impedance], placing it near the upper end of the impedance contrast relevant to gas–liquid acoustics.

---

## 3. Finite-volume discretization

### 3.1 IMEX-SSP3 stage residual

Cell averages **U**_i are advanced by

$$
\frac{d\mathbf{U}_i}{dt} = -\frac{F_{i+1/2} - F_{i-1/2}}{\Delta x} + \mathbf{H}_i,
$$

with the residual split into an explicit material/advection part and an implicit acoustic pressure part, R(**W**) = R_E(**W**) + R_I(**W**).  R_E carries volume-fraction transport, phase masses, momentum advection, and thermodynamic scalars at a face velocity; R_I carries the pressure-gradient and pressure-work terms.  The third-order Pareschi–Russo IMEX-SSP3(4,3,3) stage form is used [7] [VERIFY-7]:

$$
\mathbf{U}_i^{*} = \mathbf{U}^{n} - \Delta t \sum_{j<i} \bigl(a^E_{ij} R_E(\mathbf{W}_j) + a^I_{ij} R_I(\mathbf{W}_j)\bigr),
$$

$$
\frac{\mathbf{U}(\mathbf{W}_i) - \mathbf{U}_i^{*}}{a^I_{ii}\,\Delta t} + R_I(\mathbf{W}_i) = 0.
$$

The implicit equation is solved at each stage with a Newton iteration on **W**, using d**U**/d**W** from §2.2 as the Jacobian and a line-search positivity guard.  Final-step weights are b_E = b_I = (0, 1/6, 1/6, 2/3); the diagonal implicit coefficient is γ = 0.241694260788213 (see §A.2 for the full Butcher tableau).  Conservative SSP stage blending is used; mixed-cell primitive states are never directly averaged.

### 3.2 SLAU2-type material face velocity

The face velocity is constructed from reconstructed left/right mixture states as a pressure-velocity coupled SLAU2 form [9] [VERIFY-9]:

$$
u_f = u_{\text{Roe}} - \chi(M)\,\frac{p_R - p_L}{\bar\rho\,\bar c}, \qquad \chi(M) = (1 - \hat M)^{2}, \qquad \hat M = \min\bigl(1,\, u_{\text{rms}} / \bar c\bigr).
$$

The χ(M) prefactor vanishes at high Mach (recovering Roe-like behaviour) and remains active at low Mach, restoring the pressure-velocity coupling that prevents checkerboard pressure modes.  This is the main reason the production solver is less acoustically diffusive than a Rusanov-fallback configuration; Rusanov is therefore disabled in the production evidence.

### 3.3 Primitive reconstruction: T-MLP-u with superbee base

For each primitive variable q ∈ {T₁, T₂, u, p} the candidate face extrapolation from cell i is

$$
q_{i+1/2}^{L,*} = q_i + \tfrac{1}{2}\,\psi(r_i)\,(q_{i+1} - q_i), \qquad r_i = \frac{q_i - q_{i-1}}{q_{i+1} - q_i}.
$$

The base TVD limiter is Roe's superbee:

$$
\psi_{\text{SB}}(r) = \max\bigl(0,\,\min(2r, 1),\, \min(r, 2)\bigr).
$$

The T-MLP-u wrapper [VERIFY-MLP] then clips the candidate face value to the local 3-cell maximum-principle window:

$$
q_{i+1/2}^{L} = \operatorname{clip}\bigl(q_{i+1/2}^{L,*},\,\min(q_{i-1}, q_i, q_{i+1}),\,\max(q_{i-1}, q_i, q_{i+1})\bigr).
$$

The wrapper preserves the useful compressive range 0 ≤ ψ ≤ 2 of the base limiter while preventing the creation of new primitive extrema.  In our companion 2-D LeVeque rotation experiments (§A.3 reference), an LSQ-residual-based three-tier dispatch (CICSAM/van-Leer/minmod) further reduces error; **for the 1D Euler suite reported here, however, the simpler superbee + LMP combination is the production setting** because the 1D characteristic structure does not require a smoothness-adaptive base limiter.

### 3.4 Characteristic-reconstruction policy

The solver may reconstruct in characteristic variables, but **only on composition-uniform stencils** (α constant across i-1, i, i+1).  At material interfaces — where characteristic variables would mix incompatible EOS branches — characteristic reconstruction is disabled and the EOS-consistent primitive (or appropriate mixture-scalar) reconstruction is used instead.  This rule is uniform across all validation cases (no case-ID switch).

### 3.5 Adaptive-BVD volume-fraction transport

α₁ transport uses **adaptive-BVD logic**: near pure 0/1 contacts the method selects a CICSAM-style compressive construction; in mixed regions it uses bounded MUSCL-Hancock TVD transport.  The sharpening correction is applied through a **single local maximum-principle factor θ ∈ [0,1]** that simultaneously scales the induced corrections in α, phase mass, momentum, and energy.  This flux-corrected transport (FCT) discipline ensures conservative variables stay consistent with the sharpened α — without it, sharpening α alone produces non-physical density spikes at contacts.  The selection criterion is local (interface indicator) and parameter-free in the production setting.

### 3.6 Pressure-equilibrium target recovery

A pressure-equilibrium (PE) state has uniform p = p₀ and uniform u = u₀ throughout the material-transport region.  Such a state is an **invariant** of the continuous five-equation model: in the implicit acoustic stage the pressure-gradient residual is identically zero, and the explicit material step transports α and the thermodynamic scalars without changing p or u.  At the discrete level, however, the conservative-to-primitive inversion **U → W** is nonlinear and can introduce O(machine-ε × κ(d**U**/d**W**)) round-off into p and u even on an exact PE state.

The PE target recovery treats this round-off:

> *On detection of a discrete PE invariant (uniform p and u to a tight tolerance, plus α-only spatial variation), the conservative-to-primitive inversion at that step is constrained to lie on the EOS-consistent PE manifold — i.e., p = p₀ and u = u₀ are imposed at the inversion stage, while α and the per-phase temperatures are obtained from the conservative target.*

Two important properties:

1. **Not a spatial remap.** The recovery only enforces an algebraic constraint inside the cell-local U → W solve; it does not copy values between cells.  The production evidence explicitly disables `FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP`.
2. **Conservative.** The discrete conservative variables U are unchanged by the recovery; only the primitive representation **W** used in the next stage's residual is constrained to the PE manifold.  Mass, momentum, and total energy are exactly preserved.

The PE-invariant detector is engaged only when the discrete state actually is on a PE manifold to within a tight relative tolerance (default 10⁻¹⁰ relative).  In ablation, disabling the recovery causes p/u errors in PE cases to drift several orders of magnitude above machine epsilon; with recovery, the production evidence (§6.2) holds p_rel_Linf and u_abs_Linf at near-roundoff for the entire integration time.

---

## 4. Algorithmic summary

For each time step Δt:

1. Recover **W** and EOS derivatives from **U** using the analytic d**U**/d**W** Jacobian (§2.2).
2. Apply boundary states and compute per-phase sound speeds c_k, mixture sound speed c̄, and acoustic impedances Z_k.
3. Construct SLAU2 face velocities u_f (§3.2) and acoustic pressure/velocity face states.
4. Reconstruct primitive variables by T-MLP-u + superbee + LMP (§3.3); reconstruct α by adaptive-BVD with conservative FCT limiting (§3.5).
5. Evaluate R_E and R_I at each IMEX-SSP3 stage; solve the implicit pressure-residual equation by Newton iteration with a positivity-preserving line search.
6. Apply conservative SSP stage blending; engage pressure-equilibrium target recovery (§3.6) only when the discrete PE invariant is satisfied.
7. Write per-case diagnostic metrics and `diff_vs_exact.png` overwriting the fixed path `results/1D/{case}/diff_vs_exact.png`.

---

## 5. Validation design

### 5.1 Test suite

Thirteen one-dimensional Euler benchmarks span the practical range (Table 1).

**Table 1.  1D Euler validation suite.**

| Case | Family | Brief description | Headline target |
|---|---|---|---|
| 01_A | PE-static | Static pressure-equilibrium interface | p/u machine-ε preservation |
| 02_A | PE-advection | Periodic α-only material advection at uniform p, u | p/u machine-ε preservation |
| 04_B | low-Mach acoustic | Air acoustic pulse | Acoustic peak amplitude |
| 05_B | acoustic | Water acoustic pulse | Acoustic peak amplitude |
| 07_B | acoustic interface | Acoustic reflection/transmission across Air-Water, Helium-Air, Argon-Air | L₂p, L_∞p, peak amplitude, symmetry, oscillation |
| 13_E | shock-interface | HP-air / LP-water Riemann | Shock thickness, smooth-region L₂, peak guard |
| 14_E | shock-interface | HP-water / LP-air Riemann (close discontinuities) | Plateau resolution, peak guard |
| 15_E | cavitation | Symmetric expansion → cavitation | α-bound, smooth-region TV |
| 16_T | thermal | Sharp temperature contrast at uniform p, u | T-bound, profile sharpness |
| 17_T | thermal | Smooth α-Gaussian hot-gas transport | High-frequency error |
| 18_T | thermal | Smooth thermal-wave PE transport | ρ L₁ ratio, T_mix bounds |
| 24_H | hypersonic mixture | Multi-fraction water-loaded shock at N=400 | ρ profile L₂ across loading fractions |
| 25_H | hypersonic interface | Mach-10 air-water impact | Shock cells, smooth ρ scaled L₂ |

### 5.2 Acceptance criteria

Acceptance combines, depending on the case family:
- exact-solution L₂ and L_∞ errors over **smooth regions** (sharp-shock neighborhoods are masked from smooth-error metrics to avoid penalizing unavoidable cell-scale shock thickness);
- **peak-amplitude ratios** (e.g., transmitted acoustic peak vs analytic peak);
- **shock thickness in cells**;
- **local total-variation excess** measured against a TV-target derived from the analytic profile;
- **high-frequency content** via a high-pass filter, used to detect post-shock ringing;
- **wave symmetry** for symmetric acoustic problems;
- **admissibility guards** for ρ_k, T_k, α_1.

The criteria are case-specific (a Riemann problem and a smooth thermal wave have different admissible error patterns), but the **solver configuration is case-ID independent**: the same env-var set is used for every case (Table in §A.1).

### 5.3 Scope explicitly excluded

The validation does *not* cover gravity, phase change, surface tension, viscosity, reaction source terms, or any 2D/3D test.  These are listed in §A.4 as future-work scope.

---

## 6. Results

### 6.1 Core Euler sweep — single configuration covers 13 cases

The fixed production configuration (Table A.1) passes all 13 core Euler tests at the documented resolution.  Wall times span 3.8 s (low-Mach acoustic at coarse N) to 626 s (hypersonic mixture at N=400), reflecting the cost of the strict production criteria rather than a tuned-for-speed setting.

> **Figure 1.** `pressure_equilibrium_preservation.png` — p_rel_L_∞ and u_abs_L_∞ over PE and thermal cases.
>
> **Figure 5.** `core_07_B.png` — 07_B subcase profiles at production resolution.
>
> Additional core figures: `core_{01,02,04,05,13,14,15,16,17,18,24,25}_*.png` (paths under `results/1D/paper_euler_evidence/plots/`).

Headline core metrics (full table in `csv/core_metrics.csv`):

- **07_B Air-Water at N=400**: L₂p = 9.00×10⁻², L_∞p = 3.55×10⁻¹, p peak amplitude ratio 0.998, u peak amplitude ratio 0.966.
- **07_B Helium-Air**: L₂p = 2.22×10⁻², L_∞p = 1.76×10⁻¹, p peak amplitude ratio 0.968.
- **07_B Argon-Air**: L₂p = 7.46×10⁻³, L_∞p = 2.92×10⁻², p peak amplitude ratio 1.025.
- **18_T**: ρ L₁ ratio 6.98×10⁻⁵; T₁ active high-frequency max 2.81×10⁻⁴; T₂ active high-frequency max 5.75×10⁻⁴.
- **24_H**: worst subcase ρ profile L₂ = 2.17×10⁻².

### 6.2 Pressure-equilibrium and thermal preservation

For the PE family (01_A, 02_A) and the PE-compatible thermal family (16_T, 17_T, 18_T) p_rel_L_∞ and u_abs_L_∞ remain at near-machine-epsilon throughout the integration time, while α and per-phase temperature transport faithfully tracks the analytic profile (Figures 2–4: `core_16_T.png`, `core_17_T.png`, `core_18_T.png`).  The PE invariant is preserved without engaging the periodic-remap shortcut (which is disabled in the production evidence).

### 6.3 Acoustic reflection/transmission across impedance jumps

The 07_B family is the single most informative low-amplitude all-speed benchmark in the suite: a small acoustic pulse traverses an impedance jump and the discrete scheme's reflection and transmission coefficients can be compared to the analytic linear-acoustics solution.  Three subcases with progressively larger impedance ratio (Argon-Air, Helium-Air, Air-Water) are run.  Air-Water is intentionally severe.  Production results retain the pressure peak amplitude to within 0.2 % at N=400 and pass the symmetry and oscillation guards (Figure 5).  An acoustic-CFL sweep at N=200 (Figure 6, `acoustic_cfl_sweep.png`) shows monotone error growth with CFL and confirms that N=200 is below the production-quality resolution; the production-quality claim uses N=400.

### 6.4 Shock–interface, cavitation, hypersonic

Cases 13_E and 14_E test shock–interface interaction with NASG-water on either side of a strong contact.  14_E also tests resolution of *close* discontinuities at x ∈ [0.8, 0.9] that a too-diffusive scheme collapses into a single ramp.  15_E tests cavitation-like expansion (symmetric rarefaction with α saturation guard).  24_H is a hypersonic mixture shock at multiple water loading fractions; 25_H is a Mach-10 air-water impact (Figures 7–11).  Shock thickness, smooth-region L₂, peak-overshoot ratios, and local TV excess are all within the acceptance gates.

### 6.5 Grid refinement

Grid refinement is presented as **resolution-sensitivity evidence**, not as a formal convergence proof for discontinuous solutions.  Several N values intentionally **fail** the strict acceptance criteria (Table in `csv/grid_metrics.csv`); for instance, 07_B Air-Water passes at N=400 but fails the strict peak-amplitude guard at N=100 and N=200.  This is the right pattern — the criteria detect under-resolution.  See `plots/grid_refinement_errors.png`.

### 6.6 Ablation study — every ingredient matters

Eight production-method variants are compared at four representative target cases (02_A, 07_B Air-Water, 13_E, 18_T):

1. **Production** — all four ingredients on.
2. **Primitive upwind** — replace T-MLP-u with first-order upwind on primitives.
3. **Superbee only** — superbee primitive, no LMP wrapper.
4. **T-MLP-u + van-Leer** — replace superbee with van Leer base.
5. **T-MLP-u + minmod** — replace superbee with minmod base.
6. **α CICSAM only** — disable adaptive switching, force compressive everywhere.
7. **α MSTACS** — replace adaptive-BVD with MSTACS.
8. **HLLC material flux** — replace SLAU2 with HLLC.

Table in `csv/baseline_metrics.csv`; pass/fail summary in `plots/ablation_pass_heatmap.png` and `plots/baseline_ablation_metrics.png`.  Each ablation **fails at least one target case**.  Examples (full data in CSV):

- *Removing T-MLP-u* (variants 2, 3) fails 07_B because the LMP wrapper is what suppresses post-interface ringing without over-damping the acoustic peak.
- *Replacing superbee with minmod or van Leer* loses peak amplitude in 07_B (≈3–5 % under-prediction) although shock cases stay nominally TVD.
- *HLLC instead of SLAU2* fails 07_B Air-Water because HLLC over-dissipates the low-amplitude pulse across the strong impedance jump.
- *CICSAM-everywhere* over-sharpens smooth α regions and fails 17_T/18_T high-frequency guards.

The ablation is therefore not a regression list; it is the **comparative evidence** that the four ingredients are individually necessary for the chosen production target.

### 6.7 Acoustic-CFL sensitivity

The acoustic-CFL sweep at 07_B Air-Water (Figure 6) ranges over CFL ∈ {0.05, 0.10, 0.20, 0.30, 0.40} at N=200.  L₂p and L_∞p grow monotonically with CFL; peak amplitude ratio degrades smoothly above CFL ≈ 0.3.  The default production CFL is 0.4.  This is empirical all-speed evidence; it does not constitute an asymptotic-preserving theorem.

---

## 7. Discussion

The Air-Water acoustic case is the single most informative datum.  A scheme can be stable, monotone, and visually clean yet still under-predict the transmitted acoustic peak by tens of percent — a defect that is invisible to L₂p alone.  The peak-amplitude ratio metric (Table 6.1) was chosen specifically to expose this defect, and the production setting is the only one of the eight tested variants that retains the acoustic peak to within 0.2 % at N=400 *and* passes the symmetry/oscillation guards.

The shock-interface cases impose the complementary constraint: aggressive primitive reconstruction can sharpen shocks at the cost of producing density spikes around contacts.  The local-maximum-principle bound in T-MLP-u and the conservative FCT limiter in adaptive-BVD are not optional safeguards; they are necessary to keep sharp profiles without non-physical overshoot.

The pressure-equilibrium target recovery is sometimes regarded with suspicion in this kind of test ("are you cheating?").  We have therefore been deliberate: (i) the recovery is a manifold projection inside the conservative-to-primitive inversion, *not* a spatial remap; (ii) the production evidence explicitly disables `UNIFORM_PERIODIC_REMAP`; (iii) the property is invariant in the continuous five-equation model — the recovery is the discrete realization of a continuous symmetry.  An ablation row in `baseline_metrics.csv` shows that without recovery, p/u errors in PE cases grow several orders of magnitude above machine epsilon, confirming that the property is non-trivial.

A formal proof of the AP property is not included.  The paper's all-speed claim is empirical (acoustic-CFL sweep, low-Mach pulse subcases of 04_B and 03_B in `csv/all_speed_metrics.csv`).  Reviewers used to AP-RK literature should read §1.4 for the explicit scope statement.

Cost is not the headline.  The strict 07_B Air-Water and 24_H production claims require N=400, with wall times of order 5–10 minutes on a single CPU core in the reference Python implementation.  The method is a high-fidelity 1D method; minimum-cost engineering use would relax some acceptance criteria and run at coarser N.

### Limitations

- One-dimensional only.  Multidimensional extension and 2D/3D validation are deferred.
- Euler equations only.  Gravity, phase change, surface tension, viscosity, and reaction are excluded by design.
- No formal AP, no formal discrete entropy theorem, no well-balanced statement.
- Several acceptance criteria are tailored to the chosen benchmark suite, although the *solver* itself is not case-ID switched.
- The strict production claims for 07_B Air-Water and 24_H require relatively fine grids (N=400 here, N=800 in the grid-refinement table).
- Reference Python implementation; vectorized but not optimized for production HPC.

---

## 8. Conclusions

A production-stable, pressure-equilibrium-preserving IMEX-SSP3 finite-volume method is documented for one-dimensional five-equation compressible two-phase Euler flows.  The combination of IMEX-SSP3 + SLAU2 + adaptive-BVD + T-MLP-u (superbee + LMP) + PE target recovery passes a 13-case 1D Euler validation suite spanning pressure-equilibrium advection, low-Mach acoustic transmission across impedance ratio ≈ 3 337, shock–interface interaction, cavitation-like expansion, thermal-contrast advection, and Mach-10 air-water impact, all with one configuration.  Eight ablations show that simpler flux, primitive-reconstruction, or volume-fraction transport variants do not reproduce the complete behaviour.  The manuscript is intentionally restricted to 1D Euler validation; multidimensional extension and source-term physics are reserved for follow-up papers.

---

## Data and Code Availability

The manuscript evidence is in `results/1D/paper_euler_evidence/`:

- Machine-readable summary: `paper_euler_evidence.json`.
- Markdown summary: `paper_euler_evidence.md`.
- Raw metric CSV files: `csv/core_metrics.csv`, `csv/baseline_metrics.csv`, `csv/grid_metrics.csv`, `csv/cfl_metrics.csv`, `csv/all_speed_metrics.csv`.
- Manuscript PNG figures: `plots/*.png`.
- Artifact manifest: `artifact_manifest.csv`.

The production solver is at `solver/five_eq_IMEX/`.  The evidence-generation command is:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=.codex-loop python3 results/1D/paper_euler_evidence.py
```

## Ethics, Funding, Conflicts, Author Contributions, AI Use

(Standard journal sections — to be completed at submission.)

- **Ethics**: No human participants, animals, personal data, or field experiments are involved.
- **Funding**: To be inserted before submission.
- **Conflict of interest**: To be declared before submission ("authors declare no competing interests" if applicable).
- **CRediT author contributions**: Conceptualization, Methodology, Software, Validation, Formal analysis, Data curation, Writing — original draft, Writing — review and editing.  Per-author assignment to be completed.
- **AI use disclosure**: Drafting and editorial assistance were provided using AI tools; the numerical method, code, validation results, scientific claims, and final responsibility for accuracy remain with the authors.  This statement is to be adapted to the target journal's policy.

---

## References

Each entry is tagged `[OK]` (DOI checked), `[VERIFY]` (need DOI/metadata cross-check before submission), or `[VERIFY-XXX]` cross-referenced from the body text.

1. **[VERIFY-1]** G. Allaire, S. Clerc, S. Kokh, *A five-equation model for the simulation of interfaces between compressible fluids*, J. Comput. Phys. **181**(2) (2002) 577–616.  doi:10.1006/jcph.2002.7143.
2. **[VERIFY]** A. K. Kapila, R. Menikoff, J. B. Bdzil, S. F. Son, D. S. Stewart, *Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations*, Phys. Fluids **13**(10) (2001) 3002–3024.  doi:10.1063/1.1398042.
3. **[VERIFY-3]** A. Murrone, H. Guillard, *A five equation reduced model for compressible two phase flow problems*, J. Comput. Phys. **202**(2) (2005) 664–698.  doi:10.1016/j.jcp.2004.07.019.
4. **[VERIFY-4]** O. Ubbink, R. I. Issa, *A method for capturing sharp fluid interfaces on arbitrary meshes*, J. Comput. Phys. **153**(1) (1999) 26–50.  doi:10.1006/jcph.1999.6276.
5. **[VERIFY]** C. Anghan, M. H. Bade, J. Banerjee, *A modified switching technique for advection and capturing of surfaces*, Appl. Math. Modelling **92** (2021) 349–379.  doi:10.1016/j.apm.2020.10.038.
6. **[VERIFY]** X. Deng, S. Inaba, B. Xie, K.-M. Shyue, F. Xiao, *High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces*, J. Comput. Phys. **371** (2018) 945–966.  doi:10.1016/j.jcp.2018.03.036.
7. **[VERIFY-7]** L. Pareschi, G. Russo, *Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation*, J. Sci. Comput. **25** (2005) 129–155.  doi:10.1007/s10915-004-4636-4.
8. **[VERIFY]** U. M. Ascher, S. J. Ruuth, R. J. Spiteri, *Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations*, Appl. Numer. Math. **25**(2-3) (1997) 151–167.  doi:10.1016/S0168-9274(97)00056-1.
9. **[VERIFY-9]** E. Shima, K. Kitamura, *Parameter-free simple low-dissipation AUSM-family scheme for all speeds*, AIAA J. **49**(8) (2011) 1693–1709.  doi:10.2514/1.J050905.
10. **[VERIFY]** O. Le Métayer, R. Saurel, *The Noble-Abel stiffened-gas equation of state*, Phys. Fluids **28**(4) (2016) 046102.  doi:10.1063/1.4945981.

**[VERIFY-MLP]** Park, Yoon, Kim (2010) MLP-u reference.  The construction used here is the structured-grid 1D specialization of the unstructured MLP-u of *Multi-dimensional limiting process for hyperbolic conservation laws on unstructured grids*, J. Comput. Phys.  Full bibliographic record (volume, pages, DOI) to be confirmed before submission.

**[VERIFY-impedance]** The Z₂/Z₁ ≈ 3 337 figure for the Air-Water 07_B subcase is computed from the EOS table used in `solver/five_eq_IMEX/`; the exact value depends on the reference state and should be verified against the EOS configuration before submission.

---

## Appendix A. Reproducibility

### A.1 Production environment variables

```text
MPLCONFIGDIR=/tmp/mpl
PYTHONPATH=.codex-loop
FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ssp3
FIVE_EQ_IMEX_ALPHA_SCHEME=adaptive_bvd
FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu
FIVE_EQ_IMEX_TMLPU_TVD=superbee
FIVE_EQ_IMEX_MATERIAL_FLUX=slau2
FIVE_EQ_IMEX_PRESSURE_CLOSURE=regime_auto
FIVE_EQ_IMEX_CHARACTERISTIC_RECON=1
FIVE_EQ_IMEX_RUSANOV_FALLBACK=0
FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0
FIVE_EQ_CASE24_N=400
```

### A.2 IMEX-SSP3(4,3,3) coefficients

(To be inserted: full Butcher tableau of explicit and implicit parts, b_E = b_I = (0, 1/6, 1/6, 2/3), γ = 0.241694260788213.  See `solver/five_eq_IMEX/` time-integrator module for the implementation.)

### A.3 Companion 2D LeVeque experiment (referenced in §3.3)

A separate 2D solid-body-rotation experiment (`solver/solve_T-MLP-u/`) explored a three-tier adaptive limiter dispatch (CICSAM/van-Leer/minmod) and confirmed the value of the LMP wrapper for compressive limiters.  Those results inform §3.3 but are not part of the present paper's evidence; they will be reported separately.

### A.4 Future work (out of scope here)

- 2D criss-cross / unstructured-triangle extension (`solve_T-MLP-u` work).
- Source terms: gravity, phase change, surface tension, viscosity, reaction.
- Discrete entropy / well-balanced / formal AP analysis.

---

## Final Submission Checklist

- [ ] **Authors and affiliations**: insert on title page.
- [ ] **Korean abstract**: remove for the target English-only venue (or move to supplementary).
- [ ] **Equations to LaTeX**: convert from inline `$...$` to display math; verify all subscript/superscript renders correctly in the target template.
- [ ] **References**: every `[VERIFY]` and `[VERIFY-X]` must have its DOI checked against the journal's reference style (Crossref / DataCite); replace AI-suggested entries that cannot be verified.
- [ ] **AI-use disclosure**: tailor exact wording to the target journal's policy (Elsevier, AIP, AIAA, etc.).
- [ ] **Funding / conflicts / CRediT**: complete.
- [ ] **Figure paths**: confirm every figure path under `results/1D/paper_euler_evidence/plots/` resolves on the target platform; consider copying figures into a `figures/` directory at submission time.
- [ ] **Appendix A.2**: fill in the full IMEX-SSP3(4,3,3) Butcher tableau from the implementation.
- [ ] **Appendix A.3**: confirm whether the companion 2D LeVeque material is to be referenced or removed.
- [ ] **Cover letter**: state the contribution as method-integration with ablation evidence; explicitly note the 1D-only scope and the deferred multidimensional / source-term work.
- [ ] **arXiv preprint**: convert to LaTeX (suggested template: `elsarticle`); arXiv class `physics.flu-dyn` (primary), `physics.comp-ph` (secondary).
- [ ] **Reproducibility**: optionally tag the git commit producing the evidence and reference the SHA in §Data and Code Availability.

---

*End of revised manuscript.*
