---
title: ""
---

**A fully-coupled pressure-based algorithm for compressible two-phase flows at all speeds: L-stable acoustic time integration, semi-Lagrangian THINC interface capturing and an exact analytical Jacobian with Noble–Abel stiffened-gas thermodynamics**

Young Lin Yoo^a,\*^

^a^ [Department, Institution, Address — to be completed]

^\*^ Corresponding author. E-mail: younglin90@gmail.com

# Abstract

A fully-coupled pressure-based finite-volume algorithm for inviscid compressible two-phase flows at all speeds is presented, building on the acoustically-conservative interface discretisation (ACID) framework of Denner et al. [J. Comput. Phys. 367 (2018) 192–234]. The governing four-equation model — mixture mass, momentum, total enthalpy and volume-fraction transport under mechanical and thermal equilibrium — is closed by the Noble–Abel stiffened-gas (NASG) equation of state, for which a complete closed-form set of thermodynamic derivatives is provided. The discretised equations are solved with a Newton method for the primitive unknowns (velocity, pressure, total enthalpy), for which we derive an exact, closed-form, block-pentadiagonal analytical Jacobian including the cross-coupling terms introduced by the momentum-weighted interpolation (MWI) of the advecting velocity. Three developments are reported and verified. First, the acoustic regime is advanced with a two-stage TR-BDF2 scheme ($\gamma = 2-\sqrt{2}$, L-stable, second order) applied to a conservative total-energy residual $\rho E = \rho h - p$, which absorbs the transient pressure-work source and removes the residual pressure wake left by a single-step second-order backward scheme, while leaving every shock-capturing (backward-Euler) computation byte-for-byte unchanged. Second, the volume fraction is transported with a THINC hyperbolic-tangent reconstruction evaluated as a conservative semi-Lagrangian face flux (the tanh profile integrated over the departure interval, verified in closed form to a relative $4\times10^{-15}$ against quadrature) and protected by a parameter-free density-monotonicity (boundary-variation-diminishing) guard, sharpening material contacts to a few cells without spurious density oscillation. Third, the entire verification programme — nineteen one-dimensional benchmarks from a machine-precision quiescent interface to a Mach 100 shock — is run with a single global parameter set (Courant number 0.45, tanh steepness $\beta = 3.5$, mathematically-fixed integrator constants), all per-case numerical coefficients and tuned dissipation multipliers having been removed. A systematic negative-result study of time-step-independent MWI dissipation is included: transient, advective, acoustic-signal-speed and linearized-Riemann forms are each shown to satisfy at most two of {per-step time-step independence, low-Mach stability, near-vacuum robustness}, so that the single-cell reflected-shock imprint remaining in the shock–interface case is a start-up feature also visible in the reference computation of Denner et al. The exact Jacobian reduces the Jacobian-assembly cost from fifteen residual evaluations to one while leaving the converged solution unchanged.

**Keywords:** pressure-based algorithm; all-Mach number flow; compressible two-phase flow; momentum-weighted interpolation; THINC interface capturing; TR-BDF2; Noble–Abel stiffened-gas; analytical Jacobian

# 1. Introduction

Compressible two-phase flows in which the local Mach number spans several orders of magnitude arise in underwater explosions, cavitating hydraulic machinery, fuel injection and bubble dynamics: an essentially incompressible liquid coexists with a highly compressible gas while acoustic waves, shocks and material interfaces interact in one domain, so numerical methods must remain accurate and stable from the incompressible limit to the strongly supersonic regime.

Density-based methods suit transonic and supersonic flows but lose accuracy and become stiff as the Mach number vanishes. Pressure-based methods retain the pressure as a primary variable, degenerate gracefully to the incompressible limit, and extend to compressible flow from the ICE method of Harlow and Amsden [1] through the pressure-correction families [2–6], with semi-implicit variants removing the acoustic time-step restriction [7]; Ref. [8] reviews the finite-volume machinery. On collocated grids a momentum-consistent face-velocity interpolation is required to avoid pressure–velocity decoupling, most commonly the momentum-weighted interpolation (MWI) of Rhie and Chow [9], whose sensitivity to under-relaxation [10] and to the time-step [11–13] is established; Bartholomew et al. [14] unified the formulation and documented the small-time-step decoupling. Fully-coupled pressure-based algorithms, solving momentum, continuity and energy as one linear system, were developed for all-speed single-phase flow by Xiao et al. [15], and Denner [16] showed that Newton linearisation of the transient and advection terms substantially improves their convergence at large Mach and Courant numbers.

For flows with material interfaces the central difficulty is spurious pressure oscillation between fluids with different equations of state, analysed in the quasi-conservative framework of Abgrall [21] and addressed by interface-capturing methods [22, 23, 25–28] derived from the multiphase models of Refs. [23–26]. Algebraic interface sharpening by the hyperbolic-tangent (THINC) reconstruction [44, 45] with boundary-variation-diminishing (BVD) selection [46] keeps a captured interface compact without a Riemann solver and is adopted here for the volume-fraction transport (Section 3.6). Within the pressure-based line, Denner et al. [17] proposed the acoustically-conservative interface discretisation (ACID), whose face density and enthalpy are built from upwind-evaluated partial quantities of the two fluids so that acoustic properties are preserved across the interface (corrigendum [18]); extensions cover the conservative all-speed NASG formulation [19, 29], coexisting incompressible and compressible fluids [20], implicit surface tension at large density ratios [43] and cavitation-bubble acoustics [42].

The present work builds directly on this literature, claims no new discretisation concept, and develops four practical aspects. First, the complete closed-form Jacobian of the coupled (velocity, pressure, enthalpy) system, including the MWI cross-coupling — the sensitivity of the advecting velocity to neighbouring pressures through the compact and wide gradient stencils and to densities through the interpolation coefficient — terms usually treated by deferred (Picard) iteration or finite differences; the algorithm is in defect-correction form, so the converged solution is rigorously independent of the Jacobian approximation, which affects only convergence behaviour and cost. Second, an L-stable second-order TR-BDF2 integration [47] of a conservative total-energy residual for the acoustic regime, removing the pressure wake of the single-step backward scheme without first-order diffusion. Third, a conservative semi-Lagrangian THINC volume-fraction transport with a parameter-free density-monotonicity guard, sharpening contacts while preserving the discrete pressure-equilibrium property. Fourth, a nineteen-case verification — machine-precision interface preservation to a Mach 100 shock — run by one locally-adaptive scheme with one global parameter set and assessed with full-field metrics, including a documented negative-result quantification of the Courant-number dependence of the MWI dissipation in the strong-shock regime, complementing Ref. [14].

Section 2 summarises the governing equations and closure; Section 3 the numerical method; Section 4 the verification suite and measured results, including the THINC, TR-BDF2, ablation, robustness and MWI-dissipation studies; Section 5 conclusions and limitations. Appendices collect the thermodynamic derivatives, exact Riemann references and Jacobian entries.

# 2. Governing equations and thermodynamic closure

## 2.1 Four-equation two-phase model

We consider one-dimensional, inviscid, compressible flow of two immiscible fluids in mechanical and thermal equilibrium: both fluids share a single pressure $p$, temperature $T$ and velocity $u$. With the volume fraction $\alpha \in [0,1]$ of fluid 1, the mixture density and specific total enthalpy are

$$\rho = \alpha \rho_1 + (1-\alpha)\rho_2, \qquad \rho h = \alpha \rho_1 h_1 + (1-\alpha)\rho_2 h_2 + \tfrac{1}{2}\rho u^2,$$

where $\rho_k(p,T)$ and $h_k(p,T)$ follow from the equation of state and $h$ is the specific *total* enthalpy. The governing equations are the four-equation model of Ref. [17]:

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0, \qquad \frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2)}{\partial x} = -\frac{\partial p}{\partial x},$$

$$\frac{\partial (\rho h)}{\partial t} + \frac{\partial (\rho u h)}{\partial x} = \frac{\partial p}{\partial t}, \qquad \frac{\partial \alpha}{\partial t} + u \frac{\partial \alpha}{\partial x} = 0.$$

The total-enthalpy form is convenient for pressure-based algorithms because the transient pressure term requires no linearisation [16, 19]. The non-conservative volume-fraction advection corresponds to the five-equation model of Allaire et al. [25] with vanishing compaction terms; the model reduces to the Euler equations in each pure fluid. No phase change, surface tension, viscosity or heat conduction is considered.

## 2.2 Noble–Abel stiffened-gas equation of state

Each fluid follows the NASG equation of state of Le Métayer and Saurel [29], representing gases, liquids and their limits in one expression:

$$\rho_k(p,T) = \frac{p + \Pi_k}{c_{v,k}(\gamma_k - 1)T + b_k (p + \Pi_k)}, \qquad h_k(p,T) = \gamma_k c_{v,k} T + b_k p + \eta_k,$$

with heat-capacity ratio $\gamma_k$, pressure constant $\Pi_k$, co-volume $b_k$, isochoric heat capacity $c_{v,k}$ and enthalpy reference $\eta_k$; the phase speed of sound is $c_k^2 = \gamma_k (p+\Pi_k)/[\rho_k (1 - b_k \rho_k)]$. Setting $b_k = \eta_k = 0$ recovers the stiffened gas, and additionally $\Pi_k = 0$ the ideal gas, so one implementation covers all fluids. The closed-form derivatives required by the analytical Jacobian are collected in Appendix A. The mixture speed of sound entering the time-step selection is the mechanical-equilibrium (Wood [30, 31]) speed, $(\rho c^2)^{-1} = \alpha/(\rho_1 c_1^2) + (1-\alpha)/(\rho_2 c_2^2)$. Table 1 lists the fluid parameters.

**Table 1.** NASG parameters of the working fluids. (SG: stiffened-gas limit, $b=\eta=0$; ideal gas: additionally $\Pi=0$.)

| Fluid | $\gamma$ | $\Pi$ (Pa) | $b$ (m^3^/kg) | $c_v$ (J/kg K) | $\eta$ (J/kg) |
|---|---|---|---|---|---|
| Air (ideal gas) | 1.4 | 0 | 0 | 717.5–720.25 | 0 |
| Second gas (ideal, Ref. [17] §7.1) | 1.6 | 0 | 0 | 3472.2 | 0 |
| Helium / argon (ideal, Ref. [17] Fig. 12) | 1.667 / 1.660 | 0 | 0 | 3047 / 288.9 | 0 |
| Matched gas (ideal, Ref. [17] Eq. 74) | 1.648 | 0 | 0 | 512.41 | 0 |
| Water (SG, Ref. [17]) | 4.1 | 4.4×10^8^ | 0 | 474.2 | 0 |
| Water (NASG, Ref. [29]) | 1.187 | 7.028×10^8^ | 6.61×10^−4^ | 3610 | −1.178×10^6^ |

# 3. Numerical method

## 3.1 Finite-volume discretisation and notation

The domain is divided into $N$ uniform cells of width $\Delta x$ with cell centres $x_i$ and faces $f = i \pm 1/2$; all variables are stored at cell centres (collocated arrangement). [Fig. 1 about here: schematic of the collocated arrangement, reconstruction stencils and pentadiagonal coupling footprint.] The primitive unknowns of the coupled solution are $\mathbf{q}_i = (u_i, p_i, h_i)$, with the temperature recovered from the total enthalpy (Section 3.7) and the volume fraction updated per Section 3.6. Boundary conditions (transmissive, reflective, velocity inlet) are imposed through two layers of ghost cells.

## 3.2 Transient discretisation: backward Euler and TR-BDF2

The transient term is discretised by first-order backward Euler (BE) in the shock-capturing regime and by an L-stable second-order two-stage scheme in the acoustic regime. The regime is selected by a case-blind indicator — the presence of a time-harmonic inlet source — never by the identity of the test case.

For $\phi \in \{\rho, \rho u, \rho h\}$ the backward-Euler update is $(\phi_i^{\,n+1} - \phi_i^{\,n})/\Delta t$. Consistently with Ref. [17], the old-level mixture density and enthalpy are re-evaluated with the *updated* volume fraction at the old pressure and temperature, so a moved interface injects no spurious transient source.

In the acoustic regime the energy equation is recast in conservative total-energy form: $\rho E = \rho h - p$ moves the pressure-work source $\partial p/\partial t$ into the time derivative. This residual is advanced with TR-BDF2 [47], a composite of the trapezoidal rule and BDF2: with $U$ the coupled unknowns and $F$ the flux-divergence operator, stage one is the trapezoidal rule over $\gamma\,\Delta t$,

$$U^{\,n+\gamma} = U^{\,n} + \frac{\gamma\,\Delta t}{2}\left[ F(U^{\,n}) + F(U^{\,n+\gamma}) \right],$$

and stage two a three-point backward formula on $\{n, n+\gamma, n+1\}$,

$$U^{\,n+1} = a\,U^{\,n+\gamma} - b\,U^{\,n} + c\,\Delta t\, F(U^{\,n+1}), \qquad a = \frac{1}{\gamma(2-\gamma)}, \quad b = \frac{(1-\gamma)^2}{\gamma(2-\gamma)}, \quad c = \frac{1-\gamma}{2-\gamma},$$

with $\gamma = 2 - \sqrt{2}$ so that both stages share one iteration matrix and the composite is L-stable. TR-BDF2 is self-starting and second-order accurate; its measured stability and convergence properties are reported with the acoustic results (Section 4.11). The MWI transient memory (Section 3.5) and the inlet condition use each stage's own timescale and instant. Shock-capturing computations use BE and the enthalpy residual and are unaffected — the two regimes are byte-for-byte separable (Section 4.11).

## 3.3 Adaptive face reconstruction of the convected quantities

Convected primitives are reconstructed from the upwind side with the accuracy selected by local indicators. A face-shock sensor flags faces whose pressure ratio over the four-cell stencil exceeds 1.3; such faces use first-order upwinding. Where a propagating acoustic wave must be resolved (time-harmonic inlet source present), a Minmod-limited second-order TVD reconstruction [32–34] is upgraded to the fourth-order central face interpolation $\phi_f = (-\phi_{i-1} + 7\phi_i + 7\phi_{i+1} - \phi_{i+2})/12$ wherever the four-cell stencil lies in a single phase, reverting to second order at the interface. The face pressure in the momentum equation uses the same central or fourth-order interpolation, preserving the telescoping property of the pressure force.

## 3.4 Acoustically-conservative face density and enthalpy (ACID)

Following Ref. [17], the face mass and enthalpy fluxes of cell $i$ are built from upwind-evaluated *partial* densities and enthalpies of both fluids, combined with the cell's own volume fraction, $\tilde{\rho}_f^{(i)} = \alpha_i\, \rho_1(p_{\mathrm{up}}, T_{\mathrm{up}}) + (1-\alpha_i)\, \rho_2(p_{\mathrm{up}}, T_{\mathrm{up}})$, and analogously for $\widetilde{\rho h}_f^{(i)}$ with the phase total enthalpies $\rho_k (h_k + u_{\mathrm{up}}^2/2)$. For a material contact advected by uniform velocity and pressure this yields $\partial_x(\tilde{\rho}_f \vartheta_f) = \rho_i\, \partial_x \vartheta_f = 0$ discretely, so the contact generates no spurious pressure or velocity disturbance — the discrete analogue of the pressure-equilibrium property of Ref. [21] — while providing a consistent face impedance for acoustic transmission across the interface [17].

## 3.5 Momentum-weighted interpolation of the advecting velocity

The advecting velocity at a face is evaluated with the transient-consistent MWI [9, 14, 17],

$$\vartheta_f = \bar{u}_f - \hat{d}_f \left[ \frac{p_{i+1} - p_i}{\Delta x} - \overline{\left(\frac{\partial p}{\partial x}\right)}_f \right] + \hat{d}_f \frac{\rho_f^{*,n}}{\Delta t} \left( \vartheta_f^{\,n} - \bar{u}_f^{\,n} \right),$$

where $\bar{u}_f$ is the interpolated cell velocity, the bracketed compact-minus-wide pressure-gradient difference is a low-pass filter on the third pressure derivative suppressing pressure–velocity decoupling, the last term is the transient memory, and $\rho_f^{*}$ is the harmonic face density [17]. The interpolation coefficient follows Eq. (21) of Ref. [17],

$$\hat{d}_f = \frac{\dfrac{V_P}{e_P} + \dfrac{V_Q}{e_Q}}{2 + \dfrac{\rho_f^{*}}{\Delta t}\left( \dfrac{V_P}{e_P} + \dfrac{V_Q}{e_Q} \right)},$$

with $e_P$ the transient momentum coefficient $\rho\,\Delta x/\Delta t$, in which limit $\hat d_f \propto \Delta t$ and the MWI dissipation vanishes with the time-step (analysed in Ref. [14]; examined for the strong shocks in Section 4.12). The correction is bounded by the local speed of sound, $|\hat d_f\,[\cdot]| \le c_f$. One transient-coefficient form serves every case — no per-case dissipation scale, no advection-coefficient switch — and, the residual being exact (Section 3.7), the MWI affects oscillation content but not converged wave speeds or jump conditions.

## 3.6 Volume-fraction transport with THINC interface capturing

The volume fraction is advanced with the advecting velocity in the non-conservative form

$$\alpha_i^{\,n+1} = \alpha_i^{\,n} - \frac{\Delta t}{\Delta x}\left( \vartheta_{i+1/2}\alpha_{f,i+1/2} - \vartheta_{i-1/2}\alpha_{f,i-1/2} \right) + \Delta t\, \alpha_i^{\,n} \frac{\vartheta_{i+1/2} - \vartheta_{i-1/2}}{\Delta x},$$

followed by clamping to $[0,1]$, before the coupled solution of $(u,p,h)$. In smooth regions $\alpha_f$ is first-order upwind; at a genuine material interface it is replaced by a THINC reconstruction [44, 45].

The interface within the upwind cell is a normalised tanh profile $H(x)$ of steepness $\beta$, positioned so its cell average reproduces $\alpha$. The face value is the *conservative semi-Lagrangian* average of this profile over the departure region swept in one step — $[1-c,\,1]$ of the upwind cell for $\vartheta_f \ge 0$, $[0,\,c]$ otherwise, with local material Courant number $c = |\vartheta_f|\,\Delta t/\Delta x$ — in the closed form

$$\alpha_f = \frac{1}{c}\left[\, \bar{x} + \frac{1}{2\beta}\ln\!\frac{\cosh\beta(\bar{x}-x_c+c/2)}{\cosh\beta(\bar{x}-x_c-c/2)} \,\right],$$

evaluated through an algebraically equivalent stable form and verified against adaptive quadrature to a relative $4\times10^{-15}$. The point (non-integrated) downwind value is *not* used: it under-transports the sub-cell interface at finite Courant number, lagging a sharp contact by of order thirty cells (case 02). The reconstruction applies only where a case-blind indicator fires — neighbours straddling the half-value, steep local variation $|\alpha_{i+1}-\alpha_{i-1}| > 0.5$, monotone profile, genuinely mixed cell ($10^{-6} < \alpha_i < 1-10^{-6}$) — otherwise plain upwind. The steepness $\beta = 3.5$ is the single global scheme constant.

THINC sharpens the volume fraction while the convected pressure and temperature remain first-order, so the implied face mixture density can leave the range of the adjacent cells and oscillate. A parameter-free density-monotonicity guard of BVD type [46] suppresses this: the candidate face is accepted only if the mixture density it implies at the upwind state, $\rho_f^{\mathrm{imp}} = \alpha_f\,\rho_1(p_{\mathrm{up}}, T_{\mathrm{up}}) + (1-\alpha_f)\,\rho_2(p_{\mathrm{up}}, T_{\mathrm{up}})$, lies within the interval spanned by the two adjacent cell mixture densities; otherwise the face reverts to upwind. The bounds are neighbour values — no new constant. The guard acts only on the volume-fraction face value, never on the mass, momentum or energy fluxes, so it cannot perturb the discrete pressure-equilibrium property (Section 4.3); its measured effect is reported in Section 4.10.

## 3.7 Fully-coupled Newton solution

The discretised equations are collected in the per-cell residual $\mathbf{R}_i = (R^{\,m}_i, R^{\,c}_i, R^{\,e}_i)$:

$$R^{\,m}_i = \frac{c_0 \rho_i u_i - \Phi^{\mathrm{old}}_{m,i}}{\Delta t}\Delta x + \left( \dot m_{i+1/2}\, \tilde u_{i+1/2} - \dot m_{i-1/2}\, \tilde u_{i-1/2} \right) + \left( \bar p_{i+1/2} - \bar p_{i-1/2} \right),$$

$$R^{\,c}_i = \frac{c_0 \rho_i - \Phi^{\mathrm{old}}_{c,i}}{\Delta t}\Delta x + \left( \dot m_{i+1/2} - \dot m_{i-1/2} \right),$$

$$R^{\,e}_i = \frac{c_0 \rho_i h_i - \Phi^{\mathrm{old}}_{e,i}}{\Delta t}\Delta x + \left( \dot m_{i+1/2} \tilde h_{i+1/2} - \dot m_{i-1/2} \tilde h_{i-1/2} \right) - \frac{p_i - p_i^{\,n}}{\Delta t}\Delta x,$$

with the ACID mass flux $\dot m_f = \tilde\rho_f^{(i)} \vartheta_f$. In the acoustic regime the energy residual takes the conservative total-energy form of Section 3.2, removing the explicit pressure-work source; the shock-capturing regime retains the enthalpy form. The mixture density $\rho_i = \rho(p_i, T_i, \alpha_i)$ comes from the equation of state, rendering the continuity equation an implicit constraint on the pressure — the defining property of a pressure-based method [15, 19]. The temperature is recovered at every nonlinear iteration by inverting

$$T_i = \frac{\rho_i h_i - \tfrac{1}{2}\rho_i u_i^2 - \sum_k \alpha_k \rho_k (b_k p_i + \eta_k)}{\sum_k \alpha_k \rho_k \gamma_k c_{v,k}},$$

under-relaxed by 1/2 with rejection of transiently non-physical iterates — necessary for robustness in the strong-shock cases; at convergence the relation holds exactly. Each Newton iteration solves $\mathbf{J}\, \delta \mathbf{q} = -\mathbf{R}(\mathbf{q}^{(k)})$, $\mathbf{q}^{(k+1)} = \mathbf{q}^{(k)} + \lambda\, \delta \mathbf{q}$, with $\lambda$ from the line search of Section 3.9. The iteration is in defect-correction form: $\mathbf{R}$ is always evaluated exactly, so the converged solution is independent of any approximation in $\mathbf{J}$. Convergence requires $\|\delta u\|_\infty < 10^{-8}\, \lambda_{\max}$ and $\|\delta p\|_\infty < 10^{-8}\, p_{\max}$, with an analogous test on the enthalpy.

## 3.8 Exact analytical Jacobian

The Jacobian has, in one dimension, the block-pentadiagonal structure $\mathbf{J} = \mathrm{penta}( \mathbf{E}_i, \mathbf{A}_i, \mathbf{B}_i, \mathbf{C}_i, \mathbf{F}_i )$ with $\mathbf{B}_i = \partial \mathbf{R}_i/\partial \mathbf{q}_i \in \mathbb{R}^{3\times 3}$, the second off-diagonals arising from the wide MWI pressure-gradient stencil and the fourth-order face interpolation. The blocks are assembled in closed form from four groups: (i) *transient terms*, e.g. $\partial R^{\,c}_i / \partial p_i = c_0 (\partial \rho/\partial p)_T\, \Delta x/\Delta t$ with the NASG derivatives of Appendix A; (ii) *convective fluxes via the upwind chain*, where the temperature recovery contributes the factor $\partial T/\partial h = \rho / \sum_k \alpha_k \rho_k \gamma_k c_{v,k}$ so that e.g. $\partial \dot m_f / \partial p_{\mathrm{up}} = \vartheta_f\, \partial \tilde\rho_f / \partial p_{\mathrm{up}}$ is closed-form; (iii) *pressure-force terms* through the central or fourth-order face weights, $j = i-2,\dots,i+2$; and (iv) *MWI cross-coupling*: the advecting velocity depends on up to four cells' pressures,

$$\frac{\partial \vartheta_f}{\partial p_j} = -\hat d_f \left( \frac{\partial}{\partial p_j}\left[\frac{p_{i+1}-p_i}{\Delta x}\right] - \frac{\partial}{\partial p_j}\overline{\left(\frac{\partial p}{\partial x}\right)}_f \right) \cdot \chi_f,$$

with $\chi_f \in \{0,1\}$ deactivating the derivative where the sound-speed bound is active, and on the densities through the interpolation coefficient,

$$\frac{\partial \vartheta_f}{\partial \rho_j} = -\left( \frac{\partial \hat d_f}{\partial \rho_j} \right) \left[ \left(\frac{\partial p}{\partial x}\right)^{\mathrm{compact}}_f - \overline{\left(\frac{\partial p}{\partial x}\right)}_f \right],$$

with the gradient difference frozen at the current iterate. These terms propagate into all three residuals through the mass flux and are, to our knowledge, those most commonly replaced by deferred evaluation; the complete block entries are listed in Appendix C.

The analytical assembly requires one pass over the faces; the baseline graph-coloured finite-difference (FD) Jacobian requires 15 perturbed residual evaluations per assembly (pentadiagonal coupling, three variables), optionally reused over $K$ iterations (modified Newton). The residual being exact, all variants converge to identical solutions, permitting a controlled cost comparison (Section 4.17). The linear solve is a direct block-pentadiagonal factorisation, linear in $N$. The acoustic (TR-BDF2) regime presently uses the FD assembly — by defect-correction invariance the converged solution is unchanged, and restoring the analytical assembly there is straightforward.

## 3.9 Globalisation

Newton's method on the strong-shock and near-vacuum cases requires safeguards; the following, with all thresholds fixed once for all cases, proved sufficient: (i) a *backtracking line search* ($\lambda$ halved until the residual norm decreases); (ii) a *best-iterate fallback* (the lowest-residual iterate is accepted if the tolerance is not met); (iii) *stall detection* (no improvement over five iterations terminates early — the near-vacuum situation, where the line search pins at the volume-fraction admissibility limit); (iv) *progress-based step control* (an unconverged, no-progress step halves the time-step and retries, with a persistent scale factor on the CFL step recovering geometrically, factor 1.5 per clean step); (v) an *admissibility guard* (non-finite states or velocities an order of magnitude beyond the problem scale reject the step). The time step follows the acoustic CFL condition $\Delta t = \mathrm{Co}\, \Delta x / \max_i (|u_i| + c_i)$; the algorithm is acoustically implicit, so Co is an accuracy rather than a stability parameter.

## 3.10 Summary of the algorithm

Per time step: (1) select $\Delta t$; (2) advance $\alpha$ (Section 3.6) and re-evaluate the old-level mixture quantities with the new $\alpha$; (3) assemble transient coefficients and regime indicators; (4) Newton iterations on $(u,p,h)$ — two TR-BDF2 stages in the acoustic regime — with residual evaluation, Jacobian assembly, pentadiagonal solve, line search and temperature recovery; (5) accept subject to the admissibility guard, else halve $\Delta t$ and return to (2); (6) store the MWI memory.

## 3.11 Parameter policy

One global parameter set is used for every case: acoustic Courant number $\mathrm{Co} = 0.45$, THINC steepness $\beta = 3.5$, and the TR-BDF2 constant $\gamma = 2-\sqrt{2}$ (mathematically fixed). Regime selection — BE versus TR-BDF2, first-order versus high-order reconstruction, material versus acoustic time-step — is made from local physical indicators (pressure ratio, time-harmonic-source detector, local velocity), never from the case identity; in particular the material step of a purely advected contact is auto-detected from near-uniform pressure with no acoustic source. All per-case numerical coefficients, advection-coefficient switches and tuned dissipation multipliers of earlier versions of this work have been removed; the entire suite runs on these defaults.

# 4. Verification suite and results

## 4.1 Test matrix and references

Table 2 summarises the nineteen benchmarks; solver case identifiers are used throughout. Cases 01–15 follow the verification programme of Ref. [17]; cases 26–36 add the single-phase strong shocks (Mach 10 and 100), the Mach 1.22 shock–interface interactions, the mixture Hugoniot at two further volume fractions, and two further acoustic reflection/transmission pairs. References are exact where available: two-material NASG Riemann solutions (Appendix B), single-phase Hugoniot states, the mixture Rankine–Hugoniot relations of Ref. [17] (Eqs. 57–62), linear-acoustic d'Alembert/impedance solutions, and exact advection. The double rarefaction has no exact solution and is assessed as a grid-self-consistency and robustness test; without a phase-change model its core pressure is set by the admissibility floor of the closure, not a physical vapour pressure, so no cavitation validation is claimed [38–40].

Two implemented cases are excluded for documented blockers: the Mach 100 *water* shock (case 29), whose acoustic time step collapses to $\mathcal{O}(10^{-9})$ s at the global settings, leaving the front under-resolved (measured pressure amplitude ratio $0.35$ against the exact $7.08\times10^{7}$ post-shock ratio); and the Woodward–Colella blast interaction (case 32), whose middle-state pressure of $0.01$ Pa lies below the solver's $1.0$ Pa floor, so the initial condition is not representable. Both retain their configurations and reference machinery in the code.

The suite spans the acoustic-Courant range the all-speed claim requires: shocks and acoustics at $\mathrm{Co} = 0.45$ on $|u|+c$, while the contact case (02) advances on the auto-detected material step whose *acoustic* Courant number $\mathrm{Co}_a = 0.45\,a/|u|$ is of order $10^{2}$–$10^{3}$ (about 157 in air, $a\approx349$ m/s; 450 in the light gas, $a\approx1000$ m/s; $|u| = 1$ m/s), carried stably by the acoustic implicitness — one algorithm from $\mathrm{Co}_a = \mathcal{O}(1)$ to several hundred.

**Table 2.** Verification matrix. Co is the acoustic Courant number based on $\max(|u|+c)$; the contact-advection case (02) additionally advances the volume fraction on the auto-detected material step. References: E = exact two-material Riemann or exact advection; H = single-phase Hugoniot; RH = mixture Rankine–Hugoniot; dA = d'Alembert / linear-acoustic; SC = self-consistency.

| Case | Description | Fluids | $N$ | Co | Ref. |
|---|---|---|---|---|---|
| 01 | Static interface, $\Delta p = 0$ | air / water(SG) | 200 | 0.45 | E |
| 02 | Contact advection ($u_0 = 1$ m/s) | gas / gas | 500 | 0.45 (material) | E |
| 04 | Acoustic wave in air ($f = 2$ kHz) | air | 500 | 0.45 | dA |
| 05 | Acoustic wave in water ($f = 6$ kHz) | water(SG) | 400 | 0.45 | dA |
| 07 | Acoustic reflection–transmission | air / water(SG) | 750 | 0.45 | dA |
| 13 | Shock tube 10^9^/10^4^ Pa | air / water(SG) | 400 | 0.45 | E |
| 14 | Shock tube (reversed) 10^9^/10^5^ Pa | water(NASG) / air | 400 | 0.45 | E |
| 15 | Double rarefaction ($|u| \le 100$ m/s) | air / water(NASG) | 400 | 0.45 | SC |
| 24 | Ms = 10 homogeneous-mixture shock, $\psi_w=0.5$ | air–water mixture | 800 | 0.45 | RH |
| 25 | Ms = 10 shock / water interface | air / water(SG) | 400 | 0.45 | E |
| 26 | Ms = 10 single-phase shock | air | 1000 | 0.45 | H |
| 27 | Ms = 10 single-phase shock | water(SG) | 1000 | 0.45 | H |
| 28 | Ms = 100 single-phase shock | air | 1000 | 0.45 | H |
| 30 | Ms = 1.22 shock–interface | air / helium | 400 | 0.45 | E |
| 31 | Ms = 1.22 shock–interface | air / matched gas | 400 | 0.45 | E |
| 33 | Ms = 10 mixture shock, $\psi_w=0.25$ | air–water mixture | 800 | 0.45 | RH |
| 34 | Ms = 10 mixture shock, $\psi_w=0.75$ | air–water mixture | 800 | 0.45 | RH |
| 35 | Acoustic reflection–transmission | helium / air | 750 | 0.45 | dA |
| 36 | Acoustic reflection–transmission | argon / air | 750 | 0.45 | dA |

## 4.2 Error metrics

Full-field metrics are computed against the reference sampled on the computational grid: range-normalised $L_2$ error, Pearson correlation, amplitude ratio (solver to reference peak-to-peak), discontinuity-position error (in cells, by steepest gradient), and a spectral high-frequency indicator — the fraction of deviation energy above one quarter of the Nyquist wavenumber — which discriminates smooth error from cell-scale oscillation that amplitude comparisons alone [17] do not constrain. All numbers below are measured with the single parameter set of Section 3.11.

## 4.3 Pressure-equilibrium preservation (cases 01, 02)

For the quiescent interface (case 01) the measured $L_\infty$ errors of $p$, $u$ and $\rho$ are exactly $0$: the ACID construction cancels the interfacial flux imbalance to machine precision, and the property survives the THINC guard. For the advected contact (case 02), pressure and velocity stay uniform to machine precision ($L_\infty(p)=0$, $L_\infty(u)=10^{-14}$); the guarded THINC transport gives density correlation $0.997$ at $N = 500$ with the contact over about five cells, against roughly forty-six for first-order upwinding (Section 4.10); Figures 2 and 3 show both cases with pointwise-error panels.

![](results_cpp/figs/case01.png){width=70%} ![](results_cpp/figs/case01_err.png){width=85%}

*Figure 2: Static liquid–gas interface (case 01), solver (red) versus reference (blue), with pointwise absolute errors below: all three errors are exactly zero (the flat line is the $10^{-18}$ plotting floor) — the discrete pressure-equilibrium property.*

![](results_cpp/figs/case02.png){width=70%} ![](results_cpp/figs/case02_err.png){width=85%}

*Figure 3: Contact advection (case 02, Ref. [17] §7.1): pressure and velocity uniform to machine precision, the density contact THINC-captured (corr $\rho = 0.997$); the error panels show $p$ and $u$ at the machine floor and the density error confined to the contact band.*

## 4.4 Single-phase acoustics (cases 04, 05)

The propagating sinusoidal waves quantify dissipation and dispersion away from interfaces: pressure amplitude ratios $1.001$ (air, case 04) and $1.001$ (water, case 05), pressure correlations $0.9989$ and $0.9999$, velocity correlations $0.9988$ and $0.9999$ — no measurable amplitude loss (Figure 4).

![](results_cpp/figs/case04.png){width=70%} ![](results_cpp/figs/case05.png){width=70%}

*Figure 4: Acoustic waves versus d'Alembert references: (top) air, case 04, 2 kHz — amplitude ratio $1.001$, correlation $0.9989$; (bottom) stiffened-gas water, case 05, 6 kHz — amplitude ratio $1.001$, correlation $0.9999$.*

## 4.5 Interfacial acoustics and reflection/transmission (cases 07, 35, 36)

A single-period wave packet impinges on a material interface; reflected and transmitted amplitudes are compared with linear-acoustic impedance relations. Air–water (case 07, impedance ratio of order $3\times10^{3}$): pressure amplitude ratio $0.986$, correlation $0.9983$, velocity amplitude ratio $0.996$. Helium–air and argon–air (cases 35, 36): amplitude ratios $0.994$ and $1.001$, correlations $0.9994$ and $0.9998$. The residual post-packet ripple, of order $10^{-1}$ Pa against $10^{5}$ Pa, is the smooth dispersive tail of the high-order reconstruction (Section 4.11); the full-field metric exposes what an amplitude comparison would not (Figures 5, 6).

![Figure 5: Air–water acoustic reflection/transmission (case 07, Ref. [17] §7.3.2): near-total reflection in the gas, near-doubled transmitted pressure. Amplitude ratio $0.986$, correlation $0.9983$.](results_cpp/figs/case07.png){width=70%}

![](results_cpp/figs/case35.png){width=70%} ![](results_cpp/figs/case36.png){width=70%}

*Figure 6: Reflection/transmission siblings (Ref. [17] Fig. 12): (top) helium–air, case 35 — amplitude ratio $0.994$, correlation $0.9994$; (bottom) argon–air, case 36 — amplitude ratio $1.001$, correlation $0.9998$.*

## 4.6 Two-phase shock tubes (cases 13, 14)

The strong liquid–gas shock tubes are compared against exact two-material NASG Riemann solutions. Case 13 (air into water): correlations $0.9973$/$0.9938$/$0.9990$ ($p$/$u$/$\rho$), shock front over one to two cells (per-face velocity-jump ratio $0.15$), zero shock-band overshoot and total-variation excess. Case 14 (reversed, NASG water into air): correlations $0.9995$/$0.9723$/$0.9929$; the reduced velocity correlation is the contact-band smear of Section 4.10 (Figure 7).

![](results_cpp/figs/case13.png){width=70%} ![](results_cpp/figs/case14.png){width=70%}

*Figure 7: Shock tubes versus exact two-material Riemann solutions: (top) case 13, air into water — correlations $0.997/0.994/0.999$ ($p/u/\rho$), monotone plateaus; (bottom) case 14, NASG water into air — correlations $0.9995/0.972/0.993$, the velocity error being the contact-band smear of Section 4.10 with its oscillatory component removed by the guard.*

## 4.7 Homogeneous-mixture shocks (cases 24, 33, 34)

The mixture shocks at water fractions $\psi_w = 0.5, 0.25, 0.75$ (cases 24, 33, 34) give pressure correlations $0.9974$, $0.9971$, $0.9978$ against the mixture Rankine–Hugoniot relations, amplitude ratios at unity and monotone plateaus (density dip and overshoot within $2\%$ and $1\%$ of the jump). A front-cell pressure spike (high-frequency indicators $0.46$–$0.53$) is structural to conservative capturing of a mixture shock with a single-temperature closure and does not contaminate the plateau (Figures 8, 9).

![Figure 8: Homogeneous Mach 10 mixture shock, $\psi_w = 0.5$ (case 24), versus mixture Rankine–Hugoniot. Correlation $0.997$, amplitude ratio $1.0$.](results_cpp/figs/case24.png){width=70%}

![](results_cpp/figs/case33.png){width=70%} ![](results_cpp/figs/case34.png){width=70%}

*Figure 9: Mixture shocks of Ref. [17] Fig. 18 at $\psi_w = 0.25$ (case 33, top; correlation $0.997$) and $\psi_w = 0.75$ (case 34, bottom; correlation $0.998$).*

## 4.8 Shock–interface interaction (cases 25, 30, 31)

In the Mach 10 shock–water-interface interaction (case 25) the wave positions are within a cell of the exact Riemann reference (guarded THINC: interface at $0$–$1$ cells), correlations $0.9943$/$0.9980$/$0.9976$ ($p$/$u$/$\rho$). The reflected shock carries a single-cell overshoot, amplitude ratio $1.243$ — the residual analysed in Section 4.12; Denner's own Fig. 23 shows the same feature. The Mach 1.22 interactions (cases 30, 31): pressure correlations $0.9917$ and $0.9963$, amplitude ratios at unity, contact sharpened to one cell and three cells respectively (Figures 10, 11).

![Figure 10: Mach 10 shock / water interface (case 25) versus exact Riemann solution. Wave positions within a cell; the reflected-shock single-cell overshoot (amplitude ratio $1.24$, Section 4.12) also appears in Ref. [17], Fig. 23.](results_cpp/figs/case25.png){width=70%}

![](results_cpp/figs/case30.png){width=70%} ![](results_cpp/figs/case31.png){width=70%}

*Figure 11: Mach 1.22 shock–interface interactions: (top) air–helium, case 30 (Ref. [17] §7.4.3) — correlation $0.992$, contact sharpened to one cell; (bottom) air–matched-gas, case 31 (Ref. [17] §7.4.5) — correlation $0.996$, contact over three cells.*

## 4.9 Single-phase strong shocks (cases 26, 27, 28)

The single-phase shocks verify conservative capturing at extreme compression, away from any interface. Mach 10 in air (case 26) and water (case 27): pressure correlations $0.9975$ and $0.9985$, amplitude ratios $1.0005$ and $1.004$. Mach 100 in air (case 28), an extreme globalisation test: correlation $0.9974$, amplitude ratio $1.0005$, front over one to two cells, monotone plateau (Figures 12, 13).

![](results_cpp/figs/case26.png){width=70%} ![](results_cpp/figs/case27.png){width=70%}

*Figure 12: Single-phase Mach 10 shocks versus exact Hugoniot states: (top) air, case 26 — correlation $0.998$, amplitude ratio $1.0005$; (bottom) water, case 27 — correlation $0.998$, amplitude ratio $1.004$.*

![Figure 13: Single-phase Mach 100 air shock (case 28, Ref. [17] Fig. 17a) versus exact Hugoniot: correlation $0.997$, amplitude ratio $1.0005$ at a $\sim10^{4}$-fold pressure ratio.](results_cpp/figs/case28.png){width=70%}

## 4.10 THINC interface capturing

*Sharpening.* The semi-Lagrangian THINC flux sharpens every material contact in the suite: case 02 narrows from about $46$ to about $5$ cells (density correlation $0.980\to0.997$; an unguarded front reaches one cell at $0.9999$, traded by the guard for a one-cell offset, below); the Mach 1.22 contacts narrow from $15$ to $1$ cell (case 30) and $12$ to $3$ cells (case 31); the case 13 contact band from $14$ to $9$ cells. The point downwind value under-transports the sub-cell interface (case 02 front lag of order thirty cells), hence the conservative departure-interval average.

*Density-monotonicity guard.* In the reversed shock tube (case 14) unguarded THINC produces a genuine density oscillation: total-variation excess $44.6\%$ of the jump over the contact band, against a monotone-smear floor of $0.7\%$. The guard removes it — $0.69\%$ at $N = 400$, $0.37\%$ at $N = 800$ — while improving the velocity correlation from $0.966$ to $0.972$. It also cleans the case 25 interface by two orders of magnitude (interface pressure error $0.012 \to 0.0001$; wave positions $8/1/9 \to 0/0/1$ cells) by rejecting early-transient activations at the shocked contact. Its one cost: machine-precision-level rejections on the uniform-state contact of case 02 introduce a one-cell offset, lowering the density correlation from $0.9999$ to $0.997$ — still far above the $0.90$ gate and the $0.980$ upwind baseline; the guarded $0.997$ is the value in Table 7. Case 01 retains $L_\infty(p)=0$ throughout.

## 4.11 TR-BDF2 acoustic time integration

On a scalar linear-decay test $\dot y = -\lambda y$ the composite scheme shows an observed convergence rate of $2.000$ and an amplification factor $R(z)\to 0$ as $z=-\lambda\,\Delta t\to-\infty$ (measured $R(z)\approx -4.83/|z|$): L-stable, actively driving the highest-frequency temporal modes to zero. With the conservative total-energy residual removing the pressure-work source retained by the enthalpy form, this distinguishes TR-BDF2 from the single-step backward scheme, whose source drives a wake and damps the wave. Measured: the case 07 wake peak-to-peak falls from $1.69$ Pa to $0.80$ Pa, the velocity amplitude ratio recovers from $0.88$ to $0.996$, the case 35 pressure amplitude ratio rises from $0.71$ to $0.99$, and all five acoustic correlations (04, 05, 07, 35, 36) reach $0.999$ or above. The remaining sub-$0.1$ Pa case 07 ripple is a smooth dispersion tail of the high-order low-dissipation reconstruction — reducing the spatial order makes it several times larger — not a time-integration artefact. Every backward-Euler case is byte-for-byte identical to the pre-change implementation.

## 4.12 On time-step-independent momentum-weighted interpolation

The single residual in the suite is the case 25 reflected-shock overshoot: a single-cell pressure feature with amplitude ratio $1.243$. Its mechanism is a documented property of collocated MWI [14]: with $e_P = \rho\,\Delta x/\Delta t$ the coefficient scales as $\hat d_f \propto \Delta t$, so the MWI dissipation *vanishes* as the time-step is refined, under-damping the pressure–velocity coupling at a strong shock; the overshoot accordingly grows as the step is reduced. Table 3 (left) gives the measured Courant dependence: amplitude ratio $1.243$ at Co = 0.45 rising to $1.351$ at Co = 0.10.

We searched systematically for a parameter-free MWI dissipation independent of the time-step yet harmless to the low-Mach and near-vacuum cases. Four forms were implemented faithfully and measured; each is a negative result, and the code was reverted to the transient-coefficient baseline in every case. (i) The Bartholomew et al. [14] *unified* form (advective coefficient $a = \rho|u|$ with the paper's memory feedback) is genuinely time-step-independent and essentially removes the overshoot (case 25 amplitude ratio $1.0003$) — but its near-unity memory feedback at low Mach integrates acoustic-scale corrections into divergence, and the five acoustic cases explode. (ii) An acoustic-signal-speed coefficient $a = \rho(|u|+c)$ is time-step-independent at its fixed point and lowers the overshoot uniformly at every Courant number (Table 3, right: $1.324$/$1.277$/$1.207$ at Co = 0.10/0.20/0.45 versus baseline $1.351$/$1.314$/$1.243$) — but the *slope* of the sweep is unchanged (a moving front never reaches the fixed point), and where the local signal speed collapses (the near-vacuum core of case 15) the memory feedback approaches unity and over-filters, dropping the suite to $18/19$. (iii) A linearized acoustic-Riemann face pressure at shock faces, $p^{*}_f = (Z_R p_L + Z_L p_R)/(Z_L+Z_R) + Z_L Z_R (u_L-u_R)/(Z_L+Z_R)$, cures the overshoot (amplitude ratio $1.03$) but its jump term is $5$–$10$ times too strong for the already-clean single-phase and mixture shocks, over-diffusing their fronts past the sharpness and plateau gates. (iv) Gating that term on MWI-clamp saturation never fires on the target — a *clean* Mach 10 shock (case 26) carries a face-level saturation signature of $0.39426$, indistinguishable from case 25's $0.39430$ — and instead fires on the violent initial transients of the cases it must not touch.

The measured conclusion: for collocated implicit MWI on this suite, per-step time-step independence, low-Mach stability and near-vacuum robustness form a *pick-two* constraint. Every candidate that achieves time-step independence at the shock either destabilises the resolved acoustic field (i, ii at its extreme) or over-diffuses the clean shocks (iii), and no face-local instantaneous signal discriminates the under-damped reflected front from a clean Hugoniot front (iv) — the two are locally identical. The case 25 residual is a one-cell start-up imprint of the reflection event, locally indistinguishable from a physical front, and the same feature appears in Denner's own reference computation (Ref. [17], Fig. 23). We report it as a documented residual rather than tune it away.

**Table 3.** Case 25 reflected-shock overshoot. Left: Courant-number dependence of the transient-coefficient baseline. Right: the acoustic-signal-speed coefficient $\rho(|u|+c)$ at the same Courant numbers; the level is reduced uniformly but the slope (and hence the time-step dependence) is unchanged. Values are the pressure amplitude ratio (high-frequency indicator in parentheses).

| Co | baseline amp. (HF) | | Co | $\rho(|u|+c)$ amp. (HF) |
|---|---|---|---|---|
| 0.10 | 1.351 (1.087) | | 0.10 | 1.324 (0.996) |
| 0.20 | 1.314 (0.994) | | 0.20 | 1.277 (0.897) |
| 0.45 | 1.243 (0.842) | | 0.45 | 1.207 (0.749) |

## 4.13 Ablation of the scheme ingredients

Cases 02, 07, 14 and 25 were re-run with each of three ingredients removed in turn via the solver's environment opt-outs: the guarded THINC transport, the TR-BDF2 integrator, and the exact analytical Jacobian (replaced by the coloured FD assembly with modified-Newton reuse); Table 4 reports the metric each most affects. THINC owns the sharp contact of case 02 (density correlation $0.980\to0.997$, band $46\to5$ cells) and is inactive on the acoustic case 07; TR-BDF2 owns the case 07 wake removal (velocity amplitude ratio $0.880\to0.996$, wake $1.69\to0.80$ Pa) and leaves the backward-Euler cases 02, 14, 25 unchanged. The case 25 amplitude ratio is $1.243$ in every column: that residual belongs to the momentum interpolation (Section 4.12), orthogonal to all three ingredients.

The exact Jacobian is a cost lever, not an accuracy lever (defect correction): the FD column reproduces the production solution on the well-converged cases (case 07 byte-for-byte, maximum pressure difference exactly $0$; case 25 to $0.2$ Pa against $\sim10^{6}$ Pa). Two informative exceptions: the stiffest shock tube (case 14) accepts a marginally different iterate ($\sim3\times10^{-3}$ relative pressure difference); and on the MWI-dominant case 02 the coupled $(u,p)$ solution matches to $10^{-10}$, yet the *explicitly* advected volume fraction drifts under the FD path's looser per-step convergence, smearing the contact back toward the upwind result (correlation $0.997\to0.980$, band $5\to44$ cells). The exact Jacobian thus also protects interface sharpening on this stiff case (cost comparison: Section 4.17).

**Table 4.** Ablation on four representative cases: production scheme versus each ingredient removed. THINC = semi-Lagrangian THINC transport with the density-monotonicity guard; TR-BDF2 = L-stable acoustic integrator; AJAC = exact analytical Jacobian (removed = coloured FD assembly with modified-Newton reuse). Each cell reports the two metrics named for that case.

| Case — metric | Production | $-$THINC | $-$TR-BDF2 | $-$AJAC (FD) |
|---|---|---|---|---|
| 02 — corr $\rho$ / band (cells) | 0.997 / 5 | 0.980 / 46 | 0.997 / 5 | 0.980 / 44 † |
| 07 — amp($u$) / wake p2p (Pa) | 0.996 / 0.80 | 0.996 / 0.80 | 0.880 / 1.69 | 0.996 / 0.80 |
| 14 — corr $u$ / $\rho$ TV-excess (%) | 0.972 / 1.9 | 0.973 / 2.4 | 0.972 / 1.9 | 0.968 / 2.0 |
| 25 — amp($p$) / interface $\Delta p$ ($\times10^{-4}$) | 1.243 / 1.3 | 1.243 / 1.4 | 1.243 / 1.3 | 1.243 / 1.3 |

† The coupled $(u,p)$ solution is identical to $10^{-10}$; the explicitly-advected volume fraction drifts under the FD path's looser per-step convergence (see text).

## 4.14 Courant-number robustness

Being acoustically implicit, the scheme treats Co as an accuracy parameter (Section 3.9). Table 5 confirms this on case 07 across a three-fold Courant range: amplitude ratios and correlation change in the third decimal only, the wake decreasing modestly as the step coarsens (the MWI transient filter strengthens with $\Delta t$). The complementary shock-side sweep — the case 25 overshoot, which *grows* as the step is refined — is the baseline column of Table 3; the two sweeps bracket the acoustic and strong-shock regimes.

**Table 5.** Courant-number robustness on the acoustic case 07 (global Co varied about the default 0.45). Amplitude ratios, pressure correlation and residual wake peak-to-peak.

| Co | amp($p$) | amp($u$) | corr($p$) | wake p2p (Pa) |
|---|---|---|---|---|
| 0.20 | 0.993 | 0.999 | 0.9983 | 0.87 |
| 0.45 | 0.986 | 0.996 | 0.9983 | 0.80 |
| 0.60 | 0.983 | 0.995 | 0.9982 | 0.78 |

## 4.15 Grid refinement

Table 6 shows the error behaviour under refinement for four cases of distinct character at production settings. The acoustic waves (04, 05) are amplitude-converged at the baseline resolution — amplitude ratio unity to three decimals — so the small $L_2(p)$ plateaus: its residual is a fixed sub-grid phase error, not dissipation. The captured contact (02) shows the defining property of interface capturing: the front stays a fixed one-to-two cells wide at all $N$, holding the density correlation near $0.997$ across a four-fold refinement. The shock tube (14) converges at the honest sub-first-order rate of a captured discontinuity: velocity $L_2$ drops with observed order about $0.4$ (correlation $0.972\to0.991$ from $N=400$ to $1600$), consistent with the $\mathcal{O}(N^{-1/2})$ error of a first-order-captured jump in a global norm. No case shows the interior scheme's second-order asymptotic rate, because a discontinuity or fixed phase error sets each global norm — a limitation stated rather than obscured.

**Table 6.** Grid refinement at the production settings. Acoustic cases (04, 05) report amplitude ratio and $L_2(p)$; the contact (02) reports density correlation; the shock tube (14) reports $L_2(u)$ with the observed order between successive levels.

| Case | metric | coarse | medium | fine |
|---|---|---|---|---|
| 04 (acoustic, air) | amp($p$) | 1.001 ($N$=500) | 1.000 (1000) | 1.000 (2000) |
| 04 (acoustic, air) | $L_2(p)$ | 0.0148 | 0.0150 | 0.0151 |
| 05 (acoustic, water) | $L_2(p)$ | 0.0030 ($N$=400) | 0.0025 (800) | — |
| 02 (contact) | corr $\rho$ | 0.997 ($N$=500) | 0.998 (1000) | 0.996 (2000) |
| 14 (shock tube) | $L_2(u)$ | 0.103 ($N$=400) | 0.072 (800) | 0.059 (1600) |

For the shock tube the successive-level orders are $0.51$ ($N$: $400\to800$) and $0.31$ ($800\to1600$), averaging near the $\mathcal{O}(N^{-1/2})$ discontinuity rate.

## 4.16 Double-rarefaction robustness test (case 15)

The symmetric double rarefaction drives the mixture toward vacuum and is the case for which the globalisation of Section 3.9 is decisive: the Newton iteration stalls at the volume-fraction admissibility limit, and the best-iterate fallback with stall detection carries the computation at an unreduced time step. Agreement with the refined-grid solution: correlations $0.9986$ (velocity) and $0.9950$ (density). The core pressure sits at the EOS admissibility floor in both solver and self-consistent reference, so the pressure error is nominally zero; no cavitation claim is attached (Section 4.1).

## 4.17 Cost of the exact Jacobian

The analytical and FD assemblies converge to identical solutions by the defect-correction construction, verified on the well-converged cases (Section 4.13); they differ only in cost: $15$ perturbed residual evaluations per FD assembly against one analytical pass. Since residual evaluation dominates the per-iteration cost, the analytical assembly cuts assembly work by an order of magnitude against per-iteration FD reassembly, less against FD reuse. Newton iteration counts are comparable (both approximate the same matrix); the exact form additionally benefits stiff strong-shock steps, where FD perturbation noise degrades the Newton direction. We report structural counts rather than one wall-time ratio, as the balance depends on the EOS-evaluation cost and linear-solver share.

## 4.18 Conservation

Mixture mass, momentum and total energy are conserved to machine precision by the flux-form discretisation, the conservative total-energy residual making this exact for the energy as well; the volume-fraction transport is non-conservative by design of the model [25]. Measured global conservation errors are at accumulated round-off.

## 4.19 Summary of verification metrics

Table 7 collects the measured full-field metrics; all nineteen cases satisfy their acceptance criteria with the single parameter set of Section 3.11. The machine-precision cases (01, 02) quantify the discrete pressure-equilibrium property; the remaining error is dominated by front smearing (02, 13, 14, strong shocks), wave dissipation and dispersion (04, 05, 07, 35, 36), and the case 25 single-cell imprint.

**Table 7.** Measured full-field verification metrics: range-normalised $L_2$ of pressure, Pearson correlations of $p$, $u$ and $\rho$, and pressure amplitude ratio. (Case 15 is a self-consistency reference with a floor-limited core pressure, so $L_2(p)=0$ and the amplitude ratio is not meaningful; its discriminating fields are $u$ and $\rho$.)

| Case | $N$ | $L_2(p)$ | corr $p$ | corr $u$ | corr $\rho$ | amp($p$) |
|---|---|---|---|---|---|---|
| 01 | 200 | 0 (machine) | 1 | 1 | 1 | 1 |
| 02 | 500 | 0 (machine) | 1 | 1 | 0.997 | 1 |
| 04 | 500 | 0.0148 | 0.9989 | 0.9988 | 0.9989 | 1.001 |
| 05 | 400 | 0.0030 | 0.9999 | 0.9999 | 0.9999 | 1.001 |
| 07 | 750 | 0.0096 | 0.9983 | 0.9979 | 1.000 | 0.986 |
| 13 | 400 | 0.0216 | 0.9973 | 0.9938 | 0.9990 | 1.000 |
| 14 | 400 | 0.0141 | 0.9995 | 0.9723 | 0.9929 | 1.000 |
| 15 | 400 | 0 † | 1 | 0.9986 | 0.9950 | — |
| 24 | 800 | 0.0289 | 0.9974 | 0.9957 | 0.9972 | 1.000 |
| 25 | 400 | 0.0480 | 0.9943 | 0.9980 | 0.9976 | 1.243 |
| 26 | 1000 | 0.0282 | 0.9975 | 0.9954 | 0.9974 | 1.001 |
| 27 | 1000 | 0.0222 | 0.9985 | 0.9980 | 0.9981 | 1.004 |
| 28 | 1000 | 0.0287 | 0.9974 | 0.9952 | 0.9973 | 1.001 |
| 30 | 400 | 0.0340 | 0.9917 | 0.9844 | 0.9994 | 1.000 |
| 31 | 400 | 0.0414 | 0.9963 | 0.9961 | 0.9986 | 1.000 |
| 33 | 800 | 0.0303 | 0.9971 | 0.9950 | 0.9970 | 1.000 |
| 34 | 800 | 0.0266 | 0.9978 | 0.9967 | 0.9976 | 1.000 |
| 35 | 750 | 0.0030 | 0.9994 | 0.9994 | 1.000 | 0.994 |
| 36 | 750 | 0.0023 | 0.9998 | 0.9998 | 1.000 | 1.001 |

† Floor-limited core pressure (self-consistency reference); see Section 4.16.

## 4.20 Reproducibility and scope

Every metric above is produced by the released validation binary directly from the solver and reference states, with no post-processing beyond Section 4.2, on the single parameter set of Section 3.11. All references are exact except the stated self-consistency case 15; cases 29 and 32 are excluded per Section 4.1. The ablation, Courant and refinement studies (Sections 4.13–4.15) used the solver's environment opt-outs and a temporary, subsequently reverted measurement patch; the reverted source reproduces the byte-identical nineteen-case suite. Source and data will be made available (Acknowledgements).

# 5. Conclusions

A fully-coupled pressure-based algorithm for inviscid compressible two-phase flows at all speeds has been presented and verified, following the ACID framework of Denner et al. [17] with NASG thermodynamics [19, 29]. Four developments were measured: an exact closed-form block-pentadiagonal Jacobian of the coupled (velocity, pressure, enthalpy) system including the MWI cross-coupling, cutting Jacobian-assembly cost from fifteen residual evaluations to one with the converged solution unchanged; an L-stable second-order TR-BDF2 integration of a conservative total-energy residual that removes the acoustic pressure wake and restores wave amplitude while leaving every shock-capturing computation byte-for-byte unchanged; a conservative semi-Lagrangian THINC transport with a parameter-free density-monotonicity guard that sharpens material contacts to a few cells without density oscillation or loss of the discrete pressure-equilibrium property; and a nineteen-case verification — machine-precision interface preservation to a Mach 100 shock — carried by one locally-adaptive scheme with a single global parameter set, the Courant-number dependence of the MWI dissipation documented as a systematic negative-result analysis.

The limitations are stated explicitly. All computations are one-dimensional and inviscid; the multi-dimensional extension of the analytical Jacobian involves additional geometric terms and is left to future work. The reversed shock tube retains a smeared contact band — the four-equation model carries one mixture temperature per cell, so a cell whose volume fraction flips within one to two cells relaxes its state over the following cells; the guard removes the oscillatory component but not the smear, whose cure (phase-consistent energy transport, a five-equation-model-like extension) is a model change — and the guard costs a one-cell offset on the uniform contact (correlation $0.997$, Section 4.10). The case 25 reflected-shock overshoot is a start-up imprint shared by the reference computation of Ref. [17] (Fig. 23); a parameter-free time-step-independent MWI dissipation valid across the full Mach range is constraint-bound by the pick-two trade-off of Section 4.12 — an open problem. The double rarefaction is a robustness and self-consistency test only. Cases 29 and 32 are excluded for the documented blockers of Section 4.1.

# Appendix A. NASG thermodynamic derivatives

From the NASG relations of Section 2.2, with $D_k = c_{v,k}(\gamma_k - 1)T + b_k(p + \Pi_k)$ so that $\rho_k = (p+\Pi_k)/D_k$:

$$\left(\frac{\partial \rho_k}{\partial p}\right)_T = \frac{D_k - b_k (p+\Pi_k)}{D_k^2} = \frac{c_{v,k}(\gamma_k-1)T}{D_k^2}, \qquad \left(\frac{\partial \rho_k}{\partial T}\right)_p = -\frac{(p+\Pi_k)\, c_{v,k}(\gamma_k - 1)}{D_k^2},$$

$$\left(\frac{\partial h_k}{\partial p}\right)_T = b_k, \qquad \left(\frac{\partial h_k}{\partial T}\right)_p = \gamma_k c_{v,k}, \qquad c_{p,k} = \gamma_k c_{v,k}.$$

The mixture derivatives at fixed $\alpha$ follow by volume-fraction weighting, e.g. $(\partial \rho/\partial p)_T = \alpha (\partial \rho_1/\partial p)_T + (1-\alpha)(\partial \rho_2/\partial p)_T$. The temperature-recovery chain factor used in the Jacobian is $\partial T/\partial (\rho h_s) = 1/\sum_k \alpha_k \rho_k \gamma_k c_{v,k}$ at frozen $\rho_k$, consistent with the under-relaxed inversion of Section 3.7.

# Appendix B. Exact two-material NASG Riemann references

The references for the shock tubes (13, 14) and shock–interface interactions (25, 30, 31) are exact two-material NASG Riemann solutions. Rarefactions follow the NASG isentrope,

$$p + \Pi = K \left( \frac{\rho}{1 - b\rho} \right)^{\gamma}, \qquad c^2 = \frac{\gamma (p+\Pi)}{\rho(1-b\rho)},$$

with the Riemann invariant $u \pm \int c\, d\rho / \rho$ in closed form; shocks satisfy the NASG Rankine–Hugoniot conditions; the waves are connected across the contact by continuity of pressure and velocity, the star state found by Newton iteration on the pressure. For the shock–interface cases the incident shock is constructed in the driver gas and the interaction solved as a Riemann problem at the interface, evaluated at the final time on the grid. The single-phase shocks (26–28) use the exact Hugoniot state; the mixture shocks (24, 33, 34) the mixture Rankine–Hugoniot relations (Eqs. 57–62 of Ref. [17]) with the thermodynamically-consistent mixture sound speed and the volume fraction held across the shock.

# Appendix C. Entries of the analytical Jacobian

[This appendix tabulates the non-zero entries of the 3×3 blocks $\mathbf{E}_i, \mathbf{A}_i, \mathbf{B}_i, \mathbf{C}_i, \mathbf{F}_i$ by contribution: transient (diagonal), convective upwind chain (first neighbours), pressure force (up to second neighbours through the fourth-order face weights), and MWI cross-coupling (up to second neighbours through the wide gradient), including the gating indicators for the sound-speed bound and the reconstruction selector. To be included in the final version; the entries follow mechanically from Sections 3.3–3.8 and Appendix A.]

# Acknowledgements

[Funding and acknowledgement information to be completed.] The author declares no competing financial interests. The verification data and the solver source code will be made available upon reasonable request [or: at a public repository, DOI to be added].

# References

[1] F.H. Harlow, A.A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, J. Comput. Phys. 8 (1971) 197–213.

[2] S.V. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere, Washington, DC, 1980.

[3] R.I. Issa, Solution of the implicitly discretised fluid flow equations by operator-splitting, J. Comput. Phys. 62 (1986) 40–65.

[4] K.C. Karki, S.V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA J. 27 (1989) 1167–1174.

[5] I. Demirdžić, Ž. Lilek, M. Perić, A collocated finite volume method for predicting flows at all speeds, Int. J. Numer. Methods Fluids 16 (1993) 1029–1050.

[6] D.R. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Comput. Fluids 32 (2003) 1113–1132.

[7] N. Kwatra, J. Su, J.T. Grétarsson, R. Fedkiw, A method for avoiding the acoustic time step restriction in compressible flow, J. Comput. Phys. 228 (2009) 4146–4161.

[8] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics, Springer, 2016.

[9] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532.

[10] S. Majumdar, Role of underrelaxation in momentum interpolation for calculation of flow with nonstaggered grids, Numer. Heat Transf. 13 (1988) 125–132.

[11] B. Yu, W.-Q. Tao, J.-J. Wei, Y. Kawaguchi, T. Tagawa, H. Ozoe, Discussion on momentum interpolation method for collocated grids of incompressible flow, Numer. Heat Transf. B 42 (2002) 141–166.

[12] A. Cubero, N. Fueyo, A compact momentum interpolation procedure for unsteady flows and relaxation, Numer. Heat Transf. B 52 (2007) 507–529.

[13] A. Pascau, Cell face velocity alternatives in a structured colocated grid for the unsteady Navier–Stokes equations, Int. J. Numer. Methods Fluids 65 (2011) 812–833.

[14] P. Bartholomew, F. Denner, M.H. Abdol-Azis, A. Marquis, B.G.M. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, J. Comput. Phys. 375 (2018) 177–208.

[15] C.-N. Xiao, F. Denner, B.G.M. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, J. Comput. Phys. 346 (2017) 91–130.

[16] F. Denner, Fully-coupled pressure-based algorithm for compressible flows: linearisation and iterative solution strategies, Comput. Fluids 175 (2018) 53–65.

[17] F. Denner, C.-N. Xiao, B.G.M. van Wachem, Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, J. Comput. Phys. 367 (2018) 192–234.

[18] F. Denner, B.G.M. van Wachem, Corrigendum to "Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation" [J. Comput. Phys. 367 (2018) 192–234], J. Comput. Phys. 381 (2019) 290–291.

[19] F. Denner, F. Evrard, B.G.M. van Wachem, Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds, J. Comput. Phys. 409 (2020) 109348.

[20] F. Denner, B.G.M. van Wachem, A unified algorithm for interfacial flows with incompressible and compressible fluids, in: Advances in Fluid Mechanics: Modelling and Simulations, Springer, Singapore, 2022.

[21] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160.

[22] K.-M. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, J. Comput. Phys. 142 (1998) 208–242.

[23] R. Saurel, R. Abgrall, A multiphase Godunov method for compressible multifluid and multiphase flows, J. Comput. Phys. 150 (1999) 425–467.

[24] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: reduced equations, Phys. Fluids 13 (2001) 3002–3024.

[25] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577–616.

[26] A. Murrone, H. Guillard, A five equation reduced model for compressible two phase flow problems, J. Comput. Phys. 202 (2005) 664–698.

[27] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2006) 715–732.

[28] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible multicomponent flows, J. Comput. Phys. 274 (2014) 95–121.

[29] O. Le Métayer, R. Saurel, The Noble-Abel stiffened-gas equation of state, Phys. Fluids 28 (2016) 046102.

[30] A.B. Wood, A Textbook of Sound, G. Bell and Sons, London, 1930.

[31] G.B. Wallis, One-dimensional Two-phase Flow, McGraw-Hill, New York, 1969.

[32] B. van Leer, Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method, J. Comput. Phys. 32 (1979) 101–136.

[33] A. Harten, High resolution schemes for hyperbolic conservation laws, J. Comput. Phys. 49 (1983) 357–393.

[34] P.K. Sweby, High resolution schemes using flux limiters for hyperbolic conservation laws, SIAM J. Numer. Anal. 21 (1984) 995–1011.

[35] A. Jameson, W. Schmidt, E. Turkel, Numerical solution of the Euler equations by finite volume methods using Runge–Kutta time stepping schemes, AIAA Paper 81-1259, 1981.

[36] C.W. Gear, Numerical Initial Value Problems in Ordinary Differential Equations, Prentice-Hall, Englewood Cliffs, NJ, 1971.

[37] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, third ed., Springer, Berlin, 2009.

[38] R. Saurel, F. Petitpas, R.A. Berry, Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures, J. Comput. Phys. 228 (2009) 1678–1712.

[39] A. Zein, M. Hantke, G. Warnecke, Modeling phase transition for compressible two-phase flows applied to metastable liquids, J. Comput. Phys. 229 (2010) 2964–2998.

[40] C.E. Brennen, Cavitation and Bubble Dynamics, Oxford University Press, New York, 1995.

[41] Y. Moguen, P. Bruel, E. Dick, A combined momentum-interpolation and advection upstream splitting pressure-correction algorithm for simulation of convective and acoustic transport at all levels of Mach number, J. Comput. Phys. 384 (2019) 16–41.

[42] F. Denner, S. Schenke, Modeling acoustic emissions and shock formation of cavitation bubbles, Phys. Fluids 35 (2023) 012114.

[43] R. Janodet, B.G.M. van Wachem, F. Denner, A fully-coupled algorithm with implicit surface tension treatment for interfacial flows with large density ratios, J. Comput. Phys. (2025), in press.

[44] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, Int. J. Numer. Methods Fluids 48 (2005) 1023–1040.

[45] K.-M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for compressible two-phase flow: the algebraic THINC approach, J. Comput. Phys. 268 (2014) 326–354.

[46] Z. Sun, S. Inaba, F. Xiao, Boundary variation diminishing (BVD) reconstruction: a new approach to improve Godunov schemes, J. Comput. Phys. 322 (2016) 309–325.

[47] R.E. Bank, W.M. Coughran, W. Fichtner, E.H. Grosse, D.J. Rose, R.K. Smith, Transient simulation of silicon devices and circuits, IEEE Trans. Comput.-Aided Des. 4 (1985) 436–451.
