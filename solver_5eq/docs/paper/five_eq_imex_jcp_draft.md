---
title: "A one-dimensional all-speed implicit–explicit finite-volume method for compressible two-phase flows with the five-equation model, general equations of state, and low-diffusion interface transport"
author:
- "Younglin Yoo^a^"
- "^a^ [Affiliation, Department, Address], Republic of Korea. E-mail: younglin90@gmail.com"
abstract: |
  We present a one-dimensional finite-volume method for the five-equation diffuse-interface model of compressible two-phase flow that remains stable and accurate from nearly incompressible to hypersonic regimes under a time step restricted only by the material velocity. The method combines an implicit–explicit (IMEX) splitting, in which nonlinear material transport is advanced explicitly by a strong-stability-preserving Runge–Kutta composition and the acoustic subsystem is advanced by a single linearized implicit solve, with a thermodynamically consistent spatial discretization built on temperature-based primitive variables $(\alpha_1, T_1, T_2, u, p)$ and fifth-order weighted essentially non-oscillatory reconstruction of the pressure and velocity at material-clean acoustic faces. The primitive set admits closed-form equation-of-state (EOS) derivatives for ideal-gas, stiffened-gas, and Noble–Abel stiffened-gas fluids, an analytic $5\times 5$ transformation Jacobian, and face states that are re-evaluated on the EOS surface rather than interpolated, which together yield discrete preservation of pressure–velocity equilibrium across material interfaces without EOS iteration inside the implicit operator. Interface transport uses a topology-adaptive blend of monotone upwind-limited reconstruction and interface-compressive schemes, and the total-energy flux employs path-consistent energy coefficients that satisfy a discrete energy-consistency identity to machine precision. A linearized amplification analysis is used to select the time integrator and to identify and suppress a pressure odd–even decoupling mode. The method is assessed, with a single set of numerical parameters, on a thirteen-case one-dimensional suite spanning equilibrium interface advection at acoustic Courant numbers exceeding one hundred, linear acoustic transmission and reflection at gas–liquid interfaces, gas–liquid shock tubes, advection with large phase-temperature contrast, and strong shocks up to Mach 10.
---

**Keywords:** compressible multiphase flow; five-equation model; all-speed scheme; implicit–explicit time integration; diffuse interface; noble–abel stiffened-gas equation of state

# 1. Introduction

Compressible flows containing immiscible fluids separated by material interfaces arise in a wide range of applications, including underwater explosions, cavitation, fuel injection, and liquid–gas atomization. Diffuse-interface methods describe such flows with a single system of conservation laws in which the interface is represented by a smooth transition of a volume fraction, avoiding explicit interface reconstruction and permitting a unified treatment of interfaces and shocks. Among the hierarchy of diffuse-interface models derived from the Baer–Nunziato seven-equation system [1,2], the five-equation models of Kapila et al. [3] and Allaire et al. [4], with a single velocity and a single pressure, offer a widely used compromise between physical fidelity and computational cost, and their mathematical structure has been analyzed in detail by Murrone and Guillard [5].

Two well-known numerical difficulties motivate the present work. The first is the generation of spurious pressure oscillations at material interfaces when conservative quantities are interpolated across an interface separating fluids with different equations of state (EOS); this mechanism was identified by Abgrall [6] and has driven the development of quasi-conservative interface treatments [7,8]. The second is the acoustic time-step restriction of explicit schemes. In many two-phase applications the liquid sound speed exceeds the material velocity by two to four orders of magnitude, so an explicit scheme spends nearly all of its computational effort resolving acoustic waves that may be of little interest, and, in addition, upwind fluxes constructed for the acoustic scale introduce excessive dissipation at low Mach number [10].

Several semi-implicit and IMEX strategies have been proposed to remove the acoustic restriction for two-phase models. Chalons, Girardin, and Kokh [11] developed an all-regime Lagrange-projection scheme for the homogeneous five-equation model in which the acoustic step is treated implicitly, and Peluchon, Gallice, and Mieussens [12] constructed a robust first-order acoustic–transport splitting for the Allaire model with an implicit Godunov-type acoustic solver, later extended to second order with an interface-compressive limiter by Tallois, Peluchon, and Villedieu [15]. Related splittings were studied by Iampietro et al. [13] and, in fully explicit form for the Kapila model, by ten Eikelder et al. [14]. These schemes are formulated for stiffened-gas fluids and evolve conservative or Lagrangian variables. In the single-phase setting, semi-implicit schemes accommodating general nonideal EOS have been developed by Dumbser and Casulli [16] and extended to high order by Boscheri and Pareschi [17], using the nested-Newton solution technique of Casulli and Zanolli [18] for the mildly nonlinear pressure system.

A complementary line of work originates in pressure-based methods. Denner, Xiao, and van Wachem [19] proposed a fully coupled pressure-based algorithm for interfacial flows at all speeds with an acoustically conservative interface discretization (ACID), in which face densities and enthalpies are constructed so that acoustic compression and expansion are transmitted correctly across the interface; the underlying coupled solution strategy and its linearization were analyzed in [20]. Pressure-based and semi-implicit formulations have since been extended to Baer–Nunziato-type models with generic EOS by Re and Abgrall [22], to all-Mach interfacial bubble dynamics with a coupled implicit pressure–temperature solve and the Noble–Abel stiffened-gas (NASG) EOS by Saade, Lohse, and Fuster [24] building on [23], to real-fluid semi-implicit two-phase solvers in the subsonic regime by Urbano, Bibal, and Tanguy [25], to a hybrid four-equation multicomponent solver with NASG thermodynamics by Deng et al. [26], and to a linearly implicit all-Mach scheme for the full seven-equation model by Battisti and Boscheri [27]. In parallel, fully conservative energy-consistent flux corrections that approximately preserve pressure equilibrium in multicomponent flows have been proposed by Terashima and co-workers [28], and thermodynamically consistent interface-sharpening formulations for generic five-equation systems have been developed by He and Tan [29], Zhao et al. [30], and He and Zhao [31], the last of which introduces compact closed-form closures expressed through thermal EOS derivatives. Interface-capturing reconstructions such as THINC [32], its boundary-variation-diminishing (BVD) combination with MUSCL [33], and algebraic compressive schemes such as CICSAM [34] provide low-diffusion transport of the volume fraction in this class of methods.

Each of the ingredients above addresses part of the problem, but, to the best of our knowledge, no published method for the five-equation model simultaneously provides (i) a time step restricted only by the material velocity, (ii) thermodynamic closure for nonideal EOS beyond the stiffened gas with closed-form derivatives, and (iii) discrete preservation of pressure–velocity equilibrium at interfaces together with an energy-consistent face flux. The acoustic-CFL-free five-equation schemes [11,12,15] are restricted to stiffened gases and conservative variables; the general-EOS semi-implicit schemes [16,17] are single-phase; the pressure-based interfacial methods [19,24,26] employ mixture or four-equation models that do not conserve the mass of each phase separately or do not carry independent phase temperatures; and the energy-consistent flux corrections [28] have been developed in fully explicit settings.

This paper describes a one-dimensional method that combines these properties within a single formulation. The main elements are as follows. First, the solver evolves the primitive set $\mathbf{W} = (\alpha_1, T_1, T_2, u, p)$, i.e., the volume fraction, one temperature per phase, the velocity, and the pressure. We emphasize that the two temperatures are used as a *parameterization* of the standard mechanical-equilibrium five-equation model (the model itself is unchanged), but this choice has three practical consequences: all EOS evaluations become closed-form functions $\rho_k(p, T_k)$ and $e_k(p, T_k)$ for the ideal-gas, stiffened-gas, and NASG families; the four thermodynamic derivatives $(\partial\rho/\partial p)_T$, $(\partial\rho/\partial T)_p$, $(\partial e/\partial p)_T$, $(\partial e/\partial T)_p$ are available in closed form, so that the $5\times5$ Jacobian $\partial\mathbf{U}/\partial\mathbf{W}$ of the conservative variables is analytic and no EOS inversion is required inside the implicit operator; and face states can be re-evaluated *on the EOS surface* from face values of $(p, T_k)$, in the spirit of the ACID discretization [19] but formulated in $(p,T)$ space, which avoids the interpolation of mixture density or energy that seeds interface pressure oscillations [6]. Second, time integration uses an IMEX splitting in which the nonlinear material transport is advanced by an explicit strong-stability-preserving Runge–Kutta composition [40] and the acoustic subsystem, with coefficients frozen at the stage anchor, is advanced by a single linearized implicit solve; the design of this integrator is guided by a discrete amplification analysis that also identifies, and motivates the suppression of, a pressure odd–even decoupling mode. Third, the total-energy flux is written in terms of energy coefficients associated with the phase-mass and volume-fraction fluxes, following the pressure-equilibrium-consistent construction of [28], with a path-consistent secant form that satisfies the corresponding discrete energy identity to machine precision. Fourth, the volume fraction and the primitive fields are transported by a topology-adaptive combination of a TVD-limited MUSCL–Hancock reconstruction, supplemented by a local maximum-principle bound, and interface-compressive schemes (THINC/BVD [32,33] and CICSAM [34]), selected by the local volume-fraction topology rather than by problem-specific switches, together with a low-Mach-consistent material flux of SLAU2 type [35,36].

We deliberately restrict this paper to one space dimension. The purpose is to document the formulation and to assess its behavior on a broad, strict validation suite under a single set of numerical parameters; the multidimensional extension, which requires no new thermodynamic ingredients but a block-structured implicit acoustic solver, is left to future work. We also do not claim an asymptotic-preserving property in the formal sense; the low-Mach behavior of the scheme is assessed numerically.

The paper is organized as follows. Section 2 presents the governing model and the thermodynamic closure, including the temperature-based parameterization and the analytic Jacobian. Section 3 describes the numerical method: the flux splitting, the time integrator and its amplification analysis, the face thermodynamics, the energy-consistent flux, the interface transport, and the positivity treatment. Section 4 defines the validation suite and reports the results. Section 5 summarizes conclusions and limitations. Closed-form EOS derivatives, the Jacobian entries, and the primitive-recovery procedure are collected in the appendices.

# 2. Governing model and thermodynamic closure

## 2.1. Five-equation model

We consider two immiscible compressible fluids, indexed by $k = 1, 2$, sharing a single velocity $u$ and a single pressure $p$. The one-dimensional five-equation model reads

$$\frac{\partial (\alpha_1 \rho_1)}{\partial t} + \frac{\partial (\alpha_1 \rho_1 u)}{\partial x} = 0, \qquad (1)$$

$$\frac{\partial (\alpha_2 \rho_2)}{\partial t} + \frac{\partial (\alpha_2 \rho_2 u)}{\partial x} = 0, \qquad (2)$$

$$\frac{\partial (\rho u)}{\partial t} + \frac{\partial (\rho u^2 + p)}{\partial x} = 0, \qquad (3)$$

$$\frac{\partial (\rho E)}{\partial t} + \frac{\partial \left[ (\rho E + p) u \right]}{\partial x} = 0, \qquad (4)$$

$$\frac{\partial \alpha_1}{\partial t} + \frac{\partial (\alpha_1 u)}{\partial x} = \left( \alpha_1 + D_1 \right) \frac{\partial u}{\partial x}, \qquad (5)$$

which is the flux form, adopted here so that the volume fraction shares the discrete face velocities of the material fluxes, of the familiar advective equation $\partial_t \alpha_1 + u\, \partial_x \alpha_1 = D_1\, \partial_x u$,

with $\alpha_k$ the volume fractions ($\alpha_1 + \alpha_2 = 1$), $\rho_k$ the phase densities, $\rho = \alpha_1 \rho_1 + \alpha_2 \rho_2$ the mixture density, $E = e + u^2/2$ the specific total energy, and $\rho e = \alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2$ the mixture internal energy under the isobaric closure $p_1 = p_2 = p$. Two closures for the compaction coefficient $D_1$ are supported. The Allaire–Clerc–Kokh closure [4] sets $D_1 = 0$, so that $\alpha_1$ is passively advected; the Kapila closure [3] sets

$$D_1 = \frac{\alpha_1 \alpha_2 \left( \rho_2 c_2^2 - \rho_1 c_1^2 \right)}{\alpha_2 \rho_1 c_1^2 + \alpha_1 \rho_2 c_2^2}, \qquad (6)$$

which accounts for the differential compaction of the phases in compression and expansion. The mixture sound speed associated with the Kapila closure is the Wood speed [44],

$$\frac{1}{\rho c^2} = \frac{\alpha_1}{\rho_1 c_1^2} + \frac{\alpha_2}{\rho_2 c_2^2}, \qquad (7)$$

which is non-monotone in $\alpha_1$ and attains values far below either pure-phase sound speed in mixture regions; this property is relevant to the time-step definitions and to the acoustic test cases of Section 4. All computations reported here use the Allaire closure unless stated otherwise, with the Kapila closure exercised as an option (Section 3.7).

## 2.2. Equations of state

Each phase is closed by a Noble–Abel stiffened-gas (NASG) EOS [9], which contains the ideal gas and the stiffened gas as special cases and represents liquids at moderate pressures with good accuracy. In terms of pressure and temperature, the specific volume and specific internal energy of phase $k$ are

$$v_k(p, T_k) = \frac{(\gamma_k - 1) c_{v,k} T_k}{p + p_{\infty,k}} + b_k, \qquad \rho_k = \frac{1}{v_k}, \qquad (8)$$

$$e_k(p, T_k) = \frac{p + \gamma_k p_{\infty,k}}{p + p_{\infty,k}} \, c_{v,k} T_k + q_k, \qquad (9)$$

where $\gamma_k$, $p_{\infty,k}$, $b_k$ (covolume), $c_{v,k}$, and $q_k$ are constants. Setting $b_k = q_k = 0$ recovers the stiffened gas, and additionally $p_{\infty,k} = 0$ recovers the ideal gas. The essential feature exploited throughout this work is that both state functions (8)–(9), and their first derivatives

$$\left( \frac{\partial \rho_k}{\partial p} \right)_{T},\quad \left( \frac{\partial \rho_k}{\partial T} \right)_{p},\quad \left( \frac{\partial e_k}{\partial p} \right)_{T},\quad \left( \frac{\partial e_k}{\partial T} \right)_{p}, \qquad (10)$$

are available in closed form (Appendix A). The isentropic phase sound speed follows from the derivative set (10) as

$$c_k^2 = \left[ \left( \frac{\partial \rho_k}{\partial p} \right)_T + \left( \frac{\partial \rho_k}{\partial T} \right)_p \Theta_k \right]^{-1}, \qquad \Theta_k = \frac{\dfrac{p}{\rho_k^2} \left( \dfrac{\partial \rho_k}{\partial p} \right)_T - \left( \dfrac{\partial e_k}{\partial p} \right)_T}{\left( \dfrac{\partial e_k}{\partial T} \right)_p - \dfrac{p}{\rho_k^2} \left( \dfrac{\partial \rho_k}{\partial T} \right)_p}, \qquad (11)$$

which reduces to $c_k^2 = \gamma_k R_k T_k$ for an ideal gas. Because the framework interacts with the EOS only through the state functions and the derivative set (10), fluids described by Mie–Grüneisen-type or cubic EOS can be accommodated by supplying the corresponding derivatives, analytically or numerically; only the three analytic families above are exercised in this paper.

## 2.3. Temperature-based primitive parameterization

The solver evolves the primitive vector and the conservative vector

$$\mathbf{W} = \left( \alpha_1,\; T_1,\; T_2,\; u,\; p \right)^{\mathrm{T}}, \qquad \mathbf{U} = \left( \alpha_1 \rho_1,\; \alpha_2 \rho_2,\; \rho u,\; \rho E,\; \alpha_1 \right)^{\mathrm{T}}. \qquad (12)$$

In the five-equation model with a single pressure the phase temperatures are not additional degrees of freedom: given $\mathbf{U}$, the pair $(T_1, T_2)$ is determined (together with $p$) by the phase masses and the mixture energy. The set $\mathbf{W}$ is therefore a parameterization, not a thermal-nonequilibrium extension of the model. Its advantage is structural. The map $\mathbf{W} \mapsto \mathbf{U}$ is an explicit composition of the EOS functions (8)–(9), and its Jacobian

$$\mathbf{A}(\mathbf{W}) = \frac{\partial \mathbf{U}}{\partial \mathbf{W}} \in \mathbb{R}^{5\times 5} \qquad (13)$$

is analytic, with entries assembled from the derivative set (10) (Appendix B); it is validated against high-order finite differences in the test suite. The inverse map $\mathbf{U} \mapsto \mathbf{W}$ requires the solution of a $3\times 3$ nonlinear system for $(p, T_1, T_2)$ at fixed $\alpha_1$, namely

$$\alpha_1 \rho_1(p, T_1) = U_1, \qquad \alpha_2 \rho_2(p, T_2) = U_2, \qquad \sum_k \alpha_k \rho_k(p,T_k)\, e_k(p, T_k) = \rho e, \qquad (14)$$

which is solved by a safeguarded Newton iteration with the analytic Jacobian (Appendix C); in near-pure cells ($\alpha_k$ below a threshold) the vanishing phase is reconstructed from a single-phase fallback to avoid ill-conditioning. Two consequences of this parameterization are used repeatedly below: (i) the implicit stage never needs to invert an EOS, because all linearizations are expressed through (13); and (ii) spatial face states can be prescribed in $(p, T_1, T_2)$ and mapped through the EOS, so that face densities and energies are EOS-consistent by construction.

# 3. Numerical method

## 3.1. Flux splitting and semi-discrete form

The domain is divided into cells $I_i = [x_{i-1/2}, x_{i+1/2}]$ of uniform width $\Delta x$. System (1)–(5) is written as

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}_E(\mathbf{W})}{\partial x} + \frac{\partial \mathbf{F}_I(\mathbf{W})}{\partial x} = \mathbf{S}(\mathbf{W}), \qquad (15)$$

with the material (explicit) and acoustic (implicit) flux components and the non-conservative volume-fraction source

$$\mathbf{F}_E = \left( \alpha_1 \rho_1 u,\; \alpha_2 \rho_2 u,\; \rho u^2,\; F_{\rho E}^{E},\; \alpha_1 u \right)^{\mathrm{T}}, \qquad \mathbf{F}_I = \left( 0,\; 0,\; p,\; p u,\; 0 \right)^{\mathrm{T}}, \qquad (16)$$

$$\mathbf{S} = \left( 0,\; 0,\; 0,\; 0,\; B_i \, \frac{u_{i+1/2} - u_{i-1/2}}{\Delta x} \right)^{\mathrm{T}}, \qquad B_i = \begin{cases} \alpha_{1,i}, & \text{Allaire}, \\ \alpha_{1,i} + D_{1,i}, & \text{Kapila}, \end{cases} \qquad (17)$$

where $F_{\rho E}^{E}$ is the advective energy flux constructed in Section 3.4 and the face velocities $u_{i\pm 1/2}$ in (17) are those of the material flux, so that the advection and compaction contributions to the volume fraction are discretely compatible. The pressure-work flux is kept in the conservative product form $(p u)_f = p_f u_f$; a split form $p\,\partial_x u + u\,\partial_x p$ was found to violate discrete well-balancedness of uniform states and is not used. This splitting places all genuinely nonlinear transport in $\mathbf{F}_E$, while $\mathbf{F}_I$ carries the terms responsible for acoustic propagation; the eigenvalues of the explicit subsystem are all equal to $u$, so the explicit step is stable under a material CFL condition, and the numerical dissipation of the explicit fluxes is scaled with $|u|$ only: no term proportional to the sound speed enters the explicit operator, which is essential for accuracy at low Mach number [10].

An important discrete design constraint is *uniform-flow consistency* in the sense of Abgrall [6]: for spatially uniform $(u, p)$ and arbitrary $\alpha_1, T_1, T_2$ profiles, every component of the discrete explicit operator, the discrete pressure gradient, and the discrete pressure-work divergence must vanish identically. In the present method this is achieved by evaluating all face thermodynamic states from a single shared set of face values $(\alpha_{1,f}, T_{k,f}, u_f, p_f)$ (Section 3.3) and by the energy-coefficient construction of Section 3.4; the property holds to machine precision (it is enforced as a regression test, with all residual components identically zero in double precision).

## 3.2. Time integration and amplification analysis

**Explicit part.** Within a time step, the material subsystem $\partial_t \mathbf{U} + \partial_x \mathbf{F}_E = \mathbf{S}$ is advanced by the three-stage strong-stability-preserving Runge–Kutta method of Shu–Osher type [40], applied as a convex composition of forward-Euler substeps *in the conservative variables*. Each substep uses MUSCL–Hancock predictor–corrector fluxes (Section 3.5), so that the composed material update is second-order accurate in space and time under a material Courant number $C_u = |u| \Delta t / \Delta x \le 1$. When the pressure and velocity fields are discretely uniform (the pressure-equilibrium, or PE, regime of pure interface advection), the composition reduces to a single material step, which avoids the additional diffusion that repeated convex blending would otherwise introduce into an exactly advected profile.

**Implicit part.** The acoustic subsystem is advanced by a single linearized solve. Linearizing $\mathbf{F}_I$ about the post-transport state (the *stage anchor*) yields the coupled acoustic block for the cell-centered velocity and pressure increments,

$$\frac{\partial u}{\partial t} + \frac{1}{\rho} \frac{\partial p}{\partial x} = 0, \qquad \frac{\partial p}{\partial t} + \rho c^2 \frac{\partial u}{\partial x} = 0, \qquad (18)$$

with $\rho$ and $\rho c^2$ frozen at the anchor ($c$ the Wood speed (7)). The block is discretized in time with a Crank–Nicolson-type weighting and solved for $(u^{n+1}, p^{n+1})$; the pressure and velocity that enter the acoustic residual are reconstructed at the faces by the fifth-order scheme of Section 3.6, and elimination of the velocity yields a scalar Helmholtz-type banded system for the pressure (a seven-point stencil of half-bandwidth three, set by that reconstruction, and cyclic-banded for periodic boundaries), which is solved directly. The mixture energy and phase masses are then updated with the conservative pressure-work flux evaluated from the implicit face values. Because the acoustic solve is linear, its cost per step is comparable to one explicit stage; because it is unconditionally stable, the overall time step is restricted only by the material Courant number. The Jacobian of the acoustic block with respect to $\mathbf{W}$ is assembled from the analytic transformation (13); a Schur-complement variant that eliminates $(\delta\alpha_1, \delta T_1, \delta T_2)$ first and solves a reduced $(\delta u, \delta p)$ system is used in the fully nonlinear backward-Euler configuration of the solver and gives identical spectra in the linear regime. The fifth-order reconstruction of the acoustic faces enters the linearized solve through a straight-through Jacobian: the residual value keeps the nonlinear reconstruction, while the derivative used to assemble the operator is taken as the linear-optimal fifth-order stencil. This choice has two reasons. The adaptive-weight derivatives are ill-defined on equilibrium states that are flat to rounding error, where they react to noise; and taking the linear-optimal stencil keeps the implicit operator's stencil consistent with the assembled Jacobian, since the reconstruction reaches $i \pm 3$ and the Jacobian covers that band.

**Energy-closure regimes.** After the acoustic solve, the total energy can be closed in more than one discretely consistent way, and the appropriate choice depends on the local wave content. The solver selects among (i) a conservative implicit-energy closure, appropriate when shocks dominate and correct Rankine–Hugoniot jump speeds are required; (ii) a pressure-work-consistent closure, in which the energy update uses the face acoustic work $(p u)_f$, appropriate for collocated pressure and material jumps; and (iii) a conservative pressure-recovery closure restricted to compression regions. The selection is made from the *initial-state wave topology* (presence and orientation of pure-phase regions, impedance contrast, pressure jumps), not from a per-case switch; the same automaton is active in every computation of Section 4, and its effect is isolated in the ablation study (Section 4.6).

**Amplification analysis and integrator selection.** The integrator was selected by direct measurement of the one-step amplification operator. For a frozen configuration $\mathbf{q}^n$ (all cell values of $\mathbf{W}$), the map $\mathbf{q}^n \mapsto \mathbf{q}^{n+1}$ defined by one full time step is differentiated algorithmically to obtain $\mathbf{A} = \partial \mathbf{q}^{n+1} / \partial \mathbf{q}^{n}$, and the spectral radius $\rho(\mathbf{A})$ and leading eigenvectors are computed. Around a pressure-equilibrium state containing a volume-fraction jump between air and water (a discrete stationary interface), a classical two-stage IMEX Runge–Kutta integrator of ARS(2,2,2) type [41,42] exhibits $\rho(\mathbf{A}) \approx 8.8$ at an acoustic-scale time step (i.e., violent instability), whereas the single-stage linearized implicit update yields $\rho(\mathbf{A}) \approx 1.0009$ on the same configuration. The instability of the multi-stage integrator is traceable to the accumulation, across stages, of departures from the pressure-equilibrium manifold: each stage re-evaluates the stiff operator at a state that the previous stage has displaced from equilibrium, and the displacement is amplified geometrically. We therefore adopt the single-anchor construction above and treat higher-order-in-time acoustic integration as an open problem for this class of models; the material part remains second-order. The residual near-neutral modes of the adopted integrator ($|\lambda| \approx 1.0009$ in the configuration above) are pure-pressure odd–even (checkerboard) modes with no imprint on $\alpha_1$, $T_k$, or $u$; their treatment is described in Section 3.6.

## 3.3. EOS-consistent face thermodynamics

Face states for the material fluxes are constructed in $(p, T)$ space. At each face $f = i + 1/2$, the primitive fields are reconstructed (Section 3.5) to obtain $\alpha_{1,f}$, $T_{1,f}$, $T_{2,f}$ (upwind-biased) and $u_f$, $p_f$ (centered, with the pressure dissipation of Section 3.6). The face phase densities, internal energies, and sound speeds are then *re-evaluated from the EOS*,

$$\rho_{k,f} = \rho_k\left(p_f, T_{k,f}\right), \qquad e_{k,f} = e_k\left(p_f, T_{k,f}\right), \qquad \rho_f = \alpha_{1,f}\, \rho_{1,f} + \alpha_{2,f}\, \rho_{2,f}, \qquad (19)$$

so that every face state lies exactly on the EOS surface of each fluid. No mixture density, mixture energy, or conservative variable is interpolated across the face. This is the ACID principle of Denner et al. [19] (construct face thermodynamics so that the interface behaves as a contact discontinuity for the acoustic field), transposed to the $(p, T)$ parameterization, where it takes a particularly simple form: EOS-consistency of the face state is automatic, and the same face values $(p_f, T_{k,f}, \alpha_{1,f})$ feed every flux component, which is what makes the uniform-flow identity of Section 3.1 hold exactly. The explicit fluxes assembled from (19) are upwinded with a local Lax–Friedrichs coefficient proportional to $|u|$ only, $a_f = \max\left( |u_L|, |u_R| \right) + \varepsilon_u$, consistent with the material character of the explicit subsystem.

## 3.4. Energy-consistent advective energy flux

The advective energy flux requires care: if $F_{\rho E}^{E}$ is built independently of the phase-mass and volume-fraction fluxes, the discrete energy equation is inconsistent with the discrete mass and volume-fraction equations at an interface, and pressure equilibrium is destroyed at the rate of the inconsistency. Following the pressure-equilibrium-consistent construction of Terashima et al. [28], we write the internal-energy part of the advective flux as a linear combination of the discrete phase-mass fluxes $F_{q_1}, F_{q_2}$ and the volume-fraction flux $F_{\alpha}$,

$$F_{\rho e}^{E} = \chi_1 F_{q_1} + \chi_2 F_{q_2} + \chi_{\alpha} F_{\alpha}, \qquad (20)$$

with differential energy coefficients evaluated at the EOS-consistent face state (19),

$$\chi_k = e_{k,f} + \rho_{k,f} \left. \frac{(\partial e_k / \partial T)_p}{(\partial \rho_k / \partial T)_p} \right|_{f}, \qquad \chi_{\alpha} = \left. \rho_{2,f}^2 \frac{(\partial e_2/\partial T)_p}{(\partial \rho_2/\partial T)_p} \right|_f - \left. \rho_{1,f}^2 \frac{(\partial e_1/\partial T)_p}{(\partial \rho_1/\partial T)_p} \right|_f, \qquad (21)$$

where the ratio $(\partial e_k/\partial T)_p / (\partial \rho_k/\partial T)_p = (\partial e_k / \partial \rho_k)_p$ is the isobaric energy–density slope; a single-phase fallback ($\chi_k \to e_{k,f}$, $\chi_\alpha \to 0$) is applied when $|(\partial \rho_k/\partial T)_p|$ degenerates or the face is pure in one phase. The differential coefficients (21) are exact for infinitesimal jumps. For finite jumps we also implement a *path-consistent secant* form: defining the face internal-energy potential $g(q_1, q_2, \alpha; p_f) = q_1 e_1(q_1/\alpha, p_f) + q_2 e_2(q_2/(1-\alpha), p_f)$ with $q_k = \alpha_k \rho_k$, secant coefficients $(\bar\chi_1, \bar\chi_2, \bar\chi_\alpha)$ are constructed along a three-segment path from the left to the right face state such that the discrete identity

$$g_R - g_L = \bar\chi_1 (q_{1,R} - q_{1,L}) + \bar\chi_2 (q_{2,R} - q_{2,L}) + \bar\chi_{\alpha} (\alpha_R - \alpha_L) \qquad (22)$$

holds exactly in floating-point arithmetic. Identity (22) removes the residual mismatch between the energy flux and the mass/volume-fraction fluxes that the differential coefficients leave at finite interface jumps, which we have found to be a slow but persistent source of pressure-equilibrium drift in long computations. Both forms are available; the computations of Section 4 use the differential form with the secant form assessed in the ablation study.

## 3.5. Reconstruction and interface transport

**Primitive reconstruction.** The fields $T_1, T_2, u, p$ (and, where used, the phase densities) are reconstructed at faces with a second-order TVD-limited scheme of MUSCL type [37,38],

$$\phi_f = \phi_i + \tfrac{1}{2}\, \psi(r_i)\, (\phi_{i+1} - \phi_i), \qquad r_i = \frac{\phi_i - \phi_{i-1}}{\phi_{i+1} - \phi_i}, \qquad (23)$$

where $\psi$ is a standard flux limiter (the superbee limiter [38] in the reference configuration). Two additional bounds are applied multiplicatively: a *local maximum-principle bound* that clips $\phi_f$ to the range of the three-cell neighborhood, in the spirit of the multi-dimensional limiting process of Kim and Kim [39], and a Hancock-type time-centering factor $(1 - C_u)$ applied to the reconstructed increment, which renders the explicit substep second-order in time and non-oscillatory up to $C_u = 1$. We refer to this combination as the T-MLP-u reconstruction; the ablation study shows that both the limiter choice and the additional bounds are active ingredients rather than safety redundancies.

**Volume-fraction transport.** Low-diffusion transport of $\alpha_1$ is obtained by selecting, face by face, between a smooth-profile scheme and an interface-compressive scheme, according to the local *volume-fraction topology*: when the face lies in a resolved sharp transition between near-pure cells (adjacent cells within a pure-phase tolerance of $0$ or $1$, or a thin mixed layer), a compressive CICSAM-type flux [34] is applied; elsewhere, the MUSCL–Hancock reconstruction above is used. A THINC-based variant with boundary-variation-diminishing (BVD) selection [32,33] (choosing between the hyperbolic-tangent and MUSCL candidates so as to minimize the cell-boundary jump) is implemented and gives closely similar results; the topology-based selector is used in the reference configuration because it avoids the evaluation of both candidates at every face. We emphasize that the selector depends only on the local discrete $\alpha_1$ profile, never on the identity of the test case.

**Material face velocity.** At low Mach number, a purely centered face velocity produces pressure–velocity decoupling, while acoustic-scale upwinding is over-dissipative [10]. The material face velocity therefore includes a pressure-difference coupling of SLAU2 type [35,36],

$$u_f = \tilde{u}_f - \chi(\hat{M}) \, \frac{p_R - p_L}{\bar\rho \bar{c}}, \qquad \chi(\hat{M}) = (1 - \hat{M})^2, \qquad \hat{M} = \min\left(1, \frac{\sqrt{(u_L^2+u_R^2)/2}}{\bar{c}}\right), \qquad (24)$$

where $\tilde u_f$ is a density-weighted average of the reconstructed cell velocities and $\bar\rho, \bar c$ are face means. The scaling $\chi(\hat M)$ activates the coupling at low Mach number and removes it near sonic conditions, where the implicit pressure block provides the coupling instead; the pressure flux itself remains entirely in the implicit operator, so no double counting of the pressure gradient occurs. Term (24) plays the role of the momentum-weighted (Rhie–Chow [21]) interpolation of pressure-based methods, adapted to the present splitting.

## 3.6. Acoustic face reconstruction, pressure dissipation, and positivity

**Acoustic face reconstruction.** The face states that enter the acoustic subsystem are the pressure $p$ and the velocity $u$, and their reconstruction sets the dispersion and the dissipation of the transmitted acoustic field. On material-clean faces both the explicit acoustic face evaluation and the implicit acoustic residual reconstruct $p$ and $u$ componentwise with a fifth-order weighted essentially non-oscillatory (WENO) interpolation [45], using the optimal linear weights $(1/10, 6/10, 3/10)$, the Jiang–Shu smoothness indicators, and a dimensionless relative regularization $\varepsilon$ in the nonlinear weights, and evaluate the $Z$-weighted linearized acoustic Riemann flux (mixture impedance $Z = \rho c$) at the reconstructed left and right states. A face is reconstructed at fifth order only when its full six-cell stencil lies within a single resolved pure material, tested by a topology mask that requires all six cells to lie within a pure-phase tolerance $\max(\alpha_{\text{pure}}, \varepsilon_{\text{mach}}^{1/4})$ of $\alpha_1 = 0$ or $\alpha_1 = 1$; a face whose stencil crosses a material jump falls back to the exact two-state first-order acoustic Riemann value. All constants above are fixed by the reconstruction and the equations of state, so no user-tuned coefficient is introduced. The implicit path uses the straight-through Jacobian of Section 3.2: the residual value keeps the nonlinear reconstruction, while the operator is assembled from the linear-optimal fifth-order stencil.

The fifth-order reconstruction was adopted after the second-order reconstruction of the acoustic faces was found to leave a low-amplitude artifact at the strongest impedance contrast of the suite. With a second-order TVD-limited MUSCL reconstruction of the acoustic faces, the air–water benchmark (case 07_B; impedance ratio about $3600{:}1$) develops a low-amplitude, marginally resolved (eight to ten cells) pressure packet adjacent to the interface. Each step's reconstruction is TVD-admissible, but its dispersive error at four to five points per wavelength accumulates phase-coherently over the roughly $1600$-step run, and any wavenumber-blind dissipation strong enough to remove the packet also damps the physical pulse, which shares the same band. Reconstructing the face states at fifth order removes the packet at the source: the strict total-variation guard value drops from $0.537$ to $0.204$ (limit $0.30$) while the transmitted-peak amplitude ratio stays at $1.00$.

**Odd–even pressure mode.** The amplification analysis of Section 3.2 identifies the least-damped modes of the linearized scheme as pure-pressure odd–even oscillations (alternating sign pattern in $p$, no imprint on the other fields), a collocated-grid artifact familiar from pressure-based methods. In the backward-Euler analysis integrator used for that study (Sections 3.2 and 3.7) the mode is suppressed by a fourth-difference (biharmonic) face-pressure dissipation in the implicit block,

$$p_f = \frac{p_L + p_R}{2} - \frac{D}{8} \left( -p_{i-1} + 3 p_i - 3 p_{i+1} + p_{i+2} \right), \qquad (25)$$

with $D = 0.02$. The stencil (25) is the four-point analogue of a Rhie–Chow pressure-velocity coupling; being a fourth difference, it vanishes to second order on smooth fields and does not affect the formal accuracy of the pressure solution. The production integrator does not use (25): it advances the acoustic subsystem with the $Z$-weighted acoustic-Riemann face closure described above, whose upwind dissipation removes the odd–even mode intrinsically, so that no tuned dissipation coefficient enters any production computation.

**Positivity.** Admissibility of the explicit update ($\alpha_k \rho_k > 0$, $\alpha_1 \in (\varepsilon_\alpha, 1 - \varepsilon_\alpha)$) is enforced by a layered flux limiter: the high-order explicit flux is blended, $\mathbf{F}_f = \theta_f \mathbf{F}_f^{HO} + (1 - \theta_f) \mathbf{F}_f^{LO}$ with $\theta_f \in [0,1]$ halved on faces adjacent to a violating cell until the candidate update is admissible (at most a fixed number of sweeps). The essential design choice is the low-order flux: a conventional Rusanov flux with dissipation proportional to $(\mathbf{U}_R - \mathbf{U}_L)$ re-introduces exactly the conservative-variable interpolation that destroys pressure equilibrium at interfaces. The low-order flux is therefore taken as the *same-face-state upwind flux*, the flux (16) evaluated at the shared EOS-consistent face state with full upwinding and no conservative difference term, so that the blending cannot move the scheme off the pressure-equilibrium manifold. If the subsequent implicit stage fails to recover admissible primitives, the step is repeated with a reduced implicit weight (a bounded backtracking sequence); in the computations of Section 4 this fallback is exercised only in the strongest shock cases.

## 3.7. Options exercised as variants

Three ingredients are implemented and tested but are not part of the reference configuration: (i) the Kapila closure (6), with the compaction source integrated with face velocities of the implicit block (a semi-implicit treatment) or a path-conservative treatment restricted to resolved mixture stencils; (ii) a fully nonlinear backward-Euler integrator, in which the complete system (15) is solved by a Newton iteration with the analytic Jacobian and Schur-complement elimination (this configuration is the basis of the amplification measurements of Section 3.2 and remains useful as an analysis tool); and (iii) a pressure-equilibrium tangent projection that removes the component of the discrete residual normal to the equilibrium manifold, which is redundant in the reference configuration but documents the sensitivity of the model to equilibrium drift. These variants are reported only where explicitly noted.

# 4. Numerical results

## 4.1. Test suite, configuration, and metrics

Table 1 lists the EOS parameters. Table 2 summarizes the thirteen test cases, which are drawn from a strict internal validation suite and span five groups: (A) pressure-equilibrium (PE) preservation, (B) linear gas–liquid acoustics, (E) gas–liquid shock tubes, (T) advection with large phase-temperature contrast, and (H) strong shocks at Mach 10. Every computation uses the *same* reference configuration described in Section 3 (SSP3 material integrator with the linearized implicit acoustic solve, T-MLP-u/superbee reconstruction, topology-adaptive volume-fraction transport, SLAU2-type material face velocity, a $Z$-weighted acoustic-Riemann face closure with fifth-order WENO reconstruction of pressure and velocity on material-clean faces (Section 3.6), and the topology-based energy-closure automaton), with no per-case parameter adjustment. Errors are measured against exact solutions (advection, Riemann problems) or against linear acoustic theory (reflection–transmission), with

$$L_2(\phi) = \left[ \frac{\sum_i (\phi_i - \phi_i^{\mathrm{ref}})^2}{\sum_i (\phi_i^{\mathrm{ref}})^2} \right]^{1/2}, \qquad L_\infty(\phi) = \frac{\max_i |\phi_i - \phi_i^{\mathrm{ref}}|}{\max_i |\phi_i^{\mathrm{ref}}|}. \qquad (26)$$

**Table 1.** EOS parameters used in this work. The NASG water parameterization follows [9]. Stiffened-gas (SG) water is used only in the ultra-low-Mach case 03_B of the all-speed sweep (Section 4.6); every other water phase (cases 02, 05, 07 air–water, 13, 14, 15, 24, 25) uses the NASG parameters.

| Fluid | EOS | $\gamma$ | $p_\infty$ (Pa) | $b$ (m^3^/kg) | $c_v$ (J/(kg K)) | $q$ (J/kg) |
|---|---|---|---|---|---|---|
| Air | ideal | 1.400 | 0 | 0 | 717.5 | 0 |
| Helium | ideal | 1.667 | 0 | 0 | 3120.0 | 0 |
| Argon | ideal | 1.660 | 0 | 0 | 312.0 | 0 |
| Water (NASG) | NASG | 1.187 | $7.028\times10^{8}$ | $6.61\times10^{-4}$ | 3610.0 | $-1.177788\times10^{6}$ |
| Water (SG) | stiffened gas | 4.4 | $6.0\times10^{8}$ | 0 | 474.2 | 0 |

**Table 2.** Validation matrix. All cases use the reference configuration of Section 3. $N$ is the grid resolution; the time step is either a fixed $\Delta t$ (s) or set by the stated CFL (material Courant) number. The acoustic Courant number exceeds unity in every case; Section 4.2 quantifies it for 02_A.

| ID | Description | Fluids / EOS | $N$ | $\Delta t$ or CFL | $t_{\mathrm{end}}$ (s) | Reference solution |
|---|---|---|---|---|---|---|
| 01_A | Static air–water interface, long integration | ideal / NASG | 100 | $\Delta t = 10^{-2}$ | 1.0 | exact (steady) |
| 02_A | PE interface advection, $u = 1$ m/s, periodic | ideal / NASG | 100 | $\Delta t = 10^{-2}$ | 1.0 | exact advection |
| 04_B | Sinusoidal acoustics in air, 2 kHz | ideal / ideal | 500 | CFL $= 0.4$ | $2.3\times10^{-3}$ | linear acoustics |
| 05_B | Sinusoidal acoustics in water, 6 kHz | ideal / NASG | 400 | CFL $= 0.4$ | $5.10\times10^{-4}$ | linear acoustics |
| 07_B-1 | Acoustic pulse, air → water interface | ideal / NASG | 400 | CFL $= 0.4$ | $1.55\times10^{-3}$ | linear R/T theory |
| 07_B-2 | Acoustic pulse, helium → air interface | ideal / ideal | 400 | CFL $= 0.4$ | $1.513\times10^{-3}$ | linear R/T theory |
| 07_B-3 | Acoustic pulse, argon → air interface | ideal / ideal | 400 | CFL $= 0.4$ | $2.02\times10^{-3}$ | linear R/T theory |
| 13_E | Shock tube, high-$p$ air / water (1 GPa) | ideal / NASG | 800 | CFL $= 0.30$ | $6.7\times10^{-4}$ | exact Riemann |
| 14_E | Shock tube, high-$p$ water / air | NASG / ideal | 800 | CFL $= 0.25$ | $2.29\times10^{-4}$ | exact Riemann |
| 15_E | Air–water cavitation (double rarefaction) | ideal / NASG | 400 | CFL $= 0.01$ | $9.5\times10^{-4}$ | reference |
| 16_T | Sharp interface, hot gas / cold liquid advection | ideal / NASG | 100 | $\Delta t = 5\times10^{-4}$ | 0.1 | exact advection |
| 17_T | Smooth Gaussian $\alpha$ pulse, large $\Delta T$ | ideal / NASG | 550 | $\Delta t = 10^{-4}$ | 0.1 | exact advection |
| 18_T | Mixed smooth $\alpha$ and thermal waves | ideal / NASG | 550 | $\Delta t = 1/11000$ | 0.1 | exact advection |
| 24_H | Mixture shock, $M_s = 10$, homogeneous | ideal / NASG | 400 | CFL $= 0.10$ | $0.7/V_s$ | exact mixture Riemann |
| 25_H | Mach-10 shock into air–water | ideal / NASG | 400 | CFL $= 0.30$ | $t_{\mathrm{hit}} + 2.42\times10^{-4}$ | reference solution |

Acceptance thresholds are fixed per case family and are not adjusted between runs. For the interface-acoustic family (07_B) the pressure-error limits are $L_2(p) \le 0.216$ and $L_\infty(p) \le 0.756$ (air–water) or $0.81$ (gas–gas), with a correlation floor of $0.88$, a transmitted-energy fraction floor of $0.76$, a peak-amplitude band of $0.85$–$1.10$ for the liquid transmission and $0.80$–$1.13$ for the gas cases, a wave-symmetry limit of $0.38$, a peak-location tolerance of three cells, and a smoothed-pressure local total-variation-excess limit of $0.80$. The pressure-equilibrium family uses pressure and velocity tolerances of $10^{-10}$ with a minimum range ratio of $0.85$, and the shock-tube family uses a shock-location tolerance of three cells with density-profile $L_2$ and correlation limits of $0.03$ and $0.99$. The same configuration and the same thresholds were applied to every case in Table 2.

## 4.2. Pressure-equilibrium interface advection at large acoustic Courant number

Case 02_A advects an isolated water–air interface (NASG water, ideal-gas air) at $u = 1$ m/s across a periodic domain on $N = 100$ cells with a fixed time step $\Delta t = 10^{-2}$ s; the interface completes one full domain transit in 100 steps. With $\Delta x = 10^{-2}$ m the material Courant number is $C_u = |u|\Delta t/\Delta x = 1.0$, while the acoustic Courant number based on the NASG-water sound speed ($c \approx 1.57\times10^{3}$ m/s) is $C_a = (|u|+c)\Delta t/\Delta x \approx 1.57\times10^{3}$: the fastest acoustic waves cross of order $10^{3}$ cells per step and are carried entirely by the implicit block. This configuration is the elementary test of the claims of Sections 3.3–3.4: any inconsistency between the mass, volume-fraction, and energy fluxes, or any EOS interpolation error, appears as a growing disturbance in $p$ and $u$.

After 100 steps the relative pressure deviation is $L_\infty(p) = 5.821\times10^{-16}$ and the absolute velocity deviation is $L_\infty(u) = 5.329\times10^{-14}$ m/s in the evidence run, which was regenerated in full on 2026-07-14. The volume-fraction and density profiles are advected without change of range (range ratios $1.000$ and $1.0000000000016$) and with correlation coefficients against the exact profile equal to unity to machine precision ($\mathrm{corr}_\alpha = 1.000$, $\mathrm{corr}_\rho = 1.000$); the volume-fraction $L_1$ error ratio is $3.505\times10^{-15}$. The deviations sit at the roundoff floor rather than at a small but finite level, so the equilibrium property is a structural identity of the discretization. The static case 01_A ($u = 0$, no flow) holds the same equilibrium over 100 steps with a relative pressure drift of $4.366\times10^{-14}$, a velocity drift of $1.042\times10^{-12}$ m/s, and a pressure checkerboard measure of $2.609\times10^{-16}$.

![](figures/core_02_A.svg)
![](figures/pressure_equilibrium_preservation.svg)

**Figure 1.** Case 02_A, air/NASG-water interface advected one full transit at $C_a \approx 1.57\times10^{3}$. The upper panels show density (a top-hat, $\rho \approx 1050$ in $[0.4, 0.6]$ and $\approx 0$ outside), velocity held at $1.0$ to within $\pm10^{-13}$, and pressure held at $10^{5}$ Pa to within $\approx 10^{-7}$; the lower panels give the pointwise errors, with $|\Delta\rho|$ reaching $\approx 5\times10^{-11}$ only at the two interface cells, $|\Delta u| \approx 10^{-13}$, and $|\Delta p| \approx 10^{-10}$. The companion panel summarizes the pressure-equilibrium preservation metric, $L_\infty(p) = 5.821\times10^{-16}$.

## 4.3. Acoustic transmission and reflection at gas–liquid interfaces

The single-fluid checks come first. Cases 04_B and 05_B inject a small-amplitude sinusoidal acoustic train onto a $u = 1$ m/s flow and advect it over multiple periods, in air (2000 Hz, $N = 500$) and in NASG water (6000 Hz, $N = 400$). In air the amplitude-rescaled profile error is $L_2 = 0.03366$ with an amplitude ratio of $0.9940$ and a correlation of $0.9988$; the measured wavelength $0.1745$ matches the exact $0.17393$ to $+0.33\%$, and the wave pressure amplitude $4.011$ matches the exact $4.025$. In water the corresponding numbers are $L_2 = 0.008806$, amplitude ratio $0.9947$, correlation $0.99992$, wavelength $0.2600$ against $0.26122$ ($-0.47\%$), and pressure amplitude $15562$ Pa against $15642$ Pa. Both cases hold amplitude and phase to within about $0.6\%$, consistent with the Crank–Nicolson weighting of the implicit block.

![](figures/core_04_B.svg)
![](figures/core_05_B.svg)

**Figure 2.** Single-fluid acoustic wave trains advected on a $u = 1$ m/s flow. Left: air at 2000 Hz ($N = 500$), amplitude ratio $0.9940$. Right: NASG water at 6000 Hz ($N = 400$), amplitude ratio $0.9947$, correlation $0.99992$. Solid line, numerical; dashed line, exact.

Cases 07_B-1/2/3 propagate a Gaussian velocity pulse (amplitude $2\times10^{-2}$ m/s) toward a material interface (air→water, helium→air, argon→air) on $N = 400$ cells and compare reflected and transmitted waves with the linear d'Alembert solution, $R = (Z_2 - Z_1)/(Z_1 + Z_2)$ and $T = 2 Z_2 / (Z_1 + Z_2)$ in the impedances $Z_k = \rho_k c_k$. The air–water pair is the demanding one: the impedance ratio is about $3600$, so an inconsistent interface discretization produces order-one errors in the reflected wave; this benchmark follows the interface-acoustics assessment of [19]. The measured errors and amplitude ratios are listed below.

| Interface pair | $L_2(p)$ | $L_\infty(p)$ | $L_2(u)$ | $L_\infty(u)$ | $\mathrm{corr}(p)$ | $\mathrm{frac}(p)$ | $p$-peak $\Delta$ (cells) | $p$ amp. ratio | $u$ amp. ratio | symmetry |
|---|---|---|---|---|---|---|---|---|---|---|
| air–water | 0.06026 | 0.2331 | 0.02210 | 0.1783 | 0.9933 | 1.000 | 1 | 0.9969 | 0.9803 | 0.05534 |
| helium–air | 0.01059 | 0.06596 | 0.005599 | 0.02719 | 0.9988 | 1.000 | 0 | 0.9745 | 0.9743 | 0.03765 |
| argon–air | 0.002412 | 0.01188 | 0.002970 | 0.01583 | 0.9999 | 1.000 | 0 | 0.9920 | 0.9918 | 0.01618 |

The transmitted-peak amplitude ratios are $0.9969$, $0.9745$, and $0.9920$, all inside the acceptance bands ($0.85$–$1.10$ for the air–water liquid transmission, $0.80$–$1.13$ for the gas cases). The air–water transmitted pressure and velocity peaks are each located within one cell; the helium–air and argon–air peaks are exact to the cell. The wave-symmetry errors ($0.05534$, $0.03765$, $0.01618$) are inside the $0.38$ limit, and all three subcases pass acceptance with zero failures. Relative to the earlier second-order acoustic faces the two gas–gas pairs sharpen by a factor of about two to three in $L_2(p)$ (helium–air $0.02222 \to 0.01059$, argon–air $0.007459 \to 0.002412$), and the air–water error improves from $0.09002$ to $0.06026$.

The air–water subcase is the stiffest impedance jump and historically retained a low-amplitude pressure ringing adjacent to the interface, at the base of the transmitted peak. With the fifth-order WENO reconstruction of the acoustic faces (Section 3.6) its smoothed-pressure local total-variation excess is $0.204$, below the stricter total-variation guard of $0.30$ introduced during internal development; the second-order TVD acoustic faces of the earlier scheme gave $0.537$. The helium–air and argon–air subcases are cleaner still by this measure ($0.05714$ and $0.01177$). All three subcases pass every acceptance criterion, including the tightened total-variation guard.

![](figures/core_07_B.svg)
![](figures/all_speed_interface_acoustic_07_B.svg)

**Figure 3.** Case 07_B, acoustic reflection and transmission at a material interface for three impedance pairs (rows: air–water, helium–air, argon–air; columns: density, velocity, pressure perturbation), $N = 400$. The numerical pressure perturbation tracks the linear reflection–transmission solution in all nine panels; the air–water transmitted peak (near $x \approx 1.13$) is captured within one cell, with the interface-adjacent base ringing suppressed by the fifth-order acoustic reconstruction (Section 3.6), quantified by $L_2(p) = 0.06026$ and $\mathrm{corr}(p) = 0.9933$. The second panel is the same air–water subcase used as the interface-acoustic anchor of the all-speed sweep (Section 4.6). Solid line, numerical; dashed line, exact.

The accuracy of the air–water subcase is insensitive to the time step. On $N = 200$ with CFL of $0.2$, $0.4$, and $0.6$, the pressure error $L_2(p)$ is $0.1506$, $0.1512$, and $0.1576$ and the correlation is $0.9716$, $0.9712$, and $0.9680$, so the error metrics move by less than $5\%$ as the CFL triples. The subcase is marked FAIL at all three CFLs, but the failure is a resolution effect rather than an instability: at $N = 200$ the transmitted peak amplitude is damped to $0.707$–$0.732$, below the $0.85$ acceptance band, whereas the peak-amplitude bar is met at $N = 400$ (Section 4.6). No blow-up occurs at any CFL.

![](figures/cfl_07_AirWater_CFL0p2.svg)
![](figures/cfl_07_AirWater_CFL0p4.svg)
![](figures/cfl_07_AirWater_CFL0p6.svg)
![](figures/acoustic_cfl_sweep.svg)

**Figure 4.** CFL robustness of the air–water subcase at $N = 200$ for CFL $= 0.2$, $0.4$, and $0.6$, with the summary overlay at right. The pressure error stays within $0.1506$–$0.1576$ and the correlation within $0.968$–$0.972$ as the CFL triples; the subcase fails acceptance at $N = 200$ because the transmitted peak is damped to $0.707$–$0.732$, below the $0.85$ band, not because of instability.

## 4.4. Gas–liquid shock tubes

Cases 13_E–15_E are Riemann problems with large initial pressure ratios across gas–liquid interfaces. All three use NASG water. These exercise the interaction of the implicit acoustic block with nonlinear waves, the energy-closure automaton (the conservative closure must activate so that shock speeds satisfy the Rankine–Hugoniot conditions), and the positivity layering at the strong expansion adjacent to the interface.

Case 13_E is a 1 GPa air pocket discharging into water on $N = 800$ (CFL $0.30$). The contact sits at $x = 0.6338$ with a density peak of $6217$ against the exact $6298$ (overshoot ratio $0.01598$, limit $0.05$), and the shock at $x = 1.845$ against the exact $1.84992$, a position error of $1.970$ cells (limit three). Smooth-region relative $L_2$ errors are $0.004066$ (p), $0.003002$ ($\rho$), and $0.008479$ (u); the pressure and density checkerboard measures are $0.003124$ and $0.004761$, and the pressure spans $10^{4}$–$10^{9}$ Pa without oscillation.

Case 14_E is pressurized water discharging into air on $N = 800$ (CFL $0.25$). The contact ($x = 0.83125$ against $0.83228$, $0.8220$ cells) and shock ($x = 0.85875$ against $0.85965$, $0.7222$ cells) form a closely spaced pair that the scheme separates (split-gap ratio $1.005$, band $0.5$–$1.8$); the density peak is $736.5$ against $736.1$ (overshoot ratio $0.0007$, limit $0.001$), and the plateau $L_\infty$ ratio is $0.002436$ (limit $0.03$). The checkerboard measures are $0.0002489$ (p) and $0.007598$ ($\rho$).

Case 15_E is an air–water cavitation problem: two rarefactions pull the fluid apart at $\pm100$ m/s about $x = 0.5$ on $N = 400$. The gas fraction grows from a seed of $0.055$ to a peak of $0.9771$, and the density and pressure fall to $22.93$ and $3.067$ Pa near the center while staying positive. The checkerboard measures are $0.0006779$ (p) and $0.002824$ ($\rho$). The CFL for this case is reduced to $0.01$, because the nominal $0.25$ is unstable for the Kapila $D_1\,\partial_x u$ source at the initial velocity jump; this is a documented limitation of the present source treatment.

![](figures/core_13_E.svg)
![](figures/core_14_E.svg)
![](figures/core_15_E.svg)

**Figure 5.** Gas–liquid shock tubes at final time (solid, numerical; dashed, exact Riemann solution). Left, 13_E (1 GPa air into water, $N = 800$): left rarefaction, density plateau $\approx 6300$, contact at $x \approx 0.63$, and shock at $x \approx 1.85$, with the acoustic impedance stepping $4.05 \to 1.9 \to 2.63$. Center, 14_E (pressurized water into air, $N = 800$): closely spaced contact and shock split, density peak $736$. Right, 15_E (air–water cavitation, $N = 400$): double rarefaction with the gas fraction growing to $0.977$ and near-vacuum pressure $3.07$ Pa held positive.

## 4.5. Temperature-contrast advection and strong shocks

Cases 16_T–18_T advect interfaces and smooth volume-fraction profiles with phase-temperature contrasts of order $10^2$ K under exact pressure equilibrium, with $T_1$ and $T_2$ as primary reconstructed fields. This isolates the temperature-based parameterization: a scheme that transports $(\rho_k, p)$ and reconstructs temperatures a posteriori can preserve pressure equilibrium while distorting the phase-temperature fields. Case 16_T advects a sharp hot-gas/cold-liquid interface ($T_1 = 300$ K, $T_2 = 1200$ K, $u = 10$ m/s, $N = 100$). Pressure and velocity hold equilibrium to $L_\infty(p) = 6.112\times10^{-14}$ and $L_\infty(u) \approx 4.90\times10^{-13}$, the active-phase temperature errors are at the $10^{-12}$ level ($T_1$ $L_\infty$ ratio $2.867\times10^{-12}$, $T_2$ $7.768\times10^{-12}$), and the mixture-temperature error ($L_\infty$ ratio $9.681\times10^{-7}$) is localized to the sharp contact; the volume-fraction range is retained (ratio $1.000$, correlation $0.99999999999997$). Case 17_T advects a smooth Gaussian $\alpha$ pulse ($N = 550$): the peak is retained to a range ratio of $0.9951$, the mixture-temperature $L_\infty$ ratio is $0.004350$, and the smooth-$\alpha$ total-variation excess stays at $2.426\times10^{-4}$. Case 18_T advects mixed smooth $\alpha$ and thermal waves ($N = 550$): pressure holds to $L_\infty(p) = 4.366\times10^{-16}$, the density peak amplitude ratio is $0.9996$ (band $0.98$–$1.02$), the mixture-temperature $L_\infty$ ratio is $0.001790$, and the active-phase high-frequency temperature measures ($2.809\times10^{-4}$ for $T_1$, $5.747\times10^{-4}$ for $T_2$) stay below the $8\times10^{-4}$ guard.

![](figures/core_16_T.svg)
![](figures/core_17_T.svg)
![](figures/core_18_T.svg)

**Figure 6.** Temperature-contrast advection (solid, numerical; dashed, exact). Left, 16_T (sharp hot-gas/cold-liquid interface, $N = 100$): $\alpha_1$ and density are top-hat blocks in $[0.35, 0.65]$ and the mixture temperature is the inverted block (1200 K gas to 300 K liquid); velocity holds at $10$ m/s and pressure at $10^{5}$ Pa, with $|\Delta T|$ and $|\Delta\rho|$ peaking only at the interface cell. Center and right, 17_T and 18_T (smooth Gaussian and mixed thermal waves, $N = 550$): peak retention $0.9951$ and density-peak ratio $0.9996$.

Cases 24_H and 25_H drive $M_s = 10$ shocks through a homogeneous two-phase mixture and into an air–water configuration. Case 24_H ($N = 400$, CFL $0.10$) is run for five water mass fractions $\psi \in \{0, 0.25, 0.5, 0.75, 1\}$. The shock sits at $x \approx 0.80$ (exact $0.80$) in every subcase, with a position error of $0.50$ cells for $\psi \le 0.75$ and $1.50$ cells at $\psi = 1$. The density-profile $L_2$ errors are $0.02172$, $0.01985$, $0.01833$, $0.008273$, and $0.009474$, all below the $0.03$ limit, and the density correlations are $0.99$ or above; the post-shock pressure spans $1.17\times10^{7}$ Pa (pure air) to $7.55\times10^{10}$ Pa (pure water), four decades, without post-shock oscillation. The CFL is reduced to $0.10$ because the hypersonic mixture shock is sensitive to the source and flux time-centering. Case 25_H ($N = 400$, CFL $0.30$) sends a Mach-10 air shock into a water interface, producing reflected and transmitted shocks. The scaled-$L_2$ errors are $0.04811$ (p), $0.004469$ ($\rho$), and $0.02400$ (u) with correlations $0.9969$, $0.99999$, and $0.9990$; the shock, interface, reflected-shock, and transmitted-shock positions are captured to $0.5433$, $0.4528$, $0.7074$, and $0.5433$ cells. This is the noisiest core case, with a pressure checkerboard measure of $0.09952$ (about $10\%$); the solution stays admissible.

![](figures/core_24_H.svg)
![](figures/core_25_H.svg)
![](figures/all_speed_hypersonic_25_H.svg)

**Figure 7.** Mach-10 cases (solid, numerical; dashed, reference). Left, 24_H (homogeneous mixture shock, $N = 400$): five water mass fractions $\psi = 0, 0.25, 0.5, 0.75, 1$, each a single shock fixed at $x \approx 0.80$ with post-shock pressure ranging from $1.17\times10^{7}$ Pa (pure air) to $7.55\times10^{10}$ Pa (pure water). Center, 25_H (Mach-10 air shock into water, $N = 400$): reflected and transmitted shocks captured within about one cell, with a residual $10\%$ pressure checkerboard. Right, the $N = 200$ all-speed hypersonic anchor of 25_H.

## 4.6. Grid refinement, all-speed span, and ablation

**Grid refinement.** The table below and Figure 8 report the representative normalized error on grid sequences for six cases, with observed orders computed from consecutive resolutions. The interface-acoustic case 07_B converges monotonically in $L_2(p)$, from $0.3090$ ($N = 100$) to $0.1512$ ($N = 200$) to $0.06026$ ($N = 400$), with observed orders $1.03$ then $1.33$; the case meets the acceptance thresholds of Section 4.1 only at $N = 400$, where all criteria including the stricter total-variation guard of Section 4.3 are satisfied, while $N = 100$ and $N = 200$ fail on the coarse-grid peak-amplitude limit. The shock-tube case 13_E converges at about $0.7$ order in the smooth-region density error ($0.007635 \to 0.004878 \to 0.003002$ for $N = 200/400/800$), as expected when a shock and contact limit the rate. Case 14_E jumps from a density-plateau $L_\infty$ ratio of $0.8946$ at $N = 200$ (the two-jump split is essentially unresolved) to $0.01666$ and $0.002436$ at $N = 400$ and $800$, an apparent order above five that reflects crossing the resolution threshold; the case passes acceptance only at $N = 800$, because at $N = 400$ the physical two-jump density split is not yet resolved (the density-peak criterion fails there while the plateau ratio itself is already below its limit). Case 18_T improves from $8.866\times10^{-4}$ to $2.531\times10^{-4}$ to $6.981\times10^{-5}$ ($N = 200/400/550$), with an apparent $400 \to 550$ order near four that is inflated by the short resolution interval near the guard floor. The Mach-10 mixture case 24_H converges at sub-first order ($0.03102 \to 0.02593 \to 0.02172$, order $\approx 0.26$), as expected for an $L_2$ error that includes the shock. Case 25_H now converges monotonically in the scaled-$L_2$ pressure error, $0.1001$, $0.04811$, and $0.03988$ at $N = 200/400/800$ (observed orders $1.06$ then $0.27$), so $N = 800$ is the best resolution; the scaled density and velocity errors remain non-monotone, each smallest at $N = 400$. All three 25_H resolutions pass acceptance.

| Case | Metric | $N$ (coarse → fine) | Error sequence | Observed order |
|---|---|---|---|---|
| 07_B | $L_2(p)$, air–water | 100 / 200 / 400 | 0.3090 / 0.1512 / 0.06026 | 1.03, 1.33 |
| 13_E | smooth-region $\rho$ $L_2$ | 200 / 400 / 800 | 0.007635 / 0.004878 / 0.003002 | 0.65, 0.70 |
| 14_E | density-plateau $L_\infty$ ratio | 200 / 400 / 800 | 0.8946 / 0.01666 / 0.002436 | 5.75, 2.77 |
| 18_T | mixture-density $L_1$ ratio | 200 / 400 / 550 | $8.866\times10^{-4}$ / $2.531\times10^{-4}$ / $6.981\times10^{-5}$ | 1.81, 4.05 |
| 24_H | max $\rho$-profile $L_2$ (5 subcases) | 100 / 200 / 400 | 0.03102 / 0.02593 / 0.02172 | 0.26, 0.26 |
| 25_H | scaled-$L_2$ pressure | 200 / 400 / 800 | 0.1001 / 0.04811 / 0.03988 | 1.06, 0.27 |

![](figures/grid_refinement_errors.svg)

**Figure 8.** Representative normalized error versus resolution (log–log) for six cases. Reading downward at the right: 07_B (highest at the finest grid, $0.31 \to 0.06$); 25_H (now monotone, $0.10 \to 0.048 \to 0.040$); 24_H (nearly flat, $\approx 0.031 \to 0.022$, shock-limited); 13_E ($0.0076 \to 0.0030$); 14_E (steepest, $\approx 0.89 \to 0.017 \to 0.0024$ as it crosses its resolution threshold); and 18_T (lowest, $8.9\times10^{-4} \to 7\times10^{-5}$).

**All-speed span.** Four runs under one configuration cover the Mach range. The ultra-low-Mach case 03_B propagates a $\Delta p = 1$ Pa pulse on a $10^{5}$ Pa background (Mach $\approx 10^{-5}$) in air with SG water, giving a profile error of $0.02795$, a maximum pressure perturbation of $0.5007$ Pa, and an oscillation measure of $7.71\times10^{-8}$ ($N = 200$). The low-Mach air case 04_B ($N = 200$) gives a scaled-$L_2$ error of $0.03945$ with amplitude ratio $0.9833$ and correlation $0.9983$. The interface-acoustic anchor is the air–water 07_B subcase at $N = 400$ ($L_2(p) = 0.06026$, $L_\infty(p) = 0.2331$, correlation $0.9933$). The hypersonic anchor is 25_H at $N = 200$ (scaled-$L_2$ pressure $0.1001$, density $0.06244$, correlation $0.9866$, shock position $0.5216$ cells). The same production scheme therefore spans Mach $\approx 10^{-5}$ to Mach $10$, and all four runs pass.

![](figures/all_speed_ultra_low_mach_03_B.svg)
![](figures/all_speed_low_mach_air_04_B.svg)

**Figure 9.** All-speed low-Mach anchors. Left, 03_B (SG water, $\Delta p/p \approx 10^{-5}$, oscillation $\approx 7.7\times10^{-8}$): the extreme low-Mach case. Right, 04_B ($N = 200$, correlation $0.9983$): low-Mach air acoustics. With the Mach-10 cases of Figure 7 these bracket Mach $\approx 10^{-5}$ to Mach $10$ under one configuration.

**Ablation.** Table 3 replaces one production ingredient at a time and re-runs four representative cases at fixed, deliberately coarse resolutions (02_A at $N = 100$, 07_B at $N = 200$, 13_E and 18_T at $N = 400$). At these resolutions the production configuration itself does not pass 07_B or 18_T (both are resolution-limited, as Sections 4.3 and 4.6 show), so the table is read as a per-cell comparison against the production row, not as an absolute verdict. Two results hold across every variant: 02_A preserves machine-precision equilibrium for all eight schemes (the relative $L_\infty(p)$ stays between $4.37\times10^{-16}$ and $2.47\times10^{-15}$), and 07_B fails for all eight at $N = 200$ by the coarse-grid peak-amplitude limit. On 07_B the ordering of the pressure error separates the schemes: production, superbee_only, alpha_cicsam, and alpha_mstacs share the smallest value ($0.1512$), followed by tmlpu_vanleer ($0.1688$), tmlpu_minmod ($0.1753$), and upwind_primitive ($0.4269$), while hllc_flux diverges ($L_2(p) = 1.20\times10^{39}$). On 13_E six of the eight variants pass; upwind_primitive is too diffusive ($0.01888$) and hllc_flux is near blow-up ($0.1303$). On 18_T only upwind_primitive passes, its extra diffusion satisfying the wiggle guard while the sharper schemes trip it at $0.00446$–$0.00463$; alpha_mstacs returns NaN (terminated at 798 steps) and hllc_flux blows up ($T_2$ $L_\infty$ ratio $5.25\times10^{4}$). The hllc_flux variant is the worst overall (one of four), which motivates the SLAU2 material flux of the production configuration.

**Table 3.** Ablation matrix: eight reconstruction/flux variants on four representative cases at fixed coarse resolutions (02_A $N = 100$, 07_B $N = 200$, 13_E $N = 400$, 18_T $N = 400$). Each cell gives PASS/FAIL at that resolution and the per-cell metric (02_A relative $L_\infty(p)$; 07_B air–water $L_2(p)$; 13_E smooth-region pressure $L_2$; 18_T $T_2$ $L_\infty$ ratio). Verdicts are relative to the production row, not absolute, because the production configuration is itself resolution-limited on 07_B and 18_T at these grids.

| Variant | 02_A ($N{=}100$) | 07_B ($N{=}200$) | 13_E ($N{=}400$) | 18_T ($N{=}400$) | PASS / 4 |
|---|---|---|---|---|---|
| production | PASS $5.82\times10^{-16}$ | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004468 | 2 |
| superbee_only | PASS $5.82\times10^{-16}$ | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004468 | 2 |
| tmlpu_vanleer | PASS $2.47\times10^{-15}$ | FAIL 0.1688 | PASS 0.007252 | FAIL 0.004461 | 2 |
| tmlpu_minmod | PASS $1.02\times10^{-15}$ | FAIL 0.1753 | PASS 0.008551 | FAIL 0.004466 | 2 |
| upwind_primitive | PASS $1.89\times10^{-15}$ | FAIL 0.4269 | FAIL 0.01888 | PASS 0.01872 | 2 |
| alpha_cicsam | PASS $5.82\times10^{-16}$ | FAIL 0.1512 | PASS 0.006684 | FAIL 0.004626 | 2 |
| alpha_mstacs | PASS $1.60\times10^{-15}$ | FAIL 0.1512 | PASS 0.006684 | FAIL (NaN) | 2 |
| hllc_flux | PASS $4.37\times10^{-16}$ | FAIL $1.20\times10^{39}$ | FAIL 0.1303 | FAIL $5.25\times10^{4}$ | 1 |

Per-case pass counts (of eight variants): 02_A $8/8$, 07_B $0/8$, 13_E $6/8$, 18_T $1/8$.

![](figures/ablation_pass_heatmap.svg)
![](figures/baseline_ablation_metrics.svg)

**Figure 10.** Ablation summary. Left, the PASS/FAIL heatmap over four cases (rows) and eight variants (columns): 02_A is all PASS, 07_B all FAIL (coarse-$N$ peak-amplitude limit), 13_E PASS except upwind_primitive and hllc_flux, and 18_T FAIL except upwind_primitive. Right, the companion per-variant metrics, including the hllc_flux divergences on 07_B, 13_E, and 18_T.

## 4.7. Reproducibility and scope of the evidence

The evidence package was regenerated in full on 2026-07-14 with a single fixed configuration: the imex_ssp3 integrator, adaptive_bvd volume-fraction scheme, tmlpu primitive reconstruction with the superbee limiter, slau2 material flux, fifth-order WENO reconstruction of the acoustic faces (Section 3.6), regime_auto pressure closure, CHARACTERISTIC_RECON = 1, RUSANOV_FALLBACK = 0, UNIFORM_PERIODIC_REMAP = 0, and the core override FIVE_EQ_CASE24_N = 400. The regenerated metric values are 02_A $5.821\times10^{-16}$ / $5.329\times10^{-14}$; 07_B air–water $L_2(p) = 6.026\times10^{-2}$, $L_\infty(p) = 2.331\times10^{-1}$, correlation $0.993$; helium–air $L_2(p) = 1.059\times10^{-2}$; argon–air $L_2(p) = 2.412\times10^{-3}$. The thirteen physics cases of Table 2 all pass the acceptance criteria of Section 4.1. The auxiliary sweeps of Section 4.6 add grid, CFL, and ablation runs at deliberately coarse resolutions and with deliberately degraded variants, many of which fail by design; under the acceptance criteria of Section 4.1 the aggregate tally over all runs is 41 passes of 70 (29 failures). This tally counts intentional failures and is not the pass rate of the core suite. Two case-specific caveats carry into the interpretation of Section 4.6: cases 13_E and 14_E use $N = 800$, higher than most of the suite, and case 25_H, though now monotone in the scaled-$L_2$ pressure error (Section 4.6), remains non-monotone in the scaled density and velocity errors.

# 5. Conclusions

We have described a one-dimensional finite-volume method for the five-equation two-phase model that advances material transport explicitly at the material CFL limit and the acoustic subsystem by a single linearized implicit solve, and whose spatial discretization is organized around the temperature-based primitive parameterization $(\alpha_1, T_1, T_2, u, p)$. The parameterization provides closed-form EOS evaluations and derivatives for the ideal-gas, stiffened-gas, and NASG families, an analytic transformation Jacobian, EOS-consistent face states, and energy-flux coefficients satisfying a discrete consistency identity, which together yield preservation of pressure–velocity equilibrium across material interfaces at the roundoff level even at acoustic Courant numbers exceeding one hundred. A measured amplification analysis guided the selection of the time integrator (rejecting a standard two-stage IMEX scheme that is violently unstable on discrete interface states) and identified an odd–even pressure mode, which the parameter-free acoustic-Riemann face closure of the production scheme suppresses without a tuned coefficient. On the thirteen-case suite, run with one configuration, the method preserved pressure–velocity equilibrium to a relative pressure deviation $L_\infty(p) = 5.82\times10^{-16}$ on an interface advected at an acoustic Courant number above $10^{3}$, reproduced the linear reflection–transmission amplitudes to peak ratios of $0.997$ (air–water), $0.975$ (helium–air), and $0.992$ (argon–air) at a $3600{:}1$ impedance contrast, captured gas–liquid shock and contact positions to within about one to two cells, and remained admissible through $M_s = 10$ shocks with post-shock pressures spanning four decades.

The limitations are stated plainly. The formulation and validation are one-dimensional; the multidimensional extension requires a block implicit acoustic solver and is in progress. The acoustic update is first-order accurate in time in its present linearized single-solve form, and a higher-order acoustic integrator that preserves discrete interface equilibrium remains an open problem, as quantified by the amplification analysis. No asymptotic-preserving property is claimed. The air–water acoustic benchmark requires the finest grid of the suite, and this resolution requirement is reported rather than removed. An earlier version of the scheme, with second-order reconstruction of the acoustic faces, retained a low-amplitude pressure ringing adjacent to the air–water interface whose smoothed-pressure local total-variation excess reached $0.537$ at $N = 400$, above a stricter total-variation limit of $0.30$ introduced later in development; reconstructing the acoustic face states at fifth order (Section 3.6) removes this ringing, lowering the guard value to $0.204$ while preserving the transmitted-peak amplitude ratio at $1.00$. The Mach-10 air-into-water case (25_H) now converges monotonically in the scaled-$L_2$ pressure metric under the fifth-order acoustic reconstruction ($N = 800$ gives the smallest error, $0.0399$), although the scaled density and velocity metrics remain non-monotone (each smallest at $N = 400$), which we attribute to interface-interaction phasing and the residual pressure checkerboard rather than to smooth truncation error. Finally, although the thermodynamic framework accepts any EOS supplying the derivative set (10), only the three analytic families have been validated here; Mie–Grüneisen and cubic EOS, phase exchange source terms, and viscous and capillary effects are natural extensions of the present framework.

# Appendix A. Closed-form NASG derivatives

With $v_k$ and $e_k$ given by (8)–(9), the four derivatives of the set (10) are, dropping the phase index,

$$\left( \frac{\partial \rho}{\partial p} \right)_T = \frac{(\gamma - 1) c_v T}{\left[ (p + p_\infty) v \right]^2} \, , \qquad \left( \frac{\partial \rho}{\partial T} \right)_p = - \frac{(\gamma - 1) c_v}{(p + p_\infty) v^2}, \qquad (\mathrm{A.1})$$

$$\left( \frac{\partial e}{\partial p} \right)_T = - \frac{(\gamma - 1) p_\infty c_v T}{(p + p_\infty)^2}, \qquad \left( \frac{\partial e}{\partial T} \right)_p = \frac{p + \gamma p_\infty}{p + p_\infty} \, c_v. \qquad (\mathrm{A.2})$$

For the stiffened gas ($b = 0$) and the ideal gas ($b = p_\infty = q = 0$) these reduce to the familiar expressions; in particular, for the ideal gas $(\partial e/\partial p)_T = 0$ and $(\partial e/\partial T)_p = c_v$, and (11) recovers $c^2 = \gamma R T$. The isobaric energy–density slope appearing in (21) is

$$\left( \frac{\partial e}{\partial \rho} \right)_p = \frac{(\partial e/\partial T)_p}{(\partial \rho/\partial T)_p} = - \frac{(p + \gamma p_\infty)\, v^2}{\gamma - 1}, \qquad (\mathrm{A.3})$$

which is finite and negative for all admissible NASG states and reduces to $-p v^2/(\gamma-1)$ for the ideal gas.

# Appendix B. Analytic transformation Jacobian

Writing $\rho_{k,p} = (\partial \rho_k/\partial p)_T$, $\rho_{k,T} = (\partial \rho_k / \partial T)_p$, $e_{k,p}$, $e_{k,T}$ accordingly, and $(\rho e)_k = \rho_k e_k$, the nonzero entries of $\mathbf{A} = \partial \mathbf{U} / \partial \mathbf{W}$ with $\mathbf{U}, \mathbf{W}$ as in (12) are

$$\frac{\partial U_1}{\partial \alpha_1} = \rho_1, \quad \frac{\partial U_1}{\partial T_1} = \alpha_1 \rho_{1,T}, \quad \frac{\partial U_1}{\partial p} = \alpha_1 \rho_{1,p}; \qquad \frac{\partial U_2}{\partial \alpha_1} = -\rho_2, \quad \frac{\partial U_2}{\partial T_2} = \alpha_2 \rho_{2,T}, \quad \frac{\partial U_2}{\partial p} = \alpha_2 \rho_{2,p}; \qquad (\mathrm{B.1})$$

$$\frac{\partial U_3}{\partial \alpha_1} = (\rho_1 - \rho_2) u, \quad \frac{\partial U_3}{\partial T_k} = \alpha_k \rho_{k,T} u, \quad \frac{\partial U_3}{\partial u} = \rho, \quad \frac{\partial U_3}{\partial p} = \left( \alpha_1 \rho_{1,p} + \alpha_2 \rho_{2,p} \right) u; \qquad (\mathrm{B.2})$$

$$\frac{\partial U_4}{\partial \alpha_1} = (\rho e)_1 - (\rho e)_2 + \tfrac{1}{2} (\rho_1 - \rho_2) u^2, \quad \frac{\partial U_4}{\partial T_k} = \alpha_k \left( \rho_{k,T} e_k + \rho_k e_{k,T} + \tfrac{1}{2} \rho_{k,T} u^2 \right), \qquad (\mathrm{B.3})$$

$$\frac{\partial U_4}{\partial u} = \rho u, \quad \frac{\partial U_4}{\partial p} = \sum_k \alpha_k \left( \rho_{k,p} e_k + \rho_k e_{k,p} + \tfrac{1}{2} \rho_{k,p} u^2 \right); \qquad \frac{\partial U_5}{\partial \alpha_1} = 1. \qquad (\mathrm{B.4})$$

The matrix is lower-block-triangular up to the momentum and energy rows and is inverted analytically where needed; it is verified against sixth-order finite differences to a tolerance of $10^{-7}$ in the unit-test suite.

# Appendix C. Primitive recovery

Given $\mathbf{U}$, set $\alpha_1 = U_5$ and solve system (14) for $(p, T_1, T_2)$ by Newton iteration with the analytic $3\times3$ Jacobian assembled from (A.1)–(A.2), initialized from the previous time level. Safeguards: (i) iterates are clipped to the admissible region $p + \min_k p_{\infty,k} > 0$, $T_k > 0$, $\rho_k b_k < 1$; (ii) if $\alpha_k$ is below a pure-phase tolerance, the vanishing-phase temperature is slaved to its EOS at the recovered pressure and the system reduces to a scalar solve for the remaining phase; (iii) failure of the iteration (not observed in the reported computations outside the backtracking of Section 3.6) triggers the step-size reduction of the implicit stage rather than an ad hoc state repair. Convergence to $10^{-12}$ in the relative residual typically requires two to four iterations.

# Appendix D. Supplementary result figures

This appendix collects the per-variant and per-resolution panels behind the ablation matrix (Section 4.6, Table 3) and the grid-refinement study (Section 4.6, Figure 8), together with the static-interface case 01_A.

## D.1. Static interface (01_A)

![](figures/core_01_A.svg)

**Figure D.1.** Case 01_A, static air/NASG-water interface with $u = 0$, held over 100 steps. Pressure drift $4.366\times10^{-14}$ (relative to $10^{5}$ Pa), velocity drift $1.042\times10^{-12}$ m/s, pressure checkerboard $2.609\times10^{-16}$.

## D.2. Per-variant ablation panels

Solution profiles for each of the eight variants of Table 3 at the fixed ablation resolutions (02_A $N = 100$, 07_B $N = 200$, 13_E $N = 400$, 18_T $N = 400$), grouped by case.

![](figures/production_02_A.svg)
![](figures/superbee_only_02_A.svg)
![](figures/tmlpu_vanleer_02_A.svg)
![](figures/tmlpu_minmod_02_A.svg)
![](figures/upwind_primitive_02_A.svg)
![](figures/alpha_cicsam_02_A.svg)
![](figures/alpha_mstacs_02_A.svg)
![](figures/hllc_flux_02_A.svg)

**Figure D.2.** Case 02_A ($N = 100$) for the eight variants of Table 3. Every variant preserves pressure equilibrium to machine precision (relative $L_\infty(p)$ between $4.37\times10^{-16}$ and $2.47\times10^{-15}$).

![](figures/production_07_B.svg)
![](figures/superbee_only_07_B.svg)
![](figures/tmlpu_vanleer_07_B.svg)
![](figures/tmlpu_minmod_07_B.svg)
![](figures/upwind_primitive_07_B.svg)
![](figures/alpha_cicsam_07_B.svg)
![](figures/alpha_mstacs_07_B.svg)
![](figures/hllc_flux_07_B.svg)

**Figure D.3.** Case 07_B air–water ($N = 200$) for the eight variants. All fail acceptance at this resolution by the peak-amplitude limit; the pressure error orders the schemes, from $0.1512$ (production, superbee_only, alpha_cicsam, alpha_mstacs) to $1.20\times10^{39}$ for the diverging hllc_flux.

![](figures/production_13_E.svg)
![](figures/superbee_only_13_E.svg)
![](figures/tmlpu_vanleer_13_E.svg)
![](figures/tmlpu_minmod_13_E.svg)
![](figures/upwind_primitive_13_E.svg)
![](figures/alpha_cicsam_13_E.svg)
![](figures/alpha_mstacs_13_E.svg)
![](figures/hllc_flux_13_E.svg)

**Figure D.4.** Case 13_E ($N = 400$) for the eight variants. Six pass; upwind_primitive is over-diffusive ($0.01888$) and hllc_flux is near blow-up ($0.1303$).

![](figures/production_18_T.svg)
![](figures/superbee_only_18_T.svg)
![](figures/tmlpu_vanleer_18_T.svg)
![](figures/tmlpu_minmod_18_T.svg)
![](figures/upwind_primitive_18_T.svg)
![](figures/alpha_cicsam_18_T.svg)
![](figures/alpha_mstacs_18_T.svg)
![](figures/hllc_flux_18_T.svg)

**Figure D.5.** Case 18_T ($N = 400$) for the eight variants. Only upwind_primitive passes; alpha_mstacs returns NaN and hllc_flux blows up ($T_2$ $L_\infty$ ratio $5.25\times10^{4}$).

## D.3. Per-resolution grid panels

Solution profiles at each resolution of the grid-refinement study (Section 4.6, Figure 8), grouped by case.

![](figures/grid_07_B_N100.svg)
![](figures/grid_07_B_N200.svg)
![](figures/grid_07_B_N400.svg)

**Figure D.6.** Case 07_B air–water at $N = 100$, $200$, $400$. $L_2(p) = 0.3090$, $0.1512$, $0.06026$; PASS at $N = 400$.

![](figures/grid_13_E_N200.svg)
![](figures/grid_13_E_N400.svg)
![](figures/grid_13_E_N800.svg)

**Figure D.7.** Case 13_E at $N = 200$, $400$, $800$. Smooth-region density $L_2 = 0.007635$, $0.004878$, $0.003002$.

![](figures/grid_14_E_N200.svg)
![](figures/grid_14_E_N400.svg)
![](figures/grid_14_E_N800.svg)

**Figure D.8.** Case 14_E at $N = 200$, $400$, $800$. Density-plateau $L_\infty$ ratio $0.8946$, $0.01666$, $0.002436$; PASS at $N = 800$ (at $N = 400$ the physical two-jump density split is not yet resolved).

![](figures/grid_18_T_N200.svg)
![](figures/grid_18_T_N400.svg)
![](figures/grid_18_T_N550.svg)

**Figure D.9.** Case 18_T at $N = 200$, $400$, $550$. Mixture-density $L_1$ ratio $8.866\times10^{-4}$, $2.531\times10^{-4}$, $6.981\times10^{-5}$.

![](figures/grid_24_H_N100.svg)
![](figures/grid_24_H_N200.svg)
![](figures/grid_24_H_N400.svg)

**Figure D.10.** Case 24_H at $N = 100$, $200$, $400$. Maximum density-profile $L_2$ over the five subcases $0.03102$, $0.02593$, $0.02172$; PASS at $N = 400$.

![](figures/grid_25_H_N200.svg)
![](figures/grid_25_H_N400.svg)
![](figures/grid_25_H_N800.svg)

**Figure D.11.** Case 25_H at $N = 200$, $400$, $800$. Scaled-$L_2$ pressure $0.1001$, $0.04811$, $0.03988$: monotone, with $N = 800$ the best.

# Acknowledgements

[Funding sources and grant numbers to be added.] The author thanks [colleagues] for helpful discussions.

# References

1. M.R. Baer, J.W. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (DDT) in reactive granular materials, Int. J. Multiphase Flow 12 (1986) 861–889.
2. R. Saurel, R. Abgrall, A multiphase Godunov method for compressible multifluid and multiphase flows, J. Comput. Phys. 150 (1999) 425–467.
3. A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: reduced equations, Phys. Fluids 13 (2001) 3002–3024.
4. G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577–616.
5. A. Murrone, H. Guillard, A five equation reduced model for compressible two phase flow problems, J. Comput. Phys. 202 (2005) 664–698.
6. R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160.
7. K.-M. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, J. Comput. Phys. 142 (1998) 208–242.
8. E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2006) 715–732.
9. O. Le Métayer, R. Saurel, The Noble–Abel stiffened-gas equation of state, Phys. Fluids 28 (2016) 046102.
10. H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, Comput. Fluids 28 (1999) 63–86.
11. C. Chalons, M. Girardin, S. Kokh, An all-regime Lagrange-projection like scheme for 2D homogeneous models for two-phase flows on unstructured meshes, J. Comput. Phys. 335 (2017) 885–904.
12. S. Peluchon, G. Gallice, L. Mieussens, A robust implicit–explicit acoustic-transport splitting scheme for two-phase flows, J. Comput. Phys. 339 (2017) 328–355.
13. D. Iampietro, F. Daude, P. Galon, J.-M. Hérard, A Mach-sensitive splitting approach for Euler-like systems, J. Comput. Phys. (2017).
14. M.F.P. ten Eikelder, F. Daude, B. Koren, A.S. Tijsseling, An acoustic-convective splitting-based approach for the Kapila two-phase flow model, J. Comput. Phys. 331 (2017) 188–208.
15. L. Tallois, S. Peluchon, P. Villedieu, A second-order extension of a robust implicit–explicit acoustic-transport splitting scheme for two-phase flows, Comput. Fluids (2022).
16. M. Dumbser, V. Casulli, A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible Navier–Stokes equations with general equation of state, Appl. Math. Comput. 272 (2016) 479–497.
17. W. Boscheri, L. Pareschi, High order pressure-based semi-implicit IMEX schemes for the 3D Navier–Stokes equations at all Mach numbers, J. Comput. Phys. 434 (2021) 110206.
18. V. Casulli, P. Zanolli, Iterative solutions of mildly nonlinear systems, J. Comput. Appl. Math. 236 (2012) 3937–3947.
19. F. Denner, C.-N. Xiao, B.G.P. van Wachem, Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, J. Comput. Phys. 367 (2018) 192–234.
20. F. Denner, Fully-coupled pressure-based algorithm for compressible flows: linearisation and iterative solution methods, Comput. Fluids (2018).
21. C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532.
22. B. Re, R. Abgrall, A pressure-based method for weakly compressible two-phase flows under a Baer–Nunziato type model with generic equations of state and pressure and velocity disequilibrium, Int. J. Numer. Methods Fluids 94 (2022) 1183–1232.
23. D. Fuster, S. Popinet, An all-Mach method for the simulation of bubble dynamics problems in the presence of surface tension, J. Comput. Phys. 374 (2018) 752–768.
24. Y. Saade, D. Lohse, D. Fuster, A multigrid solver for the coupled pressure-temperature equations in an all-Mach solver with VoF, J. Comput. Phys. 476 (2023) 111865.
25. A. Urbano, M. Bibal, S. Tanguy, A semi implicit compressible solver for two-phase flows of real fluids, J. Comput. Phys. 456 (2022) 111034.
26. X. Deng, B. Xie, O.K. Matar, P. Boivin, A hybrid approach for simulating multi-component flows across all Mach numbers, J. Comput. Phys. (2025), in press; arXiv:2502.02570.
27. L. Battisti, W. Boscheri, A linearly implicit shock capturing scheme for compressible two-phase flows at all Mach numbers, J. Comput. Phys. 539 (2025) 114227.
28. H. Terashima, et al., An approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible multi-species flows, J. Comput. Phys. (2025).
29. Z. He, B. Tan, et al., A generic five-equation formulation with thermodynamically compatible interface sharpening for multimaterial flows, J. Comput. Phys. (2024).
30. F. Zhao, et al., An interface-sharpening method for multimaterial flow simulation with conservative and consistent fluxes, Phys. Fluids (2025).
31. Z. He, F. Zhao, Compact closed-form closures for the five-equation model under pressure–temperature equilibrium, Phys. Fluids (2025).
32. F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, Int. J. Numer. Methods Fluids 48 (2005) 1023–1040.
33. X. Deng, S. Inaba, B. Xie, K.-M. Shyue, F. Xiao, High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces, J. Comput. Phys. 371 (2018) 945–966.
34. O. Ubbink, R.I. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 153 (1999) 26–50.
35. E. Shima, K. Kitamura, Parameter-free simple low-dissipation AUSM-family scheme for all speeds, AIAA J. 49 (2011) 1693–1709.
36. K. Kitamura, E. Shima, Towards shock-stable and accurate hypersonic heating computations: a new pressure flux for AUSM-family schemes, J. Comput. Phys. 245 (2013) 62–83.
37. B. van Leer, Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method, J. Comput. Phys. 32 (1979) 101–136.
38. P.L. Roe, Characteristic-based schemes for the Euler equations, Annu. Rev. Fluid Mech. 18 (1986) 337–365.
39. K.H. Kim, C. Kim, Accurate, efficient and monotonic numerical methods for multi-dimensional compressible flows. Part II: multi-dimensional limiting process, J. Comput. Phys. 208 (2005) 570–615.
40. S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM Rev. 43 (2001) 89–112.
41. U.M. Ascher, S.J. Ruuth, R.J. Spiteri, Implicit–explicit Runge–Kutta methods for time-dependent partial differential equations, Appl. Numer. Math. 25 (1997) 151–167.
42. L. Pareschi, G. Russo, Implicit–explicit Runge–Kutta schemes and applications to hyperbolic systems with relaxation, J. Sci. Comput. 25 (2005) 129–155.
43. E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, third ed., Springer, Berlin, 2009.
44. A.B. Wood, A Textbook of Sound, G. Bell and Sons, London, 1930.
45. G.-S. Jiang, C.-W. Shu, Efficient implementation of weighted ENO schemes, J. Comput. Phys. 126 (1996) 202–228.
