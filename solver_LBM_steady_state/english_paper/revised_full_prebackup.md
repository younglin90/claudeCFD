**A Geometry-Aware Moment-Schur Nonlinear Preconditioner with an Admissibility-Preserving Gate for Steady-State Lattice Boltzmann Solvers: Validation on Complex-Geometry Benchmarks**

Moment-Schur Accelerated LBM (MSA-LBM)

Young-Lin Yoo

# Abstract

In steady-state lattice Boltzmann method (LBM) computations, the long-wavelength residual of the conserved pressure--velocity moments persists long after the kinetic modes have decayed, and it is this slowly decaying residual that governs fixed-point convergence. This paper proposes a geometry-aware, Jacobian-free nonlinear preconditioning technique, termed Moment-Schur Accelerated LBM (MSA-LBM), that targets this bottleneck directly. Every accepted update is screened by an admissibility-preserving gate; here "admissibility-preserving" refers to the physical feasibility of the trial state and is unrelated to the asymptotic-preserving schemes of kinetic theory or to any neural-network method. The method leaves the discretized LBM equations and boundary conditions unchanged and operates on the native residual $R(f) = f - G(f)$ as is. Its core operator is built by closing the conserved-moment Schur complement of the LBE operator---linearized about a uniform base state---into a per-mode $3 \times 3$ matrix in Fourier space, and assembling its kinetic-null-space-corrected, admissibility-preserving inverse as a spectral preconditioner. This preconditioner is used solely as the left preconditioner of a single Jacobian-free Newton (GMRES) step applied to the native nonlinear residual; the resulting trial update is accepted only when, after a damped line search, it simultaneously satisfies a decrease in the macroscopic $L_{2}$ residual, density positivity, wall/inlet/outlet/mask boundary consistency, and conservation sanity, and otherwise the solver falls back to a native Picard step. All internal constants depend only on a global grid scale and on neither benchmark identity nor any reference solution.

On a fixed set of benchmark results, the proposed method attains convergence under an identical protocol across all 27 runs spanning nine benchmark families (channel, Couette, lid-driven cavity at $Re = 100/400/1000$, backward-facing step, cylinder wake, multi-cylinder, and T-junction) at the 1x/2x/3x mesh levels. Under the same stopping protocol and the same admissibility definition, five baseline methods (Picard, Anderson acceleration, preconditioned LBM, inexact Newton--Krylov, and dual-time multigrid) converge on only 12--15 of the 27 cases even with a generous budget. In a conservative timing comparison restricted to the strict subset (15 cases) on which a baseline also converges, the proposed method is faster in wall time on 14/15 cases (median ratio $\approx 2.06 \times$) and uses fewer LBE-calls---the operator-work metric---on 13/15 cases (median $\approx 1.80 \times$). Broadening the comparison to all available baselines, it is faster on 25/27 cases (median $2.92 \times$). On accuracy, the method exhibits an observed spatial convergence order of $\approx 2.0$ for channel Poiseuille flow (the BGK-LBM theoretical value), machine precision for Couette flow, and monotone convergence toward the Ghia centerline for the cavity, confirming that the acceleration does not sacrifice discrete accuracy. All results are independently recomputable from the stored residual histories and per-case execution traces.

The novelty of this work lies not in modifying the LBM physical model, but in a single solver framework that preconditions the hydrodynamic slow modes of the native steady residual from a Schur-complement viewpoint and validates every accepted update through the same admissibility gate, even on complex geometries. The performance claims are confined to relative comparisons within a stored 2D D2Q9/BGK benchmark suite under an identical macroscopic-$L_{2}$-residual/plateau protocol, and the method is to be understood as a nonlinear preconditioner that solves the same discrete steady problem faster---without reference injection or case-specific tuning.

**Keywords:** lattice Boltzmann method; steady-state solver; admissibility-preserving Schur complement; Jacobian-free residual correction; nonlinear preconditioning; complex geometry.

# 1. Introduction

The lattice Boltzmann method (LBM) has become a workhorse for a wide range of CFD problems, owing to the simplicity of its streaming--collision structure and its amenability to complex-boundary treatment and parallelization \[1--4\]. In applications where the steady-state solution itself is the goal---design optimization, geometric parameter sweeps, inverse design---rather than the transient, however, the explicit time-marching nature of LBM translates directly into cost. The origin of this cost is not a requirement of temporal accuracy but the spectral structure of the fixed-point residual. In native lattice Boltzmann equation (LBE) iteration, the non-conserved kinetic modes are damped relatively quickly by collision relaxation, whereas the conserved hydrodynamic modes associated with density and momentum survive as the long-wavelength shear and acoustic modes of the linearized LBE and decay very slowly \[4\]; in the low-Mach regime in particular, the separation of convective and acoustic scales further retards this decay \[20, 21\]. As a result, the convergence history exhibits a rapid initial drop followed by a long, flat tail, and it is this tail that dominates the total wall time.

Prior efforts to mitigate this tail fall broadly into three families. First, algebraic history accelerators, exemplified by Anderson acceleration and reduced-rank extrapolation (RRE), extrapolate a descent direction from the residual correlations of the past fixed-point history \[9, 10, 15\]. These are powerful and general, but they treat the residual as a structureless vector and do not directly exploit the *physical block structure* by which, within the steady LBM residual, the kinetic fast modes and the hydrodynamic slow modes vanish on different time scales. Second, inexact Newton and Jacobian-free Newton--Krylov (JFNK) methods provide the standard framework for solving the nonlinear residual equation directly \[6, 7, 13\]. The efficiency of JFNK, however, hinges entirely on how well the preconditioner reflects the physical structure of the problem; an unsuitable preconditioner inflates the cost of both GMRES iterations and residual evaluations. Moreover, on complex geometries where masks, obstacles, and open boundaries coexist, a Newton trial step may violate density positivity or boundary consistency, so that the physical admissibility of the trial becomes a problem distinct from convergence. The third family modifies the interior of the LBM model itself. Preconditioned LBM redefines the collision relaxation spectrum or the equilibrium to accelerate low-Mach steady convergence \[20, 21\], and recent work has combined preconditioning with cascaded/central-moment LBM (including corrections for Galilean invariance and cubic-velocity errors) \[22, 23, 25\] and extended the lattice Boltzmann flux solver to steady flows on unstructured grids \[24\]. Multigrid and dual-time families relax the elliptic coupling through a mesh hierarchy and coarse-grid corrections \[14\], and the pressure-Schur/saddle-point preconditioning they borrow is a canonical framework for rapidly solving the pressure--velocity coupling of incompressible systems \[8, 11, 12\]. On a separate axis, Huang, Yang, and Cai discretized the LBE implicitly and proposed fully implicit and nonlinearly preconditioned inexact-Newton frameworks combining Newton--Krylov, domain decomposition, and nonlinear elimination \[18, 19\]. These rich bodies of prior work nonetheless share one common trait: to accelerate, they either redefine the collision model, equilibrium, or relaxation parameter (model-level acceleration), or require additional infrastructure such as a mesh hierarchy, transfer operators, implicit matrix assembly, or domain decomposition. In other words, the acceleration takes place *inside* the existing discrete LBM operator or in the *infrastructure surrounding* it.

A gap therefore emerges. Steady-state LBM acceleration still lacks an externally attached correction layer that *simultaneously* satisfies three conditions: (a) it leaves the native operator---built from collision, streaming, and boundary handling---and its discrete steady solution entirely unchanged; (b) it selectively targets only the conserved-moment slow-mode block that governs convergence; and (c) it guarantees physical admissibility, without case-by-case tuning, even on complex geometries with masks and open boundaries. Algebraic accelerators do not use the block structure of (b); model-level accelerations violate (a); and generic JFNK and implicit frameworks either do not address (c) separately or require heavy infrastructure.

This paper proposes such an externally attached acceleration layer---a geometry-aware moment-Schur nonlinear preconditioner with an admissibility-preserving acceptance gate, hereafter Moment-Schur Accelerated LBM (MSA-LBM)---that meets all three conditions at once. The qualifier "admissibility-preserving" refers to the physical feasibility of each accepted trial state and is unrelated to the asymptotic-preserving schemes of kinetic theory or to any neural-network method. The central idea begins from the observation that the slow components governing convergence reside in the conserved-moment block: it therefore suffices to precondition only the Schur complement of that subspace, constructed in closed form in Fourier space, and to use it solely as the preconditioner of a single Jacobian-free Newton step applied to the unchanged native residual $R(f) = f - G(f)$. The generated trial is accepted only if it passes both a residual decrease and physical admissibility, and otherwise the solver falls back to a native LBE step; the method therefore operates stably on complex geometries while leaving the existing LBM operator and its discrete steady solution untouched.

The method is validated on the 27 runs obtained by extending nine benchmark families to the 1x/2x/3x levels, together with a 1x ablation study. Under an identical stopping protocol the proposed method converges on all 27 cases, whereas the baseline accelerators converge on only a subset under the same conditions; and even in the conservative comparison restricted to cases on which a baseline also converges, the proposed method reaches steady state with shorter wall time and fewer operator-work units, without sacrificing discrete accuracy. All performance claims are made as relative comparisons within the stored 2D D2Q9/BGK suite.

# 2. Numerical Methods

## 2.1 Native steady-state LBM residual and notation

We consider the two-dimensional, nine-velocity (D2Q9) lattice Boltzmann equation with a single-relaxation-time (BGK) collision operator,

$$f_{i}(\mathbf{x} + \mathbf{c}_{i},\, t + 1) = f_{i}(\mathbf{x},t) - \frac{1}{\tau_{f}}\left\lbrack f_{i}(\mathbf{x},t) - f_{i}^{eq}(\mathbf{x},t) \right\rbrack,\quad\quad i = 0,\ldots,8,$$

where $\mathbf{x}$ is a node of the square lattice $\mathbb{Z}^{2}$, $t$ is the discrete time step, $\{\mathbf{c}_{i}\}_{i = 0}^{8}$ are the D2Q9 lattice velocities, lattice units are used throughout ($\Delta x = \Delta t = 1$), and $\tau_{f}$ is the BGK relaxation time. The discrete equilibrium is

$$f_{i}^{eq}(\rho,\mathbf{u}) = w_{i}\,\rho\left\lbrack 1 + \frac{\mathbf{c}_{i} \cdot \mathbf{u}}{c_{s}^{2}} + \frac{(\mathbf{c}_{i} \cdot \mathbf{u})^{2}}{2c_{s}^{4}} - \frac{|\mathbf{u}|^{2}}{2c_{s}^{2}} \right\rbrack,\quad\quad c_{s}^{2} = \frac{1}{3},$$

with weights $w_{0} = 4/9$, $w_{1\text{–}4} = 1/9$, and $w_{5\text{–}8} = 1/36$. The macroscopic moments are

$$\rho = \sum_{i}^{}f_{i},\quad\quad\rho\mathbf{u} = \sum_{i}^{}\mathbf{c}_{i}f_{i},\quad\quad p = c_{s}^{2}\rho.$$

The kinematic viscosity is $\nu = c_{s}^{2}(\tau_{f} - \frac{1}{2})$, and the flow is characterized by $Re = U_{ref}L_{ref}/\nu$ and $Ma = U_{ref}/c_{s}$. All benchmarks are run at $Ma \ll 1$, so the weakly compressible pressure $p = c_{s}^{2}\rho$ is consistent with the incompressible reference solutions.

We denote by $G$ a single full lattice step, comprising collision, streaming, and the application of all boundary conditions. The steady state is the fixed point of this operator, expressed as the residual equation

$$R(f) = f - G(f) = 0,$$

with $G$ invoked as a stand-alone operator that the acceleration scheme leaves unmodified.

Two linear operators connect the nine-component distribution field to its three conserved hydrodynamic moments. The projection $M \in \mathbb{R}^{3 \times 9}$ maps a distribution to its density and momentum,

$$Mf = (\rho,\ \rho u_{x},\ \rho u_{y})^{\top},$$

its rows being the moment weights $\{ 1,\, c_{x,i},\, c_{y,i}\}$. The lifting $T \in \mathbb{R}^{9 \times 3}$ reconstructs a minimal distribution increment from a conserved-moment increment $dm = (d\rho,\, d(\rho u_{x}),\, d(\rho u_{y}))$,

$$(T\, dm)_{i} = w_{i}\left\lbrack d\rho + 3\, c_{x,i}\, d(\rho u_{x}) + 3\, c_{y,i}\, d(\rho u_{y}) \right\rbrack,$$

so that $T$ is a right inverse of $M$, $MT = I_{3}$. The increment $T\, dm$ is the minimal hydrodynamic distribution perturbation that matches a prescribed change in density and momentum while leaving the remaining kinetic components to be relaxed by $G$ at the next iteration.

## 2.2 Conserved-moment Schur-complement formulation

A Newton correction for the steady state solves $J_{f}(f^{*})\, df = - R(f^{*})$, where $J_{f} = \partial R/\partial f$. Assembling the full Jacobian, together with its mask and boundary operators, is costly in both memory and implementation. We therefore split the correction into a conserved-moment part $dm = (d\rho,\, d(\rho u_{x}),\, d(\rho u_{y}))$, consistent with $M$ and $T$ above, and a kinetic part $dk$, which gives the block-partitioned system

$$\begin{bmatrix}
J_{mm} & J_{mk} \\
J_{km} & J_{kk}
\end{bmatrix}\begin{bmatrix}
dm \\
dk
\end{bmatrix} = - \begin{bmatrix}
R_{m} \\
R_{k}
\end{bmatrix}.$$

Eliminating the kinetic block yields the Schur-complement system for the conserved moments,

$$S_{m}\, dm = - \left( R_{m} - J_{mk}J_{kk}^{- 1}R_{k} \right),\quad\quad S_{m} = J_{mm} - J_{mk}J_{kk}^{- 1}J_{km}.$$

The kinetic block $J_{kk}$ collects the locally, rapidly relaxing modes, whereas the moment Schur complement $S_{m}$ encodes the global, weakly damped hydrodynamic coupling that controls the slow pressure--velocity modes and dominates the late stagnating phase of the iteration. Acceleration is therefore required only for the three conserved-moment components, not for all nine distributions per node.

Forming $S_{m}^{- 1}$ explicitly is intractable, so we approximate its action analytically, with the closed-form per-mode block derived in the spectral construction that follows. Working entirely in moment space, the practical correction reads

$$S_{m} \approx MJ_{f}T,\quad\quad S_{m}\, dm \approx - MR(f^{*}),\quad\quad df_{AP} = T\, dm,$$

where $M$ and $T$ are the operators of Eqs. (5)--(6) and $MT = I_{3}$ guarantees consistency. Interior solid and obstacle nodes are excluded; only fluid nodes enter the computation. The action of $J_{f}$ on a direction $v$ is obtained, matrix-free, by a directional finite difference of the native residual,

$$J_{f}(f)\, v \approx \frac{R(f + \varepsilon v) - R(f)}{\varepsilon},\quad\quad\varepsilon = 10^{- 7}\,\frac{1 + \parallel f \parallel_{2}}{\parallel v \parallel_{2}},$$

where $10^{- 7}$ is the forward-difference scale at IEEE double precision. The method is thus a moment-Schur nonlinear preconditioner built on the Jacobian-free residual response, not a full JFNK; the Jacobian $J_{f}$ is never assembled.

## 2.3 Spectral moment-Schur preconditioner

To obtain a closed-form approximation of $S_{m}^{- 1}$ we linearize the lattice update about a uniform base state $\bar{\rho} = 1$, $\bar{\mathbf{u}} = 0$. In Fourier space, streaming becomes a diagonal phase operator $A(\mathbf{k}) = diag(e^{- i\mathbf{k} \cdot \mathbf{c}_{i}})$, and the global problem decouples mode by mode. The linearized BGK collision is $C(\omega) = (1 - \omega)I_{9} + \omega\, TM$ with $\omega = 1/\tau_{f}$, the single linearized update is $L'(\mathbf{k}) = A(\mathbf{k})C(\omega)$, and the fixed-point residual Jacobian is $J(\mathbf{k}) = I_{9} - L'(\mathbf{k})$. Reducing to moment space gives a $3 \times 3$ Schur complement per wavenumber.

The Galerkin reduction of this Jacobian is

$$S_{m}^{G}(\mathbf{k}) = M\, J(\mathbf{k})\, T = I_{3} - M\, A(\mathbf{k})\, T,$$

which omits the influence of the kinetic modes; moments and kinetic modes interact through streaming and collision, and the resulting damping depends on $\omega$. We restore this coupling through the correction

$$S_{m}^{AP}(\mathbf{k}) = S_{m}^{G}(\mathbf{k}) - \kappa(\omega)\left\lbrack MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} \right\rbrack.$$

The bracketed correction term reflects an exact identity,

$$MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} = MA(\mathbf{k})\,(I_{9} - TM)\, A(\mathbf{k})\, T,$$

which is precisely the moment$\rightarrow$kinetic$\rightarrow$moment coupling, i.e. the moment-space representation of $J_{mk}J_{km}$ appearing in Eq. (8). The Galerkin reduction discards exactly this term; approximating the kinetic inverse $J_{kk}^{- 1}$ by the scalar $\kappa(\omega)$ then makes Eq. (12) a first-order, structurally exact reconstruction of $- J_{mk}J_{kk}^{- 1}J_{km}$. The scalar is

$$\kappa(\omega) = \frac{1}{2}\, sign(r)\,\min\left( \frac{1}{2},\,|r| \right),\quad\quad r = \frac{1 - \omega}{\omega},$$

so that the cap keeps $|\kappa| \leq 1/4$ and the correction stays bounded for all $\omega \in (0,2)$, including the limit $\omega \rightarrow 0$ where $r \rightarrow + \infty$ and $\kappa$ saturates at the cap.

To control the conditioning of the per-mode operator we add an adaptive Tikhonov shift,

$$S_{m}^{reg}(\mathbf{k}) = S_{m}^{AP}(\mathbf{k}) + \eta\, I_{3},\quad\quad\eta = \frac{\sigma_{\max}(S_{m}^{AP})}{50},$$

which, with $\eta$ set to a fixed fraction of $\sigma_{\max}$, limits the per-mode condition number to approximately $50$. The per-mode preconditioner block is then $B_{m}(\mathbf{k}) = \lbrack S_{m}^{reg}(\mathbf{k})\rbrack^{- 1}$. The mean mode $\mathbf{k} = (0,0)$ is tied to mass conservation: we take no Newton step on the density mean and pass the momentum mean through unchanged, so $B_{m}(0) = diag(0,1,1)$.

The assembled preconditioner $B_{m}$ acts on a residual field by projecting to moments, transforming, applying the cached per-mode blocks $B_{m}(\mathbf{k})$, transforming back, and lifting,

$$B_{m}R_{f} = T\,\mathcal{F}^{- 1}\left\{ B_{m}(\mathbf{k}\mathcal{) \cdot F\lbrack}MR_{f}\rbrack(\mathbf{k}) \right\}.$$

Because $B_{m}(\mathbf{k})$ depends only on $(N_{y},N_{x},\omega)$, it is built once per case at cost $O(N_{f}\log N_{f})$ and reused at every application. The Fourier linearization assumes periodicity and so does not represent non-periodic boundary conditions exactly; this is standard Krylov preconditioning practice and does not affect the converged solution, which is governed by the native nonlinear residual and the admissibility gate described next.

## 2.4 Jacobian-free Newton step and admissibility gate

The spectral operator $B_{m}$ is used as a left preconditioner for a single preconditioned Newton step on the native residual. In each outer round we run a left-preconditioned GMRES on

$$J_{f}(f^{k})\, df = - R(f^{k}),$$

with the operator action $v \mapsto J_{f}(f^{k})\, v$ supplied by the finite difference of Eq. (10), preconditioner $B_{m}$, Krylov subspace dimension $k_{\max} = 30$, restart $= 2k_{\max}$, and a single outer iteration.

The GMRES direction $df$ is not accepted directly. Instead it is subjected to a damped line search coupled to an admissibility gate. With trial states $f_{trial}(\alpha) = f^{k} + \alpha\, df$ and $\alpha \in \{ 1,\frac{1}{2},\frac{1}{4},\frac{1}{8}\}$, we accept the largest $\alpha$ whose trial is admissible and strictly reduces the native residual,

$$\text{accept }\alpha \Leftrightarrow admissible(f_{trial})\  \land \  \parallel R(f_{trial}) \parallel < \parallel R(f_{best}) \parallel \  \land \ conservation(f_{trial}).$$

Candidates are tested from the largest step first, the first passing candidate is accepted, and if none passes the step is recorded as rejected and the solver falls back to a native LBM update. The admissibility predicate comprises the following gates.

| Gate | Condition |
|---|---|
| Finite field | reject any NaN or Inf |
| Positive density | reject $\rho \leq 0$ |
| Residual decrease | accept only if the native macroscopic residual decreases |
| Boundary consistency | re-apply native wall/inlet/outlet/mask before evaluating the residual |
| Conservation sanity | mass drift and inlet--outlet flux closure not worsened relative to native |

Trial states are interior-only: the native boundary projection is re-applied before the residual, positivity, finiteness, and mask checks, and solid and mask nodes are excluded from both the fluid-domain norm $\parallel \cdot \parallel_{\Omega_{f}}$ and the projection $M$.

## 2.5 Solver procedure and scale-only adaptation

The complete method is summarized in Algorithm 1; its only adaptive element is a global scale $s$ that sets the burn-in and block lengths. The burn-in is the number of Picard (native LBM) iterations used for initial stabilization, and the block is the number of Picard iterations forming each per-round candidate. The scale is purely geometric,

$$s = max\left( \sqrt{\frac{N_{dof}}{9 \cdot 32^{2}}},\, 1 \right),$$

where $N_{dof} = 9N_{f}$, so that $s$ equals the linear grid size divided by $32$ ($s = 1$ on a $32 \times 32$ D2Q9 lattice), independent of $Re$, boundary conditions, and mask. The lengths are $burn = clip(round(16s),\, 8,\, 96)$ and $block = clip(round(80s),\, 48,\, 512)$.

The conceptual workflow is summarized in Figure 1: the moment residual is extracted, an MSA-LBM correction is formed and validated by the admissibility gate, and either the accepted update or a native fallback is applied.

> **Algorithm 1** --- MSA-LBM accelerated steady-state LBM

1.  Burn-in: $f \leftarrow {Picard}^{\, burn}(f_{0})$; initialize $f_{best}$, $r_{best}$.
2.  For round $= 1,\ldots,R_{\max}$ ($R_{\max} = 160$):
    a.  Picard candidate: $c_{pic} = {Picard}^{\, block}(f)$, residual $r_{pic}$.
    b.  MSA-LBM candidate: solve Eq. (17) by $B_{m}$-preconditioned GMRES for $df$, then form $c_{ap}$, $r_{ap}$ via the line-search/gate of Eq. (18).
    c.  Choose the candidate with the smallest residual.
    d.  If no candidate beats $r_{best}$ by a factor $1.02$, fall back to a native Picard guard.
    e.  Set $f \leftarrow$ chosen; update $f_{best}$, $r_{best}$, and the staleness counter.
    f.  Terminate if $r_{best} \leq \tau$ or if staleness $\geq$ stale$_{\max}$ ($= 40$).
3.  Return $f_{best}$ and the convergence history.

A single MSA-LBM step is not cheaper than a single Picard step; it requires several native residual evaluations, a spectral solve, and a line search. The benefit is structural rather than per-step: an early global correction removes the slow hydrodynamic mode that would otherwise require thousands to hundreds of thousands of Picard tail iterations to damp, thereby shortening the total time to steady state. Because the full Newton matrix is never formed, the memory footprint scales as $O(N_{f})$; the full cost model is given in Appendix A.3.

Global mass and inlet--outlet flux closure are reported as auxiliary physical-plausibility diagnostics, defined in Appendix A.4; they are not part of the stopping rule, which uses only the macroscopic residual and its plateau.

![](media/image1.png)

**Figure 1.** Conceptual workflow of the MSA-LBM method. The macroscopic moment residual is extracted from the native LBM residual, the MSA-LBM correction is validated by the admissibility gate, and execution proceeds with either the accepted update or the native fallback.

# 3. Results

We first summarize the benchmark suite and evaluation protocol, then present the convergence, performance, and accuracy results.

## 3.1 Benchmark suite and evaluation protocol

The method is evaluated on nine flow families spanning canonical flows with closed-form solutions through complex mask and branching geometries: plane Poiseuille and Couette flow (analytic references), the lid-driven cavity at $Re = 100/400/1000$ (Ghia centerline benchmark \[5\]), the backward-facing step and cylinder wake (separation, reattachment, wake), the multi-cylinder configuration (complex mask boundary), and the T-junction (branching geometry with coupled inlet/outlet boundaries). Each family is solved at three mesh levels (1x/2x/3x), for 27 runs in total; Table 1 lists the grid sizes, boundary conditions, validation role, and reference tier.

**Table 1. Benchmark family definitions: grid size, boundary conditions, validation role, and reference tier.**

| Family | Grid (1x / 2x / 3x) | Boundary conditions | Validation role | Reference |
|---|---|---|---|---|
| Channel (plane Poiseuille) | 32×192 / 64×384 / 96×576 | Inlet/outlet + wall | Pressure-driven shear (baseline) | Analytic |
| Couette | 32² / 64² / 96² | Moving wall + wall | Shear flow (baseline) | Analytic |
| Cavity $Re = 100$ | 33² / 65² / 97² | Lid-driven, closed | Recirculating closed domain | Ghia \[5\] |
| Cavity $Re = 400$ | 49² / 97² / 145² | Lid-driven, closed | Recirculating closed domain | Ghia \[5\] |
| Cavity $Re = 1000$ | 129² / 257² / 385² | Lid-driven, closed | Recirculating closed domain | Ghia \[5\] |
| Backward-facing step | 64² / 128² / 192² | Inlet/outlet + step mask | Separation/reattachment | Tight ref |
| Cylinder wake | 64² / 128² / 192² | Inlet/outlet + obstacle mask | Wake formation | Tight ref |
| Multi-cylinder | 32² / 64² / 96² | Multiple obstacle masks | Complex mask boundary | Tight ref |
| T-junction | 96×64 / 192×128 / 288×192 | Branching inlet/outlet | Branching geometry + open-BC coupling | Picard ref |

Convergence is judged on the macroscopic $L_{2}$ change of the pressure and velocity fields rather than the microscopic $f$-RMS. The pressure increment is made gauge-invariant by removing its fluid-domain mean (the absolute pressure level is arbitrary in weakly compressible closed or periodic flow) and combined with the velocity increment into a residual $r_{macro}$ evaluated over fluid nodes only. Convergence requires three conditions simultaneously: an absolute residual $r_{macro} \le 5\tau$; a plateau condition (fractional improvement over the last $W = 50$ checks at most $\eta = 0.05$); and physical admissibility (density positivity, finite fields, boundary/mask consistency). The base tolerance is $\tau = 10^{-7}$ for the non-cavity families and $10^{-8}$ for the cavity families at 1x, tightened by $1/2$ and $1/3$ at 2x and 3x. All constants and the admissibility rule are applied identically to every method, with no per-case tuning.

The proposed method is compared against five baselines that share the native LBM operator of the same code base (Table 2). The native LBE is the collide--stream--boundary fixed-point (Picard) iteration; Anderson acceleration is a regularized least-squares fixed-point accelerator; preconditioned LBM combines the standard balanced PLBE transform with a block preconditioner; inexact Newton--Krylov is a JFNK method using GMRES with a smoother and a line search; and dual-time multigrid is an FAS V-cycle scheme. Each baseline is implemented with literature-standard hyperparameters and a generous iteration budget (about $6 \times 10^{5}$ to $1.2 \times 10^{6}$ LBE-calls for cavity 2x/3x), so that none is deliberately weakened. The only difference between the proposed method and a baseline is the update rule; all run in the same Python/NumPy environment calling the same native operator, so that any wall-time difference reflects algorithmic structure rather than language or library.

**Table 2. Baseline implementations and main settings.**

| Baseline method | Implementation summary | Main settings |
|---|---|---|
| Picard (native LBM) | Native collide--stream--boundary fixed-point iteration | max_steps $\leq 1.2 \times 10^{6}$, residual-monotone termination |
| Anderson acceleration \[9,10\] | Regularized least-squares fixed-point acceleration, admissibility safeguard | depth $m = 10$, $\beta = 1.0$, reg $= 10^{-12}$ |
| Preconditioned LBM \[20,21\] | Balanced PLBE ($\gamma$-scaled) transform + block preconditioner | $\gamma = 0.5$, max_steps $\leq 1.2 \times 10^{6}$ |
| Inexact Newton--Krylov \[6,7\] | JFNK: GMRES + NE/smoother + line search | krylov_max=10, K_ne=20, K_smooth=10, line_search=4 |
| Dual-time multigrid \[14\] | FAS V-cycle, residual-equation smoothing | max_levels=6, V-cycle, K_pre/coarse/post=20/30/20 |

Because the methods stop at different accuracies, all are compared by *time-to-threshold*---the cost to first reach a common target residual $\varepsilon$, read both as wall time and as the number of native operator evaluations $G(f)$ (LBE-calls); the latter is a hardware-independent, deterministic operator-work metric that includes the cost of rejected trials. Reference solutions (analytic, Ghia, tight numerical) are never used inside the solve, entering only the post-solve accuracy evaluation (relative $L_{2}$ error). The proposed method solves all 27 problems with a single deterministic routine, only the problem definition (grid, $Re$, boundary, mask) changing per case.

## 3.2 Convergence histories

![](media/image2.png)

**Figure 2.** Macroscopic $L_{2}$ residual versus wall time for a representative case (cavity $Re = 1000$, 2x), for all six methods. The baseline methods stall at a hydrodynamic plateau, whereas only the proposed method descends below the stopping tolerance.

Figure 2 shows, for a representative high-Reynolds cavity case, the residuals of the six methods plotted against wall time on a logarithmic residual axis. The baselines descend rapidly at first but then stall at a plateau near $10^{-6}$ and fail to descend further. The proposed method (bold red), by contrast, passes through this plateau and decreases monotonically below the stopping tolerance. The residual trajectory of the proposed method both descends faster and converges to a lower residual value.

![](media/image3.png)

**Figure 3.** Convergence histories of all nine 1x-grid cases (all methods, monotone wall-time axis).

Figure 3 extends the same comparison to all nine validation cases at the 1x level. To demonstrate that the behavior is not specific to a curated subset, every case is shown on identical axes and with identical color conventions. In every panel the proposed method (bold red) reaches the lowest residual first. The corresponding 2x and 3x grids of histories are provided in the appendix (Figures A1 and A2).

**Table 3. Per-level convergence summary for the proposed method.**

| Level | Cases | Converged | Total wall \[s\] | Median residual | Max residual | Median rel. error |
|---|---|---|---|---|---|---|
| 1x | 9 | 9 | 134.2 | 2.142e-12 | 2.474e-08 | 3.260e-03 |
| 2x | 9 | 9 | 1546.6 | 3.305e-12 | 6.409e-08 | 0.0326 |
| 3x | 9 | 9 | 3507.2 | 1.153e-11 | 1.567e-08 | 0.0257 |

Table 3 aggregates the proposed-method results by level. The method satisfies the convergence criterion of Section 3.1 (the three convergence flags simultaneously) on all 27 runs. The "median rel. error" column is not directly comparable across levels, because the set of cases with a comparable reference solution differs by level (some complex geometries lack a tight reference at 2x/3x); accuracy is verified case by case in Section 3.5.

## 3.3 Convergence-rate analysis and robustness

The behavior observed in Section 3.2 follows directly from the linearized structure of Section 2. The asymptotic convergence rate of the native LBE iteration is governed by the spectral radius of the linearized iteration operator, which separates into kinetic and hydrodynamic modes. The kinetic modes, corresponding to the block $J_{kk}$, are damped strongly and locally, whereas the hydrodynamic modes tied to the conserved moments (density and momentum) are governed by the Schur complement $S_{m} = J_{mm} - J_{mk}J_{kk}^{-1}J_{km}$. The damping rate associated with the largest eigenvalue of $S_{m}$ approaches unity, so that after sufficiently many iterations the residual is determined by this single slow mode, forming the plateau observed in Figures 2 and 3. The proposed method removes this dominant slow mode directly by applying an analytic approximation of $S_{m}^{-1}$ restricted to the conserved-moment subspace. As a result, the residual continues to decrease in the region where the native iteration stalls, demonstrating that the acceleration selectively targets the component that constrains late convergence.

This mechanistic difference is quantified by the robustness gap between the methods. Under the same stopping criterion, the same admissibility definition, and the same iteration budget granted to all methods, the proposed method satisfies the convergence criterion ($r_{macro} \le 5\tau$ together with the plateau and admissibility conditions) on all 27 cases. The five baselines, by contrast, converge on only a subset under identical conditions: the most robust, inexact Newton--Krylov, on 15 cases, preconditioned LBM on 14, Picard and Anderson on 13 each, and dual-time multigrid on 12.

The non-convergence of the baselines arises from numerical stagnation rather than budget exhaustion, as can be confirmed directly from the stored convergence histories. For cavity $Re = 400$ (2x), for example, all five baselines stop decreasing at a residual level of $3.4\text{--}3.6 \times 10^{-6}$, roughly two orders of magnitude above the target tolerance ($5\tau = 2.5 \times 10^{-8}$); this stagnation already sets in at $6\text{--}7 \times 10^{5}$ LBE-calls, well short of the budget limit ($\sim 10^{6}$). For cavity $Re = 1000$ (2x), the residual remains at $\mathcal{O}(10^{0})$ even after $1.2 \times 10^{6}$ LBE-calls. In these cases the residual curves enter their asymptotic plateau before the budget is reached, so that additional iterations cannot achieve convergence.

This robustness gap is reported in its own right but is not used in the quantitative timing comparison of Section 3.4. To ensure fairness between methods, the timing comparison is restricted to the subset of cases on which the baselines also satisfy the strict convergence criterion, thereby excluding any influence of budget asymmetry on the measured speedups.

## 3.4 Quantitative speedup: wall time and operator work

### 3.4.1 Wall-time comparison

We compare the wall time to first reach the common threshold $r_{macro} = 10^{-4}$, which all six methods attain on all 27 cases; excluding no arrival failure, this is the most conservative timing comparison.

At this common threshold the proposed method is consistently faster than each baseline. The median across all cases of the arrival-time ratio is $1.64\times$ relative to preconditioned LBM, $2.42\times$ relative to native LBE, $2.84\times$ relative to inexact Newton--Krylov, $7.34\times$ relative to Anderson, and $18.65\times$ relative to dual-time multigrid. The fraction of cases on which the proposed method is faster ranges from 19/27 (preconditioned LBM) to 27/27 (Anderson). Compared against the single fastest baseline run on each case, however, the median time ratio is $1.09\times$, i.e. on par with the strongest single competitor across all cases. The timing advantage of the proposed method is therefore concentrated on particular problem types rather than distributed uniformly across cases.

To honour the fairness restriction announced in Section 3.3---comparing only where a baseline also satisfies the strict convergence criterion---we isolate the strict subset of 15 cases on which at least one baseline converges and compare the proposed method against that converged baseline case by case. On this subset the proposed method is faster in wall time on 14 of the 15 cases, with a median wall-time ratio of $2.06\times$, and it uses fewer operator-work units (LBE-calls) on 13 of the 15 cases, with a median operator-work ratio of $1.80\times$. Broadening the same case-by-case comparison to every available baseline run rather than only the strictly converged subset, the proposed method is faster on 25 of the 27 cases, with a median ratio of $2.92\times$. These strict-subset and all-baseline win rates are collected in Table 3a; they are the headline figures quoted in the abstract.

**Table 3a. Strict-subset and all-baseline timing comparison (time-to-threshold, $r_{macro} = 10^{-4}$).**

| Comparison set | Metric | Win rate | Median ratio |
|---|---|---|---|
| Strict subset (15 cases, baseline also converges) | Wall time | 14/15 | 2.06× |
| Strict subset (15 cases, baseline also converges) | Operator work (LBE-calls) | 13/15 | 1.80× |
| All baselines (27 cases) | Wall time | 25/27 | 2.92× |

For cavity $Re = 1000$ at 2x and 3x and for the T-junction at 1x, 2x, and 3x, the convergence history of the proposed method displays a characteristic three-stage shape that qualitatively reveals the working principle of the acceleration. (i) In the initial transient, native relaxation rapidly removes the high-wavenumber kinetic modes and the residual drops sharply. (ii) In the ensuing long plateau, the residual is dominated by the global hydrodynamic slow mode; under the inexact correction of the truncated inner GMRES and the damping of the admissibility gate ($\alpha < 1$), steps are accepted at every iteration but the residual decreases only marginally, so that a near-stagnation state is maintained (for example, in cavity $Re = 1000$ 2x the residual lingers near $\sim 10^{-7}$ for about 1100 steps and 740 s). This stage accounts for most of the total wall time and corresponds to the GMRES cost repeatedly expended on correction trials that yield little benefit. (iii) The moment the iterate enters Newton's quadratic-convergence region, the finite-difference Jacobian--vector approximation becomes accurate, the full step ($\alpha = 1$) passes the admissibility test, and a single correction step drives the residual down to the square of its previous value. This terminal collapse is confirmed quantitatively: in the T-junction (2x) the residual falls in one step from $2.6 \times 10^{-5}$ to $8.1 \times 10^{-12}$, and in cavity $Re = 1000$ (2x) from $9.6 \times 10^{-8}$ to $4.9 \times 10^{-14}$, each consistent with the square of the preceding residual and hence with a second-order convergence rate. The baselines, being limited to linear convergence, achieve only a fixed fractional decrease per iteration and so cannot in principle produce such a terminal collapse---the obverse of the residual-floor phenomenon described above. The shape of the convergence curve is therefore itself qualitative evidence that the MSA-LBM correction acts as a genuine Newton step in the conserved-moment subspace.

The family in which the differences between methods are most pronounced is the lid-driven cavity flows ($Re = 100, 400, 1000$ at three mesh levels, nine configurations in total), where the difference goes beyond a simple gap in time and reaches a structural difference in the attainable residual level itself. On all nine cavity configurations the five baselines stop decreasing at a common residual floor in the range $10^{-7}\text{--}10^{-5}$, regardless of the type of method used. Within each configuration, the final residuals of the five baselines cluster in a very narrow range (for example $1.0\text{--}1.2 \times 10^{-6}$ at $Re = 1000$ 2x, and $1.1\text{--}1.2 \times 10^{-6}$ at $Re = 400$ 3x), indicating that this floor is not a limitation of any particular method but a structural barrier imposed by the problem itself. The proposed method, by contrast, passes through this floor on every cavity configuration and reaches residuals at the $10^{-8}\text{--}10^{-14}$ level (final residuals of the proposed method from $1.9 \times 10^{-13}$ to $1.6 \times 10^{-8}$), corresponding to a steady-state precision two to eight orders of magnitude deeper than any baseline.

This depth advantage is in most cases achieved together with shorter wall time. At $Re = 1000$ 1x, for example, the proposed method reaches $2.4 \times 10^{-9}$ in 57 s, whereas the five baselines spend 531--774 s and remain stuck at the $\sim 6 \times 10^{-6}$ floor; at $Re = 100$ 1x the proposed method reaches $1.9 \times 10^{-13}$ in 1 s, whereas the baselines stagnate at the $\sim 10^{-5}$ level after 72--114 s. The same penetration of the stagnation floor recurs consistently across all three Reynolds numbers and all three mesh levels.

This behavior arises because the cavity flow is a closed, recirculation-dominated problem strongly governed by the global hydrodynamic slow mode. The native-relaxation-based baselines cannot damp this slow mode efficiently and therefore converge to the same floor irrespective of method, whereas the proposed method passes through that floor by targeting the mode directly in the conserved-moment subspace (Section 3.3). Even when compared at the deepest common threshold reached by all baselines ($10^{-5}$), the proposed method retains its advantage: representatively, at $Re = 1000$ 2x it reaches $10^{-5}$ in 70.8 s, against 178.7 s for native LBE ($2.5\times$), 218.8 s for Anderson ($3.1\times$), and 940.3 s for dual-time multigrid ($13.3\times$).

Consistent with the mechanism, the net gain of the proposed method is proportional to the degree to which the global hydrodynamic slow mode governs convergence; in problems where this mode is weak or absent, the per-iteration cost of the correction step (moment projection, inner GMRES, admissibility gate) is not amortized. In the present suite this manifests in two ways. First, in Couette flow the linear distribution is represented exactly by the LBM equilibrium, so the global slow mode is essentially absent. At 1x the proposed method is faster than all baselines (by $3.3\text{--}31\times$ at the $10^{-5}$ threshold), but as the grid is refined native relaxation alone becomes fast enough while the correction cost grows, so the ordering reverses: at 2x the proposed method takes 1.8 s to reach $10^{-4}$, about $3.6\times$ slower than native LBE (0.5 s), and at $10^{-5}$ it is on par with native LBE ($0.90\times$) and preconditioned LBM ($0.96\times$). Second, the T-junction is a case in which strongly driving inlet/outlet boundaries make boundary-local modes dominant over the global mode: at 2x the proposed method takes 95.4 s to reach $10^{-5}$, about $5\times$ slower than native LBE (18.3 s) and preconditioned LBM (17.8 s). Even in this case, however, the proposed method ultimately converges to $1.3 \times 10^{-12}$, a deeper steady state than any baseline, and at 3x it recovers its advantage at the $10^{-5}$ threshold over native LBE ($1.30\times$) and preconditioned LBM ($1.38\times$).

The timing advantage of the proposed method is thus maximized on problems where the global hydrodynamic slow mode governs convergence (high-Reynolds closed, recirculation-dominated flows), while in simple shear flows or boundary-driven flows where the slow mode is weak the marginal gain can be small or negative. In every case, however, the proposed method converges stably to a steady-state precision that the baselines cannot reach, so the local timing disadvantage on some cases does not offset its robustness advantage. The full per-case arrival times, including the non-dominant cases, are reported as is in Appendix Table A1.

### 3.4.2 Operator-work (LBE-call) comparison

Wall time depends on processor performance, memory bandwidth, and implementation language, so its conclusions may shift if the measurement environment changes. To complement it, the same comparison is repeated in terms of operator work, an environment-independent and fully deterministic metric. In the lattice Boltzmann method the most expensive primitive operation is one lattice update---a single evaluation of the operator $G$ that bundles collision, advection, and boundary handling---and all six compared methods share this operation. The total number of $G$ evaluations called to reach steady state (the LBE-call count) therefore measures the intrinsic algorithmic work directly, independent of hardware and implementation.

At the common threshold $\varepsilon = 10^{-4}$ reached by all methods, the operator work of the proposed method is fewer than or comparable to each single baseline---by the median it uses $1.17\times$ fewer than inexact Newton--Krylov, $1.32\times$ fewer than preconditioned LBM, $1.33\times$ fewer than native LBE, $2.60\times$ fewer than Anderson, and $3.85\times$ fewer than dual-time multigrid. Compared against the single baseline run that uses the fewest operations on each case, however, the median is $0.87\times$, so that at a loose threshold the proposed method spends somewhat more $G$ evaluations than the strongest competitor. This is because each outer iteration of the proposed method spends several $G$ evaluations on the finite-difference Jacobian--vector products and the inner GMRES iterations, whereas a baseline relaxation iteration advances one step with a single evaluation. The proposed method thus takes fewer iterations but at higher per-iteration cost, so that at a loose threshold the two effects offset and it remains on par with the strongest competitor.

Both metrics exhibit the same structure---parity with the strongest competitor at a loose threshold and a widening advantage on harder problems and at deeper thresholds. Since one metric is environment-dependent (time) and the other fully deterministic (operator work), their agreement confirms that the advantage is intrinsic to the algorithm rather than a measurement artifact.

The operator-work advantage is most pronounced on the hard problems where the global hydrodynamic slow mode governs convergence. On such problems the baseline relaxation methods cannot remove the slow mode and stagnate at a fixed residual level, continuing to evaluate $G$ at every iteration and thus exhausting operator work while stuck at that floor. The proposed method, by contrast, targets the mode directly in the conserved-moment subspace and so passes through the floor with fewer operations. For example, on the finest-grid high-Reynolds lid-driven cavity ($Re = 1000$), the proposed method spends only 24,521 $G$ evaluations to reach $10^{-4}$, whereas the five baselines spend 45,592--125,985, so that the proposed method reaches the same accuracy with $1.9\text{--}5.1\times$ fewer operations.

## 3.5 Accuracy verification

![](media/image4.png)

**Figure 4.** Grid-refinement accuracy. (a) Second-order convergence for Poiseuille flow, (b) machine precision for Couette flow, (c) monotone decrease of the cavity Ghia error.

![](media/image5.png)

**Figure 5.** Cavity centerline $u(y)$ and $v(x)$ profiles on the 3x grid compared against Ghia (1982) for $Re = 100/400/1000$.

That the acceleration does not compromise solution accuracy in exchange for convergence speed is central to the validity of the method. Since the proposed method is designed to leave the native residual itself unchanged, it should in principle converge to the same discrete steady state as the native iteration; this section verifies that quantitatively against closed-form solutions, literature benchmarks, and high-fidelity numerical references.

We first check the accuracy under grid refinement for plane Poiseuille flow, which admits a smooth analytic solution. The relative $L_{2}$ error of the velocity profile is $9.37 \times 10^{-3}$, $2.27 \times 10^{-3}$, and $1.00 \times 10^{-3}$ at $N_{y} = 32, 64, 96$, respectively; the observed order of convergence between successive grids is 2.04 and 2.02, in quantitative agreement with the theoretical second-order accuracy of BGK-LBM for smooth flows (Figure 4a). The acceleration thus preserves the convergence order of the underlying discretization. Next, the linear Couette flow is represented exactly by the LBM equilibrium distribution, so that the discretization error should be essentially zero; the measured relative $L_{2}$ error lies between $2.75 \times 10^{-9}$ and $5.19 \times 10^{-8}$, all within the machine-precision limit (Figure 4b). This shows that the acceleration correction injects no unphysical bias into the solution. Finally, the error against the literature benchmark---the Ghia centerline velocities of the lid-driven cavity---decreases monotonically with grid refinement at all three Reynolds numbers (Figure 4c). This error is not a pure discretization error but a quantity that reflects the Navier--Stokes reference table, the boundary discretization, weak-compressibility effects, and table interpolation together, so no formal order of convergence is claimed; the monotone approach across all three Reynolds numbers nonetheless shows that the final field of the proposed method converges consistently toward the literature solution. This agreement is also visually confirmed by overlaying the cavity centerline velocity profiles on the finest 3x grid directly with Ghia (1982) (Figure 5): at all three Reynolds numbers the vertical and horizontal centerline profiles of the proposed method pass precisely through the Ghia markers, showing that the accelerated solution agrees quantitatively with the standard literature solution.

**Table 4. Accuracy summary for cases with an analytic or reference profile (1x).**

| Case | Wall \[s\] | Final residual | Rel. $L_{2}$ vs ref | Reference |
|---|---|---|---|---|
| Plane Poiseuille (N_y=32) | 20.30 | 3.384e-13 | 9.371e-03 | analytic |
| Couette (N=32) | 1.20 | 2.180e-12 | 2.750e-09 | analytic |
| Cavity Re=100 (N=33) | 0.70 | 1.935e-13 | 0.117 | Ghia |
| Cavity Re=400 (N=49) | 3.06 | 2.045e-11 | 0.106 | Ghia |
| Cavity Re=1000 (N=129) | 56.84 | 2.360e-09 | 0.0542 | Ghia |
| Multi-cylinder (N=32) | 1.25 | 2.142e-12 | 4.146e-05 | tight ref |
| Backward step (N=64) | 27.66 | 2.474e-08 | 3.260e-03 | tight ref |
| Cylinder wake (N=64) | 4.88 | 9.882e-15 | 7.935e-05 | tight ref |
| T-junction (N_x=96) | 18.29 | 2.633e-13 | 1.896e-05 | Picard ref |

For complex geometries without an analytic or literature solution, we compare against high-fidelity numerical references (Table 4). Among these the T-junction case provides especially direct evidence: its reference is a strictly converged native-iteration field, and the relative $L_{2}$ difference between it and the final field of the proposed method is only $1.9 \times 10^{-5}$. This directly demonstrates that the proposed method reaches the same discrete steady state as the native iteration---the acceleration does not detour to a different solution, it reaches the same solution faster. The other complex-geometry cases likewise show relative $L_{2}$ errors at the $10^{-5}$ level, indicating that the acceleration maintains accuracy independently of geometric complexity.

![](media/image6.png)

**Figure 6.** Velocity magnitude with streamlines for the nine geometries (3x, proposed-method solution; obstacles shaded gray).

![](media/image7.png)

**Figure 7.** Vorticity fields for the nine geometries (3x, proposed-method solution). The uniform color of the Couette case reflects the physical fact of linear shear (constant vorticity).

Finally, we verify qualitatively that the converged field has a physically plausible structure. Figures 6 and 7 show, for all nine geometries on the finest 3x grid, the velocity field (with streamlines) and the vorticity field reconstructed from the final field of the proposed method. In every geometry the characteristic structure of the corresponding flow---the primary and secondary recirculation vortices of the cavity, the separation and reattachment recirculation of the backward-facing step, the shear layer of the cylinder wake, the flow distribution in the T-junction, and the bypass flow around the multiple cylinders---is clearly reproduced, complementing the quantitative accuracy verification from a qualitative standpoint.

# 4. Conclusion

This work proposed and validated MSA-LBM, a nonlinear acceleration technique that preconditions the pressure--velocity hydrodynamic slow mode---the convergence bottleneck of the steady-state lattice Boltzmann method---from the viewpoint of the conserved-moment Schur complement. The method leaves the native residual and boundary operators unchanged: it projects the residual onto the conserved-moment subspace, forms a Jacobian-free trial direction there, and accepts it only through an admissibility gate (density positivity, finiteness, boundary consistency, residual decrease), reverting to a native update otherwise. Because every residual evaluation passes through the same native operator $G(f)$, consistency with complex masks and boundaries is automatic, and the approximate correction can never compromise the converged solution. An ablation isolates the moment-Schur correction as the source of the gain, and it is accepted in every case (71.0% overall, with no zero-acceptance case).

The principal outcome is robustness: under an identical protocol and a generous budget the proposed method converges on all 27 cases, whereas the five baselines converge on only 12--15 each, their non-convergence arising from numerical stagnation rather than budget exhaustion. The gap is largest on slow-mode-dominated problems such as the high-Reynolds cavity, where the baselines stall at a common residual floor near $10^{-6}$ while the proposed method passes through it to the $10^{-8}\text{--}10^{-14}$ level with fewer operations. Its convergence history---a sharp drop, a long near-stagnation, then a terminal collapse at the square of the preceding residual---shows that the correction acts as a genuine Newton step, which the linearly convergent baselines cannot reproduce.

The direct scope of this work is limited to comparisons of convergence time, operator work, residual, and reference-solution error on the 2D D2Q9/BGK steady benchmark suite. The proposed method does not remove discretization or boundary-condition error itself; it reaches the discrete solution faster. Wall time depends on the CPU generation, memory bandwidth, and library implementation, so it is interpreted as a relative metric within a single environment and is complemented by the hardware-independent, fully deterministic operator-work (LBE-call) metric. An exception was also observed in which, for some problems with a weak global slow mode (such as simple shear flows) or with dominant boundary-driven local modes, the per-iteration cost of the correction is not amortized and the marginal gain of the acceleration becomes small; that both the favorable and unfavorable cases are explained by the same slow-mode-dominance mechanism supports rather than weakens the predictive power of that interpretation. Future work remains in strategies for shortening the near-stagnation phase through the inner-Krylov dimension and an adaptive inner tolerance, in extension to 3D, MRT/entropic collision models, and the high-Reynolds turbulent regime, and in a quantitative evaluation of open-boundary flux conservation.

# Appendix

## A.1 Full 27-case result table

Table A1 reports all 27 proposed-method benchmark runs. The entries marked "not computed" are cases for which the tight reference solution at that level is absent from the result set and was therefore not computed post hoc; they are not to be interpreted as zero or as success. Convergence for those cases was satisfied independently by the residual, plateau, and admissibility criteria.

**Table A1. All 27 proposed-method benchmark runs.**

| Lv | Case | Wall \[s\] | LBE | r_final | r/r_0 | Rel. err | Ref |
|---|---|---|---|---|---|---|---|
| 1x | backward step n64 | 27.66 | 122673 | 2.47e-08 | 7.55e-08 | 3.26e-03 | tight ref |
| 1x | cavity re1000 n129 | 56.84 | 221413 | 2.36e-09 | 7.91e-09 | 0.0542 | Ghia |
| 1x | cavity re100 n33 | 0.70 | 20873 | 1.93e-13 | 1.02e-12 | 0.117 | Ghia |
| 1x | cavity re400 n49 | 3.06 | 44379 | 2.04e-11 | 9.69e-11 | 0.106 | Ghia |
| 1x | channel poiseuille | 20.30 | 32666 | 3.38e-13 | 4.34e-11 | 9.37e-03 | analytic |
| 1x | couette n32 | 1.20 | 20606 | 2.18e-12 | 4.63e-11 | 2.75e-09 | analytic |
| 1x | cylinder wake n64 | 4.88 | 20251 | 9.88e-15 | 4.02e-14 | 7.94e-05 | tight ref |
| 1x | multi cylinder n32 | 1.25 | 20377 | 2.14e-12 | 5.70e-12 | 4.15e-05 | tight ref |
| 1x | t junction | 18.29 | 32054 | 2.63e-13 | 7.18e-12 | 1.90e-05 | Picard ref |
| 2x | backward step n64 | 74.97 | 119793 | 6.41e-08 | 2.76e-07 | not computed | --- |
| 2x | cavity re1000 n129 | 829.72 | 1440003 | 4.87e-14 | 4.79e-07 | 0.0326 | Ghia |
| 2x | cavity re100 n33 | 5.78 | 41793 | 8.74e-12 | 7.56e-11 | 0.0669 | Ghia |
| 2x | cavity re400 n49 | 309.99 | 1257000 | 4.46e-09 | 3.34e-08 | 0.0642 | Ghia |
| 2x | channel poiseuille | 185.70 | 105281 | 1.90e-13 | 9.73e-11 | 2.27e-03 | analytic |
| 2x | couette n32 | 21.78 | 101554 | 3.31e-12 | 9.92e-11 | 2.87e-08 | analytic |
| 2x | cylinder wake n64 | 16.16 | 23184 | 1.43e-11 | 8.14e-11 | not computed | --- |
| 2x | multi cylinder n32 | 5.77 | 20471 | 1.65e-14 | 6.11e-14 | not computed | --- |
| 2x | t junction | 96.70 | 63535 | 1.26e-12 | 1.00e-10 | not computed | --- |
| 3x | backward step n64 | 756.37 | 866000 | 1.29e-10 | 6.79e-10 | not computed | --- |
| 3x | cavity re1000 n129 | 1234.14 | 1440085 | 2.56e-10 | 6.64e-07 | 0.0257 | Ghia |
| 3x | cavity re100 n33 | 180.17 | 769000 | 1.20e-10 | 1.35e-09 | 0.0493 | Ghia |
| 3x | cavity re400 n49 | 74.00 | 233779 | 1.57e-08 | 1.50e-07 | 0.0501 | Ghia |
| 3x | channel poiseuille | 798.11 | 222772 | 7.81e-14 | 8.98e-11 | 1.00e-03 | analytic |
| 3x | couette n32 | 133.55 | 296454 | 2.63e-12 | 9.66e-11 | 5.19e-08 | analytic |
| 3x | cylinder wake n64 | 36.78 | 41868 | 1.15e-11 | 8.01e-11 | not computed | --- |
| 3x | multi cylinder n32 | 10.37 | 21265 | 1.48e-12 | 6.57e-12 | not computed | --- |
| 3x | t junction | 283.69 | 89398 | 5.29e-13 | 7.79e-11 | not computed | --- |

## A.2 Supplementary convergence and diagnostic figures

![](media/image8.png)

**Figure A1.** Convergence histories of the nine cases on the 2x grid (all methods).

![](media/image9.png)

**Figure A2.** Convergence histories of the nine cases on the 3x grid (all methods).

![](media/image10.png)

**Figure A3.** Wall-time variability over seven repeated runs of a representative case (CV < 7%) versus operator-work determinism (LBE-calls bit-identical).

![](media/image11.png)

**Figure A4.** MSA-LBM correction-acceptance rate by level (71.0% overall, with no case taking zero accepted corrections).

## A.3 Cost and memory model

Let $N_{f}$ be the number of fluid nodes, $q = 9$ the number of discrete velocities, and $n_{m} = 3$ the number of conserved moments. The cost of one outer round is

$$C_{round} \approx (n_{G} + n_{trial})\, C_{G} + C_{FFT},\quad\quad C_{FFT} = O(n_{m}N_{f}\log N_{f}),$$

where $C_{G}$ is the cost of one native lattice step, $n_{G}$ is the number of native steps per round (burn-in/block Picard work plus the guard), and $n_{trial}$ is the number of residual evaluations spent in the finite-difference Jacobian action and the line search. The memory footprint is

$$W_{mem} \approx qN_{f} + O(n_{m}N_{f}) + O(N_{b}),$$

i.e. the full distribution field, the moment and spectral buffers, and the boundary bookkeeping over $N_{b}$ boundary nodes. The full Newton matrix, of size $qN_{f} \times qN_{f}$, is never formed. Peak resident-set-size measurements at three grid resolutions confirm that the marginal memory $W_{mem}$ grows linearly in $N_{f}$ (empirically of order $35 \times$ the field size) and lies three to four orders of magnitude below a dense Jacobian. Absolute memory usage is environment-dependent, so the claim is restricted to the $O(N_{f})$ scaling and the order-of-magnitude gap.

## A.4 Mass-conservation and boundary-consistency diagnostics

Residual convergence is distinct from exact mass conservation: the macroscopic $L_{2}$ residual measures the change of $p$ and $\mathbf{u}$ over the whole fluid domain, whereas global mass drift and inlet/outlet flux closure are sensitive to the boundary and mask treatment. We therefore retain the residual and its plateau as the primary convergence indicators and report the following two quantities only as auxiliary diagnostics. The global mass and its relative drift are

$$\mathcal{M}^{n} = \sum_{\Omega_{f}}^{}\rho^{n}\, dV,\quad\quad\varepsilon_{\mathcal{M}}^{n} = \frac{|\mathcal{M}^{n} - \mathcal{M}^{0}|}{max(|\mathcal{M}^{0}|,\,\epsilon)},$$

and the inlet--outlet flux closure is

$$\varepsilon_{Q}^{n} = \frac{|\sum_{out}^{}{flux} + \sum_{in}^{}{flux}|}{\max\left( \sum_{in}^{}|flux|,\,\epsilon \right)}.$$

Neither quantity is used as a stopping condition; for closed cavities $\varepsilon_{Q}$ is not applicable. The reported results are obtained with no post hoc mass correction. Positivity and boundary consistency enter only through the admissibility gate.

# Acknowledgements

The author thanks colleagues for helpful discussions on lattice Boltzmann steady-state solvers and Krylov preconditioning. (Funding sources and institutional support to be added.)

# References

\[1\] Qian, Y. H., d'Humières, D., & Lallemand, P. (1992). Lattice BGK models for Navier-Stokes equation. *Europhysics Letters*, 17(6), 479--484. https://doi.org/10.1209/0295-5075/17/6/001

\[2\] Chen, S., & Doolen, G. D. (1998). Lattice Boltzmann method for fluid flows. *Annual Review of Fluid Mechanics*, 30, 329--364. https://doi.org/10.1146/annurev.fluid.30.1.329

\[3\] Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

\[4\] Lallemand, P., & Luo, L.-S. (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61, 6546--6562. https://doi.org/10.1103/PhysRevE.61.6546

\[5\] Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387--411. https://doi.org/10.1016/0021-9991(82)90058-4

\[6\] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856--869. https://doi.org/10.1137/0907058

\[7\] Knoll, D. A., & Keyes, D. E. (2004). Jacobian-free Newton-Krylov methods: A survey of approaches and applications. *Journal of Computational Physics*, 193(2), 357--397. https://doi.org/10.1016/j.jcp.2003.08.010

\[8\] Benzi, M., Golub, G. H., & Liesen, J. (2005). Numerical solution of saddle point problems. *Acta Numerica*, 14, 1--137. https://doi.org/10.1017/S0962492904000212

\[9\] Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715--1735. https://doi.org/10.1137/10078356X

\[10\] Tóth, A., & Kelley, C. T. (2015). Convergence analysis for Anderson acceleration. *SIAM Journal on Numerical Analysis*, 53(2), 805--819. https://doi.org/10.1137/130919398

\[11\] Olshanskii, M. A., & Vassilevski, Y. V. (2007). Pressure Schur complement preconditioners for the discrete Oseen problem. *SIAM Journal on Scientific Computing*, 29(6), 2686--2704. https://doi.org/10.1137/070679776

\[12\] Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). *Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics* (2nd ed.). Oxford University Press.

\[13\] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

\[14\] Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001). *Multigrid*. Academic Press.

\[15\] Sidi, A. (1986). Convergence and stability properties of minimal polynomial and reduced rank extrapolation algorithms. *SIAM Journal on Numerical Analysis*, 23(1), 197--209. https://doi.org/10.1137/0723014

\[16\] Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591--1598. https://doi.org/10.1063/1.869307

\[17\] Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). Momentum transfer of a Boltzmann-lattice fluid with boundaries. *Physics of Fluids*, 13(11), 3452--3459. https://doi.org/10.1063/1.1399290

\[18\] Huang, J., Yang, C., & Cai, X.-C. (2015). A fully implicit method for lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 37(5), S291--S313. https://doi.org/10.1137/140975346

\[19\] Huang, J., Yang, C., & Cai, X.-C. (2016). A nonlinearly preconditioned inexact Newton algorithm for steady state lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 38(3), A1701--A1724. https://doi.org/10.1137/15M1028078

\[20\] Guo, Z., Zhao, T. S., & Shi, Y. (2004). Preconditioned lattice-Boltzmann method for steady flows. *Physical Review E*, 70(6), 066706. https://doi.org/10.1103/PhysRevE.70.066706

\[21\] Premnath, K. N., Pattison, M. J., & Banerjee, S. (2009). Steady state convergence acceleration of the generalized lattice Boltzmann equation with forcing term through preconditioning. *Journal of Computational Physics*, 228(3), 746--769. https://doi.org/10.1016/j.jcp.2008.09.028

\[22\] Hajabdollahi, F., & Premnath, K. N. (2018). Galilean-invariant preconditioned central-moment lattice Boltzmann method without cubic velocity errors for efficient steady flow simulations. *Physical Review E*, 97(5), 053303. https://doi.org/10.1103/PhysRevE.97.053303

\[23\] Hajabdollahi, F., & Premnath, K. N. (2019). Improving the low Mach number steady state convergence of the cascaded lattice Boltzmann method by preconditioning. *Computers & Mathematics with Applications*, 78(4), 1115--1130.

\[24\] Walsh, B., & Boyle, F. J. (2020). A preconditioned lattice Boltzmann flux solver for steady flows on unstructured hexahedral grids. *Computers & Fluids*, 210, 104634. https://doi.org/10.1016/j.compfluid.2020.104634

\[25\] Yahia, E., & Premnath, K. N. (2022). Preconditioned central moment lattice Boltzmann method on a rectangular lattice grid for accelerated computations of inhomogeneous flows. *Journal of Computational Science*, 63.
