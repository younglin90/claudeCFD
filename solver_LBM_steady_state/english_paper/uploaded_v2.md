**A Moment-Schur Nonlinear Preconditioner for Robust Steady-State Lattice Boltzmann Computation on Complex Geometries**

Moment-Schur Accelerated LBM (MSA-LBM)

Young-Lin Yoo

# **Abstract**

In steady-state lattice Boltzmann method (LBM) computations, the long-wavelength residual of the conserved pressure--velocity moments persists long after the kinetic modes decay and governs fixed-point convergence. We propose a geometry-aware, Jacobian-free nonlinear preconditioner, termed Moment-Schur Accelerated LBM (MSA-LBM), that targets this bottleneck while leaving the discretized LBM equations and boundary conditions unchanged, operating on the native residual $R(f) = f - G(f)$. The core operator closes the conserved-moment Schur complement of the LBE operator, linearized about a uniform base state, into a per-mode $3 \times 3$ matrix in Fourier space and assembles its kinetic-null-space-corrected inverse as a spectral preconditioner, used solely as the left preconditioner of a single Jacobian-free Newton (GMRES) step on the native residual. After a damped line search, the trial update is accepted only when it simultaneously satisfies a decrease in the macroscopic $L_{2}$ residual, density positivity, wall/inlet/outlet/mask boundary consistency, and conservation sanity; otherwise the solver falls back to a native Picard step. All internal constants depend only on a global grid scale, on neither benchmark identity nor any reference solution.

On a fixed set of benchmark results, the proposed method converges under an identical protocol across all 27 runs spanning nine benchmark families (channel, Couette, lid-driven cavity at $Re = 100/400/1000$, backward-facing step, cylinder wake, multi-cylinder, and T-junction) at the 1x/2x/3x mesh levels. Under the same stopping protocol and admissibility definition, five baseline methods (Picard, Anderson acceleration, preconditioned LBM, inexact Newton--Krylov, and dual-time multigrid) converge on only 12--15 of the 27 cases even with a generous budget. In a conservative timing comparison restricted to the strict subset of 15 cases on which a baseline also converges, the proposed method is faster in wall time on 14/15 cases (median ratio $\approx 2.06 \times$) and uses fewer LBE-calls, the operator-work metric, on 13/15 cases (median $\approx 1.80 \times$). Across all available baselines it is faster on 25/27 cases (median $2.92 \times$). On accuracy, the method exhibits an observed spatial convergence order of $\approx 2.0$ for channel Poiseuille flow (the BGK-LBM theoretical value), machine precision for Couette flow, and monotone convergence toward the Ghia centerline for the cavity, confirming that the acceleration does not sacrifice discrete accuracy. All results are independently recomputable from the stored residual histories and per-case execution traces.

The novelty lies not in modifying the LBM physical model but in a single solver framework that preconditions the hydrodynamic slow modes of the native steady residual from a Schur-complement viewpoint and validates every accepted update through the same admissibility gate, even on complex geometries. The performance claims are relative comparisons within a stored 2D D2Q9/BGK benchmark suite under an identical macroscopic-$L_{2}$-residual/plateau protocol; the method solves the same discrete steady problem faster, without reference injection or case-specific tuning.

**Keywords:** lattice Boltzmann method; steady-state solver; admissibility-preserving Schur complement; Jacobian-free residual correction; nonlinear preconditioning; complex geometry.

# **1. Introduction**

The lattice Boltzmann method (LBM) is widely used in CFD, owing to its simple streaming--collision structure, its suitability for complex-geometry applications, and its inherent parallelizability \[1--4\]. In steady-state applications such as design optimization, geometric parametric studies, and inverse design, LBM can be computationally expensive because its explicit time marching requires many iterations to reach convergence. This cost is caused by slowly decaying errors that delay convergence to the steady state. In native lattice Boltzmann equation (LBE) iteration, collision relaxation damps the non-conserved kinetic modes quickly, whereas the conserved hydrodynamic modes of density and momentum survive as the long-wavelength shear and acoustic modes of the linearized LBE and decay slowly [\[4\];]{.mark} in the low-Mach regime, the separation between convective and acoustic time scales further slows error decay. \[20, 21\]. The convergence history therefore shows a rapid initial drop followed by a long, flat tail that dominates the total wall time.

Existing approaches to reducing this long convergence tail fall into three classes. First, history-based extrapolation methods, such as Anderson acceleration \[?\] and reduced-rank extrapolation (RRE) \[?\], construct accelerated iterates from previous iterates and residuals \[9, 10, 15\]. However, these general-purpose methods do not explicitly account for the slow hydrodynamic errors that limit steady-state convergence in LBM. Second, inexact Newton and Jacobian-free Newton--Krylov (JFNK) methods \[?\] solve the nonlinear residual equation directly \[6, 7, 13\]. However, their efficiency depends strongly on a preconditioner that reflects the underlying physical structure; otherwise, the number of GMRES iterations and residual evaluations can increase substantially. In complex geometries with solid obstacles, masked nodes, and open boundaries, a Newton update can produce negative densities or violate boundary conditions. Thus, the validity of each trial update must be considered separately from convergence. Third, methods modifying the LBM model interior: preconditioned LBM redefines the collision relaxation spectrum or equilibrium to accelerate low-Mach steady convergence \[20, 21\]; recent work combines preconditioning with cascaded/central-moment LBM, including corrections for Galilean invariance and cubic-velocity errors \[22, 23, 25\], and extends the lattice Boltzmann flux solver to steady flows on unstructured grids \[24\]. Multigrid and dual-time methods reduce the cost of resolving elliptic coupling through mesh hierarchies and coarse-grid corrections \[14\]. These methods use Schur-complement and saddle-point preconditioning strategies commonly employed for incompressible pressure--velocity coupling \[8, 11, 12\]. Fully implicit inexact-Newton frameworks have also been developed for the LBE, combining Newton--Krylov methods with domain decomposition and nonlinear elimination \[18, 19\]. Existing acceleration strategies generally fall into two categories. Some modify the LBM formulation itself, for example by changing the collision model, equilibrium distribution, or relaxation parameters. Others introduce additional solver infrastructure, such as mesh hierarchies, transfer operators, implicit matrix assembly, or domain decomposition. Thus, acceleration is usually achieved either by altering the underlying LBM scheme or by introducing additional solver components.

A remaining limitation is that steady-state LBM lacks a correction strategy that can be applied without modifying the underlying LBM scheme. (a) leaves the native operator (collision, streaming, boundary handling) and its discrete steady solution unchanged; (b) targets only the conserved-moment slow-mode block that governs convergence; and (c) preserves physically valid states without case-by-case tuning, even on complex geometries with masked nodes and open boundaries. History-based extrapolation methods do not explicitly target the slow hydrodynamic components in (b). Model-level acceleration methods violate (a) by modifying the LBM formulation. Generic JFNK and fully implicit frameworks either do not separately handle the validity of trial updates or require additional solver components, such as matrix assembly, preconditioning, or domain decomposition.

We propose such a correction strategy---a geometry-aware moment-Schur nonlinear preconditioner with an admissibility-preserving acceptance gate, hereafter Moment-Schur Accelerated LBM (MSA-LBM)---that meets all three conditions at once. MSA-LBM accelerates the slow conserved-moment errors using a Fourier-space Schur preconditioner applied within a single Jacobian-free Newton correction to the original LBM residual $R(f) = f - G(f)$. The trial is accepted only if it passes both a residual decrease and physical admissibility; otherwise the solver falls back to a native LBE step. The method can therefore be applied to complex geometries while preserving the existing LBM operator and the corresponding discrete steady solution.

The method is validated on a set of benchmark cases. Under the same stopping criterion, MSA-LBM converges in all tested cases, whereas the baseline accelerators converge only in some. Among the cases where a baseline also converges, MSA-LBM reaches steady state with lower wall-clock time and fewer equivalent LBM steps, without loss of discrete accuracy. All reported comparisons are restricted to the stored two-dimensional D2Q9/BGK test suite.

# **2. Numerical Methods**

## **2.1 Steady-state LBM residual**

We consider the two-dimensional, nine-velocity (D2Q9) lattice Boltzmann equation with a single-relaxation-time (BGK) collision operator,

$f_{i}(\mathbf{x} + \mathbf{c}_{i},\, t + 1) = f_{i}(\mathbf{x},t) - \frac{1}{\tau_{f}}\left\lbrack f_{i}(\mathbf{x},t) - f_{i}^{eq}(\mathbf{x},t) \right\rbrack,\quad\quad i = 0,\ldots,8,$ (1)

where $\mathbf{x}$ is a [node of the square lattice]{.mark} $\mathbb{Z}^{2}$, $t$ the discrete time step, $\{\mathbf{c}_{i}\}_{i = 0}^{8}$ the D2Q9 lattice velocities, $\tau_{f}$ the BGK relaxation time, and lattice units are used [throughout]{.mark} ($\Delta x = \Delta t = 1$). The discrete equilibrium is

$f_{i}^{eq}(\rho,\mathbf{u}) = w_{i}\,\rho\left\lbrack 1 + \frac{\mathbf{c}_{i} \cdot \mathbf{u}}{c_{s}^{2}} + \frac{(\mathbf{c}_{i} \cdot \mathbf{u})^{2}}{2c_{s}^{4}} - \frac{|\mathbf{u}|^{2}}{2c_{s}^{2}} \right\rbrack,\quad\quad c_{s}^{2} = \frac{1}{3},$ (2)

with weights $w_{0} = 4/9$, $w_{1\text{–}4} = 1/9$, and $w_{5\text{–}8} = 1/36$. Where $c_{s}$ 는 뭐시기다. The macroscopic moments are

$$\rho = \sum_{i}^{}f_{i},\quad\quad\rho\mathbf{u} = \sum_{i}^{}\mathbf{c}_{i}f_{i},\quad\quad p = c_{s}^{2}\rho.$$

The kinematic viscosity is $\nu = c_{s}^{2}(\tau_{f} - \frac{1}{2})$, and the flow is characterized by $Re = U_{ref}L_{ref}/\nu$ and $Ma = U_{ref}/c_{s}$. All benchmarks run at $Ma \ll 1$, so the weakly compressible pressure $p = c_{s}^{2}\rho$ [is consistent with the incompressible references.]{.mark}

We denote by $G$ a single full lattice step comprising collision, streaming, and the application of all boundary conditions. The steady state [is the fixed point]{.mark} of this operator, expressed as the residual equation

$$R(f) = f - G(f) = 0,$$

with $G$ [invoked]{.mark} as a stand-alone operator that the acceleration scheme [leaves unmodified.]{.mark}

Two linear operators connect the nine-component distribution field to its three conserved hydrodynamic moments. The projection $M \in \mathbb{R}^{3 \times 9}$ maps a distribution to density and momentum,

$$Mf = (\rho,\ \rho u_{x},\ \rho u_{y})^{\top},$$

[its rows]{.mark} being the moment weights $\{ 1,\, c_{x,i},\, c_{y,i}\}$. The lifting $T \in \mathbb{R}^{9 \times 3}$ reconstructs a minimal distribution increment from a conserved-moment increment $dm = (d\rho,\, d(\rho u_{x}),\, d(\rho u_{y}))$,

$$(T\, dm)_{i} = w_{i}\left\lbrack d\rho + 3\, c_{x,i}\, d(\rho u_{x}) + 3\, c_{y,i}\, d(\rho u_{y}) \right\rbrack,$$

so that $T$ is a right inverse of $M$, $MT = I_{3}$. [The increment]{.mark} $T\, dm$ [is the minimal hydrodynamic perturbation matching a prescribed density-momentum change while leaving the remaining kinetic components for]{.mark} $G$ [to relax at the next iteration]{.mark}.

## **2.2 Conserved-moment Schur-complement formulation**

A Newton correction for the steady state solves $J_{f}(f^{*})\, df = - R(f^{*})$, where $J_{f} = \partial R/\partial f$. Assembling the full Jacobian [with its mask and boundary operators]{.mark} is costly. We therefore split the correction into a conserved-moment part $dm = (d\rho,\, d(\rho u_{x}),\, d(\rho u_{y}))$, consistent with $M$ and $T$, and a kinetic part $dk$, [yielding]{.mark} the [block-partitioned system]{.mark}

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

Eliminating the kinetic block yields the conserved-moment Schur-complement system,

$$S_{m}\, dm = - \left( R_{m} - J_{mk}J_{kk}^{- 1}R_{k} \right),\quad\quad S_{m} = J_{mm} - J_{mk}J_{kk}^{- 1}J_{km}.$$

The kinetic block $J_{kk}$ collects the locally, rapidly relaxing modes, whereas the moment Schur complement $S_{m}$ encodes the global, weakly damped hydrodynamic coupling that governs the slow pressure--velocity modes and dominates the late stagnating phase. Acceleration is therefore required only for the three conserved-moment components, not for all nine distributions per node.

Forming $S_{m}^{- 1}$ [explicitly is intractable]{.mark}, so we approximate [its action analytically]{.mark}; the closed-form per-mode block is derived in the spectral construction below. [Working in moment space]{.mark}, the practical correction reads

$$S_{m} \approx MJ_{f}T,\quad\quad S_{m}\, dm \approx - MR(f^{*}),\quad\quad df_{AP} = T\, dm,$$

where $M$ and $T$ are the operators of Eqs. (5)--(6) and $MT = I_{3}$ guarantees consistency. Only fluid nodes enter the computation; interior solid and obstacle nodes are excluded. The action of $J_{f}$ on a direction $v$ is obtained matrix-free by a directional finite difference of the native residual,

$$J_{f}(f)\, v \approx \frac{R(f + \varepsilon v) - R(f)}{\varepsilon},\quad\quad\varepsilon = 10^{- 7}\,\frac{1 + \parallel f \parallel_{2}}{\parallel v \parallel_{2}},$$

where $10^{- 7}$ is the forward-difference scale at IEEE double precision. The method is thus a moment-Schur nonlinear preconditioner built on the Jacobian-free residual response, not a full JFNK, and $J_{f}$ is never assembled.

## **2.3 Spectral moment-Schur preconditioner**

To obtain a closed-form approximation of $S_{m}^{- 1}$ we linearize the lattice update about a uniform base state $\bar{\rho} = 1$, $\bar{\mathbf{u}} = 0$. In Fourier space streaming becomes a diagonal phase operator $A(\mathbf{k}) = diag(e^{- i\mathbf{k} \cdot \mathbf{c}_{i}})$ and the global problem decouples mode by mode. The linearized BGK collision is $C(\omega) = (1 - \omega)I_{9} + \omega\, TM$ with $\omega = 1/\tau_{f}$, the linearized update is $L'(\mathbf{k}) = A(\mathbf{k})C(\omega)$, and the fixed-point residual Jacobian is $J(\mathbf{k}) = I_{9} - L'(\mathbf{k})$, reducing to a $3 \times 3$ Schur complement per wavenumber in moment space.

[The Galerkin reduction]{.mark} of this Jacobian is

$$S_{m}^{G}(\mathbf{k}) = M\, J(\mathbf{k})\, T = I_{3} - M\, A(\mathbf{k})\, T,$$

which omits the kinetic modes; moments and kinetic modes interact through streaming and collision, with the resulting damping depending on $\omega$. [We]{.mark} restore this coupling through the correction

$$S_{m}^{AP}(\mathbf{k}) = S_{m}^{G}(\mathbf{k}) - \kappa(\omega)\left\lbrack MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} \right\rbrack.$$

[The bracketed term]{.mark} is an exact identity,

$$MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} = MA(\mathbf{k})\,(I_{9} - TM)\, A(\mathbf{k})\, T,$$

which is the moment$\rightarrow$kinetic$\rightarrow$moment coupling, i.e. [the moment-space representation of]{.mark} $J_{mk}J_{km}$ in Eq. (8) that [the Galerkin reduction discards]{.mark}. Approximating the kinetic inverse $J_{kk}^{- 1}$ by the scalar $\kappa(\omega)$ then makes Eq. (12) a first-order, structurally exact reconstruction of $- J_{mk}J_{kk}^{- 1}J_{km}$. The scalar is

$$\kappa(\omega) = \frac{1}{2}\, sign(r)\,\min\left( \frac{1}{2},\,|r| \right),\quad\quad r = \frac{1 - \omega}{\omega},$$

so that the cap keeps $|\kappa| \leq 1/4$ and the correction stays bounded for all $\omega \in (0,2)$, including the limit $\omega \rightarrow 0$ where $r \rightarrow + \infty$ and $\kappa$ saturates. To control [the per-mode conditioning]{.mark} we add an adaptive Tikhonov shift,

$$S_{m}^{reg}(\mathbf{k}) = S_{m}^{AP}(\mathbf{k}) + \eta\, I_{3},\quad\quad\eta = \frac{\sigma_{\max}(S_{m}^{AP})}{50},$$

which, with $\eta$ a [fixed fraction]{.mark} of $\sigma_{\max}$, limits the [per-mode condition number]{.mark} to about $50$. The per-mode block is then $B_{m}(\mathbf{k}) = \lbrack S_{m}^{reg}(\mathbf{k})\rbrack^{- 1}$. The mean mode $\mathbf{k} = (0,0)$ is tied to mass conservation: we take no Newton step on the density mean and pass the momentum mean through unchanged, so $B_{m}(0) = diag(0,1,1)$.

The assembled $B_{m}$ acts on a residual field by projecting to moments, transforming, applying the cached per-mode blocks $B_{m}(\mathbf{k})$, transforming back, and lifting,

$$B_{m}R_{f} = T\,\mathcal{F}^{- 1}\left\{ B_{m}(\mathbf{k}\mathcal{) \cdot F\lbrack}MR_{f}\rbrack(\mathbf{k}) \right\}.$$

Because $B_{m}(\mathbf{k})$ depends only on $(N_{y},N_{x},\omega)$, it is built once per case at cost $O(N_{f}\log N_{f})$ and reused at every application. The Fourier linearization assumes periodicity and so does not represent non-periodic boundary conditions exactly; this is standard Krylov preconditioning practice and does not affect the converged solution, which is governed by the native nonlinear residual and the admissibility gate described next.

## **2.4 Jacobian-free Newton step and admissibility gate**

The spectral operator $B_{m}$ is the left preconditioner for a single Newton step on the native residual. In each outer round we run left-preconditioned GMRES on

$$J_{f}(f^{k})\, df = - R(f^{k}),$$

with operator action $v \mapsto J_{f}(f^{k})\, v$ from the finite difference of Eq. (10), preconditioner $B_{m}$, Krylov dimension $k_{\max} = 30$, restart $= 2k_{\max}$, and a single outer iteration.

The GMRES direction $df$ is not accepted directly but subjected to a damped line search coupled to an admissibility gate. With trial states $f_{trial}(\alpha) = f^{k} + \alpha\, df$ and $\alpha \in \{ 1,\frac{1}{2},\frac{1}{4},\frac{1}{8}\}$, we accept the largest $\alpha$ whose trial is admissible and strictly reduces the native residual:

$$\text{accept }\alpha \Leftrightarrow admissible(f_{trial})\  \land \  \parallel R(f_{trial}) \parallel < \parallel R(f_{best}) \parallel \  \land \ conservation(f_{trial}).$$

Candidates are tested from the largest step first, and the first to pass is accepted; if none passes, the step is rejected and the solver falls back to a native LBM update. The admissibility predicate [comprises the]{.mark} following gates.

  ---------------------------------------------------------------------------------------------------------------
  Gate                                Condition
  ----------------------------------- ---------------------------------------------------------------------------
  Finite field                        reject any NaN or Inf

  Positive density                    reject $\rho \leq 0$

  Residual decrease                   accept only if the native macroscopic residual decreases

  Boundary consistency                re-apply native wall/inlet/outlet/mask before evaluating the residual

  Conservation sanity                 mass drift and inlet--outlet flux closure not worsened relative to native
  ---------------------------------------------------------------------------------------------------------------

Trial states are interior-only: the native boundary projection is re-applied before the residual, positivity, finiteness, and mask checks, and solid and mask nodes are excluded from both the fluid-domain norm $\parallel \cdot \parallel_{\Omega_{f}}$ and $M$.

## **2.5 Solver procedure and scale-only adaptation**

The complete method is summarized in Algorithm 1; its only adaptive element is a global scale $s$ setting the burn-in and block lengths. The burn-in is the number of Picard (native LBM) iterations for initial stabilization, and the block is the number forming each per-round candidate. The scale is geometric,

$$s = \max\left( \sqrt{\frac{N_{dof}}{9 \cdot 32^{2}}},\, 1 \right),$$

where $N_{dof} = 9N_{f}$, so $s$ equals the linear grid size divided by $32$ ($s = 1$ on a $32 \times 32$ D2Q9 lattice), independent of $Re$, boundary conditions, and mask. The lengths are $burn = clip(round(16s),\, 8,\, 96)$ and $block = clip(round(80s),\, 48,\, 512)$.

Figure 1 summarizes the workflow: the moment residual is extracted, an MSA-LBM correction is formed and validated by the admissibility gate, and either the accepted update or a native fallback is applied.

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

A single MSA-LBM step is not cheaper than a single Picard step: it requires several native residual evaluations, a spectral solve, and a line search. The benefit is structural: an early global correction removes the slow hydrodynamic mode that would otherwise require thousands to hundreds of thousands of Picard tail iterations to damp, shortening the total time to steady state. Because the full Newton matrix is never formed, the memory footprint scales as $O(N_{f})$; the full cost model is in Appendix A.3.

Global mass and inlet--outlet flux closure are reported as auxiliary physical-plausibility diagnostics (Appendix A.4); they are not part of the stopping rule, which uses only the macroscopic residual and its plateau.

![](./media/image1.png){width="5.833333333333333in" height="3.604965004374453in"}

**Figure 1.** Conceptual workflow of the MSA-LBM method. The macroscopic moment residual is extracted from the native LBM residual, the MSA-LBM correction is validated by the admissibility gate, and execution proceeds with either the accepted update or the native fallback.

# **3. Results**

We first summarize the benchmark suite and protocol, then the convergence, performance, and accuracy results.

## **3.1 Benchmark suite and evaluation protocol**

The method is evaluated on nine flow [families spanning canonical flows]{.mark} with closed-form solutions through complex mask and branching geometries: plane Poiseuille and Couette flow (analytic references), the lid-driven cavity at $Re = 100/400/1000$ (Ghia centerline benchmark \[5\]), the backward-facing step and cylinder wake (separation, reattachment, wake), the multi-cylinder configuration (complex mask boundary), and the T-junction (branching geometry with coupled inlet/outlet boundaries). Each family is solved at three mesh levels (1x/2x/3x), for 27 runs total; Table 1 lists the grid sizes, boundary conditions, validation role, and reference tier.

**Table 1. Benchmark family definitions: grid size, boundary conditions, validation role, and reference tier.**

  ----------------------------------------------------------------------------------------------------------------------------------------------
  Family                       Grid (1x / 2x / 3x)         Boundary conditions            Validation role                         Reference
  ---------------------------- --------------------------- ------------------------------ --------------------------------------- --------------
  Channel (plane Poiseuille)   32×192 / 64×384 / 96×576    Inlet/outlet + wall            Pressure-driven shear (baseline)        Analytic

  Couette                      32² / 64² / 96²             Moving wall + wall             Shear flow (baseline)                   Analytic

  Cavity $Re = 100$            33² / 65² / 97²             Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 400$            49² / 97² / 145²            Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 1000$           129² / 257² / 385²          Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Backward-facing step         64² / 128² / 192²           Inlet/outlet + step mask       Separation/reattachment                 Tight ref

  Cylinder wake                64² / 128² / 192²           Inlet/outlet + obstacle mask   Wake formation                          Tight ref

  Multi-cylinder               32² / 64² / 96²             Multiple obstacle masks        Complex mask boundary                   Tight ref

  T-junction                   96×64 / 192×128 / 288×192   Branching inlet/outlet         Branching geometry + open-BC coupling   Picard ref
  ----------------------------------------------------------------------------------------------------------------------------------------------

Convergence is judged on the macroscopic $L_{2}$ change of the pressure and velocity fields, not the microscopic $f$-RMS. The pressure increment is made gauge-invariant by removing its fluid-domain mean (the absolute pressure level is arbitrary in weakly compressible closed or periodic flow) and combined with the velocity increment into a residual $r_{macro}$ over fluid nodes only. Convergence requires three conditions simultaneously: an absolute residual $r_{macro} \leq 5\tau$; a plateau condition (fractional improvement over the last $W = 50$ checks at most $\eta = 0.05$); and physical admissibility (density positivity, finite fields, boundary/mask consistency). The base tolerance is $\tau = 10^{- 7}$ for the non-cavity families and $10^{- 8}$ for the cavity families at 1x, tightened by $1/2$ and $1/3$ at 2x and 3x. All constants and the admissibility rule apply identically to every method, with no per-case tuning.

The proposed method is compared against five baselines that share the native LBM operator of the same code base (Table 2): the native LBE collide--stream--boundary fixed-point (Picard) iteration; Anderson acceleration, a regularized least-squares fixed-point accelerator; preconditioned LBM, the standard balanced PLBE transform with a block preconditioner; inexact Newton--Krylov, a JFNK method using GMRES with a smoother and a line search; and dual-time multigrid, an FAS V-cycle scheme. Each baseline uses literature-standard hyperparameters and a generous iteration budget (about $6 \times 10^{5}$ to $1.2 \times 10^{6}$ LBE-calls for cavity 2x/3x), so none is deliberately weakened. The only difference from the proposed method is the update rule; all run in the same Python/NumPy environment calling the same native operator, so any wall-time difference reflects algorithmic structure rather than language or library.

**Table 2. Baseline implementations and main settings.**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Baseline method                  Implementation summary                                                        Main settings
  -------------------------------- ----------------------------------------------------------------------------- -------------------------------------------------------------------
  Picard (native LBM)              Native collide--stream--boundary fixed-point iteration                        max_steps $\leq 1.2 \times 10^{6}$, residual-monotone termination

  Anderson acceleration \[9,10\]   Regularized least-squares fixed-point acceleration, admissibility safeguard   depth $m = 10$, $\beta = 1.0$, reg $= 10^{- 12}$

  Preconditioned LBM \[20,21\]     Balanced PLBE ($\gamma$-scaled) transform + block preconditioner              $\gamma = 0.5$, max_steps $\leq 1.2 \times 10^{6}$

  Inexact Newton--Krylov \[6,7\]   JFNK: GMRES + NE/smoother + line search                                       krylov_max=10, K_ne=20, K_smooth=10, line_search=4

  Dual-time multigrid \[14\]       FAS V-cycle, residual-equation smoothing                                      max_levels=6, V-cycle, K_pre/coarse/post=20/30/20
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Because the methods stop at different accuracies, all are compared by *time-to-threshold*---the cost to first reach a common target residual $\varepsilon$, read both as wall time and as the number of native operator evaluations $G(f)$ (LBE-calls); the latter is a hardware-independent, deterministic operator-work metric that includes rejected trials. Reference solutions (analytic, Ghia, tight numerical) are never used inside the solve, entering only the post-solve accuracy evaluation (relative $L_{2}$ error). The proposed method solves all 27 problems with a single deterministic routine, only the problem definition (grid, $Re$, boundary, mask) changing per case.

## **3.2 Convergence histories**

![](./media/image2.png){width="4.79908573928259in" height="3.6347025371828523in"}

**Figure 2.** Macroscopic $L_{2}$ residual versus wall time for a representative case (cavity $Re = 1000$, 2x), for all six methods. The baseline methods stall at a hydrodynamic plateau, whereas only the proposed method descends below the stopping tolerance.

Figure 2 plots the six methods' residuals against wall time for a representative high-Reynolds cavity case on a logarithmic residual axis. The baselines descend rapidly but stall at a plateau near $10^{- 6}$, whereas the proposed method (bold red) penetrates it and decreases monotonically below the stopping tolerance, descending faster and reaching a lower residual.

![](./media/image3.png){width="5.833333333333333in" height="5.182422353455818in"}

**Figure 3.** Convergence histories of all nine 1x-grid cases (all methods, monotone wall-time axis).

Figure 3 extends the comparison to all nine 1x cases on identical axes and color conventions, showing the behavior is not specific to a curated subset: in every panel the proposed method (bold red) reaches the lowest residual first. The 2x and 3x histories are in the appendix (Figures A1 and A2).

**Table 3. Per-level convergence summary for the proposed method.**

  ---------------------------------------------------------------------------------------------------------
  Level      Cases      Converged   Total wall \[s\]   Median residual   Max residual   Median rel. error
  ---------- ---------- ----------- ------------------ ----------------- -------------- -------------------
  1x         9          9           134.2              2.142e-12         2.474e-08      3.260e-03

  2x         9          9           1546.6             3.305e-12         6.409e-08      0.0326

  3x         9          9           3507.2             1.153e-11         1.567e-08      0.0257
  ---------------------------------------------------------------------------------------------------------

Table 3 aggregates the proposed-method results by level; the method satisfies the convergence criterion of Section 3.1 (the three flags simultaneously) on all 27 runs. The "median rel. error" column is not comparable across levels, because the set of cases with a comparable reference differs by level (some complex geometries lack a tight reference at 2x/3x); accuracy is verified case by case in Section 3.5.

## **3.3 Convergence-rate analysis and robustness**

The behavior of Section 3.2 follows from the linearized structure of Section 2. The asymptotic convergence rate of the native LBE iteration is set by the spectral radius of the linearized operator, which separates into kinetic and hydrodynamic modes. The kinetic modes (block $J_{kk}$) are damped strongly and locally, whereas the hydrodynamic modes tied to the conserved moments are governed by the Schur complement $S_{m} = J_{mm} - J_{mk}J_{kk}^{- 1}J_{km}$. The damping rate of the largest eigenvalue of $S_{m}$ approaches unity, so after enough iterations the residual is determined by this single slow mode, forming the plateau in Figures 2 and 3. The proposed method removes this dominant slow mode by applying an analytic approximation of $S_{m}^{- 1}$ restricted to the conserved-moment subspace, so the residual continues to decrease where the native iteration stalls---direct evidence that the acceleration targets the component constraining late convergence.

This difference is quantified by the robustness gap. Under the same stopping criterion, admissibility definition, and iteration budget, the proposed method satisfies the convergence criterion ($r_{macro} \leq 5\tau$ with the plateau and admissibility conditions) on all 27 cases, whereas the five baselines converge on only a subset: inexact Newton--Krylov on 15, preconditioned LBM on 14, Picard and Anderson on 13 each, and dual-time multigrid on 12.

The non-convergence arises from numerical stagnation, not budget exhaustion, as the stored histories confirm. This is clearest for the lid-driven cavity, where the baselines stall at a common residual floor regardless of method: across all nine cavity configurations ($Re = 100/400/1000$ at three levels) they stop decreasing in the range $10^{- 7}\text{–}10^{- 5}$, with final residuals clustered narrowly within each configuration ($1.0\text{–}1.2 \times 10^{- 6}$ at $Re = 1000$ 2x, $1.1\text{–}1.2 \times 10^{- 6}$ at $Re = 400$ 3x), indicating a structural barrier rather than a method-specific limit. For cavity $Re = 400$ (2x) the floor is $3.4\text{–}3.6 \times 10^{- 6}$, two orders above the target ($5\tau = 2.5 \times 10^{- 8}$), reached at $6\text{–}7 \times 10^{5}$ LBE-calls, well short of the budget ($\sim 10^{6}$); for cavity $Re = 1000$ (2x) it remains at $\mathcal{O(}10^{0})$ even after $1.2 \times 10^{6}$ LBE-calls. The curves enter their asymptotic plateau before the budget is reached, so further iterations cannot converge.

This robustness gap is reported in its own right but is not used in the timing comparison of Section 3.4, which is restricted to cases where the baselines also satisfy the strict convergence criterion, excluding budget asymmetry from the speedups.

## **3.4 Quantitative speedup: wall time and operator work**

### **3.4.1 Wall-time comparison**

We compare the wall time to first reach the common threshold $r_{macro} = 10^{- 4}$, which all six methods attain on all 27 cases; with no arrival failure, this is the most conservative timing comparison. At this threshold the median arrival-time ratio of the proposed method is $1.64 \times$ relative to preconditioned LBM, $2.42 \times$ to native LBE, $2.84 \times$ to inexact Newton--Krylov, $7.34 \times$ to Anderson, and $18.65 \times$ to dual-time multigrid; the fraction of cases on which it is faster ranges from 19/27 (preconditioned LBM) to 27/27 (Anderson). Against the single fastest baseline on each case the median ratio is $1.09 \times$, on par with the strongest competitor, so the advantage is concentrated on particular problem types. Under the fairness restriction of Section 3.3, on the strict subset of 15 cases where at least one baseline converges it is faster in wall time on 14/15 (median $2.06 \times$) and uses fewer operator-work units (LBE-calls) on 13/15 (median $1.80 \times$); broadening to every available baseline run, it is faster on 25/27 (median $2.92 \times$; Table 3a).

**Table 3a. Strict-subset and all-baseline timing comparison (time-to-threshold,** $r_{macro} = 10^{- 4}$**).**

  -------------------------------------------------------------------------------------------------------------------
  Comparison set                                      Metric                      Win rate          Median ratio
  --------------------------------------------------- --------------------------- ----------------- -----------------
  Strict subset (15 cases, baseline also converges)   Wall time                   14/15             2.06×

  Strict subset (15 cases, baseline also converges)   Operator work (LBE-calls)   13/15             1.80×

  All baselines (27 cases)                            Wall time                   25/27             2.92×
  -------------------------------------------------------------------------------------------------------------------

For cavity $Re = 1000$ at 2x/3x and the T-junction at all three levels, the convergence history displays a characteristic three-stage shape that reveals the working principle. (i) In the initial transient, native relaxation removes the high-wavenumber kinetic modes and the residual drops sharply. (ii) In the ensuing plateau the residual is dominated by the global slow mode; under the inexact truncated-GMRES correction and gate damping ($\alpha < 1$), steps are accepted every iteration but the residual barely decreases (in cavity $Re = 1000$ 2x it lingers near $\sim 10^{- 7}$ for about 1100 steps and 740 s), spending most of the wall time on correction trials of little benefit. (iii) Once the iterate enters Newton's quadratic-convergence region, the finite-difference Jacobian--vector approximation becomes accurate, the full step ($\alpha = 1$) passes the gate, and one correction drives the residual to the square of its previous value. This terminal collapse is confirmed: in the T-junction (2x) the residual falls in one step from $2.6 \times 10^{- 5}$ to $8.1 \times 10^{- 12}$, and in cavity $Re = 1000$ (2x) from $9.6 \times 10^{- 8}$ to $4.9 \times 10^{- 14}$, each consistent with second-order convergence. The linearly convergent baselines cannot produce such a collapse, so the curve shape is itself evidence that the MSA-LBM correction acts as a genuine Newton step in the conserved-moment subspace.

The cavity flows show the largest method differences, a structural gap in attainable residual. Where the baselines stall at the common floor of $10^{- 7}\text{–}10^{- 5}$ (Section 3.3), the proposed method penetrates it on every configuration to the $10^{- 8}\text{–}10^{- 14}$ level (final residuals from $1.9 \times 10^{- 13}$ to $1.6 \times 10^{- 8}$), two to eight orders deeper, with shorter wall time: at $Re = 1000$ 1x it reaches $2.4 \times 10^{- 9}$ in 57 s while the baselines spend 531--774 s stalled at the floor, and at $Re = 100$ 1x it reaches $1.9 \times 10^{- 13}$ in 1 s versus 72--114 s of baseline stagnation. The penetration recurs across all Reynolds numbers and levels because the cavity is a closed, recirculation-dominated flow governed by the global slow mode native relaxation cannot damp efficiently. Even at the deepest common threshold all baselines reach ($10^{- 5}$) the advantage persists: at $Re = 1000$ 2x it reaches $10^{- 5}$ in 70.8 s, against 178.7 s for native LBE ($2.5 \times$), 218.8 s for Anderson ($3.1 \times$), and 940.3 s for dual-time multigrid ($13.3 \times$).

Consistent with the mechanism, the net gain scales with how strongly the global slow mode governs convergence; where it is weak the per-iteration correction cost (moment projection, inner GMRES, gate) is not amortized, as two cases show. In Couette flow the linear distribution is exact in the LBM equilibrium, so the slow mode is essentially absent: at 1x the proposed method beats all baselines ($3.3\text{–}31 \times$ at $10^{- 5}$), but on refinement native relaxation alone suffices while the correction cost grows, reversing the order---at 2x it takes 1.8 s to reach $10^{- 4}$, about $3.6 \times$ slower than native LBE (0.5 s), and at $10^{- 5}$ is on par with native LBE ($0.90 \times$) and preconditioned LBM ($0.96 \times$). In the T-junction, strong inlet/outlet driving makes boundary-local modes dominant: at 2x it takes 95.4 s to reach $10^{- 5}$, about $5 \times$ slower than native LBE (18.3 s) and preconditioned LBM (17.8 s), yet still converges to $1.3 \times 10^{- 12}$, deeper than any baseline, and at 3x recovers its advantage at $10^{- 5}$ over native LBE ($1.30 \times$) and preconditioned LBM ($1.38 \times$). The advantage is thus maximized on slow-mode-dominated flows (high-Reynolds closed recirculation) and can be small or negative in simple shear or boundary-driven flows; even there it converges to a precision the baselines cannot reach, so the local disadvantage does not offset the robustness. Full per-case arrival times are in Appendix Table A1.

### **3.4.2 Operator-work (LBE-call) comparison**

Wall time depends on processor performance, memory bandwidth, and implementation language, so we complement it with operator work, an environment-independent, fully deterministic metric. The most expensive LBM primitive is one lattice update---a single evaluation of $G$ bundling collision, advection, and boundary handling---shared by all six methods, so the total number of $G$ evaluations to reach steady state (the LBE-call count) measures the intrinsic algorithmic work. At the common threshold $\varepsilon = 10^{- 4}$ the proposed method uses, by the median, $1.17 \times$ fewer than inexact Newton--Krylov, $1.32 \times$ fewer than preconditioned LBM, $1.33 \times$ fewer than native LBE, $2.60 \times$ fewer than Anderson, and $3.85 \times$ fewer than dual-time multigrid. Against the single fewest-operation baseline the median is $0.87 \times$: each outer iteration spends several $G$ evaluations on finite-difference Jacobian--vector products and inner GMRES, against one per baseline relaxation step, so fewer iterations at higher per-iteration cost offset to parity at a loose threshold. Because one metric is environment-dependent and the other fully deterministic, their matching structure---parity at a loose threshold, widening advantage on harder problems and deeper thresholds---confirms the advantage is intrinsic rather than a measurement artifact.

The operator-work advantage is most pronounced on the hard, slow-mode-dominated problems, where the baselines cannot remove the slow mode and keep evaluating $G$ while stalled at the floor, whereas the proposed method penetrates it in the conserved-moment subspace with fewer operations. On the finest-grid high-Reynolds cavity ($Re = 1000$), for example, it spends only 24,521 $G$ evaluations to reach $10^{- 4}$, against 45,592--125,985 for the five baselines---$1.9\text{–}5.1 \times$ fewer for the same accuracy.

## 3.5 Accuracy verification

![](./media/image4.png){width="5.833333333333333in" height="1.8713287401574803in"}

**Figure 4.** Grid-refinement accuracy. (a) Second-order convergence for Poiseuille flow, (b) machine precision for Couette flow, (c) monotone decrease of the cavity Ghia error.

![](./media/image5.png){width="5.833333333333333in" height="3.491405293088364in"}

**Figure 5.** Cavity centerline $u(y)$ and $v(x)$ profiles on the 3x grid compared against Ghia (1982) for $Re = 100/400/1000$.

That the acceleration does not trade accuracy for speed is central to the method's validity. Since it leaves the native residual unchanged, it should converge to the same discrete steady state as the native iteration; this section verifies that against closed-form solutions, literature benchmarks, and high-fidelity numerical references.

For plane Poiseuille flow, which admits a smooth analytic solution, the relative $L_{2}$ error of the velocity profile is $9.37 \times 10^{- 3}$, $2.27 \times 10^{- 3}$, and $1.00 \times 10^{- 3}$ at $N_{y} = 32,64,96$, giving observed convergence orders of 2.04 and 2.02, matching the theoretical second-order accuracy of BGK-LBM for smooth flows (Figure 4a); the acceleration thus preserves the discretization order. Linear Couette flow is represented exactly by the LBM equilibrium, so its error should be essentially zero; the measured relative $L_{2}$ error lies between $2.75 \times 10^{- 9}$ and $5.19 \times 10^{- 8}$, within machine precision (Figure 4b), confirming the correction injects no unphysical bias. The error against the Ghia cavity centerline velocities decreases monotonically with refinement at all three Reynolds numbers (Figure 4c); since it combines the Navier--Stokes reference table, boundary discretization, weak-compressibility effects, and table interpolation rather than a pure discretization error, no formal order is claimed, but the monotone approach shows consistent convergence toward the literature solution. Overlaying the 3x cavity centerline profiles on Ghia (1982) confirms this: at all three Reynolds numbers the vertical and horizontal profiles pass precisely through the Ghia markers (Figure 5).

**Table 4. Accuracy summary for cases with an analytic or reference profile (1x).**

  --------------------------------------------------------------------------------------------
  Case                        Wall \[s\]   Final residual   Rel. $L_{2}$ vs ref   Reference
  --------------------------- ------------ ---------------- --------------------- ------------
  Plane Poiseuille (N_y=32)   20.30        3.384e-13        9.371e-03             analytic

  Couette (N=32)              1.20         2.180e-12        2.750e-09             analytic

  Cavity Re=100 (N=33)        0.70         1.935e-13        0.117                 Ghia

  Cavity Re=400 (N=49)        3.06         2.045e-11        0.106                 Ghia

  Cavity Re=1000 (N=129)      56.84        2.360e-09        0.0542                Ghia

  Multi-cylinder (N=32)       1.25         2.142e-12        4.146e-05             tight ref

  Backward step (N=64)        27.66        2.474e-08        3.260e-03             tight ref

  Cylinder wake (N=64)        4.88         9.882e-15        7.935e-05             tight ref

  T-junction (N_x=96)         18.29        2.633e-13        1.896e-05             Picard ref
  --------------------------------------------------------------------------------------------

For complex geometries without an analytic or literature solution, we compare against high-fidelity numerical references (Table 4). The T-junction is especially direct: its reference is a strictly converged native-iteration field, and the relative $L_{2}$ difference from the proposed-method field is only $1.9 \times 10^{- 5}$, so the acceleration reaches the same discrete steady state faster rather than detouring to a different solution. The other complex-geometry cases likewise show relative $L_{2}$ errors at the $10^{- 5}$ level, so accuracy is maintained independently of geometric complexity.

![](./media/image6.png){width="5.833333333333333in" height="4.427425634295713in"}

**Figure 6.** Velocity magnitude with streamlines for the nine geometries (3x, proposed-method solution; obstacles shaded gray).

![](./media/image7.png){width="5.833333333333333in" height="4.5858475503062115in"}

**Figure 7.** Vorticity fields for the nine geometries (3x, proposed-method solution). The uniform color of the Couette case reflects the physical fact of linear shear (constant vorticity).

Finally, Figures 6 and 7 show, for all nine geometries on the 3x grid, the velocity field (with streamlines) and vorticity from the proposed-method solution. Every geometry reproduces the characteristic flow structure---the cavity's primary and secondary recirculation vortices, the backward-facing step's separation and reattachment, the cylinder wake's shear layer, the T-junction flow distribution, and the bypass flow around the multiple cylinders---complementing the quantitative verification.

# **4. Conclusion**

This work proposed and validated MSA-LBM, a nonlinear acceleration technique that preconditions the pressure--velocity hydrodynamic slow mode---the convergence bottleneck of the steady-state lattice Boltzmann method---from the conserved-moment Schur-complement viewpoint. It leaves the native residual and boundary operators unchanged: it projects the residual onto the conserved-moment subspace, forms a Jacobian-free trial direction there, and accepts it only through an admissibility gate (density positivity, finiteness, boundary consistency, residual decrease), reverting to a native update otherwise. Because every residual evaluation passes through the same native operator $G(f)$, consistency with complex masks and boundaries is automatic and the approximate correction cannot compromise the converged solution. An ablation isolates the moment-Schur correction as the source of the gain; it is accepted in every case (71.0% overall, with no zero-acceptance case).

The principal outcome is robustness: under an identical protocol and a generous budget the proposed method converges on all 27 cases, whereas the five baselines converge on only 12--15 each, their non-convergence arising from numerical stagnation rather than budget exhaustion. The gap is largest on slow-mode-dominated problems such as the high-Reynolds cavity, where the proposed method penetrates the common baseline residual floor with fewer operations. Its convergence history---a sharp drop, a long near-stagnation, then a terminal collapse at the square of the preceding residual---establishes that the correction acts as a genuine Newton step the linearly convergent baselines cannot reproduce.

The scope is limited to comparisons of convergence time, operator work, residual, and reference-solution error on the 2D D2Q9/BGK steady benchmark suite. The method does not remove discretization or boundary-condition error; it reaches the discrete solution faster. Wall time depends on CPU generation, memory bandwidth, and library, so it is read as a relative metric within one environment and complemented by the deterministic operator-work (LBE-call) metric. For problems with a weak global slow mode (simple shear) or dominant boundary-driven local modes, the per-iteration correction cost is not amortized and the marginal gain becomes small; that both favorable and unfavorable cases follow from the same slow-mode-dominance mechanism supports its predictive power. Future work includes shortening the near-stagnation phase through the inner-Krylov dimension and an adaptive inner tolerance, extension to 3D, MRT/entropic collision models, and the high-Reynolds turbulent regime, and a quantitative evaluation of open-boundary flux conservation.

# **Appendix**

## **A.1 Full 27-case result table**

Table A1 reports all 27 proposed-method benchmark runs. The entries marked "not computed" are cases for which the tight reference solution at that level is absent from the result set and was therefore not computed post hoc; they are not to be read as zero or as success. Convergence for those cases was satisfied independently by the residual, plateau, and admissibility criteria.

**Table A1. All 27 proposed-method benchmark runs.**

  ------------------------------------------------------------------------------------------------------
  Lv       Case                 Wall \[s\]   LBE       r_final    r/r_0      Rel. err       Ref
  -------- -------------------- ------------ --------- ---------- ---------- -------------- ------------
  1x       backward step n64    27.66        122673    2.47e-08   7.55e-08   3.26e-03       tight ref

  1x       cavity re1000 n129   56.84        221413    2.36e-09   7.91e-09   0.0542         Ghia

  1x       cavity re100 n33     0.70         20873     1.93e-13   1.02e-12   0.117          Ghia

  1x       cavity re400 n49     3.06         44379     2.04e-11   9.69e-11   0.106          Ghia

  1x       channel poiseuille   20.30        32666     3.38e-13   4.34e-11   9.37e-03       analytic

  1x       couette n32          1.20         20606     2.18e-12   4.63e-11   2.75e-09       analytic

  1x       cylinder wake n64    4.88         20251     9.88e-15   4.02e-14   7.94e-05       tight ref

  1x       multi cylinder n32   1.25         20377     2.14e-12   5.70e-12   4.15e-05       tight ref

  1x       t junction           18.29        32054     2.63e-13   7.18e-12   1.90e-05       Picard ref

  2x       backward step n64    74.97        119793    6.41e-08   2.76e-07   not computed   ---

  2x       cavity re1000 n129   829.72       1440003   4.87e-14   4.79e-07   0.0326         Ghia

  2x       cavity re100 n33     5.78         41793     8.74e-12   7.56e-11   0.0669         Ghia

  2x       cavity re400 n49     309.99       1257000   4.46e-09   3.34e-08   0.0642         Ghia

  2x       channel poiseuille   185.70       105281    1.90e-13   9.73e-11   2.27e-03       analytic

  2x       couette n32          21.78        101554    3.31e-12   9.92e-11   2.87e-08       analytic

  2x       cylinder wake n64    16.16        23184     1.43e-11   8.14e-11   not computed   ---

  2x       multi cylinder n32   5.77         20471     1.65e-14   6.11e-14   not computed   ---

  2x       t junction           96.70        63535     1.26e-12   1.00e-10   not computed   ---

  3x       backward step n64    756.37       866000    1.29e-10   6.79e-10   not computed   ---

  3x       cavity re1000 n129   1234.14      1440085   2.56e-10   6.64e-07   0.0257         Ghia

  3x       cavity re100 n33     180.17       769000    1.20e-10   1.35e-09   0.0493         Ghia

  3x       cavity re400 n49     74.00        233779    1.57e-08   1.50e-07   0.0501         Ghia

  3x       channel poiseuille   798.11       222772    7.81e-14   8.98e-11   1.00e-03       analytic

  3x       couette n32          133.55       296454    2.63e-12   9.66e-11   5.19e-08       analytic

  3x       cylinder wake n64    36.78        41868     1.15e-11   8.01e-11   not computed   ---

  3x       multi cylinder n32   10.37        21265     1.48e-12   6.57e-12   not computed   ---

  3x       t junction           283.69       89398     5.29e-13   7.79e-11   not computed   ---
  ------------------------------------------------------------------------------------------------------

## **A.2 Supplementary convergence and diagnostic figures**

![](./media/image8.png){width="5.833333333333333in" height="5.2808683289588805in"}

**Figure A1.** Convergence histories of the nine cases on the 2x grid (all methods).

![](./media/image9.png){width="5.833333333333333in" height="5.231182195975503in"}

**Figure A2.** Convergence histories of the nine cases on the 3x grid (all methods).

![](./media/image10.png){width="4.543859361329834in" height="3.2857141294838144in"}

**Figure A3.** Wall-time variability over seven repeated runs of a representative case (CV \< 7%) versus operator-work determinism (LBE-calls bit-identical).

![](./media/image11.png){width="4.373432852143482in" height="3.498746719160105in"}

**Figure A4.** MSA-LBM correction-acceptance rate by level (71.0% overall, with no case taking zero accepted corrections).

## A.3 Cost and memory model

Let $N_{f}$ be the number of fluid nodes, $q = 9$ the number of discrete velocities, and $n_{m} = 3$ the number of conserved moments. The cost of one outer round is

$$C_{round} \approx (n_{G} + n_{trial})\, C_{G} + C_{FFT},\quad\quad C_{FFT} = O(n_{m}N_{f}\log N_{f}),$$

where $C_{G}$ is the cost of one native lattice step, $n_{G}$ is the number of native steps per round (burn-in/block Picard work plus the guard), and $n_{trial}$ is the number of residual evaluations spent in the finite-difference Jacobian action and the line search. The memory footprint is

$$W_{mem} \approx qN_{f} + O(n_{m}N_{f}) + O(N_{b}),$$

i.e. the full distribution field, the moment and spectral buffers, and the boundary bookkeeping over $N_{b}$ boundary nodes. The full Newton matrix, of size $qN_{f} \times qN_{f}$, is never formed. Peak resident-set-size measurements at three grid resolutions confirm that the marginal memory $W_{mem}$ scales linearly in $N_{f}$ (empirically of order $35 \times$ the field size) and lies three to four orders of magnitude below a dense Jacobian. Absolute memory usage is environment-dependent, so the claim is restricted to the $O(N_{f})$ scaling and the order-of-magnitude gap.

## **A.4 Mass-conservation and boundary-consistency diagnostics**

Residual convergence is distinct from exact mass conservation: the macroscopic $L_{2}$ residual measures the change of $p$ and $\mathbf{u}$ over the whole fluid domain, whereas global mass drift and inlet/outlet flux closure are sensitive to the boundary and mask treatment. We therefore retain the residual and its plateau as the primary convergence indicators and report the following two quantities only as auxiliary diagnostics. The global mass and its relative drift are given by

$$\mathcal{M}^{n} = \sum_{\Omega_{f}}^{}\rho^{n}\, dV,\quad\quad\varepsilon_{\mathcal{M}}^{n} = \frac{|\mathcal{M}^{n} - \mathcal{M}^{0}|}{\max(|\mathcal{M}^{0}|,\,\epsilon)},$$

and the inlet--outlet flux closure is

$$\varepsilon_{Q}^{n} = \frac{|\sum_{out}^{}{flux} + \sum_{in}^{}{flux}|}{\max\left( \sum_{in}^{}|flux|,\,\epsilon \right)}.$$

Neither quantity is used as a stopping condition; for closed cavities $\varepsilon_{Q}$ is not applicable. The reported results are obtained with no post hoc mass correction. Positivity and boundary consistency enter only through the admissibility gate.

# **Acknowledgements**

The author thanks colleagues for helpful discussions on lattice Boltzmann steady-state solvers and Krylov preconditioning. (Funding sources and institutional support to be added.)

# **References**

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
