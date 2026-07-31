**A Geometry-Aware, Admissibility-Preserving Schur-Complement Nonlinear Preconditioner for Steady-State Lattice Boltzmann Solvers: Validation on Complex-Geometry Benchmarks**

Moment-Schur Accelerated LBM

Young-Lin Yoo

# Abstract

In steady-state lattice Boltzmann method (LBM) computations, the long-wavelength residual of the conserved pressure--velocity moments persists long after the kinetic modes have decayed, and it is this slowly decaying residual that governs fixed-point convergence. This paper proposes a geometry-aware, admissibility-preserving (AP) Schur-only, Jacobian-free nonlinear preconditioning technique that targets this bottleneck directly. Here and throughout, "AP" abbreviates *admissibility-preserving*---not the asymptotic-preserving schemes of kinetic theory and not any neural-network method. The method leaves the discretized LBM equations and boundary conditions unchanged and operates on the native residual $R(f) = f - G(f)$ as is. Its core operator is built by closing the conserved-moment Schur complement of the LBE operator---linearized about a uniform base state---into a per-mode $3 \times 3$ matrix in Fourier space, and assembling its kinetic-null-space-corrected, admissibility-preserving inverse as a spectral preconditioner. This preconditioner is used solely as the left preconditioner of a single Jacobian-free Newton (GMRES) step applied to the native nonlinear residual; the resulting trial update is accepted only when, after a damped line search, it simultaneously satisfies a decrease in the macroscopic $L_{2}$ residual, density positivity, wall/inlet/outlet/mask boundary consistency, and conservation sanity, and otherwise the solver falls back to a native Picard step. All internal constants depend only on a global grid scale and on neither benchmark identity nor any reference solution.

On a fixed set of benchmark results, the proposed method attains convergence under an identical protocol across all 27 runs spanning nine benchmark families (channel, Couette, lid-driven cavity at $Re = 100/400/1000$, backward-facing step, cylinder wake, multi-cylinder, and T-junction) at the 1x/2x/3x mesh levels. Under the same stopping protocol and the same admissibility definition, five baseline methods (Picard, Anderson acceleration, preconditioned LBM, inexact Newton--Krylov, and dual-time multigrid) converge on only 12--15 of the 27 cases even with a generous budget. In a conservative timing comparison restricted to the strict subset (15 cases) on which a baseline also converges, the proposed method is faster in wall time on 14/15 cases (median ratio $\approx 2.06 \times$) and uses fewer LBE-calls---the operator-work metric---on 13/15 cases (median $\approx 1.80 \times$). Broadening the comparison to all available baselines, it is faster on 25/27 cases (median $2.92 \times$). On accuracy, the method exhibits an observed spatial convergence order of $\approx 2.0$ for channel Poiseuille flow (the BGK-LBM theoretical value), machine precision for Couette flow, and monotone convergence toward the Ghia centerline for the cavity, confirming that the acceleration does not sacrifice discrete accuracy. All results are independently recomputable from the stored residual histories and per-case execution traces.

The novelty of this work lies not in modifying the LBM physical model, but in a single solver framework that preconditions the hydrodynamic slow modes of the native steady residual from a Schur-complement viewpoint and validates every accepted update through the same admissibility gate, even on complex geometries. The performance claims are confined to relative comparisons within a stored 2D D2Q9/BGK benchmark suite under an identical macroscopic-$L_{2}$-residual/plateau protocol, and the method is to be understood as a nonlinear preconditioner that solves the same discrete steady problem faster---without reference injection or case-specific tuning.

**Keywords:** lattice Boltzmann method; steady-state solver; admissibility-preserving Schur complement; Jacobian-free residual correction; nonlinear preconditioning; complex geometry.

# 1. Introduction

The lattice Boltzmann method (LBM) has become a workhorse for a wide range of CFD problems, owing to the simplicity of its streaming--collision structure and its amenability to complex-boundary treatment and parallelization \[1--4\]. In applications where the steady-state solution itself is the goal---design optimization, geometric parameter sweeps, inverse design---rather than the transient, however, the explicit time-marching nature of LBM translates directly into cost. The origin of this cost is not a requirement of temporal accuracy but the spectral structure of the fixed-point residual. In native lattice Boltzmann equation (LBE) iteration, the non-conserved kinetic modes are damped relatively quickly by collision relaxation, whereas the conserved hydrodynamic modes associated with density and momentum survive as the long-wavelength shear and acoustic modes of the linearized LBE and decay very slowly \[4\]; in the low-Mach regime in particular, the separation of convective and acoustic scales further retards this decay \[20, 21\]. As a result, the convergence history exhibits a rapid initial drop followed by a long, flat tail, and it is this tail that dominates the total wall time.

Prior efforts to mitigate this tail fall broadly into three families. First, algebraic history accelerators, exemplified by Anderson acceleration and reduced-rank extrapolation (RRE), extrapolate a descent direction from the residual correlations of the past fixed-point history \[9, 10, 15\]. These are powerful and general, but they treat the residual as a structureless vector and do not directly exploit the *physical block structure* by which, within the steady LBM residual, the kinetic fast modes and the hydrodynamic slow modes vanish on different time scales. Second, inexact Newton and Jacobian-free Newton--Krylov (JFNK) methods provide the standard framework for solving the nonlinear residual equation directly \[6, 7, 13\]. The efficiency of JFNK, however, hinges entirely on how well the preconditioner reflects the physical structure of the problem; an unsuitable preconditioner inflates the cost of both GMRES iterations and residual evaluations. Moreover, on complex geometries where masks, obstacles, and open boundaries coexist, a Newton trial step may violate density positivity or boundary consistency, so that the physical admissibility of the trial becomes a problem distinct from convergence. The third family modifies the interior of the LBM model itself. Preconditioned LBM redefines the collision relaxation spectrum or the equilibrium to accelerate low-Mach steady convergence \[20, 21\], and recent work has combined preconditioning with cascaded/central-moment LBM (including corrections for Galilean invariance and cubic-velocity errors) \[22, 23, 25\] and extended the lattice Boltzmann flux solver to steady flows on unstructured grids \[24\]. Multigrid and dual-time families relax the elliptic coupling through a mesh hierarchy and coarse-grid corrections \[14\], and the pressure-Schur/saddle-point preconditioning they borrow is a canonical framework for rapidly solving the pressure--velocity coupling of incompressible systems \[8, 11, 12\]. On a separate axis, Huang, Yang, and Cai discretized the LBE implicitly and proposed fully implicit and nonlinearly preconditioned inexact-Newton frameworks combining Newton--Krylov, domain decomposition, and nonlinear elimination \[18, 19\]. These rich bodies of prior work nonetheless share one common trait: to accelerate, they either redefine the collision model, equilibrium, or relaxation parameter (model-level acceleration), or require additional infrastructure such as a mesh hierarchy, transfer operators, implicit matrix assembly, or domain decomposition. In other words, the acceleration takes place *inside* the existing discrete LBM operator or in the *infrastructure surrounding* it.

A gap therefore emerges. Steady-state LBM acceleration still lacks an externally attached correction layer that *simultaneously* satisfies three conditions: (a) it leaves the native operator---built from collision, streaming, and boundary handling---and its discrete steady solution entirely unchanged; (b) it selectively targets only the conserved-moment slow-mode block that governs convergence; and (c) it guarantees physical admissibility, without case-by-case tuning, even on complex geometries with masks and open boundaries. Algebraic accelerators do not use the block structure of (b); model-level accelerations violate (a); and generic JFNK and implicit frameworks either do not address (c) separately or require heavy infrastructure.

This paper proposes such an externally attached acceleration layer---a geometry-aware, admissibility-preserving Schur-complement nonlinear preconditioner (hereafter AP-Schur)---that meets all three conditions at once. Throughout, "AP" denotes *admissibility-preserving*; it is unrelated to the asymptotic-preserving (AP) schemes of kinetic theory and to any neural-network method. The central idea begins from the observation that the slow components governing convergence reside in the conserved-moment block: it therefore suffices to precondition only the Schur complement of that subspace, constructed in closed form in Fourier space, and to use it solely as the preconditioner of a single Jacobian-free Newton step applied to the unchanged native residual $R(f) = f - G(f)$. The generated trial is accepted only if it passes both a residual decrease and physical admissibility, and otherwise the solver falls back to a native LBE step; the method therefore operates stably on complex geometries while leaving the existing LBM operator and its discrete steady solution untouched.

The method is validated on the 27 runs obtained by extending nine benchmark families to the 1x/2x/3x levels, together with a 1x ablation study. Under an identical stopping protocol the proposed method converges on all 27 cases, whereas the baseline accelerators converge on only a subset under the same conditions; and even in the conservative comparison restricted to cases on which a baseline also converges, the proposed method reaches steady state with shorter wall time and fewer operator-work units, without sacrificing discrete accuracy. All performance claims are made as relative comparisons within the stored 2D D2Q9/BGK suite.

# 2. Numerical Method

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

## 2.3 Spectral AP-Schur preconditioner

To obtain a closed-form approximation of $S_{m}^{- 1}$ we linearize the lattice update about a uniform base state $\bar{\rho} = 1$, $\bar{\mathbf{u}} = 0$. In Fourier space, streaming becomes a diagonal phase operator $A(\mathbf{k}) = diag(e^{- i\mathbf{k} \cdot \mathbf{c}_{i}})$, and the global problem decouples mode by mode. The linearized BGK collision is $C(\omega) = (1 - \omega)I_{9} + \omega\, TM$ with $\omega = 1/\tau_{f}$, the single linearized update is $L'(\mathbf{k}) = A(\mathbf{k})C(\omega)$, and the fixed-point residual Jacobian is $J(\mathbf{k}) = I_{9} - L'(\mathbf{k})$. Reducing to moment space gives a $3 \times 3$ Schur complement per wavenumber.

The Galerkin reduction of this Jacobian is

$$S_{m}^{G}(\mathbf{k}) = M\, J(\mathbf{k})\, T = I_{3} - M\, A(\mathbf{k})\, T,$$

which omits the influence of the kinetic modes; moments and kinetic modes interact through streaming and collision, and the resulting damping depends on $\omega$. We restore this coupling through the correction

$$S_{m}^{AP}(\mathbf{k}) = S_{m}^{G}(\mathbf{k}) - \kappa(\omega)\left\lbrack MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} \right\rbrack.$$

The bracketed correction term reflects an exact identity,

$$MA(\mathbf{k})^{2}T - (MA(\mathbf{k})T)^{2} = MA(\mathbf{k})\,(I_{9} - TM)\, A(\mathbf{k})\, T,$$

which is precisely the moment$\rightarrow$kinetic$\rightarrow$moment coupling, i.e. the moment-space representation of $J_{mk}J_{km}$ appearing in Eq. (8). The Galerkin reduction discards exactly this term; approximating the kinetic inverse $J_{kk}^{- 1}$ by the scalar $\kappa(\omega)$ then makes Eq. (12) a first-order, structurally exact reconstruction of $- J_{mk}J_{kk}^{- 1}J_{km}$. The scalar is

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

Candidates are tested from the largest step first, the first passing candidate is accepted, and if none passes the step is recorded as rejected and the solver falls back to a native LBM update. The admissibility predicate comprises:

  ---------------------------------------------------------------------------------------------------------------
  Gate                                Condition
  ----------------------------------- ---------------------------------------------------------------------------
  Finite field                        reject any NaN or Inf

  Positive density                    reject $\rho \leq 0$

  Residual decrease                   accept only if the native macroscopic residual decreases

  Boundary consistency                re-apply native wall/inlet/outlet/mask before evaluating the residual

  Conservation sanity                 mass drift and inlet--outlet flux closure not worsened relative to native
  ---------------------------------------------------------------------------------------------------------------

Trial states are interior-only: the native boundary projection is re-applied before the residual, positivity, finiteness, and mask checks, and solid and mask nodes are excluded from both the fluid-domain norm $\parallel \cdot \parallel_{\Omega_{f}}$ and the projection $M$.

## 2.5 Solver procedure and scale-only adaptation

The complete method is summarized in Algorithm 1; its only adaptive element is a global scale $s$ that sets the burn-in and block lengths. The burn-in is the number of Picard (native LBM) iterations used for initial stabilization, and the block is the number of Picard iterations forming each per-round candidate. The scale is purely geometric,

$$s = max\left( \sqrt{\frac{N_{dof}}{9 \cdot 32^{2}}},\, 1 \right),$$

where $N_{dof} = 9N_{f}$, so that $s$ equals the linear grid size divided by $32$ ($s = 1$ on a $32 \times 32$ D2Q9 lattice), independent of $Re$, boundary conditions, and mask. The lengths are $burn = clip(round(16s),\, 8,\, 96)$ and $block = clip(round(80s),\, 48,\, 512)$.

The conceptual workflow is summarized in Figure 1: the moment residual is extracted, an AP-Schur correction is formed and validated by the admissibility gate, and either the accepted update or a native fallback is applied.

> **Algorithm 1** --- AP-Schur accelerated steady-state LBM

1.  Burn-in: $f \leftarrow {Picard}^{\, burn}(f_{0})$; initialize $f_{best}$, $r_{best}$.
2.  For round $= 1,\ldots,R_{\max}$ ($R_{\max} = 160$):
    a.  Picard candidate: $c_{pic} = {Picard}^{\, block}(f)$, residual $r_{pic}$.
    b.  AP-Schur candidate: solve Eq. (17) by $B_{m}$-preconditioned GMRES for $df$, then form $c_{ap}$, $r_{ap}$ via the line-search/gate of Eq. (18).
    c.  Choose the candidate with the smallest residual.
    d.  If no candidate beats $r_{best}$ by a factor $1.02$, fall back to a native Picard guard.
    e.  Set $f \leftarrow$ chosen; update $f_{best}$, $r_{best}$, and the staleness counter.
    f.  Terminate if $r_{best} \leq \tau$ or if staleness $\geq$ stale$_{\max}$ ($= 40$).
3.  Return $f_{best}$ and the convergence history.

A single AP-Schur step is not cheaper than a single Picard step; it requires several native residual evaluations, a spectral solve, and a line search. The benefit is structural rather than per-step: an early global correction removes the slow hydrodynamic mode that would otherwise require thousands to hundreds of thousands of Picard tail iterations to damp, thereby shortening the total time to steady state. Because the full Newton matrix is never formed, the memory footprint scales as $O(N_{f})$; the full cost model is given in the appendix.

Global mass and inlet--outlet flux closure are reported as auxiliary physical-plausibility diagnostics, defined in the appendix; they are not part of the stopping rule, which uses only the macroscopic residual and its plateau.

![](media/image1.png){width="6.159722222222222in" height="3.808333333333333in"}

**Figure 1.** Conceptual workflow of the AP-Schur-only method. The macroscopic moment residual is extracted from the native LBM residual, the AP-Schur correction is validated by the admissibility gate, and execution proceeds with either the accepted update or the native fallback.

# 4. Results

**Table 5. Benchmark family definitions: grid size, boundary conditions, validation role.**

  --------------------------------------------------------------------------------------------------------------------------------------------------
  Family                       Grid (1x / 2x / 3x)         Boundary conditions            Validation role                         Reference
  ---------------------------- --------------------------- ------------------------------ --------------------------------------- ------------------
  Channel (plane Poiseuille)   32×192 / 64×384 / 96×576    Inlet/outlet + wall            Pressure-driven shear (baseline)        Analytic

  Couette                      32² / 64² / 96²             Moving wall + wall             Shear flow (baseline)                   Analytic

  Cavity $Re = 100$            33² / 65² / 97²             Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 400$            49² / 97² / 145²            Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Cavity $Re = 1000$           129² / 257² / 385²          Lid-driven, closed             Recirculating closed domain             Ghia \[5\]

  Backward-facing step         64² / 128² / 192²           Inlet/outlet + step mask       Separation/reattachment                 Tight ref

  Cylinder wake                64² / 128² / 192²           Inlet/outlet + obstacle mask   Wake formation                          Tight ref

  Multi-cylinder               32² / 64² / 96²             Multiple obstacle masks        Complex mask boundary                   Tight ref

  T-junction                   96×64 / 192×128 / 288×192   Branching inlet/outlet         Branching geometry + open-BC coupling   Tight/Picard ref
  --------------------------------------------------------------------------------------------------------------------------------------------------

제안 기법을 아홉 개 유동 문제군에서 평가하였다: 평면 Poiseuille와 Couette(해석해 존재), lid-driven cavity Re=100/400/1000(Ghia 중심선 문헌 기준 \[5\]), 후향 계단·원기둥 후류(박리·재부착·후류), 다중 원기둥(복합 마스크 경계), T-junction(분기 + 입출구 결합)이다. 각 문제군을 1x/2x/3x 세 격자에서 풀어 총 27개 실행을 수행하였다. 표 5는 각 문제군의 격자 크기, 경계조건, 검증 역할, 기준해 등급을 정리한 것으로, 단순 전단·압력 구동 유동(해석해)에서 복합 마스크·분기 형상(고정밀 수치 기준해)에 이르기까지 서로 다른 검증 역할을 포괄하도록 구성되었다.

수렴은 미시적 \$f\$-RMS가 아니라 거시 변수의 \$L_2\$ 변화량으로 판정한다. 압력 증분은 유체 영역 평균을 제거해 게이지 불변으로 만들고(약압축성·폐쇄 유동에서 절대 압력 준위는 임의 상수), 이를 속도 증분과 결합한 거시 잔차 \$r\_{\\text{macro}}\$를 정의한다. 최종 수렴은 절대 잔차 \$r\_{\\text{macro}}\\le 5\\tau\$, 최근 구간에서 감소가 멎는 plateau 조건, 그리고 밀도 양수성·유한성·경계 정합을 포함한 물리적 허용성, 세 조건의 동시 충족으로 선언한다. 기준 허용오차 \$\\tau\$는 비-cavity 문제군에서 \$10\^{-7}\$(1x), cavity 문제군에서 \$10\^{-8}\$(1x)이며 격자 레벨마다 \$1/2\$, \$1/3\$로 강화된다. 이 상수들과 plateau 창(\$W=50\$, \$\\eta=0.05\$), 허용성 규칙은 제안 기법과 모든 기준 기법에 사례 무관하게 동일하게 적용된다.

**Table 6a. Baseline implementations and main settings.**

  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Baseline method                  Implementation summary                                                        Main settings
  -------------------------------- ----------------------------------------------------------------------------- -------------------------------------------------------------------
  Picard (native LBM)              Native collide--stream--boundary fixed-point iteration                        max_steps $\leq 1.2 \times 10^{6}$, residual-monotone termination

  Anderson acceleration \[9,10\]   Regularized least-squares fixed-point acceleration, admissibility safeguard   depth $m = 10$, $\beta = 1.0$, reg $= 10^{- 12}$

  Preconditioned LBM \[20,21\]     Balanced PLBE ($\gamma$-scaled) transform + block preconditioner              $\gamma = 0.5$, max_steps $\leq 1.2 \times 10^{6}$

  Inexact Newton--Krylov \[6,7\]   JFNK: GMRES + NE/smoother + line search                                       krylov_max=10, K_ne=20, K_smooth=10, line_search=4

  Dual-time multigrid \[14\]       FAS V-cycle, residual-equation smoothing                                      max_levels=6, V-cycle, K_pre/coarse/post=20/30/20
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

비교 대상은 동일 코드베이스의 native LBM 연산자를 공유하는 다섯 기법이다(표 6a). native LBE는 collide--stream--boundary 고정점 반복이고, Anderson 가속은 정규화 최소제곱 기반 고정점 가속, preconditioned LBM은 표준 PLBE 변환에 블록 전처리를 결합한 기법, inexact Newton--Krylov는 GMRES와 line search를 사용하는 JFNK, dual-time multigrid는 FAS V-cycle 기법이다. 다섯 기법 모두 문헌 표준 하이퍼파라미터와 넉넉한 반복 예산(cavity 2x/3x에서 \$6\\times10\^{5}\$--\$1.2\\times10\^{6}\$ LBE-call)으로 구현되어 어느 것도 의도적으로 약화되지 않았다. 제안 기법과 기준 기법의 유일한 차이는 갱신 규칙뿐이며, 동일한 잔차·plateau·허용성 프로토콜로 평가된다.

방법마다 종료 시점의 정확도가 다르면 비용을 직접 견줄 수 없으므로, 모든 방법에 공통의 목표 잔차 \$\\varepsilon\$를 정하고 그 값에 처음 도달하기까지 소모한 벽시계 시간과 native 연산자 평가 횟수(LBE-call)를 비교한다(time-to-threshold). 후자는 하드웨어에 무관한 결정론적 연산량 지표로, 거부된 보정 시도의 비용까지 포함한다. 모든 기준해(해석해·Ghia·고정밀 수치해)는 풀이 과정에는 일절 사용되지 않고 수렴 후 정확도 평가에만 쓰이며, 정확도는 상대 \$L_2\$ 오차로 측정한다. 제안 기법은 사례별 경험 계수나 형상별 분기 없이 단일 결정론적 절차로 27개 문제 전부를 풀며, 사례마다 바뀌는 것은 문제 정의(격자·Re·경계·마스크)뿐이다.

**4.1 Convergence Histories**

![](media/image2.png){width="4.7756944444444445in" height="3.6243055555555554in"}

Fig. 1. 대표 사례(캐비티 Re=1000, 2x)의 거시 L_2 잔차 대 벽시계 시간(6개 방법). 기준법은 유체역학적 정체면에서 멈추고, 제안법만 허용오차 아래로 하강한다.

그림 1은 대표 사례(고-Reynolds 캐비티)에서 6개 방법의 residuals를 wall time에 대해 그린 것이고 residuals는 로그 척도이다. 기준법들은 초기에는 빠르게 감소하다가 약 10⁻⁶ 부근에서 **정체(plateau)**하여 더 이상 내려가지 못한다. 반면 제안법(굵은 적색)은 이 정체 구간을 통과하여 정지 허용오차 아래까지 단조롭게 하강한다. 제안법이 **잔차 궤적 자체가 더 빨리 내려**가고, 더 낮은 레지듀얼 값까지 수렴한다는 것을 볼 수 있다.

![](media/image3.png){width="6.299212598425197in" height="5.596317804024497in"}

**그림 2.** 1x 격자의 9개 사례 수렴 이력(전 방법, 단조 벽시계 축).

그림 2는 같은 비교를 9개 검증 사례 전부에 대해 1x 격자에서 보인 것이다. 특정 사례만 고른 것이 아님을 보이기 위해 모든 사례를 동일 축·동일 색 규칙으로 제시하였다. 모든 패널에서 제안법(굵은 적색)이 가장 낮은 잔차에 가장 먼저 도달한다. 2x·3x 격자의 동일 그리드는 부록 그림 A1·A2에 수록한다.

**표 1.** 제안법의 레벨별 수렴 요약.

  -------------------------------------------------------------------------------------------------------------------
  **Level**   **Cases**   **Converged**   **총 wall \[s\]**   **잔차 중앙값**   **최대 잔차**   **상대오차 중앙값**
  ----------- ----------- --------------- ------------------- ----------------- --------------- ---------------------
  1x          9           9               134.2               2.142e-12         2.474e-08       3.260e-03

  2x          9           9               1546.6              3.305e-12         6.409e-08       0.0326

  3x          9           9               3507.2              1.153e-11         1.567e-08       0.0257
  -------------------------------------------------------------------------------------------------------------------

표 1은 제안법의 레벨별 집계 결과다. 제안법은 27개 실행 전부에서 식 (5)의 수렴 판정을 충족하였다(세 수렴 flag 동시 만족). 표 1의 \"상대오차 중앙값\"은 레벨마다 비교 가능한 기준해를 가진 사례 집합이 달라지므로(2x·3x에서 복잡 형상 일부 제외) 레벨 간 직접 비교 대상이 아니다. 정확도는 §4.4에서 사례별로 따로 검증한다.

**4.2 Convergence-Rate Analysis and Robustness**

§4.1에서 관측된 거동은 §2의 선형화 구조로부터 직접 설명된다. native LBE 반복의 점근 수렴률은 선형화 반복 연산자의 스펙트럼 반경에 의해 지배되며, 이 연산자는 운동학적(kinetic) 모드와 유체역학적(hydrodynamic) 모드로 분리된다. 운동학적 모드는 kinetic 블록 \$J\_{kk}\$에 대응하여 국소적으로 강하게 감쇠되는 반면, 보존 모멘트(밀도·운동량)에 결부된 유체역학적 모드는 Schur 보완 \$S_m = J\_{mm} - J\_{mk}J\_{kk}\^{-1}J\_{km}\$이 지배한다. \$S_m\$의 최대 고유값에 해당하는 감쇠율은 1에 점근하므로, 충분히 많은 반복 이후 잔차는 이 단일한 느린 모드에 의해 결정되며 그림 1·2에서 관측되는 정체면(plateau)을 형성한다. 제안 기법은 보존 모멘트 부분공간에 한정하여 \$S_m\^{-1}\$의 해석적 근사를 적용함으로써 이 지배적 느린 모드를 직접 제거한다. 그 결과 native 반복이 정체하는 영역에서도 잔차가 지속적으로 감소하며, 이는 가속이 후반 수렴을 제약하는 성분을 선택적으로 표적함을 보인다.

이 메커니즘적 차이는 방법 간 강건성 격차로 정량화된다. 동일한 정지 기준과 동일한 허용성 정의, 그리고 모든 방법에 동일하게 부여된 반복 예산 아래에서, 제안 기법은 27개 사례 전부에서 수렴 판정(\$r\_{\\text{macro}}\\le 5\\tau\$, 정체, 허용성의 동시 충족)을 만족하였다. 반면 다섯 개 기준 기법은 동일 조건에서 일부 사례에서만 수렴하였는데, 가장 강건한 inexact Newton--Krylov가 15개, preconditioned LBM이 14개, Picard와 Anderson이 각각 13개, dual-time multigrid가 12개에 그쳤다.

기준 기법의 미수렴은 반복 예산의 소진이 아니라 수치적 정체에서 기인하며, 이는 저장된 수렴 이력으로부터 직접 확인된다. 일례로 cavity Re=400(2x)에서 다섯 기준 기법은 모두 목표 허용오차(\$5\\tau = 2.5\\times10\^{-8}\$)보다 약 두 자릿수 높은 \$3.4\\text{--}3.6\\times10\^{-6}\$ 수준에서 잔차 감소가 정지하였으며, 이 정체는 \$6\\text{--}7\\times10\^{5}\$ LBE-call 시점에 이미 발생하여 예산 한계(\$\\sim\$\$10\^{6}\$)에 한참 못 미친다. cavity Re=1000(2x)에서는 \$1.2\\times10\^{6}\$ LBE-call 이후에도 잔차가 \$\\mathcal{O}(10\^{0})\$에 머물렀다. 즉 이들 사례의 잔차 곡선은 예산 도달 이전에 점근 정체에 진입하였으므로, 추가 반복으로는 수렴이 달성되지 않는다.

본 강건성 격차는 그 자체로 보고하되, 정량적 시간 성능 비교(§4.3)에서는 사용하지 않는다. 방법 간 공정성을 보장하기 위해, 시간 비교는 기준 기법 또한 엄격 수렴 판정을 만족하는 사례 부분집합으로 한정하여 수행하며, 이로써 예산 비대칭이 속도 향상 측정에 개입할 여지를 배제한다.

**4.3 Quantitative Speedup: Wall Time and Operator Work**

**4.3.1 Wall-time comparison**

방법마다 정지 규칙이 달라 종료 시점이 일치하지 않으므로, 종료 시점에 기반한 비교는 서로 다른 정확도의 해를 대조하게 되어 공정하지 않다. 이를 피하기 위해 본 연구는 모든 방법을 \*\*동일한 잔차 수준에 처음 도달하기까지의 벽시계 시간(time-to-threshold)\*\*으로 비교한다. 여섯 방법 모두 잔차를 동일 정의(\$\\texttt{macro_l2_p_ux_uy_uz}\$)로 기록하므로, 공통 임계값 \$r\_{\\text{macro}} = 10\^{-4}\$를 기준으로 각 방법의 첫 도달 시각을 직접 대조할 수 있다. 이 임계값은 여섯 방법 모두 27개 사례 전부에서 도달하는 수준으로, 어떠한 도달 실패도 비교에서 배제되지 않으므로 가장 보수적인 시간 비교를 제공한다.

이 공통 임계값에서 제안 기법은 각 기준 기법보다 일관되게 빠르다. 전 사례에 대한 도달 시간비의 중앙값은 Preconditioned LBM 대비 1.64배, native LBE 대비 2.42배, Inexact Newton--Krylov 대비 2.84배, Anderson 대비 7.34배, Dual-time multigrid 대비 18.65배이다. 제안 기법이 더 빠른 사례의 비율은 기준 기법에 따라 19/27(Preconditioned LBM)에서 27/27(Anderson)에 이른다. 다만 각 사례에서 가장 빠른 단일 기준 실행과 비교하면 시간비 중앙값은 1.09배로, 모든 사례에 걸쳐 단일 최강 경쟁자와 대등한 수준이다. 즉 제안 기법의 시간 이점은 사례 전반에 균일하게 분포하기보다, 특정 유형의 문제에 집중되어 나타난다.

Cavity Re=1000 2x, 3x와 T-junction 1x, 2x, 3x 에서는 제안 기법의 수렴 이력은 세 단계로 구성된 특징적 형태를 보이며, 이는 가속의 작동 원리를 정성적으로 드러낸다. (i) 초기 과도 구간에서는 native 완화가 고파수 운동학적 모드를 신속히 제거하여 잔차가 급격히 감소한다. (ii) 이어지는 장기 정체 구간에서는 잔차가 전역 유체역학적 느린 모드에 지배되어, 절단된 내부 GMRES에 의한 부정확(inexact) 보정과 허용성 게이트의 댐핑(\$\\alpha\<1\$) 아래 매 스텝 채택은 되나 잔차가 거의 감소하지 않는 준정체(near-stagnation) 상태가 유지된다(예: cavity Re=1000 2x에서 잔차가 \$\\sim10\^{-7}\$에 약 1100 스텝·740초 동안 머묾). 본 단계가 전체 벽시계 시간의 대부분을 차지하며, 이는 효과를 거의 내지 못하는 보정 시도에 반복적으로 소요되는 GMRES 비용에 해당한다. (iii) 반복점이 Newton의 2차 수렴 영역에 진입하는 순간, 유한차분 Jacobian--벡터 근사가 정확해지고 전스텝(\$\\alpha=1\$)이 허용성 판정을 통과하여 단일 보정 스텝이 잔차를 직전 값의 제곱 수준으로 급강하시킨다. 이 종단 붕괴는 정량적으로 확인된다. T-junction(2x)에서 잔차는 한 스텝에 \$2.6\\times10\^{-5}\$에서 \$8.1\\times10\^{-12}\$로, cavity Re=1000(2x)에서 \$9.6\\times10\^{-8}\$에서 \$4.9\\times10\^{-14}\$로 감소하여, 각각 직전 잔차의 제곱에 부합하는 2차 수렴률을 나타낸다. 선형 수렴에 그치는 기준 기법들은 매 반복에서 일정 비율의 감소만 달성하므로 이러한 종단 붕괴를 원리적으로 생성할 수 없으며, 이것이 앞서 기술한 잔차 바닥 현상의 이면이다. 따라서 수렴 곡선의 형태 자체가 AP-Schur 보정이 보존 모멘트 부분공간에서 실질적 Newton 스텝으로 작동함을 보이는 정성적 증거가 된다.

방법 간 차이가 가장 극명하게 드러나는 사례군은 lid-driven cavity 유동들(Re=100, 400, 1000 × 세 격자 레벨, 총 9개 구성)이며, 여기서는 단순한 시간 차이를 넘어 도달 가능한 잔차 수준 자체의 구조적 차이가 관측된다. 9개 cavity 구성 전부에서 다섯 기준 기법은 사용한 기법의 종류와 무관하게 약 \$10\^{-7}\\text{--}10\^{-5}\$ 범위의 공통된 잔차 바닥(residual floor)에서 감소를 멈춘다. 특히 각 구성 내에서 다섯 기준 기법의 최종 잔차는 서로 매우 좁은 범위에 모이는데(예: Re=1000 2x에서 \$1.0\\text{--}1.2\\times10\^{-6}\$, Re=400 3x에서 \$1.1\\text{--}1.2\\times10\^{-6}\$), 이는 이 바닥이 특정 기법의 한계가 아니라 문제 자체가 부과하는 구조적 장벽임을 보여준다. 반면 제안 기법은 모든 cavity 구성에서 이 바닥을 통과하여 \$10\^{-8}\\text{--}10\^{-14}\$ 수준의 잔차까지 도달하며(제안 기법 최종 잔차 \$1.9\\times10\^{-13}\$ \~ \$1.6\\times10\^{-8}\$), 이는 어떤 기준 기법보다도 2--8자릿수 깊은 정상상태 정밀도에 해당한다.

이러한 깊이 우위는 대체로 더 짧은 벽시계 시간과 함께 달성된다. 예컨대 Re=1000 1x에서 제안 기법은 57초 만에 \$2.4\\times10\^{-9}\$에 도달한 반면 다섯 기준 기법은 531--774초를 소진하고도 \$\\sim6\\times10\^{-6}\$ 바닥에 머물렀고, Re=100 1x에서는 제안 기법이 1초 만에 \$1.9\\times10\^{-13}\$에 도달한 반면 기준 기법들은 72--114초 뒤 \$\\sim10\^{-5}\$ 수준에서 정체하였다. 동일한 정체-바닥 대비 관통(penetration) 양상이 세 Reynolds 수와 세 격자 레벨 전반에서 일관되게 반복된다.

이 현상은 cavity 유동이 전역 유체역학적 느린 모드에 강하게 지배되는 폐쇄·재순환 지배 문제이기 때문이다. native 완화 기반 기준 기법들은 이 느린 모드를 효율적으로 감쇠시키지 못해 방법에 무관하게 동일한 바닥에 수렴하는 반면, 제안 기법은 보존 모멘트 부분공간에서 이 모드를 직접 표적함으로써 그 바닥을 통과한다(§4.2). 기준 기법들이 모두 도달하는 가장 깊은 공통 임계값(\$10\^{-5}\$)에서의 시간 비교에서도 제안 기법은 우위를 유지하여, 대표적으로 Re=1000 2x에서 제안 기법은 70.8초에 \$10\^{-5}\$에 도달한 반면 native LBE는 178.7초(2.5배), Anderson은 218.8초(3.1배), Dual-time multigrid는 940.3초(13.3배)를 소요하였다.

메커니즘에 따르면 제안 기법의 순이득은 전역 유체역학적 느린 모드가 수렴을 지배하는 정도에 비례하며, 이 모드가 약하거나 부재한 문제에서는 모멘트 투영·내부 GMRES·허용성 게이트로 구성된 보정 단계의 단위 반복 비용이 상쇄되지 못한다. 본 벤치마크에서 이는 두 유형으로 나타난다. 첫째, Couette 유동은 선형 분포가 LBM 평형으로 정확히 표현되어 전역 느린 모드가 본질적으로 부재한다. 1x 격자에서는 제안 기법이 모든 기준 기법보다 빠르지만(\$10\^{-5}\$ 기준 3.3--31배), 격자가 세밀해지면 native 완화만으로 충분히 빠른 반면 보정 비용은 증가하여 우열이 역전된다 --- 2x에서 제안 기법은 \$10\^{-4}\$ 도달에 1.8초로 native LBE(0.5초)보다 약 3.6배 느렸고, \$10\^{-5}\$에서는 native LBE(0.90배)·Preconditioned LBM(0.96배)과 비등하였다. 둘째, T-junction은 입출구 경계가 강하게 구동하여 경계 국소 모드가 전역 모드보다 지배적인 경우로, 2x 격자에서 제안 기법은 \$10\^{-5}\$까지 95.4초를 소요하여 native LBE(18.3초)·Preconditioned LBM(17.8초)보다 약 5배 느렸다. 다만 동일 사례에서도 제안 기법은 최종적으로 \$1.3\\times10\^{-12}\$까지 수렴하여 기준 기법보다 깊은 정상상태에 도달하였고, 3x 격자에서는 \$10\^{-5}\$ 기준 native LBE(1.30배)·Preconditioned LBM(1.38배) 대비 우위를 회복하였다.

제안 기법의 시간 이점은 전역 유체역학적 느린 모드가 수렴을 지배하는 문제(고-Reynolds 폐쇄·재순환 지배 유동)에서 극대화되며, 느린 모드가 약한 단순 전단류나 경계 구동 지배 유동에서는 한계 이득이 작거나 음일 수 있다. 그러나 모든 사례에서 제안 기법은 기준 기법이 도달하지 못하는 정상상태 정밀도까지 안정적으로 수렴하므로, 일부 사례의 국소적 시간 열세가 강건성 우위를 상쇄하지는 않는다. 비우세 사례를 포함한 전체 사례별 도달 시간은 그림 3과 부록 표 A1에 그대로 수록하였다.

### 4.3.2 연산량(LBE-call) 비교

벽시계 시간은 프로세서 성능·메모리 대역폭·구현 언어에 따라 달라지므로, 측정 환경이 바뀌면 결론이 흔들릴 수 있다. 이를 보완하기 위해 환경에 무관하고 완전히 결정론적인 지표인 연산량(operator-work)으로 동일한 비교를 반복한다. 격자 볼츠만 방법에서 가장 비용이 큰 기본 연산은 1회의 격자 업데이트, 즉 충돌·이류·경계처리를 묶은 연산자 \$G\$의 한 번 평가이며, 비교 대상 여섯 방법이 모두 이 연산을 공유한다. 따라서 정상상태에 도달하기까지 호출한 \$G\$의 총 횟수(이하 LBE-call)는 하드웨어·구현과 독립한 알고리즘 고유의 작업량을 직접 측정한다.

비교는 벽시계 시간과 동일하게, 모든 방법에 공통의 목표 잔차 \$\\varepsilon\$를 정하고 그 값에 처음 도달하는 순간까지 소모한 LBE-call로 수행한다. 즉 잔차가 처음 \$\\varepsilon\$ 이하로 떨어진 반복에서의 누적 \$G\$ 평가 횟수를 각 방법의 도달 연산량으로 정의하며, 도달하지 못하고 정체하는 방법은 비교에서 제외한다. 여섯 방법 모두 잔차를 동일한 정의로 기록하므로 이 재계산은 저장된 동일 결과 집합에서 일관되게 이루어진다.

모든 방법이 도달하는 공통 임계값 \$\\varepsilon=10\^{-4}\$에서, 제안 기법의 도달 LBE-call은 각 단일 기준 기법보다 적거나 비등하다 --- 중앙값으로 Inexact Newton--Krylov 대비 1.17배, Preconditioned LBM 대비 1.32배, native LBE 대비 1.33배, Anderson 대비 2.60배, Dual-time multigrid 대비 3.85배 적은 연산을 사용한다. 다만 각 사례에서 가장 적은 연산을 쓴 단일 기준 실행과 비교하면 중앙값은 0.87배로, 느슨한 임계값에서는 제안 기법이 최강 경쟁자보다 다소 많은 \$G\$ 평가를 소모한다. 이는 제안 기법의 매 외부 반복이 유한차분 Jacobian--벡터 곱과 내부 GMRES 반복에 복수의 \$G\$ 평가를 소모하는 반면, 기준 완화 반복은 1회 평가만으로 한 스텝을 진행하기 때문이다. 즉 제안 기법은 반복 횟수는 적지만 반복당 연산비가 높아, 느슨한 임계값에서는 두 효과가 상쇄되어 최강 경쟁자와 대등한 수준에 머문다.

이 연산량 비교가 벽시계 시간 비교와 동일한 정성적 구조를 보인다는 점은 중요하다. 두 지표 모두 느슨한 임계값에서는 제안 기법이 최강 경쟁자와 대등하고, 우위는 어려운 문제와 더 깊은 임계값으로 갈수록 확대된다. 한 지표는 환경 의존적(시간)이고 다른 하나는 완전히 결정론적(연산량)임에도 두 결론이 일치한다는 사실은, 관측된 성능 우위가 인터프리터 overhead나 일시적 스케줄링 변동 같은 측정 부수효과가 아니라 알고리즘 고유의 성질임을 확인해 준다.

연산량 이점은 전역 유체역학적 느린 모드가 수렴을 지배하는 어려운 문제에서 가장 뚜렷하다. 이러한 문제에서 기준 완화 기법들은 느린 모드를 제거하지 못해 일정 잔차 수준에서 정체하며, 그 바닥에 머무는 동안에도 매 반복마다 \$G\$를 평가하여 연산량을 계속 소진한다. 반면 제안 기법은 보존 모멘트 부분공간에서 이 모드를 직접 표적하므로 더 적은 연산으로 그 바닥을 통과한다. 예컨대 가장 세밀한 격자의 고-Reynolds lid-driven cavity(Re=1000)에서 제안 기법은 \$10\^{-4}\$ 도달에 24,521회의 \$G\$ 평가만을 소모한 반면, 다섯 기준 기법은 45,592--125,985회를 소모하여 제안 기법이 1.9--5.1배 적은 연산으로 동일 정확도에 도달하였다.

또한, 보고된 모든 연산량에는 실패한 보정 시도의 비용이 빠짐없이 포함된다. 제안 기법의 보정 시도가 허용성 또는 잔차감소 판정을 통과하지 못하면 그 시도는 폐기되고 기준 완화로 fallback하지만, 시도 과정에서 수행된 잔차 평가·경계 재적용·유한성 및 양수성 검사·fallback 단계의 \$G\$ 평가가 모두 기록된 LBE-call에 산입된다. 따라서 보고한 연산량은 성공한 보정만을 선별한 사후 측정이 아니라, 실패 시도를 포함한 실제 실행 경로 전체의 비용이다.

## 4.4 정확도 검증: 가속이 해를 왜곡하지 않는가 (Accuracy)

![](media/image4.png){width="6.299212598425197in" height="2.0207830271216096in"}

**그림 4.** 격자 미세화 정확도. (a) Poiseuille 2차 수렴, (b) Couette 기계정밀도, (c) 캐비티 Ghia 오차 단조 감소.

![](media/image5.png){width="6.299212598425197in" height="3.7702471566054245in"}

**그림 5.** 3x 격자 캐비티 중심선 u(y)·v(x) 프로파일의 Ghia(1982) 대비 비교(Re=100/400/1000).

가속 절차가 수렴 속도를 높이는 대가로 해의 정확도를 훼손하지 않는다는 점은 본 기법의 타당성에 핵심적이다. 제안 기법은 native 잔차 자체를 변경하지 않도록 설계되었으므로 원리적으로는 native 반복과 동일한 이산 정상상태로 수렴해야 하며, 본 절은 이를 폐형식 해, 문헌 기준해, 고정밀 수치 기준해에 대해 정량적으로 검증한다.

먼저 매끄러운 해석해를 갖는 평면 Poiseuille 유동에서 격자 미세화에 따른 정확도를 확인한다. 속도 프로파일의 상대 \$L_2\$ 오차는 \$N_y=32,64,96\$에서 각각 \$9.37\\times10\^{-3}\$, \$2.27\\times10\^{-3}\$, \$1.00\\times10\^{-3}\$이며, 인접 격자 간 관측 수렴 차수는 2.04와 2.02로 BGK-LBM이 매끄러운 유동에 대해 갖는 이론적 2차 정확도와 정량적으로 일치한다(그림 4a). 즉 가속 절차는 기저 이산화의 수렴 차수를 그대로 보존한다. 다음으로 선형 Couette 유동은 LBM 평형 분포로 정확히 표현되어 이산오차가 본질적으로 0이어야 하는 경우로, 측정된 상대 \$L_2\$ 오차는 \$2.75\\times10\^{-9}\$에서 \$5.19\\times10\^{-8}\$ 수준에 머물러 모두 기계정밀도 한계 안에 있다(그림 4b). 이는 가속 보정이 해에 어떠한 비물리적 편향도 주입하지 않음을 보인다. 끝으로 문헌 기준해인 lid-driven cavity의 Ghia 중심선 속도와의 오차는 세 Reynolds 수 모두에서 격자 미세화에 따라 단조 감소한다(그림 4c). 이 오차는 순수 이산오차가 아니라 Navier--Stokes 기준표·경계 이산화·약압축성 효과·표 보간이 함께 반영된 양이므로 형식적 수렴 차수를 주장하지는 않으나, 세 Reynolds 수 전반에서의 단조 접근은 제안 기법의 최종 장이 문헌 해로 일관되게 수렴함을 보인다. 이 일치는 가장 세밀한 3x 격자에서 캐비티 중심선 속도 프로파일을 Ghia(1982)와 직접 겹쳐 보면 시각적으로도 확인된다(그림 5). 세 Reynolds 수 모두에서 제안 기법의 수직·수평 중심선 프로파일이 Ghia의 표식점을 정밀하게 통과하여, 가속된 해가 표준 문헌 해와 정량적으로 부합함을 보인다.

**표 4.** 해석해/기준 프로파일이 있는 사례의 정확도 요약(1x).

  -------------------------------------------------------------------------------------------------------
  **Case**                    **Wall \[s\]**   **Final residual**   **Rel. L_2 vs ref**   **Reference**
  --------------------------- ---------------- -------------------- --------------------- ---------------
  Plane Poiseuille (N_y=32)   20.30            3.384e-13            9.371e-03             analytic

  Couette (N=32)              1.20             2.180e-12            2.750e-09             analytic

  Cavity Re=100 (N=33)        0.70             1.935e-13            0.117                 Ghia

  Cavity Re=400 (N=49)        3.06             2.045e-11            0.106                 Ghia

  Cavity Re=1000 (N=129)      56.84            2.360e-09            0.0542                Ghia

  Multi-cylinder (N=32)       1.25             2.142e-12            4.146e-05             tight ref

  Backward step (N=64)        27.66            2.474e-08            3.260e-03             tight ref

  Cylinder wake (N=64)        4.88             9.882e-15            7.935e-05             tight ref

  T-junction (N_x=96)         18.29            2.633e-13            1.896e-05             Tight ref
  -------------------------------------------------------------------------------------------------------

해석해나 문헌해가 없는 복잡 형상에 대해서는 고정밀 수치 기준해와 비교한다(표 4). 그중 T-junction 사례는 특히 직접적인 증거를 제공하는데, 그 기준해가 엄격 수렴한 native 반복 장이며 제안 기법의 최종 장과의 상대 \$L_2\$ 차이가 \$1.9\\times10\^{-5}\$에 불과하다. 이는 제안 기법이 native 반복과 동일한 이산 정상상태에 도달했음을 직접 입증한다 --- 가속은 다른 해로 우회하는 것이 아니라 동일한 해에 더 빠르게 도달하는 것이다. 다른 복잡 형상 사례들에서도 상대 \$L_2\$ 오차는 \$10\^{-5}\$ 수준으로, 가속이 형상의 복잡성과 무관하게 정확도를 유지함을 보인다.

![](media/image6.png){width="6.299212598425197in" height="4.781022528433946in"}

**그림 6.** 9개 형상의 속도 크기 + 유선(3x, 제안법 해; 장애물은 회색 음영).

![](media/image7.png){width="6.299212598425197in" height="4.952096456692914in"}

**그림 7.** 9개 형상의 와도장(3x, 제안법 해). Couette의 균일색은 선형 전단(일정 와도)이라는 물리적 사실을 반영한다.

끝으로 수렴된 장이 물리적으로 타당한 구조를 갖는지 정성적으로 확인한다. 그림 6과 7은 가장 세밀한 3x 격자에서 아홉 개 모든 형상에 대해 제안 기법의 최종 장으로부터 재구성한 속도장(유선 포함)과 와도장을 보인다. 모든 형상에서 해당 유동의 특징적 구조 --- 캐비티의 주·부 재순환 와류, 후향 계단의 박리·재부착 재순환, 원기둥 후류의 전단층, T-분기에서의 유동 분배, 다중 원기둥 주위의 우회류 --- 가 명확히 재현되어, 정량적 정확도 검증을 정성적 측면에서 보완한다.

# 6. Conclusion

본 연구는 정상상태 격자 볼츠만 방법(LBM)의 수렴 병목인 압력--속도 유체역학적 느린 모드를, 보존 모멘트 Schur 보완(Schur-complement)의 관점에서 전처리하는 비선형 가속 기법을 제안하고 검증하였다. 표준 LBM의 collide--stream 완화는 운동학적(kinetic) 성분을 국소적으로 안정하게 감쇠시키지만, 전역적으로 평형에 도달해야 하는 압력--속도 모드는 매우 느리게 수렴하여 후반 정체(plateau)를 유발한다. 제안 기법은 이 느린 성분만을 표적으로 삼아, 잔차를 보존 모멘트 공간으로 투영하고 그 부분공간에서 Jacobian-free 시행 방향을 생성한 뒤 다시 분포함수 공간으로 들어올린다. 핵심 설계 원칙은 native 정상상태 LBM 잔차와 경계 연산자를 일절 변경하지 않는다는 것으로, 모든 잔차 평가가 동일한 native 연산자 \$G(f)\$를 거치므로 복합 마스크·벽·입출구 처리와의 정합성이 자동으로 보장된다. 생성된 보정은 무조건 적용되지 않고, 밀도 양수성·유한성·경계 정합·잔차 감소를 확인하는 허용성 게이트를 통과할 때에만 채택되며, 통과하지 못하면 native 갱신으로 복귀한다. 이 구조 덕분에 가속의 국소적 불완전성이 솔버 전체의 안정성을 훼손하지 않으며, 정확도·경계조건·기준해 비교에 관한 검증을 모두 동일한 잔차·허용성 기준 안에서 일관되게 다룰 수 있다.

성능 측면에서 가장 두드러진 결과는 강건성이다. 저장된 27개 벤치마크에서 제안 기법은 모든 실행이 동일한 수렴 기준(절대 잔차·plateau·허용성의 동시 충족)을 통과한 반면, 다섯 기준 기법 --- native LBE, Anderson 가속, preconditioned LBM, inexact Newton--Krylov, dual-time multigrid --- 은 동일 프로토콜과 넉넉한 반복 예산에도 각각 12--15개 사례에서만 수렴하였다. 이 미수렴은 예산 부족이 아니라 수치적 정체에서 비롯된 것으로, 잔차가 예산 도달 한참 전에 이미 바닥에 멈추는 양상으로 확인된다. 차이는 전역 느린 모드가 수렴을 강하게 지배하는 어려운 문제에서 극대화된다. 대표적으로 고-Reynolds lid-driven cavity에서 다섯 기준 기법은 사용한 기법의 종류와 무관하게 약 \$10\^{-6}\$ 수준의 공통된 잔차 바닥에서 정체한 반면, 제안 기법은 보존 모멘트 부분공간의 보정으로 그 바닥을 통과하여 \$10\^{-8}\$--\$10\^{-14}\$ 수준까지, 더 적은 연산으로 도달하였다. 또한 제안 기법의 수렴 이력은 초기 급강하 --- 장기 준정체 --- 종단 급강하의 특징적 형태를 보이는데, 마지막 단계의 급강하가 직전 잔차의 제곱 수준으로 일어나 보정이 보존 모멘트 부분공간에서 실질적 Newton 스텝으로 작동함을 정성적으로 입증한다. 이는 선형 수렴에 그쳐 정체면을 넘지 못하는 기준 기법들과 본질적으로 구별되는 거동이다.

본 연구의 직접적 범위는 2D D2Q9/BGK 정상 벤치마크에서의 수렴 시간·연산량·잔차·기준해 오차 비교에 한정된다. 제안 기법은 이산화나 경계조건 오차 자체를 제거하지 않으며 이산 해에 더 빠르게 도달할 뿐이다. 벽시계 시간은 CPU 세대·메모리 대역폭·라이브러리 구현에 의존하므로 동일 환경 내 상대 지표로 해석하며, 이를 보완하기 위해 하드웨어에 무관하고 완전히 결정론적인 연산량(LBE-call) 지표를 함께 보고하였다. 또한 단순 전단류처럼 전역 느린 모드가 약하거나 경계 구동 국소 모드가 지배적인 일부 문제에서는 보정 단계의 반복당 비용이 상쇄되지 못해 가속의 한계 이득이 작아지는 예외도 관측되었으며, 이러한 승·패가 모두 동일한 느린-모드 지배도 메커니즘으로 설명된다는 점은 오히려 그 해석의 예측력을 뒷받침한다. 향후 연구로는 준정체 구간을 단축하기 위한 내부 Krylov 차원 및 적응적 내부 허용오차 전략, 3D·MRT/entropic 충돌 모델·고-Reynolds 난류 영역으로의 확장, 그리고 개방경계 유속 보존의 정량적 평가를 남긴다.

# 부록 (Appendix)

## 부록 A. 전체 27개 사례 결과표

**표 A1.** 27개 제안법 benchmark 실행 전체. \"미산출\"은 해당 레벨의 tight 기준해가 결과 집합에 없어 사후 계산하지 않은 항목이며 0이나 성공으로 해석하지 않는다(§5.4 데이터 무결성). 해당 사례의 수렴은 잔차·plateau·허용성 기준으로 독립 충족되었다.

  --------------------------------------------------------------------------------------------------------------
  **Lv**   **Case**             **Wall \[s\]**   **LBE**   **r_final**   **r/r_0**   **Rel. err**   **Ref**
  -------- -------------------- ---------------- --------- ------------- ----------- -------------- ------------
  1x       backward step n64    27.66            122673    2.47e-08      7.55e-08    3.26e-03       tight ref

  1x       cavity re1000 n129   56.84            221413    2.36e-09      7.91e-09    0.0542         Ghia

  1x       cavity re100 n33     0.70             20873     1.93e-13      1.02e-12    0.117          Ghia

  1x       cavity re400 n49     3.06             44379     2.04e-11      9.69e-11    0.106          Ghia

  1x       channel poiseuille   20.30            32666     3.38e-13      4.34e-11    9.37e-03       analytic

  1x       couette n32          1.20             20606     2.18e-12      4.63e-11    2.75e-09       analytic

  1x       cylinder wake n64    4.88             20251     9.88e-15      4.02e-14    7.94e-05       tight ref

  1x       multi cylinder n32   1.25             20377     2.14e-12      5.70e-12    4.15e-05       tight ref

  1x       t junction           18.29            32054     2.63e-13      7.18e-12    1.90e-05       Picard ref

  2x       backward step n64    74.97            119793    6.41e-08      2.76e-07    미산출         ---

  2x       cavity re1000 n129   829.72           1440003   4.87e-14      4.79e-07    0.0326         Ghia

  2x       cavity re100 n33     5.78             41793     8.74e-12      7.56e-11    0.0669         Ghia

  2x       cavity re400 n49     309.99           1257000   4.46e-09      3.34e-08    0.0642         Ghia

  2x       channel poiseuille   185.70           105281    1.90e-13      9.73e-11    2.27e-03       analytic

  2x       couette n32          21.78            101554    3.31e-12      9.92e-11    2.87e-08       analytic

  2x       cylinder wake n64    16.16            23184     1.43e-11      8.14e-11    미산출         ---

  2x       multi cylinder n32   5.77             20471     1.65e-14      6.11e-14    미산출         ---

  2x       t junction           96.70            63535     1.26e-12      1.00e-10    미산출         ---

  3x       backward step n64    756.37           866000    1.29e-10      6.79e-10    미산출         ---

  3x       cavity re1000 n129   1234.14          1440085   2.56e-10      6.64e-07    0.0257         Ghia

  3x       cavity re100 n33     180.17           769000    1.20e-10      1.35e-09    0.0493         Ghia

  3x       cavity re400 n49     74.00            233779    1.57e-08      1.50e-07    0.0501         Ghia

  3x       channel poiseuille   798.11           222772    7.81e-14      8.98e-11    1.00e-03       analytic

  3x       couette n32          133.55           296454    2.63e-12      9.66e-11    5.19e-08       analytic

  3x       cylinder wake n64    36.78            41868     1.15e-11      8.01e-11    미산출         ---

  3x       multi cylinder n32   10.37            21265     1.48e-12      6.57e-12    미산출         ---

  3x       t junction           283.69           89398     5.29e-13      7.79e-11    미산출         ---
  --------------------------------------------------------------------------------------------------------------

## 부록 B. 추가 수렴 이력 및 진단 그림

![](media/image8.png){width="6.299212598425197in" height="5.70262467191601in"}

**그림 A1.** 2x 격자의 9개 사례 수렴 이력(전 방법).

![](media/image9.png){width="6.299212598425197in" height="5.648970909886264in"}

**그림 A2.** 3x 격자의 9개 사례 수렴 이력(전 방법).

![](media/image10.png){width="4.330708661417323in" height="3.131582458442695in"}

**그림 A3.** 대표 사례 7회 반복의 벽시계 시간 변동성(CV\<7%) 대 연산량 결정성(LBE bit-identical).

![](media/image11.png){width="3.543307086614173in" height="2.8346456692913384in"}

**그림 A4.** 레벨별 AP-Schur 보정 채택률(전체 71.0%, 채택 0회 사례 없음).

# Appendix

## A.1 Cost and memory model

Let $N_{f}$ be the number of fluid nodes, $q = 9$ the number of discrete velocities, and $n_{m} = 3$ the number of conserved moments. The cost of one outer round is

$$C_{round} \approx (n_{G} + n_{trial})\, C_{G} + C_{FFT},\quad\quad C_{FFT} = O(n_{m}N_{f}\log N_{f}),$$

where $C_{G}$ is the cost of one native lattice step, $n_{G}$ is the number of native steps per round (burn-in/block Picard work plus the guard), and $n_{trial}$ is the number of residual evaluations spent in the finite-difference Jacobian action and the line search. The memory footprint is

$$W_{mem} \approx qN_{f} + O(n_{m}N_{f}) + O(N_{b}),$$

i.e. the full distribution field, the moment and spectral buffers, and the boundary bookkeeping over $N_{b}$ boundary nodes. The full Newton matrix, of size $qN_{f} \times qN_{f}$, is never formed. Peak resident-set-size measurements at three grid resolutions confirm that the marginal memory $W_{mem}$ grows linearly in $N_{f}$ (empirically of order $35 \times$ the field size) and lies three to four orders of magnitude below a dense Jacobian. Absolute memory usage is environment-dependent, so the claim is restricted to the $O(N_{f})$ scaling and the order-of-magnitude gap.

## A.2 Mass-conservation and boundary-consistency diagnostics

Residual convergence is distinct from exact mass conservation: the macroscopic $L_{2}$ residual measures the change of $p$ and $\mathbf{u}$ over the whole fluid domain, whereas global mass drift and inlet/outlet flux closure are sensitive to the boundary and mask treatment. We therefore retain the residual and its plateau as the primary convergence indicators and report the following two quantities only as auxiliary diagnostics. The global mass and its relative drift are

$$\mathcal{M}^{n} = \sum_{\Omega_{f}}^{}\rho^{n}\, dV,\quad\quad\varepsilon_{\mathcal{M}}^{n} = \frac{|\mathcal{M}^{n} - \mathcal{M}^{0}|}{max(|\mathcal{M}^{0}|,\,\epsilon)},$$

and the inlet--outlet flux closure is

$$\varepsilon_{Q}^{n} = \frac{|\sum_{out}^{}{flux} + \sum_{in}^{}{flux}|}{\max\left( \sum_{in}^{}|flux|,\,\epsilon \right)}.$$

Neither quantity is used as a stopping condition; for closed cavities $\varepsilon_{Q}$ is not applicable. The reported results are obtained with no post hoc mass correction. Positivity and boundary consistency enter only through the admissibility gate.

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
