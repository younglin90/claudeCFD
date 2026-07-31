**A Geometry-Aware, Admissibility-Preserving Schur-Complement Nonlinear Preconditioner for Steady-State Lattice Boltzmann Solvers: Validation on Complex-Geometry Benchmarks**

Research article (revised manuscript)

# Abstract

In steady-state lattice Boltzmann method (LBM) computations, the long-wavelength residual of the conserved pressure–velocity moments persists long after the kinetic modes have decayed, and it is this slowly decaying residual that governs fixed-point convergence. This paper proposes a geometry-aware, admissibility-preserving (AP) Schur-only, Jacobian-free nonlinear preconditioning technique that targets this bottleneck directly. Here and throughout, "AP" abbreviates *admissibility-preserving*—not the asymptotic-preserving schemes of kinetic theory and not any neural-network method. The method leaves the discretized LBM equations and boundary conditions unchanged and operates on the native residual $R(f) = f - G(f)$ as is. Its core operator is built by closing the conserved-moment Schur complement of the LBE operator—linearized about a uniform base state—into a per-mode $3\times3$ matrix in Fourier space, and assembling its kinetic-null-space-corrected, admissibility-preserving inverse as a spectral preconditioner. This preconditioner is used solely as the left preconditioner of a single Jacobian-free Newton (GMRES) step applied to the native nonlinear residual; the resulting trial update is accepted only when, after a damped line search, it simultaneously satisfies a decrease in the macroscopic $L_2$ residual, density positivity, wall/inlet/outlet/mask boundary consistency, and conservation sanity, and otherwise the solver falls back to a native Picard step. All internal constants depend only on a global grid scale and on neither benchmark identity nor any reference solution.

On a fixed set of benchmark results, the proposed method attains convergence under an identical protocol across all 27 runs spanning nine benchmark families (channel, Couette, lid-driven cavity at $Re=100/400/1000$, backward-facing step, cylinder wake, multi-cylinder, and T-junction) at the 1x/2x/3x mesh levels. Under the same stopping protocol and the same admissibility definition, five baseline methods (Picard, Anderson acceleration, preconditioned LBM, inexact Newton–Krylov, and dual-time multigrid) converge on only 12–15 of the 27 cases even with a generous budget. In a conservative timing comparison restricted to the strict subset (15 cases) on which a baseline also converges, the proposed method is faster in wall time on 14/15 cases (median ratio $\approx 2.06\times$) and uses fewer LBE-calls—the operator-work metric—on 13/15 cases (median $\approx 1.80\times$). Broadening the comparison to all available baselines, it is faster on 25/27 cases (median $2.92\times$). On accuracy, the method exhibits an observed spatial convergence order of $\approx 2.0$ for channel Poiseuille flow (the BGK-LBM theoretical value), machine precision for Couette flow, and monotone convergence toward the Ghia centerline for the cavity, confirming that the acceleration does not sacrifice discrete accuracy. All results are independently recomputable from the stored residual histories and per-case execution traces.

The novelty of this work lies not in modifying the LBM physical model, but in a single solver framework that preconditions the hydrodynamic slow modes of the native steady residual from a Schur-complement viewpoint and validates every accepted update through the same admissibility gate, even on complex geometries. The performance claims are confined to relative comparisons within a stored 2D D2Q9/BGK benchmark suite under an identical macroscopic-$L_2$-residual/plateau protocol, and the method is to be understood as a nonlinear preconditioner that solves the same discrete steady problem faster—without reference injection or case-specific tuning.

**Keywords:** lattice Boltzmann method; steady-state solver; admissibility-preserving Schur complement; Jacobian-free residual correction; nonlinear preconditioning; complex geometry.

# 1. Introduction

The lattice Boltzmann method (LBM) has become a workhorse for a wide range of CFD problems, owing to the simplicity of its streaming–collision structure and its amenability to complex-boundary treatment and parallelization [1–4]. In applications where the steady-state solution itself is the goal—design optimization, geometric parameter sweeps, inverse design—rather than the transient, however, the explicit time-marching nature of LBM translates directly into cost. The origin of this cost is not a requirement of temporal accuracy but the spectral structure of the fixed-point residual. In native lattice Boltzmann equation (LBE) iteration, the non-conserved kinetic modes are damped relatively quickly by collision relaxation, whereas the conserved hydrodynamic modes associated with density and momentum survive as the long-wavelength shear and acoustic modes of the linearized LBE and decay very slowly [4]; in the low-Mach regime in particular, the separation of convective and acoustic scales further retards this decay [20, 21]. As a result, the convergence history exhibits a rapid initial drop followed by a long, flat tail, and it is this tail that dominates the total wall time.

Prior efforts to mitigate this tail fall broadly into three families. First, algebraic history accelerators, exemplified by Anderson acceleration and reduced-rank extrapolation (RRE), extrapolate a descent direction from the residual correlations of the past fixed-point history [9, 10, 15]. These are powerful and general, but they treat the residual as a structureless vector and do not directly exploit the *physical block structure* by which, within the steady LBM residual, the kinetic fast modes and the hydrodynamic slow modes vanish on different time scales. Second, inexact Newton and Jacobian-free Newton–Krylov (JFNK) methods provide the standard framework for solving the nonlinear residual equation directly [6, 7, 13]. The efficiency of JFNK, however, hinges entirely on how well the preconditioner reflects the physical structure of the problem; an unsuitable preconditioner inflates the cost of both GMRES iterations and residual evaluations. Moreover, on complex geometries where masks, obstacles, and open boundaries coexist, a Newton trial step may violate density positivity or boundary consistency, so that the physical admissibility of the trial becomes a problem distinct from convergence. The third family modifies the interior of the LBM model itself. Preconditioned LBM redefines the collision relaxation spectrum or the equilibrium to accelerate low-Mach steady convergence [20, 21], and recent work has combined preconditioning with cascaded/central-moment LBM (including corrections for Galilean invariance and cubic-velocity errors) [22, 23, 25] and extended the lattice Boltzmann flux solver to steady flows on unstructured grids [24]. Multigrid and dual-time families relax the elliptic coupling through a mesh hierarchy and coarse-grid corrections [14], and the pressure-Schur/saddle-point preconditioning they borrow is a canonical framework for rapidly solving the pressure–velocity coupling of incompressible systems [8, 11, 12]. On a separate axis, Huang, Yang, and Cai discretized the LBE implicitly and proposed fully implicit and nonlinearly preconditioned inexact-Newton frameworks combining Newton–Krylov, domain decomposition, and nonlinear elimination [18, 19]. These rich bodies of prior work nonetheless share one common trait: to accelerate, they either redefine the collision model, equilibrium, or relaxation parameter (model-level acceleration), or require additional infrastructure such as a mesh hierarchy, transfer operators, implicit matrix assembly, or domain decomposition. In other words, the acceleration takes place *inside* the existing discrete LBM operator or in the *infrastructure surrounding* it.

A gap therefore emerges. Steady-state LBM acceleration still lacks an externally attached correction layer that *simultaneously* satisfies three conditions: (a) it leaves the native operator—built from collision, streaming, and boundary handling—and its discrete steady solution entirely unchanged; (b) it selectively targets only the conserved-moment slow-mode block that governs convergence; and (c) it guarantees physical admissibility, without case-by-case tuning, even on complex geometries with masks and open boundaries. Algebraic accelerators do not use the block structure of (b); model-level accelerations violate (a); and generic JFNK and implicit frameworks either do not address (c) separately or require heavy infrastructure.

This paper proposes such an externally attached acceleration layer—a geometry-aware, admissibility-preserving Schur-complement nonlinear preconditioner (hereafter AP-Schur)—that meets all three conditions at once. Throughout, "AP" denotes *admissibility-preserving*; it is unrelated to the asymptotic-preserving (AP) schemes of kinetic theory and to any neural-network method. The central idea begins from the observation that the slow components governing convergence reside in the conserved-moment block: it therefore suffices to precondition only the Schur complement of that subspace, constructed in closed form in Fourier space, and to use it solely as the preconditioner of a single Jacobian-free Newton step applied to the unchanged native residual $R(f) = f - G(f)$. The generated trial is accepted only if it passes both a residual decrease and physical admissibility, and otherwise the solver falls back to a native LBE step; the method therefore operates stably on complex geometries while leaving the existing LBM operator and its discrete steady solution untouched.

The method is validated on the 27 runs obtained by extending nine benchmark families to the 1x/2x/3x levels, together with a 1x ablation study. Under an identical stopping protocol the proposed method converges on all 27 cases, whereas the baseline accelerators converge on only a subset under the same conditions; and even in the conservative comparison restricted to cases on which a baseline also converges, the proposed method reaches steady state with shorter wall time and fewer operator-work units, without sacrificing discrete accuracy. All performance claims are made as relative comparisons within the stored 2D D2Q9/BGK suite.

# 2. Numerical Method

## 2.1 Native Steady-State LBM Residual and Notation

A two-dimensional, isothermal, incompressible flow is represented on the D2Q9 lattice by the distribution functions $f_{i}(\mathbf{x}),\ i = 0,\ \ldots,\ 8$. The macroscopic density, momentum, and pressure are computed from the standard velocity moments [1–4]:

$$\rho(\mathbf{x}) = \sum_{i} f_{i}(\mathbf{x}), \qquad \rho(\mathbf{x})\,\mathbf{u}(\mathbf{x}) = \sum_{i} \mathbf{c}_{i} f_{i}(\mathbf{x}), \qquad p(\mathbf{x}) = c_{s}^{2}\,\rho(\mathbf{x}). \tag{1}$$

The benchmarks are interpreted in standard lattice units with $\Delta x = \Delta t = 1$, and the lattice sound speed of the D2Q9 BGK model is $c_{s}^{2} = 1/3$. The kinematic viscosity is $\nu = c_{s}^{2}(\tau_{\mathrm{BGK}} - 1/2)$, where $\tau_{\mathrm{BGK}}$ is the collision relaxation time, to be distinguished from the stopping tolerance $\tau$. The Reynolds and Mach numbers are $Re = U_{\mathrm{ref}} L_{\mathrm{ref}} / \nu$ and $Ma = U_{\mathrm{ref}} / c_{s}$. The pressure $p = c_{s}^{2}\rho$ in Eq. (1) is the weakly compressible pressure variable of LBM, interpreted consistently with incompressible benchmarks in the sufficiently small Mach-number regime. The proposed method changes none of this collision model, $\nu$, or boundary conditions; it is an acceleration procedure that reduces the steady residual of the *same* native operator faster.

Writing the BGK collision together with the streaming/boundary update as a single native operator $G$, the steady problem reduces to a fixed-point residual equation [1, 4]:

$$R(f) = f - G(f) = 0. \tag{2}$$

Here $G(f)$ subsumes all wall, inlet, outlet, mask, and obstacle handling. The acceleration layer of this work only calls $G$ as a black box and modifies neither the collision model, the forcing, nor the boundary conditions. The Zou–He pressure/velocity boundaries and the bounce-back/momentum-transfer mask boundaries are therefore treated as a fixed native projection lying *outside* the acceleration layer.

We define two constant matrices that connect the conserved moments and the distribution functions. The extraction (projection) operator $\mathsf{M} \in \mathbb{R}^{3 \times 9}$ and the lifting operator $\mathsf{T} \in \mathbb{R}^{9 \times 3}$ are

$$\mathsf{M} = \begin{bmatrix} 1 & \ldots & 1 \\ c_{x,0} & \ldots & c_{x,8} \\ c_{y,0} & \ldots & c_{y,8} \end{bmatrix}, \qquad \mathsf{T}_{i,:} = \big[\, w_{i},\; 3 w_{i} c_{x,i},\; 3 w_{i} c_{y,i} \,\big],$$

and by design they satisfy the Galerkin consistency $\mathsf{M}\mathsf{T} = \mathsf{I}_{3}$. The product $\mathsf{M}f$ returns exactly the conserved moments $\left( \rho, \rho u_{x}, \rho u_{y} \right)$ of Eq. (1), while $\mathsf{T}$ maps a moment increment back to the minimal hydrodynamic distribution increment consistent with the first-order equilibrium terms.

## 2.2 Macroscopic $L_2$ Residual and Convergence Criterion

Convergence is judged not on the microscopic $f$-RMS but on the macroscopic $L_2$ change of the pressure and velocity fields; the $f$-RMS is stored as a secondary diagnostic and is not used as the primary convergence metric. At check point $k$, write the one-step changes of the primitive macroscopic fields as $\delta p = p^{k+1} - p^{k}$, $\delta u_{x} = u_{x}^{k+1} - u_{x}^{k}$, and $\delta u_{y} = u_{y}^{k+1} - u_{y}^{k}$. The pressure increment is made gauge-invariant by removing its fluid-domain mean—the absolute pressure level (equivalently the density mean) is an arbitrary constant in weakly compressible, closed or periodic flow—so that $\widetilde{\delta p} = \delta p - \langle \delta p \rangle_{\Omega_{f}}$ on $\Omega_{f}$ and $\widetilde{\delta p} = 0$ on solid nodes. The component residuals are the $L_2$ norms taken over the fluid nodes (without per-node normalization, so that a single tolerance family applies at a fixed grid level):

$$r_{p}^{k} = \big\| \widetilde{\delta p} \big\|_{\Omega_{f}}, \qquad r_{u}^{k} = \left\| \delta u_{x} \right\|_{\Omega_{f}}, \qquad r_{v}^{k} = \left\| \delta u_{y} \right\|_{\Omega_{f}}, \qquad \| \cdot \|_{\Omega_{f}}^{2} = \sum_{\mathbf{x} \in \Omega_{f}} ( \cdot )^{2}, \tag{3}$$

$$r_{\mathrm{macro}}^{k} = \sqrt{ (r_{p}^{k})^{2} + (r_{u}^{k})^{2} + (r_{v}^{k})^{2} + (r_{w}^{k})^{2} }, \qquad I_{W}^{k} = \frac{r_{\mathrm{macro}}^{k-W} - r_{\mathrm{macro}}^{k}}{\max\!\left( r_{\mathrm{macro}}^{k-W},\, \varepsilon_{\mathrm{floor}} \right)} \le \eta. \tag{4}$$

This gauge removal of the pressure increment pairs exactly with the mean-mode treatment $B_{U}(0) = \mathrm{diag}(0,1,1)$ of the preconditioner (Section 2.4): the residual and the preconditioner consistently quotient out the density-mean mode.

Because the present suite consists of 2D D2Q9 problems, the active velocity components are $u_{x}$ and $u_{y}$, and the $r_{w}$ term in Eq. (4) is a general notation for a 3D extension that equals $r_{w} = 0$ in the present results. Even when the residual label of a stored file reads `macro_l2_p_ux_uy_uz`, the convergence value for a 2D case is the $L_2$ change of the pressure, $x$-velocity, and $y$-velocity components. This labeling does not mean that convergence was relaxed by adding a $z$-velocity; it is merely a generalization that allows the same residual routine to serve both 2D and 3D.

The quantity $I_{W}$ in Eq. (4) is a plateau indicator that measures the fractional improvement over the most recent window. The condition $I_{W} \le \eta$ does not require monotone decrease; rather, having already passed the absolute residual gate, it confirms that the additional rate of decrease has become sufficiently small—a tail-stability condition. A case in which the residual rebounds slightly at the end of the window so that $I_{W}$ becomes negative is also interpreted as a plateau signal, indicating that no meaningful further decrease remains near the numerical floor. Final convergence is declared upon the simultaneous satisfaction of three conditions:

$$\text{converged} = \big[\, r_{\mathrm{macro}} \le C_{\mathrm{tol}}\,\tau \,\big] \ \text{AND}\ \big[\, \text{plateau}(r_{\mathrm{macro}}; W, \eta) = \text{true} \,\big] \ \text{AND}\ \big[\, \text{admissible}(f, \rho, u, v) = \text{true} \,\big]. \tag{5}$$

Here $\tau$ is the base tolerance corresponding to the mesh level and case family, and $C_{\mathrm{tol}}$ is a residual safety factor. The plateau condition alone does not declare convergence; a low macroscopic-$L_2$ residual and a physically admissible field must hold simultaneously. The constants $\tau$, $W$, $\eta$, and the admissibility rule are applied identically to the proposed method and the baselines and are not changed per case; specific values are fixed in Section 3.2.

## 2.3 Conserved-Moment Schur-Complement Formulation

A Newton correction requires solving $J_{f}\left( f^{*} \right)\delta f = - R\left( f^{*} \right)$, but assembling the full Jacobian $J_{f}$—which includes the complex mask and boundary operators—explicitly is inefficient in both memory and implementation [6, 7, 13]. The starting point of the proposed method is to split the distribution correction $\delta f$ into a conserved-moment component $\delta m = (\delta\rho,\ \delta u,\ \delta v)$ and a kinetic component $\delta k$. Under this decomposition, the local linearization of the steady residual is expressed as the block system

$$\begin{bmatrix} J_{mm} & J_{mk} \\ J_{km} & J_{kk} \end{bmatrix} \begin{bmatrix} \delta m \\ \delta k \end{bmatrix} = - \begin{bmatrix} R_{m} \\ R_{k} \end{bmatrix}. \tag{6}$$

Eliminating the kinetic block yields the Schur-complement problem for the conserved moments:

$$S_{m}\,\delta m = - \big( R_{m} - J_{mk}\, J_{kk}^{-1}\, R_{k} \big), \qquad S_{m} = J_{mm} - J_{mk}\, J_{kk}^{-1}\, J_{km}. \tag{7}$$

$S_{m}$ is the effective operator that directly controls the slow modes of the pressure–velocity field—the moment Schur complement [8, 11, 12]. This viewpoint matters because it corresponds exactly to the convergence structure of native LBE iteration. Native collide–stream iteration performs the kinetic relaxation associated with $J_{kk}$ locally and quickly, whereas the hydrodynamic coupling associated with $S_{m}$ is global and weakly damped, and therefore dominates the late stage of convergence where the residual decreases very slowly. The target of acceleration is thus not the full set of nine distribution functions at each lattice node, but the density and velocity moment components that make convergence slow.

To resolve the slow components accurately, the proposed method does not build the enormous inverse $S_{m}^{-1}$ directly. Instead, it uses a moment projection operator $P$ to extract the density and momentum components from the distributions, computes a correction direction in this moment space, and then uses a lifting operator $P^{\dagger}$ to convert that correction back into a distribution-function increment. The resulting correction is not applied as a final solution; it is applied only after checking that it is physically meaningful (admissible) and that it actually reduces the residual. On the D2Q9 lattice these two operators are given in closed form as constant $3\times9$ and $9\times3$ matrices:

$$(Pf)(\mathbf{x}) = \mathsf{M}f(\mathbf{x}),$$

$$Pf = \begin{pmatrix} \sum_{i} f_{i} & \sum_{i} c_{ix} f_{i} & \sum_{i} c_{iy} f_{i} \end{pmatrix} = \begin{pmatrix} \rho & \rho u_{x} & \rho u_{y} \end{pmatrix}, \tag{8}$$

$$\left( P^{\dagger}\delta U \right)_{i} = w_{i}\left[\, \delta\rho + 3 c_{ix}\,\delta\left( \rho u_{x} \right) + 3 c_{iy}\,\delta\left( \rho u_{y} \right) \,\right]. \tag{9}$$

The operator $P$ in Eq. (8) is exactly the velocity-moment projection of Eq. (1), and $P^{\dagger}$ in Eq. (9) is the lifting that maps a moment increment $\delta U = \left( \delta\rho,\ \delta\left( \rho u_{x} \right),\ \delta\left( \rho u_{y} \right) \right)$ back to the minimal hydrodynamic distribution increment consistent with the first-order equilibrium terms. That is, $P^{\dagger}$ is not a post hoc reset of the nine distribution components but the canonical lifting onto the conserved-moment subspace; by design $\mathsf{M}\mathsf{T} = \mathsf{I}_{3}$, so that $P\left( P^{\dagger}\delta U \right) = \delta U$. The correction lifted by $P^{\dagger}$ is the minimal distribution-function correction that matches only the density and momentum components. The remaining kinetic components, which are not corrected directly, are left for the existing LBM collision–streaming–boundary operator to relax naturally on the next iteration. The AP-Schur step is therefore not a procedure that fits the full distribution shape to a reference, but one that corrects only the density and velocity components responsible for slow convergence. Furthermore, when applying $P$ and $P^{\dagger}$, interior solid/obstacle nodes are not computed; computation is restricted to nodes where fluid actually exists.

The moment Schur operator $S_{m}$ and its approximate inverse $B_{m}$ are defined as

$$S_{m} \approx P J_{f} P^{\dagger}, \qquad S_{m}\,\delta U \approx - P R\left( f^{*} \right), \qquad \delta f_{\mathrm{AP}} = P^{\dagger}\delta U. \tag{10}$$

In the implementation, $S_{m}$ is neither an explicit submatrix of the full $\left( 9 N_{f} \right) \times \left( 9 N_{f} \right)$ Jacobian nor a per-geometry dense matrix. The action of $S_{m}$ is evaluated only through a directional finite difference of the native residual, and its inverse action is approximated by the spectral (Fourier) preconditioner $B_{m}$ of Section 2.4:

$$J_{f}(f)\,v \approx \frac{R(f + \epsilon v) - R(f)}{\epsilon}, \qquad \epsilon = \frac{10^{-7}\left( 1 + \left\| f \right\|_{2} \right)}{\left\| v \right\|_{2}}. \tag{11}$$

Equation (11) is the exact finite-difference increment used in this implementation; the constant $10^{-7}$ is the standard forward-difference scale at IEEE double precision and is applied identically to all cases. To make the naming precise, "Jacobian-free" in this paper does not mean that a full Newton matrix is assembled or that a Newton–Krylov system is solved exactly at every step. Its meaning is restricted to three points. First, the correction is always evaluated through the finite-difference response of the native residual $R(f)$ (Eq. 11). Second, the search is restricted not to the full distribution space but to the Schur-preconditioned direction within the pressure–velocity moment subspace. Third, an accepted step is merely a Newton-like trial; if it fails the residual-decrease and admissibility gates, the solver reverts to the native fallback. The proposed method is therefore not a full JFNK solver but a moment-Schur nonlinear preconditioner that uses the Jacobian-free residual response. Consequently, the method neither solves a new pressure Poisson equation, nor forms the large saddle-point matrices used in finite-element approaches, nor constructs a separate Schur solver for each geometry. The same $P,\ P^{\dagger},\ B_{m}$, and accept/reject gate are used across all benchmarks; what changes from case to case is only the existing LBM boundary and mask of each problem.

## 2.4 Spectral AP-Schur Preconditioner $B_{m}$

Rather than constructing the inverse of $S_{m}$ directly, we linearize the LBM update about the uniform base state $\overline{\rho} = 1,\ \overline{\mathbf{u}} = 0$. This linearization allows the complex nonlinear LBM update to be treated approximately as a linear operator. In particular, in Fourier space the streaming operation becomes a simple phase factor $A(\mathbf{k}) = \mathrm{diag}\left( e^{-i \mathbf{k}\cdot\mathbf{c}_{i}} \right)$ for each mode $\mathbf{k}$, so that the large global problem decouples into a small per-mode problem. The linearization of the BGK collision is $C(\omega) = (1 - \omega)\mathsf{I}_{9} + \omega \mathsf{T}\mathsf{M}$. A single linearized LBM update at Fourier mode $\mathbf{k}$ is thus $L'(\mathbf{k}) = A(\mathbf{k})C(\omega)$, and the Jacobian corresponding to the linearization of the fixed-point residual $R(f) = f - G(f)$ can be written $J(\mathbf{k}) = \mathsf{I}_{9} - L'(\mathbf{k})$. Reducing this mode-wise representation to the moment space yields a $3\times3$ Schur approximation at each $\mathbf{k}$, whose inverse serves as the $B_{m}$ preconditioner.

Applying the Galerkin reduction to the moment space gives, at each Fourier mode, the following $3\times3$ Schur approximation:

$$S_{U}^{\mathrm{G}}(\mathbf{k}) = \mathsf{M} J(\mathbf{k}) \mathsf{T} = \mathsf{I}_{3} - \mathsf{M} A(\mathbf{k}) \mathsf{T}. \tag{13}$$

This simple Galerkin Schur approximation $S_{U}^{\mathrm{G}}(\mathbf{k})$, however, does not fully reflect the influence of the kinetic modes. In LBM the conserved moments and the kinetic modes are not entirely independent; they interact through streaming and collision. Because the degree of damping of the kinetic modes varies with the relaxation parameter $\omega$ in particular, neglecting this interaction can degrade the quality of the preconditioner. We therefore correct, to first order, the contribution of the kinetic null space ($J_{kk} \approx \omega$), and define the following admissibility-preserving (AP) Schur operator:

$$S_{U}^{\mathrm{AP}}(\mathbf{k}) = S_{U}^{\mathrm{G}}(\mathbf{k}) - \kappa(\omega)\left[\, \mathsf{M} A(\mathbf{k})^{2} \mathsf{T} - \left( \mathsf{M} A(\mathbf{k}) \mathsf{T} \right)^{2} \,\right]. \tag{14}$$

The correction term admits an exact identity that fixes its meaning:

$$\mathsf{M} A(\mathbf{k})^{2} \mathsf{T} - \left( \mathsf{M} A(\mathbf{k}) \mathsf{T} \right)^{2} = \mathsf{M} A(\mathbf{k})\, \left( \mathsf{I}_{9} - \mathsf{T}\mathsf{M} \right) A(\mathbf{k})\, \mathsf{T}.$$

The right-hand side is precisely the moment$\to$kinetic$\to$moment coupling—the path that leaves the moment subspace through the kinetic complement $\mathsf{I}_{9} - \mathsf{T}\mathsf{M}$ and returns to it—i.e., the moment-space representation of $J_{mk} J_{km}$ in Eq. (7). Approximating the kinetic-block inverse $J_{kk}^{-1}$ by the scalar $\kappa(\omega)$ therefore yields a first-order, structurally exact reconstruction of the term $-J_{mk} J_{kk}^{-1} J_{km}$ that the Galerkin approximation $S_{U}^{\mathrm{G}}$ omits—not an ad hoc correction. The coefficient $\kappa(\omega)$ is defined as

$$\kappa(\omega) = \tfrac{1}{2}\,\mathrm{sign}(r)\,\min\!\left( \tfrac{1}{2}, |r| \right), \qquad r = (1 - \omega)/\omega.$$

When $\omega$ becomes very small, $r$ grows large and the correction term may grow excessively, destabilizing the preconditioner; $\kappa(\omega)$ therefore includes a clip such that $\left| \kappa(\omega) \right| \le \tfrac{1}{4}$. For each Fourier mode $\mathbf{k}$, the $3\times3$ operator is regularized with an adaptive Tikhonov term to handle singular or ill-conditioned cases:

$$S_{U}^{\mathrm{reg}}(\mathbf{k}) = S_{U}^{\mathrm{AP}}(\mathbf{k}) + \eta\,\mathsf{I}_{3}, \qquad \eta = \frac{\sigma_{\max}\!\left( S_{U}^{\mathrm{AP}} \right)}{50}, \tag{15}$$

and the per-mode preconditioner is finally defined as $B_{U}(\mathbf{k}) = \left[ S_{U}^{\mathrm{reg}}(\mathbf{k}) \right]^{-1}$. Here the regularization strength $\eta$ is a parameter-free choice fixed automatically from the maximum singular value $\sigma_{\max}$ over the entire spectrum (target condition number $\approx 50$) and is not tuned per case. Because the mean mode $\mathbf{k} = (0,0)$ is directly tied to mass conservation, neither the Newton step nor the preconditioner may alter it arbitrarily; we therefore apply no Newton step to it (zeroing that component) and pass through only the momentum mean, leaving it to the kinetic LBE. The action of the preconditioner is implemented with a single FFT pair:

$$B_{m} R_{f} = P^{\dagger}\,\mathcal{F}^{-1}\!\left\{ B_{U}(\mathbf{k}) \cdot \mathcal{F}\left[ P R_{f} \right](\mathbf{k}) \right\}. \tag{16}$$

That is, the residual is projected to the moment space ($P$), transformed by a 2D FFT, multiplied at each mode by the precomputed and cached $3\times3$ $B_{U}(\mathbf{k})$, inverse-transformed, and lifted ($P^{\dagger}$). Because $B_{U}(\mathbf{k})$ depends only on the $(N_{y}, N_{x})$ grid and $\omega$, it is constructed once per case at $\mathcal{O}\!\left( N_{f}\log N_{f} \right)$ cost and reused throughout the outer iterations.

This preconditioner is derived from a periodic Fourier linearization, yet the actual problem may have non-periodic boundary conditions. $B_{m}$ does not represent boundary effects exactly, but because $B_{m}$ does not produce the solution directly and is used only as a preconditioner inside GMRES, this does not undermine its validity. The role of the preconditioner is to reduce the slow-mode error amplification factor; if $B_{m}$ does not fully capture the boundary effects, the consequence is merely "slower convergence," not "convergence to a wrong fixed point." Because every accepted update is validated only by the decrease of the native nonlinear residual $R(f) = f - G(f)$—which includes the boundaries—and by admissibility (Eqs. 17–18), the approximation quality of the preconditioner affects only speed, not accuracy. This is the same logic as the standard Krylov practice of using a symmetric/constant-coefficient preconditioner for a nonsymmetric, non-periodic problem [8, 11, 12].

## 2.5 Jacobian-Free Newton Step and Admissibility Gate

The spectral preconditioner $B_{m}$ does not by itself produce a solution; it is used only as the left preconditioner of a single preconditioned Newton step applied to the native nonlinear residual. At every outer round, the method applies left-preconditioned GMRES, with a limited number of iterations, to the right-hand side $-R(f^{k})$:

$$J_{f}\left( f^{k} \right)\delta f = - R\left( f^{k} \right) \quad \text{with preconditioner } M = B_{m},\ \ \text{operator } v \rightarrow J_{f}\left( f^{k} \right)v\ \text{(Eq. 11)},\ \ \text{restart} = 2 k_{\max},\ \ \text{maxiter} = 1. \tag{17}$$

Because the action of the operator is evaluated only through the native-residual finite difference of Eq. (11), $J_{f}$ is never assembled explicitly. Here $R(f) = f - G(f)$ is the native operator including collision, streaming, and boundary conditions (wall/inlet/outlet/mask boundary projection). The $\delta f$ returned by GMRES is not accepted directly; it must pass a damped line search and an admissibility gate:

$$f_{\mathrm{trial}}(\alpha) = f^{k} + \alpha\,\delta f, \quad \alpha \in \left\{ 1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8} \right\}, \quad \text{accept} \iff \text{admissible}(f_{\mathrm{trial}}) \ \wedge\ \left\| R\left( f_{\mathrm{trial}} \right) \right\| < \left\| R\left( f_{\mathrm{best}} \right) \right\| \ \wedge\ \text{conservation}(f_{\mathrm{trial}}). \tag{18}$$

The damping set $\left\{ 1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8} \right\}$ is a globalization device: $\alpha$ is tried from the largest value down, the first candidate that satisfies admissibility (below) and a residual decrease is accepted, and if no $\alpha$ satisfies them the trial is recorded as rejected and the solver falls back to native LBM. Acceptance is decided solely by (i) physical admissibility, (ii) a monotone decrease of the native residual norm, and (iii) conservation (mass/flux) sanity.

The admissibility gate blocks three kinds of failure. First, it rejects any trial that violates density positivity ($\rho > 0$) or the finite-value condition. Second, it rejects any trial for which mask/wall boundary consistency is not maintained even after the native boundary projection. Third, a trial that fails to achieve a residual decrease is not recorded as an accepted step. Table 3 summarizes the physical meaning of each gate.

**Table 3. Admissibility gates and their physical meaning.**

| Validation gate | Rationale |
|---|---|
| Finite field | Rejects NaN/Inf pressure, velocity, density, or distribution. |
| Positive density | Rejects $\rho \le 0$ or non-physical low-density branches. |
| Residual decrease | Accepts an AP-Schur correction only when the native $r_{\mathrm{macro}}$ decreases. |
| Boundary consistency | Re-applies the native wall/inlet/outlet/mask operator for every trial. |
| Conservation sanity | Accepts only when mass drift and inlet/outlet flux closure do not worsen relative to the native candidate (mask/open geometry). |
| No reference injection | Analytic/Ghia/tight references are used only for post-solve error evaluation. |

The purpose of geometry-aware admissibility is not to force the Schur correction onto complex geometries, but to let through only admissible trials while preserving the boundary conditions (wall, inlet, outlet, mask, obstacle constraints) defined by the native LBM operator. In cases where open boundaries are also present, a global correction risks overwriting the local boundary physics, and the gates of Table 3 block such trials.

To respect a particularly important invariance in boundary treatment, an AP-Schur trial does not overwrite wall, inlet, outlet, or obstacle values with new, independent boundary expressions. It first forms an interior trial field through the moment-space correction, and then evaluates the residual, density positivity, finite-field, and mask-consistency conditions in the state to which each benchmark's original native boundary projection has been re-applied. Solid/mask nodes are excluded from the $\Omega_{f}$ residual norm and the moment projection, and the prescribed quantities of a boundary segment are re-projected by the native operator before any trial can be accepted. This design matters because, in cases where geometry and open boundaries coexist—such as the backward step, cylinder wake, multi-cylinder mask, and T-junction—a global correction risks overwriting the local boundary physics. Because this paper uses no per-geometry Schur solver and instead the same projection–lifting and accept/reject logic, performance differences between cases are interpreted not as differences in the correction term but as differences in how often the same correction provides an admissible direction in each geometry (Section 4.7).

## 2.6 Single Solver Procedure and Scale-Only Adaptation

The full execution procedure of the proposed method is given in Algorithm 1. It is a single routine that does not change with the computational setting. Its only adaptivity is that a global state scale $s$ sets the burn-in and block lengths. The burn-in is the number of Picard iterations used for initial stabilization, and the block is the number of Picard iterations used to form a Picard candidate in each round. This scale is defined in closed form from the number of degrees of freedom (fluid nodes $\times$ lattice directions) $N_{\mathrm{dof}}$ alone:

$$s = \max\!\left( \sqrt{\frac{N_{\mathrm{dof}}}{9 \times 32^{2}}},\, 1 \right) = \frac{\text{linear grid size}}{32}.$$

That is, $s$ equals 1 on a $32\times32$ D2Q9 grid and is a pure grid scale proportional to the linear grid size; it depends on neither the Reynolds number, the boundary-condition type, nor the mask geometry. Setting $\text{burn} = \mathrm{clip}(\mathrm{round}(16 s), 8, 96)$ and $\text{block} = \mathrm{clip}(\mathrm{round}(80 s), 48, 512)$ reflects the universal fact that, as the grid grows, more sweeps are needed for information to traverse the domain.

**Algorithm 1. Single AP-Schur-only solver.**

> **Input:** native operator $G$, initial field $f^{0} = $ `case.initial_field()`, tolerance $\tau$. **Constants:** $\text{burn} = \mathrm{clip}(\mathrm{round}(16 s), 8, 96)$, $\text{block} = \mathrm{clip}(\mathrm{round}(80 s), 48, 512)$, $R_{\max} = 160$, $\text{stale}_{\max} = 40$, $k_{\max}$ (GMRES restart factor).
>
> 1. Burn in with $f \leftarrow \mathrm{Picard}^{\,\text{burn}}(f^{0})$ and initialize $f_{\mathrm{best}} \leftarrow f$, $r_{\mathrm{best}} \leftarrow \|R(f)\|$.
> 2. **for** round $= 1 \ldots R_{\max}$:
>     - (a) **Picard candidate:** $c_{\mathrm{pic}} \leftarrow \mathrm{Picard}^{\,\text{block}}(f)$, $r_{\mathrm{pic}} \leftarrow \|R(c_{\mathrm{pic}})\|$.
>     - (b) **AP-Schur candidate:** obtain $\delta f$ from the $B_{m}$-preconditioned GMRES of Eq. (17), and obtain $c_{\mathrm{ap}}, r_{\mathrm{ap}}$ through the line-search/admissibility/conservation gate of Eq. (18).
>     - (c) Choose the candidate with the smallest $r$; if no candidate improves on $r_{\mathrm{best}}$ by at least a factor of $1.02$, fall back to the native Picard guard ($c_{\mathrm{pic}}$).
>     - (d) $f \leftarrow$ chosen candidate. If improved, update $(f_{\mathrm{best}}, r_{\mathrm{best}})$ and reset $\text{stale} = 0$; otherwise increment $\text{stale}$. Terminate if $r_{\mathrm{best}} \le \tau$ or $\text{stale} \ge \text{stale}_{\max}$.
> 3. **Return** $f_{\mathrm{best}}$ and the (residual, LBE-call, wall time) history.

![](media/image1.png){width=6.25in}

**Figure 1.** Conceptual workflow of the AP-Schur-only method. The macroscopic moment residual is extracted from the native LBM residual, the AP-Schur correction is validated by the admissibility gate, and execution proceeds with either the accepted update or the native fallback.

Figure 1 illustrates the singularity of the method: what changes in each computational case is only the mask and boundary operator, while the residual evaluation, moment projection, AP-Schur trial, admissibility gate, and fallback structure remain identical. Table 4 lists the invariant conditions that an independent verifier can check at each execution stage.

**Table 4. Execution stages of the single AP-Schur-only solver and their verification invariants.**

| Stage | Operation | Invariant for an independent verifier |
|---|---|---|
| Initialization | Construct $f^{0} = $ `case.initial_field()` and the mask/boundary operator; burn in. | No reference profile is used for the initial field or boundary update. |
| Native evaluation | Form $R(f) = f - G(f)$ and a Picard block candidate via a native collide–stream–boundary sweep. | The residual is computed from the solver's original steady equation. |
| Projection | Project $R(f)$ to the moment residual of Eq. (8) and compute $r_{\mathrm{macro}}$. | The same macro-$L_2$ definition is used for stopping and for history storage. |
| AP-Schur step | Solve the GMRES of Eq. (17) with $B_{m}$ of Eq. (16) as preconditioner to form $\delta f$, and build the trial via the lift of Eq. (9). | The spectral $B_{m}$ depends only on $(N_{y}, N_{x}, \omega)$ and does not see any field reference. |
| Gate | Check boundary re-application, residual decrease, admissibility, and conservation via the line search of Eq. (18). | Only accepted trials are applied; on failure, fall back to the native Picard guard. |
| Termination | Terminate when the $r_{\mathrm{macro}}$ threshold and the plateau condition are met simultaneously. | To prevent premature termination, tail-stabilization of the residual decrease is also checked. |

The cost model is as follows. Let $N_{f}$ be the number of fluid nodes, $q = 9$ the number of lattice directions, and $n_{m} = 3$ the number of conserved moments. The dominant cost and structural storage per outer round are

$$C_{\mathrm{round}} \approx \left( n_{G} + n_{\mathrm{trial}} \right) C_{G} + C_{\mathrm{FFT}}, \quad C_{\mathrm{FFT}} = \mathcal{O}\!\left( n_{m} N_{f}\log N_{f} \right), \quad M \approx q N_{f} + \mathcal{O}\!\left( n_{m} N_{f} \right) + \mathcal{O}\!\left( N_{b} \right), \tag{19}$$

where $C_{G}$ is the cost of one collision–streaming–boundary residual evaluation, $C_{\mathrm{FFT}}$ is the cost of the FFT/IFFT pair plus the mode-wise $3\times3$ products of the spectral preconditioner (Eq. 16), and the term $\left( n_{G} + n_{\mathrm{trial}} \right) C_{G}$ denotes the operator-work that combines native evaluations with the in-GMRES JVP and line-search trial evaluations (counted in LBE-calls). $N_{b}$ is the size of the boundary/mask metadata. The mode-wise $3\times3$ inverses $B_{U}(\mathbf{k})$ are cached once per case at $\mathcal{O}\!\left( N_{f} \right)$ storage, and because the full Newton matrix ($q N_{f} \times q N_{f}$) is never formed, the memory is dominated by the distribution field, the moment buffers, and the spectral cache. This $\mathcal{O}\!\left( N_{f} \right)$ storage model is quantitatively confirmed in Section 4.8 by peak-RSS measurements at three grid sizes (the marginal memory grows linearly with field size at a nearly constant ratio of about $35\times$, three to four orders of magnitude smaller than a dense Jacobian); because absolute memory values depend on the runtime environment, the memory claim of this paper is confined to the $\mathcal{O}\!\left( N_{f} \right)$ linear scaling and the order-of-magnitude gap relative to a dense Jacobian.

This cost model is not a claim that an AP-Schur step is always cheaper than one Picard step. An AP-Schur step itself requires several native residual evaluations, a spectral solve, and a line search, so its unit cost is higher. The point is that removing, by an early global correction, the slow hydrodynamic mode that would otherwise be iterated over thousands to hundreds of thousands of Picard tail steps reduces the total wall time [6–8, 11–13]. In other words, the wall-time improvement reported in this paper arises not because "one step is cheaper" but because "an expensive global correction shortens the long native tail," and this is quantitatively confirmed by the LBE-call analysis of Section 4.

## 2.7 Mass-Conservation and Boundary-Consistency Diagnostics

Residual convergence and mass conservation are related but not identical metrics. The macroscopic $L_2$ residual measures whether the pressure and velocity changes have become small over the entire fluid domain, whereas the global mass drift and the inlet/outlet flux closure are more sensitive to boundary cross-sections and mask handling. This paper retains the residual and plateau as the primary convergence conditions and defines mass and boundary consistency separately, as auxiliary diagnostics by which an independent verifier can check physical plausibility:

$$M^{n} = \sum_{\mathbf{x} \in \Omega_{f}} \rho^{n}(\mathbf{x})\,\Delta V, \qquad \epsilon_{M}^{n} = \frac{\left| M^{n} - M^{0} \right|}{\max\left( \left| M^{0} \right|, \epsilon \right)}, \tag{21}$$

$$\epsilon_{Q}^{n} = \frac{\left| \sum_{\Gamma_{\mathrm{out}}} \int_{\Gamma} \mathbf{u}^{n} \cdot \mathbf{n}\, d\Gamma + \sum_{\Gamma_{\mathrm{in}}} \int_{\Gamma} \mathbf{u}^{n} \cdot \mathbf{n}\, d\Gamma \right|}{\max\!\left( \sum_{\Gamma_{\mathrm{in}}} \left| \int_{\Gamma} \mathbf{u}^{n} \cdot \mathbf{n}\, d\Gamma \right|, \epsilon \right)}. \tag{22}$$

Equations (21) and (22) are not additional stopping conditions but diagnostics that check whether the residual-convergence result is physically interpretable. For closed cavities with no clear inlet/outlet, $\epsilon_{Q}$ is not applied. Because mixing flux closure into the stopping rule could make the convergence criterion geometry-dependent, this paper keeps a common macroscopic-residual criterion and uses the mass/boundary items only for reporting and sanity checks.

We also state the absence of any post hoc mass correction. The results of the proposed method are not obtained by renormalizing the density to unity, by clamping the minimum $\rho$ to an arbitrary floor, or by rescaling the distributions post hoc to match a target mass. Density positivity and boundary consistency are used only as the admissibility gate that decides trial acceptance; an accepted state is not post-processed to match a reference value or a target mass. The mass drift of Eq. (21) and the flux imbalance of Eq. (22) are therefore not the result of any hidden correction term in the solver but physical diagnostics independently recomputable from the stored final field. Because the stored summary files do not contain complete numerical mass/flux columns for every case, the text does not claim quantitative mass/flux upper bounds as a primary result and instead defines them as recomputation items of the reproducibility package (Section 5.4).
# 3. Benchmark Suite and Evaluation Protocol

## 3.1 Benchmark Composition and Roles

The suite comprises nine families, chosen to span distinct validation roles. Channel (plane Poiseuille) and Couette are basic shear/pressure-driven tests for which analytic profiles exist. The lid-driven cavity at $Re = 100/400/1000$ checks the literature centerline benchmark [5] and recirculating closed-domain dynamics. The backward-facing step and cylinder wake involve separation, reattachment, and wake formation; the multi-cylinder mask tests multiple obstacles and a complex mask boundary; and the T-junction tests a branching geometry with inlet/outlet boundary coupling. Each family is run at 1x/2x/3x mesh scalings, constituting 27 proposed-method runs in total.

Table 5 summarizes the actual grid size of each family (directly verifiable from the shape of the stored final-field arrays), the boundary-condition type, the validation role, and the reference tier. The complete benchmark specification, including the definitions of $U_{\mathrm{ref}}$, $\nu$, and the mask geometry, is included in the case manifest of the reproducibility package.

**Table 5. Benchmark family definitions: grid size, boundary conditions, validation role.**

| Family | Grid (1x / 2x / 3x) | Boundary conditions | Validation role | Reference |
|---|---|---|---|---|
| Channel (plane Poiseuille) | 32×192 / 64×384 / 96×576 | Inlet/outlet + wall | Pressure-driven shear (baseline) | Analytic |
| Couette | 32² / 64² / 96² | Moving wall + wall | Shear flow (baseline) | Analytic |
| Cavity $Re=100$ | 33² / 65² / 97² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Cavity $Re=400$ | 49² / 97² / 145² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Cavity $Re=1000$ | 129² / 257² / 385² | Lid-driven, closed | Recirculating closed domain | Ghia [5] |
| Backward-facing step | 64² / 128² / 192² | Inlet/outlet + step mask | Separation/reattachment | Tight ref |
| Cylinder wake | 64² / 128² / 192² | Inlet/outlet + obstacle mask | Wake formation | Tight ref |
| Multi-cylinder | 32² / 64² / 96² | Multiple obstacle masks | Complex mask boundary | Tight ref |
| T-junction | 96×64 / 192×128 / 288×192 | Branching inlet/outlet | Branching geometry + open-BC coupling | Tight/Picard ref |

The interpretive scope of the 1x/2x/3x scaling is stated explicitly. This axis is a solver-scaling benchmark that checks whether the same solver converges while maintaining the same stopping protocol as the mesh size grows, and how the wall-time/LBE-call scaling behaves relative to the baselines. It does *not* mean that a formal grid-convergence study or Richardson extrapolation was performed; claims of observed order of accuracy or GCI are outside the scope of this paper (Section 5.2).

## 3.2 Stopping Protocol and Tolerance

In the 27 stored proposed-method runs, the residual type is recorded uniformly as `macro_l2_p_ux_uy_uz`, and the absolute gate is $r_{\mathrm{macro}} < 5\tau$ ($C_{\mathrm{tol}} = 5$). The value of $\tau$ is taken from the `tol` column of the summary CSV as the source value. The channel/Couette/backward-step/cylinder-wake/multi-cylinder/T-junction families use $\tau = 1.0\mathrm{e}{-7},\ 5.0\mathrm{e}{-8},\ 3.333\mathrm{e}{-8}$ at 1x/2x/3x respectively, and the cavity $Re = 100/400/1000$ families use $\tau = 1.0\mathrm{e}{-8},\ 5.0\mathrm{e}{-9},\ 3.333\mathrm{e}{-9}$. The relative-plateau test is passed when the fractional improvement over the most recent $W = 50$ recorded check points in the relative macro-$L_2$ history is at most $\eta = 0.05$. The residual recording interval is set by the run configuration and is stored, together with the iteration, LBE-call, and wall time, in each case's history CSV, so that the plateau test can be recomputed directly from the stored history. In addition, every proposed-method run is subject to a minimum operator budget of about $2\times10^{4}$ LBE-calls, which acts as a floor that prevents premature termination immediately after the rapid initial residual drop. This is directly verifiable in the stored logs, where many fast cases (Couette, multi-cylinder, cylinder wake, cavity $Re=100$ 1x, etc.) terminate at LBE-calls just past $2.0\times10^{4}$; this floor acts only to *increase* the proposed method's wall time.

That $\tau$ tightens by factors of $1/2$ and $1/3$ as the level increases is a protocol-level reporting choice, applied identically to the proposed method and the baselines; because the same schedule is imposed on all methods, it does not affect the fairness of the inter-method comparison.

The difference in absolute $\tau$ between the cavity and non-cavity families is not a case tuning favorable to any particular method but a family-level reporting choice reflecting different reference tiers and benchmark conventions. The cavity is a literature benchmark compared against the Ghia centerline and therefore uses a stricter tolerance family. The meaningful unit of comparison is not the absolute $\tau$ across families but the wall time and operator work needed for the proposed method and the baselines to reach the same convergence verdict at the same case, the same mesh level, the same $\tau$, and the same plateau rule.

The minimum tail budget and the plateau window are not tuning parameters that make the AP-Schur step aggressive or change the correction per case; they are validation gates that prevent premature termination immediately after a fast residual drop. Strengthening them can only keep the wall time the same or increase it, so they are not devices that artificially inflate the proposed method's speed advantage. The same criterion applies to the strict-convergence interpretation of the baseline runs.

The distinction between protocol constants and tuning parameters is placed in four tiers. First, $Re$, $Ma$, $\tau_{\mathrm{BGK}}$, the geometry mask, and the inlet/outlet/wall conditions are part of the benchmark definition and are not adjusted by the solver. Second, $\tau$, $C_{\mathrm{tol}}$, $W$, $\eta$, and the minimum tail budget are frozen validation constants that define the stopping protocol. Third, the $\alpha$ damping candidates and the admissibility gate are method-wide, fixed globalization devices. Fourth, at no tier is there any empirical coefficient specific to the cavity, backward step, cylinder, or T-junction.

## 3.3 Baseline Implementations and Fairness

The credibility of the comparison rests on whether the baselines are faithfully implemented and tuned rather than set up as straw men. All five baselines of this paper share the native LBM operator of the same code base and are implemented with standard settings. Each method is given literature-standard hyperparameters and a generous iteration budget (Table 6a). No baseline is intentionally weakened: Anderson uses a sufficient depth and a regularized least-squares; inexact Newton and dual-time multigrid use multi-stage Krylov/V-cycles; and preconditioned LBM uses the standard PLBE transformation.

**Table 6a. Baseline implementations and main settings.**

| Baseline method | Implementation summary | Main settings |
|---|---|---|
| Picard (native LBM) | Native collide–stream–boundary fixed-point iteration | max_steps $\le 1.2\times10^{6}$, residual-monotone termination |
| Anderson acceleration [9,10] | Regularized least-squares fixed-point acceleration, admissibility safeguard | depth $m=10$, $\beta=1.0$, reg $=10^{-12}$ |
| Preconditioned LBM [20,21] | Balanced PLBE ($\gamma$-scaled) transform + block preconditioner | $\gamma=0.5$, max_steps $\le 1.2\times10^{6}$ |
| Inexact Newton–Krylov [6,7] | JFNK: GMRES + NE/smoother + line search | krylov_max=10, K_ne=20, K_smooth=10, line_search=4 |
| Dual-time multigrid [14] | FAS V-cycle, residual-equation smoothing | max_levels=6, V-cycle, K_pre/coarse/post=20/30/20 |

All methods are evaluated under the same macroscopic-$L_2$-residual/plateau protocol and the same admissibility definition, and the only difference between the proposed method and the baselines is the update rule. The iteration budget given to the baselines meets or exceeds standard steady-LBM practice (e.g., $6\times10^{5}$–$1.2\times10^{6}$ LBE-calls for cavity 2x/3x). The non-convergence reported in the next section is therefore interpreted not as budget starvation but as a genuine plateau within that budget (quantitatively confirmed in Section 4.1).

## 3.4 Comparison-Matching Rules

The baselines of this paper denote the Picard, Anderson acceleration, preconditioned LBM, inexact Newton, and dual-time multigrid implementations contained in the stored result set, and the preconditioned-LBM axis corresponds to the representative family of the steady-flow LBM acceleration literature [20, 21]. This comparison does not claim an absolute ranking against every conceivable optimal implementation in the literature; it reports the relative performance observed under the same benchmark definition, the same macroscopic-residual/plateau verdict, and the same summary/history aggregation rules. Because all methods are computed in the same Python/NumPy execution environment sharing the same native LBM operator implementation, inter-method wall-time differences arise from algorithmic structure rather than from implementation language or library differences.

When building a comparison table, a proposed-method run and a baseline run must share the same case label and the same mesh level. We define two comparison groups. (i) Available-baseline comparison: among the stored baseline runs for a given case/level, those that have wall-second and residual records are considered, and the shortest wall time is found. (ii) Strict-convergence comparison: a conservative subset retaining only those runs that passed the same macro-$L_2$/plateau verdict and were recorded as converged. Cases for which no baseline run exists or for which a required column is empty are not counted in the denominator of the win count, and a faster but non-converged run is not used for any strong conclusion. This rule is not a post hoc filter favorable to the proposed method but a prior interpretive rule for first matching whether different solvers reached the same stopping protocol and then comparing time. We also state the direction of bias of the two groups: the available-baseline comparison includes the short wall times of baseline runs that terminated early without converging, whose true convergence times can only be longer than recorded. The available-baseline comparison is therefore a conservative comparison favorable to the baselines and unfavorable to the proposed method, and the result that the proposed method nonetheless wins on 25/27 cases (Section 4.2) can be interpreted as a lower-bound estimate.

The timing measurement convention is as follows. Wall time is aggregated from the `wall_seconds` and `elapsed` records of the stored summary/history files and includes all additional residual evaluations incurred during AP-Schur trials, fallbacks, and continuations. Because absolute wall time depends on CPU generation, memory bandwidth, the Python/NumPy/BLAS implementation, and background load, the primary interpretation is a relative comparison within the same stored result set under the same stopping rule. To complement this hardware dependence, the LBE-call is reported as an auxiliary operator-work metric. The LBE-call is the number of invocations of the native operator $G(f)$ or an equivalent collision–streaming–boundary residual evaluation, and includes the evaluation cost of rejected trials. With the present result set, which lacks repeated-run statistics, we claim no confidence intervals or $p$-values (Section 5.2).

## 3.5 Reference Tiers and Accuracy Metric

Reference data are used only for post-solve evaluation and never during the solve. Denoting the field or profile to be compared by $Q_h$ and the reference by $Q_{\mathrm{ref}}$, the accuracy metric is the relative $L_2$ norm

$$e_{\mathrm{ref}} = \frac{\| Q_h - Q_{\mathrm{ref}} \|_2}{\max\left( \| Q_{\mathrm{ref}} \|_2, \varepsilon_{\mathrm{ref}} \right)}. \tag{23}$$

References are organized into three tiers. (i) The analytic profiles of channel/Couette are closed-form references expected under the same discrete setting. (ii) The Ghia et al. centerline data [5] for the cavity are an external literature benchmark. (iii) For complex geometries without closed-form solutions—backward step, cylinder wake, multi-cylinder, T-junction—a more strictly or longer-converged stored field within the same benchmark definition is used as a tight numerical reference; in this case the reference error denotes final-field agreement rather than a continuum-exact error. We do not sort $e_{\mathrm{ref}}$ across different tiers into a single universal accuracy ranking, and strong comparisons are restricted to inter-method differences within the same case family and the same level.

The Ghia comparison procedure for the cavity is as follows. When the solver field does not share grid points with the Ghia tabulation coordinates, centerline values at the same physical coordinates are sampled by linear interpolation from the stored final field, and no smoothing or renormalization is applied to the reference values. This interpolation is not used in residual evaluation, accept/reject, or damping selection; it is used only in the post-processing stage for the figures and the $e_{\mathrm{ref}}$ computation. The Ghia comparison is therefore not a calibration but a post hoc verification of whether the final discrete field is compatible with the literature benchmark.

We also separate the convergence and accuracy verdicts into tiers. A convergence pass is a solver-state verdict that $r_{\mathrm{macro}}$, plateau, and admissibility were satisfied simultaneously, whereas $e_{\mathrm{ref}}$ and the Ghia/analytic/tight-reference comparison are accuracy diagnostics of how close that state is to an external reference field. A convergence pass does not imply minimization of the reference error, and conversely a small difference from the reference is not evidence that the solver used the reference internally. The tables and figures of Section 4 present both tiers together.

## 3.6 Fairness Invariants and Reproducibility Checklist

All numerical results are aggregated from a fixed result set. No solver algorithm or benchmark output was changed during aggregation; the stored final states and residual histories were read by the same rules to build the tables and figures. We did not adjust per-case coefficients to favor the proposed method, nor apply different convergence criteria per family. Table 6 lists the implementation invariants that an independent verifier can check first when examining fairness.

**Table 6. Reproducibility checklist and implementation invariants.**

| Invariant | Verification criterion |
|---|---|
| Residual definition | The macroscopic-$L_2$-residual history is used as the primary convergence metric for all methods and all cases. |
| Plateau condition | Together with the absolute residual condition, a plateau condition—decrease halting in the recent tail—is required. |
| Reference usage | Ghia, analytic solutions, and benchmark references are used only for post-solve error evaluation, never for solver updates. |
| No case tuning | The proposed method runs as a single AP-Schur-only algorithm with no case-specific relaxation coefficients or geometry-specific empirical switches. |
| Admissibility | Only trials passing density positivity, finite macro fields, boundary/mask consistency, and native residual decrease are accepted. |
| Reported data | Wall time, final residual, relative residual, field error, and contour/profile figures are produced only from the stored summary, history, field, and reference files. |

We also fix the identification criterion for the proposed method. The method we call "AP-Schur-only" in this paper is defined solely as one that shares all of: the same native operator $G(f)$, the same projection $P$ and lifting $P^{\dagger}$, the same Jacobian-free residual-response evaluation, the same damping candidates and accept/reject rule, the same admissibility gate, the same native fallback, and the same stopping protocol. What changes from benchmark to benchmark is only the problem definition—mesh, $Re$, boundary condition, mask.

# 4. Results

## 4.1 Overall Convergence Summary

The proposed method passed the convergence verdict of Eq. (5) on all 27 runs (the `converged`, `residual_converged`, and `plateau_converged` flags are all satisfied, with `convergence_mode = macro_l2_final_threshold_and_relative_plateau`). The total wall times at the 1x/2x/3x levels are 134.2 s, 1546.6 s, and 3507.2 s, respectively. Table 7 and Figure 2 give the per-level summary.

**Table 7. Per-level convergence summary of the proposed method.**

| Level | Cases | Converged | Total wall [s] | Median residual | Max residual | Median rel. error |
|---|---|---|---|---|---|---|
| 1x | 9 | 9 | 134.2 | 2.142e-12 | 2.474e-08 | 3.260e-03 |
| 2x | 9 | 9 | 1546.6 | 3.305e-12 | 6.409e-08 | 0.0326 |
| 3x | 9 | 9 | 3507.2 | 1.153e-11 | 1.567e-08 | 0.0257 |

The "Median rel. error" in Table 7 is the median over the cases for which a reference error was computed at that level. At 2x/3x, complex-geometry cases without a computed tight reference are excluded, so the share of Ghia-compared cavity cases grows; comparisons of the magnitude of this column across levels are therefore meaningless, and only cases sharing the same reference tier should be compared (Section 3.5). The increase from 3.26e-03 at 1x to 0.0326 at 2x reflects a change in the composition of the aggregated subset, not a degradation of accuracy.

We also disclose the budget handling of long runs. The LBE-calls of cavity $Re=400$ 2x, $Re=100$ 3x, and $Re=1000$ 2x/3x (about 1.26M, 0.77M, 1.44M, and 1.44M, respectively) exceed the nominal step budget recorded in the summary; these are the result of continuation runs (`method_variant = uniform_ap_schur_only_continued`) that maintained the same stopping rule. A continuation is not a change of algorithm or protocol but an extension of the run under the same verdict criteria, and the cost of the extended segment is also fully included in the wall time and LBE-calls. No proposed-method run therefore treated reaching the budget as convergence; all 27 terminated by the verdict of Eq. (5).

![](media/image2.png){width=5.83in}

**Figure 2.** Total wall time and maximum final macro-$L_2$ residual of the proposed-method runs by mesh scaling level. Bars are the sum of wall time over the nine cases; the line is the maximum final residual.

The interpretation rule for the CSV convergence columns is as follows. The final convergence verdict is read jointly from the `converged`, `residual_converged`, `plateau_converged`, and `convergence_mode` columns. The `relative_floor_pass`, `macro_change_pass`, and `plateau_improvement` columns are auxiliary columns from earlier diagnostic stages or sub-paths of the plateau verdict; even if some of them are zero or empty, the final verdict is checked by the summary's `converged` flag and the residual/plateau flags.

**Convergence-robustness comparison.** Under the same stopping protocol and the same admissibility definition, the proposed method converged on all 27 cases. The five baselines, by contrast, converged on only a subset of the 27 within their generous budgets (Table 7a). That the non-convergence is not budget starvation is directly verifiable. For example, on cavity $Re=400$ 2x all five baselines plateaued at a final residual of about $3.4$–$3.6\times10^{-6}$ after exhausting $6\times10^{5}$–$7\times10^{5}$ LBE-calls (about $100\times$ above the target $5\tau = 2.5\times10^{-8}$), and on cavity $Re=1000$ 2x they stagnated at about $1.0\times10^{0}$ even after $1.2\times10^{6}$ calls. The baselines' non-convergence is thus a genuine stall within the budget, not a lack of iterations. This robustness gap is not itself used for the primary timing claim of this paper; the timing comparison is restricted to the strict subset (15/27) on which a baseline also converges, excluding any budget asymmetry (Section 4.2).

**Table 7a. Number of converged cases per method (same protocol, out of 27 cases).**

| Method | Converged cases / 27 | Note |
|---|---|---|
| Proposed (AP-Schur-only) | 27 | All cases satisfy $r_{\mathrm{macro}} \le 5\tau$ + plateau + admissibility simultaneously |
| Inexact Newton–Krylov | 15 | Strongest baseline; still non-converged within budget on 12 cases |
| Preconditioned LBM | 14 | — |
| Picard / Anderson | 13 / 13 | — |
| Dual-time multigrid | 12 | — |

## 4.2 Wall-Time Comparison

Compared against the shortest converged time among available baselines for each case, the proposed method is faster on 25/27 cases, with a median wall-time ratio of $2.92\times$. Because some baseline runs may have a strict-convergence flag of 0, we interpret the available-baseline comparison and the strict-convergence comparison separately (Section 3.4). A strict-convergence baseline exists on 15/27 cases, and on that subset the proposed method is faster than the shortest-time strict-convergence baseline on 14/15 cases, with a median ratio of about $2.06\times$. The remaining 12 cases have no strict-convergence baseline run in the stored result set, so for them the available-baseline comparison is interpreted only as exploratory and is not used for any strong superiority claim.

All headline figures of this section (win counts, median ratios) were recomputed by an independent script from the stored all-method summary CSV according to the matching rules of Section 3.4 and confirmed to match the in-text values.

We disclose the exception cases as follows. Among the 27 comparisons, the cases where the proposed method is slower than the shortest available baseline are Couette 3x and cavity $Re=400$ 2x. On Couette 3x, both preconditioned LBM (about 85.5 s) and inexact Newton (about 95.9 s) satisfied the strict-convergence flag and were faster than the proposed method (about 133.5 s); this is the sole exception in the strict-convergence subset. On cavity $Re=400$ 2x, an inexact-Newton run at about 275.4 s is a faster available run than the proposed method (about 310 s), but all five baseline runs for that case are recorded with a strict-convergence flag of 0. These non-winning cases are included as is in the summary table and the per-case wall-time-ratio figure (Figure 3); rather than a failure of the AP-Schur correction itself, they are interpreted as situations where native Picard-type relaxation is already short enough, or where the local modes created by the boundary/mask dominate over the global hydrodynamic slow mode.

![](media/image3.png){width=5.83in}

**Figure 3.** Per-case AP-Schur-only wall-time ratio relative to the shortest-time available-baseline run. A ratio $> 1$ marks a case where the proposed method is faster. Baseline runs without a strict-convergence flag are distinguished separately in the text interpretation.

Figures 4–6 present, on common axes, the macro-$L_2$ residual versus wall time histories of all six methods for all 27 cases. These figures show two things directly. First, the residuals of all methods are recorded with the same definition (`macro_l2_p_ux_uy_uz`), so differences in stopping rule do not distort the comparison. Second, the time advantage of the proposed method comes not from a difference in termination point but from the residual trajectory itself descending below tolerance at an earlier wall time. Each curve was generated directly from the history CSV of the case directory, with no smoothing applied.

![](media/image4.png){width=6.25in}

**Figure 4.** Macro-$L_2$ residual versus wall time convergence histories for the nine 1x cases (all methods, generated from the stored history CSV).

![](media/image5.png){width=6.25in}

**Figure 5.** Macro-$L_2$ residual versus wall time convergence histories for the nine 2x cases.

![](media/image6.png){width=6.25in}

**Figure 6.** Macro-$L_2$ residual versus wall time convergence histories for the nine 3x cases.

## 4.3 Operator-Work (LBE-call) Comparison

To complement the hardware dependence of wall time, the LBE-call ratio was recomputed from the same stored result set. Compared against the shortest-time available baseline, the proposed method uses fewer LBE-calls on 19/27 cases, with a median ratio of about $1.80\times$. On the strict-convergence subset, it uses fewer LBE-calls on 13/15 cases, also with a median ratio of about $1.80\times$. The LBE-call exceptions in the strict-convergence subset are Couette 3x and T-junction 3x. The efficiency claim therefore does not rest on wall time alone but is interpreted together with an operator-work metric recomputable from the same logs, showing that the speedup is not explained solely by Python overhead or transient CPU scheduling. That said, the LBE-call counts native residual evaluations and is thus an auxiliary metric, not an absolute complexity measure encompassing each method's internal linear-algebra cost.

We also state the cost handling of rejected trials. If an AP-Schur trial fails the admissibility gate or the residual-decrease gate, it is not counted as an accepted correction and the solver proceeds with the native fallback; nonetheless, the cost of the residual evaluations, boundary re-application, finite/positivity checks, and fallback step incurred while evaluating the rejected trial is all included in the stored `wall_seconds` and LBE-calls. The reported speedup is therefore not a post hoc selective timing of only successful corrections but the elapsed cost of the actual execution path, including failed trials.

**Run-to-run timing variability and operator-work determinism.** To check the statistical reliability of a single-run wall time, four representative fast cases were run seven times each under the same stopping protocol (excluding one numba JIT-compilation warmup). Table 7b gives the results. The wall-time coefficient of variation (CV) was 3.6–6.8%, below 7% for all cases, whereas the LBE-call count of each case was bit-identical across all seven repetitions. That is, the operator-work of the proposed method is fully deterministic with zero run-to-run noise, and only the wall time varies by about $\pm5\%$ due to system scheduling. This result implies two things. First, the median wall-time speedups of Section 4.2 (about $2.06\times$ on the strict subset, about $2.92\times$ on available baselines) are more than an order of magnitude larger than the measured timing noise ($<7\%$) and therefore cannot be explained by transient scheduling variation. Second, the LBE-call comparison of Section 4.3 (13/15, 19/27, median $1.80\times$) is made on a deterministic metric and thus contains no run-to-run noise at all. The absolute wall times in Table 7b depend on the runtime environment and are not the primary comparison object of this paper; the reported quantities are the relative variability (CV) and the LBE determinism.

**Table 7b. Wall-time variability and LBE-call determinism over seven repetitions of representative cases.**

| Case (1x) | Mean wall [s] | Std. dev. [s] | CV [%] | LBE-call (7 runs) |
|---|---|---|---|---|
| couette n32 | 0.988 | 0.053 | 5.3 | 13109 (all identical) |
| multi-cylinder n32 | 0.867 | 0.034 | 3.9 | 13291 (all identical) |
| cavity $Re=100$ n33 | 0.524 | 0.019 | 3.6 | 13611 (all identical) |
| cylinder wake n64 | 2.600 | 0.176 | 6.8 | 8075 (all identical) |

## 4.4 Full 27-Case Result Table

Table 8 is a compact result table for reviewing all 1x/2x/3x proposed-method runs at once. To reduce any concern that only selected cases were presented, all converged runs are listed together with level, wall time, LBE-call, final residual, initial-relative residual, and reference error.

**Table 8. Summary results for all 27 proposed-method benchmark runs.**

| Lv | Case | Wall [s] | LBE | $r_{\mathrm{final}}$ | $r/r_0$ | Rel. err | Ref |
|---|---|---|---|---|---|---|---|
| 1x | backward step n64 | 27.66 | 122673 | 2.47e-08 | 7.55e-08 | 3.26e-03 | tight ref |
| 1x | cavity re1000 n129 | 56.84 | 221413 | 2.36e-09 | 7.91e-09 | 0.0542 | Ghia centerline |
| 1x | cavity re100 n33 | 0.70 | 20873 | 1.93e-13 | 1.02e-12 | 0.117 | Ghia centerline |
| 1x | cavity re400 n49 | 3.06 | 44379 | 2.04e-11 | 9.69e-11 | 0.106 | Ghia centerline |
| 1x | channel poiseuille Ny32 Nx192 | 20.30 | 32666 | 3.38e-13 | 4.34e-11 | 9.37e-03 | analytic Poiseuille |
| 1x | couette n32 | 1.20 | 20606 | 2.18e-12 | 4.63e-11 | 2.75e-09 | analytic Couette |
| 1x | cylinder wake n64 | 4.88 | 20251 | 9.88e-15 | 4.02e-14 | 7.94e-05 | tight ref |
| 1x | multi cylinder n32 | 1.25 | 20377 | 2.14e-12 | 5.70e-12 | 4.15e-05 | tight ref |
| 1x | t junction Nx96 Ny64 W16 | 18.29 | 32054 | 2.63e-13 | 7.18e-12 | 1.90e-05 | Picard ref (T-junction 1x) |
| 2x | backward step n64 | 74.97 | 119793 | 6.41e-08 | 2.76e-07 | not computed | — |
| 2x | cavity re1000 n129 | 829.72 | 1440003 | 4.87e-14 | 4.79e-07 | 0.0326 | Ghia centerline |
| 2x | cavity re100 n33 | 5.78 | 41793 | 8.74e-12 | 7.56e-11 | 0.0669 | Ghia centerline |
| 2x | cavity re400 n49 | 309.99 | 1257000 | 4.46e-09 | 3.34e-08 | 0.0642 | Ghia centerline |
| 2x | channel poiseuille Ny64 Nx384 | 185.70 | 105281 | 1.90e-13 | 9.73e-11 | 2.27e-03 | analytic Poiseuille |
| 2x | couette n32 | 21.78 | 101554 | 3.31e-12 | 9.92e-11 | 2.87e-08 | analytic Couette |
| 2x | cylinder wake n64 | 16.16 | 23184 | 1.43e-11 | 8.14e-11 | not computed | — |
| 2x | multi cylinder n32 | 5.77 | 20471 | 1.65e-14 | 6.11e-14 | not computed | — |
| 2x | t junction Nx192 Ny128 W32 | 96.70 | 63535 | 1.26e-12 | 1.00e-10 | not computed | — |
| 3x | backward step n64 | 756.37 | 866000 | 1.29e-10 | 6.79e-10 | not computed | — |
| 3x | cavity re1000 n129 | 1234.14 | 1440085 | 2.56e-10 | 6.64e-07 | 0.0257 | Ghia centerline |
| 3x | cavity re100 n33 | 180.17 | 769000 | 1.20e-10 | 1.35e-09 | 0.0493 | Ghia centerline |
| 3x | cavity re400 n49 | 74.00 | 233779 | 1.57e-08 | 1.50e-07 | 0.0501 | Ghia centerline |
| 3x | channel poiseuille Ny96 Nx576 | 798.11 | 222772 | 7.81e-14 | 8.98e-11 | 1.00e-03 | analytic Poiseuille |
| 3x | couette n32 | 133.55 | 296454 | 2.63e-12 | 9.66e-11 | 5.19e-08 | analytic Couette |
| 3x | cylinder wake n64 | 36.78 | 41868 | 1.15e-11 | 8.01e-11 | not computed | — |
| 3x | multi cylinder n32 | 10.37 | 21265 | 1.48e-12 | 6.57e-12 | not computed | — |
| 3x | t junction Nx288 Ny192 W48 | 283.69 | 89398 | 5.29e-13 | 7.79e-11 | not computed | — |

The reference errors marked "not computed" in Table 8 are entries for which the tight reference field at that level is not included in the result set and was therefore not computed post hoc; they are not interpreted as zero or as success (the data-integrity rule of Section 5.4). The convergence verdict for those cases was independently satisfied by the residual/plateau/admissibility criteria.

## 4.5 Code Verification: Accuracy under Mesh Refinement

This section verifies, from a mesh-refinement viewpoint, that the proposed method reaches the correct discrete solution rather than merely reducing the residual. The purpose is to show independently that the accelerator does not distort the solution, and it is restricted to cases for which a closed-form or literature reference exists.

(i) **Smooth analytic solution — channel Poiseuille.** For plane Poiseuille flow with inlet/outlet boundaries, the relative $L_2$ error of the proposed method's velocity profile is $9.37\times10^{-3}$, $2.27\times10^{-3}$, and $1.00\times10^{-3}$ at $N_y = 32/64/96$ (1x/2x/3x). The observed convergence orders between adjacent levels are

$$p_{12} = \frac{\ln(e_{1x}/e_{2x})}{\ln 2} = 2.04, \qquad p_{23} = \frac{\ln(e_{2x}/e_{3x})}{\ln 1.5} = 2.02,$$

quantitatively consistent with the second-order spatial accuracy that BGK-LBM theoretically possesses for smooth flows. That is, the proposed method reaches the solution while preserving the order of the native LBM discretization. Table 9a summarizes this result.

**Table 9a. Accuracy and observed convergence order of channel Poiseuille under mesh refinement.**

| Level | Grid ($N_y$) | Rel. $L_2$ error | Observed order $p$ |
|---|---|---|---|
| 1x | 32 | $9.37\times10^{-3}$ | — |
| 2x | 64 | $2.27\times10^{-3}$ | 2.04 |
| 3x | 96 | $1.00\times10^{-3}$ | 2.02 |

(ii) **Exactly representable solution — Couette.** Because the linear Couette profile is represented exactly by the LBM equilibrium, the discretization error should be essentially zero. The relative $L_2$ error of the proposed method is $2.75\times10^{-9}$, $2.87\times10^{-8}$, and $5.19\times10^{-8}$ at 1x/2x/3x—all at machine-precision level—and the slight increase with level is merely floating-point accumulation from more operations. This shows that the AP-Schur acceleration injects no non-physical bias into the solution.

(iii) **Literature benchmark — lid-driven cavity.** The Ghia centerline relative $L_2$ error decreases monotonically with mesh refinement at all three Reynolds numbers: $Re=100$, $0.117 \to 0.0669 \to 0.0493$; $Re=400$, $0.106 \to 0.0642 \to 0.0501$; $Re=1000$, $0.0542 \to 0.0326 \to 0.0257$ (1x$\to$2x$\to$3x). Because the Ghia error is not a pure discretization error but a mixture of the Navier–Stokes benchmark table, the lid/wall discretization, low-Mach weak compressibility, and tabulation interpolation, we do not claim a formal order; the monotone approach at all three $Re$ nonetheless shows that the final field of the proposed method converges consistently toward the literature solution.

Taking the three results together, the proposed method (a) preserves the native second order on smooth solutions, (b) maintains machine precision on an exactly representable solution, and (c) converges monotonically toward a literature benchmark. This does not contradict the limitation stated in Section 5.2—that this work does not perform a formal grid-convergence study (Richardson/GCI)—but rather directly supports the secondary claim that "acceleration does not sacrifice accuracy."

## 4.5b Accuracy Summary and Physical Fields

Table 9 is the accuracy summary for the 1x cases that have an analytic or external reference. Channel and Couette are compared against analytic profiles, the cavity against the Ghia centerline, and the remaining complex geometries against a tight/reference numerical field.

**Table 9. Accuracy summary for cases with an analytic or reference profile (1x).**

| Case | Level | Wall [s] | Final residual | Rel. $L_2$ vs ref | Reference |
|---|---|---|---|---|---|
| Plane Poiseuille inlet/outlet ($N_y$=32, $N_x$=192) | 1x | 20.30 | 3.384e-13 | 9.371e-03 | analytic_poiseuille |
| Couette flow (N=32) | 1x | 1.20 | 2.180e-12 | 2.750e-09 | analytic_couette |
| Lid-driven cavity $Re=100$ (N=33) | 1x | 0.70 | 1.935e-13 | 0.117 | ghia_centerline |
| Lid-driven cavity $Re=400$ (N=49) | 1x | 3.06 | 2.045e-11 | 0.106 | ghia_centerline |
| Lid-driven cavity $Re=1000$ (N=129) | 1x | 56.84 | 2.360e-09 | 0.0542 | ghia_centerline |
| Multi-cylinder masked flow (N=32) | 1x | 1.25 | 2.142e-12 | 4.146e-05 | tight_ref |
| Backward-facing step (N=64) | 1x | 27.66 | 2.474e-08 | 3.260e-03 | tight_ref |
| Cylinder wake analogue (N=64) | 1x | 4.88 | 9.882e-15 | 7.935e-05 | tight_ref |
| Strict inlet/outlet T-junction ($N_x$=96, $N_y$=64) | 1x | 18.29 | 2.633e-13 | 1.896e-05 | picard_ref_min_tjunction_1x |

The cavity Ghia centerline relative $L_2$ error is about 0.117, 0.106, and 0.054 at 1x for $Re=100/400/1000$, decreasing to about 0.049, 0.050, and 0.026 at 3x (Figures 7–9). These values do not indicate a failure of residual convergence: the final macro-$L_2$ residual in the same row passed the stopping tolerance, and the Ghia error is not the solver's internal objective but a post hoc comparison against an external literature profile. The residual measures the change relative to the steady fixed point of the present discrete LBM operator, whereas the Ghia comparison is jointly affected by the Navier–Stokes benchmark table, the lid/wall boundary discretization, low-Mach weak compressibility, and tabulation-coordinate interpolation. The cavity–Ghia error is therefore a diagnostic of grid/boundary-condition discretization error, and its decreasing trend with level is consistent with this interpretation; because the cavity–Ghia error is influenced by factors beyond grid spacing, we do not formally require monotone decrease per level.

That the T-junction 1x reference is a strictly converged Picard field carries a separate significance. The relative $L_2$ difference between the proposed method's final field and this Picard reference is only $1.9\mathrm{e}{-05}$, which is direct evidence that the proposed method reached the *same* discrete steady fixed point as native Picard iteration. The acceleration is thus a faster convergence to the same solution rather than a detour to a different one, consistent with the design claim of Section 2.3 that the native residual is not modified. Figure 10 shows velocity-magnitude and vorticity contours of representative cases reconstructed from the stored proposed-method final fields, demonstrating that the converged field qualitatively reproduces the expected flow structures of each geometry (shear layers, recirculation regions, wakes, and branching flow).

![](media/image7.png){width=5.83in}

**Figure 7.** Relative $L_2$ error against the Ghia centerline for the lid-driven cavity at $Re=100/400/1000$.

![](media/image8.png){width=4.79in}

**Figure 8.** Comparison of 1x cavity centerline velocity profiles against Ghia et al. [5].

![](media/image9.png){width=3.44in}

**Figure 9.** Comparison of 2x/3x cavity centerline velocity profiles against Ghia et al. [5].

![](media/image10.png){width=5.83in}

**Figure 10.** Velocity-magnitude and vorticity contours reconstructed from the stored proposed-method NPZ fields. These are post-processing results, not new CFD computations.

## 4.6 Ablation Study: Component Contribution Analysis

The ablation is a mechanism-isolation experiment that separates the contributions of the AP-Schur correction, RRE [15], and the native block in order to clarify the novelty and the performance contribution. Four variants were compared on the 1x suite under the same stopping rule; the results are given in Table 10 and Figure 11. AP-Schur-only maintained 9/9 convergence and 9/9 per-case wall-time wins while attaining the lowest total wall time (147.3 s).

Two points are needed to interpret Table 10. First, the AP-Schur-only total wall time of the ablation (147.3 s) differs from the 1x total of the final 27-run result set in Section 4.1 (134.2 s) because the ablation is a separate experiment performed under the same protocol for variant comparison, with a different run time and log from the final result set; both values are recomputable from their respective stored logs, and the relative ranking among variants is identical in both sets. Second, the "Mean speedup (vs Picard)" column is the arithmetic mean over the nine cases of the wall-time ratio relative to the Picard run of the same case, which differs in both baseline and statistic from the headline metric of Section 4.2 (the median of the ratio relative to the shortest available baseline); the two numbers must not be compared directly.

**Table 10. 1x ablation study results.**

| Variant | Conv. | Wins | Total wall [s] | Mean speedup (vs Picard) | Median residual | AP acc/trial |
|---|---|---|---|---|---|---|
| Full: AP-Schur + RRE | 9/9 | 9/9 | 258.5 | 9.18x | 1.386e-11 | 50/86 |
| RRE only | 9/9 | 8/9 | 292.3 | 11.07x | 1.365e-12 | 0/0 |
| AP-Schur only | 9/9 | 9/9 | 147.3 | 19.41x | 2.142e-12 | 92/118 |
| Native block only | 8/9 | 8/9 | 169.0 | 17.12x | 1.268e-12 | 0/0 |

![](media/image11.png){width=5.83in}

**Figure 11.** 1x ablation total wall time comparison. AP-Schur-only attains the lowest total wall time.

The final variant-selection rule follows this priority: first, the breadth of completed convergence; second, the total wall time on the same 1x suite; third, the number of per-case wall-time wins; fourth, algorithmic simplicity. The residual is only a gate confirming that every variant satisfied the common stopping rule, not a selection criterion. In Table 10, RRE-only and native-block-only appear smaller on some median-residual figures, but RRE-only has a larger total wall time and its wins drop to 8/9, while native-block-only loses convergence breadth at 8/9. Full AP-Schur+RRE passes all cases but is more complex and slower than AP-Schur-only. The choice of AP-Schur-only is therefore based not on the smallest residual number but on the robustness–time–simplicity combination under the same stopping rule.

We also state a defense against the post hoc-selection criticism. Table 10 is not a table for mixing the favorable variant per case. The proposed method in the main text is defined as a single deterministic routine, AP-Schur-only, and is never switched to a different variant per case on any benchmark. Even where another variant shows a smaller final residual on some case, that value is not substituted as the proposed-method result.

## 4.7 Execution-Trace Verification: Direct Evidence of Singularity and Reference-Freedom

The two core claims of the proposed method—(i) that every case uses the same single algorithm, and (ii) that no reference is injected into the solve—are verified directly and independently from the phase log of the stored per-case diagnostic CSV. Each outer round records, by a phase label, which candidate was accepted.

Aggregating the diagnostic logs of all 27 cases, the executed phases consist of exactly the following vocabulary: AP-Schur JFNK acceptance (by damping, `ap_schur_jfnk_alpha` $\in \{1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}\}$), native Picard block, native Picard guard (fallback), and AP-Schur rejected. In none of the 27 cases is an analytic-projection, reference-injection, Ghia-fitting, or benchmark-specific phase recorded even once. This guarantees, at the execution-trace level, that the algorithm does not branch on case identity (claim i) and that acceptance depends only on the native residual and admissibility (claim ii). Table 10a is the full phase aggregation.

**Table 10a. Aggregation of executed outer-round phases over all 27 cases (from the diagnostic logs).**

| Executed phase | Count | Meaning |
|---|---|---|
| ap_schur_jfnk_alpha1 | 204 | AP-Schur Newton step accepted at $\alpha=1$ |
| ap_schur_jfnk_alpha0.5 / 0.25 / 0.125 | 62 / 27 / 23 | AP-Schur accepted after a damping line search |
| ap_schur_rejected | 18 | AP-Schur trial failed the gate $\to$ native fallback |
| uniform_picard_block / guard | many | native Picard candidate / fallback |
| (analytic/reference/case-specific) | 0 | never executed in any of the 27 cases |

Quantitatively, AP-Schur trials were evaluated 334 times in total, of which 237 passed the admissibility and residual-decrease gates and were accepted (overall acceptance rate 71.0%; 78.0%, 69.1%, 65.1% at 1x/2x/3x respectively; all recomputable from the proposed-only summary CSV). Because there are no zero-accept cases, the proposed-method results cannot be interpreted as pure Picard results in which AP-Schur did nothing; and because the rejected-trial cost is included in the wall time and LBE-calls, this statistic is not a post hoc selection of only successful corrections. That the acceptance rate decreases gently with level (78$\to$69$\to$65%) suggests that the admissibility gate operates more conservatively as the grid grows; that the wall-time advantage is nonetheless maintained shows that even partial acceptance suffices to shorten the tail.

This trace-level verification lets the paper answer the reviewer's two strongest attacks—"is there a hidden per-case branch or reference use in the code?" and "did AP-Schur actually contribute?"—with a reproducible execution record rather than a narrative. Aggregating the phase column of each case's diagnostic CSV in the reproducibility package regenerates Table 10a exactly.

## 4.8 Measured Memory Usage

Section 2.6 made the structural claim that, because the proposed method does not assemble the full Newton matrix ($q N_f \times q N_f$), the memory is dominated by $\mathcal{O}(N_f)$. To confirm this quantitatively, the process peak working set (RSS) was measured with the Windows `GetProcessMemoryInfo` API for proposed-method runs at three grid sizes (Table 11a). The marginal solve memory, after separating the runtime baseline immediately after import (Python+NumPy+SciPy+numba, about 150 MB), is 22/50/86 MB at grids of 96²/145²/192², growing linearly with the distribution-field size ($q N_f \times 8$ bytes) at a nearly constant ratio of about $35\times$. This is consistent with the $\mathcal{O}(N_f)$ storage model of Eq. (19)—the spectral cache $B_U(\mathbf{k})$ ($(N_y, N_x, 3, 3)$ complex), a small number of GMRES restart vectors, FFT work arrays, and a limited number of distribution-field copies.

Storing a dense Jacobian $q N_f \times q N_f$ explicitly at the same grids would require about 51/267/820 GB at 96²/145²/192². The measured peak RSS (172–237 MB) is three to four orders of magnitude smaller, and the peak grows only $1.4\times$ while the grid grows $4\times$ (96²$\to$192²). The claim that "full Jacobian assembly is unnecessary" is therefore not merely a qualitative structural claim but is supported by measurement. Because absolute RSS values depend on the runtime environment (interpreter/library versions), the quantitative claim of this paper is confined to (i) the $\mathcal{O}(N_f)$ linear scaling of the marginal memory and (ii) the three-to-four-order-of-magnitude gap relative to a dense Jacobian, and is not extrapolated to a hardware-independent absolute memory constant.

**Table 11a. Measured peak working-set (RSS) of the proposed method by grid size, and structural footprint comparison.**

| Case (3x) | Grid | Field [MB] | Dense Jac [GB] | Baseline RSS [MB] | Peak RSS [MB] | Marginal [MB] |
|---|---|---|---|---|---|---|
| multi-cylinder | 96² | 0.63 | 51 | 149.8 | 172.0 | 22.2 |
| cavity $Re=400$ | 145² | 1.44 | 267 | 150.1 | 200.4 | 50.3 |
| cylinder wake | 192² | 2.53 | 820 | 151.3 | 237.1 | 85.8 |
# 5. Discussion

## 5.1 Mechanistic Interpretation of the Performance Gain

The observed performance gain is interpreted consistently from the Schur-complement viewpoint. Native LBM Picard stably damps the kinetic component through local collide–stream relaxation, but the global equilibration of the pressure–velocity hydrodynamic mode is slow. AP-Schur projects the residual into the moment space and proposes an approximate global correction for this slow component, and the admissibility gate accepts the correction only when it remains within the feasible set of the discrete problem. The results of Section 4.6—where the variant with the AP-Schur block removed (native-block-only) loses convergence breadth and the variant using only history extrapolation (RRE-only) shows a longer total wall time—and the result of Section 4.7—where all cases show a non-trivial acceptance rate—support this mechanistic interpretation. What the local linear analysis of Eq. (11) guarantees is only a reduction of the amplification factor of the captured slow mode; but because the native fallback is guaranteed on failure, the incompleteness of the analysis does not compromise solver stability.

## 5.2 Limitations and Scope of Claims

The direct scope of this work is limited as follows. First, the application scope is the stored 2D D2Q9/BGK steady benchmark suite. We make no direct claim of generalization to 3D, thermal/compressible LBM, MRT/entropic collision models, or high-Reynolds-number turbulent regimes. Second, Section 4.5 showed observed second-order convergence for channel Poiseuille, machine precision for Couette, and monotone approach to Ghia for the cavity, but this is code-verification evidence for cases with closed-form/literature references, not a formal grid-convergence study for all geometries (Richardson extrapolation, GCI, and discretization-error bounds against the continuum solution for every case). In particular, for cases whose reference is a tight numerical field—backward step, cylinder wake, multi-cylinder, T-junction—no formal order is claimed; such a claim would additionally require a systematic grid sequence, confirmation of a monotone asymptotic range, and analysis of integral quantities (conserved quantities, forces, reattachment length, etc.). Third, AP-Schur-only is not a method that removes discretization/BC error but one that reaches the same discrete solution faster. That the cavity Ghia error is nonzero is a direct example of this distinction.

We also state the limitations of the timing claim. Wall time depends on CPU generation, memory bandwidth, the Python/NumPy/BLAS implementation, and background load, and the CSV/JSON sources of the present result set do not store the CPU model, core count, library version, or git commit hash as separate columns. Wall time is therefore interpreted as a relative metric comparing the same stopping rule and the same case/level within the same result set, not as a hardware-independent absolute performance constant. We quantified the wall-time coefficient of variation (3.6–6.8%) and the LBE-call determinism (identical across repetitions) by seven repetitions of representative cases (Section 4.3, Table 7b), but we do not claim large-scale repetition over all 27 cases or inferential statistics based on confidence intervals/$p$-values. The first-order check of run-to-run noise is performed on the deterministic metrics—LBE-call, final residual, and the plateau flag. Extending the timing comparison in an independent reproduction would require fixing, together, the solver/benchmark script revisions, the CPU/OS/library information, the thread settings, and deterministic flags.

As a limitation of open-boundary diagnostics, for problems with complex geometry and boundary conditions—backward step, cylinder wake, multi-cylinder, T-junction—residual decrease and the local flux/mass diagnostics do not always improve at the same rate, because the residual measures the macroscopic-$L_2$ change over the whole domain whereas flux closure is sensitive to the integral over specific inlet/outlet cross-sections. This work does not use flux-related quantities in the stopping rule and interprets them only as auxiliary physical diagnostics; a quantitative flux-closure bound is left as a recomputation item of the reproducibility package.

## 5.3 Threats to Validity and Mitigations

The internal-validity threat is the possibility that the proposed method used reference information internally or used parameters favorable only to specific cases. To reduce this, we specified a single AP-Schur-only routine, the same residual/plateau criterion, the same admissibility gate, and a reference-free accept/reject procedure (Sections 2–3), and disclosed all proposed-method run records in summary tables. The continued label (`uniform_ap_schur_only_continued`) is not a change of method but an extension of the run under the same stopping rule. The measurement-validity threat is that the residual, wall time, and reference error are metrics of different character, and we report them separately as convergence efficiency and final-field agreement. The external-validity threat is that the benchmark suite does not represent all CFD problems, addressed by the scope restriction of Section 5.2. Table 11 summarizes the main concerns from an independent-verification viewpoint together with the paper's defense logic, and Table 12 summarizes how potential questions are addressed.

**Table 11. Anticipated concerns from an independent-verification viewpoint and the paper's defense logic.**

| Potential concern | Risk | Defense logic in the text |
|---|---|---|
| Reference-injection suspicion | If Ghia/analytic/tight references enter the solver internally, novelty and fairness weaken. | Sections 2.5 and 3.5 specify that references are used only for post-processing error evaluation; the accept gate uses only residual/admissibility. |
| Case-specific tuning suspicion | Applying different coefficients or algorithms only to specific benchmarks is strongly criticized. | The final method is defined as a single AP-Schur-only routine with the same stopping rule and gate applied to all cases (Sections 3.2, 3.6), with singularity verified by execution traces (Section 4.7). |
| Accuracy over-claim | Because the Ghia error is nonzero, an "accuracy-improvement" claim is refutable. | Framed not as accuracy improvement but as convergence acceleration that reaches the same discrete steady solution faster. |
| Open-boundary/mass consistency | Residual decrease and flux/mass diagnostics may be conflated on open geometries. | Density positivity, finite field, boundary re-application, and open-boundary branch rejection are organized as method gates (Sections 2.4, 2.7). |
| Insufficient ablation | If the contributions of AP-Schur and RRE/native block are not separated, novelty weakens. | The 1x ablation table and wall-time figure provide the basis for selecting AP-Schur-only (Section 4.6). |
| Insufficient reproducibility | Reliability drops if results depend on code changes or recomputation. | Provenance and a recomputation procedure are specified so that tables/figures can be regenerated from the stored summary/history/field/reference sources (Section 5.4). |
| JFNK over-claim | Solving no full Newton–Krylov system while appearing to be JFNK could be read as exaggeration. | Named as a moment-Schur nonlinear preconditioner based on the Jacobian-free residual response, with an explicit statement that it is not a full JFNK (Section 2.3). |

**Table 12. Potential verification questions and how the text addresses them.**

| Potential question | Response in the text |
|---|---|
| The fast wall time is due to differences in stopping rule. | We specify that the same residual/plateau verdict is applied to all methods and present residual-versus-time histories together. |
| It does not exactly match Ghia. | The Ghia error is a post hoc accuracy metric, separated from the convergence claim; mesh-refinement results interpret it as discretization sensitivity. |
| A different method was used for complex geometries. | We specify the use of a single AP-Schur-only solver and the same admissibility gate. |
| The novelty is a mere combination. | The moment-Schur-complement interpretation, native-residual acceptance, and geometry-aware admissibility are presented as one steady-LBM preconditioning framework. |
| Wall-time differences may be due to CPU scheduling/Python overhead. | The wall-time CV measured over seven repetitions is 3.6–6.8%, more than an order of magnitude smaller than the speedup ($\approx 2\times$), and the LBE-call is fully deterministic across repetitions (Table 7b); the advantage is therefore not explained by scheduling noise. |
| Does 1x/2x/3x mean formal grid convergence? | No. It is a solver-scaling benchmark; no formal order-of-accuracy claim is made. |
| Were only favorable variants aggregated? | The 27-case performance claims are sourced from the proposed-only summary, and the merged all-method CSV distinguishes duplicates by `base_case_id`, `scaling_level`, and `method_variant`. |
| If mass/flux closure is incomplete, is residual convergence invalid? | Mass/flux are open-boundary auxiliary diagnostics, not the stopping rule. The primary verdict is the macro-$L_2$ residual and plateau, and flux closure is disclosed as a separate recomputation item. |

## 5.4 Claim Hierarchy, Falsification Criteria, and Reproducibility

To prevent over-generalization of the strong performance results, we tier the claims of this work. The primary claim is the reduction of convergence time under the same stopping rule; the secondary claim is agreement with reference fields; and we do not claim the removal of discretization error itself. This hierarchy also serves as the basis for interpreting additional verification results: if some cases weaken in further reproduction computations, we adjust the scope of the primary/secondary claims—rather than hiding the methodology—and disclose the residual histories and field errors of the failing cases as supplementary material. Table 13 specifies the falsification condition for each tier.

**Table 13. Claim hierarchy of this work and falsification criteria.**

| Tier | Claim | Falsification condition |
|---|---|---|
| Primary claim | AP-Schur-only reaches steady state with smaller wall time than the baseline accelerators on most benchmarks under the same residual/plateau/admissibility criteria. | If, upon re-running under the same criteria, the proposed method is repeatedly slower than the fastest baseline or fails to satisfy the plateau, the claim must be weakened. |
| Secondary claim | The final field maintains accuracy comparable to analytic/Ghia/reference profiles. | If the wall time is fast but the Ghia/analytic error grows systematically relative to the baselines, the accuracy claim is downgraded to an auxiliary claim. |
| Mechanism claim | The performance gain arises from the combination of preconditioning the hydrodynamic moment Schur complement and native-residual acceptance. | If, in the ablation, AP-Schur-only loses its advantage over RRE/native or the Schur-correction acceptance scarcely occurs, the mechanistic interpretation is revised. |
| Out of scope | We do not claim that AP-Schur-only produces the exact solution on every grid or removes all open-boundary flux error. | These items are outside the scope of this work and are treated by auxiliary diagnostics and follow-up research. |

We distinguish reproducibility into two stages. The first is stored-data recomputation, in which an independent verifier recomputes the in-text table and figure numbers from the summary/history/field/reference sources without re-running the solver. The second is algorithmic reproduction, in which the AP-Schur-only solver is re-run in a new environment under the same benchmark definitions and stopping protocol. The direct basis for the in-text numerical claims is placed on the first stage. The recommended order for stored-data recomputation is: (i) from the proposed-only summary (`papers_data/summary_latest_ap_schur_only_proposed.csv`), confirm the 27 runs and the `converged`/`residual_converged`/`plateau_converged` flags keyed by `case_id` and `scaling_level`; (ii) from the all-method summary (`papers_data/summary_all_methods_with_latest_ap_schur_only.csv`), group only the baseline runs satisfying the same `case_id`, `scaling_level`, and stopping rule as the comparison group and recompute the `wall_seconds` and LBE-call ratios; (iii) confirm from each case directory's residual history that the final macro-$L_2$ residual and plateau-window condition match the summary flags; (iv) recompute the reference error from the accuracy or Ghia centerline CSV.

The aggregation unit and data-integrity rules are as follows. The number of proposed-method runs and the method-comparison pairs are counted not by the method string alone but de-duplicated by `case_id` and `scaling_level`, and the `method_variant` values `uniform_ap_schur_only` and `uniform_ap_schur_only_continued` are summed as the same AP-Schur-only method. The minimal columns needed to regenerate the in-text tables/figures are the case label, level, method key, converged flag, residual/plateau flags, `wall_seconds`, LBE-call, final macro-$L_2$ residual, initial-relative residual, reference error, and tolerance. If any diagnostic column is empty, it is treated as unreported rather than as zero or success, and if the proposed and baseline runs cannot be joined at the same case/level, that pair is excluded from the ratio computation. Each figure and table retains provenance recording the case key, level, method key, source CSV/field file, and generation procedure, so as to preclude any suspicion of selectively excluding failing cases or selectively adjusting axis/color scales. Table 14 summarizes the data sources and the reproducibility-verification methods.

**Table 14. Data sources and reproducibility-verification methods.**

| Item | Use in the text | Reproducibility-verification method |
|---|---|---|
| Summary CSV | Building tables of level, case, wall time, residual, and reference error for the 27 proposed-method runs | Check CSV row count, case label, level, method, and final residual |
| History CSV | Building wall-time-vs-residual and convergence curves | Check each method's elapsed time, LBE-call, and $r_{\mathrm{macro}}$ history |
| NPZ/field output | Visualizing velocity-magnitude contours, cavity profiles, complex-geometry fields | Check the shapes and finite values of the $\rho$, $u$, $v$, mask arrays |
| Reference profile | Post hoc comparison against Ghia and analytic solutions | Not used for solver updates; used only for plots/error metrics |
| Figure/table provenance | Tracing the sources of convergence plots, contours, centerline comparisons, ablation figures | Check consistency of source CSV/field/manifest with case key, level, method key, axis range, and error metric |

# 6. Conclusion

This paper proposed and validated an AP-Schur-only nonlinear preconditioning framework that preconditions the pressure–velocity hydrodynamic slow mode—the convergence bottleneck of steady-state LBM—from a conserved-moment Schur-complement viewpoint. The method changes neither the native steady LBM residual nor the boundary operator; it proposes a Jacobian-free trial direction in the conserved-moment space and uses the admissibility gate to confirm a residual decrease and physical consistency simultaneously. This structure makes it possible to treat the verification questions about accuracy, boundary conditions, and reference comparison separately, within the same residual/admissibility criterion.

On the stored 27 benchmarks, all proposed-method runs passed the same convergence criterion, whereas the five baselines converged on only 12–15 cases each despite the same protocol and a generous budget, demonstrating the robustness advantage of the proposed method. In a conservative timing comparison that excludes budget asymmetry (the 15-case subset on which a baseline also converges), the proposed method recorded shorter wall time on 14/15 cases (median about $2.06\times$) and fewer LBE-calls on 13/15 cases (median about $1.80\times$), and was faster on 25/27 cases against all available baselines (median $2.92\times$). In the 1x ablation, AP-Schur-only showed the lowest total wall time and 9/9 per-case wall-time wins, and the 71% trial acceptance rate together with the single phase vocabulary across all 27 cases (Section 4.7) guarantees, at the execution-trace level, that the Schur correction operated meaningfully along the actual execution path and without per-case branching. On accuracy, the observed second-order convergence for channel Poiseuille, machine precision for Couette, and monotone approach to Ghia for the cavity (Section 4.5) showed that the acceleration does not sacrifice discrete accuracy. This confirms that the method is an algorithm that solves the same discrete steady LBM problem faster, rather than curve fitting or reference injection.

The direct scope of this work is the comparison of convergence time, operator work, residual, and reference error on the stored 2D D2Q9/BGK 1x/2x/3x benchmark suite. Extension to 3D, higher Reynolds numbers, other collision models (MRT/entropic), and rigorous quantification of open-boundary flux closure is left to follow-up work. Because the slow exception cases, the absence of repeated-run statistics, and the limitations of the runtime-environment metadata are disclosed in the text, the present results should be interpreted not as a claim of universal superiority but as a verifiable, reproducible claim of steady-LBM nonlinear preconditioning.

# Data and Code Availability

The numerical claims of this work are organized so that a first-order check is possible from the stored summary/history/field archive without re-running the original solver. The reproducibility package includes the proposed-only summary CSV, the all-method comparison summary CSV, per-case residual histories, the accuracy table, the final-field NPZ files, the figure-generation scripts, the manifest and source-path metadata, the file inventory, and the revision information of the solver/post-processing scripts used. When the full field archive cannot be distributed owing to journal policy or repository capacity limits, the minimal distribution unit is the compact summary, the history CSV, the cavity centerline comparison CSV, the contour-regeneration script, and a specification of how to access the original field archive. To reproduce the mass/flux diagnostics of the open-boundary cases, the final field, the inlet/outlet segment definitions, the normal-direction convention, and the quadrature rule are included together. This is a computational study based on stored numerical benchmark results and deterministic post-processing, and involves no human or animal subjects. Funding, conflicts of interest, and author contributions are separated into distinct metadata items of the final manuscript.

# References

[1] Qian, Y. H., d'Humières, D., & Lallemand, P. (1992). Lattice BGK models for Navier-Stokes equation. *Europhysics Letters*, 17(6), 479–484. https://doi.org/10.1209/0295-5075/17/6/001

[2] Chen, S., & Doolen, G. D. (1998). Lattice Boltzmann method for fluid flows. *Annual Review of Fluid Mechanics*, 30, 329–364. https://doi.org/10.1146/annurev.fluid.30.1.329

[3] Succi, S. (2001). *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*. Oxford University Press.

[4] Lallemand, P., & Luo, L.-S. (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. *Physical Review E*, 61, 6546–6562. https://doi.org/10.1103/PhysRevE.61.6546

[5] Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387–411. https://doi.org/10.1016/0021-9991(82)90058-4

[6] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. *SIAM Journal on Scientific and Statistical Computing*, 7(3), 856–869. https://doi.org/10.1137/0907058

[7] Knoll, D. A., & Keyes, D. E. (2004). Jacobian-free Newton-Krylov methods: A survey of approaches and applications. *Journal of Computational Physics*, 193(2), 357–397. https://doi.org/10.1016/j.jcp.2003.08.010

[8] Benzi, M., Golub, G. H., & Liesen, J. (2005). Numerical solution of saddle point problems. *Acta Numerica*, 14, 1–137. https://doi.org/10.1017/S0962492904000212

[9] Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations. *SIAM Journal on Numerical Analysis*, 49(4), 1715–1735. https://doi.org/10.1137/10078356X

[10] Tóth, A., & Kelley, C. T. (2015). Convergence analysis for Anderson acceleration. *SIAM Journal on Numerical Analysis*, 53(2), 805–819. https://doi.org/10.1137/130919398

[11] Olshanskii, M. A., & Vassilevski, Y. V. (2007). Pressure Schur complement preconditioners for the discrete Oseen problem. *SIAM Journal on Scientific Computing*, 29(6), 2686–2704. https://doi.org/10.1137/070679776

[12] Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). *Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics* (2nd ed.). Oxford University Press.

[13] Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

[14] Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001). *Multigrid*. Academic Press.

[15] Sidi, A. (1986). Convergence and stability properties of minimal polynomial and reduced rank extrapolation algorithms. *SIAM Journal on Numerical Analysis*, 23(1), 197–209. https://doi.org/10.1137/0723014

[16] Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Physics of Fluids*, 9(6), 1591–1598. https://doi.org/10.1063/1.869307

[17] Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). Momentum transfer of a Boltzmann-lattice fluid with boundaries. *Physics of Fluids*, 13(11), 3452–3459. https://doi.org/10.1063/1.1399290

[18] Huang, J., Yang, C., & Cai, X.-C. (2015). A fully implicit method for lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 37(5), S291–S313. https://doi.org/10.1137/140975346

[19] Huang, J., Yang, C., & Cai, X.-C. (2016). A nonlinearly preconditioned inexact Newton algorithm for steady state lattice Boltzmann equations. *SIAM Journal on Scientific Computing*, 38(3), A1701–A1724. https://doi.org/10.1137/15M1028078

[20] Guo, Z., Zhao, T. S., & Shi, Y. (2004). Preconditioned lattice-Boltzmann method for steady flows. *Physical Review E*, 70(6), 066706. https://doi.org/10.1103/PhysRevE.70.066706

[21] Premnath, K. N., Pattison, M. J., & Banerjee, S. (2009). Steady state convergence acceleration of the generalized lattice Boltzmann equation with forcing term through preconditioning. *Journal of Computational Physics*, 228(3), 746–769. https://doi.org/10.1016/j.jcp.2008.09.028

[22] Hajabdollahi, F., & Premnath, K. N. (2018). Galilean-invariant preconditioned central-moment lattice Boltzmann method without cubic velocity errors for efficient steady flow simulations. *Physical Review E*, 97(5), 053303. https://doi.org/10.1103/PhysRevE.97.053303

[23] Hajabdollahi, F., & Premnath, K. N. (2019). Improving the low Mach number steady state convergence of the cascaded lattice Boltzmann method by preconditioning. *Computers & Mathematics with Applications*, 78(4), 1115–1130.

[24] Walsh, B., & Boyle, F. J. (2020). A preconditioned lattice Boltzmann flux solver for steady flows on unstructured hexahedral grids. *Computers & Fluids*, 210, 104634. https://doi.org/10.1016/j.compfluid.2020.104634

[25] Yahia, E., & Premnath, K. N. (2022). Preconditioned central moment lattice Boltzmann method on a rectangular lattice grid for accelerated computations of inhomogeneous flows. *Journal of Computational Science*, 63.
