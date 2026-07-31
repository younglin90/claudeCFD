# 2. Numerical Method

## 2.1 Native steady-state LBM residual and notation

We consider the two-dimensional, nine-velocity (D2Q9) lattice Boltzmann equation with a single-relaxation-time (BGK) collision operator,
$$
f_i(\mathbf{x}+\mathbf{c}_i,\,t+1) = f_i(\mathbf{x},t) - \frac{1}{\tau_f}\big[f_i(\mathbf{x},t) - f_i^{\mathrm{eq}}(\mathbf{x},t)\big], \qquad i = 0,\dots,8, \tag{1}
$$
where $\mathbf{x}$ is a node of the square lattice $\mathbb{Z}^2$, $t$ is the discrete time step, $\{\mathbf{c}_i\}_{i=0}^{8}$ are the D2Q9 lattice velocities, lattice units are used throughout ($\Delta x = \Delta t = 1$), and $\tau_f$ is the BGK relaxation time. The discrete equilibrium is
$$
f_i^{\mathrm{eq}}(\rho,\mathbf{u}) = w_i\,\rho\left[1 + \frac{\mathbf{c}_i\cdot\mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_i\cdot\mathbf{u})^2}{2 c_s^4} - \frac{|\mathbf{u}|^2}{2 c_s^2}\right], \qquad c_s^2 = \tfrac{1}{3}, \tag{2}
$$
with weights $w_0 = 4/9$, $w_{1\text{–}4} = 1/9$, and $w_{5\text{–}8} = 1/36$. The macroscopic moments are
$$
\rho = \sum_i f_i, \qquad \rho\mathbf{u} = \sum_i \mathbf{c}_i f_i, \qquad p = c_s^2 \rho. \tag{3}
$$
The kinematic viscosity is $\nu = c_s^2(\tau_f - \tfrac{1}{2})$, and the flow is characterized by $\mathrm{Re} = U_{\mathrm{ref}} L_{\mathrm{ref}}/\nu$ and $\mathrm{Ma} = U_{\mathrm{ref}}/c_s$. All benchmarks are run at $\mathrm{Ma}\ll 1$, so the weakly compressible pressure $p = c_s^2\rho$ is consistent with the incompressible reference solutions.

We denote by $G$ a single full lattice step, comprising collision, streaming, and the application of all boundary conditions. The steady state is the fixed point of this operator, expressed as the residual equation
$$
R(f) = f - G(f) = 0, \tag{4}
$$
with $G$ invoked as a stand-alone operator that the acceleration scheme leaves unmodified.

Two linear operators connect the nine-component distribution field to its three conserved hydrodynamic moments. The projection $M \in \mathbb{R}^{3\times 9}$ maps a distribution to its density and momentum,
$$
M f = (\rho,\ \rho u_x,\ \rho u_y)^\top, \tag{5}
$$
its rows being the moment weights $\{1,\,c_{x,i},\,c_{y,i}\}$. The lifting $T \in \mathbb{R}^{9\times 3}$ reconstructs a minimal distribution increment from a conserved-moment increment $\mathrm{d}m = (\mathrm{d}\rho,\,\mathrm{d}(\rho u_x),\,\mathrm{d}(\rho u_y))$,
$$
(T\,\mathrm{d}m)_i = w_i\big[\mathrm{d}\rho + 3\,c_{x,i}\,\mathrm{d}(\rho u_x) + 3\,c_{y,i}\,\mathrm{d}(\rho u_y)\big], \tag{6}
$$
so that $T$ is a right inverse of $M$, $MT = I_3$. The increment $T\,\mathrm{d}m$ is the minimal hydrodynamic distribution perturbation that matches a prescribed change in density and momentum while leaving the remaining kinetic components to be relaxed by $G$ at the next iteration.

## 2.2 Conserved-moment Schur-complement formulation

A Newton correction for the steady state solves $J_f(f^\ast)\,\mathrm{d}f = -R(f^\ast)$, where $J_f = \partial R/\partial f$. Assembling the full Jacobian, together with its mask and boundary operators, is costly in both memory and implementation. We therefore split the correction into a conserved-moment part $\mathrm{d}m = (\mathrm{d}\rho,\,\mathrm{d}(\rho u_x),\,\mathrm{d}(\rho u_y))$, consistent with $M$ and $T$ above, and a kinetic part $\mathrm{d}k$, which gives the block-partitioned system
$$
\begin{bmatrix} J_{mm} & J_{mk} \\ J_{km} & J_{kk} \end{bmatrix}
\begin{bmatrix} \mathrm{d}m \\ \mathrm{d}k \end{bmatrix}
= -\begin{bmatrix} R_m \\ R_k \end{bmatrix}. \tag{7}
$$
Eliminating the kinetic block yields the Schur-complement system for the conserved moments,
$$
S_m\,\mathrm{d}m = -\big(R_m - J_{mk} J_{kk}^{-1} R_k\big), \qquad S_m = J_{mm} - J_{mk} J_{kk}^{-1} J_{km}. \tag{8}
$$
The kinetic block $J_{kk}$ collects the locally, rapidly relaxing modes, whereas the moment Schur complement $S_m$ encodes the global, weakly damped hydrodynamic coupling that controls the slow pressure–velocity modes and dominates the late stagnating phase of the iteration. Acceleration is therefore required only for the three conserved-moment components, not for all nine distributions per node.

Forming $S_m^{-1}$ explicitly is intractable, so we approximate its action analytically, with the closed-form per-mode block derived in the spectral construction that follows. Working entirely in moment space, the practical correction reads
$$
S_m \approx M J_f T, \qquad S_m\,\mathrm{d}m \approx -M R(f^\ast), \qquad \mathrm{d}f_{\mathrm{AP}} = T\,\mathrm{d}m, \tag{9}
$$
where $M$ and $T$ are the operators of Eqs. (5)–(6) and $MT = I_3$ guarantees consistency. Interior solid and obstacle nodes are excluded; only fluid nodes enter the computation. The action of $J_f$ on a direction $v$ is obtained, matrix-free, by a directional finite difference of the native residual,
$$
J_f(f)\,v \approx \frac{R(f + \varepsilon v) - R(f)}{\varepsilon}, \qquad \varepsilon = 10^{-7}\,\frac{1 + \|f\|_2}{\|v\|_2}, \tag{10}
$$
where $10^{-7}$ is the forward-difference scale at IEEE double precision. The method is thus a moment-Schur nonlinear preconditioner built on the Jacobian-free residual response, not a full JFNK; the Jacobian $J_f$ is never assembled.

## 2.3 Spectral AP-Schur preconditioner

To obtain a closed-form approximation of $S_m^{-1}$ we linearize the lattice update about a uniform base state $\bar\rho = 1$, $\bar{\mathbf{u}} = 0$. In Fourier space, streaming becomes a diagonal phase operator $A(\mathbf{k}) = \mathrm{diag}(e^{-i\mathbf{k}\cdot\mathbf{c}_i})$, and the global problem decouples mode by mode. The linearized BGK collision is $C(\omega) = (1-\omega) I_9 + \omega\,TM$ with $\omega = 1/\tau_f$, the single linearized update is $L'(\mathbf{k}) = A(\mathbf{k}) C(\omega)$, and the fixed-point residual Jacobian is $J(\mathbf{k}) = I_9 - L'(\mathbf{k})$. Reducing to moment space gives a $3\times 3$ Schur complement per wavenumber.

The Galerkin reduction of this Jacobian is
$$
S_m^{G}(\mathbf{k}) = M\,J(\mathbf{k})\,T = I_3 - M\,A(\mathbf{k})\,T, \tag{11}
$$
which omits the influence of the kinetic modes; moments and kinetic modes interact through streaming and collision, and the resulting damping depends on $\omega$. We restore this coupling through the correction
$$
S_m^{\mathrm{AP}}(\mathbf{k}) = S_m^{G}(\mathbf{k}) - \kappa(\omega)\big[M A(\mathbf{k})^2 T - (M A(\mathbf{k}) T)^2\big]. \tag{12}
$$
The bracketed correction term reflects an exact identity,
$$
M A(\mathbf{k})^2 T - (M A(\mathbf{k}) T)^2 = M A(\mathbf{k})\,(I_9 - TM)\,A(\mathbf{k})\,T, \tag{13}
$$
which is precisely the moment$\to$kinetic$\to$moment coupling, i.e. the moment-space representation of $J_{mk} J_{km}$ appearing in Eq. (8). The Galerkin reduction discards exactly this term; approximating the kinetic inverse $J_{kk}^{-1}$ by the scalar $\kappa(\omega)$ then makes Eq. (12) a first-order, structurally exact reconstruction of $-J_{mk} J_{kk}^{-1} J_{km}$. The scalar is
$$
\kappa(\omega) = \tfrac{1}{2}\,\mathrm{sign}(r)\,\min\!\big(\tfrac{1}{2},\,|r|\big), \qquad r = \frac{1-\omega}{\omega}, \tag{14}
$$
so that the cap keeps $|\kappa|\le 1/4$ and the correction stays bounded for all $\omega\in(0,2)$, including the limit $\omega\to 0$ where $r\to+\infty$ and $\kappa$ saturates at the cap.

To control the conditioning of the per-mode operator we add an adaptive Tikhonov shift,
$$
S_m^{\mathrm{reg}}(\mathbf{k}) = S_m^{\mathrm{AP}}(\mathbf{k}) + \eta\,I_3, \qquad \eta = \frac{\sigma_{\max}(S_m^{\mathrm{AP}})}{50}, \tag{15}
$$
which, with $\eta$ set to a fixed fraction of $\sigma_{\max}$, limits the per-mode condition number to approximately $50$. The per-mode preconditioner block is then $B_m(\mathbf{k}) = [S_m^{\mathrm{reg}}(\mathbf{k})]^{-1}$. The mean mode $\mathbf{k} = (0,0)$ is tied to mass conservation: we take no Newton step on the density mean and pass the momentum mean through unchanged, so $B_m(0) = \mathrm{diag}(0,1,1)$.

The assembled preconditioner $B_m$ acts on a residual field by projecting to moments, transforming, applying the cached per-mode blocks $B_m(\mathbf{k})$, transforming back, and lifting,
$$
B_m R_f = T\,\mathcal{F}^{-1}\big\{ B_m(\mathbf{k})\cdot \mathcal{F}[M R_f](\mathbf{k}) \big\}. \tag{16}
$$
Because $B_m(\mathbf{k})$ depends only on $(N_y, N_x, \omega)$, it is built once per case at cost $O(N_f \log N_f)$ and reused at every application. The Fourier linearization assumes periodicity and so does not represent non-periodic boundary conditions exactly; this is standard Krylov preconditioning practice and does not affect the converged solution, which is governed by the native nonlinear residual and the admissibility gate described next.

## 2.4 Jacobian-free Newton step and admissibility gate

The spectral operator $B_m$ is used as a left preconditioner for a single preconditioned Newton step on the native residual. In each outer round we run a left-preconditioned GMRES on
$$
J_f(f^k)\,\mathrm{d}f = -R(f^k), \tag{17}
$$
with the operator action $v \mapsto J_f(f^k)\,v$ supplied by the finite difference of Eq. (10), preconditioner $B_m$, Krylov subspace dimension $k_{\max} = 30$, restart $= 2k_{\max}$, and a single outer iteration.

The GMRES direction $\mathrm{d}f$ is not accepted directly. Instead it is subjected to a damped line search coupled to an admissibility gate. With trial states $f_{\mathrm{trial}}(\alpha) = f^k + \alpha\,\mathrm{d}f$ and $\alpha \in \{1, \tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}\}$, we accept the largest $\alpha$ whose trial is admissible and strictly reduces the native residual,
$$
\text{accept } \alpha \iff \mathrm{admissible}(f_{\mathrm{trial}}) \ \wedge\ \|R(f_{\mathrm{trial}})\| < \|R(f_{\mathrm{best}})\| \ \wedge\ \mathrm{conservation}(f_{\mathrm{trial}}). \tag{18}
$$
Candidates are tested from the largest step first, the first passing candidate is accepted, and if none passes the step is recorded as rejected and the solver falls back to a native LBM update. The admissibility predicate comprises:

| Gate | Condition |
|---|---|
| Finite field | reject any NaN or Inf |
| Positive density | reject $\rho \le 0$ |
| Residual decrease | accept only if the native macroscopic residual decreases |
| Boundary consistency | re-apply native wall/inlet/outlet/mask before evaluating the residual |
| Conservation sanity | mass drift and inlet–outlet flux closure not worsened relative to native |

Trial states are interior-only: the native boundary projection is re-applied before the residual, positivity, finiteness, and mask checks, and solid and mask nodes are excluded from both the fluid-domain norm $\|\cdot\|_{\Omega_f}$ and the projection $M$.

## 2.5 Solver procedure and scale-only adaptation

The complete method is summarized in Algorithm 1; its only adaptive element is a global scale $s$ that sets the burn-in and block lengths. The burn-in is the number of Picard (native LBM) iterations used for initial stabilization, and the block is the number of Picard iterations forming each per-round candidate. The scale is purely geometric,
$$
s = \max\!\left(\sqrt{\frac{N_{\mathrm{dof}}}{9\cdot 32^2}},\,1\right), \tag{19}
$$
where $N_{\mathrm{dof}} = 9 N_f$, so that $s$ equals the linear grid size divided by $32$ ($s = 1$ on a $32\times 32$ D2Q9 lattice), independent of $\mathrm{Re}$, boundary conditions, and mask. The lengths are $\mathrm{burn} = \mathrm{clip}(\mathrm{round}(16 s),\,8,\,96)$ and $\mathrm{block} = \mathrm{clip}(\mathrm{round}(80 s),\,48,\,512)$.

The conceptual workflow is summarized in Figure 1: the moment residual is extracted, an AP-Schur correction is formed and validated by the admissibility gate, and either the accepted update or a native fallback is applied.

> **Algorithm 1** — AP-Schur accelerated steady-state LBM
>
> 1. Burn-in: $f \leftarrow \mathrm{Picard}^{\,\mathrm{burn}}(f_0)$; initialize $f_{\mathrm{best}}$, $r_{\mathrm{best}}$.
> 2. For round $= 1,\dots,R_{\max}$ ($R_{\max} = 160$):
>     a. Picard candidate: $c_{\mathrm{pic}} = \mathrm{Picard}^{\,\mathrm{block}}(f)$, residual $r_{\mathrm{pic}}$.
>     b. AP-Schur candidate: solve Eq. (17) by $B_m$-preconditioned GMRES for $\mathrm{d}f$, then form $c_{\mathrm{ap}}$, $r_{\mathrm{ap}}$ via the line-search/gate of Eq. (18).
>     c. Choose the candidate with the smallest residual.
>     d. If no candidate beats $r_{\mathrm{best}}$ by a factor $1.02$, fall back to a native Picard guard.
>     e. Set $f \leftarrow$ chosen; update $f_{\mathrm{best}}$, $r_{\mathrm{best}}$, and the staleness counter.
>     f. Terminate if $r_{\mathrm{best}} \le \tau$ or if staleness $\ge$ stale$_{\max}$ ($= 40$).
> 3. Return $f_{\mathrm{best}}$ and the convergence history.

A single AP-Schur step is not cheaper than a single Picard step; it requires several native residual evaluations, a spectral solve, and a line search. The benefit is structural rather than per-step: an early global correction removes the slow hydrodynamic mode that would otherwise require thousands to hundreds of thousands of Picard tail iterations to damp, thereby shortening the total time to steady state. Because the full Newton matrix is never formed, the memory footprint scales as $O(N_f)$; the full cost model is given in the appendix.

Global mass and inlet–outlet flux closure are reported as auxiliary physical-plausibility diagnostics, defined in the appendix; they are not part of the stopping rule, which uses only the macroscopic residual and its plateau.

---

# Appendix

## A.1 Cost and memory model

Let $N_f$ be the number of fluid nodes, $q = 9$ the number of discrete velocities, and $n_m = 3$ the number of conserved moments. The cost of one outer round is
$$
C_{\mathrm{round}} \approx (n_G + n_{\mathrm{trial}})\,C_G + C_{\mathrm{FFT}}, \qquad C_{\mathrm{FFT}} = O(n_m N_f \log N_f), \tag{A.1}
$$
where $C_G$ is the cost of one native lattice step, $n_G$ is the number of native steps per round (burn-in/block Picard work plus the guard), and $n_{\mathrm{trial}}$ is the number of residual evaluations spent in the finite-difference Jacobian action and the line search. The memory footprint is
$$
W_{\mathrm{mem}} \approx q N_f + O(n_m N_f) + O(N_b), \tag{A.2}
$$
i.e. the full distribution field, the moment and spectral buffers, and the boundary bookkeeping over $N_b$ boundary nodes. The full Newton matrix, of size $qN_f \times qN_f$, is never formed. Peak resident-set-size measurements at three grid resolutions confirm that the marginal memory $W_{\mathrm{mem}}$ grows linearly in $N_f$ (empirically of order $35\times$ the field size) and lies three to four orders of magnitude below a dense Jacobian. Absolute memory usage is environment-dependent, so the claim is restricted to the $O(N_f)$ scaling and the order-of-magnitude gap.

## A.2 Mass-conservation and boundary-consistency diagnostics

Residual convergence is distinct from exact mass conservation: the macroscopic $L_2$ residual measures the change of $p$ and $\mathbf{u}$ over the whole fluid domain, whereas global mass drift and inlet/outlet flux closure are sensitive to the boundary and mask treatment. We therefore retain the residual and its plateau as the primary convergence indicators and report the following two quantities only as auxiliary diagnostics. The global mass and its relative drift are
$$
\mathcal{M}^n = \sum_{\Omega_f} \rho^n\,\mathrm{d}V, \qquad \varepsilon_{\mathcal{M}}^n = \frac{|\mathcal{M}^n - \mathcal{M}^0|}{\max(|\mathcal{M}^0|,\,\epsilon)}, \tag{A.3}
$$
and the inlet–outlet flux closure is
$$
\varepsilon_Q^n = \frac{\big|\sum_{\mathrm{out}} \mathrm{flux} + \sum_{\mathrm{in}} \mathrm{flux}\big|}{\max\!\big(\sum_{\mathrm{in}} |\mathrm{flux}|,\,\epsilon\big)}. \tag{A.4}
$$
Neither quantity is used as a stopping condition; for closed cavities $\varepsilon_Q$ is not applicable. The reported results are obtained with no post hoc mass correction. Positivity and boundary consistency enter only through the admissibility gate.