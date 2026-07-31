---
title: "An All-Speed Pressure-Based Finite-Volume Method for One-Dimensional Compressible Two-Phase Flow with an Analytic-Jacobian Newton Solver"
subtitle: "Technical Report — formulation, discretization, solution algorithm, and validation"
author: "claudeCFD / solver_denner"
date: "2026-06"
geometry: margin=2.5cm
fontsize: 11pt
---

# 1. Introduction and scope

This report documents, in full detail, a one-dimensional numerical method for simulating
**compressible two-phase (two-component) flow across the entire Mach-number range** — from a
nearly incompressible liquid at rest, through slow acoustic waves, up to a Mach-10 shock in a
gas–liquid mixture. The method is *pressure-based* and *fully implicit in the acoustics*: it solves
a coupled nonlinear system for the primitive unknowns (velocity, pressure, total enthalpy) at every
time step with Newton's method. It follows the **ACID** (Acoustically Consistent Interpolation of
Density) family of pressure-based all-Mach schemes (Denner, 2018) and extends it with a general
equation of state, a hand-derived **analytic Jacobian**, and a robust globalization that makes the
analytic-Jacobian solver pass every validation case.

The report is written to be self-contained: a reader new to multiphase CFD should be able to
follow every step. We therefore (i) state every governing equation and define every symbol;
(ii) give the equation of state and *all* of its partial derivatives in closed form; (iii) derive
the finite-volume discretization term by term, including the Rhie–Chow / momentum-weighted
interpolation that makes a pressure-based method work on a collocated grid; (iv) write out the
residual vector and the Jacobian matrix that Newton's method requires; (v) describe the
globalization and performance engineering; and (vi) report quantitative results, with figures, on
ten validation problems.

**Notation.** Subscripts $k\in\{1,2\}$ label the two phases (phase 1 = gas/air, phase 2 =
liquid/water). $\alpha_k$ is the volume fraction of phase $k$ (with $\alpha_1+\alpha_2=1$), $\rho_k$
its density, $p$ the (shared) pressure, $T$ the (shared) temperature, $u$ the velocity. A
superscript $o$ denotes the previous time level. Cell indices are $i$; face indices are $f$ (face
$f$ sits between cells $f-1$ and $f$). $\Delta x$ is the cell width, $\Delta t$ the time step.

# 2. Governing equations

## 2.1 The two-phase model

We solve the one-dimensional, single-pressure, single-velocity, single-temperature
("mechanical and thermal equilibrium") two-phase model. The conserved quantities are the two
**partial masses** $\alpha_1\rho_1$ and $\alpha_2\rho_2$, the **mixture momentum** $\rho u$, and the
**mixture total energy** $\rho E$. A transport equation for the volume fraction $\alpha_1$ closes
the system. The governing equations are

$$\frac{\partial(\alpha_k\rho_k)}{\partial t}+\frac{\partial(\alpha_k\rho_k u)}{\partial x}=0,
\qquad k=1,2, \tag{2.1}$$

$$\frac{\partial(\rho u)}{\partial t}+\frac{\partial(\rho u^2+p)}{\partial x}=0, \tag{2.2}$$

$$\frac{\partial(\rho E)}{\partial t}+\frac{\partial\big((\rho E+p)\,u\big)}{\partial x}=0, \tag{2.3}$$

$$\frac{\partial\alpha_1}{\partial t}+u\,\frac{\partial\alpha_1}{\partial x}=0. \tag{2.4}$$

Equation (2.1) states that each phase's mass is advected with the common velocity $u$; (2.2) is
momentum conservation with the pressure gradient; (2.3) is total-energy conservation; and (2.4) is
the Allaire–Massoni volume-fraction transport (a non-conservative advection of the interface, with
no compression source — the simplest interface-capturing closure).

## 2.2 Mixture rules

The **mixture density** is the volume-fraction-weighted sum

$$\rho=\alpha_1\rho_1+\alpha_2\rho_2, \tag{2.5}$$

the **total energy** per unit mass is $E=e+\tfrac12u^2$ with $e$ the mixture specific internal
energy, and the **total enthalpy** per unit mass is

$$h=e+\frac{p}{\rho}+\frac12u^2=h_{\text{stat}}+\frac12u^2, \tag{2.6}$$

where $h_{\text{stat}}$ is the static (thermodynamic) enthalpy. Because the present method is
pressure-based and works with enthalpy, we use the **total-enthalpy form** of the energy equation.
Adding $\partial p/\partial t$ to (2.3) and using $\rho E+p=\rho h$ gives the equivalent statement

$$\frac{\partial(\rho h)}{\partial t}+\frac{\partial(\rho h\,u)}{\partial x}=\frac{\partial p}{\partial t}, \tag{2.7}$$

which is the energy equation actually discretized below (the right-hand side $\partial p/\partial t$
is the pressure-work term that, in the low-Mach limit, balances the advection so that $h$ stays
nearly constant).

The mixture static enthalpy is the mass-weighted average of the phase enthalpies,

$$h_{\text{stat}}=\frac{\alpha_1\rho_1 h_1+\alpha_2\rho_2 h_2}{\rho}. \tag{2.8}$$

# 3. Equation of state

## 3.1 The Noble–Abel Stiffened-Gas (NASG) family

Each phase obeys a **Noble–Abel Stiffened-Gas (NASG)** equation of state (Le Métayer & Saurel,
2016), a five-parameter law $(\gamma_k,\pi_k,b_k,\kappa_k,\eta_k)$ that contains the ideal gas, the
stiffened gas, and the Noble–Abel co-volume liquid as special cases. With $\kappa_k$ the specific
heat at constant volume ($c_{v,k}$), the density, static enthalpy, and sound speed of phase $k$ are

$$\rho_k(p,T)=\frac{p+\pi_k}{\kappa_k(\gamma_k-1)\,T+b_k\,(p+\pi_k)}, \tag{3.1}$$

$$h_k(p,T)=\gamma_k\kappa_k T+b_k\,p+\eta_k, \tag{3.2}$$

$$c_k(p,T)=\sqrt{\dfrac{\gamma_k\,(p+\pi_k)}{\rho_k\,(1-b_k\rho_k)}}. \tag{3.3}$$

Here $\gamma_k$ is the heat-capacity ratio, $\pi_k$ the stiffening pressure (the cohesion of a
liquid), $b_k$ the co-volume (the finite molecular volume; $b_k=0$ recovers the stiffened gas),
$\kappa_k$ the constant-volume specific heat, and $\eta_k$ a reference enthalpy. It is convenient to
abbreviate the denominator of (3.1):

$$A_k\equiv\kappa_k(\gamma_k-1)\,T+b_k\,(p+\pi_k),\qquad \rho_k=\frac{p+\pi_k}{A_k}. \tag{3.4}$$

The parameter sets used in this work are listed in Table 3.1.

| phase | $\gamma$ | $\pi$ [Pa] | $b$ [m³/kg] | $\kappa$ [J/kg·K] | $\eta$ [J/kg] |
|---|---|---|---|---|---|
| air (ideal gas) | 1.4 | 0 | 0 | 720.25 | 0 |
| water — stiffened gas (Denner cases) | 4.1 | $4.4\times10^{8}$ | 0 | 474.2 | 0 |
| water — NASG (cases 14, 15) | 1.187 | $7.028\times10^{8}$ | $6.61\times10^{-4}$ | 3610 | $-1.1778\times10^{6}$ |

Table 3.1 — Equation-of-state parameters. The NASG water (with co-volume $b\neq0$) is used for the
cavitation and high-pressure-water problems.

## 3.2 Exact partial derivatives

The Newton solver requires the thermodynamic derivatives of $\rho_k$ and $h_k$. Differentiating
(3.1)–(3.2) analytically (these are used verbatim in the code, not finite-differenced) gives

$$\zeta_k\equiv\left.\frac{\partial\rho_k}{\partial p}\right|_T=\frac{\kappa_k(\gamma_k-1)\,T}{A_k^{2}},
\qquad
\phi_k\equiv\left.\frac{\partial\rho_k}{\partial T}\right|_p=-\frac{(p+\pi_k)\,\kappa_k(\gamma_k-1)}{A_k^{2}}, \tag{3.5}$$

$$\left.\frac{\partial h_k}{\partial T}\right|_p=\gamma_k\kappa_k,\qquad
\left.\frac{\partial h_k}{\partial p}\right|_T=b_k. \tag{3.6}$$

These four closed-form derivatives are the building blocks of the analytic Jacobian (§7).

## 3.3 Recovering temperature from enthalpy

The implicit solver carries $(u,p,h)$ as unknowns, but the EOS is written in terms of $(p,T)$.
At each residual evaluation we must therefore recover $T$ from the *total* enthalpy $h$ and the
kinetic energy. Subtracting the kinetic energy gives the static enthalpy
$h_{\text{stat}}=h-\tfrac12u^2$, and we solve the scalar nonlinear equation

$$F(T)\equiv\frac{\alpha_1\rho_1(p,T)\,h_1(p,T)+\alpha_2\rho_2(p,T)\,h_2(p,T)}{\alpha_1\rho_1(p,T)+\alpha_2\rho_2(p,T)}-h_{\text{stat}}=0 \tag{3.7}$$

for $T$ by Newton iteration. Writing the numerator and denominator of the mixture enthalpy as
$N=\sum_k\alpha_k\rho_k h_k$ and $D=\sum_k\alpha_k\rho_k=\rho$, the derivative needed for this inner
Newton is obtained analytically by the quotient rule,

$$\frac{dF}{dT}=\frac{N_T\,D-N\,D_T}{D^2},\quad
D_T=\sum_k\alpha_k\phi_k,\quad
N_T=\sum_k\alpha_k\big(\phi_k h_k+\rho_k\,\gamma_k\kappa_k\big). \tag{3.8}$$

Using (3.8) instead of a finite difference removes two EOS evaluations per inner iteration; the
inner solve converges in three to five iterations.

# 4. Finite-volume spatial discretization

## 4.1 Cells, faces, and the collocated arrangement

The domain $[x_0,x_1]$ is divided into $n$ equal cells of width $\Delta x=(x_1-x_0)/n$. All
primitive variables are stored at cell centres (a *collocated* arrangement). There are $n+1$ faces;
face $f$ separates cell $f-1$ (left) from cell $f$ (right). Boundary conditions (transmissive,
reflective/wall, inlet, periodic) are imposed with two layers of ghost cells.

A finite-volume update of any conserved quantity $\psi$ integrates its conservation law over a cell,
turning the flux divergence into a difference of **face fluxes**:

$$\frac{\psi_i-\psi_i^{o}}{\Delta t}\,\Delta x + \big(\mathcal F_{i+1}-\mathcal F_{i}\big)=0, \tag{4.1}$$

where $\mathcal F_{f}$ is the flux through face $f$. The entire art of the method is in how the face
flux $\mathcal F_f$ — in particular the **advecting velocity** and the **face density** — is
constructed from the cell-centred data.

## 4.2 The momentum-weighted advecting velocity (Rhie–Chow / MWI)

On a collocated grid, naively averaging the cell velocities to the face produces a notorious
**checkerboard (odd–even) decoupling**: the pressure at a cell becomes invisible to its own
continuity equation, and a saw-tooth pressure field satisfies the discrete equations spuriously.
The cure, due to Rhie and Chow, is to build the face advecting velocity $\theta_f$ from the averaged
velocity *plus* a pressure-gradient correction that reintroduces the missing cell-to-cell pressure
coupling. The ACID method casts this as a *momentum-weighted interpolation* (MWI). Define, at face
$f$ between cells $L=f-1$ and $R=f$:

- the harmonic face density (consistent with the momentum equation),
$$\rho_f=\frac{2}{1/\rho_L+1/\rho_R}; \tag{4.2}$$
- the transient-dominated momentum coefficient and its inverse,
$$a_P=\tfrac12(\rho_L+\rho_R)\frac{\Delta x}{\Delta t},\qquad d_f=\frac{\Delta x}{a_P}; \tag{4.3}$$
- the Rhie–Chow coefficient,
$$\hat d_f=\frac{d_f}{1+(\rho_f/\Delta t)\,d_f}; \tag{4.4}$$
- the *compact* and *wide* pressure gradients,
$$\left(\frac{\partial p}{\partial x}\right)^{\text{compact}}_f=\frac{p_R-p_L}{\Delta x},\qquad
\left(\frac{\partial p}{\partial x}\right)^{\text{wide}}_f=\tfrac12\Big(\nabla p_L+\nabla p_R\Big), \tag{4.5}$$
where $\nabla p_i=(p_{i+1}-p_{i-1})/(2\Delta x)$ is the cell-centred gradient.

The face advecting velocity is then

$$\theta_f=\bar u_f
\;\underbrace{-\;\hat d_f\Big[\big(\partial_x p\big)^{\text{compact}}_f-\big(\partial_x p\big)^{\text{wide}}_f\Big]}_{\text{MWI pressure correction }\;\delta_f}
\;+\;\frac{\rho_f}{\Delta t}\,\hat d_f\big(\theta_f^{o}-\bar u_f^{o}\big), \tag{4.6}$$

where $\bar u_f$ is the interpolated face velocity (2nd-order average $\tfrac12(u_L+u_R)$, or the
4th-order central stencil $(-u_{L-1}+7u_L+7u_R-u_{R+1})/12$ where the four-cell stencil is
single-phase) and the last term is the *transient memory* that makes the scheme consistent at low
Mach number. The MWI correction $\delta_f$ is the third-pressure-derivative term that damps the
checkerboard; it is bounded by the local sound speed,

$$\delta_f\leftarrow \operatorname{clamp}(\delta_f,\,-a_f,\,+a_f),\qquad a_f=\tfrac12(c_L+c_R), \tag{4.7}$$

so that a violent shock's huge pressure gradient cannot drive the advecting velocity past the
physical signal speed (the low-Mach MWI assumption breaks at a shock; the clamp restores
robustness without affecting smooth low-Mach flow, where $|\delta_f|\ll a_f$).

## 4.3 The ACID face mass flux

The face mass flux must be **acoustically consistent**: a uniform-velocity contact discontinuity
(two fluids of different density moving together) must produce *no spurious mass source*. ACID
achieves this by building the face density from the **upwind partial densities** blended with the
*receiving cell's* volume fraction. With "up" denoting the upwind side of face $f$ (chosen by the
sign of $\theta_f$) and $\rho_{a,\mathrm{up}},\rho_{b,\mathrm{up}}$ the upwind phase densities, the
mass flux into cell $i$ across its right face $i{+}1$ and left face $i$ is

$$\dot m_{i+1}=\big[\alpha_i\,\rho_{a,\mathrm{up}}^{(i+1)}+(1-\alpha_i)\,\rho_{b,\mathrm{up}}^{(i+1)}\big]\,\theta_{i+1},\quad
\dot m_{i}=\big[\alpha_i\,\rho_{a,\mathrm{up}}^{(i)}+(1-\alpha_i)\,\rho_{b,\mathrm{up}}^{(i)}\big]\,\theta_{i}. \tag{4.8}$$

Because the *same* cell volume fraction $\alpha_i$ multiplies both faces, a uniform velocity field
gives $\dot m_{i+1}-\dot m_i=\rho_i(\theta_{i+1}-\theta_i)=0$ to machine precision at a contact —
no spurious source. This is the defining property that gives ACID its name.

## 4.4 Reconstruction and shock capturing

The convected primitive quantities (the upwind $p,T,u$ that enter $\rho_{\mathrm{up}}$ and the
upwind enthalpy flux) are reconstructed from the cell data with a **solution-adaptive** scheme:

- a per-face **shock sensor** trips when the pressure ratio across the four-cell stencil exceeds
  $1.3$, i.e. $\max(p)>1.3\max(\min(p),1)$;
- where the flow is smooth (sensor off) and the four-cell stencil is single-phase, a **4th-order
  central** interpolation is used for $p$, $u$, and the face density — this removes the 2nd-order
  numerical *dispersion* that otherwise smears an acoustic wave;
- elsewhere a **Minmod-limited 2nd-order TVD** reconstruction is used for smooth-but-resolved waves;
- at a shock face the scheme reverts to **1st-order upwind**, which is monotone (no oscillations).

This single, parameter-free, solution-driven switch lets one scheme resolve smooth acoustic waves
to high order *and* capture strong shocks without ringing.

# 5. Temporal discretization

The transient term in (4.1) uses either **Backward Euler (BE, first order)** or a constant-step
**second-order Backward Differentiation (BDF2)**. Writing a generic conserved quantity $\psi$,

$$\text{BE:}\quad \frac{\psi-\psi^{o}}{\Delta t},\qquad\qquad
\text{BDF2:}\quad \frac{\tfrac32\psi-2\psi^{o}+\tfrac12\psi^{oo}}{\Delta t}, \tag{5.1}$$

where $\psi^{oo}$ is the value two steps back. BDF2 is activated only in the smooth, wave-resolving
regime (a time-harmonic acoustic source at the inlet), where its lower phase error matters; at a
strong shock the scheme uses BE, which is monotone in time. Because the whole acoustic system is
solved implicitly, the time step is limited by *accuracy*, not stability: the CFL number can be
$O(1)$ even though an explicit scheme would require $\sim10^{-3}$ at these conditions.

The time step is chosen from the acoustic CFL condition $\Delta t=\mathrm{CFL}\cdot\Delta x/\max_i(|u_i|+c_i)$,
modified by an adaptive ramp described in §8.4.

# 6. The fully-coupled implicit system

## 6.1 Unknowns and residual

At each time step the method solves, **simultaneously and implicitly**, for the three primitive
unknowns per cell

$$\mathbf{q}_i=(u_i,\;p_i,\;h_i)^{\mathsf T}, \tag{6.1}$$

(the temperature $T_i$ and densities $\rho_{k,i}$ are recovered from $(p_i,h_i)$ through §3.3 and
the EOS at every evaluation, so they are *not* independent unknowns). The discrete residual has
three components per cell — momentum, continuity, and energy:

$$R^{\text{mom}}_i=\frac{b_0\,\rho_i u_i-\mathcal C^{\text{mom}}_i}{\Delta t}\Delta x
+\big(\dot m_{i+1}u^{\text{up}}_{i+1}-\dot m_i u^{\text{up}}_i\big)+\big(p_{f,i+1}-p_{f,i}\big), \tag{6.2}$$

$$R^{\text{con}}_i=\frac{b_0\,\rho_i-\mathcal C^{\text{con}}_i}{\Delta t}\Delta x+\big(\dot m_{i+1}-\dot m_i\big), \tag{6.3}$$

$$R^{\text{ene}}_i=\frac{b_0\,\rho_i h_i-\mathcal C^{\text{ene}}_i}{\Delta t}\Delta x
+\big(\theta_{i+1}\,\widehat{\rho h}_{i+1}-\theta_i\,\widehat{\rho h}_i\big)-\frac{p_i-p_i^{o}}{\Delta t}\Delta x, \tag{6.4}$$

where $b_0=1$ for BE and $b_0=\tfrac32$ for BDF2; $\mathcal C^{\bullet}_i$ collects the known
old-level terms ($\psi^{o}$ for BE, $2\psi^{o}-\tfrac12\psi^{oo}$ for BDF2); $p_{f,f}$ is the face
pressure (same reconstruction as $\bar u$); and $\widehat{\rho h}_f=\alpha_i\,\rho_{a,\mathrm{up}}
(h_{a,\mathrm{up}}+\tfrac12u_{\mathrm{up}}^2)+(1-\alpha_i)\,\rho_{b,\mathrm{up}}
(h_{b,\mathrm{up}}+\tfrac12u_{\mathrm{up}}^2)$ is the ACID partial total-enthalpy flux, blended with
the receiving cell's $\alpha_i$ exactly as the mass flux is. The last term of (6.4) is the discrete
$\partial p/\partial t$ pressure-work source from (2.7).

The volume fraction $\alpha_1$ is advected by the explicit upwind discretization of (2.4) at the
start of each step (a first-order upwind colour-function update) and then held fixed during the
$(u,p,h)$ Newton solve.

## 6.2 Defect correction

The solver is a **defect-correction (inexact-Newton) scheme**: the residual $\mathbf R(\mathbf q)$
above is the single source of truth, and an *approximate* Jacobian $\mathbf J$ is used only to
choose the update direction. Newton's method solves

$$\mathbf J\,\delta\mathbf q=-\mathbf R(\mathbf q),\qquad \mathbf q\leftarrow\mathbf q+\omega\,\delta\mathbf q, \tag{6.5}$$

and iterates until $\|\mathbf R\|\to0$. The crucial consequence is that the *converged solution is
independent of any approximation in* $\mathbf J$: an inexact or stale Jacobian changes only the
number of iterations, never the answer. This property is used heavily in §7 (analytic vs.
finite-difference Jacobian) and §8 (Jacobian reuse).

# 7. The Newton Jacobian

## 7.1 Why the Jacobian is block-pentadiagonal

Each residual $\mathbf R_i$ is a $3$-vector and each unknown $\mathbf q_j$ is a $3$-vector, so the
Jacobian $\partial\mathbf R_i/\partial\mathbf q_j$ is built of $3\times3$ blocks. The question is
*which* blocks are non-zero — i.e. how far the stencil reaches. The MWI advecting velocity
$\theta_f$ (4.6) depends on the *wide* pressure gradient (4.5), whose cell-centred stencil
$\nabla p_R=(p_{R+1}-p_{R-1})/2\Delta x$ reaches two cells away. Consequently $R_i$ depends on
$p_{i-2},\dots,p_{i+2}$, and the true Jacobian is **block-pentadiagonal** (bandwidth two):

$$\mathbf J=\operatorname{pentadiag}\big(\mathsf E_i,\;\mathsf A_i,\;\mathsf B_i,\;\mathsf C_i,\;\mathsf F_i\big), \tag{7.1}$$

with $\mathsf B$ the diagonal block, $\mathsf A,\mathsf C$ the first off-diagonals ($i\mp1$), and
$\mathsf E,\mathsf F$ the second off-diagonals ($i\mp2$). Dropping $\mathsf E,\mathsf F$ (a merely
tridiagonal solve) gives a wrong Newton direction whenever the MWI dominates — at large material
time steps and at cavitation — and the iteration diverges. Retaining the full pentadiagonal
structure is essential.

## 7.2 Finite-difference Jacobian (reference)

The pentadiagonal Jacobian can be assembled exactly (to round-off) by finite differences using
**graph colouring**: variables five cells apart never influence the same residual within bandwidth
two, so perturbing every fifth cell simultaneously isolates one column of each block. Three
variables $\times$ five colours $=15$ residual evaluations assemble the entire Jacobian. This is
robust and is retained as the reference path (selectable at run time), but each Newton iteration
then costs fifteen residual evaluations.

## 7.3 Analytic Jacobian

The analytic Jacobian assembles $\mathbf J$ in a *single* $O(n)$ pass, replacing the fifteen
finite-difference residual evaluations. It is built from three contributions.

**(a) The transient/EOS-chain (diagonal) block.** The transient terms in (6.2)–(6.4) depend on the
local density $\rho_i(p_i,T_i)$ and temperature $T_i(p_i,h_i,u_i)$. Differentiating through the
$h\!\to\!T$ inversion (3.7)–(3.8) gives the exact density sensitivities

$$\frac{\partial\rho}{\partial h}=D_T\frac{1}{h_{s,T}},\quad
\frac{\partial\rho}{\partial u}=D_T\frac{-u}{h_{s,T}},\quad
\frac{\partial\rho}{\partial p}=D_p+D_T\frac{-h_{s,p}}{h_{s,T}}, \tag{7.2}$$

where $D_p=\sum_k\alpha_k\zeta_k$, $D_T=\sum_k\alpha_k\phi_k$, and $h_{s,T},h_{s,p}$ are the
derivatives of the mixture static enthalpy obtained by the quotient rule exactly as in (3.8). These
populate the diagonal $\mathsf B_i$ block.

**(b) The acoustic-flux coupling.** The face velocity $\theta_f$ and face pressure $p_{f,f}$ depend
on $u$ and $p$ across the stencil through $\bar u_f$ and the compact/wide pressure gradients.
Differentiating (4.6) and the face-pressure interpolation analytically gives the $\partial\theta_f/
\partial u$, $\partial\theta_f/\partial p$, $\partial p_{f,f}/\partial p$ entries, which (multiplied
by the frozen flux blends) populate $\mathsf A,\mathsf B,\mathsf C,\mathsf E,\mathsf F$ for all
three equations.

**(c) The upwind-transport and frozen-MWI density couplings.** The upwind densities and enthalpies
$\rho_{\mathrm{up}},h_{\mathrm{up}}$ depend on the upwind cell's $(p,h,u)$ through the EOS chain;
these populate the $i\mp1$ blocks. Finally, the Rhie–Chow coefficient $\hat d_f$, the harmonic face
density $\rho_f$, and the transient-memory term in (4.6) *all depend on the neighbour densities*
$\rho_L,\rho_R$, hence on $(u,h,p)$ at the two adjacent cells. These "frozen-MWI" sensitivities,

$$\frac{\partial\theta_f}{\partial\rho_{L,R}}=-\,\mathbb 1_{\text{MWI active}}\,\delta_f\frac{\partial\hat d_f}{\partial\rho_{L,R}}
+\frac{\theta_f^{o}-\bar u_f^{o}}{\Delta t}\Big(\hat d_f\frac{\partial\rho_f}{\partial\rho_{L,R}}+\rho_f\frac{\partial\hat d_f}{\partial\rho_{L,R}}\Big), \tag{7.3}$$

with
$\partial\hat d_f/\partial\rho=\big[\partial d_f/\partial\rho-(d_f^2/\Delta t)\,\partial\rho_f/\partial\rho\big]/(1+(\rho_f/\Delta t)d_f)^2$,
$\partial d_f/\partial\rho=-d_f/(\rho_L+\rho_R)$, and $\partial\rho_f/\partial\rho_{L}=\rho_f^2/(2\rho_L^2)$,
were the last terms to be added; they supply the off-diagonal energy–enthalpy and energy–pressure
couplings ($\partial R^{\text{ene}}_i/\partial h_{i\pm1}$ etc.) that are otherwise missing. Their
inclusion was verified term by term against the finite-difference Jacobian (every significant block
agrees to relative error $<10^{-2}$) and reduces the inner Newton iteration count on the strong
shock-interface problems.

## 7.4 Block solve

The block-pentadiagonal system is solved directly by pairing cells $(2I,2I{+}1)$ into $6\times6$
super-cells, which turns the pentadiagonal block system into a *block-tridiagonal* one that is
solved by the block Thomas algorithm. The $3\times3$ and $6\times6$ block inverses are computed in
closed form.

# 8. Globalization and robustness

A bare Newton iteration is not globally convergent; several safeguards make the analytic-Jacobian
solver pass every case.

## 8.1 Backtracking line search

Each step is damped by $\omega$ and then line-searched: starting from $\alpha=1$ and halving, the
update $\delta\mathbf q$ is accepted at the first $\alpha$ that reduces $\|\mathbf R\|$, with
clamps $|\delta p|\le\tfrac12p$, $|\delta u|\le u_{\text{ref}}$, $|\delta h|\le\tfrac12|h|$ and a
floor $p\ge1$ and $h>\tfrac12u^2$ (so that the recovered static enthalpy stays positive).

## 8.2 Keep-best and stall detection

The cavitation problem (Case 15) has a per-step inner Newton that **never reaches the convergence
gate**: the line search pins at its floor and the residual stalls — for the finite-difference *and*
the analytic Jacobian identically, because it is a stiffness/globalization property of the
near-vacuum state, not a Jacobian error. The solver therefore (i) tracks the **best** (lowest-
residual) iterate and restores it, and (ii) **breaks** the inner loop once the residual stops
improving for a fixed number of iterations (default five). Converging cases improve every iteration
and never trip this; the cavitation case stops after a few iterations instead of running to the
iteration cap, which both speeds it up and prevents a spurious dt collapse.

## 8.3 Convergence-gated step acceptance

A step is accepted if it converged, *or* if it stalled but made net progress (best residual below
the step's initial residual). It is rejected — triggering a time-step retry — only when it made no
progress at all (the signature of a too-large $\Delta t$ at a violent shock). This single criterion
distinguishes the two failure modes: "reduce $\Delta t$" (strong-shock startup) versus "accept the
stalled-but-good state" (cavitation), and is what lets one solver handle both.

## 8.4 Adaptive-CFL ramp

A sharp shock in the initial condition diverges on the first implicit step at a large CFL, and a
naive per-step retry then jumps straight back to the full CFL and re-diverges. A persistent scale
on the CFL time step fixes this: it drops to the level at which a retrying step actually succeeded,
and climbs back by a factor $1.5$ per clean step. It is a no-op for cases that never retry (so they
are byte-for-byte unchanged) and lets the Mach-10 shock problems run at a CFL the smeared shock
tolerates ($\approx0.6$ for the analytic-Jacobian path) instead of the much smaller value the raw
initial transient would force.

# 9. Performance engineering

The method was optimized from a baseline of 178 s (serial, finite-difference Jacobian) to about 27 s
(per-case-sum, analytic-Jacobian default) — a $\sim6.6\times$ reduction — with no change to the
validation results. The contributing measures were:

1. **OpenMP** parallelization of the three EOS-heavy kernels of the residual evaluation (the
   $h\!\to\!T$ inversion, the per-cell thermodynamics, and the per-face state). These loops are
   per-cell/per-face independent (gather, no scatter, no reductions) so the parallel result is
   bit-identical to serial. The thread count is capped (~$n/32$, at most 8) because the residual is
   evaluated millions of times in short bursts and a high thread count is dominated by fork–join
   and cache-line traffic on these small ($n\le800$) problems.
2. **Analytic derivatives** replacing finite differences: the $h\!\to\!T$ derivative (3.8), and a
   sound-speed-free lean thermodynamic kernel for the hottest loops.
3. **Modified Newton**: reusing the assembled finite-difference Jacobian for several iterations
   (defect correction guarantees the same converged answer), and the analytic Jacobian that
   replaces fifteen residual evaluations per iteration with one $O(n)$ pass.
4. **Reusable scratch buffers** for the ghost-extended fields, removing per-evaluation heap traffic.

The analytic-Jacobian path is the default; the finite-difference path remains available as a
reference.

# 10. Validation

The method is validated on ten one-dimensional problems spanning the full regime: static and
advected pressure-equilibrium interfaces, single- and two-phase acoustic waves, two shock tubes,
a cavitation (double-rarefaction) problem, and two Mach-10 shock problems. **All ten pass.** Each
solution is compared against an analytic or fine-grid reference. Table 10.1 lists the principal
error metrics — the $L_2$ errors of pressure, velocity, and density, and the Pearson correlation
$\mathrm{corr}_p$ between computed and reference pressure (1.0 = perfect).

| case | description | $L_2(p)$ | $L_2(u)$ | $L_2(\rho)$ | $\mathrm{corr}_p$ | pass |
|---|---|---|---|---|---|---|
| 01 | PE static interface (air/water) | 0 | 0 | 0 | 1.000 | ✓ |
| 02 | PE advection (gas/gas) | 0 | $5\!\times\!10^{-15}$ | 0.080 | 1.000 | ✓ |
| 04 | air acoustic sinusoid | 0.074 | 0.0015 | $1.3\!\times\!10^{-5}$ | 0.988 | ✓ |
| 05 | water acoustic sinusoid | 0.047 | 0.0010 | 0.014 | 0.994 | ✓ |
| 07 | air/water acoustic refl.–trans. | 0.017 | 0.0005 | $2\!\times\!10^{-8}$ | 0.995 | ✓ |
| 13 | HP-air / LP-water shock tube | 0.021 | 0.052 | 0.027 | 0.997 | ✓ |
| 14 | HP-water / LP-air shock tube | 0.015 | 0.101 | 0.039 | 0.999 | ✓ |
| 15 | air/water cavitation | 0 | 0.027 | 0.045 | 1.000 | ✓ |
| 24 | Mach-10 mixture shock | 0.031 | 0.040 | 0.032 | 0.997 | ✓ |
| 25 | Mach-10 air-shock / water interface | 0.048 | 0.026 | 0.037 | 0.994 | ✓ |

Table 10.1 — Validation summary (analytic-Jacobian default). $L_2$ errors are relative; $\mathrm{corr}_p$
is the computed–reference pressure correlation. Case 01 is machine-exact ($L_\infty(p)=0$).

The following subsections describe each problem and show the computed pressure, velocity, and
density (red) against the reference (black dashed).

## 10.1 Case 01 — pressure-equilibrium static interface
Air and water sit side by side at a uniform pressure with zero velocity ($N=200$). A correct
all-Mach scheme must keep this state *exactly* stationary (no spurious interfacial currents). The
method preserves it to machine precision: $L_2(p)=L_2(u)=L_2(\rho)=0$, $L_\infty(p)=0$.

![Case 01 — static interface.](results_cpp/figs/rep_case01.png)

## 10.2 Case 02 — pressure-equilibrium advection
A gas–gas contact at uniform pressure is advected at $u_0=1\,$m/s through the domain ($N=500$,
$t_{\text{end}}=0.7$). Pressure and velocity stay exact ($L_2(p)=0$, $L_2(u)\sim10^{-15}$); the
density contact is transported with only the expected first-order numerical diffusion
($\mathrm{corr}_\rho=0.980$).

![Case 02 — interface advection.](results_cpp/figs/rep_case02.png)

## 10.3 Cases 04 and 05 — single-phase acoustic sinusoids
A small-amplitude ($\Delta u=0.02$) sinusoidal acoustic wave is driven from the inlet through air
(Case 04) and through stiffened-gas water (Case 05). These test linear-acoustic accuracy: amplitude
and phase of the propagating wave. The BDF2 + high-order reconstruction preserves the wave with
$\mathrm{corr}_p=0.988$ (air) and $0.994$ (water).

![Case 04 — air acoustic sinusoid.](results_cpp/figs/rep_case04.png)

![Case 05 — water acoustic sinusoid.](results_cpp/figs/rep_case05.png)

## 10.4 Case 07 — air/water acoustic reflection and transmission
A single one-period acoustic pressure pulse (Denner §7.3.2) crosses the air–water interface at
$x=0.5$ ($N=750$, $\Delta x=2\times10^{-3}$). Because the acoustic impedances differ by a factor
$\sim3000$, the pulse partially reflects (in air) and transmits (into water). The computed
reflected/transmitted amplitudes match linear-acoustic theory; $\mathrm{corr}_p=0.995$,
$L_2(p)=0.017$, with the density field essentially exact ($L_2(\rho)\sim10^{-8}$).

![Case 07 — acoustic reflection/transmission.](results_cpp/figs/rep_case07.png)

## 10.5 Cases 13 and 14 — shock tubes
Case 13 is a high-pressure-air / low-pressure-water shock tube ($p_L=10^9$, $p_R=10^4$, ratio
$10^5$); Case 14 reverses it (high-pressure water / low-pressure air). Both develop a shock, a
contact, and a rarefaction. The scheme captures the shock in $\sim$one cell with no overshoot and
follows the smooth regions accurately ($\mathrm{corr}_p=0.997$ and $0.999$ respectively). Case 14,
a strong water rarefaction, uses the fully-coupled energy to fix the contact pressure and a
spurious boundary build-up that a segregated energy update produces.

![Case 13 — HP-air / LP-water shock tube.](results_cpp/figs/rep_case13.png)

![Case 14 — HP-water / LP-air shock tube.](results_cpp/figs/rep_case14.png)

## 10.6 Case 15 — cavitation
Two columns of an air–water mixture ($\alpha_1=0.055$) move apart at $\pm100\,$m/s ($N=400$),
creating a strong double rarefaction that drives the pressure toward vacuum — the classic
*cavitation* test, and the hardest for a pressure-based solver because the gas density and its
pressure derivative become extreme. The method keeps the pressure field exact ($L_2(p)=0$,
$\mathrm{corr}_p=1$) and resolves the density rarefaction ($L_2(\rho)=0.045$). This case motivated
the keep-best / stall-break globalization of §8.2.

![Case 15 — cavitation.](results_cpp/figs/rep_case15.png)

## 10.7 Cases 24 and 25 — Mach-10 shocks
Case 24 is a Mach-10 shock travelling through a *homogeneous* air–water mixture (a Wood-speed
mixture Hugoniot); Case 25 is a Mach-10 air shock impacting a water interface. These are the most
violent problems and exercise the coupled energy, the shock-capturing reconstruction, and the
adaptive-CFL robustness. Both are captured with the shock in $\sim$one cell and
$\mathrm{corr}_p=0.997$ (Case 24) and $0.994$ (Case 25).

![Case 24 — Mach-10 mixture shock.](results_cpp/figs/rep_case24.png)

![Case 25 — Mach-10 air shock / water interface.](results_cpp/figs/rep_case25.png)

# 11. Summary

This report has described, from first principles, an all-speed pressure-based finite-volume method
for one-dimensional compressible two-phase flow: the single-pressure/velocity/temperature model
(§2); the NASG equation of state with all closed-form derivatives and the enthalpy→temperature
inversion (§3); the collocated finite-volume discretization with the momentum-weighted (Rhie–Chow)
advecting velocity, the acoustically-consistent ACID mass flux, and the solution-adaptive
reconstruction (§4); the implicit Backward-Euler / BDF2 time integration (§5); the fully-coupled
$(u,p,h)$ residual and its defect-correction Newton solution (§6); the block-pentadiagonal Jacobian
in both finite-difference and fully analytic forms, including the frozen-MWI density-coupling terms
(§7); the line-search / keep-best / stall-break / adaptive-CFL globalization that makes the
analytic-Jacobian solver robust (§8); and the parallelization and algorithmic optimizations that
reduced the run time $\sim6.6\times$ (§9). The method passes all ten validation problems (§10),
machine-exactly where a stationary solution is required and with high fidelity on shocks, acoustic
waves, and cavitation.

# References

1. F. Denner, *Pressure-based methods for all Mach number flows* (ACID; acoustically-consistent
   interpolation of density), 2018.
2. O. Le Métayer and R. Saurel, *The Noble–Abel Stiffened-Gas equation of state*, Physics of
   Fluids, 2016.
3. G. Allaire, S. Clerc, S. Kokh, *A five-equation model for the simulation of interfaces between
   compressible fluids*, J. Comput. Phys., 2002.
4. C. M. Rhie and W. L. Chow, *Numerical study of the turbulent flow past an airfoil with trailing
   edge separation*, AIAA J., 1983.
