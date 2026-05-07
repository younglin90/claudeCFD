![](_page_0_Picture_0.jpeg)

# **Linearly implicit all Mach number shock capturing schemes for the Euler equations**

Stavros Avgerinos, Florian Bernard, Angelo Iollo, Giovanni Russo

### **To cite this version:**

Stavros Avgerinos, Florian Bernard, Angelo Iollo, Giovanni Russo. Linearly implicit all Mach number shock capturing schemes for the Euler equations. Journal of Computational Physics, 2019, ⟨10.1016/j.jcp.2019.04.020⟩. ⟨hal-02419411⟩

# **HAL Id: hal-02419411 <https://inria.hal.science/hal-02419411v1>**

Submitted on 19 Dec 2019

**HAL** is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire **HAL**, est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

![](_page_0_Picture_9.jpeg)

[HAL Authorization](https://about.hal.science/hal-authorisation-v1/)

# Linearly Implicit All Mach Number Shock Capturing Schemes

Stavros Avgerinos, Florian Bernard, Angelo Iollo, Giovanni Russo

June 29, 2018

#### Abstract

We propose a family of simple second order accurate schemes for the numerical solution of Euler equation of gas dynamics that are (linearly) implicit in the acoustic waves, eliminating the acoustic CFL restriction on the time step. The general idea is that explicit differential operators in space relative to convective or material speeds are discretized by upwind schemes or local Lax-Friedrics fluxes and the linear implicit operators, pertaining to acoustic waves, are discretized by central differences. We have compared the results of such schemes on a series of one-dimensional test problems including classical shock tube configurations. Also we have considered low-Mach number acoustic wave propagation tests as well as nozzle flows in various Mach regimes. The results show that these schemes do not introduce excessive numerical dissipation at low Mach number providing an accurate solution in such regimes. They perform reasonably well also when the Mach number are not too small.

<sup>\*</sup>Department of Mathematics and Computer Science, University of Catania, email: stavrosavg.unict.it

<sup>&</sup>lt;sup>†</sup>Institut de Mathématiques de Bordeaux, University of Bordeaux, email: florian.bernard@math.u-bordeaux.fr

<sup>&</sup>lt;sup>‡</sup>Institut de Mathématiques de Bordeaux, University of Bordeaux, and Inria Bordeaux Sud-Ouest, email: angelo.iollo@math.u-bordeaux.fr

<sup>§</sup>Department of Mathematics and Computer Science, University of Catania, email: russo@dmi.unict.it

# Contents

| 1 | Introduction                                                                                                                                                                                                     |                                        |  |  |  |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|--|--|--|
| 2 | Euler equations                                                                                                                                                                                                  | 7                                      |  |  |  |
| 3 | One dimensional first-order schemes<br>3.1<br>Pressure splitting<br>3.2<br>Flux splitting                                                                                                                        | 8<br>9<br>10                           |  |  |  |
| 4 | Second order scheme<br>4.1<br>High order time discretization                                                                                                                                                     | 11<br>12                               |  |  |  |
| 5 | Numerical tests on 1D Euler equations<br>5.1<br>Sod shock tube<br>5.2<br>Lax shock tube<br>5.3<br>High Mach test case<br>5.4<br>Acoustic waves<br>5.4.1<br>Case<br><br>= 1/11<br>5.4.2<br>Case<br><br>= 1/1000 . | 14<br>14<br>15<br>16<br>18<br>19<br>20 |  |  |  |
| 6 | Application to nozzle flow<br>6.1<br>Pressure splitting<br>6.2<br>Flux splitting<br>6.3<br>Equilibria<br>6.4<br>Approximate well-balanced scheme                                                                 | 21<br>22<br>23<br>23<br>25             |  |  |  |
| 7 | Numerical tests for the nozzle flow<br>7.1<br>Subsonic flow<br>10−1<br>7.1.1<br>M<br>'<br>10−3<br>7.1.2<br>M<br>'<br>7.2<br>Transonic flow with a shock<br>7.3<br>Boundary conditions                            | 25<br>26<br>26<br>28<br>28<br>30       |  |  |  |
| 8 | 2D Model<br>8.1<br>Pressure spitting in 2D                                                                                                                                                                       | 31<br>31                               |  |  |  |
| 9 | Numerical tests on 2D Euler equations<br>9.1<br>Sod shock tube<br>9.2<br>Gresho vortex (Convergence test)<br>9.3<br>Vortex dipole                                                                                | 32<br>33<br>34<br>37                   |  |  |  |

| A |     | Well balanced methods                                           | 40 |
|---|-----|-----------------------------------------------------------------|----|
|   | A.1 | General finite volume scheme                                    | 40 |
|   | A.2 | Equilibrium and conservative variables                          | 40 |
|   | A.3 | Reconstruction of the field variables at cell boundaries        | 41 |
|   | A.4 | First order in space and time, perfectly well-balanced scheme . | 41 |
|   | A.5 | Application to nozzle flow                                      | 42 |
|   | A.6 | Approximate well-balanced schemes .                             | 43 |

# 1 Introduction

Numerical methods for the solution of hyperbolic systems of conservation laws has been a very active field of research in the last decades. Several very effective schemes are nowadays treated in textbooks which became a classic on the topic [26, 39, 15]. Because of the hyperbolic nature, all such systems develop waves that propagate at finite speeds. If one wants to accurately compute all the waves in a hyperbolic system, then one has to resolve all the space and time scales that characterize it. Most schemes devoted to the numerical solution of such systems are obtained by explicit time discretization, and the time step has to satisfy a stability condition, known as CFL condition, which states that the time step should be limited by the space step divided by the fastest wave speed (times a constant of order 1). Usually such a restriction is not a problem: because of the hyperbolic nature of the system, if the order of accuracy is the same in space and time, accuracy restriction and stability restrictions are almost the same, and the system is not stiff. There are, however, cases in which some of the waves are not particularly relevant and one is not interested in resolving them. Let us consider as a prototype model the classical Euler equations of compressible gas dynamics. In the low Mach number regimes, it may happen that the acoustic waves carry a negligible amount of energy, and one is mainly interested in accurately capturing the motion of the fluid. In such a case the system becomes stiff: classical CFL condition on the time step is determined by the acoustic waves which have a negligible influence on the solution, but which deeply affect the efficiency of the method itself.

Another difficulty arising with standard Godunov-type schemes for low-Mach flows is that the amount of numerical viscosity on the slow waves introduced by upwind-type discretization of the system would heavily degrade the accuracy. An account of the latter effect is analyzed in [14], where the relevance of centering pressure gradients in the limit of small Mach number is emphasized.

In order to overcome the drawback of the stiffness, one has to resort to implicit strategies for time discretization, which avoid the acoustic CFL restriction and allow the use of a much large time step. Naive implementation of implicit schemes for the solution of the Euler equations presents however two kinds of problems. First, classical upwind discretization (say Godunov methods based on exact or approximate Riemann solvers) are highly nonlinear and very difficult to solve implicitly. Second, the implicit version of classical schemes may introduce an excessive numerical dissipation on the slow wave, resulting in loss of accuracy. Investigation of the effect on fully implicit schemes (and preconditioning techniques adopted to cure the large numerical diffusion) are discussed for example in [42] and in [27], both inspired by an early work of Turkel [40]. In both cases, a modification to the absolute value of the Roe matrix is proposed by a suitable preconditioner that avoids excessive numerical diffusion of upwind-type discretization at very low Mach.

Several techniques have been devised to treat problems in the low Mach number regimes, that alleviate both drawbacks, see for example [24]. However, some of such techniques have been explicitly designed to treat low Mach number regimes, and are based on low Mach number asymptotics ([22], [23]). There are cases in which the Mach number can change by several orders of magnitude. The biggest challenges come from gas dynamic problems in astrophysics, where the range of scales of virtually all parameters vary over many orders of magnitude. An adaptive low Mach number scheme, based on a non conservative formulation, has been developed with the purpose of tackling complex gas dynamics problems in astrophysics (see [31] and references therein). When Mach number is very low the flow does not develop shock discontinuities, and the conservation form of the schemes is not mandatory. When Mach number is not small, then shock discontinuities may form. In such a cases it is necessary to resort to conservative schemes (see for example [27] for other astrophysical applications).

Some hyperbolic systems other than gas dynamics may be affected by the stiffness due to a large range of wave speed. In magneto-hydrodynamics, for example, fast magneto acoustic waves may be much faster than Alfv´en waves, and in case they carry very little energy, they do not need to be resolved. A pioneering paper in this direction was written by Harned and Kerner [18], who proposed a semi-implicit method for compressible MHD, which was able to filter out fast magneto-acoustic waves, so that the restriction on the time step was due to the much slower Alfv´en waves.

Other physical systems, still in the context of gas dynamics, are affected by drastic changes of the sound speed. Such large variations may be due to geometrical effects, as for example in the case of the nozzle flow (see Section 6) or to heterogeneity of the media. Air-water systems, for example, are characterized by density ratio of three orders of magnitude, while the ratio of sound speed is about five. Waves in heterogeneous solid materials may travel at very different speeds, depending on the local stiffness of the medium. The motivation for the construction of effective all Mach number solver is twofold: on one hand it is relevant to accurately simulate waves in heterogeneous materials without small time step restriction suffered by explicit schemes, on the other hand such simulations can be adopted as a tool to validate homogenized models, which at a more macroscopic scale can be described as a homogeneous medium with different mechanical properties. For example, in air-water flows, for a range of values of the void fraction, the measured sound speed is lower than both water and air sound speed [11].

Motivated by the above arguments, several researchers have devoted a lot of effort in the development of all Mach number solvers for gas dynamics. An early all Mach number scheme has been described in [41]. The method is based on a MAC-type staggered discretization in space. A conservative scheme is stabilized by a pressure-correction technique. The method is applied to several one and two dimensional problems, although no numerical convergence studies are reported.

Another attempt in this direction is presented in [25], where the authors adopt a pressure stabilization technique to be able to go beyond the classical CFL restriction. The technique works well for moderate Mach number, but is not specifically designed to deal with very small Mach numbers.

A different stabilization technique has been proposed by Kadioglu and collaborators [21]. Here the authors present a stabilization method based on an implicit step (on the primitive variables) which is performed after a second order explicit prediction. The technique is successfully applied to single fluid as well as multi-fluid test cases. Related methods by the same author have been developed in [19], where an IMEX strategy has been adopted to solve hydrodynamical problems with non linear heat conduction, and in [20], where the implicit-explicit schemes in time have been used in the context of radiation hydrodynamics.

In an impressive sequence of papers and conference proceedings, [9, 5, 7, 8, 6], F.Coquel and collaborators proposed a semi-implicit strategy, coupled with a multi resolution approach, for the numerical solution of hyperbolic systems of conservation laws with very well separated wave propagation speeds. In particular, they considered application to fluid mixtures, in which the propagation speed of acoustic waves, often carrying a negligible amount of energy, is much larger than the speed of the material wave traveling at the fluid velocity. The basic framework is set in [9]. The method is first explained in the context of linear hyperbolic systems. The eigenvalues are sorted and it is assumed that there is a clear separation between slow and fast waves. The Jacobian matrix is split into a slow and fast component, using the characteristic decomposition. The flux at cell boundaries is consequently split into a slow and fast term. The fast term is treated implicitly, while the slow one is treated explicitly. The approach is then generalized to the quasilinear case, making use of Roe-type approximation of flux difference. This allows to construct a simple semi-implicit formulation by leaving the Roe matrix of the fast waves at the previous time step, while only the field is computed at the new time step, leading to a linearly implicit scheme. The effectiveness of the approach is further improved by adopting spatial multi resolution: given a multi scale expansion of the numerical solution, the finest scale is maintained locally only where needed, while coarser scales are adaptively adopted in smoother regions, with a great savings in computational time. Different schemes, still adopting implicit-explicit time differentiation to filter out fast waves, are considered in [5], where a sort of arbitrary Lagrangian-Eulerian scheme is constructed: a fractional time step strategy is composed by an implicit Lagrangian step, which filters out acoustic waves, and an explicit Eulerian step, which takes into account the contribution of slow waves. The main application is

still on a model for the evolution of gas-oil mixture. In order to simplify the treatment of a general equation of state, a relaxation method is adopted (which of course satisfies the Chen-Levermore-Liu sub-charactertistic condition [4]). The problem of developing an adaptive (local) time step strategy is considered in the proceedings [7], and fully exploited in [8]. In [6], the authors further refine the technique, thus producing a positivity preserving, entropic semi-implicit scheme for Euler-like equations. The approach developed by Coquel and collaborators is certainly valuable, although it may be quite involved to be efficiently implemented for more complex, multidimensional situations.

A different approach has been adopted by Munz and collaborators, starting from the low Mach number asymptotic of Kleinerman and Majda. In [29], the authors develop a very effective semi-implicit method which can be viewed as a generalization of a compressible solver to weakly compressible flows. The method is based on the asymptotic behavior of the Euler equation for low Mach number. Two pressures are defined, a thermodynamic one, which is essentially constant in space, and a dynamic one, which accounts for fluid motion. The method is based on a discretization of the system written in primitive variables. The approach, designed for low Mach flow, cannot be directly used when compressive effects are more pronounced. In a subsequent paper [34], Park and Munz extend the method, still using the pressure as basic unknown in place of the energy, but now they adopt a conservative formulation, thus being able to capture shocks when the Mach number is not so small. Several space discretizations as well as time discretization strategies are discussed, which allow to obtain second order accuracy in space and time. In addition, the paper contains a nice overview of other works on low Mach number flow.

In [16] and in [13] the authors explore the construction of an all Mach-number finite volume scheme for the isentropic Euler and Navier-Stokes equations. In both cases, the approach consists in a sort of hyperbolic splitting, obtained by adding and subtracting a gradient-type term to the momentum equation. Such a term is an approximation of the pressure gradient, and is treated implicitly, while the (relatively small) difference with the physical pressure gradient is treated explicitly. The authors show the asymptotic preserving (AP) property of the schemes: when the Mach number approaches zero the schemes become a consistent and stable discretization of the incompressible Euler and Navier-Stokes equations. In a more recent paper, Cordier et al.[10] extend the technique to the full Euler and Navier-Stokes equations. In paper [12] a different approach has been adopted for the construction of asymptotic preserving schemes for the gas dynamics. The authors perform a gauge decomposition of the momentum density into a solenoidal and irrotational field. They show that this corresponds to a sort of micro-macro decomposition, in which the macroscopic variable describe the slow material wave, while the fast variable accounts for the fast acoustic waves. They apply their technique to isentropic and full Euler and Navier-Stokes, as well as to the isentropic Navier-Stokes-Poisson system.

A slightly different approach is adopted in [30], where the author propose methods based on the flux splitting: the flux is split in two terms, one of which is treated explicitly and the other implicitly.

In most schemes, ad hoc procedures have been adopted to pass from first to second order accuracy in time.

Objective of the present work is to propose and compare some simple schemes for the numerical solution of Euler equation in gas dynamics that are (linearly) implicit in the acoustic waves, eliminating the acoustic CFL restriction, and that do not introduce excessive numerical dissipation at low Mach number, thus providing accurate solution in such regimes. A further requirement is that the schemes should perform reasonably well when the Mach number are not too small. In view of the more demanding applications, simplicity is one of the key features of the proposed schemes.

The outcome of this work will be used to select a candidate scheme which will be generalized to solve more challenging problems, such as multi-material ones (possibly with a parallel architecture and with the use of adaptive grids) described before.

In addition, at variance with previous works on the topic, a systematic technique is adopted for the construction of high order schemes in time. Although the methods in the paper are limited to second order accuracy, the technique can be adopted for the construction of more accurate schemes.

The plan of the paper is the following: next section introduces the problem and the key ideas behind the schemes. Then we present first order schemes, in which we discretize time by implicit-explicit Euler scheme, and space by either upwind or central difference, according to the term. Higher order discretization in space and time are reported in Section 4. In the last section, numerical results are presented on one dimensional test cases. In particular, the nozzle flow is investigated for a wide range of Mach numbers.

# 2 Euler equations

Let us consider the Euler equations of compressible gas dynamics in one space dimension:

$$\begin{cases} \frac{\partial \rho}{\partial t} + \frac{\partial m}{\partial x} = 0, \\ \frac{\partial m}{\partial t} + \frac{\partial}{\partial x} (mu + p) = 0, \\ \frac{\partial E}{\partial t} + \frac{\partial \rho hu}{\partial x} = 0, \end{cases}$$
 (1)

where ρ is the density, u the velocity, m = ρu the momentum, p the pressure, E the total energy and h = (E + p)/ρ the total enthalpy. The system is closed with the equation of state for a perfect gas

$$p = (\gamma - 1)(E - \frac{\rho u^2}{2}). \tag{2}$$

The idea of the schemes is the following. We should try to identify the terms that need to be treated implicitly, and the ones that can be dealt with explicitly. Roughly speaking, we would like to treat acoustic waves implicitly, while material waves should be treated explicitly. Another difference concerns space discretization. The terms that describe material waves in the limit of small Mach number will be discretized by some upwind discretization in space, while the terms that are responsible of acoustic waves will be discretized by central difference.

System (1) has the structure of a quasilinear hyperbolic system of conservation laws. Let us write it in the form:

$$\frac{\partial W}{\partial t} = -\frac{\partial \mathcal{F}(W)}{\partial x},\tag{3}$$

where W = (ρ, m, E) <sup>&</sup>gt;. Assuming we approximate the spatial derivatives that appear in the system, by suitable discrete operators, we can formally rewrite system (3) in the form

$$\begin{cases}
\frac{dU}{dt}(t) = \mathcal{H}(U_E(t), U_I(t)), & \forall t \ge t_0, \\
U(t_0) = U_0.
\end{cases}$$
(G)

Here U is a discrete approximation of W that can be decomposed in a non stiff part (for example corresponding to the material wave) and a stiff part (corresponding to the acoustic waves). It appears therefore natural to treat the first variable explicitly (UE), and the second one implicitly (U<sup>I</sup> ). H denotes some approximation of −∂F(W)/∂x obtained by a spatial discretization (that we shall specify later).

# 3 One dimensional first-order schemes

We present different schemes based on two approaches. The first family of schemes will be called "Pressure splitting" and is based on an explicit treatment of the convective terms and an implicit handling of the pressure terms. The second family,called "Flux splitting", is based on a splitting of the fluxes along the characteristics. The flux corresponding to the material wave is treated explicitly while the other is treated implicitly.

In the following, we denote by Dˆ <sup>x</sup> the flux derivative that will be discretized as the difference of the numerical fluxes between i + 1/2 and i − 1/2 and D<sup>x</sup> the flux derivatives discretized with a centered scheme.

Two approaches are used to compute the numerical fluxes in Dˆ <sup>x</sup>. The first one is an upwind discretization based on the sign of the gas velocity, i.e.

$$\Phi_{i+1/2} = \begin{cases} \frac{u_{i+1/2} + |u_{i+1/2}|}{2|u_{i+1/2}|} \mathcal{F}(U_i) + \frac{u_{i+1/2} - |u_{i+1/2}|}{2|u_{i+1/2}|} \mathcal{F}(U_{i+1}) & \text{if } u_{i+1/2} \neq 0, \\ 0 & \text{otherwise,} \end{cases}$$
(4)

where ui+1/<sup>2</sup> is the velocity at the cell interface i + 1/2.

The second approach uses the local Lax-Friedrichs fluxes.

$$\Psi_{i+1/2} = \frac{\mathcal{F}(U_i) + \mathcal{F}(U_{i+1})}{2} - \frac{\alpha_{j+1/2}}{2} (U_{i+1} - U_i), \tag{5}$$

where αj+1/<sup>2</sup> has to be optimally chosen. In classical explicit schemes it is a bound on the maximum wave speeds, and is given by

$$\alpha_{j+1/2} = \max(|u_j| + a_j, |u_{j+1}| + a_{j+1}),$$

where a <sup>2</sup> = γp/ρ denotes the square of the sound speed. In our case, since the acoustic waves are treated implicitly, we use α proportional to the material speed. We expect that for very low Mach number, α ≈ |u| should be sufficient, while for Mach number larger than one, the speed of sound is bounded by the fluid speed. For this reason, we choose

$$\alpha_{j+1/2} = \max(|u_j|, |u_{j+1}|), \tag{6}$$

We shall denote by ∆x the space step discretization. Then the upwind discrete derivative is given by

$$\hat{D}_x(F_i^j) = \frac{F_{i+1/2}^j - F_{i-1/2}^j}{\Delta x},\tag{7}$$

F being either Φ or Ψ, and superscript j denotes the j-th component of the flux vector.

On the other hand, D<sup>x</sup> will be computed with a centred scheme:

$$D_x(F_i^j) = \frac{\mathcal{F}^j(U_{i+1/2}) - \mathcal{F}^j(U_{i-1/2})}{\Delta x}.$$
 (8)

In practice, F j (Ui−1/2) will be approximated by (F j (Ui−1) + F j (Ui))/2, therefore Eq.(8) becomes the classical centered approximation of the first derivative. However, we prefer to use definition (8) which provides a more compact discrete second-order operator, such as a second derivative.

# 3.1 Pressure splitting

We first present the schemes derived from an implicit treatment of the pressure terms.

We discretize the system of equations (1) between a time t <sup>n</sup> and a time t n+1

$$\begin{cases}
\rho^{n+1} = \rho^n - \Delta t \hat{D}_x(m^n), \\
m^{n+1} = m^n - \Delta t \hat{D}_x(m^n u^n) - \Delta t D_x(p^{n+1}), \\
E^{n+1} = E^n - \Delta t D_x(h^n m^{n+1}).
\end{cases} \tag{9}$$

One can note that in this case, ρ <sup>n</sup>+1 is computed explicitly. p <sup>n</sup>+1 and E <sup>n</sup>+1 are linearly linked through the equation of state so one can now solve the problem on the energy or on the pressure. It appears more natural to solve the equation on the energy, which is the conservative variable. Indeed we explored the use of schemes obtained by solving the linear system on the pressure, and we found them to be less accurate and more oscillatory.

Substituting p <sup>n</sup>+1 in the equation of the momentum by (γ − 1)(E <sup>n</sup>+1 − m<sup>n</sup>u <sup>n</sup>/2) and treating the new derivative on m<sup>n</sup>u <sup>n</sup> upwind we obtain

$$m^{n+1} = m^n - \frac{3-\gamma}{2} \Delta t \hat{D}_x(m^n u^n) - (\gamma - 1) \Delta t D_x(E^{n+1}).$$
 (10)

Plugging this expression in the equation on the energy, we obtain

$$E^{n+1} = E^* + (\gamma - 1)\Delta t^2 D_x(h^n D_x(E^{n+1})), \tag{11}$$

with 
$$E^* = E^n - \Delta t \hat{D}_x(h^n m^*)$$
 and  $m^* = m^n - \frac{3 - \gamma}{2} \Delta t \hat{D}_x(m^n u^n)$ .

E <sup>n</sup>+1 can now be computed by solving a tridiagonal system (11) and plugged into the momentum equation to find m<sup>n</sup>+1 .

### 3.2 Flux splitting

The fluxes are decomposed in two parts according to the characteristics. The first part corresponding to the material flux (characteristic velocity u) is treated explicitly with fully upwind or local Lax-Friedrichs fluxes. The second part, corresponding to the acoustic fluxes is computed with centred derivatives. Such a splitting is obtained as follows. First, observe that in the case of the Euler equation for a polytropic gas one has

$$\mathcal{F}(U) = \mathcal{A}U,$$

where A(U) = ∇<sup>U</sup> F(U) denotes the Jacobian matrix of the system. Let us denote by Λ = diag(u − a, u, u + a) the diagonal eigenvalue matrix, and let Q denote the matrix containing its right eigenvectors. Then we can write:

$$\mathcal{A} = Q\Lambda Q^{-1}.$$

Let us partition the Jacobian matrix as A = A<sup>A</sup> + A<sup>F</sup> , where A<sup>A</sup> = QΛAQ<sup>−</sup><sup>1</sup> is the acoustic component and A<sup>F</sup> = QΛFQ<sup>−</sup><sup>1</sup> is the fluid component, and Λ<sup>F</sup> = diag(0, u, 0), and Λ<sup>F</sup> = diag(u − a, 0, u + a). Then we partition the flux as

$$\mathcal{F} = \mathcal{F}_F + \mathcal{F}_A$$
.

Straightforward calculation shows that

$$\mathcal{F}_{F} = \frac{\gamma - 1}{\gamma} \begin{pmatrix} m \\ mu \\ mu^{2} \end{pmatrix}, \ \mathcal{F}_{A} = \frac{1}{\gamma} \begin{pmatrix} m \\ mu \\ mu^{2} \end{pmatrix} + \begin{pmatrix} 0 \\ p \\ \frac{\gamma}{\gamma - 1} pu \end{pmatrix}. \tag{12}$$

Using this spitting, first order implicit-explicit time discretization reads

$$\begin{cases}
\rho^{n+1} = \rho^{n} - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_{x}(m^{n}) - \frac{\Delta t}{\gamma} D_{x}(m^{n+1}), \\
m^{n+1} = m^{n} - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_{x}(m^{n}u^{n}) - \frac{\Delta t}{\gamma} D_{x}(m^{n}u^{n}) - \Delta t D_{x}(p^{n+1}), \\
E^{n+1} = E^{n} - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_{x}(\rho^{n}(u^{n})^{3}) - \frac{\Delta t}{\gamma} D_{x}(\rho^{n}(u^{n})^{3}) - \Delta t \frac{\gamma}{\gamma - 1} D_{x}(\frac{p^{n}}{\rho^{n}} m^{n+1}).
\end{cases} (13)$$

Let us substitute  $p^{n+1}$  by the equation of state (with only E at time  $t^{n+1}$  in the equation of m)

$$m^{n+1} = m^n - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_x(m^n u^n) + \frac{\gamma^2 - \gamma - 2}{2\gamma} \Delta t D_x(m^n u^n) - (\gamma - 1) \Delta t D_x(E^{n+1}),$$
  
=  $m^* - (\gamma - 1) \Delta t D_x(E^{n+1})$ 

with 
$$m^* = m^n - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_x(m^n u^n) + \frac{\gamma^2 - \gamma - 2}{2\gamma} \Delta t D_x(m^n u^n)$$
.

Plugging this in the expression of the energy, one gets

$$E^{n+1} = E^* + \Delta t^2 \gamma D_x(\frac{p^n}{\rho^n} D_x(E^{n+1})), \tag{14}$$

where

$$E^* = E^n - \frac{\gamma - 1}{\gamma} \Delta t \hat{D}_x(\rho^n(u^n)^3) - \frac{\Delta t}{\gamma} D_x(\rho^n(u^n)^3) - \Delta t \frac{\gamma}{\gamma - 1} D_x(\frac{p^n}{\rho^n} m^*).$$

One can note that if all the explicit derivatives are treated with fully upwind fluxes or local Lax-Friedrichs fluxes, we obtain the pressure splitting scheme of Section 3.1.

#### 4 Second order scheme

High order shock-capturing finite volume schemes are usually obtained by adopting some high order non-oscillatory reconstruction in space, such as ENO or WENO [38]. In this paper we limit ourselves to first and second order schemes. Second-order schemes in space are obtained by using a piecewise conservative linear reconstruction in order to evaluate all *upwind* derivatives. First and second order derivative approximated by classical three point central schemes are automatically second order accurate. In order to achieve second order in the upwind cases we use:

$$\Psi_{i+1/2} = \frac{\mathcal{F}_{i+1/2}^{-} + \mathcal{F}_{i+1/2}^{+}}{2} - \frac{\alpha_{j+1/2}}{2} (v_{i+1/2}^{+} - v_{i+1/2}^{-}), \tag{15}$$

where  $\mathcal{F}_{i+1/2}^{\pm} = f(v_{i+1/2}^{\pm})$ . For each conservative field v, we use the following reconstruction in cell j:

$$v_{j+1/2}^n(x)^{\mp} = \bar{v}_j^n \pm v_j'(x - x_j),$$

where the approximation of the derivative is obtained by

$$v_j' = \operatorname{MimMod}\left(\theta \frac{\overline{v}_{j+1} - \overline{v}_j}{\Delta x}, \frac{\overline{v}_{j+1} - \overline{v}_{j-1}}{2\Delta x}, \theta \frac{\overline{v}_j - \overline{v}_{j-1}}{\Delta x}\right),\,$$

and the function MinMod is defined as

$$\operatorname{MimMod}(a_1, \dots, a_m) = \begin{cases} \operatorname{sign}(a_1) \min_{i=1}^m |a_i| & \text{if } \{a_i\} \text{ have all the same sign} \\ 0 & \text{otherwise} \end{cases}$$

#### High order time discretization 4.1

The method proposed in the previous section is only first order accurate in time (and second order accurate in space, thanks to the piecewise linear reconstructions).

High order in time is obtained by suitable use of implicit-explicit Runge-Kutta schemes. Here we use the technique described in detail in [3]. Once the system is discretized in space, we can write it as a large set of ordinary differential equations for a vector  $U(t) \in \mathbb{R}^{Jm}$ , where J is the number of space cells, and m denotes the number of equations of the system (m = 3 for the Euler equations in one space)dimension).

For such a purpose we use Implicit-Explicit Runge-Kutta schemes [1], [33]. An s-stage IMEX-RK scheme is represented by the double Butcher tableaux.

$$\begin{array}{c|c}
\hat{c} & \hat{A} \\
\hline
 & \hat{b}^{\top} \\
\end{array}$$

$$\begin{array}{c|c}
c & A \\
\hline
 & b^{\top}
\end{array},$$
(16)

where  $\hat{A}, A \in \mathcal{M}_{s,s}(\mathbb{R}), \hat{c}, c, \hat{b}, b \in \mathbb{R}^s$ . The coefficients  $\hat{c}$  and c, used in the case of non-autonomous systems, are related to matrices  $\hat{A}$  and A by

$$\hat{c}_i = \sum_j \hat{a}_{ij}, \quad c_i = \sum_j a_{ij}.$$

Matrices  $\hat{A}$  and A are lower triangular matrix, and in particular  $\hat{A}$  will have zero elements on the diagonal. Here we adopt schemes with b = b, and in particular all second order computations are performed with the specific scheme:

$$\begin{array}{c|ccccc}
0 & 0 & 0 & \beta & \beta & 0 \\
\hat{c} & \hat{c} & 0 & 1 & 1 - \beta & \beta \\
\hline
& 1 - \beta & \beta & 1 - \beta & \beta
\end{array}$$
(17)

where β is chosen as the smallest root of the polynomial β <sup>2</sup> − 2β + 1/2 = 0, i.e. β = 1 − 1/ √ 2 and ˆc = 1/(2β). This scheme is the combination of a second order Runge-Kutta method (explicit part) and an L-stable second order singly diagonal stiffly accurate RK method (SDIRK) in the implicit part, and we call it LSDIRK2 (see for instance [17]).

The scheme works as follows: given U n , the numerical solution at the next time step, U <sup>n</sup>+1, is computed as follows:

#### • Stage values:

for i = 1 to s compute:

$$U_E^{(i)} = U^n + \Delta t \sum_{j=1}^{i-1} \hat{a}_{ij} \mathcal{H}(U_E^{(j)}, U_I^{(j)})$$
(18)

$$U_I^{(i)} = U_*^{(i)} + \Delta t a_{ii} \mathcal{H}(U_E^{(i)}, U_I^{(i)})$$
(19)

where

$$U_*^{(i)} = U^n + \Delta t \sum_{j=1}^{i-1} a_{ij} \mathcal{H}(U_E^{(j)}, U_I^{(j)})$$

• Numerical solution:

$$U^{n+1} = U^n + \Delta t \sum_{j=1}^{s} b_i \mathcal{H}(U_E^{(i)}, U_I^{(i)})$$

Notice that the only step requiring an implicit evaluation is the step in Eq. (19), that computes U (i) I from aii∆t, U (i) E and U (i) <sup>∗</sup> , and which is equivalent to an implicit Euler step.

In practice the algorithm to reach higher order can be easily constructed as follows. Let us define

$$U_I = \mathcal{S}(U_*, U_E, \Delta t)$$

the function that gives the solution to the problem

$$U_I = U_* + \Delta t \mathcal{H}(U_E, U_I);$$

then, the method corresponding to the imex scheme (17) can be effectively implemented as

$$U_I^{(1)} = \mathcal{S}(U^n, U^n, \beta \Delta t)$$

$$U_E^{(2)} = \left(1 - \frac{\hat{c}}{\beta}\right) U^n + \frac{\hat{c}}{\beta} U_I^{(1)}$$

$$U_*^{(2)} = \frac{2\beta - 1}{\beta} U^n + \frac{1 - \beta}{\beta} U_I^{(1)}$$

$$U_I^{(2)} = \mathcal{S}(U_*^{(2)}, U_E^{(2)}, \beta \Delta t)$$

At the end, the numerical solution is computed as U <sup>n</sup>+1 = U (2) I .

Higher order discretization in time can be obtained by the same approach, using higher order IMEX schemes. However, in order to obtain the same accuracy in space, high order space discretization should be used. This can be obtained by high order non-oscillatory reconstructions for the computation of the numerical flux, and with high order central space discretization for the stiff term. Such high order schemes are however beyond the scope of the present paper.

# 5 Numerical tests on 1D Euler equations

In this section we present classical test cases on 1D Euler equation to show the efficiency of the different schemes in several Mach regimes.

First we consider the classical Sod and Lax tests, in order to verify the shock capturing capability of the schemes for intermediate Mach numbers. Here the objective is not to be able to use CFL numbers larger than one, but rather to check the robustness of the new schemes with classical shock tube test problems.

Later we show the results for colliding acoustic waves at small Mach numbers, in order to test the capability of the schemes to filter out acoustic waves. In particular, we shall show that, for low Mach number, the proposed schemes are stable under the much less restrictive material CFL condition, which means basically that (∆t max |u|)/∆x < C, with C a constant of order 1. Unless when otherwise stated, by CFL we denote the classical Courant number, namely CFL = (∆t max(|u| + a))/∆x, so that we shall see that for small Mach number we can get values of CFL well above one.

### 5.1 Sod shock tube

In this test Mach numbers are of order 1. The domain is [0,1] and is discretized with 100 cells. The discontinuity is initially at x = 0.5 and the initial condition on the left and on the right (with index L and R respectively) are:

$$u_L = 0$$
  $u_R = 0$   
 $\rho_L = 1$  and  $\rho_R = 0.125$   
 $p_L = 1$   $p_R = 0.1$ 

Five schemes are compared: two explicit schemes, with numerical flux, respectively, Osher [32] and local Lax-Friedrichs (continuous and dashed line), and three semi-implicit schemes, namely P-split-up, P-split-LF (for pressure splitting with upwind fluxes and local Lax-Friedrichs fluxes respectively), F-split-LF (for flux splitting with local Lax-Friedrichs fluxes). Density profiles are shown for the first order schemes in Figure 1a and for the second order in Figure 1b. The same for pressure profiles in Figure 2a and for the second order in Figure 2b.

Note that the results of the explicit and P-split schemes that use local Lax-Friedrichs flux are almost identical. Scheme F-split-LF shows a spurious overshoot, while F-split-up is unstable. This is not surprising, since the scheme contain an explicit centred derivative in the convective part.

![](_page_15_Figure_1.jpeg)

Figure 1: Sod shock tube. Density solution with the different methods. CFL=0.5, 100 grid points.

![](_page_15_Figure_3.jpeg)

Figure 2: Sod shock tube. Pressure solution with the different methods. CFL=0.5, 100 grid points.

### 5.2 Lax shock tube

The Lax shock tube test case is similar to the previous one, but with a stronger shock. The domain is [0,1] and is discretized with 200 grid points to correctly

capture the phenomena. The discontinuity is initially at x = 0.5 and the initial conditions on the left and on the right (with index L and R respectively) are:

$$\rho_L = 0.445$$
 $(\rho u)_L = 0.311$ 
and
 $\rho_R = 0.5$ 
 $(\rho u)_R = 0$ 
 $E_L = 8.928$ 
 $E_R = 1.4275$ 

![](_page_16_Figure_2.jpeg)

Figure 3: Lax shock tube. First order schemes. Density profiles obtained with the various methods. CFL=0.5, 200 grid points.

Figure 3 reports the density result of the first order schemes. Explicit scheme with Osher flux and P-split-upwind give similar results. Likewise, Explicit-LF and P-split-LF produce comparable answers.

Figure 4 shows the density results on the Lax test problem obtained with the various second-order schemes considered in the paper. Once again, the zoom in Fig. 5 shows that explicit and P-split schemes based on the local Lax-Friedrics fluxes give similar results. Explicit-Osher and P-split-up give comparable results, showing the good shock capability of the semi-implicit schemes. An overshoot is observed in the F-split scheme. The same is observed for pressure profiles in Figure 6a and for the second order in Figure 6b. If the discretization increases, e.g. 400 grid points, the amplitude of the oscillation remains basically constant and it occurs on the same number of grid points.

# 5.3 High Mach test case

This extreme case is studied to test the robustness of the proposed schemes. Among the semi-implicit schemes only the P split LF is stable. Without surprise, the ones

![](_page_17_Figure_0.jpeg)

![](_page_17_Figure_1.jpeg)

Figure 4: Lax shock tube. Second order schemes. Density profiles obtained with various methods. CFL=0.5, 200 grid points.

Figure 5: Zoom of the previous figure.

![](_page_17_Figure_4.jpeg)

![](_page_17_Figure_5.jpeg)

Figure 6: Lax shock tube. Pressure profiles obtained with various methods. CFL=0.5, 200 grid points.

specifically conceived for low-Mach regimes are unstable. Both explicit schemes (Osher and L-X) are stable. See figures 7a and 7b for first and second-order results. The initial conditions are those presented in [21]:

$$\rho_L = 10 \nu_L = 2000 and  $u_R = 0 
p_L = 500$ 
 $\rho_R = 20 \nu_R = 0 
p_R = 500$$$

![](_page_18_Figure_0.jpeg)

Figure 7: High Mach shock. First order and second order schemes. Density profiles obtained with various methods. CFL=0.5, 200 grid points.

![](_page_18_Figure_2.jpeg)

Figure 8: High Mach shock. First order and second order schemes. Density profiles obtained with various methods. CFL=0.5, 200 grid points.

Note how the P-split LF scheme has better resolution than the explicit LF scheme.

#### 5.4 Acoustic waves

This test is taken from [10], and is adopted to check the ability of the schemes to filter out acoustic waves in low Mach regimes, when adopting a material rather than an acoustic CFL restriction.

In paper [10], the equations are rescaled, and a parameter , related to the Mach number, appears explicitly in the equations. Such a parameter is the ratio between a typical fluid velocity u<sup>0</sup> and the thermal speed p p0/ρ0:

$$\epsilon = \frac{u_0}{\sqrt{p_0/\rho_0}} = \sqrt{\gamma}M$$

where M is the Mach number. In order to have an immediate comparison with the results from the literature, in this section we shall use in place of M in order to identify the various regimes.

The computational domain goes from −L/2 = −1/ to L/2 = 1/ and the initial condition is given as follows:

$$\rho_0 = 0.955; \rho_1 = 2 \nu_0 = 2\sqrt{\gamma} 
p_0 = 1; p_1 = 2 
\rho = \rho_0 + 0.5\rho_1\epsilon (1 - \cos\frac{2\pi x}{L}) 
p = p_0 + 0.5p_1\epsilon (1 - \cos\frac{2\pi x}{L})$$

Two test cases are considered, with = 1/11 and = 10<sup>−</sup><sup>3</sup> , respectively.

#### 5.4.1 Case = 1/11

Figures (9a–10a) show the results obtained with the various schemes when Mach number is <sup>M</sup> = 1/(11√γ).

The figures show the results obtained with a small time step, using CFL = 0.5 for all schemes. The P-split up schemes tends to oscillate, even with a first order schemes, and is not displayed on these figures. For first order schemes, P-split-LF is less accurate than the other schemes, because it uses local Lax-Friedrichs fluxe which is more dissipative. However, for the same CFL value, the second order implicit schemes give results which are comparable with those obtained by the explicit schemes, showing the ability of the semi-implicit schemes to correctly capture the acoustic waves if a suitably small CFL number is adopted.

Figure 10a and 10b show the pressure profiles on the same test case but with a CFL equal to 3 (corresponding to a material CFL of 0.44). Here, we compare the results of the P-split and F-split schemes with upwind fluxes (respectively P split up and F split up), the F-split scheme with the local Lax-Friedrichs fluxes (F split LF) with the explicit solution computed with the local Lax-Friedrichs fluxes with CFL equal to 0.5. At first order, the profiles are distinct while with second order schemes, the solution is well captured.

![](_page_20_Figure_0.jpeg)

Figure 9: Pressure at t=1.63, CFL=0.5, 400 grid points, = 1/11.

![](_page_20_Figure_2.jpeg)

Figure 10: Pressure at t=1.63, CFL=3 (material CFL=0.44), 400 grid points, = 1/11.

#### 5.4.2 Case = 1/1000

The same test is performed with = 10<sup>−</sup><sup>3</sup> . We first compare the results to the explicit solution for a CFL of 0.5. For all schemes, the solution is well captured, especially at second order. With an explicit CFL number, the acoustic waves are not filtered.

For a high CFL number (200, corresponding to a material CFL of 0.39), the acoustic waves are completely filtered out as we can see on Figure 12a with first order schemes but the solution remains stable. At second order, the acoustic waves are not completely filtered out since the schemes are more accurate. However, most of the acoustic signal is lost.

![](_page_21_Figure_0.jpeg)

Figure 11: Pressure at t=1.63, CFL=0.5, 400 grid points,  $\epsilon = 10^{-3}$ .

![](_page_21_Figure_2.jpeg)

Figure 12: Pressure ar t=1.63, CFL=200 (material CFL=0.39, 400 grid points,  $\epsilon = 10^{-3}$ .

This test case has shown the ability of the schemes to capture correctly the acoustic waves with the acoustic CFL number. Moreover, as expected, when adopting a material CFL rather than an acoustic CFL, the schemes remain stable, and the acoustic waves are filtered out.

# 6 Application to nozzle flow

We now consider a Laval nozzle through the quasi-1D Euler equations [37]. The system is similar to the classical 1D Euler equations with the addition of a source term, which accounts for the variable cross section of the nozzle.

The equations are:

$$\begin{cases}
\frac{\partial \rho}{\partial t} + \frac{\partial m}{\partial x} = -\frac{m}{A} \frac{\partial A}{\partial x}, \\
\frac{\partial m}{\partial t} + \frac{\partial}{\partial x} (mu + p) = -\frac{mu}{A} \frac{\partial A}{\partial x}, \\
\frac{\partial E}{\partial t} + \frac{\partial \rho hu}{\partial x} = -\frac{\rho hu}{A} \frac{\partial A}{\partial x}.
\end{cases} (20)$$

The same schemes developed in Sections 3 and 4 can be applied on this model, with slight generalization, because of the geometric source term.

#### 6.1 Pressure splitting

First order in time implicit-explicit schemes can be written as

$$\rho^{n+1} = \rho^n - \Delta t \widehat{D}_x(m)^n - \Delta t \frac{A_x}{A} m^{n+1}, \tag{21}$$

$$m^{n+1} = m^n - \Delta t \widehat{D}_x(mu)^n - \Delta t D_x p^{n+1} - \Delta t \frac{A_x}{A} u^n m^{n+1},$$
 (22)

$$E^{n+1} = E^n - \Delta t D_x(h^n m^{n+1}) - \Delta t \frac{A_x}{A} h^n m^{n+1}.$$
 (23)

Here  $\widehat{D}$  denotes the discrete derivative of the fluxes, and is treated explicitly and upwind. Expressing  $p^{n+1}$  with the equation of state (2),  $m^{n+1}$  can be recast as:

$$m^{n+1} = m^* - \frac{\Delta t(\gamma - 1)}{1 + \Delta t u^n A_x / A} D_x E^{n+1}, \tag{24}$$

with

$$m^* = \frac{1}{1 + \Delta t u^n A_x / A} \left( m^n - \Delta t \frac{3 - \gamma}{2} \widehat{D}_x(m^n u^n) \right)$$

Then, plugging this expression in the expression of the energy:

$$E^{n+1} = E^{n} - \Delta t (\widehat{D}_{x}(h^{n}m^{*}) + \frac{A_{x}}{A}h^{n}m^{*}) + \Delta t^{2}(\gamma - 1)D_{x} \left(\frac{h^{n}}{1 + \Delta t u^{n}A_{x}/A}D_{x}E^{n+1}\right) + \frac{\Delta t^{2}}{1 + \Delta t u^{n}A_{x}/A}\frac{A_{x}}{A}h^{n}D_{x}E^{n+1}$$
(25)

#### 6.2 Flux splitting

In this case the scheme takes the form

$$\rho^{n+1} = \rho^n - \Delta t \frac{\gamma - 1}{\gamma} \widehat{D}_x(m)^n - \frac{\Delta t}{\gamma} D_x m^{n+1} - \Delta t \frac{A_x}{A} m^{n+1}, \tag{26}$$

$$m^{n+1} = m^n - \Delta t \frac{\gamma - 1}{\gamma} \widehat{D}_x(mu)^n - \frac{\Delta t}{\gamma} D_x(mu)^n - \Delta t D_x p^{n+1} - \Delta t \frac{A_x}{A} u^n m^{n+1},$$
(27)

$$E^{n+1} = E^n - \Delta t \frac{\gamma - 1}{2\gamma} \widehat{D}_x (mu^3)^n - \frac{\Delta t}{2\gamma} D_x (mu^3)^n$$
(28)

$$-\Delta t \frac{\gamma}{\gamma - 1} D_x \left( \frac{p^n}{\rho^n} m^{n+1} \right) - \Delta t \frac{A_x}{A} h^n m^{n+1}. \tag{29}$$

As usual,  $\widehat{D}$  denotes the flux difference that are treated explicitly and upwind.

Let us denote by \* intermediate quantities computed with explicit part. Expressing  $p^{n+1}$  in the momentum equation with the equation of state (2) we get

$$\rho^* = \rho^n - \Delta t \frac{\gamma - 1}{\gamma} \widehat{D}_x(m^n),$$

$$m^* = \frac{1}{1 + \Delta t u^n A_x / A} (m^n - \Delta t \frac{\gamma - 1}{\gamma} \widehat{D}_x(mu)^n + \Delta t \frac{\gamma^2 - \gamma - 2}{2\gamma} D_x(mu)^n),$$

$$E^* = E^n - \Delta t \frac{\gamma - 1}{2\gamma} \widehat{D}_x(mu^3)^n - \frac{\Delta t}{2\gamma} D_x(mu^3)^n,$$

and we have

$$m^{n+1} = m^* - \frac{(\gamma - 1)\Delta t}{1 + \Delta t u^n A_x / A} D_x E^{n+1}.$$
 (30)

Plugging this expression of  $m^{n+1}$  in the equation of energy, we obtain

$$E^{n+1} = E^{**} + \Delta t^2 \gamma D_x \left( \frac{p}{\rho} \frac{1}{1 + \Delta t \frac{A_x}{A} u^n} D_x E^{n+1} \right) + \frac{\Delta t^2 (\gamma - 1)}{1 + \Delta t \frac{A_x}{A} u^n} \frac{A_x}{A} h^n D_x E^{n+1}, \quad (31)$$

where 
$$E^{**} = E^* - \frac{\Delta t \gamma}{\gamma - 1} D_x \left( \frac{p}{\rho} m^* \right) - \Delta t \frac{A_x}{A} h^n m^*.$$

## 6.3 Equilibria

Special solutions of the nozzle flow are stationary equilibria, in which the field variables do not depend on time. Such solutions are relevant *per se*, and their study is important in order to construct *well balanced schemes*, i.e. schemes that are accurate even when the solution is a small deviation from the stationary one.

Stationary equations are given by:

$$\begin{cases}
\frac{\partial m}{\partial x} = -\frac{m}{A} \frac{\partial A}{\partial x}, \\
\frac{\partial}{\partial x} (mu + p) = -\frac{mu}{A} \frac{\partial A}{\partial x}, \\
\frac{\partial \rho h u}{\partial x} = -\frac{\rho h u}{A} \frac{\partial A}{\partial x}.
\end{cases} (32)$$

The first equation gives:

$$Am_x + mA_x = 0, (33)$$

from which we deduce that, for stationary flow

$$Q \equiv m A = \text{const.} \tag{34}$$

The third equation gives

$$m_x h + m h_x + h m A_x / A = 0,$$
 (35)

which, making use of Eq. (33) becomes m h<sup>x</sup> = 0, which, for m 6= 0, gives

$$h = \text{const.}$$
 (36)

Making use of Eq. (33), the second equation gives

$$m u_x + p_x = 0. (37)$$

This equation, when combined with the relation

$$h = \frac{1}{2}u^2 + \frac{\gamma}{\gamma - 1}\frac{p}{\rho} = \text{const},$$

gives

$$p_x - a^2 \rho_x = 0, (38)$$

where a <sup>2</sup> = γp/ρ is the square of the sound speed. This differential relation means that (dp/dρ) = a <sup>2</sup> = (dp/dρ)S, which implies that the flow is isentropic, i.e.

$$S = \frac{p}{\rho^{\gamma}} = \text{const.} \tag{39}$$

Here S is a function of the physical entropy density η only. More precisely, S = κ exp(η/cv), where κ is a constant (in classical thermodynamics the entropy is defined up to an additive constant), and c<sup>v</sup> denotes the specific heat at constant volume.

Stationary solutions are therefore characterized by three invariants, Q, h, and S, expressed by relations (34,36,39). We shall make use of this property as a key ingredient for the construction of well balanced schemes.

#### 6.4 Approximate well-balanced scheme

A first order scheme in space and time, which preserves equilibria to second order accuracy, is obtained by adopting, in each cell j, at each time step  $t^n$ , a piecewise linear reconstruction of the conservative variables:

$$\rho_i^n(x) = \bar{\rho}_i^n + \rho_i'(x - x_j), \quad m_i^n(x) = \bar{m}_i^n + m_i'(x - x_j), \quad E_i^n(x) = \bar{E}_i^n + E_i'(x - x_j), \quad (40)$$

where

$$m' = -m\frac{A_x}{A}, \rho' = \frac{um'}{a^2 - u^2}, \quad E' = (h - a^2)\rho'$$
 (41)

and  $a^2 = \gamma p/\rho$ .

Such expressions are deduced by imposing that the derivatives of Q, h and S expressed in terms of the derivatives of conservative variables, are zero at cell center. The Appendix is devoted to the construction of (exact and approximate) well balanced schemes. In particular, equations (41) are deduced in the Appendix, see Eqs. (66,67,68).

Using reconstruction (40) and (41) provides a scheme which is first order accurate in space and time for a general time-dependent solution, but which captures stationary solutions to second order accuracy. Note that because of the implicit-explicit nature of the method, the numerical solution depends on the time step, even when looking for a stationary solution. For this reason we expect slight differences in the stationary solutions produced by the various schemes, even if they are all based on the same second order approximate well-balanced reconstruction.

### 7 Numerical tests for the nozzle flow

In this section we perform some tests on stationary nozzle flows. We solve the initial boundary value problem for system (20) with initial conditions and boundary conditions given by the analytic stationary solution. Then we evolve the system until steady state is reached. As a test for convergence, we stop the code at time  $t^n$  such that

$$\max_{j} |\rho_{j}^{n} - \rho_{j}^{n-1}| \le 10^{-7} \max_{j} |\rho_{j}^{1} - \rho_{j}^{0}| \tag{42}$$

An analytic solution can be computed for any type of flow inside the nozzle (subsonic or transonic with a shock). Thus, error and convergence can be studied.

The geometry is first chosen such that for given total pressure  $P_0$  and total temperature  $T_0$ , and an isentropic flow, the Mach is linear inside the nozzle with the desired Mach number at the inlet and at the outlet with M=1 at the throat. It gives a direct relation between the area A and the Mach number M (considered linear). We have (mass conservation and isentropic relations).

$$Q \equiv \rho u A = \text{const},$$

$$P_0 = p(1 + \frac{\gamma - 1}{2} M^2)^{\frac{\gamma}{\gamma - 1}},$$

$$T_0 = T(1 + \frac{\gamma - 1}{2} M^2),$$
(43)

where total pressure  $P_0$  and total temperature  $T_0$  can be linked to the invariant h and S as

$$S = P_0 \left(\frac{R^* T_0}{P_0}\right)^{\gamma},$$
  
$$h = \frac{\gamma}{\gamma - 1} R^* T_0,$$

where  $R^*$  is the specific gas constant.

The area-Mach number relation becomes

$$\frac{A}{A_{throat}} = \frac{1}{M} \left( \frac{2 + (\gamma - 1)M^2}{2 + (\gamma - 1)} \right)^{\frac{\gamma + 1}{2(\gamma - 1)}}$$
(44)

where  $A_{throat}$  is the nozzle area at the throat where M=1.

The regime in the nozzle can be managed through the pressure imposed at the outlet  $p_{\text{out}}$ . The analytical solution for M at a point x where the area is A is then the solution of

$$(1 + \frac{\gamma - 1}{2}M^2)^{\frac{\gamma + 1}{2(\gamma - 1)}} - \frac{MA}{M_{\text{out}}A_{\text{out}}} (1 + \frac{\gamma - 1}{2}M_{\text{out}}^2)^{\frac{\gamma + 1}{2(\gamma - 1)}} = 0$$
 (45)

where  $M_{\text{out}}$ ,  $A_{\text{out}}$  are the Mach number and the area at the outlet known through the isentropic relations and the pressure imposed at the outlet.

In the case of a stationary shock, the analytical solution is constructed by matching piecewise smooth solutions (as in the previous subsection) with a stationary shock satisfying Rankine-Hunoniot relations.

#### 7.1 Subsonic flow

In this section we present a subsonic flow in a nozzle, for various Mach numbers. Thus, we can compare the accuracy of the different schemes (explicit and semi-implicit) in different subsonic regimes with respect to the analytical solution.

#### 7.1.1 $M \simeq 10^{-1}$

We first consider the nozzle flow in a subsonic regime without any shock such that the Mach number varies between 0.1 and 0.3. To do so, the total pressure and total

![](_page_27_Figure_0.jpeg)

Figure 13: Geometry of the nozzle for subsonic flows.

![](_page_27_Figure_2.jpeg)

Figure 14: Mach profile.

![](_page_27_Figure_4.jpeg)

Figure 15: Convergence of the different methods.

temperature are set to 1 and the pressure at the outlet is set to 0.99. The Mach profile obtained is shown in Figure 14. The CFL number is set to 0.8 for all schemes.

Figure 15 shows the convergence rates (in  $L_{\infty}$  norm of the normalized pressure error with respect to the variation of the analytical solution) for the different methods. We can observe that all the methods converge with second order accuracy. However, the errors amplitudes are very different: in this regime explicit schemes are usually more accurate than their implicit counterpart.

#### 7.1.2 $M \simeq 10^{-3}$

We now have 0.001 < M < 0.0035 by setting the pressure at the outlet at 0.999999.

![](_page_28_Figure_2.jpeg)

![](_page_28_Figure_3.jpeg)

Figure 16: Mach profile.

Figure 17: Convergence of the different methods.

As in the previous case, the convergence test shows a second order for all the methods (see Figure 17). We observe that implicit schemes are more accurate than explicit ones. In particular, P-split with local Lax-Friedrichs fluxes works with a CFL 50 (corresponding to a material CFL of 0.27), resulting in the most cost effective scheme in this regime. Figure 18 shows the comparison of this scheme with respect to the most effective explicit one (with Osher fluxes) in terms of error and CPU time. For the explicit scheme we used 100, 100, 250, 500 and 1000 points, while for the semi-implicit scheme we used 50, 100 and 200 points. The results show, that for a given accuracy, the implicit is faster in terms of CPU time and the explicit scheme requires a finer grid. In practice, to get the same accuracy that the implicit scheme reaches with 100 grid points, the explicit scheme needs about 1000 grid points and is three times slower. For the same number of grid point, the explicit scheme is faster but much less accurate (two orders of magnitude). The computations have been done on a intel Core i7, 2.8GHz. The code is written in Matlab using vectorization and sparse matrix for the linear solver. To reduce the computation times, the tolerance used in the convergence criteria in  $10^{-4}$ .

One should also note that these results are valid only in 1D. The implicit scheme requires to solve a linear system which can become costly in multi dimension. This aspect will be investigated in a future work.

#### 7.2 Transonic flow with a shock

We now consider a shock at x = 0.8 (see geometry on Figure 19). The analytical solution is computed with the isentropic relationships until x = 0.8. Then, the

![](_page_29_Figure_0.jpeg)

Figure 18: Pressure error as function of CPU time for explicit (with Osher fluxes) for 50, 100, 250, 500, 100 grid points and implpicit P-split-LF scheme for 50, 100 and 200 grid points.

Rankine-Hugoniot conditions are used to find the solution after the shock. The solution until the outlet is again computed thanks to the isentropic relationships. The Mach profile is shown on Figure 20.

![](_page_29_Figure_3.jpeg)

Figure 19: Geometry of the nozzle for a transonic flow. Figure 20: Mach profile with a shock at x=0.8.

We observe in Figure 21 that upwind-based schemes are able to well resolve the shock. In particular, in this regime P-split-upwind scheme is just slightly more dissipative than the explicit upwind scheme.

![](_page_30_Figure_0.jpeg)

![](_page_30_Figure_1.jpeg)

Figure 21: Right panel: Pressure profiles. Right panel: zoom on the pressure profiles around the shock at x = 0.8.

### 7.3 Boundary conditions

In previous examples we impose the exact solution at the boundary. In applications of course exact boundary conditions are unknown and therefore they must be modeled. Because of the charcateristic pattern, for a subsonic flow at inlet two boundary conditions are needed. Similarly, for a subsonic flow at the outlet, only one boundary is needed. A rather general boundary model is represented by giving total temperature and pressure at the inlet and static pressure at the outlet. Physically this models the outflow from an infinite reservoir where the fluid is at rest into an environment at given pressure. Similar conditions can be imposed for external flows.

In the same spirit of the semi-implicit discretization scheme, the boundary conditions are imposed thanks to an explicit extrapolation from the interior domain and the solution of an implicit step for the elliptic problem. Let us consider the inlet. The Mach number is extrapolated from the interior in the explicit step and from equations (??) we get static pressure and temperature. Hence we can compute all the conservative variables at the inlet. Similarly for the outlet, where the Mach number and the velocity are extrapolated from the interior domain. Using the given static pressure all the conservative variables are obtained.

In terms of the number of iterations to reach convergence, using these boundary conditions we need about three times more iterations for the compressible regime and about ten time more iterations for the low Mach case, for given convergence threshold.

In Figure 22 we report the convergence rates obtained using the P split LF scheme. As expected the solution is less accurate than that obtained with exact boundary conditions, but second-order accuracy is still recovered for both in the compressible and the low Mach regime.

![](_page_31_Figure_0.jpeg)

Figure 22: Pressure errors imposing  $P_0$  and  $T_0$  at inlet and p at outlet.  $M = 10^{-1}$  (left panel) and  $M = 10^{-3}$  (right panel).

# 8 2D Model

Now we consider the Euler equations of compressible gas dynamics in two dimensions defined in a square domain  $\Omega = [a, b] \times [a, b]$ . The equations are given by:

$$\begin{cases}
\frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{m} = 0, \\
\frac{\partial \mathbf{m}}{\partial t} + \nabla \cdot \left(\frac{\mathbf{m} \otimes \mathbf{m}}{\rho}\right) + \nabla p = 0, \\
\frac{\partial E}{\partial t} + \nabla \cdot (h\mathbf{m}) = 0,
\end{cases} (46)$$

where  $\rho$  is the density,  $\mathbf{m} = (\mathsf{m}, \mathsf{n}) = (\rho u, \rho v)$  the vector of the momentum with the x-direction and y-direction components respectively, p the pressure, E the total energy and  $h = (E + p)/\rho$  the enthalpy. System (46) is closed by the equation of state for a perfect gas in two dimensions:

$$p = (\gamma - 1) \left( E - \frac{\rho}{2} (u^2 + v^2) \right). \tag{47}$$

## 8.1 Pressure spitting in 2D

In the same philosophy as 3.1 we choose to treat with an implicit way the pressure terms. Now, we discretize the system (46) between a time  $t^n$  and a time  $t^{n+1}$ 

$$\begin{cases}
\rho^{n+1} = \rho^{n} - \Delta t \hat{D}_{x}(\mathsf{m}^{n}) - \Delta t \hat{D}_{y}(\mathsf{n}^{n}), \\
\mathsf{m}^{n+1} = \mathsf{m}^{n} - \Delta t \hat{D}_{x}(\mathsf{m}^{n}u^{n}) - \Delta t \hat{D}_{y}(\mathsf{m}^{n}v^{n}) - \Delta t D_{x}(p^{n+1}), \\
\mathsf{n}^{n+1} = \mathsf{n}^{n} - \Delta t \hat{D}_{y}(\mathsf{n}^{n}v^{n}) - \Delta t \hat{D}_{x}(\mathsf{n}^{n}u^{n}) - \Delta t D_{y}(p^{n+1}), \\
E^{n+1} = E^{n} - \Delta t D_{x}(h^{n}\mathsf{m}^{n+1}) - \Delta t D_{y}(h^{n}\mathsf{n}^{n+1}).
\end{cases} (48)$$

Then we make use of the Equation of State (47),

$$p^{n+1}=(\gamma-1)\big(E^{n+1}-\frac{1}{2}((\mathbf{m}^nu^n+\mathbf{n}^nv^n)\big)$$

and we substitute this expression in (48). We choose to treat implicitly the energy term and upwind the rest. Thus, we obtain:

$$\begin{cases} \mathbf{m}^{n+1} = \mathbf{m}^{n} - \frac{3 - \gamma}{2} \Delta t \hat{D}_{x}(\mathbf{m}^{n}u^{n}) - \frac{1 - \gamma}{2} \Delta t \hat{D}_{x}(\mathbf{n}^{n}v^{n}) \\ -\Delta t \hat{D}_{y}(\mathbf{m}^{n}v^{n}) - (\gamma - 1)\Delta t D_{x}(E^{n+1}), \end{cases}$$

$$\mathbf{n}^{n+1} = \mathbf{n}^{n} - \frac{3 - \gamma}{2} \Delta t \hat{D}_{y}(\mathbf{n}^{n}v^{n}) - \frac{1 - \gamma}{2} \Delta t \hat{D}_{y}(\mathbf{m}^{n}u^{n}) \\ -\Delta t \hat{D}_{x}(\mathbf{n}^{n}u^{n}) - (\gamma - 1)\Delta t D_{y}(E^{n+1}) \end{cases}$$

$$(49)$$

Plugging these expressions in the equation for the Energy one obtains:

$$E^{n+1} = E^{n} - \Delta t \hat{D}_{x}(h^{n} \mathsf{m}^{*}) + (\gamma - 1) \Delta t^{2} D_{x}(h^{n} D_{x}(E^{n+1})) - \Delta t \hat{D}_{y}(h^{n} \mathsf{n}^{*}) + (\gamma - 1) \Delta t^{2} D_{y}(h^{n} D_{y}(E^{n+1})),$$
(50)

where  $\mathsf{m}^* = \mathsf{m}^n - \frac{3-\gamma}{2} \Delta t \hat{D}_x(\mathsf{m}^n u^n) - \frac{1-\gamma}{2} \Delta t \hat{D}_x(\mathsf{n}^n v^n) - \Delta t \hat{D}_y(\mathsf{m}^n v^n)$  along with  $\mathsf{n}^* = \mathsf{n}^n - \frac{3-\gamma}{2} \Delta t \hat{D}_y(\mathsf{n}^n v^n) - \frac{1-\gamma}{2} \Delta t \hat{D}_y(\mathsf{m}^n u^n) - \Delta t \hat{D}_x(\mathsf{n}^n u^n)$  are the terms treated explicitly in the momentum equations.

Posing 
$$E^* = E^n - \Delta t \hat{D}_x(h^n \mathsf{m}^*) - \Delta t \hat{D}_y(h^n \mathsf{n}^*)$$
 the equation (50) becomes:  

$$E^{n+1} = E^* + (\gamma - 1)\Delta t^2 D_x(h^n D_x(E^{n+1})) + (\gamma - 1)\Delta t^2 D_y(h^n D_y(E^{n+1})), \quad (51)$$

Now we can solve this system in order to compute  $E^{n+1}$  and then plug it in the momentum equations to compute  $\mathbf{m}^{n+1}$  and  $\mathbf{n}^{n+1}$ .

# 9 Numerical tests on 2D Euler equations

We perform three test cases in order to test the robustness of the scheme over a wide range of Mach numbers.

#### 9.1 Sod shock tube

This test shows that the scheme is able to work on a wide range of Mach numbers. We compare the 2D scheme with the 1D scheme by doing the following. We choose a square domain  $\Omega = [0, \sqrt{0.5}] \times [0, \sqrt{0.5}]$ . In order to initialize the test, we place the discontinuity along the main diagonal, thus the domain is divided into the upper and lower part and the initial conditions ( with index U and L respectively, Figure 23 ) are:

$$u_U = 0$$
  $u_L = 0$   
 $v_U = 0$   $v_L = 0$   
 $\rho_U = 1$   $\rho_L = 0.125$   
 $p_U = 1$   $\rho_L = 0.1$ 

![](_page_33_Picture_3.jpeg)

Figure 23: Initial conditions of Sod shock tube in 2D.

![](_page_33_Figure_5.jpeg)

Figure 24: Density Surf at T=0.168 and the solution vector we keep in order to compare with the 1D case.

Then we keep as solution the vector that contains the elements of the counter diagonal of the solution matrix (Figure 24). This test is performed in order to compare in a quantitatively way the solutions obtained with the 1D and 2D code. We regulate the CFL number in such a way that the timestep in both cases is the same (0.7 for the 2D scheme). We observe that the 2D code is much more accurate even if we are using half of the points we are using in the 1D computation. In figures 25a and 25b we see the comparison between the two schemes. Here,  $D_x$  and  $D_y$  denote second order central difference approximation of space derivatives.  $\hat{D}_x$  and  $\hat{D}_y$  are second order finite volume discretization obtained just as in 1D case.

![](_page_34_Figure_1.jpeg)

Figure 25: Sod shock tube. Comparison between 1D and 2D code

## 9.2 Gresho vortex (Convergence test)

In this test we apply our scheme to the Gresho vortex problem [28]. This is done in order to check the effect of the numerical diffusion to the solution at the final time  $T=0.4\pi$ . The Gresho vortex is a stationary solution of the Euler equations. We run the test with different values of Mach number M=0.1,0.01,0.001 in order to compare the results of the numerical scheme with the initial conditions. To perform this test we assume a square domain  $\Omega=[-0.5,0.5]\times[-0.5,0.5]$  and we center the vortex to (x,y)=(0,0). The initial conditions of the problem in polar coordinates are given by:

$$(u_{\phi}(r), p(r)) = \begin{cases} (5r, p_0 + \frac{25}{2}r^2), & 0 \le r < 0.2, \\ (2 - 5r, p_0 + \frac{25}{2}r^2 + 4(1 - 5r - \ln(0.2) + \ln(r)) & 0.2 \le r < 0.4, \\ (0, p_0 - 2 + 4\ln(2)) & 0.4 \le r. \end{cases}$$

where  $p_0 = \frac{\rho}{\gamma M^2}$  and the density is constant  $\rho = 1$  in the whole domain. We introduce a modified CFL number called  $CFL_{IM}$  and we calculate the timestep as follows:

$$\Delta t = CFL_{IM} \frac{\Delta x}{\max|u| + \max|v|}$$
(52)

The  $CFL_{IM}$  number used for this test is 0.15. In figures 26 and 27 we plot the pressure profiles at the center of the domain in both directions. We observe that the scheme preserves the stationary solution for a wide range of Mach numbers.

![](_page_35_Figure_3.jpeg)

Figure 26: Pressure Profiles, M = 0.1 at  $T = 0.4\pi$  (320pts)

![](_page_35_Figure_5.jpeg)

Figure 27: Pressure Profiles, M = 0.001 at  $T = 0.4\pi$  (320pts)

We perform a convergence test by computing the so-called EOC (experimental order of convergence). In order to compute the EOC we use as reference solution the initial conditions of the problem. Thus we calculate the error by using the following formula:

![](_page_36_Figure_0.jpeg)

Figure 28: Pseudocontour plot of pressure, M = 0.1 at T = 0.4π (320pts)

![](_page_36_Figure_2.jpeg)

Figure 29: Pseudocontour plot of pressure, M = 0.001 at T = 0.4π (320pts)

$$e_N = \frac{||U_N - U_I||_{L^1}}{||U_I||_{L^1}} \tag{53}$$

where U<sup>N</sup> is the numerical solution calculated on a grid with N × N points and U<sup>I</sup> is the initial condition of the problem taken as a reference solution. Then the EOC is calculated:

$$EOC := \log_2\left(\frac{e_N}{e_{2N}}\right) \tag{54}$$

Pressure errors and the corresponding EOC for the Gresho vortex test are presented in table 1. In Figure 30 we plot the evolution of the kinetic energy EKin(t), normalized with respect to the initial value EKin(0), for two different meshes 40×40(dotted line) and 80 × 80 (cross line) with CF LIM = 0.25. For each mesh we use all the values of = 10<sup>−</sup><sup>1</sup> , 10<sup>−</sup><sup>2</sup> , 10<sup>−</sup><sup>3</sup> and we observe that the lines are indistinguishable for each mesh.

![](_page_37_Figure_1.jpeg)

Figure 30: Evolution of the total Kinetic energy normalized with respect to the initial Kinetic energy. The dotted line is for the 40 × 40 and the cross line is for the 80 grid. We use = 10<sup>−</sup>1, 10<sup>−</sup>2, 10<sup>−</sup>3 for both meshes. The points for the different values of are indistinguishable

| N   | M=0.1 (T<br>= 0.4π) |              | M=0.01 (T<br>= 0.4π) |              | M=0.001 (T<br>= 0.4π) |              |
|-----|---------------------|--------------|----------------------|--------------|-----------------------|--------------|
|     | 1<br>L<br>error     | 1 order<br>L | 1<br>L<br>error      | 1 order<br>L | 1<br>L<br>error       | 1 order<br>L |
| 40  | 1.95e-04            | -            | 3.38e-06             | -            | 1.35e-07              | -            |
| 80  | 5.50e-05            | 1.8277       | 3.80e-07             | 3.1549       | 3.72e-09              | 5.1851       |
| 160 | 1.77e-05            | 1.6380       | 1.21e-07             | 1.6455       | 1.03e-09              | 1.8525       |
| 320 | 4.24e-06            | 2.0586       | 7.97e-08             | 0.6079       | 2.75e-10              | 1.9018       |

Table 1: Convergence table for the Gresho vortex

## 9.3 Vortex dipole

Here we compare the result of the scheme to an accurate solution of incompressible Euler equations. We use the same Low Mach number formulation as [2] and we introduce a parameter which is a global Mach number. The accurate solution is obtained by a spectral method applied to incompressible in the vorticity stream function formulation where:

$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y} \tag{55}$$

we initialize the test as following:

$$\omega(x, y, 0) = \begin{cases} \delta \cos x - \frac{1}{\rho} \operatorname{sech}^{2}((y - \pi/2)/\rho)), & y \leq \pi, \\ \delta \cos x - \frac{1}{\rho} \operatorname{sech}^{2}((3\pi/2 - y)/\rho)), & y > \pi, \end{cases}$$

where δ = 0.05 and ρ = π/15. In this case because ∇ · u = 0 there is a function ψ such that:

$$\mathbf{u} = \left(-\frac{\partial \psi}{\partial y}, \frac{\partial \psi}{\partial x}\right)$$

Plugging this expression to (55) we obtain the Poisson equation:

$$-\Delta\psi = \omega$$

The density and the pressure for this test are set equal to 1 on the whole domain and we assume periodic boundary conditions. The final time is T = 6, the CF LIM number is 0.35 and as reference solution we consider a very accurate solution obtained by a spectral method and a fourth order Runge-Kutta method in time.

![](_page_38_Figure_6.jpeg)

Figure 31: Vortex dipole

In Figure 32 we show the behavior of the L <sup>1</sup> norm as the difference between the velocities of the numerical solution of the compressible Euler equations with a reference solution obtained by the aforementioned spectral method in a very fine grid. For this test we use ρ = π/10 and the final time is T = 1.

# Conclusions

The main goal of the paper was to identify an all Mach number scheme which is sufficiently robust to work on a large range of Mach number, and which is simple enough in view of more demanding applications.

![](_page_39_Figure_0.jpeg)

Figure 32: L <sup>1</sup> norm of the velocities compared with a very accurate solution obtained by a spectral method.

We propose a novel family of potentially all Mach number schemes for gas dynamics. The general idea is that explicit differential operators in space relative to convective or material speeds are discretized by upwind schemes or local Lax-Friedrics fluxes. The linear implicit operators, pertaining to acoustic waves, are discretized by central differences.

We have compared the results of such schemes on a series of one-dimensional test problems including classical shock tube configurations. Also we have considered lowmach number acoustic wave propagation tests as well as nozzle flows in various Mach regimes.

We found that there is no scheme that outperforms the others over the whole Mach number range. In contrast, there exist implicit schemes that are robust enough to work in all regimes, like for example the local Lax-Friedrichs pressure-splitting schemes. Furthermore, for low-mach number flows, implicit schemes are far more accurate and efficient compared to explicit ones for given precision.

The perspectives of this work are multiple. We plan to consider multi material flows where several time scales are induced by different wave speeds and space-time adaptivity in multi dimensions.

# Acknowledgments

The work has been partially supported by ITN-ETN Horizon 2020 Project Mod-CompShock, Modeling and Computation on Shocks and Interfaces, Project Reference 642768, and by the Visiting Scholars Position of the Excellence Initiative of the Universit´e de Bordeaux.

### A Well balanced methods

Consider a hyperbolic system of balance laws

$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = g(x, u), \qquad (t, x) \in [0, \infty) \times [a, b], \tag{56}$$

with the initial conditions

$$u(0,x) = u_0(x), \qquad x \in [a,b].$$

In this section we describe an approach for the construction of well-balanced schemes for system (56) with regard to finite volume methods. For the details about well-balanced schemes see, for example [36] and [35], where this technique was presented in a different context, and references therein.

#### A.1 General finite volume scheme

We divide the computational domain [a, b] into J equal intervals  $I_j = [x_{j-1/2}, x_{j+1/2}]$  (j = 1, ..., J) of length  $\Delta x = (b - a)/J$ . Let  $x_j = a + (j - 1/2)\Delta x$  be the centers of the cells,  $x_{j+1/2} = a + j\Delta x$  be the intercell boundaries. We denote a cell average of u(x, t) over the cell  $I_j$  by  $\bar{u}_j(t)$ .

Integration of Eq. (56) over cell  $I_i$  gives a semidiscrete equation

$$\frac{d\bar{u}_j}{dt} = \frac{1}{\Delta x} \left( F_{j-1/2} - F_{j+1/2} \right) + \langle g \rangle_j(t), \tag{57}$$

where  $F_{j+1/2} = F(u_{j+1/2}^-, u_{j+1/2}^+)$  is a numerical flux at the intercell boundary  $x_{j+1/2}$ ;  $u_{j+1/2}^-$  and  $u_{j+1/2}^+$  are approximations of the limiting values of u at  $x_{j+1/2}$ , obtained by some suitable reconstruction;  $\langle g \rangle_j$  is the cell average of the source. The numerical flux  $F_{j+1/2}$  can be computed by any appropriate Riemann solver, given by the numerical flux function  $F(u^-, u^+)$ .

A numerical scheme (57) will be well-balanced if we define  $u_{j+1/2}^{\pm}$  and  $\langle g \rangle_j$  in such a way that the right-hand side of Eq. (57) vanishes at steady-state solutions. One way this is actually implemented is illustrated below.

## A.2 Equilibrium and conservative variables

To make the scheme well-balanced we use so called *equilibrium variables* v in addition to conservative ones u. The equilibrium variables are defined as such variables which are constant at stationary solutions. We suppose that there exists a one-to-one mapping u = U(x, v) between equilibrium variables v and conservative ones u. If v = const then  $u^{e}(x) = U(x, v)$  is a stationary solution of Eq. (56):

$$\frac{\partial f(u^{\mathbf{e}})}{\partial x} = g(x, u^{\mathbf{e}}). \tag{58}$$

The idea is to use cell averages of the conservative variables u for the evolution, by solving system (57) and, at the same time, to use equilibrium variables v for the computation of the intercell limits  $u_{j+1/2}^{\pm}$  and source cell averages  $\langle g \rangle_j$ . How this is done is explained below.

#### A.3 Reconstruction of the field variables at cell boundaries

Given the cell averages  $\bar{u}_j^n$ , we define equilibrium cell averages  $\bar{v}_j$  as constants which satisfy the equation

$$\frac{1}{\Delta x} \int_{I_j} U(x, \bar{v}_j) \, dx = \bar{u}_j^n. \tag{59}$$

Then using these values  $\bar{v}_j$  we define intercell boundary values of conservative variables  $u_{i+1/2}^{\pm}$  as

$$u_{j+1/2}^- = U(x_{j+1/2}, \bar{v}_j), \quad u_{j+1/2}^+ = U(x_{j+1/2}, \bar{v}_{j+1}),$$
 (60)

and the average of the source term as

$$\langle g \rangle_j^n = \frac{1}{\Delta x} \int_{I_i} g(x, U(x, \bar{v}_j)) dx. \tag{61}$$

# A.4 First order in space and time, perfectly well-balanced scheme

A first order scheme is obtained by applying forward Euler time discretization to system (57)

$$\bar{u}_j^{n+1} = \bar{u}_j^n + \frac{\Delta t}{\Delta x} (F_{j-1/2} - F_{j+1/2}) + \Delta t \langle g \rangle_j^n,$$
 (62)

where the fluxes at cell edges are obtained from the numerical flux function, and the values at the edge of the cells,  $u_{j+1/2}^{\pm}$ , and the average of the source are obtained from the conservative reconstruction described above, Eqs.(59,60,61).

Note that for any constant  $\bar{v}_j$ , the reconstruction  $u_j^n(x) = U(x, \bar{v}_j)$  is a local equilibrium, which satisfies Eq.(58) in each cell. The reconstruction

$$u^{n}(x) = \sum_{j=1}^{J} u_{j}^{n}(x) \chi_{I_{j}}(x),$$

where  $\chi_I$  denotes the characteristic function of interval I, is therefore composed by piecewise equilibria. Such reconstruction can be considered as the and generalization to systems with source terms of the piecewise constant reconstruction that is usually adopted in first order Godunov-like schemes for systems of conservation laws.

If all values  $\bar{v}_j$  are the same, say  $\bar{v}_j = \bar{v}, j = 1, \dots, J$ , then the profile  $u^n(x)$  would be a global equilibrium, since

$$\sum_{j=1}^{J} U(x, \bar{v}_j) \chi_{I_j}(x) = \sum_{j=1}^{J} U(x, \bar{v}) \chi_{I_j}(x) = U(x, \bar{v}).$$

That a scheme defined by (62,59,60,61) with  $F_{j+1/2} = F(u_{j+1/2}^-, u_{j+1/2}^+)$  is well balanced can be verified by inspection: if the state  $\{\bar{u}_j^n\}$  represents an equilibrium, by definition the equilibrium variables will be constant, therefore  $\bar{v}_j = \bar{v}$ , the reconstruction  $u^n(x) = U(x,\bar{v})$  will be a global equilibrium,  $U(x,\bar{v}) = u^e(x)$ . By consistency of the numerical flux function,  $F_{j\pm 1/2} = F(u_{j\pm 1/2}^-, u_{j\pm 1/2}^+) = F(u^e(x_{j\pm 1/2}), u^e(x_{j\pm 1/2})) = f(u^e(x_{j\pm 1/2}))$  and therefore we have

$$\frac{\Delta t}{\Delta x} \left( F_{j-1/2} - F_{j+1/2} \right) + \Delta t \left\langle g \right\rangle_j^n = \Delta t \int_{I_j} \left[ g(x, u^{\mathrm{e}}(x)) - \frac{\partial f(u^{\mathrm{e}}(x))}{\partial x} \right] dx,$$

where the term on the right hand side vanishes because of Eq. (58).

The above scheme is only first order accurate in space and time, but it is in principle perfectly well-balanced: it preserves equilibria exactly.

### A.5 Application to nozzle flow

In the case of the nozzle flow, a well balanced scheme can be constructed by looking for a local reconstruction that at the same time preserves the cell averages and such that the invariants of the stationary flow are piecewise constant.

This can be obtained as follows. Let us assume we are able to invert the relation between the conservative and equilibrium variables:

$$\rho = \rho(x; Q, h, S), \quad m = m(x; Q, h, S), \quad E = E(x; Q, h, S).$$

Then, at each time  $t^n$ , for each cell j, we look for three constants,  $Q_j^n$ ,  $h_j^n$ , and  $S_j^n$ , such that the average of the conservative variables in each cell have the prescribed value, i.e. we impose

$$\frac{1}{\Delta x} \int_{I_j} \rho(x; Q_j^n, h_j^n, S_j^n) \, dx = \bar{\rho}_j^n, \quad \frac{1}{\Delta x} \int_{I_j} m(x; Q_j^n, h_j^n, S_j^n) \, dx = \bar{m}_j^n, 
\frac{1}{\Delta x} \int_{I_j} E(x; Q_j^n, h_j^n, S_j^n) \, dx = \bar{E}_j^n.$$
(63)

Once such quantities are found, then we use the obtained reconstructions  $\rho = \rho(x;Q,h,S)$ , m=m(x;Q,h,S), E=E(x;Q,h,S) in order to compute the values of the conservative variables at each side of each cell, and the average of the source. With all these values we use Euler scheme in time with any consistent numerical flux function, and construct a numerical solution which will be automatically well-balanced.

In practice, conditions (63) are imposed as follows: From the expression of m,  $m_i(x) = Q_i/A(x)$ , using the second equation of (63), we obtain:

$$Q_j^n \langle A^{-1} \rangle_j = \bar{m}_j^n.$$

Then, replacing the expression  $p = S\rho^{\gamma}$  in the expression of h, expressing m in terms of Q and A, and imposing that h is a constant, we obtain:

$$\frac{Q_j^2}{2A(x)^2\rho^2} + \frac{\gamma}{\gamma - 1}S_j\rho^{\gamma - 1} = h_j.$$

This is a nonlinear equation for  $\rho$ , which depends on x because of the x dependence of A. Once it is solved, it allows to express  $\rho$  as a function of  $x, Q_j, S_j, h_j$ , i.e.  $\rho = \rho(x; Q_j, S_j, h_j)$ . Energy can be also expressed as function of such quantities:

$$E = E(x; Q, S, h) = \frac{m^2}{2\rho} + \frac{S\rho^{\gamma}}{\gamma - 1}.$$

The equations for  $S_j$  and  $h_j$  are finally obtained by imposing

$$\langle \rho \rangle_j = \bar{\rho}_j^n, \quad \langle E \rangle_j = \bar{E}_j^n.$$
 (64)

Notice that by solving exactly the equation for  $\rho$  and by imposing the conditions (64) one obtains a scheme that is first order in space and time for the evolution system (20), but which preserves equilibria exactly. The construction of such scheme is however almost impossible, since it requires the exact solution of several non-linear equations. Several approximate schemes can however be adopted, in order to produce approximate well-balanced solutions. Such approximate schemes are described in the next subsection.

## A.6 Approximate well-balanced schemes

The construction of exactly well-balanced schemes presents two difficulties. The first is the solution of nonlinear equations that allow to express the conservative quantities as a function of the equilibrium variables. The second is that the conditions required to impose that the average of the reconstructions coincides with cell average are of integral nature, and it is difficult to impose them exactly. Approximate well-balanced schemes can be constructed in several ways. One possibility is to approximate, for example, the integrals appearing in (59) by quadrature formulas. In such a case, condition (59) is replaced by

$$\sum_{i=0}^{\nu} b_i U(x_{i-1/2} + c_i \Delta x, \bar{v}_j) = \bar{u}_j^n.$$
(65)

where the integral is replaced by a quadrature formula in [0,1], with nodes and weights, respectively,  $c_i$  and  $b_i$ ,  $i = 0, ..., \nu$ . Using for example the two node

(ν = 1) Gauss-Legendre quadrature formula would guarantee fourth order accuracy. Assuming we can compute Q<sup>n</sup> j explicitly, application of the method to our case still requires the solution of a set of two nonlinear equations for S n j and h n j .

In order to write such equations one has to solve the nonlinear equation for ρ in each cell in two different points. A simpler approach can be obtained by a collocation method. Since Q<sup>j</sup> , S<sup>j</sup> , and h<sup>j</sup> have to be constant, then their first derivative has to vanish identically in each interval. If we express the derivatives of the equilibrium variables in terms of the conservative variables, we obtain a set of ordinary differential equations, the solution of which provides local equilibria. Rather than imposing that such differential system is satisfied for all points x in I<sup>j</sup> , we impose the condition on some collocation nodes. The simplest choice is to impose that, in each cell,

$$Q'(x_j) = 0$$
,  $S'(x_j) = 0$ ,  $h'(x_j) = 0$ .

The expression of the derivatives of the conservative variables is easily obtained from equations of Section 6.3: from (33) one obtains

$$m_x = -m\frac{A_x}{A},\tag{66}$$

from (37) and (38) one obtains

$$\rho_x = \frac{um_x}{a^2 - u^2},\tag{67}$$

and differentiating the relation E = hρ − p, and making use of the fact that h is constant and of Eq. (38),

$$E_x = h\rho_x - p_x = (h - a^2)\rho_x. (68)$$

Relations (?? are used in Sec. 6.4 in the construction of scheme which are wellbalanced to second order.

The procedure outlined above can be adopted as a building block for the construction of arbitrary high order well-balanced schemes. This requires two major ingredients. The first one is to compute higher order reconstructions by suitable combination of piecewise equilibria. The second ingredient is to adopt equilibrium variables to compute predictor values at cell edges. Such predictor values are then adopted in order to compute fluxes at cell edges, necessary for the high order update of the equilibrium variables. Application of such procedure to the shallow water equations are presented in [36] and [35].

# References

[1] Uri M Ascher, Steven J Ruuth, and Raymond J Spiteri. Implicit-explicit rungekutta methods for time-dependent partial differential equations. Applied Numerical Mathematics, 25(2):151–167, 1997.

- [2] S. Boscarino, G. Russo, and L. Scandurra. All mach number second order semi-implicit scheme for the euler equations of gasdynamics. 2017.
- [3] Sebastiano Boscarino, Francis Filbet, and Giovanni Russo. High order semiimplicit schemes for time dependent partial differential equations. Journal of Scientific Computing, pages 1–27, 2016.
- [4] Gui Qiang Chen, C David Levermore, and Tai-Ping Liu. Hyperbolic conservation laws with stiff relaxation terms and entropy. Communications on Pure and Applied Mathematics, 47(6):787–830, 1994.
- [5] Fr´ed´eric Coquel, Q-L Nguyen, Marie Postel, and Q-H Tran. Large time step positivity-preserving method for multiphase flows. In Hyperbolic Problems: Theory, Numerics, Applications, pages 849–856. Springer Berlin Heidelberg, 2008.
- [6] Fr´ed´eric Coquel, Quang Nguyen, Marie Postel, and Quang Tran. Entropysatisfying relaxation method with large time-steps for euler ibvps. Mathematics of Computation, 79(271):1493–1533, 2010.
- [7] Fr´ed´eric Coquel, Quang Long Nguyen, Marie Postel, and Quang Huy Tran. Local time stepping with adaptive time step control for a two-phase fluid system. In ESAIM: Proceedings, volume 29, pages 73–88. EDP Sciences, 2009.
- [8] Fr´ed´eric Coquel, Quang Long Nguyen, Marie Postel, and Quang Huy Tran. Local time stepping applied to implicit-explicit methods for hyperbolic systems. Multiscale Modeling & Simulation, 8(2):540–570, 2010.
- [9] Fr´ed´eric Coquel, Marie Postel, Nicole Poussineau, and Quang-Huy Tran. Multiresolution technique and explicit–implicit scheme for multicomponent flows. Journal of Numerical Mathematics jnma, 14(3):187–216, 2006.
- [10] Floraine Cordier, Pierre Degond, and Anela Kumbaro. An asymptoticpreserving all-speed scheme for the euler and navier–stokes equations. Journal of Computational Physics, 231(17):5685–5704, 2012.
- [11] G. Costigan and P.B. Whalley. Measurements of the speed of sound in air-water flows. Chemical Engineering Journal, 66(2):131 – 135, 1997.
- [12] P. Degond, S. Jin, and J.-G. Liu. Mach-number uniform asymptotic-preserving gauge schemes for compressible flows. Bull. Inst. Math., Acad. Sin., Vol. 2(No. 4):pp. 851–892, 2007.
- [13] Pierre Degond and Min Tang. All speed scheme for the low mach number limit of the isentropic euler equation. arXiv preprint arXiv:0908.1929, 2009.

- [14] St´ephane Dellacherie. Analysis of godunov type schemes applied to the compressible euler system at low mach number. Journal of Computational Physics, 229(4):978–1016, 2010.
- [15] Edwige Godlewski and Pierre-Arnaud Raviart. Numerical Approximation of Hyperbolic Systems of Conservation Laws. Springer, 2014.
- [16] Jeffrey Haack, Shi Jin, and Jian-Guo Liu. An all-speed asymptotic-preserving method for the isentropic euler and navier-stokes equations. Communications in Computational Physics, 12(04):955–980, 2012.
- [17] E. Hairer and G. Wanner. Solving Ordinary Differential Equations II. Stiff and Differential-Algebraic Problems. (2Nd Revised. Ed.), volume 14 of Springer Series in Comput. Mathematics. Springer-Verlag New York, Inc., New York, NY, USA, 1996.
- [18] Douglas S. Harned and W. Kerner. Semi-implicit method for three-dimensional resistive magnetohydrodynamic simulation of fusion plasmas. Nuclear Science and Engineering, 92(1):119–125, 1986.
- [19] Samet Y Kadioglu and Dana A Knoll. A fully second order implicit/explicit time integration technique for hydrodynamics plus nonlinear heat conduction problems. Journal of Computational Physics, 229(9):3237–3249, 2010.
- [20] Samet Y Kadioglu, Dana A Knoll, Robert B Lowrie, and Rick M Rauenzahn. A second order self-consistent imex method for radiation hydrodynamics. Journal of Computational Physics, 229(22):8313–8332, 2010.
- [21] Samet Y Kadioglu, Mark Sussman, Stanley Osher, Joseph P Wright, and Myungjoo Kang. A second order primitive preconditioner for solving all speed multi-phase flows. Journal of computational physics, 209(2):477–503, 2005.
- [22] Sergiu Klainerman and Andrew Majda. Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible fluids. Communications on Pure and Applied Mathematics, 34(4):481–524, 1981.
- [23] Sergiu Klainerman and Andrew Majda. Compressible and incompressible fluids. Communications on Pure and Applied Mathematics, 35(5):629–651, 1982.
- [24] R. Klein. Semi-implicit extension of a godunov-type scheme based on low mach number asymptotics. i: One-dimensional flow. J. Comput. Phys., Vol. 121(No. 2):pp. 213–237, 1995.
- [25] Nipun Kwatra, Jonathan Su, J´on T Gr´etarsson, and Ronald Fedkiw. A method for avoiding the acoustic time step restriction in compressible flow. Journal of Computational Physics, 228(11):4146–4161, 2009.

- [26] Randall J LeVeque. Finite volume methods for hyperbolic problems, volume 31. Cambridge university press, 2002.
- [27] F. Miczek, F.K. R¨opke, and P.V.F. Edelmann. A new numerical solver for flows at various mach numbers. Astronomy & Astrophysics, Vol. 576:A50, 2015.
- [28] Miczek, F., R¨opke, F. K., and Edelmann, P. V. F. New numerical solver for flows at various mach numbers. A&A, 576:A50, 2015.
- [29] C-D Munz, Sabine Roller, Rupert Klein, and Karl J Geratz. The extension of incompressible flow solvers to the weakly compressible regime. Computers & Fluids, 32(2):173–196, 2003.
- [30] S. Noelle, G. Bispen, K. R. Arun, and C.-D. Munz Luk´aˇcov´a-Medvidov´a, M. An asymptotic preserving all mach number scheme for the euler equations of gas dynamics. Technical Report 348, IGPM , RWTH-Aachen, Germany, 2012.
- [31] Andrew Nonaka, AS Almgren, JB Bell, MJ Lijewski, CM Malone, and M Zingale. Maestro: An adaptive low mach number hydrodynamics algorithm for stellar flows. The Astrophysical Journal Supplement Series, 188(2):358, 2010.
- [32] S Osher and F Solomon. Upwind difference schemes for hyperbolic systems of conservation laws. Mathematics of Computation, 1982.
- [33] Lorenzo Pareschi and Giovanni Russo. Implicit-explicit runge-kutta schemes and applications to hyperbolic systems with relaxation. Journal of Scientific computing, 25(1-2):129–155, 2005.
- [34] JH Park and C-D Munz. Multiple pressure variables methods for fluid flow at all mach numbers. International journal for numerical methods in fluids, 49(8):905–931, 2005.
- [35] Giovanni Russo and Alexander Khe. High order well balanced schemes for systems of balance laws. In Hyperbolic problems: theory, numerics and applications, volume 67 of Proc. Sympos. Appl. Math., pages 919–928. Amer. Math. Soc., Providence, RI, 2009.
- [36] Giovanni Russo and Alexander Khe. High order well-balanced schemes based on numerical reconstruction of the equilibrium variables. In Waves and Stability in Continuous Media, volume 1, pages 230–241, 2010.
- [37] A H Shapiro. The Dynamics and Thermodynamics of Compressible Fluid Flow, May 1953.
- [38] Chi-Wang Shu. Essentially non-oscillatory and weighted essentially nonoscillatory schemes for hyperbolic conservation laws. In Advanced numerical approximation of nonlinear hyperbolic equations, pages 325–432. Springer Berlin Heidelberg, 1998.

- [39] Eleuterio F Toro. Riemann solvers and numerical methods for fluid dynamics: a practical introduction. third edition, 2009.
- [40] Eli Turkel. Preconditioned methods for solving the incompressible and low speed compressible equations. Journal of computational physics, 72(2):277–298, 1987.
- [41] D.R. van der Heul, C. Vuik, and P. Wesseling. A conservative pressurecorrection method for flow at all speeds. Computers & Fluids, 32(8):1113 – 1132, 2003.
- [42] C´ecile Viozat. Implicit Upwind Schemes for Low Mach Number Compressible Flows. Technical Report RR-3084, INRIA, January 1997.