# High order pressure-based semi-implicit IMEX schemes for the 3D Navier-Stokes equations at all Mach numbers

Walter Boscheri∗<sup>a</sup> , Lorenzo Pareschi<sup>a</sup>

*<sup>a</sup>Department of Mathematics and Computer Science, University of Ferrara, Ferrara, Italy*

# Abstract

This article aims at developing a high order pressure-based solver for the solution of the 3D compressible Navier-Stokes system at all Mach numbers. We propose a cell-centered discretization of the governing equations that splits the fluxes into a fast and a slow scale part, that are treated implicitly and explicitly, respectively. A novel semiimplicit discretization is proposed for the kinetic energy as well as the enthalpy fluxes in the energy equation, hence avoiding any need of iterative solvers. The implicit discretization yields an elliptic equation on the pressure that can be solved for both ideal gas and general equation of state (EOS). A nested Newton method is used to solve the mildly nonlinear system for the pressure in case of nonlinear EOS. High order in time is granted by implicit-explicit (IMEX) time stepping, whereas a novel CWENO technique efficiently implemented in a dimension-by-dimension manner is developed for achieving high order in space for the discretization of explicit convective and viscous fluxes. A quadrature-free finite volume solver is then derived for the high order approximation of numerical fluxes. Central schemes with no dissipation of suitable order of accuracy are finally employed for the numerical approximation of the implicit terms. Consequently, the CFL-type stability condition on the maximum admissible time step is based only on the fluid velocity and not on the sound speed, so that the novel schemes work uniformly for all Mach numbers. Convergence and robustness of the proposed method are assessed through a wide set of benchmark problems involving low and high Mach number regimes, as well as inviscid and viscous flows.

*Keywords:* All Mach number flow solver, Asymptotic preserving methods, Semi-implicit IMEX schemes, 3D compressible Euler and Navier-Stokes equations, Quadrature-free WENO, General equation of state (EOS)

# 1. Introduction

The unsteady compressible Navier-Stokes equations constitute a mathematical model for the simulation of a wide set of applications in fluid mechanics, that involve aerospace and mechanical engineering as well as environmental engineering [\[67,](#page-36-0) [34,](#page-35-0) [48,](#page-35-1) [2,](#page-34-0) [50,](#page-35-2) [56\]](#page-35-3). Atmospheric flows, geophysical flows in oceans, rivers and lakes can be described relying on the Navier-Stokes model, which is also used in industrial applications such as the design of wind or water turbines, aircraft engines and cars. The governing equations are based on the physical principle of conservation and they can be derived from the conservation of mass, momentum and total energy. The compressible Navier-Stokes equations already embed several simplified sub-systems, such as the compressible Euler equations in the case of inviscid flows or the incompressible Navier-Stokes equations, that can be retrieved in the zero Mach number limit. The Mach number, which is the ratio between the fluid velocity and the sound speed, describes the regime of the fluid under consideration. High Mach number situations are typically encountered in industrial engineering, whereas geophysical phenomena mostly involve low Mach number flows.

The numerical methods developed for the solution of high and low Mach number problems are quite different, because of the nature of the governing equations. For the high Mach number case explicit upwind finite difference and Godunov-type finite volume methods are very popular [\[33,](#page-35-4) [1,](#page-34-1) [43,](#page-35-5) [44,](#page-35-6) [49,](#page-35-7) [52\]](#page-35-8). On the other hand, in the incompressible

*Email addresses:* walter.boscheri@unife.it (Walter Boscheri<sup>∗</sup> ), lorenzo.pareschi@unife.it (Lorenzo Pareschi)

<sup>∗</sup>Corresponding author

regime the elliptic behavior of the pressure introduces a very severe restriction on the maximum admissible time step for low Mach number flows. Indeed, the CFL-type stability condition for explicit methods depends also on the sound speed which becomes predominant in the zero Mach limit. Furthermore, in [\[20\]](#page-34-2) the effect of numerical viscosity on the slow waves introduced by upwind-type schemes is proven to degrade the accuracy. As a consequence, implicit strategies for time discretization have been proposed in order to avoid the acoustic CFL restriction and enlarge the time step. However, fully implicit methods imply the solution of large nonlinear systems that are computationally very expensive and in which the convergence is numerically very difficult to control. In addition, in many realistic scenarios, both high and low Mach regimes coexist and can arise during the simulation without being predicted in advance, thus needing the design of numerical methods that can deal with all Mach numbers.

This is the reason behind the research activity carried out in the recent past for investigating an alternative strategy to treat problems with multiple time scales. A successful idea consists in treating implicitly only one part of the system to be solved while keeping the remaining explicit, thus both incompressible and compressible regimes can be handled [\[56,](#page-35-3) [65,](#page-36-1) [53,](#page-35-9) [19,](#page-34-3) [14,](#page-34-4) [46,](#page-35-10) [31\]](#page-35-11). This approach permits to design space and time discretizations in which the implicit part of the system is relatively easy to be inverted, typically avoiding nonlinear systems, while keeping robustness and shock-capturing properties in the explicit part. There are mainly two classes of schemes that allows to deal with split sub-systems, namely one for the fast and the other for the slow scale phenomena, that are treated implicitly and explicitly, respectively. The first class is given by implicit-explicit (IMEX) methods [\[66,](#page-36-2) [8,](#page-34-5) [55,](#page-35-12) [9\]](#page-34-6) or, more in general, by the so-called partitioned schemes [\[51\]](#page-35-13). IMEX schemes are proven to be very effective for many applications. Their main feature is to achieve high order under a time step stability constraint independent of the values of the fast scale, and to satisfy the Asymptotic Preserving (AP) property, meaning that the limit model is consistently reproduced at the discrete level [\[40,](#page-35-14) [41\]](#page-35-15). The other class is represented by semi-implicit methods [\[42,](#page-35-16) [16,](#page-34-7) [17,](#page-34-8) [58,](#page-35-17) [59\]](#page-35-18), which have also gained visibility in the past years. Here, the idea is to obtain a linearly implicit scheme for the stiff terms in the governing equations, thus avoiding any need of iterative methods. In [\[27\]](#page-35-19) a flux splitting for the Euler equations is proposed that aims at obtaining an advection and a pressure sub-system. Both sub-systems are demonstrated to be hyperbolic and in [\[23\]](#page-34-9) the advection system is treated explicitly, while the pressure sub-system is discretized implicitly, so that the time step is only limited by the fluid velocity and not by the acoustic speed. Staggered meshes are employed, hence allowing for compact stencils and permitting to recover by construction the divergence free constraint of the velocity field in the low Mach limit, along the lines of [\[16,](#page-34-7) [17\]](#page-34-8). Following these ideas, high order semi-implicit methods coupled with discontinuous Galerkin (DG) space discretizations on unstructured staggered meshes have been forwarded for compressible and incompressible flows [\[63,](#page-35-20) [4\]](#page-34-10), on dynamic adaptive meshes [\[29,](#page-35-21) [30\]](#page-35-22) and for axially symmetric flows [\[25,](#page-35-23) [38\]](#page-35-24).

It is worth to notice, that the two above approaches, IMEX and semi-implicit, have been generalized under a unified framework in [\[58\]](#page-35-17). This generalization, permits the construction of high order linearly implicit schemes by using the standard formalism of IMEX Runge-Kutta methods. In this paper, we will rely on such methodology to achieve efficient high-order accurate semi-implicit discretizations of the Navier-Stokes system for all Mach number flows.

Recently, an all Mach solver for the 3D Euler equations has been designed [\[68\]](#page-36-3), which is a cell-centered second order accurate finite volume method with IMEX time stepping. An elliptic equation on the pressure is solved at the aid of an iterative Picard algorithm. Once the pressure is computed, it is used for advancing in time the momentum and the total energy. The work has then been extended in [\[12\]](#page-34-11) to the full Navier-Stokes equations with implicit viscosity treatment. In [\[3\]](#page-34-12) the two-dimensional Euler equations are considered at all Mach number regimes, presenting second order IMEX schemes based on the solution of an elliptic equation on the energy. A finite volume solver has been forwarded in [\[28\]](#page-35-25) for inviscid and viscous compressible fluids in two space dimensions at high and low Mach flows. There, an elliptic equation on the enthalpy allows to treat also general equations of state (EOS) that link the internal energy with the density and the enthalpy. A similar approach based on the solution of a pressure wave equation can be found in [\[23\]](#page-34-9), where ideal gas and general cubic EOS are considered. The aforementioned references are at most second order accurate in space and time. High order semi-implicit schemes for the isentropic Euler equations on two-dimensional Cartesian meshes are described in [\[59\]](#page-35-18), where the seminal work presented in [\[58\]](#page-35-17) is applied to low Mach flows.

In this article we present a third-order semi-implicit scheme on collocated Cartesian grids for the solution of the compressible Navier-Stokes equations at all Mach numbers. The flux splitting proposed in [\[27\]](#page-35-19) requires an implicit sub-system to be solved for the pressure. The novel semi-implicit discretization proposed in this work splits the kinetic energy contribution as well as the enthalpy fluxes in the energy equation into an explicit and an implicit part. Differently from what presented in [23, 68], no iterative solvers are needed anymore and the solution of the pressure system is directly computed. General equations of state yield a mildly nonlinear system that can then be handled relying on a nested Newton technique developed in [13]. An efficient CWENO reconstruction that is carried out dimension-by-dimension is used for achieving high order of accuracy in space, while IMEX Runga-Kutta time stepping is adopted following [58]. Finally, a quadrature-free finite volume scheme is developed for the convective explicit part of the system, which also contains the viscous fluxes of the Navier-Stokes equations. The scheme is also proven to be asymptotically preserving, hence it recovers the limit model in the stiff limit. To overcome the appearance of spurious oscillations due to high order discretization in time, a new limiting strategy is proposed which is based on a convex combination between high and first order numerical solution. Applications to inviscid and viscous compressible flows in low and high Mach regimes are shown, demonstrating the accuracy and the robustness of the novel method.

The rest of the paper is organized as follows. In Section 2 the compressible Navier-Stokes equations are described. We also introduce the low Mach scaling of the governing partial differential equations (PDE) and the continuous model retrieved in the stiff limit. The novel semi-implicit scheme is detailed in Section 3. Firstly, a semi-discretization in time is explained, then the fully discrete first order scheme is derived. Details of the high order extension in time and in space are then given and finally the quadrature-free finite volume scheme for the treatment of explicit terms is presented. The limiting strategy adopted to reduce spurious oscillations when high order time discretizations are adopted is detailed at the end of this section. Numerical convergence studies and applications to a wide set of test problems is shown in Section 4. A concluding section finalizes the article where we draw some conclusions and present an outlook to future research.

# 2. Governing equations

Let  $\Omega \in \mathbb{R}^d$  represent a bounded domain in the space dimension  $d \in \{1, 2, 3\}$ , which is defined by spatial position vector  $\mathbf{x} \in \Omega$  and time variable  $t \in \mathbb{R}_+$ . The compressible Navier-Stokes equations write

$$\frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho \mathbf{u} \\ \rho E \end{pmatrix} + \nabla \cdot \begin{pmatrix} \rho \mathbf{u} \\ \rho \mathbf{u} \otimes \mathbf{u} + p \mathbf{I} \\ \rho k \mathbf{u} + h \rho \mathbf{u} \end{pmatrix} = \nabla \cdot \begin{pmatrix} 0 \\ \sigma \\ \sigma \mathbf{u} + \lambda \nabla T \end{pmatrix}, \tag{1}$$

with **I** being the identity matrix.  $\rho(t, x) > 0$  is the density of the fluid,  $\mathbf{u}(t, x) \in \mathbb{R}^d$  denotes the velocity vector,  $\rho E(t, x)$  represents the total energy with the specific kinetic energy k, the specific internal energy e and the specific enthalpy h. The fluid pressure is denoted by p(t, x) > 0 and T(t, x) > 0 refers to the fluid temperature with  $\lambda$  representing the thermal conductivity. In  $\mathbb{R}^3$  one has  $\mathbf{x} = (x, y, z)$  and  $\mathbf{u} = (u, v, w)$ . The right hand side of system (1) is conveniently formulated by introducing the stress tensor  $\sigma$  which under Stokes hypothesis is

$$\sigma = \mu \left( \nabla \boldsymbol{u} + \nabla \boldsymbol{u}^{\mathsf{T}} \right) - \frac{2}{3} \left( \mu \nabla \cdot \boldsymbol{u} \right) \mathbf{I}, \tag{2}$$

where  $\mu$  is the viscosity of the fluid. A thermal equation of state  $p = p(T, \rho)$  and a caloric equation of state  $e = e(T, \rho)$  are required to close system (1). Typically, the temperature is canceled from these two equations of state, yielding one single relation of the form  $e = e(p, \rho)$ , which will be adopted in this work. We assume that the internal energy is a non-negative and non-decreasing function of the fluid pressure p. Furthermore, the relation between the viscosity coefficient and the fluid temperature is governed by Sutherland's law, that is

$$\mu(T) = \mu_0 \left(\frac{T}{T_0}\right)^{\beta} \frac{T_0 + s}{T + s},\tag{3}$$

with parameters  $\mu_0$ ,  $T_0$ ,  $\beta$  and s. Notice that constant viscosity is retrieved if  $\beta = 1$  and s = 0. The ratio of specific heats of the gas at constant pressure  $c_p$  and at constant volume  $c_v$  is  $\gamma = c_p/c_v$  and the specific heat at constant volume  $c_v$  is determined by  $c_v = R/(\gamma - 1)$  with R being the gas constant which is assumed to be R = 0.4. Finally, the specific kinetic energy k and the specific enthalpy k are given by the following relations:

$$k = \frac{1}{2}u^2, \qquad h = e + \frac{p}{\rho}. \tag{4}$$

Let observe that the total energy flux in (1) is written as

$$\mathbf{u}(\rho E + p) = \rho k \, \mathbf{u} + \rho h \, \mathbf{u},\tag{5}$$

according to [27], thus introducing a *flux splitting* which will be extremely important for the numerical methods developed in this work.

# 2.1. Ideal gas EOS

If an ideal gas is considered, the thermal and caloric equation of state (EOS) are given by

$$\frac{p}{\rho} = RT, \qquad e = c_{\nu}T. \tag{6}$$

The temperature can be eliminated using both expressions in (6), thus leading to an equation of state of the form  $e(p,\rho)$ , that is

$$e(p,\rho) = \frac{p}{(\gamma - 1)\rho}. (7)$$

Here the relation between pressure p and internal energy e is linear

#### 2.2. Redlich-Kwong EOS

Let now consider a general cubic equation of state, which according to [67] takes the form

$$p(T,\rho) = \frac{RT}{v-b} - \frac{a(T)}{(v-br_1)(v-br_2)}.$$
 (8)

Here,  $v = 1/\rho$  is the specific volume, b represents the co-volume and  $r_1, r_2$  are two parameters. The function a(T) is related to the attraction term in the EOS. The caloric equation of state which corresponds to (8) writes [67]

$$e(T,\rho) = c_v T + \frac{a(T) - Ta'(T)}{b} U(v, b, r_1, r_2), \tag{9}$$

with

$$a'(T) = \frac{da(T)}{dT}, \qquad U(v, b, r_1, r_2) = \frac{1}{r_1 - r_2} \ln\left(\frac{v - br_1}{v - br_2}\right). \tag{10}$$

Different equation of states can be derived by appropriate choices of the parameters in (8)-(9). For instance, the Redlich-Kwong EOS is obtained by setting  $r_1 = 0$  and  $r_2 = -1$ , with the attraction term given by  $a(T) = 1/(2\sqrt{T})$ . In this case the EOS yields a nonlinear relation between internal energy and pressure. In order to compute a function of the form  $e(p,\rho)$ , we first need to determine the temperature from the thermal equation of state (8). In our approach we solve the nonlinear equation  $p(T,\rho)$  numerically relying on a simple and efficient Newton method. Once the temperature is known, it can be inserted into the caloric EOS (9) to obtain the relation  $e(p,\rho)$ .

# 2.3. Scaling of the Navier-Stokes equations

The governing equations (1) can be rescaled relying on the following scaled variables:

$$\tilde{\rho} = \rho/\rho_0, \qquad \tilde{\boldsymbol{u}} = \boldsymbol{u}/u_0, \qquad \tilde{p} = p/p_0, \qquad \tilde{E} = \frac{\rho_0 E}{p_0}, \qquad \tilde{T} = T/T_0, \qquad \tilde{\boldsymbol{x}} = \boldsymbol{x}/x_0, \qquad \tilde{t} = t/t_0,$$
 (11)

where  $\rho_0$ ,  $p_0$ ,  $x_0$ ,  $t_0$ ,  $u_0 = x_0/t_0$  and  $T_0 = p_0/\rho_0$  are typical values referred to the problems under consideration. Furthermore, let  $\tilde{\mu} = \mu/\mu_0$  and  $\tilde{\lambda} = \lambda/\lambda_0$  be the rescaled coefficients for the viscosity and the thermal conductivity, respectively, and let us introduce a stiffness parameter  $\varepsilon$ 

$$\varepsilon = \frac{\rho_0 u_0^2}{p_0},\tag{12}$$

which is related to the global Mach number  $M = u_0/c$  and characterizes the flow and the nondimensionalisation. The sound speed c is then given by  $c^2 = \left(\frac{\partial p}{\partial \rho}\right)_s$  with s representing the entropy. Using the definitions (11)-(12) and omitting the tildes, the rescaled Navier-Stokes equations read

$$\frac{\partial}{\partial t} \begin{pmatrix} \rho \\ \rho \mathbf{u} \\ \rho E \end{pmatrix} + \nabla \cdot \begin{pmatrix} \rho \mathbf{u} \\ \rho \mathbf{u} \otimes \mathbf{u} + \frac{1}{\varepsilon} \rho \mathbf{I} \\ \varepsilon \rho k \mathbf{u} + h \rho \mathbf{u} \end{pmatrix} = \nabla \cdot \begin{pmatrix} 0 \\ \sigma \\ \varepsilon \sigma \mathbf{u} + \lambda \nabla T \end{pmatrix}. \tag{13}$$

The right hand side of (13) can be expanded at the aid of the Reynolds and Prandtl numbers

$$Re = \frac{u_0 x_0}{v}, \qquad Pr = \frac{\mu \gamma c_v}{\lambda},$$
 (14)

with the kinematic viscosity  $\nu = \frac{\mu}{\rho}$ , hence yielding

$$\nabla \cdot \begin{pmatrix} 0 \\ \sigma \\ \varepsilon \, \sigma \boldsymbol{u} + \lambda \nabla T \end{pmatrix} = \nabla \cdot \begin{pmatrix} 0 \\ \frac{1}{Re} \nabla \cdot \left[ \rho \left( \nabla \boldsymbol{u} + \nabla \boldsymbol{u}^{\top} \right) - \frac{2}{3} \left( \rho \nabla \cdot \boldsymbol{u} \right) \mathbf{I} \right] \\ \frac{\varepsilon}{Re} \nabla \cdot \left[ \left( \rho \left( \nabla \boldsymbol{u} + \nabla \boldsymbol{u}^{\top} \right) - \frac{2}{3} \left( \rho \nabla \cdot \boldsymbol{u} \right) \mathbf{I} \right) \cdot \boldsymbol{u} \right] + \frac{c_{p}}{Re \cdot Pr} \nabla T \end{pmatrix}. \tag{15}$$

Looking at the rescaled equations (13) it is evident that the stiffness is originated in the momentum equations by the pressure waves. As a consequence, the energy equation turns out to be stiff as well. In particular, the energy can be decomposed into a pressure and a kinetic part, corresponding to internal and kinetic energy contribution. In the stiff regime the pressure evolves very fast, implying the same for the component of the energy related to the pressure, i.e. the internal energy e.

# 2.4. Low Mach limit of the Navier-Stokes equations

Let us now investigate the limit of the Navier-Stokes equations in the case  $\varepsilon \to 0$ . In [15] the low Mach limit is studied in a bounded domain, while a fully three dimensional space is considered in [2]. Here, we only briefly recall the formal limit obtained with an ideal gas EOS (7). On the boundary  $\partial\Omega$  of the computational domain the following conditions must be imposed:

$$\mathbf{u}(t, \mathbf{x}) \cdot \mathbf{n} = 0, \quad \frac{\partial T}{\partial n}(t, \mathbf{x}) = 0, \quad \forall \mathbf{x} \in \partial \Omega, \ t > 0,$$
 (16)

with **n** denoting the unit outward normal vector to the boundary and *n* its direction. The limit for  $\varepsilon \to 0$  of system (1) then writes [2]

$$\partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0, \tag{17}$$

$$\partial_t(\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \nabla p_1 = \nabla \cdot \boldsymbol{\sigma},\tag{18}$$

$$\gamma \nabla \cdot \boldsymbol{u} = (\gamma - 1) \nabla \cdot \left( \frac{\lambda}{R} \nabla \left( \frac{1}{\rho} \right) \right), \tag{19}$$

assuming that the limit pressure  $p_1 = \lim_{\varepsilon \to 0} \frac{1}{\varepsilon} (p - p_0)$  exists. Notice that if we set  $\mu = 0$  and  $\lambda = 0$ , that is viscous forces and thermal conductivity are neglected, the well known low Mach limit for the compressible Euler equations is retrieved. Specifically, this implies a divergence free condition on the velocity field, i.e.  $\nabla \cdot \boldsymbol{u} = 0$ , which derives from the energy equation. In the case of the Navier-Stokes equations, in the low Mach limit the fluid is no more incompressible because of large temperature variations and heat conduction effects.

Regardless the viscous or inviscid property of the fluid, in the low Mach regime the sound speed is much bigger than the fluid velocity, thus it corresponds to small values of  $\varepsilon$  in (13). From the numerical viewpoint, the maximum admissible time step  $\Delta t = t^{n+1} - t^n$  for fully explicit schemes is given by a CFL-type stability condition that writes

$$\Delta t \le \text{CFL} \min_{\Omega} \left( \frac{\max(|u \pm c/\sqrt{\varepsilon}|)}{\Delta x} + \frac{\max(|v \pm c/\sqrt{\varepsilon}|)}{\Delta y} + \frac{\max(|w \pm c/\sqrt{\varepsilon}|)}{\Delta z} + \max\left(\frac{\lambda_{v}}{\Delta x^{2}} + \frac{\lambda_{v}}{\Delta y^{2}} + \frac{\lambda_{v}}{\Delta z^{2}}\right) \right)^{-1}, \quad (20)$$

where  $\Delta x$ ,  $\Delta y$ ,  $\Delta z$  are the characteristic mesh spacing along each spatial direction in 3D. The eigenvalues of the viscous sub-system for an ideal gas are given according to [21] by

$$\lambda_{\nu} = \max\left(\frac{4}{3}\frac{\mu}{\rho}, \frac{\gamma\mu}{Pr\rho}\right). \tag{21}$$

 $\Delta t$  is of order  $\sqrt{\varepsilon}$  and tends to 0 with  $\varepsilon$ , thus dictating severe limits in the maximum size of the time step. Furthermore, even if this constraint is satisfied and the scheme runs with very small time steps, explicit schemes are not capable to capture the correct asymptotic regime as discussed in [36, 35, 20].

#### 3. Numerical scheme

For the sake of simplicity we present the discretization for the compressible Euler equations, that are retrieved by neglecting the terms on the right hand side of system (1). The compressible Navier-Stokes model will then be included with fully explicit discretization of the viscous forces in the momentum equations and the work of the viscous stress tensor in the energy equation, while keeping untouched the semi-implicit scheme developed for the inviscid hydrodynamics model. For an ideal gas (7) the heat flux in the energy equation can be treated implicitly because temperature can be written as  $T = p/(R\rho)$ , therefore it can be easily embedded in the semi-implicit solver for the pressure.

# 3.1. First order semi-discrete scheme in time

The time discretization is based on a *semi-implicit* approach which leads to the following scheme:

$$\frac{\rho^{n+1} - \rho^n}{\Delta t} + \nabla \cdot (\rho \mathbf{u})^n = 0, \tag{22}$$

$$\frac{(\rho \boldsymbol{u})^{n+1} - (\rho \boldsymbol{u})^n}{\Delta t} + \nabla \cdot ((\rho \boldsymbol{u})^n \otimes (\rho \boldsymbol{u})^n) + \frac{1}{\varepsilon} \nabla p^{n+1} = 0, \tag{23}$$

$$\frac{(\rho e)^{n+1} + \frac{\varepsilon}{2} \frac{(\rho u)^n}{2\rho^n} (\rho u)^{n+1} - (\rho E)^n}{\Delta t} + \nabla \cdot (\varepsilon \rho k u)^n + \nabla \cdot \left( h^n (\rho u)^{n+1} \right) = 0, \tag{24}$$

(25)

where the kinetic energy in the total energy definition splits into an explicit and an implicit contribution, namely

$$(\rho E)^{n+1} := (\rho e)^{n+1} + \varepsilon \frac{(\rho u)^n}{2\rho^n} (\rho u)^{n+1}. \tag{26}$$

The scheme (22)-(24) is written in flux form for all variables, hence it is locally and globally conservative. The terms involving pressure are treated implicitly, while the convective part of the system is discretized explicitly. This allows the time step to be free from any restriction based on acoustic waves, that is a desirable property in the low Mach regime when  $\varepsilon \to 0$ . As a consequence, the time step must satisfy a milder CFL stability condition which is based only on the material speed of the flow |u|, that is

$$\Delta t \le \mathrm{CFL} \frac{\min_{\Omega}(\Delta x, \Delta y, \Delta z)}{\max_{\Omega}(|\boldsymbol{u}|)}, \tag{27}$$

with CFL < 1. The algorithm for the solution of the semi-implicit numerical scheme (22)-(24) is made of the following steps.

- 1. The density equation can be solved explicitly, thus  $\rho^{n+1}$  is readily obtained from (22).
- 2. The momentum equation (23) is then inserted into the energy equation (24) yielding an *elliptic* equation on the pressure:

$$(\rho e)^{n+1} + \varepsilon \frac{(\rho u)^n}{2\rho^n} \left( (\rho u)^n - \Delta t \nabla \cdot ((\rho u)^n \otimes (\rho u)^n) - \frac{\Delta t}{\varepsilon} \nabla p^{n+1} \right) =$$

$$(\rho E)^n - \Delta t \nabla \cdot (\varepsilon \rho k u)^n - \Delta t \nabla \cdot \left( h^n \left( (\rho u)^n - \Delta t \nabla \cdot ((\rho u)^n \otimes (\rho u)^n) - \frac{\Delta t}{\varepsilon} \nabla p^{n+1} \right) \right).$$
(28)

Notice that a semi-implicit discretization of the enthalpy flux in the energy equation (24) leads to an explicit evaluation of the enthalpy, i.e.  $h^n$ , and an implicit treatment of the momentum, that is  $(\rho u)^{n+1}$ . Shifting the unknowns on the left hand side and multiplying by the stiffness factor  $\varepsilon$ , the pressure wave equation (28) writes

$$\varepsilon(\rho e)^{n+1} + \varepsilon \frac{\Delta t}{2} \frac{(\rho \mathbf{u})^n}{\rho^n} \nabla p^{n+1} - \Delta t^2 \left( \nabla \cdot h^n \nabla p^{n+1} \right) = \varepsilon \left[ (\rho E)^* - \varepsilon \frac{\Delta t}{2} \frac{(\rho \mathbf{u})^n}{\rho^n} (\rho \mathbf{u})^* - \Delta t \nabla \cdot (h^n (\rho \mathbf{u})^*) \right], \tag{29}$$

with the explicit quantities

$$(\rho E)^* = (\rho E)^n - \Delta t \nabla \cdot (\varepsilon \rho k \mathbf{u})^n, \qquad (30)$$

$$(\rho \mathbf{u})^* = (\rho \mathbf{u})^n - \Delta t \nabla \cdot ((\rho \mathbf{u})^n \otimes (\rho \mathbf{u})^n). \tag{31}$$

The internal energy  $(\rho e)^{n+1}$  must now be written in terms of the new pressure  $p^{n+1}$  using the equation of state.

Ideal gas EOS. According to (7) for a perfect gas the internal energy is given by

$$(\rho e)^{n+1} = \frac{p^{n+1}}{\gamma - 1},\tag{32}$$

thus the elliptic equation (29) constitutes a linear system that can be directly solved.

*General EOS.* For a general equation of state the relation between internal energy and pressure might be non-linear, hence requiring the solution of the following nonlinear equation for the pressure:

$$g(p^{n+1}) = \mathcal{P}(p^{n+1}) + \mathcal{R} p^{n+1} - b^n = 0, \tag{33}$$

with the definitions

$$\mathcal{R} p^{n+1} := \varepsilon \frac{\Delta t}{2} \frac{(\rho \mathbf{u})^n}{\rho^n} \nabla p^{n+1} - \Delta t^2 \left( \nabla \cdot h^n \nabla p^{n+1} \right), \tag{34}$$

$$b^{n} := \varepsilon \left[ (\rho E)^{*} - \varepsilon \frac{\Delta t}{2} \frac{(\rho \mathbf{u})^{n}}{\rho^{n}} (\rho \mathbf{u})^{*} - \Delta t \nabla \cdot (h^{n} (\rho \mathbf{u})^{*}) \right]. \tag{35}$$

The term  $\mathcal{P}(p^{n+1})$  contains the nonlinearity of (29) due to the EOS (9) for the internal energy  $(\rho e)^{n+1}$ . Recall that the new density  $\rho^{n+1}$  is already known thanks to (22). A Newton method is then used for solving the *piecewise linear* equation (33), along the lines of the algorithm presented in [13]. The solution for the new pressure is iteratively obtained as

$$g(p^{n+1,k+1}) = g(p^{n+1,k}) + \Delta p^k \frac{dg(p^{n+1,k})}{p^{n+1,k}} = 0,$$
(36)

with k denoting the iteration index and  $\Delta p^k = (p^{n+1,k+1} - p^{n+1,k})$ . In practice, equation (36) is directly solved for  $\Delta p^k$ , then the new pressure at the next Newton iteration is given by  $p^{n+1,k+1} = p^{n+1,k} - \Delta p^k$ . The Newton method stops when the prescribed tolerance  $\delta = 10^{-10}$  has been reached, e.g.  $\Delta p^k < \delta$ .

- 3. The new pressure  $p^{n+1}$  is used in (23) to compute the momentum  $(\rho u)^{n+1}$  at the next time level.
- 4. Finally, the total energy is simply updated relying on (26), which ensures thermodynamic compatibility between the new pressure  $p^{n+1}$  and momentum  $(\rho u)^{n+1}$ .

For the full Navier-Stokes system, the viscous contribution  $\sigma$  in the momentum equation as well as the work of the viscous stress tensor  $\sigma \cdot u$  in the energy equation are discretized explicitly and are formally embedded in the explicit quantities  $(\rho u)^*$  and  $(\rho E)^*$ , respectively. If an ideal gas is considered, an implicit discretization is likely to be assumed for the temperature gradient in the energy equation. Since temperature can be easily written in terms of pressure, i.e.  $T^{n+1} = p^{n+1}/(R\rho^{n+1})$ , this contribution is added to the pressure wave equation (29) and implicitly solved.

Asymptotic preserving property. The limit of the governing equations (13) is given when  $\varepsilon \to 0$ . The expansion of a generic variable m in powers of the stiffness parameter  $\varepsilon$ , that is  $m = m_{(0)} + \varepsilon m_{(1)} + \varepsilon^2 m_{(2)} + \ldots$ , is applied to all variables involved in the governing rescaled Navier-stokes model, which is here assumed with  $\mu = \lambda = 0$ . These expressions are then inserted into the semi-discrete scheme (22)-(24) and only leading order terms are considered, thus obtaining

$$\frac{\rho_{(0)}^{n+1} - \rho_{(0)}^n}{\Delta t} + \nabla \cdot (\rho \mathbf{u})_{(0)}^n = 0, \tag{37}$$

$$\frac{(\rho \boldsymbol{u})_{(0)}^{n+1} - (\rho \boldsymbol{u})_{(0)}^{n}}{\Delta t} + \nabla \cdot \left( (\rho \boldsymbol{u})_{(0)}^{n} \otimes (\rho \boldsymbol{u})_{(0)}^{n} \right) + \nabla p_{(1)}^{n+1} = 0, \tag{38}$$

$$\frac{(\rho e)_{(0)}^{n+1} - (\rho e)_{(0)}^{n}}{\Delta t} + \nabla \cdot \left( h_{(0)}^{n} \rho_{(0)}^{n+1} \boldsymbol{u}_{(0)}^{n+1} \right) = 0, \tag{39}$$

$$\nabla p_{(0)}^{n+1} = 0. (40)$$

The incompressibility constraint

$$\nabla \cdot \boldsymbol{u}_{(0)}^{n+1} = O(\Delta t), \tag{41}$$

with  $O(\Delta t)$  independent of  $\varepsilon$ , must now be retrieved in order to demonstrate the asymptotic preserving property of the scheme. Indeed, this recovers the limit of the energy equation at the continuous level (19). To ease the notation the subscript (0) is removed and all terms estimated by  $C\Delta t$  with the constant  $C \neq C(\varepsilon)$  will simply be addressed with  $O(\Delta t)$ . Notice that from (37)-(38) one has

$$\rho^{n+1} = \rho^n + O(\Delta t), \qquad u^{n+1} = u^n + O(\Delta t). \tag{42}$$

Recalling that  $e = e(\rho, p)$  according to (4), the first term in the limit energy equation (39) can be written as

$$\frac{(\rho e)^{n+1} - (\rho e)^n}{\Delta t} = \frac{1}{\Delta t} \left( \frac{\partial (\rho e)}{\partial \rho} (\rho^{n+1} - \rho^n) + \frac{\partial (\rho e)}{\partial p} (p^{n+1} - p^n) + O(\rho^{n+1} - \rho^n)^2 + O(p^{n+1} - p^n)^2 \right)$$

$$= \frac{\partial (\rho e)}{\partial \rho} \frac{\rho^{n+1} - \rho^n}{\Delta t} + O(\Delta t), \tag{43}$$

where  $p^{n+1} - p^n = 0$  because of (40) which implies that pressure is constant when  $\varepsilon \to 0$ . Now, using the limit continuity equation (37) and the observation (42), from expression (43) we get

$$\frac{(\rho e)^{n+1} - (\rho e)^n}{\Delta t} = \frac{\partial(\rho e)}{\partial \rho} \frac{\rho^n - \Delta t \nabla(\rho^n \mathbf{u}^n) - \rho^n}{\Delta t} + O(\Delta t)$$

$$= -\frac{\partial(\rho e)}{\partial \rho} (\mathbf{u}^n \cdot \nabla \rho^n + \rho^n \nabla \cdot \mathbf{u}^n) + O(\Delta t)$$

$$= -\frac{\partial(\rho e)}{\partial \rho} (\mathbf{u}^{n+1} \cdot \nabla \rho^n + \rho^n \nabla \cdot \mathbf{u}^{n+1}) + O(\Delta t). \tag{44}$$

Since  $(\rho h) = (\rho e) + p$ , the divergence flux in (39) is rewritten at the aid of (42) as

$$\nabla \cdot \left(h^{n} \rho^{n+1} \boldsymbol{u}^{n+1}\right) = \boldsymbol{u}^{n+1} \cdot \nabla (h^{n} \rho^{n+1}) + h^{n} \rho^{n+1} \nabla \cdot \boldsymbol{u}^{n+1}$$

$$= \boldsymbol{u}^{n+1} \cdot \nabla (h^{n} \rho^{n}) + h^{n} \rho^{n} \nabla \cdot \boldsymbol{u}^{n+1} + O(\Delta t)$$

$$= \boldsymbol{u}^{n+1} \cdot \nabla ((\rho e)^{n} + p^{n}) + h^{n} \rho^{n} \nabla \cdot \boldsymbol{u}^{n+1} + O(\Delta t)$$

$$= \boldsymbol{u}^{n+1} \left(\frac{\partial (\rho e)}{\partial \rho} \nabla \rho^{n} + \frac{\partial (\rho e)}{\partial \rho} \nabla p^{n}\right) + h^{n} \rho^{n} \nabla \cdot \boldsymbol{u}^{n+1} + O(\Delta t)$$

$$= \frac{\partial (\rho e)}{\partial \rho} \boldsymbol{u}^{n+1} \cdot \nabla \rho^{n} + h^{n} \rho^{n} \nabla \cdot \boldsymbol{u}^{n+1} + O(\Delta t). \tag{45}$$

The energy equation (39) can therefore be formulated by summing up the terms (44)-(45), which yields

$$-\frac{\partial(\rho e)}{\partial\rho}\left(\boldsymbol{u}^{n+1}\cdot\nabla\rho^{n}+\rho^{n}\nabla\cdot\boldsymbol{u}^{n+1}\right)+\frac{\partial(\rho e)}{\partial\rho}\boldsymbol{u}^{n+1}\cdot\nabla\rho^{n}+h^{n}\rho^{n}\nabla\cdot\boldsymbol{u}^{n+1}=O(\Delta t)$$

$$-\frac{\partial(\rho e)}{\partial\rho}\rho^{n}\nabla\cdot\boldsymbol{u}^{n+1}+((\rho e)^{n}+p^{n})\nabla\cdot\boldsymbol{u}^{n+1}=O(\Delta t)$$

$$\left(-(\rho^{n})^{2}\frac{\partial e}{\partial\rho}+p^{n}\right)\nabla\cdot\boldsymbol{u}^{n+1}=O(\Delta t). \tag{46}$$

The incompressibility constraint (41) is therefore satisfied because  $\left(-(\rho^n)^2 \frac{\partial e}{\partial \rho} + p^n\right) \neq 0$ . For an ideal gas EOS it holds  $\frac{\partial e}{\partial \rho} = -\frac{p}{\rho^2(\gamma-1)}$ , thus equation (46) becomes

$$h^n \nabla \cdot \boldsymbol{u}^{n+1} = O(\Delta t). \tag{47}$$

# 3.2. First order fully discrete scheme in space and time

Let us consider a three-dimensional computational domain  $\Omega(x) = [x_{\min}; x_{\max}] \times [y_{\min}; y_{\max}] \times [z_{\min}; z_{\max}]$  which is discretized by a Cartesian grid composed of a total number  $N_e = N_x \times N_y \times N_z$  of cells  $C_{i,j,k}$  with volume  $|C_{i,j,k}| = \Delta x \Delta y \Delta z$ . Specifically, the characteristic mesh sizes are given by

$$\Delta x = \frac{x_{\text{max}} - x_{\text{min}}}{N_x}, \qquad \Delta y = \frac{y_{\text{max}} - y_{\text{min}}}{N_y}, \qquad \Delta z = \frac{z_{\text{max}} - z_{\text{min}}}{N_z}.$$
 (48)

A triple index (i, j, k) referred to each space direction allows a cell to be uniquely identified. The faces in x, y and z direction are referred to as (i + 1/2, j, k), (i, j + 1/2, k) and (i, j, k + 1/2), respectively. The associated normal vectors are the canonical unit vectors, that is  $\mathbf{n}_x = (1, 0, 0)$ ,  $\mathbf{n}_y = (0, 1, 0)$ ,  $\mathbf{n}_z = (0, 0, 1)$ . The cell center is located at point  $\mathbf{x}_{i,j,k} = (x_i, y_j, z_k)$  and a face center is at point  $\mathbf{x}_{i+1/2,j,k} = \left(\frac{x_i + x_{i+1}}{2}, y_j, z_k\right)$ . The spatial discretization is based on *collocated grids*, in which all variables of the governing equations are defined at the cell centers of the control volumes. Implicit fluxes are discretized with finite differences with no numerical dissipation, while we rely on finite volume schemes based on numerical fluxes for the explicit terms.

For the sake of clarity and to improve the readability, the fully discrete scheme will be presented for the onedimensional case, since extension to multiple space dimensions on Cartesian meshes follows straightforward. Let us introduce the explicit operator  $m_i^*$  which applies to a generic cell quantity  $m_i^n$ :

$$m_i^* = m_i^n - \frac{\Delta t}{\Delta r} \left( f_{i+1/2}^m - f_{i-1/2}^m \right). \tag{49}$$

Here,  $f_{i\pm 1/2}^m$  denote the numerical fluxes, that are explicitly given by a Rusanov-type approximate Riemann solver, thus leading to

$$f_{i+1/2}^{m} = \frac{1}{2} \left( f(m_{i+1}^{n}) + f(m_{i}^{n}) \right) - \frac{1}{2} a_{i+1/2}^{n} \left( m_{i+1}^{n} - m_{i}^{n} \right), \qquad a_{i+1/2}^{n} = \max \left( |u_{i+1}^{n}|, |u_{i}^{n}| \right),$$

$$f_{i-1/2}^{m} = \frac{1}{2} \left( f(m_{i}^{n}) + f(m_{i-1}^{n}) \right) - \frac{1}{2} a_{i-1/2}^{n} \left( m_{i}^{n} - m_{i-1}^{n} \right), \qquad a_{i-1/2}^{n} = \max \left( |u_{i}^{n}|, |u_{i-1}^{n}| \right),$$

$$(50)$$

where  $f(\cdot)$  represents the physical flux related to variable m. Notice that the numerical viscosity is chosen to be proportional to the material speed, so that if the Mach number is high the speed of sound is bounded by the fluid velocity, whereas for very low Mach number this choice should be sufficient to guarantee stability with the dissipation  $a_{i+1/2}^n \approx |u|$ .

Now, a spatial discretization of the time semi-discrete scheme (22)-(24) is presented. Let us start following the steps of the algorithm detailed in 3.1.

1. The new density  $\rho^{n+1}$  is immediately available solving the continuity equation (22) with  $m_i^n \equiv \rho_i^n$  in (49):

$$\rho_i^{n+1} = (\rho_i^n)^* \,. \tag{51}$$

2. The next step requires the solution of the elliptic equation for the pressure (29). Two different flux derivative operators need to be discretized, namely  $\frac{\partial p}{\partial x}$  and  $\frac{\partial}{\partial x} \left( h \frac{\partial p}{\partial x} \right)$ . A central finite difference discretization is then given by

$$\left. \frac{\partial p}{\partial x} \right|_{x_i}^{n+1} = \frac{p_{i+1}^{n+1} - p_{i-1}^{n+1}}{2\Delta x} + O(\Delta x^2), \tag{52}$$

$$\frac{\partial}{\partial x} \left( h \frac{\partial p}{\partial x} \right) \Big|_{x_i}^{n,n+1} = \frac{1}{\Delta x^2} \left[ h_{i-1}^n \quad h_i^n \quad h_{i+1}^n \right] \left[ \begin{array}{ccc} 3/4 & -1 & 1/4 \\ 0 & 0 & 0 \\ 1/4 & -1 & 3/4 \end{array} \right] \left[ \begin{array}{c} p_{i-1}^{n+1} \\ p_{i+1}^{n+1} \\ p_{i+1}^{n+1} \end{array} \right] + O(\Delta x^2), \tag{53}$$

which provides up to second order of accuracy. The approximate derivative in (53) is based on a finite difference approach with Lagrange interpolation polynomials of degree 2 on the stencil composed by cells [i-1,i,i+1]. This makes the scheme very compact even on collocated grids, hence achieving the same properties typically related to the usage of staggered meshes. Similar discretization is applied to all derivative operators that are discretized implicitly, namely the pressure gradient in the momentum equation, the implicit part of the kinetic energy and the term  $\nabla \cdot (h^n(\rho u)^*)$  in the energy equation. The pressure wave equation (29) can now be expressed after multiplication by the cell volume  $\Delta x$  as

$$\varepsilon \Delta x \rho_{i}^{n+1} e_{i}^{n+1} + \varepsilon \frac{\Delta t}{4} \frac{(\rho u)_{i}^{n}}{\rho_{i}^{n}} \left( p_{i+1}^{n+1} - p_{i-1}^{n+1} \right) - \frac{\Delta t^{2}}{\Delta x} \left( p_{i-1}^{n+1} \left( \frac{3}{4} h_{i-1}^{n} + \frac{1}{4} h_{i+1}^{n} \right) - p_{i}^{n+1} \left( h_{i-1}^{n} + h_{i+1}^{n} \right) + p_{i+1}^{n+1} \left( \frac{1}{4} h_{i-1}^{n} + \frac{3}{4} h_{i+1}^{n} \right) \right) = \varepsilon \Delta x b_{i}^{n},$$
 (54)

with the known right hand side

$$b_i^n = (\rho E)_i^* - \varepsilon \frac{\Delta t}{2} \frac{(\rho u)_i^n}{\rho_i^n} (\rho u)_i^* - \frac{\Delta t}{2\Delta x} \left( h_{i+1}^n (\rho u)_{i+1}^* - h_{i-1}^n (\rho u)_{i-1}^* \right). \tag{55}$$

Depending on the equation of state, the pressure wave equation (54)-(55) involves either a linear or a nonlinear system that is solved following the ansatz given by (32) or (36), respectively. This allows to compute the new pressure  $p_i^{n+1}$ .

3. The momentum is updated at the next time level as follows:

$$(\rho u)_{i}^{n+1} = (\rho u)_{i}^{*} - \frac{1}{\varepsilon} \frac{\Delta t}{2\Delta x} \left( p_{i+1}^{n+1} - p_{i-1}^{n+1} \right). \tag{56}$$

4. The total energy is then given by

$$(\rho E)_i^{n+1} = \rho_i^{n+1} e^{n+1} + \varepsilon \frac{(\rho u)_i^n}{2\rho_i^n} (\rho u)_i^{n+1}, \tag{57}$$

where the internal energy is computed at the aid of the equation of state  $e = e(\rho, p)$ .

Remark on the discretization of the enthalpy. The enthalpy in the energy flux is discretized explicitly in each control volume, i.e.  $h_i^n$ . Particular care must be taken in order to achieve preservation of constant velocity and pressure flows at the discrete level. Therefore, the enthalpy is not simply evaluated according to the definition (4), but it is discretized as

$$h_i^n = \frac{\rho_i^n h_i^n}{\rho_i^{n+1}},\tag{58}$$

which guarantees structure preserving properties that will be explained in Section 3.3.

*Remark on the compactness of the stencil.* A direct discretization of the total energy equation in [\(13\)](#page-4-0) would lead to

$$(\rho E)_{i}^{n+1} = (\rho E)_{i}^{*} - \frac{\Delta t}{2\Delta x} \left( h_{i+1}^{n} (\rho u)_{i+1}^{n+1} - h_{i-1}^{n} (\rho u)_{i-1}^{n+1} \right)$$

$$= (\rho E)_{i}^{*} - \frac{\Delta t}{2\Delta x} \left[ h_{i+1}^{n} \left( (\rho u)_{i+1}^{*} - \frac{\Delta t}{2\Delta x} \left( p_{i+2}^{n+1} - p_{i}^{n+1} \right) \right) - h_{i-1}^{n} \left( (\rho u)_{i-1}^{*} - \frac{\Delta t}{2\Delta x} \left( p_{i}^{n+1} - p_{i-2}^{n+1} \right) \right) \right]$$

$$:= E_{1}, \qquad (59)$$

where the viscous source terms have been neglected and the discretization of the momentum equation [\(56\)](#page-9-3) has been directly inserted. This is a consequence of the fully discrete approach, in which both momentum and energy equations are first discretized in space and time, and then formal substitution of the discrete momentum into the energy equation yields the scheme [\(59\)](#page-10-1), as done in [\[23,](#page-34-9) [68\]](#page-36-3). It is evident that in this case the stencil is larger, therefore the pressure wave equation [\(28\)](#page-5-2) would involve those cells spanning the interval [*<sup>i</sup>* <sup>−</sup> <sup>2</sup>, *<sup>i</sup>* <sup>−</sup> <sup>1</sup>, *<sup>i</sup>*, *<sup>i</sup>* <sup>+</sup> <sup>1</sup>, *<sup>i</sup>* <sup>+</sup> 2]. On the other hand, the total energy which results from the wave equation for the pressure [\(54\)](#page-9-1) with the discrete operator [\(53\)](#page-9-0) is indeed given by

$$(\rho E)_{i}^{n+1} = (\rho E)_{i}^{*} - \frac{\Delta t}{2\Delta x} \left( h_{i+1}^{n} (\rho u)_{i+1}^{*} - h_{i-1}^{n} (\rho u)_{i-1}^{*} \right) + \frac{\Delta t^{2}}{\Delta x^{2}} \left( p_{i-1}^{n+1} \left( \frac{3}{4} h_{i-1}^{n} + \frac{1}{4} h_{i+1}^{n} \right) - p_{i}^{n+1} \left( h_{i-1}^{n} + h_{i+1}^{n} \right) + p_{i+1}^{n+1} \left( \frac{1}{4} h_{i-1}^{n} + \frac{3}{4} h_{i+1}^{n} \right) \right) := E_{2},$$

$$(60)$$

where we have first performed only a time discretization, then plugged the momentum into the energy equation and finally discretized in space. It is also possible to quantify a kind of correction C*<sup>E</sup>* to retrieve equation [\(59\)](#page-10-1) from [\(60\)](#page-10-2), that is

$$C_{E} = E_{2} - E_{1}$$

$$= \frac{\Delta t^{2}}{\Delta x^{2}} \left( h_{i-1}^{n} p_{i}^{n+1} - h_{i-1}^{n} p_{i-2}^{n+1} + h_{i+1}^{n} p_{i}^{n+1} - h_{i+1}^{n} p_{i+2}^{n+1} \right)$$

$$+ \frac{\Delta t^{2}}{\Delta x^{2}} \left[ 4 p_{i-1}^{n+1} \left( \frac{3}{4} h_{i-1}^{n} + \frac{1}{4} h_{i+1}^{n} \right) - 4 p_{i}^{n+1} (h_{i-1}^{n} + h_{i+1}^{n}) + 4 p_{i+1}^{n+1} \left( \frac{1}{4} h_{i-1}^{n} + \frac{3}{4} h_{i+1}^{n} \right) \right]. \tag{61}$$

We point out that this discretization does not compromise the incompressibility constraint that must be preserved in the low Mach limit, as shown in the previous Section [3.1.](#page-5-4) Indeed, it allows to maintain the stencil more compact, thus improving the computational efficiency for parallel simulations.

# *3.3. Exact preservation of pressure and velocity across a contact discontinuity*

As pointed out in [\[7\]](#page-34-16), a consistent numerical scheme should be able to preserve a constant pressure and a constant velocity field through a discontinuity in the fluid density during the time evolution of the solution. Let us consider a one-dimensional computational domain Ω = [−*xL*; *xR*] which is filled with an ideal fluid assigned with the following initial condition (*<sup>t</sup>* <sup>=</sup> *<sup>t</sup>*0) for all control volumes *<sup>i</sup>* <sup>=</sup> <sup>1</sup>, . . . *<sup>N</sup>x*:

$$\rho_i(t_0) = \rho_i^0 = \begin{cases} \rho_L & x \le x_D \\ \rho_R & x > x_D \end{cases}, \quad u_i(t_0) = u_i^0 = u_0, \quad p_i(t_0) = p_i^0 = p_0, \quad i = 1, \dots N_x,$$
 (62)

with  $x_D$  representing the location of the discontinuity inside the domain and  $\rho_L \neq \rho_R$  being non-negative real numbers. The explicit operator (49) for density, momentum and energy explicitly writes

$$(\rho_i)^* = \rho_i^n - \frac{\Delta t}{\Delta x} \left( f_{i+1/2}^{\rho,n} - f_{i-1/2}^{\rho,n} \right) = \rho_i^{n+1}, \tag{63}$$

$$(\rho u)_{i}^{*} = (\rho u)_{i}^{n} - \frac{\Delta t}{\Delta x} \left( f_{i+1/2}^{q,n} - f_{i-1/2}^{q,n} \right) = \rho_{i}^{n} u_{i}^{n} - \frac{\Delta t}{\Delta x} u_{0} \left( f_{i+1/2}^{\rho,n} - f_{i-1/2}^{\rho,n} \right) = \rho_{i}^{n+1} u_{0}, \tag{64}$$

$$(\rho E)_{i}^{*} = \underbrace{\rho_{i}^{n} e_{i}^{n} + \rho_{i}^{n} k_{i}^{n}}_{=E_{i}^{n}} - \frac{\Delta t}{\Delta x} \frac{u_{0}^{2}}{2} \left( f_{i+1/2}^{\rho,n} - f_{i-1/2}^{\rho,n} \right)$$

$$= \rho_i^n e_i^n + \rho_i^n \frac{(u_i^n)^2}{2} - \frac{\Delta t}{\Delta x} \frac{(u_i^n)^2}{2} \left( f_{i+1/2}^{\rho,n} - f_{i-1/2}^{\rho,n} \right) = \rho_i^n e_i^n + \rho_i^{n+1} \frac{u_0^2}{2}. \tag{65}$$

Assuming an ideal gas EOS (7) and using the enthalpy definition (58) together with the above explicit operators, the right hand side (55) of the pressure system reads

$$b_{i} = \left(\frac{p_{0}}{\gamma - 1} + \rho_{i}^{n+1} \frac{u_{0}^{2}}{2} - \varepsilon \frac{\rho_{i}^{n} u_{0}}{2\rho_{i}^{n}} \rho_{i}^{n+1} u_{0}\right) - \frac{\Delta t}{2\Delta x} \left(\frac{\gamma p_{0}^{n}}{(\gamma - 1)\rho_{i+1}^{n+1}} \rho_{i+1}^{n+1} u_{0} - \frac{\gamma p_{0}^{n}}{(\gamma - 1)\rho_{i-1}^{n+1}} \rho_{i-1}^{n+1} u_{0}\right)$$

$$= \frac{p_{0}}{\gamma - 1} + (1 - \varepsilon)\rho_{i}^{n+1} \frac{u_{0}^{2}}{2}, \tag{66}$$

while the left hand side (54) simply reduces to

$$\frac{\varepsilon \Delta x}{\gamma - 1} p_0 + \varepsilon \frac{\Delta t}{4} \frac{(\rho u)_i^n}{\rho_i^n} (p_0 - p_0) - \left[ \frac{3}{4} h_{i-1}^n + \frac{1}{4} h_{i+1}^n - h_{i-1}^n - h_{i+1}^n + \frac{1}{4} h_{i-1}^n + \frac{3}{4} h_{i+1}^n \right] p_0 = \varepsilon \Delta x b_i,$$

$$\frac{\varepsilon \Delta x}{\gamma - 1} p_0 + 0 + [0] p_0 = \varepsilon \Delta x b_i,$$
(67)

thus  $p_0$  is the solution of the linear system (54)-(55) independently of  $\varepsilon$  and the constant pressure field is preserved. As a consequence, the update of the momentum with (56) writes

$$(\rho u)_i^{n+1} = \rho_i^{n+1} u_0 - \frac{\Delta t}{2 \varepsilon \Delta x} \cdot 0, \tag{68}$$

hence the constant velocity is maintained as well. The test case RP0 in Section 4.2 gives numerical evidences of this property achieved by the novel semi-implicit scheme (54)-(56).

# 3.4. Extension to high order of accuracy

To reduce the effects of numerical dissipation the first order semi-implicit scheme detailed in Section 3.2 is extended to high order of accuracy in space and time. A semi-implicit IMEX discretization [58] is adopted for achieving high order in time, while we rely on a CWENO reconstruction [45] for gaining high accuracy in space. Finally, the numerical scheme for the explicit convective terms is implemented in a new *quadrature-free* formulation.

The governing equations (1) can be cast into a compact and general form that writes

$$\frac{\partial \mathbf{Q}}{\partial t} + \nabla \cdot \mathbf{F} + \nabla p = \mathbf{0},\tag{69}$$

where  $\mathbf{Q} = (\rho, \rho \mathbf{u}, \rho E)$  is the vector of conserved variables and  $\mathbf{F} = \mathbf{F}(\mathbf{Q}, \nabla \mathbf{Q})$  represents the nonlinear flux tensor which includes both convective and viscous fluxes of the Navier-Stokes equations, that is

$$\mathbf{F} = \begin{pmatrix} \rho \mathbf{u} \\ \rho \mathbf{u} \otimes \mathbf{u} - \boldsymbol{\sigma} \\ \rho k \mathbf{u} + h \rho \mathbf{u} - \boldsymbol{\sigma} \mathbf{u} - \lambda \nabla T \end{pmatrix}. \tag{70}$$

## 3.4.1. High order in time

Following [58], the governing PDE are written under the form of an autonomous system, that is

$$\frac{\partial \mathbf{Q}}{\partial t} = \mathcal{H}(\mathbf{Q}(t), \mathbf{Q}(t)), \quad \forall t > t_0, \quad \text{with} \quad \mathbf{Q}(t_0) = \mathbf{Q}_0,$$
 (71)

with the initial condition  $\mathbf{Q}_0$  defined at time  $t_0$ . The function  $\mathcal{H}$  represents the spatial approximation of the terms  $\nabla \cdot \mathbf{F} + \nabla p$  in (69). An explicit treatment is assumed for the first argument of  $\mathcal{H}$  denoted with  $\mathbf{Q}_E$ , whereas an implicit discretization is adopted for the second argument referred to as  $\mathbf{Q}_I$ , thus obtaining a partitioned system with  $\mathbf{Q} = (\mathbf{Q}_E, \mathbf{Q}_I)$ , hence

$$\begin{cases} \frac{\partial \mathbf{Q}_E}{\partial t} = \mathcal{H}(\mathbf{Q}_E, \mathbf{Q}_I) \\ \frac{\partial \mathbf{Q}_I}{\partial t} = \mathcal{H}(\mathbf{Q}_E, \mathbf{Q}_I) \end{cases}$$
(72)

where the number of unknowns has been doubled. However, for a specific choice of time discretizations and for autonomous systems this duplication is indeed only apparent [58]. The Navier-Stokes equations with the flux splitting (5) fulfill the formalism (72), i.e.

$$\mathcal{H}(\mathbf{Q}_{E}, \mathbf{Q}_{I}) = \begin{cases} (\rho \mathbf{u})_{E} \\ (\rho \mathbf{u} \otimes \mathbf{u})_{E} + p_{I} - \sigma_{E} \\ \rho_{I}(k\mathbf{u})_{E} + \rho_{I}(h\mathbf{u})_{E} - (\sigma \mathbf{u})_{E} - \lambda \nabla T_{I} \end{cases}, \tag{73}$$

where  $\mathbf{Q}_E = (\rho_E, (\rho \mathbf{u})_E, (\rho E)_E)$  and  $\mathbf{Q}_I = (\rho_I, (\rho \mathbf{u})_I, (\rho E)_I)$ . High order in time is achieved making use of implicit-explicit (IMEX) Runge-Kutta schemes [55], that are multi-step methods based on s stages and typically represented with the double Butcher tableau:

$$\begin{array}{c|c}
\tilde{c} & \tilde{A} \\
\hline
& \tilde{b}^{\top}
\end{array} \qquad \frac{c & A}{b^{\top}}, \tag{74}$$

with the matrices  $(\tilde{A}, A) \in \mathbb{R}^{s \times s}$  and the vectors  $(\tilde{c}, c, \tilde{b}, b) \in \mathbb{R}^{s}$ . The tilde symbol refers to the explicit scheme and matrix  $\tilde{A} = (\tilde{a}_{ij})$  is a lower triangular matrix with zero elements on the diagonal, while  $A = (a_{ij})$  is a triangular matrix which accounts for the implicit scheme, thus having non-zero elements on the diagonal. Here, we adopt IMEX schemes with  $\tilde{b} = b$  and the stiffly accurate property in the implicit part, that is crucial for assuring asymptotic consistency and accuracy of the scheme [60]. Applying the partitioned Runge-Kutta method to (73) under the assumption that the system is autonomous, only one set of stage fluxes needs to be computed and the fluxes at each stage  $i = 1, \ldots, s$  can be evaluated as

$$k_i = \mathcal{H}\left(\mathbf{Q}_E^n + \Delta t \sum_{i=1}^s \tilde{a}_{ij} k_j, \quad \mathbf{Q}_I^n + \Delta t \sum_{i=1}^s a_{ij} k_j\right), \qquad 1 \le i \le s.$$
 (75)

A semi-implicit IMEX Runge-Kutta method is obtained as follows. Let us first set  $\mathbf{Q}_E^n = \mathbf{Q}_I^n = \mathbf{Q}^n$ , then the stage fluxes for  $i = 1, \dots, s$  are calculated as

$$\mathbf{Q}_E^i = \mathbf{Q}_E^n + \Delta t \sum_{j=1}^{i-1} \tilde{a}_{ij} k_j, \qquad 2 \le i \le s,$$
(76a)

$$\tilde{\mathbf{Q}}_{I}^{i} = \mathbf{Q}_{E}^{n} + \Delta t \sum_{i=1}^{i-1} a_{ij} k_{j}, \qquad 2 \le i \le s, \tag{76b}$$

$$k_i = \mathcal{H}\left(\mathbf{Q}_E^i, \tilde{\mathbf{Q}}_I^i + \Delta t \, a_{ii} \, k_i\right), \qquad 1 \le i \le s. \tag{76c}$$

Finally, the numerical solution is updated with

$$\mathbf{Q}^{n+1} = \mathbf{Q}^n + \Delta t \sum_{i=1}^s b_i k_i. \tag{77}$$

Notice that equation (76c) implies an implicit step with the solution of a system for  $k_i$ , that corresponds to the pressure wave equation (54)-(55). The final update of the solution (77) is done using the implicit weights  $b^{T}$  that are assumed to be equal to the explicit ones  $\tilde{b}^{T}$ . Furthermore, the stage fluxes  $k_i$  in (75) are the same for both explicit and implicit conserved vectors  $\mathbf{Q}_E$  and  $\mathbf{Q}_I$ , therefore the system is actually not doubled, since there is indeed only one set of numerical solution.

The IMEX schemes used in this work have been developed in [55, 54] and are listened hereafter. Stiffly accurate schemes are addressed with SA, while SSP stands for Strong Stability Preserving methods, which perform better if shock waves or strong discontinuities appear in the flow. Each scheme is described with a triplet  $(s, \tilde{s}, p)$  which characterizes the number s of stages of the implicit scheme, the number  $\tilde{s}$  of stages of the explicit scheme and the order p of the resulting scheme.

• SP(1,1,1)

$$\begin{array}{c|ccccccccccccccccccccccccccccccccccc$$

• SA SSP(3,3,2)

$$\begin{array}{c|ccccccccccccccccccccccccccccccccccc$$

• SA DIRK(3,4,3)

| 0        | 0        | 0        | 0         | 0 | δ        | δ | 0        | 0         | 0        |
|----------|----------|----------|-----------|---|----------|---|----------|-----------|----------|
| $\delta$ | δ        | 0        | 0         | 0 | $\delta$ | 0 | $\delta$ | 0         | 0        |
| 0.717933 | 1.437745 | 0.719812 | 0         | 0 | 0.717933 | 0 | 0.282066 | $\delta$  | 0        |
| 1        | 0.916993 | 1/2      | 0.416993  | 0 | 1        | 0 | 1.208496 | -0.644363 | 0.416993 |
|          | 0        | 1.208496 | -0.644363 | δ |          | 0 | 1.208496 | -0.644363 | δ        |
|          | '        |          |           |   | ·        |   |          |           | (80)     |

 $\delta = 0.435866$ 

• SSP3(4,3,3)

$$\begin{array}{c|ccccccccccccccccccccccccccccccccccc$$

 $\alpha = 0.241694, \delta = 0.060424, \eta = 0.129153$ 

The first order scheme (78) corresponds to the implicit Euler method and is stiffly accurate and stability preserving. Both properties are also exhibited by the second order scheme (79), while for third order accurate IMEX RK methods we use either (80) or (81) for low or high Mach number flows, respectively. Indeed, in the stiff limit only (80) can be used in order to be consistent with the limit model at the discrete level [20, 60].

#### 3.4.2. High order in space

To achieve high order of accuracy in space centered finite difference schemes are adopted for the treatment of the implicit terms, whereas a novel CWENO reconstruction is employed for the explicit convective and viscous terms in the governing equations (1).

*Implicit terms*. The spatial discretization of the flux derivative operators (52)-(53) presented in Section 3.2 accounts for up to second order spatial accuracy. Standard finite differences are employed for higher order discretizations, hence yielding

$$\frac{\partial p}{\partial x}\Big|_{x_{i}}^{n+1} = \frac{-p_{i+2}^{n+1} + 8p_{i+1}^{n+1} - 8p_{i-1}^{n+1} + p_{i-2}^{n+1}}{12\Delta x} + O(\Delta x^{4}), \tag{82}$$

$$\frac{\partial}{\partial x}\left(h\frac{\partial p}{\partial x}\right)\Big|_{x_{i}}^{n,n+1} = \frac{1}{\Delta x^{2}}\left[\begin{array}{cccccccccccccccccccccccccccccccccccc$$

Let observe that the operator (83) is an extension of the second order operator (53), hence it is derived in the same way by using Lagrange interpolation polynomials of higher order for the derivative of the pressure. We recover the same discretization proposed in [60] and the scheme maintains the compactness of the stencil, that is now bounded in the interval [i-2, i-1, i, i+1, i+2]. The fourth order finite difference approximation (82) is applied to all first derivatives that appear in the elliptic equation on the pressure (54)-(55) as well as for the pressure flux in the momentum equation (56).

Explicit terms. High order shock-capturing finite volume methods are usually built upon a nonlinear reconstruction procedure that allows to stabilize the numerical scheme and avoid spurious oscillations in the vicinity of strong discontinuities. Here, we propose to develop a CWENO-type algorithm [45, 18] because it permits to keep a compact stencil, that is determined by the polynomial degree M of the reconstruction. The spatial discretization makes use of Cartesian control volumes, so that the entire reconstruction algorithm can be performed in a reference element with dimensional splitting, that is we first obtain a high order polynomial of degree M in x direction, then in y and finally in z direction. This results in a computationally more efficient method compared to fully multidimensional reconstruction algorithms. A similar approach has been forwarded in [47] for WENO reconstructions, requiring larger stencils and thus more memory consumption compared to the algorithm proposed in the following.

The reconstruction procedure aims at generating high order polynomials w(t, x), which are written using a *nodal* basis of polynomials of degree M defined in a reference unit interval I = [0; 1]. A cell is rescaled on the reference interval at the aid of the following change of coordinates:

$$\xi = \xi(x, i) = \frac{1}{\Delta x} (x - x_{i-1/2}), \quad \eta = \eta(y, j) = \frac{1}{\Delta y} (y - y_{i-1/2}), \quad \zeta = \zeta(z, k) = \frac{1}{\Delta z} (z - z_{i-1/2}). \tag{84}$$

In particular, since the reconstruction will then be employed for the evaluation of explicit numerical fluxes across cell boundaries, the basis consists of M+1 linearly independent Lagrange interpolating polynomials of maximum degree M, i.e.  $\{\psi_l\}_{l=1}^{M+1}$ , passing through a set of M+1 nodal points  $\{\xi_k\}_{k=1}^{M+1}$ , which are assumed to be the Gauss-Lobatto nodes. The interpolation property holds by construction, that is

$$\psi_l(\xi_k) = \delta_{lk}, \qquad l, k = 1, \dots, M + 1.$$
 (85)

In this way the reconstruction degrees of freedom automatically provide the values of the high order numerical solution for each conserved variable at the nodes. Furthermore, with our choice the degrees of freedom coincide with the Gauss-Lobatto nodes, thus very efficient quadrature-free computations can be designed for general integration over the reference interval, e.g. the numerical flux integration. The final reconstruction polynomial will then take the form

$$\mathbf{w}(t^n, \mathbf{x}) = \psi_l(\xi) \, \psi_q(\eta) \, \psi_r(\zeta) \, \hat{\mathbf{w}}_{ijk,pqr}^n, \tag{86}$$

with the unknown degrees of freedom  $\hat{w}_{ijk,pqr}^n$  that must be determined. Einstein summation convention, implying summation over indices appearing twice, is adopted. The CWENO reconstruction is carried out in a dimension-by-dimension manner for each cell  $C_{ijk}$  and the starting point is the definition of the one-dimensional reconstruction

stencils. Contrarily to WENO schemes [47], here we always consider a total number of  $N_s = 3$  reconstruction stencils, namely one central stencil (s = 0) for reconstructing a polynomial of degree M and two fully one-sided stencils, one to the left (s = 1) and the other one to the right (s = 2), for obtaining second order polynomials that are only used for nonlinear stabilization in the presence of discontinuous profiles of the numerical solution. The central stencil s = 0 is assembled for each Cartesian direction as

$$S_{ijk}^{0,x} = \bigcup_{e=i-L}^{i+L} C_{ejk}, \quad S_{ijk}^{0,y} = \bigcup_{e=j-L}^{j+L} C_{iek}, \quad S_{ijk}^{0,z} = \bigcup_{e=k-L}^{k+L} C_{ije},$$
(87)

where the spatial extension of the stencil to the left L and to the right R is given by

The low order one-sided stencil s = 1 (left-sided) and s = 2 (right-sided) are simply assembled with the element under consideration  $C_{ijk}$  and the direct neighbor to the left and to the right, that is

$$S_{ijk}^{1,x} = \bigcup_{e=i-1}^{i} C_{ejk}, \quad S_{ijk}^{2,x} = \bigcup_{e=i}^{i+1} C_{ejk},$$
(88)

the same holding for y and z direction. The reconstruction is based on integral conservation of all conserved quantities stored in the state vector  $\mathbf{Q}^n$  and is firstly performed along the x direction. Therefore, we look for a reconstruction polynomial defined on each reconstruction stencil s = 0, 1, 2 of the form

$$\mathbf{w}^{s,x}(t^n, x) = \psi_p(\xi) \,\hat{\mathbf{w}}_{ijk,p}^{n,s},\tag{89}$$

for which integral conservation holds, that is

$$\frac{1}{\Delta x} \int_{x_{-1/2}}^{x_{e+1/2}} \psi_p(\xi(x)) \, \hat{\mathbf{w}}_{ijk,p}^{n,s} = \mathbf{Q}_{ejk}^n, \qquad \forall C_{ejk} \in \mathcal{S}_{ijk}^{s,x}, \tag{90}$$

that must be prescribed for each stencil  $s \in [0, 1, 2]$ . Recall that for the one-sided stencils (s = 1, 2) the reconstruction polynomial is of degree one, therefore the nodal basis are defined accordingly by locally setting M = 1. Equations (90) lead to a linear system which might become overdetermined in the case of even order schemes. We rely on a constrained least squares (CLSQ) technique [26, 10] for determining the unknowns  $\hat{w}_{ijk,p}^{n,s}$ , where the linear constraint is given by requiring that integral conservation (90) exactly holds true for the cell  $C_{ijk}$  under consideration. In the CWENO framework the polynomial  $w^{0,x}(t^n,x)$  defined on the central stencil is often referred to as *optimal polynomial*, because among all the possible polynomials of degree M, it is the only one that shares the same cell average  $\mathbf{Q}_{ijk}^n$  in the element, while being close in the least-square sense to the other cell averages in the stencil. According to [22], the central polynomial  $\tilde{w}^{0,x}(t^n,x)$  is then obtained by difference between the polynomial  $w^{0,x}(t^n,x)$  and the linear combination of the one-sided polynomials  $w^{1,x}(t^n,x)$  and  $w^{2,x}(t^n,x)$  of lower degree [18], that is

$$\tilde{\mathbf{w}}^{0,x}(t^n, x) = \frac{1}{\lambda_0} \left( \mathbf{w}^{0,x}(t^n, x) - \sum_{s=1}^2 \lambda_s \mathbf{w}^{s,x}(t^n, x) \right), \qquad \tilde{\mathbf{w}}^{1,x}(t^n, x) = \mathbf{w}^{1,x}(t^n, x), \qquad \tilde{\mathbf{w}}^{2,x}(t^n, x) = \mathbf{w}^{2,x}(t^n, x),$$
(91)

where  $\lambda_s$  with s = 0, 1, 2 are positive coefficients such that

$$\sum_{s=0}^{2} \lambda_s = 1 := \lambda_{sum}. \tag{92}$$

Here, we do not use the pointwise WENO formulation originally introduced in [39], but the polynomial WENO schemes forwarded in [26]. As a consequence, the linear weights are a normalization which sums up to unity and

we set  $\lambda_0 = 200/\lambda_{sum}$  for  $S^{0,x}$  and  $\lambda_1 = \lambda_2 = 1/\lambda_{sum}$  for all one-sided polynomials [22]. Once the polynomials  $w^{s,x}(t^n,x)$  in (89) are available, we proceed by constructing a nonlinear data-dependent hybridization among the three polynomials obtained for each stencil, that is

$$\tilde{\mathbf{w}}^{x}(t^{n}, x) = \psi_{p}(\xi) \,\hat{\mathbf{w}}_{ijk,p}^{n}, \quad \text{with} \quad \hat{\mathbf{w}}_{ijk,p}^{n} = \sum_{s=0}^{2} \omega_{s} \tilde{\mathbf{w}}^{s,x}(t^{n}, x),$$
 (93)

where the nonlinear weights  $\omega_s$  are given by

e given by
$$\omega_s = \frac{\tilde{\omega}_s}{\sum\limits_{s=0}^{2} \tilde{\omega}_s}, \quad \text{with} \quad \tilde{\omega}_s = \frac{\lambda_s}{(\sigma_s + \epsilon)^r}. \tag{94}$$

The parameter  $\epsilon = 10^{-14}$  avoids division by zero and the exponent r = 4 is chosen according to [22]. The oscillation indicators  $\sigma_s$  are given by

$$\sigma_s = \Sigma_{lm} \hat{\mathbf{w}}_l^{n,s} \hat{\mathbf{w}}_m^{n,s}, \tag{95}$$

where the oscillation matrix  $\Sigma_{lm}$  can be computed as done in [24] once and for all on the reference interval I, hence

$$\Sigma_{lm} = \sum_{\alpha=1}^{M} \int_{0}^{1} \frac{\partial^{\alpha} \psi_{l}(\xi)}{\partial \xi^{\alpha}} \cdot \frac{\partial^{\alpha} \psi_{m}(\xi)}{\partial \xi^{\alpha}} d\xi. \tag{96}$$

Notice that the integrals appearing in (90), which then constitute the so-called *reconstruction matrix*, only depend on the geometry, i.e. on the interval over which integration is carried out. Since this corresponds to the reference element I, the reconstruction matrix can be evaluated, inverted and stored during the pre-processing stage and it remains the same throughout the entire computation. Furthermore, only one reconstruction matrix is needed because all control volumes are rescaled to the reference element and our CWENO reconstruction is carried out one by one for each spatial dimension.

The polynomials  $\mathbf{w}^x(t^n, x)$  obtained so far are high order accurate in x direction, but they still remain a cell average along the y and z direction. Therefore the CWENO reconstruction procedure illustrated above needs to be performed again along y and finally along z direction (see [47] for further details). The final element-wise reconstruction polynomials  $\mathbf{w}(t^n, \mathbf{x})$  in (86) represent entire polynomials defined by a nodal basis, which makes use of the Gauss-Lobatto interpolation points. Consequently, the degrees of freedom associated to the high order reconstruction are nothing but the high order extrapolated values of the conserved quantities at a set of quadrature nodes, thus they are ready for performing integration as a direct result of the reconstruction procedure. In other words, no further reconstruction evaluations will be needed while integrating over cell boundaries for the numerical flux computation.

Quadrature-free finite volume scheme for the explicit fluxes. The interpolation property of the CWENO reconstruction polynomials, that are expressed in terms of a nodal basis defined through a set of Gauss-Lobatto points in the reference interval I, can be fully exploited for designing a quadrature-free finite volume solver for the computation of the explicit fluxes in (49). For the sake of clarity let us consider a one-dimensional setting with the generic cell quantity  $m_i^n$ . The integral of the basis functions  $\psi_I(\xi)$  over the reference interval I is simply given by

$$\mathcal{F}_{l} := \int_{0}^{1} \psi_{l}(\xi) \, d\xi, \qquad l = 1, \dots, M + 1, \tag{97}$$

which will then be used as universal flux matrix. The high order version of the finite volume scheme (49) writes

$$m_i^* = m_i^n - \frac{\Delta t}{\Delta x} \left( \mathcal{F}_l \hat{f}_{l,i+1/2}^m - \mathcal{F}_q \hat{f}_{q,i-1/2}^m \right),$$
 (98)

where the expansion coefficients of the fluxes, i.e.  $\hat{f}_{l,i+1/2}^m$  and  $\hat{f}_{q,i-1/2}^m$ , are obtained by computing the corresponding fluxes defined in (50) at the Gauss-Lobatto nodes, that is l and q for the face i+1/2 and i-1/2, respectively. Because the reconstruction values are directly available at quadrature points, no computation is needed but it is sufficient to pick the correct degree of freedom (either l or q in (98)) out of the CWENO polynomial  $w(t^n, x)$  and calculate the Rusanov flux (50). From (98) it is evident that the high order finite volume scheme is quadrature-free, thus requiring only a matrix-vector multiplication for obtaining high order numerical fluxes for all conserved variables. The same applies for the fluxes in y and z direction.

## 3.5. Viscous fluxes of the Navier-Stokes equations

The extension of the algorithm to the Navier-Stokes model is simply performed relying on an explicit discretization of the viscous fluxes which are then added to the explicit operators  $m_i^*$ . For first order schemes the discrete velocity gradients on the control volume boundaries are computed as

$$\nabla u_{i+1/2}^n = \frac{1}{\Lambda x} \left( \nabla u_{i+1}^n - \nabla u_i^n \right), \qquad \nabla u_{i-1/2}^n = \frac{1}{\Lambda x} \left( \nabla u_i^n - \nabla u_{i-1}^n \right). \tag{99}$$

For higher order (M > 0) the CWENO reconstruction polynomials are exploited, since they automatically provide gradients of the conserved variables. Specifically, the Rusanov flux (50) is slightly modified along the lines of [32, 21] in order to include both the convective and the viscous terms, hence obtaining the following numerical flux  $f^w$  for the reconstructed conserved variables  $w(t^n, x)$  at the interface  $x_{i+1/2}$ :

$$f_{i+1/2}^{\mathbf{w}} = \frac{1}{2} \left[ f\left(\mathbf{w}_{i+1}(t^n, x_{i+1/2}), \nabla \mathbf{w}_{i+1}(t^n, x_{i+1/2})\right) + f\left(\mathbf{w}_i(t^n, x_{i+1/2}), \nabla \mathbf{w}_i(t^n, x_{i+1/2})\right) \right] - \frac{1}{2} \left( a_{i+1/2}^n + 2\eta \lambda_{\nu, i+1/2}^n \right) \cdot \left[ \mathbf{w}_{i+1}(t^n, x_{i+1/2}) - \mathbf{w}_i(t^n, x_{i+1/2}) \right],$$
 (100)

with  $\lambda_{v,i+1/2}^n$  representing the maximum eigenvalue of the viscous operator defined in (21). Here, the physical fluxes  $f(\cdot)$  contain all terms of the nonlinear flux tensor **F** (70) and the numerical viscosity is supplemented with a dissipative coefficient  $\eta$  which arises from the solution of the generalized diffusive Riemann problem [32] and is evaluated as

$$\eta = \frac{2M+1}{\Delta x \sqrt{\pi/2}}.\tag{101}$$

# 3.6. A posteriori stabilization at high Mach number

In the case of strong discontinuities in the flow, implicit high order time discretizations are not able to remove overshooting and undershooting of the numerical solution. This aspect has been investigated in [61] and numerically observed in [68] for a second order IMEX scheme applied to the Euler equations. Spurious oscillations are generated by the violation of the explicit CFL stability condition and they do not vanish but remain limited in time. In this sense high order implicit schemes are not  $L_{\infty}$  stable but only  $L_2$  stable.

To overcome this problem, we propose to use a stabilization technique that is based on a *convex combination* of high order and first order schemes, which are proven to ensure monotonicity. Differently from the *a posteriori* approach that has been developed in [31], here we employ a limiting procedure which makes use of an *a priori* strategy. The first objective is to detect troubled cells, i.e. those regions of the computational domain  $\Omega$  which are characterized by strong shocks. We rely on the flattener variable described in [5] as *shock indicator*. A shock can be identified by comparing the divergence of the velocity field  $\nabla \cdot \boldsymbol{u}^n$  with the minimum of the sound speed  $c_{\min}^n$  obtained by considering the element  $C_{ijk}$  itself as well as its Neumann neighborhood  $\mathcal{D}_{ijk}$ , i.e. all elements which share one face with  $C_{ijk}$ :

$$\mathcal{D}_{ijk} = \bigcup_{e=i-1}^{e=i+1} C_{ejk} \cup \bigcup_{e=j-1}^{e=j+1} C_{iek} \cup \bigcup_{e=k-1}^{e=k+1} C_{ije} \quad \text{with} \quad e \neq \{i, j, k\}.$$
 (102)

The divergence of the velocity field is then evaluated as follows:

$$(\nabla \cdot \boldsymbol{u}^{n})_{ijk} = \frac{1}{\Delta x} \left[ (u_{i+1jk}^{n} - u_{ijk}^{n}) - (u_{i-1jk}^{n} - u_{ijk}^{n}) \right] + \frac{1}{\Delta y} \left[ (v_{ij+1k}^{n} - v_{ijk}^{n}) - (v_{ij-1k}^{n} - v_{ijk}^{n}) \right] + \frac{1}{\Delta z} \left[ (w_{ijk+1}^{n} - w_{ijk}^{n}) - (w_{ijk-1}^{n} - w_{ijk}^{n}) \right].$$

$$(103)$$

Among all neighbors we compute the minimum sound speed  $c_{\min}^n \in \mathcal{D}_{ijk}$ , which is a function of the pressure and the density. The divergence of the velocity field (103) is estimated from the cell-averaged states  $\mathbf{Q}^n$  which are known at

the current time. The flattener variable  $\chi_{ijk}^n$  can now be computed:

$$\chi^{n} = \min\left[1, \max\left(0, -\frac{\nabla \cdot \boldsymbol{u}^{n} + k_{1}c_{\min}^{n}}{k_{1}c_{\min}^{n}}\right)\right],\tag{104}$$

with the coefficient  $k_1 = 10^{-3}$  set for all our computations. To ensure further stabilization, the flattener is extended also to those elements which are about to be crossed by a shock, but have still to enter the wave, as done in [11]. The flattener variable is interpreted as a *detector*, therefore the cell is flagged as troubled if  $\chi^n > 0$ . Let observe that in the case of rarefaction waves, where the divergence of the velocity field is positive in (104), and when shocks of modest strength occur, that is  $-k_1c_{\min}^n \leq \nabla \cdot \boldsymbol{u}^n \leq 0$ , the flattener variable is zero. Moreover, the flattener is bounded in the interval [0:1].

Once the flattener indicator has been computed for all cells, the semi-implicit scheme presented in Section 3.2 is run with high order time and space discretizations following the algorithm detailed in Section 3.4. As a result one obtains a so-called *candidate solution*  $\mathbf{Q}^{n+1,O(M+1)}$  that is of order M+1. Then, if at least one cell is marked as troubled by the flattener, a first order numerical solution is computed, i.e.  $\mathbf{Q}^{n+1,O(1)}$ . Finally, the new solution at the next time level is given by the convex combination

$$\mathbf{Q}^{n+1} = \chi^n \mathbf{Q}^{n+1,O(1)} + (1 - \chi^n) \mathbf{Q}^{n+1,O(M+1)}.$$
 (105)

If no cells are marked as troubled, then the new solution corresponds to the fully high order candidate solution. We underline that very few cells are typically flagged by the flattener and if no shocks occur, like in the low Mach regime, the flattener is never activated and the semi-implicit scheme is always run without any further stabilization.

#### 4. Numerical results

The new high order semi-implicit pressure solver (SI-P) is applied to a large set of different test cases in order to asses the accuracy and the robustness of the numerical scheme. Firstly, the accuracy of the method is validated at different Mach number regimes. Secondly, shock tube problems with ideal gas law (7) and Redlich-Kwong EOS (8)-(9) are considered, thus showing the capability of the high order semi-implicit method to deal with both linear and nonlinear equation of state. Finally, multidimensional test cases for inviscid and viscous flows involving shocks and other discontinuities are presented. All simulations are run in a fully three-dimensional setting and the time step is always computed according to a CFL-type stability condition that is only based on the maximum absolute value of the flow velocity and eventually the viscous eigenvalues, i.e.

$$\Delta t \le \text{CFL} \min_{\Omega} \left( \frac{|u|}{\Delta x} + \frac{|v|}{\Delta y} + \frac{|w|}{\Delta z} + \left( \frac{\lambda_{v}}{\Delta x^{2}} + \frac{\lambda_{v}}{\Delta y^{2}} + \frac{\lambda_{v}}{\Delta z^{2}} \right) \right)^{-1}, \tag{106}$$

which does no longer involve any dependency on the rescaled sound speed  $c/\sqrt{\varepsilon}$  compared to the time step (20) of fully explicit discretizations. The computational domain is addressed with  $\Omega$  and is discretized with a total number of  $N_x \times N_y \times N_z$  Cartesian control volumes. If not specified, the ideal gas EOS is assumed with  $\gamma = 1.4$ , the flattener variable presented in Section 3.6 is not activated and the third order version of the method in space and time is adopted. For viscous flows we assume constant viscosity, hence we set  $\beta = 1$  and s = 0 in (3). The vector of conserved variables is  $\mathbf{Q} = (\rho, \rho u, \rho v, \rho w, \rho E)$ , while the vector of primitive variables is addressed with  $\mathbf{U} = (\rho, u, v, w, p)$ .

### 4.1. Numerical convergence studies

The convergence of the novel semi-implicit pressure solver presented in this article is studied by considering a modified version of the smooth isentropic vortex [37] governed by the compressible Euler equations, thus we set  $\mu = \lambda = 0$  in the Navier-Stokes system (1). The computational domain is given by  $\Omega = [0; 10] \times [0; 10] \times [0; 1]$  with periodic boundaries. The fluid is characterized by a homogeneous background field on the top of which some perturbations are added, thus

$$\mathbf{U}(t_0, \mathbf{x}) = (1 + \delta \rho, 1, 1, 0, 1 + \delta p), \tag{107}$$

with the perturbations for temperature  $\delta T$ , density  $\delta \rho$  and pressure  $\delta p$  that read

$$\delta T = -\frac{(\gamma - 1)\epsilon^2}{8\gamma\pi^2}e^{1-r^2}, \quad \delta \rho = (1 + \delta T)^{\frac{1}{\gamma - 1}} - 1, \quad \delta p = (1 + \delta T)^{\frac{\gamma}{\gamma - 1}} - 1.$$
 (108)

The vortex maintains perfect equilibrium and the flow is stationary, thus the exact solution  $\mathbf{U}_{ex}$  is simply given by the initial condition at any time t > 0, i.e.  $\mathbf{U}_{ex} = \mathbf{U}(t_0, x)$ . The final time of the simulation is  $t_f = 1$  and the test is run on a sequence of successively refined computational meshes. The grids are refined in the x-y plane while keeping constant the number of cells  $N_z = 4$  along the z direction. The error  $L_m$  is normalized with respect to the exact solution, hence it is computed at the final time as

$$L_m(\mathbf{Q}) = \frac{\sqrt[m]{\int_{\Omega} \left\| \mathbf{w}(t_f, \mathbf{x}) - \mathbf{Q}(t_0, \mathbf{x}) \right\|^m d\mathbf{x}}}{\sqrt[m]{\int_{\Omega} \left\| \mathbf{Q}(t_0, \mathbf{x}) \right\|^m d\mathbf{x}}},$$
(109)

where the integrals are evaluated with Gaussian quadrature formulae of suitable order of accuracy (see [62]) and the exponent m determines the type of error norm that is computed. The numerical solution  $\mathbf{Q}(t_f, \mathbf{x})$  is reconstructed with the high order accurate CWENO procedure detailed in Section 3.4, that is  $\mathbf{w}(t_f, \mathbf{x})$ . The time step is computed according to (106) with CFL = 0.9. Numerical convergence studies are firstly carried out in the normal regime  $\varepsilon = 1$  for second and third order accurate semi-implicit schemes. The results are reported in Table 1 where errors are measured in  $L_1$ ,  $L_2$  and  $L_\infty$  norm for the conserved variables  $(\rho, \rho u, \rho E)$ . Both IMEX schemes (80) and (81) are proven to achieve the formal order of accuracy as well as the second order SA-SSP2 scheme (79).

Table 1: Numerical convergence results for the compressible Euler equations using second and third order SI-P schemes with  $\varepsilon=1$  and different IMEX time stepping. The errors are measured in  $L_1$  norm and refer to the variables  $\rho$  (density),  $\rho u$  (horizontal momentum) and  $\rho E$  (energy) at time t=1

SI-P *O*2

|                                     |             |           | -             |             |               |             |  |  |  |  |
|-------------------------------------|-------------|-----------|---------------|-------------|---------------|-------------|--|--|--|--|
| $N_x(N_y)$                          | $L_1(\rho)$ | $O(\rho)$ | $L_1(\rho u)$ | O(pu)       | $L_1(\rho E)$ | $O(\rho E)$ |  |  |  |  |
| 16                                  | 8.274E-04   | -         | 1.861E-02     | -           | 8.897E-04     | -           |  |  |  |  |
| 32                                  | 3.374E-04   | 1.29      | 4.307E-03     | 2.11        | 3.041E-04     | 1.55        |  |  |  |  |
| 64                                  | 8.838E-05   | 1.93      | 1.045E-03     | 2.04        | 8.468E-05     | 1.84        |  |  |  |  |
| 128                                 | 2.199E-05   | 2.01      | 2.583E-04     | 2.02        | 2.195E-05     | 1.95        |  |  |  |  |
|                                     |             |           |               |             |               |             |  |  |  |  |
| SI-P <i>O</i> 3 with SA-DIRK(3,4,3) |             |           |               |             |               |             |  |  |  |  |
| $N_x(N_y)$                          | $L_1(\rho)$ | $O(\rho)$ | $L_1(\rho u)$ | $O(\rho u)$ | $L_1(\rho E)$ | $O(\rho E)$ |  |  |  |  |
| 16                                  | 7.677E-05   | -         | 2.134E-03     | -           | 9.448E-05     | -           |  |  |  |  |
| 32                                  | 1.164E-05   | 2.72      | 2.884E-04     | 2.89        | 1.296E-05     | 2.87        |  |  |  |  |
| 64                                  | 1.508E-06   | 2.95      | 3.898E-05     | 2.89        | 1.806E-06     | 2.84        |  |  |  |  |
| 128                                 | 1.741E-07   | 3.11      | 6.320E-06     | 2.62        | 8.737E-08     | 4.37        |  |  |  |  |
|                                     |             |           |               |             |               |             |  |  |  |  |
| SI-P <i>O</i> 3 with SSP(4,3,3)     |             |           |               |             |               |             |  |  |  |  |
| $N_x(N_y)$                          | $L_1(\rho)$ | $O(\rho)$ | $L_1(\rho u)$ | $O(\rho u)$ | $L_1(\rho E)$ | $O(\rho E)$ |  |  |  |  |
| 16                                  | 7.428E-05   | -         | 2.137E-03     | -           | 9.495E-05     | -           |  |  |  |  |
| 32                                  | 1.128E-05   | 2.72      | 2.880E-04     | 2.89        | 1.292E-05     | 2.88        |  |  |  |  |
| 64                                  | 1.498E-06   | 2.91      | 3.900E-05     | 2.88        | 1.806E-06     | 2.84        |  |  |  |  |
| 128                                 | 2.346E-07   | 2.67      | 5 887E-06     | 2.73        | 2.856E-07     | 2.66        |  |  |  |  |

Secondly, the behavior of the scheme at low Mach regimes is investigated by considering different values of the stiffness parameter  $\varepsilon$ , namely we consider  $\varepsilon \in [10^{-6}; 10^{-1}]$  and the convergence rates are shown for the horizontal momentum  $\rho u$  in Table (2). Second and third order of accuracy are well preserved even in the limit  $\varepsilon = 10^{-6}$  in which pressure is almost constant and the total energy is entirely constituted by its kinetic part. Figure 1 depicts the pressure contours and the velocity stream-traces for the smooth isentropic vortex in the low Mach regime, highlighting that the numerical solution is independent of the Mach number, as expected. The stiffly accurate IMEX scheme (80) has been used at third order for retrieving the correct asymptotic behavior in the stiff limit.

![](_page_20_Figure_0.jpeg)

Figure 1: Third order pressure contours (20 levels have been used in the range bounded by the minimum and maximum value of the pressure) for the isentropic vortex test with  $N_x = N_y = 128$  at time t = 1. Stream-traces of the velocity field for  $\varepsilon = 10^0$  (left),  $\varepsilon = 10^{-3}$  (middle) and  $\varepsilon = 10^{-6}$  (right).

Table 2: Numerical convergence results for the compressible Euler equations using second and third order SI-P schemes at different low Mach regimes with stiffness parameters ranging in the interval  $[\varepsilon = 10^{-1}; \varepsilon = 10^{-6}]$ . The errors are measured in  $L_1$  norm and refer to the variables  $\rho u$  (horizontal momentum) at time t = 1.

|            | $\varepsilon = 10$ | <del>-</del> 1       | $\varepsilon = 10$ | <del>-2</del> | $\varepsilon = 10^{-3}$ |       |  |
|------------|--------------------|----------------------|--------------------|---------------|-------------------------|-------|--|
| $N_x(N_y)$ | $L_1$              | Order                | $L_1$              | Order         | $L_1$                   | Order |  |
| 16         | 1.791E-02          | -                    | 1.731E-02          | -             | 1.744E-02               | -     |  |
| 32         | 3.558E-03          | 2.33                 | 3.875E-03          | 2.16          | 3.442E-03               | 2.34  |  |
| 64         | 7.835E-04          | 2.18                 | 9.854E-04          | 1.98          | 7.485E-04               | 2.20  |  |
| 128        | 1.856E-04          | 2.08                 | 2.116E-04          | 2.22          | 1.811E-04               | 2.05  |  |
|            |                    |                      |                    |               |                         |       |  |
|            | $\varepsilon = 10$ | -4                   | $\varepsilon = 10$ | -5            | $\varepsilon = 10^{-6}$ |       |  |
| $N_x(N_y)$ | $L_1$              | Order                | $L_1$              | Order         | $L_1$                   | Order |  |
| 16         | 1.743E-02          | -                    | 1.738E-02          | -             | 1.738E-02               | -     |  |
| 32         | 3.473E-03          | 2.33                 | 3.474E-03          | 2.32          | 3.473E-03               | 2.32  |  |
| 64         | 7.303E-04          | 4 2.25 7.327H        |                    | 2.25          | 7.333E-04               | 2.24  |  |
| 128        | 1.663E-04          | 1.663E-04 2.13 1.668 |                    | 2.14          | 1.669E-04               | 2.14  |  |
|            |                    |                      |                    |               |                         |       |  |
|            |                    | SI-P <i>O</i> 3 v    | with SA-DIRE       | ζ(3,4,3)      |                         |       |  |
|            | $\varepsilon = 10$ | -1                   | $\varepsilon = 10$ | -2            | $\varepsilon = 10^{-3}$ |       |  |
| $N_x(N_y)$ | $L_1$              | Order                | $L_1$              | Order         | $L_1$                   | Order |  |
| 16         | 2.222E-03          | -                    | 2.227E-03          | -             | 2.188E-03               | -     |  |
| 32         | 3.381E-04          | 2.72                 | 3.319E-04          | 2.75          | 3.292E-04               | 2.73  |  |
| 64         | 4.608E-05          | 2.88                 | 4.440E-05          | 2.90          | 4.350E-05               | 2.92  |  |
| 128        | 6.320E-06 2.87     |                      | 5.746E-06 2.95     |               | 5.511E-06               | 2.98  |  |
|            |                    |                      |                    |               |                         |       |  |
|            | $\varepsilon = 10$ | -4                   | $\varepsilon = 10$ | -5            | $\varepsilon = 10^{-6}$ |       |  |
| $N_x(N_y)$ | $L_1$              | Order                | $L_1$              | Order         | $L_1$                   | Order |  |
| 16         | 2.188E-03          | -                    | 2.188E-03          | -             | 2.325E-03               | -     |  |
| 32         | 3.280E-04          | 2.74                 | 3.328E-04          | 2.74          | 3.279E-04               | 2.83  |  |
| 64         | 4.342E-05          | 2.92                 | 4.342E-05          | 2.92          | 4.341E-05               | 2.92  |  |
| 128        | 5.448E-06 2.99     |                      | 5.448E-06          | 2.99          | 5.448E-06               | 2.99  |  |
|            |                    |                      |                    |               |                         |       |  |

## 4.2. Shock tube problems

The novel numerical method is here validated against a set of well-known Riemann problems for the compressible Euler equations taken from [64]. The initial condition of the gas consists in a left (L) and a right (R) state that are separated by a discontinuity located at  $x = x_d$ . The computational domain is the box  $\Omega = [xL; xR] \times [0; 0.1] \times [0; 0.1]$  with Dirichlet boundary conditions imposed along the x direction and periodic boundaries set elsewhere. Table 3 summarizes the extension of the computational domain as well as the initial condition for density, horizontal velocity and pressure for all shock tube problems considered in the following. Riemann problems RK1 and RK2 are concerned with the nonlinear Redlich-Kwong EOS and the computational domain is discretized with  $N_x \times N_y \times N_z = 400 \times 4 \times 4$  control volumes, while the other tests involve an ideal gas and the computational grid is composed of  $N_x \times N_y \times N_z = 200 \times 4 \times 4$  cells. The computation stops at the final time indicated in Table 3 and we set CFL = 0.9 for the first four test cases, whereas CFL = 0.5 is adopted for the simulations involving the Redlich-Kwong EOS. The reference solution for all test problems is computed with an explicit second order MUSCL-TVD scheme run on a very fine mesh composed of 10'000 cells. The numerical solution is plot considering a 1D cut through the x direction of the computational domain with 200 equidistant sample points.

Table 3: Initialization of shock tube problems. Initial states left (L) and right (R) are reported as well as the final time of the simulation  $t_f$ , the computational domain  $[x_L; x_R]$  and the position of the initial discontinuity  $x_d$ . The equation of state (EOS) is also specified.

| Name                   | $t_f$ | $x_L$ | $x_R$ | $x_d$ | $\rho_L$ | $u_L$ | $p_L$           | $\rho_R$ | $u_R$ | $p_R$           | EOS           |
|------------------------|-------|-------|-------|-------|----------|-------|-----------------|----------|-------|-----------------|---------------|
| RP0 (Contact)          | 0.50  | 0.0   | 1.0   | 0.25  | 1000     | 0     | 10 <sup>5</sup> | 0.01     | 0.0   | 10 <sup>5</sup> | ideal gas     |
| RP1 (Lax)              | 0.14  | 0.0   | 1.0   | 0.50  | 0.445    | 1.698 | 3.528           | 0.5      | 0.0   | 0.571           | ideal gas     |
| RP2 (Two shocks)       | 0.80  | 0.0   | 1.0   | 0.50  | 1.0      | 2.0   | 0.1             | 1.0      | -2.0  | 0.1             | ideal gas     |
| RP3 (Two rarefactions) | 0.15  | 0.0   | 1.0   | 0.50  | 1.0      | -1.0  | 0.4             | 1.0      | 1.0   | 0.4             | ideal gas     |
| RK1                    | 0.10  | -0.5  | 0.5   | 0.0   | 1.0      | 1.0   | 2.0             | 1.0      | -1.0  | 1.0             | Redlich-Kwong |
| RK2                    | 0.20  | -0.5  | 0.5   | 0.0   | 1.0      | 0.0   | 1.0             | 0.125    | 0.0   | 0.1             | Redlich-Kwong |

Test RP0 provides numerical evidences about the property of the SI-P scheme of maintaining an exact preservation of constant pressure and velocity across a contact discontinuity, see Section 3.3. Indeed, a contact discontinuity involving a density step of five orders of magnitude is moving at constant velocity and pressure to the right of the domain at Mach number ranging in the interval  $[2.7 \cdot 10^{-4}; 8.5 \cdot 10^{-2}]$ . Figure 2 shows a comparison between first and third order numerical solution and it highlights that pressure and horizontal velocity are kept constant up to machine precision.

The second Riemann problem represents a benchmark test for Godunov-type finite volume methods, namely the Lax shock tube problem. The results are gathered in Figure 3 where a very good agreement with the reference solution can be appreciated. Moreover, the high order solution is much less dissipative compared to the first order results, hence enhancing the benefits of high order discretizations in terms of accuracy especially across rarefaction waves.

RP2 and RP3 are concerned with two strong colliding shocks and a symmetric double rarefaction, respectively. Figures 4 and 5 plot the numerical solution at the final time of the simulation for RP2 and RP3, respectively. For both tests some nonphysical oscillations can be noticed in the density profile which are also present in [23], while the numerical solution for velocity and pressure is overall in good agreement with the reference solution. Very small perturbations occur at the shock waves, but this is well known also for explicit Godunov-type finite volume schemes as pointed out in [64].

A nonlinear equation of state is considered in Riemann problems RK1 and RK2, which results are shown in Figures 6 and 7, respectively. In particular, RK2 is the Sod shock tube problem that has been run using the Redlich-Kwong EOS, thus obtaining very different waves in terms of profile and location compared to the same test run with classical ideal gas EOS (see [64]). The temperature distribution is also shown and the results reasonably match the reference solution. The Newton algorithm (36) for the solution of the pressure wave equation (54)-(55) has always converged to a tolerance  $\delta = 10^{-10}$  in at most four iterations.

Finally, Figure 8 show the regions of the computational domain where the flattener  $\chi^n$  has been activated and the solution has been updated with the convex combination (105). The troubled cells are indeed identified only where

shocks are located and do not involve a large number of control volumes, thus the majority of the computational cells evolve the solution using the fully high order space and time scheme.

![](_page_22_Figure_1.jpeg)

Figure 2: RP0 involving a moving contact discontinuity at final time  $t_f = 0.5$ . From top left to bottom right: 3d computational grid with density contours and comparison of density, velocity and pressure (symbols) versus the reference solution (straight line).

# 4.3. Gresho vortex

The so-called Gresho vortex problem [34] is a known stationary solution of the Euler equation that is typically used for assessing the behavior of numerical methods at different Mach number. The computational domain is defined by  $\Omega = [-0.5; 0.5]^3$  with Dirichlet boundaries, where the initial condition is imposed. This is given in polar coordinates for density, angular velocity and pressure with  $r = \sqrt{x^2 + y^2}$  denoting the generic radial position on the x - y plane

![](_page_23_Figure_0.jpeg)

Figure 3: Lax shock tube problem (RP1) at final time *<sup>t</sup><sup>f</sup>* <sup>=</sup> <sup>0</sup>.14. Comparison of density, velocity and pressure (symbols) versus the reference solution (straight line) for first and third order SI-P schemes.

![](_page_24_Figure_0.jpeg)

Figure 4: Colliding shock test (RP2) at final time *<sup>t</sup><sup>f</sup>* <sup>=</sup> <sup>0</sup>.8. Comparison of density, velocity and pressure (symbols) versus the reference solution (straight line) for third order SI-P schemes.

![](_page_24_Figure_2.jpeg)

Figure 5: Double rarefaction test (RP3) at final time *<sup>t</sup><sup>f</sup>* <sup>=</sup> <sup>0</sup>.15. Comparison of density, velocity and pressure (symbols) versus the reference solution (straight line) for third order SI-P schemes.

![](_page_25_Figure_0.jpeg)

Figure 6: Shock tube problem RK1 at final time *<sup>t</sup><sup>f</sup>* <sup>=</sup> <sup>0</sup>.1 for the Redlich-Kwong EOS. Comparison of density, velocity, pressure and temperature (symbols) versus the reference solution (straight line) for third order SI-P schemes.

![](_page_26_Figure_0.jpeg)

Figure 7: Shock tube problem RK2 at final time *<sup>t</sup><sup>f</sup>* <sup>=</sup> <sup>0</sup>.2 for the Redlich-Kwong EOS. Comparison of density, velocity, pressure and temperature (symbols) versus the reference solution (straight line) for third order SI-P schemes.

![](_page_27_Figure_0.jpeg)

Figure 8: Flattener indicator for RP1 (top left), RP2 (top right), RK1 (bottom left) and RK2 (bottom right) test at the final time of the simulation.

and  $\theta = \arctan(y/x)$  is the corresponding angle:

$$\rho(r) = 1$$

$$\nu_{\theta}(r) = \begin{cases}
5r & 0 \le r < 0.2 \\
2 - 5r & 0.2 \le r < 0.4 \\
0 & r \ge 0.4
\end{cases}$$

$$p(r) = \begin{cases}
p_0 + \frac{25}{2}r^2 & 0 \le r < 0.2 \\
p_0 + \frac{25}{2}r^2 + 4[1 - 5r - \ln(0.2) + \ln(r)] & 0.2 \le r < 0.4 \\
p_0 - 2 + 4\ln(2) & r \ge 0.4
\end{cases}$$
(110)

where the background pressure  $p_0 = \rho/(\gamma M^2)$  is expressed in terms of the Mach number. The velocity field with Cartesian components can be easily obtained from  $u_\theta$  with a rotation, that is  $(u, v) = u_\theta/r \cdot (-y, x)$ . This test is run until the final time  $t_f = 0.4 \pi$  with different magnitudes of the Mach number, namely  $M = 10^{-1}$ ,  $M = 10^{-2}$  and  $M = 10^{-3}$ . The computational mesh is composed of  $N_x \times N_y \times N_z = 80 \times 80 \times 4$  control volumes and the time step is evaluated with CFL = 0.15 according to [3]. Figure 9 depicts the velocity magnitude contours together with the stream-traces of the velocity field for each Mach number regime. The pressure profile along the x direction is also shown and compared against the exact solution. An excellent agreement can be appreciated, hence concluding that the novel semi-implicit pressure scheme preserves the stationary solution for a wide range of Mach numbers.

Figure 10 depicts the evolution of the total kinetic energy normalized with with respect to the initial kinetic energy. Two different grids are used with characteristic mesh size of  $h_1 = 1/40$  and  $h_2 = 1/80$  for different values of the Mach number. The time step here is evaluated with CFL = 0.25. We consider both second order and third order schemes, in order to give evidences of the less dissipative behavior of the higher order scheme compared to the widespread second order solvers available in the literature for low Mach flows [3, 68, 28]. The results do not depend on the stiffness regime because of the asymptotic property of the schemes, that allows all these simulations to be run with the same time step. Indeed, one can notice that the lines are almost overlapping, thus demonstrating that the low Mach regime does not affect neither the stability nor the accuracy of the numerical scheme. Compared to other results in the literature [3], the high order accuracy of the SI-P method reduces the numerical dissipation, which is particularly evident on the coarser mesh  $h_1$ . The kinetic energy dissipation measures 0.997 and 0.998 for second and third order scheme, respectively, in the case of the finest mesh. The advantage induced by a higher order discretization is more evident on the coarse mesh, where the ratio  $K/K_0$  is 0.984 for M = 2 and 0.988 for M = 3.

# 4.4. Viscous shock

Now we consider the full Navier-Stokes system (1) in the case of supersonic viscous flows. Specifically, we propose to solve the problem of an isolated viscous shock wave which is traveling into a medium at rest with a shock Mach number of  $M_s = 2$ . The analytical solution of this problem has been obtained in [6] where the compressible Navier-Stokes equations are solved for the special case of a stationary shock wave at Prandtl number Pr = 0.75 with constant viscosity. According to [6, 21], the exact solution is given in terms of dimensionless variables, namely density, pressure and velocity. The dimensionless velocity  $\bar{u} = \frac{u}{M_s c_0}$  is related to the stationary shock wave. This can then be computed as the root of the following equation:

$$\frac{|\bar{u} - 1|}{|\bar{u} - \lambda^2|^{\lambda^2}} = \left| \frac{1 - \lambda^2}{2} \right|^{(1 - \lambda^2)} \exp\left( \frac{3}{4} \operatorname{Re}_s \frac{M_s^2 - 1}{\gamma M_s^2} x \right), \tag{111}$$

with

$$\lambda^2 = \frac{1 + \frac{\gamma - 1}{2} M_s^2}{\frac{\gamma + 1}{2} M_s^2}.$$
 (112)

The solution of equation (111) permits to express the dimensionless velocity  $\bar{u}$  as a function of x. The form of the viscous profile of the dimensionless pressure  $\bar{p} = \frac{p - p_0}{\rho_0 c_0^2 M_s^2}$  is given by the relation

$$\bar{p} = 1 - \bar{u} + \frac{1}{2\gamma} \frac{\gamma + 1}{\gamma - 1} \frac{(\bar{u} - 1)}{\bar{u}} (\bar{u} - \lambda^2). \tag{113}$$

![](_page_29_Figure_0.jpeg)

Figure 9: Gresho vortex problem with third order SI-P scheme at the final time  $t_f = 0.4 \,\pi$  with Mach number  $M = 10^{-1}$  (top),  $M = 10^{-2}$  (middle),  $M = 10^{-3}$  (bottom). Left: stream-traces of the velocity field with velocity magnitude contours (30 levels have been used in the range  $[3 \cdot 10^{-6}; 1]$  for all Mach numbers). Right: pressure distribution versus reference solution along a 1D cut in x-direction (y = z = 0) with 80 interpolation points.

![](_page_30_Figure_0.jpeg)

Figure 10: Evolution of the total kinetic energy K normalized with respect to the initial kinetic energy  $K_0$  of the Gresho vortex problem computed with second order (left) and third order (right) semi-implicit scheme. Dashed lines refer to the mesh size  $N_x = N_y = 40$ , while dash-dot lines refer to the mesh size  $N_x = N_y = 80$ . Mach number 0.1 (red), 0.01 (blue), 0.001 (black) are considered.

Finally, the profile of the dimensionless density  $\bar{\rho} = \frac{\rho}{\rho_0}$  is derived from the integrated continuity equation:  $\bar{\rho}\bar{u} = \frac{\rho}{\rho_0}$ 1. In order to simulate an unsteady shock wave traveling into a medium at rest, one can simply superimpose a constant velocity field  $u = M_s c_0$  to the solution of the stationary shock wave found in the previous steps. The computational domain is the rectangular box  $\Omega = [0; 1] \times [0; 0.2] \times [0; 0.2]$  which is discretized with a total number of cells  $N_x \times N_y \times N_z = 200 \times 4 \times 4$ . Periodic boundaries are imposed in y and z direction, while the constant inflow velocity is prescribed for x = 0 and outflow boundary condition is set at x = 1. The time step is evaluated with CFL = 0.5 and the final time of the simulation is  $t_f = 0.2$ . The initial condition is given by a shock wave centered at x = 0.25which is propagating at Mach  $M_s = 2$  from left to right with a Reynolds number of Re = 100. The upstream shock state is defined by  $\rho_0 = 1$ ,  $u_0 = v_0 = 0$ ,  $p_0 = 1/\gamma$  and  $c_0 = 1$ , while the fluid viscosity is  $\mu = 2 \times 10^{-2}$  and the ideal gas law is adopted, thus allowing the heat flux in (1) to be treated implicitly in the pressure wave equation (54). The third order SI-P schemes is used to run the simulation and the results are depicted in Figure 11, which match very closely the analytical solution for density, horizontal velocity, pressure and heat flux in x direction computed as  $q_x = \lambda \frac{\partial T}{\partial x}$ Though being a one-dimensional problem, let observe that this test case involves all terms contained in the governing equations, hence including convective and viscous fluxes, pressure gradients as well as temperature gradients and heat fluxes. Furthermore, an analytical solution does exist which permits to compare the numerical results. Looking at the excellent matching between numerical and exact solution we can conclude that the Navier-Stokes system is properly discretized by the novel semi-implicit pressure solver proposed in this article.

#### 4.5. 3D Taylor-Green vortex

As last test case we solve the well-known 3D Taylor-Green vortex, that is a widespread test problem used in the context of incompressible flows. The initial condition of the fluid according to [50] writes

$$\rho(\mathbf{x}, 0) = \rho_0,$$

$$\mathbf{u}(\mathbf{x}, 0) = (\sin(x)\cos(y)\cos(z), -\cos(x)\sin(y)\cos(z), 0),$$

$$p(\mathbf{x}, 0) = p_0 + \frac{\rho_0}{16}(\cos(2x) + \cos(2y))(\cos(2z) + 2)),$$
(114)

with  $\rho_0 = 1$ . The computational domain is the cube  $\Omega = [-\pi, \pi]^3$  and periodic boundary conditions are imposed everywhere. Starting from this smooth initial condition, the flow quickly degenerates into very complex small scale

![](_page_31_Figure_0.jpeg)

Figure 11: Viscous shock problem with shock Mach number  $M_s = 2$  and Prandtl number Pr = 0.75. Third order SI-P solution compared against analytical solution for density (top left), horizontal velocity (top right), pressure (bottom left) and heat flux in x-direction (bottom right). One-dimensional cut of 200 equidistant points along the x-direction at y = z = 0.

structures, depending on the Reynolds number. Consequently, no analytical solution is available for this highly unsteady flow. Nevertheless, in [\[48\]](#page-35-1) well-resolved DNS studies for an incompressible fluid are available, hence we consider those results as a very accurate reference solution. In order to mimic the incompressible property of the fluid we set *p*<sup>0</sup> = 10<sup>3</sup> in our semi-implicit compressible solver. The final time of the simulation is set to *t<sup>f</sup>* = 10 and the time step is evaluated with CFL <sup>=</sup> <sup>0</sup>.5. Two different values of Reynolds number are taken into account, namely *Re* = 100 and *Re* = 200. The computational domain is discretized with a total number of 107280000 control volumes, obtained by setting *N<sup>x</sup>* × *N<sup>y</sup>* × *N<sup>z</sup>* = 120 × 120 × 120. Figure [12](#page-32-0) depicts the vorticity isosurfaces together with the velocity magnitude for *Re* = 200, clearly showing the development of the small-scale structures that arise from the fluid flow.

![](_page_32_Picture_1.jpeg)

Figure 12: 3D Taylor-Green vortex at *Re* <sup>=</sup> 200. Velocity magnitude (left) and vorticity iso-surfaces at levels {2, <sup>3</sup>, <sup>5</sup>} (right) at time *<sup>t</sup>* <sup>=</sup> 4 (top) and *t* = 8 (bottom).

Finally, Figure [13](#page-33-0) plots the time series of the calculated total kinetic energy dissipation rates <sup>−</sup>*dK*/*dt* compared against the DNS data [\[48\]](#page-35-1). Also for this rather complex test case, the numerical results obtained with the third order SI-P schemes fit well with the reference solution for all the considered Reynolds numbers. Notice that the spatial resolution used for running this test is relatively coarse compared to existing second order solvers. However, the results are of good quality because of the high order discretization achieved by the novel semi-implicit schemes. This has also been observed in the case of higher order DG schemes [\[63,](#page-35-20) [4\]](#page-34-10) where the spatial resolution could be reduced even further. On the other hand, second order schemes as the method presented in [\[68\]](#page-36-3) require much more computational cells to carry out the same simulations of the 3D Taylor-Green vortex test case.

![](_page_33_Figure_1.jpeg)

Figure 13: Time evolution of the kinetic energy dissipation rate <sup>−</sup>*dK*/*dt* for the 3D Taylor-Green vortex compared with available DNS data of brachet et al. [\[48\]](#page-35-1).

# 5. Conclusions

A novel high-order semi-implicit numerical method for the solution of the compressible Navier-Stokes equations at all Mach numbers has been derived and discussed. A high order cell-centered quadrature-free finite volume scheme is used for the approximation of the explicit fluxes, whereas finite differences are employed for the discretization of the implicit terms. Collocated Cartesian grids are used in a fully three-dimensional setting and the fluid can be modeled with ideal gas as well as with general equations of state that might lead to a nonlinear relation between internal energy and pressure. The new semi-implicit method splits the kinetic energy and the enthalpy fluxes into an explicit and an implicit part, making the usage of an iterative solver for the pressure unnecessary. Formal analysis of the scheme at the discrete level reveals the asymptotic property of the algorithm, which is capable of retrieving at the discrete level a consistent discretization of the limit model in the zero Mach number regime. High order time stepping is performed relying on IMEX schemes and the resulting stability condition requires a time step limitation based only on the fluid velocity (and eventually the viscous eigenvalues) and not on the acoustic speed, thus making the novel numerical method very efficient and suitable for the simulation of low Mach flows. The implicit sub-system requires the solution of an elliptic equation for the pressure, which allows to easily include nonlinear EOS by adopting a nested Newton solver for the resulting mildly nonlinear system. This would not hold true in the case of the derivation of an implicit equation for the total energy, where the system would become fully nonlinear. A wide set of benchmark problems is proposed to test the accuracy and the robustness of the new algorithm, involving low and high Mach number flows as well as viscous and inviscid fluid simulations.

Future research will concern the solution of more complex systems of hyperbolic equations with nonconservative products and stiff source terms like the GPR model [\[57\]](#page-35-39), which would simultaneously allow a unified formulation for continuum mechanics, including fluids and solids. Another potential future topic of research may be the application of the new semi-implicit pressure-based IMEX solver to the equations of magnetohydrodynamics with involution constraints, which require the additional property of keeping a zero divergence of the magnetic field at the discrete level.

# Acknowledgments

The Authors would like to thank the Italian Ministry of Instruction, University and Research (MIUR) to support this research with funds coming from PRIN Project 2017 (No. 2017KKJP4X entitled Innovative numerical methods for evolutionary partial differential equations and applications). The Authors also acknowledge the CINECA supercomputing center in Bologna (Italy) for awarding access to the MARCONI100 cluster under the project *IscrC HiPPUM 0*.

# References

- [1] P. Lax A. Harten and B. van Leer. On upstream differencing and godunov-type schemes for hyperbolic conservation laws. SIAM Rev., 25:35–61, 1983.
- [2] T. Alazard. Low mach number limit of the full navier-stokes equations. Arch. Ration. Mech. Anal., 180:1–73, 2006.
- [3] S. Avgerinos, F. Bernard, A. Iollo, and G. Russo. Linearly implicit all mach number shock capturing schemes for the euler equations. J. Comput. Phys., 393:278–312, 2019.
- [4] P. Roe B. Einfeldt, C. Munz and B. Sjogreen. On godunov-type methods near low densities. ¨ J. Comp. Phys., 92:273–295, 1991.
- [5] D.S. Balsara. Self-adjusting, positivity preserving high order schemes for hydrodynamics and magnetohydrodynamics. J. Comp. Phys., 231:7504 – 7517, 2012.
- [6] R. Becker. Stosswelle und detonation. Physik, 8:321–1923, 1923.
- [7] G. Billet and R. Abgrall. An adaptive shock-capturing algorithm for solving unsteady reactive flows. Computers & Fluids, 32:1473–1495, 2003.
- [8] S. Boscarino and G. Russo. On a class of uniformly accurate IMEX Runge-Kutta schemes and applications to hyperbolic systems with relaxation. SIAM J. Sci. Comput., 31:1926–1945, 2009.
- [9] Sebastiano Boscarino, Lorenzo Pareschi, and Giovanni Russo. A unified IMEX Runge-Kutta approach for hyperbolic systems with multiscale relaxation. SIAM J. Numer. Anal., 55(4):2085–2109, 2017.
- [10] W. Boscheri. A space-time semi-lagrangian advection scheme on staggered voronoi meshes applied to free surface flows. Computers & Fluids, 202:104503, 2020.
- [11] W. Boscheri and D.S. Balsara. High order direct Arbitrary-Lagrangian-Eulerian (ALE) *P<sup>N</sup> P<sup>M</sup>* schemes with WENO Adaptive-Order reconstruction on unstructured meshes. J. Comp. Phys., 398:108899, 2019.
- [12] W. Boscheri, G. Dimarco, and M. Tavelli. An efficient all mach second order finite volume solver for compressible navier-stokes equations. Computer Methods in Applied Mechanics and Engineering. submitted.
- [13] L. Brugnano and V. Casulli. Iterative solution of piecewise linear systems. SIAM J. Sci. Comput., 30:463472, 2007.
- [14] M. Girardin C. Chalons and S. Kokh. Large time step and asymptotic preserving numerical schemes for the gas dynamics equations with source terms. SIAM J. Sci. Comput., 35:2874–2902, 2013.
- [15] S. Jiang C. Dou and Y. Ou. Low mach number limit of full navier-stokes equations in a 3d bounded domain. J. Diff. Equat., 258:379–398, 2015.
- [16] V. Casulli. Semi-implicit finite difference methods for the two–dimensional shallow water equations. J. Comp. Phys., 86:56–74, 1990.
- [17] V. Casulli. A semi-implicit finite difference method for non-hydrostatic free-surface flows. Int. J. Num. Meth. in Fluids, 30:425–440, 1999.
- [18] I. Cravero, G. Puppo, M. Semplice, and G. Visconti. CWENO: uniformly accurate reconstructions for balance laws. Math. Comp., 87:1689– 1719, 2018.
- [19] P. Degond and M. Tang. All speed scheme for the low Mach number limit of the isentropic Euler equations. Commun. Comput. Phys., 10:1–31, 2011.
- [20] S. Dellacherie. Analysis of godunov type schemes applied to the compressible euler system at low mach number. J. Comp. Phys., 229:978– 1016, 2010.
- [21] M. Dumbser. Arbitrary high order PNPM schemes on unstructured meshes for the compressible Navier–Stokes equations. Computers & Fluids, 39:60–76, 2010.
- [22] M. Dumbser, W. Boscheri, M. Semplice, and G. Russo. Central weighted ENO schemes for hyperbolic conservation laws on fixed and moving unstructured meshes. SIAM J. Sci. Comput., 39(6):A2564–A2591, 2017.
- [23] M. Dumbser and V. Casulli. A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible navier-stokes equations with general equation of state. Applied Mathematics and Computation, 272:479–497, 2016.

- [24] M. Dumbser, C. Enaux, and E.F. Toro. Finite volume schemes of very high order of accuracy for stiff hyperbolic balance laws. J. Comp. Phys., 227:3971–4001, 2008.
- [25] M. Dumbser, U. Iben, and M. Ioriatti. An efficient semi-implicit finite volume method for axially symmetric compressible flows in compliant tubes. Applied Numerical Mathematics, 89:24–44, 2015.
- [26] M. Dumbser and M. Kaser. Arbitrary high order non-oscillatory finite volume schemes on unstructured meshes for linear hyperbolic systems. ¨ J. Comp. Phys., 221:693–723, 2007.
- [27] M.E. Vazquez-Cend ´ on E.F. Toro. Flux splitting schemes for the euler equations. ´ Computers & Fluids, 70:1–12, 2012.
- [28] P. Degond F. Cordier and A. Kumbaro. An asymptotic-preserving all-speed scheme for the euler and navier stokes equations. J. Comp. Phys., 231:5685–5704, 2012.
- [29] F. Fambri and M. Dumbser. Semi-implicit discontinuous galerkin methods for the incompressible navier–stokes equations on adaptive staggered cartesian grids. Computer Methods in Applied Mechanics and Engineering, 324:170–203, 2017.
- [30] F. Fambri, M. Dumbser, and O. Zanotti. Space–time adaptive ader-dg schemes for dissipative flows: Compressible navier–stokes and resistive mhd equations. Computer Physics Communications, 220:297–318, 2017.
- [31] V.M. Dansac G. Dimarco, R. Loubere and M.H. Vignal. Second-order implicit-explicit total variation diminishing schemes for the euler ` system in the low mach regime. J. Comp. Phys., 372:178–201, 2018.
- [32] F. Lorcher G. Gassner and C.D. Munz. A contribution to the construction of di ¨ ffusion fluxes for finite volume and discontinuous galerkin schemes. J. Comp. Phys., 224:1049–1063, 2007.
- [33] S. Godunov. Finite difference methods for the computation of discontinuous solutions of the equations of fluid dynamics. Mat. Sb., 47:271– 306, 1959.
- [34] P. M. Gresho and S.T. Chan. On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. part 2: Implementation.
- [35] H. Guillard and A. Murrone. On the behavior of upwind schemes in the low mach number limit : Ii. godunov type schemes. Computers & Fluids, 33:655–675, 2004.
- [36] H. Guillard and C. Viozat. On the behavior of upwind schemes in the low mach limit. Computers & Fluids, 28:63–86, 1999.
- [37] C. Hu and C.W. Shu. Weighted essentially non-oscillatory schemes on triangular meshes. J. Comp. Phys., 150:97–127, 1999.
- [38] M. Ioriatti and M. Dumbser. Semi-implicit staggered discontinuous galerkin schemes for axially symmetric viscous compressible flows in elastic tubes. Computers and Fluids, 167:166–179, 2018.
- [39] G.-S. Jiang and C.W. Shu. Efficient implementation of weighted ENO schemes. J. Comp. Phys., 126:202–228, 1996.
- [40] Shi Jin. Efficient asymptotic-preserving (AP) schemes for some multiscale kinetic equations. SIAM J. Sci. Comput., 21(2):441–454, 1999.
- [41] Axel Klar. An asymptotic preserving numerical scheme for kinetic equations in the low Mach number limit. SIAM J. Numer. Anal., 36(5):1507–1527, 1999.
- [42] R. Klein. Semi-implicit extension of a godunov-type scheme based on low mach number asymptotics i: One-dimensional flow. J. Comp. Phys., 121:213–237, 1995.
- [43] P. Lax and B. Wendroff. Systems of conservation laws. J. Comp. Phys., 13:217–237, 1960.
- [44] R.J. LeVeque. Finite Volume Methods for Hyperbolic Problems. Cambridge University Press, 2002.
- [45] D. Levy, G. Puppo, and G. Russo. Central WENO schemes for hyperbolic systems of conservation laws. M2AN Math. Model. Numer. Anal., 33(3):547–571, 1999.
- [46] R. Klein M. Boger, F. Jaegle and C.-D. Munz. Coupling of compressible and incompressible flow regions using the multiple pressure variables approach. Math. Methods Appl. Sci., 38:458–477, 2015.
- [47] A. Hidalgo M. Dumbser, O. Zanotti. Ader-weno finite volume schemes with space-time adaptive mesh refinement. J. Comp. Phys., 248:257– 286, 2013.
- [48] D.I. Meiron M.E. Brachet and S.A. Orszag. Small-scale structure of the taylor-green vortex. Journal of Fluid Mechanics, 130:411–452, 1983.
- [49] C.D. Munz. On godunov-type schemes for lagrangian gas dynamics. SIAM J. Numer. Anal., 31:17–42, 1994.
- [50] R. Codina O. Colomes, S. Badia and J. Principe. Assessment of variational multiscale models for the large eddy simulation of turbulent incompressible flows. Comp. Methods in App. Mech. and Eng., 285:32–63, 2015.
- [51] S. Osher and F. Solomon. A partially implicit method for large stiff systems of ode's with only few equations introducing small time-constants. SIAM J. Numer. Anal., 13:645–663, 1976.
- [52] S. Osher and F. Solomon. Upwind difference schemes for hyperbolic conservation laws. Math. Comput., 38:339–374, 1997.
- [53] A. Sangam P. Degond, F. Deluzet and M.-H. Vignal. An asymptotic preserving scheme for the euler equations in a strong magnetic field. J. Comp. Phys., 228:3540–3558, 2009.
- [54] L. Pareschi and G. Russo. High order asymptotically strong-stability-preserving methods for hyperbolic systems with stiff relaxation. in: Hou t.y., tadmor e. (eds) hyperbolic problems: Theory, numerics, applications. 2003.
- [55] L. Pareschi and G. Russo. Implicit-explicit runge-kutta schemes and applications to hyperbolic systems with relaxation. J. Sci. Comput., 25:129–155, 2005.
- [56] S.V. Patankar. Numerical heat transfer and fluid flow. New York: McGraw-Hill, 1980.
- [57] I. Peshkov and E. Romenski. A hyperbolic model for viscous newtonian flows. Continuum Mech Thermodyn, 28:85104, 2016.
- [58] F. Filbet S. Boscarino and G. Russo. High order semi-implicit schemes for time dependent partial differential equations. J. Sci. Comput., 68:975–1001, 2016.
- [59] G. Russo S. Boscarino, J.-M. Qiu and T. Xiong. A high order semi-implicit imex weno scheme for the all-mach isentropic euler system. J. Comp. Phys., 392:594–618, 2019.
- [60] G. Russo S. Boscarino, J.-M. Qiu and T. Xiong. A high order semi-implicit imex weno scheme for the all-mach isentropic euler system. J. Comp. Phys., 392:594–618, 2019.
- [61] C.-W. Shu S. Gottlieb and E. Tadmor. Strong stability-preserving high-order time discretization methods. SIAM Rev., 43:89–112, 2001.
- [62] A.H. Stroud. Approximate Calculation of Multiple Integrals. Prentice-Hall Inc., Englewood Cliffs, New Jersey, 1971.
- [63] M. Tavelli and M. Dumbser. A staggered space-time discontinuous galerkin method for the three-dimensional incompressible navier-stokes

- equations on unstructured tetrahedral meshes. J. Comp. Phys., 319:294–323, 2016.
- [64] E.F. Toro. Riemann Solvers and Numerical Methods for Fluid Dynamics: a Practical Introduction. Springer, 2009.
- [65] E. Turkel. Preconditioned methods for solving the incompressible and low speed compressible equations. J. Comp. Phys., 72:277–298, 1987.
- [66] S. J. Ruuth U. M. Ascher and R. J. Spiteri. Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations. Appl. Numer. Math., 25:151–167, 1982.
- [67] J. Vidal. Thermodynamics: Applications in Chemical Engineering and the Petroleum Industry. Editions Technip, 2003.
- [68] R. Loubere M. Tavelli W. Boscheri, G. Dimarco and M.H. Vignal. A second order all mach number imex finite volume solver for the three ` dimensional euler equations. J. Comp. Phys., 415:109486, 2020.