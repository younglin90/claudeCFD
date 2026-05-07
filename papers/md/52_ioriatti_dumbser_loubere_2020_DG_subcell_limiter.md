![](_page_0_Picture_0.jpeg)

# **A Staggered Semi-implicit Discontinuous Galerkin Scheme with a Posteriori Subcell Finite Volume Limiter for the Euler Equations of Gasdynamics**

Matteo Ioriatti, Michael Dumbser, Raphaël Loubère

## **To cite this version:**

Matteo Ioriatti, Michael Dumbser, Raphaël Loubère. A Staggered Semi-implicit Discontinuous Galerkin Scheme with a Posteriori Subcell Finite Volume Limiter for the Euler Equations of Gasdynamics. Journal of Scientific Computing, 2020, 83 (2), ⟨10.1007/s10915-020-01209-w⟩. ⟨hal-02618986⟩

# **HAL Id: hal-02618986 <https://hal.science/hal-02618986v1>**

Submitted on 22 Dec 2020

**HAL** is a multi-disciplinary open access archive for the deposit and dissemination of scientific research documents, whether they are published or not. The documents may come from teaching and research institutions in France or abroad, or from public or private research centers.

L'archive ouverte pluridisciplinaire **HAL**, est destinée au dépôt et à la diffusion de documents scientifiques de niveau recherche, publiés ou non, émanant des établissements d'enseignement et de recherche français ou étrangers, des laboratoires publics ou privés.

![](_page_0_Picture_9.jpeg)

[HAL Authorization](https://about.hal.science/hal-authorisation-v1/)

## A staggered semi-implicit discontinuous Galerkin scheme with a posteriori subcell finite volume limiter for the Euler equations of gasdynamics

Matteo Ioriatti<sup>1</sup> , Michael Dumbser<sup>1</sup> , Raphael Loub ¨ ere ` b

*<sup>a</sup>Department of Civil, Environmental and Mechanical Engineering University of Trento, via Mesiano 77, I-38123 Trento, Italy b Institut de Mathematiques de Bordeaux (IMB), 33405 Talence, France `*

#### Abstract

In this paper we propose a novel semi-implicit Discontinuous Galerkin (DG) finite element scheme on staggered meshes with *a posteriori* subcell finite volume limiting for the one and two dimensional Euler equations of compressible gasdynamics. We therefore extend the strategy adopted by Dumbser and Casulli (2016), where the Euler equations have been solved solved using a semi-implicit finite volume scheme based on the flux-vector splitting method recently proposed by Toro and Vazquez-Cend ´ on (2012). In our scheme, the nonlinear convective terms are discretized explic- ´ itly, while the pressure terms are discretized implicitly. As a consequence, the time step is restricted only by a mild CFL condition based on the fluid velocity, which makes this method particularly suitable for simulations in the low Mach number regime. However, the conservative formulation of the scheme, together with the novel subcell finite volume limiter allows also the numerical simulation of high Mach number flows with shock waves. Inserting the discrete momentum equation into the discrete total energy conservation law yields a mildly nonlinear system with the scalar pressure as the only unknown. The resulting pressure system can be efficiently solved with modern iterative methods.

In order to deal with shock waves or steep gradients, the new semi-implicit DG scheme proposed in this paper includes an *a posteriori* subcell finite volume limiting technique. This strategy was first proposed by Dumbser et al. in 2014 for explicit DG schemes on collocated grids and is based on the *a posteriori* MOOD approach of Clain, Loubere and ` Diot. Recently, this approach was also extended to semi-implicit DG schemes on staggered meshes for the shallow water equations in [44]. Within the MOOD approach, an *unlimited* DG scheme first produces a so-called *candidate solution* for the next time level t <sup>n</sup>+1. Later on, the control volumes with a non-admissible candidate solution are identified by using physical and numerical detection criteria, such as the positivity of the solution, the absence of floating point errors and the satisfaction of a relaxed discrete maximum principle (DMP). Then, in the detected troubled cells a more robust first order semi-implicit finite volume (FV) method is applied on a sub-grid composed of 2P + 1 subcells, where P denotes the polynomial degree used in the DG scheme. For that purpose, the nonlinear convective terms are recomputed in the troubled cells using an explicit finite volume scheme on the subcell level. Also the linear system for the pressure needs to be assembled and solved again, but where now a low order semi-implicit finite volume scheme is used on the sub-cell level in all troubled DG cells, instead of the original high order DG method. Finally, the higher order DG polynomials are reconstructed from the piecewise constant subcell finite volume averages and the scheme proceeds to the next time step.

In this paper we present, discuss and test this novel family of methods and simulate a set of classical numerical benchmark problems of compressible gasdynamics. Great attention is dedicated to 1D and 2D Riemann problems and we also show that for these test cases the scheme responds appropriately in the presence of shock waves and does not produce any non-physical spurious numerical oscillations.

*Keywords:* high order semi-implicit discontinuous Galerkin schemes, a posteriori sub-cell finite volume limiter, staggered grids, MOOD paradigm, sub-cell finite volume limiting, Euler equations of compressible gasdynamics

*Email addresses:* matteo.ioriatti@unitn.it (Matteo Ioriatti), michael.dumbser@unitn.it (Michael Dumbser), raphael.loubere@math.u-bordeaux.fr (Raphael Loub ¨ ere) `

#### 1. Introduction

Computational fluid dynamics is a branch of applied mathematics that demands the development of advanced accurate and efficient algorithms for the numerical solution of partial differential equations (PDEs) in multiple space dimensions. These mathematical tools have become widespread in many scientific applications in the contexts of <sup>5</sup> computational physics, aerospace engineering and bio-medical sciences. Nowadays, there exist several dedicated solvers capable of simulating a large range of physical flow configurations. However, an ultimate numerical method for all the possible flow conditions still does not exist. In this paper we develop a novel algorithm that is at the same time suitable for low Mach number (or nearly incompressible) flows, where classical explicit methods are typically not efficient due to the CFL timestep restriction based on the speed of sound, as well as for high Mach number flows <sup>10</sup> with shock waves and contact discontinuities. It is therefore our declared goal to develop a new all Mach number flow solver that is able to cope with a large range of different possible flow regimes of the compressible Euler equations. The Discontinuous Galerkin (DG) method is a finite element method where the numerical solution is approximated by high order piecewise polynomials. It was first developed by Reed and Hill for the discretization of the governing equations of neutron transport phenomena [57]. Later on, this method rapidly gained popularity after the works of <sup>15</sup> Cockburn and Shu. In particular, in [19, 18, 17, 20, 20] the DG method was extended to general systems of nonlinear hyperbolic PDE. While the aforementioned DG schemes were based on TVD Runge-Kutta time integrators, spacetime DG methods were first developed by Van der Vegt in [73, 74]. DG schemes were applied for the first time to the Navier-Stokes equations by Bassi and Rebay [3], Baumann and Oden [4, 5], Hartmann and Houston [39, 40], Gassner et al. [36] and Klaij et al. [47]. Moreover, a new family of fully-discrete explicit one-step DG schemes was proposed <sup>20</sup> in [30] and [56]. A central DG method that employs overlapping meshes was introduced by Liu et al. in [51, 52]. Most of the DG methods mentioned so far, apart from the space-time DG approach, which is by construction fully implicit, are characterized by an explicit time integration; this feature makes them inefficient for situations characterized by very low Mach numbers. An implicit discretization allows to avoid this problem. In this framework, we mention the implicit DG schemes on *collocated grids* developed by Bassi et al. [1, 2] and the semi-implicit DG scheme of Dolejsi <sup>25</sup> and Feistauer [23]. Other semi-implicit DG methods on collocated grids have been presented for the shallow water equations in [72, 71].

A particularly efficient family of semi-implicit finite volume and finite difference algorithms on staggered meshes for computational fluid dynamics has been developed in a series of papers by Casulli et al. in [13, 7, 11, 15, 8, 14, 9]. Later, this family of algorithms was extended to incompressible and compressible fluid flow in elastic pipes, see e.g. <sup>30</sup> [12, 68, 35, 27, 45], and in [26, 24] these schemes were extended to the compressible Navier-Stokes equations and to viscous and resistive magnetohydrodynamics (MHD), respectively. The first high-order semi-implicit DG scheme on staggered meshes for the shallow water equations has been proposed in [25] and was later successfully extended to the compressible and incompressible Navier-Stokes equations on unstructured meshes [63, 62, 64, 65, 66], to adaptive Cartesian grids (AMR) [33, 34], as well as to compressible fluid flow in elastic pipes [43] and linear elasticity [67]. A <sup>35</sup> drawback of high order DG schemes is the generation of spurious oscillations near discontinuities or strong gradients due to Gibbs' phenomenon. This fact is justified by Godunov's theorem [37], according to which *linear* monotone schemes can be at most first order accurate. In order to overcome this problem, a simple strategy consists in adding some artificial viscosity to the scheme [54]. However, determining the precise location and amount of the artificial viscosity to be added is not a trivial task, and is the subject of many publications in the CFD literature. Recently, a <sup>40</sup> more sophisticated technique, called *a posteriori subcell finite volume limiting*, was presented in [32, 78, 29, 44]. This new limiter is based on the ideas of the *a posteriori* MOOD approach that was introduced in [16, 21, 22, 53] for finite volume schemes.

In the present work we propose a novel high order *semi-implicit* DG finite element method on staggered meshes with *a posteriori* subcell finite volume limiting for the Euler equations of compressible gasdynamics. The method represents <sup>45</sup> the natural high order extension to staggered Cartesian grids of the schemes proposed in [26, 66]. In addition, the limiting approach is carried out following the ideas introduced in [44] for the shallow water equations.

The rest of paper is structured as follows. In Sections 2 and 3 we derive the new *a posteriori* subcell finite volume limiter for semi-implicit staggered DG methods for the Euler equations of gasdynamics in one and two space dimensions. In Sections 4 and 5 the numerical method is validated for several classical benchmarks that involve <sup>50</sup> smooth and discontinuous solutions. Finally, in Section 6 we give some conclusions and provide an outlook to future research.

# 2. Staggered semi-implicit DG schemes with a posteriori sub-cell finite volume limiting for the Euler equations in 1D

In this section we introduce our staggered semi-implicit DG schemes with *a posteriori* subcell finite volume limiting for the 1D Euler equations. For the sake of a comprehensive presentation, we first derive the unlimited semi-implicit 1D DG scheme in section 2.2; then in section 2.3 we introduce a semi-implicit FV method within a proper subcell formulation. Later on, we recall the MOOD algorithm and we conclude in section 2.5 with the general scheme that combines these two methods. In the following, several universal tensors will be used and we report them in the Appendix A.

#### 2.1. Governing equations of the 1D model

The Euler equations of compressible gasdynamics in one space dimension are a well-known system of hyperbolic PDEs and read as follows

$$\frac{\partial \rho}{\partial t} + \frac{\partial (\rho u)}{\partial x} = 0, 
\frac{\partial (\rho u)}{\partial t} + \frac{\partial}{\partial x} (\rho u^2 + p) = 0, 
\frac{\partial (\rho E)}{\partial t} + \frac{\partial}{\partial x} (u(\rho E + p)) = 0.$$
(1)

The equations (1) state the principles of mass, momentum and total energy conservation. The spatial and the temporal coordinates are x and t and the computational domain is denoted by  $\Omega_x = [x_L, x_R]$  in the following. Furthermore,  $\rho(x,t)$  is the fluid density, u(x,t) is the velocity, p(x,t) is the pressure and E(x,t) is the specific total energy. Additionally, we have E=e+k, where e is the specific internal energy and  $k=\frac{1}{2}u^2$  is the specific kinetic energy. We introduce another quantity called specific enthalpy  $h=e+p/\rho$  and consequently we rewrite the flux of the energy equation as the sum of two different contributions  $u(\rho E+p)=u\rho k+h\rho u$ , see [55, 70, 26]. In order to close the system we use the equation of state (EOS) for an ideal gas, which reads

$$e = e(p, \rho) = \frac{p}{(\gamma - 1)\rho}.$$
 (2)

Here,  $\gamma = c_p/c_v$  is the ratio of specific heats, which typically lies in the range  $1 < \gamma < 3$ . In this paper we will consider this quantity as a constant and equal to  $\gamma = 1.4$ , which is the adiabatic index of a diatomic gas and thus a reasonable value for air at moderate pressures and temperatures.

The system in eq. (1) can be written in more compact matrix-vector notation as

$$\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{Q})}{\partial x} = 0,\tag{3}$$

where  $\mathbf{Q} = [\rho, \rho u, \rho E]^T$  is the vector of conserved variables and  $\mathbf{F} = [\rho u, \rho u^2 + p, u(\rho E + p)]^T$  is the physical flux vector. In quasi-linear form the above system reads

$$\frac{\partial \mathbf{Q}}{\partial t} + \mathbf{A}(\mathbf{Q}) \frac{\partial \mathbf{Q}}{\partial x} = 0, \tag{4}$$

where  $\mathbf{A}(\mathbf{Q}) = \partial \mathbf{F}(\mathbf{Q})/\partial \mathbf{Q}$  is the Jacobian matrix of the system. For the Euler equations,  $\mathbf{A}(\mathbf{Q})$  has three real eigenvalues:  $\lambda_1 = u - a$ ,  $\lambda_2 = u$  and  $\lambda_3 = u + a$ , where a is the *sound speed*. For an ideal gas the sound speed is equal to  $a = \sqrt{\gamma p/\rho}$ . Moreover, the matrix  $\mathbf{A}$  has a complete set of linearly independent eigenvectors, hence the system is hyperbolic. For a thorough discussion of the Euler equations of gasdynamics and numerical methods for their discretization, the reader is referred to the textbook of Toro [69]. Following the flux vector splitting proposed by Toro and Vázquez-Cendón in [70] and the ideas outlined in [55, 26, 24], the PDE system (3) can now be rewritten as

$$\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}^c(\mathbf{Q})}{\partial x} + \frac{\partial \mathbf{F}^p(\mathbf{Q})}{\partial x} = 0,$$
(5)

with the nonlinear convective flux  $\mathbf{F}^c(\mathbf{Q}) = [\rho u, \rho u^2, u \rho k]^T$  that does not contain any pressure terms, and the pressure flux  $\mathbf{F}^p(\mathbf{Q}) = [0, p, h \rho u]^T$ . Following [26] and [24], in this paper we will discretize  $\mathbf{F}^c$  explicitly and  $\mathbf{F}^p$  implicitly. In [70] it was shown that the Jacobian of the convective flux  $\mathbf{A}^c(\mathbf{Q}) = \partial \mathbf{F}^c(\mathbf{Q})/\partial \mathbf{Q}$  has the eigenvalues  $\lambda_1^c = 0$  and  $\lambda_{2,3}^c = u$ , i.e. the CFL condition on the explicit part of the scheme will depend only on the bulk flow velocity u and not on the sound speed a. For this reason, the method presented in this paper can also be used for the simulation of low Mach number flows.

#### 2.2. Unlimited staggered semi-implicit DG scheme for the 1D Euler equations

We consider a computational domain  $\Omega_x = [x_L, x_R]$  composed of two overlapping grids (see Fig. 1). The first one is called main grid, which contains  $N_x$  cells characterized by a constant length  $\Delta x = \frac{L}{N_x} = \frac{x_R - x_L}{N_x}$ , while the second grid is called dual mesh and it has  $N_x + 1$  elements. On this last grid there are  $N_x - 1$  equally spaced elements, while the length of the cells on the left and on the right boundary is only  $\Delta x/2$ . A generic element of the main grid is denoted by  $T_i = [x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}]$ , whereas a control volume on the staggered grid is indicated as  $T_{i+\frac{1}{2}} = [x_i, x_{i+1}]$  and its edges are the barycenters of two consecutive cells of the main grid. In order to represent our numerical solution

![](_page_4_Figure_3.jpeg)

Figure 1: Staggered grids for the one-dimensional DG scheme. Main grid used for the free surface (top) and dual mesh for the velocity (bottom).

via piecewise polynomials of degree P, we use a set of basis functions  $\varphi(\xi)$  defined on the reference element [0,1]. In particular, we choose a *nodal basis* given by the Lagrange interpolation polynomials passing through the Gauss-Legendre quadrature points of the reference element [0,1]. This basis is by construction orthogonal. For any physical control volume we generate the basis functions  $\phi_l(x)$  on the main grid and the basis functions  $\psi_l(x)$  on the dual grid from  $\varphi_l(\xi)$  as

$$\phi_l(x) = \varphi_l(\xi) \quad \text{with} \quad x = x_{i-\frac{1}{2}} + \xi \Delta x, \quad \text{and} \quad \psi_l(x) = \varphi_l(\xi) \quad \text{with} \quad x = x_i + \xi \Delta x, \qquad 0 \le \xi \le 1. \tag{6}$$

Then, a quantity located on the main grid, for example the pressure, is approximated within  $T_i$  as follows

$$p_i(x,t) = \sum_{l=1}^{P+1} \phi_l(x)\hat{p}_{i,l}(t) := \boldsymbol{\phi}(\boldsymbol{x}) \cdot \hat{\boldsymbol{p}}_i(t), \tag{7}$$

while for variables that are defined on the staggered dual mesh, such as the fluid velocity, we have within  $T_{i+\frac{1}{2}}$ 

$$u_{i+\frac{1}{2}}(x,t) = \sum_{l=1}^{P+1} \psi_l(x) \hat{u}_{i+\frac{1}{2},l}(t) := \psi(x) \cdot \hat{u}_{i+\frac{1}{2}}(t)$$
(8)

where  $\hat{p}$  and  $\hat{u}$  are the degrees of freedom for the pressure and for the velocity, respectively. Following the ideas of semi-implicit methods, see e.g. [26, 70, 55], the Euler equations can be split into a convective sub-system, which contains the non linear terms and into a quasi-linear pressure subsystem.

#### 2.2.1. Explicit discretization of the nonlinear convective terms

In the numerical scheme proposed in this paper, the convective subsystem

$$\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}^c(\mathbf{Q})}{\partial x} = 0 \tag{9}$$

is integrated using an *explicit* Runge-Kutta discontinuous Galerkin (RKDG) scheme on the main grid, based on a third order TVD Runge-Kutta method [38]. For this purpose, we define the discrete solution for the vector  $\mathbf{Q}$  within an element  $T_i$  as

$$\mathbf{Q}_i(t) = \phi_l(x)\hat{\mathbf{Q}}_{i,l}(t),\tag{10}$$

where we assume the Einstein summation convention over repeated indices. Multiplication of eq. (9) with a test function  $\phi_k$ , integration over the primary control volume  $T_i$  with subsequent integration by parts and introduction of a numerical flux lead to the following semi-discrete method

$$\int_{T_i} \phi_k(x) \frac{\partial \mathbf{Q}_i}{\partial t} dx + \phi_k(x_{i+\frac{1}{2}}) \mathbf{F}_{i+\frac{1}{2}}^c - \phi_k(x_{i-\frac{1}{2}}) \mathbf{F}_{i-\frac{1}{2}}^c - \int_{T_i} \frac{\partial \phi_k}{\partial x} \mathbf{F}^c(\mathbf{Q}_i) dx = 0.$$
 (11)

The numerical fluxes  $\mathbf{F}^c_{i\pm \frac{1}{2}}$  at the cell interfaces  $x_{i\pm \frac{1}{2}}$  are given by a local Lax-Friedrichs (Rusanov) Riemann solver

$$\mathbf{F}_{i+\frac{1}{2}}^{c} = \frac{1}{2} \left( \mathbf{F}^{c} \left( \mathbf{Q}_{i+\frac{1}{2}}^{+} \right) + \mathbf{F}^{c} \left( \mathbf{Q}_{i+\frac{1}{2}}^{-} \right) \right) - \frac{1}{2} s_{\max} \left( \mathbf{Q}_{i+\frac{1}{2}}^{+} - \mathbf{Q}_{i+\frac{1}{2}}^{-} \right), \tag{12}$$

where  $\mathbf{Q}_{i+\frac{1}{2}}^{\pm}$  are the boundary extrapolated values at the element interfaces from the right and left, respectively. Since in this explicit part of the scheme we only consider the convective subsystem, the maximal signal speed  $s_{\max}$  is given by the eigenvalues  $\lambda^c$  of the convective flux  $\mathbf{F}^c$ , i.e.

$$s_{\max} = \max\left(\max|\lambda^{c}\left(\mathbf{Q}_{i+\frac{1}{2}}^{+}\right)|, \max|\lambda^{c}\left(\mathbf{Q}_{i+\frac{1}{2}}^{-}\right)|\right) = \max\left(|u_{i+\frac{1}{2}}^{+}|, |u_{i+\frac{1}{2}}^{-}|\right). \tag{13}$$

After integration with a third order TVD Runge-Kutta scheme [38] the degrees of freedom of the conservative variables of the convective subsystem at the new time are denoted by  $\widehat{\mathbf{Q}}^*$  and read as follows<sup>1</sup>:

$$\widehat{\mathbf{k}}_{1} = \widehat{\mathbf{Q}}^{n} + \Delta t \mathbf{L}_{h} \left( \widehat{\mathbf{Q}}^{n} \right),$$

$$\widehat{\mathbf{k}}_{2} = \frac{3}{4} \widehat{\mathbf{Q}}^{n} + \frac{1}{4} \widehat{\mathbf{k}}_{1} + \frac{1}{4} \Delta t \mathbf{L}_{h} \left( \widehat{\mathbf{k}}_{1} \right),$$

$$\widehat{\mathbf{Q}}^{*} = \frac{1}{3} \widehat{\mathbf{Q}}^{n} + \frac{2}{3} \widehat{\mathbf{k}}_{2} + \frac{2}{3} \Delta t \mathbf{L}_{h} \left( \widehat{\mathbf{k}}_{2} \right).$$
(14)

In eq. (14) the spatial discretization operator  $\mathbf{L}_h\left(\widehat{\mathbf{Q}}\right)$  is defined for each element  $T_i$  as

$$\mathbf{L}_{h}\left(\widehat{\mathbf{Q}}\right)\Big|_{T_{i}} = -\frac{1}{\Delta x}\mathbf{M}^{-1}\left(\boldsymbol{\varphi}(1)\mathbf{F}_{i+\frac{1}{2}}^{c} - \boldsymbol{\varphi}(0)\mathbf{F}_{i-\frac{1}{2}}^{c} - \int_{0}^{1}\boldsymbol{\varphi}'(\xi)\boldsymbol{\varphi}(\xi)d\xi \cdot \widehat{\mathbf{F}}_{i}^{c}\right),\tag{15}$$

with the element mass matrix on the reference element given by

$$\mathbf{M} = \int_0^1 \boldsymbol{\varphi} \boldsymbol{\varphi} d\xi = \int_0^1 \varphi_k \varphi_l d\xi, \tag{16}$$

and the vector of degrees of freedom of the convective flux simply defined as  $\widehat{\mathbf{F}}_i^c = \mathbf{F}^c \left( \widehat{\mathbf{Q}}_i \right)$ , since we use an orthogonal nodal basis.

 $<sup>^{1}</sup>$ when the index i is omitted in the vector of degrees of freedom we intend the entire set of all degrees of freedom of all elements

#### 2.2.2. Derivation of the semi-implicit staggered DG scheme in 1D

We first discretize the total energy equation obtained after the flux splitting procedure (5). We multiply the total energy equation by the vector of test functions  $\phi$  and we integrate over a control volume of the main grid  $T_i$ :

$$\int_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}} \phi\left(\frac{\partial}{\partial t}(\rho e + \rho k) + \frac{\partial}{\partial x}(\rho k u + h\rho u)\right) dx = 0.$$
(17)

In the time derivative, the term of the total energy term at  $t^{n+1}$  is divided considering the contribution of the specific internal energy and the one of the kinetic energy; in addition we make use of the term  $\widehat{\rho E_i}^*$  that has been obtained after the explicit discretization of the nonlinear convective terms discussed before. Then, we integrate the spatial derivative by parts in space and, in order to achieve second order of accuracy in time, we use the so-called  $\theta$ -method. This leads to

$$\begin{pmatrix}
\int_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}} \phi \phi dx \\
\int_{x_{i-\frac{1}{2}}}^{\infty} \phi \phi dx \end{pmatrix} \cdot \left( \frac{\widehat{\rho} e_{i}^{n+1} + \widehat{\rho} k_{i}^{n+1} - \widehat{\rho} E_{i}^{*}}{\Delta t} \right) \\
+ \phi(x_{i+\frac{1}{2}}^{-}) \psi(x_{i+\frac{1}{2}}) \psi(x_{i+\frac{1}{2}}) \cdot \hat{h}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho} \widehat{u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \phi(x_{i-\frac{1}{2}}^{+}) \psi(x_{i-\frac{1}{2}}) \psi(x_{i-\frac{1}{2}}) \cdot \hat{h}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \widehat{\rho} \widehat{u}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \\
- \begin{pmatrix}
\int_{x_{i+\frac{1}{2}}}^{x_{i}} \frac{\partial \phi}{\partial x} \psi \psi dx \\
\int_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \widehat{\rho} \widehat{u}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} - \begin{pmatrix}
\int_{x_{i+\frac{1}{2}}}^{x_{i+\frac{1}{2}}} \frac{\partial \phi}{\partial x} \psi \psi dx \\
\int_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho} \widehat{u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} = 0,
\end{pmatrix} (18)$$

where  $\widehat{\rho u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}}=(1-\theta_{i+\frac{1}{2}})\widehat{\rho u}_{i+\frac{1}{2}}^{n}+\theta_{i+\frac{1}{2}}\widehat{\rho u}_{i+\frac{1}{2}}^{n+1}$  and  $0.5\leq\theta\leq1$  is an implicitness parameter. When  $\theta=1$  the scheme is the implicit Euler scheme which is first order accurate in time; when  $\theta=0.5$  we have the second-order Crank-Nicolson time discretization [10]. According to the Godunov theorem [37], in this last case the method is not monotone and can generate unphysical oscillations in the vicinity of strong discontinuities. As we will see later, in the framework of sub-cell FV limiters for semi-implicit DG schemes (see [44]) we therefore want to vary the parameter  $\theta$  in the domain and this is done assigning different values of  $\theta$  on the dual mesh. Eq (18) can be rewritten in the compact matrix-vector form

$$\mathbf{M} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{e}}_{i}^{n+1} + \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}_{i}^{n+1} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*}) + \frac{\Delta t}{\Delta x} (\mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}}) = 0.$$
 (19)

where  $\mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}}$  is the product of a rank 3 tensor  $\mathbf{R}_{\mathbf{e}}^{\mathrm{DG}}$  with the degrees of freedom  $\hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}}$ . Moving the quantities of eq. (19) at time  $t^n$  to the right hand side yields

$$\mathbf{M} \cdot \frac{\widehat{\boldsymbol{p}}_{i}^{n+1}}{\gamma - 1} + \mathbf{M} \cdot \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}_{i}^{n+1} + \frac{\Delta t}{\Delta x} (\theta_{i+\frac{1}{2}} \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+1} - \theta_{i-\frac{1}{2}} \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n+1}) = \\ \mathbf{M} \cdot \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*} - \frac{\Delta t}{\Delta x} ((1 - \theta_{i+\frac{1}{2}}) \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n} - (1 - \theta_{i-\frac{1}{2}}) \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n})$$

$$(20)$$

where we expressed the internal energy e as function of p using the ideal gas equation of state. In addition, the meaning of the symbol under-tilde will be clarified later. The momentum equation is multiplied by the test function  $\psi$  and it is integrated over a control volume  $T_{i+\frac{1}{n}}$  that belongs to the dual grid

$$\int_{x_i}^{x_{i+1}} \psi\left(\frac{\partial \rho u}{\partial t} + \frac{\partial \rho u^2}{\partial x}\right) dx = -\int_{x_i}^{x_{i+1}} \psi\frac{\partial p}{\partial x} dx. \tag{21}$$

We carry out the integration of (21) following the same procedure used in [25, 66, 43, 44]. The nonlinear convective terms have already been discretized explicitly, leading to  $\widehat{\rho u}_{i+\frac{1}{2}}^*$ , while the pressure gradient is approximated by splitting it into a smooth contribution and a jump term across the interfaces of the primary grid, which are located in the interior of the dual mesh. This yields

$$\begin{pmatrix}
\int_{x_{i}}^{x_{i+1}} \boldsymbol{\psi} \boldsymbol{\psi} dx \\
\int_{x_{i}}^{x_{i+\frac{1}{2}}} \boldsymbol{\psi} \boldsymbol{\psi} dx \end{pmatrix} \cdot \begin{pmatrix}
\widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2}}^{n+1} - \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2}}^{*} \\
\Delta t
\end{pmatrix} = -\boldsymbol{\psi}(x_{i+\frac{1}{2}}) \begin{pmatrix} \boldsymbol{\phi}(x_{i+\frac{1}{2}}^{+}) \cdot \hat{\boldsymbol{p}}_{i+1}^{n+\theta_{i+\frac{1}{2}}} - \boldsymbol{\phi}(x_{i+\frac{1}{2}}^{-}) \cdot \hat{\boldsymbol{p}}_{i}^{n+\theta_{i-\frac{1}{2}}} \\
- \begin{pmatrix}
\int_{x_{i+\frac{1}{2}}}^{x_{i+\frac{1}{2}}} \boldsymbol{\psi} \frac{\partial \boldsymbol{\phi}}{\partial x} dx \\
\int_{x_{i+\frac{1}{2}}}^{n+\theta_{i-\frac{1}{2}}} - \begin{pmatrix}
\int_{x_{i+\frac{1}{2}}}^{x_{i+1}} \boldsymbol{\psi} \frac{\partial \boldsymbol{\phi}}{\partial x} dx \\
\int_{x_{i+\frac{1}{2}}}^{n+\theta_{i+\frac{1}{2}}} \boldsymbol{\psi} \frac{\partial \boldsymbol{\phi}}{\partial x} dx
\end{pmatrix} \cdot \hat{\boldsymbol{p}}_{i+1}^{n+\theta_{i+\frac{1}{2}}} \tag{22}$$

and the more compact matrix-tensor formulation of the same equation reads as follows:

$$\mathbf{M} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+1} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^*) + \frac{\Delta t}{\Delta x} \left( \mathbf{R}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i+1}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i}^{n+\theta_{i+\frac{1}{2}}} \right) = 0.$$
 (23)

Later on, we introduce the term  $\hat{G}_{i+\frac{1}{2}}^n$  that collects the convective terms of the momentum equation and the discrete pressure gradient at the time level  $t^n$ 

$$\widehat{G}_{i+\frac{1}{2}}^{n} = \widehat{\rho u}_{i+\frac{1}{2}}^{*} - \frac{\Delta t}{\Delta x} (1 - \theta_{i+\frac{1}{2}}) \mathbf{M}^{-1} \cdot (\mathbf{R}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{p}_{i+1}^{n} - \mathbf{L}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{p}_{i}^{n}), \tag{24}$$

so that the final semi-implicit momentum equation reads

$$\widehat{\rho u}_{i+\frac{1}{2}}^{n+1} = \widehat{G}_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x} \theta_{i+\frac{1}{2}} \mathbf{M}^{-1} \cdot (\mathbf{R}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{p}_{i+1}^{n+1} - \mathbf{L}_{\mathbf{p}}^{\mathrm{DG}} \cdot \widehat{p}_{i}^{n+1}). \tag{25}$$

Later, we will insert the discrete momentum equation (25) into the discrete total energy equation (20). However, the enthalpy h is a function of the pressure, and in order to avoid a strongly non-linear system in p we adopt an iterative Picard technique. Hence, the symbol under-tilde means that a variable is evaluated at the previous Picard iteration. In the following we will use the superscript r as the iteration index of the Picard process. For the sake of clarity we rewrite eq. (20) as

$$\mathbf{M} \cdot \frac{\hat{\boldsymbol{p}}_{i}^{n+1,r+1}}{\gamma - 1} + \frac{\Delta t}{\Delta x} (\theta_{i+\frac{1}{2}} \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1,r} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+1,r+1} - \theta_{i-\frac{1}{2}} \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1,r} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n+1,r+1}) = \\ \mathbf{M} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}_{i}^{n+1,r}) - \frac{\Delta t}{\Delta x} ((1 - \theta_{i+\frac{1}{2}}) \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n} - (1 - \theta_{i-\frac{1}{2}}) \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n}).$$
(26)

The discrete momentum equation (25) becomes

$$\widehat{\rho u}_{i+\frac{1}{2}}^{n+1,r+1} = \widehat{G}_{i+\frac{1}{2}}^{n} - \frac{\Delta t}{\Delta x} \theta_{i+\frac{1}{2}} \mathbf{M}^{-1} \cdot (\mathbf{R}_{\mathbf{p}}^{DG} \cdot \widehat{p}_{i+1}^{n+1,r+1} - \mathbf{L}_{\mathbf{p}}^{DG} \cdot \widehat{p}_{i}^{n+1,r+1}).$$
(27)

Therefore, after inserting (27) into (26) one gets a discrete wave equation for the unknown pressure  $p^{n+1,r+1}$  at the next Picard iteration, which can be solved by using a Thomas algorithm for linear block three-diagonal systems, and which reads

$$\mathcal{L}_{i}^{\text{DG}} \cdot \hat{p}_{i-1}^{n+1,r+1} + \mathcal{C}_{i}^{\text{DG}} \cdot \hat{p}_{i}^{n+1,r+1} + \mathcal{R}_{i}^{\text{DG}} \cdot \hat{p}_{i+1}^{n+1,r+1} = \hat{b}_{i}^{r}.$$
(28)

Here,  $\hat{b}_{i}^{r}$  is the known right hand side term

$$\hat{\boldsymbol{b}}_{i}^{r} = \mathbf{M} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}_{i}^{n+1,r}) - \frac{\Delta t}{\Delta x} (\theta_{i+\frac{1}{2}} \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1,r} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{*} - \theta_{i-\frac{1}{2}} \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1,r} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{*}) - \frac{\Delta t}{\Delta x} ((1 - \theta_{i+\frac{1}{2}}) \mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2}}^{n} - (1 - \theta_{i-\frac{1}{2}}) \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2}}^{n}). \tag{29}$$

The block matrices  $\mathcal{L}_i^{\mathrm{DG}}$ ,  $\mathcal{C}_i^{\mathrm{DG}}$  and  $\mathcal{R}_i^{\mathrm{DG}}$  read

$$\mathcal{L}_{i}^{\text{DG}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i-\frac{1}{2}}^{2} \mathbf{L}_{\mathbf{e}}^{\text{DG}} \hat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1,r} \mathbf{M}^{-1} \mathbf{L}_{\mathbf{p}}^{\text{DG}},$$
(30)

$$C_{i}^{DG} = \frac{1}{\gamma - 1} \mathbf{M} + \frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i + \frac{1}{2}}^{2} \mathbf{R}_{e}^{DG} \hat{\boldsymbol{h}}_{i + \frac{1}{2}}^{n+1,r} \mathbf{M}^{-1} \mathbf{L}_{p}^{DG} + \frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i - \frac{1}{2}}^{2} \mathbf{L}_{e}^{DG} \hat{\boldsymbol{h}}_{i - \frac{1}{2}}^{n+1,r} \mathbf{M}^{-1} \mathbf{R}_{p}^{DG},$$
(31)

$$\mathcal{R}_{i}^{\text{DG}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i+\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\text{DG}} \hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1,r} \mathbf{M}^{-1} \mathbf{R}_{\mathbf{p}}^{\text{DG}}.$$
 (32)

For every Picard iteration, once the pressure is known, the momentum update is done using eq. (27). Subsequently, also the enthalpy  $\hat{h}_{i+\frac{1}{2}}^{n+1,r+1}$  has to be computed and, in order to get a quantity from the main grid to the dual one and back, we use the following averaging operator based on  $L_2$  projection:

$$\hat{\boldsymbol{p}}_{i+\frac{1}{2}} = \mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{p}}_i + \mathbf{M}_{\mathbf{R}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{p}}_{i+1}). \tag{33}$$

In addition, since in eq. (29) the kinetic energy is needed on the main grid, we first compute the degrees of freedom of the velocity as  $\hat{u}_{i+\frac{1}{2}}^{n+1,r+1} = \widehat{\rho u}_{i+\frac{1}{2}}^{n+1,r+1}/\widehat{\rho}_{i+\frac{1}{2}}^{n+1}$  and then we apply the  $L_2$  averaging operator from the dual to the main grid

$$\hat{\boldsymbol{u}}_{i} = \mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{u}}_{i-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2}}). \tag{34}$$

Numerical experiments indicate that only very few Picard iterations, typically 2 or 3, are enough to obtain a good solution. Finally, at the end of the Picard loop the update of the total energy is carried out and from eq. (19) we get

$$\widehat{\rho E}_{i}^{n+1} = \widehat{\rho E}_{i}^{*} - \frac{\Delta t}{\Delta x} \mathbf{M}^{-1} \cdot (\mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \widehat{\rho u}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}}).$$
(35)

For this semi-implicit DG scheme, the maximum time step is given by the usual CFL condition of the RKDG scheme which reads

$$\Delta t < \text{CFL} \frac{\Delta x}{\max(|u|)} \quad \text{with} \quad \text{CFL} < (2P+1)^{-1}.$$
 (36)

Note that this is the stability condition due to the explicit discretization of the nonlinear convective terms and which is based only on the fluid velocity. Due to the implicit treatment of the pressure terms, the speed of sound does not influence the choice of Δt. Consequently, this method becomes suitable for the numerical simulation of low Mach number flows. However, due to its conservative formulation, the method is also able to deal with shock waves, as shown later. For other types of semi-implicit DG schemes on staggered meshes, the reader is referred to [25, 43, 63, 65, 66, 33, 34]. In addition, the scheme derived here represents the natural extension to arbitrary order of accuracy in space of the semi-implicit 1D finite volume method introduced in [26].

### 2.3. A sub-cell formulation for the semi-implicit finite volume method for the 1D Euler equations

In this subsection we introduce a sub-cell formulation of the semi-implicit finite volume method for the 1D Euler equations. In particular this approach was adopted in [44] for the shallow water equations but other similar methods have been proposed also in [7, 61, 48]. The computational domain is  $\Omega_x = [x_L, x_R]$  and, similarly to the method in subsection 2.2, we consider two staggered meshes. For an element  $T_i$  of the main grid, given a positive integer P, there are 2P+1 piecewise constant sub-cell averages that represent the FV data on 2P+1 sub-cells  $T_{i,s}$  characterized by length  $\Delta x_s = \Delta x/(2P+1)$ . Consequently we have that  $T_i = \bigcup_s T_{i,s}$  with s=1,...,(2P+1). Hence, for P=2,  $\overline{p}_i = [\overline{p}_{i,1}, \overline{p}_{i,2}, \overline{p}_{i,3}, \overline{p}_{i,4}, \overline{p}_{i,5}]$  denotes the set of sub-cell data in the cell  $T_i$ . For a schematic representation of these grids see Fig.2 The quantities assigned to the main grid and to the dual grid are the same discussed in the previous section 2.2. For the convective sub-system, the explicit finite volume update on a collocated grid at the subcell level is given by

$$\overline{Q}_{i,s}^* = \overline{Q}_{i,s}^n - \frac{\Delta t}{\Delta r} (\mathbf{F}_{i,s+\frac{1}{2}}^c - \mathbf{F}_{i,s-\frac{1}{2}}^c), \tag{37}$$

![](_page_9_Figure_0.jpeg)

Figure 2: Staggered grids for the sub-cell finite volume scheme in one dimension in the case P = 2 ( $N_s = 5$ ). Main grid used for the free surface (top) and staggered dual mesh for the velocity (bottom).

where we used again the Rusanov method in order to evaluate the numerical fluxes  $\mathbf{F}_{i,s\pm\frac{1}{2}}^c$ . We start with the semi-implicit approximation of the momentum equation. Using the adaptive  $\theta$ -method, the discretization reads as follows

$$(\overline{\rho}\overline{u}_{i+\frac{1}{2}}^{n+1} - \overline{\rho}\overline{u}_{i+\frac{1}{2}}^*) + \frac{\Delta t}{\Delta x}(\mathbf{R}^{\text{FV}} \cdot \overline{p}_{i+1}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}^{\text{FV}} \cdot \overline{p}_{i}^{n+\theta_{i+\frac{1}{2}}}) = 0$$

$$(38)$$

where  $\overline{\rho u}_{i+\frac{1}{2}}^*$  is the contribution of the explicit discretization of the nonlinear convective term; in addition, the tensors  $\mathbf{R}^{\mathrm{FV}}$  and  $\mathbf{L}^{\mathrm{FV}}$  (see Appendix Appendix A.2) approximate the derivative in x-direction. For the sake of clarity, we give the approximation of the pressure gradient  $\Delta x \partial p / \partial x$  with P=2 and  $\theta_{i+\frac{1}{n}}=1$ :

$$\mathbf{R}^{\text{FV}} \cdot \overline{p}_{i+\frac{1}{2}}^{n+1} - \mathbf{L}^{\text{FV}} \cdot \overline{p}_{i-\frac{1}{2}}^{n+1} = 5(\overline{p}_{i-\frac{1}{2},4}^{n+1} - \overline{p}_{i-\frac{1}{2},5}^{n+1} - \overline{p}_{i-\frac{1}{2},4}^{n+1}, \overline{p}_{i+\frac{1}{2},1}^{n+1} - \overline{p}_{i-\frac{1}{2},5}^{n+1}, \overline{p}_{i+\frac{1}{2},2}^{n+1} - \overline{p}_{i+\frac{1}{2},2}^{n+1}, \overline{p}_{i+\frac{1}{2},2}^{n+1} - \overline{p}_{i+\frac{1}{2},2}^{n+1}).$$
(39)

The semi-implicit sub-cell finite volume discretization of the total energy equation takes the form

$$\frac{1}{\gamma - 1} \overline{\boldsymbol{p}}_{i}^{n+1,r+1} + \frac{\Delta t}{\Delta x} \left( \boldsymbol{\theta}_{i+\frac{1}{2}} \mathbf{R}^{\text{FV}} \cdot \overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1,r} \overline{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+1,r+1} - \boldsymbol{\theta}_{i-\frac{1}{2}} \mathbf{L}^{\text{FV}} \cdot \overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1,r} \overline{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i-\frac{1}{2}}^{n+1,r+1} \right) =$$

$$\left( \overline{\boldsymbol{\rho}} \overline{\boldsymbol{E}}_{i}^{*} - \overline{\boldsymbol{\rho}} \overline{\boldsymbol{k}}_{i}^{n+1,r} \right) - \frac{\Delta t}{\Delta x} \left( (1 - \boldsymbol{\theta}_{i+\frac{1}{2}}) \mathbf{R}^{\text{FV}} \cdot \overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n} \overline{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i+\frac{1}{2}}^{n} - (1 - \boldsymbol{\theta}_{i-\frac{1}{2}}) \mathbf{L}^{\text{FV}} \cdot \overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n} \overline{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i-\frac{1}{2}}^{n} \right).$$

$$(40)$$

Inserting eq. (38) into eq. (40) yields a linear three-diagonal system which at each Picard level reads

$$\mathcal{L}_{i}^{\text{FV}} \cdot \overline{p}_{i-1}^{n+1,r+1} + \mathcal{C}_{i}^{\text{FV}} \cdot \overline{p}_{i}^{n+1,r+1} + \mathcal{R}_{i}^{\text{FV}} \cdot \overline{p}_{i+1}^{n+1,r+1} = \overline{b}_{i}^{r}, \tag{41}$$

and which can be solved either by a Thomas algorithm or by an iterative conjugate gradient method. In each Picard iteration the enthalpy, which is defined on the staggered mesh, has to be updated and one can use the following FV average in order to interpolate the pressure from the main grid to the dual mesh

$$\overline{p}_{i+\frac{1}{2}} = (\mathbf{M}_L^{\text{FV}} \cdot \overline{p}_i + \mathbf{M}_R^{\text{FV}} \cdot \overline{p}_{i+1}). \tag{42}$$

Similarly, the kinetic energy is discretized on the main grid. So, in order to get the degrees of freedom of the velocity  $\overline{u}_i$ , the projection from the dual grid to main one reads

$$\overline{\boldsymbol{u}}_{i} = (\mathbf{M}_{L}^{\text{FV}} \cdot \overline{\boldsymbol{u}}_{i-\frac{1}{2}} + \mathbf{M}_{R}^{\text{FV}} \cdot \overline{\boldsymbol{u}}_{i+\frac{1}{2}}). \tag{43}$$

For the matrices introduced in equations (42) and (43), see appendix Appendix A.2. Finally, the total energy update reads

$$\overline{\rho}\overline{E}_{i}^{n+1} = \overline{\rho}\overline{E}_{i}^{*} - \frac{\Delta t}{\Delta x} \cdot (\mathbf{R}^{\text{FV}} \cdot \overline{h}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \overline{\rho} \overline{u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}^{\text{FV}} \cdot \overline{h}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \overline{\rho} \overline{u}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}}). \tag{44}$$

In this case, the CFL condition of the scheme is given by the stability condition for the computation of the explicit nonlinear convective terms:

$$\Delta t < \text{CFL} \frac{\Delta x_s}{\max(|u|)}$$
 with CFL < 1. (45)

We remind that here we introduced an alternative data representation for the FV method which is necessary in order to combine the DG and FV schemes on staggered meshes for the *a posteriori* limiting strategy introduced in the next section. For the method discussed in this subsection, the solution algorithm is the same adopted for the 1D pure DG scheme in Section 2.2. For a better understanding of this particular semi-implicit FV method see [26, 44].

#### 2.4. MOOD algorithm and detection criteria - 1D case

In this paper, we employ the limiting strategy proposed in [44] for staggered SIDG schemes in the context of the shallow water equations. Similarly to the *a posteriori* sub-cell limiter for explicit DG schemes exposed in [32, 29, 6], the starting point is the MOOD paradigm [16, 21, 22, 53]: at each time step the algorithm is composed of two stages. In the first one, the unlimited DG scheme presented in section 2.2 generates a so-called *candidate solution*  $Q^{\circ,n+1} = (\rho^{\circ,n+1},\rho u^{\circ,n+1},\rho E^{\circ,n+1})^T$  at time  $t^{n+1}$ . Then, the troubled zones are detected using physical and numerical admissibility criteria. In the cells flagged as troubled zones, the more robust staggered semi-implicit finite volume subcell scheme introduced in the previous section 2.3 is used, while we continue using the unlimited high order DG method in those control volumes which are not troubled. Afterwards, in the second part of the procedure, a valid solution at time  $t^n$  is recovered and then the update is carried out by a *mixed* scheme, which uses the more robust subcell finite volume method in troubled cells and the unlimited DG method in the other ones. To do this, the linear system for the pressure is reassembled and solved again. Moreover, if no troubled cells have been identified after the first stage of the algorithm, the scheme directly proceeds to the next time level.

#### 2.4.1. Data representation, projection and reconstruction

For a generic variable q of the approximated solution given by the DG scheme in the cell  $T_i$  at time  $t^n$ , the degrees of freedom used for the piecewise polynomial data representation are denoted by  $\hat{q}_i^n$ . Similarly  $\overline{q}_i^n$  denote the piecewise constant cell averages in the subcells  $T_{i,s}$ . These subcell averages  $\overline{q}_i^n$  are computed from  $\hat{q}_i^n$  using the following  $L_2$  projection

$$\overline{q}_{i,s}^n = \frac{1}{|T_{i,s}|} \int_{T_{i,s}} \phi_l(x) dx \, \hat{q}_{i,l}^n, \qquad \forall T_{i,s} \in T_i, \tag{46}$$

and consequently we introduce the projection operator  $\mathcal{P}$  so that  $\overline{q}_i^n = \mathcal{P} \cdot \hat{q}_i^n$ .

Then, one can gather back the piecewise constant sub-cell averages  $\bar{q}_{i,s}^n$  into the degrees of freedom  $\hat{q}_{i,l}^n$  of a high order DG polynomial by solving again (46). However, these procedure leads to an *over-determined* system because there are 2P+1 equations for P+1 unknowns. In order to overcome this problem a *constrained least-squares* approach (see [28, 46]) is adopted. In particular, the linear constraint is integral conservation on the big cell  $T_i$  (see [32]), i.e.

$$\sum_{j} |T_{i,s}| \overline{q}_{i,s}^n = \int_{T_i} \phi_l(x) dx \, \hat{q}_{i,l}^n. \tag{47}$$

This operation is expressed in a tensorial formulation as  $\hat{q}_i^n = \mathcal{W} \cdot \overline{q}_i^n$ , with  $\mathcal{W} \cdot \mathcal{P} = \mathcal{I}$ , where  $\mathcal{I}$  denotes the identity matrix. Moreover, in some situations we will need to recover a polynomial from P+1 piecewise constant sub-cell data. Here we consider two situations: one is at the left part,  $T_i^L = \bigcup_{s=1}^{P+1} T_{i,s}$  (see 3b), and the other at the right side,  $T_i^R = \bigcup_{s=P+1}^{2P+1} T_{i,s}$  (see 3c). So we introduce the matrices  $\mathcal{W}_{\mathcal{L}}$  and  $\mathcal{W}_{\mathcal{R}}$  that are computed using reconstruction procedure (46) only on the left and right part of cell  $T_i$ . Note that here the least squares procedure is not necessary because now the number of equations is equal to the number of unknowns. For a schematic summary of the projection and reconstruction procedures described above, see Figure 3a.

#### 2.4.2. Detection criteria

In this subsection we discuss the physical and numerical admissibility criteria. All the quantities are analyzed on the *main grid* and those variables which are located on the staggered dual grid are extrapolated onto the main grid. This operation is done by using the algorithm given in eq. (33).

At first, we want that the candidate solution, given numerically by the unlimited staggered DG scheme, satisfies some *physical admissibility criteria* (PAD). For the Euler equations we check that both density and pressure are positive, hence we require

$$\hat{\rho}_{i,l}^{\circ,n+1} > 0 \quad \text{and} \quad \hat{p}_{i,l}^{\circ,n+1} > 0.$$
 (48)

In addition, similarly to [32, 29, 6], the detector is supplemented by *numerical admissibility criteria* (NAD) based on a relaxed version of the discrete maximum principle (DMP). For a generic variable q this criterion reads as follows

$$\min_{\forall T_j \in \mathcal{V}_i} (\hat{q}_{j,l}^n) - \delta \le \hat{q}_{i,l}^{\circ, n+1} \le \max_{\forall T_j \in \mathcal{V}_i} (\hat{q}_{j,l}^n) + \delta, \tag{49}$$

where  $V_i = \{T_{i-1}, T_i, T_{i+1}\}$  is the set of neighbour cells of  $T_i$ , which is the set of Voronoi neighbours in the multidimensional case. In equation (49) we introduce two tolerances:  $\delta$  is a relaxation parameter that allows some very small oscillations or numerical noise; it reads

$$\delta = \max[\delta_0, \epsilon(\max_{\forall T_i \in \mathcal{V}_i} (\hat{q}_{j,l}^n) - \min_{\forall T_i \in \mathcal{V}_i} (\hat{q}_{j,l}^n))]$$
(50)

where  $\delta_0$  is chosen in the interval  $[10^{-4}, 10^{-3}]$  and typically  $\epsilon = 5 \cdot 10^{-4}$ . In this algorithm we check the DMP componentwise for all conserved variables. Moreover, an additional detection of troubled cells is carried out by projecting all numerical quantities from the main grid to the dual mesh and then projecting them back onto the main grid. This procedure creates non-physical oscillations near discontinuities and thus leads to a more robust detection of troubled cells, especially for the initial condition.

Successively, a cell is flagged as troubled if it does not respect the *a posteriori* admissibility criteria applied to the candidate solution  $Q^{\circ,n+1}$ . We therefore introduce a troubled cell indicator parameter  $\beta_i$ . If  $\beta_i=1$  then a cell is marked as troubled and the solution at time  $t^{n+1}$  will be recomputed using the more robust subcell finite volume scheme presented in Section 2.3. On the contrary, if  $\beta_i=0$  in  $T_i$  it means that the cell is not troubled. In addition, if the cell  $T_i$  is flagged as troubled we also mark both dual control volumes  $T_{i-\frac{1}{2}}$  and  $T_{i+\frac{1}{2}}$  as troubled control volumes, see Figure 4. As consequence, we impose  $\beta_{i-\frac{1}{3}}=1$  and  $\beta_{i+\frac{1}{3}}=1$ .

Finally, for the sake of clarity we remind that in the first MOOD iteration the implicitness parameter  $\theta$  is a constant value  $\theta_{DG}$  in the whole domain. It is a user-defined parameter which is typically chosen very close to 1/2 in order to minimize the effects of the numerical viscosity caused by the time integration. At the second MOOD level, the value of the parameter  $\theta$  in the troubled cells of dual grid is imposed equal to 1. Consequently, the new numerical solution is given by using a monotone and robust finite volume method in the troubled cells.

#### 2.5. Sub-cell limiting of the semi-implicit DG scheme for the 1D Euler equations

Finally, we complete the derivation of the 1D algorithm for the finite volume limiting of the Euler equations. The computation of the nonlinear convective terms is carried out according to the MOOD strategy used for explicit high order schemes, see [16, 21, 22, 53, 32, 29, 6]. Hence, at the beginning of the first MOOD step all the cells are flagged with  $\beta_i = 0$  and the nonlinear convective terms are computing via the high order unlimited RKDG algorithm explained in subsection 2.2.1. Later on, after the detection of the troubled control volumes, we use the explicit finite volume update given in 2.3 in order to recompute the solution in the cells marked with  $\beta_i = 1$ , while for the cells which are not troubled we keep the results of the first MOOD step.

Now, we consider the semi-implicit approximation of the one-dimensional total energy and momentum equations for the complete algorithm. For the general case we distinguish four situations. The first one is when the control volume  $T_i$  and the corresponding dual cells  $T_{i\pm}$  are not marked as troubled. In this case  $\beta_{i-\frac{1}{2}}=\beta_i=\beta_{i+\frac{1}{2}}=0$  and the equations are discretized using the pure DG method introduced in subsection 2.2. The opposite case is when the troubled cell indicator  $\beta_{i-\frac{1}{2}}=\beta_i=\beta_{i+\frac{1}{2}}=1$  is equal to one in all the control volumes  $T_{i-\frac{1}{2}},T_i,T_{i+\frac{1}{2}};$  for this situation, we use the finite volume method discussed in 2.3.

![](_page_12_Figure_0.jpeg)

Figure 3: Projection and reconstruction operators.

![](_page_12_Figure_2.jpeg)

Figure 4: Staggered grids for the semi-implicit staggered DG scheme with sub-cell limiter active in cell  $T_i$ . Main grid used for the free surface (top) and staggered dual mesh for the velocity (bottom). If a cell  $T_i$  on the main grid is flagged as troubled, then also the two overlapping staggered velocity control volumes  $T_{i\pm\frac{1}{2}}$  are flagged as troubled. The data representation in troubled cells is changed from high order polynomials to piecewise constant subcell averages.

![](_page_13_Picture_0.jpeg)

![](_page_13_Picture_1.jpeg)

- (a) DG method used in  $T_i$  and  $T_{i-\frac{1}{2}}$ ; finite volumes in  $T_{i+\frac{1}{2}}$
- (b) DG method used in  $T_i$  and  $T_{i+\frac{1}{2}}$ ; finite volumes in  $T_{i-\frac{1}{2}}$ .

Figure 5: Control volumes for the total energy equation for the staggered semi-implicit DG scheme in the limited case using P=2, i.e.  $N_s=2P+1=5$ . Finite volume method used in  $T_{i+\frac{1}{5}}$  (left) and in  $T_{i-\frac{1}{5}}$  (right).

In the following we provide two examples when cell  $T_i$  is not troubled but one of its neighbors is marked as troubled. In this case, the discrete energy conservation law reads

$$\mathbf{M} \cdot (\frac{\widehat{\rho E}_{i}^{n+1} - \widehat{\rho E}_{i}^{*}}{\Delta t}) + \frac{\mathbf{R}_{\mathbf{e}}^{\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \widehat{\rho u}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathrm{Lim}} \cdot \overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \overline{\rho u}_{i-\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}}}{\Delta x} = 0, \quad \text{with} \quad \mathbf{L}_{\mathbf{e}}^{\mathrm{Lim}} = \mathbf{L}_{\mathbf{e}}^{\mathrm{DG}} \cdot \mathcal{W}_{\mathcal{R}}$$

$$(51)$$

for  $\beta_i = 0$ ,  $\beta_{i-1} = 0$  and  $\beta_{i+1} = 1$  and

$$\mathbf{M} \cdot (\frac{\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{n+1} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*}}{\Delta t}) + \frac{\mathbf{R}_{\mathbf{e}}^{\text{Lim}} \cdot \overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} \overline{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i+\frac{1}{2}}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\text{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}} \widehat{\boldsymbol{\rho}} \overline{\boldsymbol{u}}_{i-\frac{1}{2}}^{n+\theta_{i-\frac{1}{2}}}}{\Delta x} = 0, \text{ with } \mathbf{R}_{\mathbf{e}}^{\text{Lim}} = \mathbf{R}_{\mathbf{e}}^{\prime \text{DG}} \cdot \mathcal{W}_{\mathcal{L}}.$$
(52)

for  $\beta_i=0$ ,  $\beta_{i-1}=1$  and  $\beta_{i+1}=0$ . These two approximations are very similar to eq. (19) the discretization of the energy equation in the pure DG case. In fact, considering the scheme in Figure 5a, the operator  $\mathcal{W}_{\mathcal{L}}$  reconstructs the polynomial data using the first P+1 subcell averages of the cell  $T_{i+\frac{1}{2}}$ . Similarly  $\mathcal{W}_{\mathcal{R}}$  reconstructs the polynomial data from the last P+1 piecewise constant data of the control volume  $T_{i-\frac{1}{2}}$ , see Figure 52. In addition, in equations (51) and (51) we introduced the matrices  $\mathbf{R}'_{\mathbf{e}}^{\mathrm{DG}}$  and  $\mathbf{L}'_{\mathbf{e}}^{\mathrm{DG}}$  that are rank 2 tensors and which are explicitly given in Appendix Appendix A.3.

Now we discuss the two limited cases of the discrete momentum equation. Here, we consider the cases when  $T_{i+\frac{1}{2}}$  is a troubled cell and only one of the control volumes  $T_i$  and  $T_{i+1}$  is a limited control volume. Hence the numerical approximation of the momentum equation reads as follows

$$\overline{\rho}\overline{u}_{i+\frac{1}{2}}^{n+1} = \overline{\rho}\overline{u}_{i+\frac{1}{2}}^* - \frac{\Delta t}{\Delta x} (\mathbf{R}_{\mathbf{u}}^{\text{Lim}} \cdot \hat{p}_{i+1}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}^{\text{FV}} \cdot \overline{p}_{i}^{n+\theta_{i+\frac{1}{2}}}) \quad \text{with} \quad \mathbf{R}_{\mathbf{u}}^{\text{Lim}} = \mathbf{R}^{\text{FV}} \cdot \mathcal{P}$$
(53)

when  $\beta_i=0$  and  $\beta_{i+1}=1$  (see Fig. 6a) and

$$\overline{\rho}\overline{u}_{i+\frac{1}{2}}^{n+1} = \overline{\rho}\overline{u}_{i+\frac{1}{2}}^* - \frac{\Delta t}{\Delta r} (\mathbf{R}^{\text{FV}} \cdot \overline{p}_{i+1}^{n+\theta_{i+\frac{1}{2}}} - \mathbf{L}_{\mathbf{u}}^{\text{Lim}} \cdot \hat{p}_{i}^{n+\theta_{i+\frac{1}{2}}}) \quad \text{with} \quad \mathbf{L}_{\mathbf{u}}^{\text{Lim}} = \mathbf{L}^{\text{FV}} \cdot \mathcal{P}$$
(54)

when  $\beta_i=1$  and  $\beta_{i+1}=0$  (see Fig. 6b). Note that equations (53) and 53 can be seen as particular case of the discrete momentum equation (38) for the sub-cell finite volume method. In addition, in eq. (53), the product  $\mathbf{R}_{\mathbf{u}}^{\mathrm{Lim}} \cdot \hat{\boldsymbol{p}}_{i+1}^{n+\theta_{i+\frac{1}{2}}} = \mathbf{R}^{\mathrm{FV}} \cdot \overline{\boldsymbol{p}}_{i+1}^{n+\theta_{i+\frac{1}{2}}}$  is equal to the multiplication of the tensor  $\mathbf{R}^{\mathrm{FV}}$  with the vector of piecewise constant subcell averages

![](_page_14_Figure_0.jpeg)

Figure 6: Control volumes for the momentum equation for the staggered semi-implicit DG scheme in the limited case using P=2 ( $N_s=5$ ).

 $\overline{p}_{i+1}^{n+\theta_{i+\frac{1}{2}}} = \mathcal{P} \cdot \hat{p}_{i+1}^{n+\theta_{i+\frac{1}{2}}}$  given by the projection of the DG degrees of freedom onto the set of the subcell finite volume degrees of freedom. This observation is valid also for the tensor product  $\mathbf{L}_{\mathbf{u}}^{\mathrm{Lim}} \cdot \hat{p}_{i}^{n+\theta_{i+\frac{1}{2}}} = \mathbf{L}^{\mathrm{FV}} \cdot \overline{p}_{i}^{n+\theta_{i+\frac{1}{2}}}$  in eq. (54). The substitution of the momentum equation into the energy equation yields the following block three-diagonal linear system (55):

$$\mathcal{L}_{i}^{\operatorname{Lim}} \cdot \tilde{\boldsymbol{p}}_{i-1}^{n+1,r+1} + \mathcal{C}_{i}^{\operatorname{Lim}} \cdot \tilde{\boldsymbol{p}}_{i}^{n+1,r+1} + \mathcal{R}_{i}^{\operatorname{Lim}} \cdot \tilde{\boldsymbol{p}}_{i+1}^{n+1,r+1} = \tilde{\boldsymbol{b}}_{i}^{r}. \tag{55}$$

We denote with the symbol  $\tilde{p}_i^{n+1}$  the generic degrees of freedom of the pressure which represent either a DG polynomial or a set of subcell averages depending on the value of the troubled cell indicator  $\beta$ . In the following, we will give the expressions for the blocks  $\mathcal{L}_i^{\mathrm{Lim}}$ ,  $\mathcal{C}_i^{\mathrm{Lim}}$ ,  $\mathcal{R}_i^{\mathrm{Lim}}$  and for the right hand side term  $\tilde{\boldsymbol{b}}_i^r$ . For

clarity,  $C_i^{\text{Lim}}$  is divided in two contributions

$$C_i^{\text{Lim}} = C_i^{0,\text{Lim}} + C_i^{x,\text{Lim}},\tag{56}$$

where  $C_i^{0,\mathrm{Lim}}$  is equal to the mass matrix in the DG case or equal to the identity for the subcell FV scheme:

$$C_i^{0,\text{Lim}} = \begin{cases} \mathbf{M} & \text{if} \quad \beta_i = 0, \\ \mathbf{I} & \text{if} \quad \beta_i = 1. \end{cases}$$
 (57)

The remaining blocks  $\mathcal{L}_i^{\text{Lim}}$ ,  $C_i^{x,\text{Lim}}$ ,  $\mathcal{L}_i^{\text{Lim}}$  can be expressed considering four different cases depending on the value of

$$\mathcal{L}_{i}^{\text{Lim}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i-\frac{1}{2}}^{2} \begin{cases}
\mathbf{L}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}_{i}^{n+1} \cdot \mathbf{M}^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\text{DG}} & \text{if} & \beta_{i-1} = \beta_{i} = 0, \\
\mathbf{L}^{\text{FV}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1}) \cdot \mathbf{L}^{\text{FV}} & \text{if} & \beta_{i-1} = \beta_{i} = 1, \\
\mathbf{L}_{\mathbf{e}}^{\text{Lim}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1}) \cdot \mathbf{L}^{\text{FV}} & \text{if} & \beta_{i-1} = 1, \beta_{i} = 0, \\
\mathbf{L}^{\text{FV}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i-\frac{1}{2}}^{n+1}) \cdot \mathbf{L}_{\mathbf{u}}^{\text{Lim}} & \text{if} & \beta_{i-1} = 0, \beta_{i} = 1,
\end{cases} (58)$$

$$\mathcal{C}_{i}^{x,\text{Lim}} = \frac{\Delta t^{2}}{\Delta x^{2}} \begin{cases} \theta_{i-\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot \mathbf{M}^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\text{DG}} + \theta_{i+\frac{1}{2}}^{2} \mathbf{L}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot \mathbf{M}^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\text{DG}} & \text{if} \quad \beta_{i} = \beta_{i-\frac{1}{2}} = \beta_{i+\frac{1}{2}} = 0, \\ \theta_{i-\frac{1}{2}}^{2} \mathbf{L}^{\text{FV}} \cdot \text{diag}(\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{R}^{\text{FV}} + \theta_{i+\frac{1}{2}}^{2} \mathbf{R}^{\text{FV}} \cdot \text{diag}(\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}^{\text{FV}} & \text{if} \quad \beta_{i} = \beta_{i-\frac{1}{2}} = \beta_{i+\frac{1}{2}} = 1, \\ \theta_{i-\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot \mathbf{M}^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\text{DG}} + \theta_{i+\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\text{Lim}} \cdot \text{diag}(\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}_{\mathbf{u}}^{\text{Lim}} & \text{if} \quad \beta_{i} = \beta_{i-\frac{1}{2}} = 0, \beta_{i+\frac{1}{2}} = 1, \\ \theta_{i-\frac{1}{2}}^{2} \mathbf{L}_{\mathbf{e}}^{\text{Lim}} \cdot \text{diag}(\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{R}_{\mathbf{u}}^{\text{Lim}} + \theta_{i+\frac{1}{2}}^{2} \mathbf{L}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot \mathbf{R}_{\mathbf{u}}^{\text{DG}} & \text{if} \quad \beta_{i} = \beta_{i+\frac{1}{2}} = 0, \beta_{i-\frac{1}{2}} = 1, \end{cases}$$

$$(59)$$

$$\mathcal{R}_{i}^{\text{Lim}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i+\frac{1}{2}}^{2} \begin{cases}
\mathbf{R}_{\mathbf{e}}^{\text{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1} \cdot \mathbf{M}^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\text{DG}} & \text{if} & \beta_{i+1} = \beta_{i} = 0, \\
\mathbf{R}^{\text{FV}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1}) \cdot \mathbf{R}^{\text{FV}} & \text{if} & \beta_{i+1} = \beta_{i} = 1, \\
\mathbf{R}_{\mathbf{e}}^{\text{Lim}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1}) \cdot \mathbf{R}^{\text{FV}} & \text{if} & \beta_{i+1} = 1, \beta_{i} = 0, \\
\mathbf{R}^{\text{FV}} \cdot \operatorname{diag}(\overline{\boldsymbol{h}}_{i+\frac{1}{2}}^{n+1}) \cdot \mathbf{R}_{\mathbf{u}}^{\text{Lim}} & \text{if} & \beta_{i+1} = 0, \beta_{i} = 1.
\end{cases} (60)$$

Finally the right hand side term reads

$$\tilde{\boldsymbol{b}}_{i}^{n} = \tilde{\boldsymbol{b}}_{i}^{0,n} + \tilde{\boldsymbol{b}}_{i}^{x,n} \tag{61}$$

with

$$\widetilde{\boldsymbol{b}}_{i}^{0,n} = \begin{cases}
\mathbf{M} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}^{n+1}) & \text{if} \quad \beta_{i} = 0, \\
\overline{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i}^{*} - \overline{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}^{n+1} & \text{if} \quad \beta_{i} = 1,
\end{cases}$$
(62)

We conclude with the algorithms for the interpolation from the main grid to the dual mesh

$$\tilde{\boldsymbol{\rho}}_{i+\frac{1}{2}} = \begin{cases}
\mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i} + \mathbf{M}_{\mathbf{R}}^{\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i+1}) & \text{if} & \beta_{i+\frac{1}{2}} = \beta_{i} = \beta_{i+1} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i} + \mathbf{M}_{\mathbf{R}}^{\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1}) & \text{if} & \beta_{i+\frac{1}{2}} = \beta_{i} = \beta_{i+1} = 1, \\
(\mathbf{M}_{\mathbf{L}}^{\mathrm{FV}} \cdot \mathcal{P} \cdot \hat{\boldsymbol{\rho}}_{i} + \mathbf{M}_{\mathbf{R}}^{\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1}) & \text{if} & \beta_{i+\frac{1}{2}} = \beta_{i+1} = 1, \beta_{i} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i} + \mathbf{M}_{\mathbf{R}}^{\mathrm{FV}} \cdot \mathcal{P} \cdot \hat{\boldsymbol{\rho}}_{i+1}) & \text{if} & \beta_{i+\frac{1}{2}} = \beta_{i} = 1, \beta_{i+1} = 0
\end{cases} (64)$$

and with the one for the opposite projection from the staggered grid to main one

$$\tilde{U}_{i} = \begin{cases}
\mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\text{DG}} \cdot \hat{U}_{i-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\text{DG}} \cdot \hat{U}_{i+\frac{1}{2}}) & \text{if} & \beta_{i} = \beta_{i-\frac{1}{2}} = \beta_{i+\frac{1}{2}} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\text{FV}} \cdot \overline{U}_{i-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\text{FV}} \cdot \overline{U}_{i+\frac{1}{2}}) & \text{if} & \beta_{i} = \beta_{i-\frac{1}{2}} = \beta_{i+\frac{1}{2}} = 1, \\
\mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\text{DG}} \cdot \hat{U}_{i-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\text{DG}} \cdot \mathcal{W}_{\mathcal{L}} \cdot \overline{U}_{i+\frac{1}{2}}) & \text{if} & \beta_{i} = \beta_{i-\frac{1}{2}} = 0, \beta_{i+\frac{1}{2}} = 1, \\
\mathbf{M}^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\text{DG}} \cdot \mathcal{W}_{\mathcal{R}} \cdot \overline{U}_{i-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\text{DG}} \cdot \hat{U}_{i+\frac{1}{2}}) & \text{if} & \beta_{i} = \beta_{i+\frac{1}{2}} = 0, \beta_{i-\frac{1}{2}} = 1,
\end{cases}$$
(65)

Finally note that, due to the choice of  $N_s=2P+1$  subcell averages, the CFL condition for the explicit part of the high order DG scheme and the CFL condition for the explicit part of the subgrid finite volume method yield the same maximum admissible time step size  $\Delta t$  for both semi-implicit schemes.

#### 3. Staggered semi-implicit DG schemes with sub-cell finite volume limiting for the Euler equations in 2D

In this section we now present the staggered semi-implicit DG scheme with *a posteriori* subcell finite volume limiter for the two-dimensional compressible Euler equations. We start by introducing the unlimited semi-implicit DG scheme in Section 3.2.2; then in Section 3.3 we briefly focus the attention on the subcell formulation for the semi-implicit FV method in 2D. Later on, we discuss the *a posteriori* subcell limiter for the 2D case, which is again based on the MOOD paradigm (see subsection 3.4) and which combines the two methods previously mentioned (see subsection 3.5).

#### 3.1. Governing equations of the 2D model

The two dimensional Euler equations are a nonlinear system of four PDEs

$$\frac{\partial \rho}{\partial t} + \frac{\partial \rho u}{\partial x} + \frac{\partial \rho v}{\partial y} = 0,$$

$$\frac{\partial \rho u}{\partial t} + \frac{\partial \rho u u}{\partial x} + \frac{\partial \rho u v}{\partial y} + \frac{\partial p}{\partial x} = 0,$$

$$\frac{\partial \rho v}{\partial t} + \frac{\partial \rho u v}{\partial x} + \frac{\partial \rho v v}{\partial y} + \frac{\partial p}{\partial y} = 0,$$

$$\frac{\partial \rho E}{\partial t} + \frac{\partial}{\partial x} u(\rho E + p) + \frac{\partial}{\partial y} v(\rho E + p) = 0,$$
(66)

where  ${\bf Q}$  is the vector of the conserved variables  ${\bf Q}=[\rho,\rho u,\rho v,\rho E]^T$ . The total energy is split into two contributions  $\rho E=\rho e+\rho k$ , where  $k=\frac{1}{2}(u^2+v^2)$  is the specific kinetic energy. In order to close the system we use again the ideal gas equation of state; hence, similar to the 1D case the internal energy is expressed as  $e=\frac{p}{(\gamma-1)\rho}$ . The system of PDEs (66) is written in the quasi-linear form

$$\frac{\partial \mathbf{Q}}{\partial t} + \mathbf{A}(\mathbf{Q}) \frac{\partial \mathbf{Q}}{\partial x} + \mathbf{B}(\mathbf{Q}) \frac{\partial \mathbf{Q}}{\partial y} = 0$$
 (67)

where  $\mathbf{A}(\mathbf{Q})$  and  $\mathbf{B}(\mathbf{Q})$  are the two Jacobian matrices expressed as  $\mathbf{A}(\mathbf{Q}) = \partial \mathbf{F}(\mathbf{Q})/\partial \mathbf{Q}$  and  $\mathbf{B}(\mathbf{Q}) = \partial \mathbf{G}(\mathbf{Q})/\partial \mathbf{Q}$  where  $\mathbf{F} = [\rho u, \rho u^2 + p, \rho u v, u(E+p)]^T$  and  $\mathbf{G} = [\rho v, \rho u v, \rho v^2 + p, v(E+p)]^T$  are the physical fluxes in x and y direction, respectively. Also in this case it is possible to verify that eq. (66) is a hyperbolic system of PDEs, see [69]. In the two dimensional case the flux splitting of Toro and Vázquez yields the following PDE for the total energy equation.

$$\frac{\partial}{\partial t}(\rho e + \rho k) + \frac{\partial}{\partial x}(\rho k u + h \rho u) + \frac{\partial}{\partial y}(\rho k v + h \rho v) = 0.$$
 (68)

## 3.2. Unlimited staggered semi-implicit DG scheme for the 2D Euler equations

First we introduce the main grid for the computational domain  $\Omega_{xy} = [x_L, x_R] \times [y_B, y_T]$  which is divided into  $N_x \times N_y$  control volumes denoted as  $T_{i,j} = [x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}] \times [y_{j-\frac{1}{2}}, j_{j+\frac{1}{2}}]$ , see Fig. 7a. In addition, the lengths of the elements  $T_{i,j}$  are  $\Delta x = \frac{x_R - x_L}{N_x}$  and  $\Delta y = \frac{y_T - y_B}{N_y}$  for the x and y directions, respectively. Later on we consider two staggered grids and their cells are denoted by  $T_{i+\frac{1}{2},j} = [x_i, x_{i+1}] \times [y_{j-\frac{1}{2}}, j_{j+\frac{1}{2}}]$  and  $T_{i,j+\frac{1}{2}} = [x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}] \times [y_j, j_{j+1}]$ , see Fig. 7b. Now we consider the same polynomial basis functions  $\phi$  and  $\psi$  of degree P used in the 1D semi-implicit DG scheme. Hence, a quantity defined on the main grid, for example the pressure, is represented within element the  $T_{i,j}$  as follows

$$p_{i,j}(x,y,t) = \sum_{l}^{P+1} \sum_{m}^{P+1} \phi_l(x) \phi_m(y) \hat{p}_{i,j,l,m}(t) = \phi(x) \phi(y) \cdot \hat{p}_{i,j}(t),$$
(69)

![](_page_17_Figure_0.jpeg)

![](_page_17_Figure_1.jpeg)

(a) Main grid for  $\eta$ 

(b) Staggered dual meshes for U and V

Figure 7: Computational grids for the 2D semi-implicit staggered DG scheme.

then, according to the philosophy of staggered methods the velocities are located on the dual meshes and they are defined as

$$u_{i+\frac{1}{2},j}(x,y,t) = \sum_{l}^{P+1} \sum_{m}^{P+1} \phi_{l}(x)\psi_{m}(y)\hat{u}_{i+\frac{1}{2},j,l,m}(t) = \phi(x)\psi(y) \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}(t)$$
(70)

$$v_{i,j+\frac{1}{2}}(x,y,t) = \sum_{l}^{P+1} \sum_{m}^{P+1} \psi_{l}(x)\phi_{m}(y)\hat{v}_{i,j+\frac{1}{2},l,m}(t) = \psi(x)\phi(y) \cdot \hat{v}_{i,j+\frac{1}{2}}(t).$$
 (71)

#### 3.2.1. Convective terms

Similar to the 1D staggered semi-implicit DG method, also in the 2D case the nonlinear convective terms are updated on the main grid and integrated using the RKDG method where the spatial operator  $\mathbf{L}_h\left(\widehat{\mathbf{Q}}\right)$  reads

$$\mathbf{L}_{h}\left(\widehat{\mathbf{Q}}\right)|_{T_{i}} = \mathbf{L}_{h}^{x}\left(\widehat{\mathbf{Q}}\right)|_{T_{i}} + \mathbf{L}_{h}^{y}\left(\widehat{\mathbf{Q}}\right)|_{T_{i}} \tag{72}$$

with

$$\mathbf{L}_{h}^{x}\left(\widehat{\mathbf{Q}}\right)|_{T_{i}} = -\frac{\Delta t}{\Delta x \Delta y} \left(\mathbf{M}^{xy}\right)^{-1} \left(\int_{\partial T_{i,j}} \phi \mathbf{F}_{\mathbf{u}} \cdot \mathbf{n} dS - \int_{T_{i,j}} \nabla \phi \cdot \mathbf{F}_{\mathbf{u}} dx dy\right),\tag{73}$$

$$\mathbf{L}_{h}^{y}\left(\widehat{\mathbf{Q}}\right)|_{T_{i}} = -\frac{\Delta t}{\Delta x \Delta y} \left(\mathbf{M}^{\mathbf{x}\mathbf{y}}\right)^{-1} \left(\int_{\partial T_{i,j}} \phi \mathbf{F}_{\mathbf{v}} \cdot \mathbf{n} dS - \int_{T_{i,j}} \nabla \phi \cdot \mathbf{F}_{\mathbf{v}} dx dy\right),\tag{74}$$

where  $F_{\boldsymbol{u}} = [\rho u, \rho uu, \rho uv, \rho ku]$  and  $F_{\boldsymbol{v}} = [\rho v, \rho uv, \rho vv, \rho kv]$  are the physical convective fluxes and for the computation of the mass matrix  $\mathbf{M}^{\mathbf{x}\mathbf{y}}$  we will give more explanations in the next subsection. In addition, the numerical fluxes are computed using the Rusanov method

$$F_{u} \cdot n = \frac{1}{2} (F_{u}^{+} + F_{u}^{-}) \cdot n - \frac{1}{2} s_{q} (q^{+} - q^{-}), \text{ with } s_{q} = 2 \max(|u^{+}|, |u^{-}|).$$
 (75)

#### 3.2.2. Derivation of the 2D semi-implicit staggered DG scheme

The integration of the energy equation is carried out by multiplying eq. (68) with the test functions  $\phi\phi$  and integrating over a control volume  $T_{i,j}$  of the main grid

$$\int_{x_{i-\frac{1}{2},j}}^{x_{i+\frac{1}{2},j}} \int_{y_{i,j-\frac{1}{2}}}^{y_{i,j+\frac{1}{2}}} \phi \phi \left( \frac{\partial}{\partial t} (\rho e + \rho k) + \frac{\partial}{\partial y} (\rho k v + h \rho v) \right) dx dy = 0.$$
 (76)

Similarly, the momentum equations in x and y direction are discretized on the dual meshes as follows

$$\int_{x_{i,j}}^{x_{i+1,j}} \int_{y_{i,j-\frac{1}{2}}}^{y_{i,j+\frac{1}{2}}} \psi \phi \left( \frac{\partial \rho u}{\partial t} + \frac{\partial \rho u^2}{\partial x} + \frac{\partial \rho u v}{\partial y} \right) dx dy = - \int_{x_{i,j}}^{x_{i+1,j}} \int_{y_{i,i-\frac{1}{2}}}^{y_{i,j+\frac{1}{2}}} \psi \phi \frac{\partial p}{\partial x} dx dy.$$
 (77)

$$\sum_{i_{1}=\frac{1}{3},j}^{x_{i+\frac{1}{2},j}} \int_{y_{i,j}}^{y_{i,j+1}} \phi \psi \left( \frac{\partial \rho v}{\partial t} + \frac{\partial \rho u v}{\partial x} + \frac{\partial \rho v^{2}}{\partial y} \right) dx dy = - \int_{x_{i-\frac{1}{3},j}}^{x_{i+\frac{1}{2},j}} \int_{y_{i,j}}^{y_{i,j+1}} \phi \psi \frac{\partial p}{\partial y} dx dy.$$
 (78)

We proceed with eq. (76) by splitting the integration of the time derivative in three contributions

$$\begin{pmatrix}
x_{i+\frac{1}{2}} \\
\int_{x_{i-\frac{1}{2}}}^{y_{j+\frac{1}{2}}} \phi \phi dx
\end{pmatrix} \cdot \begin{pmatrix}
y_{j+\frac{1}{2}} \\
\int_{y_{j-\frac{1}{2}}}^{y_{j+\frac{1}{2}}} \phi \phi dy
\end{pmatrix} \cdot \begin{pmatrix}
\widehat{\rho} \widehat{e}_{i,j}^{n+1} + \widehat{\rho} \widehat{k}_{i,j}^{n+1} - \widehat{\rho} \widehat{E}_{i,j}^{*} \\
\Delta t
\end{pmatrix}$$
(79)

where  $\widehat{\rho E}_{i,j}^{n+1} = \widehat{\rho e}_{i,j}^{n+1} + \widehat{\rho k}_{i,j}^{n+1}$  is the total energy density at the new time and the term  $\widehat{\rho E}_{i,j}^*$  is the result of the explicit discretization of the convection of the kinetic energy. Now we consider the integration of the derivative in x direction  $\frac{\partial h \rho u}{\partial x}$ . In this case it is possible to isolate the integral of the basis functions that depends on y while for the remaining part the procedure is carried out similarly to eq. (18); hence integration by parts and the adaptive  $\theta$ -method in order to get second order of accuracy in time yield

$$\left(+\phi(x_{i+\frac{1}{2}}^{-})\psi(x_{i+\frac{1}{2}})\psi(x_{i+\frac{1}{2}})\cdot\hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}}\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}}-\phi(x_{i-\frac{1}{2}}^{+})\psi(x_{i-\frac{1}{2}})\psi(x_{i-\frac{1}{2}})\cdot\hat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}}\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{u}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}}-\left(\int_{x_{i}}^{x_{i+\frac{1}{2}}}\frac{\partial\phi}{\partial x}\psi\psi dx\right)\cdot\hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}}\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}}\right)\left(\int_{y_{j-\frac{1}{2}}}^{y_{j+\frac{1}{2}}}\phi\phi dy\right).$$
(80)

The integration of the the derivative  $\frac{\partial h \rho v}{\partial y}$  can be done in the same way. Later on, we continue the integration of eq. (77). The time derivative is discretized introducing the nonlinear convective term  $\widehat{\rho u}_{i+\frac{1}{2},j}^*$  and separating the integrals in x and y direction. Then, the contribution of the pressure gradient is approximated by splitting it into a smooth contribution and a jump term across the interface of the primary grid, which is located in the interior of the dual mesh. Furthermore, we use a spatially adaptive  $\theta$ -method. The same procedure can be properly repeated for eq. (78).

$$\begin{pmatrix}
\int_{x_{i}}^{x_{i+1}} \boldsymbol{\psi} \boldsymbol{\psi} dx \end{pmatrix} \begin{pmatrix}
\int_{y_{i-\frac{1}{2}}}^{y_{i+\frac{1}{2}}} \boldsymbol{\phi} \boldsymbol{\phi} dy \\
\int_{y_{i-\frac{1}{2}}}^{y_{i+\frac{1}{2}}} \boldsymbol{\phi} \boldsymbol{\phi} dy \end{pmatrix} \cdot \begin{pmatrix}
\widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+1} - \widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{*} \\
\Delta t
\end{pmatrix} = -\begin{pmatrix}
\int_{y_{i+\frac{1}{2}}}^{y_{i+\frac{1}{2}}} \boldsymbol{\phi} \boldsymbol{\phi} dy \\
\int_{y_{i-\frac{1}{2}}}^{y_{i+\frac{1}{2}}} \boldsymbol{\phi} \boldsymbol{\phi} dy \end{pmatrix} \cdot \begin{pmatrix}
\widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+1} - \widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{*} \\
\int_{y_{i+\frac{1}{2}}}^{y_{i+\frac{1}{2}}} \boldsymbol{\phi} \boldsymbol{\phi} dx \end{pmatrix} \cdot \hat{\boldsymbol{p}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \begin{pmatrix}
\int_{x_{i+1}}^{x_{i+\frac{1}{2}}} \boldsymbol{\phi} \frac{\boldsymbol{\phi}}{\boldsymbol{\phi}} dx \\
\widehat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} \end{pmatrix} - \begin{pmatrix}
\int_{x_{i+1}}^{x_{i+\frac{1}{2}}} \boldsymbol{\phi} \frac{\boldsymbol{\phi}}{\boldsymbol{\phi}} dx \\
\widehat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{\rho}} \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \hat{\boldsymbol{v}}_{i+\frac{1}{2},j}^{n+\theta_{i+$$

Finally, the integration of equations (76), (77) and (78) is completed and written in the compact matrix-vector notation as follows

$$\mathbf{M}^{\mathbf{x}\mathbf{y}} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{e}}_{i,j}^{n+1} + \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{k}}_{i,j}^{n+1} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{E}}_{i,j}^{*})$$

$$+ \frac{\Delta t}{\Delta x} \mathbf{M}^{\mathbf{y}} \cdot (\mathbf{R}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}})$$

$$+ \frac{\Delta t}{\Delta y} \mathbf{M}^{\mathbf{x}} \cdot (\mathbf{R}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{v}}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}}) = 0$$

$$(82)$$

$$\mathbf{M}^{\mathbf{x}\mathbf{y}} \cdot (\widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+1} - \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^*) + \frac{\Delta t}{\Delta x} \mathbf{M}^{\mathbf{y}} \cdot (\mathbf{R}_{\mathbf{p}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}_{\mathbf{p}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i,j}^{n+\theta_{i+\frac{1}{2},j}}) = 0$$
(83)

$$\mathbf{M}^{\mathbf{x}\mathbf{y}} \cdot (\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^{n+1} - \widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^*) + \frac{\Delta t}{\Delta u} \mathbf{M}^{\mathbf{x}} \cdot (\mathbf{R}_{\mathbf{p}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i,j+1}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}_{\mathbf{p}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{p}}_{i,j}^{n+\theta_{i,j+\frac{1}{2}}}) = 0$$
(84)

The matrices used in eq. (82), (83), (84), (73), (74) are obtained using the following tensor products and assuming the Einstein convection of summation over repeated indexes

$$\mathbf{Z}^{\mathbf{x}} = Z_{mn} I_{m'n'} X_{nn'}, \quad \mathbf{Z}^{\mathbf{y}} = I_{mn} Z_{m'n'} X_{nn'}, \quad \mathbf{Z}^{\mathbf{x}\mathbf{y}} = \mathbf{Z}^{\mathbf{x}} \cdot \mathbf{Z}^{\mathbf{y}}, \tag{85}$$

where  $\widehat{X}$  is a generic vector of degrees of freedom,  $\mathbf{Z}$  is a square matrix and  $\mathbf{I}$  is the identity matrix. Consequently, these tensor products are carried out involving the same tensors introduced in the 1D case for the unlimited semi-implicit staggered DG scheme. See also [33] for a similar notation employed in the context of high order semi-implicit staggered DG methods for the incompressible Navier-Stokes equations.

Then, from equations (83) and (84) one can get an expression of the degrees of freedom of  $\widehat{\rho u}_{i,j\pm\frac{1}{2}}^{n+1}$  and  $\widehat{\rho v}_{i,j\pm\frac{1}{2}}^{n+1}$  and cast them into eq. (82). Later on, similar to the 1D case, an iterative *Picard* procedure is introduced in order to avoid the solution of a strongly nonlinear system of equations. Consequently, the linear system reads

$$\mathcal{L}_{i,j}^{x,\text{DG}} \cdot \hat{\boldsymbol{p}}_{i-1,j}^{n+1,r+1} + \mathcal{L}_{i,j}^{y,\text{DG}} \cdot \hat{\boldsymbol{p}}_{i,j-1}^{n+1,r+1} + \mathcal{C}_{i,j}^{\text{DG}} \cdot \hat{\boldsymbol{p}}_{i,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{x,\text{DG}} \cdot \hat{\boldsymbol{p}}_{i+1,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{y,\text{DG}} \cdot \hat{\boldsymbol{p}}_{i,j+1}^{n+1,r+1} = \hat{\boldsymbol{b}}_{i,j}^{r}, \quad (86)$$

and is solved using modern iterative Krylov subspace methods based on a matrix-free implementation. Numerical experiments have shown that system 86 can be solved using the conjugate gradient method, but we stress that we do not have a rigorous proof yet that the system is symmetric and positive definite. A deeper analysis will be carried out in further works.

Once the pressure in known, in the Picard loop the degrees of freedom of the momentum in x and y direction are readily updated as follows

$$\widehat{\rho u}_{i+\frac{1}{2},j}^{n+1} = \widehat{Gu}_{i+\frac{1}{2},j}^{n+1} - \theta_{i+\frac{1}{2},j} \frac{\Delta t}{\Delta x} (\mathbf{M}^{\mathbf{x}-1}) \cdot (\mathbf{R}_{\mathbf{p}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{p}_{i+1,j}^{n+1,r+1} - \mathbf{L}_{\mathbf{p}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{p}_{i,j}^{n+1,r+1})$$
(87)

$$\widehat{\rho v}_{i,j+\frac{1}{2}}^{n+1} = \widehat{Gv}_{i,j+\frac{1}{2}}^{n+1} - \theta_{i,j+\frac{1}{2}} \frac{\Delta t}{\Delta y} (\mathbf{M}^{\mathbf{y}-1}) \cdot (\mathbf{R}_{\mathbf{p}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{p}_{i,j+1}^{n+1,r+1} - \mathbf{L}_{\mathbf{p}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{p}_{i,j}^{n+1,r+1})$$
(88)

and the two components of the velocity vector are computed component-wise as

$$\widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+1} = \frac{\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+1}}{\widehat{\boldsymbol{\rho}}_{i+\frac{1}{2},j}^{n+1}}, \qquad \widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^{n+1} = \frac{\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^{n+1}}{\widehat{\boldsymbol{\rho}}_{i,j+\frac{1}{2}}^{n+1}}.$$
(89)

In the previous equation the density is extrapolated on the staggered meshes using the following  $L_2$ -based projection

$$\hat{\boldsymbol{\rho}}_{i+\frac{1}{2},j} = (\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i+1,j})$$
(90)

$$\hat{\boldsymbol{\rho}}_{i,j+\frac{1}{2}} = (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j+1}). \tag{91}$$

Similarly, we interpolate the velocities on the main grid from the dual control volumes

$$\hat{\boldsymbol{u}}_{i,j} = (\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{u}}_{i-\frac{1}{2},j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j})$$
(92)

$$\hat{\boldsymbol{v}}_{i,j} = (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{v}}_{i,j-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}})$$

$$(93)$$

and then it is possible to compute the kinetic energy as

$$\hat{\mathbf{k}}_{i,j} = \frac{1}{2} (\hat{\mathbf{u}}_{i,j}^2 + \hat{\mathbf{v}}_{i,j}^2). \tag{94}$$

Finally, once the Picard loop is terminated, it is possible to update the degrees of freedom of the total energy density by the following expression

$$\widehat{\rho}\widehat{E}_{i,j}^{n+1} = \widehat{\rho}\widehat{E}_{i,j}^{*}$$

$$-\frac{\Delta t}{\Delta x}\mathbf{M}^{\mathbf{x}-1} \cdot (\mathbf{R}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} \widehat{\rho}\widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}} \widehat{\rho}\widehat{\boldsymbol{u}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}})$$

$$-\frac{\Delta t}{\Delta y}\mathbf{M}^{\mathbf{y}-1} \cdot (\mathbf{R}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} \widehat{\rho}\widehat{\boldsymbol{v}}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}} \widehat{\rho}\widehat{\boldsymbol{v}}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}}),$$
(95)

which is easily obtained modifying properly eq. (82). We conclude discussing the stability condition of the method. The maximum time step is given by the classical CFL condition of the DG method based on the maximum velocities  $|u_{\rm max}|$  and  $|v_{\rm max}|$ 

$$\Delta t = \text{CFL} \left( \frac{|u_{\text{max}}|}{\Delta x} + \frac{|v_{\text{max}}|}{\Delta y_s} \right)^{-1}, \quad \text{with} \quad \text{CFL} < \frac{1}{2P+1}. \tag{96}$$

This is possible due to the implicit treatment of the pressure system and, as we will see later, this makes the staggered semi-implicit 2D DG scheme a good candidate for solving the 2D Euler equations in the low Mach number regime.

#### 3.3. A sub-cell formulation for the semi-implicit finite volume method for the 2D Euler equations

Now the two dimensional case of the semi-implicit sub-cell finite volume scheme is presented. This method represents the two dimensional extension of the method proposed in sub-section 2.3. Similarly to the previous sub-section the control volumes of the the grids are  $T_{i,j}$ ,  $T_{i+\frac{1}{2},j}$  and  $T_{i,j+\frac{1}{2}}$  see Figs. 8 and 9. The sub-grid is composed of  $(2P+1)^2$  finite volume sub-elements of size  $\Delta x_s = \Delta x/(2P+1)$  and  $\Delta y_s = \Delta y/(2P+1)$ , where P is a positive integer, see Fig. 8b.

Here the quantities assigned to the main grid and to the the dual meshes are chosen in the same way discussed in the previous subsection 3.2.2. Then, the explicit convective terms are updated at the subcell level using a first order finite volume scheme on collocated grids with a simple Rusanov-type method for the numerical fluxes

$$\overline{\boldsymbol{Q}}_{i,j,p,q}^* = \overline{\boldsymbol{Q}}_{i,j,p,q}^n - \frac{\Delta t}{\Delta x_s} (\mathbf{F}_{i,j,p+\frac{1}{2},q}^c - \mathbf{F}_{i,j,p-\frac{1}{2},q}^c) - \frac{\Delta t}{\Delta y_s} (\mathbf{G}_{i,j,p,q+\frac{1}{2}}^c - \mathbf{G}_{i,j,p,q-\frac{1}{2}}^c). \tag{97}$$

![](_page_21_Figure_0.jpeg)

| pi,j,h,k     | pi,j,h,k     | pi,j,h,k     |
|--------------|--------------|--------------|
| h = 1, k = 3 | h = 2, k = 3 | h = 3, k = 3 |
| pi,j,h,k     | pi,j,h,k     | pi,j,h,k     |
| h = 1, k = 2 | h = 2, k = 2 | h = 3, k = 2 |
| pi,j,h,k     | pi,j,h,k     | pi,j,h,k     |
| h = 1, k = 1 | h = 2, k = 1 | h = 3, k = 1 |

(a) Stencil

(b) Ordering of the subcell degrees of freedom within one cell

Figure 8: Main grid and sub-cells for the 2D semi-implicit sub-cell finite volume scheme.

![](_page_21_Figure_5.jpeg)

Figure 9: Staggered grids for the 2D semi-implicit sub-cell finite volume scheme.

Later on, the semi-implicit discretization of the energy equation is carried out on the main grid and it reads

$$(\overline{\rho}\overline{e}_{i,j}^{n+1} + \overline{\rho}\overline{k}^{n+1} - \overline{\rho}\overline{E}_{i,j}^{*})$$

$$+ \frac{\Delta t}{\Delta x} (\mathbf{R}^{\mathbf{x},FV} \cdot \overline{h\rho u}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}^{\mathbf{x},FV} \cdot \overline{h\rho u}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}})$$

$$+ \frac{\Delta t}{\Delta y} (\mathbf{R}^{\mathbf{y},FV} \cdot \overline{h\rho v}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}^{\mathbf{y},FV} \cdot \overline{h\rho v}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}}) = 0.$$

$$(98)$$

while the momentum equations are approximated on the control volumes of the dual grids as follows

$$\left(\overline{\rho}\overline{u}_{i+\frac{1}{2},j}^{n+1} - \overline{\rho}\overline{u}_{i+\frac{1}{2},j}^{*}\right) + \frac{\Delta t}{\Delta x}\left(\mathbf{R}^{\mathbf{x},FV} \cdot \overline{p}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}^{\mathbf{x},FV} \cdot \overline{p}_{i,j}^{n+\theta_{i+\frac{1}{2},j}}\right) = 0$$

$$(99)$$

$$(\overline{\rho}\overline{v}_{i,j+\frac{1}{2}}^{n+1} - \overline{\rho}\overline{v}_{i,j+\frac{1}{2}}^*) + \frac{\Delta t}{\Delta u}(\mathbf{R}^{\mathbf{y},FV} \cdot \overline{p}_{i,j+1}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}^{\mathbf{y},FV} \cdot \overline{p}_{i,j}^{n+\theta_{i,j+\frac{1}{2}}}) = 0$$

$$(100)$$

The tensor products in equations (98), (99), (100) are carried out using the same approach explained in the previous subsection. So matrix multiplications involve only the tensors introduced for the 1D case of the subcell finite volume method and they are reported in Appendix Appendix A.2.

Then, at every iteration of the Picard loop we solve the following linear system

$$\mathcal{L}_{i,j}^{x,\text{FV}} \cdot \overline{p}_{i-1,j}^{n+1,r+1} + \mathcal{L}_{i,j}^{y,\text{FV}} \cdot \overline{p}_{i,j-1}^{n+1,r+1} + \mathcal{C}_{i,j}^{\text{FV}} \cdot \overline{p}_{i,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{x,\text{FV}} \cdot \overline{p}_{i+1,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{y,\text{FV}} \cdot \overline{p}_{i,j+1}^{n+1,r+1} = \overline{b}_{i,j}^{r}, \quad (101)$$

which is obtained coupling the energy equation (98) with the momentum equations (99) and (100). System in eq. (101) is symmetric and positive definite since that the method presented in this subsection is an extension at the subgrid level of the FV method proposed [26]. Hence a conjugate gradient method gives efficiently the pressure  $\overline{p}^{n+1,r+1}$  and then the momentum is readily updated using equation (99) and 100. Then, one can compute the kinetic energy where the velocities are extrapolated from the dual grid using the following projections

$$\overline{\boldsymbol{u}}_{i,j} = (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathsf{FV}} \cdot \overline{\boldsymbol{u}}_{i-\frac{1}{2},j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathsf{FV}} \cdot \overline{\boldsymbol{u}}_{i+\frac{1}{2},j}) \tag{102}$$

$$\overline{\boldsymbol{v}}_{i,j} = (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathsf{FV}} \cdot \overline{\boldsymbol{v}}_{i,j-\frac{1}{2}} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathsf{FV}} \cdot \overline{\boldsymbol{v}}_{i,j+\frac{1}{2}}). \tag{103}$$

where  $\overline{u}_{i+\frac{1}{2},j}=\overline{\rho}\overline{u}_{i+\frac{1}{2},j}/\overline{\rho}_{i+\frac{1}{2},j}$  and  $\overline{v}_{i,j+\frac{1}{2}}=\overline{\rho}\overline{v}_{i,j+\frac{1}{2}}/\overline{\rho}_{i,j+\frac{1}{2}}$  are computed component-wise, respectively. Similarly, the densities on the dual grids  $\overline{\rho}_{i+\frac{1}{2},j}$  and  $\overline{\rho}_{i,j+\frac{1}{2}}$  are computed from  $\overline{p}_{i,j}$  using these subcell finite volume averages

$$\overline{\boldsymbol{\rho}}_{i+\frac{1}{2},j} = (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1,j})$$
(104)

$$\overline{\boldsymbol{\rho}}_{i,j+\frac{1}{2}} = (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j+1}). \tag{105}$$

Finally, the update for the total energy reads

$$\overline{\rho}\overline{E}_{i,j}^{n+1} - \overline{\rho}\overline{E}_{i,j}^{*} = -\frac{\Delta t}{\Delta x} (\mathbf{R}^{\mathbf{x},FV} \cdot \overline{h\rho u}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}^{\mathbf{x},FV} \cdot \overline{h\rho u}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}}) 
- \frac{\Delta t}{\Delta y} (\mathbf{R}^{\mathbf{y},FV} \cdot \overline{h\rho v}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}^{\mathbf{y},FV} \cdot \overline{h\rho v}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}})$$
(106)

In this method,  $\Delta t$  is computed according to the following expression for the CFL condition

$$\Delta t = \text{CFL} \left( \frac{|u_{\text{max}}|}{\Delta x_s} + \frac{|v_{\text{max}}|}{\Delta y_s} \right)^{-1}, \quad \text{with} \quad \text{CFL} < 1.$$
 (107)

#### 3.4. MOOD algorithm and detection criteria - 2D case

The MOOD algorithm for the two-dimensional case is based on the same approach of Section 2.4. Consequently, the two dimensional extension of the considerations made in Section 2.4 is straightforward. We introduce the projection and reconstruction matrices  $\mathcal{P}^{xy}$  and  $\mathcal{W}^{xy}$  for a generic variable q. Hence, we have  $\overline{q}_{i,j}^n = \mathcal{P}^{xy} \cdot \hat{q}_{i,j}^n$ ,  $\hat{q}_{i,j}^n = \mathcal{W}^{xy} \cdot \overline{q}_{i,j}^n$ .

In addition, we introduce also the operators  $\mathbb{W}^x$ ,  $\mathbb{W}^y$ ,  $\mathbb{P}^x$ ,  $\mathbb{P}^y$  and, for the sake of simplicity, their role will be explained in the next subsection. The detection criteria are based on the positivity of the density and of the pressure and on the relaxed discrete maximum principle (DMP), i.e.

$$\hat{\rho}_{i,j,m}^{\circ,n+1} > 0, \quad \hat{p}_{i,j,l,k}^{\circ,n+1} > 0, \quad \text{and} \quad \min_{\forall T_{k,l} \in \mathcal{V}_{i,j}} (\hat{q}_{k,l,m}^n) - \delta \leq \hat{q}_{i,j,m}^{\circ,n+1} \leq \max_{\forall T_{k,l} \in \mathcal{V}_{i,j}} (\hat{q}_{k,l,m}^n) + \delta, \quad (108)$$

where  $\mathcal{V}_{i,j} = \{T_{i-1,j-1}, T_{i,j-1}, T_{i+1,j-1}, T_{i-1,j}, T_{i,j}, T_{i+1,j}, T_{i-1,j+1}, T_{i,j+1}, T_{i+1,j+1}\}$  is the set of Voronoi neighbours of  $T_{i,j}$  and  $\delta$  is the same relaxation parameter for the DMP as in the 1D case. The DMP is applied to the set of conserved variables  $\rho$ ,  $\rho u$ ,  $\rho v$ ,  $\rho E$  as well to the primitive variables  $\rho$ , u, v, u. The control volumes that do not pass the admissibility criteria outlined in eq. (108) are marked as a troubled cells by assigning a limiter status  $\beta_{i,j} = 1$ , while in the cells that do not need to be limited we impose  $\beta_{i,j} = 0$ . Then, similarly to the 1D case we limit properly also the cells of the staggered dual grids. For example, if the control volume  $T_{i,j}$  is flagged as limited zone with  $\beta_{i,j} = 1$ , then also the four corresponding cells on the dual grids  $T_{i+\frac{1}{2},j}, T_{i-\frac{1}{2},j}, T_{i,j+\frac{1}{2}}$  and  $T_{i,j-\frac{1}{2}}$  need to be limited by assigning  $\beta_{i+\frac{1}{2},j} = 1$ ,  $\beta_{i-\frac{1}{2},j} = 1$ ,  $\beta_{i,j+\frac{1}{2}} = 1$  and  $\beta_{i,j-\frac{1}{2}} = 1$ . Finally, in the first MOOD step the implicitness parameter  $\theta$  is chosen between 0.5 and 1. In general, in order to reduce the numerical viscosity of the time integration, we impose  $\theta_{DG} = 0.55$  in the entire domain. However, at the second MOOD phase  $\theta$  is changed and imposed equal to  $\theta = 1$  in the troubled control volumes, in order to avoid the generation of Gibbs oscillations, while it is kept equal to  $\theta = \theta_{DG}$  in all the unlimited cells.

#### 3.5. Sub-cell limiting of the semi-implicit DG scheme for the 2D Euler equations

Now, the total energy equation is discretized in the presence of subcell limiter, where we have consider four possible configurations. In all cases  $T_{i,j}$  is unlimited ( $\beta_{i,j} = 0$ ), but one of its edge neighbours is a troubled cell. We write the equation as follows

$$\mathbf{M}^{\mathbf{x}\mathbf{y}} \cdot (\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{e}}_{i,j}^{n+1} + \widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{k}}_{i,j}^{n+1} - \widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{E}}_{i,j}^{*}) + \frac{\Delta t}{\Delta x}\mathbf{M}^{\mathbf{y}} \cdot \Delta \widehat{\boldsymbol{h}}\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{u}}_{i,j}^{n+\theta_{i,j}} + \frac{\Delta t}{\Delta y}\mathbf{M}^{\mathbf{y}} \cdot \Delta \widehat{\boldsymbol{h}}\widehat{\boldsymbol{\rho}}\widehat{\boldsymbol{v}}_{i,j}^{n+\theta_{i,j}} = 0.$$
(109)

We distinguish two cases for  $\Delta \widehat{h\rho u}_{i,j}^{n+\theta_{i,j}}$ :

$$\Delta \widehat{\boldsymbol{h}\rho u}_{i,j}^{n+\theta_{i,j}} = \begin{cases} \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y} (\overline{\boldsymbol{h}\rho u}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}}) - \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \widehat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}} & \text{if } \beta_{i+\frac{1}{2},j} = 1, \beta_{i-\frac{1}{2},j} = 0, \\ \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \widehat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} \widehat{\boldsymbol{\rho}} \widehat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y} (\overline{\boldsymbol{h}\rho u}_{i-\frac{1}{2},j}^{n+\theta_{i-\frac{1}{2},j}}) & \text{if } \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1. \end{cases}$$

$$(110)$$

The first corresponds to the case when the cell  $T_{i+1,j}$  is troubled and consequently  $\beta_{i+1,j} = \beta_{i+\frac{1}{2},j} = 1$ ; on the contrary the other situation is when the control volume  $T_{i-1,j}$  is a troubled cell. Similarly, the two cases for  $\Delta \widehat{h\rho v}_{i,j}^{n+\theta_{i,j}}$ 

$$\Delta \widehat{\boldsymbol{h}\rho v}_{i,j}^{n+\theta_{i,j}} = \begin{cases} \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{Lim}} \cdot \mathbb{W}^{x} (\overline{\boldsymbol{h}\rho v}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}}) - \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}} \widehat{\rho v}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}} & \text{if} \quad \beta_{i,j+\frac{1}{2}} = 1, \beta_{i,j-\frac{1}{2}} = 0, \\ \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \widehat{\boldsymbol{h}}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} \widehat{\rho v}_{i,j+\frac{1}{2}}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\text{Lim}} \cdot \mathbb{W}^{x} (\overline{\boldsymbol{h}\rho v}_{i,j-\frac{1}{2}}^{n+\theta_{i,j-\frac{1}{2}}}) & \text{if} \quad \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j-\frac{1}{2}} = 1. \end{cases}$$

$$(111)$$

In equations (110) and (111) we used the two operators  $\mathbb{W}^x$  and  $\mathbb{W}^y$  that are used in order to have consistent tensor products. Looking at Fig. 10 the operator  $\mathbb{W}^y$  works on the finite volume subcell averages and does a reconstruction of the DG polynomial only y direction producing the red symbols. The operator  $\mathbb{W}^y$  does a similar reconstruction in x direction. We remind that when all the four control volumes  $T_{i\pm\frac{1}{2},j}$  and  $T_{i,j\pm\frac{1}{2}}$  are not marked the discrete total

![](_page_24_Picture_0.jpeg)

![](_page_24_Picture_1.jpeg)

Figure 10: Reconstruction done by the operator  $\mathbb{W}^y$  used in the continuity equation of the limited scheme. The red crosses in  $T_{i+\frac{1}{2},j}$  are the new degrees of freedom obtained applying the  $\mathbb{W}^y$  operator the DG degrees of freedom in  $T_{i+\frac{1}{2},j}$ 

energy equation is discretized using the pure DG scheme and it corresponds to eq. (82) derived in subsection 3.2.2. On the contrary, if all the four cells are troubled control volumes, then the subcell averages of the total energy are given by eq. (98), see subsection 3.3.

Now we consider the limited case of the momentum equation in x direction. When  $T_{i+\frac{1}{2},j}$  is limited ( $\beta_{i+\frac{1}{2},j}=1$ ), we consider two cases. If  $\beta_{i+\frac{1}{2},j}=\beta_{i,j}=1$  and  $\beta_{i+1,j}=0$ , the momentum update reads

$$\overline{\rho}\overline{u}_{i+\frac{1}{2},j}^{n+1} = \overline{\rho}\overline{u}_{i+\frac{1}{2},j}^{*} - \frac{\Delta t}{\Delta x} (\mathbf{R}_{\mathbf{u}}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbb{P}^{y}(\hat{\boldsymbol{p}}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}}) - \mathbf{L}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{p}}_{i,j}^{n+\theta_{i+\frac{1}{2},j}})$$

$$(112)$$

while for the other case,  $\beta_{i+\frac{1}{2},j}=\beta_{i+1,j}=1$  and  $\beta_{i,j}=0,$  we have

$$\overline{\rho}\overline{u}_{i+\frac{1}{2},j}^{n+1} = \overline{\rho}\overline{u}_{i+\frac{1}{2},j}^* - \frac{\Delta t}{\Delta x} (\mathbf{R}^{\mathbf{x},\mathbf{FV}} \cdot \overline{p}_{i+1,j}^{n+\theta_{i+\frac{1}{2},j}} - \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\mathbf{Lim}} \cdot \mathbb{P}^y(\hat{p}_{i,j}^{n+\theta_{i+\frac{1}{2},j}})). \tag{113}$$

Similarly in y direction the two possibilities of the momentum equation are

$$\overline{\rho v}_{i,j+\frac{1}{2}}^{n+1} = \overline{\rho v}_{i,j+\frac{1}{2}}^* - \frac{\Delta t}{\Delta y} (\mathbf{R}_{\mathbf{u}}^{\mathbf{y},\mathsf{Lim}} \cdot \mathbb{P}^x (\hat{\boldsymbol{p}}_{i,j+1}^{n+\theta_{i,j+\frac{1}{2}}}) - \mathbf{L}^{\mathbf{y},\mathsf{FV}} \cdot \overline{\boldsymbol{p}}_{i,j}^{n+\theta_{i,j+\frac{1}{2}}})$$
(114)

if  $\beta_{i,j+\frac{1}{2}}=\beta_{i,j}=1$  and  $\beta_{i,j+1}=0$ , and for the other case

$$\overline{\rho v}_{i,j+\frac{1}{2}}^{n+1} = \overline{\rho v}_{i,j+\frac{1}{2}}^* - \frac{\Delta t}{\Delta y} (\mathbf{R}^{\mathbf{y},\mathbf{FV}} \cdot \overline{p}_{i,j+1}^{n+\theta_{i,j+\frac{1}{2}}} - \mathbf{L}_{\mathbf{u}}^{\mathbf{y},\mathbf{Lim}} \cdot \mathbb{P}^x (\hat{p}_{i,j}^{n+\theta_{i,j+\frac{1}{2}}}))$$
(115)

![](_page_25_Picture_0.jpeg)

 $T_{i+\frac{1}{2},j}$ 

![](_page_25_Picture_2.jpeg)

Figure 11: Projection done by the operator  $\mathbb{P}^y$  used in the momentum equation of the limited scheme. The red crosses in  $T_{i,j}$  are the new degrees of freedom obtained applying the  $\mathbb{P}^y$  operator the DG degrees of freedom in  $T_{i,j}$ 

if  $\beta_{i,j+\frac{1}{2}}=\beta_{i,j+1}=1$  and  $\beta_{i,j}=0$ . Also for the limited momentum equations we introduced the operators  $\mathbb{P}^y$  and  $\mathbb{P}^x$  that are necessary in order to multiply properly the tensors and the degrees of freedom. See Figure 11 for a schematic representation of the projection done in y direction by  $\mathbb{P}^y$ . Then, the linear system for the 2D semi-implicit limited staggered DG reads

$$\mathcal{L}_{i,j}^{x,\text{Lim}} \cdot \tilde{\boldsymbol{p}}_{i-1,j}^{n+1,r+1} + \mathcal{L}_{i,j}^{y,\text{Lim}} \cdot \tilde{\boldsymbol{p}}_{i,j-1}^{n+1,r+1} + \mathcal{C}_{i,j}^{\text{Lim}} \cdot \tilde{\boldsymbol{p}}_{i,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{x,\text{Lim}} \cdot \tilde{\boldsymbol{p}}_{i+1,j}^{n+1,r+1} + \mathcal{R}_{i,j}^{y,\text{Lim}} \cdot \tilde{\boldsymbol{p}}_{i,j+1}^{n+1,r+1} = \tilde{\boldsymbol{b}}_{i,j}^{r}, \quad (116)$$

which appears to be symmetric in practice, but we are not able to provide a rigorous proof of this property yet. The blocks  $\mathcal{R}_{i,j}^{x,\text{Lim}}$ ,  $\mathcal{L}_{i,j}^{x,\text{Lim}}$ ,  $\mathcal{R}_{i,j}^{y,\text{Lim}}$ ,  $\mathcal{L}_{i,j}^{y,\text{Lim}}$  are computed as follows

$$\mathcal{R}_{i,j}^{x,\text{Lim}}, \mathcal{L}_{i,j}^{x,\text{Lim}}, \mathcal{L}_{i,j}^{y,\text{Lim}}, \mathcal{R}_{i,j}^{y,\text{Lim}} = \text{computed as follows}$$

$$\mathcal{R}_{i,j}^{x,\text{Lim}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i+\frac{1}{2},j}^{2} \begin{cases} M^{y} \cdot \mathbf{R}_{\mathbf{e}}^{x,\text{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1} \cdot (\mathbf{M}^{x})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{x,\text{DG}} & \text{if} & \beta_{i+1,j} = \beta_{i,j} = 0, \\ \mathbf{R}^{x,\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{R}^{x,\text{FV}} & \text{if} & \beta_{i+1,j} = \beta_{i,j} = 1, \\ M^{y} \cdot \mathbf{R}_{\mathbf{e}}^{x,\text{Lim}} \cdot \mathcal{D}^{x} (\mathbb{W}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1})) \cdot \mathbf{R}^{x,\text{FV}} & \text{if} & \beta_{i+1,j} = 1, \beta_{i,j} = 0, \\ \mathbf{R}^{x,\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{R}_{\mathbf{u}}^{x,\text{Lim}} & \text{if} & \beta_{i+1,j} = 0, \beta_{i,j} = 1. \end{cases}$$

$$\mathcal{R}_{i,j}^{y,\text{Lim}} = -\frac{\Delta t^{2}}{\Delta y^{2}} \theta_{i,j+\frac{1}{2}}^{2} \begin{cases} \mathbf{M}^{x} \cdot \mathbf{R}_{\mathbf{e}}^{y,\text{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1} \cdot (\mathbf{M}^{y})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{y,\text{DG}} & \text{if} & \beta_{i,j+1} = \beta_{i,j} = 0, \\ \mathbf{R}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{R}^{y,\text{FV}} & \text{if} & \beta_{i,j+1} = \beta_{i,j} = 0, \\ \mathbf{R}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{R}_{\mathbf{u}}^{y,\text{Lim}} & \text{if} & \beta_{i,j+1} = 0, \beta_{i,j} = 1. \end{cases}$$

$$\mathcal{L}_{i,j}^{x,\text{Lim}} \cdot \mathcal{D}^{x} (\mathbf{M}^{y})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{x,\text{DG}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\ \mathbf{R}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{FV}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\ \mathbf{R}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{R}^{y,\text{Lim}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\ \mathbf{R}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{FV}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\ \mathbf{L}^{x,\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{FV}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\ \mathbf{L}^{x,\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{Lim}} & \text{if} & \beta_{i-1,j} = 0, \beta_{i,j} = 1. \end{cases}$$

$$\left\{ \mathbf{M}^{y} \cdot \mathbf{L}_{\mathbf{e}}^{x,\text{Lim}} \cdot \mathcal{D}^{x} (\mathbf{M}^{y})^{-1} \cdot \mathbf{L}^{x,\text{Lim}} & \text{if} & \beta_{i-1,j} = 0, \beta_{i,j} = 0, \\ \mathbf{L}^{x,\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{FV}} (\overline{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1}) \cdot \mathbf{L}^{x,\text{F$$

$$\mathcal{R}_{i,j}^{y,\text{Lim}} = -\frac{\Delta t^{2}}{\Delta y^{2}} \theta_{i,j+\frac{1}{2}}^{2} \begin{cases}
\mathbf{M}^{\mathbf{x}} \cdot \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\mathbf{y},\text{DG}} & \text{if} & \beta_{i,j+1} = \beta_{i,j} = 0, \\
\mathbf{R}^{\mathbf{y},\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}^{n+1}_{i+1}) \cdot \mathbf{R}^{\mathbf{y},\text{FV}} & \text{if} & \beta_{i,j+1} = \beta_{i,j} = 1, \\
\mathbf{M}^{\mathbf{x}} \cdot \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{Lim}} \cdot \mathcal{D}^{y} (\mathbf{W}^{x} (\overline{\boldsymbol{h}}^{n+1}_{i,j+\frac{1}{2}})) \cdot \mathbf{R}^{\mathbf{y},\text{FV}} & \text{if} & \beta_{i,j+1} = 1, \beta_{i,j} = 0, \\
\mathbf{R}^{\mathbf{y},\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}^{n+1}_{i,j+\frac{1}{2}}) \cdot \mathbf{R}_{\mathbf{u}}^{\mathbf{y},\text{Lim}} & \text{if} & \beta_{i,j+1} = 0, \beta_{i,j} = 1.
\end{cases} (118)$$

$$\mathcal{L}_{i,j}^{x,\text{Lim}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i-\frac{1}{2},j}^{2} \begin{cases}
\mathbf{M}^{\mathbf{y}} \cdot \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{x}})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\text{DG}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 0, \\
\mathbf{L}^{\mathbf{x},\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}^{\mathbf{x},\text{FV}} & \text{if} & \beta_{i-1,j} = \beta_{i,j} = 1, \\
\mathbf{M}^{\mathbf{y}} \cdot \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{Lim}} \cdot \mathcal{D}^{x} (\mathbf{W}^{y} (\overline{\boldsymbol{h}}^{n+1})) \cdot \mathbf{L}^{\mathbf{x},\text{FV}} & \text{if} & \beta_{i-1,j} = 1, \beta_{i,j} = 0, \\
\mathbf{L}^{\mathbf{x},\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\text{Lim}} & \text{if} & \beta_{i-1,j} = 0, \beta_{i,j} = 1.
\end{cases} (119)$$

$$\mathcal{L}_{i,j}^{y,\text{Lim}} = -\frac{\Delta t^{2}}{\Delta x^{2}} \theta_{i,j-\frac{1}{2}}^{2} \begin{cases}
\mathbf{M}^{\mathbf{x}} \cdot \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{y},\text{DG}} & \text{if} & \beta_{i,j-1} = \beta_{i,j} = 0, \\
\mathbf{L}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}^{y,\text{FV}} & \text{if} & \beta_{i,j-1} = \beta_{i,j} = 1, \\
\mathbf{M}^{\mathbf{x}} \cdot \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\text{Lim}} \cdot \mathcal{D}^{y} (\mathbf{W}^{x} (\overline{\boldsymbol{h}}^{n+1})) \cdot \mathbf{L}^{y,\text{FV}} & \text{if} & \beta_{i,j-1} = 1, \beta_{i,j} = 0, \\
\mathbf{L}^{y,\text{FV}} \cdot \mathcal{D}^{y} (\overline{\boldsymbol{h}}^{n+1}) \cdot \mathbf{L}^{y,\text{Lim}} & \text{if} & \beta_{i,j-1} = 0, \beta_{i,j} = 1.
\end{cases} (120)$$

while a proper splitting for  $C_{i,j}^{\text{Lim}}$  yields

$$C_{i,j}^{\text{Lim}} = C_{i,j}^{0,\text{Lim}} + C_{i,j}^{x,\text{Lim}} + C_{i,j}^{y,\text{Lim}}, \tag{121}$$

with

$$C_{i,j}^{0,\text{Lim}} = \begin{cases} \mathbf{M}^{\mathbf{x}\mathbf{y}} & \text{if } \beta_{i,j} = 0, \\ \mathbf{I} & \text{if } \beta_{i,j} = 1. \end{cases}$$
(122)

$$C_{i,j}^{x,\text{Lim}} = \frac{\Delta t^{2}}{\Delta x^{2}} \begin{cases} \mathbf{M}^{\mathbf{y}} (\theta_{i-\frac{1}{2},j}^{2} \cdot \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{h}}_{n}^{n+1} \cdot (\mathbf{M}^{\mathbf{x}})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\text{DG}} + \theta_{i+\frac{1}{2},j}^{2} \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{h}}_{n+1}^{n+1} \cdot (\mathbf{M}^{\mathbf{x}})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\mathbf{x},\text{DG}}) \\ \text{if } \beta_{i,j} = \beta_{i-\frac{1}{2},j} = \beta_{i+\frac{1}{2},j} = 0, \\ \theta_{i-\frac{1}{2},j}^{2} \mathbf{L}^{\mathbf{x},\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{n+1}^{n+1}) \cdot \mathbf{R}^{\mathbf{x},\text{FV}} + \theta_{i+\frac{1}{2},j}^{2} \mathbf{R}^{\mathbf{x},\text{FV}} \cdot \mathcal{D}^{x} (\overline{\boldsymbol{h}}_{n+1}^{n+1}) \cdot \mathbf{L}^{\mathbf{x},\text{FV}} \\ \text{if } \beta_{i,j} = \beta_{i-\frac{1}{2},j} = \beta_{i+\frac{1}{2},j} = 1, \\ \mathbf{M}^{\mathbf{y}} (\theta_{i-\frac{1}{2},j}^{2} \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{h}}_{n+1}^{n+1} \cdot (\mathbf{M}^{\mathbf{x}})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\text{DG}} + \theta_{i+\frac{1}{2},j}^{2} \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\text{Lim}} \cdot \mathcal{D}^{x} (\mathbf{W}^{y} (\overline{\boldsymbol{h}}_{n+1}^{n+1})) \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{x},\text{Lim}} ) \\ \text{if } \beta_{i,j} = \beta_{i-\frac{1}{2},j} = 0, \beta_{i+\frac{1}{2},j} = 1, \\ \mathbf{M}^{\mathbf{y}} (\theta_{i-\frac{1}{2},j}^{2} \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{Lim}} \cdot \mathcal{D}^{x} (\mathbf{W}^{y} (\overline{\boldsymbol{h}}_{n+1}^{n+1})) \cdot \mathbf{R}_{\mathbf{u}}^{\text{Lim}} + \theta_{i+\frac{1}{2},j}^{2} \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{h}}_{n+1}^{n+1} \cdot (\mathbf{M}^{\mathbf{x}})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\mathbf{x},\text{DG}} ) \\ \text{if } \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \end{cases}$$

$$(123)$$

$$\mathcal{C}_{i,j}^{y,\text{Lim}} = \frac{\Delta t^{2}}{\Delta y^{2}} \begin{cases}
\mathbf{M}^{\mathbf{x}}(\theta_{i,j-\frac{1}{2}}^{2} \cdot \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{L}_{\mathbf{u}}^{\mathbf{y},\text{DG}} + \theta_{i,j+\frac{1}{2}}^{2} \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{R}_{\mathbf{u}}^{\mathbf{y},\text{DG}}) \\
\text{if } \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = \beta_{i,j+\frac{1}{2}} = 0, \\
\theta_{i,j-\frac{1}{2}}^{2} \mathbf{L}^{\mathbf{y},\text{FV}} \cdot \mathcal{D}^{y}(\overline{\boldsymbol{h}}^{n+1}_{n+1}) \cdot \mathbf{R}^{\mathbf{y},\text{FV}} + \theta_{i,j+\frac{1}{2}}^{2} \mathbf{R}^{\mathbf{y},\text{FV}} \cdot \mathcal{D}^{y}(\overline{\boldsymbol{h}}^{n+1}_{n+1}) \cdot \mathbf{L}^{\mathbf{y},\text{FV}} \\
\text{if } \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = \beta_{i,j+\frac{1}{2}} = 1, \\
\mathbf{M}^{\mathbf{x}}(\theta_{i,j-\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{L}^{\mathbf{y},\text{DG}} + \theta_{i,j+\frac{1}{2}}^{2} \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\text{Lim}} \cdot \mathcal{D}^{y}(\mathbf{W}^{x}(\overline{\boldsymbol{h}}^{n+1}_{n+1})) \cdot \mathbf{L}^{\mathbf{y},\text{Lim}}) \\
\text{if } \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1, \\
\mathbf{M}^{\mathbf{y}}(\theta_{i-\frac{1}{2},j}^{2} \mathbf{L}^{\mathbf{y},\text{Lim}} \cdot \mathcal{D}^{y}(\mathbf{W}^{x}(\overline{\boldsymbol{h}}^{n+1}_{n+1})) \cdot \mathbf{R}^{\text{Lim}} + \theta_{i,j+\frac{1}{2}}^{2} \mathbf{L}^{\mathbf{y},\text{DG}} \cdot \hat{\boldsymbol{h}}^{n+1}_{n+1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot \mathbf{R}^{\mathbf{y},\text{DG}}) \\
\text{if } \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j-\frac{1}{2}} = 1.
\end{cases}$$
(124)

We consider three contributions for the right hand side term

$$\tilde{\boldsymbol{b}}_{i,j}^{r} = \tilde{\boldsymbol{b}}_{i,j}^{0,n} + \tilde{\boldsymbol{b}}_{i,j}^{x,n} + \tilde{\boldsymbol{b}}_{i,j}^{y,n}$$
(125)

with

$$\widetilde{\boldsymbol{b}}_{i,j}^{0,r} = \begin{cases}
\mathbf{M}^{\mathbf{x}\mathbf{y}} \cdot (\widehat{\boldsymbol{\rho}} \widetilde{\boldsymbol{E}}_{i,j}^* - \widehat{\boldsymbol{\rho}} \widetilde{\boldsymbol{k}}^{n+1}) & \text{if} & \beta_{i,j} = 0, \\
\overline{\boldsymbol{\rho}} \widetilde{\boldsymbol{E}}_{i,j}^* - \overline{\boldsymbol{\rho}} \widetilde{\boldsymbol{k}}^{n+1} & \text{if} & \beta_{i,j} = 1,
\end{cases}$$
(126)

$$\tilde{b}_{i,j}^{x,r} = -\frac{\Delta t}{\Delta x} \begin{cases}
(\theta_{i+\frac{1}{2},j} \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n+1} \hat{\boldsymbol{G}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n+1} \hat{\boldsymbol{G}}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n} \widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}^{n} - (1-\theta_{i-\frac{1}{2},j}) \mathbf{L}_{\mathbf{e}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n} \widehat{\boldsymbol{\rho}} \hat{\boldsymbol{u}}_{i-\frac{1}{2},j}^{n}) \\
\text{if } \beta_{i,j} = \beta_{i-\frac{1}{2},j} = \beta_{i+\frac{1}{2},j} = 0, \\
(\theta_{i+\frac{1}{2},j} \mathbf{R}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{h}}_{i}^{n} \overline{\boldsymbol{G}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{h}}_{i}^{n} \overline{\boldsymbol{G}}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{h}} \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2},j}^{n} - (1-\theta_{i-\frac{1}{2},j}) \mathbf{L}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{h}} \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{EV}} \cdot \overline{\boldsymbol{h}} \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2},j}^{n} - (1-\theta_{i-\frac{1}{2},j}) \mathbf{L}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{h}} \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2},j}^{n}) - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n} \hat{\boldsymbol{\rho}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{\rho}} \boldsymbol{u}_{i+\frac{1}{2},j}^{n}) - (1-\theta_{i-\frac{1}{2},j}) \mathbf{L}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i-\frac{1}{2},j}^{n} \widehat{\boldsymbol{\rho}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{h}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{h}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{h}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2},j}) \mathbf{R}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{h}}_{i+\frac{1}{2},j}^{n} - \theta_{i-\frac{1}{2},j} \mathbf{L}^{\mathbf{x},\mathrm{Lim}} \cdot \mathbf{W}^{y} (\overline{\boldsymbol{h}} \hat{\boldsymbol{h}} \boldsymbol{u}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i+\frac{1}{2$$

$$\tilde{b}_{i,j}^{y,r} = -\frac{\Delta t}{\Delta y} \begin{cases}
(\theta_{i,j+\frac{1}{2}} \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n+1} \hat{G}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i-\frac{1}{2},j}^{n+1} \hat{G}_{i-\frac{1}{2},j}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n} - \Omega_{i,j+\frac{1}{2}}^{n} - (1-\theta_{i,j-\frac{1}{2}}) \mathbf{L}_{\mathbf{e}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j-\frac{1}{2}}^{n} \hat{\rho} \hat{\mathbf{u}}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\mathbf{h}}_{n}^{n} - \overline{\mathbf{G}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\mathbf{h}}_{n}^{n} - \overline{\mathbf{G}}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\mathbf{h}} \rho \mathbf{u}_{i,j+\frac{1}{2}}^{n} - (1-\theta_{i,j-\frac{1}{2}}) \mathbf{L}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}}_{n}^{n+1} - \overline{\mathbf{G}}_{i,j+\frac{1}{2}}^{n}) - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{n}^{n+1} \hat{\mathbf{G}}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j+\frac{1}{2}}^{n}) - (1-\theta_{i,j-\frac{1}{2}}) \mathbf{L}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j-\frac{1}{2}}^{n} \hat{\rho} \mathbf{u}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{n}^{n+1} \cdot \overline{\mathbf{G}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{n}^{n+1} \cdot \overline{\mathbf{G}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n}) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n})) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}} \mathbf{L}^{\mathbf{y},\mathrm{Lim}} \cdot \mathbb{W}^{\mathbf{y}} (\overline{\mathbf{h}} \rho \mathbf{u}_{i,j-\frac{1}{2}}^{n})) \\
+((1-\theta_{i,j+\frac{1}{2}}) \mathbf{R}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\mathbf{h}}_{i,j+\frac{1}{2}}^{n} - \theta_{i,j-\frac{1}{2}}^$$

Then, the projections from the main grid to staggered grids in x and y direction are

$$\tilde{\boldsymbol{\rho}}_{i+\frac{1}{2},j} = \begin{cases}
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i+1,j}) & \text{if} & \beta_{i+\frac{1}{2},j} = \beta_{i,j} = \beta_{i+1,j} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1,j}) & \text{if} & \beta_{i+\frac{1}{2},j} = \beta_{i,j} = \beta_{i+1,j} = 1, \\
(\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{FV}} \cdot \mathbb{P}^{xy}(\hat{\boldsymbol{\rho}}_{i,j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1,j}) & \text{if} & \beta_{i+\frac{1}{2},j} = \beta_{i+1,j} = 1, \beta_{i,j} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\mathrm{FV}} \cdot \mathbb{P}^{xy}(\hat{\boldsymbol{\rho}}_{i+1,j})) & \text{if} & \beta_{i+\frac{1}{2},j} = \beta_{i,j} = 1, \beta_{i+1,j} = 0.
\end{cases} (129)$$

$$\tilde{\boldsymbol{\rho}}_{i,j+\frac{1}{2}} = \begin{cases}
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{DG}} \cdot \hat{\boldsymbol{\rho}}_{i,j+1}) & \text{if} & \beta_{i,j+\frac{1}{2}} = \beta_{i,j} = \beta_{i,j+1} = 0, \\
 (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i+1,j}) & \text{if} & \beta_{i,j+\frac{1}{2}} = \beta_{i,j} = \beta_{i,j+1} = 1, \\
 (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{FV}} \cdot \mathbb{P}^{xy}(\hat{\boldsymbol{\rho}}_{i,j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j+1}) & \text{if} & \beta_{i,j+\frac{1}{2}} = \beta_{i,j+1} = 1, \beta_{i,j} = 0, \\
 (\mathbf{M}_{\mathbf{L}}^{\mathbf{y},\mathrm{FV}} \cdot \overline{\boldsymbol{\rho}}_{i,j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{y},\mathrm{FV}} \cdot \mathbb{P}^{xy}(\hat{\boldsymbol{\rho}}_{i,j+1})) & \text{if} & \beta_{i,j+\frac{1}{2}} = \beta_{i,j} = 1, \beta_{i,j+1} = 0.
 \end{cases}$$
(130)

Reciprocally, the projections from the staggered grids to the main mesh are

$$\tilde{\boldsymbol{u}}_{i,j} = \begin{cases}
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i-\frac{1}{2},j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i-\frac{1}{2},j} = \beta_{i+\frac{1}{2},j} = 0, \\
(\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{FV}} \cdot \overline{\boldsymbol{u}}_{i-\frac{1}{2},j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{FV}} \cdot \overline{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i-\frac{1}{2},j} = \beta_{i+\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i-\frac{1}{2},j} + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i+\frac{1}{2},j})) & \text{if} & \beta_{i,j} = \beta_{i-\frac{1}{2},j} = 0, \beta_{i+\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i-\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1}{2},j}) + \mathbf{M}_{\mathbf{R}}^{\mathbf{x},\text{DG}} \cdot \hat{\boldsymbol{u}}_{i+\frac{1}{2},j}) & \text{if} & \beta_{i,j} = \beta_{i+\frac{1}{2},j} = 0, \beta_{i+\frac{1}{2},j} = 0, \beta_{i+\frac{1}{2},j} = 1, \\
(\mathbf{M}^{\mathbf{x}})^{-1} \cdot (\mathbf{M}_{\mathbf{L}}^{\mathbf{x},\text{Lim}} \cdot \mathbb{W}^{y}(\overline{\boldsymbol{u}}_{i-\frac{1$$

$$\tilde{\boldsymbol{v}}_{i,j} = \begin{cases}
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \hat{\boldsymbol{v}}_{i,j-\frac{1}{2}} + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = \beta_{i,j+\frac{1}{2}} = 0, \\
 (\mathbf{M}^{\mathbf{y},FV}_{\mathbf{L}} \cdot \overline{\boldsymbol{v}}_{i,j-\frac{1}{2}} + \mathbf{M}^{\mathbf{y},FV}_{\mathbf{R}} \cdot \overline{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = \beta_{i,j+\frac{1}{2}} = 1, \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \hat{\boldsymbol{v}}_{i,j-\frac{1}{2}} + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j+\frac{1}{2}})) & \text{if} & \beta_{i,j} = \beta_{i,j-\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1, \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{R}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j-\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}}) & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \mathbb{W}^{\mathbf{y}}(\overline{\boldsymbol{v}}_{i,j+\frac{1}{2}}) + \mathbf{M}^{\mathbf{y},DG}_{\mathbf{L}} \cdot \hat{\boldsymbol{v}}_{i,j+\frac{1}{2}} & \text{if} & \beta_{i,j} = \beta_{i,j+\frac{1}{2}} = 0, \beta_{i,j+\frac{1}{2}} = 1. \\
 (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y}})^{-1} \cdot (\mathbf{M}^{\mathbf{y}$$

The description of the two-dimensional limited method is concluded by discussing the CFL stability condition. The restriction on the time step is given by the stability condition for the explicit update of the nonlinear convective terms. We can easily verify that  $\Delta x_s = \Delta x (2P+1)^{-1}$  and  $\Delta y_s = \Delta y (2P+1)^{-1}$ . Consequently, the time step  $\Delta t$  given in 107 for the subcell finite volume scheme is equivalent to the one expressed in 96 for the pure unlimited high order DG method. Moreover the 2D limited method acknowledges the same time step restriction.

#### 4. Numerical tests for the one-dimensional method

In this section we simulate several test cases in order to validate the new staggered semi-implicit discontinuous Galerkin method presented in section 2 and referred to as SIDG in the following. The aim of the first test, which is <sup>165</sup> reported in subsection 4.1, is to verify the correctness of the implementation of the high order unlimited DG scheme. Then, in subsection 4.2 the numerical results are compared against the exact solution of the Riemann problem in order to check the robustness of the limiter in presence of discontinuities. We remind that in all the benchmarks we do neither use artificial viscosity nor any smoothing or filtering of the solution. All numerical solutions are displayed on the figures as a set of equidistant samples in the DG cell by evaluating the underlying DG polynomial. More precisely <sup>170</sup> N sample points are displayed in 1D for a P<sup>N</sup> DG polynomial. For a FV subcell one sample centered point is plotted. Moreover each troubled DG cell and its 2P + 1 FV subcells, are colored in red, while unlimited DG cells are either plotted in blue or in black. When available, the numerical results are compared to an exact solution.

#### *4.1. Advection of a smooth density wave in 1D*

Here, we consider a sanity test used to validate the correctness of the formulation of the pure DG scheme derived in subsection 2.2. As initial condition, in the computational domain Ω<sup>x</sup> = [−1, +1] we impose a smooth Gaussian profile for the density

> ρ(x, 0) = ρ0(1 + e − <sup>0</sup>.5<sup>x</sup> 2 <sup>0</sup>.<sup>12</sup> ) (133)

with ρ<sup>0</sup> = 0.01 and a constant value for the velocity u(x, 0) = u<sup>0</sup> = 2 and for the pressure p(x, 0) = p<sup>0</sup> = 1. <sup>175</sup> Consequently, the density wave is transported without deformation. If we prescribe periodic boundary conditions, the exact solution at the final time tend = 1 coincides with the initial condition. Two simulations are performed: first with a grid made of 50 cells with a polynomial degree P = 5 while, in the second case with 300 control volumes and P = 0 corresponding to a low order finite volume scheme. Note that in both cases the degrees of freedom are equal to 300. The implicitness parameter θ is fixed equal to 0.55. The results for density, velocity and pressure are presented <sup>180</sup> in Fig. 12. It is possible to observe, that for this smooth test the results obtained using the high order semi-implicit DG scheme are extremely accurate and they reach an excellent agreement with the analytical solution. On the contrary, for the finite volume case we notice that the numerical solution for the density is drastically diffused. In addition, in the plots for the velocity and for the pressure, the numerical noise of the FV method is several orders of magnitude bigger than the noise produced by the DG scheme with P = 5.

![](_page_29_Figure_6.jpeg)

Figure 12: Reference and numerical solutions for the Advection of a smooth density wave in 1D problem at tend = 1.

#### <sup>185</sup> *4.2. One-dimensional Riemann problems*

Here we apply the novel semi-implicit DG scheme to a set of Riemann problems taken from [69, 26, 66]. These simulations are carried out in order to check the correct propagation of the shock waves and to verify that the new method does not produce spurious or dispersive oscillations in the vicinity of discontinuities.

The computational domain is Ω<sup>x</sup> = [−0.5, +0.5] and the initial condition consists in a discontinuity centred in x<sup>0</sup>

$$\mathbf{V}(x,0) = \begin{cases} \mathbf{V_L} & \text{if} \quad x \le x_0, \\ \mathbf{V_R} & \text{if} \quad x > x_0, \end{cases}$$
 (134)

where V indicates the vector of primitive variables V = [ρ, u, p]. In addition the simulation run up to t = tend. The parameters for the initial conditions are listed in Table 1 and the results are shown in Figs. 13,14,15,16,17,18 and 19 respectively. RP1 (see Fig. 13) is known as Sod shock tube [60]. We observe an excellent agreement with the

| Case | ρL      | uL      | pL      | ρR      | uR       | pR     | x0   | Nx  | ∆x   | tend  |
|------|---------|---------|---------|---------|----------|--------|------|-----|------|-------|
| RP1  | 1.0     | 0.0     | 1.0     | 0.125   | 0.0      | 0.1    | 0.0  | 100 | 0.01 | 0.2   |
| RP2  | 0.445   | 0.698   | 3.528   | 0.5     | 0.0      | 0.571  | 0.0  | 100 | 0.01 | 0.14  |
| RP3  | 1.0     | 0.0     | 1000.0  | 1.0     | 0.0      | 0.01   | 0.1  | 100 | 0.01 | 0.012 |
| RP4  | 5.99924 | 19.5975 | 460.894 | 5.99924 | -6.19633 | 46.095 | 0.0  | 100 | 0.01 | 0.035 |
| RP5  | 1.0     | -1.0    | 0.4     | 1.0     | +1.0     | 0.4    | 0.0  | 100 | 0.01 | 0.15  |
| RP6  | 1.0     | +2.0    | 0.1     | 1.0     | -2.0     | 0.1    | 0.0  | 100 | 0.01 | 0.5   |
| RP7  | 1.0     | 0.75    | 1.0     | 0.125   | 0.0      | 0.1    | -0.1 | 100 | 0.01 | 0.2   |

Table 1: Initial and scheme conditions for the Riemann problems for semi-implicit DG schemes on staggered grid with subcell limiter - Left and right states for ρ,U and p, initial discontinuity location x0, number of control volumes Nx, cell size ∆<sup>x</sup> and final time of the simulations tend.

reference solution and, moreover, the limiter is activated in the region of the shock wave while the rarefaction and the <sup>190</sup> contact waves are solved only by using the pure DG method.

RP2 (see Fig. 14) is called Lax problem [50]. Also here we see a good fit with the exact solution The new staggered semi-implicit DG scheme activates the *a posteriori* subcell FV limiter appropriately only at the shock wave.

RP3 and RP4 (see Figs. 15 and 16) are taken from the well-known textbook of Toro [69]; these test cases involve strong shock waves and consequently they are suitable for checking the robustness of the scheme. In RP3 despite <sup>195</sup> a small overshoot on the shock wave, in the other parts of the domain the agreement is excellent. Also in RP4 (see Fig. 16) the shocks are very well resolved but we observe some additional troubled cells, probably because of some numerical noise in the plateau region.

RP5 (see Fig. 17) consists of two symmetric rarefaction waves. We notice a small spurious peak close to the origin but elsewhere the numerical solution agrees with the exact one. Moreover, troubled cells are detected only in the first <sup>200</sup> part of the simulation because the initial condition is discontinuous.

In RP6 (see Fig. 18) two jets collide and a double shock is generated; we observe that the waves are perfectly symmetric and the troubled cell are optimally detached and solved without occurrence of any Gibbs phenomena. Note that the glitch in the density has been observed also in [26]

RP7 (see Fig. 19) is a modified Sod problem proposed by Toro in [69] used to verify the presence of an entropy glitch <sup>205</sup> inside the left moving rarefaction, sometimes observed in some entropy-violating approximate Riemann solver [58]. Again, the agreement is good and we can conclude that the semi-implicit finite volume subcell limiter applied to the staggered semi-implicit DG scheme is an appropriate shock capturing strategy. Moreover, our numerical experiments have shown that the use of the limiter was necessary for all these test cases in order to successfully reach the final time tend. The unlimited DG scheme alone produces strong spurious oscillations in the vicinity of discontinuities that <sup>210</sup> may lead to negative densities and pressures, which then make the code crash. Contrarily, our *a posteriori* subcell finite volume limiter successfully cures these spurious oscillations and renders the code at the same time robust and accurate on these demanding test cases.

## 5. Numerical tests for the two-dimensional method

Finally, we carry out the numerical validation of the two-dimensional semi-implicit staggered DG scheme with <sup>215</sup> *a posteriori* subcell finite volume limiting. First we consider two smooth problems: in section 5.1 we compute the

![](_page_31_Figure_0.jpeg)

Figure 13: RP1 - Reference and numerical solutions for ρ, u and p at tend = 0.2

![](_page_31_Figure_2.jpeg)

Figure 14: RP2 - Reference and numerical solutions for ρ, u and p at tend = 0.14

![](_page_31_Figure_4.jpeg)

Figure 15: RP3 - Reference and numerical solutions for ρ, u and p at tend = 0.012

![](_page_32_Figure_0.jpeg)

Figure 16: RP4 - Reference and numerical solutions for ρ, u and p at tend = 0.035

![](_page_32_Figure_2.jpeg)

Figure 17: RP5 - Reference and numerical solutions for ρ, u and p at tend = 0.15

![](_page_32_Figure_4.jpeg)

Figure 18: RP6 - Reference and numerical solutions for ρ, u and p at tend = 0.5

![](_page_33_Figure_0.jpeg)

Figure 19: RP7 - Reference and numerical solutions for ρ, u and p at tend = 0.2

convergence table and the numerical order of convergence of the 2D scheme; then the test case in section 5.3 can be considered as a sanity check analogous to the one of section 4.1, while, in section 5.4, some low Mach number regime simulations are run. We conclude this test suite with the 2D Riemann problems in section 5.6 where the use of the limiter is fundamental in order to complete the simulations, and, obtain a clean, robust and accurate solution of the <sup>220</sup> PDEs.

#### *5.1. Isentropic vortex*

The first two-dimensional test is the well known *isentropic vortex* (see [41]). For the 2D Euler equations, this is a fundamental benchmark because this test checks the correctness of the implementation and the numerical convergence. In addition, the boundary conditions can be either set to periodic or transmissive. The initial condition for the density and for the pressure are ρ(x, y, 0) = ρ<sup>∞</sup> + δρ, p(x, y, 0) = p<sup>∞</sup> + δp, where δρ and δp read as follows

$$\delta \rho = (1 + \delta T)^{\frac{1}{\gamma - 1}} - 1, \quad \delta p = (1 + \delta T)^{\frac{\gamma}{\gamma - 1}} - 1, \quad \text{with} \quad \delta T = \frac{(\gamma - 1)\beta^2}{8\gamma \pi^2} e^{1 - r^2},$$
 (135)

where we have introduced the radial coordinate r = p (x − x0) <sup>2</sup> + (y − y0) <sup>2</sup>, the vortex center (x0, y0) = (0, 0) and, β, the vortex strength, here imposed equal to 5. The initial velocity is [u(x, y, 0), v(x, y, 0)] = [u<sup>∞</sup> + δu, v<sup>∞</sup> + δv] where δu and δv are given as

$$\delta u = -\frac{\beta}{2\pi} e^{\frac{1-r^2}{2}} (y - y_0), \quad \delta v = \frac{\beta}{2\pi} e^{\frac{1-r^2}{2}} (x - x_0).$$
 (136)

If the undisturbed velocity components u<sup>∞</sup> and v<sup>∞</sup> are imposed equal to 0, then the vortex is stationary. In this particular case we computed the L<sup>2</sup> error between the numerical and the exact solutions on a series of successively refined grids characterized by square cells in a domain Ωxy = [−5, +5] × [−5, +5] with N<sup>x</sup> = Ny. For this steady <sup>225</sup> state test, the choice of θ = 1 makes sense because the problem is stationary. The results for the errors and for the convergence rates are reported in the following table and it is possible to state that the semi-implicit DG method on staggered meshes is indeed arbitrary high order accurate in space, see also [25] for the shallow water equations. In particular for a given polynomial degree P, we observe that the convergence rate is equal to P + 1 for the absolute value of the velocity (V = √ u <sup>2</sup> + v <sup>2</sup>). For the pressure, an optimal convergence rate is reported for P even while <sup>230</sup> sub-optimal for P odd. We attribute this behaviour to the choice of the basis functions; more detailed analyses will be carried out in the future.

Later on, we impose the components of the velocity [u∞, v∞] equal to [1, 1]. The computational grid is composed of 625 square cells with N<sup>x</sup> = N<sup>y</sup> = 25, the degree of the polynomial basis functions is P = 5 and Ωxy = [−5, +5] × [−5, +5]. We impose the implicitness parameter equal to θ = 0.7 and periodic boundary conditions are

| P | Nx<br>= Ny | P<br><br>L2 | Vt<br><br>L2 | OP<br>L2 | Vt<br>O<br>L2 |
|---|------------|-------------|--------------|----------|---------------|
| 0 | 100        | 1.94E-2     | 3.90E-2      | —        | —             |
|   | 200        | 9.69E-3     | 1.71E-2      | 0.99     | 1.18          |
|   | 300        | 6.58E-3     | 1.10E-2      | 0.95     | 1.09          |
|   | 400        | 4.99E-3     | 8.10E-3      | 0.95     | 1.06          |
|   | 100        | 2.44E-2     | 3.14E-3      | —        | —             |
|   | 200        | 1.25E-2     | 9.83E-4      | 0.96     | 1.67          |
| 1 | 300        | 8.40E-3     | 4.64E-4      | 0.98     | 1.85          |
|   | 350        | 7.21E-3     | 3.52E-4      | 0.99     | 1.77          |
|   | 400        | 6.32E-3     | 2.77E-4      | 0.99     | 1.80          |
|   | 50         | 3.43E-3     | 1.67E-3      | —        | —             |
|   | 100        | 6.31E-4     | 2.76E-4      | 2.44     | 2.59          |
| 2 | 150        | 1.79E-4     | 9.03E-5      | 3.10     | 2.75          |
|   | 200        | 7.69E-5     | 3.88E-5      | 2.93     | 2.93          |
|   | 25         | 8.89E-4     | 2.90E-4      | —        | —             |
|   | 50         | 1.14E-4     | 2.49E-5      | 2.96     | 3.53          |
| 3 | 75         | 3.35E-5     | 6.36E-6      | 3.02     | 3.36          |
|   | 100        | 1.41E-5     | 2.51E-6      | 3.01     | 3.23          |
| 4 | 15         | 9.82E-4     | 3.42E-4      | —        | —             |
|   | 30         | 7.67E-5     | 2.14E-5      | 3.67     | 4.00          |
|   | 45         | 1.73E-5     | 3.04E-6      | 3.66     | 4.86          |
|   | 60         | 5.58E-6     | 1.03E-6      | 3.93     | 3.75          |
| 5 | 15         | 8.39E-4     | 3.35E-4      | —        | —             |
|   | 30         | 4.97E-5     | 5.27E-6      | 4.07     | 5.98          |
|   | 45         | 7.02E-6     | 5.89E-7      | 4.82     | 5.40          |
|   | 60         | 1.46E-6     | 2.85E-7      | 5.44     | 2.52          |

Table 2: Shu-Osher vortex — Numerical convergence rates computed with respect to the L<sup>2</sup> error norms of the pressure and velocity for the two-dimensional semi-implicit staggered DG scheme.

<sup>235</sup> prescribed. Hence, at the final time tend = 10 the vortex is centred again at its initial position at (x0, y0) = (0, 0) and the exact solution corresponds to the initial one (see (135) and (136)). In Fig. 20 we report the contour plots for the density and pressure where it can be observed that the numerical solution is symmetric and free from spurious oscillations.

#### <sup>240</sup> *5.2. Steady Taylor-Green vortex at low Mach number*

Here, we carry out a benchmark characterized by nearly incompressible regime which is the stationary and inviscid case of the well known Taylor-Green vortex (see e.g. [63, 33, 31, 76]). The initial condition is given as follows

$$\rho(x, y, t) = \rho_0 \nu(x, y, t) = \sin(x)\cos(y) 
v(x, y, t) = -\cos(x)\sin(y) 
p(x, y, t) = C + \frac{1}{4}(\cos(2x) + \sin(2y))$$
(137)

with ρ<sup>0</sup> = 1 and C = 10<sup>4</sup> . With this choice of parameters, the Mach number of this test case is of the order M ≈ 10<sup>−</sup><sup>2</sup> , i.e. the use of an *explicit* scheme would be computationally very inefficient due to the CFL condition on the time step based on the sound speed, while the time step of our semi-implicit staggered DG scheme is only limited by a CFL condition based on the flow velocity. We consider a computational domain Ωxy = [0, 2π] × [0, 2π]

![](_page_35_Figure_0.jpeg)

Figure 20: Shu-Osher vortex - Contour plots for the density and the pressure at time tend = 10.

composed of 50<sup>2</sup> <sup>245</sup> cells (N<sup>x</sup> = N<sup>y</sup> = 50) and a polynomial approximation degree of P = 5. The final time of the simulation is tend = 10 and the reference solution at this time is equal to the initial condition, since the inviscid Taylor Green vortex is a stationary solution of the incompressible Euler equations. In Fig. 21 we report the contour plots for the components of the velocity and for the pressure and we can observe a very clean solution for all the quantities. In addition, in Fig. 22 we expose 1D cuts along the x and the y axis for the velocity components u and v and for the <sup>250</sup> pressure p. For all quantities we note an excellent agreement between our numerical results and the reference solution.

![](_page_35_Figure_3.jpeg)

Figure 21: Taylor-Green vortex - Contour plots for u, v and p at time tend = 10.0.

## *5.3. Advection of a smooth density bell in 2D*

Similarly to the 1D method in 4.1, this test consists in a two dimensional smooth Gaussian bell moving in a uniform pressure and velocity flow. Hence the initial condition reads as follows

$$\rho(x,y,0) = \rho_0(1 + e^{-\frac{0.5r^2}{0.1^2}}), \quad p(x,y,0) = 1, \quad u(x,y,0) = 2, \quad v(x,y,0) = 2.$$
(138)

![](_page_36_Figure_0.jpeg)

Figure 22: Taylor-Green vortex - Reference and cut-of the numerical solutions along the x axis for u, v and p at time tend = 10.0.

The computational domain is Ωxy = [−1, +1] × [−1, +1] and tend = 1; consequently, when imposing periodic boundary conditions the reference solution at the final time coincides with the initial condition in (138). We consider two configurations both with θ = 0.55; in the first one the polynomial degree is P = 5 and N<sup>x</sup> = N<sup>y</sup> = 50 and <sup>255</sup> for the second case N<sup>x</sup> = N<sup>y</sup> = 300 using P = 0, which corresponds to the piece-wise constant finite volume data. Note that in both situations the number of degrees of freedom is the same and equal to N<sup>x</sup> × N<sup>y</sup> × P <sup>2</sup> = 9 × 10<sup>4</sup> . The results for the contour plot at time tend = 1 are shown in Fig. 23. Here we observe that for the high order DG case the solution is centred and symmetric, while we can see the spurious effects of the numerical viscosity in the first order situation. Moreover, in Fig. 24, we report a comparison between the numerical and the exact solutions both for <sup>260</sup> a slice along the x axis. Similarly to the 1D case in section 4.1, in the FV case the dissipation is important while the semi-implicit staggered DG scheme produces a very accurate solution. Consequently, this underlines the importance of the high order approach for the numerical solution of nonlinear partial differential equations.

![](_page_36_Figure_3.jpeg)

Figure 23: Advection of a smooth density bell in 2D - Density for P = 5 (high order DG) and for P = 0 (first order FV) at time tend = 1.

![](_page_37_Figure_0.jpeg)

Figure 24: Advection of a smooth density bell in 2D - Reference and numerical solutions along the x axis for  $\rho$ , velocity component u and p at time  $t_{end}=1$ .

#### 5.4. Two-dimensional smooth acoustic wave propagation

Let us consider another smooth test for the two-dimensional Euler equations. The initial pressure is a smooth surface,  $p(x,y,0)=1+e^{-\alpha r^2}$ , while the density is constant,  $\rho(x,y,0)=1$ , and both components of velocity are null, u(x,y,0)=0, v(x,y,0)=0. For this test  $\alpha=40$  and  $r=\sqrt{x^2+y^2}$  is the radial coordinate. This simulation is characterized by a cylindrically symmetric solution with a low Mach number. However, because the stability condition of the semi-implicit DG scheme is based on the fluid velocity, consequently this scheme is not overly constrained by these kind of acoustic regimes. On the contrary, the same test solved numerically by an explicit method would be very inefficient; in fact, for this last case the stability condition is based on the maximum eigenvalue ||U||+c.

For this simulation the computational domain is  $\Omega_{xy} = [-2, +2] \times [-2, +2]$  and we use  $N_x = N_y = 25$  with a total number of 625 cells using a polynomial degree P equal to 5. Hence, the total number of degrees of freedom is 22500 and the final time is set to 1. The implicitness parameter  $\theta$  is chosen equal to 0.55 in order to get as close as possible to the second order accurate Crank-Nicholson scheme in time.

In Fig. 25 we can observe that the color contour plots for the density and the pressure perfectly respect the cylindrical symmetry. In addition, in Fig. 26 we compare the numerical results against a reference solution obtained running a second order explicit TVD scheme on a very fine mesh. Except for some slight numerical dissipation close the to peaks, the agreement is excellent and all the acoustic waves travel with the proper speed. According to [66], we can state that the semi-implicit staggered DG is a very suitable numerical method for CFD simulation in the low Mach number regime.

#### 5.5. Circular explosion

Here we consider a well known two dimensional Riemann problem characterized by a cylindrical symmetry. Given a computational domain  $\Omega_{xy} = [-1, +1] \times [-1, +1]$  the initial condition reads

$$\mathbf{Q}(x, y, 0) = \begin{cases} \mathbf{Q_{in}} & \text{if} \quad r \le r_c \\ \mathbf{Q_{out}} & \text{if} \quad r > r_c. \end{cases}$$
 (139)

where  $r=\sqrt{x^2+y^2}$  is the radial coordinate and  $\mathbf{Q}=[\rho,u,v,p]$  is state vector. For this simulation we impose the polynomial degree P equal to 5 and we use  $N_x=N_y=100$  with  $\theta_{DG}=0.55$  and  $t_{end}=0.2$ . In Fig. 27 we see that the density has a very clean contour plot. In addition we report a map of the troubled cells; most of them are located in the vicinity of the shock wave and few other control volumes are limited in the zone of the contact wave. We can observe that a vast majority of the cells are dealt with the unlimited optimal DG scheme, while few troubled cells demand a re-calculation. Moreover, a cut of the solution along the x axis is compared against the reference solution

![](_page_38_Figure_0.jpeg)

Figure 25: Two-dimensional smooth acoustic wave propagation - Contour plots for the density and pressure at time tend = 1.

![](_page_38_Figure_2.jpeg)

Figure 26: Two-dimensional smooth acoustic wave propagation - Reference and numerical solutions along the x axis for ρ, u and p at time tend = 1.

in Fig. 28. These last data are obtained using a robust second order TVD scheme (see [75]) on a fine grid solving an <sup>290</sup> equivalent 1D Euler system of PDEs with a source term taking into account the effect of the radial symmetry [69]. We observe an excellent agreement between the solutions; furthermore in the numerical solution no oscillation is observed, and, all waves are solved with their correct speed of propagation.

#### *5.6. Two dimensional Riemann problems*

For the last benchmarks we consider a class of two dimensional Riemann problems presented in [59] and studied in detail in [49]. The computational domain Ωxy = [−0.5, +0.5] × [−0.5, +0.5] is paved using N<sup>x</sup> = N<sup>y</sup> = 50 cells with a polynomial degree P = 5. The initial condition reads as follows

$$(\rho, u, v, p) = \begin{cases} (\rho_1, u_1, v_1, p_1) & \text{if} \quad x > 0 \land y > 0, \\ (\rho_2, u_2, v_2, p_2) & \text{if} \quad x \le 0 \land y > 0, \\ (\rho_3, u_3, v_3, p_3) & \text{if} \quad x \le 0 \land y \le 0, \\ (\rho_4, u_4, v_4, p_4) & \text{if} \quad x > 0 \land y \le 0, \end{cases}$$

$$(140)$$

![](_page_39_Figure_0.jpeg)

Figure 27: Circular explosion - Contour plots for the density and map of the troubled zones (red cells) at time tend = 0.2.

![](_page_39_Figure_2.jpeg)

Figure 28: Circular explosion - Reference and cut-of the numerical solutions along the x axis for ρ, u and p at time tend = 0.2.

We consider four initial configurations, the same ones carried out in [26], the initial parameters of which are listed in <sup>295</sup> Table 3. From Figs. 29,30,31, 32 we can state that the numerical result are in good agreement when compared to the simulations carried out in [49, 26]. In addition we observe that most of the troubled control volumes are located on the discontinuities but, probability due to numerical noise, few bad cells are also detected in the plateau regions and at boundaries. Moreover, since this method is at most second order accurate in time we observe a relevant numerical viscosity which, here, is clearly higher than the one observed in explicit DG schemes with *a posteriori* subcell WENO <sup>300</sup> limiter, see [32, 78]. In order to achieve higher order in time, the present staggered semi-implicit DG scheme could be extended to a semi-implicit staggered space-time DG method, see [33, 64, 66].

## 6. Conclusions and outlook

In this paper we have proposed a novel family of semi-implicit DG schemes on staggered meshes with *a posteriori* subcell finite volume limiter applied to the 1D and 2D Euler equations of gasdynamics. In particular, we have extended <sup>305</sup> to high order of accuracy the FV method presented in [26] and then, in order to develop a robust shock-capturing

| Configuration C4  |               |               |         |      |  |  |  |  |
|-------------------|---------------|---------------|---------|------|--|--|--|--|
|                   | ρ             | u<br>v        |         | p    |  |  |  |  |
| (x > 0, y > 0)    | 1.1           | 0.0           | 0.0     | 1.1  |  |  |  |  |
| (x < 0, y > 0)    | 0.5065        | 0.8939<br>0.0 |         | 0.35 |  |  |  |  |
| (x < 0, y < 0)    | 1.1<br>0.8939 |               | 0.8939  | 1.1  |  |  |  |  |
| (x > 0, y < 0)    | 0.5065        | 0.0           | 0.8939  | 0.35 |  |  |  |  |
| Configuration C7  |               |               |         |      |  |  |  |  |
|                   | ρ             | U             | V       | p    |  |  |  |  |
| (x > 0, y > 0)    | 0.5197        | -0.6259       | 0.1     | 0.4  |  |  |  |  |
| (x < 0, y > 0)    | 1.0           | 0.1           | 0.1     | 1.0  |  |  |  |  |
| (x < 0, y < 0)    | 0.8           | 0.1           | 0.1     | 0.4  |  |  |  |  |
| (x > 0, y < 0)    | 0.5197        | 0.1           | -0.6259 | 0.4  |  |  |  |  |
| Configuration C8  |               |               |         |      |  |  |  |  |
|                   | ρ             | U             | V       | p    |  |  |  |  |
| (x > 0, y > 0)    | 0.5197        | 0.1           | 0.1     | 0.4  |  |  |  |  |
| (x < 0, y > 0)    | 1.0           | -0.6259       | 0.1     | 1.0  |  |  |  |  |
| (x < 0, y < 0)    | 0.8           | 0.1           | 0.1     | 1.0  |  |  |  |  |
| (x > 0, y < 0)    | 1.0           | 0.1           | -0.6259 | 1.0  |  |  |  |  |
| Configuration C16 |               |               |         |      |  |  |  |  |
|                   | ρ             | U             | V       | p    |  |  |  |  |
| (x > 0, y > 0)    | 0.5313        | 0.1           | 0.1     | 0.4  |  |  |  |  |
| (x < 0, y > 0)    | 1.0222        | -0.6179       | 0.1     | 1.0  |  |  |  |  |
| (x < 0, y < 0)    | 0.8           | 0.1           | 0.1     | 1.0  |  |  |  |  |
| (x > 0, y < 0)    | 1.0           | 0.1           | 0.8276  | 1.0  |  |  |  |  |

Table 3: Initial conditions for the 2D Riemann problems.

![](_page_41_Figure_0.jpeg)

Figure 29: Two dimensional Riemann problem RP2D C4 - Contour plots for the density, map of the troubled zones (red cells) and contour lines at time tend = 0.2.

![](_page_41_Figure_2.jpeg)

Figure 30: Two dimensional Riemann problem RP2D C7 - Contour plots for the density, map of the troubled zones (red cells) and contour lines at time tend = 0.2.

strategy, we followed the approach used in [44] where the *a posteriori* subcell liming was extended to staggered semiimplicit schemes for the first time.

Similarly to the other *a posteriori* subcell finite volume limiters for explicit DG schemes on collocated grids developed in [32, 29, 6], this approach is based on the MOOD paradigm introduced in [16, 21, 22, 53]. Hence, an unlimited <sup>310</sup> semi-implicit staggered DG method is first applied in order to produce a so-called discrete *candidate solution* at time t <sup>n</sup>+1. Then, applying physical and numerical admissibility criteria, *troubled cells* characterized by a non-valid solution are detected. Then, a more robust first order semi-implicit staggered finite volume scheme is applied in the control volumes flagged as troubled cells. Successively, the linear system for the pressure is solved again involving the unlimited DG cells together with the finite volume subcells. The algorithm is concluded with the reconstruction <sup>315</sup> of the DG polynomial from the piecewise constant subcell averages in the troubled cells.

Several benchmarks have been carried out in order to confirm that the new schemes behave well both for the low Mach number regime, due to the implicit treatment of the pressure term, and for several kinds of Riemann problems, since the *a posteriori* finite volume limiter stabilizes the DG scheme in the presence of shock waves and steep gradients.

In further works we will consider the investigation of cavitating compressible flows for industrial applications, see <sup>320</sup> [27, 45], implementing the *a posteriori* subcell finite volume limiting for the semi-implicit DG proposed in [43]. In addition, we will extend the present subcell FV limiter for semi-implicit staggered DG schemes to adaptive Cartesian

![](_page_42_Figure_0.jpeg)

Figure 31: Two dimensional Riemann problem RP2D C8 - Contour plots for the density, map of the troubled zones (red cells) and contour lines at time tend = 0.2.

![](_page_42_Figure_2.jpeg)

Figure 32: Two dimensional Riemann problem RP2D C16 - Contour plots for the density, map of the troubled zones (red cells) and contour lines at time tend = 0.2.

meshes (AMR), see [78, 77, 77, 34]. Finally, another path to explore consists in the extension of the present staggered semi-implicit FV and DG methods to the compressible Navier-Stokes equations and to other nonlinear hyperbolic systems, such as the Baer-Nunziato model for compressible multi-phase flows and the unified first order hyperbolic <sup>325</sup> GPR model proposed in [42, 31], which unifies a large range of different models of continuum mechanics.

#### 7. Acknowledgments

The authors acknowledge funding from the Istituto Nazionale di Alta Matematica (INdAM) via the GNCS group and the program *Young Researchers Funding 2018* via the research project *Semi-implicit structure preserving schemes for continuum mechanics*. MD acknowledges the financial support received from the Italian Ministry of Education, <sup>330</sup> University and Research (MIUR) in the frame of the Departments of Excellence Initiative 2018–2022 attributed to DICAM of the University of Trento (grant L. 232/2016) and in the frame of the PRIN 2017 project *Innovative numerical methods for evolutionary partial differential equations and applications*. MD has also received funding from the University of Trento via the Strategic Initiative *Modeling and Simulation*.

#### 8. References

- <sup>335</sup> [1] F. Bassi, A. Crivellini, D.A. Di Pietro, and S. Rebay. An artificial compressibility flux for the discontinuous Galerkin solution of the incompressible Navier–Stokes equations. *Journal of Computational Physics*, 218:208–221, 2006.
  - [2] F. Bassi, A. Crivellini, D.A. Di Pietro, and S. Rebay. An implicit high-order discontinuous Galerkin method for steady and unsteady incompressible flows. *Computers and Fluids*, 36:1529–1546, 2007.
- [3] F. Bassi and S. Rebay. A high-order accurate discontinuous finite element method for the numerical solution of the compressible Navier-<sup>340</sup> Stokes equations. *Journal of Computational Physics*, 131:267–279, 1997.
  - [4] C. Baumann and J. Oden. A discontinuous hp finite element method for convection-diffusion problems. *Computer Methods in Applied Mechanics and Engineering*, 175:311–341, 1999.
  - [5] C. Baumann and J. Oden. A discontinuous hp finite element method for the Euler and Navier-Stokes equation. *International Journal for Numerical Methods in Fluids*, 31:79–95, 1999.
- <sup>345</sup> [6] W. Boscheri and M. Dumbser. Arbitrary-LagrangianEulerian Discontinuous Galerkin schemes with a posteriori subcell finite volume limiting on moving unstructured meshes. *Journal of Computational Physics*, 346:449 – 479, 2017.
  - [7] V. Casulli. Semi-implicit finite difference methods for the two–dimensional shallow water equations. *Journal of Computational Physics*, 86:56–74, 1990.
- [8] V. Casulli. A high-resolution wetting and drying algorithm for free-surface hydrodynamics. *International Journal for Numerical Methods in* <sup>350</sup> *Fluids*, 60:391–408, 2009.
  - [9] V. Casulli. A semi–implicit numerical method for the free–surface Navier–Stokes equations. *International Journal for Numerical Methods in Fluids*, 74:605–622, 2014.
  - [10] V. Casulli and E. Cattani. Stability, accuracy and efficiency of a semi-implicit method for three-dimensional shallow water flow. *Computers and Mathematics with Applications*, 27(4):99 – 112, 1994.
- <sup>355</sup> [11] V. Casulli and R. T. Cheng. Semi-implicit finite difference methods for three–dimensional shallow water flow. *International Journal for Numerical Methods in Fluids*, 15:629–648, 1992.
  - [12] V. Casulli, M. Dumbser, and E. F. Toro. Semi-implicit numerical modeling of axially symmetric flows in compliant arterial systems. *International Journal for Numerical Methods in Biomedical Engineering*, 28:257–272, 2012.
- [13] V. Casulli and D. Greenspan. Pressure method for the numerical solution of transient, compressible fluid flows. *International Journal for* <sup>360</sup> *Numerical Methods in Fluids*, 4(11):1001–1012, 1984.
  - [14] V. Casulli and S. Stelling. Semi-implicit subgrid modelling of three-dimensional free-surface flows. *International Journal for Numerical Methods in Fluids*, 67:441–449, 2010.
  - [15] V. Casulli and R. A. Walters. An unstructured grid, three–dimensional model based on the shallow water equations. *International Journal for Numerical Methods in Fluids*, 32:331–348, 2000.
- <sup>365</sup> [16] S. Clain, S. Diot, and R. Loubre. A high-order finite volume method for systems of conservation lawsMulti-dimensional Optimal Order Detection (MOOD). *Journal of Computational Physics*, 230(10):4028 – 4050, 2011.
  - [17] B. Cockburn, S. Y. Lin, and C. W. Shu. TVB Runge-Kutta local projection discontinuous Galerkin finite element method for conservation laws III: one dimensional systems. *Journal of Computational Physics*, 84:90–113, 1989.
- [18] B. Cockburn and C. W. Shu. TVB Runge-Kutta local projection discontinuous Galerkin finite element method for conservation laws II: <sup>370</sup> general framework. *Mathematics of Computation*, 52:411–435, 1989.
  - [19] B. Cockburn and C. W. Shu. The Runge-Kutta local projection P1-Discontinuous Galerkin finite element method for scalar conservation laws. *Mathematical Modelling and Numerical Analysis*, 25:337–361, 1991.
  - [20] B. Cockburn and C. W. Shu. The Runge-Kutta discontinuous Galerkin method for conservation laws V: multidimensional systems. *Journal of Computational Physics*, 141:199–224, 1998.
- <sup>375</sup> [21] S. Diot, S. Clain, and R. Loubere. Improved detection criteria for the Multi-dimensional Optimal Order Detection (MOOD) on unstructured ` meshes with very high-order polynomials. *Computers and Fluids*, 64(Supplement C):43 – 63, 2012.
  - [22] S. Diot, R. Loubere, and S. Clain. The Multidimensional Optimal Order Detection method in the three-dimensional case: very high-order ` finite volume method for hyperbolic systems. *International Journal for Numerical Methods in Fluids*, 73(4):362–392, 2013.
- [23] V. Dolejsi and M. Feistauer. A semi-implicit discontinuous Galerkin method for the numerical solution of inviscid compressible flows. <sup>380</sup> *Journal of Computational Physisc*, 198:727–746, 2004.
  - [24] M. Dumbser, D.S. Balsara, M. Tavelli, and F. Fambri. A divergence-free semi-implicit finite volume scheme for ideal, viscous and resistive magnetohydrodynamics. *International Journal for Numerical Methods in Fluids*, 89:16–42, 2018.
  - [25] M. Dumbser and V. Casulli. A staggered semi-implicit spectral discontinuous Galerkin scheme for the shallow water equations. *Applied Mathematics and Computation*, 219(15):8057–8077, 2013.
- <sup>385</sup> [26] M. Dumbser and V. Casulli. A conservative, weakly nonlinear semi-implicit finite volume scheme for the compressible Navier-Stokes equations with general equation of state. *Applied Mathematics and Computation*, 272:479–497, 2016.
  - [27] M. Dumbser, U. Iben, and M.Ioriatti. An efficient semi-implicit finite volume method for axially symmetric compressible flows in compliant tubes . *Applied Numerical Mathematics*, 89:24–44, 2015.
- [28] M. Dumbser and M. Kaser. Arbitrary High Order Non-oscillatory Finite Volume Schemes on Unstructured Meshes for Linear Hyperbolic ¨ <sup>390</sup> Systems. *J. Comput. Phys.*, 221(2):693–723, February 2007.
  - [29] M. Dumbser and R. Loubere. A simple robust and accurate a posteriori sub-cell finite volume limiter for the discontinuous Galerkin method ` on unstructured meshes. *Journal of Computational Physics*, 319(Supplement C):163 – 199, 2016.
  - [30] M. Dumbser and C.D. Munz. Building blocks for arbitrary high order discontinuous Galerkin schemes. *Journal of Scientific Computing*, 27:215–230, 2006.
- <sup>395</sup> [31] M. Dumbser, I. Peshkov, E. Romenski, and O. Zanotti. High order ADER schemes for a unified first order hyperbolic formulation of continuum mechanics: Viscous heat-conducting fluids and elastic solids. *Journal of Computational Physics*, 314:824 – 862, 2016.

- [32] M. Dumbser, O. Zanotti, R. Loubere, and S. Diot. A posteriori subcell limiting of the discontinuous Galerkin finite element method for ` hyperbolic conservation laws. *Journal of Computational Physics*, 278:47 – 75, 2014.
- [33] F. Fambri and M. Dumbser. Spectral semi-implicit and spacetime discontinuous Galerkin methods for the incompressible NavierStokes <sup>400</sup> equations on staggered Cartesian grids. *Applied Numerical Mathematics*, 110:41–74, 2016.
  - [34] F. Fambri and M. Dumbser. Semi-implicit discontinuous Galerkin methods for the incompressible NavierStokes equations on adaptive staggered Cartesian grids. *Computer Methods in Applied Mechanics and Engineering*, 324:170 – 203, 2017.
  - [35] F. Fambri, M. Dumbser, and V. Casulli. An Efficient Semi-Implicit Method for Three-Dimensional Non-Hydrostatic Flows in Compliant Arterial Vessels. *International Journal for Numerical Methods in Biomedical Engineering*, 30:1170–1198, 2014.
- <sup>405</sup> [36] G. Gassner, F. Lorcher, and C.D. Munz. A contribution to the construction of diffusion fluxes for finite volume and discontinuous Galerkin ¨ schemes. *Journal of Computational Physics*, 224:1049–1063, 2007.
  - [37] S.K. Godunov. Finite difference methods for the computation of discontinuous solutions of the equations of fluid dynamics. *Math. USSR Sb.*, 47:271306, 1959.
  - [38] S. Gottlieb and C.W. Shu. Total Variation Diminishing Runge-Kutta Schemes. *Math. Comput.*, 67(221):73–85, January 1998.
- <sup>410</sup> [39] R. Hartmann and P. Houston. Symmetric interior penalty DG methods for the compressible navier–stokes equations I: Method formulation. *Int. J. Num. Anal. Model.*, 3:1–20, 2006.
  - [40] R. Hartmann and P. Houston. An optimal order interior penalty discontinuous galerkin discretization of the compressible navier-stokes equations. *Journal of Computational Physics*, 227:9670–9685, 2008.
- [41] C. Hu and C.W. Shu. Weighted essentially non-oscillatory schemes on triangular meshes. *Journal of Computational Physics*, 150:97–127, <sup>415</sup> 1999.
  - [42] I. Peshkov and E. Romenski. A hyperbolic model for viscous Newtonian flows. *Continuum Mechanics and Thermodynamics*, 28:85–104, 2016.
  - [43] M. Ioriatti and M. Dumbser. Semi-implicit staggered Discontinuous Galerkin schemes for axially symmetric viscous compressible flows in elastic tubes. *Computer and Fluids*.
- <sup>420</sup> [44] M. Ioriatti and M. Dumbser. A posteriori sub-cell finite volume limiting of staggered semi-implicit discontinuous galerkin schemes for the shallow water equations. *Applied Numerical Mathematics*, 2018. submitted to.
  - [45] M. Ioriatti, M. Dumbser, and U. Iben. A comparison of explicit and semi-implicit finite volume schemes for viscous compressible flows in elastic pipes in fast transient regime. *Zeitschrift fuer Angewandte Mathematik und Mechanik*, 97:1358–1380, 2017.
- [46] M. Kaser and A. Iske. ADER Schemes on Adaptive Triangular Meshes for Scalar Conservation Laws. ¨ *J. Comput. Phys.*, 205(2):486–508, <sup>425</sup> May 2005.
  - [47] C.M. Klaij, J.J.W. van der Vegt, and H. van der Ven. Spacetime discontinuous Galerkin method for the compressible NavierStokes equations. *Journal of Computational Physics*, 217(2):589 – 611, 2006.
  - [48] S. C. Kramer and G. S. Stelling. A conservative unstructured scheme for rapidly varied flows. *International Journal for Numerical Methods in Fluids*, 58:183–212, 2008.
- <sup>430</sup> [49] Alexander Kurganov and Eitan Tadmor. Solution of two-dimensional riemann problems for gas dynamics without riemann problem solvers. *Numerical Methods for Partial Differential Equations*, 18(5):584–608.
  - [50] Peter Lax and Burton Wendroff. Systems of conservation laws. *Communications on Pure and Applied Mathematics*, 13(2):217–237.
  - [51] Y. J. Liu, C. W. Shu, E. Tadmor, and M. Zhang. Central discontinuous Galerkin methods on overlapping cells with a non-oscillatory hierarchical reconstruction. *SIAM Journal on Numerical Analysis*, 45:2442–2467, 2007.
- <sup>435</sup> [52] Y. J. Liu, C. W. Shu, E. Tadmor, and M. Zhang. L2-stability analysis of the central discontinuous Galerkin method and a comparison between the central and regular discontinuous Galerkin methods. *Mathematical Modeling and Numerical Analysis*, 42:593–607, 2008.
  - [53] R. Loubere, M. Dumbser, and S. Diot. A New Family of High Order Unstructured MOOD and ADER Finite Volume Schemes for Multidi- ` mensional Systems of Hyperbolic Conservation Laws. *Communications in Computational Physics*, 16(3):718763, 2014.
- [54] J. Von Neumann and R. D. Richtmyer. A method for the numerical calculation of hydrodynamic shocks. *Journal of Applied Physics*, <sup>440</sup> 21:232–237, 1950.
  - [55] J.H. Park and C.D. Munz. Multiple pressure variables methods for fluid flow at all mach numbers. *International Journal for Numerical Methods in Fluids*, 49:905–931, 2005.
  - [56] J. Qiu, M. Dumbser, and C.W. Shu. The discontinuous Galerkin method with Lax-Wendroff type time discretizations. *Computer Methods in Applied Mechanics and Engineering*, 194:4528–4543, 2005.
- <sup>445</sup> [57] W. H. Reed and T. R. Hill. Triangular mesh methods for the neutron transport equation. . Technical Report LA-UR-73-479, Los Alamos Scientific Laboratory, 1973.
  - [58] P.L. Roe. Approximate Riemann solvers, parameter vectors, and difference schemes. *Journal of Computational Physics*, 43:357–372, 1981.
  - [59] C. Schulz-Rinne. Classification of the riemann problem for two-dimensional gas dynamics. *SIAM Journal on Mathematical Analysis*, 24(1):76–88, 1993.
- <sup>450</sup> [60] Gary A Sod. A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws. *Journal of Computational Physics*, 27(1):1 – 31, 1978.
  - [61] G. S. Stelling and S. P. A. Duynmeyer. A staggered conservative scheme for every Froude number in rapidly varied shallow water flows. *International Journal for Numerical Methods in Fluids*, 43:1329–1354, 2003.
- [62] M. Tavelli and M. Dumbser. A high order semi-implicit discontinuous Galerkin method for the two dimensional shallow water equations on <sup>455</sup> staggered unstructured meshes. *Applied Mathematics and Computation*, 234:623–644, 2014.
  - [63] M. Tavelli and M. Dumbser. A staggered semi-implicit discontinuous Galerkin method for the two dimensional incompressible NavierStokes equations. *Applied Mathematics and Computation*, 248:70–92, 2014.
  - [64] M. Tavelli and M. Dumbser. A staggered space-time discontinuous Galerkin method for the incompressible Navier-Stokes equations on two-dimensional triangular meshes. *Computers and Fluids*, 119:235–249, 2015.
- <sup>460</sup> [65] M. Tavelli and M. Dumbser. A staggered space-time discontinuous Galerkin method for the three-dimensional incompressible Navier-Stokes equations on unstructured tetrahedral meshes. *Journal of Compuational Physics*, 319:294–323, 2016.

- [66] M. Tavelli and M. Dumbser. A pressure-based semi-implicit spacetime discontinuous Galerkin method on staggered unstructured meshes for the solution of the compressible NavierStokes equations at all Mach numbers. *Journal of Computational Physics*, 341:341 – 376, 2017.
- [67] M. Tavelli and M. Dumbser. Arbitrary high order accurate space-time discontinuous galerkin finite element schemes on staggered unstructured <sup>465</sup> meshes for linear elasticity. *Journal of Computational Physics*, 366:386 – 414, 2018.
  - [68] M. Tavelli, M. Dumbser, and V. Casulli. High resolution methods for scalar transport problems in compliant systems of arteries. *Applied Numerical Mathematics*, 74:62–82, 2013.
  - [69] E. F. Toro. *Riemann Sovlers and Numerical Methods for Fluid Dynamics*. Springer, 2009.
  - [70] E.F. Toro and M.E. Vazquez-Cend ´ on. Flux splitting schemes for the Euler equations. ´ *Computers and Fluids*, 70:1 12, 2012.
- <sup>470</sup> [71] G. Tumolo and L. Bonaventura. A semi–implicit, semi–Lagrangian discontinuous Galerkin framework for adaptive numerical weather prediction. *Quarterly Journal of the Royal Meteorological Society*, 141(692):2582–2601.
  - [72] G. Tumolo, L. Bonaventura, and M. Restelli. A semi-implicit, semi-Lagrangian,p-adaptive discontinuous Galerkin method for the shallow water equations. *Journal of Compuatational Physics*, 232:46–67, 2013.
- [73] J.J.W. van der Vegt and H. van der Ven. SpaceTime Discontinuous Galerkin Finite Element Method with Dynamic Grid Motion for Inviscid <sup>475</sup> Compressible Flows: I. General Formulation. *Journal of Computational Physics*, 182(2):546 – 585, 2002.
  - [74] H. van der Ven and J.J.W. van der Vegt. Spacetime discontinuous galerkin finite element method with dynamic grid motion for inviscid compressible flows: Ii. efficient flux quadrature. *Computer Methods in Applied Mechanics and Engineering*, 191(41):4747 – 4780, 2002.
  - [75] B. van Leer. Toward the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method. *Journal of Computational Physics*, 32:101 – 136, 1979.
- <sup>480</sup> [76] W.Boscheri, M. Dumbser, and R. Loubere. Cell centered direct arbitrary-lagrangian-eulerian ader-weno finite volume schemes for nonlinear ` hyperelasticity. *Computers and Fluids*, 134-135:111 – 129, 2016.
  - [77] O. Zanotti, M. Dumbser, and F. Fambri. Solving the relativistic magnetohydrodynamics equations with ADER discontinuous Galerkin methods, a posteriori subcell limiting and adaptive mesh refinement. *Monthly Notices of The Royal Astronomical Society*, 452:3010–3029, 2015.
- <sup>485</sup> [78] O. Zanotti, F. Fambri, M. Dumbser, and A. Hidalgo. Space-time adaptive ADER discontinuous Galerkin finite element schemes with a posteriori sub-cell finite volume limiting. *Computer and Fluids*, 118:204–224, 2015.

### Appendix A. Matrices and tensors for the 1D semi-implicit schemes

*Appendix A.1. Matrices and tensors for the 1D semi-implicit DG method*

$$\mathbf{M} = \int_{0}^{1} \boldsymbol{\varphi}(\xi) \boldsymbol{\varphi}(\xi) d\xi \tag{A.1}$$

$$\mathbf{K} = \int_{0}^{1} \boldsymbol{\varphi}'(\xi) \boldsymbol{\varphi}(\xi) d\xi \tag{A.2}$$

$$\mathbf{R_e^{DG}} = \varphi(1)\varphi\left(\frac{1}{2}\right)\varphi\left(\frac{1}{2}\right) - \int_{\frac{1}{2}}^{1} \varphi'(\xi)\varphi\left(\xi - \frac{1}{2}\right)\varphi\left(\xi - \frac{1}{2}\right)d\xi \tag{A.3}$$

$$\mathbf{L}_{\mathbf{e}}^{\mathbf{DG}} = \boldsymbol{\varphi}(0)\boldsymbol{\varphi}\left(\frac{1}{2}\right)\boldsymbol{\varphi}\left(\frac{1}{2}\right) + \int_{0}^{\frac{1}{2}} \boldsymbol{\varphi}'(\xi)\boldsymbol{\varphi}\left(\xi + \frac{1}{2}\right)\boldsymbol{\varphi}\left(\xi + \frac{1}{2}\right)d\xi \tag{A.4}$$

$$\mathbf{R}_{\mathbf{p}}^{\mathbf{DG}} = \varphi\left(\frac{1}{2}\right)\varphi(0) + \int_{\frac{1}{2}}^{1} \varphi(\xi)\varphi'\left(\xi - \frac{1}{2}\right)d\xi \tag{A.5}$$

$$\mathbf{L}_{\mathbf{p}}^{\mathbf{DG}} = \varphi\left(\frac{1}{2}\right)\varphi(1) - \int_{0}^{\frac{1}{2}}\varphi(\xi)\varphi'\left(\xi + \frac{1}{2}\right)d\xi \tag{A.6}$$

$$\mathbf{M_{L}^{DG}} = \int_{0}^{\frac{1}{2}} \varphi(\xi)\varphi\left(\xi + \frac{1}{2}\right)d\xi \tag{A.7}$$

$$\mathbf{M}_{\mathbf{R}}^{\mathbf{DG}} = \int_{\frac{1}{2}}^{1} \varphi(\xi) \varphi\left(\xi - \frac{1}{2}\right) d\xi \tag{A.8}$$

*Appendix A.2. Matrices and tensors for the 1D semi-implicit sub-cell FV method for P=2*

$$\mathbf{R^{FV}} = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 5 & 0 & 0 & 0 & 0 \\ -5 & 5 & 0 & 0 & 0 \\ 0 & -5 & 5 & 0 & 0 \end{pmatrix} \quad \mathbf{L^{FV}} = \begin{pmatrix} 0 & 0 & 5 & -5 & 0 \\ 0 & 0 & 0 & 5 & -5 \\ 0 & 0 & 0 & 0 & 5 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0$$

$$\mathbf{M}_{\mathbf{R}}^{\mathbf{FV}} = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 \\ 0.5 & 0 & 0 & 0 & 0 \\ 0.5 & 0.5 & 0 & 0 & 0 \\ 0 & 0.5 & 0.5 & 0 & 0 \end{pmatrix} \quad \mathbf{M}_{\mathbf{L}}^{\mathbf{FV}} = \begin{pmatrix} 0 & 0 & 0.5 & 0.5 & 0 \\ 0 & 0 & 0 & 0.5 & 0.5 \\ 0 & 0 & 0 & 0 & 0.5 \\ 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0$$

<sup>490</sup> *Appendix A.3. Tensors for the limited 1D semi-implicit DG method*

$$\mathbf{R'_{e}^{DG}} = \varphi(1)\varphi\left(\frac{1}{2}\right) - \int_{\frac{1}{2}}^{1} \varphi'(\xi)\varphi\left(\xi - \frac{1}{2}\right)d\xi \tag{A.11}$$

$$\mathbf{L}_{\mathbf{e}}^{\prime \mathbf{DG}} = \boldsymbol{\varphi}(0)\boldsymbol{\varphi}\left(\frac{1}{2}\right) + \int_{0}^{\frac{1}{2}} \boldsymbol{\varphi}'(\xi)\boldsymbol{\varphi}\left(\xi + \frac{1}{2}\right)d\xi \tag{A.12}$$