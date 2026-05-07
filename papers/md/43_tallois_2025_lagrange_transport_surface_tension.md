![](_page_0_Picture_1.jpeg)

Contents lists available at [ScienceDirect](http://www.ScienceDirect.com/)

journal homepage: [www.elsevier.com/locate/jcp](http://www.elsevier.com/locate/jcp)

![](_page_0_Picture_5.jpeg)

![](_page_0_Picture_6.jpeg)

# Non-conservative Godunov-type schemes: Application to two-phase flows with surface tension using Lagrange-transport splitting strategy

Lucas Tallois <sup>a</sup>*,*b*[,](http://orcid.org/0000-0003-1418-0244) ,*∗, Simon Peluchon <sup>a</sup>, Gérard Gallice <sup>d</sup>, Philippe Villedieu <sup>b</sup>*,*<sup>c</sup>

- <sup>a</sup> *CEA-CESTA, 15 avenue des sablières CS 60001, 33116 Le Barp Cedex, France*
- <sup>b</sup> *INSA, 135 Avenue de Rangueil, 31400 Toulouse, France*
- <sup>c</sup> *DMPE, ONERA, Université de Toulouse, 31000, Toulouse, France*
- <sup>d</sup> *CEA-CESTA, France*

# A R T I C L E I N F O A B S T R A C T

*Keywords:* Godunov-type scheme Surface tension Two-phase flows Riemann solver Implicit-explicit scheme This paper aims to present one-dimensional and multi-dimensional non-conservative Godunovtype schemes in a general framework. These schemes are designed to preserve equilibrium solutions discretely. Gallice's theory of simple solvers is used to solve the Riemann problem approximately. These numerical schemes are applied to compute two-phase flows with surface tension effects. Time integration is based on the Lagrange-Transport splitting strategy, allowing to solve the acoustic waves with an implicit time scheme.

# **1. Introduction**

This work presents multi-dimensional Godunov-type scheme based on an approximate Riemann solver for non-conservative hyperbolic systems of equations. The main difficulty lies in the fact that the non-conservative product is not defined by the usual concepts of distribution theory. This point has been discussed in [\[28](#page-30-0)], and will not be addressed here. The construction of non-conservative Godunov-type schemes based on Riemann solvers has already been addressed by several authors, whether for Roe-type, Osher-type, or simple solvers [\[74](#page-31-0)[,40,39](#page-30-0)[,56](#page-31-0)[,16,18](#page-30-0)[,1](#page-29-0)[,19,17\]](#page-30-0). The advantage of taking into account source terms or non-conservative products directly in the construction of the scheme is that it makes it possible to guarantee equilibrium solutions discretely. The best-known case of source term is probably that of the shallow-water equations with bottom, for which the lake at rest solution must be preserved. Dealing with non-conservative products, one can also call two phaseflow models with surface tension, for which the Laplace law constitutes an equilibrium solution. Schemes that preserve equilibrium solutions are generally referred to as *well-balanced*.

Generally speaking, extension to a dimension greater than one is achieved by solving the approximate Riemann problem on each face separating the mesh cells. The scheme is therefore not truly multi-dimensional since it only uses the neighborhood of faces, and not the neighborhood of nodes. Truly multi-dimensional schemes were first introduced for the resolution of Lagrangian hydrodynamics (see [\[35](#page-30-0)[,52](#page-31-0)[,15,27](#page-30-0)]). Several recent works have extended the notion of multi-dimensional schemes to the Eulerian framework [\[7](#page-30-0)[,69\]](#page-31-0). Recently, Gallice et al. [\[42](#page-31-0)] have shown that it is possible to rewrite the Lagrangian scheme EUCCLHYD [\[52\]](#page-31-0) as a Godunov-type scheme. They then introduced multi-dimensional Godunov-type schemes, to obtain a multi-dimensional, positive, entropic numerical

<https://doi.org/10.1016/j.jcp.2025.113958>

Received 15 May 2024; Received in revised form 22 March 2025; Accepted 24 March 2025

<sup>\*</sup> Corresponding author at: CEA-CESTA, 15 avenue des sablières - CS 60001, 33116 Le Barp Cedex, France. *E-mail addresses:* [tallois.lucas@gmail.fr](mailto:tallois.lucas@gmail.fr) (L. Tallois), [simon.peluchon@cea.fr](mailto:simon.peluchon@cea.fr) (S. Peluchon), [gerard.gallice@gmail.com](mailto:gerard.gallice@gmail.com) (G. Gallice), [philippe.villedieu@onera.fr](mailto:philippe.villedieu@onera.fr) (P. Villedieu).

scheme based on the notion of a simple solver. The key to the construction of these multi-dimensional schemes lies in the notion of non-classical conservation. Numerical fluxes are not equal on either side of the mesh faces. However, it is possible to obtain a globally conservative scheme by assuming that the sum of fluxes around a node cancels out. This ingenious idea makes it possible, by adding a nodal dependency to the numerical fluxes, to construct fluxes with a nodal stencil. Its extension to problems with source term was made in [23,30]. Moreover, as for the surface tension effects, many non-conservative products are a truly multi-dimensional phenomenon. Therefore, the extension of multi-dimensional Godunov-type scheme theory to the non-conservative framework based on an approximate Riemann solver is of interest, and a comparative study of the one-dimensional and multi-dimensional approaches has to be carried out.

This paper aims to present the theory of multi-dimensional non-conservative Godunov-type schemes with extensive details. These multi-dimensional schemes are based on a one-dimensional framework using non-conservative 1D schemes. These non-conservative one-dimensional schemes generalize those developed in [40] for which the numerical fluxes are conservative in the classical sense, i.e. they are equal on both sides of a face. Here an extension to conservative numerical fluxes in a non-classical sense is done. The multi-dimensional schemes rely on this notion. The construction of the scheme will be carried out, to obtain numerical fluxes based on the Riemann solver. Stability conditions, conservation, and entropy inequality will be presented, as the key point of equilibrium solutions. Those results will be applied to solve the five-equation system developed concurrently by Allaire et al. [3] and by Massoni et al. [53] with surface tension effects [64] modeled by the CSF formulation (Continuum Surface Force) [13]. The non-conservative expression of the CSF formulation makes it difficult to take into account numerically. However, this model is widely used in the literature and has been the subject of the development of numerous numerical methods. Périgaud et al. [64] used a Godunov-type scheme based on the exact Riemann solver including surface tension. Later, Braconnier et al. [14] proposed an approximate Riemann solver with surface tension, based on a relaxation scheme. Nguyen et al. [54] solved the five-equation model with a scheme based on a balanced Osher-type Riemann solver. Then, the HLLC [73] scheme was extended to take into account the surface tension force by Garrick et al. [43]. In the context of the Level-Set method, Zou et al. [77,78] used the Lagrange-Transport splitting strategy of Chalons et al. [20] where the fluxes at the interface contain the surface tension force.

In this work, the Lagrange-Transport splitting method [20,21] is used. Chalons' approach consists of using an operator decomposition that allows two subsystems to be solved successively in two distinct steps. The first is called **Lagrange step** and consists in solving the first subsystem in Lagrangian form while the second is called **Transport step** and corresponds to the transport of the conservative quantities at material velocity. This approach is very similar to the well-known Lagrange-Remap method [10,33], even if there is no remap involved in the splitting strategy. Initially used for the Euler equations [20], two-phase flows with four equations [21] and shallow-water equations with bottom [22], Peluchon et al. [58] extended this method to the five-equation system, then added the effects of viscous dissipation and thermal dissipation [59]. More recently, this approach has been applied to the Baer & Nunziato barotropic model [2]. The interest of this method lies in the possibility of solving the **Lagrange step** by an implicit scheme. Indeed, the targeted applications involve very large density, pressure, sound velocity, or viscosity ratios, which can be extremely restrictive in terms of time step. A semi-implicit approach allows the global scheme to be constrained only by the material velocity since all physics is solved in the implicit Lagrange step. This approach has proved to be very robust, especially for solving viscous and thermal boundary layer flows with a refined near-wall mesh, with a time step based on the material velocity. Here, we will add surface tension effects in the CSF form into the implicit Lagrange step. The extension of surface tension in the Lagrangian framework has already been studied in [26], but to date has never been realized in the Eulerian framework.

It is known that Godunov-type schemes require special treatment in the low-Mach regime [45,44,66,31,32]. Indeed, when the Mach number tends towards 0, the discrete solution of the scheme does not converge towards the asymptotic limit of the solved model. However, the targeted applications require an all-speed numerical scheme, which is accurate for supersonic or hypersonic flows as well as for quasi-incompressible flows.

After presenting the theory of multi-dimensional non-conservative Godunov-type schemes, we will apply these results to the five-equation system with surface tension in CSF form. Finally, we will study schemes correction to correctly capture the incompressible limit as the Mach number tends towards 0. Finally, some test cases will be presented to validate the method.

# 2. Non-conservative Godunov-type scheme

Let  $d \in \mathbb{N}$  be the dimension of the physical space and  $\Omega$  an open of  $\mathbb{R}^d$ . Let m be the size of the system,  $U(\mathbf{x},t) \in \mathbb{R}^m$  the vector of conservative variables with  $(\mathbf{x},t) \in \Omega \times \mathbb{R}^+$ , and F of size  $m \times d$  the flux tensor whose columns are regular functions  $(f_1,...,f_d)$  such that  $\forall i = 1, d, f_i : \Omega \to \mathbb{R}^m$ . Non-conservative hyperbolic systems are given by

$$\partial_t U + \nabla \cdot F(U) + P(U)\nabla \cdot B(U) = 0$$
, in  $\Omega \times \mathbb{R}^+$ , (1)

where B a matrix of size  $m \times d$ , and P, a matrix of size  $m \times m$ , constitute the non-conservative differential term. Results on entropy will be omitted in this paper. Propositions and proofs, analogous to the system studied, can be found in [42,70,30].

Since multi-dimensional schemes are based on a one-dimensional framework using non-conservative 1D schemes, we consider first the one-dimensional case.

#### 2.1. One-dimensional Godunov-type scheme

Let  $\Omega$  be an open space of  $\mathbb{R}$ . This space is discretized, into N cells  $([x_{i-\frac{1}{2}},x_{i+\frac{1}{2}}])_{i\in\mathbb{Z}}$  of size  $\Delta x_i=x_{i+\frac{1}{2}}-x_{i-\frac{1}{2}}$ . For the sake of simplicity, constant cell size  $\Delta x$  is assumed in the following. Time is discretized by  $t^n=n\Delta t$  for  $n\in\mathbb{N}$ , where  $\Delta t>0$  is the time step.

![](_page_2_Figure_2.jpeg)

Fig. 1. Illustration of a Godunov-type scheme based on a Riemann solver.

In the one-dimensional case, system (1) is given by

$$\partial_t U + \partial_x F(U) + P(U) \partial_x B(U) = 0,$$
 (2)

with U, F and B vectors of  $\mathbb{R}^m$  and P a matrix  $m \times m$ . The construction of numerical schemes for non-conservative systems of the form (2) remains an open problem, especially when the quantities P(U) and B(U) can be discontinuous at the same time. The class of Path-Conservative schemes introduced by Parès [56] and Castro [16] is based on the work of Dal Maso et al. [28]. These schemes are an extension of Toumi's work on the weak formulation of the Roe solver [74]. It is shown in [18] that convergence of such schemes to the weak solution can only be obtained if the numerical diffusion is zero. This result is illustrated in [1], where even with the correct path, the scheme written in non-conservative form does not converge. The idea of having a scheme without numerical diffusion was taken up again in [19], in order to construct a Path-Conservative scheme based on Glimm's random-drawing method, applied to gas dynamics in the Lagrangian framework. In this work, we will only deal with configurations where such theories and schemes are not necessary, i.e. when the definition of the non-conservative product is clearly defined. Examples include Baer & Nunziato sevenequation model [6], Kapila's five-equation system, [48], the shallow-water equations with bottom [68], Magnetohydrodynamics and Powell's system [62], for which the quantities P(U) and B(U) are not discontinuous at the same time.

A Godunov-type scheme is based on the solution of the following Riemann problem

$$\partial_t \mathbf{U} + \partial_x \mathbf{F}(\mathbf{U}) + \mathbf{P}(\mathbf{U}) \partial_x \mathbf{B}(\mathbf{U}) = 0, \ \mathbf{U}(x,0) = \mathbf{U}^0(x) = \begin{cases} \mathbf{U}_I & \text{if } x < 0, \\ \mathbf{U}_r & \text{if } 0 \le x. \end{cases}$$
(3)

Let  $W(x/t; U_t, U_r)$  be an approximation of the Riemann problem defined by the system (3) at each interface between the left and right states  $U_1, U_r$ . The self-similar function W is such that

- W (x/t; U<sub>1</sub>, U<sub>r</sub>) = U<sub>1</sub> for -x/t large enough,
   W (x/t; U<sub>1</sub>, U<sub>r</sub>) = U<sub>r</sub> for x/t large enough,

In addition, the solver must be consistent in the sense that W(x/t; U, U) = U. On a face  $i + \frac{1}{2}$ , we regularly use the notation

$$\boldsymbol{W}_{i+\frac{1}{2}} = \boldsymbol{W}\left(\frac{x - x_{i+\frac{1}{2}}}{\Delta t}; \boldsymbol{U}_i, \boldsymbol{U}_{i+1}\right).$$

We will now set out some general definitions.

**Definition 1** (Godunov-type scheme [46]). The Riemann solver  $W_{i+\frac{1}{2}}$  induces a Godunov-type scheme given by

$$U_i^{n+1} = \frac{1}{2} \left( U_{i-\frac{1}{2}}^+ + U_{i+\frac{1}{2}}^- \right),\tag{4}$$

with

$$\boldsymbol{U}_{i-\frac{1}{2}}^{+} = \frac{2}{\Delta x} \int_{x_{i-\frac{1}{2}}}^{x_{i}} \boldsymbol{W} \left( \frac{x - x_{i-\frac{1}{2}}}{\Delta t}; \boldsymbol{U}_{i-1}, \boldsymbol{U}_{i} \right) dx, \ \boldsymbol{U}_{i+\frac{1}{2}}^{-} = \frac{2}{\Delta x} \int_{x_{i}}^{x_{i+\frac{1}{2}}} \boldsymbol{W} \left( \frac{x - x_{i+\frac{1}{2}}}{\Delta t}; \boldsymbol{U}_{i}, \boldsymbol{U}_{i+1} \right) dx.$$
 (5)

According to Definition 1, the solution is obtained by averaging the Riemann solvers on each face over the considered cell (see Fig. 1). The solution is updated in each cell by solving a Riemann problem at each interface. The Riemann solver can be either exact or approximate. In the finite volume paradigm, this form in itself is not easy to use in practice. It is preferable to define numerical fluxes based on the Riemann solver.

**Definition 2** (*Numerical fluxes associated with the Riemann solver*). Let  $\overline{F}_{i-\frac{1}{2}}^+$  and  $\overline{F}_{i+\frac{1}{2}}^-$  be the numerical fluxes associated with the Riemann solver on the  $i-\frac{1}{2}$  and  $i+\frac{1}{2}$  faces of the i cell. By integrating (2) over their respective space-time domain, these numerical fluxes are given by

$$\overline{F}_{i-\frac{1}{2}}^{+} = F(U_i) + \frac{1}{\Delta t} \int_{x_{i-\frac{1}{2}}}^{x_i} \left( W_{i-\frac{1}{2}} - U_i \right) dx + \left( \overline{P\Delta B} \right)_{i-\frac{1}{2}}^{+},$$

$$\overline{F}_{i+\frac{1}{2}}^{-} = F(U_i) - \frac{1}{\Delta t} \int_{x_i}^{x_{i+\frac{1}{2}}} \left( W_{i+\frac{1}{2}} - U_i \right) dx - \left( \overline{P\Delta B} \right)_{i+\frac{1}{2}}^{-},$$
(6)

where  $\left(\overline{P\Delta B}\right)_{i-\frac{1}{2}}^+$  and  $\left(\overline{P\Delta B}\right)_{i+\frac{1}{2}}^-$  are two approximations of the non-conservative term satisfying

$$\left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{i-\frac{1}{2}}^{+} \simeq \int_{x_{i-\frac{1}{2}}}^{x_{i}} \boldsymbol{P}(\boldsymbol{U}) \partial_{x} \boldsymbol{B}(\boldsymbol{U}) \mathrm{d}x \text{ and } \left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{i+\frac{1}{2}}^{-} \simeq \int_{x_{i}}^{x_{i+\frac{1}{2}}} \boldsymbol{P}(\boldsymbol{U}) \partial_{x} \boldsymbol{B}(\boldsymbol{U}) \mathrm{d}x.$$

This definition of fluxes is identical to [42] without a non-conservative product and similar to [30] in the case of a source term. We adopt the notation  $F_i = F(U_i)$  in the following.

Using numerical fluxes definition (6), one can then rewrite the Godunov-type scheme (4) in the following non-conservative form

$$\boldsymbol{U}_{i}^{n+1} = \boldsymbol{U}_{i}^{n} - \frac{\Delta t}{\Delta x} \left( \overline{\boldsymbol{F}}_{i+\frac{1}{2}}^{-} - \overline{\boldsymbol{F}}_{i-\frac{1}{2}}^{+} \right) - \frac{\Delta t}{\Delta x} \left( \left( \overline{\boldsymbol{P} \Delta \boldsymbol{B}} \right)_{i+\frac{1}{2}}^{-} + \left( \overline{\boldsymbol{P} \Delta \boldsymbol{B}} \right)_{i-\frac{1}{2}}^{+} \right).$$

The numerical fluxes (6) are non-conservative in the sense that  $\overline{F}_{i+\frac{1}{2}}^-$  can be different from  $\overline{F}_{i+\frac{1}{2}}^+$  unlike in the classical conservative case. According to the definition of fluxes (6), we have the relation

$$\overline{F}_{i+\frac{1}{2}}^{+} - \overline{F}_{i+\frac{1}{2}}^{-} = F_{i+1} - F_{i} + \overline{P\Delta B}_{i+\frac{1}{2}} + \frac{1}{\Delta t} \int_{x_{i}}^{x_{i+1}} \left( \boldsymbol{W}_{i+\frac{1}{2}} - \boldsymbol{U}^{0} \right) dx, \tag{7}$$

with the notations  $\overline{P\Delta B}_{i+\frac{1}{2}} = (\overline{P\Delta B})_{i+\frac{1}{2}}^- + (\overline{P\Delta B})_{i+\frac{1}{2}}^+$  and

$$\boldsymbol{U}^0 = \left\{ \begin{array}{ll} \boldsymbol{U}_i & \text{if} \quad \boldsymbol{x} < \boldsymbol{x}_{i+\frac{1}{2}}, \\ \boldsymbol{U}_{i+1} & \text{if} \quad \boldsymbol{x} \geq \boldsymbol{x}_{i+\frac{1}{2}}. \end{array} \right.$$

Since the form of the numerical fluxes  $\overline{F}_{i+\frac{1}{2}}^{\pm}$  have been formally defined, it remains to define the approximations of the non-conservative products

$$\left(\overline{P\Delta B}\right)_{i+\frac{1}{2}}^{+}$$
 and  $\left(\overline{P\Delta B}\right)_{i+\frac{1}{2}}^{-}$ . (8)

Several choices are possible. The conservative case will enable us to give a good definition. Indeed, in the degenerate case where  $P = I_d$  the identity matrix, it is enough to define

$$\overline{\Delta \boldsymbol{B}}_{i\pm\frac{1}{2}}^{\mp}=\pm(\overline{\boldsymbol{B}}_{i\pm\frac{1}{2}}^{\mp}-\boldsymbol{B}_{i}),$$

with  $\overline{B}_{i+\frac{1}{2}}^{\mp} = \overline{B}_{i+\frac{1}{2}}^{\mp} \left( U_i, U_{i+1} \right)$  a consistent approximation depending on left and right states, and the notation  $B_i = B \left( U_i \right)$ , so that the definition of fluxes is consistent with system (2). In the case where  $P \neq I_d$ , we will see in the following that the consistency relation requires us to define the term  $\overline{P\Delta B}_{i+\frac{1}{2}}$  which corresponds to the approximation of the non-conservative product on the  $i+\frac{1}{2}$  face. We assume the existence of a matrix  $\overline{P}$  such that  $\overline{P}(U,U) = P(U)$ , and such that by construction

$$\overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i+\frac{1}{2}} \simeq \int_{y_{i}}^{x_{i+1}} \boldsymbol{P}(\boldsymbol{U}) \partial_{x} \boldsymbol{B}(\boldsymbol{U}) dx \simeq \overline{\boldsymbol{P}}_{i+\frac{1}{2}} \left( \boldsymbol{U}_{i}, \boldsymbol{U}_{i+1} \right) \left( \boldsymbol{B}_{i+1} - \boldsymbol{B}_{i} \right). \tag{9}$$

It is also possible to define non-conservative terms (8) as follows

L. Tallois, S. Peluchon, G. Gallice et al.

$$\overline{{\textbf P}\Delta {\textbf B}}_{i+\frac{1}{2}}^+ = \overline{{\textbf P}}_{i+\frac{1}{2}}^+ \left( {\textbf B}_{i+1} - \overline{{\textbf B}}_{i+\frac{1}{2}}^+ \right) \ \ \text{and} \ \ \overline{{\textbf P}\Delta {\textbf B}}_{i+\frac{1}{2}}^- = \overline{{\textbf P}}_{i+\frac{1}{2}}^- \left( \overline{{\textbf B}}_{i+\frac{1}{2}}^- - {\textbf B}_i \right),$$

with  $\overline{P}_{i+\frac{1}{2}}^{\pm} = \overline{P}_{i+\frac{1}{2}}^{\pm} \left( U_i, U_{i+1} \right)$ . By choosing a centered discretization of  $\overline{B}_{i+\frac{1}{2}}^{\pm}$ , such that  $\overline{B}_{i+\frac{1}{2}}^{\pm} = \left( B_{i+1} + B_i \right)/2$ , we then have

$$\overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i+\frac{1}{2}} = \frac{1}{2} \left( \overline{\boldsymbol{P}}_{i+\frac{1}{2}}^+ + \overline{\boldsymbol{P}}_{i+\frac{1}{2}}^- \right) \left( \boldsymbol{B}_{i+1} - \boldsymbol{B}_i \right).$$

The relation with the definition (9) being obvious, we have

$$\overline{P}_{i+\frac{1}{2}} = \frac{1}{2} \left( \overline{P}_{i+\frac{1}{2}}^+ + \overline{P}_{i+\frac{1}{2}}^- \right). \tag{10}$$

**Remark 1.** The discretization of this term will be discussed later, concerning the preservation of equilibrium solutions. We therefore have a first-order approximation of the non-conservative term. To obtain a second-order scheme for the non-conservative term, it is necessary to introduce a centered approximation of  $P\partial_x B$  in each cell (see [12] for example). This point will be dealt with later.

## 2.1.1. Stability

The stability of the numerical scheme is guaranteed if the solution remains in the set of definition domain, which is assumed convex here. Let  $\xi_{i-\frac{1}{2}}^{\max}$  and  $\xi_{i+\frac{1}{2}}^{\min}$  two positive velocities such that

$$\bullet \ \forall \xi \geq \xi_{i-\frac{1}{2}}^{\max}, \ \boldsymbol{W}_{i-\frac{1}{2}} - \boldsymbol{U}_{i} = 0,$$

• 
$$\forall \xi \leq -\xi_{i+\frac{1}{2}}^{\min}$$
,  $\boldsymbol{W}_{i+\frac{1}{2}}^{2} - \boldsymbol{U}_{i} = 0$ .

The scheme defined by (4) and (5) is stable under the condition [42,70,30]

$$\forall i, \frac{1}{\Delta t} \ge \frac{\xi_{i-\frac{1}{2}}^{\max} + \xi_{i+\frac{1}{2}}^{\min}}{\Delta x}.$$

$$(11)$$

#### 2.1.2. Conservation

In the framework of classical Godunov-type schemes, conservation is obtained in the classical way by equality on each interface of the fluxes given by (6) i.e.  $\overline{F}_{i+\frac{1}{2}}^+ = \overline{F}_{i+\frac{1}{2}}^-$  which is not yet the case here. The definition of consistency with the integral form, similar to [47], will allow us to state a proposition about flux conservation. Even if the system under study is non-conservative, we speak of a conservative flux in the sense that, without a non-conservative product, the fluxes are equal on either side of the interface.

**Definition 3** (Consistency with integral form [39]). The Riemann solver  $W_{i+\frac{1}{2}}$  is consistent with the integral form of (2) if and only if there exists an approximation of the non-conservative term  $\overline{P\Delta B}_{i+\frac{1}{2}}$  such that

$$\frac{1}{\Delta t} \int_{\mathbf{x}}^{x_{i+1}} \left( \mathbf{W} \left( \frac{\mathbf{x} - \mathbf{x}_{i+\frac{1}{2}}}{\Delta t}; \mathbf{U}_i, \mathbf{U}_{i+1} \right) - \mathbf{U}^0 \right) d\mathbf{x} + \mathbf{F}_{i+1} - \mathbf{F}_i + \overline{\mathbf{P} \Delta \mathbf{B}}_{i+\frac{1}{2}} = 0.$$
 (12)

In this case, (4) is called a conservative Godunov-type scheme.

**Proposition 1** ([39]). The Godunov-type scheme induced by the Riemann solver  $W_{i+\frac{1}{2}}$  associated with the approximation of the non-conservative term  $P\Delta B_{i+\frac{1}{2}}$  is consistent with the integral form if and only if

$$\overline{\boldsymbol{F}}_{i+\frac{1}{2}}^{+} = \overline{\boldsymbol{F}}_{i+\frac{1}{2}}^{-}.$$

**Proof.** According to the equation (7), the relation (12) is true if and only if  $\overline{F}_{i+\frac{1}{2}}^+ = \overline{F}_{i+\frac{1}{2}}^-$ .

In the same way as for the conservation, the entropy fluxes and the consistency with the integral form of the entropy inequality can be defined for Godunov-type schemes in the case of non-conservative systems. For sake of readability, this part is detailed in [70].

# 2.1.3. Equilibrium solutions

Schemes that preserve stationary solutions are said to be well-balanced. First, we recall the notion of equilibrium solutions given by Gallice [39].

![](_page_5_Picture_2.jpeg)

**Fig. 2.** Notations associated with the 2D mesh concerning the face f (a) and the node n (b).

**Definition 4** (Equilibrium and strong equilibrium solutions). We call **equilibrium solution** a sequence  $(U_i)_i$  satisfying at each interface

$$\boldsymbol{U}_{i+\frac{1}{2}}^{+} = \boldsymbol{U}_{i+1}, \ \boldsymbol{U}_{i+\frac{1}{2}}^{-} = \boldsymbol{U}_{i+1}.$$

We call **strong equilibrium solution** a sequence  $(U_i)_i$  satisfying

$$\left\{ \begin{array}{ll} \boldsymbol{W}\left(x/t;\boldsymbol{U}_{i},\boldsymbol{U}_{i+1}\right) = \boldsymbol{U}_{i} & \text{if} \quad x/t \leq 0, \\ \boldsymbol{W}\left(x/t;\boldsymbol{U}_{i},\boldsymbol{U}_{i+1}\right) = \boldsymbol{U}_{i+1} & \text{if} \quad x/t > 0. \end{array} \right.$$

A strong equilibrium solution is an equilibrium solution. We have the following proposition.

**Proposition 2** (Consistency relation of an equilibrium solution [39]). Equilibrium and strong equilibrium solution verify the consistency relation

$$\boldsymbol{F}_{i+1} - \boldsymbol{F}_i + \overline{\boldsymbol{P}} \Delta \boldsymbol{B}_{i+\frac{1}{2}} = 0.$$

**Proof.** Recall that the consistency relation (12) is given by

$$\frac{1}{\Delta t} \int_{\mathbf{x}}^{x_{i+1}} \left( \mathbf{W} \left( \frac{\mathbf{x} - \mathbf{x}_{i+\frac{1}{2}}}{\Delta t}; \mathbf{U}_i, \mathbf{U}_{i+1} \right) - \mathbf{U}^0 \right) d\mathbf{x} + \mathbf{F}_{i+1} - \mathbf{F}_i + \overline{\mathbf{P}} \Delta \overline{\mathbf{B}}_{i+\frac{1}{2}} = 0.$$

If we consider a strong equilibrium solution, the integral disappears, and the result is obtained.  $\Box$ 

This result shows that equilibrium solutions verify equilibrium at the faces. Having a well-balanced scheme that verifies equilibrium and/or a strong equilibrium solution is far more powerful than simply verifying equilibrium at the cell level. It is easy to check that such a scheme preserves stationary states

$$\forall i, \ \boldsymbol{U}_{i}^{n+1} = \boldsymbol{U}_{i}^{n}.$$

This question was addressed in [12].

#### 2.2. Multi-dimensional Godunov-type scheme

This section aims to generalize the previous results to higher dimensions. Here, we will restrict ourselves to dimension 2. Let us first consider an open domain  $\Omega \in \mathbb{R}^2$  discretized into N cells  $\Omega_c$  such that  $\mathcal{V}(c)$  is the set of cells neighboring the cell  $\Omega_c$  by faces.  $|\Omega_c|$  is the surface of the cell  $\Omega_c$ . The centroid  $\mathbf{x}_c$  of the cell  $\Omega_c$  is defined as follows

$$\mathbf{x}_c = \frac{1}{|\Omega_c|} \int_{\Omega} \mathbf{x} d\mathbf{x}.$$

Let us note  $\mathcal{N}(c)$  the set of nodes of the cell  $\Omega_c$ . The length of the face f common to the cells c and d is  $|\Gamma_f| = |\Gamma_{cd}|$ , and  $|\Gamma_{nf}| = |\Gamma_{ncd}| = |\Gamma_{ncd}| = |\Gamma_{cd}|/2$  is the half-length attached to the node n, and  $n_f = n_{cd}$  is the unit vector normal to  $\Gamma_{cd}$  such that  $n_{cd} = -n_{dc}$  (see Fig. 2). Then,  $\mathcal{SF}(nc)$  is the set of  $\Omega_c$  faces attached to the n node. In two dimensions, there are always only two. We sometimes use the notation  $\mathcal{SF}(n)$  to define the set of faces attached to the n node. In the case of a two-dimensional structured mesh, this set is made up of four faces. Finally, we denote  $\mathbf{x}_{n^+}$  and  $\mathbf{x}_{n^-}$  the nodes that form the face f.

Time is always discretized by  $t^n = n\Delta t$  for  $n \in \mathbb{N}$ , where  $\Delta t > 0$  is the time step.

A direct extension of classical Godunov-type schemes to the multi-dimensional case consists of considering the Riemann solver Win the direction associated with the face normal. Thus,  $\mathbf{W}_{cf} = \mathbf{W}_{cf} \left( \xi; \mathbf{U}_t, \mathbf{U}_r, \mathbf{n}_f \right)$  where  $\xi = (\mathbf{x} - \mathbf{x}_f) \cdot \mathbf{n}_f / t$  is a self-similar variable.  $W_{c,f}$  is the exact or approximate solution of the Riemann problem on the f face of the c cell.

$$\begin{cases} \partial_t \boldsymbol{U} + \nabla \cdot \boldsymbol{F}(\boldsymbol{U}) + \boldsymbol{P}(\boldsymbol{U}) \nabla \cdot \boldsymbol{B}(\boldsymbol{U}) = 0, \\ \boldsymbol{U}(\mathbf{x} \cdot \boldsymbol{n}_f, 0) = \boldsymbol{U}^0(\mathbf{x} \cdot \boldsymbol{n}_f) = \begin{cases} \boldsymbol{U}_c = \boldsymbol{U}_l & \text{if } \mathbf{x} \cdot \boldsymbol{n}_f < 0, \\ \boldsymbol{U}_d = \boldsymbol{U}_r & \text{if } 0 \leq \mathbf{x} \cdot \boldsymbol{n}_f. \end{cases}$$

However, it is also possible to associate a nodal dependence with the Riemann solver, in the manner of the Lagrangian schemes GLACE [35] or EUCCLHYD [52]. This formulation has been newly introduced in [42]. It implies that the Riemann solver becomes  $\boldsymbol{W}_{ncf} = \boldsymbol{W}_{ncf} \left( \xi; \boldsymbol{U}_c, \boldsymbol{U}_d, \mathcal{P}_n, \boldsymbol{n}_f \right)$  where  $\mathcal{P}_n$  is a nodal parameter associated with n. In [35,52,42],  $\mathcal{P}_n$  corresponds to the velocity of the node. In a Lagrangian framework, it seems natural to define this notion of nodal velocity as the mesh moves. From an Eulerian perspective, the need to introduce this nodal parameter seems less obvious. Its utility has been demonstrated in [42]. However, we can imagine other choices for the nodal parameter, such as the pressure at the node. This choice was operated in a recent work [9] for the acoustic system and in [29] for Euler system. The self-similar function  $\boldsymbol{W}_{ncf}$  is always such that

- $W_{ncf}(\xi; U_c, U_d, \mathcal{P}_n, n_f) = U_c$  for  $-\xi$  large enough,
- $\boldsymbol{W}_{ncf}\left(\xi;\boldsymbol{U}_{c},\boldsymbol{U}_{d},\mathcal{P}_{n},\boldsymbol{n}_{f}\right)=\boldsymbol{U}_{d}$  for  $\xi$  large enough,  $\boldsymbol{W}_{ncf}\left(\xi;\boldsymbol{U},\boldsymbol{U},\mathcal{P}_{n},\boldsymbol{n}_{f}\right)=\boldsymbol{U}$ .

Furthermore, the solver is assumed to be symmetrical in the sense that

$$\boldsymbol{W}_{ncf}\left(\xi;\boldsymbol{U}_{c},\boldsymbol{U}_{d},\mathcal{P}_{n},\boldsymbol{n}_{f}\right)=\boldsymbol{W}_{ndf}\left(-\xi;\boldsymbol{U}_{d},\boldsymbol{U}_{c},\mathcal{P}_{n},\boldsymbol{n}_{f}\right).$$

The normal  $n_f$  acts as follows

$$\boldsymbol{W}_{ncf}\left(\boldsymbol{\xi};\boldsymbol{U}_{c},\boldsymbol{U}_{d},\boldsymbol{\mathcal{P}}_{n},-\boldsymbol{n}_{f}\right)=\boldsymbol{W}_{ncf}\left(\boldsymbol{\xi};\boldsymbol{U}_{c},\boldsymbol{U}_{d},\boldsymbol{\mathcal{P}}_{n},\boldsymbol{n}_{f}\right).$$

Now, by analogy with the one-dimensional case, we will set out the definition of a multi-dimensional Godunov-type scheme.

**Definition 5** (Multi-dimensional Godunov-type scheme). Let  $(\alpha_{ncf})_{ncf}$  be a sequence of positive reals associated with cell c such that

$$\sum_{n \in \mathcal{N}(c)} \sum_{f \in \mathcal{SF}(nc)} \alpha_{ncf} = 1.$$

The Riemann solver  $W_{ncf}$  induces a multi-dimensional Godunov scheme such that

$$\boldsymbol{U}_{c}^{n+1} = \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} \alpha_{ncf} \boldsymbol{U}_{ncf}, \tag{13}$$

with

$$\boldsymbol{U}_{ncf} = \boldsymbol{U}_{c} + \frac{|\Gamma_{ncf}|\Delta t}{\alpha_{ncf}|\Omega_{c}|} \int_{-\infty}^{0} \left(\boldsymbol{W}_{ncf}\left(\xi; \boldsymbol{U}_{c}, \boldsymbol{U}_{d}, \boldsymbol{\mathcal{P}}_{n}, \boldsymbol{n}_{f}\right) - \boldsymbol{U}_{c}\right) d\xi.$$

$$(14)$$

Generally, we introduce the perimeter  $|\partial \Gamma_c|$  of the cell c and define the sequence  $(\alpha_{ncf})_{ncf}$  such that

$$\alpha_{ncf} = \frac{\left|\Gamma_{ncf}\right|}{\left|\partial\Gamma_{c}\right|}, \ \left|\partial\Gamma_{c}\right| = \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} \left|\Gamma_{ncf}\right|.$$

A Godunov-type scheme based on a Riemann  $W_{ncf}$  solver requires the introduction of an associated numerical flux, to write the scheme in the finite volume formalism.

**Definition 6** (Flux associated with the multi-dimensional Riemann solver). Let  $\overline{F}_{ncf} = \overline{F}_{ncf} (U_c, U_d, \mathcal{P}_n, \mathbf{n}_f)$  be the numerical flux associated with the multi-dimensional Riemann solver). ciated with the Riemann solver  $\boldsymbol{W}_{ncf}$ . By spatio-temporal integration, this flux is given by

$$\overline{\boldsymbol{F}}_{ncf} = \boldsymbol{F}_c \cdot \boldsymbol{n}_f - \int_{-\infty}^{0} \left( \boldsymbol{W}_{ncf} \left( \xi; \boldsymbol{U}_c, \boldsymbol{U}_d, \boldsymbol{P}_n, \boldsymbol{n}_f \right) - \boldsymbol{U}_c \right) d\xi - \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{ncf} \cdot \boldsymbol{n}_f, \tag{15}$$

where  $\left(\overline{P\Delta B}\right)_{ncf} \cdot n_f$  is an approximation of the non-conservative term such that

![](_page_7_Picture_2.jpeg)

Fig. 3. For a 2D mesh, schematic representation of fluxes evaluated on a face f (a) and fluxes evaluated around a node n (b).

$$\Delta t \int_{0}^{0} P(U) \partial_{n} B(U) d\xi \simeq \left( \overline{P \Delta B} \right)_{ncf} \cdot n_{f}.$$
(16)

An abuse of notation is made when integrating (16). Indeed, the integral is not really on an infinite domain but rather on the support of  $\boldsymbol{W}_{ncf}\left(\boldsymbol{\xi};\boldsymbol{U}_{c},\boldsymbol{U}_{d},\mathcal{P}_{n},\boldsymbol{n}_{f}\right)-\boldsymbol{U}_{c}$ . For simplicity's sake, we will keep this notation.

As in the one-dimensional case, using numerical fluxes definition (15) and noting that

$$\sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \boldsymbol{n}_f = 0,$$

one can rewrite the multi-dimensional Godunov-type scheme (13) as

$$\boldsymbol{U}_{c}^{n+1} = \boldsymbol{U}_{c}^{n} - \frac{\Delta t}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left( \overline{\boldsymbol{F}}_{ncf} + \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{ncf} \cdot \boldsymbol{n}_{f} \right). \tag{17}$$

The definition of the Riemann solver with a nodal parameter induces a *Cell-Centered* finite volume type scheme where fluxes are evaluated around the nodes (see Fig. 3). Thus, in our 2D configuration, on the same face, there are two numerical fluxes on each side of the face (see Fig. 3.a). In addition, around a node attached to a cell, there are two distinct fluxes (see Fig. 3.b) for the same face.

**Remark 2.** For the non-conservative product, the same precepts apply as in the one-dimensional case. At first order in space, we will assume the existence of a matrix  $\overline{P}$  consistent with P. In the multi-dimensional context, this matrix  $\overline{P}$  can be associated with a face, or with a node. In addition, there is no reason why it should not also depend on the nodal parameter  $\mathcal{P}_n$ . The influence of this choice will be discussed in the section on equilibrium solutions. The second-order case will also be dealt with later.

One can also introduce the flux of the cell d attached to the node n on the face f denoted by  $\overline{F}_{ndf} = \overline{F}_{ndf} (U_d, U_c, \mathcal{P}_n, -n_f)$ . This flux is given by

$$\overline{\boldsymbol{F}}_{ndf} = \boldsymbol{F}_{d} \cdot (-\boldsymbol{n}_{f}) - \int_{-\infty}^{0} \left( \boldsymbol{W}_{ndf} \left( \xi; \boldsymbol{U}_{d}, \boldsymbol{U}_{c}, \boldsymbol{\mathcal{P}}_{n}, -\boldsymbol{n}_{f} \right) - \boldsymbol{U}_{d} \right) d\xi + \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{ndf} \cdot (-\boldsymbol{n}_{f}). \tag{18}$$

As in the one-dimensional case, let  $\overline{F}_{nf}^{\pm}$  be the fluxes on the face f and associated with the node n such that  $\overline{F}_{nf} = \overline{F}_{ncf}$  and  $\overline{F}_{nf}^{+} = \overline{F}_{ndf}$ . We do not consider a priori the equality of the fluxes in the sense that  $\overline{F}_{nf}^{-}$  is different from  $\overline{F}_{nf}^{+}$ . Indeed, we have the relation

$$\overline{F}_{nf}^{+} - \overline{F}_{nf}^{-} = \left[ \left( F_c - F_d + \overline{P \Delta B}_{nf} \right) \cdot n_f + \int_{-\infty}^{\infty} \left( W_{ncf}(\xi) - U^0(\xi) d\xi \right) \right],$$

with the notations  $\boldsymbol{W}_{ncf}(\xi) = \boldsymbol{W}_{ncf}(\xi; \boldsymbol{U}_c, \boldsymbol{U}_d, \mathcal{P}_n, \boldsymbol{n}_f)$  and

$$\overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{nf} = \left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{ncf} + \left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{ndf}.$$

# 2.2.1. Stability

We will now state the stability condition for the scheme defined by (13) and (14). Let  $\xi_{ncf}^{\min}$  be a positive real such that  $\forall \xi \leq -\xi_{ncf}^{\min}$ ,  $\boldsymbol{W}_{ncf}(\xi) - \boldsymbol{U}_c = 0$ . The scheme defined by (13) and (14) is stable under the following condition [42,70,30]

L. Tallois, S. Peluchon, G. Gallice et al.

$$\frac{1}{\Delta t} > \max_{c} \left( \frac{1}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SP(nc)} |\Gamma_{ncf}| \xi_{ncf}^{\min} \right). \tag{19}$$

#### 2.2.2. Conservation

It is necessary to define the meaning of a conservative scheme in the present context, where the system studied is not conservative.

Definition 7 (Conservative scheme). The finite volume scheme (17) for the non-conservative system (1) is conservative if and only if

$$\sum_{c} |\Omega_{c}| U_{c}^{n+1} = \sum_{c} |\Omega_{c}| U_{c}^{n} - \Delta t \sum_{c} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left( \overline{P \Delta B} \right)_{ncf} \cdot \boldsymbol{n}_{f}.$$

This definition is consistent with the definition of conservation without a non-conservative term.

The Multi-dimensional Godunov-type scheme can be conservative in an unclassical sense, i.e.  $\overline{F}_{ncf} \neq -\overline{F}_{ndf}$ . This idea was used to develop schemes for Lagrangian hydrodynamics in [35,52], and formalized in defining a new class of Godunov-type schemes in [42]. We will first state the unclassical consistency of the Riemann solver with the integral form and show that the flux associated with the Riemann solver  $W_{ncf}$  induces a Multi-dimensional Godunov-type scheme with **unclassical conservative fluxes**.

**Definition 8** (*Unclassical consistency with integral form*). The Riemann solver  $\underline{W}_{ncf}$  is unclassically consistent with the integral form (1) if and only if there exists an approximation to the non-conservative term  $\underline{P}\Delta \underline{B}_{ncf}$  such that

$$\sum_{f \in SF(nc)} |\Gamma_{ncf}| \left[ \int_{-\infty}^{\infty} \left( \boldsymbol{W}_{ncf} \left( \boldsymbol{\xi}; \boldsymbol{U}_{c}, \boldsymbol{U}_{d}, \boldsymbol{\mathcal{P}}_{n}, \boldsymbol{n}_{f} \right) - \boldsymbol{U}^{0}(\boldsymbol{\xi}) \right) d\boldsymbol{\xi} + \left( \boldsymbol{F}_{c} - \boldsymbol{F}_{d} + \overline{\boldsymbol{P}\Delta \boldsymbol{B}}_{nf} \right) \cdot \boldsymbol{n}_{f} \right] = 0.$$
 (20)

In this case, (13) is called unclassical conservative Godunov-type scheme.

**Proposition 3** (Unclassical conservative flux). If the Riemann solver  $W_{ncf}$  is unclassically consistent with the integral form, then the Godunov-type scheme (13) induced by the Riemann solver has unclassical conservative fluxes associated with  $W_{ncf}$  and  $\overline{P\Delta B}_{ncf}$ .

**Proof.** Here, the flux is always non-conservative in the classical sense. According to the definition of a conservative scheme (see Definition 7), it suffices to have the following equality

$$\sum_{c} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \overline{F}_{ncf} = 0,$$

which is written, by manipulating the sums

$$\sum_{n} \sum_{c \in C(n)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \overline{F}_{ncf} = 0.$$

Here, C(n) designates the set of cells attached to the node n. A sufficient condition to obtain a conservative scheme is that the flux verifies the relation

$$\forall n \in \mathcal{N}(c), \ \sum_{c \in C(n)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \overline{F}_{ncf} = 0.$$

Using the flux  $\overline{F}_{nf}^{\pm}$  introduced earlier and defined by (15) and (18), and noting that summing over the cells  $c \in C(n)$  attached to the node n then summing over the faces  $f \in S\mathcal{F}(n)$  of the cell c attached to the node n is equivalent to summing directly over all the  $f \in S\mathcal{F}(n)$  faces attached to the node, the previous equation becomes

$$\forall n \in \mathcal{N}(c), \sum_{f \in SF(n)} |\Gamma_{ncf}| \left( \overline{F}_{nf}^{\dagger} - \overline{F}_{nf} \right) = 0. \tag{21}$$

Here,  $\overline{F}_{nf}^-$  and  $\overline{F}_{nf}^+$  are the fluxes on either side of the face f attached to the node n. Their dependence on the nodal parameter  $\mathcal{P}_n$  means that, a priori, these fluxes are not equal. The relationship (21) expresses the fact that conservation can be achieved if, around each node, the sum of the fluxes on the faces attached to that node is zero. The difference in fluxes in (21) leads to the unclassical conservation relation (20).  $\square$ 

The reciprocal is probably true in our two-dimensional configuration, but difficult to justify. It is easy to see that the definition of consistency with the integral form (see Definition 3) as defined by [47] implies unclassical consistency with the integral form.

As in the one-dimensional case, the extension of unclassical conservation can be done naturally to the definition of an unclassical entropic scheme. Definitions and proposition for entropy in the context of multi-dimensional Godunov-type scheme are given in [42,70,30].

![](_page_9_Figure_2.jpeg)

![](_page_9_Figure_3.jpeg)

Fig. 4. Riemann problem structure (a) when the solution is piecewise constant and (b) when the solution is not piecewise constant.

# 2.2.3. Equilibrium solutions

The extension of the equilibrium solutions from the one-dimensional to the multi-dimensional case is as follows.

**Definition 9** (Equilibrium and strong equilibrium solutions). We call **equilibrium solution** a sequence  $(U_c)_i$  that verifies at each cell c and each interface

$$\boldsymbol{U}_{ncf} = \boldsymbol{U}_{c}$$
.

We call **strong equilibrium solution** a sequence  $(U_c)_i$  that verifies at each interface

$$\left\{ \begin{array}{ll} \boldsymbol{W}_{ncf} \left( \boldsymbol{\xi}; \boldsymbol{U}_{c}, \boldsymbol{U}_{d}, \boldsymbol{\mathcal{P}}_{n}, \boldsymbol{n} \right) = \boldsymbol{U}_{c} & \text{if} \quad \boldsymbol{\xi} \leq 0, \\ \boldsymbol{W}_{ncf} \left( \boldsymbol{\xi}; \boldsymbol{U}_{c}, \boldsymbol{U}_{d}, \boldsymbol{\mathcal{P}}_{n}, \boldsymbol{n} \right) = \boldsymbol{U}_{d} & \text{if} \quad \boldsymbol{\xi} > 0. \end{array} \right.$$

**Proposition 4.** In the multi-dimensional case, equilibrium and strong equilibrium solutions verify the consistency relation

$$\sum_{f \in SF(nc)} |\Gamma_{ncf}| \left( \boldsymbol{F}_c - \boldsymbol{F}_d + \overline{\boldsymbol{P}\Delta \boldsymbol{B}}_{ncf} \right) \cdot \boldsymbol{n}_f = 0.$$
(22)

**Proof.** This result is demonstrated as in the one-dimensional case.

The previous result shows the importance of defining the approximation of the non-conservative term  $\overline{P}$ . If  $\overline{P}$  is defined at the faces, then the equilibrium is verified on each face as in the one-dimensional case. The previous proposition is trivial. However, constructing a matrix  $\overline{P}$  at faces verifying equilibrium at node (22) but not verifying equilibrium on faces seems impossible. It is also possible to define  $\overline{P}$  at the node. Such a choice is not absurd, since the matrix itself can depend on the nodal parameter  $\mathcal{P}_n$ . It thus depends on the nodal parameter and all the cells attached to the node. In this way, equilibrium is truly achieved at the nodes. We then have

$$\overline{P} = \overline{P}_n (..., U_c, U_d, ...; \mathcal{P}_n).$$

Such a choice has no consequence on the previous definitions and results but will enable us to construct different numerical schemes.

#### 2.3. Non-conservative Godunov-type scheme: second-order extension

In the previous section, we introduced the notion of a non-conservative Godunov-type scheme. The previous scheme was constructed by considering the solution to be constant in each cell. Extension to second order requires attention to this consideration (see Remark 1 and Remark 2). We shall see precisely in the one-dimensional framework that the non-conservative product requires dealing with a Riemann problem within the cell itself, to bring out a second-order contribution. It was discuss in [12] for hyperbolic systems with source terms. The extension to the multi-dimensional framework will follow naturally.

# 2.3.1. Second-order extension: one-dimensional case

This time, we consider the problem illustrated on Fig. 4, where the solution is not piecewise constant. Indeed, using a MUSCL-type scheme for example, where fluxes are evaluated by a polynomial reconstruction of the solution within cells, we can consider that within cell i there are two constant states  $U_i^L$  and  $U_i^R$ , calculated from the reconstructed variables such that

$$\boldsymbol{U}_{i}^{L} + \boldsymbol{U}_{i}^{R} = 2\boldsymbol{U}_{i}.$$

This idea was also used in [51]. Thus, we will solve three Riemann problems. According to the definition of a Godunov-type scheme, we have

L. Tallois, S. Peluchon, G. Gallice et al.

$$U_i^{n+1} = \frac{1}{4} \left( U_{i-\frac{1}{2}}^+ + U_i^- + U_i^+ + U_{i+\frac{1}{2}}^- \right), \tag{23}$$

with

$$\begin{split} & \boldsymbol{U}_{i-\frac{1}{2}}^{+} = \frac{4}{\Delta x} \int\limits_{x_{i-\frac{1}{2}}}^{x_{i-\frac{1}{4}}} \boldsymbol{W} \left( \frac{x - x_{i-\frac{1}{2}}}{\Delta t}; \boldsymbol{U}_{i-1}^{R}, \boldsymbol{U}_{i}^{L} \right) \mathrm{d}x, \\ & \boldsymbol{U}_{i}^{-} = \frac{4}{\Delta x} \int\limits_{x_{i-\frac{1}{4}}}^{x_{i}} \boldsymbol{W} \left( \frac{x - x_{i-\frac{1}{4}}}{\Delta t}; \boldsymbol{U}_{i}^{L}, \boldsymbol{U}_{i}^{R} \right) \mathrm{d}x, \\ & \boldsymbol{U}_{i}^{+} = \frac{4}{\Delta x} \int\limits_{x_{i}}^{x_{i+\frac{1}{4}}} \boldsymbol{W} \left( \frac{x - x_{i+\frac{1}{4}}}{\Delta t} \boldsymbol{U}_{i}^{L}, \boldsymbol{U}_{i}^{R} \right) \mathrm{d}x, \\ & \boldsymbol{U}_{i+\frac{1}{2}}^{-} = \frac{4}{\Delta x} \int\limits_{x_{i+\frac{1}{4}}}^{x_{i+\frac{1}{2}}} \boldsymbol{W} \left( \frac{x - x_{i+\frac{1}{4}}}{\Delta t} \boldsymbol{U}_{i}^{L}, \boldsymbol{U}_{i}^{R} \right) \mathrm{d}x. \end{split}$$

According to the definition of fluxes (6), we then directly have the expressions

$$\begin{array}{llll} \boldsymbol{U}_{i-\frac{1}{2}}^{+} & = & \boldsymbol{U}_{i}^{L} & + & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{F}}_{i-\frac{1}{2}}^{+} & - & \boldsymbol{F}_{i-\frac{1}{4}} \right) & - & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}} \right)_{i-\frac{1}{2}}^{+}, \\ \boldsymbol{U}_{i}^{-} & = & \boldsymbol{U}_{i} & + & \frac{4\Delta t}{\Delta x} \left( \boldsymbol{F}_{i-\frac{1}{4}}^{-} & - & \overline{\boldsymbol{F}}_{i}^{-} \right) & - & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}} \right)_{i}^{-}, \\ \boldsymbol{U}_{i}^{+} & = & \boldsymbol{U}_{i} & + & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{F}}_{i}^{+} & - & \boldsymbol{F}_{i+\frac{1}{4}} \right) & - & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}} \right)_{i}^{+}, \\ \boldsymbol{U}_{i+\frac{1}{2}}^{-} & = & \boldsymbol{U}_{i}^{R} & + & \frac{4\Delta t}{\Delta x} \left( \boldsymbol{F}_{i-\frac{1}{4}}^{-} & - & \overline{\boldsymbol{F}}_{i+\frac{1}{2}}^{-} \right) & - & \frac{4\Delta t}{\Delta x} \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}} \right)_{i+\frac{1}{2}}^{-}. \end{array}$$

Here, the consistency relation with the integral form (see Definition 3) is assumed to be verified by the intermediate fluxes  $\overline{F}_i^-$  and  $\overline{F}_i^+$ . They are then equal. Injecting the above relations into the definition of a Godunov-type scheme (23), we obtain

$$\boldsymbol{U}_{i}^{n+1} = \boldsymbol{U}_{i}^{n} - \frac{\Delta t}{\Delta x} \left( \overline{\boldsymbol{F}}_{i+\frac{1}{2}}^{-} - \overline{\boldsymbol{F}}_{i-\frac{1}{2}}^{+} \right) - \frac{\Delta t}{\Delta x} \left( \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{i+\frac{1}{2}}^{-} + \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{i-\frac{1}{2}}^{+} \right) - \frac{\Delta t}{\Delta x} \overline{\boldsymbol{P}} \Delta \boldsymbol{B}_{i},$$

where we see a second-order contribution from the non-conservative product, centered on the cell, and given by

$$\overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i} = \overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i}^{-} + \overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i}^{+}$$

The necessity of having a centered contribution of the non-conservative product for second-order finite volume schemes is not new [12]. This result is consistent with the scheme constructed by Audusse et al. [4,5] for the shallow-water equations with bottom. Here, we have highlighted its appearance in the context of non-conservative Godunov-type schemes. Applying the same approximations as in the first-order case, we define the centered non-conservative products by

$$\overline{P}\Delta \overline{B}_{i}^{-} = \overline{P}_{i}^{-} \left( \overline{B}_{i}^{-} - \overline{B}_{i}^{L} \right) \text{ et } \overline{P}\Delta \overline{B}_{i}^{+} = \overline{P}_{i}^{+} \left( \overline{B}_{i}^{R} - \overline{B}_{i}^{+} \right).$$

The flux  $\overline{B}_{i}^{\pm}$  can be taken centered, i.e.,  $\overline{B}_{i}^{\pm} = B_{i} = (B_{i}^{L} + B_{i}^{R})/2$ . The centered term thus becomes

$$\overline{\boldsymbol{P}\Delta\boldsymbol{B}}_{i} = \frac{1}{2} \left( \overline{\boldsymbol{P}}_{i}^{-} + \overline{\boldsymbol{P}}_{i}^{+} \right) \left( \boldsymbol{B}_{i}^{R} - \boldsymbol{B}_{i}^{L} \right).$$

We then see that, at first order, without reconstruction, this term is zero since  $\mathbf{B}_i^R = \mathbf{B}_i^L = \mathbf{B}_i$ . However, if we also wish to apply a high-order scheme to the non-conservative product, of the MUSCL type for example, it is necessary to take this centered term into account. However, it does not affect the construction of fluxes or the notion of consistency. Finally, the discretization of terms  $\overline{P}_i$  and  $\overline{P}_i^+$  can be made identically to the first-order one, taking care to be consistent with P and checking that the scheme guarantees equilibrium solutions.

# 2.3.2. Second-order extension: multi-dimensional case

By analogy with the one-dimensional case, we can show that the non-conservative multi-dimensional Godunov-type scheme of order two also has a centered second term of the non-conservative product approximation. The second-order numerical flux  $\overline{F}_{ncf}\left(U_{cd},U_{dc},\mathcal{P}_{n},n_{f}\right)$  associated with the Riemann solver  $W_{ncf}$  is always defined by

L. Tallois, S. Peluchon, G. Gallice et al.

$$\overline{\boldsymbol{F}}_{ncf} = \boldsymbol{F}_{c} \cdot \boldsymbol{n}_{f} - \int_{-\infty}^{0} \left( \boldsymbol{W}_{ncf} \left( \boldsymbol{\xi}; \boldsymbol{U}_{cd}, \boldsymbol{U}_{dc}, \boldsymbol{\mathcal{P}}_{n}, \boldsymbol{n}_{f} \right) - \boldsymbol{U}_{c} \right) \mathrm{d}\boldsymbol{\xi} - \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{ncf} \cdot \boldsymbol{n}_{f}.$$

The overall scheme becomes

$$\boldsymbol{U}_{c}^{n+1} = \boldsymbol{U}_{c}^{n} - \frac{\Delta t}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left(\overline{\boldsymbol{F}}_{ncf} + \left(\overline{\boldsymbol{P}\Delta \boldsymbol{B}}\right)_{ncf} \cdot \boldsymbol{n}_{f} + \left(\overline{\boldsymbol{P}\Delta \boldsymbol{B}}\right)_{cf} \cdot \boldsymbol{n}_{f}\right),$$

with

$$\left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{ncf} = \overline{\boldsymbol{P}}_{ncf}^{-}\left(\boldsymbol{U}_{cd}, \boldsymbol{U}_{dc}\right)\left(\boldsymbol{B}\left(\boldsymbol{U}_{dc}\right) - \boldsymbol{B}\left(\boldsymbol{U}_{cd}\right)\right),$$

and the centered term defined by the same approximation

$$\left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{cf} = \overline{\boldsymbol{P}}_{cf}^{-}\left(\boldsymbol{U}_{c}, \boldsymbol{U}_{cd}\right) \left(\boldsymbol{B}\left(\boldsymbol{U}_{cd}\right) - \overline{\boldsymbol{B}}_{c}^{+}\right).$$

This last term is the same as in the one-dimensional case. It then remains to approximate the flux  $\overline{B}_c^+$  by  $B(U_c)$ , and discretize the term  $\overline{P}_{cf}$  in a way consistent with P.

#### 2.4. Simple Riemann solver

The previous definitions of Godunov-type schemes are based on the notion of a Riemann solver. From there, any type of exact or approximate Riemann solver can be used, such as the HLL solver [47] or HLLC solver [73], the Roe solver [67] or the Osher solver [36] (see [72]). To construct our numerical scheme, we use a simple Riemann solver, known as a simple solver. This notion was introduced by Gallice [38,39,41]. This is the simplest form of approximate Riemann solver. Famous solvers such as Roe [67], HLL [47] or HLLC [73] are simple Riemann solvers. We first recall the definition of a simple solver, given in [38,39,41], then use them first in the one-dimensional and then multi-dimensional framework. Simple solvers allow us to explicitly calculate the integrals in the flux expression.

**Definition 10** (Simple solver). A simple solver is a Riemann solver defined simply by (m+1) constant states  $(U)_{k=1}^{m+1}$  separated by discontinuities of slope  $(\lambda)_{k=1}^m$ , i.e.

$$\boldsymbol{W}\left(x/t;\boldsymbol{U}_{l},\boldsymbol{U}_{r}\right) = \begin{cases} \boldsymbol{U}_{l} & \text{if } x/t \leq \lambda_{1},\\ \boldsymbol{U}_{k} & \text{if } \lambda_{k-1} < x/t \leq \lambda_{k}, \ k=2,m,\\ \boldsymbol{U}_{r} & \text{if } \lambda_{m} < x/t. \end{cases}$$

# 2.4.1. One-dimensional case

We can then characterize the Godunov-type scheme induced by a simple solver for non-conservative system.

Proposition 5 ([39]). A simple solver induces a non-conservative Godunov-type scheme consistent with the integral form if and only if

$$\overline{F}_{i+\frac{1}{2}}^{+} - \overline{F}_{i+\frac{1}{2}}^{-} = F_{i+1} - F_i + \overline{P\Delta B}_{i+\frac{1}{2}} - \sum_{k=1}^{m} \lambda_k \left( U_{k+1} - U_k \right) = 0.$$
 (24)

If the fluxes are conservative in the classical sense, they are given by

$$\overline{F}_{i+\frac{1}{2}} = \frac{1}{2} \left( F_i + F_{i+1} \right) - \frac{1}{2} \sum_{k=1}^{m} |\lambda_k| \left( U_{k+1} - U_k \right) - \frac{1}{2} \left( \left( \overline{P \Delta B} \right)_{i+\frac{1}{2}}^+ - \left( \overline{P \Delta B} \right)_{i+\frac{1}{2}}^- \right). \tag{25}$$

In this case, the resulting numerical scheme is written as

$$\boldsymbol{U}_{i}^{n+1} = \boldsymbol{U}_{i}^{n} - \frac{\Delta t}{\Delta x} \left( \overline{\boldsymbol{F}}_{i+\frac{1}{2}} - \overline{\boldsymbol{F}}_{i-\frac{1}{2}} \right) - \frac{\Delta t}{2\Delta x} \left( \left( \overline{\boldsymbol{P}} \Delta \boldsymbol{B} \right)_{i+\frac{1}{2}}^{-} + \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}} \right)_{i-\frac{1}{2}}^{+} \right).$$

In practice we use a more convenient form. It consists in writing the scheme as

$$\boldsymbol{U}_{i}^{n+1} = \boldsymbol{U}_{i}^{n} - \frac{\Delta t}{\Delta x} \left( \overline{\boldsymbol{H}}_{i+\frac{1}{2}} - \overline{\boldsymbol{H}}_{i-\frac{1}{2}} \right) - \frac{\Delta t}{2\Delta x} \left( \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}}_{i+\frac{1}{2}} + \overline{\boldsymbol{P}} \Delta \overline{\boldsymbol{B}}_{i-\frac{1}{2}} \right), \tag{26}$$

with  $\overline{H}_{i+\frac{1}{2}}$  the flux given by

$$\overline{H}_{i+\frac{1}{2}} = \frac{1}{2} \left( F_i + F_{i+1} \right) - \frac{1}{2} \sum_{k=1}^{m} |\lambda_k| \delta U_k. \tag{27}$$

# 2.4.2. Multi-dimensional case

In the case of a multi-dimensional Godunov-type scheme, we have the following proposition.

**Proposition 6** ([70]). A simple solver induces a non-classical non-conservative Godunov-type scheme consistent with the integral form if and only if

$$\sum_{f \in SF(nc)} |\Gamma_{nf}| \left[ \left( \boldsymbol{F}_c - \boldsymbol{F}_d + \overline{\boldsymbol{P} \Delta \boldsymbol{B}}_{nf} \right) \cdot \boldsymbol{n}_f - \sum_{k=1}^m \lambda_k \delta \boldsymbol{U}_k \right] = 0.$$

In the case where the fluxes are conservative in the non-classical sense, the induced Godunov-type scheme is given by

$$\boldsymbol{U}_{c}^{n+1} = \boldsymbol{U}_{c}^{n} - \frac{\Delta t}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left(\overline{\boldsymbol{F}}_{ncf} + \frac{1}{2} \left(\overline{\boldsymbol{P} \Delta \boldsymbol{B}}\right)_{ncf} \cdot \boldsymbol{n}_{f}\right),$$

with

$$\overline{\boldsymbol{F}}_{ncf} = \frac{1}{2} \left[ \left( \boldsymbol{F}_c + \boldsymbol{F}_d \right) \cdot \boldsymbol{n}_f - \sum_{k=1}^m |\lambda_k| \delta \boldsymbol{U}_k - \overline{\Delta (\boldsymbol{P} \Delta \boldsymbol{B})}_{nf} \right] - \frac{1}{2} \left[ \left( \boldsymbol{F}_c - \boldsymbol{F}_d + \overline{\boldsymbol{P} \Delta \boldsymbol{B}}_{nf} \right) \cdot \boldsymbol{n}_f - \sum_{k=1}^m \lambda_k \delta \boldsymbol{U}_k \right],$$

and

$$\overline{\Delta (\boldsymbol{P} \Delta \boldsymbol{B})}_{nf} = \frac{1}{2} \left( \left( \overline{\boldsymbol{P} \Delta \boldsymbol{B}} \right)^{-} - \left( \overline{\boldsymbol{P} \Delta \boldsymbol{B}} \right)^{+} \right)_{ncf} \cdot \boldsymbol{n}_{f}.$$

Again, a more convenient and compact form will be used. It is given by

$$\boldsymbol{U}_{c}^{n+1} = \boldsymbol{U}_{c}^{n} - \frac{\Delta t}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SP(nc)} |\Gamma_{ncf}| \overline{\boldsymbol{H}}_{ncf}, \tag{28}$$

with

$$\overline{\boldsymbol{H}}_{ncf} = \boldsymbol{F}_c \cdot \boldsymbol{n}_f - \sum_{k=1}^m \lambda_k^- \delta \boldsymbol{U}_k,$$

where  $\lambda_k^- = \min(\lambda_k, 0)$ . This form is very interesting. The non-conservative flux is composed of the evaluation of the physical flux plus a second part of intermediate states related to cell c. Implementing the scheme in this form requires only the calculation of intermediate states, which will contain the contribution of the non-conservative product.

If the nodal dependence of the Riemann solver is dropped, the previous scheme is exactly the one-dimensional scheme, since it becomes conservative in the classical sense.

# 2.5. Application to general fluid mechanics systems

General fluid mechanics systems are very often described by a hyperbolic system of the form (2) in the one-dimensional case and (1) in the multi-dimensional case. The shallow-water equations with bottom, the Euler equations with gravity, or the Magnetohydrodynamics equations are well-studied systems that fit into this framework [41]. Several assumptions are made in the context of general gas dynamics systems. These general assumptions will enable the construction of Godunov-type schemes based on simple Riemann solvers in the Lagrangian form of the equations. Such systems are given by

$$\partial_{\tau} V + \partial_{M} G(V) + P(V) \partial_{M} B(V) = 0,$$
 (29)

where the first equation of the system is nothing other than the law of conservation of volume  $v = 1/\rho$ .

$$\partial_{\tau}v - \partial_{M}u_{n} = 0, M = \int \rho dx \text{ and } \tau = t,$$
 (30)

with  $\rho$  the density. Assume that a simple solver can be defined following the Definition 10. This solver is assumed to satisfy the following properties (H1) and (H2) [38,41]

$$u_{n,k+1} + \lambda_k v_{k+1} = u_{n,k} + \lambda_k v_k, (H1).$$
 (31)

The assumption (H1) is nothing other than the discrete form of (30). It is a natural assumption, since the volume conservation equation is linear, and in particular allows us to link the Eulerian and Lagrangian solvers [38,41]. Furthermore, (H2) is a natural assumption that the volume of each intermediate state cannot be negative.

$$v_k \ge 0, \ k = 1, m + 1, \ (H2).$$

The non-conservative term must therefore satisfy condition (H3) [39]

$$(\overline{P}\Delta B)_{v} = 0$$
,  $(H3)$ ,

where (), indicates the equation of the quantity • of the system (29). This means that the non-conservative term does not affect the volume equation.

Assuming these properties to be true, it will then be possible to determine the intermediate states of the simple solver. Then, explicit conditions on the slopes of the solver will guarantee the positivity of the intermediate states. Since, for Godunov-type schemes, the solution given after a time step is a convex combination of the intermediate states, this positivity guarantees that the solution remains in the set of admissible solutions, under an explicit condition on the time step.

# 3. Application to the five-equation system with surface tension

Now that the theory of the Godunov-type scheme in the context of a conservation law system with a non-conservative term has been presented in oneand multi-dimensional frameworks, the results are applied to the five-equation system with surface tension using the CSF model. However, they can be applied to various hyperbolic systems with a non-conservative term, such as the shallow-water equations with bottom, Euler's equations with gravity, the Kapila system or Magnetohydrodynamics [38,41,57,24,42,23,30].

The construction of a numerical scheme applied to general fluid dynamics systems is easier in its Lagrangian form than in its Eulerian one. Although usable in a purely Lagrangian framework, we want to be able to handle large interface deformations, which is simpler in an Eulerian formalism. To return to the Eulerian framework, several approaches can be followed. Firstly, the Lagrange-Remap approach is used, for example, in [10,33]. After a purely Lagrangian step, the variables are projected onto a fixed Eulerian mesh. Another approach is to use Gallice's Lagrange-Euler transformation [41], which is particularly well-suited to gas dynamics. From the scheme in its Lagrangian form, we can directly reduce to the Eulerian frame via a discrete relation between the intermediate states of the Lagrangian solver and its Eulerian equivalent. Latige extended this approach to the five-equation system [50], but his extension is not very robust with an implicit time scheme for liquid-gas mixtures. The approach used here is called the Lagrange-Transport splitting method of Chalons et al. [20,21]. The approach consists of using an operator splitting that allows two subsystems to be solved successively in two distinct stages. The first, called Lagrange step, consists in solving the first subsystem in Lagrangian variable. The second, called **Transport step**, involves transporting the conservative quantities at material velocity.

In the family of diffuse interface methods, one of the most comprehensive two-phase models is the Baer & Nunziato model [6]. In this model, the two phases are in pressure and velocity disequilibrium. Unlike some models where the two phases are assumed to be intimately mixed at the mesoscopic scale (solid particles, bubbles, or drops transported in a fluid, for example), we consider here that the two phases are separate, and that it is only for numerical reasons that we have to consider the possibility of mixing between the phases. The model result is therefore not assumed to depend on the closure laws chosen for the two-phase mixture. It is sufficient that the laws chosen ensure the mathematically well-posedness of the model (hyperbolicity, existence of entropy in the Lax sense). Several possibilities have been explored in the literature. One of the most frequently used is Kapila's model [48], which assumes that the volume fraction z adjusts instantaneously to ensure equality of pressures between the two phases, leading to the introduction of a source term in div(V) in the transport equation for z. Another solution proposed independently by Allaire et al. [3] and by Massoni et al. [53] consists in assuming that z is simply convected at the common velocity of the two phases and that it is the internal energies of the two phases (or their temperatures) that adjust instantaneously to ensure the local equilibrium of the pressures between the two phases. This is the model we have chosen to use here, as it leads to a simpler mathematical system. This does not, of course, mean that it is the best model for dealing with hybrid problems involving both zones where the phases are separated and zones where two-phase mixing occurs at the mesoscopic scale, i.e. at a scale smaller than or comparable to the mesh size. But it's important to stress that in this case, the exchanges between the phases depend on the topology of the interface between the phases, and it is necessary either to add equations to the model (for interfacial area density, for example), or to make additional assumptions to ensure that the model obtained is physically correct. These issues are beyond the scope of the present work.

# 3.1. Recalling equations

In the model, each phase k = 1, 2 is defined by its volume fraction  $z_k$ , its density  $\rho_k$  and its internal energy  $\varepsilon_k = E_k - \frac{1}{2}||\boldsymbol{u}||^2$  where  $E_k$  is its total energy and  $\boldsymbol{u}$  is the velocity. An equation of state for each fluid enables to compute its thermodynamic pressure  $p_k(\rho_k, \varepsilon_k)$ . The five-equation system [3,53] with the surface tension force in CSF form is written as

$$\begin{cases} \partial_{t}\rho + \nabla \cdot (\rho \mathbf{u}) &= 0, \\ \partial_{t}(\rho y) + \nabla \cdot (\rho y \mathbf{u}) &= 0, \\ \partial_{t}(\rho \mathbf{u}) + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \nabla p &= \sigma \kappa \nabla z, \\ \partial_{t}(\rho E) + \nabla \cdot (\rho E \mathbf{u} + \rho \mathbf{u}) &= \sigma \kappa \nabla z \cdot \mathbf{u}, \\ \partial_{t}z &+ \mathbf{u} \cdot \nabla z &= 0, \end{cases}$$
(32)

with  $z=z_1$ ,  $y=\rho_1z/\rho$  the mass fraction of phase 1,  $\sigma$  the surface tension coefficient and  $\kappa$  the interface curvature. The mixing density  $\rho$  and the internal energy of mixing  $\varepsilon$  are deduced by weighting the densities and energies of each phase by the volume fraction. We thus have

$$\begin{array}{lcl} \rho & = & \rho_1 z_1 + \rho_2 (1-z_1), \\ \rho \varepsilon & = & \rho_1 \varepsilon_1 z_1 + \rho_2 \varepsilon_2 (1-z_1). \end{array}$$

The total energy of mixing E is thus easily deduced by

$$\rho E = \rho_1 E_1 z_1 + \rho_2 E_2 (1 - z_1).$$

An isobaric closure allows us to deduce the mixture pressure. This closure is given by the following system of unknowns  $\rho_1 \varepsilon_1$  and  $\rho_2 \varepsilon_2$ .

$$\begin{cases} p_1(\rho_1, \rho_1 \varepsilon_1) = p_2(\rho_2, \rho_2 \varepsilon_2), \\ \rho \varepsilon = \rho_1 \varepsilon_1 z_1 + \rho_2 \varepsilon_2 (1 - z_1), \end{cases}$$
(33)

where  $\rho_1$ ,  $\rho_1$  and  $z_1$  are known. When considering perfect gas, stiffened gas, or Mie-Grüneisen gas equation of state, the system (33) can be solved explicitly and the following expression for the mixture pressure is obtained

$$p(\rho, \varepsilon, z_1) = \rho \varepsilon (\gamma(z_1) - 1) - \gamma(z_1) \pi(z_1),$$
 (34)

where  $\gamma$  the mixing heat capacity ratio and  $\pi$  the reference pressure are given by

$$\frac{1}{\gamma - 1} = \sum_{k=1}^{2} \frac{z_k}{\gamma_k - 1} \quad \text{and} \quad \frac{\gamma \pi}{\gamma - 1} = \sum_{k=1}^{2} \frac{z_k \gamma_k \pi_k}{\gamma_k - 1}.$$

The expression for the mixing pressure therefore has the same form as that for a pure phase (34) with the mixing coefficients defined above. The speed of sound is finally given by

$$c^2 = \frac{\gamma(p+\pi)}{a}$$
.

The major drawback of the five-equation system is that entropies of the system are not phase entropies, as opposed to the Kapila model [48]. They will therefore not be studied numerically.

In our Lagrange-Transport splitting approach, system (32) is split into two subsystems. The one studied in **the Lagrange step** is given in Lagrangian variables by

$$\begin{cases}
\partial_{t}v - v\nabla \cdot \boldsymbol{u} &= 0, \\
\partial_{t}y &= 0, \\
\partial_{t}\boldsymbol{u} + v\nabla p &= v\sigma\kappa\nabla z, \\
\partial_{t}E + v\nabla \cdot (p\boldsymbol{u}) &= v\sigma\kappa\nabla z \cdot \boldsymbol{u}, \\
\partial_{t}z &= 0.
\end{cases} \tag{35}$$

Here, we will construct a Godunov-type scheme based on a simple Riemann solver. This solver will be derived from the solution of an approximate Riemann problem in the frame normal to a mesh face. The resulting numerical flux will be consistent with the system associated with the normal  $n = (n_x, n_y)$ .

$$\rho \partial_t \mathbf{V} + \partial_n \mathbf{G}(\mathbf{V}) + \mathbf{P}(\mathbf{V}) \partial_n \mathbf{B}(\mathbf{V}) = 0,$$

with  $V^T = (v, y, u, v, E, z)$ ,  $G(V)^T = (-u, 0, pn_x, pn_v, pu \cdot n, 0)$ ,  $B^T = (0, 0, z, z, z, 0)$  and

$$\boldsymbol{P} = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 &$$

Assuming first-order surface tension effects, i.e. that the curvature is fixed and no longer depends on the volume fraction, this system is hyperbolic and has eigenvalues  $(-\rho c, 0, 0, 0, \rho c)$  with c the speed of sound. We aim to develop an equilibrium scheme that preserves stationary solutions, in particular Laplace's law.

The hyperbolicity set of the five-equation system is

$$\mathcal{D}^{E} = \{ (\rho, \rho y, \rho u, \rho E, z)^{t} \in \mathbb{R}^{6} \mid \rho > 0, c^{2} > 0 \}.$$

This set is determined by the positivity of the density and the square of the speed of sound. It is given in Lagrangian form by

$$\mathcal{D}^{L} = \left\{ (v, y, \mathbf{u}, E, vz)^{t} \in \mathbb{R}^{6} \mid v > 0, c^{2} > 0 \right\}.$$

In particular, we can show that the Euler-Lagrange transformation

$$\mathcal{L} : \mathcal{D}^E \to \mathcal{D}^L$$

$$U = (\rho, \rho v, \rho u, \rho E, z)^l \mapsto V = (v, v, u, E, vz)^l,$$

preserves convex sets [57]. The hyperbolicity domain is not sufficient to define a physically admissible set of solutions. For example, we want to preserve the saturation relation and impose that the volume fraction z and the mass fraction y must lie between 0 and 1.

![](_page_15_Figure_2.jpeg)

Fig. 5. Riemann solver for the system (35).

We also need the specific internal energy to be positive. We can thus define the domains of admissible physical solutions written in Lagrangian form

$$\mathcal{A}^{L} = \left\{ (v, y, \mathbf{u}, E, vz)^{t} | v > 0, y \in (0, 1), \varepsilon > 0, z \in (0, 1), p + \pi(z) > 0 \right\},\$$

and in Eulerian form

$$\mathcal{A}^{E} = \left\{ (\rho, \rho y, \rho \mathbf{u}, \rho E, z)^{t} | \rho > 0, y \in (0, 1), \varepsilon > 0, z \in (0, 1), \rho + \pi(z) > 0 \right\}.$$

The positivity of the quantity  $p + \pi(z)$  ensures that the square of the speed of sound is positive, and thus preserves hyperbolicity. For a stiffened-gas equation of state [57], the sets  $A^E$  and  $A^L$  are convex under the condition

$$(\gamma_2 - \gamma_1)(\pi_2 - \pi_1) \ge 0.$$

It is therefore necessary to use ordered parameters  $\gamma_k$  and  $\pi_k$ , k=1,2. In liquid/gas flows, we generally have  $\pi_g=0$  and  $\gamma_l \geq \gamma_g$ . Convexity is thus assured.

#### 3.2. Simple Lagrangian Riemann solver: one-dimensional case

A scheme is said to be "balanced" or "well-balanced" if it preserves equilibrium solutions. In our case, these states are continuously given by u=0 and  $\nabla p=\sigma\kappa\nabla z$ , which corresponds to Laplace's law. A naive consideration of non-conservative terms, i.e. a simple centered discretization of  $\sigma\kappa\nabla z$ , does not allow us to discretely verify Laplace's law given by u=0 and  $\Delta p=\sigma\kappa\Delta z$ . Such discretization generates parasitic currents, even if the curvature is exactly calculated. We will therefore build a Godunov-type scheme based on a simple Riemann solver, ensuring that the equilibrium conditions are verified. In the following, we consider  $\sigma=1$  for the sake of clarity. In addition, as the relations to the mass and volume fraction equations y and z are trivial, they will be omitted.

# 3.2.1. Construction of the Lagrangian Riemann solver

Let  $\boldsymbol{W}\left(x/t;\boldsymbol{V}_{l},\boldsymbol{V}_{r}\right)$  be an approximation of the Riemann problem defined by the system (35) at each interface between the left and right states  $\boldsymbol{V}_{l},\boldsymbol{V}_{r}$ . The  $\boldsymbol{W}$  self-similar function consists of four intermediate states separated by discontinuities propagating at  $-\lambda^{-}v_{l}$ , 0 and  $\lambda^{+}v_{r}$  (see Fig. 5).  $\boldsymbol{W}$  is defined by

$$\boldsymbol{W} \left( \boldsymbol{x}/t; \boldsymbol{V}_l, \boldsymbol{V}_r \right) = \begin{cases} \boldsymbol{V}_l & \text{if } \boldsymbol{x}/t \leq -\lambda^- v_l, \\ \boldsymbol{V}_l^* & \text{if } -\lambda^- v_l < x/t \leq 0, \\ \boldsymbol{V}_r^* & \text{if } 0 < x/t \leq \lambda^+ v_r, \\ \boldsymbol{V}_r & \text{if } \lambda^+ v_r < x/t. \end{cases}$$

In the following, the notation  $\lambda^- = \lambda_l$  and  $\lambda^+ = \lambda_r$  will sometimes be used. First of all, since property (H1) (31) is assumed to be true, we have

$$u_{l}^{*} - \lambda^{-}v_{l}^{*} = u_{l} - \lambda^{-}v_{l},$$

$$u_{r}^{*} + \lambda^{+}v_{r}^{*} = u_{r} + \lambda^{+}v_{r},$$

$$u_{l}^{*} = u_{r}^{*}.$$
(36)

The intermediate velocities of the left and right states are equal, allowing us to write  $u_l^* = u_r^* = u^*$ . Now we introduce the fluxes  $\overline{G}_s = (-u^*, 0, \overline{p}_s, \overline{pu}_s, 0)^t$ , s = l, r. If the Riemann solver is of type Godunov for the system (35) then the relations (24) give

$$\begin{cases} -\lambda^{-}(v_{l}^{*} - v_{l}) + \lambda^{+}(v_{r} - v_{r}^{*}) = -\Delta u, \\ -\lambda^{-}(u_{l}^{*} - u_{l}) + \lambda^{+}(u_{r} - u_{r}^{*}) = \Delta p - \overline{\kappa \Delta z}, \\ -\lambda^{-}(E_{l}^{*} - E_{l}) + \lambda^{+}(E_{r} - E_{r}^{*}) = \Delta (pu) - \overline{\kappa u \Delta z}, \end{cases}$$
(37)

where  $\overline{\kappa \Delta z}$  is an approximation of the non-conservative product. From (36) and (37) the expressions for  $u^*$ ,  $v_i^*$  and  $v_i^*$  can be deduced

$$\begin{split} u^* &= \frac{\lambda^+ u_r + \lambda^- u_l}{\lambda^+ + \lambda^-} - \frac{\Delta p - \overline{\kappa \Delta z}}{\lambda^+ + \lambda^-}, \\ v_l^* &= v_l - \frac{1}{\lambda^- (\lambda^- + \lambda^+)} \left( \Delta p - \overline{\kappa \Delta z} - \lambda^+ \Delta u \right), \\ v_r^* &= v_r + \frac{1}{\lambda^+ (\lambda^- + \lambda^+)} \left( \Delta p - \overline{\kappa \Delta z} + \lambda^- \Delta u \right). \end{split}$$

From the flux equations (6), we have the left and right relations

$$\begin{cases} u^* &= u_l + \lambda^-(v_l^* - v_l), \\ \overline{p}_l &= p_l - \lambda^-(u^* - u_l) \\ \overline{p}\overline{u}_l &= p_l u_l - \lambda^-(E_l^* - E_l) \\ \end{cases} + \underbrace{(\overline{\kappa}\Delta z)_l}_{(\overline{\kappa}u\Delta z)_l}, \begin{cases} u^* &= u_r - \lambda^+(v_r^* - v_r), \\ \overline{p}_r &= p_r + \lambda^+(u^* - u_r) \\ \overline{p}\overline{u}_r &= p_r u_r + \lambda^+(E_r^* - E_r) \\ \end{cases} - \underbrace{(\overline{\kappa}\Delta z)_r}_{(\overline{\kappa}u\Delta z)_r},$$

where  $\overline{\kappa \Delta z_s}$ , s = l, r are approximations of the non-conservative product. As explained in Sec. 2.1, those approximations do not need to be given, only there sum  $\overline{\kappa \Delta z}$  needs to be defined. Its approximation will be examined later.

Assuming on the discretization of the total energy flux that  $\overline{pu}_s = \overline{p}_s u^*$  and  $\kappa u \Delta z_s = \kappa \Delta z_s u^*$ , s = l, r, the intermediate states of the Riemann solver are then fully characterized. We then have the following remark.

**Remark 3.** The Riemann solver obtained for the five-equation system with surface tension is exactly the extension of Peluchon's solver et al. without surface tension [58], and Gallice's solver in the context of gas dynamics [41]. We can summarize W as

$$\boldsymbol{W}\left(x/t;\boldsymbol{V}_{l},\boldsymbol{V}_{r}\right) = \begin{cases} \boldsymbol{V}_{l} & \text{if } x/t \leq -\lambda^{-}v_{l}, \\ \boldsymbol{V}_{l}^{*} &= \boldsymbol{V}_{l} + \phi_{-}R_{-} & \text{if } -\lambda^{-}v_{l} < x/t \leq 0, \\ \boldsymbol{V}_{r}^{*} &= \boldsymbol{V}_{r} - \phi_{+}R_{+} & \text{if } 0 < x/t \leq \lambda^{+}v_{r}, \\ \boldsymbol{V}_{r} & \text{if } \lambda^{+}v_{r} < x/t, \end{cases}$$

$$(38)$$

where

$$\begin{array}{lcl} R_{\pm} & = & (-1,0,\pm\lambda_{\pm},p_{1-\alpha}\pm\lambda_{\pm}u_{\alpha},0)^t, \\ \phi_{\pm} & = & \frac{\Delta p - \kappa\Delta z \pm \lambda_{\mp}\Delta u}{\lambda_{-}\lambda_{+} + \lambda_{+}^2}, \end{array}$$

and

$$\overline{\kappa \Delta z} = (\overline{\kappa \Delta z})_t + (\overline{\kappa \Delta z})_x$$

with  $\alpha = \lambda^-/(\lambda^+ + \lambda^-)$  and the notation  $x_\alpha = (1 - \alpha)x_r + \alpha x_l$ .

# 3.2.2. Overall scheme

The overall scheme under the form (26) is simply

$$\begin{cases} v_{i}^{n+1} &= v_{i}^{n} + v_{i}^{n} \frac{\Delta t}{\Delta x} \left( \overline{u}_{i+\frac{1}{2}}^{n} - \overline{u}_{i-\frac{1}{2}}^{n} \right), \\ y_{i}^{n+1} &= y_{i}^{n}, \\ u_{i}^{n+1} &= u_{i}^{n} - v_{i}^{n} \frac{\Delta t}{\Delta x} \left( \overline{p}_{i+\frac{1}{2}}^{n} - \overline{p}_{i-\frac{1}{2}}^{n} \right) + v_{i}^{n} \frac{\Delta t}{2\Delta x} \left( \overline{\kappa \Delta z}_{i+\frac{1}{2}}^{n} + \overline{\kappa \Delta z}_{i-\frac{1}{2}}^{n} \right), \\ E_{i}^{n+1} &= E_{i}^{n} - v_{i}^{n} \frac{\Delta t}{\Delta x} \left( \overline{p}_{i+\frac{1}{2}}^{n} \overline{u}_{i+\frac{1}{2}}^{n} - \overline{p}_{i-\frac{1}{2}}^{n} \overline{u}_{i-\frac{1}{2}}^{n} \right) + v_{i}^{n} \frac{\Delta t}{2\Delta x} \left( \overline{\kappa \Delta z}_{i+\frac{1}{2}}^{n} \overline{u}_{i+\frac{1}{2}}^{n} + \overline{\kappa \Delta z}_{i-\frac{1}{2}}^{n} \overline{u}_{i-\frac{1}{2}}^{n} \right), \\ z_{i}^{n+1} &= z_{i}^{n}, \end{cases}$$

$$(39)$$

with

$$\overline{\boldsymbol{H}}_{i+\frac{1}{2}} = (-\overline{u}, 0, \overline{p}, \overline{pu}, 0)_{i+\frac{1}{2}}^{t},$$

where  $\overline{u} = u^*$  and

$$\left(\overline{\boldsymbol{P}\Delta\boldsymbol{B}}\right)_{i+\frac{1}{2}} = -\overline{\kappa}\Delta\overline{z} \left(0,0,1,\overline{u},0\right)_{i+\frac{1}{2}}^{t}.$$

The relation between  $\overline{p}_s$ , s = l, r and  $\overline{p}$  is just the difference between (25) and (27), that is

$$\overline{p}_s = \overline{p} - \frac{1}{2} \left( (\overline{\kappa \Delta z})_r - (\overline{\kappa \Delta z})_l \right), \quad s = l, r.$$

We have therefore constructed a numerical scheme based on a four-state Riemann solver, taking into account the non-conservative surface tension effects. We still need to discretize the non-conservative term to preserve stationary states and to determine conditions on the Riemann solver slopes that guarantee the positivity of the intermediate states.

# 3.2.3. Equilibrium solutions

So far, no choice has been made regarding the discretization of the non-conservative terms  $\overline{P\Delta B}$ . In our case, the natural choice, as (10), is to consider  $\overline{P\Delta B} = \overline{P}\Delta B$  where  $\overline{P}$  is the arithmetic mean

$$\overline{\boldsymbol{P}}(\boldsymbol{U}_i, \boldsymbol{U}_{i+1}) = \frac{1}{2} \left( \boldsymbol{P}(\boldsymbol{U}_i) + \boldsymbol{P}(\boldsymbol{U}_{i+1}) \right).$$

This approximation is consistent in the sense that  $\overline{P}(U,U) = P(U)$ . We can deduce that

$$\left(\overline{\kappa\Delta z}\right)_{i+\frac{1}{2}} = \left(\frac{\kappa_i + \kappa_{i+1}}{2}\right)(z_{i+1} - z_i).$$

**Proposition 7.** Discrete solutions satisfying Laplace's law  $(u=0,\ \Delta p=\overline{\kappa}\Delta z)$  are equilibrium solutions. In addition, the previous choice preserves the equilibrium solutions  $u=0,\ \Delta\kappa=0$  and  $\Delta p=\kappa\Delta z$ .

**Proof.** Using the Remark 3, it's easy to show that Laplace's law is a strong equilibrium solution. If the curvature is constant, we have  $\overline{\kappa} = \kappa$ . The relation  $\Delta p = \kappa \Delta z$  is the application of Proposition 2.

#### 3.2.4. Positivity conditions

There are various ways of exhibiting conditions on the slopes that guarantee the positivity of the intermediate states [24]. One method was introduced by Gallice [41] for gas dynamics and extended in [58] for the five-equation system without surface tension. It consists of introducing the ratio of the slopes  $r = \lambda^+/\lambda^-$  and reducing it to the resolution of quadratic equations [70]. Here, we will adapt one of the methods presented in [24] for gas dynamics and in [30] for shallow-water equations.

To guarantee the positivity of the density, we can use the fact that

$$\begin{split} v_l^* &= \frac{\lambda^-}{\lambda^- + \lambda^+} \left( v_l - \frac{\Delta p - \overline{\kappa} \Delta z}{\lambda^{-2}} \right) + \frac{\lambda^+}{\lambda^- + \lambda^+} \left( v_l + \frac{\Delta u}{\lambda^-} \right), \\ v_r^* &= \frac{\lambda^-}{\lambda^- + \lambda^+} \left( v_r + \frac{\Delta u}{\lambda^+} \right) + \frac{\lambda^+}{\lambda^- + \lambda^+} \left( v_r + \frac{\Delta p - \overline{\kappa} \Delta z}{\lambda^{+2}} \right). \end{split}$$

Thus, by ensuring that the terms in brackets are positive, the intermediate states of the volume fraction are positive. Since a Godunov-type scheme is made up of convex combinations of positive states, the solution is positive.

We can also guarantee the positivity of the internal energy. Multiplying the pressure flux  $\overline{p}_s$  by  $\frac{1}{2}(u^* + u_s)$ , s = l, r, substituting it for the total energy flux  $\overline{p}_s u^*$ , s = l, r, and using the relation on intermediate states of velocity and volume, we obtain

$$\varepsilon_s^* = \varepsilon_s - p_s(v_s^* - v_s) + \frac{\lambda_s^2}{2}(v_s^* - v_s)^2, \ s = l, r.$$
 (40)

This second-degree equation in  $(v_s^* - v_s)$  allows us to deduce that if the discriminant is negative, then  $\varepsilon_s^* > 0$ . Thus,

$$\varepsilon_s^* > 0 \text{ if } \lambda_s \ge \frac{p_s}{\sqrt{2\varepsilon_s}}, \ s = l, r.$$

In the case of a stiffened gas equation, the above equation becomes

$$\varepsilon_s^* > 0 \ \text{if} \ \lambda_s \geq \sqrt{\frac{\gamma-1}{2\gamma}} \rho_s c_s \left( c_s - \frac{\gamma_s \pi_s}{\rho_s c_s} \right) \sqrt{\frac{\rho_s}{\rho_s c_s^2 + \gamma_s (\gamma_s - 1) \pi_s}}, \ s = l, r.$$

However, for the five-equation system, hyperbolicity is assured if  $p + \pi > 0$  in the case of two stiffened gases. Since  $p = \rho \varepsilon (\gamma - 1) - \gamma \pi$ , we have

$$p + \pi > 0$$
 if and only if  $\varepsilon - v\pi > 0$ .

Since the volume fraction does not change during the **Lagrange step**, we have  $\pi_s^* = \pi_s$ . Using (40), we have

$$\varepsilon_{s}^{*} - v_{s}^{*} \pi_{s} = \varepsilon_{s} - v_{s} \pi_{s} - (p_{s} + \pi_{s})(v_{s}^{*} - v_{s}) + \frac{\lambda_{s}^{2}}{2}(v_{s}^{*} - v_{s})^{2}, \ s = l, r.$$

This time, the speed of sound is positive if

$$p_s^* + \pi_s > 0$$
 if  $\lambda_s \ge \frac{p_s + \pi_s}{\sqrt{2(\varepsilon_s - \upsilon_s \pi_s)}}$ ,  $s = l, r$ .

Using the equation of state, we obtain

$$p_s^* + \pi_s > 0 \text{ if } \lambda_s \ge \sqrt{\frac{\gamma - 1}{2\gamma}} \rho_s c_s, \ s = l, r.$$

Putting together the conditions of positivity of the volume v and of  $p + \pi$ , we deduce that the choice of slopes  $\lambda_l$  and  $\lambda_r$  such that

$$\lambda_l = \max\left(\rho_l c_l, \sqrt{\frac{|\Delta p - \overline{\kappa} \Delta z|}{\upsilon_l}}, -\frac{\Delta u}{\upsilon_l}\right), \text{ and } \lambda_r = \max\left(\rho_r c_r, \sqrt{\frac{|-\Delta p + \overline{\kappa} \Delta z|}{\upsilon_r}}, -\frac{\Delta u}{\upsilon_r}\right),$$

makes it possible to guarantee that the intermediate states of the volume as well as the speed of sound are positive.

**Remark 4.** The positivity conditions presented previously were obtained by assuming an explicit temporal discretization. Generally, it is not possible to derive such conditions on slopes with an implicit time integration procedure. Recent work, however, has introduced a class of unconditionally stable implicit schemes to solve the gas dynamics equations [60].

We therefore constructed a Godunov-type scheme based on a simple solver to solve the five-equation system with surface tension. The scheme is positive, preserves stationary states, and is conditionally stable over the time step. Now, we will extend this scheme to the multi-dimensional framework.

# 3.3. Simple Lagrangian Riemann solver: multi-dimensional case

This time we are in a two-dimensional configuration. The velocity becomes  $\mathbf{u} = (u, v)^t$ . We want to build a numerical scheme for the system (35).

#### 3.3.1. Construction of the Lagrangian Riemann solver

First of all, the property (H1) (31) is again considered true.

$$u_{n,l}^* - \lambda^- v_l^* = u_{n,l} - \lambda^- v_l, u_{n,r}^* + \lambda^+ v_r^* = u_{n,r} + \lambda^+ v_r, u_{n,l}^* = u_{n,r}^*,$$

$$(41)$$

with  $u_{n,s} = u_s \cdot n_f$ , s = l, r where  $n_f = n_{cd}$  is the outgoing normal of the cell c. It is then possible to define  $u_{n,l}^* = u_n^* = u_{n,r}^*$ . Secondly, assuming that the tangential velocity only changes on either side of the contact discontinuity, we obtain

$$u_{t,l}^* = u_{t,l}, u_{t,r}^* = u_{t,r},$$

with  $u_{t,s} = u_s \cdot t$ , s = l, r where t is the orthogonal vector of  $n_f$ . We introduce left and right fluxes

$$\overline{\boldsymbol{G}}_{nsf} = (-u_{\boldsymbol{n}}^*, 0, \overline{p}_s n_x, \overline{p}_s n_y, \overline{p}_s u_{\boldsymbol{n}}^*, 0)^t, \ s = l, r,$$

where the assumption on the discretization of the energy flux  $\overline{pu}_s = \overline{p}_s u_n^*$  is made. These fluxes satisfy the following relationships

$$\overline{G}_{nf}^{-} - G_{l} \cdot n_{f} = -\lambda^{-} (V_{l}^{*} - V_{l}) - (\overline{P\Delta B})_{l} \cdot n_{f}, 
\overline{G}_{nf}^{+} - G_{r} \cdot n_{f} = -\lambda^{+} (V_{r} - V_{r}^{*}) + (\overline{P\Delta B})_{r} \cdot n_{f}, \tag{42}$$

where we recall that  $\overline{G}_{ncf} = \overline{G}_{nf}^-$  and  $\overline{G}_{ndf} = -\overline{G}_{nf}^+$ . The equations (42) represent a system of 8 equations with 9 unknowns. By introducing  $u_n^*$  the nodal parameter such that  $u_n^* = u_n^* \cdot n_f$ ,  $u_t^* = u_n^* \cdot t$  we obtain

$$\begin{cases} u_{n}^{*} = u_{n,l} + \lambda^{-}(v_{l}^{*} - v_{l}), \\ \overline{p}_{l}n_{f} = p_{l}n_{f} - \lambda^{-}(u_{n}^{*} - u_{l}) + (\overline{\kappa\Delta z})_{l}n_{f}, \\ \overline{p}_{l}u_{n}^{*} = p_{l}u_{l} - \lambda^{-}(E_{l}^{*} - E_{l}) + (\overline{\kappa u\Delta z})_{l}, \end{cases} \begin{cases} u_{n}^{*} = u_{n,r} - \lambda^{+}(v_{r}^{*} - v_{r}), \\ \overline{p}_{r}n_{f} = p_{r}n_{f} + \lambda^{+}(u_{n}^{*} - u_{r}) - (\overline{\kappa\Delta z})_{r}n_{f}, \\ \overline{p}_{r}u_{n}^{*} = p_{r}u_{r} + \lambda^{+}(E_{r}^{*} - E_{r}) - (\overline{\kappa u\Delta z})_{r}, \end{cases}$$
(43)

by omitting the trivial relations between the volume fraction and the mass fraction. Additionally, one of the two pressure flux equations is redundant because the states  $u_{t,l}^*$  and  $u_{t,r}^*$  are already known. We still have a system of 6 equations and 7 unknowns  $u_n^*, v_l^*, v_r^*, \overline{p}_l, \overline{p}_r, E_l^*, E_r^*$ . Still making the discretization hypothesis  $\overline{\kappa u \Delta z} = u_n^* \overline{\kappa \Delta z}$ , we notice that the matrix  $\overline{P}$  depends on the nodal parameter. We can write the difference of the two fluxes in the more compact form

$$\overline{\boldsymbol{G}}_{nf}^{+} - \overline{\boldsymbol{G}}_{nf}^{-} = (\overline{p}_{r} - \overline{p}_{l}) \begin{pmatrix} 0 \\ 0 \\ \boldsymbol{n}_{f} \\ \boldsymbol{u}_{n}^{*} \\ 0 \end{pmatrix}.$$

Using the left and right pressure fluxes given by the relations (43), and after having developed the calculations, we deduce

$$\overline{p}_r - \overline{p}_l = (\lambda^+ + \lambda^-) \left( u_n^* - \left( \frac{\lambda^+ u_{n,r} + \lambda^- u_{n,l} - \Delta p + \overline{\kappa \Delta z}}{\lambda^+ + \lambda^-} \right) \right),$$

with

$$\overline{\kappa \Delta z} = (\overline{\kappa \Delta z})_l + (\overline{\kappa \Delta z})_r$$

In the same way, as in [42], we introduce the normal velocity to a face

$$\overline{u}_{n} = \frac{\lambda^{+} u_{n,r} + \lambda^{-} u_{n,l} - \Delta p + \overline{\kappa} \Delta z}{\lambda^{+} + \lambda^{-}},$$

which corresponds to the normal velocity of the one-dimensional acoustic solver with surface tension effects. The value given to  $u_n^*$  defines the schema type

- If  $u_n^* = \overline{u}_n$ , we find the one-dimensional Godunov type scheme of the previous part, provided we define  $\overline{\kappa \Delta z}$  on the faces. Pressure fluxes are therefore conservative in the classical sense.
- If  $u_n^* \neq \overline{u}_n$ , we find a multi-dimensional or nodal Godunov type scheme of the [42] type. This scheme is therefore not conservative in the classic sense of the term. We can choose to define  $\overline{\kappa \Delta z}$  on the faces or on the nodes.

#### 3.3.2. Conservation

For a multi-dimensional Godunov-type scheme, the conservation condition is obtained by summing the contributions of the flux around a node. According to (20), we have

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( \overline{\boldsymbol{G}}_{nf}^{+} - \overline{\boldsymbol{G}}_{nf}^{-} \right) = 0,$$

thus

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( -\sum_{k=1}^{m} \lambda_{k} \left( \boldsymbol{V}_{k+1} - \boldsymbol{V}_{k} \right) + \left( \Delta \boldsymbol{G}_{nf} + \left( \overline{\boldsymbol{P} \Delta \boldsymbol{B}} \right)_{nf} \right) \cdot \boldsymbol{n}_{f} \right) = 0.$$

The previous calculations lead to

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left(\overline{p}_{df} - \overline{p}_{cf}\right) \begin{pmatrix} 0 \\ 0 \\ n_f \\ u_n^* \\ 0 \end{pmatrix} = 0.$$

By retaining only the velocity equation, we obtain

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( \overline{p}_{df} - \overline{p}_{cf} \right) \boldsymbol{n}_f = 0.$$

Here,  $u_n^*$  is always an unknown of the problem, attached to each face of the cells connected to the node n. We had previously assumed the existence of a nodal velocity  $u_n^*$  such that

$$u_n^* = u_n^* \cdot n_f$$
.

This closure is identical to that carried out in the construction of the multi-dimensional schemes GLACE [35], EUCCLHYD [52], and more recently in [42]. We will omit in the following the exponent \* on the nodal velocity  $u_n^*$ . The pressure flux difference is now transformed into a nodal conservation condition

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( \lambda_r + \lambda_l \right) \left( \boldsymbol{u}_n \cdot \boldsymbol{n}_f - \overline{\boldsymbol{u}}_f \right) \boldsymbol{n}_f = 0,$$

with

$$\overline{u}_f = \frac{\lambda_r u_{n,r} + \lambda_l u_{n,l} - \Delta p + \overline{\kappa} \Delta z}{\lambda_r + \lambda_l},$$

where we again used the notation  $\lambda^- = \lambda_l$  and  $\lambda^+ = \lambda_r$  on the face f. The equation (20) is nothing other than a  $2 \times 2$  system having as unknown the nodal velocity  $u_n$ 

$$\sum_{f \in SF(n)} \left| \Gamma_{nf} \right| \left( \lambda_r + \lambda_l \right) \left( \boldsymbol{n}_f \otimes \boldsymbol{n}_f \right) \boldsymbol{u}_n = \sum_{f \in SF(n)} \left| \Gamma_{nf} \right| \left( \lambda_r + \lambda_l \right) \overline{\boldsymbol{u}}_f \boldsymbol{n}_f.$$

This system is invertible and therefore admits a unique solution [52]. The determined nodal velocity depends on the quantities of all cells attached to the same node. The numerical scheme is therefore truly multi-dimensional.

**Remark 5.** The Riemann solver  $W_{ncf}$  is given by

$$\boldsymbol{W}\left(\boldsymbol{\xi};\boldsymbol{V}_{c},\boldsymbol{V}_{d},\boldsymbol{u}_{n},\boldsymbol{n}_{f}\right) = \begin{cases} \boldsymbol{V}_{c} & \text{if} & \boldsymbol{\xi} \leq -\lambda_{l}v_{c}, \\ \boldsymbol{V}_{c}^{*} & = & \boldsymbol{V}_{c} + \phi_{c}\boldsymbol{R}_{c}^{-} & \text{if} & -\lambda_{l}v_{c} < \boldsymbol{\xi} \leq 0, \\ \boldsymbol{V}_{d}^{*} & = & \boldsymbol{V}_{d} - \phi_{d}\boldsymbol{R}_{d}^{+} & \text{if} & 0 < \boldsymbol{\xi} \leq \lambda_{r}v_{d}, \\ \boldsymbol{V}_{d} & & \text{if} & \lambda_{r}v_{d} < \boldsymbol{\xi}, \end{cases}$$

with

$$\begin{array}{lcl} \boldsymbol{R}_{c,d}^{\pm} & = & (-1,0,\pm \lambda_{l,r}\boldsymbol{n}_f, p_{c,d} \pm \lambda_{l,r}\boldsymbol{u}_n \cdot \boldsymbol{n}_f, 0)^t, \\ \boldsymbol{\phi}_{c,d} & = & -(\boldsymbol{u}_n - \boldsymbol{u}_{c,d}) \cdot \boldsymbol{n}_f/\lambda_{l,r}. \end{array}$$

The R vector has the same structure as that obtained in the one-dimensional case in the Remark 3.

#### 3.3.3. Overall scheme

Once the nodal velocity has been determined, we can easily obtain the fluxes of the multi-dimensional scheme. Namely, the overall scheme under the form (28) is as follows

$$\begin{cases} v_c^{n+1} &= v_c^n &+ v_c^n \frac{\Delta t}{|\Omega_c|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \mathbf{u}_n \cdot \mathbf{n}_f, \\ y_c^{n+1} &= y_c^{n+1}, \\ \mathbf{u}_c^{n+1} &= \mathbf{u}_c^n &- v_c^n \frac{\Delta t}{|\Omega_c|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left( p_c \cdot \mathbf{n}_f - \lambda_l \left( \mathbf{u}_n - \mathbf{u}_c \right) \mathbf{n}_f \right), \\ E_c^{n+1} &= E_c^n &- v_c^n \frac{\Delta t}{|\Omega_c|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{ncf}| \left( p_c \cdot \mathbf{n}_f - \lambda_l \left( \mathbf{u}_n - \mathbf{u}_c \right) \mathbf{n}_f \right) \mathbf{u}_n, \\ z_c^{n+1} &= z_c^{n+1}. \end{cases}$$

In this form, we can see that calculating fluxes only requires determining the nodal parameter  $u_n$ , surface tension effects being entirely included in this parameter.

We have therefore constructed a multi-dimensional scheme to solve the five-equation system with surface tension effects, in its Lagrangian form. Unlike the one-dimensional scheme constructed in the previous subsection, this scheme depends on a stencil not only on the neighbors of the faces but also on the neighbors of the nodes. The final form of the scheme is similar to that obtained in [30] for source terms. However, surface tension effects are taken into account in intermediate states and nodal velocity. Finally, if we consider classical conservative fluxes, i.e.  $u_n \cdot n_f = \overline{u}_f$ , the scheme is strictly equivalent to (39).

# 3.3.4. Equilibrium solutions

To deal with equilibrium solutions, we always assume that the discretization of the non-conservative product  $\overline{P\Delta B} = \overline{P}\Delta B$ . It is necessary to define the parameter  $\overline{\kappa}$ . As seen in Sec. 2.2.3, it can be defined at the faces, as in the one-dimensional case, or at the nodes. If defined at the faces, we have

$$\overline{\kappa}_f = \frac{\kappa_c + \kappa_d}{2}$$
.

This verifies equilibrium on each side. So, when u=0 and  $\kappa=cste$ , the scheme preserves Laplace's law. If we define the curvature approximation at nodes, then we have  $\overline{\kappa}_n$  such that

$$\overline{\kappa}_n = \frac{1}{\operatorname{card}\left\{C(n)\right\}} \sum_{c \in C(n)} \kappa_c. \tag{44}$$

The approximation of the curvature at a node is the average of the curvatures of the cells attached to that node. Finally, we have the following proposition.

**Proposition 8.** Discrete solutions satisfying Laplace's law  $(u = 0, \Delta p = \overline{\kappa} \Delta z)$  are equilibrium solutions. Moreover, the two choices of approximation of  $\overline{\kappa}$ , on faces or on nodes, preserve the equilibrium solutions  $u = 0, \Delta \kappa = 0$  and  $\Delta p = \kappa \Delta z$ .

**Proof.** The demonstration is analogous to the one-dimensional case.  $\Box$ 

# 3.3.5. Positivity conditions

The positivity conditions for the intermediate states of the multi-dimensional scheme are more complicated to derive explicitly. For example, for the specific volume, using (41)

$$v_l^* = v_l + \frac{u_{n,l}^* - u_{n,l}}{\lambda^-}$$
 and  $v_r^* = v_r - \frac{u_{n,r}^* - u_{n,r}}{\lambda^+}$ .

The positivity of intermediate states of specific volume is ensured by taking

$$\lambda^- \ge -\frac{u_n^* - u_{n,l}}{v_l}$$
 and  $\lambda^+ \ge \frac{u_n^* - u_{n,r}}{v_r}$ .

The calculation of the nodal velocity  $u_n^*$  makes the intermediate velocity state nonlinear regarding solver slopes. In the context of gas dynamics, Chan used a fixed-point method [23] in her thesis to ensure the positivity of the specific volume. In our case, the slopes are calculated as in the one-dimensional scheme case. Moreover, since the aim is to use an implicit time scheme for which no conditions can be given, this strategy is sufficient.

#### 3.4. Stability

The stability conditions for both the one-dimensional (11) and multi-dimensional (19) approaches can be simplified as follows

$$\frac{1}{\Delta t} > \max_{c} \left( \frac{v_{c}}{|\Omega_{c}|} \sum_{n \in \mathcal{N}(c)} \sum_{f \in SF(nc)} |\Gamma_{cf}| \lambda^{-} \right).$$

As the terms related to surface tension are directly taken into account in the solver, they are also taken into account in the slopes and therefore in the stability condition. This was made possible by considering curvature as a first-order term. However, in practice, we can also add the stability condition for surface tension given by Brackbill [13].

$$\frac{1}{\Delta t} > \max_{c} \sqrt{2\pi\sigma v_{c} \sum_{d \in \mathcal{V}(c)} \left(\frac{\left|\Gamma_{cd}\right|}{\left|\Omega_{c}\right|}\right)^{3}}.$$

#### 3.5. Lagrange-transport splitting scheme

The Lagrange-Transport splitting method of Chalons et al. [20,21] for a time step between  $t^n$  and  $t^{n+1}$  is as follows

- **Step 1:** From a state  $U^n$ , calculate  $U^{\dagger}$ , the approximation of the Lagrange system.
- **Step 2:** Find the fluid state  $U^{n+1}$  by solving the Transport system with the initial state  $U^{\dagger}$ .

The study of the five-equation system with dissipative effects was carried out in [59]. The same method is used here. Viscous effects are taken into account in the **Lagrange step**. Velocity gradients are discretized using a transformation between a curvilinear grid and a Cartesian reference grid. Stability constraints related to viscous dissipation is

$$\frac{1}{\Delta t} > \max_{c} \left( \frac{1}{|\Omega_{c}|} \sum_{d \in \mathcal{V}(c)} |\Gamma_{cd}|^{2} \left( \frac{\mu}{|\Omega|} \right)_{cd} \right).$$

The **Transport step** system in its compact form is given by

$$\partial_t \mathbf{U} + \nabla \cdot (\mathbf{u}\mathbf{U}) - \mathbf{U}\nabla \cdot \mathbf{u} = 0,$$

where  $U = (\rho, \rho y, \rho u, \rho E, z)^t$  corresponds to the vector of conservative variables of the global system. The **Transport step** is solved as

$$\boldsymbol{U}_{c}^{n+1} = \boldsymbol{U}_{c}^{\dagger} + \boldsymbol{U}_{c}^{\dagger} \frac{\Delta t}{|\Omega_{c}|} \sum_{d \in \mathcal{V}(c)} \left| \Gamma_{cd} \right| \tilde{u}_{cd} - \sum_{d} \left| \Gamma_{cd} \right| \boldsymbol{U}_{cd}^{\dagger} \tilde{u}_{cd},$$

with  $U_{cd}^{\dagger}$  the upwind flux. The choice of discretization of the material velocity  $\tilde{u}_{cd}$  is decisive for a globally conservative scheme [20]. To achieve this, we simply take the opposite of the first flux component  $\overline{u}_{cd}$  as  $\tilde{u}_{cd}$ , i.e.  $\overline{u}_{cd}$ . This scheme is stable under the time-step condition

$$\frac{1}{\Delta t} > \max_{c} \left( \frac{1}{|\Omega_{c}|} \sum_{d \in \mathcal{V}(c)} |\Gamma_{cd}| |\overline{u}_{cd}| \right).$$

**Remark 6.** The *Lagrange step* will usually be solved using an implicit time scheme. A quasi-Newton method is employed on a pressure-velocity system, where the Jacobian is exactly given (assuming frozen speed of sound), as the system solved is linear. Density and total energy are deduced after convergence of the solver. This approach is extensively detailed in [58,59,71]. The overall scheme is extended to second-order using a MUSCL-type scheme, as detailed in [71]. A parameter  $\beta$  is introduced in the gradient limiter of the volume fraction, such that  $\beta = 1$  gives a classical MUSCL-type scheme while  $\beta = 2$  gives a compressive reconstruction method that sharpen the numerical interface between fluids. Details of the seconder order schemes can be found in [70].

#### 4. Low-Mach regime

The low-Mach regime corresponds to the incompressible limit of the equations. It is well known in the literature that Godunov-type schemes do not asymptotically converge to this limit [45,44,66,31,32]. If the mesh cannot be sufficiently refined, it is necessary to use a low-Mach correction. This is possible by making the scheme *all-Mach*, i.e., capable of accurately treating both low Mach number, quasi-incompressible zones, and high Mach number, supersonic or hypersonic zones.

The strategy used to solve the two-phase system is based on the Lagrange-Transport operator decomposition of Chalons et al. [20,21]. This decomposition is solved in two successive steps, the Lagrange step and the Transport step. The behavior of this strategy within the framework of the Euler equations concerning the low-Mach regime has already been studied in [20,21]. Excessive dissipation of the pressure flux in the Lagrange step is identified as the cause of the method's poor convergence. An asymptotic study of the numerical scheme shows that it is not possible to converge to the incompressible limit of the equations, i.e. constant density, zero divergence velocity field, and velocity time evolution equation. To rectify this shortcoming, a low-Mach correction is generally used, consisting of centering the pressure flux according to a local Mach number [44,66,31,32,20,21,58,59,77,78]. Another idea is to construct a numerical scheme for a system preconditioned by a cleverly chosen matrix. This method, introduced by Turkel [75,76], when applied to a Godunov-type scheme, amounts to modifying the slopes of the Riemann solver as a function of the Mach number. It has the disadvantage of requiring the definition of a threshold limit on this Mach number so as not to impact the time step too much when using an explicit approach.

#### 4.1. One-dimensional scheme

The low-Mach correction used for the one-dimensional scheme is inspired by that used by Chalons et al. [20,21]. The numerical scheme used to solve the two-phase system in its Lagrangian form with capillary effects is a Godunov-type scheme based on a four-state Riemann solver. The correction used consists in slightly modifying the Riemann solver (38)

$$W(x/t; V_l, V_r) = \begin{cases} V_l & \text{if } x/t \le -\lambda^- v_l, \\ V_l^* = V_l + \phi_R_- + LM_- & \text{if } -\lambda^- v_l < x/t \le 0, \\ V_r^* = V_r - \phi_+ R_+ - LM_+ & \text{if } 0 < x/t \le \lambda^+ v_r, \\ V_r & \text{if } \lambda^+ v_r < x/t, \end{cases}$$
(45)

where

$$\begin{array}{lcl} R_{\pm} & = & (-1,0,\pm\lambda_{\pm},p_{1-\alpha}\pm\lambda_{\pm}u_{\alpha},0), \\ LM_{\pm} & = & \frac{\lambda_{\mp}(\theta-1)\Delta u}{\lambda_{-}+\lambda_{+}}(0,0,1,\overline{u},0), \\ \phi_{\pm} & = & \frac{\Delta p - \tilde{\kappa}\Delta z \pm \lambda_{\mp}\Delta u}{\lambda_{-}\lambda_{+}+\lambda_{+}^{2}}, \end{array}$$

with

$$\theta = \min\left(1, \frac{|\overline{u}|}{\max(c_l, c_r)}\right).$$

The correction used amounts to modifying the pressure flux  $\overline{p}$  simply by

$$\overline{p}^{\theta} = \frac{\lambda^{-}p_{r} + \lambda^{+}p_{l}}{\lambda^{+} + \lambda^{-}} - \frac{\theta\lambda^{+}\lambda^{-}\Delta u}{\lambda^{+} + \lambda^{-}} - \frac{\overline{\kappa}\Delta z}{2} \frac{\lambda^{-} - \lambda^{+}}{\lambda^{+} + \lambda^{-}}.$$

We can see that the low-Mach correction only comes into play in the subsonic regime when the local Mach number is lower than 1. Moreover, Laplace's law is still a strong equilibrium solution of the Riemann solver.

**Proposition 9.** Laplace's law is a strong equilibrium solution of the Riemann solver (45).

**Proof.** We can immediately see that the  $LM_{\pm}$  term related to the low-Mach correction cancels out when u is constant. Furthermore, if  $p - \sigma \kappa z = cste$ , then  $\phi_{\pm} = 0$ . This amounts to  $V_l^* = V_l$  and  $V_r^* = V_r$ , which is the definition of a strong equilibrium.

# 4.2. Multi-dimensional scheme

The multi-dimensional scheme can be corrected in several ways. In recent years, many authors have proposed low-Mach corrections for compressible models. To our knowledge, Labourasse's work [49] is the only one to address the case of Lagrangian schemes. This correction has been applied in the context of Lagrangian hydrodynamics with surface tension in [26]. However, one class of multi-dimensional schemes seems capable of dealing directly with low Mach number flows. This is the case of Corot's recently developed Lagrangian scheme [27], for which pressure fluxes are based on nodal pressure. The good behavior of Corot's scheme in the low-Mach regime had already been noted in [49]. Interestingly, the dissipation of this nodal pressure takes the form of a discrete nodal divergence, which cancels out in the low-Mach regime. This idea was used by Barsukow [8] to build a multi-dimensional solver in

the Cartesian case, and employed in [27] to reduce the dissipation of the GLACE [35] and EUCCLHYD [52] schemes. One of the corrections proposed here and developed in [70] is also based on this observation. Moreover, it was asserted in [70] that in the Cartesian case without surface tension, when  $\theta_n = 0$ , the (46), (47) scheme becomes exactly the multi-dimensional low-Mach scheme of Barsukow [8]. This was recently proved in [9] for linear acoustics, where a multi-dimensional Godunov-type scheme with nodal pressure was employed. The same scheme with a nodal pressure was extended in [29] for the Euler system.

The first idea is to replace the multi-dimensional pressure flux with a convex combination of the multi-dimensional flux and the one-dimensional flux corrected according to the local Mach number. This correction still maintains a well-balanced scheme. However, as the pressure flux becomes one-dimensional, the multi-dimensional character of the scheme is lost. Thus, it is more interesting to explore a correction that allows us to have a multi-dimensional low-Mach scheme. We can draw inspiration from the correction proposed by [49] and used in [26].

First, we define  $q_n$  as a quantity at node n such that

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( \frac{1}{\lambda_l} + \frac{1}{\lambda_r} \right) q_n = \sum_{c \in C(n)} \sum_{f \in SF(nc)} |\Gamma_{nf}| \left( \frac{p_c - \sigma \kappa_c z_c}{\lambda^-} + \boldsymbol{u}_c \cdot \boldsymbol{n}_f \right). \tag{46}$$

Recall that the notation  $\lambda^-$  corresponds to the value of the slope associated with the left state of the face f, oriented by the outgoing normal  $n_f$  from c to d. This definition is based on the construction of the nodal velocity  $u_n$ . Indeed, we saw in the previous section that the nodal velocity is a combination of the one-dimensional velocity flux on the faces attached to the node. Thus, the second member of (46) corresponds to a one-dimensional pressure flux. From here, we define the new pressure flux as

$$\overline{p}_l = \theta_n \left( p_l - \lambda^- (\boldsymbol{u}_n - \boldsymbol{u}_l) \cdot \boldsymbol{n}_f \right) + (1 - \theta_n) \left( q_n + \sigma \kappa_n z_c \right), \tag{47}$$

where

$$\theta_n = \min\left(1, \frac{||\boldsymbol{u}_n||}{c_{\max}}\right),$$

and  $c_{\max}$  corresponds to the maximum speed of sound in the nodal neighborhood. The pressure flux thus defined is a convex combination of the original multi-dimensional pressure flux and a multi-dimensional flux for resolving low-Mach flows with surface tension. The parameter  $\kappa_n$  corresponds to a nodal curvature. If the curvature  $\overline{\kappa}$  is defined at nodes, then  $\overline{\kappa} = \kappa_n$ . In the opposite case, where the curvature  $\overline{\kappa}$  has been defined at the faces, the nodal curvature  $\kappa_n$  is given by (44). The nodal quantity  $q_n$  is therefore composed of a term centered at  $q = p - \sigma \kappa z$  and a velocity diffusion term. This  $\mathcal D$  diffusion term of the nodal pressure is given by

$$\mathcal{D} = \frac{1}{\sum_{f \in SF(n)} |\Gamma_{nf}| \left(\frac{1}{\lambda_l} + \frac{1}{\lambda_r}\right)} \sum_{c \in C(n)} \sum_{f \in SF(nc)} |\Gamma_{nf}| \boldsymbol{u}_c \cdot \boldsymbol{n}_f.$$

It is easy to see that the  $\mathcal D$  diffusion term is a discretization of the discrete divergence of the velocity at node n. We know that in the low-Mach regime, the asymptotic limit of the system as the Mach number tends towards 0 gives a velocity field with zero divergence. Thus, the diffusion term also cancels out, and the dimensionless scheme of **the Lagrange step** is consistent with a  $\mathcal O(\Delta x)$  truncated error on the velocity equation. The theoretical proof of this result will not be given here. To prove that  $\mathcal D$  is a first-order approximation to divergence at the node, we can use the same tools as in [34,49]. It is also possible to construct the nodal quantity  $q_n$  without the velocity terms. We then have

$$\sum_{f \in SF(n)} |\Gamma_{nf}| \left( \frac{1}{\lambda_l} + \frac{1}{\lambda_r} \right) q_n = \sum_{c \in C(n)} \sum_{f \in SF(nc)} |\Gamma_{nf}| \left( \frac{p_c - \sigma \kappa_c z_c}{\lambda^-} \right). \tag{48}$$

We can then show that the dimensioned scheme (47) with nodal pressure (46) or (48) is consistent with a  $\mathcal{O}(\Delta x)$  truncated error on the velocity equation. We will now look at the proposed correction for equilibrium solutions.

Proposition 10. Laplace's law is a strong equilibrium solution for the scheme (47), (46), and (44).

**Proof.** When the velocity u, curvature  $\kappa$  and quantity  $q = p - \sigma \kappa z$  are constant, we have  $u_n = u$ . Moreover,  $q_n = q$  so  $q_n + \sigma \kappa_n z_c = p_c$ . Thus,  $\overline{p}_l = p_l$  and  $\overline{p}_r = p_r$ . Injecting these relations into the definitions of (43), we then have  $v_s^* = v_s$  and  $E_s^* = E_s$ , s = l, r.  $\square$ 

This scheme retains the property of being multi-dimensional, consistent with the asymptotic limit of the low-Mach regime, and preserves stationary solutions. By defining the nodal quantities with the same slopes, i.e.  $\forall f \in SF(nc)$ ,  $\lambda^- = \lambda^+$ , we then find Labourasse's low-Mach correction [49], used notably in [26].

# 5. Lagrange-transport splitting scheme applied to the five-equation system with surface tension and viscous dissipation

For the multi-dimensional scheme with surface tension in the form CSF (32), we saw in Sec. 3.3.4 that two choices were possible for defining curvature: at faces or at nodes. Our numerical experiments have shown that the choice of curvature at the nodes is numerically more diffusive than the choice at the faces. For example, in the case of the oscillation of an ellipsoidal droplet presented

Table 1
Water droplet at equilibrium: initial data.

|               | P (Pa)                             | $\rho$ (kg.m <sup>-3</sup> ) | <b>u</b> (m.s <sup>-1</sup> ) | z | γ | π (Pa)              |
|---------------|------------------------------------|------------------------------|-------------------------------|---|---|---------------------|
| liquid<br>gas | 10 <sup>5</sup><br>10 <sup>5</sup> | 10 <sup>3</sup>              | •                             | 1 |   | $6.8 \times 10^{8}$ |

![](_page_24_Figure_4.jpeg)

Fig. 6. Water droplet at equilibrium: evolution of total kinetic energy in the domain.

in the Sec. 5.2, the amplitude of the oscillations was smaller with the choice of the nodes. So, unless otherwise stated, curvature will be defined at the faces. Curvature is always estimated at cell centers from the volume fraction moments, calculated using a quadratic least squares method, using node-based stencil, without weights [70]. Thus, we have

$$\begin{split} \kappa_c &= -\nabla \cdot \left( \frac{\boldsymbol{\nabla} z}{|\boldsymbol{\nabla} z|} \right) \\ &= \left( \frac{2\partial_x z \partial_y z \partial_{xy} z - \partial_x^2 z \partial_{yy} z - \partial_y^2 z \partial_{xx} z}{(\partial_x^2 z + \partial_y^2 z)^{3/2}} \right). \end{split}$$

The test cases will use either wall boundary conditions or periodic conditions. Fluxes at the boundaries are calculated using ghost cells, in which the quantities required to evaluate the fluxes are imposed. Since the mesh is structured and all boundaries considered in this work are aligned with the cartesian referential axes, this procedure is conservative and sufficient.

# 5.1. A drop of water at equilibrium

This two-dimensional academic case will allow us to test the ability of numerical schemes to preserve equilibrium solutions. We initially place a drop of water of radius  $R=1.8\times 10^{-5}$  m at the center of a domain of size  $\Omega=[0,0.9\times 10^{-5}]\times[0,0.9\times 10^{-5}]$  m<sup>2</sup> of air at rest. Initial data are given in the Table 1. The surface tension coefficient is  $\sigma=0.072$  N.m<sup>-1</sup>. The domain is divided into  $128\times 128$  cells. Since the curvature of the drop is non-zero, the surface tension must induce a pressure jump at the interface, which must respect Laplace's law  $\Delta p=\sigma\kappa\Delta z$ . We then propose to study the amplitude of the parasitic currents via the total kinetic energy in the domain, as well as to calculate the error on the pressure jump, for 6 different schemes: one-dimensional and multi-dimensional schemes in CSF form with exact curvature imposed and one-dimensional and multi-dimensional schemes in CSF form with numerically calculated curvature. The final time is  $t_f=5\times 10^{-5}$  s. Calculations are performed with the implicit scheme, first order in space, without Mach low correction. Wall conditions are used on all boundaries.

To calculate the volume fraction derivatives, the cells crossed by the circle defining the drop are initialized as mixing meshes, for which the volume fraction between 0 and 1 in a mixing mesh is calculated by subdividing this mesh into  $100 \times 100$  worth of sub-cells, estimating their number in the circle and dividing this number by the total number of sub-cells.

The Fig. 6 shows the evolution of kinetic energy in the domain over time. Firstly, it can be seen that the CSF approach preserves the equilibrium solutions: the exact curvature (CSF uniD +  $\kappa_{th}$  & CSF multiD +  $\kappa_{th}$ ) allows the total kinetic energy to tend towards the machine error. Next, we observe that if the curvature is not calculated exactly, the one-dimensional (CSF uniD) scheme does not converge to a stationary solution, unlike the multi-dimensional (CSF multiD) scheme. Numerically, we observe a velocity field completely aligned with the mesh, leading to instabilities at the interface. These are completely absent with the multi-dimensional scheme, and the velocity field is also completely different. To illustrate this, we plot Fig. 7 the velocity amplitude at  $t = 10^{-5}$  s, before the appearance of these instabilities, for the one-dimensional and multi-dimensional schemes in CSF form. While the velocity field is truly multi-dimensional for the multi-dimensional scheme, the one-dimensional scheme has a velocity field aligned with the mesh. This disturbs the interface and causes major instabilities.

![](_page_25_Figure_2.jpeg)

**Fig. 7.** Equilibrium water drop: velocity amplitude at = 10−5 s for oneand multi-dimensional schemes in CSF form. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)

![](_page_25_Figure_6.jpeg)

**Fig. 8.** Water droplet at equilibrium: evolution of the difference between theoretical and calculated pressure.

To check that the right pressure jump has been obtained, we plot on the Fig. 8 the deviation such that

$$\xi_p = \frac{|[p_{sim}] - [p_{th}]|}{[p_{th}]},$$

where

$$[p_{sim}] = \frac{\sum_{c} pz}{\sum_{c} z} - \frac{\sum_{c} p(1-z)}{\sum_{c} (1-z)} \quad \text{and} \quad [p_{th}] = \sigma \sqrt{\frac{\pi}{\sum_{c} z |\Omega_{c}|}}.$$

To compare the deviations before the onset of these instabilities, we look at the results up to time = 10−5 s. In general, the deviations order are about a percent (4% max). Laplace's law is therefore well reproduced by both schemes. Given that the pressure jump estimate is calculated by taking into account all the cells in the domain, the one-dimensional and multi-dimensional schemes in CSF form with exact curvature (CSF uniD+ *ℎ* & CSF multiD+ *ℎ*) do not give an exact 0% deviation on the Fig. 8. This would require the use of a threshold on the estimate of [] to exclude mixing cells.

*Extension to second order and low-Mach correction* To check that the extension of the oneand multi-dimensional schemes to second order, together with the low Mach correction, preserves the stationary solutions, we now impose the theoretical pressure jump Δ = 4000 Pa and the exact curvature = 1∕ directly on the water droplet. Indeed, since the domain is closed and there is no viscous dissipation, initializing the droplet without the pressure jump would cause waves to appear in the domain which, bouncing around without ever being damped, would make it impossible to obtain a stationary solution. Moreover, since we are directly imposing

![](_page_26_Figure_2.jpeg)

Fig. 9. Equilibrium water droplet: evolution of total kinetic energy for the one-dimensional and multi-dimensional scheme with the centered non-conservative term (o2) and without the centered non-conservative term (o1).

 Table 2

 Oscillation of an ellipsoidal droplet : initial data.

|        | P (Pa)          | $\rho$ (kg.m <sup>-3</sup> ) | <b>u</b> (m.s <sup>-1</sup> ) | z. | γ   | π (Pa)             |
|--------|-----------------|------------------------------|-------------------------------|----|-----|--------------------|
| liquid | 10 <sup>5</sup> | $10^{2}$                     | 0                             | 1  | 2.4 | $1. \times 10^{7}$ |
| gas    | $10^{5}$        | 1                            | 0                             | 0  | 1.4 | 0                  |

the stationary solution, for which the velocity is zero throughout the domain, we're using the explicit scheme. Indeed, the iterative process required to solve **the Lagrange step** in the implicit approach does not in practice converge quickly to a zero-velocity solution. This is also caused by the time step dependency on the velocity material, which is zero.

The simulation is run up to  $t = 5 \times 10^{-8}$  s, which is a sufficiently long final time to observe any imbalance in the schemes. For the one-dimensional and multi-dimensional second-order schemes with low Mach correction, the total kinetic energy at the final time is less than  $10^{-18}$  J, which means that the schemes are in equilibrium. Second-order equilibrium is ensured in particular by taking into account the second-order centered terms of the non-conservative product, here is the surface tension in CSF form. Neglecting these terms does not preserve Laplace's law, and the total kinetic energy increases substantially up to the final time. This result is illustrated in Fig. 9.

# 5.2. Oscillation of an ellipsoidal droplet

This two-dimensional academic case was introduced in [64], and studied in [54,26,25,55]. An elliptical droplet is positioned at the center of a square domain of size  $\Omega = [0,1] \times [0,1]$  m filled with gas at rest. The ellipse is defined by

$$\frac{(x-0.5)^2}{0.2^2} + \frac{(y-0.5)^2}{0.12^2} = 1.$$

The ellipse, considered liquid, is modeled by the stiffened gas equation of state, while the perfect gas equation of state represents air. Initial data are given in the Table 2. The surface tension coefficient is  $\sigma=340~\rm N.m^{-1}$ . The simulation time is  $t_f=0.5~\rm s.$  As in the previous case, wall conditions are used on all four boundaries. The mesh is composed of  $128\times128~\rm cells$ . The non-uniform curvature of the droplet induces a periodic motion of the interface. Initially, the potential energy associated with surface tension is maximal, while kinetic energy is zero. The transfer of potential energy into kinetic energy will then deform the ellipse into a sphere, for which the kinetic energy is maximum, which will then deform into the same ellipse as before. This oscillation process, with no dissipative effects, is expected to last indefinitely, and only the numerical diffusion of the scheme should attenuate the amplitude of the oscillations. Périgaud [64] uses the modified Rayleigh formula [65,37] to analytically estimate the oscillation period as a function of the densities of the two fluids, the radius R of the drop at rest and the surface tension coefficient  $\sigma$ . This relationship is given by

$$\omega^2 = (o^3 - o) \frac{\sigma}{(\rho_l + \rho_o)R^3}$$
, with  $\omega = \frac{2\pi}{T}$ ,

with o the oscillation mode equals 2. If we estimate, by conservation of its volume, that the ellipse has an average radius R = 0.15 m, then we have a theoretical period of 85 ms.

First, we will compare the one-dimensional and multi-dimensional schemes for different resolutions of the Lagrange step, namely:

- the first-order resolution (denoted by o1 in the legend of the Fig. 10),
- the first-order resolution with low Mach correction (lm+o1),
- the second-order resolution (o2),

![](_page_27_Figure_2.jpeg)

**Fig. 10.** Oscillation of an ellipsoidal drop: evolution of the volumetric kinetic energy in the domain for oneand multi-dimensional schemes, where the Lagrange step is solved at order one (o1), at order one with low Mach correction (lm+o1), at order two (o2) and at order two with low Mach correction (lm+o2).

**Table 3** Oscillation of an ellipsoidal droplet: period of numerically observed oscillations.

|       | 1D      | 2D      |
|-------|---------|---------|
| lm+o1 | 0.090 s | 0.090 s |
| lm+o2 | 0.089 s | 0.090 s |

• the second-order resolution with low Mach correction (lm+o2).

**The Transport step** is always solved at second-order with = 2 to limit interface diffusion [\[71,70\]](#page-31-0) (see Remark [6\)](#page-21-0). We do not use the second-order correction on the non-conservative term at the second order on **the Lagrange step**. Indeed, it has been observed that the iterative process convergence deteriorates using a second-order scheme when this term is considered. This is not a problem here, as we don't necessarily want to be well-balanced in such a case, since in theory, the drop should oscillate indefinitely, without ever returning to equilibrium. The results on the Fig. 10 show the evolution of the total volumetric kinetic energy in the domain for the different schemes. First, it can be seen that the first-order and second-order schemes without low-Mach correction fail to reproduce the phenomenon of periodic droplet oscillation. At second-order without low-Mach correction, the results are worse. Numerically, this is due to the use of = 2, a choice that tends to sharpen the interface and greatly reduce the amplitude of oscillations. With = 1, we find the results of the literature [\[64,54](#page-31-0)[,26,25](#page-30-0)[,55\]](#page-31-0), namely that the damping of oscillations is preserved over several periods. In general, the low-Mach correction preserves the amplitude of oscillations very well. Oscillations are barely attenuated when the second-order scheme is used in addition to low-Mach correction. Generally, the multi-dimensional scheme is more diffusive than the one-dimensional one. For such a case, we can see that the dissipation of the scheme is responsible for the damping of oscillations and that the use of a low-Mach correction with a high-order scheme is necessary to recover the theoretical behavior of the drop.

The Table 3 gives the oscillation period obtained for the four schemes. This is measured as the average of the 5 oscillation periods observed in the figures, starting from the first and last maximum. The average period is virtually the same for all schemes. The deviation from the theoretical period is less than 6%. The one-dimensional and multi-dimensional schemes are therefore capable of reproducing dynamic phenomena linked to surface tension, provided a low-Mach correction is employed.

### *5.3. Viscous damping of a capillary wave*

In this test case, we are interested in the attenuation of a capillary wave. The linear theory of small oscillations of an interface between two viscous fluids with the same viscosity was studied theoretically by [\[63](#page-31-0)], who gave an analytical solution for the case where the kinematic viscosities of the two fluids are equal, i.e. = . For this test case, studied in particular in [\[61](#page-31-0)[,11\]](#page-30-0), we consider two fluids, separated by a sinusoidal interface.

$$a(\mathbf{x}) = a_0 \cos(k\mathbf{x}),$$

where <sup>0</sup> is the initial amplitude of the oscillations and is the wave number (see Fig. [11\)](#page-28-0). When <sup>0</sup> *<<* 2, the time evolution of the oscillation amplitude is given by Prosperetti [\[63\]](#page-31-0)

![](_page_28_Figure_2.jpeg)

Fig. 11. Viscous damping of a capillary wave: geometric description of the case.

**Table 4**Viscous damping of a capillary wave: initial data.

|       | $\rho$ (kg.m <sup>-3</sup> ) | p (Pa)          | $\mu (kg.m^{-1}.s^{-1})$    | γ   | $\pi$ (Pa)          |
|-------|------------------------------|-----------------|-----------------------------|-----|---------------------|
| Water | $10^{3}$                     | 10 <sup>5</sup> | $1.68495882 \times 10^{2}$  | 4.4 | $6.8 \times 10^{8}$ |
| Air   | 1                            | $10^{5}$        | $1.68495882 \times 10^{-1}$ | 1.4 | 0                   |

$$a(x,t) = \frac{4(1-4\beta)\epsilon^2}{8(1-4\beta)\epsilon^2 + \omega_0^2} a_0 \operatorname{erfc}(\sqrt{\epsilon t}) + \sum_{i=1}^4 \frac{z_i}{Z_i} \left( \frac{\omega_0^2 a(x,0)}{z_i^2 - \epsilon} \right) \exp\left((z_i^2 - \epsilon)t\right) \operatorname{erfc}(\mathcal{R}(z_i)\sqrt{t}),$$

with  $\beta = \rho_1 \rho_2 / (\rho_1 + \rho_2)^2$ ,  $\omega_0 = \sigma \kappa / (\rho_1 + \rho_2)$ ,  $\epsilon = \nu k^2$ ,  $Z_i = \prod_{j=1, j\neq i}^4 (z_j - z_i)$  and  $z_i$ , i = 1, 4 the complex conjugate roots of the equation

$$z^{4} - 4\beta\sqrt{\epsilon}z^{3} + 2(1 - 6\beta)\epsilon z^{2} + 4(1 - 3\beta)\epsilon^{3/2}z + (1 - 4\beta)\epsilon^{2} + \omega_{0} = 0.$$

The notation  $\mathcal{R}(z_i)$  designates the real part of  $z_i$ , and erfc is the complementary error function. Based on the work of Popinet and Zaleski [61], we take the data from Table 4, with a perfect gas equation of state for air, and a stiff-gas equation for liquid. The surface tension coefficient is  $\sigma = 13.556 \times 10^3 \text{ N.m}^{-1}$ . The domain is rectangular,  $[0, L] \text{ m} \times [0, 2L] \text{ m}$  with  $L = 2\pi$ . The amplitude  $a_0$  is such that the ratio  $a_0/2L = 10^{-2}$  is small enough to fall within the linear theory of small oscillations. The Ohnesorge number, the ratio of viscous forces to surface tension and inertial forces, is given by

$$Oh = \frac{\mu}{\sqrt{\rho\sigma L}}$$

Here, we have  $Oh = 1/\sqrt{3000}$  for the liquid and  $Oh = 1/\sqrt{3 \times 10^6}$  for the gas. The dimensioned kinematic viscosity is  $\epsilon = 6.472 \times 10^{-2}$  for both fluids. Heat conduction is assumed to be zero in this case. The final simulation time is  $t_f = 10$  s. Periodic boundary conditions are used on the left and right edges, while wall conditions are imposed at the top and bottom of the domain.

The five-equation system with viscous dissipation is solved as described in [59,70]. We will compare the evolution of the oscillation amplitude over time with the theoretical solution for different mesh sizes. The calculations are performed using the implicit Lagrange-Transport decomposition approach. **The Lagrange step** is always solved at first-order with low-Mach correction, while the Transport step is solved at second-order with the compressive limiter. Given the results obtained for the case of an ellipsoidal drop oscillation, this approach is sufficient to accurately reproduce oscillations over several periods. To resolve capillary waves more precisely, we take a time step such that  $\Delta t = \min \left( \Delta t_t, 50 \Delta t_c \right)$ , where  $\Delta t_t$  is the time step related to **the Transport step** and  $\Delta t_c$  is that related to surface tension. Adding capillary stress to the time step by a factor of 50 allows us to better resolve the dynamics of this case since more iterations will be made between each oscillation period. Indeed, using only the time step of the **transport step** would not allow us to accurately resolve each oscillation period.

Mesh convergence is performed. As the initialization, the amplitude is smaller than the size of a cell. We thus initialize the cells in which the sinusoidal interface is contained in them in the manner of the equilibrium Water Drop case. Again, we use  $100 \times 100$  subcells to estimate the volume fraction of the mixing cells. This produces a more accurate estimation of the curvature at the first iteration. It is necessary because, as the initial amplitude is very small, the interface is contained in only one cell. The results presented in Fig. 12 compare one-dimensional and multi-dimensional schemes, for different meshes. The results are in good agreement with

![](_page_29_Figure_2.jpeg)

**Fig. 12.** Viscous damping of a capillary wave: evolution of the amplitude of interface oscillations for oneand multi-dimensional schemes, for different mesh sizes.

the exact solution given by Prosperetti. The more refined the mesh, the closer the oscillation frequency and the damping ratio are to the theoretical solution.

# **6. Conclusion**

In this work, we have extended the notion of a Godunov-type scheme to the non-conservative framework. Our use of the EUC-CLHYD multi-dimensional scheme [\[52\]](#page-31-0) in the conservative case, i.e. without surface tension, prompted us to use Gallice's notion of a multi-dimensional Godunov-type scheme et al. [\[42](#page-31-0)] to construct a non-conservative multi-dimensional Godunov-type scheme. The one-dimensional and multi-dimensional schemes based on a simple Riemann solver [\[38,40,39,41](#page-30-0)] are positive, entropic, and conditionally stable over the time step. They have been used to construct under Lagrangian form a one-dimensional numerical scheme and its multi-dimensional extension for the reduced five-equation system with surface tension in CSF form. We call this a multidimensional scheme because the flux construction stencil is based on a nodal neighborhood, unlike the one-dimensional scheme where the numerical flux stencil consists of neighbors at faces. They are well-balanced, i.e. they preserve the stationary solutions linked to surface tension in the case of CSF modeling. The use of the Lagrange-Transport splitting method extensively detailed in [\[20,21](#page-30-0)[,58,59,71](#page-31-0)] brings us back to the Eulerian framework. Finally, a set of test cases enabled us to compare and validate the numerical schemes developed here, as well as the low Mach correction employed for each of them. The use of a truly multi-dimensional scheme seems advantageous in this kind of two-phase flow as it reduces parasitic currents and produces a really multi-dimensional velocity field. Indeed, surface tension is a truly multi-dimensional effect and pathologies can arise when it is used within a onedimensional framework, as highlighted by the first test case. Being more diffusive than the one-dimensional scheme, the extension to second-order as well as a low Mach correction is essential to capture all the phenomena involved.

## **CRediT authorship contribution statement**

**Lucas Tallois:** Writing -- review & editing, Writing -- original draft, Software, Investigation, Conceptualization. **Simon Peluchon:** Writing -- review & editing, Supervision, Investigation, Conceptualization. **Gérard Gallice:** Writing -- review & editing, Investigation, Conceptualization. **Philippe Villedieu:** Writing -- review & editing, Supervision.

#### **Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

#### **Data availability**

Data will be made available on request.

# **References**

- [1] R. Abgrall, S. Karni, A comment on the computation of non-conservative products, J. Comput. Phys. 229 (2010) 2759--2763, [https://doi.org/10.1016/j.jcp.2009.](https://doi.org/10.1016/j.jcp.2009.12.015) [12.015](https://doi.org/10.1016/j.jcp.2009.12.015), [https://www.sciencedirect.com/science/article/pii/S0021999109006949.](https://www.sciencedirect.com/science/article/pii/S0021999109006949)
- [2] K. Ait-Ameur, S. Kokh, M. Massot, M. Pelanti, T. Pichard, An acoustic-transport splitting method for the barotropic Baer-Nunziato two-phase flow model, ESAIM Proc. Surv. 72 (2023) 93--116, [https://doi.org/10.1051/proc/202372093.](https://doi.org/10.1051/proc/202372093)

- [3] G. Allaire, S. Clerc, S. Kokh, A [five-equation](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib2706F718EE510989862917A04EE36987s1) model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577--616.
- [4] E. Audusse, F. Bouchut, M.O. Bristeau, R. Klein, B. Perthame, A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows, SIAM J. Sci. Comput. 25 (2004) 2050--2065, <https://doi.org/10.1137/S1064827503431090>.
- [5] E. Audusse, M.O. Bristeau, A 2d Well-balanced Positivity Preserving Second Order Scheme for Shallow Water Flows on Unstructured Meshes, Research Report RR-5260, INRIA, 2004, [https://inria.hal.science/inria-00070738.](https://inria.hal.science/inria-00070738)
- [6] M. Baer, J. Nunziato, A two-phase mixture theory for the [deflagration-to-detonation](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibFD1CCF94E905CFA5B78AC32C921E8F54s1) transition (DDT) in reactive granular materials, Int. J. Multiph. Flow 12 (1986) [861--889.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibFD1CCF94E905CFA5B78AC32C921E8F54s1)
- [7] D. Balsara, Multidimensional HLLE Riemann solver: application to Euler and magnetohydrodynamic flows, J. Comput. Phys. 229 (2010) 1970--1993, [https://](https://doi.org/10.1016/j.jcp.2009.11.018) [doi.org/10.1016/j.jcp.2009.11.018,](https://doi.org/10.1016/j.jcp.2009.11.018) <https://www.sciencedirect.com/science/article/pii/S0021999109006378>.
- [8] W. Barsukow, Truly multi-dimensional all-speed schemes for the Euler equations on Cartesian grids, J. Comput. Phys. 435 (2021) 110216, [https://doi.org/10.](https://doi.org/10.1016/j.jcp.2021.110216) [1016/j.jcp.2021.110216](https://doi.org/10.1016/j.jcp.2021.110216), <https://www.sciencedirect.com/science/article/pii/S002199912100111X>.
- [9] W. Barsukow, R. Loubère, P.H. Maire, A node-conservative [vorticity-preserving](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib466DDDD4817852091A24F075F88403DAs1) finite volume method for linear acoustics on unstructured grids, Math. Comput. [\(2024\).](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib466DDDD4817852091A24F075F88403DAs1)
- [10] D. Benson, Computational methods in Lagrangian and Eulerian hydrocodes, Comput. Methods Appl. Mech. Eng. 99 (1992) 235--394, [https://doi.org/10.1016/](https://doi.org/10.1016/0045-7825(92)90042-I) [0045-7825\(92\)90042-I](https://doi.org/10.1016/0045-7825(92)90042-I), [https://www.sciencedirect.com/science/article/pii/004578259290042I.](https://www.sciencedirect.com/science/article/pii/004578259290042I)
- [11] G. Blanchard, Modélisation et simulation [multi-échelles](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibF1963A723F453B5C63E6F4D1B95CE1E4s1) de l'atomisation d'une nappe liquide cisaillée, Thèse de doctorat, Université de Toulouse, 2015.
- [12] F. Bouchut, Nonlinear Stability of Finite Volume Methods for Hyperbolic Conservation Laws and [Well-Balanced](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib3AF2297FC2CEA908E04DF18CC97F3303s1) Schemes for Sources, Frontiers in Mathematics, vol. 2/2004, [Birkhäuser,](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib3AF2297FC2CEA908E04DF18CC97F3303s1) Basel, 2004.
- [13] J. Brackbill, D. Kothe, C. Zemach, A continuum method for modeling surface tension, J. Comput. Phys. 100 (1992), [https://doi.org/10.1016/0021-9991\(92\)](https://doi.org/10.1016/0021-9991(92)90240-Y) [90240-Y,](https://doi.org/10.1016/0021-9991(92)90240-Y) <https://www.sciencedirect.com/science/article/pii/002199919290240Y>.
- [14] B. Braconnier, Modélisation numérique d'écoulements multiphasiques pour des fluides compressibles, non miscibles et soumis aux effets capillaires, Thèse de doctorat, Université Sciences et Technologies-Bordeaux I, 2007, <http://www.theses.fr/2007BOR13381>.
- [15] D. Burton, T. Carney, N. Morgan, S. Sambasivan, M. Shashkov, A cell-centered Lagrangian Godunov-like method for solid dynamics, in: Numerical Methods for Highly Compressible Multi-Material Flow Problems, Comput. Fluids 83 (2013) 33--47, [https://doi.org/10.1016/j.compfluid.2012.09.008,](https://doi.org/10.1016/j.compfluid.2012.09.008) [https://](https://www.sciencedirect.com/science/article/pii/S0045793012003647) [www.sciencedirect.com/science/article/pii/S0045793012003647](https://www.sciencedirect.com/science/article/pii/S0045793012003647).
- [16] M. Castro, J. Gallardo, C. Parés, High order finite volume schemes based on reconstruction of states for solving hyperbolic systems with nonconservative products. Applications to shallow-water systems, Math. Comput. 75 (2006) 1103--1134, [http://www.jstor.org/stable/4100267.](http://www.jstor.org/stable/4100267)
- [17] M. Castro, A. Kurganov, T. Morales de Luna, Path-conservative central-upwind schemes for nonconservative hyperbolic systems, ESAIM: Math. Model. Numer. Anal. 53 (2019), [https://doi.org/10.1051/m2an/2018077.](https://doi.org/10.1051/m2an/2018077)
- [18] M. Castro, P. LeFloch, M. Muñoz-Ruiz, C. Parés, Why many theories of shock waves are necessary: convergence error in formally path-consistent schemes, J. Comput. Phys. 227 (2008) 8107--8129, [https://doi.org/10.1016/j.jcp.2008.05.012,](https://doi.org/10.1016/j.jcp.2008.05.012) <https://www.sciencedirect.com/science/article/pii/S0021999108002842>.
- [19] C. Chalons, F. Coquel, A new comment on the computation of non-conservative products using Roe-type path conservative schemes, J. Comput. Phys. 335 (2017) 592--604, <https://doi.org/10.1016/j.jcp.2017.01.016>, [https://www.sciencedirect.com/science/article/pii/S0021999117300268.](https://www.sciencedirect.com/science/article/pii/S0021999117300268)
- [20] C. Chalons, M. Girardin, S. Kokh, An all-regime Lagrange-projection like scheme for the gas dynamics equations on unstructured meshes, Commun. Comput. Phys. 20 (2016) 188--233, [https://doi.org/10.4208/cicp.260614.061115a.](https://doi.org/10.4208/cicp.260614.061115a)
- [21] C. Chalons, M. Girardin, S. Kokh, An all-regime Lagrange-projection like scheme for 2D homogeneous models for two-phase flows on unstructured meshes, J. Comput. Phys. 335 (2017) 885--904, [https://doi.org/10.1016/j.jcp.2017.01.017,](https://doi.org/10.1016/j.jcp.2017.01.017) [https://www.sciencedirect.com/science/article/pii/S002199911730027X.](https://www.sciencedirect.com/science/article/pii/S002199911730027X)
- [22] C. Chalons, P. Kestener, S. Kokh, M. Stauffert, A large time-step and well-balanced Lagrange-projection type scheme for the shallow-water equations, Commun. Math. Sci. 15 (2016), <https://doi.org/10.4310/CMS.2017.v15.n3.a9>.
- [23] A. Chan, Innovative numerical schemes for 3D supersonic aerodynamics on unstructure mesh, Thèse de doctorat, Université Sciences et Technologies-Bordeaux I, 2022, <http://www.theses.fr/2022BORD0303/document>.
- [24] A. Chan, G. Gallice, R. Loubère, P.H. Maire, Positivity preserving and entropy consistent approximate Riemann solvers dedicated to the high-order MOOD-based finite volume discretization of Lagrangian and Eulerian gas dynamics, Comput. Fluids 229 (2021) 105056, <https://doi.org/10.1016/j.compfluid.2021.105056>, <https://www.sciencedirect.com/science/article/pii/S0045793021002206>.
- [25] P. Cordesse, Contribution to the study of combustion instabilities in cryotechnic rocket engines: coupling diffuse interface models with kinetic-based moment methods for primary atomization simulations, Thèse de doctorat, Université Paris-Saclay, 2020, <https://theses.hal.science/tel-02948195>.
- [26] T. Corot, P. Hoch, E. Labourasse, Surface tension for compressible fluids in ALE framework, J. Comput. Phys. 407 (2020) 109247, [https://doi.org/10.1016/j.](https://doi.org/10.1016/j.jcp.2020.109247) [jcp.2020.109247](https://doi.org/10.1016/j.jcp.2020.109247), [https://www.sciencedirect.com/science/article/pii/S0021999120300218.](https://www.sciencedirect.com/science/article/pii/S0021999120300218)
- [27] T. Corot, B. Mercier, A new nodal solver for the two dimensional Lagrangian hydrodynamics, J. Comput. Phys. 353 (2018) 1--25, [https://doi.org/10.1016/j.jcp.](https://doi.org/10.1016/j.jcp.2017.09.053) [2017.09.053](https://doi.org/10.1016/j.jcp.2017.09.053), <https://www.sciencedirect.com/science/article/pii/S002199911730726X>.
- [28] G. Dal Maso, P. LeFloch, F. Murat, Definition and weak stability of [nonconservative](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib2F6D7ACC383E8889667F5C4FCBF68395s1) products, J. Math. Pures Appl. 74 (1995) 483--548.
- [29] A. Del Grosso, W. Barsukow, R. Loubère, P.H. Maire, An asymptotic-preserving [multidimensionality-aware](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibE28F9D34FC1F3DAC25911A1F076CA912s1) finite volume numerical scheme for Euler equations, in: ICCFD12 - International Conference on [Computational](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibE28F9D34FC1F3DAC25911A1F076CA912s1) Fluid Dynamics, 2024.
- [30] A. Del Grosso, M. Castro, A. Chan, G. Gallice, R. Loubère, P.H. Maire, A well-balanced, positive, entropy-stable, and multi-dimensional-aware finite volume scheme for 2D shallow-water equations with unstructured grids, J. Comput. Phys. 503 (2024) 112829, <https://doi.org/10.1016/j.jcp.2024.112829>.
- [31] S. Dellacherie, Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number, J. Comput. Phys. 229 (2010) 978--1016, [https://](https://doi.org/10.1016/j.jcp.2009.09.044) [doi.org/10.1016/j.jcp.2009.09.044,](https://doi.org/10.1016/j.jcp.2009.09.044) <https://www.sciencedirect.com/science/article/pii/S0021999109005361>.
- [32] S. Dellacherie, J. Jung, P. Omnes, P.A. Raviart, Construction of modified [Godunov-type](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibABC897EEF6DE260EADEA9FBEF41C5ED5s1) schemes accurate at any Mach number for the compressible Euler system, Math. Models Methods Appl. Sci. 26 (2016) [2525--2615.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibABC897EEF6DE260EADEA9FBEF41C5ED5s1)
- [33] B. Desprès, Lois de Conservations Eulériennes, Lagrangiennes et Méthodes Numériques. Mathématiques et Applications, Springer Berlin, Heidelberg, 2010, <https://link.springer.com/book/10.1007/978-3-642-11657-5>.
- [34] B. Després, Weak consistency of the cell-centered Lagrangian GLACE scheme on general meshes in any dimension, Comput. Methods Appl. Mech. Eng. 199 (2010) 2669--2679, [https://doi.org/10.1016/j.cma.2010.05.010,](https://doi.org/10.1016/j.cma.2010.05.010) <https://www.sciencedirect.com/science/article/pii/S0045782510001593>.
- [35] B. Després, C. Mazeran, Lagrangian gas dynamics in two dimensions and Lagrangian systems, Arch. Ration. Mech. Anal. 178 (2005) 327--372, [https://doi.org/](https://doi.org/10.1007/s00205-005-0375-4) [10.1007/s00205-005-0375-4](https://doi.org/10.1007/s00205-005-0375-4).
- [36] B. Engquist, S. Osher, One-sided difference [approximations](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibCD1219CC289F2B5B9C135BF4E3E73C66s1) for nonlinear conservation laws, J. Comput. Phys. 36 (1981) 321--351, NR 20140805.
- [37] D. Fyfe, E. Oran, M. Fritts, Surface tension and viscosity with Lagrangian [hydrodynamics](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibE8EE3B2553F8EEF8B96996AFA2A6322Es1) on a triangular mesh, J. Comput. Phys. 76 (1988) 349--384.
- [38] G. Gallice, Schémas de type Godunov entropiques et positifs préservant les [discontinuités](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib15F66CDD8AE2032B641FECD72ECF96FEs1) de contact, C. R. Acad. Sci. Paris, Série I 331 (2000) 149--152.
- [39] G. Gallice, Numerical Approximation of Conservative or Nonconservative Non-linear Hyperbolic Systems. Habilitation à Diriger des Recherches, Université de Bordeaux I, 2002, [https://hal-cea.archives-ouvertes.fr/tel-01320526.](https://hal-cea.archives-ouvertes.fr/tel-01320526)
- [40] G. Gallice, Solveurs simples positifs et entropiques pour les systèmes hyperboliques avec terme source, C. R. Acad. Sci. Paris, Série I 334 (2002) 713--716, [https://](https://doi.org/10.1016/S1631-073X(02)02307-5) [doi.org/10.1016/S1631-073X\(02\)02307-5](https://doi.org/10.1016/S1631-073X(02)02307-5), <https://www.sciencedirect.com/science/article/pii/S1631073X02023075>.
- [41] G. Gallice, Positive and entropy stable [Godunov-type](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibBD82C862B4ECCA1439309572123B09B9s1) schemes for gas dynamics and MHD equations in Lagrangian or Eulerian coordinates, Numer. Math. 94 (2003) [673--713.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibBD82C862B4ECCA1439309572123B09B9s1)

- [42] G. Gallice, A. Chan, R. Loubère, P.H. Maire, Entropy stable and positivity preserving Godunov-type schemes for multidimensional hyperbolic systems on unstructured grid, J. Comput. Phys. 468 (2022) 111493, [https://doi.org/10.1016/j.jcp.2022.111493.](https://doi.org/10.1016/j.jcp.2022.111493)
- [43] D.P. Garrick, M. Owkes, J.D. Regele, A finite-volume HLLC-based scheme for compressible interfacial flows with surface tension, J. Comput. Phys. 339 (2017) 46--67, <https://doi.org/10.1016/j.jcp.2017.03.007>, [https://www.sciencedirect.com/science/article/pii/S0021999117301948.](https://www.sciencedirect.com/science/article/pii/S0021999117301948)
- [44] H. Guillard, A. Murrone, On the behavior of upwind schemes in the low Mach number limit: II. Godunov type schemes, Comput. Fluids 33 (2004) 655--675, <https://doi.org/10.1016/j.compfluid.2003.07.001>, [https://www.sciencedirect.com/science/article/pii/S0045793003000781.](https://www.sciencedirect.com/science/article/pii/S0045793003000781)
- [45] H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, Comput. Fluids 28 (1999) 63--86, [https://doi.org/10.1016/S0045-](https://doi.org/10.1016/S0045-7930(98)00017-6) [7930\(98\)00017-6](https://doi.org/10.1016/S0045-7930(98)00017-6), [https://www.sciencedirect.com/science/article/pii/S0045793098000176.](https://www.sciencedirect.com/science/article/pii/S0045793098000176)
- [46] A. Harten, High resolution schemes for hyperbolic conservation laws, J. Comput. Phys. 49 (1983) 357--393, [https://doi.org/10.1016/0021-9991\(83\)90136-5](https://doi.org/10.1016/0021-9991(83)90136-5), <https://www.sciencedirect.com/science/article/pii/0021999183901365>.
- [47] A. Harten, P. Lax, B. van Leer, On upstream differencing and [Godunov-type](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib1B1F78EAA1B6CA641ABD1470D99D0DF8s1) schemes for hyperbolic conservation laws, SIAM Rev. 25 (1983) 35--61.
- [48] A. Kapila, R. Menikoff, J. Bdzil, S. Son, D.S. Stewart, Two-phase modeling of [deflagration-to-detonation](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibEEC576C524EC51EE05EB0DFEC0400DA6s1) transition in granular materials: reduced equations, Phys. Fluids 13 (2001) [3002--3024.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibEEC576C524EC51EE05EB0DFEC0400DA6s1)
- [49] E. Labourasse, A low-Mach correction for multi-dimensional finite volume shock capturing schemes with application in Lagrangian frame, Comput. Fluids 179 (2019) 372--393, <https://doi.org/10.1016/j.compfluid.2018.11.005>.
- [50] M. Latige, Simulation numérique de l'ablation liquide, Thèse de doctorat, Université Sciences et [Technologies-Bordeaux](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib2E1FB9A0AE30D32B0213B7C46DA89982s1) I, 2013.
- [51] R. LeVeque, Balancing source terms and flux gradients in high-resolution Godunov methods: the quasi-steady wave-propagation algorithm, J. Comput. Phys. 146 (1998) 346--365, [https://doi.org/10.1006/jcph.1998.6058,](https://doi.org/10.1006/jcph.1998.6058) <https://www.sciencedirect.com/science/article/pii/S0021999198960582>.
- [52] P.H. Maire, R. Abgrall, J. Breil, J. Ovadia, A cell-centered Lagrangian scheme for [two-dimensional](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib5B9E347C8A0D7CD4C471A525072F7F41s1) compressible flow problems, SIAM J. Sci. Comput. 29 (2007) [1781--1824.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib5B9E347C8A0D7CD4C471A525072F7F41s1)
- [53] J. Massoni, R. Saurel, B. Nkonga, R. Abgrall, Proposition de méthodes et modèles Eulériens pour les problèmes à interfaces entre fluides compressibles en présence de transfert de chaleur: Some models and Eulerian methods for interface problems between compressible fluids with heat transfer, Int. J. Heat Mass Transf. 45 (2002) 1287--1307, [https://doi.org/10.1016/S0017-9310\(01\)00238-1](https://doi.org/10.1016/S0017-9310(01)00238-1).
- [54] T. Nguyen, M. Dumbser, A path-conservative finite volume scheme for compressible multi-phase flows with surface tension, Appl. Math. Comput. 271 (2015) 959--978, [https://doi.org/10.1016/j.amc.2015.09.026,](https://doi.org/10.1016/j.amc.2015.09.026) <https://www.sciencedirect.com/science/article/pii/S009630031501262X>.
- [55] A. Panchal, S. Bryngelson, S. Menon, A seven-equation diffused interface method for resolved multiphase flows, J. Comput. Phys. 475 (2023) 111870, [https://](https://doi.org/10.1016/j.jcp.2022.111870) [doi.org/10.1016/j.jcp.2022.111870,](https://doi.org/10.1016/j.jcp.2022.111870) <https://www.sciencedirect.com/science/article/pii/S0021999122009330>.
- [56] C. Parés, Numerical methods for nonconservative hyperbolic systems: a theoretical framework, SIAM J. Numer. Anal. 44 (2006) 300--321, [https://doi.org/10.](https://doi.org/10.1137/050628052) [1137/050628052](https://doi.org/10.1137/050628052).
- [57] S. Peluchon, Approximation numérique et modélisation de l'ablation liquide, Thèse de doctorat, Université Sciences et Technologies-Bordeaux I, 2017, [http://](http://www.theses.fr/2017BORD0739) [www.theses.fr/2017BORD0739.](http://www.theses.fr/2017BORD0739)
- [58] S. Peluchon, G. Gallice, L. Mieussens, A robust implicit–explicit [acoustic-transport](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib1F1B7B8DDAF908335B831834B4898CEAs1) splitting scheme for two-phase flows, J. Comput. Phys. 339 (2017) 328--355.
- [59] S. Peluchon, G. Gallice, L. Mieussens, Development of numerical methods to simulate the melting of a thermal protection system, J. Comput. Phys. (2021) 110753, [https://doi.org/10.1016/j.jcp.2021.110753,](https://doi.org/10.1016/j.jcp.2021.110753) <https://www.sciencedirect.com/science/article/pii/S0021999121006483>.
- [60] A. Plessier, S. del Pino, B. Després, Implicit discretization of Lagrangian gas dynamics, ESAIM: Math. Model. Numer. Anal. 57 (2023) 717--743, [https://doi.org/](https://doi.org/10.1051/m2an/2022102) [10.1051/m2an/2022102](https://doi.org/10.1051/m2an/2022102), <https://hal.science/hal-04048418>.
- [61] S. Popinet, S. Zaleski, A front-tracking algorithm for accurate [representation](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib6F11151E3EA58F3474376CFC3DB1A4B2s1) of surface tension, Int. J. Numer. Methods Fluids (1999).
- [62] K.G. Powell, An approximate Riemann solver for [magnetohydrodynamics](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib6E5410FB7E55555CA1D66C69623AFDCBs1) (that works in more than one dimension), Technical Report, 1994.
- [63] A. [Prosperetti,](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibABBDE8FDE123F669702208A4563E1D8Fs1) Motion of two superposed viscous fluids, Phys. Fluids (1981).
- [64] G. Périgaud, R. Saurel, A compressible flow model with capillary effects, J. Comput. Phys. 209 (2005) 139--178, <https://doi.org/10.1016/j.jcp.2005.03.018>, <https://www.sciencedirect.com/science/article/pii/S0021999105001853>.
- [65] L. Rayleigh, On the capillary phenomena of jets, Proc. R. Soc. Lond. 29 (1879) 71--97, [https://doi.org/10.1098/rspl.1879.0015,](https://doi.org/10.1098/rspl.1879.0015) [https://royalsocietypublishing.](https://royalsocietypublishing.org/doi/abs/10.1098/rspl.1879.0015) [org/doi/abs/10.1098/rspl.1879.0015.](https://royalsocietypublishing.org/doi/abs/10.1098/rspl.1879.0015)
- [66] F. Rieper, A low-Mach number fix for Roe's approximate Riemann solver, J. Comput. Phys. 230 (2011) 5263--5287, <https://doi.org/10.1016/j.jcp.2011.03.025>, <https://www.sciencedirect.com/science/article/pii/S0021999111001689>.
- [67] P. Roe, Approximate Riemann solvers, parameter vectors, and difference schemes, J. Comput. Phys. 43 (1981) 357--372, [https://doi.org/10.1016/0021-9991\(81\)](https://doi.org/10.1016/0021-9991(81)90128-5) [90128-5](https://doi.org/10.1016/0021-9991(81)90128-5), <https://www.sciencedirect.com/science/article/pii/0021999181901285>.
- [68] A.B. de Saint-Venant, Théorie du mouvement non permanent des eaux avec applications aux crues des rivières et à [l'introduction](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibC281C1211E4B0E162702B23A80A1AED1s1) des marées dans leur lit, C. R. Acad. Sci. Paris 73 (1871) [148--154.](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibC281C1211E4B0E162702B23A80A1AED1s1)
- [69] Z. Shen, W. Yan, G. Yuan, A robust and contact resolving Riemann solver on unstructured mesh, Part I, Euler method, J. Comput. Phys. 268 (2014) 432--455, [https://doi.org/10.1016/j.jcp.2014.02.020,](https://doi.org/10.1016/j.jcp.2014.02.020) <https://www.sciencedirect.com/science/article/pii/S0021999114001405>.
- [70] L. Tallois, [Simulation](http://refhub.elsevier.com/S0021-9991(25)00241-4/bib8CC6645FC9FF594BE404530F814B1E35s1) numérique de l'ablation liquide, Thèse de doctorat, INSA Toulouse, 2023.
- [71] L. Tallois, S. Peluchon, P. Villedieu, A second-order extension of a robust implicit–explicit acoustic-transport splitting scheme for two-phase flows, Comput. Fluids 244 (2022), <https://doi.org/10.1016/j.compfluid.2022.105531> 105531, <https://www.sciencedirect.com/science/article/pii/S0045793022001633>.
- [72] E. Toro, Riemann Solvers and Numerical Methods for Fluid [Dynamics,](http://refhub.elsevier.com/S0021-9991(25)00241-4/bibC7DB8535E60CB29E5D2332828BE87AFBs1) Springer, 1997.
- [73] E. Toro, M. Spruce, W. Speares, Restoration of the contactsurface in the HLL-Riemann solver, Shock Waves 4 (1994) 25--34, <https://doi.org/10.1007/BF01414629>.
- [74] I. Toumi, A weak formulation of Roe's approximate Riemann solver, J. Comput. Phys. 102 (1992) 360--373, [https://doi.org/10.1016/0021-9991\(92\)90378-C](https://doi.org/10.1016/0021-9991(92)90378-C), <https://www.sciencedirect.com/science/article/pii/002199919290378C>.
- [75] E. Turkel, Preconditioned methods for solving the incompressible and low speed compressible equations, J. Comput. Phys. 72 (1987) 277--298, [https://doi.org/](https://doi.org/10.1016/0021-9991(87)90084-2) [10.1016/0021-9991\(87\)90084-2](https://doi.org/10.1016/0021-9991(87)90084-2), <https://www.sciencedirect.com/science/article/pii/0021999187900842>.
- [76] E. Turkel, Review of preconditioning methods for fluid dynamics, Appl. Numer. Math. 12 (1993) 257--284, [https://doi.org/10.1016/0168-9274\(93\)90122-8](https://doi.org/10.1016/0168-9274(93)90122-8), <https://www.sciencedirect.com/science/article/pii/0168927493901228>.
- [77] Z. Zou, E. Audit, N. Grenier, C. Tenaud, An accurate sharp interface method for two-phase compressible flows at low-Mach regime, Flow Turbul. Combust. 105 (2020), [https://doi.org/10.1007/s10494-020-00125-1.](https://doi.org/10.1007/s10494-020-00125-1)
- [78] Z. Zou, N. Grenier, S. Kokh, C. Tenaud, E. Audit, Compressible solver for two-phase flows with sharp interface and capillary effects preserving accuracy in the low Mach regime, J. Comput. Phys. 448 (2021) 110735, [https://doi.org/10.1016/j.jcp.2021.110735.](https://doi.org/10.1016/j.jcp.2021.110735)