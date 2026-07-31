Contents lists available at [ScienceDirect](http://www.ScienceDirect.com/)

[www.elsevier.com/locate/jcp](http://www.elsevier.com/locate/jcp)

![](_page_0_Picture_5.jpeg)

# Low-dissipation BVD schemes for single and multi-phase compressible flows on unstructured grids

![](_page_0_Picture_7.jpeg)

Lidong Cheng <sup>a</sup>*,*d, Xi Deng <sup>b</sup>*,*∗, Bin Xie <sup>c</sup>*,*∗, Yi Jiang <sup>a</sup>*,*∗, Feng Xiao <sup>d</sup>*,*<sup>∗</sup>

- <sup>a</sup> *School of Aerospace Engineering, Beijing Institute of Technology, Beijing, 10-0081, China*
- <sup>b</sup> *Department of Aeronautics, Imperial College London, SW7 2AZ, United Kingdom*
- <sup>c</sup> *School of Naval Architecture, Department of Ocean and Civil Engineering, Shanghai Jiaotong University, Shanghai, 200240, China*
- <sup>d</sup> *School of Engineering, Department of Mechanical Engineering, Tokyo Institute of Technology, Tokyo, 152-8550, Japan*

#### a r t i c l e i n f o a b s t r a c t

#### *Article history:* Available online 28 December 2020

*Keywords:* Compressible flows Single and multi-phase fluids Discontinuities Low-dissipation BVD algorithm Unstructured grids

Solving compressible flows containing both smooth and discontinuous flow structures still remains a big challenge for finite volume methods, especially on unstructured grids where one faces more difficulties in building high-order polynomial reconstruction and limiting projection to suppress numerical oscillations in comparison with the case of structured grids. As a result, most of the current finite volume schemes on unstructured grids are of second order and too dissipative to resolve fine structures of complex flows. In this paper, we report two novel hybrid schemes to resolve vortical and discontinuous solutions on unstructured grids by reducing numerical dissipation. Different from conventional shock capturing schemes that use polynomials and limiting projections for reconstruction, the proposed schemes employ two second-order schemes, i.e. a polynomial and a sigmoid function as candidate reconstruction functions to approximate smooth and discontinuous solutions respectively. As the polynomial function, the MUSCL (Monotone Upstreamcentered Schemes for Conservation law) scheme with the MLP (Multi-dimensional Limiting Process) slope limiter is adopted, while being a sigmoid function, the multi-dimensional THINC (Tangent of Hyperbola for INterface Capturing) function with quadratic surface representation and Gaussian quadrature, so-called THINC/QQ, is used to mimic the discontinuous solution structure. With these candidates for reconstruction, a single-step boundary variation diminishing (BVD) algorithm, which aims to minimize numerical dissipation, is designed on unstructured grids to select the final reconstruction function. The resulting two variant schemes, MUSCL-THINC/QQ-BVD schemes with two and three candidates respectively, are algorithmically simple and show great superiority to other existing schemes in capturing discontinuous and vortical flow structures for single and multiphase compressible flows on unstructured grids. The performance of the proposed schemes has been extensively verified through benchmark tests of single and multiphase compressible flows, where discontinuous and vortical flow structures, like shock waves, contact discontinuities and material interfaces, as well as vortices and shear instabilities of different scales, coexist simultaneously. The numerical results show that the proposed schemes that hybrids two second-order schemes are capable of capturing sharp discontinuous profiles without numerical oscillations and resolving vortical structures

*E-mail addresses:* [x.deng@imperial.ac.uk](mailto:x.deng@imperial.ac.uk) (X. Deng), [xie.b.aa@sjtu.edu.cn](mailto:xie.b.aa@sjtu.edu.cn) (B. Xie), [jy2818@163.com](mailto:jy2818@163.com) (Y. Jiang), [xiao.f.aa@m.titech.ac.jp](mailto:xiao.f.aa@m.titech.ac.jp) (F. Xiao).

<sup>\*</sup> Corresponding authors.

along shear layers and material interfaces with significantly improved solution quality superior to other schemes of even higher order reconstructions.

© 2020 Elsevier Inc. All rights reserved.

#### **1. Introduction**

Discontinuous solutions, such as shock waves, are commonly found in compressible flows of high Mach number. The flow structures become even more complex when multi-phase fluids with moving interfaces are involved. Mixed flow features such as shock waves, contact discontinuities, moving material interfaces, strong species gradients and shear layers become typical in flows of these kinds. In general, flows of such complexity are far beyond the reach of theoretical analysis, and hard for experimental approach as well in many situations. The numerical simulation becomes an effective and probably the only available alternative in some cases to provide flow information for scientific researches or engineering applications. In spite of the great importance, the numerical methods to solve compressible flows of single and multiple phases are far from a mature stage. Particularly, the solvers on unstructured grids are less developed compared to their counterparts on structured grids although the unstructured grids are much more demanded in industrial applications that usually involve complex geometrical configurations.

The finite volume method (FVM) has been widely accepted as the basic numerical framework to solve compressible flows on unstructured grids because of its numerical conservativeness and grid adaptivity. Implementing low-order FVM on unstructured grids is straightforward and easy, while high-order unstructured FVM might be quite challenging. Over the decades, a number of high-order schemes have been proposed on unstructured grids to improve numerical solutions for compressible flows. The well-known *k*-exact least-square method has been proposed and developed in [\[1–3](#page-25-0)]. In this method, a stencil consisting of the target cell and its neighbors is employed to construct a polynomial of degree *k*. For high-order polynomials, however, the stencil has to be extended to include more neighboring cells, which increases the complexity of numerical algorithm and parallelism. So, high-order finite volume schemes on unstructured grids are not popularly used in the codes of practical interests. Different from conventional finite volume method, Multi-moment Constrained finite Volume method (MCV) [[4–8\]](#page-25-0), high order Discontinuous Galerkin (DG) [\[9,10\]](#page-26-0), Flux Reconstruction (FR) [\[11](#page-26-0)–[15](#page-26-0)] and its variants for unstructured grids [[16](#page-26-0)–[19](#page-26-0)] realize high-order reconstructions by using locally defined degrees of freedom (DOFs). These methods have received particular attention in recent years because of their superior convergence property for smooth solutions, as well as the compact stencil which might be beneficial for parallel processing.

Although these schemes based on high-order polynomial reconstructions show superiority in simulating smooth flow features such as acoustic waves, vortices and turbulence, they generally face challenges to obtain accurate and stable solutions around discontinuities. Special techniques such as limiting projection or artificial viscosity must be designed to prevent Gibbs phenomenon and its associated spurious numerical oscillations arising from high order polynomial interpolations in the presence of discontinuities. Usually, as the reconstruction and the numerical limiter devised in 1D cannot be directly used in unstructured grids, solving discontinuous flow features on unstructured grids imposes additional difficulty. Based on the idea of TVD (Total Variation Diminishing) scheme [[20](#page-26-0)], multi-dimensional slope limiting processes on unstructured grids have been proposed in [\[21,22\]](#page-26-0) and improved recently in [\[23,24](#page-26-0)]. In order to reduce the numerical dissipation of TVD schemes, WENO schemes (Weighted Essentially Non-Oscillatory) have been extended to unstructured grids, for example [[25](#page-26-0)–[29\]](#page-26-0), to cite but a few. The general idea of WENO schemes is to construct a weighted average of the polynomial approximations over all candidate stencils. However, it is not a trivial work to construct efficient WENO scheme on unstructured grids since dealing with wide stencils and choosing admissible stencil cells increase the algorithmic complexity. The difficulty of solving discontinuities also exists and brings about even more serious problems for high-order schemes using local high-order reconstructions, like DG and FR methods. Although several strategies such as artificial viscosity [\[30,31\]](#page-26-0) and subcell finite volume formulation [\[32,33](#page-26-0)] have been proposed and improved, solving discontinuities accurately and robustly remains a challenge for high order local reconstruction schemes. Moreover, in spite of the efforts aforementioned, limiting processes or artificial viscosity methods usually introduce excessive numerical dissipation which continuously smears and blurs flow structures. Especially, the resolution of discontinuous flow features such as contact surfaces, shear waves, reaction fronts and material interfaces may evolve from bad to worse due to the limiting process.

Although polynomial-based reconstruction is well suited for smooth solutions, it may not be a proper choice for solutions containing discontinuities. So, in addition to the polynomial function, it may be sensible to prepare another non-polynomial interpolation function to mimic discontinuity. The sigmoid function for example serves well for this purpose. Hence, making use of a hybrid reconstruction that includes polynomials and sigmoid functions can be expected to provide better solutions to problems which contain both smooth and discontinuous flow structures. This idea has been proven very successful in [[34](#page-26-0)] where a novel algorithm called boundary variation diminishing (BVD) was proposed to use a non-polynomial reconstruction scheme, the THINC (Tangent of Hyperbola for INterface Capturing) scheme [[35](#page-26-0),[36](#page-26-0)], to solve discontinuous flow structures, while high order polynomial-based WENO scheme [[37](#page-26-0)] for smooth flow regions. The proposed methodology significantly reduces the numerical dissipation and produces much improved solutions for both smooth and discontinuous flow structures. Following [[34](#page-26-0)], the BVD algorithm has been improved and applied for more challenging problems involving stiff source terms and material interfaces [\[38,39\]](#page-26-0). Based on the formulation in [\[34](#page-26-0)], BVD algorithm has also been tested on unstructured grids [\[40,41\]](#page-26-0). More recently, higher-order shock capturing schemes have been devised using the BVD algorithm [[42](#page-26-0)–[44](#page-26-0)], where unlimited polynomials of high order and the THINC function are used as the candidate functions for reconstruction. These schemes do not use any conventional limiting projections which have been extensively investigated as a necessary tool to eliminate spurious oscillations and stabilize numerical solutions in vicinity of discontinuities. Consequently, the BVD schemes give high-fidelity solutions to both smooth and discontinuous flow structures showing significant superiority compared to conventional schemes using polynomial-based reconstructions. Although being mainly practiced on structured grids, the above works show the great potential of the BVD algorithm as an alternative to design accurate and robust numerical schemes for compressible flows.

In this work, we make efforts to extend the BVD algorithm to unstructured grids and devise new shock capturing schemes which have largely reduced numerical dissipation and are thus capable of computing vortical fluid structures and discontinuities with substantially improved solution quality. For sake of algorithmic simplicity and practical utility, the unstructured-grid BVD algorithm uses linear polynomial and THINC function as the reconstruction candidates. For linear polynomial, the second order MUSCL (Monotone Upstream-centered Schemes for Conservation law) scheme with the MLP (Multi-dimensional Limiting Process) slope limiter [\[23,24\]](#page-26-0) is adopted. The multi-dimensional THINC (Tangent of Hyperbola for INterface Capturing) function with quadratic surface representation and Gaussian quadrature [\[45,46\]](#page-26-0), so-called THINC/QQ, is used as the non-polynomial reconstruction candidate. With the above reconstruction candidates, a single-step BVD algorithm with multiple candidate member functions is devised to select the final reconstruction function in line with the principle of minimizing the boundary variations. The resultant numerical schemes with two and three members are named as MUSCL-THINC/QQ-BVD. The performance of these MUSCL-THINC/QQ-BVD schemes is verified through benchmark tests of single and multiple phase compressible flows. The numerical results show that the proposed schemes are able to capture sharp discontinuous profiles without numerical oscillation. Meanwhile, it is able to resolve vortical flow structures associated with Kelvin-Helmholtz instabilities along the shear layer and material interface. With largely reduced numerical dissipation, the proposed MUSCL-THINC/QQ-BVD schemes significantly improve the resolution for both smooth and discontinuous flow structures in comparison with schemes only relying on polynomial reconstructions. The proposed schemes are expected to serve as accurate, robust and practical schemes for compressible flows of smooth and discontinuous structures.

The rest of this paper is organized as follows. Mathematical models of the targeted physical problems of single and multiphase compressible flows are introduced in section 2. In section [3](#page-3-0), after a brief introduction to the MUSCL scheme and the THINC/QQ scheme which are the candidate functions for reconstruction on unstructured grids, two novel BVD schemes are described in detail. Numerical results and discussions are presented in section [4](#page-8-0), followed by some concluding remarks in section [5.](#page-22-0)

# **2. Mathematical models**

#### *2.1. Governing equations*

In this work, our numerical schemes are tested with benchmark problems of linear advection equation and Euler equations of inviscid single and two-phase compressible flows. We focus on the two dimensional problems in this paper, and the three dimensional implementation is straightforward without substantial difficulty.

A general form of conservation laws can be written as:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} + \frac{\partial \mathbf{G}(\mathbf{U})}{\partial y} = \mathbf{S},\tag{1}$$

which are concretized as follows.

• The linear advection equation

The advection transport of a scalar *φ(x, y,t)*, with a specified flow velocity *V* = *(u, v)*, can be formulated by reducing Eq. (1) into

$$\mathbf{U} = \phi, \ \mathbf{F}(\mathbf{U}) = u\phi, \ \mathbf{G}(\mathbf{U}) = v\phi, \ \mathbf{S} = \phi \nabla \cdot \mathbf{V}$$
 (2)

It is widely used to compute free interfaces transported by flow motion, and *φ* can be either the volume fraction of the target fluid species in the volume of fluid method or the signed distance to the interface in the level set method.

• The Euler equations

Inviscid single-phase compressible flows are modeled by the Euler equations, which consist of equations for conservation of mass, momentum and energy respectively.

$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ E \end{pmatrix}, \ \mathbf{F}(\mathbf{U}) = \begin{pmatrix} \rho u \\ \rho u u + p \\ \rho u v \\ u(E+p) \end{pmatrix}, \ \mathbf{G}(\mathbf{U}) = \begin{pmatrix} \rho v \\ \rho v u \\ \rho v v + p \\ v(E+p) \end{pmatrix}, \ \mathbf{S} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$
(3)

where *ρ* is the density, *p* the pressure field and *E* is the total energy.

![](_page_3_Picture_2.jpeg)

Fig. 1. Two dimensional elements.

• The five-equation model for two-phase inviscid compressible flows
Inviscid two-phase compressible flows under mechanical equilibrium are modeled by the five-equation model developed\nin [47]. It assumes that the interface cells containing two kinds of fluids are in equilibrium of pressure. Thus, the
governing equations consist of two mass conservation laws, one momentum equations (with two components in 2D),
one energy equation and an equation for the transport of volume fraction.

$$\mathbf{U} = \begin{pmatrix} \alpha_1 \\ \rho_1 \alpha_1 \\ \rho_2 \alpha_2 \\ \rho u \\ \rho v \\ E \end{pmatrix}, \ \mathbf{F}(\mathbf{U}) = \begin{pmatrix} u \alpha_1 \\ \rho_1 \alpha_1 u \\ \rho_2 \alpha_2 u \\ \rho u u + p \\ \rho u v \\ u (E + p) \end{pmatrix}, \ \mathbf{G}(\mathbf{U}) = \begin{pmatrix} v \alpha_1 \\ \rho_1 \alpha_1 v \\ \rho_2 \alpha_2 v \\ \rho v u \\ \rho v v + p \\ v (E + p) \end{pmatrix}, \ \mathbf{S} = \begin{pmatrix} \alpha_1 \nabla \cdot \mathbf{V} \\ 0 \\ 0 \\ 0 \\ 0 \end{pmatrix}$$

$$(4)$$

where  $\alpha_k \in [0, 1]$  and  $\rho_k$  is the volume fraction and density of the kth (k = 1, 2) fluid.  $\mathbf{V} = (u, v)$  is the velocity field.

#### 2.2. The closure strategy

To close the Euler equations and the five-equation model, fluids are assumed to satisfy the following ideal gas law:

$$p = \rho e(\gamma - 1) \tag{5}$$

where e is the internal energy, and  $\gamma$  is the ratio of the specific heats.

For two-component flows, conservative constraints lead to the following mixing formula of volume fraction, density and internal energy:

$$\alpha_1 + \alpha_2 = 1$$

$$\alpha_1 \rho_1 + \alpha_2 \rho_2 = \rho$$

$$\alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2 = \rho e$$
(6)

As derived in [48,49], in order to prevent spurious oscillation at material interface, the mixed ratio of the specific heats can be calculated as

$$\frac{1}{\gamma - 1} = \frac{\alpha_1}{\gamma_1 - 1} + \frac{\alpha_2}{\gamma_2 - 1}.\tag{7}$$

#### 3. Numerical methods

#### 3.1. Computational grids

Two dimensional computational domains are divided into non-overlapping triangular or quadrilateral elements  $\Omega_i$  (i=0,1,2,...,N). Vertices and edges are denoted by  $\vartheta_{ik}(k=1,2,...,K)$  and  $\Gamma_{ij}(j=1,2,...,J)$ , where K=J=3 for triangular meshes and K=J=4 for quadrilateral meshes. The cell center is denoted by  $\vartheta_{ic}(x_{ic},y_{ic})$ . We define the area of element  $\Omega_i$  as  $|\Omega_i|$ , length and unit normal vector of edge  $\Gamma_{ij}$  as  $|\Gamma_{ij}|$  and  $\boldsymbol{n}_{ij}=(n_{ijx},n_{ijy})$  (see Fig. 1).

#### 3.2. The Godunov-type finite volume method

The general formulation of the finite volume method is obtained by integrating Eq. (1) over a finite volume element  $\Omega_i$ , yielding the following semi-discrete form for cell-average values.

$$\frac{d\overline{\boldsymbol{U}}_{i}}{dt} = -\frac{1}{|\Omega_{i}|} \oint_{\partial \Omega_{i}} \boldsymbol{F}_{n}(\boldsymbol{U}) d\Gamma + \frac{1}{|\Omega_{i}|} \oint_{\Omega_{i}} \boldsymbol{S} d\Omega, \tag{8}$$

where  $\mathbf{F}_n(\mathbf{U}) = (\mathbf{F}(\mathbf{U}), \mathbf{G}(\mathbf{U})) \cdot \mathbf{n}$  is the flux in the normal direction,  $\mathbf{n} = (n_x, n_y)$ , of the volume surface. The integration along the element surface can be calculated by the summation over all edge segments,

$$\oint_{\partial \Omega_i} \mathbf{F}_n(\mathbf{U}) d\Gamma = \sum_{j=1}^J \int_{\Gamma_{ij}} \mathbf{F}_n(\mathbf{U}) d\Gamma \approx \sum_{j=1}^J \mathbf{F}_{nij}(\mathbf{U}) \left| \Gamma_{ij} \right|, \tag{9}$$

and the key in a finite volume method is how to calculate the numerical flux  $F_{nij}(U)$  across each boundary edge.

The Godunov-type finite volume method has evolved into the mainstream numerical framework for compressible fluid dynamics, following the pioneer work of Godunov [50]. In a Godunov finite volume method, the numerical flux across edge  $\Gamma_{ii}$  can be cast into

$$\mathbf{F}_{nij}(\mathbf{U}) = \mathbf{F}_{nij}^{\text{Riemann}}(\mathbf{U}_{ij}^{R}, \mathbf{U}_{ij}^{L}), \tag{10}$$

where  $\boldsymbol{U}_{ij}^{L}$  and  $\boldsymbol{U}_{ij}^{R}$  stand for the left and right-side values of  $\boldsymbol{U}(x,y)$  across edge  $\Gamma_{ij}$ , while  $\boldsymbol{F}_{nij}^{Riemann}(\boldsymbol{U}_{ij}^{R},\boldsymbol{U}_{ij}^{L})$  denotes a Riemann solver to calculate the numerical flux from the left and right-side values.

A Godunov finite volume method consists of three major steps to advance the solution of Eq. (8).

- (i) Given the cell-average values  $\overline{\boldsymbol{U}}_i$ , find the values at cell boundaries,  $\boldsymbol{U}_{ij}^L$  and  $\boldsymbol{U}_{ij}^R$ , via spatial reconstruction. It is the core part of the present paper, and will be detailed in section 3.3.
- (ii) Compute the numerical flux from the reconstructed values across cell boundary, using either exact or approximate Riemann solver. We use either the HLL [51] or HLLC Riemann solver [52] to solve the Euler equations of single phase inviscid compressible flows and the five-equation model for two-phase multiphase compressible flows.
- (iii) Solve Eq. (8) as ordinary differential equations to update the numerical solution in time. In present work, we make use of the third-order SSP Runge-Kutta [53] method for time integration.

#### 3.3. Reconstruction schemes

As the central part of this paper, we present a novel spatial reconstruction method, so-called MUSCL-THINC/QQ-BVD method, in this subsection to compute the left and right states  $\boldsymbol{U}_{ij}^L$  and  $\boldsymbol{U}_{ij}^R$  across a boundary edge of an unstructured grid cell element, from the known cell-averaged values  $\overline{\boldsymbol{U}}$  in the target cell  $\Omega_i$  and its surrounding cells.

Although the BVD algorithm has been devised and successfully applied to the structured grids, its implementation on unstructured grids still remains an issue not fully resolved. For simplicity and computational efficiency, we use the MUSCL scheme with a multi-dimensional limiter and the multi-dimensional THINC/QQ scheme on unstructured grids as the candidate interpolation functions for reconstruction.

Next, we first present the MUSCL and THINC reconstruction functions, and then devise two variants of the BVD scheme to select the final reconstruction function from the candidates on unstructured grids.

#### 3.3.1. The MUSCL scheme

The basic idea of a MUSCL-type scheme is to reconstruct the spatial distribution of a physical variable q(x, y) for a certain cell by a linear polynomial with a slope limiter. For two dimensional unstructured grids, the reconstruction function can be written as.

$$Q_i^M(x, y) = \bar{q}_i + \phi_i \left( q_{xi}(x - x_{ic}) + q_{yi}(y - y_{ic}) \right) \tag{11}$$

where  $\bar{q}_i$  is the cell-average value,  $\phi_i$  is the slope limiter to suppress numerical oscillation and keep monotonicity. As a common practice on unstructured grids, the cell-averaged gradient,  $(\overline{\nabla q})_i = (q_{xi}, q_{yi})$ , is determined from a least square method.

We use the multi-dimensional MLP-u2 limiter in [24] as the slope limiter for the MUSCL scheme.

$$\phi_i = \min_{k=1}^K \begin{cases} \Phi(R_{ik}) & \text{if } (\overline{\nabla q})_i \cdot \mathbf{r}_{ik} \neq 0 \\ 1 & \text{otherwise,} \end{cases}$$

where  $\mathbf{r}_{ik}(k=1,2,\ldots,K)$  is the vector from  $\vartheta_{ic}$  to vertex  $\vartheta_{ik}$ , and  $R_{ik}$  is the ratio of the maximum or minimum allowable variation to the estimated variation at  $\vartheta_{ik}$ ,

$$R_{ik} = \max\left(\frac{\bar{q}_{ik}^{\min} - \bar{q}_i}{(\overline{\nabla q})_i \cdot \mathbf{r}_{ik}}, \frac{\bar{q}_{ik}^{\max} - \bar{q}_i}{(\overline{\nabla q})_i \cdot \mathbf{r}_{ik}}\right)$$

Here,  $\bar{q}_{ik}^{min}$  and  $\bar{q}_{ik}^{max}$  are the minimum and maximum cell-average values of cells around  $\vartheta_{ik}$ . For the MLP-u2 limiter,

![](_page_5_Figure_2.jpeg)

Fig. 2. A illustration of possible reconstructions by the MUSCL scheme (left) and the THINC/QQ scheme (right). (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)

$$\Phi(R_{ik}) = \frac{R_{ik}^2 + 2R_{ik} + \epsilon}{R_{ik}^2 + R_{ik} + 2.0 + \epsilon}$$

where  $\epsilon$  is a small positive number valued as  $\epsilon = 1.0 \times 10^{-15}$  in the present work.

#### 3.3.2. The THINC/QQ scheme

For unstructured grids, we use the multi-dimensional THINC/QQ (THINC method with Quadratic surface representation and Gaussian Quadrature) scheme [46,45] as another candidate in the BVD schemes. As shown in [46,45], THINC/QQ is able to achieve highly accurate representation for discontinuity by accounting of geometrical information such as normal direction and curvature of the discontinuity, which is much beneficial to improve the numerical results of continuous solutions when the THINC method is applied to smooth distributions by choosing a proper steepness parameter  $\beta$  [38]. The 2D THINC/QQ interpolation function reads

$$Q_{i}^{T(\beta)}(x, y) = \bar{q}_{i}^{\min} + \frac{\bar{q}_{i}^{\max} - \bar{q}_{i}^{\min}}{2} \left( 1 + \tanh\left(\frac{\beta}{H_{i}} \left(P_{i}(x, y) + d_{i}\right)\right) \right)$$
(12)

where  $\bar{q}_i^{min}$  and  $\bar{q}_i^{max}$  are the maximum and minimum cell-average values of cells sharing the vertices of cell  $\Omega_i$ .

$$\bar{q}_i^{max} = \max_{k=1}^K \left\{ \bar{q}_{ik}^{\max} \right\}, \bar{q}_i^{min} = \min_{k=1}^K \left\{ \bar{q}_{ik}^{\min} \right\}$$

and  $\beta$  is a parameter to control the steepness.  $H_i$  is the hydraulic diameter of  $\Omega_i$  defined by

$$H_i = \frac{4|\Omega_i|}{\sum_{i=1}^J |\Gamma_{ij}|}.$$

 $P_i(x, y) + d_i$  is a fully quadratic polynomial including geometrical information of the reconstructed profile as

$$P_i(x, y) = a_{20}(x - x_{ic})^2 + a_{11}(x - x_{ic})(y - y_{ic}) + a_{02}(y - y_{ic})^2 + a_{10}(x - x_{ic}) + a_{01}(y - y_{ic}).$$

The coefficients  $a_{st}(0 < s + t \le 2)$  can be calculated using a least square method. The only unknown  $d_i$  is determined from the conservation constraint condition

$$\frac{1}{|\Omega_i|} \oint_{\Omega_i} Q_i(x, y) dx dy = \bar{q}_i.$$

This integration is calculated with Gaussian quadrature. For more details, please refer to [45,46].

It is noted that the steepness of the THINC function can be controlled by parameter  $\beta$ . Shown later, different candidate functions in the BVD algorithm can be created by modifying  $\beta$ .

Fig. 2 shows the reconstructions by the MUSCL scheme and the THINC/QQ scheme. As the THINC/QQ function of Eq. (12) includes not only gradient terms but also curvature terms, it has more geometrical information of the solution profile. So, it should be more beneficial and desirable to preserve curved flow structures such as vortices in comparison with the MUSCL scheme.

![](_page_6_Picture_2.jpeg)

Fig. 3. Computation of the TBV for candidate function  $Q_i^{\xi}$  on cell  $\Omega_i$ . The same reconstruction function is used for the target cell and its immediate neighboring cells.

#### 3.3.3. The BVD algorithm

As aforementioned, given the left and right-side values of the conservative variables of a cell boundary,  $\mathbf{U}^R$  and  $\mathbf{U}^L$ , the numerical flux across the boundary can be computed from by the Riemann solver. An approximate Riemann solver is expressed in the following canonical formulation [34,54]:

$$\mathbf{F}(\mathbf{U}^L, \mathbf{U}^R) = \frac{1}{2} \left( \mathbf{F}(\mathbf{U}^L) + \mathbf{F}(\mathbf{U}^R) \right) - \mathbf{A}(\mathbf{U}^L, \mathbf{U}^R) (\mathbf{U}^R - \mathbf{U}^L), \tag{13}$$

 $A(U^L, U^R)$  is a matrix computed from  $U^R$  and  $U^L$ . The last term of Eq. (13) can be interpreted as a dissipation term. It is noted that different approximate Riemann solver may have different coefficient for each term, we just use Eq. (13) as a generic form to illustrate that a Riemann solver basically includes two part: a central part and a dissipation part. As shown in Eq. (10) and Eq. (14) in [55], for example, the HLL-family Riemann solvers can be written as a classic central flux with dissipation terms. These dissipation terms may introduce excessive dissipation in numerical solutions. The BVD algorithm is designed to select the reconstruction function from multiple candidates so as to minimize the dissipation term. Hence, choosing a final reconstruction that generates smaller boundary variation tends to reduce numerical dissipation and preserve the solution structures [56].

The BVD concept stated above provides a general guideline to devise numerical schemes. Several variants of BVD schemes have been developed in our previous works [34,38,39,42,43] on structured grids using different candidate reconstruction functions.

We denote the union of admissible reconstruction functions by

$$\Xi := \left\{ Q_i^{\xi 1}(x, y), Q_i^{\xi 2}(x, y), \cdots, Q_i^{\xi N}(x, y) \right\}$$
 (14)

for cell  $\Omega_i$  with  $\xi N$  being the total number of the candidate functions.

A BVD algorithm is needed to determine one  $Q_i^{\xi}(x,y) \in \Xi$  from the multiple candidate reconstruction functions. The first question is how to measure the BV-equivalent quantity. Some of those working well on structured grids might be difficult to be implemented on unstructured grids. We make use of the total boundary variation (TBV) to measure the difference of the reconstructed values across cell boundary. Similar to [38,42], we compute the TBV of  $Q_i^{\xi}(x,y)$  for the target cell  $\Omega_i$  with the assumption that  $\Omega_i$  and its immediate neighbors,  $\Omega_{ij}$  ( $j=1,2,\cdots,J$ ) which share the boundary segment  $\Gamma_{ij}$  with  $\Omega_{ij}$ , use the same reconstruction scheme as shown in Fig. 3 for the triangular grid. Thus, the BV at cell edge  $\Gamma_{ij}$  is computed by

$$BV_{ij}^{\xi} = \left| q_{ij,L}^{\xi} - q_{ij,R}^{\xi} \right|,$$

where  $q_{ij,L}^{\xi}$  and  $q_{ij,R}^{\xi}$  are the integrated value of the reconstruction functions on  $\Omega_i$  and  $\Omega_{ij}$  respectively,

$$q_{ij,L}^{\xi} = \int_{\Gamma_{ij}} Q_i^{\xi}(x,y) d\Gamma, \text{ and } q_{ij,R}^{\xi} = \int_{\Gamma_{ij}} Q_{ij}^{\xi}(x,y) d\Gamma.$$

$$\tag{15}$$

It is noted that the BV in unstructured grid includes the length of the edge segment, which differs from the structured grid that allows the spatial reconstruction being done dimension-wisely.

Then, the total boundary variation (TBV) of  $Q_i^{\xi}(x, y)$  for cell  $\Omega_i$  is obtained as

$$TBV_{i}^{\xi} = \sum_{j=1}^{J} BV_{ij}^{\xi}.$$
 (16)

 $TBV_i^\xi$  can be computed for all  $Q_i^\xi(x,y) \in \Xi$  with  $\xi = \xi 1, \xi 2, \cdots, \xi N$ . A one-step BVD algorithm to determine the final reconstruction function  $Q_i^{\text{final}}(x,y)$  for cell  $\Omega_i$  can be devised by simply choosing the one that has the minimum TBV,

$$Q_i^{\text{final}}(x, y) = Q_i^{\tilde{\xi}}(x, y) \in \Xi, \text{ if } TBV_i^{\tilde{\xi}} \le TBV_i^{\tilde{\xi}} \text{ for all } Q_i^{\tilde{\xi}}(x, y) \in \Xi.$$
 (17)

Given the TBVs of all reconstruction functions, (17) provides a straightforward and general BVD formulation to determine the reconstruction functions. High-fidelity numerical results can be obtained if candidate functions are properly included in union  $\Xi$ . As shown in the following variant schemes, existing methods for reconstruction can be used as the candidate functions. Our previous practice and experience show that the numerical solution can be significantly improved by combining the existing schemes using the BVD concept, even although using each of them alone might not produce satisfactory results

It is noted that, this one-step BVD algorithm is different from and simpler than the multi-step BVD schemes in [42,43], where the BVD algorithm is implemented in sequential steps. We compare the TBVs of all candidates directly and choose the scheme having the smallest TBV. The present one-step algorithm is more efficient and easier to implement on unstructured grids.

As long as the reconstruction function is determined for each cell, the left-side (inner-side) values along the boundary segments are obtained by

$$q_{ij}^{L} = \frac{1}{|\Gamma_{ij}|} \int_{\Gamma_{ij}} Q_i^{\text{final}}(x, y) d\Gamma$$
 (18)

By repeating this process cell-wisely, we get both  $q_{ij}^L$  and  $q_{ij}^R$  across boundary segment  $\Gamma_{ij}$ , which are then used to approximate the numerical fluxes in Riemann solvers.

Different schemes can be made by including different numbers of candidate functions in  $\Xi$ . In this work, we proposed and tested two different schemes with two ( $\xi N = 2$ ) and three ( $\xi N = 3$ ) members in the candidate union.

## 3.3.4. The two-member BVD scheme

The candidate union contains two members  $\Xi := \left\{Q_i^{\xi 1}(x,y), Q_i^{\xi 2}(x,y)\right\}$  as the simplest case. The first candidate is the MUSCL scheme  $Q_i^{\xi 1}(x,y) = Q_i^M(x,y)$  shown above. The MUSCL reconstruction in form of Eq. (11) or others has been widely used in research and commercial CFD code for its robustness and simplicity on unstructured grids. However, the excessive numerical dissipation limits its utility in applications where vortical and smooth structure need to be adequately resolved. Shown in [39], the numerical dissipation in MUSCL scheme can be substantially reduced by implementing the BVD principle with the THINC function as another candidate.

The analysis on numerical dispersion and dissipation in [38] reveals that the THINC reconstruction with the steepness parameter  $\beta$  ranging from 0.8 to 1.3 can retrieve a class of MUSCL schemes of different slope limiters, and the discontinuous solution structures can be well preserved if  $\beta$  is larger than 1.3. We choose  $\beta = \beta_l = 1.4$  in Eq. (12) and denote the resulting THINC function by  $Q_i^{T(\beta_l)}(x, y)$ , which is used as the second candidate  $Q_i^{\xi 2}(x, y) = Q_i^{T(\beta_l)}(x, y)$ .

In summary, the two-member one-step BVD scheme with MUSCL and THINC reconstructions, referred to as the two-member MUSCL-THINC/QQ-BVD, is formulated as Eq. (17) with

$$\Xi := \left\{ Q_i^M(x, y), Q_i^{T(\beta_l)}(x, y) \right\}. \tag{19}$$

#### 3.3.5. The three-member BVD scheme

Numerical tests show that the above two-member BVD scheme can effectively reduce numerical diffusion at strong discontinuities such as shock waves and material interfaces. However, since a relatively large steepness  $\beta_l$  is used, it can not capture some slight discontinuities such as shear waves and vortices where the MUSCL scheme is chosen by the present BVD algorithm. In order to improve the solutions for shear waves and vortical structures, we add another THINC reconstruction with a smaller  $\beta_s = 0.8$ ,  $Q_i^{T(\beta_s)}(x, y)$ , as a new candidate function. Thus, the candidate union becomes,

$$\Xi := \left\{ Q_i^{\xi 1}(x, y), Q_i^{\xi 2}(x, y), Q_i^{\xi 3}(x, y) \right\} = \left\{ Q_i^M(x, y), Q_i^{T(\beta_s)}(x, y), Q_i^{T(\beta_l)}(x, y) \right\}, \tag{20}$$

and the three-member MUSCL-THINC/QQ-BVD scheme is obtained by Eq. (17) with Eq. (20).

Shown later in numerical results, adding a THINC reconstruction with a relatively smaller  $\beta$  can effectively improve the resolution of the curved flow structures such as vortices, which may be attributed to the fact that the multi-dimensional THINC/QQ function of Eq. (12) includes not only gradient terms but also curvature terms.

![](_page_8_Picture_2.jpeg)

Fig. 4. The initial condition of the solid rotation test.

![](_page_8_Picture_4.jpeg)

Fig. 5. The solution of the MUSCL scheme after one rotation.

When applied to system equations, the reconstructed quantity q can be different variables. In this work, we choose the primitive variables as the reconstructed variables [57,58], i.e.  $(\rho, u, v, p)$  for the single phase Euler equations and  $(\alpha_1, \alpha_1\rho_1, \alpha_2\rho_2, u, v, p)$  for the five-equation two-phase inviscid compressible flow.

#### 4. Numerical results

In this section, we extensively verify the proposed unstructured-grid BVD schemes with various benchmark tests for scalar advection equation and conservation laws for inviscid single and two-phase compressible flows.

#### 4.1. Solid rotation of a complex profile

We consider a complex 2D benchmark problem used by [59,46]. This test case assesses the ability of schemes to resolve both smooth and sharply discontinuous solutions. The 2D computational domain,  $[0,1]^2$ , is divided into quasi-uniform 54, 604 triangular elements. Shown in Fig. 4, the initial condition consisting three shapes of different smoothness, i.e. a Zalesak discontinuous slotted disk [60], a smooth hump and a cone with slight discontinuities around bottom and apex, are defined by

$$\phi = \begin{cases} 1, & \text{if } |x - 0.5| > 0.025, \text{ or } y > 0.83 & r_1(x, y) < r_0 \\ \frac{1}{4}(1 + \cos(\pi \min(r_2(x, y)/r_0, 1))) & r_2(x, y) < r_0 \\ 1 - r_3(x, y)/r_0 & r_3(x, y) < r_0 \\ 0 & \text{otherwise,} \end{cases}$$

where  $r_i(x, y) = \sqrt{(x - x_i)^2 + (y - y_i)^2}$  and  $r_0 = 0.15$ . The centers of the Zalesak disk, smooth hump and cone are  $(x_1, y_1) = (0.5, 0.78)$ ,  $(x_2, y_2) = (0.31, 0.39)$  and  $(x_3, y_3) = (0.69, 0.39)$  respectively.

The initial shapes are rotated as solid bodies by a velocity field of

$$u = 0.5 - y$$
,  $v = x - 0.5$ 

![](_page_9_Figure_2.jpeg)

**Fig. 6.** Results of the two-member BVD scheme: (a) The solution after one rotation; (b) Red colored are the cells where THINC/QQ (*βl*) scheme is chosen by the BVD algorithm for reconstruction.

![](_page_9_Figure_4.jpeg)

**Fig. 7.** Same as Fig. 6, but for the three-member BVD scheme, where yellow colored cells using the THINC/QQ (*β<sup>s</sup>* ).

![](_page_9_Figure_6.jpeg)

**Fig. 8.** Three contours (0.05, 0.5, 0.95) of the Zalesak disk: left-the MUSCL scheme; right-BVD schemes; middle-zoomed in with meshes.

![](_page_10_Figure_2.jpeg)

**Fig. 9.** Density of the Riemann problem at *t* = 0*.*012.

![](_page_10_Figure_4.jpeg)

**Fig. 10.** Cells using THINC/QQ schemes to reconstruct density.

![](_page_10_Figure_6.jpeg)

**Fig. 11.** The geometry of the scram-jet engine.

with the maximum Courant number of <sup>0</sup>*.*2. Results after one revolution at *<sup>t</sup>* = <sup>2</sup>*π* by different schemes are shown in Figs. [5](#page-8-0), [6](#page-9-0) and [7.](#page-9-0)

For the smooth regions of hump and cone, three schemes give almost the same results, where the BVD schemes choose the MUSCL scheme in these regions for reconstruction, as shown in Figs. [6](#page-9-0) and [7.](#page-9-0) BVD schemes can accurately identify the cells including strong discontinuities (marked with red color in Figs. [6\(](#page-9-0)b) and [7\(](#page-9-0)b)) around the edge of the slotted disk, and select THINC/QQ scheme with *β<sup>l</sup>* which preserves the step-like solution structure as expected. This step-like structure is heavily smeared by the MUSCL scheme as shown in Figs. [5](#page-8-0) and [8](#page-9-0) due to the excessive numerical dissipation. The numerical dissipation error can be quantitatively estimated by checking the width of jump transition in VOF field computed by numerical schemes. It is observed from Fig. [8](#page-9-0) that BVD schemes can resolve the discontinuity within three cells, but the MUSCL scheme spreads the jump over more than 10 cells. We further calculated and show in Table [1](#page-12-0) the numerical errors and convergence rates of different schemes by grid-refinement numerical experiments. The numerical errors shown in Table [1](#page-12-0) are defined by

![](_page_11_Figure_2.jpeg)

Fig. 12. Contours of Mach number of the final steady solution.

![](_page_11_Figure_4.jpeg)

Fig. 13. Contours of Mach number of the final steady solution on a coarse mesh.

$$E_R = \frac{\sum_i \left| \phi_i - \phi_i^h \right| |\Omega_i|}{\sum_i \left| \phi_i \right| |\Omega_i|} \tag{21}$$

where  $\phi_i$  is the exact solution and  $\phi_i^h$  the numerical one. It is found that the BVD schemes have much smaller errors and higher convergence rates than the MUSCL scheme.

![](_page_12_Figure_2.jpeg)

**Fig. 14.** Mesh of wind tunnel around the corner.

![](_page_12_Figure_4.jpeg)

**Fig. 15.** Density contour of results at *t* = 4*.*0.

![](_page_12_Figure_6.jpeg)

**Fig. 16.** Cells where THINC/QQ scheme is used to reconstruct *ρ*.

**Table 1** Numerical errors (*E <sup>R</sup>* ) and convergence rates for the rotating test.

| Schemes          | Ne = 3, 262 | Rate | Ne = 13, 568 | Rate | Ne = 54, 604 |
|------------------|-------------|------|--------------|------|--------------|
| MUSCL            | 0.519099    | 0.72 | 0.315153     | 0.74 | 0.188273     |
| Two-member BVD   | 0.153037    | 1.42 | 0.057204     | 1.28 | 0.023559     |
| Three-member BVD | 0.141065    | 1.38 | 0.054019     | 1.03 | 0.026574     |

![](_page_13_Figure_2.jpeg)

**Fig. 17.** A bird's view of the density field.

It is interesting to see that the three-member BVD scheme chooses THINC/QQ scheme with *β<sup>s</sup>* in the regions where the solution is slightly discontinuous (marked by yellow color) around hump and cone, which is much beneficial to resolve the vortical flow structions shown later. It reveals that the THINC function with properly chosen *β* can provide a spectrum of reconstructions fitting well structures of different smoothness using the BVD principle.

#### *4.2. The single phase compressible flow*

In this section, we assess the ability of BVD schemes to capture shock waves and vortices in single phase gas dynamic problems. The Euler equations of ideal gas with the ratio of specific heats of *γ* = <sup>1</sup>*.*4 are solved by the proposed BVD schemes. The HLL Riemann solver [\[51](#page-27-0)] is applied to compute the numerical flux.

#### *4.2.1. The one dimensional shock tube problem*

Two-dimensional schemes are applied to a one-dimensional shock tube problem. The computational domain is [0*,* 1] × [0*,* 0*.*1] with 100 triangular elements in *x*-direction and 10 triangular elements in *y*-direction. We consider the following initial conditions [[52](#page-27-0)]:

$$(\rho, u, v, p) = \begin{cases} (1.0, 0.0, 0.0, 1000.0) & \text{if } x < 0.5\\ (1.0, 0.0, 0.0, 0.01) & \text{otherwise} \end{cases}$$

It is the left half of the blast wave problem of Woodward and Colella [\[61\]](#page-27-0). Its solution contains a left rarefaction, a contact wave and a right-moving shock wave. Density of the exact solution and numerical results at *t* = 0*.*012 are shown in Fig. [9](#page-10-0). It is obvious that BVD schemes can resolve the shock and contact wave more sharply than the MUSCL scheme, and there is no oscillation around discontinuities. Fig. [10](#page-10-0) shows cells using THINC/QQ schemes for reconstructing *ρ* when BVD schemes are implemented. For cells around the shock and contact waves, the THINC/QQ with *β<sup>l</sup>* is used (red cells), which can effectively preserve the step-like flow structure.

#### *4.2.2. Supersonic inflow passing through a scram-jet engine*

In scram-jet engine, the interactions between extreme combustion and complicated geometries have not been fully addressed due to numerical difficulties [[62](#page-27-0)–[64\]](#page-27-0). Here, we use benchmark test from [\[65](#page-27-0)]. It includes complex geometry boundaries and complex flow structures of discontinuity. The geometrical configuration shown in Fig. [11](#page-10-0) is divided into 52*,* 724 quasi-uniform triangular elements. Inflow and outflow conditions are imposed on the left and right boundaries respectively. Other boundaries are assumed as slip walls. A supersonic flow of Mach 3.0 enters the engine from the left boundary. The initial condition is <sup>a</sup> uniform flow with *(ρ, <sup>u</sup>, <sup>v</sup>, <sup>p</sup>)* = *(*1*.*4*,* <sup>3</sup>*.*0*,* <sup>0</sup>*,* <sup>1</sup>*.*0*)*.

Fig. [12](#page-11-0) shows Mach contours of the final steady solutions, ranging from 0.0 to 3.0 with 60 constant intervals. It is obvious that the BVD schemes can resolve shock waves better than the MUSCL scheme. Especially for reflected weak shocks around point *(*6*.*5*,* 0*.*0*)*, the MUSCL scheme significantly smears out the discontinuous flow structure, while the BVD schemes provide adequate numerical results. In order to compare to other existing high-order method, we also plot the numerical results from a coarse mesh with 10*,* 417 uniform triangular elements in Fig. [13](#page-11-0), which has the same grid resolution as Fig. 8 in [\[65](#page-27-0)] where a second-order DG method is evaluated. It is observed that although the overall flow structure of two results is quite similar, the three-member BVD scheme can give better resolved flow structures.

#### *4.2.3. A Mach 3 wind tunnel with a forward step*

This benchmark problem was proposed in [\[61\]](#page-27-0) and widely used to verify the capability of numerical schemes in capturing strong shocks and vortical structures [[27](#page-26-0),[66](#page-27-0)–[69](#page-27-0)]. The wind tunnel is 1*.*0 unit high and 3*.*0 unit long. A forward step is located 0*.*6 unit from the left boundary with a height of 0*.*2 unit. The computational domain is divided into triangular elements with a size of 1*/*160 away from the corner but 1*/*320 around the corner, as shown in Fig. [14.](#page-12-0) This mesh was used in [\[66](#page-27-0)] to deal with the singularity around the corner point. The initial condition is the same as the inflow passing through the scram-jet engine in the previous test.

![](_page_14_Figure_2.jpeg)

**Fig. 18.** Density contours at *t* = 0*.*2 computed by MUSCL scheme (a-b), two-member BVD scheme (c-d) and, three-member BVD scheme (e-f).

Fig. [15](#page-12-0) is the density contour at *t* = 4*.*0, ranging from 0.32 to 6.15 with 30 constant intervals. It is seen that with the help of THINC/QQ schemes, BVD schemes can resolve shock waves better than pure MUSCL scheme. Red colored in Fig. [16](#page-12-0) are the cells that use THINC/QQ with *βl*, which coincide well with the regions where shock waves exist. Furthermore, the three-member BVD scheme can resolve the vortices with largely improved solution quality, as shown in Fig. [17.](#page-13-0) As discussed before, it is due in part to the fact that the THINC/QQ with *β<sup>s</sup>* as a reconstruction function includes not only the slope but also the curvature information of second order derivative terms, thus is able to represent better complex flow structures, like vortices. By comparing with the result of a 3rd-order WENO scheme [[66](#page-27-0)], we can see that although the order of both

![](_page_15_Picture_2.jpeg)

Fig. 19. The baroclinic effect to generate vortex chain on material interface in interaction between air shock and R22 cylinder.

![](_page_15_Figure_4.jpeg)

Fig. 20. The computational configuration of shock-R22 bubble interaction.

the MUSCL scheme and the THINC/QQ scheme [45] is around 2nd-order, the combination of them by BVD algorithms can give much better results than a 3rd-order scheme that is more numerically dissipative.

#### 4.2.4. Double Mach reflection

This test was originally proposed in [61]. It is a more challenging case in sense that in addition to strong shock waves the numerical solution of this test includes richer vortical structures, which are sensitive to the intrinsic dissipation of numerical schemes. Thus, it is now used as a standard benchmark test to evaluate a numerical scheme for compressible flow to see if it can simultaneously resolve shocks without spurious oscillation and the abundant vortical structures along the slip lines. Since we use unstructured grids, we can solve this problem directly in its original physical setup, as in [70]. Here, a right-moving Mach 10 shock wave hits a 30 deg ramp. The flow on the left inflow boundary remains the post-shock state, and a outflow condition is imposed on the right boundary. Other parts are all reflective walls. The initial condition is given as:

$$(\rho, u, v, p) = \begin{cases} (8.0, 8.25, 0.0, 116.5) & \text{if } x < 0.1\\ (1.4, 0.0, 0.0, 1.0) & \text{otherwise} \end{cases}$$

A uniform triangular mesh with h = 1/200 is used and the density contours, ranged from 1.5 to 21.5 with 30 equidistant levels, of the numerical results at t = 0.2 are shown in Fig. 18. The overall flow structures agree well with the results of most existing numerical methods, for example [70] where a third-order quadrature-free ADER-FV scheme is used. The shock waves in the results of BVD schemes have more compact thickness than those from the MUSCL scheme. Enlarged in Fig. 18(b)(d)(f), a slip line with density difference and shearing velocity tends to roll up and develop into a chain of vortices due to some amount of numerical viscosity. Whether these vortices can be adequately reproduced or not is usually used as an indicator of numerical dissipation of numerical schemes. The MUSCL and two-member BVD schemes fail in reproducing the vortical structures because of excessive numerical dissipation, which might be fatal in applications where these structures play an important role in the dynamic processes. The three-member BVD scheme largely reduces the numerical dissipation and resolves the vortices sufficiently. A direct comparison reveals that the three-member BVD appears to be less dissipative and provides better resolved vortical structures than many existing high-order schemes on unstructured grids, such as the third-order quadrature-free ADER-FV scheme [70], the fourth-order WENO scheme [66], the fourth-order CLSFV and k-exact FV scheme in [71].

# 4.3. Two-fluid compressible multiphase flows

The BVD schemes presented in this paper can be straightforwardly applied to multiphase compressible flows with moving interfaces. Previous researches show that the THINC reconstruction scheme can effectively prevent smearing-out of moving

![](_page_16_Figure_2.jpeg)

**Fig. 21.** Numerical schlierens for results of interaction between air shock and R22 cylinder. Panels of each row are at the same instant. Left column: the results of the MUSCL scheme (upper half) and the two-member BVD scheme (lower half); Right column: the results of the three-member BVD scheme (upper half) and the two-member BVD scheme (lower half).

interfaces, and the transition layer of the material interface can be kept within a few mesh cells even over long-term simulations [\[39](#page-26-0)[,72](#page-27-0)–[77](#page-27-0)], which is essential for simulating interfacial multiphase fluid dynamics. The combination of the THINC scheme and the MUSCL scheme by BVD algorithm in the structured-grid framework [\[39](#page-26-0)] shows that it can not only resolve clearly material interfaces with compact thickness, but also capture flow more accurately with finer structures than conventional polynomial-based reconstructions such as the MUSCL scheme and the WENO scheme.

In this section, we verify that the presented unstructured MUSCL-THINC/QQ-BVD schemes have excellent performance for interfacial multiphase compressible flow simulations. We solve the five-equation model described above with the BVD schemes proposed in this paper for spatial reconstruction and the HLLC Riemann solver [[78](#page-27-0)] for numerical flux.

#### *4.3.1. Two dimensional shock–R22-cylinder interaction*

As the first benchmark test of two-component flows, we consider a well known shock-bubble interaction problem involving interaction between a shock in air and a cylindrical bubble of R22 gas [[39](#page-26-0)[,79](#page-27-0)–[82](#page-27-0)]. As analyzed in [[83,84](#page-27-0)], vortices will be generated by the baroclinic effect when the shock wave passes through the surface of R22 cylinder. We consider the equation of vorticity *ω* without viscous terms [\[83\]](#page-27-0):

$$\frac{d\boldsymbol{\omega}}{dt} + \boldsymbol{\omega}\nabla \cdot \boldsymbol{V} = \boldsymbol{\omega} \cdot \nabla \boldsymbol{V} + \frac{\nabla \rho \times \nabla p}{\rho^2}.$$
 (22)

![](_page_17_Figure_2.jpeg)

Fig. 22. Continue of Fig. 21.

The last term of Eq. (22) is a source term due to the baroclinic effect. Misalignment of the local gradient of pressure and local gradient of density will lead to a generation of vorticity. Fig. 19 shows the direction of vorticity generated in the interaction between the right-moving air shock and the R22 cylinder bubble. Resultantly, a vortex chain will develop along the material interface. However, the vortices might be suppressed and invisible in simulations if the dissipation of numerical method or model is too strong. Another observation is that if the material interface is smeared out, the gradients in the baroclinic term will be reduced and thus hamper the development of vortices.

Refer to [85] for the experimental work. The computational setup is shown in Fig. 20. A planar right-moving Mach 1.22 shock in air hits a stationary R22 gas cylinder with a diameter d = 50 mm. Both air and R22 gas are treated as ideal gases. The initial condition is given by

$$(\alpha_1,\rho,u,\nu,p,\gamma) = \begin{cases} (\epsilon,3.863~\text{kg/m}^3,0.0,0.0,1.01325\times 10^5~\text{Pa},1.249) & \text{In the R22 cylinder} \\ (1.0-\epsilon,1.686~\text{kg/m}^3,113.5~\text{m/s},0.0,1.59\times 10^5~\text{Pa},1.4) & \text{Post-shock} \\ (1.0-\epsilon,1.225~\text{kg/m}^3,0.0,0.0,1.01325\times 10^5~\text{Pa},1.4) & \text{Otherwise,} \end{cases}$$

where  $\epsilon = 10^{-8}$ . A uniform triangular mesh with h = 0.1875 mm is used. The corresponding number mesh elements is 1,894,892. Reflective wall boundary conditions are implemented to the top and bottom boundaries. Inflow and outflow conditions are imposed on the left and right boundaries respectively.

Numerical schlierens of density,  $\ln(1.0 + |\nabla \rho|)$ , at different instants are shown in Figs. 21 and 22 where the numerical results of different schemes are plotted against each other. It is observed that the BVD schemes proposed in this work can not only resolve sharply the material interface, but also reproduce the vortices generated by the baroclinic mechanism, whereas the MUSCL scheme largely smears the interface and fails to produce the fine vortical structures due to excessive numerical dissipation. Examining the right column of Fig. 21 and 22 reveals that both BVD schemes with two and three members keep adequate sharpness of the material interface as both choose the THINC scheme of  $\beta_l$  for reconstruction.

![](_page_18_Figure_2.jpeg)

Fig. 23. Results from the three-member BVD scheme (upper half) and Fig. 8 of [39] (lower half) with the same mesh number.

Meanwhile, the three-member BVD scheme, with less numerical dissipation and better capability of resolving finer flow structures, gives improved clarity for the complex waves generated from reflection and transmission. Similar to the conclusion for single phase flow, inclusion of the THINC/QQ scheme with  $\beta_s$  in the three-member BVD scheme enriches the vortical structures of small scale.

We also compare our results on unstructured grid with those from the structured BVD scheme in [39] where the reconstruction is conducted in a dimension-wise manner. Shown in Fig. 23, although there are some difference in detail solution structures, the BVD schemes on structured and unstructured grids give similar numerical solutions, and possess common desirable properties, i.e. capturing sharply material interface and resolving fine flow structures with less numerical dissipation.

#### 4.3.2. Two dimensional shock-Helium-bubble interaction

This is another well-known shock-bubble interaction benchmark test used in [79,81,82,86] and others for visual comparison with the experimental images in [85]. It is also used to assess the ability of numerical schemes to capture material interfaces and resolve small vortices due to instability of interfaces. Since the density of Helium gas is lighter than surrounding air, the bubble deforms in a pattern completely different from the R22 bubble case. The shock penetrates through the bubble which is deformed to a toroidal shape and then divides into two parts. Because of the reversal density gradient across the interface, vorticity will be generated in the direction opposite to that in Fig. 19.

The computational domain is the same as that used in the above shock and R22 bubble interaction test shown in Fig. 20. A mesh with uniform triangular cells of h = 0.35 mm is used. The corresponding number of mesh cells is 545,632. Both air and Helium gas are treated as ideal gas. The initial condition is given as

$$(\alpha_1,\rho,u,\nu,p,\gamma) = \begin{cases} (\epsilon,0.167 \text{ kg/m}^3,0.0,0.0,1.01325 \times 10^5 \text{ Pa},1.667) & \text{In the Helium cylinder} \\ (1.0-\epsilon,1.686 \text{ kg/m}^3,113.5 \text{ m/s},0.0,1.59 \times 10^5 \text{ Pa},1.4) & \text{Post-shock} \\ (1.0-\epsilon,1.225 \text{ kg/m}^3,0.0,0.0,1.01325 \times 10^5 \text{ Pa},1.4) & \text{Otherwise}. \end{cases}$$

We plot the experimental results in [85] and our numerical results of volume fraction of Helium gas (including three contour lines of 0.05, 0.5 and 0.95) in Fig. 24. As analyzed in [79], velocities in simulation are faster than experiments. Time indicated in Fig. 24 is the physical time of numerical simulation. It can be seen that the evolution of the bubble shape of numerical simulations agrees well with the experimental images. BVD schemes can keep very thin material interfaces, whereas the MUSCL scheme diffuses them to several cell elements and suppresses the small vortices along the interface generated from the baroclinic effect and shear instability. Results of two BVD schemes are almost visually identical, except

![](_page_19_Figure_2.jpeg)

**Fig. 24.** Volume fraction of Helium gas of the air-shock and Helium bubble interaction test. Left column: experimental images; Middle column: numerical results of MUSCL scheme (upper half) and the two-member BVD scheme (lower half); Right column: numerical results of the three-member BVD scheme (upper half) and the two-member BVD scheme (lower half).

![](_page_20_Figure_2.jpeg)

Fig. 25. The computational domain and an illustration of the triangular unstructured mesh for divergent RMI simulation.

**Table 2**Initial conditions for divergent RMI.

| Region | ρ     | р       | и     | γ    |
|--------|-------|---------|-------|------|
| Α      | 2.139 | 2.317e5 | 217.5 | 1.4  |
| В      | 1.204 | 1.013e5 | 0.0   | 1.4  |
| C      | 6.143 | 1.013e5 | 0.0   | 1.09 |

for some fine structures such as the tail of bubbles and shape of vortices. It should be noticed that, the two-member BVD scheme slightly smears the interface in a few cells around the tail of the bubble where the interface is largely deformed, while the three-member BVD scheme can keep the interface sharp almost everywhere.

#### 4.3.3. Divergent Richtmyer-Meshkov instability (RMI) of a light/heavy interface

This test case is used to show the advantage of proposed schemes in dealing with geometrical configurations which are difficult for schemes developed on structured grids. The experimental setup and results are described in [87]. As shown in Fig. 25, the divergent region of shock tube is separated into three parts: region A for post-shock air, B for pre-shock air and C for SF<sub>6</sub> gas. Table 2 summarizes the initial conditions of each part for the corresponding experimental case 4-24 in [87]. The initial interface between regions B and C has single-mode perturbation defined by  $r(\theta) = R_0 + a_0 cos(n\theta)$ , where  $R_0 = 130$  mm,  $a_0 = 4$  mm and n = 24. A quasi-uniform triangular mesh was used to precisely represent the divergent walls of the channel. The size of mesh cells is about 1.0 mm, and 39,006 elements are used to partition the whole computational domain

Fig. 26 shows the numerical results of MUSCL, twoand three-member BVD schemes against the experimental schlierens at three instances. The structures of right-moving shock and interfaces captured by all three numerical schemes are in good agreement with experimental results. At the early stage, the interfaces are resolved with reasonable transition thickness. However, due to excessive numerical dissipation, the MUSCL scheme smears the interface over a wider range as computation proceeds. As shown in Fig. 27, the BVD schemes can select the THINC/QQ ( $\beta_l$ ) reconstruction function for volume fraction function at interface regions. As a consequence, the BVD schemes can resolve sharp interfaces through the whole simulation. This is very important in RMI simulations since density gradient essentially affects the mixing process. A smeared interface weakens the interface instabilities and the growth of vortices, which can be seen more evidently at the later stage of the interface evolution as shown in Fig. 28. Besides, three-member BVD scheme performs better than the two-member BVD scheme in resolving weak discontinuities. We plot the density gradient and cells where the THINC/QQ is selected to reconstruct  $\alpha_1\rho_1$  for both twoand three-member BVD schemes in Fig. 29. In the three-member BVD scheme, the THINC/QQ ( $\beta_s$ ) is chosen for weak shocks, which are otherwise computed by the MUSCL scheme. It effectively improves solution of the shock structures.

### 4.3.4. Compressible triple point problem

As the last numerical test, we simulated a two-material three-state 2D Riemann problem to assess the performance of BVD schemes on quadrilateral elements. This benchmark test is widely used to validate the robustness of Lagrangian or ALE methods for both single-phase and multi-phase compressible flows [88–91] and the ability of interface capturing schemes to resolve sharp interfaces [72,92]. An important feature in the results of this test case is rich vortices generated at material interfaces due to the Kelvin-Helmholtz instability [92], which provides a good test bed to evaluate both the robustness and dissipation error of numerical schemes.

The computational domain and initial condition are given in Fig. 30. An outflow boundary condition is applied to the right boundary, and slip-wall boundary for all other boundaries. A uniform quadrilateral mesh of  $1792 \times 768$  is used for all computations.

Fig. 31 shows the density gradient and vorticity of results at t = 5.0 computed by the MUSCL scheme. Figs. 32 and 33 display density, density gradient, footprint of the THINC/QQ scheme and vorticity of the results computed by the two BVD schemes proposed in this paper. It is obvious that the MUSCL scheme smears the interface between two fluids, while the

![](_page_21_Figure_2.jpeg)

**Fig. 26.** Density gradient (|∇*ρ*|) of divergent RMI. Left column: numerical results of the two-member BVD scheme (upper half) vs experimental images; Right column: numerical results of the three-member BVD scheme (upper half) vs the MUSCL scheme.

BVD schemes can keep the interface very sharp. In order to access quantitatively the numerical diffusion that smears the interface, we show in Fig. [34](#page-24-0) the local distribution of volume fraction value (*α*1) in the transition region of the interface computed by MUSCL scheme in Fig. [31](#page-23-0)(a) and the two-member BVD scheme (same as that of the three-member BVD scheme). It is obvious that the MUSCL scheme diffuses the interface over 30 cells between *α*<sup>1</sup> = 0 (blue) and *α*<sup>1</sup> = 1 (red), whereas the BVD schemes keep the interface thickness within 3 ∼ 5 cells through the whole computation.

![](_page_22_Figure_2.jpeg)

**Fig. 27.** Volume fraction fields at *t* = 950 μs. Left: numerical results of the three-member BVD scheme (upper half) vs the MUSCL scheme (lower half). Right: numerical results of the two-member BVD scheme (upper half) and cells where THINC/QQ was used to reconstruct *α*<sup>1</sup> (Red cells).

![](_page_22_Figure_4.jpeg)

**Fig. 28.** Volume fraction fields at *t* = 1100 μs and *t* = 1200 μs computed by the three-member BVD scheme (upper half) and the MUSCL scheme (lower half).

From Fig. [31\(](#page-23-0)b) we can see that almost all the small flow structures around the interfaces are smeared by the MUSCL scheme because of excessive numerical dissipation. However, as shown in Figs. [32](#page-24-0)(b) and [33](#page-24-0)(b), the BVD schemes reproduce the complex flow structures with substantially improved resolution, and are able to capture the small vortical structures, which might be partly due to the high order geometric terms in THINC/QQ reconstruction and reduced numerical dissipation. Our results of BVD schemes are comparable with results in [\[92](#page-27-0)], where a finer mesh of 3584 × 1536 was used (see Fig. [35\)](#page-25-0).

# **5. Conclusion remarks**

In this work, we present two novel variants of the BVD schemes for unstructured grids. The novelty of the new schemes lies in (i) a multi-dimensional formation for boundary variation (BV) calculation has been proposed to take account of the geometrical configuration of unstructured mesh element, which has not been considered so far in the formulations for structured grids, and (ii) a one-step BVD algorithm is devised to choose the most suitable function for spatial reconstruction from multiple candidate functions.

For sake of algorithmic simplicity and practical utility for unstructured grids, the MUSCL scheme and the multidimensional THINC/QQ scheme with different smoothness are used as the candidate functions for spatial reconstruction based on the BVD principle. The one-step BVD algorithm proposed in this paper gives reliable selection of the most suitable function for reconstruction, without any ad hoc threshold. The resulting finite volume formulation effectively minimizes

![](_page_23_Figure_2.jpeg)

**Fig. 29.** Density gradient (|∇*ρ*|) at *<sup>t</sup>* = 540 μs. Left: numerical results of the three-member BVD scheme (upper half) vs the two-member one (lower half); Right: cells using THINC/QQ to reconstruct *α*1*ρ*<sup>1</sup> (Red cells use *β<sup>l</sup>* and green cells use *β<sup>s</sup>* ).

![](_page_23_Figure_4.jpeg)

**Fig. 30.** The computational domain and initial conditions of the triple point problem.

![](_page_23_Figure_6.jpeg)

**Fig. 31.** Results of the MUSCL scheme at *t* = 5*.*0.

numerical dissipation. In spite of the second-order accuracy of these candidate functions, the resultant BVD schemes can resolve the vortical and discontinuous flow structures with superior solution quality in comparison with other existing methods using even higher-order polynomial reconstructions.

The proposed schemes have been extensively verified with various challenging benchmark tests, ranging from scalar advection equation to single and multi-phase inviscid compressible flows, where solutions include strong discontinuities and smooth flow structures. Results show that our BVD algorithm can preserve solution properties and reduce numerical dissipation effectively. For advection or interface tracking problem, BVD schemes can limit discontinuities into few cells which can be controlled by parameter of steepness. In gas dynamic problems, the BVD scheme can simultaneously resolve sharp shock waves and vortical flow structures which are heavily smeared out by other existing methods. It is also found that the three-member BVD scheme that including an additional THINC reconstruction with a mild slope can significantly improve the ability to capture vortical structures, surpassing other conventional methods that use higher polynomial and limiting

![](_page_24_Picture_2.jpeg)

![](_page_24_Picture_3.jpeg)

**Fig. 32.** Results of the two-member BVD scheme at *t* = 5*.*0.

![](_page_24_Picture_7.jpeg)

![](_page_24_Picture_8.jpeg)

**Fig. 33.** Results of the three-member BVD scheme at *t* = 5*.*0.

![](_page_24_Figure_12.jpeg)

**Fig. 34.** Volume fraction of fluid 1 (*γ* = <sup>1</sup>*.*5).

![](_page_25_Picture_2.jpeg)

**Fig. 35.** Vorticity: the left part is picked from [[92](#page-27-0)] with a finer mesh (3584 × 1536); the right part is the results of our three-member BVD scheme on a 1792 × 768 mesh.

projection for reconstructions. For compressible multi-component flows with moving interface, BVD schemes can capture well the material interface with narrow transition thickness through the whole simulations, which is quite challenging for other conventional high resolution schemes, particularly for the simulations on unstructured grids.

This paper presents a new path to construct robust, reliable and efficient finite volume method on unstructured grids with superior accuracy for both smooth and discontinuous solutions. The two proposed BVD schemes have been verified as competitive alternatives to other existing numerical methods for simulating the complex flows targeted in this paper.

### **CRediT authorship contribution statement**

**Lidong Cheng:** Methodology, Validation, Visualization, Writing, Investigation. **Xi Deng:** Methodology, Conceptualization, Formal analysis, Supervision, Writing. **Bin Xie:** Methodology, Formal analysis, Software, Writing, Supervision. **Yi Jiang:** Validation, Supervision, Investigation. **Feng Xiao:** Methodology, Conceptualization, Supervision, Writing, Project administration

#### **Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# **Acknowledgement**

Lidong Cheng and Yi Jiang would like to thank International Graduate Exchange Program of Beijing Institute of Technology for funding this collaborative research. Xi Deng would like to express his gratitude for funding from the Engineering and Physical Sciences Research Council (EP/R030340/1) that supports his research career at Imperial College London. Bin Xie is funded by National Natural Science Foundation of China (grant no. 11802178). Feng Xiao was supported in part by the fund from JSPS (Japan Society for the Promotion of Science) under Grant Nos. 17K18838, 18H01366 and 19H05613.

Lidong Cheng would like to give special thanks to Dr. Peng Jin and Dr. Siengdy Tann at Tokyo Institute of Technology for their inspirational discussions.

Xi Deng greatly appreciates the strong support he has received from Dr. Peter Vincent.

# **References**

- [1] T. Barth, P. Frederickson, Higher order solution of the Euler equations on unstructured grids using quadratic [reconstruction,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB637B17AF08ACED8850C18CCCDE915DAs1) in: 28th Aerospace Sciences [Meeting,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB637B17AF08ACED8850C18CCCDE915DAs1) vol. 13, 1990.
- [2] T. Barth, Recent developments in high order k-exact [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib61620957A1443C946A143CF99A7D24FAs1) on unstructured meshes, in: 31st Aerospace Sciences Meeting, vol. 668, 1993.
- [3] M. Delanaye, Y. Liu, Quadratic reconstruction finite volume schemes on 3D arbitrary unstructured polyhedral grids, in: 14th [Computational](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF7AB469D1DC79166FC874DADCC0DD854s1) Fluid Dynamics [Conference,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF7AB469D1DC79166FC874DADCC0DD854s1) 1999, p. 3259.
- [4] B. Xie, S. Ii, A. Ikebata, F. Xiao, A multi-moment finite volume method for [incompressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib055FE8002082BF36D2F3BF976695DE94s1) Navier–Stokes equations on unstructured grids: volume[average/point-value](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib055FE8002082BF36D2F3BF976695DE94s1) formulation, J. Comput. Phys. 277 (2014) 138–162.
- [5] B. Xie, F. Xiao, Two and three dimensional multi-moment finite volume solver for [incompressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib463B62C3521E1362024AAF86E1159A6As1) Navier–Stokes equations on unstructured grids with arbitrary [quadrilateral](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib463B62C3521E1362024AAF86E1159A6As1) and hexahedral elements, Comput. Fluids 104 (2014) 40–54.
- [6] B. Xie, F. Xiao, A multi-moment constrained finite volume method on arbitrary unstructured grids for [incompressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib38D0AD1F8D2E15FB536ABED62765AA8As1) flows, J. Comput. Phys. 327 (2016) [747–778.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib38D0AD1F8D2E15FB536ABED62765AA8As1)
- [7] B. Xie, X. Deng, Z. Sun, F. Xiao, A hybrid [pressure–density-based](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibAC2A6C671B105F3F00EE3FBE72584CF2s1) Mach uniform algorithm for 2D Euler equations on unstructured grids by using [multi-moment](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibAC2A6C671B105F3F00EE3FBE72584CF2s1) finite volume method, J. Comput. Phys. 335 (2017) 637–663.
- [8] X. Deng, B. Xie, H. Teng, F. Xiao, High resolution [multi-moment](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4F7A41F8421B3C08EF3519AFB7A97BBBs1) finite volume method for supersonic combustion on unstructured grids, Appl. Math. Model. 66 (2019) [404–423.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4F7A41F8421B3C08EF3519AFB7A97BBBs1)

- [9] W. Reed, T. Hill, Triangular mesh methods for the neutron transport equation, Technical Report LA-UR-73-479, Los Alamos Scientific Laboratory, Los Alamos.
- [10] B. Cockburn, S. Hou, C.-W. Shu, The Runge-Kutta local projection [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib21B0A67A3DEC3C8FE0B3DFC02E7098EBs1) Galerkin finite element method for conservation laws. IV. The multidi[mensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib21B0A67A3DEC3C8FE0B3DFC02E7098EBs1) case, Math. Comput. 54 (190) (1990) 545–581.
- [11] H.T. Huynh, A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods, in: 18th AIAA [Computational](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8B5E3D09558F38A9EA48869564838B5Es1) Fluid Dynamics [Conference,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8B5E3D09558F38A9EA48869564838B5Es1) 2007, p. 4079.
- [12] P.E. Vincent, P. Castonguay, A. Jameson, A new class of high-order energy stable flux [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib2BB43A7AC4D0DF618338EE7A5B05CE99s1) schemes, J. Sci. Comput. 47 (1) (2011) 50–72.
- [13] P. Castonguay, P.E. Vincent, A. Jameson, A new class of high-order energy stable flux [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib950A2B2BD0F931A2D9E7B234E2895629s1) schemes for triangular elements, J. Sci. Comput. 51 (1) (2012) [224–256.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib950A2B2BD0F931A2D9E7B234E2895629s1)
- [14] F.D. Witherden, A.M. Farrington, P.E. Vincent, PyFR: an open source framework for solving [advection–diffusion](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB65AC5E25E96C7AB6AE756BC72C2ABD7s1) type problems on streaming architectures using the flux [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB65AC5E25E96C7AB6AE756BC72C2ABD7s1) approach, Comput. Phys. Commun. 185 (11) (2014) 3028–3040.
- [15] S. Lou, C. Yan, L.-B. Ma, Z.-H. Jiang, The flux [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibC6CAF8A926F68A34BB38CC60E1767493s1) method with Lax–Wendroff type temporal discretization for hyperbolic conservation laws, J. Sci. [Comput.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibC6CAF8A926F68A34BB38CC60E1767493s1) 82 (2) (2020) 42.
- [16] Z. Wang, H. Gao, A unifying lifting collocation penalty formulation including the discontinuous Galerkin, spectral [volume/difference](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib6B3E4FBEFADA3D072873D75927D87BF6s1) methods for [conservation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib6B3E4FBEFADA3D072873D75927D87BF6s1) laws on mixed grids, J. Comput. Phys. 228 (21) (2009) 8161–8186.
- [17] T. Haga, H. Gao, Z. Wang, High-order unifying [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF224C0022E761DA21AF8A22D8C86A38Cs1) formulation for 3D mixed grids, in: 48th AIAA Aerospace Sciences Meeting, 2010, AIAA Paper [2010–540.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF224C0022E761DA21AF8A22D8C86A38Cs1)
- [18] H.T. Huynh, High-order methods including discontinuous Galerkin by [reconstructions](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib3A9FCFBCCC5F091FF7CC6F14074164DAs1) on triangular meshes, in: 49th AIAA Aerospace Sciences Meeting, 2011, AIAA paper [2011–44.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib3A9FCFBCCC5F091FF7CC6F14074164DAs1)
- [19] H. Huynh, Z. Wang, P. Vincent, High-order methods for [computational](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib58F422919BAA2CCBEDD7C165C795B58Bs1) fluid dynamics: a brief review of compact differential formulations on unstructured grids, Comput. Fluids 98 (2) (2014) [209–220.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib58F422919BAA2CCBEDD7C165C795B58Bs1)
- [20] A. Harten, High resolution schemes for hyperbolic [conservation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8014B540B4E94DEC0213393FE4ABA0A3s1) laws, J. Comput. Phys. 49 (3) (1983) 357–393.
- [21] T. Barth, D. Jespersen, The design and application of upwind schemes on [unstructured](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF99E57EC2A34412E7973FA11EAC61CEDs1) meshes, in: 27th Aerospace Sciences Meeting, vol. 366, 1989.
- [22] S. Spekreijse, Multigrid solution of monotone second-order [discretizations](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib0CBC6DD2F5F9B0FA17261F04502C77C1s1) of hyperbolic conservation laws, Math. Comput. 49 (179) (1987) 135–155.
- [23] J.S. Park, S.-H. Yoon, C. Kim, [Multi-dimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib011546E025FF2E0C90265BA274D1D9F7s1) limiting process for hyperbolic conservation laws on unstructured grids, J. Comput. Phys. 229 (3) (2010) [788–812.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib011546E025FF2E0C90265BA274D1D9F7s1)
- [24] J.S. Park, C. Kim, [Multi-dimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8746848D3FC0F1B420114F42F82F17B2s1) limiting process for finite volume methods on unstructured grids, Comput. Fluids 65 (2012) 8–24.
- [25] O. Friedrich, Weighted essentially [non-oscillatory](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib064A635CF42C793FBA202D2BBAF7AE18s1) schemes for the interpolation of mean values on unstructured grids, J. Comput. Phys. 144 (1) (1998) [194–212.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib064A635CF42C793FBA202D2BBAF7AE18s1)
- [26] A. Haselbacher, A WENO [reconstruction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib566C4816C91CB10FD30062833AF7B6B7s1) algorithm for unstructured grids based on explicit stencil construction, in: 43rd AIAA Aerospace Sciences [Meeting](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib566C4816C91CB10FD30062833AF7B6B7s1) and Exhibit, 2005, p. 879.
- [27] W. Wolf, J. Azevedo, High-order ENO and WENO schemes for [unstructured](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib85970C5ADF7B1C5403CA30EA40698F5Ds1) grids, Int. J. Numer. Methods Fluids 55 (10) (2007) 917–943.
- [28] V. Titarev, P. Tsoutsanis, D. Drikakis, WENO schemes for [mixed-element](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib45A8ED3E8B6774DA1A2463713DF21FD2s1) unstructured meshes, Commun. Comput. Phys. 8 (3) (2010) 585.
- [29] P. Tsoutsanis, V.A. Titarev, D. Drikakis, WENO schemes on arbitrary [mixed-element](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib33FCF1F5046A43CBD95D70E10CC27847s1) unstructured meshes in three space dimensions, J. Comput. Phys. 230 (4) (2011) [1585–1601.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib33FCF1F5046A43CBD95D70E10CC27847s1)
- [30] P.-O. Persson, J. Peraire, Sub-cell shock capturing for [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib99CDCA539FE9BC33448B6545A4232B39s1) Galerkin methods, in: 44th AIAA Aerospace Sciences Meeting and Exhibit, vol. 112, [2006.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib99CDCA539FE9BC33448B6545A4232B39s1)
- [31] G.E. Barter, D.L. Darmofal, Shock capturing with PDE-based artificial viscosity for DGFEM: part I. [Formulation,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8A4A488B9744F493372791B87FCE70CCs1) J. Comput. Phys. 229 (5) (2010) [1810–1827.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8A4A488B9744F493372791B87FCE70CCs1)
- [32] M. Dumbser, R. Loubère, A simple robust and accurate a posteriori sub-cell finite volume limiter for the [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibD1633D2F9C2CBA987E7E2FA2E3415F34s1) Galerkin method on unstructured meshes, J. Comput. Phys. 319 (2016) [163–199.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibD1633D2F9C2CBA987E7E2FA2E3415F34s1)
- [33] M. Dumbser, O. Zanotti, R. Loubère, S. Diot, A posteriori subcell limiting of the [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib607293AF18D7BFAE51FBF025AA077BD5s1) Galerkin finite element method for hyperbolic conservation laws, J. [Comput.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib607293AF18D7BFAE51FBF025AA077BD5s1) Phys. 278 (2014) 47–75.
- [34] Z. Sun, S. Inaba, F. Xiao, Boundary Variation Diminishing (BVD) [reconstruction:](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib0A56605B98F9E957938BD3388733C902s1) a new approach to improve Godunov schemes, J. Comput. Phys. 322 (2016) [309–325.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib0A56605B98F9E957938BD3388733C902s1)
- [35] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using [hyperbolic](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib838817D04619752DB486809B40FA10E6s1) tangent function, Int. J. Numer. Methods Fluids 48 (9) (2005) [1023–1040.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib838817D04619752DB486809B40FA10E6s1)
- [36] F. Xiao, S. Ii, C. Chen, Revisit to the THINC scheme: a simple algebraic VOF algorithm, J. Comput. Phys. 230 (2011) [7086–7092.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF08FEECCB787FFC0D29987A9BC60FAB0s1)
- [37] G.-S. Jiang, C.-W. Shu, Efficient [implementation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibA5621200215DFFD0614591E738773A94s1) of weighted ENO schemes, J. Comput. Phys. 126 (1) (1996) 202–228.
- [38] X. Deng, B. Xie, R. Loubère, Y. Shimizu, F. Xiao, Limiter-free [discontinuity-capturing](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib2B947BBD274DCE5D858B17BC14863427s1) scheme for compressible gas dynamics with reactive fronts, Comput. Fluids 171 [\(2018\)](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib2B947BBD274DCE5D858B17BC14863427s1) 1–14.
- [39] X. Deng, S. Inaba, B. Xie, K.-M. Shyue, F. Xiao, High fidelity [discontinuity-resolving](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibE7FE48506637DFC439EBAE0D585FA977s1) reconstruction for compressible multiphase flows with moving [interfaces,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibE7FE48506637DFC439EBAE0D585FA977s1) J. Comput. Phys. 371 (2018) 945–966.
- [40] X. Deng, B. Xie, F. Xiao, [Multimoment](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib256EC32E1FD4A7B2A23668D83F63370Es1) finite volume solver for Euler equations on unstructured grids, AIAA J. 55 (8) (2017) 2617–2629.
- [41] X. Deng, B. Xie, F. Xiao, A finite volume multi-moment method with boundary variation diminishing principle for Euler equation on [three-dimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibEE5A163FFBE564E6901B78FFEAD0437Bs1) hybrid [unstructured](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibEE5A163FFBE564E6901B78FFEAD0437Bs1) grids, Comput. Fluids 153 (2017) 85–101.
- [42] X. Deng, Y. Shimizu, F. Xiao, A fifth-order shock capturing scheme with two-stage boundary variation [diminishing](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibFBC546FCA28177BD37644E92424A81A3s1) algorithm, J. Comput. Phys. 386 (2019) [323–349.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibFBC546FCA28177BD37644E92424A81A3s1)
- [43] X. Deng, Y. Shimizu, B. Xie, F. Xiao, Constructing higher order [discontinuity-capturing](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib12197E98A26B6B12A103CF71DD1BDF44s1) schemes with upwind-biased interpolations and boundary variation [diminishing](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib12197E98A26B6B12A103CF71DD1BDF44s1) algorithm, Comput. Fluids (2020) 104433.
- [44] X. Deng, Z.-H. Jiang, F. Xiao, C. Yan, Implicit large eddy simulation of [compressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBA5A88750CBEA0942D11A01412DED442s1) turbulence flow with PnTmBVD scheme, Appl. Math. Model. 77 [\(2020\)](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBA5A88750CBEA0942D11A01412DED442s1) 17–31.
- [45] B. Xie, F. Xiao, Toward efficient and accurate interface capturing on arbitrary hybrid [unstructured](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib00C85DE98200ECCDFD801A04B519558As1) grids: the THINC method with quadratic surface [representation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib00C85DE98200ECCDFD801A04B519558As1) and Gaussian quadrature, J. Comput. Phys. 349 (2017) 415–440.
- [46] B. Xie, X. Deng, H. Nakayama, S. Liao, F. Xiao, High-order [multi-moment](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib013C952254501C4BBB5E4801E73487D3s1) finite volume method with smoothness adaptive fitting reconstruction for [compressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib013C952254501C4BBB5E4801E73487D3s1) viscous flow, J. Comput. Phys. 394 (2019) 559–593.
- [47] G. Allaire, S. Clerc, S. Kokh, A [five-equation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib80753285D088CEFB88BF152006163F26s1) model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2) (2002) [577–616.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib80753285D088CEFB88BF152006163F26s1)
- [48] R. Abgrall, How to prevent pressure oscillations in [multicomponent](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib524240F0AE5F547F302AF6F5A5DEA614s1) flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) [150–160.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib524240F0AE5F547F302AF6F5A5DEA614s1)
- [49] K.-M. Shyue, A fluid-mixture type algorithm for compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibA02725A23A6D8F19B6195669A1F780A4s1) flow with Mie–Grüneisen equation of state, J. Comput. Phys. 171 (2) (2001) [678–707.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibA02725A23A6D8F19B6195669A1F780A4s1)
- [50] S. Godunov, A finite difference method for the computation of [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB0CD0BA55679AF79160DFF1060BA8A2Es1) solutions of the equations of fluid dynamics, Sb. Math. 47 (8–9) (1959) [357–393.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB0CD0BA55679AF79160DFF1060BA8A2Es1)

- [51] A. Harten, P. Lax, B. van Leer, On upstream differencing and [Godunov-type](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibA3C1910067B42D9D3C273B386D91CEB4s1) schemes for hyperbolic conservation laws, SIAM Rev. 25 (1983) 35–61.
- [52] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical [Introduction,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8A6B9A998B5B18A5007296AB5E7C42A5s1) Springer Science & Business Media, 2013.
- [53] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong [stability-preserving](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib9F48B9F9C882AD65A5F5D489E6B1EFF8s1) high-order time discretization methods, SIAM Rev. 43 (1) (2001) 89–112.
- [54] A. Harten, P.D. Lax, B.v. Leer, On upstream differencing and [Godunov-type](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibCA5F5AC6E9890C1CB5A0A26F9203652As1) schemes for hyperbolic conservation laws, SIAM Rev. 25 (1) (1983) 35–61.
- [55] X. Deng, P. Boivin, F. Xiao, A new [formulation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib9C5DC974D1FC405125C41A5AF8C51F5Bs1) for two-wave Riemann solver accurate at contact interfaces, Phys. Fluids 31 (4) (2019) 046102.
- [56] S. Tann, X. Deng, Y. Shimizu, R. Loubère, F. Xiao, Solution Property Preserving Reconstruction for Finite Volume Scheme: a BVD+ MOOD framework, Int. J. Numer. Methods Fluids.
- [57] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibDCF6F51A42B518C6858FB38058C0D5B5s1) flow problems, J. Comput. Phys. 219 (2) (2006) 715–732.
- [58] E. Johnsen, On the treatment of contact [discontinuities](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibDB38D4F9EECD1DB3448BEA5C328F4840s1) using WENO schemes, J. Comput. Phys. 230 (24) (2011) 8665–8668.
- [59] R.J. LeVeque, [High-resolution](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib011ADD9F3BA56CEC94E5DDE4280E070Bs1) conservative algorithms for advection in incompressible flow, SIAM J. Numer. Anal. 33 (2) (1996) 627–665.
- [60] S.T. Zalesak, Fully [multidimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib43A393FE9E832107E50BA1165A053E09s1) flux-corrected transport algorithms for fluids, J. Comput. Phys. 31 (3) (1979) 335–362.
- [61] P. Woodward, P. Colella, The numerical simulation of [two-dimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4ED454C95DA6624B33F1645BD8F378DEs1) fluid flow with strong shocks, J. Comput. Phys. 54 (1) (1984) 115–173.
- [62] K. Wang, H. Teng, P. Yang, H.D. Ng, Numerical [investigation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4B2684E0F0674738490C6B30C4D9F19Es1) of flow structures resulting from the interaction between an oblique detonation wave and an upper [expansion](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4B2684E0F0674738490C6B30C4D9F19Es1) corner, J. Fluid Mech. 903 (2020).
- [63] P. Yang, H.D. Ng, H. Teng, Numerical study of [wedge-induced](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibF3E8EC4E3F2B79B963B8D5076E0ED7FEs1) oblique detonations in unsteady flow, J. Fluid Mech. 876 (2019) 264–287.
- [64] X. Deng, P. Boivin, Diffuse interface modelling of reactive multi-phase flows applied to a sub-critical cryogenic jet, Appl. Math. Model.
- [65] S. Tu, S. Aliabadi, et al., A slope limiting procedure in [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibE380A0CF78947D47AF638D498DE6F202s1) Galerkin finite element method for gasdynamics applications, Int. J. Numer. Anal. Model. 2 (2) (2005) [163–178.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibE380A0CF78947D47AF638D498DE6F202s1)
- [66] C. Hu, C.-W. Shu, Weighted essentially [non-oscillatory](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib4D0DF1E2D174D52AFCE7C199690C94B5s1) schemes on triangular meshes, J. Comput. Phys. 150 (1) (1999) 97–127.
- [67] W. Li, Y.-X. Ren, High-order k-exact WENO finite volume schemes for solving gas dynamic Euler equations on [unstructured](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib406A54994B3216BD4D5C70DD58D2DBC5s1) grids, Int. J. Numer. Methods Fluids 70 (6) (2012) [742–763.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib406A54994B3216BD4D5C70DD58D2DBC5s1)
- [68] J. Zhu, J. Qiu, C.-W. Shu, M. Dumbser, Runge–Kutta [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8860FC36B144AC50D2B8B7B1D6B05E57s1) Galerkin method using WENO limiters II: unstructured meshes, J. Comput. Phys. 227 (9) (2008) [4330–4353.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib8860FC36B144AC50D2B8B7B1D6B05E57s1)
- [69] J. Zhu, J. Qiu, Hermite WENO schemes and their application as limiters for Runge-Kutta [discontinuous](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib98E1B516792DC83AAD20354C7DB3E5AAs1) Galerkin method, III: unstructured meshes, J. Sci. Comput. 39 (2) (2009) [293–321.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib98E1B516792DC83AAD20354C7DB3E5AAs1)
- [70] M. Dumbser, M. Käser, V.A. Titarev, E.F. Toro, [Quadrature-free](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB5160D34FA184B51BA6222CE994B0583s1) non-oscillatory finite volume schemes on unstructured meshes for nonlinear hyperbolic systems, J. Comput. Phys. 226 (1) (2007) [204–243.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB5160D34FA184B51BA6222CE994B0583s1)
- [71] Q. Wang, Y.-X. Ren, W. Li, Compact high order finite volume method on unstructured grids II: extension to [two-dimensional](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib529A6FA66D9CAEDE2110E330C9B0904Es1) Euler equations, J. Comput.
- Phys. 314 (2016) [883–908.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib529A6FA66D9CAEDE2110E330C9B0904Es1) [72] K.-M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for [compressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib7ADCBFD013F4FC76D600F02A631A335Es1) two-phase flow: the algebraic THINC approach, J. Comput. Phys. 268
- (2014) [326–354.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib7ADCBFD013F4FC76D600F02A631A335Es1) [73] B. Xie, P. Jin, F. Xiao, An [unstructured-grid](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibEC62A7238EA9AE4FFFF5ADB0561382D1s1) numerical model for interfacial multiphase fluids based on multi-moment finite volume formulation and
- THINC method, Int. J. Multiph. Flow 89 (2017) [375–398.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibEC62A7238EA9AE4FFFF5ADB0561382D1s1) [74] D.P. Garrick, W.A. Hagen, J.D. Regele, An interface capturing scheme for modeling atomization in [compressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib94E3BAA42FFFC5572BEB9B10706937EEs1) flows, J. Comput. Phys. 334 (2017)
- [260–280.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib94E3BAA42FFFC5572BEB9B10706937EEs1) [75] Y. Niu, Y. Chen, T. Yang, F. Xiao, Development of a less-dissipative hybrid AUSMD scheme for [multi-component](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib1EE6D2C14FB235D7D0E6905F54D83150s1) flow simulations, Shock Waves 29
- (2019) [691–704.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib1EE6D2C14FB235D7D0E6905F54D83150s1) [76] A.K. Pandare, H. Luo, J. Bakosi, An enhanced AUSM+up scheme for high-speed [compressible](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib011C33DBF78F994DC9330279DC8D0F1As1) two-phase flows on hybrid grids, Shock Waves 29 (5)
- (2019) [629–649.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib011C33DBF78F994DC9330279DC8D0F1As1) [77] B. Xie, P. Jin, S. Liao, F. Xiao, A conservative solver for surface-tension-driven multiphase flows on collocated unstructured grids, J. Comput. Phys. 401 (2020), <https://doi.org/10.1016/j.jcp.2019.109025>.
- [78] E. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the [HLL-Riemann](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib20BB6697AA40AE71C0D31FDA70CB2BA2s1) solver, Shock Waves 4 (1994) 25–34.
- [79] J.J. Quirk, S. Karni, On the dynamics of a [shock–bubble](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib92C0EDA5717309C2366ABB10B1BF9163s1) interaction, J. Fluid Mech. 318 (1996) 129–163.
- [80] K.-M. Shyue, A [wave-propagation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib2CDBE2DF2304DA0072A5DC4AFC7AB758s1) based volume tracking method for compressible multicomponent flow in two space dimensions, J. Comput. Phys. 215 (1) (2006) [219–244.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib2CDBE2DF2304DA0072A5DC4AFC7AB758s1)
- [81] S. Shankar, S. Kawai, S. Lele, Numerical simulation of [multicomponent](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib77383947022331803461CD287419320Bs1) shock accelerated flows and mixing using localized artificial diffusivity method, in: 48th AIAA Aerospace Sciences Meeting Including the New Horizons Forum and Aerospace [Exposition,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib77383947022331803461CD287419320Bs1) vol. 352, 2010.
- [82] K. So, X. Hu, N.A. Adams, [Anti-diffusion](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib3B342518F9790D99B18CBAC37F134071s1) interface sharpening technique for two-phase compressible flow simulations, J. Comput. Phys. 231 (11) (2012) [4304–4323.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib3B342518F9790D99B18CBAC37F134071s1)
- [83] J. Picone, J. Boris, Vorticity generation by shock [propagation](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibB65BDB18062C5E44608288111D2AA8DAs1) through bubbles in a gas, J. Fluid Mech. 189 (1988) 23–51.
- [84] J. Giordano, Y. Burtschell, [Richtmyer-Meshkov](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib03A945333423A93FE17DF35DE840645Cs1) instability induced by shock-bubble interaction: numerical and analytical studies with experimental [validation,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib03A945333423A93FE17DF35DE840645Cs1) Phys. Fluids 18 (3) (2006) 036102.
- [85] J.-F. Haas, B. Sturtevant, Interaction of weak shock waves with cylindrical and spherical gas [inhomogeneities,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibDBE365D70E82F2B770939875450C45BCs1) J. Fluid Mech. 181 (1987) 41–76.
- [86] C. Liu, C. Hu, Adaptive THINC-GFM for compressible [multi-medium](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib6E12A2E47C6962C7C46538B51544FDC9s1) flows, J. Comput. Phys. 342 (2017) 43–65.
- [87] M. Li, J. Ding, Z. Zhai, T. Si, N. Liu, S. Huang, X. Luo, On divergent [Richtmyer–Meshkov](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib98878F29ED335AB3947B113C67A16E76s1) instability of a light/heavy interface, J. Fluid Mech. 901 (2020) [A38.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib98878F29ED335AB3947B113C67A16E76s1)
- [88] X. Zeng, G. Scovazzi, A frame-invariant vector limiter for flux corrected nodal remap in arbitrary [Lagrangian–Eulerian](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib398764DF33E5C37F2A20DD5628F8AF82s1) flow computations, J. Comput. Phys. 270 (2014) [753–783.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib398764DF33E5C37F2A20DD5628F8AF82s1)
- [89] M. Kucharik, R.V. Garimella, S.P. Schofield, M.J. Shashkov, A comparative study of interface reconstruction methods for [multi-material](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBEF437B6A81CDA5782927A58868D32AAs1) ALE simulations, J. Comput. Phys. 229 (7) (2010) [2432–2452.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBEF437B6A81CDA5782927A58868D32AAs1)
- [90] R. Loubère, P.-H. Maire, M. Shashkov, J. Breil, S. Galera, ReALE: a reconnection-based [arbitrary-Lagrangian–Eulerian](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBE52C5E31523EAC425B17F406C4DFCA0s1) method, J. Comput. Phys. 229 (12) (2010) [4724–4761.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bibBE52C5E31523EAC425B17F406C4DFCA0s1)
- [91] V.A. Dobrev, T.V. Kolev, R.N. Rieben, High-order curvilinear finite element methods for Lagrangian [hydrodynamics,](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib6CC1A9ADD702A6E22C0B38D161CAED32s1) SIAM J. Sci. Comput. 34 (5) (2012) [B606–B641.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib6CC1A9ADD702A6E22C0B38D161CAED32s1)
- [92] S. Pan, L. Han, X. Hu, N.A. Adams, A conservative [interface-interaction](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib924B4498C8376A0ED8AE9E8DA0133392s1) method for compressible multi-material flows, J. Comput. Phys. 371 (2018) [870–895.](http://refhub.elsevier.com/S0021-9991(20)30862-7/bib924B4498C8376A0ED8AE9E8DA0133392s1)