# Accepted Manuscript

Toward efficient and accurate interface capturing on arbitrary hybrid unstructured grids: The THINC method with quadratic surface representation and Gaussian quadrature

Bin Xie, Feng Xiao

PII: S0021-9991(17)30599-5

DOI: <http://dx.doi.org/10.1016/j.jcp.2017.08.028>

Reference: YJCPH 7530

To appear in: *Journal of Computational Physics*

Received date: 3 March 2017 Revised date: 24 July 2017 Accepted date: 14 August 2017

![](_page_0_Picture_8.jpeg)

Please cite this article in press as: B. Xie, F. Xiao, Toward efficient and accurate interface capturing on arbitrary hybrid unstructured grids: The THINC method with quadratic surface representation and Gaussian quadrature, *J. Comput. Phys.* (2017), http://dx.doi.org/10.1016/j.jcp.2017.08.028

This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

## Highlights

- A novel VOF (volume of fluid) scheme for arbitrary unstructured grids.
- Complete quadratic representation of curved interface.
- A straightforward and easy-to-use VOF scheme of great practical significance.
- Verified solution quality competitive to other existing methods.

Toward efficient and accurate interface capturing on arbitrary hybrid unstructured grids: The THINC method with quadratic surface representation and Gaussian quadrature

> Bin Xiea,b,<sup>∗</sup> , Feng Xiaob,<sup>∗</sup>

*aSchool of Naval Architecture, Department of Ocean and Civil Engineering, Shanghai Jiaotong University, Shanghai 200240, China; bSchool of Engineering, Department of Mechanical Engineering, Tokyo Institute of Technology, 4259 Nagatsuta Midori-ku, Yokohama, 226-8502, Japan.*

### Abstract

A novel interface capturing scheme is proposed to compute moving interface on arbitrary hybrid unstructured grids. Different from conventional volume of fluid (VOF) schemes that require complicated geometric manipulations for interface reconstructions, the present method can be viewed as a hybrid geometric/algebraic type VOF approach, i.e. the interface is implictly retrieved from an algebraic function that effectively makes use of the geometrical information, such as normal direction and curvature of the interface. Unlike previous versions of the THINC (tangent of hyperbola interface capturing) method, the interface is represented by a quadratic surface for grid cells of arbitrary shapes in this scheme. The Gaussian quadrature is used to estimate the integration of the cell-wise multi-dimensional hyperbolic tangent reconstruction function which is then used to retrieve the interface from the volume fraction value of the target cell. The Gaussian quadrature is also used to compute the numerical fluxes from the reconstruction function. Numerical accuracy for reconstruction and flux computation can be effectively improved by increasing quadrature points. The whole solution procedure follows the finite volume method for advection transport, and is thus simple and easy to use for cells of arbitrary shapes. As verified in the benckmark tests in this paper, the presented scheme, so-called THINC/QQ(THINC method with quadratic surface representation and Gaussian quadrature) scheme, shows significantly improved geometrical fidelity of interface representation particularly for curved surface. Despite algorithmic simplicity, the solution quality of THINC/QQ is comparable to other existing VOF methods with PLIC geometrical interface reconstructions, and thus an accurate and efficient VOF scheme of great practical significance for unstructured grids.

*Keywords:* multi-phase flow, quadratic interface fitting, VOF method, unstructured hybrid grid, multidimensional THINC method, interface capturing.

## 1. Introduction

The one-fluid model [45] has gained increasing popularity in the simulations of multiphase fluid dynamics owing to the recent innovative progress in the numerical technique for capturing moving interfaces that separate difference fluids and need to be solved as part of the whole numerical solutions. Among the methods of this kind, the volume of fluid (VOF) method[13] is of particular interest since it can ensure rigorous numerical conservation. In a VOF method, volume fraction is defined for a certain fluid and updated by solving a transport equation based on finite volume formulation on a fixed Eulerian mesh. In order to calculate numerical fluxes, special care is required to identify and reconstruct the moving interface to avoid numerical diffusion which tends to smear out the compactness of the transition layer of VOF field.

<sup>∗</sup>Corresponding author: Dr. B. Xie (Email: xie.b.aa@m.titech.ac.jp, Dr. F. Xiao (Email: xiao@es.titech.ac.jp)

In the early version of VOF method, the interface is explicitly reconstructed by simple line interface calculation (SLIC) scheme [26, 18] which makes use of a straight line (2D) or a plane (3D) segment aligning with the coordinate axis to approximate the interface in each cell. More precise representation of the interface can be made by considering the orientation of interface in the geometric reconstructions. The efforts to this end have led to the great success of piecewise linear interface calculation (PLIC) method [56, 57] as well as its advanced variants [29, 35, 37, 38, 30, 36, 22, 4, 3, 41, 24, 10]. Among others, Parker–Youngs method [29] uses simple difference formula to approximate the interface normal, which is at most first order accuracy. The least squares VOF interface reconstruction algorithm (LVIRA) [30] enhances the accuracy by iterative calculation to minimize numerical error of interface reconstruction based on least-square method. To improve efficiency, efficient LVIRA (ELVIRA) [30] is proposed to replace the iterative calculation by selecting the best choice from several reconstruction candidates. An elegent formulation to identify the location of interface from a given volume fraction value was presented in [40]. All these PLIC type VOF schemes are characterized with a common feature that the interface is identified by a line (2D) or a plane (3D) within each cell and requires geometric manipulations. A remarkable implementation of the PLIC type VOF algorithms is found in the AMR (adaptive mesh refinement) Cartesian solvers for incompressible multiphase flows in freewares *Gerris* [31, 32, 33] and *Basilisk* [34], which facilitate numerical simulations of interfacial flows of a broad scale in space. Nevertheless, the numerical procedure for geometric reconstructions involves computational complexity which becomes more substantial when extended to unstructured grids. To our knowledge, there are only a few existing works [25, 1, 42, 16, 23, 5] of PLIC type VOF schemes on unstructured grids and very limited applications [16, 5] in the literature.

As the moving interfaces between different phases of fluids are curved surfaces in general cases, it is natural to see the possibility of representing the interfaces with curved surfaces in a quadratic or higher order form rather than the PLIC reconstructions. Some efforts can be found in the literature to make use of the quadratic surfaces to fit the interfaces in geometric reconstruction [36, 41, 21, 9] where the curvature of interface is taken into account. The numerical results demonstrated that reconstructions with a quadratic fitting can produce more accurate results in comparison with the PLIC type algorithm. Unfortunately, as one can imagine, the quadratic fitting greatly increases the computational complexity much more beyond the linear fitting (plane) in the PLIC algorithm, and all those existing studies are limited to the Cartesian grid in two dimensions.

Being the alternative of the VOF methods that use geometric reconstructions, some schemes have been devised to solve the VOF field without explicit geometric reconstructions as the PLIC algorithm aforementioned. This type of methods, so-called algebraic VOF methods, are in principle based on the finite volume advection schemes, and thus appealing in computational efficiency and algorithmic simplicity, which allows the implementation on unstructured grids being conducted easily and straightforwardly. However, the intrinsic numerical diffusion in Eulerian advection schemes tends to smear out the interface jumps in the volume fraction field, thus degrades the VOF function as an identification function for immiscible multiphase fluids. Thus, amendment must be made by some special techniques to remove numerical diffusion, which is in practice realized through either an intrinsic compressive mechanism or an additional post-processing step. Among the representatives of this kind methods, the flux-corrected transport (FCT) VOF scheme [38] introduces an anti-diffusive flux to counteract the excessive diffusion. Alternatively, the compressive interface capturing scheme for arbitrary meshes (CICSAM) algorithm[46] was proposed to switch between the upwinding and the ULTIMATE-QUICKEST (UQ)[19] schemes based on the normalised variable diagram (NVD) concept. This scheme has been successively improved in various variants (e.g. [7, 59] as well as the references therein) and widely adopted in many mainstream CFD codes on unstructured grids. More recently, Olsson et al. have presented the conservative level-set (CLS) method [27, 28] where re-initialization equation is added as an artificial steepening treatment to keep the thickness of interface compact. Without geometric manipulations, the algebraic type VOF schemes significantly ameliorate algorithmic complexity and make the extension to unstructured grids much more straightforward. However, solution quality of algebraic VOF methods is usually observed inferior to the PLIC type VOF schemes.

As a different strategy to design algebraic VOF methods, tangent of hyperbola interface capturing (THINC) scheme [48] was proposed by making use of a particular interpolation function, i.e. the hyperbolic tangent function, under the framework of conventional finite volume method. The numerical flux computed from the step-like hyperbolic tangent function effectively eliminates the numerical diffusion. This method is a pure advection transport scheme and does not need any numerical manipulation to handle geometric components , like line or surface segments cutting through grid cells, nor special numerical treatment to re-enforce the steepness of the interface jump, hence is

perhaps the most algorithmically simplest one among the existing VOF schemes. Meanwhile, the transition zone of jump can be effectively controlled by adjusting the steepness (or slope) parameter, and gets favorable thickness with specification of an appropriate value. Multi-dimensional interface capturing computation can be implemented by using the 1D THINC method in [48] as a building block through a dimensional splitting procedure. As shown in [53] and [49], numerical solutions of high quality comparable to the PLIC type VOF methods can be obtained if an adaptive steepness parameter is used in the 1D THINC reconstruction. The real-case applications for incompressible multiphase flows have been reported in [54, 55] with appealing numerical results.

Using multi-dimensional hyperbolic tangent function, more sophisticated scheme, the multidimensional THINC (MTHINC) method, is proposed to reconstruct interface in multi-dimensions [14]. A remarkable advantage of the multi-dimensional THINC reconstruction is that it enables including the geometric information, such as the normal direction and curvature of the interface, into the reconstruction function while retaining the algorithmic simplicity. Nevertheless, integrating the multi-dimensional hyperbolic tangent function comes as another technique issue in implementation. In [14], a hybrid integration was introduced to blend a 1D exact integration and Gaussian quadrature. Following the similar idea, unstructured versions, i.e. unstructured MTHINC(UMTHINC) methods, have been also developed for unstructured grids with different element shapes, such as triangular and quadrilateral in 2D and tetrahedral and hexahedral in 3D [50, 52]. All the variants show accuracy remarkably superior to other existing algebraic VOF schemes and rival PLIC type schemes.

Despite the considerable success gained in the above schemes of THINC family, the main bottleneck regarding the integration of multi-dimensional THINC function still demands for further improvements where the second order mixed derivatives regarding to the principle orientation of interface is omitted in MTHINC method to facilitate integration calculation. Another important issue remains unresolved is that the hybrid integration in the existing THINC schemes don't accommodate well the quadratic interface representation for arbitrary grid elements. The extension of quadratic reconstruction to grids other than quadrilateral and hexahedral elements [14, 50], suffers significantly increasing difficulties due to the stumbling block that mainly comes from how to split multi-dimensional integration of hyperbolic tangent function into 1D integral operator on the non-square domain of standard element.

![](_page_4_Figure_4.jpeg)

Figure 1: The computational grid elements and related definitions. From left to right are triangular, quadrilateral, tetrahedral, hexahedral, prismatic and pyramidal elements.

Toward the establishment of efficient and accurate numerical solver for interface capturing on arbitrary hybrid unstructured grids, we present and verify in this paper another novel variant of the THINC method, i.e. THINC/QQ (THINC method with quadratic surface representation and Gaussian quadrature) scheme, for all basic grid elements shown in Fig. 1. As an important update of great practical significance, the THINC/QQ scheme has the following new features: 1) curved surface reconstructed by completely quadratic function is used to cell-wisely fit the moving interface in the THINC formulation for all grid elements in Fig. 1; 2) all integration in multi-dimensional THINC reconstruction is computed by Gaussian quadrature, which applies straightforwardly to grid cells of arbitrary shapes with substantially simplified numerical procedure; and 3) refined reconstruction can be easily achieved by simply increasing the points for quadrature.

The rest of this paper is organized as follows. The numerical details of the new THINC/QQ scheme is introduced in section 2, where the mathematical formulation of quadratic reconstruction is described, followed by numerical procedure of quadrature for multi-dimensional hyperbolic tangent function. The numerical tests are presented in section 3 to verify the solution quality of the presented scheme. We end the paper with some conclusion remarks in section 4.

### 2. Quadratic unstructured multi-dimensional THINC scheme on arbitrary unstructured grids

#### 2.1. Computational grids and preliminary definition

The computational domain is divided into non-overlapped discrete grid cells  $\Omega = \bigcup_{i=1}^N \Omega_i$  including triangular, quadrilateral elements for two dimensions and tetrahedral, hexahedral, prismatic, pyramidal elements for three dimensions. Regarding element  $\Omega_i$  as shown in Fig. 1, we denote vertices by  $\theta_{ik}$  located at  $(x_{ik}, y_{ik}, z_{ik})$  (k = 1, 2, ..., K) where K stands for the number of the cell vertices. The boundary surface segments of element are denoted by  $\Gamma_{ij}$  where subscript ij represents the index of the jth surface (j = 1, 2, ..., J) and J the total number of the cell surfaces. We also denote the central points of boundary segment  $\Gamma_{ij}$  by  $\theta_{ij}$  with outward normal unit vector  $\mathbf{n}_{ij} = (n_{xij}, n_{yij}, n_{zij})$  respectively. We further introduce the notation of  $\theta_{ic}$  for the mass center  $(x_{ic}, y_{ic}, z_{ic})$ ,  $|\Omega_i|$  for the volume, and  $|\Gamma_{ij}|$  for the area of boundary surface  $\Gamma_{ij}$ .

For the purpose of simplicity, we transform arbitrary unstructured grid element  $\Omega_i$  into standard reference element denoted by local coordinate system  $\boldsymbol{\xi} = (\boldsymbol{\xi}, \eta, \zeta)$ . Given the physical coordinate  $(x_{ik}, y_{ik}, z_{ik})$  and variable  $\phi_{ik}$  at each vertex ik, any quantity in the local coordinate can be mapped to that of physical coordinate by the basis function  $\mathcal{N}_{ik}(\boldsymbol{\xi})$  as

$$\begin{bmatrix} 1 \\ x \\ y \\ z \\ \phi \end{bmatrix} = \begin{bmatrix} 1 & 1 & \cdots & 1 & 1 \\ x_{i1} & x_{i2} & \cdots & x_{iK-1} & x_{iK} \\ y_{i1} & y_{i2} & \cdots & y_{iK-1} & y_{iK} \\ z_{i1} & z_{i2} & \cdots & z_{iK-1} & z_{iK} \\ \phi_{i1} & \phi_{i2} & \cdots & \phi_{iK-1} & \phi_{iK} \end{bmatrix} \begin{bmatrix} \mathcal{N}_{i1}(\boldsymbol{\xi}) \\ \mathcal{N}_{i2}(\boldsymbol{\xi}) \\ \vdots \\ \mathcal{N}_{iK-1}(\boldsymbol{\xi}) \\ \mathcal{N}_{iK}(\boldsymbol{\xi}) \end{bmatrix}.$$
(1)

With transformation metric (1), the first-order derivatives can be transformed from global coordinate  $(\phi_{xi}, \phi_{yi}, \phi_{zi})$  to local coordinate  $(\phi_{\xi i}, \phi_{\eta i}, \phi_{\xi i})$  by (2) and (3) as

$$\begin{bmatrix} \phi_{\xi i} \\ \phi_{\eta i} \\ \phi_{\zeta i} \end{bmatrix} = \begin{bmatrix} x_{\xi i} & y_{\xi i} & z_{\xi i} \\ x_{\eta i} & y_{\eta i} & z_{\eta i} \\ x_{\zeta i} & y_{\zeta i} & z_{\zeta i} \end{bmatrix} \begin{bmatrix} \phi_{x i} \\ \phi_{y i} \\ \phi_{z i} \end{bmatrix}$$
(2)

and vice versa

$$\begin{bmatrix} \phi_{xi} \\ \phi_{yi} \\ \phi_{zi} \end{bmatrix} = \begin{bmatrix} \xi_{xi} & \eta_{xi} & \zeta_{xi} \\ \xi_{yi} & \eta_{yi} & \zeta_{yi} \\ \xi_{zi} & \eta_{zi} & \zeta_{zi} \end{bmatrix} \begin{bmatrix} \phi_{\xi i} \\ \phi_{\eta i} \\ \phi_{\zeta i} \end{bmatrix}.$$
(3)

Interested readers may consult [51] for more details.

### 2.2. Volume fraction function and transport equation

Considering computational domain  $\Omega = \Omega^1 \cup \Omega^2$  filled with two kinds of physically immiscible fluids, and fluid  $1(\mathbf{x} \in \Omega^1)$  transits to fluid  $2(\mathbf{x} \in \Omega^2)$  through the interface region  $(\mathbf{x} \in \Gamma)$  of a finite thickness. We define a time-dependent smooth indicator function  $H(\mathbf{x},t)$  of fluid 1 as

$$H(\mathbf{x},t) = \begin{cases} 1 & \mathbf{x} \in \Omega^1, \\ 0 & \mathbf{x} \in \Omega^2, \\ 0 < \phi < 1 & \mathbf{x} \in \Gamma. \end{cases}$$
 (4)

In two-phase case, the indicator function of fluid 2 is simply obtained by  $1 - H(\mathbf{x}, t)$ . In the present work, we consider the immiscible interface that moves at the flow velocity. Given flow velocity field  $\mathbf{u} = (u, v, w)$ , the indicator function is updated in the Eulerian form by the following advection equation,

$$\frac{\partial H}{\partial t} + \nabla \cdot (\mathbf{u}H) = H\nabla \cdot \mathbf{u}. \tag{5}$$

Within each discrete grid element, the volume fraction of fluid 1 then can be define by

$$\overline{\phi}_i(t) = \frac{1}{|\Omega_i|} \int_{\Omega_i(\mathbf{x})} H(\mathbf{x}, t) d\mathbf{x}.$$
 (6)

The governing equation of volume fraction  $\overline{\phi}_i(t)$  can be obtained from the finite volume formulation of advection equation (5) over control volume  $\Omega_i$ ,

$$\frac{\partial \overline{\phi}_{i}(t)}{\partial t} + \frac{1}{|\Omega_{i}|} \sum_{j=1}^{J} \left( v_{n_{ij}} \int_{\Gamma_{ij}} H(\mathbf{x}, t)_{iup} d\Gamma \right) = \frac{\overline{\phi}_{i}(t)}{|\Omega_{i}|} \sum_{j=1}^{J} \left( v_{n_{ij}} \left| \Gamma_{ij} \right| \right), \tag{7}$$

where  $v_{n_{ij}} = \mathbf{u}_{ij} \cdot \mathbf{n}_{ij}$  denotes the surface normal velocity on surface  $\Gamma_{ij}$ . In this work, we employ linear(plane) element and the surface normal velocity  $v_{n_{ij}}$  is assumed to be constant on each surface segment.  $H(\mathbf{x}, t)_{iup}$  stands for the reconstruction function in the upwinding cell of cell surface  $\Gamma_{ij}$  which will be elaborated in the next subsection.

## 2.3. Reconstruction of piecewise indicator function with quadratic multi-dimensional THINC formulation

In order to evaluate the numerical flux  $\int_{\Gamma_{ij}} H(\mathbf{x}, t)_{iup} d\Gamma$  in (7), the indicator function  $H(\mathbf{x}, t)$  is piecewisely approximated by a hyperbolic tangent function in the local coordinate  $\tilde{H}_i(\xi, \eta, \zeta)$  for the target cell  $\Omega_i$  at each time step,

$$\tilde{H}_i(\xi, \eta, \zeta) = \frac{1}{2} \left( 1 + \tanh \left( \beta \left( \mathcal{P}_i(\xi, \eta, \zeta) + d_i \right) \right) \right), \tag{8}$$

where  $\beta$  is the steepness parameter to control the thickness of the transition jump and  $\mathcal{P}_i(\xi, \eta, \zeta) + d_i = 0$  is the equation of interface surface in standard element on local coordinate.

Different from [14, 15, 50], we approximate the interface by using a fully quadratic polynomial that includes the geometric information of the interface as a curved surface,

$$\mathcal{P}_{i}(\xi,\eta,\zeta) = a_{200}\xi^{2} + a_{020}\eta^{2} + a_{002}\zeta^{2} + a_{110}\xi\eta + a_{011}\eta\zeta + a_{101}\xi\zeta$$

$$+ a_{100}\xi + a_{010}\eta + a_{001}\zeta,$$
(9)

where coefficients  $a_{str}(s, r, t = 0, 1, 2 \text{ and } s + r + t \le 2)$  are calculated from the interface normal and curvature tensor which will be described in section 2.3.1. It is noted that including above geometric information in (8) substantially improves the solution quality of numerical results, and makes the scheme more like a hybrid of geometric and algebraic approaches, i.e. the geometric information is effectively but implicitly integrated in an algebraic computation procedure.

The only unknown  $d_i$  in (8) indicates the location of the interface which is determined from the volume fraction values using constrained condition (10),

$$\frac{1}{|\Omega_i|} \int_{\Omega_i(\xi)} \tilde{H}_i(\xi, \eta, \zeta) d\xi d\eta d\zeta = \overline{\phi}_i. \tag{10}$$

Once the indicator function  $\tilde{H}_i(\xi, \eta, \zeta)$  is constructed, the numerical fluxes can be computed by Gauss quadrature formula on cell boundary surfaces and then used to update the volume fraction value  $\overline{\phi}_i$  by the finite volume formulation (7). It should be noted that the reconstruction is only required for cells containing interface which is identified by  $\epsilon \leqslant \phi_i \leqslant 1 - \epsilon$ , where  $\epsilon$  is set to  $10^{-8}$  by default.

### 2.3.1. The unit normal vector and curvature of the interface

Given the VOF value in each cell, we firstly calculate the gradient  $\nabla \phi_{ik} = (\phi_{xik}, \phi_{yik}, \phi_{zik})$  at each vertex point  $(x_{ik}, y_{ik}, z_{ik})$  from surrounding volume fraction values  $\overline{\phi}_{ikl}$ ,  $(l = 1, 2, \dots, L)$ , where ikl is the local index of the neighbouring cells sharing vertex  $\theta_{ik}$ . Considering a linear polynomial at  $(x_{ik}, y_{ik}, z_{ik})$ ,

$$\phi(x, y, z) = \phi_{ik} + \phi_{vik}(x - x_{ik}) + \phi_{vik}(y - y_{ik}) + \phi_{zik}(z - z_{ik}), \tag{11}$$

the unknowns  $\phi_{ik}$ ,  $\phi_{xik}$ ,  $\phi_{yik}$ ,  $\phi_{zik}$  can be evaluated by minimizing the following error functional,

$$I_{ik} = \sum_{l=1}^{L} \left( \frac{1}{|\Omega_{ikl}|} \int_{\Omega_{ikl}} \phi(x, y, z) d\Omega_{ikl} - \overline{\phi}_{ikl} \right)^{2}. \tag{12}$$

It is noted that for linear approximation (11), the volume integral average of  $\phi(x, y, z)$  is equivalent to its value at cell mass center, namely

$$\frac{1}{|\Omega_{ikl}|} \int_{\Omega_{ikl}} \phi(x, y, z) d\Omega_{ikl} = \phi(x_{iklc}, y_{iklc}, z_{iklc})$$
(13)

where  $(x_{iklc}, y_{iklc}, z_{iklc})$  denotes the mass center of cell  $\Omega_{ikl}$ . Given VOF values  $\overline{\phi}_{ikl}$  for the surrounding cells of vertex  $\theta_{ik}$  (the total number of the surrounding cells is L), we formulate the over-determined linear system according to the following constraints

$$\frac{1}{|\Omega_{ikl}|} \int_{\Omega_{ikl}} \phi(x, y, z) d\Omega_{ikl} = \overline{\phi}_{ikl}, \ l = 1, 2, \cdots L, \tag{14}$$

which yielding an over-determined linear system  $\mathbf{M} \cdot \mathbf{A} = \mathbf{B}$ . The  $\mathbf{M}$  denotes the coefficient matrix depending only on the mesh geometry,  $\mathbf{A}$  the unknown variables in (11), i.e. the point values and gradients at vertex point vertex  $\theta_{ik}$  ( $\phi_{ik}$ ,  $\phi_{xik}$ ,  $\phi_{yik}$ ,  $\phi_{zik}$ ).  $\mathbf{B}$  stands for the right-hand-side source vector. The unknowns  $\mathbf{A}$  are uniquely determined by

$$\mathbf{A} = \left(\mathbf{M}^T \mathbf{M}\right)^{-1} \mathbf{M}^T \mathbf{B} \tag{15}$$

following the least-square method. For more details, see Appendix E in [51] as an example for 2D case.

Then the first derivatives on local coordinate  $(\phi_{\xi ik}, \phi_{\eta ik}, \phi_{\zeta ik})$  can be transformed from  $(\phi_{xik}, \phi_{yik}, \phi_{zik})$  straightforwardly by formula (2). The normalized components of derivatives at each vertex is then computed by

$$\begin{cases}
\varphi_{\xi ik} = \phi_{\xi ik} / |\hat{\nabla}\phi_{ik}|, \\
\varphi_{\eta ik} = \phi_{\eta ik} / |\hat{\nabla}\phi_{ik}|, & \text{for } k = 1, 2, \dots K, \\
\varphi_{\xi ik} = \phi_{\xi ik} / |\hat{\nabla}\phi_{ik}|,
\end{cases} \tag{16}$$

where  $\hat{\nabla}\phi_{ik} = (\phi_{\xi ik}, \phi_{\eta ik}, \phi_{\zeta ik})$ .

With  $(\varphi_{\xi ik}, \varphi_{\eta ik}, \varphi_{\xi ik})$  at each vertex, the unit normal vector  $(\varphi_{\xi ic}, \varphi_{\eta ic}, \varphi_{\xi ic})$  and curvature tensors at mass center given by

$$\varphi_{\vartheta\delta ic} = \frac{1}{2} \left( \frac{\partial \varphi_{\vartheta}}{\partial \delta} + \frac{\partial \varphi_{\delta}}{\partial \vartheta} \right) \Big|_{\theta_{ic}}, \quad (\vartheta, \delta = \xi, \, \eta, \, \zeta)$$
(17)

can be approximated by interpolation and differentiation from the piecewise polynomial  $\varphi_{\gamma}(\xi) = \sum_{k=1}^{K} \varphi_{\gamma ik} \mathcal{N}_{ik}(\xi)$  with  $\gamma$  denoting either  $\xi$ ,  $\eta$  or  $\zeta$ , where  $\mathcal{N}_{ik}(\xi)$  is the basis function for the kth vertex as shown in (1). For convenience of readers, we summarized the formulations of unit normals and curvature tensors for different types of elements in Appendix A.

The coefficients  $a_{str}$  of (9), which depend directly on the geometrical features of the interface, i.e. the interface orientation and curvature, then can be uniquely determined via the following constraint conditions,

$$\begin{cases}
\frac{\partial \mathcal{P}_{i}}{\partial \xi}(\boldsymbol{\xi}_{ic}) = \varphi_{\xi ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \xi^{2}}(\boldsymbol{\xi}_{ic}) = \varphi_{\xi^{2}ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \xi \partial \eta}(\boldsymbol{\xi}_{ic}) = \frac{1}{2}\left(\varphi_{\xi \eta ic} + \varphi_{\eta \xi ic}\right), \\
\frac{\partial \mathcal{P}_{i}}{\partial \eta}(\boldsymbol{\xi}_{ic}) = \varphi_{\eta ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \eta^{2}}(\boldsymbol{\xi}_{ic}) = \varphi_{\eta^{2}ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \eta \partial \xi}(\boldsymbol{\xi}_{ic}) = \frac{1}{2}\left(\varphi_{\eta \zeta ic} + \varphi_{\zeta \eta ic}\right), \\
\frac{\partial \mathcal{P}_{i}}{\partial \zeta}(\boldsymbol{\xi}_{ic}) = \varphi_{\zeta ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \zeta^{2}}(\boldsymbol{\xi}_{ic}) = \varphi_{\zeta^{2}ic}, & \frac{\partial^{2} \mathcal{P}_{i}}{\partial \zeta \partial \xi}(\boldsymbol{\xi}_{ic}) = \frac{1}{2}\left(\varphi_{\xi \zeta ic} + \varphi_{\zeta \xi ic}\right).
\end{cases} \tag{18}$$

We give below the explicit expressions of  $a_{str}$  for the convenience of readers,

$$\begin{cases} a_{100} = \varphi_{\xi ic} - \xi_{ic}\varphi_{\xi^{2}ic} - \frac{\eta_{ic}}{2} \left( \varphi_{\xi \eta ic} + \varphi_{\eta \xi ic} \right) - \frac{\zeta_{ic}}{2} \left( \varphi_{\xi \zeta ic} + \varphi_{\zeta \xi ic} \right), \\ a_{010} = \varphi_{\eta ic} - \eta_{ic}\varphi_{\eta^{2}ic} - \frac{\xi_{ic}}{2} \left( \varphi_{\xi \eta ic} + \varphi_{\eta \xi ic} \right) - \frac{\zeta_{ic}}{2} \left( \varphi_{\eta \zeta ic} + \varphi_{\zeta \eta ic} \right), \\ a_{001} = \varphi_{\zeta ic} - \zeta_{ic}\varphi_{\zeta^{2}ic} - \frac{\xi_{ic}}{2} \left( \varphi_{\xi \zeta ic} + \varphi_{\zeta \xi ic} \right) - \frac{\eta_{ic}}{2} \left( \varphi_{\eta \zeta ic} + \varphi_{\zeta \eta ic} \right), \\ a_{200} = \frac{1}{2}\varphi_{\xi^{2}ic}, \ a_{110} = \frac{1}{2} \left( \varphi_{\xi \eta ic} + \varphi_{\eta \xi ic} \right), \\ a_{020} = \frac{1}{2}\varphi_{\eta^{2}ic}, \ a_{011} = \frac{1}{2} \left( \varphi_{\eta \zeta ic} + \varphi_{\zeta \eta ic} \right), \\ a_{002} = \frac{1}{2}\varphi_{\zeta^{2}ic}, \ a_{101} = \frac{1}{2} \left( \varphi_{\xi \zeta ic} + \varphi_{\zeta \xi ic} \right). \end{cases}$$

$$(19)$$

Once  $a_{str}$  are determined, the only unknown  $d_i$  in (8) can be computed by (10) for each cell which will be described in next section. It is obvious that the 1D (in  $\xi$  direction) formulation in [48] is a special case where the interface degrades to a jump plane normal to  $\xi$  direction, and all  $a_{str}$  become zero except  $a_{100} = 1$ .

### 2.3.2. The THINC/QQ reconstruction function

As no general analytical expression is available for integration of multi-dimensional hyperbolic tangent function (8), a hybrid approach is proposed in [14] by making 1D exact integration in the principal direction and computing the integration in other two directions through numerical quadrature.

In this work, we use a fully multi-dimensional Gaussian quadrature to approximate the integration of hyperbolic tangent function (8), which largely simplifies the numerical procedure and applies straightforwardly to the case where the interface is represented by a quadratic surface for arbitrary unstructured grids.

Denoting the coordinates and weights of Gaussian points by  $\xi_{ig}$  and  $\omega_{ig}$  (g = 1, 2, ..., G) in element  $\Omega_i$ , we approximate (10) by Gaussian quadrature as follows,

$$\sum_{g=1}^{G} \omega_{ig} \left( \frac{1}{2} \left( 1 + \tanh \left( \beta \left( \mathcal{P}_{i}(\boldsymbol{\xi}_{ig}) + d_{i} \right) \right) \right) \right) = \overline{\phi}_{i}. \tag{20}$$

It is noted that the weights satisfy  $\sum_{g=1}^{G} \omega_{ig} = 1$ . We re-write  $\tanh \left(\beta \left(\mathcal{P}_{i}(\xi_{ig}) + d_{i}\right)\right)$  in (20) as

$$\tanh\left(\beta\mathcal{P}_{i}(\boldsymbol{\xi}_{ig}) + \beta d_{i}\right) = \frac{\tanh(\beta\mathcal{P}_{i}(\boldsymbol{\xi}_{ig})) + \tanh(\beta d_{i})}{1 + \tanh(\beta\mathcal{P}_{i}(\boldsymbol{\xi}_{ig})) \cdot \tanh(\beta d_{i})}.$$
(21)

We then recast (20) into

$$\sum_{g=1}^{G} \omega_{ig} \frac{\tanh(\beta \mathcal{P}_{i}(\xi_{ig})) + \tanh(\beta d_{i})}{1 + \tanh(\beta \mathcal{P}_{i}(\xi_{ig})) \cdot \tanh(\beta d_{i})} = 2\left(\overline{\phi}_{i} - \frac{1}{2}\right). \tag{22}$$

Given the quadratic function for fitting the interface  $\mathcal{P}_i(\xi)$  as discussed above, we can solve  $d_i$  as the only unknown from (22).

For brevity, we denote  $A_g = \tanh(\beta \mathcal{P}_i(\xi_{ig}))$ ,  $D = \tanh(\beta d_i)$  and  $Q = 2(\overline{\phi}_i - \frac{1}{2})$ , and rewrite (22) as

$$\sum_{g=1}^{G} \omega_{ig} \frac{A_g + D}{1 + A_g D} = Q,\tag{23}$$

which yields a rational equation regarding the unknown D. Since D only resides in range of [-1, 1], we solve (23) by Newton-Raphson method which takes a few iterations to converge in practice.

The above numerical procedure leaves an arbitrariness in choosing the number of quadrature points. In principle, increasing the points for quadrature improves numerical accuracy for reconstruction as will be shown in section 3.1 and 3.2, which, however, increases the computational cost as well. All numerical tests presented in this paper were carried out with the Gaussian quadrature configurations elaborated in Appendix B.

#### 2.4. Updating of the volume fraction

Once the hyperbolic tangent reconstruction function in each cell is determined, volume fraction  $\overline{\phi}_i(t)$  can be updated by a finite volume formulation as,

$$\frac{\partial \overline{\phi}_{i}(t)}{\partial t} = -\frac{1}{|\Omega_{i}|} \sum_{j=1}^{J} \left( v_{n_{ij}} \int_{\Gamma_{ij}} H_{i}(\mathbf{x}, t)_{iup} d\Gamma \right) + \frac{\overline{\phi}_{i}(t)}{|\Omega_{i}|} \sum_{j=1}^{J} \left( v_{n_{ij}} \left| \Gamma_{ij} \right| \right). \tag{24}$$

The index *iup* is determined by

$$iup = \begin{cases} = i, \text{ for } v_{n_{ij}} > 0; \\ = ij, \text{ otherwise,} \end{cases}$$
 (25)

where ij denotes the index of the neighboring cell that shares surface  $\Gamma_{ij}$  with cell  $\Omega_i$ . The surface-average value is computed in the local coordinate system by Gaussian quadrature formula as

$$\int_{\Gamma_{ij}} H_i(\mathbf{x}, t)_{iup} d\Gamma = \left| \Gamma_{ij} \right| \sum_{q=1}^m \omega_q \tilde{H}_i(\xi_q, \eta_q, \gamma_q)_{iup}, \tag{26}$$

where  $(\xi_q, \eta_q, \gamma_q)$  is Gaussian points on surface  $\Gamma_{ij}$  and  $\omega_q$  is the corresponding weighting coefficient. We employ linear element, and use 4-point Gaussian quadrature(m=4) for 2D edge line, 6-point/9-point Gaussian quadrature for 3D triangular/quadrilateral surface plane respectively. The third-order TVD Runge-Kutta (RK3) scheme is used for time integration [43] as a standard time integration scheme. We have also experimented the THINC/QQ reconstruction with other time integration schemes, such as the 2th-order Runge-Kutta (RK2) method and the 3-level 2th-order Back-Difference Formula (BDF2) schemes. The results reveal that THINC/QQ reconstruction can provide accurate and reliable solutions when combined with different time integration schemes.

### 2.5. Summary of the numerical procedure

In order to facilitate the readers to follow the algorithmic flow, we give a brief summary for the entire solution procedure as follows.

- 1. Given VOF field  $\bar{\phi}_i$  in each cell, compute the gradients  $\nabla \phi_{ik}$  at cell vertices via (12) following least-square method and then map them to the local coordinate;
- 2. Compute unit normal vectors at cell vertices by Eq. (16). Then evaluate the unit normal vectors ( $\varphi_{\xi ic}, \varphi_{\eta ic}, \varphi_{\zeta ic}$ ) and curvature tensors  $\varphi_{\vartheta \delta ic}$  at mass center by performing interpolation and differentiation from piecewise polynomial with basis function  $\mathcal{N}_{ik}(\boldsymbol{\xi})$ .
- 3. Calculate the coefficients  $a_{str}$  by Eq. (19) and approximate the interface with quadratic polynomial  $\mathcal{P}_i(\xi, \eta, \zeta)$  in the form of (9).
- 4. Define indicator function  $\tilde{H}_i$  and determine the unknown  $d_i$  from the volume fraction value by Eq. (23).
- 5. Update the volume fraction by a finite volume formulation, namely Eq. (24) with RK3 time integration.

The above solution procedure provides quadratic reconstruction with substantially reduced algorithmic complexity, and can be implemented on unstructured grids of arbitrary shapes. As verified later, it significantly improves the geometric fidelity of curved interfaces.

### 3. Numerical results

The solution quality of present method is evaluated by benchmark tests in this section. In order to quantify accuracy of numerical results, we define  $L_1$  and  $L_2$  errors as follows,

$$E(L_1) = \frac{\sum_{i=1}^{N_e} (|\phi_{ni} - \phi_{ei}||\Omega_i|)}{\sum_{i=1}^{N_e} (|\phi_{ei}||\Omega_i|)} \text{ and } E(L_2) = \sqrt{\frac{\sum_{i=1}^{N_e} ((\phi_{ni} - \phi_{ei})^2 |\Omega_i|)}{\sum_{i=1}^{N_e} (\phi_{ei}^2 |\Omega_i|)}},$$
(27)

where  $\phi_{ni}$  and  $\phi_{ei}$  stand for numerical and exact solutions respectively.

#### 3.1. Interface reconstructions for a circle

In order to evaluate the accuracy of the reconstruction of the multi-dimensional hyperbolic tangent function with the quadratic surface representation, we reconstruct a circular interface from the volume fraction values on different grids as shown in Fig. 2. A circle with the radius of R = 0.4 is centered on square computational domain of  $[0, 1] \times [0, 1]$ , and the indication function is given by

$$H(\mathbf{x}) = \begin{cases} 1, & \text{if } \sqrt{(x - 0.5)^2 + (y - 0.5)^2} \le R. \\ 0, & \text{otherwise.} \end{cases}$$
 (28)

We divide the computational domain into quadrilateral and triangular elements respectively with ten vertices on each boundary edge. The volume fraction values  $\overline{\phi}_i$  are obtained by (6) and computed from  $H(\mathbf{x})$  with  $100 \times 100$  sampling points in each element, which gives adequate accuracy for initial approximation.

Knowing the volume fraction in each cell, we can piece-wisely reconstruct the THINC function  $\tilde{H}_i(\xi)$  from (10) and retrieve interface from 0.5 contour of  $\tilde{H}_i(\xi)$  where the interface equation  $\mathcal{P}_i(\xi,\eta) + d_i = 0$  is hold. We choose  $\beta = 6$  for example, and plot the reconstructed interfaces on the quadrilateral and triangular grids in Fig.2. It is seen that the curved interface segments in each cell are reproduced with accurate location from the given volume fraction field

In order to quantitatively examine the accuracy of the reconstructed interface, we sampled points  $(x_p, y_p)$ ,  $p = 1, 2, \dots, P$ , on the 0.5-contour line of the reconstructed THINC function  $\tilde{H}_i$  and calculated the deviation from the true interface by

$$E_{\text{intface}} = \frac{\sum_{p=1}^{P} \left| \sqrt{(x_p - 0.5)^2 + (y_p - 0.5)^2} - R \right|}{P}.$$
 (29)

In order to evaluate the dependency of reconstruction accuracy on the number of Gaussian quadrature points, we give the interface errors computed from (29) with different numbers of quadrature points for both quadratical and triangular elements in Table 1. We also include the results from the UMTHINC scheme [15, 50] for comparison. It is observed that with less quadrature points (3 points for triangular and 4 points for quadrilateral elements) the error of the present reconstruction is relatively higher than the UMTHINC scheme. Increasing the quadrature points significantly improves the numerical accuracy. The numerical errors in the cases of using 6 points for triangular element and 9 points are much smaller than those of UMTHINC scheme. The numerical errors are further reduced when we increase the number of quadrature points to 16 and 12 for quadrilateral and triangular elements respectively, which however is somewhat more computationally expensive.

Table 1: Numerical error for interface reconstruction by UMTHINC[52] and THINC/QQ schemes on quadrilateral and triangular element grids. NOQP is the abbreviation of "number of quadrature points".

| schemes  | quadril | ateral element         | triangular element |                        |  |
|----------|---------|------------------------|--------------------|------------------------|--|
| schemes  | NOQP    | Eintface               | NOQP               | Eintface               |  |
| UMTHINC  | _       | $2.460 \times 10^{-3}$ | _                  | $2.004 \times 10^{-3}$ |  |
|          | 4       | $2.905 \times 10^{-3}$ | 3                  | $2.471 \times 10^{-3}$ |  |
| THINC/QQ | 9       | $1.521 \times 10^{-3}$ | 6                  | $1.456 \times 10^{-3}$ |  |
|          | 16      | $1.392 \times 10^{-3}$ | 12                 | $1.351 \times 10^{-3}$ |  |

We also evaluate the accuracy of reconstruction function  $\tilde{H}_i(\xi)$  by computing  $L_1$  and  $L_2$  error of volume fraction value for interface cells identified within  $\epsilon \leqslant \frac{1}{|\Omega_i|} \int_{\Omega_i} H(\mathbf{x}) d\Omega \leqslant 1 - \epsilon$ . The  $\phi_{ni}$  and  $\phi_{ei}$  of (27) are volume fractions defined by

$$\phi_{ni} = \frac{\int_{\Omega_i} \tilde{H}_i(\xi) d\Omega}{|\Omega_i|}, \ \phi_{ei} = \frac{\int_{\Omega_i} H(\mathbf{x}) d\Omega}{|\Omega_i|},$$
(30)

and numerically integrated with sufficient sampling points (about  $10^6$ ) in each grid element. From Table 2, it is seen that with quadrature points increased, present method effectively reduces numerical errors in both  $L_1$  and  $L_2$  norms and shows evident advantage over UMTHINC scheme in numerical accuracy. Particularly, the numerical error on

![](_page_11_Figure_1.jpeg)

![](_page_11_Figure_2.jpeg)

Figure 2: Interface reconstructions for a circular shape on quadrilateral (Left) and triangular (Right) element grids. The red line stands for the true interface, and the black line segments are the cell-wisely retrieved interface, which is obtained by connecting the points sampled on the 0.5-contour of the reconstructed THINC function  $\tilde{H}_i(\xi)$ .

triangular grid is reduced down to nearly 1/4 compared to UMTHINC scheme where a linear surface (plane) approximation is used to fit the interface. The significant improvement in numerical accuracy might be partly attributed to the quadratic fitting of the interface in reconstruction of the THINC/QQ scheme.

Table 2: Numerical errors for reconstructed volume fraction value by UMTHINC[52] and THINC/QQ schemes on quadrilateral and triangular element grids. NOQP is the abbreviation of "number of quadrature points".

| schemes  |      | quadrilateral el       | ement                  | triangular element |                        |                        |  |
|----------|------|------------------------|------------------------|--------------------|------------------------|------------------------|--|
| schemes  | NOQP | $L_1$ error            | $L_2$ error            | NOQP               | $L_1$ error            | $L_2$ error            |  |
| UMTHINC  | _    | $8.626 \times 10^{-3}$ | $1.012 \times 10^{-2}$ | -                  | $2.371 \times 10^{-2}$ | $3.047 \times 10^{-2}$ |  |
|          | 4    | $4.336 \times 10^{-2}$ | $3.799 \times 10^{-2}$ | 3                  | $3.360 \times 10^{-2}$ | $3.392 \times 10^{-2}$ |  |
| THINC/QQ | 9    | $7.181 \times 10^{-3}$ | $6.583 \times 10^{-3}$ | 6                  | $7.336 \times 10^{-3}$ | $7.862 \times 10^{-3}$ |  |
|          | 16   | $9.108 \times 10^{-4}$ | $9.666 \times 10^{-4}$ | 12                 | $1.382 \times 10^{-3}$ | $1.522 \times 10^{-3}$ |  |

### 3.2. Advection of a circle in a uniform velocity field

We considered the translation advection of a round circle on unstructured grids in [59]. A circle of diameter of 0.3 was initially centered at (0.3, 0.3) in a square domain of  $[0, 1.2]^2$ . Following the grid configuration used in [59], we divided the domain by an unstructured grid with 14418 triangular elements generated with Delaunay algorithm. The circle was transported by a uniform velocity of (2, 1) with its exact position at (0.9, 0.6) after 0.3 time unit.

We firstly compute with  $\beta = 6.0$  and CFL(Courant-Friedrichs-Lewy) number of 0.6 and depict the contour plots in Fig. 3. It is seen that the interface shape is perfectly preserved which is nearly identical to the exact solution. It reveals that the present scheme can effectively remain the interface sharpness even at relatively large CFL number which is superior to the results in [59] (see Fig.33 in [59] for comparison).

To further illustrate the performance of present scheme, we repeat the test with different CFL numbers and Gaussian points distribution and compare with some other mainstream algebraic VOF schemes available in [59]. Configured with three, six and twelve Gaussian points, the resultant THINC/QQ scheme is denoted by 3P, 6P and 12P

![](_page_12_Figure_1.jpeg)

Figure 3: Contour plots of the advected round droplet at time t = 0.3. Plotted are the exact solution(Left) and the numerical solution by THINC/QQ scheme under CFL number of 0.6 (Right).

Table 3: Numerical errors of THINC/QQ( $\beta = 6.0$ ) scheme at different CFL numbers and Gaussian points in comparison with other algebraic VOF schemes available in [59].

| schemes       | CFL=0.2                | CFL=0.3                | CFL=0.4                | CFL=0.5                | CFL=0.6                | CFL=0.7                | CFL=0.8                |
|---------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| CICSAM[59]    | $3.406 \times 10^{-3}$ | $3.858 \times 10^{-3}$ | $5.295 \times 10^{-3}$ | $1.056 \times 10^{-2}$ | $1.956 \times 10^{-2}$ | $2.553 \times 10^{-2}$ | $2.918 \times 10^{-2}$ |
| HRIC[59]      | $5.122 \times 10^{-3}$ | $6.208 \times 10^{-3}$ | $1.519 \times 10^{-2}$ | $2.490 \times 10^{-2}$ | $2.997 \times 10^{-2}$ | $3.142 \times 10^{-2}$ | $3.170 \times 10^{-2}$ |
| THOR[59]      | $3.967 \times 10^{-3}$ | $6.376 \times 10^{-3}$ | $1.189 \times 10^{-2}$ | $1.747 \times 10^{-2}$ | $2.575 \times 10^{-2}$ | $3.270 \times 10^{-2}$ | $3.708 \times 10^{-2}$ |
| STACS[59]     | $6.038 \times 10^{-3}$ | $6.184 \times 10^{-3}$ | $6.366 \times 10^{-3}$ | $6.554 \times 10^{-3}$ | $6.727 \times 10^{-3}$ | $6.980 \times 10^{-3}$ | $7.138 \times 10^{-3}$ |
| M-CICSAM[59]  | $3.353 \times 10^{-3}$ | $3.471 \times 10^{-3}$ | $4.236 \times 10^{-3}$ | $4.651 \times 10^{-3}$ | $4.662 \times 10^{-3}$ | $5.556 \times 10^{-3}$ | $5.725 \times 10^{-3}$ |
| THINC/QQ(3P)  | $4.636 \times 10^{-4}$ | $4.674 \times 10^{-4}$ | $4.832 \times 10^{-4}$ | $5.215 \times 10^{-4}$ | $5.845 \times 10^{-4}$ | $7.049 \times 10^{-4}$ | $8.549 \times 10^{-4}$ |
| THINC/QQ(6P)  | $3.819 \times 10^{-4}$ | $3.807 \times 10^{-4}$ | $3.790 \times 10^{-4}$ | $3.812 \times 10^{-4}$ | $3.939 \times 10^{-4}$ | $4.369 \times 10^{-4}$ | $5.132 \times 10^{-4}$ |
| THINC/QQ(12P) | $3.689 \times 10^{-4}$ | $3.678 \times 10^{-4}$ | $3.661 \times 10^{-4}$ | $3.666 \times 10^{-4}$ | $3.757 \times 10^{-4}$ | $4.154 \times 10^{-4}$ | $4.948 \times 10^{-4}$ |

respectively. The numerical error is defined the same as in [59] for comparison,

$$Err = \frac{1}{N_e} \sum_{i=1}^{N_e} (|\phi_{ni} - \phi_{ei}||\Omega_i|).$$
 (31)

We summarize the results in Table 4 together with those available in [59]. It shows that the present scheme significantly improves the numerical accuracy for all CFL numbers tested. The numerical errors of the present scheme are overall less than 1/10 in comparison to the referenced solutions of other existing schemes in [59]. It is noted that the numerical results of present scheme are less dependent on the CFL number. Same as what observed in section 3.1, using more quadrature points effectively reduces the numerical error, and the configuration of 6 Gaussian points for triangular element can give sufficient accuracy.

To investigate the influence of the sharpness parameter ( $\beta$ ) on the numerical solutions, we have also carried out a series of computations with different  $\beta$  values ( $\beta = 2.0, 5.0$  and 8.0) as well as CFL numbers (CFL=0.2 and 0.8) separately. We plot the enlarged color-map views of VOF field in the range of [0.001, 0.999] for different  $\beta$  values in Fig.4. It is observed that the thickness of jump transition in the VOF field can be effectively adjusted by  $\beta$  value. The interface region can be limited within 2  $\sim$  3 cells when  $\beta$  is 4 or larger. It should be noted that the thickness of jump transition remains constant for a given  $\beta$  without smeared-out even for large number steps of computation. We also

![](_page_13_Figure_1.jpeg)

Figure 4: Enlarged view of contour plots within the range of  $0.001 \le \phi \le 0.999$  for different  $\beta$  values at time t = 0.3 under time step of CFL=0.2.

| Table 4: Under/overshoots and errors | (31) | ) at time $t = 0.3$ with differ | rent $\beta$ values and CFL | numbers for advection test of a round circle. |
|--------------------------------------|------|---------------------------------|-----------------------------|-----------------------------------------------|
|                                      |      |                                 |                             |                                               |

|     |                         | CFL=0.2                 |                        | CFL=0.8                             |                         |                        |  |
|-----|-------------------------|-------------------------|------------------------|-------------------------------------|-------------------------|------------------------|--|
| β   | $\min \phi$             | $\max \phi - 1$         | Err                    | $\min \phi$                         | $\max \phi - 1$         | Err                    |  |
| 2.0 | 0                       | $-3.056 \times 10^{-7}$ | $2.179 \times 10^{-3}$ | 0                                   | $-2.781 \times 10^{-7}$ | $2.182 \times 10^{-3}$ |  |
| 3.0 | 0                       | $-1.701 \times 10^{-7}$ | $1.179 \times 10^{-3}$ | $-8.549 \times 10^{-5}$             | $1.799 \times 10^{-4}$  | $1.176 \times 10^{-3}$ |  |
| 4.0 | 0                       | $-1.141 \times 10^{-7}$ | $7.309 \times 10^{-4}$ | $-7.049 \times 10^{-3}$             | $3.818 \times 10^{-3}$  | $7.337 \times 10^{-4}$ |  |
| 5.0 | 0                       | $-8.663 \times 10^{-8}$ | $5.026 \times 10^{-4}$ | $-1.372 \times 10^{-2}$             | $6.503 \times 10^{-3}$  | $5.494 \times 10^{-4}$ |  |
| 6.0 | 0                       | $-6.931 \times 10^{-8}$ | $3.819 \times 10^{-4}$ | $\rightarrow -2.572 \times 10^{-2}$ | $8.267 \times 10^{-3}$  | $5.132 \times 10^{-4}$ |  |
| 7.0 | 0                       | $-5.602 \times 10^{-8}$ | $3.240 \times 10^{-4}$ | $-3.385 \times 10^{-2}$             | $1.265 \times 10^{-2}$  | $5.646 \times 10^{-4}$ |  |
| 8.0 | $-2.369 \times 10^{-5}$ | $-8.201 \times 10^{-7}$ | $3.165 \times 10^{-4}$ | $-3.937 \times 10^{-2}$             | $1.759 \times 10^{-2}$  | $6.598 \times 10^{-4}$ |  |

summarize the undershoots  $(\min \phi)$ , overshoots  $(\max \phi - 1)$  and numerical errors (31) in Table 4 for different  $\beta$  values and CFL numbers. It is found that the THINC/QQ scheme can get well-regulated solution for VOF fields for wide range values of  $\beta$  and CFL number. We also observe that the over/undershoots become more visible when larger  $\beta$  and CFL values are used, but are still within an acceptable range even in the case of  $\beta = 8$  and CFL=0.8. Our numerical experiments for real-case applications suggest that the  $\beta$  values from 2 to 6 can be chosen for a targeted thickness of the interface in practice.

To evaluate the computational cost for different cases, we measured the elapse times on PC with a single CPU of Intel(R) i7 870, 2.93GHZ and show the results in Table 5. Compared with UMTHINC scheme with the linear representation of the interface, the present scheme with the quadratic representation causes a slight increase in CPU time. The CPU time also marginally increases when more quadrature points are used, which shows a potential capability of more accurate reconstructions in real-case applications. We summarize the quadrature configurations for different types of elements in Appendix B as the practical trade-off with adequate solution quality, which will be used in the numerical tests hereafter.

### 3.3. Rotation of an asteroid-like body

As a convincing example to illustrate the superiority of quadratic reconstruction over linear reconstruction for interface, we solved the solid-rotation of an asteroid-like body. Following [14], the asteroid-like body is initially plotted by  $r(\theta) = 0.3 \times (1 + \cos(10\theta)/2)$  with  $\tan \theta = (y - 0.5)/(x - 0.5)$  on a square domain of  $[0, 1]^2$ , and rotated by a counter-clockwise velocity (0.5 - y, x - 0.5) field around the center of domain. We partitioned the domain with triangular grids with number of cells doubly increased along each side of computational domain as  $N_s = 50$ , 100 and

Table 5: Computational cost for UMTHINC and THINC/QQ schemes with different numbers of quadrature points and CFL conditions.

| schemes       | CFL=0.2 | CFL=0.4 | CFL=0.6 | CFL=0.8 |
|---------------|---------|---------|---------|---------|
| UMTHINC       | 6.15s   | 3.07s   | 2.29s   | 1.74s   |
| THINC/QQ(3P)  | 6.34s   | 3.29s   | 2.37s   | 1.82s   |
| THINC/QQ(6P)  | 6.91s   | 3.55s   | 2.48s   | 1.95s   |
| THINC/QQ(12P) | 7.90s   | 3.75s   | 2.65s   | 2.12s   |

![](_page_14_Figure_3.jpeg)

Figure 5: Interface computed by UMTHINC scheme with linear representation for interface on grids of different resolutions after five revolutions. Dashed red line is the numerical solution and solid black line the exact solution.

200, which correspond to the grids with total element numbers of 5612, 22464 and 90198. The time step is adjusted so that CFL number is 0.2 for all grids, and  $\beta = 6.0$  is used in this test.

We firstly plot the computed interface by UMTHINC scheme, where a linear fitting is employed for the moving interface, after five revolutions at  $t=10\pi$  in Fig. 5. Obvious distortions are observed even with refined meshes. The corresponding results of THINC/QQ scheme are shown in Fig. 6. It is found that the symmetry of the numerical solution has been substantially improved by THINC/QQ scheme with a quadratic representation of the interface that captures curved surfaces with much better geometrical fidelity. The superiority of THINC/QQ scheme in solution accuracy is also substantiated by examining numerical errors in term of  $L_1$  error and convergence rate as given in Table 6. Moreover, we include the results of algebraic MULES[8] scheme in Table 6 for comparison which is an recent variant of CICSAM scheme. It is verified that the THINC/QQ scheme with quadratic reconstruction substantially improves the geometric faithfulness as well as solution accuracy.

### 3.4. Zalesak solid-body rotation on hybrid unstructured grids

The well-known Zalesak's solid rotation problem [58] is tested on different grid resolutions. Initially, a slotted disk is centered at (0.5, 0.75) in the computational domain  $[0, 1]^2$  by specifying the indication function as,

$$H(\mathbf{x}) = \begin{cases} 1, & \text{if } \left\{ \sqrt{(x - 0.5)^2 + (y - 0.75)^2} \le 0.15 \right\} \setminus \{|x - 0.5| \le 0.025 \text{ and } y \le 0.85\} \\ 0, & \text{otherwise.} \end{cases}$$
(32)

Table 6: Numerical errors and convergence rates after five-round rotations of an asteroid-like body.

| ١. | racie of rear | Table of Transcriber errors and convergence rates area in a round rotations of an asteroid into cody. |      |                        |      |                        |      |  |
|----|---------------|-------------------------------------------------------------------------------------------------------|------|------------------------|------|------------------------|------|--|
|    | elements      | UMTHINC                                                                                               |      | THINC/QQ               |      | MULES                  |      |  |
|    | Cicilicitis   | $L_1$ error                                                                                           | Rate | $L_1$ error            | Rate | $L_1$ error            | Rate |  |
|    | 5612          | $2.861 \times 10^{-1}$                                                                                | _    | $1.712 \times 10^{-1}$ | _    | $2.813 \times 10^{-1}$ | _    |  |
|    | 22464         | $1.452 \times 10^{-1}$                                                                                | 0.98 | $5.947 \times 10^{-2}$ | 1.52 | $1.173 \times 10^{-1}$ | 1.26 |  |
|    | 90198         | $6.735 \times 10^{-2}$                                                                                | 1.11 | $2.252 \times 10^{-2}$ | 1.40 | $4.440 \times 10^{-2}$ | 1.40 |  |

![](_page_15_Figure_1.jpeg)

Figure 6: Same as Fig. 5 but computed by THINC/QQ scheme with the quadratic representation for interface.

The velocity field is given as (u, v) = (y - 0.5, 0.5 - x) and time increment  $\delta t$  is set so as to let the maximum CFL number be 0.25. The initial volume fraction values  $\phi_i$  are obtained from sampling  $H(\mathbf{x})$  at  $100 \times 100$  points in each cell.

We compute this test problem on both Cartesian quadrilateral grid and unstructured triangular grid with  $\beta$  specified to be 6. After one revolution of rotation, we plot the enlarged views of the interface identified by the 0.5-contour line in Fig. 7 for both grids. It is observed that the present results recover the exact interface with sufficient accuracy comparable to other PLIC VOF schemes in [2, 35, 37, 4] on Cartesian grid. There is no significant difference in solutions quality between the results on both grids which indicates excellent adaptivity of present scheme to hybrid unstructured grids. In order to see the thickness of the jump transition in the volume fraction field, we also plot the contours of 0.05, 0.5 and 0.95 in Fig. 8. The background grids are included to show that the jump transitions maintain a compact thickness within 2 to 3 cells across the interface. In addition, we have experimented with different values of  $\beta$  up to 10 and observed a sharp interface within thickness of 3 cell if  $\beta$  is larger than 4. This finding is in consistent with that of previous example (section 3.2) which also agrees well with existing studies [49, 14, 15].

![](_page_15_Figure_5.jpeg)

![](_page_15_Figure_6.jpeg)

Figure 7: Enlarged view of interface(0.5 contour) after one revolution for Zalesak slotted-cylinder test computed by THINC/QQ scheme on 40000 quadrilateral grid (Left) and 57518 triangular grid (Right). The red dashed line is retrieved from numerical solution and black solid line is the exact solution.

We have investigated  $L_1$  errors and convergence rates in respect to different grid resolutions. The numerical results

![](_page_16_Figure_1.jpeg)

![](_page_16_Figure_2.jpeg)

Figure 8: Enlarged view of 0.05, 0.5, 0.95 contour lines after one revolution for Zalesak slotted-cylinder test on 40000 quadrilateral grid (Left) and 57518 triangular grid (Right) computed by THINC/QQ scheme.

Table 7: Numerical errors and convergence orders of Zalesak solid rotation on grid A and grid B in comparison with the UMTHINC scheme.

| Resolution | UMTHINC ( $\beta = 6$ ) | Order      | THINC/QQ ( $\beta = 6$ ) | Order |
|------------|-------------------------|------------|--------------------------|-------|
|            | Quadri                  | lateral gr | ids                      |       |
| 2500       | $8.12 \times 10^{-2}$   |            | $8.96 \times 10^{-2}$    | _     |
| 10000      | $2.61 \times 10^{-2}$   | 1.63       | $3.22 \times 10^{-2}$    | 1.47  |
| 40000      | $1.33 \times 10^{-2}$   | 0.97       | $1.67 \times 10^{-2}$    | 0.95  |
|            | Triang                  | gular gric | ls                       |       |
| 3588       | $1.19 \times 10^{-1}$   | _          | $7.71 \times 10^{-2}$    | _     |
| 14412      | $4.11 \times 10^{-2}$   | 1.78       | $2.42 \times 10^{-2}$    | 1.66  |
| 57518      | $1.76 \times 10^{-2}$   | 1.22       | $1.12 \times 10^{-2}$    | 1.13  |

are given in Table 7 and compared with those of UMTHINC scheme. It is noted that UMTHINC scheme implements quadratic reconstruction on quadrilateral grid while linear reconstruction on triangular grid. The numerical accuracy of THINC/QQ on quadrilateral grids is comparable to UMTHINC scheme and substantially improved on triangular elements, which justifies the superiority of the quadratic reconstruction over the linear one.

Given errors of several existing VOF schemes for Rudman-Zalesak test[38, 6], we computed the solid rotation of slotted disk by the present method on Cartesian grid (grid A) and two unstructured grids (grid B and grid C) as shown in Fig. 9. The grid B and grid C are generated by Delaunay and subsplit algorithms respectively. On domain boundary, 200 points are uniformly distributed along each edge, which is equivalent to Rudman's configuration. To facilitate comparison of numerical results on different type of grids, we set a square region containing the initial disk with identical Cartesian grid elements. After one round revolution, we plot interface profile (0.5 contour line) in Fig. 10 and 0.05, 0.5 and 0.95 contours in Fig. 11. We can clearly observe that the interface is favorably tracked with indistinguishable deformation compared with initial condition. In additional, the compact transition of jump region is effectively preserved within  $2 \sim 3$  cells even for longer computation. The present results look competitive to reference solution of other existing VOF methods as reported in literature [38, 26, 13, 46, 21, 11, 4, 6, 16]. We also include  $L_1$  errors in Table 8 for further comparison. The results demonstrate that the THINC/QQ scheme is more appealing than PLIC type VOF scheme particular for unstructured grids since the same level of high accuracy can be readily obtained on all grids tested and complicated manipulation in geometric reconstruction is entirely circumvented.

![](_page_17_Figure_1.jpeg)

Figure 9: The computational grid for Rudman-Zalesak slotted disk rotation test: structured quadrilateral grid (a), hybrid unstructured grid mixed with quadrilateral and triangular elements generated by Delaunay (b) and subsplit (c) algorithm respectively. For clarity, the grids displayed here are coaser than those actually used in the numerical tests.

![](_page_17_Figure_3.jpeg)

Figure 10: Enlarged view of interface (0.5 contour) after one revolution for Rudman-Zalesak slotted disk rotation test on different grids.

![](_page_17_Figure_5.jpeg)

Figure 11: The same as Fig. 10 but for enlarged view of 0.05, 0.5, 0.95 contour lines.

Table 8: Numerical errors of Rudman-Zalesak slotted disk rotation test.

| Algorithms             | Errors                |
|------------------------|-----------------------|
| THINC/QQ(grid A)       | $1.42 \times 10^{-2}$ |
| THINC/QQ(grid B)       | $9.79 \times 10^{-3}$ |
| THINC/QQ(grid C)       | $9.65 \times 10^{-3}$ |
| FCT-VOF[38]            | $3.29 \times 10^{-2}$ |
| SLIC[26]               | $8.38 \times 10^{-2}$ |
| SOLA-VOF[13]           | $9.62 \times 10^{-2}$ |
| Stream/Youngs[10]      | $1.07 \times 10^{-2}$ |
| Stream/Puckett[10]     | $1.00 \times 10^{-2}$ |
| EMFPA/Youngs[21]       | $1.06 \times 10^{-2}$ |
| EMFPA/Puckett[21]      | $9.73 \times 10^{-3}$ |
| EMFPA-SIR[21]          | $8.74 \times 10^{-3}$ |
| DDR/Youngs[11]         | $1.56 \times 10^{-2}$ |
| DDR/Puckett[11]        | $1.50 \times 10^{-2}$ |
| Youngs[38]             | $1.09 \times 10^{-2}$ |
| ELVIR[4]               | $1.00 \times 10^{-2}$ |
| GPCA[6]                | $9.79 \times 10^{-3}$ |
| CICSAM[46]             | $2.02 \times 10^{-2}$ |
| PLIC [16]              | $1.09 \times 10^{-2}$ |
| PLIC(Unstructured)[16] | $1.50 \times 10^{-2}$ |
| THINC[48]              | $3.52 \times 10^{-2}$ |
| THINC/WLIC[53]         | $1.96 \times 10^{-2}$ |
| THINC/SW[49]           | $1.34 \times 10^{-2}$ |

#### 3.5. Time-dependent single vortex flow on hybrid unstructured grids

The capability of present scheme to capture heavily deformed and stretched interfaces is verified by computing the shearing flow benchmark test introduced in [37]. As displayed in Fig. 12, two computational grids, i.e. Cartesian grid and unstructured triangular grid denoted by grid A and grid B respectively, are used in this test. An initial circle with radius 0.15 is centered at (0.5,0.75) on the computational domain of an unit square  $[0,1]^2$ . The circle is transported by a time-dependent velocity field defined by the stream function as,

$$\psi(x, y, t) = \frac{1}{\pi} \sin^2(\pi x) \sin^2(\pi y) \cos\left(\frac{\pi t}{T}\right), \quad x, y \in [0, 1],$$
(33)

where T = 8 is specified in this test.

The initial volume fraction profile is deformed and stretched into a spiral with a thin tail by the velocity field up to the half period T/2, and then returned back to its initial state until t = T by a reversed velocity. The maximum CFL number is 0.15 in this test. The test is carried out for one period on grids of different resolutions that have 32, 64 and 128 vertices evenly configured on each edge of computational domain.

The interface at time of T/2 computed by THINC/QQ scheme is plotted in Fig. 13. It is seen that, the interface is largely stretched and broken into small droplets since the distorted tail becomes too thin to be resolvable by finite resolution of the grids. It is also a typical observation for PLIC type VOF schemes. As shown also in Fig. 13, the interface after one period (t = T) is transported back to the initial position with good solution quality comparable to other existing PLIC-VOF schemes [39, 38, 10].

In order to quantitatively compare the numerical error and convergence rate of THINC/QQ scheme with other VOF schemes, we use the following measurement of numerical error (34)

$$Error = \sum_{i=1}^{N_e} (|\phi_{ni} - \phi_{ei}||\Omega_i|). \tag{34}$$

![](_page_19_Figure_1.jpeg)

![](_page_19_Figure_2.jpeg)

Figure 12: The computational grids for time-dependent single vortex flow test. Left (grid A): a structured quadrilateral grid. Right (grid B) unstructured triangular grid.

![](_page_19_Figure_4.jpeg)

![](_page_19_Figure_5.jpeg)

Figure 13: Numerical results of time-dependent single vortex flow test problem for *T* = 8 on a Cartesian grid (left) and an unstructured triangular grid (right) with 128 vertices in each edge. The red lines indicate the numerical solutions at *T*/2 and *T*. The black dot line represents the initial profile (exact solution).

As shown in Table 9, THINC/QQ scheme achieves nearly the same level accuracy on the unstructured grid compared with the referenced PLIC-VOF schemes which are performed on the Cartesian grid (grid A). The numerical solution converges at a rate around 2nd order. However, the numerical errors of THINC/QQ scheme are larger than those of PLIC scheme on all grids resolution. If we look at Fig. 13, we can observe that the interface is deformed into a spiral with the tail stretched into very thin film at time of t = T/2. Since the reconstruction of THINC/QQ scheme requires at least of  $2 \sim 3$  cells, where the diffused interface might result in larger errors evaluated by (34). For such a case, the PLIC scheme is more appealing because it is capable to identify sharper interface within fewer grid cells.

Table 9: Numerical errors (34) and convergence rates of THINC/QQ scheme in comparison with other existing PLIC VOF schemes reported in [37, 10, 42, 21] for the single-vortex deformational flow test[37] at T = 8.

| . <u> </u>                |                       |       |                       |       |                       |
|---------------------------|-----------------------|-------|-----------------------|-------|-----------------------|
| Grid resolution           | $32 \times 32$        | Order | $64 \times 64$        | Order | $128 \times 128$      |
| Rider-Kothe/Puckett [37]  | $4.78 \times 10^{-2}$ | 2.78  | $6.96 \times 10^{-3}$ | 2.27  | $1.44 \times 10^{-3}$ |
| Stream/Puckett [10]       | $3.72 \times 10^{-2}$ | 2.45  | $6.79 \times 10^{-3}$ | 2.53  | $1.18 \times 10^{-3}$ |
| Stream/Youngs [10]        | $3.61 \times 10^{-2}$ | 1.85  | $1.00 \times 10^{-2}$ | 2.22  | $2.16 \times 10^{-3}$ |
| EMFPA/Puckett [21]        | $3.77 \times 10^{-2}$ | 1.85  | $6.58 \times 10^{-3}$ | 2.22  | $1.07 \times 10^{-3}$ |
| Shahbazi/Paraschivoiu[42] | $3.55 \times 10^{-2}$ | 2.78  | $7.17 \times 10^{-3}$ | 2.27  | $1.44 \times 10^{-3}$ |
| THINC/QQ(grid A)          | $6.70 \times 10^{-2}$ | 1.98  | $1.52 \times 10^{-2}$ | 2.33  | $3.06 \times 10^{-3}$ |
| THINC/QQ(grid B)          | $6.54 \times 10^{-2}$ | 2.12  | $1.36 \times 10^{-2}$ | 2.65  | $2.58 \times 10^{-3}$ |

For further comparison, we also computed another similar benchmark test with single vortex shearing velocity introduced in [38]. An initial circle with a radius of  $0.2\pi$  was centered at  $(0.5\pi, 0.2(\pi + 1))$  on a  $[0, \pi]^2$  computational domain. The interface is stretched into a spiral structure by a shearing velocity field  $(u, v) = (\sin(x)\cos(y), -\cos(x)\sin(y))$ up to T/2 and transported back by a reversed velocity field. The test is carried out with the maximum CFL number up to 0.25 for one period with  $T = 10\pi$ . We generate grids with 100 vertices on each edge following the same manner as for Fig. 9 and denote them as grid A, grid B and grid C respectively. As shown in Fig. 14, we plot profiles of interface at instant of t = T/2 and T which are comparable to existing VOF type schemes [38, 11]. We can observe some differences in interface configuration among three type grids which are attributed to small droplets beyond the resolvable scale of the finite grid resolutions. To verify the capability of THINC/QQ to preserve compactness of the interface jump, we plot the enlarged view of contours  $\phi = 0.05$ , 0.5 and 0.95 for the restored cylinder in Fig. 15. It is observed that the thickness of the jump transition remains sufficiently compact throughout the computation even for the heavily distorted interface. We compare the numerical error measured by (34) with the results of PLIC type VOF schemes as well as the previous variants of THINC type schemes in Table 10. It is found that the present THINC/QQ scheme possesses the highest accuracy among all other VOF schemes computed on Cartesian grid (grid A). For hybrid unstructured grids (grid B and grid C), THINC/QQ scheme shows sufficiently high accuracy comparable to its results on grid A and outperforms the UMTHINC scheme on the same computational condition where the improvements are considered to be mainly coming from the superiority of quadratic reconstruction.

#### 3.6. 3D solid rotation of slotted-sphere on hybrid unstructured grids

The three dimensional scheme was firstly tested by solving the solid rotation transport of a slotted-sphere [14] in domain of  $[0, 1]^3$ . The initial profile of the volume fraction is specified as,

$$H(\mathbf{x}) = \begin{cases} 1, & \text{if } (x - 0.5)^2 + (y - 0.75)^2 + (z - 0.5)^2 \le 0.15^2 \land (|x - 0.5| \ge 0.025 \lor y > 0.725); \\ 0, & \text{otherwise.} \end{cases}$$
(35)

We generate three of computational grids as shown in Fig. 16. Grid A is a structured Cartesian grid, grid B and grid C are hybrid unstructured grids of hexahedral element and prismatic element generated by Delaunay and subsplit algorithms respectively. The solid slotted-sphere is rotated by velocity of  $\mathbf{u} = (y - 0.5, 0.5 - x, 0)$  for one revolution with the maximum CFL number of 0.25. The numerical accuracy is quantified by  $L_1$  error and compared with the MTHINC scheme [14] on Cartesian grid as well as UMTHINC scheme [52] on all three type grids.

From Table 11, we find that the present scheme gets similar accuracy to MTHINC and UMTHINC schemes on grid A where quadratic reconstruction function is used for all computation. The numerical errors of THINC/QQ on grids

![](_page_21_Figure_1.jpeg)

Figure 14: Numerical solution of Rudman-shearing vortex flow test problem for *T* = 10π on different grids. The red lines indicate the numerical solutions at *T*/2 and *T*. The black dot line represents the initial profile (exact solution).

![](_page_21_Figure_3.jpeg)

Figure 15: Enlarged view of 0.05, 0.5, 0.95 contour lines ofthe numerical solution *t* = *T* = 10π to Rudman-shearing vortex flow test problem on different grids.

| Table 10: | Numerical | errors of Rudma | n-shearing | vortex flow test |
|-----------|-----------|-----------------|------------|------------------|

| Algorithms       | Errors                |
|------------------|-----------------------|
| THINC/QQ(grid A) | $3.12 \times 10^{-2}$ |
| THINC/QQ(grid B) | $2.74 \times 10^{-2}$ |
| THINC/QQ(grid C) | $4.25 \times 10^{-2}$ |
| Youngs[38]       | $3.85 \times 10^{-2}$ |
| FCT-VOF[38]      | $1.44 \times 10^{-1}$ |
| Hirt-Nichols[38] | $1.09 \times 10^{-1}$ |
| SLIC[38]         | $9.02 \times 10^{-2}$ |
| DDR/Youngs[11]   | $5.15 \times 10^{-2}$ |
| DDR/Puckett[11]  | $4.51 \times 10^{-2}$ |
| THINC[48]        | $6.64 \times 10^{-2}$ |
| THINC/WLIC[53]   | $4.03 \times 10^{-2}$ |
| VOF/WLIC[53]     | $6.31 \times 10^{-2}$ |
| UMTHINC(grid A)  | $3.71 \times 10^{-2}$ |
| UMTHINC(grid B)  | $3.40 \times 10^{-2}$ |
| UMTHINC(grid C)  | $5.02 \times 10^{-2}$ |

![](_page_22_Picture_3.jpeg)

![](_page_22_Picture_4.jpeg)

![](_page_22_Picture_5.jpeg)

Figure 16: The computational meshes for Zalesak 3D rotation test. Grid A (left): a structured Cartesian grid; Grid B (middle): a hybrid unstructured grid of hexahedral and prismatic elements generated by Delaunay algorithm.; Grid C (right): a hybrid unstructured grid generated by subsplit algorithm.

B and grid C are slightly smaller than those of UMTHINC in [52], which reveals that the quadratic representation of the interface can consistently improve the accuracy on gradually refined resolution. Fig. 17 depicts the 0.5-isosurfaces of rotated solid bodies after one revolution. We can see the geometrically faithful solutions which are competitive to other VOF type schemes.

### 3.7. Time-dependent 3D vortical deformation flow on hybrid unstructured grids

To verify the capability of present scheme to resolve heavily distorted thin interface in 3D, we performed the three dimensional time-dependent vortical deformational transport of the interface in domain of [0, 1]<sup>3</sup> [20]. A sphere with radius of 0.15 is initially centered at (0.35,0.35,0.35) and transported by a time-dependent deformational reversing velocity field,

$$\begin{cases} u(x, y, z) = 2\sin^2(\pi x)\sin(2\pi y)\sin(2\pi z)\cos(\pi t/T), \\ v(x, y, z) = -\sin(2\pi x)\sin^2(\pi y)\sin(2\pi z)\cos(\pi t/T), \\ w(x, y, z) = -\sin(2\pi x)\sin(2\pi y)\sin^2(\pi z)\cos(\pi t/T). \end{cases}$$

The interface is stretched until T/2 and returned back to its initial location after one period T=3. On grid B, the sphere was initially located in a region which is filled with Cartesian grid for comparison. The test is carried out

Table 11: Numerical errors and convergence orders for the Zalesak 3D solid rotation test on grids A-C.

| Revolution           | 1/50                  | Order | 1/100                 | Order | 1/200                 |
|----------------------|-----------------------|-------|-----------------------|-------|-----------------------|
| MTHINC(grid A) [14]  | $1.19 \times 10^{-3}$ | 1.09  | $5.60 \times 10^{-4}$ | 0.91  | $3.56 \times 10^{-4}$ |
| UMTHINC(grid A) [52] | $1.11 \times 10^{-3}$ | 1.38  | $4.27 \times 10^{-4}$ | 0.93  | $2.24 \times 10^{-4}$ |
| UMTHINC(grid B) [52] | $1.05 \times 10^{-3}$ | 1.19  | $4.58 \times 10^{-4}$ | 0.95  | $2.37 \times 10^{-4}$ |
| UMTHINC(grid C) [52] | $1.07 \times 10^{-3}$ | 1.24  | $4.52 \times 10^{-4}$ | 1.03  | $2.22 \times 10^{-4}$ |
| THINC/QQ(grid A)     | $1.19 \times 10^{-3}$ | 1.35  | $4.67 \times 10^{-4}$ | 0.89  | $2.51 \times 10^{-4}$ |
| THINC/QQ(grid B)     | $8.46 \times 10^{-4}$ | 1.19  | $3.72 \times 10^{-4}$ | 0.90  | $1.99 \times 10^{-4}$ |
| THINC/QQ(grid C)     | $8.78 \times 10^{-4}$ | 1.20  | $3.81 \times 10^{-4}$ | 0.89  | $2.05 \times 10^{-4}$ |

![](_page_23_Picture_3.jpeg)

![](_page_23_Picture_4.jpeg)

![](_page_23_Picture_5.jpeg)

![](_page_23_Picture_6.jpeg)

Figure 17: Numerical results for Zalesak 3D rotation test problem on the finest grids at  $t = 2\pi$ . Shown from left to right are the exact solution (a), and numerical solutions on grid A (b), grid B (c) and grid C (d).

with maximum CFL number of 0.25. We used two grids as shown in Fig. 18. One is a Cartesian grid denoted by grid A and the other, denoted by grid B, is a hybrid unstructured grid mixed with hexahedral, tetrahedral and pyramidal elements. We generated grids of three levels of resolutions by evenly locating 32, 64, 128 vertices on each boundary edge of domain, which are then denoted by 32<sup>3</sup>, 64<sup>3</sup> and 128<sup>3</sup> grids respectively.

![](_page_23_Picture_9.jpeg)

![](_page_23_Picture_10.jpeg)

Figure 18: The computational grids for 3D shear deformational transport test. Left (grid A): a Cartesian grid. Right (grid B): a hybrid unstructured grid.

We plot iso-surface of 0.5 at instant of t = 0, T/2 and T on Cartesian and unstructured grids in Fig. 19 and Fig. 20. As discussed before, the thickness of the jump transition spans over two to three grid cells, which makes the 0.5 iso-surface for part of the thin film invisible due to the inadequate grid resolution. Nevertheless, the VOF field is exactly conserved. The numerical solutions for the largely deformed interface and the final restored sphere look also very competitive to other existing schemes for this 3D benchmark test involving complex flow and large interface distortion.

The numerical errors and corresponding convergence rates are given in Table 12 and compared with other VOF

![](_page_24_Picture_1.jpeg)

![](_page_24_Picture_2.jpeg)

![](_page_24_Picture_3.jpeg)

Figure 19: Numerical results for time-dependent 3D shear deformational transport test on 1283 Cartesian grid. Displayed are the iso-surface of contour 0.5 of the instantaneous solutions at time *t* = 0, *<sup>T</sup>* <sup>2</sup> , *T* with *T* = 3.

![](_page_24_Picture_5.jpeg)

![](_page_24_Picture_6.jpeg)

![](_page_24_Picture_7.jpeg)

Figure 20: Same as Fig. 19, but on a hybrid unstructured grid of 1243312 cells.

schemes. We observe that accuracy of present scheme on structured grid is almost the same as UMTHINC scheme and competitive to the PLIC type schemes, such as the RK-3D (the 3D version of Rider and Kothe PLIC scheme [37]), FMFPA-3D (face-matched flux polyhedron advection [12]). It is found that the numerical errors of present scheme on unstructured grid are reduced with nearly half compared to the results in Cartesian grid, which might be due to the relatively finer grid resolution. It is noted that in this test the quadratic interface representation is used for all types of grid elements. The numerical results substantiate the capability and good solution quality of the present scheme to capture 3D complex moving interfaces on hybrid arbitrary unstructured grids.

#### 4. Conclusion remarks

We have proposed a novel interface capturing method, THINC/QQ method, on hybrid arbitrary unstructured grids including triangular, quadrilateral elements in 2D and tetrahedral, hexahedral, prismatic and pyramidal elements in 3D. Based on the fully multi-dimensional reconstruction using the hyperbolic tangent function, the present method is able to get superior solution quality with obvious advantage in algorithmic simplicity.

The THINC/QQ method is different from the previous MTHINC scheme[14] and other existing variants in the following aspects:

- A fully multi-dimensional Gaussian quadrature is used in THINC/QQ method to compute the integration of multi-dimensional hyperbolic function rather than the hybrid integration in MTHINC scheme[14]. It considerably eases the numerical algorithm and facilitates the quadratic representation of the interface on arbitrary grid elements.
- The interface is represented through a complete quadratic function that includes geometrical information of the interface, such as normal direction and curvatures. As verified in the numerical tests, making use of these geometrical information in the reconstruction process substantially improves the geometrical faithfulness for curved interfaces.

Table 12: Numerical errors and convergence orders for the time-dependent 3D vortical deformation flow test.

| Revolution                | $32^{3}$              | Order | 64 <sup>3</sup>       | Order | $128^{3}$             |
|---------------------------|-----------------------|-------|-----------------------|-------|-----------------------|
| RK-3D [12]                | $7.85 \times 10^{-3}$ | 1.51  | $2.75 \times 10^{-3}$ | 1.89  | $7.41 \times 10^{-4}$ |
| FMFPA-3D [12]             | $7.44 \times 10^{-3}$ | 1.42  | $2.79 \times 10^{-3}$ | 1.97  | $7.14 \times 10^{-4}$ |
| Youngs [17]               | $7.47 \times 10^{-3}$ | 1.43  | $2.77 \times 10^{-3}$ | 1.77  | $8.14 \times 10^{-4}$ |
| LVIRA [17]                | $6.92 \times 10^{-3}$ | 1.51  | $2.43 \times 10^{-3}$ | 1.93  | $6.37 \times 10^{-4}$ |
| Youngs(Unstructured) [17] | $1.02 \times 10^{-2}$ | 1.20  | $4.45 \times 10^{-3}$ | 2.24  | $9.43 \times 10^{-4}$ |
| LVIRA(Unstructured) [17]  | $1.02 \times 10^{-2}$ | 1.53  | $3.54 \times 10^{-3}$ | 2.30  | $7.20 \times 10^{-4}$ |
| UMTHINC(gridA) [52]       | $8.06 \times 10^{-3}$ | 1.41  | $3.04 \times 10^{-3}$ | 1.69  | $9.40 \times 10^{-4}$ |
| UMTHINC(gridB) [52]       | $6.95 \times 10^{-3}$ | 1.89  | $1.87 \times 10^{-3}$ | 1.83  | $5.25 \times 10^{-4}$ |
| THINC/QQ(grid A)          | $7.96 \times 10^{-3}$ | 1.46  | $2.89 \times 10^{-3}$ | 1.67  | $9.05 \times 10^{-4}$ |
| THINC/QQ(grid B)          | $6.22 \times 10^{-3}$ | 1.98  | $1.57 \times 10^{-3}$ | 1.96  | $4.04 \times 10^{-4}$ |

Numerical accuracy can be enhanced by adding quadrature points in a straightforward way at a modest increase\nin computational cost.

The THINC/QQ method is carefully verified by various benchmark tests in this paper. The numerical results demonstrate the capability of the scheme in capturing moving interfaces of large distortions or even topological changes on unstructured grids of hybrid arbitrary elements. The quantitative comparisons with other existing methods indicate that THINC/QQ can provide high solution quality that is much superior than other algebraic VOF schemes and comparable to PLIC type VOF schemes. The later are limited nearly to Cartesian grid.

The THINC/QQ method does not require the geometric manipulations as the PLIC reconstructions, thus is easy and straightforward to be implemented on unstructured grids of various types of elements. Meanwhile, the geometrical information, such as normal direction and curvature of the interface are effectively used in the reconstruction procedure, which enables the THINC/QQ method to generate better solution quality in comparison with other existing algebraic VOF schemes. Unlike other algebraic VOF methods, THINC/QQ method does not involve any post-processing step, such as anti-diffusive treatment or reinitialization, to regularize the computed VOF field. Consequently, the THINC/QQ method act as a desirable compromise between algebraic-type and geometric-type interface capturing schemes that well balances the numerical accuracy and algorithmic complexity. The jump thickness is "semi-diffusive" and effectively preserved within  $2 \sim 3$  cells even after long time computation, which in many cases, contributes to improving numerical stability. Nevertheless, for some problems that require to identify sharper interface within less grid resolution, the VOF methods with geometrical reconstructions are more appealing where the interface is uniquely determined under the volume conservation constraint condition.

Given the adequate accuracy, algorithmic simplicity as well as the easiness to implement to arbitrary unstructured grids, the present THINC/QQ method is highly appealing in the real-case applications of multiphase simulations involving complex geometrical configurations.

#### Acknowledgment

This work was supported in part by JSPS KAKENHI Grant Numbers 15H03916 and 15J09915.

#### Appendix A. The unit normal vector and tensor in local THINC/QQ reconstruction

For simplicity, we transform grid element into local coordinate and piecewisely construct THINC/QQ formulation on these standard reference elements including triangular, quadrilateral elements for two dimensions and tetrahedral, hexahedral, prismatic, pyramidal elements for three dimensions. As shown in Table A.1, the reference elements are denoted with coordinate region and vertices  $\theta_{ik}$  located at  $(\xi_{ik}, \eta_{ik}, \zeta_{ik})$  (k = 1, 2, ..., K) where K stands for the number of the cell vertices. For 2D, component in third dimension  $\zeta$  is discarded by default.

In the standard reference element, local THINC/QQ scheme can be piecewicely constructed with given unit normal vector and curvature tensor of interface configuration. As described in Section 2.3.1, these mass-central quantities are approximated from linear polynomial function reconstructed by unit normal vector on cell vertices which are transformed from global coordinate. To facilitate reader to follow the scheme readily, we summarized the formulations for each element as follows.

- 1. We first compute first-order derivatives of VOF function on vertices  $\nabla \phi_{ik} = (\phi_{xik}, \phi_{yik}, \phi_{zik})$  by least-square method.
- 2. We transform  $\nabla \phi_{ik}$  to local coordinate  $\hat{\nabla} \phi_{ik} = (\phi_{\xi ik}, \phi_{\eta ik}, \phi_{\xi ik})$  and compute unit normal vector on vertices as  $\hat{\nabla} \varphi_{ik} = (\phi_{\xi ik}, \phi_{\eta ik}, \phi_{\xi ik})/|\hat{\nabla} \phi_{ik}|$ .
- 3. We compute unit normal vector and curvature tensor by linear approximation as shown in Table A.2 and A.3 for all type grid element.

### Appendix B. Gaussian quadrature for arbitrary unstructured grids

For THINC/QQ scheme, gaussian quadrature is required to make integration of hyperbolic tangent function where analytical expression is not available. We denote gaussian points and weights as  $p_g$  and  $\omega_g$ ,  $(g=1,2,\ldots,G)$  where G is total number of gaussian points and  $\omega_g$  satisfies the relation of  $\sum_{g=1}^G \omega_g = 1$ . Integration of hyperbolic tangent function can be expressed by sum of weighted value of indicator function on selected gaussian points as

$$\int_{\Omega_i(\xi)} \tilde{H}_i(\xi,\eta,\zeta) d\xi d\eta d\zeta = \sum_{g=1}^G \omega_g \tilde{H}_i(\xi_g,\eta_g,\zeta_g).$$

As shown in this paper, the performance of THINC/QQ scheme depends to some extent on the choice of quadrature points. In practice, we make use of standard 6, 9, 11 and 12 Gaussian quadrature points for triangular, quadrilateral, tetrahedral and pyramidal elements respectively. To reconcile accuracy and efficiency, particular attention is paid for hexahedral and prismatic element so that refined Gaussian points are arranged in the principle direction of interface for the purpose of capturing abruptly-varying VOF distribution with higher accuracy. Besides, we can use alternative set of symmetric quadrature rules [47] to ease the programming effort where the free package "polyquad" can be used to determine the quadrature configuration for any finite element grid.

#### · Hexahedral element

The integration for hexahedral element is performed based on the strategy that 4-point Gaussian quadrature is preferred for the principle direction of interface while 2-point Gaussian quadrature is used for the rest directions. The principle is determined by the maximum magnitude value of unit normal vector which yields three different formula regarding to  $(\xi, \eta, \zeta)$  direction respectively.

Case 1: 
$$|\varphi_{\xi i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$$

|               | Table A.1: Local coordina                                                                                               | **                                                                                                                                           |                                                                                                                                                                                                                                                                                                                                         |
|---------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Element       | Cell shape                                                                                                              | Element region                                                                                                                               | $\theta_{ik} (\xi_{ik}, \eta_{ik}, \zeta_{ik})$                                                                                                                                                                                                                                                                                         |
| Triangular    | $\theta_{i3}$ $\theta_{i3}$ $\theta_{i2}$                                                                               | $0 \leqslant \xi \leqslant 1$                                                                                                                | $ \theta_{i1}: (0, 0, 0, ) $ $ \theta_{i2}: (1, 0, ) $ $ \theta_{i3}: (0, 1, ) $                                                                                                                                                                                                                                                        |
| Quadrilateral | $\theta_{i1}$ $\eta$ $\theta_{i3}$ $\xi$ $\theta_{i2}$                                                                  | $0 \leqslant \xi \leqslant 1$ $0 \leqslant \eta \leqslant 1$                                                                                 | $ \theta_{i1}: ( -1, -1,  )  \theta_{i2}: ( 1, -1,  )  \theta_{i3}: ( 1,  1,  )  \theta_{i4}: ( -1,  1,  ) $                                                                                                                                                                                                                            |
| Tetrahedral   | $\theta_{IA}$ $\theta_{I3}$ $\eta$ $\theta_{I3}$                                                                        | $\delta = 1 - \xi - \eta$ $0 \le \xi \le 1$ $0 \le \eta \le 1$ $0 \le \zeta \le \delta$                                                      | $\theta_{i1}: (0, 0, 0)$ $\theta_{i2}: (1, 0, 0)$ $\theta_{i3}: (0, 1, 0)$ $\theta_{i4}: (1, 0, 1)$                                                                                                                                                                                                                                     |
| Hexahedral    | $\theta_{i5}$ $\zeta$ $\theta_{i8}$ $\theta_{i8}$ $\theta_{i9}$ $\theta_{i1}$ $\theta_{i4}$ $\theta_{i2}$ $\theta_{i3}$ | $ \begin{array}{c} -1 \leqslant \xi \leqslant 1 \\ -1 \leqslant \eta \leqslant 1 \\ -1 \leqslant \zeta \leqslant 1 \end{array} $             | $\begin{array}{c} \theta_{i1}: ( & -1, & -1, & -1 & ) \\ \theta_{i2}: ( & 1, & -1, & -1 & ) \\ \theta_{i3}: ( & 1, & 1, & -1 & ) \\ \theta_{i4}: ( & -1, & 1, & -1 & ) \\ \theta_{i5}: ( & -1, & -1, & 1 & ) \\ \theta_{i6}: ( & 1, & -1, & 1 & ) \\ \theta_{i7}: ( & 1, & 1, & 1 & ) \\ \theta_{i8}: ( & -1, & 1, & 1 & ) \end{array}$ |
| Prismatic     | $\theta_{i3}$ $\zeta$ $\theta_{i6}$ $\theta_{i6}$ $\theta_{i6}$ $\theta_{i6}$                                           |                                                                                                                                              | $\theta_{i1}: ( 1, 0, -1 ) \\ \theta_{i2}: ( 0, 0, -1 ) \\ \theta_{i3}: ( 0, 1, -1 ) \\ \theta_{i4}: ( 1, 0, 1 ) \\ \theta_{i5}: ( 0, 0, 1 ) \\ \theta_{i6}: ( 0, 1, 1 ) $                                                                                                                                                              |
| Pyramidal     | $\theta_{i3}$ $\zeta$ $\theta_{i4}$ $\eta$ $\theta_{i2}$ $\xi$                                                          | $\delta = (1 - \zeta)/2$ $-\delta \leqslant \xi \leqslant \delta$ $-\delta \leqslant \eta \leqslant \delta$ $-1 \leqslant \zeta \leqslant 1$ | $\theta_{i1}: ( -1, -1, -1 ) \\ \theta_{i2}: ( 1, -1, -1 ) \\ \theta_{i3}: ( 1, 1, -1 ) \\ \theta_{i4}: ( -1, 1, -1 ) \\ \theta_{i5}: ( 0, 0, 1 )$                                                                                                                                                                                      |

| Table A.2: Unit normal vector at cell center for each type element. |               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |
|---------------------------------------------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| Dimensionality                                                      | Element       | Gradient components                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |  |
| Two dimension                                                       | Trianglar     | $\varphi_{\xi ic} = \frac{1}{K} \sum_{k=1}^{K} \varphi_{\xi ik},$                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |  |
|                                                                     | Quadrilateral | $\varphi_{\eta ic} = \frac{1}{K} \sum_{k=1}^{K} \varphi_{\eta ik}.$                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |  |
| Three dimension                                                     | Tetrahedral   | $\varphi_{\xi ic} = \frac{1}{K} \sum_{k=1}^{K} \varphi_{\xi ik},$                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |  |
|                                                                     | Hexahedral    | $\varphi_{\eta ic} = \frac{1}{K} \sum_{k=1}^{K} \varphi_{\eta ik},$                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |  |
|                                                                     | Prismatic     | $ \varphi_{\zeta ic} = \frac{1}{K} \sum_{k=1}^{K} \varphi_{\zeta ik}. $                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |  |
|                                                                     |               | $\varphi_{\xi i c} = \frac{1}{16} \left( 3 \left( \varphi_{\xi i 1} + \varphi_{\xi i 2} + \varphi_{\xi i 3} + \varphi_{\xi i 4} \right) + 4 \varphi_{\xi i 5} \right),$ $\varphi_{\eta i c} = \frac{1}{16} \left( 3 \left( \varphi_{\eta i 1} + \varphi_{\eta i 2} + \varphi_{\eta i 3} + \varphi_{\eta i 4} \right) + 4 \varphi_{\eta i 5} \right),$ $\varphi_{\zeta i c} = \frac{1}{16} \left( 3 \left( \varphi_{\zeta i 1} + \varphi_{\zeta i 2} + \varphi_{\zeta i 3} + \varphi_{\zeta i 4} \right) + 4 \varphi_{\zeta i 5} \right).$ |  |
|                                                                     | Pyramidal     | $\varphi_{\eta ic} = \frac{1}{16} \left( 3 \left( \varphi_{\eta i1} + \varphi_{\eta i2} + \varphi_{\eta i3} + \varphi_{\eta i4} \right) + 4 \varphi_{\eta i5} \right),$                                                                                                                                                                                                                                                                                                                                                                   |  |
|                                                                     |               | $\varphi_{\zeta ic} = \frac{1}{16} \left( 3 \left( \varphi_{\zeta i1} + \varphi_{\zeta i2} + \varphi_{\zeta i3} + \varphi_{\zeta i4} \right) + 4 \varphi_{\zeta i5} \right).$                                                                                                                                                                                                                                                                                                                                                             |  |

$$\begin{aligned} p_1 &= (a_1,b_1,b_1), \ p_2 &= (a_1,b_2,b_1), \ p_3 &= (a_1,b_2,b_2), \ p_4 &= (a_1,b_1,b_2), \\ p_5 &= (a_2,b_1,b_1), \ p_6 &= (a_2,b_2,b_1), \ p_7 &= (a_2,b_2,b_2), \ p_8 &= (a_2,b_1,b_2), \\ p_9 &= (a_3,b_1,b_1), \ p_{10} &= (a_3,b_2,b_1), \ p_{11} &= (a_3,b_2,b_2), \ p_{12} &= (a_3,b_1,b_2), \\ p_{13} &= (a_4,b_1,b_1), \ p_{14} &= (a_4,b_2,b_1), \ p_{15} &= (a_4,b_2,b_2), \ p_{16} &= (a_4,b_1,b_2), \\ \omega_1 &= \omega_2 &= \omega_3 &= \omega_4 &= \omega_{13} &= \omega_{14} &= \omega_{15} &= \omega_{16} &= \left(3 - \sqrt{5/6}\right) / 48, \\ \omega_5 &= \omega_6 &= \omega_7 &= \omega_8 &= \omega_9 &= \omega_{10} &= \omega_{11} &= \omega_{12} &= \left(3 + \sqrt{5/6}\right) / 48, \end{aligned}$$

Case 2: 
$$|\varphi_{\eta i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$$

$$\begin{split} p_1 &= (b_1, a_1, b_1), \ p_2 = (b_2, a_1, b_1), \ p_3 = (b_2, a_1, b_2), \ p_4 = (b_1, a_1, b_2), \\ p_5 &= (b_1, a_2, b_1), \ p_6 = (b_2, a_2, b_1), \ p_7 = (b_2, a_2, b_2), \ p_8 = (b_1, a_2, b_2), \\ p_9 &= (b_1, a_3, b_1), \ p_{10} = (b_2, a_3, b_1), \ p_{11} = (b_2, a_3, b_2), \ p_{12} = (b_1, a_3, b_2), \\ p_{13} &= (b_1, a_4, b_1), \ p_{14} = (b_2, a_4, b_1), \ p_{15} = (b_2, a_4, b_2), \ p_{16} = (b_1, a_4, b_2). \\ \omega_1 &= \omega_2 = \omega_3 = \omega_4 = \omega_{13} = \omega_{14} = \omega_{15} = \omega_{16} = \left(3 - \sqrt{5/6}\right) / 48, \\ \omega_5 &= \omega_6 = \omega_7 = \omega_8 = \omega_9 = \omega_{10} = \omega_{11} = \omega_{12} = \left(3 + \sqrt{5/6}\right) / 48, \end{split}$$

Case 3: 
$$|\varphi_{\zeta i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$$

$$\begin{split} p_1 &= (b_1,b_1,a_1)\,,\; p_2 = (b_2,b_1,a_1)\,,\; p_3 = (b_2,b_2,a_1)\,,\; p_4 = (b_1,b_2,a_1)\,,\\ p_5 &= (b_1,b_1,a_2)\,,\; p_6 = (b_2,b_1,a_2)\,,\; p_7 = (b_2,b_2,a_2)\,,\; p_8 = (b_1,b_2,a_2)\,,\\ p_9 &= (b_1,b_1,a_3)\,,\; p_{10} = (b_2,b_1,a_3)\,,\; p_{11} = (b_2,b_2,a_3)\,,\; p_{12} = (b_1,b_2,a_3)\,,\\ p_{13} &= (b_1,b_1,a_4)\,,\; p_{14} = (b_2,b_1,a_4)\,,\; p_{15} = (b_2,b_2,a_4)\,,\; p_{16} = (b_1,b_2,a_4)\,,\\ \omega_1 &= \omega_2 = \omega_3 = \omega_4 = \omega_{13} = \omega_{14} = \omega_{15} = \omega_{16} = \left(3-\sqrt{5/6}\right)\!\!\Big/48,\\ \omega_5 &= \omega_6 = \omega_7 = \omega_8 = \omega_9 = \omega_{10} = \omega_{11} = \omega_{12} = \left(3+\sqrt{5/6}\right)\!\!\Big/48, \end{split}$$

where

$$a_1 = -\left(3 + 2\sqrt{1.2}\right)/7, \ a_2 = -\left(3 - 2\sqrt{1.2}\right)/7, \ b_1 = -1/\sqrt{3}, \ a_3 = \left(3 - 2\sqrt{1.2}\right)/7, \ a_4 = \left(3 + 2\sqrt{1.2}\right)/7, \ b_2 = 1/\sqrt{3}.$$

#### • Prismatic element

The integration for prismatic element is implemented by splitting volume integration into area integration on triangular face of  $\xi\eta$  plane and line integration along  $\zeta$  direction which is perpendicular to  $\xi\eta$  plane. The integration on two segments uses different resolutions of quadrature points which yields two distinct configurations

Table A.3: Curvature tensor at the cell center for each element type.

| Table A.3: Curvature tensor at the cell center for each element type. |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |  |  |  |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|--|--|
| Element                                                               | Curvature tensor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |  |  |  |
| Triangular                                                            | $\varphi_{\xi^{2}ic} = \varphi_{\xi i2} - \varphi_{\xi i1},$ $\varphi_{\eta^{2}ic} = \varphi_{\eta i3} - \varphi_{\eta i1},$ $\varphi_{\xi \eta ic} = \varphi_{\xi i3} - \varphi_{\xi i1},$ $\varphi_{\xi \eta ic} = \varphi_{\xi i3} - \varphi_{\xi i1},$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |  |  |  |
| Quadrilateral                                                         | $\varphi_{\eta\xi ic} = \varphi_{\eta i2} - \varphi_{\eta i1}.$ $\varphi_{\xi^{2}ic} = \frac{1}{4}(\varphi_{\xi i2} + \varphi_{\xi i3} - \varphi_{\xi i1} - \varphi_{\xi i4}),$ $\varphi_{\eta^{2}ic} = \frac{1}{4}(\varphi_{\eta i3} + \varphi_{\eta i4} - \varphi_{\eta i1} - \varphi_{\eta i2}),$ $\varphi_{\xi\eta ic} = \frac{1}{4}(\varphi_{\xi i3} + \varphi_{\xi i4} - \varphi_{\xi i1} - \varphi_{\xi i2}),$ $\varphi_{\eta\xi ic} = \frac{1}{4}(\varphi_{\eta i2} + \varphi_{\eta i3} - \varphi_{\eta i1} - \varphi_{\eta i4}).$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |  |  |  |
| Tetrahedral                                                           | $\varphi_{\xi^{2}ic} = \varphi_{\xi^{1}2} - \varphi_{\xi^{1}1},$ $\varphi_{\eta^{2}ic} = \varphi_{\eta^{1}3} - \varphi_{\eta^{1}1},$ $\varphi_{\zeta^{2}ic} = \varphi_{\zeta^{1}4} - \varphi_{\zeta^{1}1},$ $\varphi_{\xi\eta^{ic}} = \varphi_{\xi^{1}3} - \varphi_{\xi^{1}1},$ $\varphi_{\eta\xi^{ic}} = \varphi_{\eta^{1}2} - \varphi_{\eta^{1}1},$ $\varphi_{\eta\xi^{ic}} = \varphi_{\eta^{1}2} - \varphi_{\eta^{1}1},$ $\varphi_{\zeta\eta^{ic}} = \varphi_{\zeta^{1}3} - \varphi_{\zeta^{1}1},$ $\varphi_{\xi\eta^{ic}} = \varphi_{\xi^{1}3} - \varphi_{\zeta^{1}1},$ $\varphi_{\xi\xi^{ic}} = \varphi_{\xi^{1}2} - \varphi_{\xi^{1}1},$ $\varphi_{\xi\xi^{ic}} = \varphi_{\zeta^{1}2} - \varphi_{\zeta^{1}1}.$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |  |  |  |
| Hexahedral                                                            | $\begin{split} \varphi_{\xi^{2}ic} &= \frac{1}{8}(\varphi_{\xi i2} + \varphi_{\xi i3} + \varphi_{\xi i6} + \varphi_{\xi i7} - \varphi_{\xi i1} - \varphi_{\xi i4} - \varphi_{\xi i5} - \varphi_{\xi i8}), \\ \varphi_{\eta^{2}ic} &= \frac{1}{8}(\varphi_{\eta i3} + \varphi_{\eta i4} + \varphi_{\eta i7} + \varphi_{\eta i8} - \varphi_{\eta i1} - \varphi_{\eta i2} - \varphi_{\eta i5} - \varphi_{\eta i6}), \\ \varphi_{\zeta^{2}ic} &= \frac{1}{8}(\varphi_{\zeta i5} + \varphi_{\zeta i6} + \varphi_{\zeta i7} + \varphi_{\zeta i8} - \varphi_{\zeta i1} - \varphi_{\zeta i2} - \varphi_{\zeta i3} - \varphi_{\zeta i4}), \\ \varphi_{\xi \eta ic} &= \frac{1}{8}(\varphi_{\xi i3} + \varphi_{\xi i4} + \varphi_{\xi i7} + \varphi_{\xi i8} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i5} - \varphi_{\xi i6}), \\ \varphi_{\eta \xi ic} &= \frac{1}{8}(\varphi_{\eta i2} + \varphi_{\eta i3} + \varphi_{\eta i6} + \varphi_{\eta i7} - \varphi_{\eta i1} - \varphi_{\eta i4} - \varphi_{\eta i5} - \varphi_{\eta i8}), \\ \varphi_{\eta \zeta ic} &= \frac{1}{8}(\varphi_{\eta i5} + \varphi_{\eta i6} + \varphi_{\eta i7} + \varphi_{\eta i8} - \varphi_{\eta i1} - \varphi_{\eta i2} - \varphi_{\eta i3} - \varphi_{\eta i4}), \\ \varphi_{\zeta \eta ic} &= \frac{1}{8}(\varphi_{\xi i3} + \varphi_{\zeta i4} + \varphi_{\tau i7} + \varphi_{\zeta i8} - \varphi_{\zeta i1} - \varphi_{\zeta i2} - \varphi_{\zeta i5} - \varphi_{\zeta i6}), \\ \varphi_{\xi \zeta ic} &= \frac{1}{8}(\varphi_{\xi i5} + \varphi_{\xi i6} + \varphi_{\xi i7} + \varphi_{\xi i8} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i3} - \varphi_{\xi i4}), \\ \varphi_{\zeta \xi ic} &= \frac{1}{8}(\varphi_{\zeta i2} + \varphi_{\zeta i3} + \varphi_{\tau i6} + \varphi_{\zeta i7} - \varphi_{\zeta i1} - \varphi_{\zeta i4} - \varphi_{\zeta i5} - \varphi_{\zeta i8}). \end{split}$                                                                                                                 |  |  |  |
| Prismatic                                                             | $\varphi_{\xi^{2}ic} = \frac{1}{2} \left( \varphi_{\xi i1} + \varphi_{\xi i4} - \varphi_{\xi i2} - \varphi_{\xi i5} \right),$ $\varphi_{\eta^{2}ic} = \frac{1}{2} \left( \varphi_{\eta i3} + \varphi_{\eta i6} - \varphi_{\eta i2} - \varphi_{\eta i5} \right),$ $\varphi_{\xi^{2}ic} = \frac{1}{6} \left( \varphi_{\xi i4} + \varphi_{\xi i5} + \varphi_{\xi i6} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i3} \right),$ $\varphi_{\xi \eta ic} = \frac{1}{2} \left( \varphi_{\xi i3} + \varphi_{\xi i6} - \varphi_{\xi i2} - \varphi_{\xi i5} \right),$ $\varphi_{\eta \xi ic} = \frac{1}{2} \left( \varphi_{\eta i1} + \varphi_{\eta i4} - \varphi_{\eta i2} - \varphi_{\eta i5} \right),$ $\varphi_{\eta \xi ic} = \frac{1}{6} \left( \varphi_{\eta i4} + \varphi_{\eta i5} + \varphi_{\eta i6} - \varphi_{\eta i1} - \varphi_{\eta i2} - \varphi_{\eta i3} \right),$ $\varphi_{\xi \eta ic} = \frac{1}{2} \left( \varphi_{\xi i3} + \varphi_{\xi i6} - \varphi_{\xi i2} - \varphi_{\xi i5} \right),$ $\varphi_{\xi \xi ic} = \frac{1}{6} \left( \varphi_{\xi i4} + \varphi_{\xi i5} + \varphi_{\xi i6} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i3} \right),$ $\varphi_{\xi \xi ic} = \frac{1}{2} \left( \varphi_{\xi i1} + \varphi_{\xi i4} - \varphi_{\xi i2} - \varphi_{\xi i5} \right).$ $\varphi_{\xi^{2}ic} = \frac{1}{4} \left( \varphi_{\xi i2} + \varphi_{\xi i3} - \varphi_{\xi i1} - \varphi_{\xi i4} \right),$ $\varphi_{\eta^{2}ic} = \frac{1}{4} \left( \varphi_{\eta i3} + \varphi_{\eta i4} - \varphi_{\eta i1} - \varphi_{\eta i2} \right),$ $\varphi_{\xi^{2}ic} = \frac{1}{8} \left( 4\varphi_{\xi i5} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i3} - \varphi_{\xi i4} \right),$ $\varphi_{\xi \eta ic} = \frac{1}{4} \left( \varphi_{\xi i3} + \varphi_{\xi i4} - \varphi_{\xi i1} - \varphi_{\xi i2} \right),$ $\varphi_{\eta \xi ic} = \frac{1}{4} \left( \varphi_{\eta i2} + \varphi_{\eta i3} - \varphi_{\eta i1} - \varphi_{\eta i4} \right),$ |  |  |  |
|                                                                       | $ \varphi_{\eta\zeta ic} = \frac{1}{8} \left( 4\varphi_{\eta i5} - \varphi_{\eta i1} - \varphi_{\eta i2} - \varphi_{\eta i3} - \varphi_{\eta i4} \right),  \varphi_{\zeta\eta ic} = \frac{1}{4} \left( \varphi_{\zeta i3} + \varphi_{\zeta i4} - \varphi_{\zeta i1} - \varphi_{\zeta i2} \right),  \varphi_{\xi\zeta ic} = \frac{1}{8} \left( 4\varphi_{\xi i5} - \varphi_{\xi i1} - \varphi_{\xi i2} - \varphi_{\xi i3} - \varphi_{\xi i4} \right),  \varphi_{\zeta\xi ic} = \frac{1}{4} \left( \varphi_{\zeta i2} + \varphi_{\zeta i3} - \varphi_{\zeta i1} - \varphi_{\zeta i4} \right). $                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |  |  |  |

according to principle direction of interface. For the first case that the principle direction lies on the plane of triangular face segment i.e.  $\xi$  or  $\eta$  direction, 6-point Gaussian quadrature is preferred to integrate over the  $\xi\eta$  plane and 2-point Gaussian quadrature for the  $\zeta$  directions. Otherwise for the case that the principle direction stands perpendicular to  $\xi\eta$  plane, we prefer 3-point Gaussian quadrature for  $\xi\eta$  plane and 4-point Gaussian quadrature for the  $\zeta$  directions. The formula are summarized for these two cases as follows.

Case 1: 
$$|\varphi_{\eta i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$$
 or  $|\varphi_{\zeta i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$   
 $p_1 = (a_1, a_1, b_1), \ p_2 = (a_3, a_1, b_1), \ p_3 = (a_1, a_3, b_1), \ p_4 = (a_2, a_2, b_1),$   
 $p_5 = (a_4, a_2, b_1), \ p_6 = (a_2, a_4, b_1), \ p_7 = (a_1, a_1, b_2), \ p_8 = (a_3, a_1, b_2),$   
 $p_9 = (a_1, a_3, b_2), \ p_{10} = (a_2, a_2, b_2), \ p_{11} = (a_4, a_2, b_2), \ p_{12} = (a_2, a_4, b_2),$   
 $a_1 = \left(8 - \sqrt{10} + \sqrt{38 - 44\sqrt{2/5}}\right) \left|18, \ a_2 = \left(8 - \sqrt{10} - \sqrt{38 - 44\sqrt{2/5}}\right)\right|18,$   
 $a_3 = 1 - 2a_1, \ a_4 = 1 - 2a_2, \ b_1 = -1 \left|\sqrt{3}, \ b_2 = 1 \left|\sqrt{3}\right|.$   
 $\omega_1 = \omega_2 = \omega_3 = \omega_4 = \omega_5 = \omega_6 = \left(620 + \sqrt{213125 - 53320\sqrt{10}}\right) \left|7440,$   
 $\omega_7 = \omega_8 = \omega_9 = \omega_{10} = \omega_{11} = \omega_{12} = \left(620 - \sqrt{213125 - 53320\sqrt{10}}\right) \left|7440,$ 

Case 2: 
$$|\varphi_{\xi i}| = \max(|\varphi_{\xi i}|, |\varphi_{\eta i}|, |\varphi_{\zeta i}|)$$

$$\begin{aligned} p_1 &= (a_1,a_1,b_1)\,,\; p_2 = (a_1,a_2,b_1)\,,\; p_3 = (a_2,a_1,b_1)\,,\; p_4 = (a_1,a_1,b_2)\,,\\ p_5 &= (a_1,a_2,b_2)\,,\; p_6 = (a_2,a_1,b_2)\,,\; p_7 = (a_1,a_1,b_3)\,,\; p_8 = (a_1,a_2,b_3)\,,\\ p_9 &= (a_2,a_1,b_3)\,,\; p_{10} = (a_1,a_1,b_4)\,,\; p_{11} = (a_1,a_2,b_4)\,,\; p_{12} = (a_2,a_1,b_4)\,,\\ a_1 &= 1/6,\; a_2 = 0,\; a_3 = 2/3,\; b_1 = -\left(3+2\sqrt{1.2}\right)\!\!\big/\, 7,\; b_2 = -\left(3-2\sqrt{1.2}\right)\!\!\big/\, 7,\\ b_3 &= \left(3-2\sqrt{1.2}\right)\!\!\left/\, 7,\; b_4 = \left(3+2\sqrt{1.2}\right)\!\!\left/\, 7,\; b_2 = -\left(3-2\sqrt{1.2}\right)\!\!\right/\, 7,\\ \omega_1 &= \omega_2 = \omega_3 = \omega_{10} = \omega_{11} = \omega_{12} = \left(3-\sqrt{5/6}\right)\!\!\middle/\, 36,\\ \omega_4 &= \omega_5 = \omega_6 = \omega_7 = \omega_8 = \omega_9 = \left(3+\sqrt{5/6}\right)\!\!\middle/\, 36. \end{aligned}$$

- [1] N. Ashgriz, T. Barbat, G. Wang, A computational lagrangian-eulerian advection remap for free surface flows, Int. J. Numer. Methods Fluids 44 (2004) 1–32.
- [2] A. Ashgriz, J.P Poo, FLAIR: flux line-segment model for adevction and interface reconstruction, J. Comput. Phys. 93 (1991) 449–468.
- [3] E. Aulisa, S. Manservisi, R. Scardovelli, S. Zaleski, Interface reconstruction with least-square fit and split advection in three-dimensional Cartesian geometry, J. Comput. Phys. 225 (2007) 2301–2319.
- [4] E. Aulisa, S. Manservisi, R. Scardovelli, S. Zaleski, A geometrical area-preserving volume-of-fluid advection method, J. Comput. Phys. 192 (2003) 355–364.
- [5] N. Balcazar, O. Lehmkuhl, L. Jofre, J. Rigola, A. Oliva, A coupled volume-of-fluid ´ /level-set method for simulation of two-phase flows on unstructured meshes, Comput. and Fluids 124 (2016) 12–29.
- [6] A. Cervone, S. Manservisi, R. Scardovelli, S. Zaleski, A geometrical predictor-corrector advection scheme and its application to the volume fraction function, J. Comput. Phys. 228 (2009) 406–419.
- [7] F. Denner, B. G.M. van Wachem, Compressive VOF method with skewness correction to capture sharp interfaces on arbitrary meshes, J. Comput. Phys. 279 (2014) 127–144.
- [8] S.S. Deshpande, L. Anumolu, M.F. Trujillo, Evaluating the performance of the two-phase flow solver interFoam, Comput. Sci. Disc. 5 (2012) 014016.
- [9] S.V. Diwakar, Sarit K. Das, T. Sundararajan, A Quadratic Spline based Interface (QUASI) reconstruction algorithm for accurate tracking of two-phase flows, J. Comput. Phys. 228 (2009) 9107–9130.
- [10] D.J.E. Harvie, D.F. Fletcher, A new volume of fluid advection algorithm: the stream scheme, J. Comput. Phys. 162 (2000) 1–32.
- [11] D.J.E. Harvie, D. F Fletcher, A new volume of fluid advection algorithm: the defined donating region scheme, Int. J. Numer. Methods Fluids 35 (2001) 151–172.
- [12] J. Hernandez, J. Lopez, P. Gomez, C. Zanzi, F. Faura, A new volume of fluid method in three dimensions Part I: multidimensional advection method with face-matched flux polyhedra, Int. J. Numer. Methods Fluids 58 (2008) 897–921.
- [13] C.W. Hirt, D.B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, J. Comput. Phys. 39 (1981) 201–251.
- [14] S. Ii, K. Sugiyama, S. Takeuchi, S. Takagi, Y. Matsumoto, F. Xiao, An interface capturing method with a continuous function: the THINC method with multi-dimensional reconstruction, J. Comput. Phys. 231 (2012) 2328–2358.
- [15] S. Ii, B. Xie, F. Xiao, An interface capturing method with a continuous function: The THINC method on unstructured triangular and tetrahedral meshes, J. Comput. Phys. 259 (2014) 260–269.
- [16] K. Ito, T. Kunugi, H. Ohshima, T. Kawamura, A volume-conservative PLIC algorithm on three-dimensional fully unstructured meshes, Comput. and Fluids 88 (2013) 250-261.
- [17] L. Jofre, O. Lehmkuhl, J. Castro, A. Oliva, A 3-D volume-of-fluid advection method based on cell-vertex velocities for unstructured meshes, Comput. and Fluids 94 (2014) 14–29.
- [18] B. Lafaurie, C. Nardone, R. Scardovell, S. Zaleski, G. Zanetti, Modeling merging and fragmentation in multiphase flows with SURFER, J. Comput. Phys. 113 (1994) 134–147.
- [19] B. P. Leonard, The ULTIMATE conservative difference scheme applied to unsteady one dimensional advection, Comput. Methods Appl. Mech. Engrg. 88 (1991) 17-74.
- [20] R. LeVeque, High-resolution conservative algorithms for advection in incompressible flow, SIAM Journal on Numerical Analysis, 33 (1996) 627–665.
- [21] J. Lopez, J. Hernandez, P. Gomez, F. Faura, A volume of fluid method based on multidimensional advection and spline interface reconstruction, J. Comput. Phys. 195 (2004) 718–742.
- [22] J. Lopez, J. Hernandez, P. Gomez, F. Faura, An improved PLIC-VOF method for tracking thin fluid structures in incompressible two-phase flows, J. Comput. Phys. 208 (2005) 51–74.
- [23] T. Maric, H. Marschall, D. Bothe, voFoam-a geometrical volume of fluid algorithm on arbitrary unstructured meshes with local dynamic adaptive mesh refinement using OpenFOAM, arXiv preprint arXiv:1305.3417 (2013)
- [24] M. Meier, G. Yadigaroglu, B. L. Smith, A novel technique for including surface tension in PLIC-VOF methods, Eur. J. Mech. B/Fluids 21 (2002) 61–73.
- [25] S.J. Mosso, B.K. Swartz, D.B. Kothe, S.P. Clancy, Recent enhancements of volume tracking algorithms for irregular grids, Technical Report LA-CP-96-227, Los Alamos National Laboratory, 1996.
- [26] W.F. Noh, P. Woodward, SLIC (simple line interface method), Lect. Notes Phys. 24 (1976) 330–340.
- [27] E. Olsson, G. Kreiss, A conservative level set method for two phase flow, J. Comput. Phys. 210 (2005) 225–246.
- [28] E. Olsson, G. Kreiss, S. Zahedi, A conservative level set method for two phase flow II, J. Comput. Phys. 225 (2007) 785–807.
- [29] B.J. Parker, D.L. Youngs, Two and three dimensional Eulerian simulation of fluid flow with material interfaces, UK Atomic Weapons Establishment, 1992.
- [30] J.E. Pilliod, E.G. Puckett, Second-order accurate volume-of-fluid algorithms for tracking material interfaces, J. Comput. Phys. 199 (2004) 464–502.
- [31] S. Popinet, Gerris: a tree-based adaptive solver for the incompressible euler equations in complex geometries, J. Comput. Phys. 190 (2003) 572–600.
- [32] S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows. J. Comput. Phys. 228 (2009) 583-5866.
- [33] S. Popinet, http://gfs.sourceforge.net
- [34] S. Popinet, http://basilisk.fr
- [35] E.G. Puckett, A.S. Almgren, J.B. Bell, D.L. Marcus, W.J. Rider, A high-order projection method for tracking fluid interfaces in variable density incompressible flows, J. Comput. Phys. 130 (1997) 269–282.
- [36] Y. Renardy, M. Renardy, Prost: a parabolic reconstruction of surface tension of the volume-of-fluid method, J. Comput. Phys. 183 (2002) 400–421.
- [37] W.J. Rider, D.B. Kothe, Reconstructing volume tracking, J. Comput. Phys. 141 (1998) 112–152.
- [38] M. Rudman, Volume-tracking methods for interfacial flow calculations, Int. J. Numer. Methods Fluids 24 (1997) 671–691.

- [39] G. Ryskin, L.G. Leal, Numerical solution of free-boundary problems in fluid mechanics. Part 2. Buoyancy-driven motion of a gas bubble through a quiescent liquid, J. Fluid Mech. 148 (1984) 19–35.
- [40] R. Scardovelli, S. Zaleski, Analytical Relations Connecting Linear Interfaces and Volume Fractions in Rectangular Grids, J. Comput. Phys. 164 (2000) 228–237.
- [41] R. Scardovelli, S. Zaleski, Interface reconstruction with least-square fit and split EulerianLagrangian advection, Int. J. Numer. Methods Fluids 41 (2003) 251–274.
- [42] K. Shahbazi, M. Paraschivoiu, J. Mostaghimi, Second order accurate volume tracking based on remapping for triangular neshes, J. Comput. Phys. 188 (2003) 100–122.
- [43] C.W. Shu, Total-variation-diminishing time discretizations, SIAM J. Sci. Stat. Comput. 9 (1988) 1073–1084.
- [44] M. Sussman, P. Smereka, Axisymmetric free boundary problems, J. Fluid Mech. 341 (1997) 269.
- [45] G. Tryggvason, R. Scardovelli, S. Zaleski, Direct Numerical Simulations of Gas–Liquid Multiphase Flows, Cambridge University Press, 2011.
- [46] C. Ubbink, R.I. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 153 (1999) 26–50.
- [47] F.D. Witherden, P.E. Vincent, On the identification of symmetric quadrature rules for finite element methods, Comput. Math Appl., 69 (2015) 1232–1241.
- [48] F. Xiao, Y. Honma, K. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, Int. J. Numer. Methods Fluids 48 (2005) 1023–1040.
- [49] F. Xiao, S. Ii, C.G. Chen, Revisit to the THINC scheme: a simple algebraic VOF algorithm, J. Comput. Phys. 230 (2011) 7086–7092.
- [50] B. Xie, S. Ii and F. Xiao, An efficient and accurate algebraic interface capturing method for unstructured grids in 2 and 3 dimensions: The THINC method with quadratic surface representation, Int. J. Numer. Methods Fluids 76 (2014) 1025–1042.
- [51] B. Xie, F. Xiao, A multi-moment constrained finite volume method on arbitrary unstructured grid for incompressible flows, J. Comput. Phys. 327 (2016) 747–778.
- [52] B. Xie, P. Jing, F. Xiao, An unstructured-grid numerical model for interfacial multiphase fluids based on multi-moment finite volume formulation and THINC method, Int. J. Multiphase Flow, 89 (2017) 375–398.
- [53] K. Yokoi, Efficient implementation of THINC scheme: A simple and practical smoothed VOF algorithm, J. Comput. Phys. 226 (2007) 1985–2002.
- [54] K. Yokoi, A practical numerical framework for free surface flows based on CLSVOF method, multi-moment methods and density-scaled CSF model: Numerical simulations of droplet splashing, J. Comput. Phys. 232 (2013) 252–271.
- [55] K. Yokoi, A density-scaled continuum surface force model within a balanced force formulation, J. Comput. Phys. 278 (2014) 221–228.
- [56] D.L. Youngs, Time-dependent multi-material flow with large fluid distortion, Numerical methods for fluid dynamics, Americal Press, New York (1982) 273-486.
- [57] D.L. Youngs, An interface tracking method for a 3D Eulerian hydrodynamics code, Technical Report 44/92/35, AWRE, 1984.
- [58] S.T. Zalesak, Fully multi-dimensional flux corrected transport algorithm for fluid flow, J. Comput. Phys. 31 (1979) 335–362.
- [59] D. Zhang, C.B. Jiang, D.F. Liang, Z.B. Chen, Y. Yang, A refined volume-of-fluid algorithm for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 274 (2014) 709–736.