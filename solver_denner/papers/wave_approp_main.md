# Wave-appropriate reconstruction of compressible flows: physics-constrained acoustic dissipation and rank-1 entropy wave correction

# Amareshwara Sainadh Chamarthi<sup>∗</sup>

Division of Engineering and Applied Science, California Institute of Technology, Pasadena, CA, 91125, USA

# Abstract

Within the finite volume framework, the wave-appropriate reconstruction approach [\[1,](#page-38-0) [2,](#page-38-1) [3\]](#page-38-2) decomposes the reconstruction procedure into characteristic wave families, centralizing non-acoustic waves to minimize dissipation while retaining an upwind bias for acoustic waves. In all previous implementations, the acoustic upwind parameter η<sup>a</sup> was fixed at its maximum value of 1.0; however, this choice is conservative and can be further improved, motivating a systematic search for the minimum value of η<sup>a</sup> that preserves robust stability across flow regimes. To this end, the CFD solver is treated as a black box within Brent's bounded scalar minimization, which minimizes an accuracy objective for the subsonic inviscid Taylor–Green vortex subject to a stability constraint enforced by the supersonic viscous Taylor–Green vortex. Because the waveappropriate framework leaves η<sup>a</sup> as the sole degree of freedom, the optimization converges in approximately 25 evaluations, with the objective function plateauing after roughly 12 iterations. The resulting optimal values, η ∗ <sup>a</sup> = 0.54 for the third-order scheme and η ∗ <sup>a</sup> = 0.6010 for the fifth-order scheme, generalize without retuning across the full range from subsonic turbulence to hypersonic flows with shocks and contact discontinuities. Critically, the optimized nonlinear Nth-order scheme consistently matches or outperforms the standard linear (N +2)th-order scheme at full upwinding.

The second contribution focuses on eliminating the need for an explicit contact-discontinuity detector, which is commonly required in flows involving both shock waves and contact discontinuities. In such cases, the reconstruction deficiency appears solely within the entropy characteristic wave and can be corrected by a rank-1 update along the entropy right eigenvector. The proposed algorithm, WA-CR, relies only on the Ducros sensor and is limiter-agnostic, facilitating direct use in other schemes, such as WENO, while maintaining the same η ∗ a . This approach reduces wall time by 29–41% compared to full characteristic decomposition. To further demonstrate the method's generality, introducing a controlled acoustic bias exclusively to the normal momentum in a kinetic-energy-preserving scheme eliminates spurious vortices in periodic shear layers, confirming that the acoustic stability mechanism operates independently of the discretization framework.

Keywords: Physics-constrained optimization, Low dissipation, Data-driven, Wave-appropriate reconstruction, Conservative-characteristic reconstruction, ILES.

# 1. Introduction

The simulation of compressible turbulent flows demands a careful balance between two competing requirements: resolving turbulent structures with minimal artificial dissipation while maintaining numerical stability near discontinuities, such as shocks and contact discontinuities. High-order shock-capturing schemes typically address stability by applying upwind reconstruction uniformly across all flow variables, but this uniform treatment imposes unnecessary dissipation where none is required. The fundamental question is therefore not only how much dissipation to add, but where and to which flow structures it should be directed.

<sup>∗</sup>Corresponding author.

# 1.1. Background and motivation

The wave-appropriate reconstruction framework answers this question by exploiting the characteristic structure of the governing equations. The compressible flow field is decomposed into its characteristic wave families, and each family is treated by the reconstruction scheme most appropriate to its physical character, as illustrated in Figure [1.](#page-1-0) For the three-dimensional compressible Euler equations, there are five characteristic waves: two acoustic waves that carry pressure fluctuations and are responsible for shocks, one entropy/contact wave that carries density jumps across material interfaces, and two shear/vortical waves that carry transverse momentum and are responsible for turbulent structures. This decomposition is not merely a mathematical convenience; it reflects the distinct physical roles that each wave family plays in the flow. Each family also has distinct numerical characteristics that require a specific reconstruction approach. The framework was developed over a series of papers [\[1,](#page-38-0) [2,](#page-38-1) [4,](#page-38-3) [5,](#page-38-4) [3\]](#page-38-2), with each contribution identifying one more wave family whose treatment could be made more physically consistent, as summarized below and in Figure [1.](#page-1-0)

Figure 1: Wave-appropriate reconstruction algorithm.

#### Part 1 (2023): Wave-appropriate discontinuity sensor [\[1\]](#page-38-0)

The first contribution established the basic wave-appropriate framework. The key observation was that shock capturing should be performed selectively by wave type rather than uniformly across all characteristic fields. The Ducros sensor [\[6\]](#page-38-5) detects shocks and activates nonlinear MP limiting for the acoustic and shear/vortical waves; a separate density-based criterion detects contact discontinuities, carried by the entropy wave, and activates limiting for that wave independently, regardless of the Ducros sensor state. Despite this selective limiting, all five characteristic waves were still reconstructed with the same fully upwind interpolation (η<sup>a</sup> = 1), leaving the dissipation level of non-acoustic waves unnecessarily high. This contribution is represented by the grey box in Figure [1.](#page-1-0)

#### Part 2 (2024): Centralization of vortical waves [\[2\]](#page-38-1)

The second contribution [\[2\]](#page-38-1) showed that shear and vortical waves, which travel at the local advection velocity and carry only transverse momentum perturbations, can be reconstructed using a central scheme (η<sup>a</sup> = 0.5) in smooth regions. The Ducros sensor activates MP limiting for these waves near shocks, as in Part 1, but outside shock regions, the central scheme is used for the shear/vorticity waves. This reduced unnecessary dissipation on vortical structures and had a significant practical consequence: in the oblique shock impingement case of Sandham et al. [\[7\]](#page-38-6), only the centralized scheme achieved full laminar-to-turbulent transition on wall-modeled LES grids. This work is represented by the green box in Figure [1.](#page-1-0)

Part 3 (2025): Extension to multicomponent and multiphase flows [5, 3]

The third contribution extended the wave-appropriate framework to multicomponent and multiphase flows, where material interfaces introduce an additional wave family that requires specialized treatment. The entropy/contact wave carries jumps in species volume fraction across material interfaces. Rather than applying a standard polynomial reconstruction to this wave, the Tangent of Hyperbola for INterface Capturing (THINC) method was applied selectively to the entropy characteristic, sharpening the interface representation without introducing oscillations in the acoustic or vortical fields. This work also established a key physical constraint for 2-D and 3-D flows: the tangential velocity components are continuous across a contact discontinuity (in viscous flows [8] or if there is artificial viscosity [9]), which means the shear characteristic waves carry no jump at the interface, and a central scheme is therefore appropriate. This contribution is represented by the blue and green boxes in Figure 1.

#### The remaining open question

The preceding work in this series [1, 2, 4, 5, 3] established the wave-appropriate reconstruction framework: acoustic waves are reconstructed with an upwind scheme, shear and vortical waves with a central scheme, and the entropy wave with selective limiting near contact discontinuities. Notably, in all prior implementations, the acoustic upwind bias,  $\eta_a \in [0.5, 1]$ , was typically set to 1.0. Since non-acoustic waves are already centralized, reducing  $\eta_a < 1$  could be stable; yet, no theoretical analysis precisely predicts the minimum achievable value. This is due to the complex nonlinear interactions among the limiter, the Ducros sensor, the Riemann solver, and the characteristic projection, which collectively determine stability in a way that linear theory cannot capture. Consequently, the minimum  $\eta_a$  must be found empirically.

A key advantage of the wave-appropriate framework is that only a *single* parameter,  $\eta_a$ , requires empirical determination; all other dissipation sources are fixed during design. As a result, tuning focuses solely on  $\eta_a$ . The present optimization process reduces to a bounded scalar minimization, which converges in about 25 CFD evaluations: the objective function typically plateaus after approximately 12 iterations, and subsequent evaluations serve only to refine  $\eta_a^*$  to four decimal places.

Building on this, the present paper addresses the open question for both the thirdand fifth-order schemes. It identifies the physical mechanism by which optimized nonlinear Nth-order schemes can outperform standard (N+2)th-order linear schemes. In addition, it introduces a second contribution: the rank-1 entropy correction, which further reduces the computational cost of the wave-appropriate framework by removing the need for explicit contact detection. Both contributions are highlighted by the red box in Figure 1.

A related but distinct issue concerns the stability of kinetic-energy-preserving (KEP) and entropy-stable schemes, which employ a non-dissipative central discretization of the convective flux in smooth flow regions and switch to a characteristic-based upwind or WENO scheme near shocks via the Ducros sensor. This approach has been employed in several high-fidelity compressible flow solvers [10], including the formulation of Subbareddy and Candler [11] and the CHAMPS immersed boundary solver of van Noordt et al. [12]. The total numerical flux in such formulations is written as

$$\hat{f}^{conv} = (1 - \alpha)\hat{f}^{cent} + \alpha \hat{f}^{diss}, \tag{1}$$

where  $\hat{f}^{cent}$  is a non-dissipative kinetic-energy-preserving flux computed directly in physical space,  $\hat{f}^{diss}$  is the dissipative flux computed using characteristic-based upwind or WENO reconstruction, and  $\alpha$  is the Ducros sensor output. When the Ducros sensor is inactive ( $\alpha=0$ ), as is the case in smooth shear-layer flows, the flux reduces to the purely central kinetic-energy-preserving scheme with zero dissipation across all wave types. Because the central component is computed in physical space without characteristic decomposition, the characteristic decomposition is invoked only through the dissipative path  $\hat{f}^{diss}$  and is therefore available only when the Ducros sensor is active. The Ducros threshold has been reported to require case-by-case adjustment in practice; van Noordt et al. [12] found that values of  $\alpha < 0.0125$  generate noise in the solution while higher values cause the flow to deviate from the expected behavior, ultimately selecting  $\alpha = 0.0125$  empirically. Furthermore, De Vanna et al. [13] combined a kinetic-energy-preserving scheme with a WENO scheme using a modified Ducros sensor and similarly reported that the threshold value may be case-dependent.

Ghate and Lele [\[14\]](#page-39-2) identified two important limitations of such KEP schemes. First, the resulting solutions may suffer from energy pile-up near Nyquist scales in the absence of a subgrid-scale closure, since the kinetic energy transfer associated with the inviscid fluxes is non-dissipative. Second, kinetic energy conservation holds only in a semi-discrete sense at low Mach numbers, with many past applications requiring additional de-aliasing filtering for robustness and most numerical investigations conducted at CFL numbers below 0.1 [\[14\]](#page-39-2). A further limitation, not explicitly discussed in the literature to the authors' knowledge, is that the fully central acoustic treatment can be unstable in shear-layer flows, even in the absence of shocks. As demonstrated in Section [7.2,](#page-22-0) the unmodified KEP scheme produces significant oscillations in the periodic shear layer despite the flow being entirely shock-free and the Ducros sensor remaining inactive throughout. These limitations motivate applying the wave-appropriate principle to KEP schemes. Section [6](#page-16-0) demonstrates that introducing a controlled acoustic bias exclusively through the normal-momentum component of the Riemann solver's dissipative flux, while leaving the non-dissipative KEP treatment intact for all non-acoustic variables, eliminates the shear-layer instability without compromising energy conservation. This extends the wave-appropriate framework beyond reconstruction-based schemes to encompass energy-preserving formulations.

Remark 1.1 (On the Ducros sensor). A related point concerns the formulation of the Ducros sensor itself. The original sensor of Ducros et al. [\[6\]](#page-38-5) is the product of a pressure-based shock indicator and a dilatation-to-vorticity ratio. In much of the subsequent literature, only the dilatation-to-vorticity component has been retained, while the pressure-based indicator has been dropped. This simplification is consequential: the dilatation-to-vorticity ratio alone can misidentify intense vortical regions as shocked [\[15\]](#page-39-3), which may partly explain the need for threshold tuning in practice. Sciacovelli et al. [\[15\]](#page-39-3) demonstrated this deficiency directly, showing that the dilatation-to-vorticity ratio alone is insufficient for reliable shock detection and that incorporating Jameson's pressure-based indicator substantially improves sensor specificity, as illustrated in Figure [2.](#page-3-0) Notably, this combined formulation is precisely what Ducros et al. [\[6\]](#page-38-5) originally proposed; the pressure-based component appears to have been inadvertently dropped in much of the subsequent literature. The present work, consistent with the authors' earlier contributions [\[1,](#page-38-0) [2,](#page-38-1) [5\]](#page-38-4), employs the complete twocomponent sensor with a fixed threshold Ω<sup>d</sup> > 0.01, which Sciacovelli et al. [\[15\]](#page-39-3) independently identify as appropriate. The robustness of this fixed threshold, applied uniformly across all test cases from subsonic turbulence to Mach 10 flows without case-by-case adjustment, may be attributed in part to the use of the complete two-component formulation, as detailed in Section [3.2.2.](#page-9-0)

![](_page_3_Figure_2.jpeg)

Figure 2: Shock sensor components for underexpanded jet, reproduced from Sciacovelli et al. [\[15\]](#page-39-3): (a) dilatation-to-vorticity ratio alone, (b) Bhagatwala and Lele's indicator, (c) Jameson's pressure-based indicator, and (d) combination of all three components. Panel (a) clearly misidentifies intense vortical regions as shocked, while the combined sensor in panel (d) correctly localizes activation in the shock regions.

# 1.2. Prior work on scheme optimization

Prior work has explored data-driven optimization of numerical scheme parameters for compressible flows using surrogate-based Bayesian optimization [\[16,](#page-39-4) [17,](#page-39-5) [18\]](#page-39-6) and multi-objective frameworks [\[19,](#page-39-7) [20,](#page-39-8) [21\]](#page-39-9). The most relevant is Feng et al. [\[20\]](#page-39-8), which optimized four free parameters—a Ducros cutoff threshold C<sup>D</sup> and three upwind bias coefficients (ηeno, ηlnr, ηv)—using 100 CFD evaluations with Gaussian process surrogates. Examining the requirement for four parameters is instructive from the point of view of the current work. First, the Ducros sensor implementation in [\[20\]](#page-39-8) employs only the dilatation-to-vorticity ratio, omitting the pressure-based shock indicator in the original formulation of Ducros et al. [\[6\]](#page-38-5) (see Remark [3.3\)](#page-10-0). As discussed in Section [3.2.2,](#page-9-0) the complete two-component sensor admits a fixed threshold (Ω<sup>d</sup> > 0.01) requiring no tuning, indicating C<sup>D</sup> may lack practical significance if the sensor is implemented in full. Second, the three upwind bias coefficients (ηeno, ηlnr, ηv) regulate dissipation uniformly across all characteristic wave families instead of distinguishing by physical character. The wave-appropriate framework addresses two of these issues by design: shear/vortical waves are centralized (η = 0.5), and the entropy wave is treated with selective limiting, both of which are informed by eigenvector structure. Feng et al. thus empirically recovered what the wave-appropriate framework derives from first principles: their optimized η<sup>v</sup> converges to a small value aligning with the central treatment of vortical structures—the assignment determined by physical reasoning in the present framework. Only the acoustic upwind bias η<sup>a</sup> remains to be determined. Consequently, the degrees of freedom are reduced from four empirically optimized parameters requiring 100 surrogateassisted evaluations to a single bounded scalar addressed with approximately 25 direct evaluations via Brent's method. This reduction is due to resolving three degrees of freedom through physical reasoning rather than to the optimizer's efficiency. Additionally, Feng et al. calibrate with three configurations—the inviscid Taylor-Green vortex and both 2D and 3D implosion problems—because their dispersion objective cannot be isolated with the Taylor-Green vortex alone, requiring data-driven dissipation learning for wave specificity. The present optimization uses only Taylor-Green vortex variants: the subsonic inviscid case supplies the accuracy objective, while the supersonic viscous case sets the stability constraint. This sufficiency results from addressing all non-acoustic waves physically, narrowing the remaining task to determining the acoustic stability boundary.

A recent claim in the machine-learning-for-CFD literature is that classical shock-capturing schemes lack the integration of physical knowledge in cell-face reconstruction and require learned operators [\[19,](#page-39-7) [22\]](#page-39-10). A complementary view from the classical numerical methods community is provided by Van Leer [\[23\]](#page-39-11), who stated that a numerical method should use upwinding for advection and a central scheme for subsonic acoustic propagation, thus identifying the need for wave-specific treatment well before the application of machine learning to CFD. Roe [\[24\]](#page-39-12) observed that the two-dimensional Euler equations involve eight distinct reconstruction parameters corresponding to various wave families and sweep directions, parameters inherently absent from schemes designed for the scalar linear advection equation [\[3\]](#page-38-2). The wave-appropriate reconstruction framework [\[1,](#page-38-0) [3\]](#page-38-2) was developed from physical reasoning regarding the Euler equations' characteristic structure, and it was subsequently noted that it closely aligns with Roe's multidimensional upwinding concepts [\[24\]](#page-39-12). The advantage here is clear: since all wave-specific dissipation requirements are addressed at the design stage, only η<sup>a</sup> remains to be optimized, reducing CFD evaluations compared to end-to-end DNS training.

All comparisons and discussions in this paper refer to the published descriptions of the cited methods. The author has aimed to describe prior work accurately and objectively; any differences highlighted here reflect technical distinctions only, not judgments of the quality of the respective work.

#### 1.3. Contributions of this work

The first contribution identifies the minimum acoustic upwind bias η ∗ a for the wave-appropriate framework. The optimization yields η ∗ <sup>a</sup> = 0.54 for the third-order scheme and η ∗ <sup>a</sup> = 0.6010 for the fifth-order scheme. Both values transfer without retuning from subsonic turbulence to supersonic flows at no additional computational cost.

Remark 1.2 (On η ∗ a ). The value η<sup>a</sup> = 0.5 (fully central acoustics) was tested during the development of the original wave-appropriate framework [\[1\]](#page-38-0) aand found to be unstable, though this result was not reported therein. This prompted the present optimization, aimed at identifying the minimum stable value

in the interval (0.5, 1]. A further motivation arises in hypersonic flows, where boundary-layer transition is dominated by acoustic instability modes (Mack modes), and the acoustic upwind bias directly controls the numerical dissipation acting on these modes. Lowering η ∗ a could therefore enhance the fidelity of hypersonic transition simulations such as those in [\[2\]](#page-38-1). This potential improvement remains an avenue for future work.

The second contribution shows that the contact discontinuity detector used in the prior approach [\[4\]](#page-38-3) is unnecessary. Near a contact discontinuity, the reconstruction of a conservative variable suffers from an error in the entropy characteristic amplitude. This error can be corrected by performing a rank-1 update along the entropy right eigenvector at a cost of one dot product and five multiply-accumulate operations per interface. The resulting algorithm uses only the Ducros sensor and resolves contacts algebraically rather than through explicit detection. This approach reduces wall time by 29–41% across various benchmark configurations. Notably, the correction is limiter-agnostic and can be seamlessly integrated into any scheme that utilizes conservative-characteristic reconstruction. Unlike previous optimization efforts that modify existing dissipation operators, this rank-1 entropy wave correction introduces a novel algebraic mechanism for contact resolution. Consequently, any conservative-characteristic scheme can now inherit wave-appropriate contact treatment without the need for explicit detection or additional empirical parameters.

The third contribution builds on this by demonstrating that the wave-appropriate principle extends beyond reconstruction-based schemes. For instance, the kinetic-energy-preserving (KEP) scheme [\[25\]](#page-39-13) employs first-order piecewise-constant states and constructs a non-dissipative flux using logarithmic-mean density and arithmetic velocity and pressure averages. By construction, this approach ensures zero numerical dissipation across all wave families (fluxes). However, acoustic dissipation is reintroduced at the Riemann solver level through the normal-momentum component of the dissipative flux. This targeted correction eliminates the spurious vortices produced by the unmodified KEP scheme in shear-layer flows. It confirms that the acoustic stability mechanism identified by the optimization is independent of the discretization framework.

### 1.4. Organization

The paper proceeds as follows. Section [2](#page-5-0) states the governing equations. Section [3](#page-6-0) presents the numerical methods, including the wave-appropriate reconstruction framework and the parameterized acoustic upwind bias ηa. Section [4](#page-11-0) presents the physics-constrained optimization formulation and results. Section [5](#page-13-0) presents the conservative variable reconstruction with rank-1 entropy wave correction. Section [6](#page-16-0) extends the waveappropriate acoustic dissipation principle to kinetic-energy-preserving schemes. Section [7](#page-17-0) validates the proposed schemes across a suite of benchmark test cases. Concluding remarks are given in Section [8.](#page-37-0)

#### 2. Governing Equations

In this study, the three-dimensional compressible Navier-Stokes equations are solved in Cartesian coordinates:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}^c}{\partial x} + \frac{\partial \mathbf{G}^c}{\partial y} + \frac{\partial \mathbf{H}^c}{\partial z} = \frac{\partial \mathbf{F}^v}{\partial x} + \frac{\partial \mathbf{G}^v}{\partial y} + \frac{\partial \mathbf{H}^v}{\partial z},\tag{2}$$

where U is the conservative variable vector, F c , G<sup>c</sup> , H<sup>c</sup> and F v , G<sup>v</sup> , H<sup>v</sup> , are the convective (superscript c) and viscous (superscript v) flux vectors in each coordinate direction, respectively. The conservative variable, convective, and viscous flux vectors are given as:

$$\mathbf{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho E \end{pmatrix}, \quad \mathbf{F}^c = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ \rho u v \\ \rho u w \\ \rho u H \end{pmatrix}, \quad \mathbf{G}^c = \begin{pmatrix} \rho v \\ \rho v u \\ \rho v^2 + p \\ \rho v w \\ \rho v H \end{pmatrix}, \quad \mathbf{H}^c = \begin{pmatrix} \rho w \\ \rho w u \\ \rho w v \\ \rho w^2 + p \\ \rho w H \end{pmatrix}, \quad (3a-3d)$$

$$\mathbf{F}^{v} = \begin{bmatrix} 0, \tau_{xx}, \tau_{xy}, \tau_{xz}, u\tau_{xx} + v\tau_{xy} + w\tau_{xz} - q_{x} \end{bmatrix}^{T},$$

$$\mathbf{G}^{v} = \begin{bmatrix} 0, \tau_{xy}, \tau_{yy}, \tau_{yz}, u\tau_{yx} + v\tau_{yy} + w\tau_{yz} - q_{y} \end{bmatrix}^{T},$$

$$\mathbf{H}^{v} = \begin{bmatrix} 0, \tau_{xz}, \tau_{yz}, \tau_{zz}, u\tau_{zx} + v\tau_{zy} + w\tau_{zz} - q_{z} \end{bmatrix}^{T},$$

$$(4)$$

where  $\rho$  is density, u, v, and w are the velocities in the x, y, and z directions, respectively, p is the pressure,  $E = e + \left(u^2 + v^2 + w^2\right)/2$  is the specific total energy, and  $H = E + p/\rho$  is the specific total enthalpy. The equation of state is for a calorically perfect gas, so that  $e = \frac{p}{\rho(\gamma-1)}$  is the internal energy, where  $\gamma$  is the ratio of specific heats. The components of the viscous stress tensor  $\tau$  and the heat flux q are defined in tensor notations as:

$$\tau_{ij} = \frac{\mu}{\text{Re}} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} - \frac{2}{3} \frac{\partial u_k}{\partial x_k} \delta_{ij} \right), \tag{5}$$

$$q_i = -\frac{\mu}{\text{RePrMa}(\gamma - 1)} \frac{\partial T}{\partial x_i}, \quad T = \text{Ma}^2 \gamma \frac{p}{\rho},$$
 (6)

where  $\mu$  is the dynamic viscosity, T is the temperature, Ma and Re are the Mach number and Reynolds number, and Pr is the Prandtl number.

#### 3. Numerical methods

Using a conservative numerical method, the governing equations cast in semi-discrete form for a Cartesian cell  $I_{i,j,k} = \left[x_{i-\frac{1}{2}},x_{i+\frac{1}{2}}\right] \times \left[y_{j-\frac{1}{2}},y_{j+\frac{1}{2}}\right] \times \left[z_{k-\frac{1}{2}},z_{k+\frac{1}{2}}\right]$  can be expressed via the following ordinary differential equation:

$$\frac{\mathrm{d}}{\mathrm{d}t}\check{\mathbf{U}}_{i,j,k} = \mathbf{Res}_{i,j,k} = -\left.\frac{\mathrm{d}\check{\mathbf{F}}^c}{\mathrm{d}x}\right|_{i,j,k} - \left.\frac{\mathrm{d}\check{\mathbf{G}}^c}{\mathrm{d}y}\right|_{i,j,k} - \left.\frac{\mathrm{d}\check{\mathbf{H}}^c}{\mathrm{d}z}\right|_{i,j,k} + \left.\frac{\mathrm{d}\check{\mathbf{F}}^v}{\mathrm{d}z}\right|_{i,j,k} + \left.\frac{\mathrm{d}\check{\mathbf{H}}^v}{\mathrm{d}z}\right|_{i,j,k},$$
(7)

where the check accent,  $(\dot{\cdot})$ , indicates a numerical approximation of physical quantities,  $\mathbf{Res}_{i,j,k}$  is the residual function, and the remaining terms are cell centre numerical flux derivatives of the physical fluxes in Eq. (2). For brevity, we continue with only the x-direction as it is straightforward to extend to all three dimensions in a dimension-by-dimension manner. The indices j and k are also dropped for simplicity. In the following sections, viscous and convective flux discretization will be presented.

# 3.1. Viscous flux spatial discretization

Viscous fluxes,  $\check{\mathbf{F}}^v$ , are computed using the fourth-order  $\alpha$ -damping scheme of Nishikawa [26]. In the one-dimensional scenario, the cell center numerical viscous flux derivative is:

$$\frac{\mathrm{d}\check{\mathbf{F}}^v}{\mathrm{d}x}\Big|_{i} = \frac{1}{\Delta x} \left( \check{\mathbf{F}}^v_{i+\frac{1}{2}} - \check{\mathbf{F}}^v_{i-\frac{1}{2}} \right). \tag{8}$$

The cell interface numerical viscous flux is computed as,

$$\check{\mathbf{F}}_{i+\frac{1}{2}}^{v} = \begin{pmatrix} 0 \\ \tau_{i+\frac{1}{2}} \\ \tau_{i+\frac{1}{3}} u_{i+\frac{1}{3}} + q_{i+\frac{1}{3}} \end{pmatrix}, \tau_{i+\frac{1}{2}} = \frac{4}{3} \mu_{i+\frac{1}{2}} \left. \frac{\partial u}{\partial x} \right|_{i+\frac{1}{2}}, \quad q_{i+\frac{1}{2}} = -\kappa_{i+\frac{1}{2}} \left. \frac{\partial T}{\partial x} \right|_{i+\frac{1}{2}}. \tag{9}$$

For an arbitrary variable,  $\phi$ , the  $\alpha$ -damping approach computes cell interface gradients as:

$$\frac{\partial \phi}{\partial x}\Big|_{i+\frac{1}{2}} = \frac{1}{2} \left( \frac{\partial \phi}{\partial x}\Big|_{i} + \frac{\partial \phi}{\partial x}\Big|_{i+1} \right) + \frac{\alpha}{2\Delta x} \left( \phi_{R} - \phi_{L} \right), \phi_{L} = \phi_{i} + \frac{\partial \phi}{\partial x}\Big|_{i} \frac{\Delta x}{2}, \phi_{R} = \phi_{i+1} - \frac{\partial \phi}{\partial x}\Big|_{i+1} \frac{\Delta x}{2}, \quad (10)$$

where, in this work,  $\alpha = 8/3$ . The gradients at cell centers are computed using the second-order central-difference approximation, as in [26], which is as follows:

$$\left. \frac{\partial \phi}{\partial x} \right|_{i} = \frac{\phi_{i+1} - \phi_{i-1}}{2\Delta x}.\tag{11}$$

#### 3.2. Convective flux spatial discretization

Similar to the viscous flux discretization, the cell centre numerical convective flux derivative is expressed as:

$$\frac{\mathrm{d}\check{\mathbf{F}}^c}{\mathrm{d}x}\Big|_i = \frac{1}{\Delta x} \left( \check{\mathbf{F}}^c_{i+\frac{1}{2}} - \check{\mathbf{F}}^c_{i-\frac{1}{2}} \right),\tag{12}$$

where  $i \pm \frac{1}{2}$  indicates right and left cell interface values, respectively.  $\check{\mathbf{F}}^c_{i\pm\frac{1}{2}}$  are computed using an approximate Riemann solver. This work uses the HLLC [27] or HLL [28] approximate Riemann solvers unless otherwise explicitly stated. The numerical fluxes at cell boundaries computed using a Riemann solver can be expressed in the following standard form:

$$\check{\mathbf{F}}_{i\pm\frac{1}{2}}^{c} = \frac{1}{2} \left[ \check{\mathbf{F}}^{c} \left( \mathbf{U}_{i\pm\frac{1}{2}}^{L} \right) + \check{\mathbf{F}}^{c} \left( \mathbf{U}_{i\pm\frac{1}{2}}^{R} \right) \right] - \frac{1}{2} \left| \mathbf{A}_{i\pm\frac{1}{2}} \right| \left( \mathbf{U}_{i\pm\frac{1}{2}}^{R} - \mathbf{U}_{i\pm\frac{1}{2}}^{L} \right), \tag{13}$$

where the L and R superscripts denote the leftand right-biased states, respectively, and  $\left|\mathbf{A}_{i\pm\frac{1}{2}}\right|$  denotes the convective flux Jacobian. The objective is to obtain the leftand right-biased states,  $\mathbf{U}_{i\pm\frac{1}{2}}^{L}$  and  $\mathbf{U}_{i\pm\frac{1}{2}}^{R}$ . The procedure to obtain these interface values is described in the following sections.

#### 3.2.1. Linear and nonlinear schemes

In this subsection, we provide the details of the calculations of candidate polynomials that can be used to approximate the values of  $\mathbf{U}_{i\pm\frac{1}{2}}^L$  and  $\mathbf{U}_{i\pm\frac{1}{2}}^R$ . While the original wave-appropriate reconstruction approach of Chamarthi et al. [1, 2] used a gradient-based reconstruction approach [29, 30, 31], here we consider the widely used standard third and fifth-order schemes [32], which are as follows:

**Third-order linear and nonlinear schemes:** The third-order upwind schemes for obtaining the values of the left and right interfaces are as follows:

$$\phi_{i+1/2}^{L3,Linear} = -\frac{1}{6}\phi_{i-1} + \frac{5}{6}\phi_{i+0} + \frac{1}{3}\phi_{i+1}, 
\phi_{i+1/2}^{R3,Linear} = -\frac{1}{6}\phi_{i+2} + \frac{5}{6}\phi_{i+1} + \frac{1}{3}\phi_{i+0},$$
(14)

where  $\phi$  is an arbitrary variable, either conservative (**U**) or characteristic (**C**) variables are used in the present paper. The superscripts L3 and R3 denote left-biased and right-biased third-order formulas, respectively. The Linear superscript indicates that the scheme is linear; for brevity, the third-, fifth-, and seventh-order linear upwind schemes will be referred to as U-3, U-5, and U-7, respectively, throughout the remainder of the paper. As widely recognized from Godunov's barrier theorem [33], any linear scheme leads to oscillations near discontinuities and necessitates a nonlinear approach [34]. The third-order nonlinear scheme for the linear scheme mentioned above is the MUSCL scheme [35], which is briefly outlined below:

$$\phi_{i+1/2}^{L,MUSCL} = \phi_{i+0} + \frac{1}{4} \left[ (1 - \bar{\kappa}) \ddot{\Delta}_{i-1/2} \phi + (1 + \bar{\kappa}) \tilde{\Delta}_{i+1/2} \phi \right]$$
(15)

$$\phi_{i+1/2}^{R,MUSCL} = \phi_{i+1} - \frac{1}{4} \left[ (1 - \bar{\kappa}) \tilde{\Delta}_{i+3/2} \phi + (1 + \bar{\kappa}) \ddot{\Delta}_{i+1/2} \phi \right]$$
(16)

$$\tilde{\Delta}_{i+1/2}\phi = \operatorname{minmod}\left(\Delta_{i+1/2}\phi, 2\Delta_{i-1/2}\phi\right),$$

$$\tilde{\Delta}_{i+1/2}\phi = \operatorname{minmod}\left(\Delta_{i+1/2}\phi, 2\Delta_{i+3/2}\phi\right),$$
(17)

where minmod  $(a,b) = \frac{1}{2} [\operatorname{sgn}(a) + \operatorname{sgn}(b)] \min(|a|,|b|)$ . The coefficient  $\bar{\kappa}$  determines the accuracy of the approximation, and when  $\bar{\kappa} = \frac{1}{3}$ , the approximation has third-order accuracy.

**Fifth-order linear and nonlinear schemes:** Similarly, the fifth-order upwind schemes for obtaining the values of the left and right interfaces are as follows:

$$\phi_{i+1/2}^{L5,Linear} = \frac{1}{30}\phi_{i-2} - \frac{13}{60}\phi_{i-1} + \frac{47}{60}\phi_{i+0} + \frac{9}{20}\phi_{i+1} - \frac{1}{20}\phi_{i+2},$$

$$\phi_{i+1/2}^{R5,Linear} = \frac{1}{30}\phi_{i+3} - \frac{13}{60}\phi_{i+2} + \frac{47}{60}\phi_{i+1} + \frac{9}{20}\phi_{i+0} - \frac{1}{20}\phi_{i-1},$$
(18)

The superscripts L5 and R5 denote leftand right-biased fifth-order formulas, respectively. Similar to the third-order scheme, the fifth-order linear scheme, Eq. 18, leads to oscillations, and we utilized the MP limiting approach of Suresh and Huynh [36]. The following details the MP limiting procedure specifically for the left-biased state, although the procedure is identical for the right-biased state.

$$\phi_{i+\frac{1}{2}}^{L,MP5} = \begin{cases} \phi_{i+\frac{1}{2}}^{L,\text{Linear}} & \text{if } \left(\phi_{i+\frac{1}{2}}^{L,\text{Linear}} - \phi_i\right) \left(\phi_{i+\frac{1}{2}}^{L,\text{Linear}} - \phi_{i+\frac{1}{2}}^{L,\text{MP}}\right) \leq 10^{-40}, \\ \phi_{i+\frac{1}{2}}^{L,\text{NL}} & \text{otherwise,} \end{cases}$$
(19)

where  $\phi_{i+\frac{1}{2}}^{L,Linear}$  corresponds to Eq. 18 and the remaining terms are:

$$\phi_{i+\frac{1}{2}}^{L,\text{NL}} = \phi_{i+\frac{1}{2}}^{L,\text{lin}} + \text{minmod} \left( \phi_{i+\frac{1}{2}}^{L,\text{MIN}} - \phi_{i+\frac{1}{2}}^{L,\text{lin}}, \phi_{i+\frac{1}{2}}^{L,\text{MAX}} - \phi_{i+\frac{1}{2}}^{L,\text{lin}} \right),$$

$$\phi_{i+\frac{1}{2}}^{L,\text{MP}} = \phi_{i} + \text{minmod} \left[ \phi_{i+1} - \phi_{i}, 4 \left( \phi_{i} - \phi_{i-1} \right) \right],$$

$$\phi_{i+\frac{1}{2}}^{L,\text{MIN}} = \text{max} \left[ \min \left( \phi_{i}, \phi_{i+1}, \phi_{i+\frac{1}{2}}^{L,\text{MD}} \right), \min \left( \phi_{i}, \phi_{i+\frac{1}{2}}^{L,\text{UL}}, \phi_{i+\frac{1}{2}}^{L,\text{LC}} \right) \right],$$

$$\phi_{i+\frac{1}{2}}^{L,\text{MAX}} = \min \left[ \max \left( \phi_{i}, \phi_{i+1}, \phi_{i+\frac{1}{2}}^{L,\text{MD}} \right), \max \left( \phi_{i}, \phi_{i+\frac{1}{2}}^{L,\text{UL}}, \phi_{i+\frac{1}{2}}^{L,\text{LC}} \right) \right],$$

$$\phi_{i+\frac{1}{2}}^{L,\text{MD}} = \frac{1}{2} \left( \phi_{i} + \phi_{i+1} \right) - \frac{1}{2} d_{i+\frac{1}{2}}^{L,M}, \quad \phi_{i+\frac{1}{2}}^{L,\text{UL}} = \phi_{i} + 4 \left( \phi_{i} - \phi_{i-1} \right),$$

$$\phi_{i+\frac{1}{2}}^{L,\text{LC}} = \frac{1}{2} \left( 3\phi_{i} - \phi_{i-1} \right) + \frac{4}{3} d_{i-\frac{1}{2}}^{L,M},$$

$$d_{i+\frac{1}{2}} = \min \text{minmod} \left( d_{i}, d_{i+1} \right),$$

$$d_{i} = \phi_{i-1} - 2\phi_{i} + \phi_{i+1}.$$

$$(20)$$

The fifth-order upwind method with MP limiting is referred to as MP5 in this paper, as in [36].

**Central schemes:** The above-mentioned schemes are leftand right-biased upwind schemes. They provide necessary dissipation in some regions, but are unsuitable for turbulent simulations because they require a low-dissipation central scheme. A central scheme can be obtained by averaging the leftand right-biased upwind reconstructions (they can also be derived in other possible ways).

$$\phi_{i+\frac{1}{2}}^{L} = \phi_{i+\frac{1}{2}}^{C} = (1 - \eta) \,\phi_{i+\frac{1}{2}}^{L,Linear} + \eta \phi_{i+\frac{1}{2}}^{R,Linear}, \tag{21a}$$

$$\phi_{i+\frac{1}{2}}^{R} = \phi_{i+\frac{1}{2}}^{C} = \eta \phi_{i+\frac{1}{2}}^{L,Linear} + (1 - \eta) \phi_{i+\frac{1}{2}}^{R,Linear}, \tag{21b}$$

where  $\eta = 0.5$  and  $(\cdot)^C$  denotes the centralized reconstruction. An Upwind scheme implies that the value of  $\eta$  will be one for all the waves, and for  $\eta = 0.5$  it will be a central scheme. The well-known fourth-order central scheme can be obtained by averaging the leftand right-biased reconstruction formulas given by Equation (14), and is as follows:

$$\phi_{i+\frac{1}{2}}^{C4,Linear} = \frac{1}{2} \left( \phi_{i+1/2}^{L3,Linear} + \phi_{i+1/2}^{R3,Linear} \right) = \frac{1}{12} \left( -\phi_{i-1} + 7\phi_i + 7\phi_{i+1} - \phi_{i+2} \right). \tag{22}$$

Similarly, the sixth-order central scheme can be obtained by averaging the leftand right-biased reconstruction formulas given by Equation (18), and is as follows:

$$\phi_{i+\frac{1}{2}}^{C6,Linear} = \frac{1}{2} \left( \phi_{i+1/2}^{L5,Linear} + \phi_{i+1/2}^{R5,Linear} \right) = \frac{1}{60} \left( \phi_{i-2} - 8\phi_{i-1} + 37\phi_i + 37\phi_{i+1} - 8\phi_{i+2} + \phi_{i+3} \right). \tag{23}$$

How these linear and nonlinear schemes are utilized in the wave-appropriate reconstruction approach of Hoffmann, Chamarthi, and Frankel [2] will be presented in the following section.

**Remark 3.1.** The numerical schemes presented in this paper are described in a self-contained manner. Standard formulas for the thirdand fifth-order upwind reconstructions are reproduced for completeness and to establish the notation used throughout.

#### 3.2.2. Wave-appropriate reconstruction approach

In Ref. [1], the authors took advantage of the wave structure of the Euler equations. Once the variables are transformed from physical (U) to characteristic space (C), the characteristic variables have specific properties. The first and last variables are acoustic waves; the second variable is the entropy wave; and the rest are shear waves. Together, the entropy and shear waves are known as linearly degenerate waves. The density varies across the entropy wave in characteristic space, and the rest of the variables remain unchanged. Ref. [1] took advantage of this and reduced the frequent activation of the MP criterion, Equation (19). Shock waves are detected by the Ducros sensor, which is used for waves (characteristic variables) other than entropy waves. However, all the waves are computed using the upwind, inherently dissipative schemes,  $\phi_{i+\frac{1}{2}} = (1-\eta) \phi_{i+\frac{1}{2}}^L + \eta \phi_{i+\frac{1}{2}}^R$ , where  $\eta = 1$ .

In [2], all the linearly degenerate waves are computed using a central scheme,  $\eta = 0.5$ , if the necessary discontinuity detection criteria are met. The acoustic waves are still computed using an upwind scheme for stability, and the proposed approach significantly improved the results on benchmark test cases and predicted hypersonic transitional flows. The algorithm is outlined below:

Step 1. Compute Roe-averaged variables at the interface to construct the left,  $\mathbf{L}_n$ , and right,  $\mathbf{R}_n$ , eigenvectors of the normal convective flux Jacobian.

**Step 2.** For the fifth-order upwind scheme transform  $\mathbf{U}_i$  to characteristic space by multiplying them by  $\mathbf{L}_n$ 

$$\mathbf{C}_{i+m,b} = \mathbf{L}_{n,i+\frac{1}{2}} \mathbf{U}_{i+m},\tag{24a}$$

for m = -2, -1, 0, 1, 2, 3 and b = 1, 2, 3, 4, 5, representing the vector of characteristic variables which are defined as follows in the current implementation:

Table 1: Characteristic wave types.

| b = 1, 5 | b=2             | b = 3, 4       |
|----------|-----------------|----------------|
| Acoustic | Entropy/Contact | Shear/Vortical |

For the third-order scheme, m = -1, 0, 1, 2 is sufficient, as the stencil is smaller.

**Step 3.** The left-biased reconstruction for the fifth-order scheme is then treated by the following algorithm:

$$C_{i+1/2,\,b}^{L} = \begin{cases} \text{if } b = 1,5 \text{ (acoustic):} & \begin{cases} C_{i+1/2,\,b}^{L,\,\text{MP5}} & \text{if } \Omega_d > 0.01, \text{( Refer Eq. 19)} \\ C_{i+1/2,\,b}^{L5,\,\text{Linear}} & \text{otherwise, ( Refer Eq. 18)} \end{cases}$$

$$C_{i+1/2,\,b}^{L} = \begin{cases} \text{if } b = 2 \text{ (entropy):} & \begin{cases} C_{i+1/2,\,b}^{L,\,\text{MP5}}, \text{( Refer Eq. 19)} \\ C_{i+1/2,\,b}^{L,\,\text{MP5}} & \text{if } \Omega_d > 0.01, \text{( Refer Eq. 19)} \\ C_{i+1/2,\,b}^{C6,\,\text{Linear}} & \text{otherwise, ( Refer Eq. 23)} \end{cases}$$

$$(25)$$

where the Ducros sensor is computed as follows:

$$\Omega_d = \frac{|-p_{i-2} + 16p_{i-1} - 30p_i + 16p_{i+1} - p_{i+2}|}{|p_{i-2} + 16p_{i-1} + 30p_i + 16p_{i+1} + p_{i+2}|} \frac{(\nabla \cdot \mathbf{u})^2}{(\nabla \cdot \mathbf{u})^2 + |\nabla \times \mathbf{u}|^2},$$
(26)

where **u** represents the velocity vector, and the derivatives of velocities are computed using Equation (11), which involves second-order finite-difference approximation of the first derivative. The sensor value is taken as the maximum of  $\Omega_d$  within a three-cell neighborhood.

Likewise, the algorithm for the third-order MUSCL scheme is as follows:

$$C_{i+1/2, b}^{L} = \begin{cases} \text{if } b = 1, 5 \text{ (acoustic):} & \begin{cases} C_{i+1/2, b}^{L, \text{MUSCL}} & \text{if } \Omega_d > 0.01, (\text{ Refer Eq. 15}) \\ C_{i+1/2, b}^{L3, \text{ Linear}} & \text{otherwise, (Refer Eq. 14}) \end{cases} \\ C_{i+1/2, b}^{L} = \begin{cases} \text{if } b = 2 \text{ (entropy):} & \begin{cases} C_{i+1/2, b}^{L, \text{MUSCL}}, (\text{ Refer Eq. 15}) \\ if b = 3, 4 \text{ (shear/vortical):} \end{cases} \\ \begin{cases} C_{i+1/2, b}^{L, \text{MUSCL}} & \text{if } \Omega_d > 0.01, (\text{ Refer Eq. 15}) \\ C_{i+1/2, b}^{C4, \text{Linear}} & \text{otherwise, (Refer Eq. 22}) \end{cases} \end{cases}$$

$$\Omega_d = \max(\Omega_{i+m}), \quad \text{for } m = -1, 0, 1.$$
 (28)

A similar procedure is carried out for the right-biased reconstruction.

**Step 4.** After obtaining  $C_{i+\frac{1}{2},b}^{L,R}$ , the variables are transformed back to physical fields:

$$\mathbf{U}_{i+\frac{1}{2}}^{L,R} = \mathbf{R}_{n,i+\frac{1}{2}} \mathbf{C}_{i+\frac{1}{2}}^{L,R}.$$
 (29)

Remark 3.2 (Naming convention of the schemes). The wave-appropriate algorithm employing the fifth-order (and sixth) scheme, as outlined in Eq. 25, will be referred to as WA-5 in this paper. Similarly, the algorithm using the third-order (and fourth-order) scheme, as presented in Eqs. 27, will be denoted WA-3 in this paper.

As shown in the above algorithms, for both fifthand third-order algorithms, the acoustic waves (b = 1, 5) are always computed using an upwind scheme,  $C^{L3,Linear}$  or  $C^{L5,Linear}$ . It implies that in the following equation.

$$\phi_{i+1/2}^{L} = \eta_a \,\phi_{i+1/2}^{L, \, \text{upwind}} + (1 - \eta_a) \,\phi_{i+1/2}^{R, \, \text{upwind}}, \qquad \phi_{i+1/2}^{R} = (1 - \eta_a) \,\phi_{i+1/2}^{L, \, \text{upwind}} + \eta_a \,\phi_{i+1/2}^{R, \, \text{upwind}}, \qquad (30)$$

 $\eta_a=1$ , acoustic dissipation is at its maximum. In [1, 2], the amount of acoustic dissipation is always taken as one due to the stability issues. For  $\eta_a=0.5$ , all characteristic waves are centralized, and the scheme carries no dissipation on any field. For compressible flows, this leads to energy accumulation at the grid scale and eventual divergence. Therefore, there exists a minimum value  $\eta_{a,\min}$  below which the scheme is unstable for a wide range of flow regimes. The optimization procedure that identifies  $\eta_a^*$  (i.e., the stable optimum) for each scheme is described in Section 4.

Remark 3.3 (On the Ducros sensor formulation). The sensor  $\Omega_d$  defined in Eq. (26) is the product of two components: a pressure-based shock indicator and the dilatation-to-vorticity ratio. This is the formulation originally proposed by Ducros et al. [6], where the pressure-based component corresponds to the Jameson sensor  $\Psi$  (Eqs. 12–13 of [6]) and the dilatation-to-vorticity ratio corresponds to the correction  $\Phi$  (Eq. 22 of [6]). The sensor used throughout that reference is the product  $\Phi\Psi$ , as recorded in Tables I–III of [6]. In much of the subsequent literature, only the dilatation-to-vorticity component  $\Phi$  has been retained and referred to as the Ducros sensor, while the pressure-based component  $\Psi$  has been dropped. This simplification appears, for example, in the formulation of Feng et al. [20]. The pressure-based component is not redundant: the dilatation-to-vorticity ratio alone can misidentify strong vortical regions as shocked, because intense vortices in compressible flow produce non-negligible dilatation even in the absence of discontinuities [15]. The pressure indicator filters such regions because vortices do not produce the sharp pressure

jumps characteristic of shocks. The robustness of the fixed threshold  $\Omega_d > 0.01$  used throughout this work, applied uniformly across all test cases from subsonic turbulence to Mach 10 flows without case-by-case adjustment, may be attributed in part to the use of the complete two-component formulation. Readers seeking to reproduce the algorithms presented here should note that  $\Omega_d$  as defined in Eq. (26) includes both components. Using only the dilatation-to-vorticity ratio would yield a different and potentially less robust sensor.

### 4. Physics-Constrained Optimization of $\eta_a$

#### 4.1. Problem formulation

The stability boundary with respect to  $\eta_a$  cannot be determined analytically for the full nonlinear scheme, because the nonlinear interactions between the spatial reconstruction, limiting procedures, shock sensor, and approximate Riemann solver collectively determine stability in a way that is not captured by linear modified wavenumber analysis or even approximate dispersion analysis [37]. We therefore treat the discrete operator as a black box and seek.

$$\eta_a^* = \underset{\eta_a \in \mathcal{S}}{\arg \min} \ \mathcal{J}_{\text{acc}}(\eta_a), \tag{31}$$

where  $S = \{\eta_a \in [0.5, 1] : \text{all stability tests pass} \}$  is the feasible set and  $\mathcal{J}_{acc}$  is the accuracy objective defined below. The lower bound  $\eta_a = 0.5$  corresponds to a perfectly symmetric reconstruction carrying zero acoustic upwind bias. Throughout the optimization (and for all the cases shown later in the paper), shocked cells (identified by  $\sigma > 0.01$ ) always retain  $\eta_a = 1$  via the runtime selector; the optimization therefore acts exclusively on the smooth-region acoustic bias.

#### 4.1.1. Accuracy objective

The three-dimensional inviscid Taylor-Green vortex (TGV) at Ma = 0.1 on a  $64^3$  grid serves as the accuracy benchmark. The accuracy objective is the time-integrated absolute error in volume-averaged turbulent kinetic energy:

$$\mathcal{J}_{\rm acc}(\eta_a) = \int_0^{t_f} \left| \overline{E}_k^{\rm num}(t; \eta_a) - \overline{E}_k^{\rm ref}(t) \right| dt, \tag{32}$$

where  $t_f = 10$  is the final simulation time. Excess acoustic dissipation manifests as kinetic energy that decays faster than the reference, producing a large  $\mathcal{J}_{acc}$ . The reference kinetic energy profile  $\overline{E}_k^{ref}(t)$  is obtained from a linear fully upwinded scheme of order N+2 on the same  $64^3$  grid: a fifth-order linear upwind scheme for optimizing WA-3, and a seventh-order linear upwind scheme for optimizing WA-5. The optimization, therefore, targets the spectral resolution of the next standard scheme in the order hierarchy (the least dissipative linear baseline one would choose).

Remark 4.1. Since no DNS exists for the inviscid Taylor-Green vortex, the (N+2)th-order linear scheme on the same  $64^3$  grid serves as the accuracy reference. The energy spectra in Figure 4c confirm that the optimized scheme resolves finer scales than this reference, validating the choice of  $\mathcal{J}_{acc}$ . It should further be stressed that WA-3 and WA-5 are nonlinear schemes being judged against a linear baseline; that the optimized nonlinear Nth-order scheme matches or outperforms the linear (N+2)th-order scheme is therefore a non-trivial result.

#### 4.1.2. Stability constraint

A simulation is declared unstable if it encounters non-finite values or terminates prematurely. The stability test suite consists of the three-dimensional supersonic viscous TGV (Re = 1600) on both a coarse ( $64^3$ ) and a refined ( $128^3$ ) grid. This problem subjects the scheme to strong compressibility effects, vortex stretching, and under-resolved features at grid scales where the physical viscous dissipation is insufficient to ensure numerical stability. Testing at two resolutions ensures that apparent stability at the coarser resolution is not an artifact of excessive grid-scale dissipation.

As documented by Lusher and Sandham [38], the supersonic Taylor-Green vortex at Ma = 1.25 develops eight distinct shock waves due to the symmetries of the initial condition, which propagate across the periodic

boundaries and interact with one another to form complex, highly unsteady shock systems, with local Mach numbers peaking at Ma  $\approx 2$  at t=6. This constitutes a genuinely three-dimensional scenario in which multiple strong interacting shock waves coexist with turbulent structures throughout the domain, testing the full wave-appropriate algorithm, including the Ducros sensor, the characteristic reconstruction path in shocked regions, and the conservative reconstruction with rank-1 entropy wave correction in smooth regions (to be presented later).

The subsonic TGV also introduces a second, physically distinct instability criterion: at  $\eta_a=0.5$  the scheme carries no acoustic dissipation and kinetic energy grows monotonically rather than decaying, which at sufficiently small  $\eta_a$  eventually produces non-finite values. A subsonic simulation is therefore declared unstable if  $\overline{E}_k^{\text{num}}$  increases monotonically after the initial transient or if non-finite values are encountered.

#### 4.2. Optimization algorithm

The scalar bounded minimization in Eq. (31) is solved with Brent's method [39].<sup>1</sup> Each function evaluation proceeds sequentially: the supersonic TGV is run first on the coarse grid, then on the refined grid. If either of the two crashes, the objective function is set to  $\mathcal{J}_{acc} = +1$ , and the subsonic TGV is completely skipped. Since all feasible evaluations satisfy  $\mathcal{J}_{acc} \ll 1$ , +1 serves as an unambiguous indicator of an infeasible region. The subsonic TGV case is run only if both stability tests pass. This ordering minimizes wall time per function evaluation because a crashed supersonic simulation runs much faster than subsonic runs.

#### Algorithm 1 Physics-constrained optimization of acoustic upwind bias

```
Require: Bounds [\eta_{a,\min},\eta_{a,\max}]=[0.5,1.0]; reference \overline{E}_k^{\,\mathrm{ref}}(t)
Ensure: Optimal smooth-region acoustic bias \eta_a^*
 1: function Objective(\eta_a)
         Run supersonic viscous TGV (coarse, 64^3) with smooth-region bias \eta_a
 2:
            (shocked cells \sigma > 0.01 use \eta_a = 1 via runtime selector)
         if crashed then return +1
 3:
         end if
 4:
         Run supersonic viscous TGV (refined, 128<sup>3</sup>) with smooth-region bias \eta_a
 5:
            (shocked cells \sigma > 0.01 use \eta_a = 1 via runtime selector)
         if crashed then return +1
 6:
         end if
 7:
         Run subsonic inviscid TGV (64<sup>3</sup>) with bias \eta_a
 8:
            (\sigma \leq 0.01 \text{ everywhere, so } \eta_a \text{ applies globally})
         if crashed or \overline{E}_k^{\text{num}} increases monotonically then return +1
 9:
         end if
10:
         return \mathcal{J}_{acc}(\eta_a)
11:
12: end function
13: \eta_a^* \leftarrow \text{BrentMinimize}(\text{Objective}, 0.5, 1.0)
```

#### 4.3. Results

Figure 3 shows how the optimizer progresses (the objective value  $\mathcal{J}_{\rm acc}(\eta_a)$  at each step) for both orders. Early steps with  $\mathcal{J}_{\rm acc}(\eta_a) = +1$  happen when  $\eta_a$  is in the unstable range, triggering an early stop. When Brent's method finds the stability limit, the next tries are in the safe zone, and  $\mathcal{J}_{\rm acc}(\eta_a)$  quickly drops to its lowest point. Both optimizations finish in about 25 tries. The objective function levels off after about 10–12 steps; later tries fine-tune  $\eta_a^*$  to four decimal places.

#### 4.3.1. Third-order scheme

For WA-3, the optimizer converges to  $\eta_a^* = 0.54$  with  $\mathcal{J}_{acc}^* = 0.0274$ . The feasible region begins sharply at  $\eta_a \approx 0.54$ : below this value, every supersonic TGV terminates with non-finite values regardless of resolution,

<sup>&</sup>lt;sup>1</sup>Implemented as scipy.optimize.minimize\_scalar with method='bounded' in SciPy.

so the stability boundary is a hard threshold rather than a gradual degradation. Once in the feasible region,  $\mathcal{J}_{acc}$  increases monotonically with  $\eta_a$ , meaning additional acoustic dissipation beyond the stability threshold provides no accuracy benefit and only degrades resolution. The optimal value sits precisely at the stability boundary, only 4% above the fully central scheme ( $\eta_a = 0.5$ ).

#### 4.3.2. Fifth-order scheme

For WA-5, the optimizer converges to  $\eta_a^* = 0.6010$  with  $\mathcal{J}_{\rm acc}^* = 0.0115$ . The stability boundary is shifted to a higher value of  $\eta_a$  relative to WA-3. This shift is a direct consequence of the lower dissipation and better dispersion properties of the fifth-order scheme compared with the third-order scheme. Therefore, the fifth-order scheme requires a larger explicit acoustic upwind bias to stabilize it in the supersonic regime near discontinuities. Nevertheless, the fifth-order scheme accepts a higher stability threshold in exchange for significantly lower dissipation within the feasible region, as reflected by the smaller optimal objective value ( $\mathcal{J}_{\rm acc}^* = 0.0115$  versus 0.0274 for WA-3).

![](_page_13_Figure_3.jpeg)

Figure 3: Optimizer convergence traces for the physics-constrained optimization of  $\eta_a$ . Each point represents one function evaluation by Brent's method. Points at  $\mathcal{J}_{acc}(\eta_a) = +1$  correspond to simulations that triggered the early-return criterion: either a supersonic TGV crash or monotonically growing kinetic energy in the subsonic TGV. The red star marks the optimal value  $\eta_a^*$ .

The values of  $\eta_a^*$  determined by the optimizer replace the default upwind value of one for the acoustic waves, as specified in Equation 30. The values for the third and fifth-order schemes are summarized in Table 2.

Table 2: Optimal acoustic upwind bias  $\eta_a^*$  for each scheme, determined by physics-constrained optimization. For both schemes, the optimum coincides with the supersonic stability boundary.  $\mathcal{J}_{\text{acc}}^*$  is the minimum accuracy objective achieved.

| Scheme             | $\eta_a^*$ | $\mathcal{J}^*_{\mathrm{acc}}$ |
|--------------------|------------|--------------------------------|
| WA-3 (third-order) | 0.54       | 0.0274                         |
| WA-5 (fifth-order) | 0.6010     | 0.0115                         |

#### 5. Conservative variable reconstruction with Rank-1 Entropy-Wave Correction

The wave-appropriate scheme of Section 3.2.2 performs the full characteristic transformation at every interface: the stencil is projected to characteristic space via  $\mathbf{L}_n$ , each of the five wave families is reconstructed independently, and the result is projected back via  $\mathbf{R}_n$ . While physically consistent and accurate, it carries a non-trivial cost: two full matrix-vector products per stencil cell are required at every interface regardless of whether a discontinuity is present. The prior conservative-characteristic scheme of [4] reduced this cost by switching to a cheaper conservative variable reconstruction in smooth regions, but required two independent sensors to do so safely: the Ducros shock sensor detected shocks, and a separate density-based criterion detected contact discontinuities.

The primary motivation behind the current algorithm is the observation that explicit contact detection is unnecessary. Near a contact discontinuity, the density varies across the discontinuity while the velocity and pressure remain continuous. In characteristic space, this variation is solely carried by the entropy wave  $C_2$ , and all other characteristic variables remain smooth. Consequently, the error introduced by a conservative reconstruction near a contact is a rank-1 perturbation along the entropy right eigenvector, which can be corrected algebraically at negligible cost without the need for a sensor. The algorithm presented in this section, Wave-appropriate Conservative Reconstruction (WA-CR), replaces the two-sensor approach with a single Ducros shock threshold and rank-1 entropy wave correction.

Before presenting the algorithm, it is instructive to examine the right eigenvector matrix. The right eigenvector matrix of the 3-D Euler equations in the x-direction, ordered as  $(\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5) = (u - \lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5)$ c, u, u, u, u + c, is

$$\mathbf{R}_{n} = \begin{bmatrix} 1 & 1 & 0 & 0 & 1 \\ u - c & u & 0 & 0 & u + c \\ v & v & 1 & 0 & v \\ w & w & 0 & 1 & w \\ H - uc & \frac{1}{2}q^{2} & v & w & H + uc \end{bmatrix},$$
(33)

where  $q^2 = u^2 + v^2 + w^2$ , c is the speed of sound, and H is the specific total enthalpy. The columns correspond to: left-running acoustic  $(\mathbf{r}_1)$ , entropy  $(\mathbf{r}_2)$ , and two shear waves  $(\mathbf{r}_3, \mathbf{r}_4)$ , and right-running acoustic ( $\mathbf{r}_5$ ). Expanding  $\mathbf{U} = \mathbf{R}_n \mathbf{C}$  in density gives

$$\rho = C_1 + C_2 + C_5. \tag{34}$$

The shear characteristics  $C_3$  and  $C_4$  are entirely absent. The zero entries  $R_{13} = R_{14} = 0$  are not numerical coincidences but the algebraic statement that shear waves are density-neutral. This property permits central reconstruction of the shear waves without any adverse effect on the density field. Furthermore, at a contact discontinuity there is no shock, so the Ducros sensor is inactive and the acoustic waves  $C_1$  and  $C_5$  are reconstructed with the conservative path at the optimized bias  $\eta_a^*$ . The density error is therefore carried entirely by the entropy wave  $C_2$  alone. It is not necessary to detect the contact or deploy a specialized sensor: correcting  $C_2$  is both necessary and sufficient, which is the key insight behind WA-CR.

# 5.1. Conservative variable reconstruction with rank-1 correction (WA-CR)

In smooth regions ( $\Omega_d \leq 0.01$ ), WA-CR avoids the full eigenvector projections and proceeds in two sub-steps.

Step 1: Direction-dependent conservative variable reconstruction.. All five conservative variables are reconstructed directly in physical space, without eigenvector projection. The upwind bias  $\eta_a^*$  is applied to the density, energy, and normal momentum; the tangential momentum use the central blend  $\eta_a = 0.5$ :

x-sweep: 
$$\rho$$
,  $\rho u$ ,  $\rho E \leftarrow \eta_a^*$ ,  $\rho v$ ,  $\rho w \leftarrow 0.5$ , (35a)

y-sweep: 
$$\rho$$
,  $\rho w$ ,  $\rho E \leftarrow \eta_a^*$ ,  $\rho u$ ,  $\rho w \leftarrow 0.5$ , (35b)  
z-sweep:  $\rho$ ,  $\rho w$ ,  $\rho E \leftarrow \eta_a^*$ ,  $\rho u$ ,  $\rho v \leftarrow 0.5$ .

z-sweep: 
$$\rho$$
,  $\rho w$ ,  $\rho E \leftarrow \eta_a^*$ ,  $\rho u$ ,  $\rho v \leftarrow 0.5$ . (35c)

This direction-dependent biasing mirrors the wave-appropriate dissipation pattern of the full characteristic path: normal momentum behaves like an acoustic variable (requiring upwinding), while tangential momentum behaves like a shear variable (suffices to central differencing). No eigenvector projection is required. Denote the resulting interface states  $\mathbf{U}_{i+1/2}^{L,c}$  and  $\mathbf{U}_{i+1/2}^{R,c}$  (superscript "c" for *conservative*).

Step 2: Rank-1 entropy wave correction.. The conservative variable reconstruction of Step 1 carries no guarantee that the entropy wave  $C_2$  is oscillation-free near a contact discontinuity. The following procedure corrects this at minimal cost.

Entropy stencil. Using only the entropy left eigenvector l<sub>2</sub>, project the six stencil cells onto the entropy wave:

$$C_{2,j} = \mathbf{l}_2 \cdot \mathbf{U}_{i+j}, \qquad j \in \{-2, -1, 0, 1, 2, 3\}.$$
 (36)

Limited interface value. Apply the MP5 limiter to the scalar stencil  $\{C_{2,i}\}$ :

$$\hat{C}_2^L, \quad \hat{C}_2^R = \text{MP5-limit}(C_{2,-2}, \dots, C_{2,3}).$$
 (37)

Rank-1 update. Compute the deficit between the limited entropy value and the entropy content already present in the conservative state,

$$\delta^{L} = \hat{C}_{2}^{L} - \mathbf{l}_{2} \cdot \mathbf{U}_{i+1/2}^{L,c}, \qquad \delta^{R} = \hat{C}_{2}^{R} - \mathbf{l}_{2} \cdot \mathbf{U}_{i+1/2}^{R,c}, \tag{38}$$

and apply the rank-1 correction along the entropy right eigenvector  $\mathbf{r}_2 = (1, \ \bar{u}, \ \bar{v}, \ \bar{w}, \ \frac{1}{2}\bar{q}^2)^{\top}$ :

$$\mathbf{U}_{i+1/2}^{L} = \mathbf{U}_{i+1/2}^{L,c} + \delta^{L} \mathbf{r}_{2}, \qquad \mathbf{U}_{i+1/2}^{R} = \mathbf{U}_{i+1/2}^{R,c} + \delta^{R} \mathbf{r}_{2}. \tag{39}$$

In component form:

$$\rho^{L} = \rho^{L,c} + \delta^{L},$$

$$(\rho u)^{L} = (\rho u)^{L,c} + \delta^{L} \bar{u},$$

$$(\rho v)^{L} = (\rho v)^{L,c} + \delta^{L} \bar{v},$$

$$(\rho w)^{L} = (\rho w)^{L,c} + \delta^{L} \bar{w},$$

$$(\rho E)^{L} = (\rho E)^{L,c} + \delta^{L} \cdot \frac{1}{2} \bar{q}^{2}.$$

$$(40)$$

Because r<sub>2</sub> spans the entropy eigenspace, the update leaves all acoustic and shear waves exactly unchanged. Near a contact discontinuity, the MP5 limiter is applied to  $C_2$ , acting to suppress nonphysical overshoots in this variable. This limiter ensures monotonic behavior and prevents spurious oscillations near discontinuities. The correction from this clipping is then propagated back into all five conservative variables, with adjustments exactly proportional to their entropy eigenvector components, thereby preserving the proper jump relations across the contact. The computational overhead relative to a pure conservative reconstruction is one dot product per stencil cell (six operations) to form the entropy stencil, one scalar MP5 pass, and five multiply-adds per interface side. This is a small fraction of the cost of the full  $\mathbf{L}_n$  and  $\mathbf{R}_n$  projections it replaces, which require five dot products per stencil cell for the forward projection and a full matrix-vector product for the back projection.

The complete procedure is summarized in Algorithm 2. The optimized bias  $\eta_a^*$ , determined in Section 4, appears in both paths. On the smooth path, it governs the conservative-variable reconstruction of density, energy, and normal momentum. On the shocked path, it governs the reconstruction of acoustic characteristics. The single value  $\eta_a^*$  remains consistent throughout the entire domain. Two sources account for the cost reduction relative to WA-5. The smooth path avoids the full  $\mathbf{L}_n$  and  $\mathbf{R}_n$  projections at most interfaces. The contact sensor is eliminated entirely.

# **Algorithm 2** WA-CR reconstruction at interface $i + \frac{1}{2}$

```
Require: Conservative states U(j), j = i - 2, ..., i + 3; sensor \sigma = \max_{m=-1,0,1} \Omega_d(i+m)
Ensure: \mathbf{U}_{i+1/2}^L, \mathbf{U}_{i+1/2}^R
  1: Compute Roe-averaged state; form l_2, r_2
  2: C_{2,j} \leftarrow \mathbf{l}_2 \cdot \mathbf{U}(j) for j = i - 2, \dots, i + 3
                                                                                                                                        ▷ entropy stencil, shared by both paths
  3: \hat{C}_2^L, \hat{C}_2^R \leftarrow \text{MP5}(C_{2,-2}, \dots, C_{2,3})
                                                                                                                                                                       ▷ limited entropy values
  4: if \sigma \leq 0.01 then
                                                                                                                                               ⊳ smooth region - conservative path
             Reconstruct \rho, \rho u_n, \rho E with bias \eta_a^*; \rho u_t, \rho u_{t'} with \eta_a = 0.5 \rightarrow \mathbf{U}^{L,c}, \mathbf{U}^{R,c} \delta^L \leftarrow \hat{C}_2^L - \mathbf{l}_2 \cdot \mathbf{U}^{L,c}; \delta^R \leftarrow \hat{C}_2^R - \mathbf{l}_2 \cdot \mathbf{U}^{R,c} \mathbf{U}^L \leftarrow \mathbf{U}^{L,c} + \delta^L \mathbf{r}_2; \mathbf{U}^R \leftarrow \mathbf{U}^{R,c} + \delta^R \mathbf{r}_2
  6:
  7:
                                                                                                                          ▷ regions with shocks - full characteristic path
  8: else
              C_{b,j} \leftarrow \mathbf{l}_b \cdot \mathbf{U}(j) for b=1,3,4,5; C_{2,j} reused from line 2 C_1^{L,R}, C_5^{L,R} \leftarrow \text{Algorithm from Eq. 25} C_2^{L,R} \leftarrow \hat{C}_2^{L,R}
  9:
10:
                                                                                                                                                                       ▶ acoustic: upwind bias
                                                                                                                                                               ▶ entropy: reuse from line 3
11:
              C_3^2, C_4^{L,R}, C_4^{L,R} \leftarrow Algorithm from Eq. 25 \mathbf{U}^{L,R} \leftarrow \mathbf{R}_n \mathbf{C}^{L,R}
                                                                                                                                                                                        ▷ shear: central
12:
13:
14: end if
```

Generality of the rank-1 correction: Although the present work uses the MP5 limiter as the baseline reconstruction, the rank-1 entropy wave correction is not tied to any particular nonlinear scheme. The correction requires only that a scalar stencil of entropy characteristic values {C2,j} be available and that a limited interface value CˆL,R <sup>2</sup> be computed from it. Any monotonicity-preserving or essentially non-oscillatory reconstruction can serve this role. Replacing MP5 with a WENO scheme [\[40\]](#page-40-10) throughout yields the variant denoted WA-WENO-CR, which inherits the same rank-1 correction structure and the same cost savings relative to its full characteristic counterpart. Notably, WA-WENO-CR uses the same optimized bias η ∗ <sup>a</sup> = 0.6010 as WA-CR. The correction is equally applicable to the third-order MUSCL variant, though the third-order scheme's higher background dissipation makes the accuracy benefit less pronounced.

The cost of the full characteristic decomposition has been recognized since the earliest high-order schemes. Jiang and Shu [\[41\]](#page-40-11) noted in their original WENO paper: "For Euler systems of gas dynamics, we suggest computing the weights from pressure and entropy instead of the characteristic values" to simplify the costly characteristic procedure. Jiang and Shu denoted their approach as WENO-PS in the corresponding paper. The rank-1 entropy wave correction proposed here directly addresses this cost. Instead of abandoning characteristic reconstruction or approximating it with scalar surrogates, it replaces the complete decomposition with a single entropy-wave correction whenever the Ducros sensor is inactive. This approach significantly reduces the characteristic work to a single dot product and a rank-1 update per interface.

#### 6. Wave-Appropriate acoustic dissipation for kinetic-energy-preserving Schemes

The kinetic-energy-preserving (KEP) scheme of Chandrashekar [\[25\]](#page-39-13) constructs the convective flux from piecewise-constant states using logarithmic-mean density and arithmetic velocity and pressure averages, ensuring zero numerical dissipation across all wave families by construction. This property is desirable in smooth turbulent regions, but it implies that no dissipation acts on acoustic waves even when the Ducros sensor is inactive, as is the case throughout shock-free shear-layer flows. The wave-appropriate framework identifies acoustic upwind bias as the minimal stabilizing mechanism: a small bias η<sup>a</sup> above the central value of 0.5 is both necessary and sufficient for stability, while leaving non-acoustic waves unaffected. In the KEP scheme, acoustic bias cannot be introduced during the reconstruction step, as in WA-3 or WA-5, because the states are piecewise-constant. Instead, it is introduced through the dissipative component of the numerical flux, applied exclusively to the normal momentum equation. The density, tangential momentum, and energy equations retain the pure KEP flux without modification, preserving the scheme's non-dissipative character across all non-acoustic wave families. The validation of this approach is given in Section [7.2.](#page-22-0)

Specifically, the total interface flux is decomposed as

$$\hat{\mathbf{F}}_{i+1/2} = \hat{\mathbf{F}}_{i+1/2}^{\text{KEP}} + \hat{\mathbf{D}}_{i+1/2},\tag{41}$$

where Fˆ KEP is the non-dissipative kinetic-energy-preserving flux of Chandrashekar [\[25\]](#page-39-13), and Dˆ is a selective dissipation term that acts only on the normal momentum component.

The dissipation is constructed as follows. For each sweep direction, the normal momentum is reconstructed using the third-order scheme of Eq. [14](#page-7-1) with an upwind bias ηa:

$$(\rho u_n)_{i+1/2}^L = \eta_a \left(\rho u_n\right)_{i+1/2}^{L3,\text{Linear}} + (1 - \eta_a) \left(\rho u_n\right)_{i+1/2}^{R3,\text{Linear}},\tag{42}$$

$$(\rho u_n)_{i+1/2}^R = (1 - \eta_a) (\rho u_n)_{i+1/2}^{L3, \text{Linear}} + \eta_a (\rho u_n)_{i+1/2}^{R3, \text{Linear}}, \tag{43}$$

where u<sup>n</sup> is the velocity component normal to the interface (i.e., u for the x-sweep, v for the y-sweep, w for the z-sweep). The Rusanov-type dissipation for the normal momentum is then

$$\hat{D}_{n-\text{mom}} = -\frac{1}{2} \lambda_{\text{max}} \left[ (\rho u_n)_{i+1/2}^R - (\rho u_n)_{i+1/2}^L \right], \tag{44}$$

where λmax = |u˜n|+ ˜c is the maximum wave speed at the interface, computed from Roe-averaged quantities. All other components of Dˆ are identically zero:

$$\hat{D}_k = \begin{cases} \hat{D}_{n\text{-mom}} & \text{if } k = \text{normal momentum equation,} \\ 0 & \text{otherwise (density, tangential momentum, energy).} \end{cases}$$
(45)

The direction-dependent application mirrors the wave-appropriate principle: normal momentum carries the acoustic wave content, while tangential momentum carries shear wave content, and density carries entropy wave content. Adding dissipation exclusively to the normal momentum, therefore targets acoustic waves without contaminating non-acoustic fields. The procedure is summarized in Algorithm 3.

Remark 6.1 (Optimum  $\eta_a$  for the KEP scheme). An optimum  $\eta_a$  for the KEP scheme cannot be determined via the optimization procedure used in this paper for two reasons. First, the KEP scheme is linear and therefore cannot simulate the supersonic Taylor-Green vortex, which serves as the stability constraint in the optimization. Second,  $\eta_a = 0.5$  is always stable for the subsonic Taylor-Green vortex since the scheme never crashes for that case. Determining an optimal  $\eta_a$  for the KEP scheme would require a suitable nonlinear stability test, which is beyond the scope of the present work. The values  $\eta_a = 0.56$  and  $\eta_a = 1.0$  are chosen here solely to demonstrate that introducing dissipation exclusively through the normal momentum flux eliminates the spurious vortices produced by the unmodified KEP scheme in the periodic shear layer, and should not be interpreted as optimized values in the sense of Section 4.

**Algorithm 3** Wave-appropriate KEP (WA-KEP) flux at interface  $i + \frac{1}{2}$ , n-direction sweep

**Require:** Left/right states  $\mathbf{U}_L = \mathbf{U}_i$ ,  $\mathbf{U}_R = \mathbf{U}_{i+1}$ ; bias  $\eta_a$ ;

**Ensure:** Interface flux  $\hat{\mathbf{F}}_{i+1/2}$ 

- 1: Compute KEP flux  $\hat{\mathbf{F}}^{\mathrm{KEP}}$  using logarithmic-mean (or geometric-mean) density and arithmetic averages
- 2: Reconstruct normal momentum  $(\rho u_n)$  using third-order scheme (Eq. 14) with bias  $\eta_a$ :

$$(\rho u_n)_{i+1/2}^{L,R}$$
 via Eqs. (42)-(43)

- 3: Compute Roe-averaged wave speed:  $\lambda_{\text{max}} = |\tilde{u}_n| + \tilde{c}$
- 4: Selective dissipation:

$$\hat{D}_{n\text{-mom}} = -\frac{1}{2}\lambda_{\max} \big[ (\rho u_n)^R - (\rho u_n)^L \big], \qquad \hat{D}_k = 0 \ \text{ for all other } k$$

5:  $\hat{\mathbf{F}}_{i+1/2} = \hat{\mathbf{F}}_{i+1/2}^{\text{KEP}}; \quad \hat{F}_{n\text{-mom}} += \hat{D}_{n\text{-mom}}$ 

#### 7. Results

This section validates the proposed schemes across various test cases spanning smooth turbulence, shear instabilities, and shocked flows. The following schemes are compared: **WA-3** (third-order MUSCL, full characteristic,  $\eta_a^* = 0.54$ ), **WA-5** (fifth-order MP5, full characteristic,  $\eta_a^* = 0.6010$ ), and **WA-CR** (fifth-order MP5, conservative reconstruction with rank-1 entropy wave correction,  $\eta_a^* = 0.6010$ ). For selected cases, **WA-WENO-CR** (fifth-order WENO, conservative reconstruction with rank-1 entropy wave correction,  $\eta_a^* = 0.6010$ ) is also included. The **WA-KEP** scheme (Section 6) is included in the double periodic shear layer test to demonstrate the generality of the wave-appropriate acoustic dissipation principle beyond reconstruction-based schemes. Where relevant, the unoptimized baseline WA-5 at  $\eta_a = 1$  and the fifth-order TENO5 scheme [20] are included for reference. All simulations use the HLLC approximate Riemann solver [27] and the third-order TVD Runge–Kutta scheme [41] at CFL = 0.4, unless otherwise stated.

#### 7.1. Inviscid Taylor-Green vortex

The three-dimensional inviscid Taylor-Green vortex (TGV) serves both as the calibration benchmark for the optimization of  $\eta_a$  and as the primary accuracy assessment. The initial conditions are

$$\begin{pmatrix} \rho \\ u \\ v \\ w \\ p \end{pmatrix} = \begin{pmatrix} 1 \\ \sin x \cos y \cos z \\ -\cos x \sin y \cos z \\ 0 \\ 100 + \frac{(\cos(2z) + 2)(\cos(2x) + \cos(2y)) - 2}{16} \end{pmatrix}. \tag{46}$$

on the triply periodic domain  $[0,2\pi)^3$  with  $\gamma=5/3$ . The subsonic variant (Ma = 0.1) is used to assess accuracy. All simulations are conducted on a 64<sup>3</sup> grid until t=10. Figure 4a shows the kinetic energy decay for the third-order schemes. WA-3 at full upwinding ( $\eta_a=1$ ) is excessively dissipative, decaying significantly faster than the reference U-5. WA-3 at  $\eta_a=0.5$  (central) is unstable and diverges at  $t\approx 5$ , demonstrating the sharp stability cliff identified by the optimizer. WA-3 at  $\eta_a^*=0.54$  closely matches the U-5 reference throughout, confirming that the optimized third-order scheme matches the standard linear fifth-order scheme at no additional computational cost. Figure 4a also shows the result obtained by the KEP scheme, where the kinetic energy remains constant, confirming that the scheme has been correctly implemented.

![](_page_18_Figure_1.jpeg)

Figure 4: Subsonic inviscid Taylor-Green vortex (Ma = 0.1,  $p_0 = 100$ ,  $64^3$  grid, Sec. 7.1): time evolution of volume-averaged kinetic energy for the (a) third-order and (b) fifth-order schemes, and (c) the kinetic energy spectrum at t = 10.

Three observations from the third-order results are worth noting. First, the instability at  $\eta_a=0.5$  is driven entirely by the acoustic waves: the entropy wave in WA-3 is always reconstructed by the nonlinear MUSCL limiter regardless of  $\eta_a$ , which is among the most dissipative reconstruction strategies available. Despite this, setting  $\eta_a=0.5$  causes the scheme to crash, confirming that the acoustic upwind bias is the sole mechanism controlling stability in smooth flows and that the entropy wave plays no role here. Second, the existence of a sharp stability boundary at  $\eta_a\approx 0.5$  justifies the use of a scalar optimization: the feasible region has a clean interior and a well-defined boundary, making Brent's method both appropriate and efficient. Finally, the Ducros sensor detects no shocks throughout this case, as the  $\eta_a=0.5$  instability confirms: if the sensor had activated and switched to  $\eta_a=1$ , the scheme would not have crashed. This

validates both the sensor formulation and the Ω<sup>d</sup> > 0.01 threshold.

Figure [4b](#page-18-0) shows the corresponding results for the fifth-order schemes. WA-5 and WA-CR at η ∗ <sup>a</sup> = 0.6010 both overlap the U-7 reference to plotting accuracy. WA-CR matches WA-5 exactly since the flow is shockfree and contact-free throughout: the conservative reconstruction path is always active, and the rank-1 correction is never triggered. This confirms that the conservative variable path introduces no accuracy penalty in smooth flows.

Figure [4c](#page-18-0) shows the kinetic energy spectra at t = 10. All optimized schemes follow the Kolmogorov k −5/3 scaling over the resolved inertial range. The spectra of WA-3 and WA-5 both extend further than their respective linear references U-5 and U-7 at high wavenumbers, confirming that the optimized nonlinear Nthorder schemes resolve small-scale structures at least as well as the standard linear (N +2)th-order schemes they were calibrated against. The KEP spectrum is omitted from Figure [4c](#page-18-0) as it exhibits pronounced energy pile-up near the Nyquist scale, consistent with the known limitation identified by Ghate and Lele [\[14\]](#page-39-2), and its inclusion would obscure the comparison among the wave-appropriate schemes.

#### 7.1.1. Viscous Taylor-Green vortex

The three-dimensional viscous Taylor-Green vortex at Re = 1600 and Ma = 0.1 is a standard benchmark for implicit LES schemes, providing a DNS reference [\[42\]](#page-40-12) against which the dissipation rate ϵ = −dEk/dt can be compared directly. The initial conditions are identical to Eq. [\(46\)](#page-17-3). Results are presented on both a coarse (64<sup>3</sup> ) and a refined (96<sup>3</sup> ) grid.

Figures [5a](#page-19-0) and [5b](#page-19-0) show the dissipation rate for the third-order schemes. On the coarse grid, WA-3 and U-5 both overpredict the dissipation rate and miss the peak, which is a known limitation of third-order accuracy at this resolution. On the refined 96<sup>3</sup> grid, both schemes improve considerably and approach the DNS profile, with WA-3 and U-5 producing nearly identical results, consistent with the inviscid TGV finding that WA-3 matches its (N + 2)th-order linear reference.

![](_page_19_Figure_6.jpeg)

![](_page_19_Figure_7.jpeg)

Figure 5: Viscous Taylor-Green vortex (Re = 1600, Ma = 0.1, Sec. [7.1.1\)](#page-19-1): time evolution of the volume-averaged dissipation rate ϵ on 64<sup>3</sup> and 96<sup>3</sup> grids for WA-3 scheme. DNS reference from [\[42\]](#page-40-12).

Figures [6a](#page-20-0) and [6b](#page-20-0) show the fifth-order results. On both grids, WA-5 and WA-CR overlap in plotting accuracy and closely follow the DNS profile through the dissipation peak at t ≈ 9. The linear U-7 scheme is slightly more dissipative on the coarse grid but agrees well with DNS on the refined grid. The KEP scheme [\[25\]](#page-39-13), a second-order finite-volume formulation, matches the DNS dissipation rate well until t ≈ 5 but deviates at later times on both grids. The KEP scheme is included in the comparison precisely because it represents the limiting case of zero acoustic dissipation by construction, and its competitive performance here suggests that zero acoustic dissipation is not inherently problematic in smooth shock-free flows. The wave-appropriate schemes are competitive with KEP in this setting and remain applicable to flows with shocks and contact discontinuities, making them more general in scope. The periodic shear layer test in Section [7.2](#page-22-0) demonstrates where the fully central scheme approach breaks down.

![](_page_20_Figure_1.jpeg)

accuracy otherwise lost to unnecessary upwinding.

![](_page_20_Figure_2.jpeg)

Figure 6: Viscous Taylor-Green vortex (Re = 1600, Ma = 0.1, Sec. [7.1.1\)](#page-19-1): time evolution of the volume-averaged dissipation rate ϵ on 64<sup>3</sup> and 96<sup>3</sup> grids for WA-5 scheme. DNS reference from [\[42\]](#page-40-12).

Figure [7a](#page-21-0) shows a broader comparison on the 64<sup>3</sup> grid including WA-WENO-CR, Feng et al. [\[20\]](#page-39-8)

(TENO5DV), and ALDM (Adaptive Local Deconvolution Method) [\[43\]](#page-40-13). WA-5, WA-CR, and WA-WENO-CR produce nearly identical results and outperform both TENO5DV and ALDM. The agreement between WA-CR and WA-WENO-CR confirms that the rank-1 correction is limiter-agnostic: replacing MP5 with WENO does not affect the dissipation rate for discontinuity-free cases, and both use the same η ∗ <sup>a</sup> = 0.6010. Figure [7b](#page-21-0) is taken from Feng et al. [\[20\]](#page-39-8) and Figure [7c](#page-21-0) from Liang and Fu [\[44\]](#page-40-14). Their results provide a useful basis for comparison to understand the current approach. Liang and Fu [\[44\]](#page-40-14) and Feng et al. both use TENO8 for comparison, but their results differ greatly. Feng et al. use a finite-volume framework, where the scheme is formally only second-order accurate. This is because they do not use high-order Gaussian quadrature for flux integration, even though high-order reconstruction is used. In contrast, Liang and Fu employ a genuinely high-order finite-difference formulation of TENO8 that computes fluxes directly. When comparing the two figures, the formally high-order finite-difference scheme is far more dissipative than the finite-volume approach on this grid. Further, the WENO-CU6 and TENO6 results in Figure [7c](#page-21-0) show substantial departure from the DNS, while even WA-3 remains much closer. This confirms a well-known result: on coarse grids, dissipation level matters more than formal order of accuracy. The current fifth-order wave-appropriate schemes are competitive with both TENO8 variants. The reason is the same in both cases: the optimized bias η ∗ a reduces acoustic dissipation to the minimum required for stability, thereby restoring

![](_page_21_Figure_0.jpeg)

Figure 7: Viscous Taylor-Green vortex (Re = 1600, Ma = 0.1,  $64^3$  grid, Sec. 7.1.1): time evolution of volume-averaged kinetic energy dissipation rate  $\epsilon(t)$ . WA-3 and WA-5 use the optimized bias  $\eta_a^*$ . WA-CR and WA-WENO-CR overlap WA-5 to plotting accuracy, confirming that the rank-1 correction introduces no accuracy penalty in smooth flows. DNS reference from Brachet et al. [42]. Figure 7c is reproduced with permission from Elsevier, License number 6226831189460.

In implicit large-eddy simulation (ILES), the numerical truncation error acts as the subgrid-scale model [45]. In the wave-appropriate framework, as the results suggest,  $\eta_a$  is the only tunable dissipation mechanism in smooth regions. Non-acoustic waves are centralized. The Ducros sensor and MP limiting (or WENO) contribute only near discontinuities. The optimized value  $\eta_a^*$  marks the least dissipative implicit subgrid model the scheme can support without instability.

#### 7.1.2. Supersonic viscous Taylor-Green vortex (Ma = 1.25)

The supersonic viscous TGV at Ma = 1.25 and Re = 1600 tests the scheme in a compressible turbulent regime. Here, acoustic effects are significant. Unlike the subsonic case, the volume-averaged kinetic energy initially increases above its initial value. This increase is due to the conversion of acoustic energy before viscous dissipation takes over [38]. Simulations use  $64^3$  and  $128^3$  grids and are compared against the DNS reference [38].

Figure 8 shows the kinetic energy evolution at both resolutions. On the 64<sup>3</sup> grid, WA-5 and WA-CR overlap to plotting accuracy. WA-3 decays slightly faster at late times due to higher background dissipation. On the 128<sup>3</sup> grid, all three schemes converge and give nearly identical results. Both the initial kinetic energy overshoot and the subsequent decay closely match the DNS profile. The agreement between WA-CR and WA-5 on both grids confirms that the conservative reconstruction path introduces no accuracy penalty in compressible turbulent flows.

![](_page_22_Figure_1.jpeg)

![](_page_22_Figure_2.jpeg)

Figure 8: Supersonic viscous Taylor-Green vortex (Ma = 1.25, Re = 1600, Sec. 7.1.2): time evolution of volume-averaged kinetic energy at  $64^3$  and  $128^3$ . The initial increase above unity reflects acoustic-to-kinetic energy conversion at supersonic Mach number. WA-CR overlaps WA-5 on both grids. All schemes converge toward the DNS reference with mesh refinement.

#### 7.2. Double periodic shear layer

The double periodic shear layer serves as the counterpart to the inviscid Taylor-Green vortex in understanding the effects of acoustic wave dissipation. The inviscid TGV demonstrated that the KEP scheme with zero dissipation can perform adequately in smooth homogeneous turbulence; the shear layer demonstrates that the same approach is unstable when vorticity gradients are sharp and sustained. Schranner et al. [16] noted that central discretization requires artificial viscosity for stability while upwind discretization is excessively dissipative, identifying the challenge of devising a scheme that simultaneously provides high wavenumber resolution and sufficient dissipation to prevent vortical instability. The wave-appropriate framework resolves this by restricting upwinding to acoustic characteristic waves, leaving the vortical waves discretized centrally. The initial conditions [46] for this case over  $[0,1]^2$  are

$$\rho = 1, \quad p = \frac{1}{\gamma \operatorname{Ma}^{2}}, \quad u = \begin{cases} \tanh[\theta(y - 0.25)], & y \le 0.5, \\ \tanh[\theta(0.75 - y)], & y > 0.5, \end{cases}$$
(47a)

$$v = 0.05\sin[2\pi(x + 0.25)],$$
 (47b)

with Ma = 0.1,  $\gamma$  = 1.4, and shear layer width  $\theta$  = 80. The flow is inviscid. A reference solution is computed on a 1600 × 1600 grid, and test simulations use a 320 × 320 grid. Since this test has no shocks or contact discontinuities, the Ducros sensor is never triggered, and all numerical dissipation is attributable solely to the acoustic upwind bias  $\eta_a$ , making it a clean diagnostic for the effect of  $\eta_a$  on vortical structures. Importantly,  $\eta_a^*$  was determined exclusively from the Taylor-Green vortex suite and was not tuned for the shear layer in any way. The fact that it prevents spurious vortex generation on this qualitatively different flow suggests that  $\eta_a^*$  is intrinsic to the scheme.

Figure [9](#page-23-0) presents the reference z-vorticity and compares three baseline variants on the coarse grid to clarify the specific behaviors of each scheme. The z-vorticity field is calculated using spectral derivatives via the fast Fourier transform, eliminating numerical differentiation error from the vorticity calculation regardless of the flow solver's order. When using the KEP scheme, which conserves energy exactly, spurious braid vortices do appear, underscoring the need for a controlled dissipation to suppress vortices and that energy conservation alone is insufficient. With WA-3 at η<sup>a</sup> = 0.5 (fully central acoustics), also produces oscillations characteristic of those discussed in Section [1.](#page-0-0) In contrast, WA-3 at η<sup>a</sup> = 1.0 is overly dissipative but effectively eliminates both spurious vortices and oscillations.

![](_page_23_Figure_1.jpeg)

Figure 9: Double periodic shear layer (θ = 80, inviscid, t = 1, 320 × 320 grid, Sec. [7.2\)](#page-22-0): z-vorticity contours for the reference solution and three baseline variants. The KEP scheme and WA-3 at ηa = 0.5 are both unstable with oscillations; WA-3 at full upwinding (ηa = 1) is stable but overly dissipative.

Figure [10](#page-24-0) shows the results for the optimized schemes. WA-3 at η ∗ <sup>a</sup> = 0.54 and WA-5 at η ∗ <sup>a</sup> = 0.6010 both reproduce the two clean primary vortices without spurious structures on the 320 × 320 grid. WA-CR matches WA-5 to plotting accuracy, as expected for a shock-free contact-free flow where the conservative path is always active and the rank-1 correction has no effect. WA-WENO-CR also produces clean results, consistent with the limiter-agnostic property of the rank-1 correction. TENO5 produces spurious braid vortices on the 320 × 320 grid, requiring a finer grid to obtain an acceptable result.

![](_page_24_Figure_0.jpeg)

Figure 10: Double periodic shear layer ( $\theta=80$ , inviscid, t=1,  $320\times320$  grid, Sec. 7.2): z-vorticity contours for the optimized schemes. WA-3 at  $\eta_a^*=0.54$  and WA-5 at  $\eta_a^*=0.6010$  both reproduce two clean primary vortices. WA-CR and WA-WENO-CR match WA-5. TENO5 produces spurious braid vortices on this grid.

The WA-KEP scheme, formulated in Section 6, is tested here on the double periodic shear layer. The unmodified KEP scheme produces spurious braid vortices ( $\omega_{z,\text{max}} \approx 192.9$  versus a reference of 81) because there is no dissipation for any of the fluxes or waves. Figure 11 shows the z-vorticity contours at t=1 on the 320 × 320 grid for  $\eta_a=0.56$  and  $\eta_a=1.0$ . Introducing dissipation ( $\eta_a=0.56$ ) exclusively through the normal momentum flux recovers the two clean primary vortices without spurious structures; the maximum z-vorticity is 77.7, within 4% of the reference. At  $\eta_a=1.0$ , it is 72.5, and the shear layers are visibly diffused due to excessive dissipation. Figure 11c, adapted from [20], shows their results on a 512 × 512 grid, which is 2.5 times finer than the grid used here. Despite the higher resolution and the optimized TENO8 scheme, spurious vortices are still visible. The present schemes eliminate these structures on the coarser 320 × 320 grid, demonstrating that physics-based wave-appropriate dissipation is more effective than increasing resolution or reconstruction order alone.

![](_page_25_Figure_0.jpeg)

(c) Figure is taken from Reference [20], where the simulations are computed on a grid size of  $512^2$ .

Figure 11: Double periodic shear layer ( $\theta = 80$ , inviscid, t = 1,  $320 \times 320$  grid, Sec. 7.2): z-vorticity contours for the WA-KEP approach and that of Feng et al. [20].

The value  $\eta_a = 0.56$  is slightly above the WA-3 threshold of  $\eta_a^* = 0.54$ . This is expected: WA-3 applies either a third-order upwind or fourth-order central scheme to the flow variables, whereas KEP is purely second-order central apart from the normal momentum flux. In WA-KEP, the acoustic bias is the sole source of dissipation in the entire scheme, so it must compensate for the complete absence of dissipation on all other wave families. The upward shift from 0.54 to 0.56 is consistent with this reasoning.

- First, the acoustic dissipation enters through a completely different mechanism than in WA-3 or WA-5. In the wave-appropriate reconstruction framework, the bias is applied during the reconstruction step by blending leftand right-biased polynomials in characteristic space. In WA-KEP, the reconstruction is first-order piecewise constant, and the bias is instead applied through the Rusanov dissipative flux acting on the normal momentum alone. Despite this difference in mechanism, the stabilizing effect is identical: a controlled acoustic upwind bias suppresses the vortical instability.
- Second, the fact that  $\eta_a \approx 0.56$  stabilizes a scheme with zero background dissipation on any wave family, while  $\eta_a^* = 0.54$  suffices for WA-3 where the entropy wave has MUSCL dissipation and  $\eta_a^* = 0.6010$  suffices for WA-5 where the higher order reduces intrinsic dissipation, provides further evidence that the acoustic stability boundary is governed by the interaction between acoustic upwinding and the base scheme's dissipation level. All three values lie in the range  $\eta_a \in [0.54, 0.6010]$ , confirming that a small acoustic upwind bias above the central value of 0.5 is both necessary and sufficient for stability across fundamentally different discretization strategies.

Table 3 reports the maximum z-vorticity at t = 1 on the  $320 \times 320$  grid. The reference value computed on the  $1600 \times 1600$  grid is  $\omega_{z,\text{max}} = 81$ . Physically acceptable results should reproduce this value within a reasonable tolerance; values significantly above the reference indicate spurious vorticity amplification due to numerical oscillations rather than physical roll-up.

Table 3: Maximum z-vorticity at t = 1 for the double periodic shear layer (θ = 80, 320×320 grid). Reference value: ωz,max = 81 (1600 × 1600 grid).

| Scheme                  | ωz,max     |
|-------------------------|------------|
| Reference (16002<br>)   | 81         |
| WA-3                    | 75.1       |
| WA-5                    | 76.1       |
| WA-CR                   | 76.1       |
| TENO5                   | ≈<br>101.8 |
| KEP                     | ≈<br>192.9 |
| WA-3,<br>ηa<br>= 0.5    | ≈<br>258.4 |
| WA-KEP,<br>ηa<br>= 0.56 | 77.7       |

WA-3, WA-5, and WA-CR all produce values within 7% of the reference, confirming that the optimized acoustic bias yields the correct dissipation level for this flow. The cases of insufficient dissipation, KEP and WA-3 at η<sup>a</sup> = 0.5, produce values far above the reference due to unrestrained growth of numerical oscillations at the shear interface. The elevated value for TENO5 (≈ 101.8) has a different origin: rather than a global lack of upwinding, the TENO smoothness indicators may misidentify the sharp vorticity gradients at the shear interface as near-discontinuous, triggering low-order stencil selection and generating spurious small-scale oscillations along the interface. These oscillations introduce spurious vorticity on top of the physical roll-up, elevating the maximum above the reference level. The wave-appropriate schemes avoid this by design: the shear/vortical waves are always reconstructed with the central scheme in smooth regions, and the Ducros sensor correctly identifies the shear layer as shock-free and keeps the full characteristic path inactive. WA-KEP at η<sup>a</sup> = 0.56 also produces a maximum vorticity within 4% of the fine-grid reference, consistent with the wave-appropriate schemes.

#### 7.3. Rayleigh-Taylor instability

The Rayleigh-Taylor instability develops when a heavy fluid is accelerated into a lighter one, producing a hierarchy of mushroom-cap structures whose fine-scale detail is highly sensitive to numerical dissipation. It is a particularly challenging test for the rank-1 entropy wave correction because the interface between the two fluids is a pure contact discontinuity: density jumps sharply while velocity and pressure remain continuous, which is precisely the configuration that a conservative reconstruction without entropy wave correction would mishandle. The Ducros sensor remains inactive throughout, since there are no shocks (the pressure-based Ducros sensor cannot activate at a contact discontinuity because pressure remains continuous there [\[1\]](#page-38-0)), so all interface treatment is performed solely by the rank-1 correction. The initial conditions on [0, 4 ] × [0, 1] are

$$(\rho, u, v, p) = \begin{cases} (2, 0, -0.025c\cos(8\pi x), 2y + 1), & 0 \le y < 0.5, \\ (1, 0, -0.025c\cos(8\pi x), y + 1.5), & y \ge 0.5, \end{cases}$$

$$(48)$$

where c = p γp/ρ and γ = 1.4, with appropriate source terms added to the momentum and energy equations. Boundary conditions are (ρ, p, u, v) = (1, 2.5, 0, 0) at the top and (2, 1, 0, 0) at the bottom. Figure [12](#page-27-0) shows density contours at t = 1.95 on the 128 × 512 and 512 × 2048 grids. Several observations follow from the results:

![](_page_27_Figure_0.jpeg)

Figure 12: Rayleigh-Taylor instability (t = 1.95, Sec. [7.3\)](#page-26-1): density contours at 128 × 512 (a,b) and 512 × 2048 (c,d,e). WA-CR matches WA-5 at both resolutions. WA-WENO-CR matches WA-CR on the fine grid.

On the coarse 128 × 512 grid, WA-5 and WA-CR produce nearly identical density contours. The mushroom cap structure, rolled-up vortex sheets, and interface symmetry are preserved equally well. This result confirms that the rank-1 correction replicates the accuracy of the full characteristic decomposition at the material interface—but at lower cost. On the fine 512 × 2048 grid, both WA-5 and WA-CR fully resolve secondary instabilities and fine-scale filaments along the stem. Spurious density oscillations do not appear at the interface. This indicates that the rank-1 entropy wave correction works as intended. Purely conservative reconstructions without this correction would introduce density overshoots at the contact; these are not present. WA-WENO-CR on the fine grid also shows no oscillations. This confirms the rank-1 correction is independent of the limiter. The quality of the interface representation depends on the MP5 or WENO limiter applied to the scalar entropy wave stencil.

Figure [13](#page-28-0) shows density contours at t = 1.95, comparing the present WA-CR scheme at 1024 × 4096 resolution to reference results from Fleischmann et al. [\[47\]](#page-40-17) on the same configuration using WENO5-JS, TENO5, WENOCU6, and WENO9. The WA-CR solution preserves the instability's symmetry along the vertical centerline throughout the simulation. The mushroom cap geometry, rolled-up vortex sheets, and secondary instabilities all appear symmetrically, with no signs of symmetry-breaking due to numerical noise. No spurious density oscillations occur at the material interface, confirming that the rank-1 entropy wave correction properly limits the contact discontinuity without affecting the smooth flow. At 1024 × 4096 resolution, finer secondary instabilities and filaments are fully resolved, with detail comparable to the reference schemes from Fleischmann et al.

![](_page_28_Figure_0.jpeg)

Figure 13: Rayleigh-Taylor instability (t = 1.95, Sec. [7.3\)](#page-26-1): density contours. Left: reference results from Fleischmann et al. [\[47\]](#page-40-17) on the same configuration. Right: WA-CR at 1024 × 4096 resolution. The present scheme resolves finer secondary instabilities and filaments along the stem while remaining free of spurious density oscillations at the interface.

#### 7.4. Explosion problem

In this example, the initial condition consists of two constant states for the flow variables, a circular region of radius r = 0.4 centered at (1, 1) and the region outside of it as mentioned in Toro [\[48\]](#page-40-18). The initial conditions are given as:

$$(\rho, u, v, p) = \begin{cases} (1, 0, 0, 1), & \text{if } (x - 1)^2 + (y - 1)^2 < r^2, \\ (0.125, 0, 0, 0.1), & \text{otherwise.} \end{cases}$$
(49)

In the present case, numerical simulations are carried out over a square domain of size [0, 2] × [0, 2] until a final time t = 0.25 on a uniform grid of resolution 400 × 400. Figure [14](#page-29-0) shows density contours and the cross-sectional density profile along y = 0 for all four schemes. Each scheme maintains circular symmetry and resolves the shock and contact discontinuity cleanly without oscillations. The cross-sectional profile shows sharp capture of the shock and contact in all cases. WA-5, WA-CR, and WA-WENO-CR yield nearly identical results, while WA-3 offers slightly less resolution at the contact, as expected from its lower order.

![](_page_29_Figure_0.jpeg)

Figure 14: Explosion problem (400 × 400 grid, t = 0.25, Sec. [7.4\)](#page-28-1): density contours with the cross-sectional density profile along y = 0 shown in blue. The shock and contact discontinuity are captured cleanly by all schemes.

### 7.5. 2-D shock-entropy wave interaction

The two-dimensional shock-entropy wave interaction of Acker et al. [\[49\]](#page-41-0) tests the ability of a scheme to resolve fine-scale entropy waves. This occurs in the presence of a strong moving shock. The initial conditions are

$$(\rho, u, p) = \begin{cases} (3.857143, \ 2.629369, \ 10.3333), & x < -4, \\ (1 + 0.2\sin(10x\cos\theta + 10y\sin\theta), \ 0, \ 1), & \text{otherwise,} \end{cases}$$
 (50)

with θ = π/6 over the domain [−5, 5] × [−1, 1]. The initial sine waves make an angle of θ with the xaxis. Following Deng et al. [\[50\]](#page-41-1), a higher frequency is used for the initial sine waves compared to the original formulation to provide a more demanding test of small-scale resolution. A mesh of 400 × 80 (∆x = ∆y = 1/40) is used.

Figure [15](#page-30-0) shows the density contours and local density profile at t = 1.8. The shock is captured cleanly on the right side of the domain. Behind the shock, the entropy wave structures are well resolved by WA-CR, with the local density profile closely following the exact solution across the full range of oscillations. This case exercises both paths of the WA-CR algorithm simultaneously: the Ducros sensor activates the full characteristic reconstruction in the vicinity of the shock, while the conservative reconstruction with rank-1 entropy correction handles the smooth entropy wave region. The clean resolution of the entropy wave structures confirms that the rank-1 correction applies the appropriate limiting to the entropy characteristic without contaminating the smooth flow away from the shock.

![](_page_30_Figure_1.jpeg)

Figure 15: 2-D shock-entropy wave interaction (t = 1.8, 400 × 80 grid, Sec. [7.5\)](#page-29-1): density contours and local density profile along y = 0 for WA-CR. The exact solution is shown for reference in (b).

# 7.6. Double Mach reflection

The double Mach reflection [\[51\]](#page-41-2) involves a Mach 10 shock impinging on a 30◦ wedge, testing both shockcapturing fidelity and the resolution of slip-layer vortices formed behind the Mach stem. For this scenario, the domain [0, 3] × [0, 1] is discretized on a 768 × 256 grid, and the simulation runs to t = 0.3. The initial conditions for this test case are as follows:

$$(\rho, u, v, p) = \begin{cases} (1.4, 0, 0, 1), & y < 1.732(x - 0.1667), \\ (8, 7.145, -4.125, 116.8), & \text{otherwise.} \end{cases}$$
 (51)

To prevent the carbuncle instability, the HLL Riemann solver is used. Reflecting wall conditions are applied on the bottom boundary for x > 0.1667, while post-shock conditions hold for x ≤ 0.1667. The top boundary follows the exact moving-shock solution. Throughout the shock system, the Ducros sensor activates, ensuring the full characteristic reconstruction path is used in shocked regions. Conversely, the conservative path with rank-1 correction is active only in the smooth post-shock regions away from the Mach stem. As illustrated in Figure [16,](#page-31-0) which shows density contours zoomed into the Mach stem region, WA-3 resolves fewer slip-layer vortices than WA-5. This is consistent with lower formal order of WA-3 and the increased dissipation of the MUSCL limiter applied to the entropy wave. Both WA-5 and WA-WENO-CR produce sharp, well-resolved roll-up structures. Notably, WA-CR matches WA-5 in the shock region, as expected, since the Ducros sensor activates the full characteristic path there. However, the cost reduction of WA-CR relative to WA-5 is realized in the smooth regions with improvements in the slip lines.

![](_page_31_Figure_0.jpeg)

Figure 16: Double Mach reflection (Ma = 10, 768 × 256 grid, t = 0.3, Sec. [7.6\)](#page-30-1): density contours zoomed into the Mach stem region. WA-5, WA-CR, and WA-WENO-CR all resolve the slip-layer roll-up structures. WA-3 resolves fewer vortices due to its higher background dissipation.

### 7.7. Two-dimensional Riemann problem

The 2-D Riemann problem of configuration 3 [\[52\]](#page-41-3) initiates four shocks at the quadrant boundaries and develops Kelvin-Helmholtz instabilities along the resulting slip lines, making it sensitive to numerical dissipation in the shock-adjacent vortical regions. The slip lines are contact discontinuities along which density jumps while velocity and pressure remain continuous, so both the shock-capturing and rank-1 entropy wave correction paths of WA-CR and WA-WENO-CR are tested simultaneously. To clarify the approach used in this study, the computational setup is described below. The initial conditions on [0, 1.2]<sup>2</sup> are as follows:

$$(\rho, u, v, p) = \begin{cases} (1.5, 0, 0, 1.5), & x > 1, y > 1, \\ (0.5323, 1.206, 0, 0.3), & x < 1, y > 1, \\ (0.138, 1.206, 1.206, 0.029), & x < 1, y < 1, \\ (0.5323, 0, 1.206, 0.3), & x > 1, y < 1, \end{cases}$$
(52)

on a 512 × 512 grid, run to t = 1.1. Figure [17](#page-32-0) shows density contours computed by various schemes.

![](_page_32_Figure_0.jpeg)

Figure 17: Two-dimensional Riemann problem, configuration 3 (512 × 512, t = 1.1, [7.7\)](#page-31-1): density contours. Base WENO5 smears the slip-line vortices compared to that of WA-WENO-CR.

WA-3 captures the primary shock structure correctly; however, it resolves less fine-scale vortical detail along the slip lines than the fifth-order schemes. In contrast, WA-5 and WA-CR produce nearly identical results, featuring sharp and well-resolved Kelvin-Helmholtz roll-up along the slip lines with no spurious oscillations at the shocks. Notably, base WENO5 produces the least resolved result of all schemes tested: the slip-line vortices are heavily smeared, and even WA-3 resolves more fine-scale structure. This confirms that the wave-appropriate design with optimized η ∗ a is more effective than increasing the reconstruction order alone. Furthermore, WA-WENO-CR recovers the resolution of WA-5 and WA-CR, demonstrating that the rank-1 entropy correction compensates for the excess dissipation of the baseline WENO scheme. Overall, this result provides the clearest illustration in the paper of the benefit of the rank-1 correction approach.

# 7.8. Shock-bubble interaction

The shock-bubble interaction tests the combined handling of a strong shock, a material interface, and the Richtmyer-Meshkov roll-up vortices that develop at the bubble boundary. A Mach 6 shock in air interacts with a helium bubble of radius 0.15 centred at (0.25, 0) in the domain [0, 1] × [−0.5, 0.5]:

$$(\rho, u, v, p) = \begin{cases} (1.0, -3, 0, 1), & \text{pre-shocked air,} \\ (\frac{216}{41}, \frac{1645}{286} - 3, 0, \frac{251}{6}), & \text{post-shocked air,} \\ (0.138, -3, 0, 1), & \text{helium bubble.} \end{cases}$$
(53)

Simulations use a 400 × 400 grid and the HLL Riemann solver. Inflow and outflow conditions are imposed at the left and right boundaries; Neumann conditions apply elsewhere. Figure [18](#page-33-0) shows density contours at the final time, t = 0.15.

![](_page_33_Figure_3.jpeg)

Figure 18: Shock-bubble interaction (Ma = 6 shock, helium bubble, 400 × 400 grid, Sec. [7.8\)](#page-32-1): density contours at the final time. WA-WENO-CR confirms the limiter-agnostic property of the rank-1 correction.

This configuration tests the algorithm for both reconstruction paths simultaneously: the Ducros sensor activates in the shock region, where the full characteristic path is used, while the rank-1 entropy wave correction handles the helium/air material interface in the smooth post-shock region. WA-3 captures the primary shock and broad bubble deformation correctly. However, it resolves less fine-scale Richtmyer-Meshkov roll-up at the bubble boundary than the fifth-order schemes. By comparison, WA-5 and WA-CR produce results with sharp vortical structures at the interface, confirming that the rank-1 correction accurately handles the material interface without degrading shock-region accuracy. Likewise, WA-WENO-CR closely matches WA-CR, again confirming the limiter-agnostic property of the correction.

#### 7.9. Viscous shock tube

The viscous shock-tube problem of Daru and Tenaud [\[53\]](#page-41-4) involves the propagation of a Mach 2.37 shock wave and contact discontinuity that form a thin boundary layer at the bottom wall. The shock-boundary layer interaction produces a complex vortex system, a separation region, and a lambda-shaped shock pattern, making it a demanding test for schemes that must handle both viscous boundary layers and strong shocks simultaneously [\[54\]](#page-41-5). The initial conditions are

$$(\rho, u, v, p) = \begin{cases} (120, 0, 0, 120/\gamma), & 0 < x < 0.5, \\ (1.2, 0, 0, 1.2/\gamma), & 0.5 \le x < 1, \end{cases}$$

$$(54)$$

and the simulation runs to t<sup>f</sup> = 1. Two Reynolds numbers are considered: Re = 1000 on a 1280 × 640 grid and Re = 2500 on a 2500 × 1250 grid. For reference, Kundu et al. [\[55\]](#page-41-6) used 109 million cells for the Re = 2500 case; the present results are obtained on a grid nearly 35 times coarser. Figure [19](#page-34-0) shows flow-field contours and the wall-normal density profile at Re = 1000.

![](_page_34_Figure_3.jpeg)

Figure 19: Viscous shock tube (Re = 1000, 1280 × 640 grid, t = 1, Sec. [7.9\)](#page-33-1): flow-field density contours and density profile along the wall (y = 0). WA-CR matches WA-5.

WA-3 captures the primary shock structure and the lambda-shock pattern correctly, but provides less fine-scale vortical detail in the separation region compared to the fifth-order schemes. WA-5 and WA-CR yield nearly identical contours, both offering sharper vortical structures and better-resolved secondary shock features. The wall-normal density profile in panel (d) demonstrates that all three schemes match the reference solution well, with WA-5 and WA-CR overlapping and WA-3 displaying slight differences at the density peaks. This case relies heavily on the full characteristic reconstruction path due to the strong shock and lambda structure, whereas the rank-1 correction consistently addresses the contact discontinuity. Notably, the shock wave thickness differs between WA-3 and WA-5, with WA-5 producing a thinner shock, indicating lower dissipation.

Figure [20](#page-35-0) shows results at the higher Reynolds number Re = 2500. Under these conditions, the separation region contains finer vortical structures that are more sensitive to numerical dissipation. WA-3 captures the primary lambda shock and the broad separation bubble correctly, but it resolves the fine-scale Kelvin-Helmholtz roll-up in the shear layer near x = 0.8–1.0 less clearly than WA-5 or WA-CR. Both WA-5 and WA-CR resolve these structures clearly and are visually indistinguishable. This confirms that the rank-1 correction maintains full accuracy at higher Reynolds numbers, where the flow structures are finer, and the contact discontinuity is sharper. The wall-normal density profile in panel (d) shows that all three schemes closely match the reference, with WA-5 and WA-CR overlapping throughout and WA-3 exhibiting minor deviations at the density peaks.

![](_page_35_Figure_2.jpeg)

Figure 20: Viscous shock tube (Re = 2500, 2500 × 1250 grid, t = 1, Sec. [7.9\)](#page-33-1): flow-field density contours and density profile along the wall (y = 0). WA-5 and WA-CR resolve finer vortical structures in the separation region than WA-3 at this higher Reynolds number. WA-CR matches WA-5 in accuracy.

Figure [21](#page-36-0) shows the Ducros sensor fields in the stream-wise and wall-normal directions. These are presented alongside a fine-grid reference solution, 4000 × 2000. The sensor activates only in the narrow shock and shock/boundary-layer interaction regions. The bulk of the boundary layer, the separation zone, and the subsonic region are handled entirely by the conservative reconstruction path with rank-1 entropy wave correction. This localization explains the wall-time reduction reported in Table [4.](#page-37-1) The characteristic path is invoked at only a small fraction of the domain interfaces. As noted in Remark [3.3,](#page-10-0) the complete two-component formulation of Ducros et al. [\[6\]](#page-38-5) is used throughout this work:

$$\Omega_d = \underbrace{\frac{\left| -p_{i-2} + 16p_{i-1} - 30p_i + 16p_{i+1} - p_{i+2} \right|}{\left| p_{i-2} + 16p_{i-1} + 30p_i + 16p_{i+1} + p_{i+2} \right|}_{\text{pressure sensor}} \cdot \underbrace{\frac{\left( \nabla \cdot \mathbf{u} \right)^2}{\left( \nabla \cdot \mathbf{u} \right)^2 + \left| \nabla \times \mathbf{u} \right|^2}}_{\text{dilatation-to-vorticity ratio}}.$$
(55)

As discussed in Section [1,](#page-0-0) some implementations in the literature employ only the dilatation-to-vorticity component. Feng et al. [\[20\]](#page-39-8) are among those who omit the pressure-based indicator. Sciacovelli et al. [\[15\]](#page-39-3) demonstrated that the dilatation-to-vorticity ratio alone can misidentify intense vortical regions as shocked. In contrast, the combined formulation provides reliable localization of genuine shock regions, as shown in Figure [21.](#page-36-0)

![](_page_36_Figure_3.jpeg)

Figure 21: Viscous shock tube (Re = 2500), Sec. [7.9:](#page-33-1) Ducros sensor fields in the (a) stream-wise and (b) wall-normal directions, and (c) fine-grid reference solution computed with WA-CR. The sensor activates only in the shock and shock/boundary-layer interaction regions; the remainder of the domain uses the conservative reconstruction path, accounting for the cost reduction of WA-CR relative to WA-5.

#### 7.10. Computational cost summary

Table [4](#page-37-1) summarises the wall time reduction achieved by WA-CR relative to WA-5 across all test cases. Simulations are carried out on an Apple Mac Mini (M1 chip) using the Intel Fortran compiler (ifort) 2021 under Rosetta 2 emulation. All runs are single-threaded. The reported times reflect serial wall-clock performance. The savings range from 29% to 41% and correlate with the fraction of interfaces where the Ducros sensor is inactive. In these cases, the conservative variable reconstruction is used. The largest reductions occur in the shock-bubble interaction (41%) and the 2-D Riemann problem (36%), where the shock occupies only a small fraction of the domain. Here, the conservative path handles the bulk of the interfaces. The double shear layer (29%) and the double Mach reflection (31%) show smaller but still substantial savings. The supersonic Taylor-Green vortex (35%) falls between these, as shocks develop and interact throughout the domain, but occupy only a minority of interfaces at any given time. In all cases, the cost reduction comes at no accuracy penalty. This is demonstrated by the results in the preceding sections.

Table 4: Computational wall time for WA-5 and WA-CR across all test cases. The cost reduction reflects the fraction of interfaces where the conservative variable reconstruction replaces the full characteristic projection.

| Test case                         | WA-5     | WA-CR    | Cost reduction |
|-----------------------------------|----------|----------|----------------|
| Taylor-Green vortex (Ma = 1.25)   | 14,251 s | 9,237 s  | ≈<br>35%       |
| Double shear layer                | 743 s    | 531 s    | ≈<br>29%       |
| Rayleigh-Taylor instability       | 210 s    | 145 s    | ≈<br>31%       |
| Double Mach reflection            | 718 s    | 499 s    | ≈<br>31%       |
| 2-D Riemann problem               | 1,201 s  | 763 s    | ≈<br>36%       |
| Shock-bubble interaction          | 424 s    | 251 s    | ≈<br>41%       |
| Viscous shock tube (Re<br>= 1000) | 36,571 s | 24,532 s | ≈<br>34%       |

#### 8. Conclusions

This paper identifies the minimum acoustic upwind bias that yields stable, accurate results across subsonic to hypersonic flow regimes in the wave-appropriate reconstruction framework and introduces a cheaper, conservative-characteristic reconstruction algorithm with a rank-1 entropy wave correction.

- The acoustic upwind bias ηa, previously fixed at 1.0 as a conservative default [\[1,](#page-38-0) [2\]](#page-38-1), is identified as the sole free parameter in the wave-appropriate framework once all other dissipation sources are eliminated by design. A physics-constrained scalar optimization using Brent's method converges in approximately 25 CFD evaluations, far fewer than multi-parameter Bayesian frameworks, because the wave-appropriate design reduces the problem to a bounded scalar minimization.
- The optimization yields η ∗ <sup>a</sup> = 0.54 for WA-3 and η ∗ <sup>a</sup> = 0.6010 for WA-5. Below these values, the scheme is unstable across a wide range of flow regimes regardless of grid resolution. Both values transfer without retuning across the full range, from subsonic turbulence to hypersonic flows, including discontinuities. The optimized nonlinear Nth-order scheme consistently matches or outperforms the standard linear (N+2)th-order scheme at full acoustic upwinding, at no additional computational cost.
- The prior conservative-characteristic scheme [\[4\]](#page-38-3) required two sensors, a Ducros shock sensor and a density-based contact detector, to switch between reconstruction paths. It is shown that the contact detector is unnecessary. Near a contact discontinuity, the deficiency of a conservative variable reconstruction is entirely a rank-1 perturbation in the entropy characteristic amplitude C2, correctable by a scalar entropy projection, one MP5 pass, and a rank-1 update of five multiply-add operations per interface side.
- The resulting algorithm, WA-CR, reduces wall time by 29 to 41% relative to the full characteristic scheme across various benchmark configurations while matching or exceeding its accuracy. The contact is handled structurally rather than by detection, eliminating the need for a sensor that would otherwise require empirical calibration. The rank-1 correction is limiter-agnostic, as the WENO variant WA-WENO-CR also produces results without oscillations for cases with contact discontinuities.
- The wave-appropriate principle is not restricted to reconstruction-based schemes. Applied to the KEP scheme, where the base discretization provides zero dissipation for all fluxes, a controlled acoustic upwind bias of η<sup>a</sup> = 0.56 introduced exclusively through the normal-momentum component of the dissipative flux eliminated the spurious vortices observed in the unmodified KEP scheme. The mechanism is direction-dependent dissipation of normal momentum without any eigenvector projection. The value η<sup>a</sup> = 0.56 is consistent with the optimized values for reconstruction-based schemes, placing all three schemes in the narrow band η<sup>a</sup> ∈ [0.54, 0.6010] and confirming that a small acoustic upwind bias above the central value of 0.5 is both necessary and sufficient for stability across fundamentally different discretization strategies.

It is important to note that all the algorithms presented in this paper require **conservative** variable reconstruction, which was also the choice of reconstruction in [1]. The wave-appropriate framework and the rank-1 entropy correction are derived from the characteristic structure of the conservative-variable system and do not apply to primitive-variable reconstruction (see Ref. [2], where it has been shown that there will be significant differences between results obtained by conservative and primitive variable reconstruction). If pressure must be considered in place of total energy, then the recommended variable set is  $[\rho, \rho u, \rho v, p]$ ; this means the reconstruction operates directly on momentum components  $(\rho u, \rho v)$  and not on velocity components (u, v), which would be the case in primitive variable reconstruction.

WA-5, WA-CR, and WA-WENO-CR with  $\eta_a^* = 0.6010$  are the recommended schemes. The framework can be straightforwardly extended to curvilinear coordinates, and the rank-1 correction principle may apply to any scheme employing conservative-characteristic reconstruction. Future work may focus on theoretically characterizing the acoustic stability boundary, potentially deriving  $\eta_a^*$  analytically from the interaction between acoustic waves and the Kelvin-Helmholtz instability. Such an analysis might reveal that  $\eta_a^*$  should vary spatially, which could further improve results.

#### **Appendix**

A two-dimensional implementation of the solver is provided as supplementary material. It is implemented using the NVIDIA Warp [56]. The code can be executed after installing the dependency via pip install warp-lang. It reproduces the results obtained by the WA-3 scheme, shown in the figure. 17a.

#### References

- [1] A. S. Chamarthi, N. Hoffmann, S. Frankel, A wave appropriate discontinuity sensor approach for compressible flows, Physics of Fluids 35 (6) (2023).
- [2] N. Hoffmann, A. S. Chamarthi, S. H. Frankel, Centralized gradient-based reconstruction for wall modelled large eddy simulations of hypersonic boundary layer transition, Journal of Computational Physics (2024) 113128.
- [3] A. S. Chamarthi, Wave-appropriate multidimensional upwinding approach for compressible multiphase flows, Journal of Computational Physics 538 (2025) 114157.
- [4] A. S. Chamarthi, A generalized adaptive central-upwind scheme for compressible flow simulations and preventing spurious vortices, arXiv preprint arXiv:2409.02340 (2024).
- [5] A. S. Chamarthi, Physics appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows, Computers & Fluids 303 (2025) 106858.
- [6] F. Ducros, V. Ferrand, F. Nicoud, C. Weber, D. Darracq, C. Gacherieu, T. Poinsot, Large-eddy simulation of the shock/turbulence interaction, Journal of Computational Physics 152 (2) (1999) 517–549.
- [7] N. Sandham, E. Schuelein, A. Wagner, S. Willems, J. Steelant, Transitional shock-wave/boundary-layer interactions in hypersonic flow, Journal of Fluid Mechanics 752 (2014) 1–33.
- [8] G. K. Batchelor, An introduction to fluid dynamics, Cambridge university press, 1967.
- [9] J. C. Meng, T. Colonius, Numerical simulation of the aerobreakup of a water droplet, Journal of Fluid Mechanics 835 (2018) 1108–1135.
- [10] F. De Vanna, F. Picano, E. Benini, A sharp-interface immersed boundary method for moving objects in compressible viscous flows, Computers & Fluids 201 (2020) 104415.
- [11] P. K. Subbareddy, G. V. Candler, A fully discrete, kinetic energy consistent finite-volume scheme for compressible flows, Journal of Computational Physics 228 (5) (2009) 1347–1364.

- [12] W. van Noordt, S. Ganju, C. Brehm, An immersed boundary method for wall-modeled large-eddy simulation of turbulent high-mach-number flows, Journal of Computational Physics 470 (2022) 111583.
- [13] F. De Vanna, F. Avanzi, M. Cogo, S. Sandrin, M. Bettencourt, F. Picano, E. Benini, Uranos: A gpu accelerated navier-stokes solver for compressible wall-bounded flows, Computer Physics Communications 287 (2023) 108717.
- [14] A. Ghate, S. K. Lele, Finite difference methods for turbulence simulations, in: Numerical Methods in Turbulence Simulation, Elsevier, 2023, pp. 235–284.
- [15] L. Sciacovelli, D. Passiatore, P. Cinnella, G. Pascazio, Assessment of a high-order shock-capturing central-difference scheme for hypersonic turbulent flow simulations, Computers & Fluids 230 (2021) 105134.
- [16] F. S. Schranner, X. Y. Hu, N. A. Adams, A physically consistent weakly compressible high-resolution approach to underresolved simulations of incompressible flows, Computers & Fluids 86 (2013) 109–124.
- [17] F. S. Schranner, V. Rozov, N. A. Adams, Optimization of an implicit large-eddy simulation method for underresolved incompressible flow simulations, AIAA Journal 54 (5) (2016) 1567–1577.
- [18] J. M. Winter, F. S. Schranner, N. A. Adams, Iterative bayesian optimization of an implicit les method for under-resolved simulations of incompressible flows, in: en. In: 10th International Symposium on Turbulence and Shear Flow Phenomena, TSFP, Vol. 10, 2016.
- [19] Y. Feng, F. S. Schranner, J. Winter, N. A. Adams, A multi-objective bayesian optimization environment for systematic design of numerical schemes for compressible flow, Journal of Computational Physics 468 (2022) 111477.
- [20] Y. Feng, J. Winter, N. A. Adams, F. S. Schranner, A general multi-objective bayesian optimization framework for the design of hybrid schemes towards adaptive complex flow simulations, Journal of Computational Physics 510 (2024) 113088.
- [21] Y. Feng, F. S. Schranner, J. Winter, N. A. Adams, A deep reinforcement learning framework for dynamic optimization of numerical schemes for compressible flow simulations, Journal of Computational Physics 493 (2023) 112436.
- [22] D. A. Bezgin, A. B. Buhendwa, S. J. Schmidt, N. A. Adams, Ml-iles: End-to-end optimization of data-driven high-order godunov-type finite-volume schemes for compressible homogeneous isotropic turbulence, Journal of Computational Physics 522 (2025) 113560.
- [23] B. Van Leer, Upwind and high-resolution methods for compressible flow: From donor cell to residualdistribution schemes, in: 16th AIAA Computational Fluid Dynamics Conference, 2003, p. 3559.
- [24] P. L. Roe, Discrete models for the numerical analysis of time-dependent multidimensional gas dynamics, Journal of Computational Physics 63 (2) (1986) 458–476.
- [25] P. Chandrashekar, Kinetic energy preserving and entropy stable finite volume schemes for compressible euler and navier-stokes equations, Communications in Computational Physics 14 (5) (2013) 1252–1286.
- [26] H. Nishikawa, Beyond Interface Gradient: A General Principle for Constructing Diffusion Schemes, 40th Fluid Dynamics Conference and Exhibit (2010).
- [27] E. F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the hll-riemann solver, Shock waves 4 (1) (1994) 25–34.
- [28] A. Harten, P. D. Lax, B. v. Leer, On upstream differencing and godunov-type schemes for hyperbolic conservation laws, SIAM review 25 (1) (1983) 35–61.
- [29] A. S. Chamarthi, Gradient based reconstruction: Inviscid and viscous flux discretizations, shock capturing, and its application to single and multicomponent flows, Computers & Fluids 250 (2023) 105706.

- [30] A. S. Chamarthi, Efficient high-order gradient-based reconstruction for compressible flows, Journal of Computational Physics 486 (2023) 112119.
- [31] A. S. Chamarthi, N. Hoffmann, H. Nishikawa, S. H. Frankel, Implicit gradients based conservative numerical scheme for compressible flows, Journal of Scientific Computing 95 (1) (2023) 17.
- [32] C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws, in: Advanced Numerical Approximation of Nonlinear Hyperbolic Equations: Lectures given at the 2nd Session of the Centro Internazionale Matematico Estivo (CIME) held in Cetraro, Italy, June 23–28, 1997, Springer, 2006, pp. 325–432.
- [33] S. K. Godunov, A difference method for numerical calculation of discontinuous solutions of the equations of hydrodynamics, Matematicheskii Sbornik 89 (3) (1959) 271–306.
- [34] B. Van Leer, Towards the ultimate conservative difference scheme. iv. a new approach to numerical convection, Journal of Computational Physics 23 (3) (1977) 276–299.
- [35] B. Van Leer, Towards the ultimate conservative difference scheme. v. a second-order sequel to godunov's method, Journal of computational Physics 32 (1) (1979) 101–136.
- [36] A. Suresh, H. Huynh, Accurate monotonicity-preserving schemes with runge-kutta time stepping, Journal of Computational Physics 136 (1) (1997) 83–99.
- [37] S. Pirozzoli, On the spectral properties of shock-capturing schemes, Journal of Computational Physics 219 (2) (2006) 489–497.
- [38] D. J. Lusher, N. D. Sandham, Assessment of low-dissipative shock-capturing schemes for the compressible taylor–green vortex, AIAA Journal 59 (2) (2021) 533–545.
- [39] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, et al., Scipy 1.0: fundamental algorithms for scientific computing in python, Nature methods 17 (3) (2020) 261–272.
- [40] R. Borges, M. Carmona, B. Costa, W. S. Don, An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws, Journal of Computational Physics 227 (6) (2008) 3191–3211.
- [41] G.-S. Jiang, C.-W. Shu, Efficient Implementation of Weighted ENO Schemes, Journal of Computational Physics 126 (126) (1995) 202–228.
- [42] M. E. Brachet, D. I. Meiron, S. A. Orszag, B. Nickel, R. H. Morf, U. Frisch, Small-scale structure of the taylor–green vortex, Journal of Fluid Mechanics 130 (1983) 411–452.
- [43] S. Hickel, N. A. Adams, J. A. Domaradzki, An adaptive local deconvolution method for implicit les, Journal of Computational Physics 213 (1) (2006) 413–436.
- [44] T. Liang, L. Fu, A new high-order shock-capturing teno scheme combined with skew-symmetric-splitting method for compressible gas dynamics and turbulence simulation, Computer Physics Communications 302 (2024) 109236.
- [45] F. F. Grinstein, L. G. Margolin, W. J. Rider, Implicit large eddy simulation, Vol. 10, Cambridge university press Cambridge, 2007.
- [46] M. L. Minion, D. L. Brown, Performance of under-resolved two-dimensional incompressible flow simulations, ii, Journal of Computational Physics 138 (2) (1997) 734–765.
- [47] N. Fleischmann, S. Adami, N. A. Adams, Numerical symmetry-preserving techniques for low-dissipation shock-capturing schemes, Computers & Fluids 189 (2019) 94–107.
- [48] E. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer Berlin Heidelberg, 2009.

- [49] F. Acker, R. d. R. Borges, B. Costa, An improved weno-z scheme, Journal of Computational Physics 313 (2016) 726–753.
- [50] X. Deng, Y. Shimizu, F. Xiao, A fifth-order shock capturing scheme with two-stage boundary variation diminishing algorithm, Journal of Computational Physics 386 (2019) 323–349.
- [51] P. Woodward, P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, Journal of Computational Physics 54 (1) (1984) 115–173.
- [52] C. W. Schulz-Rinne, J. P. Collins, H. M. Glaz, Numerical solution of the riemann problem for twodimensional gas dynamics, SIAM Journal on Scientific Computing 14 (6) (1993) 1394–1414.
- [53] V. Daru, C. Tenaud, Numerical simulation of the viscous shock tube problem by using a high resolution monotonicity-preserving scheme, Computers & Fluids 38 (3) (2009) 664–676.
- [54] A. S. Chamarthi, S. Bokor, S. H. Frankel, On the importance of high-frequency damping in highorder conservative finite-difference schemes for viscous fluxes, Journal of Computational Physics (2022) 111195.
- [55] A. Kundu, M. Thangadurai, G. Biswas, Investigation on shear layer instabilities and generation of vortices during shock wave and boundary layer interaction, Computers & Fluids 224 (2021) 104966.
- [56] M. Macklin, Warp: A high-performance python framework for gpu simulation and graphics, in: NVIDIA GPU Technology Conference (GTC), Vol. 3, 2022.