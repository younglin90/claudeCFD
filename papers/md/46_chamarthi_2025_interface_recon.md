Contents lists available at [ScienceDirect](https://www.elsevier.com/locate/compfluid)

# Computers and Fluids

journal homepage: [www.elsevier.com/locate/compfluid](https://www.elsevier.com/locate/compfluid)

![](_page_0_Picture_5.jpeg)

![](_page_0_Picture_6.jpeg)

# Physics appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows

Amareshwara Sainadh Chamarthi

*Division of Engineering and Applied Science, California Institute of Technology, Pasadena, CA, USA*

## A R T I C L E I N F O

*Keywords:* THINC Multicomponent flows Interface-capturing Tangential velocities Viscous flows

### A B S T R A C T

The paper proposes a physically consistent numerical discretization approach for simulating viscous compressible multicomponent flows. It has two main contributions. First, a contact discontinuity (and material interface) detector is developed. In those regions of contact discontinuities, the THINC (Tangent of Hyperbola for INterface Capturing) approach is used for reconstructing appropriate variables (phasic densities). For other flow regions, the variables are reconstructed using the Monotonicity-preserving (MP) scheme (or Weighted essentially non-oscillatory scheme (WENO)). For reconstruction in the characteristic space, the THINC approach is used only for the contact (or entropy) wave and volume fractions. For the reconstruction of primitive variables, the THINC approach is used for phasic densities and volume fractions only, offering an effective solution for reducing dissipation errors near contact discontinuities. The numerical results of the benchmark tests show that the proposed method captured the material interface sharply compared to existing methods. The second contribution is the development of an algorithm that uses a central reconstruction scheme for the tangential velocities, as they are continuous across material interfaces in viscous flows. In this regard, the Ducros sensor (a shock detector that cannot detect material interfaces) is employed to compute the tangential velocities using a central scheme across material interfaces. Using the central scheme does not produce any oscillations at the material interface. The proposed approach is thoroughly validated with several benchmark test cases for compressible multicomponent flows, highlighting its advantages. The physics appropriate approach also shown to prevent spurious vortices, despite being formally second-order accurate for nonlinear problems, on a coarser mesh than a genuinely high-order accurate method.

## **1. Introduction**

Compressible flows can manifest two primary types of discontinuities: shocks and contact discontinuities, the latter often representing the boundaries between different materials or phases. Density, pressure, and normal velocity are discontinuous across shocks, and only the density and volume fractions are discontinuous across a material interface [[1](#page-28-0)]. On the other hand, while tangential velocities are continuous in viscous flow simulations [\[2\]](#page-28-1) ( also when there is artificial viscosity [\[3\]](#page-28-2)), they are discontinuous across contact discontinuities in inviscid simulations [\[1\]](#page-28-0). Addressing these discontinuities with one approach has become common in numerical simulations. Various methodologies have been devised to obtain oscillation-free results for these discontinuities, including Weighted Essentially Non-Oscillatory (WENO) schemes [\[4–](#page-28-3) [6](#page-28-4)], the Monotonicity Preserving (MP) approach [[7](#page-28-5)–[9](#page-28-6)], and localized artificial dissipation methods [\[10](#page-28-7)[,11](#page-28-8)]. This study's focus is on treating these discontinuities with different approaches rather than a single approach, while considering the physics of these discontinuities. The paper has three objectives: (a) develop a contact and material interface detector such that those regions can be computed with reduced numerical dissipation, (b) use central scheme across a material interface in viscous flow simulations and show that the algorithm is free of oscillations and (c) show that the proposed algorithm that considers the physical aspects of the various variables can give superior results over a genuinely high-order method (prevents spurious vortices in periodic shear layer).

To achieve these objectives, our method is built upon the inherent wave structure of the Euler equations. In a one-dimensional scenario, this structure consists of three characteristic wave types: two acoustic waves, which travel at the local speed of sound relative to the fluid (u±c) carrying pressure and velocity changes, and one entropy wave, which travels with the fluid velocity (u) and carries variations in density and temperature. In two or more dimensions, the wave structure also includes shear or vorticity waves, which transport changes in the tangential velocity. This work leverages the above-stated physical

*E-mail address:* [sainath@caltech.edu](mailto:sainath@caltech.edu).

![](_page_1_Figure_2.jpeg)

**Fig. 1.** Test case with three gases (<sup>1</sup> , <sup>2</sup> and <sup>3</sup> , [Example](#page-21-0) [4.10\)](#page-21-0) will be tedious to check with the criteria (+1 - )( - −1) *>* 0.

knowledge, specifically how each variable behaves along these distinct waves, to develop the proposed algorithm.

Objective 1 - Contact discontinuity sensor and the corresponding work in the literature: As explained in the above paragraph, there are two types of discontinuities: shocks and contact discontinuities (material interfaces). There are many shock-capturing schemes in the literature [\[4,](#page-28-3)[7,](#page-28-5)[12–](#page-28-9)[20\]](#page-28-10) that can capture the shocks crisply and are non-oscillatory. It is well-known from the literature that contact discontinuities are captured with excessive numerical dissipation than the shocks using these schemes. The earliest works that addressed these deficiencies are those of Harten [\[21](#page-28-11)] and Huynh [\[22](#page-28-12)]. Harten proposed a subcell resolution approach such that the contact discontinuities are captured within one or two cells, whereas Huynh has proposed a slope steepening approach for the contact discontinuity. Both Harten and Huynh have applied their algorithms only to the entropy wave of the Euler equations and thereby modifying only the variable density (which is the only variable that has a jump at the contact discontinuity, [Fig.](#page-2-0) [2](#page-2-0)). Both works have demonstrated their improvements for 1D scenarios for single-component flows. Huynh's approach to detecting contact discontinuities is based on the wave strengths of characteristic waves with ad hoc parameters. Yang [\[23](#page-28-13)] further improved the approach of Harten, and the improved version was used in conjunction with the WENO scheme by Balsara and Shu [\[20](#page-28-10)].

In a more recent past, several approaches were proposed in the literature to reduce the numerical dissipation around contact discontinuities, and these algorithms also extend to multicomponent and multiphase flows (where the control of numerical diffusion of the material interface is far more important). In this regard, Shukla et al. [\[24](#page-28-14)] have proposed an interface compression approach to minimize numerical diffusion across material interfaces. Chiapolino et al. [\[25](#page-28-15)] have proposed an Overbee limiter that captured the interfaces sharply compared to the Superbee limiter. He et al. [[26\]](#page-28-16) used the approach of Yang [[23\]](#page-28-13) in multicomponent flows to capture the material interfaces sharply. On the other hand, Shyue and Xiao [[27\]](#page-28-17) proposed an interfacesharpening approach for multiphase flows using the THINC scheme. The key idea of their interface sharpening approach is to replace the standard shock-capturing schemes with the THINC scheme (a nonpolynomial reconstruction approach). Shyue and Xiao [[27\]](#page-28-17) applied THINC for any computational cell containing the material interface, which is defined as any cell satisfying the condition  *<*  ≤ 1 - , where = 10−5 and is the volume fraction of the fluid, and a monotonicity constraint (+1 - )( - −1) *>* 0. They developed a homogeneousequilibrium-consistent reconstruction scheme for interface sharpening. Inspired by their work, Garrick et al. [\[28](#page-28-18)] improved the interface capturing approach by using the THINC only for the volume fraction and phasic densities, and the other variables are reconstructed by using the standard MUSCL or WENO schemes. Zhang et al. [\[29](#page-28-19)] further improved the approach of Garrick et al. by using the THINC scheme subjected to the same criterion  *<*  <sup>≤</sup> <sup>1</sup> - and (+1 - )( - −1) *>* 0 and away from the interfaces the WENOIS approach is used. The main drawback of this approach in these studies is that the detection criterion checks for the material interfaces using volume fractions. This type of checking can become tedious if there are several species/gases in the domain, and it will also fail to detect the contact discontinuity within a material (jump in density within the material, for example, the compressible triple point test case considered in this study). A test case with three species is shown in [Fig.](#page-1-0) [1,](#page-1-0) which is also the initial condition for the test case considered in this paper.

More recently, Takagi et al. [\[30](#page-28-20)] have developed a TENO-based discontinuity sensor such that all the discontinuities are computed with the THINC scheme as opposed to that of Garrick et al. [[28\]](#page-28-18), who computed only phasic densities and volume fractions using THINC. A significant flaw in Takagi et al.'s approach is that the proposed discontinuity sensor detects both shocks and contact discontinuities, and the THINC scheme is applied to all the variables regardless of whether the variable is continuous or discontinuous. However, across the contact discontinuities, the pressure and velocity are continuous (see [Fig.](#page-2-0) [2,](#page-2-0) which represents the classic Sod test case), and applying THINC (a discontinuity capturing and interface sharpening approach) could lead to the failure of the simulation.

The key takeaways from the above discussion are:

- 1. Researchers have exploited the characteristic wave structure of the Euler equations and the reconstruction of contact waves are often used to improve the resolution of the contact discontinuities and material interfaces. [[21](#page-28-11)[,22](#page-28-12)].
- 2. The material interfaces are identified based on the volume fractions, but such an approach might not be able to detect a contact discontinuity within a material. For test cases with several species, it can be computationally intensive to check for all the material interfaces [[27–](#page-28-17)[29](#page-28-19)].
- 3. The approach of Takagi et al. can detect contact discontinuities (ignoring the fact that the THINC scheme is applied to all the variables, even if they are continuous) and probably can identify material interfaces even if there are multiple gases. However, it also detects regions that are not contact discontinuities, like high-frequency regions in the shock-entropy wave test case (discussed in [Example](#page-8-0) [4.3](#page-8-0) and in [[31\]](#page-28-21)). Applying THINC to such regions can also lead to inconsistent results. Their approach depends on the TENO scheme for interface detection and may not be generalized to other schemes.

Therefore, it is desirable to develop (a) an approach to detect contact discontinuities (that do not necessarily depend on volume fractions or TENO-based detector of Takagi et al.) and reconstruct those regions using the THINC scheme to minimize the numerical dissipation, and (b) the developed detector thus can be used with any of the existing shock-capturing schemes (MP, WENO, MUSCL, etc.). The proposed algorithm should work for both characteristic and primitive variable reconstruction while considering the appropriate physics.

**Remark 1.1.** The proposed contact or material interface detector can also detect gas-liquid interfaces, as shown in [[31\]](#page-28-21). The current paper shows the results and analysis for the gas-gas test cases only.

![](_page_2_Figure_2.jpeg)

**Fig. 2.** Pressure and velocity are continuous across a contact discontinuity and, therefore, do not require an interface sharpening approach in those regions.

Objective 2 - Using central scheme for tangential velocities: The present work is related to the wave-appropriate discontinuity detector approach presented in Ref. [[32\]](#page-28-22) to multicomponent flows. In [\[32](#page-28-22)], Chamarthi et al. used the Ducros sensor [\[33](#page-28-23)], a shock sensor for acoustic and shear waves and a density-based sensor for the entropy wave to simulate compressible turbulent flows with shocks. It is well known that the Ducros sensor cannot detect contact discontinuities, and the wave-appropriate sensor overcame the Ducros sensor's deficiency. Using only a density-based sensor can detect shocks and contact discontinuities, but it would be too dissipative for turbulent flows. Using two different discontinuity sensors, depending on the wave structure, significantly improved the numerical results. The approach in [[32\]](#page-28-22) was further improved in [[34\]](#page-28-24). In [\[34](#page-28-24)], Hoffmann, Chamarthi and Frankel made one key modification from that of the [[32\]](#page-28-22). The modification is the treatment of shear or vorticity waves. In [\[32](#page-28-22)] all the waves are computed using an upwind biased scheme, but in [[34\]](#page-28-24) the shear waves are computed using a central scheme, which leads to the accurate prediction of the transition to turbulence in hypersonic flow over a ramp. The key thought process for the modification is as follows:

*''One cannot use a central scheme all through out for all the waves as it is found to be unstable. But let us consider the physics of the wave structure or variables in the equations. We know that across a shockwave, the tangential velocity is continuous and the other variables are discontinuous. Therefore applying a central scheme for the shear wave, corresponding to the tangential velocity, could improve the results''.*

The above-mentioned modification of applying the central scheme for shear waves in [\[34](#page-28-24)] was non-oscillatory, low dissipation, stable for discontinuities and therefore was able to predict the transition to turbulence in hypersonic flows. It was also mentioned in [\[34](#page-28-24)] that the reconstruction of conservative variables was beneficial over primitive variable reconstruction. Ref. [\[32](#page-28-22)] was titled *''wave appropriate discontinuity sensor''* and similarly, the present methodology can be called *''wave or physics appropriate reconstruction approach''*, as either MP/WENO or THINC is used, depending on the characteristics of the wave structure. In the algorithm presented later, the choice of reconstruction scheme depends on the Euler equation's characteristic waves. Likewise, if the primitive variables are reconstructed, then the appropriate physics will again be considered. The objectives of Ref. [\[32](#page-28-22)[,34](#page-28-24)] were different from those of the present study; Ref. [[32,](#page-28-22)[34\]](#page-28-24) addressed the simulation of single-species hypersonic turbulent flows, and the present study targets the simulation of high-speed multicomponent flows with material interfaces. Extending the idea of using a central scheme for tangential velocities to a five-equation model, used in the current paper, with multiple gases, has faced some difficulties. Upon literature survey, the following observations were made:

• The key issue here is that across a material interface, the tangential velocity can be discontinuous [[1](#page-28-0)[,35](#page-28-25)] in inviscid scenario. But, Meng and Colonius [\[3\]](#page-28-2) wrote that in the presence of artificial viscosity, the tangential velocity can be continuous.

- Upon further investigation (*''fortunately''*), Batchelor (''An Introduction to Fluid Dynamics'') [[2](#page-28-1)] wrote that in the presence of physical viscosity, the tangential velocities are indeed continuous.
- Chamarthi has used the MP sensor for tangential velocities in [\[31](#page-28-21)] for both gas-gas and gas-liquid flows and used a central scheme. However, the sensor cannot guarantee that it always uses a central scheme, and the approach can be considered as ad hoc even though it significantly improved the results.

Therefore the second objective of the paper is to show that the tangential velocities are indeed continuous across material interface if there is physical viscosity (viscous simulations) and the best way to show that is to use shock detector (as it cannot detect contact discontinuities) for shear waves or tangential velocity and compute them using a central scheme. Such an approach also helped prevent spurious vortices in the periodic shear layer case on coarser grid points than that of the Bayesian optimization approach [[36](#page-28-26)[,37](#page-28-27)] and deep reinforcement learning approach [\[38](#page-28-28)].

Objective 3 - Physics consistency approach over high-order accuracy: In the above two objectives, the physics of the equations are used for the reconstruction process, i.e. across a material interface, pressure and velocity can be continuous, and in the presence of physical viscosity, tangential velocity can be continuous. It implies that some variables are computed using an upwind scheme, while others are computed using a central scheme. Typically in the literature, one single approach will be used for all the variables, for example, a fifth order upwind scheme. But, Chamarthi [\[31](#page-28-21)] has shown that using different reconstruction schemes for variables according to their physics will lead to better results. The idea has similarities to the multidimensional upwinding approach of Roe [[39\]](#page-28-29), and the details are presented in [\[31](#page-28-21)]. The proposed algorithm in this paper is only second-order accurate. Yet, by accounting for the physics of the variables, it can prevent the spurious vortices in the periodic shear layer case, which is not possible even for a mathematically high-order accurate approach. It will be shown that an algorithm, which is only second-order accurate, that accounts for the physics of the various variables can obtain superior results than a higher-order accurate approach.

The rest of the paper is organized as follows. The governing equations of the compressible multicomponent flows are present in Section [2](#page-2-1). Details of the numerical discretization of the equations, including the details of the physically consistent novel adaptive interface capturing scheme, are described in Section [3](#page-3-0). Several oneand two-dimensional test cases for compressible multicomponent flows are presented in Section [4,](#page-8-1) and Section [5](#page-23-0) summarizes the findings.

### **2. Governing equations**

In this study, the viscous compressible multi-component flows as described by the quasi-conservative five equation model of Allaire et al. [[40\]](#page-28-30) including viscous effects [\[5\]](#page-28-31) are considered. A system consisting of two fluids has two continuity equations, one momentum and one energy equation. In addition, an advection equation for the volume fraction of one of the two fluids is considered. The governing equations are:

$$\frac{\partial \mathbf{Q}}{\partial t} + \frac{\partial \mathbf{F}^{\mathbf{c}}}{\partial x} + \frac{\partial \mathbf{G}^{\mathbf{c}}}{\partial y} + \frac{\partial \mathbf{F}^{\mathbf{v}}}{\partial x} + \frac{\partial \mathbf{G}^{\mathbf{v}}}{\partial y} = \mathbf{S},\tag{1}$$

where the state vector, flux vectors and source term, S, are given by:

$$\mathbf{Q} = \begin{bmatrix} \alpha_{1}\rho_{1} \\ \alpha_{2}\rho_{2} \\ \rho u \\ \rho v \\ E \\ \alpha_{1} \end{bmatrix}, \quad \mathbf{F}^{\mathbf{c}} = \begin{bmatrix} \alpha_{1}\rho_{1}u \\ \alpha_{2}\rho_{2}u \\ \rho v u \\ (E+p)u \\ \alpha_{1}u \end{bmatrix},$$

$$\mathbf{G}^{\mathbf{c}} = \begin{bmatrix} \alpha_{1}\rho_{1}v \\ \alpha_{2}\rho_{2}v \\ \rho uv \\ \rho v^{2} + p \\ (E+p)v \\ \rho v v \end{bmatrix}, \quad \mathbf{S} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \alpha_{1}\nabla \cdot \mathbf{B} \end{bmatrix},$$

$$(2)$$

$$\mathbf{F}^{\mathbf{v}} = \begin{bmatrix} 0 \\ 0 \\ -\tau_{xx} \\ -\tau_{yx} \\ -\tau_{xx}u - \tau_{xy}v \end{bmatrix}, \mathbf{G}^{\mathbf{v}} = \begin{bmatrix} 0 \\ 0 \\ -\tau_{xy} \\ -\tau_{yy}u - \tau_{yy}v \\ 0 \end{bmatrix}, \tag{3}$$

where  $\rho_1$  and  $\rho_2$  correspond to the densities of fluids 1 and 2,  $\alpha_1$  and  $\alpha_2$  are the volume fractions of the fluids 1 and 2,  $\rho$ , u, v,  $\rho$  and E are the density, x– and y– velocity components, pressure, total energy per unit volume of the mixture, respectively. When employing a diffuse interface approach for mathematical modelling, the five-equation model becomes incomplete near the material interface, as the fluids exist in a mixed state. A set of mixture rules for various fluid properties must be established to close the model. These rules include the mixture rules for the volume fractions of the two fluids, denoted as  $\alpha_1$  and  $\alpha_2$ , the density, and the mixture rules for the ratio of specific heats  $(\gamma)$  of the mixture. The mixture rules are as follows:

$$\alpha_2 = 1 - \alpha_1,\tag{4}$$

$$\rho = \rho_1 \alpha_1 + \rho_2 \alpha_2,\tag{5}$$

$$\frac{1}{\gamma - 1} = \frac{\alpha_1}{\gamma_1 - 1} + \frac{\alpha_2}{\gamma_2 - 1},\tag{6}$$

where  $\gamma_1$  and  $\gamma_2$  are the specific heat ratios of fluids 1 and 2, respectively. Under the isobaric assumption, the equation of state used to close the system is as follows:

$$p = (\gamma - 1)(E - \rho \frac{(u^2 + v^2 + w^2)}{2}). \tag{7}$$

The viscous terms are given by:

$$\tau_{xx} = \frac{2}{3}\mu \left( 2\frac{\partial u}{\partial x} - \frac{\partial v}{\partial y} \right), \quad \tau_{xy} = \tau_{yx} = \mu \left( \frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right), 
\tau_{yy} = \frac{2}{3}\mu \left( 2\frac{\partial v}{\partial y} - \frac{\partial u}{\partial x} \right).$$
(8)

## 3. Numerical discretization

Eqs. (1) are discretized using a finite volume method on a uniform Cartesian grid of cell sizes  $\Delta x$  and  $\Delta y$  in the xand ydirections, respectively. The conservative variables **Q** are stored at the centre of the cell  $I_{i,j}$  and the indices i and j denote the i-th cell in xdirection and

*j*—th cell in *y*— direction. The time evolution of the cell-centred conservative variables  $\mathbf{Q}_{i,j}$  is given by the following semi-discrete equation:

$$\frac{d\mathbf{Q}_{i,j}}{dt} = -\left[\frac{\left(\hat{\mathbf{f}}^{\mathbf{c}}_{i+\frac{1}{2},j} - \hat{\mathbf{f}}^{\mathbf{c}}_{i-\frac{1}{2},j}\right) - \left(\hat{\mathbf{f}}^{\mathbf{v}}_{i+\frac{1}{2},j} - \hat{\mathbf{f}}^{\mathbf{v}}_{i-\frac{1}{2},j}\right)}{\Delta x}\right] - \left[\frac{\left(\hat{\mathbf{G}}^{\mathbf{c}}_{i,j+\frac{1}{2}} - \hat{\mathbf{G}}^{\mathbf{c}}_{i,j-\frac{1}{2}}\right) - \left(\hat{\mathbf{G}}^{\mathbf{v}}_{i,j+\frac{1}{2}} - \hat{\mathbf{G}}^{\mathbf{v}}_{i,j-\frac{1}{2}}\right)}{\Delta y}\right] + \mathbf{S}_{i,j}$$

$$= \mathbf{Res}\left(\mathbf{Q}_{i,i}\right), \tag{9}$$

where  $\hat{\mathbf{F}}^{\mathbf{c}}$ ,  $\hat{\mathbf{G}}^{\mathbf{c}}$  and  $\hat{\mathbf{F}}^{\mathbf{v}}$ ,  $\hat{\mathbf{G}}^{\mathbf{v}}$  are the numerical approximations of the convective and viscous fluxes in the x-, and ydirections, respectively, at the cell interfaces,  $i\pm\frac{1}{2}$  and  $j\pm\frac{1}{2}$ . **Res**  $(\mathbf{Q}_{i,j})$  is the residual function. The conserved variables are then integrated in time using the following third order strong stability-preserving (SSP) Runge–Kutta scheme [4]:

$$\mathbf{Q}_{i,j}^{(1)} = \mathbf{Q}_{i,j}^n + \Delta t \operatorname{Res}\left(\mathbf{Q}_{i,j}^n\right)$$
 (10)

$$\mathbf{Q}_{i,j}^{(2)} = \frac{3}{4} \mathbf{Q}_{i,j}^{n} + \frac{1}{4} \mathbf{Q}_{i,j}^{(1)} + \frac{1}{4} \Delta t \operatorname{Res} \left( \mathbf{Q}_{i,j}^{(1)} \right)$$
 (11)

$$\mathbf{Q}_{i,j}^{n+1} = \frac{1}{3} \mathbf{Q}_{i,j}^{n} + \frac{2}{3} \mathbf{Q}_{i,j}^{(2)} + \frac{2}{3} \Delta t \operatorname{Res} \left( \mathbf{Q}_{i,j}^{(2)} \right). \tag{12}$$

The superscripts (1) - (2) denote the intermediate steps, and the superscripts n and n + 1 denote the current and the next time-steps. The time-step,  $\Delta t$ , is computed as:

$$\Delta t = \text{CFL} \cdot \min \left( \frac{1}{\alpha_v} \min_{i,j} \left( \frac{\Delta x^2}{\mu_{i,j}}, \frac{\Delta y^2}{\mu_{i,j}} \right), \min_{i,j} \left( \frac{\Delta x}{\left| u_{i,j} \right| + c_{i,j}}, \frac{\Delta y}{\left| v_{i,j} \right| + c_{i,j}} \right) \right), (13)$$

where  $\alpha_v = 3$ ,  $\mu$  is the dynamic viscosity, and c is the speed of sound. The following sections present the computation of numerical approximations of viscous and convective fluxes.

## 3.1. Spatial discretization of viscous fluxes

In this section, the discretization of the viscous fluxes is presented. For simplicity, a one-dimensional scenario is considered. The grid is discretized on a uniform grid with N cells on a spatial domain spanning  $x \in [x_l, x_r]$ . The cell centre locations are at  $x_i = x_l + (i - 1/2)\Delta x$ ,  $\forall j \in \{1, 2, ..., N\}$ , where  $\Delta x = (x_r - x_l)/N$ . In one-dimension, the viscous flux at the interface is as follows:

$$\hat{\mathbf{F}}_{i+\frac{1}{2}}^{\mathbf{v}} = \begin{bmatrix} 0 \\ -\tau_{i+\frac{1}{2}} \\ -\tau_{i+\frac{1}{2}} u_{i+\frac{1}{2}} \end{bmatrix}, \tau_{i+1/2} = \frac{4}{3} \mu_{i+1/2} \left( \frac{\partial u}{\partial x} \right)_{i+1/2}$$
 (14)

As it can be seen from Eq. (14) the viscous fluxes at cell interfaces  $x_{i+\frac{1}{2}}$ ,  $\forall i \in \{0, 1, 2, \dots, N\}$ , has to be evaluated. For this purpose, we consider the  $\alpha$ -damping approach of Nishikawa [41]. The following equation computes the velocity gradient at the cell interface,

$$\left(\frac{\partial u}{\partial x}\right)_{i+\frac{1}{2}} = \frac{1}{2} \left[ \left(\frac{\partial u}{\partial x}\right)_i + \left(\frac{\partial u}{\partial x}\right)_{i+1} \right] + \frac{\alpha}{2\Delta x} \left(u_{i+\frac{1}{2}}^R - u_{i+\frac{1}{2}}^L\right),\tag{15}$$

where

$$u_{i+\frac{1}{2}}^{L} = \hat{u}_i + \frac{\Delta x}{2} \left( \frac{\partial u}{\partial x} \right)_i, u_{i+\frac{1}{2}}^{R} = \hat{u}_{i+1} - \frac{\Delta x}{2} \left( \frac{\partial u}{\partial x} \right)_{i+1} . \tag{16}$$

By substituting the Eqs. (16) in Eq. (15) we get the following equation:

$$\left(\frac{\partial u}{\partial x}\right)_{i+1/2} = \frac{1}{2}\left(u'_i + u'_{i+1}\right) + \frac{\alpha}{2\Delta x}\left(\hat{u}_{i+1} - \frac{\Delta x}{2}u'_{i+1} - \hat{u}_i - \frac{\Delta x}{2}u'_i\right),\tag{17}$$

where  $u_i'$  represents  $\left(\frac{\partial u}{\partial x}\right)_i$ . The gradients at the cell-centres,  $\left(\frac{\partial \phi}{\partial x}\right)_i$ , are computed by the following second-order formula:

$$\left(\frac{\partial u}{\partial x}\right)_i = \frac{1}{2} \left[ \frac{\hat{u}_{i+1} - \hat{u}_{i-1}}{\Delta x} \right]. \tag{18}$$

By substituting  $\alpha = 3$  in the cell interface gradients, the second derivative can be explicitly written as follows:

$$\left(\frac{\partial^2 u}{\partial x^2}\right)_i = \frac{\left(\frac{\partial u}{\partial x}\right)_{i+1/2} - \left(\frac{\partial u}{\partial x}\right)_{i-1/2}}{\Delta x} = \frac{-\hat{u}_{i-2} + 12\hat{u}_{i-1} - 22\hat{u}_i + 12\hat{u}_{i+1} - \hat{u}_{i+2}}{8\Delta x^2}.$$

The second derivatives thus computed are cell-averaged second derivatives as explained in [42].

#### 3.2. Convective flux discretization

In this section, the discretization of the viscous convective is presented. The determination of convective fluxes involves two essential stages: first, a reconstruction phase where the solution vector at the cell centre is reconstructed to the cell interfaces, and second, an evolution of approximate Riemann solver phase in which the average fluxes at each interface are assessed using a procedure that considers the directions of the "waves". The convective flux at the interface can then be expressed as:

$$\hat{\mathbf{F}}_{i+\frac{1}{2}}^{\mathbf{c}} = F_{i+\frac{1}{2}}^{Riemann} \left( \mathbf{U}_{i+\frac{1}{2}}^{L}, \mathbf{U}_{i+\frac{1}{2}}^{R} \right), \tag{20}$$

where  $\mathbf{U}=(\alpha_1\rho_1,\alpha_2\rho_2,u,v,p,\alpha_1)^T$  is the primitive variable vector (in the two-dimensional scenario), and the superscripts L, and R denote the left and right-sided reconstructed solution vectors respectively. The Hartex–Lax–van Leer-Contact (HLLC) [43] approximate Riemann solver is used in this study.

The objective of the paper is to compute the values of  $\mathbf{U}_{-1}^{L}$ ,  $\mathbf{U}_{-1}^{R}$ such that they are physically consistent and the contact discontinuities are captured sharply. Before presenting the details of the numerical discretization, the physics is briefly discussed. The computation of  $\mathbf{U}_{i+\frac{1}{2}}^{L}, \mathbf{U}_{i+\frac{1}{2}}^{R}$  is typically carried out in two ways. The primitive variables are directly reconstructed at the interfaces, or the primitive variables are transformed to characteristic space, and characteristic variables are reconstructed (they are transformed back to primitive variable space). For conversion from physical to characteristic space, the primitive variables are multiplied with the left eigenvectors obtaining the characteristic variables, denoted by W, where  $W_m = L_n U$ . Once the shock-capturing procedure is completed, the characteristic variables are multiplied with the right eigenvectors, thereby recovering the primitive variables. The left and right eigenvectors of the two-dimensional multicomponent equations, denoted by  $L_n$  and  $R_n$ , used for characteristic variable projection are as follows:

$$\mathbf{R}_{n} = \begin{bmatrix}
\frac{\alpha_{1}\rho_{1}}{c^{2}\rho} & 1 & 0 & 0 & 0 & \frac{\alpha_{1}\rho_{1}}{c^{2}\rho} \\
\frac{\alpha_{2}\rho_{2}}{c^{2}\rho} & 0 & 1 & 0 & 0 & \frac{\alpha_{2}\rho_{2}}{c^{2}\rho} \\
-\frac{n_{x}}{c\rho} & 0 & 0 & n_{y} & 0 & \frac{n_{x}}{c\rho} \\
-\frac{n_{y}}{c\rho} & 0 & 0 & n_{x} & 0 & \frac{n_{y}}{c\rho} \\
1 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix},$$

$$\mathbf{L}_{n} = \begin{bmatrix}
0 & 0 & -\frac{n_{x}c\rho}{2} & -\frac{n_{y}c\rho}{2} & \frac{1}{2} & 0 \\
1 & 0 & 0 & 0 & -\frac{\alpha_{1}\rho_{1}}{c^{2}\rho} & 0 \\
0 & 1 & 0 & 0 & -\frac{\alpha_{2}\rho_{2}}{c^{2}\rho} & 0 \\
0 & 0 & n_{y} & n_{x} & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & \frac{n_{x}c\rho}{2} & \frac{n_{y}c\rho}{2} & \frac{1}{2} & 0
\end{bmatrix}, \tag{21}$$

where  $\mathbf{n} = [n_x \ n_y]^t$  and  $[l_x \ l_y]^t$  is a tangent vector (perpendicular to  $\mathbf{n}$ ) such as  $[l_x \ l_y]^t = [-n_y \ n_x]^t$ . By taking  $\mathbf{n} = [1, 0]^t$  and  $[0, 1]^t$  we obtain

the corresponding eigenvectors in x and y directions. Let  $\mathbf{W_b}$ , where b=1,2,3,4,5,6, represent the vector of characteristic variables for the multi-species system; and these variables have the following features:

- The first and sixth characteristic variables,  $W_1$  and  $W_6$ , corresponds to acoustic waves (shocks or rarefactions) [1,44].
- The second and third characteristic variables,  $W_2$  and  $W_3$ , corresponds to what is known as the entropy or contact wave. Across the contact discontinuity, there will be a jump in density, but the pressure and the normal velocity are continuous.
- The fourth characteristic variable,  $W_4$ , corresponds to what is known as the shear wave. When there is a contact discontinuity, there will be a jump in tangential velocity [1] in inviscid scenario but the tangential velocities are continuous in viscous scenario [2]. In xdirection the left eigenvector is as follows (and it can be observed that  $W_4$  is v velocity):

$$\boldsymbol{W}_{x} = \mathbf{L}_{xU}, \text{ where } \mathbf{L}_{x} = \begin{bmatrix} 0 & 0 & \frac{-c\rho}{2} & 0 & \frac{1}{2} & 0 \\ 1 & 0 & 0 & 0 & -\frac{\alpha_{1}\rho_{1}}{\rho c^{2}} & 0 \\ 0 & 1 & 0 & 0 & -\frac{\alpha_{1}\rho_{1}}{\rho c^{2}} & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & \frac{c\rho}{2} & 0 & \frac{1}{2} & 0 \end{bmatrix},$$

and 
$$\mathbf{U} = \begin{bmatrix} \alpha_1 \rho_1 \\ \alpha_2 \rho_2 \\ u \\ v \\ p \\ \alpha_1 \end{bmatrix}$$
. (22)

• Finally, the volume fraction  $\alpha_1$  or characteristic variable  $W_5$  is constant across shocks or rarefactions, and only varies across the contact discontinuity [44]. As the volume fractions are multiplied by unity, the shock-capturing is performed directly on the physical values, i.e. there is no characteristic transformation.

In primitive variable space, shocks lead to discontinuities in phasic densities, pressure, and normal velocity, while material interfaces cause discontinuities in phasic densities and volume fractions [1]. In viscous flow simulations, tangential velocities remain continuous [2], but they become discontinuous across contact discontinuities in inviscid simulations [1]. The following subsections present three algorithms that account for these physical phenomena in order to obtain  $\mathbf{U}^L_{i+\frac{1}{2}}$ ,  $\mathbf{U}^R_{i+\frac{1}{2}}$ .

- In Section 3.2.1, algorithm with characteristic variable reconstruction in inviscid scenario is presented.
- In Section 3.2.2, algorithm with characteristic variable reconstruction in viscous scenario is presented.
- In Section 3.2.3, algorithm with primitive variable reconstruction in viscous scenario is presented.

The main difference between inviscid and viscous algorithms is how tangential velocities are computed, taking appropriate physics into account. In all the algorithms, the same contact discontinuity sensor will be used. The details are presented below.

3.2.1. Algorithm for characteristic variable reconstruction (inviscid scenario)

The complete numerical algorithm for characteristic variable reconstruction for inviscid flow simulations is summarized below, which includes the transforming of primitive variables into characteristics variables necessary for capturing discontinuities. The computations are shown for  $\mathbf{U}^L_{i+\frac{1}{2}}.$ 

1. Compute the arithmetic or Roe averages at the interface  $(x_{i+\frac{1}{2}})$  by using neighbouring cells,  $(x_i)$  and  $(x_{i+1})$ . Compute the left  $L_n$  and right  $R_n$  eigenvectors. Transform the variables, U, into characteristic space by multiplying with the left eigenvectors

$$W_{m,b} = L_{n_{i+\frac{1}{2}}} \mathbf{U}_{m,b} \tag{23}$$

The transformed variables are denoted as  $W_{m,b}$ , where m is  $\{i-2,i-1,i,i+1,i+2,i+3\}$  and b is  $\{1,2,3,4,5,6\}$ .

2. Carry out the appropriate reconstruction procedure for each variable, as explained below, and obtain the interface values denoted by  $\pmb{W}^L_{i+\frac{1}{L},b}$ .

$$\begin{aligned} \mathbf{W}_{i+\frac{1}{2},b}^{L} &= \\ & \begin{cases} \mathbf{W}_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(\mathbf{W}_{i+\frac{1}{2}}^{L,Linear} - \mathbf{W}_{i}\right) \\ & \left(\mathbf{W}_{i+\frac{1}{2}}^{L,Linear} - \mathbf{W}_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ \mathbf{W}_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise.} \\ & \mathbf{W}_{i+\frac{1}{2},b}^{L,T} & \text{if } \min\left(\psi_{i-1},\psi_{i},\psi_{i+1}\right) < \psi_{c}. \end{cases} \\ & \text{if } b = 5: & \begin{cases} \mathbf{W}_{i+\frac{1}{2},b}^{L,T} & \\ & \\ & \\ \end{cases} \end{aligned}$$

$$\begin{aligned} \mathbf{W}_{i+\frac{1}{2},b}^{L} &= \\ & \begin{cases} \mathbf{W}_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(\mathbf{W}_{i+\frac{1}{2}}^{L,Linear} - \mathbf{W}_{i}\right) \\ & \left(\mathbf{W}_{i+\frac{1}{2}}^{L,Linear} - \mathbf{W}_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ & \mathbf{W}_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise}. \end{cases} \end{aligned}$$

$$\mathbf{W}_{i+\frac{1}{2},b}^{L} = \begin{cases} \mathbf{W}_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(\mathbf{W}_{i+\frac{1}{2},b}^{L,Linear} - \mathbf{W}_{i}\right) \\ & \left(\mathbf{W}_{i+\frac{1}{2},b}^{L,Linear} - \mathbf{W}_{i+\frac{1}{2}}^{L,MP}\right) \ge 10^{-40}, \end{cases}$$

$$(26)$$

$$\mathbf{W}_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise.}$$

3. After obtaining interface values, the reconstructed states are then recovered by projecting the characteristic variables back to physical fields:

$$\mathbf{U}_{i+\frac{1}{2}}^{L} = \mathbf{R}_{n_{i+\frac{1}{2}}} \mathbf{W}_{i+\frac{1}{2}}^{L}, 
\mathbf{U}_{i+\frac{1}{2}}^{R} = \mathbf{R}_{n_{i+\frac{1}{2}}} \mathbf{W}_{i+\frac{1}{2}}^{R}.$$
(27)

In the above-presented algorithm, there are two different reconstruction procedures. The first reconstruction approach is the MP5 scheme ( $\mathbf{W}^{L,Non-Linear}$  and  $\mathbf{W}^{L,Linear}$ ) of Suresh and Hyunh [7]. The computation of the MP5 scheme is as follows:

$$\begin{split} \mathbf{W}_{i+\frac{1}{2}}^{L,Linear} &= \frac{1}{30} \mathbf{W}_{i-2} - \frac{13}{60} \mathbf{W}_{i-1} + \frac{47}{60} \mathbf{W}_{i+0} + \frac{9}{20} \mathbf{W}_{i+1} - \frac{1}{20} \mathbf{W}_{i+2}, \\ \mathbf{W}_{j+1/2}^{\text{Non-Linear}} &= \mathbf{W}_{j+1/2}^{\text{L, Linear}} \\ &+ \min \left( \mathbf{W}_{j+1/2}^{\min} - \mathbf{W}_{j+1/2}^{\text{L, Linear}} \right), \\ \mathbf{W}_{j+1/2}^{MP} &= \mathbf{W}_{j} + \min \left( \hat{\mathbf{W}}_{j+1/2} - \hat{\mathbf{W}}_{j+1/2}^{\text{L, Linear}} \right), \\ \mathbf{W}_{j+1/2}^{\min} &= \max \left[ \min \left( \hat{\mathbf{W}}_{j}, \hat{\mathbf{W}}_{j+1}, \mathbf{W}_{j+1/2}^{MD} \right), \\ &\min \left( \hat{\mathbf{W}}_{j}, \mathbf{W}_{j+1/2}^{UL}, \mathbf{W}_{j+1/2}^{LC} \right) \right], \\ \mathbf{W}_{j+1/2}^{\max} &= \min \left[ \max \left( \hat{\mathbf{W}}_{j}, \hat{\mathbf{W}}_{j+1/2}^{LC}, \mathbf{W}_{j+1/2}^{MD} \right), \\ &\max \left( \hat{\mathbf{W}}_{j}, \mathbf{W}_{j+1/2}^{UL}, \mathbf{W}_{j+1/2}^{LD} \right) \right], \\ \mathbf{W}_{j+1/2}^{MD} &= \frac{1}{2} \left( \hat{\mathbf{W}}_{j} + \hat{\mathbf{W}}_{j+1} \right) - \frac{1}{2} d_{j+1/2}^{M}, \\ \mathbf{W}_{j+1/2}^{UL} &= \hat{\mathbf{W}}_{j} + 4 \left( \hat{\mathbf{W}}_{j} - \hat{\mathbf{W}}_{j-1} \right), \\ \mathbf{W}_{j+1/2}^{LC} &= \frac{1}{2} \left( 3\hat{\mathbf{W}}_{j} - \hat{\mathbf{W}}_{j-1} \right) + \frac{4}{3} d_{j-1/2}^{M}, \\ d_{j+1/2}^{M} &= \min \left( 4d_{j} - d_{j+1}, 4d_{j+1} - d, d_{j}, d_{j+1} \right), \\ d_{j} &= \hat{\mathbf{W}}_{j-1} - 2\hat{\mathbf{W}}_{j} + \hat{\mathbf{W}}_{j+1}, \\ \text{where,} \\ \min \left( mod(a, b) = \frac{1}{2} \left( sign(a) + sign(b) \right) \min (|a|, |b|). \end{aligned} \tag{29} \end{split}$$

Remark 3.1. One can also use other shock-capturing methods like WENO [45], Montonocity preserving explicit or implicit gradient based methods [9,46,47], MUSCL scheme [48] apart from the MP5 scheme presented above. The contact discontinuity sensor, to be presented later, will work without any modifications, including all the parameters. Results using the WENO [45] scheme instead of the MP scheme for certain test cases are shown in Appendix A of this paper.

The second candidate reconstruction function is the THINC reconstruction ( $\mathbf{W}^{L,T}$ ), a differentiable and monotone Sigmoid function [49]. Unlike the MP5 discussed above, the THINC scheme is a non-polynomial function. The explicit formula for the left and right interface for the THINC function are as follows [50]:

$$\begin{aligned} \mathbf{W}_{i+1/2}^{L,T} &= \left\{ \begin{array}{l} \mathbf{u_a} + \mathbf{u_d} \frac{K_1 + (K_2/K_1)}{1 + K_2} & \text{if } \left( \mathbf{W}_{i+1} - \mathbf{W}_i \right) \left( \mathbf{W}_i - \mathbf{W}_{i-1} \right) > 0, \\ \mathbf{U}_i & \text{otherwise} \end{array} \right. \\ \mathbf{W}_{i-1/2}^{R,T} &= \left\{ \begin{array}{l} \mathbf{u_a} - \mathbf{u_d} \frac{K_1 - (K_2/K_1)}{1 - K_2} & \text{if } \left( \mathbf{W}_{i+1} - \mathbf{W}_i \right) \left( \mathbf{W}_i - \mathbf{W}_{i-1} \right) > 0, \\ \mathbf{U}_i & \text{otherwise} \end{array} \right. \end{aligned}$$

$$(30)$$

where

$$\begin{split} K_1 &= \tanh\left(\frac{\beta}{2}\right), \ K_2 = \tanh\left(\frac{\alpha_i \beta}{2}\right), \ \alpha_i = \frac{\mathbf{W}_i - \mathbf{u_a}}{\mathbf{u_d}}, \\ \mathbf{u_a} &= \frac{\mathbf{W}_{i+1} + \mathbf{W}_{i-1}}{2}, \ \mathbf{u_d} = \frac{\mathbf{W}_{i+1} - \mathbf{W}_{i-1}}{2}. \end{split}$$

The performance of the THINC function depends on the value of the steepness parameter  $\beta$  as discussed in [51,52]. The parameter  $\beta$  controls the jump thickness, i.e., a small value of  $\beta$  leads to a smooth profile, while a large one leads to a sharp jump-like distribution. When  $\beta$  is set to 1.8, the reconstruction function becomes closer to a step-like profile, and the discontinuous solution can be resolved within about four mesh cells [49]. In this study, the value of  $\beta$  is set to 1.8 for one-dimensional cases and 1.9 for multi-dimensional test cases. The minimum value is 1.6, and the maximum is 2.0 with the proposed sensor.

Finally, the proposed contact discontinuity sensor for the fiveequation model is as follows:

$$\psi_i = \frac{2ab + \varepsilon}{\left(a^2 + b^2 + \varepsilon\right)}, \text{ where } \quad \varepsilon = \frac{0.9\psi_c}{1 - 0.9\psi_c} \xi, \quad \xi = 10^{-2}, \quad \psi_c = 0.35,$$
(32)

$$a = \frac{13}{12} |s_{i-2} - 2s_{i-1} + s_i| + \frac{1}{4} |s_{i-2} - 4s_{i-1} + 3s_i|,$$

$$b = \frac{13}{12} |s_i - 2s_{i+1} + s_{i+2}|$$

$$+ \frac{1}{4} |3s_i - 4s_{i+1} + s_{i+2}|, \text{ where } s = \frac{p}{\rho^{\gamma}}, \text{ and } \rho = \rho_1 \alpha_1 + \rho_2 \alpha_2.$$

$$(33)$$

The variables a and b in Eq. (33) are (inspired by) the smoothness indicators of the WENO scheme (refer [4,53,54] for further details regarding these equations and also the Appendix of the current paper) given by the following equations.

$$\beta_{0} = \frac{1}{4} \left( U_{i-2} - 4U_{i-1} + 3U_{i} \right)^{2} + \frac{13}{12} \left( U_{i-2} - 2U_{i-1} + U_{i} \right)^{2}, 
\beta_{1} = \frac{1}{4} \left( U_{i-1} - U_{i+1} \right)^{2} + \frac{13}{12} \left( U_{i-1} - 2\bar{U}_{i} + U_{i+1} \right)^{2}, 
\beta_{2} = \frac{1}{4} \left( 3U_{i} - 4U_{i+1} + U_{i+2} \right)^{2} + \frac{13}{12} \left( U_{i} - 2U_{i+1} + U_{i+2} \right)^{2},$$
(34)

where U can be any primitive variable and  $\beta$  is known as the smoothness indicator in WENO schemes. a and b (in Eq. (33)) are in fact  $\beta_0$  and  $\beta_2$  without the squares and different choice of variable. To the author's knowledge, these smoothness indicators are not used to detect contact discontinuities in this manner, so the THINC scheme can be applied robustly to improve the resolution of contact discontinuities. Furthermore, the word "smoothness indicator" is a misnomer (they are discontinuity detectors) as explained in [54]. The sensor is based on discontinuity sensor of [47] with modifications in variable (density is used in [47] and here it is s), the parameter  $\xi$  is  $10^{-2}$  instead of  $10^{-6}$ . Using a value of  $10^{-6}$  for  $\xi$  will detect regions that are not contact discontinuities, and the simulations will crash. It will be shown later that the variable s better captures the contact discontinuities than density. The sensor is designed such that

- it can detect the material interfaces and contact discontinuities,
- it should avoid the frequency regions of the Shu-Osher test case (Example 4.3),
- it should work for single species cases as well and avoid pitfalls of existing methods, as shown in Appendix B,
- · it should work in combination with other shock capturing schemes,
- the parameters ( $\xi$  and  $\psi_c$ ) should not change from scheme to scheme and case to case,
- · and should not lead to oscillatory results.

The variable s in Eq. (33), where  $s=\frac{p}{\rho^r}$ , is loosely based on entropy. It has similar variables as that of the physical entropy,  $S=ln\left(\frac{p}{\rho^r}\right)$ , but it is not the same. It was chosen based on the physics of the Euler equations (along with some numerical experiments) that the second (and also third in the five-equation model for two-species case) characteristic variable,  $W_2$ , is known as the *entropy wave*, which corresponds to the entropy change. As it is not exactly entropy, and there are several definitions for entropy in literature, it is not defined as entropy in this manuscript. It is referred to as the variable s to avoid confusion.

**Remark 3.2.** The selection of the variable s for detection is grounded in the observation, which will be demonstrated later, that the jump or variation in s is most pronounced near the contact discontinuity (see Figs. 3(e) and 34(c)). Additionally, it was mentioned in [55] by Roe that the gradients of entropy can serve as indicators for detecting contact discontinuities, but no specific approach or relevant formulas were provided.

**Remark 3.3.** The sensor presented in Eq. (32) have parameters  $\xi$  and  $\psi$ . While this sensor and its variants are widely used in literature, the values of the parameters depend on the chosen reconstruction scheme. The value of the parameter  $\psi$  was considered as 0.25 and 0.27 by Fu, Hu and Adams [56] for their TENO scheme, 0.5 by Chamarthi [47], 0.23 by Song et al. [57], 0.4 by Li et al. [58], 0.17 and 0.24 by Li and Fu for the multiphase flows [59], 0.5 and 0.6 by Huang and Fu [60] etc. In the current paper, the value of  $\psi$  suitable for detecting contact discontinuities, along with the variable s, was 0.35 based on the analysis shown in Example 4.3. Value remains the same for all the cases and schemes in this paper and was also used for the simulations of gas-liquid compressible flows in [31].

3.2.2. Algorithm for characteristic variable reconstruction (viscous scenario)

In an inviscid scenario presented above, the tangential velocities are discontinuous across a contact discontinuity [1]. However, for a viscous flow scenario (the intended target for the proposed algorithm and the physically realistic scenario), the tangential velocities are continuous across a contact discontinuity [2]. If a variable is continuous, one may use a central scheme. In this regard, the following approach is used for the fourth characteristic variable ( $W_4$ ) to ensure that a central scheme ( $W^{C,Linear}$ ) computes the tangential velocities:

$$\mathbf{W}_{i+\frac{1}{2},b}^{L} = \begin{cases} \text{if } b = 4: & \begin{cases} \mathbf{W}_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \Omega_d > 0.01 \\ \mathbf{W}_{i+\frac{1}{2},b}^{C,Linear} & \text{otherwise,} \end{cases}$$

$$(35)$$

where

$$\mathbf{W}_{i+\frac{1}{2}}^{C,Linear} = \frac{1}{60} \left( \mathbf{W}_{i-2} - 8\mathbf{W}_{i-1} + 37\mathbf{W}_i + 37\mathbf{W}_{i+1} - 8\mathbf{W}_{i+2} + \mathbf{W}_{i+3} \right). \quad (36)$$

 $\Omega_d$  is the Ducros sensor [32,33,61]

$$\Omega_d = \max\left(\Omega_{i+m}\right), \quad \text{for } m = -1, 0, 1, \tag{37}$$

and computed as follows:

$$\Omega_{i} = \frac{\left| -p_{i-2} + 16p_{i-1} - 30p_{i} + 16p_{i+1} - p_{i+2} \right|}{\left| +p_{i-2} + 16p_{i-1} + 30p_{i} + 16p_{i+1} + p_{i+2} \right|} \frac{(\nabla \cdot \mathbf{u})^{2}}{(\nabla \cdot \mathbf{u})^{2} + |\nabla \times \mathbf{u}|^{2}},$$
 (38)

where  $\mathbf{u}$  is the velocity vector, and the derivatives of velocities are computed by the fourth-order implicit gradient approach of [62], which is as follows:

$$\frac{1}{6} \left. \frac{\partial u}{\partial x} \right|_{i-1} + \frac{2}{3} \left. \frac{\partial u}{\partial x} \right|_{i} + \frac{1}{6} \left. \frac{\partial u}{\partial x} \right|_{i+1} = \frac{1}{2\Delta x} \left( \hat{u}_{i+1} - \hat{u}_{i-1} \right). \tag{39}$$

Ducros sensor is a shock detector and cannot detect contact discontinuities as shown in [32]. In [32], authors have used a separate discontinuity sensor for shocks and contact discontinuities and showed that using the Ducros sensor in the presence of contact discontinuities will lead to oscillations. Therefore, a central scheme will always compute the tangential velocities ( $\mathbf{W_4}$ ) because it cannot detect the contact discontinuities. In Refs. [32,34], the Ducros sensor is also considered for acoustic waves, but this manuscript does not consider such an approach. The paper aims to improve the contact discontinuities and material interfaces only. Finally, the above algorithm is denoted as HY-THINC-D and is used only for viscous test cases.

Remark 3.4. One could argue that tangential velocities are continuous across a shockwave and should (or might) always be computed using a central scheme. The current approach uses a dimension-by-dimension approach, and the shockwaves are not always aligned with the grid. If a shockwave is at an angle with the grid, there will be oscillations if a limiter is not applied. An approach that aligns the grid with a shockwave might (or can) compute tangential velocities always with a central scheme. The author tried using a central scheme for tangential velocities without any sensor; while it worked in some cases, it crashed in many cases (due to oscillations near shocks).

![](_page_7_Figure_2.jpeg)

Fig. 3. Numerical solution for multi-species shock tube problem in Example 4.1 on a grid size of N = 200. Solid line: Reference solution; green stars: MP5; red circles: HY-THINC.

## 3.2.3. Algorithm for primitive variable reconstruction (viscous scenario)

Shock-capturing is typically performed on characteristic variables for coupled hyperbolic equations like the Euler equations to achieve the cleanest results, as explained by van Leer in [63]. When interface values are directly reconstructed using primitive variables, it can lead to small oscillations, particularly for high-resolution schemes. Reconstructing primitive variables  $\mathbf{U}=(\alpha_1\rho_1,\alpha_2\rho_2,u,v,p,\alpha_1)^T$  implies that the THINC scheme is applied to phasic densities and volume fractions. However, the proposed algorithm can be used with primitive variable reconstruction, especially in viscous flow scenarios. The algorithm for primitive variable reconstruction for viscous flow simulations is as follows:

In x-direction:

$$\begin{split} \mathbf{U}_{i+\frac{1}{2},b}^{L} &= \\ & \text{if } b = 3 \text{:} \quad \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i}\right) \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ U_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise}. \end{cases} \\ & \text{if } b = 4 \text{:} \quad \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \Omega_{d} > 0.01 \\ U_{i+\frac{1}{2},b}^{C,Linear} & \text{otherwise}. \end{cases} \end{split}$$

In y-direction:

$$\mathbf{U}_{i+\frac{1}{2},b}^{L} =$$

$$\begin{cases} \text{if } b = 4\text{:} & \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i}\right) \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ V_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise.} \end{cases} \\ \text{if } b = 3\text{:} & \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \Omega_{d} > 0.01 \\ U_{i+\frac{1}{2},b}^{C,Linear} & \text{otherwise.} \end{cases} \end{cases}$$

In all directions:

$$\begin{aligned} & \mathcal{J}_{i+\frac{1}{2},b}^{L} = \\ & \text{ if } b = 5 \text{:} & \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{ if } \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i}\right) \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ & U_{i+\frac{1}{2},b}^{L,Linear} & \text{ otherwise.} \end{cases} \\ & \text{ if } b = 1,2 \text{:} & \begin{cases} U_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{ if } \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i}\right) \left(U_{i+\frac{1}{2}}^{L,Linear} - U_{i+\frac{1}{2}}^{L,MP}\right) \geq 10^{-40}, \\ & U_{i+\frac{1}{2},b}^{L,Linear} & \text{ otherwise.} \end{cases} \\ & U_{i+\frac{1}{2},b}^{L,T} & \text{ if } \min\left(\psi_{i-1},\psi_{i},\psi_{i+1}\right) < \psi_{c}. \end{cases} \\ & \text{ if } b = 6 \text{:} & \begin{cases} U_{i+\frac{1}{2},b}^{L,T} & \text{ if } \min\left(\psi_{i-1},\psi_{i},\psi_{i+1}\right) < \psi_{c}. \end{cases} \end{aligned}$$

The above algorithm is still denoted as HY-THINC-D, but it will be indicated in the results that the primitive variables are reconstructed. The similarity between the characteristic and primitive variable reconstruction is that in xdirection, v is reconstructed in the characteristic

![](_page_8_Figure_2.jpeg)

**Fig. 4.** Numerical solution for isolated contact test case using = 200 grid points at = 1*.*0, [Example](#page-8-3) [4.2,](#page-8-3) where Solid line: Reference solution; green stars: MP5; red circles: HY-THINC.

space and centralized if the Ducros sensor criterion is satisfied. Likewise, in the − direction, is reconstructed in the characteristic space and centralized if the Ducros sensor criterion is satisfied. Similarly, the variable is centralized in primitive variable space in − direction (Eqs. [\(40](#page-7-2))), and is centralized in primitive variable space in − direction (Eqs. [\(41](#page-7-3))), respectively. As explained earlier, the proposed contact discontinuity sensor may detect shockwaves in some test cases, and in those cases (if detected by the sensor), density is also reconstructed using THINC if primitive variables are reconstructed. Such a reconstruction is also physically consistent as density is discontinuous across the shock. It must be repeated here that velocities and pressure are continuous across a contact discontinuity, and THINC should not be applied in these regions.

## **4. Results and discussion**

In this section, the proposed spatial discretization schemes are tested for a set of benchmark cases to assess the performance of both single and multi-dimensional test cases.

## **Example 4.1.** Multi-species shock tube

The first one-dimensional test case is the two-fluid modified shock tube of Abgrall and Karni [[64\]](#page-29-23). The initial conditions of the test case are as follows:

$$\left(\alpha_{1}\rho_{1}, \alpha_{2}\rho_{2}, u, p, \alpha_{1}, \gamma\right) = \begin{cases} (1, 0, 0, 1, 1, 1.4) & \text{for } x < 0\\ (0, 0.125, 0, 0.1, 0, 1.6) & \text{for } x \ge 0. \end{cases}$$
(43)

Simulations are carried out on a uniformly spaced grid with 200 cells on the spatial domain −0*.*5 ≤ ≤ 0*.*5 with a constant CFL of 0.4 and the final time is = 0*.*2.

[Fig.](#page-7-4) [3](#page-7-4) shows various schemes' density, pressure, volume fraction profiles, sensor variables and the profile of variable . The proposed scheme, HY-THINC, captures the material interface without oscillations and uses fewer points than the MP5 scheme, [Fig.](#page-7-5) [3\(a\)](#page-7-5). [Fig.](#page-7-1) [3\(b\)](#page-7-1) shows no visible difference in the pressure profiles. The volume fraction profiles shown in [Fig.](#page-7-1) [3\(c\)](#page-7-1) indicate that the HY-THINC scheme captures the volume fractions within a few points compared to MP5. [Fig.](#page-7-1) [3\(d\)](#page-7-1) shows the effect of using different variables in the proposed contact discontinuity sensor. [Fig.](#page-7-1) [3\(d\)](#page-7-1) indicates that using variable detects the contact discontinuity, whereas if the density is used as a variable, the contact discontinuity is not detected as indicated by the number of points used for resolving them. Finally, the variation of variable is shown in [Fig.](#page-7-1) [3\(e\),](#page-7-1) indicating a significant jump in variable across the material interface.

## **Example 4.2.** Material interface advection

This two-species one-dimensional problem is the advection of an isolated material interface [\[65](#page-29-24)[,66](#page-29-25)]. The initial conditions for this test case are given by

$$\left(\alpha_{1}\rho_{1},\alpha_{2}\rho_{2},u,p,\alpha_{1},\gamma\right) = \begin{cases} (10,0,0.5,1/1.4,1.0,1.6), & 0.25 \le x < 0.75\\ (00,1,0.5,1/1.4,0.0,1.4), & x < 0.25 \text{ or } x \ge 0.75 \end{cases} \tag{44}$$

The simulation was conducted on a computational domain spanning from = 0 to = 1, utilizing a total of = 200, as in [\[66](#page-29-25)], uniformly distributed grid points until reaching a final time of = 2*.*0. Both boundaries were subject to periodic conditions. [Fig.](#page-8-4) [4](#page-8-4) illustrates the comparison between the exact and numerical solutions of various schemes, which all precisely captured the material interface without any unwanted oscillations. The HY-THINC scheme accomplished this with fewer points than the MP5 scheme, which indicates that the sensor reliably detected the material interface. Pressure remained constant without oscillation and remained close to machine precision.

## **Example 4.3.** Shock/density wave interaction

In this test case, the Shu–Osher problem [[67\]](#page-29-26) extended for a binary mixture of Helium and Nitrogen by [[68\]](#page-29-27), that is initially separated by a shock, is considered. The initial conditions are as follow:

$$\left(\rho, u, p, \alpha_{\text{He}}, \alpha_{\text{N}_2}\right) = \begin{cases} (3.8571, 2.6294, 10.3333, 0, 1) & \text{for } x < -4\\ (1 + 0.2\sin(5x), 0, 1, 1, 0) & \text{for } x \ge -4. \end{cases}$$
(45)

The case was solved on a computational domain = [−5*,* 5] with = 200 and 800 , as in [[68\]](#page-29-27), uniformly distributed grid points until a final time, = 1*.*8. The Shu–Osher problem is commonly used to evaluate a scheme's one-dimensional shock and density perturbation capturing capabilities. As shown in [Fig.](#page-9-0) [5,](#page-9-0) the proposed HY-THINC

![](_page_9_Figure_2.jpeg)

Fig. 5. Numerical solution for multicomponent shock-density wave interaction using N = 200 and 800 grid points at t = 2.0, for Example 4.3, where Solid line: Reference solution; red circles: HY-THINC density; cyan stars: HY-THINC volume fractions; yellow circles: sensor location.

scheme matches the "exact" solution well, computed on a grid size of 3200 by the WENO scheme, both for density and volume fractions. The proposed contact discontinuity sensor correctly identifies the material interface and does not alter the high-frequency regions, as they are not discontinuities at all.

The contact discontinuity sensor presented earlier, Eq. (33), is also analysed here in this example. Aspects of interest are the location of the sensor detection regions, the motivation for the current sensor and the differences from the sensor used in [47] are presented. First, the modifications in the proposed discontinuity detector compared to that of Li et al. [58] and Chamarthi [47] are examined. The Li et al. detector used fluxes as the variable (Eq. (47)) to detect the discontinuities, and fluxes cannot be used for multicomponent flows as primitive variables are to be reconstructed to avoid oscillations [64,65,69]. Li's objective for proposing the detector was to choose between linear and WENO schemes, an objective which is different from that of the present study. Chamarthi [47] used Li's detector but with density as the variable in the detector to choose between variable or flux reconstruction in the regions away from discontinuities for high-order accuracy. Li's detector with density as the variable (as in [47]) is given below:

$$\psi_{j} = \frac{2ab + \varepsilon}{\left(a^{2} + b^{2} + \varepsilon\right)}, \varepsilon = \frac{0.9\psi_{c}}{1 - 0.9\psi_{c}} \xi^{2}, \quad \xi = 10^{-3}, \quad \psi_{c} = 0.5, \tag{46}$$

$$a = |\rho_{j} - \rho_{j-1}| + |\rho_{j} - 2\rho_{j-1} + \rho_{j-2}|,$$

$$b = |\rho_{j} - \rho_{j+1}| + |\rho_{j} - 2\rho_{j+1} + \rho_{j+2}|.$$
(47)

Li detector for reconstructing the entropy wave with THINC would be as follows:

A one-dimensional scenario is here considered. Therefore, only the entropy wave is reconstructed with THINC if a discontinuity is detected.

The current detector is also presented here for comparison.

$$\psi_j = \frac{2ab + \varepsilon}{\left(a^2 + b^2 + \varepsilon\right)}, \text{ where } \quad \varepsilon = \frac{0.9\psi_c}{1 - 0.9\psi_c}\xi, \quad \xi = 10^{-2}, \quad \psi_c = 0.35,$$
(49)

$$a = \frac{13}{12} |s_{i-2} - 2s_{i-1} + s_i| + \frac{1}{4} |s_{i-2} - 4s_{i-1} + 3s_i|,$$

$$b = \frac{13}{12} |s_i - 2s_{i+1} + s_{i+2}| + \frac{1}{4} |3s_i - 4s_{i+1} + s_{i+2}|, \text{ where } s = \frac{p}{\rho^{\gamma}}.$$
(50)

$$\boldsymbol{W}_{i+\frac{1}{2},b}^{L,MP5} = \begin{cases} \boldsymbol{W}_{i+\frac{1}{2},b}^{L,T} & \text{if } b = 2 \text{ and } \underbrace{\min\left(\psi_{i-1},\psi_{i},\psi_{i+1}\right) < \psi_{c}}_{\text{Current sensor}} \\ \boldsymbol{W}_{i+\frac{1}{2},b}^{L,MP5} & \text{otherwise.} \end{cases}$$
(51)

The parameters a and b in Li's detector given by Eq. (47) are different from the current detector given by Eq. (50). The variable used in Li's sensor is density (as modified by Chamarthi in [47]),  $\rho$ , and the current sensor is s, and  $s = \frac{\rho}{\rho^r}$ . It is not known if Li's sensor performs appropriately for multicomponent flows. As explained in Appendix A, the variables a and b in Eq. (50) are inspired by the smoothness indicators of the WENO scheme (Eq. (68)). a and b are infact  $\beta_0$  and  $\beta_2$  without the squares and a different choice of variable. The final difference between the two detectors is the number of cells that detect the discontinuity. Increasing the value of  $\psi_c$  detects the regions that might not be discontinuities, and reducing it below a threshold will not detect the discontinuities. To understand the advantages and differences of the proposed sensor, the single-species shock/entropywave problem Shu and Osher [67] is considered. The initial conditions of the test case are as follows:

$$(\rho, u, p) = \begin{cases} (3.857, 2.629, 10.333), & \text{if } -5 \le x < -4, \\ (1 + 0.2\sin(5(x - 5)), 0, 1), & \text{if } -4 \le x \le 5. \end{cases}$$
 (52)

This test case is about shockwave interaction with high-frequency oscillating sinusoidal waves as it evaluates a scheme's capability to capture both shock and avoid high-frequency regions simultaneously. Li et al. [58] also devised their sensor to avoid activation of WENO in high-frequency regions (see Fig. 10 in [58] and the corresponding discussion). The computational domain of this test case is x = [-5, 5], and the final time is t = 1.8. Simulations are carried out on a grid size

![](_page_10_Figure_2.jpeg)

**Fig. 6.** Density profiles obtained for Shu–Osher test case using Li and new sensors. Solid line: Reference solution; green stars: density with Li sensor; red circles: density with new sensor; cyan squares: location of Li sensor's detection region and blue circles: location of new sensor's detection region.

![](_page_10_Figure_4.jpeg)

**Fig. 7.** The TENO discontinuity sensor from [[70\]](#page-29-29).

of = 900, as in [\[9\]](#page-28-6), using both the sensors with the MP5 scheme. The *exact solution* is obtained on a fine grid resolution of 40,000 points using the WENO scheme. As shown in [Fig.](#page-10-0) [6\(b\)](#page-10-0), the proposed sensor matches the reference solution well, whereas Li's sensor with density as the variable modified the solution profile, squaring effect, compared to the reference result. Observing [Fig.](#page-10-1) [6\(a\)](#page-10-1), Li's sensor, shown with cyan squares, flagged several regions as discontinuities, whereas the current sensor, shown with blue circles, detected only the shockwave at ≈ 2*.*4.

The Li sensors' detection of high-frequency regions as discontinuities required further attention. It has been observed in the literature that some studies (and the corresponding discontinuity detectors) considered those regions not to be discontinuities. In [\[70](#page-29-29)], one of the authors of the TENO-THINC scheme [[30\]](#page-28-20) proposed a discontinuity detector based on the TENO scheme, and it has been mentioned in the corresponding paper that the number of cells detected as troubled cells decreases with increasing resolution. [Fig.](#page-10-2) [7](#page-10-2) shows results from [\[70](#page-29-29)] with TENO sensor. In the rightmost figure, with 800 grid points, the TENO discontinuity sensor did not detect the high-frequency region as discontinuities. The current approach did not alter the high-frequency region regardless of the grid resolution as shown in [Fig.](#page-11-0) [8.](#page-11-0)

- • Chamarthi and Frankel [[8](#page-28-32)] also made similar observations in the work that the limiting process should be avoided in the high-frequency region by conducting simulations with a linear scheme in their paper (readers can refer to Fig. 14 in [[8](#page-28-32)] and the corresponding discussion).
- Furthermore, the TENO-THINC approach of [[30\]](#page-28-20) detected the high-frequency region between ≈ 0.7 and 2.4 as a discontinuity and applied THINC, as shown in [Fig.](#page-12-0) [9\(a\)](#page-12-0). Whereas the discontinuity detector of Krividonova et al. [[71\]](#page-29-30) did not detect the high-frequency region with either density or entropy as a variable used in their shock detector.
- Finally, Zhao et al. [[72\]](#page-29-31) studied several discontinuity sensors, and their new proposed sensor also did not alter the high-frequency region, as shown in [Fig.](#page-12-1) [9\(c\).](#page-12-1) [Figs.](#page-11-1) [8\(a\),](#page-11-1) [8\(b\),](#page-11-2) and [8\(c\)](#page-11-2) show the density profile for the current sensor, and regardless of the resolution, the sensor did not detect the high-frequency region and applied THINC. The sensor detected only shockwave (but the THINC is applied only to the entropy wave and not to shockwave in this study), as indicated by the blue circles, similar to that of Krividonova et al. [\[71](#page-29-30)].

![](_page_11_Figure_2.jpeg)

Fig. 8. Density profile for Shu–Osher test case with the proposed sensor using 200, 400 and 800 grid points and HY-THINC scheme: Figs. 8(a)–8(c). Red circles: HY-THINC; blue circles: Sensor location; and dashed line: Reference solution.

• Even for the multi-species case the sensor did not modify the regions of high-frequency, as shown in Fig. 5. Using density as the variable to detect the interface also failed with the current sensor, as shown in Fig. 3(d). All these observations played a role in devising the new sensor. Furthermore, the high-frequency region in the Shu–Osher test case is not a *contact discontinuity at all*, also discussed in Ref. [8,31].

The developers of the TENO scheme also integrated it with the THINC scheme using Discontinuous Galerkin methods. Although employing a distinct algorithm, the THINC scheme tended to activate in high-frequency regions, leading to less accurate results than those obtained with the WENO scheme, as illustrated in Fig. 10. The authors acknowledged that their proposed TENO-THINC limiter did not surpass the performance of the WENO limiter [60]. In contrast, the current algorithm avoids using THINC in high-frequency regions. These findings underscore the robustness and reliability of the present method while highlighting the potential issues of previous approaches.

#### 4.1. Multi-dimensional test cases

This section carries out numerical simulations for multi-dimensional test cases. Each example highlights the advantages of the proposed algorithms.

- Examples 4.4, 4.5, and 4.6 show that the proposed contact discontinuity sensor does not falsely detect and modify the results if there are no contact discontinuities. These test cases are single species to evaluate the proposed sensor.
- In Example 4.7, it is demonstrated that tangential velocities across
  the contact discontinuity can be accurately reconstructed using a
  central scheme, effectively preventing oscillations. This example
  also illustrates that pressure can be accurately computed using a
  central scheme when reconstructing primitive variables.
- Examples 4.8 and 4.9 show that even in the case of shock-material interface interaction, the tangential velocities can be computed using a central scheme.
- Example 4.10 is to show that the proposed sensor can detect contact discontinuities even if there are more than two species in the flow, and the sensor can be used for hypersonic flows.

#### Example 4.4. Isentropic vortex (Inviscid case)

In this test 2D inviscid case, the proposed numerical scheme is evaluated for the two-dimensional vortex evolution problem [73,74]. This test case is typically considered for verifying the order of accuracy of a proposed test case. Here, it is used to confirm whether the sensor is falsely getting activated as no discontinuities are present. The

computation domain is  $[-5, 5] \times [-5, 5]$  with periodic boundaries on all sides. To the mean flow, an isentropic vortex is added, and the initial flow field is initialized as follows:

flow field is initialized as follows:  

$$p = \rho^{\gamma}, T = 1 - \frac{(\gamma - 1)\epsilon^{2}}{8\gamma\pi^{2}}e^{(1-r^{2})}, u = 1 - \frac{\epsilon}{2\pi}e^{\frac{1}{2}(1-r^{2})}y,$$

$$v = 1 + \frac{\epsilon}{2\pi}e^{\frac{1}{2}(1-r^{2})}x,$$
(53)

where  $r^2 = x^2 + y^2$  and the vortex strength  $\epsilon$  is taken as 5. The computations are performed to reach a final time t=10. The simulation is conducted on a grid size of  $100 \times 100$ . The density contours are shown in Fig. 11(a) and the sensor correctly detected no contact discontinuities in the flow and preserved the uniform entropy condition.

#### Example 4.5. Inviscid Taylor-Green Vortex (Inviscid case)

In this example, the performance of the contact discontinuity sensor in solving the three-dimensional inviscid Taylor–Green vortex problem, a classical benchmark problem in computational fluid dynamics, is investigated. This test case is a discontinuity-free test case. The proposed sensor should not have any effect on the solution. The initial conditions for the simulation are set on a periodic domain of size  $x, y, z \in [0, 2\pi)$ , and the simulations are run until time t = 10 on a grid size of  $64^3$ , with a specific heat ratio of  $\gamma = 5/3$ . The flow problem is considered incompressible since the mean pressure is significantly large. The initial conditions of the test case are as follows:

$$\begin{pmatrix} \rho \\ u \\ v \\ w \\ p \end{pmatrix} = \begin{pmatrix} 1 \\ \sin x \cos y \cos z \\ -\cos x \sin y \cos z \\ 0 \\ 100 + \frac{(\cos(2z) + 2)(\cos(2x) + \cos(2y)) - 2}{16} \end{pmatrix}.$$
 (54)

Fig. 12 indicates that the proposed scheme (the contact discontinuity sensor) did not affect this test case as it should. The results obtained by the MP5 and HY-THINC scheme are one over the other for kinetic energy and enstrophy. Even though there are no discontinuities in this test case, the TENO-THINC scheme of Takagi et al. [30] has improved the results (see Fig. 21 of [30]), which indicates that either the TENO based indicator is falsely detecting smooth flow regions as discontinuities or it could be an issue of reconstructing all the variables using the THINC scheme. It is beyond the scope of the paper to analyse the TENO-THINC, but the current approach is free of such unexpected results.

#### Example 4.6. Periodic double-shear layer (Viscous case)

In this 2D viscous test case, the impact of applying THINC to all the waves, as in [30], is investigated. The test involves two initially parallel

![](_page_12_Figure_2.jpeg)

**Fig. 9.** Discontinuity detection locations in various papers from the literature. [Fig.](#page-12-0) [9\(a\)](#page-12-0) is reproduced from [[30\]](#page-28-20) with permission from Elsevier BV 2024, License number 5734370293424. [Fig.](#page-12-1) [9\(b\)](#page-12-1) is reproduced from [[71\]](#page-29-30) with permission from Elsevier BV 2024, License number 5734370669124. [Fig.](#page-12-1) [9\(c\)](#page-12-1) is reproduced from [\[72](#page-29-31)] with permission from Elsevier BV 2024, License number 5734370461931.

**Table 1** Parameters of the periodic double shear layer test case.

| Ma  | Re     | Pr   | 𝛾   |
|-----|--------|------|-----|
| 0.1 | 10,000 | 0.73 | 1.4 |

shear layers that develop into two significant vortices at = 1. The non-dimensional parameters for this test case are presented in [Table](#page-12-2) [1](#page-12-2).

The initial conditions were:

$$p = \frac{1}{\gamma \text{Ma}^2}, \rho = 1, \ u = \begin{cases} \tanh \left[\theta(y - 0.25)\right], & \text{if } (y \le 0.5), \\ \tanh \left[\theta(0.75 - y)\right], & \text{if } (y > 0.5), \end{cases}$$
(55a)

$$v = 0.05 \sin [2\pi(x)] \text{ and } \theta = 120 \text{ or } 80,$$
 (55b)

Simulations are conducted for = 1*.*0 × 10−4 and = 3*.*0 × 10−5 . The reference solution = 1*.*0 × 10−4 and =120, shown in [Fig.](#page-14-1) [13\(a\)](#page-14-1), was computed with the MP scheme on a 800 × 800 grid. Unphysical braid vortices and oscillations can occur on the shear layers if the grid is under-resolved for this test case. [Figs.](#page-14-2) [13\(b\),](#page-14-2) [13\(c\),](#page-14-2) and [13\(d\)](#page-14-2) displays the -vorticity computed by MP, MP6 - Ducros and HY-THINC-D schemes on a grid size of 196 × 196. As expected, the upwind scheme, MP5, gave unphysical braid vortices and the MP6 - Ducros and HY-THINC-D schemes, where the central scheme computes the tangential velocities, are similar to the fine grid results. In this test case, the THINC scheme is not activated as there are no contact discontinuities, which indicates the proposed sensor works reliably.

![](_page_13_Figure_2.jpeg)

**Fig. 10.** Results of Huang et al. [\[60](#page-29-19)] using THINC with permission from Elsevier BV 2024, License number 5936770943086.

![](_page_13_Figure_4.jpeg)

**Fig. 11.** Numerical solution (density contours) for Isentropic vortex, [Example](#page-11-3) [4.4.](#page-11-3)

Observing [Figs.](#page-14-2) [13\(e\)](#page-14-2) and [13\(f\),](#page-14-2) the TENO and TENO-THINC results are not identical. While the results obtained by the TENO scheme are, as expected, similar to the MP5 scheme as all the variables are computed using the upwind scheme, the TENO-THINC scheme results further deviate from the TENO5 itself. It indicates the TENO-based discontinuity sensor is falsely getting activated and is affecting the results where there are no discontinuities. Simulations with the proposed contact discontinuity sensor and Ducros sensor are also free of spurious vortices even if reconstruction is directly carried out for the primitive variables, shown in [Fig.](#page-15-0) [14\(a\).](#page-15-0)

For comparison, [Fig.](#page-16-1) [15,](#page-16-1) adapted from [\[37](#page-28-27)], shows simulations performed on a grid of 320 × 320 points—2.6 times larger than the current approach. Despite this higher resolution and the optimized TENO8 scheme, braid vortices still exist in their results. These results underscore the advantages of the current approach of applying a central scheme to the tangential velocities.

Finally, the simulation results for = 3*.*0 × 10−5 and = 80 are shown in [Fig.](#page-16-2) [16.](#page-16-2) [Figs.](#page-16-3) [16\(a\),](#page-16-3) [16\(b\)](#page-16-4), and [16\(c\)](#page-16-4) displays the vorticity computed by MP, MP6 - Ducros and HY-THINC-D schemes

**Table 2** Order of accuracy, [Example](#page-11-3) [4.4](#page-11-3).

| N    | Fifth-order | Order | HY-THINC   | Order |
|------|-------------|-------|------------|-------|
| 252  | 3.8E − 03   | –     | 2.41E − 03 | –     |
| 502  | 3.47E − 04  | 3.46  | 6.47E − 04 | 1.90  |
| 1002 | 1.56E − 05  | 4.48  | 1.63E − 04 | 1.98  |
| 2002 | 5.10E − 07  | 4.91  | 4.10E − 05 | 1.99  |

on a grid size of 320 × 320. Once again, the upwind scheme, MP5, gave unphysical braid vortices and the MP6 - Ducros and HY-THINC-D schemes are similar to the fine grid results. For this scenario, the TENO-THINC scheme failed to pass the test (severe oscillations and eventually crashed).

On the order of accuracy and physical consistency: As explained in the introduction the third objective of the paper is to show the proposed algorithms which consider the physical consistency of the variables (some are continuous and some are discontinuous depending on the situation) can lead to superior results than an approach that is mathematically high-order accurate. The following points explain the reasons with the order of accuracy analysis and results for the periodic shear layer case.

- [Table](#page-13-3) [2](#page-13-3) shows the order of accuracy analysis for the isentropic vortex test case, [Example](#page-11-3) [4.4.](#page-11-3) For this analysis, the fifth-order WENO interpolation scheme is considered (readers can refer to the review article by Shu [[53,](#page-29-12)[75–](#page-29-34)[77\]](#page-29-35). The approach is sometimes referred to as AWENO and WCNS in the literature.) As expected, the WENO interpolation scheme showed fifth-order convergence, and the current approach is only second-order accurate.
- But, the same fifth-order scheme when used for the periodic shear layer case simulation did produce spurious vortices, shown in [Fig.](#page-16-5) [17](#page-16-5). The results obtained by the MP6-Ducros and HY-THINC-D schemes, shown in [Fig.](#page-14-3) [13](#page-14-3), are free of spurious vortices. The main difference is that the proposed schemes in this paper use a central scheme for reconstruction of tangential velocities (they are continuous). In contrast, the fifth-order scheme uses a dissipative upwind scheme for all the variables, leading to inferior results.
- Furthermore, the results shown in [Fig.](#page-16-1) [15](#page-16-1), taken from [\[37](#page-28-27)], are computed using highly optimized TENO schemes, yet they produce spurious vortices on much finer grid points than those of the current paper. Readers may also refer [[38\]](#page-28-28) where a deep reinforcement learning approach was used for the same test case and yet produced inferior results than the current ones. While one may not conclusively state that the high-order accuracy is not important, including the behaviour of the physical variables along with the order of accuracy may be necessary. The current approach has similarities to that of the multidimensional upwinding approach of Roe [[39\]](#page-28-29), as mentioned in [[31\]](#page-28-21), which suggests treating each variable differently depending on the direction and the physics.
- It was also mentioned in [\[31](#page-28-21)] that a scheme that can be highorder accurate, prevent spurious vortices (considering the physics of the variables), prevent oscillations across shocks and interfaces, etc, and it can be considered as a future work. Lastly, a recent paper by Li and Fu [[59\]](#page-29-18), which used TENO-THINC for multiphase flows, also employed the same fiveand six-point stencils used in this paper and is second-order accurate (page 12 of [\[59](#page-29-18)]).

It has been explained in [[61\]](#page-29-20) that computing tangential velocities using a central scheme will prevent unphysical vortices for this test case. In the later test cases, numerical examples will show that the tangential velocities can be computed using a central scheme, even across material interfaces.

![](_page_14_Figure_2.jpeg)

**Fig. 12.** Normalized kinetic energy and enstrophy using HY-THINC and MP5 schemes for [Example](#page-11-4) [4.5](#page-11-4) on grid size of 64<sup>3</sup> . Solid red line: MP5 and dashed blue line: HY-THINC.

![](_page_14_Figure_4.jpeg)

**Fig. 13.** Figures show -vorticity contours of the considered schemes on a grid size of 196<sup>2</sup> for = 1*.*0 × 10−4 , [Example](#page-11-5) [4.6.](#page-11-5)

## **Example 4.7.** Kelvin Helmholtz instability (Viscous case)

The Kelvin–Helmholtz instability (KHI) is a hydrodynamic instability that occurs when there is a velocity shear between two fluids of different densities. This instability arises due to an unstable velocity gradient at the interface between the two fluids, leading to vortices and turbulent mixing patterns. It plays a crucial role in the evolution of the mixing layer and the transition to turbulence. This test case is typically a single-species test case but can be modified to a multi-species one.

![](_page_15_Figure_2.jpeg)

**Fig. 14.** z-vorticity contours of the HY-THINC-D scheme with primitive variable reconstruction on a grid size of  $196^2$  for  $\mu=1.0\times10^{-4}$  and  $\theta=120$ , Example 4.6.

The initial conditions are set over a periodic domain of  $[0, 1] \times [0, 1]$ .

$$p = 2.5, \quad \alpha_{1}\rho_{1}(x, y) = \begin{cases} 2, & \text{if } 0.25 < y \le 0.75, \\ 0, & \text{else.} \end{cases},$$

$$\alpha_{2}\rho_{2}(x, y) = \begin{cases} 0, & \text{if } 0.25 < y \le 0.75, \\ 1, & \text{else.} \end{cases}$$

$$u(x, y) = \begin{cases} 0.5, & \text{If } 0.25 < y \le 0.75, \\ -0.5, & \text{else } . \end{cases}$$

$$\alpha_{1} = \begin{cases} 1.0, & \text{If } 0.25 < y \le 0.75, \\ 0.0, & \text{else } . \end{cases}$$

$$v(x, y) = 0.1 \sin(4\pi x) \left\{ \exp\left[-\frac{(y - 0.75)^{2}}{2\sigma^{2}}\right] + \exp\left[-\frac{(y - 0.25)^{2}}{2\sigma^{2}}\right] \right\}, \text{ where } \sigma = 0.05 / \sqrt{2}.$$

Fig. 18(a) shows the initial conditions, where the green and blue colours indicate volume fractions of two different species. The specific heat ratio of the first species is taken as 1.5, and for the second species is taken as 1.4. The test case is computed using the HY-THINC-D scheme on three different grid sizes ( $512^2$ ,  $1024^2$ , and  $2048^2$ ) until a final time of t = 0.8 for  $\mu = 1.0 \times 10^{-4}$ . Figs. 18(b) and 18(c) shows the density gradient contours on grid sizes of  $512^2$  and  $2048^2$ , respectively. While the coarse grid simulation shows more vortical structures, they disappear with increased grid resolution.

Fig. 19(a) shows the density gradient contours on a grid size of  $1024^2$ . Figs. 19(b) and 19(c) show regions where the THINC scheme is used; the contours indicate the sensor locations in xand ydirections. The proposed contact discontinuity sensor correctly identified the interfaces. Figs. 19(d)–19(f) show the pressure contours overlayed on volume fractions, u-velocity contours overlayed on volume fractions, and v-velocity contours overlayed on volume fractions, respectively. These figures plot the species volume fractions using green and white colours for better visualization. In this test case, there are no shocks, and the Ducros sensor did not detect any contact discontinuities; therefore, tangential velocities are computed using central schemes. From Figs. 19(e) and 19(f) one can see no oscillations in either velocities and the contours pass through from one material to the other.

In Figs. 18 and 19, the characteristic variables are reconstructed, and it can be assured that the tangential velocities are reconstructed using the central scheme. The following *extreme* algorithm is considered for this test case to show that even pressure can be reconstructed using a central scheme. As explained in Section 2 the primitive variable vector is  $\mathbf{U} = (\alpha_1 \rho_1, \alpha_2 \rho_2, u, v, p, \alpha_1)^T$ . In the following algorithm, pressure (*b*=5) is computed using a central scheme, and the tangential velocities in each direction are computed using central schemes.

$$U_{i+\frac{1}{2},b}^{L} =$$

$$\begin{cases} \text{if } b = 5 \colon & \boldsymbol{U}_{i+\frac{1}{2},b}^{C,Linear} \\ \\ \\ \text{if } b = 1,2 \colon \begin{cases} \boldsymbol{U}_{i+\frac{1}{2},b}^{L,Non-Linear} & \text{if } \left(\boldsymbol{U}_{i+\frac{1}{2}}^{L,Linear} - \boldsymbol{U}_{i}\right) \\ \\ \\ \boldsymbol{U}_{i+\frac{1}{2},b}^{L,Linear} - \boldsymbol{U}_{i+\frac{1}{2}}^{L,MP} \right) \geq 10^{-40}, \\ \\ \boldsymbol{U}_{i+\frac{1}{2},b}^{L,Linear} & \text{otherwise.} \\ \\ \boldsymbol{U}_{i+\frac{1}{2},b}^{L,T} & \text{if } \min\left(\psi_{i-1},\psi_{i},\psi_{i+1}\right) < \psi_{c}. \end{cases}$$

$$\text{if } b = 6 \colon \quad \boldsymbol{U}_{i+\frac{1}{2},b}^{L,T}.$$

$$\text{(57)}$$

In x-direction:

$$\mathbf{U}_{i+\frac{1}{2},b}^{L} = \begin{cases} \text{if } b = 3: & U_{i+\frac{1}{2},b}^{L,Linear} \\ \text{if } b = 4: & U_{i+\frac{1}{2},b}^{C,Linear} \end{cases}$$
(58)

In *y*-direction:

$$\mathbf{U}_{i+\frac{1}{2},b}^{L} = \begin{cases} \text{if } b = 4: & \mathbf{U}_{i+\frac{1}{2},b}^{L,Linear} \\ \text{if } b = 3: & \mathbf{U}_{i+\frac{1}{2},b}^{C,Linear} \end{cases}$$
(59)

Fig. 20 indicates that the algorithm mentioned above would work without any issues, and even pressure can be computed using a central scheme. These reconstructions are physically consistent as only density and volume fractions are discontinuous across the material interface, and the rest of the variables are continuous.

Motivation for the above-mentioned algorithm, where even pressure was computed using a central scheme, is as follows:

- It is well known in the fluid dynamics [40,69,78] that the pressure and velocity are continuous across a material interface and a contact discontinuity.
- As explained in the introduction, there are several interface sharpening methods, and THINC, which is used in the current paper, is a popular approach. Adams and collaborators [29,79] have used THINC only to volume fractions and partial densities and mentioned that, "It is not necessary to correct other variables since pressure and velocity should be continuous at the fluid interface".
- At the same time, Fu and his collaborators have used THINC for all the variables wherever their discontinuity detector has identified a discontinuity, regardless of the physics of the concerned discontinuity. In their first paper, where they proposed the detector such that THINC can be used [30], all the variables are computed using THINC regardless of the physical characteristics of the discontinuity. Later, they further refined the discontinuity detector approach in Liang et al. [80]. Finally, they extended their work to multiphase flows [59,81] based on the original detector of Takagi et al. as the improved approach in Liang et al. failed to detect the contact discontinuity approach in the blast wave test case. The excerpt from [59] is as follows:

"The other component, THINC reconstruction with improved simplicity, is utilized for discontinuities including shock and contact waves as well as material interfaces, following the activation criteria similar to Takagi et al."

which indicates that the THINC reconstruction is applied to both shock and contact waves as well as material interfaces, which is in contrast with the current work and that of Adams and collaborators [29,79]. It is also important to note that the steepness parameter of the THINC scheme  $\beta$ , which affects the interface thickness, has varied from the original paper of Takagi et al. [30] to the recent paper of Li and Fu [59]. Takagi et al. have used a value of 1.6, Liang et al. [80] have used 2.4, which therefore improved the vortical structures, and Li and Fu [59] have used a

![](_page_16_Figure_2.jpeg)

**Fig. 15.** Figure is taken from Ref. [[37](#page-28-27)], where the simulations are computed on a grid size of 320<sup>2</sup> .

![](_page_16_Figure_4.jpeg)

**Fig. 16.** Figures show -vorticity contours of the considered schemes computed on a grid size of 320<sup>2</sup> for = 3*.*0 × 10−5 and = 80, [Example](#page-11-5) [4.6](#page-11-5).

![](_page_16_Figure_6.jpeg)

**Fig. 17.** Spurious vortices using fifth-order scheme [\[53](#page-29-12)], [Example](#page-11-5) [4.6.](#page-11-5)

value of 1.5. The higher the value of the parameter , the sharper the interface and the more vortical structures. For the multiphase test cases, Li and Fu have reduced the steepness parameter and also multiplied the parameter by unit normals, which would potentially reduce the value of .

• The TENO-THINC of [\[30](#page-28-20)] failed for the Kelvin Helmholtz instability test case, even on a coarse grid. The possible reason for failure is applying an interface sharpening approach for continuous variables across the interface (pressure and velocities). Even for the proposed sensor, applying THINC for pressure and velocities in the regions of contact discontinuities crashed. It was shown in [\[31\]](#page-28-21) that applying THINC to all the variables gave inferior results for the periodic shear layer case and even the inviscid Taylor Green Vortex.

- In the above algorithm (variable five in Eq. [\(57](#page-15-1))), the pressure is computed with a central scheme, and the results are without any oscillations, as it is continuous, as indicated by the contours. Given the two different philosophies in the literature, the purpose of the above algorithm and results is to justify the proposed approach of computing only phasic densities and volume fractions using the THINC scheme. The comments about the TENO-THINC are not to critique the method but to support the present algorithm. Authors of TENO-THINC might not have considered all the possibilities (physical nature of the variables) during their work.
- It is also important to note that Chamarthi (author of the current paper) and Frankel [\[8\]](#page-28-32) (see the Appendix of the concerned paper) did use THINC as a reconstruction approach, indicating the familiarity with the approach, with improved results for certain test cases.

**Example 4.8.** Compressible triple point problem (Inviscid and viscous case)

In this test case, the multi-species compressible triple point problem is considered. This scenario poses a challenging three-state, twodimensional Riemann problem involving two distinct materials. This benchmark test is widely used to validate the ability of interfacecapturing schemes to resolve sharp interfaces. This test case emphasizes the generation of fine small-scale vortical structures along the contact discontinuities due to Kelvin–Helmholtz instabilities. The computational domain spans [0, 7] × [0, 3]. The initial conditions for this test case [\[9](#page-28-6)[,82](#page-29-40)] are as follows:

$$(\alpha_1 \rho_1, \alpha_2 \rho_2, u, v, p, \gamma) =$$

![](_page_17_Figure_2.jpeg)

Fig. 18. Figures show initial condition (Fig. 18(a) and density gradient contours for grid sizes 512' (Fig. 18(b)) and 2048' (Fig. 18(c)), for Example 4.7.

![](_page_17_Figure_4.jpeg)

Fig. 19. Figures show density gradient contours (Fig. 19(a)), contact discontinuity sensor locations in x– (Fig. 19(b)) and y– (Fig. 19(c)) directions, pressure contours overlayed on volume fractions (Fig. 19(d)), u-velocity contours overlayed on volume fractions (Fig. 19(e)), and v-velocity contours overlayed on volume fractions (Fig. 19(f)), for Example 4.7, on a grid size of  $1024^2$ .

$$\begin{cases} (1.0, 0.0, 0, 0, 1.0, 1.5), & \text{sub-domain } [0, 1] \times [0, 3], \\ (0.0, 1.0, 0, 0, 0.1, 1.4), & \text{sub-domain } [1, 1] \times [0, 1.5], \\ (0.125, 0.0, 0, 0, 0.1, 1.5), & \text{sub-domain } [1, 7] \times [1.5, 3]. \end{cases}$$

In the first and third subdomains,  $\alpha_1=1$ , and in the second subdomain,  $\alpha_1=0$ . Simulations are carried out in both inviscid and viscous scenarios. First, the inviscid simulation is carried out on a grid size of 3584  $\times$  1536 (as in [82]), and the final time considered is 5.0. Reflective boundary conditions are imposed for all the boundaries. Fig. 21 shows the results obtained by the HY-THINC scheme at t=5.0.

Figs. 21(a) and 21(b) show the density gradient contours and vorticity contours, respectively. The results are similar and competitive

compared to those obtained by the multi-resolution approach of Pan et al. [82], whose grid resolution is the same (see Fig. 12 of Ref. [69]). In Fig. 21(a), various contact discontinuities are denoted by C1, C2 and C3. Likewise, shockwaves are denoted by S1, RS1, TS1 and TS2. Figs. 21(c) and 21(d) show the locations of the THINC scheme activation, which indicates that the sensor has detected all the contact discontinuities and did not detect shockwaves at all. These results justify the name *contact discontinuity sensor* used for the sensor instead of being called a discontinuity detector (that detects both shocks and contacts).

![](_page_18_Figure_2.jpeg)

**Fig. 20.** Figures show density gradient contours ([Fig.](#page-18-4) [20\(a\)\)](#page-18-4) and pressure contours ([Fig.](#page-18-5) [20\(b\)](#page-18-5)) overlayed on volume fractions, for [Example](#page-13-1) [4.7](#page-13-1), on a grid size of 1024<sup>2</sup> using the *extreme* algorithm.

![](_page_18_Figure_4.jpeg)

**Fig. 21.** Density gradient contours [\(Fig.](#page-18-2) [21\(a\)](#page-18-2)), vorticity contours [\(Fig.](#page-18-3) [21\(b\)](#page-18-3)), contact discontinuity sensor location in − direction ([Fig.](#page-18-3) [21\(c\)](#page-18-3)) at time = 5 using the HY-THINC scheme and the contact discontinuity sensor location ([Fig.](#page-18-3) [21\(d\)\)](#page-18-3) in − direction at time = 5 using the HY-THINC scheme, [Example](#page-16-0) [4.8,](#page-16-0) on a grid resolution of 3584 × 1536 for inviscid simulation.

[Fig.](#page-19-0) [22](#page-19-0) shows the density gradient contours obtained by the MP5 and HY-THINC schemes at = 5 on a grid size of 1792 × 768. It can be observed that the HY-THINC scheme captured the material interface within a few cells compared to the MP5 scheme based on the contact discontinuity thickness. Contact discontinuity within the material (indicated by the red arrow in [Fig.](#page-19-1) [22\(b\)\)](#page-19-1) is also detected and is computed using THINC.

![](_page_19_Figure_2.jpeg)

**Fig. 22.** Density gradient contours obtained by using the MP5 scheme, shown in [Fig.](#page-19-2) [22\(a\),](#page-19-2) and HY-THINC, shown in [Fig.](#page-19-1) [22\(b\),](#page-19-1) at time = 5, [Example](#page-16-0) [4.8](#page-16-0), on a grid resolution of 1792 × 768 for inviscid simulation.

![](_page_19_Figure_4.jpeg)

**Fig. 23.** Numerical Schlieren images at various time instances = 0.2, 1.0, 3.0, 3.5, 4.0 and 5.0 for the compressible triple point problem using HY-THINC scheme, [Example](#page-16-0) [4.8,](#page-16-0) on a grid resolution of 3584 × 1536 for inviscid simulation.

Numerical Schlieren images at various time instances = 0.2, 1.0, 3.0, 3.5, 4.0 and 5.0 are shown in [Fig.](#page-19-3) [23,](#page-19-3) which depicts the development of the shock system and are consistent with Figure 11 of [\[82](#page-29-40)] and Figure 24 of [\[9\]](#page-28-6). The initial conditions give rise to contact discontinuities denoted as C1 and C2 in [Fig.](#page-19-3) [23\(](#page-19-3)a). In the vicinity of the triple point, a distinctive roll-up region takes shape due to the faster advancement of shockwave S1 compared to S2, as illustrated in [Fig.](#page-19-3) [23\(](#page-19-3)b). As S1 continues its trajectory, it comes into contact with the contact discontinuity labelled as C3, as depicted in [Fig.](#page-19-3) [23\(](#page-19-3)c), resulting in distinct vortical structures. When = 3*.*5, shockwave S1 reaches the right boundary and reflects into the domain, [Fig.](#page-19-3) [23](#page-19-3) (d). This inward movement of S1 gives rise to the formation of transmitted shock waves, denoted as TS1 and TS2, which, in turn, interact with the contact

discontinuity C1, as depicted in [Fig.](#page-19-3) [23](#page-19-3)(e), resulting in complex vortical structures.

Next, viscous simulations are carried out using the HY-THINC-D (to show that a central scheme across material interfaces can compute the tangential velocities). [Fig.](#page-20-1) [24](#page-20-1) shows the fine grid solution, the grid size of 3584 × 1536, for = 1*.*0× 10−4 . [Fig.](#page-20-2) [24\(a\)](#page-20-2) show the density gradient contours, and many of the vortical structures that are observed in the *inviscid* solution are non-existent, yet the contact discontinuities are captured sharply. [Fig.](#page-20-3) [24\(b\)](#page-20-3) shows the velocity contours, plotted in red colour, are continuous across the material interfaces, and there are no oscillations.

[Fig.](#page-21-1) [25\(a\)](#page-21-1) shows the density gradient contours on a grid size of 1792 × 768 computed using characteristic variable reconstruction. As

![](_page_20_Figure_2.jpeg)

**Fig. 24.** Density gradient contours, shown in Fig. 24(a), and v-velocity contours, shown in Fig. 24(b), at time t = 5 using HY-THINC-D scheme, Example 4.8, on a grid resolution of  $3584 \times 1536$  for viscous simulation.

expected, the characteristic variable reconstruction is free of oscillations. Fig. 25(b) shows the density gradient contours on a grid size of 1792 × 768 computed using primitive variable reconstruction, and there are mild oscillations behind the shock. Figs. 25(c) and 25(d) show regions where the Ducros sensor is activated; the contours indicate the Ducros sensor locations in xand ydirections, respectively. The Ducros sensor correctly identified the shocks and did not detect the contact discontinuities, which indicates the tangential velocities computed using a central scheme across the material interfaces, and there are no oscillations. Figs. 25(e) and 25(f) show the v-velocity contours overlayed on density gradient contours and pressure contours overlayed on density gradients, respectively. The TENO-THINC scheme [30] failed even for this test case, as the THINC scheme is used for velocity and pressure across material interfaces. These results indicate the proposed approach is physically consistent, oscillation free and robust for viscous compressible multi-species flows.

Comments on the results obtained with the proposed algorithms and the results in the literature:

- Fig. 26(a) shows the density gradient contours of the simulations carried out by Paula et al. [83] using WENO3 and THINC, and Fig. 26(b) shows the results obtained by Zhang et al. [79] using WENO5 and THINC. While both results are inviscid simulations on the same grid, the combination of WENO5 and THINC showed rich vortical structures, whereas the combination of WENO3 and THINC was more dissipative, as it lacked vortical structures.
- It is interesting to note that the inviscid simulation using the proposed approach, shown in Fig. 22(b), is similar to that of Zhang et al. [79]. Whereas the result of the viscous simulation, shown in Fig. 25(a) using HY-THINC-D approach, is similar to that of Paula et al. [83].
- These results indicate that it is difficult to compare two different numerical methods based solely on the vortical structures in the results. In a recent review article by Garcia et al. [84], the authors have noted that:

"for model problems like triple-point shear layers and shock/heliumbubble interactions show varying amplitudes of instability waves at the interface of a shocked bubble that depend on the specific method and fundamental work is required to establish benchmark problems with converged, validated solutions".

It is beyond the scope of the paper to identify and explain these differences. Authors of [84] have also noted that many papers have neglected mass diffusion, heat conduction, and, to their knowledge, no diffuse interface model consistently treats all diffusive effects. The current paper also neglected these potentially important aspects, but has at least included the viscous effects, and the other physical effects may be considered in future works.

**Example 4.9.** Two dimensional multi-species viscous Richtmyer–Meshkov instability (Viscous case)

In this test case, the two-dimensional viscous Richtmyer–Meshkov (RMI) instability is computed [9]. RMI occurs when an incident shock accelerates an interface between two fluids of different densities. As the shock wave hits the perturbed interface, it deforms and generates vortices due to the baroclinic effect. As time progresses, the  $SF_6$ , which is heavier, penetrates the air, the lighter fluid, leading to the formation of a spike. The computational domain, shown in Fig. 27, for this test case, extends from  $0.0 \le x \le 16.0\lambda$  and  $0.0 \le y \le 1.0\lambda$  where  $\lambda$  is the initial perturbation wavelength and the initial shape of the interface is given by

$$\frac{x}{\lambda} = 0.4 - 0.1 \sin\left(2\pi \left(\frac{y}{\lambda} + 0.25\right)\right),\tag{61}$$

where  $\lambda = 1$ 

The initial conditions for this test case are as follows:

 $(\rho, u, v, p, \gamma) =$ 

$$\begin{cases} (1, 1.24, 0, 1/1.4, 1.4), & \text{for pre-shock air} \\ (1.4112, 0.8787, 0, 1.6272/1.4, 1.4), & \text{for post-shock air} \\ (5.04, 1.24, 0, 1/1.4, 1.093), & \text{for } SF_6 \end{cases}$$
 (62)

In the current simulations, a constant dynamic viscosity of  $\mu=1.0\times10^{-4}$  is considered [9,85]. Periodic boundary conditions are applied at the top and bottom boundaries of the domain, and the initial values are set at the left and right boundaries. As a Cartesian grid is used in the present simulation, it may lead to generating secondary instabilities at the material interface, and to mitigate these secondary instabilities, the initial perturbation is smoothened by incorporating an artificial diffusion layer as proposed in Ref. [17]:

$$f_{sm} = \frac{1}{2} \left( 1 + \operatorname{erf} \left( \frac{\Delta D}{E_i \sqrt{\Delta x \Delta y}} \right) \right)$$

$$u = u_x \left( 1 - f_x \right) + u_D f$$
(63)

where u represents the primitive variables near the initial interface, the parameter  $E_i$  introduces additional thickness to the initial material interface,  $\Delta D$  is the distance from the initial perturbed material interface, and subscripts L and R denote the left and right interface conditions. Parameter  $E_i$  is chosen as 5 in this test case. Simulations are conducted on two different grid sizes,  $4096 \times 256$  cells and  $8192 \times 512$ , with a constant CFL of 0.4. Computational results of normalized density gradient magnitude  $\phi = \exp(|\nabla \rho|/|\nabla \rho|_{max})$  obtained at t = 11.0 by various schemes are shown in Fig. 28.

Figs. 28(b) and 28(d) indicate no noticeable spurious oscillations for the HY-THINC-D scheme, and the interface thickness is thinner than the MP5 scheme (28(a) and 28(c)) as the THINC computes the interfaces. HY-THINC-D scheme also shows improved resolution regarding

![](_page_21_Figure_2.jpeg)

Fig. 25. Figures show density gradient contours obtained by characteristic variable reconstruction (Fig. 25(a)), density gradient contours for primitive variable reconstruction (Fig. 25(b)), Ducros sensor locations in x– (Fig. 25(c), red lines) and y– (Fig. 25(d), blue lines) directions, v-velocity contours overlayed on density gradient contours (Fig. 25(e)) and pressure contours overlayed on density gradients (Fig. 25(f)), Example 4.8, on a grid resolution of 1792 × 768.

the roll-up vortices, indicating the scheme's low numerical dissipation compared to the MP5 scheme. These results indicate that the proposed interface sharpening approach, with a contact discontinuity sensor, can capture material interface within a few cells, even for long-duration simulations. The simulations have no oscillations despite using a central scheme across the material interface.

**Example 4.10.** Shock wave interaction with a multi-material bubble (Inviscid case)

In this test case, a Mach 6.0 shock wave in the air meets a cylindrical helium bubble enclosed by an R22 shell considered in Ref. [82]. It is a

complex flow combining two test cases: shock interaction with a helium bubble [5,86] and shock interaction with the R22 bubble [27]. The computational domain for this test case spans  $[0, 0.356] \times [0, 0.089]$ . The initial states for this test case are given by:

$$(\rho,u,v,p,\gamma) = \left\{ \begin{array}{ll} (5.268,5.752,0,41.83,1.400), & \text{for post-shock air,} \\ (1.000,0.000,0,\ 1.00,1.400), & \text{for pre-shock air,} \\ (3.154,0.000,0,\ 1.00,1.249), & \text{for R22 shell,} \\ (0.138,0.000,0,\ 1.00,1.667), & \text{for helium bubble,} \end{array} \right.$$

(64)

![](_page_22_Figure_2.jpeg)

**Fig. 26.** [Fig.](#page-22-0) [26\(a\)](#page-22-0) shows results of Paula et al. [\[83](#page-29-41)] using WENO3 and THINC with permission from Elsevier BV 2023, License number 6079050360185. [Fig.](#page-22-1) [26\(b\)](#page-22-1) shows results of Zhang et al. [\[79](#page-29-37)] using WENO5 and THINC with permission from Elsevier BV 2025, License number 6079060211989.

![](_page_22_Figure_4.jpeg)

**Fig. 27.** Schematic of initial condition of Richtmyer–Meshkov instability, [Example](#page-20-0) [4.9](#page-20-0).

![](_page_22_Figure_6.jpeg)

**Fig. 28.** Comparison of normalized density gradient magnitude, , contours for two-dimensional viscous Richtmyer–Meshkov instability problem in [Example](#page-20-0) [4.9](#page-20-0) on a grid size of 4096 × 256 and 8192 × 512. Contours are from 1 to 1.7 at time = 11*.*0 using the proposed scheme.

where the shock is placed initially at = 0*.*1. A helium bubble with an initial radius of 0.15 is placed at = 0*.*15, = 0*.*0445. The outer R22 shell has a radius of 0.30 at the exact location. Symmetry boundary conditions are set for upper and lower edges, while the left and right boundaries have inflow and outflow conditions. [Figs.](#page-23-1) [29](#page-23-1)[–31](#page-24-0) show the evolving density gradients and vorticity contours at five different time instances, offering insight into the progression of the shock system and the deformation of the helium and R22 bubbles obtained by the HY-THINC approach.

At = 5*.*0 × 10−3 , shown in [Fig.](#page-23-2) [29\(a\)](#page-23-2), we observe the shock wave's behaviour as it refracts upon encountering the R22 shell, creating a concave transmitted shock wave due to differences in acoustic impedance. This behaviour aligns with findings in previous research [\[27](#page-28-17),[51](#page-29-10)]. At = 1*.*0 × 10−2 , shown in [Fig.](#page-23-3) [29\(b\)](#page-23-3), we observe the shock wave's behaviour as it impacts the helium bubble. Unlike the R22 bubble, the air-helium interface produces a convex transmitted shock inside the helium bubble. As this shock propagates further downstream, it subsequently impacts the aft end of the helium bubble. From = 1*.*5 × 10−2 to = 3*.*0 × 10−2 , R22 and helium are mixed with the ambient air

![](_page_23_Figure_2.jpeg)

**Fig. 29.** Numerical Schlieren images at times = 5.0 × 10<sup>−</sup><sup>3</sup> , 1.0 × 10<sup>−</sup><sup>2</sup> , and 1.5 × 10<sup>−</sup><sup>2</sup> for the shock multiple bubble test case using HY-THINC scheme, [Example](#page-21-0) [4.10,](#page-21-0) on a grid resolution of 8192 × 2048, as in [\[82](#page-29-40)].

![](_page_23_Figure_4.jpeg)

**Fig. 30.** Numerical Schlieren images and corresponding vorticity contours at = 2.0 × 10<sup>−</sup><sup>2</sup> for the shock multiple bubble test case using HY-THINC scheme, [Example](#page-21-0) [4.10,](#page-21-0) on a grid resolution of 8192 × 2048, as in [\[82](#page-29-40)].

resulting in complex vortical structures, as shown in [Figs.](#page-23-4) [30](#page-23-4) and [31](#page-24-0). The deformed helium bubble and R22 have a similar shape to previous numerical results of Pan et al. [\[82](#page-29-40)]. Numerical simulations carried out with the standard MP failed to pass this test case due to negative density and pressure beyond = 1.0 × 10−<sup>2</sup> , indicating the proposed approach's robustness. For this test case, the ''carbuncle'' phenomenon is observed due to the use of the HLLC Riemann solver, and it *may be* avoided by using a rotated HLLC-HLL or any other carbuncle-free approach. In a recent paper, Zhang et al. [[79\]](#page-29-37) have also simulated the concerned test case using HLLC Riemann solver, and their results also showed the carbuncle phenomenon. Readers can see Figure 15 of the concerned paper. For this test case, Paula et al. [[83\]](#page-29-41) have conducted a threedimensional inviscid simulation using the discrete-equations method on a grid size of 452 million grid points. It is beyond the scope of the paper and the resources available to the author to conduct a viscous

simulation for this test case, as it requires hundreds of millions of grid points.

#### **5. Conclusions**

The paper proposed a physically consistent, to the extent possible, numerical discretization approach for simulating viscous compressible multicomponent flows. The key contributions and observations of the manuscript are summarized below:

• A contact discontinuity detector is proposed such that the THINC scheme can be used for the contact discontinuities and material interfaces. The detector uses the variable , where = . The detector is devised in such a way that it avoids high-frequency regions so that THINC is not applied in those regions. The study

![](_page_24_Figure_2.jpeg)

![](_page_24_Figure_3.jpeg)

**Fig. 31.** Numerical Schlieren images and corresponding Vorticity contours at = 3.0 × 10<sup>−</sup><sup>2</sup> for the shock multiple bubble test case using HY-THINC scheme, [Example](#page-21-0) [4.10,](#page-21-0) on a grid resolution of 8192 × 2048, as in [\[82](#page-29-40)].

demonstrated the effectiveness of this approach through a series of benchmark tests, showcasing its ability to capture material interfaces and significantly outperform existing methods (WENO and MP).

- The proposed approach does not rely on volume fraction criteria to identify interfaces, which can be tedious if several species are in the domain. It can robustly identify the material interfaces (and contact discontinuity within a material) even if there are more than two species. For the hypersonic flow test case, [Example](#page-21-0) [4.10](#page-21-0), where the standard shock-capturing technique failed, the proposed approach completed simulations, indicating the method's robustness.
- For both viscous and inviscid simulations, the THINC is applied only to the phasic densities and the volume fractions in physical space and only to entropy wave and volume fractions in characteristic space (such an approach is also consistent with the physics across a material interface [\[2\]](#page-28-1)).
- For viscous simulations, the tangential velocities are computed using a central scheme across contact discontinuities using the Ducros sensor (a shock detector that cannot detect material interfaces) as they are continuous across the contact discontinuities. Using a central scheme did not lead to any oscillations.
- For shock-free viscous test cases, tangential velocities and pressure can be computed using a central scheme without any sensor, as they are continuous across the contact discontinuities. Nonlinear reconstruction techniques are required only for phasic densities and volume fractions.
- It was also shown in the paper that a physically consistent algorithm, yet only second-order accurate, may produce superior results than a numerical scheme that is mathematically high-order accurate. The present contact discontinuity sensor, along with some algorithmic simplifications, was able to simulate gas-liquid flows [\[31](#page-28-21)] and match experimental results qualitatively, and is also second-order accurate. Finally, the algorithm presented by Hoffmann, Chamarthi, and Frankel [\[34](#page-28-24)], which was used for the prediction of hypersonic transitional flows and has some similarities with the HY-THINC-D scheme, was also second-order accurate for nonlinear test cases. It may be possible to obtain high-order accuracy yet prevent vortices in periodic shear layer, prevent oscillatory results for test cases with shocks and material interfaces, predict transition to turbulence, simulate gas-liquid

flows with shocks, etc, is beyond the scope of the paper and can be taken up by someone with more resources, GPUs, permanent position, etc.

#### **Declaration of competing interest**

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Amareshwara Sainadh Chamarthi has conflicts of interest with the authors of the following papers, specifically Nikolaus Adams, Xiangyu Hu, Feng Xiao, and Lin Fu: 1. Y. Feng, J. Winter, N. A. Adams, F. S. Schranner, A general multi-objective bayesian optimization framework for the design of hybrid schemes towards adaptive complex flow simulations, Journal of Computational Physics 510 (2024) 113088. 2. S. Takagi, L. Fu, H. Wakimura, F. Xiao, A novel high-order low-dissipation teno-thinc scheme for hyperbolic conservation laws, Journal of Computational Physics 452 (2022) 110899. 3. Q. Li, Y. Lv, L. Fu, A high-order diffuseinterface method with teno-thinc scheme for compressible multiphase flows, International Journal of Multiphase Flow 173 (2024) 104732. 4. W. Zhang, N. Fleischmann, S. Adami, N. A. Adams, A hybrid weno5isthinc reconstruction scheme for compressible multiphase flows, Journal of Computational Physics 498 (2024) 112672. 5. Zhiwei Hea, Yucang Ruanb, Yaqun Yu, Baolin Tian, Feng Xiao, Self-adjusting steepnessbased schemes that preserve discontinuous structures in compressible flows, Journal of Computational Physics, Volume 463, 15 August 2022 If there are other authors, they declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### **Acknowledgments**

This work is not funded by anyone. The concerned work was started in April 2019. Some preliminary results, shown in [Appendix](#page-26-0) [B](#page-26-0), were obtained in the first week of January 2021. Inviscid results were submitted to the APS conference (not attended due to lack of funding) in 2023 [[87\]](#page-29-45), and final results are presented in this paper. All the simulations are carried out on the author's computer, a Mac Mini M1 with 8 GB RAM or the now-defunct MacBook Pro with 16 GB RAM. Some of the results took six months or more to compute. A.S. thanks his wife (sorry you could not even go to your father's funeral) and his kid for their support despite the horrible situations they have gone through during these six years for the sake of ''research and science''.

![](_page_25_Figure_2.jpeg)

**Fig. 32.** -vorticity contours of the considered schemes computed on a grid size of 196<sup>2</sup> , [Example](#page-11-5) [4.6](#page-11-5).

![](_page_25_Figure_4.jpeg)

**Fig. 33.** Density gradient contours at time = 5 using various schemes, [Example](#page-16-0) [4.8,](#page-16-0) on a grid resolution of 1792 × 768.

## **Appendix A. Application of contact discontinuity sensor with WENO scheme**

This appendix presents the results obtained using the WENO scheme (instead of the MP scheme) in conjunction with the proposed contact discontinuity detector and the THINC scheme . The WENO scheme is also briefly explained for clarity. In the WENO scheme, the fifthorder upwind-biased reconstruction is nonlinearly weighted from three different third-order sub-stencils. For simplicity, the reconstruction polynomials to the left side of the cell interface at + are only presented here. The three-third order reconstruction formula of variable is given by

$$\bar{U}_{i+\frac{1}{2}}^{(0)} = \frac{1}{6} \left( 2U_{i-2} - 7U_{i-1} + 11U_i \right), 
\bar{U}_{i+\frac{1}{2}}^{(1)} = \frac{1}{6} \left( -U_{i-1} + 5U_i + 2U_{i+1} \right), 
\bar{U}_{i+\frac{1}{2}}^{(2)} = \frac{1}{6} \left( 2U_i + 5U_{i+1} - U_{i+2} \right).$$
(65)

The values *̄* () + at cell interfaces are approximated from different substencils, while represents the cell-averaged values at cell centres. The three third-order upwind approximation polynomials in Eq. ([65\)](#page-25-1) are dynamically chosen through a nonlinear convex combination. This adaptation occurs to employ a lower-order spatial discretization that avoids interpolation across discontinuities and provides the necessary numerical dissipation for shock capturing. The fifth-order WENO-Z scheme [[45\]](#page-29-4) used in this paper is as follows:

$$\bar{U}_{i+\frac{1}{2}} = \sum_{k=0}^{2} \omega_k^z \bar{U}_{i+\frac{1}{2}}^{(k)},\tag{66}$$

where are the nonlinear weights which are given by,

$$\omega_k^z = \frac{\alpha_k^z}{\sum_{k=0}^2 \alpha_k^z}, \quad \alpha_k^z = \gamma_k \left( 1 + \left( \frac{\tau_5}{\epsilon + \beta_k} \right)^p \right), \tau_5 = |\beta_0 - \beta_2|, p = 1, \quad (67)$$

where and are ideal linear weights and smoothness indicators, respectively. = 10−20 is a small constant to prevent division by zero.

![](_page_26_Figure_2.jpeg)

**Fig. 34.** Numerical solution for blast wave problem using N = 400 grid points at t = 0.038, for the Blast wave case. Figs. 34(a) and 34(b) shows the density profiles obtained by the MP5/HY-THINC and WENO5Z/WENO5Z-THINC, respectively. Fig. 34(c) shows the profile of variable s. Solid line: Reference solution; blue squares: WENO5-Z; red stars: WENO5-Z-THINC; green stars: HY-THINC and red circles: MP5.

The non-linear weights of the convex combination are based on local smoothness indicators  $\beta_k$ . These indicators measure the sum of the normalized squares of the scaled  $L^2$  norms of all derivatives of the lower-order polynomials. The goal is to assign small weights to lower-order polynomials with discontinuities in their underlying stencils, resulting in a non-oscillatory solution. Smoothness indicators  $\beta_k$  are as follows:

$$\beta_0 = \frac{1}{4} \left( U_{i-2} - 4U_{i-1} + 3U_i \right)^2 + \frac{13}{12} \left( U_{i-2} - 2U_{i-1} + U_i \right)^2, 
\beta_1 = \frac{1}{4} \left( U_{i-1} - U_{i+1} \right)^2 + \frac{13}{12} \left( U_{i-1} - 2\bar{U}_i + U_{i+1} \right)^2, 
\beta_2 = \frac{1}{4} \left( 3U_i - 4U_{i+1} + U_{i+2} \right)^2 + \frac{13}{12} \left( U_i - 2U_{i+1} + U_{i+2} \right)^2.$$
(68)

Fig. 32 shows the *z*-vorticity contours of the WENO5-Z and the WENO5-Z-THINC-Ducros schemes for the periodic shear layer test case, Example 4.6. As expected, the WENO5-Z-THINC-Ducros approach is free of spurious vortices, unlike that of WENO5-Z, as the tangential velocities are computed using the central scheme.

Fig. 33 shows the density gradient contours, inviscid scenario, obtained by the WENO5-Z and WENO-Z THINC scheme at t=5 for Example 4.8. It can be observed that the WENO5-Z-THINC scheme, Fig. 33(b), captured the material interface within a few cells compared to the base WENO5-Z scheme, Fig. 33(a), based on the contact discontinuity thickness. These results indicate the benefits of the proposed approach, where the THINC scheme can be used with two different approaches and produce oscillation-free and sharp numerical results.

#### Appendix B. Further analysis of the contact discontinuity sensor

The primary objective of the paper was to simulate compressible multicomponent flows with material interfaces and to reduce dissipation across them. However, as part of the research, the contact discontinuity sensor was first tested for the single species cases, and some of the results are shown here in this Appendix B. Here we will consider two test cases, the Blast wave and the Le Blanc problems, which have contact discontinuities and are commonly used for sensor development in the literature. First, the Blast wave problem of

Ref. [88], with the following initial conditions

$$(\rho, u, p) = \begin{cases} (1, 0, 1000), & \text{if } 0.0 \le x < 0.1, \\ (1, 0, 0.01), & \text{if } 0.1 \le x < 0.8, \\ (1, 0, 100), & \text{if } 0.8 \le x \le 1.0, \end{cases}$$

$$(69)$$

is considered. The computational domain for this test case is x = [0, 1] and the final time is t = 0.038. The simulation is conducted with N = 400 uniformly distributed grid points, as in [6], and the results are shown in Fig. 34. The WENO5-Z scheme is used to obtain the reference solution for this case and is on a grid size of 3200.

This test case showcases the advantages of the proposed approach and the location of the sensor activation. All numerical simulations are conducted with a CFL of 0.2. All the parameters of the contact discontinuity sensor are the same as those of the 1D multi-species cases considered in Section 4. The following observations can be made:

- From Figs. 34(a) and 34(b), it can be seen that all considered schemes perform well and are without oscillations. When examining the local density profiles in the inset of both the figures, the contact discontinuity, at  $x \approx 0.6$ , is better predicted by the WENO5-Z-THINC and HY-THINC schemes than the standard WENO and MP5 schemes, respectively. These results indicate the low dissipation property of the present approach in capturing the contact discontinuities. Fig. 34(c) shows the profile of variable s, which indicates that there is a large variation in s at the contact discontinuity, which is what the sensor presented in Eq. (33) is measuring (the gradients of the variable s).
- Fig. 35(a) shows the results obtained on a grid size of 3200 using the HY-THINC scheme, and the contact discontinuity is again sharply captured without any oscillations. The sensor is shown to perform without any issues. The critical observation is that the contact discontinuity at  $x \approx 0.6$  is captured sharply at both coarse and fine grid points.
- Fig. 35(b) shows the results obtained using HY-THINC, WENO5-Z-THINC and the TENO-THINC (as implemented by the author) of Takagi et al. [30]. One can observe that the TENO-THINC scheme is activated at the contact discontinuity as indicated by the blue squares in Fig. 35(b), but it has oscillations. However, both HY-THINC and WENO5-Z-THINC are free of oscillations at

![](_page_27_Figure_2.jpeg)

**Fig. 35.** Figs. 35(a) and 35(b) show numerical solution for blast wave problem using 3200 grid points at t = 0.038 using various schemes. Figs. 35(c) and 35(d) are reproduced from [30,80], respectively.

that location, which indicates the better performance (oscillations free) and versatility (works with two different reconstruction schemes) of the proposed sensor. Fig. 35(c) was reproduced from Takagi et al. [30], with permission from Elsevier BV 2024, License number 6086631087891, does show oscillations at the exact location. Similarly, Fig. 35(d) was reproduced from Liang et al. [80], with permission from Elsevier BV 2024, License number 6086630988282, which does not capture the contact discontinuity crisply (denoted as *present* in the concerned figure) despite the claims of improvement in sub-cell resolution.

For the Le Blanc [89] test case, the initial conditions are as follows:

$$(\rho, u, p) = \begin{cases} (1.0, 0, \frac{2}{3} \times 10^{-1}), & 0 < x < 3.0, \\ (10^{-3}, 0, \frac{2}{3} \times 10^{-10}), & 3.0 < x < 9, \end{cases}$$
 (70)

and the final time of simulation is t=6. The specific heat ratio of this test case is taken as  $\frac{5}{3}$  as in [89]. Numerical results for density and velocity obtained for WENO5-Z, WENO5-Z-THINC, MP5 and HY-THINC

schemes computed on a grid size of N=900, as in [89], are shown in Fig. 36. Zero gradient boundary conditions on both ends of the domain are used for this test case.

Density profiles obtained by the WENO5-Z and MP5 schemes show more dissipative profiles than those obtained by the schemes that use THINC, respectively, specifically near the contact discontinuity at  $x \approx 6.7$ . The proposed approach captures the contact discontinuity using fewer points than the standard schemes. The discontinuity at  $x \approx 6.7$  is a jump in density as the velocity is unperturbed/flat. These figures indicate the low dissipation property of the proposed approach near the contact discontinuities. The velocity profiles of all the schemes are similar, indicating that the proposed approach improves the density profiles of the solution, leaving the other variables unperturbed.

## Data availability

No data was used for the research described in the article.

![](_page_28_Figure_2.jpeg)

**Fig. 36.** Numerical solution for Le Blanc problem with the initial conditions given by Eq. ([70\)](#page-27-2). Readers can refer to the figure for the legend.

#### **References**

- [1] Hirsch C. Numerical [computation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb1) of internal and external flows, volume 2: [computational](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb1) methods for inviscid and viscous flows. Wiley; 1990.
- [2] Batchelor GK. An [introduction](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb2) to fluid dynamics. Cambridge University Press; [1967.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb2)
- [3] Meng JC, Colonius T. Numerical simulation of the [aerobreakup](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb3) of a water droplet. J Fluid Mech [2018;835:1108–35.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb3)
- [4] Jiang G-S, Shu C-W. Efficient [implementation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb4) of weighted ENO schemes. J Comput Phys [1995;126\(126\):202–28.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb4)
- [5] Coralic V, Colonius T. [Finite-volume](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb5) WENO scheme for viscous compressible
- multicomponent flows. J Comput Phys [2014;274:95–121.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb5) [6] Hu XY, Wang Q, Adams NA. An adaptive central-upwind weighted essentially non-oscillatory scheme. J Comput Phys 2010;229(23):8952–65. [http://dx.doi.](http://dx.doi.org/10.1016/j.jcp.2010.08.019)
- [org/10.1016/j.jcp.2010.08.019](http://dx.doi.org/10.1016/j.jcp.2010.08.019). [7] Suresh A, Huynh H. Accurate [monotonicity-preserving](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb7) schemes with Runge-Kutta time stepping. J Comput Phys [1997;136\(1\):83–99.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb7)
- [8] Chamarthi AS, Frankel SH. High-order [central-upwind](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb8) shock capturing scheme using a Boundary Variation [Diminishing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb8) (BVD) algorithm. J Comput Phys [2021;427:110067.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb8)
- [9] Chamarthi AS. Gradient based [reconstruction:](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb9) Inviscid and viscous flux discretizations, shock capturing, and its application to single and [multicomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb9) flows. Comput & Fluids [2023;250:105706.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb9)
- [10] Kawai S, Shankar SK, Lele SK. [Assessment](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb10) of localized artificial diffusivity scheme for large-eddy simulation of [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb10) turbulent flows. J Comput Phys [2010;229\(5\):1739–62.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb10)
- [11] Kawai S, Terashima H. A [high-resolution](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb11) scheme for compressible mul[ticomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb11) flows with shock waves. Internat J Numer Methods Fluids [2011;66\(10\):1207–25.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb11)
- [12] Hu XY, Adams NA. Scale separation for implicit large eddy simulation. J Comput Phys 2011;230(19):7240–9. <http://dx.doi.org/10.1016/j.jcp.2011.05.023>.
- [13] Adams NA, Shariff K. A [high-resolution](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb13) hybrid compact-ENO scheme for shock-turbulence interaction problems. J Comput Phys [1996;127\(1\):27–51.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb13)
- [14] Liu X, Zhang S, Zhang H, Shu CW. A new class of central [compact](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb14) schemes with [spectral-like](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb14) resolution II: Hybrid weighted nonlinear schemes. J Comput Phys [2015;284:133–54.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb14)
- [15] Balsara DS, Garain S, Shu CW. An efficient class of WENO schemes with adaptive order. J Comput Phys 2016;326:780–804. [http://dx.doi.org/10.1016/j.jcp.2016.](http://dx.doi.org/10.1016/j.jcp.2016.09.009) [09.009](http://dx.doi.org/10.1016/j.jcp.2016.09.009).
- [16] Van Leer B. Towards the ultimate [conservative](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb16) difference scheme. IV. A new approach to numerical convection. J Comput Phys [1977;23\(3\):276–99.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb16)
- [17] Wong ML, Lele SK. High-order localized dissipation weighted compact nonlinear scheme for shockand interface-capturing in compressible flows. J Comput Phys 2017;339:179–209. [http://dx.doi.org/10.1016/j.jcp.2017.03.008.](http://dx.doi.org/10.1016/j.jcp.2017.03.008)
- [18] Nonomura T, Fujii K. Characteristic finite-difference WENO scheme for multicomponent compressible fluid analysis: Overestimated quasi-conservative formulation maintaining equilibriums of velocity, pressure, and temperature. J Comput Phys 2017;340:358–88. <http://dx.doi.org/10.1016/j.jcp.2017.02.054>.
- [19] Nonomura T, Morizawa S, [Terashima](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb19) H, Obayashi S, Fujii K. Numerical (error) issues on compressible [multicomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb19) flows using a high-order [differencing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb19) scheme: Weighted compact nonlinear scheme. J Comput Phys [2012;231\(8\):3181–210.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb19)

- [20] Balsara DS, Shu C-W. Monotonicity preserving weighted essentially nonoscillatory schemes with increasingly high order of accuracy. J Comput Phys 2000;160:405–52. <http://dx.doi.org/10.1006/jcph.2000.6443>.
- [21] Harten A. ENO schemes with subcell [resolution.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb21) J Comput Phys [1989;83\(1\):148–84.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb21)
- [22] Huynh HT. Accurate upwind methods for the Euler [equations.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb22) SIAM J Numer Anal [1995;32\(5\):1565–619.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb22)
- [23] Yang H. An artificial [compression](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb23) method for ENO schemes: the slope modification method. J Comput Phys [1990;89\(1\):125–60.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb23)
- [24] Shukla RK, Pantano C, Freund JB. An interface [capturing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb24) method for the simulation of multi-phase compressible flows. J Comput Phys [2010;229\(19\):7411–39.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb24)
- [25] Chiapolino A, Saurel R, Nkonga B. [Sharpening](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb25) diffuse interfaces with compressible fluids on unstructured meshes. J Comput Phys [2017;340:389–417.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb25)
- [26] He Z, Tian B, Zhang Y, Gao F. Characteristic-based and [interface-sharpening](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb26) algorithm for high-order simulations of immiscible compressible [multi-material](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb26) flows. J Comput Phys [2017;333:247–68.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb26)
- [27] Shyue K-M, Xiao F. An Eulerian interface [sharpening](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb27) algorithm for compressible [two-phase](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb27) flow: The algebraic THINC approach. J Comput Phys [2014;268:326–54.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb27)
- [28] Garrick DP, Hagen WA, Regele JD. An interface [capturing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb28) scheme for modeling atomization in compressible flows. J Comput Phys [2017;344:260–80.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb28)
- [29] Zhang W, Fleischmann N, Adami S, Adams NA. A hybrid [WENO5is-THINC](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb29) [reconstruction](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb29) scheme for compressible multiphase flows. J Comput Phys [2024;498:112672.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb29)
- [30] Takagi S, Fu L, Wakimura H, Xiao F. A novel high-order [low-dissipation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb30) [TENO-THINC](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb30) scheme for hyperbolic conservation laws. J Comput Phys [2022;452:110899.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb30)
- [31] Chamarthi AS. [Wave-appropriate](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb31) multidimensional upwinding approach for compressible multiphase flows. J Comput Phys [2025;114157.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb31)
- [32] Chamarthi AS, Hoffmann N, Frankel S. A wave appropriate [discontinuity](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb32) sensor approach for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb32) flows. Phys Fluids 2023;35(6).
- [33] Ducros F, Ferrand V, Nicoud F, Weber C, Darracq D, [Gacherieu](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb33) C, Poinsot T. Large-eddy simulation of the [shock/turbulence](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb33) interaction. J Comput Phys [1999;152\(2\):517–49.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb33)
- [34] Hoffmann N, Chamarthi AS, Frankel SH. Centralized [gradient-based](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb34) reconstruction for wall modelled large eddy [simulations](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb34) of hypersonic boundary layer transition. J Comput Phys [2024;113128.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb34)
- [35] [Masatsuka](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb35) K. I do like CFD, vol. 1. vol. 1, Lulu. com; 2009.
- [36] Feng Y, Schranner FS, Winter J, Adams NA. A [multi-objective](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb36) Bayesian optimization environment for systematic design of numerical schemes for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb36) flow. J Comput Phys [2022;468:111477.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb36)
- [37] Feng Y, Winter J, Adams NA, Schranner FS. A general [multi-objective](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb37) Bayesian [optimization](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb37) framework for the design of hybrid schemes towards adaptive complex flow simulations. J Comput Phys [2024;510:113088.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb37)
- [38] Feng Y, Schranner FS, Winter J, Adams NA. A deep [reinforcement](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb38) learning framework for dynamic optimization of numerical schemes for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb38) flow simulations. J Comput Phys [2023;493:112436.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb38)
- [39] Roe PL. Discrete models for the numerical analysis of [time-dependent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb39) multidimensional gas dynamics. J Comput Phys [1986;63\(2\):458–76.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb39)
- [40] Allaire G, Clerc S, Kokh S. A [five-equation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb40) model for the simulation of interfaces between compressible fluids. J Comput Phys [2002;181\(2\):577–616.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb40)

- [41] Nishikawa H. Two ways to extend diffusion schemes to [navier-stokes](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb41) schemes: Gradient formula or upwind flux. In: 20th AIAA [computational](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb41) fluid dynamics [conference](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb41) 2011. 2011, p. 27–30.
- [42] [Buchmüller](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb42) P, Helzel C. Improved accuracy of high-order WENO finite volume methods on Cartesian grids. J Sci Comput [2014;61\(2\):343–68.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb42)
- [43] Toro E. Riemann solvers and numerical methods for fluid [dynamics:](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb43) A practical [introduction.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb43) Springer Berlin Heidelberg; 2009.
- [44] Chargy D, Abgrall R, Fezoui LF, Larrouturou B. [Comparisons](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb44) of several upwind schemes for [multi-component](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb44) one-dimensional inviscid flows [Ph.D. thesis], [INRIA;](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb44) 1990.
- [45] Borges R, Carmona M, Costa B, Don WS. An [improved](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb45) weighted essentially [non-oscillatory](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb45) scheme for hyperbolic conservation laws. J Comput Phys [2008;227\(6\):3191–211.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb45)
- [46] Chamarthi AS, Hoffmann N, [Nishikawa](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb46) H, Frankel SH. Implicit gradients based conservative numerical scheme for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb46) flows. J Sci Comput [2023;95\(1\):17.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb46)
- [47] Chamarthi AS. Efficient high-order [gradient-based](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb47) reconstruction for compressible flows. J Comput Phys [2023;486:112119.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb47)
- [48] van Leer B. Towards the ultimate [conservative](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb48) difference scheme. V. A second-order sequel to Godunov's method. J Comput Phys [1979;32\(1\):101–36.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb48)
- [49] Xiao F, Ii S, Chen C. Revisit to the THINC scheme: a simple [algebraic](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb49) VOF algorithm. J Comput Phys [2011;230\(19\):7086–92.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb49)
- [50] Wakimura H, Takagi S, Xiao F. [Symmetry-preserving](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb50) enforcement of lowdissipation method based on boundary variation [diminishing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb50) principle. Comput & Fluids [2022;233:105227.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb50)
- [51] Deng X, Inaba S, Xie B, Shyue K-M, Xiao F. High fidelity [discontinuity-resolving](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb51) [reconstruction](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb51) for compressible multiphase flows with moving interfaces. J Comput Phys [2018;371:945–66.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb51)
- [52] Deng X, Shimizu Y, Xiao F. A [fifth-order](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb52) shock capturing scheme with two-stage boundary variation diminishing algorithm. J Comput Phys [2019;386:323–49.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb52)
- [53] Shu C-W. High order weighted essentially nonoscillatory schemes for convection dominated problems. SIAM Rev 2009;51(1):82–126. [http://dx.doi.org/10.1137/](http://dx.doi.org/10.1137/070679065) [070679065.](http://dx.doi.org/10.1137/070679065)
- [54] Balsara DS. Higher-order accurate space-time schemes for [computational](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb54) [astrophysics—Part](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb54) I: finite volume methods. Living Rev Comput Astrophys [2017;3\(1\):2.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb54)
- [55] [Multidimensional](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb55) upwinding. In: Handbook of numerical analysis, vol. 18, [Elsevier;](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb55) 2017, p. 53–80.
- [56] Fu L, Hu XY, Adams NA. A new class of adaptive [high-order](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb56) targeted ENO schemes for hyperbolic conservation laws. J Comput Phys [2018;374:724–51.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb56)
- [57] Song H, Ghate AS, Matsuno KV, West JR, [Subramaniam](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb57) A, Lele SK. A robust compact finite difference framework for simulations of [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb57) turbulent flows. J Comput Phys [2024;519:113419.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb57)
- [58] Li Y, Chen C, Ren Y-X. A class of [high-order](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb58) finite difference schemes with minimized dispersion and adaptive dissipation for solving [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb58) flows. J Comput Phys [2022;448:110770.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb58)
- [59] Li Q, Fu L. A family of [TENOA-THINC-MOOD](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb59) schemes based on diffuse-interface method for compressible multiphase flows. J Comput Phys [2024;519:113375.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb59)
- [60] Huang H, Li X, Fu L. A new high-order RKDG method based on the [TENO-THINC](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb60) scheme for shock-capturing. J Comput Phys [2025;520:113459.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb60)
- [61] Chamarthi AS. A generalized adaptive central-upwind scheme for compressible flow simulations and preventing spurious vortices. 2024, arXiv preprint [arXiv:](http://arxiv.org/abs/2409.02340) [2409.02340.](http://arxiv.org/abs/2409.02340)
- [62] Nishikawa H. From [hyperbolic](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb62) diffusion scheme to gradient method: Implicit Green–Gauss gradients for unstructured grids. J Comput Phys [2018;372:126–60.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb62)
- [63] Van Leer B. Upwind and [high-resolution](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb63) methods for compressible flow: From donor cell to [residual-distribution](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb63) schemes. In: 16th AIAA computational fluid dynamics [conference.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb63) 2003, p. 3559.
- [64] Abgrall R, Karni S. [Computations](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb64) of compressible multifluids. J Comput Phys [2001;169\(2\):594–623.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb64)

- [65] Johnsen E, Colonius T. [Implementation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb65) of WENO schemes in compressible multicomponent flow problems. J Comput Phys [2006;219\(2\):715–32.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb65)
- [66] Wong ML, Lele SK. Improved weighted compact nonlinear scheme for flows with shocks and material interfaces: Algorithm and assessment. In: 54th AIAA aerospace sciences meeting. (no. January):2016, p. 1807. [http://dx.doi.org/10.](http://dx.doi.org/10.2514/6.2016-1807) [2514/6.2016-1807](http://dx.doi.org/10.2514/6.2016-1807).
- [67] Shu CW, Osher S. Efficient implementation of essentially non-oscillatory shockcapturing schemes. J Comput Phys 1988;77(2):439–71. [http://dx.doi.org/10.](http://dx.doi.org/10.1016/0021-9991(88)90177-5) [1016/0021-9991\(88\)90177-5](http://dx.doi.org/10.1016/0021-9991(88)90177-5).
- [68] Lv Y, Ihme M. Discontinuous Galerkin method for [multicomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb68) chemically reacting flows and combustion. J Comput Phys [2014;270:105–37.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb68)
- [69] Abgrall R. How to prevent pressure oscillations in [multicomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb69) flow calculations: a quasi conservative approach. J Comput Phys [1996;125\(1\):150–60.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb69)
- [70] Fu L. A hybrid method with TENO based [discontinuity](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb70) indicator for hyperbolic conservation laws. Commun Comput Phys [2019;26\(4\):973–1007.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb70)
- [71] [Krivodonova](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb71) L, Xin J, Remacle J-F, Chevaugeon N, Flaherty JE. Shock detection and limiting with [discontinuous](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb71) Galerkin methods for hyperbolic conservation laws. Appl Numer Math [2004;48\(3–4\):323–38.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb71)
- [72] Zhao G-Y, Sun M-B, Pirozzoli S. On shock sensors for hybrid [compact/WENO](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb72) schemes. Comput & Fluids [2020;199:104439.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb72)
- [73] Balsara DS, Shu C-W. [Monotonicity](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb73) preserving weighted essentially nonoscillatory schemes with [increasingly](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb73) high order of accuracy. J Comput Phys [2000;160\(2\):405–52.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb73)
- [74] Yee HC, Sandham ND, Djomehri MJ. Low-dissipative high-order [shock-capturing](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb74) methods using characteristic-based filters. J Comput Phys [1999;150\(1\):199–238.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb74)
- [75] Deng X, Maekawa H. Compact high-order accurate nonlinear schemes. J Comput Phys 1997;130:77–91. <http://dx.doi.org/10.1006/jcph.1996.5553>.
- [76] Chamarthi AS, Nishikawa H, [Komurasaki](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb76) K. First order hyperbolic approach for anisotropic diffusion equation. J Comput Phys [2019;396:243–63.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb76)
- [77] Nonomura T, Iizuka N, Fujii K. Freestream and vortex [preservation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb77) properties of high-order WENO and WCNS on [curvilinear](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb77) grids. Comput & Fluids [2010;39\(2\):197–214.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb77)
- [78] Laney CB. [Computational](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb78) gasdynamics. Cambridge University Press; 1998.
- [79] Zhang W, Paula T, Bußmann A, Adami S, Adams NA. [Extension](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb79) of the hybrid [WENO5is-THINC](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb79) scheme to compressible multiphase flows with an arbitrary number of components. J Comput Phys [2025;524:113702.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb79)
- [80] Liang T, Xiao F, Shyy W, Fu L. A fifth-order [low-dissipation](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb80) discontinuityresolving TENO scheme for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb80) flow simulation. J Comput Phys [2022;467:111465.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb80)
- [81] Li Q, Lv Y, Fu L. A high-order [diffuse-interface](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb81) method with TENO-THINC scheme for compressible multiphase flows. Int J Multiph Flow [2024;173:104732.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb81)
- [82] Pan S, Han L, Hu X, Adams NA. A conservative [interface-interaction](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb82) method for compressible multi-material flows. J Comput Phys [2018;371:870–95.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb82)
- [83] Paula T, Adami S, Adams NA. A robust high-resolution [discrete-equations](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb83) method for [compressible](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb83) multi-phase flow with accurate interface capturing. J Comput Phys [2023;491:112371.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb83)
- [84] [Garcia-Villalba](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb84) M, Colonius T, Desjardins O, Lucas D, Mani A, Marchisio D, Matar OK, Picano F, Zaleski S. Numerical methods for [multiphase](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb84) flows. Int J Multiph Flow [2025;105285.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb84)
- [85] Yee HC, Sjögreen B. Simulation of [Richtmyer–Meshkov](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb85) instability by sixth-order filter methods. Shock Waves [2007;17\(3\):185–93.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb85)
- [86] Quirk JJ, Karni S. On the dynamics of a [shock–bubble](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb86) interaction. J Fluid Mech [1996;318:129–63.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb86)
- [87] Chamarthi AS. Adaptive interface capturing approach for [multicomponent](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb87) flows. Bull Am Phys Soc [2023.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb87)
- [88] Woodward P, Colella P. The numerical simulation of [two-dimensional](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb88) fluid flow with strong shocks. J Comput Phys [1984;54\(1\):115–73.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb88)
- [89] Loubère R, Shashkov MJ. A subcell [remapping](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb89) method on staggered polygonal grids for [arbitrary-Lagrangian–Eulerian](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb89) methods. J Comput Phys [2005;209\(1\):105–38.](http://refhub.elsevier.com/S0045-7930(25)00318-4/sb89)