## Accepted Manuscript

Limiter-free discontinuity-capturing scheme for compressible gas dynamics with reactive fronts

Xi Deng, Bin Xie, Raphael Loub ¨ ere, Yuya Shimizu, Feng Xiao `

PII: S0045-7930(18)30258-5

DOI: [10.1016/j.compfluid.2018.05.015](https://doi.org/10.1016/j.compfluid.2018.05.015)

Reference: CAF 3900

To appear in: *Computers and Fluids*

Received date: 12 January 2018 Revised date: 17 April 2018 Accepted date: 15 May 2018

![](_page_0_Picture_8.jpeg)

Please cite this article as: Xi Deng, Bin Xie, Raphael Loub ¨ ere, Yuya Shimizu, Feng Xiao, Limiter-free ` discontinuity-capturing scheme for compressible gas dynamics with reactive fronts, *Computers and Fluids* (2018), doi: [10.1016/j.compfluid.2018.05.015](https://doi.org/10.1016/j.compfluid.2018.05.015)

This is a PDF file of an unedited manuscript that has been accepted for publication. As a service to our customers we are providing this early version of the manuscript. The manuscript will undergo copyediting, typesetting, and review of the resulting proof before it is published in its final form. Please note that during the production process errors may be discovered which could affect the content, and all legal disclaimers that apply to the journal pertain.

## Highlights

- A novel reconstruction scheme without limiters is designed for compressible flow.
- Simplified boundary variation diminishing algorithm is devised.
- The resolution around discontinuities is significantly improved.
- Spurious waves can be prevented for stiff detonation problems.
- The new scheme is algorithmically simple.

# Limiter-free discontinuity-capturing scheme for compressible gas dynamics with reactive fronts

Xi Deng<sup>a,\*</sup>, Bin Xie<sup>b</sup>, Raphaël Loubère<sup>c</sup>, Yuya Shimizu<sup>a</sup>, Feng Xiao<sup>a</sup>

<sup>a</sup>Department of Mechanical Engineering, Tokyo Institute of Technology, 2-12-1 Ookayama, Meguro-ku, Tokyo, 152-8550, Japan.

<sup>b</sup>School of Naval Architecture, Department of Ocean and Civil Engineering, Shanghai Jiaotong University, Shanghai, 200240, China.

<sup>c</sup> Université de Bordeaux Institut de Mathématiques de Bordeaux, UMR 5251, 33 405 Talence, France.

#### **Abstract**

This work proposes a new spatial reconstruction scheme in finite volume frameworks. Different from long-lasting reconstruction processes which employ high order polynomials enforced with some carefully designed limiting projections to seek stable solutions around discontinuities, the current discretized scheme employs THINC (Tangent of Hyperbola for INterface Capturing) functions with adaptive sharpness to solve both smooth and discontinuous solutions. Due to the essentially monotone and bounded properties of THINC function, difficulties to solve sharp discontinuous solutions and complexities associated with designing limiting projections can be prevented. A new simplified BVD (Boundary Variations Diminishing) algorithm, so-called adaptive THINC-BVD, is devised to reduce numerical dissipations through minimizing the total boundary variations for each cell. Verified through numerical tests, the present method is able to capture both smooth and discontinuous solutions in Euler equations for compressible gas dynamics with excellent solution quality competitive to other existing schemes. More profoundly, it provides an accurate and reliable solver for a class of reactive compressible gas flows with stiff source terms, such as the gaseous detonation waves, which are quite challenging to other high-resolution schemes. The stiff C-J detonation benchmark test reveals that the adaptive THINC-BVD scheme can accurately capture the reacting front of the gaseous detonation, while the WENO scheme with the same grid resolution generates unacceptable results. Owing also to its algorithmic simplicity, the proposed method can become as a practical and promising numerical solver for compressible gas dynamics, particularly for simulations involving strong discontinuities and reacting fronts with stiff source term.

Keywords: Euler equations, stiff detonation waves, discontinuities, boundary variation diminishing, THINC scheme

Preprint submitted to Elsevier May 18, 2018

<sup>\*</sup>Corresponding author: Mr. X. Deng (Email: deng.x.aa@m.titech.ac.jp)

#### 1. Introduction

Flow structures containing discontinuities are found in a wide spectrum of phenomena. For example, one of the main features of high speed compressible flows, which can be described by nonlinear hyperbolic conservation laws, is that discontinuous solution such as shock waves may appear in spite of smooth initial conditions. Other types of discontinuous structures, such as the contact discontinuities in convection-dominant flows and the material interfaces between different immiscible fluids, are also widely observed. These discontinuities play crucial roles in fluid dynamics, and have thus been extensively investigated [1, 2].

Any effort to investigate the fluid dynamics using numerical simulations, however, has to face challenges to obtain accurate and stable solutions around discontinuities. For most of existing spatial reconstruction schemes in finite volume method (FVM) or finite difference method (FDM), special techniques known as the limiting projection or artificial dissipation must be designed to prevent Gibbs phenomenon and its associated spurious numerical oscillations arising from high order interpolations. However, to design such techniques is not straightforward. A large amount of work has been done to explore the possibilities of reconstruction schemes to pursue high accuracy with an essentially non-oscillatory behavior (ENO). TVD (Total Variation Diminishing) schemes, such as the MUSCL (Monotone Upstream-centered Schemes for Conservation Law) scheme [3], can resolve discontinuities without numerical oscillations by introducing slope or flux limiters. However, although TVD schemes can ensure the physical fields to be bounded and monotonic in the transition region, they typically suffer from excessive numerical dissipation.

To improve the general accuracy, high order WENO (Weighted Essentially Non-Oscillatory) schemes have been introduced in [4] and improved in [5] as well as many other works thereafter. Through assigning weights to different candidate stencils according to smoothness indicators, WENO schemes can substantially prevent numerical oscillations in the vicinity of discontinuity. However, it has been recognized in [6] that the original WENO scheme generates excessive numerical dissipation that tends to smear out contact discontinuities or jumps in variables across material interface. Since then numerous studies have been contributed to further improve the accuracy of WENO schemes. For example, a series of new smoothness indicators have been proposed in [7, 8, 9, 10] where contributions of the less smooth candidate stencils are optimized to reduce numerical dissipation. Other variants have also been devised to improve the performance of original WENO scheme. For example, in [11] numerical dissipation is reduced through employing central discretization. Combination of high order flux with a high order smooth indicator is designed to improve the accuracy around discontinuities in [12]. All these works conceptually follow the original WENO scheme and make use of polynomial interpolations.

Although the WENO-type schemes are proven to be a great success in computing smooth solutions with the highest possible order of accuracy (in terms of convergence rate) while effectively suppressing numerical oscillations

around discontinuous solutions, the intrinsic numerical dissipation still persists and pollutes the numerical solution in spite of the efforts aforementioned. As shown in the numerical tests later in this paper, the numerical dissipation continuously smears and blurs flow structures, which can be disastrous for long time evolution problems in which cases the resolution of physical features such as contact surfaces, shear waves and material interfaces evolves from bad to worse [14, 13].

More profoundly, for the problems involving reactive fronts that associate with strong mass transfer and heat release, accurately resolving the reactive fronts as well-defined sharp discontinuities is crucial to get reliable numerical results. Any numerical error or deviation in the calculation of reactive fronts will trigger unphysical mass and heat transfers which, in turn, deviate the numerical results even further away from the correct solution. For example, unphysical phenomena, in which the positions of waves are predicted incorrectly, arise in simulation of stiff detonation waves as reported in [15, 16] where the conventional high-resolution schemes, like the WENO scheme, failed in reproducing correct numerical results to stiff detonation waves unless very high grid resolution is used. As discussed in [16], existing high-resolution schemes lack enough spatial accuracy to capture truly sharp discontinuities and resolve correct stiff detonation waves.

As aforementioned, the common practice to suppress spurious numerical oscillations associated with high order polynomial-based reconstruction schemes is to introduce certain amount of numerical dissipations by projecting high order polynomials to lower order or smoother ones, which, as a consequence, essentially prevents discontinuous solutions from being resolved sharply. Realizing that the polynomial-based reconstruction may not be the proper choice when the solution includes discontinuities, several authors [17, 18, 19] have proposed other non-polynomial functions with possible better monotonicity-preserving property. It has been shown that these non-polynomial functions have better performance in capturing discontinuities with monotone distributions and are able to provide oscillation-free solutions without relying on classical limiting projections. However, these schemes still have significant numerical dissipation.

In the present work, a new spatial reconstruction scheme is proposed. Without relying on high order polynomials, the reconstruction scheme employs THINC (Tangent of Hyperbola for INterface Capturing) functions [20, 21] with adaptive sharpness to solve both smooth and discontinuous solutions. Because THINC is a bounded and monotone function, the introduction of any extra limiting projection is no more mandatory. The numerical dissipation is further reduced by boundary variation diminishing (BVD) algorithm [22]. Different from the original BVD in [22], a simplified algorithm is devised in this work, where the total boundary variations of each cell are minimized. The resultant scheme is named adaptive THINC-BVD. The performance of this new scheme is verified through solving Euler equations and reacting Euler equations with comparisons against the solutions obtained by 5th order WENO schemes. It

is observed that the proposed scheme can resolve more small-scale flow structures and prevents the occurrence of spurious waves for stiff denotation problems through a significant improvement of the resolution around discontinuities. Due to its relative simple formulation, the adaptive THINC-BVD is a possible alternative non-polynomial based scheme for simulations of flows including discontinuities, particularly for those associated with stiff sources of chemical reaction and phase change.

The outline of the paper is as follows. In Section 2, after a brief introduction of the governing equations and a review of the finite volume method in wave-propagation form, the details of the new scheme for spatial reconstruction are presented. In Section 3, numerical results of benchmark tests in 1D then in 2D are presented with a comparison with high-order schemes. Some concluding remarks and perspectives are given in Section 4.

#### 2. Numerical methods

We introduce the numerical method in one dimension for the sake of simplicity. Our numerical method can be extended to the multidimensions on structured grids directly in dimension-wise reconstruction fashion. After introduction of control equations and review of the finite volume method in the wave propagation form [23], the details about the new reconstruction scheme will be presented.

#### 2.1. Control equations

The general form which describes the time-dependent inviscid compressible flow with and without reaction in one space dimension can be written in the following form

$$\frac{\partial \mathbf{q}}{\partial t} + \frac{\partial f(\mathbf{q})}{\partial x} = \phi(\mathbf{q}),\tag{1}$$

where the vectors of physical variables  $\mathbf{q}$ , flux functions f and source terms  $\phi$  are

$$\mathbf{q} = (\rho, \rho u, E)^T, \quad f = (\rho u, \rho u u + p, E u + p u)^T, \quad \phi = (0, 0, 0)^T,$$
 (2)

for Euler equations while

$$\mathbf{q} = (\rho, \rho u, E, \rho \alpha)^T, \quad f = (\rho u, \rho u u + \rho, E u + \rho u, \rho u \alpha)^T, \quad \phi = (0, 0, 0, -K(T)\rho \alpha)^T, \tag{3}$$

for reacting Euler equations with only two chemical states. The dependent variables  $\rho$ , u, E and  $\alpha$  are the density, velocity component in x direction, total energy and mass fraction of unreacted gas, respectively. p is the pressure, T the temperature and K the chemical reaction rate. The pressure is obtained through an equation of state of the form

$$p = (\gamma - 1)(E - \frac{1}{2}\rho u^2 + \mathcal{R}),$$
 (4)

![](_page_6_Figure_1.jpeg)

Figure 1: Sketch of the finite volume representation of data in three cells  $\bar{\mathbf{q}}_i$ . Reconstruction functions are denoted by  $\tilde{\mathbf{q}}_i(x)$  and intercell values by  $\mathbf{q}_{i-1/2}^L$  and  $\mathbf{q}_{i-1/2}^R$ . A Riemann solver is used to determine the fluctuation at cell interfaces.

where  $\mathcal{R}=0$  for Euler equation and  $\mathcal{R}=-q_0\rho\alpha$  for reacting Euler equations which involves the heat release from chemical reaction processes, where  $q_0$  denotes chemical heat release and  $\gamma$  is the ratio of specific heats. The temperature is calculated by

$$T = \frac{p}{\rho}. (5)$$

For reactive Euler equations, the reaction rate can be modeled by Heaviside kinetics with

$$K(T) = -\frac{1}{\xi}H(T - T_{ign}),\tag{6}$$

where H(x) = 1 for  $x \ge 0$ , and H(x) = 0 for x < 0 and  $\xi$  represents the reaction time. Last,  $T_{ign}$  is the given ignition temperature. Generally, the stiffness issue becomes severe with such Heaviside kinetics.

## 2.2. Finite volume wave propagation method

To solve the above model system, we choose the wave propagation method which is a more flexible framework to solve hyperbolic systems in comparison with the classical finite volume method [23, 24]. Nevertheless, being a spatial reconstruction scheme for hyperbolic fluxes, adaptive THINC-BVD can be implemented in any finite volume framework straightforwardly. We divide the computational domain  $\Omega$  into N non-overlapping uniform elements or cells,  $C_i : x \in [x_{i-1/2}, x_{i+1/2}], i = 1, 2, ..., N$ , with a uniform grid with the spacing  $\Delta x = x_{i+1/2} - x_{i-1/2}$ . For a standard finite volume method, the volume-integrated average value  $\bar{q}_i(t)$  in a generic cell  $C_i$  is defined as

$$\bar{\mathbf{q}}_{i}(t) = \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \mathbf{q}(x,t) \, dx. \tag{7}$$

Denoting the spatial discretization operator for convection terms in (1) by  $\mathcal{L}(\bar{\mathbf{q}}(t))$ , the semi-discrete version of the finite volume formulation can be expressed as a system of ordinary differential equations (ODEs)

$$\frac{d\bar{\mathbf{q}}(t)}{dt} = \mathcal{L}(\bar{\mathbf{q}}(t)), \tag{8}$$

In the wave-propagation method, the spatial discretization operator for convection terms in cell  $C_i$  is computed by

$$\mathcal{L}(\bar{\mathbf{q}}_{i}(t)) = -\frac{1}{\Delta x} \left( \mathcal{A}^{+} \Delta \mathbf{q}_{i-1/2} + \mathcal{A}^{-} \Delta \mathbf{q}_{i+1/2} + \mathcal{A} \Delta \mathbf{q}_{i} \right)$$
(9)

where  $\mathcal{A}^+\Delta\mathbf{q}_{i-1/2}$  and  $\mathcal{A}^-\Delta\mathbf{q}_{i+1/2}$ , are the rightand left-moving fluctuations, respectively, which enter into cell  $C_i$ , and  $\mathcal{A}\Delta\mathbf{q}_i$  is the total fluctuation within  $C_i$ . At any interface between two cells  $C_{i-1}$  and  $C_i$ , discontinuous data is to be solved. A Riemann problem is then solved to determine these fluctuations. The rightand left-moving fluctuations can be calculated by

$$\mathcal{H}^{\pm} \Delta \mathbf{q}_{i-1/2} = \sum_{k=1}^{3} \left[ s^{k} \left( \mathbf{q}_{i-1/2}^{L}, \mathbf{q}_{i-1/2}^{R} \right) \right]^{\pm} \mathcal{W}^{k} \left( \mathbf{q}_{i-1/2}^{L}, \mathbf{q}_{i-1/2}^{R} \right), \tag{10}$$

where moving speeds  $s^k$  and the jumps  $W^k$  (k = 1, 2, 3) of three propagating discontinuities can be solved by Riemann solvers [25] given the pointwise reconstructed values  $\mathbf{q}_{i-1/2}^L$  and  $\mathbf{q}_{i-1/2}^R$  which are computed from the reconstruction functions  $\tilde{\mathbf{q}}_{i-1}(x)$  and  $\tilde{\mathbf{q}}_i(x)$  to the left and right sides of cell edge  $x_{i-1/2}$ , respectively. The reconstruction process in finite volume method is illustrated in Fig. 1. In our simulations, the HLLC Riemann solver [25] is employed. Similarly, the total fluctuation can be determined by

$$\mathcal{A}\Delta\mathbf{q}_{i} = \sum_{k=1}^{3} \left[ s^{k} \left( \mathbf{q}_{i-1/2}^{R}, \mathbf{q}_{i+1/2}^{L} \right) \right]^{\pm} \mathcal{W}^{k} \left( \mathbf{q}_{i-1/2}^{R}, \mathbf{q}_{i+1/2}^{L} \right). \tag{11}$$

Given the spatial discretization, the three-stage third-order SSP (Strong Stability-Preserving) Runge-Kutta scheme [26] is employed for time marching. The Euler first-order explicit and second order Runge-Kutta schemes with proposed spatial discretized scheme also produce stable numerical results as the third order SSP Runge-Kutta does. However it is observed that the results from first and second-order temporal schemes are slightly diffusive. In the following subsection we will describe thoroughly the reconstructions used to get values,  $\mathbf{q}_{i-1/2}^L$  and  $\mathbf{q}_{i-1/2}^R$ , at cell boundaries. This is the core of our work.

## 2.3. Adaptive THINC-BVD reconstruction

In this subsection, the process to reconstruct the boundary values  $\mathbf{q}_{i+1/2}^L$  and  $\mathbf{q}_{i-1/2}^R$  is presented. We denote any single variable for reconstruction by q, which can be one of the primitive, conservative or characteristic variable. In the present work, the reconstruction applies to the primitive variables.

Instead of polynomial functions, the present reconstruction process makes use of the hyperbolic tangent function in the THINC method [20, 21], which is a differentiable and monotone Sigmoid function. The piecewise THINC

reconstruction function is written as

$$\tilde{q}_i(x) = \bar{q}_{min} + \frac{\bar{q}_{max}}{2} \left[ 1 + \theta \tanh \left( \beta \left( \frac{x - x_{i-1/2}}{x_{i+1/2} - x_{i-1/2}} - \tilde{x}_i \right) \right) \right],$$
(12)

where  $\bar{q}_{min} = \min(\bar{q}_{i-1}, \bar{q}_{i+1})$ ,  $\bar{q}_{max} = \max(\bar{q}_{i-1}, \bar{q}_{i+1}) - \bar{q}_{min}$  and  $\theta = sgn(\bar{q}_{i+1} - \bar{q}_{i-1})$ . The jump thickness is controlled by the parameter  $\beta$ , i.e. a small value of  $\beta$  leads to a smooth profile while a large one leads to a sharp jump-like distribution. The unknown  $\tilde{x}_i$ , which represents the location of the jump center, is computed from constraint condition  $\bar{q}_i = \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \tilde{q}_i(x) \, dx$ .

Since the value given by hyperbolic tangent function  $\tanh(x)$  lays in the region of [-1,1], the value of THINC reconstruction function  $\tilde{q}_i(x)$  is rigorously bounded by  $\bar{q}_{i-1}$  and  $\bar{q}_{i+1}$ . Thus a limiting process to avoid extrema generated by the reconstructions is not needed anymore. Given the reconstruction function  $\tilde{q}_i(x)$ , we calculate the boundary values  $q_{i+1/2}^L$  and  $q_{i-1/2}^R$  by  $q_{i+1/2}^L = \tilde{q}_i(x_{i+1/2})$  and  $q_{i-1/2}^R = \tilde{q}_i(x_{i-1/2})$  respectively.

As demonstrated in our previous works, adjusting the value of parameter  $\beta$  can modify the sharpness of the jump transition in the THINC reconstruction. In order to elucidate the effect of the sharpness parameter  $\beta$  on numerical dissipation of the THINC scheme, we applied the approximate dispersion relation (ADR) analysis described in [27] to the THINC scheme with different wavenumbers w. The spectral properties of THINC schemes with different  $\beta$ are shown in Fig. 2, in which the numerical dissipation is quantified through the imaginary parts of the modified wavenumber. It can be seen that dissipation errors decrease as  $\beta$  increases from 1.0 to 1.3. However, it is observed that in a band of wavenumbers the THINC scheme with  $\beta = 1.2$  or  $\beta = 1.3$  produces values larger than zero, which implies that the THINC scheme with these  $\beta$  values may modulate the waves in this band. It is observed as the compressive or anti-diffusion effect in the numerical results. In order to compare with other popularly used schemes, we also show numerical dissipations of the TVD schemes with Minmod, Van Leer and Superbee limiters [3, 28]. As shown in the right side of Fig. 2, the THINC scheme with  $\beta_s = 1.1$  has much smaller numerical dissipation than TVD scheme with Minmod limiter, and has similar but slightly better performance than the Van Leer limiter. THINC with a larger slope parameter ( $\beta_s = 1.3$ ) has the similar spectral property to the TVD method with Superbee limiter. They both show a positive imaginary part at low wavenumber band, which leads to the well-known squaring effect [27] on the solution profile. This squaring effect is preferred for discontinuous solutions. The above analysis reveals that it is possible to design a reconstruction strategy for smooth and discontinuous solutions by adaptively choosing the sharpness parameter  $\beta$ . We present next such an algorithm using the BVD principle.

For each cell i, we prepare two THINC reconstruction functions with different values of slope parameter  $\beta$ . A THINC reconstruction function  $\tilde{q}_i^s(x)$  with a small  $\beta_s$  is prepared to represent a smooth solution, which produces the reconstruction values  $q_{i+1/2}^{L,s}$  and  $q_{i-1/2}^{R,s}$ . To capture discontinuities, another THINC reconstruction function  $\tilde{q}_i^l(x)$  with

a large  $\beta_l$  is employed as an alternative candidate for reconstruction. The values at cell boundaries  $q_{i+1/2}^{l,l}$  and  $q_{i-1/2}^{R,l}$  are then obtained from  $\tilde{q}_i^l(x)$  accordingly.

The final reconstruction function is adaptively determined by a new version of BVD algorithm, see an earlier version in [22]. Let us define the total boundary variations (TBV) for a target cell *i* as

$$TBV_{i}^{p} = \left| q_{i-1/2}^{L,p} - q_{i-1/2}^{R,p} \right| + \left| q_{i+1/2}^{L,p} - q_{i+1/2}^{R,p} \right|, \qquad p = \{s, l\}.$$
 (13)

Then the final reconstruction function is determined by

$$\tilde{q}_{i}^{f}(x) = \begin{cases} \tilde{q}_{i}^{s}(x) & \text{if } TBV_{i}^{s} < TBV_{i}^{l}, \\ \tilde{q}_{i}^{l}(x) & \text{otherwise} \end{cases}, \tag{14}$$

by which the reconstruction values at cell boundaries can be calculated. The resultant scheme is named adaptive THINC-BVD, which has the following characteristics:

- (I) Since THINC functions with adaptive sharpness are employed as the spatial discretization to deal with both smooth and discontinuous solutions, extra limiting process is not needed to prevent numerical oscillations;
- (II) Numerical dissipation is reduced by minimizing the total boundary variation for each cell. Different from the BVD algorithm in [22], the current procedure simplifies the algorithm significantly and thus can be extended to unstructured grids.

It is noted that some arbitrariness is left in choosing  $\beta_s$  and  $\beta_l$ . Based on the numerical dissipation analysis aforementioned, we use  $\beta_s = 1.1$  to resolve the smooth solutions. A larger value can be chosen for  $\beta_l$  to represent a jump-like solution. In this paper  $\beta_l = 2.0$  is used for all numerical tests. From our numerical experiments, a  $\beta_l$  valued from 1.6 to 2.2 produces good or acceptable results.

In spite of its simplicity, the adaptive THINC-BVD scheme can produce accurate results for both smooth and discontinuous solutions competitive to most existing high-resolution schemes. We present numerical verifications in the following section.

#### 3. Numerical results

In this section, some comparative tests in oneand twodimensions are conducted between the 5th order WENO scheme [5] and the proposed adaptive BVD-THINC scheme. The reconstruction process is conducted in terms of primitive variables and the ratio of specific heats is set to  $\gamma = 7/5$ .

![](_page_10_Figure_1.jpeg)

Figure 2: Imaginary parts of modified wavenumber from THINC schemes with different  $\beta_s$ . A comparison with the TVD scheme is also included. The left one shows the whole resolvable wavenumber whereas the right one shows the zoomed region of the small wavenumber.

#### 3.1. Accuracy test for density propagation waves

In order to investigate the accuracy of the proposed scheme, propagating density wave is studied in this test. The initial pressure and velocity field are set constant as  $p_0 = 1.0$  and  $u_0 = 1.0$ , while an initial perturbation of the density field is given by  $\rho_0 = \sin(\pi x) + 1.0$ . The computation domain is [-1,1] and periodic boundary conditions are applied to both left and right ends. The  $L_1$ ,  $L_\infty$  errors and corresponding convergence rates are calculated on gradually refined meshes after one period at t = 2.0 with CFL=0.4. Comparisons are made amongst the proposed adaptive THINC-BVD, TVD schemes with different limiters and the 5th order accurate WENO scheme. We observe in Table. 1 that adaptive THINC-BVD scheme has much smaller errors in both  $L_1$  and  $L_\infty$  norms compared to TVD schemes with Minmod, Van Leer or Superbee limiters. Moreover, the proposed scheme shows a uniform 2nd order convergence rate in  $L_1$  for this test with smooth solution. The WENO scheme demonstrates higher-order accuracy and less numerical errors for the smooth solution. As showcased in [22], by using WENO or other high-order schemes as one of the candidates for reconstruction, the BVD algorithm can realize higher-order and more accurate numerical results for smooth solutions. Nevertheless, as shown in following numerical tests, adaptive THINC-BVD will outperform high order WENO schemes when solving problems involving discontinuities.

## 3.2. Sod's problem

As one of famous benchmark tests for shock-capturing schemes, the Sod problem is employed here to test the performance of schemes in capturing the shock front as well as the contact discontinuity. The initial distribution on

Table 1: Numerical errors and convergence rate for density propagation waves problem. Comparisons are made among different schemes.

| Schemes   | Mesh | $L_1$ errors           | $L_1$ order | $L_{\infty}$ errors    | $L_{\infty}$ order |
|-----------|------|------------------------|-------------|------------------------|--------------------|
| Minmod    | 40   | $4.547 \times 10^{-2}$ |             | $1.025 \times 10^{-1}$ |                    |
|           | 80   | $1.337 \times 10^{-2}$ | 1.77        | $4.347 \times 10^{-2}$ | 1.23               |
|           | 160  | $3.812 \times 10^{-3}$ | 1.81        | $1.795 \times 10^{-2}$ | 1.27               |
|           | 320  | $1.031 \times 10^{-3}$ | 1.89        | $7.298 \times 10^{-3}$ | 1.30               |
| Van Leer  | 40   | $2.101 \times 10^{-2}$ |             | $5.151 \times 10^{-2}$ |                    |
|           | 80   | $5.568 \times 10^{-3}$ | 1.92        | $1.952 \times 10^{-2}$ | 1.40               |
|           | 160  | $1.408 \times 10^{-3}$ | 1.98        | $7.302 \times 10^{-3}$ | 1.42               |
|           | 320  | $3.423 \times 10^{-4}$ | 2.04        | $2.715 \times 10^{-3}$ | 1.43               |
| Superbee  | 40   | $2.134 \times 10^{-2}$ |             | $6.087 \times 10^{-2}$ |                    |
|           | 80   | $9.024 \times 10^{-3}$ | 1.24        | $3.443 \times 10^{-2}$ | 0.82               |
|           | 160  | $2.642 \times 10^{-3}$ | 1.77        | $1.487 \times 10^{-2}$ | 1.21               |
|           | 320  | $7.159 \times 10^{-4}$ | 1.88        | $6.651 \times 10^{-3}$ | 1.16               |
| THINC-BVD | 40   | $1.518 \times 10^{-2}$ |             | $4.721 \times 10^{-2}$ |                    |
|           | 80   | $3.766 \times 10^{-3}$ | 2.01        | $1.821 \times 10^{-2}$ | 1,37               |
|           | 160  | $8.969 \times 10^{-4}$ | 2.07        | $6.866 \times 10^{-3}$ | 1.41               |
|           | 320  | $2.198 \times 10^{-4}$ | 2.03        | $2.545 \times 10^{-3}$ | 1.43               |
| WENO      | 40   | $4.473 \times 10^{-5}$ |             | $8.799 \times 10^{-5}$ |                    |
|           | 80   | $1.396 \times 10^{-6}$ | 5.00        | $2.822 \times 10^{-6}$ | 4.96               |
|           | 160  | $4.361 \times 10^{-8}$ | 5.00        | $8.487 \times 10^{-8}$ | 5.06               |
|           | 320  | $1.361 \times 10^{-9}$ | 5.00        | $2.544 \times 10^{-9}$ | 5.06               |

computational domain [0, 1] is specified as [29]

$$(\rho_0, u_0, p_0) = \begin{cases} (1, 0, 1) & 0 \le x \le 0.5\\ (0.125, 0, 0.1) & \text{otherwise} \end{cases}$$
 (15)

The computation is carried out with 200 uniform cells and with time integrated up to t = 0.25. The comparative numerical results calculated from WENO scheme and the proposed solver are shown in Fig. 3 for density and pressure fields respectively. From the results, it can be seen that standard 5th order WENO scheme can solve both the shock wave and contact discontinuity without numerical oscillations. However, just like other standard shock-capturing schemes, the WENO schemes present difficulties to sharply solve discontinuous linearly degenerate fields which correspond to contact discontinuities. However it can get satisfying results for the shock since shock waves have the compressive character. On the contrary, the proposed scheme can solve contact discontinuities on one or two cells only. Moreover, it reduces the numerical diffusion around shock as well. These results show that the proposed scheme has good performance in capturing discontinuous solutions for this only midly difficult test.

![](_page_12_Figure_1.jpeg)

Figure 3: Numerical results of Sod's problem for density field (top) and pressure (bottom) at t = 0.25 with 200 cells. Comparisons are made between the proposed scheme (left panels) and WENO scheme (right panels) and the exact solution (straight line).

![](_page_13_Figure_1.jpeg)

Figure 4: Numerical results of Lax's problem for density field at time t = 0.16 with 200 cells. Comparisons are made between the proposed THINC-BVD scheme (left panel) and WENO scheme (right panel) against the exact solution (straight line).

#### 3.3. Lax's problem

The Lax problem is employed here to check the ability of the proposed numerical scheme to capture relatively strong shock [30]. The initial condition is given by

$$(\rho_0, u_0, p_0) = \begin{cases} (0.445, 0.698, 3.528) & 0 \le x \le 0.5\\ (0.5, 0, 0.571) & \text{otherwise} \end{cases}$$
 (16)

With the same number of cells as in previous case, the computation lasts until time t = 0.16. From the result shown in Fig. 4, we can see that the contact discontinuous solution can be resolved sharply while a much diffused profile is produced by WENO scheme since around strong discontinuities WENO scheme, just like other high order polynomial-based reconstruction schemes, does sacrifice some accuracy preventing the occurence of spurious numerical oscillations.

#### 3.3.1. Collela-Woodward blastwaves

In [31] Collela and Woodward introduced their famous blastwave involving multiple interactions of strong shocks and rarefaction waves. The initial distribution has uniform density of  $\rho = 1$  and zero velocity. The domain  $\Omega = [0, 1]$  is split into three parts  $\Omega_L = [0, 0.1]$ ,  $\Omega_M = [0.1, 0.9]$  and  $\Omega_C = [0.9, 1]$  for which the constant pressure values are  $p_L = 1000$ ,  $p_M = 0.01$  and  $p_R = 100$ . Reflective boundary conditions are imposed at the two ends of computational domain. Two blast waves are generated by the initial jumps and then evolve and violently interact. Classically 400 mesh cells are used in the literature for this test problem when high order schemes are tested. We depict the numerical density at time t = 0.038 in Fig. 5 against a reference solution obtained with WENO scheme with a very fine mesh.

![](_page_14_Figure_1.jpeg)

Figure 5: Numerical results of two interacting blast waves problem for density field at t = 0.038 with 400 cells. Comparisons are made between the proposed scheme (left), WENO scheme (middle) and a discrete staggered compatible Lagrangian scheme (right).

The solution produced by WENO scheme for 400 cells presents discontinuities, contact and shock which are smeared, especially the first contact discontinuity around  $x \approx 0.6$ . On the contrary the proposed scheme can capture this contact discontinuity with only three points, while polynomial based high order scheme may demand several tenths of cells. For comparison purposes we also provide the results obtained by a discrete staggered compatible Lagrangian scheme on the right panel of Fig. 5, see [32]. Such mesh moving (Lagrangian) numerical method preserves exactly the contact discontinuities because the mesh moves with the fluid velocity. Therefore this scheme is one of the most accurate tools to capture those contact discontinuities. It is seen that the proposed BVD scheme's results compare well with Lagrangian ones<sup>1</sup>.

## 3.3.2. Sedov blast wave

We verify the robustness of the present scheme on Sedov blast wave which involves low density and low pressure. The initial condition is set as the same as [33, 34] in which  $(\rho_0, u_0, p_0)$  is  $(1.0, 0.0, 10^{-12})$  everywhere except that the energy in the center cell is given by the constant  $\frac{E_0}{\Delta x}$  with  $E_0 = 3.2 \times 10^6$ . Such high energy deposition generates strong left and right moving shocks, followed by exponential decay of density and pressure. Those exponential decays leads to almost near vacuum region in the center of the domain. The mesh size is  $\Delta x = \frac{1}{200}$  and the mesh is uniform. Since this problem involves near-vacuum condition, numerical oscillations given by the high order reconstruction schemes may break up the computation. Thus positivity-preserving techniques are usually demanded to preserve positivity of density. Without relying on high order polynomials as reconstruction function, the present scheme is able to solve this near-vacuum state without any extra positivity-preserving technique. The computational result is presented in the Fig. 6 where we plot the density and pressure vs the exact solution. Compared with the Fig. 5.1 in [33] the results of

<sup>&</sup>lt;sup>1</sup>Note that the Lagrangian results present an overshoot of three cells at about x = 0.765 of maximal value  $\rho = 9.5$  which are omited here because it does not alter our comments.

![](_page_15_Figure_1.jpeg)

Figure 6: Numerical results of Sedov blast wave problems. The solution of density field is presented on the left while pressure field on the right vs the exact solution (straight line).

which are obtained by a positivity-preserving WENO scheme, the present scheme produces sharper solutions around the shocks without the need to add extra dissipation for the near-vaccum states.

## 3.4. C-J detonation wave with the Heaviside model

In this example, the C-J detonation for which the chemical reaction is modeled by a simple Heaviside function is considered. Likewise as in [35, 36, 15], the parameter values of the model are  $\frac{1}{\xi} = 0.5825 \times 10^{10}$ ,  $\gamma = 1.4$ ,  $q_0 = 0.5196 \times 10^{10}$  and  $T_{ign} = 0.1155 \times 10^{10}$ . The computational domain is [0, 0.05] and the initial discontinuity is set at x = 0.005. The totally burnt gas is set on the right side with the initial state:  $\rho_0 = 1.201 \times 10^{-3}$ ,  $u_0 = 0.0$ and  $p_0 = 8.321 \times 10^5$ . The totally unburnt gas is set on the left side where  $(\rho_{CJ}, u_{CJ}, p_{CJ}, 0.0)$  are determined by the C-J detonation model [37, 38]. The computation is performed with uniform mesh of cell number N = 300 until final time  $t = 3 \times 10^{-7}$ . At this time the exact position of the burning wave is at x = 0.03764 and the states to the left and right are known. When dealing with such stiff hyperbolic system, standard shock-capturing schemes with insufficient grid resolution may produce incorrect propagation speed and non-physical spurious waves even with enough temporal resolution. Due to the presence of large numerical dissipation around the discontinuity for classical shock-capturing schemes, the chemical reaction may be triggered too early in the adjacent cell of an numerically smeared discontinuity for which the temperature profile contains the value above the ignition temperature. If the reaction is fast, the wrongly triggered chemical reaction does shift the discontinuity, and consequently produces non-physical reflected or refracted waves and a wrong chronometry for physical ones. To verify that our scheme can prevent the creation of non-physical spurious waves by virtue of its low numerical dissipation around discontinuity, we solve this problem by setting the CFL= 0.01 for the 5th order WENO scheme, while we fix the CFL= 0.1 for the proposed scheme.

The results of density, temperature and mass fraction are plotted in Fig. 7 where the reference solution is calculated by the standard 5th order WENO scheme with N = 10000 cells. We observe that while the proposed approach can correctly capture the wave front, WENO presents the previously described drawbacks even with smaller CFL numbers. For WENO scheme, CFL= 0.1 number will make spurious phenomena more severe. In fact, as reported in [15] wrong results are produced by standard WENO schemes no matter how small the time step is since the stiffness problem is due to spatial rather than the temporal errors. It is noted that being the benchmark case to test numerical methods, the stiff detonation model used here presents more severe conditions than the physically realizable cases. Having verified with the stiff C-J detonation wave test, the proposed scheme also works well for non-stiff problems which have more physical significance.

#### 3.5. Interaction between a detonation wave and an oscillatory profile

Interaction between a detonation wave and an oscillatory profile is considered in this section, see [39, 15]. The parameters of the model are  $\gamma = 1.2$ ,  $q_0 = 50$ ,  $\frac{1}{\xi} = 1000$  and  $T_{ign} = 3$ . The computational domain is set as  $[0, 2\pi]$  divided by N = 200 uniform cells. The initial state is

$$(\rho_0, u_0, p_0, \alpha_0) = \begin{cases} (1.79463, 3.0151, 21.53134, 0.0) & x \le \frac{\pi}{2} \\ (1.0 + 0.5\sin(2x), 0.0, 1.0, 1.0) & \text{otherwise} \end{cases}$$
 (17)

The simulation is run to final time  $t = \frac{\pi}{5}$  with CFL= 0.1. The numerical results of density, temperature and mass fraction fields are plotted in Fig. 8. We observe that the proposed scheme can effectively prevent the occurrence of spurious waves which are produced by standard shock capturing WENO scheme. Also, the complicated flow field produced by the interaction between the detonation wave and the oscillatory profiles can be sharply resolved by the proposed scheme, while the effect of wrong wave computation does dramatically pollute WENO results.

This test concludes our 1D validation tests where we have seen that excessive numerical dissipation can smear discontinuities (Sod, Lax, blastwave problems) with a possible dramatic effect on the capability of the numerical method to capture the correct physical phenomena (detonation tests), or to complete the simulation (Sedov problem). The proposed scheme seems to perform well for these 1D situations. It is also noted that the BVD algorithm presented in this paper is algorithmically simpler compared to the previous work [22], and can be implemented to unstructured grids straightforwardly.

#### 3.6. 2D Riemann problems

A set of two-dimensional Riemann problems which have been proposed and extensively studied in [1, 40] is employed to verify that the behaviors of the proposed THINC-BVD scheme on Euler equations. This test is employed

![](_page_17_Figure_1.jpeg)

Figure 7: Numerical results of density, temperature and mass fraction fields for C-J detonation wave with the Heaviside chemical reaction model. Reference solutions are represented by black solid lines. Comparisons are made between the WENO (right panels) and the proposed scheme (left panels).

![](_page_18_Figure_1.jpeg)

Figure 8: Numerical results of density, temperature and mass fraction fields for C-J detonation wave with the Heaviside chemical reaction model for the interaction between a detonation wave and an oscillatory profile problem. Reference solutions are represented by black solid lines. Comparisons are made between the WENO (right panels) and the proposed scheme (left panels).

to prove that the extension of the THINC-BVD scheme to two dimensions can produce accurate, robust and non-oscillatory solutions. Involving different configurations based on an initial condition made of  $2 \times 2$  constant states, the 2D Riemann problems are usually employed as benchmark tests to examine the behavior of new schemes. Note that most of the interfaces between those states are Kelvin-Helmholtz instable. As a consequence it has been reported by several works [41, 14, 42, 43, 44, 45] that the sharp capture of discontinuities along with the resolution of small-scale features generated by the Kelvin-Helmholtz instability does require fine enough grids or very high order schemes. The computational domain is set to  $[-0.5, 0.5] \times [-0.5, 0.5]$ , and four different configurations are studied in this work, the initial conditions of which are as follows,

Riemann problem #1:

$$(\rho_0, u_0, v_0, p_0) = \begin{cases} (2.0, 0.75, 0.5, 1.0) & x \le 0.0, y \ge 0.0 \\ (1.0, -0.75, 0.5, 1.0) & x < 0.0, y < 0.0 \\ (1.0, 0.75, -0.5, 1.0) & x > 0.0, y > 0.0 \\ (3.0, -0.75, -0.5, 1.0) & x > 0.0, y < 0.0 \end{cases}$$
(18)

Riemann problem #2:

$$(\rho_0, u_0, v_0, p_0) = \begin{cases} (2.0, -0.75, 0.5, 1.0) & x \le 0.0, y \ge 0.0 \\ (1.0, 0.75, 0.5, 1.0) & x < 0.0, y < 0.0 \\ (1.0, -0.75, -0.5, 1.0) & x > 0.0, y > 0.0 \\ (3.0, 0.75, -0.5, 1.0) & x > 0.0, y < 0.0 \end{cases}$$

$$(19)$$

Riemann problem #3:

$$(\rho_0, u_0, v_0, p_0) = \begin{cases} (0.5323, 1.206, 0.0, 0.3) & x \le 0.3, y \ge 0.3 \\ (0.138, 1.206, 1.206, 0.029) & x < 0.3, y < 0.3 \\ (1.5, 0.0, 0.0, 1.5) & x > 0.3, y > 0.3 \\ (0.5323, 0.0, 1.206, 0.3) & x > 0.3, y < 0.3 \end{cases}$$
(20)

Riemann problem #4:

$$(\rho_0, u_0, v_0, p_0) = \begin{cases} (0.5197, -0.6259, 0.1, 0.4) & x \le 0.0, y \ge 0.0 \\ (0.8, 0.1, 0.1, 0.4) & x < 0.0, y < 0.0 \\ (1.0, 0.1, 0.1, 1.0) & x > 0.0, y > 0.0 \\ (0.5197, 0.1, -0.6259, 0.4) & x > 0.0, y < 0.0 \end{cases}$$
(21)

A uniform grid made of  $600 \times 600$  cells is employed in all calculation. The numerical results calculated by the

proposed scheme and the WENO scheme are all presented in Fig. 9. The density variable contours are plotted with an identical color scale for both schemes. The reference results can be found in [45] where grid of 1200 × 1200 and six order WENO schemes are used to solve those 2D Riemann problems. From the results of Fig. 9 we can first observe that the two schemes produce comparable numerical solutions for the main flow structures. However the proposed scheme does capture sharper discontinuities than WENO scheme. Moreover the occurrence and the number of small scale structures are more pronounced by THINC-BVD. Contrarily to the statement made in [14] that the number of vortices will depend on the order of the scheme and cell size, we show that a supposedly low-order scheme such as THINC-BVD can nonetheless significantly reduce the numerical viscosity around discontinuities, and, also, can produce more small-scale flow structures. In addition, the performance of proposed scheme is evaluated through the comparison of the CPU times of three methods: a second-order classical TVD scheme, the 5th order WENO and the current THINC-BVD scheme. All methods have been developed under the same framework by the same developer so that this comparison is rather fair.

In Fig 10 we gather the CPU times and the ratio between those CPU times. It can be seen that on average and consistently THINC-BVD is about 2 times more expensive than TVD, while WENO costs more than 5 times. As a consequence WENO is nearly 3 times slower than THINC-BVD. The results substantiate that the proposed THINC-BVD scheme improves the numerical accuracy at a very low computational cost leading to a genuine appealing computational efficiency.

![](_page_21_Figure_1.jpeg)

 $Figure \ 9: \ Density \ contour \ for \ Riemann \ problems \ 1, 2, 3, 4. \ Left: \ 5th \ order \ WENO \ scheme \ results. \ Right: \ Adaptive \ THINC-BVD \ scheme \ results.$ 

![](_page_22_Figure_1.jpeg)

Figure 10: CPU times recorded for Riemann problems #1, #2, #3, and #4 simulated by the TVD, WENO and BVD schemes. Comparisons are made amongst TVD, adaptive BVD and WENO schemes. Histograms (left axis): CPU time — Symbol-lines (right axis): ratio between the CPU times.

![](_page_23_Figure_1.jpeg)

Figure 11: Density contours for double Mach reflection at time t = 0.2 with grid resolution  $\Delta x = \Delta y = \frac{1}{100}$  (top panels) and  $\Delta x = \Delta y = \frac{1}{200}$  (bottom panels). The figures are drawn with 30 density contours between 1.9 and 21.0

## 3.7. Double Mach reflection

A Mach 10 hypersonic propagating planar shock reflected by 30° ramp [46] is simulated in this test case. Although there exists no exact solution, this test has been adopted as a standard benchmark test to assess the ability of a numerical scheme to capture accurately both the strong reflected and refracted shocks as well as the small-scale structures in the re-circulation zone. The low dissipation of a numerical scheme can be visually estimated by the number vortex structures resulting from Kelvin-Helmholtz instabilities along the slip line in the re-circulation zone. The numerical schemes with large numerical viscosity tend to smear out these structures. The computational domain is  $[0,3.2] \times [0,1]$ . A right-moving Mach 10 shock is imposed with 60° angle relative to *x*-axis. At the right boundary of the computational domain, the boundary condition is given by setting all gradients to be zero. The solution is computed up to time t = 0.2 with two different grid resolutions,  $\Delta x = \Delta y = \frac{1}{100}$  for the top panels of Fig. 11 and  $\Delta x = \Delta y = \frac{1}{200}$  for the bottom ones. The results are in good agreement with those computed with high order schemes. For instance, compared with the Fig. 17 in [47] where the computation is conducted with several high order schemes with grid size  $\Delta x = \Delta y = \frac{1}{240}$ , the present scheme seems to resolve more vortices along the slip line. These results show that without relying on high order polynomial reconstructions, the proposed scheme can nonetheless reach high resolution capability (i.e. low dissipation), comparably with the state-of-the-art high order schemes.

#### 3.8. 2D detonation waves

Two dimensional detonation waves problem which was also investigated in [35, 15, 16] is considered here. A twodimensional shock tube with  $[0,0.025] \times [0,0.005]$  is set as computational domain. Reflective boundary conditions are prescribed to upper and lower boundaries, while zero-gradient boundary conditions are imposed to the left and the right boundaries. The parameters  $q_0$ ,  $\frac{1}{\xi}$  and  $T_{ign}$  in Heaviside chemical reaction model are same as in Subsection 3.4. The initial conditions are

$$(\rho_0, u_0, v_0, p_0, \alpha) = \begin{cases} (\rho_l, u_l, 0, p_l, 0) & \text{if } x \le \psi(y) \\ (\rho_r, u_r, 0, p_r, 1) & \text{if } x > \psi(y), \end{cases}$$
(22)

where

$$\psi(y) = \begin{cases} 0.004 & \text{if } |y - 0.0025| \ge 0.001\\ 0.005 - |y - 0.0025| & \text{if } |y - 0.0025| < 0.001. \end{cases}$$
(23)

The right states  $(\rho_r, u_r, 0, p_r, 1)$  are the same as in Subsection 3.4 and  $\rho_l = \rho_{CJ}$ ,  $p_l = p_{CJ}$  while  $u_l = 8.162 \times 10^4 > u_{CJ}$ . As stated in [48], one important feature of this solution is that a cellular pattern will form after the triple point has traveled in the transverse direction and hereafter reflected back and forth against the upper and lower boundaries.

Because there exists no known exact solution to this problem, we have simulated a fine solution obtained by the 5th order WENO scheme on  $2000 \times 400$  mesh cells, see left-panels of Fig. 12-13. The numerical density is displayed on Fig. 12 for WENO (middle panels) and BVD (right panels) schemes at time  $t = 0.3 \times 10^{-7}$  on top and  $t = 1.7 \times 10^{-7}$  on bottom. A 3D elevation plot and a projected 2D plot are simultaneously presented where color and elevation represent the density. The same scale is used for all panels. From this figure we observe that the coarse grid WENO results present spurious oscillations which are not visible on THINC-BVD results and very less pronounced on the fine WENO solution. Moreover some of the finer structures seen just after the up-front reaction front seem more accurately captured by THINC-BVD. More importantly we can see that the reaction front structure is incorrectly captured by WENO which smears excessively the wave front. On Fig. 13 we propose a different angle of visualization to emphasize this front structure. This numerical diffusion of the front has an important impact on the form of the resulting fine structures. BVD results seem to be less impacted by numerical dissipation and spurious phenomena than WENO for this test case.

#### 4. Conclusion remarks

An alternative spatial reconstruction scheme named adaptive THINC-BVD is proposed in this work in the context of finite volume schemes to solve hyperbolic systems of PDEs. Instead of relying on high order polynomial reconstructions, the current scheme employs THINC (hyperbolic tangent shape) function with adaptive sharpness and a newly proposed BVD algorithm to reconstruct values at cell boundaries by reducing the total amount of dissipation.

![](_page_25_Figure_1.jpeg)

Figure 12: Detonation problem — Density field at  $t = 0.3 \times 10^{-7}$  (top panels) and  $t = 1.7 \times 10^{-7}$  (bottom panels) — Left panels: Reference solution calculated by the 5th order WENO scheme with  $2000 \times 400$  mesh cells — Middle panel: Solution obtained by 5th order WENO scheme with  $400 \times 80$  mesh cells — Right panel: Solution obtained by the proposed BVD scheme with  $400 \times 80$  mesh cells.

![](_page_25_Figure_3.jpeg)

Figure 13: Detonation problem — Density field at  $t = 1.7 \times 10^{-7}$  seen from a different angle compared to Fig. 12 — Left panels: Reference solution calculated by the 5th order WENO scheme with  $2000 \times 400$  mesh cells — Middle panel: Solution obtained by 5th order WENO scheme with  $400 \times 80$  mesh cells — Right panel: Solution obtained by the proposed BVD scheme with  $400 \times 80$  mesh cells.

Through 1D and 2D benchmark tests, it is shown that the proposed scheme can capture more small-scale flow structures and, at the same time, prevent the occurrence of spurious waves which can pollute stiff detonation problems. The resolution around discontinuities is significantly improved by the proposed approach. Systematic comparisons against exact/reference solutions and 5th order WENO numerical solutions have been presented. Unlike most existing high order polynomial-based reconstruction schemes which may demand the development of complex limiting procedures to suppress those spurious numerical oscillations, THINC-BVD is formulated in a simple, efficient and easy-to-code form. This makes this scheme a promising alternative for simulation of flows involving discontinuity. Moreover the reconstruction procedures only demand a local neighborhood to the current cell which is harmless for parallel framework. However, it is noted that for smooth profiles the polynomial-based scheme can achieve higher order for convergence rate. It indicates a new path to investigate the coupling of BVD and an *a posteriori* approach like MOOD [49, 41] to establish a more flexible numerical framework for high-fidelity computations for both smooth and non-smooth solutions. It should be also noted that in our present tests THINC reconstructions for each cell are conducted with fixed  $\beta$  values, and further efforts might be worth to optimize the reconstructions by using a more flexible formulation for  $\beta$ .

## Acknowledgment

This work was supported in part by the fund from JSPS (Japan Society for the Promotion of Science) under Grant Nos. 15H03916, 15J09915 and 17K18838.

#### References

- [1] C.W. Schulz-Rinne, Classification of the Riemann problem for two-dimensional gas dynamics, SIAM J. Math. Anal 24 (1993), pp.76-88.
- [2] Ma. Brouillette, The richtmyer-meshkov instability, Annu. Rev. Fluid Mech. 34 (2002), pp. 445-468.
- [3] B. Van Leer, Towards the ultimate conservative difference scheme. V. A second-order sequel to Godunov's method, J. Comput. Phys. 32 (1979), pp. 101-136.
- [4] X. Liu, S. Osher, T. Chan, Weighted essentially non-oscillatory schemes, J. Comput. Phys. 115 (1994), pp. 200-212.
- [5] G.S. Jiang, C.W. Shu, Efficient implementation of weighted ENO schemes, J. Comput. Phys. 126 (1996), pp. 202-228.
- [6] A.K. Henrick, T.D. Aslam, J.M. Powers, Mapped weighted essentially non-oscillatory schemes: achieving optimal order near critical points, J. Comput. Phys. 207 (2005), pp. 542-567.
- [7] R. Borges, M. Carmona, B. Costa, W.S. Don, An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws, J. Comput. Phys. 227 (2008), pp. 3191-3211.
- [8] Y.Ha, C.H. Kim, Y.J. Lee, J. Yoon, An improved weighted essentially non-oscillatory scheme with a new smoothness indicator, J. Comput. Phys. 232 (2013), pp. 68-86.
- [9] P.Fan, Y. Shen, B. Tian, C. Yang, A new smoothness indicator for improving the weighted essentially non-oscillatory scheme, J. Comput. Phys. 269 (2014), pp. 329-354.
- [10] F. Acker, R. Borges, B. Costa, An improved WENO-Z scheme, J. Comput. Phys. 313 (2016), pp. 726-753.
- [11] X.Y. Hu, Q. Wang, N.A. Adams, An adaptive central-upwind weighted essentially non-oscillatory scheme, J. Comput. Phys. 229 (2010), pp. 8952-8965.
- [12] Y. Shen, G. Zha, Improvement of weighted essentially non-oscillatory schemes near discontinuities, Comput. Fluids 96 (2014), pp. 1-9.
- [13] R.K. Shukla, C. Pantano, J.B. Freund, An interface capturing method for the simulation of multi-phase compressible flows, J. Comput. Phys. 229 (2010), pp. 7411-7439.
- [14] G.A. Gerolymos, D. Sénéchal, I. Vallet, Very-high-order WENO schemes, J. Comput. Phys. 228 (2009), pp. 8481-8524.
- [15] W. Wang, C.W. Shu, H.C. Yee, B. Sjögreen, High order finite difference methods with subcell resolution for advection equations with stiff source terms, J. Comput. Phys. 231 (2012), pp. 190-214.
- [16] H.C. Yee, D.V. Kotov, W. Wang, C.W. Shu, Spurious behavior of shock-capturing methods by the fractional step approach: Problems containing stiff source terms and discontinuities, J. Comput. Phys. 241 (2013), pp. 266-291.

- [17] A. Marquina, Local piecewise hyperbolic reconstruction of numerical fluxes for nonlinear scalar conservation laws, SIAM J. Sci. Comput. 15 (1994), pp. 892-915.
- [18] F. Xiao, X. Peng, A convexity preserving scheme for conservative advection transport, J. Comput. Phys. 198 (2004), pp. 389-402.
- [19] R. Artebrant, H.J. Schroll, Limiter-free third order logarithmic reconstruction, SIAM J. Sci. Comput. 28 (2006), pp. 359-381.
- [20] F. Xiao, S. Ii, C. Chen, Revisit to the THINC scheme: a simple algebraic VOF algorithm, J. Comput. Phys. 230 (2011), pp. 7086-7092.
- [21] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, Int. J. Numer. Methods Fluids 48 (2005), pp. 1023-1040.
- [22] Z. Sun, S. Inaba, F. Xiao, Boundary Variation Diminishing (BVD) reconstruction: A new approach to improve Godunov schemes, J. Comput. Phys. 322 (2016), pp. 309-325.
- [23] D.I. Ketcheson, M. Parsani, R.J. LeVeque, High-order wave propagation algorithms for hyperbolic systems, SIAM J. Sci. Comput. 35 (2013), pp. A351-A377.
- [24] R.J.LeVeque, Wave propagation algorithms for multidimensional hyperbolic systems, J. Comput. Phys. 131 (1997), pp. 327-353
- [25] E.F. Toro, Riemann solvers and numerical methods for fluid dynamics: a practical introduction, Springer Science & Business Media, 2013.
- [26] S. Gottlieb, C.W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM Rey. 43 (2001), pp. 89-112.
- [27] S. Pirozzoli, On the spectral properties of shock-capturing schemes, J. Comput. Phys. 219 (2006), pp. 489-497.
- [28] C. Hirsch, Numerical computation of internal and external flows: The fundamentals of computational fluid dynamics, Butterworth-Heinemann (2007).
- [29] G.A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, J. Comput. Phys. 27 (1978), pp. 1-31.
- [30] C.W. Shu, S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, J. Comput. Phys. 77 (1988), pp. 439-471.
- [31] P. Woodward, P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, J. Comput. Phys. 54 (1984), pp. 115-173.
- [32] P.H. Maire, R. Loubère, P. Vachal, Staggered Lagrangian discretization based on cell-centered Riemann solver and associated hydro-dynamics scheme, Comput. Phys. Commun. 10 (2011), pp. 940-978.
- [33] X. Zhang, C.W. Shu, Positivity-preserving high order finite difference WENO schemes for compressible Euler equations, J. Comput. Phys. 231 (2012), pp. 2245-2258.
- [34] X.Y. Hu, N.A. Adams, C.W. Shu, Positivity-preserving method for high-order conservative schemes solving compressible Euler equations, J. Comput. Phys. 242 (2013), pp. 169-180.
- [35] W. Bao, S. Jin, The random projection method for hyperbolic conservation laws with stiff reaction terms, J. Comput. Phys. 163 (2000), pp. 216-248.
- [36] L. Tosatto, L. Vigevano, Numerical solution of under-resolved detonations, J. Comput. Phys. 227 (2008), pp. 2317-2343.
- [37] A.J. Chorin, Random choice methods with applications to reacting gas flow, J. Comput. Phys. 25 (1977), pp. 253-272.
- [38] R. Courant, K.O. Friedrichs, Supersonic flow and shock waves, Springer Science & Business Media, 1999.
- [39] W. Bao, S. Jin, The random projection method for stiff detonation capturing, SIAM J. Sci. Comput. 23 (2001), pp. 1000-1026.
- [40] A. Kurganov, E. Tadmor, Solution of two-dimensional Riemann problems for gas dynamics without Riemann problem solvers, Numer. Methods Partial Differential Equations 18 (2002), pp. 584-608.
- [41] M. Dumbser, O. Zanotti, R. Loubère, S. Diot, A posteriori subcell limiting of the discontinuous Galerkin finite element method for hyperbolic conservation laws, J. Comput. Phys. 278 (2014), pp. 47-75.
- [42] Y. Ha, C.H. Kim, Y.J. Lee, J. Yoon, An improved weighted essentially non-oscillatory scheme with a new smoothness indicator, J. Comput. Phys. 232 (2013), pp. 68-86.
- [43] P. Buchmüller, C. Helzel, Improved accuracy of high-order WENO finite volume methods on Cartesian grids, J. Sci. Comput. 61 (2014), pp. 343-368
- [44] R. Abedian, H. Adibi, M. Dehghan, A high-order symmetrical weighted hybrid ENO-flux limiter scheme for hyperbolic conservation laws, Comput. Phys. Comm. 185 (2014), pp. 106-127.
- [45] C.Y. Jung, T.B. Nguyen, Fine structures for the solutions of the two-dimensional Riemann problems by high-order WENO schemes, Adv. Comput. Math. (2017), pp. 1-28.
- [46] P. Woodward, P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, J. Comput. Phys. 54 (1984), pp. 115-173.
- [47] L. Fu, X.Y. Hu, N.A. Adams, A family of high-order targeted ENO schemes for compressible-fluid simulations, J. Comput. Phys. 305 (2016), pp. 333-359.
- [48] R.A. Strehlow, Nature of transverse waves in detonations, Astronaut. Acta 14 (1969), pp. 539.
- [49] S. Diot, R. Loubère, S. Clain, The MOOD method in the three-dimensional case: Very-High-Order Finite Volume Method for Hyperbolic Systems, Int. J. Numer. Methods Fluids 73 (2013), pp. 362-392.