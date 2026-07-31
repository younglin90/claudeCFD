![](_page_0_Picture_1.jpeg)

Contents lists available at [ScienceDirect](http://www.ScienceDirect.com/)

journal homepage: [www.elsevier.com/locate/jcp](http://www.elsevier.com/locate/jcp)

![](_page_0_Picture_5.jpeg)

# A novel steepness-adjustable harmonic volume-of-fluid method for interface capturing ✩

Weidan Ni <sup>a</sup>, Qinghong Zeng <sup>a</sup>*,*∗, Yucang Ruan <sup>b</sup>, Zhiwei He <sup>a</sup>*,*c*,*∗∗

- <sup>a</sup> *Institute of Applied Physics and Computational Mathematics, Fenghaodong Road, Haidian District, Beijing 100094, China*
- <sup>b</sup> *State Key Laboratory for Turbulence and Complex Systems, College of Engineering, Peking University, Beijing 100871, China*
- <sup>c</sup> *National Key Laboratory of Computational Physics, Beijing 100088, China*

## A R T I C L E I N F O A B S T R A C T

*Keywords:* Volume-of-fluid method Steepness-adjustable harmonic scheme Interface capturing scheme

The algebraic volume-of-fluid (VOF) method based on the construction strategy of switching technique has received extensive attention recently. However, all the previous algebraic VOF methods were constructed by combining different methods of compressible differencing scheme (CDS) and the high resolution scheme (HRS), which would switch back and forth in most cases. Thus the final algebraic VOF method usually becomes to be complicated, and the numerical instabilities would be increased. In this paper, a novel algebraic VOF method, without the operations of switching back and forth, is constructed within a unified framework of the steepness-adjustable harmonic (SAH) scheme (He et al. (2022) [\[33\]](#page-26-0)). A thorough validation of the present method is conducted, examining the pure advection of the interface indicator function. The results indicate that the present method can resolve the interface capturing with substantially low numerical diffusion and low numerical oscillations.

# **1. Introduction**

Two-phase flow separated by a material interface ubiquitously exists in practical applications, such as the marine environment [\[1\]](#page-25-0) and the chemical industry [[2](#page-25-0)]. In the last several decades, different methods have been proposed to capture the interfaces where the most popular ones are the front tracking scheme [[3](#page-25-0)], level set method [\[4\]](#page-25-0) and volume-of-fluid (VOF) method [\[5,6](#page-25-0)]. Since the VOF method could conserve mass and handle the topological changes of the interface, it has received extensive attention from researchers.

In the VOF methods, the material interface is approximated by a phase indicator function ∈ [0*,* 1], popularly known as volume fraction. In the pure substances, = 0 or 1 otherwise refers to the interfaces. Depending on the methodology used to treat the interface, the VOF method is usually divided into two branches: geometric type and algebraic type.

Typically, the geometric VOF methods include the earliest proposed method of the simple line interface calculation [\[7\]](#page-25-0) and the most widely used method of the piecewise linear interface calculation (PLIC) [\[8,9\]](#page-25-0). The former utilizes the horizontal or vertical lines to approximate the interface whereas the latter adopts the linear oblique line segments. The geometric VOF methods could advect the interface very accurately. However, in the three-dimensional (3D) flows, the reconstruction of the interface requires significantly

*E-mail addresses:* [zeng\\_qinghong@iapcm.ac.cn](mailto:zeng_qinghong@iapcm.ac.cn) (Q. Zeng), [he\\_zhiwei@iapcm.ac.cn](mailto:he_zhiwei@iapcm.ac.cn) (Z. He).

<sup>✩</sup> We acknowledge financial support from the National Natural Science Foundation of China (NSFC) under No. 12372285 and the Presidential Foundation of CAEP under No. YZJJLX2018012.

<sup>\*</sup> Corresponding author at: Institute of Applied Physics and Computational Mathematics, Fenghaodong Road, Haidian District, Beijing 100094, China.

<sup>\*\*</sup> Corresponding author.

larger computational resources compared to two-dimensional (2D) cases [\[10](#page-25-0)]. It would be even worse on arbitrary meshes, due to the added complexity of the geometric primitives used to reconstruct the interface [\[11](#page-26-0)].

The algebraic VOF methods capture the interface by algebraically discretizing the scalar advection transport equation of the volume fraction. Over years, researchers have proposed a variety of specific techniques, including high resolution differencing schemes, sharpening schemes, flux-limited (anti-diffusion) methods, analytical-function fitted methods and blended high resolution differencing methods [[12\]](#page-26-0). High resolution differencing schemes [[13–16\]](#page-26-0) are inclined to introduce excessive numerical diffusion and dispersion. In sharpening schemes [\[17,18](#page-26-0)], an extra artificial compressive term is added into the VOF advection equation for the purpose of compressing the interface, instead of just employing a compressive differencing scheme (CDS). Flux-limited methods [[19–21\]](#page-26-0) consist of a basic high resolution advection scheme and a multi-dimensional flux limiter, which are conservative, monotonic and shape-preserving for both continuous and discontinuous density fields. In analytical-function fitted methods [\[22](#page-26-0),[23\]](#page-26-0), different smooth basic functions, such as the hyperbolic tangent function and the cubic polynomial function, are adopted to represent a discontinuity at the grid scale in the flux computation of the volume fraction. Blended high resolution differencing methods continuously switch between a CDS and a diffusive high resolution scheme (HRS), according to the angle between the interface direction and the grid orientation.

The major advantages of algebraic VOF methods compared to geometric VOF methods are the straightforward implementation on arbitrary meshes and the computational efficiency. Recent studies [\[17,24](#page-26-0),[25\]](#page-26-0) show that algebraic VOF methods are generally capable of advecting sharp, evolving interfaces with similar accuracy as state-of-the-art PLIC methods.

In this paper, we focus on the algebraic VOF methods, especially the blended high resolution differencing methods. The blended high resolution differencing methods directly capture the interface by introducing proper differencing schemes. The equation of the volume fraction is carefully discretized in both temporal and spatial domains to reduce the impact of the numerical diffusion on the interface. To be specific, a CDS would be used when the interface normal is aligned with the normal of the cell face to keep the sharpness of the interface. Otherwise, a HRS should be adopted when two normals are perpendicular to each other. Therefore, the switching technique and the forms of CDS and HRS constitute the two basic elements of the blended high resolution differencing methods. The previous algebraic VOF schemes, including CICSAM [[26\]](#page-26-0), STACS [\[27](#page-26-0)], FBICS [\[28](#page-26-0)], CUIBS [\[29](#page-26-0)], MSTACS [[30\]](#page-26-0) and SAISH [\[31](#page-26-0)], adopt the same strategy of construction where the face values of the volume fraction calculated by CDS and HRS are blended according to the angle between interface normal and the direction of cell face. This strategy is widely used in the algebraic VOF method and receives continuous attention from researchers.

Although so many achievements have been made, we must point out that (1) for a complex practical flow, the relation between the interface normal and the orientation of the cell face is neither parallel nor perpendicular in most cases and (2) all the CDS and HRS adopted in the blended high resolution differencing methods are constructed by different methods. In other words, the CDS and HRS in all the previous schemes come from different framework. It would result in different expressions when calculating the face value of the volume fraction with the variation of the angle between the interface normal and the orientation of the cell face. In addition, the CDS and HRS in the previous schemes usually have the form of piecewise function and several piecewise points without theoretical determination are involved in each scheme. This kind of blending would increase the complexity of the schemes. Furthermore, Larsson and Gustafsson [\[32](#page-26-0)] pointed out that this kind of construction will decrease the instability of the method. Therefore, it is of importance to construct a simple algebraic VOF method within a unified framework rather than combining two different schemes.

Recently, a novel reconstruction theory called self-adjusting steepness (SAS)-based schemes is proposed by He et al. [\[33\]](#page-26-0). Such schemes are constructed following four main steps: (1) design a slope limiter containing a steepness parameter that provides a mechanism enabling the scheme to accurately solve both smooth and discontinuous problems with proper values; (2) determine the infimum of steepness parameter such that the scheme is order-optimized; (3) determine the supremum of steepness parameter such that the scheme has a nonlinear stable anti-diffusion/compression effect; (4) calculate the steepness parameter in terms of the infimum and supremum using an adaptive algorithm to ensure that the final scheme obtains essentially non-oscillatory and sharp resolutions for various discontinuities while maintaining the nominal second-order accuracy for smooth regions. In their work, the above-mentioned construction strategy was implemented to two specific schemes: THINC scheme [\[34\]](#page-26-0) and harmonic limiter [[35\]](#page-26-0). In principle, the SAS-based schemes can be applied to arbitrary physical variable. Indeed, it was found that for one-dimensional cases the schemes can not only obtain second-order accuracy in smooth regions but also preserve discontinuous flow structures, especially the contact discontinuities, even after long computation time [\[33](#page-26-0)]. A major character of such schemes is that we can achieve second-order accuracy or anti-dissipation under a unified framework just by adjusting the steepness parameter.

In this paper, we extend this idea to the algebraic VOF methods, and propose a novel steepness-adjustable harmonic VOF (SAH-VOF) method. This method is efficiently realized in three steps: (1) the same expression (i.e., the total variation diminishing (TVD) scheme with the steepness-adjustable harmonic limiter, namely SAH) is utilized for both HRS and CDS with only different steepness parameter (namely infimum and supremum, respectively); (2) the switching technique developed for the blended high resolution differencing methods is extended to obtain the final steepness parameter in terms of the infimum and supremum; (3) determine the infimum and supremum of the steepness parameter with a strategy based on least squares approximation for the latter being proposed. The derived SAH-VOF method, inheriting and developing the traditional algebraic VOF methods, is quite simple. However, the numerical tests in 2D and 3D cases over a wide range of numbers show that SAH-VOF is generally capable of advecting sharp and evolving interfaces with comparable or even better accuracy, in comparisons with the previous algebraic VOF methods.

This paper is organized as follows. In Section [2](#page-2-0), the governing equation is described first and the previous algebraic VOF methods, including CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH, are reviewed. In Section [3,](#page-5-0) the novel SAH-VOF method is described in detail. Further, SAH-VOF is compared against the previous algebraic VOF methods with the help of 2D and 3D cases as shown in

![](_page_2_Picture_2.jpeg)

Fig. 1. Notation of upwind, donor and acceptor cell based on the flow direction. The angle  $\theta$  between the interface normal, namely the direction of the gradient of the volume fraction  $\nabla \bar{\phi}$ , and the normal of the cell face  $\mathbf{n}_r$  is also shown in this plot.

Section 4. Besides, we also analyze our method on two benchmark problems [36–38] with the exact vortex solutions referred to the nonlinear Euler equations so as to check the performance of SAH-VOF in the complex and real problems. Conclusions will be given in Section 5.

## 2. Methodology

In this section, the governing equation, namely the scalar advection equation of the volume fraction for two-phase incompressible flows, will be introduced first in Subsection 2.1. Subsequently, the discretization form of the governing equation adopting the algebraic VOF method based on the widely-used switching technique in the finite volume frame is described. The specific forms of the previous algebraic VOF methods to be compared in this study, including CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH, are tabulated. It is followed by the brief descriptions for all the above-mentioned schemes as given in Subsection 2.2.

#### 2.1. Basic transport equation of volume fraction

The two-phase fluids, assumed to be immiscible and incompressible, are solved by finite volume method with regards to the volume fraction equation of fluid 1 with given velocity field. The governing equation reads as,

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\phi) = 0 \tag{1}$$

where  $\phi$  and t refer to the volume fraction of fluid 1 and the time, respectively. The variable  $\mathbf{u} = (u, v)$  represents the velocity vector, where u and v symbolize the velocity component in the x and y direction, respectively. Assuming the volume of the i-th grid cell is  $\Omega_i$ , the average of the volume fraction of i-th grid cell at time t is defined as,

$$\bar{\phi}_i(t) = \frac{1}{|\Omega_i|} \int_{\Omega_i(\mathbf{x})} \phi(\mathbf{x}, t) d\mathbf{x}$$
 (2)

where the coordinate vector  $\mathbf{x} = (x, y, z)$ . Integrating Eq. (1) over the *i*-th grid cell with Gauss' law applied,

$$\frac{\partial \bar{\phi}_i(t)}{\partial t} = -\frac{1}{|\Omega_i|} \sum_{f=1}^m \phi_{if}^* A_{if} \mathbf{u}_{if} \cdot \mathbf{n}_{if}$$
(3)

where the subscript f represents the f-th face of the i-th grid cell and m denotes the total number of the cell faces relying on the dimension of the geometry. The variables of  $\phi_{if}^*$ ,  $\mathbf{a}_{if}$ ,  $\mathbf{u}_{if}$  and  $\mathbf{n}_{if}$  refer to the volume fraction, area, velocity vector and normal vector of the f-th face of the i-th grid cell, respectively. In this paper, the volume fraction  $\phi_{if}^*$  and the normal velocity  $\mathbf{u}_{if} \cdot \mathbf{n}_{if}$  of cell face are considered to be equal to the corresponding values at the centroid of the cell face. In Eq. (3), a third-order TVD Runge-Kutta (RK) method [39] is adopted for the time integration. Compared with the Crank-Nicholson method that is utilized in the conventional algebraic VOF schemes such as CICSAM, the explicit scheme of third-order RK method is easier to be applied in solving the compressible Navier-Stokes equations. As a part of a long-term research, the third-order RK method is adopted in this study.

It can be seen that the key of solving Eq. (3) is to construct the face value of the volume fraction  $\phi_{if}^*$  if the time discretization scheme is given. As illustrated in Fig. 1, we take the calculation of the cell face value of the volume fraction of the donor cell as an example and thus the subscript i is omitted in  $\phi_{if}^*$ . The face value of the volume fraction  $\phi_f^*$  can be constructed with the help of the average value of the volume fraction of acceptor and donor cell, namely  $\bar{\phi}_A$  and  $\bar{\phi}_D$ , respectively. It reads as,

$$\phi_f^* = \gamma_f \bar{\phi}_{\mathcal{A}} + (1 - \gamma_f) \bar{\phi}_{\mathcal{D}} \tag{4}$$

A cell being an acceptor or a donor depends on the direction of the advection velocity, as illustrated in Fig. 1. The weight factor  $\gamma_f$  in Eq. (4) is calculated as,

$$\gamma_f = \frac{\tilde{\phi}_f - \tilde{\phi}_D}{1 - \tilde{\phi}_D} \tag{5}$$

where  $\tilde{\phi}_f$  and  $\tilde{\phi}_D$  represent the normalized values of the volume fraction of the cell face and the donor cell, respectively. They are defined as

$$\tilde{\phi}_f = \frac{\phi_f^* - \bar{\phi}_{\mathrm{U}}}{\bar{\phi}_{\mathrm{A}} - \bar{\phi}_{\mathrm{U}}} \tag{6}$$

$$\tilde{\phi}_{\mathrm{D}} = \frac{\bar{\phi}_{\mathrm{D}} - \bar{\phi}_{\mathrm{U}}}{\bar{\phi}_{\mathrm{A}} - \bar{\phi}_{\mathrm{U}}} \tag{7}$$

where  $\bar{\phi}_{\rm U}$  denotes the average value of the volume fraction of the upwind cell. Actually, the normalized variable [40,41] is utilized in Eqs. (6) and (7) for the construction of high resolution schemes. The corresponding normalized variable diagram (NVD) for each scheme will be compared in Subsection 3.2.3. In the previous study, including CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH, the normalized value  $\tilde{\phi}_f$  is constructed based on the strategy of switching technique as follows,

$$\tilde{\phi}_f = \alpha(\theta)\tilde{\phi}_{fCDS} + [1 - \alpha(\theta)]\tilde{\phi}_{fHRS} \tag{8}$$

where  $\tilde{\phi}_{f\text{CDS}}$  and  $\tilde{\phi}_{f\text{HRS}}$  refer to the normalized value of the volume fraction of the cell face calculated by CDS and HRS, respectively. The blending function  $\alpha(\theta)$  is a detector to identify the direction of the interface and thus determines that either  $\tilde{\phi}_{f\text{CDS}}$  or  $\tilde{\phi}_{f\text{HRS}}$  acts the key role on the calculation of  $\tilde{\phi}_f$ . It reads as,

$$\alpha(\theta) = \min[(\cos \theta)^p, 1.0] \tag{9}$$

where the parameter p is set to be different values in different schemes. Tsui et al. [28] mentioned that the results are not sensitive to the choices of  $\alpha(\theta)$ . The angle  $\theta$  refers to the one between the interface normal, namely the direction of the gradient of the volume fraction of the donor cell  $(\nabla \bar{\phi})_D$ , and the normal of the cell face  $\mathbf{n}_f$  as illustrated in Fig. 1. Therefore, the angle  $\theta$  can be obtained via Eq. (10),

$$\cos\theta = \left| \frac{(\nabla \bar{\phi})_{\mathrm{D}} \cdot \mathbf{n}_{f}}{|(\nabla \bar{\phi})_{\mathrm{D}}| |\mathbf{n}_{f}|} \right| \tag{10}$$

where  $(\nabla \bar{\phi})_D$  are calculated by adopting Parker-Youngs method [42].

#### 2.2. Previous typical algebraic VOF methods

In this subsection, the previous algebraic VOF methods to be compared in the present study will be briefly described and compared, including CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH. The differences in the previous algebraic VOF methods are the selection of  $\tilde{\phi}_{f\text{CDS}}$ ,  $\tilde{\phi}_{f\text{HRS}}$  and the value of p in the blending function  $\alpha(\theta)$ . The specific forms of the above-mentioned schemes are summarized in Table 1.

• Compressive Interface Capturing Scheme for Arbitrary Meshes, CICSAM [26]

In the original algebraic VOF method [43], the switching is activated suddenly when the angle between two normals is more (or less) than  $45^{\circ}$ . This work is followed by Lafaurie et al. [44], who further conducted the switching conditions and their research indicates that the accuracy of the method greatly depends on the switching angle. Ubbink and Issa [26] pointed out that it is not "when" to switch but rather "how" to switch. They proposed CICSAM preserving both the smoothness and sharpness of the interface. In CICSAM, HYPER-C [45] and ULTIMATE-QUICKEST [46] are used as CDS and HRS, respectively. CICSAM exhibits good performance up to Courant number (Co) 0.3 while it introduces too much undesirable numerical dissipation for the moderate and high Co numbers [47,12]. It is worth mentioning that ULTIMATE-QUICKEST is extremely diffusive on its own and the numerical diffusion induced by HYPER-C grows as Co number increases. This is also the reason why the modified CICSAM proposed by Chakraborty and Banerjee [48] failed to perform well at high Co numbers.

• Switching Technique for Advection and Capturing of Surfaces, STACS [27]

Darwish and Moukalled [27] pointed out that the use of transient bounding in CICSAM results in the severe numerical diffusion at high Co number. It should be noted that the transient bounding is originally designed for explicit QUICKEST by Leonard [41]. It will increase the numerical diffusion if the implicit time discretization is adopted. Therefore, Darwish and Moukalled [27] proposed STACS where SUPERBEE [45] and STOIC [49] are employed as CDS and HRS, respectively, to decrease the numerical diffusion at high Co number. In STACS, less numerical diffusion is introduced at high Co numbers compared with CICSAM. However, STACS is more diffusive at low Co numbers than CICSAM [12]. Anghan et al. [30] argue that it is the use of SUPERBEE in STACS that leads to its unsatisfactory performance at low Co number, since the bounded scheme of SUPERBEE falls within the TVD region.

· Flux-Blending Interface-Capturing Scheme, FBICS [28]

Table 1
Schemes in normalized form.

| Schemes | Value of p | Normalized form                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|---------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CICSAM  | 2          | $\tilde{\phi}_{f\text{CDS}} = \begin{cases} \min(\frac{\tilde{\phi}_{\text{D}}}{Co_{\text{D}}}, 1.0), & \text{if } 0 \leq \tilde{\phi}_{\text{D}} \leq 1\\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|         |            | $\begin{split} \tilde{\phi}_{f\text{CDS}} &= \begin{cases} \min(\frac{\tilde{\phi}_{\text{D}}}{Co_{\text{D}}}, 1.0), & \text{if } 0 \leq \tilde{\phi}_{\text{D}} \leq 1 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases} \\ \tilde{\phi}_{f\text{HRS}} &= \begin{cases} \min(\frac{8Co_{\text{D}}\tilde{\phi}_{\text{D}} + (1 - Co_{\text{D}})(6\tilde{\phi}_{\text{D}} + 3)}{8}, \tilde{\phi}_{f\text{BD}}), & \text{if } 0 \leq \tilde{\phi}_{\text{D}} \leq 1 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases} \end{split}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| STACS   | 4          | $\begin{split} \tilde{\phi}_{fHRS} &= \begin{cases} \min \\ \bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} < \frac{1}{3} \\ \frac{1+\bar{\phi}_{D}}{2}, & \text{if } 1_{3} \leq \bar{\phi}_{D} < \frac{1}{2} \\ \frac{1+\bar{\phi}_{D}}{2}, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} < \frac{1}{2} \\ \frac{3}{2}\bar{\phi}_{D}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} < \frac{2}{3} \\ 1, & \text{if } \frac{2}{3} \leq \bar{\phi}_{D} \leq 1 \\ \bar{\phi}_{D}, & \text{if } \bar{\phi}_{D} < 0 \text{ or } \bar{\phi}_{D} > 1 \end{cases} \\ \tilde{\phi}_{fHRS} &= \begin{cases} 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} < \frac{1}{5} \\ \frac{1+\bar{\phi}_{D}}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} < \frac{1}{5} \\ \frac{1+\bar{\phi}_{D}}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} < \frac{1}{5} \\ \frac{1+\bar{\phi}_{D}}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} < \frac{1}{5} \\ 1, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq \frac{1}{6} \\ 1, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq \frac{1}{6} \end{cases} \\ 1, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq 1 \end{cases} \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} \leq 1 \\ \bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} < \frac{1}{8} \end{cases} \\ \tilde{\phi}_{D} + \frac{1}{4}, & \text{if } \frac{1}{8} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{8} \end{cases} \\ \tilde{\phi}_{D} + \frac{1}{4}, & \text{if } \frac{1}{8} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \end{cases} \\ 1, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \end{cases} \\ \tilde{\phi}_{fCDS} = \begin{cases} 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \end{cases} \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \end{cases} \\ \tilde{\phi}_{D} = \begin{cases} 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \leq \bar{\phi}_{D} \leq 1 \\ \tilde{\phi}_{D}, & \text{if } \bar{\phi}_{D} < 0 \text{ or } \bar{\phi}_{D} > 1 \end{cases} \\ \tilde{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq 1 \\ 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq 1 \\ 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq 1 \\ 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq 1 \\ 3\bar{\phi}_{D}, & \text{if } 0 \leq \bar{\phi}_{D} \leq \frac{1}{5} \end{cases} \\ \frac{1}{2}\bar{\phi}_{D} + \frac{1}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq \frac{1}{5} \\ \frac{1}{2}\bar{\phi}_{D} + \frac{1}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq \frac{1}{5} \end{cases} \\ \frac{1}{2}\bar{\phi}_{D} + \frac{1}{2}, & \text{if } \frac{1}{2} \leq \bar{\phi}_{D} \leq \frac{1}{5} \end{cases} $ |
|         |            | $\tilde{\phi}_{fHRS} = \begin{cases} 3\tilde{\phi}_{\rm D}, & \text{if } 0 \leq \tilde{\phi}_{\rm D} < \frac{1}{5} \\ \frac{1+\tilde{\phi}_{\rm D}}{2}, & \text{if } \frac{1}{5} \leq \tilde{\phi}_{\rm D} < \frac{1}{2} \\ \frac{3+6\tilde{\phi}_{\rm D}}{8}, & \text{if } \frac{1}{2} \leq \tilde{\phi}_{\rm D} < \frac{5}{6} \\ 1, & \text{if } \frac{5}{6} \leq \tilde{\phi}_{\rm D} \leq 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| FBICS   | 4          | $\tilde{\phi}_{f\text{CDS}} = \begin{cases} 3\tilde{\phi}_{\text{D}}, & \text{if } 0 \le \tilde{\phi}_{\text{D}} < \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \le \tilde{\phi}_{\text{D}} \le 1 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|         |            | $\tilde{\phi}_{fHRS} = \begin{cases} 3\tilde{\phi}_{\rm D}, & \text{if } 0 \leq \tilde{\phi}_{\rm D} < \frac{1}{8} \\ \tilde{\phi}_{\rm D} + \frac{1}{4}, & \text{if } \frac{1}{8} \leq \tilde{\phi}_{\rm D} < \frac{3}{4} \\ 1, & \text{if } \frac{1}{4} \leq \tilde{\phi}_{\rm D} \leq 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| CUIBS   | 4          | $\tilde{\phi}_{f\text{CDS}} = \begin{cases} 3\tilde{\phi}_{\text{D}}, & \text{if } 0 \leq \tilde{\phi}_{\text{D}} < \frac{1}{3} \\ 1, & \text{if } \frac{1}{3} \leq \tilde{\phi}_{\text{D}} \leq 1 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|         |            | $\tilde{\phi}_{fHRS} = \begin{cases} 3\tilde{\phi}_{\rm D}, & \text{if } 0 \leq \tilde{\phi}_{\rm D} < \frac{2}{13} \\ \frac{5}{6}\tilde{\phi}_{\rm D} + \frac{1}{3}, & \text{if } \frac{2}{13} \leq \tilde{\phi}_{\rm D} < \frac{4}{5} \\ 1, & \text{if } \frac{4}{5} \leq \tilde{\phi}_{\rm D} \leq 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| MSTACS  | 4          | $\tilde{\phi}_{f\text{CDS}} = \begin{cases} \min(\frac{\bar{\phi}_{\text{D}}}{Co_{\text{D}}}, 1.0), & \text{if } 0 \leq \tilde{\phi}_{\text{D}} < 1 \text{ and } 0 < Co_{\text{D}} \leq 0.33 \\ \min(3\tilde{\phi}_{\text{D}}, 1.0), & \text{if } 0 \leq \tilde{\phi}_{\text{D}} \leq 1 \text{ and } 0.33 < Co_{\text{D}} \leq 1.0 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|         |            | $ \tilde{\phi}_{fHRS} = \begin{cases} 3\tilde{\phi}_{\rm D}, & \text{if } 0 \le \tilde{\phi}_{\rm D} < \frac{1}{5} \\ \frac{1}{2}\tilde{\phi}_{\rm D} + \frac{1}{2}, & \text{if } \frac{1}{5} \le \tilde{\phi}_{\rm D} < \frac{1}{2} \\ \frac{3}{8}\tilde{\phi}_{\rm D} + \frac{3}{4}, & \text{if } \frac{1}{2} \le \tilde{\phi}_{\rm D} < \frac{5}{6} \\ 1, & \text{if } \frac{5}{6} \le \tilde{\phi}_{\rm D} \le 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases} $ $ \tilde{\phi}_{fCDS} = \begin{cases} 4\tilde{\phi}_{\rm D}, & \text{if } 0 \le \tilde{\phi}_{\rm D} < \frac{1}{4} \\ 1, & \text{if } \frac{1}{4} \le \tilde{\phi}_{\rm D} \le 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases} $                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| SAISH   | 2          | $\tilde{\phi}_{f\text{CDS}} = \begin{cases} 4\tilde{\phi}_{\text{D}}, & \text{if } 0 \leq \tilde{\phi}_{\text{D}} < \frac{1}{4} \\ 1, & \text{if } \frac{1}{4} \leq \tilde{\phi}_{\text{D}} \leq 1 \\ \tilde{\phi}_{\text{D}}, & \text{if } \tilde{\phi}_{\text{D}} < 0 \text{ or } \tilde{\phi}_{\text{D}} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|         |            | $\tilde{\phi}_{fHRS} = \begin{cases} \tilde{\phi}_{\rm D}(2 - \tilde{\phi}_{\rm D}), & \text{if } 0 \leq \tilde{\phi}_{\rm D} < \frac{1}{2} \\ \tilde{\phi}_{\rm D} + \frac{1}{4}, & \text{if } \frac{1}{2} \leq \tilde{\phi}_{\rm D} < \frac{3}{4} \\ 1, & \text{if } \frac{3}{4} \leq \tilde{\phi}_{\rm D} \leq 1 \\ \tilde{\phi}_{\rm D}, & \text{if } \tilde{\phi}_{\rm D} < 0 \text{ or } \tilde{\phi}_{\rm D} > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

Tsui and co-workers [28] constructed two novel schemes for interface captures, namely FBICS-A and FBICS-B. It features the use of flux limiters to blend CDS and HRS to determine the convective fluxes through cell faces. The FBICS-A scheme (hereinafter referred to simply as FBICS) is more efficient in capturing the discontinuities with low errors compared with the other. In FBICS, bounded downwind (BD) scheme and Fromm scheme are adopted as CDS and HRS, respectively. Tests on typical advection problems indicate that FBICS could maintain high-accuracy performance for *Co* number up to one.

• Cubic Upwind Interpolation based Blending Scheme, CUIBS [29]

CUIBS, which was proposed by Patel and Natarajan [29], also employs BD as CDS in accordance with the selection of FBICS. The selection of HRS in CUIBS was inspired by the work of Waterson and Deconinck [50] which shows that the best convective schemes belong to the GPL(Generalized Piecewise Linear)- $\kappa$  class of schemes, including MUSCL, SMART and Koren's limited CUI scheme. The first two have been utilized in the previous schemes (FBICS [28] and STACS [27] for instance) and the Koren's limited CUI is first adopted as HRS in CUIBS. Patel and Natarajan [29] also pointed out that interface capturing schemes proposed in literature may be encompassed into a single class of GPL- $\kappa$  schemes, which allows for a unified approach for development of such schemes.

Modified Switching Technique for Advection and Capturing of Surfaces, MSTACS [30]

MSTACS [30] is built upon a framework of STACS [27]. The same HRS, namely STOIC, has been incorporated in MSTACS where CDS is employed according to the Co number. For  $Co \le 0.33$ , HYPER-C is adopted as CDS otherwise BD (in accordance with that in FBICS/CUIBS) is utilized. It is worth noticing that for the six schemes to be compared in this study, only MSTACS and CICSAM are dependent on Co number. MSTACS could capture sharp interfaces with low numerical diffusion over a wide range of Co numbers.

· Smoothly Adapting Interfacial Scheme based on Hybridization, SAISH [31]

In the practical applications, it is challenging for most algebraic VOF method that how to realize the interface capture with low numerical diffusion, mass conservation and high computational efficiency. Recently, Arote et al. [31] proposed SAISH, which performs well on the aspect of mass conservation and computational efficiency. In SAISH, BD scheme (different from that in FBICS/CUIBS) is adopted as CDS and the combination of the hybrid linear/parabolic approximation (HLPA) [51] and the Fromm scheme [52] as HRS. Arote et al. [31] compared SAISH with the previous algebraic VOF schemes, including CICSAM, CUIBS, FBICS and M-CICSAM, based on standard pure advection test problems. The results demonstrate minimal numerical diffusion is achieved by SAISH as compared to the other methods considered in their study.

#### 3. New method

In Subsection 3.1, the comments on the previous algebraic VOF methods are conducted and their deficiencies are pointed out. Subsequently, a novel interface capturing scheme is proposed aiming to overcome the outlined shortcomings in Subsection 3.2.

#### 3.1. Comments on previous algebraic VOF methods

According to the discussions above, we can sum up two characters of the previous algebraic VOF methods. (1) In CICSAM, STACS, FBICS, CUIBS, MSTACS or SAISH, the CDS and HRS in each scheme are constructed by different method, resulting in complex expressions; (2) All the methods need a large number of back-and-forth switching operations. As mentioned in Section 1, this kind of construction will decrease the instability of the new scheme [32].

Recently, He et al. [33] proposed a new concept, namely steepness-adjustable harmonic (SAH) limiters where the steepness is measured by a parameter  $\beta$ . Such limiter exhibits different desired behaviors including a second-order property with theoretically determined infimum of  $\beta$  or an anti-diffusive/compressive property with a larger steepness parameter  $\beta$  under a unified framework. In this paper, we extend this idea to the algebraic VOF methods, and propose a novel SAH-VOF method.

# 3.2. Steepness-adjustable harmonic VOF (SAH-VOF) method

The construction of SAH-VOF method includes three key points: (1) using the TVD scheme with the steepness-adjustable harmonic limiter, namely SAH scheme, as CDS and HRS; (2) developing a switching technique to combine them; (3) determining the infimum and supremum of the steepness parameter  $\beta$ . The idea of construction for SAH-VOF scheme will be described in detail.

# 3.2.1. Normalized variable form of steepness-adjustable harmonic (SAH) scheme

The classical three-point TVD scheme is widely adopted in the numerical simulations of hyperbolic conservation laws to suppress spurious oscillations with second-order accuracy achieved. For convenience of description, Eq. (1) is simplified to the one-dimensional form as an example. It reads as,

$$\frac{\partial \phi}{\partial t} + u \frac{\partial \phi}{\partial x} = 0 \tag{11}$$

Without loss of generality, we assume that the advection velocity is a positive constant (u > 0). Integrating Eq. (11) over the i-th grid cell, we could obtain

$$\frac{\mathrm{d}\bar{\phi}_i}{\mathrm{d}t} \approx -u \frac{\phi_f^* - \phi_{f-1}^*}{\Delta x} \tag{12}$$

where  $\phi_f^*$  and  $\phi_{f-1}^*$  denote the right and left cell face value of volume fraction of the *i*-th grid cell, respectively, and  $\Delta x$  represents the width of the *i*-th grid cell. The three-point TVD scheme is constructed by adopting a three-point stencil around the *i*-th grid cell. Specially, for the donor cell,  $\phi_f^*$  can be written as,

W. Ni, Q. Zeng, Y. Ruan et al.

$$\phi_f^* = \bar{\phi}_D + \frac{1}{2}\Phi(r_f)(\bar{\phi}_A - \bar{\phi}_D) \tag{13}$$

where  $r_f = \frac{\bar{\phi}_D - \bar{\phi}_U}{\bar{\phi}_L - \bar{\phi}_D}$  and  $\Phi(r_f)$  is the general form of the limiter. For harmonic limiter [35],  $\Phi(r_f)$  is expressed as,

$$\Phi(r_f) = \frac{|r_f| + r_f}{1 + r_f} \tag{14}$$

Recently, the limiter in Eq. (14) is extended to the one with an adjustable parameter  $\beta$ , namely SAH limiter [33]. It is given as,

$$\Phi(r_f) = \frac{|r_f| + r_f}{\frac{1}{\beta} + r_f} \tag{15}$$

Combining the standard form (Eq. (13)) and the SAH limiter (Eq. (15)), we could obtain a compact form [53] for easier realization,

$$\phi_{fSAH}^* = \bar{\phi}_{D} + \frac{sgn(\bar{\phi}_{D} - \bar{\phi}_{U}) + sgn(\bar{\phi}_{A} - \bar{\phi}_{D})}{2} \frac{|\bar{\phi}_{D} - \bar{\phi}_{U}||\bar{\phi}_{A} - \bar{\phi}_{D}|}{|\bar{\phi}_{D} - \bar{\phi}_{U}| + \frac{1}{\beta}|\bar{\phi}_{A} - \bar{\phi}_{D}|}$$
(16)

Since the NVD could reflect the performance of the schemes, Eq. (16) is rewritten using the normalized variable of  $\tilde{\phi}_f$  and  $\tilde{\phi}_D$  as defined in Eqs. (6) and (7) for comparing with the above-mentioned algebraic VOF methods. The normalized value  $\tilde{\phi}_{fSAH}$  is expressed as,

$$\tilde{\phi}_{fSAH} = \tilde{\phi}_{D} + \frac{\text{sgn}(\tilde{\phi}_{D}) + \text{sgn}(1 - \tilde{\phi}_{D})}{2} \frac{|\tilde{\phi}_{D}||1 - \tilde{\phi}_{D}|}{|\tilde{\phi}_{D}| + \frac{1}{\beta}|1 - \tilde{\phi}_{D}|}$$
(17)

The value of  $\beta$  could reflect the compressibility of the SAH scheme. The larger  $\beta$  is, the stronger compressibility the scheme has. Due to this property, the CDS and HRS could be represented with the same expression with only the steepness parameter different.

#### 3.2.2. How to combine the compressible differencing scheme (CDS) and the high resolution scheme (HRS)

Inspired by the switching technique in the classical algebraic VOF method, we have two ways to develop the switching technique in the present study: (1) direct application of the classical switching technique based on the SAH schemes with different values of the steepness parameter  $\beta$ ; (2) extending the classical switching technique to the determination of  $\beta$ . However, the former would still result in the combination of different schemes although only the values of  $\beta$  are different in the schemes for construction. In contrast, the latter provides a satisfactory strategy to construct a unified and simple algebraic VOF method. For comparison, both strategies above would be given in the following.

Since the value of the steepness parameter  $\beta$  could reflect the character of the SAH scheme, we assume that there exists a small value of  $\beta_s$  and a large one of  $\beta_t$ , resulting in the corresponding HRS and CDS. They can be written as,

$$\tilde{\phi}_{f\text{SAH-BS}} = \tilde{\phi}_{\text{D}} + \frac{\text{sgn}(\tilde{\phi}_{\text{D}}) + \text{sgn}(1 - \tilde{\phi}_{\text{D}})}{2} \frac{|\tilde{\phi}_{\text{D}}||1 - \tilde{\phi}_{\text{D}}|}{|\tilde{\phi}_{\text{D}}| + \frac{1}{\tilde{\theta}_{s}}|1 - \tilde{\phi}_{\text{D}}|}$$

$$(18)$$

$$\tilde{\phi}_{fSAH-BL} = \tilde{\phi}_D + \frac{\operatorname{sgn}(\tilde{\phi}_D) + \operatorname{sgn}(1 - \tilde{\phi}_D)}{2} \frac{|\tilde{\phi}_D||1 - \tilde{\phi}_D|}{|\tilde{\phi}_D| + \frac{1}{a}|1 - \tilde{\phi}_D|}$$
(19)

Naturally, we could utilize the HRS and CDS as shown in Eqs. (18) and (19), respectively, to construct a new scheme resembling the strategy of classical algebraic VOF method. However, the resultant scheme would still struggle with the back-and-forth switching between two different schemes in spite of only the steepness parameter being different.

Therefore, in the present study, the switching technique of the classical algebraic VOF method is extended to the calculation of the steepness parameter  $\beta$  rather than the direct hybrid of two different schemes. Specifically,  $\beta$  is automatically adjusted according to the relation between the interface normal and the direction of the cell face, namely the angle  $\theta$ . The steepness parameter  $\beta$  is calculated by,

$$\beta = \alpha(\theta)\beta_t + (1 - \alpha(\theta))\beta_t \tag{20}$$

where the blending function  $\alpha(\theta)$  is calculated using Eq. (9) with p=2. It can be seen from Eq. (20) that  $\beta$  could automatically choose a value closer to  $\beta_s$  or  $\beta_l$  in terms of the angle  $\theta$ . It is worth pointing out that although the steepness parameter  $\beta$  in the SAS-based schemes [33] has a similar formulation with Eq. (20), it is implemented dimension by dimension in the multi-dimensional calculation. Xiao et al. [54] also proposed a dimension-by-dimension THINC-based VOF scheme, which could maintain a proper thickness in the normal direction of the interface. However, the dimension-by-dimension schemes are not suitable for the unstructured grids. Our new scheme will not be limited by the pattern of the grid discretization. Next, the values of  $\beta_s$  and  $\beta_l$  will be determined.

# 3.2.3. Determination of the infimum $\beta_s$ and the supremum $\beta_l$

It can be seen from Eq. (20) that it is of great importance to properly determine the values of  $\beta_s$  and  $\beta_l$ . Naturally, the infimum  $\beta_s$  is set to be 1.0 because the SAH limiter is reduced to the classical harmonic limiter with second-order accuracy when the steepness

![](_page_7_Figure_2.jpeg)

**Fig. 2.** Normalized variable diagram (NVD) for all schemes. (a) CICSAM; (b) STACS; (c) FBICS; (d) CUIBS; (e) MSTACS; (f) SAISH; (g) SAH-VOF.

parameter = = 1. However, the determination of the supremum is complicated, and has not been solved so far. We propose a new method to solve this problem in the present study where the least squares approximation is adopted to determine . The implement details will be given in the following.

First of all, the NVDs for all schemes are illustrated in Fig. 2. It can be seen from Fig. 2a that both the CDS and HRS in CICSAM, namely HYPER-C and UQ, respectively, approach the first-order upwind differencing as increases and become identical at = 1*.*0. It means that CICSAM would induce high numerical dissipation and weak compressibility at high number, leading to the severe smearing of the interface. In the -independent method of STACS as shown in Fig. 2b, the high dissipation is alleviated at high number whereas STACS is too diffusive at low number [\[12\]](#page-26-0). Anghan et al. [\[30\]](#page-26-0) argued that it is the CDS falling within the TVD region that makes the interface capturing scheme diffusive at lower number. Unfortunately, in STACS, its CDS of SUPERBEE totally lies in the TVD region. It is interesting to find that for another three -independent schemes, namely FBICS, CUIBS and SAISH, all choose BD as CDS with only the compressibility different in the last one. MSTACS also utilizes BD as CDS for *>* 0*.*33 whereas HYPER-C with higher compressibility is adopted for lower numbers. Generally, FBICS, CUIBS, SAISH and MSTACS could induce proper numerical dissipation over a wide range of numbers compared with CICSAM and STACS. Arote et al. [[31\]](#page-26-0) have demonstrated that SAISH induces minimal numerical diffusion as compared to FBICS and CUIBS based on standard pure advection test problems. As discussed in Section [4](#page-8-0), SAISH performs better than MSTACS for most of the test cases over a wide range of numbers. On the other hand, SAISH has a simpler formulation in compared with -dependent MSTACS as shown in Table [1.](#page-4-0) However, it can be seen from Fig. 2f that SAISH is not differentiable everywhere in its form. The research has shown that

![](_page_8_Figure_2.jpeg)

**Fig. 3.** Variation of integral value in Eq. (22) with  $\beta$ .

the non-differentiability of the schemes would cause the greatest degradation in convergence performance [55,56]. For this reason, Venkatakrishnan [55] introduces a smooth alternative by replacing the original non-differential function.

Inspired by their work, the CDS in SAISH, namely BD as given in Table 1, is optimized by SAH scheme and thus the supremum  $\beta_l$  is determined. Suppose in an interval  $x \in [a,b]$ ,  $f_{app}(x)$ ,  $f_{des}(x) \in \mathbb{R}[a,b]$ . If there exists  $f_{app}^*(x) \in \mathbb{R}[a,b]$  satisfying

$$\min_{f_{\text{app}}(x) \in \mathbb{R}[a,b]} \int_{a}^{b} [f_{\text{app}}(x) - f_{\text{des}}(x)]^2 dx \tag{21}$$

then  $f_{\rm app}^*(x)$  is the least squares approximation of  $f_{\rm des}(x)$ . let  $f_{\rm app}(x) = \tilde{\phi}_{\rm fSAH}(\tilde{\phi}_{\rm D}), \ f_{\rm des}(x) = \tilde{\phi}_{\rm fSAISH-BD}(\tilde{\phi}_{\rm D}),$  and the integral interval be  $\tilde{\phi}_{\rm D} \in [0,1]$ . We can get

$$\min_{\tilde{\phi}_f(\tilde{\phi}_D) \in \mathbb{R}[0,1]} \int_0^1 [\tilde{\phi}_{fSAH}(\tilde{\phi}_D) - \tilde{\phi}_{fSAISH-BD}(\tilde{\phi}_D)]^2 d\tilde{\phi}_D$$
(22)

The supremum  $\beta_l$  can be then obtained from the least squares approximation function. The variation of integral value in Eq. (22) with  $\beta$  is plotted in Fig. 3 and the integral value obtains the minimum when  $\beta = 8.7$ . Therefore, the supremum  $\beta_l$  is set to be 8.7 in this paper.

It can be seen that, unlike the previous algebraic VOF methods, the SAH-VOF method is constructed with a single scheme of SAH with an automatically adjustable steepness parameter  $\beta$ , rather than adopting the different complex schemes of HRS and CDS.

In conclusion, the novel algebraic VOF method of SAH-VOF proposed in this paper is summarized as follows. The formulation of SAH-VOF is the same as that of SAH scheme as shown in Eq. (17) with only the adjustable steepness parameter  $\beta$  different. In SAH-VOF, the parameter  $\beta$  is calculated via Eq. (20) with  $\beta_s = 1.0$  and  $\beta_l = 8.7$ . The blending function  $\alpha(\theta)$  is calculated using Eq. (9) with p = 2. It can be seen that compared with the previous algebraic VOF methods, the variation of the angle  $\theta$  is involved in the construction of  $\beta$  and thus the form of SAH-VOF is unified with regards to arbitrary angle  $\theta$ . In the next section, the comparisons between SAH-VOF and the previous algebraic VOF methods will be conducted in detail.

**Remark 1.** The strategy as shown in Eq. (21) can also be applied in the determination of the supremum of  $\beta$  in the construction of SAS-based schemes proposed by He et al. [33]. In their study, the supremum of  $\beta$  is selected with the limiter generally lying around the upper bound of the TVD region. The strategy in Eq. (21) provides a quantitative method to determine the supremum of  $\beta$  for the SAS-based schemes. By applying Eq. (21), the supremum of  $\beta$  is equal to 2.1 and 2.9 for the SAS-THINC and SAS-SAH schemes, respectively, which is in accordance with their study.

# 4. Numerical tests

In this section, the performance of SAH-VOF will be conducted in both 2D and 3D cases as well as two benchmark problems from Meleshko and Van Heijst [38] over a wide range of Co numbers and the comparisons with CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH will be made as well. Quantitative comparison is carried out with the help of  $L_1$  norm of error, namely  $E_{avg}$ . It is defined as [30],

$$E_{avg} = \frac{\sum_{j=1}^{N} |\bar{\phi}_{j}^{t} - \bar{\phi}_{j}^{a}|}{N}$$
 (23)

where N refers to the number of the grid points. The superscripts "t" and "a" represent the volume fraction at time t and the corresponding analytical solution in each case. The order of convergence [31] of SAH-VOF scheme is also assessed using

![](_page_9_Figure_2.jpeg)

**Fig. 4.** Contour level of volume fraction for hollow square translation problem solved by different interface capturing schemes at various *Co* numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

$$O = \frac{\ln(\frac{E_{avg(2h)}}{E_{avg(h)}})}{\ln(r)}$$
(24)

where h and r denote the grid size and ratio, respectively. In order to check the mass conservation property of the scheme, the mass error [57] is also calculated which is defined as,

$$E_{m} = \frac{|\sum_{j=1}^{N} \bar{\phi}_{j}^{t} - \sum_{j=1}^{N} \bar{\phi}_{j}^{i}|}{\sum_{j=1}^{N} \bar{\phi}_{j}^{i}}$$
(25)

where the superscript "i" refers to the initial distribution of the volume fraction in each case. In addition, for the sake of comparing the computational efficiency, the computational time (CPU time) taken by each scheme is also compared in 3D test cases.

# 4.1. Oblique translation of hollow square

This case stems from Rudman [58], which is adopted to test the performance of the numerical schemes in the convection problem. The centroids of the hollow square are located at (0.8,0.8). The side lengths of the outer and inner side of the hollow square are 0.4 and 0.2, respectively. The domain size is  $4\times4$ , discretized by  $200\times200$  nodes. A time-independent oblique velocity field (u,v)=(2,1) is imposed on the geometry for a period of t=1.25 with the centroid convecting to the final position of (3.3,2.05). The case of hollow square translation will be analyzed in detail to shed light on the benefits by adopting SAH-VOF scheme.

Fig. 4 shows the contour levels of the volume fraction in the case of hollow square translation at the final position, solved by different interface capturing schemes at various Co numbers. The numerical diffusion induced by different schemes can be intuitively exhibited by the shape and distribution of the contour levels of the volume fraction. The denser the contour levels are, the lower the numerical diffusion is while the interface geometry should approach the exact one as much as possible. It can be seen from Fig. 4 that CICSAM shows a good performance at low Co number whereas the numerical diffusion rises as the Co number increases. The interface shapes of the hollow square distort significantly at high Co number in CICSAM. The severe numerical diffusion at high Co number is improved by using STACS, leading to the interface shapes well captured. However, STACS introduces pronounced numerical diffusion at low Co number compared with CICSAM scheme. Compared with CICSAM and STACS, the other five schemes, including FBICS, CUIBS, MSTACS, SAISH and SAH-VOF, could capture the interface with lower numerical diffusion over a wide range of Co numbers. It can be seen from Fig. 4 that a major difference in the interface capturing by using the latter five schemes is their performance at high Co number.

Table 2 compares the  $L_1$  norm of error (namely  $E_{avg}$ ) in the case of hollow square translation simulated by different numerical schemes at various Co numbers. It can be observed that  $E_{avg}$  increases with Co number with CICSAM scheme employed. Compared with CICSAM, a lower  $E_{avg}$  is obtained by STACS at Co=0.75. STACS gets the greatest  $E_{avg}$  at Co=0.25 among all the schemes compared in the present study. The other five schemes perform better than CICSAM and STACS over a wide range of Co numbers. The numerical diffusion in SAH-VOF is close to that in SAISH and the former acts better at Co=0.25. Table 3 shows the order of convergence for SAH-VOF in the case of hollow square translation at Co=0.25, which indicates that SAH-VOF is a first-order scheme. Fig. 5 shows the mass error against time for the case of hollow square translation. It is observable from Fig. 5 that the mass error induced by all schemes (except CICSAM at high Co number) approaches the machine accuracy leading to the mass quite efficiently preserved.

**Table 2** Comparisons of <sup>1</sup> norm of error ( ) in the hollow square translation problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔                                | 𝐶𝑜 = 0.25                                            | 𝐶𝑜 = 0.5                                             | 𝐶𝑜 = 0.75                                            |
|-------------------------------------|------------------------------------------------------|------------------------------------------------------|------------------------------------------------------|
| CICSAM<br>STACS<br>FBICS            | 2.917×10−3<br>5.576×10−3<br>2.438×10−3<br>2.440×10−3 | 4.034×10−3<br>5.642×10−3<br>2.544×10−3<br>2.573×10−3 | 2.501×10−2<br>6.067×10−3<br>3.044×10−3<br>2.999×10−3 |
| CUIBS<br>MSTACS<br>SAISH<br>SAH-VOF | 2.265×10−3<br>2.879×10−3<br>2.445×10−3               | 2.532×10−3<br>2.298×10−3<br>2.382×10−3               | 2.929×10−3<br>2.752×10−3<br>2.962×10−3               |

**Table 3** Order of convergence for SAH-VOF in the hollow square translation problem at = 0*.*25.

| Grids              | 𝐸𝑎𝑣𝑔                     | Order |
|--------------------|--------------------------|-------|
| 100×100<br>200×200 | 2.445×10−3<br>8.504×10−4 | 1.52  |
| 400×400            | 4.296×10−4               | 0.99  |

**Table 4** Comparisons of <sup>1</sup> norm of error ( ) in the hollow square translation problem solved by different interface capturing schemes at = 0*.*25 using grids of varying aspect ratio.

| 𝐸𝑎𝑣𝑔    | Grids      |            |            |  |
|---------|------------|------------|------------|--|
|         | 100 × 200  | 200 × 100  | 200 × 200  |  |
| CICSAM  | 2.688×10−3 | 2.241×10−3 | 2.917×10−3 |  |
| STACS   | 9.574×10−3 | 8.576×10−3 | 5.576×10−3 |  |
| FBICS   | 3.458×10−3 | 2.806×10−3 | 2.438×10−3 |  |
| CUIBS   | 3.442×10−3 | 2.783×10−3 | 2.440×10−3 |  |
| MSTACS  | 2.812×10−3 | 2.173×10−3 | 2.265×10−3 |  |
| SAISH   | 2.807×10−3 | 2.165×10−3 | 2.879×10−3 |  |
| SAH-VOF | 2.819×10−3 | 2.353×10−3 | 2.445×10−3 |  |

As described in Subsection [3.2.3,](#page-6-0) since the parameter of supremum , which determines the CDS in SAH-VOF, is selected based on the CDS in SAISH using the least squares approximation, the comparison between SAISH and SAH-VOF is further conducted. Fig. [6](#page-12-0) plots the contour level of volume fraction in 3D view for the case of hollow square translation solved by SAISH and SAH-VOF where the direction represents the variation of the volume fraction. Since the interface captured by SAH-VOF is similar to that by SAISH when = 0*.*25 and 0*.*5, only the results at = 0*.*75 are illustrated in Fig. [6](#page-12-0) to highlight the difference between these two schemes. Compared with SAH-VOF, an obvious numerical oscillation is induced by SAISH whilst much severer overshoots in the values of the volume fraction can also be observed in Fig. [6a](#page-12-0). The good performance on alleviating the numerical oscillations and the overshoots in the values of the volume fraction exhibited by SAH-VOF may be creditable to the unified form of our novel scheme, rather than the combination of two different schemes in the construction of the classical algebraic VOF method. Further, Fig. [7](#page-13-0) illustrates the comparisons between the interface position solved by different schemes and the analytical solution. The interface position is identified by the contour level of the volume fraction *̄* = 0*.*95. For CICSAM, a disastrous numerical diffusion is induced at high number, resulting in unsuccessful interface capturing. It can be clearly seen that a smoother interface is captured by SAH-VOF at high number in comparison with SAISH, which illuminates the former has a better numerical stability. Besides, the sensibility of different schemes on the grids of varying aspect ratio is also compared in Fig. [8](#page-13-0) with the corresponding <sup>1</sup> norm of error summarized in Table 4. A group of grids with various aspect ratio, namely 100 × 200, 200 × 100 and 200 × 200, is tested for all schemes at = 0*.*25. It is observable from Fig. [8](#page-13-0) and Table 4 that the stability of the resistance on the deformation of the grids for SAH-VOF is comparable to SAISH as well.

# *4.2. Zalesak's slotted disk problem*

In Zalesak's slotted disk problem [[16\]](#page-26-0), a slotted disk rotates by one revolution under a given velocity field and returns to its initial position. This case is utilized to study the performance of the scheme in the shearing problem. The initial center of the disk is situated at (2.0,2.75) with diameter 1 unit. The width and depth of the slot occupy 0.12 and 0.6, respectively. The domain size is 4×4, discretized by 200×200 nodes. The slotted disk suffers a time-independent velocity field, which reads as,

![](_page_11_Figure_2.jpeg)

**Fig. 5.** Comparison of the mass Error between various interface capturing schemes for hollow square translation problem at (a) = 0*.*25, (b) = 0*.*5 and (c) = 0*.*75.

**Table 5** Comparisons of <sup>1</sup> norm of error ( ) in the Zalesak's slotted disk problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔    | 𝐶𝑜 = 0.25  | 𝐶𝑜 = 0.5   | 𝐶𝑜 = 0.75  |
|---------|------------|------------|------------|
| CICSAM  | 1.746×10−3 | 5.328×10−3 | 3.039×10−2 |
| STACS   | 5.334×10−3 | 5.365×10−3 | 5.495×10−3 |
| FBICS   | 2.136×10−3 | 2.224×10−3 | 2.579×10−3 |
| CUIBS   | 2.088×10−3 | 2.168×10−3 | 2.507×10−3 |
| MSTACS  | 1.818×10−3 | 2.123×10−3 | 2.438×10−3 |
| SAISH   | 1.729×10−3 | 1.726×10−3 | 1.902×10−3 |
| SAH-VOF | 1.713×10−3 | 1.797×10−3 | 2.177×10−3 |

$$u = -0.5(y - 2.0)$$
  $v = 0.5(x - 2.0)$  (26)

Fig. [9](#page-14-0) illustrates the final positions of the slotted disk after one revolution, simulated by different interface capturing schemes at various numbers and the corresponding <sup>1</sup> norm of errors are listed in Table 5. In accordance with the translation problem, the numerical diffusion in CICSAM increases with number. In STACS, the numerical diffusion exhibits a weak dependence on number and an unsatisfactory result is obtained at = 0*.*25. Compared with FBICS, CUIBS and MSTACS, the numerical diffusion induced by SAISH and SAH-VOF is smaller over a wide range of numbers as shown in Table 5. Specially, SAH-VOF shows the best performance at low number whereas SAISH exhibits a better character in the middle and high numbers. Table [6](#page-12-0) shows the order of convergence for SAH-VOF in the case of slotted disk at = 0*.*25. It indicates that SAH-VOF yields a first-order accuracy. The mass error against time for all schemes is plotted in Fig. [10.](#page-14-0) In line with the translation problem, the property of mass conservation could be properly preserved by all schemes over a wide range of numbers except CICSAM at high number.

![](_page_12_Figure_2.jpeg)

**Fig. 6.** Contour level of volume fraction for hollow square translation problem solved by (a) SAISH and (b) SAH-VOF at = 0*.*75. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)

**Table 6** Order of convergence for SAH-VOF in the Zalesak's slotted disk problem at = 0*.*25.

| Grids              | 𝐸𝑎𝑣𝑔                     | Order |
|--------------------|--------------------------|-------|
| 100×100<br>200×200 | 4.824×10−3<br>1.713×10−3 | 1.49  |
| 400×400            | 9.218×10−4               | 0.89  |

#### *4.3. 2D shearing field*

This case also comes from Rudman [[58\]](#page-26-0), aiming for investigating the performance of the scheme in the 2D shearing problem. The center of the circle initially lies at (0*.*5*,* 0*.*2(1+)) with diameter ∕5. The domain size is × , discretized by 100×100 nodes. The circle, subjected to a time-independent velocity field, reaches the maximum deformation position at = ∕2. It is followed by returning to the initial flow field during the same time interval with the sign of the velocity field reversed. The velocity field at the first stage is given as,

![](_page_13_Figure_2.jpeg)

**Fig. 7.** Contour level of volume fraction *̄* = 0*.*95 for hollow square translation problem solved by different interface capturing schemes at various numbers. The dash line represents the analytical solution.

![](_page_13_Figure_4.jpeg)

**Fig. 8.** Contour level of volume fraction for hollow square translation problem solved by different interface capturing schemes at = 0*.*25 using grids of varying aspect ratio. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

$$u = cos(x)sin(y) v = -sin(x)cos(y) (27)$$

In the present test, the time period is equal to 31.41.

Figs. [11](#page-15-0) and [12](#page-15-0) show the maximum deformation and final positions of the circle for all schemes at various numbers and the corresponding <sup>1</sup> norm of errors are listed in Table [7.](#page-15-0) In line with the cases of translation and slotted disk, CICSAM performs better at low number than the high one. Since the interface at the final position solved by CICSAM is too diffusive at = 0*.*75, only part of its plot can be shown in Fig. [12.](#page-15-0) In STACS, the interface shapes at = 0*.*25*,* 0*.*5 and 0*.*75 are quite similar to each other for both the maximum deformation and final positions. It is also corresponding with the variation of with number as shown in [7](#page-15-0). The great numerical diffusion over a wide range of numbers in STACS leads to the distortion of the interface shapes at the final position. For the other five schemes, all could capture the interface well with low numerical diffusion at all numbers. Table [8](#page-16-0) shows the order of convergence for SAH-VOF in the case of shearing field at = 0*.*25. It indicates that SAH-VOF could reach first order in this case. Fig. [13](#page-16-0) illustrates the comparison of the mass error between various interface capturing schemes. As indicated in Fig. [13](#page-16-0), the property of mass conservation could be well satisfied by SAH-VOF at all numbers.

#### *4.4. Kawano's translation problem*

This test case is constructed by Kawano [[59\]](#page-26-0), which is employed to study the performance of the numerical schemes in the 3D convection problems. The initial shape is composed of the union of a rectangular parallelepiped and a sphere. The former occupies

![](_page_14_Figure_2.jpeg)

**Fig. 9.** Contour level of volume fraction for Zalesak's slotted disk problem solved by different interface capturing schemes at various numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

![](_page_14_Figure_4.jpeg)

**Fig. 10.** Comparison of the mass Error between various interface capturing schemes for Zalesak's slotted disk problem at (a) = 0*.*25, (b) = 0*.*5 and (c) = 0*.*75.

![](_page_15_Figure_2.jpeg)

**Fig. 11.** Contour level of volume fraction for 2D shearing field problem at the maximum deformation position solved by different interface capturing schemes at various numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

![](_page_15_Figure_4.jpeg)

**Fig. 12.** Contour level of volume fraction for 2D shearing field problem at the final position solved by different interface capturing schemes at various numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

**Table 7** Comparisons of <sup>1</sup> norm of error ( ) in the 2D shearing field problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔    | 𝐶𝑜 = 0.25  | 𝐶𝑜 = 0.5   | 𝐶𝑜 = 0.75  |
|---------|------------|------------|------------|
| CICSAM  | 7.387×10−3 | 2.613×10−2 | 0.156      |
| STACS   | 3.973×10−2 | 3.991×10−2 | 4.019×10−2 |
| FBICS   | 7.696×10−3 | 8.790×10−3 | 1.054×10−2 |
| CUIBS   | 8.144×10−3 | 9.283×10−3 | 1.125×10−2 |
| MSTACS  | 6.593×10−3 | 9.763×10−3 | 1.156×10−2 |
| SAISH   | 5.768×10−3 | 7.087×10−3 | 8.976×10−3 |
| SAH-VOF | 5.921×10−3 | 7.243×10−3 | 9.272×10−3 |

**Table 8** Order of convergence for SAH-VOF in the 2D shearing field problem at = 0*.*25.

| Grids                       | 𝐸𝑎𝑣𝑔                                   | Order        |
|-----------------------------|----------------------------------------|--------------|
| 50×50<br>100×100<br>200×200 | 1.326×10−2<br>5.921×10−3<br>2.594×10−3 | 1.16<br>1.19 |

![](_page_16_Figure_4.jpeg)

**Fig. 13.** Comparison of the mass Error between various interface capturing schemes for 2D shearing field problem at (a) = 0*.*25, (b) = 0*.*5 and (c) = 0*.*75.

the region of [0.08,0.48]×[0.2,0.36]×[0.2,0.36] while the center of the sphere is situated at (0.28,0.28,0.28) with radius 0.15. The computational domain is [0,1]×[0,1]×[0,1], discretized by 128×128×128 nodes. A time-independent oblique velocity field (*, ,*) = (1*,* 1*,* 1) is imposed on the geometry for a time interval of = ∕2 and subsequently the shape convects back to the initial position during the same time interval with the sign of the velocity field reversed. In the present study, the time period is equal to 0.8.

Figs. [14](#page-17-0) and [15](#page-17-0) illustrate the final position of the shape and the corresponding plane ( = 0*.*28) view for different interface capturing schemes at various numbers, respectively. Table [9](#page-17-0) shows in Kawano's translation problem for all schemes at various numbers. It can be seen from Figs. [14](#page-17-0) and [15](#page-17-0) that the shape of the rectangular parallelepiped is well reserved by CICSAM whereas the surface of the sphere is over-compressed especially at middle number. The unsatisfactory resolution for the surface of the sphere leads to the largest in CICSAM at = 0*.*5 among all the simulated schemes. A better capturing for the shape of the sphere is realized by STACS. Unfortunately, it still achieves the largest numerical diffusion at low number, which is in accordance with the trend of in Table [9](#page-17-0). Compared with CICSAM and STACS, the other five schemes could capture the interface well at all numbers. Table [10](#page-18-0) shows the order of convergence of SAH-VOF at = 0*.*25. We can see from Table [10](#page-18-0) that although the order of convergence in this case is a tad lower than first order, it tends to reach first-order accurate as the cell numbers increase. Fig. [16](#page-18-0) illustrates the mass error against time in the case of Kawano's translation. It can be seen from Fig. [16](#page-18-0) that the mass error of all the

![](_page_17_Figure_2.jpeg)

**Fig. 14.** Iso-surface of volume fraction *̄* = 0*.*5 for Kawano's translation problem at the final position solved by different interface capturing schemes at various numbers.

![](_page_17_Figure_4.jpeg)

**Fig. 15.** Contour level of volume fraction for Kawano's translation problem at the final position solved by different interface capturing schemes at various numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

#### **Table 9** Comparisons of <sup>1</sup> norm of error ( ) in the Kawano's translation problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔    | 𝐶𝑜 = 0.25  | 𝐶𝑜 = 0.5   | 𝐶𝑜 = 0.75  |
|---------|------------|------------|------------|
| CICSAM  | 1.459×10−3 | 3.413×10−3 | 1.409×10−2 |
| STACS   | 2.603×10−3 | 2.652×10−3 | 2.735×10−3 |
| FBICS   | 1.114×10−3 | 1.227×10−3 | 1.456×10−3 |
| CUIBS   | 1.065×10−3 | 1.187×10−3 | 1.431×10−3 |
| MSTACS  | 1.016×10−3 | 1.247×10−3 | 1.483×10−3 |
| SAISH   | 9.023×10−4 | 1.069×10−3 | 1.334×10−3 |
| SAH-VOF | 1.072×10−3 | 1.235×10−3 | 1.494×10−3 |

schemes increases considerably when the fluid body moves back to its initial position. It is also visible that good performance on preserving the mass conservation property could be achieved by using SAH-VOF. The computational time using different interface capturing schemes as summarized in Table [11](#page-18-0) shows that SAH-VOF consumes the least time at the same number among all the schemes compared in this study.

**Table 10** Order of convergence for SAH-VOF in the Kawano's translation problem at = 0*.*25.

| Grids       | 𝐸𝑎𝑣𝑔       | Order |
|-------------|------------|-------|
| 64×64×64    | 1.838×10−3 |       |
| 128×128×128 | 1.072×10−3 | 0.78  |
| 256×256×256 | 5.884×10−4 | 0.87  |

![](_page_18_Figure_4.jpeg)

**Fig. 16.** Comparison of the mass Error between various interface capturing schemes for Kawano's translation problem at (a) = 0*.*25, (b) = 0*.*5 and (c) = 0*.*75.

**Table 11** Computation time elapsed in case of Kawano's translation using various interface capturing schemes at different numbers.

| Computation time(s) | 𝐶𝑜 = 0.25 | 𝐶𝑜 = 0.5 | 𝐶𝑜 = 0.75 |
|---------------------|-----------|----------|-----------|
| CICSAM              | 5779.33   | 2933.19  | 2051.06   |
| STACS               | 5781.53   | 2947.57  | 2038.99   |
| FBICS               | 5745.81   | 2935.53  | 2024.89   |
| CUIBS               | 5743.05   | 2933.44  | 2023.31   |
| MSTACS              | 5781.27   | 2947.74  | 2031.01   |
| SAISH               | 5748.14   | 2933.74  | 2025.59   |
| SAH-VOF             | 5678.62   | 2914.87  | 2003.14   |

![](_page_19_Figure_2.jpeg)

**Fig. 17.** Iso-surface of volume fraction *̄* = 0*.*5 for slotted sphere problem at the final position solved by different interface capturing schemes at various numbers.

**Table 12** Comparisons of <sup>1</sup> norm of error ( ) in the slotted sphere problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔    | 𝐶𝑜 = 0.25  | 𝐶𝑜 = 0.5   | 𝐶𝑜 = 0.75  |
|---------|------------|------------|------------|
| CICSAM  | 3.963×10−3 | 1.109×10−2 | 2.776×10−2 |
| STACS   | 7.753×10−3 | 7.762×10−3 | 7.948×10−3 |
| FBICS   | 4.275×10−3 | 4.271×10−3 | 4.860×10−3 |
| CUIBS   | 4.140×10−3 | 4.135×10−3 | 4.652×10−3 |
| MSTACS  | 3.645×10−3 | 4.029×10−3 | 4.464×10−3 |
| SAISH   | 3.511×10−3 | 3.482×10−3 | 4.156×10−3 |
| SAH-VOF | 2.990×10−3 | 3.004×10−3 | 4.133×10−3 |

# *4.5. Slotted sphere problem*

The slotted sphere problem is proposed by Enright et al. [\[60\]](#page-26-0), which can be regarded as the 3D extension of Zalesak's slotted disk problem [\[16\]](#page-26-0). This case is adopted to investigate the characteristic of the schemes in the 3D shearing problem. The center of the sphere is initially located at (0.5,0.72,0.24) with diameter 0.16. The width and depth of the slot are 0.04 and 0.2, respectively. The computational domain is [0,1]×[0,1]×[0,1], discretized by 100×100×48 nodes. The slotted sphere, subjected to a given velocity field, rotates by one revolution and occupies its initial position finally. The velocity field reads as,

$$u = 2\pi/T(0.5 - y)$$
  $v = 2\pi/T(x - 0.5)$   $w = 0$  (28)

In this case, the time period is equal to 6.

Figs. 17 and [18](#page-20-0) show the final position of the slotted sphere after one revolution and the corresponding plane ( = 0*.*24) view for different interface capturing schemes at various numbers, respectively. The <sup>1</sup> norm of errors for all the simulations are listed in Table 12 accordingly. In line with Zalesak's slotted disk problem [\[16](#page-26-0)], a severe numerical diffusion is induced by CICSAM at high number, leading to the distortion of the interface. The interface shape can be recognized by STACS at = 0*.*75 whereas it does not perform well at low number with large numerical diffusion generated. The other five schemes resolve this problem better in comparison with CICSAM and STACS. It can be seen from Fig. 17 that the interface shape captured by SAH-VOF is better than FBICS, CUIBS, MSTACS and SAISH over a wide range of numbers (especially when = 0*.*75). The <sup>1</sup> norm of errors as listed in Table 12 further validates that SAH-VOF induces the least bulk diffusion of interface at the same number for all schemes compared. Table [13](#page-20-0) indicates that SAH-VOF could reach first order in this case. It is clearly observable from Fig. [19](#page-21-0) that compared with the other interface capturing schemes, the mass error induced by SAH-VOF always retains relatively low level at all numbers, which is close to the machine accuracy. On the aspect of the computational efficiency, the time spent by SAH-VOF is almost as much as that by SAISH. See also Table [14.](#page-20-0)

#### *4.6. 3D shearing field*

The case of 3D shearing field performed in this study is first given by Liovic et al. [\[61](#page-26-0)], which is analogous with 2D shearing field problem [[58\]](#page-26-0). In this case, a sphere suffers a given velocity and reaches the maximum deformation position at = ∕2. Subsequently,

![](_page_20_Figure_2.jpeg)

**Fig. 18.** Contour level of volume fraction for slotted sphere problem at the final position solved by different interface capturing schemes at various *Co* numbers. The contour level changes from 0.05 to 0.95 with a step size of 0.1.

**Table 13** Order of convergence for SAH-VOF in the slotted sphere problem at Co = 0.25.

| Grids      | $E_{avg}$              | Order |
|------------|------------------------|-------|
| 50×50×24   | $7.765 \times 10^{-3}$ | _     |
| 100×100×48 | $2.990 \times 10^{-3}$ | 1.38  |
| 200×200×96 | $1.443 \times 10^{-3}$ | 1.05  |

**Table 14**Computation time elapsed in case of slotted sphere using various interface capturing schemes at different *Co* numbers.

| Computation time(s) | Co = 0.25 | Co = 0.5 | Co = 0.75 |
|---------------------|-----------|----------|-----------|
| CICSAM              | 1602.00   | 833.67   | 575.12    |
| STACS               | 1593.85   | 827.54   | 573.11    |
| FBICS               | 1590.59   | 854.17   | 568.23    |
| CUIBS               | 1578.99   | 820.48   | 568.59    |
| MSTACS              | 1599.02   | 826.10   | 572.03    |
| SAISH               | 1587.21   | 823.87   | 570.13    |
| SAH-VOF             | 1582.85   | 824.25   | 570.01    |

it travels back to its initial position along the same path. The center of the sphere initially lies at (0.5,0.75,0.25) with diameter 0.15. The domain size is  $1\times1\times1$ , discretized by  $128\times128\times128$  nodes. The velocity field is given as,

$$u = -\sin(2\pi y)\sin^2(\pi x)\cos(\pi t/T)$$

$$v = \sin(2\pi x)\sin^2(\pi y)\cos(\pi t/T)$$

$$w = (1 - 2r)^2\cos(\pi t/T)$$
(29)

where  $r = \sqrt{(x - 0.5)^2 + (y - 0.5)^2}$  and T = 3.

Figs. 20 and 21 show the iso-surfaces of the volume fraction ( $\bar{\phi}=0.5$ ) at the maximum deformation position for different interface capturing schemes at various Co numbers in the XY and YZ planes, respectively. The corresponding  $L_1$  norm of errors are listed in Table 15. It can be seen from Fig. 21 that CICSAM performs satisfactorily at low Co number whereas the fluid body can not be completely captured at Co=0.75 due to the large numerical diffusion. Accordingly,  $E_{avg}$  increases with Co number as shown in Table 15. The  $L_1$  norm of error in STACS mildly increases with Co number. Compared with the other cases, STACS induces the largest numerical diffusion at Co=0.25. The iso-surfaces of volume fraction in the other five schemes are quite similar to each other over a wide range of Co numbers. However, compared with the other interface capturing schemes, SAH-VOF induces the least numerical diffusion when Co=0.25 and 0.5 as shown in Table 15. It is visible from Table 16 that SAH-VOF yields a first-order accuracy. Fig. 22 plots the variation of mass error with time using different interface capturing schemes. Obviously, the mass conservation property can be preserved as well in this case. Specially, the least mass error is induced by adopting SAH-VOF at all Co numbers during the

![](_page_21_Figure_2.jpeg)

**Fig. 19.** Comparison of the mass Error between various interface capturing schemes for slotted sphere problem at (a) = 0*.*25, (b) = 0*.*5 and (c) = 0*.*75.

![](_page_21_Figure_4.jpeg)

**Fig. 20.** Iso-surface of volume fraction *̄* = 0*.*5 in the plane view for 3D shearing field problem at the maximum deformation position solved by different interface capturing schemes at various numbers.

stage of the fluid body with maximum deformation returning back to its initial position. In accordance with the case of Kawano's translation, the least computational time is used by SAH-VOF at the same number among all the schemes as summarized by Table [17.](#page-22-0)

![](_page_22_Figure_2.jpeg)

**Fig. 21.** Iso-surface of volume fraction *̄* = 0*.*5 in the  plane view for 3D shearing field problem at the maximum deformation position solved by different interface capturing schemes at various numbers.

**Table 15** Comparisons of <sup>1</sup> norm of error ( ) in the 3D shearing field problem using various interface capturing schemes at different numbers.

| 𝐸𝑎𝑣𝑔    | 𝐶𝑜 = 0.25  | 𝐶𝑜 = 0.5   | 𝐶𝑜 = 0.75  |
|---------|------------|------------|------------|
| CICSAM  | 1.163×10−3 | 3.882×10−3 | 1.575×10−2 |
| STACS   | 3.672×10−3 | 3.676×10−3 | 3.682×10−3 |
| FBICS   | 1.026×10−3 | 1.028×10−3 | 1.171×10−3 |
| CUIBS   | 1.003×10−3 | 1.007×10−3 | 1.122×10−3 |
| MSTACS  | 9.220×10−4 | 9.940×10−4 | 1.111×10−3 |
| SAISH   | 8.386×10−4 | 8.441×10−4 | 1.136×10−3 |
| SAH-VOF | 7.799×10−4 | 8.316×10−4 | 1.208×10−3 |
|         |            |            |            |

**Table 16** Order of convergence for SAH-VOF in the 3D shearing field problem at = 0*.*25.

| Grids                      | 𝐸𝑎𝑣𝑔                     | Order        |
|----------------------------|--------------------------|--------------|
| 64×64×64                   | 1.909×10−3               |              |
| 128×128×128<br>256×256×256 | 7.799×10−4<br>3.679×10−4 | 1.29<br>1.08 |

**Table 17** Computation time elapsed in case of 3D shearing field using various interface capturing schemes at different numbers.

| Computation time(s) | 𝐶𝑜 = 0.25 | 𝐶𝑜 = 0.5 | 𝐶𝑜 = 0.75 |
|---------------------|-----------|----------|-----------|
| CICSAM              | 7732.16   | 4003.89  | 2771.52   |
| STACS               | 7721.67   | 4032.00  | 2793.74   |
| FBICS               | 7735.25   | 4009.63  | 3020.20   |
| CUIBS               | 7668.58   | 4037.96  | 2784.82   |
| MSTACS              | 7698.63   | 4006.27  | 2787.64   |
| SAISH               | 7710.50   | 4003.46  | 2784.38   |
| SAH-VOF             | 7659.92   | 3963.55  | 2766.05   |

#### *4.7. Elliptical vortex in a shear flow*

The detailed descriptions of the exact solution with regards to elliptical vortex in a shear flow were given by Chaplygin [\[36\]](#page-26-0), and Meleshkov and Heijst [\[38](#page-26-0)] gave a brief account of Chaplygin's results. This benchmark aims to investigate the performance of SAH-VOF in the complex and real problem. The center of the initial elliptical vortex lies at (0*.*0*,* 0*.*0) with its principal axes <sup>0</sup> = 2*.*0

![](_page_23_Figure_2.jpeg)

Fig. 22. Comparison of the mass Error  $E_m$  between various interface capturing schemes for 3D shearing field problem at (a) Co = 0.25, (b) Co = 0.5 and (c) Co = 0.75.

and  $b_0 = 1.0$ , where the subscript "0" represents the time t = 0.0. The domain size is  $[-5.0, 5.0] \times [-5.0, 5.0]$ , discretized by 500×500 nodes. The velocity field inside (index i) and outside (index e) the elliptical cylinder is given by,

$$u_{i} = -2Ay - \frac{2\omega ab}{a+b} \left(\frac{x'\sin\phi'}{a} + \frac{y'\cos\phi'}{b}\right)$$

$$v_{i} = \frac{2\omega ab}{a+b} \left(\frac{x'\cos\phi'}{a} - \frac{y'\sin\phi'}{b}\right)$$

$$u_{e} = -2Ay - \frac{2\omega ab}{\alpha'+\beta'} \left(\frac{x'\sin\phi'}{\alpha'} + \frac{y'\cos\phi'}{\beta'}\right)$$

$$v_{e} = \frac{2\omega ab}{\alpha'+\beta'} \left(\frac{x'\cos\phi'}{\alpha'} - \frac{y'\sin\phi'}{\beta'}\right)$$
(30)

where A=1.0,  $\omega=100.0$  and  $\phi_0'=0.0$ . The definitions of the variables in Eq. (30) could be referred to the study of Meleshkov and Van Heijst [38]. In order to eliminate ambiguity, the variables of  $\alpha$ ,  $\beta$  and  $\phi$  in Eqs. (2.7)-(2.8) in [38] are substituted by  $\alpha'$ ,  $\beta'$  and  $\phi'$ , respectively, in Eq. (30). Chaplygin showed that for  $m=\omega/A>0$  there would be three different types of motion for the elliptical vortex, relying on the value of  $m \ln K/4$ . The definition of  $m \ln K/4$  is given as follows,

$$m\ln K/4 = \frac{\sin^2 \phi_0'}{z_0} + z_0 \cos^2 \phi_0' - m\ln \frac{4z_0}{(1+z_0)^2}$$
(31)

where the initial eccentricity  $z_0 = b_0/a_0 = 0.5$ . It can be seen from Eq. (31) that the value of  $m \ln K/4$  only depends on the initial values of  $\phi'_0$ ,  $z_0$  and the parameter m. In the present study, it is easy to get the value of  $m \ln K/4 \approx 12.28 > 1$ , leading to the elliptical patch obeying the rule of rotating counterclockwise with non-uniform angular velocity.

Fig. 23 shows the contour of volume fraction when the elliptical patch rotates 180 degrees counterclockwise solved by different interface capturing schemes at various Co numbers. In line with the above-mentioned cases, CICSAM performs better at low and middle Co number and severe dissipation is induced as the Co number increases. In STACS, the numerical dissipation still exhibits

![](_page_24_Figure_2.jpeg)

Fig. 23. Contour of volume fraction when the elliptical patch rotates 180 degrees counterclockwise solved by different interface capturing schemes at various Conumbers.

quite similar extent at various Co numbers and a better interface capturing is obtained by STACS in comparison with CICSAM at Co = 0.75. An interface with low numerical diffusion could be captured by all the other five schemes at all Co numbers. Specially, among these five schemes, a lower dissipation is induced by SAH-VOF at Co = 0.25 and 0.5. Since the Co number is usually restricted to a middle or low value in the practical simulations to guarantee the stability of the computation, SAH-VOF has a better applicability in the real problems.

#### 4.8. Dipolar vortex moving along a straight line

This case also comes from Chaplygin [37], aiming for checking the performance of the scheme in the practical problem. The center of the circle initially lies at (0.0,0.0) with diameter R = 2.0. The domain size is  $[-5.0,5.0] \times [-5.0,5.0]$ , discretized by  $500 \times 500$  nodes. The motion outside the circle is irrotational and the dipolar vortex uniformly translates with a continuous velocity distribution. The stream-functions inside (index i) and outside (index e) the circle are given as follows,

$$\psi_{i} = -\frac{2UJ_{1}(kr)}{kJ_{0}(kR)} \frac{y}{r}$$

$$\psi_{e} = U(r - \frac{R^{2}}{r}) \frac{y}{r}$$
(32)

where  $r = \sqrt{x^2 + y^2}$ , U = 1.0,  $kR/\pi = 1.2197$  and  $J_s$  is a Bessel's function (s = 0, 1). Thus the velocity components (u, v) can be obtained via,

$$u = \frac{\partial \psi}{\partial v} \qquad v = -\frac{\partial \psi}{\partial x} \tag{33}$$

Fig. 24 shows the contour of volume fraction after the dipolar vortex translates 10 seconds for all Co numbers using different interface capturing schemes. It can be seen from Fig. 24 that the variation of the numerical dissipation by using CICSAM and STACS with the Co numbers still has the same trend with the previous cases. For the other five schemes, the numerical dissipation is well dominated especially at the middle and high Co numbers in comparison with CICSAM and STACS. Besides, it is worth mentioning that the symmetry of the flow field is better maintained by SAH-VOF among these five schemes and an interface with low dissipation could still be obtained by SAH-VOF even for the flow with high Co number.

#### 5. Conclusion

An interface capturing scheme called SAH-VOF is proposed in this paper. This method is efficiently realized in three steps: (1) unlike the classical algebraic VOF method using two different schemes to implement the construction, the same expression, namely SAH scheme, is utilized for both HRS and CDS with only different steepness parameter (namely infimum and supremum, respectively); (2) the switching technique in the classical algebraic VOF method is extended to obtain the final steepness parameter in terms of the infimum and supremum, leading to the resultant SAH-VOF scheme with a quite simple expression; (3) the infimum and supremum of  $\beta$  are properly determined with a strategy based on least squares approximation for the latter being proposed. The SAH-VOF scheme has been tested in 2D and 3D cases as well as two benchmark problems with the exact vortex solutions referred to the nonlinear Euler equations over a wide range of Co numbers with the help of contour levels and iso-surfaces of volume fraction,

![](_page_25_Figure_2.jpeg)

**Fig. 24.** Contour of volume fraction at =10 s solved by different interface capturing schemes at various numbers.

error analysis and computational time, and the corresponding simulations were also made by using CICSAM, STACS, FBICS, CUIBS, MSTACS and SAISH for comparisons. The results indicate that SAH-VOF can realize the interface capturing with low numerical diffusion for all the tests over a wide range of numbers. The numerical oscillations could be reduced along with the phenomenon of the overshoots in the values of the volume fraction alleviated since a unified form is used by SAH-VOF, rather than the direct combination of two different schemes which is widely adopted in the classical algebraic VOF method. Besides, SAH-VOF shows an acceptable computational efficiency for the tests conducted in this study. The order of convergence reveals that SAH-VOF yields a first-order accuracy. In future, SAH-VOF will be applied in practical problems to further investigate its performance in real complex flows.

# **CRediT authorship contribution statement**

**Weidan Ni:** Formal analysis, Methodology, Writing – original draft, Writing – review & editing, Software. **Qinghong Zeng:** Supervision, Writing – review & editing. **Yucang Ruan:** Writing – review & editing, Software. **Zhiwei He:** Conceptualization, Methodology, Supervision, Writing – review & editing.

# **Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

#### **Data availability**

Data will be made available on request.

# **References**

- [1] A. [Haselsteiner,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibD08A480B1E9A6D68F9FEB2A8A33C9AECs1) K.-D. Thoben, Predicting wave heights for marine design by prioritizing extreme events in a global model, Renew. Energy 156 (2020) 1146–1157.
- [2] N. Eric, M. Kinyanjui, J. Abonyo, Two-phase turbulent fluid flow in a [geothermal](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib3032847D30A9703805D449721FA2A131s1) pipe with chemical reaction, J. Appl. Math. 2022 (2022) 7617017.
- [3] D.-A. Bi, M. Tavares, E. Chénier, S. Vincent, Accuracy and convergence of the curvature and normal vector [discretizations](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE639480CF30E19B095256A6F032A0834s1) for 3d static and dynamic front-tracking [interfaces,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE639480CF30E19B095256A6F032A0834s1) J. Comput. Phys. 461 (2022) 111197.
- [4] M. Hossain, S. Pimentel, J. Stockie, Simulating surface height and terminus position for marine outlet glaciers using a level set method with data [assimilation,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibA2C4B2CC43B431B4732F73DF73D27E01s1) J. [Comput.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibA2C4B2CC43B431B4732F73DF73D27E01s1) Phys. 474 (2022) 111766.
- [5] A. Mohan, G. Tomar, Interface [reconstruction](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE85D9FED533DA619AF6360DF6D889336s1) and advection schemes for volume of fluid method in axisymmetric coordinates, J. Comput. Phys. 446 (1) (2021) [110663.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE85D9FED533DA619AF6360DF6D889336s1)
- [6] F. Denner, F. Evrard, B. van Wachem, [Breaching](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib72C3FC6FC19668FA2B35A69C8B506E49s1) the capillary time-step constraint using a coupled vof method with implicit surface tension, J. Comput. Phys. 459 (2022) [111128.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib72C3FC6FC19668FA2B35A69C8B506E49s1)
- [7] W.F. Noh, P. Woodward, Slic (simple line interface calculation), in: Proceedings of the Fifth [International](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib078A61F682ADC51E27FF57282F18A871s1) Conference on Numerical Methods in Fluid Dynamics June 28–July 2, 1976 Twente University, Enschede, Springer, 1976, [pp. 330–340.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib078A61F682ADC51E27FF57282F18A871s1)
- [8] E.G. Puckett, A.S. Almgren, J.B. Bell, D.L. Marcus, W.J. Rider, A high-order projection method for tracking fluid interfaces in variable density [incompressible](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibD904B610A3080891033EBB4F0B9A42B5s1) flows, J. Comput. Phys. 130 (2) (1997) [269–282.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibD904B610A3080891033EBB4F0B9A42B5s1)
- [9] W.J. Rider, D.B. Kothe, [Reconstructing](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibCCAEB1ED96B16122F0ECABB077B61476s1) volume tracking, J. Comput. Phys. 141 (2) (1998) 112–152.
- [10] C. Wu, D. Young, H. Wu, Simulations of [multidimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1D2E4C5FFEA4D68B2D0CFFE40E456A45s1) interfacial flows by an improved volume-of-fluid method, Int. J. Heat Mass Transf. 60 (2013) 739–755.

- [11] F. Denner, B. van Wachem, [Compressive](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib87B4C0AF0011B562071B64FC760C3FEEs1) vof method with skewness correction to capture sharp interfaces on arbitrary meshes, J. Comput. Phys. 279 (2014) [127–144.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib87B4C0AF0011B562071B64FC760C3FEEs1)
- [12] D. Zhang, C. Jiang, D. Liang, Z. Chen, Y. Yang, Y. Shi, A refined [volume-of-fluid](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibEBAE7F803442DD8806EED0DD2926C4EAs1) algorithm for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 274 (2014) [709–736.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibEBAE7F803442DD8806EED0DD2926C4EAs1)
- [13] K. [Pericleous,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibB8DEC1B7858F3D55901451EABA5DE0C1s1) K. Chan, M. Cross, Free surface flow and heat transfer in cavities: the sea algorithm, Numer. Heat Transf. 27 (4) (1995) 487–507.
- [14] R. LeVeque, [High-resolution](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibB2E19BB6269EC9CC28DA7DB726F41F89s1) conservative algorithms for advection in incompressible flow, SIAM J. Numer. Anal. 33 (2) (1996) 627–665.
- [15] T. Bonometti, J. Magnaudet, An [interface-capturing](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibEC7FD269DFB820CF69725967A06240DEs1) method for incompressible two-phase flows. Validation and application to bubble dynamics, Int. J. Multiph. Flow 33 (2) (2007) [109–133.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibEC7FD269DFB820CF69725967A06240DEs1)
- [16] S. Zalesak, Fully [multidimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib66F0C5F6A50A2774CBB7EE8B5BFD4DF4s1) flux-corrected transport algorithms for fluids, J. Comput. Phys. 31 (3) (1979) 335–362.
- [17] V. Gopala, B. vanWachem, Volume of fluid methods for [immiscible-fluid](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib79680C3DF645501B959F59D9E5904962s1) and free-surface flows, Chem. Eng. J. 141 (1–3) (2008) 204–221.
- [18] H. Rusche, [Computational](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib01010A42E60948C7A97E30BD25E7BB1Fs1) fluid dynamics of dispersed two-phase flows at high phase fractions, Ph.D. thesis, Imperial College of Science, Technology and [Medicine,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib01010A42E60948C7A97E30BD25E7BB1Fs1) 2002.
- [19] D. Walters, N. Wolgemuth, A new [interface-capturing](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib6A60EB7E03CD3F0C085D6F50CE5167D2s1) discretization scheme for numerical solution of the volume fraction equation in two-phase flows, Int. J. Numer. Methods Fluids 60 (8) (2009) [893–918.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib6A60EB7E03CD3F0C085D6F50CE5167D2s1)
- [20] J. Thuburn, [Multidimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib2DE8F4007B88E8536D8A0B45FD1A2385s1) flux-limited advection schemes, J. Comput. Phys. 123 (1) (1996) 74–83.
- [21] E. Dendy, N. Padial-Collins, W. VanderHeyden, A [general-purpose](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibC99CC96F0F4EED6928EBF5599D9F6F68s1) finite-volume advection scheme for continuous and discontinuous fields on unstructured grids, J. Comput. Phys. 180 (2) (2002) [559–583.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibC99CC96F0F4EED6928EBF5599D9F6F68s1)
- [22] F. Xiao, A. Ikebata, An efficient method for capturing free boundaries in multi-fluid [simulations,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib784769FEC51E96B31FD83566533E5E64s1) Int. J. Numer. Methods Fluids 42 (2) (2003) 187–210.
- [23] K. Yokoi, Efficient [implementation](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibB8936525E12BF337C1A609A6A06D3A86s1) of thinc scheme: a simple and practical smoothed vof algorithm, J. Comput. Phys. 226 (2) (2007) 1985–2002.
- [24] I. Park, K. Kim, J. Kim, A [volume-of-fluid](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibC67694090F9A5DE7304C26E58F828FCEs1) method for incompressible free surface flows, Int. J. Numer. Methods Fluids 61 (2009) 1331–1362.
- [25] F. Denner, D. van der Heul, G. Oud, M. Villar, A. da Silveira Neto, B. van Wachem, Comparative study of [mass-conserving](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib33F65B3015676A1B58A1155DE0D62CADs1) interface capturing frameworks for [two-phase](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib33F65B3015676A1B58A1155DE0D62CADs1) flows with surface tension, Int. J. Multiph. Flow 61 (2014) 37–47.
- [26] O. Ubbink, R. Issa, A method for capturing sharp fluid [interfaces](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib90F988AE18E909F2C357CCEA08A4795Cs1) on arbitrary meshes, J. Comput. Phys. 153 (1) (1999) 26–50.
- [27] M. Darwish, F. Moukalled, Convective schemes for capturing interfaces of free-surface flows on [unstructured](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib10AA1099DACEB2F4E21ECB3468801C96s1) grids, Numer. Heat Transf., Part B, Fundam. 49 (1) [\(2006\)](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib10AA1099DACEB2F4E21ECB3468801C96s1) 19–42.
- [28] Y.-Y. Tsui, S.-W. Lin, T.-T. Cheng, T.-C. Wu, [Flux-blending](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib45E9E621CF2802FB6F5256F814482D6Es1) schemes for interface capture in two-fluid flows, Int. J. Heat Mass Transf. 52 (23–24) (2009) [5547–5556.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib45E9E621CF2802FB6F5256F814482D6Es1)
- [29] J. Patel, G. Natarajan, A generic [framework](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib399201F1696AA792309E05EC56A21599s1) for design of interface capturing schemes for multi-fluid flows, Comput. Fluids 106 (2015) 108–118.
- [30] C. Anghan, M. Bade, J. Banerjee, A modified switching [technique](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib2D52D8E030662A80C9E095FE8E76783As1) for advection and capturing of surfaces, Appl. Math. Model. 92 (2021) 349–379.
- [31] A. Arote, M. Bade, J. Banerjee, An improved compressive volume of fluid scheme for capturing sharp interfaces using [hybridization,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1A92F54177200088990C4585C172A230s1) Numer. Heat Transf., Part B, [Fundam.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1A92F54177200088990C4585C172A230s1) (2020) 1–25.
- [32] J. Larsson, B. Gustafsson, Stability criteria for hybrid difference methods, J. Comput. Phys. 227 (2008) [2886–2898.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib43ADAA83A1068D61BA527C624705318Ds1)
- [33] Z. He, Y. Ruan, Y. Yu, B. Tian, F. Xiao, Self-adjusting [steepness-based](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib82B8337EB030F8412A150D2D1908C38Es1) schemes that preserve discontinuous structures in compressible flows, J. Comput. Phys. 463 (2022) [111268.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib82B8337EB030F8412A150D2D1908C38Es1)
- [34] F. Xiao, T. Honma, T. Kono, A simple algebraic interface capturing scheme using [hyperbolic](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib33E74FCEAE5CFAAB099DF254B22FFCC0s1) tangent function, Int. J. Numer. Methods Fluids 48 (2005) [1023–1040.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib33E74FCEAE5CFAAB099DF254B22FFCC0s1)
- [35] B. van Leer, Towards the ultimate conservative difference scheme. II. [Monotonicity](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibC372DAC84D6E54F77AD72FA2E114DEECs1) and conservation combined in a second-order scheme, J. Comput. Phys. 14 (4) (1974) [361–370.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibC372DAC84D6E54F77AD72FA2E114DEECs1)
- [36] S. [Chaplygin,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib24D095B9A97550E51FFE0D0CCA6221ADs1) On a pulsating cylindrical vortex, Trans. Phys. Sect. Mosc. Soc. Friends Nat. Sci. 10 (1899) 13–22.
- [37] S. [Chaplygin,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibDE4418E4E905698AD6CCAE4B936D819Fs1) One case of vortex motion in fluid, Trans. Phys. Sect. Mosc. Soc. Friends Nat. Sci. 11 (1903) 11–14.
- [38] V. Meleshko, G. Van Heijst, On Chaplygin's investigations of [two-dimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibD41D1B956C76F254FFBB2F8EC6E9AB46s1) vortex structures in an inviscid fluid, J. Fluid Mech. 272 (1994) 157–182.
- [39] S. Gottlieb, C.W. Shu, Total variation diminishing [Runge-Kutta](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibB1BF1D10F04F31FDD713A2035E1CE405s1) schemes, Math. Comput. 67 (1998) 73–85.
- [40] H. Gaskell, A. Lau, Curvature-compensated convective transport: smart, a new [boundedness-preserving](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE04DF328515DC041B9B05BFCE8B7662Ds1) transport algorithm, Int. J. Numer. Methods Fluids 8 (1988) [617–641.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibE04DF328515DC041B9B05BFCE8B7662Ds1)
- [41] B. Leonard, The ultimate conservative difference scheme applied to unsteady [one-dimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibA1D0BC55E66343CBD944413F49DBF8ECs1) advection, Comput. Methods Appl. Mech. Eng. 88 (1) (1991) [17–74.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibA1D0BC55E66343CBD944413F49DBF8ECs1)
- [42] B. Parker, D. Youngs, Two and three [dimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibA6200775F293600266DA758A0BEC44EEs1) Eulerian simulation of fluid flow with material interfaces, Atom. Weapons Establish. (1992).
- [43] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free [boundaries,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib4A6E82832EE49CC14CE5E1958C0E7080s1) J. Comput. Phys. 39 (1) (1981) 201–225.
- [44] B. Lafaurie, C. Nardone, R. Scardovelli, S. Zaleski, G. Zanetti, Modelling merging and [fragmentation](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1BE9336A7E86A48E359D8982EE423399s1) in multiphase flows with surfer, J. Comput. Phys. 113 (1) (1994) [134–147.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1BE9336A7E86A48E359D8982EE423399s1)
- [45] B. Leonard, H. Niknafs, Sharp monotonic resolution of [discontinuities](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib3051FFCB687629475FA2639DB3780C91s1) without clipping of narrow extrema, Comput. Fluids 19 (1) (1991) 141–154, special Issue [CTAC-89.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib3051FFCB687629475FA2639DB3780C91s1)
- [46] B.P. Leonard, A stable and accurate convective modelling procedure based on quadratic upstream [interpolation,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib663178C16F49560A4A37988E01B27F54s1) Comput. Methods Appl. Mech. Eng. 19 (1)
- [\(1979\)](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib663178C16F49560A4A37988E01B27F54s1) 59–98. [47] Y.Y. Tsui, S.W. Lin, T.T. Cheng, T.C. Wu, [Flux-blending](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib257079860A683A07EA25B9D875CB1EBDs1) schemes for interface capture in two-fluid flows, Int. J. Heat Mass Transf. 52 (23–24) (2009) 5547–5556.
- [48] B. [Chakraborty,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibCDB4989FEE81ADB7F9A6C850B96DD403s1) J. Banerjee, A sharpness preserving scheme for interfacial flows, Appl. Math. Model. 40 (21–22) (2016) 9398–9426.
- [49] M. Darwish, A new [high-resolution](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib123436C8B9385DB14E4E2F9D22E1647Es1) scheme based on the normalized variable formulation, Numer. Heat Transf., Part B, Fundam. 24 (3) (1993) 353–371.
- [50] N. Waterson, H. Deconinck, Design principles for bounded [higher-order](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib271DD29626E9BF727F576305942686DDs1) convection schemes a unified approach, J. Comput. Phys. 224 (2007) 182–207.
- [51] J. Zhu, A low-diffusive and [oscillation-free](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib3FB95FEB8A8B35A77CEC94EDCEE8909Ds1) convection scheme, Commun. Appl. Numer. Methods 7 (3) (1991) 225–232.
- [52] J. Fromm, A method for reducing dispersion in [convective](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibB8E528577446199A2DE02D72AB048B3Bs1) difference schemes, J. Comput. Phys. 3 (2) (1968) 176–189.
- [53] Z. He, Y. Zhang, F. Gao, X. Li, B. Tian, An improved accurate [monitonicity-preserving](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib79D9E1AFE4E52DB80EC08FAFC23D2734s1) scheme for the Euler equations, Comput. Fluids 140 (2016) 1–10.
- [54] F. Xiao, S. Ii, C. Chen, Revisit to the thinc scheme: a simple algebraic vof algorithm, J. Comput. Phys. 230 (19) (2011) [7086–7092.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibFD7CBC2ECA05BF73216DD04802AF69B9s1)
- [55] V. [Venkatakrishnan,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib272929B2EF1B5DB7C4CC62D800E4D282s1) On the accuracy of limiters and convergence to steady-state solutions, AIAA paper 93 (1993) 0880.
- [56] C. Michalak, C. [Ollivier-Gooch,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib61E9FC115CA2317D8ACC2C7E90F0CD55s1) Accuracy preserving limiter for the high-order accurate solution of the Euler equations, J. Comput. Phys. 228 (2009) 8693–8711.
- [57] A. Baraldi, M. Dodd, A. Ferrante, A [mass-conserving](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib531A4DE0F9B38FD5185E87F19293F235s1) volume-of-fluid method: volume tracking and droplet surface-tension in incompressible isotropic turbulence, Comput. Fluids 96 (2014) [322–337.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib531A4DE0F9B38FD5185E87F19293F235s1)
- [58] M. Rudman, [Volume-tracking](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib26632D8CE70675B680BB8E17401B6DF9s1) methods for interfacial flow calculations, Int. J. Numer. Methods Fluids 24 (7) (1997) 671–691.
- [59] A. Kawano, A simple volume-of-fluid reconstruction method for [three-dimensional](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib1B6A1C539E18DBFDF5213D6FBD6F8822s1) two-phase flows, Comput. Fluids 134–135 (2016) 130–145.
- [60] D. Enright, R. Fedkiw, J. Ferziger, I. Mitchell, A hybrid particle level set method for improved interface [capturing,](http://refhub.elsevier.com/S0021-9991(24)00014-7/bib301015ED04129293905C1ABA4DE2865Cs1) J. Comput. Phys. 183 (1) (2002) 83–116.
- [61] P. Liovic, M. Rudman, J.-L. Liow, D. Lakehal, D. Kothe, A 3d unsplit-advection volume tracking algorithm with [planarity-preserving](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibADC33C23E7AE4A002E36571710C486C6s1) interface reconstruction, Comput. Fluids 35 (10) (2006) [1011–1032.](http://refhub.elsevier.com/S0021-9991(24)00014-7/bibADC33C23E7AE4A002E36571710C486C6s1)