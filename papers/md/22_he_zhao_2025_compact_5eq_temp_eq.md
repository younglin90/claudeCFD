LETTER | DECEMBER 03 2025

# Compact representation of the generic five-equation model with additional temperature-equilibrium

Zhiwei He 🛂 🕞 ; Shuo Zhao 🕞

![](_page_0_Picture_5.jpeg)

Physics of Fluids 37, 121701 (2025) https://doi.org/10.1063/5.0304718

![](_page_0_Picture_7.jpeg)

![](_page_0_Picture_8.jpeg)

# **Articles You May Be Interested In**

A multicomponent real-fluid fully compressible four-equation model for two-phase flow with phase change Physics of Fluids (February 2019)

Microfluidics-enabled functional 3D printing

Biomicrofluidics (March 2022)

A high-fidelity interface sharpening method for the five-equation model in compressible multiphase flows Physics of Fluids (July 2025)

![](_page_0_Figure_14.jpeg)

![](_page_0_Picture_15.jpeg)

# Compact representation of the generic five-equation model with additional temperature-equilibrium 📵

Cite as: Phys. Fluids 37, 121701 (2025); doi: 10.1063/5.0304718 Submitted: 30 September 2025 · Accepted: 9 November 2025 · Published Online: 3 December 2025

![](_page_1_Picture_6.jpeg)

![](_page_1_Picture_7.jpeg)

![](_page_1_Picture_8.jpeg)

Zhiwei He<sup>1,2,a)</sup> (in) and Shuo Zhao<sup>1,2</sup> (in)

![](_page_1_Picture_10.jpeg)

![](_page_1_Picture_12.jpeg)

#### **AFFILIATIONS**

- <sup>1</sup>Institute of Applied Physics and Computational Mathematics, Beijing 100094, China
- <sup>2</sup>National Key Laboratory of Computational Physics, Beijing 100088, China

### **ABSTRACT**

This paper presents a compact formulation of the generic five-equation (GFE) model for compressible multimaterial flows under additional temperature-equilibrium. By analyzing the pressureand temperature-equilibrium closure used in its equivalent model (i.e., four-equation model), a novel method for deriving the distribution coefficient (DC) that governs the evolution of volume fractions in the GFE model is presented. Subsequently, through the introduction of three commonly used thermodynamic derivatives (i.e., the coefficient of thermal expansion, the isothermal compressibility, and the specific heat at constant pressure), we ultimately obtain the formula of DC in the GFE model with additional temperature-equilibrium that is not only simple but also highly compact in form. Numerical verifications confirm the correctness of the derivation.

### I. INTRODUCTION

Compressible flows involving multiple materials are ubiquitous in both fundamental and applied sciences, including shock-interface interactions, underwater explosions, and astrophysical phenomena. Based on their importance and wide applicability, modeling such flows

Recently, a generic five-equation (GFE) model with instantaneous velocityand pressure-equilibrium closure has been proposed to describe the above flows. Considering cases with an arbitrary number K of materials, the GFE model is composed of K phasic mass balance equations, one mixture momentum equation, one mixture total energy equation, and K evolution equations for the volume fraction of each material with a source term including a distribution coefficient (DC) to characterize the interactions between materials. This model is general in that it can recover two existing specific five-equation models<sup>2</sup> and generate new specific models with proper DCs derived by additional assumptions.1,

On the other hand, instantaneous velocity-, pressure-, and temperature-(PT) equilibrium closure continues to be a popular choice in the majority of engineering applications, such as detonation waves in condensed explosives (although the assumption of thermal equilibrium is physically questionable<sup>4,8</sup>) phase change,<sup>9</sup> and conductive heat transfer. 9,10 This temperature-equilibrium model (four-equation model) involves four partial differential equations (K phasic mass balance equations, one mixture momentum equation, and one mixture energy equation) and is fully conservative and hyperbolic. However, this model is notorious for generating spurious pressure and temperature oscillations at material interfaces.<sup>2</sup>

Noting that the additional temperature-equilibrium condition implies that one of the partial differential equations is not required, we can obtain the four-equation model by removing the volume fraction equation, which is nonconservative in the GFE model.<sup>6</sup> That is to say, the GFE model with additional temperature-equilibrium is theoretically equivalent to the classical four-equation model. The corresponding DC that arises under this additional temperature-equilibrium is theoretically obtained<sup>6</sup> with the help of the formula of the mixed speed of sound (SoS) in the four-equation model. 13,14 However, the derived DC is too complicated to be efficiently implemented due to the complex form of the mixed SoS formula.<sup>13</sup>

In this article, by carefully analyzing the PT equilibrium closure mentioned above and some relevant studies, 15-17 we propose a novel method for calculating the DC that arises in the GFE model with additional temperature-equilibrium. The new method theoretically provides the DC in a very simple and compact form.

<sup>&</sup>lt;sup>a)</sup>Author to whom correspondence should be addressed: he\_zhiwei@iapcm.ac.cn

# II. GENERIC FIVE-EQUATION MODEL

In this section, we briefly provide the basic knowledge of the GFE model and some corresponding DCs and mixed SoSs in specific cases. More information can be found in Refs. 1 and 6.

#### A. Governing equations

The GFE model for compressible multimaterial hydrodynamics with an arbitrary number of materials K is given by

$$\begin{cases} \frac{\partial(\alpha_{1}\rho_{1})}{\partial t} + \nabla \cdot (\alpha_{1}\rho_{1}\mathbf{u}) = 0, \\ \vdots \\ \frac{\partial(\alpha_{K}\rho_{K})}{\partial t} + \nabla \cdot (\alpha_{K}\rho_{K}\mathbf{u}) = 0, \\ \frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla \cdot (\rho\mathbf{u} \otimes \mathbf{u} + p\mathbf{I}) = 0, \\ \frac{\partial(\rho E)}{\partial t} + \nabla \cdot (\rho E\mathbf{u} + p\mathbf{u}) = 0, \\ \frac{\partial\alpha_{1}}{\partial t} + \mathbf{u} \cdot \nabla\alpha_{1} = \alpha_{1}(\lambda_{1} - 1)\nabla \cdot \mathbf{u}, \\ \vdots \\ \frac{\partial\alpha_{K-1}}{\partial t} + \mathbf{u} \cdot \nabla\alpha_{K-1} = \alpha_{K-1}(\lambda_{K-1} - 1)\nabla \cdot \mathbf{u}, \end{cases}$$
(1)

where  $\alpha_k$  denotes the volume fraction of the kth material with the saturation condition  $\sum_{k=1}^{K} \alpha_k = 1$ . The density  $\rho$ , pressure p, and total energy E can be expressed as follows:

$$\begin{cases}
\rho = \sum_{k=1}^{K} \alpha_k \rho_k, \\
p = \sum_{k=1}^{K} \alpha_k p_k, \\
\rho E = \sum_{k=1}^{K} \alpha_k \rho_k e_k + \frac{1}{2} \rho \mathbf{u} \cdot \mathbf{u}.
\end{cases} (2)$$

Here,  $\rho_k$ ,  $p_k$ , and  $e_k$  represent the phasic density, pressure, and internal energy of the kth material, respectively. In addition,  $\mathbf{u}$  is the velocity, shared by each material due to the instantaneous velocity-equilibrium. These equations are supplemented by the isobaric closure law, i.e.,  $p_k = p$ . Thus, we have

$$\rho e = \sum_{k=1}^{K} \alpha_k \rho_k e_k(\rho_k, p), \tag{3}$$

leveraging the equations of state (EoSs) of each material. The model is hyperbolic, and the characteristic SoS c (i.e., the mixed SoS) is given by

$$\xi \rho c^2 = \sum_{k=1}^K \xi_k \lambda_k \alpha_k \rho_k c_{s,k}^2. \tag{4}$$

In Eq. (4),  $\xi_k$  is defined as

$$\xi_k = \frac{\partial \rho_k e_k}{\partial p_k} \bigg|_{\rho_k} = \rho_k \frac{\partial e_k}{\partial p_k} \bigg|_{\rho_k}, \tag{5}$$

which turns out to be exactly the reciprocal of the Grüneisen coefficient  $\Gamma_k$  in the EoS of the kth material (i.e.,  $\xi_k = 1/\Gamma_k$ ). The isentropic SoS of the kth material is defined as

$$c_{s,k}^2 = \frac{\partial p_k}{\partial \rho_k} \bigg|_{s},\tag{6}$$

where  $s_k$  is the phasic entropy of this material. In addition, the mixed  $\xi$  in Eq. (4) is defined as

$$\xi = \sum_{k=1}^{K} \alpha_k \xi_k. \tag{7}$$

The DC  $\lambda_k$ , satisfying  $\sum_{k=1}^K \alpha_k \lambda_k = 1$ , determines the specific path along which the kth material evolves. It is problem-dependent and requires physical knowledge, and will be studied in this paper analytically.

Finally, the corresponding evolution equation of the total energy for each material can be obtained as<sup>6</sup>

$$\frac{\partial(\alpha_k \rho_k E_k)}{\partial t} + \nabla \cdot (\alpha_k \rho_k E_k \mathbf{u} + p \mathbf{u})$$

$$= (1 - Y_k) \mathbf{u} \cdot \nabla p + \alpha_k \xi_k \Big( \rho_k c_{s,k}^2 \lambda_k - \rho c^2 \Big) \nabla \cdot \mathbf{u}$$

$$+ (1 - \alpha_k \lambda_k) p \nabla \cdot \mathbf{u}, \tag{8}$$

where  $E_k = \frac{\mathbf{u} \cdot \mathbf{u}}{2} + e_k$  and  $Y_k = (\alpha_k \rho_k)/(\sum_{l=1}^K \alpha_l \rho_l)$  is the mass fraction (or concentration) of the *k*th material.

# B. Analysis of distribution coefficients under varying physical assumptions

In Refs. 1 and 6, it is found that the corresponding DC  $\lambda_k$  and mixed SoS c for certain specific situations can be analytically derived, and they are summarized here briefly.

### 1. Isentropic interaction

In the first case, the phasic entropy  $s_k$  remains constant in the process of material interactions. In this case, the following results are obtained:

$$\lambda_k = \frac{1}{\rho_k c_{s,k}^2 \sum_{l=1}^K \frac{\alpha_l}{\rho_l c_{s,l}^2}} \tag{9}$$

and

$$\frac{1}{\rho c^2} = \sum_{l=1}^{K} \frac{\alpha_l}{\rho_l c_{s,l}^2}.$$
 (10)

#### 2. Interaction under equal compressibility

In the second case, all materials exhibit the same compressibility. In this case, the following results are obtained:

$$\lambda_k = 1 \tag{11}$$

and

$$\xi \rho c^2 = \sum_{l=1}^K \xi_l \alpha_l \rho_l c_{s,l}^2. \tag{12}$$

### 3. Interaction under equal velocity variation

In the third case, the velocity variations of all materials are equal. In this case, the following results are obtained:

$$\lambda_k = \frac{1}{c_{s,k} \sum_{l=1}^K \frac{\alpha_l}{c_s}} \tag{13}$$

and

$$\xi \rho c^2 = \left(\frac{1}{\sum_{l=1}^K \frac{\alpha_l}{c_{s,l}}}\right) \left(\sum_{l=1}^K \xi_l \alpha_l \rho_l c_{s,l}\right). \tag{14}$$

# 4. General expression of distribution coefficient under two material condition

For the case of two materials, the corresponding distribution coefficient  $\lambda_k$  can be analytically written in the following form given the formula for  $\rho c^2$ :

$$\begin{cases} \alpha_{1}(\lambda_{1}-1) = \frac{\xi\rho c^{2} - \alpha_{1}\xi_{1}\rho_{1}c_{s,1}^{2} - \alpha_{2}\xi_{2}\rho_{2}c_{s,2}^{2}}{\xi_{1}\rho_{1}c_{s,1}^{2} - \xi_{2}\rho_{2}c_{s,2}^{2}}, \\ \lambda_{2} = \frac{1 - \alpha_{1}\lambda_{1}}{\alpha_{2}}. \end{cases}$$
(15)

In addition to the above-mentioned formula for the mixed SoS, two other well-known formulas for the mixed SoS exist.

The first one is the frozen SoS  $\rho c^2 = \alpha_1 \rho_1 c_{s,1}^2 + \alpha_2 \rho_2 c_{s,2}^2$ . By substituting this formula into Eq. (15), we can obtain

$$\lambda_1 = \frac{\sum_{l=1}^2 \alpha_1 \rho_l c_{s,l}^2 (\xi - \xi_l)}{\alpha_1 \left( \xi_1 \rho_1 c_{s,1}^2 - \xi_2 \rho_2 c_{s,2}^2 \right)} + 1.$$
 (16)

The above expression can be further simplified as

$$\lambda_1 = -\alpha_2 \frac{(\xi_1 - \xi_2)(\rho_1 c_{s,1}^2 - \rho_2 c_{s,2}^2)}{\xi_1 \rho_1 c_{s,1}^2 - \xi_2 \rho_2 c_{s,2}^2} + 1.$$
 (17)

The second one is the mixed SoS for the stratified flow,

$$\left(\frac{\alpha_1}{\rho_1} + \frac{\alpha_2}{\rho_2}\right) \frac{1}{c^2} = \frac{\alpha_1}{\rho_1 c_{s,1}^2} + \frac{\alpha_2}{\rho_2 c_{s,2}^2}.$$
 (18)

By substituting this formula into Eq. (15), we can obtain<sup>6</sup>

$$\lambda_{1} = \frac{\xi \rho \frac{c_{s,1}^{2} c_{s,2}^{2} (\alpha_{1} \rho_{2} + \alpha_{2} \rho_{1})}{\alpha_{1} \rho_{2} c_{s,2}^{2} + \alpha_{2} \rho_{1} c_{s,1}^{2}} - \alpha_{1} \xi_{1} \rho_{1} c_{s,1}^{2} - \alpha_{2} \xi_{2} \rho_{2} c_{s,2}^{2}}{\alpha_{1} (\xi_{1} \rho_{1} c_{s,1}^{2} - \xi_{2} \rho_{2} c_{s,2}^{2})} + 1. \quad (19)$$

# III. GENERIC FIVE-EQUATION MODEL WITH ADDITIONAL TEMPERATURE-EQUILIBRIUM

It is known that the GFE model with additional temperatureequilibrium is theoretically equivalent to the four-equation model with the PT equilibrium closure. In this section, we perform an analysis on the PT equilibrium closure. First, we list the theoretical results presented in Ref. 6. Then, we present a novel method to analyze this closure, and derive a compact representation of the GFE model with additional temperature-equilibrium.

# A. Interaction under pressureand temperatureequilibrium

The classical four-equation model is expressed as follows:

$$\begin{cases} \frac{\partial(\alpha_{1}\rho_{1})}{\partial t} + \nabla \cdot (\alpha_{1}\rho_{1}\mathbf{u}) = 0, \\ \vdots \\ \frac{\partial(\alpha_{K}\rho_{K})}{\partial t} + \nabla \cdot (\alpha_{K}\rho_{K}\mathbf{u}) = 0, \\ \frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla \cdot (\rho\mathbf{u} \otimes \mathbf{u} + p\mathbf{I}) = 0, \\ \frac{\partial(\rho E)}{\partial t} + \nabla \cdot (\rho E\mathbf{u} + p\mathbf{u}) = 0, \end{cases}$$
(20)

with the "volume-separated and PT equilibrium" closure (i.e., the materials occupy disjointed volumes at the same temperature  $T_k = T$  and pressure  $p_k = p$  inside the microstructure),

$$\begin{cases}
\sum_{l=1}^{K} \frac{Y_{l}}{\rho_{l}} = \frac{1}{\rho}, \\
p_{1}(\rho_{1}, T) = p, \\
\vdots \\
p_{K}(\rho_{K}, T) = p, \\
\sum_{l=1}^{K} Y_{l}e_{l}(\rho_{l}, T) = e.
\end{cases} (21)$$

The corresponding mixed SoS is given by

(17) 
$$\frac{1}{\rho c^2} = \sum_{l=1}^{K} \frac{\alpha_l}{\rho_l c_{s,l}^2} + \frac{1}{T \sum_{l=1}^{K} \alpha_l \rho_l C_{p,l}} \sum_{l=1}^{K} \sum_{j>l}^{K} (\alpha_l \rho_l C_{P,l}) (\alpha_j \rho_j C_{P,j}) (\zeta_j - \zeta_l)^2,$$
(22)

where  $C_{P,k}$  is the specific heat at constant pressure and the parameter  $C_{P,k}$  is  $C_{P,k} = \frac{\partial T_k}{\partial t_k} | t_k^{-1}$ 

 $\zeta_k$  is  $\zeta_k = \frac{\partial T_k}{\partial p_k}|_{s_k}$ .

This model [Eq. (20)] can be obtained by removing the volume fraction equation that is nonconservative in the GFE model. That is to say the GFE model with additional temperature-equilibrium is theoretically equivalent to the classical four-equation model. However, the corresponding DC that arises under this additional temperature-equilibrium is unclear.

By differentiating the closure [Eq. (21)] and incorporating the mixed SoS [Eq. (22)], the DC  $\lambda_k$  that arises under PT equilibrium is theoretically obtained,<sup>6</sup>

$$\lambda_{k} = \left(\frac{\mathfrak{A}_{k}}{\rho_{k}} + \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{\sum_{l=1}^{K} \alpha_{k} \rho_{k} \mathfrak{B}_{l}}{\sum_{l=1}^{K} \alpha_{l} \rho_{l} \mathfrak{C}_{l}}\right) \rho c^{2} - \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{p}{\sum_{l=1}^{K} \alpha_{l} \rho_{l} \mathfrak{D}_{l}}, \quad (23)$$

where

$$\begin{cases}
\mathfrak{A}_{k} = \frac{\partial \rho_{k}}{\partial p_{k}} \Big|_{T_{k}} = \frac{1}{c_{s,k}^{2}} + \frac{\rho_{k}^{2} \zeta_{k}^{2} C_{P,k}}{T_{k}}, \\
\mathfrak{B}_{k} = \frac{\frac{\partial e_{k}}{\partial \rho_{k}} \Big|_{T_{k}}}{\frac{\partial \rho_{k}}{\partial \rho_{k}} \Big|_{T_{k}}} = \frac{\partial e_{k}}{\partial p_{k}} \Big|_{T_{k}} = \frac{p}{(\rho_{k} c_{s,k})^{2}} - \zeta_{k} C_{P,k} \left(1 - \zeta_{k} \frac{p_{k}}{T_{k}}\right), \\
\mathfrak{C}_{k} = -\mathfrak{B}_{k} \frac{\partial p_{k}}{\partial r_{k}} \Big|_{\rho_{k}} + \frac{\partial e_{k}}{\partial T_{k}} \Big|_{\rho_{k}} = \frac{\partial e_{k}}{\partial T_{k}} \Big|_{\rho} = C_{P,k} \left(1 - \zeta_{k} \frac{p_{k}}{T_{k}}\right), \\
\mathfrak{D}_{k} = \frac{\frac{\partial p_{k}}{\partial r_{k}}}{\frac{\partial \rho_{k}}{\partial \rho_{k}} \Big|_{T_{k}}} = -\frac{\partial \rho_{k}}{\partial T_{k}} \Big|_{\rho_{k}} = \frac{\rho_{k}^{2} \zeta_{k} C_{P,k}}{T_{k}}.
\end{cases} (24)$$

We can see the derived DC is too complicated to be efficiently implemented.

### **B.** Compact representation

Defining the specific volume  $\nu_k = \frac{1}{\rho_k}$  and  $\nu = \frac{1}{\rho}$ , we can reduce Eq. (21) into a system of nonlinear equations with two unknowns (p and T) as follows:

$$\begin{cases} \sum_{l=1}^{K} Y_{l} \nu_{l}(p, T) = \nu, \\ \sum_{l=1}^{K} Y_{l} e_{l}(p, T) = e. \end{cases}$$
 (25)

Differentiating Eq. (25), we can obtain

$$\sum_{l=1}^{K} \left( \nu_l(p, T) \frac{dY_l}{dt} + Y_l \frac{d\nu_l(p, T)}{dt} \right) = \frac{d\nu}{dt}, \tag{26}$$

$$\sum_{l=1}^{K} \left( e_l(p, T) \frac{dY_l}{dt} + Y_l \frac{de_l(p, T)}{dt} \right) = \frac{de}{dt}, \tag{27}$$

where  $\frac{d}{dt} = \frac{\partial}{\partial t} + \mathbf{u} \cdot \nabla$  is the Lagrangian full derivative. From the GFE model, we know<sup>1</sup>

$$\begin{cases} \frac{dY_k}{dt} = 0, \\ \frac{d\rho}{dt} = -\rho \nabla \cdot \mathbf{u}, \\ \frac{de}{dt} = -\frac{p}{\rho} \nabla \cdot \mathbf{u}. \end{cases}$$
 (28)

Substituting Eq. (28) into Eqs. (26) and (27), we can obtain

$$\sum_{l=1}^{K} Y_l \frac{d\nu_l(p, T)}{dt} = \frac{1}{\rho} \nabla \cdot \mathbf{u}, \tag{29}$$

$$\sum_{l=1}^{K} Y_{l} \frac{de_{l}(p,T)}{dt} = -\frac{p}{\rho} \nabla \cdot \mathbf{u}. \tag{30}$$

Expanding the total differentials in Eqs. (29) and (30) gives

$$\left(\sum_{l=1}^{K} Y_{l} \frac{\partial \nu_{l}}{\partial p} \Big|_{T}\right) \frac{dp}{dt} + \left(\sum_{l=1}^{K} Y_{l} \frac{\partial \nu_{l}}{\partial T} \Big|_{p}\right) \frac{dT}{dt} = \frac{1}{\rho} \nabla \cdot \mathbf{u}, \quad (31)$$

$$\left(\sum_{l=1}^{K} Y_{l} \frac{\partial e_{l}}{\partial p} \Big|_{T}\right) \frac{dp}{dt} + \left(\sum_{l=1}^{K} Y_{l} \frac{\partial e_{l}}{\partial T} \Big|_{p}\right) \frac{dT}{dt} = -\frac{p}{\rho} \nabla \cdot \mathbf{u}. \tag{32}$$

Solving this system of nonlinear equations, we can obtain

$$\frac{dp}{dt} = \frac{1}{\rho} \frac{F_{22} + pF_{12}}{F_{11}F_{22} - F_{12}F_{21}} \nabla \cdot \mathbf{u},\tag{33}$$

$$\frac{dT}{dt} = -\frac{1}{\rho} \frac{pF_{11} + F_{21}}{F_{11}F_{22} - F_{12}F_{21}} \nabla \cdot \mathbf{u},\tag{34}$$

where

$$\begin{cases}
F_{11} = \sum_{l=1}^{K} Y_{l} \frac{\partial \nu_{l}}{\partial p} \Big|_{T}, \\
F_{12} = \sum_{l=1}^{K} Y_{l} \frac{\partial \nu_{l}}{\partial T} \Big|_{p}, \\
F_{21} = \sum_{l=1}^{K} Y_{l} \frac{\partial e_{l}}{\partial p} \Big|_{T}, \\
F_{22} = \sum_{l=1}^{K} Y_{l} \frac{\partial e_{l}}{\partial T} \Big|_{p}.
\end{cases} (35)$$

On the other hand, in the GFE model, the pressure evolution satisfies the following equation:

$$\frac{dp}{dt} = -\rho c^2 \nabla \cdot \mathbf{u}. \tag{36}$$

Comparing Eq. (36) and Eq. (33), we can obtain

$$\rho c^2 = -\frac{1}{\rho} \frac{F_{22} + pF_{12}}{F_{11}F_{22} - F_{12}F_{21}}.$$
 (37)

In the GFE model, we further define

$$\vartheta_{k} = \frac{1}{\nu_{k}} \frac{d\nu_{k}}{dt} = \frac{1}{\nu_{k}} \left( \frac{\partial \nu_{k}}{\partial p} \Big|_{T} \frac{dp}{dt} + \frac{\partial \nu_{k}}{\partial T} \Big|_{p} \frac{dT}{dt} \right)$$

$$= \frac{1}{\rho \nu_{k}} \left( \frac{\partial \nu_{k}}{\partial p} \Big|_{T} \frac{F_{22} + pF_{12}}{F_{11}F_{22} - F_{12}F_{21}} - \frac{\partial \nu_{k}}{\partial T} \Big|_{p} \frac{pF_{11} + F_{21}}{F_{11}F_{22} - F_{12}F_{21}} \right) \nabla \cdot \mathbf{u}.$$
(38)

It is further assumed that  $\vartheta_k$  satisfies the following relationship:

$$\vartheta_k = \lambda_k \nabla \cdot \mathbf{u}. \tag{39}$$

Therefore, we have

$$\lambda_k = \frac{1}{\rho \nu_k} \left( \frac{\partial \nu_k}{\partial p} \Big|_T \frac{F_{22} + pF_{12}}{F_{11}F_{22} - F_{12}F_{21}} - \frac{\partial \nu_k}{\partial T} \Big|_p \frac{pF_{11} + F_{21}}{F_{11}F_{22} - F_{12}F_{21}} \right). \tag{40}$$

Next, we introduce three thermodynamic derivatives and systematically simplify the forms of Eqs. (37) and (40).

The coefficient of thermal expansion  $\beta_k$  of the kth material is given by

$$\beta_k = \frac{1}{\nu_k} \frac{\partial \nu_k}{\partial T_k} \bigg|_{p_k}. \tag{41}$$

The isothermal compressibility  $\kappa_{T,k}$  of the kth material is given by

$$\kappa_{T,k} = -\frac{1}{\nu_k} \frac{\partial \nu_k}{\partial p_k} \bigg|_{T_L}. \tag{42}$$

The specific heat at constant pressure  $C_{P,k}$  of the kth material is given by

$$C_{P,k} = \frac{\partial h_k}{\partial T_k} \bigg|_{p_k},\tag{43}$$

where  $h_k$  is the specific enthalpy of material k. Using these definitions, we can obtain

$$\frac{\partial \nu_k}{\partial p}\Big|_T = -\nu_k \kappa_{T,k},$$
 (44)

$$\left. \frac{\partial \nu_k}{\partial T} \right|_p = \nu_k \beta_k,\tag{45}$$

$$\frac{\partial e_k}{\partial p}\Big|_T = \nu_k (p\kappa_{T,k} - T\beta_k),$$
 (46)

$$\left. \frac{\partial e_k}{\partial T} \right|_p = C_{P,k} - p\nu_k \beta_k. \tag{47}$$

Using these relationships, we can obtain

$$\begin{cases} F_{11} = -\sum_{l=1}^{K} Y_{l}\nu_{l}\kappa_{T,l}, \\ F_{12} = \sum_{l=1}^{K} Y_{l}\nu_{l}\beta_{l}, \\ F_{21} = p\sum_{l=1}^{K} Y_{l}\nu_{l}\kappa_{T,l} - T\sum_{l=1}^{K} Y_{l}\nu_{l}\beta_{l} = -pF_{11} - TF_{12}, \\ F_{22} = \sum_{l=1}^{K} Y_{l}C_{P,l} - p\sum_{l=1}^{K} Y_{l}\nu_{l}\beta_{l} = \sum_{l=1}^{K} Y_{l}C_{P,l} - pF_{12}. \end{cases}$$

$$(48)$$

In addition, by defining

$$\begin{cases} \kappa_T = \sum_{l=1}^K \alpha_l \kappa_{T,l}, \\ \beta = \sum_{l=1}^K \alpha_l \beta_l, \\ C_P = \sum_{l=1}^K Y_l C_{P,l}, \end{cases}$$
(49)

we can obtain

$$\begin{cases}
\nu \kappa_T = \sum_{l=1}^K Y_l \nu_l \kappa_{T,l}, \\
\nu \beta = \sum_{l=1}^K Y_l \nu_l \beta_l.
\end{cases} (50)$$

By substituting Eqs. (49) and (50) into Eq. (48), we can finally obtain

$$\begin{cases} F_{11} = -\nu \kappa_T, \\ F_{12} = \nu \beta, \\ F_{21} = p\nu \kappa_T - T\nu \beta, \\ F_{22} = C_P - p\nu \beta. \end{cases}$$

$$(51)$$

Using Eq. (51), we obtain

$$\begin{cases} F_{11}F_{22} - F_{12}F_{21} = -\nu\kappa_T C_P + T(\nu\beta)^2, \\ F_{22} + pF_{12} = C_P, \\ pF_{11} + F_{21} = T\nu\beta. \end{cases}$$
 (52)

Finally, substituting Eqs. (51) and (52) into Eq. (40), we can ultimately obtain

$$\lambda_k = \frac{\kappa_{T,k} C_P - T \nu \beta \beta_k}{\kappa_T C_P - T \nu \beta^2}.$$
 (53)

In addition, we theoretically prove that Eq. (4) which is the general formula of the mixed SoS in the GFE model is still valid. Substituting Eqs. (51) and (52) into Eq. (37), we can obtain

$$\frac{1}{\rho c^2} = \kappa_T - \frac{T\beta^2}{\rho C_P},\tag{54}$$

which has also been derived by some other researchers through alternative methods. <sup>15,17</sup> On the other hand, by using Maxwell's relations and thermodynamic identities, we can obtain the following relationships:

$$\rho_k c_{s,k}^2 = \frac{C_{P,k}}{\kappa_{T,k} C_{P,k} - T \nu_k \beta_k^2},\tag{55}$$

$$\xi_k = \frac{\kappa_{T,k} C_{P,k} - T \nu_k \beta_k^2}{\nu_k \beta_k} = \frac{\kappa_{T,k} C_{P,k}}{\nu_k \beta_k} - T \beta_k.$$
 (56)

Substituting the above relationships into the right-hand side of Eq. (4), we obtain

$$\sum_{l=1}^{K} \xi_{l} \lambda_{l} \alpha_{l} \rho_{l} c_{s,l}^{2} = \frac{1}{\kappa_{T} C_{P} - T \nu \beta^{2}} \sum_{l=1}^{K} \frac{\alpha_{l} C_{P,l} (\kappa_{T,l} C_{P} - T \nu \beta \beta_{l})}{\nu_{l} \beta_{l}}$$

$$= \frac{C_{P}}{\kappa_{T} C_{P} - T \nu \beta^{2}} \left( \sum_{l=1}^{K} \frac{\alpha_{l} \kappa_{T,l} C_{P,l}}{\nu_{l} \beta_{l}} - T \beta \right). \tag{57}$$

Noting that

$$\xi = \sum_{l=1}^{K} \alpha_l \xi_l = \sum_{l=1}^{K} \frac{\alpha_l \kappa_{T,l} C_{P,l}}{\nu_l \beta_l} - T\beta, \tag{58}$$

and from Eq. (54), we can easily see that  $\sum_{l=1}^{K} \xi_l \lambda_l \alpha_l \rho_l c_{s,l}^2 = \xi \rho c^2$ , and this is exactly the formula of the mixed SoS [Eq. (4)] in the GFE model. Therefore, Eqs. (4) and (54) are equivalent, they are different expressions of the same characteristic SoS in the generic five-equation model with additional temperature-equilibrium.

![](_page_6_Figure_4.jpeg)

**FIG. 1.** Comparisons of numerical results of (a) density, (b) velocity, (c) pressure, and (d) temperature for 1D shock tube problem between the four-equation model and the generic five-equation model with additional temperature at t=0.2. The insets show enlarged views of the regions where numerical oscillations occur.

### IV. NUMERICAL TESTS

Numerical results for 1D and 2D problems are presented to validate the above conclusions. The classical/extended Godunov-type finite-volume scheme is used, with third-order SSP Runge–Kutta time integration, TVD reconstruction using the van Leer limiter, and the HLLC Riemann solver. Reconstruction variables are  $\rho$ ,  $\mathbf{u}$ , p,  $Y_k$  for the four-equation model (denoted by "four-equation model"), and  $\rho_k$ ,  $\mathbf{u}$ , p,  $\alpha_k$  for the generic five-equation model with additional temperature-equilibrium (denoted by "GFE with PT"). The CFL number is set to 0.4.

### A. One-dimensional shock tube problem

First, we consider a 1D shock tube problem tested in Ref. 18. In Fig. 1, we present results for the Sod problem. In particular, the equilibrated temperature T for the GFE with PT is derived by solving  $e = \sum_{l=1}^{K} Y_k e_l(\rho_l, T)$ . As reported in many studies, <sup>2,11,12,18</sup> the four-equation model does indeed exhibit numerical oscillations across the interface. However, although theoretically equivalent, the generic five-equation model with additional temperature-equilibrium can directly

provide the phase density  $\rho_k$  and volume fraction  $\alpha_k$ , and using these quantities for reconstruction can allow to effectively eliminate the numerical oscillations across the interface caused by the nonlinear reconstruction process. <sup>18</sup>

# B. Two-dimensional shock-bubble interaction

Here, we will show results for a two-dimensional simulation of a shock tube experiment.<sup>19</sup> We base our initial conditions on those calculated by Refs. 16 and 18 with resolution of 320 cells across the shock tube width.

The comparisons of numerical results between the four-equation model and the generic five-equation model with additional temperature equilibrium are presented in Figs. 2 and 3. Significant oscillations are observed in the temperature fields when the four-equation model is employed, as shown in Figs. 2(a), 2(c), and 2(e) (with further cross-sectional analysis provided in Fig. 3). In contrast, the generic five-equation model with additional temperature equilibrium exhibits much superior solution quality, with no obvious numerical oscillations. These results further substantiate the conclusions drawn earlier.

![](_page_7_Figure_4.jpeg)

FIG. 2. The temperature field for the air–He shock–bubble interaction problem at time instants (a) and (b) t ¼ 72ls, (c) and (d) t ¼ 245ls, and (e) and (f) t ¼ 674ls. The left column is obtained with the four-equation model, while the right column is calculated with the generic five-equation model with additional temperature-equilibrium. The colorbar is consistent for both models.

![](_page_7_Figure_6.jpeg)

FIG. 3. Distribution of temperature field along y ¼ 0 cross section for the air–He shock–bubble interaction problem at (a) t ¼ 72ls and (b) t ¼ 674ls.

# V. CONCLUSION

By introducing a novel method grounded in thermodynamic analysis, we have developed a compact and unified formulation for the distribution coefficient in the generic five-equation model with additional temperature equilibrium. The above result establishes a theoretical linkage between the generic five-equation

model incorporating thermal equilibrium and the conventional four-equation model with pressure–temperature equilibrium closure.

Looking ahead, this work provides a solid foundation for further extensions, such as incorporation of phase change, as well as the development of advanced numerical schemes.

# ACKNOWLEDGMENTS

This work was supported by the Science Challenge Project (Grant No. TZ2025007) and the National Natural Science Foundation of China (NSFC) (Grant Nos. 12372285 and 12502332).

# AUTHOR DECLARATIONS Conflict of Interest

The authors have no conflicts to disclose.

#### Author Contributions

Zhiwei He: Conceptualization (equal); Formal analysis (equal); Funding acquisition (equal); Investigation (equal); Methodology (equal); Project administration (equal); Resources (equal); Supervision (equal); Writing – original draft (equal); Writing – review & editing (equal). Shuo Zhao: Software (equal); Validation (equal); Visualization (equal); Writing – original draft (equal).

### DATA AVAILABILITY

The data that support the findings of this study are available from the corresponding author upon reasonable request.

# REFERENCES

- 1 Z. He, H. Liu, and L. Li, "Generic five-equation model for compressible multimaterial flows and its corresponding high-fidelity numerical algorithms," [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2023.112154) <sup>487</sup>, 112154 (2023). <sup>2</sup>
- G. Allaire, S. Clerc, and S. Kokh, "A five-equation model for the simulation of interfaces between compressible fluids," [J. Comput. Phys.](https://doi.org/10.1006/jcph.2002.7143) <sup>181</sup>, 577–616 (2002). <sup>3</sup> J. Massoni, R. Saurel, B. Nkonga, and R. Abgrall, "Some models and Eulerian methods for interface problems between compressible fluids with heat transfer," [Int. J. Heat Mass Transfer](https://doi.org/10.1016/S0017-9310(01)00238-1) 45, 1287–1307 (2002).

- 4 A. Kapila, R. Menikoff, J. Bdzil, S. Son, and D. Stewart, "Two-phase modeling of deflagration-to-detonation transition in granular materials: Reduced equations," [Phys. Fluids](https://doi.org/10.1063/1.1398042) <sup>13</sup>, 3002–3024 (2001). <sup>5</sup>
- A. Murrone and H. Guillard, "A five equation reduced model for compressible two phase flow problems," [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2004.07.019) <sup>202</sup>, 664–698 (2005). <sup>6</sup>
- Z. He and S. Tan, "On immiscibility preservation conditions of material interfaces in the generic five-equation model," [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2024.113192) <sup>513</sup>, 113192 (2024). <sup>7</sup>
- A. Chiapolino, R. Saurel, and S. Bodard, "Existence condition for detonations in condensed explosives with pressure-temperature equilibrium models," [Phys.](https://doi.org/10.1063/5.0238486) [Fluids](https://doi.org/10.1063/5.0238486) <sup>36</sup>, 116124 (2024). <sup>8</sup>
- F. Petitpas, R. Saurel, E. Franquet, and A. Chinnayya, "Modelling detonation waves in condensed energetic materials: Multiphase CJ conditions and multidimensional computations," [Shock Waves](https://doi.org/10.1007/s00193-009-0217-7) <sup>19</sup>, 377–401 (2009). <sup>9</sup>
- S. Le Martelot, R. Saurel, and B. Nkonga, "Towards the direct numerical simu-
- lation of nucleate boiling flows," [Int. J. Multiphase Flow](https://doi.org/10.1016/j.ijmultiphaseflow.2014.06.010) <sup>66</sup>, 62–78 (2014). <sup>10</sup>R. Saurel, P. Boivin, and O. Le Metayer, "A general formulation for cavitating,
- boiling and evaporating flows," [Comput. Fluids](https://doi.org/10.1016/j.compfluid.2016.01.004) <sup>128</sup>, 53–64 (2016). <sup>11</sup>R. Abgrall, "How to prevent pressure oscillations in multicomponent flow calcu-
- lations: A quasi conservative approach," [J. Comput. Phys.](https://doi.org/10.1006/jcph.1996.0085) <sup>125</sup>, 150–160 (1996). <sup>12</sup>E. Johnsen and F. Ham, "Preventing numerical errors generated by interfacecapturing schemes in compressible multi-material flows," [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2012.04.048) 231,
- <sup>5705</sup>–5717 (2012). <sup>13</sup>T. Flåtten and H. Lund, "Relaxation two-phase flow models and the subcharac-
- teristic condition," [Math. Models Methods Appl. Sci.](https://doi.org/10.1142/S0218202511005775) <sup>21</sup>, 2379–2407 (2011). <sup>14</sup>T. Flåtten, A. Morin, and S. Munkejord, "Wave propagation in multicomponent
- flow models," [SIAM J. Appl. Math.](https://doi.org/10.1137/090777700) <sup>70</sup>, 2861–2882 (2010). <sup>15</sup>R. Menikoff, "Empirical equations of state for solids," in ShockWave Science and Technology Reference Library, edited by Y. Horie (Springer, Berlin, Heidelberg, 2007).
- <sup>16</sup>A. Cook, "Enthalpy diffusion in multicomponent flows," [Phys. Fluids](https://doi.org/10.1063/1.3139305) 21, 055109 (2009).
- <sup>17</sup>J. Grove, "Pressure-velocity equilibrium hydrodynamic models," [Acta Math.](https://doi.org/10.1016/S0252-9602(10)60063-X)
- [Sci.](https://doi.org/10.1016/S0252-9602(10)60063-X) <sup>30</sup>, 563–594 (2010). <sup>18</sup>R. J. R. Williams, "Fully-conservative contact-capturing schemes for multi-
- material advection," [J. Comput. Phys.](https://doi.org/10.1016/j.jcp.2019.07.008) <sup>398</sup>, 108809 (2019). <sup>19</sup>J. F. Haas and B. Sturtevnt, "Interaction of weak shock waves with cylindrical and spherical gas inhomogeneities," [J. Fluid Mech.](https://doi.org/10.1017/S0022112087002003) 181, 41–76 (1987).