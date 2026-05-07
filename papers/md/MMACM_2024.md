![](_page_0_Picture_1.jpeg)

Contents lists available at [ScienceDirect](http://www.ScienceDirect.com/)

journal homepage: [www.elsevier.com/locate/jcp](http://www.elsevier.com/locate/jcp)

![](_page_0_Picture_5.jpeg)

# On immiscibility preservation conditions of material interfaces in the generic five-equation model

![](_page_0_Picture_7.jpeg)

- <sup>a</sup> *Institute of Applied Physics and Computational Mathematics, Beijing 100094, China*
- <sup>b</sup> *National Key Laboratory of Computational Physics, Beijing 100088, China*

### A R T I C L E I N F O A B S T R A C T

#### *Keywords:* Interface-sharpening technique Diffuse interface method Consistent and conservative schemes Compressible multimaterial flows

Generic five-equation model

Interfaces separating pure materials and mixtures tend to be severely smeared with interfacecapturing methods for compressible multi-material flows, necessitating the requirement of various interface-sharpening techniques. However, these techniques have various problems related to consistency, conservation, and thermodynamic compatibility. In this work, we derive a general theoretical formulation of interface-sharpening techniques for the generic five-equation model. This theoretical formulation is not only conservative in mass, momentum, and total energy but also asymptotically compatible with the thermodynamic mixture laws of the mixture model upon which it is constructed, and is independent of various specific numerical algorithms. We further propose a general numerical method to solve this theoretical formulation. The proposed method is consistent and conservative, and it prevents spurious errors at the interfaces. Examples of oneand two-dimensional multimaterial compressible flow problems, including shocks and interfaces, are considered to verify the analysis and demonstrate the efficiency of the method.

## **1. Introduction**

Numerous natural and industrial processes involve compressible flows with distinct material interfaces. Typical applications include underwater bubble dynamics, cavitation flows, inertial confinement fusion, Rayleigh-Taylor instabilities, and Richtmyer-Meshkov instabilities. Thus, investigating these flow mechanisms through numerical modeling and simulation of such flows is critical [\[1,2\]](#page-28-0).

One of the most important aspects of numerical modeling multimaterial flows is the method employed to describe the movement of the material interface. One method is to consider interfaces as numerically diffused zones of artificial mixtures using a color function that acquires different values for each fluid and assists in identifying the interface. This method is the so-called diffuse interface method (DIM), which can be further categorized into two subclasses: multicomponentand multiphase-based DIM [\[2\]](#page-28-0). A more detailed description of these subclasses can be found in our previous study [[2](#page-28-0)]. Additional details can be found in [\[3,4](#page-28-0)] and the references therein.

A common problem with DIM is that numerical implementations utilizing shock-/interface-capturing methods tend to produce solutions that exhibit excessive numerical diffusion. Minimizing numerical smearing within DIM has recently become an active topic, and interface-sharpening techniques/methods for various five-equation-type models [[5–12,1\]](#page-28-0) have been proposed.

<https://doi.org/10.1016/j.jcp.2024.113192>

Received 30 October 2023; Received in revised form 23 May 2024; Accepted 6 June 2024

Available online 12 June 2024

0021-9991/© 2024 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

<sup>\*</sup> Corresponding author at: Institute of Applied Physics and Computational Mathematics, Beijing 100094, China. *E-mail addresses:* [he\\_zhiwei@iapcm.ac.cn](mailto:he_zhiwei@iapcm.ac.cn) (Z. He), [tan\\_shuang@iapcm.ac.cn](mailto:tan_shuang@iapcm.ac.cn) (S. Tan).

These techniques/methods can be roughly divided into at least five categories [[13\]](#page-28-0). (1) **Anti-diffusion method**. The basic idea of this method is to directly solve the anti-diffusion equation to sharpen the interface [\[5\]](#page-28-0). A special discretization scheme [[14–16\]](#page-28-0) is employed to ensure numerical stability and volume fraction boundedness when solving the anti-diffusion equation. This method is essentially a flux modification or flux-corrected transport (FCT)-based technique [\[15](#page-28-0)]. It is intricately and inevitably tied to the underlying numerical scheme; thus, it is difficult to generalize to different discretizations, such as an increase in the order of accuracy [\[17](#page-28-0)]. Moreover, it is unknown whether these fluxes satisfy the compatibility between the equations. Numerical oscillations have been shown to exist at the interfaces [\[5](#page-28-0)], and there is a risk that anti-diffusive fluxes can oversharpen the interface in flow regions already drawn thin by the resolved strain field [[17\]](#page-28-0). (2) **Limited downwind Lagrange-remap method**. The basic idea of this method is to employ the downwind scheme as much as possible on the basis of satisfying the total variation diminishing (TVD) stability theory [\[18](#page-28-0)]. This method has been extended to a Lagrange-remap scheme [\[11,9\]](#page-28-0), and the resulting limited downwind Lagrange-remap scheme [\[11,9](#page-28-0)] is an operator split scheme comprising two steps: (a) the Lagrange step, in which the equation of the Allaire-Massoni model [\[19,20\]](#page-28-0) is advanced to a new time on a grid that moves with the fluid, and (b) the remap step, in which the solution is remapped onto the original mesh via advection over a pseudo-time step. While the Lagrange step is standard, the remap step is built with the limited downwind scheme [\[18](#page-28-0)] to ensure two types of features [\[11,9\]](#page-28-0): (i) it provides some consistency and stability properties for the scheme and (ii) it minimizes the diffusion of the variables that are utilized to locate the interface. The limited downwind fluxes in the remap stage for the other equations are designed in a manner similar to the anti-diffusion method to preserve the consistency between the volume fraction equation and the other equations [[5](#page-28-0)]. The final limited downwind Lagrange-remap method generates impressive numerical results [\[11,9\]](#page-28-0). However, this method is confined to the Lagrange-remap method and has shortcomings in the appearance and fragmentation of fine structures (filaments and high-frequency instabilities) [\[21](#page-28-0)]. (3) **Interface renormalization method**. This method relies on the artificial movement of the interface, as the diffusive and sharpening (compressive) fluxes can balance each other, ensuring convergence to a particular profile of interfaces. The conservative level set (CLS) and phase-field (PF) methods are typical. To maintain a consistently sharp interface in incompressible flows, Olsson et al. [\[22,23](#page-28-0)] first developed the CLS method, which has garnered significant interest [\[6,7,24–28](#page-28-0)]. This method is a two-step advection/artificial-compression interface-sharpening algorithm that was reformulated and extended [\[6,7](#page-28-0)] to compressible two-material flows governed by the Allaire-Massoni model [\[19,20](#page-28-0)]. A standard second-order center scheme was proposed to solve this type of interface-sharpening technique [[6,17](#page-28-0)]. However, this method led to qualitatively incorrect results due to thermodynamic inconsistencies [[17\]](#page-28-0). Tiwari et al. [\[17](#page-28-0)] theoretically proposed the immiscibility preservation conditions for the five-equation model proposed by Kapila et al. [\[29](#page-28-0)]. Nevertheless, their results are non-conservative, even for mass equations. The PF methods are based on the Cahn-Hilliard and Allen-Cahn equations, which were originally developed to model phase separation and coarsening phenomena in solids and the motion of antiphase boundaries in crystalline solids, respectively [[27\]](#page-28-0). Recently, these methods have been adopted to model the interface between two fluids [\[30,31,26\]](#page-28-0). The Cahn-Hilliard PF model is conservative but involves a fourth-order spatial derivative in the equation, which requires careful construction of the numerical methods [[26\]](#page-28-0). By contrast, the Allen-Cahn PF model does not involve fourth-order derivatives in the equation; however, it is not conservative [\[26\]](#page-28-0). Starting from the Allen-Cahn equation, Sun and Beckerman [[30\]](#page-28-0) employed the hyperbolic tangent equilibrium profile to introduce a term that canceled the curvature-driven incompressible flow. Inspired by the conservative level set of Olsson and Kreiss [[22\]](#page-28-0), Chiu and Lin [[31\]](#page-28-0) reformulated this equation in a conservative form. Jain et al. [[32,27](#page-28-0)] extended the aforementioned result to a compressible two-material flow governed by the Allaire-Massoni model [\[19](#page-28-0),[20\]](#page-28-0). However, the utilization of <sup>0</sup> (i.e., the density of phase within the incompressible limit) is not apparent in compressible flows. Recently, Huang and Johnsen [\[33](#page-28-0)] proposed another interface-sharpening algorithm based on the PF methods. (4) **Modified reconstruction method**. Shyue and Xiao [\[8](#page-28-0)] proposed a hybrid method in which a low-order semi-discrete wave propagation method [\[34,35](#page-28-0)] was utilized in single-material regions, and a tangent of hyperbola for interface capturing (THINC) scheme [\[36](#page-28-0)] was utilized in the interface zone. Recently, Deng et al. [\[12](#page-28-0)] proposed utilizing a boundary variation diminishing (BVD) framework [[37\]](#page-28-0) to realize the above hybridization. Chiapolino et al. [\[10](#page-28-0)] proposed a specific limiter and inserted it into conventional MUSCL-type schemes [\[38](#page-28-0)] to significantly improve the resolution of the interfaces. For these types of methods, the modified reconstructed schemes are often anti-diffusive; it is not certain whether these methods are capable of maintaining consistency between equations [\[8,13](#page-28-0)]. Furthermore, it was also found [\[1\]](#page-28-0) that the diffusion of the interface cannot be completely controlled by modifying only the reconstruction schemes, and that the approximate Riemann solvers with different dissipative properties still have significant effects. (5) **Artificial compression method**. Harten [\[39](#page-28-0)] first proposed the artificial compression method (ACM) to enhance the resolution of contact discontinuities in the context of the Euler equations. The essence of ACM is to solve an original equation with an added term (called an artificial compression flux, which can be designed theoretically or numerically [\[39](#page-28-0),[40\]](#page-29-0)) so that the numerical characteristics slightly converge toward the contact discontinuity (instead of being in parallel or diverging from the contact discontinuity) to maintain its sharpness [\[39](#page-28-0)[,41](#page-29-0),[42\]](#page-29-0). Yang [\[43](#page-29-0)] proposed another simple ACM for higher-order finite-volume ENO schemes via slope modification. In our previous study [[1](#page-28-0)], we extended Yang's method to compressible multimaterial flows and found that the numerical diffusion cannot be completely controlled solely by pure slope modification, and the approximate Riemann solvers with different dissipative properties also have significant effects. Moreover, the employment of the immiscibility preservation conditions [\[17](#page-28-0)] to maintain consistency among equations led this method to be non-conservative.

Currently, all these methods encounter a common and still unsolved problem: there is no well-established means for sharpening other equations (mass, momentum, and total energy equations) when the equation describing the interface evolution (such as the volume fraction equation) adopts the above interface-sharpening methods. This problem is very important for compressible multimaterial flows because (1) For incompressible multimaterial flows with moving interfaces, the density and other physical properties, such as viscosity and thermal conductivity, are constant in each fluid [[12\]](#page-28-0). However, a substantial barrier exists when compressible multimaterial flows are considered. For compressible multimaterial flows, the volume fraction of each material is no longer conservative, and the phasic density is no longer constant. The density and energy in a compressible flow must be solved along with the volume fraction. Moreover, special formulations are required to maintain physical consistency, resulting in a balanced state among all variables for a well-defined interface cell [12]. (2) If the consistency between the equations is broken, the result worsens. Moreover, disunity phenomena exist in the physical models used in these studies. Some researchers [6,32] utilized the five-equation model proposed by Allaire et al. [19] and Massoni et al. [20], whereas others utilized the model proposed by Kapila et al. [29,44].

It is difficult but important to systematically investigate the immiscibility preservation conditions for compressible multimaterial interfacial flows governed by various five-equation models. There are two difficulties. The first is the disunity phenomena that exist in the models, and the second is how to obtain an interface-sharpening technology that is thermodynamically compatible and completely conservative in terms of mass, momentum, and total energy. In our previous study [2], a unified formulation, including a distribution coefficient to characterize the interactions between materials, was theoretically derived and called a generic five-equation model (GFE) [2]. This model is general in that it can recover two specific five-equation models [19,20,29,44] and generate new models. A more detailed description can be found in our previous study [2].

Therefore, it is natural to explore interface-sharpening techniques for the generic five-equation model. In this study, we derive a general theoretical formulation of interface-sharpening techniques for the generic five-equation model. The theoretical formulation is not only conservative in mass, momentum, and total energy, but it is also asymptotically compatible with the thermodynamic mixture laws of the mixture model upon which it is constructed and independent of various specific numerical algorithms. A general numerical strategy called the multimaterial artificial compression method (MMACM) is further proposed to numerically solve the theoretical formulation. Examples of oneand two-dimensional multimaterial compressible flow problems, including shocks and interfaces, are considered to verify the analysis and demonstrate the efficiency of the method.

The remainder of this paper is organized as follows. In Section 2, we review some of the basic aspects of the generic five-equation model and introduce the topic of the present study. In Section 3, a general theoretical formulation of the interface-sharpening techniques for the generic five-equation model is presented in detail. In Section 4, we propose a high-fidelity numerical algorithm to solve the theoretical formulation numerically. In Section 5, examples of multimaterial compressible flow problems, including shocks and interfaces, are presented to verify the analysis and demonstrate the efficiency of the method. Finally, the conclusions are presented in Section 6.

#### 2. Topic of present work

In this section, we summarize the physical model utilized in this paper, and discuss the topic of the present work.

### 2.1. Generic five-equation model

In the literature, there are two typical five-equation models: the Allaire-Massoni model [19,20] and the Kapila model [29,44]. Different researchers have employed different models, leading to disunity phenomena. Recently, we [2] derived a unified formulation for the five-equation model, establishing a GFE model. This GFE model can recover existing typical five-equation models [19,20,29,44] and also generate new models [2]. In this study, we consider the GFE model [2] for compressible multimaterial hydrodynamics with an arbitrary number of materials K, which can be given by

$$\frac{\partial \left(\alpha_{1}\rho_{1}\right)}{\partial t} + \nabla \cdot \left(\alpha_{1}\rho_{1}\mathbf{u}\right) = 0,$$
...
$$\frac{\partial \left(\alpha_{K}\rho_{K}\right)}{\partial t} + \nabla \cdot \left(\alpha_{K}\rho_{K}\mathbf{u}\right) = 0,$$

$$\frac{\partial \left(\rho\mathbf{u}\right)}{\partial t} + \nabla \cdot \left(\rho\mathbf{u} \otimes \mathbf{u}\right) + \nabla p = 0,$$

$$\frac{\partial \left(\rho E\right)}{\partial t} + \nabla \cdot \left(\rho E\mathbf{u}\right) + \nabla \cdot \left(p\mathbf{u}\right) = 0,$$

$$\frac{\partial \alpha_{1}}{\partial t} + \mathbf{u} \cdot \nabla \alpha_{1} = \alpha_{1}(\lambda_{1} - 1)\nabla \cdot \mathbf{u},$$
...
$$\frac{\partial \alpha_{K-1}}{\partial t} + \mathbf{u} \cdot \nabla \alpha_{K-1} = \alpha_{K-1}(\lambda_{K-1} - 1)\nabla \cdot \mathbf{u},$$

where  $\alpha_k$  denotes the volume fraction of the k-th material with the saturation condition  $\sum_{k=1}^{K} \alpha_k = 1$ . The density  $\rho$ , pressure p, and total energy E can be expressed as

$$\rho = \sum_{k} \alpha_{k} \rho_{k}, p = \sum_{k} \alpha_{k} p_{k}, \rho E = \sum_{k} \alpha_{k} \rho_{k} e_{k} + \frac{1}{2} \rho \mathbf{u} \cdot \mathbf{u},$$
(2)

where  $\rho_k$ ,  $p_k$ , and  $e_k$  represent the density, pressure, and specific internal energy of the k-th material, respectively. These equations are supplemented by the isobaric closure law  $\rho e = \sum_{k=1}^K \alpha_k \rho_k e_k(\rho_k, p)$ , leveraging the equations of state (EOSs) of each material.

**Table 1** Special cases of distribution coefficients  $\lambda_k$  ( $k=1,\cdots,K$ ). Symmetry means that all materials are treated in the same manner, while the absence of symmetry denotes that the importance of material is heterogeneous in interaction. For more details, please see ref. [2].

| Assumption                 | Expression of $\lambda_k$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Symmetry                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| equal compressibility      | $\lambda_k = 1$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Yes                                                           |
| isentropic                 | $\lambda_k = \frac{1}{\rho_k c_{s,k}^2 \sum_{k'=1}^K \frac{a_{k'}}{\rho_{k'} c_{s,k'}^2}}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Yes                                                           |
| equal velocity variation   | $\lambda_k = \frac{1}{c_{s,k} \sum_{k'=1}^K \frac{a_{k'}}{c_{s,k'}}}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Yes                                                           |
| frozen flow( $K = 2$ )     | $\lambda_1 = \frac{\sum_{k=1}^2 \alpha_k \rho_k c_{s,k}^2 (\xi - \xi_k)}{\alpha_1 \left( \rho_1 \xi_1 c_{s,1}^2 - \rho_2 \xi_2 c_{s,2}^2 \right)} + 1$                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | No with $\lambda_2 = \frac{1 - \alpha_1 \lambda_1}{\alpha_2}$ |
| stratified flow( $K = 2$ ) | $\begin{split} \lambda_k &= \frac{\rho_k c_{i,k}^* \sum_{k'=1}^{k'-1} \frac{\alpha_{k'}}{\rho_{k'} c_{i,k'}^2}}{c_{i,k}} \\ \lambda_1 &= \frac{\sum_{l=1}^{k} \frac{\alpha_{k'}}{\alpha_{k'}}}{a_1 \left( \rho_1 \xi_1 c_{i,l}^2 - \rho_2 \xi_2 c_{i,2}^2 \right)} + 1 \\ \lambda_1 &= \frac{\sum_{l=1}^{k} \alpha_k \rho_k c_{i,k}^2 \left( -\rho_2 \xi_2 c_{i,2}^2 \right)}{a_1 \left( \rho_1 \xi_1 c_{i,l}^2 - \rho_2 \xi_2 c_{i,2}^2 \right)} + 1 \\ \lambda_1 &= \frac{a_1 \left( \rho_1 \xi_1 c_{i,1}^2 - \rho_2 \xi_2 c_{i,2}^2 \right)}{a_1 \left( \rho_1 \xi_1 c_{i,1}^2 - \rho_2 \xi_2 c_{i,2}^2 \right)} + 1 \end{split}$ | No with $\lambda_2 = \frac{1 - \alpha_1 \lambda_1}{\alpha_2}$ |

In this study, if not specifically mentioned, we chose the stiffened-gas EOS introduced by Harlow and Amsden [45]. This EOS is a combination of the perfect gas law and barotropic Tait equation, supplemented with an appropriate energy law [46]. The corresponding thermal and calorific EOS are

$$p_{\nu}(\rho_{\nu}, e_{\nu}) = (\gamma_{\nu} - 1)\rho_{\nu}(e_{\nu} - q_{\nu}) - \gamma_{\nu}P_{m,\nu},$$
 (3)

$$T_k(\rho_k, e_k) = (e_k - q_k - P_{m,k}/\rho_k)/C_{n,k},$$
 (4)

where  $T_k$  denotes the temperature and the material parameters are the ratio of specific heat  $\gamma_k$ , specific heat at constant volume  $C_{v,k}$ , minimal pressure  $P_{\infty,k}$ , and heat of formation  $q_k$  [46]. The distribution coefficients  $\lambda_k$  ( $k=1,\cdots,K$ ), which determine the specific path along which materials evolve, are problem-dependent and require physical knowledge [2]. However, certain special cases can be mathematically derived, as listed in Table 1. Finally, the corresponding mixture sound speed  $c_{eff}$  of this model is [2]

$$\rho c_{eff}^2 = \sum_{k=1}^K \left( \frac{\lambda_k \alpha_k \rho_k \xi_k}{\xi} \right) c_{s,k}^2, \tag{5}$$

where

$$\xi_{k} \triangleq \frac{\partial \rho_{k} e_{k}}{\partial p_{k}}\Big|_{\rho_{k}},$$

$$\xi \triangleq \sum_{k=1}^{K} \alpha_{k} \xi_{k},$$

$$c_{s,k}^{2} \triangleq \frac{\partial p}{\partial \rho_{k}}\Big|_{s_{k}} = \frac{p}{\rho_{s}^{2}} \frac{\partial p}{\partial e_{k}}\Big|_{\rho_{k}} + \frac{\partial p}{\partial \rho_{k}}\Big|_{e_{k}}.$$
(6)

### 2.2. Interface sharpening techniques to preserve immiscibility conditions

This study focuses on constructing interface-sharpening techniques for the GFE model. Specifically, we investigate the following two problems in detail.

(I) What terms should be added to each equation in the GFE model such that the consistency among the equations (of mass, momentum, total energy, and volume fraction), conservation (of mass, momentum, and total energy), and more importantly, thermodynamic compatibility can be preserved? That is, in the following model,

$$\frac{\partial \left(\alpha_1 \rho_1\right)}{\partial t} + \nabla \cdot \left(\alpha_1 \rho_1 \mathbf{u}\right) = M_1,\tag{7}$$

$$\frac{\partial \left(\alpha_K \rho_K\right)}{\partial t} + \nabla \cdot \left(\alpha_K \rho_K \mathbf{u}\right) = M_K,\tag{9}$$

$$\frac{\partial \left(\rho \mathbf{u}\right)}{\partial t} + \nabla \cdot \left(\rho \mathbf{u} \otimes \mathbf{u}\right) + \nabla p = \mathbf{P},\tag{10}$$

$$\frac{\partial \left(\rho E\right)}{\partial t} + \nabla \cdot \left(\rho E \mathbf{u}\right) + \nabla \cdot \left(\rho \mathbf{u}\right) = \Theta,\tag{11}$$

$$\frac{\partial \alpha_1}{\partial t} + \mathbf{u} \cdot \nabla \alpha_1 = \alpha_1 (\lambda_1 - 1) \nabla \cdot \mathbf{u} + A_1, \tag{12}$$

$$\frac{\partial \alpha_{K-1}}{\partial t} + \mathbf{u} \cdot \nabla \alpha_{K-1} = \alpha_{K-1} (\lambda_{K-1} - 1) \nabla \cdot \mathbf{u} + A_{K-1},\tag{14}$$

what are the general constraints between  $M_k$ , **P**,  $\Theta$ , and  $A_k$ ?

In this study, we theoretically and concretely deduce the general constraints on  $M_k$ , P,  $\Theta$ , and  $A_k$ . The final result forms a generic five-equation model that includes the interface-sharpening effect, which not only conserves mass, momentum, and total energy exactly but is also compatible with thermodynamics.

(II) After obtaining the model (Eqs. (7)-(14)), the next problem is obtaining theoretically or numerically specific expressions of these added terms, especially the term added to the volume fraction equations.

In this work, we utilize the original ACM framework [39] and set  $A_k$  as the difference between the limited downwind flux (modified with the steepness-adjustable harmonic limiter [47,48]) and the traditional upwind numerical flux provided by various Riemann solvers. Coupled with the generic five-equation model that includes the interface-sharpening effect (Eqs. (7)-(14)), we propose a refined multimaterial artificial compression method (MMACM) for compressible multimaterial flows.

These two topics form the subject of this study, and the details are described in the following sections.

#### 3. Modeling

In this section, we theoretically and concretely deduce the general constraints of  $M_k$ , P,  $\Theta$ , and  $A_k$ . The final result is an extended GFE model that includes the interface-sharpening effect, which not only exactly conserves mass, momentum, and total energy but is also compatible with thermodynamics.

#### 3.1. Mixture entropy of generic five-equation model

The mixture entropy s is defined as  $s = \sum_{k=1}^{K} Y_k s_k$ , where  $Y_k = \alpha_k \rho_k / \rho$  is the mass fraction (or concentration) of the k-th material. In this section, we analyze the mixture entropy s in the GFE model.

From the equation of the total energy of the GFE model, we have [2]

$$\frac{d\rho e}{dt} + (\rho e + p)\nabla \cdot \mathbf{u} = 0,\tag{15}$$

where  $\frac{d}{dt}$  denotes the Lagrangian total derivative  $\frac{d}{dt} = \frac{\partial}{\partial t} + \mathbf{u} \cdot \nabla$ . Eq. (15) can be further expanded to

$$\sum_{k=1}^{K} \alpha_k \rho_k \frac{de_k}{dt} + \sum_{k=1}^{K} e_k \frac{d\alpha_k \rho_k}{dt} + (\rho e + p) \nabla \cdot \mathbf{u} = 0.$$
 (16)

Using the Gibb's relation for each material,  $de_k = T_k ds_k + \frac{p}{\rho_k^2} d\rho_k$ , we obtain

$$\frac{de_k}{dt} = T_k \frac{ds_k}{dt} - p \frac{d}{dt} \left( \frac{1}{\rho_k} \right). \tag{17}$$

By using the GFE model [2], we get

$$\frac{d}{dt}\left(\frac{1}{a_t}\right) = \frac{\lambda_k}{a_t} \nabla \cdot \mathbf{u},\tag{18}$$

and

$$\frac{d\alpha_k \rho_k}{dt} = -\alpha_k \rho_k \nabla \cdot \mathbf{u}. \tag{19}$$

By substituting Eqs. (17), (18), and (19) into Eq. (16), we obtain

$$\sum_{k=1}^{K} \alpha_k \rho_k T_k \frac{ds_k}{dt} - \left(\sum_{k=1}^{K} \alpha_k \lambda_k\right) p \nabla \cdot \mathbf{u} + p \nabla \cdot \mathbf{u} = 0.$$
(20)

From our previous study [2], we know that  $\sum_{k=1}^{K} \alpha_k \lambda_k = 1$ . Therefore, Eq. (20) can be simplified as

$$\sum_{k=1}^{K} \alpha_k \rho_k T_k \frac{ds_k}{dt} = 0, \text{ i.e. } \sum_{k=1}^{K} T_k \frac{d(Y_k s_k)}{dt} = 0.$$
 (21)

Eq. (21) reveals two special cases: the isentropic assumption ( $\frac{ds_k}{dt} = 0$ ) and the pressure-temperature equilibrium assumption (sharing a common temperature  $T_k = T$ , see Remark 1) satisfy the characteristic equation  $\frac{ds}{dt} = 0$ , which is consistent with existing research results [29,44,49]. For general situations, Eq. (21) can be further rewritten as

$$\frac{ds}{dt} = \sum_{k=1}^{K} \frac{T - T_k}{T} Y_k \frac{ds_k}{dt}.$$
 (22)

Eq. (22) is just the equation of the mixture entropy s in the GFE model and implies that heat transfer among materials will affect the final evolution mode of the mixture entropy s. Any specific path along which materials evolve depends on the problem and requires physical knowledge. Qualitatively,  $s_k$  should be a function of  $\rho_k$ , p (i.e.,  $s_k(\rho_k, p)$ ) in the GFE model. Therefore, we have

$$\frac{ds_k}{dt} = \frac{\partial s_k}{\partial p} \bigg|_{\rho_k} \frac{dp}{dt} + \frac{\partial s_k}{\partial \rho_k} \bigg|_{\rho_k} \frac{d\rho_k}{dt}.$$
 (23)

From our previous study [2], Eq. (23) can be further written as

$$\frac{ds_k}{dt} = -\frac{\partial s_k}{\partial p}\Big|_{\rho_k} \rho c_{eff}^2 \nabla \cdot \mathbf{u} - \frac{\partial s_k}{\partial \rho_k}\Big|_{p} \rho_k \lambda_k \nabla \cdot \mathbf{u}. \tag{24}$$

By substituting this equation into Eq. (21), we obtain

$$\left(\sum_{k=1}^{K} \alpha_k \rho_k T_k \left. \frac{\partial s_k}{\partial p} \right|_{\rho_k} \right) \rho c_{eff}^2 = -\sum_{k=1}^{K} \alpha_k \rho_k^2 T_k \lambda_k \left. \frac{\partial s_k}{\partial \rho_k} \right|_{p}, \tag{25}$$

where  $\frac{\partial s_k}{\partial p}\Big|_{\rho_k}$  and  $\frac{\partial s_k}{\partial \rho_k}\Big|_{\rho_k}$  rely on the specific path along which the materials evolve. This path can be analyzed more quantitatively

by considering the heat-transfer effect. Utilizing the well-known relation  $dp = c_{s,k}^2 d\rho_k + \Gamma_k \rho_k T_k ds_k$  where  $\Gamma_k = \frac{1}{\rho_k} \left. \frac{\partial p}{\partial e_k} \right|_{\rho_k}$  (i.e.  $\frac{1}{\xi_k}$ ) represents the Grüneisen coefficient [49], we can obtain

$$\frac{dp}{dt} = c_{s,k}^2 \frac{d\rho_k}{dt} + \Gamma_k \rho_k T_k \frac{ds_k}{dt}.$$
 (26)

If we start with the assumption that entropy change in Eq. (21) (or Eq. (22)) is only due to the heat-transfer effect,  $\alpha_k \rho_k T_k \frac{ds_k}{dt} = H_k(T-T_k)$ , in which the relaxation coefficients  $H_k$  are independent of the temperatures  $T_k$  and satisfy  $\sum_{k=1}^K H_k(T-T_k) = 0$ , we can further reformulate Eq. (26) with the help of Eq. (18), as

$$\frac{dp}{dt} = -\rho_k c_{s,k}^2 \lambda_k \nabla \cdot \mathbf{u} + \Gamma_k \frac{H_k (T - T_k)}{\alpha_k}.$$
(27)

By utilizing the pressure evolution equation of the GFE model [2] and Eq. (27), we can obtain

$$H_k(T - T_k) = \frac{\alpha_k}{\Gamma_k} \left( \rho_k c_{s,k}^2 \lambda_k - \rho c_{eff}^2 \right) \nabla \cdot \mathbf{u}. \tag{28}$$

Therefore, the internal energy evolution equation for each material (Eq. (17)) in the GFE model is as follows:

$$\alpha_k \rho_k \frac{de_k}{dt} = \frac{\alpha_k}{\Gamma_L} \left( \rho_k c_{s,k}^2 \lambda_k - \rho c_{eff}^2 \right) \nabla \cdot \mathbf{u} - \alpha_k \lambda_k p \nabla \cdot \mathbf{u}. \tag{29}$$

The corresponding evolution equation of the total energy for each material can be given by

$$\frac{\partial(\alpha_k \rho_k E_k)}{\partial t} + \nabla \cdot (\alpha_k \rho_k E_k \mathbf{u} + p\mathbf{u})$$

$$= (1 - Y_k)\mathbf{u} \cdot \nabla p + \frac{\alpha_k}{\Gamma_k} \left( \rho_k c_{s,k}^2 \lambda_k - \rho c_{eff}^2 \right) \nabla \cdot \mathbf{u} + (1 - \alpha_k \lambda_k) p \nabla \cdot \mathbf{u}, \tag{30}$$

where  $E_k = \frac{\mathbf{u} \cdot \mathbf{u}}{2} + e_k$ . When the specific path along which each material evolves is isentropic, the second term on the right side of Eq. (30) is zero, and Eq. (30) degenerates to the result obtained by some researchers [50,51]. However, this path is not always isentropic in the GFE model but is generally determined by Eq. (30). For problems with an interface separating pure fluids, numerical requirements (e.g., the monotonic sound speed) may be utilized to characterize interfaces when they are numerically treated as artificial mixtures [2]. In these cases, the heat-transfer terms in the GFE model may not be physical but purely artificial.

**Remark 1.** The generic five-equation model with pressure-temperature equilibrium is theoretically equivalent to the classical four-equation model with kinetic equilibrium (same velocity), mechanical equilibrium (same pressure), and thermal equilibrium (same temperature) [52–58]. The distribution coefficient  $\lambda_k$  that arises under pressure-temperature equilibrium is derived in Appendix A.

### 3.2. Constraints corresponding to immiscibility preservation conditions of generic five-equation model

In this section, we derive the immiscibility preservation conditions for the GFE model.

#### 3.2.1. Elementary constraint

First, the saturation condition  $\sum_{k=1}^{K} \alpha_k = 1$  imposes the elementary constraint given by

$$\sum_{k=1}^{K} A_k = 0. \tag{31}$$

Further, by summing Eqs. (7)-(9) for all materials, we can derive the modified version of the continuity equation, which can be given by

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = \sum_{k=1}^{K} M_k,\tag{32}$$

where  $\sum_{k=1}^K M_k$  is the net mass interface-sharpening term and  $\sum_{k=1}^K M_k \neq 0$ .

#### 3.2.2. Consistency between equations for mass and volume fractions

From the phasic mass equations (Eqs. (7)-(9)), we can obtain

$$\frac{d\alpha_k}{dt} = -\frac{\alpha_k}{\rho_k} \frac{d\rho_k}{dt} - \alpha_k \nabla \cdot \mathbf{u} + \frac{M_k}{\rho_k}.$$
(33)

By defining  $v_k = \frac{1}{a_k}$ , we obtain

$$\vartheta_k = \frac{1}{v_L} \frac{dv_k}{dt} = \frac{1}{v_L} \left( \frac{\partial v_k}{\partial t} + \mathbf{u} \cdot \nabla v_k \right). \tag{34}$$

Eq. (33) can be further expressed as

$$\frac{d\alpha_k}{dt} = \alpha_k(\theta_k - \nabla \cdot \mathbf{u}) + \frac{M_k}{\alpha_k}.$$
(35)

In our previous study [2], we assumed  $\theta_k = \lambda_k \nabla \cdot \mathbf{u}$ . Given that the interface-sharpening technique is not a real physical effect but purely a numerical operator, this assumption still holds true. By substituting this assumption into Eq. (35), we obtain

$$\frac{\partial \alpha_k}{\partial t} + \mathbf{u} \cdot \nabla \alpha_k = \alpha_k (\lambda_k - 1) \nabla \cdot \mathbf{u} + \frac{M_k}{\rho_k}.$$
(36)

By comparing Eq. (36) with Eqs. (12)-(14), we obtain

$$A_k = \frac{M_k}{\rho_k}. (37)$$

Considering the constraint (Eq. (31)), the constraint that

$$\sum_{k=1}^{K} \frac{M_k}{\rho_k} = 0 {38}$$

must be fulfilled.

#### 3.2.3. Entropy inequalities

In our previous study [2],  $\frac{dp}{dt}$  in the GFE model was expressed as

$$\frac{dp}{dt} = \frac{1}{\xi} \frac{d(\rho e)}{dt} - \sum_{k=1}^{K} \frac{\delta_k}{\xi} \frac{d(\alpha_k \rho_k)}{dt} + \sum_{k=1}^{K} \frac{\rho_k \delta_k - \rho_k e_k}{\xi} \frac{d\alpha_k}{dt}.$$
 (39)

In Eq. (39), we need to know  $\frac{d(\rho e)}{dt}$ ,  $\frac{d(\alpha_k \rho_k)}{dt}$ , and  $\frac{d\alpha_k}{dt}$ ;  $\frac{d(\rho e)}{dt}$  and  $\frac{d(\alpha_k \rho_k)}{dt}$  can be directly obtained from Eqs. (7)-(11), and  $\frac{d\alpha_k}{dt}$  is given in Eqs. (12)-(14). By substituting these results into Eq. (39), we get

$$\frac{dp}{dt} = -\frac{1}{\xi} \left(\rho e + p\right) \nabla \cdot \mathbf{u} + \sum_{k=1}^{K} \frac{\delta_k}{\xi} \alpha_k \rho_k \nabla \cdot \mathbf{u} + \sum_{k=1}^{K} \frac{\rho_k \delta_k - \rho_k e_k}{\xi} \alpha_k \left(\lambda_k - 1\right) \nabla \cdot \mathbf{u} + \frac{\Theta - \mathbf{u} \cdot \mathbf{P}}{\xi} - \sum_{k=1}^{K} \frac{\delta_k}{\xi} M_k + \sum_{k=1}^{K} \frac{\rho_k \delta_k - \rho_k e_k}{\xi} A_k. \tag{40}$$

The above formula can be written as

$$\frac{dp}{dt} = -\rho c_{eff}^2 \nabla \cdot \mathbf{u} + \frac{\Theta - \mathbf{u} \cdot \mathbf{P}}{\xi} - \sum_{k=1}^K \frac{\delta_k}{\xi} M_k + \sum_{k=1}^K \frac{\rho_k \delta_k - \rho_k e_k}{\xi} A_k. \tag{41}$$

By substituting this equation (Eq. (41)) into Eq. (23), we obtain

$$\frac{ds_k}{dt} = \frac{\partial s_k}{\partial p} \Big|_{\rho_k} \left( -\rho c_{eff}^2 \nabla \cdot \mathbf{u} + \frac{\Theta - \mathbf{u} \cdot \mathbf{P}}{\xi} - \sum_{k=1}^K \frac{\delta_k}{\xi} M_k + \sum_{k=1}^K \frac{\rho_k \delta_k - \rho_k e_k}{\xi} A_k \right) - \rho_k \lambda_k \frac{\partial s_k}{\partial \rho_k} \Big|_{p} \nabla \cdot \mathbf{u}.$$
(42)

By utilizing Eq. (42), we can obtain  $Y_k T_k \frac{ds_k}{dt}$  as

$$Y_{k}T_{k}\frac{ds_{k}}{dt} = -\left(Y_{k}T_{k}\frac{\partial s_{k}}{\partial p}\Big|_{\rho_{k}}\rho c_{eff}^{2} + Y_{k}T_{k}\rho_{k}\lambda_{k}\frac{\partial s_{k}}{\partial \rho_{k}}\Big|_{p}\right)\nabla \cdot \mathbf{u}$$

$$+Y_{k}T_{k}\frac{\partial s_{k}}{\partial p}\Big|_{\rho_{k}}\left(\frac{\Theta - \mathbf{u} \cdot \mathbf{P}}{\xi} + \sum_{k=1}^{K} \frac{\left(\rho_{k}\delta_{k} - \rho_{k}e_{k}\right)A_{k} - \delta_{k}M_{k}}{\xi}\right). \tag{43}$$

By inserting Eq. (37) into this equation (Eq. (43)), we obtain

$$Y_{k}T_{k}\frac{ds_{k}}{dt} = -\underbrace{\left(Y_{k}T_{k}\frac{\partial s_{k}}{\partial p}\Big|_{\rho_{k}}\rho c_{eff}^{2} + Y_{k}T_{k}\rho_{k}\lambda_{k}\frac{\partial s_{k}}{\partial \rho_{k}}\Big|_{p}\right)\nabla\cdot\mathbf{u}}_{\text{part II}}$$

$$+\underbrace{Y_{k}T_{k}\frac{\partial s_{k}}{\partial p}\Big|_{\rho_{k}}\left(\frac{\Theta-\mathbf{u}\cdot\mathbf{P}}{\xi} - \sum_{k=1}^{K}\frac{e_{k}M_{k}}{\xi}\right)}_{\text{part II}}.$$

$$(44)$$

Part I of Eq. (44) is consistent with the GFE model sharing the mixture entropy (see the discussion in Section 3.1). As discussed in Section 3.1, the interface-sharpening is not a real physical effect, and Part II of Eq. (44) should always be zero. Therefore, we obtain

$$\Theta = \mathbf{u} \cdot \mathbf{P} + \sum_{k=1}^{K} e_k M_k. \tag{45}$$

### 3.3. Generic five-equation model with immiscibility preservation conditions

The above results show that there is a consistent relationship among the equations for the mass, momentum, total energy, and volume fractions. To achieve this consistency, the following constraints must be satisfied:

$$\begin{cases} A_k = \frac{M_k}{\rho_k}, \\ \Theta = \mathbf{u} \cdot \mathbf{P} + \sum_{k=1}^K e_k M_k. \end{cases}$$
(46)

Based on these constraints, we further explore the specific expressions of  $M_k$ , P,  $\Theta$ , and  $A_k$ .

First, following the original ACM idea [39,41], we suppose there exists some kind of artificial compression flux vector for each phasic equation, and we assume  $M_k = \nabla \cdot (\rho_k \mathbf{J}_k)$  in this work. For problems with an interface separating pure or nearly pure fluids, the phasic density  $\rho_k$  is constant across the interfaces [19,59]. Therefore, we further obtain  $A_k = \nabla \cdot \mathbf{J}_k$ .

Subsequently, we derive the expression of **P**. In the past, the consistency between mass and momentum transport has received minimal attention, mostly in the incompressible regime for low Reynolds numbers and low density ratios [32]. However, this consistency correction to the momentum is crucial for compressible flows, without which the spurious momentum (or velocity) contribution to the kinetic energy may eventually lead to unbounded solutions [32]. The consistency of mass and momentum transport guarantees the physical coupling between the mass conservation equations (Eqs. (7)-(9)) and the momentum conservation equation (Eq. (10)). The momentum flux should be correlated with the mass flux, which has been guaranteed at a continuous level [32,33]. Therefore, it is easy to see that each material has a corresponding momentum flux as  $(\rho_k \mathbf{J}_k) \otimes \mathbf{u}$ , and correspondingly  $\mathbf{P} = \sum_{k=1}^K \nabla \cdot (\rho_k \mathbf{J}_k \otimes \mathbf{u})$ .

Finally, we have  $\Theta = \sum_{k=1}^K \mathbf{u} \cdot \nabla \cdot \left( \rho_k \mathbf{J}_k \otimes \mathbf{u} \right) + \sum_{k=1}^K \rho_k e_k \nabla \cdot \mathbf{J}_k$ . In compressible flows, the internal energy is not a conserved quantity because of the reversible exchange of compression/expansion work between the internal and kinetic energies. However, the sum of the internal and kinetic energies is conserved [32]. Here, we consider an interface having a unit normal  $\mathbf{n}_k$ . As the interface-sharpening effect is employed to control the thickness of the interface in the normal direction,  $\mathbf{J}_k$  should be  $\mathbb{J}_k \mathbf{n}_k$  in which  $\mathbb{J}_k$  is an undetermined but scalar quantity. Moreover, it is known that  $\mathbf{u}$ , p,  $p_k$ ,  $e_k$  are also constant along the normal direction of the interface [19]. Therefore, we have  $\mathbf{J}_k \cdot \nabla(\rho_k e_k) = \mathbb{J}_k \mathbf{n}_k \cdot \nabla(\rho_k e_k) = 0$  (for  $\mathbf{u}$ , the situation is similar). By bringing these zero-valued quantities into the expression of  $\Theta$ , we obtain  $\Theta = \sum_{k=1}^K \nabla \cdot \left( \left( \frac{\mathbf{u} \cdot \mathbf{u}}{2} + e_k \right) \rho_k \mathbf{J}_k \right)$ .

In summary, the interface-sharpening effect of the GFE model obtained in this study can be represented as

$$M_{\nu} = \nabla \cdot (\rho_{\nu} \mathbf{J}_{\nu}),$$
 (47)

$$\mathbf{P} = \sum_{k=1}^{K} \nabla \cdot \left( \rho_k \mathbf{J}_k \otimes \mathbf{u} \right), \tag{48}$$

$$\Theta = \sum_{k=1}^{K} \nabla \cdot \left( \left( \frac{\mathbf{u} \cdot \mathbf{u}}{2} + e_k \right) \rho_k \mathbf{J}_k \right), \tag{49}$$

$$A_k = \nabla \cdot \mathbf{J}_k. \tag{50}$$

This result not only conserves mass, momentum, and total energy but also demonstrates compatibility with thermodynamics. Moreover, the above results are general (in contrast to those in [6,32]) and independent of the definition of  $J_k$ . However, it is worth pointing out that such result is suitable only for mixtures in the interfacial zone.

#### 4. Numerical method

In this section, a general numerical approach is proposed to solve Eqs. (7)-(14) (with expressions of  $M_k$ ,  $\mathbf{P}$ ,  $\Theta$ , and  $A_k$  given in Eqs. (47)-(50) respectively). We consider the one-dimensional case of this model (Eqs. (7)-(14)) to highlight the concepts of the proposed numerical approach without losing generality:

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \left( \mathbf{F}(\mathbf{U}) + \mathbf{G} \right)}{\partial x} = \mathbf{S}(\mathbf{U}, \frac{\partial \mathbf{U}}{\partial x}),\tag{51}$$

where

$$\mathbf{U} = \begin{pmatrix} \alpha_1 \rho_1 \\ \dots \\ \alpha_K \rho_K \\ \alpha \rho u \\ \alpha \rho E \\ \alpha_1 \\ \dots \\ \alpha_{K-1} \end{pmatrix}, \mathbf{F}(\mathbf{U}) = \begin{pmatrix} \alpha_1 \rho_1 u \\ \dots \\ \alpha_K \rho_K u \\ \rho u^2 + p \\ \rho E u + p u \\ \alpha_1 u \\ \dots \\ \alpha_{K-1} u \end{pmatrix},$$

and

$$\mathbf{S}(\mathbf{U}, \frac{\partial \mathbf{U}}{\partial x}) = \begin{pmatrix} 0 & & & \\ & \dots & & \\ & 0 & & \\ & 0 & & \\ & \alpha_1 \lambda_1 \frac{\partial u}{\partial x} & & \\ & \dots & & \\ & \alpha_{K-1} \lambda_{K-1} \frac{\partial u}{\partial x} & & \\ & & \dots & & \\ & & \alpha_{K-1} \lambda_{K-1} \frac{\partial u}{\partial x} & & \\ \end{pmatrix}, \mathbf{G} = \begin{pmatrix} \rho_1 J_1^x & & & \\ & \dots & & \\ & \rho_K J_K^x & & \\ & \sum_{k=1}^K \rho_k \mathbf{U} J_k^x & & \\ & \sum_{k=1}^K \rho_k \mathbf{U} \frac{u^2}{2} + e_k \end{pmatrix} J_k^x \\ & & \dots \\ & J_k^x & & \dots \\ & J_k^x & & \dots \end{pmatrix}.$$

The spatial domain is discretized into N computational cells  $I_i = [x_i - \Delta x/2, x_i + \Delta x/2]$ , where  $\Delta x$  indicates the width of cell  $I_i$  and the location of the cell center is denoted as  $x_i$ . Without losing generality, we consider here the specific process from time  $t^n$  to time  $t^{n+1}$  with a timestep  $\Delta t$ .

The proposed numerical approach is based on the classical finite-volume method. Specifically, Eq. (51) in the computational cells  $I_i$  can be discretized as

$$\frac{\overline{\mathbf{U}}_{i}^{n+1} - \overline{\mathbf{U}}_{i}^{n}}{\Delta t} = -\frac{\widehat{\mathbb{F}}_{i+1/2}^{n} - \widehat{\mathbb{F}}_{i-1/2}^{n}}{\Delta x} + \widehat{\mathbf{S}}_{i}^{n},\tag{52}$$

where  $\overline{\mathbf{U}}_i^n$  and  $\overline{\mathbf{U}}_i^{n+1}$  denote the cell average of  $\mathbf{U}$  in  $I_i$  at times  $t^n$  and  $t^{n+1}$ , respectively,  $\widehat{\mathbb{F}}_{i\pm 1/2}^n$  denotes the net flux of the upwind numerical fluxes  $\widehat{\mathbf{F}}_{i\pm 1/2}^n$  and the interface-sharpening flux  $\widehat{\mathbf{G}}_{i\pm 1/2}^n$  at cell boundaries, that is,

$$\widehat{\mathbb{F}}_{i\pm 1/2}^n = \widehat{\mathbf{F}}_{i\pm 1/2}^n + \widehat{\mathbf{G}}_{i\pm 1/2}^n. \tag{53}$$

For clarity, the superscript n is omitted in  $\hat{\mathbf{F}}_{i\pm1/2}^n$ ,  $\hat{\mathbf{G}}_{i\pm1/2}^n$ , and  $\hat{\mathbf{S}}_i^n$ , hereafter. Calculating  $\hat{\mathbf{F}}_{i\pm1/2}$  and  $\hat{\mathbf{S}}_i$  follows the extended Godunov-type finite-volume method [2]. Using a suitable reconstruction scheme, we obtain the left and right states on either side of each cell edge. Riemann solvers are then utilized to derive the upwind numerical fluxes  $\hat{\mathbf{F}}_{i\pm1/2}$ , and the source term  $\hat{\mathbf{S}}_i$  containing the velocity divergence term is consistently discretized using  $\hat{u}_{i+1/2}$  [2]. A detailed description of the numerical approach is provided in our previous studies [2,1]. To avoid spurious pressure oscillations for isolated interfaces between fluids with different material properties, the consistency among physical quantities must be maintained in the reconstruction process. Therefore, the reconstruction variables must be cautiously selected [1,56,13], or a common reconstruction scheme must be utilized for  $\alpha_k$  and  $\alpha_k \rho_k$  [60,61] in the GFE model. In fact, numerical analysts have developed numerical schemes that preserve some of the key mathematical and physical properties of

the differential models they aim to approximate in their finite-dimensional algebraic representations [62,63]. Such numerical schemes are called structure-preserving/physical-compatible discretizing methods, which preserve properties such as energy, monotonicity, maximum principles, symmetries, and involutions of the continuum models at the discrete level [63]. In the context of DIM, designing structure-preserving/physical-compatible discretizing algorithms for these models [64,59,65,8,1,66,67,60,68] has long been an active topic. In our opinion, the aforementioned specific skills required for reconstructing the GFE model are a concrete example of this research direction. Following our previous work [1], we utilize the variables  $\mathbf{W} = (\overline{\rho}_1, \cdots, \overline{\rho}_K, \overline{u}, \overline{p}, \overline{\alpha}_1, \cdots, \overline{\alpha}_{K-1})^T$  to perform the reconstruction for obtaining the left and right states at the cell boundaries, which are then utilized to provide the upwind numerical fluxes  $\hat{\mathbf{F}}_{i\pm1/2}=(\hat{F}_{i+1/2}^{\alpha_1\rho_1},\cdots,\hat{F}_{i+1/2}^{\alpha_K\rho_K},\hat{F}_{i+1/2}^{\rho\mu},\hat{F}_{i+1/2}^{\rho\mu},\hat{F}_{i+1/2}^{\rho\mu},\hat{F}_{i+1/2}^{\rho\mu},\hat{F}_{i+1/2}^{\alpha_K-1})^T$  and  $\hat{u}_{i+1/2}$  [2,1]. Based on the above methodology, we further propose a conservative and consistent multimaterial artificial compression method

(MMACM) to solve the interface-sharpening flux  $\hat{\mathbf{G}}_{i+1/2}$ .

#### 4.1. Consistent numerical framework

First, we introduce an operator. For any variable f, the operator  $(\widetilde{f})_{i+1/2}$  is defined as

$$(\widetilde{f})_{i+1/2} = \begin{cases} f_i, & \text{if } \widehat{u}_{i+1/2} > 0, \\ f_{i+1}, & \text{else}, \end{cases}$$
 (54)

where  $\hat{u}_{i+1/2}$  associated with the Rusanov, HLL, and HLLC Riemann solvers was obtained in our previous study [2]. Using this definition, we further design

$$\widehat{G}_{k,i+1/2}^{\alpha_k \rho_k} = \widetilde{(\rho_k)}_{i+1/2} \widehat{G}_{k,i+1/2}^{\alpha_k},$$

$$\widehat{G}_{k,i+1/2}^{\rho u} = \sum_{k=1}^K \widetilde{(\rho_k u)}_{i+1/2} \widehat{G}_{k,i+1/2}^{\alpha_k},$$

$$\widehat{G}_{k,i+1/2}^{\rho E_k} = \sum_{k=1}^K \widetilde{(\rho_k E_k)}_{i+1/2} \widehat{G}_{k,i+1/2}^{\alpha_k}.$$
(55)

Numerical tests showed that this framework preserves the consistency of the GFE model with the interface-sharpening effect proposed in the previous section. The remaining problem involves determining  $\hat{G}_{k,i+1/2}^{\alpha_k}$ , which is discussed in the next section.

## 4.2. Numerical determination of $\hat{G}_{k,i+1/2}^{\alpha_k}$

To construct a simple and efficient method, we do not look for the theoretical expression of  $J_k$  but directly determine  $\hat{G}_{k,i+1/2}^{a_k}$  via a purely numerical approach. Using the same concept as in [1],  $\hat{G}_{k,i+1/2}^{\alpha_k}$  is defined as

$$\widehat{G}_{k,i+1/2}^{\alpha_k} = \widetilde{(\mathcal{H}_k)}_{i+1/2} \left[ \widehat{u}_{i+1/2} \check{\alpha}_{k,i+1/2} - \widehat{F}_{i+1/2}^{\alpha_k} \right], \tag{56}$$

where

$$\mathcal{H}_{k}(\alpha_{k,i}) = \begin{cases} 1 - \frac{\left\| |\alpha_{k,i+1} - \alpha_{k,i}| - \left| \alpha_{k,i} - \alpha_{k,i-1} \right| \right\|^{q}}{\left( \left| \alpha_{k,i} - \alpha_{k,i-1} \right| + \left| \alpha_{k,i+1} - \alpha_{k,i} \right| \right)^{q} + \epsilon}, & \text{for } \epsilon < \alpha_{k,i} < 1 - \epsilon \\ 0, & \text{otherwise} \end{cases}$$

$$(57)$$

is a newly introduced characteristic function utilized in our previous work [1] to measure the configuration of this material in cell i with  $\epsilon = 10^{-6}$ . In this characteristic function, the sensitivity parameter  $\epsilon = 10^{-12}$  has a fixed value to avoid the division of zero, and q has a user-defined power parameter that is usually set between 2 and 4. Ideally,  $\mathcal{H}_k(\alpha_{k,i})$  is nothing but a Heaviside function to detect whether this kth material exists or not. However, it was found [17] that the standard formula  $\rho_k = \alpha_k \rho_k / \alpha_k$  could lead to spurious oscillatory behavior near the interface where both  $\alpha_k \rho_k$  and  $\alpha_k$  have large gradients. Our numerical experiments confirm this statement and further find that this phenomenon tends to occur at the boundary between the interface mixing region and the pure material region, rather than in the core region of the interface mixing region. In these regions,  $\alpha_k \rho_k$  has a finite value, but  $\alpha_k$  is often very small. In this sense, this phenomenon is similar to the small cell problem in the Cartesian cut-cell method [69]. Under this condition, we consider the interface to be an indistinguishable flow structure on this mesh. Thus, we introduce a shock sensor [41] to modify the Heaviside function as in Eq. (57), thus introducing some numerical dissipation to enhance the stability of the MMACM. In addition,  $\check{\alpha}_{k,i+1/2}$  in Eq. (56) is set as

$$\check{\alpha}_{k,i+1/2} = \begin{cases}
\check{\alpha}_{k,i+1/2}^{L}, & \text{if } \widehat{u}_{i+1/2} > 0, \\
\check{\alpha}_{k,i+1/2}^{R}, & \text{else},
\end{cases}$$
(58)

where  $\check{\alpha}_{k,i+1/2}^L$  and  $\check{\alpha}_{k,i+1/2}^R$  are obtained using a discontinuity-preserving scheme based on a steepness-adjustable harmonic limiter [47] (or utilize its self-adjusting steepness-based version [47] directly to ensure that the final scheme obtains essentially non-oscillatory and sharp resolutions for various discontinuities while maintaining the nominal second-order accuracy for smooth regions). For  $\check{\alpha}_{k\,i+1/2}^L$  (the result for  $\check{\alpha}_{k\,i+1/2}^L$  is omitted here due to symmetry), the result is

$$\check{\alpha}_{k,i+1/2}^{L} = \alpha_{k,i} + \frac{1}{2}\psi\left(r_{i+1/2}\right)(\alpha_{k,i+1} - \alpha_{k,i}),\tag{59}$$

where  $r_{i+1/2} = \frac{\alpha_{k,i} - \alpha_{k,i-1}}{\alpha_{k,i+1} - \alpha_{k,i}}$  and

$$\psi\left(r_{i+1/2}\right) = \frac{\left|r_{i+1/2}\right| + r_{i+1/2}}{1/\beta + r_{i+1/2}}.$$
(60)

In Eq. (60), the steepness parameter  $\beta$  enables the final scheme to exhibit different behaviors [47]. This parameter can be set to a fixed value using an optimization method [47,70] or theoretically determined values related to the local CFL number [48]. The detailed relationship between this scheme and the limited downwind scheme presented by Després and Lagoutière [18] was examined in our previous study [48]. Thus, our method can be viewed as an extension of the limited downwind Lagrange-remap method to the Eulerian framework. Moreover, the adjustability of the steepness parameter  $\beta$  enables us to alleviate the appearance and fragmentation of fine structures (filaments and high-frequency instabilities [21]) in the limited downwind scheme. In this study, a fixed value of  $\beta = 2.9$  is utilized for this steepness-adjustable harmonic limiter, which generally lies near the upper bound of the TVD region for semi-discrete schemes [47]. This value is further confirmed by a least squares optimization method [70].

Given the definition of  $\hat{G}_{k,i+1/2}^{a_k}$  (Eq. (56)), Eq. (55) provides a novel interface-sharpening technique for the GFE model, in contrast to the results in existing studies. The main characteristic of this method is that it preserves the conservation and consistency properties of the physical models.

#### 4.3. Multi-dimensional case

This work aims to construct an interface-sharpening method that is as simple and efficient as possible. However, directly utilizing the dimension-by-dimension technique often results in ruffling the interface that aligns near the direction of the velocity, seriously reducing the confidence in numerical simulation results. In this section, we further explore the multi-dimensional MMACM calculation by utilizing the attractive feature of the above one-dimensional building block.

Specifically, we introduce the geometric information of the interface into the  $\mathcal{H}_k$  function (Eq. (57)). For the two-dimensional case, the unit normal  $\mathbf{n}_{k,(i,j)} = (n_{k,(i,j)}^x, n_{k,(i,j)}^y)$  at cell (i,j) is computed by differencing the  $\alpha_k$ , and we utilize Parker and Youngs' algorithm described in [71]. The  $\mathcal{H}_k(\alpha_{k,(i,j)})$  for the x direction is then calculated by replacing Eq. (57) with

$$\mathcal{H}_{k}(\alpha_{k,(i,j)}) = \begin{cases} \left(1 - \frac{\left\|\alpha_{k,(i+1,j)} - \alpha_{k,(i,j)}\right\| - \left|\alpha_{k,(i,j)} - \alpha_{k,(i-1,j)}\right|\right\|^{q}}{\left(\left|\alpha_{k,(i,j)} - \alpha_{k,(i-1,j)}\right| + \left|\alpha_{k,(i+1,j)} - \alpha_{k,(i,j)}\right|\right)^{q} + \varepsilon}\right) \left|n_{k,(i,j)}^{x}\right|, & \text{for } \epsilon < \alpha_{k,(i,j)} < 1 - \epsilon \\ 0, & \text{otherwise} \end{cases}$$

The same applies to the y direction. The resulting interface-sharpening technique, called MMACM, is extremely simple; however, its numerical accuracy is competitive. Moreover, MMACM can be straightforwardly extended to three dimensions (x, y, z) by adding another sweep in the z direction.

Remark 2. Increasing the steepness parameter  $\beta$  in the steepness-adjustable harmonic limiter (Eq. (60)) will generate a compressive or anti-diffusion effect on numerical solutions, effectively preserving discontinuous flow structures [47]. However, this compressive or anti-diffusion effect will significantly affect the stability of the final scheme. Therefore, nonlinear stability must be considered in the final numerical schemes. Various nonlinear stability theories, such as monotonicity preservation, maximum principle preserving, positivity, and TVD, can be utilized to determine the supremum of the steepness parameter [47]. To satisfy the TVD theory, the discontinuity-preserving scheme based on the steepness-adjustable harmonic limiter should be CFL-dependent [48]. Therefore, temporal discretization must also be carefully considered beyond spatial discretization. The dissipation of the explicit forward Euler scheme of time integration can be considered negative when coupled with this discontinuity-preserving scheme, and the stability requirements can be slightly relaxed by utilizing high-order temporal discretization methods [48]. Therefore, when MMACM is utilized, the third-order Runge-Kutta scheme [72,73] is recommended to achieve higher numerical stability [48].

#### 5. Numerical tests

This method was integrated into the in-house code called  $\mathrm{H}^3\mathrm{Flow}$  [2], which was developed by the first author. In this section, numerical tests in one and two dimensions are presented to verify the proposed method. This study primarily considers the pure interface problem. Therefore, the GFE model with the equal compressibility assumption is employed (unless otherwise specified). In terms of numerical methods, the third-order SSP Runge-Kutta scheme [72,73], a TVD reconstruction with the Minmod limiter [74], and the HLLC solver [75,76,2] are adopted. The CFL number is set to 0.2 due to the fixed value of  $\beta = 2.9$  [48,47].

![](_page_11_Figure_2.jpeg)

**Fig. 1.** Advection of interface in uniform pressure and velocity flow at = 0*.*2 on 400, 800, and 1600 grid cells. Numerical results without interface-sharpening (without IS) techniques and with interface-sharpening (with IS) techniques are presented. Top-left: density; top-right: volume fraction of air; bottom-left: velocity; and bottom-right: pressure.

Moreover, the numerical schlieren-type image of any physical variable :

$$\exp\left(-10\frac{\left|\nabla f\right| - \left|\nabla f\right|_{\min}}{\left|\nabla f\right|_{\max} - \left|\nabla f\right|_{\min}}\right),\,$$

is utilized in the study.

#### *5.1. Advection of interface in uniform pressure and velocity flow*

The following moving material interface problem involving two ideal gases was first examined:

$$(\rho_1, \rho_2, u, p, \alpha_1, \gamma) = \begin{cases} (1, 0.125, 2, 1/1.4, 1 - \epsilon, 1.667), & 0 \le x \le 0.5, \\ (1, 0.125, 2, 1/1.4, \epsilon, 1.4), & 0.5 < x \le 3. \end{cases}$$

The boundary conditions were constant on both the left and right sides of the domain. The computation was performed on three meshes of 400, 800, and 1600 cells, respectively. Fig. 1 presents the flow variables at = 0*.*2.

Fig. 1 indicates that a good agreement with the exact solution is obtained. No spurious oscillations were observed near the interface, regardless of whether the proposed interface-sharpening method (i.e., MMACM) was applied. Compared to the results without MMACM, where tens of points were found in the transition zone of the diffused interface, only approximately three cells were found in the transition zone of the interface when the proposed interface-sharpening technique was added. Moreover, the

**Table 2** Convergence of  $\alpha_1$  for advection of interface in uniform pressure and velocity flow. Error for solution variable  $\alpha_1$  is defined by  $\sum_{l}^{N} \left| \alpha_{1,l} - \alpha_{1,l}^{\text{exact}} \right| / N$ .

| Grid cells | Without interface-sharpening |       | With interface-sharpening |       |  |
|------------|------------------------------|-------|---------------------------|-------|--|
| oria cens  | Error                        | Order | Error                     | Order |  |
| 300        | 7.62×10 <sup>-3</sup>        | -     | 2.03×10 <sup>-3</sup>     | -     |  |
| 600        | $4.87 \times 10^{-3}$        | 0.65  | $1.01\times10^{-3}$       | 1.01  |  |
| 1200       | $3.10 \times 10^{-3}$        | 0.65  | $5.10 \times 10^{-4}$     | 0.99  |  |
| 2400       | $1.97 \times 10^{-3}$        | 0.65  | $2.65 \times 10^{-4}$     | 0.95  |  |

**Table 3** Conservation of mass of each material and total energy of mixture for advection of interface in uniform pressure and velocity flow with  $\Delta m_k = \left| \int \left( \rho_k(x,t) \alpha_k(x,t) - \rho_k(x,0) \alpha_k(x,0) \right) dx + \int \left( \rho_k(3,t) \alpha_k(3,t) u(3,t) - \rho_k(0,t) \alpha_k(0,t) u(0,t) \right) dt \right|$  and  $\Delta \mathfrak{E} = \left| \int \left( \rho(x,t) E(x,t) - \rho(x,0) E(x,0) \right) dx + \int \left( (\rho(3,t) E(3,t) + p(3,t)) u(3,t) - (\rho(0,t) E(0,t) + p(0,t)) u(0,t) \right) dt \right|$  at t = 0.2.

|   | Grid cells          | Without interface-sharpening                                         |                                                                            |                                                                      | With interface-sharpening                                                  |                                                                      |                                                                      |
|---|---------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
|   | orra cons           | $\Delta m_1$                                                         | $\Delta m_2$                                                               | Δ&                                                                   | $\Delta m_1$                                                               | $\Delta m_2$                                                         | Δ&                                                                   |
| • | 600<br>1200<br>2400 | $9.44 \times 10^{-16}$ $1.05 \times 10^{-14}$ $4.61 \times 10^{-15}$ | $1.57 \times 10^{-15}$<br>$1.39 \times 10^{-15}$<br>$1.89 \times 10^{-15}$ | $1.38 \times 10^{-13}$ $3.82 \times 10^{-13}$ $7.15 \times 10^{-13}$ | $3.89 \times 10^{-16}$<br>$5.22 \times 10^{-15}$<br>$4.16 \times 10^{-15}$ | $2.98 \times 10^{-15}$ $1.03 \times 10^{-15}$ $8.47 \times 10^{-15}$ | $2.17 \times 10^{-13}$ $4.97 \times 10^{-13}$ $8.14 \times 10^{-13}$ |

results indicate that the present approach maintains the pressure and velocity equilibrium and introduces errors only in the order of round-off.

Table 2 gives the convergence of  $\alpha_1$ . The error for the solution variable  $\alpha_1$  is defined by  $\sum_i^N \left| \alpha_{1,i} - \alpha_{1,i}^{\rm exact} \right| / N$ . It is observed that the convergence rates for the method without the interface-sharpening method are substantially lower than one. However, the magnitude of errors is significantly less and a first-order convergence rate is observed when the interface-sharpening method is applied. The results in Table 2 demonstrate that the reduction in error indicates that the discrepancy between the exact and computed solution is reduced.

The new interface-sharpening method proposed in this study follows the standard Godunov-type finite volume framework; thus, its conservation is natural. To demonstrate this statement, Table 3 lists the conservation of mass of each material and the total energy of the mixture. It is observed that the proposed method has good conservation properties up to the order of round-off.

#### 5.2. Water-air shock tube test

We considered a one-dimensional water-air shock tube test [2]. A 1-m long shock tube containing two chambers was separated by an interface at x=0.8 m, with each chamber containing a mixture of water and air. The stiffened gas EOS  $p_k=(\gamma_k-1)\rho_k e_k-\gamma_k P_{\infty,k}$  was adopted for water (k=1), with  $\gamma_1=4.4$  and  $P_{\infty,1}=6\times10^8$  Pa. The ideal gas EOS  $p_k=(\gamma_k-1)\rho_k e_k$  was adopted for air (k=2) with  $\gamma_2=1.4$ . The initial densities were  $\rho_1=1000$  kg/m³ for water and  $\rho_2=10$  kg/m³ for air. The left chamber contained a very small volume fraction of air,  $\alpha_2=\epsilon$ , and the pressure was  $10^9$  Pa. The volume fractions were reversed in the right chamber, and the pressure was  $10^5$  Pa. Three meshes of 200, 400, and 800 cells were adopted to show the convergence of the proposed method, and a reference solution was obtained using a finer mesh with 50 000 uniform cells.

Fig. 2 presents the numerical results at  $t = 220 \,\mu s$  for the one-dimensional water-air shock tube problem without interface-sharpening (without IS) techniques and with interface-sharpening (with IS) techniques. It clearly shows that the proposed method generated essentially non-oscillatory and interface-sharpened results.

However, it should be noted that an inaccurate numerical result in the shock location is obtained in Fig. 2. Moreover, it is found that there is a slight overshoot on the right side of the interface (see Fig. 3, which shows enlarged views at the interface location in Fig. 2). For this problem, there is a very strong rarefaction wave due to the high density ratio. This leads to a numerical problem that will even exist for Euler equations that an inaccurate numerical result (especially in the shock location and overshoot near the contact/interface), even with a very fine grid [77]. This finding, in accordance with the report [78], but beyond the scope of the present study, reveals that the Eulerian shock-capturing methods work very inefficiently when applied to problems with an initial high density ratio as well as a high pressure ratio, and may give inaccurate numerical results even over a very fine mesh.

#### 5.3. Rayleigh collapse

Rayleigh collapse refers to the symmetric collapse process of a gas bubble subjected to high pressure in the surrounding still fluid. This flow problem can be utilized to check the accuracy of numerical methods with respect to the interface evolution [17,79]. In this work, we consider the two-dimensional Rayleigh collapse problem and adopt the setup in [80]. Due to the symmetry of the flow problem, only a quarter of the bubble is computed. The computational domain is  $[0,200] \, \text{mm} \times [0,200] \, \text{mm}$ , in which a gas bubble with radius  $R_0 = 1 \, \text{mm}$  is initially placed at  $[0 \, \text{mm}, 0 \, \text{mm}]$ . A uniform fine mesh is employed, and the symmetric boundary

![](_page_13_Figure_2.jpeg)

Fig. 2. One-dimensional water-air shock tube problem at  $t = 220 \mu s$  on 200, 400, and 800 grid cells. Numerical results without interface-sharpening (without IS) techniques and with interface-sharpening (with IS) techniques are presented. Top-left: density; top-right: volume fraction of water; bottom-left: velocity; and bottom-right: pressure.

condition is utilized for the left and bottom sides while the extrapolation boundary condition is utilized for the others. The states of the bubble are  $p_0 = 1 \times 10^5$  Pa,  $\rho = 0.19$  kg/m<sup>3</sup> and u = v = 0 m/s. The states of the surrounding still liquid are  $\rho = 1000$  kg/m<sup>3</sup> and  $p(r) = p_{\infty} + R_0(p_0 - p_{\infty})/r$ , where r is the distance to the origin, and  $r \ge R_0$ . The pressure in the far field  $p_{\infty} = 1 \times 10^7$  Pa.

For this problem, Tiwari et al. [17] found, as confirmed further in [79], that the Allaire-Massoni five-equation model (equivalent to the GFE model with the equal compressibility assumption) is unable to accurately represent a spherical bubble collapse. The Kapila five-equation model (equivalent to the GFE model with the isentropic assumption) is required to ensure good agreement with the theoretical solution. Therefore, the "isentropic assumption" is utilized in this study. Numerical results on successively refined meshes are shown in Fig. 4, with respect to the time evolution of the effective radius of the bubble. The effective radius  $R_e$  is defined as  $R_e = 2\sqrt{V/\pi}/R_0$ , where the instantaneous volume V of the gas bubble can be calculated by integrating the gas volume fraction over the domain. The "exact solution" is the one-dimensional axisymmetric result provided in [80]. From Fig. 4, we can see that the two-dimensional results converge to the exact solution with mesh refinement.

To quantitatively estimate the accuracy, we define the error of the volume fraction of gas  $\alpha_{2,i,j}$  as:  $\frac{\sum_{i,j} \left| \alpha_{2,i,j} - \alpha_{2,i,j}^{\text{exact}} \right|}{N}$ , where  $\alpha_{2,i,j}^{\text{exact}}$  represents the volume fraction reconstructed from the exact solution of the bubble radius (0.227298 mm [80]). Table 4 shows the error. It is clear that MMACM is more accurate than the standard method on the same mesh. Specifically, the absolute error is greatly reduced, and the convergence accuracy is improved, which is close to the first-order accuracy.

In addition, we calculated this problem with the equal compressibility assumption and the isentropic assumption on  $10000 \times 10000$  grid cells to further prove that the newly proposed MMACM interface-sharpening technique is independent of different assumptions

![](_page_14_Figure_2.jpeg)

Fig. 3. Zoomed-in image of area near interface in Fig. 2. Left: density; right: volume fraction of water.

![](_page_14_Figure_4.jpeg)

Fig. 4. Numerical results of two-dimensional Rayleigh collapse problem with/without interface-sharpening (IS) techniques. Time evolution of effectively radius of bubble on  $2500 \times 2500$ ,  $5000 \times 5000$  and  $10000 \times 10000$  grid cells.

**Table 4** Convergence of  $\alpha_2$  for two-dimensional Rayleigh collapse problem. Error is defined as  $\frac{\sum_{i,j} |a_{2i,j} - a_{2inj}^{\text{exer}}|}{N}$ , where  $a_{2i,i,j}^{\text{exer}}$  represents volume fraction reconstructed from exact solution of bubble radius.

| Grid cells  | Without inte          | erface-sharpening | With interface-sharpening |       |  |
|-------------|-----------------------|-------------------|---------------------------|-------|--|
| GIId Collo  | Error                 | Order             | Error                     | Order |  |
| 2500×2500   | $9.75 \times 10^{-7}$ | -                 | $6.75 \times 10^{-7}$     | -     |  |
| 5000×5000   | $5.45 \times 10^{-7}$ | 0.84              | $3.01\times10^{-7}$       | 1.17  |  |
| 10000×10000 | $3.31\times10^{-7}$   | 0.72              | $1.23 \times 10^{-7}$     | 1.29  |  |

in the GFE model. Fig. 5 presents the results, which are in accordance with the reports [17,79]. Our results reveal that the Allaire-Massoni five-equation model (equivalent to the GFE model with the equal compressibility assumption) is unable to accurately represent a spherical bubble collapse.

#### 5.4. Air-R22 shock-bubble interaction

The next problem was the classical air-R22 shock-bubble interaction problem, involving the collision of a shockwave in air with a circular R22 bubble [81]. In this test, we utilized the ideal gas EOS  $p_k = (\gamma_k - 1)\rho_k e_k$  to model the materials, setting the

![](_page_15_Figure_2.jpeg)

Fig. 5. Time evolution of effective radius of bubble on  $10000 \times 10000$  grid cells with interface-sharpening (IS) technique. Equal compressibility assumption and isentropic assumption are utilized, respectively.

ratio of the specific heat  $\gamma_k$  to 1.4 for air (k=1) and 1.249 for R22 (k=2). This test was conducted as follows [1]: Initially, a planar leftward-moving Mach 1.22 shockwave in air at x=275 mm traveled toward a stationary R22 gas bubble with a center at (x,y)=(225 mm, 44.5 mm) and radius of  $r_0=25 \text{ mm}$  in front of the shockwave.

The initial conditions are given by

$$(\rho_1, \rho_2, u, v, p, \alpha_1) = \begin{cases} (1.225, 3.863, 0, 0, 1.01325 \times 10^5, \epsilon), & \text{Bubble}, \\ (1.225, 3.863, 0, 0, 1.01325 \times 10^5, 1 - \epsilon), & \text{Pre-shock}, \\ (1.686, 3.863, -113.5, 0, 1.59 \times 10^5, 1 - \epsilon), & \text{Post-shock}. \end{cases}$$

The computational domain was  $(x, y) \in [0, 445] \times [0, 89] \text{ mm}^2$ , where the solid wall boundary condition was imposed on the top and bottom boundaries and an outflow boundary with zero gradient was imposed on the remaining sides. The shockwave reached the R22 bubble after approximately 60 µs [1,51], and we considered this instant as t = 0.

We utilized two sets of grids to verify the convergence of the new method. Fig. 6 presents the numerical schlieren-type images of the mixture density at four different times (i.e.,  $t = 55 \mu s$ ,  $t = 247 \mu s$ ,  $t = 417 \mu s$ , and  $t = 1020 \mu s$ ) on a uniform mesh with  $3200 \times 640$  cells. Fig. 7 shows the same results except on a uniform mesh with  $6400 \times 1280$  cells.

First, for the results without the application of MMACM, the large-scale structures and interface evolution agree well with the reference literature, which confirms the validity of the underlying governing equations and numerical method. Second, with the application of MMACM, the interfaces are significantly better resolved while the large-scale structures remain consistent with the reference solution where no sharpening is applied and with the reference literature. Third, while no pointwise convergence for the numerical solution of the compressible Euler-like equations as an initial-value problem can be expected [5], the interface evolution with the application of the interface-sharpening technique is consistently resolved, with increased recovery of small-scale structures as the grid resolution increases.

#### 5.5. Collapse of air bubble under strong shock in water

We considered the collapse of an air cavity in water by a Mach 1.72 shock [82,7]. The rectangular computational domain for this problem was  $24 \times 24$  mm<sup>2</sup>. An air bubble with a radius of 3 mm was placed in water with its center at (12 mm, 12 mm), and the shock wave was initiated at x = 6.6 mm. The thermodynamic behavior of water (k = 1) was modeled by the stiffened gas with  $\gamma_1 = 4.4$ ,  $q_1 = 0$  J/kg, and  $P_{\infty,1} = 6 \times 10^8$  Pa. The air (k = 2) was modeled as an ideal gas with  $\gamma_2 = 1.4$ . The rest of the initial condition is

$$(\rho_1,\rho_2,u,v,p,\alpha_1) = \begin{cases} (1000 \text{ kg/m}^3, 1 \text{ kg/m}^3, 0, 0, 1 \times 10^5 \text{ Pa}, 1 - \epsilon), & \text{air bubble,} \\ (1000 \text{ kg/m}^3, 1 \text{ kg/m}^3, 0, 0, 1 \times 10^5 \text{ Pa}, \epsilon), & \text{Pre-shock,} \\ (1323.65 \text{ kg/m}^3, 1 \text{ kg/m}^3, 681.58 \text{ m/s}, 0, 1.9 \times 10^9 \text{ Pa}, \epsilon), & \text{Post-shock.} \end{cases}$$

The computational domain was discretized using a Cartesian grid with uniform mesh spacings of 400 and 800 cells per initial diameter of the air bubble, and extrapolation boundary conditions were applied on all the boundaries.

![](_page_16_Picture_2.jpeg)

**Fig. 6.** Numerical results of air-R22 shock-bubble interaction problem without interface-sharpening (left column) techniques and with interface-sharpening (right column) techniques. Numerical schlieren-type images of mixture density at = 55 μs (first row), = 247 μs (second row), = 417 μs (third row), and = 1020 μs (fourth row) on uniform mesh with 3200 × 640 cells are presented.

![](_page_17_Picture_2.jpeg)

**Fig. 7.** The same as Fig. [6](#page-16-0) except on uniform mesh with 6400 × 1280 cells.

![](_page_18_Figure_2.jpeg)

**Fig. 8.** Numerical results of collapse of air bubble under strong shock in water at = 2*.*2 μs (first column), = 3*.*5 μs (second column), = 3*.*8 μs (third column), and = 4*.*5 μs (fourth column). First and second rows are numerical schlieren-type images (upper half) of mixture density and contours of the pressure (lower half) without interface-sharpening and with interface-sharpening, respectively. All computations were performed on uniform mesh of 1600 × 1600 cells (corresponding to 400 cells per initial diameter of air bubble).

Computation was first performed utilizing a uniform mesh of 1600 × 1600 cells (corresponding to 400 cells per initial diameter of the air bubble). Numerical results at four different times ( = 2*.*2 μs, = 3*.*5 μs, = 3*.*8 μs, and = 4*.*5 μs), including schlieren-type images of density (upper half in each figure) and the contours of pressure (lower half in each figure), are shown in Fig. 8. Our results are in good agreement with [\[7\]](#page-28-0) at the same grid resolution, while the MMACM provides a sharp description of the material interface.

Next, computation was further performed under the same conditions except utilizing a uniform mesh of 3200 × 3200 cells (corresponding to 800 cells per initial diameter of the air bubble). Accordingly, the results of schlieren-type images of density and contours of pressure are shown in Fig. [9](#page-19-0). Compared to Fig. 8, we can see again that the interface evolution with the application of the interface-sharpening technique is consistently resolved, with increased recovery of small-scale structures as the grid resolution increases.

### *5.6. Underwater explosion*

We then considered the underwater explosion problem [[6,7,11,12](#page-28-0)]. This test case involved complicated interactions of an air cavity generated from an initial high-pressure region with a planar water-air interface lying above it. The computational domain was [−2*,* 2] × [−1*.*5*,* 2*.*5] m2. The cylindrical air cavity was 0.24 m in diameter and as initially centered at (0 m*,*−0*.*3 m) with a high pressure of 1 × 10<sup>9</sup> Pa and a high density of 1250 kg∕m3. The planar water-air interface was in equilibrium under standard atmospheric conditions at = 0 m. The thermodynamic behavior of water was modeled by the stiffened gas with <sup>1</sup> = 4*.*4, <sup>1</sup> = 0 J/kg, and ∞*,*<sup>1</sup> = 6 × 10<sup>8</sup> Pa. The air was modeled as an ideal gas with <sup>2</sup> = 1*.*4. <sup>A</sup> transparent boundary condition was imposed on the top, left, and right boundaries, whereas a reflection condition was implemented on the bottom boundary. We conducted the simulation using a uniform mesh with 600 × 600 cells.

In Fig. [10](#page-20-0), numerical schlieren-type images of the mixture density and contours of the volume fraction of water at four different times (i.e., = 20 ms, = 95 ms, = 190 ms, and = 316 ms) on a uniform mesh with 600 × 600 cells are presented. Computations using the MMACM prevent numerical smearing of the interface, and the material interfaces are sharply resolved. Overall, the results are also in good agreement with [\[7](#page-28-0)] at the same grid resolution.

For comparison, we recomputed the same problem using a finer mesh with 1200 × 1200 cells. The results are shown in Fig. [11](#page-21-0). It can be seen that the MMACM provides a sharp description of the material interface and maintains its width even as the air cavity undergoes expansion and significantly deforms the initially planar water-air interface. The solution is in good qualitative agreement with the results published in [\[7\]](#page-28-0). These results further confirm the aforementioned conclusions.

It should be pointed out that we cannot theoretically prove that MMACM can strictly maintain the thickness of the interface. In Figs. [10](#page-20-0) and [11,](#page-21-0) a portion of the interface initially has no thickness as the velocity is zero, since the HLLC solver can precisely keep a stationary interface [\[75,76](#page-29-0)]. However, after the interface flows, numerical dissipation occurs, and the interface is smeared. Previous interface-sharpening techniques, although they can theoretically keep the thickness, usually smear the initial interface by utilizing a hyperbolic tangent function [\[6,7,17\]](#page-28-0). Otherwise, unphysical oscillations or numerical instabilities may occur [\[79\]](#page-29-0). However, such

![](_page_19_Figure_2.jpeg)

Fig. 9. Same as Fig. 8, except on uniform mesh with 3200 × 3200 cells (corresponding to 800 cells per initial diameter of air bubble).

**Table 5** Interaction of shockwave in molybdenum with midocean ridge basalt (MORB) sample. Material quantities for MORB (k = 1) and molybdenum (k = 2) in HOM EOS.

| k | $\rho_{0,k}({\rm kg/m^3})$ | $c_{0,k}(\mathrm{m/s})$ | $\varsigma_k$ | $\Gamma_{0,k}$ | $\eta_k$ | $p_{0,k}(Pa)$ | $e_{0,k}(J/kg)$ |
|---|----------------------------|-------------------------|---------------|----------------|----------|---------------|-----------------|
| 1 | 2660                       | 2100                    | 1.68          | 1.18           | 1        | 0             | 0               |
| 2 | 9961                       | 4770                    | 1.43          | 2.56           | 1        | 0             | 0               |

smearing procedure may cause physical artifacts [79]. By contrast, the MMACM does not need the smearing procedure, but the price paid is that the thickness of the interface cannot be strictly guaranteed. However, numerical tests show that the MMACM can keep the interface thickness at 2-4 mesh widths, and the interface smearing does not continue essentially.

### 5.7. Interaction of a shock in molybdenum with a MORB sample

We considered the two-dimensional interaction between a shockwave in molybdenum and a mid-ocean ridge basalt (MORB) sample [1] to demonstrate that the proposed method can handle more complicated equations of state. In this problem, a right-traveling planar Mach 1.163 shockwave in molybdenum was initially located at x = 0.5 m within a rectangular block ([0.55, 0.85] × [0, 0.5] m<sup>2</sup>) of MORB liquid in a box domain of size [0, 1.125] × [0, 1.125] m<sup>2</sup>. We utilized the HOM EOS to model the MORB and molybdenum. The shockwave or HOM EOS, which is typically used for solid media (e.g., metals), had the form as follows:

$$p_k = p_{c,k}(\rho_k) + \Gamma_k(\rho_k)\rho_k(e_k - e_{c,k}(\rho_k)),$$

where

$$\begin{split} &\Gamma_k(\rho_k) = \Gamma_{0,k} \left(\frac{\rho_{0,k}}{\rho_k}\right)^{\eta_k}, \\ &p_{c,k}(\rho_k) = p_{0,k} + \frac{\rho_{0,k}c_{0,k}^2 \left(1 - \frac{\rho_{0,k}}{\rho_k}\right)}{1 - \varsigma_{0,k} \left(1 - \frac{\rho_{0,k}}{\rho_k}\right)}, \\ &e_{c,k}(\rho_k) = e_{0,k} + \frac{1}{2\rho_{0,k}} \left(1 - \frac{\rho_{0,k}}{\rho_k}\right) (p_{0,k} + p_{c,k}(\rho_k)). \end{split}$$

The typical values of material properties (i.e.,  $\Gamma_{0,k}$ ,  $\rho_{0,k}$ ,  $\rho_{0,k}$ ,  $\rho_{0,k}$ ,  $\rho_{0,k}$ ,  $\rho_{0,k}$ , and  $e_{0,k}$ ) for MORB and molybdenum were the same as those in our previous study [1] and are listed in Table 5.

![](_page_20_Figure_2.jpeg)

**Fig. 10.** Numerical results of underwater explosion problem on uniform mesh with  $600 \times 600$  cells. Two columns on the left are numerical schlieren-type images of mixture density and contours of volume fraction of water without interface-sharpening at t = 20 ms (first row), t = 95 ms (second row), t = 190 ms (third row), and t = 316 ms (fourth row). Two columns on the right are numerical schlieren-type images of mixture density and contours of volume fraction of water with interface-sharpening at t = 20 ms (first row), t = 95 ms (second row), t = 190 ms (third row), and t = 316 ms (fourth row).

We utilized the setup described in [1]. The initial states of the two fluids are given by

$$(\rho_1,\rho_2,u,v,p,\alpha_1) = \begin{cases} (9961,2660,0,0,0,1-\epsilon), & \text{MORB}, \\ (9961,2660,0,0,0,\epsilon), & \text{Pre-shock}, \\ (11042,2660,543,0,3\times10^{10},\epsilon), & \text{Post-shock}. \end{cases}$$

Computation was performed using three uniform meshes of  $400 \times 400$ ,  $800 \times 800$ , and  $1600 \times 1600$  cells, with a slip wall boundary at the bottom, an inflow boundary on the left side, and an outflow boundary with no gradient on the remaining sides. Numerical results at  $t = 100 \, \mu \text{s}$ , including schlieren-type images of density and pressure, with and without the application of the interface-sharpening technique, are shown in Fig. 12.

It can be seen that the interface is significantly sharper when the interface sharpening is applied, based on the comparison of the numerical schlieren-type images of the mixture density with and without the interface sharpening. Schlieren-type images of density

![](_page_21_Figure_2.jpeg)

**Fig. 11.** The same as Fig. [10](#page-20-0) except on uniform mesh with 1200 × 1200 cells.

and pressure are also in good agreement with [\[7](#page-28-0)] at the same grid resolution. The successful simulation of this example demonstrates that the proposed method is suitable for compressible multimaterial flows with complicated equations of state.

#### *5.8. Three-material triple-point problem*

We considered a two-dimensional problem involving the interaction of three Riemann problems across three initial material discontinuities and a triple point [\[2\]](#page-28-0) to demonstrate that the proposed method can handle more than two materials. The computational domain was [0*,* 7] m × [0*,* 3] m, and it was occupied by three perfect gases. All three materials (modeled with the ideal gas EOS = ( − 1)) were initially at rest. The inital data and EOS parameters are listed in Table [6.](#page-23-0) At the start of the simulation, the pressure in fluid = 1 (in the box domain [0*,* 1] × [0*,* 3] m2) was greater than that in the rest of the domain. This generated a set of waves, including two shocks that traveled toward the right end of the domain, one of which traveled within fluid = 2 (in a box domain [1*,* 7] × [1*.*5*,* 3] m2), and the other of which traveled within fluid = 3 (in a box domain [1*,* 7] × [0*,* 1*.*5] m2). The jump between the densities and material properties across the material interface separating fluids 2 and 3 generated an instability. For this simulation, we utilized a regular mesh and imposed slip wall boundary conditions.

![](_page_22_Figure_2.jpeg)

**Fig. 12.** Numerical results of interaction between shock in molybdenum and a MORB sample at = 100 μs on three uniform meshes of 400 × 400 cells (left column), 800 × 800 (middle column), and 1600 × 1600 (right column) cells. First and second rows are numerical schlieren-type images of mixture density without interfacesharpening and with interface-sharpening, respectively. Third and fourth rows are numerical schlieren-type images of pressure without interface-sharpening and with interface-sharpening, respectively.

**Table 6**Triple point problem. Initial data and EOS parameters.

| k | Density (kg/m³) | Pressure (Pa) | Velocity (m/s, m/s) | $\gamma_k$ |
|---|-----------------|---------------|---------------------|------------|
| 1 | 1.0             | 1.0           | (0,0)               | 1.6        |
| 2 | 0.125           | 0.1           | (0,0)               | 1.5        |
| 3 | 1.0             | 0.1           | (0,0)               | 1.4        |

![](_page_23_Figure_4.jpeg)

Fig. 13. Numerical results of three-material triple-point problem without interface-sharpening (left) techniques and with interface-sharpening (right) techniques on uniform mesh with  $140 \times 60$  cells. First and second rows are numerical schlieren-type images of mixture density at t = 3.5 s and t = 5 s, respectively. Third and fourth rows are contours of  $\sum_{k=1}^{K} k\alpha_k$  at t = 3.5 s and t = 5 s, respectively.

Fig. 13 illustrates the numerical schlieren-like images of the mixture density field and contours of  $\sum_{k=1}^{K} k\alpha_k$  on a uniform mesh with  $140 \times 60$  cells at two instants (t = 3.5 s and t = 5.0 s). This example confirms that the proposed method is suitable for compressible multimaterial flows with more than two materials. This confirms the universality of the proposed MMACM.

Fig. 14 is the same as Fig. 13, except on a uniform mesh with  $700 \times 300$  cells. The interface is sharply captured in a consistent manner with increased recovery of small-scale structures as the grid resolution increases. This further confirms the above conclusions.

![](_page_24_Figure_2.jpeg)

**Fig. 14.** Same as Fig. [13,](#page-23-0) except on uniform mesh with 700 × 300 cells.

### **6. Conclusion**

The immiscibility preservation conditions for compressible multimaterial interfacial flows governed by five-equation-type models are usually destroyed by the numerical implementation of shock-/interface-capturing numerical methods that produce solutions exhibiting excessive numerical diffusion. Utilizing interface-sharpening techniques to stop the progressive smearing of interfaces for longer simulation times is a simple and efficient approach.

However, almost all interface-sharpening techniques encounter a common and still unsolved problem: there is no clear technique for sharpening other equations (mass, momentum, and total energy equations) when the equation describing the interface evolution (such as the volume fraction equation) adopts the above interface-sharpening technique. This problem is very important for compressible multimaterial flows because the volume fraction of each material is no longer conservative, and the phasic density is no longer constant for compressible multimaterial flows. The density and energy in a compressible flow must be solved along with the volume fraction, and special formulations are required to maintain physical consistency, resulting in a balanced state among all variables for a well-defined interface cell. Moreover, the numerical results worsen if the consistency among the equations is broken.

Systematically investigating the immiscibility preservation conditions for compressible multimaterial interfacial flows is very difficult but important. There are two difficulties. The first is the disunity phenomena that exist in the model, and the second involves obtaining an interface-sharpening technique that is thermodynamically compatible and completely conservative in terms of mass, momentum, and total energy.

The disunity of the model is solved by deriving a unified formulation for the five-equation model from our previous study [\[2](#page-28-0)]. This unified formulation is called the generic five-equation model, which can recover existing five-equation models and generate new models. Therefore, it is natural to explore methods for obtaining an interface-sharpening technique that is thermodynamically compatible or completely conservative in terms of mass, momentum, and total energy for a generic five-equation model.

In this study, we first derived a general theoretical formulation of interface-sharpening techniques for the generic five-equation model. The theoretical formulation was conservative in mass, momentum, and total energy. Moreover, it was asymptotically compatible with the thermodynamic mixture laws of the mixture model upon which it was constructed and independent of various specific numerical algorithms. A general numerical strategy called the multimaterial artificial compression method (MMACM) was proposed to numerically solve the theoretical formulation. Numerical tests in oneand two-dimensions were performed to verify the proposed method.

In the future, the proposed interface-sharpening technique can be extended to multidimensional problems beyond the dimensionby-dimension technique to further improve the fidelity of the numerical simulations. Moreover, an extension of the numerical method involving higher order methods is in progress.

### **CRediT authorship contribution statement**

**Zhiwei He:** Writing – review & editing, Writing – original draft, Validation, Software, Resources, Project administration, Methodology, Investigation, Funding acquisition, Formal analysis, Data curation, Conceptualization. **Shuang Tan:** Writing – review & editing, Validation, Investigation, Formal analysis.

### **Declaration of competing interest**

The authors declared no potential conflicts of interest with respect to the research.

#### **Data availability**

No data was used for the research described in the article.

#### **Acknowledgements**

The authors thank the reviewers for their careful reading of the article and their valuable remarks. This work was supported by the Nation Key R&D Program of China (2022YFA1004500) and the National Natural Science Foundation of China (NSFC) under Grant Nos. 12372285 and 12102062.

#### **Appendix A. Pressure-temperature equilibrium case of generic five-equation model**

The four-equation model with kinetic equilibrium (same velocity), mechanical equilibrium (same pressure), and thermal equilibrium (same temperature) [[52–58\]](#page-29-0) involves four partial differential equations (i.e., two mass equations, one mixture momentum equation, and one mixture energy equation) and is fully conservative and hyperbolic. In this section, we show that this is a special case of the generic five-equation model. The additional temperature equilibrium condition implies that one of the partial differential equations is not required. The optimal view is that the four-equation model is obtained by removing the volume fraction equation that is nonconservative [\[3\]](#page-28-0) in the GFE. The details are as follows:

The four-equation model is expressed as

$$\begin{split} &\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0, \\ &\frac{\partial \rho \mathbf{u}}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u} + \rho \mathbf{I}) = 0, \\ &\frac{\partial \rho E}{\partial t} + \nabla \cdot (\rho E \mathbf{u} + \rho \mathbf{u}) = 0, \\ &\frac{\partial \rho Y_1}{\partial t} + \nabla \cdot (\rho u Y_1) = 0, \\ &\dots \\ &\frac{\partial \rho Y_{K-1}}{\partial t} + \nabla \cdot (\rho u Y_{K-1}) = 0, \end{split} \tag{A.1}$$

where the mass fraction or concentration of each material has the following constraint:

$$\sum_{k=1}^{K} Y_k = 1. \tag{A.2}$$

Moreover, this model is usually utilized with the "volume-separated and pressure-temperature-equilibrium" closure (i.e., the materials occupy disjointed volumes at the same temperature and pressure inside the microstructure):

$$\begin{cases} \sum_{k=1}^{K} \frac{Y_k}{\rho_k} = \frac{1}{\rho}, \\ p_1(\rho_1, T) = p, \\ \dots, \\ p_K(\rho_K, T) = p, \\ \sum_{k=1}^{K} Y_k e_k(\rho_k, T) = e. \end{cases}$$
(A.3)

From  $\sum_{k=1}^{K} Y_k e_k(\rho_k, T) = e$  in Eq. (A.3), we have

$$d\sum_{k=1}^{K} Y_k e(\rho_k, T) = de. \tag{A.4}$$

Expanding this equation (Eq. (A.4)), we have

$$\sum_{k=1}^{K} \left( Y_k \left. \frac{\partial e_k}{\partial \rho_k} \right|_T d\rho_k + Y_k \left. \frac{\partial e_k}{\partial T} \right|_{\rho_k} dT + e_k dY_k \right) = de. \tag{A.5}$$

From  $p_k(\rho_k, T) = p$  in Eq. (A.3), we further obtain

$$dp_{\nu}(\rho_{\nu}, T) = dp, \ \forall k. \tag{A.6}$$

Thus, we have

$$\frac{\partial p_k}{\partial \rho_k}\bigg|_T d\rho_k + \frac{\partial p_k}{\partial T}\bigg|_{\rho_k} dT = dp, \ \forall k. \tag{A.7}$$

Therefore, Eqs. (A.5) and (A.7) form a system of K + 1 equations expressed as

$$\begin{cases} \sum_{k=1}^{K} \left( Y_{k} \frac{\partial e_{k}}{\partial \rho_{k}} \Big|_{T} \frac{d\rho_{k}}{dt} + Y_{k} \frac{\partial e_{k}}{\partial T} \Big|_{\rho_{k}} \frac{dT}{dt} + e_{k} \frac{dY_{k}}{dt} \right) = \frac{de}{dt}, \\ \frac{\partial p_{1}}{\partial \rho_{1}} \Big|_{T} \frac{d\rho_{1}}{dt} + \frac{\partial p_{1}}{\partial T} \Big|_{\rho_{1}} \frac{dT}{dt} = \frac{dp}{dt}, \\ \dots \\ \frac{\partial p_{K}}{\partial \rho_{K}} \Big|_{T} \frac{d\rho_{K}}{dt} + \frac{\partial p_{K}}{\partial T} \Big|_{\rho_{K}} \frac{dT}{dt} = \frac{dp}{dt}, \end{cases}$$
(A.8)

with K+2 unknowns  $\left(\frac{d\rho_1}{dt}, \cdots, \frac{d\rho_K}{dt}, \frac{dp}{dt}, \frac{dT}{dt}\right)$ , and  $\frac{dY_k}{dt}$  and  $\frac{de}{dt}$  can be easily obtained from the model (Eq. (A.1)):

$$\frac{dY_k}{dt} = 0,$$

$$\frac{de}{dt} = -\frac{p}{o}\nabla \cdot \mathbf{u}.$$
(A.9)

If we obtain the speed of sound  $c_{eff}$  from the four-equation model, we can obtain  $\frac{dp}{dt}$ , that is,

$$\frac{dp}{dt} = -\rho c_{eff}^2 \nabla \cdot \mathbf{u}. \tag{A.10}$$

By inserting Eqs. (A.10) and (A.9) into Eq. (A.8), the above system (Eq. (A.8)) is closed, and we can easily obtain  $\frac{dT}{dt}$  and  $\frac{d\rho_k}{dt}$  as

$$\frac{dT}{dt} = \frac{1}{\sum_{k=1}^{K} Y_k \mathfrak{B}_k} \left( \left( \sum_{k=1}^{K} Y_k \mathfrak{B}_k \right) \rho c_{eff}^2 - \frac{p}{\rho} \right) \nabla \cdot \mathbf{u}, \tag{A.11}$$

$$\frac{d\rho_k}{dt} = \left( -\left( \mathfrak{A}_k + \mathfrak{D}_k \frac{\sum_{k=1}^K Y_k \mathfrak{B}_k}{\sum_{k=1}^K Y_k \mathfrak{C}_k} \right) \rho c_{eff}^2 + \mathfrak{D}_k \frac{p}{\rho \sum_{k=1}^K Y_k \mathfrak{C}_k} \right) \nabla \cdot \mathbf{u}, \tag{A.12}$$

where

$$\mathfrak{A}_{k} \triangleq \frac{\partial \rho_{k}}{\partial p_{k}}\Big|_{T} = \frac{1}{c_{s,k}^{2}} + \frac{\rho_{k}^{2}\zeta_{k}^{2}C_{p,k}}{T},$$

$$\mathfrak{B}_{k} \triangleq \frac{\frac{\partial \rho_{k}}{\partial \rho_{k}}\Big|_{T}}{\frac{\partial \rho_{k}}{\partial \rho_{k}}\Big|_{T}} = \frac{\partial e_{k}}{\partial p}\Big|_{T} = \frac{p}{(\rho_{k}c_{s,k})^{2}} - \zeta_{k}C_{p,k}\left(1 - \zeta_{k}\frac{p}{T}\right),$$

$$\mathfrak{C}_{k} \triangleq -\mathfrak{B}_{k}\frac{\partial p_{k}}{\partial T}\Big|_{\rho_{k}} + \frac{\partial e_{k}}{\partial T}\Big|_{\rho_{k}} = \frac{\partial e_{k}}{\partial T}\Big|_{\rho} = C_{p,k}\left(1 - \zeta_{k}\frac{p}{T}\right),$$

$$\mathfrak{D}_{k} \triangleq \frac{\frac{\partial p_{k}}{\partial T}\Big|_{\rho_{k}}}{\frac{\partial \rho_{k}}{\partial \rho_{k}}\Big|_{T}} = -\frac{\partial \rho_{k}}{\partial T}\Big|_{p} = \frac{\rho_{k}^{2}\zeta_{k}C_{p,k}}{T}.$$
(A.13)

The specific heat at constant pressure  $C_{p,k}$  and the parameter  $\zeta_k$  introduced by Flåtten et al. [49] for each phase k are respectively defined as

$$C_{p,k} \triangleq T \left. \frac{\partial s_k}{\partial T} \right|_p,$$

$$\zeta_k \triangleq \left. \frac{\partial T}{\partial p} \right|_{s_k}.$$
(A.14)

Differentiating  $\alpha_k \rho_k = \rho Y_k$  yields  $\rho_k d\alpha_k + \alpha_k d\rho_k = d(\rho Y_k)$ , and we obtain

$$\frac{d\alpha_{k}}{dt} = -\frac{\alpha_{k}}{\rho_{k}} \frac{d\rho_{k}}{dt} + \frac{1}{\rho_{k}} \frac{d(\rho Y_{k})}{dt},$$

$$= \alpha_{k} \left( \left( \frac{\mathfrak{A}_{k}}{\rho_{k}} + \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{\sum_{k=1}^{K} Y_{k} \mathfrak{B}_{k}}{\sum_{k=1}^{K} Y_{k} \mathfrak{C}_{k}} \right) \rho c_{eff}^{2} - \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{p}{\rho \sum_{k=1}^{K} Y_{k} \mathfrak{C}_{k}} - 1 \right) \nabla \cdot \mathbf{u}.$$
(A.15)

By inserting Eq. (A.12) and  $\frac{d(\rho Y_k)}{dt}$  (obtained from Eq. (A.1)) into Eq. (A.16), we can obtain

$$\frac{\partial \alpha_k}{\partial t} + \mathbf{u} \cdot \nabla \alpha_k = \alpha_k \left( \left( \frac{\mathfrak{A}_k}{\rho_k} + \frac{\mathfrak{D}_k}{\rho_k} \frac{\sum_{k=1}^K \alpha_k \rho_k \mathfrak{B}_k}{\sum_{k=1}^K \alpha_k \rho_k \mathfrak{C}_k} \right) \rho c_{eff}^2 - \frac{\mathfrak{D}_k}{\rho_k} \frac{p}{\sum_{k=1}^K \alpha_k \rho_k \mathfrak{C}_k} - 1 \right) \nabla \cdot \mathbf{u}. \tag{A.16}$$

Finally, we provide an exact expression for the mixed sound speed of the four-equation model. Such expression has already been proposed in the literature [83,49]

$$\frac{1}{\rho c_{eff}^2} = \sum_{k=1}^K \frac{\alpha_k}{\rho_k c_{s,k}^2} + \frac{1}{T \sum_{k=1}^K \alpha_k \rho_k C_{p,k}} \sum_{k=1}^K \sum_{j>k}^K (\alpha_k \rho_k C_{p,k}) (\alpha_j \rho_j C_{p,j}) (\zeta_j - \zeta_k)^2, \tag{A.17}$$

where  $\zeta_k = \frac{\Gamma_k T}{\rho_k c_{s,k}^2}$  and  $\Gamma_k = \frac{\rho_k}{T} \left. \frac{\partial T}{\partial \rho_k} \right|_{s_k}$  denotes the Grüneisen coefficient. The above equation (Eq. (A.17)) for K=2 can also be written as

$$\frac{1}{c_{eff}^2} = \rho \left( \frac{\alpha_1}{\rho_1 c_{s,1}^2} + \frac{\alpha_2}{\rho_2 c_{s,2}^2} \right) + \rho T \frac{(\alpha_1 \rho_1 C_{p,1})(\alpha_2 \rho_2 C_{p,2})}{\alpha_1 \rho_1 C_{p,1} + \alpha_2 \rho_2 C_{p,2}} \left( \frac{\Gamma_2}{\rho_2 c_{s,2}^2} - \frac{\Gamma_1}{\rho_1 c_{s,1}^2} \right)^2, \tag{A.18}$$

which has been derived by many researchers [83,53].

Therefore, the classical four-equation model implies that the volume fraction of each material evolves according to Eq. (A.16). Hence, this is a special case of the generic five-equation model with a distribution coefficient  $\lambda_k$  defined as

$$\lambda_{k} = \left(\frac{\mathfrak{A}_{k}}{\rho_{k}} + \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{\sum_{k=1}^{K} \alpha_{k} \rho_{k} \mathfrak{B}_{k}}{\sum_{k=1}^{K} \alpha_{k} \rho_{k} \mathfrak{G}_{k}}\right) \rho c_{eff}^{2} - \frac{\mathfrak{D}_{k}}{\rho_{k}} \frac{p}{\sum_{k=1}^{K} \alpha_{k} \rho_{k} \mathfrak{G}_{k}}.$$
(A.19)

Remark 3. For the stiffened gas EOS considered in this study, the closure (i.e., Eq. (A.3)) can be written as

$$\begin{cases} \sum_{k=1}^{K} Y_k / \rho_k = 1/\rho, \\ (\gamma_1 - 1)\rho_1 C_{v,1} T - P_{\infty,1} = p, \\ \dots \\ (\gamma_K - 1)\rho_K C_{v,K} T - P_{\infty,K} = p, \\ \sum_{k=1}^{K} Y_k (C_{v,k} T + P_{\infty,k} / \rho_k + q_k) = e. \end{cases}$$
(A.20)

Using this system and the stiffened gas EOS, we can obtain three important and widely utilized equations as follows:

$$p = \frac{1}{\sum_{k=1}^{K} \frac{\alpha_k}{\gamma_{k} - 1}} \rho\left(e - \sum_{k=1}^{K} Y_k q_k\right) - \frac{\sum_{k=1}^{K} \frac{\alpha_k}{\gamma_{k} - 1} \gamma_k P_{\infty, k}}{\sum_{k=1}^{K} \frac{\alpha_k}{\gamma_{k} - 1}},$$
(A.21)

$$T = \frac{e - \sum_{k=1}^{K} Y_k q_k - \left(\sum_{k=1}^{K} \alpha_k P_{\infty, k}\right) / \rho}{\sum_{k=1}^{K} Y_k C_{v, k}},$$
(A.22)

$$\alpha_{k} = \frac{\gamma_{k} - 1}{\gamma_{k}} \frac{\alpha_{k} \rho_{k} C_{p,k}}{\sum_{k'=1}^{K} \alpha_{k'} \rho_{k'} C_{p,k'}} \frac{\rho e + p - \sum_{k'=1}^{K} \alpha_{k'} \rho_{k'} q_{k'}}{p + P_{\infty,k'}}, \tag{A.23}$$

where  $C_{n,k} = \gamma_k C_{n,k}$ .

#### References

- Z. He, B. Tian, Y. Zhang, F. Gao, Characteristic-based and interface-sharpening algorithm for high-order simulations of immiscible compressible multi-material flows, J. Comput. Phys. 333 (2017) 247–268.
- [2] Z. He, H. Liu, L. Li, Generic five-equation model for compressible multi-material flows and its corresponding high-fidelity numerical algorithms, J. Comput. Phys. 487 (2023) 112154.
- [3] R. Saurel, C. Pantano, Diffuse-interface capturing methods for compressible two-phase flows, Annu. Rev. Fluid Mech. 50 (2018) 105-130.
- [4] V. Maltsev, M. Skote, P. Tsoutsanis, High-order methods for diffuse-interface models in compressible multi-medium flows: a review, Phys. Fluids 34 (2022)
- [5] K. So, X. Hu, N. Adams, Anti-diffusion interface sharpening technique for two-phase compressible flow simulations, J. Comput. Phys. 231 (2012) 4304-4323.
- [6] R.K. Shukla, C. Pantano, J.B. Freund, An interface capturing method for the simulation of multi-phase compressible flows, J. Comput. Phys. 229 (2010) 7411–7439
- [7] R. Shukla, Nonlinear preconditioning for efficient and accurate interface capturing in simulation of multicomponent compressible flows, J. Comput. Phys. 276 (2014) 508–540.
- [8] K.-M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for compressible two-phase flow: the algebraic THINC approach, J. Comput. Phys. 268 (2014) 326–354.
- [9] M. Friess, S. Kokh, Simulation of sharp interface multi-material flows involving an arbitrary number of components through an extended five-equation model, J. Comput. Phys. 273 (2014) 488–519.
- [10] A. Chiapolino, R. Saurel, B. Nkonga, Sharpening diffuse interfaces with compressible fluids on unstructured meshes, J. Comput. Phys. 2017 (2017) 389-417.
- [11] S. Kokh, F. Lagoutière, An anti-diffusive numerical scheme for the simulation of interfaces between compressible fluids by means of a five-equation model, J. Comput. Phys. 229 (2010) 2773–2809.
- [12] X. Deng, S. Inaba, B. Xie, K.-M. Shyue, F. Xiao, High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces, J. Comput. Phys. 371 (2018) 945–966.
- [13] Z.-W. He, B.-L. Tian, L. Li, H.-F. Li, Y.-S. Zhang, B.-Q. Meng, High-order numerical simulation method for compressible multi-material flow problems (in Chinese), Acta Aerodyn. Sin. 39 (2021) 177–190.
- [14] K. So, X. Hu, N. Adams, Anti-diffusion method for interface steepening in two-phase incompressible flow, J. Comput. Phys. 230 (2011) 5155–5177.
- [15] J. Boris, D. Book, Flux-corrected transport. I. SHASTA, a fluid transport algorithm that works, J. Comput. Phys. 11 (1973) 38-69.
- [16] M. Breuß, T. Brox, T. Sonar, J. Weickert, Stabilized nonlinear inverse diffusion for approximating hyperbolic PDEs, scale space and PDE methods in computer vision, Proceedings 3459 (2005) 536–547.
- [17] A. Tiwari, J.B. Freund, C. Pantano, A diffuse interface model with immiscibility preservation, J. Comput. Phys. 252 (2013) 290-309.
- [18] B. Després, F. Lagoutière, Contact discontinuity capturing schemes for linear advection and compressible gas dynamics, J. Sci. Comput. 16 (2001) 479-524.
- [19] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577-616.
- [20] J. Massoni, R. Saurel, B. Nkonga, R. Abgrall, Some models and Eulerian methods for interface problems between compressible fluids with heat transfer (in French), Int. J. Heat Mass Transf. 45 (2002) 1287–1307.
- [21] A. Bernard-Champmartin, F.D. Vuyst, A low diffusive Lagrange-remap scheme for the simulation of violent air-water free-surface flows, J. Comput. Phys. 274 (2014) 19–49.
- [22] E. Olsson, G. Kreiss, A conservative level set method for two phase flow, J. Comput. Phys. 210 (2005) 225-246.
- [23] E. Olsson, G. Kreiss, S. Zahedi, A conservative level set method for two phase flow II, J. Comput. Phys. 225 (2007) 785-807.
- [24] T. Wacławczyk, A consistent solution of the re-initialization equation in the conservative level-set method, J. Comput. Phys. 299 (2015) 487-525.
- [25] R. Chiodi, O. Desjardins, A reformulation of the conservative level set reinitialization equation for accurate and robust simulation of complex multiphase flows, J. Comput. Phys. 343 (2017) 186–200.
- [26] S. Mirjalili, C.B. Ivey, A. Mani, A conservative diffuse interface method for two-phase flows with provable boundedness properties, J. Comput. Phys. 401 (2020) 109006.
- [27] S.S. Jain, Accurate conservative phase-field method for simulation of two-phase flows, J. Comput. Phys. 469 (2022) 111529.
- [28] S. Parameswaran, J. Mandal, A stable interface-preserving reinitialization equation for conservative level set method, Eur. J. Mech. B, Fluids 98 (2023) 40–63.
- [29] A. Kapila, R. Menikoff, J. Bdzil, S. Son, D. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: reduced equations, Phys Fluids 13 (2001) 3002.
- [30] Y. Sun, C. Beckermann, Sharp interface tracking using the phase-field equation, J. Comput. Phys. 220 (2007) 626-653.
- [31] P.H. Chiu, Y.T. Lin, A conservative phase field method for solving incompressible two-phase flows, J. Comput. Phys. 230 (2011) 185–204.
- [32] S.S. Jain, A. Mani, P. Moin, A conservative diffuse-interface method for compressible two-phase flows, J. Comput. Phys. 418 (2020) 109606.
- [33] Z. Huang, E. Johnsen, A consistent and conservative Phase-Field method for compressible multiphase flows with shocks, J. Comput. Phys. 488 (2023) 112195.
- [34] R.J. LeVeque, Wave propagation algorithms for multidimensional hyperbolic systems, J. Comput. Phys. 131 (1997) 327-353.
- [35] D.I. Ketcheson, M. Parsani, R. LeVeque, High-order wave propagation algorithms for hyperbolic systems, SIAM J. Sci. Comput. 35 (2013) A351-A377.
- [36] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, Int. J. Numer. Methods Fluids 48 (2005) 1023–1040.
- [37] Z. Sun, S. Inaba, F. Xiao, Boundary Variation Diminishing (BVD) reconstruction: a new approach to improve Godunov schemes, J. Comput. Phys. 322 (2016)
- [38] B. van Leer, Towards the ultimate conservative difference scheme. V. A second order sequel to Godunov's method, J. Comput. Phys. 32 (1979) 101-136.
- [39] A. Harten, The artificial compression method for computation of shocks and contact discontinuities. I. Single conservation laws, Commun. Pure Appl. Math. 30 (1977) 611–638.

- [40] K. Lie, S. Noelle, One the artificial compression method for second-order [nonoscillatory](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibB86B2DD49B7984972B896682A969E030s1) central difference schemes for systems of conservation laws, SIAM J. Sci. Comput. 24 (2003) [1157–1174.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibB86B2DD49B7984972B896682A969E030s1)
- [41] A. Harten, The artificial compression method for computation of shocks and contact [discontinuities:](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib5C0D8E27B049584C6E402F064263BFBDs1) III. Self-adjusting hybrid schemes, Math. Comput. 32 (1978) [363–389.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib5C0D8E27B049584C6E402F064263BFBDs1)
- [42] A. Harten, High resolution schemes for hyperbolic [conservation](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib2C19D691FF7B8FA46CFB6739AE879E96s1) laws, J. Comput. Phys. 49 (1983) 357–393.
- [43] H. Yang, An artificial compression method for ENO schemes: the slope [modification](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib06D6B2C69B689567CC3195E3E695A700s1) method, J. Comput. Phys. 89 (1990) 125–160.
- [44] A. Murrone, H. Guillard, A five equation reduced model for [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib985E3AB4BCAE89D7F21933BF23F8B5E8s1) two phase flow problems, J. Comput. Phys. 202 (2005) 664–698.
- [45] F. Harlow, A. Amsden, Fluid Dynamics, Technical Report LA-4700, Los Alamos National [Laboratory,](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib7D48FFA36DD441EB5879C27EC537B797s1) 1971.
- [46] E. Han, M. Hantke, S. Müller, Efficient and robust relaxation procedures for [multi-component](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibD1CE6D348C0F922252C4051DD6F64618s1) mixtures including phase transition, J. Comput. Phys. 338 (2017) [217–239.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibD1CE6D348C0F922252C4051DD6F64618s1)
- [47] Z. He, Y. Ruan, Y. Yu, B. Tian, X. Feng, Self-adjusting [steepness-based](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibEA29330D2C029030E49BE77CF0D0FF80s1) schemes that preserve discontinuous structures in compressible flows, J. Comput. Phys. 463 (2022) [111268.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibEA29330D2C029030E49BE77CF0D0FF80s1)
- [48] Y. Ruan, B. Tian, X. Zhang, Z. He, On the supremum of the steepness parameter in self-adjusting [discontinuity-preserving](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib6EE7781A48E229CB1DCE663DFCE2039Bs1) schemes, Comput. Fluids 245 (2022) [105588.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib6EE7781A48E229CB1DCE663DFCE2039Bs1)
- [49] T. Flåtten, A. Morin, S.T. Munkejord, Wave propagation in [multicomponent](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib47B1709262794FC4D6BA5EA5349F3F56s1) flow models, SIAM J. Appl. Math. 70 (8) (2010) 2861–2882.
- [50] G.H. Miller, E.G. Puckett, A [high-order](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib243C61081B96CD5E1C8BEEF954D512CAs1) Godunov method for multiple condensed phases, J. Comput. Phys. 128 (1996) 134–164.
- [51] J. Kreeft, B. Koren, A new formulation of Kapila's [five-equation](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib351CA4DADE288EC22CC75471E0DE7A13s1) model for compressible two-fluid flow, and its numerical treatment, J. Comput. Phys. 229 (2010) [6220–6242.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib351CA4DADE288EC22CC75471E0DE7A13s1)
- [52] S. Le Martelot, S. Saurel, B. Nkonga, Towards the direct numerical [simulation](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib2557FD599B221FC269BAC28C5CC224BFs1) of nucleate boiling flows, Int. J. Multiph. Flow 66 (2014) 62–78.
- [53] A.D. Demou, N. Scapin, M. Pelanti, L. Brandt, A [pressure-based](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib70321DB7FF8B5CFF3C2187F8CD006629s1) diffuse interface method for low-Mach multiphase flows with mass transfer, J. Comput. Phys. 448 (2022) [110730.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib70321DB7FF8B5CFF3C2187F8CD006629s1)
- [54] R. Saurel, P. Boivin, O. Le Métayer, A general formulation for cavitating, boiling and [evaporating](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibFD350070C3F499011A9A8D222BA28B00s1) flows, Comput. Fluids 128 (2016) 53–64.
- [55] M. Pelanti, [Arbitrary-rate](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibD7319D8107C11ACC04BC0E07C7127522s1) relaxation techniques for the numerical modeling of compressible two-phase flows with heat and mass transfer, Int. J. Multiph. Flow 153 (2022) [104097.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibD7319D8107C11ACC04BC0E07C7127522s1)
- [56] R.J.R. Williams, [Fully-conservative](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibE96A5A98F18E68C9717F25A76F75AC00s1) contact-capturing schemes for multi-material advection, J. Comput. Phys. 398 (2019) 108809.
- [57] D.W. Schwendeman, A.K. Kapila, W.D. Henshaw, A comparative study of two macro-scale models of [condensed-phase](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibC669287B400A8B7D719DBA4C45E0EDADs1) explosives, IMA J. Appl. Math. 77 (2012) [2–17.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibC669287B400A8B7D719DBA4C45E0EDADs1)
- [58] G. Linga, T. Flåtten, A hierarchy of [non-equilibrium](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib1446AAE645EBE0CC051A4C75B94E2FB3s1) two-phase flow models, ESAIM Proc. Surv. 66 (2019) 109–143.
- [59] R. Abgrall, How to prevent pressure oscillations in [multicomponent](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib5A04FA06C201A121E8828527D7DD8636s1) flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160.
- [60] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible [multicomponent](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib45C96EB5442D55F0F5FBA764E994D265s1) flow problems, J. Comput. Phys. 219 (2006) 715–732.
- [61] C. Zhang, I. Menshov, L. Wang, Z. Shen, Diffuse interface relaxation model for two-phase [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibB5DF4759F3FDA6691B1DF614624AE7FBs1) flows with diffusion processes, J. Comput. Phys. 466 (2022) [111356.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibB5DF4759F3FDA6691B1DF614624AE7FBs1)
- [62] S.H. Christiansen, H.Z. Munthe-Kaas, B. Owren, Topics in [structure-preserving](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib1C96558AFA1170506003086C60A66004s1) discretization, Acta Numer. 20 (2011) 1–119.
- [63] B. Koren, R. Abgrall, P. Bochev, J. Frank, B. Perot, [Physics-compatible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib8D92A6DC6F19C1EC54A774E487FCA621s1) numerical methods, J. Comput. Phys. 257 (2014) 1039.
- [64] B. Larrouturou, How to preserve the mass fractions positivity when computing compressible [multi-component](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibBD73B8C02D95BC994D9FB40E3107C425s1) flows, J. Comput. Phys. 95 (1991) 59–84.
- [65] Z.-W. He, L. Li, Y.-S. Zhang, B.-L. Tian, Consistent [implementation](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib617D748EA88FCD5CCE5E1625154D4341s1) of characteristic flux-split based finite difference method for compressible multi-material gas flows, Comput. Fluids 168 (2018) [190–200.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib617D748EA88FCD5CCE5E1625154D4341s1)
- [66] S. Alahyari Beig, E. Johnsen, Maintaining interface equilibrium conditions in [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibF13D4401DE47C3E2699324DEC8D29FF9s1) multiphase flows using interface capturing, J. Comput. Phys. 302 (2015) [548–566.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibF13D4401DE47C3E2699324DEC8D29FF9s1)
- [67] Z.-W. He, Y.-S. Zhang, X.-L. Li, B.-L. Tian, Preventing numerical oscillations in the flux-split based finite difference method for [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibC7D8B28F4ADB5FCE7D41F6EDFD4AAF65s1) flows with discontinuities, J. Comput. Phys. 300 (2015) [269–287.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibC7D8B28F4ADB5FCE7D41F6EDFD4AAF65s1)
- [68] Z.-W. He, Y.-S. Zhang, X.-L. Li, L. Li, B.-L. Tian, Preventing numerical oscillations in the flux-split based finite difference method for [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib79D9E1AFE4E52DB80EC08FAFC23D2734s1) flows with [discontinuities,](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib79D9E1AFE4E52DB80EC08FAFC23D2734s1) II, Int. J. Numer. Methods Fluids 80 (2016) 306–316.
- [69] M. Berger, A. Giuliani, A state [redistribution](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib3ABFD44D346CDF243AB436EB617AE681s1) algorithm for finite volume schemes on cut cell meshes, J. Comput. Phys. 428 (2021) 109820.
- [70] W. Ni, Q. Zeng, Y. Ruan, Z. He, A novel [steepness-adjustable](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib031B246142575A1ACD0185A1C8D2E6A7s1) harmonic volume-of-fluid method for interface capturing, J. Comput. Phys. 501 (2024) 112765.
- [71] J.E. Pilliod Jr., E.G. Puckett, Second-order accurate [volume-of-fluid](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib15B37A6448FF81F7DF69E87C85A66A87s1) algorithms for tracking material interfaces, J. Comput. Phys. 199 (2004) 465–502.
- [72] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong [stability-preserving](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibCB2754C01B5559FDF4FDDCA6ACC0B8A8s1) high-order time discretization methods, SIAM Rev. 43 (2001) 89–112.
- [73] S. Gottlieb, C.-W. Shu, Total variation diminishing [Runge-Kutta](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibB1BF1D10F04F31FDD713A2035E1CE405s1) schemes, Math. Comput. 67 (1998) 73–85.
- [74] R.J. LeVeque, Finite Volume Methods for [Hyperbolic](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibE04C520E50C2890192F33EC9B6E4F24Cs1) Problems, Cambridge University Press, 2002.
- [75] E.F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the [HLL-Riemann](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib07CA4E9D3C6CB6A3CB28E786BFE20E1Ds1) solver, Shock Waves 4 (1994) 25–34.
- [76] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, third edition, [Springer-Verlag,](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib59E9E1102F35CA34EBE106A719516694s1) 2009.
- [77] Z.-W. He, Y.-S. Zhang, F.-J. Gao, X.-L. Li, B.-L. Tian, An improved accurate [monotonicity-preserving](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibCAA9F4FE5F8A24959DDC1D57F573F386s1) scheme for the Euler equations, Comput. Fluids 140 (2016) [1–10.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibCAA9F4FE5F8A24959DDC1D57F573F386s1)
- [78] H. Tan, T. Liu, A note on the [conservative](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib340837106A37EDE03434356BB9CC3360s1) schemes for the Euler equations, J. Comput. Phys. 218 (2006) 451–459.
- [79] K. Schmidmayer, S.H. Bryngelson, T. Colonius, An assessment of [multicomponent](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib502960325AFCA7C0BD6C434C31B68E72s1) flow models and interface capturing schemes for spherical bubble dynamics, J. [Comput.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib502960325AFCA7C0BD6C434C31B68E72s1) Phys. 402 (2020) 109080.
- [80] J.-Y. Lin, Y. Shen, H. Ding, N.-S. Liu, X.-Y. Lu, Simulation of [compressible](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib6E869FBF48ADA74DEDB27B22AEBF19A0s1) two-phase flows with topology change of fluid-fluid interface by a robust cut-cell method, J. Comput. Phys. 328 (2017) [140–159.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib6E869FBF48ADA74DEDB27B22AEBF19A0s1)
- [81] J. Haas, B. Sturtevant, Interaction of a weak shock wave with cylindrical and spherical gas [inhomogeneities,](http://refhub.elsevier.com/S0021-9991(24)00441-8/bib4D48A1ABEEE87A3430B66023FCE774C1s1) J. Fluid Mech. 390 (1987) 41–76.
- [82] R.R. Nourgaliev, T.N. Dinh, T.G. Theofanous, Adaptive [characteristics-based](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibA20DD85C294F81544E49DEE29162B888s1) matching for compressible multifluid dynamics, J. Comput. Phys. 213 (2006) [500–529.](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibA20DD85C294F81544E49DEE29162B888s1)
- [83] T. Flåtten, H. Lund, Relaxation two-phase flow models and the [subcharacteristic](http://refhub.elsevier.com/S0021-9991(24)00441-8/bibCF8D2CD8D1856C2418ED0C5AB1F56688s1) condition, Math. Models Methods Appl. Sci. 21 (12) (2011) 2379–2407.