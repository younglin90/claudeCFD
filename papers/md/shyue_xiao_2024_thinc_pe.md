Journal of Computational Physics 513 (2024) 113192

Available online 12 June 2024 0021-9991/© 2024 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies.


### Contents lists available at ScienceDirect


# Journal of Computational Physics


### journal homepage: www.elsevier.com/locate/jcp


# On immiscibility preservation conditions of material interfaces in the generic five-equation model


# Zhiwei He a,b, Shuang Tan a,b,∗

a Institute of Applied Physics and Computational Mathematics, Beijing 100094, China b National Key Laboratory of Computational Physics, Beijing 100088, China


## A R T I C L E I N F O A B S T R A C T

Keywords: Interface-sharpening technique Diffuse interface method Consistent and conservative schemes Compressible multimaterial flows Generic five-equation model

Interfaces separating pure materials and mixtures tend to be severely smeared with interfacecapturing methods for compressible multi-material flows, necessitating the requirement of various interface-sharpening techniques. However, these techniques have various problems related to consistency, conservation, and thermodynamic compatibility. In this work, we derive a general theoretical formulation of interface-sharpening techniques for the generic five-equation model. This theoretical formulation is not only conservative in mass, momentum, and total energy but also asymptotically compatible with the thermodynamic mixture laws of the mixture model upon which it is constructed, and is independent of various specific numerical algorithms. We further propose a general numerical method to solve this theoretical formulation. The proposed method is consistent and conservative, and it prevents spurious errors at the interfaces. Examples of oneand two-dimensional multimaterial compressible flow problems, including shocks and interfaces, are considered to verify the analysis and demonstrate the efficiency of the method.


### 1. Introduction

Numerous natural and industrial processes involve compressible flows with distinct material interfaces. Typical applications include underwater bubble dynamics, cavitation flows, inertial confinement fusion, Rayleigh-Taylor instabilities, and RichtmyerMeshkov instabilities. Thus, investigating these flow mechanisms through numerical modeling and simulation of such flows is critical [1,2].

One of the most important aspects of numerical modeling multimaterial flows is the method employed to describe the movement of the material interface. One method is to consider interfaces as numerically diffused zones of artificial mixtures using a color function that acquires different values for each fluid and assists in identifying the interface. This method is the so-called diffuse interface method (DIM), which can be further categorized into two subclasses: multicomponentand multiphase-based DIM [2]. A more detailed description of these subclasses can be found in our previous study [2]. Additional details can be found in [3,4] and the references therein.

A common problem with DIM is that numerical implementations utilizing shock-/interface-capturing methods tend to produce solutions that exhibit excessive numerical diffusion. Minimizing numerical smearing within DIM has recently become an active topic, and interface-sharpening techniques/methods for various five-equation-type models [5–12,1] have been proposed.

* Corresponding author at: Institute of Applied Physics and Computational Mathematics, Beijing 100094, China. E-mail addresses: he_zhiwei@iapcm.ac.cn (Z. He), tan_shuang@iapcm.ac.cn (S. Tan).

https://doi.org/10.1016/j.jcp.2024.113192 Received 30 October 2023; Received in revised form 23 May 2024; Accepted 6 June 2024

Journal of Computational Physics 513 (2024) 113192

2

Z. He and S. Tan

These techniques/methods can be roughly divided into at least five categories [13]. (1) Anti-diffusion method. The basic idea of this method is to directly solve the anti-diffusion equation to sharpen the interface [5]. A special discretization scheme [14–16] is employed to ensure numerical stability and volume fraction boundedness when solving the anti-diffusion equation. This method is essentially a flux modification or flux-corrected transport (FCT)-based technique [15]. It is intricately and inevitably tied to the underlying numerical scheme; thus, it is difficult to generalize to different discretizations, such as an increase in the order of accuracy [17]. Moreover, it is unknown whether these fluxes satisfy the compatibility between the equations. Numerical oscillations have been shown to exist at the interfaces [5], and there is a risk that anti-diffusive fluxes can oversharpen the interface in flow regions already drawn thin by the resolved strain field [17]. (2) Limited downwind Lagrange-remap method. The basic idea of this method is to employ the downwind scheme as much as possible on the basis of satisfying the total variation diminishing (TVD) stability theory [18]. This method has been extended to a Lagrange-remap scheme [11,9], and the resulting limited downwind Lagrange-remap scheme [11,9] is an operator split scheme comprising two steps: (a) the Lagrange step, in which the equation of the Allaire-Massoni model [19,20] is advanced to a new time on a grid that moves with the fluid, and (b) the remap step, in which the solution is remapped onto the original mesh via advection over a pseudo-time step. While the Lagrange step is standard, the remap step is built with the limited downwind scheme [18] to ensure two types of features [11,9]: (i) it provides some consistency and stability properties for the scheme and (ii) it minimizes the diffusion of the variables that are utilized to locate the interface. The limited downwind fluxes in the remap stage for the other equations are designed in a manner similar to the anti-diffusion method to preserve the consistency between the volume fraction equation and the other equations [5]. The final limited downwind Lagrange-remap method generates impressive numerical results [11,9]. However, this method is confined to the Lagrange-remap method and has shortcomings in the appearance and fragmentation of fine structures (filaments and high-frequency instabilities) [21]. (3) Interface renormalization method. This method relies on the artificial movement of the interface, as the diffusive and sharpening (compressive) fluxes can balance each other, ensuring convergence to a particular profile of interfaces. The conservative level set (CLS) and phase-field (PF) methods are typical. To maintain a consistently sharp interface in incompressible flows, Olsson et al. [22,23] first developed the CLS method, which has garnered significant interest [6,7,24–28]. This method is a two-step advection/artificial-compression interface-sharpening algorithm that was reformulated and extended [6,7] to compressible two-material flows governed by the Allaire-Massoni model [19,20]. A standard second-order center scheme was proposed to solve this type of interface-sharpening technique [6,17]. However, this method led to qualitatively incorrect results due to thermodynamic inconsistencies [17]. Tiwari et al. [17] theoretically proposed the immiscibility preservation conditions for the five-equation model proposed by Kapila et al. [29]. Nevertheless, their results are non-conservative, even for mass equations. The PF methods are based on the Cahn-Hilliard and Allen-Cahn equations, which were originally developed to model phase separation and coarsening phenomena in solids and the motion of antiphase boundaries in crystalline solids, respectively [27]. Recently, these methods have been adopted to model the interface between two fluids [30,31,26]. The Cahn-Hilliard PF model is conservative but involves a fourth-order spatial derivative in the equation, which requires careful construction of the numerical methods [26]. By contrast, the Allen-Cahn PF model does not involve fourth-order derivatives in the equation; however, it is not conservative [26]. Starting from the Allen-Cahn equation, Sun and Beckerman [30] employed the hyperbolic tangent equilibrium profile to introduce a term that canceled the curvature-driven incompressible flow. Inspired by the conservative level set of Olsson and Kreiss [22], Chiu and Lin [31] reformulated this equation in a conservative form. Jain et al. [32,27] extended the aforementioned result to a compressible two-material flow governed by the Allaire-Massoni model [19,20]. However, the utilization of 𝜌𝑘0 (i.e., the density of phase 𝑘within the incompressible limit) is not apparent in compressible flows. Recently, Huang and Johnsen [33] proposed another interface-sharpening algorithm based on the PF methods. (4) Modified reconstruction method. Shyue and Xiao [8] proposed a hybrid method in which a low-order semi-discrete wave propagation method [34,35] was utilized in single-material regions, and a tangent of hyperbola for interface capturing (THINC) scheme [36] was utilized in the interface zone. Recently, Deng et al. [12] proposed utilizing a boundary variation diminishing (BVD) framework [37] to realize the above hybridization. Chiapolino et al. [10] proposed a specific limiter and inserted it into conventional MUSCL-type schemes [38] to significantly improve the resolution of the interfaces. For these types of methods, the modified reconstructed schemes are often anti-diffusive; it is not certain whether these methods are capable of maintaining consistency between equations [8,13]. Furthermore, it was also found [1] that the diffusion of the interface cannot be completely controlled by modifying only the reconstruction schemes, and that the approximate Riemann solvers with different dissipative properties still have significant effects. (5) Artificial compression method. Harten [39] first proposed the artificial compression method (ACM) to enhance the resolution of contact discontinuities in the context of the Euler equations. The essence of ACM is to solve an original equation with an added term (called an artificial compression flux, which can be designed theoretically or numerically [39,40]) so that the numerical characteristics slightly converge toward the contact discontinuity (instead of being in parallel or diverging from the contact discontinuity) to maintain its sharpness [39,41,42]. Yang [43] proposed another simple ACM for higher-order finite-volume ENO schemes via slope modification. In our previous study [1], we extended Yang’s method to compressible multimaterial flows and found that the numerical diffusion cannot be completely controlled solely by pure slope modification, and the approximate Riemann solvers with different dissipative properties also have significant effects. Moreover, the employment of the immiscibility preservation conditions [17] to maintain consistency among equations led this method to be non-conservative.

Currently, all these methods encounter a common and still unsolved problem: there is no well-established means for sharpening other equations (mass, momentum, and total energy equations) when the equation describing the interface evolution (such as the volume fraction equation) adopts the above interface-sharpening methods. This problem is very important for compressible multimaterial flows because (1) For incompressible multimaterial flows with moving interfaces, the density and other physical properties, such as viscosity and thermal conductivity, are constant in each fluid [12]. However, a substantial barrier exists when compressible multimaterial flows are considered. For compressible multimaterial flows, the volume fraction of each material is no longer

Journal of Computational Physics 513 (2024) 113192

3

Z. He and S. Tan

conservative, and the phasic density is no longer constant. The density and energy in a compressible flow must be solved along with the volume fraction. Moreover, special formulations are required to maintain physical consistency, resulting in a balanced state among all variables for a well-defined interface cell [12]. (2) If the consistency between the equations is broken, the result worsens. Moreover, disunity phenomena exist in the physical models used in these studies. Some researchers [6,32] utilized the five-equation model proposed by Allaire et al. [19] and Massoni et al. [20], whereas others utilized the model proposed by Kapila et al. [29,44].

It is difficult but important to systematically investigate the immiscibility preservation conditions for compressible multimaterial interfacial flows governed by various five-equation models. There are two difficulties. The first is the disunity phenomena that exist in the models, and the second is how to obtain an interface-sharpening technology that is thermodynamically compatible and completely conservative in terms of mass, momentum, and total energy. In our previous study [2], a unified formulation, including a distribution coefficient to characterize the interactions between materials, was theoretically derived and called a generic fiveequation model (GFE) [2]. This model is general in that it can recover two specific five-equation models [19,20,29,44] and generate new models. A more detailed description can be found in our previous study [2].

Therefore, it is natural to explore interface-sharpening techniques for the generic five-equation model. In this study, we derive a general theoretical formulation of interface-sharpening techniques for the generic five-equation model. The theoretical formulation is not only conservative in mass, momentum, and total energy, but it is also asymptotically compatible with the thermodynamic mixture laws of the mixture model upon which it is constructed and independent of various specific numerical algorithms. A general numerical strategy called the multimaterial artificial compression method (MMACM) is further proposed to numerically solve the theoretical formulation. Examples of oneand two-dimensional multimaterial compressible flow problems, including shocks and interfaces, are considered to verify the analysis and demonstrate the efficiency of the method.

The remainder of this paper is organized as follows. In Section 2, we review some of the basic aspects of the generic five-equation model and introduce the topic of the present study. In Section 3, a general theoretical formulation of the interface-sharpening techniques for the generic five-equation model is presented in detail. In Section 4, we propose a high-fidelity numerical algorithm to solve the theoretical formulation numerically. In Section 5, examples of multimaterial compressible flow problems, including shocks and interfaces, are presented to verify the analysis and demonstrate the efficiency of the method. Finally, the conclusions are presented in Section 6.


### 2. Topic of present work


### In this section, we summarize the physical model utilized in this paper, and discuss the topic of the present work.


### 2.1. Generic five-equation model

In the literature, there are two typical five-equation models: the Allaire-Massoni model [19,20] and the Kapila model [29,44]. Different researchers have employed different models, leading to disunity phenomena. Recently, we [2] derived a unified formulation for the five-equation model, establishing a GFE model. This GFE model can recover existing typical five-equation models [19,20,29, 44] and also generate new models [2]. In this study, we consider the GFE model [2] for compressible multimaterial hydrodynamics with an arbitrary number of materials 𝐾, which can be given by


## 𝜕(𝛼1𝜌1 )


## 𝜕𝑡 + ∇⋅(𝛼1𝜌1u) = 0,


## ⋯


## 𝜕 ( 𝛼𝐾𝜌𝐾 )


## 𝜕𝑡 + ∇⋅(𝛼𝐾𝜌𝐾u) = 0,


## 𝜕(𝜌u)


## 𝜕𝑡 + ∇⋅(𝜌u ⊗u) + ∇𝑝= 0,


## 𝜕(𝜌𝐸)


## 𝜕𝑡 + ∇⋅(𝜌𝐸u) + ∇⋅(𝑝u) = 0,


## 𝜕𝛼1


## 𝜕𝑡+ u ⋅∇𝛼1 = 𝛼1(𝜆1 −1)∇⋅u,


## ⋯


## 𝜕𝛼𝐾−1


## 𝜕𝑡 + u ⋅∇𝛼𝐾−1 = 𝛼𝐾−1(𝜆𝐾−1 −1)∇⋅u,


### (1)


## where 𝛼𝑘denotes the volume fraction of the 𝑘-th material with the saturation condition ∑𝐾 𝑘=1 𝛼𝑘= 1. The density 𝜌, pressure 𝑝, and total energy 𝐸can be expressed as


## 𝜌= ∑

𝑘 𝛼𝑘𝜌𝑘,𝑝= ∑

𝑘 𝛼𝑘𝑝𝑘,𝜌𝐸= ∑

𝑘 𝛼𝑘𝜌𝑘𝑒𝑘+ 1


## 2𝜌u ⋅u, (2)

where 𝜌𝑘, 𝑝𝑘, and 𝑒𝑘represent the density, pressure, and specific internal energy of the 𝑘-th material, respectively. These equations are supplemented by the isobaric closure law 𝜌𝑒 = ∑𝐾 𝑘=1 𝛼𝑘𝜌𝑘𝑒𝑘(𝜌𝑘, 𝑝), leveraging the equations of state (EOSs) of each material.

Journal of Computational Physics 513 (2024) 113192

4

Z. He and S. Tan


> **Table 1 Special cases of distribution coefficients 𝜆𝑘(𝑘 = 1, ⋯ , 𝐾). Symmetry means that all materials are treated in the same manner, while the absence of symmetry denotes that the importance of material is heterogeneous in interaction. For more details, please see ref. [2].**

Assumption Expression of 𝜆𝑘 Symmetry

equal compressibility 𝜆𝑘= 1 Yes

isentropic 𝜆𝑘= 1 𝜌𝑘𝑐2 𝑠,𝑘 ∑𝐾 𝑘′ =1 𝛼𝑘′

𝜌𝑘′ 𝑐2 𝑠,𝑘′ Yes

equal velocity variation 𝜆𝑘= 1 𝑐𝑠,𝑘 ∑𝐾 𝑘′=1 𝛼𝑘′ 𝑐𝑠,𝑘′ Yes

frozen flow(𝐾= 2) 𝜆1 =


$$
∑2
𝑠,𝑘(𝜉−𝜉𝑘)
$$

𝛼1 ( 𝜌1𝜉1𝑐2 𝑠,1−𝜌2𝜉2𝑐2 𝑠,2

) + 1 No with 𝜆2 = 1−𝛼1𝜆1

𝛼2

stratified flow(𝐾= 2) 𝜆1 =

−∑2 𝑘=1 𝛼𝑘𝜌𝑘𝜉𝑘𝑐2 𝑠,𝑘+ 𝜌𝜉𝑐2 𝑠,1𝑐2 𝑠,2 (𝛼1𝜌2+𝛼2𝜌1 )

𝛼1𝜌2𝑐2 𝑠,2+𝛼2𝜌1𝑐2 𝑠,1

𝛼1 ( 𝜌1𝜉1𝑐2 𝑠,1−𝜌2𝜉2𝑐2 𝑠,2

) + 1 No with 𝜆2 = 1−𝛼1𝜆1

𝛼2

In this study, if not specifically mentioned, we chose the stiffened-gas EOS introduced by Harlow and Amsden [45]. This EOS is a combination of the perfect gas law and barotropic Tait equation, supplemented with an appropriate energy law [46]. The corresponding thermal and calorific EOS are


## 𝑝𝑘(𝜌𝑘,𝑒𝑘) = (𝛾𝑘−1)𝜌𝑘(𝑒𝑘−𝑞𝑘) −𝛾𝑘𝑃∞,𝑘, (3)


## 𝑇𝑘(𝜌𝑘,𝑒𝑘) = (𝑒𝑘−𝑞𝑘−𝑃∞,𝑘∕𝜌𝑘)∕𝐶𝑣,𝑘, (4)

where 𝑇𝑘denotes the temperature and the material parameters are the ratio of specific heat 𝛾𝑘, specific heat at constant volume 𝐶𝑣,𝑘, minimal pressure 𝑃∞,𝑘, and heat of formation 𝑞𝑘[46]. The distribution coefficients 𝜆𝑘(𝑘 = 1, ⋯ , 𝐾), which determine the specific path along which materials evolve, are problem-dependent and require physical knowledge [2]. However, certain special cases can be mathematically derived, as listed in Table 1. Finally, the corresponding mixture sound speed 𝑐𝑒𝑓𝑓of this model is [2]


## 𝜌𝑐2 𝑒𝑓𝑓=

𝐾 ∑

𝑘=1


## (𝜆𝑘𝛼𝑘𝜌𝑘𝜉𝑘


## 𝜉


## ) 𝑐2 𝑠,𝑘, (5)


### where


## 𝜉𝑘≜𝜕𝜌𝑘𝑒𝑘


## 𝜕𝑝𝑘


## ||||𝜌𝑘 ,


## 𝜉≜

𝐾 ∑

𝑘=1 𝛼𝑘𝜉𝑘,


## 𝑐2 𝑠,𝑘≜𝜕𝑝


## 𝜕𝜌𝑘


## ||||𝑠𝑘 = 𝑝


## 𝜌2 𝑘


## 𝜕𝑝 𝜕𝑒𝑘


## ||||𝜌𝑘 + 𝜕𝑝


## 𝜕𝜌𝑘


## ||||𝑒𝑘 .


### (6)


### 2.2. Interface sharpening techniques to preserve immiscibility conditions


### This study focuses on constructing interface-sharpening techniques for the GFE model. Specifically, we investigate the following two problems in detail.

(I) What terms should be added to each equation in the GFE model such that the consistency among the equations (of mass, momentum, total energy, and volume fraction), conservation (of mass, momentum, and total energy), and more importantly, thermodynamic compatibility can be preserved? That is, in the following model,


## 𝜕(𝛼1𝜌1 )


## 𝜕𝑡 + ∇⋅(𝛼1𝜌1u) = 𝑀1, (7)


## ⋯ (8)


## 𝜕(𝛼𝐾𝜌𝐾 )


## 𝜕𝑡 + ∇⋅(𝛼𝐾𝜌𝐾u) = 𝑀𝐾, (9)


## 𝜕(𝜌u)


## 𝜕𝑡 + ∇⋅(𝜌u ⊗u) + ∇𝑝= P, (10)


## 𝜕(𝜌𝐸)


## 𝜕𝑡 + ∇⋅(𝜌𝐸u) + ∇⋅(𝑝u) = Θ, (11)


## 𝜕𝛼1


## 𝜕𝑡+ u ⋅∇𝛼1 = 𝛼1(𝜆1 −1)∇⋅u + 𝐴1, (12)


## ⋯ (13)

Journal of Computational Physics 513 (2024) 113192

5

Z. He and S. Tan


## 𝜕𝛼𝐾−1


## 𝜕𝑡 + u ⋅∇𝛼𝐾−1 = 𝛼𝐾−1(𝜆𝐾−1 −1)∇⋅u + 𝐴𝐾−1, (14)


## what are the general constraints between 𝑀𝑘, P, Θ, and 𝐴𝑘?

In this study, we theoretically and concretely deduce the general constraints on 𝑀𝑘, P, Θ, and 𝐴𝑘. The final result forms a generic five-equation model that includes the interface-sharpening effect, which not only conserves mass, momentum, and total energy exactly but is also compatible with thermodynamics.

(II) After obtaining the model (Eqs. (7)-(14)), the next problem is obtaining theoretically or numerically specific expressions of these added terms, especially the term added to the volume fraction equations.

In this work, we utilize the original ACM framework [39] and set 𝐴𝑘as the difference between the limited downwind flux (modified with the steepness-adjustable harmonic limiter [47,48]) and the traditional upwind numerical flux provided by various Riemann solvers. Coupled with the generic five-equation model that includes the interface-sharpening effect (Eqs. (7)-(14)), we propose a refined multimaterial artificial compression method (MMACM) for compressible multimaterial flows.


### These two topics form the subject of this study, and the details are described in the following sections.


### 3. Modeling

In this section, we theoretically and concretely deduce the general constraints of 𝑀𝑘, P, Θ, and 𝐴𝑘. The final result is an extended GFE model that includes the interface-sharpening effect, which not only exactly conserves mass, momentum, and total energy but is also compatible with thermodynamics.


### 3.1. Mixture entropy of generic five-equation model


## The mixture entropy 𝑠is defined as 𝑠 = ∑𝐾 𝑘=1 𝑌𝑘𝑠𝑘, where 𝑌𝑘= 𝛼𝑘𝜌𝑘∕𝜌is the mass fraction (or concentration) of the 𝑘-th material. In this section, we analyze the mixture entropy 𝑠in the GFE model.


### From the equation of the total energy of the GFE model, we have [2]


## 𝑑𝜌𝑒


## 𝑑𝑡+ (𝜌𝑒+ 𝑝)∇⋅u = 0, (15)


### where 𝑑


### 𝑑𝑡denotes the Lagrangian total derivative 𝑑


## 𝑑𝑡= 𝜕


## 𝜕𝑡+ u ⋅∇. Eq. (15) can be further expanded to

𝐾 ∑

𝑘=1 𝛼𝑘𝜌𝑘 𝑑𝑒𝑘


## 𝑑𝑡+

𝐾 ∑

𝑘=1 𝑒𝑘 𝑑𝛼𝑘𝜌𝑘


## 𝑑𝑡 + (𝜌𝑒+ 𝑝)∇⋅u = 0. (16)


## Using the Gibb’s relation for each material, 𝑑𝑒𝑘= 𝑇𝑘𝑑𝑠𝑘+ 𝑝

𝜌2 𝑘 𝑑𝜌𝑘, we obtain


## 𝑑𝑒𝑘


## 𝑑𝑡= 𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡−𝑝𝑑


## 𝑑𝑡


## ( 1 𝜌𝑘


## ) . (17)


### By using the GFE model [2], we get


## 𝑑 𝑑𝑡


## ( 1 𝜌𝑘


## ) = 𝜆𝑘


## 𝜌𝑘 ∇⋅u, (18)


### and


## 𝑑𝛼𝑘𝜌𝑘


## 𝑑𝑡 = −𝛼𝑘𝜌𝑘∇⋅u. (19)


### By substituting Eqs. (17), (18), and (19) into Eq. (16), we obtain

𝐾 ∑

𝑘=1 𝛼𝑘𝜌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡−


## ( 𝐾 ∑

𝑘=1 𝛼𝑘𝜆𝑘


## )


## 𝑝∇⋅u + 𝑝∇⋅u = 0. (20)


## From our previous study [2], we know that ∑𝐾 𝑘=1 𝛼𝑘𝜆𝑘= 1. Therefore, Eq. (20) can be simplified as

𝐾 ∑

𝑘=1 𝛼𝑘𝜌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡= 0, i.e.


$$
𝐾 ∑
$$

𝑘=1 𝑇𝑘 𝑑(𝑌𝑘𝑠𝑘 )


## 𝑑𝑡 = 0. (21)


### Eq. (21) reveals two special cases: the isentropic assumption ( 𝑑𝑠𝑘


## 𝑑𝑡= 0) and the pressure-temperature equilibrium assumption (sharing


## a common temperature 𝑇𝑘= 𝑇, see Remark 1) satisfy the characteristic equation 𝑑𝑠


## 𝑑𝑡= 0, which is consistent with existing research results [29,44,49]. For general situations, Eq. (21) can be further rewritten as


## 𝑑𝑠


## 𝑑𝑡=

𝐾 ∑

𝑘=1


## 𝑇−𝑇𝑘


## 𝑇 𝑌𝑘 𝑑𝑠𝑘


## 𝑑𝑡. (22)

Journal of Computational Physics 513 (2024) 113192

6

Z. He and S. Tan

Eq. (22) is just the equation of the mixture entropy 𝑠in the GFE model and implies that heat transfer among materials will affect the final evolution mode of the mixture entropy 𝑠. Any specific path along which materials evolve depends on the problem and requires physical knowledge. Qualitatively, 𝑠𝑘should be a function of 𝜌𝑘, 𝑝(i.e., 𝑠𝑘(𝜌𝑘, 𝑝)) in the GFE model. Therefore, we have


## 𝑑𝑠𝑘


## 𝑑𝑡= 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘


## 𝑑𝑝


## 𝑑𝑡+ 𝜕𝑠𝑘


## 𝜕𝜌𝑘


## ||||𝑝


## 𝑑𝜌𝑘


## 𝑑𝑡. (23)


### From our previous study [2], Eq. (23) can be further written as


## 𝑑𝑠𝑘


## 𝑑𝑡= −𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘 𝜌𝑐2 𝑒𝑓𝑓∇⋅u −𝜕𝑠𝑘


## 𝜕𝜌𝑘


## ||||𝑝 𝜌𝑘𝜆𝑘∇⋅u. (24)


### By substituting this equation into Eq. (21), we obtain


## ( 𝐾 ∑

𝑘=1 𝛼𝑘𝜌𝑘𝑇𝑘 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘


## )


## 𝜌𝑐2 𝑒𝑓𝑓= −


$$
𝐾 ∑
$$

𝑘=1 𝛼𝑘𝜌2 𝑘𝑇𝑘𝜆𝑘 𝜕𝑠𝑘 𝜕𝜌𝑘


## ||||𝑝 , (25)


### where 𝜕𝑠𝑘

𝜕𝑝 |||𝜌𝑘 and 𝜕𝑠𝑘

𝜕𝜌𝑘 ||||𝑝 rely on the specific path along which the materials evolve. This path can be analyzed more quantitatively


## by considering the heat-transfer effect. Utilizing the well-known relation 𝑑𝑝 = 𝑐2 𝑠,𝑘𝑑𝜌𝑘+ Γ𝑘𝜌𝑘𝑇𝑘𝑑𝑠𝑘where Γ𝑘= 1

𝜌𝑘

𝜕𝑝 𝜕𝑒𝑘 ||||𝜌𝑘 (i.e. 1


### 𝜉𝑘)


### represents the Grüneisen coefficient [49], we can obtain


## 𝑑𝑝


## 𝑑𝑡= 𝑐2 𝑠,𝑘 𝑑𝜌𝑘


## 𝑑𝑡+ Γ𝑘𝜌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡. (26)


## If we start with the assumption that entropy change in Eq. (21) (or Eq. (22)) is only due to the heat-transfer effect, 𝛼𝑘𝜌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡=


## 𝐻𝑘(𝑇−𝑇𝑘), in which the relaxation coefficients 𝐻𝑘are independent of the temperatures 𝑇𝑘and satisfy ∑𝐾 𝑘=1 𝐻𝑘(𝑇−𝑇𝑘) = 0, we can further reformulate Eq. (26) with the help of Eq. (18), as


## 𝑑𝑝


## 𝑑𝑡= −𝜌𝑘𝑐2 𝑠,𝑘𝜆𝑘∇⋅u + Γ𝑘 𝐻𝑘(𝑇−𝑇𝑘)


## 𝛼𝑘 . (27)


### By utilizing the pressure evolution equation of the GFE model [2] and Eq. (27), we can obtain


## 𝐻𝑘(𝑇−𝑇𝑘) = 𝛼𝑘


## Γ𝑘


## ( 𝜌𝑘𝑐2 𝑠,𝑘𝜆𝑘−𝜌𝑐2 𝑒𝑓𝑓 ) ∇⋅u. (28)


### Therefore, the internal energy evolution equation for each material (Eq. (17)) in the GFE model is as follows:


## 𝛼𝑘𝜌𝑘 𝑑𝑒𝑘


## 𝑑𝑡= 𝛼𝑘


## Γ𝑘


## ( 𝜌𝑘𝑐2 𝑠,𝑘𝜆𝑘−𝜌𝑐2 𝑒𝑓𝑓 ) ∇⋅u −𝛼𝑘𝜆𝑘𝑝∇⋅u. (29)


### The corresponding evolution equation of the total energy for each material can be given by


## 𝜕(𝛼𝑘𝜌𝑘𝐸𝑘)


## 𝜕𝑡 +∇⋅(𝛼𝑘𝜌𝑘𝐸𝑘u + 𝑝u)


## = (1 −𝑌𝑘)u ⋅∇𝑝+ 𝛼𝑘


## Γ𝑘


## ( 𝜌𝑘𝑐2 𝑠,𝑘𝜆𝑘−𝜌𝑐2 𝑒𝑓𝑓 ) ∇⋅u + (1 −𝛼𝑘𝜆𝑘)𝑝∇⋅u, (30)


## where 𝐸𝑘= u⋅u

2 + 𝑒𝑘. When the specific path along which each material evolves is isentropic, the second term on the right side of Eq. (30) is zero, and Eq. (30) degenerates to the result obtained by some researchers [50,51]. However, this path is not always isentropic in the GFE model but is generally determined by Eq. (30). For problems with an interface separating pure fluids, numerical requirements (e.g., the monotonic sound speed) may be utilized to characterize interfaces when they are numerically treated as artificial mixtures [2]. In these cases, the heat-transfer terms in the GFE model may not be physical but purely artificial.

Remark 1. The generic five-equation model with pressure-temperature equilibrium is theoretically equivalent to the classical fourequation model with kinetic equilibrium (same velocity), mechanical equilibrium (same pressure), and thermal equilibrium (same temperature) [52–58]. The distribution coefficient 𝜆𝑘that arises under pressure-temperature equilibrium is derived in Appendix A.


### 3.2. Constraints corresponding to immiscibility preservation conditions of generic five-equation model


### In this section, we derive the immiscibility preservation conditions for the GFE model.


### 3.2.1. Elementary constraint


## First, the saturation condition ∑𝐾 𝑘=1 𝛼𝑘= 1 imposes the elementary constraint given by

𝐾 ∑

𝑘=1 𝐴𝑘= 0. (31)

Journal of Computational Physics 513 (2024) 113192

7

Z. He and S. Tan


### Further, by summing Eqs. (7)-(9) for all materials, we can derive the modified version of the continuity equation, which can be given by


## 𝜕𝜌


## 𝜕𝑡+ ∇⋅(𝜌u) =

𝐾 ∑

𝑘=1 𝑀𝑘, (32)


## where ∑𝐾 𝑘=1 𝑀𝑘is the net mass interface-sharpening term and ∑𝐾 𝑘=1 𝑀𝑘≠0.


### 3.2.2. Consistency between equations for mass and volume fractions


### From the phasic mass equations (Eqs. (7)-(9)), we can obtain


## 𝑑𝛼𝑘


## 𝑑𝑡= −𝛼𝑘


## 𝜌𝑘


## 𝑑𝜌𝑘


## 𝑑𝑡−𝛼𝑘∇⋅u + 𝑀𝑘


## 𝜌𝑘 . (33)


## By defining 𝜈𝑘= 1


### 𝜌𝑘, we obtain


## 𝜗𝑘= 1


## 𝜈𝑘


## 𝑑𝜈𝑘


## 𝑑𝑡= 1


## 𝜈𝑘


## (𝜕𝜈𝑘


## 𝜕𝑡+ u ⋅∇𝜈𝑘


## ) . (34)


### Eq. (33) can be further expressed as


## 𝑑𝛼𝑘


## 𝑑𝑡= 𝛼𝑘(𝜗𝑘−∇⋅u) + 𝑀𝑘


## 𝜌𝑘 . (35)

In our previous study [2], we assumed 𝜗𝑘= 𝜆𝑘∇ ⋅u. Given that the interface-sharpening technique is not a real physical effect but purely a numerical operator, this assumption still holds true. By substituting this assumption into Eq. (35), we obtain


## 𝜕𝛼𝑘


## 𝜕𝑡+ u ⋅∇𝛼𝑘= 𝛼𝑘(𝜆𝑘−1)∇⋅u + 𝑀𝑘


## 𝜌𝑘 . (36)


### By comparing Eq. (36) with Eqs. (12)-(14), we obtain


## 𝐴𝑘= 𝑀𝑘


## 𝜌𝑘 . (37)


### Considering the constraint (Eq. (31)), the constraint that

𝐾 ∑

𝑘=1


## 𝑀𝑘


## 𝜌𝑘 = 0 (38)


### must be fulfilled.


### 3.2.3. Entropy inequalities


### In our previous study [2], 𝑑𝑝


### 𝑑𝑡in the GFE model was expressed as


## 𝑑𝑝


## 𝑑𝑡= 1


## 𝜉 𝑑(𝜌𝑒)


## 𝑑𝑡 −

𝐾 ∑

𝑘=1


## 𝛿𝑘


## 𝜉 𝑑(𝛼𝑘𝜌𝑘)


## 𝑑𝑡 +


$$
𝐾 ∑
$$

𝑘=1


## 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘


## 𝜉 𝑑𝛼𝑘


## 𝑑𝑡. (39)


### In Eq. (39), we need to know 𝑑(𝜌𝑒)


### 𝑑𝑡, 𝑑(𝛼𝑘𝜌𝑘)

𝑑𝑡 , and 𝑑𝛼𝑘


### 𝑑𝑡; 𝑑(𝜌𝑒)

𝑑𝑡 and 𝑑(𝛼𝑘𝜌𝑘)

𝑑𝑡 can be directly obtained from Eqs. (7)-(11), and 𝑑𝛼𝑘


### 𝑑𝑡is given in Eqs. (12)-(14). By substituting these results into Eq. (39), we get


## 𝑑𝑝


## 𝑑𝑡= −1


## 𝜉(𝜌𝑒+ 𝑝)∇⋅u +


$$
𝐾 ∑
$$

𝑘=1


## 𝛿𝑘


## 𝜉𝛼𝑘𝜌𝑘∇⋅u +


$$
𝐾 ∑
$$

𝑘=1


## 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘


## 𝜉 𝛼𝑘 (𝜆𝑘−1)∇⋅u


## + Θ −u ⋅P


## 𝜉 −

𝐾 ∑

𝑘=1


## 𝛿𝑘


## 𝜉𝑀𝑘+


$$
𝐾 ∑
$$

𝑘=1


## 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘


## 𝜉 𝐴𝑘.


### (40)


### The above formula can be written as


## 𝑑𝑝


## 𝑑𝑡= −𝜌𝑐2 𝑒𝑓𝑓∇⋅u + Θ −u ⋅P


## 𝜉 −


$$
𝐾 ∑
$$

𝑘=1


## 𝛿𝑘


## 𝜉𝑀𝑘+


$$
𝐾 ∑
$$

𝑘=1


## 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘


## 𝜉 𝐴𝑘. (41)


### By substituting this equation (Eq. (41)) into Eq. (23), we obtain

Journal of Computational Physics 513 (2024) 113192

8

Z. He and S. Tan


## 𝑑𝑠𝑘


## 𝑑𝑡= 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘


## (


## −𝜌𝑐2 𝑒𝑓𝑓∇⋅u + Θ −u ⋅P


## 𝜉 −


$$
𝐾 ∑
$$

𝑘=1


## 𝛿𝑘


## 𝜉𝑀𝑘+


$$
𝐾 ∑
$$

𝑘=1


## 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘


## 𝜉 𝐴𝑘


## )


## −𝜌𝑘𝜆𝑘 𝜕𝑠𝑘 𝜕𝜌𝑘


## ||||𝑝 ∇⋅u.


### (42)


## By utilizing Eq. (42), we can obtain 𝑌𝑘𝑇𝑘 𝑑𝑠𝑘


### 𝑑𝑡as


## 𝑌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡= −


## (


## 𝑌𝑘𝑇𝑘 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘 𝜌𝑐2 𝑒𝑓𝑓+ 𝑌𝑘𝑇𝑘𝜌𝑘𝜆𝑘 𝜕𝑠𝑘 𝜕𝜌𝑘


## ||||𝑝


## )


## ∇⋅u


## + 𝑌𝑘𝑇𝑘 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘


## ( Θ −u ⋅P


## 𝜉 +


$$
𝐾 ∑
$$

𝑘=1


## ( 𝜌𝑘𝛿𝑘−𝜌𝑘𝑒𝑘 ) 𝐴𝑘−𝛿𝑘𝑀𝑘 𝜉


## )


## .


### (43)


### By inserting Eq. (37) into this equation (Eq. (43)), we obtain


## 𝑌𝑘𝑇𝑘 𝑑𝑠𝑘


## 𝑑𝑡= −


## (


## 𝑌𝑘𝑇𝑘 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘 𝜌𝑐2 𝑒𝑓𝑓+ 𝑌𝑘𝑇𝑘𝜌𝑘𝜆𝑘 𝜕𝑠𝑘 𝜕𝜌𝑘


## ||||𝑝


## )


## ∇⋅u


## ⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟ part I


## + 𝑌𝑘𝑇𝑘 𝜕𝑠𝑘


## 𝜕𝑝 ||||𝜌𝑘


## ( Θ −u ⋅P


## 𝜉 −


$$
𝐾 ∑
$$

𝑘=1


## 𝑒𝑘𝑀𝑘


## 𝜉


## )


## ⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏞⏟ part II


## .


### (44)

Part I of Eq. (44) is consistent with the GFE model sharing the mixture entropy (see the discussion in Section 3.1). As discussed in Section 3.1, the interface-sharpening is not a real physical effect, and Part II of Eq. (44) should always be zero. Therefore, we obtain


## Θ = u ⋅P +

𝐾 ∑

𝑘=1 𝑒𝑘𝑀𝑘. (45)


### 3.3. Generic five-equation model with immiscibility preservation conditions

The above results show that there is a consistent relationship among the equations for the mass, momentum, total energy, and volume fractions. To achieve this consistency, the following constraints must be satisfied:


## { 𝐴𝑘= 𝑀𝑘


## 𝜌𝑘,


## Θ = u ⋅P + ∑𝐾 𝑘=1 𝑒𝑘𝑀𝑘. (46)


## Based on these constraints, we further explore the specific expressions of 𝑀𝑘, P, Θ, and 𝐴𝑘.

First, following the original ACM idea [39,41], we suppose there exists some kind of artificial compression flux vector for each phasic equation, and we assume 𝑀𝑘= ∇ ⋅ ( 𝜌𝑘J𝑘 ) in this work. For problems with an interface separating pure or nearly pure fluids, the phasic density 𝜌𝑘is constant across the interfaces [19,59]. Therefore, we further obtain 𝐴𝑘= ∇ ⋅J𝑘.

Subsequently, we derive the expression of P. In the past, the consistency between mass and momentum transport has received minimal attention, mostly in the incompressible regime for low Reynolds numbers and low density ratios [32]. However, this consistency correction to the momentum is crucial for compressible flows, without which the spurious momentum (or velocity) contribution to the kinetic energy may eventually lead to unbounded solutions [32]. The consistency of mass and momentum transport guarantees the physical coupling between the mass conservation equations (Eqs. (7)-(9)) and the momentum conservation equation (Eq. (10)). The momentum flux should be correlated with the mass flux, which has been guaranteed at a continuous level [32,33]. Therefore, it is easy to see that each material has a corresponding momentum flux as (𝜌𝑘J𝑘 ) ⊗u, and correspondingly


## P = ∑𝐾 𝑘=1 ∇ ⋅(𝜌𝑘J𝑘⊗u) .

Finally, we have Θ = ∑𝐾 𝑘=1 u ⋅∇ ⋅(𝜌𝑘J𝑘⊗u) + ∑𝐾 𝑘=1 𝜌𝑘𝑒𝑘∇ ⋅J𝑘. In compressible flows, the internal energy is not a conserved quantity because of the reversible exchange of compression/expansion work between the internal and kinetic energies. However, the sum of the internal and kinetic energies is conserved [32]. Here, we consider an interface having a unit normal n𝑘. As the interface-sharpening effect is employed to control the thickness of the interface in the normal direction, J𝑘should be 𝕁𝑘n𝑘in which 𝕁𝑘is an undetermined but scalar quantity. Moreover, it is known that u, 𝑝, 𝜌𝑘, 𝑒𝑘are also constant along the normal direction of the interface [19]. Therefore, we have J𝑘⋅∇(𝜌𝑘𝑒𝑘) = 𝕁𝑘n𝑘⋅∇(𝜌𝑘𝑒𝑘) = 0 (for u, the situation is similar). By bringing these zero-valued


## quantities into the expression of Θ, we obtain Θ = ∑𝐾 𝑘=1 ∇ ⋅ (( u⋅u


## 2 + 𝑒𝑘 ) 𝜌𝑘J𝑘 ) .


### In summary, the interface-sharpening effect of the GFE model obtained in this study can be represented as


## 𝑀𝑘= ∇⋅(𝜌𝑘J𝑘 ), (47)

Journal of Computational Physics 513 (2024) 113192

9

Z. He and S. Tan


## P =

𝐾 ∑

𝑘=1 ∇⋅ ( 𝜌𝑘J𝑘⊗u ) , (48)


## Θ =

𝐾 ∑


$$
∇⋅
(( u ⋅u
$$


## 2 + 𝑒𝑘 ) 𝜌𝑘J𝑘 ) , (49)


## 𝐴𝑘= ∇⋅J𝑘. (50)

This result not only conserves mass, momentum, and total energy but also demonstrates compatibility with thermodynamics. Moreover, the above results are general (in contrast to those in [6,32]) and independent of the definition of J𝑘. However, it is worth pointing out that such result is suitable only for mixtures in the interfacial zone.


### 4. Numerical method

In this section, a general numerical approach is proposed to solve Eqs. (7)-(14) (with expressions of 𝑀𝑘, P, Θ, and 𝐴𝑘given in Eqs. (47)-(50) respectively). We consider the one-dimensional case of this model (Eqs. (7)-(14)) to highlight the concepts of the proposed numerical approach without losing generality:


## 𝜕U


## 𝜕𝑡+ 𝜕(F(U) + G)


## 𝜕𝑥 = S(U, 𝜕U


## 𝜕𝑥), (51)


### where


## U =


## ⎛ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜⎝


## 𝛼1𝜌1 ⋯ 𝛼𝐾𝜌𝐾 𝛼𝜌𝑢 𝛼𝜌𝐸 𝛼1 ⋯ 𝛼𝐾−1


## ⎞ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟⎠


## ,F(U) =


## ⎛ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜⎝


## 𝛼1𝜌1𝑢 ⋯ 𝛼𝐾𝜌𝐾𝑢 𝜌𝑢2 + 𝑝 𝜌𝐸𝑢+ 𝑝𝑢 𝛼1𝑢 ⋯ 𝛼𝐾−1𝑢


## ⎞ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟⎠


## ,


### and


## S(U, 𝜕U


## 𝜕𝑥) =


## ⎛ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜⎝


## 0 ⋯ 0 0 0 𝛼1𝜆1 𝜕𝑢 𝜕𝑥 ⋯ 𝛼𝐾−1𝜆𝐾−1 𝜕𝑢 𝜕𝑥


## ⎞ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟⎠


## ,G =


## ⎛ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜⎝


## 𝜌1𝐽𝑥 1 ⋯ 𝜌𝐾𝐽𝑥 𝐾 ∑𝐾 𝑘=1 𝜌𝑘𝑢𝐽𝑥 𝑘 ∑𝐾 𝑘=1 𝜌𝑘 ( 𝑢2


## 2 + 𝑒𝑘 ) 𝐽𝑥 𝑘 𝐽𝑥 1 ⋯ 𝐽𝑥 𝐾−1


## ⎞ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟⎠


## .

The spatial domain is discretized into 𝑁computational cells 𝐼𝑖= [𝑥𝑖−Δ𝑥∕2, 𝑥𝑖+ Δ𝑥∕2], where Δ𝑥indicates the width of cell 𝐼𝑖and the location of the cell center is denoted as 𝑥𝑖. Without losing generality, we consider here the specific process from time 𝑡𝑛to time 𝑡𝑛+1 with a timestep Δ𝑡.


### The proposed numerical approach is based on the classical finite-volume method. Specifically, Eq. (51) in the computational cells 𝐼𝑖can be discretized as


### U 𝑛+1 𝑖 −U 𝑛 𝑖 Δ𝑡 = − ̂𝔽𝑛 𝑖+1∕2 −̂𝔽𝑛 𝑖−1∕2 Δ𝑥 + ̂S 𝑛 𝑖, (52)


### where U 𝑛 𝑖and U 𝑛+1 𝑖 denote the cell average of U in 𝐼𝑖at times 𝑡𝑛and 𝑡𝑛+1, respectively, ̂𝔽𝑛 𝑖±1∕2 denotes the net flux of the upwind


## numerical fluxes ̂F 𝑛 𝑖±1∕2 and the interface-sharpening flux ̂G 𝑛 𝑖±1∕2 at cell boundaries, that is,


## ̂𝔽𝑛 𝑖±1∕2 = ̂F 𝑛 𝑖±1∕2 + ̂G 𝑛 𝑖±1∕2. (53)

For clarity, the superscript 𝑛is omitted in ̂F 𝑛 𝑖±1∕2, ̂G 𝑛 𝑖±1∕2, and ̂S 𝑛 𝑖, hereafter. Calculating ̂F𝑖±1∕2 and ̂S𝑖follows the extended Godunovtype finite-volume method [2]. Using a suitable reconstruction scheme, we obtain the left and right states on either side of each cell edge. Riemann solvers are then utilized to derive the upwind numerical fluxes ̂F𝑖±1∕2, and the source term ̂S𝑖containing the velocity divergence term is consistently discretized using ̂𝑢𝑖+1∕2 [2]. A detailed description of the numerical approach is provided in our previous studies [2,1]. To avoid spurious pressure oscillations for isolated interfaces between fluids with different material properties, the consistency among physical quantities must be maintained in the reconstruction process. Therefore, the reconstruction variables must be cautiously selected [1,56,13], or a common reconstruction scheme must be utilized for 𝛼𝑘and 𝛼𝑘𝜌𝑘[60,61] in the GFE model. In fact, numerical analysts have developed numerical schemes that preserve some of the key mathematical and physical properties of

Journal of Computational Physics 513 (2024) 113192

10

Z. He and S. Tan

the differential models they aim to approximate in their finite-dimensional algebraic representations [62,63]. Such numerical schemes are called structure-preserving/physical-compatible discretizing methods, which preserve properties such as energy, monotonicity, maximum principles, symmetries, and involutions of the continuum models at the discrete level [63]. In the context of DIM, designing structure-preserving/physical-compatible discretizing algorithms for these models [64,59,65,8,1,66,67,60,68] has long been an active topic. In our opinion, the aforementioned specific skills required for reconstructing the GFE model are a concrete example of this research direction. Following our previous work [1], we utilize the variables W = (𝜌1, ⋯ , 𝜌𝐾, 𝑢, 𝑝, 𝛼1, ⋯ , 𝛼𝐾−1)𝑇to perform the reconstruction for obtaining the left and right states at the cell boundaries, which are then utilized to provide the upwind numerical fluxes ̂F𝑖±1∕2 = ( ̂𝐹𝛼1𝜌1 𝑖+1∕2, ⋯ , ̂𝐹𝛼𝐾𝜌𝐾 𝑖+1∕2 , ̂𝐹𝜌𝑢 𝑖+1∕2, ̂𝐹𝜌𝐸 𝑖+1∕2, ̂𝐹𝛼1 𝑖+1∕2, ⋯ , ̂𝐹𝛼𝐾−1 𝑖+1∕2)𝑇and ̂𝑢𝑖+1∕2 [2,1]. Based on the above methodology, we further propose a conservative and consistent multimaterial artificial compression method (MMACM) to solve the interface-sharpening flux ̂G𝑖±1∕2.


### 4.1. Consistent numerical framework


## First, we introduce an operator. For any variable 𝑓, the operator ̃ (𝑓)𝑖+1∕2 is defined as


## ̃ (𝑓)𝑖+1∕2 = { 𝑓𝑖, if ̂𝑢𝑖+1∕2 > 0, 𝑓𝑖+1, else, (54)


## where ̂𝑢𝑖+1∕2 associated with the Rusanov, HLL, and HLLC Riemann solvers was obtained in our previous study [2]. Using this definition, we further design


## ̂𝐺𝛼𝑘𝜌𝑘 𝑘,𝑖+1∕2 = ̃ (𝜌𝑘)𝑖+1∕2 ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2,


## ̂𝐺𝜌𝑢 𝑘,𝑖+1∕2 =

𝐾 ∑

𝑘=1 ̃ (𝜌𝑘𝑢)𝑖+1∕2 ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2,


## ̂𝐺𝜌𝐸𝑘 𝑘,𝑖+1∕2 =

𝐾 ∑

𝑘=1 ̃ (𝜌𝑘𝐸𝑘)𝑖+1∕2 ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2.


### (55)

Numerical tests showed that this framework preserves the consistency of the GFE model with the interface-sharpening effect proposed in the previous section. The remaining problem involves determining ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2, which is discussed in the next section.


## 4.2. Numerical determination of ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2


## To construct a simple and efficient method, we do not look for the theoretical expression of J𝑘but directly determine ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2 via


## a purely numerical approach. Using the same concept as in [1], ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2 is defined as


## ̂𝐺𝛼𝑘 𝑘,𝑖+1∕2 = ̃ (𝑘)𝑖+1∕2 [ ̂𝑢𝑖+1∕2 ̆𝛼𝑘,𝑖+1∕2 −̂𝐹𝛼𝑘 𝑖+1∕2


## ] , (56)


### where


## 𝑘(𝛼𝑘,𝑖) =


## ⎧ ⎪ ⎨ ⎪⎩


## 1 −

||| |||𝛼𝑘,𝑖+1−𝛼𝑘,𝑖|||−|||𝛼𝑘,𝑖−𝛼𝑘,𝑖−1||| |||

𝑞


$$
(|||𝛼𝑘,𝑖−𝛼𝑘,𝑖−1|||+|||𝛼𝑘,𝑖+1−𝛼𝑘,𝑖|||
$$

)𝑞 +𝜀, for 𝜖< 𝛼𝑘,𝑖< 1 −𝜖


## 0, otherwise


### (57)

is a newly introduced characteristic function utilized in our previous work [1] to measure the configuration of this material in cell 𝑖with 𝜖= 10−6. In this characteristic function, the sensitivity parameter 𝜀 = 10−12 has a fixed value to avoid the division of zero, and 𝑞has a user-defined power parameter that is usually set between 2 and 4. Ideally, 𝑘(𝛼𝑘,𝑖) is nothing but a Heaviside function to detect whether this 𝑘th material exists or not. However, it was found [17] that the standard formula 𝜌𝑘= 𝛼𝑘𝜌𝑘∕𝛼𝑘could lead to spurious oscillatory behavior near the interface where both 𝛼𝑘𝜌𝑘and 𝛼𝑘have large gradients. Our numerical experiments confirm this statement and further find that this phenomenon tends to occur at the boundary between the interface mixing region and the pure material region, rather than in the core region of the interface mixing region. In these regions, 𝛼𝑘𝜌𝑘has a finite value, but 𝛼𝑘is often very small. In this sense, this phenomenon is similar to the small cell problem in the Cartesian cut-cell method [69]. Under this condition, we consider the interface to be an indistinguishable flow structure on this mesh. Thus, we introduce a shock sensor [41] to modify the Heaviside function as in Eq. (57), thus introducing some numerical dissipation to enhance the stability of the MMACM. In addition, ̆𝛼𝑘,𝑖+1∕2 in Eq. (56) is set as


## ̆𝛼𝑘,𝑖+1∕2 =


## { ̆𝛼𝐿 𝑘,𝑖+1∕2, if ̂𝑢𝑖+1∕2 > 0, ̆𝛼𝑅 𝑘,𝑖+1∕2, else, (58)

where ̆𝛼𝐿 𝑘,𝑖+1∕2 and ̆𝛼𝑅 𝑘,𝑖+1∕2 are obtained using a discontinuity-preserving scheme based on a steepness-adjustable harmonic limiter [47] (or utilize its self-adjusting steepness-based version [47] directly to ensure that the final scheme obtains essentially

