
## A thermodynamically consistent and robust four-equation model for multi-phase multi-component compressible flows using ENO-type schemes including interface regularization


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq001.png)

aDepartment of Mechanical Engineering, Stanford, CA 94305, USA bTechnical University of Munich, School of Engineering and Design, Chair of Aerodynamics and Fluid Mechanics, Boltzmannstraße 15, 85748 Garching bei M¨unchen, Germany cFLOW, Department of Engineering Mechanics, KTH Royal Institute of Technology, SE-10044 Stockholm, Sweden

Abstract

In this work, a concise and robust computational framework is proposed to simulate compressible multi-phase multi-component flows. To handle both shocks and material interfaces, a positivity-preserving ENO-type scheme is coupled with multi-phase interface regularization terms. The positivity-preserving limiter is conservative and is applied locally for minimal degradation of the baseline ENO-type scheme. The interface regularization terms are extended from the conservative diffuse interface (CDI) model to accommodate multiphase, multi-component flows. The ENO-type scheme is designed to be consistent with the thermodynamic equilibrium assumptions of the four-equation multi-phase model, naturally enforcing the interface equilibrium condition — preventing oscillations in pressure, velocity, and temperature around isothermal material interfaces — without requiring additional equations for volume fraction or mixture equation of state parameters, as is commonly done for the five-equation model. Additionally, non-dilute species diffusion models are extended to the multi-phase, multi-component setting. We show that this consistent framework is equally applicable for regimes ranging from single-phase to multi-phase multi-component flows. The proposed models and numerical schemes are implemented in the highly parallel Hypersonic Task based Research (HTR) Solver, and high-resolution simulations are performed using both CPUs and GPUs.

Keywords: ENO-type, compressible, diffuse interface, four-equation model, multicomponent, multiphase

1. Introduction

The present work is motivated by the prevalence of multi-phase compressible flows in nature and industrial applications. These flows commonly feature interactions between shocks and material interfaces and create challenges for numerical methods. These challenges include obtaining a discrete representation of discontinuities while remaining conservative in total mass, momentum, and energy. Designing a robust scheme near discontinuities while retaining a high-resolution solution in other regions is critical to properly represent prevalent physics in many flow regimes, including turbulence. Current techniques for numerical simulation of interfaces can be broadly placed in two categories: interface tracking and interface capturing methods [1].

1.1. Interface tracking Interface tracking methods discretely represent interfaces as sharp discontinuities, where the interface is tracked using a Lagrangian representation. Some methods include Arbitrary Lagrangian-Eulerian (ALE)

∗Corresponding author Email addresses: hcollis@stanford.edu (Henry Collis), deniz.bezgin@tum.de (Deniz A. Bezgin), msey@kth.se (Shahab Mirjalili), alimani@stanford.edu (Ali Mani)

Preprint submitted to Journal of Computational Physics April 22, 2025


# arXiv:2504.14063v1  [physics.flu-dyn]  18 Apr 2025

[2], front-tracking [3, 4, 5], marker-and-cell (MAC) [6], and certain approaches to ghost-fluid schemes [7]. Advantages of the sharp interface include the representation of drastically different equations of state for each fluid while avoiding spurious oscillations across phase boundaries. Although interface tracking has its advantages, achieving discrete conservation is difficult and an ongoing area of research. In addition, the complexity of these methods increases when representing more complex interfacial systems, including systems with strong interfacial deformations or topological changes, high density ratios, and interactions between shocks and material interfaces.

1.2. Interface capturing

Interface capturing schemes are based on an Eulerian equation to represent the location of the interface throughout time. The formulation can represent interfaces as both sharp and diffuse, where the sharp-interface methods include standard level-set/ghost-fluid schemes, [8, 9, 10, 11, 12] and geometric volume-of-fluid (VOF) [13, 14, 15, 16, 17, 18, 19]. The sharp interface achieves similar advantages as the interface-tracking method, but the accurate geometrical representation of the interface throughout simulations without losing conservation generally becomes complex, computationally expensive, and introduces issues for scalability[20]. On the other hand, interface capturing methods including conservative level-set (CLS) [21], algebraic VOF methods [22, 23, 24], and diffuse interface methods avoid the additional complexity of the sharp-interface representations by diffusing the interface over a finite number of discrete cells. The diffuse interface methods, in particular, have become a popular method due to their simplicity of implementation and load balancing, conservation guarantees, and applicability for large density ratio flows.

1.2.1. Implicit interface capturing A common diffuse interface approach uses the dissipation of a numerical scheme to implicitly represent interfaces. These include total-variational-dimensioning (TVD) schemes which will keep all interfaces bounded and stable, though are generally diffusive and limited to low-order accuracy. These include fluxlimiter schemes such as minmod, van Leer, and Superbee [25, 26, 27]. High-order analogies of the TVD schemes include essentially non-oscillatory (ENO) type schemes [28, 29, 30, 31, 32]. ENO-type schemes (WENO and TENO) use smoothness indicators to locally switch between non-dissipative high-order and dissipative low-order stencils to capture interfaces without adding substantial dissipation far from discontinuities. These techniques have traditionally been used as shock-capturing schemes for compressible flows. When ENO-type schemes are used to model immiscible interfaces, they have sometimes been referred to as interface capturing methods in analogy to their shock-capturing property [33, 34]. As described above, this terminology is also used to broadly refer to Eulerian methods for representation of interfaces, such as VOF and level-set methods. To avoid ambiguity, the term implicit interface capturing will be used in the remainder of this work to describe methods that rely on numerical dissipation, like WENO, to provide a diffuse representation of interfaces. An advantage of many implicit interface capturing methods is their applicability to handle regions of sharp gradients due to their localized upwind-biased numerical discretization. Even so, when implicit interface capturing schemes are applied to high-Mach, high-density ratio multi-phase flow (even if combined with a phase field model [35]), the small oscillations produced by the scheme can lead to simulation failure due to the development of zones which involve either negative density or internal energy. Obtaining robust solutions for high-Mach, high density ratio multi-phase flows requires enforcing positivity of mass, the squared speed-of-sound, and boundedness of phase volume fraction. These conditions are commonly achieved through the minimal and localized use of flux limiters [36] in a manner that does not affect discrete conservation. A disadvantage of implicit interface capturing methods is that their dissipation can smear the material interfaces indefinitely. However, recent advancements in implicit interface capturing offer a remedy to this problem through the use of sharpening techniques from algebraic VOF methods, specifically the tangent of hyperbola for interface capturing (THINC) scheme [37, 38, 39, 40].

2

1.2.2. Phase field methods In addition to implicit interface capturing methods, PDE-based diffuse interface models exist which regularize material interfaces to a finite and resolvable thickness. In particular, this work is focused on a class of diffuse interface methods known as phase field models, which includes models based on the CahnHilliard [41] and Allen-Cahn equations [42]. Recent work has proposed general implementation strategies for phase field models to provide consistent and globally conservative solutions in both incompressible and compressible flows [43, 44]. Further advances in phase field modeling have resulted in locally conservative forms of the Allen-Cahn equation [45] known as the conservative diffuse interface (CDI) method. In recent years, significant progress has been made with the CDI model, including proving bounded volume fractions [46], providing required consistency conditions for both incompressible and compressible flows [46, 47], and generalizing the CDI model for simulation of N-phase immiscible flows [48] as well as on generalized curvilinear grids [49]. Additionally, the CDI model combined with shock-capturing schemes has been shown to be highly effective in capturing shock-interface interactions [35, 50].

1.3. Equilibrium conditions In addition to a robust numerical scheme, diffuse interface methods require equilibrium conditions to define fields in the diffuse zone. The most concise model is known as the four-equation model and requires thermo-mechanical equilibrium between phases within a computational element by enforcing that one temperature, pressure, and velocity vector is shared between phases [51, 52, 53, 54, 55, 56, 57]. Other popular equilibrium conditions include a five-equation model that does not enforce thermal equilibrium [58]; the six-equation model which additionally relaxes pressure equilibrium [59, 60]; and the seven-equation model, which further relaxes momentum equilibrium between phases [61]. We note that when modeling immiscible interfaces, diffuse interface methods artificially thicken the zone of coexisting phases solely to ease the numerical resolution requirements. Noting that physical interfaces have thicknesses on the order of a nanometer, thermo-mechanical equilibrium in the interface zone is practically instantaneous compared to the temporal resolution of numerical simulations. Therefore, using a diffuse interface model in this manner should not change the equilibrium conditions present in the sharp interface limit. However, in practice models that use additional equations compared to the four-equation model commonly sacrifice this desired thermo-mechanical equilibrium property as a trade-off to gain numerical robustness [62]. In some cases achieving robustness requires either sacrificing discrete conservation (of mass [63, 64] or energy [65]) or solving additional equations beyond the number of relaxed equilibrium conditions, e.g. equations involving the ratio of specific heats [66, 67, 68], which are redundant on the PDE level and lead to non-unique solutions. Aside from departing from the desired thermo-mechanical equilibrium conditions, these models incur more cost proportional to the number of additional equations which must be solved. In this sense, development of a robust and discretely conservative four-equation model without redundant PDEs addresses a physics-based need while reducing the computational cost of simulations.

1.3.1. Interface equilibrium condition The main robustness issue with the numerical simulations of the four-equation model is formulating a scheme that satisfies the interface equilibrium condition while retaining a unique and discretely conservative solution. In the general case, the interface equilibrium condition (IEC) is defined as: for a flow with uniform pressure, temperature, and velocity, the numerical discretization should not introduce spurious oscillations in any of these fields across material interfaces. If a scheme does not satisfy the IEC, spurious oscillations around material interfaces grow with time, potentially causing unphysical results and eventually code failure. Past work focused on satisfying the IEC with a unique and conservative four-equation model includes, [69], which proposed a conservative IEC preserving flux-splitting scheme. However, the scheme from [69] is only applicable to flows involving single-phase multi-component mixtures of perfect gases without shocks. A similar extension of the method of [69] to real-gas equations of state has recently been proposed [70], but it can only approximately satisfy the IEC with a limited spatial order of accuracy. Other past works using the four-equation model in the compressible regime either suffered from oscillations due to lack of satisfying the IEC [71], did not achieve uniqueness or discrete conservation [63, 66], used dissipative low-order schemes [57], or resorted to dissipative spatial filters to dampen growth of the oscillations throughout time [50].

3

One of the main contributions of this work is providing a computational strategy to satisfy the interface equilibrium condition near machine precision using a four-equation model. The ENO-type scheme is consistently formulated with the thermodynamic assumptions of the four-equation multi-phase model to prevent oscillations around material interfaces without additional equations, sacrificing conservation, or adding spatial filters. Additionally, the proposed IEC ENO-type scheme is applicable for general equations of state and complex mixing rules. In addition to the IEC ENO-type scheme, this work includes extensions to the CDI phase field model to multi-phase multi-component interfaces and an extension of the positivity preserving flux limiter of [36] to the four-equation setting for robust treatment of the interactions of high-Mach shocks with high-density ratio material interfaces. The presented formulation and discretization is inclusive of an implicit interface capturing method which satisfies the IEC with the four-equation model. This can be realized by omitting the CDI regularization terms in our model.

1.4. Outline

The remainder of this paper is outlined below. In Section 2 we introduce the system of equations describing multi-phase multi-component flows, the mixing rules used for interphase phases and intraphase components, and the equation of state used to closed the system. In Section 3 the spatial-temporal discretization is described, including the method used to obtain oscillation-free solutions across material interfaces using ENO-type schemes, the extension of the CDI model used to multi-phase multi-component systems, and the positivity-preserving flux-limiter for the four-equation model. Section 4 presents several one-dimensional and two-dimensional cases ranging from multi-component to multi-phase multi-component flows in order to verify the application of the proposed framework. Finally, in Section 5, we summarize the results and provide an outlook for future work.

2. Governing equations

2.1. Physical model

The multi-phase, multi-component form of the compressible Navier-Stokes equations will be studied in this work. In addition, we present a material-interface regularization model which can be added to the system to represent immiscible material interfaces. Although we can represent immiscible interfaces, the proposed model is general and can be used for modeling of miscible interfaces (similar to implicit interface capturing) by omitting the regularization terms and relying on a species diffusion model. To clearly distinguish between different phases, we introduce a notation in which the phase of a material will be indicated by p, where the superscript p = 1, 2, and the components within a given phase will be indicated by the subscript c, where 1 ≤c ≤N. Although we will utilize index notation for coordinates and tensor components, we do not imply index notation for p and c. The fluid state at a given time t can be described at position x = [x, y, z]T by the vector of primitive variables W = [T, Y c p , u, v, w, P]T or by the vector of conserved variables U = [ρY c p , ρu, ρv, ρw, E]T . In these definitions, Y c p is the mass of component c of phase p per total mass, known as the mixture mass fraction, u = [u, v, w]T is the velocity vector, P is the pressure, T is the temperature, ρ is the mixture density, and E is the total energy per unit volume defined as E = ρe + 1

2ρu · u where e is the internal energy per unit volume. Using these variables, the multi-phase multi-component Navier-Stokes equations with interface regularization terms can be compactly written in differential form in terms of the conserved variables U,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq002.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq003.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq004.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq005.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq006.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq007.png)

for p = 1, 2 and 1 ≤c ≤N. The convective fluxes F, G, H are defined as,

4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq008.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq009.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq010.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq011.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq012.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq013.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq014.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq015.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq016.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq017.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq018.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq019.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq020.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq021.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq022.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq023.png)

and the viscous fluxes, Fν, Gν, Hν can be expressed as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq024.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq025.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq026.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq027.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq028.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq029.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq030.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq031.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq032.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq033.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq034.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq035.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq036.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq037.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq038.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq039.png)

where τij is the viscous stress tensor,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq040.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq041.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq042.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq043.png)

with µ as the dynamic viscosity, and δij as the Kronecker delta. Additionally, the intraphase species mass diffusion vector can be derived for non-dilute species diffusion from the Stefan-Maxwell diffusion model (with certain equilibrium assumptions) as a Fickian diffusion term and a mass corrector [72]. This well-known model for non-dilute species diffusion can be extended to the multi-phase multi-component context using a confined scalar argument. The resulting model can be written as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq044.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq045.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq046.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq047.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq048.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq049.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq050.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq051.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq052.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq053.png)

ln


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq054.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq055.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq056.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq057.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq058.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq059.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq060.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq061.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq062.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq063.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq064.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq065.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq066.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq067.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq068.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq069.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq070.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq071.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq072.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq073.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq074.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq075.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq076.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq077.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq078.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq079.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq080.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq081.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq082.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq083.png)

where Dc p is the mass diffusivity of component c within phase p, Xc p is the mixture molar fraction of component c of phase p, Xp is the mixture molar fraction of phase p, Yp is the mixture mass fraction of phase p, W c p is the molecular weight of component c, and Wp is the molecular weight of phase p. Formulas describing the computation of Wp, Yp, and Xp are in Appendix E. One key difference between this formulation and the common non-dilute Stefan-Maxwell model is the incorporation of the denominators (Xp and Yp) in the gradient terms. This restricts the volume of influence of species diffusion to be within each phase and prevents unphysical leakage of components across phases. This formulation can be derived using the consistent transport models for confined scalars introduced in previous work [73, 74]. Any physical exchange across phases (e.g. phase change) would require additional models to explicitly represent mass-transfer between phases. Lastly, the heat flux qi is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq084.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq085.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq086.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq087.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq088.png)

where λ is the heat conductivity of the mixture and hc p is the specific enthalpy of component c of phase p. The final terms in Eq. 1, FDI, GDI, HDI, are the interface regularization terms (i.e. the diffuse interface terms) and are described in Section 3.2.

5

2.2. Mixture rules

2.2.1. Equilibrium assumptions In this work, thermal and mechanical equilibrium is assumed between phases within a computational element by enforcing that one temperature, pressure, and velocity vector is shared between phases. In the literature, models which enforce thermo-mechanical equilibrium are often known as the four-equation multi-phase model [51, 52, 53, 54, 55, 56, 57]. One of the main contributions of this work is providing a computational strategy to satisfy the discrete interface equilibrium condition near machine precision with the four-equation model. The proposed scheme is presented in Section 3.1. The following sections overview the mixing rules required to obtain the equation of state closure for Eq. 1. Section 2.2.2 starts by defining the mixing rules describing multi-phase mixtures and Section 2.2.3 extends this to a general multi-phase multi-component model.

2.2.2. Interphase mixing rules In this work, multi-phase mixtures are assumed to be immiscible. To describe immiscible phases, separate phases are assumed to occupy their own individual volumes and share a common pressure (Amagat’s law). The mixing rules are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq089.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq090.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq091.png)

where Tp, Pp, ep, and vp are the temperature, pressure, internal energy, and specific volume for phase p. These thermodynamic variables can be obtained from the equation of states for each phase p. These mixing rules allow for each phase p to be governed by different equation of states as well as by unique intraphase mixing rules to describe multi-component mixtures. Other quantities that are treated with interphase mixing rules include the mixture viscosity µ and mixture thermal conductivity λ as, µ = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq092.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq093.png)

where µp and λp are the phasic viscosity and thermal conductivity respectively, and ϕp is the phase volume fraction. In this work λp and µp are assumed constant in the liquid phase. For the gaseous phase, λg is defined as, λg = µg

Pr


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq094.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq095.png)

and µg is defined using Wilke’s rule [75] as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq096.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq097.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq098.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq099.png)

where,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq100.png)

8


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq101.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq102.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq103.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq104.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq105.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq106.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq107.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq108.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq109.png)

and the Prandtl number, Pr, the dynamic viscosity of species µi g, and the specific heats, CP c g, are assumed constant. Although not explored in this work, the formulation does not restrict these quantities from being extended to more advanced definitions, including defining the components with a non-linear dependence on temperature.

6

2.2.3. Intraphase mixing rules Classical intraphase mixing rules include assuming the individual components within a phase occupy their own volume (Amagat’s law) or assuming components share volumes (Dalton’s law). Traditionally, mixtures of gas components are assumed to follow Dalton’s law, with each individual component contributing a partial pressure towards the mixture pressure [76]. For a real gas equation of state (e.g. including the compressiblity factor), Dalton’s law and Amagat’s law will give different results. However, if the gas components are modeled as ideal gases, both representations obtain equivalent thermodynamic states [76]. The governing equations in Section 2.1 are written to allow either intraphase mixing rule to be applied. In this work, though the gas components will follow the ideal gas law, both mixing rules will be summarized to show the generality of the four-equation model and provide a background for extensions to more complex equations of state. The ideal mixing rules associated with assuming all intraphase components occupy the same volume (Dalton’s law) are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq110.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq111.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq112.png)

where (assuming the ideal gas law) the partial pressure of component c is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq113.png)

where R is the universal gas constant and W c is the molecular weight of component c. Additionally, the ideal mixing rules associated with assuming all components occupy the individual volumes (Amagat’s law) are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq114.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq115.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq116.png)

Lastly, the mass diffusivity of component c within phase p is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq117.png)

where, in this work, a constant Schmidt number, Scc p, is assumed for each component within a phase.

2.3. Noble Abel stiffened gas EOS To close the system illustrated in Eq. 1 the Nobel-Abel equation of state (NASG EOS) [77] is used for all components in the multi-phase multi-component mixture. For a pure component, the general NASG EOS assuming constant heat capacity reads,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq118.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq119.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq120.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq121.png)

where for component c in phase p, CP c p is the heat capacity at constant pressure, Cvc p is the heat capacity at constant volume, γc p is the heat capacity ratio, bc p is the co-volume of the component molecules, P ∞c p is the stiffened pressure to model the attractive forces between molecules in a material, and qc p is a reference energy. If bc p and P ∞c p are taken as zero (as is done for components in the gas phase) the NASG EOS reduces to the ideal gas equation of state. During a simulation, the internal energy and volume (or density) of the mixture can be determined from the conserved vector U and the mixing rules in Section 2.2. To obtain expressions for the pressure and

7

temperature of the mixture in terms of internal energy and volume, the mixing rules discussed in Section 2.2 can be combined with the NASG EOS. For a system with multiple components in both phases, an iterative procedure is required to find the pressure and temperature that satisfy the equilibrium conditions[53]. In this work, only one component will be modeled in the liquid state, and the gaseous state will be composed of ideal gas components. Given these simplifications, we can use Amagat’s law for both intraphase and interphase mixing and the NASG EOS to obtain closed expressions for the mixture pressure. An expression for the mixture pressure can be found by enforcing the equality between the two definitions for mixture temperature found from Eq. 18,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq122.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq123.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq124.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq125.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq126.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq127.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq128.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq129.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq130.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq131.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq132.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq133.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq134.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq135.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq136.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq137.png)

Solving for a common pressure between Eq. 19 and 20 leads to


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq138.png)

with


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq139.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq140.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq141.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq142.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq143.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq144.png)

and Cv = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq145.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq146.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq147.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq148.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq149.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq150.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq151.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq152.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq153.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq154.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq155.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq156.png)

where for this work p = 1 is defined as the liquid phase and P ∞≡P ∞c=1 p=1 as there is only one component in the liquid. Once pressure is determined using Eq. 21, either Eq. 19 or Eq. 20 can be used to obtain the mixture temperature. For mixtures with multiple components in each phase or specific heats dependent on temperature, iterative solvers can be used to determine the equilibrium state [53]. Unless otherwise specified, the NASG EOS parameters used for all materials throughout this work are included in Table 1.

Material CP [J kg−1K−1] γ [-] q [J kg−1] b [kg−1] P∞[Pa]

Water (Liquid) 4.185 × 103 1.0123 −1.143 × 106 9.203 × 10−4 1.835 × 108

Air 1.011 × 103 1.4 0 0 0 Helium 5.091 × 103 1.66 0 0 0 SF6 0.661 × 103 1.093 0 0 0


> **Table 1: NASG Parameters used in this work**

8

3. Numerical framework

3.1. Spatial discretization

In this work, the integral form of the partial differential equations in Eq. 1 will be solved using the finite-volume method. The differential form of Eq. 1 is assumed to admit smooth solutions for which partial derivatives exist. A general cuboid cell (i, j, k) has the spatial width defined by ∆x, ∆y, and ∆z and a cell volume defined as ∆x∆y∆z. In this work, isotropic Cartesian grids (∆x = ∆y = ∆z) will be used to spatially discretize the system. In the finite-volume formulation the cell averaged values of the conserved variables can be obtained as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq157.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq158.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq159.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq160.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq161.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq162.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq163.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq164.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq165.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq166.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq167.png)

Applying the volume integration to Eq. 1 leads to a set of ordinary differential equations,

d dt ¯Ui,j,k = −1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq168.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq169.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq170.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq171.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq172.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq173.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq174.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq175.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq176.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq177.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq178.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq179.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq180.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq181.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq182.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq183.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq184.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq185.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq186.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq187.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq188.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq189.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq190.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq191.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq192.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq193.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq194.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq195.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq196.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq197.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq198.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq199.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq200.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq201.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq202.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq203.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq204.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq205.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq206.png)

where, Fi+1/2,j,k, Fνi+1/2,j,k, and FDIi+1/2,j,k are the averaged cell face fluxes on the xi+1/2 face for the convective, diffusive, and regularization terms respectively. The definitions of the fluxes on the y−and z−faces are analogous. For multi-dimensional problems Eq. 25 is solved using a dimension-by-dimension approach where each dimension is evaluated separately. Additionally, in this work all cell-face averaged variables are approximated using a one-point quadrature rule. Higher-order representations are possible and have been explored in other works [34]. To calculate the convective flux on the cell-faces the Godunov approach [78] is used by solving a Riemann problem on a cell face. The Riemann problem is defined by fluid states on the left and right sides of a cell face, which are obtained by numerical interpolations. Across discontinuities, including shocks and material interfaces, the method of interpolation can be critical to obtaining a robust and oscillation-free solution. A brief review of previous approaches to limiting oscillations across discontinues and interfaces was presented in Section 1. In this work, an ENO-type (WENO-type, or TENO-type) interpolation scheme is used in combination with a HLLC Riemann solver. A key challenge when applying this methodology to material interfaces is formulating the interpolations to satisfy the IEC. Details on the variables which are reconstructed by the ENO-type scheme while satisfying the IEC can be found in Section 3.1.2 and details on the HLLC form is in Appendix B. The viscous fluxes are interpolated to faces using a standard second-order central scheme. The interface regularization fluxes are interpolated to faces using a second-order kinetic energy and entropy preserving (KEEP) central interpolation that satisfies the IEC [79]. More detail on the regularization terms and the KEEP interpolation is in Section 3.2.

3.1.1. Avoiding spurious oscillations High-order interpolation of the fluid state directly to the cell-faces can result in spurious oscillations around interfaces due to the interactions of discontinuities in dependent fields. Additionally, the nonlinearity of ENO-type schemes can lead to different numerical stencils for each fluid field. As these fields are dependent on one another in physical space, the inconsistency between the stencils (and the stencil’s numerical dissipation) can result in spurious oscillations. Instead, projecting the fields to characteristic space before interpolation decouples the fluid fluids into independent characteristic variables and avoids the issues associated with using ENO-type schemes [28, 29, 30, 80].

9

Although interpolating the characteristic variables reduces spurious oscillations, it does not cause the system to inherently satisfy the interface equilibrium condition and will still result in oscillations around material interfaces. As discussed in Sections 1.3 and 1.3.1 past works have presented strategies for satisfying the IEC. However, these strategies either require loss of conservation, add redundant equations, or restrict mixing rule and thermodynamic equilibrium assumptions.

3.1.2. Satisfying the interface equilibrium condition Notably, in this work, the IEC is satisfied to near-machine precision using the four-equation model without redundant equations. The idea is based on the use of the primitive vector W = [T, Y c p , u, v, w, P]T , as the basis for building the interpolations to cell faces. An important clarification must be made that this set of primitive variables includes temperature instead of density (unlike what is done for the volume-fractionbased five-equation model [34]). Interpolating (including with a nonlinear ENO-type scheme) directly based on P, T, and u enforces the interface equilibrium condition numerically, since a well-formulated numerical scheme will keep a constant field constant after interpolation. Replacement of T with ρ, as done in other studies [34], will no longer maintain a constant temperature after interpolation (since ρ contains a jump across the interface) and results in oscillations of P, T, and u (shown in Table 7). Instead, using the primitive vector, W = [T, Y c p , u, v, w, P]T , removes oscillations around isothermal material interfaces and discretely satisfies the IEC. Though the strategy outlined above will satisfy the IEC, as discussed in Section 3.1.1, it is advantageous (especially around shocks) to decompose the fluid-state into characteristic variables in order to avoid additional spurious oscillations from the non-linearity of the ENO-type schemes. In this work, the proposed primitive basis of W = [T, Y c p , u, v, w, P]T (not including one Y c p ) is used to define characteristic variables. The characteristic matrices used for the projection of W = [T, Y c p , u, v, w, P]T into characteristic space are included in Appendix A. Additionally, using an interpolation based on characteristic variables with a nonlinear ENO-type scheme can be analytically shown to satisfy the IEC ( Appendix C). A numerical example showcasing the successful application of this IEC ENO-type scheme to solve an inviscid multi-phase droplet advection with oscillations in pressure, velocity, or temperature, near machine precision is in Section 4.2.2.

3.1.3. Godunov Approach The resulting numerical strategy for obtaining the flux on cell faces is summarized below.

1. Project W into characteristic space to obtain ˜ W

(a) Define an average of U on a face: Ui+ 1

2 (using either arithmetic or Roe average)

(b) Define the left eigenvector, S−1 from the averaged state of Ui+ 1

2 (c) ˜ W = S−1W

2. Obtain left and right states of ˜ W on the cell face using ENO-type scheme

3. Project ˜ W L i+ 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq207.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq208.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq209.png)

2

(a) Define the right eigenvector, S from the averaged state of Ui+ 1

2 (b) W L i+ 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq210.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq211.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq212.png)

2

4. Obtain the left and right states for density using the EOS

5. Find the left and right states of the conserved variables from primitives

6. Find the cell-face flux using a HLLC Riemann solver

(a) Fi+ 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq213.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq214.png)

2 )

10

The approach outlined above will result in high-resolution and low-dissipation capturing of material interfaces and shocks. However, unlike shocks, material interfaces are not self-sharpening. Though highorder ENO-type interpolations can be used across material interfaces as an implicit interface capturing scheme, the numerical dissipation from the upwind-biased interpolations will cause mixing of immiscible phases throughout time. To enforce immiscibility between phases, interface regularization terms can be added to the system to sharpen multi-phase interfaces while allowing intraphase species diffusion.

3.2. Interface regularization

To counteract the intrinsic numerical diffusion caused by both the ENO-type interpolation and the approximate Riemann solver [78] from the Godunov approach described in Section 3.1.3, material interface regularization terms can be added to enforce a finite thickness interface throughout a simulation. A comparison of results achieved using an implicit interface capturing scheme (WENO5Z) and using an explicit phase field method (WENO5Z+CDI) is shown for a Rayleigh-Taylor instability in Figure 1. The setup for the case follows the details presented in [49] and Figure 1 shows the density field at t = 1.0125s. As shown, the two approaches have different effects. The implicit interface scheme represents subgrid interfaces with mixing, whereas the explicit interface regularization model will keep interfaces immiscible. However, the interface regularization terms will result in preemptive breakup of under-resolved interfaces even in the absence of surface tension. The applicability of adding interface regularization terms can be dependent on the spatial and temporal scales of the problem of interest.


> **Figure 1: Comparison of density field for RT-instability with (right) and without (left) interface regularization terms.**

The interface regularization proposed in this work is an extension of the conservative diffuse interface (CDI) model [45, 46, 81, 47] to incorporate multi-component mixtures. The multi-phase multi-component CDI model is designed to capture immiscible interfaces between phases and treat all components within phases as confined scalars [73] which can go through intraphase diffusion without restriction. The model form of the multi-phase multi-component CDI model can be written as,

FDI(U) =


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq215.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq216.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq217.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq218.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq219.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq220.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq221.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq222.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq223.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq224.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq225.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq226.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq227.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq228.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq229.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq230.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq231.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq232.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq233.png)

11

with,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq234.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq235.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq236.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq237.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq238.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq239.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq240.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq241.png)

and k = 1

2u · u as the kinetic energy, Rp,x = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq242.png)

p Rp,x, and ρp and hp are defined using the EOS and governing mixing rules of the given phase p. Additionally, Γ, ϵ, and ϕmin are user-set parameters. The parameter Γ is a velocity scale which determines the speed the interface equilibrates towards a hyperbolic tangent profile. The parameter ϵ is a length scale and determines the thickness of the interface. The parameter ϕmin determines the volume fraction floor which the regularization term relaxes towards. In this work, ϕmin = 10−8 for all cases. The multi-phase form of Eq. 27 has been studied extensively and it has been shown that properly choosing ϵ ∼∆and Γ = max(|ui,j,k|, |vi,j,k|, |wi,j,k|), ∀i, j, k results in bounded volume fraction for a large range of circumstances, including incompressible flows [46, 81], compressible flows [47], and recently for geometries using generalized curvilinear grids [49]. However, when numerically solving Eq. 27 interpolation errors of ϕp can occur close to the interface since the regularization sharpens the interface towards a hyperbolic tangent profile of thickness ϵ. Interpolations of ϕp, and its derivatives, are needed to calculate the non-linear sharpening term and the interface normal on a cell face. In order to reduce numerical errors during these interpolations, ϕp can be transformed to an approximate signeddistance function ψ [82, 12, 83, 84] using,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq243.png)

where δ is a very small number (taken as δ = 10−100 in this work). Near the interface the ψ field is approximately linear and will not suffer from numerical error associated with interpolation. An analytical reformulation of Eq. 27 in terms of ψ has been previously completed [84] and results in,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq244.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq245.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq246.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq247.png)

4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq248.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq249.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq250.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq251.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq252.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq253.png)

Both models shown in Eq. 27 and Eq. 29 are equivalent on the continuous level, but the numerical advantages of transforming the nonlinear term from ϕ to ψ make the transformed model more computationally attractive. As such, the regularization model in Eq. 29 will be used for all simulations in this work.

3.2.1. Numerical implementation of regularization model Analogous to the formulation of the ENO-type scheme to satisfy the IEC, the thermodynamic quantities in the regularization flux can be independently interpolated and subsequently averaged to construct a face flux that adheres to the IEC. Accordingly, the regularization flux in Eq. 26 is implemented using a secondorder skew-symmetric split-form centered scheme that satisfies the IEC while preserving kinetic energy and entropy (KEEP) [79] As an example, the numerical flux for the regularization terms on a cell face in the x-direction can be written as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq254.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq255.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq256.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq257.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq258.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq259.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq260.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq261.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq262.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq263.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq264.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq265.png)

12

where

� ap,x (i±1/2) = Γ


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq266.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq267.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq268.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq269.png)

4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq270.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq271.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq272.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq273.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq274.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq275.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq276.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq277.png)

� Rcp,x (i±1/2) = −Y c p Yp ρp


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq278.png)

� R,x (i±1/2) = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq279.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq280.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq281.png)

k (i±1/2) = 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq282.png)

� hpRp,x (i±1/2) = −hpρp (i±1/2) � ap,x (i±1/2). (35)

In the notation above, � (·) (i±1/2) denotes a field which consists of a numerical derivative operation onto a

face, and (·) (i±1/2) represents an interpolation operation onto a face. The critical part of skew-symmetric split-form, which allows the scheme to satisfy IEC, is keeping all velocity and density interpolations to faces independent. Furthermore, it is critical to interpolate the product of enthalpy and density together, as this will retain a consistent pressure field at the cell face. Details on the formulations for each specific term of the CDI model can be found in the following reference [79], and an example on how to extend split form schemes to higher-order spatial accuracy is shown here [85].

3.3. Positivity preservation

Section 3.1 describes the numerical methods that provide nearly non-oscillatory solutions for flows with shocks and material interfaces. However, the high-order ENO-type schemes still lead to small oscillations which can result in inadmissible thermodynamic states (e.g. negative internal energy or density) and code failure. The positivity of the speed of sound is a critical metric for simulation robustness. To guarantee a positive sound speed the equation of state can be used where the hyperbolic speed of sound is defined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq283.png)

where β = 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq284.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq285.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq286.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq287.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq288.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq289.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq290.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq291.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq292.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq293.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq294.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq295.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq296.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq297.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq298.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq299.png)

For a single-phase flow where P ∞c p = 0 ∀c, p, the temperature, pressure, and density must be positive to keep the square of the sound speed positive. For a multi-phase flow it is possible for the squared speed of sound to be positive even if mixture pressure is negative. Physical situations like cavitation can result in negative pressures, and the NASG equation of state used in this work permits this. From Eq. 19 and Eq. 20, to guarantee an admissible temperature, both the internal energy and the density of the mixture must be positive. Additionally, a positive temperature and pressure will guarantee positive phasic internal energies and densities from Eq. 18. For both singleand multi-phase flows, to guarantee the positivity of the squared speed of sound with the NASG equation of state the following two requirements must hold,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq300.png)

13

With these requirements satisfied the solution will have an admissible speed of sound, temperature, pressure, and mixture density. In order to guarantee boundedness for mass fractions and volume fraction between zero and one, a separate check must take place,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq301.png)

To ensure a physically admissible solution at all times, three limiters are added to the numerical procedure. In the Godunov approach described in Section 3.1.3, an interpolation limiter is added after the ENO-type high-order interpolation to the face, and a flux limiter is added after the approximate Riemann solver. Additionally, a flux limiter is added after the interface regularization flux described in Section 3.2 due to the addition of the sharpening term.

3.3.1. Interpolation limiter After high-order ENO-type interpolation (step 4 of the Godunov approach), the boundedness of the mass fractions is achieved by limiting the mass fractions using an approach described in [87]. As an example, the procedure for interpolation of the mass-fraction on the minus side of the (i + 1/2) face will be described. First, the mass fraction field left out of the characteristic projection is individually interpolated to the cell face using an ENO-type scheme. Then, the mass fractions can be limited with the following steps,

1. If � c � p Y c p − (i+1/2) > 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq302.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq303.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq304.png)

(b) ϵc p = � 1−�


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq305.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq306.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq307.png)

2. If �

c �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq308.png)

(a) Σ+ = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq309.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq310.png)

(b) ϵc p = � 1−�


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq311.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq312.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq313.png)

3. Y c p − (i+1/2) = Y c p − (i+1/2) + ϵc p.

The mass fraction limiter for the positive side of the (i + 1/2) cell face is done symmetrically. One of the advantages of this limiter is that for multi-component systems it will ensure that interpolation error does not impact inert species in the stencil and only those with varying profiles. It will also spread out error between multiple species instead of forcing all interpolation errors on the species left out of the interpolation. Additionally, the positivity of pressure and temperature can be independently checked after the ENO-type interpolation and mass-fraction limiter. If a field is inadmissible, a first-order interpolation (cell-centered value) is taken as the corresponding left/right state to build the flux in the Riemann solver. For example,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq314.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq315.png)

Note that unlike works with similar limiters [36, 40], in this work θ and ϵc p are kept independent for each field after interpolation, allowing for a less dissipative positivity-preserving interpolation. Once all fields in the ENO-type interpolation have been limited, the state on the cell face can be used to build admissible conserved variables (step 5 of Godunov approach) and a Riemann solver can be used to find a flux (step 6 of the Godunov approach).

14

3.3.2. Flux limiter After using the Riemann solver to find a flux on the cell face, a flux limiter is used to guarantee that the state at the following time step is admissible [36, 40]. An outline for limiting the advection flux in the x-direction with an approximate HLLC Riemann solver is shown below.

1. Calculate FHLLC(U − (i+1/2), U + (i+1/2))

2. Use pseudo-time integration within the RK sub-step to check that FHLLC(U − (i+1/2), U + (i+1/2)) is admissible. Here D is the spatial dimension of the simulation and RK0, RK1, and RK2 are coefficients for a given SSP RK sub-step as shown in Eq. 43.

(a) ˜U (RKstep+1) i = RK0U (n) i + RK1U (RKstep) i + 2RK2∆tDFHLLC

(b) ˜U (RKstep+1) i+1 = RK0U (n) i+1 + RK1U (RKstep) i+1 −2RK2∆tDFHLLC

3. If Eqs. 39 and 40 hold for both ˜U (RKstep+1) i and ˜U (RKstep+1) i+1 , FHLLC(U − (i+1/2), U + (i+1/2)) is admissible. Otherwise, a first-order positivity-preserving flux, FHLLC(U(i), U(i+1)), is used.

Step 3 of the outline above can be augmented with a blending operation between FHLLC(U − (i+1/2), U + (i+1/2)) and FHLLC(U(i), U(i+1)). Initial tests using blending did not show a noticeable improvement in the results. In order to reduce the complexity and cost of the positivity preserving routine, blending was not used in this work. It is also useful to observe that checking the solution satisfies the constraints in Eqs. 39 and 40 only requires the conserved vector U without the evaluation of the NASG EOS. For this work, the NASG EOS is analytical for all fields, but for more complex systems, including those with non-constant specific heats, an expensive iteration procedure could be required to evaluate pressure and temperature [53]. Evaluating Eqs. 39 and 40 remains inexpensive, even when the EOS becomes more complex. After calculating the diffuse interface flux from Section 3.2, the diffuse interface flux can be deemed admissible by using the same flux-limiter algorithm described above. In step 1, replace FHLLC with FHLLC + FDI, where FHLLC is the admissible flux from the advection term. If the diffuse-interface flux is inadmissible it is not added to the solution. Restricting the use of the diffuse-interface flux is infrequent and does not result in any visible smearing/mixing of immiscible multi-phase interfaces for the simulations in Section 4.

3.4. Temporal integration After discretization, the system is expressed as a set of ordinary differential equations which can be integrated in time. In this work we use a third-order strong-stability preserving (SSP) Runge-Kutta method expressed below [88],


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq316.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq317.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq318.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq319.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq320.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq321.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq322.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq323.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq324.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq325.png)

where L[·] is the operator to evaluate the numerical approximation of the spatial operations. The superscripts with parenthesis, (1) and (2), denote the first and second denote the sub-steps within time step n. The superscripts without parenthesis, n and n+1, denote the current time step n as well as the subsequent time step n + 1. Additionally, the time step size ∆t is given by either the advection or diffusion CFL criterion expressed as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq326.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq327.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq328.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq329.png)

where 0 < CFL ≤1 and a is the hyperbolic speed of sound given by Eq. 36.

15

3.5. Boundary Conditions: Navier-Stokes characteristic boundary conditions The Navier-Stokes characteristic boundary conditions [89, 90](NSCBC) are used for non-reflecting inflow and outflow conditions. Details on single-phase multi-component uses of the NSCBC conditions can be found in [91]. Extension of the classic multi-component mixing rules to include multi-phase multi-component flows for the NASG equation of state has been completed [86] and is used in this work.

3.6. High performance computing The implementation of the proposed models and numerical schemes was completed in the highly parallel Hypersonic Task-based Regent (HTR) solver [92, 93, 94]. HTR is a task-based solver built on the Legion runtime which provides portability to run distributed simulations on both CPUs and GPUs [95].

4. Numerical results

The formulation presented in Section 2.1 is written in a general form and is applicable to flows ranging from single-component single-phase to multi-phase multi-component without changing the numerical scheme, adding redundant equations, or changing the equilibrium assumptions of the four-equation model. The applicability of the formulation for these regimes will be shown with simulations focused on single-phase flows in Appendix D, single-phase multi-component tests in Section 4.1, multi-phase single-component tests in Section 4.2, and multi-phase multi-component tests in Section 4.3. All results, unless otherwise specified, were created using WENO5Z [96] for spatial interpolation and a temporal CFL of 0.5.

4.1. Multi-component tests In this section two single-phase multi-component simulations are shown without interface regularization to verify the applicability of the framework for multi-component gaseous flows. The first test is a shockbubble interaction between helium and and air bubble and is compared against an experiment [97]. The second test is a single-mode Richtmyer-Meshkov instability and compared to past computational studies [7, 98].

4.1.1. Shock-bubble interaction: air-helium bubble The interaction of a shock in air with a helium bubble is a well-documented test case for multi-component flows. The schematic showing the setup for this case is shown in Figure 2. The initial thermodynamic state as well as the component viscosity and Schmidt numbers used in this test are shown in Table 2. Lastly, a mixture Prandtl number of 0.71 is used to define the heat conduction and a spatial resolution of (4096×1024) was used.


> **Figure 2: Shock-bubble interaction between air and helium. All units are in meters.**

16


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq330.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq331.png)


> **Table 2: Initial conditions for 2D shock-bubble interaction between air and helium**

For this case the shock-wave speed is 423 m/s and for the setup shown in Figure 2, the shock impacts the helium bubble after 23.6 µs. As this case is between two miscible species the interface regularization term is not active and species mass diffusion is present between the air and helium. Figure 3 compares the evolution of the helium bubble with an experiment [97] at times after the shock first makes contact with the helium bubble. All temporal snapshots show good visual agreement between the numerical simulation and the experiment.


> **Figure 3: Shock-bubble interaction between air and helium. First row: experiment [97] (not included in arXiv due to limited**

copyright). Second row: numerical Schlieren ln � ||∇ρ||

ρ � . Third row: temperature. Final row: density. From left to right the snapshots are taken at times of 72, 102, 245, 427, and 674µs after the shock impacts the helium cylinder.

4.1.2. Single-mode Richtmyer-Meshkov instability We consider a 2D single-mode Richtmyer-Meshkov simulation following the computational setup used in past works for additional code-to-code verification [7, 98]. The thermodynamic quantities used in this test are shown in Table 3. The test consists of a Mach 1.24 shock in air passing through a perturbed interface of SF6 and details on the initial condition are shown in Figure 4. Figure 5 shows the development of the RMI over time. After the shock passes through the perturbed interface the characteristic mushroom shape appears and a mixing layer forms between the SF6 and air. The locations of the spike (farthest left point of the interface), the bubble (farthest right point of the interface), and the mixing zone (difference between the spike and bubble location) are reported for three resolutions. These resolutions are (512×128), (1024×256), and (2048×512). Furthermore, the RMI results are compared

17


> **Figure 4: Schematic of initial condition for single-mode Rightmyer-Meshkov instability between air and SF6.**


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq332.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq333.png)


> **Table 3: Initial conditions for 2D single-mode Richtmyer-Meshkov instability**

with previous numerical experiments in Figure 6. The axis in Figure 6 is scaled to consistently compare with the non-dimensionalization used in previous studies.


> **Figure 5: Single-mode Rightmyer-Meshkov instability between air and SF6 over time. First row: density. Second row: numerical Schlieren � ln �||∇ρ|| ρ �� . Last row: temperature. The snapshots are taken at rescaled times t∗from left to right of 0, 3.2, and 10.**


> **Figure 6 shows that the three numerical resolutions simulated in this work have converged relative to the locations of the spike, bubble, and mixing zone over time. Additionally, it shows that the results from this work match well with results from past published works [7, 98], verifying the implementation and application of the proposed numerical scheme.**

4.2. Multi-phase tests To illustrate the applicability of the regularization model combined with ENO-type schemes to singlecomponent multi-phase flows four tests will be shown in this section. The first is a classic one-dimensional gas-liquid Riemann problem to verify the implementation of the model for multi-phase flows with shocks. The second is an isothermal and inviscid water droplet advection in air to verify the interface equilibrium

18


> **Figure 6: Characteristic locations over time for single-mode Rightmyer-Meshkov instability between air and SF6. The labels are listed as location of the (spike, bubble, mixing layer) for the fine resolution (2048 × 512)( , , ), medium resolution (1024 × 256) ( , , ), coarse resolution (512 × 128) ( , , ), Terishima data [7] ( , , ), and Adams data [98]( , , ).**

condition is satisfied. The third case is a shock in air interacting with a water droplet to verify the implementation of the positivity preserving algorithm as this cases will fail without it. Lastly, an inviscid Mach 100 water jet will be simulated to illustrate the robustness of the framework even when applied to unrealistically difficult flows.

4.2.1. Gas-liquid Riemann problem We consider a common gas-liquid Riemann problem which was first analyzed by [10]. This problem is formulated as a model problem for an underwater explosion in which the left state is highly compressed air and the right state is water at atmospheric pressure. The same setup used by [34] is repeated here with a spatial resolution of (200 × 1) and final time of t = 0.2. The initial condition and material parameters are listed in Table 4.


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq334.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq335.png)


> **Table 4: Initial conditions for 1D gas-liquid Riemann problem**


> **Figure 7 shows the results for TENO6 [32], WENO5JS [31], and WENO5Z [96] compared to the exact solution. All schemes perform well without excessive oscillations in density, pressure, and temperature. Small oscillations can be seen in the phasic density plots which are larger for TENO6 than the WENO-type schemes. Additionally, the regularization of volume fraction keeps the interface at nearly constant thickness for all ENO-type schemes.**

4.2.2. Inviscid droplet advection As discussed in Section 3.1.2 it is critical to ensure that the numerical scheme satisfies the interface equilibrium condition (IEC) and does not introduce unphysical oscillations around material interfaces. The standard test for ensuring the scheme satisfies the IEC is an inviscid advection case of a water droplet

19


> **Figure 7: Gas-Liquid Riemann problem for three ENO-type schemes. Exact solution: ( ), WENO5JS [31] ( ), WENO5Z [96]( ), TENO6 [32]( ).**

in air. This test is used extensively for validating the 5-equation model but is more difficult in the 4equation context. As discussed in Section 3.1.2, the 4-equation model requires an oscillation-free temperature field across an isothermal material interface. The additional coupling requires oscillation-free pressure, temperature, and velocity fields. Table 5 outlines the initial setup for this 1D droplet advection case. Two WENO-type schemes and two TENO-type schemes, each coupled to the interface regularization terms described in Section 3.2, will be used to verify that the proposed scheme satisfies the IEC for a spatial resolution of (100 × 1).


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq336.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq337.png)

2 � 1 + tanh � 0.25−(x−0.5)

2ϵ∆ �� Water 997 5.0 101325.0 297 0.0 ϕg = 1 −ϕl Air 1.18 5.0 101325.0 297 0.0


> **Table 5: Initial conditions for 2D droplet advection problem**

4-eq IEC ENO-type Scheme max � |P −Pexact|


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq338.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq339.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq340.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq341.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq342.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq343.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq344.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq345.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq346.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq347.png)


> **Table 6: Maximum normalized IEC error after one flow-through time for multiple ENO-type schemes using W = [T, Y c p , u, v, w, P]**

After advection for one flow-through time, the pressure, temperature and velocity errors are shown in Table 6 for the proposed IEC ENO-type schemes. The errors for all fields remain near machine precision for all

20

5-eq IEC ENO-type Scheme max � |P −Pexact|


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq348.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq349.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq350.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq351.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq352.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq353.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq354.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq355.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq356.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq357.png)


> **Table 7: Maximum normalized IEC error after one flow-through time for multiple ENO-type schemes using W = [ρY c p , u, v, w, P] which satisfies IEC for the 5-equation model, but not for the 4-equation. For ease of implementation, the results in this table are run using the Godunov approach without the characteristic decomposition shown in [34].**

WENO-type schemes and the TENO5 scheme after one period of advection, which verifies that the proposed scheme satisfies the interface equilibrium condition. The TENO6 scheme remains near machine precision for pressure error across the interface but shows a slightly higher error for the velocity and temperature fields due to the sensitivity of TENO6 to the pile-up of round-off errors [99]. For comparison, Table 7 shows the results of the advection test using the basis suggested for the 5-equation model, W = [ρY c p , u, v, w, P]. Though using the basis, W = [ρY c p , u, v, w, P], satisfies IEC for the 5-equation model, enforcing thermodynamic equilibrium to obtain the 4-equation model results in oscillations across the interface many orders of magnitude larger than with the proposed basis of W = [T, Y c p , u, v, w, P] for all ENO-type schemes.

4.2.3. Shock-droplet interaction: air-water For this test we consider a Mach 1.47 shock-wave in air interacting with a water droplet at atmospheric conditions. To remain stable for late times, the test case requires the positivity-preserving algorithm described in Section 3.3. The initial thermodynamic conditions used in this simulation are in Table 8. A diagram showing the setup of this case is shown in Figure 8. Three spatial resolutions of (512 × 512), (1024 × 1024), and (2048 × 2048) where used.


> **Figure 8: Schematic of initial condition for shock-droplet interaction between air and water.**


> **Figure 9 shows the volume fraction, pressure, temperature, and Schlieren throughout time. After the shock travels through the water droplet a rarefaction wave is formed at the trailing edge. The corresponding pressure wave will travel towards the front of the droplet and create a low pressure zone at the trailing**

21


> **Figure 9: Time evolution for shock-droplet interaction between air and water. First row: volume fraction. Second row: pressure. Third row: numerical Schlieren � ln �||∇ρ|| ρ �� . Fourth row: temperature. Normalized time from left to right: 0, 0.15, 0.4, 0.8.**


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq358.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq359.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq360.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq361.png)


> **Table 8: Initial conditions for 2D shock-bubble: water-air problem**

edge. In order to simulate through this point in the simulation the positivity limiter was active for both the WENO5Z interpolation as well as the flux limiter. Figure 10 shows the activation of the flux-limiter over time. In general, the activation of the limiter is very sparse in the domain, with a maximum percentage of activation for 0.008% of faces. The early sharp rise occurs when the initial rarefaction wave leaves the trailing edge of the drop. These pressure waves will reflect from within the droplet with decreasing magnitude throughout the simulation. Later, a wake region forms behind the water droplet and shear forces

22

deform the droplet creating thin features, resulting in spatially dispersed activation of the limiter. To keep the interface finite and resolvable during deformation of the droplet, the CDI model turns the sheared thin features into small secondary droplets. The ability of the CDI model to form droplets can be interpreted as numerical surface tension. As this problem does not include physical surface tension (Weber number is infinity) the interface deformation and formation of interfacial features smaller than the grid-size will exist at any resolution. As shown in Figure 9 the CDI model will represent these unresolved features as immiscible droplets of finite size. Since the CDI model is locally conservative, these droplets do not disappear with time (as opposed to other popular interface capturing schemes which are only globally conservative) and instead are transported throughout the simulation. Without the CDI model the unresolved breakup of the main drop would be modeled by the implicit diffusion of the numerical scheme and result in mixing of immiscible phases. The center of mass of the main water droplet is tracked throughout time and compared to three resolutions simulated in this work, as well as with two past computational simulations. These results are reported in Figure 11 and show good agreement between all cases.


> **Figure 10: Percentage of faces which required flux limiter over time for medium resolution (1024 × 1024) of the shock-droplet interaction between water and air.**

4.2.4. Mach 100 water column in air The final single-component multi-phase simulation we consider is the injection of a Mach 100 water column into ambient air. The test shows the applicability of the framework to cover very large density ratios and strong shocks. The initial conditions for this test are given in Table 9 and a spatial resolution of (2048 × 1025) was used with a constant time step of 7.5 × 10−4µs.


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq362.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq363.png)


> **Table 9: Initial condition for 2D Mach 100 water column**

The volume fraction, pressure, schlieren and temperature, for this test are shown in Figure 12. The simulation requires the positivity limiting procedure to remain stable. Even with the limiter present the shocks remain sharply captured by the WENO5Z [96] scheme and the volume fraction field shows that the interface stays immiscible and does not contain spurious oscillations even when interacting with large temperature and pressure gradients.

23


> **Figure 11: Evolution of centroid over time for shock-droplet case. Coarse resolution (512 × 512) ( ), medium resolution (1024 × 1024) ( ), fine resolution (2048 × 2048) ( ), Colonius data [100] ( ), and Man Long data [36]( ).**


> **Figure 12: Time evolution for Mach 100 water jet into air. First row: volume fraction. Second row: pressure. Third row: numerical Schlieren � ln �||∇ρ|| ρ �� . Fourth row: temperature. Time from left to right: 1µs, 3µs, and 6µs.**

4.3. Multi-phase multi-component tests This Section contains three multi-phase multi-component tests to explore the full abilities of the proposed formulation. The first case is a newly proposed one-dimensional Shu-Osher problem. The second case is a multi-component phase constrained diffusion problem. The final test is a two-layer cylindrical RichtmyerMeshkov implosion to show interactions with interfaces, shocks, and component mixing.

24

4.3.1. Multi-phase multi-component Shu-Osher To test the presence of shocks interacting with a regularized multi-phase multi-component interface, a modified Shu-Osher problem is proposed. The materials used in this test are defined using the ideal gas equation of state with different γ as, γ1 = 1.4, γ2 = 1.3, γ3 = 1.885. Though the materials are all gaseous, this case is computationally considered a multi-phase multi-component test as the interface between γ1 is enforced as immiscible with γ2 and γ3 using the interface regularization terms. The initial condition is shown in Table 10. The profile defining the interface between γ2 and γ3 for x ≥1 can be found using the relations, X2 = 1/(γ−1)−1/(γ3−1) 1/(γ2−1)−1/(γ3−1), and X3 = 1.0 −X2.

Location Material ρ u P µ γ

0 ≤x < 1 X1 3.857143 2.629369 10.3333 0.0 1.4 1 ≤x < 20.0 X2 & X3 1 + 0.2 sin (5(x −5)) 0.0 1.0 0.0 1 + 1/ (1.33 + 0.2 sin (5(x −5)))


> **Table 10: Initial conditions for 1D multi-phase multi-component Shu Osher**

0 5 10 15 20 0.0

0.5

1.0 X1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq364.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq365.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq366.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq367.png)

X2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq368.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq369.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq370.png)

0 5 10 15 20 x

0

1

2

u

0 5 10 15 20 x

5

10

P

0 5 10 15 20 x

1

2

3


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq371.png)


> **Figure 13: Multi-phase multi-component modified Shu-Osher problem at time t = 4. Reference case ( ), with CDI interface regularization ( ), and without interface regularization (implicit interface capturing) ( ).**


> **Figure 13 shows the results of the multi-phase multi-component Shu Osher simulation using a spatial resolution of (400×1) with and without interface regularization compared to a reference of spatial resolution (12800 × 1) without regularization at a time t = 4.0. The regularization sharpens the interface between the immiscible phases and leads to a closer match with the reference compared to only using the ENO-type scheme. Additionally, the components within the gas phase individually behave as expected and do not contain spurious oscillation due to the introduction of the interphase regularization term.**

4.3.2. Phase constrained multi-component mixing The model proposed to capture intraphase diffusion without introducing leakage between phases was described in Eqs. 5 and the final form is included here,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq372.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq373.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq374.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq375.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq376.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq377.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq378.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq379.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq380.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq381.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq382.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq383.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq384.png)

25

To test this model, we simulate the diffusion between air and helium within a phase-constrained bubble surrounded by SF6. The goal of this case is to showcase the ability of Eq. 5 to allow for intraphase diffusion without leaking material across the immiscible interface boundary. To study the behavior of the model we propose a case which consists of an immiscible interface between SF6 and air/helium being enforced throughout time. Additionally, within the helium bubble, five air pockets are initialized to induce intraphase mixing. The initial condition for this case is shown in Figure 14 and the spatial resolution is (400×400). The thermodynamic state for this problem is defined in Table 11, where the physical viscosity of all components was increased by a factor of 10,000 to speed up the simulation by ensuring that the time step is limited by diffusion.


> **Figure 14: Schematic of initial condition for phase constrained diffusion between air-He-SF6.**


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq385.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq386.png)


> **Table 11: Thermodynamic state for intraphase diffusion between air and helium surrounded by SF6. For this problem only, viscosity values are artificially increased by a factor of 10000 to speed up convergence to the equilibrium state by making the solution diffusion dominated.**


> **Figure 15 shows the evolution of air, helium, and SF6 throughout time. Throughout the diffusion process there is no leakage of air/helium into the SF6 and the interface remains immiscible. Additionally, the intraphase diffusion model results in the pockets of air mixing with the helium and eventually converging to a constant state.**

4.3.3. Two-layer Richtmyer-Meshkov instability implosion This section contains the final multi-phase multi-component test of a two-layer Rightmyer-Meshkov implosion problem. This type of flow can be thought of as a model problem to represent many of the hydrodynamic physics present in Inertial Confinement Fusion (ICF). The initial condition for this case is

26


> **Figure 15: Time evolution of phase constrained diffusion between air-He-SF6. First row: 1D line plot of mass fraction profiles for air ( ), Helium ( ), and SF6 ( ) along the diagonal. Second row: air. Third row: Helium. Bottom row: SF6. Times from left to right: 0 ms, 2 ms, 6 ms, 41.5 ms.**

shown in Figure 16, where there are three materials, air, helium, and SF6, separated by perturbed material interfaces which will be simulated with a spatial resolution of (3200 × 3200). The initial thermodynamic state is defined in Table 12. In this simulation the air and helium are treated as one phase where intraphase mixing occurs, and the air/helium-SF6 interface is treated as immiscible. A Mach 1.22 shock is initialized in air and will first cross the air-helium interface and induce a RMI which will lead to intraphase diffusion between the air and helium. Later, the shock passes through the interface between the air-helium phase with the SF6 phase creating another RMI where the immiscibility condition is enforced. At later times the shock implodes at the center of the domain and reverses into a outward traveling pressure wave which breaks the existing RMI interface structure into a chaotic field. Figures 17 and 18 show the evolution of the mass-fraction of air and Schlieren, respectively. The evolution of the mass-fraction and Schlieren both show the regimes of the implosion problem, including the formation of the intraphase and interphase RMIs as well as the eventual transition to a chaotic state. In particular, Figure 17 shows the mixing of air and helium through the simulation, whereas the interface between air and SF6 remains immiscible due to

27

the multi-component extension of the CDI interface regularization model. The regimes of this simulation include interactions between shocks, immiscible interfaces, and multi-component mixing. Specifically, the shock remains stable and non-oscillatory by the ENO-type scheme, the SF6-air/helium interface remains immiscible due to the regularization terms, and the air and helium are mixed due to the phase-constrained diffusion model.


> **Figure 16: Schematic of initial condition for two-layer Rightmyer-Meshkov implosion between air-He-SF6.**


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq387.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq388.png)


> **Table 12: Initial conditions for two-layer RMI. The inward radial velocity is denoted ur**

.

5. Conclusions

We have presented a robust and concise computational framework to simulate multi-phase multi-component flows. A positivity-preserving ENO-type scheme is consistently designed with the thermal-mechanical assumptions of the multi-phase model to achieve oscillation free solutions for shocks and material interfaces. Specifically, with the proposed method the interface equilibrium condition (IEC) was discretely satisfied without requiring additional of redundant PDEs. Additionally, the conservative diffuse interface (CDI) method was generalized to multi-phase multi-component systems by assuming intraphase mixing was occurring between scalar fields confined to immiscible phases. The extended CDI model allows for the simultaneous representation of immiscible interfaces and intraphase mixing. Similar ideas were used to confine the physical species diffusion to individual phases without leakage across immiscible interfaces.

28


> **Figure 17: Mass fraction of air during two-layer Rightmyer-Meshkov implosion between air-He-SF6 at times 11.6µs, 18.7µs, 27.1µs, and 47µs.**

The proposed framework was used to simulate problems ranging in complexity, from single-phase to multi-phase multi-component flows. Single-phase tests included a 2D Riemann problem which showed the high-resolution capabilities of the proposed framework. Multi-component tests included a single-mode Richtmyer-Meshkov instability which verified the results of the proposed framework with those of former computational studies. The multi-phase tests showed the ability of the framework to handle the interaction of high-density immiscible phase interfaces with shocks. Furthermore, a simulation of a Mach 100 water column was completed without simulation failure, showing the usefulness of the positivity preserving procedure. Lastly, multiple multi-phase multi-component simulations showcased the interactions of the generalized CDI model, the intraphase species diffusion model, and the ENO-type shock capturing. In particular, a high-resolution ICF-like RMI implosion problem showed the ability of the framework to simulate complex multi-physics systems involving shocks, intraphase mixing, and interphase immiscibility. Future work includes incorporating phase change, surface tension, and reactions. Additionally, extensions

29


> **Figure 18: Numerical Schlieren � ln �||∇ρ|| ρ �� results for two-layer Rightmyer-Meshkov implosion between air-He-SF6 at times 11.6µs, 18.7µs, 27.1µs, and 47µs..**

of the equation of state to include non-linear equation of state dependence on temperature will be required to study combustion and other high-temperature related problems. The applicability of these additions will provide a path for high-resolution simulations of many engineering systems, including engines and rocket combustors.

Appendix A. Characteristic decomposition

The Euler equations (ignoring viscosity and interface regularization in Eq. 1) is defined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq389.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq390.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq391.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq392.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq393.png)

30

where it can be re-written as in terms of primitive variables, W = [T, Y c p , u, v, w, P], using a Jacobian matrix of conserved and primitive variables defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq394.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq395.png)

and flux Jacobian matrices defined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq396.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq397.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq398.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq399.png)

Now the quasi-linearized Euler equation in terms of primitive variables can be written as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq400.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq401.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq402.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq403.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq404.png)

where A = P −1Qx, B = P −1Qy, and C = P −1Qz are defined as,

A =





u . . . 0 ρa2β−1

α 0 0 0 ...0 ...u ...0 ...0 ...0 ...0 0 0 u 0 0 1 ρ 0 0 0 u 0 0 0 0 0 0 u 0 0 0 ρa2 0 0 u


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq405.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq406.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq407.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq408.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq409.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq410.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq411.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq412.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq413.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq414.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq415.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq416.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq417.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq418.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq419.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq420.png)

(A.5) where the (· · · ) represent the number of additional entries required to represent all but one Y c p species in the system. The primitive system can be diagonalized using a characteristic decomposition,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq421.png)

where, for A, the right-eigenvectors SA and left-eignevectors S−1 A are defined as,

SA =






![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq422.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq423.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq424.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq425.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq426.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq427.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq428.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq429.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq430.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq431.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq432.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq433.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq434.png)

Similar characteristic decomposition can be performed for B as,

SB =






![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq435.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq436.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq437.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq438.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq439.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq440.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq441.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq442.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq443.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq444.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq445.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq446.png)

(A.7)

and additionally for C,

SC =






![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq447.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq448.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq449.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq450.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq451.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq452.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq453.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq454.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq455.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq456.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq457.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq458.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq459.png)

31

With the definitions for the characteristic composition the projections to/from characteristic space in the Godunov algorithm described in Section 3.1.3 can be completed.

Appendix B. HLLC Riemann solver

To solve the Riemann problem at the cell-faces we use the approximate 1D HLLC Riemann solver. Extension to multi-dimensional flows is done using a dimension-by-dimension approach. As an example, the following section will define the 1D HLLC approximate Riemann solver in the x-direction. The wave speeds are defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq460.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq461.png)

and


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq462.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq463.png)

where the overbar represents the arithmetic average across the xi+1/2 interface. The intermediate wave speed is chosen as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq464.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq465.png)

Lastly, the intermediate conservative state is defined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq466.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq467.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq468.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq469.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq470.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq471.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq472.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq473.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq474.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq475.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq476.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq477.png)

With the wave speeds and the intermediate state defined the local Riemann problem at cell face xi+1/2 can be solved using the HLLC Riemann solver to determine the convective flux, Fi+1/2 = HLLC(U L i+1/2, U R i+1/2) where the HLLC flux can be defined symmetrically as,

HLLC(U L, U R) = 1 + sign(s∗)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq478.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq479.png)

Appendix C. IEC analysis with characteristic variables

As discussed in Section 3.1.2 the clearest way to satisfy IEC is by directly interpolating the set of primitive variables W = [T, Y c p , u, v, w, P]. Instead, in Section 4.2.2 the characteristic variables projected from the primitive variables W is shown to provide acceptable levels of IEC error (near machine precision for multiple WENO-type schemes tested). The analysis showing how interpolations based on the characteristic variables remains IEC is shown below. As an example, the characteristic variables in the x-direction based on W = [T, Y c p , u, v, w, P] is defined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq480.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq481.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq482.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq483.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq484.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq485.png)

=


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq486.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq487.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq488.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq489.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq490.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq491.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq492.png)

32

where the (·) represents the arithmetic (or Roe) average onto the face. These quantities are constant across the full stencil. Following the Godunov approach described in Section 3.1.3 ˜ W is projected using an ENOtype scheme to the left and right states across a cell-face. ˜ W L/R are projected to W L/R using,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq493.png)

To satisfy IEC after interpolation the following conditions must be met (assuming an initially constant T, P, and u), T = T L/R, P = P L/R, u = uL/R. (C.3)

First, the pressure and normal velocity conditions can be analyzed,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq494.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq495.png)

To show that pressure and velocity stay constant after ENO-type interpolation, we can note that both ˜w5 and ˜w6 are constant across an isothermal material interface by seeing that pressure, velocity, and the averaged quantity, (aρ)/2, are constant over the stencil. With both ˜wL/R 5 = ˜w5 and ˜wL/R 6 = ˜w6 we can see that P L/R = P and uL/R = u. The temperature on the cell-face is determined by,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq496.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq497.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq498.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq499.png)

=


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq500.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq501.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq502.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq503.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq504.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq505.png)

To see how temperature equilibrium is maintained, note that ˜w1 is also constant (across an isothermal interface), so ˜w(L/R) 1 = ˜w1 leading to,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq506.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq507.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq508.png)

where if the condition for P = P L/R is satisfied, T L/R = T. The analysis above proves that using the characteristic variables based on the primitive basis vector W satisfies IEC by keeping pressure, temperature, and velocity oscillation free across an isothermal material interface. The theoretical analysis was confirmed in practice as the results of the IEC test in Section 4.2.2 provided levels of error near machine precision for all fields.

Appendix D. Single phase tests

This Section contains three single-phase verification cases. The first is a two-dimensional vortex advection problem to verify the convergence of the numerical scheme in multiple dimensions. The second test is the standard Shu-Osher shock tube problem. The third is a common 2D Riemann problem to show the highresolution capabilities of the Godunov implementation described in Section 3.1.3.

Appendix D.1. Two-dimensional vortex advection To verify the implementation of the numerical scheme, a 2D vortex advection test was completed. Details on the initial condition for the test can be found in the following reference [92]. As mentioned in section 3.1 a 1st-order quadrature rule was used to construct the cell-face averages. This common approach keeps

33

the Godunov scheme at a reasonable cost, but reduces the order of accuracy to 2nd-order in multiple dimensions. Even so, the high-order ENO-type interpolations result in a low-dissipation and high-resolution scheme which, in practice, resembles the truly high-order counterpart at a far reduced cost. Figure D.19 shows the expected order of convergence for the vortex advection problem using the WENO5Z [96] scheme with the Godunov method.

Figure D.19: Convergence of WENO5Z scheme for advection of 2D vortex.

Appendix D.2. Shu-Osher shock tube Another standard single-phase test to show the applicability of the proposed scheme to handle shocks in single-phase flow is the Shu-Osher problem. The initial condition is given in Table D.13 with a spatial resolution of (200×1) and a constant time step of 4×10−3. The resulting density field is shown at t = 1.8 in Figure D.20 to compare the influence of WENO5JS [31], WENO5Z [96], and TENO6 [32] on the solution. As expected, the lowest dissipation scheme is TENO6 with a very close match to the high-resolution reference solution. Additionally, the improvements associated with the WENO5Z compared to the WENO5JS scheme are visually confirmed.


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq509.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq510.png)

Table D.13: Initial conditions for 1D multi-phase multi-component Shu Osher

Appendix D.3. 2D Riemann problem The final single-phase test is a two-dimensional Riemann problem [101, 99] which will showcase the highresolution capability of the proposed scheme. The initial conditions for the simulation are listed in Table D.14 for a domain of size [0, 2] × [0, 2] with a spatial resolution of (2000 × 2000). Figure D.21 compares the density field for four schemes at a non-dimensional time t = 1.1 over a subset of the domain defined by [0, 1.2] × [0, 1.2]. The test case is without physical viscosity (Re = ∞), so small-scale structures are only destroyed by numerical dissipation. Figure D.21 shows the expected trends of more small-scale features for the lower dissipation TENO-type schemes compared to the WENO-type. Even so, all schemes showcase high-resolution solutions with both oscillation-free shocks and a richness of fine-scale structures.

34

Figure D.20: Shu-Osher problem for (a) density and (b) density zoomed. Scheme comparisons for WENO5JS, WENO5Z, and TENO6 using Godunov scheme.


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq511.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq512.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq513.png)

11 0.0 0.3 0.0 1.4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq514.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq515.png)

11 9/310 0.0 1.4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq516.png)

11 0.3 0.0 1.4

Table D.14: Initial conditions for the 2D Riemann problem

Appendix E. Mixing rule definitions and identities

Conversions between mass-fraction, density, and volume fraction:

1. �

p �

c Y c p = 1

2. �

c Y c p = Yp. Recall, c represents the components of phase p.


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq517.png)

4. �

p �

c ρY c p = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq518.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq519.png)

5. ρY c p = ρpϕpY c p /Yp

Relations between mass-fraction, molar-fraction, and molecular weight:

1. W = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq520.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq521.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq522.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq523.png)

2. Wp = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq524.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_ENO_v1_eq525.png)

3. Xp c = Y p c W/W c p


## References

[1] S. Mirjalili, S. S. Jain, M. Dodd, Interface-capturing methods for two-phase flows: An overview and recent developments, Center for Turbulence Research Annual Research Briefs 2017 (117-135) (2017) 13. [2] H. Luo, J. D. Baum, R. L¨ohner, On the computation of multi-material flows using ale formulation, Journal of Computational Physics 194 (1) (2004) 304–328.

35

Figure D.21: Riemann problem comparing density fields using the high-resolution Godunov method with TENO5 [32], TENO6 [32], WENO5JS [31], and WENO5Z [96] schemes. .

[3] J. Glimm, X. Li, Y. Liu, Z. Xu, N. Zhao, Conservative front tracking with improved accuracy, SIAM Journal on Numerical Analysis 41 (5) (2003) 1926–1947. [4] J.-P. Cocchi, R. Saurel, A riemann problem based method for the resolution of compressible multimaterial flows, Journal of Computational Physics 137 (2) (1997) 265–298. [5] J. Glimm, J. W. Grove, X. L. Li, K.-M. Shyue, Y. Zeng, Q. Zhang, Three-dimensional front tracking, SIAM Journal on Scientific Computing 19 (3) (1998) 703–727. [6] S. McKee, M. F. Tom´e, V. G. Ferreira, J. A. Cuminato, A. Castelo, F. Sousa, N. Mangiavacchi, The mac method, Computers & Fluids 37 (8) (2008) 907–930. [7] H. Terashima, G. Tryggvason, A front-tracking/ghost-fluid method for fluid interfaces in compressible flows, Journal of Computational Physics 228 (11) (2009) 4012–4037. [8] W. Mulder, S. Osher, J. A. Sethian, Computing interface motion in compressible gas dynamics, Journal of computational physics 100 (2) (1992) 209–228. [9] R. P. Fedkiw, T. Aslam, B. Merriman, S. Osher, A non-oscillatory eulerian approach to interfaces in multimaterial flows (the ghost fluid method), Journal of computational physics 152 (2) (1999) 457–492. [10] T. Liu, B. Khoo, K. Yeo, Ghost fluid method for strong shock impacting on material interface, Journal of computational physics 190 (2) (2003) 651–681. [11] X. Y. Hu, B. C. Khoo, An interface interaction method for compressible multifluids, Journal of Computational Physics 198 (1) (2004) 35–64. [12] R. Chiodi, O. Desjardins, A reformulation of the conservative level set reinitialization equation for accurate and robust simulation of complex multiphase flows, Journal of Computational Physics 343 (2017) 186–200. [13] R. DeBar, Fundamentals of the kraken code, Technical Report UCIR-760. [14] R. Scardovelli, S. Zaleski, Interface reconstruction with least-square fit and split eulerian–lagrangian advection, International Journal for Numerical Methods in Fluids 41 (3) (2003) 251–274. [15] E. Aulisa, S. Manservisi, R. Scardovelli, S. Zaleski, A geometrical area-preserving volume-of-fluid advection method, Journal of Computational Physics 192 (1) (2003) 355–364.

36

[16] L. Jofre, O. Lehmkuhl, J. Castro, A. Oliva, A 3-d volume-of-fluid advection method based on cell-vertex velocities for unstructured meshes, Computers & Fluids 94 (2014) 14–29. [17] J. L´opez, J. Hern´andez, P. G´omez, F. Faura, A volume of fluid method based on multidimensional advection and spline interface reconstruction, Journal of Computational Physics 195 (2) (2004) 718–742. [18] M. Owkes, O. Desjardins, A computational framework for conservative, three-dimensional, unsplit, geometric transport with application to the volume-of-fluid (vof) method, Journal of Computational Physics 270 (2014) 587–612. [19] C. B. Ivey, P. Moin, Conservative and bounded volume-of-fluid advection on unstructured grids, Journal of Computational Physics 350 (2017) 387–419. [20] S. Mirjalili, C. B. Ivey, A. Mani, Comparison between the diffuse interface and volume of fluid methods for simulating two-phase flows, International Journal of Multiphase Flow 116 (2019) 221–238. [21] E. Olsson, G. Kreiss, A conservative level set method for two phase flow, Journal of computational physics 210 (1) (2005) 225–246. [22] O. Ubbink, R. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, Journal of computational physics 153 (1) (1999) 26–50. [23] D. Zhang, C. Jiang, D. Liang, Z. Chen, Y. Yang, Y. Shi, A refined volume-of-fluid algorithm for capturing sharp fluid interfaces on arbitrary meshes, Journal of Computational Physics 274 (2014) 709–736. [24] B. Xie, S. Ii, F. Xiao, An efficient and accurate algebraic interface capturing method for unstructured grids in 2 and 3 dimensions: The thinc method with quadratic surface representation, International Journal for Numerical Methods in Fluids 76 (12) (2014) 1025–1042. [25] B. Van Leer, Towards the ultimate conservative difference scheme. ii. monotonicity and conservation combined in a second-order scheme, Journal of computational physics 14 (4) (1974) 361–370. [26] P. L. Roe, Characteristic-based schemes for the euler equations, Annual review of fluid mechanics 18 (1) (1986) 337–365. [27] G. D. Van Albada, B. Van Leer, W. Roberts Jr, A comparative study of computational methods in cosmic gas dynamics, in: Upwind and high-resolution schemes, Springer, 1997, pp. 95–103. [28] A. Harten, S. Osher, Uniformly high-order accurate nonoscillatory schemes. i, SIAM Journal on Numerical Analysis 24 (2) (1987) 279–309. [29] C.-W. Shu, S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, Journal of computational physics 77 (2) (1988) 439–471. [30] C.-W. Shu, S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, ii, Journal of computational physics 83 (1) (1989) 32–78. [31] G.-S. Jiang, C.-W. Shu, Efficient implementation of weighted eno schemes, Journal of computational physics 126 (1) (1996) 202–228. [32] L. Fu, X. Y. Hu, N. A. Adams, A family of high-order targeted eno schemes for compressible-fluid simulations, Journal of Computational Physics 305 (2016) 333–359. [33] E. Johnsen, T. Colonius, Implementation of weno schemes in compressible multicomponent flow problems, Journal of Computational Physics 219 (2) (2006) 715–732. [34] V. Coralic, T. Colonius, Finite-volume weno scheme for viscous compressible multicomponent flows, Journal of computational physics 274 (2014) 95–121. [35] H. Collis, S. Mirjalili, S. Jain, A. Mani, Assessment of weno and teno schemes for the four-equation compressible two-phase flow model with regularization terms, Center for Turbulence Research Annual Research Briefs (2022) 151–165. [36] M. L. Wong, J. B. Angel, C. C. Kiris, A positivity-preserving eulerian two-phase approach with thermal relaxation for compressible flows with a liquid and gases, arXiv preprint arXiv:2208.04488. [37] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, International journal for numerical methods in fluids 48 (9) (2005) 1023–1040. [38] F. Xiao, S. Ii, C. Chen, Revisit to the thinc scheme: a simple algebraic vof algorithm, Journal of Computational Physics 230 (19) (2011) 7086–7092. [39] K.-M. Shyue, F. Xiao, An eulerian interface sharpening algorithm for compressible two-phase flow: The algebraic thinc approach, Journal of Computational Physics 268 (2014) 326–354. [40] D. A. Bezgin, A. B. Buhendwa, N. A. Adams, Jax-fluids 2.0: Towards hpc for differentiable cfd of compressible two-phase flows, 2024, URL https://arxiv. org/abs/2402.05193. [41] J. W. Cahn, J. E. Hilliard, Free energy of a nonuniform system. i. interfacial free energy, The Journal of chemical physics 28 (2) (1958) 258–267. [42] S. M. Allen, J. W. Cahn, A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening, Acta metallurgica 27 (6) (1979) 1085–1095. [43] Z. Huang, G. Lin, A. M. Ardekani, A consistent and conservative phase-field method for multiphase incompressible flows, Journal of Computational and Applied Mathematics 408 (2022) 114116. [44] Z. Huang, E. Johnsen, Bound preservation for the consistent and conservative phase-field method for compressible single-, two-, and n-phase flows, Journal of Computational Physics (2025) 113783. [45] P. H. Chiu, Y. T. Lin, A conservative phase field method for solving incompressible two-phase flows, Journal of Computational Physics 230 (2011) 185–204. doi:10.1016/J.JCP.2010.09.021. [46] S. Mirjalili, C. B. Ivey, A. Mani, A conservative diffuse interface method for two-phase flows with provable boundedness properties, Journal of Computational Physics 401 (2020) 109006. doi:10.1016/J.JCP.2019.109006. [47] S. S. Jain, A. Mani, P. Moin, A conservative diffuse-interface method for compressible two-phase flows, Journal of Computational Physics 418 (2020) 109606. doi:10.1016/J.JCP.2020.109606. [48] S. Mirjalili, A. Mani, A conservative second order phase field model for simulation of n-phase flows, Journal of Compu-

37

tational Physics 498 (2024) 112657. [49] H. Collis, S. Mirjalili, A. Mani, Diffuse interface treatment in generalized curvilinear coordinates with grid-adapting interface thickness, arXiv preprint arXiv:2411.18770. [50] S. S. Jain, M. C. Adler, J. R. West, A. Mani, P. Moin, S. K. Lele, Assessment of diffuse-interface methods for compressible multiphase fluid flows and elastic-plastic deformation in solids, Journal of Computational Physics 475 (2023) 111866. [51] I. Kataoka, Local instant formulation of two-phase flow, International Journal of Multiphase Flow 12 (5) (1986) 745–758. [52] K.-M. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, Journal of Computational Physics 142 (1) (1998) 208–242. [53] A. W. Cook, Enthalpy diffusion in multicomponent flows, Physics of Fluids 21 (5). [54] T. Fl˚atten, A. Morin, T. Munkejord, On solutions to equilibrium problems for systems of stiffened gases *, Society for Industrial and Applied Mathematics 71 (2011) 41–67. doi:10.1137/100784321. URL http://www.siam.org/journals/siap/71-1/78432.html [55] S. LeMartelot, R. Saurel, O. Le M´etayer, Steady one-dimensional nozzle flow solutions of liquid–gas mixtures, Journal of fluid mechanics 737 (2013) 146–175. [56] S. Le Martelot, R. Saurel, B. Nkonga, Towards the direct numerical simulation of nucleate boiling flows, International Journal of Multiphase Flow 66 (2014) 62–78. [57] R. Saurel, P. Boivin, O. Le M´etayer, A general formulation for cavitating, boiling and evaporating flows, Computers & Fluids 128 (2016) 53–64. [58] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, Journal of Computational Physics 181 (2) (2002) 577–616. [59] G.-S. Yeom, K.-S. Chang, A modified hllc-type riemann solver for the compressible six-equation two-fluid model, Computers & Fluids 76 (2013) 86–104. [60] O. Haimovich, S. H. Frankel, Numerical simulations of compressible multicomponent and multiphase flow using a highorder targeted eno (teno) finite-volume method, Computers & Fluids 146 (2017) 105–116. [61] M. R. Baer, J. W. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials, International journal of multiphase flow 12 (6) (1986) 861–889. [62] S. A. Beig, E. Johnsen, Maintaining interface equilibrium conditions in compressible multiphase flows using interface capturing, Journal of Computational Physics 302 (2015) 548–566. [63] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, Journal of Computational Physics 125 (1) (1996) 150–160. [64] R. Abgrall, S. Karni, Computations of compressible multifluids, Journal of computational physics 169 (2) (2001) 594–623. [65] P. C. Ma, Y. Lv, M. Ihme, An entropy-stable hybrid scheme for simulations of transcritical real-fluid flows, Journal of Computational Physics 340 (2017) 330–357. [66] E. Johnsen, F. Ham, Preventing numerical errors generated by interface-capturing schemes in compressible multi-material flows, Journal of Computational Physics 231 (17) (2012) 5705–5717. [67] H. Terashima, S. Kawai, M. Koshi, Consistent numerical diffusion terms for simulating compressible multicomponent flows, Computers & Fluids 88 (2013) 484–495. [68] T. Nonomura, K. Fujii, Characteristic finite-difference weno scheme for multicomponent compressible fluid analysis: Overestimated quasi-conservative formulation maintaining equilibriums of velocity, pressure, and temperature, Journal of Computational Physics 340 (2017) 358–388. [69] Y. Fujiwara, Y. Tamaki, S. Kawai, Fully conservative and pressure-equilibrium preserving scheme for compressible multicomponent flows, Journal of Computational Physics 478 (2023) 111973. [70] H. Terashima, N. Ly, M. Ihme, Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible multi-species and real-fluid interfacial flows, Journal of Computational Physics 524 (2025) 113701. [71] P. Yi, S. Yang, C. Habchi, R. Lugo, A multicomponent real-fluid fully compressible four-equation model for two-phase flow with phase change, Physics of Fluids 31 (2). [72] T. Coffee, J. Heimerl, Transport algorithms for premixed, laminar steady-state flames, Combustion and Flame 43 (1981) 273–289. [73] S. Mirjalili, S. S. Jain, A. Mani, A computational model for interfacial heat and mass transfer in two-phase flows using a phase field method, International Journal of Heat and Mass Transfer 197 (2022) 123326. [74] S. Mirjalili, M. Khanwale, A. Mani, Consistent modeling of scalar transport in multiphase flows using conservative phase field methods, 2022. [75] C. R. Wilke, A viscosity equation for gas mixtures, Journal of Chemical physics 18 (4) (1950) 517–519. [76] A. Chiapolino, P. Boivin, R. Saurel, A simple and fast phase transition relaxation solver for compressible multicomponent two-phase flows, Computers & Fluids 150 (2017) 31–45. doi:https://doi.org/10.1016/j.compfluid.2017.03.022. [77] O. Le M´etayer, R. Saurel, The noble-abel stiffened-gas equation of state, Physics of Fluids 28 (4). [78] E. F. Toro, Riemann solvers and numerical methods for fluid dynamics: a practical introduction, Springer Science & Business Media, 2013. [79] S. S. Jain, P. Moin, A kinetic energy–and entropy-preserving scheme for compressible two-phase flows, Journal of Computational Physics 464 (2022) 111307. doi:10.1016/J.JCP.2022.111307. [80] C.-W. Shu, T. A. Zang, G. Erlebacher, D. Whitaker, S. Osher, High-order eno schemes applied to two-and threedimensional compressible flow, Applied Numerical Mathematics 9 (1) (1992) 45–71. [81] S. Mirjalili, A. Mani, Consistent, energy-conserving momentum transport for simulations of two-phase flows using the phase field equations, Journal of Computational Physics 426. doi:10.1016/j.jcp.2020.109918. [82] R. K. Shukla, Nonlinear preconditioning for efficient and accurate interface capturing in simulation of multicomponent

38

compressible flows, Journal of Computational Physics 276 (2014) 508–540. [83] T. Wac�lawczyk, A consistent solution of the re-initialization equation in the conservative level-set method, Journal of Computational Physics 299 (2015) 487–525. [84] S. S. Jain, Accurate conservative phase-field method for simulation of two-phase flows, Journal of Computational Physics 469 (2022) 111529. doi:10.1016/J.JCP.2022.111529. [85] Y. Kuya, S. Kawai, High-order accurate kinetic-energy and entropy preserving (keep) schemes on curvilinear grids, Journal of Computational Physics 442 (2021) 110482. doi:10.1016/J.JCP.2021.110482. [86] B. P´eden, J. Carmona, P. Boivin, T. Schmitt, B. Cuenot, N. Odier, Numerical assessment of diffuse-interface method for air-assisted liquid sheet simulation, Computers & Fluids 266 (2023) 106022. [87] A. Baumgart, G. Blanquart, Ensuring � s Ys = 1 in transport of species mass fractions, Journal of Computational Physics 513 (2024) 113199. doi:https://doi.org/10.1016/j.jcp.2024.113199. URL https://www.sciencedirect.com/science/article/pii/S0021999124004480 [88] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM review 43 (1) (2001) 89–112. [89] K. W. Thompson, Time-dependent boundary conditions for hyperbolic systems, ii, Journal of computational physics 89 (2) (1990) 439–461. [90] T. J. Poinsot, S. Lelef, Boundary conditions for direct simulations of compressible viscous flows, Journal of computational physics 101 (1) (1992) 104–129. [91] N. Okong’o, J. Bellan, Consistent boundary conditions for multicomponent real gas mixtures based on characteristic waves, Journal of Computational Physics 176 (2) (2002) 330–344. [92] M. D. Renzo, L. Fu, J. Urzay, Htr solver: An open-source exascale-oriented task-based multi-gpu high-order code for hypersonic aerothermodynamics, Computer Physics Communications 255 (2020) 107262. doi:10.1016/J.CPC.2020. 107262. [93] M. Di Renzo, S. Pirozzoli, Htr-1.2 solver: Hypersonic task-based research solver version 1.2, Computer Physics Communications 261 (2021) 107733. [94] M. Di Renzo, Htr-1.3 solver: Predicting electrified combustion using the hypersonic task-based research solver, Computer Physics Communications 272 (2022) 108247. [95] M. Bauer, S. Treichler, E. Slaughter, A. Aiken, Legion: Expressing locality and independence with logical regions, in: SC’12: Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis, IEEE, 2012, pp. 1–11. [96] R. Borges, M. Carmona, B. Costa, W. S. Don, An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws, Journal of computational physics 227 (6) (2008) 3191–3211. [97] J.-F. Haas, B. Sturtevant, Interaction of weak shock waves with cylindrical and spherical gas inhomogeneities, Journal of Fluid Mechanics 181 (1987) 41–76. [98] N. Hoppe, J. M. Winter, S. Adami, N. A. Adams, Alpaca-a level-set based sharp-interface multiresolution solver for conservation laws, Computer Physics Communications 272 (2022) 108246. [99] N. Fleischmann, S. Adami, N. A. Adams, Numerical symmetry-preserving techniques for low-dissipation shock-capturing schemes, Computers & Fluids 189 (2019) 94–107. [100] J. Meng, T. Colonius, Numerical simulations of the early stages of high-speed droplet breakup, Shock waves 25 (4) (2015) 399–414. [101] C. W. Schulz-Rinne, J. P. Collins, H. M. Glaz, Numerical solution of the riemann problem for two-dimensional gas dynamics, SIAM Journal on Scientific Computing 14 (6) (1993) 1394–1414.

39

