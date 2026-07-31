
## A thermodynamically consistent and robust four-equation model for multi-phase multi-component compressible flows using ENO-type schemes including interface regularization


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq001.png)

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


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq002.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq003.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq004.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq005.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq006.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq007.png)

for p = 1, 2 and 1 ≤c ≤N. The convective fluxes F, G, H are defined as,

4


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq008.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq009.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq010.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq011.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq012.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq013.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq014.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq015.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq016.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq017.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq018.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq019.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq020.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq021.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq022.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq023.png)

and the viscous fluxes, Fν, Gν, Hν can be expressed as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq024.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq025.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq026.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq027.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq028.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq029.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq030.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq031.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq032.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq033.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq034.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq035.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq036.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq037.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq038.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq039.png)

where τij is the viscous stress tensor,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq040.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq041.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq042.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq043.png)

with µ as the dynamic viscosity, and δij as the Kronecker delta. Additionally, the intraphase species mass diffusion vector can be derived for non-dilute species diffusion from the Stefan-Maxwell diffusion model (with certain equilibrium assumptions) as a Fickian diffusion term and a mass corrector [72]. This well-known model for non-dilute species diffusion can be extended to the multi-phase multi-component context using a confined scalar argument. The resulting model can be written as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq044.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq045.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq046.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq047.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq048.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq049.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq050.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq051.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq052.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq053.png)

ln


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq054.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq055.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq056.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq057.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq058.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq059.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq060.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq061.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq062.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq063.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq064.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq065.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq066.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq067.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq068.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq069.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq070.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq071.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq072.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq073.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq074.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq075.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq076.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq077.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq078.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq079.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq080.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq081.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq082.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq083.png)

where Dc p is the mass diffusivity of component c within phase p, Xc p is the mixture molar fraction of component c of phase p, Xp is the mixture molar fraction of phase p, Yp is the mixture mass fraction of phase p, W c p is the molecular weight of component c, and Wp is the molecular weight of phase p. Formulas describing the computation of Wp, Yp, and Xp are in Appendix E. One key difference between this formulation and the common non-dilute Stefan-Maxwell model is the incorporation of the denominators (Xp and Yp) in the gradient terms. This restricts the volume of influence of species diffusion to be within each phase and prevents unphysical leakage of components across phases. This formulation can be derived using the consistent transport models for confined scalars introduced in previous work [73, 74]. Any physical exchange across phases (e.g. phase change) would require additional models to explicitly represent mass-transfer between phases. Lastly, the heat flux qi is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq084.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq085.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq086.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq087.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq088.png)

where λ is the heat conductivity of the mixture and hc p is the specific enthalpy of component c of phase p. The final terms in Eq. 1, FDI, GDI, HDI, are the interface regularization terms (i.e. the diffuse interface terms) and are described in Section 3.2.

5

2.2. Mixture rules

2.2.1. Equilibrium assumptions In this work, thermal and mechanical equilibrium is assumed between phases within a computational element by enforcing that one temperature, pressure, and velocity vector is shared between phases. In the literature, models which enforce thermo-mechanical equilibrium are often known as the four-equation multi-phase model [51, 52, 53, 54, 55, 56, 57]. One of the main contributions of this work is providing a computational strategy to satisfy the discrete interface equilibrium condition near machine precision with the four-equation model. The proposed scheme is presented in Section 3.1. The following sections overview the mixing rules required to obtain the equation of state closure for Eq. 1. Section 2.2.2 starts by defining the mixing rules describing multi-phase mixtures and Section 2.2.3 extends this to a general multi-phase multi-component model.

2.2.2. Interphase mixing rules In this work, multi-phase mixtures are assumed to be immiscible. To describe immiscible phases, separate phases are assumed to occupy their own individual volumes and share a common pressure (Amagat’s law). The mixing rules are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq089.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq090.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq091.png)

where Tp, Pp, ep, and vp are the temperature, pressure, internal energy, and specific volume for phase p. These thermodynamic variables can be obtained from the equation of states for each phase p. These mixing rules allow for each phase p to be governed by different equation of states as well as by unique intraphase mixing rules to describe multi-component mixtures. Other quantities that are treated with interphase mixing rules include the mixture viscosity µ and mixture thermal conductivity λ as, µ = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq092.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq093.png)

where µp and λp are the phasic viscosity and thermal conductivity respectively, and ϕp is the phase volume fraction. In this work λp and µp are assumed constant in the liquid phase. For the gaseous phase, λg is defined as, λg = µg

Pr


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq094.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq095.png)

and µg is defined using Wilke’s rule [75] as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq096.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq097.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq098.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq099.png)

where,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq100.png)

8


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq101.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq102.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq103.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq104.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq105.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq106.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq107.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq108.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq109.png)

and the Prandtl number, Pr, the dynamic viscosity of species µi g, and the specific heats, CP c g, are assumed constant. Although not explored in this work, the formulation does not restrict these quantities from being extended to more advanced definitions, including defining the components with a non-linear dependence on temperature.

6

2.2.3. Intraphase mixing rules Classical intraphase mixing rules include assuming the individual components within a phase occupy their own volume (Amagat’s law) or assuming components share volumes (Dalton’s law). Traditionally, mixtures of gas components are assumed to follow Dalton’s law, with each individual component contributing a partial pressure towards the mixture pressure [76]. For a real gas equation of state (e.g. including the compressiblity factor), Dalton’s law and Amagat’s law will give different results. However, if the gas components are modeled as ideal gases, both representations obtain equivalent thermodynamic states [76]. The governing equations in Section 2.1 are written to allow either intraphase mixing rule to be applied. In this work, though the gas components will follow the ideal gas law, both mixing rules will be summarized to show the generality of the four-equation model and provide a background for extensions to more complex equations of state. The ideal mixing rules associated with assuming all intraphase components occupy the same volume (Dalton’s law) are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq110.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq111.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq112.png)

where (assuming the ideal gas law) the partial pressure of component c is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq113.png)

where R is the universal gas constant and W c is the molecular weight of component c. Additionally, the ideal mixing rules associated with assuming all components occupy the individual volumes (Amagat’s law) are summarized as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq114.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq115.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq116.png)

Lastly, the mass diffusivity of component c within phase p is defined as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq117.png)

where, in this work, a constant Schmidt number, Scc p, is assumed for each component within a phase.

2.3. Noble Abel stiffened gas EOS To close the system illustrated in Eq. 1 the Nobel-Abel equation of state (NASG EOS) [77] is used for all components in the multi-phase multi-component mixture. For a pure component, the general NASG EOS assuming constant heat capacity reads,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq118.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq119.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq120.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq121.png)

where for component c in phase p, CP c p is the heat capacity at constant pressure, Cvc p is the heat capacity at constant volume, γc p is the heat capacity ratio, bc p is the co-volume of the component molecules, P ∞c p is the stiffened pressure to model the attractive forces between molecules in a material, and qc p is a reference energy. If bc p and P ∞c p are taken as zero (as is done for components in the gas phase) the NASG EOS reduces to the ideal gas equation of state. During a simulation, the internal energy and volume (or density) of the mixture can be determined from the conserved vector U and the mixing rules in Section 2.2. To obtain expressions for the pressure and

7

temperature of the mixture in terms of internal energy and volume, the mixing rules discussed in Section 2.2 can be combined with the NASG EOS. For a system with multiple components in both phases, an iterative procedure is required to find the pressure and temperature that satisfy the equilibrium conditions[53]. In this work, only one component will be modeled in the liquid state, and the gaseous state will be composed of ideal gas components. Given these simplifications, we can use Amagat’s law for both intraphase and interphase mixing and the NASG EOS to obtain closed expressions for the mixture pressure. An expression for the mixture pressure can be found by enforcing the equality between the two definitions for mixture temperature found from Eq. 18,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq122.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq123.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq124.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq125.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq126.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq127.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq128.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq129.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq130.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq131.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq132.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq133.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq134.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq135.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq136.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq137.png)

Solving for a common pressure between Eq. 19 and 20 leads to


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq138.png)

with


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq139.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq140.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq141.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq142.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq143.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq144.png)

and Cv = �


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq145.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq146.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq147.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq148.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq149.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq150.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq151.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq152.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq153.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq154.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq155.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq156.png)

where for this work p = 1 is defined as the liquid phase and P ∞≡P ∞c=1 p=1 as there is only one component in the liquid. Once pressure is determined using Eq. 21, either Eq. 19 or Eq. 20 can be used to obtain the mixture temperature. For mixtures with multiple components in each phase or specific heats dependent on temperature, iterative solvers can be used to determine the equilibrium state [53]. Unless otherwise specified, the NASG EOS parameters used for all materials throughout this work are included in Table 1.

Material CP [J kg−1K−1] γ [-] q [J kg−1] b [kg−1] P∞[Pa]

Water (Liquid) 4.185 × 103 1.0123 −1.143 × 106 9.203 × 10−4 1.835 × 108

Air 1.011 × 103 1.4 0 0 0 Helium 5.091 × 103 1.66 0 0 0 SF6 0.661 × 103 1.093 0 0 0


> **Table 1: NASG Parameters used in this work**

8

3. Numerical framework

3.1. Spatial discretization

In this work, the integral form of the partial differential equations in Eq. 1 will be solved using the finite-volume method. The differential form of Eq. 1 is assumed to admit smooth solutions for which partial derivatives exist. A general cuboid cell (i, j, k) has the spatial width defined by ∆x, ∆y, and ∆z and a cell volume defined as ∆x∆y∆z. In this work, isotropic Cartesian grids (∆x = ∆y = ∆z) will be used to spatially discretize the system. In the finite-volume formulation the cell averaged values of the conserved variables can be obtained as,


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq157.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq158.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq159.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq160.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq161.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq162.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq163.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq164.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq165.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq166.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq167.png)

Applying the volume integration to Eq. 1 leads to a set of ordinary differential equations,

d dt ¯Ui,j,k = −1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq168.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq169.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq170.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq171.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq172.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq173.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq174.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq175.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq176.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq177.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq178.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq179.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq180.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq181.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq182.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq183.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq184.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq185.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq186.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq187.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq188.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq189.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq190.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq191.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq192.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq193.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq194.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq195.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq196.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq197.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq198.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq199.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq200.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq201.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq202.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq203.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq204.png)

2


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq205.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq206.png)

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


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq207.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq208.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq209.png)

2

(a) Define the right eigenvector, S from the averaged state of Ui+ 1

2 (b) W L i+ 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq210.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq211.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq212.png)

2

4. Obtain the left and right states for density using the EOS

5. Find the left and right states of the conserved variables from primitives

6. Find the cell-face flux using a HLLC Riemann solver

(a) Fi+ 1


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq213.png)


![Equation](images/2025_Collis_Bezgin_Mirjalili_Mani_four_equation_thermodynamic_ENO_eq214.png)

2 )

10

