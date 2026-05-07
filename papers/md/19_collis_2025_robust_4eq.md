
## A robust four-equation model for compressible multi-phase multi-component flows satisfying interface equilibrium and phase-immiscibility conditions

Henry Collisa,∗, Deniz A. Bezginb, Shahab Mirjalilic,a, Ali Mania

aDepartment of Mechanical Engineering, Stanford, CA 94305, USA bTechnical University of Munich, School of Engineering and Design, Chair of Aerodynamics and Fluid Mechanics, Boltzmannstraße 15, 85748 Garching bei M¨unchen, Germany cFLOW, Department of Engineering Mechanics, KTH Royal Institute of Technology, SE-10044 Stockholm, Sweden

Abstract

In this work, a robust computational method with the four-equation model is proposed to simulate compressible multi-phase multi-component flows. An ENO-type numerical scheme is designed to be consistent with the thermodynamic equilibrium assumptions of the four-equation multi-phase model, discretely enforcing the interface equilibrium condition — preventing numerical oscillations in pressure, velocity, and temperature around isothermal material interfaces. Critically, the proposed numerical method for the four-equation model accomplishes this without requiring explicit equations for volume fraction or other redundant transport equations for variables including mixture equation of state parameters, as is commonly done for the five-equation model. Additionally, consistent mixing rules are used to derive a non-dilute species diffusion model and thereby extend the conservative diffuse interface (CDI) model to multi-component systems. Together, these models prevent unphysical numerical leakage of species across phase interfaces. The presented test cases show that this consistent numerical method is equally applicable for regimes ranging from singlephase to multi-phase multi-component flows without retuning numerical parameters. When augmented with an existing positivity-preserving limiter for handling compressible multiphase flows, we show that the proposed computational method can robustly handle extreme conditions including strong shock-interface interactions, within the four-equation modeling framework. The proposed models and numerical schemes are implemented in the highly parallel Hypersonic Task based Research (HTR) Solver, and high-resolution simulations are performed using both CPUs and GPUs.

Keywords: ENO-type, compressible, diffuse interface, four-equation model, multicomponent, multiphase

1. Introduction

The present work is motivated by the prevalence of multi-phase compressible flows in nature and industrial applications. These flows commonly feature interactions between shocks and material interfaces and create challenges for numerical methods. These challenges include obtaining a discrete representation of discontinuities while remaining conservative in total mass, momentum, and energy. Designing a robust scheme near discontinuities while retaining a high-resolution solution in other regions is critical to properly represent prevalent physics in many flow regimes, including turbulence. Current techniques for numerical simulation of interfaces can be broadly placed in two categories: interface tracking and interface capturing methods [1].

∗Corresponding author Email addresses: hcollis@stanford.edu (Henry Collis), deniz.bezgin@tum.de (Deniz A. Bezgin), msey@kth.se (Shahab Mirjalili), alimani@stanford.edu (Ali Mani)

Preprint submitted to Journal of Computational Physics July 10, 2025

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

1.1. Interface tracking Interface tracking methods discretely represent interfaces as sharp discontinuities, where the interface is tracked using a Lagrangian representation. Some methods include Arbitrary Lagrangian-Eulerian (ALE) [2], front-tracking [3, 4, 5], marker-and-cell (MAC) [6], and certain approaches to ghost-fluid schemes [7]. Advantages of the sharp interface include the representation of drastically different equations of state for each fluid while avoiding spurious oscillations across phase boundaries. Although interface tracking has its advantages, achieving discrete conservation is difficult and an ongoing area of research. In addition, the complexity of these methods increases when representing more complex interfacial systems, including systems with strong interfacial deformations or topological changes, high density ratios, and interactions between shocks and material interfaces.

1.2. Interface capturing Interface capturing schemes are based on an Eulerian equation to represent the location of the interface throughout time. The formulation can represent interfaces as both sharp and diffuse, where the sharp-interface methods include standard level-set/ghost-fluid schemes, [8, 9, 10, 11, 12] and geometric volume-of-fluid (VOF) methods [13, 14, 15, 16, 17, 18, 19]. The sharp interface achieves similar advantages as the interface tracking method, but the accurate geometrical representation of the interface throughout simulations without losing conservation generally becomes complex, computationally expensive, and introduces issues for scalability [20]. On the other hand, interface capturing methods including conservative level-set (CLS) [21], algebraic VOF methods [22, 23, 24], and diffuse interface methods avoid the additional complexity of the sharp-interface representations by diffusing the interface over a finite number of discrete cells. The diffuse interface methods, in particular, have become a popular method due to their simplicity of implementation and load balancing, conservation guarantees, and applicability for large density ratio flows.

1.2.1. Implicit interface capturing A common diffuse interface approach uses the dissipation of a numerical scheme to implicitly represent interfaces. These include total-variational-diminishing (TVD) schemes which will keep all interfaces bounded and stable, though are generally excessively diffusive and limited to low-order accuracy. These include fluxlimiter schemes such as minmod, van Leer, and Superbee [25, 26, 27]. High-order analogies of the TVD schemes include essentially non-oscillatory (ENO) type schemes [28, 29, 30, 31, 32]. ENO-type schemes (WENO and TENO) use smoothness indicators to locally switch between lower-dissipation high-order and higher-dissipation low-order stencils to capture interfaces without adding substantial dissipation far from discontinuities. These techniques have traditionally been used as shock-capturing schemes for compressible flows. When ENO-type schemes are used to model immiscible interfaces, they have sometimes been referred to as interface capturing methods in analogy to their shock-capturing property [33, 34]. As described above, this terminology is also used to broadly refer to Eulerian methods for representation of interfaces, such as VOF and level-set methods. To avoid ambiguity, the term implicit interface capturing will be used in the remainder of this work to describe methods that rely on numerical dissipation, like WENO, to provide a diffuse representation of interfaces. An advantage of many implicit interface capturing methods is their applicability to handle regions of sharp gradients due to their localized upwind-biased numerical discretization. Even so, when implicit interface capturing schemes are applied to high-Mach, high-density ratio multi-phase flow (even if combined with a phase field model [35]), the small oscillations produced by the scheme can lead to simulation failure due to the development of zones which involve either negative density or internal energy. Obtaining robust solutions for high-Mach, high density ratio multi-phase flows requires enforcing positivity of mass, the squared speed-of-sound, and boundedness of phase volume fraction. These conditions are commonly achieved through the minimal and localized use of flux limiters [36] in a manner that does not affect discrete conservation. A disadvantage of implicit interface capturing methods is that their dissipation can smear the material interfaces indefinitely. However, recent advancements in implicit interface capturing offer a remedy to this problem through the use of sharpening techniques from algebraic VOF methods, specifically the tangent of hyperbola for interface capturing (THINC) scheme [37, 38, 39, 40].

2

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

1.2.2. Phase field methods In addition to implicit interface capturing methods, partial differential equation (PDE) based diffuse interface models exist which regularize material interfaces to a finite and resolvable thickness to enforce immiscibility conditions between phases. In particular, this work is focused on a class of diffuse interface methods known as phase field models, which includes models based on the Cahn-Hilliard [41] and AllenCahn equations [42]. Recent work has proposed general implementation strategies for phase field models to provide consistent and globally conservative solutions in both incompressible and compressible flows [43, 44]. Further advances in phase field modeling have resulted in locally conservative forms of the AllenCahn equation [45] known as the conservative diffuse interface (CDI) method. In recent years, significant progress has been made with the CDI model, including proving bounded volume fractions [46], providing required consistency conditions for both incompressible and compressible flows [46, 47], and generalizing the CDI model for simulation of N-phase immiscible flows [48] as well as on generalized curvilinear grids [49]. Additionally, the CDI model combined with shock-capturing schemes has been shown to be highly effective in capturing shock-interface interactions [35, 50].

1.3. Equilibrium conditions In addition to a robust numerical scheme, diffuse interface methods require equilibrium conditions to define fields in the diffuse zone. The most concise model is known as the four-equation model and requires thermo-mechanical equilibrium between phases within a computational element by enforcing that one temperature, pressure, and velocity vector is shared between phases [51, 52, 53, 54, 55, 56, 57]. Other popular equilibrium conditions include a five-equation model that does not enforce thermal equilibrium [58]; the six-equation model which additionally relaxes pressure equilibrium [59, 60]; and the seven-equation model, which further relaxes momentum equilibrium between phases [61]. We note that when modeling immiscible interfaces, diffuse interface methods artificially thicken the zone of coexisting phases solely to ease the numerical resolution requirements. Noting that physical interfaces have thicknesses on the order of a nanometer, thermo-mechanical equilibrium in the interface zone is practically instantaneous compared to the temporal resolution of numerical simulations. Therefore, using a diffuse interface model in this manner should not change the equilibrium conditions present in the sharp interface limit. However, in practice models that use additional equations compared to the four-equation model commonly sacrifice this desired thermo-mechanical equilibrium property as a trade-off to gain numerical robustness [62]. In some cases achieving robustness requires either sacrificing discrete conservation (of mass [63, 64] or energy [65]) or solving additional equations beyond the number of relaxed equilibrium conditions, e.g. equations involving the ratio of specific heats [66, 67, 68], which are redundant on the PDE level and lead to non-unique solutions. Aside from departing from the desired thermo-mechanical equilibrium conditions, these models incur more cost proportional to the number of additional equations which must be solved. In this sense, development of a robust and discretely conservative four-equation model without redundant PDEs addresses a physics-based need while reducing the computational cost of simulations.

1.3.1. Interface equilibrium condition The main robustness issue with the numerical simulations of the four-equation model is formulating a scheme that satisfies the interface equilibrium condition while retaining a unique and discretely conservative solution. In the general case, the interface equilibrium condition (IEC) is defined as: for a flow with uniform pressure, temperature, and velocity, the numerical discretization should not introduce spurious oscillations in any of these fields across material interfaces. If a scheme does not satisfy the IEC, spurious oscillations around material interfaces grow with time, potentially causing unphysical results and eventually code failure. Past work focused on satisfying the IEC with a unique and conservative four-equation model includes [69], which proposed a conservative IEC preserving flux-splitting scheme. However, the scheme from [69] is only applicable to flows involving single-phase multi-component mixtures of perfect gases without shocks. A similar extension of the method of [69] to real-gas equations of state has recently been proposed [70], but it can only approximately satisfy the IEC with a limited spatial order of accuracy. Other past works using the four-equation model in the compressible regime either suffered from oscillations due to lack of satisfying the

3

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

IEC [71], did not achieve uniqueness or discrete conservation [63, 66], used dissipative low-order schemes [57], or resorted to dissipative spatial filters to dampen growth of the oscillations throughout time [50]. Recent work has provided a localized dissipation strategy to dampen growth of the IEC oscillations for a range of density ratios, though robustness issues are shown to exist for very large density ratios [72]. One of the main contributions of this work is providing a numerical scheme that satisfies the interface equilibrium condition near machine precision using a four-equation model. The ENO-type scheme is consistently formulated with the thermodynamic assumptions of the four-equation multi-phase model to prevent oscillations around material interfaces without additional equations, sacrificing conservation, or adding spatial filters. Additionally, the proposed IEC-satisfying ENO-type scheme is applicable for general equations of state and complex mixing rules. In addition to the IEC-satisfying ENO-type scheme, this work includes extensions of the CDI phase field model to capture immiscibility for multi-phase multi-component interfaces, and proposes a phase constrained species diffusion transport model to successfully enforce immiscibility by avoiding numerical leakage of species during the diffusion process. Furthermore, this work includes an extension of the positivity preserving flux limiter of [36] to the four-equation setting for robust treatment of the interactions of high-Mach number shocks with high-density ratio material interfaces. The presented formulation and discretization is inclusive of an implicit interface capturing method which satisfies the IEC with the four-equation model. This can be realized by omitting the CDI regularization terms in our model.

1.4. Outline

The remainder of this paper is outlined below. In Section 2 we introduce the system of equations describing multi-phase multi-component flows, the mixing rules used for interphase phases and intraphase components, and the equation of state used to closed the system. In Section 3 the spatial-temporal discretization is described, including the method used to obtain oscillation-free solutions across material interfaces using ENO-type schemes, the extension of the CDI model used to multi-phase multi-component systems, and the positivity-preserving flux-limiter for the four-equation model. Section 4 presents several one-dimensional and two-dimensional cases ranging from multi-component to multi-phase multi-component flows in order to verify the application of the proposed framework. Finally, in Section 5, we summarize the results and provide an outlook for future work.

2. Governing equations

2.1. Physical model

The multi-phase, multi-component form of the compressible Navier-Stokes equations will be studied in this work. In addition, we present a material-interface regularization model which can be added to the system to represent immiscible material interfaces. Although we can represent immiscible interfaces, the proposed model is general and can be used for modeling of miscible interfaces (similar to implicit interface capturing) by omitting the regularization terms and relying on a species diffusion model. To clearly distinguish between different phases, we introduce a notation in which the phase of a material will be indicated by the superscript p, where p = [1, 2], and the components within a given phase will be indicated by the subscript c, where 1 ≤c ≤N. Although we will utilize index notation for coordinates and tensor components, we do not imply index notation for p and c. The fluid state at a given time t can be described at position x = [x, y, z]T by the vector of primitive variables W = [T, Y c p , u, v, w, P]T or by the vector of conserved variables U = [ρY c p , ρu, ρv, ρw, E]T . In these definitions, Y c p is the mass of component c of phase p per total mass, known as the mixture mass fraction, u = [u, v, w]T is the velocity vector, P is the pressure, T is the temperature, ρ is the mixture density, and E is the total energy per unit volume defined as E = ρe + 1

2ρu · u where e is the internal energy per unit volume. Using these variables, the multi-phase multi-component Navier-Stokes equations with interface regularization terms can be compactly written in differential form in terms of the conserved variables U,

4

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed


$$
∂U
$$


$$
∂t + ∂[F(U) + Fν(U) + FDI(U)]
$$


$$
∂x + ∂[G(U) + Gν(U) + GDI(U)]
$$


$$
∂y + ∂[H(U) + Hν(U) + HDI(U)]
$$


$$
∂z = 0,
$$


$$
(1)
$$

for p = 1, 2 and 1 ≤c ≤N. The convective fluxes F, G, H are defined as,

F(U) =






$$
ρY c
ρuu + P
ρuv
ρuw
$$



 G(U) =






$$
ρY c
ρuv
ρvv + P
ρvw
$$



 H(U) =






$$
ρY c
ρuw
ρvw
ρww + P
$$




$$
 , (2)
$$

and the viscous fluxes, Fν, Gν, Hν can be expressed as,


$$
Fν(U) =
$$






$$
−τ11
−τ12
−τ13
−Σiuiτ1i + q1
$$




$$
 Gν(U) =
$$






$$
−τ21
−τ22
−τ23
−Σiuiτ2i + q2
$$




$$
 Hν(U) =
$$






$$
−τ31
−τ32
−τ33
−Σiuiτ3i + q3
$$




$$
 , (3)
$$

where τij is the viscous stress tensor,


$$
τij = µ
�∂ui
∂xj
+ ∂uj
$$


$$
∂xi
−2/3∂uk
$$


$$
∂xk
δij
$$


$$
� , (4)
$$

with µ as the dynamic viscosity, and δij as the Kronecker delta. Additionally, we propose a leakage-free intraphase species mass diffusion model as an extension of a well-known version of the Stefan-Maxwell diffusion model [73] using a confined scaler argument [74, 75]. The resulting model can be written as a phase constrained Fickian diffusion term and a mass corrector as,

Jc p,i = −ρY c p




$$
Dc p ∂ ∂xi
$$

� ln �Xc p Xp


$$
�� − �
$$

j

� Y j p Yp

�


$$
Dj p ∂ ∂xi
$$

�

ln

� Xj p Xp

��


$$
 (5)
$$


$$
= −ρY c p
$$



Dc p Xp Xcp


$$
∂
∂xi
$$

�Xc p Xp


$$
� − �
$$

j

� Y j p Yp

Xp Xj p

�


$$
Dj p ∂ ∂xi
$$

� Xj p Xp

�


$$
 (6)
$$


$$
= −ρ
$$



Dc pYp W c p Wp


$$
∂
∂xi
$$

�Xc p Xp


$$
� −Y c p �
$$

j

� W j p Wp

�


$$
Dj p ∂ ∂xi
$$

� Xj p Xp

�


$$
 (7)
$$

where Dc p is the mass diffusivity of component c within phase p, Xc p is the mixture molar fraction of component c of phase p, Xp is the mixture molar fraction of phase p, Yp is the mixture mass fraction of phase p, W c p is the molecular weight of component c, and Wp is the molecular weight of phase p. Formulas describing the computation of Wp, Yp, and Xp are in Appendix E. One key difference between this formulation and the common non-dilute Stefan-Maxwell model is the incorporation of the denominators (Xp and Yp) in the gradient terms. This enforces phase-constrained diffusion of intraphase species by restricting the volume of influence of species diffusion to be within each phase, preventing unphysical leakage of components across phases. This formulation can be derived using the consistent transport models for confined scalars introduced in previous work [74, 75]. Any physical exchange across phases (e.g. phase change) would require additional models to explicitly represent masstransfer between phases. Lastly, the heat flux qi is defined as,

5

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed


$$
qi = −λ ∂T
$$


$$
∂xi + �
$$

p

�


$$
c Jc p,i Y c p Yp hc p, (8)
$$

where λ is the heat conductivity of the mixture and hc p is the specific enthalpy of component c of phase p. The final terms in Eq. 1, FDI, GDI, HDI, are the interface regularization terms (i.e. the diffuse interface terms) and are described in Section 3.2.

2.2. Mixture rules 2.2.1. Equilibrium assumptions In this work, thermal and mechanical equilibrium is assumed between phases within a computational element by enforcing that one temperature, pressure, and velocity vector is shared between phases. In the literature, models which enforce thermo-mechanical equilibrium are often known as the four-equation multi-phase model [51, 52, 53, 54, 55, 56, 57]. One of the main contributions of this work is providing a computational strategy to satisfy the discrete interface equilibrium condition near machine precision with the four-equation model. The proposed scheme is presented in Section 3.1. The following sections overview the mixing rules required to obtain the equation of state closure for Eq. 1. Section 2.2.2 starts by defining the mixing rules describing multi-phase mixtures and Section 2.2.3 extends this to a general multi-phase multi-component model.

2.2.2. Interphase mixing rules In this work, multi-phase mixtures are assumed to be immiscible. To describe immiscible phases, separate phases are assumed to occupy their own individual volumes and share a common pressure (Amagat’s law). The mixing rules are summarized as,


$$
Tp = T ∀p, Pp = P ∀p, 1/ρ = �
$$

p Ypvp, e = �


$$
p Ypep (9)
$$

where Tp, Pp, ep, and vp are the temperature, pressure, internal energy, and specific volume for phase p. These thermodynamic variables can be obtained from the equation of states for each phase p. These mixing rules allow for each phase p to be governed by different equation of states as well as by unique intraphase mixing rules to describe multi-component mixtures. Other quantities that are treated with interphase mixing rules include the mixture viscosity µ and mixture thermal conductivity λ as, µ = �


$$
p ϕpµp, λ = �
$$


$$
ϕpλp
(10)
$$

where µp and λp are the phasic viscosity and thermal conductivity respectively, and ϕp is the phase volume fraction. In this work λp and µp are assumed constant in the liquid phase. For the gaseous phase, λg is defined as, λg = µg

Pr

�


$$
c (Y c g /Yg)CP c g (11)
$$

and µg is defined using Wilke’s rule [76] as,

µg = �

i

(Y i g /Yg)µi g �


$$
j GijWc/Wj(Y j g /Yg) (12)
$$

where,


$$
Gij = 1 √
$$

8

� 1 + Wi

Wj


$$
�−1/2 
$$

1 +

� µi g µj g


$$
�−1/2 �Wi Wj
$$

�1/4 



2


$$
(13)
$$

and the Prandtl number, Pr, the dynamic viscosity of species µi g, and the specific heats, CP c g, are assumed constant. Although not explored in this work, the formulation does not restrict these quantities from being extended to more advanced definitions, including defining the components with a non-linear dependence on temperature.

6

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

2.2.3. Intraphase mixing rules Classical intraphase mixing rules include assuming the individual components within a phase occupy their own volume (Amagat’s law) or assuming components share volumes (Dalton’s law). Traditionally, mixtures of gas components are assumed to follow Dalton’s law, with each individual component contributing a partial pressure towards the mixture pressure [77]. For a real gas equation of state (e.g. including the compressiblity factor), Dalton’s law and Amagat’s law will give different results. However, if the gas components are modeled as ideal gases, both representations obtain equivalent thermodynamic states [77]. The governing equations in Section 2.1 are written to allow either intraphase mixing rule to be applied. In this work, although the gas components follow the ideal gas law, both mixing rules are summarized to show the generality of the four-equation model and provide a background for extensions to more complex equations of state. The ideal mixing rules associated with assuming all intraphase components occupy the same volume (Dalton’s law) are summarized as,

Tp = T c ∀c, Pp = �

c P c partial, vp = vc p ∀c, ep = �


$$
c (Y c p /Yp)ec p (14)
$$

where (assuming the ideal gas law) the partial pressure of component c is defined as,


$$
P c partial = ρY p c RT/W c (15)
$$

where R is the universal gas constant and W c is the molecular weight of component c. Additionally, the ideal mixing rules associated with assuming all components occupy the individual volumes (Amagat’s law) are summarized as,

Tp = T c ∀c, Pp = P c ∀c, vp = �

c (Y c p /Yp)vc p, ep = �


$$
c (Y c p /Yp)ec p. (16)
$$

Lastly, the mass diffusivity of component c within phase p is defined as,


$$
Dc p = µp ρpScc p (17)
$$

where, in this work, a constant Schmidt number, Scc p, is assumed for each component within a phase.

2.3. Noble Abel stiffened gas EOS To close the system illustrated in Eq. 1 the Nobel-Abel equation of state (NASG EOS) [78] is used for all components in the multi-phase multi-component mixture. For a pure component, the general NASG EOS assuming constant heat capacity reads,

vc p(P, T) = (CP c p −Cvc p)T P + P ∞cp + bc p


$$
ec p(P, T) = P + γc pP ∞c p P + P ∞cp Cv c pT + qc p
$$

hc p(P, T) = CP c pT + bc pP + qc p


$$
(18)
$$

where for component c in phase p, CP c p is the heat capacity at constant pressure, Cvc p is the heat capacity at constant volume, γc p is the heat capacity ratio, bc p is the co-volume of the component molecules, P ∞c p is the stiffened pressure to model the attractive forces between molecules in a material, and qc p is a reference energy. If bc p and P ∞c p are taken as zero (as is done for components in the gas phase) the NASG EOS reduces to the ideal gas equation of state. During a simulation, the internal energy and volume (or density) of the mixture can be determined from the conserved vector U and the mixing rules in Section 2.2. To obtain expressions for the pressure and

7

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

temperature of the mixture in terms of internal energy and volume, the mixing rules discussed in Section 2.2 can be combined with the NASG EOS. For a system with multiple components in both phases, an iterative procedure is required to find the pressure and temperature that satisfy the equilibrium conditions [53]. In this work, only one component will be modeled in the liquid state, and the gaseous state will be composed of ideal gas components. Given these simplifications, we can use Amagat’s law for both intraphase and interphase mixing and the NASG EOS to obtain closed expressions for the mixture pressure. An expression for the mixture pressure can be found by enforcing the equality between the two definitions for mixture temperature found from Eq. 18,

T(e, P, Y c p ) = (e −q)

��

p

�

c


$$
Y c p Cvc p(P + γc pP ∞c p) P + P ∞cp
$$


$$
�−1
$$


$$
(19)
$$


$$
T(ρ, P, Y c
p ) = (1/ρ −b)
$$

��

p

�

c


$$
p(γc
p −1)
P + P ∞cp
$$


$$
�−1
$$


$$
. (20)
$$

Solving for a common pressure between Eq. 19 and 20 leads to


$$
P = a2 + � a2 2 + 4a1a3 2a1 (21)
$$

with

a1 = Cv


$$
�e −q
1/ρ −b
$$


$$
� (CP −Cv) −P ∞Cv −P ∞Y1(CP 1 −Cv1)
$$


$$
�e −q
1/ρ −b
$$


$$
� P ∞[CP −Cv −Y1(CP 1 −Cv1)]
$$


$$
(22)
$$

and Cv = �

p

�

c Y c p Cv c p, CP = �

p

�

c Y c p CP c p, q = �

p

�

c Y c p qc p, b = �

p

�


$$
c Y c p bc p (23)
$$

where for this work p = 1 is defined as the liquid phase and P ∞≡P ∞c=1 p=1 as there is only one component in the liquid. Once pressure is determined using Eq. 21, either Eq. 19 or Eq. 20 can be used to obtain the mixture temperature. For mixtures with multiple components in each phase or specific heats dependent on temperature, iterative solvers can be used to determine the equilibrium state [53]. Unless otherwise specified, the NASG EOS parameters used for all materials throughout this work are included in Table 1.

Material CP [J kg−1K−1] γ [-] q [J kg−1] b [m3 kg−1] P∞[Pa]

Water (Liquid) 4.185 × 103 1.0123 −1.143 × 106 9.203 × 10−4 1.835 × 108

Air 1.011 × 103 1.4 0 0 0 Helium 5.091 × 103 1.66 0 0 0 SF6 0.661 × 103 1.093 0 0 0


> **Table 1: NASG Parameters used in this work**

8

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

3. Numerical framework

3.1. Spatial discretization

In this work, the integral form of the partial differential equations in Eq. 1 will be solved using the finite-volume method. The differential form of Eq. 1 is assumed to admit smooth solutions for which partial derivatives exist. A general cuboid cell (i, j, k) has the spatial width defined by ∆x, ∆y, and ∆z and a cell volume defined as ∆x∆y∆z. In this work, isotropic Cartesian grids (∆x = ∆y = ∆z) will be used to spatially discretize the system. In the finite-volume formulation, the cell-averaged values of the conserved variables can be obtained as,

¯Ui,j,k = 1

V

�xi+ 1 2 ,j,k


$$
xi−1
$$

2 ,j,k

�yi,j+ 1 2 ,k


$$
yi,j−1
$$

2 ,k

�zi,j,k+ 1 2


$$
zi,j,k−1
$$


$$
2 Udxdydz. (24)
$$

Applying the volume integration to Eq. 1 leads to a set of ordinary differential equations,

d dt ¯Ui,j,k = −1


$$
∆x
$$

� Fi+ 1


$$
2 ,j,k −Fi−1
$$


$$
2 ,j,k � −1
$$


$$
∆y
$$

� Gi,j+ 1


$$
2 ,k −Gi.j−1
$$


$$
2 ,k � −1
$$


$$
∆z
$$

� Hi,j,k+ 1


$$
2 −Hi,j,k−1
$$

2

�


$$
−1
$$


$$
∆x
$$

� Fνi+ 1


$$
2 ,j,k −Fνi−1
$$


$$
2 ,j,k � −1
$$


$$
∆y
$$


$$
� Gνi,j+ 1
$$


$$
2 ,k −Gνi.j−1
$$


$$
2 ,k � −1
$$


$$
∆z
$$


$$
� Hνi,j,k+ 1
$$


$$
2 −Hνi,j,k−1
$$

2

�


$$
−1
$$


$$
∆x
$$

� FDIi+ 1


$$
2 ,j,k −FDIi−1
$$


$$
2 ,j,k � −1
$$


$$
∆y
$$

� GDIi,j+ 1


$$
2 ,k −GDIi.j−1
$$


$$
2 ,k � −1
$$


$$
∆z
$$

� HDIi,j,k+ 1


$$
2 −HDIi,j,k−1
$$

2

�


$$
(25)
$$

where, Fi+1/2,j,k, Fνi+1/2,j,k, and FDIi+1/2,j,k are the averaged cell face fluxes on the xi+1/2 face for the convective, diffusive, and regularization terms respectively. The definitions of the fluxes on the y−and z−faces are analogous. For multi-dimensional problems Eq. 25 is solved using a dimension-by-dimension approach where each dimension is evaluated separately. Additionally, in this work all cell face averaged variables are approximated using a one-point quadrature rule. Higher-order representations are possible and have been explored in other works [34]. To calculate the convective flux on the cell faces the Godunov approach [79] is used by solving a Riemann problem on a cell face. The Riemann problem is defined by fluid states on the left and right sides of a cell face, which are obtained by numerical interpolations. Across discontinuities, including shocks and material interfaces, the method of interpolation can be critical to obtaining a robust and oscillation-free solution. A brief review of previous approaches to limiting oscillations across discontinues and interfaces was presented in Section 1. In this work, an ENO-type (WENO-type, or TENO-type) interpolation scheme is used in combination with a HLLC Riemann solver for the hyperbolic terms. A key challenge when applying this methodology to material interfaces is formulating the interpolations to satisfy the IEC. Details on the variables which are reconstructed by the ENO-type scheme while satisfying the IEC can be found in Section 3.1.2 and details on the HLLC form is in Appendix B. The viscous fluxes are interpolated to faces using a standard secondorder central scheme. The interface regularization fluxes are interpolated to faces using a second-order kinetic energy and entropy preserving (KEEP) central interpolation that satisfies the IEC [80]. More detail on the regularization terms and the KEEP interpolation is in Section 3.2.

3.1.1. Avoiding spurious oscillations High-order interpolation of the fluid state directly to the cell faces can result in spurious oscillations around interfaces due to the interactions of discontinuities in dependent fields. Additionally, the nonlinearity of ENO-type schemes can lead to different numerical stencils for each fluid field. As these fields are dependent on one another in physical space, the inconsistency between the stencils (and the stencil’s numerical dissipation) can result in spurious oscillations. Instead, projecting the fields to characteristic space before interpolation decouples the fluid states into independent characteristic variables and avoids the issues associated with using ENO-type schemes [28, 29, 30, 81].

9

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

Although interpolating the characteristic variables reduces spurious oscillations, it does not cause the system to inherently satisfy the interface equilibrium condition and will still result in oscillations around material interfaces. As discussed in Sections 1.3 and 1.3.1, past works have presented strategies for satisfying the IEC. However, these strategies either require loss of conservation, add redundant equations, or restrict mixing rules and thermodynamic equilibrium assumptions.

3.1.2. Satisfying the interface equilibrium condition Notably, in this work, the IEC is satisfied to near-machine precision using the four-equation model without redundant equations. The idea is based on the use of the primitive vector W = [T, Y c p , u, v, w, P]T , as the basis for building the interpolations to cell faces. An important clarification must be made that this set of primitive variables includes temperature instead of density (unlike what is done for the volume-fractionbased five-equation model [34]). Interpolating (including with a nonlinear ENO-type scheme) directly based on P, T, and u enforces the interface equilibrium condition numerically, since a well-formulated numerical scheme will keep a constant field constant after interpolation. Replacement of T with ρ, as done in other studies [34], will no longer maintain a constant temperature after interpolation (since ρ contains a jump across the interface) and results in oscillations of P, T, and u (shown in Table 7). Instead, using the primitive vector, W = [T, Y c p , u, v, w, P]T , removes oscillations around isothermal material interfaces and discretely satisfies the IEC. Though the strategy outlined above will satisfy the IEC, as discussed in Section 3.1.1, it is advantageous (especially around shocks) to decompose the fluid-state into characteristic variables in order to avoid additional spurious oscillations from the non-linearity of the ENO-type schemes. In this work, the proposed primitive basis of W = [T, Y c p , u, v, w, P]T (not including one Y c p ) is used to define characteristic variables. The characteristic matrices used for the projection of W = [T, Y c p , u, v, w, P]T into characteristic space are included in Appendix A. Additionally, using an interpolation based on characteristic variables with a nonlinear ENO-type scheme can be analytically shown to satisfy the IEC ( Appendix C). A numerical example showcasing the successful application of this IEC ENO-type scheme to solve an inviscid multi-phase droplet advection with oscillations in pressure, velocity, or temperature, near machine precision is given in Section 4.2.2.

3.1.3. Godunov Approach The resulting numerical strategy for obtaining the flux on cell faces is summarized below.

1. Project W into characteristic space to obtain: ˜ W

(a) Define an average of U on a face: Ui+ 1

2 (using either arithmetic or Roe average)

(b) Define the matrix of left eigenvectors, S−1 from the averaged state of Ui+ 1

2 (c) ˜ W = S−1W

2. Obtain left and right states of ˜ W on the cell face using ENO-type interpolation

3. Project ˜ W L i+ 1

2 and ˜ W R i+ 1

2 to physical space to obtain: W L i+ 1

2 and W R i+ 1

2

(a) Define the matrix of right eigenvectors, S from the averaged state of Ui+ 1

2 (b) W L i+ 1

2 = S ˜ W L i+ 1

2 , W R i+ 1

2 = S ˜ W R i+ 1

2

4. Obtain the left and right states for density using the EOS

5. Find the left and right states of the conserved variables from primitives

6. Find the cell face flux using a HLLC Riemann solver: Fi+ 1

2 = FHLLC(U L i+ 1

2 , U R i+ 1

2 )

10

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

The approach outlined above will result in high-resolution capturing of material interfaces and shocks with minimal dissipation. However, unlike shocks, material interfaces are not self-sharpening. Though high-order ENO-type interpolations can be used across material interfaces as an implicit interface capturing scheme, the numerical dissipation from the upwind-biased interpolations will cause mixing of immiscible phases throughout time. To enforce immiscibility between phases, interface regularization terms can be added to the system to sharpen multi-phase interfaces while allowing intraphase species diffusion.

3.2. Interface regularization

To counteract the intrinsic numerical diffusion caused by both the ENO-type interpolation and the approximate Riemann solver [79] from the Godunov approach described in Section 3.1.3, material interface regularization terms can be added to enforce a finite thickness interface throughout a simulation. A comparison of results achieved using an implicit interface capturing scheme (WENO5Z) and using an explicit phase field method (WENO5Z+CDI) is shown for a Rayleigh-Taylor instability in Figure 1. The setup for the case follows the details presented in [49] and Figure 1 shows the density field at t = 1.0125s. As shown, the two approaches have different effects. The implicit interface capturing scheme represents subgrid interfaces with mixing, whereas the explicit interface regularization model will keep interfaces phaseimmiscible. However, the interface regularization terms result in preemptive breakup of under-resolved interfaces even in the absence of surface tension. Generally, for scenarios with short residence times and negligible surface tension, using an implicit interface capturing could be acceptable, since material interfaces will not be substantially thickened by the numerical scheme over short periods. However, for scenarios with long residence time or non-negligible surface tension, adding regularization terms to represent immiscibility is appropriate.


> **Figure 1: Comparison of density field for Rayleigh-Taylor instability with (right) and without (left) interface regularization terms.**

The interface regularization proposed in this work is an extension of the conservative diffuse interface (CDI) model [45, 46, 82, 47] to incorporate multi-component mixtures. The multi-phase multi-component CDI model is designed to capture immiscible interfaces between phases and treat all components within phases as confined scalars [74]. Because intraphase components are treated as phase-confined scalars, they do not leak across phase interfaces, even when intraphase diffusion is included. The model form of the multi-phase multi-component CDI model can be written as,

11

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

FDI(U) =





Rc p,x uR,x vR,x wR,x kR,x + �

p hpRp,x



 GDI(U) =





Rc p,y uR,y vR,y wR,y kR,y + �

p hpRp,y



 HDI(U) =





Rc p,z uR,z vR,z wR,z kR,z + �

p hpRp,z





(26) with,


$$
Rc p,x = −Y c p Yp ρpΓ
$$

�


$$
ϵ∂ϕp
$$


$$
∂x −(ϕp −ϕmin)((1 −ϕmin)) −ϕp)
$$


$$
∂ϕp
$$


$$
∂x
|⃗∇ϕp|
$$

�


$$
, (27)
$$

and k = 1

2u · u as the kinetic energy, Rp,x = �

c Rc p,x, R,x = �

p Rp,x, and ρp and hp are defined using the EOS and governing mixing rules of the given phase p. The regularization term in the yand z-directions is written similarly to the x-direction shown. Additionally, Γ, ϵ, and ϕmin are user-set parameters. The parameter Γ is a velocity scale which determines the speed at which the interface equilibrates towards a hyperbolic tangent profile. The parameter ϵ is a length scale and determines the thickness of the interface. The parameter ϕmin determines the volume fraction floor and ceiling which the regularization term relaxes towards. In this work, ϕmin = 10−8 for all cases. The multi-phase form of Eq. 27 has been studied extensively and it has been shown that properly choosing ϵ ∼∆and Γ = max(|ui,j,k|, |vi,j,k|, |wi,j,k|), ∀i, j, k results in bounded volume fraction for a large range of circumstances, including incompressible flows [46, 82], compressible flows [47], and recently for geometries using generalized curvilinear grids [49] for central discretizations. However, when numerically solving Eq. 27 interpolation errors of ϕp can occur close to the interface since the regularization sharpens the interface towards a hyperbolic tangent profile of thickness ϵ. Interpolations of ϕp, and its derivatives, are needed to calculate the non-linear sharpening term and the interface normal on a cell face. In order to reduce numerical errors during these interpolations, ϕp can be transformed to an approximate signed-distance function ψp [83, 12, 84, 85] using,


$$
ψp = ϵ ln
ϕp + δ
1 −ϕp + δ
(28)
$$

where δ is a very small number (taken as δ = 10−100 in this work, similar to [85]). Near the interface the ψp field is approximately linear and will not suffer from numerical error associated with interpolation. An analytical reformulation of Eq. 27 in terms of ψp has been previously completed [85] and results in,

Rc p,x = −Y c p Yp ρpΓ

�


$$
ϵ∂ϕp
$$


$$
∂x −1
$$

4


$$
� 1 −tanh2 �ψp
$$


$$
2ϵ
$$


$$
� −4ϕmin(1 −ϕmin) � ∂ψp
$$


$$
∂x
|⃗∇ψp|
$$

�


$$
. (29)
$$

Both models shown in Eq. 27 and Eq. 29 are equivalent on the continuous level (absent the effects of δ), but the numerical advantages of transforming the nonlinear term from ϕp to ψp make the transformed model more computationally attractive. As such, the regularization model in Eq. 29 will be used for all simulations in this work.

3.2.1. Numerical implementation of regularization model Analogous to the formulation of the ENO-type scheme to satisfy the IEC, the thermodynamic quantities in the regularization flux can be independently interpolated and subsequently averaged to construct a face flux that adheres to the IEC. Accordingly, the regularization flux in Eq. 26 is implemented using a secondorder skew-symmetric split-form centered scheme that satisfies the IEC while preserving kinetic energy and entropy (KEEP) [80]. As an example, the numerical flux for the regularization terms on a cell face in the x-direction can be

12

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

written as,


$$
� FDI(U) (i±1/2) =
$$






$$
� Rcp,x (i±1/2)
$$


$$
u(i±1/2)� R,x (i±1/2)
$$


$$
v(i±1/2)� R,x (i±1/2)
$$


$$
w(i±1/2)� R,x (i±1/2)
$$


$$
k (i±1/2)� R,x (i±1/2) + � p � hpRp,x (i±1/2)
$$






$$
(30)
$$

where

� ap,x (i±1/2) = Γ




$$
ϵ
∂ϕp
$$


$$
∂x
$$


$$
(i±1/2) −1
$$

4

�


$$
1 −tanh2 �ψp (i±1/2)
$$


$$
2ϵ
$$


$$
� −4ϕmin(1 −ϕmin)
$$


$$
�� ∂ψp
$$


$$
∂x
|⃗∇ψp|
$$


$$
(i±1/2)
$$


$$
 (31)
$$

� Rcp,x (i±1/2) = −Y c p Yp ρp


$$
(i±1/2) � ap,x (i±1/2) (32)
$$

� R,x (i±1/2) = �

p

�


$$
c � Rcp,x (i±1/2) (33)
$$

k (i±1/2) = 1

2(u(i±1)u(i) + v(i±1)v(i) + w(i±1)w(i)) (34)

� hpRp,x (i±1/2) = −hpρp (i±1/2) � ap,x (i±1/2). (35)

In the notation above, � (·) (i±1/2) denotes a field which consists of a numerical derivative operation onto a

face, and (·) (i±1/2) represents an interpolation operation onto a face. The critical part of skew-symmetric split-form, which allows the scheme to satisfy IEC, is keeping all velocity and density interpolations to

faces separate. For example, the momentum flux in the x-direction is written as u(i±1/2)� R,x (i±1/2), instead


$$
of −�
$$

p �

c

Y c p Yp ρpu (i±1/2) � ap,x (i±1/2). Furthermore, it is critical to interpolate the product of enthalpy and density together, as this will retain a consistent pressure field at the cell face. In this work, we use secondorder accuracy for the CDI regularization terms. However, the implementation of skew-symmetric terms can be extended to arbitrary orders of accuracy. An example of applying a high-order skew-symmetric KEEP scheme to convective terms can be found in [86].

3.3. Positivity preservation Section 3.1 describes the numerical methods that provide nearly non-oscillatory solutions for flows with shocks and material interfaces. However, the high-order ENO-type schemes still lead to small oscillations which can result in inadmissible thermodynamic states (e.g. negative internal energy or density) and code failure. The positivity of the speed of sound is a critical metric for simulation robustness. To guarantee a positive speed of sound the equation of state can be used where the hyperbolic speed of sound is defined by,


$$
a2 = CP ρβCP −α2T (36)
$$

where β = 1


$$
ρ
∂ρ
∂P
$$


$$
T,Y and α = −1
$$


$$
ρ
∂ρ
∂T
$$

P,Y . For the NASG EOS used in this work [87],


$$
β = ρ �
$$

p

�

c Y c p


$$
�(γc p −1)Cvc pT (P + P ∞cp)2
$$


$$
� (37)
$$


$$
α = ρ �
$$

p

�

c Y c p


$$
�(γc p −1)Cvc p (P + P ∞cp)
$$


$$
� . (38)
$$

13

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

For a single-phase flow where P ∞c p = 0 ∀c, p, the temperature, pressure, and density must be positive to keep the square of the speed of sound positive. For a multi-phase flow it is possible for the squared speed of sound to be positive even if mixture pressure is negative. Physical situations like cavitation can result in negative pressures, and the NASG equation of state used in this work permits this. From Eq. 19 and Eq. 20, to guarantee an admissible temperature, both the internal energy and the density of the mixture must be positive. Additionally, a positive temperature and pressure will guarantee positive phasic internal energies and densities from Eq. 18. For both singleand multi-phase flows, to guarantee the positivity of the squared speed of sound with the NASG equation of state the following two requirements must hold,


$$
(e −q) > 0 and (1/ρ −b) > 0. (39)
$$

With these requirements satisfied the solution will have an admissible speed of sound, temperature, pressure, and mixture density. In order to guarantee boundedness for mass fractions and volume fraction between zero and one, a separate check must take place,


$$
(Y c p > 0) ∀ c, p. (40)
$$

To ensure a physically admissible solution at all times, three limiters are added to the numerical procedure. In the Godunov approach described in Section 3.1.3, an interpolation limiter is added after the ENO-type high-order interpolation to the face, and a flux limiter is added after the approximate Riemann solver. Additionally, a flux limiter is added after the interface regularization flux described in Section 3.2 due to the addition of the sharpening term. All three of these limiters are discussed below.

3.3.1. Interpolation limiter After high-order ENO-type interpolation (step 4 of the Godunov approach), the boundedness of the mass fractions is achieved by limiting the mass fractions using an approach described in [88]. As an example, the procedure for interpolation of the mass-fraction on the minus side of the (i + 1/2) face will be described. First, the mass fraction field left out of the characteristic projection is individually interpolated to the cell face using an ENO-type scheme. Then, the mass fractions can be limited with the following steps,

1. If �

c �

p Y c p − (i+1/2) > 1


$$
(a) Σ−= �
$$

c �


$$
p � min � Y c p − (i+1/2), Y c p (i) � −Y c p − (i+1/2) �
$$

(b) ϵc p = � 1−�

c �


$$
p Y c p − (i+1/2) Σ−
$$


$$
�� min � Y c p − (i+1/2), Y c p (i) � −Y c p − (i+1/2) �
$$

2. If � c � p Y c p − (i+1/2) < 1

(a) Σ+ = �

c �


$$
p � max � Y c p − (i+1/2), Y c p (i) � −Y c p − (i+1/2) �
$$

(b) ϵc p = � 1−�

c �


$$
p Y c p − (i+1/2) Σ+
$$


$$
�� max � Y c p − (i+1/2), Y c p (i) � −Y c p − (i+1/2) �
$$

3. Y c p − (i+1/2) = Y c p − (i+1/2) + ϵc p.

The mass fraction limiter for the positive side of the (i + 1/2) cell face is done symmetrically. One of the advantages of this limiter is that for multi-component systems it will ensure that interpolation error does not impact inert species in the stencil and only those with varying profiles. It will also spread out error between multiple species instead of forcing all interpolation errors on the species left out of the interpolation. Additionally, we note that the limiter is used to construct an admissible state on a face to eventually construct a flux, so the numerical scheme remains fully mass conserving, as the conserved values of ρY c p are unchanged by the limiter.

14

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

Additionally, the positivity of pressure and temperature can be independently checked after the ENO-type interpolation and mass-fraction limiter. If a field is inadmissible, a first-order interpolation (cell-centered value) is taken as the corresponding left/right state to build the flux in the Riemann solver. For example,


$$
P −
i+1/2 = Pi(1 −θP ) + P −
(41)
$$


$$
T −
i+1/2 = Ti(1 −θT ) + T −
(42)
$$

where θP or θT are either taken as 0 or 1 depending on if the interpolated value is admissible. Note that unlike works with similar limiters [36, 40], in this work θP , θT , and ϵc p are kept independent for each field after interpolation, allowing for a less dissipative positivity-preserving interpolation. Once all fields in the ENO-type interpolation have been limited, the state on the cell face can be used to build admissible conserved variables (step 5 of Godunov approach) and a Riemann solver can be used to find a flux (step 6 of the Godunov approach).

3.3.2. Flux limiter After using the Riemann solver to find a flux on the cell face, a flux limiter is used to guarantee that the state at the following time step is admissible [36, 40]. An outline for limiting the advection flux at cell face xi+1/2 in the x-direction with an approximate HLLC Riemann solver is shown below.

1. Calculate FHLLC(U L (i+1/2), U R (i+1/2))

2. Use pseudo-time integration within the RK sub-step to check that FHLLC(U L (i+1/2), U R (i+1/2)) is admissible. Below, D is the spatial dimension of the simulation and RK0, RK1, and RK2 are coefficients for a given SSP RK sub-step as shown in Eq. 43.

(a) ˜U (RKstep+1) i = RK0U (n) i + RK1U (RKstep) i + 2RK2∆tDFHLLC

(b) ˜U (RKstep+1) i+1 = RK0U (n) i+1 + RK1U (RKstep) i+1 −2RK2∆tDFHLLC

3. If Eqs. 39 and 40 hold for both ˜U (RKstep+1) i and ˜U (RKstep+1) i+1 , FHLLC(U L (i+1/2), U R (i+1/2)) is admissible. Otherwise, a first-order positivity-preserving flux, FHLLC(U(i), U(i+1)), is used.

Step 3 of the outline above can be augmented with a blending operation between the higher-order flux FHLLC(U L (i+1/2), U R (i+1/2)) and the first-order flux FHLLC(U(i), U(i+1)). Initial tests using blending did not show a noticeable improvement in the results. In order to reduce the complexity and cost of the positivity preserving routine, blending was not used in this work. It is also useful to observe that checking the solution satisfies the constraints in Eqs. 39 and 40 only requires the conserved vector U without the evaluation of the NASG EOS. For this work, the NASG EOS is analytical for all fields, but for more complex systems, including those with non-constant specific heats, an expensive iterative procedure could be required to evaluate pressure and temperature [53]. Evaluating Eqs. 39 and 40 remains inexpensive, even when the EOS becomes more complex. After calculating the diffuse interface flux from Section 3.2, the diffuse interface flux can be deemed admissible by using the same flux-limiter algorithm described above. In step 1, replace FHLLC with FHLLC + FDI, where FHLLC is the admissible flux from the advection term. If the diffuse-interface flux is inadmissible it is not added to the solution. Restricting the use of the diffuse-interface flux is infrequent and does not result in any visible smearing/mixing of immiscible multi-phase interfaces for the simulations in Section 4.

3.4. Temporal integration

After spatial discretization, the system is expressed as a set of ordinary differential equations which can be integrated in time. In this work, we use a third-order strong-stability preserving (SSP) Runge-Kutta

15

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

method expressed below [89],


$$
u(1) = un + ∆tL[un]
$$

u(2) = 3

4un + 1

4u(1) + 1


$$
4∆tL[u(1)]
$$

un+1 = 1

3un + 2

3u(2) + 2


$$
3∆tL[u(2)]
$$


$$
(43)
$$

where L[·] is the operator to evaluate the numerical approximation of the spatial operations. The superscripts with parenthesis, (1) and (2), denote the first and second sub-steps within time step n. The superscripts without parenthesis, n and n + 1, denote the current time step n as well as the subsequent time step n + 1. Additionally, the time step size ∆t is given by either the advection or diffusion CFL criterion expressed as,


$$
∆t = CFL
$$

maxi=1:3 � |ui+a|


$$
∆xi , 4µ/ρ
$$


$$
∆x2 i , 4 max{Dcp}
$$


$$
∆x2 i
$$


$$
� (44)
$$

where 0 < CFL ≤1 and a is the hyperbolic speed of sound given by Eq. 36.

3.5. Boundary Conditions: Navier-Stokes characteristic boundary conditions

The Navier-Stokes characteristic boundary conditions [90, 91](NSCBC) are used for non-reflecting inflow and outflow conditions. Details on single-phase multi-component uses of the NSCBC conditions can be found in [92]. Extension of the classic multi-component mixing rules to include multi-phase multi-component flows for the NASG equation of state has been completed [87] and is used in this work.

3.6. High performance computing

The implementation of the proposed models and numerical schemes was completed in the highly parallel Hypersonic Task-based Regent (HTR) solver [93, 94, 95]. HTR is a task-based solver built on the Legion runtime which provides portability to run distributed simulations on both CPUs and GPUs [96].

4. Numerical results

The formulation presented in Section 2.1 is written in a general form and is applicable to flows ranging from single-component single-phase to multi-phase multi-component without changing the numerical scheme, adding redundant equations, or changing the equilibrium assumptions of the four-equation model. The applicability of the formulation for these regimes will be shown with simulations focused on single-phase flows in Appendix D, single-phase multi-component tests in Section 4.1, multi-phase single-component tests in Section 4.2, and multi-phase multi-component tests in Section 4.3. All results, unless otherwise specified, were created using WENO5Z [97] for spatial interpolation, a HLLC solver, and a SSP-RK3 with a temporal CFL of 0.5.

4.1. Single-phase multi-component tests

In this section two single-phase multi-component simulations are shown without interface regularization to verify the applicability of the framework for multi-component gaseous flows. The first test is a shockbubble interaction between helium and an air bubble and is compared against an experiment [98]. The second test is a single-mode Richtmyer-Meshkov instability and compared to past computational studies [7, 99].

16

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed


> **Figure 2: Shock-bubble interaction between air and helium. All units are in meters.**

Material ρ [kg/m3] u [m/s] P [Pa] µ [Pa-s] Sc [-]

Helium 0.166 0.0 101325.0 1.96 × 10−5 0.70 pre-shock air 1.18 0.0 101325.0 1.81 × 10−5 1.0 post-shock air 1.624 115.65 159050.0 1.81 × 10−5 1.0


> **Table 2: Initial conditions for 2D shock-bubble interaction between air and helium**

4.1.1. Shock-bubble interaction: air-helium bubble The interaction of a shock in air with a helium bubble is a well-documented test case for multi-component flows [7, 99, 40]. The schematic showing the setup for this case is shown in Figure 2. The initial thermodynamic state as well as the component viscosity and Schmidt numbers used in this test are shown in Table 2. Lastly, a mixture Prandtl number of 0.71 is used to define the heat conduction and a spatial resolution of (4096 × 1024) was used. For this case the shock-wave speed is 423 m/s and for the setup shown in Figure 2, the shock impacts the helium bubble after 23.6 µs. As this case is between two miscible species the interface regularization term is not active and species mass diffusion is present between the air and helium. Figure 3 compares the evolution of the helium bubble with an experiment [98] at times after the shock first makes contact with the helium bubble. All temporal snapshots show good visual agreement between the numerical simulation and the experiment.

4.1.2. Single-mode Richtmyer-Meshkov instability We consider a 2D single-mode Richtmyer-Meshkov instability (RMI) simulation following the computational setup used in past works for additional code-to-code verification [7, 99]. The thermodynamic quantities used in this test are shown in Table 3. The test consists of a Mach 1.24 shock in air passing through a perturbed interface of SF6 and details on the initial condition are shown in Figure 4. Figure 5 shows the development of the RMI over time. After the shock passes through the perturbed interface the characteristic mushroom shape appears and a mixing layer forms between the SF6 and air. The locations of the spike (farthest left point of the interface), the bubble (farthest right point of the interface), and the mixing zone (difference between the spike and bubble location) are reported for three resolutions. These resolutions are (512×128), (1024×256), and (2048×512). Furthermore, the RMI results are compared with previous numerical experiments in Figure 6. The axis in Figure 6 is scaled to consistently compare with the non-dimensionalization used in previous studies. Figure 6 shows that the three numerical resolutions simulated in this work have converged relative to

17

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed


> **Figure 3: Shock-bubble interaction between air and helium. First row: experiment [98] (with permission). Second row:**

numerical schlieren ln �||∇ρ|| ρ � . Third row: temperature. Final row: density. From left to right the snapshots are taken at times of 72, 102, 245, 427, and 674µs after the shock impacts the helium cylinder.


> **Figure 4: Schematic of initial condition for single-mode Rightmyer-Meshkov instability between air and SF6.**

the locations of the spike and bubble, and the length of the mixing zone over time. Generally, the results from this work match well with results from past published works [7, 99], verifying the implementation and application of the proposed numerical scheme. The small discrepancies between this work and the two references could be attributed to the higher resolutions explored in this work compared to those of the previous studies. As shown in Figure 5, the simulations in this work achieve a strong mesh convergence at the reported resolutions.

18

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

Material ρ [kg/m3] u [m/s] P [Pa] µ [Pa-s] Sc [-]

SF6 6.06 0.0 101325.0 1.5 × 10−5 1.66 pre-shock air 1.18 0.0 101325.0 1.81 × 10−5 1.0 post-shock air 1.66 125.3 164957.0 1.81 × 10−5 1.0


> **Table 3: Initial conditions for 2D single-mode Richtmyer-Meshkov instability**


> **Figure 5: Single-mode Rightmyer-Meshkov instability between air and SF6 over time. First row: density. Second row: numerical schlieren � ln �||∇ρ|| ρ �� . Last row: temperature. The snapshots are taken at rescaled times t∗from left to right of 0, 3.2, and 10.**


> **Figure 6: Characteristic locations over time for single-mode Rightmyer-Meshkov instability between air and SF6. The labels are listed as location of the (spike, bubble, mixing layer) for the fine resolution (2048 × 512)( , , ), medium resolution (1024 × 256) ( , , ), coarse resolution (512 × 128) ( , , ), Terishima data [7] ( , , ), and Hoppe data [99]( , , ).**

4.2. Multi-phase tests To illustrate the applicability of the regularization model combined with ENO-type schemes to singlecomponent multi-phase flows four tests will be shown in this section. The first is a classic one-dimensional gas-liquid Riemann problem to verify the implementation of the model for multi-phase flows with shocks. The second is an isothermal and inviscid water droplet advection in air to verify the interface equilibrium

19

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

condition is satisfied. The third case is a shock in air interacting with a water droplet to verify the implementation of the positivity preserving algorithm as this cases will fail without it. Lastly, an inviscid Mach 100 water jet will be simulated to illustrate the robustness of the framework even when applied to unrealistically difficult flows.

4.2.1. Gas-liquid Riemann problem We consider a common gas-liquid Riemann problem which was first analyzed by [10]. This problem is formulated as a model problem for an underwater explosion in which the left state is highly compressed air and the right state is water at atmospheric pressure. The same setup used by [34] is repeated here with a spatial resolution of (200 × 1) and final time of t = 0.2. The initial condition and material parameters are listed in Table 4 where for this case we use the stiffened gas equation of state for the liquid where b = q = 0. The initial condition is nondimensionalized consistently as done in [10, 34].

Location Material ρ [-] u [-] P [-] µ [Pa-s] P∞ γ CP −1 ≤x < 0.0 post-shock air 1.241 1.0 2.753 0.0 0 1.4 7.756 0 ≤x < 1.0 water 0.991 0.0 3.059 × 10−4 0.0 1.505 5.5 15.4715


> **Table 4: Initial conditions for 1D gas-liquid Riemann problem**


> **Figure 7 shows the results for TENO6 [32], WENO5JS [31], and WENO5Z [97] compared to the exact solution. All schemes perform well without excessive oscillations in density, pressure, and temperature. Small oscillations can be seen in the phasic density plots which are larger for TENO6 than the WENO-type schemes. Additionally, the regularization of volume fraction keeps the interface at nearly constant thickness for all ENO-type schemes.**


> **Figure 7: Gas-Liquid Riemann problem for three ENO-type schemes. Exact solution: ( ), WENO5JS [31] ( ), WENO5Z [97]( ), TENO6 [32]( ).**

20

This preprint research paper has not been peer reviewed. Electronic copy available at: https://ssrn.com/abstract=5372486


# Preprint not peer reviewed

