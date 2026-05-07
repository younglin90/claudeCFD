
# A Finite-Volume Method for Compressible Viscous Multiphase Flows


### Aditya K. Pandare ∗and Hong Luo †

North Carolina State University, Raleigh, NC 27695, United States

A robust and efficient density-based finite volume method is developed for solving the six-equation single pressure system of two-phase flows based on the stratified flow model at all speeds on hybrid unstructured grids. Unlike conventional approaches where an expensive exact Riemann solver is normally required for computing numerical fluxes at the two-phase interfaces in addition to AUSM-type fluxes for single-phase interfaces in order to maintain stability and robustness in cases involving interactions of strong pressure and void-fraction discontinuities, a volume-fraction coupling term for the AUSM+-up fluxes is introduced in this work to impart the required robustness without the need of the exact Riemann solver. The resulting method is significantly less expensive in regions where otherwise the Riemann solver would be invoked. Since traditional multiphase applications involve very low Mach number flows, legacy multiphase codes use pressure-based methods and resort to numerical diffusion and Picard-type iterations to make the system stable. These methods use a segregation procedure to decouple the momentum and energy equations leading to temporal inaccuracies in addition to compressiblity errors. These type of flows cannot be solved using the standard density-based algorithms. The presented method uses a transformation from conservative to primitive variables, and then solves for the primitive variables implicity using a Jacobian-free approach, to be able to efficiently solve for such low Mach number flows. The proposed method is shown to perform well for inviscid and viscous two-phase test cases at a range of Mach numbers. The results indicate that the current density-based method provides an attractive and viable alternative to its pressure-based countepart for compressible two-phase flows at all speeds.


### I. Introduction

Multiphase flows are encountered in various engineering applications such as fuel injection and atomization, flow of coolant around fuel rods in reactors, cavitation around hydrofoils etc. Various ways exist to simulate these multiphase flows. Two broad categories can be identified depending on the way the interfaces between phases are resolved: the Sharp Interface Methods (SIM) or interface tracking methods and the Effective Field Modeling (EFM) or the interpenetrating continua approach. The interface tracking methods, as the name suggests, keep track of every individual interface between the phases through approaches like level-set, volume-of-fluid and front-tracking. This approach is extremely effective when bubble/droplet interface phenomena such as coalescence and break-up are important and necessary to be resolved. On the other hand, when these phenomena aren’t very important and the averaged flow field is of interest (especially in applications with high density of tiny bubbles/droplets), the interpenetrating continua approach is more efficient. The interpenetrating continua approach (or the EFM) is an Eulerian-Eulerian multiphase method, in which interfaces are resolved, not tracked. This makes the method less computation-extensive in regions of high bubble/droplet density. Also, the right usage of phase-transfer terms ensures spontaneous interface generation or dissolution. The two-fluid model1 is one of such EFMs. The two-fluid system is incomplete due to more unknowns than number of equations. Various approaches to resolve this exist. One of them is including an additional equation for volume fraction. This additional equation leads to a 7-equation two-pressure system, which is hyperbolic.2 Although this is a desirable

∗Ph.D Student, Department of Mechanical and Aerospace Engineering, Student Member AIAA. †Professor, Department of Mechanical and Aerospace Engineering, Associate Fellow AIAA.

1 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

2018 AIAA Aerospace Sciences Meeting 8–12 January 2018, Kissimmee, Florida

10.2514/6.2018-1814

Copyright © 2018 by Aditya K Pandare, Hong Luo. Published by the American Institute of Aeronautics and Astronautics, Inc., with permission.

AIAA SciTech Forum

property, the 7-equation model is physical only if applied to flows with closely packed particles/suspensions.3

Another approach is to adopt a single-pressure assumption leading to the Wallis model.4 This model is nonhyperbolic, and non-conservative. A detailed eigenvalue analysis of the Wallis model can be found in St¨adtke’s work.5 Even so, the non-hyperbolic 6-equation model has been used to simulate high Ma inviscid two-fluid flow by Chang, Liou and co-workers.6–8 By adding an interface pressure term suggested by Stuhmiller,9 they prove that the system can be rendered hyperbolic.10

Another way to resolve the non-hyperbolicity and low Ma convergence issue is use of non-hyperbolic methods to solve the equations. These methods are usually termed as pressure-based algorithms since the working variable is pressure, and not the density-fraction (αρ) as used by density-based methods. Pressurebased methods are better conditioned for low Ma flows, since at these speeds, the density is practically constant, and small errors in density (due to spatial or temporal discretizations) translate to large errors in pressure. Also, The pressure-based methods eliminate the speed of sound by reformulating the continuity and momentum equations by assuming that the velocity field is solenoidal (∇·v = 0). This results in a CFL criterion based solely on the flow velocity, independent of the acoustic speed, and larger time-step sizes leading to faster convergence. These pressure-based algorithms have been used successfully to obtain very accurate results for low speed flows, both for single-phase flows11,12 and two-phase flows.13,14 The drawback of the pressure-based methods is that compressiblity can not be rigorously treated, since the incompressibility assumption is deeply ingrained in their formulation. This results in significant errors from the pressurebased algorithms when high-temperature phenomena such as boiling need to be simulated. A better solution algorithm would thus be to use the density-based (fully compressible) method for the multiphase flows, using a preconditioning to “step over” the acoustic time-step limitations. One of the ways to do this is solve the two-fluid system in a fully-implicit way using a primitive variable transformation matrix. Nourgaliev and co-workers15,16 have used a [p, v, T]-formulation to solve near-incompressible flows using fully compressible density-based solver. Using pressure as a primitive variable makes the implicit density-based system better conditioned. Treating the source terms due to interfacial forces and phase-changes implicitly also relaxes the time-step restrictions, leading to faster convergence times. This idea is utilized in this article, improving upon the work by Liou, Chang et al.6 They use the stratified-flow model to discretize the non-conservative terms in the system and an all-speed two-phase extension of the AUSM flux, called the AUSM+-up, in addition to an exact Riemann solver to discretize the inviscid fluxes. The exact Riemann solver was only used in situations where the jump in the volumefraction at the cell-faces is substantial. It has also been proven6 that this choice of flux, coupled with the stratified flow discretization of the pressure flux and other non-conservative terms, satisfies the pressure non-disturbance condition (also known as Abgrall’s criterion or well-balancedness), which is essential for the solver to maintain a stationary contact discontinuity. However, the iterative procedure involved in the exact Riemann solver makes this method quite expensive. Kitamura and Nonomura17 replaced the exact Riemann solver with a two-fluid HLLC flux. This resulted in a relatively inexpensive flux function. However, knowledge of the complete eigenstructure of the system is a must for such a flux function. This might not be easily available, when more complex terms such as virtual mass are included in the governing equations. Other approaches to induce hyperbolicity into the Wallis model include work by Dinh et al.18 which involves an iterative regularization procedure. Also, viscous flows have not been solved using this approach. To the best of the authors’ knowledge, no prior research focusses on this type of density-based solution method for the compressible viscous two-fluid model. In an attempt to obviate the need of the complete eigenstructure, it is desirable to use a numerical flux function such as AUSM. However, the regular AUSM+-up flux suffers from negative pressures in the regions of high pressure shocks interacting with material interfaces. This has been shown by studies by Kitamura.8,17

The solution proposed by Chang and Liou6 was to use an exact Riemann solver, as mentioned before. The exact Riemann solver requires large number of Newton iterations to get a correct middle-zone pressure p∗in the regions of question, thus making it expensive. A flux scheme which is robust enough in these regions, yet not invoking an iterative procedure, is presented here. This involves a modification to the baseline AUSM+- up fluxes via a volume-fraction coupling term in the mass-flux, resulting in what is called the AUSM+-upf flux in this work. This is similar to the Lax-Friedrich type dissipation employed by Houim and Oran19 in the context of granular flows. Further, viscous flows, which are common in practical multiphase scenarios, are also solved using this method. In addition, an implicit primitive variable formulation15 described previously is used to resolve the low Ma convergence issue of the density-based methods. The objective of this work thus, is to develop a density-based method for the solution of the viscous two-fluid single pressure model

2 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

which can be used as a baseline to develop a solver for more complex multiphase scenarios. The target applications for this work involve flows that require interphasic coupling using the drag force. Interphasic mass transfer effects are neglected in the current work. The rest of the article is organised as follows: the governing equations of the two-fluid model and closure laws are discussed in the next section. This is followed by a description of the spatial and temporal discretization and other numerics. The results obtained using the proposed method are presented next. First, inviscid cases are presented to validate the discretization of the stratified-flow model as proposed by Chang and Liou.6 Here, the importance of the additional volume-fraction coupling term in robustness is discussed, using tests that involve strong shock and material interface interactions. Next, viscous cases are presented and compared with boilEulerFOAM 14 (an open-source pressure-based multiphase solver) to ascertain the correctness of the results obtained using the proposed solution method. This is followed by concluding remarks.


### II. The two-fluid model

The two-fluid model uses the interpenetrating continua approach to model two-phase flows. This model requires an averaging procedure to filter-out the local instantaneous fluctations very similar to the Reynolds’ averaging in turbulence. As mentioned earlier, interphasic mass-transfer terms have not been considered in this work. The resulting two-fluid model given by Ishii1 and concisely by Staedtke,5 using k as the index for the two fluids, is as follows:


$$
∂Uk
$$


$$
∂t
+ ∂Fkj
$$


$$
∂xj
= ∂Gkj
$$


$$
∂xj + Pint k + Mk + Sk (1)
$$

where,

Uk =




$$
αkρk
αkρkuki
αkρkEk
$$




$$
 (2)
$$

Fkj =




$$
αkρkukj
αkρkukiukj
αkρkukjHk
$$



+




$$
 0 αkpkδij 0
$$




$$
 (3)
$$

Gkj =




$$
 0 αkτkij αkuklτklj + αkqkj
$$




$$
 (4)
$$

Pint k =




$$
 0 pint k ∂αk
$$


$$
∂xi
−pint
∂αk
$$


$$
∂t
$$




$$
 k = 1, 2. (5)
$$

Mk are interface forces such as drag and virtual mass. Sk represent source terms, for example due to phase transitions and body forces. The viscous stress tensor and heat flux vector are given as,


$$
τij = µ
�∂ui
∂xj
+ ∂uj
$$


$$
∂xi
$$


$$
� −2
$$


$$
3µ∂ul
$$


$$
∂xl
δij,
$$


$$
∂T
∂xj
(6)
$$

where the fluid-index k has been dropped for convenience. Note that the fluid properties (viz. µ, Pr , Cp) are fluid specific. These properties are specified for each phase and kept constant in the scope of this work. The system (1) has 12 unknowns (αk, ρk, uk, pk, Ek, pint k with k = 1, 2) and a total of 9 equations with 6 PDEs, 2 equations of state (EoS) and the constraint on volume fractions,

2 �


$$
αk = 1.
(7)
$$

3 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

The EoS are given in the next subsection. The pressure-equilibrium assumption which yields the Wallis model of two-fluid flow is, pg = pl ≡p. (8)

Another condition that the interfacial pressures should cancel each other if no other stresses such as surface tension are considered at the interface gives,


$$
pint g = pint l ≡pint. (9)
$$

These equations constitute the Wallis two-fluid model or the 6-equation single-pressure model of two-phase flows. pint is explicitly given as a function of the other unknowns. Here, we use the relation given by Stuhmiller,9


$$
pint = p −σ αgαlρgρl
$$


$$
αgρl + αlρg
(10)
$$

where ur = |ul −ug|. This interface pressure term helps restore hyperbolicity to the system.7 This provides the additional 3 equations and the system of equations is closed.


### III. Additional closures

A. Equations of state

The stiffened equations of state are used in this work.


$$
p = (γk −1)(ρE −ρ|u|2
$$


$$
−Pc) −Pc
(11)
$$

a =

�


$$
γ p + Pc
$$


$$
ρ
(12)
$$


$$
T = � γ γ −1
$$


$$
�(p + Pc) ρCp , (13)
$$

with the following properties for the liquid phase:


$$
γl = 2.8
$$


$$
Pcl = 8.5 × 108 Pa
$$

Cpl = 4186 J/(kg · K),

and ideal-gas properties of air for the gas phase.

B. Drag force model

The drag-force interfacial momentum transfer term is modeled as,

Mdrag 1 = −Mdrag 2 = 3


$$
4Cd ρ1α2
$$


$$
db |Ur|Ur, (14)
$$

where the relative velocity is,


$$
Ur = U2 −U1. (15)
$$

Note that subscript 2 represents the dispersed phase and subscript 1 represents the continuous phase. The drag coefficient is set as a constant Cd = 2.0 in this work. The bubble/droplet diameter is denoted by db. A value of db = 1.0 mm is used for the test problems presented here.

C. Virtual mass force model

The presence of virtual mass force improves the stability of the numerical scheme in regions of high accelerations, for flows with a high density ratio.20 This force is modeled as,


$$
Mvm 1 = −Mvm 2 = Cvmρ1α1α2
$$


$$
�D(u2) Dt −D(u1)
$$

Dt


$$
� , (16)
$$

4 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

with the same notation for the subscripts as the drag force. The substantial derivatives are treated partimplicitly, such that they improve the condition number of the system. In other words, the positive contribution to the diagonal entries of the system matrix are treated implicitly, and the rest, explicitly. The virtual mass coefficient is set as Cvm = 0.5.


### IV. Discretization and numerical fluxes

A brief description about the spatial and temporal discretization is given in this section. The reader is referred to the cited articles for additional details.

A. Conservative spatial derivatives

A finite volume approach is used to discretize the 6-equation two-fluid system in space. A least-squares reconstruction on the primitive variables V = (T1, u1, v1, T2, u2, v2, p, α1)T is used to obtain 2nd order accuracy in space. A vertex-based limiter by Kuzmin21 is used to suppress spurious oscillations caused by the reconstruction. The convective fluxes are discretized using a AUSM+-up numerical scheme, which has been detailed in the work of Kitamura et al.8 However, certain modifications are made in the standard single-phase AUSM+-up fluxes. Details about these modifications are given here. It has been noted8 that a hybrid scheme employing exact Riemann solutions is necessary for situations where a strong pressure discontinuity interacts with a void-fraction discontinuity. A high pressure-ratio water-air shocktube, a shock/water-column interaction and a shock/air-bubble interaction have been used to illustate this. However, in this work, a modification to the AUSM+-up flux is used, which makes it possible to solve the above mentioned problems. The new scheme involves an additional coupling between the mass-flux and the volume-fraction of the dispersed phase. The AUSM+-up flux, with this additional coupling term is referred to as the AUSM+-upf; where the f stands for the volume-fraction coupling. Let’s first look at the standard AUSM+-up flux. The AUSM+-up flux developed by Liou22 specifically for all-speed application is a wise choice for two-fluid problems. In the AUSM-type of flux methods, the fluxes are written as,


$$
Fk,L/R = ˙mk,1/2ψk,1/2 + αk,L/Rpk,1/2n1/2, (17)
$$

where ψk,1/2 = (1, u, H)T k,1/2 and it is upwinded in the standard way,


$$
ψk,1/2 =
$$

 

 ψk,L if ˙mk,1/2 > 0


$$
ψk,R
(18)
$$

Note that the pressure flux contribution to the left and right elements is different due to the difference in the volume fractions at the face: αk,L ̸= αk,R. The mass flux and the Mach number of phase-k is given as,

˙mk,1/2 = Mk,1/2ac

 


$$
 αk,Lρk,L if Mk,1/2 > 0,
$$


$$
αk,Rρk,R
(19)
$$


$$
Mk,1/2 = M+ (4)(Mk,L) + M− (4)(Mk,R) + Mk,p, (20)
$$

where the split Mach numbers M ± (m) are,


$$
M± (1)(M) = 1
$$


$$
2(M ± |M|), (21)
$$


$$
M± (2)(M) = 1
$$


$$
4(M ± 1)2, (22)
$$


$$
M± (4)(M) =
$$

 


$$
 M± (1)(M) if |M| ≥1,
$$


$$
M±
(2)(M)(1 ∓16βM∓
(23)
$$

the Mach numbers are defined as,

Mk,L/R = uk,L/R


$$
ac . (24)
$$

5 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

and the common speed of sound given by Chang and Liou6 is used:

1 ac


$$
�α1
ρ1
+ α2
$$


$$
ρ2
$$


$$
� = α1 ρ1a2 1 + α2
$$


$$
ρ2a2 2 . (25)
$$

A value of β = 1/8 is used. The pressure diffusion term Mk,p introduced to treat low Mach number flows is,


$$
Mk,p = −Kp max(1 −M
k, 0)pR −pL
$$


$$
ρk,1/2a2c
(26)
$$


$$
M 2 k = u2 k,L + u2 k,R 2a2c . (27)
$$

The subscript ‘1/2’ has been used for definitions of a and ρ to represent averages of the left and right states. This term is especially important for the two-fluid system where the stiffened gas equation is used. The pressure flux is given as,


$$
pk,1/2 = P+ (5)(Mk,L)pL + P− (5)(Mk,R)pR + pk,u, (28)
$$

where the split Mach numbers for pressure are,


$$
P± (5)(M) =
$$

 




$$
1 M M± (1) if |M| ≥1,
$$


$$
1 M M± (2)[(±2 −M) ∓3MM∓ (2)] otherwise, . (29)
$$

The velocity diffusion pk,u is then defined as,


$$
pk,u = −KuP+ (5)(Mk,L)P− (5)(Mk,R) · � ρk,1/2ac � (uk,R −uk,L). (30)
$$

This choice of fluxes yielded a stable scheme for most of the cases presented. However, for some extreme cases, such as the high pressure-ratio water-air shocktube, shock-watercolumn and shock-bubble interaction and the very low Mach number channel flows, it was observed that a small amount of dissipation in the dispersed phase was necessary to obtain stability in regions of high relative velocities. Note that the two-fluid model with equal phase velocities and pressure, known as the homogeneous two-phase model, is hyperbolic in nature. The Wallis model, which doesn’t assume this however, is nonhyperbolic in its original form.5 It is surmised that when the relative velocities increase to high values, the stratified fluid model tends more to the non-hyperbolic nature and the hyperbolic correction (10) proves to be insufficient to ensure real eigenvalues. Note here that inspite of adding the hyperbolic correction term, the explicit eigen-structure (and hence the exact acoustic speed) for this system is unknown. There is a possibility that, in assuming that the system acoustic speed as a function of the acoustic speeds of the two individual phases alone, a possible effect of relative velocities on the acoustic properties of the system is being neglected. If this is indeed the case, a dissipation term proportional to the relative velocity of the two-phases, would stabilize the model. The dissipation needs to be in the form of an additional coupling between the mass-flux and the volume-fraction. This approach has been utilized by Houim and Oran19 in case of granular two-fluid flows. A similar form of the dissipation term has been utilized in the dispersed phase in this work. The dispersed phase mass flux is given as,


$$
˙m2,1/2 = M2,1/2acα2ρ2 −D2,f, (31)
$$

where Df is the volume-fraction coupling term,

Df = 1


$$
2λr
max(αL, αR)
$$


$$
αcrit
(αRρR −αLρL),
(32)
$$

where the phase subscripts are omitted, and the maximum normal relative velocity is,


$$
λr = max(ur,L, ur,R), (33)
$$


$$
ur = Ur · n. (34)
$$

6 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

Clearly, the relative velocity λr is taken into consideration via this coupling term. Note that although this term derives its true form from the Lax-Friedrichs flux, unlike the latter, it is well-balanced in nature. This means that it satisfies Abgrall’s criterion, also known as the pressure non-disturbance condition.6 Another way to stabilize this model is increasing the drag coefficient. This way, although found to be effective, might yield unphysical results. Saurel and Lemetayer23 use an infinite drag coefficient thus resulting in an absolute velocity relaxation (Ur ≈0), and a well-posed model. This assumption is not valid for applications targeted here. Hence we resort to the volume-fraction coupling for stability purposes. The AUSM+ flux which includes this coupling is referred to as the AUSM+-upf flux henceforth. The effects of this additional term can be seen for high pressure ratio shocks as encountered in the waterair shocktube and the shock-bubble interaction problems. These problems cannot be solved without the aforementioned dissipative terms accounted for, as also reported by Kitamura et al.8 Various other ways have also been used to solve this issue. Chang and Liou6 use an expensive exact Riemann solution in the region |α1,L −α1,R|. Kitamura et al.17 use an HLLC flux in this region to replace the exact Riemann solver used by Chang and Liou. The HLLC flux, albeit more economical than the exact Riemann solver, still requires knowledge of the eigenstructure of the system. The modifications proposed here do not require this, and could potentially be used for other types of equations of state. A detailed discussion of the effects of the proposed modifications will be provided in the numerical results section. The viscous fluxes require a unique value of gradients of velocity components and temperature at the cell-faces. These gradients are computed using the modified-gradient approach as,


$$
∇u|Γ = ∇u|Γ −
∇u|Γ · rΓ −uR −uL
$$


$$
|rR −rL|
$$


$$
rΓ,
(35)
$$

where,


$$
∇u|Γ = 1
$$


$$
2(∇u|L + ∇u|R) (36)
$$


$$
rΓ = rR −rL
$$


$$
|rR −rL|.
(37)
$$

B. Non-conservative spatial derivative

The term Pint is comprised of non-conservative first derivatives in space and time. The non-conservative spatial derivative term p ∂α

∂x is discretized using the stratified flow model. Consider element j for instance, with its faces nface. Then, this term is discretized for element j as,

�


$$
Ωj
pint∇αkdΩ= pint
$$

nface �


$$
αkj→iniΓi,
(38)
$$

where αkj→i is the value of αk reconstructed from the element j to its face i. This clearly shows that the nonconservative term has non-unique values at the element faces, one from each side of the face. The combined discretization of this term and the the pressure flux ∇(αp) ensures that the method is well-balanced24 and that it satisfies the pressure non-disturbance condition.6

Interface momentum transfer and other source terms Mk and Sk are treated as volume-averaged source terms.

C. Time integration

The nonconservative time-derivative term pint ∂α

∂t is treated implicitly. This is done by modifying the unknown vector as,

ˆUk =




$$
αρ
αρui
αρE + pintα
$$





k


$$
. (39)
$$

Now a choice can be made of whether an implicit or explicit time-integration scheme is to be used. Depending on whether an implicit or explicit discretization is used, pint is at time-level n or n + 1 respectively, at all the stages of the (possibly) multi-stage method, as suggested by Kitamura et al.8

7 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

The explicit 3rd order TVD Runge-Kutta25 method is used for time integration for the unsteady testcases. Implementation details can be found in the references by Chang, Kitamura, etc.6,8 Note that a decoding procedure is then used to obtain primitive variables α, p from the conservative variables from Eq. (39). The decoding procedure involves the solution to a nonlinear equation for pressure. For steady-state problems however, the implicit Euler method is used to reach steady-state faster. Flows at very low Mach numbers (Ma < 0.01) are practically incompressible since the density is almost constant. At such minimal compressibilities, small errors in densities translate to very large errors in pressure. Thus, working on primitive variables [p, v, T]15,16 rather than the conservative variables from Eq. (39) leads to a better conditioned (less stiff) system for the linear solver to handle. A transformation to primitive variables, V = (T1, u1, v1, T2, u2, v2, p, α1)T is used in this case. The linear system is then solved by a LUSGS preconditioned GMRES solver as proposed by Luo et al.26 This is a Jacobian-free method, where the left-hand side matrix is computed using divided differencing,


$$
∂R
∂V j
$$


$$
���� n = R(V + ϵ · ej) −R(V )
$$


$$
ϵ
(40)
$$

where, ej is the jth component of the unit vector associated with the component of the variable vector V (i.e. vj) with respect to which the derivative is required. The scalar ϵ is a problem dependent parameter usually set to a small number between 10−6 −10−8. The approximate first-order preconditioner matrix is hand-derived from the spatial-discretization described in the previous section. The previously mentioned α, p decoding is completely avoided when working with primitive variables. Another advantage of using the primitive variables becomes apparent here. The preconditioning matrix involves the Jacobian with respect to the primitive variables, ∂R


$$
∂V as opposed to ∂R
$$

∂ˆU. Considering the above

pressure decoding procedure, the derivative of the pressure flux with respect to the conserved variables, ∂p ∂ˆU is a formidable task, if not impossible. On the other hand, ∂p

∂V is trivial, since pressure is a primitive variable itself. Thus, the calculation of the analytical Jacobian is significantly simplified when primitive variables are considered.


### V. Numerical examples

The method described in the previous sections has been validated by several one and two-dimensional test cases. The results of these tests are presented in this section. First, several inviscid tests from references6,8

are carried out to validate the inviscid two-fluid solver. This is followed by a few viscous test cases, which is the focus of this article. Comparisons with boilEulerFOAM are made to ascertain the correctness of the results obtained using the proposed method. All the unsteady problems use the TVDRK3, whereas the steady-state problems use the implicit Euler method for faster convergence. It should be noted that the authors were unable to find previous work with laminar two-phase flow results to compare with. Due to this, the test cases available to validate the method for these kind of flows are limited. Future work involving inclusion of turbulence models will enable further validation of the method.

A. Inviscid tests

These tests validate the implementation of the stratified flow model for inviscid flows as described by Liou and co-workers.6,8 It consists of problems which test various properties of the discretization of the stratified flow model. The 3-stage Runge-Kutta method is used to integrate in time due to the unsteady nature of these problems.

1. Moving contact discontinuity problem

This test case is used to verify that the solver satisfies the pressure non-disturbance condition. The following initial conditions are used:


$$
(p, αg, uk, Tk)L = (105Pa, 1 −ϵ, 100m/s, 300.0K)
$$

(p, αg, uk, Tk)R = (105Pa, ϵ, 100m/s, 300.0K)


$$
ϵ = 1.0 × 10−7
$$

k = 1, 2

8 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

where the L and R states are to the left and right of x = 0.5 respectively. Fig. 1 shows the void-fraction and pressure obtained after running upto t = 0.003 using a ∆t = 1e −6 on 200 cells. It can be seen that the contact discontinuity is transported to the expected location and the uniform pressure is kept undisturbed.

0

0.2

0.4

0.6

0.8

1

0 0.2 0.4 0.6 0.8 1

Void-fraction

x-coordinate

99000

99500

100000

100500

101000

0 0.2 0.4 0.6 0.8 1

Pressure

x-coordinate


> **Figure 1: Void fraction (left) and Pressure (right) for the moving contact discontinuity**

2. Air/water shock-tube problem

This problem tests the capability of the method to capture shocks. It is a one-dimensional test case with the following initial conditions:


$$
(p, αg, uk, Tk)L = (109Pa, 1 −ϵ, 0m/s, 308.15K)
$$

(p, αg, uk, Tk)R = (105Pa, ϵ, 0m/s, 308.15K)


$$
ϵ = 1.0 × 10−7
$$

k = 1, 2.

A mesh with 500 elements is used. The results at t = 0.2 × 10−3 using a ∆t = 1e −7 are shown in Fig. 2 and 3. Results compare well with references.6,8

0

0.2

0.4

0.6

0.8

1

0 0.2 0.4 0.6 0.8 1

Void-fraction

x-coordinate

0

2x108

4x108

6x108

8x108

1x109

0 0.2 0.4 0.6 0.8 1

Pressure

x-coordinate


> **Figure 2: Void fraction (left) and Pressure (right) for the air-water shocktube**

9 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

-50

0

50

100

150

200

250

0 0.2 0.4 0.6 0.8 1

Uavg

x-coordinate

220

240

260

280

300

320

340

360

380

400

0 0.2 0.4 0.6 0.8 1

Tavg

x-coordinate


> **Figure 3: Average velocity (left) and temperature (right) for the air-water shocktube**

3. Water/air shock-tube problem

This is a one-dimensional test case with the following initial conditions:


$$
(p, αg, uk, Tk)L = (1.0 × 107Pa, ϵ, 0m/s, 308.15K)
$$


$$
(p, αg, uk, Tk)R = (5.0 × 106Pa, 1 −ϵ, 0m/s, 308.15K)
$$


$$
ϵ = 1.0 × 10−7
$$

k = 1, 2.

A mesh with 500 elements is used. The results at t = 0.2 × 10−3 using a ∆t = 1e −7 are shown in Fig. 4 and 5. The results show a good match with the references.

0

0.2

0.4

0.6

0.8

1

0 0.2 0.4 0.6 0.8 1

Void-fraction

x-coordinate

4x106

5x106

6x106

7x106

8x106

9x106

1x107

1.1x107

0 0.2 0.4 0.6 0.8 1

Pressure

x-coordinate


> **Figure 4: Void fraction (left) and Pressure (right) for the water-air shocktube**

10 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

-0.5

0

0.5

1

1.5

2

2.5

3

3.5

0 0.2 0.4 0.6 0.8 1

Uavg

x-coordinate

306.5

307

307.5

308

308.5

309

309.5

0 0.2 0.4 0.6 0.8 1

Tavg

x-coordinate


> **Figure 5: Average velocity (left) and temperature (right) for the water-air shocktube**

The pressure ratio in this problem is 2. A water/air shocktube with a pressure ratio of 103 is also considered in the references. The initial conditions are,


$$
(p, αg, uk, Tk)L = (1.0 × 108Pa, ϵ, 0m/s, 308.15K)
$$


$$
(p, αg, uk, Tk)R = (1.0 × 105Pa, 1 −ϵ, 0m/s, 308.15K)
$$


$$
ϵ = 1.0 × 10−7
$$

k = 1, 2.

As mentioned by Kitamura, this high pressure-ratio water/air shocktube cannot be solved with the AUSM+up fluxes. However, the additional dissipation term of the AUSM+-upf flux allows this high pressure-ratio shocktube to be solved. Using the same mesh and time-steps as before, the results at t = 0.2 × 10−3 are shown in Fig. 6 and 7. Low-frequency oscillations are observed in the pressure profile.

0

0.2

0.4

0.6

0.8

1

0 0.2 0.4 0.6 0.8 1

Void-fraction

x-coordinate

1.0e+04

1.0e+05

1.0e+06

1.0e+07

1.0e+08

1.0e+09

0 0.2 0.4 0.6 0.8 1

Pressure

x-coordinate


> **Figure 6: Void fraction (left) and Pressure (right) for the high PR water-air shocktube**

11 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

-10

0

10

20

30

40

50

60

70

0 0.2 0.4 0.6 0.8 1

Uavg

x-coordinate

285

290

295

300

305

310

315

320

325

330

335

340

0 0.2 0.4 0.6 0.8 1

Tavg

x-coordinate


> **Figure 7: Average velocity (left) and temperature (right) for the high PR water-air shocktube**

A grid convergence study was performed to check whether these oscillations vanish as the grid is refined. It was observed that as finer meshes were used, these oscillations do vanish (see Fig. 8), indicating that the method is indeed grid convergent.

1.0e+04

1.0e+05

1.0e+06

1.0e+07

1.0e+08

1.0e+09

0 0.2 0.4 0.6 0.8 1

Pressure

x-coordinate

500 cells

1000 cells

2000 cells


> **Figure 8: Pressure profiles showing grid-convergence for the high PR water-air shocktube**

4. Ransom’s faucet

This one-dimensional unsteady test case models a jet of air and water confined in a channel. Gravity force (9.8 m/s2) is then applied which accelerates the water column and a void wave is generated. The following initial conditions6 are used:

(p, αg, ug, ul, Tk) = (105Pa, 0.2, 0m/s, 10m/s, 300.00K)

The same inlet conditions are applied. The channel is 12m long. The exact solution for the void fraction is given as,


$$
αg(x, t) =
$$

 




$$
1 −(1−αg(0,0))ul(0,0) √
$$

(ul(0,0))+2gxx , if x < gt2

2 + ul(0, 0) · t


$$
αg(0, 0), otherwise
$$

A grid of 500 cells and a time-step of ∆t = 2.5e −5 s are used. The void fractions at t = 0.1s, 0.3s and 0.5s are shown in Fig. 9. It can be seen that the propagation of the void wave is computed accurately.

12 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

0.15

0.2

0.25

0.3

0.35

0.4

0.45

0.5

0 2 4 6 8 10 12

Void-fraction

x-coordinate

t = 0.1 s t = 0.3 s t = 0.5 s

Exact


> **Figure 9: Void fraction for the Faucet problem at t = 0.1s, 0.3s and 0.5s with exact solution at 0.5s**

0.15

0.2

0.25

0.3

0.35

0.4

0.45

0.5

0 2 4 6 8 10 12

Void-fraction

x-coordinate

nx = 500 nx = 1000 nx = 10000

Exact


> **Figure 10: Void fraction for the Faucet problem using 500, 1000 and 10000 cells at 0.5s**

It is well known that the faucet problem diverges if the mesh used is too fine. A grid refinement study was performed to test this. It can be seen from Fig. 10 that the solver is able to obtain a stable solution even on a grid with 10000 cells. However, convergence to the exact solution is not observed even with such a fine grid. This effect is due to the interface pressure term, which adds dissipation to enhance the stability of the method, somewhat sacrificing the accuracy.

5. Shock/water-column interation

A shock in air impacting a water-column (or a 2D droplet) is simulated in this problem. The droplet has radius r = 3.2mm and is centered at the origin. Since this problem is symmetric about the X-axis, flow over only the top half of the droplet is simulated, and the symmetry condition is imposed at the bottom boundary. The droplet is resolved using 200 × 100 isotropic cells in the domain [−5mm, 5mm] × [0mm, 5mm], so that the grid spacing is ∆xmin = ∆ymin = 0.05mm in this region. The rest of the grid is such that 450 × 150 total cells are used in the overall domain [−15mm, 20mm] × [0mm, 15mm]. The initial conditions are:7

(p, αg, uk, Tk)L = (2.35438 × 105Pa, ϵ, 225.86m/s, 381.85K) for x ≤4mm

(p, αg, uk, Tk)R = (1.0 × 105Pa, ϵ, 0m/s, 293.15K) for x > 4mm, except for


$$
x2 + y2 < (3.2mm)2, where αg = 1 −ϵ
$$


$$
ϵ = 1.0 × 10−5
$$

k = 1, 2.

13 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

These conditions result in a shock moving at Ma = 1.47, which impacts the droplet at t ≈1.5µs. A time-step of ∆t = 1.25e −9 s is used. A smooth transition of the volume fraction at the interface of the droplet is necessary. A width of ±2∆xmin is used for the transition region. The curve used to fit the volume fraction in this region is the same as the blending function used for the vanishing phase.


$$
αg|blended = G(ψ2)ϵ + (1 −G(ψ2))(1 −ϵ),
$$


$$
G(ψ2) = −ψ2
2(2ψ2 −3),
$$


$$
ψ2 =
$$


$$
� x2 + y2 −(r −2∆xmin)
$$


$$
4∆xmin
r −2∆xmin ≤
x2 + y2 ≤r + 2∆xmin.
$$

The left boundary is set as the inlet and the right boundary is the outlet. The top boundary is a slip-wall. These boundaries are sufficiently far from the droplet, such that their influence can be neglected. The AUSM+-up fluxes diverge after ≈6.25µs of flow-time. However, the AUSM+-upf fluxes can solve this problem upto a flow-time of ≈15µs. Thus, only the results using the AUSM+-upf are reported. Pressure and numerical Schlieren contours at t = 6.25µs, t = 10µs and t = 18.75µs are shown in Fig. 11, 12 and 13 respectively. Pressure contours are plotted between 1e + 5 Pa and 4e + 5 Pa. The numerical Schlieren function is computed as (1 + α2 l ) log(1 + |∇ρ|) and the range used for plotting its contours is 4 to 20.

6. Shock/air-bubble interation

The same grid-setup used for the shock-droplet interaction problem is used to simulate a shock in water impacting an air-bubble. The initial conditions are:6

(p, αg, uk, Tk)L = (1.6 × 109Pa, ϵ, 661.81m/s, 595.14K) for x ≤4mm

(p, αg, uk, Tk)R = (1.01325 × 105Pa, ϵ, 0m/s, 292.98K) for x > 4mm, except for


$$
x2 + y2 < (3.2mm)2, where αg = 1 −ϵ
$$


$$
ϵ = 1.0 × 10−3
$$

k = 1, 2.

These conditions result in a shock moving at Ma = 1.51, which impacts the droplet at t ≈0.3µs. A time-step of ∆t = 3.125e −10 s is used. Again for this problem, the AUSM+-up fluxes diverge as soon as the shock hits the bubble. However, the AUSM+-upf fluxes can solve this problem without any divergence. Results using the AUSM+-upf are reported here. Pressure and numerical Schlieren contours at t = 1.25µs, t = 2.5µs and t = 3.75µs are shown in Fig. 14, 15 and 16 respectively. Pressure contours are plotted between 1e + 8 Pa and 2e + 9 Pa. The numerical Schlieren function, computed as log(1 + |∇ρ|), is plotted between 8 to 14. The contours show reasonable agreement with the references.6,8,17

This test problem is now solved on an unstructured mesh with 150,000 elements, a section of which is shown in Fig. 17. The mesh is refined near the bubble location so that complicated physical phenomena can be resolved. The numerical Schlieren contours at 2.4 µs and 3.824 µs are shown in Fig. 18. The water-shock can be clearly seen transmitted into the air-bubble in the first picture. In the second picture, this air-shock is seen partially reflected into the bubble. The rest of it is transmitted out of the bubble. It is noteworthy here that the bubble has collapsed onto itself, without any divergence of the solver. This robustness is the effect of the modification of the fluxes.

14 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814


> **Figure 11: Pressure and numerical Schlieren contours for the shock/water-column interaction at t = 6.25µs**


> **Figure 12: Pressure and numerical Schlieren contours for the shock/water-column interaction at t = 10µs**


> **Figure 13: Pressure and numerical Schlieren contours for the shock/water-column interaction at t = 18.75µs**

15 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814


> **Figure 14: Pressure and numerical Schlieren contours for the shock/bubble interaction at t = 1.25µs**


> **Figure 15: Pressure and numerical Schlieren contours for the shock/bubble interaction at t = 2.5µs**


> **Figure 16: Pressure and numerical Schlieren contours for the shock/bubble interaction at t = 3.75µs**

16 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814


> **Figure 17: Unstructured mesh used for shock-bubble interaction problem**


> **Figure 18: Numerical Schlieren: Left: 2.4µs: Water-shock transmitted into bubble. Right: 3.825 µs: Bubble collapses onto itself and the air-shock is transmitted out.**

This ends the validation of the inviscid two-fluid method proposed by Liou, Chang et al. The next few test cases attempt at validating the viscous flow solver proposed in this article.

B. Viscous tests

These problems test the viscous capabilities of the presented method. The problems considered here are steady-state and the implicit solver is used for faster convergence. It was observed that the AUSM+-up fluxes diverge as large gradients in volume fraction start developing near stagnation points. This has originally led the belief that a volume-fraction coupling is necessary in the first place. In the following sections, only the results using the AUSM+-upf are reported.

1. Bubbly flow in a channel

A bubbly flow comprising of water and 2% air is simulated through a channel of height 25.4 mm and length 4 m at standard atmospheric pressure (1.013 × 105 Pa). A mesh with 20 cells in the height-wise direction and 200 cells in the length-wise direction is used. The Reynolds number of 100 based on water properties and channel height is used. The other problem details are given in Table 1.

17 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

Property Water Air T 298 298 Pr 6.15305 0.7 cp 4186.0 1004.5 µ 8.9313E-04 1.78978E-05 γ 2.8 1.4 Pc 8.5e+8 0.0


> **Table 1: Bubbly channel flow problem details**

Subsonic inlet and outlet boundary conditions based on the above Reynolds number and pressure are used. The velocity profile obtained at the outlet boundary after steady-state convergence is shown in Fig. 19.

0

0.001

0.002

0.003

0.004

0.005

0.006

-0.015 -0.01 -0.005 0 0.005 0.01 0.015

x-Velocity

x-coordinate

Exact boilEulerFOAM

Density-based


> **Figure 19: Velocity profiles compared with BoilEulerFOAM results and “exact” solution**

The results are compared with those obtained by BoilEulerFOAM 14 and the exact solution for a singlephase flow of water in a channel at Re = 100. The results are expected to be close to the exact solution for the single-phase flow because of the low void-fraction and absence of any phase-change terms. This is found to be true for both, the current density-based method and the pressure-based BoilEulerFOAM solver. This problem was solved with gravitational acceleration g = 9.81 m/s. The pressure drop for a single phase water flow using the same conditions can be analytically determined to be 9815.82 Pa/m. For this two-phase flow, the pressure-drop obtained by the simulation was 10694.29 Pa/m. The difference is surmised to be due to the drag-force, which would increase the pressure-drop. Spatial-discretization errors also contribute to this difference. The important observation from this test is that the current density-based method is able to solve flows at very low Mach numbers (current mach number for continuous phase Ma1 = 2.3349 × 10−4). It was observed that this is possible only when the dissipation given in Eq. (31) is used and the 6 equation system is solved for primitive variables. The all-speed flux modification alone is not sufficient to be able to solve viscous flows at such low Mach numbers.

2. Droplet flow in a channel

A droplet flow comprising of 2% water and 98% air is simulated through a channel of same dimensions as the one used for the previous test. The Reynolds number of 5.89 based on air properties and channel height is used. Fluid properties from Table 1 are used. Subsonic inlet and outlet boundary conditions based on the above Reynolds number and pressure are used. The velocity profile obtained at the outlet boundary after steady-state convergence is shown in Fig. 20.

18 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

0

0.001

0.002

0.003

0.004

0.005

0.006

-0.015 -0.01 -0.005 0 0.005 0.01 0.015

x-Velocity

x-coordinate

Exact boilEulerFOAM

Density-based


> **Figure 20: Velocity profiles compared with BoilEulerFOAM results and “exact” solution**

The results are compared with those obtained by BoilEulerFOAM 14 and the exact solution for a singlephase flow of air in a channel at Re = 5.89. The results show good agreement with boilEulerFOAM and the exact single-phase velocity profiles.

3. Flow over a flat plate

The flow of 2% water dispersed in air over a flat plate is considered in this test. The flow is simulated at Re = 66178 and Ma = 0.10115, which is based on the continuous phase (gas) properties. The same fluid properties given in Table 1 are used here. The left and right boundaries as set as subsonic inlet and subsonic outlet respectively. The top boundary is set as a freestream. The bottom boundary is a symmetry wall for x ∈[−0.5, 0] and an adiabatic no-slip wall for x ∈[0, 1]. Standard atmospheric pressure is used for the subsonic outlet boundary condition. The x−and y−velocity profiles obtained from the proposed method and boilEulerFOAM at various stations along the plate are compared in Fig. 21, 22 and 23 respectively. The velocities and y−coordinates have been nondimensionalized by the freestream mixture Reynolds number Rem as,


$$
η = y � Rem/x, (41)
$$


$$
˜u = u U∞ , (42)
$$


$$
˜v = v U∞
$$


$$
� xRem, (43)
$$


$$
Rem = ρmU∞
$$


$$
µm , (44)
$$

where the subscript m denotes mixture properties, weighted according to volume fractions.

0

0.2

0.4

0.6

0.8

1

0 1 2 3 4 5 6 7 8 9

Vx

eta-coordinate

DensityBased boilEulerFOAM

0

0.2

0.4

0.6

0.8

1

0 2 4 6 8 10

Vy

eta-coordinate

DensityBased boilEulerFOAM


> **Figure 21: X-component (left) and Y-component (right) of the velocity at x = 0.25**

19 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

0

0.2

0.4

0.6

0.8

1

0 1 2 3 4 5 6 7 8 9

Vx

eta-coordinate

DensityBased boilEulerFOAM

0

0.2

0.4

0.6

0.8

1

0 2 4 6 8 10

Vy

eta-coordinate

DensityBased boilEulerFOAM


> **Figure 22: X-component (left) and Y-component (right) of the velocity at x = 0.5**

0

0.2

0.4

0.6

0.8

1

0 1 2 3 4 5 6 7 8 9

Vx

eta-coordinate

DensityBased boilEulerFOAM

0

0.2

0.4

0.6

0.8

1

0 2 4 6 8 10

Vy

eta-coordinate

DensityBased boilEulerFOAM


> **Figure 23: X-component (left) and Y-component (right) of the velocity at x = 1.0**

Note that there is no ‘exact’ solution for this test case to compare these results with. The x-component velocity profiles obtained from the current method show good agreement with boilEulerFOAM. However, the y-component velocities show some differences, although they follow the same general trend as those obtained by boilEulerFOAM.


### VI. Conclusions and Outlook

A density-based finite volume method has been developed for compressible two-phase flows based on the effective-field model at all speeds. The regular AUSM+-up suffers from negative pressure in regions where strong shocks interact with material interfaces. A modification to the AUSM+-up mass-fluxes has been proposed to address this issue by adding appropriate diffusion via a volume-fraction coupling term. This modification is shown to make the method effective and robust for such situations without any iterative procedures, as needed by the exact Riemann solver, thus keeping the computational cost low. A primitive variable formulation has been presented in the implicit method in order to be able to effectively solve low Mach number two-phase viscous flows. A number of numerical experiments for a wide range of Mach numbers have been conducted to assess the performance and robustness of the developed finite volume method for both inviscid and viscous two-phase flow problems. The numerical results demonstrate the great potential of this density-based finite volume method as a true all-speed two-phase solution algorithm, which on one hand is able to compute high-speed flows like a conventional density-based method; and on the other, is able to solve low-speed viscous flows with an accuracy comparable to pressure-based methods. Further work will focus on extension of the method by inclusion of interphase mass-transfer and turbulence modeling.

20 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814


### Acknowledgements

This research was partially supported by the Consortium for Advanced Simulation of Light Water Reactors (http://www.casl.gov), an Energy Innovation Hub (http://www.energy.gov/hubs) for Modeling and Simulation of Nuclear Reactors under U.S. Department of Energy Contract No. DE-AC05-00OR22725. The authors would like to thank Dr. C.-H. Chang, Dr. N. Dinh, Dr. J. R. Edwards, and Dr. R. Nourgaliev for fruitful discussions. The authors would also like to thank Mr. Chad Rollins for providing us the results by boilEulerFOAM which were used for comparison for the viscous test cases.


## References

1M Ishii and T Hibiki. Thermo-fluid dynamics of two-phase flow. Springer Science & Business Media, 2010. 2R Saurel and R Abgrall. A multiphase Godunov method for compressible multifluid and multiphase flows. Journal of Computational Physics, 150(2):425–467, 1999.

3D Lhuillier, C-H Chang, and TG Theofanous. On the quest for a hyperbolic effective-field model of disperse flows. Journal of Fluid Mechanics, 731:184–194, 2013.

4GB Wallis. One-dimensional two-phase flow. McGraw-Hill Companies, 1969. 5H St¨adtke. Gasdynamic aspects of two-phase flow: Hyperbolicity, wave propagation phenomena and related numerical methods. John Wiley & Sons, 2006.

6C-H Chang and M-S Liou. A robust and accurate approach to computing compressible multiphase flow: Stratified flow model and AUSM+-up scheme. Journal of Computational Physics, 225(1):840–873, 2007.

7M-S Liou, C-H Chang, L Nguyen, and T G Theofanous. How to solve compressible multifluid equations: a simple, robust, and accurate method. AIAA Journal, 46(9):2345–2356, 2008.

8K Kitamura, M-S Liou, and C-H Chang. Extension and comparative study of AUSM-family schemes for compressible multiphase flow simulations. Communications in Computational Physics, 16(03):632–674, 2014.

9JH Stuhmiller. The influence of interfacial pressure forces on the character of two-phase flow model equations. International Journal of Multiphase Flow, 3(6):551–560, 1977.

10C-H Chang, S Sushchikh, L Nguyen, M-S Liou, and T Theofanous. Hyperbolicity, discontinuities, and numerics of the two-fluid model. In 5th ASME/JSME Fluids Engineering Summer Conference, 10th International Symposium on Gas-Liquid Two-phase Flows. San Diego, 2007.

11L Botti and D A Di Pietro. A pressure-correction scheme for convection-dominated incompressible flows with discontinuous velocity and continuous pressure. Journal of computational physics, 230(3):572–585, 2011.

12AK Pandare and H Luo. A hybrid reconstructed discontinuous Galerkin and continuous Galerkin finite element method for incompressible flows on unstructured grids. J. Comput. Phys., 322:491–510, 2016.

13A Guelfi, D Bestion, M Boucker, P Boudier, P Fillion, M Grandotto, J-M H´erard, E Hervieu, and P P´eturaud. Neptune: a new software platform for advanced nuclear thermal hydraulics. Nuclear Science and Engineering, 156(3):281–324, 2007.

14C Rollins, H Luo, and N Dinh. Development of multiphase CFD flow solver in OpenFOAM. In APS Meeting Abstracts, 2016.

15R Nourgaliev, H Luo, B Weston, A Anderson, S Schofield, T Dunn, and J-P Delplanque. Fully-implicit orthogonal reconstructed discontinuous Galerkin method for fluid dynamics with phase change. Journal of Computational Physics, 305:964– 996, 2016.

16H Park, R R Nourgaliev, R C Martineau, and D A Knoll. On physics-based preconditioning of the Navier-Stokes equations. Journal of Computational Physics, 228(24):9131–9146, 2009.

17K Kitamura and T Nonomura. Simple and robust HLLC extensions of two-fluid AUSM for multiphase flow computations. Computers & Fluids, 100:321–335, 2014.

18TN Dinh, RR Nourgaliev, and TG Theofanous. Understanding the ill-posed two-fluid model. In Proceedings of the 10th international topical meeting on nuclear reactor thermal-hydraulics (NURETH03), 2003.

19RW Houim and ES Oran. A multiphase model for compressible granular–gaseous flows: formulation and initial tests. Journal of Fluid Mechanics, 789:166–220, 2016.

20RT Lahey, LY Cheng, DA Drew, and JE Flaherty. The effect of virtual mass on the numerical stability of accelerating two-phase flows. International Journal of Multiphase Flow, 6(4):281–294, 1980.

21D Kuzmin. A vertex-based hierarchical slope limiter for p-adaptive discontinuous Galerkin methods. Journal of computational and applied mathematics, 233(12):3077–3085, 2010.

22M-S Liou. A sequel to AUSM, part II: AUSM+-up for all speeds. Journal of Computational Physics, 214(1):137–170, 2006.

23R Saurel and O Lemetayer. A multiphase model for compressible flows with interfaces, shocks, detonation waves and cavitation. Journal of Fluid Mechanics, 431:239–271, 2001.

24C Par´es. Numerical methods for nonconservative hyperbolic systems: a theoretical framework. SIAM Journal on Numerical Analysis, 44(1):300–321, 2006.

25S Gottlieb and C-W Shu. Total variation diminishing Runge-Kutta schemes. Mathematics of computation of the American Mathematical Society, 67(221):73–85, 1998.

26H Luo, JD Baum, and R L¨ohner. A fast, matrix-free implicit method for compressible flows on unstructured grids. Journal of Computational Physics, 146(2):664–690, 1998.

21 of 21

American Institute of Aeronautics and Astronautics

Downloaded by COLUMBIA UNIVERSITY on January 24, 2018 | http://arc.aiaa.org | DOI: 10.2514/6.2018-1814

