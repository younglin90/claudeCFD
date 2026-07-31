Journal of Computational Physics 302 (2015) 548–566 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0001-01.png)


Contents lists available at ScienceDirect 

www.elsevier.com/locate/jcp 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0001-05.png)


## Maintaining interface equilibrium conditions in compressible multiphase flows using interface capturing 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0001-07.png)


## Shahaboddin Alahyari Beig[∗] , Eric Johnsen 

_Mechanical Engineering Department, University of Michigan, Ann Arbor, MI 48109, USA_ 

a r t i c l e i n f o a b s t r a c t _Article history:_ An accurate treatment of material interfaces in compressible multiphase flows poses Received 14 April 2015 important challenges for high-resolution numerical methods. Although high-order interAccepted 8 September 2015 face-capturing schemes have been used to accurately simulate gas/liquid interfaces Available online 15 September 2015 with the Euler equations, these methods can result in temperature spikes at material discontinuities. While this phenomenon is not problematic for Euler simulations, it gives _Keywords:_ Compressible multiphase flows rise to numerical errors when heat conduction is included. In this work, we identify the Shock waves and interfaces source of these errors and propose a methodology to prevent their occurrence for various Stiffened equation of state models used to represent gas/liquid interfaces in compressible flows based on a “singleInterface capturing fluid” formulation, in which interfaces are represented by discontinuities in the material Solution-adaptive method properties. Our focus lies in materials (gases and liquids primarily, but also solids) that can be described by a stiffened equation of state, though our approach is generalizable to other equations. We show that numerical approaches that prevent pressure oscillations at interfaces may generate temperature errors, which affect the energy (and pressure) through the heat conduction term. We demonstrate that the material properties entering the equation of state must be computed according to suitable transport equations in conservative or non-conservative forms; the pressure and temperature must be calculated based on the appropriate properties. To verify the analysis and compute problems with gas/liquid interfaces of relevance, we develop a three-dimensional, high-order accurate, solution-adaptive finite difference framework. In particular, we show that temperatures and pressures may be significantly overestimated in calculations of shock-induced bubble collapse in water if temperature errors are not prevented. 

© 2015 Elsevier Inc. All rights reserved. 

## **1. Introduction** 

Compressible multiphase flows are central to a number of engineering applications, including cavitation erosion and high-speed combustion. One of the main challenges in accurately simulating these flows lies in simultaneously representing shock waves, interfaces separating fluids of large density ratios and physical diffusion processes, due to spurious numerical errors commonly generated at interfaces, which may eventually affect the entire flow field. The present work focuses on developing Eulerian approaches to accurately simulate shock waves and gas/liquid interfaces, with viscous and heat diffusion included. 

- [Corresponding][author.] 

> _E-mail addresses:_ alahyari@umich.edu (S. Alahyari Beig), ejohnsen@umich.edu (E. Johnsen). 

> 0021-9991/© 2015 Elsevier Inc. All rights reserved. 

http://dx.doi.org/10.1016/j.jcp.2015.09.018 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

549 

Numerical methods for Eulerian simulations of compressible flows with interfaces typically fall in one of two categories, tracking or capturing. In this article, we focus on the latter because it is relatively simple to implement even for high-order methods and is a logical approach to treat physical diffusion; tracking, which includes front-tracking [1] and level-set [2] methods, will not be discussed further here. Similarly to shock capturing, interfaces between fluids of different composition can be captured by regularizing them over a few grid points, while maintaining the correct jump conditions. By adding one transport equation for mass conservation of one of the fluids, an extension of the Euler equations to multiple fluids/phases is seemingly straightforward, as such an equation can be solved in conservative form with standard shock-capturing techniques. However, such a naive implementation has long been known to give rise to spurious pressure oscillations for isolated interfaces between fluids of different material properties (i.e., properties entering the equation of state) [3,4]. Furthermore, since material interfaces are linearly degenerate, there is no physical mechanism to steepen interfaces, unlike shock waves. Thus, to prevent interfaces from being overly smeared by numerical diffusion, high-order solution-adaptive [5,6] or sharpening [7–9] techniques are often used in practice. 

In this context, Abgrall [3] was the first to recognize that, for interfaces separating two gases of different specific heats ratios _γ_ , an additional transport equation solved for a distinct function of _γ_ in non-conservative (advection) form prevents such oscillations. Shyue [4] later expanded this idea to solving a transport equation for the mass fraction, again in non-conservative form, and to liquids and solids obeying a stiffened equation of state. Johnsen and Colonius [10] further extended these approaches to high-order Weighted Essentially Non-Oscillatory (WENO, [11]) methods to simulate nonspherical bubble collapse [12], which Coralic and Colonius [13] further refined. Such high-order finite volume methods can be computationally expensive in multiple dimensions. To address this difficulty, finite difference (for gases only, [5,14,15]) and discontinuous Galerkin [6] methods have been proposed, in which high-order limiting is applied only at discontinuities. In simulations of the compressible Navier–Stokes equations for two gases with different specific heats ratios, Johnsen and Ham [16] noticed that an inconsistent treatment of temperature causes similar errors and significant temperature undershoots due to the coupling via the heat diffusion term; they proposed approaches to overcome these problems based on _γ_ or mass fraction formulations. Although temperature errors may occur in Euler simulations, they have no influence on the results since temperature is a derived quantity. However, such temperature errors are problematic when accounting for heat diffusion, reacting flows, phase change and other temperature-dependent phenomena. 

Recent developments in interface capturing for compressible multiphase flows originated from the seven-equation twophase flow model [17], in which balance equations for mass, momentum and energy of each phase, as well as an equation for volume fraction evolution, are solved. The additional volume fraction equation prevents the occurrence of spurious pressure oscillations. For many problems of practical importance, five-equation models (e.g., that in [18]) describe the physics accurately, in which pressure and velocity equilibria between the phases are assumed; thus, equations for mass balance of each phase, for total momentum and energy, and for the volume fraction evolution are solved. This latter model and extensions thereof have been used to study a wide range of phenomena [18–26]. With these models, a consistent and efficient high-order extension to accurately include heat diffusion and predict temperature has yet to be proposed. 

The objective of the present work is to develop a treatment for temperature in compressible multiphase flows that is physically consistent and efficient, and that does not produce spurious errors in simulations of gas/liquid interfaces and shocks, with viscous and heat diffusion included. Our approach is general in that it applies to _γ_ (as in [3]), mass fraction (as in [4]) and volume fraction (e.g., five-equation, [18]) models. Through our analysis, we identify the causes for numerical errors caused by an inconsistent treatment of temperature using high-order shock-capturing schemes and show how these errors can be prevented so that accurate simulations with physical diffusion can be performed. The resulting 3D finite difference scheme is high-order accurate, conservative and prevents pressure and temperature errors. Our contribution advances the current understanding of compressible multiphase flows in that it generalizes the methodology of Johnsen and Ham [16] for gases, in which temperature errors are prevented, to gas/liquid flows and different capturing approaches and extends the work of Coralic and Colonius [13] to prevent temperature errors in such flows. The article is organized as follows. In Section 2, the physical model is presented, followed by the numerical models (Section 3). In Section 4, we discuss the causes for spurious pressure and temperature errors in the presence of heat diffusion and propose an approach to prevent them. We briefly describe the numerical implementation in Section 5 and verify it with rigorous test problems in Section 6. 

## **2. Physical model** 

## _2.1. Equations of motion_ 

Assuming no mass transfer or surface tension, the compressible Navier–Stokes equations govern the gas/liquid flows of interest: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0002-09.png)


_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

550 

## **Table 1** 

Relevant constants for the stiffened equation of state [29]. 

|Parameter|Air||Water|
|---|---|---|---|
|_n_|1.4||2.35|
|_B_|0||1 GPa|
|_c_|0.718|kJ/kg K|1.816 kJ/kg K|
|_q_|0||−1_._167 kJ/kg|


where _ρ_ is the density, _ui_ the velocity vector, _p_ the pressure, _E_ = _ρe_ + _ρu_[2] _j[/]_[2][the][total][energy,] _[e]_[the][internal][energy][and] _δij_ the identity tensor. The viscous stress tensor _τij_ and heat flux _Q j_ are given by: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-06.png)


where _μ_ is the dynamic shear viscosity, _μB_ the bulk viscosity and _κ_ the heat conductivity. 

## _2.2. Equation of state_ 

A relation between pressure, temperature and internal energy valid for gases and liquids is required. Although homogeneous equilibrium and tabular relationships have been used for cavitating flows [27,28], the stiffened equation of state introduced by Le Métayer et al. [29] is a simple and sufficiently accurate model, which has been validated against experiments for shock propagation in water and certain solids, and has been used to simulate cavitating flows [30,31]. The relationships between pressure, temperature and internal energy are given by: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-10.png)


where _n_ , _B_ , _q_ , and _c_ are material properties fit to experimental data. For air and water, the relevant constants take the values shown in Table 1. In the limit of ideal gases, _n_ = _γ_ represents the specific heat ratio, _c_ = _c v_ is the specific heat at constant volume, and _B_ and _q_ are zero, such that the ideal gas law is recovered. For multiphase flows, we follow a “single-fluid” formulation, in which the same thermodynamic relationship holds in the entire domain, with material interfaces denoted by changes in the material properties, which are advected by the flow. 

## _2.3. Multifluid modeling_ 

## _2.3.1. Definitions and basic relations_ 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-14.png)


where the superscripts _(k)_ denote phase/fluid _k_ . Similarly, the internal energy of the mixture per unit mass _e_ is: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-16.png)


The mass fraction and volume fraction are related by: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-18.png)


For ideal gases, with _γ_ equivalent to _n_ , the following relationship further holds: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0003-20.png)


In the above relationships, _z[(][k][)]_ , _α[(][k][)]_ , _ρ[(][k][)]_ and _e[(][k][)]_ may vary in space and time, but _γ[(][k][)]_ and _M[(][k][)]_ do not. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

551 

## _2.3.2. Mixture relations_ 

Although we focus on gas/liquid interfaces that are initially sharp, capturing regularizes these interfaces over a few grid points, so that mixture regions exist. For the transport coefficients, we use the mixture relations of Perigaud and Saurel [21]: _μ_ =[�] _k[α][(][k][)][μ][(][k][)]_[and] _[κ]_[=][ �] _k[α][(][k][)][κ][(][k][)]_[.][For][the][thermodynamic][quantities,][we][seek][to][express][the][material][properties] that enter the equation of state in an appropriate form for mixtures. Substituting the pressure-based stiffened equation (3a) into Eq. (5) yields: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-04.png)


while substituting the temperature-based stiffened equation (3b) into Eq. (5) yields: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-06.png)


Assuming isobaric ( _p[(][k][)]_ = _p_ ) and isothermal ( _T[(][k][)]_ = _T_ ) closure between the phases [19], it follows that 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-08.png)


As a result, we can reformulate _ρe_ in terms of volume fractions: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-10.png)


A similar procedure can be followed using the energy relation based on the mass fraction in Eq. (5). For the pressure-wise case, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-12.png)


while for the temperature-wise relation: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-14.png)


Likewise, Eqs. (12) and (13) can be rearranged as: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-16.png)


Thus, _ρe_ can be written in terms of mass fractions: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-18.png)


## _2.3.3. Transport equation_ 

With the present “single-fluid” formulation, changes in composition are denoted by changes in material properties, which depend on the mass and/or volume fraction. From fundamental principles, the mass conservation equation for either of the phases, assuming no mass transfer, is: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0004-21.png)


_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

552 

The continuity equation can be used to write this equation in advection form: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0005-03.png)


where _f_ is any function of _z_ . 

## **3. Numerical models** 

Eqs. (1), (2), (3) and (16), along with appropriate relations between the mass fraction and the material properties in the equation of state form a closed system. However, the discretization of this system may result in spurious pressure oscillations for flows with variable _n_ if care is not taken [3,4]. Three main interface-capturing approaches have been used to prevent such errors, by solving the transport equation in a different form, usually non-conservative: 

- _γ_ -based approach: Here, transport equations for specific functions of the material properties entering the equation of state must be solved: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0005-08.png)


This approach was proposed by Abgrall [3] for gases and extended to the stiffened equation of state by Shyue [4]. If needed, mass and/or volume fraction can be computed from the expressions in Section 2.3.2, as long as there are only two components with different _n_ . It requires additional transport equations for different material properties in the equation of state (e.g., [32]). 

- Volume fraction approach (five-equation model): Rather than solving transport equations for each of the material properties, Eq. (16) is rewritten in terms of the volume fraction (see Appendix A): 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0005-11.png)


where _a[(][k][)]_ is the sound speed in phase/fluid _k_ . This approach written as such was introduced by Murrone and Guillard [23] and is not restricted to the stiffened equation of state. Eq. (19b) is required to prevent pressure oscillations. For sharp-interface problems, _�kk_ ′ is commonly set to zero [19,21,33], which corresponds to the limit of infinite resolution. We also follow this convention. 

- Mass fraction approach: Eq. (20) is solved, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0005-14.png)


This approach was introduced by Shyue [4]; an alternate form of Eq. (7) to relate _z_ to _γ_ (for gases) in which it is implicitly assumed _M_ 1 = _M_ 2 is necessary to prevent pressure oscillations, which is not true in general. This approach is not restricted to the stiffened equation of state, and only one transport equation is needed for each extra component/phase. 

Although these approaches prevent the generation of pressure errors, they do not necessarily maintain temperature equilibrium. In the case of gases for instance, Johnsen and Ham [16] showed that the mass fraction approach must be modified to prevent temperature errors. This issue is not problematic for Euler simulations since temperature is a derived quantity that does not enter the equations; however, they matter for Navier–Stokes simulations, as failure to maintain temperature equilibrium generates errors in the energy (and thus pressure) through the heat conduction term, which then affect all variables. In the next section, we identify the origin of temperature errors in gas/liquid flows and discuss how to eliminate them. 

## **4. Analysis of the temperature errors** 

## _4.1. Occurrence of temperature errors_ 

To illustrate the occurrence of temperature errors, we consider the 1D advection of an isolated material interface between a gas and a liquid at a constant velocity, pressure, and temperature, inspired by the analysis of Johnsen and Ham [16]. Initially, _p_ , _u_ and _T_ are constant, and _ρ_ and the material properties entering the equation of state are discontinuous. The exact solution for this problem is that this discontinuous front propagates at speed _u_ ; from the computational viewpoint, although the initially discontinuous profiles diffuse numerically, _p_ , _u_ and _T_ are expected to remain constant (to round-off). 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

553 

We start with the compressible Navier–Stokes equations (1) and discretize them spatially using any of the three approaches in Section 3. Since these approaches are all designed to preserve velocity and pressure equilibrium across the interface with no heat transfer, the continuity and momentum equations demonstrate that velocity equilibrium is preserved. The energy equation simplifies to the following semi-discrete form, with the interface lying in cell _j_ : 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-03.png)


where _D[a] j_[and] _[D][d] j_[are][spatial][difference][operators][for][advection][and][diffusion][that][are][assumed][to][have][the][following] properties [16]: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-05.png)


for _c_ constant, _A_ and _B_ variable; such properties are not trivial but can be enforced, even with high-order methods [10,13]. Substituting the stiffened equation of state (3) into Eq. (21) yields for the pressure (assuming _ρq_ is treated appropriately, as described in the next section): 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-07.png)


Pressure equilibrium (i.e., the pressure at the next time step remains the same across the entire domain) is maintained if (i) the difference operators obey Eq. (22), (ii) the material properties entering the equation of state are evolved in a consistent fashion and (iii) the pressure is computed from the appropriate quantities [3,4,10]. Similarly, for the temperature: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-09.png)


Temperature equilibrium (i.e., the temperature at the next time step remains the same across the entire domain) is maintained if (i) the difference operators obey Eq. (22), (ii) the material properties entering the equation of state are evolved in a consistent fashion and (iii) the temperature is computed from the appropriate quantities. Following the results in the previous section, the difference operator for diffusion must be constructed such that _D j(c)_ = 0 for _c_ constant (e.g., see [16]). There are thus two main sources for temperature errors: spatial discretization (i) and numerical model (ii and iii). A failure to maintain temperature equilibrium produces energy (and thus pressure) errors via the heat diffusion term in Eq. (23), which then affect the continuity and momentum equations. This issue is relevant only to problems in which heat diffusion is present. 

## _4.2. Eliminating temperature errors for the different approaches_ 

The goal is to determine the form of the transport equations to be solved and the appropriate relationships between quantities of interest to maintain temperature equilibrium for this isolated interface advection problem based on the approaches described in Section 3. 

## _4.2.1. γ -based approach_ 

The pressure–internal energy relation (23) can be re-written: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-15.png)


Similarly, the temperature–internal energy relation (24) can be re-written: 

Eqs. (25) and (26) hold for any material, pressure and temperature, therefore each term in brackets must be zero for the pressure and temperature equilibria to be maintained, in which case _p_ and _T_ can be factored out of the time derivative in Eqs. (25) and (26). Thus, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0006-18.png)


Eqs. (27) indicate that, to maintain pressure equilibrium in time and space with the _γ_ -based approach, the transport equations for 1 _/(n_ − 1 _)_ and _nB/(n_ − 1 _)_ must be solved in non-conservative form and that for _q_ in conservative form; pressure must be computed from these specific quantities using Eq. (3a). To maintain temperature equilibrium, the transport equations for _B_ must be solved in non-conservative form and those for _c_ and _q_ in conservative form; temperature must be computed from these specific quantities in Eq. (3b). 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

554 

## _4.2.2. Volume fraction approach_ 

Considering the mixture relations for volume fraction (11), the pressure–internal energy relation (23) can be re-written: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0007-04.png)


Likewise, the temperature–internal energy relation (24) can be expressed as: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0007-06.png)


Since pressure and temperature must remain constant in time and space, arguments similar to those made in the previous section hold, so that _p_ and _T_ can be factored out of Eqs. (28) and (29). Thus, 

Since only mixture density, species density and volume fraction vary in time and space, Eqs. (30) are discretizations of the following two transport equations for volume fraction: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0007-09.png)


Eqs. (31) indicate that, to maintain pressure equilibrium in time and space for the volume fraction approach, the calculation of 1 _/(n_ − 1 _)_ and _nB/(n_ − 1 _)_ in Eq. (10) must be done using volume fraction computed from the non-conservative form of the transport equation for _α[(][k][)]_ , and that of _ρq_ in Eq. (10) using the conservative form of the transport equation for _α[(][k][)]_ . To maintain temperature equilibrium in time and space, the calculation of _B_ in Eq. (10) must be done using volume fraction computed from the non-conservative form of the transport equation for _α[(][k][)]_ , and that of _ρq_ and _ρc_ in Eq. (10) using the conservative form of the transport equation for _α[(][k][)]_ . 

We note that Eqs. (31) hold for the present interface advection problem only, in which the velocity is constant for all time and space. Although Eq. (31a) is exact, the non-conservative form of the transport equation for volume fraction is Eq. (19b) [34]; volume fraction is not simply advected, but is additionally modified by a dilatation-dependent source term that represents the mixture compressibility. This equation can be solved but may lead to numerical difficulties, e.g., positivity of the volume fraction or maintaining correct shock jump conditions [21,35]. We note that, in the limit of sharp interface, numerical mixture regions vanish and the right-hand side of Eq. (19b) goes to zero. Past studies have indeed followed such an approach (five-equation model, [13,19,21,33,36]), which we do as well. 

## _4.2.3. Mass fraction approach_ 

Using the mixture relations for mass fraction (15), the pressure–internal energy equation (23) can be re-written: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0007-14.png)


Similarly, the temperature–internal energy relation (24) can be re-written: 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

555 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0008-02.png)


Following the same arguments as in the previous sections, Eqs. (32) and (33) can be re-organized: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0008-04.png)


Since only mixture density, species density, and species mass fraction vary in time and space, Eqs. (34) are discretizations of the following two transport equations for mass fraction: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0008-06.png)


From Eq. (6), Eqs. (35) are mathematically equivalent to those for volume fraction, such that the same conditions as those listed in Section 4.2.2 hold to maintain pressure and temperature equilibria. 

## _4.3. Summary of the analysis_ 

The analysis in the previous section indicates that all three approaches ( _γ_ , volume fraction and mass fraction) can be designed to prevent pressure and temperature errors. For two fluids, the _γ_ -based approach is computationally more expensive because one transport equation must be solved for each property in the equation of state (five here); for more than two fluids it may become more attractive, but if fluids have the same properties additional transport equations must be solved to distinguish the different fluids. 

The volume fraction and mass fraction approaches are mathematically equivalent. These approaches are not tailored to a given equation of state; however, analysis is required to determine how to calculate the material properties entering the equation of state. For each additional fluid, two additional transport equations must be computed. The reduced five-equation model (with _�kk_ ′ = 0) is strictly applicable only to flows of immiscible fluids (no physical mixture regions); for miscible flows, _�kk_ ′ ̸= 0 and the source term in the transport equation must be computed, which may lead to difficulties with shock jump conditions and positivity of volume fraction. These approaches can be extended to more general equations of state, such as Mie–Grüneisen (e.g., see [6]). 

## **5. Numerical implementation** 

For the simulations presented in this work, time marching is handled with a third-order accurate explicit strong stability preserving Runge–Kutta scheme [37]. For the spatial discretization, a solution-adaptive high-order accurate central difference/discontinuity-capturing method is proposed. This method can represent both broadband flow motions and discontinuities accurately and efficiently. The basic idea is that non-dissipative methods are used where the solution is smooth, while the more dissipative and computationally expensive capturing schemes are applied near discontinuous regions. For this purpose, a discontinuity sensor discriminates between smooth and discontinuous (shocks, contacts and interfaces) regions, which all require a different treatment; smooth regions are computed using central differences, a finite difference weighted essentially non-oscillatory (WENO [38]) scheme with Lax–Friedrichs flux splitting handles shock waves, and the approach of Johnsen and Colonius [10] is used for material interfaces. 

To illustrate the specifics, we consider the semi-discrete form of the 1D Euler equations for simplicity, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0008-14.png)


where _u_ is the vector of conserved variables and _F_ is the numerical flux, which can be written 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0008-16.png)


where _bi_ represent the value of the sensor in different regions. The sensor values are _b_ 1 = 1 and _b_ 2 _, b_ 3 = 0 for smooth regions, _b_ 2 = 1 and _b_ 1 _, b_ 3 = 0 for shocks and _b_ 3 = 1 and _b_ 1 _, b_ 2 = 0 for interfaces. The capability of the sensor to distinguish 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

556 

between discontinuous and smooth regions highly affects the overall accuracy and performance [39]. We adapt the sensor of Henry de Frahan et al. [6] to finite differences. At each cell edge, _L_ and _R_ denote the value of the corresponding variable at the left and right of the computational cell respectively. Accordingly, shocks are detected using the function below: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-03.png)


If _�_ is greater than 0 _._ 01, the corresponding cell is flagged to be treated by shock capturing. Contact discontinuities, also treated by shock capturing, are detected as follows: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-05.png)


Finally, material interfaces, to be handled with the method of Johnsen and Colonius [10] are detected as follows: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-07.png)


The thresholds for _�_ or _Z_ are set to be 0 _._ 01. Although the optimal threshold values for both shocks and interfaces may be problem-dependent, our numerical experiments show a robust detection of discontinuities for the chosen values, which are slightly different from those in Henry de Frahan et al. [6] because of the different spatial discretization. This approach is easily applicable to multiple dimensions. 

In smooth regions, fourth-order central differences are used for the convective terms, both for the conservative and non-conservative equations as proposed by Movahed and Johnsen [5]: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-10.png)


Second derivatives (e.g., for diffusion) are also treated with fourth-order differences; for an arbitrary variable _A_ , 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-12.png)


Shock waves are captured using the fifth-order accurate finite difference WENO of Jiang and Shu [38]. For this purpose, we use Lax–Friedrichs flux splitting, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-14.png)


where _λ_ = max _u_ | _F_[′] _(u)_ | over the relevant range of _u_ . Since these regions are not flagged as material discontinuities, the central scheme can be used to solve the advection equation for the (constant) material properties. 

For material interfaces, the WENO procedure proposed by Johnsen and Colonius [10] is used, which corresponds to a second-order finite difference approximation; however, this reduction in order is not problematic since this approach is used at material discontinuities only, where the solution reduces to first order anyways. The HLL Riemann solver [40] is used for upwinding. To correctly treat the non-conservative transport equations, we extend the expression in Saurel and Abgrall [41] to high-order WENO: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-17.png)


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-18.png)


where _u_ and _a_ stand for velocity and sound speed, respectively. The discretized form of the non-conservative transport equation for an arbitrary variable _A_ is 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0009-20.png)


_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

557 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0010-02.png)


**Fig. 1.** _L_ ∞ error for the 1D smooth advection problem. Red squares: _n_ ; blue diamonds: pressure; green circles: temperature. 

## **6. Results** 

The compressible Navier–Stokes equations, non-dimensionalized by the density and sound speed of air at atmospheric pressure, characteristic length _L_ = 0 _._ 2 mm, and _T_ = 300 K, are solved for all problems. The time step is adaptively set to satisfy the advection and diffusion constraints, with CFL number 0.95 and VNN 0.475. We consider the _γ_ , volume fraction (or “ _α_ ”) and mass fraction (or “ _z_ ”) models described in Section 4.2; when referring to our proposed approach, we mean an approach that preserves velocity, pressure and temperature equilibria (for an isolated interface). We make comparisons to current schemes in the literature designed to maintain velocity and pressure equilibria in the absence of heat conduction, which we call “pressure only” (e.g., the methods in [3,4]); with these approaches, temperature is computed from the available data; e.g., for the _γ_ -model, _B_ would be computed from _nB/(n_ − 1 _)_ and 1 _/(n_ − 1 _)_ , rather than being advected as we propose. The _α_ and _z_ “pressure only” approaches are identical, so only the _z_ approach is considered. Water and air have values taken from Table 1. 

## _6.1. 1D smooth advection problem_ 

We consider the advection of a smooth distribution in density and _n_ to show that our solution-adaptive method achieves the correct convergence rate for smooth problems. The following initial conditions are used 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0010-08.png)


This distribution moves at constant speed _u_ , with constant pressure and temperature in the periodic domain _x_ ∈[0 _,_ 1]. The _L_ ∞ errors in _n_ , pressure and temperature are shown in Fig. 1 after one period and for different resolutions. Pressures and temperatures remain near round-off, thus demonstrating that pressure and temperature equilibria are maintained. The convergence rate (in _n_ ) is fourth, the order of the finite difference scheme. 

## _6.2. 1D air/water interface advection_ 

We numerically verify our theoretical development for the isolated interface advection problem in Section 4.1, e.g., for a 1D air bubble in water. We consider an initially sharp top-hat distribution of air in water at the same temperature and pressure, moving at a constant speed _u_ in the periodic domain _x_ ∈[0 _,_ 1]. The initial conditions are, 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0010-12.png)


The properties entering the equation of state are initialized using the same top-hat distribution. The results for all three models with the “pressure only” and our proposed approaches using 200 points are shown in Fig. 2 (density, pressure and temperature profiles after one period) and Fig. 3 (time evolution of the _L_ ∞ error in _u_ , _p_ , _T_ ). The results clearly indicate that our proposed approaches do not introduce errors in pressure, temperature or velocity. On the other hand, if using approaches that are designed to only preserve pressure and velocity equilibria, and using only the available data to compute temperature, then errors are produced in the temperature, which then propagate to the other fields due to the heat diffusion and pressure terms. The resulting errors are non-negligible, particularly for the temperature. If the Fourier heat conduction term had not been included no such errors would occur. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

558 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0011-02.png)


**Fig. 2.** Profile of the advection of an air/water interface after one period. Black solid line: initial and exact solution; green circles: “pressure only” _γ_ approach; blue squares: “pressure only” _z_ approach; orange diamonds: proposed _γ_ approach; red triangles: proposed _α_ approach; pink pluses: proposed _z_ approach. 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0011-04.png)


**Fig. 3.** Time histories of normalized _L_ ∞ errors for the advection of an air/water interface after one period. Black solid line: initial and exact solution; green circles: “pressure only” _γ_ approach; blue squares: “pressure only” _z_ approach; orange diamonds: proposed _γ_ approach; red triangles: proposed _α_ approach; pink pluses: proposed _z_ approach. 

## _6.3. 1D gas–liquid Riemann problem_ 

We consider gas–liquid Riemann problems to compare the “pressure only” and our proposed _α_ approaches for shockdominated interfacial flows; similar results are obtained with the other models. The initial conditions (with water on the left, air on the right) are [19,23]: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0011-08.png)


The domain is discretized with 1000 cells and the exact solution is the converged solution on 5000 cells. The profiles of density, pressure, temperature, velocity, and volume fraction for both approaches are shown in Fig. 4. Our proposed approach shows good agreement with the exact solution. For the “pressure only” approach at this high pressure ratio (10,000:1) and heavy gas, temperature errors are clear at the interface, but pressure errors are not substantial. 

Another gas–liquid Riemann problem is considered with initial conditions more relevant to our interests (1D shock– bubble interaction): 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0011-11.png)


The domain is discretized with 1000 cells and the exact solution is the converged solution on 5000 cells. The profiles of density, pressure, temperature, velocity, and volume fraction for both approaches are shown in Fig. 5. The numerical solution agrees well with the exact solution. In this problem, the effect of heat diffusion is significant inside the bubble. On the other hand, the “pressure only” approach gives rise to a large temperature error at the interface, as well as erroneous density, velocity, pressure and temperature between the interface and shock. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

559 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0012-02.png)


**Fig. 4.** Gas–liquid Riemann problem [19,23]. Black solid line: exact solution; blue filled circles: proposed approach; red filled diamonds: “pressure only” approach. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

560 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0013-02.png)


**Fig. 5.** Gas–liquid Riemann problem (1D shock-interface). Black solid line: exact solution; blue filled circles: proposed approach; red filled diamonds: “pressure only” approach. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

561 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0014-02.png)


**Fig. 6.** Shock-induced bubble collapse (case 1, _ps/po_ = 100) at different times _t_ = 0 _._ 04 _,_ 1 _._ 49 _,_ 2 _._ 05. Top row: “pressure only” approach; bottom row: proposed approach; top contour: pressure gradient magnitude; bottom contour: temperature. 

## _6.4. 3D shock–bubble interaction_ 

To determine the implications for relevant bubble dynamics problems, we consider the interaction of a shock wave in water with an air bubble near a rigid wall, as in Johnsen et al. [12]. Initially, the spherical bubble is in equilibrium with its surroundings: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0014-06.png)


Two different shock strengths are considered: 

- Case 1: weak shock wave with pressure ratio of 100 (Mach 1.0035) 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0014-09.png)


- Case 2: strong shock wave with pressure ratio of 1000 (Mach 1.035) 

_(ρ, u, v, w, p, T )_ = _(_ 881 _._ 6 _,_ 0 _._ 184 _,_ 0 _,_ 0 _,_ 714 _,_ 1 _._ 06 _)_ in the water downstream of the shock. (53) 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

562 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0015-02.png)


**Fig. 7.** Shock-induced bubble collapse (case 2, _ps/po_ = 1000) at different times _t_ = 0 _._ 03 _,_ 0 _._ 6 _,_ 0 _._ 92. Top row: “pressure only” approach; bottom row: proposed approach; top contour: pressure gradient magnitude; bottom contour: temperature. 

This problem is simulated using the “pressure only” and our proposed _α_ approaches on a 500 × 400 × 400 uniform grid for both cases. The initial stand-off distance of the bubble from the wall is 1.1. By symmetry, only a quarter of the bubble is computed, with symmetry boundary conditions along the relevant planes. The wall is perfectly reflecting, with no slip. Zero gradient conditions are used along the remaining boundaries. 

The results are compared to evaluate the effects of temperature errors; quantities related to pressures and temperatures along the wall, as well as bubble dynamics are of particular interest. Figs. 6 and 7 show the pressure gradient magnitude and temperature contours at different times during the process. The right-moving shock interacts with the bubble, thus producing a reflected rarefaction wave. As the bubble starts its collapse, the incoming shock hits the rigid wall and reflects back onto the bubble. During the collapse, a re-entrant jet directed toward the wall is produced, which, upon impact with the distal side, generates an outward propagating shock. Even though the “pressure only” approach is designed to prevent pressure errors, such errors are generated because of the large temperature spike across the bubble interface. The most striking discrepancies lie in the temperature contours, particularly in the region just outside the bubble after collapse. The pressure gradient magnitude shows how these temperature errors propagate in the pressure field. These temperature errors strongly affect the simulations and may ultimately cause the code to fail. 

To quantitatively evaluate these errors, Figs. 8 and 9 show the pressure and temperature along the centerline at different times, and Figs. 10 and 11 plot time histories of the wall pressure and temperature at point _A_ (along the centerline and on the wall), and total enstrophy for both cases. The temperature errors initially consist of spikes along the interface of 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

563 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0016-02.png)


**Fig. 8.** Shock-induced bubble collapse (case 1, _ps/po_ = 100) – Centerline properties (top row: pressure; bottom row: temperature) at times 0 _._ 04 _,_ 1 _._ 49 _,_ 2 _._ 05. Red diamonds: “pressure only” approach; blue circles: proposed approach. 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0016-04.png)


**Fig. 9.** Shock-induced bubble collapse (case 2, _ps/po_ = 1000) – Centerline properties (top row: pressure; bottom row: temperature) at times 0 _._ 03 _,_ 0 _._ 6 _,_ 0 _._ 73. Red diamonds: “pressure only” approach; blue circles: proposed approach. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

564 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0017-02.png)


**Fig. 10.** Shock-induced bubble collapse (case 1, _ps/po_ = 100) – Time histories of the pressure (left) and temperature (center) at point A and total enstrophy (right). Red diamonds: “pressure only” approach; blue circles: proposed approach. 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0017-04.png)


**Fig. 11.** Shock-induced bubble collapse (case 2, _ps/po_ = 1000) – Time histories of the pressure (left) and temperature (center) at point A and total enstrophy (right). Red diamonds: “pressure only” approach; blue circles: proposed approach. 

over 100% error. Much of the temperature discrepancies appear to be localized at the interface, though some regions in which the interface was previously located appear to still bear memory of these errors. Temperature errors are particularly important for case 1 ( _ps/po_ = 100). The pressure oscillations emanating from these temperature errors exhibit the largest discrepancies after collapse in the region between the bubble and the wall, with local errors nearly 100%. Along the wall, the discrepancies in pressure are on the order of 10%, while the temperature errors are more than 100%, always overshoots: for case 1, the maximum wall pressure and temperature in the simulation for the proposed approach are 2356 and 1.2, while the “pressure only” approaches yield 2550 and 2.2; for case 2, the maximum wall pressure and temperature in the simulation for the proposed approach are 14,450 and 4.1, while the “pressure only” approaches yield 15,300 and 10.2. These discrepancies even affect the vorticity contents of the flow and possibly generation of small-scale features, due to baroclinic vorticity generated along the interface, due to pressure oscillations. 

## **7. Conclusions** 

In this work, we identify and propose a strategy to prevent numerical errors produced by interface-capturing schemes in simulations of gas/liquid interfaces with heat diffusion. We consider here the compressible Navier–Stokes equations and materials (gases, liquids and solids) that can be described by a stiffened equation of state. For the common _γ_ , volume fraction and mass fraction models, we show that an incorrect calculation of the temperature (because of the discretization and/or numerical model) gives rise to errors in energy due to the heat conduction term. Such errors in temperature are not relevant in Euler simulations but may become problematic in Navier–Stokes calculations. The errors further modify the pressure and produce spurious oscillations at interfaces, even for approaches designed to prevent such errors _when no heat diffusion is considered_ . These errors can be prevented by computing the material properties in the equation of state based on appropriate transport equations in conservative and non-conservative forms, depending on the multiphase model; the appropriate properties must then be used to calculate the pressure and temperature. We further reconcile the _γ_ , volume fraction and mass fraction approaches, and demonstrate that the latter two are equivalent. We note that the proposed numerical model (“reduced” five-equation model) is applicable to sharp interfaces; for miscible or homogeneous multiphase problems, care must be taken to include the required dilatational source term in the volume fraction equation in advection form. To verify the analysis and compute gas/liquid problems of interest, we develop a three-dimensional, high-order accurate, solution-adaptive finite difference framework, which we tested using several 1D problems. Finally, we use it to 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

565 

compute the 3D shock-induced collapse of a gas bubble, to illustrate that the numerical errors described herein can significantly overpredict temperature and pressure (sometimes with over 100% error), and affect vortical structures, in situations of engineering relevance. This overall computational framework will serve as the basis for investigations of bubble dynamics. 

## **Acknowledgements** 

This research was supported by ONR grant N00014-12-1-0751 under Dr. Ki-Han Kim. This work used resources from the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number OCI-1053575. 

## **Appendix A** 

To derive the transport equations for the volume fraction formulation, we follow Miller and Puckett [34] and start with Eq. (16), which can be expanded 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-07.png)


We define the isentropic bulk modulus for each fluid, _K s[(][k][)]_ , assuming isotropic stresses during advection and isentropic processes in any compression of the individual components. These assumptions imply that the pressure change associated with compression of the bulk ( _∂ p_ ) is equal to the pressure change associated with compression of each components ( _∂ p[(][k][)]_ ). Then, _K s[(][k][)]_ can be defined as: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-09.png)


Differentiating Eq. (6) with respect to pressure and assuming constant entropies for each fluid results in: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-11.png)


By combining Eqs. (A.2) and (6), the isentropic bulk modulus and sound speed of the mixture are: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-13.png)


thus recovering Wallis’ relation [42]. According to the definition of isentropic bulk modulus and assuming pressure equilibrium between the phases, we can write _K s∂ρ/ρ_ = _∂ p_ = _∂ p[(][k][)]_ = _K s[(][k][)][∂][ρ][(][k][)][/∂][ρ][(][k][)]_[.][Using][(16)][yields:] 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-15.png)


Finally, substituting Eqs. (A.2) and (A.4) into Eq. (A.5) yields, after appropriate manipulations: 


![](images/2015_AlahyariBeig_Johnsen_interface_equilibrium_compressible_multiphase.pdf-0018-17.png)


## **References** 

> [1] J. Glimm, J.W. Grove, X.L. Li, K.M. Shyue, Y. Zeng, Q. Zhang, Three-dimensional front tracking, SIAM J. Sci. Comput. 19 (1998) 703–727, http://dx.doi.org/10.1137/S1064827595293600. 

> [2] S. Osher, J.A. Sethian, Fronts propagating with curvature dependent speed: algorithms based on Hamilton–Jacobi formulations, J. Comput. Phys. 49 (1987) 12–49, http://dx.doi.org/10.1016/0021-9991(88)90002-2. 

> [3] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160, http://dx.doi.org/10.1006/jcph.1996.0085. [4] K.M. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, J. Comput. Phys. 142 (1998) 208–242, http://dx.doi.org/ 10.1006/jcph.1998.5930. 

> [5] P. Movahed, E. Johnsen, A solution-adaptive method for efficient compressible multifluid simulations, with application to the Richtmyer–Meshkov instability, J. Comput. Phys. 239 (2013) 166–186, http://dx.doi.org/10.1016/j.jcp.2013.01.016. 

> [6] M.T. Henry de Frahan, S. Varadan, E. Johnsen, A new limiting procedure for discontinuous Galerkin methods applied to compressible multiphase flows with shocks and interfaces, J. Comput. Phys. 280 (2015) 489–509, http://dx.doi.org/10.1016/j.jcp.2014.09.030. 

_S. Alahyari Beig, E. Johnsen / Journal of Computational Physics 302 (2015) 548–566_ 

566 

- [7] S. Kokh, F. Lagoutière, An anti-diffusive numerical scheme for the simulation of interfaces between compressible fluids by means of a five-equation model, J. Comput. Phys. 229 (2010) 2773–2809, http://dx.doi.org/10.1016/j.jcp.2009.12.003. 

- [8] K.M. Shyue, F. Xiao, An Eulerian interface sharpening algorithm for compressible two-phase flow: the algebraic THINC approach, J. Comput. Phys. 268 (2014) 326–354, http://dx.doi.org/10.1016/j.jcp.2014.03.010. 

- [9] A. Tiwari, J.B. Freund, C. Pantano, A diffuse interface model with immiscibility preservation, J. Comput. Phys. 252 (2013) 290–309, http://dx.doi.org/ 10.1016/j.jcp.2013.06.021. 

- [10] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2006) 715–732, http://dx.doi.org/10.1016/j.jcp.2006.04.018. 

- [11] C.W. Shu, Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws, Lecture Notes in Mathematics, vol. 1697, Springer, Heidelberg, 1998. 

- [12] E. Johnsen, T. Colonius, Numerical simulations of non-spherical bubble collapse, J. Fluid Mech. 629 (2009) 231–261, http://dx.doi.org/10.1017/ S0022112009006351. 

- [13] V. Coralic, T. Colonius, Finite-volume WENO scheme for viscous compressible multicomponent flows, J. Comput. Phys. 274 (2014) 95–121, http://dx.doi.org/10.1016/j.jcp.2014.06.003. 

- [14] S. Kawai, H. Terashima, A high-resolution scheme for compressible multicomponent flows with shock waves, Int. J. Numer. Methods Fluids 66 (10) (2011) 1207–1225, http://dx.doi.org/10.1002/fld.2306. 

- [15] H. Terashima, S. Kawai, M. Koshi, Consistent numerical diffusion terms for simulating compressible multicomponent flows, Comput. Fluids 88 (2013) 484–495, http://dx.doi.org/10.1016/j.compfluid.2013.10.007. 

- [16] E. Johnsen, F. Ham, Preventing numerical errors generated by interface-capturing schemes in compressible multi-material flows, J. Comput. Phys. 231 (2012) 5705–5717, http://dx.doi.org/10.1016/j.jcp.2012.04.048. 

- [17] M. Baer, J. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials, Int. J. Multiph. Flow 12 (1986) 861–889, http://dx.doi.org/10.1016/0301-9322(86)90033-9. 

- [18] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deflagration-to-detonation transition in granular materials: reduced equations, Phys. Fluids 13 (2001) 3002–3024, http://dx.doi.org/10.1063/1.1398042. 

- [19] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2002) 577–616, http://dx.doi.org/10.1006/jcph.2002.7143. 

- [20] R. Saurel, R. Abgrall, A multiphase Godunov method for compressible multifluid and multiphase flows, J. Comput. Phys. 150 (1999) 425–467, http://dx.doi.org/10.1006/jcph.1999.6187. 

- [21] G. Perigaud, R. Saurel, A compressible flow model with capillary effects, J. Comput. Phys. 209 (2005) 139–178, http://dx.doi.org/10.1016/j.jcp.2005. 03.018. 

- [22] J.J. Kreeft, B. Koren, A new formulation of Kapila’s five-equation model for compressible two-fluid flow, and its numerical treatment, J. Comput. Phys. 229 (2010) 6220–6242, http://dx.doi.org/10.1016/j.jcp.2010.04.025. 

- [23] A. Murrone, H. Guillard, A five equation reduced model for compressible two phase flow problems, J. Comput. Phys. 202 (2005) 664–698, http:// dx.doi.org/10.1016/j.jcp.2004.07.019. 

- [24] F. Petitpas, J. Massoni, R. Saurel, E. Lapebie, L. Munier, Diffuse interface model for high speed cavitating underwater systems, Int. J. Multiph. Flow 35 (2009) 747–759, http://dx.doi.org/10.1016/j.ijmultiphaseflow.2009.03.011. 

- [25] T. Flatten, A. Morin, S.T. Munkejord, Wave propagation in multicomponent flow models, SIAM J. Appl. Math. 70 (2010) 2861–2882, http://dx.doi.org/ 10.1137/090777700. 

- [26] B. Braconnier, B. Nkonga, An all-speed relaxation scheme for interface flows with surface tension, J. Comput. Phys. 228 (2009) 5722–5739, http://dx.doi.org/10.1016/j.jcp.2009.04.046. 

- [27] G.H. Schnerr, I.H. Sezal, S.J. Schmidt, Numerical investigation of three-dimensional cloud cavitation with special emphasis on collapse induced shock dynamics, Phys. Fluids 20 (2008), http://dx.doi.org/10.1063/1.2911039. 

- [28] K.H. Kim, G. Chahine, J.P. Franc, A. Karimi, Advanced Experimental and Numerical Techniques for Cavitation Erosion Prediction, Fluid Mechanics and Its Applications, vol. 106, Springer, Netherlands, 2014. 

- [29] O. Le Métayer, J. Massoni, R. Saurel, Modelling evaporation fronts with reactive Riemann solvers, J. Comput. Phys. 205 (2005) 567–610, http://dx.doi.org/ 10.1016/j.jcp.2004.11.021. 

- [30] R. Saurel, F. Petitpas, R. Abgrall, Modelling phase transition in metastable liquids: application to cavitating and flashing flows, J. Fluid Mech. 607 (2008) 313–350, http://dx.doi.org/10.1017/S0022112008002061. 

- [31] E. Goncalvès, R.F. Patella, Numerical study of cavitating flows with thermodynamic effect, Comput. Fluids 39 (2010) 99–113, http://dx.doi.org/10.1016/ j.compfluid.2009.07.009. 

- [32] K.M. Shyue, A fluid-mixture type algorithm for compressible multicomponent flow with van der Waals equation of state, J. Comput. Phys. 156 (1999) 43–88, http://dx.doi.org/10.1006/jcph.2001.6801. 

- [33] R.K. Shukla, C. Pantano, J.B. Freund, An interface capturing method for the simulation of multi-phase compressible flows, J. Comput. Phys. 229 (2010) 7411–7439, http://dx.doi.org/10.1016/j.jcp.2010.06.025. 

- [34] G.H. Miller, E.G. Puckett, A high-order Godunov method for multiple condensed phases, J. Comput. Phys. 128 (1996) 134–164, http://dx.doi.org/10.1006/ jcph.1996.0200. 

- [35] R. Abgrall, V. Perrier, Asymptotic expansion of a multiscale numerical scheme for compressible multiphase flow, Multiscale Model. Simul. 5 (2006) 84–115, http://dx.doi.org/10.1137/050623851. 

- [36] R.K. Shukla, Nonlinear preconditioning for efficient and accurate interface capturing in simulation of multicomponent compressible flows, J. Comput. Phys. 276 (2014) 508–540, http://dx.doi.org/10.1016/j.jcp.2014.07.034. 

- [37] S. Gottlieb, C.W. Shu, Total variation diminishing Runge–Kutta schemes, Math. Comput. 67 (1996) 73–85, http://dx.doi.org/10.1090/S0025-5718-9800913-2. 

- [38] G.S. Jiang, C.W. Shu, Efficient implementation of weighted ENO schemes, J. Comput. Phys. 228 (1996) 202–228, http://dx.doi.org/10.1006/jcph.1996.0130. 

- [39] E. Johnsen, J. Larsson, A.V. Bhagatwala, W.H. Cabot, P. Moin, B.J. Olson, P.S. Rawat, S.K. Shankar, B. Sjögreen, H.C. Yee, X. Zhong, S.K. Lele, Assessment of high-resolution methods for numerical simulations of compressible turbulence with shock waves, J. Comput. Phys. 229 (2010) 1213–1237, http://dx.doi.org/10.1016/j.jcp.2009.10.028. 

- [40] A. Harten, P.D. Lax, B. Van Leer, On upstream differencing and Godunov-type schemes for hyperbolic conservation laws, SIAM Rev. 25 (1982) 35–61, http://dx.doi.org/10.1137/1025002. 

- [41] R. Saurel, R. Abgrall, A simple method for compressible multifluid flows, SIAM J. Sci. Comput. 21 (1999) 1115–1145, http://dx.doi.org/10.1137/ S1064827597323749. 

- [42] G.B. Wallis, One-Dimensional Two-Phase Flow, McGraw–Hill, New York, 1969. 

