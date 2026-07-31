Journal of Computational Physics 524 (2025) 113701 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0001-01.png)


Contents lists available at ScienceDirect Journal of Computational Physics journal homepage: www.elsevier.com/locate/jcp 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0001-03.png)


## Approximately pressure-equilibrium-preserving scheme for fully conservative simulations of compressible multi-species and real-fluid interfacial flows 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0001-05.png)


## H. Terashima[a] _[,]_[b] _[,]_[∗] , N. Ly[b] , M. Ihme[b] 

> a _Division of Mechanical and Aerospace Engineering, Hokkaido University, N13 W8, Kita-ku, Sapporo, 060-8628, Hokkaido, Japan_ 

> b _Department of Mechanical Engineering, Stanford University, Stanford, 94305, CA, USA_ 

|A R T I C L E<br>I N F O<br>_Keywords:_<br>Compressible multi-species fows<br>Real-fuid equations of state<br>Energy conservation<br>Pressure equilibrium|A B S T R A C T|
|---|---|
||This study proposes a numerical method for fuid interfaces in compressible multi-species<br>and real-fuid fow simulations. The proposed method preserves the full conservation (species-<br>mass, momentum, and energy) property of compressible fow equations while approximately<br>maintaining the pressure equilibrium condition at fuid interfaces. The numerical fuxes of<br>internal energy and species-mass are newly constructed to satisfy the pressure equilibrium<br>condition approximately. The modifed equation for the pressure equilibrium condition shows<br>that the proposed numerical fuxes introduce diferent coefcients in the second-order error<br>term, compared to standard numerical fuxes, thereby reducing the pressure equilibrium error.<br>The conservation and pressure equilibrium properties of the proposed method are validated<br>through one-dimensional and two-dimensional smooth interface advection problems using the<br>compressible multi-species Euler equations with the Soave-Redlich-Kwong equation of state.|


## **1. Introduction** 

It is well known that spurious pressure oscillations occur at the inviscid interfaces in the simulation of compressible multi-species flows [1] and severe pressure oscillations may induce computational instabilities by disturbing the flow field. Such spurious pressure oscillations can arise even at interfaces in single-species flow simulations, especially when real-fluid equations of state are employed where large variations of thermodynamic properties, such as density and speed of sound, exist across the interfaces [2]. 

Effective methods to prevent spurious pressure oscillations at the inviscid interfaces have been proposed. For the calorically perfect gas, Abgrall [1] identified the generation of spurious pressure oscillation at the interfaces and introduced a non-conservative form of the transport equation of the specific heat ratio to maintain the pressure equilibrium across the interfaces (called a quasi-conservative form). Note that the pressure equilibrium denotes the property of maintaining constant pressure when both pressure and velocity are initially uniform in space. Shyue [3] extended Abgrall’s idea to more complicated equations of state such as the Mie-Grüneisen equation of state, solving additional transport equations of thermodynamic properties in the equation of state. Extensions to higherorder schemes such as upwind-based schemes [4,5] or central differencing schemes with consistent numerical diffusion [6] were proposed. Karni [7] introduced a non-conservative model using primitive variables to solve spurious oscillations and later proposed a hybrid algorithm using the equation of state and pressure evolution equation in the pressure update procedure to minimize the 

- Corresponding author. 

_E-mail address:_ htera@eng.hokudai.ac.jp (H. Terashima). 

https://doi.org/10.1016/j.jcp.2024.113701 

Received 21 February 2024; Received in revised form 14 November 2024; Accepted 21 December 2024 

0021-9991/© 2024 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies. 

Available online 27 December 2024 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

conservation error [8]. Another well-known approach concerning the interface equilibrium in compressible flows is a five-equation model developed by Allaire et al. [9]. 

Spurious pressure oscillations may arise at the fluid interfaces even in single-species flow simulations when complicated equations of state are applied [2] since the pressure updated from equations of state cannot guarantee the pressure equilibrium condition at the interfaces. A typical example is the fluid interface at supercritical pressures: it is known that, in a simulation where a cryogenic N2 is injected into a warm N2 environment, spurious pressure oscillations are generated at the N2-N2 interface, causing serious numerical instabilities. These spurious oscillations worsen with larger variations of thermodynamic variables such as density or temperature across the interface. To reduce the numerical instabilities associated with spurious pressure oscillations, Schmitt et al. [10] used a quasi-conservative formulation of the governing equations with a consistent modification of the numerical dissipation terms. Terashima and Koshi [2] introduced a pressure evolution equation and consistent numerical diffusion to maintain the pressure and velocity equilibria at the fluid interfaces. In their studies, a cryogenic N2 jet into ambient environments (large density and temperature variations at the interface) was successfully simulated with a high-order compact scheme [2,11]. Ma et al. [12] developed a double-flux model for real-fluid simulations by extending the idea of Abgrall and Karni [13] for the calorically perfect gas equation of state. In their method, an entropy-stable flux was introduced to dampen the numerical oscillations at the interfaces associated with large density gradients. Lacaze et al. [14] proposed an enthalpy-based approach in comparison with fully-conservative energybased and pressure-based approaches for modeling supercritical flows. While those methods have provided sufficient robustness for real-fluid simulations under severe thermodynamic conditions, unfortunately, a concern remains: the lack of energy conservation. Numerical compatibility on preserving both the pressure equilibrium at interfaces and energy conservation has been a long-standing issue in compressible multi-species and real-fluid simulations. 

Recently, Fujiwara et al. [15] derived a pressure equilibrium condition from the governing equations of compressible multispecies flows and constructed the numerical fluxes (half-point values) to implicitly maintain the pressure equilibrium condition. The full conservation properties are preserved, and no additional transport equations for auxiliary quantities, such as the transport equation of the specific heat ratio [1], are required to maintain the pressure equilibrium. Their scheme is the first one in which the full conservation and pressure equilibrium properties are both preserved. However, while the derived pressure equilibrium condition was general to any equation of state, unfortunately, the constructed numerical fluxes were limited to the calorically perfect gas equation of state. Following the work of Fujiwara et al. [15], Bernades et al. [16] proposed a kinetic-energyand pressure-equilibrium-preserving scheme for real-fluid simulations. They abandoned the total energy conservation and solved the pressure evolution equation. The study stated that the extension of Fujiwara’s approach to real-fluid simulations is challenging due to the nonlinear relationship between pressure and internal energy. Thus, at this point, no numerical methods exist to preserve the full conservation and pressure equilibrium properties for compressible multi-species and real-fluid flow simulations. 

This study attempts to address this issue and develops a numerical method for fluid interfaces in compressible multi-species and real-fluid flow simulations, which preserves the full conservation of governing equations, particularly total energy conservation and pressure equilibrium at fluid interfaces. The compressible Euler equations for _𝑁_ -component multi-species flows are solved with the Soave-Redlich-Kwong (SRK) equation of state [17]. The proposed method is not limited to a specific equation of state, such as the calorically perfect gas equation of state. 

## **2. Numerical method** 

In this section, the one-dimensional equations are used to derive the proposed scheme. The extension to multi-dimension is straightforward. 

## _2.1. Governing equations_ 

The governing equations in this study are the compressible Euler equations. The one-dimensional compressible Euler equations for _𝑁_ -component multi-species flows are written as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0002-10.png)


where _𝜌_ is the density, _𝑌𝑖_ is the mass fraction of species _𝑖_ ( _𝑖_ = 1 ∶ _𝑁_ ), _𝑢_ the velocity, _𝑝_ the pressure, and _𝐸_ (= _𝑒_ +[1] 2 _[𝑢𝑢]_ ) the total energy per unit mass. _𝑒_ is the internal energy per unit mass. Note that[∑] _[𝑁] 𝑖_ =1 _[𝑌][𝑖]_[= 1][ and][thus][∑] _[𝑁] 𝑖_ =1 _[𝜌𝑌][𝑖]_[=] _[ 𝜌]_[are][satisfied.][We][may][use] _[𝜌][𝑖]_ instead of _𝜌𝑌𝑖_ for the density of species _𝑖_ in the discussion about the numerical method. 

In the derivation of a numerical scheme, we describe an equation of state in terms of the conservative variables as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0002-13.png)


and the flux may be denoted as 

2 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

(5) 

_𝐹𝜌𝑌𝑖_ = _𝜌𝑌𝑖𝑢, 𝐹𝜌𝑢_ = _𝜌𝑢𝑢_ + _𝑝, 𝐹𝜌𝐸_ = ( _𝜌𝐸_ + _𝑝_ ) _𝑢._ 

## _2.2. Spatial discretization_ 

In this study, the governing equations are spatially discretized by using the numerical flux [15,18] as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0003-06.png)


where _𝑚_ is a cell index and Δ _𝑥_ is the grid spacing. The numerical flux at a cell interface needs to be constructed to obtain the solutions. 

## _2.3. Pressure equilibrium_ 

From Eq. (4), the partial time derivative of the total energy can be written as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0003-10.png)


Then, an equation to satisfy the pressure equilibrium (PE) ( _𝜕𝑡𝑝_ = 0) is obtained as 

Further, using Eqs. (1) to (3), Eq. (10) may be rewritten using the spatial derivative of flux as follows: 

Equation (11) is the PE condition imposed on the flux of multi-species flows. 

## _2.4. Velocity equilibrium_ 

The velocity equilibrium (VE) condition ( _𝜕𝑡𝑢_ = 0) is obtained using Eqs. (1) and (2) as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0003-16.png)


## _2.5. Discrete form for PE and VE conditions_ 

If Eqs. (11) and (12) are satisfied at the discrete level, the PE and VE in flow fields can be maintained. For a cell index _𝑚_ , assuming a unit grid spacing (Δ _𝑥_ = 1), a discrete form for Eq. (11) can be written as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0003-19.png)


For a flow field with constant initial pressure _𝑝_ ( _𝑡_ = 0 _,𝑥_ ) = _𝑝_ 0 and velocity _𝑢_ ( _𝑡_ = 0 _,𝑥_ ) = _𝑢_ 0, the numerical flux at _𝑚_ ±[1] 2[reduces][to] 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0003-21.png)


and the partial derivative terms in Eq. (13) also reduce to 

3 

_H. Terashima, N. Ly and M. Ihme_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-01.png)


Then, substituting Eqs. (14) to (18) into Eq. (13) provides a discrete form for the PE condition, which corresponds to the PE compatibility condition derived in [15], as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-03.png)


Thus, the PE condition reduces to a condition on the half-point values _𝜌𝑌𝑖_[|] | _𝑚_ ± 12 and _𝜌𝑒_ | _𝑚_ ± 12 . However, as recognized in Eq. (19), the _𝜕𝜌𝑒_ asymmetry associated with the partial derivative term defined at _𝑚_ means that no locally-defined half-point values ( _𝜕𝜌𝑌𝑖_ ) _𝜌𝑌𝑗_ ≠ _𝑖,𝑝_ ||||| _𝑚_ 

satisfy Eq. (19) for all cells. 

Regarding the VE condition, substituting Eqs. (14) and (15) into Eq. (12) demonstrates that the VE condition is automatically satisfied at the discrete level. 

## _2.6. Approximating PE condition_ 

To satisfy the PE condition at the discrete level, the half-point values that satisfy Eq. (19) need to be determined. However, such half-point values are not apparent. Moreover, as stated in the previous section, the asymmetric form of (19) suggests that no locally-defined half-point values satisfy Eq. (19), and thereby no conservative schemes may be constructed. Therefore, we first derive a symmetric form for the half-point values _𝜌𝑌𝑖_[|] | _𝑚_ ± 12 and _𝜌𝑒_ | _𝑚_ ± 12 to approximately satisfy Eq. (19). 

Following the work by Fujiwara et al. [15], we introduce the following two equations from Eq. (19): 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-10.png)


where _𝜖𝑖_[|] | _𝑚_ = ( _𝜕𝜌𝑌𝜕𝜌𝑒𝑖_ ) _𝜌𝑌𝑗_ ≠ _𝑗 ,𝑝_ ||||| _𝑚_ . These equations are one of the sufficient conditions for Eq. (19), meaning that they are an approximation for Eq. (19). Then, Eq. (21) is shifted by one cell as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-12.png)


Here, different from the procedure by Fujiwara et al. [15] (they eliminated _𝜌𝑒_ | _𝑚_ + 12 ), we combine Eqs. (20) and (22) by taking the difference to obtain a half-point value of _𝜌𝑒_ | _𝑚_ + 12 : 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-14.png)


Eq. (23) has a symmetric form so that a set of half-point values can be readily determined at all cells and becomes an approximate PE condition for the half-point values, _𝜌𝑌𝑖_[|] | _𝑚_ ± 12 and _𝜌𝑒_ | _𝑚_ ± 12 . The choice for the half-point value of species density in Eq. (23) should be arbitrary, as there are no other PE conditions to determine the half-point value, _𝜌𝑌𝑖_[|] | _𝑚_ + 12 . Thus, this study constructs it using a simple average of the neighboring points, 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-16.png)


With Eq. (24), the half-point value of _𝜌𝑒_ | _𝑚_ + 12 in Eq. (23) becomes 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0004-18.png)


4 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

Equation (25) retains the symmetric form and the second term in the right hand side (RHS) serves as a sensor to approximately preserve the PE condition. Note that the half-point values derived in this study are different from those in Fujiwara et al. [15], which we will discuss later. 

## _2.7. Error analysis of approximate PE condition_ 

We assess the error induced by the approximation of the PE condition, specifically the newly proposed half-point values in Eqs. (24) and (25). The PE condition, as given by Eq. (19), can be rewritten by adding the grid spacing Δ _𝑥_ as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-05.png)


where the PE condition is satisfied when _𝑓𝑃𝐸_[|] | _𝑚_ = 0. Substituting the half-point values from Eqs. (24) and (25) into Eq. (26) yields the discrete form of _𝑓𝑃𝐸_[|] | _𝑚_ : 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-07.png)


Then, applying the Taylor series expansion to Eq. (27) provides a modified equation for the PE condition as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-09.png)


In contrast, if the half-point value of _𝜌𝑒_ is evaluated as ( _𝜌𝑒_ | _𝑚_ + _𝜌𝑒_ | _𝑚_ +1)∕2 (a standard scheme) instead of using Eq. (25), the modified equation becomes: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-11.png)


Therefore, by comparing Eq. (28) with Eq. (29), it is evident that the terms in the third line of Eq. (28) arise from the proposed half-point values in Eqs. (24) and (25). As a result, the error introduced by the approximation of the PE condition is associated with these terms, and the error exhibits second-order accuracy. 

The PE condition, as given by Eq. (19), corresponds to the following partial differential equation: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-14.png)


From Eq. (30), the third derivative of _𝜌𝑒_ with respect to _𝑥_ can be computed analytically, yielding the following relationship: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0005-16.png)


Then, substituting Eq. (31) into the modified equation for the proposed half-point values in Eq. (28) yields 

5 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-02.png)


Similarly, by substituting Eq. (31) into the modified equation for the standard half-point values in Eq. (29), we obtain the following: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-04.png)


Thus, the leading PE-preserving error is related to the first and second derivatives of _𝜖𝑖_ and _𝜌𝑌𝑖_ , with the coefficients of these derivative terms differing between the proposed and standard half-point values. 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-06.png)


the second-order error is eliminated, leading to the satisfaction of the PE condition with improved accuracy. 

## _2.8. APEC: proposed numerical flux_ 

This study employs a kinetic energy-preserving scheme based on split forms [19,20,15,16,18]. The numerical fluxes are written as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-10.png)


and the newly proposed internal energy flux that approximately satisfies the PE condition is 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-12.png)


## _2.9. Equation of state_ 

This study applies the Soave-Redlich-Kwong (SRK) equation of state [17] (here the equation of state is described using primitive variables): 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0006-15.png)


6 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

where _𝑅𝑢_ is the universal gas constant ( _𝑅𝑢_ = 8 _._ 314 J mol[−1] K[−1] ), _𝑀[̄]_ is the mean molecular weight of the mixture, and _𝑇_ is the temperature. The parameters for the SRK equation of state are defined using a classical mixing rule as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-03.png)


where _𝑋𝑖_ is the mole fraction of species _𝑖_ . The binary parameters are given as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-05.png)


and the species-dependent parameters are 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-07.png)


The function _𝑓_ ( _𝜔𝑖_ ) in Eq. (45) is defined with _𝑓_ ( _𝜔𝑖_ ) = 0 _._ 480 + 1 _._ 574 _𝜔𝑖_ −0 _._ 176 _𝜔_[2] _𝑖_[,][where] _[𝜔][𝑖]_[is][the][acentric][factor][of][species] _[𝑖]_[.][The] subscript _𝑐_ refers to the critical point. 

## _2.10. Partial derivatives for APEC_ 

The partial derivative _𝜖𝑖_ needs to be calculated to obtain the internal energy flux of Eq. (40). The partial derivative can be obtained as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-11.png)


The detailed derivation of Eq. (48) is provided in Appendix B. For the SRK equation of state, the partial derivatives in Eq. (48) are obtained, respectively, as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-13.png)


where _𝑒𝑖,_ 0( _𝑇_ ) is the internal energy obtained under the assumption of a thermally perfect gas (the internal energy is a function of temperature and species composition). The specific heat at constant volume is 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0007-15.png)


where _𝐶𝑣,_ 0( _𝑇_ ) is the specific heat capacity for a thermally perfect gas. In this study, the thermally perfect gas terms such as _𝑒𝑖,_ 0( _𝑇_ ) and _𝐶𝑣,_ 0( _𝑇_ ) are calculated using Chemkin II [21]. 

## _2.11. Pressure equilibrium with quasi-conservation_ 

This study also provides a way to maintain the PE condition, although the scheme becomes quasi-conservative. The PE condition of Eq. (19) can be rewritten under the VE (constant velocity) as follows: 

7 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-02.png)


Therefore, if the internal energy flux term _[𝜕] 𝜕𝑥[𝜌][𝑒𝑢]_[in][the][total][energy][equation][is][replaced][with][∑] _[𝑁] 𝑖_ =1 _[𝜖][𝑖] 𝜕𝜌𝜕𝑥𝑌𝑖𝑢_[,][the][pressure][equilibrium] can be preserved precisely. In this way, the total energy equation may be recast to 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-04.png)


The discretization for the RHS terms should be consistent with that for the left hand side (LHS) terms. The second term on the RHS can be evaluated as 

where 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-07.png)


Equation (54) is the quasi-conservative total energy equation to preserve the PE condition. One can retain the form of the total energy equation, and the modification from conservative schemes is easy. Moreover, the derivation and use of the pressure evolution equation [2,8], which is used to preserve the PE condition, are not needed. This would be convenient in viscous flow cases since the original forms of viscous, heat, and diffusion terms in the conservative governing equations can be retained. In this study, the method to use Eq. (54) is denoted as PEqC (pressure equilibrium with quasi-conservation). 

## **3. Results** 

## _3.1. Calorically perfect gas case_ 

We first investigate the performance of APEC under a calorically perfect gas assumption following the previous study [15]. Note that the half-point values of _𝜌𝑌𝑖_ and _𝜌𝑒_ proposed in this study differ from the previous study [15]. 

The calorically perfect gas equation of state for _𝑁_ -component multi-species flows is written as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-13.png)


The specific heat ratio for the mixture _̄𝛾_ is obtained through 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-15.png)


where _𝛾𝑖_ is the specific heat ratio of species _𝑖_ . Then, the partial derivative for APEC is calculated with 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-17.png)


Fujiwara et al. [15] proposed the half-point values to satisfy the PE condition of Eq. (19) in a different manner. Their half-point values are 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0008-19.png)


It is worth noting that if Eq. (60) is substituted into Eq. (23) of the proposed _𝜌𝑒_ | _𝑚_ + 12 half-point form, Eq. (23) yields Eq. (61). Here, it should also be noted that APEC does not reduce to Fujiwara’s half-point values in the calorically perfect gas case. Fujiwara’s half-point values were derived by assuming constant specific heat and the mixing rule described in Eq. (58). Further, with Eqs. (57) and (61), we interpret Fujiwara’s method that the half-point value of _𝜌𝑌𝑖_ | _𝑚_ + 12 in Eq. (60) was designed so that the transport equation of the specific heat ratio [1] was solved in the system. Thus, in the derivation of Fujiwara’s half-point values, it can be speculated that the transport equation of the specific heat ratio was implicitly included as a condition to be satisfied. 

A one-dimensional (1-D) smooth interface advection between two species is simulated. The computational conditions are the same as those in the previous study [15]. The species 1 has the specific heat ratio of _𝛾_ 1 = 1 _._ 4 and the molecular weight of _𝑀_ 1 = 28 while the species 2 has _𝛾_ 2 = 1 _._ 66 and _𝑀_ 2 = 4. The simulation is performed in a _𝑥_ = [0 ∶1] region using 501 grid points. Periodic boundary conditions are used. The initial profiles are generated with 

8 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0009-02.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0009-03.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0009-04.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0009-05.png)


**Fig. 1.** Comparison of distributions of species density, velocity, and pressure at _𝑡_ = 8 _._ 0 for the 1-D interface advection in the calorically perfect gas case. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.) 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0009-07.png)


where _𝑟_ = | _𝑥_ − _𝑥𝑐_ |. _𝑥𝑐_ is the position of wave center and _𝑟𝑐_ is the distance between wave center and interface. This study uses the same values as [15]: _𝑥𝑐_ = 0 _._ 5, _𝑟𝑐_ = 0 _._ 25, _𝑤_ 1 = 0 _._ 6, _𝑤_ 2 = 0 _._ 2, and _𝑘_ = 20. The third-order TVD Runge-Kutta scheme [22] is used for the time integration with the CFL number 0.6. 

Here, three methods are compared: APEC, a fully-conservation but no pressure-equilibrium preserving scheme (FC-NPE), and Fujiwara’s scheme [15]. FC-NPE is a standard scheme in the split form: the half-point value for _𝜌𝑌𝑖_ and _𝜌𝑒_ are evaluated with ( _𝜌𝑌𝑖_[|] | _𝑚_ + _𝜌𝑌𝑖_[|] | _𝑚_ +1)∕2 and ( _𝜌𝑒_ | _𝑚_ + _𝜌𝑒_ | _𝑚_ +1)∕2, respectively. 

Fig. 1 shows a comparison of the profiles of _𝜌_ 1, _𝜌_ 2, _𝑢_ , and _𝑝_ at _𝑡_ = 8 _._ 0 (eight flow-through times). The results show that FC-NPE only generates spurious oscillations in the pressure and velocity profiles, which harmfully affect the species density profiles. Since the oscillation amplitudes significantly grow with time, the computation with FC-NPE eventually blows up after _𝑡_ = 9 _._ 0. In contrast, the APEC results show that the pressure and velocity equilibria are sufficiently maintained with no significant oscillations. Thus, the 

9 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0010-02.png)


**Fig. 2.** Comparisons of error histories for the 1-D interface advection in the calorically perfect gas case. Note that the pressure-error result of Fujiwara is calculated in this study and slightly different from the original result of Fujiwara et al. [15]. 

validity of the proposed half-point values of Eqs. (24) and (25) for preserving the pressure and velocity equilibria is demonstrated. The species density profiles are smooth without visible oscillations. The result of APEC is very similar to that of Fujiwara et al. [15]. Fig. 2 shows the time histories of the energy conservation and pressure equilibrium (PE) errors with three methods. The time evolution of the energy conservation error is obtained with[∑] _𝑡_ {(∑ _𝑁𝑚_ =1 _𝑔[𝐸][𝑚]_[(] _[𝑡]_[)∕][∑] _[𝑁] 𝑚_ =1 _[𝑔][𝐸]_[0] _[,𝑚]_ ) −1} where _𝑚_ is the grid index and _𝑁𝑔_ is the total number of grid points. The PE error is defined with ~~√∑~~ _𝑁𝑚_ =1 _𝑔_ ~~(~~ _𝑝𝑚_ ( _𝑡_ )∕ _𝑝_ 0 −1 ~~)~~ 2 ∕ _𝑁𝑔_ . It is demonstrated that all three methods satisfy the total energy conservation at a sufficiently small order due to using a conservative form in the governing equations. On the other hand, the PE error shows that the error with FC-NPE starts to diverge around _𝑡_ = 4 _._ 0, and finally, the computation fails owing to the severe spurious oscillations, as discussed. Although the PE error with APEC is more significant than that with Fujiwara’s scheme, the error is sufficiently small at the order of 10[−5] to 10[−4] and more importantly, no divergence behavior is observed during the long-time computation. The results in this section demonstrate that the proposed APEC retains the energy conservationand approximate pressure-equilibrium-preserving properties. 

The PE error in APEC results from the approximation of the PE condition, as discussed in Section 2.7. From Eqs. (32) and (33), the leading PE-preserving error in APEC is given by 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0010-06.png)


whereas the error in FC-NPE is 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0010-08.png)


Fig. 3 shows the distributions of the PE-preserving error for APEC (Eq. (63)) and FC-NPE (Eq. (64)) at _𝑡_ = 0, corresponding to the initial profile. The derivatives in these equations are evaluated using central differencing. As expected, the error is most pronounced in the interface regions, where _𝜖𝑖_ and _𝜌𝑌𝑖_ exhibit the largest variations. The results show that APEC exhibits a smaller error than FC-NPE, due to the construction of half-point values that account for the PE condition. The error for APEC is symmetrically distributed, consistent 

10 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0011-02.png)


**Fig. 3.** Comparison of PE-preserving error distributions between APEC and FC-NPE in the calorically perfect gas case at _𝑡_ = 0. 

**Fig. 4.** Time histories of the error norm for APEC and FC-NPE in the calorically perfect gas case. 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0011-05.png)


**Fig. 5.** The trend of approximation error against grid spacing in APEC. The circle symbol denotes the computational result. The error and grid spacing are scaled by the values with the coarsest grid (251 points). The error is evaluated at _𝑡_ = 20 _._ 0. 

with the symmetric form of the error term in Eq. (63). Fig. 4 shows the time histories of the error norm for APEC and FC-NPE. The error norm is computed as √∑ _𝑁𝑚_ =1 _𝑔_ { _𝑒_ APEC( _𝑥_ )}2||| _𝑚_ ∕ _𝑁𝑔_ and ~~√~~ ∑ _𝑁𝑚_ =1 _𝑔_ { _𝑒_ FC−NPE( _𝑥_ )}2||| _𝑚_ ∕ _𝑁𝑔_ . The results indicate that the error norm for APEC is smaller than that for FC-NPE and remains nearly constant over the long-time computation period. This constant behavior suggests that the profiles of _𝜖𝑖_ and _𝜌𝑌𝑖_ are smoothly maintained, with no significant pressure oscillations. In contrast, the error norm for FC-NPE shows a diverging trend around _𝑡_ = 4 _._ 0, consistent with the PE error observed in Fig. 2(b). Fig. 5 shows the error trend of _𝑓𝑃𝐸_ from Eq. (32) as a function of grid spacing. The error of _𝑓𝑃𝐸_ is computed as √∑ _𝑁𝑚_ =1 _𝑔[𝑓] 𝑃𝐸_[2] ||| _𝑚_ ∕ _𝑁𝑔_ . The results demonstrate that the error of APEC decreases with grid spacing at second-order accuracy, consistent with Eq. (32) and Eq. (63). 

On the other hand, the half-point values by Fujiwara et al. (Eqs. (60) and (61)) precisely satisfy the PE condition. However, their half-point values are only applicable to the calorically perfect gas equation of state in Eq. (57) and the mixing rule in Eq. (58). 

11 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

**Fig. 6.** Comparison of distributions of total density, temperature, velocity, and pressure at _𝑡_ = 0 _._ 07 s in the 1-D CH4/N2 interface advection problem. The density, temperature, velocity, and pressure are normalized by _𝜌_ CH4 _,_ ∞, _𝑇_ N2 _,_ ∞, _𝑢_ ∞, and _𝑝_ ∞, respectively. The velocity and pressure profiles are offset by 1.5 and 1.2 times, respectively, for visibility. 

## _3.2. Real-fluid flow cases_ 

In this section, we consider a real-fluid flow case, consisting of CH4/N2 interface advection problems at a supercritical pressure, in which the SRK equation of state is applied. The critical values of CH4 are _𝑝𝑐,_ CH4 = 4 _._ 599 MPa, _𝑇𝑐,_ CH4 = 190 _._ 56 K, and _𝜌𝑐,_ CH4 = 162 _._ 66 kg/m[3] ; those of N2 are _𝑝𝑐,_ N2 = 3 _._ 396 MPa, _𝑇𝑐,_ N2 = 126 _._ 19 K, and _𝜌𝑐,_ N2 = 313 _._ 3 kg/m[3] . 

## _3.2.1. 1-D smooth interface advection_ 

A 1-D CH4/N2 smooth interface advection under a transcritical condition is simulated. The simulation is performed in a _𝑥_ = [0 ∶1] m region with periodic boundaries. The initial profiles are generated with 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0012-07.png)


12 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0013-02.png)


**Fig. 7.** Comparisons of error histories in the 1-D CH4/N2 interface advection problem (twenty flow-through times). In the energy conservation error, the profiles of APEC and FC-NPE are overlapped each other. 

In this case, the uniform velocity _𝑢_ ∞ is 100 m/s, and the uniform pressure _𝑝_ ∞ is a supercritical pressure of 5 MPa. The species densities are set to _𝜌_ CH4 _,_ ∞ = 400 kg/m[3] and _𝜌_ N2 _,_ ∞ = 100 kg/m[3] , respectively. The temperatures of each species calculated with the SRK equation of state are _𝑇_ CH4 _,_ ∞ = 128 _._ 12 K and _𝑇_ N2 _,_ ∞ = 190 _._ 18 K, and thus the condition corresponds to a transcritical condition. The parameters for the hyperbolic tangent are _𝑟_ = | _𝑥_ − _𝑥𝑐_ |, _𝑥𝑐_ = 0 _._ 5 m, _𝑟𝑐_ = 0 _._ 25 m, and _𝑘_ = 15. The number of grid points is 501. Note that the smoother profile with _𝑘_ = 15, compared to the previous calorically perfect gas case, is used to properly resolve the liquid-like real-fluid interface. In this case, the effects of grid resolutions are also addressed. 

Fig. 6 shows the distributions of _𝜌_ , _𝑇_ , _𝑢_ , and _𝑝_ obtained using three methods: APEC, FC-NPE, and PEqC. PEqC is a quasi-conservative scheme that preserves the PE condition presented in Section 2.11. Note that, for the real-fluid case, Fujiwara’s scheme cannot be applied because its half-point values (numerical fluxes) are only for the calorically perfect gas in Eq. (57) and the mixing rule in Eq. (58). Similar to the calorically perfect gas case, FC-NPE suffers from significant spurious pressure and velocity oscillations. The oscillations disturb the temperature and density distributions, and eventually, the computation blows up after _𝑡_ = 0 _._ 09 s. In contrast, APEC generates no significant spurious oscillations in the pressure and velocity fields, demonstrating its applicability to a real-fluid simulation with transcritical conditions. The result from PEqC shows that the pressure and velocity equilibria are maintained, and the density and temperature distributions are similar to those of APEC. 

Fig. 7 compares the time histories of PE and energy conservation errors for three methods (the error definitions are provided in the previous section). In the PE error, the result for PEqC demonstrates that the PE condition is maintained at a minimal level of 10[−8] throughout the computation. Although the PE error of APEC is larger than PEqC due to the approximation of PE condition, no diverged behaviors are observed during the long-term computation, which spans twenty flow-through times. Regarding energy conservation, APEC and FC-NPE maintain the energy conservation. In contrast, the energy conservation error increases with PEqC due to the use of the quasi-conservative form in the total energy equation. 

Fig. 8 shows the distributions of the PE-preserving error for APEC (Eq. (63)) and FC-NPE (Eq. (64)) at _𝑡_ = 0 s. As observed in the previous case of the calorically perfect gas (Fig. 3), APEC exhibits a smaller PE-preserving error than FC-NPE in this real-fluid case. The maximum PE error for APEC is approximately one-quarter of that for FC-NPE, consistent with the results from the previous case. Thus, the reduction in error may be attributed to the difference in the coefficients of the derivative terms: 12[1][for][APEC][and][1] 3[for][FC-] NPE. The error norm for APEC in Eq. (63) and FC-NPE in Eq. (64) is compared in Fig. 9. Similar to the previous case of the calorically perfect gas, the error norm for APEC is smaller than that for FC-NPE. The error norm for APEC remains nearly constant throughout 

13 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0014-02.png)


**Fig. 8.** Comparison of PE-preserving error distributions between APEC and FC-NPE in the 1-D CH4/N2 interface advection problem at _𝑡_ = 0 s. 

**Fig. 9.** Time histories of the error norm for APEC and FC-NPE in the 1-D CH4/N2 interface advection problem. 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0014-05.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0014-06.png)


**Fig. 10.** Convergence of approximation error against grid spacing with APEC in the 1-D CH4/N2 interface advection problem. The circle symbol denotes the computational result. The error is scaled by the result with the coarsest grid (251 points) and is evaluated at _𝑡_ = 0 _._ 05 s. 

the computation period. In contrast, the error norm for FC-NPE gradually increases and shows diverging behavior, accompanied by spurious pressure oscillations. 

Fig. 10 shows the trend of the PE error against the grid spacing with APEC. The results demonstrate that APEC has second-order accuracy in the PE approximation. Since FC-NPE also has second-order accuracy in the PE approximation, the improved performance of APEC over FC-NPE is attributed to the modified coefficient in the second-order error term in Eq. (63). 

Fig. 11 shows the trends of PE error for four grid resolutions: 501, 1001, 2001, and 4001 grid points. The results with APEC demonstrate that increasing the grid resolution effectively reduces the PE error, and with 4001 grid points, the error remains almost constant at its initial value. In contrast, the PE error diverges in all grid cases with FC-NPE. Even with the 4001 grid points, the computation experiences blow-up around _𝑡_ = 0 _._ 11 s due to the increase of PE error. Therefore, the results indicate that FC-NPE cannot control the diverging behavior in the PE error by increasing the grid resolution. 

14 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0015-02.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0015-03.png)


**Fig. 11.** Effects of grid resolutions on the pressure equilibrium error in the 1-D CH4/N2 interface advection problem. 

## _3.2.2. Multi-dimensional interface advection_ 

A two-dimensional (2-D) CH4/N2 smooth interface advection problem is solved to demonstrate the multi-dimensional applicability of APEC. The problem is similar to the 1-D CH4/N2 smooth interface advection, but the blob of CH4 is advected in a diagonal direction of the 2-D space. A similar simulation was performed in the previous study [15] to demonstrate the multi-dimensional applicability of their proposed method under the calorically perfect gas assumption. The simulation is performed in a domain of _𝑥_ = [0 ∶1] m and _𝑦_ = [0 ∶1] m using 501 × 501 grid points. Periodic boundary conditions are used. The initial profiles are generated with 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0015-07.png)


The initial _𝑦_ -velocity _𝑣_ ∞ is set to 100 m/s while the others ( _𝑢_ ∞, _𝑝_ ∞, _𝜌_ CH4 _,_ ∞, and _𝜌_ N2 _,_ ∞) are the same as those in the 1-D case. The parameters for the hyperbolic tangent are _𝑟_ = √( _𝑥_ − _𝑥𝑐_ )[2] + ( _𝑦_ − _𝑦𝑐_ )[2] , _𝑥𝑐_ = 0 _._ 5 m, _𝑦𝑐_ = 0 _._ 5 m, _𝑟𝑐_ = 0 _._ 25 m, and _𝑘_ = 15. The third-order TVD Runge-Kutta scheme is used for the time integration with the CFL number 0.6. 

Fig. 12 shows the fields of temperature, pressure, and _𝑥_ -velocity component at _𝑡_ = 0 _._ 09 s (nine flow-through times) in comparison between FC-NPE and APEC. The results of FC-NPE indicate that severe pressure and velocity oscillations are generated, harmfully disturbing the temperature field with a maximum overshoot error of approximately 10%. The maximum error at _𝑡_ = 0 _._ 09 s is approximately 10% in pressure and 5% in velocity (the contour range in Fig. 12 is limited to 1% to highlight the oscillations), and eventually the computation blows up around _𝑡_ = 0 _._ 1 s. In contrast, the pressure and velocity errors with APEC are significantly smaller than for FC-NPE. The maximum error at _𝑡_ = 0 _._ 09 s with APEC is approximately 0.02% in pressure and 0.01% in velocity. Large-scale disturbances with small errors are only generated in the pressure and velocity field, and the temperature field is maintained correctly without oscillations. Note that the trends of pressure equilibrium and energy conservation errors in the 2-D simulation are very similar to those in the 1-D simulation shown in Fig. 7, and thus not shown here. 

15 

_H. Terashima, N. Ly and M. Ihme_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-01.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-02.png)


<br>


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-03.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-04.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-05.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-06.png)


**Fig. 12.** Comparisons of temperature, pressure, and _𝑥_ -velocity component distributions at _𝑡_ = 0 _._ 09 s in the 2-D CH4/N2 advection simulation. Left: FC-NPE; Right: APEC. 

## _3.2.3. Multi-dimensional interface advection under tangential velocity profile_ 

The last example is a 2-D interface advection problem with a tangential velocity profile, which is introduced in [15]. The simulation is performed in a region of _𝑥_ = [0 ∶1] m and _𝑦_ = [0 ∶1] m and the periodic condition is used for all boundaries. The third-order TVD Runge-Kutta scheme is used for the time integration with the CFL number 0.6. The initial profiles are generated with 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0016-10.png)


16 

_H. Terashima, N. Ly and M. Ihme_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0017-01.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0017-02.png)


<br>


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0017-03.png)


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0017-04.png)


**Fig. 13.** Sequential distributions of the mass fraction of CH4 in the 2-D CH4/N2 advection with tangential velocity profile obtained with APEC. 

The parameters in Eq. (67) are the same as those in the previous 2-D problem in Section 3.2.2 except _𝑣_ ∞ = 15 m/s for the _𝑦_ -velocity. In this problem, the initial _𝑥_ -velocity, _𝑦_ -velocity, and pressure profiles are analytically constant in time while other variable profiles, such as the species density or temperature, change from the initial profile due to the _𝑦_ -velocity varying in _𝑥_ -direction. Further, unlike the previous case without tangential velocity profiles, the interface thickness can be thin during the advection. Here, 2001 × 2001 grid points are used. This high grid resolution ensures that the thinning interface thickness (e.g., mass fraction profiles) is sufficiently resolved during the advection. The computation is performed for four flow-through times. 

Fig. 13 shows sequential distributions of the mass fraction of CH4 obtained with APEC. The fluid of CH4 is deformed owing to the _𝑦_ -velocity stratification varying in _𝑥_ -direction, and the interface thickness in the left and right sides of the blob becomes thinner with time, unlike the previous advection cases. Fig. 14 compares the profiles of pressure, _𝑦_ -velocity, and temperature at _𝑦_ = 0 _._ 5 m and _𝑡_ = 0 _._ 04 s (four flow-through times) between FC-NPE and APEC. The results with FC-NPE exhibit spurious pressure and velocity oscillations even with the employed high grid resolution of 2001 × 2001. The pressure oscillations lead to the oscillations in the temperature profile. Although those oscillations are not severe at _𝑡_ = 0 _._ 04 s, they grow rapidly, causing the computation to fail before _𝑡_ = 0 _._ 044 s. In contrast, APEC is free from such severe pressure and velocity oscillations, and the smooth temperature profile is obtained with no significant oscillations. The pressure and the _𝑦_ -velocity component properly maintain the initial profiles. Although not shown here, the total energy is conserved during the advection with APEC. 

It should be noted that since the interface thickness becomes thin, numerical instabilities eventually occur with APEC which solves the Euler equations. In such situations, the numerical instabilities are triggered by oscillations in the species density (or temperature) due to the lack of grid resolution to resolve the interface thickness, not oscillations driven by PE errors. The computation with APEC would not blow up as long as the interface is sufficiently resolved by employed grids. In contrast, FC-NPE suffers from spurious pressure oscillations even if the interface is sufficiently resolved: even with a higher grid resolution of 3001 × 3001, spurious pressure oscillations appear around _𝑡_ = 0 _._ 03 s and similar results to those in Fig. 14 are obtained at _𝑡_ = 0 _._ 04 s. Thus, as demonstrated in the grid resolution study in the 1-D CH4/N2 interface advection problem, increasing the grid resolution would not solve the PE error problem in FC-NPE. 

17 

_H. Terashima, N. Ly and M. Ihme_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0018-01.png)


<br>


**Fig. 14.** Comparisons of profiles of pressure, _𝑦_ -velocity component, and temperature at _𝑦_ = 0 _._ 5 m and _𝑡_ = 0 _._ 04 s in the 2-D CH4/N2 advection with tangential velocity profile. Left: FC-NPE; Right: APEC. 

The results of this study demonstrate that APEC exhibits improved PE-preserving properties compared to FC-NPE. However, APEC still shows a second-order error in the PE condition, which could lead to numerical instabilities if the grid resolution is insufficient, especially in cases involving large thermodynamic variations, such as those with large density ratios across interfaces. Therefore, future studies will focus on reducing the second-order PE error in extensions of APEC. Additionally, enhancing the robustness of APEC for discontinuous flow problems will be a key focus for its practical applications in future work. 

## **4. Conclusions** 

This study has proposed a conservativeand approximately pressure-equilibrium-preserving scheme for fluid interfaces in compressible multi-species and real-fluid flow simulations. The compressible Euler equations and the SRK equation of state were used in this study. The proposed scheme, named APEC, maintains the full conservation property of the governing equations while approximately satisfying the pressure equilibrium property at fluid interfaces. The numerical flux of internal energy was constructed so that the pressure equilibrium condition was approximately satisfied. The analysis using the modified equation for the pressure equilibrium condition showed that APEC produces different coefficients for the second-order error term compared to the standard scheme, FC-NPE. Comparisons of the second-order error distributions between APEC and FC-NPE in the 1-D interface advection problems demonstrated that the error in APEC is smaller than that in FC-NPE, with the maximum error peak being approximately four 

18 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 

times smaller. The energy conservation and pressure equilibrium properties of APEC were validated through one-dimensional and two-dimensional interface advection problems. The simulation results demonstrated the improved pressure equilibrium properties of APEC compared to FC-NPE. However, APEC still exhibits a second-order pressure equilibrium error, which could lead to numerical instabilities if an adequate grid resolution is not employed, particularly in cases involving large thermodynamic variations, such as those with large density ratios across interfaces. Future studies will focus on reducing the pressure equilibrium error in extensions of APEC. 

## **CRediT authorship contribution statement** 

**H. Terashima:** Writing – original draft, Visualization, Supervision, Software, Methodology, Investigation, Funding acquisition, Formal analysis, Conceptualization. **N. Ly:** Writing – review & editing, Methodology, Investigation, Conceptualization. **M. Ihme:** Writing – review & editing, Supervision, Methodology, Investigation, Conceptualization. 

## **Declaration of competing interest** 

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

## **Acknowledgements** 

This research was supported by JSPS KAKENHI Grant Numbers 21H01522, and 21KK0250. NL and MI acknowledge financial support by NASA Space Technology Graduate Research Opportunities, NASA Award #80NSSC20K1171. 

## **Appendix A. APEC for upwind schemes** 

The idea of APEC can be applied to conventional upwind type schemes such as HLLC [23]. Using Eq. (17) and (18), Eq. (13) is rewritten as 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0019-11.png)


Then, the following two equations are introduced similar to Eq. (20) and (22), 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0019-13.png)


and 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0019-15.png)


Combining Eq. (A.2) and (A.3) provides the numerical flux for the total energy that approximately maintains the pressure equilibrium as follows: 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0019-17.png)


where the numerical fluxes of _𝐹𝜌𝑌𝑖_ ||| _𝑚_ + 12 and _𝐹𝜌𝑢_[|] || _𝑚_ + 12 can be determined with employed upwind schemes. 

## **Appendix B. Partial derivative calculation** 

This section provides a derivation of Eq. (48). Since _𝜌𝑒_ may be a function of _𝜌𝑖_ and _𝑇_ : _𝜌𝑒_ = _𝜌𝑒_ ( _𝜌𝑖,𝑇_ ), the variation of _𝜌𝑒_ is written as follows: 

19 

_H. Terashima, N. Ly and M. Ihme_ 

_Journal of Computational Physics 524 (2025) 113701_ 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0020-02.png)


The pressure may be also described using _𝜌𝑖_ and _𝑇_ : _𝑝_ = _𝑝_ ( _𝜌𝑖,𝑇_ ), and thus the variation of pressure is 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0020-04.png)


Combining Eq. (B.1) and (B.2) while eliminating d _𝑇_ yields 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0020-06.png)


Under the conditions of d _𝑝_ = 0 and d _𝜌𝑗_ ≠ _𝑖_ = 0, Eq. (B.3) becomes Eq. (48), 


![](images/2025_Terashima_Ly_Ihme_approximately_pressure_equilibrium_real_fluid.pdf-0020-08.png)


## **Data availability** 

Data will be made available on request. 

## **References** 

- [1] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1) (1996) 150–160. 

- [2] H. Terashima, M. Koshi, Approach for simulating gas–liquid-like flows under supercritical pressures using a high-order central differencing scheme, J. Comput. Phys. 231 (20) (2012) 6907–6923. 

- [3] K.-M. Shyue, A fluid-mixture type algorithm for compressible multicomponent flow with Mie–Grüneisen equation of state, J. Comput. Phys. 171 (2) (2001) 678–707. 

- [4] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2) (2006) 715–732. 

- [5] T. Nonomura, S. Morizawa, H. Terashima, S. Obayashi, K. Fujii, Numerical (error) issues on compressible multicomponent flows using a high-order differencing scheme: weighted compact nonlinear scheme, J. Comput. Phys. 231 (8) (2012) 3181–3210. 

- [6] H. Terashima, S. Kawai, M. Koshi, Consistent numerical diffusion terms for simulating compressible multicomponent flows, Comput. Fluids 88 (2013) 484–495. 

- [7] S. Karni, Multicomponent flow calculations by a consistent primitive algorithm, J. Comput. Phys. 112 (1) (1994) 31–43. 

- [8] S. Karni, Hybrid multifluid algorithms, SIAM J. Sci. Comput. 17 (5) (1996) 1019–1039. 

- [9] G. Allaire, S. Clerc, S. Kokh, A five-equation model for the simulation of interfaces between compressible fluids, J. Comput. Phys. 181 (2) (2002) 577–616. 

- [10] T. Schmitt, L. Selle, A. Ruiz, B. Cuenot, Large-eddy simulation of supercritical-pressure round jets, AIAA J. 48 (9) (2010) 2133–2144. 

- [11] H. Terashima, M. Koshi, Unique characteristics of cryogenic nitrogen jets under supercritical pressures, J. Propuls. Power 29 (6) (2013) 1328–1336. 

- [12] P.C. Ma, Y. Lv, M. Ihme, An entropy-stable hybrid scheme for simulations of transcritical real-fluid flows, J. Comput. Phys. 340 (2017) 330–357. 

- [13] R. Abgrall, S. Karni, Computations of compressible multifluids, J. Comput. Phys. 169 (2) (2001) 594–623. 

- [14] G. Lacaze, T. Schmitt, A. Ruiz, J.C. Oefelein, Comparison of energy-, pressureand enthalpy-based approaches for modeling supercritical flows, Comput. Fluids 181 (2019) 35–56. 

- [15] Y. Fujiwara, Y. Tamaki, S. Kawai, Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component flows, J. Comput. Phys. 478 (2023) 111973. 

- [16] M. Bernades, L. Jofre, F. Capuano, Kinetic-energyand pressure-equilibrium-preserving schemes for real-gas turbulence in the transcritical regime, J. Comput. Phys. 493 (2023) 112477. 

- [17] G. Soave, Equilibrium constants from a modified Redlich–Kwong equation of state, Chem. Eng. Sci. 27 (6) (1972) 1197–1203. 

- [18] N. Shima, Y. Kuya, Y. Tamaki, S. Kawai, Preventing spurious pressure oscillations in split convective form discretization for compressible flows, J. Comput. Phys. 427 (2021) 110060. 

- [19] A. Jameson, Formulation of kinetic energy preserving conservative schemes for gas dynamics and direct numerical simulation of one-dimensional viscous compressible flow in a shock tube using entropy and kinetic energy preserving schemes, J. Sci. Comput. 34 (2008) 188–208. 

- [20] S. Pirozzoli, Generalized conservative approximations of split convective derivative operators, J. Comput. Phys. 229 (19) (2010) 7180–7190. 

- [21] R.J. Kee, F.M. Rupley, J.A. Miller, Chemkin-II: a fortran chemical kinetics package for the analysis of gas-phase chemical kinetics, SAND89-8009, 1989. 

- [22] S. Gottlieb, C.-W. Shu, Total variation diminishing Runge–Kutta schemes, Math. Comput. 67 (221) (1998) 73–85. 

- [23] E.F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the HLL-Riemann solver, Shock Waves 4 (1994) 25–34. 

20 

