Journal of Computational Physics 478 (2023) 111973 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0001-01.png)


Contents lists available at ScienceDirect 

journal homepage: www.elsevier.com/locate/jcp 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0001-05.png)


## Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component flows 

## Yuji Fujiwara, Yoshiharu Tamaki, Soshi Kawai[∗] 

_Department of Aerospace Engineering, Tohoku University, 6-6-01 Aramaki-Aza-Aoba, Aobaku, Sendai, Miyagi 980-8579, Japan_ 

|a r t i c l e<br>i n f o<br>_Article_ _history:_<br>Received 16 June 2022<br>Received in revised form 12 December 2022<br>Accepted 22 January 2023<br>Available online 27 January 2023<br>_Keywords:_<br>Multi-component fow<br>Conservative scheme<br>Pressure equilibrium|a b s t r a c t|
|---|---|
||In compressible fows with a variable ratio of specifc heats, such as multi-component<br>fows, the conventional conservative schemes generate spurious pressure oscillations at<br>multi-component interfaces even when the interfaces are smooth. In this study, we propose<br>a novel spatial discretization scheme that maintains both the primary conservation and<br>pressure equilibrium in discontinuity-free compressible multi-component fows. The key<br>to this study is the compatibility condition, which is the discrete condition for implicitly<br>satisfying the pressure equilibrium at the discrete level. The compatibility condition is<br>derived based on the analytical relation between the mass of species and the ratio of<br>specifc heats in the governing equations. The proposed scheme is constructed by the<br>spatial discretization that satisfes the compatibility condition and can be applied to<br>the arbitrary number of species components. The inviscid smooth interfaces advection<br>problems verify that the proposed scheme satisfes both the pressure equilibrium and<br>primary conservation only by solving the conservation equations, unlike the existing<br>schemes that solve non-conservative or overspecifed equations.<br>© 2023 Elsevier Inc. All rights reserved.|


## **1. Introduction** 

Mixing of multiple fluid components with different properties is an important physical phenomenon in engineering applications, such as combustion and chemical reaction flows. One of the crucial physics in fluid mixing is the equilibrium (i.e., the temporal and spatial balance of the velocity, pressure, etc.) at the fluid interfaces. In inviscid flows without surface tension, since there is no diffusion, the physical velocity and pressure equilibriums must be maintained. However, in compressible flows with a non-constant ratio of specific heats, such as multi-component or supercritical fluid flows, the conventional conservative schemes which solve the conservation equations of the mass, momentum, total energy, and species generate spurious pressure oscillations at the interfaces [1–8]. Such oscillations result from numerical errors with the variation of the ratio of specific heats. In multi-component flows, since the ratio of specific heats depends on the composition of the mixture fluids, the pressure equilibrium cannot be maintained when a scheme does not represent the fluids mixing. Once the spurious pressure oscillations generate, they induce oscillations of other physical quantities such as the density and velocity through the mass and momentum equations. Note that such serious oscillations occur even when the specific heats vary smoothly in space, as will be discussed in this paper. Therefore, the conventional interface-capturing methods (i.e., numerically diffusing the interfaces) are not a fundamental solution to this oscillation problem. In addition, even in viscous 

> *[Corresponding][author.] 

> _E-mail address:_ kawai@tohoku.ac.jp (S. Kawai). 

https://doi.org/10.1016/j.jcp.2023.111973 0021-9991/© 2023 Elsevier Inc. All rights reserved. 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

flow calculations, such oscillations may occur and often lead to the numerical instabilities [6]. Therefore, the equilibrium preserving scheme is essential for high-fidelity simulations. 

Regarding the diffuse-interface methods, previous studies on compressible multi-component flows are dedicated to avoiding spurious oscillations at fluid interfaces. Karni et al. [1,9] proposed a discretization scheme based on the system of primitive variables evolution equations (i.e., density, velocity, and pressure equations) instead of the equations for the conservative quantities (i.e., density, momentum, and total energy equations). Abgrall [2] derived the analytical conditions for preserving the equilibrium in multi-component flows and proposed the equilibrium preserving scheme that directly solves the transport equation of the ratio of specific heats function instead of the conservation equation of the mass of species. Shyue [3] extended the method of Abgrall [2] to the mass fraction form. Saurel et al. [10] also extended the method to a spatially second-order scheme with a general equation of state (EoS). However, these methods do not solve the conservation equations of species. As pointed out by Johnsen [11], the difficulty of these schemes is that the conservation errors of the mass of species increase with time. The conservation errors for each species can be a critical issue in mixing and combustion phenomena. Therefore, to achieve the conservation of primary conservative variables (i.e., mass, momentum, and total energy), the double-flux methods are proposed. The double-flux methods solve both the pressure evolution and total-energy conservation equations [12,13], or both the mass conservation equation and the transport equations of the ratio of specific heats function [4,5]. In these schemes, one extra transport equation is solved in addition to the conservation equations. Although these methods satisfy both the equilibrium and primary conservation, the double-flux methods overspecify the equations for the system. Therefore, two independent solutions of pressure are obtained from the pressure and total-energy equations (similarly, for the mass of species in Ref. [5]). Analytically, the solution of the pressure or mass of species is unique. In addition, the hybrid method [14] that switches between quasi-conservative double-flux and the fullyconservative method has also been proposed, although it does not fundamentally resolve the overspecification problem. Note that several studies have also presented extensions of the methods to higher-order schemes (e.g., the WENO scheme [15] and the compact difference scheme [16]) and the applications to the localized artificial diffusivity (LAD) methods [17,18], supercritical fluids [6–8] and the phase-field methods that contain terms for diffusing and sharping interfaces for compressible multi-phase flows [19,20]. However, to the authors’ knowledge, none of the existing schemes maintains the pressure equilibrium by solving only the conservation equations of the mass of each species, momentum, and total energy. 

To avoid spurious oscillations and overspecified solutions, spatial discretization based on the analytical relations between the governing equations may be one of the solutions. Several studies in recent years have focused on the spatial discretization approach, which implicitly satisfies the equations of secondary quantities (e.g., kinetic energy, entropy, and pressure not directly solved in the system). For example, several researchers proposed kinetic energy preserving (KEP), e.g., [21,22], entropy conserving (EC), e.g., [23,24], and kinetic energy and entropy preserving (KEEP) schemes [25–29]. In these schemes, the analytical relation of the secondary quantities is also satisfied at the discrete level. In addition, Shima et al. [30] proposed a modified KEEP scheme that preserves the pressure equilibrium in single-component flows. These studies demonstrated that a well-designed discretization may implicitly satisfy a certain analytical relation without solving additional equations. This idea may be a key to an equilibrium-preserving approach in multi-component flows. 

In this paper, we propose a fully conservative and pressure-equilibrium preserving scheme for compressible multicomponent flows with smooth interfaces. Here, the terminology “fully conservative” means solving the same number of conservation equations (i.e., the mass of each species, momentum, and total energy) as the unknowns without any extra transport equation. The idea is to implicitly satisfy the equilibrium condition [2,4] at the discrete level. We focus on the mixing rule of specific heats and design the spatial discretization based on the “compatibility condition” that describes the analytical relations between the governing equations and the equilibrium condition. Unlike the previous studies, the proposed scheme maintains both conservation and equilibrium without overspecifying equations. As shown later, the proposed scheme can be applied to an arbitrary number of species components. Note that this paper focuses on the discretization of the inviscid terms in compressible multi-component flows without sharp fluid interfaces or shocks. To our knowledge, even in the discontinuity-free smooth interface conditions considered in this study, no existing scheme simultaneously satisfies the conservation and pressure-equilibrium preservation. Therefore, the proposed scheme will be a baseline scheme to simulate compressible multi-component flows with sharp interfaces or shocks by properly designing and adding numerical diffusion terms to the proposed scheme to capture the numerical discontinuities. 

The structure of this paper is as follows. In Section 2, we show the mathematical model for multi-component compressible inviscid flows and introduce the analytical equilibrium condition for the pressure and velocity. Based on the derivation of the analytical condition, the causes of spurious pressure errors in the conventional conservative methods are discussed. In Section 3, a method that satisfies both the conservation and equilibrium is proposed. Here, we discuss the discrete condition to satisfy the equilibrium condition implicitly. For simplicity, we first derive the fully conservative and pressure-equilibrium preserving scheme from the compatibility condition in two-component flows and then extend it to _N_ -component flows. The inviscid flow problems are conducted to demonstrate the capability of the proposed scheme in Section 4. In Section 5, we discuss the behavior of the proposed schemes in terms of velocity disturbances. Finally, in Section 6, conclusions are drawn. 

2 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

## **2. Mathematical models for compressible multi-component flows** 

## _2.1. Governing equations_ 

The conservation equations for the mass of species, momentum, and total energy for _N_ -component compressible inviscid flows are given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-05.png)


where _ρ_ is the density, _**u**_ is the velocity vector, _p_ is the static pressure, _E_ = _ρ(e_ + _**u**_ · _**u** /_ 2 _)_ is the total energy, _e_ is the internal energy per unit mass, _Y i_ is the mass fraction of species _i (i_ = 1 _,_ 2 _..., N)_ , and _**δ**_ is the unit tensor. In the multicomponent flow system, the mass conservation of each species must be satisfied. Note that the sum of Eq. (1a) for each species yields the mass conservation equation 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-07.png)


because[�] _i[N][Y][i]_[=][ 1][and][�] _i[N][ρ][Y][i]_[=] _[ ρ]_[.][Thus,][there][is][no][need][to][solve][Eq.][(][2][)][directly][when][we][solve][Eq.][(][1][a)][up][to][the] _N_ th species. Note that some studies [5,17,31] solve the _(ρ, ρ_ _**u** , E, ρ Y i)_ system that includes Eq. (2) and Eq. (1a) up to the _N_ − 1th species. They are analytically equivalent to solving the _(ρ Y i , ρ_ _**u** , E)_ system. Therefore, although Eq. (1) is used in this study, the following discussion is valid also for the _(ρ, ρ_ _**u** , E, ρ Y i)_ system (see Appendix A). We assume that the species are calorically perfect gases in thermal equilibrium. The equation of state (EoS) for ideal gases is _p_ = _ρ[R][u] T_ = _ρe(γ_ − 1 _),_ (3) _M_ 

where _R u_ is the universal gas constant, _T_ is the temperature, _M_ is the molar weight for the fluid mixture, and _γ_ is the ratio of specific heats for the fluid mixture. In this paper, the variables with subscript _i_ denote the quantities of each species, and those without any subscript denote the quantities of the fluid mixture. 

## _2.2. Mixing rule_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-11.png)


where _c p_ is the specific heat per mass at constant pressure, and _c v_ is the specific heat per mass at constant volume. These specific heats are calculated using the ideal gases EoS as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-13.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-14.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0003-15.png)


Note that _Mi_ and _γi_ are constant and unique to each species, while _M_ and _γ_ for the mixture are variable in terms of the mass fraction _Y i_ . Eq. (4) is the mixing rule of _G_ for multi-component flows. Although the mixing rule for ideal gases is shown in this section, the mixing rule for other EoS may be defined similarly (see Appendix B for the case of the stiffened EoS). 

3 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

## _2.3. Pressure and velocity equilibriums at fluid interfaces_ 

This section summarizes the analytical derivations of the pressure and velocity equilibriums as the physical phenomena in inviscid flows. First, we derive the equilibrium conditions for an arbitrary EoS by introducing the Jacobian. Then, we show that the extended equilibrium condition encompasses the conventional equilibrium conditions proposed by Abgrall et al. [4] that assume ideal gases. For simplicity, the one-dimensional equations where the initial pressure _p_ 0 and velocity _u_ 0 are at the equilibrium condition, i.e., spatially constant as in prior studies [3,4,18], are considered. At this condition, the time evolution of _ρ Y i, ρu, E_ at the initial state are written as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0004-04.png)


Taking the sum of Eq. (5a) for each species yields the mass-evolution equation 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0004-06.png)


Here, for an arbitrary EoS _ρe_ = _ρe(p, ρ Y i)_ , the derivative of _ρe_ is analytically replaced by the derivatives of _p_ and _ρ Y i_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0004-08.png)


By introducing the Jacobian � _∂∂ρpe_ � _ρY i_ and � _∂∂ρρYei_ � _p,ρY j_ ̸= _i_ , Eq. (7) clarifies the relationships between variables in the governing equations. These transformations of the substitution of derivatives in Eq. (7) are analytically exact. 

The initial constant velocity and pressure distributions must maintain physically in time in inviscid flows without surface tension. The time derivative of velocity and pressure are derived analytically from Eqs. (5), (6), and (7) as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0004-11.png)


These transformations are analytically exact. In the flow with constant initial pressure _p_ 0 and velocity _u_ 0, substituting Eqs. (5) and (6) into Eqs. (8) and (9) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0004-13.png)


If the time derivatives of velocity and pressure are zero, the equilibriums are maintained. The right hand side (RHS) of Eq. (10) contains the transport equation of _ρ_ . The transport equation of the density _ρ_ is automatically satisfied by solving 

4 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

Eq. (5a) in compressible flow simulations as Eq. (6). To maintain the pressure equilibrium, Eq. (9) requires the following relationship between _ρe_ and _ρ Y i_ , 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0005-03.png)


Equation (12) is the substitution derivative of _ρe_ and _ρ Y i_ in space and the analytical relationship clearly holds in pressure equilibrium _(p_ = _p_ 0 _)_ . Since Eq. (12) is the key to derive _[∂] ∂[p] t_[=][0][in][Eq.][(][11][),][the][spurious][pressure][oscillations][may][be] generated when Eq. (12) is not discretely satisfied. Therefore, Eq. (12) is the equilibrium condition that is valid for an arbitrary EoS. 

Note that assuming ideal gases _(ρe_ = _p/(γ_ − 1 _)_ = _pG)_ , the time evolution of the pressure Eq. (11) is represented using Eq. (12) as 

where 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0005-07.png)


which is analytically correct because _G_ = _G(Y i)_ = _G(ρ Y i)_ from the mixing rule (Eq. (4)). The RHS of Eq. (13) contains the transport equation of _G_ . To satisfy the equilibrium for ideal-gas flows, Eq. (13) requires 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0005-09.png)


Equation (15) is the conventional equilibrium condition as suggested by Abgrall et al. [4]. Some of the conventional equilibrium preserving schemes directly solve Eq. (15) as one of the governing equations. Substituting Eqs. (5a) and (15) into Eq. (14) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0005-11.png)


which is the analog to Eq. (12) for the ideal gas EoS. 

## **3. Equilibrium preserving scheme** 

In order to maintain the pressure equilibrium at the interfaces of multi-component fluids, the pressure equilibrium condition Eq. (12) must be satisfied at the discrete level. A simple approach in the ideal gases case is to solve Eq. (15) directly in addition to the governing equations (1) or by replacing the mass of species equation (1a), as in Refs. [4,5]. However, this approach is either overspecified (since the system is already closed by the governing equation (1), EoS (3), and the mixing rule (4)) or non-conservative for the mass of species. In other words, to satisfy the fully conservative and pressure-equilibrium characteristics without overspecification, the pressure equilibrium condition must be satisfied by solving only the conservation equations in Eq. (1). In this study, we propose a spatial discretization that implicitly satisfies the pressure equilibrium condition by solving the equations of primary conservative variables ( _ρ Y i, ρ_ _**u** , E_ ) in Eq. (1). 

To find such a discretization, we derive the discrete condition implicitly satisfying the pressure equilibrium, which represents the transport of _ρe_ as the transport of _ρ Y i_ at the discrete level. For general discussion, first, we will proceed with an arbitrary EoS and then assume the ideal gas flow to determine the detailed discretization. In this study, we treat the time derivative analytically and focus on the spatial discretization. 

5 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

## _3.1. Spacial discretization_ 

In this study, we focus on the spatial discretization and propose a fully conservative and pressure-equilibrium preserving scheme. Although the following equations are described in the one-dimension forms for simplicity, the scheme may be extended straightforwardly to multi-dimension by applying the numerical fluxes dimension-by-dimension as demonstrated in Sections 4.2 and 4.3. In conservative schemes, the spatial derivative terms must be discretized in a numerical flux form. The governing equations (1) are spatially discretized by the half-point numerical fluxes as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0006-04.png)


where 

and _m_ is the cell index. A fully conservative scheme is constructed by solving these equations without any extra transport equations. Since the form of the numerical fluxes at _m_ ±[1] 2[point][is][arbitrary,][we][propose][the][numerical][fluxes][that][preserve] the pressure equilibrium based on the analytical relations between the governing equations. 

## _3.2. Compatibility condition for spatial discretization_ 

Here, we investigate the pressure equilibrium condition (12) at the discrete level. In the flow with constant _p_ 0 and _u_ 0, substituting Eq. (17) into Eqs. (8) and (9) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0006-09.png)


Equation (18) shows that the velocity equilibrium holds at the discrete level, while the pressure equilibrium requires 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0006-11.png)


6 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

Equation (20) is the discrete analog to Eq. (12). Moreover, multiplying _�x_ to Eq. (20) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0007-03.png)


where _βi_ | _m_ ≡ � _∂∂ρρYei_ � _p,ρY j_ ̸= _i_ ���� _m_ . This equation is satisfied analytically but is not guaranteed generally at the discrete level. Therefore, our idea here is to replace the derivatives of _ρe_ with the derivatives of _ρ Y i_ at the discrete level to satisfy Eq. (21). The advantage of this approach is that the proposed method avoids overspecification while maintaining the pressure equilibrium. Therefore, by proposing the spatial discretizations that satisfy Eq. (21), we will achieve both the conservation of _ρ Y i_ , _ρu_ , _E_ and pressure-equilibrium preservation in the governing equations (1) without solving any extra equation. Equation (21) represents the condition of the numerical fluxes, which means that the transports of _ρ Y i_ are compatible with that of _ρe_ . Therefore, we call Eq. (21) the compatibility condition. The numerical fluxes based on the compatibility condition implicitly satisfy Eq. (11) at the discrete level, and as a result, the pressure equilibrium is achieved. 

## _3.3. Compatible half-point values maintaining equilibriums_ 

In this section, we derive the half-point values _ρ Y i_ | _m_ ± 12[and] _[ρ][e]_[|] _[m]_[±][1] 2[that][are][used][for][the][numerical][fluxes][and][satisfy] the compatibility condition (Eq. (21)). To derive the conservative numerical fluxes, the half-point values at _m_ +[1] 2[must][be] equivalent when seeing from the left point _m_ and the right point _m_ + 1. We first derive the modified compatibility condition at _m_ +[1] 2[from][Eq.][(][21][).][Then,][we][propose][the][half-point][values] _[ρ][Y][i]_[|] _m_ +[1] 2[and] _[ρ][e]_[|] _[m]_[+][1] 2[to][maintain][the][pressure][equilibrium.] 

## _3.3.1. Two-component flows_ 

For simplicity, we first consider a two-components flow ( _N_ = 2). First, the modified compatibility condition at _m_ +[1] 2[is] derived from Eq. (21). Equation (21) is decomposed into two equations at _m_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0007-09.png)


These equations are the sufficient conditions to satisfy Eq. (21). Then, Eq. (23) is shifted by one in _m_ direction as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0007-11.png)


Considering the local conservation, _ρe_ | _m_ + 12[in][Eqs.][(][22][)][and][(][24][)][must][be][the][same.][Hence,][eliminating] _[ρ][e]_[|] _[m]_[+][1] 2[from] Eqs. (22) and (24) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0007-13.png)


The discussion of the compatibility condition holds for the Jacobian of an arbitrary EoS until Eq. (25). By calculating the Jacobian from the EoS and substituting it into Eq. (25), the numerical fluxes that satisfy the compatibility condition are obtained. In this paper, although the derivations will be discussed under the assumption of ideal gases in the following, the derivations for EoS other than the ideal gases, the stiffened EoS often used for multi-phase flow simulations, are presented as an example in Appendix B. The derivation of the numerical fluxes for more complex EoS will be future work. 

Assuming an ideal gas _(ρe_ = _p/(γ_ − 1 _)_ = _pG)_ , _ρe_ and _βi_ are described analytically using _ρ Y_ 1 and _ρ Y_ 2 by the mixing rule of _G_ (Eq. (4)) as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0007-16.png)


where the following relationship is employed: 

7 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-02.png)


Since Eqs. (26)–(28) contain _p_ 0, Eq. (25) is described in terms of _G_ and _ρ Y i_ by dividing _p_ 0 as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-04.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-05.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-06.png)


Substituting Eqs. (31)–(33) into Eq. (25) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-08.png)


Equation (34) is the sufficient condition for the half-point values _ρ Y_ 1| _m_ + 12[and] _[ρ][Y]_[2][|] _[m]_[+][1] 2[to][realize][the][pressure][equilibrium] at the discrete level. 

_3.3.2. Derivation of the half-point values_ The half-point values _ρ Y_ 1| _m_ + 12[and] _[ρ][Y]_[2][|] _[m]_[+][1] 2[are][arbitrary,][and][therefore,][we][propose][simple][forms][of] _[ρ][Y]_[1][|] _[m]_[+][1] 2[and] _ρ Y_ 2| _m_ + 12[that][satisfy][Eq.][(][34][).][To][simplify][the][representation,][Eq.][(][34][)][is][rewritten][by][introducing][coefficients] _[φ]_[±][as] _ρY_ 1| _m_ +1 _ρY_ 2| _m_ − _ρY_ 1| _mρY_ 2| _m_ +1 = � _φ_[−] _ρY_ 2| _m_ − _φ_[+] _ρY_ 2| _m_ +1� _ρY_ 1| _m_ + 12[−] � _φ_[−] _ρY_ 1| _m_ − _φ_[+] _ρY_ 1| _m_ +1� _ρY_ 2| _m_ + 12 _[,]_ (35) 

where 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0008-12.png)


8 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

Equation (36) shows that once either _ρ Y_ 1| _m_ + 12[or] _[ρ][Y]_[2][|] _[m]_[+][1] 2[is][determined,][the][other][is][obtained][uniquely.][While][Eq.][(][36][)] holds for any _ρ Y_ 1 and _ρ Y_ 2, they become discontinuous when each denominator is zero. Therefore, we derive the half-point value that is available continuously by considering the discontinuous limit. In Eq. (36b), _ρ Y_ 2| _m_ + 12[is][discontinuous][when][the] denominator become zero (i.e., _φ_[−] _ρ Y_ 1| _m_ = _φ_[+] _ρ Y_ 1| _m_ +1), and then Eq. (36a) asymptotically approaches the following value 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-03.png)


Under this discontinuity condition, the half-point value _ρ Y_ 1| _m_ + 12[should][be][automatically][reduced][to][Eq.][(][37][),][and][a][similar] argument can be made for _ρ Y_ 2| _m_ + 12[as][well.] We propose the half-point values in the numerical fluxes that satisfy Eq. (34) as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-05.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-06.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-07.png)


The derived _G_ | _m_ + 12[is][used][as][a][part][of] _[ρ][e]_[|] _[m]_[+][1] 2[to][calculate][�] _[I]_[in][Eq.][(][17][c).][Equations][(][38][)][and][(][39][)][satisfy][the][compatibility] condition Eq. (21) so that by using Eqs. (38) and (39) to compute the numerical fluxes, the equilibrium condition is implicitly satisfied at the discrete level by only solving the conservation equations (1). Therefore, the proposed scheme may maintain both the conservation and pressure equilibrium. 

## _3.3.3. Extension to N-component flows_ 

Next, we extend the proposed scheme to flows with an arbitrary number _N_ of components. The compatibility condition in the _N_ -component system may be derived similarly to Eq. (30) as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-11.png)


where 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0009-13.png)


9 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0010-02.png)


From Eqs. (41)–(43), each term in Eq. (40) is calculated as 

Substituting Eqs. (44)–(46) into Eq. (40) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0010-05.png)


Equation (47) is an identical equation for each _i, j_ component. Because each _i, j_ component equation is consistent with the two-component case, we can derive the numerical flux similarly. 

The half-point values that satisfy Eq. (47) are 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0010-08.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0010-09.png)


When _N_ = 2, Eq. (48) becomes identical with Eqs. (38) and (39). The proposed half-point values have the same form for each species _i_ and satisfy the compatibility condition in the _N_ -component system. In other words, the proposed scheme is consistent with the fact that the mass of each species is independent of each other and that all species are equal for _i_ (following Dalton’s law). Also, considering the sum of each species, the total mass _ρ_ | _m_ + 12[(which][is][not][solved][directly][in] this system) is written as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0010-11.png)


10 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

**Table 1** 

Proposed fully conservative and pressure-equilibrium (FC/PE) preserving numerical fluxes used in compressible multi-component Euler equations in Eq. (17). Although both FC/PE (S) and FC/PE (D) satisfy the compatibility condition Eq. (21), as will be discussed in Section 5, the proposed FC/PE (S) maintains both the conservation and pressure equilibrium. 

|Numerical fux|FC/PE (S) (Proposed)|FC/PE (S) (Proposed)|FC/PE (S) (Proposed)|FC/PE (S) (Proposed)||FC/PE (D)|
|---|---|---|---|---|---|---|
|�<br>_Ci_|_m_+ 1<br>2<br>�<br>_Mu_|_m_+ 1<br>2<br>�_�_|_m_+ 1<br>2<br>�<br>_K_|_m_+ 1<br>2<br>�_I_|_m_+ 1<br>2<br>�<br>_P_|_m_+ 1<br>2|_φ_−_ρYi_|_m_+_φ_+_ρYi_|_m_+1<br>|||_u_|_m_+_u_|<br>|_m_+1<br>_u_|_m_+_u_|_m_+1<br>2<br>_u_|_mu_|_m_+1<br>2|_φ_−_(ρYiu)_|_m_+_φ_+_(ρYiu)_|_m_+1<br>2<br>_φ_−_(ρuu)_|_m_+_φ_+_(ρuu)_|_m_+1<br>2<br>_p_|_m_+_p_|_m_+1<br>2<br>_φ_−_(ρu_ _uu_<br>2 _)_|_m_+_φ_+_(ρu_ _uu_<br>2 _)_|_m_+1<br>2<br>_(pGu)_|_m_+_(pGu)_|_m_+1<br>2<br>_(up)_|_m_+_(up)_|_m_+1<br>2|
||2<br>_φ_−_ρ_|_m_+_φ_+_ρ_|_m_+1||_u_|_m_|2<br>+_u_|_m_+1|||
||2<br>_p_|_m_+_p_|_m_+1<br>2<br>_φ_−_ρ_|_m_+_φ_+_ρ_|_m_+1||_u_|_m_|2<br>+_u_|_m_+1|||
||2<br>_pG_|_m_+_pG_|_m_+1<br>|_u_||_m_+_u_<br>|2<br>|_m_+1|||
||2<br>_u_|_m p_|_m_+1+_u_|_m_<br>|+1_p_|2<br> |_m_||||
||2||||||


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0011-05.png)


Even in the _(ρ, ρ_ _**u** , E, ρ Y i)_ system, the proposed schemes are valid if the _ρ_ | _m_ + 12[in][Eq.][(][49][)][is][used][instead][of] _[ρ][Y][ N]_[|] _[m]_[+][1] 2[.] This fact indicates that solving the _(ρ, ρ_ _**u** , E, ρ Y i)_ system is discretely equivalent to _(ρ Y i, ρ_ _**u** , E)_ system for the proposed scheme (see Appendix A). In conclusion, the proposed scheme is generalized and applicable to multi-component flows for any number of species. 

## _3.4. Complete forms of the proposed scheme satisfying pressure equilibrium_ 

Although the half-point values of _ρ Y i_ | _m_ + 12[and] _[G]_[|] _[m]_[+][1] 2[are][derived][from][the][compatibility][condition][in][Eq.][(][48][),][the] numerical flux in Eq. (17) is not yet uniquely determined, e.g., the mass of species flux _C_[�] _i_ = _ρ Y i u_ may be evaluated in the divergence form _C_[�] _i_ | _m_ ± 12[=] _[ (][ρ][Y][i][u][)]_[|] _[m]_[±][1] 2[or][the][split][form] _C_[�] _i_ | _m_ ± 12[=] _[ ρ][Y][i]_[|] _[m]_[±][1] 2 _[u]_[|] _[m]_[±][1] 2[.][Therefore,][we][examine][the][following] two schemes that satisfy the compatibility condition Eq. (21) using Eqs. (48) and (49). Table 1 summarizes the numerical fluxes of the proposed scheme. First, the fully conservative and pressure-equilibrium preserving scheme in the split form (FC/PE (S)) is designed to return to KEEPPE scheme [30] in a single-component flow. The half-point values in Eq. (48) are used in _C_[�] _i_ and[�] _I_ . _C_[�] _i_ is described in the split form _C_[�] _i_ | _m_ + 12[=] _[ ρ][Y][i]_[|] _[m]_[+][1] 2 _[u]_[|] _[m]_[+][1] 2[.][The][mass-related][fluxes] _M_[�] _u_ and _K_[�] are also �written _I_ | _m_ + 12[=] in _[pG]_ the[|] _[m]_[+] split[1] 2 _[u]_[|] _[m]_ form[+][1] 2[,][to] including[maintain] _ρ_ | _m_[the] + 12[pressure][in][Eq.][(][49][equilibrium][)][to][be][consistent][as][mentioned][with] _C_[�] _i_[in] .[�] _I_[Ref.] is split[[][30][].] into[Other] the[fluxes] internal _[�]_[�][and] energy _P_[�] areandthevelocity,same as the KEEP scheme [25] based on the split form. In addition, the fully conservative and pressure-equilibrium preserving scheme in the divergence form (FC/PE (D)) can be constructed to return to the divergence scheme in a single-component flow. _C_[�] _i_ is described by multiplying the standard divergence form flux by the coefficient _φ_[±] . _M_[�] _u_ and _K_[�] are determined in the same way as _C_[�] _i_ , and other fluxes _�_[�] ,[�] _I_ , and _P_[�] are standard central fluxes in the divergence form. In the flow with constant _p_ = _p_ 0 and _u_ = _u_ 0, the fluxes in both FC/PE (S) and FC/PE (D) reduce to the half-point values in Eq. (48). These two schemes are constructed in a central difference form with second-order spatial accuracy. Higher-order accuracy may be achieved by increasing the number of stencils, similar to the existing high-order KEEP scheme [27] for single-component flows. 

Also, as shown later in Section 5, in terms of the consistency of the compatibility condition to velocity fields, in this study, we recommend the fully conservative and pressure-equilibrium preserving scheme in the split form (FC/PE (S)) as the proposed scheme. The proposed scheme may be extended straightforwardly to multi-dimension by replacing the scalar _u_ with vector _**u**_ and adding _**δ**_ to _�_[�] , similarly to the existing KEEP schemes [26,29]. Note that _u_ | _mu_ | _m_ +1 in _K_[�] represents the inner product of the velocity vectors. 

Next, we discuss the behavior of the proposed scheme outside the interfaces where the composition and specific heats do not change. In other words, _Y i_ is constant (a single-component or well-mixed fluid), and thus, the following relationship holds 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0011-11.png)


11 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0012-02.png)


**Fig. 1.** Initial distributions of the mass of each species in the 1D inviscid smooth interfaces advection problem ( , _ρY_ 1; , _ρY_ 2). (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.) 

where 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0012-05.png)


Here, the proposed numerical fluxes are reduced to a standard central form outside the interfaces. Therefore, FC/PE (S) behaves as the KEEPPE [30] scheme outside the interfaces automatically and can maintain the equilibrium without any special treatment such as switching fluxes by sensors. 

## **4. Numerical experiments** 

In this section, we first verify the proposed scheme defined in Table 1 through a one-dimensional smooth interfaces advection problem in pressure and velocity equilibrium. Then, similar problems in the two-dimensional domain are computed to show the multi-dimensional and multi-component capacity of the proposed scheme. 

## _4.1. 1D inviscid smooth interfaces advection problem in pressure and velocity equilibriums_ 

In this problem, the advection of 1D interfaces between two species is simulated. Here, we assume that the two species have different ratios of specific heats, _γ_ 1 = 1 _._ 4 and _γ_ 2 = 1 _._ 66, and molar weights, _M_ 1 = 28 _._ 0 and _M_ 2 = 4 _._ 0. Since the initial velocity and pressure are constant in space, the pressure and velocity equilibriums should preserve. The simulation is performed in a periodic region [0 _,_ 1] with 501 grid points, where sufficient points are located within the initial smooth interfaces. The initial conditions are given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0012-11.png)


where _r_ = | _x_ − _xc_ |, _xc_ is the coordinate of the wave center, _rc_ is the radius of the wave center to the interfaces, _w_ is the weight coefficient of the density and _k_ is the coefficient to adjust the smoothness. In this case, _xc_ = 0 _._ 5, _rc_ = 0 _._ 25, _(w_ 1 _, w_ 2 _)_ = _(_ 0 _._ 6 _,_ 0 _._ 2 _)_ , and _k_ = 20. Here, the initial pressure is set to 0 _._ 9 just for the visibility of the figures, although it can be any other value. The initial density distributions are shown in Fig. 1. The _ρ Y_ 1 is distributed in 0 _._ 25 _< x <_ 0 _._ 75, and _ρ Y_ 2 is distributed in _x <_ 0 _._ 25 and 0 _._ 75 _< x_ , touching each other through smooth interfaces. The four-stage fourth-order Runge– Kutta method is used with the CFL number of 0.05 ( _�t_ = 10[−][4] ) for the time integration. The proposed scheme (FC/PE (S)) that is described in Table 1 is compared to FC/PE (D), the conventional non-conservative and pressure-equilibrium preserving schemes solving the pressure-evolution equation [1] (NC_p_ ), solving _G_ directly [2] (NC_G_ ), and the divergence form scheme which is fully conservative but not pressure-equilibrium preserving (NPE). Note that NPE is obtained by applying _φ_[+] = _φ_[−] = 1 to FC/PE (D). NPE is conservative but does not satisfy the compatibility condition (Eq. (21)). 

Fig. 2 shows the distributions of _u_ and _p_ at an early stage _t_ = 1 _._ 0. In the results by FC/PE (S) and FC/PE (D), the velocity and pressure equilibriums preserve at a sufficiently small order (≃ 10[−][13] ). When using NPE, the equilibrium is not maintained and spurious oscillations occur in the velocity and pressure. These oscillations originate from the numerical errors of specific heats that are generated by violating the compatibility condition Eq. (21). The result shows that both FC/PE (S) and FC/PE (D) schemes preserve the equilibrium by satisfying the compatibility condition without solving the 

12 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0013-02.png)


**Fig. 2.** Velocity and pressure distributions of 1D inviscid smooth interfaces advection problem at _t_ = 1 _._ 0 ( , exact; , NPE; , FC/PE (S) (Proposed); The results by FC/PE (D), NC_p_ , and NC_G_ are almost the same as that by FC/PE (S)). 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0013-04.png)


**Fig. 3.** Time evolutions of L2 pressure errors of the 1D inviscid smooth interfaces advection problem ( , NPE; , FC/PE (D); , FC/PE (S) (Proposed)). 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0013-06.png)


**Fig. 4.** Distributions of velocity, pressure, and mass of species in the 1D inviscid smooth interfaces advection problem at _t_ = 6 _._ 0 ( , exact; , _u_ ; , _p_ ; , _ρY_ 1; , _ρY_ 2). 

equilibrium condition Eq. (15) directly. However, FC/PE (D) has a stability issue in the pressure-equilibrium preservation at the later stage. Fig. 3 shows the time evolution of pressure errors. FC/PE (S) and FC/PE (D) have almost negligible errors for the pressure in comparison to NPE at the beginning of the calculation ( _t_ ≤ 1). However, the pressure errors of FC/PE (D) rapidly grow with time steps and finally lead to the divergence of computation at _t_ ≈ 3 _._ 8. The causes of this error growth are discussed in Section 5. On the other hand, FC/PE (S) maintains the pressure equilibrium without developing numerical errors. Fig. 4 shows the distributions of _ρ Y i_ , _u_ and _p_ at _t_ = 6 _._ 0. In the result using NPE, spurious oscillations occur in the 

13 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0014-02.png)


**Fig. 5.** Conservation errors in total energy and mass of species 1 in the 1D inviscid smooth interfaces advection problem ( , exact; , FC/PE (S) (Proposed); , NC_p_ [1]; , NC_G_ [2]; Conservation errors by FC/PE (D) and NPE are also exactly zero, while these schemes blow up at _t_ ≈ 3 _._ 8 and _t_ ≈ 6 _._ 5, respectively). 

velocity and pressure and grow with time, which also disturbs the density fields. Finally, the computation by NPE blows up at _t_ ≈ 6 _._ 5. In contrast, FC/PE (S) maintains the equilibrium at the machine zero order for a long time. Therefore, the proposed scheme can maintain the equilibrium, and the proposed FC/PE (S) is superior to FC/PE (D) because the errors do not grow. 

Fig. 5 shows the time evolution of the conservation errors in the total energy _E_ and mass of species _ρ Y_ 1. Analytically, the total energy and mass of species conserve across the entire computational domain. Since the existing pressure-equilibrium preserving schemes (NC_p_ [1] and NC_G_ [2]) are non-conservative, the total energy does not conserve when solving the pressure-evolution equation (NC_p_ ) instead of the total energy equation, while the mass does not conserve when solving _G_ equation (NC_G_ ) instead of the mass of species equation. In contrast, FC/PE (S) (and also FC/PE (D)) satisfies the primary conservation because the proposed scheme solves the equations of the primary conservative quantities directly. Therefore, in this experiment, the proposed scheme (FC/PE (S)) is the only one that satisfies both the conservation and pressureequilibrium preservation for a long time. 

## _4.2. Multi-dimensional multi-component inviscid smooth interfaces advection problem in the pressure and velocity equilibriums_ 

In this problem, we demonstrate that the proposed scheme is applicable to multi-dimensional multi-component interface problems. Here, we assume that the three species have different ratios of specific heats, _γ_ 1 = 1 _._ 4, _γ_ 2 = 1 _._ 66, and _γ_ 3 = 1 _._ 29, and molar weights, _M_ 1 = 28 _._ 0, _M_ 2 = 4 _._ 0, and _M_ 3 = 44 _._ 0. Since the initial pressure and velocity are constant in space, the pressure and velocity equilibriums are maintained analytically. The simulation is performed in a periodic plane of dimension [0 _,_ 1] with 501[2] grid points, where sufficient grid points are located within the smooth interfaces. The initial conditions are given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0014-08.png)


where _r_ = � _(x_ − _xc)_[2] + _( y_ − _yc)_[2] , _(xc, yc)_ = _(_ 0 _._ 5 _,_ 0 _._ 5 _)_ , _(rc_ 1 _, rc_ 2 _, rc_ 3 _)_ = _(_ 0 _._ 3 _,_ 0 _._ 3 _,_ 0 _._ 2 _)_ , _(w_ 1 _, w_ 2 _, w_ 3 _)_ = _(_ 0 _._ 4 _,_ 0 _._ 2 _,_ 0 _._ 1 _)_ , and _k_ = 15. The initial density distributions are shown in Fig. 6. The four-stage fourth-order Runge–Kutta method is used with the CFL number of 0.05 ( _�t_ = 10[−][4] ) for the time integration. Here, the proposed scheme (FC/PE (S)) is compared to the divergence form scheme (NPE). 

Fig. 7 shows the distributions of total density, velocity, and pressure at _t_ = 5 _._ 0, when the circular interface advects through the domain five times. In the results of FC/PE (S), the pressure and velocity equilibriums are maintained at a sufficiently small order (≃ 10[−][12] ). When using NPE, spurious oscillations disturb the structure of waves in the whole region, and the computation blows up at _t_ ≈ 5 _._ 5. These results show that the proposed scheme can be applied in dimension by dimension to multi-dimensional multi-component flows. 

14 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-02.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-03.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-04.png)


**Fig. 6.** Initial conditions of the multi-dimensional multi-component inviscid smooth interfaces advection problem ( , _ρ_ ; , _ρY_ 1; , _ρY_ 2; , _ρY_ 3 in (b)). 

## _4.3. 2D inviscid smooth interfaces advection problem with tangential velocity jump_ 

This section presents a two-dimensional interface advection problem with the tangential velocity varying in space [10] as a multi-dimensional test case. Here, we assume that the two species have different ratios of specific heats, _γ_ 1 = 1 _._ 4 and _γ_ 2 = 1 _._ 66, and molar weights, _M_ 1 = 28 _._ 0 and _M_ 2 = 4 _._ 0. The simulations are performed in a periodic region [0 _,_ 1] with 501[2] grid points, where sufficient points are located within the initial smooth interfaces. The initial conditions are given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-08.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-09.png)


where _xc_ = 0 _._ 5, _rc_ = 0 _._ 25, _(w_ 1 _, w_ 2 _, w v )_ = _(_ 0 _._ 6 _,_ 0 _._ 2 _,_ 0 _._ 5 _)_ , and _k_ = 20. The four-stage fourth-order Runge–Kutta method is used with the CFL number of 0.05 ( _�t_ = 10[−][4] ) for the time integration. Here, the proposed scheme (FC/PE (S)) is compared to the divergence form scheme (NPE). 

Fig. 8 shows the distributions of pressure and the slice of tangential velocity at _t_ = 5 _._ 0, when the interface advects through the domain five times. In the FC/PE (S) results, the pressure equilibrium is maintained, and the tangential velocity is advected correctly. When using NPE, spurious oscillations of pressure and velocity occur. These results indicate that the proposed scheme is applicable to flows with the tangential velocity jump. 

## **5. Difference between FC/PE (S) and FC/PE (D)** 

In the numerical experiments conducted in Section 4, only FC/PE (S) preserves the equilibrium for a long time. Although FC/PE (D) preserves the equilibrium at the beginning of the calculation as analyzed in Section 3, the computation diverges after a short time. In this section, therefore, we investigate the difference between FC/PE (S) and FC/PE (D). 

In Section 3, we discuss the equilibrium condition under the assumption of the constant velocity _u_ = _u_ 0 and pressure _p_ = _p_ 0. However, in the calculation, _u_ and _p_ have errors essentially due to round-off. The errors may affect the compatibility condition discussed with _u_ = _u_ 0 in Section 3 and trigger serious errors. Therefore, we reconsider the equilibrium condition with a nonuniform _u_ . We discretize the equations of _G_ and _ρ Y i_ with a nonuniform _u_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0015-15.png)


Substituting Eqs. (53) and (54) into Eq. (12) yields 

15 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-02.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-03.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-04.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-05.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-06.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-07.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-08.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-09.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-10.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-11.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-12.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0016-13.png)


**Fig. 7.** Total density and relative errors of velocity, _(u_ − _u_ exact _)/u_ exact, and pressure, _(p_ − _p_ exact _)/p_ exact, in the multi-dimensional multi-component inviscid smooth interfaces advection problem. 

16 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-02.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-03.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-04.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-05.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-06.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-07.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-08.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-09.png)


**Fig. 8.** Relative errors of pressure _(p_ − _p_ exact _)/p_ exact and the distributions of tangential velocity _v_ in the 2D inviscid smooth interface advection problem with tangential velocity at _t_ = 5 _._ 0 ( , exact). 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-11.png)


Equation (55) is the extended (more generalized) compatibility condition for a nonuniform velocity flow. Using Eq. (55), we verify the long-time pressure-equilibrium preserving characteristics of the proposed scheme (FC/PE (S)). 

Let us first consider FC/PE (D), which can preserve the equilibrium only at the initial state. Substituting the mass and internal energy fluxes of FC/PE (D) (see Table 1) into the left-hand-side (LHS) and RHS of Eq. (55) yields 

## FC/PE (D) : 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0017-15.png)


17 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0018-02.png)


with assuming the pressure is constant. In the analysis, _(_ LHS _)_ ̸= _(_ RHS _)_ , which indicates that FC/PE (D) does not satisfy the extended compatibility condition Eq. (55). Therefore, FC/PE (D) can preserve the equilibrium only when the velocity is constant. Once the velocity is disturbed even with a small amplitude, the scheme no longer preserves the pressure equilibrium and the computation diverges. 

A similar analysis verifies the long-time pressure-equilibrium preserving characteristics of FC/PE (S): 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0018-05.png)


Here, _(_ LHS _)_ = _(_ RHS _)_ is satisfied. In other words, the convective velocity of _G_ on the LHS and the one of _ρ Y i_ on the RHS are consistent regardless of the velocity fields. Therefore, FC/PE (S) can maintain the pressure equilibrium without developing errors even in disturbed velocity fields. Furthermore, FC/PE (S) is designed to reduce to the KEEPPE in a single-component constant _γ_ flow, which is stable as shown in Ref. [30]. Thus, we recommend FC/PE (S) as the proposed scheme. 

## **6. Conclusions** 

This study proposed a fully conservative and pressure-equilibrium preserving scheme for discontinuity-free compressible multi-component flows with an arbitrary number of species. The proposed scheme solves only the conservaton equations of the mass of species, momentum, and total energy in contrast to the conventional non-conservative or overspecified pressure-equilibrium preserving schemes that directly solve the transport equation of pressure or the equilibrium condition. The key to this study is the compatibility condition, which is the condition for implicitly satisfying the pressure-equilibrium condition at the discrete level. The compatibility condition is derived based on the analytical relation between the derivatives of the specific heats _G_ and mass of species _ρ Y i_ . The proposed scheme employs spatial discretization that implicitly satisfies the compatibility condition and does not require any special treatment such as switching fluxes at interfaces. 

We verified the proposed scheme through the numerical test of the inviscid smooth interfaces advection problem in the pressure and velocity equilibriums. The proposed scheme maintains both the conservation and pressure-equilibrium preservation, while the existing conservative scheme generates spurious oscillations. Also, the conventional pressure-equilibrium preserving but non-conservative schemes do not conserve the total energy or mass of species. Furthermore, we confirmed that the proposed scheme is straightforwardly extendable to multi-dimension by the dimension-by-dimension approach. According to a nonuniform-velocity analysis, FC/PE (S) can maintain the pressure equilibrium even when the velocity is disturbed in contrast to FC/PE (D). Even for discontinuity-free flows, the proposed scheme is the first to achieve both conservation and equilibrium. In conclusion, we recommend FC/PE (S) as the baseline scheme for compressible multi-component flow simulations. 

## **CRediT authorship contribution statement** 

**Yuji Fujiwara:** Conceptualization, Formal analysis, Investigation, Methodology, Software, Visualization, Writing – original draft, Writing – review & editing. **Yoshiharu Tamaki:** Conceptualization, Methodology, Writing – original draft, Writing – review & editing. **Soshi Kawai:** Conceptualization, Funding acquisition, Project administration, Resources, Supervision, Writing – review & editing. 

18 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

## **Declaration of competing interest** 

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Soshi Kawai reports financial support was provided by Japan Society for the Promotion of Science. 

## **Data availability** 

Data will be made available on request. 

## **Acknowledgements** 

This work was supported in part by Japan Society for the Promotion of Science (JSPS) KAKENHI Grant Number JP19K21927 and JP21H01522. 

## **Appendix A. Proposed scheme in** _**(ρ, ρu, E, ρY i)**_ **system** 

Here, we describe the derivation of the proposed scheme in the _(ρ, ρ_ _**u** , E, ρ Y i)_ system. The conservation equations for the total mass, momentum, total energy, and mass of species for _N_ -component compressible inviscid flows are given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0019-10.png)


where _i_ = 1 _,_ 2 _..., N_ − 1. 

## _A.1. Compatibility condition_ 

The mixing rule Eq. (4) is described in terms of _ρ_ and _ρ Y i_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0019-14.png)


From Eq. (A.2), _G_ = _G(ρ, ρ Y i, G i, Mi)_ = _G(ρ, ρ Y i)_ . Note that _Mi_ and _γi_ are constant and unique to each species. The time derivative of pressure is analytically derived similarly to the derivations in Section 2.3 for the _(ρ Y i, ρ_ _**u** , E)_ system. Since the internal energy is defined as _ρe_ = _ρe(p, ρ, ρ Y i)_ , the derivative of _ρe_ is analytically replaced by the derivatives of _p_ , _ρ_ and _ρ Y i_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0019-16.png)


From Eq. (A.3), the time derivative of pressure is described as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0019-18.png)


Equation (A.4) shows that the pressure equilibrium requires 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0019-20.png)


_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

Equation (A.5) is the pressure equilibrium condition for the _(ρ, ρ_ _**u** , E, ρ Y i)_ system. Although Eq. (A.5) holds at the analytical level, it is not guaranteed generally at the discrete level. Substituting the discretized form of Eq. (A.1) into Eq. (A.5), the convective terms are discretized in space as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0020-03.png)


where _α_ | _m_ ≡ � _∂∂ρρe_ � _p,ρY i_ ���� _m_ and _βi_ | _m_ ≡ � _∂∂ρρYei_ � _p,ρ,ρY j_ ̸= _i_ ���� _m_ . Equation (A.6) is the compatibility condition in the _(ρ, ρ_ _**u** , E, ρ Y i)_ system. 

## _A.2. Derivation of the proposed numerical fluxes_ 

Next, we derive the half-point values _ρe_ | _m_ + 12[,] _[ρ]_[|] _[m]_[+][1] 2[,][and] _[ρ][Y][i]_[|] _[m]_[+][1] 2[that][satisfy][the][compatibility][condition][(Eq.][(][A.6][)).] Considering the fluxes conservation at _m_ +[1] 2[,][we][transform][Eq.][(][A.6][)][similar][to][the][derivations][in][the][Section][3][for][the] _(ρ Y i, ρ_ _**u** , E)_ system assuming ideal gases, 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0020-07.png)


where _α_[′] | _m_ ≡ � _∂∂ρG_ � _ρY i_ ���� _m_ and _βi_[′][|] _[m]_[ ≡] � _∂∂ρGY i_ � _ρ,ρY j_ ̸= _i_ ���� _m_ . In addition, _G_ , _α_[′] , and _βi_[′][may][be][described][analytically][using] _[ρ]_[and] _ρ Y_ 1 by the mixing rule Eq. (4) as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0020-09.png)


where 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0020-11.png)


Substituting Eqs. (A.8)–(A.11) into Eq. (A.7) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0020-13.png)


In Eq. (A.12), only _ρ_ | _m_ + 12[and] _[ρ][Y][i]_[|] _[m]_[+][1] 2[are][the][unknowns][and][the][others][are][known.][Therefore,][once][one][of] _[ρ]_[|] _[m]_[+][1] 2[and] _ρ Y i_ | _m_ + 12[is][determined,][the][other][is][obtained][uniquely.] 

20 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

The proposed half-point values used for the numerical fluxes in the _(ρ, ρ_ _**u** , E, ρ Y i)_ system are derived as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0021-03.png)


In addition, _ρ Y N_ , which is not solved directly in this system, is written as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0021-05.png)


Equation (A.14) is the same form as _ρ Y i_ | _m_ + 12[in][the] _[(][ρ][Y][i][,][ρ]_ _**[u]**[,][E][)]_[system][in][Eq.][(][48][).][This][result][indicates][that][instead][of] solving the _(ρ Y i, ρ_ _**u** , E)_ system, we can also solve the _(ρ, ρ_ _**u** , E, ρ Y i)_ system using the proposed scheme. In conclusion, the same fluxes are derived in the _(ρ, ρ_ _**u** , E, ρ Y i)_ and _(ρ Y i, ρ_ _**u** , E)_ system, and they are equivalent at the discrete level. 

## **Appendix B. Equilibrium preserving scheme for the stiffened EoS** 

Here, we describe the derivation of the proposed scheme for the stiffened EoS, which is often used in the simulations for two-phase flows [10,32]. The definition of the stiffened EoS for _N_ -component compressible flows is given by 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0021-09.png)


where _π_ is a constant parameter characteristic of the material (e.g., _πi_ is an extremely high value in a liquid phase and zero in a gas phase). The stiffened EoS is reduced to the ideal gas EoS when _π_ is zero. 

## _B.1. Mixing rule_ 

The mixing rule for the stiffened EoS is described as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0021-13.png)


where _Mi_ , _G i_ and _Ai (_ = _γiπi/(γi_ − 1 _))_ are constant and unique to each species. _ρ Y i_ is the only variable that describes the fluids mixing under the pressure-equilibrium condition _(p_ = _p_ 0 _)_ since Eq. (B.2) indicates _ρe_ = _ρe(p, ρ Y i)_ . 

## _B.2. Derivation of the numerical fluxes from the compatibility condition_ 

The compatibility condition (21) in Section 3 holds for the Jacobian � _∂∂ρρYei_ � _p,ρY j_ ̸= _i_ of an arbitrary EoS. Therefore, the proposed method can be applied to stiffened EoS as well. From the mixing rule Eq. (B.2), the Jacobian is analytically calculated as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0021-17.png)


where 

21 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-02.png)


Substituting Eqs. (B.1) and (B.3) into Eq. (21) yields 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-04.png)


Here, we derive the compatible _ρ Y i_ | _m_ + 12[,] _[G]_[|] _[m]_[+][1] 2[and] _[A]_[|] _[m]_[+][1] 2[from][Eq.][(][B.6][).][For][simplicity,][Eq.][(][B.6][)][is][divided][into][the][parts] of _G_ and _A_ as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-06.png)


The compatibility condition for the ideal gas part (Eq. (B.7a)) is already discussed in Section 3. Therefore, we consider the half point values _A_ | _m_ + 12[and] _[ρ][Y][i]_[|] _[m]_[+][1] 2[satisfying][Eq.][(][B.7][b).][Similar][to][the][ideal][gas][case,][Eq.][(][B.7][b)][is][transformed][using] _[m]_ and _m_ + 1 point values and substituted by Eqs. (B.2) and (B.5). Consequently, the following equation is obtained: 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-08.png)


Since Eq. (B.8) is the same form as for the ideal gas case by just replacing _G i_ with _Ai_ , we can derive fluxes similarly. Here, the half-point values used for the numerical fluxes satisfying the equilibrium are derived as 


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-10.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-11.png)


![](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent.pdf-0022-12.png)


and the other fluxes are the same as in Table 1. In conclusion, based on the compatibility condition derived with an arbitrary EoS, we successfully extend the proposed scheme to the stiffened EoS. 

## **References** 

> [1] S. Karni, Multicomponent flow calculations by a consistent primitive algorithm, J. Comput. Phys. 112 (1994) 31–43. 

> [2] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160. [3] K. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, J. Comput. Phys. 142 (1998) 208–242. 

> [4] R. Abgrall, S. Karni, Computations of compressible multifluids, J. Comput. Phys. 169 (2001) 594–623. 

22 

_Y. Fujiwara, Y. Tamaki and S. Kawai_ 

_Journal of Computational Physics 478 (2023) 111973_ 

- [5] E. Johnsen, F. Ham, Preventing numerical errors generated by interface-capturing schemes in compressible multi-material flows, J. Comput. Phys. 231 (2006) 5705–5717. 

- [6] S. Kawai, H. Terashima, H. Negishi, A robust and accurate numerical method for transcritical turbulent flows at supercritical pressure with an arbitrary equation of state, J. Comput. Phys. 300 (2015) 116–135. 

- [7] H. Terashima, S. Kawai, N. Yamanishi, High-resolution numerical method for supercritical flows with large density variations, AIAA J. 49 (12) (2011) 2658–2672. 

- [8] H. Terashima, M. Koshi, Approach for simulating gas–liquid-like flows under supercritical pressures using a high-order central differencing scheme, J. Comput. Phys. 231 (2012) 6907–6923. 

- [9] J.J. Quirk, S. Karni, On the dynamics of a shock–bubble interaction, J. Fluid Mech. 318 (1996) 129–163. 

- [10] R. Saurel, R. Abgrall, A simple method for compressible multifluid flows, SIAM J. Sci. Comput. 21 (3) (1999) 1115–1145. 

- [11] E. Johnsen, Spurious oscillations and conservation errors in interface-capturing schemes, CTR Ann. Res. Briefs (2008) 115–126. 

- [12] R.P. Fedkiw, X.D. Liu, S. Osher, A general technique for eliminating spurious oscillations in conservative schemes for multiphase and multispecies Euler equations, Int. J. Nonlinear Sci. Numer. Simul. 3 (2002) 99–105. 

- [13] S. Karni, Hybrid multifluid algorithms, SIAM J. Sci. Comput. 17 (5) (1996) 1019–1039. 

- [14] B. Boyd, D. Jarrahbashi, A diffuse-interface method for reducing spurious pressure oscillations in multicomponent transcritical flow simulations, Comput. Fluids 222 (2021) 104924. 

- [15] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2006) 715–732. 

- [16] A.W. Cook, Enthalpy diffusion in multicomponent flows, Phys. Fluids 21 (5) (2009) 055109. 

- [17] S. Kawai, H. Terashima, A high-resolution scheme for compressible multicomponent flows with shock waves, Int. J. Numer. Methods Fluids 66 (2011) 1207–1225. 

- [18] H. Terashima, S. Kawai, M. Koshi, Consistent numerical diffusion term for simulating compressible multicomponent flows, Comput. Fluids 88 (2013) 484–495. 

- [19] S. Mirjalili, C.B. Ivey, A. Mani, A conservative diffuse interface method for two-phase flows with provable boundedness properties, J. Comput. Phys. 401 (2020) 109006. 

- [20] S.S. Jain, A. Mani, P. Moin, A conservative diffuse-interface method for compressible two-phase flows, J. Comput. Phys. 418 (2020) 109606. 

- [21] A. Jameson, Formulation of kinetic energy preserving conservative schemes for gas dynamics and direct numerical simulation of one-dimensional viscous compressible flow in a shock tube using entropy and kinetic energy preserving schemes, J. Sci. Comput. 34 (2) (2008) 188–208. 

- [22] S. Pirozzoli, Generalized conservative approximations of split convective derivative operators, J. Comput. Phys. 229 (19) (2010) 7180–7190. 

- [23] F. Ismail, P.L. Roe, Affordable, entropy-consistent Euler flux functions II: entropy production at shocks, J. Comput. Phys. 228 (2009) 5410–5436. 

- [24] P. Chandrashekar, Kinetic energy preserving and entropy stable finite volume schemes for compressible Euler and Navier–Stokes equations, Commun. Comput. Phys. 14 (5) (2013) 1252–1286. 

- [25] Y. Kuya, K. Totani, S. Kawai, Kinetic energy and entropy preserving schemes for compressible flows by split convective forms, J. Comput. Phys. 375 (2018) 823–853. 

- [26] Y. Kuya, S. Kawai, A stable and non-dissipative kinetic energy and entropy preserving (KEEP) scheme for non-conforming block boundaries on cartesian grids, Comput. Fluids 200 (2020) 104427. 

- [27] Y. Kuya, S. Kawai, High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids, J. Comput. Phys. (2021) 110482. 

- [28] Y. Kuya, S. Kawai, Modified wavenumber and aliasing errors of split convective forms for compressible flows, J. Comput. Phys. (2022) 111336. 

- [29] Y. Tamaki, K. Yuichi, S. Kawai, Comprehensive analysis of entropy conservation property of non-dissipative schemes for compressible flows: KEEP scheme redefined, J. Comput. Phys. 468 (2022) 111494. 

- [30] N. Shima, Y. Kuya, Y. Tamaki, S. Kawai, Preventing spurious pressure oscillations in split convective form discretization for compressible flows, J. Comput. Phys. 427 (2021) 110060. 

- [31] W. Mulder, S. Osher, J.A. Sethian, Computing interface motion in compressible gas dynamics, J. Comput. Phys. 100 (2) (1992) 209–228. 

- [32] S.A. Beig, E. Johnsen, Maintaining interface equilibrium conditions in compressible multiphase flows using interface capturing, J. Comput. Phys. 302 (2015) 548–566. 

23 

