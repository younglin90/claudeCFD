Journal of Computational Physics 478 (2023) 111973


### Contents lists available at ScienceDirect


### journal homepage: www.elsevier.com/locate/jcp


# Fully conservative and pressure-equilibrium preserving scheme for compressible multi-component flows


# Yuji Fujiwara, Yoshiharu Tamaki, Soshi Kawai ∗

Department of Aerospace Engineering, Tohoku University, 6-6-01 Aramaki-Aza-Aoba, Aobaku, Sendai, Miyagi 980-8579, Japan


## a r t i c l e i n f o a b s t r a c t

Article history: Received 16 June 2022 Received in revised form 12 December 2022 Accepted 22 January 2023 Available online 27 January 2023

Keywords: Multi-component flow Conservative scheme Pressure equilibrium

In compressible flows with a variable ratio of specific heats, such as multi-component flows, the conventional conservative schemes generate spurious pressure oscillations at multi-component interfaces even when the interfaces are smooth. In this study, we propose a novel spatial discretization scheme that maintains both the primary conservation and pressure equilibrium in discontinuity-free compressible multi-component flows. The key to this study is the compatibility condition, which is the discrete condition for implicitly satisfying the pressure equilibrium at the discrete level. The compatibility condition is derived based on the analytical relation between the mass of species and the ratio of specific heats in the governing equations. The proposed scheme is constructed by the spatial discretization that satisfies the compatibility condition and can be applied to the arbitrary number of species components. The inviscid smooth interfaces advection problems verify that the proposed scheme satisfies both the pressure equilibrium and primary conservation only by solving the conservation equations, unlike the existing schemes that solve non-conservative or overspecified equations.


### © 2023 Elsevier Inc. All rights reserved.


### 1. Introduction

Mixing of multiple fluid components with different properties is an important physical phenomenon in engineering applications, such as combustion and chemical reaction flows. One of the crucial physics in fluid mixing is the equilibrium (i.e., the temporal and spatial balance of the velocity, pressure, etc.) at the fluid interfaces. In inviscid flows without surface tension, since there is no diffusion, the physical velocity and pressure equilibriums must be maintained. However, in compressible flows with a non-constant ratio of specific heats, such as multi-component or supercritical fluid flows, the conventional conservative schemes which solve the conservation equations of the mass, momentum, total energy, and species generate spurious pressure oscillations at the interfaces [1–8]. Such oscillations result from numerical errors with the variation of the ratio of specific heats. In multi-component flows, since the ratio of specific heats depends on the composition of the mixture fluids, the pressure equilibrium cannot be maintained when a scheme does not represent the fluids mixing. Once the spurious pressure oscillations generate, they induce oscillations of other physical quantities such as the density and velocity through the mass and momentum equations. Note that such serious oscillations occur even when the specific heats vary smoothly in space, as will be discussed in this paper. Therefore, the conventional interface-capturing methods (i.e., numerically diffusing the interfaces) are not a fundamental solution to this oscillation problem. In addition, even in viscous


# * Corresponding author. E-mail address: kawai@tohoku.ac.jp (S. Kawai).

https://doi.org/10.1016/j.jcp.2023.111973 0021-9991/© 2023 Elsevier Inc. All rights reserved.

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### flow calculations, such oscillations may occur and often lead to the numerical instabilities [6]. Therefore, the equilibrium preserving scheme is essential for high-fidelity simulations.

Regarding the diffuse-interface methods, previous studies on compressible multi-component flows are dedicated to avoiding spurious oscillations at fluid interfaces. Karni et al. [1,9] proposed a discretization scheme based on the system of primitive variables evolution equations (i.e., density, velocity, and pressure equations) instead of the equations for the conservative quantities (i.e., density, momentum, and total energy equations). Abgrall [2] derived the analytical conditions for preserving the equilibrium in multi-component flows and proposed the equilibrium preserving scheme that directly solves the transport equation of the ratio of specific heats function instead of the conservation equation of the mass of species. Shyue [3] extended the method of Abgrall [2] to the mass fraction form. Saurel et al. [10] also extended the method to a spatially second-order scheme with a general equation of state (EoS). However, these methods do not solve the conservation equations of species. As pointed out by Johnsen [11], the difficulty of these schemes is that the conservation errors of the mass of species increase with time. The conservation errors for each species can be a critical issue in mixing and combustion phenomena. Therefore, to achieve the conservation of primary conservative variables (i.e., mass, momentum, and total energy), the double-flux methods are proposed. The double-flux methods solve both the pressure evolution and total-energy conservation equations [12,13], or both the mass conservation equation and the transport equations of the ratio of specific heats function [4,5]. In these schemes, one extra transport equation is solved in addition to the conservation equations. Although these methods satisfy both the equilibrium and primary conservation, the double-flux methods overspecify the equations for the system. Therefore, two independent solutions of pressure are obtained from the pressure and total-energy equations (similarly, for the mass of species in Ref. [5]). Analytically, the solution of the pressure or mass of species is unique. In addition, the hybrid method [14] that switches between quasi-conservative double-flux and the fullyconservative method has also been proposed, although it does not fundamentally resolve the overspecification problem. Note that several studies have also presented extensions of the methods to higher-order schemes (e.g., the WENO scheme [15] and the compact difference scheme [16]) and the applications to the localized artificial diffusivity (LAD) methods [17,18], supercritical fluids [6–8] and the phase-field methods that contain terms for diffusing and sharping interfaces for compressible multi-phase flows [19,20]. However, to the authors’ knowledge, none of the existing schemes maintains the pressure equilibrium by solving only the conservation equations of the mass of each species, momentum, and total energy.

To avoid spurious oscillations and overspecified solutions, spatial discretization based on the analytical relations between the governing equations may be one of the solutions. Several studies in recent years have focused on the spatial discretization approach, which implicitly satisfies the equations of secondary quantities (e.g., kinetic energy, entropy, and pressure not directly solved in the system). For example, several researchers proposed kinetic energy preserving (KEP), e.g., [21,22], entropy conserving (EC), e.g., [23,24], and kinetic energy and entropy preserving (KEEP) schemes [25–29]. In these schemes, the analytical relation of the secondary quantities is also satisfied at the discrete level. In addition, Shima et al. [30] proposed a modified KEEP scheme that preserves the pressure equilibrium in single-component flows. These studies demonstrated that a well-designed discretization may implicitly satisfy a certain analytical relation without solving additional equations. This idea may be a key to an equilibrium-preserving approach in multi-component flows.

In this paper, we propose a fully conservative and pressure-equilibrium preserving scheme for compressible multicomponent flows with smooth interfaces. Here, the terminology “fully conservative” means solving the same number of conservation equations (i.e., the mass of each species, momentum, and total energy) as the unknowns without any extra transport equation. The idea is to implicitly satisfy the equilibrium condition [2,4] at the discrete level. We focus on the mixing rule of specific heats and design the spatial discretization based on the “compatibility condition” that describes the analytical relations between the governing equations and the equilibrium condition. Unlike the previous studies, the proposed scheme maintains both conservation and equilibrium without overspecifying equations. As shown later, the proposed scheme can be applied to an arbitrary number of species components. Note that this paper focuses on the discretization of the inviscid terms in compressible multi-component flows without sharp fluid interfaces or shocks. To our knowledge, even in the discontinuity-free smooth interface conditions considered in this study, no existing scheme simultaneously satisfies the conservation and pressure-equilibrium preservation. Therefore, the proposed scheme will be a baseline scheme to simulate compressible multi-component flows with sharp interfaces or shocks by properly designing and adding numerical diffusion terms to the proposed scheme to capture the numerical discontinuities.

The structure of this paper is as follows. In Section 2, we show the mathematical model for multi-component compressible inviscid flows and introduce the analytical equilibrium condition for the pressure and velocity. Based on the derivation of the analytical condition, the causes of spurious pressure errors in the conventional conservative methods are discussed. In Section 3, a method that satisfies both the conservation and equilibrium is proposed. Here, we discuss the discrete condition to satisfy the equilibrium condition implicitly. For simplicity, we first derive the fully conservative and pressure-equilibrium preserving scheme from the compatibility condition in two-component flows and then extend it to N-component flows. The inviscid flow problems are conducted to demonstrate the capability of the proposed scheme in Section 4. In Section 5, we discuss the behavior of the proposed schemes in terms of velocity disturbances. Finally, in Section 6, conclusions are drawn.

2

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### 2. Mathematical models for compressible multi-component flows


### 2.1. Governing equations


### The conservation equations for the mass of species, momentum, and total energy for N-component compressible inviscid flows are given by


## ⎧ ⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎩


# ∂ρYi


## ∂t + ∇· (ρYiu) = 0, (a)


# ∂ρu


## ∂t + ∇· (ρu ⊗u + pδ) = 0, (b)


## ∂E


## ∂t + ∇· ((E + p)u) = 0, (c)


## (1)

where ρ is the density, u is the velocity vector, p is the static pressure, E = ρ(e + u · u/2) is the total energy, e is the internal energy per unit mass, Yi is the mass fraction of species i (i = 1, 2..., N), and δ is the unit tensor. In the multicomponent flow system, the mass conservation of each species must be satisfied. Note that the sum of Eq. (1a) for each species yields the mass conservation equation


# ∂ρ


# ∂t + ∇· (ρu) = 0, (2)

because �N i Yi = 1 and �N i ρYi = ρ. Thus, there is no need to solve Eq. (2) directly when we solve Eq. (1a) up to the Nth species. Note that some studies [5,17,31] solve the (ρ, ρu, E, ρYi) system that includes Eq. (2) and Eq. (1a) up to the N −1th species. They are analytically equivalent to solving the (ρYi, ρu, E) system. Therefore, although Eq. (1) is used in this study, the following discussion is valid also for the (ρ, ρu, E, ρYi) system (see Appendix A).


### We assume that the species are calorically perfect gases in thermal equilibrium. The equation of state (EoS) for ideal gases is


# p = ρ Ru


## M T = ρe(γ −1), (3)

where Ru is the universal gas constant, T is the temperature, M is the molar weight for the fluid mixture, and γ is the ratio of specific heats for the fluid mixture. In this paper, the variables with subscript i denote the quantities of each species, and those without any subscript denote the quantities of the fluid mixture.


### 2.2. Mixing rule

One of the difficulties of the multi-component flow systems is that M and γ are not constant in time and space. The mass fraction Yi (= ρi/ρ = ρYi/ρ) is used to describe the mixing. From the thermodynamic relationships, the ratio of specific heats for the mixture γ is defined as


# γ ≡cp


## cv = �N i cp,iYi �N i cv,iYi ,


### where cp is the specific heat per mass at constant pressure, and cv is the specific heat per mass at constant volume. These specific heats are calculated using the ideal gases EoS as


## cp,i = γi γi −1


## Ru Mi , cv,i = 1 γi −1


## Ru Mi .


# In prior studies [4,5,18], the function G ≡(γ −1)−1 is used to describe fluid mixing. Using the molar weight for the mixture M = (�


## i Yi/Mi)−1, the function G can be described as


## G ≡ 1 γ −1 = M

N �

i


## 1 γi −1


## Yi Mi . (4)

Note that Mi and γi are constant and unique to each species, while M and γ for the mixture are variable in terms of the mass fraction Yi. Eq. (4) is the mixing rule of G for multi-component flows. Although the mixing rule for ideal gases is shown in this section, the mixing rule for other EoS may be defined similarly (see Appendix B for the case of the stiffened EoS).

3

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### 2.3. Pressure and velocity equilibriums at fluid interfaces

This section summarizes the analytical derivations of the pressure and velocity equilibriums as the physical phenomena in inviscid flows. First, we derive the equilibrium conditions for an arbitrary EoS by introducing the Jacobian. Then, we show that the extended equilibrium condition encompasses the conventional equilibrium conditions proposed by Abgrall et al. [4] that assume ideal gases. For simplicity, the one-dimensional equations where the initial pressure p0 and velocity u0 are at the equilibrium condition, i.e., spatially constant as in prior studies [3,4,18], are considered. At this condition, the time evolution of ρYi, ρu, E at the initial state are written as


## ⎧ ⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎩


# ∂ρYi


## ∂t + u0 ∂ρYi


## ∂x = 0, (a)


# ∂ρu


## ∂t + u0u0 ∂ρ


## ∂x = 0, (b)


## ∂E


## ∂t + u0 ∂ρe


## ∂x + u0 u2 0 2 ∂ρ


## ∂x = 0. (c)


## (5)


### Taking the sum of Eq. (5a) for each species yields the mass-evolution equation


# ∂ρ


## ∂t + u0 ∂ρ


## ∂x = 0. (6)


# Here, for an arbitrary EoS ρe = ρe(p, ρYi), the derivative of ρe is analytically replaced by the derivatives of p and ρYi as


# ∂ρe


## ∂t = �∂ρe ∂p


## �


### ρYi


## ∂p


## ∂t +

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t . (7)


### By introducing the Jacobian � ∂ρe


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq001.png)


### ρYi and � ∂ρe ∂ρYi


## �


### p,ρY j̸=i , Eq. (7) clarifies the relationships between variables in the governing


### equations. These transformations of the substitution of derivatives in Eq. (7) are analytically exact.

The initial constant velocity and pressure distributions must maintain physically in time in inviscid flows without surface tension. The time derivative of velocity and pressure are derived analytically from Eqs. (5), (6), and (7) as


## ∂u


## ∂t = ∂ �ρu ρ �


## ∂t


## = 1


# ρ ∂ρu


## ∂t −u


# ρ ∂ρ


## ∂t , (8)


## ∂p


## ∂t = �∂ρe ∂p


## �−1


### ρYi


## � ∂ρe


## ∂t −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t


## �


## = �∂ρe ∂p


## �−1


### ρYi


## � ∂ ∂t


## �


# E −(ρu)2


# 2ρ


## � −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t


## �


## = �∂ρe ∂p


## �−1


### ρYi


## � ∂E


# ∂t −u ∂ρu


## ∂t + u2


## 2 ∂ρ


## ∂t −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t


## �


## . (9)


### These transformations are analytically exact. In the flow with constant initial pressure p0 and velocity u0, substituting Eqs. (5) and (6) into Eqs. (8) and (9) yields


## ∂u


## ∂t = −u0


# ρ


# �∂ρ ∂t + u0 ∂ρ


## ∂x


## � = 0, (10)


## ∂p


## ∂t = �∂ρe ∂p


## �−1


### ρYi


## ��


## −u0 ∂ρe


## ∂x −u0 u2 0 2 ∂ρ


## ∂x + u3 0 ∂ρ


## ∂x −u0 u2 0 2 ∂ρ


## ∂x


## �


## + u0

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂x


## �


## = −u0


# �∂ρe ∂p


## �−1


### ρYi


## � ∂ρe


## ∂x −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂x


## �


## . (11)

If the time derivatives of velocity and pressure are zero, the equilibriums are maintained. The right hand side (RHS) of Eq. (10) contains the transport equation of ρ. The transport equation of the density ρ is automatically satisfied by solving

4

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### Eq. (5a) in compressible flow simulations as Eq. (6). To maintain the pressure equilibrium, Eq. (9) requires the following relationship between ρe and ρYi,


# ∂ρe


## ∂x =

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂x . (12)


# Equation (12) is the substitution derivative of ρe and ρYi in space and the analytical relationship clearly holds in pressure equilibrium (p = p0). Since Eq. (12) is the key to derive ∂p


## ∂t = 0 in Eq. (11), the spurious pressure oscillations may be generated when Eq. (12) is not discretely satisfied. Therefore, Eq. (12) is the equilibrium condition that is valid for an arbitrary EoS.


# Note that assuming ideal gases (ρe = p/(γ −1) = pG), the time evolution of the pressure Eq. (11) is represented using Eq. (12) as


## ∂p


## ∂t = 1


## G


## � ∂E


# ∂t −u ∂ρu


## ∂t + u2


## 2 ∂ρ


## ∂t −

N �

i


## �∂pG ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t


## �


## = 1


## G


## �


## −u0 ∂ρe


## ∂x −p0

N �

i


## �∂G ∂ρYi


## �


### ρY j̸=i


# ∂ρYi


## ∂t


## �


## = −p0


## G


## �


## u0 ∂G


## ∂x +

N �

i


## �∂G ∂ρYi


## �


### ρY j̸=i


# ∂ρYi


## ∂t


## �


## = −p0


## G


## �∂G ∂t + u0 ∂G


## ∂x


## � , (13)


### where


## ∂G


## ∂t =

N �

i


## �∂G ∂ρYi


## �


### ρY j̸=i


# ∂ρYi


## ∂t , (14)

which is analytically correct because G = G(Yi) = G(ρYi) from the mixing rule (Eq. (4)). The RHS of Eq. (13) contains the transport equation of G. To satisfy the equilibrium for ideal-gas flows, Eq. (13) requires


## ∂G


## ∂t + u0 ∂G


## ∂x = 0. (15)

Equation (15) is the conventional equilibrium condition as suggested by Abgrall et al. [4]. Some of the conventional equilibrium preserving schemes directly solve Eq. (15) as one of the governing equations. Substituting Eqs. (5a) and (15) into Eq. (14) yields


## ∂G


## ∂x =

N �

i


## �∂G ∂ρYi


## �


### ρY j̸=i


# ∂ρYi


## ∂x , (16)


### which is the analog to Eq. (12) for the ideal gas EoS.


### 3. Equilibrium preserving scheme

In order to maintain the pressure equilibrium at the interfaces of multi-component fluids, the pressure equilibrium condition Eq. (12) must be satisfied at the discrete level. A simple approach in the ideal gases case is to solve Eq. (15) directly in addition to the governing equations (1) or by replacing the mass of species equation (1a), as in Refs. [4,5]. However, this approach is either overspecified (since the system is already closed by the governing equation (1), EoS (3), and the mixing rule (4)) or non-conservative for the mass of species. In other words, to satisfy the fully conservative and pressure-equilibrium characteristics without overspecification, the pressure equilibrium condition must be satisfied by solving only the conservation equations in Eq. (1). In this study, we propose a spatial discretization that implicitly satisfies the pressure equilibrium condition by solving the equations of primary conservative variables (ρYi, ρu, E) in Eq. (1).

To find such a discretization, we derive the discrete condition implicitly satisfying the pressure equilibrium, which represents the transport of ρe as the transport of ρYi at the discrete level. For general discussion, first, we will proceed with an arbitrary EoS and then assume the ideal gas flow to determine the detailed discretization. In this study, we treat the time derivative analytically and focus on the spatial discretization.

5

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### 3.1. Spacial discretization

In this study, we focus on the spatial discretization and propose a fully conservative and pressure-equilibrium preserving scheme. Although the following equations are described in the one-dimension forms for simplicity, the scheme may be extended straightforwardly to multi-dimension by applying the numerical fluxes dimension-by-dimension as demonstrated in Sections 4.2 and 4.3. In conservative schemes, the spatial derivative terms must be discretized in a numerical flux form. The governing equations (1) are spatially discretized by the half-point numerical fluxes as


## ⎧ ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩


# ∂ρYi


## ∂t + � Ci|m+ 1


## 2 −� Ci|m−1

2 �x = 0, (a)


# ∂ρu


## ∂t + � Mu|m+ 1


## 2 −� Mu|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq002.png)

2 �x = 0, (b)


## ∂E


## ∂t + � K|m+ 1


## 2 −� K|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq003.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq004.png)


## 2 −� P|m−1

2 �x = 0, (c)


## (17)


### where


## � Ci ≡ρYiu, � Mu ≡ρuu, ��≡p,


## � K ≡ρ uu


## 2 u, �I ≡ρeu, � P ≡up,


### and m is the cell index. A fully conservative scheme is constructed by solving these equations without any extra transport equations. Since the form of the numerical fluxes at m ± 1


### 2 point is arbitrary, we propose the numerical fluxes that preserve the pressure equilibrium based on the analytical relations between the governing equations.


### 3.2. Compatibility condition for spatial discretization


### Here, we investigate the pressure equilibrium condition (12) at the discrete level. In the flow with constant p0 and u0, substituting Eq. (17) into Eqs. (8) and (9) yields


## ∂u


## ∂t = 1


# ρ ∂ρu


## ∂t −u


# ρ ∂ρ


## ∂t


## = −1


# ρ


## �� Mu|m+ 1


## 2 −� Mu|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq005.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq006.png)


## 2 −� C|m−1

2 �x


## �


## = − u2 0 ρ


# �ρ|m+ 1 2 −ρ|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq007.png)


# 2 −ρ|m−1

2 �x


## �


## = 0, (18)


## ∂p


## ∂t = �∂ρe ∂p


## �−1


### ρYi


## ��∂E ∂t −u ∂ρu


## ∂t + u2


## 2 ∂ρ


## ∂t


## � −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


# ∂ρYi


## ∂t


## �


## = − �∂ρe ∂p


## �−1


### ρYi


## ��� K|m+ 1


## 2 −� K|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq008.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq009.png)


## 2 −� P|m−1

2 �x


## −u � Mu|m+ 1


## 2 −� Mu|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq010.png)

2 �x + u2


## 2


## � C|m+ 1


## 2 −� C|m−1

2 �x


## �


## −

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


## ����� m


## � Ci|m+ 1


## 2 −� Ci|m−1

2 �x


## �


## = −u0


# �∂ρe ∂p


## �−1


### ρYi


# �ρe|m+ 1 2 −ρe|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq011.png)

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


## ����� m


# ρYi|m+ 1


# 2 −ρYi|m−1

2 �x


## �


## . (19)


### Equation (18) shows that the velocity equilibrium holds at the discrete level, while the pressure equilibrium requires


# ρe|m+ 1


# 2 −ρe|m−1

2 �x =

N �

i


# �∂ρe ∂ρYi


## �


### p,ρY j̸=i


## ����� m


# ρYi|m+ 1


# 2 −ρYi|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq012.png)

6

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


## Equation (20) is the discrete analog to Eq. (12). Moreover, multiplying �x to Eq. (20) yields


# ρe|m+ 1


# 2 −ρe|m−1


## 2 =

N �

i


## � βi|m � ρYi|m+ 1


# 2 −ρYi|m−1

2


## �� , (21)


## where βi|m ≡ � ∂ρe ∂ρYi


## �


### p,ρY j̸=i


## ���� m . This equation is satisfied analytically but is not guaranteed generally at the discrete level.

Therefore, our idea here is to replace the derivatives of ρe with the derivatives of ρYi at the discrete level to satisfy Eq. (21). The advantage of this approach is that the proposed method avoids overspecification while maintaining the pressure equilibrium. Therefore, by proposing the spatial discretizations that satisfy Eq. (21), we will achieve both the conservation of ρYi, ρu, E and pressure-equilibrium preservation in the governing equations (1) without solving any extra equation. Equation (21) represents the condition of the numerical fluxes, which means that the transports of ρYi are compatible with that of ρe. Therefore, we call Eq. (21) the compatibility condition. The numerical fluxes based on the compatibility condition implicitly satisfy Eq. (11) at the discrete level, and as a result, the pressure equilibrium is achieved.


### 3.3. Compatible half-point values maintaining equilibriums


# In this section, we derive the half-point values ρYi|m± 1


# 2 and ρe|m± 1


### 2 that are used for the numerical fluxes and satisfy


## the compatibility condition (Eq. (21)). To derive the conservative numerical fluxes, the half-point values at m + 1


### 2 must be equivalent when seeing from the left point m and the right point m +1. We first derive the modified compatibility condition at m + 1


# 2 from Eq. (21). Then, we propose the half-point values ρYi|m+ 1


# 2 and ρe|m+ 1


### 2 to maintain the pressure equilibrium.


### 3.3.1. Two-component flows


## For simplicity, we first consider a two-components flow (N = 2). First, the modified compatibility condition at m + 1


### 2 is derived from Eq. (21). Equation (21) is decomposed into two equations at m as


# ρe|m+ 1


# 2 −ρe|m = β1|m � ρY1|m+ 1


# 2 −ρY1|m � + β2|m � ρY2|m+ 1


# 2 −ρY2|m � , (22)


# ρe|m −ρe|m−1


## 2 = β1|m � ρY1|m −ρY1|m−1

2


## � + β2|m � ρY2|m −ρY2|m−1

2


## � . (23)


### These equations are the sufficient conditions to satisfy Eq. (21). Then, Eq. (23) is shifted by one in m direction as


# ρe|m+1 −ρe|m+ 1


## 2 = β1|m+1 � ρY1|m+1 −ρY1|m+ 1

2


## � + β2|m+1 � ρY2|m+1 −ρY2|m+ 1

2


## � . (24)


# Considering the local conservation, ρe|m+ 1


# 2 in Eqs. (22) and (24) must be the same. Hence, eliminating ρe|m+ 1


### 2 from


### Eqs. (22) and (24) yields


# ρe|m+1 −ρe|m = (β1|m −β1|m+1)ρY1|m+ 1


# 2 + (β2|m −β2|m+1)ρY2|m+ 1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq013.png)

The discussion of the compatibility condition holds for the Jacobian of an arbitrary EoS until Eq. (25). By calculating the Jacobian from the EoS and substituting it into Eq. (25), the numerical fluxes that satisfy the compatibility condition are obtained. In this paper, although the derivations will be discussed under the assumption of ideal gases in the following, the derivations for EoS other than the ideal gases, the stiffened EoS often used for multi-phase flow simulations, are presented as an example in Appendix B. The derivation of the numerical fluxes for more complex EoS will be future work.


# Assuming an ideal gas (ρe = p/(γ −1) = pG), ρe and βi are described analytically using ρY1 and ρY2 by the mixing rule of G (Eq. (4)) as


# ρe|m = p0G|m = p0M|m

2 �

i


## Gi Mi Yi|m = p0


## M|m


# ρ|m


## �G1 M1 ρY1|m + G2


# M2 ρY2|m


## � , (26)


## β1|m = p0


## �∂G ∂ρY1


## �


### ρY2


## ����� m = p0


## M|2 m ρ|2 m


## �G1 −G2 M1M2 ρY2|m


## � , (27)


## β2|m = p0


## �∂G ∂ρY2


## �


### ρY1


## ����� m = p0


## M|2 m ρ|2 m


## �G2 −G1 M1M2 ρY1|m


## � , (28)


### where the following relationship is employed:

7

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


## M|m =


## �2 �

i


## 1


## Mi Yi|m


## �−1


# = ρ|m


# �ρY1|m M1 + ρY2|m


## M2


## �−1 . (29)


# Since Eqs. (26)–(28) contain p0, Eq. (25) is described in terms of G and ρYi by dividing p0 as


## G|m+1 −G|m = � β′ 1|m −β′ 1|m+1 �ρY1|m+ 1 2 + � β′ 2|m −β′ 2|m+1 �ρY2|m+ 1 2 + β′ 1|m+1ρY1|m+1 + β′ 2|m+1ρY2|m+1 −β′ 1|mρY1|m −β′ 2|mρY2|m, (30)


## where β′ i ≡ � ∂G ∂ρYi


## �


### ρY j̸=i = βi/p0. Substituting Eqs. (26)–(29), each term in Eq. (30) is calculated as


## G|m+1 −G|m = M|m+1


# ρ|m+1


## �G1 M1 ρY1|m+1 + G2


# M2 ρY2|m+1


## � −M|m


# ρ|m


## �G1 M1 ρY1|m −G2


# M2 ρY2|m


## �


## = M|m+1


# ρ|m+1


## M|m


# ρ|m


# ��ρY1|m M1 + ρY2|m


## M2


## ��G1 M1 ρY1|m+1 + G2


# M2 ρY2|m+1


## �


## − �ρY1|m+1 M1 + ρY2|m+1


## M2


## ��G1 M1 ρY1|m + G2


# M2 ρY2|m


## ��


## = M|m+1


# ρ|m+1


## M|m


# ρ|m


## G1 −G2


## M1M2 (ρY1|m+1ρY2|m −ρY1|mρY2|m+1), (31)


## � β′ 1|m −β′ 1|m+1 �ρY1|m+ 1 2 =


## � M|2 m ρ|2 m


## �G1 −G2 M1M2 ρY2|m


## � −


## M|2 m+1 ρ|2 m+1


## �G1 −G2 M1M2 ρY2|m+1


## ��


# ρY1|m+ 1

2


## = G1 −G2


## M1M2


## � M|2 m ρ|2 m ρY2|m −


## M|2 m+1 ρ|2 m+1 ρY2|m+1


## �


# ρY1|m+ 1


## 2 , (32)


## β′ 1|mρY1|m + β′ 2|mρY2|m = M|2 m ρ|2 m


## �G1 −G2 M1M2 ρY2|m


## � ρY1|m + M|2 m ρ|2 m


## �G2 −G1 M1M2 ρY1|m


## � ρY2|m = 0. (33)


### Substituting Eqs. (31)–(33) into Eq. (25) yields


# ρY1|m+1ρY2|m −ρY1|mρY2|m+1 =


## � M|m


# ρ|m ρ|m+1


## M|m+1 ρY2|m −M|m+1


# ρ|m+1 ρ|m


## M|m ρY2|m+1


## �


# ρY1|m+ 1

2


## −


## � M|m


# ρ|m ρ|m+1


## M|m+1 ρY1|m −M|m+1


# ρ|m+1 ρ|m


## M|m ρY1|m+1


## �


# ρY2|m+ 1


## 2 . (34)


# Equation (34) is the sufficient condition for the half-point values ρY1|m+ 1


# 2 and ρY2|m+ 1


### 2 to realize the pressure equilibrium


### at the discrete level.


### 3.3.2. Derivation of the half-point values


# The half-point values ρY1|m+ 1


# 2 and ρY2|m+ 1


# 2 are arbitrary, and therefore, we propose simple forms of ρY1|m+ 1


### 2 and ρY2|m+ 1


## 2 that satisfy Eq. (34). To simplify the representation, Eq. (34) is rewritten by introducing coefficients φ± as


# ρY1|m+1ρY2|m −ρY1|mρY2|m+1 = � φ−ρY2|m −φ+ρY2|m+1 �ρY1|m+ 1 2 − � φ−ρY1|m −φ+ρY1|m+1 �ρY2|m+ 1 2 ,


## (35)


### where


## φ+ ≡M|m+1


# ρ|m+1 ρ|m


## M|m , φ−≡M|m


# ρ|m ρ|m+1


## M|m+1 , φ+φ−= 1.


# Here, ρY1|m+ 1


# 2 and ρY2|m+ 1


### 2 are described as


## ⎧ ⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎩


# ρY1|m+ 1


## 2 = (ρY1|m+1ρY2|m −ρY1|mρY2|m+1) + � φ−ρY1|m −φ+ρY1|m+1 �ρY2|m+ 1 2 � φ−ρY2|m −φ+ρY2|m+1 � , (a)


# ρY2|m+ 1


## 2 = −(ρY1|m+1ρY2|m −ρY1|mρY2|m+1) + � φ−ρY2|m −φ+ρY2|m+1 �ρY1|m+ 1 2 � φ−ρY1|m −φ+ρY1|m+1 � . (b)


## (36)

8

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


# Equation (36) shows that once either ρY1|m+ 1


# 2 or ρY2|m+ 1


### 2 is determined, the other is obtained uniquely. While Eq. (36)

holds for any ρY1 and ρY2, they become discontinuous when each denominator is zero. Therefore, we derive the half-point value that is available continuously by considering the discontinuous limit. In Eq. (36b), ρY2|m+ 1


### 2 is discontinuous when the


# denominator become zero (i.e., φ−ρY1|m = φ+ρY1|m+1), and then Eq. (36a) asymptotically approaches the following value


# ρY1|m+ 1


# 2 ∼(ρY1|m+1ρY2|m −ρY1|mρY2|m+1) � φ−ρY2|m −φ+ρY2|m+1 �


# = ρY1|m (ρY1|m+1ρY2|m −ρY1|mρY2|m+1)


# φ+ (ρY1|m+1ρY2|m −ρY1|mρY2|m+1)


# = φ−ρY1|m = φ+ρY1|m+1. (37)


# Under this discontinuity condition, the half-point value ρY1|m+ 1


### 2 should be automatically reduced to Eq. (37), and a similar


# argument can be made for ρY2|m+ 1


### 2 as well.


### We propose the half-point values in the numerical fluxes that satisfy Eq. (34) as


## ⎧ ⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎩


# ρY1|m+ 1


# 2 = φ−ρY1|m + φ+ρY1|m+1


## 2 ,


# ρY2|m+ 1


# 2 = φ−ρY2|m + φ+ρY2|m+1


## 2 . (38)

Equation (38) are the central symmetric forms that satisfy the compatibility condition Eq. (34). In this discretization, the left numerical flux on m and the right flux on m + 1 are equivalent, which implies that these numerical fluxes satisfy the local conservation. The derived ρY1|m+ 1


# 2 and ρY2|m+ 1


## 2 are used for the numerical flux � Ci in Eq. (17a). Using these discretizations


## in Eq. (38), G|m+ 1


### 2 can be obtained from Eq. (22) as


## G|m+ 1


## 2 = G|m + β′ 1|m � ρY1|m+ 1


# 2 −ρY1|m � + β′ 2|m � ρY2|m+ 1


# 2 −ρY2|m �


## = G|m + M|2 m ρ|2 m


## G1 −G2


## M1M2


## � ρY2|mρY1|m+ 1


# 2 −ρY1|mρY2|m+ 1

2


## �


## = G|m + 1


## 2


## M|2 m ρ|2 m


## G1 −G2


## M1M2


## � M|m+1


# ρ|m+1 ρ|m


## M|m ρY2|mρY1|m+1 −M|m+1


# ρ|m+1 ρ|m


## M|m ρY1|mρY2|m+1


## �


## = G|m + 1


## 2


## M|m+1


# ρ|m+1


## M|m


# ρ|m


## G1 −G2


## M1M2 (ρY1|m+1ρY2|m −ρY1|mρY2|m+1)


## = G|m + G|m+1 −G|m


## 2


## = G|m + G|m+1


## 2 . (39)


## The derived G|m+ 1


# 2 is used as a part of ρe|m+ 1


## 2 to calculate �I in Eq. (17c). Equations (38) and (39) satisfy the compatibility

condition Eq. (21) so that by using Eqs. (38) and (39) to compute the numerical fluxes, the equilibrium condition is implicitly satisfied at the discrete level by only solving the conservation equations (1). Therefore, the proposed scheme may maintain both the conservation and pressure equilibrium.


### 3.3.3. Extension to N-component flows


### Next, we extend the proposed scheme to flows with an arbitrary number N of components. The compatibility condition in the N-component system may be derived similarly to Eq. (30) as


## G|m+1 −G|m =

N �

i


## �� β′ i|m −β′ i|m+1 �ρYi|m+ 1 2 + β′ i|m+1ρYi|m+1 −β′ i|mρYi|m � , (40)


### where


## G|m = M|m


# ρ|m

N �

i


## Gi Mi ρYi|m, (41)

9

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


## β′ i|m = �∂G ∂ρYi


## �


### ρY j̸=i


## ����� m = M|2 m ρ|2 m

N �

j


## �Gi −G j MiM j ρY j|m


## � , (42)


# M|m = ρ|m


## �N �

i


# ρYi|m


## Mi


## �−1


## . (43)


### From Eqs. (41)–(43), each term in Eq. (40) is calculated as


## G|m+1 −G|m = M|m+1


# ρ|m+1

N �

i


## Gi Mi ρYi|m+1 −M|m


# ρ|m

N �

i


## Gi Mi ρYi|m


## = M|m+1


# ρ|m+1


## M|m


# ρ|m

N �

i


# �ρ|m


## M|m


## Gi Mi ρYi|m+1 −ρ|m+1


## M|m+1


## Gi Mi ρYi|m


## �


## = M|m+1


# ρ|m+1


## M|m


# ρ|m

N �

i

N �

j


## � Gi MiM j


# �ρYi|m+1ρY j|m −ρYi|mρY j|m+1 ��


## = 1


## 2


## M|m+1


# ρ|m+1


## M|m


# ρ|m

N �

i

N �

j


## �Gi −G j MiM j


# �ρYi|m+1ρY j|m −ρYi|mρY j|m+1 �� , (44)

N �

i


## �� β′ i|m −β′ i|m+1 �ρYi|m+ 1 2


## � =

N �

i


## ⎡


## ⎣


## ⎛


## ⎝M|2 m ρ|2 m

N �

j


## �Gi −G j MiM j ρY j|m


## � −


## M|2 m+1 ρ|2 m+1

N �

j


## �Gi −G j MiM j ρY j|m+1


## �⎞


# ⎠ρYi|m+ 1

2


## ⎤


## ⎦


## =

N �

i

N �

j


## � Gi −G j


## MiM j


## � M|2 m ρ|2 m ρY j|m −


## M|2 m+1 ρ|2 m+1 ρY j|m+1


## �


# ρYi|m+ 1

2


## �


## , (45)

N �

i


## � β′ i|mρYi|m � =

N �

i


## ⎡


## ⎣M|2 m ρ|2 m

N �

j


## �Gi −G j MiM j ρY j|m


## � ρYi|m


## ⎤


## ⎦= M|2 m ρ|2 m

N �

i

N �

j


## �Gi −G j MiM j ρYi|mρY j|m


## � = 0. (46)


### Substituting Eqs. (44)–(46) into Eq. (40) yields


## 1


## 2

N �

i

N �

j


## �Gi −G j MiM j


# �ρYi|m+1ρY j|m −ρYi|mρY j|m+1 �� =

N �

i

N �

j


## �Gi −G j MiM j


## � φ−ρY j|m −φ+ρY j|m+1 �ρYi|m+ 1 2


## � .


## (47)


## Equation (47) is an identical equation for each i, j component. Because each i, j component equation is consistent with the two-component case, we can derive the numerical flux similarly.


### The half-point values that satisfy Eq. (47) are


## ⎧ ⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎩


# ρYi|m+ 1


# 2 = φ−ρYi|m + φ+ρYi|m+1


## 2 , (i = 1, 2,... N),


## G|m+ 1


## 2 = G|m + G|m+1


## 2 . (48)

When N = 2, Eq. (48) becomes identical with Eqs. (38) and (39). The proposed half-point values have the same form for each species i and satisfy the compatibility condition in the N-component system. In other words, the proposed scheme is consistent with the fact that the mass of each species is independent of each other and that all species are equal for i (following Dalton’s law). Also, considering the sum of each species, the total mass ρ|m+ 1


### 2 (which is not solved directly in


### this system) is written as


# ρ|m+ 1


## 2 =

N �

i ρYi|m+ 1

2


## = φ−��N i ρYi|m � + φ+ ��N i ρYi|m+1 �


## 2

10

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Table 1 Proposed fully conservative and pressure-equilibrium (FC/PE) preserving numerical fluxes used in compressible multi-component Euler equations in Eq. (17). Although both FC/PE (S) and FC/PE (D) satisfy the compatibility condition Eq. (21), as will be discussed in Section 5, the proposed FC/PE (S) maintains both the conservation and pressure equilibrium.**

Numerical flux FC/PE (S) (Proposed) FC/PE (D)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq014.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq015.png)

2 u|m+u|m+1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq016.png)

2


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq017.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq018.png)

2 u|m+u|m+1

2 u|m+u|m+1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq019.png)

2 ��|m+ 1 2

p|m+p|m+1

2 p|m+p|m+1

2


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq020.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq021.png)

2 u|m+u|m+1

2 u|mu|m+1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq022.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq023.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq024.png)

pG|m+pG|m+1

2 u|m+u|m+1

2 (pGu)|m+(pGu)|m+1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq025.png)

2

u|m p|m+1+u|m+1 p|m

2 (up)|m+(up)|m+1

2


# = φ−ρ|m + φ+ρ|m+1


## 2 . (49)


# Even in the (ρ, ρu, E, ρYi) system, the proposed schemes are valid if the ρ|m+ 1


# 2 in Eq. (49) is used instead of ρY N|m+ 1


### 2 .

This fact indicates that solving the (ρ, ρu, E, ρYi) system is discretely equivalent to (ρYi, ρu, E) system for the proposed scheme (see Appendix A). In conclusion, the proposed scheme is generalized and applicable to multi-component flows for any number of species.


### 3.4. Complete forms of the proposed scheme satisfying pressure equilibrium


# Although the half-point values of ρYi|m+ 1


## 2 and G|m+ 1


### 2 are derived from the compatibility condition in Eq. (48), the


## numerical flux in Eq. (17) is not yet uniquely determined, e.g., the mass of species flux � Ci = ρYiu may be evaluated in the divergence form � Ci|m± 1


# 2 = (ρYiu)|m± 1


## 2 or the split form � Ci|m± 1


# 2 = ρYi|m± 1


## 2 u|m± 1


### 2 . Therefore, we examine the following

two schemes that satisfy the compatibility condition Eq. (21) using Eqs. (48) and (49). Table 1 summarizes the numerical fluxes of the proposed scheme. First, the fully conservative and pressure-equilibrium preserving scheme in the split form (FC/PE (S)) is designed to return to KEEPPE scheme [30] in a single-component flow. The half-point values in Eq. (48) are used in � Ci and �I. � Ci is described in the split form � Ci|m+ 1


# 2 = ρYi|m+ 1


## 2 u|m+ 1


## 2 . The mass-related fluxes � Mu and � K are also


# written in the split form including ρ|m+ 1


## 2 in Eq. (49) to be consistent with � Ci. �I is split into the internal energy and velocity, �I|m+ 1 2 = pG|m+ 1


## 2 u|m+ 1


## 2 , to maintain the pressure equilibrium as mentioned in Ref. [30]. Other fluxes ��and � P are the same

as the KEEP scheme [25] based on the split form. In addition, the fully conservative and pressure-equilibrium preserving scheme in the divergence form (FC/PE (D)) can be constructed to return to the divergence scheme in a single-component flow. � Ci is described by multiplying the standard divergence form flux by the coefficient φ±. � Mu and � K are determined in the same way as � Ci, and other fluxes ��, �I, and � P are standard central fluxes in the divergence form. In the flow with constant p = p0 and u = u0, the fluxes in both FC/PE (S) and FC/PE (D) reduce to the half-point values in Eq. (48). These two schemes are constructed in a central difference form with second-order spatial accuracy. Higher-order accuracy may be achieved by increasing the number of stencils, similar to the existing high-order KEEP scheme [27] for single-component flows.

Also, as shown later in Section 5, in terms of the consistency of the compatibility condition to velocity fields, in this study, we recommend the fully conservative and pressure-equilibrium preserving scheme in the split form (FC/PE (S)) as the proposed scheme. The proposed scheme may be extended straightforwardly to multi-dimension by replacing the scalar u with vector u and adding δ to ��, similarly to the existing KEEP schemes [26,29]. Note that u|mu|m+1 in � K represents the inner product of the velocity vectors.

Next, we discuss the behavior of the proposed scheme outside the interfaces where the composition and specific heats do not change. In other words, Yi is constant (a single-component or well-mixed fluid), and thus, the following relationship holds


# ρYi|m+ 1


## 2 =

M|m


### ρ|m ρ|m+1 M|m+1 ρYi|m + M|m+1


### ρ|m+1 ρ|m M|m ρYi|m+1


## 2 →ρYi|m + ρYi|m+1


## 2 ,


# ρ|m+ 1


## 2 =

M|m


### ρ|m ρ|m+1 M|m+1 ρ|m + M|m+1


### ρ|m+1 ρ|m M|m ρ|m+1


## 2 →ρ|m + ρ|m+1


## 2 ,


## G|m+ 1


## 2 = G|m + G|m+1


## 2 → 1 γ −1,

11

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 1. Initial distributions of the mass of each species in the 1D inviscid smooth interfaces advection problem ( , ρY1; , ρY2). (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)**


### where


# ρYi|m


# ρ|m = ρYi|m+1


# ρ|m+1 = Yi = const.,


## M|m = M|m+1 = M = const.,


## G|m = G|m+1 = 1 γ −1 = const.

Here, the proposed numerical fluxes are reduced to a standard central form outside the interfaces. Therefore, FC/PE (S) behaves as the KEEPPE [30] scheme outside the interfaces automatically and can maintain the equilibrium without any special treatment such as switching fluxes by sensors.


### 4. Numerical experiments

In this section, we first verify the proposed scheme defined in Table 1 through a one-dimensional smooth interfaces advection problem in pressure and velocity equilibrium. Then, similar problems in the two-dimensional domain are computed to show the multi-dimensional and multi-component capacity of the proposed scheme.


### 4.1. 1D inviscid smooth interfaces advection problem in pressure and velocity equilibriums

In this problem, the advection of 1D interfaces between two species is simulated. Here, we assume that the two species have different ratios of specific heats, γ1 = 1.4 and γ2 = 1.66, and molar weights, M1 = 28.0 and M2 = 4.0. Since the initial velocity and pressure are constant in space, the pressure and velocity equilibriums should preserve. The simulation is performed in a periodic region [0, 1] with 501 grid points, where sufficient points are located within the initial smooth interfaces. The initial conditions are given by


## ⎡


## ⎢⎢⎣


# (ρY1)0 (ρY2)0 u0 p0


## ⎤


## ⎥⎥⎦=


## ⎡


## ⎢⎢⎣

w1


## 2 (1 −tanh(k(r −rc)))

w2


## 2 (1 + tanh(k(r −rc))) 1.0 0.9


## ⎤


## ⎥⎥⎦ (50)

where r = |x −xc|, xc is the coordinate of the wave center, rc is the radius of the wave center to the interfaces, w is the weight coefficient of the density and k is the coefficient to adjust the smoothness. In this case, xc = 0.5, rc = 0.25, (w1, w2) = (0.6, 0.2), and k = 20. Here, the initial pressure is set to 0.9 just for the visibility of the figures, although it can be any other value. The initial density distributions are shown in Fig. 1. The ρY1 is distributed in 0.25 < x < 0.75, and ρY2 is distributed in x < 0.25 and 0.75 < x, touching each other through smooth interfaces. The four-stage fourth-order Runge– Kutta method is used with the CFL number of 0.05 (�t = 10−4) for the time integration. The proposed scheme (FC/PE (S)) that is described in Table 1 is compared to FC/PE (D), the conventional non-conservative and pressure-equilibrium preserving schemes solving the pressure-evolution equation [1] (NC-p), solving G directly [2] (NC-G), and the divergence form scheme which is fully conservative but not pressure-equilibrium preserving (NPE). Note that NPE is obtained by applying φ+ = φ−= 1 to FC/PE (D). NPE is conservative but does not satisfy the compatibility condition (Eq. (21)).


> **Fig. 2 shows the distributions of u and p at an early stage t = 1.0. In the results by FC/PE (S) and FC/PE (D), the velocity and pressure equilibriums preserve at a sufficiently small order (≃10−13). When using NPE, the equilibrium is not maintained and spurious oscillations occur in the velocity and pressure. These oscillations originate from the numerical errors of specific heats that are generated by violating the compatibility condition Eq. (21). The result shows that both FC/PE (S) and FC/PE (D) schemes preserve the equilibrium by satisfying the compatibility condition without solving the**

12

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 2. Velocity and pressure distributions of 1D inviscid smooth interfaces advection problem at t = 1.0 ( , exact; , NPE; , FC/PE (S) (Proposed); The results by FC/PE (D), NC-p, and NC-G are almost the same as that by FC/PE (S)).**


> **Fig. 3. Time evolutions of L2 pressure errors of the 1D inviscid smooth interfaces advection problem ( , NPE; , FC/PE (D); , FC/PE (S) (Proposed)).**


> **Fig. 4. Distributions of velocity, pressure, and mass of species in the 1D inviscid smooth interfaces advection problem at t = 6.0 ( , exact; , u; , p; , ρY1; , ρY2).**

equilibrium condition Eq. (15) directly. However, FC/PE (D) has a stability issue in the pressure-equilibrium preservation at the later stage. Fig. 3 shows the time evolution of pressure errors. FC/PE (S) and FC/PE (D) have almost negligible errors for the pressure in comparison to NPE at the beginning of the calculation (t ≤1). However, the pressure errors of FC/PE (D) rapidly grow with time steps and finally lead to the divergence of computation at t ≈3.8. The causes of this error growth are discussed in Section 5. On the other hand, FC/PE (S) maintains the pressure equilibrium without developing numerical errors. Fig. 4 shows the distributions of ρYi, u and p at t = 6.0. In the result using NPE, spurious oscillations occur in the

13

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 5. Conservation errors in total energy and mass of species 1 in the 1D inviscid smooth interfaces advection problem ( , exact; , FC/PE (S) (Proposed); , NC-p [1]; , NC-G [2]; Conservation errors by FC/PE (D) and NPE are also exactly zero, while these schemes blow up at t ≈3.8 and t ≈6.5, respectively).**

velocity and pressure and grow with time, which also disturbs the density fields. Finally, the computation by NPE blows up at t ≈6.5. In contrast, FC/PE (S) maintains the equilibrium at the machine zero order for a long time. Therefore, the proposed scheme can maintain the equilibrium, and the proposed FC/PE (S) is superior to FC/PE (D) because the errors do not grow.


> **Fig. 5 shows the time evolution of the conservation errors in the total energy E and mass of species ρY1. Analytically, the total energy and mass of species conserve across the entire computational domain. Since the existing pressure-equilibrium preserving schemes (NC-p [1] and NC-G [2]) are non-conservative, the total energy does not conserve when solving the pressure-evolution equation (NC-p) instead of the total energy equation, while the mass does not conserve when solving G equation (NC-G) instead of the mass of species equation. In contrast, FC/PE (S) (and also FC/PE (D)) satisfies the primary conservation because the proposed scheme solves the equations of the primary conservative quantities directly. Therefore, in this experiment, the proposed scheme (FC/PE (S)) is the only one that satisfies both the conservation and pressureequilibrium preservation for a long time.**


### 4.2. Multi-dimensional multi-component inviscid smooth interfaces advection problem in the pressure and velocity equilibriums

In this problem, we demonstrate that the proposed scheme is applicable to multi-dimensional multi-component interface problems. Here, we assume that the three species have different ratios of specific heats, γ1 = 1.4, γ2 = 1.66, and γ3 = 1.29, and molar weights, M1 = 28.0, M2 = 4.0, and M3 = 44.0. Since the initial pressure and velocity are constant in space, the pressure and velocity equilibriums are maintained analytically. The simulation is performed in a periodic plane of dimension [0, 1] with 5012 grid points, where sufficient grid points are located within the smooth interfaces. The initial conditions are given by


## ⎡


## ⎢⎢⎢⎢⎢⎣


# (ρY1)0 (ρY2)0 (ρY3)0 u0 v0 p0


## ⎤


## ⎥⎥⎥⎥⎥⎦ =


## ⎡


## ⎢⎢⎢⎢⎢⎢⎢⎣

w1


## 2 (1 −tanh(k(r −rc1)))

w2


## 2 (1 + tanh(k(r −rc2)))

w3


## 2 (1 + tanh(k(r −rc3))) 1.0 1.0 0.9


## ⎤


## ⎥⎥⎥⎥⎥⎥⎥⎦


## (51)

where r = � (x −xc)2 + (y −yc)2, (xc, yc) = (0.5, 0.5), (rc1, rc2, rc3) = (0.3, 0.3, 0.2), (w1, w2, w3) = (0.4, 0.2, 0.1), and k = 15. The initial density distributions are shown in Fig. 6. The four-stage fourth-order Runge–Kutta method is used with the CFL number of 0.05 (�t = 10−4) for the time integration. Here, the proposed scheme (FC/PE (S)) is compared to the divergence form scheme (NPE).


> **Fig. 7 shows the distributions of total density, velocity, and pressure at t = 5.0, when the circular interface advects through the domain five times. In the results of FC/PE (S), the pressure and velocity equilibriums are maintained at a sufficiently small order (≃10−12). When using NPE, spurious oscillations disturb the structure of waves in the whole region, and the computation blows up at t ≈5.5. These results show that the proposed scheme can be applied in dimension by dimension to multi-dimensional multi-component flows.**

14

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 6. Initial conditions of the multi-dimensional multi-component inviscid smooth interfaces advection problem ( , ρ; , ρY1; , ρY2; , ρY3 in (b)).**


### 4.3. 2D inviscid smooth interfaces advection problem with tangential velocity jump

This section presents a two-dimensional interface advection problem with the tangential velocity varying in space [10] as a multi-dimensional test case. Here, we assume that the two species have different ratios of specific heats, γ1 = 1.4 and γ2 = 1.66, and molar weights, M1 = 28.0 and M2 = 4.0. The simulations are performed in a periodic region [0, 1] with 5012


### grid points, where sufficient points are located within the initial smooth interfaces. The initial conditions are given by


## ⎡


## ⎢⎢⎢⎣


# (ρY1)0 (ρY2)0 u0 v0 p0


## ⎤


## ⎥⎥⎥⎦=


## ⎡


## ⎢⎢⎢⎢⎣

w1


## 2 (1 −tanh(k(r −rc)))

w2


## 2 (1 + tanh(k(r −rc))) 1.0 w v tanh(k(r −rc)) 0.9


## ⎤


## ⎥⎥⎥⎥⎦ (52)

where xc = 0.5, rc = 0.25, (w1, w2, w v) = (0.6, 0.2, 0.5), and k = 20. The four-stage fourth-order Runge–Kutta method is used with the CFL number of 0.05 (�t = 10−4) for the time integration. Here, the proposed scheme (FC/PE (S)) is compared to the divergence form scheme (NPE).


> **Fig. 8 shows the distributions of pressure and the slice of tangential velocity at t = 5.0, when the interface advects through the domain five times. In the FC/PE (S) results, the pressure equilibrium is maintained, and the tangential velocity is advected correctly. When using NPE, spurious oscillations of pressure and velocity occur. These results indicate that the proposed scheme is applicable to flows with the tangential velocity jump.**


### 5. Difference between FC/PE (S) and FC/PE (D)

In the numerical experiments conducted in Section 4, only FC/PE (S) preserves the equilibrium for a long time. Although FC/PE (D) preserves the equilibrium at the beginning of the calculation as analyzed in Section 3, the computation diverges after a short time. In this section, therefore, we investigate the difference between FC/PE (S) and FC/PE (D).

In Section 3, we discuss the equilibrium condition under the assumption of the constant velocity u = u0 and pressure p = p0. However, in the calculation, u and p have errors essentially due to round-off. The errors may affect the compatibility condition discussed with u = u0 in Section 3 and trigger serious errors. Therefore, we reconsider the equilibrium condition with a nonuniform u. We discretize the equations of G and ρYi with a nonuniform u as


## ∂G


## ∂t = −∂Gu


## ∂x


## ���� m ≃− (Gu)|m+ 1


## 2 −(Gu)|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq026.png)


# ∂ρYi


## ∂t = −∂ρYiu


## ∂x


## ���� m ≃− (ρYiu)|m+ 1


# 2 −(ρYiu)|m−1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq027.png)


### Substituting Eqs. (53) and (54) into Eq. (12) yields

15

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 7. Total density and relative errors of velocity, (u −uexact)/uexact, and pressure, (p −pexact)/pexact, in the multi-dimensional multi-component inviscid smooth interfaces advection problem.**

16

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


> **Fig. 8. Relative errors of pressure (p −pexact)/pexact and the distributions of tangential velocity v in the 2D inviscid smooth interface advection problem with tangential velocity at t = 5.0 ( , exact).**


## (Gu)|m+ 1


## 2 −(Gu)|m−1


## 2 =

N �

i


## � β′ i|m � (ρYiu)|m+ 1


# 2 −(ρYiu)|m−1

2


## �� . (55)

Equation (55) is the extended (more generalized) compatibility condition for a nonuniform velocity flow. Using Eq. (55), we verify the long-time pressure-equilibrium preserving characteristics of the proposed scheme (FC/PE (S)).

Let us first consider FC/PE (D), which can preserve the equilibrium only at the initial state. Substituting the mass and internal energy fluxes of FC/PE (D) (see Table 1) into the left-hand-side (LHS) and RHS of Eq. (55) yields


## FC/PE (D) :


## (LHS) = (Gu)|m+1 + (Gu)|m


## 2 −(Gu)|m + (Gu)|m−1


## 2


## = (Gu)|m+1 −(Gu)|m−1


## 2 , (56)


## (RHS) =

N �


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq028.png)


## � � Ci|FC/PE (D)

m+ 1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq029.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq030.png)

2


## �


## = 1


## 2


## M|m


# ρ|m

N �

i


## �Gi Mi −1


## Mi G|m


## �� M|m+1


# ρ|m+1 ρ|m


## M|m (ρYiu)|m+1 + M|m


# ρ|m ρ|m+1


## M|m+1 (ρYiu)|m


## −M|m


# ρ|m ρ|m−1


## M|m−1 (ρYiu)|m −M|m−1


# ρ|m−1 ρ|m


## M|m (ρYiu)|m−1


## �

17

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


## = (Gu)|m+1 −(Gu)|m−1


## 2 −G|m u|m+1 −u|m−1


## 2 , (57)

with assuming the pressure is constant. In the analysis, (LHS) ̸= (RHS), which indicates that FC/PE (D) does not satisfy the extended compatibility condition Eq. (55). Therefore, FC/PE (D) can preserve the equilibrium only when the velocity is constant. Once the velocity is disturbed even with a small amplitude, the scheme no longer preserves the pressure equilibrium and the computation diverges.


### A similar analysis verifies the long-time pressure-equilibrium preserving characteristics of FC/PE (S):


## FC/PE (S) :


## (LHS) = G|m+1 + G|m


## 2


## u|m+1 + u|m


## 2 −G|m + G|m−1


## 2


## u|m + u|m−1


## 2


## = 1


## 2


## �(Gu)|m+1 −(Gu)|m−1 2 + G|m u|m+1 −u|m−1


## 2 + u|m G|m+1 −G|m−1


## 2


## � , (58)


## (RHS) =

N �


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq031.png)


## � � Ci|FC/PE (S)

m+ 1


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq032.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq033.png)

2


## �


## = 1


## 2


## M|m


# ρ|m

N �

i


## �Gi Mi −1


## Mi G|m


## ��� M|m+1


# ρ|m+1 ρ|m


## M|m ρYi|m+1 + M|m


# ρ|m ρ|m+1


## M|m+1 ρYi|m


## � u|m+1 + u|m


## 2


## −


## � M|m


# ρ|m ρ|m−1


## M|m−1 ρYi|m + M|m−1


# ρ|m−1 ρ|m


## M|m ρYi|m−1


## � u|m + u|m−1


## 2


## �


## = 1


## 2


## �(Gu)|m+1 −(Gu)|m−1 2 + G|m u|m+1 −u|m−1


## 2 + u|m G|m+1 −G|m−1


## 2


## � . (59)

Here, (LHS) = (RHS) is satisfied. In other words, the convective velocity of G on the LHS and the one of ρYi on the RHS are consistent regardless of the velocity fields. Therefore, FC/PE (S) can maintain the pressure equilibrium without developing errors even in disturbed velocity fields. Furthermore, FC/PE (S) is designed to reduce to the KEEPPE in a single-component constant γ flow, which is stable as shown in Ref. [30]. Thus, we recommend FC/PE (S) as the proposed scheme.


### 6. Conclusions

This study proposed a fully conservative and pressure-equilibrium preserving scheme for discontinuity-free compressible multi-component flows with an arbitrary number of species. The proposed scheme solves only the conservaton equations of the mass of species, momentum, and total energy in contrast to the conventional non-conservative or overspecified pressure-equilibrium preserving schemes that directly solve the transport equation of pressure or the equilibrium condition. The key to this study is the compatibility condition, which is the condition for implicitly satisfying the pressure-equilibrium condition at the discrete level. The compatibility condition is derived based on the analytical relation between the derivatives of the specific heats G and mass of species ρYi. The proposed scheme employs spatial discretization that implicitly satisfies the compatibility condition and does not require any special treatment such as switching fluxes at interfaces.

We verified the proposed scheme through the numerical test of the inviscid smooth interfaces advection problem in the pressure and velocity equilibriums. The proposed scheme maintains both the conservation and pressure-equilibrium preservation, while the existing conservative scheme generates spurious oscillations. Also, the conventional pressure-equilibrium preserving but non-conservative schemes do not conserve the total energy or mass of species. Furthermore, we confirmed that the proposed scheme is straightforwardly extendable to multi-dimension by the dimension-by-dimension approach. According to a nonuniform-velocity analysis, FC/PE (S) can maintain the pressure equilibrium even when the velocity is disturbed in contrast to FC/PE (D). Even for discontinuity-free flows, the proposed scheme is the first to achieve both conservation and equilibrium. In conclusion, we recommend FC/PE (S) as the baseline scheme for compressible multi-component flow simulations.


### CRediT authorship contribution statement

Yuji Fujiwara: Conceptualization, Formal analysis, Investigation, Methodology, Software, Visualization, Writing – original draft, Writing – review & editing. Yoshiharu Tamaki: Conceptualization, Methodology, Writing – original draft, Writing – review & editing. Soshi Kawai: Conceptualization, Funding acquisition, Project administration, Resources, Supervision, Writing – review & editing.

18

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


### Declaration of competing interest

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Soshi Kawai reports financial support was provided by Japan Society for the Promotion of Science.


### Data availability


### Data will be made available on request.


### Acknowledgements


### This work was supported in part by Japan Society for the Promotion of Science (JSPS) KAKENHI Grant Number JP19K21927 and JP21H01522.


# Appendix A. Proposed scheme in (ρ, ρu, E, ρYi) system

Here, we describe the derivation of the proposed scheme in the (ρ, ρu, E, ρYi) system. The conservation equations for the total mass, momentum, total energy, and mass of species for N-component compressible inviscid flows are given by


## ⎧ ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩


# ∂ρ


# ∂t + ∇· (ρu) = 0, (a)


# ∂(ρu)


## ∂t + ∇· (ρu ⊗u + pδ) = 0, (b)


## ∂E


## ∂t + ∇· ((E + p)u) = 0, (c)


# ∂ρYi


## ∂t + ∇· (ρYiu) = 0, (d)


## (A.1)


## where i = 1, 2..., N −1.


### A.1. Compatibility condition


# The mixing rule Eq. (4) is described in terms of ρ and ρYi as


## G ≡ 1 γ −1 = M

N �

i


## Gi Mi Yi =

N �

i


## Gi Mi ρYi


# ρ

N �

i


## 1


## Mi ρYi


# ρ


## =


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq034.png)

i


## �Gi Mi −GN


## MN


# �ρYi ρ + GN


## MN


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq035.png)

i


## �1


## Mi − 1


## MN


# �ρYi ρ + 1


## MN


## . (A.2)


# From Eq. (A.2), G = G(ρ, ρYi, Gi, Mi) = G(ρ, ρYi). Note that Mi and γi are constant and unique to each species.

The time derivative of pressure is analytically derived similarly to the derivations in Section 2.3 for the (ρYi, ρu, E) system. Since the internal energy is defined as ρe = ρe(p, ρ, ρYi), the derivative of ρe is analytically replaced by the derivatives of p, ρ and ρYi as


# ∂ρe


## ∂t = �∂ρe ∂p


## �


### ρ,ρYi


## ∂p


## ∂t + �∂ρe ∂ρ


## �


### p,ρYi


# ∂ρ


## ∂t +


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq036.png)

i


# �∂ρe ∂ρYi


## �


### p,ρ,ρY j̸=i


# ∂ρYi


## ∂t (A.3)


### From Eq. (A.3), the time derivative of pressure is described as


## ∂p


## ∂t = �∂ρe ∂p


## �−1


### ρ,ρYi


## � ∂ρe


## ∂t − �∂ρe ∂ρ


## �


### p,ρYi


# ∂ρ


## ∂t −


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq037.png)

i


# �∂ρe ∂ρYi


## �


### p,ρ,ρY j̸=i


# ∂ρYi


## ∂t


## �


## . (A.4)


### Equation (A.4) shows that the pressure equilibrium requires


# ∂ρe


## ∂x = �∂ρe ∂ρ


## �


### p,ρYi


# ∂ρ


## ∂x +


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq038.png)

i


# �∂ρe ∂ρYi


## �


### p,ρ,ρY j̸=i


# ∂ρYi


## ∂x . (A.5)

19

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973

Equation (A.5) is the pressure equilibrium condition for the (ρ, ρu, E, ρYi) system. Although Eq. (A.5) holds at the analytical level, it is not guaranteed generally at the discrete level. Substituting the discretized form of Eq. (A.1) into Eq. (A.5), the convective terms are discretized in space as


# ρe|m+ 1


# 2 −ρe|m−1


# 2 = α|m � ρ|m+ 1


# 2 −ρ|m−1

2


## � +


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq039.png)

i


## � βi|m � ρYi|m+ 1


# 2 −ρYi|m−1

2


## �� , (A.6)


# where α|m ≡ � ∂ρe


### ∂ρ �


### p,ρYi


## ���� m and βi|m ≡ � ∂ρe ∂ρYi


## �


### p,ρ,ρY j̸=i


## ���� m . Equation (A.6) is the compatibility condition in the (ρ, ρu, E, ρYi)


### system.


### A.2. Derivation of the proposed numerical fluxes


# Next, we derive the half-point values ρe|m+ 1


# 2 , ρ|m+ 1


# 2 , and ρYi|m+ 1


### 2 that satisfy the compatibility condition (Eq. (A.6)).


## Considering the fluxes conservation at m + 1


### 2 , we transform Eq. (A.6) similar to the derivations in the Section 3 for the (ρYi, ρu, E) system assuming ideal gases,


## G|m+1 −G|m = �α′|m −α′|m+1 �ρ|m+ 1 2 + α′|m+1ρ|m+1 −α′|mρ|m


## +


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq040.png)

i


## �� β′ i|m −β′ i|m+1 �ρYi|m+ 1 2 + β′ i|m+1ρYi|m+1 −β′ i|mρYi|m � , (A.7)


# where α′|m ≡ � ∂G ∂ρ �


### ρYi


## ���� m and β′ i|m ≡ � ∂G ∂ρYi


## �


### ρ,ρY j̸=i


## ���� m . In addition, G, α′, and β′ i may be described analytically using ρ and


# ρY1 by the mixing rule Eq. (4) as


## G|m = M|m

N �

i


## Gi Mi Yi|m = M|m


## �N−1 �

i


## �Gi Mi −GN


## MN


# �ρYi|m ρ|m + GN


## MN


## �


## , (A.8)


# α′|m = �∂G ∂ρ


## �


### ρYi


## ����� m = −M|2 m


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq041.png)

i


## �Gi −GN MiMN ρYi|m


# ρ|2 m


## � , (A.9)


## β′ i|m = �∂G ∂ρYi


## �


### ρ,ρY j̸=i


## ����� m = M|2 m


## ⎡


## ⎣Gi −GN


## MiMN


## 1 ρ|m +


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq042.png)

j


## �Gi −G j MiM j + G j −GN


## M jMN + GN −Gi


## MN Mi


# �ρY j|m ρ|m


## ⎤


## ⎦, (A.10)


### where


## M|m =

N �

i


## 1


## Mi Yi|m =


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq043.png)

i


## �1


## Mi − 1


## MN


# �ρYi|m ρ|m + 1


## MN . (A.11)


### Substituting Eqs. (A.8)–(A.11) into Eq. (A.7) yields

N−1 �

i


## �Gi −GN MiMN


# ��ρYi|m+1 ρ|m+1 −ρYi|m


# ρ|m


## �


## =

N−1 �

i


## �Gi −GN MiMN


## ��� M|m+1


## M|m


# ρYi|m+1


# ρ|2 m+1 − M|m


## M|m+1


# ρYi|m


# ρ|2 m


## �


# ρ|m+ 1


## 2 −


## � M|m+1


## M|m


## 1 ρ|m+1 − M|m


## M|m+1


## 1 ρ|m


## �


# ρYi|m+ 1

2


## �


## +

N−1 �

i

N−1 �

j


## �Gi −G j MiM j + G j −GN


## M jMN + GN −Gi


## MN Mi


## �


## �� M|m+1


## M|m


# ρY j|m+1


# ρ|2 m+1 − M|m


## M|m+1


# ρY j|m


# ρ|2 m


## �


# ρYi|m+ 1


## 2 + 1


## 2


# �ρYi|m+1 ρ|m+1 ρY j|m


# ρ|m −ρYi|m


# ρ|m ρY j|m+1


# ρ|m+1


## ��


## . (A.12)


# In Eq. (A.12), only ρ|m+ 1


# 2 and ρYi|m+ 1


# 2 are the unknowns and the others are known. Therefore, once one of ρ|m+ 1


### 2 and ρYi|m+ 1


### 2 is determined, the other is obtained uniquely.

20

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


# The proposed half-point values used for the numerical fluxes in the (ρ, ρu, E, ρYi) system are derived as


## ⎧ ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩


# ρ|m+ 1


# 2 = φ−ρ|m + φ+ρ|m+1


## 2 ,


# ρYi|m+ 1


# 2 = φ−ρYi|m + φ+ρYi|m+1


## 2 , (i = 1, 2,... N −1),


## G|m+ 1


## 2 = G|m + G|m+1


## 2 .


## (A.13)


# In addition, ρY N, which is not solved directly in this system, is written as


# ρY N|m+ 1


# 2 = ρ|m+ 1


## 2 −


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq044.png)


![Equation](images/2023_Fujiwara_Tamaki_Kawai_pressure_equilibrium_preserving_multicomponent_eq045.png)

2


## = φ−� ρ|m −�N−1 i ρYi|m � + φ+ � ρ|m+1 −�N−1 i ρYi|m+1 �


## 2


# = φ−ρY N|m + φ+ρY N|m+1


## 2 . (A.14)


# Equation (A.14) is the same form as ρYi|m+ 1


# 2 in the (ρYi, ρu, E) system in Eq. (48). This result indicates that instead of

solving the (ρYi, ρu, E) system, we can also solve the (ρ, ρu, E, ρYi) system using the proposed scheme. In conclusion, the same fluxes are derived in the (ρ, ρu, E, ρYi) and (ρYi, ρu, E) system, and they are equivalent at the discrete level.


### Appendix B. Equilibrium preserving scheme for the stiffened EoS

Here, we describe the derivation of the proposed scheme for the stiffened EoS, which is often used in the simulations for two-phase flows [10,32]. The definition of the stiffened EoS for N-component compressible flows is given by


# ρe = p γ −1 + γ π


# γ −1


## = pG + A, (B.1)

where π is a constant parameter characteristic of the material (e.g., πi is an extremely high value in a liquid phase and zero in a gas phase). The stiffened EoS is reduced to the ideal gas EoS when π is zero.


### B.1. Mixing rule


### The mixing rule for the stiffened EoS is described as


# ρe = pG + A


## = p M


# ρ

N �

i


## Gi Mi ρYi + M


# ρ

N �

i


## Ai Mi ρYi, (B.2)

where Mi, Gi and Ai (= γiπi/(γi −1)) are constant and unique to each species. ρYi is the only variable that describes the fluids mixing under the pressure-equilibrium condition (p = p0) since Eq. (B.2) indicates ρe = ρe(p, ρYi).


### B.2. Derivation of the numerical fluxes from the compatibility condition


### The compatibility condition (21) in Section 3 holds for the Jacobian � ∂ρe ∂ρYi


## �


### p,ρY j̸=i of an arbitrary EoS. Therefore, the pro-


### posed method can be applied to stiffened EoS as well. From the mixing rule Eq. (B.2), the Jacobian is analytically calculated as


## βi|m ≡ �∂ρe ∂ρYi


## �


### p,ρY j̸=i


## ����� m = p0


## �∂G ∂ρYi


## �


### ρY j̸=i


## ����� m + �∂A ∂ρYi


## �


### ρY j̸=i


## ����� m , (B.3)


### where

21

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973


## �∂G ∂ρYi


## �


### ρY j̸=i


## ����� m = M|2 m ρ|2 m

N �

j


## �Gi −G j MiM j ρY j|m


## � , (B.4)


## �∂A ∂ρYi


## �


### ρY j̸=i


## ����� m = M|2 m ρ|2 m

N �

j


## �Ai −A j MiM j ρY j|m


## � . (B.5)


### Substituting Eqs. (B.1) and (B.3) into Eq. (21) yields


## p0 � G|m+ 1


## 2 −G|m−1

2


## � + A|m+ 1


## 2 −A|m−1


## 2 =

N �

i


## �


## p0


## �∂G ∂ρYi


## �


### ρY j̸=i


## ����� m + �∂A ∂ρYi


## �


### ρY j̸=i


## ����� m


## �� ρYi|m+ 1


# 2 −ρYi|m−1

2


## � .


## (B.6)


# Here, we derive the compatible ρYi|m+ 1


## 2 , G|m+ 1


## 2 and A|m+ 1


### 2 from Eq. (B.6). For simplicity, Eq. (B.6) is divided into the parts


### of G and A as


## ⎧ ⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎩


## G|m+ 1


## 2 −G|m−1


## 2 =

N �

i


## �∂G ∂ρYi


## �


### ρY j̸=i


## ����� m


## � ρYi|m+ 1


# 2 −ρYi|m−1

2


## � , (a)


## A|m+ 1


## 2 −A|m−1


## 2 =

N �

i


## �∂A ∂ρYi


## �


### ρY j̸=i


## ����� m


## � ρYi|m+ 1


# 2 −ρYi|m−1

2


## � . (b)


## (B.7)


### The compatibility condition for the ideal gas part (Eq. (B.7a)) is already discussed in Section 3. Therefore, we consider the half point values A|m+ 1


# 2 and ρYi|m+ 1


### 2 satisfying Eq. (B.7b). Similar to the ideal gas case, Eq. (B.7b) is transformed using m


## and m + 1 point values and substituted by Eqs. (B.2) and (B.5). Consequently, the following equation is obtained:


## 1


## 2

N �

i

N �

j


## �Ai −A j MiM j


# �ρYi|m+1ρY j|m −ρYi|mρY j|m+1 �� =

N �

i

N �

j


## �Ai −A j MiM j


## � φ−ρY j|m −φ+ρY j|m+1 �ρYi|m+ 1 2


## � .


## (B.8)

Since Eq. (B.8) is the same form as for the ideal gas case by just replacing Gi with Ai, we can derive fluxes similarly. Here, the half-point values used for the numerical fluxes satisfying the equilibrium are derived as


## ⎧ ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎨


## ⎪⎪⎪⎪⎪⎪⎪⎪⎪⎪⎩


# ρYi|m+ 1


# 2 = φ−ρYi|m + φ+ρYi|m+1


## 2 , (i = 1, 2,... N),


## G|m+ 1


## 2 = G|m + G|m+1


## 2 .


## A|m+ 1


## 2 = A|m + A|m+1


## 2 .


## (B.9)


# Equation (B.9) shows that ρYi|m+ 1


### 2 is the same form to the ideal gas case. These half-point values are used in the internal


## energy flux �I|m+ 1


### 2 as


## �I|m+ 1 2 = �pG|m + pG|m+1 2 + A|m + A|m+1


## 2


## �u|m + u|m+1 2


# = ρe|m + ρe|m+1


## 2


## u|m + u|m+1


## 2 , (B.10)


### and the other fluxes are the same as in Table 1. In conclusion, based on the compatibility condition derived with an arbitrary EoS, we successfully extend the proposed scheme to the stiffened EoS.


## References

[1] S. Karni, Multicomponent flow calculations by a consistent primitive algorithm, J. Comput. Phys. 112 (1994) 31–43. [2] R. Abgrall, How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach, J. Comput. Phys. 125 (1996) 150–160. [3] K. Shyue, An efficient shock-capturing algorithm for compressible multicomponent problems, J. Comput. Phys. 142 (1998) 208–242. [4] R. Abgrall, S. Karni, Computations of compressible multifluids, J. Comput. Phys. 169 (2001) 594–623.

22

Y. Fujiwara, Y. Tamaki and S. Kawai Journal of Computational Physics 478 (2023) 111973

[5] E. Johnsen, F. Ham, Preventing numerical errors generated by interface-capturing schemes in compressible multi-material flows, J. Comput. Phys. 231 (2006) 5705–5717. [6] S. Kawai, H. Terashima, H. Negishi, A robust and accurate numerical method for transcritical turbulent flows at supercritical pressure with an arbitrary equation of state, J. Comput. Phys. 300 (2015) 116–135. [7] H. Terashima, S. Kawai, N. Yamanishi, High-resolution numerical method for supercritical flows with large density variations, AIAA J. 49 (12) (2011) 2658–2672. [8] H. Terashima, M. Koshi, Approach for simulating gas–liquid-like flows under supercritical pressures using a high-order central differencing scheme, J. Comput. Phys. 231 (2012) 6907–6923. [9] J.J. Quirk, S. Karni, On the dynamics of a shock–bubble interaction, J. Fluid Mech. 318 (1996) 129–163. [10] R. Saurel, R. Abgrall, A simple method for compressible multifluid flows, SIAM J. Sci. Comput. 21 (3) (1999) 1115–1145. [11] E. Johnsen, Spurious oscillations and conservation errors in interface-capturing schemes, CTR Ann. Res. Briefs (2008) 115–126. [12] R.P. Fedkiw, X.D. Liu, S. Osher, A general technique for eliminating spurious oscillations in conservative schemes for multiphase and multispecies Euler equations, Int. J. Nonlinear Sci. Numer. Simul. 3 (2002) 99–105. [13] S. Karni, Hybrid multifluid algorithms, SIAM J. Sci. Comput. 17 (5) (1996) 1019–1039. [14] B. Boyd, D. Jarrahbashi, A diffuse-interface method for reducing spurious pressure oscillations in multicomponent transcritical flow simulations, Comput. Fluids 222 (2021) 104924. [15] E. Johnsen, T. Colonius, Implementation of WENO schemes in compressible multicomponent flow problems, J. Comput. Phys. 219 (2006) 715–732. [16] A.W. Cook, Enthalpy diffusion in multicomponent flows, Phys. Fluids 21 (5) (2009) 055109. [17] S. Kawai, H. Terashima, A high-resolution scheme for compressible multicomponent flows with shock waves, Int. J. Numer. Methods Fluids 66 (2011) 1207–1225. [18] H. Terashima, S. Kawai, M. Koshi, Consistent numerical diffusion term for simulating compressible multicomponent flows, Comput. Fluids 88 (2013) 484–495. [19] S. Mirjalili, C.B. Ivey, A. Mani, A conservative diffuse interface method for two-phase flows with provable boundedness properties, J. Comput. Phys. 401 (2020) 109006. [20] S.S. Jain, A. Mani, P. Moin, A conservative diffuse-interface method for compressible two-phase flows, J. Comput. Phys. 418 (2020) 109606. [21] A. Jameson, Formulation of kinetic energy preserving conservative schemes for gas dynamics and direct numerical simulation of one-dimensional viscous compressible flow in a shock tube using entropy and kinetic energy preserving schemes, J. Sci. Comput. 34 (2) (2008) 188–208. [22] S. Pirozzoli, Generalized conservative approximations of split convective derivative operators, J. Comput. Phys. 229 (19) (2010) 7180–7190. [23] F. Ismail, P.L. Roe, Affordable, entropy-consistent Euler flux functions II: entropy production at shocks, J. Comput. Phys. 228 (2009) 5410–5436. [24] P. Chandrashekar, Kinetic energy preserving and entropy stable finite volume schemes for compressible Euler and Navier–Stokes equations, Commun. Comput. Phys. 14 (5) (2013) 1252–1286. [25] Y. Kuya, K. Totani, S. Kawai, Kinetic energy and entropy preserving schemes for compressible flows by split convective forms, J. Comput. Phys. 375 (2018) 823–853. [26] Y. Kuya, S. Kawai, A stable and non-dissipative kinetic energy and entropy preserving (KEEP) scheme for non-conforming block boundaries on cartesian grids, Comput. Fluids 200 (2020) 104427. [27] Y. Kuya, S. Kawai, High-order accurate kinetic-energy and entropy preserving (KEEP) schemes on curvilinear grids, J. Comput. Phys. (2021) 110482. [28] Y. Kuya, S. Kawai, Modified wavenumber and aliasing errors of split convective forms for compressible flows, J. Comput. Phys. (2022) 111336. [29] Y. Tamaki, K. Yuichi, S. Kawai, Comprehensive analysis of entropy conservation property of non-dissipative schemes for compressible flows: KEEP scheme redefined, J. Comput. Phys. 468 (2022) 111494. [30] N. Shima, Y. Kuya, Y. Tamaki, S. Kawai, Preventing spurious pressure oscillations in split convective form discretization for compressible flows, J. Comput. Phys. 427 (2021) 110060. [31] W. Mulder, S. Osher, J.A. Sethian, Computing interface motion in compressible gas dynamics, J. Comput. Phys. 100 (2) (1992) 209–228. [32] S.A. Beig, E. Johnsen, Maintaining interface equilibrium conditions in compressible multiphase flows using interface capturing, J. Comput. Phys. 302 (2015) 548–566.

23

