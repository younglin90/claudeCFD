# A New Asymptotic-Preserving Dual Formulation Finite-Volume Method for the Compressible Euler Equations 

Alina Chertock[∗] , Smadar Karni[†] , Alexander Kurganov,[‡] and Lorenzo Micalizzi _[§]_ 

## **Abstract** 

The paper focuses on the development of numerical methods for the compressible Euler equations. It is well-known that if the Mach number is small, the system becomes stiff and hence explicit schemes suffer from severe time-step restrictions, making them inefficient or even impractical. Our objective is to develop an asymptotic preserving (AP) scheme that remains uniformly accurate and stable across all Mach numbers. 

Instead of the conservative hyperbolic flux splitting approach, which is widely used to design AP schemes, we consider a primitive (nonconservative) formulation and introduce a nonconservative hyperbolic splitting. The resulting system is discretized using a semi-implicit approach: the stiff part is handled semi-implicitly using second-order central differences, while the nonstiff part is treated explicitly using a second-order path-conservative centralupwind (CU) discretization. A key feature of our method is that the pressure at each time level is computed by solving a well-posed Poisson-type elliptic equation, thereby enforcing the AP property. Simultaneously, we evolve the conservative form of the system using a semi-discrete CU scheme. At the end of each stage of the time discretization, we perform a special post-processing that selects the appropriate numerical solution depending on the Mach number. This guarantees that in low-Mach-number regimes, the solution is obtained by the AP nonconservative scheme, while in higher-Mach-number regimes, a sharp and physically relevant solution is computed by the conservative CU scheme. 

Numerical experiments confirm that the proposed AP scheme achieves the expected second order of accuracy and that the time-step constraint is independent of the Mach number, making it a robust and efficient alternative to conventional explicit methods. 

**Key words:** Compressible Euler equations; low Mach number; asymptotic preserving (AP) schemes; hyperbolic splitting; semi-implicit methods; deferred correction. 

**AMS subject classification:** 76M12, 65M08, 65L04, 35B40, 35L60. 

> ∗Department of Mathematics, North Carolina State University, Raleigh, NC 27695, USA; `chertock@math.ncsu.edu` 

> †Department of Mathematics, University of Michigan, 48109, USA; `karni@umich.edu` 

> ‡Department of Mathematics and Shenzhen International Center for Mathematics, Southern University of Science and Technology, Shenzhen, 518055, China; `alexander@sustech.edu.cn` 

> _§_ Department of Mathematics, North Carolina State University, Raleigh, NC 27695, USA; `lmicali@ncsu.edu` 

1 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

2 

## **1 Introduction** 

The paper focuses on the compressible Euler equations, which, like any other hyperbolic system of PDEs, are characterized by a finite speed of propagation. This plays a crucial role in the development of explicit numerical methods, for which a major stability requirement is to keep the time steps inversely proportional to the maximum wave speed over the entire computational domain. 

It is well-known that low-Mach-number flows pose several major challenges for numerical simulations. A distinctive feature of such regimes is the appearance of both slow material waves, which transport quantities like entropy and vorticity, and fast acoustic waves, whose speeds scale inversely with the Mach number. As the Mach number decreases, the resulting stiffness imposes severe time-step restrictions on explicit methods and leads to excessive numerical diffusion, making such schemes inefficient or even impractical for real applications. Fully-implicit methods can address the stiffness, but have their own drawbacks: they tend to oversmear material waves (see, e.g., [10]), require the solution of large nonlinear systems, and may fail to capture the correct solution in the zero-Mach-number limit. 

To overcome these difficulties, a widely adopted strategy is to use either implicit-explicit (IMEX) or semi-implicit (SI) methods based on conservative hyperbolic flux splitting. This approach decomposes the hyperbolic flux into stiff and nonstiff components in a manner that preserves the conservative structure of the original system. The fast (stiff) part, associated with acoustic waves, is treated (semi-)implicitly to relax time-step limitations, while the slow (nonstiff) part is handled explicitly to accurately capture the evolution of material waves without excessive numerical diffusion. 

It is also known that, as the Mach number tends to zero, the compressible Euler equations reduce to the incompressible Euler equations. It is essential to ensure that numerical schemes also exhibit the same limiting behavior at the discrete level and provide a consistent discretization of the incompressible Euler equations as the Mach number tends to zero. Schemes that maintain this property are called asymptotic-preserving (AP). They were originally introduced to capture steadystate solutions for neutron transport in the diffusive regime [33,34], but the specific definition was introduced in [22,28,30] in the context of stiff kinetic equations. In recent years, AP schemes have been extensively studied and applied for simulating low-Mach-number flows; see, e.g., [7,9–11,13, 16–20,26,31,41,45,47,49] for a non-exhaustive list of references. 

All of the aforementioned AP schemes, which were designed for either the isentropic or full compressible Euler system, are based on different flux splitting strategies. A very simple and robust flux splitting, which was proposed in [26] for the isentropic Euler equations and later extended to the rotating shallow water equations in [36], seems to be rather optimal in the sense that it very accurately identifies and separates a linear stiff pressure term, which is then discretized implicitly. However, extending this flux splitting to the full Euler equations presents significant challenges. 

In this paper, we propose an alternative way of accurately identifying and separating a stiff part of the full Euler system: we first rewrite the studied system in a nonconservative form and then introduce a nonconservative hyperbolic splitting, which may be naturally viewed as an extension of the flux splitting from [26]. We integrate the nonconservative system using a SI method implemented as follows. The stiff part is approximated semi-implicitly using a simple second-order accurate central-differencing, and the nonstiff part is handled explicitly using a second-order pathconservative (PC) central-upwind (CU) discretization. The resulting SI approach is then realized in such a way that the pressure update consists of solving a Poisson-type elliptic equation: this is 

A New AP Method 

3 

used to enforce the AP property. 

However, the resulting SI method can only be applied to low-Mach-number regimes, where the magnitude of discontinuous waves is small. For large Mach numbers, solving nonconservative formulations of the Euler equations in the presence of discontinuities typically leads to nonphysical computed solutions, as was demonstrated in [4,27]. We therefore apply a dual formulation (DF) approach, which has been recently introduced in [14] (for other recent works on DF methods, we refer the reader to [2,3,5,43]), and solve the nonconservative and conservative formulations simultaneously. The latter one is discretized in a fully-explicit manner using the second-order semi-discrete CU discretization from [32]. This way, at each stage of a multi-stage SI time discretization (we have used the deferred correction (DeC) time discretization from [42]), two copies of the computed solution are evolved: one is AP, but nonconservative, while the second one is conservative, but nonAP. Hence, upon the completion of each stage of the time evolution, we post-process the obtained solutions to automatically ensure that in low-Mach-number regimes, the overall numerical solution is obtained by the AP nonconservative scheme, while in large (intermediate)-Mach-number regimes, the solution reduces to the sharp and conservative solution obtained by the CU scheme. 

The rest of the paper is organized as follows. In _§_ 2, we give the necessary preliminaries: we introduce the governing equations, namely, non-dimensional conservative and nonconservative (primitive) formulations of the full Euler equations, discuss their zero-Mach-number limit, and briefly review the considered DF framework. In _§_ 3, we introduce the novel AP DF finite-volume (DF-FV) scheme for compressible Euler equations, providing a rigorous proof of its AP character. In _§_ 4, we demonstrate the performance of the proposed scheme on a number of challenging numerical examples. Finally, concluding remarks can be found in _§_ 5. 

## **2 Preliminaries** 

The main goal of this section is to provide the background needed for presenting the proposed AP scheme. Specifically, we will describe: 

_•_ a nonconservative reformulation of the Euler equations in terms of primitive variables, which allows for a natural decomposition of the terms, which are stiff and nonstiff in the low-Machnumber regime; 

_•_ a formal asymptotic analysis of the studied equations in low-Mach-number regimes, providing the incompressible system that the AP scheme must accurately approximate in the zero-Mach-number limit; 

_•_ a DF framework, in which both conservative and nonconservative formulations of the studied system are numerically solved simultaneously exploiting the advantages of each of them in the corresponding Mach-number regimes. 

## **2.1 Conservative and Primitive Formulations** 

After suitable non-dimensionalization and rescaling, the two-dimensional (2-D) compressible Euler equations can be written in the conservative form as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0003-12.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

4 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-02.png)


Here, _ρ_ , _**u**_ := ( _u, v_ ) _[⊤]_ , and _E_ denote the density, velocity, and total energy, respectively, _p_ is the pressure, _ε_ is the reference Mach number, and the system is closed by the equation of state, which, in the case of an ideal gas, reads as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-04.png)


with _γ_ being the specific heat ratio. This system is hyperbolic and features acoustic waves traveling with (maximum) speed _|_ _**u** |_ + _c_ , where _c_ is the speed of sound given by _c_ :=[1] _ε_ � _γp/ρ_ . Notice that in low-Mach-number regimes, the acoustic waves travel at a high (maximum) speed proportional to 1 _/ε_ . 

For the purpose of deriving our AP scheme, we also consider an equivalent nonconservative formulation of the system (2.1)–(2.4) in terms of the primitive variables _ρ_ , _**u**_ , and _p_ : 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-07.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-08.png)


We emphasize that this formulation is equivalent to the conservative system (2.1)–(2.4) only for smooth solutions, and numerical approximations of (2.5)–(2.7) typically converge to nonphysical solutions when discontinuities are present; see [4,27] for a detailed discussion. 

## **2.2 Zero-Mach-Number Limit** 

It is well-known (see, e.g., a formal analysis in [6, 37]) that in the zero-Mach-number limit the compressible Euler equations reduce to the incompressible ones. To illustrate this, we examine the formal behavior of the primitive system (2.5)–(2.7) as _ε →_ 0. We substitute the formal expansions 

_ρ_ = _ρ_[(0)] + _ερ_[(1)] + _ε_[2] _ρ_[(2)] + _. . . ,_ _**u**_ = _**u**_[(0)] + _ε_ _**u**_[(1)] + _ε_[2] _**u**_[(2)] + _. . . , p_ = _p_[(0)] + _εp_[(1)] + _ε_[2] _p_[(2)] + _. . ._ into (2.5)–(2.7) and collect terms by powers of _ε_ . This yields 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-13.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-14.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-15.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-16.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-17.png)


It follows from (2.8)–(2.9) that _p_[(0)] ( _x, y, t_ ) = _p_[(0)] ( _t_ ) and _p_[(1)] ( _x, y, t_ ) = _p_[(1)] ( _t_ ) are spatially uniform. One can also show that both _p_[(0)] and _p_[(1)] are independent of time, provided the following Dirichlet boundary condition holds: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0004-19.png)


A New AP Method 

5 

where _p_ 0 _>_ 0 and _p_ 1 are constants, _pk_ ( _x, y, t_ ), _k ≥_ 2 are bounded functions, and Ωis the spatial domain with boundary _∂_ Ω. In fact, such a boundary condition implies _p_[(0)] _≡ p_ 0 and _p_[(1)] _≡ p_ 1. Using this in (2.12), one concludes that _**∇** ·_ _**u**_[(0)] = 0, which further implies _**∇** ·_ _**u**_[(1)] = 0 thanks to (2.13). Hence, the zero-Mach-number limiting equations are 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-03.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-04.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-05.png)


and the correct low-Mach-number scaling for the _O_ ( _ε_ ) terms is 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-07.png)


## **2.3 Dual Formulation (DF) Framework** 

As outlined above, a new AP scheme for the compressible Euler equations will be constructed within the DF framework, in which both conservative and primitive formulations are numerically solved simultaneously. The conservative form ensures proper handling of discontinuities, while the primitive form is used to achieve the AP property in the zero-Mach-number limit. While the DF methodology is not specifically developed to handle multiscale features, it lays the groundwork for the AP scheme developed in subsequent sections by enabling an efficient and accurate treatment of the studied Euler equations in both compressible and nearly incompressible flow regimes. 

We consider a general 2-D hyperbolic system of conservation laws, 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-11.png)


where _**U**_ is the vector of conservative variables and _**F**_ and _**G**_ are fluxes, and rewrite it in an equivalent nonconservative form 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-13.png)


where _**V**_ is the vector of nonconservative variables, _**F**_[�] and _**G**_[�] are the corresponding fluxes, and _B_ ( _**V**_ ) _**V** x_ and _C_ ( _**V**_ ) _**V** y_ are the nonconservative product terms. 

The key idea of the DF approach is to evolve the solutions of (2.19) and (2.20) simultaneously. A crucial step in DF-based methods is a post-processing, in which the evolved values of the nonconservative solution _**V**_ are replaced with a more reliable approximation after the update. This step is necessary because long-term evolutions of _**V**_ by directly solving the nonconservative system (2.20) may lead to nonphysical solutions in the presence of discontinuities, which typically appear when the studied Euler system is considered in the compressible (large/medium-Machnumber) regime. 

The post-processing can be described as follows. After advancing the solutions of (2.19) and (2.20) from a certain time level _t_ to the next time level _t_ + ∆ _t_ , the evolved values of _**V**_ ( _t_ + ∆ _t_ ) are replaced with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-17.png)


where _r_ is a suitable replacement function and _**V**_ ( _**U**_ ) is a conservative-primitive variable transformation. In the simplest nonstiff case, one can set 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0005-19.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

6 

However, in the development of the AP scheme below, we will modify the post-processing (2.22) by taking an appropriate function _r_ in (2.21) to ensure that in the nearly incompressible (lowMach-number) regime the AP _**V**_ -solution is not overwritten by the non-AP conservative one. 

## **3 Novel AP Scheme for Compressible Euler Equations** 

Building on the DF framework described in _§_ 2.3, we now present a novel AP scheme designed for the compressible Euler equations across all Mach-number regimes, from fully compressible to nearly incompressible flows. The proposed method couples the conservative (2.1)–(2.4) and primitive (2.5)–(2.7) formulations of the system, ensuring stability, accuracy, and consistency with the analytical asymptotic behavior as _ε →_ 0. 

In this section, we provide a complete description of the proposed space-time discretization, starting with the primitive system (2.5)–(2.7). In _§_ 3.1, we outline its temporal integration, which is based on a new hyperbolic splitting and an SI approach. In _§_ 3.2, we present a fully discrete second-order AP scheme for the primitive system, and in _§_ 3.3, we describe the semi-discrete CU scheme employed for the conservative system. _§_ 3.4 is devoted to the clarification of important implementation details. Finally, in _§_ 3.5, we present the Mach-number dependent post-processing strategy used to reconcile primitive and conservative variables. 

## **3.1 Novel AP Time Discretization of the Primitive System** 

We begin by providing a precise definition of an asymptotic-preserving (AP) time discretization in the context of the zero-Mach-number limit. 

**Definition 3.1 (AP time discretization)** _Assume that the computed solution at time t[n] can be expanded as_ 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0006-09.png)


_which is compatible with the asymptotic limits (2.17) and (2.18), that is,_ 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0006-11.png)


_and satisfies the Dirichlet boundary condition (2.14), namely,_ 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0006-13.png)


_Let us consider a one-step time discretization that produces at time t[n]_[+1] = _t[n]_ + ∆ _t an approximation_ ( _ρ[n]_[+1] _,_ _**u**[n]_[+1] _, p[n]_[+1] ) _. We say that such time discretization is AP if it admits an asymptotic expansion of the type (3.1) and yields a consistent discretization of (2.15)–(2.18) as ε →_ 0 _._ 

To construct an AP time discretization, we first perform a hyperbolic splitting of the primitive system (2.5)–(2.7), separating the stiff pressure-driven terms from the nonstiff convective terms. This splitting enables the use of an SI integration strategy in which the stiff and nonstiff terms are treated semi-implicitly and explicitly, respectively, ensuring uniform stability and asymptotic consistency as _ε →_ 0. 

A New AP Method 

7 

## **3.1.1 A New Hyperbolic Splitting** 

We follow the idea from [26,36] and split the nonconservative system into two parts corresponding to the slow and fast dynamics as follows. We first define the time-dependent variables 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-04.png)


and then add and subtract 1[and] _[γp]_[min] _**[∇]**[·]_ _**[ u]**_[from][(2.6)][and][(2.7),][respectively,][to][rewrite] _ε_[2] _ρ_ max _[∇][p]_ system (2.5)–(2.7) as follows: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-06.png)


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-07.png)


This system can be put in the following vector form: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-09.png)


where _**V**_ := ( _ρ, u, v, p_ ) _[⊤]_ , the nonlinear nonstiff (slow dynamics) part consists of the fluxes 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-11.png)


and the nonstiff nonconservative terms _B_[�] ( _**V**_ ) _**V** x_ + _C_[�] ( _**V**_ ) _**V** y_ with matrices 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-13.png)


while,� the linear stiff (fast dynamics) part consists of the stiff nonconservative terms _B_[�] ( _**V**_ ) _**V** x_ + _C_ ( _**V**_ ) _**V** y_ with matrices 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-15.png)


We stress that the subsystem _**V** t_ + _**F**_[�] ( _**V**_ ) _x_ + _**G**_[�] ( _**V**_ ) _y_ = _B_[�] ( _**V**_ ) _**V** x_ + _C_[�] ( _**V**_ ) _**V** y_ is indeed nonstiff _∂_ _**F**_[�] _∂_ _**G**_[�] as the eigenvalues of the matrices _∂_ _**V**_[(] _**[V]**_[ )] _[−][B]_[�][(] _**[V]**_[ )][and] _∂_ _**V**_[(] _**[V]**_[ )] _[−][C]_[�][(] _**[V]**_[ ),][are] _[{][u][ ±]_[ ˜] _[c, u, u][}]_[and] _{v ±_ ˜ _c, v, v}_ , respectively, with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0007-17.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

8 

which are real and of size _O_ (1) thanks to the definitions of _ρ_ max and _p_ min in (3.5) and to the asymptotic analysis in _§_ 2.2, which ensure that 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0008-03.png)


In the next subsection, we will utilize this splitting and design an AP time discretization based on an explicit approximation of the nonstiff subsystem and an SI discretization of the stiff terms on the right-hand sides (RHSs) of (3.7) and (3.8). 

## **3.1.2 First-Order AP SI Time Discretization** 

The simplest first-order AP SI time discretization of the system (3.9) reads as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0008-07.png)


which can also be written in the following vector form: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0008-09.png)


Notice that in (3.14), _**L**[n,n]_[+1] is discretized in an SI (rather than fully implicit) manner, where both _B_[�] and _C_[�] are evaluated at _**V**[n]_ (and not at _**V**[n]_[+1] ), which prevents from numerically solving complicated systems of nonlinear algebraic equations. 

We shall now prove that the time discretization (3.12) is indeed AP, provided that the time step is computed based on the wave speeds of the nonstiff subsystem, that is, provided that 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0008-12.png)


where _K_ CFL is a CFL number and ∆ _x_ and ∆ _y_ are mesh sizes used in the spatial discretization. Notice that selecting the time step ∆ _t_ according to (3.15) makes it asymptotically independent of ˜ _ε_ as, according to (3.10)–(3.11), _c_ = _O_ (1). 

A New AP Method 

9 

**Theorem 3.1** _The first-order SI time discretization (3.12) is AP according to Definition 3.1, provided that_ ∆ _t is computed as in (3.15)._ 

**Proof:** We begin by formally showing that the computed solution _ρ[n]_[+1] , _**u**[n]_[+1] , _p[n]_[+1] admits an expansion of the type (3.1) satisfying (3.2) and (3.3) in the limit as _ε →_ 0. We substitute the corresponding expansion of the numerical solution at time _t[n]_ into the scheme (3.12) and use (3.2)–(3.3) to obtain 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0009-04.png)


Thanks to the explicit nature of the density update in (3.16), we conclude that indeed _ρ[n]_[+1] admits the required asymptotic expansion _ρ[n]_[+1] = _ρ_[(0)] _[,n]_[+1] + _ερ_[(1)] _[,n]_[+1] + _ερ_[(2)] _[,n]_[+1] + _. . ._ , where the different terms of the expansion are obtained by collecting corresponding powers of _ε_ : 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0009-06.png)


In order to show that also _p[n]_[+1] admits an expansion of the same type, we take the divergence of the velocity equation (3.17), substitute _**∇** ·_ _**u**[n]_[+1] into the pressure equation (3.18), and use the divergence-free assumption (3.3) to obtain 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0009-08.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

10 

which implies that _p[n]_[+1] is the solution of the elliptic equation 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-03.png)


subject to the boundary condition (3.4). According to the theory of perturbed linear operators [29], one can conclude that 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-05.png)


and hence 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-07.png)


which we substitute into (3.17) to obtain the velocity expansion 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-09.png)


with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-11.png)


We now need to show that (3.2) and (3.3) hold for the updated solution, along with the consistency of the scheme (3.12) with (2.15) and (2.16) as _ε →_ 0. We have already shown that (3.2) holds; see (3.21). The divergence-free conditions (3.3) can be deduced from the pressure update (3.18), which in view of the obtained results yields 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-13.png)


Collecting the power-like terms of _ε_ , we deduce _**∇** ·_ _**u**_[(0)] _[,n]_[+1] = _**∇** ·_ _**u**_[(1)] _[,n]_[+1] = 0. 

The consistency with (2.15) immediately follows from the first equation in (3.19). To show the consistency with (2.16), we rewrite the first equation in (3.22) as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0010-16.png)


which is a consistent discretization of (2.16). 

We remark that since the zeroth and first modes of the pressure are constant, the evolution of the pressure in the zero-Mach-number limit essentially consists of the evolution of the second mode. Thus, _∇p_[(2)] _[,n]_[+1] _−∇p_[(2)] _[,n] ≈O_ (∆ _t_ ) and the last term in (3.23), in fact, represents a temporal diffusion term, which is proportional to _O_ ((∆ _t_ )[2] ). We also remark that according to (3.15), the time step ∆ _t_ is asymptotically independent of _ε_ . ■ 

A New AP Method 

11 

## **3.1.3 Second-Order AP SI Time Discretization** 

We now introduce a second-order AP SI time discretization, which is based on the DeC approach, which was originally introduced in [21]. Our second-order AP SI-DeC time discretization is directly related to the IMEX-DeC methods presented in [42] and based on the DeC formulation introduced in [1]; see also [38,39]. 

According to the second-order AP SI-DeC time discretization, the solution of (3.9) is evolved from _t_ = _t[n]_ to _t_ = _t[n]_[+1] through the following two stages: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0011-05.png)


where the upper index _∗_ is associated with the intermediate solution _**V**[∗]_ , and the definitions of the operators _**R**[∗]_ and _**L**[n,n]_ , _**L**[∗][,][∗]_ , and _**L**[∗][,n]_[+1] are analogous to those given in (3.13) and (3.14), respectively. 

The scheme (3.24) can be equivalently written as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0011-08.png)


and 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0011-10.png)


This second-order time discretization is indeed AP as shown in the next theorem. 

**Theorem 3.2** _The second-order SI-DeC discretization (3.25)–(3.26) is AP according to Definition 3.1, provided that_ ∆ _t is computed according to (3.15)._ 

**Proof:** The proof proceeds along the same lines and uses the same arguments as in the proof of Theorem 3.1. 

We begin by observing that the first stage of the second-order SI-DeC discretization coincides with the first-order AP SI time discretization studied before. Therefore, according to Theorem 3.1, the intermediate solution _ρ[∗]_ , _**u**[∗]_ , _p[∗]_ , obtained by (3.25) admits an expansion of the type (3.1), that is, 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0011-15.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

12 

with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0012-03.png)


We then substitute the expansions (3.1) and (3.27) into (3.26) and use the conditions (3.2)–(3.3) and (3.28) to obtain 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0012-05.png)


The explicit nature of the density update (3.29) implies that _ρ[n]_[+1] admits the required expansion _ρ[n]_[+1] = _ρ_[(0)] _[,n]_[+1] + _ερ_[(1)] _[,n]_[+1] + _ε_[2] _ρ_[(2)] _[,n]_[+1] + _. . ._ with _ρ_[(0)] _[,n]_[+1] satisfying 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0012-07.png)


and other coefficients satisfying the equations, which can be easily obtained by grouping the corresponding powers of _ε_ . 

A New AP Method 

13 

As in the proof of Theorem 3.1, we show that _p[n]_[+1] admits the asymptotic expansion by proving that it satisfies a well-posed elliptic problem with suitable boundary conditions. Taking the divergence of the velocity equation (3.30) and substituting _**∇** ·_ _**u**[n]_[+1] into the pressure equation (3.31) yields 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0013-03.png)


This together with the boundary conditions (2.14), results in the same expansion for _p[n]_[+1] , which we have established in (3.20)–(3.21) for the first-order SI method, leading to 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0013-05.png)


Next, we substitute (3.33) into the the velocity equation (3.30) and a straightforward grouping of the power-like terms of _ε_ gives the equations for the coefficients of the velocity expansion _**u**[n]_[+1] = _**u**_[(0)] _[,n]_[+1] + _ε_ _**u**_[(1)] _[,n]_[+1] + _ε_[2] _**u**_[(2)] _[,n]_[+1] + _. . ._ . The equation for _**u**_[(0)] _[,n]_[+1] is 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0013-07.png)


and the other equations can be obtained similarly. 

Let us now show the consistency with the asymptotic limit. The required conditions (3.21) on the pressure modes have been already shown. The divergence-free conditions for the velocity modes are then established from the pressure update (3.31), which becomes 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0013-10.png)


It is clear that the _O_ (1) and _O_ ( _ε_ ) terms here vanish, that is, _**∇** ·_ _**u**_[(0)] _[,n]_[+1] = _**∇** ·_ _**u**_[(1)] _[,n]_[+1] = 0. 

Finally, we notice that (3.32) and (3.34) are consistent discretizations of (2.15) and (2.16) with the last term in (3.34) representing a temporal diffusion, which is consistent with the order of accuracy of the scheme and thus proportional to _O_ ((∆ _t_ )[3] ). ■ 

**Remark 3.1** _The described AP SI-DeC time discretization can be extended to arbitrarily high order in a straightforward way within the DeC framework. For the sake of brevity, we restrict our consideration to the second order of accuracy, which matches the accuracy that will be used in the spatial discretization discussed in §3.2._ 

## **3.2 Fully Discrete Second-Order AP Scheme for the Primitive System** 

In this section, we construct a fully discrete scheme based on the second-order AP SI time discretization presented in _§_ 3.1.3. To this end, we first introduce uniform Cartesian cells _Ij,k_ := [ _xj−_ 12 _[, x][j]_[+][1] 2[]] _[ ×]_[ [] _[y][k][−]_[1] 2 _[, y][k]_[+][1] 2[]][with] _[x][j]_[+][1] 2 _[−][x][j][−]_[1] 2 _[≡]_[∆] _[x]_[and] _[y][k]_[+][1] 2 _[−][y][k][−]_[1] 2 _[≡]_[∆] _[y]_[,][centered][at][(] _[x][j][, y][k]_[)] with _xj_ = � _xj−_ 21[+] _[x][j]_[+][1] 2 � _/_ 2 and _yk_ = � _yk−_ 12[+] _[y][k]_[+][1] 2 � _/_ 2, and assume that the cell averages _n_ 1 _**V** j,k_[:] _[≈]_ ∆ _x_ ∆ _y_ �� _Ij,k_ _**[V]**_[ (] _[x, y, t][n]_[) d] _[x]_[d] _[y]_[are][available][at][time] _[t][n]_[.] 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

14 

The fully discrete FV version of the second-order AP scheme (3.24) reads as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-03.png)


where _**R**[n] j.k_[and] _**[R]**[∗] j.k_[are][obtained][using][the][PCCU][discretization][from][[3, 15],][which][is][a][low-] dissipation generalization of the PCCU discretization from [12], while _**L**[n,] j.k[∗]_[,] _**[ L]**[n,n] j.k_[,] _**[ L]**[∗] j.k[,][∗]_[, and] _**[ L]**[∗] j.k[,n]_[+1] are obtained using central differences. In what follows, for the sake of brevity, we provide details on _**R**[n] j.k_[and] _**[L]**[n,] j.k[∗]_[only,][whereas][the][remaining][discretizations][are][obtained][in][a][similar][manner.] We begin with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-05.png)


where _**F**_[�] _nj_ +[1] 2 _[,k]_[and] _**[G]**_[�] _nj,k_ + 2[1][are][the][CU][numerical][fluxes] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-07.png)


and _**V**[±][,n][±][,n]_[at the midpoints of the cell interfaces,] _[ δ]_ _**[V]**[n] j_ +[1] 2 _[,k]_[and] _**[ V]** j,k_ + 2[1][are reconstructed values of] _**[ V]** j_ + 2[1] _[,k]_ and _δ_ _**V**[n]_[terms,][and] _[a]_[˜] _[±][,n][a]_[˜] _[±][,n]_[the][one-sided][local] _j,k_ +[1] 2[are “built-in” anti-diffusion] _j_ +[1] 2 _[,k]_[and] _j,k_ +[1] 2[denote] propagation speeds of the nonstiff subsystem in the _x_ - and _y_ -direction, respectively. The point values 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-09.png)


are computed using the piecewise linear reconstruction 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-11.png)


in which the slopes ( _**V** x_ ) _[n] j,k_[and][(] _**[V]**[y]_[)] _[n] j,k_[are][approximated][using][the][generalized][minmod][limiter] (see, e.g., [35,40,44]): 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0014-13.png)


A New AP Method 

15 

where the minmod function, defined by 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0015-03.png)


is applied in a componentwise manner. The parameter _θ ∈_ [1 _,_ 2] in (3.39) is to be chosen to adjust the amount of numerical dissipation present in the resulting scheme, with larger values of _θ_ leading to sharper but, in general, more oscillatory solutions. 

The one-sided local speeds of propagation are estimated using the smallest and largest eigenvalues of the matrices _∂[∂]_ _**VF**_[�][(] _**[V]**_[ )] _[ −][B]_[�][(] _**[V]**_[ )][and] _∂[∂]_ _**VG**_[�][(] _**[V]**_[ )] _[ −][C]_[�][(] _**[V]**_[ )][as][follows:] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0015-06.png)


where the sound speeds 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0015-08.png)


and _δ_ is a small positive parameter introduced to prevent divisions by 0 (we have taken _δ_ := 10 _[−]_[15] in the numerical experiments reported in _§_ 4). 

The “built-in” anti-diffusion terms are 

where 

Finally, 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0015-13.png)


where _**∇** ·_ _**u** j,k_ denotes the discrete divergence operator computed using second-order central differences: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0015-15.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

16 

## **3.3 Semi-Discrete CU Scheme for the Conservative System** 

We now consider the conservative formulation (2.1)–(2.3), which can be put into the following vector form: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0016-04.png)


1 In the semi-discrete CU scheme, the cell averages _**U** j,k_ ( _t_ ) : _≈_ ∆ _x_ ∆ _y_ �� _Ij,k_ _**[U]**_[(] _[x, y, t]_[) d] _[x]_[d] _[y]_[are] evolved in time by numerically solving the following system of ODEs: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0016-06.png)


where _**F** j_ + 21 _[,k]_[and] _**[G]**[j,k]_[+] 2[1][are][the][CU][numerical][fluxes][from][[32]][defined][as] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0016-08.png)


Here, the interface values _**U** j[±]_ +[1] 2 _[,k]_[:=] _**[U]**_ � _**V** j[±]_ +[1] 2 _[,k]_ � and _**U** j,k[±]_ +[1] 2[:=] _**[U]**_ � _**V** j,k[±]_ +[1] 2 � are computed from the reconstructed primitive variables _**V** j[±]_ + 2[1] _[,k]_[and] _**[V]** j,k[±]_ + 2[1][(see] _[§]_[3.2)][at][the][corresponding][time][level][via] a straightforward transformation _**U**_ ( _**V**_ ) from _**V**_ to _**U**_ . The quantities _a[±] j_ +[1] 2 _[,k]_[and] _[b][±] j,k_ + 2[1][are][the] one-sided local speeds of propagation for the conservative system (3.44) in the _x_ - and _y_ -direction, respectively. They are estimated using the largest and smallest eigenvalues of the corresponding flux Jacobians as follows: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0016-10.png)


where _δ_ := 10 _[−]_[15] is used to avoid divisions by 0. 

A New AP Method 

17 

The “built-in” anti-diffusion terms are 

with 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-04.png)


Note that most of the indexed quantities in the semi-discrete setting above are time-dependent, but we have omitted this dependence to ease the notation. 

Finally, the system of ODEs (3.45) has to be integrated in time using an appropriate ODE solver. Its solution is performed simultaneously with one of the primitive systems using the explicit counterpart of the SI-DeC scheme, and a post-processing is performed at each stage, as explained in _§_ 3.4. 

## **3.4 Implementation Details** 

In our DF-FV approach, the solutions of the primitive and conservative systems are evolved simultaneously according to the following algorithm. 

_•_ **Step 1 (Compute** _ρj,k[∗]_ **[).]**[We][use][the] _[ρ]_[-equation][in][(3.35)][to][obtain] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-10.png)


_•_ **Step 2 (Solve the linear elliptic equation for** _pj,k[∗]_ **[).]**[We][apply][the][discrete][divergence][operator] (3.43) to the _**u**_ -equations in (3.35) and substitute them into the _p_ -equation in (3.35) to obtain the following linear system of algebraic equations for _pj,k[∗]_[,][which][is][a][discretization][of][the][linear] elliptic equation for _p[∗]_ : 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-12.png)


where the discrete Laplacian ∆ _pj,k_ is defined as 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-14.png)


_•_ **Step 3 (Compute** _**u** j,k[∗]_ **[).]**[Once] _pj,k[∗]_[is][available,][we][use][the] _**[u]**_[-equations][in][(3.35)][to][obtain] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-16.png)


_∗ •_ **Step 4 (Compute** _**U** j,k_ **[).]**[We][perform][the][conservative][update][with][the][explicit][counterpart][of] _∗_ the SI-DeC scheme to obtain the solution _**U** j,k_[at][the][intermediate][stage] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0017-18.png)


A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

18 

_∗ ∗ ∗_ and then post-process the primitive solution by replacing _**V** j,k_[with] _[r]_ _**V**_ � _**U** j,k_ � _,_ _**V** j,k_ �; see _§_ 3.5. � 

_•_ **Step 5 (Compute** _ρj,k[n]_[+1] **[).]**[We][solve][the] _[ρ]_[-equation][in][(3.36)][to][obtain] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0018-04.png)


_•_ **Step 6 (Solve the linear elliptic equation for** _pj,k[n]_[+1] **[).]**[We][apply][the][discrete][divergence][operator] (3.43) to the _**u**_ -equations in (3.36) and substitute them into the _p_ -equation in (3.36) to obtain the following linear system of algebraic equations for _pj,k[n]_[+1][,][which][is][a][discretization][of][the][linear] elliptic equation for _p[n]_[+1] : 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0018-06.png)


_•_ **Step 7 (Compute** _**u** j,k[n]_[+1] **[).]**[Once] _pj,k[n]_[+1] is available, we compute 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0018-08.png)


_n_ +1 _•_ **Step 8 (Compute** _**U** j,k_ **[).]**[Finally,][we][use][the][explicit][part][of][the][SI-DeC][scheme][to][evaluate] 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0018-10.png)


_n_ +1 _n_ +1 _n_ +1 and then post-process of the primitive solution by replacing _**V** j,k_ with _r_ � _**V**_ � _**U** j,k_ � _,_ _**V** j,k_ �; see _§_ 3.5. 

We recall that the interface values of _**U**_ needed for the computation of the numerical fluxes in (3.50) and (3.51) are obtained from the reconstructed primitive variables _**V**_ at the corresponding time levels. Note that the conservative updates are, as a matter of fact, explicit, since they are performed using the explicit part of the SI-DeC scheme. 

## **3.5 Post-Processing** 

Upon completion of Steps 4 and 8, we obtain two sets of numerical solutions: the _**V**_ -solution, which is AP but nonconservative, and the _**U**_ -solution, which is conservative but non-AP. We therefore design the post-processing (2.21) using their convex combination with coefficients dependent on _ε_ , leveraging the AP SI method in the low-Mach-number regime and the sharp conservative CU scheme in the moderateand high-Mach-number regimes—thus ensuring accuracy, stability, and physical consistency across all flow regimes. Specifically, we select the following function _r_ in (2.21): 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0018-15.png)


A New AP Method 

19 

where _s_ is a suitable switching function, which is supposed to be increasing, continuous, and satisfy _s_ (1) = 0 and _s_ (0) = 1. Moreover, in the high-Mach-number regime, _s_ should be _∼_ 0 so that the primitive variables _**V**_ are almost completely overwritten by _**V**_ ( _**U**_ ), while, in the low-Mach-number regime, _s_ should be _∼_ 1 so that the primitive variables _**V**_ stay almost unchanged. For intermediate values of _ε_ , a smooth transition between 1 and 0 is expected. 

## **4 Numerical Examples** 

In this section, we verify the accuracy and robustness of the proposed AP scheme on a variety of numerical examples across different values of _ε_ . In all of the numerical examples, we: 

- Take the minmod parameter _θ_ = 1 _._ 3; 

- Adaptively select time steps based on the time-step restriction (3.15) for the nonstiff part of the 

- primitive system; 

- Set _γ_ = 1 _._ 4 (except for Example 1, in which _γ_ = 2); 

- Modify (3.5) to 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0019-09.png)


Notice that this modification has almost no impact in the low-Mach-number regime, but it aims at adding more upwinding and thus improving the stability property of the resulting AP scheme when _ε_ is large; 

- Choose the following switching function: 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0019-12.png)


where _ε_ 0, _ε_ 1, and _α_ are positive constants taken to be _ε_ 0 = 0 _._ 15, _ε_ 1 = 0 _._ 4, and _α_ = 14 in all of the numerical examples below. This switching function is plotted in Figure 4.1. 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0019-14.png)


<br>


Figure 4.1: Switching function _s_ ( _ε_ ) plotted with respect to 1 _/ε_ . 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

20 

## **Example 1—Accuracy Test for Low-Mach-Number Smooth Vortex** 

In this example taken from [49], we consider a smooth, unsteady Mach dependent vortex over the computational domain [ _−_ 10 _,_ 10] _×_ [ _−_ 10 _,_ 10] subject to the periodic boundary conditions. The analytical solution is given, modulo the periodicity, by 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0020-04.png)


where _**x** r_ ( _x, y, t_ ) = ( _xr, yr_ ) _[⊤]_ := ( _x − t, y − t_ ) _[⊤]_ . 

We take the CFL number _K_ CFL = 0 _._ 475 and compute the numerical solution until the final time _t_ = 0 _._ 1 on a series of uniform _N × N_ meshes with _N_ = 64, 128, 256, and 512 for _ε_ = 1, 0 _._ 1, 0 _._ 01, and 0 _._ 001. We study the convergence and present the obtained results in Figure 4.2, where one can see that the expected second-order convergence rate has been achieved in all variables for all considered _ε_ . One can also observe that, for fixed mesh refinement, the error decreases for decreasing _ε_ as a result of the convergence of the analytical solution to the incompressible limit ( _ρ, u, v, p_ ) _[⊤]_ = (1 _,_ 1 _,_ 1 _,_ 1) _[⊤]_ and of the AP character of the proposed AP DF-FV scheme. 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0020-07.png)


<br>


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0020-08.png)


<br>


Figure 4.2: Example 1: Convergence analysis. 

A New AP Method 

21 

## **Example 2—Gresho Vortex** 

This example was introduced in [25] and, since then, it has been widely used as a common benchmark to numerically validate the AP property. We consider a steady vortex over the computational domain [0 _,_ 1] _×_ [0 _,_ 1] subject to the periodic boundary conditions. At any time _t_ , the shape of the vortex is given by 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0021-04.png)


where 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0021-06.png)


We take the CFL number _K_ CFL = 0 _._ 475 and compute the numerical solution until the final time _t_ = 1 on a uniform 128 _×_ 128 mesh for _ε_ = 10 _[−][α]_ with _α_ = 1 _, . . . ,_ 6, and report the obtained local Mach number, defined as _∥_ _**u** ∥_ 2 _/[√] γ_ , in Figure 4.3 along with its initial distribution. According to what is expected due to the AP feature of the scheme, the shape of the vortex is preserved and no evident dependency on _ε_ can be observed. 

## **Example 3—Baroclinic Vorticity Generation** 

In this example taken from [41], we consider a low-Mach-number flow with _ε_ = 0 _._ 05 involving an acoustic wave, which moves within two density layers in the computational domain [ _−_[1] _ε[,]_[1] _ε_[]] _[×]_[[0] _[,]_ 5[2] _ε_[]] subject to the periodic boundary conditions. The acoustic wave induces different accelerations in the two density layers, which results in rotational excitation and in the formation of a longwavelength sinusoidal shear layer. Due to the interaction with the acoustic wave, such a shear layer becomes unstable, and several Kelvin-Helmholtz-type unstable structures originate from it. The initial conditions are 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0021-10.png)


The numerical solution is computed with the CFL number _K_ CFL = 0 _._ 475 until the final time _t_ = 20 on a 800 _×_ 160 uniform mesh. The density at times _t_ = 0, 10, and 20 is plotted in Figure 4.4. Since the solution develops instabilities, no strong convergence is expected in this example; see [49]. One can, however, observe that the underlying physics is correctly captured. 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

22 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0022-02.png)


<br>


Figure 4.3: Example 2: Initial local Mach number independently of _ε_ (top) and local Mach number at _t_ = 1 for different values of _ε_ . 

## **Example 4—Double Shear Layer Problem** 

In the following test case, originally introduced in [8] for the incompressible Navier–Stokes equations and subsequently adopted in, e.g., [10,48,49] in the context of compressible Euler equations in the low-Mach-number regime, a shear layer develops, and the AP property of the proposed scheme can be assessed. In particular, we would like to check whether the scheme maintains its consistency for small values of _ε_ , that is, in the almost incompressible regime. 

The initial conditions, 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0022-07.png)


A New AP Method 

23 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0023-02.png)


<br>


Figure 4.4: Example 3: Density at different times. 

_p_ ( _x, y,_ 0) _≡_[1] _γ[,]_ 

are prescribed in the computational domain [0 _,_ 2 _π_ ] _×_ [0 _,_ 2 _π_ ] subject to the periodic boundary conditions. The initial vorticity _ω_ := _vx − uy_ , where the derivatives are approximated using second-order central differences, is plotted in Figure 4.5. 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0023-06.png)


<br>


Figure 4.5: Example 4: Initial vorticity. 

We compute the numerical solutions for _ε_ = 10 _[−][α]_ with _α_ = 1 _, . . . ,_ 6 until the final time _t_ = 10 on a 256 _×_ 256 uniform mesh using _K_ CFL = 0 _._ 1. Figures 4.6 and 4.7 display the vorticity at times _t_ = 6 and _t_ = 10, respectively, for different _ε_ . The obtained results are consistent with those 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

24 

reported in [10, 49]. Moreover, no macroscopic dependence on _ε_ is observed, providing further evidence of the AP property of the proposed DF-FV scheme. 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0024-03.png)


<br>


Figure 4.6: Example 4: Vorticity at _t_ = 6 for different values of _ε_ . _K_ CFL = 0 _._ 1. 

We remark that the simulations remain stable for larger CFL numbers. However, the use of larger _K_ CFL may lead to a noticeable increase in the amount of the numerical diffusion for very small values of _ε_ ≲ 10 _[−]_[3] . To illustrate this, we recompute the solution with _K_ CFL = 0 _._ 475 for _ε_ = 10 _[−]_[4] , 10 _[−]_[5] , and 10 _[−]_[6] and plot the obtained results (for _t_ = 6) in Figure 4.8. As one can clearly see, the numerical solution is now substantially more diffusive compared with those reported in the bottom ’row of Figure 4.6. 

## **Example 5—Explosion Problem** 

In the last numerical example, we consider an explosion problem taken from [46]. The initial data, 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0024-08.png)


are prescribed in the computational domain [ _−_ 1 _,_ 1] _×_ [ _−_ 1 _,_ 1] subject to the free boundary conditions. 

The main objective of this test is to verify that the proposed AP DF-FV scheme remains accurate and stable, also in the high-Mach-number regime, in which strong shocks and contact discontinuities may be present. To this end, we perform simulations for several values of _ε_ . For _ε_ = 1, 0 _._ 9, 0 _._ 6, and 0 _._ 3 the final times are _t_ = 0 _._ 25, 0 _._ 2, 0 _._ 15, and 0 _._ 08, respectively. The surface plots of the density _ρ_ computed on a uniform mesh with 400 _×_ 400 cells using _K_ CFL = 0 _._ 475 are 

A New AP Method 

25 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0025-02.png)


<br>


Figure 4.7: Example 4: The same as in Figure 4.6, but at _t_ = 10. 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0025-04.png)


<br>


Figure 4.8: Example 4: The same as in Figure 4.6, but for _K_ CFL = 0 _._ 475 (left). 

reported in Figure 4.9, where one can see that the obtained solutions are oscillation-free and their nonsmooth features are accurately resolved for all values of _ε_ . 

To further assess the correctness of the computed solutions, we plot their one-dimensional (1-D) slices along the diagonal _y_ = _x_ in Figure 4.10 together with the corresponding slices of the reference solution, which was obtained using the second-order semi-discrete CU scheme from [32] on a much finer mesh with 2000 _×_ 2000 cells using the CFL number 0 _._ 2 and the three-stage third-order strong stability preserving (SSP) Runge-Kutta method [23, 24]. As one can clearly see, the computed solutions show a perfect agreement with the reference ones, and the discontinuities locations are correctly captured. 

**Remark 4.1** _We stress that in this example, both terms in the numerator in (3.10) will vanish_ 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

26 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0026-02.png)


Figure 4.9: Example 5: Surface plot of density for different values of _ε_ and corresponding times. 

_if ρ_ max _and p_ min _are computed using (3.5). While the modification (4.1) ensures positivity of c_ ˜ _, the resulting time steps might still be too big to guarantee stability of the AP DF-FV method. Therefore, we set_ ∆ _t_ = 10 _[−]_[4] _for the first 10 time steps for the simulations involving ε_ = 0 _._ 6 _and_ 0 _._ 3 _._ 

**Remark 4.2** _Let us remark that discontinuities are unlikely to occur in low-Mach-number flows. Consequently, the above tests with ε_ = 0 _._ 6 _and_ 0 _._ 3 _should be regarded as “academic” and are primarily intended to demonstrate that the proposed AP DF-FV scheme is capable of handling discontinuities even in the low-Mach-number regime._ 

## **5 Conclusion** 

We have presented a novel asymptotic-preserving (AP) numerical method for the compressible Euler equations that is effective across all Mach-number regimes, including the low-Mach-number one, where standard explicit schemes become inefficient. The key idea is a new hyperbolic splitting, inspired by the flux-splitting approach introduced in [26]. The new splitting is applied to a primitive (nonconservative) formulation of the Euler equations, which enables one to design an 

A New AP Method 

27 


![](images/-2026-_AP_+_FVM_+_comp_+_Euler.pdf-0027-02.png)


<br>


Figure 4.10: Example 5: 1-D slices of the computed solutions along _y_ = _x_ for different values of _ε_ and at different times: _ε_ = 1, _t_ = 0 _._ 25 (top row), _ε_ = 0 _._ 9, _t_ = 0 _._ 2 (second row), _ε_ = 0 _._ 6, _t_ = 0 _._ 15 (third row), and _ε_ = 0 _._ 3, _t_ = 0 _._ 08 (bottom row). 

efficient semi-implicit (SI) time discretization. Our splitting isolates stiff linear terms, which are discretized semi-implicitly: this leads to a well-posed linear elliptic problem, which ensures the AP property of the resulting scheme. 

To overcome the well-known difficulties associated with the use of nonconservative formulations in the presence of discontinuities, we implement the proposed AP scheme within the recently introduced dual formulation framework [3, 14]. In this approach, the conservative and primitive 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

28 

systems are solved simultaneously, and their resulting solutions are post-processed to ensure the correct capturing of discontinuities while retaining the AP property of the primitive-based SI approach. 

The proposed AP dual formulation finite-volume (DF-FV) method has been thoroughly validated on several benchmarks ranging from the fully compressible to the nearly incompressible regime, demonstrating both high accuracy and robustness of the method. Future work will focus on extending the AP DF-FV framework to more complex systems and on developing higher-order spatial and temporal discretizations. 

**Acknowledgment:** The work of A. Chertock was supported in part by NSF grant DMS-2208438. The work of A. Kurganov was supported in part by NSFC grant W2431004. The work of L. Micalizzi was supported in part by the LeRoy B. Martin, Jr. Distinguished Professorship Foundation. 

## **References** 

- [1] R. Abgrall, _High order schemes for hyperbolic problems using globally continuous approximation and avoiding mass matrices_ , J. Sci. Comput., 73 (2017), pp. 461–494. 

- [2] ~~,~~ _A combination of residual distribution and the active flux formulations or a new class of schemes that can combine several writings of the same hyperbolic problem: application to the 1D Euler equations_ , Commun. Appl. Math. Comput., 5 (2023), pp. 370–402. 

- [3] R. Abgrall, A. Chertock, A. Kurganov, and L. Micalizzi, _Dual formulation finitevolume methods on overlapping meshes for hyperbolic conservation laws_ , Comput. & Fluids, 307 (2026). Paper No. 106952. 

- [4] R. Abgrall and S. Karni, _A comment on the computation of non-conservative products_ , J. Comput. Phys., 229 (2010), pp. 2759–2763. 

- [5] R. Abgrall and Y. Liu, _A new approach for designing well-balanced schemes for the shallow water equations: a combination of conservative and primitive formulations_ , SIAM J. Sci. Comput., 46 (2024), pp. A3375–A3400. 

- [6] T. Alazard, _Incompressible limit of the nonisentropic Euler equations with the solid wall boundary conditions_ , Adv. Differential Equations, 10 (2005), pp. 19–44. 

- [7] P. Allegrini and M.-H. Vignal, _Study of a new low-oscillating second-order all-Mach number IMEX finite volume scheme for the full Euler equations_ , SIAM J. Sci. Comput., 47 (2025), pp. A268–A299. 

- [8] J. B. Bell, P. Colella, and H. M. Glaz, _A second-order projection method for the incompressible Navier-Stokes equations_ , J. Comput. Phys., 85 (1989), pp. 257–283. 

- [9] S. Boscarino, J.-M. Qiu, G. Russo, and T. Xiong, _A high order semi-implicit IMEX WENO scheme for the all-Mach isentropic Euler system_ , J. Comput. Phys., 392 (2019), pp. 594–618. 

A New AP Method 

29 

- [10] S. Boscarino, G. Russo, and L. Scandurra, _All Mach number second order semiimplicit scheme for the Euler equations of gas dynamics_ , J. Sci. Comput., 77 (2018), pp. 850– 884. 

- [11] W. Boscheri, G. Dimarco, R. Loub`ere, M. Tavelli, and M.-H. Vignal, _A second order all Mach number IMEX finite volume solver for the three dimensional Euler equations_ , J. Comput. Phys., 415 (2020). Paper No. 109486. 

- [12] M. J. Castro D´ıaz, A. Kurganov, and T. Morales de Luna, _Path-conservative central-upwind schemes for nonconservative hyperbolic systems_ , ESAIM Math. Model. Numer. Anal., 53 (2019), pp. 959–985. 

- [13] C. Chalons, M. Girardin, and S. Kokh, _An all-regime Lagrange-projection like scheme for the gas dynamics equations on unstructured meshes_ , Commun. Comput. Phys., 20 (2016), pp. 188–233. 

- [14] A. Chertock, Q. Fu, A. Kurganov, and L. Micalizzi, _New adaptive numerical methods based on dual formulation of hyperbolic conservation laws_ . Submitted; arXiv:2601.20000. 

- [15] S. Chu, A. Kurganov, and M. Na, _Fifth-order A-WENO schemes based on the pathconservative central-upwind method_ , J. Comput. Phys., 469 (2022). Paper No. 111508. 

- [16] F. Cordier, P. Degond, and A. Kumbaro, _An asymptotic-preserving all-speed scheme for the Euler and Navier-Stokes equations_ , J. Comput. Phys., 231 (2012), pp. 5685–5704. 

- [17] P. Degond, S. Jin, and J.-G. Liu, _Mach-number uniform asymptotic-preserving gauge schemes for compressible flows_ , Bull. Inst. Math. Acad. Sin. (N.S.), 2 (2007), pp. 851–892. 

- [18] P. Degond and M. Tang, _All speed scheme for the low Mach number limit of the isentropic Euler equations_ , Commun. Comput. Phys., 10 (2011), pp. 1–31. 

- [19] G. Dimarco, R. Loub`ere, V. Michel-Dansac, and M.-H. Vignal, _Second-order implicit-explicit total variation diminishing schemes for the Euler system in the low Mach regime_ , J. Comput. Phys., 372 (2018), pp. 178–201. 

- [20] G. Dimarco, R. Loub`ere, and M.-H. Vignal, _Study of a new asymptotic preserving scheme for the Euler system in the low Mach number limit_ , SIAM J. Sci. Comput., 39 (2017), pp. A2099–A2128. 

- [21] L. Fox and E. T. Goodwin, _Some new methods for the numerical integration of ordinary differential equations_ , Proc. Cambridge Philos. Soc., 45 (1949), pp. 373–388. 

- [22] F. Golse, S. Jin, and C. D. Levermore, _The convergence of numerical transfer schemes in diffusive regimes. I. Discrete-ordinate method_ , SIAM J. Numer. Anal., 36 (1999), pp. 1333– 1369. 

- [23] S. Gottlieb, D. Ketcheson, and C.-W. Shu, _Strong stability preserving Runge-Kutta and multistep time discretizations_ , World Scientific Publishing Co. Pte. Ltd., Hackensack, NJ, 2011. 

A. Chertock, S. Karni, A. Kurganov & L. Micalizzi 

30 

- [24] S. Gottlieb, C.-W. Shu, and E. Tadmor, _Strong stability-preserving high-order time discretization methods_ , SIAM Rev., 43 (2001), pp. 89–112. 

- [25] P. M. Gresho and S. T. Chan, _On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. II. Implementation_ , Internat. J. Numer. Methods Fluids, 11 (1990), pp. 621–659. Computational methods in flow analysis (Okayama, 1988). 

- [26] J. Haack, S. Jin, and J.-G. Liu, _An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations_ , Commun. Comput. Phys., 12 (2012), pp. 955–980. 

- [27] T. Y. Hou and P. G. LeFloch, _Why nonconservative schemes converge to wrong solutions: error analysis_ , Math. Comp., 62 (1994), pp. 497–530. 

- [28] S. Jin, _Efficient asymptotic-preserving (AP) schemes for some multiscale kinetic equations_ , SIAM J. Sci. Comput., 21 (1999), pp. 441–454 (electronic). 

- [29] T. Kato, _Perturbation theory for linear operators_ , Classics in Mathematics, Springer-Verlag, Berlin, 1980 ed., 1995. 

- [30] A. Klar, _An asymptotic preserving numerical scheme for kinetic equations in the low Mach number limit_ , SIAM J. Numer. Anal., 36 (1999), pp. 1507–1527. 

- [31] R. Klein, _Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics, I: One-dimensional flow_ , J. Comput. Phys., 121 (1995), pp. 213–237. 

- [32] A. Kurganov and C.-T. Lin, _On the reduction of numerical dissipation in central-upwind schemes_ , Commun. Comput. Phys., 2 (2007), pp. 141–163. 

- [33] E. W. Larsen and J. E. Morel, _Asymptotic solutions of numerical transport problems in optically thick, diffusive regimes. II_ , J. Comput. Phys., 83 (1989), pp. 212–236. 

- [34] E. W. Larsen, J. E. Morel, and W. F. Miller, Jr., _Asymptotic solutions of numerical transport problems in optically thick, diffusive regimes_ , J. Comput. Phys., 69 (1987), pp. 283– 324. 

- [35] K.-A. Lie and S. Noelle, _An improved quadrature rule for the flux-computation in staggered central difference schemes in multidimensions_ , J. Sci. Comput., 63 (2003), pp. 1539– 1560. 

- [36] X. Liu, A. Chertock, and A. Kurganov, _An asymptotic preserving scheme for the two-dimensional shallow water equations with Coriolis forces_ , J. Comput. Phys., 391 (2019), pp. 259–279. 

- [37] G. M´etivier and S. Schochet, _The incompressible limit of the non-isentropic Euler equations_ , Arch. Ration. Mech. Anal., 158 (2001), pp. 61–90. 

- [38] L. Micalizzi and D. Torlo, _A new efficient explicit deferred correction framework: analysis and applications to hyperbolic PDEs and adaptivity_ , Commun. Appl. Math. Comput., 6 (2024), pp. 1629–1664. 

A New AP Method 

31 

- [39] L. Micalizzi, D. Torlo, and W. Boscheri, _Efficient iterative arbitrary high-order methods: an adaptive bridge between low and high order_ , Commun. Appl. Math. Comput., 7 (2025), pp. 40–77. 

- [40] H. Nessyahu and E. Tadmor, _Nonoscillatory central differencing for hyperbolic conservation laws_ , J. Comput. Phys., 87 (1990), pp. 408–463. 

- [41] S. Noelle, G. Bispen, K. R. Arun, M. Luk´aˇcov´a-Medvid’ov´a, and C.-D. Munz, _A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas dynamics_ , SIAM J. Sci. Comput., 36 (2014), pp. B989–B1024. 

- [42] P. Offner, L. Petri, and D. Torlo[¨] , _Analysis for implicit and implicit-explicit ADER and DeC methods for ordinary differential equations, advection-diffusion and advection-dispersion equations_ , Appl. Numer. Math., 212 (2025), pp. 110–134. 

- [43] R. M. Pidatella, G. Puppo, G. Russo, and P. Santagati, _Semi-conservative finite volume schemes for conservation laws_ , SIAM J. Sci. Comput., 41 (2019), pp. B576–B600. 

- [44] P. K. Sweby, _High resolution schemes using flux limiters for hyperbolic conservation laws_ , SIAM J. Numer. Anal., 21 (1984), pp. 995–1011. 

- [45] M. Tang, _Second order all speed method for the isentropic Euler equations_ , Kinet. Relat. Models, 5 (2012), pp. 155–184. 

- [46] E. F. Toro, _Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction_ , Springer-Verlag, Berlin, third ed., 2009. 

- [47] E. F. Toro and M. E. V´azquez-Cend´on, _Flux splitting schemes for the Euler equations_ , Comput. & Fluids, 70 (2012), pp. 1–12. 

- [48] E. Weinan and C.-W. Shu, _A numerical resolution study of high order essentially nonoscillatory schemes applied to incompressible flow_ , J. Comput. Phys., 110 (1994), pp. 39–46. 

- [49] J. Zeifang, J. Sch¨utz, K. Kaiser, A. Beck, M. Luk´aˇcov´a-Medvid’ov´a, and S. Noelle, _A novel full-Euler low Mach number IMEX splitting_ , Commun. Comput. Phys., 27 (2020), pp. 292–320. 

