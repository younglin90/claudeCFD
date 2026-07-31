
# A New Asymptotic-Preserving Dual Formulation Finite-Volume Method for the Compressible Euler Equations


### Alina Chertock∗, Smadar Karni†, Alexander Kurganov‡, and Lorenzo Micalizzi§

Abstract

The paper focuses on the development of numerical methods for the compressible Euler equations. It is well-known that if the Mach number is small, the system becomes stiff and hence explicit schemes suffer from severe time-step restrictions, making them inefficient or even impractical. Our objective is to develop an asymptotic preserving (AP) scheme that remains uniformly accurate and stable across all Mach numbers. Instead of the conservative hyperbolic flux splitting approach, which is widely used to design AP schemes, we consider a primitive (nonconservative) formulation and introduce a nonconservative hyperbolic splitting. The resulting system is discretized using a semi-implicit approach: the stiff part is handled semi-implicitly using second-order central differences, while the nonstiff part is treated explicitly using a second-order path-conservative centralupwind (CU) discretization. A key feature of our method is that the pressure at each time level is computed by solving a well-posed Poisson-type elliptic equation, thereby enforcing the AP property. Simultaneously, we evolve the conservative form of the system using a semi-discrete CU scheme. At the end of each stage of the time discretization, we perform a special post-processing that selects the appropriate numerical solution depending on the Mach number. This guarantees that in low-Mach-number regimes, the solution is obtained by the AP nonconservative scheme, while in higher-Mach-number regimes, a sharp and physically relevant solution is computed by the conservative CU scheme. Numerical experiments confirm that the proposed AP scheme achieves the expected second order of accuracy and that the time-step constraint is independent of the Mach number, making it a robust and efficient alternative to conventional explicit methods.

Key words: Compressible Euler equations; low Mach number; asymptotic preserving (AP) schemes; hyperbolic splitting; semi-implicit methods; deferred correction.

AMS subject classification: 76M12, 65M08, 65L04, 35B40, 35L60.

∗Department of Mathematics, North Carolina State University, Raleigh, NC 27695, USA; chertock@math.ncsu.edu †Department of Mathematics, University of Michigan, 48109, USA; karni@umich.edu ‡Department of Mathematics and Shenzhen International Center for Mathematics, Southern University of Science and Technology, Shenzhen, 518055, China; alexander@sustech.edu.cn §Department of Mathematics, North Carolina State University, Raleigh, NC 27695, USA; lmicali@ncsu.edu

1


# arXiv:2604.26111v1  [math.NA]  28 Apr 2026

2 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi


## 1 Introduction

The paper focuses on the compressible Euler equations, which, like any other hyperbolic system of PDEs, are characterized by a finite speed of propagation. This plays a crucial role in the development of explicit numerical methods, for which a major stability requirement is to keep the time steps inversely proportional to the maximum wave speed over the entire computational domain. It is well-known that low-Mach-number flows pose several major challenges for numerical simulations. A distinctive feature of such regimes is the appearance of both slow material waves, which transport quantities like entropy and vorticity, and fast acoustic waves, whose speeds scale inversely with the Mach number. As the Mach number decreases, the resulting stiffness imposes severe time-step restrictions on explicit methods and leads to excessive numerical diffusion, making such schemes inefficient or even impractical for real applications. Fully-implicit methods can address the stiffness, but have their own drawbacks: they tend to oversmear material waves (see, e.g., [10]), require the solution of large nonlinear systems, and may fail to capture the correct solution in the zero-Mach-number limit. To overcome these difficulties, a widely adopted strategy is to use either implicit-explicit (IMEX) or semi-implicit (SI) methods based on conservative hyperbolic flux splitting. This approach decomposes the hyperbolic flux into stiff and nonstiff components in a manner that preserves the conservative structure of the original system. The fast (stiff) part, associated with acoustic waves, is treated (semi-)implicitly to relax time-step limitations, while the slow (nonstiff) part is handled explicitly to accurately capture the evolution of material waves without excessive numerical diffusion. It is also known that, as the Mach number tends to zero, the compressible Euler equations reduce to the incompressible Euler equations. It is essential to ensure that numerical schemes also exhibit the same limiting behavior at the discrete level and provide a consistent discretization of the incompressible Euler equations as the Mach number tends to zero. Schemes that maintain this property are called asymptotic-preserving (AP). They were originally introduced to capture steadystate solutions for neutron transport in the diffusive regime [33,34], but the specific definition was introduced in [22,28,30] in the context of stiff kinetic equations. In recent years, AP schemes have been extensively studied and applied for simulating low-Mach-number flows; see, e.g., [7,9–11,13, 16–20,26,31,41,45,47,49] for a non-exhaustive list of references. All of the aforementioned AP schemes, which were designed for either the isentropic or full compressible Euler system, are based on different flux splitting strategies. A very simple and robust flux splitting, which was proposed in [26] for the isentropic Euler equations and later extended to the rotating shallow water equations in [36], seems to be rather optimal in the sense that it very accurately identifies and separates a linear stiff pressure term, which is then discretized implicitly. However, extending this flux splitting to the full Euler equations presents significant challenges. In this paper, we propose an alternative way of accurately identifying and separating a stiff part of the full Euler system: we first rewrite the studied system in a nonconservative form and then introduce a nonconservative hyperbolic splitting, which may be naturally viewed as an extension of the flux splitting from [26]. We integrate the nonconservative system using a SI method implemented as follows. The stiff part is approximated semi-implicitly using a simple second-order accurate central-differencing, and the nonstiff part is handled explicitly using a second-order pathconservative (PC) central-upwind (CU) discretization. The resulting SI approach is then realized in such a way that the pressure update consists of solving a Poisson-type elliptic equation: this is

A New AP Method 3

used to enforce the AP property. However, the resulting SI method can only be applied to low-Mach-number regimes, where the magnitude of discontinuous waves is small. For large Mach numbers, solving nonconservative formulations of the Euler equations in the presence of discontinuities typically leads to nonphysical computed solutions, as was demonstrated in [4,27]. We therefore apply a dual formulation (DF) approach, which has been recently introduced in [14] (for other recent works on DF methods, we refer the reader to [2,3,5,43]), and solve the nonconservative and conservative formulations simultaneously. The latter one is discretized in a fully-explicit manner using the second-order semi-discrete CU discretization from [32]. This way, at each stage of a multi-stage SI time discretization (we have used the deferred correction (DeC) time discretization from [42]), two copies of the computed solution are evolved: one is AP, but nonconservative, while the second one is conservative, but nonAP. Hence, upon the completion of each stage of the time evolution, we post-process the obtained solutions to automatically ensure that in low-Mach-number regimes, the overall numerical solution is obtained by the AP nonconservative scheme, while in large (intermediate)-Mach-number regimes, the solution reduces to the sharp and conservative solution obtained by the CU scheme. The rest of the paper is organized as follows. In §2, we give the necessary preliminaries: we introduce the governing equations, namely, non-dimensional conservative and nonconservative (primitive) formulations of the full Euler equations, discuss their zero-Mach-number limit, and briefly review the considered DF framework. In §3, we introduce the novel AP DF finite-volume (DF-FV) scheme for compressible Euler equations, providing a rigorous proof of its AP character. In §4, we demonstrate the performance of the proposed scheme on a number of challenging numerical examples. Finally, concluding remarks can be found in §5.


## 2 Preliminaries

The main goal of this section is to provide the background needed for presenting the proposed AP scheme. Specifically, we will describe: • a nonconservative reformulation of the Euler equations in terms of primitive variables, which allows for a natural decomposition of the terms, which are stiff and nonstiff in the low-Machnumber regime; • a formal asymptotic analysis of the studied equations in low-Mach-number regimes, providing the incompressible system that the AP scheme must accurately approximate in the zero-Mach-number limit; • a DF framework, in which both conservative and nonconservative formulations of the studied system are numerically solved simultaneously exploiting the advantages of each of them in the corresponding Mach-number regimes.


### 2.1 Conservative and Primitive Formulations

After suitable non-dimensionalization and rescaling, the two-dimensional (2-D) compressible Euler equations can be written in the conservative form as


![Equation](images/[2026] AP + FVM + comp + Euler_eq001.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq002.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq003.png)

4 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi


![Equation](images/[2026] AP + FVM + comp + Euler_eq004.png)

Here, ρ, u := (u, v)⊤, and E denote the density, velocity, and total energy, respectively, p is the pressure, ε is the reference Mach number, and the system is closed by the equation of state, which, in the case of an ideal gas, reads as


![Equation](images/[2026] AP + FVM + comp + Euler_eq005.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq006.png)

with γ being the specific heat ratio. This system is hyperbolic and features acoustic waves traveling with (maximum) speed |u| + c, where c is the speed of sound given by c := 1

ε � γp/ρ. Notice that in low-Mach-number regimes, the acoustic waves travel at a high (maximum) speed proportional to 1/ε. For the purpose of deriving our AP scheme, we also consider an equivalent nonconservative formulation of the system (2.1)–(2.4) in terms of the primitive variables ρ, u, and p:


![Equation](images/[2026] AP + FVM + comp + Euler_eq007.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq008.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq009.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq010.png)

We emphasize that this formulation is equivalent to the conservative system (2.1)–(2.4) only for smooth solutions, and numerical approximations of (2.5)–(2.7) typically converge to nonphysical solutions when discontinuities are present; see [4,27] for a detailed discussion.


### 2.2 Zero-Mach-Number Limit

It is well-known (see, e.g., a formal analysis in [6, 37]) that in the zero-Mach-number limit the compressible Euler equations reduce to the incompressible ones. To illustrate this, we examine the formal behavior of the primitive system (2.5)–(2.7) as ε →0. We substitute the formal expansions


![Equation](images/[2026] AP + FVM + comp + Euler_eq011.png)

into (2.5)–(2.7) and collect terms by powers of ε. This yields


![Equation](images/[2026] AP + FVM + comp + Euler_eq012.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq013.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq014.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq015.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq016.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq017.png)

It follows from (2.8)–(2.9) that p(0)(x, y, t) = p(0)(t) and p(1)(x, y, t) = p(1)(t) are spatially uniform. One can also show that both p(0) and p(1) are independent of time, provided the following Dirichlet boundary condition holds:


![Equation](images/[2026] AP + FVM + comp + Euler_eq018.png)

A New AP Method 5

where p0 > 0 and p1 are constants, pk(x, y, t), k ≥2 are bounded functions, and Ωis the spatial domain with boundary ∂Ω. In fact, such a boundary condition implies p(0) ≡p0 and p(1) ≡p1. Using this in (2.12), one concludes that ∇·u(0) = 0, which further implies ∇·u(1) = 0 thanks to (2.13). Hence, the zero-Mach-number limiting equations are


![Equation](images/[2026] AP + FVM + comp + Euler_eq019.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq020.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq021.png)

and the correct low-Mach-number scaling for the O(ε) terms is


![Equation](images/[2026] AP + FVM + comp + Euler_eq022.png)


### 2.3 Dual Formulation (DF) Framework

As outlined above, a new AP scheme for the compressible Euler equations will be constructed within the DF framework, in which both conservative and primitive formulations are numerically solved simultaneously. The conservative form ensures proper handling of discontinuities, while the primitive form is used to achieve the AP property in the zero-Mach-number limit. While the DF methodology is not specifically developed to handle multiscale features, it lays the groundwork for the AP scheme developed in subsequent sections by enabling an efficient and accurate treatment of the studied Euler equations in both compressible and nearly incompressible flow regimes. We consider a general 2-D hyperbolic system of conservation laws,


![Equation](images/[2026] AP + FVM + comp + Euler_eq023.png)

where U is the vector of conservative variables and F and G are fluxes, and rewrite it in an equivalent nonconservative form


![Equation](images/[2026] AP + FVM + comp + Euler_eq024.png)

where V is the vector of nonconservative variables, �F and � G are the corresponding fluxes, and B(V )Vx and C(V )Vy are the nonconservative product terms. The key idea of the DF approach is to evolve the solutions of (2.19) and (2.20) simultaneously. A crucial step in DF-based methods is a post-processing, in which the evolved values of the nonconservative solution V are replaced with a more reliable approximation after the update. This step is necessary because long-term evolutions of V by directly solving the nonconservative system (2.20) may lead to nonphysical solutions in the presence of discontinuities, which typically appear when the studied Euler system is considered in the compressible (large/medium-Machnumber) regime. The post-processing can be described as follows. After advancing the solutions of (2.19) and (2.20) from a certain time level t to the next time level t + ∆t, the evolved values of V (t + ∆t) are replaced with r � V (U(t + ∆t)), V (t + ∆t) � , (2.21)

where r is a suitable replacement function and V (U) is a conservative-primitive variable transformation. In the simplest nonstiff case, one can set


![Equation](images/[2026] AP + FVM + comp + Euler_eq025.png)

6 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

However, in the development of the AP scheme below, we will modify the post-processing (2.22) by taking an appropriate function r in (2.21) to ensure that in the nearly incompressible (lowMach-number) regime the AP V -solution is not overwritten by the non-AP conservative one.


## 3 Novel AP Scheme for Compressible Euler Equations

Building on the DF framework described in §2.3, we now present a novel AP scheme designed for the compressible Euler equations across all Mach-number regimes, from fully compressible to nearly incompressible flows. The proposed method couples the conservative (2.1)–(2.4) and primitive (2.5)–(2.7) formulations of the system, ensuring stability, accuracy, and consistency with the analytical asymptotic behavior as ε →0. In this section, we provide a complete description of the proposed space-time discretization, starting with the primitive system (2.5)–(2.7). In §3.1, we outline its temporal integration, which is based on a new hyperbolic splitting and an SI approach. In §3.2, we present a fully discrete second-order AP scheme for the primitive system, and in §3.3, we describe the semi-discrete CU scheme employed for the conservative system. §3.4 is devoted to the clarification of important implementation details. Finally, in §3.5, we present the Mach-number dependent post-processing strategy used to reconcile primitive and conservative variables.


### 3.1 Novel AP Time Discretization of the Primitive System

We begin by providing a precise definition of an asymptotic-preserving (AP) time discretization in the context of the zero-Mach-number limit.

Definition 3.1 (AP time discretization) Assume that the computed solution at time tn can be expanded as

ρn = ρ(0),n + ερ(1),n + ε2ρ(2),n + . . . , un = u(0),n + εu(1),n + ε2u(2),n + . . . ,

pn = p(0),n + εp(1),n + ε2p(2),n + . . . , (3.1)

which is compatible with the asymptotic limits (2.17) and (2.18), that is,


![Equation](images/[2026] AP + FVM + comp + Euler_eq026.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq027.png)

and satisfies the Dirichlet boundary condition (2.14), namely,


![Equation](images/[2026] AP + FVM + comp + Euler_eq028.png)

Let us consider a one-step time discretization that produces at time tn+1 = tn + ∆t an approximation (ρn+1, un+1, pn+1). We say that such time discretization is AP if it admits an asymptotic expansion of the type (3.1) and yields a consistent discretization of (2.15)–(2.18) as ε →0.

To construct an AP time discretization, we first perform a hyperbolic splitting of the primitive system (2.5)–(2.7), separating the stiff pressure-driven terms from the nonstiff convective terms. This splitting enables the use of an SI integration strategy in which the stiff and nonstiff terms are treated semi-implicitly and explicitly, respectively, ensuring uniform stability and asymptotic consistency as ε →0.

A New AP Method 7

3.1.1 A New Hyperbolic Splitting

We follow the idea from [26,36] and split the nonconservative system into two parts corresponding to the slow and fast dynamics as follows. We first define the time-dependent variables


![Equation](images/[2026] AP + FVM + comp + Euler_eq029.png)

and then add and subtract 1 ε2ρmax∇p and γpmin∇· u from (2.6) and (2.7), respectively, to rewrite system (2.5)–(2.7) as follows:


![Equation](images/[2026] AP + FVM + comp + Euler_eq030.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq031.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq032.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq033.png)

This system can be put in the following vector form:


![Equation](images/[2026] AP + FVM + comp + Euler_eq034.png)

where V := (ρ, u, v, p)⊤, the nonlinear nonstiff (slow dynamics) part consists of the fluxes


![Equation](images/[2026] AP + FVM + comp + Euler_eq035.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq036.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq037.png)

and the nonstiff nonconservative terms �B(V )Vx + �C(V )Vy with matrices


![Equation](images/[2026] AP + FVM + comp + Euler_eq038.png)



     

0 0 0 0


![Equation](images/[2026] AP + FVM + comp + Euler_eq039.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq040.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq041.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq042.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq043.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq044.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq045.png)

0 0 0 0


![Equation](images/[2026] AP + FVM + comp + Euler_eq046.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq047.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq048.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq049.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq050.png)

while, the linear stiff (fast dynamics) part consists of the stiff nonconservative terms �B(V )Vx + �C(V )Vy with matrices


![Equation](images/[2026] AP + FVM + comp + Euler_eq051.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq052.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq053.png)

0 0 0 0


![Equation](images/[2026] AP + FVM + comp + Euler_eq054.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq055.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq056.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq057.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq058.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq059.png)

0 0 0 0

0 0 0 0


![Equation](images/[2026] AP + FVM + comp + Euler_eq060.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq061.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq062.png)

We stress that the subsystem Vt + �F (V )x + � G(V )y = �B(V )Vx + �C(V )Vy is indeed nonstiff as the eigenvalues of the matrices ∂� F ∂V (V ) −�B(V ) and ∂� G ∂V (V ) −�C(V ), are {u ± ˜c, u, u} and {v ± ˜c, v, v}, respectively, with


![Equation](images/[2026] AP + FVM + comp + Euler_eq063.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq064.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq065.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq066.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq067.png)

8 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

which are real and of size O(1) thanks to the definitions of ρmax and pmin in (3.5) and to the asymptotic analysis in §2.2, which ensure that


![Equation](images/[2026] AP + FVM + comp + Euler_eq068.png)

In the next subsection, we will utilize this splitting and design an AP time discretization based on an explicit approximation of the nonstiff subsystem and an SI discretization of the stiff terms on the right-hand sides (RHSs) of (3.7) and (3.8).

3.1.2 First-Order AP SI Time Discretization

The simplest first-order AP SI time discretization of the system (3.9) reads as


![Equation](images/[2026] AP + FVM + comp + Euler_eq069.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq070.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq071.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq072.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq073.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq074.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq075.png)

(3.12)

which can also be written in the following vector form:


![Equation](images/[2026] AP + FVM + comp + Euler_eq076.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq077.png)

where V n := (ρn, un, pn)⊤≈V (tn), ρn max := ρmax(tn), pn min := pmin(tn), and

Rn := �F (V n)x + � G(V n)y −�B(V n)V n x −�C(V n)V n y

=


![Equation](images/[2026] AP + FVM + comp + Euler_eq078.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq079.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq080.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq081.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq082.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq083.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq084.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq085.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq086.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq087.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq088.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq089.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq090.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq091.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq092.png)

Ln,n+1 := −�B(V n)V n+1 x −�C(V n)V n+1 y =


![Equation](images/[2026] AP + FVM + comp + Euler_eq093.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq094.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq095.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq096.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq097.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq098.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq099.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq100.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq101.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq102.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq103.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq104.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq105.png)

Notice that in (3.14), Ln,n+1 is discretized in an SI (rather than fully implicit) manner, where both �B and �C are evaluated at V n (and not at V n+1), which prevents from numerically solving complicated systems of nonlinear algebraic equations. We shall now prove that the time discretization (3.12) is indeed AP, provided that the time step is computed based on the wave speeds of the nonstiff subsystem, that is, provided that


![Equation](images/[2026] AP + FVM + comp + Euler_eq106.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq107.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq108.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq109.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq110.png)

where KCFL is a CFL number and ∆x and ∆y are mesh sizes used in the spatial discretization. Notice that selecting the time step ∆t according to (3.15) makes it asymptotically independent of ε as, according to (3.10)–(3.11), ˜c = O(1).

A New AP Method 9

Theorem 3.1 The first-order SI time discretization (3.12) is AP according to Definition 3.1, provided that ∆t is computed as in (3.15).

Proof: We begin by formally showing that the computed solution ρn+1, un+1, pn+1 admits an expansion of the type (3.1) satisfying (3.2) and (3.3) in the limit as ε →0. We substitute the corresponding expansion of the numerical solution at time tn into the scheme (3.12) and use (3.2)–(3.3) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq111.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq112.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq113.png)

(3.16)

un+1 = u(0),n + εu(1),n + ε2u(2),n −∆t � (u(0),n·∇)u(0),n + ρn max −ρ(0),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq114.png)

−ε∆t � (u(1),n·∇)u(0),n + (u(0),n·∇)u(1),n − ρ(1),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq115.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq116.png)

−ε2∆t � (u(2),n·∇)u(0),n + (u(1),n·∇)u(1),n + (u(0),n·∇)u(2),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq117.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq118.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq119.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq120.png)

(3.17)

pn+1 = p0 + εp1 + ε2p(2),n −ε2∆tu(0),n·∇p(2),n −∆tγpn min∇·un+1 + O(ε3). (3.18)

Thanks to the explicit nature of the density update in (3.16), we conclude that indeed ρn+1 admits the required asymptotic expansion ρn+1 = ρ(0),n+1 + ερ(1),n+1 + ερ(2),n+1 + . . . , where the different terms of the expansion are obtained by collecting corresponding powers of ε:


![Equation](images/[2026] AP + FVM + comp + Euler_eq121.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq122.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq123.png)

(3.19)

In order to show that also pn+1 admits an expansion of the same type, we take the divergence of the velocity equation (3.17), substitute ∇·un+1 into the pressure equation (3.18), and use the divergence-free assumption (3.3) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq124.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq125.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq126.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq127.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq128.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq129.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq130.png)

+ ε2(∆t)2γpn min∇· � (u(2),n·∇)u(0),n + (u(1),n·∇)u(1),n + (u(0),n·∇)u(2),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq131.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq132.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq133.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq134.png)

10 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

which implies that pn+1 is the solution of the elliptic equation


![Equation](images/[2026] AP + FVM + comp + Euler_eq135.png)

subject to the boundary condition (3.4). According to the theory of perturbed linear operators [29], one can conclude that


![Equation](images/[2026] AP + FVM + comp + Euler_eq136.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq137.png)

and hence ∇pn+1 = ε2∇p(2),n+1 + ε3∇p(3),n+1 + ε4∇p(4),n+1 + . . . ,

which we substitute into (3.17) to obtain the velocity expansion


![Equation](images/[2026] AP + FVM + comp + Euler_eq138.png)

with

u(0),n+1 = u(0),n −∆t � (u(0),n·∇)u(0),n + ρn max −ρ(0),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq139.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq140.png)

u(1),n+1 = u(1),n −∆t � (u(1),n·∇)u(0),n + (u(0),n·∇)u(1),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq141.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq142.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq143.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq144.png)

u(2),n+1 = u(2),n −∆t � (u(2),n·∇)u(0),n + (u(1),n·∇)u(1),n + (u(0),n·∇)u(2),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq145.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq146.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq147.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq148.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq149.png)

(3.22)

We now need to show that (3.2) and (3.3) hold for the updated solution, along with the consistency of the scheme (3.12) with (2.15) and (2.16) as ε →0. We have already shown that (3.2) holds; see (3.21). The divergence-free conditions (3.3) can be deduced from the pressure update (3.18), which in view of the obtained results yields

ε2p(2),n+1 = ε2p(2),n −ε2∆tu(0),n·∇p(2),n −∆tγpn min∇·u(0),n+1


![Equation](images/[2026] AP + FVM + comp + Euler_eq150.png)

Collecting the power-like terms of ε, we deduce ∇·u(0),n+1 = ∇·u(1),n+1 = 0. The consistency with (2.15) immediately follows from the first equation in (3.19). To show the consistency with (2.16), we rewrite the first equation in (3.22) as

u(0),n+1 = u(0),n −∆t � (u(0),n·∇)u(0),n + ∇p(2),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq151.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq152.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq153.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq154.png)

which is a consistent discretization of (2.16). We remark that since the zeroth and first modes of the pressure are constant, the evolution of the pressure in the zero-Mach-number limit essentially consists of the evolution of the second mode. Thus, ∇p(2),n+1−∇p(2),n ≈O(∆t) and the last term in (3.23), in fact, represents a temporal diffusion term, which is proportional to O((∆t)2). We also remark that according to (3.15), the time step ∆t is asymptotically independent of ε. ■

A New AP Method 11

3.1.3 Second-Order AP SI Time Discretization

We now introduce a second-order AP SI time discretization, which is based on the DeC approach, which was originally introduced in [21]. Our second-order AP SI-DeC time discretization is directly related to the IMEX-DeC methods presented in [42] and based on the DeC formulation introduced in [1]; see also [38,39]. According to the second-order AP SI-DeC time discretization, the solution of (3.9) is evolved from t = tn to t = tn+1 through the following two stages:


![Equation](images/[2026] AP + FVM + comp + Euler_eq155.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq156.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq157.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq158.png)

where the upper index ∗is associated with the intermediate solution V ∗, and the definitions of the operators R∗and Ln,n, L∗,∗, and L∗,n+1 are analogous to those given in (3.13) and (3.14), respectively. The scheme (3.24) can be equivalently written as


![Equation](images/[2026] AP + FVM + comp + Euler_eq159.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq160.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq161.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq162.png)

(3.25)

and


![Equation](images/[2026] AP + FVM + comp + Euler_eq163.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq164.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq165.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq166.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq167.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq168.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq169.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq170.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq171.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq172.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq173.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq174.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq175.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq176.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq177.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq178.png)

(3.26)

This second-order time discretization is indeed AP as shown in the next theorem.

Theorem 3.2 The second-order SI-DeC discretization (3.25)–(3.26) is AP according to Definition 3.1, provided that ∆t is computed according to (3.15).

Proof: The proof proceeds along the same lines and uses the same arguments as in the proof of Theorem 3.1. We begin by observing that the first stage of the second-order SI-DeC discretization coincides with the first-order AP SI time discretization studied before. Therefore, according to Theorem 3.1, the intermediate solution ρ∗, u∗, p∗, obtained by (3.25) admits an expansion of the type (3.1), that is, ρ∗= ρ(0),∗+ ερ(1),∗+ ε2ρ(2),∗+ . . . , u∗= u(0),∗+ εu(1),∗+ ε2u(2),∗+ . . . ,

p∗= p(0),∗+ εp(1),∗+ ε2p(2),∗+ . . . , (3.27)

12 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi


![Equation](images/[2026] AP + FVM + comp + Euler_eq179.png)

We then substitute the expansions (3.1) and (3.27) into (3.26) and use the conditions (3.2)–(3.3) and (3.28) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq180.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq181.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq182.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq183.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq184.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq185.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq186.png)

(3.29)


![Equation](images/[2026] AP + FVM + comp + Euler_eq187.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq188.png)

2

� (u(0),n·∇)u(0),n + (u(0),∗·∇)u(0),∗+ ρn max −ρ(0),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq189.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq190.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq191.png)

2

� (u(1),n·∇)u(0),n + (u(0),n·∇)u(1),n + (u(1),∗·∇)u(0),∗+ (u(0),∗·∇)u(1),∗


![Equation](images/[2026] AP + FVM + comp + Euler_eq192.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq193.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq194.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq195.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq196.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq197.png)

2

� (u(2),n·∇)u(0),n + (u(1),n·∇)u(1),n + (u(0),n·∇)u(2),n


![Equation](images/[2026] AP + FVM + comp + Euler_eq198.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq199.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq200.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq201.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq202.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq203.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq204.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq205.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq206.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq207.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq208.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq209.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq210.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq211.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq212.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq213.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq214.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq215.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq216.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq217.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq218.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq219.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq220.png)

(3.30)


![Equation](images/[2026] AP + FVM + comp + Euler_eq221.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq222.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq223.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq224.png)

The explicit nature of the density update (3.29) implies that ρn+1 admits the required expansion ρn+1 = ρ(0),n+1 + ερ(1),n+1 + ε2ρ(2),n+1 + . . . with ρ(0),n+1 satisfying


![Equation](images/[2026] AP + FVM + comp + Euler_eq225.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq226.png)

and other coefficients satisfying the equations, which can be easily obtained by grouping the corresponding powers of ε.

A New AP Method 13

As in the proof of Theorem 3.1, we show that pn+1 admits the asymptotic expansion by proving that it satisfies a well-posed elliptic problem with suitable boundary conditions. Taking the divergence of the velocity equation (3.30) and substituting ∇·un+1 into the pressure equation (3.31) yields


![Equation](images/[2026] AP + FVM + comp + Euler_eq227.png)

This together with the boundary conditions (2.14), results in the same expansion for pn+1, which we have established in (3.20)–(3.21) for the first-order SI method, leading to


![Equation](images/[2026] AP + FVM + comp + Euler_eq228.png)

Next, we substitute (3.33) into the the velocity equation (3.30) and a straightforward grouping of the power-like terms of ε gives the equations for the coefficients of the velocity expansion un+1 = u(0),n+1 + εu(1),n+1 + ε2u(2),n+1 + . . . . The equation for u(0),n+1 is


![Equation](images/[2026] AP + FVM + comp + Euler_eq229.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq230.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq231.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq232.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq233.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq234.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq235.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq236.png)

and the other equations can be obtained similarly. Let us now show the consistency with the asymptotic limit. The required conditions (3.21) on the pressure modes have been already shown. The divergence-free conditions for the velocity modes are then established from the pressure update (3.31), which becomes


![Equation](images/[2026] AP + FVM + comp + Euler_eq237.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq238.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq239.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq240.png)

It is clear that the O(1) and O(ε) terms here vanish, that is, ∇·u(0),n+1 = ∇·u(1),n+1 = 0. Finally, we notice that (3.32) and (3.34) are consistent discretizations of (2.15) and (2.16) with the last term in (3.34) representing a temporal diffusion, which is consistent with the order of accuracy of the scheme and thus proportional to O((∆t)3). ■

Remark 3.1 The described AP SI-DeC time discretization can be extended to arbitrarily high order in a straightforward way within the DeC framework. For the sake of brevity, we restrict our consideration to the second order of accuracy, which matches the accuracy that will be used in the spatial discretization discussed in §3.2.


### 3.2 Fully Discrete Second-Order AP Scheme for the Primitive System

In this section, we construct a fully discrete scheme based on the second-order AP SI time discretization presented in §3.1.3. To this end, we first introduce uniform Cartesian cells Ij,k := [xj−1

2, xj+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq241.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq242.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq243.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq244.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq245.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq246.png)

2 ≡∆y, centered at (xj, yk) with xj = � xj−1


![Equation](images/[2026] AP + FVM + comp + Euler_eq247.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq248.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq249.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq250.png)

V n j,k :≈ 1 ∆x∆y ��


![Equation](images/[2026] AP + FVM + comp + Euler_eq251.png)

14 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

The fully discrete FV version of the second-order AP scheme (3.24) reads as


![Equation](images/[2026] AP + FVM + comp + Euler_eq252.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq253.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq254.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq255.png)

where Rn j.k and R∗ j.k are obtained using the PCCU discretization from [3, 15], which is a lowdissipation generalization of the PCCU discretization from [12], while Ln,∗ j.k , Ln,n j.k , L∗,∗ j.k, and L∗,n+1 j.k are obtained using central differences. In what follows, for the sake of brevity, we provide details on Rn j.k and Ln,∗ j.k only, whereas the remaining discretizations are obtained in a similar manner. We begin with

Rn j,k := 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq256.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq257.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq258.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq259.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq260.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq261.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq262.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq263.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq264.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq265.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq266.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq267.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq268.png)

+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq269.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq270.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq271.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq272.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq273.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq274.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq275.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq276.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq277.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq278.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq279.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq280.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq281.png)

(3.37)

where �F n j+ 1

2 ,k and �G n j,k+ 1

2 are the CU numerical fluxes

�F n j+ 1

2 ,k := ˜a+,n j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq282.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq283.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq284.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq285.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq286.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq287.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq288.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq289.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq290.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq291.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq292.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq293.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq294.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq295.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq296.png)

�G n j,k+ 1

2 := ˜b+,n j,k+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq297.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq298.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq299.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq300.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq301.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq302.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq303.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq304.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq305.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq306.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq307.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq308.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq309.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq310.png)

and V ±,n j+ 1

2 ,k and V ±,n j,k+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq311.png)

2 ,k and δV n j,k+ 1

2 are “built-in” anti-diffusion terms, and ˜a±,n j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq312.png)

2 denote the one-sided local propagation speeds of the nonstiff subsystem in the xand y-direction, respectively. The point values


![Equation](images/[2026] AP + FVM + comp + Euler_eq313.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq314.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq315.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq316.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq317.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq318.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq319.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq320.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq321.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq322.png)

are computed using the piecewise linear reconstruction


![Equation](images/[2026] AP + FVM + comp + Euler_eq323.png)

in which the slopes (Vx)n j,k and (Vy)n j,k are approximated using the generalized minmod limiter (see, e.g., [35,40,44]):

(Vx)n j,k := minmod


![Equation](images/[2026] AP + FVM + comp + Euler_eq324.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq325.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq326.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq327.png)

(Vy)n j,k := minmod


![Equation](images/[2026] AP + FVM + comp + Euler_eq328.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq329.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq330.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq331.png)

(3.39)

A New AP Method 15

where the minmod function, defined by


![Equation](images/[2026] AP + FVM + comp + Euler_eq332.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq333.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq334.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq335.png)

is applied in a componentwise manner. The parameter θ ∈[1, 2] in (3.39) is to be chosen to adjust the amount of numerical dissipation present in the resulting scheme, with larger values of θ leading to sharper but, in general, more oscillatory solutions. The one-sided local speeds of propagation are estimated using the smallest and largest eigenvalues of the matrices ∂� F ∂V (V ) −�B(V ) and ∂� G ∂V (V ) −�C(V ) as follows:


![Equation](images/[2026] AP + FVM + comp + Euler_eq336.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq337.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq338.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq339.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq340.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq341.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq342.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq343.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq344.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq345.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq346.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq347.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq348.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq349.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq350.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq351.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq352.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq353.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq354.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq355.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq356.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq357.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq358.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq359.png)

(3.40)

where the sound speeds

˜c ±,n j+ 1

2 ,k := 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq360.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq361.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq362.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq363.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq364.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq365.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq366.png)

2 := 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq367.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq368.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq369.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq370.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq371.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq372.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq373.png)

are computed using the following discrete versions of ρn max and pn min ρn max := max j,k ρ n j,k, pn min := min j,k p n j,k, (3.41)

and δ is a small positive parameter introduced to prevent divisions by 0 (we have taken δ := 10−15

in the numerical experiments reported in §4). The “built-in” anti-diffusion terms are


![Equation](images/[2026] AP + FVM + comp + Euler_eq374.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq375.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq376.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq377.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq378.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq379.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq380.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq381.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq382.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq383.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq384.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq385.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq386.png)

where


![Equation](images/[2026] AP + FVM + comp + Euler_eq387.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq388.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq389.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq390.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq391.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq392.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq393.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq394.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq395.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq396.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq397.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq398.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq399.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq400.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq401.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq402.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq403.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq404.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq405.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq406.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq407.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq408.png)

Finally,


![Equation](images/[2026] AP + FVM + comp + Euler_eq409.png)

where ∇· uj,k denotes the discrete divergence operator computed using second-order central differences: ∇· uj,k := uj+1,k −uj−1,k


![Equation](images/[2026] AP + FVM + comp + Euler_eq410.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq411.png)

16 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi


### 3.3 Semi-Discrete CU Scheme for the Conservative System

We now consider the conservative formulation (2.1)–(2.3), which can be put into the following vector form:

Ut + F (U)x + G(U)y = 0, U := (ρ, ρu, ρv, E)⊤,

F (U) := � ρu, ρu2 + p


![Equation](images/[2026] AP + FVM + comp + Euler_eq412.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq413.png)

In the semi-discrete CU scheme, the cell averages U j,k(t) :≈ 1 ∆x∆y ��

Ij,k U(x, y, t) dxdy are evolved in time by numerically solving the following system of ODEs:


![Equation](images/[2026] AP + FVM + comp + Euler_eq414.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq415.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq416.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq417.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq418.png)

where F j+ 1

2 ,k and Gj,k+ 1

2 are the CU numerical fluxes from [32] defined as


![Equation](images/[2026] AP + FVM + comp + Euler_eq419.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq420.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq421.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq422.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq423.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq424.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq425.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq426.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq427.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq428.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq429.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq430.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq431.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq432.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq433.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq434.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq435.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq436.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq437.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq438.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq439.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq440.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq441.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq442.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq443.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq444.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq445.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq446.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq447.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq448.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq449.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq450.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq451.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq452.png)

(3.46)

Here, the interface values U ± j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq453.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq454.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq455.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq456.png)

reconstructed primitive variables V ± j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq457.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq458.png)

a straightforward transformation U(V ) from V to U. The quantities a± j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq459.png)

2 are the one-sided local speeds of propagation for the conservative system (3.44) in the xand y-direction, respectively. They are estimated using the largest and smallest eigenvalues of the corresponding flux Jacobians as follows:

a− j+ 1

2 ,k := min � u− j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq460.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq461.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq462.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq463.png)

a+ j+ 1

2 ,k := max � u− j+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq464.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq465.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq466.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq467.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq468.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq469.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq470.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq471.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq472.png)

b− j,k+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq473.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq474.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq475.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq476.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq477.png)

b+ j,k+ 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq478.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq479.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq480.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq481.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq482.png)

2 := 1


![Equation](images/[2026] AP + FVM + comp + Euler_eq483.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq484.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq485.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq486.png)

(3.47)

where δ := 10−15 is used to avoid divisions by 0.

A New AP Method 17

The “built-in” anti-diffusion terms are


![Equation](images/[2026] AP + FVM + comp + Euler_eq487.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq488.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq489.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq490.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq491.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq492.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq493.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq494.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq495.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq496.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq497.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq498.png)

with


![Equation](images/[2026] AP + FVM + comp + Euler_eq499.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq500.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq501.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq502.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq503.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq504.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq505.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq506.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq507.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq508.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq509.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq510.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq511.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq512.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq513.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq514.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq515.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq516.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq517.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq518.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq519.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq520.png)

(3.49)

Note that most of the indexed quantities in the semi-discrete setting above are time-dependent, but we have omitted this dependence to ease the notation. Finally, the system of ODEs (3.45) has to be integrated in time using an appropriate ODE solver. Its solution is performed simultaneously with one of the primitive systems using the explicit counterpart of the SI-DeC scheme, and a post-processing is performed at each stage, as explained in §3.4.


### 3.4 Implementation Details

In our DF-FV approach, the solutions of the primitive and conservative systems are evolved simultaneously according to the following algorithm. • Step 1 (Compute ρ ∗ j,k). We use the ρ-equation in (3.35) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq521.png)

• Step 2 (Solve the linear elliptic equation for p ∗ j,k). We apply the discrete divergence operator (3.43) to the u-equations in (3.35) and substitute them into the p-equation in (3.35) to obtain the following linear system of algebraic equations for p ∗ j,k, which is a discretization of the linear elliptic equation for p∗:


![Equation](images/[2026] AP + FVM + comp + Euler_eq522.png)

where the discrete Laplacian ∆pj,k is defined as


![Equation](images/[2026] AP + FVM + comp + Euler_eq523.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq524.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq525.png)

• Step 3 (Compute u ∗ j,k). Once p ∗ j,k is available, we use the u-equations in (3.35) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq526.png)

• Step 4 (Compute U ∗ j,k). We perform the conservative update with the explicit counterpart of the SI-DeC scheme to obtain the solution U ∗ j,k at the intermediate stage


![Equation](images/[2026] AP + FVM + comp + Euler_eq527.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq528.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq529.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq530.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq531.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq532.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq533.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq534.png)

18 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

and then post-process the primitive solution by replacing V ∗ j,k with r � V � U ∗ j,k � , V ∗ j,k � ; see §3.5.

• Step 5 (Compute ρ n+1 j,k ). We solve the ρ-equation in (3.36) to obtain


![Equation](images/[2026] AP + FVM + comp + Euler_eq535.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq536.png)

• Step 6 (Solve the linear elliptic equation for p n+1 j,k ). We apply the discrete divergence operator (3.43) to the u-equations in (3.36) and substitute them into the p-equation in (3.36) to obtain the following linear system of algebraic equations for p n+1 j,k , which is a discretization of the linear elliptic equation for pn+1:

p n+1 j,k −(∆t)2γp∗ min ε2ρ∗ max ∆p n+1 j,k = p n j,k −∆t


![Equation](images/[2026] AP + FVM + comp + Euler_eq537.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq538.png)

−∆tγp∗ min∇· u n j,k + (∆t)2γp∗ min 2 ∇· � (Ru)n j,k + (Ru)∗ j,k � −(∆t)2γp∗ min 2 ∇· � (Lu)n,n j,k −(Lu)∗,∗ j,k � .

• Step 7 (Compute u n+1 j,k ). Once p n+1 j,k is available, we compute

u n+1 j,k = u n j,k −∆t


![Equation](images/[2026] AP + FVM + comp + Euler_eq539.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq540.png)

• Step 8 (Compute U n+1 j,k ). Finally, we use the explicit part of the SI-DeC scheme to evaluate


![Equation](images/[2026] AP + FVM + comp + Euler_eq541.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq542.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq543.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq544.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq545.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq546.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq547.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq548.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq549.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq550.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq551.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq552.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq553.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq554.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq555.png)

(3.51)

and then post-process of the primitive solution by replacing V n+1 j,k with r � V � U n+1 j,k � , V n+1 j,k � ; see §3.5. We recall that the interface values of U needed for the computation of the numerical fluxes in (3.50) and (3.51) are obtained from the reconstructed primitive variables V at the corresponding time levels. Note that the conservative updates are, as a matter of fact, explicit, since they are performed using the explicit part of the SI-DeC scheme.


### 3.5 Post-Processing

Upon completion of Steps 4 and 8, we obtain two sets of numerical solutions: the V -solution, which is AP but nonconservative, and the U-solution, which is conservative but non-AP. We therefore design the post-processing (2.21) using their convex combination with coefficients dependent on ε, leveraging the AP SI method in the low-Mach-number regime and the sharp conservative CU scheme in the moderateand high-Mach-number regimes—thus ensuring accuracy, stability, and physical consistency across all flow regimes. Specifically, we select the following function r in (2.21): r � V � U j,k � , V j,k � = (1 −s(ε)) V � U j,k � + s(ε)V j,k, ∀j, k,

A New AP Method 19

where s is a suitable switching function, which is supposed to be increasing, continuous, and satisfy s(1) = 0 and s(0) = 1. Moreover, in the high-Mach-number regime, s should be ∼0 so that the primitive variables V are almost completely overwritten by V (U), while, in the low-Mach-number regime, s should be ∼1 so that the primitive variables V stay almost unchanged. For intermediate values of ε, a smooth transition between 1 and 0 is expected.


## 4 Numerical Examples

In this section, we verify the accuracy and robustness of the proposed AP scheme on a variety of numerical examples across different values of ε. In all of the numerical examples, we:

• Take the minmod parameter θ = 1.3;

• Adaptively select time steps based on the time-step restriction (3.15) for the nonstiff part of the primitive system;

• Set γ = 1.4 (except for Example 1, in which γ = 2);

• Modify (3.5) to


![Equation](images/[2026] AP + FVM + comp + Euler_eq556.png)

Notice that this modification has almost no impact in the low-Mach-number regime, but it aims at adding more upwinding and thus improving the stability property of the resulting AP scheme when ε is large;

• Choose the following switching function:


![Equation](images/[2026] AP + FVM + comp + Euler_eq557.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq558.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq559.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq560.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq561.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq562.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq563.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq564.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq565.png)

where ε0, ε1, and α are positive constants taken to be ε0 = 0.15, ε1 = 0.4, and α = 14 in all of the numerical examples below. This switching function is plotted in Figure 4.1.

2 4 6 8 10 1

0.0

0.2

0.4

0.6

0.8

1.0

s( )


> **Figure 4.1: Switching function s(ε) plotted with respect to 1/ε.**

20 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

Example 1—Accuracy Test for Low-Mach-Number Smooth Vortex

In this example taken from [49], we consider a smooth, unsteady Mach dependent vortex over the computational domain [−10, 10] × [−10, 10] subject to the periodic boundary conditions. The analytical solution is given, modulo the periodicity, by


![Equation](images/[2026] AP + FVM + comp + Euler_eq566.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq567.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq568.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq569.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq570.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq571.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq572.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq573.png)

where xr(x, y, t) = (xr, yr)⊤:= (x −t, y −t)⊤. We take the CFL number KCFL = 0.475 and compute the numerical solution until the final time t = 0.1 on a series of uniform N × N meshes with N = 64, 128, 256, and 512 for ε = 1, 0.1, 0.01, and 0.001. We study the convergence and present the obtained results in Figure 4.2, where one can see that the expected second-order convergence rate has been achieved in all variables for all considered ε. One can also observe that, for fixed mesh refinement, the error decreases for decreasing ε as a result of the convergence of the analytical solution to the incompressible limit (ρ, u, v, p)⊤= (1, 1, 1, 1)⊤and of the AP character of the proposed AP DF-FV scheme.

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

u

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

v

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

10 19

10 16

10 13

10 10

10 7

10 4

10 1

400

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

u

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

v

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 19

10 16

10 13

10 10

10 7

10 4

10 1 p

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

u

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

v

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 19

10 16

10 13

10 10

10 7

10 4

10 1 p

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

u

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 9

10 8

10 7

10 6

10 5

10 4

10 3

10 2

v

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2

100 200 400 N

10 19

10 16

10 13

10 10

10 7

10 4

10 1 p

= 1.0 = 0.1 = 0.01 = 0.001

ref. order 1 ref. order 2


> **Figure 4.2: Example 1: Convergence analysis.**

A New AP Method 21

Example 2—Gresho Vortex

This example was introduced in [25] and, since then, it has been widely used as a common benchmark to numerically validate the AP property. We consider a steady vortex over the computational domain [0, 1] × [0, 1] subject to the periodic boundary conditions. At any time t, the shape of the vortex is given by


![Equation](images/[2026] AP + FVM + comp + Euler_eq574.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq575.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq576.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq577.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq578.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq579.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq580.png)

where

xr := x −0.5, yr := y −0.5, r := � x2 r + y2 r, ψ(r) :=


![Equation](images/[2026] AP + FVM + comp + Euler_eq581.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq582.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq583.png)

We take the CFL number KCFL = 0.475 and compute the numerical solution until the final time t = 1 on a uniform 128 × 128 mesh for ε = 10−α with α = 1, . . . , 6, and report the obtained local Mach number, defined as ∥u∥2/√γ, in Figure 4.3 along with its initial distribution. According to what is expected due to the AP feature of the scheme, the shape of the vortex is preserved and no evident dependency on ε can be observed.

Example 3—Baroclinic Vorticity Generation

In this example taken from [41], we consider a low-Mach-number flow with ε = 0.05 involving an acoustic wave, which moves within two density layers in the computational domain [−1


![Equation](images/[2026] AP + FVM + comp + Euler_eq584.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq585.png)

5ε] subject to the periodic boundary conditions. The acoustic wave induces different accelerations in the two density layers, which results in rotational excitation and in the formation of a longwavelength sinusoidal shear layer. Due to the interaction with the acoustic wave, such a shear layer becomes unstable, and several Kelvin-Helmholtz-type unstable structures originate from it. The initial conditions are


![Equation](images/[2026] AP + FVM + comp + Euler_eq586.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq587.png)

u(x, y, 0) = √γ


![Equation](images/[2026] AP + FVM + comp + Euler_eq588.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq589.png)

The numerical solution is computed with the CFL number KCFL = 0.475 until the final time t = 20 on a 800 × 160 uniform mesh. The density at times t = 0, 10, and 20 is plotted in Figure 4.4. Since the solution develops instabilities, no strong convergence is expected in this example; see [49]. One can, however, observe that the underlying physics is correctly captured.

22 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

Initial condition

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 1

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 2

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 3

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 4

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 5

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0.2 0.4 0.6 0.8

0.2

0.4

0.6

0.8

= 10 6

0.0

0.2

0.4

0.6

0.8

1.0

1.2


> **Figure 4.3: Example 2: Initial local Mach number independently of ε (top) and local Mach number at t = 1 for different values of ε.**

Example 4—Double Shear Layer Problem

In the following test case, originally introduced in [8] for the incompressible Navier–Stokes equations and subsequently adopted in, e.g., [10,48,49] in the context of compressible Euler equations in the low-Mach-number regime, a shear layer develops, and the AP property of the proposed scheme can be assessed. In particular, we would like to check whether the scheme maintains its consistency for small values of ε, that is, in the almost incompressible regime. The initial conditions,


![Equation](images/[2026] AP + FVM + comp + Euler_eq590.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq591.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq592.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq593.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq594.png)

2


![Equation](images/[2026] AP + FVM + comp + Euler_eq595.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq596.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq597.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq598.png)

A New AP Method 23

15 10 5 0 5 10 15

2

4

6

t=0

0.0

0.5

1.0

1.5

2.0

15 10 5 0 5 10 15

2

4

6

t=10

0.0

0.5

1.0

1.5

2.0

15 10 5 0 5 10 15

2

4

6

t=20

0.0

0.5

1.0

1.5

2.0


> **Figure 4.4: Example 3: Density at different times.**


![Equation](images/[2026] AP + FVM + comp + Euler_eq599.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq600.png)

are prescribed in the computational domain [0, 2π] × [0, 2π] subject to the periodic boundary conditions. The initial vorticity ω := vx −uy, where the derivatives are approximated using second-order central differences, is plotted in Figure 4.5.

1 2 3 4 5 6

1

2

3

4

5

6

4

2

0

2

4


> **Figure 4.5: Example 4: Initial vorticity.**

We compute the numerical solutions for ε = 10−α with α = 1, . . . , 6 until the final time t = 10 on a 256 × 256 uniform mesh using KCFL = 0.1. Figures 4.6 and 4.7 display the vorticity at times t = 6 and t = 10, respectively, for different ε. The obtained results are consistent with those

24 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

reported in [10, 49]. Moreover, no macroscopic dependence on ε is observed, providing further evidence of the AP property of the proposed DF-FV scheme.

1 2 3 4 5 6

1

2

3

4

5

6

= 10 1

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 2

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 3

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 4

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 5

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 6

4

2

0

2

4


> **Figure 4.6: Example 4: Vorticity at t = 6 for different values of ε. KCFL = 0.1.**

We remark that the simulations remain stable for larger CFL numbers. However, the use of larger KCFL may lead to a noticeable increase in the amount of the numerical diffusion for very small values of ε ≲10−3. To illustrate this, we recompute the solution with KCFL = 0.475 for ε = 10−4, 10−5, and 10−6 and plot the obtained results (for t = 6) in Figure 4.8. As one can clearly see, the numerical solution is now substantially more diffusive compared with those reported in the bottom ’row of Figure 4.6.

Example 5—Explosion Problem

In the last numerical example, we consider an explosion problem taken from [46]. The initial data,


![Equation](images/[2026] AP + FVM + comp + Euler_eq601.png)


![Equation](images/[2026] AP + FVM + comp + Euler_eq602.png)

are prescribed in the computational domain [−1, 1] × [−1, 1] subject to the free boundary conditions. The main objective of this test is to verify that the proposed AP DF-FV scheme remains accurate and stable, also in the high-Mach-number regime, in which strong shocks and contact discontinuities may be present. To this end, we perform simulations for several values of ε. For ε = 1, 0.9, 0.6, and 0.3 the final times are t = 0.25, 0.2, 0.15, and 0.08, respectively. The surface plots of the density ρ computed on a uniform mesh with 400 × 400 cells using KCFL = 0.475 are

A New AP Method 25

1 2 3 4 5 6

1

2

3

4

5

6

= 10 1

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 2

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 3

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 4

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 5

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 6

4

2

0

2

4


> **Figure 4.7: Example 4: The same as in Figure 4.6, but at t = 10.**

1 2 3 4 5 6

1

2

3

4

5

6

= 10 4

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 5

4

2

0

2

4

1 2 3 4 5 6

1

2

3

4

5

6

= 10 6

4

2

0

2

4


> **Figure 4.8: Example 4: The same as in Figure 4.6, but for KCFL = 0.475 (left).**

reported in Figure 4.9, where one can see that the obtained solutions are oscillation-free and their nonsmooth features are accurately resolved for all values of ε. To further assess the correctness of the computed solutions, we plot their one-dimensional (1-D) slices along the diagonal y = x in Figure 4.10 together with the corresponding slices of the reference solution, which was obtained using the second-order semi-discrete CU scheme from [32] on a much finer mesh with 2000×2000 cells using the CFL number 0.2 and the three-stage third-order strong stability preserving (SSP) Runge-Kutta method [23, 24]. As one can clearly see, the computed solutions show a perfect agreement with the reference ones, and the discontinuities locations are correctly captured.

Remark 4.1 We stress that in this example, both terms in the numerator in (3.10) will vanish

26 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi


> **Figure 4.9: Example 5: Surface plot of density for different values of ε and corresponding times.**

if ρmax and pmin are computed using (3.5). While the modification (4.1) ensures positivity of ˜c, the resulting time steps might still be too big to guarantee stability of the AP DF-FV method. Therefore, we set ∆t = 10−4 for the first 10 time steps for the simulations involving ε = 0.6 and 0.3.

Remark 4.2 Let us remark that discontinuities are unlikely to occur in low-Mach-number flows. Consequently, the above tests with ε = 0.6 and 0.3 should be regarded as “academic” and are primarily intended to demonstrate that the proposed AP DF-FV scheme is capable of handling discontinuities even in the low-Mach-number regime.


## 5 Conclusion

We have presented a novel asymptotic-preserving (AP) numerical method for the compressible Euler equations that is effective across all Mach-number regimes, including the low-Mach-number one, where standard explicit schemes become inefficient. The key idea is a new hyperbolic splitting, inspired by the flux-splitting approach introduced in [26]. The new splitting is applied to a primitive (nonconservative) formulation of the Euler equations, which enables one to design an

A New AP Method 27

0.2

0.4

0.6

0.8

1.0

AP DF-FV reference

0.0

0.2

0.5

0.8

1.0

1.2

u2 + v2

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

p

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

AP DF-FV reference

0.0

0.2

0.5

0.8

1.0

1.2

1.5

u2 + v2

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

p

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

AP DF-FV reference

0.0

0.5

1.0

1.5

2.0

2.5

u2 + v2

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

p

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

AP DF-FV reference

0.0

1.0

2.0

3.0

4.0

u2 + v2

AP DF-FV reference

0.2

0.4

0.6

0.8

1.0

p

AP DF-FV reference


> **Figure 4.10: Example 5: 1-D slices of the computed solutions along y = x for different values of ε and at different times: ε = 1, t = 0.25 (top row), ε = 0.9, t = 0.2 (second row), ε = 0.6, t = 0.15 (third row), and ε = 0.3, t = 0.08 (bottom row).**

efficient semi-implicit (SI) time discretization. Our splitting isolates stiff linear terms, which are discretized semi-implicitly: this leads to a well-posed linear elliptic problem, which ensures the AP property of the resulting scheme.

To overcome the well-known difficulties associated with the use of nonconservative formulations in the presence of discontinuities, we implement the proposed AP scheme within the recently introduced dual formulation framework [3, 14]. In this approach, the conservative and primitive

28 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

systems are solved simultaneously, and their resulting solutions are post-processed to ensure the correct capturing of discontinuities while retaining the AP property of the primitive-based SI approach. The proposed AP dual formulation finite-volume (DF-FV) method has been thoroughly validated on several benchmarks ranging from the fully compressible to the nearly incompressible regime, demonstrating both high accuracy and robustness of the method. Future work will focus on extending the AP DF-FV framework to more complex systems and on developing higher-order spatial and temporal discretizations.

Acknowledgment: The work of A. Chertock was supported in part by NSF grant DMS-2208438. The work of A. Kurganov was supported in part by NSFC grant W2431004. The work of L. Micalizzi was supported in part by the LeRoy B. Martin, Jr. Distinguished Professorship Foundation.


## References

[1] R. Abgrall, High order schemes for hyperbolic problems using globally continuous approximation and avoiding mass matrices, J. Sci. Comput., 73 (2017), pp. 461–494.

[2] , A combination of residual distribution and the active flux formulations or a new class of schemes that can combine several writings of the same hyperbolic problem: application to the 1D Euler equations, Commun. Appl. Math. Comput., 5 (2023), pp. 370–402.

[3] R. Abgrall, A. Chertock, A. Kurganov, and L. Micalizzi, Dual formulation finitevolume methods on overlapping meshes for hyperbolic conservation laws, Comput. & Fluids, 307 (2026). Paper No. 106952.

[4] R. Abgrall and S. Karni, A comment on the computation of non-conservative products, J. Comput. Phys., 229 (2010), pp. 2759–2763.

[5] R. Abgrall and Y. Liu, A new approach for designing well-balanced schemes for the shallow water equations: a combination of conservative and primitive formulations, SIAM J. Sci. Comput., 46 (2024), pp. A3375–A3400.

[6] T. Alazard, Incompressible limit of the nonisentropic Euler equations with the solid wall boundary conditions, Adv. Differential Equations, 10 (2005), pp. 19–44.

[7] P. Allegrini and M.-H. Vignal, Study of a new low-oscillating second-order all-Mach number IMEX finite volume scheme for the full Euler equations, SIAM J. Sci. Comput., 47 (2025), pp. A268–A299.

[8] J. B. Bell, P. Colella, and H. M. Glaz, A second-order projection method for the incompressible Navier-Stokes equations, J. Comput. Phys., 85 (1989), pp. 257–283.

[9] S. Boscarino, J.-M. Qiu, G. Russo, and T. Xiong, A high order semi-implicit IMEX WENO scheme for the all-Mach isentropic Euler system, J. Comput. Phys., 392 (2019), pp. 594–618.

A New AP Method 29

[10] S. Boscarino, G. Russo, and L. Scandurra, All Mach number second order semiimplicit scheme for the Euler equations of gas dynamics, J. Sci. Comput., 77 (2018), pp. 850– 884.

[11] W. Boscheri, G. Dimarco, R. Loub`ere, M. Tavelli, and M.-H. Vignal, A second order all Mach number IMEX finite volume solver for the three dimensional Euler equations, J. Comput. Phys., 415 (2020). Paper No. 109486.

[12] M. J. Castro D´ıaz, A. Kurganov, and T. Morales de Luna, Path-conservative central-upwind schemes for nonconservative hyperbolic systems, ESAIM Math. Model. Numer. Anal., 53 (2019), pp. 959–985.

[13] C. Chalons, M. Girardin, and S. Kokh, An all-regime Lagrange-projection like scheme for the gas dynamics equations on unstructured meshes, Commun. Comput. Phys., 20 (2016), pp. 188–233.

[14] A. Chertock, Q. Fu, A. Kurganov, and L. Micalizzi, New adaptive numerical methods based on dual formulation of hyperbolic conservation laws. Submitted; arXiv:2601.20000.

[15] S. Chu, A. Kurganov, and M. Na, Fifth-order A-WENO schemes based on the pathconservative central-upwind method, J. Comput. Phys., 469 (2022). Paper No. 111508.

[16] F. Cordier, P. Degond, and A. Kumbaro, An asymptotic-preserving all-speed scheme for the Euler and Navier-Stokes equations, J. Comput. Phys., 231 (2012), pp. 5685–5704.

[17] P. Degond, S. Jin, and J.-G. Liu, Mach-number uniform asymptotic-preserving gauge schemes for compressible flows, Bull. Inst. Math. Acad. Sin. (N.S.), 2 (2007), pp. 851–892.

[18] P. Degond and M. Tang, All speed scheme for the low Mach number limit of the isentropic Euler equations, Commun. Comput. Phys., 10 (2011), pp. 1–31.

[19] G. Dimarco, R. Loub`ere, V. Michel-Dansac, and M.-H. Vignal, Second-order implicit-explicit total variation diminishing schemes for the Euler system in the low Mach regime, J. Comput. Phys., 372 (2018), pp. 178–201.

[20] G. Dimarco, R. Loub`ere, and M.-H. Vignal, Study of a new asymptotic preserving scheme for the Euler system in the low Mach number limit, SIAM J. Sci. Comput., 39 (2017), pp. A2099–A2128.

[21] L. Fox and E. T. Goodwin, Some new methods for the numerical integration of ordinary differential equations, Proc. Cambridge Philos. Soc., 45 (1949), pp. 373–388.

[22] F. Golse, S. Jin, and C. D. Levermore, The convergence of numerical transfer schemes in diffusive regimes. I. Discrete-ordinate method, SIAM J. Numer. Anal., 36 (1999), pp. 1333– 1369.

[23] S. Gottlieb, D. Ketcheson, and C.-W. Shu, Strong stability preserving Runge-Kutta and multistep time discretizations, World Scientific Publishing Co. Pte. Ltd., Hackensack, NJ, 2011.

30 A. Chertock, S. Karni, A. Kurganov & L. Micalizzi

[24] S. Gottlieb, C.-W. Shu, and E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM Rev., 43 (2001), pp. 89–112.

[25] P. M. Gresho and S. T. Chan, On the theory of semi-implicit projection methods for viscous incompressible flow and its implementation via a finite element method that also introduces a nearly consistent mass matrix. II. Implementation, Internat. J. Numer. Methods Fluids, 11 (1990), pp. 621–659. Computational methods in flow analysis (Okayama, 1988).

[26] J. Haack, S. Jin, and J.-G. Liu, An all-speed asymptotic-preserving method for the isentropic Euler and Navier-Stokes equations, Commun. Comput. Phys., 12 (2012), pp. 955–980.

[27] T. Y. Hou and P. G. LeFloch, Why nonconservative schemes converge to wrong solutions: error analysis, Math. Comp., 62 (1994), pp. 497–530.

[28] S. Jin, Efficient asymptotic-preserving (AP) schemes for some multiscale kinetic equations, SIAM J. Sci. Comput., 21 (1999), pp. 441–454 (electronic).

[29] T. Kato, Perturbation theory for linear operators, Classics in Mathematics, Springer-Verlag, Berlin, 1980 ed., 1995.

[30] A. Klar, An asymptotic preserving numerical scheme for kinetic equations in the low Mach number limit, SIAM J. Numer. Anal., 36 (1999), pp. 1507–1527.

[31] R. Klein, Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics, I: One-dimensional flow, J. Comput. Phys., 121 (1995), pp. 213–237.

[32] A. Kurganov and C.-T. Lin, On the reduction of numerical dissipation in central-upwind schemes, Commun. Comput. Phys., 2 (2007), pp. 141–163.

[33] E. W. Larsen and J. E. Morel, Asymptotic solutions of numerical transport problems in optically thick, diffusive regimes. II, J. Comput. Phys., 83 (1989), pp. 212–236.

[34] E. W. Larsen, J. E. Morel, and W. F. Miller, Jr., Asymptotic solutions of numerical transport problems in optically thick, diffusive regimes, J. Comput. Phys., 69 (1987), pp. 283– 324.

[35] K.-A. Lie and S. Noelle, An improved quadrature rule for the flux-computation in staggered central difference schemes in multidimensions, J. Sci. Comput., 63 (2003), pp. 1539– 1560.

[36] X. Liu, A. Chertock, and A. Kurganov, An asymptotic preserving scheme for the two-dimensional shallow water equations with Coriolis forces, J. Comput. Phys., 391 (2019), pp. 259–279.

[37] G. M´etivier and S. Schochet, The incompressible limit of the non-isentropic Euler equations, Arch. Ration. Mech. Anal., 158 (2001), pp. 61–90.

[38] L. Micalizzi and D. Torlo, A new efficient explicit deferred correction framework: analysis and applications to hyperbolic PDEs and adaptivity, Commun. Appl. Math. Comput., 6 (2024), pp. 1629–1664.

A New AP Method 31

[39] L. Micalizzi, D. Torlo, and W. Boscheri, Efficient iterative arbitrary high-order methods: an adaptive bridge between low and high order, Commun. Appl. Math. Comput., 7 (2025), pp. 40–77.

[40] H. Nessyahu and E. Tadmor, Nonoscillatory central differencing for hyperbolic conservation laws, J. Comput. Phys., 87 (1990), pp. 408–463.

[41] S. Noelle, G. Bispen, K. R. Arun, M. Luk´aˇcov´a-Medvid’ov´a, and C.-D. Munz, A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas dynamics, SIAM J. Sci. Comput., 36 (2014), pp. B989–B1024.

[42] P. ¨Offner, L. Petri, and D. Torlo, Analysis for implicit and implicit-explicit ADER and DeC methods for ordinary differential equations, advection-diffusion and advection-dispersion equations, Appl. Numer. Math., 212 (2025), pp. 110–134.

[43] R. M. Pidatella, G. Puppo, G. Russo, and P. Santagati, Semi-conservative finite volume schemes for conservation laws, SIAM J. Sci. Comput., 41 (2019), pp. B576–B600.

[44] P. K. Sweby, High resolution schemes using flux limiters for hyperbolic conservation laws, SIAM J. Numer. Anal., 21 (1984), pp. 995–1011.

[45] M. Tang, Second order all speed method for the isentropic Euler equations, Kinet. Relat. Models, 5 (2012), pp. 155–184.

[46] E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer-Verlag, Berlin, third ed., 2009.

[47] E. F. Toro and M. E. V´azquez-Cend´on, Flux splitting schemes for the Euler equations, Comput. & Fluids, 70 (2012), pp. 1–12.

[48] E. Weinan and C.-W. Shu, A numerical resolution study of high order essentially nonoscillatory schemes applied to incompressible flow, J. Comput. Phys., 110 (1994), pp. 39–46.

[49] J. Zeifang, J. Sch¨utz, K. Kaiser, A. Beck, M. Luk´aˇcov´a-Medvid’ov´a, and S. Noelle, A novel full-Euler low Mach number IMEX splitting, Commun. Comput. Phys., 27 (2020), pp. 292–320.

