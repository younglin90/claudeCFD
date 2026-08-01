
## Breaching the capillary time-step constraint using a coupled VOF method with implicit surface tension


![Equation](images/2203.01963_eq001.png)

Chair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany

Abstract

The capillary time-step constraint is the dominant limitation on the applicable time-step in many simulations of interfacial flows with surface tension and, consequently, governs the execution time of these simulations. We propose a fully-coupled pressure-based algorithm based on an algebraic Volume-of-Fluid (VOF) method in conjunction with an implicit linearised surface tension treatment that can breach the capillary time-step constraint. The advection of the interface is solved together with the momentum and continuity equations of the interfacial flow in a single system of linearised equations, providing an implicit coupling between pressure, velocity and the VOF colour function used to distinguish the interacting fluids. Surface tension is treated with an implicit formulation of the Continuum Surface Force (CSF) model, whereby both the interface curvature and the gradient of the colour function are treated implicitly with respect to the colour function. The presented results demonstrate that a time-step larger than the capillary time-step can be applied with this new numerical framework, as long as other relevant time-step restrictions are satisfied, including a time-step restriction associated with surface tension, density as well as viscosity.

Keywords: Capillary time-step constraint, Surface tension, Coupled algorithm, Volume-of-Fluid method

© 2022. This manuscript version is made available under the CC-BY-NC-ND 4.0 license. http://creativecommons.org/licenses/by-nc-nd/4.0/

1. Introduction

The temporal resolution of the propagation of the smallest capillary waves discretely resolved in space on a fluid interface presents the dominant time-step restriction for the majority of simulations of interfacial flows with surface tension [1]. It has long been speculated that an implicit formulation of surface tension must be able to eliminate, or at least mitigate, this time-step constraint [1–6]. To date, however, no numerical algorithm based on an interface capturing method, such as a Volume-of-Fluid (VOF), level-set or phasefield method, has been reported in the literature that can breach the capillary time-step constraint, while still representing the governing physics faithfully and without adding unphysical terms to the governing equations. The capillary time-step constraint arises from the phase velocity c of capillary waves, given as [7]

c =


![Equation](images/2203.01963_eq002.png)

where σ is the surface tension coefficient, λ is the wavelength, and ρa and ρb are the densities of the interacting fluids. Because the phase velocity is inversely proportional to √

λ, shorter capillary waves with an increasing phase velocity are spatially resolved as the computational mesh is refined. Brackbill et al. [2] were the first to recognise this time-step restriction. Considering that the shortest unambiguously resolved capillary waves have a wavenumber of kσ = π/∆x and, consequently, a wavelength of λσ = 2π/kσ = 2∆x, where ∆x is the mesh spacing, Brackbill et al. [2] proposed the capillary time-step constraint as


![Equation](images/2203.01963_eq003.png)


![Equation](images/2203.01963_eq004.png)


![Equation](images/2203.01963_eq005.png)


![Equation](images/2203.01963_eq006.png)

∗fabian.denner@ovgu.de

1


# arXiv:2203.01963v1  [physics.comp-ph]  3 Mar 2022

The capillary time-step constraint is, therefore, a Courant–Friedrichs–Lewy (CFL) condition [8] associated with the phase velocity of capillary waves, ∆tσ ∝∆x/c, with the factor 2 in Eq. (2) accounting for the case in which two oppositely propagating waves enter the same mesh cell simultaneously. As part of our previous work [9], we revisited the origin of the capillary time-step constraint using both numerical and signal-processing arguments, and arrived at a similar but slightly different formulation of the capillary timestep constraint than Brackbill et al. [2], given as1


![Equation](images/2203.01963_eq007.png)


![Equation](images/2203.01963_eq008.png)


![Equation](images/2203.01963_eq009.png)


![Equation](images/2203.01963_eq010.png)

We also demonstrated that, if the velocity of the flow at the interface has a magnitude similar to the phase velocity c, the Doppler shift associated with capillary waves propagating along a moving interface ought to be taken into account, and provided the first numerical results that clearly delineate the capillary time-step constraint. Contrary to the traditional CFL condition, ∆t ≤∆x/|u| [8], which arises from the flow velocity u and is proportional to the mesh spacing ∆x, the capillary time-step constraint is proportional to ∆x3/2, since c ∝∆x−1/2. As a consequence, the capillary time-step constraint dominates the maximum applicable timestep for interfacial flows at small lengthscales (e.g. microfluidics), applications with quasi-static heat and mass transfer (e.g. the evaporation of a sessile drop) and, in general, simulations with high spatial resolution, as it is now routinely afforded by adaptive mesh refinement algorithms in conjunction with modern highperformance computing resources. Even at very small scales, such as in microfluidic applications, viscous contributions with a physical origin are typically not able to mitigate the capillary time-step constraint [1, 9]. Already in their original proposition of the capillary time-step constraint, Brackbill et al. [2] hypothesised that an implicit treatment of surface tension should allow to breach or even eliminate the capillary timestep constraint. Hou et al. [10] presented a boundary integral method with an implicit surface tension treatment for irrotational incompressible flows in two dimensions that is evidently free of the capillary time-step constraint, demonstrating the necessity of an implicit surface tension treatment in eliminating the capillary time-step constraint. Following the work of B¨ansch [11], Hysing [4] proposed a semi-implicit surface tension treatment for a two-dimensional finite-element method, which, in essence, incorporates the interface position at the new time instance implicitly. Raessi et al. [12] translated this surface tension treatment subsequently to finite-volume methods. Hysing [4] and Raessi et al. [12] reported stable results for time-steps exceeding the capillary time-step constraint, although the solution was not stable for arbitrarily large time-steps (assuming other relevant time-step restrictions, e.g. the CFL condition, are satisfied). We previously proposed an algorithm in which the continuity, momentum and VOF advection equations are implicitly coupled and solved as a single system of equations [9], treating the surface tension term semiimplicit with respect to the VOF colour function. However, this method does not allow to breach the capillary time-step constraint, which led us to conclude (incorrectly) that an implicit formulation of the source term representing surface tension cannot eliminate the capillary time-step constraint using interface capturing methods [9]. A fully-Lagrangian method for incompressible free-surface flows was proposed by Zheng et al. [13], based on a Marker-and-Cell (MAC) method in conjunction with a Lagrangian fluid mesh in the vicinity of the fluid interface, which are coupled implicitly and ensure an exact balance between surface tension and pressure. To date, this is the only method for three-dimensional interfacial flows that has been demonstrated to be practically free of the capillary time-step constraint, whereas a numerical framework based on an interface capturing method with comparable capabilities has yet to be published. In this article, we propose a numerical framework for interfacial flows based on a VOF method that is able to breach the capillary time-step constraint. We achieve this by extending a fully-coupled pressure-based algorithm [14] with an algebraic VOF method that is implicitly coupled to the governing flow equations, enabling an implicit treatment of surface tension based on the Continuum Surface Force (CSF) model [2]. The presented results demonstrate that a time-step larger than the capillary time-step constraint can be applied with this new numerical framework, as long as other relevant time-step restrictions (e.g. the CFL condition) are satisfied. However, a new time-step constraint arises that depends on surface tension, density and viscosity, but which is less restrictive than the classical capillary time-step constraint.

1The effective phase velocity in the case where two oppositely propagating waves enter the same mesh cell simultaneously increases by factor √

2, a result directly following from Eq. (1), rather than factor 2 [9].

2

2. Mathematical model

In this study, we consider interfacial flows of two immiscible and incompressible Newtonian fluids. Such an interfacial flow is governed by the continuity equation


![Equation](images/2203.01963_eq011.png)

with x the spatial coordinate and u the velocity vector, and the momentum equations


![Equation](images/2203.01963_eq012.png)


![Equation](images/2203.01963_eq013.png)


![Equation](images/2203.01963_eq014.png)


![Equation](images/2203.01963_eq015.png)


![Equation](images/2203.01963_eq016.png)

where ρ is the density, t denotes time, p is the pressure and S is the source term representing surface tension. The stress tensor τ is given as


![Equation](images/2203.01963_eq017.png)


![Equation](images/2203.01963_eq018.png)


![Equation](images/2203.01963_eq019.png)

where µ is the dynamic viscosity. A VOF method [15] is adopted to model the transport and interaction of two immiscible fluids. The two fluids are distinguished by the indicator function ζ, which is defined as


![Equation](images/2203.01963_eq020.png)


![Equation](images/2203.01963_eq021.png)

where Ω= Ωa ∪Ωb is the computational domain, with Ωa and Ωb the subdomains occupied by fluids “a” and “b”, respectively. The indicator function ζ is advected by the underlying flow


![Equation](images/2203.01963_eq022.png)


![Equation](images/2203.01963_eq023.png)


![Equation](images/2203.01963_eq024.png)


![Equation](images/2203.01963_eq025.png)


![Equation](images/2203.01963_eq026.png)

The fluid properties are defined over the entire computational domain using the indicator function, e.g. for density ρ = (1 −ζ)ρa + ζρb. However, to focus the discussion on the discretisation and influence of surface tension, both interacting fluids are assumed to have the same density, ρ = ρa = ρb, and viscosity, µ = µa = µb, throughout this study.

3. Numerical framework

The proposed numerical framework builds upon a class of fully-coupled pressure-based algorithms for single-phase [14, 16] and interfacial flows [9, 17], with the aim of treating all discretised and linearised governing equations implicitly in a single linear system of governing equations, A · φ = b. This linear system of governing equations is solved simultaneously for the pressure p, the velocity u ≡(u v w)T and the discrete colour function ψ, using the Block-Jacobi pre-conditioner and the BiCGSTAB solver of the software library PETSc [18, 19]. The nonlinear nature of the governing equations is accounted for by means of an inexact Newton method [20], whereby the deferred terms resulting from the applied linearisation procedure are updated iteratively, until the nonlinear system of governing equations satisfies predefined conservation criteria, as illustrated in Figure 1. The discretisation of the governing equations is based on an established second-order finite-volume method and utilises a collocated variable arrangement [14], with the fluxes through the mesh faces computed by a momentum-weighted interpolation [21]. The proposed numerical framework is not inherently limited to VOF methods, but may similarly be used in conjunction with level-set or phase-field methods.

3.1. Discretised governing equations

In the following we assume an equidistant Cartesian mesh with mesh spacing ∆x. The continuity equation (4) for each computational cell P is readily discretised as �

f F (n+1) f = 0, (9)

3


![Equation](images/2203.01963_eq027.png)


![Equation](images/2203.01963_eq028.png)


![Equation](images/2203.01963_eq029.png)

Gather coefficients and assemble A and b


![Equation](images/2203.01963_eq030.png)


![Equation](images/2203.01963_eq031.png)


![Equation](images/2203.01963_eq032.png)

Conservation satisfied? no yes


![Equation](images/2203.01963_eq033.png)


![Equation](images/2203.01963_eq034.png)


> **Figure 1: Flow chart of the solution procedure of the discretised and linearised system of governing equations, where n is the nonlinear iteration counter, χ ∈{p, u, v, w, ψ} are the solution variables, κ is the interface curvature (see Section 3.2) and ϑf is the advecting velocity through mesh face f (see Section 3.3). The coefficient matrix A holds all coefficients for the implicitly sought solution variables χ(n+1) of the discretised governing equations (see Section 3.4) and φ is the solution vector. The right-hand side vector b holds the deferred contributions of the previous iteration (χ(n), κ(n), ϑ(n) f ) and the contributions of**

the previous time-levels (χ(t−∆t), χ(t−2∆t), ϑ(t−∆t) f ).

where subscript f denotes faces adjacent to mesh cell P, Ff = ϑfAf is the flux through face f, Af is the area of face f and ϑf is the advecting velocity (outward pointing with respect to cell P) obtained from a momentum-weighted interpolation, as further detailed in Section 3.3. The superscript (n + 1) denotes implicitly solved quantities and the superscript (n) denotes deferred quantities, where n is the nonlinear iteration counter. Applying the second-order backward Euler scheme to discretise the transient term [22], the discretised momentum equations (5) follow as


![Equation](images/2203.01963_eq035.png)



3u(n+1) j,P −4u(t−∆t) j,P + u(t−2∆t) j,P 2∆t VP + �


![Equation](images/2203.01963_eq036.png)


![Equation](images/2203.01963_eq037.png)


![Equation](images/2203.01963_eq038.png)

= − �

f


![Equation](images/2203.01963_eq039.png)


![Equation](images/2203.01963_eq040.png)


![Equation](images/2203.01963_eq041.png)


![Equation](images/2203.01963_eq042.png)


![Equation](images/2203.01963_eq043.png)


![Equation](images/2203.01963_eq044.png)


![Equation](images/2203.01963_eq045.png)


![Equation](images/2203.01963_eq046.png)


![Equation](images/2203.01963_eq047.png)


![Equation](images/2203.01963_eq048.png)

where ˜□denotes flux-limited values at face f (see below), □f = (□P + □Q)/2 denotes a linear interpolation of the cell-centred values to face f, with Q the neighbour cell of P adjacent to face f, nf is the normal vector of face f (outward pointing with respect to cell P) and, as further described in Section 3.2, S is the source term representing surface tension. A Newton linearisation is applied to the advection term of the momentum equations [16], to enable an implicit treatment of both the velocity ˜uf and the flux Ff. The two interacting fluids are represented discretely by the colour function ψ,


![Equation](images/2203.01963_eq049.png)

VP

˚


![Equation](images/2203.01963_eq050.png)

4

which is advected based on Eq. (8) by

3ψ(n+1) P −4ψ(t−∆t) P + ψ(t−2∆t) P 2∆t VP + �


![Equation](images/2203.01963_eq051.png)


![Equation](images/2203.01963_eq052.png)


![Equation](images/2203.01963_eq053.png)


![Equation](images/2203.01963_eq054.png)


![Equation](images/2203.01963_eq055.png)


![Equation](images/2203.01963_eq056.png)


![Equation](images/2203.01963_eq057.png)


![Equation](images/2203.01963_eq058.png)


![Equation](images/2203.01963_eq059.png)


![Equation](images/2203.01963_eq060.png)


![Equation](images/2203.01963_eq061.png)

In analogy to the discretisation of the momentum equations (10), the transient term is discretised with the second-order backward Euler scheme and both spatial terms are linearised with a Newton linearisation. The flux-limited face values are given as

˜ψ(n+1) f =


![Equation](images/2203.01963_eq062.png)


![Equation](images/2203.01963_eq063.png)

and

˜u(n+1) f =


![Equation](images/2203.01963_eq064.png)


![Equation](images/2203.01963_eq065.png)


![Equation](images/2203.01963_eq066.png)


![Equation](images/2203.01963_eq067.png)

where ξ(ψ) f is the flux limiter of the colour function, εtol = 10−6 is a predefined tolerance, and subscripts U and D denote the upwind and downwind cells of face f, respectively. For the purpose of this study, the CICSAM scheme [23] is used to compute the flux limiter ξ(ψ) f , but any other advection scheme may also be applied. Note that the advection scheme applied to the colour function in Eq. (13) away from the interface, where ψ = const., is irrelevant and choosing upwind differencing is here a matter of convenience.

3.2. Surface tension

Surface tension is modelled using the CSF model [2], with which the source term S representing surface tension is given as


![Equation](images/2203.01963_eq068.png)

where σ is the surface tension coefficient and κ is the interface curvature. The gradient of the colour function is discretised using the Gauss theorem, analogous to the discretisation of the pressure gradient in the momentum equations (10), thus it is given as


![Equation](images/2203.01963_eq069.png)


![Equation](images/2203.01963_eq070.png)


![Equation](images/2203.01963_eq071.png)


![Equation](images/2203.01963_eq072.png)


![Equation](images/2203.01963_eq073.png)

which facilitates a force-balanced discretisation [17]. The source term S is treated implicitly by applying a Newton linearisation of the interface curvature and the gradient of the colour function to yield


![Equation](images/2203.01963_eq074.png)

which, by inserting Eq. (16), becomes

S(n+1) P ≈σ


![Equation](images/2203.01963_eq075.png)


![Equation](images/2203.01963_eq076.png)


![Equation](images/2203.01963_eq077.png)


![Equation](images/2203.01963_eq078.png)


![Equation](images/2203.01963_eq079.png)


![Equation](images/2203.01963_eq080.png)


![Equation](images/2203.01963_eq081.png)


![Equation](images/2203.01963_eq082.png)


![Equation](images/2203.01963_eq083.png)


![Equation](images/2203.01963_eq084.png)


![Equation](images/2203.01963_eq085.png)

Various approaches have been proposed for estimating the interface curvature from the discrete colour function, e.g. [2, 5, 17, 24–28]. In this work, we employ the height-function (HF) method [28] both for its superior accuracy and ease of linearisation. In the HF method, when the z-component of the interface

5

normal is dominant, curvature reads as

κ = −Hxx � 1 + H2 y � −Hyy � 1 + H2 x � + 2HxHyHxy � H2x + H2y + 1 �3/2 , (19)

where H{x,y,xx,yy,xy} are the first and second partial derivatives of the “heights of fluid” computed along the z-direction2. On a Cartesian mesh, a height of fluid can be trivially obtained by summing the colour function in a column of computational cells. The partial derivatives of the heights are calculated using central differences, therefore they can be expressed as linear combinations of the discrete colour function values in the set of cells from which the heights are computed. Consider, for instance, an interfacial computational cell P, ψP ∈]0, 1[, within which the z-component of the interface normal is dominant. One can construct a stencil S(P) centred around P, containing at most 3 × 3 × NH cells, with NH an odd number typically chosen between 5 and 9. For the portion of interface in cell P, the first partial derivative of the heights along the x-direction is approximated as

Hx = �


![Equation](images/2203.01963_eq086.png)

where βx,N are coefficients arising from the application of central differences to the heights, themselves calculated as the sum of the discrete colour function values in the columns of NH computational cells of S(P). The other partial derivatives of the heights are expressed similarly. In the cell P, applying a Newton linearisation on curvature yields

κ(n+1) P = κ(n) P + �


![Equation](images/2203.01963_eq087.png)


![Equation](images/2203.01963_eq088.png)


![Equation](images/2203.01963_eq089.png)


![Equation](images/2203.01963_eq090.png)

Introducing the quantities

NP = −Hxx � 1 + H2 y � −Hyy � 1 + H2 x � + 2HxHyHxy (22)

DP = H2 x + H2 y + 1 (23)

the linearisation of curvature can be reformulated as

κ(n+1) P = κ(n) P + 1


![Equation](images/2203.01963_eq091.png)


![Equation](images/2203.01963_eq092.png)


![Equation](images/2203.01963_eq093.png)


![Equation](images/2203.01963_eq094.png)


![Equation](images/2203.01963_eq095.png)


![Equation](images/2203.01963_eq096.png)


![Equation](images/2203.01963_eq097.png)


![Equation](images/2203.01963_eq098.png)


![Equation](images/2203.01963_eq099.png)


![Equation](images/2203.01963_eq100.png)

The partial derivatives on the right-hand side of Eq. (24) read as

∂NP ∂ψN = − � 1 + H2 y � βxx,N − � 1 + H2 x � βyy,N


![Equation](images/2203.01963_eq101.png)

Note that for two-dimensional flow in the plane (x, z), the previous expressions reduce to


![Equation](images/2203.01963_eq102.png)

3.3. Momentum-weighted interpolation

The advecting velocity ϑf = uf·nf of the flux Ff = ϑfAf through face f is discretised using a momentumweighted interpolation (MWI) [29], which provides a direct coupling of pressure and velocity that eliminates pressure-velocity decoupling as a result of the applied collocated variable arrangement. Including the source

2In case the xor y-component of the normal is dominant, the indices of the partial derivatives are simply permuted.

6

term representing surface tension, the advecting velocity is defined as [17, 21]

ϑf = uf · nf −ˆdf � ∇pf −∇pf � · nf + ˆdf � Sf −Sf � · nf + ˆdf ρ ∆t


![Equation](images/2203.01963_eq103.png)

The coefficient ˆdf, derived in detail in [21], represents the weighting factor of the MWI correction terms and, in the context of a fully-coupled algorithm, the strength of the implicit coupling provided by the MWI. The pressure gradient at the face is discretised as


![Equation](images/2203.01963_eq104.png)


![Equation](images/2203.01963_eq105.png)

and, analogous to the pressure term in the momentum equations (10), the cell-centred pressure gradient is discretised using the Gauss theorem as


![Equation](images/2203.01963_eq106.png)


![Equation](images/2203.01963_eq107.png)


![Equation](images/2203.01963_eq108.png)


![Equation](images/2203.01963_eq109.png)


![Equation](images/2203.01963_eq110.png)

Together, the pressure terms in Eq. (29), (∇pf −∇pf), then act as a low-pass filter with respect to pressure [21]. Similarly, the surface tension terms are discretised as [17]


![Equation](images/2203.01963_eq111.png)


![Equation](images/2203.01963_eq112.png)

and


![Equation](images/2203.01963_eq113.png)


![Equation](images/2203.01963_eq114.png)


![Equation](images/2203.01963_eq115.png)


![Equation](images/2203.01963_eq116.png)


![Equation](images/2203.01963_eq117.png)

Applying the discretisation described above, along with a Newton linearisation of the surface tension terms analogous to the source term representing surface tension in the momentum equations, the advecting velocity is proposed as

ϑ(n+1) f = u(n+1) f · nf −ˆdf


![Equation](images/2203.01963_eq118.png)

2


![Equation](images/2203.01963_eq119.png)


![Equation](images/2203.01963_eq120.png)


![Equation](images/2203.01963_eq121.png)


![Equation](images/2203.01963_eq122.png)


![Equation](images/2203.01963_eq123.png)

2


![Equation](images/2203.01963_eq124.png)


![Equation](images/2203.01963_eq125.png)


![Equation](images/2203.01963_eq126.png)


![Equation](images/2203.01963_eq127.png)


![Equation](images/2203.01963_eq128.png)

2


![Equation](images/2203.01963_eq129.png)


![Equation](images/2203.01963_eq130.png)


![Equation](images/2203.01963_eq131.png)


![Equation](images/2203.01963_eq132.png)


![Equation](images/2203.01963_eq133.png)

2


![Equation](images/2203.01963_eq134.png)


![Equation](images/2203.01963_eq135.png)


![Equation](images/2203.01963_eq136.png)


![Equation](images/2203.01963_eq137.png)


![Equation](images/2203.01963_eq138.png)

With this discrete formulation, each term of the current time-level in the advecting velocity, Eq. (29), makes an implicit contribution to the solution variables χ ∈{p, u, v, w, ψ}.

3.4. Solution procedure

The discretised governing equations (9), (10) and (12) are solved simultaneously in a single linear system of discretised equations, given for a three-dimensional mesh with N cells as 

     

Ap cont. Au cont. Av cont. Aw cont. Aψ cont. Ap x-mom. Au x-mom. Av x-mom. Aw x-mom. Aψ x-mom. Ap y-mom. Au y-mom. Av y-mom. Aw y-mom. Aψ y-mom. Ap z-mom. Au z-mom. Av z-mom. Aw z-mom. Aψ z-mom. Ap vof Au vof Av vof Aw vof Aψ vof


![Equation](images/2203.01963_eq139.png)


![Equation](images/2203.01963_eq140.png)


![Equation](images/2203.01963_eq141.png)


![Equation](images/2203.01963_eq142.png)


![Equation](images/2203.01963_eq143.png)


![Equation](images/2203.01963_eq144.png)


![Equation](images/2203.01963_eq145.png)


![Equation](images/2203.01963_eq146.png)


![Equation](images/2203.01963_eq147.png)


![Equation](images/2203.01963_eq148.png)


![Equation](images/2203.01963_eq149.png)


![Equation](images/2203.01963_eq150.png)


![Equation](images/2203.01963_eq151.png)


![Equation](images/2203.01963_eq152.png)


![Equation](images/2203.01963_eq153.png)


![Equation](images/2203.01963_eq154.png)

7

for the solution variables pressure p, velocity u ≡(u v w)T and colour function ψ. In Eq. (35), Aχ eq. denotes the coefficient submatrix of size N × N of the continuity equation (eq. = cont.), the momentum equations associated with the three Cartesian coordinate axes (eq. = x-mom., eq. = y-mom., eq. = z-mom.) and the VOF advection equation (eq. = vof) for the respective solution variable χ ∈{p, u, v, w, ψ}. The solution subvectors of length N for solution variable χ are denoted as φχ and the right-hand side subvectors of length N of the five discretised governing equations, which contain all deferred contributions and contributions from previous time-levels, are denoted as beq.. This fully-coupled system of equations (35) is solved in an iterative fashion using the Block-Jacobi pre-conditioner and the BiCGSTAB solver of the software library PETSc [18, 19], as illustrated in Figure 1. The Newton linearisation of the advection terms in both the momentum equations and the VOF advection equation yields an implicit contribution of the fluxes, which in turn introduces an implicit pressure, velocity and colour function dependency in all governing equations. The implicit formulation of the fluxes by the MWI presented in Eq. (34) is, thus, the primary coupling term of the discretised governing equations. Notably, the implicit pressure and velocity dependency in the VOF advection is a novel building block for solving interfacial flows. Both the colour function gradient and the curvature are treated implicitly with respect to the colour function, yielding an implicit CSF model for surface tension.

4. Differences to previously proposed methods

Previous work aimed at breaching the capillary time-step constraint in the context of interface capturing methods, as already mentioned in the introduction of this article, has focused on incorporating the interface position at the new time instance implicitly in the momentum equations [4, 12] and on a coupled solution algorithm in which all governing equations are solved simultaneously [9]. Hysing [4] proposed a semi-implicit surface tension treatment in the context of a finite-element method, whereby the interface position at the new time instance is incorporate implicitly in the source term representing surface tension. Using the CSF method to model surface tension in a finite-volume discretisation, the source term representing surface tension, accounting for the interface position at the new time instance implicitly, is given as [12]

S(n+1) P ≈σ κ(n) P ∇ψ(n) P + σ ∆t |∇ψ(n) P | ∆su(n+1) P , (36)

where ∆s is the Laplace-Beltrami operator with respect to the interface. This formulation is convenient since the implementation in existing implicit numerical frameworks, by adding an implicit contribution of velocity to the momentum equations, is straightforward. However, a simple dimensional analysis reveals that µΣ = σ ∆t |∇ψ| represents a viscosity acting in the interface region [30] and, in conjunction with a Laplacian of velocity ∆su, the second term on the right-hand side of Eq. (36) acts as an additional viscous stress term at the interface. As discussed by Popinet [1], the resulting increase in dissipation in the vicinity of the interface is responsible for the success of this semi-implicit surface tension formulation. To this end, we demonstrated that an explicit implementation of the additional surface dissipation term in Eq. (36) also allows to breach the capillary time-step constraint by up to one order of magnitude [30], at the cost of artificially increasing the effective viscous stresses acting in the vicinity of the fluid interface. The implicit treatment of surface tension as part of the proposed algorithm, Eq. (18), does not introduce an additional viscous stress term at the interface. We previously presented a coupled implicit algorithm [9], similar to the algorithm proposed in Section 3, which also solves the VOF advection equation together with the continuity and momentum equations in a single system of linear equations, and features a semi-implicit treatment of the source term representing surface tension. Contrary to the proposed algorithm, in the previous algorithm [9] the advection terms of the momentum equations (10) and the VOF advection equation (12) were linearised with a Picard linearisation,

˚

V


![Equation](images/2203.01963_eq155.png)


![Equation](images/2203.01963_eq156.png)


![Equation](images/2203.01963_eq157.png)

where φ ∈{u, ψ}, and the source term representing surface tension only considered the gradient of the colour

8

function in an implicit manner,

S(n+1) P ≈σκ(n) P VP


![Equation](images/2203.01963_eq158.png)


![Equation](images/2203.01963_eq159.png)


![Equation](images/2203.01963_eq160.png)

In contrast, in the algorithm proposed in Section 3, a Newton linearisation is applied to linearise the advection terms and the interface curvature is also treated implicitly with respect to the colour function ψ. In the previous algorithm [9], the advecting velocity was also formulated implicit in the solution variables χ ∈ {p, u, v, w, ψ}, given as

ϑ(n+1) f = u(n+1) f · nf −ˆdf


![Equation](images/2203.01963_eq161.png)

2


![Equation](images/2203.01963_eq162.png)


![Equation](images/2203.01963_eq163.png)


![Equation](images/2203.01963_eq164.png)


![Equation](images/2203.01963_eq165.png)


![Equation](images/2203.01963_eq166.png)

2


![Equation](images/2203.01963_eq167.png)


![Equation](images/2203.01963_eq168.png)


![Equation](images/2203.01963_eq169.png)


![Equation](images/2203.01963_eq170.png)


![Equation](images/2203.01963_eq171.png)

yet, contrary to the formulation proposed in Eq. (34), only the face-based gradients of pressure and colour function were treated implicitly, while the interface curvature and the cell-centred (Gaussian) gradients were deferred. As a result of Eqs. (38) and (39), each governing equation has an implicit contribution of the colour function and the continuity equation has implicit contributions with respect to all solution variables χ ∈{p, u, v, w, ψ}. Nevertheless, the algorithm previously proposed in [9] does not allow to breach the capillary time-step constraint.

5. Results

The ability of the numerical framework proposed in Section 3 to breach the capillary time-step constraint is tested and validated using two well-defined test-cases: the Laplace equilibrium of an interface with constant curvature (i.e. a circular and a spherical interface) and a standing capillary wave. Both test-cases have frequently been used to scrutinise numerical methods, are governed by surface tension, and analytical solutions are available for comparison. Furthermore, as both considered test-cases yield relatively small interface deformations and the interface curvature is spatially well resolved at all times, these test-cases are not subject to issues associated with the fidelity of the interface transport or the well-posedness of the heights used for the evaluation of the interface curvature, issues that are outside the scope of this study.

5.1. Laplace equilibrium A circular or spherical interface subject to surface tension in a quiescent flow is in mechanical equilibrium, with zero velocity in the entire domain and a pressure difference between the outside and the inside of the interface given by the Young-Laplace equation,


![Equation](images/2203.01963_eq172.png)

However, due to errors associated with the numerical framework and the discrete representation of the interface topology, unphysical parasitic currents arise. For a force-balanced numerical framework, such as the one applied in this study [17], these parasitic currents should decay exponentially due to viscous dissipation and converge to a value commensurate with machine precision or the chosen solver tolerance, with the interface attaining a discrete equilibrium shape [5]. Following the work of Popinet [5], a circular interface with a diameter of D = 0.8 m is simulated, with ρ = 1 kg/m3 and σ = 1 N/m. The viscosity µ is defined by the considered Laplace number,


![Equation](images/2203.01963_eq173.png)

µ2 . (41)

Exploiting the symmetry of the problem, only one quarter of the circular interface is simulated, situated at the origin of a square domain with edge length 1 m. The domain is represented by an equidistant Cartesian mesh with 32 × 32 cells. Figure 2 shows the evolution of the root-mean-square (RMS) of the velocity in the domain for La ∈{120, 1200, 12000}, simulated with different time-steps ∆t. A stable solution is obtained

9

0 0.2 0.4 0.6 0.8 1


![Equation](images/2203.01963_eq174.png)


![Equation](images/2203.01963_eq175.png)


![Equation](images/2203.01963_eq176.png)


![Equation](images/2203.01963_eq177.png)


![Equation](images/2203.01963_eq178.png)


![Equation](images/2203.01963_eq179.png)

RMS (u) /Uσ


![Equation](images/2203.01963_eq180.png)


![Equation](images/2203.01963_eq181.png)


![Equation](images/2203.01963_eq182.png)


![Equation](images/2203.01963_eq183.png)


![Equation](images/2203.01963_eq184.png)

La = 120 La = 1200 La = 12000


> **Figure 2: Evolution of the root-mean-square (RMS) of the flow velocity u of the two-dimensional Laplace equilibrium with Laplace number La ∈{120, 1200, 12000}, for different time-steps ∆t, normalised by the capillary velocity Uσ = � σ/(ρD) and the viscous timescale τµ = ρD2/µ. ∆tσ refers to the capillary time-step constraint given in Eq. (3).**


![Equation](images/2203.01963_eq185.png)


![Equation](images/2203.01963_eq186.png)


![Equation](images/2203.01963_eq187.png)


![Equation](images/2203.01963_eq188.png)


![Equation](images/2203.01963_eq189.png)


![Equation](images/2203.01963_eq190.png)


![Equation](images/2203.01963_eq191.png)


![Equation](images/2203.01963_eq192.png)


![Equation](images/2203.01963_eq193.png)


> **Figure 3: Evolution of the root-mean-square (RMS) of the flow velocity u of the three-dimensional Laplace equilibrium with Laplace number La = 120, for different time-steps ∆t, normalised by the capillary velocity Uσ = � σ/(ρD) and the viscous timescale τµ = ρD2/µ. ∆tσ refers to the capillary time-step constraint given in Eq. (3).**

even if the time-step exceeds the capillary time-step constraint ∆tσ, Eq. (3), by factor 50, and the equilibrium velocity, which is of negligible magnitude, is hardly affected. Considering the same case in three-dimensions, i.e. a spherical interface, with La = 120, a similar evolution of the velocity RMS is observed in Figure 3. Hence, the capillary time-step constraint can be breached for both twoand three-dimensional interfaces, without affecting the discrete balance between surface tension and the flow significantly.

5.2. Capillary wave

A single two-dimensional capillary wave is considered to test the fidelity and robustness with which the proposed algorithm can predict surface-tension-driven motion when the capillary time-step constraint is breached. The motion of the fluid interface and the flow are driven solely by surface tension and, considering a small initial wave amplitude and equal properties of the bulk fluids, an analytical solution to the corresponding initial-value problem is available for comparison [31]. The considered capillary wave with wavelength λ = 10−4 m and initial amplitude a0 = 0.01λ is situated in a domain with dimensions λ × 3λ, illustrated in Figure 4, resolved with a mesh spacing of ∆x = λ/100. Periodic boundary conditions are applied on the side walls. Both fluids have the same density, ρ = 1 kg/m3, and the same viscosity, ranging from µ = 5 × 10−6 Pa s to µ = 5 × 10−2 Pa s. The surface tension coefficient is σ = 0.01 N/m. The capillary wave is fully characterised by its critical wavenumber kc ≃22/3σ(ρa + ρb)/(µa + µb)2, above which the oscillation of the capillary wave ceases [32], and its undamped frequency ω0 = � σk3/(ρa + ρb), where k = 2π/λ is the wavenumber.

10


![Equation](images/2203.01963_eq194.png)


![Equation](images/2203.01963_eq195.png)


![Equation](images/2203.01963_eq196.png)


> **Figure 4: Schematic of the two-dimensional capillary wave with wavelength λ and initial amplitude a0.**

0 3 6 9 12 15 −0.5

0

0.5

1

0


![Equation](images/2203.01963_eq197.png)

a/a0


![Equation](images/2203.01963_eq198.png)

0 3 6 9 12 15

0


![Equation](images/2203.01963_eq199.png)


![Equation](images/2203.01963_eq200.png)

0 3 6 9 12 15

0


![Equation](images/2203.01963_eq201.png)


![Equation](images/2203.01963_eq202.png)


> **Figure 5: Evolution of the amplitude of a capillary wave with wavelength λ = 202 λc and initial amplitude a0 = 0.01λ obtained with different time-steps ∆t ∈{0.5, 5, 50}∆tσ, where ∆tσ is the capillary time-step constraint, Eq. (3). The evolution is presented relative to the undamped frequency ω0 of the capillary wave and the analytical solution of Prosperetti [31] is shown as a reference. For the numerical results, each dot shows every 100th time-step for ∆t = 0.5 ∆tσ, every 10th time-step for ∆t = 5 ∆tσ and every time-step for ∆t = 50 ∆tσ.**


> **Figure 5 shows the evolution of the amplitude of an oscillating capillary wave with a wavelength of λ = 202 λc, obtained with different time-steps ∆t. The evolution of the wave amplitude is predicted accurately compared to the analytical solution, even with ∆t = 5 ∆tσ. Although the numerical algorithm is stable for ∆t = 50 ∆tσ, the result exhibits a visible discrepancy in comparison to the analytical solution. Nevertheless, this discrepancy is to be expected given the rather coarse temporal resolution of the oscillation, whereby each time-step is illustrated by a dot in Figure 5 for ∆t = 50 ∆tσ. Similar observations can be made for the evolution of the amplitude of a relatively shorter capillary wave with a wavelength of λ = 12.6 λc shown in Figure 6, where a time-step of ∆t = 100 ∆tσ yields a stable and reasonably accurate result. Changing the mesh resolution with which this capillary wave is resolved, but keeping the applied time-step unchanged, yields virtually identical results, as observed in Figure 7. Figure 8 shows contour plots of the colour function ψ after a single time-step with ∆t = 10∆tσ, using different modelling assumptions. Neglecting any of the three main implicit extensions proposed in comparison to our previously presented coupled algorithm [9], as detailed in Section 4, yields an unphysical interface topology even after a single time-step, if the capillary time-step is breached. Only the proposed algorithm in which all contributions of the solution variables χ ∈{p, u, v, w, ψ} are treated implicitly yields a stable result for time-steps exceeding the capillary time-step constraint.**

6. Revised time-step constraint

Although the proposed algorithm exhibits a favourable behaviour for time-steps exceeding the capillary time-step constraint, the time-step that yields a stable solution is still limited. We presume that this timestep limitation is associated with the particular discretisation and linearisation chosen to cast the governing equations in a form amenable to numerical analysis using linear algebra. Because the governing equations are linearised, we consider linear stability analysis to analyse the stability of the linearised system of governing equations. Based on a linear stability analysis under the assumption of an interface perturbation with small amplitude and sufficiently small Reynolds number, Galusinski and

11

0 3 6 9 12 15 −0.5

0

0.5

1

0


![Equation](images/2203.01963_eq203.png)

a/a0


![Equation](images/2203.01963_eq204.png)

0 3 6 9 12 15

0


![Equation](images/2203.01963_eq205.png)


![Equation](images/2203.01963_eq206.png)

0 3 6 9 12 15

0


![Equation](images/2203.01963_eq207.png)


![Equation](images/2203.01963_eq208.png)


> **Figure 6: Evolution of the amplitude of a capillary wave with wavelength λ = 12.6 λc and initial amplitude a0 = 0.01λ obtained with different time-steps ∆t ∈{1, 10, 100}∆tσ, where ∆tσ is the capillary time-step constraint, Eq. (3). The evolution is presented relative to the undamped frequency ω0 of the capillary wave and the analytical solution of Prosperetti [31] is shown as a reference. For the numerical results, each dot shows every 100th time-step for ∆t = 1 ∆tσ, every 10th time-step for ∆t = 10 ∆tσ and every time-step for ∆t = 100 ∆tσ.**


![Equation](images/2203.01963_eq209.png)

0


![Equation](images/2203.01963_eq210.png)

1

0


![Equation](images/2203.01963_eq211.png)


![Equation](images/2203.01963_eq212.png)


![Equation](images/2203.01963_eq213.png)


> **Figure 7: Evolution of the amplitude of a capillary wave with wavelength λ = 12.6 λc and initial amplitude a0 = 0.01λ. The results are obtained with different mesh resolutions ∆x ∈{λ/100, λ/200, λ/300}. The same time-step, corresponding to ∆t = 10∆tσ for the mesh with ∆x = λ/100, where ∆tσ is the capillary time-step constraint, Eq. (3), is applied regardless of the mesh resolution. The evolution is presented relative to the undamped frequency ω0 of the capillary wave and every 10th**

time-step is shown by a mark.

Vigneaux [33] proposed a maximum time-step ∆t⋆for a stable solution of surface-tension-driven flows of


![Equation](images/2203.01963_eq214.png)


![Equation](images/2203.01963_eq215.png)


![Equation](images/2203.01963_eq216.png)

where c1 and c2 are constants. With the wavelength of the shortest spatially resolved capillary waves being λσ = 2∆x, and assuming3 ˆµ = µa + µb and ˆρ = ρa + ρb, we reformulate Eq. (42) as


![Equation](images/2203.01963_eq217.png)


![Equation](images/2203.01963_eq218.png)

or, by inserting the capillary timescale τσ = � ˆρλ3σ/σ and the viscocapillary timescale τvc = ˆµλσ/σ [34],


![Equation](images/2203.01963_eq219.png)

The maximum time-step ∆t⋆follows as the positive root of Eq. (44),

∆t⋆= a2 τvc + � a2 2 τ 2vc + 4 a1 τ 2σ


![Equation](images/2203.01963_eq220.png)

This suggests that the maximum applicable time-step is proportional to the capillary timescale τσ for small Ohnesorge numbers with respect to λσ,

Oh = τvc

τσ = ˆµ √ˆρσλσ , (46)

where surface tension dominates. In contrast, a maximum time-step proportional to the viscocapillary timescale τvc is relevant for large Oh, where both viscosity and surface tension govern the interface motion.

3Galusinski and Vigneaux [33] did neither specify nor discuss how the density and viscosity of the two-phase system are defined in their linear stability analysis.

12

(a) All implicit (b) Picard linearisation of the advection terms

(c) Explicit interface curvature (d) Semi-implicit MWI formulation


> **Figure 8: Contour plots of the VOF colour function ψ after one time-step of the wave with wavelength λ = 202 λc and initial amplitude a0 = 0.01λ obtained with a time-step of ∆t = 10∆tσ using different modelling assumptions. (a) Shows the result obtained using the proposed all implicit algorithm; (b) a Picard linearisation, see Eq. (37), is applied for the linearisation of the advection terms of the momentum and VOF advection equations; (c) the interface curvature is treated explicitly, with the source term representing surface tension given by Eq. (38); (d) the semi-implicit MWI formulation given in Eq. (39) is applied, but with the proposed implicit treatment of the interface curvature.**


![Equation](images/2203.01963_eq221.png)

100 101 102 103 104 105


![Equation](images/2203.01963_eq222.png)

Oh


![Equation](images/2203.01963_eq223.png)


![Equation](images/2203.01963_eq224.png)

(a) Laplace equilibrium


![Equation](images/2203.01963_eq225.png)

100 101 102 103 104 105


![Equation](images/2203.01963_eq226.png)

Oh


![Equation](images/2203.01963_eq227.png)


![Equation](images/2203.01963_eq228.png)

(b) Capillary wave


> **Figure 9: Maximum applicable time-step ∆t, normalised by the capillary time-step constraint ∆tσ defined in Eq. (3), as a function of the Ohnesorge number Oh defined in Eq. (46), for (a) the Laplace equilibrium of Section 5.1 and (b) the capillary wave (with three different viscosities) of Section 5.2. The approximate maximum time-step ∆t⋆using Eq. (45) is shown with a1 = 6 and a2 = 98 in (a) and with a1 = 1 and a2 = 20 in (b), where suitable values for a1 and a2 are approximated.**

Note that the capillary time-step constraint ∆tσ, as presented in Eq. (3), associated with an explicit treatment of surface tension is recovered for a1 = (16π)−1 and a2 = 0. Figure 9 shows the maximum time-step that yields a stable result for the Laplace equilibrium (Section 5.1) and the capillary wave (Section 5.2), normalised by the capillary time-step constraint ∆tσ given by Eq. (3), for different Ohnesorge numbers as defined in Eq. (46). For both cases, the maximum applicable time-step exceeds the capillary time-step constraint ∆tσ and the revised time-step constraint presented in Eq. (45) is in remarkable agreement with the maximum time-step over the considered eight orders of magnitude of the Ohnesorge number. The results presented in Figure 9 further indicate that the coefficients a1 and a2 are case dependent; a1 ≈6 and a2 ≈98 for the Laplace equilibrium, whereas a1 ≈1 and a2 ≈20 for the capillary wave. More generally, the results suggest a maximum applicable time-step of O(τσ) in the surface-tension-dominated regime (Oh ≪0.01) and of O(10 τvc) −O(100 τvc) in the viscocapillary regime (Oh ≫0.01). We are currently not aware of a method to estimate precise values for a1 and a2 from first principles or based on the discretisation of the governing equations. However, given that the revised time-step constraint ∆t⋆is given by a second-order polynomial, the coefficients can be approximated for a given case for all practically relevant Ohnesorge numbers with only two results, one for Oh ≪0.01 and one for Oh ≫0.01.

13

7. Conclusions

The capillary time-step constraint, first formulated by Brackbill et al. [2], presents a severe impediment to the performance of most interfacial flow simulations with surface tension. Breaching or even eliminating the capillary time-step constraint is generally thought to be possible with an implicit treatment of surface tension. However, previous work in this direction using interface capturing methods has either been unsuccessful [9] or has introduced an additional viscous dissipation term [4, 12, 30]. This led us to conclude that it is not possible to breach the capillary time-step constraint with interface capturing methods [9], a conclusion that was (correctly) met with scepticism [1]. In this study, we have presented a fully-coupled pressure-based algorithm, based on a second-order finitevolume discretisation, featuring an implicit VOF method and an implicit linearised treatment of surface tension. Three implementation principles are at the heart of this algorithm: (i) making all governing equations implicitly dependent on pressure, velocity and the colour function (through the momentum-weighted interpolation), (ii) linearising all nonlinear terms with a Newton linearisation, and (iii) treating every term involving pressure, velocity or the colour function implicitly. The ensuing system of discretised linear equations, which includes the continuity, momentum and VOF advection equations, is then solved simultaneously for pressure, velocity and the colour function. We have shown that this algorithm is able to breach the capillary time-step constraint; hence, interface capturing methods are indeed able to breach the capillary time-step constraint, which proves our early conclusion in [9] to be incorrect. The presented results further indicate that the proposed algorithm features the minimum level of implicitness required for breaching the capillary time-step constraint. However, this study also highlights the limitation of the proposed approach; by how much the capillary time-step constraint can be breached depends on the fluid properties as well as the considered case. To this end, the maximum time-step that yields a stable solution is described accurately by a revised time-step constraint based on a linear stability analysis previously proposed by Galusinski and Vigneaux [33]. As a result, the maximum time-step depends on the surface tension coefficient, density and viscosity, as well as two coefficients that are case dependent. Nevertheless, stable results with time-steps larger than the capillary time-step constraint have been obtained for all simulations considered. With this study, we provide a proof-of-concept for a fully-coupled algorithm and an implicit treatment of surface tension, based on an interface capturing method, that allows to breach the capillary time-step constraint. While the primary aim of breaching the capillary time-step constraint has been achieved, additional work is required to make such an algorithm applicable to solve problems relevant in practice and further exploit its benefits. For instance, in this proof-of-concept we have only considered density and viscosity ratios equal to unity. Making the proposed algorithm applicable to flows with density and viscosity ratios as they occur in practice, while simultaneously allowing to breach the capillary time-step, may additionally require to treat the density and viscosity in a semi-implicit fashion as a function of the colour function. Furthermore, in the employed algebraic VOF method, the advection of the colour function is based on the CICSAM scheme, which is known for requiring very small time-steps to retain a sharp interface [35] and, thus, stands in opposition to maximising the applied time-step. The colour function in the vicinity of the interface is, hence, smeared quickly if the interface moves significantly and, as a consequence, the heightfunction method becomes ill-posed. Contemporary interface advection schemes, such as THINC [36], would perhaps be better suited to maximise the time-step effectively and robustly. Also, the proposed numerical framework is not limited to implicit algebraic VOF methods, but may be based upon an implicit phase-field or level-set method, and it remains to be determined which interface capturing method is best suited to exploit the benefits of the proposed fully-coupled approach. Even a geometric VOF method [e.g. 37] could, in principle, be used in conjunction with the proposed numerical framework, on the condition that it can be implemented implicitly and solved in a coupled system simultaneous with the governing equations.

Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), grant numbers 452916560 and 458610925.


## References

[1] S. Popinet, Numerical models of surface tension, Annual Review of Fluid Mechanics 50 (2018) 49–75. [2] J. Brackbill, D. Kothe, C. Zemach, Continuum Method for Modeling Surface Tension, Journal of Computational Physics 100 (1992) 335–354.

14

[3] D. Kothe, Perspective on Eulerian Finite Volume Methods for Incompressible Interfacial Flows, in: H. Kuhlmann, H. Rath (Eds.), Free Surface Flows, volume M, Springer, Wien, New York, 1998, pp. 267–331. [4] S. Hysing, A new implicit surface tension implementation for interfacial flows, International Journal for Numerical Methods in Fluids 51 (2006) 659–672. [5] S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows, Journal of Computational Physics 228 (2009) 5838–5866. [6] M. Sussman, M. Ohta, A stable and efficient method for treating surface tension in incompressible two-phase flow, SIAM Journal on Scientific Computing 31 (2009) 2447–2471. [7] H. Lamb, Hydrodynamics, sixth ed., Cambridge University Press, 1932. [8] R. Courant, K. Friedrichs, H. Lewy, ¨uber die partiellen Differenzengleichungen der mathematischen Physik, Mathematische Annalen 100 (1928) 32–74. [9] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, Journal of Computational Physics 285 (2015) 24–40. [10] T. Y. Hou, J. S. Lowengrub, M. J. Shelley, Removing the stiffness from interfacial flows with surface tension, Journal of Computational Physics 114 (1994) 312–338. [11] E. B¨ansch, Finite element discretization of the Navier-Stokes equations with a free capillary surface, Numerische Mathematik 88 (2001) 203–235. [12] M. Raessi, M. Bussmann, J. Mostaghimi, A semi-implicit finite volume implementation of the CSF method for treating surface tension in interfacial flows, International Journal for Numerical Methods in Fluids 59 (2009) 1093–1110. [13] W. Zheng, B. Zhu, B. Kim, R. Fedkiw, A new incompressibility discretization for a hybrid particle MAC grid representation with surface tension, Journal of Computational Physics 280 (2015) 96–142. [14] F. Denner, F. Evrard, B. van Wachem, Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds, Journal of Computational Physics 409 (2020) 109348. [15] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–225. [16] F. Denner, Fully-coupled pressure-based algorithm for compressible flows: Linearisation and iterative solution strategies, Computers & Fluids 175 (2018) 53–65. [17] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. [18] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. [19] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. [20] R. Dembo, S. Eisenstat, T. Steihaug, Inexact newton methods, SIAM Journal on Numerical Analysis 19 (1982) 400–408. [21] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177–208. [22] J. H. Ferziger, M. Peric, R. L. Street, Computational Methods for Fluid Dynamics, fourth ed., Springer International Publishing, 2020. [23] O. Ubbink, R. Issa, A Method for Capturing Sharp Fluid Interfaces on Arbitrary Meshes, Journal of Computational Physics 153 (1999) 26–50. [24] S. Cummins, M. Francois, D. Kothe, Estimating curvature from volume fractions, Computers & Structures 83 (2005) 425–434. [25] M. Raessi, J. Mostaghimi, M. Bussmann, Advecting normal vectors: A new method for calculating interface normals and curvatures when modeling two-phase flows, Journal of Computational Physics 226 (2007) 774–797. [26] M. Owkes, O. Desjardins, A mesh-decoupled height function method for computing interface curvature, Journal of Computational Physics 281 (2014) 285–300. [27] F. Evrard, F. Denner, B. van Wachem, Estimation of curvature from volume fractions using parabolic reconstruction on two-dimensional unstructured meshes, Journal of Computational Physics 351 (2017) 271–294. [28] F. Evrard, F. Denner, B. van Wachem, Height-function curvature estimation with arbitrary order on non-uniform Cartesian grids, Journal of Computational Physics: X 7 (2020) 100060. [29] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. [30] F. Denner, F. Evrard, R. Serfaty, B. van Wachem, Artificial viscosity model to mitigate numerical artefacts at fluid interfaces with surface tension, Computers & Fluids 143 (2017) 59–72. [31] A. Prosperetti, Motion of two superposed viscous fluids, Physics of Fluids 24 (1981) 1217–1223. [32] F. Denner, G. Par´e, S. Zaleski, Dispersion and viscous attenuation of capillary waves with finite amplitude, Euro. Phys. J. Spec. Top. 226 (2017) 1229–1238. [33] C. Galusinski, P. Vigneaux, On stability condition for bifluid flows with surface tension: Application to microfluidics, Journal of Computational Physics 227 (2008) 6140–6164. [34] J. Castrej´on-Pita, A. Castrej´on-Pita, S. Thete, K. Sambath, I. Hutchings, J. Hinch, J. Lister, O. Basaran, Plethora of transitions during breakup of liquid filaments., Proc. Nat. Acad. Sci. USA 112 (2015) 4582–4587. [35] V. Gopala, B. van Wachem, Volume of fluid methods for immiscible-fluid and free-surface flows, Chemical Engineering Journal 141 (2008) 204–221. [36] F. Xiao, S. Ii, C. Chen, Revisit to the THINC scheme: A simple algebraic VOF algorithm, Journal of Computational Physics 230 (2011) 7086–7092. [37] D. L. Youngs, Time-dependent multi-material flow with large fluid distortion, in: K. Morton, M. Baines (Eds.), Numerical Methods for Fluid Dynamics, Academic Press, New York, 1982, p. 273.

15

