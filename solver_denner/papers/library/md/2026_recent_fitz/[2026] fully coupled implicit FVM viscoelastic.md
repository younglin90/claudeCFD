
## Fully coupled implicit finite-volume algorithm for viscoelastic interfacial flows


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq001.png)

aChair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany bDepartment of Mechanical Engineering, Polytechnique Montréal, Montréal, H3T 1J4, Québec, Canada

Abstract

A fully coupled implicit finite-volume algorithm for incompressible viscoelastic interfacial flows is proposed, whereby the viscoelasticity of the flow is described by an upper-convected Maxwell constitutive model, including limited extensibility and shear-thinning behaviour. The governing equations describing the conservation of continuity and momentum, as well as the constitutive model are discretized using standard finite-volume methods and are solved for pressure, velocity and the polymer stress tensor in a single linear system of equations. Treating all terms of the linearized and discretized governing equations implicit in velocity, pressure and/or the components of the polymer stress tensor, a tightly coupled system of equations is obtained. The interface separating the interacting bulk phases and the surface tension acting at the fluid interface are modelled using a state-of-the-art front-tracking method. We demonstrate the capabilities of the proposed numerical framework with four representative test cases, including the deformation of a viscoelastic droplet in shear flow at large Weissenberg numbers of up to Wi = 104, and the jump discontinuity of the rise velocity of a bubble rising in a viscoelastic liquid as a result of a “negative wake”. Contrary to previous studies using segregated algorithms, the proposed fully coupled implicit algorithm does not apply or require a log-conformation approach to predict these flows. Overall, the fully implicit coupled front-tracking formulation provides a robust framework to reliable numerical predictions of strongly elastic interfacial flows at large Weissenberg numbers.

Keywords: Viscoelastic flow, Interfacial flows, Fully coupled algorithm, Shear-thinning, High Weissenberg number

1. Introduction

Classical engineering applications of viscoelastic fluids, such as the lubrication dynamics in bearings [1, 2] and binders for composite materials [3], as well as emerging manufacturing techniques by 3D printing [4, 5] and the assembly of soft materials [6, 7] have been driving an increased interest in viscoelastic interfacial flows, in which two or more immiscible fluids interact with each other, and the physical phenomena associated with these flows. The airborne dispersion of respiratory diseases, e.g. corona viruses, gave the interest in viscoelastic interfacial flows a further boost in the wake of the recent pandemic, since mucus, the primary carrier of the viral load, exhibits dominant viscoelastic properties [8]. As a result of these relevant and timely applications of viscoelastic interfacial flows, their numerical modelling has become a subject of growing activity in the scientific community. Even though computational rheology has developed into a mature discipline in recent decades, a large variety of constitutive models for the viscoelastic stress (typically referred to as the polymer stress) based on often competing assumptions [9], large differences in numerical predictions and different interpretations of the underpinning physical phenomena [10, 11], as well as convergence difficulties for flows with even moderate elasticity, known as the high-Weissenberg number problem [12–14], are still hampering a widespread adoption and application of numerical modelling tools for viscoelastic flows. State-of-the-art algorithms for incompressible viscoelastic flows mostly rely on segregated algorithms [11], in which the momentum equations, a pressure projection equation satisfying the continuity constraint and the constitutive model describing the viscoelastic stresses are solved sequentially. However, the weak explicit coupling between velocity, pressure and polymer stress in the discretized governing equations as a result of the iterative predictorcorrector solution procedure severely limits the stability and convergence of these algorithms, requiring a strong underrelaxation of the discretized equations to reach a converged solution [15–17]. Numerical methods to mitigate problems associated with a high elasticity of the fluid, most notably the log-conformation approach [18–21], have laid the foundation for new developments that substantially expanded the parameter range, especially with respect to the Weissenberg number, that can now be simulated routinely with widely available computational tools [16, 22–26], including viscoelastic multiphase flows [27–30]. Coupled implicit algorithms, whereby all governing equations are solved simultaneously in a single linear system of implicitly coupled equations, present a powerful alternative to the widely used segregated algorithms. This class of

∗Corresponding author: Email address: berend.van.wachem@multiflow.org (Berend van Wachem)


# arXiv:2602.08645v1  [physics.flu-dyn]  9 Feb 2026

algorithms, which has been applied successfully to incompressible and compressible Newtonian single-phase [31–35] and interfacial [36–40] flows, allows a tight implicit coupling of the governing equations, improving the stability and performance of the solution algorithm. Fernandes et al. [17] presented a coupled implicit finite-volume algorithm for two-dimensional viscoelastic flows, whereby the constitutive model for the polymer stress tensor is included in the linear system of implicitly coupled equations and the polymer stress tensor is treated implicitly in the momentum equations. Shortly after, Pimenta and Alves [41] proposed a fully coupled algorithm for three-dimensional electrically driven viscoelastic flows using a Poisson-Nernst-Planck model, further adding a Poisson equation governing the electrical potential and transport equations for the charge densities to the linear system of implicitly coupled equations. Recently, Fernandes [26] presented a fully coupled algorithm for two-dimensional viscoelastic flows applying a log-conformation approach to the constitutive model. All three studies [17, 26, 41] reported a robust and rapid convergence with an impressive speed-up exceeding one order of magnitude for some cases compared to a state-of-the-art segregated algorithm for viscoelastic flows. While the simulation of Newtonian interfacial flows, in which two (or more) immiscible fluids interact with each other, has become common and a large variety of numerical frameworks based on the volume-of-fluid [42], level-set [43, 44] and front-tracking [45, 46] methods are available, simulation tools that can accurately predict three-dimensional interfacial flows in which at least one of the interacting fluids is viscoelastic are still scarce. Finite-volume algorithms for viscoelastic interfacial flows using volume-of-fluid [28, 47, 48], level-set [49–51] and front-tracking [30, 52] methods have previously been presented, alongside algorithms based on the Lattice-Boltzmann method [53] and diffusive interface method [54–57]. For incompressible viscoelastic interfacial flows, state-of-the-art algorithms rely on solving for the conformation tensor of the polymer stress rather than for the polymer stress tensor directly, and current finite-volume algorithms are built upon segregated algorithms. However, a fully coupled implicit algorithm for viscoelastic interfacial flows has not yet been presented in the literature. In the context of Newtonian flows, fully coupled implicit algorithms have been demonstrated to be particularly suited for interfacial flows as they can be readily applied to interfacial flows with large density ratios. Additionally, the capillary time-step constraint, a severe impediment for the simulation of flows with surface tension [58, 59], can be breached when surface tension is treated implicitly [60]. Building upon recently published algorithms on viscoelastic single-phase flows [17, 26, 41] and our own prior work on fully coupled implicit algorithms for Newtonian flows [32, 36, 60, 61], we propose a fully coupled implicit finite-volume algorithm for incompressible viscoelastic interfacial flows. To account for the viscoelasticity of the flow, we consider the upper-convected Maxwell constitutive equation for the polymer stress tensor, including limited extensibility and shear-thinning behaviour, demonstrated with the linear and exponential Phan-Thien-Tanner models [62], as well as the Giesekus model [63]. The governing equations describing the conservation of continuity and momentum, as well as the constitutive model are discretized using standard finite-volume methods and are solved for pressure, velocity and the polymer stress tensor in a single linear system of equations. The interface separating the interacting bulk phases and the surface tension acting at the fluid interface are modelled using a state-of-the-art fronttracking method [39, 64]. We demonstrate the capabilities of the proposed numerical framework with representative test cases, including the deformation of a viscoelastic droplet in shear flow at large Weissenberg numbers and the jump discontinuity of the rise velocity of a bubble rising in a viscoelastic liquid. Notably, the proposed algorithm does not apply nor require a log-conformation approach to predict these flows, contrary to the studies presented in the literature to date. Section 2 introduces the governing conservation laws and the considered constitutive model. The discretization of these governing equations and the numerical framework are described in Section 3, where we examine the discretization of each term of the governing equations, and the front-tracking method employed to represent and transport the fluid interface is briefly reviewed in Section 4. A presentation of the complete discretized governing equations and an explanation of the solution procedure is the subject of Section 5. The results of four representative test cases are presented and discussed in Section 6, and the article is concluded in Section 7.

2. Governing equations

The considered incompressible and isothermal viscoelastic flows are governed by the continuity and momentum equations, given as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq002.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq003.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq004.png)

respectively, where t denotes time, u ≡(u v w)T denotes the velocity vector, ρ is the fluid density, Sσ is the source term representing surface tension (when an interface is present) and Sg = ρg is the source term representing gravity, with g the gravitational acceleration. The stress tensor ς = −pI + τ s + τ p comprises the normal stress tensor exerted by pressure, −pI, where I denotes the identity tensor, the solvent stress tensor τ s and the polymer stress tensor τ p,

2

such that the momentum equations follow as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq005.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq006.png)

The solvent stress tesnor is based on Newton’s law of viscosity and given as τ s = 2µD, where µ is the dynamic solvent viscosity, and D = 1

2(∇u + ∇uT) is the rate of deformation tensor. The non-Newtonian viscoelastic behaviour of the flow is expressed by the evolution of the polymer stress tensor τ p, which, in turn, is governed by a general differential constitutive equation of the form [11]


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq007.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq008.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq009.png)

where ψ is the stress function, λ is the relaxation time, ξ is the non-affine or “slip” parameter, α is the mobility parameter and η is the polymer viscosity. The first four terms gathered in big parentheses in Eq. (4) together constitute the upper-convected time derivative of the polymer stress tensor, where the third and fourth terms ensure the correct transformation of τ p under deformations by the flow [65], and the fifth term captures the slip between the molecular network and the continuum medium [66]. Note that the polymer stress tensor is symmetric and, therefore, has only 6 unique components. As constitutive models we consider the Giesekus model [63], for which 0 < α ≤0.5, ψ = 1 and ξ = 0, the linear Phan-Thien-Tanner model (LPTT) [62], for which α = 0 and ψ = 1 + λϵtr(τ p)/η, and the exponential Phan-Thien-Tanner model (EPTT) [62], for which α = 0 and ψ = exp[λϵtr(τ p)/η], where ϵ is the extensibility coefficient of the fluid. All three models are widely used in the literature and convey the primary features of modelling viscoelastic flows. In order to distinguish the interacting fluid phases, we consider a generic indicator function I that is reconstructed based on the position of the interface. The indicator function is defined as

I(x) = ⎧⎪⎪⎨⎪⎪⎩


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq010.png)

where Ω= Ωa ∪Ωb is the computational domain, with Ωa and Ωb the subdomains occupied by fluids “a” and “b”, respectively. Following previous work on viscoelastic interfacial flows [52, 56], the fluid properties ϕ ∈{α,η,λ,µ,ρ,ψ, ξ} are defined based on the indicator function as ϕ(x) = ϕa +I(x)(ϕb −ϕa). Surface tension is modelled as a volumetric source term Sσ in the momentum equations,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq011.png)

where σ is the surface tension coefficient, κ is the interface curvature, nΣ is the normal vector of the interface and δΣ is the interfacial delta function.

3. Numerical framework

The proposed numerical framework is implemented in our in-house finite-volume solver MultiFlow and is built upon a collocated second-order finite-volume discretization and a fully coupled implicit solution algorithm [32, 67], whereby the discretized governing equations are solved in a single system of linearized equations, A ⋅ζ = b, with the three velocity components, pressure and the six unique components of the polymer stress tensor as the implicitly sought solution variables. Below we describe the discretization of the individual terms of the governing conservation laws, Eqs. (1) and (2), and the constitutive model, Eq. (4), as well as the stress-velocity coupling. The interface is modelled using a front-tracking method [39], presented in Section 4, although any other suitable method to capture or track the interface, e.g., volume-of-fluid or level-set methods, may equally be applied together with the proposed numerical framework.

3.1. Discretization methods

The discretization is based on a standard second-order finite-volume method. Discretizing the generic convectiondiffusion equation of a general fluid variable ϕ,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq012.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq013.png)

where Dϕ is the diffusion coefficient of ϕ and S is a generic source term, with the employed second-order finite-volume method is given in semi-discretized form for cell P of an arbitrary computational mesh as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq014.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq015.png)

3

where f denotes all faces bounding mesh cell P, ˜◻denotes a flux-limited interpolation, nf is the normal vector of face f pointing out of cell P, and the area of face f and the volume of cell P are denoted with Af and VP , respectively. The flux Ff through face f is defined as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq016.png)

where the advecting velocity ϑf = uf ⋅nf is obtained by a momentum-weighted interpolation (MWI), as discussed in detail in Section 3.6. The transient derivative is discretized using the second-order backward Euler (BDF2) scheme for a variable time-step as [68]


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq017.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq018.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq019.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq020.png)

where ∆t1 is the current time-step and ∆t2 is the previous time-step. The superscripts (t), (t −∆t1) and (t −∆t1 − ∆t2) denote the solution at the current time-level, the previous time-level, and the previous-previous time-level, respectively. In the discretization of the advection term, ϕf is interpolated from the values at adjacent cell centers using a flux-limited interpolation scheme, given as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq021.png)

where χf denotes the flux limiter, and subscripts D and U denote the downwind cell and the upwind cell of face f, respectively. In this study, we consider the central-differencing scheme as well as the CUBISTA scheme [69], which is widely used to compute viscoelastic flows, to determine the flux limiter χf, but other suitable schemes may equally be applied. The face-centered velocity gradient projected along the normal vector of the cell face, ∇ϕf ⋅nf, in the discretized diffusion term is decomposed into an orthogonal and a non-orthogonal part to correct for any non-orthogonality of the computational mesh, following the work of Demirdžić and Muzaferija [70], as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq022.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq023.png)

where cf = (nf ⋅sf)−1 is the scaling factor of the decomposition [71] and where ◻f = (1 −ℓf) ◻P +ℓf◻Q denotes a linear interpolation, with ℓf the inverse-distance weighting coefficient with respect to cell P and face f. The vector sf is the unit vector connecting the cell centers adjacent to face f, pointing from cell P to neighbour cell Q and ∆sf = ∣sf∣is the distance between the centers of cells P and Q.

3.2. Continuity equation

Applying the divergence theorem, the continuity equation, Eq. (1), is readily discretized using the flux Ff through face f, as defined in Eq. (9), as

∭VP ∇⋅udV ≈∑ f F (n+1) f = 0. (13)

To this end, the implicitly treated advecting velocity given by the MWI, see Section 3.6, of the form

ϑ(n+1) f = u(n+1) f ⋅nf + f(∇p(n+1),u(t−∆t1)) (14)

introduces an implicit dependency of the continuity equation on pressure, such that the continuity equation can be solved implicitly for velocity and pressure in a fully coupled manner [32]. The iteration counter n is associated with nonlinear iterations performed to solve the system of discretized governing equations at each time-step, as further explained in Section 5, with superscript (n) denoting deferred quantities and superscript (n + 1) denoting quantities for which the solution is sought implicitly.

3.3. Momentum equations The momentum equations are solved implicitly in velocity, pressure and the components of the polymer stress tensor. The transient term is discretized, at mesh cell P, as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq024.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq025.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq026.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq027.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq028.png)

with the transient derivative of velocity following from Eq. (10) as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq029.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq030.png)

(n+1)

P = ( 1


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq031.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq032.png)

4

The advection term is discretized by applying a Newton linearization treating both the flow velocity u and the flux F implicitly,

∭VP ρ∇⋅(u ⊗u) dV ≈ρP ∑ f (˜u(n+1) f F (n) f + ˜u(n) f F (n+1) f −˜u(n) f F (n) f ), (17)

where ˜◻denotes the flux-limited interpolation presented in Eq. (11). The divergence of the stress tensor appearing on the right-hand side of the momentum equations, Eq. (2), is split into three parts, ∇⋅ς = −∇p + ∇⋅τ s + ∇⋅τ p, each of which is discretized separately. The pressure gradient, ∇p = ∇⋅(pI), and the divergence of the polymer stress tensor, ∇⋅τ p, are both discretized using linear interpolation of the respective cell values to the mesh faces as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq033.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq034.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq035.png)

Both p and τ p are treated implicitly, as indicated by the superscript (n + 1). The divergence of the solvent stress tensor τ s is discretized as [32]


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq036.png)

where, including the correction for non-orthogonal meshes presented in Eq. (12),


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq037.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq038.png)

A harmonic interpolation is applied to interpolate the viscosity values from the cell centers P and Q to the shared face f [72],

1 ˘µf = 1 −ℓf

µP + ℓf


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq039.png)

where ˘µf is the harmonic interpolated face-centered viscosity, and ℓf is the inverse-distance weighting coefficient with respect to cell P and face f. The solvent stress term is implemented treating all velocity terms implicitly, such that

∭VP ∇⋅τ s dV ≈∑ f µ ⎡⎢⎢⎢⎢⎣ cf u(n+1) Q −u(n+1) P ∆sf + ∇u (n+1) f ⋅(nf −cfsf) + ∇u T,(n+1) f ⋅nf ⎤⎥⎥⎥⎥⎦ Af. (23)

Contrary to previous studies on fully coupled algorithms for Newtonian flows, e.g. [32, 33], and viscoelastic flows [17], all velocity gradients of the discretized solvent shear stress in Eq. (23) are solved implicitly. As shown in previous studies [36, 67, 73], the momentum sources S have to be discretized with the same discretization as the pressure gradients, for the discretized pressure gradient ∇pP to be able to match the discretized source term S⋆ P . Following Bartholomew et al. [67], the discretized momentum source S⋆is constructed based on the untreated source term S as

S⋆ P = 1


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq040.png)

3.4. Constitutive model

The constitutive model, Eq. (4), yields six governing equations for the six unique components of the polymer stress tensor, which are solved implicitly for the components of the polymer stress tensor, as well as velocity and pressure. In the constitutive model, the first term on the left-hand side of Eq. (4) is treated implicitly as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq041.png)

and the transient term of the constitutive model, as part of the upper-convected derivative of the polymer stress tensor, is discretized in the same manner as transient term of the momentum equations,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq042.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq043.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq044.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq045.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq046.png)

5

using the second-order backward Euler scheme presented in Eq. (10). The advection term of the constitutive model, Eq. (4), arises from the material derivative of τ p and is, consequently, not in conserved form, contrary to the advection term of the momentum equations. In the interest of a discretization that is consistent with the advection of momentum, we reformulate the advection term of τ p using the product rule as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq047.png)

such that the advecting velocity can now be applied to define the fluxes through the mesh faces. Similar to the advection term, a Newton linearization is applied, with

∭VP λ(u ⋅∇τ p) dV ≈λP ⎡⎢⎢⎢⎢⎣ ∑ f (˜τ (n+1) p,f −τ (n+1) p,P )F (n) f + ∑ f (˜τ (n) p,f −τ (n) p,P )F (n+1) f −∑ f (˜τ (n) p,f −τ (n) p,P )F (n) f


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq048.png)

For the considered incompressible flows, the (∇⋅u)-term in Eq. (27) is superfluous from a mathematical viewpoint. Numerically, however, this is only true for a converged result, but may not be the case during the initial nonlinear iterations in each time-step. Including the (∇⋅u)-term, therefore, generally improves the convergence of the solution algorithm. Aiming to fully exploit the implicit coupling provided by the fully coupled solution procedure, the two remaining terms of the upper-convected time derivative of the polymer stress tensor and the non-affine response term are linearized and treated implicit using a Newton linearization respectively,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq049.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq050.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq051.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq052.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq053.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq054.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq055.png)

the quadratic stress term of the Giesekus model is treated implicitly with respect to the polymer stress tensor,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq056.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq057.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq058.png)

and the strain-rate tensor is treated implicitly with respect to the velocity,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq059.png)

Fernandes et al. [17] treated both the additional terms of the upper-convective time derivative, Eq. (29), and the strain-rate tensor, Eq. (32), explicitly. Recently, Fernandes [26] also applied a Newton linearization to the additional terms of the upper-convective time derivative in their fully coupled log-conformation algorithm, treating these terms implicitly in the conformation tensor and the velocity. Pimenta and Alves [41] treated the first term of Eq. (32) implicitly with respect to velocity, while treating the second term explicitly. Contrary to the fully coupled algorithm of Fernandes et al. [17], we do not divide the discretized constitutive model by the relaxation time λ before discretization. Hence, the constitutive model is valid for λ ≥0. Considering, for example, the upper-convected Maxwell model (ψ = 1, µ = 0), Eq. (4) reduces to τ p = η(∇u + ∇uT) for λ = 0, resulting in a Newtonian flow with shear viscosity η.

3.5. Stress-velocity coupling

The stress-velocity coupling associated with the polymer stress is a central building block of the proposed methodology. The commonly applied method to ensure a robust coupling between the polymer stress and the velocity when a collocated variable arrangement is used, is to introduce two mathematically equivalent diffusion terms with

6


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq060.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq061.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq062.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq063.png)


> **Figure 1: Interpolation stencils of the velocity at face f, with the adjacent cells P and Q, considered for the stress-velocity coupling.**

opposite signs on the right-hand side of the momentum equations, Eq. (2), to yield


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq064.png)

∂t + ∇⋅(u ⊗u)] = −∇p + ∇⋅τ s + ∇⋅τ p + Sσ + Sg + ∇⋅(ˆη ∇u) ����������������������� small stencil


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq065.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq066.png)

where ˆη is a weighting factor that is dimensionally equivalent to a dynamic viscosity. As the notation indicates, the additional terms are discretized on different computational stencils. For clarity, we first consider an equidistant Cartesian mesh with cf = 1 and ∆sf = ∆x, with the discretization for general non-orthogonal meshes given thereafter in Eqs. (37) and (38). On an equidistant Cartesian mesh, the small-stencil diffusion term is discretized as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq067.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq068.png)

and the large-stencil diffusion term as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq069.png)

as illustrated in Figure 1. In the literature, this procedure of adding two diffusion terms with opposite signs is widely referred to as both-sides diffusion (BSD) [74], typically applied with ˆη = η [16, 17, 26, 75]. Applying the conventionally used centered finite-difference approximations described above, the additional small-stencil and large-stencil diffusion terms yield, using tensor notation and the Einstein summation convention,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq070.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq071.png)

f −1

2 ⎛ ⎝ˆη ∂ui


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq072.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq073.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq074.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq075.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq076.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq077.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq078.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq079.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq080.png)

which infers that, by taking the divergence of this term in the momentum equations, the two additional diffusion terms introduce numerical diffusion (∂u4 i /∂x4 j) with a magnitude proportional to ˆη and ∆x2 [75]. In line with the discretization of the divergence of the solvent stress described in Eq. (23) and with ˆη = ˘ηf, the small-stencil and large-stencil terms are discretized as

∭VP ∇⋅(ˆη ∇u) dV ≈∑ f ˘ηf ⎡⎢⎢⎢⎢⎣ cf u(n+1) Q −u(n+1) P ∆sf + ∇u (n+1) f ⋅(nf −cfsf) ⎤⎥⎥⎥⎥⎦ Af (37)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq081.png)

respectively. The contribution of the stress-velocity coupling to the right-hand side of the discretized momentum equations can, thus, be summarized as

∭VP [∇⋅(ˆη ∇u) ����������������������� small stencil


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq082.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq083.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq084.png)

To fully exploit the implicit coupling afforded by the fully coupled algorithm and to be consistent with the treatment of the strain-rate term in the constitutive model, all velocity contributions in Eq. (39) are treated implicitly.

7

3.6. Pressure-velocity coupling

To ensure a robust pressure-velocity coupling using the employed collocated variable arrangement, the flux Ff = ϑfAf through face f is defined with an advecting velocity ϑf = uf ⋅nf that is evaluated using a momentum-weighted interpolation (MWI) [67], originally introduced by Rhie and Chow [76]. This advecting velocity allows to solve the continuity equation for pressure [32] and prevents pressure-velocity decoupling on the employed collocated variable arrangement [67]. The advecting velocity is, based on the unified formulation of the momentum-weighted interpolation proposed by Bartholomew et al. [67], defined as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq085.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq086.png)

where ˘ρf is the harmonically averaged face density. The weighting factor ˆdf defines the strength of the pressurevelocity coupling and is given as

ˆdf =

VP aP + VQ


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq087.png)

2 + ˘ρf

∆t1 (VP


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq088.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq089.png)

The coefficients aP and aQ are defined based on the diagonal matrix coefficients of the velocity arising from the advection term, see Eq. (17), the solvent stress term, see Eq. (23), and the small-stencil stress-coupling term, see Eq. (39), of the discretized momentum equations associated with the cells adjacent to face f. For the discretization presented above, the coefficient aP (and, analogously, aQ) is given as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq090.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq091.png)

where χ′ f = (1 −χf) if Ff ≥0 and χ′ f = χf if Ff < 0. For the MWI to be time-step independent, aP and aQ must not include the contribution of the transient terms to the diagonal coefficient [67]. For an arbitrary unstructured mesh and including a density-weighting of the large-stencil pressure and source term contributions, the discretized and implicitly treated advecting velocity is defined as [67]

ϑ(n+1) f = u(n+1) f ⋅nf −ˆdf ⎡⎢⎢⎢⎢⎣


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq092.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq093.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq094.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq095.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq096.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq097.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq098.png)

The discretized pressure terms together constitute a low-pass filter on the pressure field that prevents pressure-velocity decoupling [67]. Contrary to most previous work on fully coupled algorithms [17, 32, 33], we treat all pressure terms in Eq. (43) implicitly [61].

4. Front-tracking method

The numerical framework proposed in the previous section is complemented by a front-tracking method [39] to track the fluid interface separating to immiscible bulk phases. Since the precise formulation and implementation of the applied interface tracking (or interface capturing) method is not critical for the proposed numerical framework, we only provide a brief overview of the applied front-tracking method and refer the reader to our recent publications [39, 64] for more details. In front tracking [45, 46], the fluid interface is represented by a triangulated surface mesh. Each vertex i of this surface mesh is advected in a Lagrangian manner,

dxi(t)

dt = u(xi,t), (44)

where xi and u are the location and (interpolated) velocity of vertex i, respectively. The vertices of the surface mesh can, consequently, also move tangential to the interface, which may lead to vertex clustering and a deteriorating quality of the surface mesh, in turn requiring extensive remeshing of the surface mesh to retain an acceptable mesh quality. In order to address this issue, we apply the normal-only advection (NOA) of the vertices [64], with the

8

velocity at the location of the surface mesh vertices defined as

u(xi,t) = uref(t) + {[u(xi,t) −uref(t)] ⋅n(xi,t)}n(xi,t) (45)

where uref(t) is a spatially invariant reference velocity and u(xi,t) is the interpolated fluid velocity at the location of vertex i. Since the fluid velocity is only known at the cell centers of the fluid mesh, the velocity u(xi,t) at the location of the vertices of the surface mesh is interpolated from the fluid mesh using a Peskin cosine interpolation kernel [77],

u(xi,t) = ∑ L


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq099.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq100.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq101.png)

where L denotes all mesh cells in a 2∆x × 2∆x × 2∆x stencil with respect to vertex i, and the weighting kernel is

d(r) =

⎧⎪⎪⎪⎨⎪⎪⎪⎩

1 4 [1 + cos(π


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq102.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq103.png)

We define the reference velocity as the volume-averaged velocity of the body enclosed by the front,

uref ≈∑P uP IP VP


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq104.png)

where P denotes all cells of the fluid mesh, and integrate Eq. (44) using a conventional fourth-order Runge-Kutta scheme [64]. The indicator function I is reconstructed based on the location of the surface mesh by solving a Poisson equation [78]. The force due to surface tension is computed at each triangle T of the surface mesh using a Frenet-Element algorithm [78]


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq105.png)

where e denotes the edges of triangle T with length le, outward-pointing planar vector pe = ne ×te, normal vector ne and tangential vector te. Subsequently, the force due to surface tension computed on the surface mesh is interpolated to the fluid mesh using the Peskin cosine interpolation kernel to define the surface tension source term as

Sσ,P = ∑ T

⎡⎢⎢⎢⎣ Fσ,T


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq106.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq107.png)

where T are all surface triangles in a 2∆x × 2∆x × 2∆x stencil with respect to cell P. The surface mesh is dynamically adapted to ensure a sufficient mesh quality as well as an adequate resolution of the interface, including a parabolic fit vertex repositioning method that reduces shape errors of the interface, as described in detail in our previous work [39]. In addition, we apply a volume correction step [79] to improve volume conservation and treat small undulations of the surface mesh in areas where the interface strongly contracts with the TSUR3D algorithm [80].

5. Solution procedure

Combining the discretization of the individual terms presented in Section 3, we obtain a set of discretized equations governing the considered incompressible and isothermal viscoelastic flows. The discretized continuity equation is given by Eq. (13), the discretized momentum equations are

ρP ⎡⎢⎢⎢⎢⎣


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq108.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq109.png)

(n+1)

P VP + ∑ f (˜u(n+1) f F (n) f + ˜u(n) f F (n+1) f −˜u(n) f F (n) f ) ⎤⎥⎥⎥⎥⎦ = −∑ f


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq110.png)

+ ∑ f ˘µf ⎡⎢⎢⎢⎢⎣ cf u(n+1) Q −u(n+1) P ∆sf + ∇u (n+1) f ⋅(nf −cf sf) + ∇u T,(n+1) f ⋅nf ⎤⎥⎥⎥⎥⎦ Af + ∑ f (τ (n+1) p,f ⋅nf)Af


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq111.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq112.png)

and the discretized constitutive model is given as

9


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq113.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq114.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq115.png)

Advect the interface and compute the surface tension

Gather the coefficients and assemble A and b


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq116.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq117.png)

Conservation satisfied?

Adapt the fluid mesh, if applicable

no

yes


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq118.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq119.png)


> **Figure 2: Flow chart of the solution procedure of the discretized and linearized system of governing equations, where n is the nonlinear iteration counter, Γ = {u, v, w, p, τp,xx, τp,yy, τp,zz, τp,xy, τp,xz, τp,yz} are the solution variables and Ff is the flux through mesh face f (see Section 3.6). The coefficient matrix A holds all coefficients for the implicitly sought solution variables Γ(n+1) of the discretized governing equations and ζ is the solution vector. The right-hand side vector b holds the deferred contributions of the previous iteration (Γ(n), ϑ(n) f ) and the contributions of the previous time-levels (Γ(t−∆t1), Γ(t−∆t1−∆t2), ϑ(t−∆t1) f ).**

ψP τ (n+1) p,P VP + λP ⎡⎢⎢⎢⎢⎣


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq120.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq121.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq122.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq123.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq124.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq125.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq126.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq127.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq128.png)

As the notation suggests, each term of the governing equations makes an implicit contribution to at least one of the solution variables Γ = {p,u,v,w,τp,xx,τp,yy,τp,zz,τp,xy,τp,xz,τp,yz}. The solution procedure applied to solve the discretized governing equations is illustrated in Figure 2. In each time-step, the interface is advected first and, subsequently, the linearized and discretized governing equations (13), (51) and (52) are solved simultaneously in a single linear system of equations. For a three-dimensional computational

10

mesh with N cells, this linear system of equations is given as

⎛ ⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜⎜ ⎝

Ap Au Av Aw 0 0 0 0 0 0 Bp Bu Bv Bw Bτp,xx 0 0 Bτp,xy Bτp,xz 0 Cp Cu Cv Cw 0 Cτp,yy 0 Cτp,xy 0 Cτp,yz Dp Du Dv Dw 0 0 Dτp,zz 0 Dτp,xz Dτp,yz Ep Eu Ev Ew Eτp,xx 0 0 Eτp,xy Eτp,xz 0 Fp Fu Fv Fw 0 Fτp,yy 0 Fτp,xy 0 Fτp,yz Gp Gu Gv Gw 0 0 Gτp,zz 0 Gτp,xz Gτp,yz Hp Hu Hv Hw Hτp,xx Hτp,yy 0 Hτp,xy Hτp,xz Hτp,yz Ip Iu Iv Iw Iτp,xx 0 Iτp,zz Iτp,xy Iτp,xz Iτp,yz J p J u J v J w 0 J τp,yy J τp,zz J τp,xy J τp,xz J τp,yz


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq129.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq130.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq131.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq132.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq133.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq134.png)

where AΓ to J χ are the N × N coefficient submatrices of the solution variables Γ associated with the continuity equation (A), the three momentum equations (B−D) and the six constitutive equations of the polymer stress tensor (E −J ). The subvectors ζΓ of length N hold the solution of the implicitly sought variables Γ and the right-hand side vector b of length 10N holds all known contribution from previous nonlinear iterations and time-steps. The solution procedure performs nonlinear iterations in which this system of linearized and discretized governing equations, Eq. (53), is solved using the Block-Jacobi pre-conditioner and the BiCGSTAB solver of the software library PETSc [81, 82] until a pre-defined solver tolerance is satisfied. Subsequently, the deferred quantities are updated and Eq. (53) is solved again. This procedure continues until the conservation error of the nonlinear set of governing conservation laws satisfies a predefined maximum error [32], at which point the solution procedure moves to the next time-step. The Newton linearization of the advection terms in the momentum equations and the constitutive model yields an implicit contribution of the fluxes F (n+1) f . The flux, thus, introduces an implicit pressure and velocity dependency in all governing equations. Furthermore, the implicit treatment of the polymer stress term and the stress-velocity coupling terms in the momentum equations, alongside the implicit treatment of the upper-convected time derivative of the polymer stress tensor and strain-rate tensor in the constitutive model, provides a strong implicit coupling of the velocity field and the polymer stress tensor.

6. Results

Four representative test cases are considered to demonstrate the capabilities of the proposed numerical framework for single-phase and interfacial flows. First, a lid-driven cavity containing a viscoelastic fluid described by the LPTT model is considered in Section 6.1 to assess the basic predictive accuracy of the proposed algorithm. Two-dimensional Taylor vortices are simulated in Section 6.2 to quantify the influence of the stress-velocity coupling on the conservation of kinetic energy. In Section 6.3, a Newtonian droplet in a shear-thinning Giesekus fluid is subjected to a shear flow at different Weissenberg numbers, allowing a direct comparison with the results recently reported by Wang et al. [53] using a state-of-the-art Lattice-Boltzmann method. A bubble rising in a viscoelastic EPTT fluid under the action of gravity is considered in Section 6.4, where we focus particularly on the jump discontinuity in the terminal rise velocity of the bubble and the related negative-wake phenomenon, as studied in detail by Niethammer et al. [28].

6.1. Lid-driven cavity A square cavity with edge length L is considered, the top wall of which moves at a constant velocity U, with the shear rate defined as ˙γ = U/L. Following Yapici [83], we consider an LPTT fluid with β = µ/(µ+η) = 0.3, ϵ = 0.25 and ξ = 0. Different mesh resolutions ranging from 20 × 20 to 160 × 160 cells are considered and the applied time-step ∆t is defined adaptively to correspond to a Courant number of Co = u∆t/∆x ≃0.9. The flow has a Weissenberg number of Wi = ˙γλ ∈{1,5} and a Reynolds number of Re = ρ˙γL2/(µ + η) = 10−4. The contours of the velocity magnitude at steady state of both cases, with U = 1 m/s, are shown in Figure 3. Figures 4 and 5 show the velocity profiles along both centerlines, as well as the error ετp,xx in normal polymer stress component τp,xx at x = (0.9L 0.9L), for Wi = 1 and Wi = 5, respectively. For both considered Weissenberg numbers, the velocity profiles are in excellent agreement with the results reported by Yapici [83] for the same cases and using the same mesh resolution. The error ετp,xx in normal polymer stress component τp,xx converges, as expected, with second order compared to the solution on the finest mesh, if the mesh resolution is sufficiently high.

6.2. Taylor vortices The evolution of two-dimensional Taylor vortices are simulated to analyze the artificial dissipation of kinetic energy contributed by the stress-velocity coupling. With this test case we were able to demonstrate that the fully coupled algorithm for Newtonian flows that is underpinning the proposed algorithm for viscoelastic flows does not introduce numerical diffusion if central differencing is applied [32], aside from the numerical diffusion associated with the MWI used for the definition of the fluxes, which, however, decays with ∆x3.

11


> **Figure 3: Velocity contours of the considered LPTT fluid in a lid-driven cavity at steady state, for Wi ∈{1, 5}, on an equidistant Cartesian mesh with 160 × 160 cells.**

0.2 0.0 0.2 0.4 0.6 0.8 1.0

u

0.0

0.2

0.4

0.6

0.8

1.0

y


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq135.png)

(a) u-velocity

0.0 0.2 0.4 0.6 0.8 1.0 x

0.2

0.1

0.0

0.1

0.2

v


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq136.png)

10 3 10 2 10 1

x xref

10 3

10 2

10 1

10 0

10 1

p,xx

x2

(c) Mesh convergence


> **Figure 4: Velocity profiles along the respective centerlines of the lid-driven cavity, obtained on the reference mesh with 160 × 160 cells, and convergence of the error ετp,xx in normal polymer stress component τp,xx relative to the reference mesh, for Wi = 1. The results of Yapici [83] are shown for reference.**

0.2 0.0 0.2 0.4 0.6 0.8 1.0

u

0.0

0.2

0.4

0.6

0.8

1.0

y


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq137.png)

(a) u-velocity

0.0 0.2 0.4 0.6 0.8 1.0 x

0.2

0.1

0.0

0.1

0.2

v


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq138.png)

10 3 10 2 10 1

x xref

10 3

10 2

10 1

10 0

10 1

p,xx

x2

(c) Mesh convergence


> **Figure 5: Velocity profiles along the respective centerlines of the lid-driven cavity, obtained on the reference mesh with 160 × 160 cells, and covergence of the error ετp,xx in normal polymer stress component τp,xx relative to the reference mesh, for Wi = 5. The results of Yapici [83] are shown for reference.**

12

0.162

0.166

0.170

Ekin [J]

analytic

=1, =0.01Pas

s, central

s, CUBISTA

p, central

0.154

0.162

0.170

=10, =0.1Pas

0.06

0.12

0.18

=100, =1Pas

10 2 10 1 x [m]

10 8

10 6

10 4

10 2

10 0

kin

2

3

10 2 10 1 x [m]

10 8

10 6

10 4

10 2

10 0

2

10 2 10 1 x [m]

10 8

10 6

10 4

10 2

10 0

2


> **Figure 6: Results of the Newtonian Taylor vortices at t = 1s for Re = 100, applying τ = τ s, τ = τ p with the stress-velocity coupling in the momentum equations defined by Eq. (58), for different parameter sets and λ = 0 For τ = τ s results obtained with both the central differencing scheme and the CUBISTA scheme are shown, for τ = τ p only results using the central differencing scheme are shown. Top row: Kinetic energy Ekin integrated over the domain as a function of mesh spacing ∆x, where the analytic kinetic energy is shown by the dashed line. Bottom row: Error εkin, see Eq. (60), as a function of mesh spacing ∆x, incurred when using τ = τ p compared to τ = τ s, for different parameter sets.**

Following the work of Ham and Iaccarino [84] as well as our previous work [32, 67], the computational domain has the dimensions 2m × 2m and is periodic in all directions, such that no boundary conditions need to be considered. For a Newtonian fluid, the velocity and pressure are given as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq139.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq140.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq141.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq142.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq143.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq144.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq145.png)

from which the initial conditions for the simulations are readily obtained for t = 0. Integrating the kinetic energy analytically and numerically over the domain Ωyields for a Newtonian fluid with constant density

Ekin(t) = 1


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq146.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq147.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq148.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq149.png)

The fluid occupying the computational domain has a density of ρ = 1kg/m3 and the time-step applied for all simulations is ∆t = 2 × 103 s. If not stated otherwise, the central differencing scheme is applied for the discretization of the advection terms. We consider a Newtonian fluid, such that the momentum equations are given as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq150.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq151.png)

However, the stress tensor τ is now either the solvent stress tensor τ s or the polymer stress tensor τ p under the assumption of λ = 0, α = 0, and ξ = 0, for which the constitutive model reduces to


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq152.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq153.png)

In both scenarios, assuming either τ = τ s or τ = τ p in the momentum equations, the results should be identical as long as µ = η/ψ. The only difference between the two scenarios is, therefore, the coupling of the polymer stress with the velocity field described in Section 3.5, which is not required for the solvent stress. Please note, the constitutive model is solved for the polymer stress to demonstrate the influence of the stress-velocity coupling. As the weighting

13


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq154.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq155.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq156.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq157.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq158.png)

L


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq159.png)


> **Figure 7: Schematic illustration of the droplet in shear flow, where the blue color depicts the viscoelastic fluid.**

coefficient for the stress-velocity coupled we use ˆη = η, following Fernandes et al. [17], as conventionally used in the literature. Figure 6 shows the kinetic energy integrated over the domain at t = 1s for Re = 100. Applying ψ = 1 for the polymer stress tensor, all cases are in excellent agreement with each other, converging towards the analytical value given by Eq. (57). This is to be expected for using the polymer stress in conjunction with the stress-velocity coupling and ˆη = η, because in this case the stress-velocity coupling substitutes, in the momentum equations, the large-stencil diffusion term of the strain-rate tensor of the constitutive model by the corresponding small-stencil diffusion term. The errors incurred when using the polymer stress instead of the solvent stress, defined as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq160.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq161.png)

where Ekin,s is the kinetic energy obtained using the solvent stress and Ekin,p is the kinetic energy obtained using the polymer stress, are numerically negligible. Applying the CUBISTA scheme [69] instead of central differencing to discretise the advection term of the momentum equations also introduces a small amount of numerical diffusion, an error that converges with close to third order under mesh refinement. Changing the values of the stress function ψ and the polymer viscosity η concurrently, such that the ratio η/ψ remains unchanged, should yield the same results. However, the stress-velocity coupling imposes a filter on the velocity field that is proportional to ˆη and ∆x2, see Eq. (36). Figure 6 shows the kinetic energy integrated over the domain at t = 1 for Re = 100, where {ψ = 10,η = 0.1Pas} and {ψ = 100,η = 1Pas} for the viscoelastic cases. The polymer stress tensor introduces an error that is, as expected, dependent on the values of η and the mesh spacing ∆x. The difference between the results obtained using the polymer stress and using the solvent stress decays proportional to ∆x2. These results demonstrate that the stress-velocity coupling imposes a filter on velocity field and introduces an error that is proportional to ∆x2, as stipulated by Eq. (36). Hence, the employed stress-velocity coupling retains the second-order accuracy of the underlying finite-volume scheme as part of the proposed fully coupled algorithm.

6.3. Droplet in shear flow The capabilities of the proposed fully coupled algorithm with respect to interfacial flows with surface tension is demonstrated with a Newtonian droplet situated in a Giesekus fluid between two infinite parallel plates, subject to a shear flow with shear rate ˙γ = 2U/H, as illustrated in Figure 7. Following the recent work of Wang et al. [53], the initially spherical droplet with radius R is placed at the center of the computational domain with dimensions 9R × 5.5R × 4R, represented with an equidistant Cartesian mesh. The plates are modelled as no-slip walls, whereas periodicity is assigned to all other domain boundaries. The host fluid is characterized by a solvent viscosity ratio of β = µh/(µh + ηh) = 0.5, a non-affine parameter of ξ = 0, and a mobility parameter of α = 0.3, the droplet viscosity ratio is m = µd/(µh + ηh) = 1 and the surface tension coefficient σ of the fluid interface follows from the considered capillary number Ca = (µh + ηh)˙γR/σ ∈{0.15,0.25}. The shear flow is in the creeping flow regime, with a Reynolds number of Re = ρh ˙γR2/(µh + ηh) = 0.1. In order to test the mesh convergence of the proposed numerical framework, we consider a droplet with Ca = 0.15 in viscoelastic shear flow with Wi = ˙γλ = 1, using different mesh resolutions. Figure 8a shows the evolution of the Taylor deformation parameter,


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq162.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq163.png)

14

0 1 2 3 4 5 t

0.00

0.06

0.12

0.18

D

x=R/10 x=R/15 x=R/20 x=R/25 Wang et al., x=R/30


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq164.png)


### 0 1 2 3 4 5 6 7 t


### 0.0


### 0.1


### 0.2


### 0.3


### D


### Wi=1 Wi=4 Wi=50 Wi=104


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq165.png)


> **Figure 8: Evolution of the Taylor deformation parameter D, see Eq. (61), of the droplet in viscoelastic shear flow. (a) Ca = 0.15 and Wi = 1, obtained with different mesh resolutions; the circles show the reference results of Wang et al. [53]. (b) Ca = 0.25 using a mesh resolution of ∆x = R/20, for different Weissenberg numbers Wi = ˙γλ; the colored circles show the corresponding reference results of Wang et al. [53].**


> **Table 1: Material parameters of P2500 0.8% weight aqueous viscoelastic liquid.**


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq166.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq167.png)

where L and S are the semi-major and semi-minor axes of the deformed droplet, as illustrated in Figure 7. Even for a mesh resolution as small as 10 cells per initial bubble radius, the proposed numerical framework provides results of reasonable accuracy. The results converge as the mesh resolution is increased and exhibit an overall very good agreement with the reference results of Wang et al. [53]. The evolution of the Taylor deformation parameter D, see Eq. (61), for a droplet with Ca = 0.25 and different Weissenberg numbers Wi, using a mesh resolution of ∆x = R/20, is shown in Figure 8b. The results obtained with the proposed numerical framework are in excellent agreement with the results of Wang et al. [53] up to Wi = 50, which is the largest Weissenberg number considered by Wang et al. [53]. Even for a strongly elastic case, with Wi = 104, the proposed numerical framework is seen to produce physically meaningful results, converging robustly without requiring any form of underrelaxation or log-transformation.

6.4. Rising bubble To further demonstrate the capabilities of the proposed fully coupled algorithm with respect to interfacial flows, a benchmark case of a single gas bubble rising in a viscoelastic liquid is considered. It is a compelling validation case, since experimental studies show a jump in the terminal rise velocity beyond a critical bubble volume and the formation of a negative wake region behind the trailing end of the bubble. The main objective is to achieve quantitative agreement between numerical predictions and experimental measurements and to analyze the flow structures surrounding the bubble to clarify the interplay between rise velocity, bubble deformation, and viscoelastic stress.

6.4.1. Case setup and overview The experimental results of a rising bubble in an aqueous P2500 0.8% weight viscoelastic liquid of Pilz and Brenn [85] are used to validate the proposed fully coupled algorithm. As illustrated in Figure 9, the polymer solution is shear thinning, hence the exponential Phan-Thien Tanner (EPTT) model is used and the model parameters are determined by fitting the viscosity material function of the used model in steady shear flow to experimental rheology data. For the complete derivation of the viscosity material function the reader is referred to Alves et al. [66]. Liquid density, polymer viscosity, solvent viscosity, surface tension coefficient, and relaxation time are given by Pilz and Brenn [85], while the extensibility coefficient and the slip parameter are determined from the fitting procedure as illustrated in Figure 9. The resultant model parameters are shown in Table 1 following the work of Niethammer et al. [28]. The initially spherical bubble with diameter D is placed at the center of a cubic domain of size 20D × 20D × 20D to eliminate the effects of confinement from the domain boundaries, discretized using an equidistant Cartesian mesh. The bubble is rising in positive y-axis due to buoyancy. At the domain boundaries, the velocity field is prescribed using a combination of Dirichlet and Neumann conditions. On the x and z boundaries, impermeable free-slip boundaries are applied: the normal velocity component is set to zero (Dirichlet), while homogeneous Neumann conditions are imposed for the tangential components (zero normal gradients). At the lower y boundary, a no-slip wall is enforced

15

10 2 10 1 10 0 10 1 10 2 10 3

[1/s]

10 1

10 0

p [Pa s]

p =1.483 Pa s,  s =0.03 Pa s

=0.203 s,  =0.05,  =0.12


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq168.png)


> **Figure 9: Polymer viscosity as a function of shear rate of the aqueous 0.8% weight P2500 solution of Pilz and Brenn [85] with fitted data for the EPTT model.**

by prescribing all velocity components to zero (Dirichlet). At the upper y boundary, an outlet condition is used by imposing homogeneous Neumann conditions for all velocity components (zero normal gradients), while the pressure is fixed to a reference value (Dirichlet) to provide a well-defined pressure level. The pressure satisfies homogeneous Neumann conditions on all remaining boundaries. Finally, homogeneous Neumann conditions are applied to the polymer-stress components on all domain faces, enforcing zero normal gradients at the boundaries. Unlike the droplet in shear flow, where a linear interpolation of the stress function, polymer viscosity, relaxation time and non-affine parameter is applied at the interface, we employ here a smooth but sharpened nonlinear interpolation of the same polymer properties for the rising bubble simulations. The rise-velocity jump and the development of a negative wake are governed by the localization and intensity of viscoelastic stresses in a thin layer around the bubble. In particular, detailed analyses relate the regime change to polymer stretching along the bubble contour and to where the stored elastic energy is released relative to the bubble equator [86]. The present nonlinear interpolation confines intermediate properties to a narrow interfacial band while remaining continuous, thereby better preserving the stress distribution near the interface and improving agreement with the reference results. The polymer properties for the rising bubble simulations are defined as


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq169.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq170.png)

Ic )


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq171.png)

where ω is a weighting factor that sets the location of the transition in terms of the indicator field, Ic ∈(0,1) denotes the indicator value used to define the effective interface within the numerically smeared interfacial region, and n > 0 is a sharpness exponent controlling the steepness of the transition (larger n yields a more abrupt switch between phases, while smaller n produces a smoother variation). In the present work, we use Ic = 0.47 and n = 20 for all simulated volumes, as these values provided the sharpest transition and the best agreement with our reference results. A mesh and time-step convergence study was performed to assess the sensitivity of the numerical results to spatial resolution and temporal discretization. The purpose of this study is to determine an efficient spatial and temporal resolution (mesh cell size and time-step) that still provides reliable predictions of the terminal rise velocity. The bubble rise velocity is defined as the vertical velocity component of the bubble centroid, computed as the indicator weighted volume average of the velocity field. Numerical results for a bubble volume of 30mm3 are compared across three mesh resolutions and two time-step sizes to assess grid and time-step convergence, as illustrated in Figures 10a and 10b, respectively. Adaptive mesh refinement is applied in the vicinity of the bubble, providing high resolution around the interface while allowing for a coarser mesh in regions farther from the bubble. Figures 10a and 10b show that a resolution of 60 mesh cells per bubble diameter and a time-step of 1×10−4 s are sufficient, as further refinement or a smaller time-step yield negligible changes in the results. The time step is selected such that it always satisfies the CFL stability constraint of 0.2.

6.4.2. Results Building on the numerical settings and material properties established in the previous section, we report the transient rise velocity evolution and the terminal rise velocity as a function of bubble volume, and discuss the

16

0.00 0.05 0.10 0.15 t [s]

0

20

40

60

80

v [mm/s]

40 cells/D 60 cells/D 90 cells/D

(a) Grid convergence study

0.00 0.05 0.10 0.15 t [s]

0

20

40

60

80

v [mm/s]


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq172.png)

(b) Time step convergence study


> **Figure 10: Transient rise velocity of a 30 mm3 bubble as a function of time. (a) Grid-convergence study using three meshes with different numbers of mesh cells per bubble diameter at ∆t = 1 × 10−4 s. The spatial resolution is increased by a factor of 1.5 from one mesh level to the next.(b) Time-step convergence study at fixed spatial resolution of 60 mesh cells per bubble diameter.**

0.00 0.05 0.10 0.15 0.20 t [s]

0

50

100

150

200

v [mm/s]

V=8 mm3

V=30 mm3

V=40 mm3

V=60 mm3

V=70 mm3

(a) Transient rise velocity

10 1 10 2

V [mm3]

10 1

10 2

v [mm/s]

Pilz & Brenn (2007) Current work

(b) Terminal rise velocity


> **Figure 11: Transient and terminal rise velocity of different bubble volumes as a function of time. (a) Transient rise velocity of five bubble volumes 8, 30, 40, 60, and 70 mm3. (b) Terminal rise velocity of five bubble volumes 8, 30, 40, 60, and 70 mm3 compared to the experiments of Pilz and Brenn [85].**

associated negative wake appearance over a region downstream the trailing end of the supercritical bubbles. Figure 11a shows the transient rise velocity of four bubble volumes as a function of time, including three subcritical bubble volumes, V < Vc (8, 30, and 40 mm3) and two supercritical bubble volumes, V > Vc (60 and 70 mm3). Based on their experimental measurements, Pilz and Brenn [85] reported a critical volume of Vc ≈46 mm3 for the considered fluid combination. In the subcritical regime, the bubbles accelerate rapidly from rest and reach a single local maximum at early times, followed by a gradual deceleration toward a steady terminal value. The approach to the terminal state is monotonic after the first maximum and no secondary acceleration stage is observed. In contrast, the supercritical cases exhibit a distinctly different two-stage transient response. The initial acceleration and first local maximum are similar to the subcritical cases, however the subsequent deceleration continues past the terminal value and leads to a local minimum. After this minimum, the bubble undergoes a second acceleration stage and eventually reaches a terminal velocity that is higher than the first local maximum in the considered cases. This non-monotonic evolution is the characteristic signature of the supercritical regime. Figure 11b reports the terminal rise velocity as a function of bubble volume, comparing the present simulation results with the measurements of Pilz and Brenn [85]. The results capture the jump discontinuity in terminal velocity and show good overall agreement with the experimental data. In the supercritical regime, a negative wake develops behind the bubble. In the inertial frame, the vertical liquid velocity in the bubble’s wake reverses direction over a region downstream of the trailing end. Figure 12 shows the wake using vertical velocity contours to identify the onset and spatial growth of the negative wake of the 70mm3

bubble at two time instances, t = 0.05s where the onset of the negative wake and at t = 0.07s where the negative

17


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq173.png)


![Equation](images/[2026] fully coupled implicit FVM viscoelastic_eq174.png)


> **Figure 12: Vertical velocity component v (left) and velocity vector field (right) for the bubble with a volume of 70 mm3 at two representative times. At t = 0.05 s the onset of reverse flow appears downstream of the trailing end, while at t = 0.07 s a clearer reversed-flow region is established. Vector glyphs are scaled for visualization of direction only.**

wake is more developed and apparent. In agreement with prior numerical studies, the negative wake develops as a dynamical wake structure that grows downstream of the trailing tip as the bubble approaches its supercritical steady state.

7. Conclusions

In this work, we present a fully coupled implicit finite-volume framework for incompressible viscoelastic interfacial flows, in which the continuity equation, momentum equations, and an upper-convected Maxwell constitutive model, including limited extensibility and shear-thinning behaviour, are solved simultaneously for pressure, velocity, and the polymer stress tensor. The proposed discretization treats all relevant couplings implicitly, including the stress-velocity coupling and the pressure-velocity coupling, yielding a tightly coupled linear system at each nonlinear iteration in a standard finite-volume framework. In addition, a state-of-the-art front-tracking approach is used to represent the evolving interface and account for surface tension forces. The method is assessed using representative single-phase and multiphase benchmarks. For the lid-driven cavity flow with an LPTT fluid, the predicted velocity profiles show excellent agreement with the reference data and stress fields show a second-order mesh convergence when compared against the finest grid. The Taylor vortices

18

test case isolates the numerical impact of the stress-velocity coupling by tracking the domain integrated kinetic energy. The direct substitution parameter set recovers the solvent-stress baseline and converges to the analytical value with central differencing, while scaled stress function and polymer viscosity cases show a residual that decays at second order under mesh refinement, confirming that the coupling preserves second-order accuracy of the current finite-volume discretization within the proposed fully coupled algorithm. For a Newtonian droplet sheared in a shear-thinning Giesekus fluid, the predicted transient Taylor parameter deformation agrees closely with the reference results up to the highest reported Weissenberg number, and remains stable and physically consistent even when extended to substantially higher Weissenberg number. This benchmark therefore supports both interfacial accuracy and robustness with increasing elasticity, without requiring underrelaxation or a log-transformation. Finally, for a bubble rising in an EPTT viscoelastic liquid, the simulations clearly distinguish the transient rise dynamics in the subcritical and supercritical regimes. The predicted terminal rise velocity as a function of bubble volume reproduces the experimentally observed jump discontinuity, while the flow fields show the onset and downstream development of a negative wake in the supercritical case. A grid and time step sensitivity study confirms that these rise velocity predictions are numerically well resolved at the chosen resolution. This benchmark is particularly demanding because it combines strong interface deformation, sharp localization of viscoelastic stresses near the interface, and a regime transition that is highly sensitive to stress-flow coupling. Capturing all of these features consistently highlights the robustness and effectiveness of the fully implicit coupled formulation for challenging viscoelastic interfacial dynamics. Overall, the four test cases demonstrate that the proposed fully implicit coupled formulation delivers accurate and robust solutions from single phase viscoelastic flows to strongly elastic interfacial dynamics, while maintaining robustness at high Weissenberg numbers without resorting to a log-conformation method.

Data Availability Statement

The data that support the findings of this study are reproducible and data is openly available in the repository with DOI 10.5281/zenodo.18547566, available at https://doi.org/10.5281/zenodo.18547566

Declaration of Generative AI and AI-assisted technologies in the writing process

The authors used OpenAI’s ChatGPT to assist with editing and proofreading this manuscript. The authors then reviewed and revised the text as necessary and assume full responsibility for the final content of the publication.

Acknowledgements

We thank Christian Gorges, Fabien Evrard, and Bruno Blais for fruitful discussions. This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), grant numbers 420239128 and 458610925, and by the Natural Sciences and Engineering Research Council of Canada (NSERC), funding reference number RGPIN-2024-04805.


## References

[1] B. Williamson, K. Walters, T. Bates, R. Coy, A. Milton, The viscoelastic properties of multigrade oils and their effect on journal-bearing characteristics, Journal of Non-Newtonian Fluid Mechanics 73 (1997) 115–126.

[2] S. Gamaniel, D. Dini, L. Biancofiore, The effect of fluid viscoelasticity in lubricated contacts in the presence of cavitation, Tribology International 160 (2021) 107011.

[3] K. Wei, D. Liang, M. Mei, X. Yang, L. Chen, A viscoelastic model of compression and relaxation behaviors in preforming process for carbon fiber fabrics with binder, Composites Part B: Engineering 158 (2019) 1–9.

[4] H. Yuk, X. Zhao, A New 3D Printing Strategy by Harnessing Deformation, Instability, and Fracture of Viscoelastic Inks, Advanced Materials 30 (2018) 1704028.

[5] A. Z. Nelson, B. Kundukad, W. K. Wong, S. A. Khan, P. S. Doyle, Embedded droplet printing in yield-stress fluids, Proceedings of the National Academy of Sciences 117 (2020) 5671–5679.

[6] V. Frumkin, M. Bercovici, Fluidic shaping of optical components, Flow 1 (2021) E2.

[7] P.-T. Brun, Fluid-Mediated Fabrication of Complex Assemblies, JACS Au (2022) jacsau.2c00427.

[8] S. K. Lai, Y.-Y. Wang, D. Wirtz, J. Hanes, Microand macrorheology of mucus, Advanced Drug Delivery Reviews 61 (2009) 86–100.

[9] D. A. Siginer, Stability of Non-Linear Constitutive Formulations for Viscoelastic Fluids, SpringerBriefs in Applied Sciences and Technology, Springer International Publishing, Cham, 2014.

19

[10] F. Dupret, J. Marchal, M. Crochet, On the consequence of discretization errors in the numerical calculation of viscoelastic flow, Journal of Non-Newtonian Fluid Mechanics 18 (1985) 173–186.

[11] M. Alves, P. Oliveira, F. Pinho, Numerical Methods for Viscoelastic Fluid Flows, Annual Review of Fluid Mechanics 53 (2021) 509–541.

[12] R. Keunings, On the high Weissenberg number problem, Journal of Non-Newtonian Fluid Mechanics 20 (1986) 209–226.

[13] T.-P. Tsai, D. S. Malkus, Numerical breakdown at high Weissenberg number in non-Newtonian contraction flows, Rheologica Acta 39 (2000) 62–70.

[14] K. Walters, M. F. Webster, The distinctive CFD challenges of computational rheology, International Journal for Numerical Methods in Fluids 43 (2003) 577–596.

[15] I. Keshtiban, F. Belblidia, M. Webster, Numerical simulation of compressible viscoelastic liquids, Journal of Non-Newtonian Fluid Mechanics 122 (2004) 131–146.

[16] F. Habla, A. Obermeier, O. Hinrichsen, Semi-implicit stress formulation for viscoelastic models: Application to three-dimensional contraction flows, Journal of Non-Newtonian Fluid Mechanics 199 (2013) 70–79.

[17] C. Fernandes, V. Vukčević, T. Uroić, R. Simoes, O. Carneiro, H. Jasak, J. Nóbrega, A coupled finite volume flow solver for the solution of incompressible viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 265 (2019) 99–115.

[18] R. Fattal, R. Kupferman, Constitutive laws for the matrix-logarithm of the conformation tensor, Journal of Non-Newtonian Fluid Mechanics 123 (2004) 281–285.

[19] R. Fattal, R. Kupferman, Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation, Journal of Non-Newtonian Fluid Mechanics 126 (2005) 23–37.

[20] F. Becker, K. Rauthmann, L. Pauli, P. Knechtges, An eigenvalue-free implementation of the log-conformation formulation, Journal of Non-Newtonian Fluid Mechanics 322 (2023) 105133.

[21] W. Doherty, T. N. Phillips, Z. Xie, The log-conformation formulation for singleand multi-phase axisymmetric viscoelastic flows, Journal of Computational Physics 508 (2024) 113014.

[22] M. A. Hulsen, R. Fattal, R. Kupferman, Flow of viscoelastic fluids past a cylinder at high Weissenberg number: Stabilized simulations using matrix logarithms, Journal of Non-Newtonian Fluid Mechanics 127 (2005) 27–39.

[23] A. Afonso, P. Oliveira, F. Pinho, M. Alves, The log-conformation tensor approach in the finite-volume method framework, Journal of Non-Newtonian Fluid Mechanics 157 (2009) 55–65.

[24] F. Martins, C. Oishi, A. Afonso, M. Alves, A numerical study of the Kernel-conformation transformation for transient viscoelastic fluid flows, Journal of Computational Physics 302 (2015) 653–673.

[25] M. Niethammer, H. Marschall, C. Kunkelmann, D. Bothe, A numerical stabilization framework for viscoelastic fluid flow using the finite volume method on general unstructured meshes, International Journal for Numerical Methods in Fluids 86 (2018) 131–166.

[26] C. Fernandes, A Fully Implicit Log-Conformation Tensor Coupled Algorithm for the Solution of Incompressible Non-Isothermal Viscoelastic Flows, Polymers 14 (2022) 4099.

[27] M. Tomé, A. Castelo, A. Afonso, M. Alves, F. Pinho, Application of the log-conformation tensor to threedimensional time-dependent free surface flows, Journal of Non-Newtonian Fluid Mechanics 175–176 (2012) 44–54.

[28] M. Niethammer, G. Brenn, H. Marschall, D. Bothe, An extended volume of fluid method and its application to single bubbles rising in a viscoelastic liquid, Journal of Computational Physics 387 (2019) 326–355.

[29] C. Fernandes, S. Faroughi, O. Carneiro, J. M. Nóbrega, G. McKinley, Fully-resolved simulations of particle-laden viscoelastic fluids using an immersed boundary method, Journal of Non-Newtonian Fluid Mechanics 266 (2019) 80–94.

[30] H. U. Naseer, Z. Ahmed, D. Izbassarov, M. Muradoglu, Dynamics and interactions of parallel bubbles rising in a viscoelastic fluid under buoyancy, Journal of Non-Newtonian Fluid Mechanics (2023) 105000.

[31] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130.

20

[32] F. Denner, F. Evrard, B. van Wachem, Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds, Journal of Computational Physics 409 (2020) 109348.

[33] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, Journal of Computational Physics 228 (2009) 180–201.

[34] M. Darwish, F. Moukalled, A fully coupled Navier-Stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444.

[35] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165.

[36] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255.

[37] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234.

[38] F. Denner, B. van Wachem, A Unified Algorithm for Interfacial Flows with Incompressible and Compressible Fluids, in: D. Zeidan, L. T. Zhang, E. G. Da Silva, J. Merker (Eds.), Advances in Fluid Mechanics: Modelling and Simulations, Springer Nature Singapore, Singapore, 2022, pp. 179–208.

[39] C. Gorges, F. Evrard, B. van Wachem, F. Denner, Reducing volume and shape errors in front tracking by divergence-preserving velocity interpolation and parabolic fit vertex positioning, Journal of Computational Physics 457 (2022) 111072.

[40] M. Darwish, A. Aziz, F. Moukalled, A Coupled Pressure-Based Finite-Volume Solver for Incompressible TwoPhase Flow, Numerical Heat Transfer, Part B: Fundamentals 67 (2015) 47–74.

[41] F. Pimenta, M. Alves, A coupled finite-volume solver for numerical simulation of electrically-driven flows, Computers & Fluids 193 (2019) 104279.

[42] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–225.

[43] S. Osher, J. A. Sethian, Fronts Propagating with Curvature-Dependent Speed: Algorithms based on the Hamilton-Jacobi Formulation, Journal of Computational Physics 79 (1988) 12–49.

[44] M. Sussman, P. Smereka, S. Osher, A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow, Journal of Computational Physics 114 (1994) 146–159.

[45] S. Unverdi, G. Tryggvason, A Front-Tracking Method for Viscous, Incompressible, Multi-fluid Flows, Journal of Computational Physics 100 (1992) 25–37.

[46] G. Tryggvason, B. Bunner, A. Esmaeeli, D. Juric, N. Al-Rawahi, W. Tauber, J. Han, S. Nas, Y. Jan, A front-tracking method for the computations of multiphase flow, Journal of Computational Physics 169 (2001) 708–759.

[47] F. Habla, H. Marschall, O. Hinrichsen, L. Dietsche, H. Jasak, J. L. Favero, Numerical simulation of viscoelastic two-phase flows using openFOAM®, Chemical Engineering Science 66 (2011) 5487–5496.

[48] J. V. Giliberto, O. Desjardins, A sharp computational method for simulating multiphase viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 348 (2026) 105559.

[49] SB. Pillapakkam, P. Singh, A level-set method for computing solutions to viscoelastic two-phase flow, Journal of Computational Physics 174 (2001) 552–578.

[50] K. K. Kabanemi, J.-P. Marcotte, A level set method for simulating wrinkling of extruded viscoelastic sheets, Polymer Engineering & Science 60 (2020) 1662–1675.

[51] A. Amani, N. Balcázar, A. Naseri, J. Rigola, A numerical approach for non-Newtonian two-phase flows using a conservative level-set method, Chemical Engineering Journal 385 (2020) 123896.

[52] D. Izbassarov, M. Muradoglu, A front-tracking method for computational modeling of viscoelastic two-phase flow systems, Journal of Non-Newtonian Fluid Mechanics 223 (2015) 122–140.

[53] D. Wang, N. Wang, H. Liu, Droplet deformation and breakup in shear-thinning viscoelastic fluid under simple shear flow, Journal of Rheology 66 (2022) 585–603.

21

[54] P. Yue, J. J. Feng, C. Liu, J. Shen, A diffuse-interface method for simulating two-phase flows of complex fluids, Journal of Fluid Mechanics 515 (2004) 293–317.

[55] P. Yue, C. Zhou, J. J. Feng, C. F. Ollivier-Gooch, H. H. Hu, Phase-field simulations of interfacial dynamics in viscoelastic fluids using finite elements with adaptive meshing, Journal of Computational Physics 219 (2006) 47–67.

[56] M. Rodriguez, E. Johnsen, A high-order accurate five-equations compressible multiphase approach for viscoelastic fluids and solids with relaxation and elasticity, Journal of Computational Physics 379 (2019) 70–90.

[57] K. Zografos, A. M. Afonso, R. J. Poole, M. S. N. Oliveira, A viscoelastic two-phase solver using a phase-field approach, Journal of Non-Newtonian Fluid Mechanics 284 (2020).

[58] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, Journal of Computational Physics 285 (2015) 24–40.

[59] S. Popinet, Numerical models of surface tension, Annual Review of Fluid Mechanics 50 (2018) 49–75.

[60] F. Denner, F. Evrard, B. van Wachem, Breaching the capillary time-step constraint using a coupled VOF method with implicit surface tension, Journal of Computational Physics 459 (2022) 111128.

[61] R. Janodet, B. van Wachem, F. Denner, A fully-coupled algorithm with implicit surface tension treatment for interfacial flows with large density ratios, Journal of Computational Physics 520 (2025) 113520.

[62] N. Phan-Thien, R. I. Tanner, A new constitutive equation derived from network theory, Journal of NonNewtonian Fluid Mechanics 2 (1977) 353–365.

[63] H. Giesekus, A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility, Journal of Non-Newtonian Fluid Mechanics 11 (1982) 69–109.

[64] C. Gorges, A. Hodžić, F. Evrard, B. van Wachem, C. M. Velte, F. Denner, Efficient reduction of vertex clustering using front tracking with surface normal propagation restriction, Journal of Computational Physics 491 (2023) 112406.

[65] J. H. Snoeijer, A. Pandey, M. A. Herrada, J. Eggers, The relationship between viscoelasticity and elasticity, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 476 (2020) 20200419.

[66] M. A. Alves, F. T. Pinho, P. J. Oliveira, Study of steady pipe and channel flows of a single-mode Phan-Thien– Tanner fluid, Journal of Non-Newtonian Fluid Mechanics 101 (2001) 55–76.

[67] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentumweighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177– 208.

[68] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016.

[69] M. A. Alves, P. J. Oliveira, F. T. Pinho, A convergent and universally bounded interpolation scheme for the treatment of advection, International Journal for Numerical Methods in Fluids 41 (2003) 47–75.

[70] I. Demirdžić, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255.

[71] S. Mathur, J. Murthy, A pressure-based method for unstructured meshes, Numerical Heat Transfer Part B Fundamentals 31 (1997) 195–215.

[72] J. Ferziger, Interfacial transfer in Tryggvason’s method, International Journal for Numerical Methods in Fluids 41 (2003) 551–560.

[73] J. Mencinger, I. Zun, On the finite volume discretization of discontinuous body force field on collocated grid: Application to VOF method, Journal of Computational Physics 221 (2007) 524–538.

[74] R. Guénette, M. Fortin, A new mixed finite element method for computing viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 60 (1995) 27–52.

[75] F. Pimenta, M. Alves, Stabilization of an open-source finite-volume solver for viscoelastic fluid flows, Journal of Non-Newtonian Fluid Mechanics 239 (2017) 85–104.

22

[76] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532.

[77] C. S. Peskin, Numerical analysis of blood flow in the heart, Journal of Computational Physics 25 (1977) 220–252.

[78] G. Tryggvason, R. Scardovelli, S. Zaleski, Direct Numerical Simulations of Gas-Liquid Multiphase Flows, Cambridge University Press, Cambridge ; New York, 2011.

[79] M. R. Pivello, A Fully Adaptive Front-Tracking Method for the Simulation of 3D Two-Phase Flows, Ph.D. thesis, University of Uberlandia, Uberlandia, 2012.

[80] F. de Sousa, N. Mangiavacchi, L. Nonato, A. Castelo, M. Tomé, V. Ferreira, J. Cuminato, S. McKee, A front-tracking/front-capturing method for the simulation of 3D multi-fluid flows with free surfaces, Journal of Computational Physics 198 (2004) 469–499.

[81] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017.

[82] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017.

[83] K. Yapici, A comparison study on high-order bounded schemes: Flow of PTT-linear fluid in a lid-driven square cavity, Korea-Australia Rheology Journal 24 (2012) 11–21.

[84] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annual Research Briefs, Center for Turbulence (2004) 3–14.

[85] C. Pilz, G. Brenn, On the critical bubble volume at the rise velocity jump discontinuity in viscoelastic liquids, Journal of Non-Newtonian Fluid Mechanics 145 (2007) 124–138.

[86] D. Bothe, Sharp-interface continuum thermodynamics of multicomponent fluid systems with interfacial mass, International Journal of Engineering Science 179 (2022) 103731.

23

