## Fully coupled implicit finite-volume algorithm for viscoelastic interfacial flows 

Ayman Mazloum[a] , Gabriele Gennari[a] , Fabian Denner[b] , Berend van Wachem[a,][∗] 

> _aChair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany_ 

> _bDepartment of Mechanical Engineering, Polytechnique Montréal, Montréal, H3T 1J4, Québec, Canada_ 

## **Abstract** 

A fully coupled implicit finite-volume algorithm for incompressible viscoelastic interfacial flows is proposed, whereby the viscoelasticity of the flow is described by an upper-convected Maxwell constitutive model, including limited extensibility and shear-thinning behaviour. The governing equations describing the conservation of continuity and momentum, as well as the constitutive model are discretized using standard finite-volume methods and are solved for pressure, velocity and the polymer stress tensor in a single linear system of equations. Treating all terms of the linearized and discretized governing equations implicit in velocity, pressure and/or the components of the polymer stress tensor, a tightly coupled system of equations is obtained. The interface separating the interacting bulk phases and the surface tension acting at the fluid interface are modelled using a state-of-the-art front-tracking method. We demonstrate the capabilities of the proposed numerical framework with four representative test cases, including the deformation of a viscoelastic droplet in shear flow at large Weissenberg numbers of up to Wi = 10[4] , and the jump discontinuity of the rise velocity of a bubble rising in a viscoelastic liquid as a result of a “negative wake”. Contrary to previous studies using segregated algorithms, the proposed fully coupled implicit algorithm does not apply or require a log-conformation approach to predict these flows. Overall, the fully implicit coupled front-tracking formulation provides a robust framework to reliable numerical predictions of strongly elastic interfacial flows at large Weissenberg numbers. 

_Keywords:_ Viscoelastic flow, Interfacial flows, Fully coupled algorithm, Shear-thinning, High Weissenberg number 

## **1. Introduction** 

Classical engineering applications of viscoelastic fluids, such as the lubrication dynamics in bearings [1, 2] and binders for composite materials [3], as well as emerging manufacturing techniques by 3D printing [4, 5] and the assembly of soft materials [6, 7] have been driving an increased interest in viscoelastic interfacial flows, in which two or more immiscible fluids interact with each other, and the physical phenomena associated with these flows. The airborne dispersion of respiratory diseases, e.g. corona viruses, gave the interest in viscoelastic interfacial flows a further boost in the wake of the recent pandemic, since mucus, the primary carrier of the viral load, exhibits dominant viscoelastic properties [8]. As a result of these relevant and timely applications of viscoelastic interfacial flows, their numerical modelling has become a subject of growing activity in the scientific community. Even though computational rheology has developed into a mature discipline in recent decades, a large variety of constitutive models for the viscoelastic stress (typically referred to as the _polymer stress_ ) based on often competing assumptions [9], large differences in numerical predictions and different interpretations of the underpinning physical phenomena [10, 11], as well as convergence difficulties for flows with even moderate elasticity, known as the high-Weissenberg number problem [12–14], are still hampering a widespread adoption and application of numerical modelling tools for viscoelastic flows. 

State-of-the-art algorithms for incompressible viscoelastic flows mostly rely on segregated algorithms [11], in which the momentum equations, a pressure projection equation satisfying the continuity constraint and the constitutive model describing the viscoelastic stresses are solved sequentially. However, the weak explicit coupling between velocity, pressure and polymer stress in the discretized governing equations as a result of the iterative predictorcorrector solution procedure severely limits the stability and convergence of these algorithms, requiring a strong underrelaxation of the discretized equations to reach a converged solution [15–17]. Numerical methods to mitigate problems associated with a high elasticity of the fluid, most notably the log-conformation approach [18–21], have laid the foundation for new developments that substantially expanded the parameter range, especially with respect to the Weissenberg number, that can now be simulated routinely with widely available computational tools [16, 22–26], including viscoelastic multiphase flows [27–30]. 

Coupled implicit algorithms, whereby all governing equations are solved simultaneously in a single linear system of implicitly coupled equations, present a powerful alternative to the widely used segregated algorithms. This class of 

> ∗Corresponding author: _Email address:_ `berend.van.wachem@multiflow.org` (Berend van Wachem) 

algorithms, which has been applied successfully to incompressible and compressible Newtonian single-phase [31–35] and interfacial [36–40] flows, allows a tight implicit coupling of the governing equations, improving the stability and performance of the solution algorithm. Fernandes et al. [17] presented a coupled implicit finite-volume algorithm for two-dimensional viscoelastic flows, whereby the constitutive model for the polymer stress tensor is included in the linear system of implicitly coupled equations and the polymer stress tensor is treated implicitly in the momentum equations. Shortly after, Pimenta and Alves [41] proposed a fully coupled algorithm for three-dimensional electrically driven viscoelastic flows using a Poisson-Nernst-Planck model, further adding a Poisson equation governing the electrical potential and transport equations for the charge densities to the linear system of implicitly coupled equations. Recently, Fernandes [26] presented a fully coupled algorithm for two-dimensional viscoelastic flows applying a log-conformation approach to the constitutive model. All three studies [17, 26, 41] reported a robust and rapid convergence with an impressive speed-up exceeding one order of magnitude for some cases compared to a state-of-the-art segregated algorithm for viscoelastic flows. While the simulation of Newtonian interfacial flows, in which two (or more) immiscible fluids interact with each other, has become common and a large variety of numerical frameworks based on the volume-of-fluid [42], level-set [43, 44] and front-tracking [45, 46] methods are available, simulation tools that can accurately predict three-dimensional interfacial flows in which at least one of the interacting fluids is viscoelastic are still scarce. Finite-volume algorithms for viscoelastic interfacial flows using volume-of-fluid [28, 47, 48], level-set [49–51] and front-tracking [30, 52] methods have previously been presented, alongside algorithms based on the Lattice-Boltzmann method [53] and diffusive interface method [54–57]. For incompressible viscoelastic interfacial flows, state-of-the-art algorithms rely on solving for the conformation tensor of the polymer stress rather than for the polymer stress tensor directly, and current finite-volume algorithms are built upon segregated algorithms. However, a fully coupled implicit algorithm for viscoelastic interfacial flows has not yet been presented in the literature. In the context of Newtonian flows, fully coupled implicit algorithms have been demonstrated to be particularly suited for interfacial flows as they can be readily applied to interfacial flows with large density ratios. Additionally, the capillary time-step constraint, a severe impediment for the simulation of flows with surface tension [58, 59], can be breached when surface tension is treated implicitly [60]. 

Building upon recently published algorithms on viscoelastic single-phase flows [17, 26, 41] and our own prior work on fully coupled implicit algorithms for Newtonian flows [32, 36, 60, 61], we propose a fully coupled implicit finite-volume algorithm for incompressible viscoelastic interfacial flows. To account for the viscoelasticity of the flow, we consider the upper-convected Maxwell constitutive equation for the polymer stress tensor, including limited extensibility and shear-thinning behaviour, demonstrated with the linear and exponential Phan-Thien-Tanner models [62], as well as the Giesekus model [63]. The governing equations describing the conservation of continuity and momentum, as well as the constitutive model are discretized using standard finite-volume methods and are solved for pressure, velocity and the polymer stress tensor in a single linear system of equations. The interface separating the interacting bulk phases and the surface tension acting at the fluid interface are modelled using a state-of-the-art fronttracking method [39, 64]. We demonstrate the capabilities of the proposed numerical framework with representative test cases, including the deformation of a viscoelastic droplet in shear flow at large Weissenberg numbers and the jump discontinuity of the rise velocity of a bubble rising in a viscoelastic liquid. Notably, the proposed algorithm does not apply nor require a log-conformation approach to predict these flows, contrary to the studies presented in the literature to date. 

Section 2 introduces the governing conservation laws and the considered constitutive model. The discretization of these governing equations and the numerical framework are described in Section 3, where we examine the discretization of each term of the governing equations, and the front-tracking method employed to represent and transport the fluid interface is briefly reviewed in Section 4. A presentation of the complete discretized governing equations and an explanation of the solution procedure is the subject of Section 5. The results of four representative test cases are presented and discussed in Section 6, and the article is concluded in Section 7. 

## **2. Governing equations** 

The considered incompressible and isothermal viscoelastic flows are governed by the continuity and momentum equations, given as 

**==> picture [415 x 11] intentionally omitted <==**

**==> picture [482 x 23] intentionally omitted <==**

respectively, where _t_ denotes time, **u** ≡( _u v w_ )[T] denotes the velocity vector, _ρ_ is the fluid density, **S** _σ_ is the source term representing surface tension (when an interface is present) and **S** _g_ = _ρ_ **g** is the source term representing gravity, with **g** the gravitational acceleration. The stress tensor _**ς**_ = − _p_ **I** + _**τ**_ s + _**τ**_ p comprises the normal stress tensor exerted by pressure, − _p_ **I** , where **I** denotes the identity tensor, the solvent stress tensor _**τ**_ s and the polymer stress tensor _**τ**_ p, 

2 

such that the momentum equations follow as 

**==> picture [482 x 22] intentionally omitted <==**

The solvent stress tesnor is based on Newton’s law of viscosity and given as _**τ**_ s = 2 _µ_ **D** , where _µ_ is the dynamic solvent viscosity, and **D** =[1] 2[(∇] **[u]**[ + ∇] **[u]**[T][)][is][the][rate][of][deformation][tensor.] 

The non-Newtonian viscoelastic behaviour of the flow is expressed by the evolution of the polymer stress tensor _**τ**_ p, which, in turn, is governed by a general differential constitutive equation of the form [11] 

**==> picture [482 x 24] intentionally omitted <==**

where _ψ_ is the stress function, _λ_ is the relaxation time, _ξ_ is the non-affine or “slip” parameter, _α_ is the mobility parameter and _η_ is the polymer viscosity. The first four terms gathered in big parentheses in Eq. (4) together constitute the upper-convected time derivative of the polymer stress tensor, where the third and fourth terms ensure the correct transformation of _**τ**_ p under deformations by the flow [65], and the fifth term captures the slip between the molecular network and the continuum medium [66]. Note that the polymer stress tensor is symmetric and, therefore, has only 6 unique components. As constitutive models we consider the Giesekus model [63], for which 0 < _α_ ≤ 0 _._ 5, _ψ_ = 1 and _ξ_ = 0, the linear Phan-Thien-Tanner model (LPTT) [62], for which _α_ = 0 and _ψ_ = 1 + _λϵ_ tr( _**τ**_ p)/ _η_ , and the exponential Phan-Thien-Tanner model (EPTT) [62], for which _α_ = 0 and _ψ_ = exp [ _λϵ_ tr( _**τ**_ p)/ _η_ ], where _ϵ_ is the extensibility coefficient of the fluid. All three models are widely used in the literature and convey the primary features of modelling viscoelastic flows. 

In order to distinguish the interacting fluid phases, we consider a generic indicator function I that is reconstructed based on the position of the interface. The indicator function is defined as 

**==> picture [482 x 31] intentionally omitted <==**

where Ω = Ωa ∪ Ωb is the computational domain, with Ωa and Ωb the subdomains occupied by fluids “a” and “b”, respectively. Following previous work on viscoelastic interfacial flows [52, 56], the fluid properties _ϕ_ ∈{ _α,η,λ,µ,ρ,ψ, ξ_ } are defined based on the indicator function as _ϕ_ ( **x** ) = _ϕ_ a +I( **x** )( _ϕ_ b − _ϕ_ a). Surface tension is modelled as a volumetric source term **S** _σ_ in the momentum equations, 

**==> picture [482 x 10] intentionally omitted <==**

where _σ_ is the surface tension coefficient, _κ_ is the interface curvature, **n** Σ is the normal vector of the interface and _δ_ Σ is the interfacial delta function. 

## **3. Numerical framework** 

The proposed numerical framework is implemented in our in-house finite-volume solver `MultiFlow` and is built upon a collocated second-order finite-volume discretization and a fully coupled implicit solution algorithm [32, 67], whereby the discretized governing equations are solved in a single system of linearized equations, **A** ⋅ _**ζ**_ = **b** , with the three velocity components, pressure and the six unique components of the polymer stress tensor as the implicitly sought solution variables. Below we describe the discretization of the individual terms of the governing conservation laws, Eqs. (1) and (2), and the constitutive model, Eq. (4), as well as the stress-velocity coupling. The interface is modelled using a front-tracking method [39], presented in Section 4, although any other suitable method to capture or track the interface, e.g., volume-of-fluid or level-set methods, may equally be applied together with the proposed numerical framework. 

## _3.1. Discretization methods_ 

The discretization is based on a standard second-order finite-volume method. Discretizing the generic convectiondiffusion equation of a general fluid variable _ϕ_ , 

**==> picture [481 x 22] intentionally omitted <==**

where _Dϕ_ is the diffusion coefficient of _ϕ_ and _S_ is a generic source term, with the employed second-order finite-volume method is given in semi-discretized form for cell _P_ of an arbitrary computational mesh as 

**==> picture [479 x 26] intentionally omitted <==**

3 

where _f_ denotes all faces bounding mesh cell _P_ , ◻˜ denotes a flux-limited interpolation, **n** _f_ is the normal vector of face _f_ pointing out of cell _P_ , and the area of face _f_ and the volume of cell _P_ are denoted with _Af_ and _VP_ , respectively. The flux _Ff_ through face _f_ is defined as 

**==> picture [482 x 12] intentionally omitted <==**

where the advecting velocity _ϑf_ = **u** _f_ ⋅ **n** _f_ is obtained by a momentum-weighted interpolation (MWI), as discussed in detail in Section 3.6. The transient derivative is discretized using the second-order backward Euler (BDF2) scheme for a variable time-step as [68] 

**==> picture [479 x 25] intentionally omitted <==**

where ∆ _t_ 1 is the current time-step and ∆ _t_ 2 is the previous time-step. The superscripts ( _t_ ), ( _t_ − ∆ _t_ 1) and ( _t_ − ∆ _t_ 1 − ∆ _t_ 2) denote the solution at the current time-level, the previous time-level, and the previous-previous time-level, respectively. In the discretization of the advection term, _ϕf_ is interpolated from the values at adjacent cell centers using a flux-limited interpolation scheme, given as 

**==> picture [104 x 13] intentionally omitted <==**

**==> picture [19 x 11] intentionally omitted <==**

where _χf_ denotes the flux limiter, and subscripts _D_ and _U_ denote the downwind cell and the upwind cell of face _f_ , respectively. In this study, we consider the central-differencing scheme as well as the CUBISTA scheme [69], which is widely used to compute viscoelastic flows, to determine the flux limiter _χf_ , but other suitable schemes may equally be applied. The face-centered velocity gradient projected along the normal vector of the cell face, ∇ _ϕf_ ⋅ **n** _f_ , in the discretized diffusion term is decomposed into an orthogonal and a non-orthogonal part to correct for any non-orthogonality of the computational mesh, following the work of Demirdžić and Muzaferija [70], as 

**==> picture [482 x 25] intentionally omitted <==**

where _cf_ = ( **n** _f_ ⋅ **s** _f_ )[−][1] is the scaling factor of the decomposition [71] and where ~~◻~~ _f_ = (1 − _ℓf_ ) ◻ _P_ + _ℓf_ ◻ _Q_ denotes a linear interpolation, with _ℓf_ the inverse-distance weighting coefficient with respect to cell _P_ and face _f_ . The vector **s** _f_ is the unit vector connecting the cell centers adjacent to face _f_ , pointing from cell _P_ to neighbour cell _Q_ and ∆ _sf_ = ∣ **s** _f_ ∣ is the distance between the centers of cells _P_ and _Q_ . 

## _3.2. Continuity equation_ 

Applying the divergence theorem, the continuity equation, Eq. (1), is readily discretized using the flux _Ff_ through face _f_ , as defined in Eq. (9), as 

**==> picture [482 x 24] intentionally omitted <==**

To this end, the implicitly treated advecting velocity given by the MWI, see Section 3.6, of the form 

**==> picture [482 x 17] intentionally omitted <==**

introduces an implicit dependency of the continuity equation on pressure, such that the continuity equation can be solved implicitly for velocity and pressure in a fully coupled manner [32]. 

The iteration counter _n_ is associated with nonlinear iterations performed to solve the system of discretized governing equations at each time-step, as further explained in Section 5, with superscript ( _n_ ) denoting deferred quantities and superscript ( _n_ + 1) denoting quantities for which the solution is sought implicitly. 

## _3.3. Momentum equations_ 

The momentum equations are solved implicitly in velocity, pressure and the components of the polymer stress tensor. The transient term is discretized, at mesh cell _P_ , as 

**==> picture [482 x 26] intentionally omitted <==**

with the transient derivative of velocity following from Eq. (10) as 

**==> picture [479 x 44] intentionally omitted <==**

The advection term is discretized by applying a Newton linearization treating both the flow velocity **u** and the flux _F_ implicitly, 

**==> picture [482 x 24] intentionally omitted <==**

where ◻˜ denotes the flux-limited interpolation presented in Eq. (11). 

The divergence of the stress tensor appearing on the right-hand side of the momentum equations, Eq. (2), is split into three parts, ∇⋅ _**ς**_ = −∇ _p_ + ∇⋅ _**τ**_ s + ∇⋅ _**τ**_ p, each of which is discretized separately. The pressure gradient, ∇ _p_ = ∇⋅( _p_ **I** ), and the divergence of the polymer stress tensor, ∇⋅ _**τ**_ p, are both discretized using linear interpolation of the respective cell values to the mesh faces as 

**==> picture [480 x 52] intentionally omitted <==**

Both _p_ and _**τ**_ p are treated implicitly, as indicated by the superscript ( _n_ + 1). The divergence of the solvent stress tensor _**τ**_ s is discretized as [32] 

**==> picture [482 x 25] intentionally omitted <==**

where, including the correction for non-orthogonal meshes presented in Eq. (12), 

**==> picture [482 x 22] intentionally omitted <==**

A harmonic interpolation is applied to interpolate the viscosity values from the cell centers _P_ and _Q_ to the shared face _f_ [72], 

**==> picture [481 x 25] intentionally omitted <==**

where _µ_ ˘ _f_ is the harmonic interpolated face-centered viscosity, and _ℓf_ is the inverse-distance weighting coefficient with respect to cell _P_ and face _f_ . The solvent stress term is implemented treating all velocity terms implicitly, such that 

**==> picture [482 x 33] intentionally omitted <==**

Contrary to previous studies on fully coupled algorithms for Newtonian flows, e.g. [32, 33], and viscoelastic flows [17], all velocity gradients of the discretized solvent shear stress in Eq. (23) are solved implicitly. 

As shown in previous studies [36, 67, 73], the momentum sources **S** have to be discretized with the same discretization as the pressure gradients, for the discretized pressure gradient ∇ _pP_ to be able to match the discretized source term **S**[⋆] _P_[.][Following][Bartholomew][et][al.][[][67][],][the discretized momentum source] **[ S]**[⋆][is constructed based on the] untreated source term **S** as 

**==> picture [482 x 26] intentionally omitted <==**

## _3.4. Constitutive model_ 

The constitutive model, Eq. (4), yields six governing equations for the six unique components of the polymer stress tensor, which are solved implicitly for the components of the polymer stress tensor, as well as velocity and pressure. 

In the constitutive model, the first term on the left-hand side of Eq. (4) is treated implicitly as 

**==> picture [482 x 22] intentionally omitted <==**

and the transient term of the constitutive model, as part of the upper-convected derivative of the polymer stress tensor, is discretized in the same manner as transient term of the momentum equations, 

**==> picture [482 x 27] intentionally omitted <==**

5 

using the second-order backward Euler scheme presented in Eq. (10). 

The advection term of the constitutive model, Eq. (4), arises from the material derivative of _**τ**_ p and is, consequently, not in conserved form, contrary to the advection term of the momentum equations. In the interest of a discretization that is consistent with the advection of momentum, we reformulate the advection term of _**τ**_ p using the product rule as 

**==> picture [482 x 11] intentionally omitted <==**

such that the advecting velocity can now be applied to define the fluxes through the mesh faces. Similar to the advection term, a Newton linearization is applied, with 

**==> picture [482 x 33] intentionally omitted <==**

For the considered incompressible flows, the (∇⋅ **u** )-term in Eq. (27) is superfluous from a mathematical viewpoint. Numerically, however, this is only true for a converged result, but may not be the case during the initial nonlinear iterations in each time-step. Including the (∇⋅ **u** )-term, therefore, generally improves the convergence of the solution algorithm. 

Aiming to fully exploit the implicit coupling provided by the fully coupled solution procedure, the two remaining terms of the upper-convected time derivative of the polymer stress tensor and the non-affine response term are linearized and treated implicit using a Newton linearization respectively, 

**==> picture [480 x 42] intentionally omitted <==**

**==> picture [482 x 80] intentionally omitted <==**

the quadratic stress term of the Giesekus model is treated implicitly with respect to the polymer stress tensor, 

**==> picture [482 x 24] intentionally omitted <==**

and the strain-rate tensor is treated implicitly with respect to the velocity, 

**==> picture [482 x 21] intentionally omitted <==**

Fernandes et al. [17] treated both the additional terms of the upper-convective time derivative, Eq. (29), and the strain-rate tensor, Eq. (32), explicitly. Recently, Fernandes [26] also applied a Newton linearization to the additional terms of the upper-convective time derivative in their fully coupled log-conformation algorithm, treating these terms implicitly in the conformation tensor and the velocity. Pimenta and Alves [41] treated the first term of Eq. (32) implicitly with respect to velocity, while treating the second term explicitly. 

Contrary to the fully coupled algorithm of Fernandes et al. [17], we do not divide the discretized constitutive model by the relaxation time _λ_ before discretization. Hence, the constitutive model is valid for _λ_ ≥ 0. Considering, for example, the upper-convected Maxwell model ( _ψ_ = 1, _µ_ = 0), Eq. (4) reduces to _**τ**_ p = _η_ (∇ **u** + ∇ **u**[T] ) for _λ_ = 0, resulting in a Newtonian flow with shear viscosity _η_ . 

## _3.5. Stress-velocity coupling_ 

The stress-velocity coupling associated with the polymer stress is a central building block of the proposed methodology. The commonly applied method to ensure a robust coupling between the polymer stress and the velocity when a collocated variable arrangement is used, is to introduce two mathematically equivalent diffusion terms with 

6 

**==> picture [292 x 146] intentionally omitted <==**

**----- Start of picture text -----**<br>
P Q P Q<br>× f × f<br>(a) Small-stencil treatment of face f (b) Large-stencil treatment of face f<br>**----- End of picture text -----**<br>


Figure 1: Interpolation stencils of the velocity at face _f_ , with the adjacent cells _P_ and _Q_ , considered for the stress-velocity coupling. 

opposite signs on the right-hand side of the momentum equations, Eq. (2), to yield 

**==> picture [482 x 33] intentionally omitted <==**

where _η_ ˆ is a weighting factor that is dimensionally equivalent to a dynamic viscosity. As the notation indicates, the additional terms are discretized on different computational stencils. For clarity, we first consider an equidistant Cartesian mesh with _cf_ = 1 and ∆ _sf_ = ∆ _x_ , with the discretization for general non-orthogonal meshes given thereafter in Eqs. (37) and (38). On an equidistant Cartesian mesh, the _small-stencil_ diffusion term is discretized as 

**==> picture [482 x 25] intentionally omitted <==**

and the _large-stencil_ diffusion term as 

**==> picture [482 x 24] intentionally omitted <==**

as illustrated in Figure 1. In the literature, this procedure of adding two diffusion terms with opposite signs is widely ˆ referred to as _both-sides diffusion_ (BSD) [74], typically applied with _η_ = _η_ [16, 17, 26, 75]. Applying the conventionally used centered finite-difference approximations described above, the additional small-stencil and large-stencil diffusion terms yield, using tensor notation and the Einstein summation convention, 

**==> picture [482 x 32] intentionally omitted <==**

which infers that, by taking the divergence of this term in the momentum equations, the two additional diffusion terms introduce numerical diffusion ( _∂u_[4] _i_[/] _[∂x]_[4] _j_[)][with][a][magnitude][proportional][to] _[η]_[ˆ][and][∆] _[x]_[2][[][75][].] ˆ ˘ In line with the discretization of the divergence of the solvent stress described in Eq. (23) and with _η_ = _ηf_ , the small-stencil and large-stencil terms are discretized as 

**==> picture [482 x 60] intentionally omitted <==**

respectively. The contribution of the stress-velocity coupling to the right-hand side of the discretized momentum equations can, thus, be summarized as 

**==> picture [482 x 38] intentionally omitted <==**

To fully exploit the implicit coupling afforded by the fully coupled algorithm and to be consistent with the treatment of the strain-rate term in the constitutive model, all velocity contributions in Eq. (39) are treated implicitly. 

7 

## _3.6. Pressure-velocity coupling_ 

To ensure a robust pressure-velocity coupling using the employed collocated variable arrangement, the flux _Ff_ = _ϑf Af_ through face _f_ is defined with an advecting velocity _ϑf_ = **u** _f_ ⋅ **n** _f_ that is evaluated using a momentum-weighted interpolation (MWI) [67], originally introduced by Rhie and Chow [76]. This advecting velocity allows to solve the continuity equation for pressure [32] and prevents pressure-velocity decoupling on the employed collocated variable arrangement [67]. 

The advecting velocity is, based on the unified formulation of the momentum-weighted interpolation proposed by Bartholomew et al. [67], defined as 

**==> picture [482 x 24] intentionally omitted <==**

where _ρ_ ˘ _f_ is the harmonically averaged face density. The weighting factor _d_[ˆ] _f_ defines the strength of the pressurevelocity coupling and is given as 

**==> picture [482 x 53] intentionally omitted <==**

The coefficients _aP_ and _aQ_ are defined based on the diagonal matrix coefficients of the velocity arising from the advection term, see Eq. (17), the solvent stress term, see Eq. (23), and the small-stencil stress-coupling term, see Eq. (39), of the discretized momentum equations associated with the cells adjacent to face _f_ . For the discretization presented above, the coefficient _aP_ (and, analogously, _aQ_ ) is given as 

**==> picture [482 x 27] intentionally omitted <==**

where _χ_[′] _f_[= (][1][ −] _[χ][f]_[)][if] _[F][f]_[≥][0][and] _[χ]_[′] _f_[=] _[ χ][f]_[if] _[F][f]_[<][ 0][.][For][the][MWI][to][be][time-step][independent,] _[a][P]_[and] _[a][Q]_[must][not] include the contribution of the transient terms to the diagonal coefficient [67]. 

For an arbitrary unstructured mesh and including a density-weighting of the large-stencil pressure and source term contributions, the discretized and implicitly treated advecting velocity is defined as [67] 

**==> picture [482 x 69] intentionally omitted <==**

The discretized pressure terms together constitute a low-pass filter on the pressure field that prevents pressure-velocity decoupling [67]. Contrary to most previous work on fully coupled algorithms [17, 32, 33], we treat all pressure terms in Eq. (43) implicitly [61]. 

## **4. Front-tracking method** 

The numerical framework proposed in the previous section is complemented by a front-tracking method [39] to track the fluid interface separating to immiscible bulk phases. Since the precise formulation and implementation of the applied interface tracking (or interface capturing) method is not critical for the proposed numerical framework, we only provide a brief overview of the applied front-tracking method and refer the reader to our recent publications [39, 64] for more details. 

In front tracking [45, 46], the fluid interface is represented by a triangulated surface mesh. Each vertex _i_ of this surface mesh is advected in a Lagrangian manner, 

**==> picture [481 x 22] intentionally omitted <==**

where **x** _i_ and **u** are the location and (interpolated) velocity of vertex _i_ , respectively. The vertices of the surface mesh can, consequently, also move tangential to the interface, which may lead to vertex clustering and a deteriorating quality of the surface mesh, in turn requiring extensive remeshing of the surface mesh to retain an acceptable mesh quality. In order to address this issue, we apply the _normal-only advection_ (NOA) of the vertices [64], with the 

8 

