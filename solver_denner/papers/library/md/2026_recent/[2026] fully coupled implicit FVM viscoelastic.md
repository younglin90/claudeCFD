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


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0002-05.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0002-06.png)


respectively, where _t_ denotes time, **u** ≡( _u v w_ )[T] denotes the velocity vector, _ρ_ is the fluid density, **S** _σ_ is the source term representing surface tension (when an interface is present) and **S** _g_ = _ρ_ **g** is the source term representing gravity, with **g** the gravitational acceleration. The stress tensor _**ς**_ = − _p_ **I** + _**τ**_ s + _**τ**_ p comprises the normal stress tensor exerted by pressure, − _p_ **I** , where **I** denotes the identity tensor, the solvent stress tensor _**τ**_ s and the polymer stress tensor _**τ**_ p, 

2 

such that the momentum equations follow as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-01.png)


The solvent stress tesnor is based on Newton’s law of viscosity and given as _**τ**_ s = 2 _µ_ **D** , where _µ_ is the dynamic solvent viscosity, and **D** =[1] 2[(∇] **[u]**[ + ∇] **[u]**[T][)][is][the][rate][of][deformation][tensor.] 

The non-Newtonian viscoelastic behaviour of the flow is expressed by the evolution of the polymer stress tensor _**τ**_ p, which, in turn, is governed by a general differential constitutive equation of the form [11] 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-04.png)


where _ψ_ is the stress function, _λ_ is the relaxation time, _ξ_ is the non-affine or “slip” parameter, _α_ is the mobility parameter and _η_ is the polymer viscosity. The first four terms gathered in big parentheses in Eq. (4) together constitute the upper-convected time derivative of the polymer stress tensor, where the third and fourth terms ensure the correct transformation of _**τ**_ p under deformations by the flow [65], and the fifth term captures the slip between the molecular network and the continuum medium [66]. Note that the polymer stress tensor is symmetric and, therefore, has only 6 unique components. As constitutive models we consider the Giesekus model [63], for which 0 < _α_ ≤ 0 _._ 5, _ψ_ = 1 and _ξ_ = 0, the linear Phan-Thien-Tanner model (LPTT) [62], for which _α_ = 0 and _ψ_ = 1 + _λϵ_ tr( _**τ**_ p)/ _η_ , and the exponential Phan-Thien-Tanner model (EPTT) [62], for which _α_ = 0 and _ψ_ = exp [ _λϵ_ tr( _**τ**_ p)/ _η_ ], where _ϵ_ is the extensibility coefficient of the fluid. All three models are widely used in the literature and convey the primary features of modelling viscoelastic flows. 

In order to distinguish the interacting fluid phases, we consider a generic indicator function I that is reconstructed based on the position of the interface. The indicator function is defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-07.png)


where Ω = Ωa ∪ Ωb is the computational domain, with Ωa and Ωb the subdomains occupied by fluids “a” and “b”, respectively. Following previous work on viscoelastic interfacial flows [52, 56], the fluid properties _ϕ_ ∈{ _α,η,λ,µ,ρ,ψ, ξ_ } are defined based on the indicator function as _ϕ_ ( **x** ) = _ϕ_ a +I( **x** )( _ϕ_ b − _ϕ_ a). Surface tension is modelled as a volumetric source term **S** _σ_ in the momentum equations, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-09.png)


where _σ_ is the surface tension coefficient, _κ_ is the interface curvature, **n** Σ is the normal vector of the interface and _δ_ Σ is the interfacial delta function. 

## **3. Numerical framework** 

The proposed numerical framework is implemented in our in-house finite-volume solver `MultiFlow` and is built upon a collocated second-order finite-volume discretization and a fully coupled implicit solution algorithm [32, 67], whereby the discretized governing equations are solved in a single system of linearized equations, **A** ⋅ _**ζ**_ = **b** , with the three velocity components, pressure and the six unique components of the polymer stress tensor as the implicitly sought solution variables. Below we describe the discretization of the individual terms of the governing conservation laws, Eqs. (1) and (2), and the constitutive model, Eq. (4), as well as the stress-velocity coupling. The interface is modelled using a front-tracking method [39], presented in Section 4, although any other suitable method to capture or track the interface, e.g., volume-of-fluid or level-set methods, may equally be applied together with the proposed numerical framework. 

## _3.1. Discretization methods_ 

The discretization is based on a standard second-order finite-volume method. Discretizing the generic convectiondiffusion equation of a general fluid variable _ϕ_ , 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-15.png)


where _Dϕ_ is the diffusion coefficient of _ϕ_ and _S_ is a generic source term, with the employed second-order finite-volume method is given in semi-discretized form for cell _P_ of an arbitrary computational mesh as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0003-17.png)


3 

where _f_ denotes all faces bounding mesh cell _P_ , ◻˜ denotes a flux-limited interpolation, **n** _f_ is the normal vector of face _f_ pointing out of cell _P_ , and the area of face _f_ and the volume of cell _P_ are denoted with _Af_ and _VP_ , respectively. The flux _Ff_ through face _f_ is defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-01.png)


where the advecting velocity _ϑf_ = **u** _f_ ⋅ **n** _f_ is obtained by a momentum-weighted interpolation (MWI), as discussed in detail in Section 3.6. The transient derivative is discretized using the second-order backward Euler (BDF2) scheme for a variable time-step as [68] 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-03.png)


where ∆ _t_ 1 is the current time-step and ∆ _t_ 2 is the previous time-step. The superscripts ( _t_ ), ( _t_ − ∆ _t_ 1) and ( _t_ − ∆ _t_ 1 − ∆ _t_ 2) denote the solution at the current time-level, the previous time-level, and the previous-previous time-level, respectively. In the discretization of the advection term, _ϕf_ is interpolated from the values at adjacent cell centers using a flux-limited interpolation scheme, given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-05.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-06.png)


where _χf_ denotes the flux limiter, and subscripts _D_ and _U_ denote the downwind cell and the upwind cell of face _f_ , respectively. In this study, we consider the central-differencing scheme as well as the CUBISTA scheme [69], which is widely used to compute viscoelastic flows, to determine the flux limiter _χf_ , but other suitable schemes may equally be applied. The face-centered velocity gradient projected along the normal vector of the cell face, ∇ _ϕf_ ⋅ **n** _f_ , in the discretized diffusion term is decomposed into an orthogonal and a non-orthogonal part to correct for any non-orthogonality of the computational mesh, following the work of Demirdžić and Muzaferija [70], as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-08.png)


where _cf_ = ( **n** _f_ ⋅ **s** _f_ )[−][1] is the scaling factor of the decomposition [71] and where ~~◻~~ _f_ = (1 − _ℓf_ ) ◻ _P_ + _ℓf_ ◻ _Q_ denotes a linear interpolation, with _ℓf_ the inverse-distance weighting coefficient with respect to cell _P_ and face _f_ . The vector **s** _f_ is the unit vector connecting the cell centers adjacent to face _f_ , pointing from cell _P_ to neighbour cell _Q_ and ∆ _sf_ = ∣ **s** _f_ ∣ is the distance between the centers of cells _P_ and _Q_ . 

## _3.2. Continuity equation_ 

Applying the divergence theorem, the continuity equation, Eq. (1), is readily discretized using the flux _Ff_ through face _f_ , as defined in Eq. (9), as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-12.png)


To this end, the implicitly treated advecting velocity given by the MWI, see Section 3.6, of the form 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-14.png)


introduces an implicit dependency of the continuity equation on pressure, such that the continuity equation can be solved implicitly for velocity and pressure in a fully coupled manner [32]. 

The iteration counter _n_ is associated with nonlinear iterations performed to solve the system of discretized governing equations at each time-step, as further explained in Section 5, with superscript ( _n_ ) denoting deferred quantities and superscript ( _n_ + 1) denoting quantities for which the solution is sought implicitly. 

## _3.3. Momentum equations_ 

The momentum equations are solved implicitly in velocity, pressure and the components of the polymer stress tensor. The transient term is discretized, at mesh cell _P_ , as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-19.png)


with the transient derivative of velocity following from Eq. (10) as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0004-21.png)


The advection term is discretized by applying a Newton linearization treating both the flow velocity **u** and the flux _F_ implicitly, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-01.png)


where ◻˜ denotes the flux-limited interpolation presented in Eq. (11). 

The divergence of the stress tensor appearing on the right-hand side of the momentum equations, Eq. (2), is split into three parts, ∇⋅ _**ς**_ = −∇ _p_ + ∇⋅ _**τ**_ s + ∇⋅ _**τ**_ p, each of which is discretized separately. The pressure gradient, ∇ _p_ = ∇⋅( _p_ **I** ), and the divergence of the polymer stress tensor, ∇⋅ _**τ**_ p, are both discretized using linear interpolation of the respective cell values to the mesh faces as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-04.png)


Both _p_ and _**τ**_ p are treated implicitly, as indicated by the superscript ( _n_ + 1). The divergence of the solvent stress tensor _**τ**_ s is discretized as [32] 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-06.png)


where, including the correction for non-orthogonal meshes presented in Eq. (12), 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-08.png)


A harmonic interpolation is applied to interpolate the viscosity values from the cell centers _P_ and _Q_ to the shared face _f_ [72], 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-10.png)


where _µ_ ˘ _f_ is the harmonic interpolated face-centered viscosity, and _ℓf_ is the inverse-distance weighting coefficient with respect to cell _P_ and face _f_ . The solvent stress term is implemented treating all velocity terms implicitly, such that 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-12.png)


Contrary to previous studies on fully coupled algorithms for Newtonian flows, e.g. [32, 33], and viscoelastic flows [17], all velocity gradients of the discretized solvent shear stress in Eq. (23) are solved implicitly. 

As shown in previous studies [36, 67, 73], the momentum sources **S** have to be discretized with the same discretization as the pressure gradients, for the discretized pressure gradient ∇ _pP_ to be able to match the discretized source term **S**[⋆] _P_[.][Following][Bartholomew][et][al.][[][67][],][the discretized momentum source] **[ S]**[⋆][is constructed based on the] untreated source term **S** as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-15.png)


## _3.4. Constitutive model_ 

The constitutive model, Eq. (4), yields six governing equations for the six unique components of the polymer stress tensor, which are solved implicitly for the components of the polymer stress tensor, as well as velocity and pressure. 

In the constitutive model, the first term on the left-hand side of Eq. (4) is treated implicitly as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-19.png)


and the transient term of the constitutive model, as part of the upper-convected derivative of the polymer stress tensor, is discretized in the same manner as transient term of the momentum equations, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0005-21.png)


5 

using the second-order backward Euler scheme presented in Eq. (10). 

The advection term of the constitutive model, Eq. (4), arises from the material derivative of _**τ**_ p and is, consequently, not in conserved form, contrary to the advection term of the momentum equations. In the interest of a discretization that is consistent with the advection of momentum, we reformulate the advection term of _**τ**_ p using the product rule as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-02.png)


such that the advecting velocity can now be applied to define the fluxes through the mesh faces. Similar to the advection term, a Newton linearization is applied, with 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-04.png)


For the considered incompressible flows, the (∇⋅ **u** )-term in Eq. (27) is superfluous from a mathematical viewpoint. Numerically, however, this is only true for a converged result, but may not be the case during the initial nonlinear iterations in each time-step. Including the (∇⋅ **u** )-term, therefore, generally improves the convergence of the solution algorithm. 

Aiming to fully exploit the implicit coupling provided by the fully coupled solution procedure, the two remaining terms of the upper-convected time derivative of the polymer stress tensor and the non-affine response term are linearized and treated implicit using a Newton linearization respectively, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-07.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-08.png)


the quadratic stress term of the Giesekus model is treated implicitly with respect to the polymer stress tensor, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-10.png)


and the strain-rate tensor is treated implicitly with respect to the velocity, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0006-12.png)


Fernandes et al. [17] treated both the additional terms of the upper-convective time derivative, Eq. (29), and the strain-rate tensor, Eq. (32), explicitly. Recently, Fernandes [26] also applied a Newton linearization to the additional terms of the upper-convective time derivative in their fully coupled log-conformation algorithm, treating these terms implicitly in the conformation tensor and the velocity. Pimenta and Alves [41] treated the first term of Eq. (32) implicitly with respect to velocity, while treating the second term explicitly. 

Contrary to the fully coupled algorithm of Fernandes et al. [17], we do not divide the discretized constitutive model by the relaxation time _λ_ before discretization. Hence, the constitutive model is valid for _λ_ ≥ 0. Considering, for example, the upper-convected Maxwell model ( _ψ_ = 1, _µ_ = 0), Eq. (4) reduces to _**τ**_ p = _η_ (∇ **u** + ∇ **u**[T] ) for _λ_ = 0, resulting in a Newtonian flow with shear viscosity _η_ . 

## _3.5. Stress-velocity coupling_ 

The stress-velocity coupling associated with the polymer stress is a central building block of the proposed methodology. The commonly applied method to ensure a robust coupling between the polymer stress and the velocity when a collocated variable arrangement is used, is to introduce two mathematically equivalent diffusion terms with 

6 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-00.png)


<br>


Figure 1: Interpolation stencils of the velocity at face _f_ , with the adjacent cells _P_ and _Q_ , considered for the stress-velocity coupling. 

opposite signs on the right-hand side of the momentum equations, Eq. (2), to yield 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-03.png)


where _η_ ˆ is a weighting factor that is dimensionally equivalent to a dynamic viscosity. As the notation indicates, the additional terms are discretized on different computational stencils. For clarity, we first consider an equidistant Cartesian mesh with _cf_ = 1 and ∆ _sf_ = ∆ _x_ , with the discretization for general non-orthogonal meshes given thereafter in Eqs. (37) and (38). On an equidistant Cartesian mesh, the _small-stencil_ diffusion term is discretized as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-05.png)


and the _large-stencil_ diffusion term as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-07.png)


as illustrated in Figure 1. In the literature, this procedure of adding two diffusion terms with opposite signs is widely ˆ referred to as _both-sides diffusion_ (BSD) [74], typically applied with _η_ = _η_ [16, 17, 26, 75]. Applying the conventionally used centered finite-difference approximations described above, the additional small-stencil and large-stencil diffusion terms yield, using tensor notation and the Einstein summation convention, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-09.png)


which infers that, by taking the divergence of this term in the momentum equations, the two additional diffusion terms introduce numerical diffusion ( _∂u_[4] _i_[/] _[∂x]_[4] _j_[)][with][a][magnitude][proportional][to] _[η]_[ˆ][and][∆] _[x]_[2][[][75][].] ˆ ˘ In line with the discretization of the divergence of the solvent stress described in Eq. (23) and with _η_ = _ηf_ , the small-stencil and large-stencil terms are discretized as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-11.png)


respectively. The contribution of the stress-velocity coupling to the right-hand side of the discretized momentum equations can, thus, be summarized as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0007-13.png)


To fully exploit the implicit coupling afforded by the fully coupled algorithm and to be consistent with the treatment of the strain-rate term in the constitutive model, all velocity contributions in Eq. (39) are treated implicitly. 

7 

## _3.6. Pressure-velocity coupling_ 

To ensure a robust pressure-velocity coupling using the employed collocated variable arrangement, the flux _Ff_ = _ϑf Af_ through face _f_ is defined with an advecting velocity _ϑf_ = **u** _f_ ⋅ **n** _f_ that is evaluated using a momentum-weighted interpolation (MWI) [67], originally introduced by Rhie and Chow [76]. This advecting velocity allows to solve the continuity equation for pressure [32] and prevents pressure-velocity decoupling on the employed collocated variable arrangement [67]. 

The advecting velocity is, based on the unified formulation of the momentum-weighted interpolation proposed by Bartholomew et al. [67], defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0008-03.png)


where _ρ_ ˘ _f_ is the harmonically averaged face density. The weighting factor _d_[ˆ] _f_ defines the strength of the pressurevelocity coupling and is given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0008-05.png)


The coefficients _aP_ and _aQ_ are defined based on the diagonal matrix coefficients of the velocity arising from the advection term, see Eq. (17), the solvent stress term, see Eq. (23), and the small-stencil stress-coupling term, see Eq. (39), of the discretized momentum equations associated with the cells adjacent to face _f_ . For the discretization presented above, the coefficient _aP_ (and, analogously, _aQ_ ) is given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0008-07.png)


where _χ_[′] _f_[= (][1][ −] _[χ][f]_[)][if] _[F][f]_[≥][0][and] _[χ]_[′] _f_[=] _[ χ][f]_[if] _[F][f]_[<][ 0][.][For][the][MWI][to][be][time-step][independent,] _[a][P]_[and] _[a][Q]_[must][not] include the contribution of the transient terms to the diagonal coefficient [67]. 

For an arbitrary unstructured mesh and including a density-weighting of the large-stencil pressure and source term contributions, the discretized and implicitly treated advecting velocity is defined as [67] 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0008-10.png)


The discretized pressure terms together constitute a low-pass filter on the pressure field that prevents pressure-velocity decoupling [67]. Contrary to most previous work on fully coupled algorithms [17, 32, 33], we treat all pressure terms in Eq. (43) implicitly [61]. 

## **4. Front-tracking method** 

The numerical framework proposed in the previous section is complemented by a front-tracking method [39] to track the fluid interface separating to immiscible bulk phases. Since the precise formulation and implementation of the applied interface tracking (or interface capturing) method is not critical for the proposed numerical framework, we only provide a brief overview of the applied front-tracking method and refer the reader to our recent publications [39, 64] for more details. 

In front tracking [45, 46], the fluid interface is represented by a triangulated surface mesh. Each vertex _i_ of this surface mesh is advected in a Lagrangian manner, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0008-15.png)


where **x** _i_ and **u** are the location and (interpolated) velocity of vertex _i_ , respectively. The vertices of the surface mesh can, consequently, also move tangential to the interface, which may lead to vertex clustering and a deteriorating quality of the surface mesh, in turn requiring extensive remeshing of the surface mesh to retain an acceptable mesh quality. In order to address this issue, we apply the _normal-only advection_ (NOA) of the vertices [64], with the 

8 

velocity at the location of the surface mesh vertices defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-01.png)


where **u** ref ( _t_ ) is a spatially invariant reference velocity and **u** ( **x** _i,t_ ) is the interpolated fluid velocity at the location of vertex _i_ . Since the fluid velocity is only known at the cell centers of the fluid mesh, the velocity **u** ( **x** _i,t_ ) at the location of the vertices of the surface mesh is interpolated from the fluid mesh using a Peskin cosine interpolation kernel [77], 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-03.png)


where _L_ denotes all mesh cells in a 2∆ _x_ × 2∆ _x_ × 2∆ _x_ stencil with respect to vertex _i_ , and the weighting kernel is 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-05.png)


We define the reference velocity as the volume-averaged velocity of the body enclosed by the front, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-07.png)


where _P_ denotes all cells of the fluid mesh, and integrate Eq. (44) using a conventional fourth-order Runge-Kutta scheme [64]. 

The indicator function I is reconstructed based on the location of the surface mesh by solving a Poisson equation [78]. The force due to surface tension is computed at each triangle _T_ of the surface mesh using a Frenet-Element algorithm [78] 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-10.png)


where _e_ denotes the edges of triangle _T_ with length _le_ , outward-pointing planar vector **p** _e_ = **n** _e_ × **t** _e_ , normal vector **n** _e_ and tangential vector **t** _e_ . Subsequently, the force due to surface tension computed on the surface mesh is interpolated to the fluid mesh using the Peskin cosine interpolation kernel to define the surface tension source term as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-12.png)


where _T_ are all surface triangles in a 2∆ _x_ × 2∆ _x_ × 2∆ _x_ stencil with respect to cell _P_ . 

The surface mesh is dynamically adapted to ensure a sufficient mesh quality as well as an adequate resolution of the interface, including a parabolic fit vertex repositioning method that reduces shape errors of the interface, as described in detail in our previous work [39]. In addition, we apply a volume correction step [79] to improve volume conservation and treat small undulations of the surface mesh in areas where the interface strongly contracts with the TSUR3D algorithm [80]. 

## **5. Solution procedure** 

Combining the discretization of the individual terms presented in Section 3, we obtain a set of discretized equations governing the considered incompressible and isothermal viscoelastic flows. The discretized continuity equation is given by Eq. (13), the discretized momentum equations are 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0009-17.png)


and the discretized constitutive model is given as 

9 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0010-00.png)


<br>


Figure 2: Flow chart of the solution procedure of the discretized and linearized system of governing equations, where _n_ is the nonlinear iteration counter, Γ = { _u, v, w, p, τ_ p _,xx, τ_ p _,yy, τ_ p _,zz, τ_ p _,xy, τ_ p _,xz, τ_ p _,yz_ } are the solution variables and _Ff_ is the flux through mesh face _f_ (see Section 3.6). The coefficient matrix **A** holds all coefficients for the implicitly sought solution variables Γ[(] _[n]_[+][1][)] of the discretized governing equations and _**ζ**_ is the solution vector. The right-hand side vector **b** holds the deferred contributions of the previous iteration (Γ[(] _[n]_[)] , _ϑ_[(] _f[n]_[)] ) and the contributions of the previous time-levels (Γ[(] _[t]_[−][∆] _[t]_[1][)] , Γ[(] _[t]_[−][∆] _[t]_[1][−][∆] _[t]_[2][)] , _ϑ_[(] _f[t]_[−][∆] _[t]_[1][)] ). 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0010-02.png)


As the notation suggests, each term of the governing equations makes an implicit contribution to at least one of the solution variables Γ = { _p,u,v,w,τ_ p _,xx,τ_ p _,yy,τ_ p _,zz,τ_ p _,xy,τ_ p _,xz,τ_ p _,yz_ }. 

The solution procedure applied to solve the discretized governing equations is illustrated in Figure 2. In each time-step, the interface is advected first and, subsequently, the linearized and discretized governing equations (13), (51) and (52) are solved simultaneously in a single linear system of equations. For a three-dimensional computational 

10 

mesh with _N_ cells, this linear system of equations is given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0011-01.png)


where **A** Γ to **J** _χ_ are the _N_ × _N_ coefficient submatrices of the solution variables Γ associated with the continuity equation ( **A** ), the three momentum equations ( **B** − **D** ) and the six constitutive equations of the polymer stress tensor ( **E** − **J** ). The subvectors _**ζ**_ Γ of length _N_ hold the solution of the implicitly sought variables Γ and the right-hand side vector **b** of length 10 _N_ holds all known contribution from previous nonlinear iterations and time-steps. The solution procedure performs nonlinear iterations in which this system of linearized and discretized governing equations, Eq. (53), is solved using the Block-Jacobi pre-conditioner and the BiCGSTAB solver of the software library PETSc [81, 82] until a pre-defined solver tolerance is satisfied. Subsequently, the deferred quantities are updated and Eq. (53) is solved again. This procedure continues until the conservation error of the nonlinear set of governing conservation laws satisfies a predefined maximum error [32], at which point the solution procedure moves to the next time-step. The Newton linearization of the advection terms in the momentum equations and the constitutive model yields an implicit contribution of the fluxes _Ff_[(] _[n]_[+][1][)] . The flux, thus, introduces an implicit pressure and velocity dependency in all governing equations. Furthermore, the implicit treatment of the polymer stress term and the stress-velocity coupling terms in the momentum equations, alongside the implicit treatment of the upper-convected time derivative of the polymer stress tensor and strain-rate tensor in the constitutive model, provides a strong implicit coupling of the velocity field and the polymer stress tensor. 

## **6. Results** 

Four representative test cases are considered to demonstrate the capabilities of the proposed numerical framework for single-phase and interfacial flows. First, a lid-driven cavity containing a viscoelastic fluid described by the LPTT model is considered in Section 6.1 to assess the basic predictive accuracy of the proposed algorithm. Two-dimensional Taylor vortices are simulated in Section 6.2 to quantify the influence of the stress-velocity coupling on the conservation of kinetic energy. In Section 6.3, a Newtonian droplet in a shear-thinning Giesekus fluid is subjected to a shear flow at different Weissenberg numbers, allowing a direct comparison with the results recently reported by Wang et al. [53] using a state-of-the-art Lattice-Boltzmann method. A bubble rising in a viscoelastic EPTT fluid under the action of gravity is considered in Section 6.4, where we focus particularly on the jump discontinuity in the terminal rise velocity of the bubble and the related negative-wake phenomenon, as studied in detail by Niethammer et al. [28]. 

## _6.1. Lid-driven cavity_ 

A square cavity with edge length _L_ is considered, the top wall of which moves at a constant velocity _U_ , with the shear rate defined as _γ_ ˙ = _U_ / _L_ . Following Yapici [83], we consider an LPTT fluid with _β_ = _µ_ /( _µ_ + _η_ ) = 0 _._ 3, _ϵ_ = 0 _._ 25 and _ξ_ = 0. Different mesh resolutions ranging from 20 × 20 to 160 × 160 cells are considered and the applied time-step ∆ _t_ is defined adaptively to correspond to a Courant number of Co = **u** ∆ _t_ /∆ _x_ ≃ 0 _._ 9. The flow has a Weissenberg number ˙ ˙ of Wi = _γλ_ ∈{1 _,_ 5} and a Reynolds number of Re = _ργL_[2] /( _µ_ + _η_ ) = 10[−][4] . The contours of the velocity magnitude at steady state of both cases, with _U_ = 1 m/s, are shown in Figure 3. 

Figures 4 and 5 show the velocity profiles along both centerlines, as well as the error _ετ_ p _,xx_ in normal polymer stress component _τ_ p _,xx_ at **x** = (0 _._ 9 _L_ 0 _._ 9 _L_ ), for Wi = 1 and Wi = 5, respectively. For both considered Weissenberg numbers, the velocity profiles are in excellent agreement with the results reported by Yapici [83] for the same cases and using the same mesh resolution. The error _ετ_ p _,xx_ in normal polymer stress component _τ_ p _,xx_ converges, as expected, with second order compared to the solution on the finest mesh, if the mesh resolution is sufficiently high. 

## _6.2. Taylor vortices_ 

The evolution of two-dimensional Taylor vortices are simulated to analyze the artificial dissipation of kinetic energy contributed by the stress-velocity coupling. With this test case we were able to demonstrate that the fully coupled algorithm for Newtonian flows that is underpinning the proposed algorithm for viscoelastic flows does not introduce numerical diffusion if central differencing is applied [32], aside from the numerical diffusion associated with the MWI used for the definition of the fluxes, which, however, decays with ∆ _x_[3] . 

11 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0012-00.png)


Figure 3: Velocity contours of the considered LPTT fluid in a lid-driven cavity at steady state, for Wi ∈{1 _,_ 5}, on an equidistant Cartesian mesh with 160 × 160 cells. 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0012-02.png)


<br>


Figure 4: Velocity profiles along the respective centerlines of the lid-driven cavity, obtained on the reference mesh with 160 × 160 cells, and convergence of the error _ετ_ p _,xx_ in normal polymer stress component _τ_ p _,xx_ relative to the reference mesh, for Wi = 1. The results of Yapici [83] are shown for reference. 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0012-04.png)


<br>


Figure 5: Velocity profiles along the respective centerlines of the lid-driven cavity, obtained on the reference mesh with 160 × 160 cells, and covergence of the error _ετ_ p _,xx_ in normal polymer stress component _τ_ p _,xx_ relative to the reference mesh, for Wi = 5. The results of Yapici [83] are shown for reference. 

12 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0013-00.png)


<br>


Figure 6: Results of the Newtonian Taylor vortices at _t_ = 1s for Re = 100, applying _**τ**_ = _**τ**_ s, _**τ**_ = _**τ**_ p with the stress-velocity coupling in the momentum equations defined by Eq. (58), for different parameter sets and _λ_ = 0 For _**τ**_ = _**τ**_ s results obtained with both the central differencing scheme and the CUBISTA scheme are shown, for _**τ**_ = _**τ**_ p only results using the central differencing scheme are shown. Top row: Kinetic energy _E_ kin integrated over the domain as a function of mesh spacing ∆ _x_ , where the analytic kinetic energy is shown by the dashed line. Bottom row: Error _ε_ kin, see Eq. (60), as a function of mesh spacing ∆ _x_ , incurred when using _**τ**_ = _**τ**_ p compared to _**τ**_ = _**τ**_ s, for different parameter sets. 

Following the work of Ham and Iaccarino [84] as well as our previous work [32, 67], the computational domain has the dimensions 2m × 2m and is periodic in all directions, such that no boundary conditions need to be considered. For a Newtonian fluid, the velocity and pressure are given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0013-03.png)


from which the initial conditions for the simulations are readily obtained for _t_ = 0. Integrating the kinetic energy analytically and numerically over the domain Ω yields for a Newtonian fluid with constant density 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0013-05.png)


The fluid occupying the computational domain has a density of _ρ_ = 1kg/m[3] and the time-step applied for all simulations is ∆ _t_ = 2 × 10[3] s. If not stated otherwise, the central differencing scheme is applied for the discretization of the advection terms. 

We consider a Newtonian fluid, such that the momentum equations are given as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0013-08.png)


However, the stress tensor _**τ**_ is now either the solvent stress tensor _**τ**_ s or the polymer stress tensor _**τ**_ p under the assumption of _λ_ = 0, _α_ = 0, and _ξ_ = 0, for which the constitutive model reduces to 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0013-10.png)


In both scenarios, assuming either _**τ**_ = _**τ**_ s or _**τ**_ = _**τ**_ p in the momentum equations, the results should be identical as long as _µ_ = _η_ / _ψ_ . The only difference between the two scenarios is, therefore, the coupling of the polymer stress with the velocity field described in Section 3.5, which is not required for the solvent stress. Please note, the constitutive model is solved for the polymer stress to demonstrate the influence of the stress-velocity coupling. As the weighting 

13 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0014-00.png)


<br>


Figure 7: Schematic illustration of the droplet in shear flow, where the blue color depicts the viscoelastic fluid. 

ˆ coefficient for the stress-velocity coupled we use _η_ = _η_ , following Fernandes et al. [17], as conventionally used in the literature. 

Figure 6 shows the kinetic energy integrated over the domain at _t_ = 1s for Re = 100. Applying _ψ_ = 1 for the polymer stress tensor, all cases are in excellent agreement with each other, converging towards the analytical value given by Eq. (57). This is to be expected for using the polymer stress in conjunction with the stress-velocity coupling and _η_ ˆ = _η_ , because in this case the stress-velocity coupling substitutes, in the momentum equations, the large-stencil diffusion term of the strain-rate tensor of the constitutive model by the corresponding small-stencil diffusion term. The errors incurred when using the polymer stress instead of the solvent stress, defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0014-04.png)


where _E_ kin _,_ s is the kinetic energy obtained using the solvent stress and _E_ kin _,_ p is the kinetic energy obtained using the polymer stress, are numerically negligible. Applying the CUBISTA scheme [69] instead of central differencing to discretise the advection term of the momentum equations also introduces a small amount of numerical diffusion, an error that converges with close to third order under mesh refinement. 

Changing the values of the stress function _ψ_ and the polymer viscosity _η_ concurrently, such that the ratio _η_ / _ψ_ remains unchanged, should yield the same results. However, the stress-velocity coupling imposes a filter on the velocity field that is proportional to _η_ ˆ and ∆ _x_[2] , see Eq. (36). Figure 6 shows the kinetic energy integrated over the domain at _t_ = 1 for Re = 100, where { _ψ_ = 10 _,η_ = 0 _._ 1Pas} and { _ψ_ = 100 _,η_ = 1Pas} for the viscoelastic cases. The polymer stress tensor introduces an error that is, as expected, dependent on the values of _η_ and the mesh spacing ∆ _x_ . The difference between the results obtained using the polymer stress and using the solvent stress decays proportional to ∆ _x_[2] . 

These results demonstrate that the stress-velocity coupling imposes a filter on velocity field and introduces an error that is proportional to ∆ _x_[2] , as stipulated by Eq. (36). Hence, the employed stress-velocity coupling retains the second-order accuracy of the underlying finite-volume scheme as part of the proposed fully coupled algorithm. 

## _6.3. Droplet in shear flow_ 

The capabilities of the proposed fully coupled algorithm with respect to interfacial flows with surface tension is demonstrated with a Newtonian droplet situated in a Giesekus fluid between two infinite parallel plates, subject to a shear flow with shear rate _γ_ ˙ = 2 _U_ / _H_ , as illustrated in Figure 7. Following the recent work of Wang et al. [53], the initially spherical droplet with radius _R_ is placed at the center of the computational domain with dimensions 9 _R_ × 5 _._ 5 _R_ × 4 _R_ , represented with an equidistant Cartesian mesh. The plates are modelled as no-slip walls, whereas periodicity is assigned to all other domain boundaries. The host fluid is characterized by a solvent viscosity ratio of _β_ = _µ_ h/( _µ_ h + _η_ h) = 0 _._ 5, a non-affine parameter of _ξ_ = 0, and a mobility parameter of _α_ = 0 _._ 3, the droplet viscosity ratio is _m_ = _µ_ d/( _µ_ h + _η_ h) = 1 and the surface tension coefficient _σ_ of the fluid interface follows from the considered ˙ capillary number Ca = ( _µ_ h + _η_ h) _γR_ / _σ_ ∈{0 _._ 15 _,_ 0 _._ 25}. The shear flow is in the creeping flow regime, with a Reynolds number of Re = _ρ_ h ˙ _γR_[2] /( _µ_ h + _η_ h) = 0 _._ 1. 

In order to test the mesh convergence of the proposed numerical framework, we consider a droplet with Ca = 0 _._ 15 in viscoelastic shear flow with Wi = _γλ_ ˙ = 1, using different mesh resolutions. Figure 8a shows the evolution of the Taylor deformation parameter, 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0014-11.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0014-12.png)


14 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0015-00.png)


<br>


Figure 8: Evolution of the Taylor deformation parameter _D_ , see Eq. (61), of the droplet in viscoelastic shear flow. (a) Ca = 0 _._ 15 and Wi = 1, obtained with different mesh resolutions; the circles show the reference results of Wang et al. [53]. (b) Ca = 0 _._ 25 using a mesh resolution of ∆ _x_ = _R_ /20, for different Weissenberg numbers Wi = _γλ_ ˙ ; the colored circles show the corresponding reference results of Wang et al. [53]. 

Table 1: Material parameters of P2500 0.8% weight aqueous viscoelastic liquid. 

|_ρl_|[kg m−3]|_η_ [Pas]|_µ_ [Pas]|_λ_ [s]|_σ_ [Nm−1]|_ξ_|_ϵ_|
|---|---|---|---|---|---|---|---|
||1000_._9|1_._483|0_._03|0_._203|0_._07555|0_._12|0_._05|


where **L** and **S** are the semi-major and semi-minor axes of the deformed droplet, as illustrated in Figure 7. Even for a mesh resolution as small as 10 cells per initial bubble radius, the proposed numerical framework provides results of reasonable accuracy. The results converge as the mesh resolution is increased and exhibit an overall very good agreement with the reference results of Wang et al. [53]. 

The evolution of the Taylor deformation parameter _D_ , see Eq. (61), for a droplet with Ca = 0 _._ 25 and different Weissenberg numbers Wi, using a mesh resolution of ∆ _x_ = _R_ /20, is shown in Figure 8b. The results obtained with the proposed numerical framework are in excellent agreement with the results of Wang et al. [53] up to Wi = 50, which is the largest Weissenberg number considered by Wang et al. [53]. Even for a strongly elastic case, with Wi = 10[4] , the proposed numerical framework is seen to produce physically meaningful results, converging robustly without requiring any form of underrelaxation or log-transformation. 

## _6.4. Rising bubble_ 

To further demonstrate the capabilities of the proposed fully coupled algorithm with respect to interfacial flows, a benchmark case of a single gas bubble rising in a viscoelastic liquid is considered. It is a compelling validation case, since experimental studies show a jump in the terminal rise velocity beyond a critical bubble volume and the formation of a negative wake region behind the trailing end of the bubble. The main objective is to achieve quantitative agreement between numerical predictions and experimental measurements and to analyze the flow structures surrounding the bubble to clarify the interplay between rise velocity, bubble deformation, and viscoelastic stress. 

## _6.4.1. Case setup and overview_ 

The experimental results of a rising bubble in an aqueous P2500 0.8% weight viscoelastic liquid of Pilz and Brenn [85] are used to validate the proposed fully coupled algorithm. As illustrated in Figure 9, the polymer solution is shear thinning, hence the exponential Phan-Thien Tanner (EPTT) model is used and the model parameters are determined by fitting the viscosity material function of the used model in steady shear flow to experimental rheology data. For the complete derivation of the viscosity material function the reader is referred to Alves et al. [66]. Liquid density, polymer viscosity, solvent viscosity, surface tension coefficient, and relaxation time are given by Pilz and Brenn [85], while the extensibility coefficient and the slip parameter are determined from the fitting procedure as illustrated in Figure 9. The resultant model parameters are shown in Table 1 following the work of Niethammer et al. [28]. 

The initially spherical bubble with diameter _D_ is placed at the center of a cubic domain of size 20 _D_ × 20 _D_ × 20 _D_ to eliminate the effects of confinement from the domain boundaries, discretized using an equidistant Cartesian mesh. The bubble is rising in positive _y_ -axis due to buoyancy. At the domain boundaries, the velocity field is prescribed using a combination of Dirichlet and Neumann conditions. On the _x_ and _z_ boundaries, impermeable free-slip boundaries are applied: the normal velocity component is set to zero (Dirichlet), while homogeneous Neumann conditions are imposed for the tangential components (zero normal gradients). At the lower _y_ boundary, a no-slip wall is enforced 

15 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0016-00.png)


<br>


Figure 9: Polymer viscosity as a function of shear rate of the aqueous 0.8% weight P2500 solution of Pilz and Brenn [85] with fitted data for the EPTT model. 

by prescribing all velocity components to zero (Dirichlet). At the upper _y_ boundary, an outlet condition is used by imposing homogeneous Neumann conditions for all velocity components (zero normal gradients), while the pressure is fixed to a reference value (Dirichlet) to provide a well-defined pressure level. The pressure satisfies homogeneous Neumann conditions on all remaining boundaries. Finally, homogeneous Neumann conditions are applied to the polymer-stress components on all domain faces, enforcing zero normal gradients at the boundaries. 

Unlike the droplet in shear flow, where a linear interpolation of the stress function, polymer viscosity, relaxation time and non-affine parameter is applied at the interface, we employ here a smooth but sharpened nonlinear interpolation of the same polymer properties for the rising bubble simulations. The rise-velocity jump and the development of a negative wake are governed by the localization and intensity of viscoelastic stresses in a thin layer around the bubble. In particular, detailed analyses relate the regime change to polymer stretching along the bubble contour and to where the stored elastic energy is released relative to the bubble equator [86]. The present nonlinear interpolation confines intermediate properties to a narrow interfacial band while remaining continuous, thereby better preserving the stress distribution near the interface and improving agreement with the reference results. The polymer properties for the rising bubble simulations are defined as 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0016-04.png)


where _ω_ is a weighting factor that sets the location of the transition in terms of the indicator field, _Ic_ ∈(0 _,_ 1) denotes the indicator value used to define the effective interface within the numerically smeared interfacial region, and _n_ > 0 is a sharpness exponent controlling the steepness of the transition (larger _n_ yields a more abrupt switch between phases, while smaller _n_ produces a smoother variation). In the present work, we use _Ic_ = 0 _._ 47 and _n_ = 20 for all simulated volumes, as these values provided the sharpest transition and the best agreement with our reference results. 

A mesh and time-step convergence study was performed to assess the sensitivity of the numerical results to spatial resolution and temporal discretization. The purpose of this study is to determine an efficient spatial and temporal resolution (mesh cell size and time-step) that still provides reliable predictions of the terminal rise velocity. The bubble rise velocity is defined as the vertical velocity component of the bubble centroid, computed as the indicator weighted volume average of the velocity field. Numerical results for a bubble volume of 30mm[3] are compared across three mesh resolutions and two time-step sizes to assess grid and time-step convergence, as illustrated in Figures 10a and 10b, respectively. Adaptive mesh refinement is applied in the vicinity of the bubble, providing high resolution around the interface while allowing for a coarser mesh in regions farther from the bubble. Figures 10a and 10b show that a resolution of 60 mesh cells per bubble diameter and a time-step of 1×10[−][4] s are sufficient, as further refinement or a smaller time-step yield negligible changes in the results. The time step is selected such that it always satisfies the CFL stability constraint of 0.2. 

## _6.4.2. Results_ 

Building on the numerical settings and material properties established in the previous section, we report the transient rise velocity evolution and the terminal rise velocity as a function of bubble volume, and discuss the 

16 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0017-00.png)


<br>


Figure 10: Transient rise velocity of a 30 mm[3] bubble as a function of time. (a) Grid-convergence study using three meshes with different numbers of mesh cells per bubble diameter at ∆ _t_ = 1 × 10[−][4] s. The spatial resolution is increased by a factor of 1 _._ 5 from one mesh level to the next.(b) Time-step convergence study at fixed spatial resolution of 60 mesh cells per bubble diameter. 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0017-02.png)


<br>


Figure 11: Transient and terminal rise velocity of different bubble volumes as a function of time. (a) Transient rise velocity of five bubble volumes 8, 30, 40, 60, and 70 mm[3] . (b) Terminal rise velocity of five bubble volumes 8, 30, 40, 60, and 70 mm[3] compared to the experiments of Pilz and Brenn [85]. 

associated negative wake appearance over a region downstream the trailing end of the supercritical bubbles. 

Figure 11a shows the transient rise velocity of four bubble volumes as a function of time, including three subcritical bubble volumes, _V_ < _Vc_ (8, 30, and 40 mm[3] ) and two supercritical bubble volumes, _V_ > _Vc_ (60 and 70 mm[3] ). Based on their experimental measurements, Pilz and Brenn [85] reported a critical volume of _Vc_ ≈ 46 mm[3] for the considered fluid combination. In the subcritical regime, the bubbles accelerate rapidly from rest and reach a single local maximum at early times, followed by a gradual deceleration toward a steady terminal value. The approach to the terminal state is monotonic after the first maximum and no secondary acceleration stage is observed. In contrast, the supercritical cases exhibit a distinctly different two-stage transient response. The initial acceleration and first local maximum are similar to the subcritical cases, however the subsequent deceleration continues past the terminal value and leads to a local minimum. After this minimum, the bubble undergoes a second acceleration stage and eventually reaches a terminal velocity that is higher than the first local maximum in the considered cases. This non-monotonic evolution is the characteristic signature of the supercritical regime. Figure 11b reports the terminal rise velocity as a function of bubble volume, comparing the present simulation results with the measurements of Pilz and Brenn [85]. The results capture the jump discontinuity in terminal velocity and show good overall agreement with the experimental data. 

In the supercritical regime, a negative wake develops behind the bubble. In the inertial frame, the vertical liquid velocity in the bubble’s wake reverses direction over a region downstream of the trailing end. Figure 12 shows the wake using vertical velocity contours to identify the onset and spatial growth of the negative wake of the 70mm[3] bubble at two time instances, _t_ = 0 _._ 05s where the onset of the negative wake and at _t_ = 0 _._ 07s where the negative 

17 


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0018-00.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0018-01.png)


![](images/-2026-_fully_coupled_implicit_FVM_viscoelastic.pdf-0018-02.png)


<br>


Figure 12: Vertical velocity component _v_ (left) and velocity vector field (right) for the bubble with a volume of 70 mm[3] at two representative times. At _t_ = 0 _._ 05 s the onset of reverse flow appears downstream of the trailing end, while at _t_ = 0 _._ 07 s a clearer reversed-flow region is established. Vector glyphs are scaled for visualization of direction only. 

wake is more developed and apparent. In agreement with prior numerical studies, the negative wake develops as a dynamical wake structure that grows downstream of the trailing tip as the bubble approaches its supercritical steady state. 

## **7. Conclusions** 

In this work, we present a fully coupled implicit finite-volume framework for incompressible viscoelastic interfacial flows, in which the continuity equation, momentum equations, and an upper-convected Maxwell constitutive model, including limited extensibility and shear-thinning behaviour, are solved simultaneously for pressure, velocity, and the polymer stress tensor. The proposed discretization treats all relevant couplings implicitly, including the stress-velocity coupling and the pressure-velocity coupling, yielding a tightly coupled linear system at each nonlinear iteration in a standard finite-volume framework. In addition, a state-of-the-art front-tracking approach is used to represent the evolving interface and account for surface tension forces. 

The method is assessed using representative single-phase and multiphase benchmarks. For the lid-driven cavity flow with an LPTT fluid, the predicted velocity profiles show excellent agreement with the reference data and stress fields show a second-order mesh convergence when compared against the finest grid. The Taylor vortices 

18 

test case isolates the numerical impact of the stress-velocity coupling by tracking the domain integrated kinetic energy. The direct substitution parameter set recovers the solvent-stress baseline and converges to the analytical value with central differencing, while scaled stress function and polymer viscosity cases show a residual that decays at second order under mesh refinement, confirming that the coupling preserves second-order accuracy of the current finite-volume discretization within the proposed fully coupled algorithm. For a Newtonian droplet sheared in a shear-thinning Giesekus fluid, the predicted transient Taylor parameter deformation agrees closely with the reference results up to the highest reported Weissenberg number, and remains stable and physically consistent even when extended to substantially higher Weissenberg number. This benchmark therefore supports both interfacial accuracy and robustness with increasing elasticity, without requiring underrelaxation or a log-transformation. Finally, for a bubble rising in an EPTT viscoelastic liquid, the simulations clearly distinguish the transient rise dynamics in the subcritical and supercritical regimes. The predicted terminal rise velocity as a function of bubble volume reproduces the experimentally observed jump discontinuity, while the flow fields show the onset and downstream development of a negative wake in the supercritical case. A grid and time step sensitivity study confirms that these rise velocity predictions are numerically well resolved at the chosen resolution. This benchmark is particularly demanding because it combines strong interface deformation, sharp localization of viscoelastic stresses near the interface, and a regime transition that is highly sensitive to stress-flow coupling. Capturing all of these features consistently highlights the robustness and effectiveness of the fully implicit coupled formulation for challenging viscoelastic interfacial dynamics. 

Overall, the four test cases demonstrate that the proposed fully implicit coupled formulation delivers accurate and robust solutions from single phase viscoelastic flows to strongly elastic interfacial dynamics, while maintaining robustness at high Weissenberg numbers without resorting to a log-conformation method. 

## **Data Availability Statement** 

The data that support the findings of this study are reproducible and data is openly available in the repository with DOI 10.5281/zenodo.18547566, available at https://doi.org/10.5281/zenodo.18547566 

## **Declaration of Generative AI and AI-assisted technologies in the writing process** 

The authors used OpenAI’s ChatGPT to assist with editing and proofreading this manuscript. The authors then reviewed and revised the text as necessary and assume full responsibility for the final content of the publication. 

## **Acknowledgements** 

We thank Christian Gorges, Fabien Evrard, and Bruno Blais for fruitful discussions. This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), grant numbers 420239128 and 458610925, and by the Natural Sciences and Engineering Research Council of Canada (NSERC), funding reference number RGPIN-2024-04805. 

## **References** 

- [1] B. Williamson, K. Walters, T. Bates, R. Coy, A. Milton, The viscoelastic properties of multigrade oils and their effect on journal-bearing characteristics, Journal of Non-Newtonian Fluid Mechanics 73 (1997) 115–126. 

- [2] S. Gamaniel, D. Dini, L. Biancofiore, The effect of fluid viscoelasticity in lubricated contacts in the presence of cavitation, Tribology International 160 (2021) 107011. 

- [3] K. Wei, D. Liang, M. Mei, X. Yang, L. Chen, A viscoelastic model of compression and relaxation behaviors in preforming process for carbon fiber fabrics with binder, Composites Part B: Engineering 158 (2019) 1–9. 

- [4] H. Yuk, X. Zhao, A New 3D Printing Strategy by Harnessing Deformation, Instability, and Fracture of Viscoelastic Inks, Advanced Materials 30 (2018) 1704028. 

- [5] A. Z. Nelson, B. Kundukad, W. K. Wong, S. A. Khan, P. S. Doyle, Embedded droplet printing in yield-stress fluids, Proceedings of the National Academy of Sciences 117 (2020) 5671–5679. 

- [6] V. Frumkin, M. Bercovici, Fluidic shaping of optical components, Flow 1 (2021) E2. 

- [7] P.-T. Brun, Fluid-Mediated Fabrication of Complex Assemblies, JACS Au (2022) jacsau.2c00427. 

- [8] S. K. Lai, Y.-Y. Wang, D. Wirtz, J. Hanes, Microand macrorheology of mucus, Advanced Drug Delivery Reviews 61 (2009) 86–100. 

- [9] D. A. Siginer, Stability of Non-Linear Constitutive Formulations for Viscoelastic Fluids, SpringerBriefs in Applied Sciences and Technology, Springer International Publishing, Cham, 2014. 

19 

- [10] F. Dupret, J. Marchal, M. Crochet, On the consequence of discretization errors in the numerical calculation of viscoelastic flow, Journal of Non-Newtonian Fluid Mechanics 18 (1985) 173–186. 

- [11] M. Alves, P. Oliveira, F. Pinho, Numerical Methods for Viscoelastic Fluid Flows, Annual Review of Fluid Mechanics 53 (2021) 509–541. 

- [12] R. Keunings, On the high Weissenberg number problem, Journal of Non-Newtonian Fluid Mechanics 20 (1986) 209–226. 

- [13] T.-P. Tsai, D. S. Malkus, Numerical breakdown at high Weissenberg number in non-Newtonian contraction flows, Rheologica Acta 39 (2000) 62–70. 

- [14] K. Walters, M. F. Webster, The distinctive CFD challenges of computational rheology, International Journal for Numerical Methods in Fluids 43 (2003) 577–596. 

- [15] I. Keshtiban, F. Belblidia, M. Webster, Numerical simulation of compressible viscoelastic liquids, Journal of Non-Newtonian Fluid Mechanics 122 (2004) 131–146. 

- [16] F. Habla, A. Obermeier, O. Hinrichsen, Semi-implicit stress formulation for viscoelastic models: Application to three-dimensional contraction flows, Journal of Non-Newtonian Fluid Mechanics 199 (2013) 70–79. 

- [17] C. Fernandes, V. Vukčević, T. Uroić, R. Simoes, O. Carneiro, H. Jasak, J. Nóbrega, A coupled finite volume flow solver for the solution of incompressible viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 265 (2019) 99–115. 

- [18] R. Fattal, R. Kupferman, Constitutive laws for the matrix-logarithm of the conformation tensor, Journal of Non-Newtonian Fluid Mechanics 123 (2004) 281–285. 

- [19] R. Fattal, R. Kupferman, Time-dependent simulation of viscoelastic flows at high Weissenberg number using the log-conformation representation, Journal of Non-Newtonian Fluid Mechanics 126 (2005) 23–37. 

- [20] F. Becker, K. Rauthmann, L. Pauli, P. Knechtges, An eigenvalue-free implementation of the log-conformation formulation, Journal of Non-Newtonian Fluid Mechanics 322 (2023) 105133. 

- [21] W. Doherty, T. N. Phillips, Z. Xie, The log-conformation formulation for singleand multi-phase axisymmetric viscoelastic flows, Journal of Computational Physics 508 (2024) 113014. 

- [22] M. A. Hulsen, R. Fattal, R. Kupferman, Flow of viscoelastic fluids past a cylinder at high Weissenberg number: Stabilized simulations using matrix logarithms, Journal of Non-Newtonian Fluid Mechanics 127 (2005) 27–39. 

- [23] A. Afonso, P. Oliveira, F. Pinho, M. Alves, The log-conformation tensor approach in the finite-volume method framework, Journal of Non-Newtonian Fluid Mechanics 157 (2009) 55–65. 

- [24] F. Martins, C. Oishi, A. Afonso, M. Alves, A numerical study of the Kernel-conformation transformation for transient viscoelastic fluid flows, Journal of Computational Physics 302 (2015) 653–673. 

- [25] M. Niethammer, H. Marschall, C. Kunkelmann, D. Bothe, A numerical stabilization framework for viscoelastic fluid flow using the finite volume method on general unstructured meshes, International Journal for Numerical Methods in Fluids 86 (2018) 131–166. 

- [26] C. Fernandes, A Fully Implicit Log-Conformation Tensor Coupled Algorithm for the Solution of Incompressible Non-Isothermal Viscoelastic Flows, Polymers 14 (2022) 4099. 

- [27] M. Tomé, A. Castelo, A. Afonso, M. Alves, F. Pinho, Application of the log-conformation tensor to threedimensional time-dependent free surface flows, Journal of Non-Newtonian Fluid Mechanics 175–176 (2012) 44–54. 

- [28] M. Niethammer, G. Brenn, H. Marschall, D. Bothe, An extended volume of fluid method and its application to single bubbles rising in a viscoelastic liquid, Journal of Computational Physics 387 (2019) 326–355. 

- [29] C. Fernandes, S. Faroughi, O. Carneiro, J. M. Nóbrega, G. McKinley, Fully-resolved simulations of particle-laden viscoelastic fluids using an immersed boundary method, Journal of Non-Newtonian Fluid Mechanics 266 (2019) 80–94. 

- [30] H. U. Naseer, Z. Ahmed, D. Izbassarov, M. Muradoglu, Dynamics and interactions of parallel bubbles rising in a viscoelastic fluid under buoyancy, Journal of Non-Newtonian Fluid Mechanics (2023) 105000. 

- [31] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130. 

20 

- [32] F. Denner, F. Evrard, B. van Wachem, Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds, Journal of Computational Physics 409 (2020) 109348. 

- [33] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, Journal of Computational Physics 228 (2009) 180–201. 

- [34] M. Darwish, F. Moukalled, A fully coupled Navier-Stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444. 

- [35] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165. 

- [36] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. 

- [37] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234. 

- [38] F. Denner, B. van Wachem, A Unified Algorithm for Interfacial Flows with Incompressible and Compressible Fluids, in: D. Zeidan, L. T. Zhang, E. G. Da Silva, J. Merker (Eds.), Advances in Fluid Mechanics: Modelling and Simulations, Springer Nature Singapore, Singapore, 2022, pp. 179–208. 

- [39] C. Gorges, F. Evrard, B. van Wachem, F. Denner, Reducing volume and shape errors in front tracking by divergence-preserving velocity interpolation and parabolic fit vertex positioning, Journal of Computational Physics 457 (2022) 111072. 

- [40] M. Darwish, A. Aziz, F. Moukalled, A Coupled Pressure-Based Finite-Volume Solver for Incompressible TwoPhase Flow, Numerical Heat Transfer, Part B: Fundamentals 67 (2015) 47–74. 

- [41] F. Pimenta, M. Alves, A coupled finite-volume solver for numerical simulation of electrically-driven flows, Computers & Fluids 193 (2019) 104279. 

- [42] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–225. 

- [43] S. Osher, J. A. Sethian, Fronts Propagating with Curvature-Dependent Speed: Algorithms based on the Hamilton-Jacobi Formulation, Journal of Computational Physics 79 (1988) 12–49. 

- [44] M. Sussman, P. Smereka, S. Osher, A Level Set Approach for Computing Solutions to Incompressible Two-Phase Flow, Journal of Computational Physics 114 (1994) 146–159. 

- [45] S. Unverdi, G. Tryggvason, A Front-Tracking Method for Viscous, Incompressible, Multi-fluid Flows, Journal of Computational Physics 100 (1992) 25–37. 

- [46] G. Tryggvason, B. Bunner, A. Esmaeeli, D. Juric, N. Al-Rawahi, W. Tauber, J. Han, S. Nas, Y. Jan, A front-tracking method for the computations of multiphase flow, Journal of Computational Physics 169 (2001) 708–759. 

- [47] F. Habla, H. Marschall, O. Hinrichsen, L. Dietsche, H. Jasak, J. L. Favero, Numerical simulation of viscoelastic two-phase flows using openFOAM®, Chemical Engineering Science 66 (2011) 5487–5496. 

- [48] J. V. Giliberto, O. Desjardins, A sharp computational method for simulating multiphase viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 348 (2026) 105559. 

- [49] SB. Pillapakkam, P. Singh, A level-set method for computing solutions to viscoelastic two-phase flow, Journal of Computational Physics 174 (2001) 552–578. 

- [50] K. K. Kabanemi, J.-P. Marcotte, A level set method for simulating wrinkling of extruded viscoelastic sheets, Polymer Engineering & Science 60 (2020) 1662–1675. 

- [51] A. Amani, N. Balcázar, A. Naseri, J. Rigola, A numerical approach for non-Newtonian two-phase flows using a conservative level-set method, Chemical Engineering Journal 385 (2020) 123896. 

- [52] D. Izbassarov, M. Muradoglu, A front-tracking method for computational modeling of viscoelastic two-phase flow systems, Journal of Non-Newtonian Fluid Mechanics 223 (2015) 122–140. 

- [53] D. Wang, N. Wang, H. Liu, Droplet deformation and breakup in shear-thinning viscoelastic fluid under simple shear flow, Journal of Rheology 66 (2022) 585–603. 

21 

- [54] P. Yue, J. J. Feng, C. Liu, J. Shen, A diffuse-interface method for simulating two-phase flows of complex fluids, Journal of Fluid Mechanics 515 (2004) 293–317. 

- [55] P. Yue, C. Zhou, J. J. Feng, C. F. Ollivier-Gooch, H. H. Hu, Phase-field simulations of interfacial dynamics in viscoelastic fluids using finite elements with adaptive meshing, Journal of Computational Physics 219 (2006) 47–67. 

- [56] M. Rodriguez, E. Johnsen, A high-order accurate five-equations compressible multiphase approach for viscoelastic fluids and solids with relaxation and elasticity, Journal of Computational Physics 379 (2019) 70–90. 

- [57] K. Zografos, A. M. Afonso, R. J. Poole, M. S. N. Oliveira, A viscoelastic two-phase solver using a phase-field approach, Journal of Non-Newtonian Fluid Mechanics 284 (2020). 

- [58] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, Journal of Computational Physics 285 (2015) 24–40. 

- [59] S. Popinet, Numerical models of surface tension, Annual Review of Fluid Mechanics 50 (2018) 49–75. 

- [60] F. Denner, F. Evrard, B. van Wachem, Breaching the capillary time-step constraint using a coupled VOF method with implicit surface tension, Journal of Computational Physics 459 (2022) 111128. 

- [61] R. Janodet, B. van Wachem, F. Denner, A fully-coupled algorithm with implicit surface tension treatment for interfacial flows with large density ratios, Journal of Computational Physics 520 (2025) 113520. 

- [62] N. Phan-Thien, R. I. Tanner, A new constitutive equation derived from network theory, Journal of NonNewtonian Fluid Mechanics 2 (1977) 353–365. 

- [63] H. Giesekus, A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility, Journal of Non-Newtonian Fluid Mechanics 11 (1982) 69–109. 

- [64] C. Gorges, A. Hodžić, F. Evrard, B. van Wachem, C. M. Velte, F. Denner, Efficient reduction of vertex clustering using front tracking with surface normal propagation restriction, Journal of Computational Physics 491 (2023) 112406. 

- [65] J. H. Snoeijer, A. Pandey, M. A. Herrada, J. Eggers, The relationship between viscoelasticity and elasticity, Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 476 (2020) 20200419. 

- [66] M. A. Alves, F. T. Pinho, P. J. Oliveira, Study of steady pipe and channel flows of a single-mode Phan-Thien– Tanner fluid, Journal of Non-Newtonian Fluid Mechanics 101 (2001) 55–76. 

- [67] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentumweighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177– 208. 

- [68] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. 

- [69] M. A. Alves, P. J. Oliveira, F. T. Pinho, A convergent and universally bounded interpolation scheme for the treatment of advection, International Journal for Numerical Methods in Fluids 41 (2003) 47–75. 

- [70] I. Demirdžić, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255. 

- [71] S. Mathur, J. Murthy, A pressure-based method for unstructured meshes, Numerical Heat Transfer Part B Fundamentals 31 (1997) 195–215. 

- [72] J. Ferziger, Interfacial transfer in Tryggvason’s method, International Journal for Numerical Methods in Fluids 41 (2003) 551–560. 

- [73] J. Mencinger, I. Zun, On the finite volume discretization of discontinuous body force field on collocated grid: Application to VOF method, Journal of Computational Physics 221 (2007) 524–538. 

- [74] R. Guénette, M. Fortin, A new mixed finite element method for computing viscoelastic flows, Journal of Non-Newtonian Fluid Mechanics 60 (1995) 27–52. 

- [75] F. Pimenta, M. Alves, Stabilization of an open-source finite-volume solver for viscoelastic fluid flows, Journal of Non-Newtonian Fluid Mechanics 239 (2017) 85–104. 

22 

- [76] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. 

- [77] C. S. Peskin, Numerical analysis of blood flow in the heart, Journal of Computational Physics 25 (1977) 220–252. 

- [78] G. Tryggvason, R. Scardovelli, S. Zaleski, Direct Numerical Simulations of Gas-Liquid Multiphase Flows, Cambridge University Press, Cambridge ; New York, 2011. 

- [79] M. R. Pivello, A Fully Adaptive Front-Tracking Method for the Simulation of 3D Two-Phase Flows, Ph.D. thesis, University of Uberlandia, Uberlandia, 2012. 

- [80] F. de Sousa, N. Mangiavacchi, L. Nonato, A. Castelo, M. Tomé, V. Ferreira, J. Cuminato, S. McKee, A front-tracking/front-capturing method for the simulation of 3D multi-fluid flows with free surfaces, Journal of Computational Physics 198 (2004) 469–499. 

- [81] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. 

- [82] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. 

- [83] K. Yapici, A comparison study on high-order bounded schemes: Flow of PTT-linear fluid in a lid-driven square cavity, Korea-Australia Rheology Journal 24 (2012) 11–21. 

- [84] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annual Research Briefs, Center for Turbulence (2004) 3–14. 

- [85] C. Pilz, G. Brenn, On the critical bubble volume at the rise velocity jump discontinuity in viscoelastic liquids, Journal of Non-Newtonian Fluid Mechanics 145 (2007) 124–138. 

- [86] D. Bothe, Sharp-interface continuum thermodynamics of multicomponent fluid systems with interfacial mass, International Journal of Engineering Science 179 (2022) 103731. 

23 

