
# A fully-coupled algorithm with implicit surface tension treatment for interfacial flows with large density ratios


### Romain Janodeta, Berend van Wachema and Fabian Dennerb,∗

aChair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany bDepartment of Mechanical Engineering, Polytechnique Montréal, 2500 chemin de Polytechnique, Montréal, H3T 1J4, Québec, Canada

A R T I C L E I N F O

Keywords: Multiphase flows Surface tension Capillary time-step constraint Coupled algorithm Volume-of-fluid method THINC method

A B S T R A C T

The stability of most surface-tension-driven interfacial flow simulations is governed by the capillary time-step constraint. This concerns particularly small-scale flows and, more generally, highly-resolved liquid-gas simulations with moderate inertia. To date, the majority of interfacialflow simulations are performed using an explicit surface-tension treatment, which restrains the performance of such simulations. Recently, an implicit treatment of surface tension able to breach the capillary time-step constraint using the volume-of-fluid (VOF) method was proposed, based on a fully-coupled pressure-based finite-volume algorithm. To this end, the interfaceadvection equation is incorporated implicitly into the linear flow solver, resulting in a tight coupling between all implicit solution variables (colour function, pressure, velocity). However, this algorithm is limited to uniform density and viscosity fields. Here, we present a fullycoupled algorithm for interfacial flows with implicit surface tension applicable to interfacial flows with large density and viscosity ratios. This is achieved by solving the continuity and momentum equations in conservative form, whereby the density is treated implicitly with respect to the colour function, and the advection term of the interface-advection equation is discretised using the THINC/QQ algebraic VOF scheme, yielding a consistent discretisation of the advective terms. This new algorithm is tested by considering representative surfacetension-dominated interfacial flows, including the Laplace equilibrium of a stationary droplet and the three-dimensional Rayleigh-Plateau instability of a liquid filament. The presented results demonstrate that interfacial flows with large density and viscosity ratios can be simulated and energy conservation is ensured, even with a time step larger than the capillary time-step constraint, provided that other time-step restrictions are satisfied.


### 1. Introduction

The maximum stable numerical time step Δ𝑡which is possible for the simulation of surface-tension-driven interfacial flow simulations is determined by various stability criteria, each one being driven by a physical phenomenon and associated with a scaling with respect to the mesh spacing Δ𝑥. For an incompressible flow, the time-step constraint associated with inertia is determined from the well-known Courant-Friedrichs-Lewy (CFL) condition [1],


$$
Δ𝑡CFL = CFL Δ𝑥
$$


$$
∝Δ𝑥,
(1)
$$

where 𝐮is the fluid velocity vector and CFL is the CFL number, the time-step constraint associated with diffusion (mechanical, thermal, chemical) is determined by the Fourier number Fo [2],


$$
Δ𝑡Fo = Fo Δ𝑥2
$$


$$
∝Δ𝑥2,
(2)
$$

where 𝑎is the relevant diffusivity, and the time-step constraint associated with surface tension is determined, as originally formulated by Brackbill et al. [3], by the so-called capillary time-step constraint,


$$
Δ𝑡𝜎=
$$


$$
√
$$

𝜌A + 𝜌B


$$
Δ𝑥3 ∝Δ𝑥3∕2,
(3)
$$

∗Corresponding author

fabian.denner@polymtl.ca (F. Denner) ORCID(s): 0000-0001-5459-7891 (R. Janodet); 0000-0002-5399-4075 (B. van Wachem); 0000-0001-5812-061X (F. Denner)

R. Janodet et al.: Preprint submitted to Elsevier Page 1 of 25


# arXiv:2410.17757v1  [physics.flu-dyn]  23 Oct 2024

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

where 𝜌A and 𝜌B are the densities of the two interacting fluids A and B, respectively, and 𝜎is the surface tension coefficient. This criterion is in fact a CFL condition associated with the phase velocity 𝑐𝜎of the smallest unambiguously resolved capillary waves: Δ𝑡𝜎∝Δ𝑥∕𝑐𝜎. Denner and van Wachem [4] revisited the capillary time-step constraint using both numerical and signal-processing approaches, and found a slightly less restrictive formulation,


$$
Δ𝑡𝜎=
$$


$$
√
$$

𝜌A + 𝜌B


$$
Δ𝑥3 ∝Δ𝑥3∕2,
(4)
$$

compared to the original formulation given in Eq. (3). Although the diffusion time-step constraint shows the strongest scaling with respect to the mesh spacing, contemporary numerical algorithms for incompressible flows solve diffusion terms implicitly, thereby eliminating the diffusion time-step constraint. Consequently, the capillary time-step constraint is generally the dominant time-step constraint for interfacial flows at small scales [4–6]. Since Brackbill’s hypothesis that an implicit discretisation of surface tension should allow breaching or removing the capillary time-step constraint, the community has worked towards this goal. The first studies that presented numerical methods that are able to breach the capillary time-step constraint include the boundary integral method with an implicit surface tension treatment for two-dimensional irrotational incompressible flows of Hou et al. [7] and the works of Bänsch [8], Hysing [9] and Raessi et al. [10] on a semi-implicit surface tension formulation, which includes the interface location at the new time instance implicitly. With the latter, stable results for time steps larger than the capillary time-step constraint were obtained, but the solution is not stable for arbitrarily large time steps. As demonstrated by Denner et al. [11], the method of Hysing [9] and Raessi et al. [10] is equivalent to the addition of a surface viscosity that attenuates fast capillary waves. Sussman and Ohta [12] proposed a method that works in a similar manner as the scheme of Hysing [9], by adding surface damping acting as a low-pass filter [6]. Zheng et al. [13] proposed an implicit Lagrangian method that is not bound by the capillary time-step constraint and which, in contrast to the majority of methods addressing the capillary time-step constraint that express surface tension as a volumetric source term [8–10, 14], retains a sharp representation of the interface.

Recently, Denner et al. [14] proposed a fully-coupled algorithm with an implicit treatment of surface tension that is able to breach, without numerical artifacts, the capillary time-step constraint. This was achieved by employing a Newton linearisation of the surface-tension source term with the continuum surface force (CSF) approach [3], and expressing the implicit interface curvature and colour-function gradients with respect to the implicit colour function. This methodology provides a tight coupling between all implicit solution variables (colour function, pressure, velocity), by applying an algebraic volume-of-fluid (VOF) method for the discretisation of the interface-advection equation and solving the discretised interface-advection equation together with the discretised continuity and momentum equations simultaneously as an implicitly coupled system of equations. However, although the study of Denner et al. [14] demonstrates that the capillary time-step constraint can be breached by treating surface tension implicitly, their algorithm is only able to simulate interfacial flows with unit density and viscosity ratios, and the interface advection is discretised by the compressive interface-capturing scheme for arbitrary meshes (CICSAM) [15], which is known to require very small CFL numbers to retain a sharp interface [16]. In this study, a fully-coupled algorithm for interfacial flows with implicit surface tension is presented that is able to simulate interfacial flows with realistic density and viscosity ratios, and breach the capillary time-step constraint. In order to enable this algorithm for flows with realistic density and viscosity ratios, the continuity and momentum equations are discretised in conservative form, with a consistent discretisation of the advective terms in all governing equations and where density is treated implicitly with respect to the colour function in the transient terms. Moreover, in order to mitigate the CFL time-step constraint associated with the interface transport, the THINC/QQ scheme [17, 18] is applied to discretise the advection term of the interface-advection equation, as it generally can handle larger time steps than the CICSAM scheme, while retaining a sharp fluid interface. To test and scrutinize this new algorithm, twoand three-dimensional interfacial flows with surface tension and large density ratios are considered, ranging from the two-dimensional Laplace equilibrium to a three-dimensional Rayleigh-Plateau instability. The governing equations are introduced in Section 2 and the proposed numerical framework is presented in Section 3. Section 4 briefly revisits the less stringent time-step constraint associated with surface tension encountered with the proposed algorithm. The results of four representative test cases are presented and discussed in Section 5 and the article is concluded in Section 6.

R. Janodet et al.: Preprint submitted to Elsevier Page 2 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


### 2. Governing equations

The governing equations describing immiscible interfacial flows with surface tension in the one-fluid formulation are the continuity equation,

𝜕𝜌


$$
𝜕𝑡+ 𝛁⋅(𝜌𝐮) = 0,
(5)
$$

where 𝐮is the velocity vector and 𝜌is the density, and the momentum equations,

𝜕(𝜌𝐮)


$$
+ 𝛁⋅(𝜌𝐮⊗𝐮) = −𝛁𝑝+ 𝛁⋅𝝉+ 𝐒𝜎,
(6)
$$

where 𝑝is the pressure, 𝝉is the viscous stress tensor, and 𝐒𝜎is the volumetric source term representing surface tension. Since the aim of this work is to simulate incompressible interfacial flows with large density ratios, the conservative forms of both the continuity and momentum equations are considered [19]. For an isotropic Newtonian fluid, the viscous stress tensor 𝝉is defined as

𝝉= 𝜇(𝛁𝐮+ 𝛁𝐮𝑇) , (7)

with 𝜇the dynamic viscosity. The surface-tension source term 𝐒𝜎acts at the fluid interface between two immiscible liquids and is defined as [20]

𝐒𝜎= 𝜎𝜅𝐧S𝛿S, (8)

where 𝜎is the constant surface-tension coefficient, 𝜅is the interface curvature, 𝐧S is the interface normal vector, and 𝛿S is the Dirac delta function associated with the interface. A volume-of-fluid (VOF) method [21] is employed to model the transport and interaction of two immiscible fluids. The indicator function 𝐻, used here to distinguish two immiscible fluid phases, is defined at position 𝐱as

𝐻(𝐱) =

{ 1 if 𝐱in in fluid A 0 if 𝐱is in fluid B (9)

where fluids A and B may represent any immiscible combination of fluids, and advected with the flow,

𝐷𝐻

𝐷𝑡= 𝜕𝐻


$$
𝜕𝑡+ 𝐮⋅𝛁𝐻= 𝜕𝐻
$$


$$
𝜕𝑡+ 𝛁⋅(𝐻𝐮) −𝐻𝛁⋅𝐮= 0.
(10)
$$

Since the sharp Heaviside function 𝐻cannot be convected numerically on a grid, the colour function 𝜓, defined as the control-volume average of 𝐻as

𝜓= 1


$$
𝑉∫𝑉
(11)
$$

is convected instead. By integrating Eq. (10) over the control volume 𝑉with boundary 𝜕𝑉, Eq. (10) becomes the advection equation for the colour function


$$
∫𝑉
$$

𝜕𝜓


$$
𝜕𝑡d𝑉+ ∫𝜕𝑉
(𝐻𝐮) ⋅𝐝𝑨−∫𝑉
(𝐻𝛁⋅𝐮) d𝑉= 0.
(12)
$$

The fluid properties are defined over the entire computational domain based on the individual properties of the two fluids, denoted with subscripts A and B, respectively, and the colour function 𝜓as

𝜌(𝐱) = 𝜌B + 𝜓(𝐱) (𝜌A −𝜌B) (13)

𝜇(𝐱) = 𝜇B + 𝜓(𝐱) (𝜇A −𝜇B). (14)

such that 𝜌and 𝜇are consistent with the interface position.

R. Janodet et al.: Preprint submitted to Elsevier Page 3 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


### 3. Coupled numerical framework

The proposed fully-coupled algorithm for interfacial flows solves the discretised and linearised governing equations presented in Section 2 simultaneously and implicitly coupled in a single system of linear equations, ⋅𝝓= 𝐛. To this end, the governing equations are discretised using a second-order finite-volume method with a collocated variable arrangement [22], whereby the interface advection is discretised using the THINC/QQ scheme [17, 18] and the volume flux through the mesh faces is computed using a momentum-weighted interpolation (MWI) [23] and treated implicitly.

3.1. Implicit surface-tension source term With the colour function 𝜓, the volumetric source term representing surface tension, Eq. (8), is described by the CSF model [3], defined for cell 𝑃as

𝐒𝜎,𝑃= 𝜎𝜅𝑃𝛁𝜓𝑃, (15)

where the surface tension coefficient 𝜎is assumed to be constant. The gradient of the colour function 𝜓is discretised using the Gauss theorem, given for a cell 𝑃as


$$
𝛁𝜓𝑃≈1
$$

𝑉𝑃


$$
∑
$$

𝑓

𝜓𝑓𝐧𝑓𝐴𝑓, (16)

where □denotes a linear interpolation, 𝑓denotes the faces adjacent to cell 𝑃, and 𝐧𝑓and 𝐴𝑓are the outward pointing unit normal vector and the area of face 𝑓, respectively. In order to treat the source term 𝐒𝜎implicitly, a Newton linearisation is applied to the colour function gradients and the interface curvature, such that [14]

𝐒(𝑛+1) 𝜎,𝑃 ≈𝜎

𝑉𝑃

(

𝜅(𝑛) 𝑃 ∑

𝑓

𝜓(𝑛+1) 𝑓 𝐧𝑓𝐴𝑓+ 𝜅(𝑛+1) 𝑃 ∑

𝑓


$$
𝜓(𝑛) 𝑓𝐧𝑓𝐴𝑓−𝜅(𝑛) 𝑃 ∑
$$

𝑓

𝜓(𝑛) 𝑓𝐧𝑓𝐴𝑓

)


$$
, (17)
$$

where 𝑛is the nonlinear iteration counter, with superscript (𝑛) denoting deferred quantities and superscript (𝑛+ 1) denoting an implicitly solved quantity. Following Denner et al. [14], an implicit formulation of the height-function (HF) method [24] is applied to compute the interface curvature. Using the HF method and assuming the 𝑧-component of the interface normal vector is dominant, the interface curvature is given as


$$
−ℎ𝑥𝑥
1 + ℎ2
−ℎ𝑦𝑦
(1 + ℎ2
) + 2ℎ𝑥ℎ𝑦ℎ𝑥𝑦
ℎ2
𝑥+ ℎ2
)3∕2
3∕2 ,
(18)
$$

where ℎ{𝑥,𝑦,𝑥𝑥,𝑦𝑦,𝑥𝑦} denotes the first and second partial derivatives of the liquid heights computed along the 𝑧-direction. By permuting the indices, the curvature can be computed in the same way for cases in which either the 𝑥or 𝑦component of the interface normal vector is the dominant component. Applying a Newton linearisation to Eq. (18), the implicit interface curvature is given as

𝜅(𝑛+1) 𝑃 = 𝜅(𝑛) 𝑃+ 1

(𝑛) 𝑃


$$
∑
$$


$$
𝑁∈(𝑃)
$$


$$
[( 𝜓(𝑛+1) 𝑁 −𝜓(𝑛) 𝑁
$$

) ((𝜕𝑃

𝜕𝜓𝑁


$$
)(𝑛) −3
$$

2𝜅(𝑛) 𝑃


$$
√
$$

(𝑛) 𝑃

(𝜕𝑃

𝜕𝜓𝑁

)(𝑛))]


$$
, (19)
$$

which is applied on a stencil with 3 × 3 × 𝑁ℎcells, where 𝑁ℎis an odd number between 5 and 9. A detailed derivation of this implicit height-function method can be found in the work of Denner et al. [14].

3.2. Discretisation of convective terms In the proposed finite-volume framework, all advective terms take the form ∑

𝑓 (𝜙) 𝑓 = ∑

𝑓 ̃𝜙(𝑛) 𝑓𝐹(𝑛+1) 𝑓 , (20)

where ̃𝜙(𝑛) 𝑓 is the deferred face-based transported quantity 𝜙(𝜓, 𝜌, 𝜌𝐮) at nonlinear iteration 𝑛, and 𝐹(𝑛+1) 𝑓 is the implicitly solved volume flux, calculated using the momentum-weighted interpolation (MWI) presented in Section 3.2.1. Thus, there are three advective fluxes (𝜙) 𝑓 to compute, which should be all consistent [25, 26]:

R. Janodet et al.: Preprint submitted to Elsevier Page 4 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

• Colour function: (𝜓) 𝑓 = ̃𝜓(𝑛) 𝑓𝐹(𝑛+1) 𝑓 , with ̃𝜓(𝑛) 𝑓 constructed using the THINC/QQ scheme [17, 18], described in Section 3.2.2.

• Density: (𝜌) 𝑓 = ̃𝜌(𝑛) 𝑓𝐹(𝑛+1) 𝑓 = 𝜌A(𝜓) 𝑓 + 𝜌B(1−𝜓) 𝑓 , which depends on the flux of the colour function for

consistency, ̃𝜌(𝑛) 𝑓= ̃𝜌(𝑛) 𝑓(̃𝜓(𝑛) 𝑓), as further discussed in Section 3.2.3.

• Momentum: (𝜌𝐮) 𝑓 = ̃ (𝜌𝐮) (𝑛) 𝑓𝐹(𝑛+1) 𝑓 = ̃𝜌(𝑛) 𝑓̃𝐮(𝑛) 𝑓𝐹(𝑛+1) 𝑓 = (𝜌) 𝑓̃𝐮(𝑛) 𝑓, which depends on the density flux for consistency, as described in Section 3.2.3.

It should be noted that, contrary to Denner et al. [14] where we applied a Newton linearisation to the convective terms, here we apply a Picard linearisation with the volume flux 𝐹𝑓being treated implicitly. This Picard linearisation is chosen because the fluxes of the THINC/QQ scheme cannot be written implicitly and, to ensure the consistent transport of mass, momentum, and the fluid interface, this Picard linearisation is applied to all convective terms.

3.2.1. Calculation of the volume flux 𝐹(𝑛+1) 𝑓 The volume flux 𝐹𝑓= 𝜗𝑓𝐴𝑓through cell face 𝑓is defined based on an advecting velocity 𝜗𝑓= 𝐮𝑓⋅𝐧𝑓that is discretised using a momentum-weighted interpolation (MWI) [23, 27]. As previously demonstrated [14, 22, 28], in the class of fully-coupled algorithms considered in this study, the volume flux can play a major role in implicitly coupling the governing equations. Contrary to previously presented fully-coupled algorithms for interfacial flows [4, 14, 29–32] that include (partially) deferred volume fluxes, the proposed algorithm only contains fully implicit velocity and pressure terms, and the surface tension terms are made implicit using a Newton linearisation. Including the linearised surface-tension source term as presented in Section 3.1, the implicit advecting velocity is defined by the MWI as [14]

𝜗(𝑛+1) 𝑓 = 𝐮(𝑛+1) 𝑓 ⋅𝐧𝑓−̂𝑑𝑓 ⎡ ⎢ ⎢⎣


$$
𝑝(𝑛+1) 𝑄 −𝑝(𝑛+1) 𝑃 Δ𝑥 − ̆𝜌(𝑛) 𝑓 2
$$

⎛ ⎜ ⎜⎝

𝛁𝑝(𝑛+1) 𝑃 𝜌(𝑛) 𝑃 + 𝛁𝑝(𝑛+1) 𝑄

𝜌(𝑛) 𝑄


$$
⎞ ⎟ ⎟⎠ ⋅𝐧𝑓 ⎤ ⎥ ⎥⎦
$$

+ ̂𝑑𝑓𝜎 ⎡ ⎢ ⎢⎣


$$
𝜅(𝑛) 𝑓 𝜓(𝑛+1) 𝑄 −𝜓(𝑛+1) 𝑃 Δ𝑥 − ̆𝜌(𝑛) 𝑓 2
$$

⎛ ⎜ ⎜⎝ 𝜅(𝑛) 𝑃 𝛁𝜓(𝑛+1) 𝑃 𝜌(𝑛) 𝑃 + 𝜅(𝑛) 𝑄 𝛁𝜓(𝑛+1) 𝑄

𝜌(𝑛) 𝑄


$$
⎞ ⎟ ⎟⎠ ⋅𝐧𝑓 ⎤ ⎥ ⎥⎦
$$

+ ̂𝑑𝑓𝜎 ⎡ ⎢ ⎢⎣


$$
𝜅(𝑛+1) 𝑓 𝜓(𝑛) 𝑄−𝜓(𝑛) 𝑃 Δ𝑥 − ̆𝜌(𝑛) 𝑓 2
$$

⎛ ⎜ ⎜⎝ 𝜅(𝑛+1) 𝑃 𝛁𝜓(𝑛) 𝑃 𝜌(𝑛) 𝑃 + 𝜅(𝑛+1) 𝑄 𝛁𝜓(𝑛) 𝑄

𝜌(𝑛) 𝑄


$$
⎞ ⎟ ⎟⎠ ⋅𝐧𝑓 ⎤ ⎥ ⎥⎦
$$


$$
−̂𝑑𝑓𝜎 ⎡ ⎢ ⎢⎣
$$


$$
𝜅(𝑛) 𝑓 𝜓(𝑛) 𝑄−𝜓(𝑛) 𝑃 Δ𝑥 − ̆𝜌(𝑛) 𝑓 2
$$

⎛ ⎜ ⎜⎝ 𝜅(𝑛) 𝑃 𝛁𝜓(𝑛) 𝑃 𝜌(𝑛) 𝑃 + 𝜅(𝑛) 𝑄 𝛁𝜓(𝑛) 𝑄

𝜌(𝑛) 𝑄


$$
⎞ ⎟ ⎟⎠ ⋅𝐧𝑓 ⎤ ⎥ ⎥⎦
$$


$$
+ ̂𝑑𝑓 ̆𝜌(𝑡−Δ𝑡) 𝑓
$$


$$
Δ𝑡
$$


$$
( 𝜗(𝑡−Δ𝑡) 𝑓 −𝐮(𝑡−Δ𝑡) 𝑓 ⋅𝐧𝑓 )
$$


$$
(21)
$$

where coefficient ̂𝑑𝑓represents the weighting factor of the MWI correction, as derived by Bartholomew et al. [23], and the face density ̆𝜌𝑓is defined by a harmonic average.

1

̆𝜌(𝑛) 𝑓 = 1

2𝜌(𝑛) 𝑃 + 1

2𝜌(𝑛) 𝑄 . (22)

The primary motivation for including the pressure in the computation of the fluxes is to couple pressure and velocity for the employed collocated variable arrangement. To this end, the discretised pressure terms together constitute a low-pass filter on the pressure field that prevents pressure-velocity decoupling [23],


$$
𝛁𝑝𝑓−𝛁𝑝𝑓=
𝑝𝑄−𝑝𝑃
$$


$$
Δ𝑥
−
$$


$$
2 ∝𝜕3𝑝
$$

𝜕𝑥3


$$
|||||𝑓 , (23)
$$

R. Janodet et al.: Preprint submitted to Elsevier Page 5 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

where the overbar denotes a linear interpolation of cell-centred values. In order to ensure a balance between surface tension and pressure, the source term representing surface tension must be incorporated in the momentum-weighted interpolation in the same fashion as the pressure [23, 30],


$$
𝜎𝜅𝑓𝛁𝜓𝑓−𝜎𝜅𝑓𝛁𝜓𝑓= 𝜎𝜅𝑓
𝜓𝑄−𝜓𝑃
$$


$$
Δ𝑥
−𝜎
$$


$$
2 . (24)
$$

Since we wish to treat both the interface curvature and the colour function gradient implicitly with respect to the colour function, we further linearise the surface tension terms with a Newton linearisation,

𝜎𝜅(𝑛+1)𝛁𝜓(𝑛+1) ≈𝜎𝜅(𝑛+1)𝛁𝜓(𝑛) + 𝜎𝜅(𝑛)𝛁𝜓(𝑛+1) −𝜎𝜅(𝑛)𝛁𝜓(𝑛), (25)

similar to Eq. (17). The density weighting applied to both the pressure and surface tension terms in Eq. (21) reduces the errors associated with a discontinuous change in density [23].

The volume flux 𝐹(𝑛+1) 𝑓 = 𝜗(𝑛+1) 𝑓 𝐴𝑓is part of all governing equations and, thus, ensures the strong link between these equations, and prevents pressure-velocity decoupling as a consequence of the applied collocated variable arrangement.

3.2.2. Calculation of the colour function ̃𝜓(𝑛) 𝑓 The THINC/QQ interface-capturing scheme, a state-of-the-art version of the original THINC scheme of Xiao et al. [33, 34], was first introduced by Xie and Xiao [17], and recently revisited by Chen et al. [18]. It relies on the use of a smoothed phase indicator ̃𝐻to reconstruct the colour function 𝜓at face 𝑓. This indicator is defined using a trigonometric function, and reconstructed within each cell 𝑃as

̃𝐻𝑃(𝐗) = 1

2 [1 + tanh(𝛽𝑃(𝑃(𝐗) + 𝑑𝑃))] , (26)

where 𝛽𝑃is a local steepness parameter, computed in the present study from a user-defined scaling factor 𝛽𝑓, which is typically set to 6 [18], and the local cell size Δ𝑠𝑃= 3√

𝑉𝑃:

𝛽𝑃= 𝛽𝑓 Δ𝑠𝑃 . (27)

Thus, the smooth phase indicator ̃𝐻relates to the sharp indicator 𝐻as

lim 𝛽→∞ ̃𝐻= 𝐻, (28)

𝐗are the local cell-wise coordinates, which can be defined for a Cartesian cell 𝑃of dimensions (Δ𝑥𝑃, Δ𝑦𝑃, Δ𝑧𝑃) as

𝐗= ⎛ ⎜ ⎜⎝

𝑋 𝑌 𝑍

⎞ ⎟ ⎟⎠ =

⎛ ⎜ ⎜ ⎜ ⎜ ⎜ ⎜⎝


$$
2(𝑥−𝑥𝑃)
$$


$$
Δ𝑥𝑃
2(𝑦−𝑦𝑃)
$$


$$
Δ𝑦𝑃
2(𝑧−𝑧𝑃)
$$


$$
Δ𝑧𝑃
$$

⎞ ⎟ ⎟ ⎟ ⎟ ⎟ ⎟⎠


$$
, (29)
$$

with 𝐱𝑃= (𝑥𝑃, 𝑦𝑃, 𝑧𝑃) the coordinates of the cell center. Hence, −1 ≤𝑋, 𝑌, 𝑍≤1 and 𝐗0 = (0, 0, 0) corresponds to the center of the computational cell. The quadratic interface polynomial , which contains the local geometric information of the interface (normal vector, curvature), is defined as


$$
𝑃(𝐗) = 𝐶200,𝑃𝑋2+𝐶020,𝑃𝑌2+𝐶002,𝑃𝑍2+𝐶110,𝑃𝑋𝑌+𝐶101,𝑃𝑋𝑍+𝐶011,𝑃𝑌𝑍+𝐶100,𝑃𝑋+𝐶010,𝑃𝑌+𝐶001,𝑃𝑍, (30)
$$

and 𝑑𝑃is the local surface constant, which sets the position of the interface within a cell 𝑃containing the interface, computed by enforcing volume conservation for each cell as

1 𝑉𝑃∫𝑉 ̃𝐻𝑃(𝐗) d𝑉= 𝜓𝑃. (31)

R. Janodet et al.: Preprint submitted to Elsevier Page 6 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

The quadratic polynomial coefficients 𝐶in Eq. (30) embed the local geometric information of the cell-wise parabolic interface, as described in detail by Xie and Xiao [17] and Chen et al. [18]. They are computed in local coordinates for each cell 𝑃, from the colour-function gradient normalised in global coordinates, as


$$
𝐶200,𝑃= XX,𝑃∕2 =
(Δ𝑥𝑃∕2)2)
∕2
$$


$$
𝐶020,𝑃= YY,𝑃∕2 =
(Δ𝑦𝑃∕2)2)
∕2
$$


$$
𝐶002,𝑃= ZZ,𝑃∕2 =
(Δ𝑧𝑃∕2)2)
∕2
$$


$$
𝐶110,𝑃= XY,𝑃= xy,𝑃∕2 (Δ𝑦𝑃∕2)2 + yx,𝑃∕2 (Δ𝑥𝑃∕2)2
$$


$$
𝐶011,𝑃= YZ,𝑃= yz,𝑃∕2 (Δ𝑧𝑃∕2)2 + zy,𝑃∕2 (Δ𝑦𝑃∕2)2
$$


$$
𝐶101,𝑃= XZ,𝑃= xz,𝑃∕2 (Δ𝑧𝑃∕2)2 + zx,𝑃∕2 (Δ𝑥𝑃∕2)2
$$

𝐶100,𝑃= x,𝑃 (Δ𝑥𝑃∕2)

𝐶010,𝑃= y,𝑃 (Δ𝑦𝑃∕2)

𝐶001,𝑃= z,𝑃 (Δ𝑧𝑃∕2) ,


$$
(32)
$$

where = (x, y, z) is the unit normal vector in global coordinates (𝑥, 𝑦, 𝑧),

= 𝛁𝜓

|𝛁𝜓|, (33)

= 𝛁is the Hessian tensor in global coordinates (𝑥, 𝑦, 𝑧), and is the curvature tensor in local coordinates (𝑋, 𝑌, 𝑍). In order to increase the accuracy of the computation of , 5 steps of Laplace smoothing are performed on the colour function before (and only for) the present normalised gradient computation.

The THINC-based procedures are divided into two main steps: (i) algebraic reconstruction of the smooth phase indicator ̃𝐻𝑃(𝐗) for each cell, i.e. computation of and 𝑑, and (ii) calculation of the flux to advect the colour function 𝜓𝑃using ̃𝐻𝑃(𝐗). Contrary to the other THINC schemes, the THINC/QQ procedure uses both a quadratic surface representation defined by Eq. (30), as well as a Gauss-quadrature integration of ̃𝐻𝑃(𝐗) for the calculation of the surface constant 𝑑as defined by Eq. (31) and for the advection flux. The local surface constant 𝑑𝑃is computed by enforcing volume conservation as indicated by Eq. (31), which features the volume integral of the hyperbolic tangent profile ̃𝐻𝑃(𝐗). Since no analytical solution exists to perform the multidimensional integration, a fully multidimensional Gaussian quadrature is employed to approximate the integral, as proposed by Xie and Xiao [17], which greatly simplifies the numerical procedure. Let us define 𝐗𝑞and 𝜔𝑞(𝑞= 1, ..., 𝑄) the quadrature-point local coordinates and associated weights, respectively. In this work, the quadrature points and weights of Xie and Xiao [17] for hexahedral grids (𝑄= 16 points) are used. Eq. (31) is then given as

𝑄 ∑

𝑞=1 𝜔𝑞,𝑃 ( ̃𝐻𝑃 (𝐗𝑞,𝑃 )) =


$$
𝑄 ∑
$$

𝑞=1 𝜔𝑞,𝑃 (1


$$
2 (1 + tanh (𝛽𝑃 (𝑃 (𝐗𝑞,𝑃 ) + 𝑑𝑃 )))) = 𝜓𝑃, (34)
$$

where the weights are defined so that ∑𝑄 𝑞=1 𝜔𝑞,𝑃= 1 [17]. Using the identity

tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 ) + 𝛽𝑃𝑑𝑃 ) = tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 )) + tanh (𝛽𝑃𝑑𝑃 )


$$
1 + tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 )) ⋅tanh (𝛽𝑃𝑑𝑃 ), (35)
$$

Eq. (34) can be rewritten as

𝑄 ∑

𝑞=1 𝜔𝑞,𝑃 tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 )) + tanh (𝛽𝑃𝑑𝑃 )

1 + tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 )) ⋅tanh (𝛽𝑃𝑑𝑃 ) = 2 ( 𝜓𝑃−1

2


$$
) . (36)
$$

Provided the quadratic function (𝐗) is already known using Eq. (30), 𝑑𝑃is the only remaining unknown to solve. Following Xie and Xiao [17], 𝐴𝑞,𝑃= tanh (𝛽𝑃𝑃 (𝐗𝑞,𝑃 )), 𝐷𝑃= tanh (𝛽𝑃𝑑𝑃 ), and 𝑄𝑃= 2 (𝜓𝑃−1∕2), which

R. Janodet et al.: Preprint submitted to Elsevier Page 7 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

allows to write Eq. (36) as

𝑄 ∑

𝑞=1 𝜔𝑞,𝑃 𝐴𝑞,𝑃+ 𝐷𝑃 1 + 𝐴𝑞,𝑃𝐷𝑃 −𝑄𝑃= 0, (37)

which is a rational equation for the modified surface constant 𝐷𝑃∈[−1, 1]. A Newton-Raphson algorithm is employed to find 𝐷𝑃, as described by Kumar et al. [35]. Once the reconstruction of ̃𝐻𝑃(𝐗) has been performed for all interface cells (i.e. cells with 10−8 < 𝜓𝑃< 1−10−8), the associated flux can be computed to advect the colour function using Eq. (12). As mentioned above, this flux involves the quantity ̃𝜓(𝑛) 𝑓, which is computed in the THINC/QQ framework using Gauss-quadrature integration over the cell face 𝑓as [17]

̃𝜓(𝑛) 𝑓 = 1

𝐴𝑓∫𝐴𝑓 ̃𝐻(𝑛) 𝑃,upwd𝐴=


$$
 ∑
$$


$$
𝑞=1 𝜔𝑞̃𝐻(𝑛) 𝑃,upw(𝐗𝑞), (38)
$$

where ̃𝐻(𝑛) 𝑃,upw is the reconstructed smooth phase indicator at nonlinear iteration 𝑛in the upwind cell associated with face 𝑓, defined as

̃𝐻(𝑛) 𝑃,upw =

{ ̃𝐻(𝑛) 𝑃 if 𝜗(𝑛) 𝑓> 0 ̃𝐻(𝑛) 𝑄if 𝜗(𝑛) 𝑓< 0. (39)

For the Gauss-quadrature integration, = 9 Gauss-Legendre points are used in total on the cell-face surface. With the colour function at cell face 𝑓defined by Eq. (38), the colour function flux follows as (𝜓) 𝑓 = ̃𝜓(𝑛) 𝑓𝐹(𝑛+1) 𝑓 .

3.2.3. Calculation of density and momentum fluxes Following Arrufat et al. [26], the density flux is formulated consistently with the colour-function flux as

(𝜌) 𝑓 = 𝜌A(𝜓) 𝑓 + 𝜌B(1−𝜓) 𝑓 = 𝜌Ã𝜓(𝑛) 𝑓𝐹(𝑛+1) 𝑓 + 𝜌B ( 1 −̃𝜓(𝑛) 𝑓


$$
) 𝐹(𝑛+1) 𝑓 . (40)
$$

Rearranging this equation gives

(𝜌) 𝑓 = [ 𝜌B + ̃𝜓(𝑛) 𝑓 (𝜌A −𝜌B )] 𝐹(𝑛+1) 𝑓 = ̃𝜌(𝑛) 𝑓𝐹(𝑛+1) 𝑓 , (41)

where the face density at the nonlinear iteration 𝑛is

̃𝜌(𝑛) 𝑓= 𝜌B + ̃𝜓(𝑛) 𝑓Δ𝜌 (42)

and Δ𝜌= 𝜌A −𝜌B. The momentum flux follows, in a similar manner, based on the density flux as

(𝜌𝐮) 𝑓 = ̃𝜌(𝑛) 𝑓̃𝐮(𝑛) 𝑓𝐹(𝑛+1) 𝑓 = (𝜌) 𝑓̃𝐮(𝑛) 𝑓. (43)

The only quantity that remains to be calculated is the face velocity ̃𝐮(𝑛) 𝑓. To this end, we employ a Favre averaging together with total variation diminishing (TVD) differencing, following the work of Kuhn et al. [36], such that the velocity at face 𝑓is defined as

̃𝐮(𝑛) 𝑓= (𝜌𝐮)(𝑛) 𝑓,TVD

𝜌(𝑛) 𝑓,TVD , (44)

where the TVD quantities 𝜑are calculated using the Minmod flux limiter 𝜉(𝜑) 𝑓,Minmod [37, 38] as

𝜌(𝑛) 𝑓,TVD = 𝜌(𝑛) 𝑈+ 𝜉(𝜌) 𝑓,Minmod


$$
[ 𝜌(𝑛) 𝐷−𝜌(𝑛) 𝑈
$$


$$
] (45)
$$

(𝜌𝐮)(𝑛) 𝑓,TVD = (𝜌𝐮)(𝑛) 𝑈+ 𝜉(𝜌𝐮) 𝑓,Minmod

[ (𝜌𝐮)(𝑛) 𝐷−(𝜌𝐮)(𝑛) 𝑈


$$
] . (46)
$$

The subscripts 𝑈and 𝐷denote the upwind and downwind cells, respectively.

R. Janodet et al.: Preprint submitted to Elsevier Page 8 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

3.3. Discretisation of transient terms A second-order backward Euler time-integration scheme for variable time steps is employed to discretise the transient terms, given for the general fluid variable Ω as [39]


$$
∫𝑉𝑃
$$


$$
𝜕Ω
$$

𝜕𝑡d𝑉≈ [( 1


$$
Δ𝑡+ 1
$$


$$
Δ𝜏
$$


$$
) Ω(𝑛+1) 𝑃 − ( 1
$$


$$
Δ𝑡+
Δ𝑡o
$$


$$
) Ω(𝑡−Δ𝑡) 𝑃 + Δ𝑡 Δ𝑡oΔ𝜏Ω(𝑡−Δ𝜏) 𝑃
$$


$$
] 𝑉𝑃, (47)
$$

where Δ𝑡is the current time step, Δ𝑡o denotes the previous time step and Δ𝜏= Δ𝑡+ Δ𝑡o. The transient term of the advection equation of the colour function, Eq. (12), is readily discretised by Eq. (47) with Ω = 𝜓. Using the cell-based density computed from the cell-based colour function as 𝜌𝑃= 𝜌B+Δ𝜌𝜓𝑃, where Δ𝜌= 𝜌A−𝜌B, the implicit cell-based density is obtained by performing a Newton linearisation as

𝜌(𝑛+1) 𝑃 = 𝜌(𝑛) 𝑃+ ( 𝜓(𝑛+1) 𝑃 −𝜓(𝑛) 𝑃

) 𝜕𝜌

𝜕𝜓= 𝜌(𝑛) 𝑃+ ( 𝜓(𝑛+1) 𝑃 −𝜓(𝑛) 𝑃


$$
Δ𝜌.
(48)
$$

The transient term of the continuity equation is then discretised using Eq. (47) with Ω(𝑛+1) 𝑃 = 𝜌(𝑛+1) 𝑃 given by Eq. (48),

Ω(𝑡−Δ𝑡) 𝑃 = 𝜌(𝑡−Δ𝑡) 𝑃 , and Ω(𝑡−Δ𝜏) 𝑃 = 𝜌(𝑡−Δ𝜏) 𝑃 . The implicit momentum is defined as a product of density and velocity, linearised with a Newton linearisation,

(𝜌𝐮)(𝑛+1) 𝑃 = 𝜌(𝑛+1) 𝑃 𝐮(𝑛) 𝑃+ 𝜌(𝑛) 𝑃𝐮(𝑛+1) 𝑃 −𝜌(𝑛) 𝑃𝐮(𝑛) 𝑃. (49)

Rearranging the terms of this equation and inserting the density definition used for the discretisation of the transient term of the continuity equation yields

(𝜌𝐮)(𝑛+1) 𝑃 = 𝜌(𝑛) 𝑃𝐮(𝑛+1) 𝑃 + ( 𝜓(𝑛+1) 𝑃 −𝜓(𝑛) 𝑃


$$
) Δ𝜌𝐮(𝑛) 𝑃. (50)
$$

The transient term of the momentum equations is then discretised using Eq. (47) with Ω(𝑛+1) 𝑃 = (𝜌𝐮)(𝑛+1) 𝑃 given by

Eq. (50), Ω(𝑡−Δ𝑡) 𝑃 = (𝜌𝐮)(𝑡−Δ𝑡) 𝑃 , and Ω(𝑡−Δ𝜏) 𝑃 = (𝜌𝐮)(𝑡−Δ𝜏) 𝑃 .

3.4. Discretised equation system and solution procedure Applying the discretisation defined in Sections 3.1-3.3, the discretised continuity equation, Eq. (5), for mesh cell 𝑃is given as [( 1


$$
Δ𝑡+ 1
$$


$$
Δ𝜏
$$

) [ 𝜌(𝑛) 𝑃+ ( 𝜓(𝑛+1) 𝑃 −𝜓(𝑛) 𝑃


$$
) Δ𝜌 ] − ( 1
$$


$$
Δ𝑡+
Δ𝑡o
$$


$$
) 𝜌(𝑡−Δ𝑡) 𝑃 + Δ𝑡 Δ𝑡oΔ𝜏𝜌(𝑡−Δ𝜏) 𝑃
$$


$$
] 𝑉𝑃+ ∑
$$


$$
𝑓 (𝜌) 𝑓 = 0, (51)
$$

the discretised momentum equations, Eq. (6), follows as [( 1


$$
Δ𝑡+ 1
$$


$$
Δ𝜏
$$

) [ 𝜌(𝑛) 𝑃𝐮(𝑛+1) 𝑃 + ( 𝜓(𝑛+1) 𝑃 −𝜓(𝑛) 𝑃


$$
) Δ𝜌𝐮(𝑛) 𝑃
$$


$$
] − ( 1
$$


$$
Δ𝑡+
Δ𝑡o
$$


$$
) (𝜌𝐮)(𝑡−Δ𝑡) 𝑃 + Δ𝑡 Δ𝑡oΔ𝜏(𝜌𝐮)(𝑡−Δ𝜏) 𝑃
$$

] 𝑉𝑃

+ ∑

𝑓 (𝜌𝐮) 𝑓 = − ∑

𝑓

𝑝(𝑛+1)


$$
𝑓 𝐧𝑓𝐴𝑓+ ∑
$$

𝑓 𝜇𝑓 ⎛ ⎜ ⎜⎝

𝐮(𝑛+1)


$$
𝑄 −𝐮(𝑛+1)
$$


$$
𝑃 Δ𝑥 + 𝛁𝐮
$$

(𝑛+1) 𝑓 ⋅𝐧𝑓 ⎞ ⎟ ⎟⎠ 𝐴𝑓+ 𝐒(𝑛+1)


$$
𝜎,𝑃𝑉𝑃, (52)
$$

and the discretised advection equation for the colour function, Eq. (12), is given as [( 1


$$
Δ𝑡+ 1
$$


$$
Δ𝜏
$$

) 𝜓(𝑛+1) 𝑃 − ( 1


$$
Δ𝑡+
Δ𝑡o
$$


$$
) 𝜓(𝑡−Δ𝑡) 𝑃 + Δ𝑡 Δ𝑡oΔ𝜏𝜓(𝑡−Δ𝜏) 𝑃
$$


$$
] 𝑉𝑃+ ∑
$$


$$
𝑓 (𝜓) 𝑓 −𝜓(𝑛)
$$


$$
𝑃 ∑
$$

𝑓 𝐹(𝑛+1)


$$
𝑓 = 0, (53)
$$

where □𝑓denotes a linear interpolation of the cell-centred values to face 𝑓. This discretised set of governing equations is solved simultaneously for the pressure 𝑝, the velocity vector 𝐮= (𝑢𝑣𝑤)T and the colour function 𝜓in a single linear equation system, ⋅𝝓= 𝐛, given for a three-dimensional mesh with 𝑁cells as

⎛ ⎜ ⎜ ⎜ ⎜ ⎜⎝

𝑝 cont. 𝑢 cont. 𝑣 cont. 𝑤 cont. 𝜓 cont. 𝑝 𝑥-mom. 𝑢 𝑥-mom. 𝑣 𝑥-mom. 𝑤 𝑥-mom. 𝜓 𝑥-mom. 𝑝 𝑦-mom. 𝑢 𝑦-mom. 𝑣 𝑦-mom. 𝑤 𝑦-mom. 𝜓 𝑦-mom. 𝑝 𝑧-mom. 𝑢 𝑧-mom. 𝑣 𝑧-mom. 𝑤 𝑧-mom. 𝜓 𝑧-mom. 𝑝 VOF 𝑢 VOF 𝑣 VOF 𝑤 VOF 𝜓 VOF

⎞ ⎟ ⎟ ⎟ ⎟ ⎟⎠


$$
⋅
$$

⎛ ⎜ ⎜ ⎜ ⎜⎝

𝝓𝑝

𝝓𝑢

𝝓𝑣

𝝓𝑤

𝝓𝜓

⎞ ⎟ ⎟ ⎟ ⎟⎠

=

⎛ ⎜ ⎜ ⎜ ⎜⎝

𝐛cont. 𝐛𝑥-mom. 𝐛𝑦-mom. 𝐛𝑧-mom. 𝐛VOF

⎞ ⎟ ⎟ ⎟ ⎟⎠


$$
. (54)
$$

R. Janodet et al.: Preprint submitted to Elsevier Page 9 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


$$
Update previous time-levels: 𝜒(𝑡−Δ𝜏) 𝑃 ←𝜒(𝑡−Δ𝑡) 𝑃 𝜒(𝑡−Δ𝑡) 𝑃 ←𝜒(𝑛) 𝑃 𝜗(𝑡−Δ𝑡) 𝑓 ←𝜗(𝑛) 𝑓
$$

Gather coefficients and assemble and 𝐛


$$
Solve ⋅𝝓= 𝐛
$$

Update deferred quantities: 𝜒(𝑛) 𝑃 ←𝜒(𝑛+1) 𝑃

Compute 𝜅(𝑛) 𝑃 and 𝜗(𝑛) 𝑓

Conservation satisfied? no yes

𝑛= 𝑛+ 1


$$
𝑡= 𝑡+ Δ𝑡
$$


> **Figure 1: Flow chart of the solution procedure of the discretised and linearised fully-coupled system of governing equations, ⋅𝝓= 𝐛. Superscript (𝑛+1) denotes implicitly solved variables, superscript (𝑛) denotes deferred variables, and superscripts (𝑡−Δ𝑡) and (𝑡−Δ𝜏) denote quantities of previous time-levels.**

The coefficient submatrices of size 𝑁× 𝑁for each governing equation “eq.” with regard to each solution variable 𝜒∈{𝑝, 𝑢, 𝑣, 𝑤, 𝜓} are denoted as 𝜒 eq., where “eq. = cont.” refers to the continuity equation, Eq. (51), eq. = {𝑥-mom., 𝑦-mom., 𝑧-mom.} refers to the three momentum equations with respect to each spatial dimension, Eq. (52), and “eq. = VOF” refers to the advection equation of the VOF colour function, Eq. (53). 𝝓𝜒are the solution subvectors of length 𝑁for each solution variable 𝜒and all contributions from previous time-levels and contributions that are deferred are contained in the right-hand side subvectors 𝐛eq., each of length 𝑁. The linear system of discretised governing equations, Eq. (54), is solved iteratively using the Block-Jacobi preconditioner and the BiCGSTAB solver of the software library PETSc [40, 41]. To account for the nonlinearity of the governing equations, an inexact Newton method [42] is applied, whereby the deferred terms are updated iteratively until the nonlinear system of equations has converged to a predefined conservation criteria, as illustrated in Fig. 1. The nonlinear iterative process is considered to be converged when the maximum 𝐿2 error norm of the quantities 𝜙∈{𝜌, 𝜌𝑢, 𝜌𝑣, 𝜌𝑤, 𝜓} satisfies

max 𝜙 (𝐿2(𝜙)) < 𝜖nonlinear, (55)

where 𝜖nonlinear is typically set to 10−6, and 𝐿2(𝜙) is calculated as

𝐿2(𝜙) =


$$
√
√
√
√
√
√
√
√
$$

1 𝑁cells


$$
∑ 𝑃 ( 𝜙(𝑛+1) 𝑃 −𝜙(𝑛) 𝑃
$$

)2

max𝑃

(( 𝜙(𝑛+1) 𝑃


$$
)2) . (56)
$$

R. Janodet et al.: Preprint submitted to Elsevier Page 10 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

As previously reported by Denner et al. [14], since the implicit volume flux 𝐹(𝑛+1) 𝑓 is based on implicit contributions of all solution variables 𝜒∈{𝑝, 𝑢, 𝑣, 𝑤, 𝜓} and contained in the advection terms of all governing equations, it introduces a tight implicit coupling between the governing equations, which was found to be essential for breaching the capillary time-step constraint. In the proposed algorithm, this implicit coupling between the governing equations is further strengthened by treating the density of the current time-level implicit with respect to the colour function in the transient terms of the continuity and momentum equations. Furthermore, to facilitate the robust solution of interfacial flows with large density ratios, the continuity and momentum equations are discretised in conservative form, contrary to the prevailing standard for incompressible flows, and a Favre averaging is applied in the momentum advection term.


### 4. Upper capillary stability limit

Although the capillary time-step constraint has been shown to be breachable using interface-capturing methods, Denner et al. [14] observed that, even beyond the capillary time-step constraint, the time step that yields a stable solution remains limited. Following the work of Galusinski and Vigneaux [5], the maximum possible time step using the class of fully-coupled algorithms considered in this study has the form [14]


$$
Δ𝑡∗=
√
$$

(𝑎2𝜏vc)2 + 4𝑎1𝜏2 𝜎 2 , (57)

where 𝜏vc and 𝜏𝜎are the visco-capillary and capillary time scales, respectively, and 𝑎1,2 are two case-dependent constants. The capillary time-step constraint Δ𝑡𝜎, Eq. (4), is recovered for 𝑎1 = 1∕(16𝜋) and 𝑎2 = 0. By defining ̂𝜌= 𝜌A + 𝜌B and ̂𝜇= 𝜇A + 𝜇B, and with the wavelength of the shortest unambiguously resolved capillary waves given as 𝜆𝜎= 2Δ𝑥[3, 4], the reference time scales are

𝜏vc = ̂𝜇𝜆𝜎

𝜎 (58)

𝜏𝜎=


$$
√
$$

̂𝜌𝜆3 𝜎 𝜎, (59)

which yield the mesh Ohnesorge number [14]


$$
OhΔ𝑥= 𝜏vc
$$

𝜏𝜎 = ̂𝜇 √

̂𝜌𝜎𝜆𝜎 . (60)

The upper capillary stability limit can, therefore, be separated into two regimes: (i) the inviscid regime for OhΔ𝑥≪1 where Δ𝑡∗∝𝜏𝜎, and (ii) the viscous regime for OhΔ𝑥≫1 where Δ𝑡∗∝𝜏vc. In order to determine the case-dependent constants 𝑎1 and 𝑎2, the results of Denner et al. [14] suggest that only two results obtained for OhΔ𝑥≪1 and OhΔ𝑥≫1, respectively, are sufficient.


### 5. Test cases and validation

The proposed fully-coupled algorithm is validated and scrutinized using four representative test cases in which surface tension plays the dominant role and for which the time step is usually limited by the capillary time-step constraint: the Laplace equilibrium of a stationary droplet (Section 5.1), a standing capillary wave (Section 5.2), an oscillating droplet (Section 5.3), and the Rayleigh-Plateau instability (Section 5.4). The analysis of these results focuses in particular on the force balancing of the discretised governing equations, the conservation of energy, and the maximum stable time step that can be applied in these simulations.

5.1. Laplace equilibrium of a stationary droplet The common first step to validate a surface tension framework is to demonstrate its balanced-force or well-balanced property [6, 43–45] by simulating a static droplet (or bubble) and show that the exact Laplace balance Δ𝑝= 𝜎𝜅can be achieved, provided the interface shape has reached a numerical equilibrium. Given that the spherical interface should satisfy the Laplace balance and be in mechanical equilibrium, the two-phase flow should be quiescent. This is the case

R. Janodet et al.: Preprint submitted to Elsevier Page 11 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Table 1 Dimensionless physical and numerical parameters for the Laplace equilibrium case.**


$$
𝜌A∕𝜌B
𝜇A∕𝜇B
𝐷∕Δ𝑥
Δ𝑡∕Δ𝑡𝜎
$$

1000 1 120 25.6 0.5, 2, 8, 16

0.0 0.2 0.4 0.6 0.8 1.0 Dimensionless viscous time t∗= t/τµ


$$
10−15
$$


$$
10−13
$$


$$
10−11
$$


$$
10−9
$$


$$
10−7
$$


$$
10−5
$$


$$
10−3
$$

Camax

(a) 2D, La = 120


$$
∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


$$
∆t/∆tσ = 16
$$

0.0 0.2 0.4 0.6 0.8 1.0 Dimensionless viscous time t∗= t/τµ


$$
10−15
$$


$$
10−13
$$


$$
10−11
$$


$$
10−9
$$


$$
10−7
$$


$$
10−5
$$


$$
10−3
$$

Camax

(b) 3D, La = 120


$$
∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


$$
∆t/∆tσ = 16
$$


> **Figure 2: Maximum capillary number over time for the Laplace equilibrium case with density ratio 𝜌A∕𝜌B = 1000 in (a) 2D and (b) 3D.**

when the spurious currents, which are the only source of a non-zero velocity 𝐮, introduced by the initial numerical disequilibrium, are dissipated by viscosity. Hence, the relevant time scale to consider is the viscous dissipation time scale 𝜏𝜇, given for a droplet of diameter 𝐷as 𝜏𝜇= 𝜌A𝐷2∕𝜇A. The relevant dimensionless parameter to parameterize this case is the Laplace number La = 𝜌A𝜎𝐷∕𝜇2 A. It is well-known that the exact Laplace equilibrium can be retrieved in segregated-VOF/explicit-CSF frameworks [43], and more recently with the coupled-VOF/implicit-CSF framework of Denner et al. [14] for time steps larger than the capillary time-step constraint for a uniform density ratio. In this work, we consider a large density ratio of 𝜌A∕𝜌B = 1000 for both a two-dimensional (2D) and three-dimensional (3D) droplet. The dimensionless parameters of the problem considered are summarised in Table 1. The evolution of the maximum capillary number Camax = 𝜇A|𝐮|∞∕𝜎of the spurious currents as a function of the dimenonless time 𝑡∗= 𝑡∕𝜏𝜇is shown in Fig. 2 (a) for the 2D case and in Fig. 2 (b) for the 3D case. In both cases, the initial spurious currents are dissipated exponentially by viscosity, until a numerical equilibrium is reached at 𝑡∗≈0.2. For 𝑡∗> 0.2, the magnitude of the spurious currents remains negligible and is defined by the tolerance to which the nonlinear system of governing equations is solved (or machine precision).

The pressure field and spurious currents are illustrated in Fig. 3, at two time instants representing the beginning and the end of the 2D simulation with the largest applied time step, Δ𝑡∕Δ𝑡𝜎= 16. The droplet has a radius of 𝑅= 0.4m and the surface tension coefficient is 𝜎= 1N.m−1, such that the pressure jump follows as Δ𝑝= 𝜎∕𝑅= 2.5Pa, which is correctly predicted by the simulation, see Fig. 3.

The upper capillary stability limit, discussed in Section 4, for this case is displayed in Fig. 4. In the limit of OhΔ𝑥≪1, a time step smaller than 1.5Δ𝑡𝜎has to be applied to obtain a stable solution, which is significantly more restrictive than for the same case with unit density ratio for which a maximum time step of 15Δ𝑡𝜎could be used in this regime [14]. In the large-OhΔ𝑥regime, the upper capillary stability limit Δ𝑡∗is also one order of magnitude smaller than in the case with unit density ratio considered by Denner et al. [14]. Nonetheless, the proposed fully-couppled algorithm is capable to breach the capillary time-step constraint even for interfacial flows with realistic gas-liquid density ratios and still retain the force balance between pressure and surface tension.

R. Janodet et al.: Preprint submitted to Elsevier Page 12 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Figure 3: Pressure field and spurious currents at (a) 𝑡∗= 𝑡∕𝜏𝜇= 0.01 and (b) 𝑡∗= 𝑡∕𝜏𝜇= 1 in 2D for Δ𝑡∕Δ𝑡𝜎= 16.**


$$
10−6
10−5
10−4
10−3
10−2
10−1
$$


$$
Oh∆x
$$


$$
10−1
$$

100

101

102

103

104

105


$$
∆t∗/∆tσ
$$

Stability limit −Laplace equilibrium Explicit capillary time-step constraint


> **Figure 4: Approximated upper capillary stability limit as a function of the mesh Ohnesorge number, OhΔ𝑥, for the Laplace equilibrium case with density ratio 𝜌A∕𝜌B = 1000.**

5.2. Capillary wave We consider the oscillation and decay of a capillary wave. Two immiscible viscous fluids with large density and viscosity ratios are initially at rest and separated by an interface with surface tension, perturbed by a smallamplitude sinusoidal capillary wave, in a two-dimensional [0; 𝜆] × [0; 3𝜆] domain [46, 47]. This domain is periodic in the 𝑥-direction and has slip walls at its top and bottom boundaries. The initial wave amplitude is 𝐴0 = 𝜆∕100, with 𝜆= 2𝜋the length of the capillary wave, which gradually decays due to viscous dissipation. The calculations are performed on Cartesian meshes with a mesh resolution of 𝜆∕Δ𝑥= {25, 50, 100, 200}, using different time steps Δ𝑡∕Δ𝑡𝜎= {0.5, 2, 8}, and run until 𝜔0𝑡= 25, which corresponds to approximately 4 oscillations and where 𝜔0 = √

𝜎𝑘3∕(𝜌A + 𝜌B) is the undamped angular frequency of the capillary wave. The considered density and viscosity ratios are 𝜌A∕𝜌B = 1000 and 𝜇A∕𝜇B = 1000, such that the kinematic viscosity 𝜈= 𝜇∕𝜌is 𝜈A = 𝜈B. The physical and numerical parameters of this case are summarised in Table 2. The results obtained with the proposed fullycoupled algorithm are compared to the analytical solution of the temporal evolution of the wave amplitude proposed by

R. Janodet et al.: Preprint submitted to Elsevier Page 13 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Table 2 Dimensionless physical and numerical parameters for the case of the viscous damping of a capillary wave.**


$$
𝜌A∕𝜌B
𝜇A∕𝜇B
La = 𝜌𝜆𝜎∕𝜇2
𝜆∕Δ𝑥
Δ𝑡∕Δ𝑡𝜎
$$

1000 1000 300 25, 50, 100, 200 0.5, 2, 8


> **Table 3 𝐿2 error norm, see Eq. (61), for each resolution and for each prescribed time step relative to the capillary time-step constraint. The corresponding order of convergence is given in parentheses.**


$$
𝜆∕Δ𝑥
𝐿2 (Δ𝑡∕Δ𝑡𝜎= 0.5)
𝐿2 (Δ𝑡∕Δ𝑡𝜎= 2)
𝐿2 (Δ𝑡∕Δ𝑡𝜎= 8)
$$


$$
5.02 × 10−3
6.00 × 10−3
2.96 × 10−2
$$


$$
2.73 × 10−3 (0.88)
3.15 × 10−3 (0.93)
4.67 × 10−3 (2.66)
1.76 × 10−3 (0.63)
2.29 × 10−3 (0.46)
2.42 × 10−3 (0.95)
1.17 × 10−3 (0.59)
1.35 × 10−3 (0.76)
1.43 × 10−3 (0.76)
$$

Prosperetti [48]. This analytical solution is valid for capillary waves with small amplitude (𝐴≪𝜆) and for interacting fluids with equal kinematic viscosity 𝜈. The temporal evolution of the wave amplitude over time is shown in Figs. 5 (a), (c), and (e). The results obtained with the proposed fully-coupled algorithm are in excellent agreement with the analytical solution of Prosperetti [48], especially for 𝜆∕Δ𝑥≥50, irrespective of the applied time step. The corresponding instantaneous relative amplitude error |𝐴(𝑡) −𝐴th(𝑡)|∕𝐴0 shown in Figs. 5 (b), (d), and (e) suggest that time-step independence is reached with 200 points per wavelength. In order to better illustrate the present test case and confirm the robustness of the solver, the pressure and vertical velocity fields are provided in Fig. 6, at time instant 𝜔0𝑡= 2.5 (right before the wave reaches its minimum amplitude), for 𝜆∕Δ𝑥= 100 and the largest applied time step, Δ𝑡∕Δ𝑡𝜎= 8.

To further quantify the differences between analytical solutions and numerical results, the 𝐿2 error norm is computed relative to the initial amplitude 𝐴0, as previously considered by Popinet [43], defined as

𝐿2 = 1

𝐴0


$$
√
$$

1 𝑡end ∫

𝑡end

𝑡=0


$$
[𝐴(𝑡) −𝐴th(𝑡)]2 d𝑡,
(61)
$$

where 𝑡end is the end time of the simulation, 𝐴is the wave amplitude obtained with the proposed numerical framework and 𝐴th is the wave amplitude obtained with the analytical solution of Prosperetti [48]. The values of the 𝐿2 error norm are gathered in Table 3. The 𝐿2 error norms for all cases decrease with increasing mesh resolution and decreasing time step and, in general, the 𝐿2 error norms are small (< 10−2) for all considered cases, with the expectation of the wave on the coarsest mesh simulated with the largest time step. The rate of convergence is between 1∕2 and 1 for all time steps, which is low compared to the second-order convergence observed with a unit density ratio on Cartesian meshes [43]. However, the density ratio of 𝜌A∕𝜌B = 1000 is less common in the literature and existing results with explicit surface tension [49, 50] also reported a low convergence rate (order 1 or less) for this case. The limiting factor regarding the order of accuracy of this test case are the interface transport and the computation of the interface curvature. The interface transport with the employed schemes is at best second order accurate, which means that the computation of the mean curvature of the interface is at best zeroth order accurate [51]. This suggests that, for a sufficiently high spatial and temporal resolution, the order of convergence of the amplitude error should be zero, meaning that the amplitude error is constant and does not reduce with further mesh refinement.

The upper capillary stability limit for this capillary wave case is plotted in Fig. 7. Similar to the Laplace equilibrium discussed in the previous section, the stability limit is more restrictive for the density ratio of 𝜌A∕𝜌B = 1000 considered here than for the density ratio 𝜌A∕𝜌B = 1 considered by Denner et al. [14]. Nevertheless, especially cases in which viscosity dominates, either because the dynamic viscosity is large or the capillary wave is short, benefit from a drastically increased applicable time step.

5.3. Oscillating droplet The case of an oscillating droplet, which has been previously presented in [52, 53], considers the oscillations of a dense two-dimensional viscous droplet, in which the effects of the surface tension are examined. The initially

R. Janodet et al.: Preprint submitted to Elsevier Page 14 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

0 5 10 15 20 25 ω0t


$$
−1.00
$$


$$
−0.75
$$


$$
−0.50
$$


$$
−0.25
$$

0.00

0.25

0.50

0.75

1.00

A A0

(a) Evolution of the wave amplitude λ/∆x = 50


$$
Theory ∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


$$
0 5 10 15 20 25 ω0t
$$

0.000

0.002

0.004

0.006

0.008

0.010

0.012

0.014

0.016


$$
|A −Ath|/A0
$$


$$
(b) Relative amplitude error λ/∆x = 50
$$


$$
∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$

0 5 10 15 20 25 ω0t


$$
−1.00
$$


$$
−0.75
$$


$$
−0.50
$$


$$
−0.25
$$

0.00

0.25

0.50

0.75

1.00

A A0

(c) Evolution of the wave amplitude λ/∆x = 100


$$
Theory ∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


$$
0 5 10 15 20 25 ω0t
$$

0.000

0.002

0.004

0.006

0.008

0.010

0.012

0.014

0.016


$$
|A −Ath|/A0
$$


$$
(d) Relative amplitude error λ/∆x = 100
$$


$$
∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$

0 5 10 15 20 25 ω0t


$$
−1.00
$$


$$
−0.75
$$


$$
−0.50
$$


$$
−0.25
$$

0.00

0.25

0.50

0.75

1.00

A A0

(e) Evolution of the wave amplitude λ/∆x = 200


$$
Theory ∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


$$
0 5 10 15 20 25 ω0t
$$

0.000

0.002

0.004

0.006

0.008

0.010

0.012

0.014

0.016


$$
|A −Ath|/A0
$$


$$
(f) Relative amplitude error λ/∆x = 200
$$


$$
∆t/∆tσ = 0.5
$$


$$
∆t/∆tσ = 2
$$


$$
∆t/∆tσ = 8
$$


> **Figure 5: Left: Temporal evolution of the dimensionless wave amplitude 𝐴∕𝐴0 over non-dimensional time 𝑡∗= 𝜔0𝑡for 𝜆∕Δ𝑥∈{50, 100, 200}. Right: Instantaneous relative amplitude error |𝐴−𝐴th|∕𝐴0 for 𝜆∕Δ𝑥∈{50, 100, 200}, where 𝐴th is the amplitude of the analytical solution of Prosperetti [48].**

elliptical droplet is situated in a unit square domain, with initial major axis 𝑎= 0.15 m and minor axis 𝑏= 0.1 m. The physical properties and dimensionless parameters of the problem are summarised in Tables 4 and 5, respectively. The simulations are run until 𝑡∕𝜏𝜇= 1.5, where 𝜏𝜇= 𝜌A𝐷2 0∕𝜇A is the viscous time scale, to ensure that both spurious currents and physical interface oscillations are dissipated completely. The time step applied in these simulations is

R. Janodet et al.: Preprint submitted to Elsevier Page 15 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

(a) Pressure 𝑝 (b) 𝑦-Velocity


> **Figure 6: Pressure field (a) and vertical velocity field (b) at 𝑡∗= 𝜔0𝑡= 2.5 for 𝜆∕Δ𝑥= 100 and Δ𝑡∕Δ𝑡𝜎= 8.**


$$
10−6
10−5
10−4
10−3
10−2
10−1
$$


$$
Oh∆x
$$


$$
10−1
$$

100

101

102

103

104

105


$$
∆t∗/∆tσ
$$

Stability limit −Capillary wave Explicit capillary time-step constraint


> **Figure 7: Approximated upper capillary stability limit as a function of the mesh Ohnesorge number, OhΔ𝑥, for the capillary wave case with density ratio 𝜌A∕𝜌B = 1000.**

calculated as the minimum between the CFL and capillary time-step constraints: Δ𝑡= min (Δ𝑡CFL, ΣΔ𝑡𝜎 ), where Σ = Δ𝑡∕Δ𝑡𝜎is the factor by which the capillary time-step constraint Δ𝑡𝜎is breached. Three different time steps are considered for this test case, with Σ ∈{2, 5, 10}, and the maximum CFL number is 0.05. Over time, the oscillations of the droplet induced by surface tension are damped by viscous stresses. Using the theoretical work of Rush and Nadim [54] and Lamb [55], the evolution of the dimensionless total energy (i.e. potential and kinetic) of the 2D droplet for oscillation mode 𝑛is given as


$$
𝐸∗(𝑡) = 𝐸(𝑡)
$$

𝐸0 = exp

(


$$
−2𝑛(𝑛−1)𝜇A𝑡
$$


$$
√
$$

𝜌A𝜎𝑅0

)

= exp

(


$$
−2𝑛(𝑛−1)𝑡
√
$$

La

)


$$
. (62)
$$

R. Janodet et al.: Preprint submitted to Elsevier Page 16 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Table 4 Physical parameters for the oscillating droplet test.**


$$
𝜌A (kg.m−3)
𝜌B (kg.m−3)
𝜇A (kg.m−1.s−1)
𝜇B (kg.m−1.s−1)
𝜎(N.m−1)
√
$$

𝑎𝑏(m)

1000 1 7.5 × 10−2 7.5 × 10−2 0.1 0.1224744871


> **Table 5 Dimensionless parameters for the oscillating droplet test.**


$$
𝜌A∕𝜌B
𝜇A∕𝜇B
La = 𝜌A𝜎𝑅0∕𝜇2
𝑅0∕Δ𝑥
$$

1000 1 2177 7.8


> **Figure 8: Pressure field of the 2D oscillating droplet at 𝑡∗= 𝑡∕𝜏𝜇= 0.015 with Σ = 10.**

The initial total energy, 𝐸0, which corresponds to the initial potential energy associated with surface tension, is calculated as 𝐸0 = 𝜎(ellipse−circle), where ellipse is the circumference of the ellipse and circle is the circumference of the circle with the same area. Following the work of Lamb [55], the analytical expression for the oscillation frequency of the 𝑛-th mode is given as

𝜔𝑛=


$$
√
$$


$$
𝑛(𝑛−1)(𝑛+ 1)𝜎
$$

(𝜌A + 𝜌B)𝑅3 0 , (63)

where 𝑅0 = √

𝑎𝑏is the equilibrium radius of the droplet. As previously considered by Raessi and Pitsch [52] and Vaudor et al. [53], the dominant second mode (𝑛= 2) is studied here. To illustrate the test case, the pressure field during the second oscillation of the droplet, at 𝑡∗= 𝑡∕𝜏𝜇= 0.015, with the largest prescribed time step, Σ = 10, is shown in Fig. 8.

The temporal evolution of the dimensionless kinetic energy of the droplet 𝐸∗ kin = 𝐸kin∕𝐸0 for all simulations is shown in Fig. 9 (a). A zoomed view is provided in Fig. 9 (b) over the range 𝑡∗∈[0, 0.2], corresponding to the interval in which all but one millionth of the initial droplet energy 𝐸0 dissipates. The temporal evolution of the effective time step of the simulations is displayed in Fig. 10, which confirms the representative choice of the reduced interval 𝑡∗∈[0, 0.2], used for the calculations of errors following below. Despite a strong CFL constraint, the largest part of the simulations is conducted with a time step larger than the capillary time-step constraint, as observed in Fig. 10. At the end of all simulations the kinetic energy is, as expected, fully dissipated, see Fig. 9 (a). The theoretical decay rate is accurately reproduced for all time steps in the range 𝑡∗= 𝑡∕𝜏𝜇< 0.1, as seen in Fig. 9 (b). Fig. 10 indicates that 𝑡∗= 0.1 corresponds approximately to the time after which the time step reaches its maximum value Δ𝑡= ΣΔ𝑡𝜎. A more significant difference in the quality of the results (oscillation frequency and kinetic energy decay) depending on the time step is observed for 𝑡∗> 0.1.

R. Janodet et al.: Preprint submitted to Elsevier Page 17 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

0.00 0.25 0.50 0.75 1.00 1.25 1.50 Dimensionless viscous time t∗= t/τµ


$$
10−18
$$


$$
10−15
$$


$$
10−12
$$


$$
10−9
$$


$$
10−6
$$


$$
10−3
$$

100

Dimensionless kinetic energy E∗ kin = Ekin/E0


$$
(a) t∗∈[0, 1.5]
$$


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 5∆tσ)
$$


$$
∆t = min(∆tCFL, 10∆tσ) Theoretical decay rate
$$

0.00 0.05 0.10 0.15 0.20 Dimensionless viscous time t∗= t/τµ


$$
10−9
$$


$$
10−7
$$


$$
10−5
$$


$$
10−3
$$


$$
10−1
$$

Dimensionless kinetic energy E∗ kin = Ekin/E0


$$
(b) t∗∈[0, 0.2]
$$


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 5∆tσ)
$$


$$
∆t = min(∆tCFL, 10∆tσ) Theoretical decay rate
$$


> **Figure 9: Temporal evolution of the dimensionless kinetic energy of the 2D oscillating droplet over time for (a) the entire simulation and (b) a zoomed interval.**


$$
10−3
10−2
10−1
$$


$$
Dimensionless viscous time t∗= t/τµ
$$

0

1

2

3

4

5

6

7

8

9

10


$$
∆t/∆tσ
$$


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 5∆tσ)
$$


$$
∆t = min(∆tCFL, 10∆tσ) Capillary timestep constraint
$$


> **Figure 10: Temporal evolution of the applied time step, normalised by the capillary time-step constraint, for each run (Σ = 2, 5, 10) of the 2D oscillating droplet. The dotted vertical black line corresponds to 𝑡∗= 0.2, which is the upper bound of the zoomed interval in Fig. 9 (b).**

The errors of the oscillation frequency 𝜔𝑛and kinetic energy 𝐸kin are computed over the range 𝑡∗∈[0, 0.2] as

𝜖(𝜔𝑛) = 𝜔𝑛,calc −𝜔𝑛,Lamb

𝜔𝑛,Lamb =


$$
(2𝜋∕𝑛,calc
) −𝜔𝑛,Lamb
(64)
$$

and

𝜖(𝐸kin) = 𝐿2(𝐸kin) =


$$
√
√
√
√
√
$$


$$
𝑁peaks ∑
$$

𝑝=1


$$
(𝐸∗
kin,calc,𝑝−𝐸∗
𝐸∗
$$

)2


$$
(65)
$$

R. Janodet et al.: Preprint submitted to Elsevier Page 18 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Table 6 Errors 𝜖and corresponding convergence rate of the simulation of the oscillating droplet with different time steps. A negative convergence rate indicates a decrease in accuracy as the time step increases.**

Σ = Δ𝑡∕Δ𝑡𝜎 𝜖(𝑅) (𝑅) 𝜖(𝐸kin) (𝐸kin) 𝑛,calc [s] 𝜖(𝜔𝑛) (𝜔𝑛)

2 9.92 × 10−4 − 0.038 − 11.297 +0.0263 − 5 8.68 × 10−4 0.15 0.074 −0.73 11.386 +0.0339 −0.28 10 1.30 × 10−3 −0.58 0.195 −1.40 11.595 +0.0513 −0.60

with 𝑛,calc the average period calculated over the first 14 oscillations (corresponding to the range 𝑡∗∈[0, 0.2]) for each simulation, 𝑁peaks = 28 (corresponding to 𝑡∗∈[0, 0.2]), and 𝐸∗ kin,calc and 𝐸∗ kin,th are the dimensionless energy obtained from the simulations and from theory, respectively. The error of the final radius is also computed at the end of each simulation, 𝑡end = 1.5𝑡∗as

𝜖(𝑅) = ||𝑅calc(𝑡end) −𝑅0||

𝑅0 =


$$
||| √
$$


$$
𝑎calc(𝑡end) 𝑏calc(𝑡end) − √
$$


$$
𝑎𝑏||| √
$$


$$
𝑎𝑏 , (66)
$$

where 𝑎calc and 𝑏calc are calculated by summing up the colour function row-wise and column-wise on the employed Cartesian mesh, respectively. The error values and associated convergence rate are gathered in Table 6 for all considered time steps. The kinetic-energy and oscillation-frequency errors are also plotted in Fig. 11. The geometric error in final radius is small for all considered time steps, which shows that a circle faithfully representing the expected theoretical circle of radius 𝑅0 is obtained at the end of the simulations. The very low convergence rate of the radius further indicates that the final radius of the simulation does not depend strongly on the applied time step. Fig. 11 shows that the oscillation frequency is less affected than the kinetic energy by an increase in time step. Both errors are, nonetheless, small for Σ = 2 and Σ = 5, while Σ = 10 is not sufficient to faithfully capture the physics, as confirmed by Fig. 9 (a). It should be noted that the errors in oscillation frequency for Σ = 2 and Σ = 5 (approx. 3%) are smaller than those obtained by [53] with an explicit surface tension treatment at the same resolution, but over a shorter time interval (approx. 4.5%), demonstrating the good temporal accuracy of the proposed fully-coupled algorithm. It should be noted that the formal order of convergence of a finite-difference scheme can, in general, only be achieved if all relevant physical process are adequately resolved. Here, however, the surface-tension-driven interface motion is not adequately resolved in time, since the capillary time-step constraint is breached. The formal second-order accuracy of the employed temporal discretisation can, thus, not be expected to be achieved.

The performance of the simulations is quantified by the wall clock time (WCT) and the reduced computational time (RCT), calculated as [56]


$$
RCT = WCT × 𝑁cores
$$


$$
𝑁Δ𝑡× 𝑁cells
(67)
$$

where 𝑁cores is the number of processing cores, 𝑁Δ𝑡refers to the number of time steps, 𝑁cells is the number of mesh cells, and presented in Table 7. The WCT of the cases with Σ = 2 and Σ = 5 show that a speed up of factor 1.9 in total simulation time is obtained when multiplying the maximum allowed time step by 2.5. This speed up is however not maintained when increasing the maximum time step further to Σ = 10, which is explained by the on average slower convergence of the nonlinear procedure within each time step for Σ = 10. This is confirmed by the RCT values, showing a substantial increase in average reduced time spent for each time step for Σ = 10. Hence, there exists an optimal case-dependent value for Σ that maximises the performance, while still providing a sufficient resolution of the dominant transient flow features. For the present test case, Σ = 5 seems to be a good compromise.

5.4. 3D Rayleigh-Plateau instability The Rayleigh-Plateau instability is studied to demonstrate the capability of the proposed numerical framework with the implicit surface tension framework to predict surface-tension-driven flows in three dimensions that feature a significant deformation of the liquid-gas interface. To this end, a cylindrical filament of radius 𝑟0 is initialised in a cubic domain with a small longitudinal perturbation imposed at the interface,

𝑟(𝑧) = 𝑟0 + 𝜖𝑟0 cos(𝑘𝑧) = 𝑟0 + 𝐴0 cos(𝑘𝑧), (68)

R. Janodet et al.: Preprint submitted to Elsevier Page 19 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

100 101


$$
∆t/∆tσ
$$


$$
10−2
$$


$$
10−1
$$

L2 kinetic energy error

(a) Kinetic energy peaks

ϵ(Ekin) First order

100 101


$$
∆t/∆tσ
$$


$$
10−2
$$


$$
10−1
$$

Relative frequency error

(b) Oscillation frequency


$$
ϵ(ωn) First order
$$


> **Figure 11: Temporal convergence of the oscillating droplet test: (a) the error in kinetic energy, defined by Eq. (65), and (b) the error in oscillation frequency, defined by Eq. (64).**


> **Table 7 Computational performance of the simulations of the oscillating droplet test for different time steps, quantified by the wall clock time (WCT) and the reduced computational time (RCT), see Eq. (67).**

Σ WCT [s] 𝑁cores 𝑁Δ𝑡 𝑁cells RCT [ms]

2 3370 24 8180 4096 2.41 5 1790 24 3677 4096 2.85 10 1410 24 2207 4096 3.74

where 𝜖= 0.03, 𝑟0 = 0.2, and 𝑘= 𝜋. Following Popinet [43], in order to compare the results to linear theory, the velocity field is initialised as 𝐮= 𝛁𝜙, with the potential

𝜙= − 𝐴0𝑐 𝑖𝑘𝐽1(𝑖𝑘𝑟0)𝐽0(𝑖𝑘𝑟) cos(𝑘𝑧), (69)

where 𝑐is the inviscid growth rate obtained from linear stability analysis [57]

( 𝑐 𝑐0

)2 = 𝐼1(𝜉)


$$
𝐼0(𝜉)𝜉(1 −𝜉2),
(70)
$$

with 𝑐2 0 = 𝜎∕(𝜌A𝑟3 0) and 𝜉= 𝑘𝑟0 the dimensionless wavenumber. To include viscous effects, Weber reformulated the growth-rate equation as [58, 59]:

𝑐2 + 3𝜈A𝑘2𝑐= 1


$$
0𝜉2(1 −𝜉2),
(71)
$$

which allows to validate cases with low to moderate Laplace numbers. Following previous studies [43, 60], the Laplace number considered here is La = 238. A quarter of the filament is simulated in a [0; 0.5] × [0; 0.5] × [0; 1] domain with symmetric boundary conditions, using an equidistant Cartesian mesh with 50 × 50 × 100 cells, which corresponds to 20 cells per initial filament radius. The same dimensional parameters as previously considered by Popinet [43] are adopted for this test case, summarised in Table 8. The characteristic time scale 𝑇𝜎= 1∕𝑐0 of the Rayleigh-Plateau instability is defined by the growth rate 𝑐0. The time step applied in the simulations is calculated as the minimum between the CFL and capillary time-step constraints: Δ𝑡= min (Δ𝑡CFL, ΣΔ𝑡𝜎 ), where Σ = Δ𝑡∕Δ𝑡𝜎is the factor by which the capillary time-step constraint

R. Janodet et al.: Preprint submitted to Elsevier Page 20 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows


> **Table 8 Dimensionless parameters for the Rayleigh-Plateau instability.**


$$
𝜌A∕𝜌B
𝜇A∕𝜇B
La = 𝜌A𝜎𝑟0∕𝜇2
𝑟0∕Δ𝑥
$$

832 56 238 0.628 20

(a) 𝑡∕𝑇𝜎= 0 (b) 𝑡∕𝑇𝜎= 6 (c) 𝑡∕𝑇𝜎= 9

(d) 𝑡∕𝑇𝜎= 10.8 (e) 𝑡∕𝑇𝜎= 11 (f) 𝑡∕𝑇𝜎= 11.2


> **Figure 12: Interface contours of the Rayleigh-Plateau instability of a cylindrical filament for different time instances for Σ = 2, coloured by the axial velocity.**

Δ𝑡𝜎is breached. Two different time steps are considered for this test case, with Σ ∈{2, 8}, and the maximum CFL number is 0.04. The interface contours illustrating the development of the Rayleigh-Plateau instability and the induced interface deformation are shown for Σ = 2 in Fig. 12. Following Popinet [43], the relative deformation of the interface, calculated based on both the maximum and minimum radii, is compared to Weber’s viscous linear theory [58, 59] in Fig. 13. The results are in excellent agreement for both considered time steps, demonstrating the ability of the proposed coupled solver to faithfully reproduce capillary effects in three dimensions with significant interface deformation. The non-dimensional breakup time predicted with both time steps is 𝑡0∕𝑇𝜎= 11.5, which is slightly earlier than a breakup time of 𝑡0∕𝑇𝜎= 12.1 reported by Popinet [43] with explicit surface tension. We attribute this difference to the coarser mesh resolution used here compared to the work of Popinet [43], which means the neck of the filament reaches a radius that cannot be resolved and, consequently, succumbs to numerical breakup at an earlier time 𝑡0∕𝑇𝜎. Following Denner et al. [61], the evolution of the dimensionless minimum radius 𝑟min∕𝑟0 as a function of the dimensionless time to pinch-off 𝜏= (𝑡0 −𝑡)∕𝑇𝜎 is plotted in Fig. 14. Early on in the evolution of the Rayleigh-Plateau instability, the minimum radius follows the inertial scaling 𝑟min ∼(𝑡0 −𝑡)2∕3 [62], whereas shortly before pinch-off the minimum radius faithfully reproduces the theoretical inertial-viscous solution defined as [63]

𝑟min

𝑟0 = 0.0304𝜏

Oh , (72)

where Oh = 1∕ √

La is the Ohnesorge number. The temporal evolution of the applied time step is shown in Fig. 15. The calculations are conducted with a time step larger than the capillary time-step up to 𝑡∕𝑇𝜎= 8, at which point the

R. Janodet et al.: Preprint submitted to Elsevier Page 21 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

0 1 2 3 4 5 6 7 8 9 10 11 12 Dimensionless time t/Tσ

100

101

102

Relative deformation (maximum radius)

(a) Maximum radius


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 8∆tσ)
$$

Viscous theory (Weber)

Inviscid theory (Rayleigh)

Breakup time t0/Tσ

0 1 2 3 4 5 6 7 8 9 10 11 12 Dimensionless time t/Tσ

100

101

102

Relative deformation (minimum radius)

(b) Minimum radius


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 8∆tσ)
$$

Viscous theory (Weber)

Inviscid theory (Rayleigh)


$$
Breakup time t0/Tσ 1/ϵ
$$


> **Figure 13: Relative deformation of the liquid filament over dimensionless time 𝑡∕𝑇𝜎based on (a) the maximum radius (𝑟max −𝑟0)∕(𝜖𝑟0) and (b) the minimum radius (𝑟0 −𝑟min)∕(𝜖𝑟0), compared to Weber’s viscous linear theory, defined in Eq. (71). The dotted line in (b) indicates the theoretical relative deformation from breakup (i.e. when 𝑟min = 0).**


$$
10−2 10−1 100 101
$$


$$
Dimensionless time to breakup τ = (t0 −t)/Tσ
$$


$$
10−1
$$

100

Dimensionless minimum radius rmin/r0


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 8∆tσ)
τ 2/3
$$


$$
0.0304τ/Oh
$$


$$
∆x/r0
$$


> **Figure 14: Evolution of the dimensionless minimum radius 𝑟min∕𝑟0 with respect to the dimensionless time to breakup 𝜏= (𝑡0 −𝑡)∕𝑇𝜎. The dash-dotted line Δ𝑥∕𝑟0 indicates the spatial resolution limit.**

CFL constraint becomes the dominant time-step constraint. Hence, the majority of the simulations is carried out with a time step larger than the capillary time-step constraint.


### 6. Conclusions

The performance of computer simulations of surface-tension-driven phenomena is often limited, in many cases severely, by the capillary time-step constraint [3, 6]. Mitigating or eliminating this constraint, therefore, promises significant performance gains, for applications including microfluidic interfacial flows, quasi-stationary evaporation, dynamic contact line problems, and spray atomization.

R. Janodet et al.: Preprint submitted to Elsevier Page 22 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

0 1 2 3 4 5 6 7 8 9 10 11 12 Dimensionless time t/Tσ

0

1

2

3

4

5

6

7

8


$$
∆t/∆tσ
$$


$$
∆t = min(∆tCFL, 2∆tσ)
$$


$$
∆t = min(∆tCFL, 8∆tσ) Capillary timestep constraint
$$


> **Figure 15: Temporal evolution of the applied time step, normalised by the capillary time-step constraint, for both simulations, Σ ∈{2, 8}, of the Rayleigh-Plateau instability.**

In this article, a fully-coupled pressure-based algorithm for interfacial flows with implicit surface tension treatment has been derived, implemented and validated. In the present work, the THINC/QQ scheme [17, 18] has been used to discretise the advection term of the colour function transport equation, instead of the CICSAM scheme [15] considered previously in similar algorithms [14, 30], with the aim of alleviating the stringent CFL number constraint of CFL ≲0.01 associated with the CICSAM scheme. However, even with the THINC/QQ scheme, which is able to retain a sharp interface for CFL numbers of (0.1), stable results could only be obtained for a maximum CFL number of 0.05 in the current study. In conclusion, the proposed algorithm allows simulating realistic gas-liquid flows with time steps larger than the capillary time-step constraint, as long as other time-step constraints are satisfied. Nevertheless, areas of further improvement remain to maximise the potential of the proposed numerical approach. In order to handle complex topology changes, the implicit height-function method requires an implicit alternative to increase its robustness in the case of under-resolved interfaces. In addition, further improvements with respect to the robustness of the interface advection scheme should enable simulations with larger CFL numbers, bearing the potential to greatly improve the performance of the proposed algorithm.


### Acknowledgements

This research was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), grant numbers 452036112, 452916560, and 458610925. Fruitful discussions with Aman Jain and Fabien Evrard are gratefully acknowledged. Data supporting this publication can be obtained from https://zenodo.org/records/ 13215768 under a Creative Commons Attribution license.


## References

[1] R. Courant, K. Friedrichs, H. Lewy, Über die partiellen Differenzengleichungen der mathematischen Physik, Mathematische Annalen 100 (1928) 32–74. [2] S. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere Publishing Company, 1980. [3] J. Brackbill, D. Kothe, C. Zemach, Continuum Method for Modeling Surface Tension, Journal of Computational Physics 100 (1992) 335–354. [4] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, Journal of Computational Physics 285 (2015) 24–40. [5] C. Galusinski, P. Vigneaux, On stability condition for bifluid flows with surface tension: Application to microfluidics, Journal of Computational Physics 227 (2008) 6140–6164.

R. Janodet et al.: Preprint submitted to Elsevier Page 23 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

[6] S. Popinet, Numerical models of surface tension, Annual Review of Fluid Mechanics 50 (2018) 49–75. [7] T. Y. Hou, J. S. Lowengrub, M. J. Shelley, Removing the stiffness from interfacial flows with surface tension, Journal of Computational Physics 114 (1994) 312–338. [8] E. Bänsch, Finite element discretization of the Navier-Stokes equations with a free capillary surface, Numerische Mathematik 88 (2001) 203–235. [9] S. Hysing, A new implicit surface tension implementation for interfacial flows, International Journal for Numerical Methods in Fluids 51 (2006) 659–672. [10] M. Raessi, M. Bussmann, J. Mostaghimi, A semi-implicit finite volume implementation of the CSF method for treating surface tension in interfacial flows, International Journal for Numerical Methods in Fluids 59 (2009) 1093–1110. [11] F. Denner, F. Evrard, R. Serfaty, B. van Wachem, Artificial viscosity model to mitigate numerical artefacts at fluid interfaces with surface tension, Computers & Fluids 143 (2017) 59–72. [12] M. Sussman, M. Ohta, A stable and efficient method for treating surface tension in incompressible two-phase flow, SIAM Journal on Scientific Computing 31 (2009) 2447–2471. [13] W. Zheng, B. Zhu, B. Kim, R. Fedkiw, A new incompressibility discretization for a hybrid particle MAC grid representation with surface tension, Journal of Computational Physics 280 (2015) 96–142. [14] F. Denner, F. Evrard, B. van Wachem, Breaching the capillary time-step constraint using a coupled VOF method with implicit surface tension, Journal of Computational Physics 459 (2022) 111128. [15] O. Ubbink, R. Issa, A Method for Capturing Sharp Fluid Interfaces on Arbitrary Meshes, Journal of Computational Physics 153 (1999) 26–50. [16] V. Gopala, B. van Wachem, Volume of fluid methods for immiscible-fluid and free-surface flows, Chemical Engineering Journal 141 (2008) 204–221. [17] B. Xie, F. Xiao, Toward efficient and accurate interface capturing on arbitrary hybrid unstructured grids: The THINC method with quadratic surface representation and Gaussian quadrature, J. Comput. Phys. 349 (2017) 415–440. [18] D. Chen, B. Xie, F. Xiao, Revisit to the THINC/QQ scheme: Recent progress to improve accuracy and robustness, Int. J. Numer. Meth. Fluids. 94 (2022) 719–755. [19] M. Rudman, A volume-tracking method for incompressible multifluid flows with large density variations, International Journal for Numerical Methods in Fluids 28 (1998) 357–378. [20] G. Tryggvason, R. Scardovelli, S. Zaleski, Direct numerical simulations of gas-liquid multiphase flows, Cambridge University Press, Cambridge ; New York, 2011. [21] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–225. [22] F. Denner, F. Evrard, B. van Wachem, Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds, Journal of Computational Physics 409 (2020) 109348. [23] P. Bartholomew, F. Denner, M. Abdol-Azis, A. Marquis, B. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, Journal of Computational Physics 375 (2018) 177–208. [24] F. Evrard, F. Denner, B. van Wachem, Height-function curvature estimation with arbitrary order on non-uniform Cartesian grids, Journal of Computational Physics: X 7 (2020) 100060. [25] D. Fuster, S. Popinet, An all-Mach method for the simulation of bubble dynamics problems in the presence of surface tension, Journal of Computational Physics 374 (2018) 752–768. [26] T. Arrufat, M. Crialesi-Esposito, D. Fuster, Y. Ling, L. Malan, S. Pal, R. Scardovelli, G. Tryggvason, S. Zaleski, A mass-momentum consistent, Volume-of-Fluid method for incompressible flow on staggered grids, Computers & Fluids 215 (2021) 104785. [27] C. M. Rhie, W. L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA Journal 21 (1983) 1525–1532. [28] F. Denner, Fully-coupled pressure-based algorithm for compressible flows: linearisation and iterative solution strategies, Computers & Fluids 175 (2018) 53–65. [29] M. Darwish, I. Sraj, F. Moukalled, A Coupled Incompressible Flow Solver on Structured Grids, Numerical Heat Transfer, Part B: Fundamentals 52 (2007) 353–371. [30] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numerical Heat Transfer Part B: Fundamentals 65 (2014) 218–255. [31] M. Darwish, A. Aziz, F. Moukalled, A Coupled Pressure-Based Finite-Volume Solver for Incompressible Two-Phase Flow, Numerical Heat Transfer, Part B: Fundamentals 67 (2015) 47–74. [32] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234. [33] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, International Journal for Numerical Methods in Fluids 48 (2005) 1023–1040. [34] F. Xiao, S. Ii, C. Chen, Revisit to the THINC scheme: A simple algebraic VOF algorithm, Journal of Computational Physics 230 (2011) 7086–7092. [35] R. Kumar, L. Cheng, Y. Xiong, B. Xie, R. Abgrall, F. Xiao, THINC scaling method that bridges VOF and level set schemes, Journal of Computational Physics 436 (2021) 110323. [36] M. B. Kuhn, G. Deskos, M. A. Sprague, A mass–momentum consistent coupling for mesh-adaptive two-phase flow simulations, Computers & Fluids 252 (2023) 105770. [37] P. Roe, Characteristic-based schemes for the euler equations, Annual Review of Fluid Mechanics 18 (1986) 337–365. [38] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, Journal of Computational Physics 298 (2015) 466–479.

R. Janodet et al.: Preprint submitted to Elsevier Page 24 of 25

A fully-coupled algorithm with implicit surface tension treatment for interfacial flows

[39] F. Moukalled, L. Mangani, M. Darwish, The finite volume method in computational fluid dynamics: An advanced introduction with OpenFOAM and Matlab, Springer, 2016. [40] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. [41] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, 2017. [42] R. Dembo, S. Eisenstat, T. Steihaug, Inexact newton methods, SIAM Journal on Numerical Analysis 19 (1982) 400–408. [43] S. Popinet, An accurate adaptive solver for surface-tension-driven interfacial flows, Journal of Computational Physics 228 (2009) 5838–5866. [44] T. Abadie, J. Aubin, D. Legendre, On the combined effects of surface tension force calculation and interface advection on spurious currents within Volume of Fluid and Level Set frameworks, Journal of Computational Physics 297 (2015) 611–636. [45] M. O. Abu-Al-Saud, S. Popinet, H. A. Tchelepi, A conservative and well-balanced surface tension model, Journal of Computational Physics 371 (2018) 896–913. [46] S. Popinet, S. Zaleski, A front-tracking algorithm for accurate representation of surface tension, International Journal for Numerical Methods in Fluids 30 (1999) 775 – 793. [47] F. Denner, G. Paré, S. Zaleski, Dispersion and viscous attenuation of capillary waves with finite amplitude, Euro. Phys. J. Spec. Top. 226 (2017) 1229–1238. [48] A. Prosperetti, Motion of two superposed viscous fluids, Physics of Fluids 24 (1981) 1217–1223. [49] M. Herrmann, A balanced force refined level set grid method for two-phase flows on unstructured flow solver grids, Journal of Computational Physics 227 (2008) 2674–2706. [50] O. Desjardins, V. Moureau, H. Pitsch, An accurate conservative level set/ghost fluid method for simulating turbulent atomization, Journal of Computational Physics 227 (2008) 8395 – 8416. [51] F. Evrard, R. Chiodi, B. van Wachem, O. Desjardins, Simulating interfacial flows: a farewell to planes, arXiv (2024) 2401.15012. [52] M. Raessi, H. Pitsch, Consistent mass and momentum transport for simulating incompressible interfacial flows with large density ratios using the level set method, Computers & Fluids 63 (2012) 70–81. [53] G. Vaudor, T. Menard, W. Aniszewski, M. Doring, A. Berlemont, A consistent mass and momentum flux computation method for two phase flows. Application to atomization process, Comput. Fluids 152 (2017) 204–216. [54] B. M. Rush, A. Nadim, The shape oscillations of a two-dimensional drop including viscous effects, Engineering Analysis with Boundary Elements 24 (2000) 43–51. [55] H. Lamb, Hydrodynamics, Cambridge University Press, 6th edition, 1932. [56] R. Janodet, C. Guillamón, V. Moureau, R. Mercier, G. Lartigue, P. Bénard, T. Ménard, A. Berlemont, A massively parallel accurate conservative level set algorithm for simulating turbulent atomization on adaptive unstructured grids, Journal of Computational Physics 458 (2022) 111075. [57] L. Rayleigh, On the Instability of a Cylinder of Viscous Liquid under Capillary Force, Philosophical Magazine 34 (1892) 145–154. [58] C. Weber, Zum Zerfall eines Flussigkeitsstrahles, Zeitschrift fuer angewandte Mathematik und Mechanik 11 (1931) 136–154. [59] A. M. Sterling, C. A. Sleicher, The instability of capillary jets, J. Fluid Mech. 68 (1975) 477–495. [60] M. Dai, D. Schmidt, Adaptive tetrahedral meshing in free-surface flow, Journal of Computational Physics 208 (2005) 228–252. [61] F. Denner, F. Evrard, A. A. Castrejón-Pita, J. R. Castrejón-Pita, B. van Wachem, Reversal and Inversion of Capillary Jet Breakup at Large Excitation Amplitudes, Flow, Turbulence and Combustion 108 (2022) 843–863. [62] J. Castrejón-Pita, A. Castrejón-Pita, S. Thete, K. Sambath, I. Hutchings, J. Hinch, J. Lister, O. Basaran, Plethora of transitions during breakup of liquid filaments., Proc. Nat. Acad. Sci. USA 112 (2015) 4582–4587. [63] J. Eggers, Universal pinching of 3D axisymmetric free-surface flow, Physical Review Letters 71 (1993) 3458–3460.

R. Janodet et al.: Preprint submitted to Elsevier Page 25 of 25

