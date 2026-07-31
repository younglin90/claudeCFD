J. Fluid Mech. (2007), vol. 590, pp. 239–264. c⃝2007 Cambridge University Press doi:10.1017/S0022112007007999 Printed in the United Kingdom 239


# Axisymmetric deformation and stability of a viscous drop in a steady electric field

ETIENNE LAC AND G. M. H O M S Y

Department of Mechanical and Environmental Engineering, University of California, Santa Barbara, CA, USA

(Received 11 December 2006 and in revised form 14 June 2007)

We consider a neutrally buoyant and initially uncharged drop in a second liquid subjected to a uniform electric field. Both liquids are taken to be leaky dielectrics. The jump in electrical properties creates an electric stress balanced by hydrodynamic and capillary stresses. Assuming creeping flow conditions and axisymmetry of the problem, the electric and flow fields are solved numerically with boundary integral techniques. The system is characterized by the physical property ratios R (resistivities), Q (permitivities) and λ (dynamic viscosities). Depending on these parameters, the drop deforms into a prolate or an oblate spheroid. The relative importance of the electric stress and of the drop/medium interfacial tension is measured by the dimensionless electric capillary number, CaE. For λ = 1, we present a survey of the various behaviours obtained for a wide range of R and Q. We delineate regions in the (R, Q)-plane in which the drop either attains a steady shape under any field strength or reaches a fold-point instability past a critical CaE. We identify the latter with linear instability of the steady shape to axisymmetric disturbances. Various break-up modes are identified, as well as more complex behaviours such as bifurcations and transition from unstable to stable solution branches. We also show how the viscosity contrast can stabilize the drop or advance break-up in the different situations encountered for λ = 1.

1. Introduction When an electric field meets an interface separating two immiscible liquids, it undergoes a jump due to the change of physical properties from one medium to the next. One of the consequences of the field discontinuity is the presence of an electric stress on the interface. In the case of a suspended drop placed in an otherwise uniform electric field, the curvature of the interface creates surface gradients of electric field and stress which are likely to deform the drop. In the absence of free charge, if the two fluids are regarded as ideally insulating dielectrics, or if the drop phase is a perfect conductor in an insulating medium, the interfacial electric stress is normal. It may then be balanced by surface tension, and the drop always deforms into a prolate spheroid, where the axis of symmetry is the direction of the imposed electric field (Allan & Mason 1962). Since both phases are at rest if a steady state is reached, this phenomenon is referred to as electrohydrostatics. The literature on the subject is vast and is only briefly summarized in table 1. If, more realistically, the two liquids are assumed to be slightly conducting (so-called leaky dielectrics), the electric current outside and inside the drop induces a surface charge distribution which creates, in addition to the electric pressure, a tangential stress distribution on the interface (Taylor 1966; Melcher & Taylor 1969; Saville 1997). Accounting for this effect, Taylor concluded that the tangential stress could

240 E. Lac and G. M. Homsy

Experimental work Allan & Mason (1962); Torza et al. (1971); Vizika & Saville (1992); Ha & Yang (2000a, b); Sato et al. (2006).

EHS: Theoretical modelling O’Konski & Thacher (1953); Allan & Mason (1962); Taylor (1964). Numerical simulation Brazier-Smith (1971); Miksis (1981); Sherwood (1988); Dubash & Mestel (2007).

EHD: Theoretical modelling Taylor (1966); Torza et al. (1971); Ajayi (1978). Numerical simulation Sherwood (1988); Feng & Scott (1996); Baygents et al. (1998); Feng (1999). Reviews Melcher & Taylor (1969); Saville (1997).


> **Table 1. Summary of existing studies on uncharged drops suspended in a steady electric field (EHS/EHD – ElectroHydroStatics/Dynamics).**

only be balanced by viscous fluid motion and developed the electrohydrodynamic model. In the limit of creeping flow approximation and small perturbation from sphericity, he was able to determine at first order the drop deformation under a given electric field. In particular, these results show that, depending on fluid properties, the drop may remain spherical or even be deformed into an oblate spheroid. Later, Torza, Cox & Mason (1971) extended Taylor’s theory to the case of a unidirectional alternating electric field. When compared to many experimental data (e.g. Torza et al. 1971; Vizika & Saville 1992; Ha & Yang 2000a, b), Taylor’s theory is able to predict the type of deformation (prolate or oblate). However, in the experiments by Torza et al. and by Vizika & Saville, it always underestimates the drop deformation, at best by a factor 1.02 to 1.6. In an effort to predict the drop behaviour more accurately, Ajayi (1978) extended Taylor’s theory by taking into account higher-order terms in shape distortion. Even though the correction terms sometimes predict an increased drop deformation, Ajayi’s work still fails to resolve the discrepancies between theory and experiments. Ha & Yang (2000a, b) conducted experiments with a wide choice of liquid pairs, but deduced the drop/medium interfacial tension by fitting their data with Taylor’s solution in the small-deformation regime, which makes irrelevant the discussion on the validity of the linear theory. Nevertheless, it is noteworthy that their experiments cover a much wider range of drop deformations than the previous investigations. Importantly, the results clearly show a departure from linearity as the electric field is increased, which allows one to estimate the limit of the linear regime in terms of drop deformation. On the numerical side, various investigations on electrohydrodynamics have been conducted for axisymmetric configurations. Miksis (1981) developed a boundary integral method to calculate the static shape of a single dielectric drop suspended in an unbounded medium and subjected to a steady electric field. Assuming creeping flow conditions, Sherwood (1988) used similar techniques to solve both the electric and the electrohydrodynamic flow fields, and was able to simulate the response of a leaky dielectric drop in an electric field. In this pioneering work, Sherwood demonstrated the existence of highly elongated yet stable drop shapes, and identified different break-up modes. However, the study was restricted to a relatively narrow interval of electrical properties (in particular, oblate drops were not considered), and to a drop/medium viscosity ratio held at unity for reasons of computational convenience

Axisymmetric deformation of a drop in a steady electric field 241

and efficiency. Feng & Scott (1996) conducted computations based on the Galerkin finite element method, where the complete field equations are solved for the electric field and fluid motion inside and outside the drop. These authors have developed a very complete model for the fluid mechanics, including viscosity contrast and finite Reynolds number. They concluded that inertia effects were negligible for the liquid pairs commonly used in the experiments cited above. When both liquids have low viscosities (comparable with water), however, inertia was found to alter considerably the oblate drops, which can exhibit a prolate-like shape under high enough electric fields. The effect of viscosity ratio was briefly studied, considering only two liquid pairs. Later, Feng (1999) continued the work of Feng & Scott by taking into account the effect of charge convection along the interface due to fluid motion. He concluded that this phenomenon had a tendency to enhance prolate deformations and reduce oblate deformations. The main limitation of the finite element based methods remains the necessity to truncate the outer domain. Feng & Scott (1996) made a careful examination of the boundary effects by comparing the numerical solution to Taylor’s around a sherical drop for various fluid properties, and truncated their domain at a distance of ten drop radii around the drop. In the range of deformations studied by Feng & Scott (1996) and then by Feng (1999), it is certain that truncation effects were indeed negligible. However, it is not so when the drop deformation becomes very large, as in Sherwood’s simulations, where the drop length can reach up to sixteen drop radii. Our present goal is to pursue Sherwood’s work over a broader range of parameters in order to give as complete an overview as possible of the drop behaviour in the framework of leaky dielectric liquids and creeping flow motion. In particular, we aim to identify the different modes of break-up and study the influence of the viscosity ratio on the drop stability. Furthermore, we wish to compare our numerical results to the predictions of Ajayi (1978) to determine the limit of validity of the secondorder theory. This has not yet been done, and would be very useful when comparing experimental data and theoretical predictions. In § 2, we recall the governing equations of the problem and their boundary conditions, as well as the small-deformation theories by Taylor and by Ajayi. Section 3 is devoted to the presentation of the boundary integral techniques developed to find the steady solution of the problem, and to study its stability. We first present in § 4 the results obtained for a drop which has the same viscosity as the suspension medium. Then, the effect of viscosity contrast is specifically investigated in § 5. A concluding discussion and remarks on future research are finally proposed in § 6.

2. Problem statement We consider a small liquid drop of radius a suspended in another liquid subjected to a uniform electric field E∞, as shown schematically in figure 1. For any physical quantity, barred characters will refer to the drop, whereas unbarred ones will stand for the suspension liquid. The liquids inside and outside the drop are assumed to be Newtonian with density ρ, and viscosity ¯µ and µ, respectively. We assume that the liquids are both leaky dielectrics (Taylor 1966) with constant electric properties, namely the resistivity ¯χ, χ and the dielectric constant ¯ε, ε. The ratios of the physical properties are denoted by


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq001.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq002.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq003.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq004.png)

µ, (2.1)

242 E. Lac and G. M. Homsy

V

a


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq005.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq006.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq007.png)


> **Figure 1. Representation of the suspended drop.**

where ¯σ and σ are the electric conductivities. We denote by S the surface of the drop and by V its volume. Note that �� X �� will hereafter denote the jump X −¯X across S.

2.1. The electric field The fluids are devoid of charge except at the interface; thus the electric field satisfies

∇· ¯E = 0 inside the drop, ∇· E = 0 outside the drop.


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq008.png)

Hereafter, we shall denote by En the normal field E · n and by Et = E −Enn the tangential part of E (resp. ¯En, ¯Et for the drop), n being the outer unit normal vector to S. Across the drop interface, Et is continuous but En undergoes a discontinuity due to the difference of physical properties of the two media (Landau & Lifshitz 1984). Assuming that the electric charge reorganization occurs over a much shorter time scale than that of the interfacial flow, the charge flux balance simply dictates that the normal electric current be continuous across S. Ohm’s law then provides the boundary condition �� σEn �� = 0, i.e. ¯En = REn. (2.3)

The difference of permittivity creates a surface charge distribution


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq009.png)

where ε0 is the permittivity of vacuum. Far away from the drop, the perturbation on the electric field vanishes:


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq010.png)

2.2. Electric stress on the interface The electric stress is given by the Maxwell stress tensor


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq011.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq012.png)

where I denotes the identity tensor. The discontinuity of the electric field across the interface creates a jump of stress �� σ E · n �� denoted by


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq013.png)

= �pE n + qs Et (2.7)

expressed here in terms of external electric field components, taking into account the boundary condition (2.3). The normal electric stress �pE is sometimes referred to as the electric pressure.

Axisymmetric deformation of a drop in a steady electric field 243

2.3. Hydrodynamics Assuming that creeping flow conditions prevail, the interfacial velocity u may be expressed by the classical integral equation (e.g. Pozrikidis 1992)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq014.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq015.png)

S �f i(y) Jij(x, y) dS(y)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq016.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq017.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq018.png)

S ui(y) Kijk(x, y) nk(y) dS(y), (2.8)

where �f = �� σ H · n �� denotes the jump of viscous traction across the interface and σ H

represents the Newtonian stress tensor in the two phases; J and K are the free-space Green’s functions given by


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq019.png)

r + rirj

r3 , Kijk(x, y) = rirjrk

r5 , (2.9)

with r = x −y and r = |r|. The stress balance on the interface is �� (σ H + σ E) · n �� = 2κγ n, (2.10)

where κ denotes the local mean curvature of the surface and γ is the drop/medium interfacial tension. According to (2.10),


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq020.png)

The electric and velocity fields are coupled through both normal and tangential components of (2.11).

2.4. Dimensional analysis The electric field is scaled by the intensity of the applied field, E∞. Since the drop is suspended in an unbounded medium, the natural length scale of the problem is the initial drop radius, a. If U denotes a characteristic velocity, the viscous and electric stress are scaled by µU/a and εε0E2 ∞, respectively. Balancing these two stresses gives U = aεε0E2 ∞/µ. The relative importance of the electric stress and surface tension is measured by the so-called electric capillary number, defined as

CaE = µU


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq021.png)

Hence, if a star superscript indicates a dimensionless quantity, the stress balance (2.11) yields


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq022.png)

Thus, the problem depends on a single dynamic parameter, CaE, and three property ratios, R, Q and λ.

2.5. Small-deformation theory In the limit of small perturbations of a spherical drop, the problem has been solved by Taylor (1966) and then by Ajayi (1978) to the first and second order in CaE. The axisymmetric drop deformation


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq023.png)

l1 + l2 , (2.14)

244 E. Lac and G. M. Homsy

10–2 10–1 100 101 102

10–1

10–2

102

101

100

R

Q

PR+ A

PR+ B

PR– B

PR– A

PRA

PRB

OB OB+

OB–


> **Figure 2. (R, Q)-diagram for λ = 1. The solid line corresponds to the spherical drop (k1 = k2 = 0); the dashed line, k1 ̸= 0, k2 = 0; the dotted line shows RQ = 1; PR = prolate, OB = oblate; ± exponents show the sign of k2. The right-hand insets show the flow pattern around the drop for the different cases; the dash-dot lines show the axis of revolution and the direction of the electric field.**

where l1 and l2 are the drop length and breadth, respectively, was found to be D = k1 CaE + k2 Ca2 E + O(Ca3 E), with

k1 = 9


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq024.png)

(1 + 2R)2 ,

k2 = k1 (1 + 2R)2


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq025.png)

16


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq026.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq027.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq028.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq029.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq030.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq031.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq032.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq033.png)

700


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq034.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq035.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq036.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq037.png)

(2.15)

Fd(R, Q, λ) is known as Taylor’s discriminating function, for it determines at first order the sign of D, i.e. it predicts whether the drop will deform into a prolate (Fd > 0) or an oblate (Fd < 0) shape. The particular case Fd = 0 corresponds to a set of physical properties such that, at first and second order, the drop remains spherical under any electric field, and thus k1 = k2 = 0. Figure 2 is a diagram in the (R, Q)-space showing the various behaviours of the drop as predicted by the small-perturbation theory for λ = 1. It is a classic representation (e.g. Torza, Cox & Mason 1971; Baygents, Rivette & Stone 1998) that we reproduce for its remarkable clarity. In this graph, the denominations PR and OB refer to prolate and oblate deformation, respectively. The + or −exponent indicates the sign of k2, which allows one to predict if Taylor’s theory underor overestimates the deformation of the drop. At lowest order (Taylor 1966), the tangential velocity on S is


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq038.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq039.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq040.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq041.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq042.png)

Axisymmetric deformation of a drop in a steady electric field 245

10–4 10–3 10–2 10–1 100 101 102 103 104

–5

–4

–3

–2

–1

0

1

R

Q = 0.01 0.1

3

1

10 100

k1


> **Figure 3. First-order coefficient k1 as a function of R, for set values of Q and λ = 1. The dashed lines show the asymptotic value of k1 when R →∞.**

where θ is the polar angle in spherical coordinates, such that the direction θ = 0 is that of the imposed electric field. Hence, the direction of the interfacial flow on each hemisphere is determined by the sign of 1 −RQ. The particular case RQ = 1 corresponds to a vanishing surface charge distribution and tangential electric stress. Consequently, if equilibrium is reached in this case, both the internal and external fluids are at rest. In figure 2, the curve RQ = 1 separates the prolate region in two subregions denoted PRA and PRB. When RQ < 1, the external electrohydrodynamic flow goes from equator to pole, resembling a straining flow elongating the drop along the symmetry axis; if RQ > 1, conversely, the flow runs from pole to equator, as shown schematically in the three sketches in figure 2. We show in figure 3 the variation of the coefficient k1 as a function of R for λ = 1 and different values of Q. This coefficient is useful, because it represents a rough estimate of the sensitivity of the drop to an applied electric field. When R tends to zero, k1 tends to the constant value 9/16, independent of Q and λ. When R →∞, k1 tends to 9[ 1 −α(λ) Q ]/64, where α(λ) is the function between square brackets in the expression for Fd (2.15). For finite values of R, k1 is always bounded by these two limits. In many experiments where R ≪1 (e.g. the class A2 of Torza et al. 1971), it is interesting to note that the discrepancy observed between theory and experiments cannot be due to inaccuracies in the ratios R, Q and λ, because k1 is virtually insensitive to any of these parameters. On the other hand, when Q > 3, typically, one might expect possibly large errors in the calculation of k1 when 0.1 < R < 10 (most likely in the oblate regime).

3. Numerical method for arbitrary drop deformations 3.1. Steady drop shapes Given the physical property ratios R, Q and λ, we wish to find a steady drop shape (if any) which satisfies the set of equations presented in § 2. To do so, we consider an initially spherical drop at rest suddenly subjected at time t = 0 to an electric field E∞, and follow its deformation in time by integrating the advection equation


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq043.png)

∂t = un(x) n(x) + (I −nn) · uc(x), (3.1)

246 E. Lac and G. M. Homsy

where x is a marker point on the interface, un = u · n is given by (2.8), and uc is a correction velocity, classically used in Eulerian interface tracking problems to ensure a denser distribution of nodes in the high-curvature regions (e.g. Cristini, Blawzdziewicz & Loewenberg 2001). Equation (3.1) is solved by the fourth-order Runge–Kutta method. Numerically, a satisfactory steady state is obtained when the maximum dimensionless normal velocity over the profile is smaller than a chosen tolerance (typically, 10−5). To solve equation (2.8) for the calculation of the interfacial velocity, we need to determine the stress distribution �f . After equation (2.11), we first have to calculate the jump of electric stress (2.7), which requires knowledge of the external electric field on S. In a boundary integral formulation of the problem, the electric field may be represented as a surface distribution of dipoles (Baygents et al. 1998; Stone, Lister & Brenner 1999):

E∞+ �

S


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq044.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq045.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq046.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq047.png)

1 2[E(x) + ¯E(x)] if x ∈S


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq048.png)

(3.2)

The above expression exactly satisfies the far-field condition (2.5). Using the boundary condition (2.3), an integral equation for En can be deduced from (3.2) at any point x on the interface by taking its inner product with n(x):


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq049.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq050.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq051.png)

S

r · n(x)

r3 En(y) dS(y) = 1 + R

2 En(x). (3.3)

Once En is known on the surface, Et is given by

Et = E + ¯E


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq052.png)

2 Enn, (3.4)

where the mean electric field on S is provided by equation (3.2). Finally, the jump of electric stress is calculated with equation (2.7). In this work, we consider the axisymmetric motion of the drop, where the axis of revolution is the direction of E∞. The initial interface profile is divided in m equal elements defining m + 1 nodes x0, . . . , xm, where the two extreme nodes lie on the revolution axis. In the results presented hereafter, m ranges from 50 to 200. Within the elements, the profile is interpolated by a cubic B-spline:


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq053.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq054.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq055.png)

where the Bk are piecewise cubic polynomials, and ˜xk are the spline coefficients associated with x at time t. The parameter ξ runs from 0 to π and initially corresponds to the dimensionless arclength along the drop profile, defined for the node xi as iπ/m (figure 4). For each scalar variable, the spline representation (3.5) requires m + 3 spline coefficients and two boundary conditions imposed on the revolution axis. For any variable that vanishes on the axis due to the axisymmetry of the problem (such as the radial component of any vector field), we impose that the second derivative with respect to ξ be zero at x0 and xm. Otherwise, we require the first derivative to be zero. For the vector position x on the interface, in particular, this means that we impose a tangent vector perpendical to the axis, implying smooth tips. The present

Axisymmetric deformation of a drop in a steady electric field 247


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq056.png)

xm x0


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq057.png)

xi


> **Figure 4. Discretization of the initial drop profile.**

method thus excludes the possibility of conical ends, but we will see later that very high-curvature tips can be captured. Owing to the axisymmetry of the problem, the various integrals are first calculated analytically in the azimuthal direction and reduce to elliptic integrals that can be accurately estimated by convergent series (Pozrikidis 1992, pp. 38–41). For the calculation of the singleand double-layer kernels in equation (2.8), we have used routines from the open-source library BEMLIB, written by Costas Pozrikidis. The line integrals along the drop profile were then calculated numerically by Gauss–Legendre quadrature. After discretization in the spline coefficient space, the integral equations (2.8) and (3.3) are expressed as linear systems and solved using the LAPACK library.

3.2. Linear stability of the solution If steady solutions exist for a given set of parameters (R, Q, λ, CaE), it is useful to investigate their stability. A simple way to do this consists of studying the time relaxation of the drop after a slight pertubation of the numerical steady shape. We shall consider axisymmetric modes of perturbation only. The steady solution profile rs(θ), in spherical coordinates, may be decomposed into Legendre modes, i.e.

rs(θ) = �

k�0 ak Pk(cos θ) with ak = 2k + 1

2


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq058.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq059.png)

where Pk denotes the order-k Legendre polynomial of the first kind. We may then superimpose an axisymmetric perturbation r′, such that


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq060.png)

k�0 (ak + a′ k) Pk(cos θ). (3.7)

If the limit of small perturbations, the linear stability analysis predicts that the amplitude of each disturbed eigenmode should evolve in time as e ωt, where ω is independent of the amplitude. The solution is stable as long as ω remains negative. For a spherical shape, the Legendre polynomials are eigenmodes and are therefore uncoupled, but they are coupled otherwise. Perturbing the shape in terms of Legendre polynomials is computationally convenient, and the coupling remains weak in many cases, as will be seen later in the presentation of our numerical results. The advantage of such a numerical experiment is that it allows us to use the method presented in § 3.1, starting from a different initial state than the spherical drop at rest. The perturbation r′ must (a) be small compared to rs, and (b) not modify the volume of the drop. The condition of volume conservation cannot be a priori imposed on the coefficients ak, so we rescale the perturbed shape to obtain the desired volume. Physically, we expect the high-wavenumber modes to be dominated by surface tension, and therefore only tested perturbation modes up to k = 6. For simplicity, it is preferable to disturb single Legendre modes, rather than several modes at once.

248 E. Lac and G. M. Homsy

0 0.1 0.2 0.3 0.4 0.5

0.1

0.2

0.3

0.4

0.5

0.6

D

CaE


> **Figure 5. Drop deformation vs. electric capillary number for R = 0.1, Q = 0.1, and λ = 1; dotted line, small-perturbation theory by Taylor (1966); dashed line, second-order correction by Ajayi (1978). In top-left inset, �shows the position in the (R, Q)-diagram of figure 2. Top-right sequence: drop break-up when CaE is increased from 0.340 to 0.342 (the arrow shows the direction of the electric field, and the dashed line shows the undeformed drop).**

Unfortunately, the rescaling of the shape due to the volume correction is tantamount to a perturbation of all the Legendre modes. Nonetheless, if the disturbed mode has a small amplitude, the volume adjustment has an even smaller impact on the rest of the modes. Typically, the largest undesired perturbation is that of mode 0 (isotropic distortion), with an amplitude smaller by an order of magnitude than that of the initially disturbed mode.

4. Results: drop with the same viscosity as the medium We first give an overview of the drop behaviour when the viscosity ratio λ is set to unity. The results are presented with the deformation curve D(CaE), and systematically compared to the firstand second-order theory of Taylor (1966) and of Ajayi (1978).

4.1. Prolate drops Figure 5 shows D vs. CaE for R = 0.1 and Q = 0.1, and compares the numerical results to the asymptotic theories. The deformation curve quickly departs from the first-order theoretical prediction, but the numerical results show good agreement with secondorder corrections up to D ≈0.08. Note that the curvature keeps the sign predicted by the second-order theory (k2 > 0). We also observe here the existence of a capillary number above which no steady state is found. A slight increase of CaE past this maximum value causes a sudden elongation and break-up of the drop, as depicted by the profile sequence shown in figure 5. Figure 6 presents the linear stability analysis conducted for this case as explained in § 3.2. In figure 6(a), we show the typical relaxation of an imposed Legendre perturbation with various amplitudes. The perturbation exhibits an exponential decay. However, the characteristic damping time is slightly sensitive to the initial amplitude. This is presumably due to the fact that the perturbation is finite albeit small, and/or due to the coupling of different Legendre modes. Nevertheless, we only observed minor effects on the Legendre modes other

Axisymmetric deformation of a drop in a steady electric field 249

(a) (b)

0 5 10 15

10–1

10–2

10–3


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq061.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq062.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq063.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq064.png)

0 0.1 0.2 0.3 0.4

–1.5

–1.0

–0.5

0

CaE

k = 2

k = 4

k = 6

a′k


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq065.png)


> **Figure 6. (R, Q, λ) = (0.1, 0.1, 1): (a) typical time relaxation of a Legendre perturbation a′ k for different initial amplitudes (here, k = 2, CaE = 0.30); (b) damping coefficients ωk vs. CaE for modes 2, 4, 6 (the dashed line is an extrapolation as ω2 approaches zero).**

than that deliberately perturbed, which seems to indicate that the latter mode was dominant in the perturbed eigenmodes and that the coupling between the Legendre modes remained weak for the tested drop shapes. The damping coefficients ωk obtained for different perturbation modes (k = 2, 4, 6) were calculated for increasing values of the electric capillary number, and are plotted in figure 6(b). We find that the mode 2 (or at least eigenmodes dominated by the mode 2) is always the least damped for a given CaE, and that ω2 tends to zero in the vicinity of the maximum capillary number for which we could find a steady solution. We thus conclude that the drop is indeed unstable and that the deformation curve reaches a fold point when the capillary number is increased beyond a critical value. Furthermore, the fact that the mode 2 is expected to be the most unstable is consistent with the type of break-up observed numerically for this set of parameters. If we fix R at 0.1 and increase Q to 5, drop break-up is no longer observed, as shown in figure 7. Deformation continuously increases with CaE, and the deformation curve exhibits an inflection point as the drop steady profile evolves from a convex to a very elongated peanut-like shape. The transition between the two types of deformation coincides with the inflection point in the deformation curve. The effect of Q is emphasized in figure 8, where we have plotted the normal external electric field and the normal electric stress along the interface for R = 0.1 and two equally deformed steady drop shapes. The first case corresponds to Q = 0.1 and CaE = 0.34, the highest capillary number for which we could find a steady state. The second drop shape was obtained with Q = 5 and CaE = 0.435. Since the internal and external electric fields only depend on R and on the drop geometry, (3.2) and (3.3), these quantities are almost identical for the two drops (figure 8b). On the contrary, the electric stress on the interface depends on both R and Q. In case 1, we see from figure 8(c) that the electric pressure �pE almost vanishes at the equator (θ = π/2). Consequently, the electric field does not contribute to the normal stress balance at this point, and the drop exhibits a capillary break-up when the profile curvature approaches zero. Considering the fact that �pE is even slightly negative, one may argue that capillary break-up is slightly enhanced by the electric effects. In case 2, the permitivity ratio is high enough to generate a positive electric pressure, hence

250 E. Lac and G. M. Homsy

0.2 0.4 0.6 0.8 0

0.2

0.4

0.6

0.8

1.0

D

CaE


> **Figure 7. Deformation vs. CaE for (R, Q, λ) = (0.1, 5, 1). For explanation of lines, see figure 5.**

–4

–2

0

2

4

0 0.2 0.4 0.6 0.8 1.0

2

0

4

6

8


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq066.png)

Q = 0.1, CaE = 0.34 Q  = 5, CaE = 0.435

(a) (b)

(c)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq067.png)

Case 1: Case 2:


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq068.png)


> **Figure 8. Effect of Q at equivalent steady deformation (D ≈0.4121) for R = 0.1: (a) drop profiles; (b) external normal electric field on S; (c) normal electric stress.**

reducing the effect of capillarity according to (2.11), and allowing the drop to sustain larger deformations. Between the two cases discussed above, we observe a transition from a stable to an unstable branch as Q is decreased from 5 to 0.1 for R = 0.1 (figure 9). For Q = 3, the inflection point in the deformation curve corresponds to a vertical tangent, i.e. there is a sharp transition from the convex to the two-lobed solutions. For Q = 1.37, we observe a phenomenon of bifurcation, where in a narrow interval of electric capillary numbers (0.345 < CaE < 0.365), two steady shapes are admissible for the same CaE, one of which is convex, the other two-lobed. The latter case (R = 0.1, Q = 1.37) is studied in more detail in figures 10–14. This choice of R and Q corresponds approximately to the set of experiments denoted NN17–NN21 by Ha & Yang (2000a), for which the viscosity ratio varies from 0.080 to 0.874. We have only reported in figure 10 the results of experiments NN20 and

Axisymmetric deformation of a drop in a steady electric field 251

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0

0.2

0.4

0.6

0.8

1.0

D

CaE

Q = 0.1

Q = 1.37

Q = 3

Q = 5


> **Figure 9. Transition from unstable to stable branch by increasing Q for R = 0.1 and λ = 1. Note that the dashed line for Q = 1.37 has been drawn to guide the reader’s eye, but has not been investigated numerically.**

0 0.1 0.2 0.3 0.4 0.5

0.2

0.4

0.6

0.8

1.0


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq069.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq070.png)

D

CaE

NN20

NN21


> **Figure 10. Deformation vs. CaE for (R, Q, λ) = (0.1, 1.37, 1). Parameters correspond to the experiments NN20 and NN21 by Ha & Yang (2000a). Numbers between brackets indicate the number of lobes of the steady shape.**

NN21, where λ is close to unity (λ = 0.800 and 0.874, respectively). In the range of electric capillary numbers covered in the experiments, the measured deformations match very well the numerical results. Ha & Yang report that the drop breaks up past CaE ≈0.30, while we could reach steady spheroidal drop shapes up to CaE ≈0.36. When the upper branch is investigated by increasing CaE above 0.38, a family of steady three-lobed shapes is reached, as shown in the inset in figure 10 for CaE = 0.40.

252 E. Lac and G. M. Homsy

a

(a) (b)


> **Figure 11. Flow pattern for two admissible steady drop shapes subjected to the same electric capillary number (R = 0.1, Q = 1.37, λ = 1, CaE = 0.36).**

2a


> **Figure 12. Flow pattern for a three-lobed steady drop shape (R = 0.1, Q = 1.37, λ = 1, CaE = 0.40).**

For CaE = 0.45 and then CaE = 0.50, the solution has four and then five lobes †. Typical flow patterns inside and outside the drop are studied for the different solution families in figures 11–13. We recall that the parameter set (R, Q, λ) = (0.1, 1.37, 1) corresponds to the type PRA, meaning the outer flow runs from equator to poles in Taylor’s smalldeformation theory. Figure 11 shows the flow streamlines at steady state for the two solutions coexisting at CaE = 0.36. Since the drop exhibits symmetry with respect to the equatorial plane (the vertical dashed line in figure 11), the drop profiles are here cut in half. While the flow pattern around the spheroidal drop (figure 11a) is quite similar to that predicted by the small-deformation theory, the streamlines around the two-lobed drop (figure 11b) reveals the existence of a recirculation eddy, which prevents drop break-up as a result of the viscous normal stress generated by the flow at the equator. When CaE is increased to 0.40 (figure 12), a third lobe appears at the drop centre. This drop shape is sustained by the appearance of a secondary external vortex and a third toroidal vortex inside the drop which together prevent the disintegration of the drop into three smaller droplets. For CaE = 0.45 (figure 13), the drop exhibits four lobes and contains twice as many internal recirculation eddies. The external flow pattern becomes more complex, but overall, the destabilizing capillary effects expected at each neck between two lobes is again balanced by the electrohydrodynamic flow. Figure 14 summarizes the cascade of steady shapes obtained when CaE is increased from 0.36 to 0.45. It is remarkable that similar internal eddies occur as the drop elongates to more complex multi-lobe shapes.

† For these multi-lobe steady shapes, the definition of the drop breadth l2 used to measure the deformation D is ambiguous; here, we have chosen the equatorial diameter for l2, but the parameter D is no longer a good measurement of the shape distortion, which explains the peculiar variation of the deformation curve in figure 10.

Axisymmetric deformation of a drop in a steady electric field 253

2a


> **Figure 13. Flow pattern for a four-lobed steady drop shape (R = 0.1, Q = 1.37, λ = 1, CaE = 0.45).**

2a

CaE = 0.36

CaE = 0.36

CaE = 0.40

CaE = 0.45


> **Figure 14. Eddy structure as CaE increases, for (R, Q, λ) = (0.1, 1.37, 1).**

10 20 30 40 50 0

0.2

0.4

0.6

0.8

1.0

D

CaE


> **Figure 15. Deformation vs. CaE for (R, Q, λ) = (10, 0.04, 1). Inset: drop profile for CaE = 30.**


> **Figure 15 shows the deformation curve obtained for (R, Q) = (10, 0.04). Like the different cases investigated up to now, this set of parameters leads to a prolate drop of the type PRA, but this time the second-order correction coefficient k2 defined by (2.15) is negative. For prolate drops (k1 > 0), this means that Taylor’s first-order theory tends to overestimate the drop deformation. Again, the numerical results show that Ajayi’s second-order theory predicts the drop deformation well up to D ≈0.1 (CaE ≈1). The**

254 E. Lac and G. M. Homsy

0 0.1 0.2 0.3 0.4 0.5 0.6

0.2

0.4

0.6

0.8

D

CaE


> **Figure 16. Deformation vs. CaE for (R, Q, λ) = (0.04, 50, 1). The sequence shows the drop behaviour when CaE is increased from 0.46 to 0.47.**

drop deformation evolves smoothly, and more slowly as CaE increases. Consequently, break-up is never observed. The drop reaches a deformation of the order of 0.9 for CaE = 50. We now turn to the prolate type PRB, where the external flow around the drop runs from the tips to the equator of the drop. The values (R, Q) = (0.04, 50) and (100, 0.1) were chosen to explore the two sub-types PR+ B and PR− B, shown in figures 16 and 17, respectively. In the first case (figure 16), no steady shape could be found above CaE = 0.46. Past this value, the drop exhibits pointed tips, as was already observed by Sherwood (1988). At some point, as CaE increases, the electric pressure dominates the capillary effect and the pressure at the poles becomes negative, despite the high curvature in this region. However, we do not know if this awkward pressure distribution is responsible for drop break-up. One might assert that the formation of very high-curvature tips suggests the onset of cones, or of tip streaming, should the flow run from equator to pole and thus allow the ejection of droplets. The flow direction, being in the reverse direction, makes this impossible. When (R, Q) = (100, 0.1), the second-order asymptotic theory predicts that the coefficient k2 is negative. As already observed for (R, Q) = (10, 0.04) (type PR− A, figure 15), the curvature of the deformation curve remains negative, leading to smaller and smaller variations of deformation as CaE increases (figure 17). The drop is stable under any electric capillary number, and break-up is never observed. However, if qualitatively similar deformation curves are obtained for the types PR− A and PR− B (figures 15 and 17), the PR− A drops are significantly more distorted by the electric field, because the induced equator-to-pole flow tends to contribute to the drop deformation. If we maintain k2 negative but choose (R, Q) = (10, 0.1) or (R, Q) = (100, 0.01), such that the electrohydrodynamic effects are cancelled because RQ = 1, we find the same type of deformation curves as in figures 15 and 17. This proves that when k2 < 0, the drop behaviour is essentially due to the balance between surface tension and the electric pressure; as the electric field is increased, these two quantities evolves in such a way that equilibrium is always possible, which leads to the observed stability.

Axisymmetric deformation of a drop in a steady electric field 255

20 40 60 80 100 0

0.1

0.2

0.3

0.4

0.5

0

0.05

0.10

D

CaE

1 2 3


> **Figure 17. Deformation vs. CaE for (R, Q, λ) = (100, 0.1, 1).**

0 0.05 0.10 0.15 0.20 0.25 0.30

–0.5

–0.4

–0.3

–0.2

–0.1

0

D

CaE


> **Figure 18. Deformation vs. CaE for (R, Q, λ) = (10, 2, 1). The sequence shows drop break-up when CaE is increased from 0.297 (last point on the curve) to 0.300.**

4.2. Oblate drops The oblate drop domain offers fewer sub-categories because there is no ambiguity in the flow direction, which always runs from pole to equator. We chose to study two typical cases, corresponding to a negative and a positive coefficient k2, respectively. In figure 18, R = 10 and Q = 2. We find for this case a fold point around CaE = 0.297, and the stability analysis shows that the mode 2 is again the first unstable mode. Here, break-up is due to a strong negative electric pressure at the poles, where presumably the capillary effect would be too weak to cause break-up due to the locally low mean curvature. At these points (θ = 0, π), we find 2κ∗/CaE ≈1.48, and �p∗ E ≈−4.65. As CaE increases, κ∗decreases and eventually becomes negative, while �p∗ E keeps increasing. Consequently, the drop breaks up in the fashion described by the profile sequence inserted in figure 18.

256 E. Lac and G. M. Homsy

0 0.5 1.0 1.5 2.0 –1.0

–0.8

–0.6

–0.4

–0.2

0

D

CaE


> **Figure 19. Deformation vs. CaE for (R, Q, λ) = (0.5, 20, 1). Inset: steady drop shape for CaE = 2.**

The drop behaviour is radically different when (R, Q) = (0.5, 20), as shown in figure 19. Here, break-up is no longer observed. At large capillary numbers, the drop becomes like a flat disk with both faces almost homogeneously charged. In such a case, all the physical quantities undergo sharp variations in the vicinity of the equator.

4.3. Summary In the particular case where the drop and the suspension medium have equal dynamic viscosities, we have identified various behaviours by exploring the parameter map (R, Q) as given by the firstand second-order theories. For prolate drops (k1 > 0), four sub-types are a priori expected, depending on the flow direction (PRA and PRB) and the sign of the coefficient k2. When k2 is negative, our numerical results show that the deformation curve D vs. CaE remains concave even when the small-deformation theory is no longer valid. A stable steady drop shape is found for all capillary numbers. The drop deformation is mainly due to the electric pressure, while the induced flow only tends to reduce or enhance deformation depending on its direction. When k2 is positive, the drop exhibits pointed tips past a maximum electric capillary number in the somewhat counterintuitive case where the flow runs from pole to equator (type PR+ B). The most interesting phenomena are observed for the type PR+ A; here, for moderate values of R, electric and viscous stresses are strongly coupled. A positive electric pressure at the equator appears to be a stabilizing factor which allows the drop to reach highly elongated shapes. Furthermore, capillary break-up of elongated drops can be prevented by the appearance of electrically induced recirculation eddies inside and outside the drop. For oblate drops (k1 < 0), the flow direction is always the same (pole to equator), and does not allow competition between capillary and electric pressures. Two main tendencies are observed. First, when k2 > 0 (class OB+), the drop is stable for all values of CaE, as observed for the class PR−. In this case, the drop tends to behave like a flat disk with oppositely charged faces. Second, when k2 < 0, the drops breaks up due to the action of a negative electric pressure in the flattened regions near the poles. Interestingly, it seems that the drop tends to be stable when the coefficients k1 and k2 have opposite signs. This criterion, based on the second-order asymptotic theory

Axisymmetric deformation of a drop in a steady electric field 257

0 0.1 0.2 0.3 0.4 0.5

0.2

0.4

0.6

0.8

1.0


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq071.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq072.png)

D

CaE


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq073.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq074.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq075.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq076.png)


> **Figure 20. Deformation vs. CaE for (R, Q) = (0.1, 0.1) and different viscosity ratios. The inset profiles show the last steady state observed before break-up. The firstand second-order predictions correspond to λ = 1.**

of Ajayi (1978), is certainly not accurate. In particular, it is expected to fail near the regions where k2 vanishes, since the nonlinear drop deformation is then determined by O(Ca3 E) or higher-order terms. Nonetheless, it appears to be an easy way to predict the general tendency of the drop behaviour. When RQ > 1 (types PRB and OB), the drop may exhibit at some point a plausible instability to three-dimensional perturbations since the induced charge distribution is such that the resulting electric dipole moment is opposite to the imposed field (Quincke effect, e.g. Rivette & Baygents 1996; Feng 2002; Liao et al. 2005). Obviously, this case is out of reach for the present method, but it is noteworthy that electrorotation (Ha & Yang 2000b) and shape oscillations of a drop in a DC field (Sato et al. 2006) have indeed been observed experimentally.

5. Effect of viscosity contrast The previous section has highlighted the diversity of drop behaviours observed for a wide range of parameters R and Q when λ is held fixed at unity. The effect of viscosity ratio has so far been widely ignored, perhaps due to its weak influence on the small-deformation theory (§ 2.5). Yet, this parameter is known to have a great influence in problems involving drops in mechanically driven flows (Stone 1994). The various behaviours presented in § 4 are the result of the strongly nonlinear interaction between an electric forcing and the motion of two viscous liquids; the viscosity contrast is then expected to play an important role. To our knowledge, the only study dedicated to the effect of λ was conducted by Feng & Scott (1996). These authors only considered two sets of parameters (R, Q) – one prolate and one oblate drop – and concluded that an increasing λ merely shifted the critical field strength to a lower value. However, we shall see below that the viscosity contrast can drastically modify the drop deformation and stability. Figure 20 is the counterpart of figure 5, where R = Q = 0.1, but with varying values of λ. (Note that in figure 20 and in the following graphs illustrating the effect of viscosity contrast, the firstand second-order predictions are, for clarity, only shown for λ = 1, since the effect of λ on these curves is extremely small.) As predicted by the

258 E. Lac and G. M. Homsy


> **Figure 21. Drop profile at steady state; (R, Q, λ) = (0.1, 0.1, 20), CaE = 0.45. The arrow shows the field direction.**

asymptotic theory, the drop deformation increases with the viscosity ratio for a given capillary number. Overall, λ has a weak influence on the deformation as long as D does not exceed 0.2, approximately. For larger drop deformations, however, λ plays a key role in drop stability. When the drop viscosity decreases, the primary fold point is postponed to larger capillary numbers and deformations, and the break-up mode is the same as that observed for λ = 1. However, we find that a sufficiently viscous drop does not break up but undergoes the cascade of deformations described in § 4 for (R, Q, λ) = (0.1, 1.37, 1) (figures 10–14). For large electric fields (typically here, CaE = 0.45), the drop becomes almost cylindrical, with bulbous ends (figure 21). The cylindrical part is practically devoid of electric charge because the normal electric field En is extremely small. For λ = 50, we find that the drop deformation is almost the same as that obtained when λ = 20, except that the fluid velocities at steady state are naturally smaller. Overall, the drop can achieve large deformations when the viscosity of one phase becomes dominant over the other. In this case, the observed drop behaviour can be qualitatively explained by the analysis of Saville (1970) on the stability of viscous cylinders in an electric field. Indeed, Saville demonstrated that electric stresses may stabilize the interface if the inner fluid is viscous enough, and that oppositely, “if the viscosity of the outer fluid is dominant, then instability always exists for disturbances with small wavenumbers”. The same trend is found for (R, Q) = (0.1, 1.37), where stability is lost for λ < 1. For the same parameters, Ha & Yang (2000a) report break-up for their systems NN17–19, where λ ranges between 0.080 and 0.266. However, they observed no effect of λ on the critical capillary nor the drop deformation before break-up, as is the case here. We have not pursued the investigation for viscosity ratios slightly below unity (systems NN20 and NN21), but it is possible that even a small viscosity contrast in favour of the external fluid may be responsible for drop break-up. For a prolate drop with RQ > 1 (type PRB), the effect of λ is inverted, i.e. the drop deformation decreases as λ increases. This is clearly visible in figure 22, for (R, Q) = (0.04, 50). When the drop relative viscosity increases, the drop ultimately breaks up past a critical value of CaE, in the fashion depicted by the sequence in figure 16 for λ = 1. However, if the drop viscosity is low enough, the drop no longer breaks up but reaches an elongated shape with very high-curvature tips (figure 22, inset). The typical flow pattern obtained for the low-viscosity elongated drops is plotted in figure 23 for CaE = 0.6 and λ = 0.05. The inset shows the very fine discretization needed to capture the high gradients of all the physical quantities at the tips. All the charges tend to accumulate at the tips, where again the electric pressure dominates capillarity. This is why the normal stress balance cannot be achieved unless the inner fluid is much less viscous than the outer fluid. For the type PR− B, decreasing the viscosity ratio at constant capillary number increases the drop deformation (and vice versa), but the deformation curve obtained for λ = 1 (figure 17) is only shifted up or down when λ ̸= 1. The same comment can be made for (R, Q) = (10, 0.04) (type PR− A), save that this time, the deformation increases with λ. This confirms that when the inner liquid is poorly conducting and

Axisymmetric deformation of a drop in a steady electric field 259

0 0.1 0.2 0.3 0.4 0.5 0.6

0.2

0.4

0.6

0.8

D

CaE


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq077.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq078.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq079.png)


> **Figure 22. Deformation vs. CaE for (R, Q) = (0.04, 50) and different viscosity ratios.**

2a


> **Figure 23. Flow pattern at steady state for (R, Q, λ) = (0.04, 50, 0.05) and CaE = 0.60. Inset: discretization of the profile at the tips, κ∗≈63.2 (m = 150).**

Q is such that the drop deforms into a prolate shape (typically, R > 2, Q < 0.3, so that k1 > 0 and k2 < 0), the viscous stresses induced by the fluid motion have only a weak effect on the drop stability, as was already observed for λ = 1. Finally, we investigate the effect of λ on oblate drops. When (R, Q) = (10, 2) and λ = 1 (type OB−), we have seen that drop break-up was at some point triggered by a high negative electric pressure at the poles, expelling the inner liquid toward the equator (inset in figure 18). In figure 24, we see that the drop reaches a fold point whatever the value of λ. Decreasing the viscosity ratio only postpones drop break-up to a higher electric capillary number. It is not surprising that the break-up mode is the same for the various values of λ since both the tangential stress and fluid velocity are small around the equator, where the dominant destabilizing effect remains the negative electric pressure. We have not investigated the effect of λ above 20 nor below 1/20, but we believe the results shown in figure 24 reflect well the drop behaviour when one of the viscosities dominates the other.

6. Discussion We have studied numerically the deformation and stability of a suspended drop in a steady electric field in the limit of creeping flow approximation. Both the electric and the velocity field were calculated with a boundary integral technique which exactly satisfies the condition of vanishing perturbation infinitely far from the drop, where the outer liquid is at rest and the electric field uniform. When the drop and the outer phase have the same dynamic viscosity, the drop exhibits a large range of different

260 E. Lac and G. M. Homsy

0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40

–0.5

–0.4

–0.3

–0.2

–0.1

0

D

CaE


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq080.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq081.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq082.png)


> **Figure 24. Deformation vs. CaE for (R, Q) = (10, 2) and different viscosity ratios.**

behaviours when the resistivity ratio R and the permittivity ratio Q vary. For a given set of parameters (R, Q), the amplitude of the normal and tangential parts of the external electric field remains, for moderately deformed drops, of the order of Taylor’s solution:


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq083.png)

Accordingly, the electric pressure at the poles (where Et = 0) and at the equator (where En = 0) is of the order of


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq084.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq085.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq086.png)

(1 + 2R)2 , (6.2)

while the typical tangential electric stress and velocity are


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq087.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq088.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq089.png)


![Equation](images/2007_lac_homsy_leaky_dielectric_drop_eq090.png)

When the drop is much more conducting than the suspension medium, i.e. R ≪1, the electrohydrodynamic effects become vanishingly small, as already pointed out by Feng & Scott (1996). Consequently, the viscosity ratio λ hardly affects the drop behaviour. Furthermore, all the electrostatic effects tend to depend on the properties of the outer phase only, meaning that the permittivity ratio has no influence whatsoever. Accordingly, the firstand second-order theories each predict a deformation curve independent of R, Q and λ. In this situation, we find that break-up occurs past a typical deformation D = 0.3, corresponding to a dimensionless drop length of about 3. The critical capillary number is always around CaE = 0.22, which is consistent with the observations of Ha & Yang (2000a) for their systems NN1–16. When a steady state is reached, the fluids are almost at rest (depending on how small R is), and surface tension balances the positive electric pressure. At the onset of break-up, the electric pressure at the drop tips increases faster than the curvature, meaning that surface tension is no longer able to keep the drop together. The process itself is dynamic and is therefore influenced by λ, but we have not specifically investigated this point. We refer the interested reader to the study of Dubash & Mestel (2007) on

Axisymmetric deformation of a drop in a steady electric field 261

(a) (b)


> **Figure 25. Prolate drop break-up: (a) (R, Q, λ) = (0.01, 12, 0.01) when CaE is increased from 0.22 (first profile, steady state) to 0.225; (b) (R, Q, λ) = (0.04, 3, 1), CaE = 0.27 (initially spherical drop); inset: detail of the discretization (m = 200).**

the behaviour of a perfectly conducting drop in a viscous medium. For low-viscosity drops, the instability is characterized by the appearance of fast evolving fingers at the drop tips, as observed by Torza et al. (1971) and Ha & Yang (2000a) (system NN13), as well as in our simulations (figure 25a). If the drop is more conducting than the suspension medium by only one order of magnitude (R ∼0.1), the typical stresses given by (6.2) and (6.3) indicate that in the prolate regime (i.e. moderate Q), the drop deformation is still mostly driven by the electric pressure at the poles, but also that the tangential electric stress (and thus fluid motion) has a substantial role. For moderate deformations, �p∗ E(π/2) ∼(Q −1)R2 + O(R3) and is therefore negligible. Should the drop become very elongated, however, the electric field around the drop equator tends to be undisturbed by the drop extremities, and E∗ t ∼1 in this area. Consequently, �p∗ E(π/2) becomes of the order of (Q−1)/2, independent of R. For such distorted shapes, the sign and magnitude of the electric pressure at the equator is then directly given by Q. Consequently, for λ = 1, we find that the drop can reach elongated shapes when Q > 1, because the electric pressure is positive and reduces the capillary pressure around the equator. If Q < 1, oppositely, the electric pressure tends to enhance the effect of surface tension at the equator, which may help the drop collapse in two main droplets. Nevertheless, the onset of capillary break-up of a drop in elongation is a dynamic process that depends on the rate of extension of the drop, which cannot be predicted by the sign of �pE only, owing to the strong coupling between the electric field and the fluid motion. In this context, the viscosity contrast was found to have a great influence on the drop stability. If the inner liquid is sufficiently viscous, the collapse of the drop may be prevented by the appearance of a secondary flow (e.g. figure 11b), for all values of Q. On the other hand, the drop always breaks up if λ is small enough. Furthermore, at a given viscosity ratio, the drop is stabilized if the electric field is able to induce a sufficiently strong recirculation flow. Indeed, we observe drop break-up in figure 25(b) for (R, Q, λ) = (0.04, 3, 1), whereas the drop is stable under elongated shapes when R is increased to 0.1 (figure 9). When R2 ≫1, i.e. when the drop resistivity is significantly larger than that of the medium (R > 10, typically), �p∗ E(0) ∼−Q(1 −R−1), �p∗ E(π/2) ∼(Q −1)(1 −R−1), and T ∗ E ∼|Q −(1 + Q)/R|, at least when Q < O(R). The importance of the

262 E. Lac and G. M. Homsy

electrohydrodynamic phenomena is then determined by the value of Q. When the drop is prolate, Q is fairly small, and we find that viscous effects are indeed weak. This case corresponds to a negative coefficient k2, and the drop is stable under all capillary numbers, mainly as a result of the equilibrium between capillary and electric pressures. By its nature, the oblate regime is characterized by important electrohydrodynamic effects, since the drop is always prolate when these effects vanish (e.g. R ≪1, RQ = 1 . . .). Most likely, QR2 > 1, and the drop is deformed under the effect of a negative electric pressure near the poles. Depending on the electric property ratios R and Q, this pressure can either lead the drop to burst by expelling the inner liquid out of the drop centre, or progressively deform it into a flat conducting disk (see insets in figures 18 and 19, respectively). This trend pertains for different viscosity ratios. In the second case, the outer electric field in the flattened region tends to its undisturbed value E∞; consequently, E∗ n ≈1, E∗ t ≈0, and the dimensionless electric pressure tends to the constant value (1 −QR2)/2. Numerically, we find that the diameter-to-thickness ratio of these flattened drops evolves as Ca3/2 E . As a result of the axisymmetry of the system, all physical quantities (surface charge and velocities, in particular) undergo extremely high gradients in the vicinity of the equator. However, it is probable that the drop is unstable to three-dimensional perturbations before it flattens to a disk. Indeed, Ha & Yang (2000b) reported that an oblate drop may exhibit a rotational motion around an axis orthogonal to the electric field above a critical field strength. It would be interesting to see whether a non-axisymmetric disturbance (spherical harmonics) around the axisymmetric steady shape would be able to trigger new modes of deformations or would decay as was observed here for Legendre modes. Finally, we conclude with a remark on the asymptotic theories of Taylor (1966) and of Ajayi (1978). Our results show that the first-order theory of Taylor no longer predicts the drop deformation for D > 0.02, whereas the second-order correction of Ajayi remains satisfactory up to D ≈0.1, typically. The parameter D is extremely sensitive to small distortion from sphericity; for instance, a prolate deformation of 0.02 corresponds approximately to dimensionless length and breadth of 2.055 and 1.972, respectively. This means that experimentally, identifying accurately the original slope of the deformation curve is extremely difficult. In many experiments (e.g. Torza et al. 1971; Vizika & Saville 1992), the drop deformation is measured up to D ≈0.1, and a linear fit of the whole data set is used to compare experiment and theory. Except in the particular case where k2 is very small, this method cannot lead to a relevant comparison since the drop has already entered a (weakly) nonlinear regime in this range of deformations. A similar remark can be made about the method used by Ha & Yang (2000a, b), which consists of deducing the surface tension (or more generally the term εε0/γ ) from the plot D vs. aE2 ∞so that the deformation curve matches Taylor’s theory for small CaE. This method can lead to incorrect conclusions, as shown in figure 26. First, figure 26(a) shows a rescaling of the capillary number in Ha & Yang’s results in order to fit Ajayi’s theory rather than Taylor’s, keeping the assumed values (R, Q, λ) = (0.1, 1.37, 1); we observe that the experimental data diverge from our numerical results for large deformations, which might support the conclusion that the leaky dielectric model is inappropriate. However, in figure 26(b), we perform the same rescaling, assuming this time (R, Q, λ) = (0.04, 1, 1); the numerical results appear in much better agreement, and also predict break-up. This indicates the need for independent and accurate measurements of all physical properties, as well as a comparison between experiments and nonlinear theories, when assessing the

Axisymmetric deformation of a drop in a steady electric field 263

0 0.1 0.2 0.3 0.4 0 0.1 0.2 0.3 0.4

0.1

0.2

0.3

0.4 (a) (b)

*

D

CaE CaE

NN20 NN21


> **Figure 26. Rescaling of CaE in the experimental data of Ha & Yang (2000a) to match Ajayi’s theory: (a) by a factor of 0.9, with (R, Q, λ) = (0.1, 1.37, 1); (b) by a factor of 0.76, with (R, Q, λ) = (0.04, 1, 1). Solid line, our numerical results; *, drop break-up.**

validity of the leaky dielectric model. Consequently, we believe it would interesting for future work to include in the asymptotic theory O(Ca2 E) effects such as surface charge convection, in order to propose as physical a model as possible.

This work was supported by DOE, Office of Basic Energy Sciences, and by the NASA Microgravity Fluid Mechanics Program. E. L. is grateful to Costas Pozrikidis for authorizing the use of the library BEMLIB. Helpful discussions with L. Gary Leal, Todd Squires and J. D. Sherwood were also greatly appreciated.


## References

Ajayi, O. O. 1978 A note on Taylor’s electrohydrodynamic theory. Proc. R. Soc. Lond. A 364, 499–507. Allan, R. S. & Mason, S. G. 1962 Particle behaviour in shear and electric fields. i. deformation and burst of fluid drops. Proc. R. Soc. Lond. A 267, 45–61. Baygents, J. C., Rivette, N. J. & Stone, H. A. 1998 Electrohydrodynamic deformation and interaction of drop pairs. J. Fluid Mech. 368, 359–375. Brazier-Smith, P. R. 1971 Stability and shape of isolated and pairs of water drops in an electric field. Phys. Fluids 14, 1–6. Cristini, V., Blawzdziewicz, J. & Loewenberg, M. 2001 An adaptive mesh algorithm for evolving surfaces: Simulations of drop breakup and coalescence. J. Comput. Phys. 168, 445–463. Dubash, N. & Mestel, A. J. 2007 Behaviour of a conducting drop in a highly viscous fluid subject to an electric field. J. Fluid Mech. 581, 469–493. Feng, J. Q. 1999 Electrohydrodynamic behaviour of a drop subjected to a steady uniform electric field at finite electric Reynolds number. Proc. R. Soc. Lond. A 455, 2245–2269. Feng, J. Q. 2002 A 2D electrohydrodynamic model for electrorotation of fluid drops. J. Colloid Interface Sci. 246, 112–121. Feng, J. Q. & Scott, T. C. 1996 A computational analysis of electrohydrodynamics of a leaky dielectric drop in an electric field. J. Fluid Mech. 311, 289–326. Ha, J.-W. & Yang, S.-M. 2000a Deformation and breakup of Newtonian and non-Newtonian conducting drops in an electric field. J. Fluid Mech. 405, 131–156. Ha, J.-W. & Yang, S.-M. 2000b Electrohydrodynamics and electrorotation of a drop with fluid less conductive than that of the ambient fluid. Phys. Fluids 12, 764–772. Landau, L. D. & Lifshitz, E. M. 1984 Course of Theoretical Physics, Vol. 8: Electrodynamics of Continuous Media, 2nd edn. Butterworth–Heinemann.

264 E. Lac and G. M. Homsy

Liao, G., Smalyukh, I. I., Kelly, J. R., Lavrentovich, O. D. & J´akli, A. 2005 Electrorotation of colloidal particles in liquid crystals. Phys. Rev. E 72, 031704. Melcher, J. R. & Taylor, G. I. 1969 Electrohydrodynamics: A review of the role of interfacial shear stresses. Annu. Rev. Fluid Mech. 1, 111–146. Miksis, M. J. 1981 Shape of a drop in an electric field. Phys. Fluids 24, 1967–1972. O’Konski, C. T. & Thacher, H. C. 1953 The distortion of aerosol droplets by an electric field. J. Phys. Chem. 57, 955–958. Pozrikidis, C. 1992 Boundary Integral and Singularity Methods for Linearized Viscous Flow. Cambridge University Press. Rivette, N. J. & Baygents, J. C. 1996 A note on the electrostatic force and torque acting on an isolated body in an electric field. Chem. Engng Sci. 51, 5205–5211. Sato, H., Kajia, N., Mochizukib, T. & Mori, Y. H. 2006 Behavior of oblately deformed droplets in an immiscible dielectric liquid under a steady and uniform electric field. Phys. Fluids 18, 127101. Saville, D. A. 1970 Electrohydrodynamic stability: Fluid cylinders in longitudinal electric fields. Phys. Fluids 13, 2987–94. Saville, D. A. 1997 Electrohydrodynamics: the Taylor-Melcher leaky dielectric model. Annu. Rev. Fluid Mech. 29, 27–64. Sherwood, J. D. 1988 Breakup of fluid droplets in electric and magnetic fields. J. Fluid Mech. 188, 133–146. Stone, H. A. 1994 Dynamics of drop deformation and breakup in viscous fluids. Annu. Rev. Fluid Mech. 26, 65–102. Stone, H. A., Lister, J. R. & Brenner, M. P. 1999 Drops with conical ends in electric and magnetic fields. Proc. R. Soc. Lond. A 455, 329–347. Taylor, G. I. 1964 Disintegration of water drops in an electric field. Proc. R. Soc. Lond. A 291, 159–166. Taylor, G. I. 1966 Studies in electrohydrodynamics. I. The circulation produced in a drop by an electric field. Proc. R. Soc. Lond. A 280, 383–397. Torza, S., Cox, R. G. & Mason, S. G. 1971 Electrohydrodynamic deformation and burst of liquid drops. Phil. Trans. R. Soc. Lond. A 269, 295–319. Vizika, O. & Saville, D. A. 1992 The electrohydrodynamic deformation of drops suspended in liquids in steady and oscillatory fields. J. Fluid Mech. 239, 1–21.

