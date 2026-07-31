
# Two-phase electrohydrodynamic simulations using a volume-of-fluid approach


## G. Tomar a, D. Gerlach b, G. Biswas a,*, N. Alleborn b, A. Sharma c, F. Durst b, S.W.J. Welch d, A. Delgado b

a Department of Mechanical Engineering, Indian Institute of Technology Kanpur, Kanpur, UP 208 016, India b Institute of Fluid Mechanics, University of Erlangen-Nuremberg, Cauerstr. 4, 91058 Erlangen, Germany c Department of Chemical Engineering, Indian Institute of Technology Kanpur, Kanpur, UP 208 016, India d Department of Mechanical Engineering, University of Colorado at Denver and Health Sciences Center, Denver, CO 80217, USA

Received 9 May 2007; received in revised form 23 August 2007; accepted 3 September 2007 Available online 14 September 2007

Abstract

A numerical methodology to simulate two-phase electrohydrodynamic flows under the volume-of-fluid paradigm is proposed. The electric force in such systems acts only at the interface and is zero elsewhere in the two fluids. Continuum surface force representations are derived for the electric field force in a system of dielectric–dielectric and conducting–conducting fluids. On the basis of analytical calculations for simple flow problems we propose a weighted harmonic mean interpolation scheme to smoothen the electric properties in the diffused transition region (interface). It is shown that a wrong choice of interpolation scheme (weighted arithmetic mean) may lead to a transition region thickness dependent electric field in the bulk. We simulate a set of problems with exact or approximate analytical solutions to validate the numerical model proposed. A coupled level set and volume-of-fluid (CLSVOF) algorithm has been used for simulations presented here. �2007 Elsevier Inc. All rights reserved.

Keywords: Electrohydrodynamics; Volume-of-fluid; Continuum method; Surface force

1. Introduction

Electrohydrodynamics is the term used for the hydrodynamics coupled with electrostatics. In presence of an electric field, the molecules of a fluid get polarized and for a homogeneous dielectric medium, the charge because of such a polarization appears only at the surface. In addition, charged ions/free-electrons in the fluid (or from the electrode) migrate to the surface of the fluid. The charge thus accumulated at the interface because of the electric current in the two fluids across the interface is known as free charge and charge that appears because of the polarization is called bound charge [1]. The force on these charges in presence of

0021-9991/$ - see front matter �2007 Elsevier Inc. All rights reserved. doi:10.1016/j.jcp.2007.09.003

* Corresponding author. Tel.: +91 512 2597656; fax: +91 512 2590534. E-mail address: gtm@iitk.ac.in (G. Biswas).

Available online at www.sciencedirect.com

Journal of Computational Physics 227 (2007) 1267–1285

www.elsevier.com/locate/jcp

electric field yields a surface force felt by the fluid particles enclosing these charges. For perfectly dielectric fluids, namely, in the absence of any free-charge ions/electrons, the surface force is normal to the interface whereas, for conducting fluids the surface force has a tangential component too [2–4]. The electric surface force can have a significant influence on the stability of the surface of the fluid. This has been realized as an effective tool for imitating gravity [5], mixing and separation of fluids [6,7], breakup of bubbles and drops [8–10], generating droplets and bubbles from a nozzle on demand [11,12] and many more applications in space research and industry. To increase the efficiency of boiling heat exchangers especially under micro-gravity or free space conditions an electric field can be applied [13–16]. The size of a gas bubble, injected into a liquid, can be strongly influenced using an electric field [17] and thus mixing/reaction which strongly depend on the size of the bubble can be controlled. An efficient control of the above processes requires an in-depth understanding of the response of a two-fluid system in an externally applied electric field. The effect of electrohydrodynamics in such systems has been extensively studied both theoretically and experimentally [2,4,8,18–24]. Theoretical studies are limited to gross simplification of the actual problem due to extremely complicated analysis of the coupled fluid-mechanical and electrical problem. Numerical simulations of electrohydrodynamic flows have been performed. In most of the numerical investigations, boundary element method (BEM) has been used to solve for the electric field and the fluid flow assuming Stokes or inviscid flow [9,10,17,25–27]. Using finite element method (FEM) Basaran et al. [28–31] have simulated drop deformations of pendant and sessile conducting drops due to electric field and have characterized the equilibrium drop shapes as well as the morphological evolution. Fernandez et al. [32], using a front-tracking method [33], studied the effect of electrostatic forces on the distribution of drops in a channel flow. Zhang and Kwok [34] performed numerical simulations of electrohydrodynamic driven deformations in a 2D drop using a Lattice Boltzmann method. The leaky dielectric theory with diffused interfaces was used to model the electric field in the fluids separated by a diffused interface. Very recently, Welch and Biswas [35] employed a CLSVOF algorithm to simulate film boiling of water under the effect of electrostatic forces. In this paper, we present a methodology for an accurate numerical simulation of two-fluid flows influenced by electric field forces. The proposed numerical method is applicable to systems of two fluids that can be approximated as either perfectly dielectric fluids or conducting fluids. For this class of problems a method has been developed to incorporate the electric forces at the interface in the paradigm of front-capturing methods such as volume-of-fluid or level set, where the interface is tracked through an Eulerian mesh and the interface is represented by a transition region of finite thickness to smoothen the jump in fluid properties. The proposed method is based on the work of Brackbill et al. [36], in which a model for the surface tension force has been successfully developed. This model is called continuous surface tension model and is a standard method to include surface tension in volume-of-fluid and level-set methods. An analogous approach to include forces at the interface because of an applied electric field is presented here. The method has been validated using a set of problems having exact or approximate analytical solutions. The paper is organized as follows. Section 2 lays down the governing equations for the electric field and the electric forces acting at the interface for a system of two perfectly dielectric mediums as well as two conducting mediums. A general volume-of-fluid methodology for the solution of the electric field and the modeling of the electric field surface force as a volume force is described in Section 3. Results of test cases for the model suggested using a coupled level set and volume-of-fluid method are presented in Section 4. Important conclusions are stated in Section 5.

2. Formulation

The essential electrical laws are summarized below. In electrohydrodynamics, dynamic currents are small and therefore the magnetic induction effects can be ignored. Thus, the electric field intensity E is irrotational ($ · E = 0). The Gauss law in a dielectric material of relative permittivity �can be written in terms of the electric displacement (D = ��0E) as,

$ �D ¼ qv; ð1Þ

1268 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

where qv is the volume-charge density and �0 = 8.85 · 10�12 C/Vm is the permittivity of the vacuum. The charge conservation equation is given by:

$ �J þ Dqv Dt ¼ 0; ð2Þ

where J = rE is the current density due to conduction, D/Dt = o/ot + v Æ $ is the material derivative and r is the electrical conductivity of the medium. v is the velocity vector of the fluid. The tangential component of the electric field intensity is continuous across the interface, that is n · iEi = 0, where iÆi represents a jump (medium-1–medium-2) across the interface and n is the unit normal vector at the interface (Fig. 1). The normal component of the electric displacement vector is discontinuous at the interface and the jump is given by,

n �kDk ¼ qs; ð3Þ

where qs is the free charge per unit surface area. The conservation equation for the free charge at the interface is given by,

kJ �nk þ $s �K ¼ ðn �vÞkqvk �oqs ot �vs �$sqs þ qsn �ðn �$Þv; ð4Þ

where K is the surface current density, vs is the interfacial velocity and $s is the surface gradient. The terms on the left hand side of the above equation represent the jump in the current density due to conduction across the interface along the normal and surface conduction, respectively. The terms on the right hand side, from left to right, represent the jump in the convectional charge current in the bulk across the interface, temporal derivative of the interfacial charge density, surface convection of the interfacial charge density and the last term includes the effect of stretching of the interface [2,4], respectively. A two-fluid system can be broadly categorized as, dielectric–dielectric, dielectric–conducting and conducting–conducting based on the electrical conductivities and permittivities of the participating fluids. The freecharge density can be shown to decay in the neighborhood of a given fluid particle with the relaxation time tE = �0�/r using the charge conservation equation (Eq. (2)) [2]. The viscous time scale of the fluid motion is given by tv = qL2/l, where q and l are the density and the viscosity of the fluid, and L is the characteristic length scale. A weakly conducting liquid can therefore be expected to behave as a perfectly dielectric material if tE is much larger than tv (tE �tv). If the electrical conductivities of the fluids are high, the charge accumulates at the interface almost instantaneously as compared to the time scale of fluid motion (tE �tv). Such a two-fluid system can be categorized as conducting–conducting and the surface-charge conservation Eq. (4) can be assumed to reach its steady state in a time scale much smaller than the fluid response. Through a non-dimensionalization of Eq. (4), it can be shown that for cases with tE �tv in both the fluids Eq. (4) can be approximated by n Æ iJi = 0 (neglecting the surface charge conduction) [26]. If one of the fluids is a dielectric material and the other fluid is conducting, then the system can be categorized as dielectric–conducting. This case has not been considered here.


![Equation](images/2007_tomar_two_phase_ehd_vof_eq001.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq002.png)

n^

t^

1

2

Medium

Medium


> **Fig. 1. Sketch showing a jump in the properties of fluids across the interface.**

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1269

In the present paper, a methodology for modeling two-fluid systems consisting of dielectric–dielectric or conducting–conducting fluids is proposed. Governing equations for both the cases are discussed in the following.

2.1. Perfectly dielectric liquids (tE �tv)

An externally applied electric field polarizes the molecules of a dielectric material. The molecular dipoles so formed modify the electric field which again changes the polarization field. The net result of this infinite regress can be obtained directly by solving for the electric displacement vector (D) from the free-charge configuration using Eq. (1). The governing equation for the electric field in a perfectly dielectric medium having inhomogeneous isotropic linear polarizability, in absence of any free charge, can thus be written as [1],

$ �ð��0EÞ ¼ 0: ð5Þ

In absence of any time varying magnetic field, the curl of the electric field is zero ð$ �E ¼ 0Þ and the electric field can be written as the gradient of a potential function, E ¼ �$w. Eq. (5) can now be written in terms of electric potential w as,

$ �ð��0$wÞ ¼ 0: ð6Þ

The stress induced in a dielectric medium in the presence of an electric field is given by the Maxwell stress tensor sE as,

sE ¼ ��0 EE �1 2 E2I � � ; ð7Þ

where I is the identity tensor. The equivalent volume force representation of the Maxwell stress tensor for dielectric fluids (superscript d) can be obtained as [1,3],

fE;d v ¼ $ �sE ¼ �1 2 �0E2$�: ð8Þ

Note that, Eq. (8) is applicable in cases where the permittivity is a continuous function in space. In this study, we are interested in situations, where two dielectric fluids are separated by a sharp interface (see Fig. 1). Both fluids have constant dielectric permittivities, �1 and �2, respectively. Under these conditions, the equation for electric potential (6) reduces to a Laplace equation ($2w = 0) in each medium. The boundary conditions at the interface are the continuity of the electric potential (iwi = 0) and the continuity of the normal component of the electric displacement vector across the interface, i.e. Eq. (3) with qs = 0,

k��0$w �nk ¼ 0: ð9Þ

The above condition indicates a jump of E Æ n across the interface. The electric field force for perfectly dielectric fluids having homogeneous permittivities acts only at the interface. A force balance in the normal direction at the interface in the presence of an applied electric field is given by,

p1 �p2 ¼ cj þ kn �ðsv �nÞk þ fE;d s �n; ð10Þ

where pi is the pressure in medium i = 1, 2 and sv is the viscous stress tensor. The surface tension force, cj, is given by the product of the surface tension coefficient c and the local curvature j. The surface force due to the applied electric field acts normal to the interface and is represented by fE;d s �n ¼ kn �ðsE �nÞk. Thus, a pressure jump exists across the interface due to surface tension, viscous and electric forces acting normal to the interface. The jump in the tangential electric field stress is zero and thus, the tangential stress balance in the fluid yields it Æ (sv Æ n)i = 0.

2.2. Conducting liquids (tE �tv)

For a conducting fluid the charge conservation equation in the bulk of an inhomogeneous continuous media Eq. (2) can be assumed to attain steady state in a time much smaller than the time scale of the fluid motion (tE �tv) and can thus be simplified to

1270 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

$ �J ¼ $ �ðrEÞ ¼ 0: ð11Þ

The stress induced in the medium due to applied electric field is given by Maxwell stress tensor (7) and the equivalent volume force representation is given by,

fE;c v ¼ �1 2 �0E2$�þ qvE; ð12Þ

where qv = $ Æ (�0�E) is the volume charge density and the superscript c denotes the case of conducting fluids. For a two-fluid system, having constant electrical conductivities r1 and r2, Eq. (11) reduces to Laplace equations ($2w = 0) in each medium. The boundary conditions at the interface are continuity of the potential (iwi = 0) and continuity of the electric current across the interface (n Æ iJi),

kr$w �nk ¼ 0: ð13Þ

The normal stress balance for conducting fluids yields

p1 �p2 ¼ cj þ kn �ðsv �nÞk þ fE;c s �n; ð14Þ

where fE;c s is the electric force at the conducting–conducting interface. The shear stress balance equation due to the tangential component of the electric field force is given by,

kt �ðsv �nÞk þ fE;c s �t ¼ 0: ð15Þ

3. Numerical modeling

3.1. Continuous electric surface force

In a particular class of numerical algorithms for solving two-phase flow dynamics (front-capturing methods) [37–40], the sharp interface is modeled as a diffused interface spanning in a thin region (transition region) of thickness 2d around the sharp interface. The discontinuous transport and thermodynamic properties (for e.g. q, l, �and r) across the interface are smoothed along the normal direction in the transition region. In this framework, Brackbill et al. [36] introduced the concept of a continuous surface force to incorporate the pressure jump at the interface because of a surface tension force in their numerical model. They derived an equivalent volume force representation for the surface force (surface tension) in a thin transition region around the interface. The equivalent volume force in general should have the following properties as mentioned in Brackbill et al. [36]:

(a) The integration of the volume force in the normal direction across the transition region is equal to the surface force. (b) The volume force should become a surface force in the limit of vanishing transition region thickness (2d ! 0).

It can be shown that the electric field volume force as in Eqs. (8) and (12) satisfies property (a) given above. However, in the present form they are not defined at the interface because the normal component of the electric field vector has a jump across the interface (Eqs. (9) and (13)) and thus the force is undefined at the interface. Therefore, the electric field force representations Eqs. (8) and (12) do not satisfy property (b) mentioned above. In the case of two dielectric fluids we observe that the electric field force (Eq. (8)) can be re-written as (see Appendix A),

fE;d v ¼ �0 2

ðD �nÞ2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq003.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq004.png)

!


![Equation](images/2007_tomar_two_phase_ehd_vof_eq005.png)

It should be noted here that the quantities D Æ n and E Æ t are continuous for any set of orthogonal vectors n and t in an inhomogeneous medium where permittivity �is a function of space.

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1271

In the case of a two-fluid system incurring a jump in permittivity across the interface, the above representation of the volume force can be readily written as a surface force in the limit of vanishing transition region (property (b)) with n as the surface normal,

fE;d s ¼ �0 2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq006.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq007.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq008.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq009.png)

!


![Equation](images/2007_tomar_two_phase_ehd_vof_eq010.png)

where ds is the surface Dirac-delta function. The above result is obtained from Eq. (16) using the fact that $�! i�inds and similarly $(1/�) ! i1/�inds for 2d ! 0 (see Brackbill et al. [36]). Eq. (17) represents the pressure jump across the interface due to the electric Maxwell stress as used in Eq. (10). The electric field force given by Eq. (16) satisfies both the properties ((a) and (b)) given above to accurately model the surface force as a volumetric force. In the case of conducting fluids, using similar arguments the volume force equation (Eq. (12)) can be modified as (see Appendix B),

fE;c v ¼ �0 2 $ � r2

� � ðrðE �nÞÞ2 �ðE �tÞ2$� h i þ �0 ðrEÞ �$ � r


![Equation](images/2007_tomar_two_phase_ehd_vof_eq011.png)

where the normal vector is defined as n = $(�/r)/j$(�/r)j. For a two-fluid system, the above expression (Eq. (18)) readily yields the known result for surface force [4],

fE;c s ¼ �0 2 ðJ �nÞ2 �1 r2 1 ��2 r2 2

� � �ðE �tÞ2ð�1 ��2Þ � � nds þ �0ðJ �nÞ �1 r1 ��2 r2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq012.png)

In the next section we discuss the effect of smoothening of properties in the transition region on the solution of the electric field.

3.2. Smoothening of electric properties in the transition region

To solve the governing equation of the electric field (Eq. (6) or (11)) in the paradigm of front-capturing algorithms, the dielectric permittivity and electrical conductivity need to be interpolated in the diffused interface based on some indicator function (say I). The indicator function, I, indicates medium-1 (I ¼ 1) or medium-2 (I ¼ 0) and varies smoothly in the transition region. We now investigate, the accuracy of two possible interpolation schemes [41] for perfectly dielectric fluids (�¼ f ðIÞ), namely, a weighted arithmetic mean interpolation (WAM),

�¼ �1I þ �2ð1 �IÞ ð20Þ

and a weighted harmonic mean interpolation (WHM),

1 �¼ I �1 þ 1 �I �2 : ð21Þ

For the one-dimensional problem of a flat interface having a transition region thickness 2d = W1 �W2 as sketched in Fig. 2, it can be shown that the WHM interpolation (Eq. (21)) using a generic indicator function IðyÞ yields,

E1 ¼ �w0 L þ ð�1=�2 �1ÞW avg þ ð1 ��1=�2Þ R d �d I dy �d � �; ð22Þ

where E1 is the y-component of the electric field in medium-1, w0 is the electric potential of the electrode at y = L, Wavg is the location of the sharp interface around which the extent of the diffused interface is defined by W1 and W2 such that Wavg = (W1 + W2)/2. The electrode at y = 0 is earthed i.e. w = 0 at y = 0. In general, the indicator function satisfies the condition R d �d I dy ¼ d, and therefore Eq. (22) reduces to

E1 ¼ �w0 L þ ð�1=�2 �1ÞW avg ; ð23Þ

1272 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

which is also the exact solution for a sharp interface at y = (W1 + W2)/2. Using a WAM interpolation we get

E1 ¼ �w0 L þ ð�1=�2 �1ÞW avg þ R d �d dy Iþð1�IÞð�1=�2Þ �dð1 þ �1=�2Þ : ð24Þ

For a linear indicator function I ¼ ðy �W 2Þ=ðW 1 �W 2Þ, Eq. (24) is

E1 ¼ �w0 ðL þ W avgð�1=�2 �1Þ �dð1 þ �1=�2 �2 lnð�1=�2Þ=ð�1=�2 �1ÞÞÞ ; ð25Þ

whereas for a WHM interpolation (Eq. (21)), the electric field in the bulk is independent of the transition region thickness (Eq. (23)). For L = 1 cm, Wavg = 5 mm, �1/�2 = 70, w0 = 100 V and d = 0.01 L, we obtain E1 = 2.8743 · 102 V/m using Eq. (25) whereas the exact solution (Eq. (23)) is E1 = 2.8169 · 102 V/m. The error in using the WAM interpolation scheme (Eq. (20)) is analytically 2% for these parameters. This clearly shows that the WHM (Eq. (21)) is a better choice for the interpolation of the permittivity in the transition region. In Section 4, numerical results for the one-dimensional problem of a flat interface using both interpolation schemes are presented. In a similar way, it can be shown that for conducting fluids the WAM interpolation scheme for the electrical conductivity gives a transition thickness dependent solution for electric field intensity in the bulk.

3.3. Numerical implementation

For numerical validation of the model proposed above, we use a coupled level set and volume-of-fluid (CLSVOF) algorithm [38] on a staggered grid of equidistant cell width h to solve the electric field governing equation Eq. (5), (Eq. (11)) for dielectric (conducting) fluids along with the hydrodynamic governing equations,

$ �v ¼ 0; ð26Þ

qðH dÞ ov ot þ v �$v � � ¼ �$p þ $ �lðH dÞð$v þ $vTÞ � � þ qðH dÞg þ fc v þ fE v ; ð27Þ

where v = (u,v) is the velocity vector, t is time, p is the pressure, g is the gravitational acceleration. The density, q(Hd), and the dynamic viscosity, l(Hd), of the medium are interpolated in the transition region using a smoothed Heaviside function Hd as the indicator function I,

qðH dÞ ¼ q1H d þ ð1 �H dÞq2; ð28Þ

lðH dÞ ¼ l1H d þ ð1 �H dÞl2; ð29Þ

as used elsewhere [42]. The smoothed Heaviside function is defined as,

0

L

W W W

x

y


![Equation](images/2007_tomar_two_phase_ehd_vof_eq013.png)

1

2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq014.png)


> **Fig. 2. Configuration of the flat interface test case (Section 4.1).**

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1273

H dð/Þ ¼


![Equation](images/2007_tomar_two_phase_ehd_vof_eq015.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq016.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq017.png)

8 > <

> : ð30Þ

where 2d is the transition region thickness and / is a distance function (the level set function) which assumes a value of zero at the interface [38,39]. d is chosen to be 1.5h in the present work for all cases unless mentioned otherwise. The surface tension force fc v is defined as a continuous surface tension force cj(/)$Hd(/) (volume force), non-zero only in the transition region [36]. The electric field surface force for perfectly dielectric fluids is modeled as a volumetric force using Eq. (17),

fE;d v ¼ ��0 2 ðð�E �nÞ2=ð�1�2Þ þ ðE �tÞ2Þð�1 ��2Þ$H d: ð31Þ

Here, �E Æ n and E Æ t are evaluated at the computational cell faces from the gradient of the potential using a central difference scheme. �is computed using a WHM interpolation scheme (Eq. (21)). The normal unit vector in the transition is obtained as n = $Hd/j$Hdj. For conducting fluids, the electric field force using Eq. (19) can be written as

fE;c v ¼ �0 2 ðJ �nÞ2 �1 r2 1 ��2 r2 2

� � �ðE �tÞ2ð�1 ��2Þ � � $H d þ �0ðJ �$H dÞ �1 r1 ��2 r2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq018.png)

Note that in the thin transition region along the normal the quantities E Æ t, �E Æ n and rE Æ n are assumed to be constant, so that the volume force is evaluated by smoothening the surface Dirac-delta function (ds) as in Eqs. (17) and (19) for perfectly dielectric and conducting cases, respectively. This is similar to the assumption of the curvature in the continuous surface tension force model of Brackbill et al. [36]. The essential logic behind this scheme is the following. In our numerical algorithm, we first predict the provisional velocities incorporating the body forces. In the subsequent corrector step, we solve the pressure correction equation, which leads to pressure update (ensuring Navier–Stokes momentum balance) and correction in provisional velocity to ensure incompressibily of the fluid. In this step we implicitly perform integration over the body force along with inertial and viscous forces to obtain the corresponding static pressure. In the case of smoothened surface forces, this integration is performed over a few cells and thus the integration of the volume force in the transition will be inaccurate if the variation in volume force is not well behaved. For example, using a WHM scheme (to ensure accurate evaluation of electric field), if we evaluate the term (E Æ t)2$�, we get a function, (E Æ t)2(�1�2)/(�2 + (�1 ��2)Hd)2(�1 ��2)$Hd. The use of this function in evaluation of the electric field force was found to produce inaccurate pressure jump across the interface. To avoid the numerical error in integration, we chose an expression (E Æ t)2(�1 ��2)$Hd, which conserves the surface force under the approximation that E Æ t is nearly constant. A similar argument also serves the justification of the expression of volume force for conducting fluids (Eq. (32)). The CLSVOF algorithm used here has been extensively validated in Ref. [42] and applied in Refs. [43,44] for the cases not involving electric field. We have used a combined VOF and LS method in order to exploit virtues of both approaches, namely, the accurate computation of the interface normal and the curvature using the LS method and a mass conserving advection of the interface using the VOF method. However, the methodology for incorporating the electric forces acting at the interface (Eqs. (16) and (18)) is in general valid for any interface-capturing algorithm using a diffused interface model.

4. Results and discussion

4.1. Horizontal interface in an electric field (dielectric/dielectric)

The analytic result for the electric field in a confined flat film (see Fig. 2) can be easily obtained (see Section 3.2). The pressure jump can be evaluated by computing the jump in the Maxwell stress tensor across the interface (Eq. (7)). In the previous section we showed that an inappropriate interpolation of the dielectric permittivity in a thin transition region can lead to an erroneous solution of the electric field. Here, we show that the

1274 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

numerical solutions for the flat interface configuration, obtained using the CLSVOF algorithm are in agreement with the analytical predictions made in Section 3.2. The numerical results have been obtained using the interpolation schemes WHM as well as WAM for different transition region thicknesses 2d with the Heaviside function as the indicator function. Fig. 3 shows the variation of the electric field E with y. The electric field has been scaled with the analytical solution for the sharp interface in medium-1 (Eq. (23)), Eex: 1 ¼ 0:00084 V/m for �1 = 70 and �2 = 1. The interface is diffused around the sharp interface (y = Wavg = 40h for an 80 · 80 grid) using a smoothed Heaviside function. The electric field obtained using a WHM interpolation is exact, whereas, for a WAM interpolation, we obtain a transition region thickness dependent solution. For a WHM interpolation scheme, the electric field strength at y = Wavg = 40h (sharp interface) is independent of the transition region thickness in contrast to the WAM interpolation and it is equal to the average of the electric field values in the bulk of the two mediums.


> **Table 1 shows the convergence of electric field E=Eex: 1 and the pressure jump Dp obtained using both WHM and WAM interpolation schemes. Using WAM, a large error is observed in electric field which decreases with grid refinement. The error in computing the corresponding pressure jump is also large for the WAM interpolation schemes. In contrast, the results using the WHM interpolation are remarkably accurate even for very coarse grids.**

4.2. Pressure jump in spherical and spheroidal drops (dielectric/dielectric)

We show here a test of the accuracy of the present algorithm by comparing the numerical simulation result of the pressure in a spherical drop with the anayltical prediction. The governing equations are solved using an

y/h

E/E1 ex.

30 35 40 45 50

0

10

20

30

40

50

60

70

80

90


![Equation](images/2007_tomar_two_phase_ehd_vof_eq019.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq020.png)


> **Fig. 3. Distribution of the electric field (normalized by the exact value in medium-1) E=Eex: 1 in normal direction to the interface for different interpolation schemes and transition region thicknesses.**


> **Table 1 Comparison of the numerically computed values of the pressure jump across the interface Dp and the electric field E=Eex: 1 using the WAM and WHM interpolation schemes**

Grid Error (%) Dp Error (%) E=Eex: 1 40 WAM 14.74 7.00 WHM 9.70 · 10�3 4.80 · 10�3

80 WAM 6.98 3.43 WHM 6.70 · 10�3 2.10 · 10�3

160 WAM 3.4 1.69 WHM 5.48 · 10�3 8.54 · 10�4

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1275

axi-symmetric CLSVOF formulation. The pressure jump across the interface of a spherical drop in an electric field is in general non-uniform and is a function of the polar angle h (see Fig. 4). The variation in pressure jump (Dp) incorporating the surface tension and the electric force is given by [26],

p1 �p2 ¼ 2c R þ �0


![Equation](images/2007_tomar_two_phase_ehd_vof_eq021.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq022.png)

where the subscripts ‘1’ and ‘2’ denote the quantities inside and outside the drop, respectively. Static pressure is denoted by p and R is the radius of the drop. En and Et are the components of the electric field in the normal and tangential direction respectively at the surface of the drop (see Fig. 4) [3]:

E1n ¼ 3E1 �r þ 2 ny; ð34Þ

E1t ¼ E2t ¼ �3E1 �r þ 2 nx; ð35Þ

E2n ¼ �1 �2

�� 3E1 �r þ 2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq023.png)

Here, E1 is the externally applied electric field (in the y-direction) far away from the drop, �r = �1/�2 is the ratio of the electrical permittivities of the drop and the surrounding liquid, nx and ny are the x and y components of the unit normal at the interface (Fig. 4).


> **Fig. 5 shows the comparison between the pressure jump at r = R obtained using the present algorithm and the analytical result. The pressure difference is obtained across the thin transition region after one time step. The numerical algorithm converges to the analytical prediction on grid refinement. Table 2 shows the convergence of the maximum pressure jump which occurs at the equator line of the spherical drop (Fig. 5). A spherical drop deforms under an externally applied uniform electric field. The equilibrium shape can be obtained using an energy minimization approach considering the surface energy and the electrical stored energy of the drop [18]. The equilibrium shape of the drop is close to an ellipsoid elongated in the direction of the electric field and the pressure inside the drop is nearly uniform. The pressure jump across the drop interface can be obtained using Eqs. (7) and (10) as**

Dp ¼ cj þ �0 2 E2 1ð�1 ��2Þ; ð37Þ


![Equation](images/2007_tomar_two_phase_ehd_vof_eq024.png)

r^

x

y

t

n

1 2


> **Fig. 4. Schematic of a drop placed in an electric field showing the cordinate axis, normal and tangential vectors, radial vector ^r and polar angle h.**

1276 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

where, E1 = �2E1/((1 �n)�2 + n�1) with n = (1 �e2)(ln[(1 + e)/(1 �e)] �2e)/(2e3) and e ¼ ffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi 1 �a2=b2 q given by Landau and Lifshitz for an ellipsoid [3]. Here, 2b is the major axis (along the applied electric field) and 2a is the minor axis. Subscripts 1 and 2 refer to drop and surrounding fluid, respectively. E1 is the electric field strength far away from the drop. The above expression is obtained by balancing stresses only at the equator as has been done extensively in theoretical analysis [18]. Thus, this does not ensure pressure balance all along the interface. A uniform electric field is imposed at the top and bottom boundary of the computational domain. Two cases (A) and (B) have been considered for which �1/�2 = 10, �2 = 1, c = 32 · 10�4 N/m and the gravitational acceleration is neglected. The radius of the equivalent spherical drop in an electric field free environment is R = 1 mm. The shape of the deformed drop given by a and b, the electric field Bond number ðBoe ¼ �0�2E2 1R=cÞ and the predicted pressure jump for the two cases are:

(A) a = R/1.1, b = 1.21R and Boe = 0.33. The theoretical pressure jump using Eq. (37) is Dp = 5.08 Pa. (B) a = R/1.2, b = 1.44R, Boe = 0.49 and Dp = 4.24 Pa for this case.

The cases (A) and (B) have been chosen based on the stability analysis of Chen and Chaddock [18]. According to the theory presented in Chen and Chaddock [18], the drops in cases (A) and (B) should be in equilibrium and stable for the corresponding Bond numbers. In Fig. 6, the electric field potential (left half) and electric field lines (right half) are plotted for case (B) of a static ellipsoid. The electric field lines in the surrounding fluid (medium-2) are distorted in the presence of the drop (medium-1) having a different permittivity. The electric field lines as they enter the drop are aligned along the applied electric field which is in conformity with the theoretical predictions.


> **Table 2 The maximum pressure jump across the interface for a spherical drop for different grid spacing**

R/h Dpmax (Pa) Error (%)

20 5.83 4.49 40 5.97 2.24 80 6.03 1.15 160 6.09 0.28

The error is the relative error of the computed pressure jump with the analytical results Dpmax = 6.103 Pa (Eq. (33)) at h = 0.


![Equation](images/2007_tomar_two_phase_ehd_vof_eq025.png)


### p

-1 -0.5 0 0.5 1 3

4

5

6

Theory R/h = 20 R/h = 40 R/h = 80 R/h = 160

(rad.)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq026.png)


> **Fig. 5. Comparison of variation in pressure jump at the interface of a spherical drop obtained in present simulations with theory (Eq. (33)). The pressure jump in numerical simulations is obtained across the transition region from the pressure distribution after one time step.**

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1277


> **Table 3 shows the convergence of the pressure difference Dp and the standard deviation �r of the pressure inside the drop from the theoretical value. Both quantities are calculated based on all cells which are filled with liquid by 99% or more.The numerically obtained mean pressure difference Dp is close to the analytical predictions and the standard deviation of the pressure distribution inside the drop also decreases on grid refinement. We note here that since the equilibrium shape in only approximately spheroid, this test cannot be considered as a convergence test of the numerical algorithm. But, nevertheless the test shows close agreement between the computed pressure jump with the theoretical predictions which are considered to show good agreement with the experimental investigations [18].**

4.3. Conducting drops

The problem of the deformation of a spherical drop under an external uniform electric field (E1) has been used here to validate the numerical algorithm for a system of two conducting fluids. The electric field intensity inside the drop is uniform. The drop deformation D because of an applied electric field, obtained by perturbation analysis and hence valid for only small deformations, is given by [45,46]

D ¼ L �B L þ B ¼ 9Boe 16


![Equation](images/2007_tomar_two_phase_ehd_vof_eq027.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq028.png)

where

fd

r1 r2 ; �1 �2 ; l1 l2

� � ¼ r1 r2


![Equation](images/2007_tomar_two_phase_ehd_vof_eq029.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq030.png)


![Equation](images/2007_tomar_two_phase_ehd_vof_eq031.png)

0.0075

-0.0075

0

r

z


> **Fig. 6. Electric field potential (left half) and electric field lines (right half) in a perfectly dielectric ellipsoidal drop (case B) suspended in a perfectly dielectric liquid.**


> **Table 3 Deviation of the computed pressure jump across the interface from the theoretical value (Eq. (37)) and uniformity of the pressure inside the drop (standard deviation from the mean, �r) for the two cases A and B**

R/h Error DpA (%) �rA Error DpB (%) �rB

4 6.1 0.043 14.5 0.22 8 4.1 0.019 7.3 0.05 16 2.2 0.025 3.9 0.01 32 1.2 0.007 1.8 0.001

1278 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

Here, L and B are the major and minor axis from end-to-end of the deformed drop, respectively, with the major axis aligned with the applied electric field E1 and Boe ¼ �0�2E2 1R=c, R being the radius of the undeformed drop. Subscripts 1 and 2 represent the drop and the bulk fluid, respectively. The drop deformation can be categorized as prolate (fd > 0) or oblate (fd < 0) and is governed by the influence of the normal electric force and in contrast to the dielectric fluids also by the tangential electric force acting at the interface (Eq. (15)) [45]. As described in Ref. [26], for fd = 0 the drop remains spherical/undeformed despite of a non-zero velocity field at the interface because of the tangential stress. For these cases, the sense of circulation can be determined from the sign of the tangential component of the surface force �0(J Æ n)�1/r1(1 �(�2/�1)/(r2/r1))(E Æ t)t (see Eq. (19)).


> **Fig. 7a shows velocity vectors and streamlines for a case with r1/r2 = 5.1, �1/�2 = 10, l1/l2 = 1 and Boe = 0.18, which corresponds to a case of no deformation (fd = 0). The sense of circulation is clockwise in the first quadrant of the drop, which is in agreement with theory and experiments [8,26,46]. The deformation of the drop (L/2R) from the sphere measured from the computed shape is negligible. Using a grid of R/h = 10 (grid: 40 · 80) the deviation from the spherical shape L/2R = 1 was 1.6% and for R/h = 20 (grid: 80 · 160) it reduces to 0.6%. Further, simulations have been performed using electric properties for which small deformations are expected so that a comparison with Eq. (38) is possible. Simulations have been performed by varying the electric conductivity ratio r1/r2 and keeping the other parameters constant. The initial shape of the drop has been**


> **Fig. 7. Velocity vectors (left) and stream lines (right) of a conducting drop suspended in a conducting liquid for: (a) D = 0 (no deformation case) obtained using r1/r2 = 5.1 and �1/�2 = 10 and (b) oblate deformation obtained using r1/r2 = 1.81 < �1/�2 = 10.**

-0.2

-0.15

-0.1

-0.05

0

0.05

0.1

0 2 4 6 8 10 12 14

deformation D


![Equation](images/2007_tomar_two_phase_ehd_vof_eq032.png)

theory computations


> **Fig. 8. Comparison of the deformation D between the predicted values (Eq. (38)) and the computational results for different ratios of the electric conductivities r1/r2 and �1/�2 = 10.**

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1279

chosen to be spherical for all the cases. Fig. 7b shows a case of an oblate equilibrium shape of the drop obtained using r1/r2 = 1.81.


> **Fig. 8 shows a comparison of the computational results (for R/h = 20) with the predictions from Eq. (38). For small values of D, the agreement between simulations and theory is excellent. For larger values of D, the deviation increases since the predictions from Eq. (38) are accurate for small deformations only.**


> **Fig. 9 shows a comparison between the radial and polar components of the velocity at h = p/4 obtained from the numerical simulations (grid 80 · 240 with R/h = 20) and predictions of the theory [45] for the no deformation case described above (see Fig. 5a). Using the velocity stream function derived in Ref. [45], velocity components in ^r and ^h direction (Fig. 4) can be obtained as,**

r/R


![Equation](images/2007_tomar_two_phase_ehd_vof_eq033.png)

-0.01

-0.005

0

0.005


![Equation](images/2007_tomar_two_phase_ehd_vof_eq034.png)

1 0 5 2 3 4


> **Fig. 9. The comparison of the velocity profile obtained in present numerical simulation with theory (Eqs. (40) and (41)) at a polar angle h = p/4.**

r/R

z/R

0 -2 -4

10

4 2 2

6

8

4


> **Fig. 10. Drop interaction in a leaky dielectric emulsion with r1/r2 = 2, �1/�2 = 8, Boe = 1 and R ¼ 0:01. The unfilled circles show the boundary element method results of Baygents et al. (Fig. 5a in Ref. [26]).**

1280 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

ur ¼ A r R �� 1 �ðr R Þ2 � � ð3 sin2ðhÞ �1Þ r 6 R

A R r ��2 ðR r Þ2 �1 � � ð3 sin2ðhÞ �1Þ r P R

8 > <

> : ; ð40Þ

uh ¼

3A 2 ðr RÞð1 �5 3 ðr R Þ2Þ sinð2hÞ r 6 R;

�A R r ��4 sinð2hÞ r P R;

(


![Equation](images/2007_tomar_two_phase_ehd_vof_eq035.png)

where

A ¼ �0:9E2 1R�2�0ðr1=r2 ��1=�2Þ

ðr1=r2 þ 2Þ2ð1 þ kÞl2 : ð42Þ

r/R

z/R

-4 -2 0 2 4 4

6

8

10

12

14

16

r/R

z/R

-4 -2 0 2 4 4

6

8

10

12

14

16

r/R

z/R

-4 -2 0 2 4 4

6

8

10

12

14

16

r/R

z/R

-4 -2 0 2 4 4

6

8

10

12

14

16


> **Fig. 11. Drop-pair interaction in a perfectly dielectric emulsion with �1/�2 = 8 and Boe = 1.5: (a) streamlines and velocity vectors during the prolate deformation of drops, (b) drops just before coalescence, (c) coalescence and (d) stretching of coalesced drops into a cylinder.**

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1281

Here, k = l1/l2 is the ratio of the dynamic viscosity of the fluid inside and outside the drop, E1 is the external electric field far away from the drop. A high viscosity of the fluids is chosen (l1 = l2 = 2 · 10�2 Pa s) so as to be in the regime of Stokes approximation under which the above analytical results were obtained [45]. The numerical simulation results of the velocity field inside the drop match well with the analytical predictions as is evident from Fig. 9. However, the velocity outside the drop deviates slightly from the theory possibly due to the smaller computational domain chosen for simulations. The velocity field which decays slowly in analytical simulation and is zero at infinity decays faster in simulations due to imposed boundary conditions. Similar conclusions were reached by Zhang and Kwok [34] in the comparison of the Lattice Boltzmann simulations of electric field driven deformations in a 2D leaky dielectric drop with analytical results.

4.4. Interacting droplets

To show the strength of the numerical algorithm proposed, we present in this section simulation results for interacting droplets in an emulsion. Emulsion drops under applied electric field deform, coalesce and possibly break-up into smaller droplets. Between two polarized droplets in an electric field an attractive force always exists because of a dipole–dipole interaction. This phenomenon is known as dielectrophoretic effect and it can eventually lead to the coalescence of the emulsion drops. In addition, the electrohydrodynamic flow generated because of the electric stresses at the drop interfaces influences the motion of the droplets. In a leaky dielectric emulsion (conducting–conducting), drops can deform prolately or oblately and subsequently might come closer or move apart depending upon the ratio of conductivities and permittivities of the drop and the suspending medium [26]. The electrohydrodynamic problem of drop interaction may then be characterized by the dimensionless parameters, namely, ratio of dielectric permittivities �1/�2, conductivities r1/r2, viscosities l1/l2, electric bond number Boe ¼ �0�2E2 1R=c and Reynolds number R ¼ q2�2�0E2 1R2=l2 2, where R is the radius of the undeformed spherical drop. Interaction between a pair of drops has been studied in detail using a boundary element method in Baygents et al. [26] under the Stokes approximation (R �1). Fig. 10 shows a situation (r1/r2 = 2, �1/�2 = 8, Boe = 1, R ¼ 0:01) in which a pair of initially spherical droplets deform oblately and attract each other. The deformed droplet shape and position, at a non-dimensional time s ¼ t=ðl2=ð�2�0E2 1ÞÞ ¼ 30, agree well with the result of Baygents et al. (Fig. 5a in [26]) shown by unfilled circles. The computational domain chosen for the simulation is 8R · 12R and the grid size, h, is such that R/h = 10.


> **Fig. 11 shows the interaction between a pair of drops in a perfectly dielectric emulsion (�1/�2 = 8, Boe = 1.5). The drops deform and move towards each other (Fig. 11a and b), coalesce (Fig. 11c) and further elongate prolately forming a columnar structure (Fig. 11d). As in typical VOF or LS methods, we assume coalescence when the distance between the approaching interfaces reduces to less than one grid spacing. The numerical simulations for both the situations (Figs. 10 and 11) are in agreement with the theory of dielectrophoretic effect mentioned earlier in this section and in detail in Ref. [26].**

5. Conclusions

We have presented here a volume-of-fluid (VOF) based methodology to solve the governing equations for the electric field and the associated surface force in two-fluid systems, namely, dielectric–dielectric fluids and conducting–conducting fluids. We have shown that the interpolation scheme to smoothen the electric properties (conductivity and permittivity) in the interface transition region has a significant influence on the solution in the bulk. For the electric field, it is verified here that the electric permittivities and electric conductivities in the transition region between the two fluids should be interpolated using a weighted harmonic mean (WHM) interpolation scheme rather than the usual weighted arithmetic mean (WAM) scheme, although the latter is the standard method for interpolating density and viscosity in VOF methods. Brackbill et al. [36] laid down the properties of a surface force to volume force transformation used to include capillary effects in the VOF algorithm. We argue here that the usual body force representation of the electric field force does not satisfy one of the two properties. We derive a new representation of the electric field force and demonstrate that it satisfies both properties stated by Brackbill et al. [36]. The body force representations proposed are in general valid for VOF or level set (LS) based methods. Essentially, we show that

1282 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

the continuum surface tension model of Brackbill et al. [36] can be extended to include electric surface forces with a transformation of the body force representation for inhomogeneous fluids. Using a combined LS and VOF algorithm (CLSVOF), the model proposed has been validated with the help of a set of test problems. Using a test case of a flat interface placed between two electrodes we show analytically as well numerically that the WAM interpolation scheme leads to a transition region thickness dependent bulk electric field. In contrast, the solutions are independent of the transition region, when the WHM interpolation scheme is used. Another test case examines the pressure jump, as a function of the polar angle h, occurring at the interface of a spherical dielectric drop in an uniform electric field. The forces acting at the interface are electric and capillary forces. The numerically obtained pressure jump shows a good agreement with the theoretical predictions. The proposed model for conducting–conducting fluids is tested by a problem, where a conducting spherical drop suspended in a conducting medium undergoes deformation because of the electric field. We show that the numerical results of the drop deformation agree well with theory and numerical predictions of Baygents et al. [26]. Finally, interaction between two drops in an electric field have been simulated. The numerical algorithm could capture the motion of drops under the dielectrophoretic effect, showing the capability of the proposed method.

Acknowledgements

The support received from the Department of Science and Technology (DST, India) and the German Academic Exchange Program (DAAD, Germany) for the ‘‘Project Based Personnel Exchange Programme’’ is gratefully acknowledged.

Appendix A

The derivation of the modified form of the electric field volume force in an inhomogeneous (permittivity is a function of space) perfectly dielectric fluid (Eq. (16) in Section 3.1) is derived in the following. Electric field force in an inhomogeneous medium can be written as

fE;d v ¼ �1 2 �0E2$�: ðA:1Þ

Writing E2 = (E Æ n)2 + (E Æ t)2, Eq. (A.1) becomes,

fE;d v ¼ �0 2 ð�ðE �nÞ2$��ðE �tÞ2$�Þ; ðA:2Þ

which on multiplying and dividing the first term by �2 yields,

fE;d v ¼ �0 2 ð�ð�E �nÞ2$�=�2 �ðE �tÞ2$�Þ; ðA:3Þ

and thus can be readily written as,

fE;d v ¼ �0 2 ð�E �nÞ2$ 1 �

�� �ðE �tÞ2$� � � : ðA:4Þ

In the above equation substituting D = �E yields Eq. (16) presented in Section 3.1.

Appendix B

The derivation of the modified form of the electric field volume force in an inhomogeneous (conductivity and permittivity are a function of space) conducting fluid (Eq. (18) in Section 3.1) is derived in the following. Electric field force in an inhomogeneous leaky dielectric medium is written as,

fE;c v ¼ �1 2 �0E2$�þ qvE: ðB:1Þ

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1283

The volume-charge density qv = $ Æ (�0�E) can be written as,

qv ¼ �0ðrEÞ �$ð�=rÞ þ �0ð�=rÞ$ �ðrEÞ ¼ �0ðrEÞ �$ð�=rÞ; ðB:2Þ

using the electric field governing equation $ Æ (rE) = 0. Using the above expression for charge density we can write the electric field force Eq. (B.1) as,

fE;c v ¼ �0 2 �ðJ �nÞ2 $� r2 þ 2 J �$ � r

�� � �ðJ �nÞ r n �ðE �tÞ2$� � � þ �0 J �$ � r


![Equation](images/2007_tomar_two_phase_ehd_vof_eq036.png)

Choosing the orthogonal set of unit vectors n and t such that n = $(�/r)/j$(�/r)j, the above equation can be written as,

fE;c v ¼ �0 2 �ðJ �nÞ2 $� r2 þ 2ðJ �nÞ2 1 r $ � r

�� �ðE �tÞ2$� � � þ �0 J �$ � r


![Equation](images/2007_tomar_two_phase_ehd_vof_eq037.png)

which can be modified to,

fE;c v ¼ �0 2 ðJ �nÞ2 $� r2 �2�$r r3

� � �ðE �tÞ2$� � � þ �0 J �$ � r


![Equation](images/2007_tomar_two_phase_ehd_vof_eq038.png)

The above equation yields Eq. (18) on substituting,

$ � r2

� � ¼ $� r2 �2�$r r3


![Equation](images/2007_tomar_two_phase_ehd_vof_eq039.png)


## References

[1] D.J. Griffiths, Introduction to Electrodynamics, third ed., Pearson Education, Delhi, 2006. [2] J.R. Melcher, G.I. Taylor, Electrohydrodynamics: a review of the role of interfacial shear stress, Annu. Rev. Fluid Mech. 1 (1969) 111–146. [3] L.D. Landau, E.M. Lifshitz, Electrodynamics of Continuous Media, second ed., Pergamon, Oxford, 1975. [4] D.A. Saville, Electrohydrodynamics: The Taylor–Melcher leaky dielectric model, Annu. Rev. Fluid Mech. 29 (1997) 27–64. [5] J.E. Hart, G.A. Glatzmaier, J. Toomre, Space-laboratory and numerical simulations of thermal convection in a rotating hemispherical shell with radial gravity, J. Fluid Mech. 173 (1986) 519–544. [6] Y. Tsori, F. Tournilhac, L. Leibler, Demixing in simple fluids induced by electric field gradients, Nature 430 (2004) 544–547. [7] S. Shin, I. Kang, Y.-K. Cho, Mixing enhancement by using electrokinetic instability under time-periodic electric field, J. Micromech. Microeng. 15 (2005) 455–462. [8] G.I. Taylor, Disintegration of water drops in an electric field, Proc. R. Soc. Lond. A 280 (1964) 383–397. [9] J. Sherwood, Breakup of fluid droplets in electric and magnetic fields, J. Fluid Mech. 188 (1988) 133–146. [10] J.-W. Ha, S.-M. Yang, Effect of nonionic surfactant on the deformation and breakup of a drop in an electric field, J. Colloid Interface Sci. 206 (1998) 195–204. [11] A. Sou, K. Sakai, T. Nakjima, Control of ink transportation in electrostatic inkjet printer, ASME Fluid Engineering Division Summer Meeting, Montreal, Quebec, Canada, 2002. [12] S. Lee, D. Byun, S.U. Son, Y. Kim, H.S. Ko, Electrostatic droplet formation and ejection of colloid, in: International Symposium on Micro-NanoMechatronics and Human Science, The Fourth Symposium Microand Nano-Mechatronics for Information-based Society, Nagoya, 2004, pp. 249–254. [13] R.L. Johnson, Effect of an electric field on boiling heat transfer, AIAA J. 6 (8) (1968) 1456–1460. [14] J. Ogata, A. Yabe, Basic study on the enhancement of nucleate boiling heat transfer by applying electric fields, Int. J. Heat Mass Transfer 36 (1993) 775–782. [15] J. Ogata, A. Yabe, Augmentation of boiling heat transfer by utilizing the EHD effect – EHD behaviour of boiling bubbles and heat transfer characteristics, Int. J. Heat Mass Transfer 36 (1993) 783–791. [16] P.D. Marco, W. Grassi, G. Memoli, T. Takamasa, A. Tomiyama, S. Hosokawa, Influence of electric field on single gas-bubble growth and detachment in microgravity, Int. J. Multiphase Flow 29 (2003) 559–578. [17] F.J. Higuera, Injection of bubbles in a quiescent inviscid liquid under a uniform electric field, J. Fluid Mech. 568 (2006) 203–222. [18] K.J. Cheng, J.B. Chaddock, Deformation and stability of drops and bubbles in an electric field, Phys. Lett. 106 A (1984) 51–53. [19] K.J. Cheng, Capillary oscillations of a drop in an electric field, Phys. Lett. 112 A (8) (1985) 392–396. [20] K.J. Cheng, J.B. Chaddock, Effect of an electric field on bubble growth, Int. Commun. Heat Mass Transfer 12 (1985) 259–268. [21] O. Pamperin, H.-J. Rath, Influence of buoyancy on bubble formation at submerged orifices, Chem. Eng. Sci. 50 (19) (1995) 3009– 3024.

1284 G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285

[22] B. Shapiro, H. Moon, R.L. Garrell, C. Kim, Equilibrium behaviour of sessile drops under surface tension, applied external fields, and material variations, J. Appl. Phys. 93 (9) (2003) 5794–5811. [23] C. Herman, E. Iacona, Modeling of bubble detachment in reduced gravity under the influence of electric fields and experimental verification, Heat Mass Transfer 40 (2004) 943–957. [24] J. Carrera, R. Parthasarathy, S.R. Gollahalli, Bubble formation from a free-standing tube in microgravity, Chem. Eng. Sci. 61 (2006) 7007–7018. [25] H.J. Cho, I.S. Kang, Y.C. Kweon, M.H. Kim, Study of the behavior of a bubble attached to a wall in a uniform electric field, Int. J. Multiphase Flow. 22 (5) (1996) 909–922. [26] J.C. Baygents, N.J. Rivette, H.A. Stone, Electrohydrodynamic deformation and interaction of drop pairs, J. Fluid Mech. 368 (1998) 359–375. [27] H.J. Cho, I.S. Kang, Y.C. Kweon, M.H. Kim, Numerical study of the behavior of a bubble attached to a tip in a nonuniform electric field, Int. J. Multiphase Flow. 24 (3) (1998) 479–498. [28] O.A. Basaran, L.E. Scriven, Axisymmetric shapes and stability of pendant and sessile drops in an electric field, J. Colloid Interface Sci. 140 (1) (1990) 10–30. [29] M.T. Harris, O.A. Basaran, Capillary electrohydrostatics of conducting drops hanging from a nozzle in an electric field, J. Colloid Interface Sci. 161 (1993) 389–413. [30] M.T. Harris, O.A. Basaran, Equilibrium shapes and stability of nonconducting pendant drops surrounded by a conducting fluid in an electric field, J. Colloid Interface Sci. 170 (1995) 308–319. [31] P.K. Notz, O.A. Basaran, Dynamics of drop formation in an electric field, J. Colloid Interface Sci. 213 (1999) 218–237. [32] A. Fernandez, G. Tryggvason, J. Che, S. Ceccio, The effects of electrostatic forces on the distribution of drops in a channel flow: twodimensional oblate drops, Phys. Fluid 17 (2005) 093302. [33] G. Tryggvason, B. Bunner, A. Esmaeeli, D. Juric, N. Al-Rawahi, W. Tauber, J. Han, S. Nas, Y.-J. Jan, A front-tracking method for the computations of multiphase flows, J. Comp. Phys. 169 (2001) 708–759. [34] J. Zhang, D.Y. Kwok, A 2D lattice Boltzmann study on electrohydrodynamic drop deformation with the leaky dielectric theory, J. Comput. Phys. 206 (2005) 150–161. [35] S.W.J. Welch, G. Biswas, Direct simulation of film boiling including electrohydrodynamics forces, Phys. Fluid 19 (2007) 012106. [36] J.U. Brackbill, D.B. Kothe, C. Zemach, A continuum method for modeling surface tension, J. Comput. Phys. 100 (1992) 335–354. [37] S. Popinet, S. Zaleski, A front-tracking algorithm for accurate representation of surface tension, Int. J. Numer. Method Fluid 30 (1999) 775–793. [38] M. Sussman, E.G. Puckett, A coupled level set and volume-of-fluid method for computing 3D and axisymmetric incompressible twophase flows, J. Comput. Phys. 162 (2000) 301–337. [39] S. Osher, R.P. Fedkiw, Level set methods: an overview and some recent results, J. Comput. Phys. 169 (2001) 463–502. [40] G. Son, Efficient implementation of a coupled level set and volume-of-fluid method for 3-dimensional incompressible two-phase flows, Numer. Heat Transfer B 43 (6) (2003) 549–565. [41] J. Li, E. Lopez-Page´s, P. Yecko, S. Zaleski, Droplet deformation in sheared liquid–gas layers, Theor. Comput. Fluid Dyn. 21 (2007) 59–76. [42] D. Gerlach, G. Tomar, G. Biswas, F. Durst, Comparison of volume-of-fluid methods for computing surface tension dominant twophase flows, Int. J. Heat Mass Transfer 49 (2006) 740–754. [43] G. Tomar, G. Biswas, A. Sharma, A. Agrawal, Numerical simulation of bubble growth in film boiling using CLSVOF method, Phys. Fluid 17 (1) (2005) 112103. [44] D. Gerlach, N. Alleborn, V. Buwa, F. Durst, Numerical simulation of periodic bubble formation at a submerged orifice with constant gas flow rate, Chem. Eng. Sci. 62 (7) (2007) 2109–2125. [45] G.I. Taylor, Studies in electrohydrodynamics I. The circulation produced in a drop by an electric field, Proc. R. Soc. Lond. A 291 (1966) 159–166. [46] S. Torza, R.G. Cox, S.G. Mason, Electrohydrodynamic deformation and burst of liquid drops, Phil. Trans. R. Soc. Lond. A 269 (1971) 295–319.

G. Tomar et al. / Journal of Computational Physics 227 (2007) 1267–1285 1285

