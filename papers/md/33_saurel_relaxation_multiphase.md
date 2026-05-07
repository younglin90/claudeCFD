Simple and efﬁcient relaxation methods for interfaces
separating compressible ﬂuids, cavitating ﬂows and
shocks in multiphase mixtures
Richard Saurel a,b,*, Fabien Petitpas a, Ray A. Berry c
a Polytech’Marseille, Aix-Marseille University and SMASH Project UMR CNRS 6595 – IUSTI-INRIA, 5 rue E. Fermi, 13453 Marseille Cedex 13, France
b University Institute of France and SMASH Project UMR CNRS 6595 – IUSTI-INRIA, 5 rue E. Fermi, 13453 Marseille Cedex 13, France
c Multiphysics Methods Group, Advanced Nuclear Energy Systems Department, Idaho National Laboratory, P.O. Box 1625, Idaho Falls, ID 83415-3885, United States
a r t i c l e
i n f o
Article history:
Received 16 April 2008
Received in revised form 13 October 2008
Accepted 3 November 2008
Available online 13 November 2008
Keywords:
Hyperbolic systems
Multiﬂuid
Multiphase
Real gases
Cavitation
Multiphysic
Godunov
a b s t r a c t
Numerical approximation of the ﬁve-equation two-phase ﬂow of Kapila et al. [A.K. Kapila,
R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deﬂagration-to-deto-
nation transition in granular materials: reduced equations, Physics of Fluids 13(10) (2001)
3002–3024] is examined. This model has shown excellent capabilities for the numerical
resolution of interfaces separating compressible ﬂuids as well as wave propagation in com-
pressible mixtures [A. Murrone, H. Guillard, A ﬁve equation reduced model for compress-
ible two phase ﬂow problems, Journal of Computational Physics 202(2) (2005) 664–698;
R. Abgrall, V. Perrier, Asymptotic expansion of a multiscale numerical scheme for com-
pressible multiphase ﬂows, SIAM Journal of Multiscale and Modeling and Simulation (5)
(2006) 84–115; F. Petitpas, E. Franquet, R. Saurel, O. Le Metayer, A relaxation-projection
method for compressible ﬂows. Part II. The artiﬁcial heat exchange for multiphase shocks,
Journal of Computational Physics 225(2) (2007) 2214–2248]. However, its numerical
approximation poses some serious difﬁculties. Among them, the non-monotonic behavior
of the sound speed causes inaccuracies in wave’s transmission across interfaces. Moreover,
volume fraction variation across acoustic waves results in difﬁculties for the Riemann
problem resolution, and in particular for the derivation of approximate solvers. Volume
fraction positivity in the presence of shocks or strong expansion waves is another issue
resulting in lack of robustness. To circumvent these difﬁculties, the pressure equilibrium
assumption is relaxed and a pressure non-equilibrium model is developed. It results in a
single velocity, non-conservative hyperbolic model with two energy equations involving
relaxation terms. It fulﬁlls the equation of state and energy conservation on both sides
of interfaces and guarantees correct transmission of shocks across them. This formulation
considerably simpliﬁes numerical resolution. Following a strategy developed previously for
another ﬂow model [R. Saurel, R. Abgrall, A multiphase Godunov method for multiﬂuid and
multiphase ﬂows, Journal of Computational Physics 150 (1999) 425–467], the hyperbolic
part is ﬁrst solved without relaxation terms with a simple, fast and robust algorithm, valid
for unstructured meshes. Second, stiff relaxation terms are solved with a Newton method
that also guarantees positivity and robustness. The algorithm and model are compared to
exact solutions of the Euler equations as well as solutions of the ﬁve-equation model under
extreme ﬂow conditions, for interface computation and cavitating ﬂows involving dynam-
ics appearance of interfaces. In order to deal with correct dynamic of shock waves propa-
gating through multiphase mixtures, the artiﬁcial heat exchange method of Petitpas et al.
[F. Petitpas, E. Franquet, R. Saurel, O. Le Metayer, A relaxation-projection method for
0021-9991/$ - see front matter  2008 Elsevier Inc. All rights reserved.
doi:10.1016/j.jcp.2008.11.002
* Corresponding author. Address: Polytech’Marseille, Aix-Marseille University and SMASH Project UMR CNRS 6595 – IUSTI-INRIA, 5 rue E. Fermi, 13453
Marseille Cedex 13, France. Tel.: +33 339 128 8511; fax: +33 339 128 8322.
E-mail address: Richard.Saurel@polytech.univ-mrs.fr (R. Saurel).
Journal of Computational Physics 228 (2009) 1678–1712
Contents lists available at ScienceDirect
Journal of Computational Physics
journal homepage: www.elsevier.com/locate/jcp


compressible ﬂows. Part II. The artiﬁcial heat exchange for multiphase shocks, Journal of
Computational Physics 225(2) (2007) 2214–2248] is adapted to the present formulation.
 2008 Elsevier Inc. All rights reserved.
1. Introduction
Compressible multi-material ﬂows and multiphase mixtures arise in many natural and industrial situations including
bubble dynamics, shock wave interaction with material discontinuities, detonation of high energetic materials, hyperveloc-
ity impacts, cavitating ﬂows, combustion systems to name only a few. The motivation of the present work is the accurate and
computationally efﬁcient resolution of interface problems in extreme ﬂow conditions (high pressure ratios 107, high den-
sity ratios 103), as well as the computation of dynamic appearance of interfaces, that occur in cavitating ﬂows and spall-
ation phenomena. These interfaces are often separating pure media but also mixtures of materials in which wave dynamics
is also important. Such situations appear frequently in astrophysics, physics of explosives, nuclear physics, powder engineer-
ing and many other applications. The aim of the present paper is to develop a general formulation and algorithm to solve
interface problems separating compressible media or mixtures in extreme situations.
Godunov type schemes and variants have now reached a level of maturity to solve single phase ﬂows in the presence of
discontinuities. However, the presence of large discontinuities of thermodynamic variables and equations of state at material
interfaces result in numerical instabilities, oscillations and computational failure [24,1].
To circumvent these difﬁculties, two classes of methods have been developed:
– Methods that consider the interface as a sharp discontinuity (Sharp Interface Methods – SIM).
– Methods that consider the interface as a diffuse zone, like contact discontinuities in gas dynamics (Diffuse Interface
Methods – DIM).
The Lagrangian class of SIM is the most natural (see for example [21,14]). In this context, the computational mesh moves
and distorts with the material interface. However, when dealing with ﬂuid ﬂows, deformations are unbounded and resulting
mesh distortions can make the Lagrangian approach unpractical [46]. Eulerian methods use a ﬁxed mesh with an additional
equation for tracking or reconstructing the material interface. In the volume of ﬂuid (VOF) approach [20], each computa-
tional cell is assumed to possibly contain a mixture of both ﬂuids and the volume occupied by each ﬂuid is represented
by the volume fraction, transported with the ﬂow. This method is widely used for incompressible ﬂows as there is no special
thermodynamics to compute in mixture cells [19]. For compressible ﬂows, extra energy equations are used as well as pres-
sure relaxation procedures [7,32]. These methods seem efﬁcient as a result of subtle management at the discrete level of the
various equations. The literature does not provide a clear link of this discrete management to a given system of continuous
partial differential equations. In the present paper an attempt to clarify, improve and generalize these methods will be
developed.
Another class of popular Eulerian methods is based on the level-set equation [13,34,36,47] to locate the interface. Again,
for compressible ﬂows, special management of the interface is needed to guarantee interface conditions. Relevant work in
this direction was done by Fedkiw et al. [16] with the Ghost Fluid Method, Abgrall and Karni [2] with a simpliﬁed version
of this method and Khoo et al. [25]. This method is attractive for its apparent simplicity and versatility versus various prob-
lems of physics. However, its use in arbitrary conditions, with large pressure and density ratios does not seem obvious. More-
over, it is non-conservative regarding mixture variables (momentum and energy). The last class of SIM corresponds to Front
Tracking methods where the interface is explicitly tracked over a ﬁxed Eulerian mesh. Considerable efforts have been done to
develop computational codes employing this approach [18,30].
It is worth mentioning that none of these methods is able to dynamically create interfaces and to solve interfaces sepa-
rating pure media and mixtures.
The second type of methods (DIM) considers interfaces as numerically diffused zones, like contact discontinuities in gas
dynamics. Diffuse interfaces correspond to artiﬁcial mixtures created by numerical diffusion. A pioneering work in this direc-
tion was performed by Abgrall [1]. Determination of thermodynamic ﬂow variables in these zones is achieved on the basis of
multiphase ﬂow theory ([40,4,42,35,3,43,37]). The challenge is to derive physically, mathematically, and numerically consis-
tent thermodynamic laws for the artiﬁcial mixture. The key issue is to fulﬁll interface conditions within this artiﬁcial mix-
ture. This second category possesses several advantages:
– The same algorithm is implemented globally in both pure ﬂuids and in mixture zones. An extended hyperbolic system is
used to solve every location of the ﬂow.
– These models and methods are able to dynamically create interfaces that are not present initially, e.g. in cavitating ﬂows
where gas pockets dynamically appear in a liquid [41,29,45].
– These methods are also able to deal with interfaces separating pure ﬂuids and ﬂuid mixtures, e.g. in the computation of
detonation waves in condensed explosives where chemical decomposition produces multiphase mixtures of materials
[41,8,38].
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1679


Methods in this second category are based on hyperbolic multiphase ﬂow models, consisting of two main classes:
– Models for mixtures in total non-equilibrium: Baer and Nunziato [5] model and its variants, and
– Models for mixtures in mechanical equilibrium [49,23].
This paper deals with the building of a simple, robust, fast and accurate formulation for single velocity and single pressure
multiphase ﬂows. The Kapila et al. [23] model is of particular interest for the computation of interfaces separating compress-
ible ﬂuids, as well as barotropic and non-barotropic cavitating ﬂows. Speciﬁc numerical schemes have been derived recently
in Murrone and Guillard [35], Abgrall and Perrier [3], Saurel et al. [43], Petitpas et al. [37].
This model is apparently simple. In the context of two ﬂuids it is composed of two mass equations, a mixture momentum
equation and a mixture energy equation. These equations express in conservative formulation. The closure is achieved by the
pressure equilibrium condition that results in a differential transport equation for the volume fraction containing a non-con-
servative term, involving the velocity divergence and phasic bulk modulii. However this last equation poses serious compu-
tational challenges which include:
– Shock computations within the context of a non-conservative model.
– Volume fraction positivity, when dealing with shocks and strong expansion waves. The term involving a velocity diver-
gence in the volume fraction evolution equation is particularly difﬁcult to approximate [37]. This is particularly important
for the dynamic appearance of interfaces in cavitating ﬂows.
– Non-monotonic behavior of the sound speed [54] resulting in inaccurate wave transmission across diffuse interfaces. In
the diffuse interface the sound speed presents large variations resulting in wrong acoustic wave dynamics. The wave’s
chronology is thus in error, as will be shown in more details in the next paragraph.
Moreover, in order to consider future extensions with additional physics to reach multiphysics modeling of continuous
media with a multiphase approach, the computational efﬁciency of existing algorithms must be improved. The multiphysics
challenge we consider deals with:
– Sophisticated equations of state (EOS): Mie-Gruneisen for condensed materials, JWL for explosive products [28], etc.
– Granular materials that involve extra EOS expressing contact granular energy and contact pressure [6].
– Capillary effects modeling [39] with eventually phase transition [45].
– Interfaces separating compressible ﬂuids and elastic solids in extreme deformations [33,50,17,15]. This instance is partic-
ularly difﬁcult as the EOS for solids depends on the deformation tensor.
The present paper does not deal with all these extensions, but it is clear that such a goal needs simple and robust mul-
tiphase formulations. The present paper addresses this issue in the context of the simplest version of the Kapila et al. [23]
model.
The main difﬁculty with this model comes from the pressure equilibrium condition, which results in the non-conservative
equation for the volume fraction. A conservative formulation can be obtained with the help of the entropy equations. How-
ever, this conservative formulation is untenable in the presence of shocks.
To circumvent these difﬁculties, pressure non-equilibrium effects are restored in the Kapila et al. [23] model. This results
in a 6-equation model with a single velocity but with two pressures and associated relaxation terms. This extended model
was already presented as a ﬁrst reduction of the Baer and Nunziato [5] model in [23], but never considered for the descrip-
tion of diffuse interfaces. A seventh equation is added describing the mixture total energy in order to guarantee a correct
treatment of shocks in the single phase limit. This apparent complexity with an extended model actually leads to consider-
able simpliﬁcations regarding numerical resolution. Indeed, this model remains hyperbolic with only three characteristic
wave propagation speeds and volume fraction positivity is easily preserved. The building of a simple and efﬁcient method
for the numerical approximation of this ﬂow model in the context of diffuse interfaces is the aim of the present paper.
When relaxation terms are omitted the volume fraction remains constant across acoustic waves and the Riemann prob-
lem is easily solved with approximate Riemann solvers (acoustic and HLLC-type solvers, [52]). Moreover, the building of a
positivity preserving scheme guarantees robustness when considering cavitating ﬂows [26,48,37] where interfaces appear
dynamically. Dynamic appearance of these interfaces is a consequence of pressure relaxation, done at the end of each hyper-
bolic evolution step, in order to match asymptotically solutions of the Kapila et al. [23] reduced model.
This paper is organized as follows. In Section 2 the Kapila et al. [23] model is recalled and the non-equilibrium 6-equation
model is presented. This 6-equation model tends to the 5-equation model of Kapila et al. [23] in the limit of stiff pressure
relaxation. Basic properties of these models are presented: Entropy inequality and hyperbolicity. In Section 3 the numerical
method is built. Approximate Riemann solvers are presented for the hyperbolic part and a Godunov type scheme is built. The
pressure relaxation algorithm is also presented in this Section. Special attention is given to the role of the seventh equation
used to correct the computation of non-conservative energies in the single phase limit, on both sides of an interface. Various
test cases are presented in Section 4, together with validations against exact solutions of the Euler equations and of the 5-
equation model of Kapila et al. [23]. Some examples consider interfaces initially present in the ﬂow, while others involve
the dynamic appearance of interfaces. Section 5 presents the extension of the method to shock propagation in physical
1680
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


multiphase mixtures. This extension is not important for interfaces separating pure (or nearly pure) ﬂuids. But it has impor-
tance when the interface separates pure ﬂuids and mixtures of materials. Finally, conclusions and future investigations are
discussed in Section 6.
Difﬁculties are often reported to solve barotropic cavitating ﬂow models. The present method being general, it can also be
applied to this type of model. Thus, comparisons of the 6-equation model have been added with existing barotropic cavitat-
ing ﬂow models [53] in Appendix A. These models are recovered as limiting cases of the present 6-equation model. Moreover,
a simple algorithm is proposed to solve cavitating barotropic ﬂows.
2. Pressure equilibrium and non-equilibrium single velocity multiphase ﬂow models
The single velocity pressure equilibrium model corresponds to the one of Kapila et al. [23]. It has been obtained as the
asymptotic limit of the Baer and Nunziato [5] model in the limit of both stiff velocity and pressure relaxation. It involves
5 partial differential equations, one of them being non-conservative. Its resulting speed of sound corresponds to that of
[54] which exhibits a non-monotonic variation with volume fraction. These two difﬁculties (non-conservativity and non-
monotonicity) present serious computational challenges. To circumvent them, a pressure non-equilibrium 6-equation model
is constructed (ﬁrst reduced model in [23]), also non-conservative but easier to solve with a relaxation method. Both models
are presented hereafter.
2.1. Five-equation model
The Kapila et al. [23] is the zero-order approximation of the Baer and Nunziato [5] with stiff mechanical relaxation. It
reads in the context of two ﬂuids:
@a1
@t þ u @a1
@x ¼ q2c2
2  q1c2
1
q1c2
1
a1 þ q2c2
2
a2
@u
@x ;
@ðaqÞ1
@t
þ @ðaqÞ1u
@x
¼ 0;
@ðaqÞ2
@t
þ @ðaqÞ2u
@x
¼ 0;
@qu
@t þ @qu2 þ p
@x
¼ 0;
@qE
@t þ @ðqE þ pÞu
@x
¼ 0;
ðII:1Þ
where a, q, u, p, E ðE ¼ e þ 1
2 u2Þ, and e represent respectively the volume fraction, the mixture density, the velocity, the mix-
ture pressure, the mixture total energy and the mixture internal energy.
The mixture internal energy is deﬁned as
e ¼ Y1e1ðq1; pÞ þ Y2e2ðq2; pÞ
ðII:2Þ
and the mass fraction is given by: Yk ¼ ðaqÞk
q .
The mixture density is deﬁned by q = (aq)1 + (aq)2.
Each ﬂuid is governed by its own convex equation of state (EOS),
ek ¼ ekðqk; pÞ;
which allows the determination of the phases’ sound speed,
ck ¼ ckðqk; pÞ:
The mixture pressure p is determined by solving Eq. (II.2). In the particular case of ﬂuids governed by the stiffened gas EOS,
pk ¼ ðck  1Þqkek  ckp1k;
ðII:3Þ
the resulting mixture EOS reads,
pðq; e;a1;a2Þ ¼
qe 
a1c1p11
c11 þ a2c2p12
c21


a1
c11 þ
a2
c21
ðII:4Þ
It is straightforward to obtain the entropy equations:
dsk
dt ¼ 0;
k ¼ 1; 2:
Consequently, this model needs speciﬁc relations for its closure in the presence of shocks. In the limit of weak shocks, appro-
priate shock relations have been determined in [44]:
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1681


Yk ¼ Y0
k;
qðu  rÞ ¼ q0ðu0  rÞ ¼ m;
p  p0 þ m2ðv  v0Þ ¼ 0;
ek  e0
k þ p þ p0
2
ðvk  v0
kÞ ¼ 0;
ðII:5Þ
where r denotes the shock speed and the upperscript ‘0’ represents the unshocked state.
These relations have been intensively validated against a large experimental data base for weak and strong shocks in the
same reference.
Even equipped with these relations, this apparently simple model involves many difﬁculties:
– With the help of relations (II.5), it is possible to solve exactly or approximately the Riemann problem [37]. Even when this
solution is exact, it is shown in the same reference that convergence of a numerical scheme to the exact solution is extre-
mely difﬁcult as the system is non-conservative: The cell average of non-conservative variables has no physical sense. Cell
averages were replaced by a relaxation procedure in [43,37]. To reach convergence for shock propagating in multiphase
mixtures, artiﬁcial heat exchanges were needed in the shock layer [37].
– Another issue is related to the volume fraction positivity in the presence of shocks and even in the presence of strong rar-
efaction waves. Indeed, when dealing with liquid–gas mixtures for example, the liquid compressibility is so weak that the
pressure tends to become negative, resulting in computational failure in the gas sound speed computation. Such situation
occurs frequently in cavitation test problems.
– An extra difﬁculty is related to the mixture sound speed that obeys the Wood [54] formula
1
qc2
eq ¼
a1
q1c2
1 þ a2
q2c2
2. The mixture
sound speed has a non-monotonic variation with volume fraction, as shown in Fig. 1. Here ceq represents the mechanical
equilibrium mixture sound speed.
To illustrate the difﬁculties related to the non-monotonic sound speed in this model, numerical results obtained with the
method of Petitpas et al. [37] are recalled. This method solves interfaces as diffuse numerical zones with the help of a
Lagrange-relaxation algorithm. A 1-m long shock tube containing two chambers separated by an interface at the location
x = 0.8 m is considered. Each chamber contains a mixture of water and air. The initial density of the water is
qwater = 1000 kg m3 and the stiffened gas EOS parameters are cwater = 4.4 and p1,water = 6  108 Pa. The initial density of
air is qair = 10 kg m3 and EOS parameters are cair = 1.4 and p1,air = 0 Pa. The left chamber contains a very small volume frac-
tion of air aair = 106 and the pressure is equal to 109 Pa. The right chamber contains the same ﬂuids but the volume fractions
are reversed. Its pressure is equal to 105 Pa. In both chambers the initial velocity is zero. The exact solution of the single
phase Euler equations and the multiphase ﬂow model with ﬁve equations are compared in Fig. 2 at time t = 220 ls.
A bad consequence of the Wood [54] speed of sound appears when a pressure wave interacts with a diffuse interface. To
illustrate this difﬁculty, let us consider the advection of a water–air interface at the velocity of 50 m/s. The numerical solu-
tion of this advection test with a ﬁrst-order accuracy method is shown in Fig. 3 where the behaviors of the equilibrium speed
of sound [54] and another mixture sound speed (frozen) are compared. The frozen speed of sound is deﬁned by
c2
f ¼ Ywaterc2
water þ Yairc2
air and will appear as a major feature of the non-equilibrium 6-equation model.
It is clear that the use of the equilibrium speed of sound creates a zone where the speed of sound is lower than those of
the two initial media. This may have serious consequences regarding wave’s propagation. To illustrate the difﬁculty let us
consider the interaction of an acoustic wave with this diffused interface. When a wave propagates through the interface,
Fig. 1. Representation of the mixture equilibrium speed of sound ð1=qc2
eq ¼ a1=q1c2
1 þ a2=q2c2
2Þ of the 5-equation model for the liquid water–air mixture
under atmospheric conditions.
1682
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


it crosses a ﬁrst zone with stiff variation of the sound speed, possibly resulting in partial diffraction. The transmitted wave
propagates in the numerical diffusion zone with very low velocity before reaching the second stiff variation of sound speed,
resulting in a second diffraction. These various effects (multiple diffractions) and low sound speed induce delay for the
wave’s transmission (Fig. 4).
The method developed in the present paper is aimed to improve accuracy, robustness and computational efﬁciency of
existing methods for the Kapila et al. [23] model regarding:
– Volume fraction positivity. This is a particularly difﬁcult issue when dealing with dynamic appearance of interfaces in
nearly pure liquids and solids.
– Computation of cavitating ﬂows. These ﬂows involve extra difﬁculties related to the drastic Mach number evolutions
ranging from 0.01 to 100 [26,11,48]. These references report this problem in the simpler context of a cavitation model
in conservative form that will be examined in Appendix A.
– Riemann problem resolution, that is quite difﬁcult to solve with the Wood [54] sound speed.
Fig. 2. Liquid/gas shock tube. The Lagrange-relaxation method (symbols) of Petitpas et al. [37] is compared to the exact solution (solid). A 1000 cells mesh is
used. The density ratio is 100 and the pressure ratio is 10,000 at the initial discontinuity. A Mach oscillation appears in the numerical diffusion zone at the
interface and is due to the non-monotonic behavior of the speed of sound of this model.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1683


– Wave transmission through diffuse interfaces, as presented in Fig. 4.
– Computations on unstructured meshes. Not all existing methods for the Kapila et al. [23] model are able to deal with
unstructured meshes.
A pressure non-equilibrium model is considered in these aims.
2.2. Six-equation model
The 6-equation model is also derived from the 7-equation model of Baer and Nunziato [5] in the asymptotic limit of stiff
velocity relaxation only (ﬁrst reduced model in [23]). Pressure non-equilibrium effects are maintained. The 6-equation mod-
el should not be considered as a physical model, but more as a step-model to solve the 5-equations model (second reduced
model of [23]). Indeed, the model with 6-equations has better properties for numerical approximations than the mechanical
equilibrium one:
– Positivity of the volume fraction is easily preserved.
– The mixture sound speed has a monotonic behavior which seems to be more attractive regarding diffused interfaces and
acoustic wave transmission.
These two properties are key points for the building of a simple, robust and accurate hyperbolic solver. Moreover, with
proper treatment of relaxation terms, solutions of the 5-equation model will be recovered.
2.2.1. Flow model
The 6-equation model reads:
@a1
@t þ u @a1
@x ¼ lðp1  p2Þ;
Fig. 3. Comparison of equilibrium (lines) and frozen (dashed lines) speed of sound during numerical advection of a water–air interface. In the numerical
diffusion zone at the interface the equilibrium speed of sound is lower than in pure ﬂuids. This may have serious consequences on wave transmission (in
particular regarding chronology) when a pressure wave interacts with this diffused zone.
Fig. 4. Schematic representation in the (x, t) diagram of the interaction between an acoustic wave and the numerical diffusion zone of an interface
computed with the equilibrium speed of sound [54]. In the numerical diffusion zone, the transmitted wave propagates at a lower velocity than in the pure
ﬂuids. This induces a delay s in the wave’s transmission through the interface.
1684
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


@a1q1
@t
þ @a1q1u
@x
¼ 0;
@a2q2
@t
þ @a2q2u
@x
¼ 0;
@qu
@t þ @qu2 þ ða1p1 þ a2p2Þ
@x
¼ 0;
@a1q1e1
@t
þ @a1q1e1u
@t
þ a1p1
@u
@x ¼ pIlðp1  p2Þ;
@a2q2e2
@t
þ @a2q2e2u
@t
þ a2p2
@u
@x ¼ pIlðp1  p2Þ:
ðII:6Þ
The interfacial pressure pI is obtained as the asymptotic limit of the interfacial pressure of the symmetric non-equilibrium
model with 7-equations of Saurel et al. [42]. This estimate in the limit of equal velocities reads:
pI ¼ Z2p1 þ Z1p2
Z1 þ Z2
;
where Zk = qkck represents the acoustic impedance of phase k.
The combination of the two internal energy equations with mass and momentum equations results in the additional mix-
ture energy equation:
@qðY1e1 þ Y2e2 þ 1
2 u2Þ
@t
þ @u qðY1e1 þ Y2e2 þ 1
2 u2Þ þ ða1p1 þ a2p2Þ


@x
¼ 0:
ðII:7Þ
This extra equation will be important during numerical resolution, in order to correct inaccuracies due to the numerical
approximation of the two non-conservative internal energy equations in the presence of shocks.
There is no difﬁculty to check that the second law of thermodynamics is fulﬁlled by this model. The phasic entropy equa-
tions are readily obtained:
a1q1T1
ds1
dt ¼ lðp1  p2Þ2
Z1
Z1 þ Z2
;
a2q2T2
ds2
dt ¼ lðp1  p2Þ2
Z2
Z1 þ Z2
;
insuring that the mixture entropy (s = Y1s1 + Y2s2) always evolve with positive or null variations.
This model exhibits a nice feature with respect to the mixture sound speed. The mixture sound speed,
c2
f ¼ Y1c2
1 þ Y2c2
2;
has a monotonic behavior versus volume and mass fractions and represents the frozen mixture sound speed.
The model is thus strictly hyperbolic with waves speeds: u + cf, u  cf, u. A more detailed analysis of hyperbolicity and
sound speed will be carried out in Section 3 with the approximate acoustic Riemann solver.
2.2.2. About shock relations
As with the previous 5-equation model, the new model is also non-conservative, and shock relations have to be pre-
scribed. However, the preceding remarks about shock relations for the ﬁve equations model and numerical approximation
of shocks with non-conservative systems yield the following conclusion:
Even when shock relations are known or accepted for a non-conservative system, it is very difﬁcult to make the numerical solu-
tion converge to the end shock state solution.
There is thus no need to determine precise shock relations for the 6-equation model, in particular since it is intended only
to approximate the 5-equation model for which shock relations are known.
However, some admissibility conditions have to be respected by a given Hugoniot approximate model. Jump conditions
must at least respect [44]:
– Energy conservation of the mixture.
– Tangency of the mixture Hugoniot curve and mixture isentrope.
– Single phase limit for which jump conditions are unambiguously known.
– Symmetry.
– Entropy production.
Jump conditions for the mass equations are
a1q1ðu  rÞ ¼ a0
1q0
1ðu0  rÞ ¼ m1;
a2q2ðu  rÞ ¼ a0
2q0
2ðu0  rÞ ¼ m2:
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1685


Let us denote the mixture pressure by p = a1p1 + a2p2 and the shock mass ﬂow rate by m = m1 + m2. With these notations, the
momentum jump condition can be written
p  p0 þ m2ðv  v0Þ ¼ 0:
The mixture energy jump condition is:
e  e0 þ p þ p0
2
ðv  v0Þ ¼ 0;
with e = Y1e1 + Y2e2 and v = Y1v1 + Y2v2 ðvk ¼ 1
qkÞ.
In the absence of relaxation effects the volume fraction jump is simply:
a1 ¼ a0
1:
The non-conservative internal energy equations are not adapted to the determination of jump conditions. Following the pre-
ceding admissibility conditions the following jump conditions are proposed:
ek  e0
k þ pk þ p0
k
2
ðvk  v0
kÞ ¼ 0:
ðII:8Þ
The conditions that must be satisﬁed include:
 Energy conservation
The sum of the internal energy jump equations yields:
Y1ðe1  e0
1Þ þ p1 þ p0
1
2
ðY1v1  Y1v0
1Þ þ Y2ðe2  e0
2Þ þ p2 þ p0
2
2
ðY2v2  Y2v0
2Þ ¼ 0:
As Yk ¼ akv
vk , we have:
e  e0 þ p1 þ p0
1
2
ða1v  a0
1v0Þ þ p2 þ p0
2
2
ða2v  a0
2v0Þ ¼ 0:
With the volume fraction jump relation, this equation becomes
e  e0 þ a1p1 þ a2p2 þ a1p0
1 þ a2p0
2
2
ðv  v0Þ ¼ 0;
or simply
e  e0 þ p þ p0
2
ðv  v0Þ ¼ 0:
This result guarantees that the phasic energy jump conditions are compatible with the mixture energy conservation.
 Tangency of the mixture Hugoniot curve and isentrope
This is a mandatory property for the Riemann problem solution. As the volume fraction is constant across shocks and rar-
efaction waves (in absence of relaxation effects) and the phasic Hugoniots are tangent to phasic isentropes, the mixture
Hugoniot is necessarily tangent to the mixture isentrope.
 Single phase limit
When one of the phases disappears the energy jump condition of the remaining ﬂuid is in agreement with the single
phase energy jump.
 Symmetry
Symmetry in the formulation allows an easy extension to an arbitrary number of ﬂuids.
 Entropy production
As each phase evolves along its own Hugoniot (II.8) there is no doubt that the mixture entropy evolves positively.
Through application of these relations, the Riemann problem can now be solved. Numerical issues pertaining to the Rie-
mann problem solution are addressed in the next section. Let us insist on the fact that jump conditions are not the key to
shock computation in multiphase mixtures. It has been shown that even when shock relations are known, the convergence
of a numerical scheme to the exact solution is very difﬁcult. This is due to the lack of deﬁnition for cell averages of non-con-
servative variables [37].
1686
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


2.2.3. Asymptotic limit
As the method will solve the 6-equation model with stiff relaxation terms, it is important to check that in the limit of
inﬁnitely fast pressure relaxation the 5-equation model is recovered. This proof is given in Appendix B.
3. Numerical method
Numerical resolution of the 6-equation model in the limit of stiff pressure relaxation is addressed in the present section.
In regular zones, this model is self consistent. But in the presence of shocks the internal energy equations are inappropriate.
To correct the thermodynamic state predicted by these equations in the presence of shocks, the total mixture energy equa-
tion will be used. This correction will be valid on both sides of an interface, when the ﬂow tends to the single phase limits.
The details of this correction will be examined further. For now, the 6-equation system is augmented by a redundant equa-
tion regarding the total mixture energy. The system to consider during numerical resolution thus involves seven equations:
@a1
@t þ u @a1
@x ¼ lðp1  p2Þ;
@a1q1e1
@t
þ @a1q1e1u
@x
þ a1p1
@u
@x ¼ pIlðp1  p2Þ;
@a2q2e2
@t
þ @a2q2e2u
@x
þ a2p2
@u
@x ¼ pIlðp1  p2Þ;
@a1q1
@t
þ @a1q1u
@x
¼ 0;
@a2q2
@t
þ @a2q2u
@x
¼ 0;
@qu
@t þ @qu2 þ ða1p1 þ a2p2Þ
@x
¼ 0;
@qðY1e1 þ Y2e2 þ 1
2 u2Þ
@t
þ @u qðY1e1 þ Y2e2 þ 1
2 u2Þ þ ða1p1 þ a2p2Þ


@x
¼ 0
ðIII:1Þ
with pI ¼ Z2p1þZ1p2
Z1þZ2
and appropriate equations of state ek = ek(qk,pk).
This system is equipped with the approximate shock relations of the preceding section, in particular relation (II.8).
3.1. Approximate Riemann solvers
Two types of approximate Riemann solvers will be considered:
– Acoustic linearized Riemann solver,
– HLLC Riemann solver.
These two solvers are detailed in the context of the Euler equations in Toro [52].
3.1.1. Acoustic solver
This approximate solver assumes that shocks are absent or sufﬁciently weak. The last equation of system (III.1) can thus
be suppressed. Indeed, this last equation is only used to correct some deﬁciencies of the numerical resolution of phase’s
internal energy equations in the presence of shocks. The 6-equation system free of relaxation terms can thus be written with
the following variables:
@W
@t þ AðWÞ @W
@x ¼ 0;
with W = (a1,s1,s2,u,p1,p2)T and,
AðWÞ ¼
u
0
0
0
0
0
0
u
0
0
0
0
0
0
u
0
0
0
p1p2
q
0
0
u
a1
q
a2
q
0
0
0
q1c2
1
u
0
0
0
0
q2c2
2
0
u
0
B
B
B
B
B
B
B
B
B
@
1
C
C
C
C
C
C
C
C
C
A
:
Eigenvalues of the propagation matrix are:
k0 = u, four times fold, k1 = u  c, k2 = u + c, with,
c2 ¼ Y1c2
1 þ Y2c2
2
ðIII:2Þ
The frozen sound speed introduced in Section 2 is now established.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1687


The acoustic solver is based on characteristic equations that are readily obtained:
– Along trajectories deﬁned by dx
dt ¼ u
da1
dt ¼ 0;
ds1
dt ¼ 0;
ds2
dt ¼ 0:
– Along trajectories deﬁned by dx
dt ¼ u  c
 p1  p2
qc
da1
dt

uc
þ du
dt

uc
 a1
qc
dp1
dt

uc
 a2
qc
dp2
dt

uc
¼ 0
– Along trajectories deﬁned by dx
dt ¼ u þ c
p1  p2
qc
da1
dt

uþc
þ du
dt

uþc
þ a1
qc
dp1
dt

uþc
þ a2
qc
dp2
dt

uþc
¼ 0:
These relations are used to solve the linearized Riemann problem. By assuming weak variations across left- and right-facing
waves, the acoustic impedance Z = qc (with c deﬁned by (III.2) and q the mixture density) are assumed constant. The corre-
sponding jump relations are:
– Across a right-facing wave,
a
1R ¼ a1R; s
1R ¼ s1R; s
2R ¼ s2R;
ða1p1 þ a2p2Þ
R  ZRu
R ¼ ða1p1 þ a2p2ÞR  ZRuR
with ZR ¼ qRcR:
– Across a left-facing wave,
a
1L ¼ a1L; s
1L ¼ s1L; s
2L ¼ s2L;
ða1p1 þ a2p2Þ
L þ ZLu
L ¼ ða1p1 þ a2p2ÞL þ ZLuL
with ZL ¼ qLcL:
The upperscript ‘*’ stands for the perturbated state.
The velocity and pressure solution of the Riemann problem are thus easily obtained with the help of the interface
conditions:
ða1p1 þ a2p2Þ
L ¼ ða1p1 þ a2p2Þ
R ¼ ða1p1 þ a2p2Þ ¼ p;
u
L ¼ u
R ¼ u:
The velocity and pressure solution of the Riemann problem read:
u ¼ ZLuL þ ZRuR þ pL  pR
ZL þ ZR
;
p ¼ ZLpR þ ZRpL þ ZRZLðuL  uRÞ
ZL þ ZR
:
ðIII:3Þ
With
p ¼ a1p1 þ a2p2;
Z ¼ qc;
q ¼ a1q1 þ a2q2;
c2 ¼ Y1c2
1 þ Y2c2
2:
Relations (III.3) are the same for the 6-equation model and for the Euler equations. The differences appear through the def-
initions of the mixture pressure, mixture sound speed and mixture density.
Once the pressure is determined in the star region the phase’s densities are determined with the help of the entropy
jumps.
This solver is simple and efﬁcient for subsonic ﬂows or ﬂows in absence of strong shocks. Characteristic relations are also
useful for boundary conditions treatment. But we prefer a solver able to deal with arbitrary shocks, genuinely positive (and
consequently robust), able to deal with arbitrary convex EOS. The HLLC solver of Toro et al. [51] fulﬁls these requirements.
3.1.2. HLLC-type solver
Consider a cell boundary separating a left state (L) and a right state (R). The left- and right-facing waves speeds are readily
obtained, following Davis [12] estimates:
SR ¼ maxðuL þ cL; uR þ cRÞ;
SL ¼ minðuL  cL; uR  cRÞ;
where the sound speed still obeys to Relation (III.2).
The speed of the intermediate wave (or contact discontinuity) is estimated using the HLL approximation
SM ¼ ðqu2 þ pÞL  ðqu2 þ pÞR  SLðquÞL þ SRðquÞR
ðquÞL  ðquÞR  SLqL þ SRqR
;
with the mixture density and mixture pressure deﬁned previously.
1688
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


From these wave speeds, the following variable states are determined
ðakqkÞ
R ¼ ðakqkÞR
SR  uR
SR  SM
;
ðakqkÞ
L ¼ ðakqkÞL
SL  uL
SL  SM
;
p ¼ pR þ qRuRðuR  SRÞ  q
RSMðSM  SRÞ;
with q
R ¼
X
k
ðakqkÞ
R;
E
R ¼ qRERðuR  SRÞ þ pRuR  pSM
q
RðSM  SRÞ
;
E
L ¼ qLELðuL  SLÞ þ pLuL  pSM
q
LðSM  SLÞ
;
with E ¼ Y1e1 þ Y2e2 þ 1
2 u2:
The volume fraction jump is readily obtained, as in the absence of relaxation effects the volume fraction is constant along
ﬂuid trajectories
a
kR ¼ akR;
a
kL ¼ akL:
As the volume fraction is constant across left- and right-facing waves, the ﬂuid density is determined from the preceding
relations:
q
kR ¼ qkR
uR  SR
SM  SR
:
Internal energy jumps are determined with the help of the Hugoniot relation (II.8). Let us consider the example of ﬂuids gov-
erned by the stiffened gas EOS (II.3). With the help of the EOS, the phasic pressures are constrained along their Hugoniot
curves to be functions only of the corresponding phase density:
p
kðq
kÞ ¼ ðpk þ p1kÞ ðck  1Þqk  ðck þ 1Þq
k
ðck  1Þq
k  ðck þ 1Þqk
 p1k:
The phase’s internal energies are then determined from the EOS: e
kR ¼ e
kRðp
k;q
kÞ.
Equipped with these approximate Riemann solvers, the next step is to develop a Godunov type scheme.
3.2. Godunov type method
For the sake of simplicity, the method is presented at ﬁrst-order. The extension to second-order is detailed in Appendix C.
3.2.1. First-order method
In the absence of relaxation terms, the conservative part of System (III.1) is updated with the conventional Godunov
scheme:
Unþ1
i
¼ Un
i  Dt
Dx ðFðUn
i ; Un
iþ1Þ  FðUn
i1; Un
i ÞÞ;
where U = ((aq)1, (aq)2, qu, qE)T and F = ((aq)1u, (aq)1u, qu2 + p, (qE + p)u)T, E ¼ Y1e1 þ Y2e2 þ 1
2 u2and p = a1p1 + a2p2.
The volume fraction equation is also updated using the Godunov method for advection equations:
anþ1
1i
¼ an
1i  Dt
Dx ððua1Þ
iþ1=2  ðua1Þ
i1=2  an
1iðu
iþ1=2  u
i1=2ÞÞ:
This scheme guarantees volume fraction positivity during the hyperbolic step. Other options are possible, like for example,
VOF type methods [32]. Using a reconstruction algorithm may have nice features when dealing with interfaces only, these
interfaces having to be present at the initial time. As we also deal with dynamic appearance of interfaces, a capturing method
is preferred. This is not the only difference between the Miller and Puckett [32] method and the present one. The mixture
pressure and sound speed used in the present formulation are very different from the single phase estimates used by these
authors.
Regarding the non-conservative energy equations, there is no hope to determine accurate approximation in the presence
of shocks [22]. Therefore, we use the simplest approximation of the corresponding equations by assuming the product ðapÞn
ki
constant during the time step:
ðaqeÞnþ1
ki
¼ ðaqeÞn
ki  Dt
Dx ððaqeuÞ
kiþ1=2  ðaqeuÞ
ki1=2 þ ðapÞn
kiðu
iþ1=2  u
i1=2ÞÞ:
The lack of accuracy in the internal energy computation resulting from the present scheme is not so crucial. The internal
energies will be used only to estimate the phase’s pressure at the end of the hyperbolic step, before the relaxation one.
The relaxation step will give a ﬁrst correction to the internal energies, in agreement with the second law of thermodynamics.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1689


A second correction will be made with the help of the total mixture energy. The details of these two steps are described in the
next two subsections. Before giving these details, let us examine a basic situation of fundamental importance when dealing
with interface problems; namely uniform ﬂow conditions.
3.2.2. Uniform ﬂow test
The main difﬁculty in solving interface problems as diffused numerical zones lies in the building of a ﬂow model and a
numerical scheme that preserve interface conditions. The uniform ﬂow test problem was proposed by Abgrall [1] in the con-
text of the Euler equations. Let us consider a one-dimensional ﬂow in mechanical equilibrium. A volume fraction disconti-
nuity propagates at constant velocity u in a constant pressure ﬂow ﬁeld p1 = p2 = p. This ﬂow system is initially in mechanical
equilibrium and therefore must remain in mechanical equilibrium during its time evolution.
Let us examine the behavior of the present Godunov method for the conservative part of this model in the particular case
of uniform pressure and velocity ﬁelds. The Godunov method for the mass equations is:
ðaqÞnþ1
ki
¼ ðaqÞn
ki  Dt
Dx ððaquÞ
kiþ1=2  ðaquÞ
ki1=2Þ;
k ¼ 1; 2:
Because the velocity is uniform we have:
ðaqÞnþ1
ki
¼ ðaqÞn
ki  Dt
Dx uððaqÞ
kiþ1=2  ðaqÞ
ki1=2Þ:
The mixture density thus obeys to the discrete formula:
qnþ1
i
¼ qn
i  Dt
Dx uðq
iþ1=2  q
i1=2Þ:
The discrete momentum equation under the same uniform ﬂow conditions becomes
ðquÞnþ1
i
¼ ðquÞn
i  Dt
Dx u2ðq
iþ1=2  q
i1=2Þ:
That is
ðquÞnþ1
i
¼ uðqÞnþ1
i
:
Thus the ﬂow will necessarily retain its uniform velocity at the next time step: unþ1
i
¼ u.
The adopted numerical scheme for the internal energies becomes in the present situation
ðaqeÞnþ1
ki
¼ ðaqeÞn
ki  Dt
Dx uððaqeÞ
kiþ1=2  ðaqeÞ
ki1=2Þ:
Consider, for example, the stiffened gas (SG) EOS (II.3): qkek ¼ pkþckp1k
ck1 .
The discrete approximation of the internal energy now becomes
a p þ cp1
c  1

nþ1
ki
¼
a p þ cp1
c  1

n
ki
 Dt
Dx u
a p þ cp1
c  1


kiþ1=2
 a p þ cp1
c  1


ki1=2
 
!
:
As the EOS parameters are constant in each ﬂuid, this expression simpliﬁes to:
ðaðp þ cp1ÞÞnþ1
ki
¼ ðaðp þ cp1ÞÞn
ki  Dt
Dx uððaðp þ cp1ÞÞ
kiþ1=2  ðaðp þ cp1ÞÞ
ki1=2Þ;
which can be rewritten as
ðapÞnþ1
ki
þ ðcp1ÞkðaÞnþ1
i
¼ p ðaÞn
ki  Dt
Dx uððaÞ
kiþ1=2  ðaÞ
ki1=2Þ

	
þ ðcp1Þk
ðaÞn
ki  Dt
Dx uððaÞ
kiþ1=2  ðaÞ
ki1=2Þ

	
:
The adopted numerical scheme for the volume fraction evolution, in uniform velocity ﬂow conditions becomes:
anþ1
ki
¼ an
ki  Dt
Dx uððakÞ
iþ1=2  ðakÞ
i1=2Þ:
Using this, the internal energy equation reduces to:
pnþ1
ki
¼ p:
The adopted numerical approximation thus preserves interface conditions in mechanical equilibrium ﬂows.
When the EOS are more sophisticated than the SG one, i.e. Mie Gruneisen EOS for example that can be written under the
form,
qkek ¼ pk þ ckp1kðqkÞ
ck  1
:
1690
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


The same properties of interface preserving are observed experimentally. The reason is that Godunov type methods used for
mass and volume fraction equations result in prolonged density ﬁeld through the interface. Locally, these more sophisticated
EOS thus reduce to the SG one.
3.3. Relaxation step
This step is of major importance to fulﬁll interface conditions in non-uniform velocity and pressure ﬂows. It also forces the
solution of the 6-equation model to converge to that of the 5-equation model.
In the relaxation step we must solve
@a1
@t ¼ lðp1  p2Þ;
@a1q1e1
@t
¼ pIlðp1  p2Þ;
@a2q2e2
@t
¼ pIlðp1  p2Þ;
@a1q1
@t
¼ 0;
@a2q2
@t
¼ 0;
@qu
@t ¼ 0;
@qE
@t ¼ 0
with pI ¼ Z2p1þZ1p2
Z1þZ2
and in the limit l ? +1.
After some manipulations the internal energy equations become
@e1
@t þ pI
@v1
@t ¼ 0;
@e2
@t þ pI
@v2
@t ¼ 0:
This system can be written in integral formulation
ek  e0
k þ ^pIkðvk  v0
kÞ ¼ 0;
where ^pIk ¼
1
vkv0
k
R Dt
0 pI
@vk
@t dt.
Determination of pressure averages ^pIk has to be done in agreement with thermodynamic considerations. By summing the
internal energy equations we have:
Y1e1  Y1e0
1 þ Y2e2  Y2e0
2 þ ^pI1ðY1v1  Y1v0
1Þ þ ^pI2ðY2v2  Y2v0
2Þ ¼ 0:
The mixture mass equation can be written as
ðY1v1  Y1v0
1Þ þ ðY2v2  Y2v0
2Þ ¼ 0:
Using these relations the mixture energy equation becomes
e  e0 þ ð^pI1  ^pI2ÞðY1v1  Y1v0
1Þ ¼ 0:
In order that the mixture energy conservation be fulﬁlled it is necessary that: ^pI1 ¼ ^pI2 ¼ ^pI. Possible estimates are ^pI ¼ p0
I or
^pI ¼ p, the initial and relaxed pressures respectively. These estimates are compatible with the entropy inequality [43]. With
regard to the choice of one or the other estimate, upon computation of the relaxed state the resulting difference in practical
computations is negligible. This negligible inﬂuence will be illustrated in the results section. The system to solve is thus com-
posed of equations
ekðp;vkÞ  e0
kðp0
k;v0
kÞ þ ^pIðvk  v0
kÞ ¼ 0;
k ¼ 1; 2;
which involves 3 unknowns, vk(k = 1,2) and p. Its closure is achieved using the saturation constraint
X
k
ak ¼ 1;
or
X
k
ðaqÞkvk ¼ 1:
Here the (aq)k are constant during the relaxation process. This system can be replaced by a single equation with a single
unknown (p). With the help of the EOS (II.3) the energy equations become
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1691


vkðpÞ ¼ v0
k
p0 þ ckp1k þ ðck  1Þ^pI
p þ ckp1k þ ðck  1Þ^pI
;
and thus the only equation to solve (for p) is
X
k
ðaqÞkvkðpÞ ¼ 1:
ðIII:4Þ
Once the relaxed pressure is found, the phase’s speciﬁc volumes and volume fractions are determined.
In the Miller and Puckett [32] method, the relaxed pressure is used to advance the solution to the next time step. How-
ever, there is no guarantee that the mixture EOS or the mixture energy be in agreement with this relaxed pressure. In order
to respect total energy and correct shock dynamics on both sides of the interface, the following correction is employed.
3.4. Reinitialization step
As the volume fractions have been estimated previously by the relaxation method, the mixture pressure can be deter-
mined from the mixture EOS based on the mixture energy which is known from the solution of the total energy equation.
Because the mixture total energy obeys a conservation law, its evolution is accurate in the entire ﬂow ﬁeld and in particular
at shocks.
Again considering ﬂuids governed by the stiffened gas EOS, the mixture EOS in this context relates mixture energy, den-
sity and volume fractions (II.4):
pðq; e;a1;a2Þ ¼
qe 
a1c1p11
c11 þ a2c2p12
c21


a1
c11 þ
a2
c21
:
This EOS is valid in pure ﬂuids and in the diffuse interface zone. As it is valid in pure ﬂuids, and based on the total energy
equation, it guarantees correct and conservative wave dynamics on both sides of the interface. Inside the numerical diffusion
zone of the interface, numerical experiments show that the method is accurate too, as the volume fractions used in the mix-
ture EOS (II.4) have a quite accurate prediction from the relaxation method.
Once the mixture pressure is determined from (II.4) the internal energies of the phases are reinitialized with the help of
their respective EOS before going to the next time step
ek ¼ ekðp;akqk;akÞ:
ðIII:5Þ
3.5. Summary
The numerical method can be summarized as follows:
– At each cell boundary solve the Riemann problem of System (III.1) with favorite solver. The HLLC solver of Section 3.1 is
recommended.
– Evolve all ﬂow variables with the Godunov type method of Section 3.2.
– Determine the relaxed pressure and especially the volume fraction by solving Eq. (III.4). The Newton method is appro-
priate for this task.
– Compute the mixture pressure with Eq. (II.4).
– Reset the internal energies with the computed pressure with the help of their respective EOS (III.5).
– Go to the ﬁrst item for the next time step.
4. Tests and validations
4.1. Advection of an interface in a uniform pressure and velocity ﬂow
A discontinuity of volume fraction (thus a mixture density discontinuity) is moving in a uniform pressure and velocity
ﬂow at 100 m/s. Initially the discontinuity is located at x = 0.5 m in a 1 m length tube. This discontinuity separates two nearly
pure ﬂuids, liquid water on the left deﬁned by qwater = 1000 kg m3, and the stiffened gas EOS parameters cwater = 4.4,
p1,water = 6  108 Pa and air on the right deﬁned by qair = 10 kg m3 with the ideal gas EOS parameters cair = 1.4 and p1,
air = 0 Pa. In the left chamber, the water volume fraction is set to awater = 1  e and in the right chamber its value is awater = e,
with e = 108. The uniform pressure is set equal to p = 105 Pa.
The numerical solution is plotted in Fig. 5 at time t = 2.79 ms and is compared to the exact one. A mesh with 200 uniform
cells is used with a second-order extension of the method (see Appendix C for details).
The agreement between the numerical and analytical solutions is excellent and the numerical solution is oscillation free,
except for the Mach number, computed with the equilibrium sound speed ceq. For this test case, the ﬂow being in mechanical
equilibrium, relaxation terms present in the volume fraction and energy equations have no importance, as well as the pres-
1692
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


sure relaxation step. The respect of interface conditions is just a consequence of the clean numerical approximation with the
Godunov method of conservative and non-conservative equations of Section 3.2.
4.2. Shock tube with Mie-Grüneisen type EOS
In order to show the method’s capabilities, in particular when dealing with more general equations of state, a test involv-
ing the [9] EOS (CC EOS) is considered. This EOS is of Mie-Grüneisen type. The same shock tube problem presented in Saurel
et al. [43] is considered. In this example, a single ﬂuid is considered governed by CC EOS, with a density discontinuity in a
shock tube. As there is a single ﬂuid, the Godunov method is expected to be valid. However, it was shown in the same
reference that due to the nonlinearity of p1(q) in the EOS, the Godunov method produced pressure and velocity oscillations.
A cure to these difﬁculties was proposed in that same reference. Here, with the help of the multiphase ﬂow model, these
Fig. 5. Advection of a volume fraction discontinuity in a uniform pressure and velocity ﬂow. Comparison of the relaxation method with Superbee limiter
(symbols) and the exact solution (solid). A 200 cells mesh is used. Excellent agreement is observed.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1693


difﬁculties can be solved by considering the single ﬂuid as a two-phase media with the initial discontinuity in the shock tube
separating the two states.
Initially, the high pressure chamber is set to 20 GPa, while the pressure is set equal to 0.2 MPa in the low pressure cham-
ber. Both chambers are ﬁlled with liquid nitromethane, governed by the CC EOS in which densities are respectively set to
1134 kg m3 and 1200 kg m3. In the high pressure chamber, volume fraction of the ﬁrst phase is set to a1 = 1  e and in
the right chamber its value is a1 = e (e = 108). Thus, the model is used in the single phase limit, i.e. the same EOS is used
for both ﬂuids but with different initial densities:
pðq; eÞ ¼ qCðe  ekðqÞÞ þ pkðqÞ;
with
ekðqÞ ¼
A1
qref ðE1  1Þ
q
qref
 
!E11

A2
qref ðE2  1Þ
q
qref
 
!E21
;
pkðqÞ ¼ A1
q
qref
 
!E1
 A2
q
qref
 
!E2
:
The data used in the present simulation are: C = 1.19, qref = 1134 kg m3, A1 = 0.819181  109 Pa, A2 = 1.50835  109 Pa,
E1 = 4.52969 and E2 = 1.42144.
The solution is presented at time t = 67 ls in Fig. 6. The present relaxation method is compared to the exact solution of the
Euler equations. Results are similar to those of Saurel et al. [43] but the present algorithm is easier to implement. A magniﬁed
view of pressure and velocity around the contact discontinuity is given in Fig. 7. It presents a solution free of oscillations.
4.3. Water–air shock tubes
4.3.1. Water–air shock tube with moderate pressure ratio and high density ratio
A 1 m long shock tube containing two chambers separated by an interface at the location x = 0.75 m is considered. Each
chamber contains a nearly pure ﬂuid. The initial density of water is qwater = 1000 kg m3 and the stiffened gas EOS param-
eters are cwater = 4.4 and p1,water = 6  108 Pa. The initial density of air is qair = 1 kg m3 and EOS parameters are cair = 1.4 and
Fig. 6. Shock tube with Mie-Grüneisen type EOS. The present relaxation method based on the 6-equation model (symbols) is compared to the exact solution
of the Euler equations (solid). A 500 cells mesh is used. Results are in perfect agreement.
1694
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


p1, air = 0 Pa. The left chamber contains a very small volume fraction of air aair = 106 and the initial pressure is set equal to
1 GPa. The right chamber contains the same ﬂuids but the volume fractions are reversed. The initial pressure is set equal to
0.1 MPa. In both chambers the initial velocity is equal to 0.
The numerical solution of the 6-equation model is compared to the exact solution of the Euler equations. A mesh employ-
ing 1000 uniform cells is used in Fig. 8 and a mesh employing 100 cells is used in Fig. 9. Comparison with the exact solution is
shown in both ﬁgures at time t = 240 ls. Again this test poses no computational difﬁculty.
Fig. 7. Shock tube with Mie-Grüneisen type EOS. Magniﬁed view of pressure and density around the contact discontinuity. Results are in perfect agreement
with the exact solution and the solution is oscillation free.
Fig. 8. Liquid/gas shock tube. The present relaxation method is used to solve the 6-equation model. Numerical results are shown with symbols and
compared to the exact solution (solid). A 1000 cells mesh is used. The density ratio is 1000 and the pressure ratio is 10,000 at the initial discontinuity.
A second-order extension of the method with van Leer limiter is used. Results are in excellent agreement.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1695


In this test case and in all subsequent tests, strong pressure waves propagate. Relaxation terms present in the volume
fraction and energy equations become important, as well as the pressure relaxation step. Robustness and convergence of
the algorithm in the unsteady building of the solution are improved by pressure relaxation.
4.3.2. Water–air shock tube in extreme conditions
The same shock tube problem is solved, but initially, the left chamber pressure is set to 1 TPa (10 Mbars) and the density
of air is set to 10 kg m3. The exact solutions of the single phase Euler equations and the multiphase ﬂow model with 6 equa-
tions are compared in Fig. 10 at time t = 8.3 ls. This test illustrates the robustness and convergence of the algorithm.
4.4. Inﬂuence of ^pI in the relaxation method
During the relaxation step, we have highlighted different possible estimates for the pressure average ^pI. In order to dem-
onstrate the weak inﬂuence of the estimate, the liquid/gas shock tube test presented in Fig. 8 is examined with different esti-
mates of ^pI. In Fig. 11, results are presented and compared with two possible estimates: ^pI ¼ p0
I or ^pI ¼ p. No differences are
visible.
4.5. Cavitation test
A 1 m length tube is ﬁlled with liquid water at atmospheric pressure and with density q = 1000 kg m3. A small volume
fraction of air (aair = 102) is initially present everywhere. An initial velocity discontinuity is located at x = 0.5 m. On the left,
the velocity is set to u = 100 m/s and on the right, u = 100 m/s. Solution is shown in Fig. 12 at time t = 1.85 ms, using 1000
uniform mesh cells.
Strong rarefaction waves propagate in the tube and the liquid pressure decreases. As gas is present, the pressure cannot
become negative. To maintain positive pressure, the gas volume fraction increases and creates a cavitation pocket. This re-
sults in the dynamic appearance of two interfaces that were not present initially. Excellent agreement with the exact solu-
tion of the 5-equation model [37] is obtained. Interface creation is readily handled by the present algorithm.
Fig. 9. Same liquid/gas shock tube as those of Fig. 8 with 100 cells. Numerical results are shown with symbols and compared to the exact solution (solid).
A second-order extension of the method with van Leer limiter is used. Results are in good agreement.
1696
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


4.6. Multidimensional validation: shock–bubble interaction
Multidimensional ﬁnite volume extension of the method is presented in Appendix D. The method is validated against
shock tube experiments of shock–bubble interaction. The experiment is one of those proposed in Layes and Le Métayer
[27] where full description of the experimental setup is provided. The conﬁguration under study consists in a shock wave
propagating at Mach number 1.5 into air at atmospheric conditions and interacting with a helium bubble. The initial density
of air is qair = 1.29 kg m3 and the initial density of helium is qhelium = 0.167 kg m3. In the simulation both ﬂuids are con-
sidered as ideal gases with polytropic coefﬁcients cair = 1.4 and chelium = 1.67.
Fig. 10. Liquid/gas shock tube. The present relaxation method is used to solve the 6-equation model. Numerical results are shown with symbols and are
compared to the exact solution (solid). A 1000 cells mesh is used. The initial density ratio is 100 and the initial pressure ratio is 107. This test illustrates
robustness and convergence of the algorithm.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1697


The initial conﬁguration is represented in Fig. 13. Computed results are compared with experimental ones in Fig. 14.
4.7. Cavitating Richtmyer–Meshkov instability (RMI)
To illustrate the method capabilities a 2D test involving a RMI with a liquid–gas interface is considered. As the liquid is
not pure, new interfaces appear during the development of the instability, due to cavitation effects. The shape of the result-
ing interface and the entire ﬂow ﬁeld show a non-conventional behavior, that was never computed before, as the model and
method must deal with liquid gas interfaces and dynamic appearance of gas pockets in severe conditions.
Fig. 11. Comparison of two different pressure averages estimates. The test case of Fig. 8 is computed with ^pI ¼ p0
I on the left the and ^pI ¼ p on the right.
From top to bottom, pressure, velocity and mixture density remains unchanged.
1698
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


The left part of the computational domain is ﬁlled with nearly pure water and the right part with nearly pure gas. They are
initially separated by a curved interface. It is a portion of circle with 0.6 m radius centered at x = 1.2 m, y = 0.5 m. The phys-
ical domain is 3 m long and 1 m high. The mesh contains 900 cells along x-direction and 400 cells along y-direction. Both
water and gas have an initial velocity of 200 m/s. Top, bottom and left boundaries are treated as solid walls. The initial den-
sity of water is qwater = 1000 kg m3 and the stiffened gas EOS parameters are cwater = 4.4 and p1,water = 6  108 Pa. The initial
density of gas is qgas = 100 kg m3 and EOS parameters are cgas = 1.8 and p1,gas = 0 Pa. The left chamber contains a very small
volume fraction of gas agas = 106 and the right chamber contains a very small volume fraction of water awater = 106. The
initial conﬁguration is represented in Fig. 15. Results are shown in Fig. 16.
When the ﬂow impacts the left wall, a right-facing shock propagates in the domain through the water/gas discontinuity. A
conventional RMI appears ﬁrst. Then expansion waves are produced as the jet elongates. It results in expanded zones near
the solid boundary where gas in-homogeneities grow, producing dynamic appearance of gas pockets. As the pressure is very
low in these zones, the jet dynamics is modiﬁed compared to conventional RMI with pure ﬂuids. The various gas pockets
Fig. 12. Expansion tube with cavitation pocket appearance. The present relaxation method is used to solve the 6-equation model. Numerical results are
shown with symbols and are compared to the exact solution (solid) of the 5-equation model [37]. A 1000 cells mesh is used.
Fig. 13. Initial conﬁguration of the shock–bubble interaction test.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1699


near the solid boundary and in the jet core are clearly visible in Fig. 17 where the gas volume fraction is shown. Relaxation
terms present in the volume fraction and energy equations are responsible for the dynamic appearance of these gas pockets.
The link between the 6-equation model and conventional barotropic cavitating ﬂow models that are the most popular in
cavitation modeling is detailed in Appendix A. These models are composed of one or two mass conservation equations and
one momentum equation. They consist in hyperbolic systems of conservation laws. These models involve an important dif-
ﬁculty related to the non-monotonic behavior of the sound speed versus volume fraction [26,11,48]. It is thus interesting to
examine how the various ingredients developed in the context of the 6-equation model can be used for these barotropic
models in order to solve this difﬁculty.
5. Method extension for shocks in multiphase mixtures – artiﬁcial heat exchanges
The present reﬁnement of the algorithm is needed only when shock propagation in real multiphase mixtures is under
study. For other situations with interfaces separating pure ﬂuids or cavitating ﬂows, there is no need to account for the arti-
ﬁcial heat exchanges detailed hereafter. The artiﬁcial heat exchange is used to correct the partition of the energies among the
various phases in the mixture and to propagate shocks in these mixtures at the correct speed with the correct shocked state.
Fig. 15. Initial conﬁguration of the water–gas Richtmyer–Meshkov instability. Both liquid and gas have initial velocity of 200 m/s.
Fig. 14. Shock–bubble interaction test. Experimental results (left) and computed results (right) are compared at different times. Because of the difference in
gas properties, the transmitted shock wave is faster than the incident one in air. Pressure and density gradients create vorticity.
1700
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


Some preliminary observations of numerical schemes in the context of single phase ﬂows are ﬁrst necessary to introduce
the numerical issues associated with multiphase shocks.
Consider a shock wave propagating in a pure material, governed by the Euler equations. Shock capturing schemes pro-
duce a smearing of discontinuities and it is interesting to compare the thermodynamic path followed by the ﬂuid in the
shock layer to the theoretical Hugoniot curve. Such comparison is shown in Fig. 18.
Fig. 16. Water–gas Richtmyer–Meshkov instability. Mixture density contours are shown at time t0 = 0 ms, t1 = 2 ms, t2 = 3.1 ms, t3 = 4.8 ms, t4 = 6.4 ms,
t5 = 8.6 ms. The mesh contains 900  400 cells. New interfaces appear dynamically near the solid boundary as a result of expansion waves focusing. They
result in cavitation pockets that considerably modify the jet and spike shape.
Fig. 17. Water–gas Richtmyer–Meshkov instability. Volume fraction contours of gas are shown at time t5 = 8.6 ms. The gas volume fraction increase into the
liquid jet and near the solid wall boundary. The spike shape is also modiﬁed.
Fig. 18. Comparison of the numerical Hugoniot curve (symbols) and the theoretical one (lines) in the numerical diffusion zone for single phase ﬂows. The
two thermodynamic paths are different but the end states are the same.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1701


It appears clearly that the thermodynamic paths are very different. This is due to the succession of numerical weak shocks
that propagate into the cell that do not impose the same thermodynamic transformation as a single strong shock [10]. The
successive cell averages produce also transformations in disagreement with the single shock Hugoniot.
However, this numerical phenomenon has no consequence on the computation of the shocked state for single phase
ﬂows. As shown in Fig. 18, the end of the shock layer merges with the theoretical Hugoniot state. This is a consequence
of conservation properties of the Euler equations.
When dealing with multiphase mixtures, the same deviation from the theoretical Hugoniot appears and has more serious
consequences. The reason is that for each weak shock that enter the cell, the equation of state changes. Indeed, for multi-
phase mixtures, there is an extra degree of freedom characterized by the volume fraction. At a given point of the numerical
Fig. 19. Epoxy–spinel shock tube problem with moderate pressure ratio. The present relaxation method is used to solve the 6-equation model. Numerical
results are shown with symbols and are compared to the exact solution of the 5-equation model (solid). A 500 cells mesh is used. The pressure ratio is
100,000 at the initial discontinuity.
1702
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


shock, as shown in Fig. 18, there is no hope that this point belongs to the theoretical mixture Hugoniot curve. It follows that
the corresponding volume fraction is in error. Consequently the mixture EOS (II.4) is in error too. These errors cumulate in
the shock layer and, contrary to that of single phase ﬂows, the end state does not belong to the mixture Hugoniot.
To illustrate these difﬁculties, consider the following test cases.
5.1. Epoxy–spinel mixture shock tube with moderate pressure ratio
A tube of 1 m length contains two chambers separated by an interface at the location x = 0.6 m. Both chambers of the tube
are ﬁlled with the same mixture of epoxy and spinel. The initial density of the epoxy is qepoxy = 1185 kg m3 and its stiffened
Fig. 20. Epoxy–spinel shock tube problem with extreme pressure ratio. The present relaxation method is used to solve the 6-equation model. Numerical
results are shown with symbols and are compared to the exact solution of the 5-equation model (solid). A 500 cells mesh is used. The pressure ratio is
2,000,000 at the initial discontinuity.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1703


gas EOS parameters are cepoxy = 2.43 and p1, epoxy = 5.3  109 Pa. The initial density of spinel is qspinel = 3622 kg m3 and EOS
parameters
are
cspinel = 1.62
and
p1,spinel = 141  109 Pa.
The
initial
volume
fraction
in
both
chambers
are
aepoxy ¼ 0:5954ðaspinel ¼ 1  aepoxyÞ. The pressure at the left of the interface is equal to 1  1010 Pa, while the right chamber
is at atmospheric pressure. All the materials are initially at rest. Using a 500 cell uniform mesh the solution of the multiphase
ﬂow model with 6 equations is compared to the exact solution of the 5-equation model [37] in Fig. 19 at time t = 80 ls. As
the shock is of moderate strength, the present method converges to the exact solution without any artiﬁcial heat exchange.
5.2. Epoxy–spinel mixture shock tube under extreme conditions
We consider now the same shock tube problem as previously, but the initial pressure ratio is set to 2  106. Results are
shown in Fig. 20.
Important differences appear between solutions as the shock is now very strong. The numerical solution does not con-
verge to the exact solution of the 5-equation model, equipped with the shock relations summarized in System (II.5). This
is due to the incorrect partition of internal energy in the shock layer [37]. In order to partition the energies correctly, artiﬁcial
heat exchanges are now introduced.
5.3. Artiﬁcial heat exchanges in the 6-equation model
The correct partition of the shock energy among the various phases can be achieved by shock tracking methods. Shock
tracking methods have been intensively studied by Glimm et al. [18], LeVeque and Shyue [30], Massoni et al. [31]. Another
option is to correct partition of the energies in the shock layer by introducing artiﬁcial heat transfers.
Artiﬁcial heat exchanges have been introduced in Petitpas et al. [37] in the context of a Lagrange-relaxation method. In
the present Eulerian formulation they correspond to an extra pressure that appears in the internal energy equations:
@a1q1e1
@t
þ @a1q1e1u
@x
þ ða1p1 þ qÞ @u
@x ¼ 0;
@a2q2e2
@t
þ @a2q2e2u
@x
þ ða2p2  qÞ @u
@x ¼ 0:
The artiﬁcial heat exchange term q @u
@x is active only in the shock layer as it is deﬁned by
q ¼ g @u
@x


mðvÞ;
where g @u
@x
 
¼
1 if @u
@x < 0
0 otherwise

and m(v) is the heat exchange function.
It is more convenient and also accurate (regarding mesh independence of the results) to rewrite these equations into the
form:
@a1q1e1
@t
þ @ a1q1e1 þ g @u
@x
 lðvÞ


u
@x
þ a1p1
@u
@x ¼ 0;
@a2q2e2
@t
þ @ a2q2e2  g @u
@x
 lðvÞ


u
@x
þ a2p2
@u
@x ¼ 0:
The function l(v) also expresses heat exchange and must be predetermined for a given two-phase mixture. A method for its
determination is given in [37]. An example of the effects of function l(v) is shown in the following example.
Fig. 21. Values of the approximate piecewise linear function of l (symbols) and ﬁtting curve ðlðvÞ ¼ exp½5:64107  v2 þ 5:34103  v þ 25:6Þ in the
speciﬁc volume range (2.65  104 	 4.61  104 m3/kg) corresponding to piston velocity range of 0 	 4200 m/s and pressure range of 1–880,000 atm.
1704
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


5.4. Epoxy–spinel mixture shock tube with artiﬁcial heat exchanges
The test problem of Fig. 20 is rerun with the artiﬁcial heat exchanges. The procedure developed in [37] is used to deter-
mine the heat exchange function. This function depends on:
– The initial state of the mixture in which the shock propagates.
– The numerical smearing of the shock front that is inherent in a given method.
Fig. 22. Epoxy–spinel shock tube problem. The present relaxation method is used to solve the 6-equation model. Numerical results are shown with symbols
and are compared to the exact solution of the 5-equation model (solid). A 500 cells mesh is used. The pressure ratio is 2,000,000 at the initial discontinuity.
Artiﬁcial heat exchanges are used in the shock layer only. Convergence of the results is obtained.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1705


For the present algorithm, the heat exchange function has been determined and is shown in Fig. 21.
Artiﬁcial heat exchanges are used in the shock layer only. With this correction, the algorithm converges to the exact solu-
tion, as shown in Fig. 22.
It is signiﬁcant to note that the heat exchange function of Fig. 21 provides converged results for any shock strength in the
pressure range of 1–880,000 atm. Moreover, mesh independence of the solution is guaranteed.
6. Conclusion
A relaxation hyperbolic model with 6 equations was built to solve interface problems, cavitating ﬂows and shocks into
mixtures. This model considerably simpliﬁes the numerical approximation of the 5-equation model of Kapila et al. [23]. A
simple, efﬁcient and robust algorithm has been derived to solve the relaxation model. The various ingredients used by this
method are general enough to consider future extensions to problems involving complex physics and large hyperbolic sys-
tems. In particular, solid–ﬂuid coupling will be examined with the present multiphase modeling of diffuse interfaces in the
context of the elastic model of Gavrilyuk et al. [17].
Appendix A. The link between the 6-equation model and conventional barotropic cavitating ﬂow models
Barotropic ﬂow models are very popular in cavitation modeling. They are composed of one or two mass conservation
equations and one momentum equation. They consist in hyperbolic systems of conservation laws. These models involve
an important difﬁculty related to the non-monotonic behavior of the sound speed versus volume fraction [26,11,48]. It is
thus interesting to examine how the various ingredients developed in the context of the 6-equation model can be used
for these barotropic models in order to solve this difﬁculty.
A well posed barotropic ﬂow model for cavitating ﬂows can be obtained by simplifying the 5-equation model of Kapila
et al. [23]. In cavitating ﬂows, shocks are assumed absent or weak, even if there is no evidence regarding this assumption.
A ﬁrst simpliﬁcation consists in replacing the volume fraction and energy equations by entropy equations:
@s1
@t þ u @s1
@x ¼ 0;
@s2
@t þ u @s2
@x ¼ 0;
@ðaqÞ1
@t
þ @ðaqÞ1u
@x
¼ 0;
@ðaqÞ2
@t
þ @ðaqÞ2u
@x
¼ 0;
@qu
@t þ @qu2 þ p
@x
¼ 0:
ðA:1Þ
This system is closed by the pressure equilibrium condition:
p1ðq1; s1Þ ¼ p2ðq2; s2Þ ¼ p:
ðA:2Þ
Solution of this equation gives the volume fraction, and consequently the pressure p.
An extra assumption is used in conventional barotropic cavitating ﬂow models. The entropies are assumed constant in the
entire domain and not only along ﬂuid’s trajectories. The two entropy equations thus reduce to
sk ¼ s0
k;
k ¼ 1; 2:
The barotropic ﬂow model thus reduces to three conservation equations:
@ðaqÞ1
@t
þ @ðaqÞ1u
@x
¼ 0;
@ðaqÞ2
@t
þ @ðaqÞ2u
@x
¼ 0;
@qu
@t þ @qu2 þ p
@x
¼ 0:
ðA:3Þ
To illustrate the thermodynamic closure of this model, let us assume that each phase obeys the stiffened gas EOS (II.3). The
isentropes become
pk þ p1k
qck
k
¼ p0 þ p1k
qck
0k
;
and correspond to the Tait EOS.
The isentropic stiffened gas EOS (or Tait EOS), can be derived for any pure liquid and any ideal gas. It is a function of the
phase density only
1706
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


pk ¼ pS
kðqkÞ ¼
qk
q0k

ck
ðp0 þ p1kÞ  p1k:
ðA:4Þ
System (A.3) is thus closed by the relation
pS
1ðq1Þ ¼ pS
2ðq2Þ:
ðA:5Þ
In other words, the mixture evolves in mechanical equilibrium with isentropic evolutions for each phase. This assumption is
valid provided that boundary layers, heat and mass transfer, and shock waves have negligible inﬂuence.
With the use of the isentropic stiffened gas EOS the equilibrium condition (A.5) reduces to a function of volume fraction
only:
fða1Þ ¼
ðaqÞ1
a1q01

c1
ðp0 þ p11Þ  p11 
ðaqÞ2
ð1  a1Þq02

c2
ðp0 þ p12Þ þ p12 ¼ 0:
ðA:6Þ
Its resolution gives a1, then q1 as well as the pressure with the help of one of the EOS (A.4).
This model assumes that cavitation does not result from mass transfer. Cavitation pockets appear as the volume fraction
increases for a small amount of gas present initially. Cavitation is thus modeled as a mechanical relaxation process, occurring
at inﬁnite rate, and not as a mass transfer process. This corresponds to a simpliﬁed limit situation compared to reality. It also
presents a deﬁciency when pure liquid is present. Heat and mass transfers have been introduced in the 5-equation model
[45] in order to deal with more realistic cavitating situations. Furthermore, the barotropic ﬂow model, in reduced form
(A.3) involves the same numerical difﬁculties as the 5-equation model. The sound speed for this model still obeys Wood’s
formulas, whose non-monotonic behavior was shown in Fig. 1.
To circumvent these difﬁculties, especially due to the non-monotonic behavior of the sound speed, we adapt the strategy
developed in the context of the 6-equation model to this simpliﬁed situation.
A.1. A relaxation model for the barotropic cavitating ﬂow model
The non-monotonic behavior of the sound speed that causes computational difﬁculties comes from the equilibrium con-
dition (A.5). Following the analysis of Section 2, a relaxation model can be built:
@a1
@t þ u @a1
@x ¼ lðpS
1ðq1Þ  pS
2ðq2ÞÞ;
@ðaqÞ1
@t
þ @ðaqÞ1u
@x
¼ 0;
@ðaqÞ2
@t
þ @ðaqÞ2u
@x
¼ 0;
@qu
@t þ @qu2 þ a1p1 þ a2p2
@x
¼ 0:
ðA:7Þ
As the model includes pressure non-equilibrium effects, the momentum equation involves pressures from both phases. This
model is the isentropic analogue of the 6-equation model. Unlike the preceding models, the present one has a monotonic
sound speed given by
c2
f ¼ Y1c2
1 þ Y2c2
2:
It is not difﬁcult to show that in the asymptotic limit l ? +1 this model corresponds to system (A.3) with thermodynamic
closure (A.5).
The numerical method to solve System (A.7) is a simpliﬁcation of the method developed in Section 3. It can be summa-
rized as follows:
– At each cell boundary solve the Riemann problem of System (A.7) without relaxation terms with favorite solver. The
HLLC solver of Section 3.1 is recommended.
– Evolve all ﬂow variables W = (a1,(aq)1, (aq)2,qu) with the Godunov type method of Section 3.2.
– Determine the relaxed pressure and especially the volume fraction by solving Eq. (A.6). The Newton method is appro-
priate for this task.
– Go to the ﬁrst item for the next time step.
Appendix B. Asymptotic limit of the 6-equation model in the presence of stiff pressure relaxation
To perform the asymptotic analysis it is assumed that each ﬂow variable f obeys the following asymptotic expansion:
f = fo + ef1 where fo represents the equilibrium state and f1 a small perturbation around this state. Inversely to the perturba-
tions, pressure relaxation coefﬁcient l ¼ l0
e is assumed stiff with e ? 0+.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1707


With this transformation, the equations that do not contain any relaxation parameter will be unchanged. The three equa-
tions to consider are thus the internal energy equations and the volume fraction equation. These are rewritten under follow-
ing form:
da1
dt ¼ lðp1  p2Þ;
a1q1
de1
dt þ a1p1
@u
@x ¼ pIlðp1  p2Þ;
a2q2
de2
dt þ a2p2
@u
@x ¼ pIlðp1  p2Þ;
where d
dt ¼ @
@t þ u @
@x represents the Lagrangian derivative.
Some transformations with appropriate variables are necessary before doing the asymptotic analysis. Consider the inter-
nal energy equation of phase 1. It can be written as a pressure evolution equation as e1 = e1(q1,p1):
a1q1
@e1
@q1

p1
dq1
dt þ @e1
@p1

q1
dp1
dt þ p1
q1
@u
@x
 
!
¼ pIlðp1  p2Þ:
With the help of the phase 1 mass equation,
da1q1
dt
þ a1q1
@u
@x ¼ 0;
that also reads,
dq1
dt ¼ q1
a1 lðp1  p2Þ  q1
@u
@x ;
we get
dp1
dt þ q1
p1
q2
1  @e1
@q1

p1
@e1
@p1

q1
@u
@x ¼ q1
a1
pI
q2
1  @e1
@q1

p1
@e1
@p1

q1
l p1  p2
ð
Þ:
With the help of sound speed deﬁnitions,
c2
1 ¼
p1
q2
1  @e1
@q1

p1
@e1
@p1

q1
;
c2
I1 ¼
pI
q2
1  @e1
@q1

p1
@e1
@p1

q1
:
The phase 1 pressure evolution equation is obtained:
dp1
dt þ q1c2
1
@u
@x ¼ q1c2
I1
a1 lðp1  p2Þ:
Regarding phase 2, a similar result is obtained:
dp2
dt þ q2c2
2
@u
@x ¼ q2c2
I2
a2 lðp1  p2Þ:
The asymptotic analysis is now carried out on the following system:
da1
dt ¼ lðp1  p2Þ;
dp1
dt þ q1c2
1
@u
@x ¼ q1c2
I1
a1 lðp1  p2Þ;
dp2
dt þ q2c2
2
@u
@x ¼ q2c2
I2
a2 lðp1  p2Þ:
By expanding each ﬂow variable as f = fo + ef1 we get
– At order 1
e:
p0
1 ¼ p0
2 ¼ p0:
1708
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


It implies on the one hand,
p0
I ¼ p0;
and on the other hand:
c02
I1 ¼ c02
1
and
c02
I2 ¼ c02
2 :
– At zero-order the two-pressure equations become
dp
0
dt þ q0
1c02
1
@u0
@x ¼ q0
1c02
1
a0
1
ðp1
1  p1
2Þ;
dp
0
dt þ q0
2c02
2
@u0
@x ¼ q0
2c02
2
a0
2
ðp1
1  p1
2Þ:
By making the difference of these two equations, the pressure ﬂuctuation difference is readily obtained:
p1
1  p1
2 ¼ q0
2c02
2  q0
1c02
1
q0
1c02
1
a0
1 þ q0
2c00
2
a0
2
@u0
@x :
The volume fraction equation thus becomes
da0
1
dt ¼ q0
2c02
2  q0
1c02
1
q0
1c02
1
a0
1 þ q0
2c00
2
a0
2
@u0
@x :
Consequently the 5-equation model with mechanical equilibrium is recovered as the asymptotic limit of the 6-equation
model in the presence of stiff pressure relaxation.
Appendix C. Extension to second-order
The ﬁrst-order numerical method for the hyperbolic step presented in Section 3 is extended to second-order. It consists in
solving the two-pressure 6-equation model (C.1) with a MUSCL type method:
@a1
@t þ u @a1
@x ¼ 0;
@a1q1
@t
þ @a1q1u
@x
¼ 0;
@a2q2
@t
þ @a2q2u
@x
¼ 0;
@qu
@t þ @qu2 þ ða1p1 þ a2p2Þ
@x
¼ 0;
@a1q1e1
@t
þ @a1q1e1u
@x
þ a1p1
@u
@x ¼ 0;
@a2q2e2
@t
þ @a2q2e2u
@x
þ a2p2
@u
@x ¼ 0:
ðC:1Þ
In the MUSCL method, the solution is assumed regular enough so that a primitive variable formulation can be used:
@a1
@t þ u @a1
@x ¼ 0;
@q1
@t þ u @q1
@x þ q1
@u
@x ¼ 0;
@q2
@t þ u @q2
@x þ q2
@u
@x ¼ 0;
@u
@t þ u @u
@x þ 1
q
@p
@x ¼ 0
or
@u
@t þ u @u
@x þ p1  p2
q
@a1
@x þ a1
q
@p1
@x þ 1  a1
ð
Þ
q
@p2
@x ¼ 0;
@p1
@t þ u @p1
@x þ q1c2
1
@u
@x ¼ 0;
@p2
@t þ u @p2
@x þ q2c2
2
@u
@x ¼ 0:
Under compact form, this system reads
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1709


@W
@t þ AðWÞ @W
@x ¼ 0:
ðC:2Þ
With W = (a1,q1,q2,u,p1,p2)t and AðWÞ ¼
u
0
0
0
0
0
0
u
0
q1
0
0
0
0
u
q2
0
0
p1p2
q
0
0
u
a1
q
1a1
q
0
0
0
q1c2
1
u
0
0
0
0
q2c2
2
0
u
0
B
B
B
B
B
B
@
1
C
C
C
C
C
C
A
.
Second-order extension consists in applying the following sequence of operations.
C.1. Gradients limitation
In a cell i, at instant tn, primitive variables Wn
i are known. Let us denote by D
i and Dþ
i the gradients vector respectively on
the left and right neighbors of cell i. They are deﬁned by
D
i ¼ Wn
i  Wn
i1
Dx
and
Dþ
i ¼ Wn
iþ1  Wn
i
Dx
:
A slope limiter function n is used to prevent local extrema. Minmod, van Leer or Superbee limiters can be used. The limited
slope is now Di ¼ nðD
i ; Dþ
i Þ.
C.2. Variables extrapolation
Within a given cell extrapolated primitive variable vectors Wi,L and Wi,R corresponding to the left and right boundary of
cell i respectively are computed,
Wn
i;L ¼ Wn
i  Dx
2 Di
and
Wn
i;R ¼ Wn
i þ Dx
2 Di:
These variables are evolved during a half time step by
Wnþ1=2
i;L;R
¼ Wn
i;L;R þ 1
2
Dt
Dx AðWn
i Þ Wn
i;L  Wn
i;R
h
i
:
C.3. Riemann problem resolutions
The Riemann problem is now computed at each cell boundary i ± 1/2 allowing ﬂux vectors F
i1=2 computation for conser-
vative variables:
F
i1=2 ¼ F
i1=2 Wnþ1=2
i1;R ; Wnþ1=2
i;L


and
F
iþ1=2 ¼ F
iþ1=2 Wnþ1=2
i;R
; Wnþ1=2
iþ1;L


:
ðC:3Þ
It also provides the cell boundaries non-conservative variables:
a
k;i1=2; u
i1=2
and
ðaqeÞ
k;i1=2:
ðC:4Þ
C.4. Evolution step
Once the inter-cells ﬂuxes and non-conservative variables are determined, the solution is evolved on the entire time step:
Unþ1
i
¼ Un
i  Dt
Dx ðF
iþ1=2  F
i1=2Þ;
anþ1
1i
¼ a1in  Dt
Dx ððua1Þ
iþ1=2  ðua1Þ
i1=2  a1i
nðu
iþ1=2  u
i1=2ÞÞ;
ðaqeÞki
nþ1 ¼ ðaqeÞn
ki  Dt
Dx ððaqeuÞ
kiþ1=2  ðaqeuÞ
ki1=2 þ ðapÞn
kiðu
iþ1=2  u
i1=2ÞÞ;
where the ‘‘*” variables are given by (C.3) and (C.4).
Appendix D. Extension to multi-dimensions
The method is extended to multi-dimensions by a ﬁnite volume method able to deal with structured and unstructured
meshes. Thus, let us consider a control volume Vi delimited by surface A of normal unit vector ~n. The conservative part of
system (III.1) under integral form reads
@
@t
Z
Vi
U þ
Z
A
H  ~ndA ¼ 0
ðD:1Þ
with U = ((aq)1,(aq)2,qu,qv,qE)T the conservative variable vector,
1710
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712


H = (F,G) the tensor of ﬂuxes where:
F ¼ ððaqÞ1u; ðaqÞ1u;qu2 þ p; uv; ðqE þ pÞuÞT;
G ¼ ððaqÞ1v; ðaqÞ1v;quv;qv2 þ p; ðqE þ pÞvÞT;
and E ¼ Y1e1 þ Y2e2 þ 1
2 u2 þ 1
2v2 and p = a1p1 + a2p2.
Boundary A of Vi is the union of N straight segments [AsAs+1], where AN+1 = A1.
The ﬁrst term of Eq. (D.1) is interpreted as the time-rate of change of the conservative variable vector volume average:
@
@t
Z
Vi
U ¼ Vi
@U
@t :
As the normal unit vector is expressed by ~nS ¼ ðcos hs; sin hsÞ, the second term of (D.1) becomes
Z
A
H  ~ndA ¼
X
N
s¼1
Z Asþ1
As
H  ~nSdA ¼
X
N
s¼1
Z Asþ1
As
ðF 
 cos hs þ G 
 sin hsÞdA:
Assuming that the ﬂuxes are constant along each segment, it becomes
Z
A
H  ~ndA ¼
X
N
s¼1
LsðFs 
 cos hs þ Gs 
 sin hsÞ;
where Ls is the length of segment [AsAs+1].
After time integration, the evolution of the conservative part of system (III.1) is given for cell i by the scheme:
Unþ1
i
¼ Un
i  Dt
Vi
X
N
s¼1
LsðF
s 
 cos hs þ G
s 
 sin hsÞ;
where F
s and G
s represent the ﬂuxes solution of the Riemann problem between states L and R separated by the segment
[AsAs+1] with respect to normal ~n.
The scheme for the non-conservative volume fraction equation becomes
anþ1
k;i
¼ an
k;i  Dt
Vi
X
N
s¼1
Ls½ðuakÞ
s cos hs þ ðvakÞ
s sin hs  an
k;iðu
s cos hs þ v
s sin hsÞ;
and for the non-conservative energy equations it is
ðaqeÞnþ1
k;i
¼ ðaqeÞn
k;i  Dt
Vi
X
N
s¼1
Ls½ððaqeÞkuÞ
s cos hs þ ððaqeÞkvÞ
s sin hs þ ðapÞn
k;iðu
s cos hs þ v
s sin hsÞ:
References
[1] R. Abgrall, How to prevent pressure oscillations in multicomponent ﬂow calculations: a quasi conservative approach, Journal of Computational Physics
125 (1996) 150–160.
[2] R. Abgrall, S. Karni, Computations of compressible multiﬂuids, Journal of Computational Physics 169 (2) (2001) 594–623.
[3] R. Abgrall, V. Perrier, Asymptotic expansion of a multiscale numerical scheme for compressible multiphase ﬂows, SIAM Journal of Multiscale and
Modeling and Simulation 5 (2006) 84–115.
[4] R. Abgrall, R. Saurel, Discrete equations for physical and numerical compressible multiphase mixtures, Journal of Computational Physics 186 (2) (2003)
361–396.
[5] M.R. Baer, J.W. Nunziato, A two-phase mixture theory for the deﬂagration-to-detonation transition (DDT) in reactive granular materials, International
Journal of Multiphase Flow 12 (6) (1986) 861.
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
1711


[6] J.B. Bdzil, R. Menikoff, S.F. Son, A.K. Kapila, D.S. Steward, Two-phase modeling of a deﬂagration-to detonation transition in granular materials: A critical
exmination of modeling issues, Physics of Fluids 11 (2) (1999) 378–402.
[7] D.J. Benson, Computational methods in Lagrangian and Eulerian hydrocodes, Computer Methods in Applied Mechanics and Engineering 99 (1992) 235–
394.
[8] A. Chinnayya, E. Daniel, R. Saurel, Computation of detonation waves in heterogeneous energetic materials, Journal of Computational Physics 196 (2004)
490–538.
[9] G. Cochran, J. Chan, Shock initiation and detonation models in one and two dimensions, CID-18024 Lawrence National Laboratory Report 2 (1979) 1–2.
[10] R. Courant, K. Friedrichs, Supersonic Flow and Shock Waves, Springer, 1948.
[11] O. Coutier-Delgosha, R. Fortes-Patella, J.L. Reboud, N. Hakimi, C. Hirsch, Stability of preconditioned Navier-Stockes equations associated with a
cavitation model, Computers and Fluids 34 (2005) 319–349.
[12] S.F. Davis, Simpliﬁed second order Godunov type methods, SIAM Journal of Scientiﬁc and Statistical Computing 9 (1988) 445–473.
[13] A. Dervieux, F. Thomasset, A ﬁnite element method for the simulation of Rayleigh-Taylor instability, in: Approximation methods for Navier–Stokes
problems, Proceedings of the Symposium, Paderborn, West Germany, September 9–15, 1979, Springer-Verlag, Berlin, 1980, pp. 145–158.
[14] C. Farhat, F.X. Roux, A method for ﬁnite element tearing and interconnecting and its parallel solution algorithm, International Journal for Numerical
Methods in Engineering 32 (1991) 1205–1227.
[15] N. Favrie, S. Gavrilyuk, R. Saurel, Diffuse solid–ﬂuid interface model in cases of extreme deformations, Journal of Computational Physics, submitted for
publication.
[16] R. Fedkiw, T. Aslam, B. Merriman, S. Osher, A non-oscillatory Eulerian approach to interfaces in multimaterial ﬂows (the ghost ﬂuid method), Journal of
Computational Physics 152 (2) (1999) 457–492.
[17] S. Gavrilyuk, N. Favrie, R. Saurel, Modelling wave dynamics of compressible elastic materials, Journal of Computational Physics 227 (5) (2008) 2941–
2969.
[18] J. Glimm, J.W. Grove, X.L. Li, K.M. Shyue, Q. Zhang, Y. Zeng, Three dimensional front tracking, SIAM Journal of Scientiﬁc Computing 19 (1998) 703–727.
[19] D. Gueyfﬁer, L. Li, A. Nadim, R. Scardovelli, S. Zaleski, Volume-of-ﬂuid interface tracking with smoothed surface stress methods for three-dimensional
ﬂows, Journal of Computational Physics 152 (1999) 423–456.
[20] C.W. Hirt, B.D. Nichols, Volume of ﬂuid (VOF) method for the dynamics of free boundaries, Journal of Computational Physics 39 (1981) 201–255.
[21] C.W. Hirt, A.A. Amsden, J.L. Cook, An arbitrary Lagrangian Eulerian computing method for all ﬂow speeds, Journal of Computational Physics 135 (1974)
203–216.
[22] T.Y. Hou, P. Le Floch, Why non-conservative schemes converge to the wrong solution: error analysis, Mathematics of Computation 62 (1994) 497–530.
[23] A.K. Kapila, R. Menikoff, J.B. Bdzil, S.F. Son, D.S. Stewart, Two-phase modeling of deﬂagration-to-detonation transition in granular materials: reduced
equations, Physics of Fluids 13 (10) (2001) 3002–3024.
[24] S. Karni, Multicomponent ﬂow calculations by a consistent primitive algorithm, Journal of Computational Physics 112 (1994) 31–43.
[25] B.C. Khoo, T.G. Liu, C.W. Wang, The ghost ﬂuid method for compressible gas-water simulations, Journal of Computational Physics 204 (2005) 193.
[26] B. Koren, M.R. Lewis, E.H. van Brummelen, B. van Leer, Riemann-problem and level-set approaches for homentropic two-ﬂuid computations, Journal of
Computational Physics 181 (2002) 654–674.
[27] G. Layes, O. Le Métayer, Quantitative numerical and experimental studies of the shock accelerated heterogeneous bubbles motion, Physics of Fluids 19
(2007) 042105.
[28] E.L. Lee, H.C. Horning, J.W. Kury, Adiabatic Expansion of High Explosives Detonation Products, Lawrence Radiation Laboratory, University of California,
Livermore, TID 4500-UCRL 50422, 1968.
[29] O. Le Metayer, J. Massoni, R. Saurel, Modeling evaporation fronts with reactive Riemann solvers, Journal of Computational Physics 205 (2005) 567–610.
[30] R.J. LeVeque, Keh-Ming Shyue, Two-dimensional front tracking based on high resolution wave propagation methods, Journal of Computational Physics
123 (2) (1996) 354–368.
[31] J. Massoni, R. Saurel, G. Baudin, G. Demol, A mechanistic model for shock initiation of solid explosives, Physics of Fluids 11 (3) (1999) 710–736.
[32] G.H. Miller, E.G. Puckett, A high-order Godunov method for multiple condensed phases, Journal of Computational Physics 128 (1) (1996) 134–164.
[33] G.H. Miller, P. Colella, A high-order Eulerian Godunov method for elastic–plastic ﬂows in solids, Journal of Computational Physics 167 (2001) 131–176.
[34] W. Mulder, S. Osher, J.A. Sethian, Computing interface motion: the compressible Rayleigh-Taylor and Kelvin-Helmholtz instabilities, Journal of
Computational Physics 100 (1992) 209.
[35] A. Murrone, H. Guillard, A ﬁve equation reduced model for compressible two phase ﬂow problems, Journal of Computational Physics 202 (2) (2005)
664–698.
[36] S. Osher, R. Fedkiw, Level set methods: an overview and some recent results, Journal of Computational Physics 169 (2001) 463–502.
[37] F. Petitpas, E. Franquet, R. Saurel, O. Le Metayer, A relaxation-projection method for compressible ﬂows. Part II. The artiﬁcial heat exchange for
multiphase shocks, Journal of Computational Physics 225 (2) (2007) 2214–2248.
[38] F. Petitpas, R. Saurel, E. Franquet, A. Chinnayya, A., Modelling detonation waves in condensed materials: multiphase CJ conditions and
multidimensional computations, Shock waves, submitted for publication.
[39] G. Perigaud, R. Saurel, A compressible ﬂow model with capillary effects, Journal of Computational Physics 209 (2005) 139–178.
[40] R. Saurel, R. Abgrall, A multiphase Godunov method for multiﬂuid and multiphase ﬂows, Journal of Computational Physics 150 (1999) 425–467.
[41] R. Saurel, O. Le Metayer, A multiphase model for interfaces, shocks, detonation waves and cavitation, Journal of Fluid Mechanics 431 (2001) 239–271.
[42] R. Saurel, S. Gavrilyuk, F. Renaud, A multiphase model with internal degrees of freedom: application to shock-bubble interaction, Journal of Fluid
Mechanics 495 (2003) 283–321.
[43] R. Saurel, E. Franquet, E. Daniel, O. Le Metayer, A relaxation-projection method for compressible ﬂows. Part I. The numerical equation of state for the
Euler equations, Journal of Computational Physics 223 (2) (2007) 822–845.
[44] R. Saurel, O. Le Metayer, J. Massoni, S. Gavrilyuk, Shock jump relations for multiphase mixtures with stiff mechanical relaxation, Shock Waves 16 (3)
(2007) 209–232.
[45] R. Saurel, F. Petitpas, R. Abgrall, Modeling phase transition in metastable liquids. Application to cavitating and ﬂashing ﬂows, Journal of Fluid
Mechanics 607 (2008) 313–350.
[46] D.R. Scheffer, J.A. Zukas, Practical aspects of numerical simulation of dynamic events: material interfaces, International Journal of Impact Engineering
24 (5–6) (2000) 821–842.
[47] J. Sethian, Evolution, implementation and application of level set and fast marching methods for advancing fronts, Journal of Computational Physics
169 (2001) 503–555.
[48] E. Sinibaldi, Implicit preconditioned numerical schemes for the simulation of three-dimensional barotropic ﬂows, PhD Thesis, Scuola Normale
Superiore di Pisa, Italy, 2006.
[49] H.B. Stewart, B. Wendroff, Two-phase ﬂow: models and methods, Journal of Computational Physics 56 (3) (1984) 363–409.
[50] V.A. Titarev, E. Romenski, E.F. Toro, MUSTA type upwind ﬂuxes for nonlinear elasticity, International Journal of Numerical Methods in Engineering 73
(7) (2007) 897–926.
[51] E.F. Toro, M. Spruce, W. Speares, Restoration of the contact surface in the HLL Riemann solver, Shock Waves 4 (1994) 25–34.
[52] E.F. Toro, Riemann Solvers and Numerical Methods for Fluids Dynamics, Springer-Verlag, Berlin, 1997.
[53] E.H. van Brummelen, B. Koren, A pressure-invariant conservative Godunov-type method for barotropic two-ﬂuid ﬂows, Journal of Computational
Physics 185 (2003) 289–308.
[54] A.B. Wood, A Textbook of Sound, G. Bell and Sons Ltd., London, 1930.
1712
R. Saurel et al. / Journal of Computational Physics 228 (2009) 1678–1712
