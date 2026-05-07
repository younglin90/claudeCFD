Accepted version
Int J Numer Meth Fluids (2022) DOI: 10.1002/ﬂd.5087
The ﬁnal publication is available at onlinelibrary.wiley.com
Received: Added at production
Revised: Added at production
Accepted: Added at production
DOI: xxx/xxxx
RESEARCH ARTICLE: ACCEPTED VERSION
A pressure-based method for weakly compressible two-phase ﬂows
under a Baer-Nunziato type model with generic equations of state
and pressure and velocity disequilibrium
Barbara Re*1 | Rémi Abgrall2
1Department of Aerospace Science and
Technology, Politecnico di Milano, Italy
2Institute of Mathematics, University of
Zürich, Switzerland
Correspondence
*Email: barbara.re@polimi.it
Summary
Within the framework of diﬀuse interface methods, we derive a pressure-based Baer-
Nunziato type model well-suited to weakly compressible multiphase ﬂows. The
model can easily deal with diﬀerent equation of states and it includes relaxation
terms characterized by user-deﬁned ﬁnite parameters, which drive the pressure and
velocity of each phase toward the equilibrium. There is no clear notion of speed
of sound, and thus, most of the classical low Mach approximation cannot easily
be cast in this context. The proposed solution strategy consists of two operators: a
semi-implicit ﬁnite-volume solver for the hyperbolic part and an ODE integrator for
the relaxation processes. Being the acoustic terms in the hyperbolic part integrated
implicitly, the stability condition on the time step is lessened. The discretization
of non-conservative terms involving the gradient of the volume fraction fulﬁlls by
construction the non-disturbance condition on pressure and velocity to avoid oscilla-
tions across the multimaterial interfaces. The developed simulation tool is validated
through one-dimensional simulations of shock-tube and Riemann-problems, involv-
ing water-aluminum and water-air mixtures, vapor-liquid mixture of water and of
carbon dioxide, and almost pure ﬂows. The numerical results match analytical and
reference ones, except some expected discrepancies across shocks, which however
remain acceptable (errors within some percentage points). All tests were performed
with acoustic CFL numbers greater than one, and no stability issues arose, even
for CFL greater than 10. The eﬀects of diﬀerent values of relaxation parameters
and of diﬀerent amount equations of state—stiﬀened gas and Peng-Robinson—were
investigated.
KEYWORDS:
Baer-Nunziato type model, pressure formulation, compressible two-phase ﬂows, pressure and velocity
relaxation with ﬁnite parameters, semi-implicit ﬁnite-volume scheme, Peng-Robinson equation of state
arXiv:2107.12408v2  [physics.flu-dyn]  23 May 2022


2
B. Re and R. Abgrall
1
INTRODUCTION
Compressible multiphase ﬂows may manifest themselves in a variety of conﬁgurations, ranging from dispersed ﬂows (e.g.,
bubble or spray ﬂows) to interface problems involving two nearly pure ﬂuids (e.g., liquid accumulation or sloshing of a liquid
in a tank). From a numerical point of view, a distinguishing and challenging feature of such ﬂows is the presence of dynamic
interfaces that separate immiscible ﬂuids with diﬀerent physical or chemical properties. The several ways this challenge can be
answered has led to the development of diﬀerent multi-phase simulation strategies. The ﬁrst one that can come to mind is the
explicit tracking of the interface, either by deforming the grid to preserve interfaces as resolved surfaces, e.g. in1,2, or by tracking
their motion indirectly by means of Lagrangian markers, e.g. in3,4. These methods can be very accurate in well-resolved interface
problems with limited deformations, but cannot easily handle signiﬁcant interface distortions or topological modiﬁcations. A
diﬀerent strategy is pursued by interface capturing methods, which reconstruct the interfaces from the solution according to an
indicator function. Popular instances in this class are the level-set methods (e.g., see the reviews5 and6), in which an interface
is described by a zero-level curve of a continuous function expressing the (signed) distance from the interface, and the jump
conditions can be transferred across the interface by the ghost ﬂuid method7. This strategy facilitates the tracking of complex
interfaces, but it may prevent mass conservation and robustness8.
In this work, we focus on diﬀuse-interface methods (DIMs)9, which are another class of interface capturing methods, initiated
by the volume of ﬂuids method of Hirt and Nichols10 for incompressible ﬂows, and extended to compressible ﬂows by Saurel and
Abgrall11, and Kapila et al.12. DIMs rely on an augmented system of governing equations that speciﬁcally model the behavior of
the continuum close to the interfaces, while they aim to recover the pure ﬂuid behavior far from them. In practice, DIMs assume
that at least a small quantity of all ﬂuids coexist in each computational cell, and, rather than local instantaneous realizations of the
multiphase ﬂows, they aim to describe its behavior on average (in time, space, an ensemble, or in some combination of those)13,
which is usually the quantity of interest in industrial applications. Finally, DIMs appear particularly suited for ﬂuids governed
by diﬀerent equations of state (EOSs), since the behavior of each ﬂuid is described through its own thermodynamic model11.
Baer-Nunziato model
The cornerstone of the DIM class is the Baer-Nunziato (BN) two-phase model14, which was originally developed for reactive
granular materials and allows unequal phase pressures, velocities, and internal energies. The BN model consists of a set of mass,
momentum, and total energy equations for each phase and a topological equation for the volume fraction, so seven equations for a
one-dimensional problem. Starting from the original one, a wide set of BN-type models have been proposed11,15–19, according to
diﬀerent modeling and closure assumptions. While using diﬀerent deﬁnitions for the interface and relaxation terms, these models
typically share the same homogeneous and hyperbolic part. Thus, they require to face similar analytical and computational
challenges, which concern the presence of non-conservative terms, the large number of waves and the requirement to deal with
many equations. To mitigate the last two shortcomings, reduced models have been also proposed.
Five-equations models have been derived by means of asymptotic expansions of the BN model in the limit of stiﬀmechani-
cal (i.e., pressure and velocity) equilibrium12,20–23. Although these models are simpliﬁed, they have diﬀerent diﬃculties, as for
instance, the discretization of a non-conservative term involving the divergence of the velocity in the transport equation and the
non-monotonic behavior of the mixture sound speed with volume fraction, which may lead to an erroneous wave propagation
speed through the diﬀuse interface24. The roots of these issues are found in the pressure-equilibrium condition, which can be
thus removed, as in the pressure non-equilibrium 6-equation models24,25, which however need to be augmented by an energy
conservation law for the mixture to correct the predicted thermodynamic states, unless they use the formulation recently pro-
posed by Pelanti and Shyue for simpliﬁed EOSs (i.e., stiﬀened gas)26,27. A diﬀerent choice underlies the six-equation two-ﬂuid
models28,29, in which the ﬂuids have same pressure, but other thermodynamic quantities are in non-equilibrium. These models
are generally considered as ill-posed30, but recently Hantke and co-authors31 have proposed some constraints on the interfacial
pressure that can ensure hyperbolicity.
Even from this short and basic outline about two-phase models, it appears evident that each model has its own strengths and
weaknesses, and which is the best one clearly depends on the application under investigation. However, from a general BN-
type model, a hierarchy of hyperbolic multiphase models can be derived on the basis of asymptotic analysis32, so it is possible
to derive the simplest model involving the relevant physical eﬀects. Keeping into account these considerations, in this work,
we propose a full non-equilibrium, BN-type model, to provide the widest applicability within the class of DIMs, and eventual
reduced models will be considered in future works. Nevertheless, the selected BN-type model includes terms for pressure and


B. Re and R. Abgrall
3
velocity relaxation determined by ﬁnite parameters, which could be tuned to manage how the mechanical equilibrium between
phases is reached.
Pressure formulation
Most of the literature about DIMs for BN-type models solve the governing equations for the conservative variables, that is
volume fraction, density, momentum, and total energy, and contribute to the development and improvement of the so-called
density-based methods. These are the solvers of choice for ﬂows characterized by signiﬁcant compressibility, but they suﬀer
from ill-conditioning and accuracy problems at low Mach number33, that is when the ﬂow speed is considerably lower than the
speed of sound. In these conditions, the stability constraint on the time step becomes stringent and sophisticated preconditioning
techniques are required to recover the correct scaling of the pressure ﬂuctuation with the Mach number33,34. Because of diﬀerent
thermo-physical properties, two-phase ﬂow ﬁelds, especially when involving gas and liquid mixtures, often exhibit a wide range
of Mach numbers, including also the low Mach limit. A classical way to take into account the stiﬀness due to low Mach eﬀects
is the dimensionless scaling of the system of partial diﬀerential equations according to a reference density, a reference speed
and a reference speed of sound33,35. This leads to a system that looks similar to the original one, but that is able to describe
incompressible ﬂows. However, this approach is diﬃcult to apply to non-equilibrium multi-phase models, because it is not
possible to deﬁne a unique, unambiguous reference speed of sound. In addition, in density-based method, the pressure ﬁeld is
generally updated by means of an EOS, an operation that, in compressible multi-phase ﬂows, may generate spurious oscillations
at material interfaces36,37.
On the other hand, using the pressure rather than the density as a solution variable in the governing equations could circum-
vent most of the issues arising from the weak pressure-density coupling at low Mach numbers, because pressure variations are
signiﬁcant at all speeds. Thus, a unique reference pressure can be easily identiﬁed for the non-dimensional scaling of the govern-
ing equations, as in38. Moreover, solving for pressure (a primitive base) rather than total energy (a conservative variable) could
facilitate the achievement of mechanical equilibrium across interfaces and regions with varying thermo-physical properties36,
and paves the wave for a straightforward implementation of arbitrary EOSs39. These features could be substantially beneﬁcial
for the simulation of compressible multiphase ﬂows and thus have prompted us to study a pressure-based BN-type model.
Pressure-based methods have their roots in numerical methods for incompressible single-phase ﬂows, which have been
extended to compressible ﬂows following the general idea to replace the divergence-free condition on the velocity ﬁeld of
standard incompressible solvers by a modiﬁed continuity equation40,41. This concept has been extensively applied to the semi-
implicit method for pressure linked equations (SIMPLE)42, to projection or fraction-step methods43, and to the MAC method44,
leading to a large variety of pressure-based formulations, e.g.38,45–53. Although the research area of pressure-based formulation
has been very active for decades, most of the available techniques consider single-phase ﬂows and, but for a few excep-
tions39,54–57, they are valid only under the assumption of polytropic ideal gas. Recently, some examples of pressure-based
methods have been proposed in the framework of volume of ﬂuid methods, e.g.,58 and59, while Zhang et al.60 have developed
a pressure-based solver for the two-ﬂuid six-equation model, and Abgrall et al.61 have used the non-conservative pressure for-
mulation of Kapila’s model. However, according to our knowledge, no pressure-based algorithms have been proposed for a full
non-equilibrium BN-type model, except a preliminary work for the homogeneous part62.
Although pressure-based methods oﬀer several potential advantages, they are non-conservative, so they are not able to cor-
rectly predict the propagation speed of shock waves. Some techniques have been proposed to cure this inherent drawback: for
instance, it would be possible to switch to a fully-conservative formulation far from material interfaces36, correction terms can
be added to the pressure equation61,63, or the pressure equation can be considered only as a predictor for the updated value to be
inserted in the conservative energy equation60. However, in this work, we do not resort to any corrective measures, because we
focus here on the validation of the proposed pressure-based BN-type model and on the convergence to the correct solution in the
low-Mach regime using a simple numerical technique, while we leave all the numerical advancements for a further work. Never-
theless, we solve conservatively the part of density and momentum equations related to the Euler equations, so the conservation
of mass and momentum of the two-phase mixture mitigate the error in the shock propagation, unless very strong discontinuities
are involved.
Weakly compressible multiphase ﬂows
Our research about a pressure-based solver well-suited for weakly compressible two-phase ﬂows was motivated also by a speciﬁc
application, the pipeline transport of pressurized carbon dioxide (CO2) within the carbon-capture and storage framework, a
promising measure to mitigate climate change64. In standard working conditions, CO2 is transported in liquid or dense gas state,
but two-phase ﬂows may occur because of transient events such as start-up, de-pressurization, or oscillations in the supply chain.


4
B. Re and R. Abgrall
In these situations, the Mach number is low, but if we treated the ﬂows as incompressible, pressure waves would generate no
changes in the density. On the contrary, the capability to correctly evaluate the density and temperature variation is of paramount
importance for a safe design of the pipeline and for ﬂow metering65. Other examples of low-Mach multiphase problems involve
sloshing phenomena, and boiling or cavitating ﬂows, which are encountered in various applications, such as combustion engines,
pumps, heat exchange, nuclear power plants, transportation and storage systems. Here, the liquid phase is almost incompressible,
but the liquid-gas mixture is highly compressible and the presence of bubbles or entrapped air, especially if close to a wall,
impacts on the ﬂow behavior and on the structural loads. Hence, we need to take into account the compressibility of both phases
to correctly evaluate the thermodynamic pressure and the wave propagation66–68.
Goal and highlights
As anticipated before, the goal of our work is to develop a pressure-based formulation of a BN-type model, well suited for the
simulation of unsteady weakly-compressible multiphase ﬂows. The rationale behind the pressure-based formulation is to avoid
preconditioning—required by a standard density-based approach—which could change the topology, as well as to have a clear
scaling, even if more than one speed of sound characterizes the ﬂow ﬁeld.
In this paper, we describe the model and validate it in simpliﬁed test cases. Therefore, we focus only on one-dimensional
problems. Nevertheless, the software implementation keeps into consideration the possibility to extend the method to two- and
three-dimensions in the future. We are particularly interested in creating a ﬂexible and robust simulation tool, which is able to
deal with diﬀerent ﬂuids and ﬂow conﬁgurations, but, whenever possible, we pursue the strategy to combine existing tools to
address speciﬁc problems. These guidelines motivate the following modeling choices, which characterize the method proposed
in this work.
• We adopt the BN-type model proposed by Saurel and Abgrall11, but we consider ﬁnite parameters for pressure and velocity
relaxation terms.
• A general thermodynamic description is assumed during the derivation of the governing equations and the numerical
method, which are thus valid for diﬀerent EOSs, such as stiﬀened gas, Peng-Robinson, and more complex, multi-parameter
EOSs.
• Two diﬀerent pressure variables are deﬁned according to the scaling proposed by Bijl and Wesseling38, so that the acoustics
is ﬁltered out from the model.
• Staggered grids are used to prevent stability issues related to the checker-board problem.
• A robust, but scheme-dependent, discretization of the non-conservative terms has been derived following the pressure
non-disturbance condition69, to avoid spurious oscillations across multi-material interfaces.
These choices are explained and justiﬁed better in the next sections.
Paper structure
The next section concisely reviews some key features of pressure-based formulations proposed for single-phase ﬂows, which
are important to support the modeling choices made in this work. Section 3 begins with the presentation of the underlying BN-
type model, continues with the derivation of the dimensionless pressure-based model, and ends with a short digression about
the thermodynamic models. Section 4 presents the numerical method developed to solve the resulting BN-type model and it
is split in four subsections: Sec. 4.1 introduces the semi-implicit time integration of the hyperbolic operator, Sec. 4.2 deﬁnes
the organization of the variables over the grids, Sec. 4.3 details the spatial discretization of each equation of the hyperbolic
part of the model, and Sec. 4.4 explains how the relaxation terms are treated. To have an organized framework, the results are
presented in three diﬀerent sections. First, Sec. 5 concerns the veriﬁcation of the proposed numerical method for single-phase
ﬂows, and compares some possible alternatives in the solution strategy. Section 6 moves to two-phase simulations but still on a
veriﬁcation level, as it presents some water-air problems without relaxation, to validate the behavior of the hyperbolic operator.
Finally, Sec. 7 presents the results of the complete model. Sections 7.1 and 7.2 give also an illustration of the eﬀects of diﬀerent
values of ﬁnite relaxation parameters, Sec. 7.4 considers almost pure ﬂuids, and Sec. 7.5 compares the results obtained with two
diﬀerent thermodynamic models. Lastly, in Sec. 8 we draw the conclusions of our works and we discuss future development
and potential exploitation. The manuscript includes also A and B, which contain some passages omitted in the derivation of the
model and the numerical discretization for reason of space.


B. Re and R. Abgrall
5
2
SOME KEY FEATURES OF STANDARD PRESSURE-BASED APPROACHES FOR
SINGLE PHASE FLOWS
Many researchers have proposed diverse pressure-based formulations for the Euler equations, able to address the challenges of
the low-Mach limits. These studies serve as a precious basis for our work, in which we attempt to blend together some key features
of these previous studies and to extend them to multiphase ﬂows. For this reason, before explaining our work, we brieﬂy review
here, without claiming to be exhaustive, some fundamental concepts widely used in numerical methods for weakly compressible
single-phase ﬂows.
One of the most challenging aspects of low Mach ﬂows is that the governing equations change their character: the system of
equations of the compressible gas-dynamics is purely hyperbolic, while its incompressible counterpart has a mixed hyperbolic-
elliptic character with inﬁnite propagation speed. As a consequence, pressure and density are weakly coupled, so the problem
of retrieving the pressure from the density becomes ill-conditioned. This explains, at least partially, the misbehavior of standard
density-based methods at low Mach33 and has motivated the widespread of pressure-based formulations. These latter strategies
reﬂect, in general, the weak pressure-density coupling by solving the governing equations in a segregate approach: the velocity
is ﬁrst predicted using a pressure approximation in the momentum equation, then a correction step is carried out to update
the pressure and, ﬁnally, the velocity is corrected for the new pressure38,70,71. Segregate solution strategies prompt the use of
staggered spatial discretization where thermodynamic variables are stored at cell centers, while velocity variables are stored
at cell faces44,72,73. Contrary to co-located formulations, staggered formulations ﬁlter spurious pressure modes providing an
improved stability, at a similar level of eﬃciency and conservation properties74.
Asymptotic analyses have suggested the use of multiple pressure variables, which account for the diﬀerent physical roles
played by the diﬀerent orders of the pressure in the low Mach limit46,75. Performing a single time scale/multiple space scale
asymptotic analysis of the Euler equations, in which the pressure for small Mach numbers (푀) is expressed as
푃= 푃(0) + 푀푃(1) + 푀2푃(2) + (푀3) ,
(1)
Klein75 showed that a scheme for low Mach ﬂows should take into consideration at least two pressure variables: the leading
order 푃(0) which plays the role of the thermodynamic variable, and the second-order term 푃(2) which is the “standard pressure”
that accounts for local force balancing and, for 푀→0, satisﬁes the Poisson equation. Instead, the ﬁrst-order term 푃(1) is
associated with long wave acoustics and it should be taken into account when pressure waves of order () are important. The
pressure decomposition (1) allows the compressible equations to converge toward the correct limit, namely to the solution of
the incompressible ones for 푀→046,71. Standard numerical methods for compressible ﬂows use on a non-dimensionalization
based on a single reference velocity (e.g. computed from a set of reference pressure, density, and length), which introduces,
in the low Mach limit, a singularity in the momentum equation, due to the presence of the term
1
푀2 in front of the pressure
gradient71. The pressure decomposition (1) cures this singularity. As a ﬁnal remark, this asymptotic analysis highlights that the
divergence condition for incompressible ﬂows, that is 훁⋅풖= 0 (where 풖is the ﬂow velocity), results from the energy equation,
not from the continuity equation, which, in the zero Mach limit, simply describes the advection of density ﬂuctuations.
The idea of multiple pressure variables can be implemented in several ways38,46,76. Here, we follow the strategy proposed by
Bijl and Wesseling38,71, who deﬁned the pressure scaling as
푃=
̃푃−̃푃r
̃휌r̃푢2
r
,
(2)
where 휌is the density, 푢is a scalar velocity, the tilde indicates dimensional variables, and the subscript r denotes reference
quantities. The scaling (2) is characterized by the parameter 푀r (reference Mach number), deﬁned as
푀2
r =
̃휌r̃푢2
r
̃푃r
,
(3)
which expresses the overall compressibility of the ﬂow ﬁeld. This strategy does not take into account the ﬁrst order pressure 푃(1).
In the low-Mach limit, the system of diﬀerential equations describing the evolution of the ﬂow becomes stiﬀ. Hence, the
explicit time stepping schemes, routinely used for highly non-linear compressible ﬂows, become ineﬃcient46,54, because the
CFL (Courant-Friedrichs-Lewy) condition imposes a severe limitation of the maximum admissible time step. To circumvents
the most stringent time step limitation, the acoustic terms should be integrated implicitly, while the convective and diﬀusive
terms can be treated explicitly, since they impose only a mild stability limitation of the time step based on the ﬂow velocity. This
strategy is called semi-implicit time integration75,77,78, and it is a common feature of pressure-based schemes, shared especially


6
B. Re and R. Abgrall
by the schemes that are able to represent all-Mach numbers—i.e., from very small to Mach numbers of order one. The concept can
been easily implemented in a semi-discrete, fractional step projection method, in which the equations to be solved sequentially
contain both implicit and explicit terms52,55,57,70,76,79. This is the strategy adopted in the present work, but, alternatively, a similar
idea can be enforced by splitting the ﬂuxes in two parts, advective and non-advective47,54,80. Both strategies can be used to derive
asymptotic preserving schemes49,55,81.
3
A BAER AND NUNZIATO TYPE MODEL FOR NON-EQUILIBRIUM MULTIPHASE
FLOWS AT LOW MACH NUMBER
In this section, we derive the set of equations that is the basis of the proposed numerical method, explaining the starting point and
the modeling choices. We start presenting the BN-type model and the relevant notation, then we derive the pressure formulation
and we apply the scaling (2).
3.1
The Baer and Nunziato type model
The non-equilibrium multiphase model derived by Saurel and Abgrall in11 assumes that each phase is compressible and evolves
with its own pressure, temperature, and velocity. The model does not consider heat or mass transfer and it tends to the Euler
equations far from the interfaces. The system of 7 governing equations can be written in the following compact non-conservative
form:
휕퐔
휕푡+ 휕퐅(퐔)
휕푥
+ 퐁(퐔)휕훼1
휕푥= 퐒푃(퐔) + 퐒푢(퐔)
(4)
퐔=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
훼1
훼휌1
훼푚1
훼퐸1
훼휌2
훼푚2
훼퐸2
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 퐅=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
0
훼휌1푢1
훼푚1푢1 + 훼푃1
(훼퐸1 + 훼푃1)푢1
훼휌2푢2
훼푚2푢2 + 훼푃2
(훼퐸2 + 훼푃2)푢2
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 퐁=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
푢I
0
−푃I
−푃I푢I
0
푃I
푃I푢I
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 퐒푃=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
휇(푃1 −푃2)
0
0
−휇푃I(푃1 −푃2)
0
0
휇푃I(푃1 −푃2)
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 퐒푢=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
0
0
−휆(푢1 −푢2)
−휆푢I(푢1 −푢2)
0
휆(푢1 −푢2)
휆푢I(푢1 −푢2)
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
where 퐔is the vector of evolution variables, 퐅is the ﬂux function (the pure conservative part), 퐁contains the non-conservative
part, 퐒푃and 퐒푢are vectors of source terms modeling, respectively, pressure and velocity relaxation. According to the standard
notation, the variables are: 훼the volume fraction (note that 훼1 + 훼2 = 1), 휌the density, 푢the velocity, 푚= 휌푢the momentum,
푃the pressure, and 퐸the total energy, deﬁned as 퐸= 푒+ 1
2휌푢2, where 푒is the internal energy. The numerical subscript of each
variable denotes the phase to which it refers1. The pressure 푃I and the velocity 푢I model the average interface values over the
two-phase control volume, and they are estimated as
푃I = 훼푃1 + 훼푃2 ,
푢I = 훼푚1 + 훼푚2
훼휌1 + 훼휌2
,
(5)
while 휇and 휆are relaxation parameters that express, respectively, how fast the pressure and velocity equilibrium is reached11,14.
These variables depends on the nature of each ﬂuid as well as on the topology of the multiphase ﬂow. For this reason, in this
work, they are user-deﬁnite ﬁnite and positive parameters. All variables in Eqs. (4) and (5) are dimensional.
Even if presented for two phases, this model can be adapted to three or more phases, provided a deﬁnition of interface and
relaxation terms is given. Moreover, considering 훼1 = 1 and 훼2 = 0, the model simpliﬁes to the classical Euler equations for
single-phase ﬂows. Furthermore, if we sum the equations per variables, we have the Euler equations for the mixture conservative
variables, that is for the mixture density, i.e., ̄휌= 훼휌1 + 훼휌2, the mixture momentum, and the mixture total energy.
Thermodynamic models are required to close the model. Each component obeys its own EOS as a pure material, so for each
ﬂuid or phase, we consider a generic EOS in the shape 푒= 푒(휌, 푃), where 푒is the internal energy per unit of volume, namely
푒= 휀휌with 휀the speciﬁc internal energy. We remind here some thermodynamic deﬁnitions and relations which are of interest
1When a single numerical subscript is written at the end of a group of variables starting with 훼, it refers to all variables in the group, e.g., 훼휌1 means 훼1휌1.


B. Re and R. Abgrall
7
in the next sections. First, we introduce the following thermodynamic derivatives
휒=
(
휕푃
휕휌
)
푒
휅=
(휕푃
휕푒
)
휌.
(6)
Accordingly, the deﬁnition of the speed of sound reads for each ﬂuid
푐2 =
(
휕푃
휕휌
)
푠
=
(
휕푃
휕휌
)
푒
+
(휕푃
휕푒
)
휌
(
휕푒
휕휌
)
푠
= 휒+ 휅
( 휕푒
휕푣
)
푠
d푣
d휌= 휒−휅1
휌2
[
휌
(휕휀
휕푣
)
푠+ 휀d휌
d푣
]
= 휒−휅1
휌2
[−휌푃−휀휌2] = 휒+ 휅푃+ 푒
휌
(7)
Deﬁnitions (6) and (7) are valid for each ﬂuid separately, although we have omitted the subscript denoting the phase to simplify
the notation. For later convenience, we deﬁne also, for each phase, an interface speed of sound as
푐2
I,휎= 휒휎+ 휅휎
푃I + 푒휎
휌휎
.
(8)
3.2
Pressure-based formulation
To formulate a pressure-based BN-type model, we need to derive an equation for the pressure evolution from the conservative
form (4). Here, we describe only the main steps and the results, while the step-by-step derivation is given in A.
The ﬁrst step consists in expressing the total energy in terms of pressure, density, momentum, and energy. Given an EOS in
the form 푒= 푒(휌, 푃), we can express the partial derivatives of 퐸with respect to 휉= {푡, 푥} as
휕퐸
휕휉=
[
1
휅
휕푃
휕휉−휒
휅
휕휌
휕휉
]
+ 푢휕푚
휕휉−푢2
2
휕휌
휕휉.
Thus, we insert this into the energy equation for phase 휎= {1, 2} in (4), which can be re-written as2
훼휎
[
1
휅휎
휕푃휎
휕푡−휒휎
휅휎
휕휌휎
휕푡+ 푢휎
휕푚휎
휕푡−
푢2
휎
2
휕휌휎
휕푡
]
+
(
푒휎+
푚2
휎
2휌휎
) [휕훼휎
휕푡+ 휕훼휎푢휎
휕푥
]
+훼휎푢휎
[
1
휅휎
휕푃휎
휕푥−휒휎
휅휎
휕휌휎
휕푥+ 푢휎
휕푚휎
휕푥−
푢2
휎
2
휕휌휎
휕푥
]
+ 휕(훼푃휎푢휎)
휕푥
−푃I푢I
휕훼휎
휕푥= 푃I휇Δ휎푃−푢I휆Δ휎푢
(9)
where we have introduced the operator Δ휎which takes the diﬀerence between the phase 휎and the opposite one, i.e., Δ1푃=
푃1 −푃2 and Δ2푢= 푢2 −푢1.
As detailed in A, we replace the temporal derivatives of 휌휎, 푚휎, and 훼휎according to the respective equations in (4). Then,
re-arranging the terms and recalling the deﬁnitions of the speed of sound (7) and interface speed of sound (8), we write the
pressure formulation of the BN-type model (4) as
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥+ 훼휌휎푐2
휎
휕푢휎
휕푥−휌휎푐2
I,휎(푢I −푢휎)휕훼휎
휕푥= −휌휎푐2
I,휎휇Δ휎푃−휅휎(푢I −푢휎)휆Δ휎푢.
(10)
3.3
Dimensionless pressure-based BN-type model
In this section, we proceed to make the system of governing equations dimensionless, according to the pressure scaling (2)
proposed by Bijl and Wesseling38. For the sake of clarity, we re-write here the volume fraction, density and momentum equation
of system (4), along with the pressure equation (10), for one phase only, highlighting the dimensional variables with a tilde. We
2Note that, since 훼1 + 훼2 = 1, 휕훼1
휕휉= −휕훼2
휕휉, so the change of sign between phase 1 and 2 in 퐁(퐔) is correctly reproduced in Eq. (9) by using 휕훼휎
휕휉.


8
B. Re and R. Abgrall
TABLE 1 This table summaries the dimensionless scaling of the variables and of some operators in equations (11)–(14). In
particular, we show the combination of reference quantities (̃휌r, ̃푢r, ̃퐿r, and ̃푃r) required to express each dimensional variable
in terms of its dimensional counterpart. The ﬁrst two columns report the variables that are not aﬀected by the pressure, whose
scaling is standard. The last column refers to the thermodynamic variables that require particular care because of the presence
of the pressure in their deﬁnition.
̃푚= 푚̃휌r̃푢r
̃푒= 푒̃휌r̃푢2
r
̃푃= 푃̃휌r̃푢2
r + ̃푃r
̃휇= 휇
1
̃퐿r ̃휌r̃푢r
̃휆= 휆̃휌r̃푢r
̃퐿r
̃푐2 =
(
푐2 +
1
푀2
r
휅
휌
)
̃푢2
r
휕
휕̃푡= ̃푢r
̃퐿r
휕
휕푡
휕
휕̃푥= 1
̃퐿r
휕
휕푥
̃푐2
I =
(
푐2
I +
1
푀2
r
휅
휌
)
̃푢2
r
Δ휎̃푢= ̃푢r Δ휎푢
Δ휎̃푃= ̃휌r̃푢2
r Δ휎푃
remind that the volume fraction 훼is, by deﬁnition, a dimensionless variable.
휕훼휎
휕̃푡+ ̃푢I
휕훼휎
휕̃푥= ̃휇Δ휎̃푃
(11)
휕훼̃휌휎
휕̃푡
+ 휕(훼̃휌휎̃푢휎)
휕̃푥
= 0
(12)
휕훼̃푚휎
휕̃푡
+ 휕(훼̃푚휎̃푢휎+ 훼̃푃휎)
휕̃푥
−̃푃I
휕훼휎
휕̃푥= −̃휆Δ휎̃푢
(13)
훼휎
휕̃푃휎
휕̃푡+ 훼̃푢휎
휕̃푃휎
휕̃푥+ 훼̃휌휎̃푐2
휎
휕̃푢휎
휕̃푥−̃휌휎̃푐2
I,휎(̃푢I −̃푢휎)휕훼휎
휕̃푥= −̃휌휎̃푐2
I,휎̃휇Δ휎̃푃−̃휅휎(̃푢I −̃푢휎)̃휆Δ휎̃푢
(14)
The scaling procedure requires the deﬁnition of the set of (dimensional) reference variables. The ﬁrst entries in this set are: a
density ̃휌r, a length ̃퐿r, and a velocity ̃푢r. Conventionally, we deﬁne dimensionless density, length, and velocity as
휌= ̃휌
̃휌r
,
푥= ̃푥
̃퐿r
푢= ̃푢
̃푢r
.
Combinations of these three reference variables are suﬃcient to make dimensionless all the variables in Eqs. (11)–(14), as shown
in Tab. 1. However, as anticipated in Sec. 2, we adopt a special scaling for the pressure to ﬁlter out the long-wave acoustics
and to cure the singularity in the momentum equations in the zero Mach limit. Indeed, we introduce also a pressure reference
variable ̃푃r, and we deﬁne the dimensionless pressure as 푃=
̃푃−̃푃r
̃휌r̃푢2
r .
Let us illustrate how this choice inﬂuences the scaling of the thermodynamic variables. To preserve the relation between the
internal and total energy at dimensionless level, we deﬁne
푒=
̃푒
̃휌r̃푢2
r
,
and
퐸=
̃퐸
̃휌r̃푢2
r
=
̃푒
̃휌r̃푢2
r
+ 1
2
̃휌̃푢2
̃휌r̃푢2
r
= 푒+ 1
2휌푢2 .
Consequently, the pressure derivatives deﬁned in (6) are scaled as
̃휒=
(
휕̃푃
휕̃휌
)
̃푒
=
(
휕푃
휕휌
)
푒
̃푢2
r = 휒̃푢2
r ,
̃휅=
(
휕̃푃
휕̃푒
)
̃휌
=
(휕푃
휕푒
)
휌= 휅.
(15)
More care is required for the speed of sound, which depends explicitly on the pressure. We want to preserve the deﬁnitions (7)
and (8) also at dimensionless level, so we deﬁne
푐2 = 휒+ 휅푃+ 푒
휌
and
푐2
I = 휒+ 휅푃I + 푒
휌
.
(16)
But, with this choice, the dimensional speed of sound in terms of dimensionless variables reads
̃푐2 = ̃휒+ ̃휅̃푃+ ̃푒
̃휌
=
[
휒+ 휅(푃+ 푒)̃휌r
휌̃휌r
]
̃푢2
r + 휅
̃푃r
휌̃휌r
= 푐2̃푢2
r + 휅
휌
̃푃r
̃휌r̃푢2
r
̃푢2
r =
[
푐2 +
1
푀2
r
휅
휌
]
̃푢2
r
(17)
where we have introduced the reference Mach number deﬁned in (3). A similar expression is found for the interface speed of
sound, which is reported in Tab. 1. As we show in the next paragraph, the additional term
1
푀2
r
휅
휌plays a fundamental role in the
scaling of the pressure equation (10).


B. Re and R. Abgrall
9
Following the deﬁnitions given above and in Tab. 1, we express all variables in Eqs. (11)–(14) in terms of their dimensionless
counterpart. By using a verbose notation to show all substitutions, we obtain
̃푢r
̃퐿r
휕훼휎
휕푡+ ̃푢r
̃퐿r
푢I
휕훼휎
휕푥=
̃휌r̃푢2
r
̃퐿r̃휌r̃푢r
휇Δ휎푃
(18)
̃휌r̃푢r
̃퐿r
휕훼휌휎
휕푡
+ ̃휌r̃푢r
̃퐿r
휕(훼휌휎푢휎)
휕푥
= 0
(19)
̃휌r̃푢2
r
̃퐿r
휕훼푚휎
휕푡
+
̃휌r̃푢2
r
̃퐿r
휕(훼푚휎푢휎+ 훼푃휎)
휕푥
+
̃푃r
̃퐿r
휕훼휎
휕푥−
푃Ĩ휌r̃푢2
r + ̃푃r
̃퐿r
휕훼휎
휕푥= −
̃휌r̃푢2
r
̃퐿r
휆Δ휎푢
(20)
̃휌r̃푢3
r
̃퐿r
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥
]
+
̃휌r̃푢3
r
̃퐿r
[
훼휌휎
(
푐2
휎+
1
푀2
r
휅휎
휌휎
) 휕푢휎
휕푥−휌휎
(
푐2
I,휎+
1
푀2
r
휅휎
휌휎
)
(푢I −푢휎)휕훼휎
휕푥
]
= −
̃휌r̃푢2
r ̃휌r̃푢2
r
̃퐿r̃휌r̃푢r
휌휎
(
푐2
I,휎+
1
푀2
r
휅휎
휌휎
)
휇Δ휎푃−
̃휌r̃푢3
r
̃퐿r
휅휎(푢I −푢휎)휆Δ휎푢.
(21)
Clearly, the previous equations can be simpliﬁed. Noting that in Eq. (20) the two terms involving ̃푃r cancel out, we can imme-
diately simplify Eqs. (18)–(21) by deleting the factors comprising ̃휌r, ̃푢r, and ̃퐿r. Then, we multiply Eq. (21) by 푀2
r and
we re-arrange the terms, gathering those with this factor together. In summary, the ﬁnal system of equations expressing the
pressure-based formulation of the BN-type model deﬁned in Eq. (4) reads
휕훼휎
휕푡+ 푢I
휕훼휎
휕푥= 휇Δ휎푃
(22)
휕훼휌휎
휕푡
+ 휕(훼휌휎푢휎)
휕푥
= 0
(23)
휕훼푚휎
휕푡
+ 휕(훼푚휎푢휎+ 훼푃휎)
휕푥
−푃I
휕훼휎
휕푥= −휆Δ휎푢
(24)
푀2
r
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥+ 훼휌휎푐2
휎
휕푢휎
휕푥−휌휎푐2
I,휎(푢I −푢휎)휕훼휎
휕푥
]
+ 휅휎
[
훼휎
휕푢휎
휕푥−(푢I −푢휎)휕훼휎
휕푥
]
= −푀2
r
[
휌휎푐2
I,휎휇Δ휎푃+ 휅휎(푢I −푢휎)휆Δ휎푢
]
−휅휎휇Δ휎푃.
(25)
We highlight that the adopted pressure scaling, by means of the additional term proportional to 휅in the deﬁnition of 푐2
휎and
푐2
I,휎, is directly responsible for the peculiar expression of Eq. (25), in which terms proportional to 푀0
r and 푀2
r coexist. The
fundamental beneﬁt of this choice is expressed by the following Remark.
Remark 1 (Multiphase incompressibility constraint). From Eq. (25), we can derive the multiphase counterpart of the kinematic
constraint for incompressible ﬂows, which for a 1D single-phase ﬂow reads 휕푢
휕푥= 0. In the limit for 푀r →0, Eq. (25) simpliﬁes
to 훼휎
휕푢휎
휕푥−(푢I −푢휎) 휕훼휎
휕푥= −휇Δ휎푃, which, exploiting Eq. (22), can be re-written as
휕훼휎
휕푡+ 휕훼푢휎
휕푥
= 0 .
(26)
Equation (26) can be considered the multiphase incompressibility condition, since, if we sum Eq. (26) for 휎= 1 and 휎= 2, we
have 휕훼푢1+훼푢2
휕푥
= 휕̄푢
휕푥= 0, where ̄푢is the mixture velocity. This result reminds us that the incompressibility condition comes from
the energy equation, and not from the mass equation.
Remark 2 (Symmetry). The model expressed by Eqs.(22)–(25) is symmetric, in the sense that an exchange in the phase index
휎does not change the set of governing equations.
Remark 3 (Uniform pressure and velocity ﬁeld). From Eqs.(22)–(25), we can see that, if the initial state is (spatially) uniform
in pressure and velocity, this condition is preserved in time. Actually, if 푢1 = 푢2 = 푢I = 푢and 푃1 = 푃2 = 푃I = 푃, we have:
휕훼휎
휕푡+ 푢휕훼휎
휕푥= 0 ;
휕휌휎
휕푡+ 푢휕휌휎
휕푥= 0 ;
휕푢휎
휕푡= 0 ;
휕푃휎
휕푡= 0 ,
so no pressure or velocity variations are generated11.


10
B. Re and R. Abgrall
3.4
Thermodynamic models used in this work
The system of governing equations presented above and the numerical method described in the next section are derived without
any speciﬁc assumption on the thermodynamic models, as long as the EOS of ﬂuid can be expressed as 푒= 푒(휌, 푃). Since
this requirement is pretty easy to meet, most of the EOSs used for academic and industrial purposes can be adopted to model
the behavior of each component within the proposed pressure-based BN-type model. To introduce the nomenclature used in
the following sections, we brieﬂy introduce here the models used in for the simulations presented in result sections 5, 6 and 7,
namely the stiﬀened gas model82,83 and the polytropic Peng-Robinson EOS84.
A complete thermodynamic model of a pure ﬂuid at equilibrium can be obtained from two independent EOSs, the thermal
and caloric one. For the stiﬀened gas, their dimensional expressions read (omitting the tilde to lighten the notation)
푒휎(휌휎, 푃휎) =
푃휎+ 훾휎푃∞,휎
훾휎−1
+ 휌휎푞휎
and
푇휎(휌휎, 푃휎) =
푃휎+ 푃∞,휎
푐푣,휎휌휎(훾휎−1) ,
(27)
where 푇is the temperature and 푐푣=
(
휕휀
휕푇
)
푣is the speciﬁc heat capacity at constant volume, which is constant in the stiﬀened gas
approximation. The parameters 훾(ratio of speciﬁc heat capacity), 푃∞, and 푞depend on the material and can be determined by
ﬁtting experimental data, e.g., the saturation curve22,85. The expressions of other thermodynamic variables can be found in86,87.
Stiﬀened gas model can be considered an extension of the polytropic ideal gas (which is recovered when 푃∞= 푞= 0) able
to take into consideration the repulsive eﬀects present in all states of matter (modeled by the term 푒−휌푞
훾−1 ) and the cohesive forces
typical of liquid and solid states (thanks to the term 훾푃∞)85. This capability together with its simplicity accounts for its wide
use in the research activities focused on the development of models and numerical tools for two-phase ﬂows, as the present
one. When the focus is the study of complex two-phase ﬂow behavior, e.g., for the investigation of water cavitation problems
or in process simulation of renewable energy technologies, more accurate EOSs may be required. An answer to this demand
may come from cubic EOSs, which are widely used also in industrial applications because they combine a decent accuracy with
computational eﬃciency. A popular instance in this class is the Peng-Robinson84 EOS, which is used in this work to model
liquid and vapor CO2 in Sec. 7.5. The expression of thermal and caloric EOSs for this model can be found in86,88.
All thermodynamic variables are made dimensionless following the scaling rules deﬁned in Tab. 1 and a standard scaling for
the temperature according to a reference dimensional value ̃푇r, that is ̃푇= 푇̃푇r.
4
NUMERICAL METHOD
For the aim of this work, a numerical method that is ﬁrst order accurate, both in time and in space, is considered. The system of
governing equations (22)–(25) is solved according to the Strang splitting approach, as in11,25,26,89. Hence, given the solution 퐔푛
at a initial time 푡푛, the solution 퐔푛+1 after a time interval Δ푡is obtained by the sequence of operators
퐔푛+1 = 퐿relax 퐿hyp(퐔푛) ,
(28)
where 퐿hyp is the operator that solves the hyperbolic part of the system over a time step Δ푡, while 퐿relax is the relaxation operator
that solves the system of ordinary diﬀerential equations (ODEs) considering only the relaxation terms for the velocity and the
pressure. We describe 퐿hyp in the sec. 4.1–4.3, while 퐿relax in sec. 4.4.
4.1
Temporal discretization of the hyperbolic operator
For the numerical discretization of the hyperbolic operator 퐿hyp, we start from the time integration, keeping the spatial derivatives
continuous. To mitigate the time step restriction imposed by the CFL constraint, we use a semi-implicit temporal discretization
where the acoustic eﬀects are treated implicitly. This requires to integrate implicitly the pressure gradient in the momentum
equations and the divergence of the velocity in the pressure equations. To easily handle the ﬁrst task, we adopt a time splitting
in which the momenta (and the velocities) are ﬁrst estimated by treating explicitly the pressure gradient, then they are corrected
according to the updated pressure values, as done for instance in47,54. Moreover, at the end of the time step, we recompute the


B. Re and R. Abgrall
11
density with the current advection velocity, to have a better accuracy when the Mach number is particularly low. The semi-
discretization of the governing equations per each phase reads
훼푛+1
휎
−훼푛
휎
Δ푡
+ 푢푛
I
휕훼푛+1
휎
휕푥
= 0
(29)
훼휌푛∗
휎−훼휌푛
휎
Δ푡
+
휕(훼휌푛∗
휎푢푛
휎)
휕푥
= 0
(30)
훼푚푛∗
휎−훼푚푛
휎
Δ푡
+
휕(훼푚푛∗
휎푢푛
휎+ 훼푛+1
휎
푃푛
휎)
휕푥
−푃푛
I
휕훼푛+1
휎
휕푥
= 0
(31)
푀2
r 훼푛+1
휎
[
푃푛+1
휎
−푃푛
휎
Δ푡
+ 푢푛∗
휎
휕푃푛+1
휎
휕푥
]
+ (푀2
r 휌휎푐2
휎+ 휅휎
)푛훼푛+1
휎
휕푢푛+1
휎
휕푥
−
(
푀2
r 휌휎푐2
I,휎+ 휅휎
)푛
(푢I −푢휎)푛∗휕훼푛+1
휎
휕푥
= 0
(32)
훼푚푛∗∗
휎
−훼푚푛∗
휎
Δ푡
+
휕[(훼푚푛∗∗
휎
−훼푚푛∗
휎)푢푛
휎
]
휕푥
+
휕[훼푛+1
휎
(푃푛+1
휎
−푃푛
휎)]
휕푥
−(푃푛+1
I
−푃푛
I )
휕훼푛+1
휎
휕푥
= 0
(33)
훼휌푛+1
휎
−훼휌푛
휎
Δ푡
+
휕(훼휌푛+1
휎
푢푛+1
휎
)
휕푥
= 0 ,
(34)
where 훼휌푛∗, 훼푚푛∗= (훼휌)푛∗푢푛∗and 푢푛∗are the predicted density, momentum and velocity. The superscripts 푛and 푛+ 1 indicate,
as usual, variables at the previous time step, 푡푛, and at the end of the hyperbolic operator, 푡푛+1. The double star in the momentum
correction equation (33) highlight that it is related to the predicted density, i.e. 훼푚푛∗∗
휎
= (훼휌휎)푛∗푢푛+1.
We can interpret the previous set of equations also in the framework of multiple-pressure variables. Indeed, this semi-implicit
discretization computes the convective and thermodynamic eﬀects in a predictor step, composed by Eqs. (29)–(31), while the
high-order pressure eﬀects, that is the ones due to the term 푃(2) in Eq. (1), are corrected implicitly through the Eqs. (32) and
(33)75. Moreover, if we sum (31) and (33), we get the equation for the momentum 훼푚푛∗∗with implicit pressure gradient, i.e.,
훼푚푛∗∗
휎
−훼푚푛
휎
Δ푡
+
휕(훼푚푛∗∗
휎푢푛
휎+ 훼푛+1
휎
푃푛+1
휎
)
휕푥
−푃푛+1
I
휕훼푛+1
휎
휕푥
= 0 .
The ﬁnal momentum is computed after solving (34) as
훼푚푛+1
휎
= (훼휌휎)푛+1푢푛+1 = (훼휌휎)푛+1 훼푚푛∗∗
휎
훼휌푛∗
휎
(35)
On the other hand, we use a diﬀerent approach for the density equations (30) and (34): the results of the former one, i.e., 훼휌푛∗
휎,
are used only while solving (31)–(33); but, at the end of the time step, while solving (34), the densities 훼휌푛+1
휎
are computed
starting from 훼휌푛
휎, discharging 훼휌푛∗
휎. This re-computation allows the use of the most updated advection velocity, 푢푛+1
휎
, which is
particularly important in ﬂow problems close to the incompressibility limit, where the density equations simplify to transport
equations. We compare the results obtained with and without density re-computation in the ﬁrst numerical test, in Sec. 5.1.
A ﬁnal remark concerns the divergence of the velocity in (32), which needs to be treated implicitly to overcome acoustic
CFL limitations54. However, since Eqs. (29)–(34) are solved in a segregate approach in the order they appear, the value of the
velocity 푢푛+1 is not known while solving the pressure equation. Inspired by the use of the momentum equation to derive an
implicit pressure equation54,55, we use Eq. (33) to approximate the value of 푢푛+1. Indeed, to a ﬁrst approximation, the diﬀerence
in the convective terms can be neglected38, so the ﬁnal velocity can be approximated as
푢푛+1
휎
= 푢푛∗
휎+ Δ푡
훼휌푛∗
휎
[
−
휕[훼푛+1
휎
(푃푛+1
휎
−푃푛
휎)]
휕푥
+ (푃푛+1
I
−푃푛
I )
휕훼푛+1
휎
휕푥
]
.
(36)
As explain better in the following, this choice, and in particular the term 푃푛+1
I
= ∑
휎훼휎푃푛+1
휎
, couples the pressure equations for
all phases together. On the contrary, Eqs. (30),(31), (33), and (34), are solved per each phase independently.
Remark 4 (Alternative formulation). Considering the deﬁnition of 훼푚푛∗∗
휎
and that (훼휌휎)푛∗is already known, instead of the
momentum correction (33), we could also correct directly the velocity by solving
푢푛+1
휎
−푢푛∗
휎
Δ푡
+
휕[푢푛+1
휎
−푢푛∗
휎)푢푛
휎
]
휕푥
+
1
훼휌푛∗
휎
휕[훼푛+1
휎
(푃푛+1
휎
−푃푛
휎)]
휕푥
−
1
훼휌푛∗
휎
(푃푛+1
I
−푃푛
I )
휕훼푛+1
휎
휕푥
= 0 .
(37)


12
B. Re and R. Abgrall
P1, αρ1
P2, αρ2
PN−1
αρN−1
PN, αρN
αm1, u1
αm0, u0
αmN−1, uN−1 αmN, uN
P0, αρ0
PN+1
αρN+1
ζ0
ζ1
ζN
C1
CN
x0
x1
x2
xN−1
xN
FIGURE 1 Spatial discretization and variables positioning. At the bottom, the computational domain Ω = [푥0, 푥푁] is drawn,
along with the position of the grid nodes 푥푖and the boundary domains sketched with grey dashed lines. The upper part of
the picture illustrates the primary and staggered grids. They are drawn with a vertical development and separately from the
computational domain only for greater clarity, but all grids and cells here deﬁned should be considered as one-dimensional. In
the upper part of the picture, blue color refers to the primary cells 푖and to the quantities 훼, 훼휌, and 푃, which are stored at their
centers, represented by blue square marks. Green color refers to the staggered cells, at the center of which the kinetic variables
훼푚and 푢are stored; these cells are sketched by dashed lines and their centers are represented by green circle marks. In red, the
boundary values for the primary grid complement the picture. To lighten the notation of variables, phase indication is omitted
and the numerical subscripts refer simply to the spatial cells.
4.2
Variables positioning: primary and staggered grids
For the spatial discretization of the pressure-based BN-type model, we consider two ﬁnite-volume schemes based on staggered
grids: one for the thermodynamic variables, and one for the kinematic variables. FIGURE 1 shows how the staggered grids
are deﬁned. We split the computational domain Ω = [푥0, 푥푁] in 푁intervals, deﬁned by the equidistant grid nodes 푥푖, with
푖= 0, … , 푁. Then, the grid for the ﬁnite-volume discretization of the thermodynamic quantities (hereafter called primary
grid) is built by deﬁning each cell 푖corresponding to the grid element [푥푖−1, 푥푖]. Conversely, the grid for the ﬁnite-volume
discretization of the kinematic variables (hereafter called staggered grid) is built by centering each cell 휁푘around the grid nodes
푥푘, between the centroids of the adjacent grid elements (or the boundary). In summary, the cells on primary and staggered grids
are deﬁned as
primary:
푖= [푥푖−1, 푥푖]
∀푖= 1, … , 푁;
staggered: 휁푘=
[푥푘+ 푥푘−1
2
, 푥푘+1 + 푥푘
2
]
∀푘= 1, … , 푁−1;
휁0 =
[
푥0, 푥1 + 푥0
2
]
;
휁푁=
[푥푁+ 푥푁−1
2
, 푥푁
]
.
As it appears from the given deﬁnitions, starting from grid nodes equally spaced by a distance Δ푥, all primary cells have the
same size ||푖|| = Δ푥, while the staggered cells have the same size ||휁푘|| = Δ푥only far from boundary. Indeed, the ﬁrst and last
staggered cells are half the size, i.e., ||휁0|| = ||휁푁|| = Δ푥∕2.
A cell-centered ﬁnite-volume discretization over the primary grid is used to solve the volume fraction, density, and pressure
equations, that is Eqs. (29), (30), (34), and (32). So, the thermodynamic variables (sometimes called “scalar” in contrast to the
kinematic, vectorial variables) are approximated over the cell 푖as
(휉휎)푛
푖=
1
||푖|| ∫
푖
휉휎(푥, 푡푛)d푥,
with
휉∈{훼휎, 훼휌휎, 푃휎} .
On the other hand, the momentum and momentum update equations, (31) and (33), are discretized over the staggered grid. Using
a ﬁnite-volume scheme, we deﬁne the cell value of the momentum as
(훼푚휎)푛
푘=
1
||휁푘|| ∫
휁푘
훼푚휎(푥, 푡푛)d푥.
These are the ﬁnite-volume cell values, illustrated also in Fig. 1. However, it is often required to map variables from “their” grid
to the other. In this case, we perform a weighted average, which, being the grid nodes equidistant, simply results in an arithmetic
mean.
Remark 5 (Notation and mapping). To have a clear notation, we use the subscript 푖for quantities over the primary grid, and
푘for quantities over the staggered one. Accordingly, a thermodynamic variable with a subscript 푘refers to its mapped value


B. Re and R. Abgrall
13
over the staggered cell 푘, and vice versa. To clarify this point, consider the following example. The notation (훼휌휎)푘indicates the
mapped density over the cell 휁푘, computed as (훼휌휎)푘= [(훼휌휎)푖+ (훼휌휎)푖+1]∕2 for 푖= 푘. We can then use this mapped density to
estimate the velocity in the cell 휁푘as (푢휎)푘= (훼푚휎)푘∕(훼휌휎)푘.
4.3
Spatial discretization of the hyperbolic operator
Each hyperbolic diﬀerential equation in the model is integrated in time for the interval Δ푡= 푡푛+1 −푡푛and in space over all cells
(푗or 휁푘). The spatial derivative of the convective ﬂuxes is approximated through numerical evaluations of the ﬂuxes at the cell
interfaces. In particular, we use a ﬁrst-order approximation based on the Rusanov ﬂux, as in11,19. This choice is motivated by
simplicity, as it avoids the complexities related to the solution of local Riemann problems with several waves90–92.
The use of staggered grids makes the discretization of some speciﬁc terms easy and natural.
• The convective velocity to be used in the ﬂux computation on the primary grid is directly the velocity deﬁned over the
staggered grid. For instance, the Rusanov ﬂux for 훼휌휎at the interface between 푖and 푖+1 is computed as
퐹Rus
푖+ 1
2
(훼휌휎, 푢휎) = 1
2(푢휎)푘
[(훼휌휎)푖+1 + (훼휌휎)푖
] −1
2|(푢휎)푘| [(훼휌휎)푖+1 −(훼휌휎)푖
] .
(38)
• The pressure gradient in the momentum equation is readily approximated by a centered diﬀerence scheme, since the values
of the pressure at the faces of the staggered cells are available:
∫
휁푘
휕푃휎
휕푥d푥≈(푃휎)푖+1 −(푃휎)푖.
(39)
• The divergence of the velocity in the pressure equation is easily discretized through a centered diﬀerence scheme:
∫
푖
휕푢휎
휕푥d푥≈(푢휎)푘−(푢휎)푘−1 .
(40)
A major complexity of the spatial discretization of Eqs. (29)–(34) concerns the presence of non-conservative terms involving
the gradient of the volume fraction. This is a challenge common to all BN-type models, which include the term 퐁(퐔) 휕훼1
휕푥that
models the momentum and energy transfer among phases but prevents to write Eq. (4) in divergence form. This means that
it is not possible to deﬁne weak solutions in the standard sense of distribution and to determine unique wave speeds. From a
numerical point of view, these non-conservative products have to be integrated as source terms, rather than as ﬂuxes. Since a
naive discretization may introduce spurious oscillations across material interfaces between phases with diﬀerent speciﬁc heat
ratios, we seek a robust discretization of non-conservative terms involving the volume fraction gradient by explicitly enforcing
that uniform velocity and pressure proﬁles are maintained69. Honestly, diﬀerent strategies can be followed to integrate the non-
conservative terms associated to the linearly degenerate ﬁelds, as, in particular, path conservative schemes93. However, this
approach does not guarantee to always converge to the correct weak solution of non-conservative hyperbolic problems94. In
addition, a primitive formulation of the governing equations, such as the pressure one here considered, facilitates preserving
pressure equilibrium near material interfaces36. All in all, for weak discontinuities, as the ones considered in the framework of
weakly compressible ﬂows, any consistent and accurate enough method would be adequate to achieve a satisfactory solution95.
4.3.1
Volume fraction and density equations
Without any non-conservative term, the density equations (30) and (34) are easily discretized in space as
||푖||
Δ푡
[(훼휌휎)⋄
푖−(훼휌휎)푛
푖
] = −
[
퐹Rus
푖+ 1
2
(훼휌⋄
휎, 푢푛
휎) −퐹Rus
푖−1
2
(훼휌⋄
휎, 푢푛
휎)
]
(41)
where the expression for the Rusanov ﬂuxes is given in Eq. (38), and the superscript ⋄corresponds to 푛∗and to 푛+ 1 in the
spatial discretization of (30) and (34), respectively.
The discrete volume fraction equation is
||푖||
Δ푡
[(훼휎)푛+1
푖
−(훼휎)푛
푖
] = −퐻푢(훼푛+1
휎
, 푢푛
I )푖,
(42)


14
B. Re and R. Abgrall
where 퐻푢(훼푛+1
휎
, 푢푛
I )푖≈∫푖푢푛
I
휕훼푛+1
휎
휕푥d푥is a suitable approximation of the non-conservative term. To deﬁne this operator, we follow
the idea that starting from a uniform pressure and velocity, no variations in these variables should be generated11,19,69, see also
Remark 3.
If we assume a uniform velocity ﬁeld, e.g. (푢휎)푘= (푢휎)푘+1 = (푢I) = 푢, the discrete mass equation reads
||푖||
Δ푡
[(훼휌휎)푛+1
푖
−(훼휌휎)푛
푖
] = −1
2푢[(훼휌휎)푖+1 −(훼휌휎)푖−1
] + 1
2|푢| [(훼휌휎)푖+1 −2(훼휌휎)푖+ (훼휌휎)푖−1
] ,
(43)
where we have dropped the superscripts in the left hand side to lighten the notation. Let us consider now the special case when
also the density ﬁeld is uniform19. If (휌휎)푖= (휌휎)푖−1 = (휌휎)푖+1, the mass equation reads
||푖||
Δ푡
[
(훼휌휎)푛+1
푖
−(훼휌휎)푛
푖
]
= −(휌휎)푢
2
[
(훼휎)푖+1 −(훼휎)푖−1
]
+ (휌휎)|푢|
2
[
(훼휎)푖+1 −2(훼휎)푖+ (훼휎)푖−1
]
.
(44)
If velocity and density are uniform, the density should remain constant, i.e., (휌휎)푛+1
푖
= (휌휎)푛
푖. So, in order to make Eq. (42)
compatible with Eq. (44) in this speciﬁc case, we need that
퐻푢(훼푛+1
휎
, 푢푛
I )푖= 푢
2
[
(훼휎)푖+1 −(훼휎)푖−1
]
−|푢|
2
[
(훼휎)푖+1 −2(훼휎)푖+ (훼휎)푖−1
]
.
From this, we deﬁne the following non-conservative operator 퐻푢:
퐻푢(훼푛+1
휎
, 푢푛
I )푖= ̂퐹Rus
푖+ 1
2
(훼푛+1
휎
, (푢I)푛
푖
) −̂퐹Rus
푖−1
2
(훼푛+1
휎
, (푢I)푛
푖
) ,
(45)
where
̂퐹Rus
푖+ 1
2
(훼푛+1
휎
, (푢I)푛
푖
) = 1
2(푢I)푛
푖
[(훼휎)푛+1
푖+1 + (훼휎)푛+1
푖
] −1
2|(푢I)푛
푖| [(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
]. We use the notation ̂퐹Rus to highlight that
that the resulting discretization of 퐻푢depends on the discretization for the convective ﬂux in the mass equation, but, at the same
time, the ̂⋅indicates that it is not a proper ﬂux, as (푢I)푖is the mapping of the interface velocity over the primary cell 푖, not an
interface velocity. This choice guarantees that 퐻푢= 0, if the volume fraction is uniform, as expected by the integration of 푢I
휕훼휎
휕푥.
4.3.2
Momentum equations
The spatial discretization of the momentum equations (31) and (33) requires the integration of three terms: the convective ﬂux,
for which we adopt a Rusanov ﬂux; the pressure gradient, discretized by the central ﬁnite diﬀerence deﬁned in (39); and the
non-conservative term, for which we deﬁne the operator 퐻푃(훼푛+1
휎
, 푃푛
I )푘≈∫휁푘푃푛
I
휕훼푛+1
휎
휕푥d푥exploiting the non-disturbance pressure
and velocity condition, as explained in the following. Accordingly, the discrete equations of the predicted momentum and of the
corrected momentum read
||휁푘||
Δ푡
[
(훼푚휎)푛∗
푘−(훼푚휎)푛
푘
]
= −
[
퐹Rus
푘+ 1
2
(훼푚푛∗
휎, 푢푛
휎
) −퐹Rus
푘−1
2
(훼푚푛∗
휎, 푢푛
휎
)]
−[(훼휎)푛+1
푖+1 (푃휎)푛
푖+1 −(훼휎)푛+1
푖
(푃휎)푛
푖
] + 퐻푃(훼푛+1
휎
, 푃푛
I )푘
(46)
||휁푘||
Δ푡
[
(훼푚휎)푛∗∗
푘
−(훼푚휎)푛∗
푘
]
= −
[
퐹Rus
푘+ 1
2
(훿훼푚푛∗∗
휎, 푢푛
휎
) −퐹Rus
푘−1
2
(훿훼푚푛∗∗
휎, 푢푛
휎
)]
−[(훼휎)푛+1
푖+1 (훿푃휎)푛+1
푖+1 −(훼휎)푛+1
푖
(훿푃휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 훿푃푛+1
I
)푘
(47)
where we have used the operator 훿to identify the jump in the kinetic and pressure variables between the prediction and correction
step. More precisely,
훿훼푚푛∗∗
휎
= 훼푚푛∗∗
휎
−훼푚푛∗
휎= 훼휌푛∗
휎훿푢푛+1 = 훼휌푛+1
휎
(푢푛
휎−푢푛∗
휎) ,
훿푃푛+1
휎
= 푃푛+1
휎
−푃푛
휎,
and
훿푃푛+1
I
= 푃푛+1
I
−푃푛
I .
The Rusanov ﬂuxes are deﬁned, as usual, as
퐹Rus
푘+ 1
2
(훼푚푛∗
휎, 푢푛
휎
) = 1
2
[(훼푚휎)푛∗
푘+1(푢휎)푛
푘+1 + (훼푚휎)푛∗
푘(푢휎)푛
푘
] −1
2푆푛
푘+ 1
2
[(훼푚휎)푛∗
푘+1 −(훼푚휎)푛∗
푘
]
(48)
where 푆푛
푘+ 1
2
=
max
(|||(푢휎)푛
푘+1
||| , |||(푢휎)푛
푘
|||
)
. The same expression but with (훿훼푚휎)푛∗∗instead of (훼푚휎)푛∗is used for
퐹Rus
푘+ 1
2
(훿훼푚푛∗∗
휎, 푢푛
휎
).


B. Re and R. Abgrall
15
The discretization of the non-conservative term 퐻푃(훼휎, 푃I)푘is derived, similarly to 퐻푢(훼휎, 푢I)푖, by imposing the non-
disturbance pressure and velocity constraint. The whole process is detailed in B. For conciseness, we report here only the ﬁnal
deﬁnition of the non-conservative operator 퐻푃:
퐻푃(훼푛+1
휎
, 푃푛
I )푘= (푃I)푛
푘
[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] ,
(49)
퐻푃(훼푛+1
휎
, 훿푃푛
I )푘= [(푃I)푛+1
푘
−(푃I)푛
푘
] [(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] ,
(50)
where (푃I)푘= 1
2
[(푃I)푖+ (푃I)푖+1
] is the interface pressure mapped at the staggered cell 휁푘.
Finally, we highlight that Eqs. (46) and (47) are solved only over the internal cells 휁푘, with 푘= 1, … , 푁−1; while the values
of the predicted and updated momentum on 휁0 and 휁푁are imposed through the boundary treatment described in Sec. 4.3.5.
4.3.3
Velocity correction equation
If we consider the velocity correction equation (37), its discretization is straightforwardly derived from the spatially discrete
equation of momentum correction (47) and it reads
||휁푘||
Δ푡
[
(푢휎)푛+1
푘
−(푢휎)푛∗
푘
]
= −
[
퐹Rus
푘+ 1
2
(훿푢푛∗∗
휎, 푢푛
휎
) −퐹Rus
푘−1
2
(훿푢푛∗∗
휎, 푢푛
휎
)]
−
1
(훼휌휎)푛∗
푘
[(훼휎)푛+1
푖+1 (훿푃휎)푛+1
푖+1 −(훼휎)푛+1
푖
(훿푃휎)푛+1
푖
] +
1
(훼휌휎)푛∗
푘
퐻푃(훼푛+1
휎
, 훿푃푛+1
I
)푘,
(51)
where the Rusanov ﬂuxes are deﬁned, similarly to (48), as
퐹Rus
푘+ 1
2
(푢푛+1
휎
, 푢푛
휎
) = 1
2
[(푢휎)푛+1
푘+1(푢휎)푛
푘+1 + (푢휎)푛+1
푘
(푢휎)푛
푘
] −1
2푆푛
푘+ 1
2
[(훼푚휎)푛∗
푘+1 −(훼푚휎)푛∗
푘
] .
4.3.4
Pressure equation
We develop now the discrete version of the non-conservative pressure equation (32). First, we observe that, in the considered
ﬁnite volume context, the thermodynamic variables and the volume fraction are constant within the primary cell, as in38. So,
integrating Eq. (32) over a cell 푖, we can write
푀2
r (훼휎)푛+1
푖
||푖||
Δ푡
[(푃휎)푛+1
푖
−(푃휎)푛
푖
] + 푀2
r (훼휎)푛+1
푖
∫
푖
푢푛∗
휎
휕푃푛+1
휎
휕푥
d푥
+ (K퐹
휎)푛
푖(훼휎)푛+1
푖
∫
푖
휕푢푛+1
휎
휕푥d푥−(K퐻
휎)푛
푖∫
푖
(푢I −푢휎)푛∗휕훼푛+1
휎
휕푥d푥= 0 ,
(52)
where, to have a more compact expression, we have introduced the two coeﬃcients
(K퐹
휎)푛
푖= 푀2
r (휌휎)푛
푖(푐2
휎)푛
푖+ (휅휎)푛
푖,
(K퐻
휎)푛
푖= 푀2
r (휌휎)푛
푖(푐2
I,휎)푛
푖+ (휅휎)푛
푖,
which are known, because the variables (푐2
휎)푛
푖, (푐2
I,휎)푛
푖, and (휅휎)푛
푖are computed using the thermodynamic state at cell 푖and at
time 푡푛. For instance, (휅휎)푛
푖= 휅((휌휎)푛
푖, (푒휎)푛
푖
), according to deﬁnition (15).
The second step concerns the discretization of the ﬁrst integral term, which, thanks to the product rule, is re-written as
∫
푖
푢푛∗
휎
휕푃푛+1
휎
휕푥
d푥= ∫
푖
휕(푃휎)푛+1(푢휎)푛∗
휕푥
d푥−∫
푖
(푃휎)푛+1 휕푢푛∗
휎
휕푥d푥.
To approximate the ﬁrst term in the previous expression, we deﬁne the following ﬂux (similar to (38))
퐹Rus
푖+ 1
2
(푃푛+1
휎
, 푢푛∗
휎
) = 1
2(푢휎)푛∗
푘
[(푃휎)푛+1
푖+1 + (푃휎)푛+1
푖
] −1
2
|||(푢휎)푛∗
푘
|||
[(푃휎)푛+1
푖+1 −(푃휎)푛+1
푖
] ,
while for the second one, we rely on the central approximation scheme for the divergence of the velocity given in (40). We obtain:
∫
푖
푢푛∗
휎
휕푃푛+1
휎
휕푥
d푥≈퐹Rus
푖+ 1
2
(푃푛+1
휎
, 푢푛∗
휎
) −퐹Rus
푖−1
2
(푃푛+1
휎
, 푢푛∗
휎
) −(푃휎)푛+1
푖
[(푢휎)푛∗
푘−(푢휎)푛∗
푘−1
]
= 1
2
[(푃휎)푛+1
푖+1 −(푃휎)푛+1
푖
] [
(푢휎)푛∗
푘−|||(푢휎)푛∗
푘
|||
]
+ 1
2
[(푃휎)푛+1
푖
−(푃휎)푛+1
푖−1
] [
(푢휎)푛∗
푘−1 + |||(푢휎)푛∗
푘−1
|||
]
.


16
B. Re and R. Abgrall
A third aspect to be considered is the approximation of the non-conservative term involving the gradient of the volume
fraction. Given the similarities with the non-conservative term in the volume fraction equation, we adopt the same operator 퐻푢
deﬁned in (45), but for the velocity jump. Thus,
∫
푖
(푢I −푢휎)푛∗휕훼푛+1
휎
휕푥d푥≈퐻푢
(훼푛+1
휎
, (푢I −푢휎)푛∗)
푖,
with 퐻푢
(훼푛+1
휎
, (푢I −푢휎)푛∗)
푖= ̂퐹Rus
푖+ 1
2
(훼푛+1
휎
, ((푢I)푛∗
푖−(푢휎)푛∗
푖
)) −̂퐹Rus
푖−1
2
(훼푛+1
휎
, ((푢I)푛∗
푖−(푢휎)푛∗
푖
)).
The remaining integral term in Eq. (52) is easily approximated by a central diﬀerence scheme, but it requires an expression
for the velocities at the time step 푡푛+1. This latter is derived from the discretization of the velocity update, Eq. (51), discharging
the diﬀerences in the convective terms. It reads
(푢휎)푛+1
푘
= (푢휎)푛∗
푘+
Δ푡
||휁푘|| (훼휌휎)푛∗
푘
[
−[(훼휎)푛+1
푖+1 (훿푃휎)푛+1
푖+1 −(훼휎)푛+1
푖
(훿푃휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 훿푃푛+1
I
)푘
]
.
(53)
In conclusion, the discrete version of the pressure equation is
푀2
r (훼휎)푛+1
푖
||푖||
Δ푡
[(푃휎)푛+1
푖
−(푃휎)푛
푖
]
= −푀2
r (훼휎)푛+1
푖
1
2
{[(푃휎)푛+1
푖+1 −(푃휎)푛+1
푖
] [
(푢휎)푛∗
푘−|||(푢휎)푛∗
푘
|||
]
+ [(푃휎)푛+1
푖
−(푃휎)푛+1
푖−1
] [
(푢휎)푛∗
푘−1 + |||(푢휎)푛∗
푘−1
|||
]}
−(K퐹
휎)푛
푖(훼휎)푛+1
푖
[
(푢휎)푛+1
푘
−(푢휎)푛+1
푘−1
]
+ (K퐻
휎)푛
푖퐻푢
(훼푛+1
휎
, (푢I −푢휎)푛∗)
푖.
(54)
Remark 6 (Equation coupling). The implicit treatment of the velocity divergence in the pressure equation determines the cou-
pling of the discrete pressure equations for both phases. In (54), the velocities (푢휎)푛+1
푘
and (푢휎)푛+1
푘−1 depend on (훿푃I)푛+1
푘
and (훿푃I)푛+1
푘−1
(cfr. Eqs. (36) and(53)). Recalling the deﬁnition of (푃I) and the mapping from the primary to the staggered, we have
(푃I)푛+1
푘
= 1
2
[(푃I)푛+1
푖+1 + (푃I)푛+1
푖
] = 1
2
∑
휎
[(훼휎)푛+1
푖+1 (푃휎)푛+1
푖+1 + (훼휎)푛+1
푖
(푃휎)푛+1
푖
] ,
from which it appears evident the involvement of the pressure of both phases in the deﬁnition of the velocity (푢휎)푛+1.
Consequently, we need to solve the pressure equations (54) for both phase together, i.e., in a coupled way.
4.3.5
Boundary conditions
To impose boundary conditions, we distinguish between primary and staggered grid. For the primary grid, we use a standard
method based on two ghost states deﬁned outside the computational domain. With reference to Fig. 1, these states are denoted
by subscripts 0 and 푁+ 1, on the left and right boundary, respectively, and are deﬁned as
(퐖휎
)푛+1
B
=
⎡
⎢
⎢⎣
(훼휎)푛+1
B
(훼휌휎)푛+1
B
(푃휎)푛+1
B
⎤
⎥
⎥⎦
for 휎= {1, 2} , and B = {0, 푁+ 1} .
According to the physical boundary condition we need to model, the value of the variables in (퐖휎
)푛+1
B
mirrors the state of the
adjacent internal cell (1 or 푁), or it is directly imposed as boundary value (for the details about this selection process, see
for instance96). The boundary state (퐖휎
)푛+1
B
is then used in the discrete equations (41), (42), (46),(54), and (47) to evaluate the
ﬂuxes, the non-conservative terms, and the central diﬀerence schemes at the boundary interfaces.
For the staggered grid, we use a diﬀerent strategy, because the ﬁrst and the last staggered cells (휁0 and 휁푁) are boundary cells.
In addition, the velocities (푢휎)0 and (푢휎)푁are already stored at the boundary interfaces (see Fig. 1). Thus, the momentum and
velocity in these two cells are not computed by solving Eqs. (46) and (47), but they are computing according to the physical
boundary condition. In particular, we distinguish two cases: if the boundary velocity (푢휎)B is known, its value is imposed;
otherwise the velocity is extrapolated from the two closest internal cells. For instance, considering the left boundary:
(푢휎)0 =
{
(푢휎)B
if 푢휎known at 푥0 ,
2(푢휎)1 −(푢휎)2
otherwise.
Then (훼푚휎)0 = (훼휌휎)0 + (훼휌휎)1
2
(푢휎)0 ,
where (훼휌휎)0, (훼휌휎)1 are the density values on the primary cells. This deﬁnition applies also to the implicit velocity in the
pressure equations, i.e., while solving the equation (54) for (푃휎)푛+1
1
and (푃휎)푁) the expressions for the velocity (푢휎)푛+1
0
and
(푢휎)푛+1
푁
are the ones given above, instead of (53).


B. Re and R. Abgrall
17
4.3.6
Solution of implicit system
The implicit treatment of some terms in the discretization of the hyperbolic operator makes the equations coupled between
adjacent cells. Indeed, the structure of mass, volume fraction, and momentum equations, e.g.(41), (42), (46), and (47), can be
approximately represented as
Δ푥푗
Δ푡
[
푤푛+1
푗
−푤푛
푗
]
= −
[
퐹R
(
푤푛+1
푗
, 푤푛+1
푗+1
)
−퐹L
(
푤푛+1
푗
, 푤푛+1
푗−1
)]
−[퐷R −퐷L
] + 퐻(훼푛+1) ,
(55)
where 푤is the unknown of a speciﬁc phase in the cell 푗(over the primary or staggered grid), Δ푥푗is the cell volume, 퐹L and 퐹R
are ﬂuxes across the left and right cell face, 퐷R and 퐷L refer to the cell centered discretization and the term 퐻represents the
discretization of the non-conservative terms (involving diﬀerent values of 훼푛+1 according to the equation we are considering).
The superscripts 푛and 푛+ 1 indicates, generally, known and unknown values, respectively. Obviously, not all right hand side
terms are present in every equation, but there is always at least one term that generates the cross-coupling.
The previous expression can be further simpliﬁed as
Δ푥푗
Δ푡
[
푤푛+1
푗
−푤푛
푗
]
= −Φ푗
(
푤푛+1
푗
, 푤푛+1
푗+1, 푤푛+1
푗−1
)
+ R푗(푤)
(56)
where Φ includes the ﬂuxes or the non-conservative terms that are function of the unknowns themselves and R includes all the
known terms. We use a ﬁrst order Taylor expansion to approximate Φ as
Φ푛+1 = Φ
(
푤푛+1
푗
, 푤푛+1
푗+1, 푤푛+1
푗−1
)
≈Φ푛+ 휕Φ
휕푤푗
훿푤푛+1
푗
+
휕Φ
휕푤푗+1
훿푤푛+1
푗+1 +
휕Φ
휕푤푗−1
훿푤푛+1
푗−1 ,
where 훿푤푛+1
푗
= 푤푛+1
푗
−푤푛
푗. Since we use Rusanov ﬂuxes, the derivatives of ﬂuxes and the non-conservative term (required only
in the volume fraction equation) can be easily computed analytically. Hence, for Eqs. (41), (42), (46), and (47), for each phase
separately, we need to solve a set of equations in the form
[Δ푥푗
Δ푡+ 휕Φ
휕푤푗
]
훿푤푛+1
푗
+
휕Φ
휕푤푗+1
훿푤푛+1
푗+1 +
휕Φ
휕푤푗−1
훿푤푛+1
푗−1 = Φ푛
푗+ R푗(푤) ,
(57)
which is comprised of 푁or 푁+ 1 equations, depending on whether we are considering the primary or the staggered grid.
The resulting systems are linear and they can be written, in a compact form, as [퐴]휹풘= 퐑, where [퐴] is a tridiagonal matrix
including the derivatives of Φ, 훿풘is the vector of unknown, and 퐑is the known term. These systems are solved through the
Generalized Minimal Residual (GMRES) algorithm provided by the PETSc library97.
Similar observations can be drawn also for the pressure equation (54), which however is solved for both phases together. In
this case, the generalized term Φ푃includes also the unknown terms deriving from (53), that is
(Φ푃)푛+1
푖
= Φ ((푃1)푛+1
푖
, (푃1)푛+1
푖+1 , (푃1)푛+1
푖−1 , (푃2)푛+1
푖
, (푃2)푛+1
푖+1 , (푃2)푛+1
푖−1
) .
Consequently, the ﬁrst-order Taylor expansion involves six diﬀerent unknowns, which can be organized in a vector in this
order: [… , (훿푃1)푖−1, (훿푃2)푖−1, (훿푃1)푖, (훿푃2)푖, (훿푃1)푖+1, (훿푃2)푖+1, … ], so that the resulting ﬁnal linear system can be written as
[퐴]휹풘= 퐑, where [퐴] is now a banded matrix with an upper bandwidth of 3 and a lower bandwidth of 2.
4.4
Relaxation operator
According to the Strang splitting introduced in (28), the solution of the hyperbolic operator described in Secs. 4.1–4.3 provides
a known set of variables that are used as initial data to solve the system of ODEs associated with the relaxation terms. For this
reason, in this section, we re-deﬁne the notation to distinguish the intermediate solutions after the hyperbolic operator 퐿hyp, and
the relaxation operator 퐿relax as follows
퐔◦= 퐿hyp(퐔푛)
and
퐔∙= 퐿relax(퐔◦) .
(58)
In practice, in this subsection the superscript ◦denotes what in subsections 4.1 and 4.3 was denoted by 푛+1, and the superscript
∙refers to the variables computed during the relaxation processes.
The relaxation operator 퐿relax plays a fundamental role in driving phasic velocities and pressures toward the equilibrium, close
to interfaces. The characteristic time of these processes depend on many factors, as the ﬂuids features and the multiphase ﬂow
topology. For instance, the parameter 휇, which expresses the velocity of the pressure relaxation, may depend on the compress-
ibility of the ﬂuid and the parameter 휆, which governs the rate of the velocity homogenization, may depend on ﬂuid viscosity11.


18
B. Re and R. Abgrall
In general, pressure and velocity relaxation are much faster than the dynamics associated to the wave propagation, to the point
that they are something modeled as instantaneous phenomena, by assuming inﬁnite 휇and 휆11,25,27. However, in this work, we use
ﬁnite relaxation parameters, as in2,98, to allow wider modeling possibilities. Indeed, we could deﬁne the relaxation parameters
in terms of the average interfacial area of bubbles99, or, if we had experimental data about diﬀerent multiphase ﬂow topologies,
we could tune the relaxation parameters in our model to match the data.
Assuming a characteristic time much shorter than the one characterizing the hyperbolic operator, the ODE system associated
to the relaxation operator is derive from the continuous governing equations (22)–(25) neglecting convective and transport terms.
It reads
d훼1
d푡= 휇Δ1푃
(59)
d훼휌휎
d푡
= 0
for 휎= {1, 2}
(60)
d훼푚휎
d푡
= −휆Δ휎푢
for 휎= {1, 2}
(61)
푀2
r 훼휎
d푃휎
d푡= −
[
푀2
r 휌휎푐2
I,휎+ 휅휎
]
휇Δ휎푃−푀2
r 휅휎(푢I −푢휎)휆Δ휎푢
for 휎= {1, 2} .
(62)
This system is characterized by a high degree of stiﬀness, so we use the implicit Backward Euler scheme for the time integration.
Equations (60) give immediately 훼휌푛+1
휎
= 훼휌◦
휎. If we use this result in (61) and we integrate in time, we have
훼휌∙
1
푢∙
1 −푢◦
1
Δ푡
= −휆(푢∙
1 −푢∙
2
)
훼휌∙
2
푢∙
2 −푢◦
2
Δ푡
= +휆(푢∙
1 −푢∙
2
) ,
where the only unknowns are the velocities. The solution of this system, expressed in term of Δ푢, is
Δ푢= (푢∙
1 −푢∙
2
) = (푢◦
1 −푢◦
2
) / [
1 + 휆Δ푡
훼휌∙
1 + 훼휌∙
2
훼휌∙
1훼휌∙
2
]
(63)
which, as expected, gives Δ푢→0 when 휆→∞, so that 푢∙
1 = 푢∙
2 =
훼휌∙
1푢◦
1+훼휌∙
2푢◦
2
훼휌∙
1+훼휌∙
2
= 푢∙
I. In the opposite case, for 휆= 0, we have
푢∙
휎= 푢◦
휎.
The remaining part in the ODE system comprises the volume fraction equation (59) and the two pressure equations (62).
After the discretization of the time derivatives, these equations can be re-written as
훼∙
1 −훼◦
1 + 휇Δ푡Δ푃= 0
(64)
푀2
r 훼∙
1(푃∙
1 −푃◦
1 ) + K퐻
1 휇Δ푡Δ푃+ K푈
1 휆Δ푡Δ푢= 0
(65)
푀2
r (푃∙
1 −Δ푃−푃◦
2 ) + 훼∙
1(Δ푃+ 푃◦
2 −푃◦
1 ) + (K퐻
1 −K퐻
2
) 휇Δ푡Δ푃+ (K푈
1 −K푈
2
) 휆Δ푡Δ푢= 0
(66)
where K퐻
휎=
[
푀2
r 휌휎푐2
I,휎+ 휅휎
]
, as in (52), and K푈
휎= 푀2
r 휅휎(푢I −푢휎). Reminding that the velocities 푢∙
휎are given by (63), the last
terms in (65) and (66) are known. For the discretization of the coeﬃcient K퐻
휎, we approximate the thermodynamic variables
and the interface pressure by using the values at the end of the hyperbolic operator. This choice is a simplifying assumption,
which slightly mitigates the non-linearity of the system (64)–(66), and it is motivated by the absence of diﬀerences noted in24
while solving the pressure relaxation system approximating the integral value of the interface pressure by 푃∙
퐼or 푃◦
퐼.
From a numerical point of view, the non-linear system (64)–(66) presents some unfavorable features, such as the simultaneous
presence of very small and very large terms, which could cause a loss of accuracy, the stiﬀness and the non-linearity. To tackle
these aspects, we rely also for the relaxation operator on the PETSc non-linear solver97 and, in particular, on the trust-region
Newton-based solver.
5
VERIFICATION FOR SINGLE-PHASE FLOWS
Since both the model and the numerical method proposed in this work are new, before focusing on two-phase simulations, we
present in this section some single-phase tests, to verify the numerical method and, in particular, the low-Mach treatment and
the core part of the hyperbolic operator. The governing equations (29)–(34) (in their fully discrete versions given in Sec. 4.3)
are here solved only for one phase, but the numerical solution algorithm is kept unaltered. And by that, we mean that the volume


B. Re and R. Abgrall
19
TABLE 2 Stiﬀened gas parameters for the ﬂuids (air and water) used in the numerical tests for single-ﬂuid ﬂows of Sec. 5 and
for two-phase ﬂows without relaxation of Sec. 6
훾[−]
푃∞[Pa]
푐푣[J∕kg K]
푞[J∕kg]
Air:
1.4
0
717.6
0
Water:
4.4
6.8 ⋅108
4178.0
0
fraction equation is solved and the non-conservative terms are included while building the system matrix [퐴], even if we expect
them to be identically null. However, in single phase simulations, the relaxation operator is not applied, or, in other words, 휆= 0
and 휇= 0.
5.1
Low Mach Riemann problem for a perfect gas
We start with a Riemann problem test at particularly low Mach number, presented in100. The pipe is ﬁlled with a perfect gas,
i.e. air with the parameters given in Tab. 2, at very low pressures and the left and right chambers features weak pressure and
velocity jumps, according to the data given in the row lmAir in Tab. 3. The solution is represented by two rarefaction waves, plus
a central contact discontinuity which moves at 푢푠= 4.7 10−3 m∕s. The Mach number is lower than 0.012 all over the domain.
Figure 2 displays the results at the ﬁnal time obtained with the standard formulation described by (29)–(34) considering only
one phase. Six simulations are run imposing six diﬀerent time steps Δ푡, deﬁned as Δ푡= 푡퐹∕푁푡where 푡퐹is the ﬁnal time and
푁푡is the number of integration steps used to reach the ﬁnal time 푡퐹. The simulations are labeled in the picture according to the
number 푁푡, which goes from 500 to 15 (from the smallest to the largest time step). As reported in the caption, some of them
lead to an acoustic CFL greater than one, in short, CFL(|푢| + 푐) > 1. The contact discontinuity, which has moved by only one
cell, is sharply represented, while the rarefaction waves are smeared because of the ﬁrst-order accuracy of the Rusanov ﬂux.
We use this test also to show the role played by the density correction and the velocity formulation. Figure 2 compares the
results obtained with and without density re-computation. In particular, we compare three formulations:
휌햺헌햿헂헋헌헍: solves only the mass equation (30) at the beginning of the time step, i.e. 훼휌푛+1
휎
←훼휌푛∗
휎and (34) is skipped;
휌햺헌헅햺헌헍: solves only the mass equation (34) at the end of the time step, i.e. 훼휌푛∗
휎←훼휌푛
휎and (30) is skipped;
휌햼허헋헋: solves the mass equation (30) at the beginning of the time step and then re-computes density at the end through (30),
i.e. the standard formulation.
From the density proﬁle, we can notice that solving the mass equation only at the beginning of the time step, so that using
the convective velocity 푢푛
휎, leads to some oscillations across the contact discontinuity, which are ampliﬁed if the CFL number
increases. The old value of the velocity 푢푛
휎does not account for the pressure correction, which is responsible for enforcing the
incompressibility condition (see Remark 1). On the other hand, from the velocity proﬁle, we can notice that the computation of
the density only at the end of the time produces slightly worse results than the standard formulation with density re-computation.
Since the density equation (41) does not present numerical diﬃculties, e.g. it does not include non-conservative terms, and its
computational eﬀort is almost negligible with respect to the solution of the other equations, we adopt the re-computation as the
standard formulation.
A further open question in the development of the numerical method here proposed concerns the momentum or velocity
correction, that is whether (33) can be substituted by (37). For this reason, we have re-run the simulations presented in Figures 2
and 3 with the velocity correction, where (33) is solved instead of (37). No notable diﬀerences are detected, and, in particular,
the same conclusions about the density re-computation are drawn, as shown by Fig. 4.
5.2
Low Mach Riemann problem for a stiﬀened gas
In this section, we address the simulation of a water pipe ﬂow under the stiﬀened gas model, proposed in100. The thermodynamic
parameters are given in Tab. 2 and the initial data are reported in the row lmWater in Tab. 3. The Riemann problem is charac-
terized by a weak pressure ratio and has the structure of the test presented in Sec. 5.1, with the contact discontinuity that moves
at 푢푠= 8.04 m∕s. Although we observe a higher speed in this test with respect to the test considering air, the Mach number is


20
B. Re and R. Abgrall
TABLE 3 Initial conditions for the single-phase tests presented in Sec.5. The coordinates 푥0 and 푥푁delimits the domain, which
is split in 푁primary cells. The initial position of the discontinuity is 푥d and the ﬁnal time is 푡퐹. The six rightmost columns report
the velocity, pressure, and density characterizing the left and right states of the Riemann problem, denoted by the subscripts 퐿
and 푅, respectively.
Test
푥0
푥푁
푁
푥d
푡퐹
푢퐿
푢푅
푃퐿
푃푅
휌퐿
휌푅
[m]
[m]
[−]
[m]
[s]
[m∕s]
[m∕s]
[Pa]
[Pa]
[kg∕m3]
[kg∕m3]
lmAir
-0.5
0.5
1000
0
0.25
0
0.008
0.4
0.399
1.0
1.0
lmWater
-0.5
0.5
1000
0
10−4
0
15
108
0.98 ⋅108
103
103
lmWaterLong
-250
250
5000
0
0.095
0
15
108
0.98 ⋅108
103
103
Lax
-0.5
0.5
1000
0
0.12
0.698
0
3.528
0.571
0.445
0.5
FIGURE 2 Low Mach Air: results at 푡퐹= 0.25 s, obtained in six simulations, each one considering a diﬀerent time step
Δ푡= 푡퐹∕푁푡, with 푁푡the number of steps as indicated in the legend. The standard formulation with density re-computation and
momentum correction is used. The analytical solution of the Riemann problem (initial conditions given in Tab. 3) is shown as
a dashed line. The six numbers of steps 푁푡correspond to the following CFL numbers.
푁푡
500
250
100
50
25
15
max CFL(|푢| + 푐)
0.4
0.8
1.9
3.8
7.6
12.6
max CFL(|푢|)
0.004
0.008
0.02
0.04
0.08
0.13
even lower, below 0.01, due to the high speed of sound. Figure 5 displays the results obtained by solving (29)–(34) for one single
phase. We have considered diﬀerent time steps, which correspond also to acoustic CFL numbers greater than one (up to 12), as
indicated in the caption by CFL(|푢| + 푐). The smaller is the time step, the better is the agreement with the analytical solution.
Additionally, we test the capability to capture travelling material waves over a long simulation, by repeating the same Riemann
problem test but over a longer time, that is 푡퐹= 0.095 s, as proposed in100. All test information are reported in the row
lmWaterLong in Tab. 3. Figure 6 displays the detail of the solution ﬁeld close to the contact discontinuity, which at the end
of this test has reached 푥푠= 0.7638 m, so it has crossed 7 grid cells. The quality of our results compares well with the ones
reported by Abbate et al.100 for their implicit scheme, so we can state that our scheme is able to correctly compute the position
and the velocity of the moving material wave.


B. Re and R. Abgrall
21
FIGURE 3 Low Mach Air: comparison of diﬀerent treatments of density equation, at ﬁnal time 푡퐹. Initial conditions are given
in Tab. 3 and they are the same of Fig. 2, and we use 푁푡= 50. The two pictures in the second row show two details of density and
velocity proﬁles, corresponding to the regions indicated by the rectangles in the plots of the ﬁrst row. In 휌햺헌햿헂헋헌헍and 휌햺헌헅햺헌헍,
the mass equation is solved only once, at the beginning and at the end of the time step, respectively. 휌햼허헋헋refers to the standard
formulation with density correction. In all cases, the momentum correction equation is used.
FIGURE 4 Low Mach Air: results at 푡퐹obtained with the alternative formulation, involving velocity correction. The test is
the same of Fig. 3, i.e., lmAir test of Tab. 3 with 푁푡= 50. In 휌햺헌햿헂헋헌헍and 휌햺헌헅햺헌헍, the mass equation is solved only once,
at the beginning and at the end of the time step, respectively. 휌햼허헋헋refers to the standard formulation with density correction.
Diﬀerently from Fig. 3, here the velocity correction equation (37) is used. Since the results do not diﬀer notably from the ones
obtained with the momentum correction, we report here only the zoomed regions, highlighted by rectangles in the ﬁrst row of
Fig. 3.
5.3
Lax problem
Finally, we end this single-phase section presenting the results for the Lax shock-tube test101, routinely used to validate standard
compressible schemes, to investigate the behavior of the proposed scheme at Mach numbers between 0.3 and 1, so not so low.
This shock test is characterized by an initial discontinuity also in the velocity, which is not null in the left chamber. Initial
conditions and test data are given in Table 3, in the row Lax. When the diaphragm bursts, the initial discontinuity evolves in a


22
B. Re and R. Abgrall
FIGURE 5 Low Mach Water: results at 푡퐹= 0.1 ms, obtained in six simulations, each one considering a diﬀerent time step
Δ푡= 푡퐹∕푁푡, with 푁푡the number of steps as indicated in the legend. The standard formulation with density re-computation and
momentum correction is used. The analytical solution of the Riemann problem (initial conditions given in Tab. 3) is shown as
a dashed line. The six numbers of steps 푁푡correspond to the following CFL numbers.
푁푡
500
250
100
50
25
15
max CFL(|푢| + 푐)
0.4
0.7
1.9
3.7
7.5
12.4
max CFL(|푢|)
0.003
0.006
0.015
0.03
0.06
0.1
FIGURE 6 Low Mach Water Long: the same Riemann problem test of Fig. 5 is run over a longer time 푡퐹= 0.095 s, using
푁푡= 900, on a grid with Δ푥= 0.1. Initial conditions are given in Tab. 3. The analytical solution of the Riemann problem is
shown as a dashed line. The right picture shows the detail of the density ﬁeld in proximity of the traveling contact discontinuity,
i.e. the region enclosed in the rectangle in the left picture.
leftward moving rarefaction waves and a rightward moving shock wave, with a contact discontinuity in between. The results at
푡퐹= 0.12 s are shown in Fig. 7 for diﬀerent time steps Δ푡. In the right part of the domain, where the Mach number is higher,
the numerical solution does not agree well with the analytical one. However, this discrepancy is an expected manifestation of
the non-conservation of the total energy, but beyond that, the numerical results of the proposed scheme show an acceptable
agreement with the analytical ones although we are not operating within the target regime of weakly compressible ﬂows.


B. Re and R. Abgrall
23
FIGURE 7 Lax test: numerical results at 푡퐹= 0.12 s, obtained in four simulations, each one considering a diﬀerent time step
Δ푡= 푡퐹∕푁푡, with 푁푡the number of steps as indicated in the legend. The standard formulation with density re-computation and
momentum correction is used. The analytical solution of the Riemann problem (initial conditions given in Tab. 3) is shown as
a dashed line. The four numbers of steps 푁푡correspond to the following CFL numbers.
푁푡
1000
750
500
250
max CFL(|푢| + 푐)
0.57
0.75
1.13
2.26
max CFL(|푢|)
0.18
0.25
0.37
0.74
6
NUMERICAL RESULTS FOR THE HYPERBOLIC OPERATOR FOR TWO-PHASE
FLOWS
In this section, we present two-phase ﬂow results computed by using only the hyperbolic operator, without any relaxation process.
Results of the complete numerical methods are shown in next section. Taking into consideration the conclusions of the previous
section, we use the standard formulation with density re-computation, i.e. (29)-(34). We organize our analysis in subsequent
steps, starting from the numerical validation of two fundamental properties: the behavior of the hyperbolic operator without
mixing in Sec. 6.1, and the fulﬁllment of the pressure non-disturbance condition in Sec. 6.2. Then, we present the results of
the proposed method on some reference Riemann problems available in the literature about BN-type models, in Sec. 6.4, and,
ﬁnally, on a water-air mixture problem in Sec. 6.5.
6.1
No mixing water-air test
The ﬁrst two-phase test involves liquid water and air governed by the stiﬀened gas model with the parameters listed in Tab. 2.
As initial condition, the phases are uniformly dispersed with equal volume fraction 훼1 = 훼2 = 0.5, in a shock-tube where a mild
pressure jump is imposed between the two chambers: 푃퐿= 100 bar at the left, and 푃푅= 50 bar at the right. A null velocity and
the temperature 푇= 270 K are applied uniformly in the domain. The initial position of the discontinuity is at 푥d = 0 and the
grid spacing is Δ푥= 0.001 m.
Being the volume fraction uniform and given the absence of relaxation terms, in this test, each phase evolves independently
from the other one. Thus, the exact solution can be computed by solving the Riemann problem for the Euler equations. Figure 8
shows the results at the ﬁnal time of 0.16 ms computed with two diﬀerent time steps: the smallest one corresponds to an acoustic
CFL number slightly above 1 only for the liquid, while the biggest time step results in acoustic CFL number greater than 2 for


24
B. Re and R. Abgrall
FIGURE 8 No mixing test: results of the two-phase shock-tube Riemann problem with uniform volume fraction 훼1 = 훼2 = 0.5
and with no relaxation, at 푡퐹= 0.16 ms using 푁푡= 500 and 푁푡= 25 time steps. The absence of relaxation and the uniform
volume fraction make the ﬂuids (air and water) evolve separately, so the numerical results are compared to the analytical solutions
of two single-phase Riemann problems. The acoustic CFL conditions are
for 푁푡= 500,
liquid:
max CFL(|푢| + 푐) = 1.3
max CFL(|푢|) = 0.001
gas:
max CFL(|푢| + 푐) = 0.14
max CFL(|푢|) = 0.03
for 푁푡= 25,
liquid:
max CFL(|푢| + 푐) = 26.4
max CFL(|푢|) = 0.02
gas:
max CFL(|푢| + 푐) = 2.7
max CFL(|푢|) = 0.5
both phases. Although the shock and the rarefaction waves appear smeared in liquid phase, the numerical results agree well with
the analytical solutions, both in terms of position of the waves and of downstream conditions.
6.2
Pure advection water-air problem
Here, we investigate a pure advection problem: a column of water-air mixture with a liquid volume fraction 훼1,c = 0.9 is
transported at a velocity of 100 m∕s in a uniform pressure ﬁeld at 푃= 1 bar, involving a mixture with 훼1,L = 훼1,R = 0.1.
The initial temperature is 270 K for both phases. The parameters of the stiﬀened gas model for the ﬂuids are the same as in the
previous test, and are listed in Tab. 2. Initially, the column is located at 0.2 < 푥< 0.4, within the domain Ω = [0, 1].
This test is performed considering diﬀerent discretizations, all imposing the convective CFL = 0.5. The results at time
푡퐹= 3 ms over three grids (with 400, 800, and 1600 cells) are shown and compared to the exact solution in Fig. 9. From the
second row of the picture, we can appreciate that the no pressure or velocity oscillations arise and the initially uniform ﬁelds are
correctly preserved during the time evolution. This achievement is crucial for a correct discretization of the non-conservative
terms69. Beyond pressure and velocity, a good agreement between the numerical and the exact solution is observed also for
the volume fraction and mixture density variables, for which the smearing of the contact discontinuity decreases with the grid
reﬁnement. To conﬁrm this behavior, we have performed also a grid convergence study, presented in Fig. 10, computing the
discrete 퐿1 error between the numerical and the exact mixture density at the ﬁnal time, normalized by the 퐿1 norm of the initial
mixture density, as
퐸̄휌(Δ푥) =
∫Ω || ̄휌(Δ푥, 푡퐹) −̄휌ex(푥, 푡퐹)|| d푥
∫Ω || ̄휌ex(푥, 푡0)|| d푥
=
∑
푖|| ̄휌푖(푡퐹) −̄휌ex(푥푖, 푡퐹)|| Δ푥
∫Ω || ̄휌ex(푥, 푡0)|| d푥
.
(67)
The numerical error converges with the order of Δ푥1∕2, as expected when using a ﬁrst-order scheme for BN-type models89.


B. Re and R. Abgrall
25
FIGURE 9 Pure advetion test of a column of water-air mixture at uniform velocity, results at 푡퐹= 3 ms, with CFL(|푢|) = 0.5.
The initial position of the column is shown as a dashed line in the top row, which displays also the water volume fraction 훼1
on the left, and the mixture density ̄휌= 훼휌1 + 훼휌2 on the right; the results at 푡퐹obtained over three diﬀerent grids (with 햭헑
the number of primary cells) are compared with the exact solution shown by a solid black line. The second row presents the
pressure and velocity ﬁelds at 푡퐹for both phases; only the results computed over the coarsest grid are shown for brevity, as no
diﬀerences are observed using ﬁner grids.
FIGURE 10 Pure advetion test: grid convergence study. The considered grid spacings correspond to 100⋅2[1∶7] points and the 퐿1
norm of the error on the mixture density is computed according to (67). The dashed line displays the convergence rate of Δ푥1∕2.
6.3
Veriﬁcation using a manufactured solution
In this subsection, we perform a grid convergence study for a test with non-uniform pressure and velocity. To do that, we switch
oﬀall relaxation terms and add to the right-hand side of the equations a source 휙(푥, 푡) to be determined by a manufactured
solution. We follow the strategy proposed by Hennessey et al.102, and we express the exact solution in terms of the primitive
variables 퐖= [훼1, 휌1, 푢1, 푃1, 휌2, 푢2, 푃2
]T as
푊ex
푗(푥, 푡) = 훽푗+ 훿푗
(1 + 푎1,푗푥+ 푎2,푗푥2) (1 + 푏1,푗푡+ 푏2,푗푡2) ,
푗= 1, … , 7
(68)
where [훽푗, 훿푗, 푎1,푗, 푎2,푗, 푏1,푗, 푏2,푗
] are constant. The set of dimensionless values for 훽푗is [0.5, 0.5, 0, 0, 0.4, 0, −0.2], while the
remaining parameters are chosen randomly within the interval [0.1, 0.2]. These choices yield to a solution that varies smoothly


26
B. Re and R. Abgrall
FIGURE 11 Manufactured solution test: grid convergence study. Inﬁnity-norm of the error on the mixture density, normalized
with respect to the integral of the density over the domain. The ﬁnal time step 푡퐹= 0.5 is reached in 푁푇= 2400 time steps,
which correspond to a CFL(|푢|) = 0.25 for air, on the smallest grid. The dashed line displays the convergence rate of 1. A
least-square ﬁt of the logarithmic values of the errors gives a slope of 1.0024.
and monotonically within the domain, starting from the constant states expressed by 훽푗coeﬃcients. The ﬂuids are air and water,
deﬁned according to the stiﬀened gas model, with the parameters listed in Tab. 2.
We run this test over diﬀerent grids representing the domain 0 ≤푥≤1, and we compute the ﬁnal solution at time 푡퐹= 0.5.
The exact solution is given by evaluating Eq. (68) at the ﬁnal time. The 퐿-inﬁnity norm of the error on the mixture density, is
shown in Fig. 11. The order of convergence is close to 1, as expected while using the Rusanov scheme, as here.
6.4
Reference Riemann problems with perfect gases
The goal of this section is to validate the proposed approach through some tests commonly used in the research community
devoted to the development of one-dimensional numerical schemes for the BN-type models. Neglecting tests involving strong
shock waves or vanishing phases, we have selected from the literature three Riemann problems for which the analytical solution
is given: the ﬁrst two, i.e., the sonic point and the 123-problem, are taken from103,104 (named there Test 3 and Test 4, respectively);
the third one reproduces the Test-case 1 in105, and it is called solid contact in the following.
Before describing each one, let us remark that these tests are not properly representative of low-Mach problems, but they
provide anyhow an important contribution for the veriﬁcation of the hyperbolic operator. Moreover, in these three tests, both
ﬂuids follow the perfect gas model, with the air parameters given in Tab. 2, so we prefer to use the notation phase 1 and phase
2, rather than solid and gas. Finally, for the sake of completeness, we report here the estimate for the shock speed103 we have
used to draw the analytical solutions:
푢s = 푢pre ± 푐pre
√
1 + 훾+ 1
2훾
(푃post + 푃∞
푃pre + 푃∞
−1
)
,
(69)
where the subscript pre and post refer to the pre- and post-shock states and the plus and minus sign is used for a right and a left
traveling shock, respectively.
Sonic point
This test was presented in103 to assess the correct resolution of a sonic rarefaction. The two phases have initially the same
pressure, density and velocity, but the mixture composition diﬀers between the left and right states:
left:
푃= 1.0 Pa
휌= 1.0 kg∕m3
푢= 0.75 m∕s
훼1 = 0.8,
right:
푃= 0.1 Pa
휌= 0.125 kg∕m3
푢= 0.0 m∕s
훼1 = 0.3.
Therefore, the solution of the two phases is the same except for the volume fraction, and it is composed by a shock wave and a
contact discontinuity, both right-traveling, and a left sonic rarefaction wave. The numerical results shown in Fig. 12 at the time
푡퐹= 0.15 s agree fairly well with the exact solution given in103, except for the intermediate state after the shock, where, however,


B. Re and R. Abgrall
27
FIGURE 12 Sonic point test: results at the time 푡퐹= 0.15 s. Both phases are air. The initial solution for the phase 1 is shown by
a dashed grey line; the initial pressure, density and velocity of phase 2 are the same. The numerical results obtained over a grid
with Δ푥= 0.0025 m and considering 푁푡= 500 time steps are compared with the exact solution of this Riemann problem given
in103. The convective CFL with respect to the shock speed, which is 푉푠≈2 푚∕푠, is CFL(푉푠) = 0.24, which is pretty similar to
the acoustic one since CFL(|푢| + 푐) = 0.32 for both phases.
some discrepancies are expected as we are using a pressure-based, that is non-conservative, solver. Moreover, the asymmetry
between phases in the density proﬁles is simply due to the smearing of the volume fraction discontinuity. More important, in
this test, is the correct resolution of the sonic rarefaction, without any non-physical entropy glitch at the sonic point.
Two-phase 123-problem
This test involves a region close to vacuum, so it is useful to assess the pressure positivity. Initially, the ﬂuids are at uniform
pressure 푃= 0.4 Pa and density 휌= 1.0 kg∕m3. A discontinuity is imposed in the middle of the domain, 푥퐷= 0: on the left,
the volume fraction is 훼1 = 0.8 and the velocity is −2.0 m∕s, on the right, the volume fraction is 훼1 = 0.5 and the velocity is
+2.0 m∕s. The solution consists in two symmetric rarefactions and a stationary contact discontinuity in between, where, at the
ﬁnal time 푡퐹= 0.15 s, the pressure and density are extremely small: 푃(푥퐷, 푡퐹) = 0.0019 Pa and 휌(푥퐷, 푡퐹) = 0.0219 kg∕m3 103.
The numerical results are displayed in Fig. 13. The pressure and the density are computed accurately, preserving the positivity.
However, the discontinuity in the volume fraction appears to be very diﬀused, but a similar behavior for the Rusanov’s scheme
is reported also by Coquel et al.105.
Solid contact
The last Riemann problem we present was proposed by Coquel et. al.105 and, diﬀerently from the previous ones, it does not
involve an initial symmetry between the two phases. The initial ﬁeld is described in Tab. 4, and its evolution encompasses seven
diﬀerent types of waves: for phase 1, a left-traveling shock, a material contact discontinuity moving at velocity 푢1, a phase
fraction discontinuity moving with velocity 푢2 and a right-traveling rarefaction wave; for phase 2, a left-traveling rarefaction fan,
the phase fraction discontinuity, and a right-traveling shock. To be able to compare our results with the analytical and numerical
solution in105, we deﬁne the interface velocity and pressure as 푃I = 푃1 and 푢I = 푢2. The solution computed after 0.15 s is


28
B. Re and R. Abgrall
FIGURE 13 Two-phase 123-problem: results at the time 푡퐹= 0.15 s. Both phases consist in air. The initial solution for the
phase 1 is shown by a dashed grey line; the initial pressure, density and velocity of phase 2 are the same. The numerical results
obtained over a grid with Δ푥= 0.0024 m and considering 푁푡= 250 time steps are compared with the exact solution of this
Riemann problem given in103. The convective and acoustic CFLs are max CFL(|푢|) = 0.5 and max CFL(|푢| + 푐) = 0.69, for
both phases.
TABLE 4 Initial conditions for the Solid contact Riemann problem in Sec. 6.4. The subscripts 퐿and 푅refer to the left and
right state with respect to the initial position of the discontinuity 푥퐷= 0.
훼퐿
푢퐿
푃퐿
휌퐿
훼푅
푢푅
푃푅
휌푅
[−]
[m∕s]
[Pa]
[kg∕m3]
[−]
[m∕s]
[Pa]
[kg∕m3]
Phase 1
0.2
-0.02609
0.3
0.21430
0.7
-0.03629
0.95776
0.96964
Phase 2
0.8
0.00007
1.0
1.00003
0.3
-0.00004
1.0
0.99993
displayed in Fig. 14. A good agreement with the analytical solution both in terms of intermediate values and wave positions
conﬁrms the correctness of the numerical implementation of the hyperbolic operator. This positive outcome is also justiﬁed
by the fact that in this test, we have the lowest maximum Mach numbers among the three Riemann problems presented in this
section, that is max 푀1 = 0.72 for phase 1 and max 푀2 = 0.06 for phase 2.
6.5
Water-air mixture test
In this section, we reproduce the test proposed in19 under the name Smooth shock tube test case. The ﬂuids are water and air
but, diﬀerently from Sec. 6.1, an initial discontinuity is imposed also in the volume fraction. The water is modeled under the
stiﬀened gas model, using 푃∞= 6.0⋅108 Pa as in19, while the remaining EOS parameters are the same given in Tab. 2. Initially,
the ﬂuids are at rest and the densities, 휌w = 1050 kg∕m3 for the water and 휌a = 1.2 kg∕m3 for the air, are uniform along the tube.


B. Re and R. Abgrall
29
FIGURE 14 Solid contact test: results at the time 푡퐹= 0.15 s. Both phases consist in air and the initial solution is given in Tab. 4.
The numerical results obtained over a grid with Δ푥= 0.001 m and considering 푁푡= 200 time steps are compared with the
analytical solution of this Riemann problem given in105. The shock positions in the analytical solution are estimated with (69).
On the top right, the solution variables 훼휌1 and 훼휌2 (called partial densities) are displayed. The acoustic CFLs max CFL(|푢|+푐)
are 1.7 for phase 1 and 0.95 for phase 2.
Pressure and volume fraction are diﬀerent: in the left chamber (푥< 0), 푃= 106 Pa and 훼w = 0.3, whereas in the right chamber
(푥> 0), 푃= 105 Pa and 훼w = 0.7. The domain Ω = [−0.65, 0.65] m is divided in 650 primary cells. The results computed at
푡퐹= 350 μs, with two diﬀerent numbers of time steps (푁푡= 500 and 푁푡= 125), are shown in Fig. 15. In the ﬁrst case, the
maximum acoustic CFL is smaller than one (max 퐶퐹퐿(푢+|푐|) ≈0.5 for both phases), whereas in the second case, the acoustic
CFL is greater than 2 for both phases, with convective CFL reaching maxa CFL(|푢|) = 0.75 for air and maxw CFL(|푢|) = 0.0005
for water. An estimation based on the values of the solution variables across the volume fraction discontinuity leads to a value
for the interface velocity 푢퐼≈0.6 m∕s; since at the ﬁnal time step it has moved only about Δ푥∕10, its displacement cannot
be distinguished in Fig. 15. As a reference, Fig. 15 displays also the results for the air reported in19. The match is reasonably
good in proximity to the rarefaction wave and the contact discontinuity. On the contrary, the shock position is not captured
correctly by the present method. This is an inherent limitation of the adopted pressure-based formulation and we are aware that
the introduced error increases with the shock strength, however this test is on the boundary of the target application area, as the
maximum Mach number of the air is well above one. Concerning the water results, we cannot compare them with the results
in19, as their model is based on diﬀerent assumptions which make the dispersed phase—water in this test—invariant across
the shock. Nevertheless, as pointed out also in19, such a large pressure disequilibrium between water and air in this test is not
much physically reasonable. Indeed, this test serves mainly as a further validation for the hyperbolic operator, especially when
dealing with diﬀerent scale velocities between the phases and acoustic CFLs greater than one: even with the largest time step,
the scheme is stable and no spurious oscillations appear in the solution.
7
NUMERICAL RESULTS FOR TWO-PHASE FLOWS WITH RELAXATION
In this section, we ﬁnally present the results obtained with the full numerical scheme, that is including velocity and pressure
relaxation. In Sec. 7.1, the results of the BN-type model with pressure and velocity relaxation are compared to the analytical
results of Kapila’s model for two-phase ﬂows in mechanical equilibrium, and the role of the ﬁnite relaxation parameters is
investigated. Then, the water-air mixture test of Sec. 6.5 is re-run in Sec. 7.2 with pressure relaxation to compare the results of
the proposed model to the ones achieved through a diﬀerent numerical method for a BN-type model. The last three subsections


30
B. Re and R. Abgrall
FIGURE 15 Water-air mixture test: results at the time 푡퐹= 350 μs, over a uniform grid with 650 cells, in absence of relaxation
terms. Two diﬀerent time steps are considered: the smallest one (resulting from 푁푇= 500 steps, and displayed in blue for water,
and yellow for air) corresponds to max CFL(|푢| + 푐) ≈0.5 for both phases; the largest one (resulting from 푁푇= 125 steps, and
displayed in orange for water, and violet for air) corresponds to max CFL(|푢| + 푐) ≈2.2 for both phases. The initial condition
exhibits a pressure and volume fraction discontinuity at 푥= 0, but its displacement (approximately 0.21 mm) is too small to be
observed in these graphics. The left column illustrates the water volume fraction and the pressures; the right column displays
the partial densities 훼휌, the velocity and the Mach number of water (scales on left axis) and air (scales on the right axis). The
pressure and the velocity of the air, as well as the liquid volume fraction, given in Saurel et al.19 are displayed as reference.
TABLE 5 Stiﬀened gas parameters for the pure ﬂuids used in the two-phase numerical tests with relaxation involving a water-
aluminum mixture (Sec. 7.1), a water-air mixture (Sec. 7.2), and almost-pure water and air ﬂows (Sec. 7.4).
훾[−]
푃∞[Pa]
푐푣[J∕kg K]
푞[J∕kg]
Aluminum:
3.4
21.5 ⋅109
897.0
0
Water:
4.4
6.0 ⋅108
4178.0
0
Air:
1.4
0
717.6
0
refers to speciﬁc features: a strong rarefaction that generates a gas pocket in Sec. 7.3, the simulation of almost-pure ﬂuids in
Sec. 7.4, and the use of cubic equation of states in Sec. 7.5.
7.1
Water-aluminum mixture test
The test presented in this section was proposed by Furfaro et al.106 and involves a mixture of two condensed phases, water
and aluminum. The parameters of the stiﬀened gas models used in this test are summarized in Tab. 5. Initially, the phases are
uniformly dispersed with equal volume fraction 훼1 = 훼2 = 0.5 in a shock-tube with two chambers: the left one at high pressure


B. Re and R. Abgrall
31
TABLE 6 Reference solution for the water-aluminum mixture test, computed according to the Riemann solver proposed by
Petitpas et al. in107 for the Kapila’s model, which assumes mechanical equilibrium between phases. The solution consists of a
right-traveling rarefaction, a contact surface, and a left-traveling shock, and it is characterized by the four constant states given
in this table. The two intermediate states, labeled left* and right*, are separated by the contact surface, across which the pressure
푃and velocity 푢are continuous. In addition to them, the densities of each ﬂuid and the volume fraction of the water are given
in the table.
left
left*
right*
right
푃
[Pa]
109
4.583 ⋅108
4.583 ⋅108
105
푢
[m∕s]
0
124.1
124.1
0
휌w [kg∕m3]
1000
910.3
1134.0
1000
휌al [kg∕m3]
2700
2680.7
2716.8
2700
훼w [−]
0.5
0.5217
0.4701
0.5
(푃퐿= 109 Pa), and the right one at low pressure (푃푅= 105 Pa). The densities are uniform: 휌w = 1000 kg∕m3 for water and
휌al = 2700 kg∕m3 for aluminum. The initial velocity is zero everywhere. Due to relaxation processes, as the time evolves, the
volume fraction changes across the expansion and compression waves, besides the contact discontinuity originating from the
initial pressure jump. To build a reference solution, we consider the mechanical equilibrium model of Kapila12, which is the
limit model of BN model considering instantaneous pressure and mechanical relaxation, and the exact Riemann solver proposed
in107. The computed reference solution is given in Tab. 6. The maximum Mach number is 0.06 (achieved by water) and this
motivates the choice of this test to illustrate the capabilities of the proposed pressure-based method.
We start the illustration of the results considering a set of relaxation parameters suﬃciently large to drive water and aluminum
toward mechanical equilibrium, namely 휆= 109 kg∕(m3s) and 휇= 105 m s∕kg. The results at the time 푡퐹= 111 μs obtained
with the proposed method correlate well with the reference solution, as Fig. 16 shows. A good match is reached also considering
acoustic CFLs higher than one, indeed in the simulation carried out using 푁푡= 200 time steps we have max CFL(|푢|+푐)w = 1.5
and max CFL(|푢| + 푐)al = 3.0. Considering these parameters, we conducted also a grid convergence study, illustrated in Fig. 17.
As expected, a coarse grid leads to smoother results, but with grid reﬁnement, the solution converges toward the reference one.
After the preliminary veriﬁcation of the results, we use this test to illustrate now the eﬀects of ﬁnite relaxation parameters.
In the previous tests, we have seen how ﬁnite, large values can successfully replicate the mechanical equilibrium. Intuitively,
smaller relaxation parameters leave a higher level of dis-equilibrium between phases. This behavior is conﬁrmed by Fig. 18,
which displays the results obtained with diﬀerent sets of relaxation parameters:
• the strong set with 휆= 109 kg∕(m3s) and 휇= 105 m s∕kg used also in the previous tests,
• an intermediate set with 휆= 108 kg∕(m3s) and 휇= 104 m s∕kg,
• a mild set with 휆= 107 kg∕(m3s) and 휇= 103 m s∕kg.
These tests are performed over a grid with Δ푥= 0.001 m at acoustic CFLs higher than one. From Fig 18, we note a substantial
dis-equilibrium in the phasic velocity, in particular while using the mild set, but it can be observed also for the intermediate set.
To have a quantitative idea, the maximum velocity diﬀerence in these sets is
푖푛푡푒푟푚푒푑푖푎푡푒∶max(푢w −푢al) = 15.8 m∕s,
푚푖푙푑∶max(푢w −푢al) = 78.4 m∕s .
Conversely, no disequilibrium in the pressure can be distinguished, but the eﬀect of decreasing relaxation parameters is a smooth-
ing of the expansion and compression waves, which causes the disappearance of the intermediate uniform regions between them
(that is the regions called left* and right* in the mechanical equilibrium solution in Tab. 6). This behavior is reﬂected also by
the volume fraction proﬁle, where this smoothing diminishes also the degree of the mixing, as with the mild set the water vol-
ume fraction 훼w reaches only a maximum of 0.519 after the rarefaction and a minimum of 0.475 after the compression, instead
of max(훼w) = 0.522 and min(훼w) = 0.470 as in other cases.
Finally, we better investigate the eﬀects of pressure relaxation only, without relaxing the velocities (i.e., using 휆= 0). Figure 19
compares the results obtained without pressure relaxation (so full disequilibrium, considering the hyperbolic operator only), with
a weak relaxation parameter 휇= 10−1 m s∕kg and with a strong one 휇= 105 m s∕kg. To allow for the faster waves in the ﬁrst


32
B. Re and R. Abgrall
FIGURE 16 Water-aluminum mixture test, with 휆= 109 kg∕(m3s) and 휇= 105 m s∕kg: results at 푡퐹= 111 μs obtained in
two simulations characterized by a diﬀerent number of time steps 푁푡, over a uniform mesh with 푁푥= 1000 cells. The choice
푁푡= 1000 corresponds to max CFL(|푢| + 푐) of 0.3 for water and of 0.6 for aluminum, while 푁푡= 200 leads to 1.5 for water
and 3.0 for aluminum. The results are compared with the analytical solution of Kapila’s model given in Tab. 6. The top-left
panel shows how the initially uniform volume fractions change due to the relaxation processes. The top-right panel displays the
phasic densities: please, note the diﬀerent scales for water (left axis) and aluminum (right axis). In the bottom line, we have
the pressure and the velocity of each phase: for each simulation, it is impossible to distinguish between water and aluminum
as the relaxation processes drive them toward the equilibrium. In these plots, the reference pressure and velocity are only one,
displayed as solid line.
FIGURE 17 Water-aluminum mixture test, with 휆= 109 kg∕(m3s) and 휇= 105 m s∕kg: grid convergence study at 푡퐹= 111 μs
considering max CFL(|푢| + 푐w) = 0.3 and max CFL(|푢| + 푐al) = 0.6. The results of three simulations over uniform grids with
푁푥= {500, 2000, 8000} cells are displayed, along with the reference solution of Tab. 6. The variables are the water volume
fraction (on the left) and the mixture density ̄휌= 훼휌w + 훼휌al (on the right).
test, we consider a longer domain, Ω = [−0.8, 0.8] m, but the same grid spacing and time step as before. The ﬁrst observation
we can make from Fig. 19 is that the weak parameter is suﬃcient to drive the phasic pressures to the equilibrium, as the lines
for water and aluminum appear to be overlapped also in the zoomed view. These lines overlap also to the ones corresponding
to the strong relaxation and the Kapila’s model, conﬁrming that the pressure equilibrium is achieved in both tests. Similar
observations between the two diﬀerent relaxation parameters can be drawn also for the volume fraction and the velocity ﬁelds,


B. Re and R. Abgrall
33
FIGURE 18 Water-aluminum mixture test: eﬀects of diﬀerent ﬁnite relaxation parameters. Results are computed at 푡퐹= 111 μs,
using 1000 cells and 푁푡= 200 time steps (i.e., approximately max CFL(|푢|+푐w) = 1.5 and max CFL(|푢|+푐al) = 3.0 in all tests).
The results obtained with three diﬀerent sets of relaxation parameters 휆and 휇are displayed, along with the reference solution
of Tab. 6. The units of the 휆and 휇(omitted in the legend for brevity) are, respectively, kg∕(m3s) and m s∕kg. On the top-left,
only the volume fraction of water is displayed for clarity. On the bottom-left, the pressure proﬁles of water and aluminum of
each single test are indistinguishable, but they diﬀer among tests. On the top-right, a zoomed view of the velocities near and
after the shock is displayed to highlight the dis-equilibrium.
which, conversely, diﬀer signiﬁcant from the Kapila’s model because of the velocity dis-equilibrium. Furthermore, comparing
the results with and without pressure relaxation, we can observe that wave speeds in the relaxed condition are similar to the wave
speeds in non-equilibrium water, that is much slower than the aluminum ones. The equilibrium pressure is also pretty similar to
the non-equilibrium water, whereas the equilibrium velocity is an intermediate value between the ones of water and aluminum.
7.2
Water-air mixture test with pressure relaxation
In this section, we reconsider the Smooth shock tube test case proposed in19, but diﬀerently from Sec. 6.5, we use now pressure
relaxation, and we compare our results again with the ones shown in19 under stiﬀpressure relaxation. The initial conditions and
the stiﬀened gas parameters of air and water are those given in Sec. 6.5 and in Tab. 5. In summary: 휌w(푥) = 1050 kg∕m3 and
휌a(푥) = 1.2 kg∕m3, with
헅햾햿헍(푥< 0) ∶푃= 106 Pa , 훼w = 0.3 ;
헋헂헀헁헍(푥> 0) ∶푃= 105 Pa , 훼w = 0.7 .
The domain Ω = [−0.6, 0.6] m is divided in 600 primary cells. Figure 20 shows the results computed at 푡퐹= 350 μs, with
two diﬀerent time steps—the larger one resulting in an acoustic CFL of about 1.85 for both phases—and a relaxation parameter
휇= 105 m s∕kg. The agreement with the reference results is not excellent, probably because the maximum Mach number for air
is higher than 1.2, so our non-conservative scheme is not able to correctly capture the shock speed and this inaccuracy aﬀects
also the pressure proﬁles, where we have a diﬀerence of about 0.3 bar (i.e., 4%) after the rarefaction and of 0.2 bar (i.e., 3%)
after the shock. We also remind that the model used in19 is not symmetric and it is based on diﬀerent modeling hypotheses for
the water and we are not able to estimate whether and how this variation in the models may justify the discrepancy in the results.
Nevertheless, the general behavior of the pressure and velocity of both phases is consistent with the reference results, and this
serves as a conﬁrmation of the correctness of the pressure relaxation scheme.
To better investigate the eﬀect of the ﬁnite pressure relaxation parameter, we have re-run this test considering the largest
time step and diﬀerent values of 휇. Figure 21 compares the pressure proﬁles and the diﬀerences in the phasic pressure obtained
with two values of 휇. The value 휇= 105 m s∕kg is large enough to drive the phasic pressures toward the equilibrium, and
the phase diﬀerence is null everywhere. On the other hand, the small value 휇= 10−1 m s∕kg allows a certain degree of phase


34
B. Re and R. Abgrall
FIGURE 19 Water-aluminum mixture test: eﬀects of pressure relaxation, without velocity relaxation (휆= 0). Results are
computed at 푡퐹= 111 μs, using 1600 cells and 푁푡= 200 time steps (i.e., approximately max CFL(|푢| + 푐w) = 1.5 and
max CFL(|푢| + 푐al) = 3.0 in all tests). A ﬁrst test is run without pressure relaxation (휇= 0), then two tests are run using
휇∈{10−1, 105} m s∕kg (units are omitted in the legend for brevity). The solution of Tab. 6 is also displayed for comparison
with previous ﬁgures, but here it is labeled as “Kapila’s model” as it is not a reference solution for this non-equilibrium test.
Markers distinguish lines which, especially for pressure, overlap between water and aluminum (ﬁlled vs. empty symbols) and
between the two tests with pressure relaxation (diﬀerentiated by yellow squares vs blue downward-pointing triangles). A zoomed
view of the pressure ﬁeld close to the rarefaction end is shown in the top-left corner.
disequilibrium, especially across the shock, where it reaches the maximum (206.7 Pa), and across the material discontinuity,
where there is also a sign change. Indeed, due to the smearing of material discontinuity, the water on its left undergoes an
expansion (the increase of the volume fraction has an eﬀect similar to an expanding nozzle) and the water on its right is slightly
compressed, and the weak pressure relaxation is not suﬃcient to overcome this phenomenon.
7.3
Two-phase water expansion tube
The benchmark presented in this section was originally proposed in22 and it has been investigated also in25,87, with and without
phase transition. Here, we consider it without phase transition. This test involves a tube ﬁlled with liquid water at pressure
푃= 1 bar and density 휌liq = 1150 kg∕m3, to which a weak volume fraction of vapor 훼vap = 0.01 is added. As in87, we compute
the vapor initial conditions from the pressure and temperature of the liquid, considering the stiﬀened gas parameters given in
Tab. 7. An initial velocity discontinuity is located at the center of the tube (푥= 0): the left velocity is −2 m∕s, the right one is
2 m∕s.
The solution is computed at time 푡퐹= 3.2 ms and it consists of two symmetric rarefaction waves moving outwards. At the
center of the domain, the volume fraction increases because of the vapor mechanical expansion and a gas pocket is dynamically
generated22. As for the water-aluminum test in Sec. 7.1, a reference solution is computed in the limit of stiﬀmechanic relaxation
according to the Riemann solver proposed by Petitpas et al107. The numerical results computed using a grid spacing Δ푥= 10−4 m
are shown in Fig. 22. First, we compute the solution using 푁푇= 160000 time steps, corresponding to a maximal acoustic CFL
of 0.3 (for the liquid). With this choice, we achieve a good match with the reference solution, especially in terms of pressure,
capturing correctly the constant region between the rarefaction waves, while a little overshoot aﬀects both the density and the
volume fraction, but this behavior has been exhibited also in the previous works. Considering the small velocities and the severe
rarefactions involved in this test, the capability of the proposed low-Mach scheme to correctly compute the solution at mild CFL
numbers is a notable result. Remarkably, we perform a second test enforcing a 100 times bigger time step, leading to maximal
acoustic CFL number of 28.6 for the liquid and 15.8 for the vapor. As we can see from Fig. 22, the dissipation at this level
prevents to fully capture the intermediate state, being all proﬁles smeared with respect to the reference ones, but the general


B. Re and R. Abgrall
35
FIGURE 20 Water-air mixture test with pressure relaxation (휇= 105 m s∕kg): results at the time 푡퐹= 350 μs, over a uniform
grid with 600 cells, in absence of velocity relaxation (휆= 0). Two diﬀerent time steps are considered: the smallest one (resulting
from 푁푇= 500 steps, and displayed in blue and yellow solid lines) corresponds to max CFL(|푢|+푐) ≈0.55 for both phases; the
largest one (resulting from 푁푇= 150 steps, and displayed in orange and violet dashed lines) corresponds to max CFL(|푢|+푐) ≈
1.85 for both phases. In the left column, we see the water volume fraction and the pressures, which overlap between phases
because of relaxation; in the right column, we have the partial densities 훼휌and the velocity of each phase (note the diﬀerent
scales: for water on left axis, for air on the right axis). Thin, black lines display the pressure, the velocity and the volume fraction
in the limit of stiﬀpressure relaxation given by Saurel et al.19.
FIGURE 21 Water-air mixture test with diﬀerent pressure relaxation parameters: results at the time 푡퐹= 350 μs, over a uniform
grid with 600 cells, in absence of velocity relaxation (휆= 0), considering 푁푇= 150 time steps. The compared 휇values are 105
(blue) and 10−1 m s∕kg (orange). On the left, the water and air pressure proﬁles (in bar) are displayed, but it is impossible to
distinguish any diﬀerence. On the right, the pressure disequilibrium between water and air (in Pa) is plotted over the domain.
behavior of the ﬂuids is well captured. By comparison, Zein et al.25 used a CFL equal to 0.03 to obtain a stable solution, while
Han et al.87 used a maximum CFL of 0.9.
7.4
Almost pure ﬂuids test
Here, we present a test involving almost pure ﬂuids, water and air, with material parameters given in Tab. 5. The conﬁguration
involves a shock-tube with a left chamber (푥< 0) ﬁlled with air at 푃퐿= 100 bar and the right chamber (푥> 0) ﬁlled with water
at 푃푅= 50 bar. Since the proposed method is not able to deal with null volume fractions, we deﬁne the air volume fraction in
the left chamber as 훼a = 1−휀and the water one in the right chamber as 훼w = 1−휀, where 휀= 10−4. The air and water densities


36
B. Re and R. Abgrall
TABLE 7 Stiﬀened gas parameters for the two-phase water expansion tube, presented in Sec. 7.3, as given in22.
훾[−]
푃∞[Pa]
푐푣[J∕kg K]
푞[J∕kg]
Liquid water:
2.35
109
1816
−1167 ⋅103
Vapor water:
1.43
0
1040
2030 ⋅103
FIGURE 22 Water expansion tube test: results at the time 푡퐹= 3.2 ms, over a uniform grid with 20000 cells, with pressure and
velocity relaxation (휆= 107 kg∕(m3s) and 휇= 105 m s∕kg). The results obtained with 푁푇= 160000 and with 푁푇= 1600 time
steps are displayed, along with the reference solution, computed assuming mechanical equilibrium between phases according
to107. The displayed variables are: on the top, vapor volume fraction (initially, 훼vap = 0.01 everywhere) and mixture density
̄휌= 훼휌liq + 훼휌vap (it follows the color of Vapor in the legend); on the bottom, pressure and velocity of each phases (Water in the
legend refers to liquid water).
are 휌a = 100 kg∕m3 and 휌w = 1000 kg∕m3, respectively. Starting from a state of rest, the almost pure liquid on the right is set
into motion by the almost pure gas at higher pressure on the left. Considering pure ﬂuids separated by a material discontinuity,
it is possible to compute the analytical solution according to the Euler equations. It comprises a left-traveling rarefaction for air
and a right-traveling shock wave for water, while the intermediate state is characterized by a pressure 푃⋆= 98.887 bar and a
velocity 푢⋆= 2.989 m∕s. This velocity conﬁrms the low-Mach regime, as the maximum Mach number is 0.008, and the shock
speed is 1636 m∕s (1.0025 times the pre-shock speed of sound).
The numerical results at 푡퐹= 0.8 ms, obtained with pressure and velocity relaxation, are displayed in Fig. 23 and they reach
an excellent agreement with the analytical, pure-ﬂuid solution, except the shock wave which is considerably smeared. Figure 23
shows also how the solution varies for higher values of 휀, that is for higher levels of mixing. The ﬁgure contains some zoomed
frames to highlight the details close to the most important ﬂow structures. The level of mixture given by 휀= 10−2, that is the
same as the one considered for the liquid-water expansion test in sec. 7.3, is already enough to signiﬁcantly depart from the
pure-ﬂuid behavior. Indeed, the intermediate velocity is higher, resulting in a faster material discontinuity and a slower shock.
On the other hand, 휀= 10−3 is already suﬃcient to capture qualitatively the behavior of the pure ﬂuids, but it leads to some
discrepancies in the values of the intermediate state. These are overcome by using the value 휀= 10−4.
To investigate better the capability to correctly capture the interface between almost pure ﬂuids, we repeat this simulation, for
a long time. We consider a longer domain, Ω = [−20, 60] m, with the same grid spacing (Δ푥= 10−3 m) and we compute the
solution at 푡퐹= 0.03 s, using the time steps Δ푡= 0.2 휇푠and Δ푡= 1.0 휇푠, corresponding to a maximal acoustic CFL for water
of 0.33 and 1.64, respectively. The results are displayed in Fig. 24. The ﬁnal position of the material interface is 푥⋆= 0.0897 m


B. Re and R. Abgrall
37
FIGURE 23 Almost pure ﬂuids test: results at the time 푡퐹= 0.8 ms, over a uniform grid with 2000 cells (Δ푥= 10−3 m),
considering 푁푡= 1200 time steps, and imposing pressure and velocity relaxation. On the top-left, the water volume fraction
is displayed, along with a detailed view close to the material discontinuity, where the dashed vertical line illustrates the initial
position of the material discontinuity. On the top-right, the density, or more precisely the mixture density ̄휌, is displayed, with a
zoom of the region close to the shock; the disagreement among the initial densities (clearly visible for the blue line at 푥= 1.4)
is due to the diﬀerent values of 휀, which lead to diﬀerent weights in the average of the densities. On the bottom line, we have
the pressure proﬁles, with a zoomed view of the rarefaction, and the velocities. In all frames, the solid black line displays the
pure-ﬂuid solution computed analytically.
and it is computed correctly, as it can be observed in the zoomed view in the top-right corner of Fig. 24. Similarly, also the
intermediate state and the rarefaction fan are in excellent agreement with the analytical solution. On the contrary, the position of
the shock is not computed accurately, because the non-conservativeness of the scheme introduces an error in the shock velocity,
which, after a long time, results visible in the position. However, the relative error in the shock position is about 1.0%, so it is
acceptable.
7.5
Two-phase carbon dioxide test in saturation conditions, with accurate EOS
In this ﬁnal section, we compare the results computed under the stiﬀened gas approximation with the ones computed using
a more accurate thermodynamic model for CO2, based on the Peng-Robinson (PG) EOS84. In particular, we take advantage
of the implementation of this EOS tailor-made for CO2 ﬂows, provided by an in-house thermodynamic library developed at
SINTEF Energy108, that exploits the concept of the corresponding states to enhance the accuracy of speciﬁc properties, such as
the density and the speed of sound, for the liquid phase.
The set-up in which this comparison is carried out consists in a pipe, 600 m long, ﬁlled with two-phase CO2 mixture at
saturation conditions, that is the initial conditions of the ﬂuids, represented in the thermodynamic plane, lie along the saturation,
or vapor-liquid equilibrium, curve.3 In the left part of the domain, the CO2 mixture is composed by 75% of liquid (훼liq = 0.75)
and 25% of vapor (훼vap = 0.25), at the saturation temperature 푇퐿= 260 K. In the right part, the composition is inverted and the
saturation temperature is 20 K higher, that is 훼liq = 0.25, 훼vap = 0.75 and 푇푅= 280 K. We remind that, at saturation conditions,
deﬁning the temperature unequivocally deﬁnes also the pressure and the density of each phase. Hence, in the test with the Peng-
Robinson EOS, no other inputs are deﬁned. Conversely, in the test with the stiﬀened gas model, as we do not have a saturation
3A representation of the initial conditions in the pressure-volume plane is given in Fig. 27; for illustrations of the saturation curve of CO2 with diﬀerent EOSs, see 109.


38
B. Re and R. Abgrall
FIGURE 24 Almost pure ﬂuids test: results at the time 푡퐹= 0.03 s, over a uniform grid with 80000 cells (Δ푥= 10−3 m),
with pressure and velocity relaxation. The results obtained with two time steps (leading to max(|푢| + 푐)w of 0.33 and 1.64) are
compared. On the top-right, the water volume fraction is displayed, along with an highlight of the displacement of the material
discontinuity (here, the grey dashed line illustrates its initial position, the solid lines the ﬁnal one). On the bottom, the pressure
and velocity are displayed and the ﬁnal position of the material discontinuity is shown by a yellow vertical line. On the top-
left corner, two zoomed views of the pressure ﬁelds in proximity of the rarefaction fan and of the shock wave are shown. In all
frames, the solid black line displays the pure-ﬂuid solution computed analytically.
model, we do need to specify also the initial pressure, for which we use the saturated values computed according to the Peng-
Robinson EOS, that are 푃퐿= 23.98 bar and 푃푅= 41.5 bar. The parameters of the stiﬀened gas model for CO2 are taken from110
and reported in Tab. 8.
We compute the ﬂow ﬁeld for 푡퐹= 1 s from the moment the diaphragm separating the two mixtures, initially at rest, is
removed. We consider 푁푇= 2000 time steps and a grid spacing Δ푥= 0.1 m, which result in max 퐶퐹퐿(|푢| + 푐)liq = 2.3 and
max 퐶퐹퐿(|푢|+푐)vap = 1.2. Higher values of CFL could be enforced without preventing stability, but we found that these values
leads to a good compromise between accuracy and eﬃciency. Figure 25 illustrates the proﬁles of the liquid volume fraction,
the pressure and the velocity, which are driven toward the equilibrium by relaxation processes with 휆= 107 kg∕(m3s) and
휇= 103 m s∕kg. The diﬀerent thermodynamic models lead to a diﬀerence in the minimum velocity reached in the intermediate
region between the shock and the rarefaction wave. This is reﬂected on the position of the material discontinuity: as highlighted
by the zoomed view on the top-right box in Fig. 25, the stiﬀened gas predicts a larger displacement (in the negative 푥-direction)
of the contact discontinuity due to faster ﬂuids velocities. Quantitatively, the position of the contact discontinuity is −11 m using
the Peng-Robinson EOS, while it is −11.3 m using the stiﬀened gas.
The diﬀerences introduced by the thermodynamic models are clearer in the density proﬁles, displayed in Fig. 26. Given as
input the same temperatures and pressures, the initial conditions entail already discrepancies in the density:
for liquid: 휌PG
퐿,liq > 휌Stiff
퐿,liq
and
휌PG
푅,liq < 휌Stiff
푅,liq ;
for vapor: 휌PG
퐿,vap < 휌Stiff
퐿,vap
and
휌PG
푅,vap < 휌Stiff
푅,vap .
Moreover, Peng-Robinson EOS predicts stronger initial density jumps for both phases. This leads to a notable disagreement
in the behavior of the CO2 vapor after the diaphragm rupture, when the two thermodynamic models predict opposite (in sign)
density jumps across the material discontinuity.
Finally, we plot the results obtained with the Peng-Robinson EOS in the pressure-volume thermodynamic plane, in Fig. 27.
This illustration gives a clear idea of how the proposed full non-equilibrium BN-type model allows each phase to evolve indepen-
dently according to its own thermodynamic model, although the phasic pressures (and velocity) are immediately driven toward


B. Re and R. Abgrall
39
TABLE 8 Stiﬀened gas parameters for the carbon dioxide tests presented in Sec. 7.5, as given in110
훾[−]
푃∞[Pa]
푐푣[J∕kg K]
푞[J∕kg]
Liquid CO2:
1.23
1.32 ⋅108
2440
−6.23 ⋅105
Vapor CO2:
1.06
8.86 ⋅105
2410
−3.01 ⋅105
FIGURE 25 Two-phase CO2 test: results at the time 푡퐹= 1.0 s, over a uniform grid with 6000 cells (Δ푥= 0.1 m), considering
푁푡= 2000 time steps. Initially, the left (푥< 0) and right (푥> 0) parts of the tube contain a saturated mixture with diﬀerent
composition and at diﬀerent temperatures. The results obtained with the Peng-Robinson EOS (blue and triangular markers, PG
EOS in the legend) are compared to the ones obtained with the stiﬀened gas model (orange and square markers, StiﬀGas in the
legend). On the top, the liquid volume fraction is displayed, with a detailed view close to the material discontinuity in the box
on the right. On the bottom, the pressure and the velocities are displayed: the liquid (full markers) and vapor (empty markers)
CO2 are driven to the equilibrium by means of relaxation.
the equilibrium. The resulting ﬂow states lie in close proximity to one side (liquid on the left, vapor on the right) of the dome,
and only the evolution of the mixture density develops in the fully two-phase region. However, no special treatment, such as the
deﬁnition of a speed of sound for the mixture, is required for this situation.
8
SUMMARY AND CONCLUSIONS
Starting from the symmetric variant proposed by Saurel and Abgrall11, we derived a pressure-based BN-type model for weakly
compressible two-phase ﬂows, which, as illustrated by Remark 1, encompasses the divergence-free condition for multiphase
ﬂows in the zero-Mach limit, recovering the correct scaling of the pressure. Although the single components of the ﬁnite-volume
solver we built to solve the 1D pressure-based BN-type model are not novel by themselves, the resulting numerical method is
unique and includes notable features. For instance, it preserves by construction the non-disturbance condition on pressure and
velocity, it overcomes the stringent limitation on the time step imposed by the acoustics, it allows for diﬀerent thermodynamic
models, and it provides a wide variety of modeling choices thanks to the relaxation terms with ﬁnite parameters, which can be
used to control whether and how the phasic pressure and/or velocity are driven toward the equilibrium.
The main drawback of the proposed method relates to the non-conservative character of the pressure-based formulation,
which prevents to resolve accurately shock waves. However, this behavior was expected and numerical tests showed that, for
moderate intensities, the error in the position of the shock is between 1 and 2%, consistently with the predictions made by
Karni36. Notwithstanding the acceptable impact on weakly compressible ﬂows, a remedy for this limitation is imperative to


40
B. Re and R. Abgrall
FIGURE 26 Two-phase CO2 test: comparison of the density (휌liq and 휌vap) at the initial time (black dashed and dotted lines)
and at 푡퐹= 1.0 s (colored lines with markers) computed with Peng-Robinson EOS (black dashed; blue and yellows lines with
triangular markers) and with stiﬀened gas model (black dotted; orange and violet lines with square markers). The upper part of
the plot refers to the liquid CO2, while the bottom part refers to the vapor CO2. Note that the 푦-axis is discontinuous, but the
tick spacing is preserved.
FIGURE 27 Two-phase CO2 test: visualization of the ﬂow ﬁeld computed using the Peng-Robinson EOS in the thermodynamic
plane pressure–speciﬁc volume. The saturation curve is plotted on the background with a grey line. The initial states for the
liquid (blue) and vapor (orange) phase, as well as the CO2 mixture (yellow), are displayed by full triangles: left-pointing triangles
refer to the left state (saturated temperature 푇퐿= 260 K) and right-pointing ones refer to the right state (saturated temperature
푇푅= 280 K). The results at 푡퐹= 1.0 s are displayed by lines. The evolution of the mixture (dashed line) is plotted as 1∕̄휌versus
the mixture pressure ̄푃= 훼푃liq + 훼푃vap, where however the phasic pressures 푃liq and 푃vap are equal thanks to relaxation.
move toward an all-speed scheme, so this matter will be given a high priority in the future development of the method. To
mitigate the eﬀects of non-conservativeness, we will consider strategies already proposed for single-phase ﬂows—for instance,
an additional scheme-dependent viscous term that provides a correction at the leading order of the discrete approximation111,
or the use of the pressure equation only as a prediction for variables in the total energy equation54—and for reduced two-phase
ﬂows, e.g., considering a supplementary, explicit correction step60 or moving to residual distribution schemes61.
As told in the introduction, this work describes the ﬁrst stage of a longer project, which aims to develop a reliable and robust
tool for multi-phase simulations of weakly-compressible ﬂows. To reach this aim, the numerical scheme underlying the solution
strategy described in this work should be enhanced: higher-order discretization techniques will be considered, in addition to the
previously mentioned correction for non-conservativeness. Afterwards, we will extend the proposed numerical tool to multi-
dimensional grids. In particular, to account for complex geometries, we will look at unstructured grids, for which a face-based


B. Re and R. Abgrall
41
staggered discretization has been recently proposed by Bermúdez et al.52 in the framework of weakly compressible single-phase
ﬂows.
In conclusion, we propose here the ﬁrst numerical tool based on the BN-type model that includes simultaneously all the
features outlined in the ﬁrst paragraph of this conclusive section. As identiﬁed above, the proposed method is not free from
defects, but the main limitations could be overcome by adapting the solution strategy to include some of the techniques already
proposed in the literature for other models. Given these points, we think this work could pave the way for a wider application
of BN-type models, especially to investigate two-phase ﬂows involving liquid components, which generally exhibit low Mach
number and require speciﬁc, accurate thermodynamic models. The ingredients of the solution strategy, taken individually, are
not complex, so the main ideas presented in this work could be applicable also in software used in applied research and industry,
where simplicity and eﬃciency are of utmost importance. Similarly, the modular nature of the presented method could facilitate
its integration in other BN solvers, mainly used in academic research, so far. Furthermore, the use of ﬁnite relaxation parameters
may enhance the modeling capabilities of BN–type models. Indeed, if experimental data were available, parametric analyses
could be carried out to deﬁne the best values (or range of values) for each multiphase regimes, contributing to increase the
ﬁdelity of numerical results.
ACKNOWLEDGEMENT
This publication has been produced with support from the NCCS Centre, performed under the Norwegian research program Cen-
tres for Environment-friendly Energy Research (FME). The authors acknowledge the following partners for their contributions:
Aker Solutions, Ansaldo Energia, CoorsTek Membrane Sciences, EMGS, Equinor, Gassco, Krohne, Larvik Shipping, Norcem,
Norwegian Oil and Gas, Quad Geometrics, Shell, Total, Vår Energi, and the Research Council of Norway (257579/E20).
This work was initiated while B. Re was a post-doctoral researcher at University of Zürich (UZH). The author gratefully
acknowledges the ﬁnancial support received under the grant Forschungskredit of the University of Zurich, grant no. [FK-20-121].
The authors would like to thank Svend Tollak Munkejord and Morten Hammer (SINTEF, Norway) for the constructive
discussions about multiphase CO2 ﬂows and the help with the implementation of the thermodynamic library.
References
1. Barlow AJ, Maire PH, Rider WJ, Rieben RN, Shashkov MJ. Arbitrary Lagrangian–Eulerian methods for modeling high-
speed compressible multimaterial ﬂows. J. Comput. Phys. 2016; 322: 603–665. doi: 10.1016/j.jcp.2016.07.001
2. Dumbser M, Boscheri W. High-order unstructured Lagrangian one-step WENO ﬁnite volume schemes for non-
conservative hyperbolic systems: Applications to compressible multi-phase ﬂows. Comput. Fluids 2013; 86: 405–432.
doi: 10.1016/j.compﬂuid.2013.07.024
3. Glimm J, Graham M, Grove J, et al. Front tracking in two and three dimensions. Comput. Math. with Appl. 1998; 35(7):
1–11. doi: 10.1016/S0898-1221(98)00028-5
4. Tryggvason G, Bunner B, Esmaeeli A, et al. A Front-Tracking Method for the Computations of Multiphase Flow. J.
Comput. Phys. 2001; 169(2): 708–759. doi: 10.1006/jcph.2001.6726
5. Sethian JA, Smereka P. Level Set Methods for Fluid Interfaces. Annu. Rev. Fluid Mech. 2003; 35(1): 341–372. doi:
10.1146/annurev.ﬂuid.35.101101.161105
6. Saye RI, Sethian JA. A review of level set methods to model interfaces moving under complex physics: Recent challenges
and advances. Handb. Numer. Anal. 2020; 21: 509–554. doi: 10.1016/BS.HNA.2019.07.003
7. Fedkiw RP, Aslam T, Merriman B, Osher S. A Non-oscillatory Eulerian Approach to Interfaces in Multimaterial Flows
(the Ghost Fluid Method). J. Comput. Phys. 1999; 152(2): 457–492. doi: 10.1006/jcph.1999.6236
8. Owkes M, Desjardins O. A computational framework for conservative, three-dimensional, unsplit, geometric trans-
port with application to the volume-of-ﬂuid (VOF) method. J. Comput. Phys. 2014; 270: 587–612.
doi:
10.1016/j.jcp.2014.04.022


42
B. Re and R. Abgrall
9. Saurel R, Pantano C. Diﬀuse-Interface Capturing Methods for Compressible Two-Phase Flows. Annu. Rev. Fluid Mech.
2018; 50: 105–130. doi: 10.1146/annurev-ﬂuid-122316-050109
10. Hirt C, Nichols B. Volume of ﬂuid (VOF) method for the dynamics of free boundaries. J. Comput. Phys. 1981; 39(1):
201–225. doi: 10.1016/0021-9991(81)90145-5
11. Saurel R, Abgrall R. A Multiphase Godunov Method for Compressible Multiﬂuid and Multiphase Flows. J. Comput. Phys.
1999; 150(2): 425–467. doi: 10.1006/JCPH.1999.6187
12. Kapila AK, MenikoﬀR, Bdzil JB, Son SF, Stewart DS. Two-phase modeling of deﬂagration-to-detonation transition in
granular materials: Reduced equations. Phys. Fluids 2001; 13(10): 3002–3024. doi: 10.1063/1.1398042
13. Drew DA, Passman SL. Theory of Multicomponent Fluids. 135 of Applied Mathematical Sciences.
New York, NY:
Springer New York . 1999
14. Baer MR, Nunziato JW. A two-phase mixture theory for the Deﬂagration-to-Detonation Transition (DDT) in reactive
granular materials. Int. J. Multiphase Flow 1986; 6: 861-889. doi: 10.1016/0301-9322(86)90033-9
15. Romenski E, Resnyansky AD, Toro EF. Conservative hyperbolic formulation for compressible two-phase ﬂow with
diﬀerent phase pressures and temperatures. Q. Appl. Math. 2007; 65(2): 259–279. doi: 10.1090/S0033-569X-07-01051-2
16. Ambroso A, Chalons C, Raviart PA. A Godunov-type method for the seven-equation model of compressible two-phase
ﬂow. Comput. Fluids 2012; 54(1): 67–91. doi: 10.1016/j.compﬂuid.2011.10.004
17. Saurel R, Le Martelot S, Tosello R, Lapebie E. Symmetric model of compressible granular mixtures with permeable
interfaces. Phys. Fluids 2014; 26(12): 123304. doi: 10.1063/1.4903259
18. Müller S, Hantke M, Richter P. Closure conditions for non-equilibrium multi-component models. Contin. Mech. Thermo-
dyn. 2016; 28(4): 1157–1189. doi: 10.1007/s00161-015-0468-8
19. Saurel R, Chinnayya A, Carmouze Q. Modelling compressible dense and dilute two-phase ﬂows. Phys. Fluids 2017; 29(6):
1–18. doi: 10.1063/1.4985289
20. Allaire G, Clerc S, Kokh S. A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids. J.
Comput. Phys. 2002; 181(2): 577–616. doi: 10.1006/jcph.2002.7143
21. Murrone A, Guillard H. A ﬁve equation reduced model for compressible two phase ﬂow problems. J. Comput. Phys. 2005;
202(2): 664–698. doi: 10.1016/J.JCP.2004.07.019
22. Saurel R, Petitpas F, Abgrall R. Modelling phase transition in metastable liquids: Application to cavitating and ﬂashing
ﬂows. J. Fluid Mech. 2008; 607(2008): 313–350. doi: 10.1017/S0022112008002061
23. Kreeft JJ, Koren B. A new formulation of Kapila’s ﬁve-equation model for compressible two-ﬂuid ﬂow, and its numerical
treatment. J. Comput. Phys. 2010; 229(18): 6220–6242. doi: 10.1016/J.JCP.2010.04.025
24. Saurel R, Petitpas F, Berry RA. Simple and eﬃcient relaxation methods for interfaces separating compressible
ﬂuids, cavitating ﬂows and shocks in multiphase mixtures. J. Comput. Phys. 2009; 228(5): 1678–1712.
doi:
10.1016/J.JCP.2008.11.002
25. Zein A, Hantke M, Warnecke G. Modeling phase transition for compressible two-phase ﬂows applied to metastable liquids.
J. Comput. Phys. 2010; 229(8): 2964–2998. doi: 10.1016/j.jcp.2009.12.026
26. Pelanti M, Shyue KM. A mixture-energy-consistent six-equation two-phase numerical model for ﬂuids with interfaces,
cavitation and evaporation waves. J. Comput. Phys. 2014; 259: 331–357. doi: 10.1016/j.jcp.2013.12.003
27. Pelanti M, Shyue KM. A numerical model for multiphase liquid–vapor–gas ﬂows with interfaces and cavitation. Int. J.
Multiph. Flow 2019; 113: 208–230. doi: 10.1016/j.ijmultiphaseﬂow.2019.01.010
28. Ishii M. Thermo-ﬂuid Dynamic Theory of Two-phase Flow. Collection de la Direction des études et recherches d’Électricité
de FranceEyrolles . 1975.


B. Re and R. Abgrall
43
29. Staedtke H, Franchello G, Worth B, et al. Advanced three-dimensional two-phase ﬂow simulation tools for application to
reactor safety (ASTAR). Nucl. Eng. Des. 2005; 235(2-4): 379–400. doi: 10.1016/j.nucengdes.2004.08.052
30. Stewart HB, WendroﬀB. Two-phase ﬂow: Models and methods. J. Comput. Phys. 1984; 56(3): 363–409.
doi:
10.1016/0021-9991(84)90103-7
31. Hantke M, Müller S, Grabowsky L. News on Baer–Nunziato-type model at pressure equilibrium. Contin. Mech.
Thermodyn. 2021; 33(3): 767–788. doi: 10.1007/s00161-020-00956-3
32. Lund H. A Hierarchy of Relaxation Models for Two-Phase Flow. SIAM J. Appl. Math. 2012; 72(6): 1713–1741. doi:
10.1137/12086368X
33. Guillard H, Viozat C. On the behaviour of upwind schemes in the low Mach number limit. Comput. Fluids 1999; 28(1):
63–86. doi: 10.1016/S0045-7930(98)00017-6
34. Guillard H, Murrone A. On the behavior of upwind schemes in the low Mach number limit: II. Godunov type schemes.
Comput. Fluids 2004; 33(4): 655–675. doi: 10.1016/j.compﬂuid.2003.07.001
35. Dellacherie S, Jung J, Omnes P, Raviart PA. Construction of modiﬁed Godunov-type schemes accurate at any
Mach number for the compressible Euler system. Math. Model. Methods Appl. Sci. 2016; 26(13): 2525–2615.
doi:
10.1142/S0218202516500603
36. Karni S. Hybrid Multiﬂuid Algorithms. SIAM J. Sci. Comput. 1996; 17(5): 1019–1039. doi: 10.1137/S106482759528003X
37. Lv Y. Development of a nonconservative discontinuous Galerkin formulation for simulations of unsteady and turbulent
ﬂows. Int. J. Numer. Methods Fluids 2020; 92(5): 325–346. doi: 10.1002/ﬂd.4785
38. Bijl H, Wesseling P. A Uniﬁed Method for Computing Incompressible and Compressible Flows in Boundary-Fitted
Coordinates. J. Comput. Phys. 1998; 141(2): 153–173. doi: 10.1006/JCPH.1998.5914
39. Kawai S, Terashima H, Negishi H. A robust and accurate numerical method for transcritical turbulent ﬂows at supercritical
pressure with an arbitrary equation of state. J. Comput. Phys. 2015; 300: 116–135. doi: 10.1016/j.jcp.2015.07.047
40. Karki KC, Patankar SV. Pressure based calculation procedure for viscous ﬂows at all speeds in arbitrary conﬁgurations.
AIAA J. 1989; 27(9): 1167–1174. doi: 10.2514/3.10242
41. Demirdžić I, Muzaferija S. Numerical method for coupled ﬂuid ﬂow, heat transfer and stress analysis using unstructured
moving meshes with cells of arbitrary topology. Comput. Methods Appl. Mech. Eng. 1995; 125(1-4): 235–255. doi:
10.1016/0045-7825(95)00800-G
42. Patankar S, Spalding D. A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic
ﬂows. Int. J. Heat Mass Transf. 1972; 15(10): 1787–1806. doi: 10.1016/0017-9310(72)90054-3
43. Kim J, Moin P. Application of a fractional-step method to incompressible Navier-Stokes equations. J. Comput. Phys. 1985;
59(2): 308–323. doi: 10.1016/0021-9991(85)90148-2
44. Harlow FH, Welch JE. Numerical Calculation of Time-Dependent Viscous Incompressible Flow of Fluid with Free
Surface. Phys. Fluids 1965; 8(12): 2182. doi: 10.1063/1.1761178
45. Buﬀard T, Gallouët T, Hérard JM. A sequel to a rough Godunov scheme: Application to real gases. Comput. Fluids 2000;
29(7): 813–847. doi: 10.1016/S0045-7930(99)00026-2
46. Munz CD, Roller S, Klein R, Geratz KJ. The extension of incompressible ﬂow solvers to the weakly compressible regime.
Comput. Fluids 2003; 32(2): 173–196. doi: 10.1016/S0045-7930(02)00010-5
47. Xiao F. Uniﬁed formulation for compressible and incompressible ﬂows by using multi-integrated moments I: one-
dimensional inviscid compressible ﬂow. J. Comput. Phys. 2004; 195(2): 629–654. doi: 10.1016/j.jcp.2003.10.014
48. Park JH, Munz CD. Multiple pressure variables methods for ﬂuid ﬂow at all Mach numbers. International Journal for
Numerical Methods in Fluids 2005; 49: 905-931. doi: 10.1002/ﬂd.1032


44
B. Re and R. Abgrall
49. Degond P, Tang M. All speed scheme for the low mach number limit of the Isentropic Euler equations. Commun. Comput.
Phys. 2011; 10(1): 1–31. doi: 10.4208/cicp.210709.210610a
50. Xiao CN, Denner F, Wachem vBG. Fully-coupled pressure-based ﬁnite-volume framework for the simulation of ﬂuid ﬂows
at all speeds in complex geometries. J. Comput. Phys. 2017; 346: 91–130. doi: 10.1016/j.jcp.2017.06.009
51. Xie B, Deng X, Sun Z, Xiao F. A hybrid pressure–density-based Mach uniform algorithm for 2D Euler equations
on unstructured grids by using multi-moment ﬁnite volume method. J. Comput. Phys. 2017; 335: 637–663.
doi:
10.1016/j.jcp.2017.01.043
52. Bermúdez A, Busto S, Dumbser M, Ferrín J, Saavedra L, Vázquez-Cendón M. A staggered semi-implicit hybrid FV/FE
projection method for weakly compressible ﬂows. J. Comput. Phys. 2020; 421: 109743. doi: 10.1016/j.jcp.2020.109743
53. Busto S, Río-Martín L, Vázquez-Cendón M, Dumbser M. A semi-implicit hybrid ﬁnite volume/ﬁnite element scheme
for all Mach number ﬂows on staggered unstructured meshes. Appl. Math. Comput. 2021; 402: 126117.
doi:
10.1016/j.amc.2021.126117
54. Kwatra N, Su J, Grétarsson JT, Fedkiw R. A method for avoiding the acoustic time step restriction in compressible ﬂow.
J. Comput. Phys. 2009; 228(11): 4146–4161. doi: 10.1016/J.JCP.2009.02.027
55. Cordier F, Degond P, Kumbaro A. An Asymptotic-Preserving all-speed scheme for the Euler and Navier-Stokes equations.
J. Comput. Phys. 2012; 231(17): 5685–5704. doi: 10.1016/j.jcp.2012.04.025
56. Terashima H, Koshi M. Approach for simulating gas–liquid-like ﬂows under supercritical pressures using a high-order
central diﬀerencing scheme. J. Comput. Phys. 2012; 231(20): 6907–6923. doi: 10.1016/J.JCP.2012.06.021
57. Dumbser M, Casulli V. A conservative, weakly nonlinear semi-implicit ﬁnite volume scheme for the compressible Navier-
Stokes equations with general equation of state. Appl. Math. Comput. 2016; 272: 479–497. doi: 10.1016/j.amc.2015.08.042
58. Duret B, Canu R, Reveillon J, Demoulin FX. A pressure based method for vaporizing compressible two-phase ﬂows with
interface capturing approach. Int. J. Multiph. Flow 2018; 108: 42–50. doi: 10.1016/j.ijmultiphaseﬂow.2018.06.022
59. Denner F, Xiao CN, Wachem vBG. Pressure-based algorithm for compressible interfacial ﬂows with acoustically-
conservative interface discretisation. J. Comput. Phys. 2018; 367: 192–234. doi: 10.1016/j.jcp.2018.04.028
60. Zhang L, Kumbaro A, Ghidaglia JM. A conservative pressure based solver with collocated variables on unstructured grids
for two-ﬂuid ﬂows with phase change. J. Comput. Phys. 2019; 390: 265–289. doi: 10.1016/j.jcp.2019.04.007
61. Abgrall R, Bacigaluppi P, Tokareva S. A high-order nonconservative approach for hyperbolic equations in ﬂuid dynamics.
Comput. Fluids 2018; 169: 10–22. doi: 10.1016/j.compﬂuid.2017.08.019
62. Re B, Abgrall R. Non-equilibrium Model for Weakly Compressible Multi-component Flows: The Hyperbolic Operator.
2020: 33-45. doi: 10.1007/978-3-030-49626-5_3
63. Heul v. dDR, Vuik C, Wesseling P. A Conservative Pressure Correction Method for Flow at All Speeds. Comput. Fluids
2003; 32: 1113–1132. doi: 10.1016/S0045-7930(02)00086-5
64. Metz B, Davidson O, Coninck dH, Loos M, Meyer L., eds.IPCC, 2005: IPCC Special Report on Carbon Dioxide Capture
and Storage. Cambridge University Press, UK . 2005.
65. Munkejord ST, Hammer M, Løvseth SW. CO2transport: Data and models - A review. Appl. Energy 2016; 169: 499–523.
doi: 10.1016/j.apenergy.2016.01.100
66. Daru V, Le Quéré P, Duluc MC, Le Maître O. A numerical method for the simulation of low Mach number liquid-gas
ﬂows. J. Comput. Phys. 2010; 229(23): 8844–8867. doi: 10.1016/j.jcp.2010.08.013
67. LeMartelot S, Nkonga B, Saurel R. Liquid and liquid-gas ﬂows at all speeds. J. Comput. Phys. 2013; 255: 53–82. doi:
10.1016/j.jcp.2013.08.001


B. Re and R. Abgrall
45
68. Pelanti M. Low Mach number preconditioning techniques for Roe-type and HLLC-type methods for a two-phase
compressible ﬂow model. Appl. Math. Comput. 2017; 310: 112–133. doi: 10.1016/j.amc.2017.04.014
69. Abgrall R. How to Prevent Pressure Oscillations in Multicomponent Flow Calculations: A Quasi Conservative Approach.
J. Comput. Phys. 1996; 125(1): 150–160. doi: 10.1006/JCPH.1996.0085
70. Wall C, Pierce CD, Moin P. A semi-implicit method for resolution of acoustic waves in low Mach number ﬂows. J. Comput.
Phys. 2002; 181(2): 545–563. doi: 10.1006/jcph.2002.7141
71. Wenneker I, Segal A, Wesseling P. A Mach-uniform unstructured staggered grid method. Int. J. Numer. Methods Fluids
2002; 40(9): 1209–1235. doi: 10.1002/ﬂd.417
72. Perot B. Conservation Properties of Unstructured Staggered Mesh Schemes. J. Comput. Phys. 2000; 159(1): 58–89. doi:
10.1006/jcph.2000.6424
73. Zhang X, Schmidt D, Perot B. Accuracy and Conservation Properties of a Three-Dimensional Unstructured Staggered
Mesh Scheme for Fluid Dynamics. J. Comput. Phys. 2002; 175(2): 764–791. doi: 10.1006/jcph.2001.6973
74. Perot B, Nallapati R. A moving unstructured staggered mesh method for the simulation of incompressible free-surface
ﬂows. J. Comput. Phys. 2003; 184(1): 192–214. doi: 10.1016/S0021-9991(02)00027-X
75. Klein R. Semi-implicit extension of a godunov-type scheme based on low mach number asymptotics I: One-dimensional
ﬂow. J. Comput. Phys. 1995; 121(2): 213–237. doi: 10.1016/S0021-9991(95)90034-9
76. Ventosa-Molina J, Chiva J, Lehmkuhl O, Muela J, Pérez-Segarra CD, Oliva A. Numerical analysis of conservative
unstructured discretisations for low Mach ﬂows. Int. J. Numer. Methods Fluids 2017; 84(6): 309–334. doi: 10.1002/ﬂd.4350
77. Casulli V, Greenspan D. Pressure method for the numerical solution of transient, compressible ﬂuid ﬂows. Int. J. Numer.
Methods Fluids 1984; 4(11): 1001–1012. doi: 10.1002/ﬂd.1650041102
78. Boscarino S, Russo G, Scandurra L. All Mach Number Second Order Semi-implicit Scheme for the Euler Equations of
Gas Dynamics. J. Sci. Comput. 2018; 77: 850-884. doi: 10.1007/s10915-018-0731-9
79. Bermúdez A, Ferrín JL, Saavedra L, Vázquez-Cendón ME. A projection hybrid ﬁnite volume/element method for low-
Mach number ﬂows. J. Comput. Phys. 2014; 271: 360–378. doi: 10.1016/j.jcp.2013.09.029
80. Lentine M, Grétarsson JT, Fedkiw R. An unconditionally stable fully conservative semi-Lagrangian method. J. Comput.
Phys. 2011; 230(8): 2857–2879. doi: 10.1016/j.jcp.2010.12.036
81. Noelle S, Bispen G, Arun KR, Lukáčová-Medviďová M, Munz CD. A Weakly Asymptotic Preserving Low Mach
Number Scheme for the Euler Equations of Gas Dynamics. SIAM J. Sci. Comput. 2014; 36(6): B989–B1024.
doi:
10.1137/120895627
82. Harlow FH, Amsden AA. Fluid Dynamics. Tech. Rep. LA-4700, Los Alamos Scientiﬁc Lab.; Los Alamos: 1971.
83. MenikoﬀR, Plohr BJ. The {R}iemann problem for ﬂuid ﬂow of real materials. Rev. Mod. Phys. 1989; 61(1): 75–130. doi:
10.1103/RevModPhys.61.75
84. Peng DY, Robinson DB. A New Two-Constant Equation of State. Ind. Eng. Chem. Fundam. 1976; 15(1): 59–64. doi:
10.1021/i160057a011
85. Le Métayer O, Saurel R. The Noble-Abel Stiﬀened-Gas equation of state. Phys. Fluids 2016; 28(4): 046102.
doi:
10.1063/1.4945981
86. Rodio MG, Congedo PM, Abgrall R. Two-phase ﬂow numerical simulation with real-gas eﬀects and occurrence of
rarefaction shock waves. Eur. J. Mech. B/Fluids 2014; 45: 20–35. doi: 10.1016/j.euromechﬂu.2013.11.007
87. Han E, Hantke M, Müller S. Eﬃcient and robust relaxation procedures for multi-component mixtures including phase
transition. J. Comput. Phys. 2017; 338: 217–239. doi: 10.1016/j.jcp.2017.02.066


46
B. Re and R. Abgrall
88. Re B, Guardone A. An adaptive ALE scheme for non-ideal compressible ﬂuid dynamics over dynamic unstructured
meshes. Shock Waves 2019; 29(1): 73–99. doi: 10.1007/s00193-018-0840-2
89. Hérard JM, Hurisse O. A fractional step method to compute a class of compressible gas–liquid ﬂows. Comput. Fluids
2012; 55: 57–69. doi: 10.1016/j.compﬂuid.2011.11.001
90. Andrianov N, Warnecke G. The Riemann problem for the Baer-Nuziato two-phase ﬂow model. J. Comput. Phys. 2004;
195(2): 434–464. doi: 10.1016/j.jcp.2003.10.006
91. Schwendeman DW, Wahle CW, Kapila AK. The Riemann problem and a high-resolution Godunov method for a model of
compressible two-phase ﬂow. J. Comput. Phys. 2006; 212(2): 490–526. doi: 10.1016/j.jcp.2005.07.012
92. Deledicque V, Papalexandris MV. An exact Riemann solver for compressible two-phase ﬂow models containing non-
conservative products. J. Comput. Phys. 2007; 222(1): 217–245. doi: 10.1016/j.jcp.2006.07.025
93. Parés C. Numerical methods for nonconservative hyperbolic systems: a theoretical framework. SIAM J. Numer. Anal. 2006;
44(1): 300–321. doi: https://doi.org/10.1137/050628052
94. Abgrall R, Karni S. A comment on the computation of non-conservative products. J. Comput. Phys. 2010; 229(8): 2759–
2763. doi: 10.1016/j.jcp.2009.12.015
95. Toro EF, Siviglia A. PRICE: primitive centred schemes for hyperbolic systems. Int. J. Numer. Methods Fluids 2003;
42(12): 1263–1291. doi: 10.1002/ﬂd.491
96. Bermúdez A, Lopóz X, Vázquez-Cendón ME. Numerical solution of non-isothermal non-adiabatic ﬂow of real gases in
pipelines. J. Comput. Phys. 2016; 323: 126–148. doi: 10.1016/j.jcp.2016.07.020
97. Balay S, Abhyankar S, Adams M, et al. {PETS}c {W}eb page. http://www.mcs.anl.gov/petsc; 2019.
98. Saurel R, Le Métayer O, Massoni J, Gavrilyuk S. Shock jump relations for multiphase mixtures with stiﬀmechanical
relaxation. Shock Waves 2007; 16(3): 209–232. doi: 10.1007/s00193-006-0065-7
99. Abgrall R, Saurel R. Discrete equations for physical and numerical compressible multiphase mixtures. J. Comput. Phys.
2003; 186(2): 361–396. doi: 10.1016/S0021-9991(03)00011-1
100. Abbate E, Iollo A, Puppo G. An all-speed relaxation scheme for gases and compressible materials. J. Comput. Phys. 2017;
351: 1–24. doi: 10.1016/j.jcp.2017.08.052
101. Lax PD. Weak solutions of nonlinear hyperbolic equations and their numerical computation. Commun. Pure Appl. Math.
1954; 7(1): 159–193. doi: 10.1002/cpa.3160070112
102. Hennessey M, Kapila A, Schwendeman D. An HLLC-type Riemann solver and high-resolution Godunov method for a
two-phase model of reactive ﬂow with general equations of state. Journal of Computational Physics 2020; 405: 109180.
doi: 10.1016/j.jcp.2019.109180
103. Tokareva S, Toro E. HLLC-type Riemann solver for the Baer–Nunziato equations of compressible two-phase ﬂow. J.
Comput. Phys. 2010; 229(10): 3573–3604. doi: 10.1016/j.jcp.2010.01.016
104. Tokareva S, Toro E. A ﬂux splitting method for the Baer–Nunziato equations of compressible two-phase ﬂow. J. Comput.
Phys. 2016; 323: 45–74. doi: 10.1016/j.jcp.2016.07.019
105. Coquel F, Hérard JM, Saleh K. A positive and entropy-satisfying ﬁnite volume scheme for the Baer–Nunziato model. J.
Comput. Phys. 2017; 330: 401–435. doi: 10.1016/j.jcp.2016.11.017
106. Furfaro D, Saurel R. A simple HLLC-type Riemann solver for compressible non-equilibrium two-phase ﬂows. Comput.
Fluids 2015; 111: 159–178. doi: 10.1016/j.compﬂuid.2015.01.016
107. Petitpas F, Franquet E, Saurel R, Le Metayer O. A relaxation-projection method for compressible ﬂows. Part II: Artiﬁcial
heat exchanges for multiphase shocks. J. Comput. Phys. 2007; 225(2): 2214–2248. doi: 10.1016/j.jcp.2007.03.014


B. Re and R. Abgrall
47
108. Wilhelmsen Ø, Aasen A, Skaugen G, et al. Thermodynamic Modeling with Equations of State: Present Challenges with
Established Methods. Ind. Eng. Chem. Res. 2017; 56(13): 3503–3515. doi: 10.1021/acs.iecr.7b00317
109. Hammer M, Ervik Å, Munkejord ST. Method using a density-energy state function with a reference equation of state
for ﬂuid-dynamics simulation of vapor-liquid-solid carbon dioxide. Ind. Eng. Chem. Res. 2013; 52(29): 9965–9978. doi:
10.1021/ie303516m
110. Lund H, Aursand P. Two-Phase Flow of CO2 with Phase Transfer. Energy Procedia 2012; 23: 246–255.
doi:
10.1016/j.egypro.2012.06.034
111. Karni S. Viscous Shock Proﬁles and Primitive Formulations. SIAM J. Numer. Anal. 1992; 29(6): 1592–1609.
doi:
10.1137/0729092
APPENDIX
A DERIVATION OF THE PRESSURE FORMULATION FOR THE BN-TYPE MODEL
In this appendix, we show step by step how we have achieved the pressure formulation of Eq. (10), starting from the equation
of the total energy in (4).
For a generic ﬂuid satisfying the EOS 푒= 푒(휌, 푃), we can express the partial derivative of 퐸= 푒+ 푚2
2휌with respect to a
generic variable 휉as
휕퐸
휕휉=
[( 휕푒
휕푃
)
휌
휕푃
휕휉+
(
휕푒
휕휌
)
푃
휕휌
휕휉
]
+ 푢휕푚
휕휉−푢2
2
휕휌
휕휉
where the terms in square brackets are the partial derivative 휕푒
휕휉. Reminding deﬁnitions (6) and the triple product rule, i.e.,
(
휕푒
휕푃
)
휌
(
휕푃
휕휌
)
푒
(
휕휌
휕푒
)
푃= −1, we can re-write the derivative of 퐸as
휕퐸
휕휉=
[
1
휅
휕푃
휕휉−휒
휅
휕휌
휕휉
]
+ 푢휕푚
휕휉−푢2
2
휕휌
휕휉.
(A1)
Now, we insert Eq. (A1) into the total energy equation for phase 휎in (4), re-formulating it as
훼휎
[
1
휅휎
휕푃휎
휕푡−휒휎
휅휎
휕휌휎
휕푡+ 푢휎
휕푚휎
휕푡−
푢2
휎
2
휕휌휎
휕푡
]
+
(
푒휎+
푚2
휎
2휌휎
) [휕훼휎
휕푡+ 휕훼푢휎
휕푥
]
+훼푢휎
[
1
휅휎
휕푃휎
휕푥−휒휎
휅휎
휕휌휎
휕푥+ 푢휎
휕푚휎
휕푥−
푢2
휎
2
휕휌휎
휕푥
]
+ 휕(훼푃휎푢휎)
휕푥
−푃I푢I
휕훼휎
휕푥= 푃I휇Δ휎푃−푢I휆Δ휎푢
(A2)
where we have introduced the operator Δ휎which takes the diﬀerence between the phase 휎and the opposite one, i.e., Δ1푃=
푃1 −푃2 and Δ2푢= 푢2 −푢1.
Terms in Eq. (A2) can be re-arranged as
1
휅휎
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥
]
−
(
휒휎
휅휎
+
푢2
휎
2
) [
훼휎
휕휌휎
휕푡+ 훼푢휎
휕휌휎
휕푥
]
+푢휎
[
훼휎
휕푚휎
휕푡+ 훼푢휎
휕푚휎
휕푥
]
+
(
푒휎+
푚2
휎
2휌휎
) [휕훼휎
휕푡+ 휕훼푢휎
휕푥
]
+휕(훼푃휎푢휎)
휕푥
−푃I푢I
휕훼휎
휕푥= −휇푃IΔ휎푃−휆푢IΔ휎푢.


48
B. Re and R. Abgrall
Now, we use the density and momentum equations in (4) to replace the derivative of 휌휎and 푚휎:
1
휅휎
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥
]
+
(
휒휎
휅휎
+
푢2
휎
2
)
휌휎
[휕훼휎
휕푡+ 휕훼푢휎
휕푥
]
−푢휎
[
푚휎
휕훼휎
휕푡+ 푚휎
휕훼푢휎
휕푥
+ 휕훼푃휎
휕푥
−푃I
휕훼휎
휕푥+ 휆Δ휎푢
]
+
(
푒휎+
푚2
휎
2휌휎
) [휕훼휎
휕푡+ 휕훼푢휎
휕푥
]
+휕(훼푃휎푢휎)
휕푥
−푃I푢I
휕훼휎
휕푥= −휇푃IΔ휎푃−휆푢IΔ휎푢.
(A3)
Expanding the derivative of 훼푃휎푢휎and noting that
푢2
휎
2 휌휎−푢휎푚휎+
푚2
휎
2휌휎= 0, the previous equation reads
1
휅휎
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥
]
+
(
푒휎+ 휒휎휌휎
휅휎
) [휕훼휎
휕푡+ 휕훼푢휎
휕푥
]
+
[
푢휎푃I
휕훼휎
휕푥−푢휎휆Δ휎푢
]
+ 훼푃휎
휕푢휎
휕푥−푃I푢I
휕훼휎
휕푥= −휇푃IΔ휎푃−휆푢IΔ휎푢.
(A4)
Now, we use also the volume fraction equation in (4) to replace the temporal derivative of 훼휎, and we re-arrange the terms, to
have
1
휅휎
[
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥
]
+
(휒휎휌휎
휅휎
+ 푃휎+ 푒휎
)
훼휎
휕푢휎
휕푥−
(휒휎휌휎
휅휎
+ 푃I + 푒휎
)
(푢I −푢휎)휕훼휎
휕푥
= −
(휒휎휌휎
휅휎
+ 푃I + 푒휎
)
휇Δ휎푃−(푢I −푢휎)휆Δ휎푢.
(A5)
Finally, recalling the deﬁnitions of the speed of sound (7) and (8), we arrive to
훼휎
휕푃휎
휕푡+ 훼푢휎
휕푃휎
휕푥+ 훼휌휎푐2
휎
휕푢휎
휕푥−휌휎푐2
I,휎(푢I −푢휎)휕훼휎
휕푥= −휌휎푐2
I,휎휇Δ휎푃−휅휎(푢I −푢휎)휆Δ휎푢,
(A6)
which is the pressure formulation of the BN-type model given in Eq. (10).
B DEFINITION OF NON-CONSERVATIVE OPERATOR IN MOMENTUM EQUATIONS
We show here the process that has led to the deﬁnition of the operator 퐻푃(훼푛+1
휎
, 푃푛
I )푘≈∫휁푘푃푛
I
휕훼푛+1
휎
휕푥d푥in the momentum
equations.
Assuming (푢휎)푘= (푢휎)푘+1 = (푢휎)푘−1 = 푢and (푃휎)푖= (푃휎)푖+1 = 푃and taking into account that that (훼푚휎)푛∗
푘= (훼휌휎)푛∗
푘(푢휎)푛∗
푘,
the momentum equation (46) reads
||휁푘||
Δ푡
[
(훼휌휎)푛∗
푘(푢)푛∗
푘−(훼휌휎)푛
푘(푢)푛
푘
]
= −(푢)푛∗
[
퐹Rus
푘+ 1
2
(훼휌푛∗
휎, 푢푛) −퐹Rus
푘−1
2
(훼휌푛∗
휎, 푢푛)]
−(푃)푛[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 푃푛
I )푘.
(B7)
We recall the mapping from the primary to the staggered (see Remark 5): (훼휌휎)푘= 0.5 [(훼휌휎)푖+ (훼휌휎)푖+1
]. Therefore, the
Rusanov ﬂuxes in Eq. (B7) are
퐹Rus
푘+ 1
2
= 1
2푢푛[(훼휌휎)푛∗
푘+1 + (훼휌휎)푛∗
푘
] −1
2|푢푛| [(훼휌휎)푛∗
푘+1 −(훼휌휎)푛∗
푘
]
= 1
2푢푛
[(훼휌휎)푛∗
푖+2 + (훼휌휎)푛∗
푖+1
2
+
(훼휌휎)푛∗
푖+1 + (훼휌휎)푛∗
푖
2
]
−1
2
|||푢푛|||
[(훼휌휎)푛∗
푖+2 + (훼휌휎)푛∗
푖+1
2
−
(훼휌휎)푛∗
푖+1 + (훼휌휎)푛∗
푖
2
]
= 1
2
[
퐹Rus
푖+ 3
2
(훼휌휎, 푢) + 퐹Rus
푖+ 1
2
(훼휌휎, 푢)
]
;
퐹Rus
푘−1
2
= 1
2
[
퐹Rus
푖+ 1
2
(훼휌휎, 푢) + 퐹Rus
푖−1
2
(훼휌휎, 푢)
]
,
where 퐹Rus
푖−1
2
(훼휌휎, 푢), 퐹Rus
푖+ 1
2
(훼휌휎, 푢) and 퐹Rus
푖+ 3
2
(훼휌휎, 푢) are the density ﬂux at primary cell interfaces, as deﬁned in Eq. (38).


B. Re and R. Abgrall
49
Now, we substitute the previous ﬂux forms in the right hand side of Eq. (B7) and the expressions for (훼휌휎)푛+1
푘
and (훼휌휎)푛
푘in
the left hand side, obtaining
||휁푘||
Δ푡
(훼휌휎)푛∗
푖+ (훼휌휎)푛∗
푖+1
2
푢푛∗−
||휁푘||
Δ푡
(훼휌휎)푛
푖+ (훼휌휎)푛
푖+1
2
푢푛
= −1
2(푢)푛∗
[
퐹Rus
푖+ 3
2
(훼휌푛∗
휎, 푢푛) + 퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛) −퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛) −퐹Rus
푖−1
2
(훼휌푛∗
휎, 푢푛)
]
−(푃)푛[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 푃푛
I )푘.
(B8)
Finally, we substitute in the left hand side also the expressions for (훼휌휎)푛∗
푖
and (훼휌휎)푛∗
푖+1 given by Eq. (43)). Considering an
internal cell of a uniform grid4, for which ||휁푘|| = ||푖|| = ||푖+1|| = Δ푥, we have
||휁푘||
Δ푡
1
2
[ [(훼휌휎)푛
푖+ (훼휌휎)푛
푖+1
] 푢푛∗−[(훼휌휎)푛
푖+ (훼휌휎)푛
푖+1
] 푢푛
]
−1
2푢푛∗
[
퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛)) −퐹Rus
푖−1
2
(훼휌푛∗
휎, 푢푛)) + 퐹Rus
푖+ 3
2
(훼휌푛∗
휎, 푢푛)) −퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛))]
= −1
2푢푛∗
[
퐹Rus
푖+ 3
2
(훼휌푛∗
휎, 푢푛) + 퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛) −퐹Rus
푖+ 1
2
(훼휌푛∗
휎, 푢푛) −퐹Rus
푖−1
2
(훼휌푛∗
휎, 푢푛)
]
−(푃)푛[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 푃푛
I )푘.
(B9)
All the ﬂux terms cancel out, so, to ensure that no velocity variations arise, it is requires that
−(푃)푛[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 푃푛
I )푘= 0 .
In light of this, we deﬁne the non-conservative operator 퐻푃as
퐻푃(훼푛+1
휎
, 푃푛
I )푘= (푃I)푛
푘
[(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
]
where (푃I)푛
푘= 1
2
[(푃I)푛
푖+ (푃I)푛
푖+1
] is the interface pressure mapped at the staggered cell 휁푘.
The same reasoning applies to the equation of the momentum update (47). In this case, we obtain intermediate equations very
similar to Eqs. (B7), (B8), (B9), but with the term 훿푢푛+1 instead of 푢푛. The constraint to have 훿푢푛+1 = 0 is
−(훿푃)푛+1 [(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] + 퐻푃(훼푛+1
휎
, 훿푃푛+1
I
)푘= 0 .
so, we deﬁne
퐻푃(훼푛+1
휎
, 훿푃푛+1
I
)푘= [(푃I)푛+1
푘
−(푃I)푛
푘
] [(훼휎)푛+1
푖+1 −(훼휎)푛+1
푖
] ,
which, clearly, ensures also that, if 푃푛+1 = 푃푛, 퐻푃(훼휎, 훿푃I) = 0. Indeed, if we have a uniform pressure and velocity ﬁeld, the
equations in the correction step, that is the pressure and the update momentum ones, should be identically null.
4Considering only an internal cell is motivated by the diﬀerent computation of momentum on the boundary cells, which does not require the solution of Eqs. (46) and
(47), as explained in Sec. 4.3.5.
