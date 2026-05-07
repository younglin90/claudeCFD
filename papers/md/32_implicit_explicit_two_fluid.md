Journal of Computational Physics 498 (2024) 112696
Available online 7 December 2023
0021-9991/© 2023 Elsevier Inc. All rights reserved.
Contents lists available at ScienceDirect
Journal of Computational Physics
journal homepage: www.elsevier.com/locate/jcp
An implicit-explicit solver for a two-ﬂuid single-temperature 
model
Mária Lukáˇcová-Medvid’ová a, Ilya Peshkov b, Andrea Thomann a,c,∗
a Institut für Mathematik, Johannes-Gutenberg-Universität Mainz, Staudingerweg 9, 55099 Mainz, Germany
b Department of Civil, Environmental and Mechanical Engineering, University of Trento, Via Mesiano 77, I-38123 Trento, Italy
c Université de Strasbourg, CNRS, Inria, IRMA, F-67000 Strasbourg, France
A R T I C L E 
I N F O
A B S T R A C T
Keywords:
All-speed scheme
IMEX method
Reference state strategy
Single temperature two-ﬂuid ﬂow
Asymptotic preserving property
Symmetric hyperbolic thermodynamically 
compatible model
We present an implicit-explicit ﬁnite volume scheme for two-ﬂuid single-temperature ﬂow in all 
Mach number regimes which is based on a symmetric hyperbolic thermodynamically compatible 
description of the ﬂuid ﬂow. The scheme is stable for large time steps controlled by the interface 
transport and is computational eﬃcient due to a linear implicit character. The latter is achieved 
by linearizing along constant reference states given by the asymptotic analysis of the single-
temperature model. Thus, the use of a stiﬄy accurate IMEX Runge Kutta time integration and the 
centered treatment of pressure based quantities provably guarantee the asymptotic preserving 
property of the scheme for weakly compressible Euler equations with variable volume fraction. 
The properties of the ﬁrst and second order scheme are validated by several numerical test cases.
1. Introduction
In continuum mixture theory, the constituents of a multiphase system, also called mixture, are present at every material element 
even if an element represents a pure phase. This approach is applicable to model both situations – the case of miscible [26] and 
immiscible [33,5] multicomponent systems. The material interfaces, if present, are zones of rapid but smooth changes of a parameter 
distinguishing the phases of the mixture, typically the volume or mass fraction.
Despite the fact that almost any application in science and engineering deals with multiphase systems and there is an obvious 
need for a consistent and reliable mathematical model to describe such multicomponent systems, the continuum mixture theory is far 
from being complete and no widely accepted model exists. Perhaps, the most widely used approach is based on equations for every 
individual constituent of the system, i.e. phase mass balance, phase momentum balance, phase energy balance, etc. The key problem 
here is to ﬁnd a closure for this system of equations which is represented by the coupling terms describing the exchange of mass, 
energy, and momenta between the mixture constituents. Note that a ﬁrst-principle theory to provide such a closure is currently not 
available. Consequently, various heuristic and phenomenological approaches are used. The Baer-Nunziato (BN) model [1] introduced 
in 1986 is a representative of this class of mathematical formulations and since then an active line of research has been done to adapt 
it to various applications, [32,10,33] and recently to low Mach number ﬂows [27].
In this paper, we deal with another class of governing equations for mixtures, here in the single-temperature simpliﬁcation, 
which represents an attempt to build a mixture theory based on the ﬁrst-principle reasoning. The equations belong to the class 
* Corresponding author at: Université de Strasbourg, CNRS, Inria, IRMA, F-67000 Strasbourg, France.
E-mail addresses: lukacova@mathematik.uni-mainz.de (M. Lukáˇcová-Medvid’ová), ilya.peshkov@unitn.it (I. Peshkov), andrea.thomann@inria.fr (A. Thomann).
https://doi.org/10.1016/j.jcp.2023.112696
Received 12 May 2023; Received in revised form 25 September 2023; Accepted 29 November 2023


Journal of Computational Physics 498 (2024) 112696
2
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
of the so-called Symmetric Hyperbolic Thermodynamically Compatible (SHTC) equations [11,13,12]. The key ingredients are the 
variational principle and the second law of thermodynamics. The variational principle is used to deduce a reversible part of the 
evolution equations that is subject to entropy conservation. The second law of thermodynamics yields an irreversible part of a model 
and controls entropy production. In contrast, to the BN class of mixture models, the governing equations of the SHTC model are 
formulated not directly in terms of the phase quantities but mainly in terms of mixture quantities such as mixture mass density, 
mixture momentum, mixture energy, etc. The SHTC equations can be rewritten in terms of the phase balance equations, i.e. in a BN 
form, see [30,28]. In this way, a new term appears in the phase balance equations, that is usually missing in the BN-type models, see 
[28]. The latter can be identiﬁed as lift forces [9] acting on a rotating ﬂuid element of one phase immersed in another one.
In this work, we are interested in a single-temperature SHTC mixture model [20] which is a special case of the two-velocity, two-
pressure, two-entropy SHTC model of two-phase ﬂows derived in [30,29]. In [34] the full model has been numerically solved based 
on an explicit time integration. The diﬃculty lies in handling the two entropy balance laws of the full two-ﬂuid SHTC model and only 
one energy conservation law. On the other hand, applications such as sediment transport, granular ﬂows or aerosol transport can be 
modeled with the single temperature approach resulting in one mixture entropy balance law associated to one energy conservation 
law. These applications lie within weakly compressible ﬂow regimes, characterized by small Mach numbers. A severe diﬃculty in the 
construction of a numerical scheme applied to weakly compressible ﬂow regimes is posed by the scale diﬀerences between acoustic 
and material waves. The focus of the numerical simulation usually lies on the evolution of the slower material waves following the 
two-ﬂuid interface for which a time step controlled by the local ﬂow speed is suﬃcient. The time step of an explicit scheme, as 
proposed in [29,31] for compressible two-phase ﬂow in the SHTC framework, is bounded by the smallest Mach number. This leads to 
very restrictive time steps in the low Mach number regime and consequently to long computational times, especially when long time 
periods are considered. This problem can be overcome by considering implicit-explicit (IMEX) time integrators, where fast waves are 
treated implicitly leading to a Courant-Friedrichs-Lewy (CFL) condition that is restricted only by the local ﬂow velocity. It allows 
larger time steps while keeping the material waves well resolved. Additionally, an implicit treatment of the associated stiﬀ pressure 
terms, which trigger fast acoustic waves, has the advantage that centered ﬁnite diﬀerences can be applied without loss of stability 
while guaranteeing a Mach number independent numerical diﬀusion of the scheme, see e.g. [8,14,18] for a discussion on upwind 
schemes. Indeed, the correct amount of numerical diﬀusion is crucial to obtain the so-called asymptotic preserving (AP) schemes 
[15]. Since the ﬂow regime of the two-phase ﬂow considered here is characterized by two potentially distinct phase Mach numbers, 
diﬀerent singular Mach number limits can be obtained depending on the constitution of the mixture. For their formal derivation we 
apply asymptotic expansions, as done for the (isentropic) Euler equations, see [6–8,14,17,18,25] and the references therein. We refer 
the reader to our recent work on the isentropic SHTC two-ﬂuid model [21]. To obtain physically admissible solutions, especially in 
the weakly compressible ﬂow regime, the numerical scheme has to yield correct asymptotic behavior. This means a uniformly stable 
and consistent approximation of the limit equations as the Mach numbers tend to zero.
The profound knowledge of the structure of well-prepared initial data can be used to construct an AP scheme by applying a 
reference solution (RS)-IMEX approach. This approach was successfully applied to construct AP schemes for the (isentropic) Euler 
equations [3,16,19,35] and isentropic two-ﬂuid ﬂow [21]. This leads to a stiﬀ linear part which is then treated implicitly whereas 
the nonlinear higher order terms are integrated explicitly respecting the asymptotes in the low Mach number limit. By doing this, 
nonlinear implicit solvers can be avoided which are computationally costly.
The paper is structured as follows. In Section 2, we brieﬂy recall the model and give its non-dimensional formulation. For well-
prepared initial data, we analyze its singular Mach number limit towards the incompressible Euler equations with variable density. 
Using the knowledge of the limit reference state, we construct ﬁrst a semi discrete scheme in Section 3.2 and derive a fully discrete 
scheme in Section 3.3. The construction of higher order schemes within this framework is shortly discussed, too. Further, the AP 
property of the scheme is proven in Section 4. Finally, in Section 5, a series of 1D and 2D test problems is presented to numerically 
verifying the convergence of the proposed scheme and its behavior in compressible and weakly compressible ﬂow regime.
2. Single temperature two-ﬂuid ﬂow
In this section we recall the SHTC two-ﬂuid model derived in [30,29]. We concentrate on the model in the thermal equilibrium 
regime [20] which is a legitimate approximation for many applications mentioned in the Introduction. Thus, we deal with the 
mixture of two ﬂuids in which every material element (control volume) is characterized by the temperature 𝑇with 𝑇= 𝑇1 = 𝑇2, 
where the lower indices denote the respective phase 𝑙= 1, 2. Moreover, we assume that every material element of volume 𝒱and 
mass ℳis occupied by both ﬂuids, i.e. 𝒱= 𝜈1 + 𝜈2 and ℳ= 𝑚1 + 𝑚2, with 𝜈𝑙and 𝑚𝑙being the volume and mass of the 𝑙-th phase 
in the control volume 𝑉. However, to characterize the ﬂuid content in a control volume, it is convenient to use non-dimensional 
scalars: the volume fractions 𝛼𝑙and mass fractions 𝑐𝑙deﬁned as
𝛼𝑙= 𝜈𝑙
𝒱,
𝑐𝑙= 𝑚𝑙
ℳ= 𝜚𝑙
𝜌= 𝛼𝑙𝜌𝑙
𝜌,
(1)
where
𝜌= ℳ
𝒱= 𝜚1 + 𝜚2 = 𝛼1𝜌1 + 𝛼2𝜌2
(2)
is the mass density of the mixture, 𝜚𝑙is the mass density of the 𝑙-th phase in the control volume 𝑉, and 𝜌𝑙is the mass density of the 
𝑙-th phase. The volume and mass fractions obey the constraints


Journal of Computational Physics 498 (2024) 112696
3
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
𝛼1 + 𝛼2 = 1,
𝑐1 + 𝑐2 = 1.
(3)
Moreover, each phase is equipped with its own velocity ﬁeld 𝒗𝑙∈ℝ𝑑, where 𝑑denotes the space dimension, and the mixture control 
volume is assumed to have the velocity deﬁned as the center of mass velocity, i.e. as the weighted average given by
𝒗= 𝑐1𝒗1 + 𝑐2𝒗2.
(4)
The mixture momentum 𝜌𝒗= 𝜚1𝒗1 + 𝜚2𝒗2 is equal to the sum of the phase momenta. Additionally, one needs to characterize the 
relative motion of the phases which, in the SHTC theory, is done using the relative velocity ﬁeld
𝒘= 𝒗1 −𝒗2.
(5)
For each phase, an entropy 𝑠𝑙and internal energy 𝑒𝑙(𝜌𝑙, 𝑠𝑙) is prescribed yielding the phase pressures
𝑝𝑙= 𝜌2
𝑙
𝜕𝑒𝑙
𝜕𝜌𝑙
,
𝑙= 1,2.
(6)
We consider an ideal gas equation of state (EOS) given in terms of the respective density and single temperature resulting in
𝑠𝑙(𝜌𝑙,𝑇) = 𝑐𝑣,𝑙log
(
𝑇
𝑇0,𝑙
(
1
𝜌𝑙
)𝛾𝑙−1)
,
𝑇0,𝑙=
1
(𝛾𝑙−1)𝑐𝑣,𝑙
,
𝑒𝑙(𝜌𝑙,𝑇) = 𝑐𝑣,𝑙𝑇,
𝑝𝑙(𝜌𝑙,𝑇) = (𝛾𝑙−1)𝑐𝑣,𝑙𝜌𝑙𝑇,
(7)
where 𝛾𝑙denotes the ratio of speciﬁc heats and 𝑐𝑣,𝑙the speciﬁc heat at constant volume for each phase 𝑙= 1, 2. To conclude 
the deﬁnition of the mixture state variables, we have the speciﬁc mixture internal energy 𝑒 = 𝑐1𝑒1 + 𝑐2𝑒2, the mixture pressure 
𝑝 = 𝜌2 𝜕𝑒
𝜕𝜌= 𝛼1𝑝1 + 𝛼2𝑝2 and the total energy density of the mixture given by
𝜌𝐸= 𝜌𝑒+ 𝜌‖𝒗‖2
2
+ 𝜌𝑐1𝑐2
‖𝒘‖2
2
.
(8)
The total mixture entropy reads 𝑆= 𝑐1𝑠1 + 𝑐2𝑠2. All state variables are summarized in the vector
𝒒= (𝛼1,𝛼1𝜌1,𝛼2𝜌2,𝜌𝒗,𝒘,𝜌𝐸)𝑇.
(9)
The SHTC model with a single temperature can be written in the following way
𝜕𝛼1
𝜕𝑡+ 𝒗⋅∇𝛼1 = −𝑝1 −𝑝2
𝜏(𝛼)𝜌,
(10a)
𝜕(𝛼1𝜌1)
𝜕𝑡
+ ∇⋅(𝛼1𝜌1𝒗1) = 0,
(10b)
𝜕(𝛼2𝜌2)
𝜕𝑡
+ ∇⋅(𝛼2𝜌2𝒗2) = 0,
(10c)
𝜕(𝜌𝒗)
𝜕𝑡
+ ∇⋅(𝜌𝒗⊗𝒗+ 𝑝𝑰+ 𝜌𝑐1𝑐2𝒘⊗𝒘) = 0,
(10d)
𝜕𝒘
𝜕𝑡+ ∇⋅
([
𝒘⋅𝒗+ 𝜇1 −𝜇2 + (1 −2𝑐1)‖𝒘‖2
2
]
𝑰
)
+ (∇× 𝒘) × 𝒗= −𝑐1𝑐2𝒘
𝜏(𝑤) ,
(10e)
𝜕(𝜌𝐸)
𝜕𝑡
+ ∇⋅
(
𝒗(𝜌𝐸+ 𝑝) + 𝜌
[
𝒘⋅𝒗+ 𝜇1 −𝜇2 + (1 −2𝑐1)‖𝒘‖2
2
]
𝑐1𝑐2𝒘
)
= 0.
(10f)
Here 𝜇𝑙= 𝑒𝑙+ 𝑝𝑙
𝜌𝑙−𝑠𝑙𝑇, 𝑙= 1, 2 denote the chemical potentials. In the above formulation, the volume fraction is advected in a 
non-conservative way with the ﬂuid ﬂow 𝒗balanced by a pressure relaxation source term. The mixture mass is conserved due to 
(10b), (10c), the momentum due to (10d) and the total energy due to (10f). The relative velocity is not conserved and driven by the 
diﬀerence in the chemical potentials 𝜇= 𝜇1 −𝜇2 and a friction source term. The relaxation parameters 𝜏(𝛼) and 𝜏(𝒘) characterize the 
relaxation rates of the mixture towards pressure (𝑝1 = 𝑝2) and relative velocity (𝒗1 = 𝒗2) equilibrium.
Moreover, the model is equipped with the entropy balance law
𝜕(𝜌𝑆)
𝜕𝑡
+ ∇⋅(𝜌𝑆𝒗) = Π ≥0
(11)
with the entropy production term
Π =
1
𝑇𝜏(𝛼)𝜌2 (𝑝1 −𝑝2)2 +
1
𝑇𝜏(𝑤)𝜌2 𝑐2
1𝑐2
2‖𝒘‖2.
(12)
For details on the derivation of the model and its thermodynamical properties we refer to [31,30,29,28].
Since each phase is equipped with a respective pressure and density, a sound speed for each phase 𝑎𝑙, as well as a mixture sound 
speed 𝑎can be deﬁned by
(𝑎𝑙
)2 = 𝜕𝑝𝑙
𝜕𝜌𝑙
= 𝛾𝑙
𝑝𝑙
𝜌𝑙
and
𝑎2 = 𝜕𝑝
𝜕𝜌= 𝑐1
(𝑎1
)2 + 𝑐2
(𝑎2
)2 .
(13)


Journal of Computational Physics 498 (2024) 112696
4
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Accordingly, a Mach number can be assigned to each phase. As usual it is deﬁned by the ratio between the ﬂow velocity 𝒗and the 
sound speed 𝑎𝑙. In the case that the ﬂow is characterized by (at least) one small Mach number, diﬀerent scales arise in the model 
that yield stiﬀness in the governing equations (10). To obtain a better understanding of the scales which are present in the model, 
we will rewrite system (10) in a non-dimensional form.
2.1. Non-dimensional formulation of the two-ﬂuid model
Let us denote the non-dimensional quantities by ( ̃∙) and the corresponding reference value by (∙)ref. We assume that the 
convective scales are of the same order, i.e. 𝒗𝑙,ref = 𝒗ref = 𝑥ref∕𝑡ref. The ratio of the phase densities, however, can be large, especially 
when considering a mixture of a light gas and liquid. To take this potentially large diﬀerence into account, we deﬁne two diﬀerent 
reference densities 𝜌𝑙,ref. Note, that the volume fraction and mass fractions are already non-dimensional quantities. Further, we deﬁne 
two reference pressures 𝑝𝑙,ref from which we can compute the reference sound speeds 𝑎𝑙,ref and reference internal energies 𝑒𝑙,ref via 
the EOS (7). They are given by
(𝑎𝑙,ref
)2 =
𝑝𝑙,ref
𝜌𝑙,ref
,
𝑒𝑙,ref =
𝑝𝑙,ref
𝜌𝑙,ref
,
𝑇ref =
1
𝛾𝑙(𝛾𝑙−1)𝑐𝑣,𝑙
𝑝𝑙,ref
𝜌𝑙,ref
,
𝑙= 1,2.
(14)
The dimensional state variables 𝒒are then expressed as the product of non-dimensional quantities and reference values as follows
𝜌𝑙= ̃𝜌𝑙𝜌𝑙,ref,
𝑝𝑙= ̃𝑝𝑙𝑝𝑙,ref,
𝑒𝑙= ̃𝑒𝑙
𝑝𝑙,ref
𝜌𝑙,ref
,
𝜇𝑙= ̃𝜇𝑙
𝑝𝑙,ref
𝜌𝑙,ref
,
𝒗𝑙= ̃𝒗𝑙𝒗ref,
𝒗= ̃𝒗𝒗ref,
𝒘= ̃𝒘𝒗ref.
(15)
Further, a respective reference Mach number 𝑀𝑙is assigned to each phase 𝑙= 1, 2
𝑀𝑙= 𝒗ref
𝑎𝑙,ref
.
(16)
Inserting expressions (15) into the dimensional equations (10), dropping the tilde ( ̃∙) and using (16), we obtain the following 
non-dimensional formulation
𝜕𝛼1
𝜕𝑡+ 𝒗⋅∇𝛼1 = −
1
𝜏(𝛼)𝜌
(𝜌1,ref
𝜌ref
𝑝1
(𝑀1)2 −
𝜌2,ref
𝜌ref
𝑝2
(𝑀2)2
)
,
(17a)
𝜕(𝛼1𝜌1)
𝜕𝑡
+ ∇⋅(𝛼1𝜌1𝒗1) = 0,
(17b)
𝜕(𝛼2𝜌2)
𝜕𝑡
+ ∇⋅(𝛼2𝜌2𝒗2) = 0,
(17c)
𝜕(𝜌𝒗)
𝜕𝑡
+ ∇⋅
(
𝜌𝒗⊗𝒗+
(
𝛼1
𝜌1,ref
𝜌ref
𝑝1
(𝑀1)2 + 𝛼2
𝜌2,ref
𝜌ref
𝑝2
(𝑀2)2
)
𝑰+ 𝜌𝑐1𝑐2𝒘⊗𝒘
)
= 0,
(17d)
𝜕𝒘
𝜕𝑡+ ∇⋅
([
𝒘⋅𝒗+
𝜇1
(𝑀1)2 −
𝜇2
(𝑀2)2 + (1 −2𝑐) ‖𝒘‖2
2
]
𝑰
)
+ (∇× 𝒘) × 𝒗= −𝑐1𝑐2𝒘
𝜏(𝑤) ,
(17e)
𝜕(𝜌𝐸)
𝜕𝑡
+ ∇⋅
(
𝒗
(
𝜌𝐸+ 𝛼1
𝜌1,ref
𝜌ref
𝑝1
(𝑀1)2 + 𝛼2
𝜌2,ref
𝜌ref
𝑝2
(𝑀2)2
))
(17f)
+∇⋅
(
𝜌
[
𝒘⋅𝒗+
𝜇1
(𝑀1)2 −
𝜇2
(𝑀2)2 + (1 −2𝑐1
) ‖𝒘‖2
2
]
𝑐1𝑐2𝒘
)
= 0
(17g)
with the scaled total energy
𝐸= 𝑐1
𝜌1,ref
𝜌ref
𝑒1(𝜌1,𝑇)
(𝑀1)2
+ 𝑐2
𝜌2,ref
𝜌ref
𝑒2(𝜌2,𝑇)
(𝑀2)2
+ ‖𝒗‖2
2
+ 𝑐1𝑐2
‖𝒘‖2
2
(18)
and the mixture density 𝜌 = ̃𝜌𝜌ref = 𝛼1 ̃𝜌1𝜌1,ref+𝛼2 ̃𝜌2𝜌2,ref with ̃𝜌= 𝛼1 ̃𝜌1
𝜌1,ref
𝜌ref +𝛼2 ̃𝜌2
𝜌2,ref
𝜌ref . In the next section we introduce well-prepared 
initial data that will be used for the formal asymptotic analysis of (17) in the low Mach number limits.
2.2. Well-prepared data and low Mach number limits
As we have seen from the Mach number deﬁnition (16), the diﬀerence in the ﬂow regimes of the two phases depends mainly on 
the material constants 𝛾𝑙and 𝑐𝑣,𝑙. In particular, from the single temperature EOS (7), we obtain
𝑎2
1
𝛾1(𝛾1 −1)𝑐𝑣,1
=
𝑎2
2
𝛾2(𝛾2 −1)𝑐𝑣,2
⇔𝑎2
1 =
𝛾1(𝛾1 −1)𝑐𝑣,1
𝛾2(𝛾2 −1)𝑐𝑣,2
𝑎2
2
(19)
and consequently, with 𝑎𝑙,ref = √𝛾𝑙(𝛾𝑙−1)𝑐𝑣,𝑙𝑇ref, we ﬁnd a direct relation between two Mach numbers
𝑀1 = 𝒞𝑀2,
𝒞=
√
𝛾2(𝛾2 −1)𝑐𝑣,2
𝛾1(𝛾1 −1)𝑐𝑣,1
> 0.
(20)


Journal of Computational Physics 498 (2024) 112696
5
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
In the following, for simplicity, we consider the case 𝑀1 = 𝑀2 = 𝑀, where 0 < 𝑀≪1, i.e. 𝒞= 1. The cases 𝒞> 1 and 𝒞< 1 can 
be treated in a similar manner. For a full analysis of model (17) in the isentropic case we refer the reader to [21], where the singular 
limit for two diﬀerent Mach numbers 1 ≫𝑀1 > 𝑀2 > 0 and 1 ≈𝑀1 ≫𝑀2 > 0 are considered.
We proceed by expanding suﬃciently smooth phase state variables with respect to 𝑀. Note that the volume and mass fractions 
are non-dimensional quantities and are not expanded with respect to the Mach number.
𝜌𝑙= 𝜌𝑙,(0) + 𝑀𝜌𝑙,(1) + 𝒪(𝑀2),
𝑙= 1,2,
𝑇= 𝑇(0) + 𝑀𝑇(1) + 𝑀2𝑇(2) + 𝒪(𝑀3),
𝒗= 𝒗(0) + 𝑀𝒗(1) + 𝒪(𝑀2).
(21)
Since the relative velocity is subject to a relaxation process, we set 𝜏(𝒘) = 𝑀leading to the desired zero background relative velocity 
𝒘(0) = 0 in the limit, thus
𝒘= 𝑀𝒘(1) + 𝒪(𝑀2).
(22)
To obtain Mach number expansions also for the remaining thermodynamical quantities, we apply EOS (7) which yields
𝑝𝑙= (𝑐𝑣,𝑙(𝛾𝑙−1)𝜌𝑙,(0)𝑇(0)
) + 𝑀𝑐𝑣,𝑙(𝛾𝑙−1)(𝜌𝑙,(0)𝑇(1) + 𝜌𝑙,(1)𝑇(0)
) + 𝒪(𝑀2),
𝜇𝑙= (𝑇(0)
(𝑐𝑣,𝑙𝛾𝑙−𝑠𝑙(𝜌𝑙,(0),𝑇(0)))) + 𝑀
(
𝑐𝑣,𝑙(𝛾𝑙−1)(𝜌𝑙,(0)𝑇(1) + 𝜌𝑙,(1)𝑇(0)
)
𝜌𝑙,(0)
−𝑇(1)𝑠𝑙(𝜌𝑙,(0),𝑇(0))
)
+ 𝒪(𝑀2),
𝜌𝑙𝑒𝑙= (𝑐𝑣,𝑙𝜌𝑙,(0)𝑇(0)
) + 𝑀𝑐𝑣,𝑙
(𝜌𝑙,(0)𝑇(1) + 𝜌𝑙,(1)𝑇(0)
) + 𝒪(𝑀2)
(23)
and imply the following expansions
𝑝𝑙= 𝑝𝑙,(0) + 𝑀𝑝𝑙,(1) + 𝑀2𝑝𝑙,(2) + 𝒪(𝑀3),
𝜇𝑙= 𝜇𝑙,(0) + 𝑀𝜇𝑙,(1) + 𝑀2𝜇𝑙,(2) + 𝒪(𝑀3),
𝜌𝑙𝑒𝑙= (𝜌𝑙𝑒𝑙
)
(0) + 𝑀(𝜌𝑙𝑒𝑙
)
(1) + 𝑀2 (𝜌𝑙𝑒𝑙
)
(2) + 𝒪(𝑀3).
(24)
We insert the Mach number expansions (21), (24) and 𝜌(0) = 𝛼1𝜌1
(0) + 𝛼2𝜌2
(0) in the non-dimensional formulation (17) and sort by 
the equal order of the Mach number. Terms of the order 𝒪(𝑀−2) and 𝒪(𝑀−1) arise in the relaxation source term of equation (17b)
yielding
𝜌2,ref
𝜌ref
𝑝1,(0) =
𝜌1,ref
𝜌ref
𝑝2,(0)
and
𝜌2,ref
𝜌ref
𝑝1,(1) =
𝜌1,ref
𝜌ref
𝑝2,(1),
(25)
as well as in the momentum equation (17d)
∇
(
𝛼1
𝜌1,ref
𝜌ref
𝑝1,(0) + 𝛼2
𝜌2,ref
𝜌ref
𝑝2,(0)
)
= 0 ⇔∇𝑝1,(0) = 0, ∇𝑝2,(0) = 0
and
(26)
∇
(
𝛼1
𝜌1,ref
𝜌ref
𝑝1,(1) + 𝛼2
𝜌2,ref
𝜌ref
𝑝2,(1)
)
= 0 ⇔∇𝑝1,(1) = 0, ∇𝑝2,(1) = 0.
(27)
This implies that 𝑝𝑙and 𝜌𝑙𝑒𝑙are constant in space up to the second order perturbation 𝑝𝑙,(2) and 
(
𝜌𝑙𝑒𝑙
)
(2). Furthermore, from the 
relative velocity equation (17e) we have the following conditions
∇𝜇1,(0) = ∇𝜇2,(0),
and
∇𝜇1,(1) = ∇𝜇2,(1).
(28)
In particular this means that the diﬀerences of the chemical potentials 𝜇(0) and 𝜇(1) are constant in space. Taking these observations 
into account as well as 𝒘(0) = 0, 𝒪(𝑀−2) order terms in the energy equation (17g) reduce to
𝜕𝜌(0)𝑒(0)
𝜕𝑡
+ ∇⋅(𝜌(0)𝑒(0)𝒗(0)) + 𝑝(0)∇⋅𝒗(0) = 0.
(29)
We further assume that the phase pressure relaxation towards a common pressure is faster than the characteristic time of pressure 
wave propagation, and one obtains a uniform mixture pressure 𝑝(2) = 𝛼1
𝜌1,ref
𝜌ref 𝑝1,(2) + 𝛼2
𝜌2,ref
𝜌ref 𝑝2,(2) = 𝑝2,(2). This motivates the following 
assumption on the dynamics of the volume fraction in the low Mach number limit.
Assumption 2.1 (Transport of interfaces). In the low Mach number limit we assume
𝜕𝛼1
𝜕𝑡+ 𝒗(0) ⋅∇𝛼1 = 0.
(30)


Journal of Computational Physics 498 (2024) 112696
6
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
With this assumption, we can rewrite the energy equation at the leading order (29) as follows
𝛼1
𝜕(𝜌1𝑒1)(0)
𝜕𝑡
+ 𝛼2
𝜕(𝜌2𝑒2)(0)
𝜕𝑡
+ 𝜌(0)𝑒(0)∇⋅𝒗(0) + 𝑝(0)∇⋅𝒗(0) = 0.
(31)
In fact, (31) can be written as a convex combination with respect to the volume fraction
𝛼1
(𝜕(𝜌1,(0)𝑒1,(0))
𝜕𝑡
+ (𝜌1,(0)𝑒1,(0) + 𝑝1,(0)
)∇⋅𝒗(0)
)
+ 𝛼2
(𝜕(𝜌2,(0)𝑒2,(0))
𝜕𝑡
+ (𝜌2,(0)𝑒2,(0) + 𝑝2,(0)
)∇⋅𝒗(0)
)
= 0.
(32)
Since the volume fraction can be arbitrary under the constraint 0 < 𝛼𝑙< 1, 𝑙= 1, 2, (32) implies
𝜕(𝜌1,(0)𝑒1,(0))
𝜕𝑡
+
(
𝜌1,(0)𝑒1,(0) + 𝑝1,(0)
)
∇⋅𝒗(0) = 0
and
𝜕(𝜌2,(0)𝑒2,(0))
𝜕𝑡
+
(
𝜌2,(0)𝑒2,(0) + 𝑝2,(0)
)
∇⋅𝒗(0) = 0,
(33)
which is consistent with the limit of single phase ﬂow of the Euler equations, see e.g. [8,17,18].
Then, analogously to [8] for the case of the Euler equations, we obtain from (33), that 𝜌𝑙,(0)𝑒𝑙,(0) are constant in space and time 
and consequently we obtain from (24) that also the phase pressures at leading order 𝑝𝑙,(0) and 𝜌𝑙,(0)𝑇(0) are constant in space and 
time. Furthermore, we obtain the divergence free mixture velocity constraint ∇⋅𝒗(0) = 0. Summarizing, we can formally write the 
following expansions for the pressure and internal energy
𝑝= 𝑝(0) + 𝒪(𝑀2),
𝑝(0) = constant,
(34a)
𝜌𝑙𝑒𝑙= 𝜌𝑙,(0)𝑒𝑙,(0) + 𝒪(𝑀2),
𝜌𝑙,(0)𝑒𝑙,(0) = constant.
(34b)
To obtain the expansion for the temperature, we look at the constraint for the chemical potentials. First, (34b) implies that 𝜌𝑙,(0)𝑇(0)
are constant. Therefore, we can deﬁne two constants ℰ𝑙> 0 such that
𝑇(0)𝜌1,(0) = ℰ1,
𝑇(0)𝜌2,(0) = ℰ2.
(35)
Then it follows from ∇(𝜇1,(0) −𝜇2,(0)) = 0 that
0 =
(
𝑐𝑣,2 log
((𝜌0,2
ℰ2
)𝛾2−1 𝑇𝛾2
(0)
𝑇0
)
−𝑐𝑣,𝑙log
((𝜌0,1
ℰ1
)𝛾1−1 𝑇𝛾1
(0)
𝑇0
))
∇𝑇(0)
(36)
and consequently 𝑇(0) is constant unless both phases coincide. Since we consider a general case of diﬀerent phase densities, it follows 
from (35), that the phase densities 𝜌𝑙,(0) are constant as well. From relation (34a) follows 𝑝(1) = 0 thus 𝜌𝑙,(0)𝑇(1) + 𝜌𝑙,(1)𝑇(0) = 0 which 
implies
𝑇(1) = −
𝜌𝑙,(1)
𝜌𝑙,(0)
𝑇(0).
(37)
Relation (28) yields 𝑇(1) is constant and together with (37) implies
𝜇𝑙= 𝜇𝑙,(0) + 𝒪(𝑀2),
𝜌𝑙= 𝜌𝑙,(0) + 𝒪(𝑀2),
𝑇= 𝑇(0) + 𝒪(𝑀2),
{𝜇𝑙,(0), 𝜌𝑙,(0) ,𝑇(0)} = constant.
(38)
Plugging these expansions in (17b) and (17c) and single out the ﬁrst order perturbations for the phase densities, it follows 𝒗(1) = 0
and 𝒘(1) = 0. Consequently, we have for the friction source term 𝜏(𝒘) = 𝒪(𝑀2). We proceed by deﬁning a set of well-prepared initial 
data.
Deﬁnition 2.2 (Well-prepared initial data for variable volume fraction). Let 𝒒∈ℝ4+2𝑑denote the state vector and let both phases be in 
the same Mach number regime denoted by 𝑀. Let Assumption 2.1 hold. Then the set of well-prepared initial data is given as
Ω𝑀
𝑤𝑝=
{
𝒒∈ℝ4+2𝑑∶
𝜌1,ref
𝜌ref
𝑝2
(𝑘) =
𝜌1,ref
𝜌ref
𝑝1,(𝑘), 𝑘= 0,1,2;
∇𝑝1,(0) = 0, ∇𝑝2,(0) = 0;
∇𝑝1,(1) = 0, ∇𝑝2,(1) = 0;
∇𝜇1,(0) = ∇𝜇1,(0), ∇𝜇1,(1) = ∇𝜇2,(1);
∇⋅𝒗(0) = 0,
𝒗(1) = 0;
𝜏(𝒘) = 𝒪(𝑀2) }
(39)
using the Mach number expansions (21), (24).
For well-prepared initial data, we obtain formally for 𝑀→0 and 𝜏(𝒘) = 𝒪(𝑀2) the following incompressible limit equations with 
variable volume fraction
𝜕𝛼1
𝜕𝑡+ 𝒗(0) ⋅∇𝛼1 = 0,
𝜌(0) = 𝛼1𝜌1,(0) + 𝛼2𝜌2,(0),
(40a)
𝜕𝒗(0)
𝜕𝑡
+ 𝒗(0) ⋅∇𝒗(0) +
∇𝑝(2)
𝜌(0)
= 0,
∇⋅𝒗(0) = 0,
(40b)
where 𝑝(2) is the second order perturbation of the uniform pressure given by −∇⋅(∇𝑝(2)∕𝜌(0)
) = ∇2 ∶(𝒗(0) ⊗𝒗(0)
)
acting as the 
Lagrangian multiplier. Note that the limit velocity equation is derived applying 𝜕𝑡𝛼= −𝒗(0) ⋅∇𝛼in the momentum formulation (17d).


Journal of Computational Physics 498 (2024) 112696
7
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
3. Numerical scheme
Let us write the two-ﬂuid model (17) in the following compact form
𝜕𝒒
𝜕𝑡+ ∇⋅𝒇(𝒒) + 𝑩(𝒒) ⋅∇𝒒= 𝒓(𝒒),
(41)
were 𝒒denotes the vector of state variables deﬁned in (9), 𝒇the ﬂux function consisting of the conservative terms, 𝑩(𝒒) is the matrix 
that contains the non-conservative contributions and 𝒓the relaxation source terms acting on the volume fraction and the relative 
velocity.
In the following, we construct a numerical scheme for the two ﬂuid single-temperature model (41) which is stable independently 
of the Mach numbers 𝑀1 and 𝑀2. This allows to follow the dynamics associated with the ﬂow velocity 𝒗, especially the transport of 
the volume fraction that represents the interface between the two phases. In addition we require the new scheme to be asymptotically 
preserving (AP), meaning that the numerical scheme in the singular limit as 𝑀→0 has to be consistent with a discretization of the 
incompressible limit equations (40).
To achieve this goal, we use an operator splitting approach, dividing the ﬂux 𝒇into terms 𝒇ex treated explicitly and 𝒇im integrated 
implicitly. The components of the non-conservative terms 𝑩only involve terms with respect to the velocities 𝒗and 𝒘and are treated 
explicitly. The resulting implicit system is in general nonlinear due to the nonlinearity of the ﬂux function 𝒇im and the EOS (7). This, 
however, implies a huge computational overhead since nonlinear solvers would be required to solve large coupled implicit systems. 
To reduce computational costs, we construct a linear implicit numerical scheme whose implicit part can be solved by direct or 
iterative linear solvers. To avoid that the AP property is lost during the linearization process, we use the so called reference solution 
(RS) approach detailed in the subsequent section that has been successfully applied to construct schemes for the Euler equations and 
isentropic two-phase ﬂows [24,3,16,21].
3.1. Reference solution approach
In the singular limit as 𝑀→0, the stiﬀness in the system is mainly connected to the pressure and chemical potential terms in the 
momentum and relative velocity equation. Further, these terms are coupled with the evolution equation for the total energy density 
𝜌𝐸. Therefore, to obtain a time step that is dominated by the mixture velocity 𝒗, these terms need to be treated implicitly. It follows 
from the EOS (7), that the mixture pressure 𝑝depends linearly on 𝜌𝐸and we can write
𝑝= (𝜙𝑝−1)(𝜌𝐸−𝜌𝐸kin
),
with
𝜙𝑝(𝛼1,𝛼2,𝜌1,𝜌2) =
𝛾1𝛼1𝜌1𝑐𝑣,1 + 𝛾2𝛼2𝜌2𝑐𝑣,2
𝛼1𝜌1𝑐𝑣,1 + 𝛼2𝜌2𝑐𝑣,2
,
(42)
where 𝐸kin contains all kinetic energy contributions
𝐸kin = ‖𝒗‖2
2
+ 𝑐1𝑐2
‖𝒘‖2
2
.
(43)
For the chemical potentials we have a nonlinear dependence on 𝜌𝐸via the phase entropies 𝑠𝑙(𝒒), 𝑙= 1, 2, given by
𝜇= 𝜇1
𝑀2
1
−𝜇2
𝑀2
2
= 𝜙𝜇(𝜌𝐸−𝜌𝐸kin),
where
𝜙𝜇(𝒒) =
𝛾1𝑐𝑣,1 −𝑠1(𝒒) −𝒞2(𝛾2𝑐𝑣,2 −𝑠2(𝒒))
𝛼1𝜌1𝑐𝑣,1 + 𝒞2𝛼2𝜌2𝑐𝑣,2
(44)
with 𝒞being the ratio of Mach numbers deﬁned in (20).
Note that for single phase ﬂow, 𝜙𝑝= 𝛾−1 is constant and 𝜙𝜇= 0. Consequently, formulations (42), (44) reduce to the Euler case 
studied in [4], and are consistent with single phase ﬂow. Therefore, we linearize only the diﬀerence of the chemical potentials 𝜇
with respect to 𝜌𝐸around a reference state 𝒒RS as follows
𝜇= 𝜇RS +
(
𝜕𝜇
𝜕(𝜌𝐸)
)
RS
(
𝜌𝐸−(𝜌𝐸)RS
)
+ 𝒪
((
𝜌𝐸−(𝜌𝐸)RS
)2)
,
(45)
where
𝜕𝜇
𝜕(𝜌𝐸) =
𝑐𝑣,1(𝛾1 −1) −𝑠1(𝜌1,𝑇) −𝒞2(𝑐𝑣,2(𝛾2 −1) −𝑠2(𝜌2,𝑇))
𝛼1𝜌1𝑐𝑣,1 + 𝒞2𝛼2𝜌2𝑐𝑣,2
= 𝒪(1)
(46)
for all 𝒞= 𝑀1
𝑀2 . The reference state is set to
𝒒RS = (𝛼1,𝛼1𝜌1,(0),𝛼2𝜌2,(0),𝜌(0)𝒗,𝒘,(𝜌𝐸)RS)𝑇,
(𝜌𝐸)RS = 𝜌(0)𝑒(0) + 𝜌(0)𝐸kin,
(47)
where 𝜌𝑙,(0), 𝜌𝑙,(0)𝑒𝑙,(0) are constant leading order states from (38) and
𝜌(0) = 𝛼1𝜌1,(0) + 𝛼2𝜌2,(0),
𝜌(0)𝑒(0) = 𝛼1
𝜌1,(0)𝑒1,(0)
𝑀2
1
+ 𝛼2
𝜌2,(0)𝑒2,(0)
𝑀2
2
.
We split 𝜇into a part that is linear in 𝜌𝐸given by


Journal of Computational Physics 498 (2024) 112696
8
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
̂𝜇= ̂𝜇RS +
(
𝜕𝜇
𝜕(𝜌𝐸)
)
RS
𝜌𝐸,
̂𝜇RS = 𝜇(𝒒RS) −
(
𝜕𝜇
𝜕(𝜌𝐸)
)
RS
× (𝜌𝐸)RS
(48)
and a nonlinear part
̄𝜇= 𝜇−̂𝜇= 𝒪
((𝜌𝐸−(𝜌𝐸)RS
)2)
.
(49)
Note that if considering well-prepared initial data, ̄𝜇is of order 𝑀2
𝑙, 𝑙= 1, 2. This can be seen by multiplying 𝜇= 𝜇1
𝑀2
1
−𝜇2
𝑀2
2
, without 
loss of generality, by 𝑀2
1 . Then we obtain
𝜇1 −𝒞2𝜇2 = 𝜇1,RS −𝒞2𝜇2,RS +
(
𝜕𝜇
𝜕(𝜌𝐸)
)
RS
× (𝜌̌𝐸−(𝜌̌𝐸)RS
) + 𝒪((𝜌̌𝐸−(𝜌̌𝐸)RS)2),
(50)
where
𝜌̌𝐸= 𝛼1𝜌1𝑒1 + 𝒞2𝛼2𝜌2𝑒2 + 𝑀2
1𝜌𝐸kin.
(51)
For the diﬀerence between the total energy density and its reference value we obtain
𝜌̌𝐸−(𝜌̌𝐸)RS = 𝑀2
1𝛼1
(𝜌1𝑒1
)
(2) + (𝑀2𝒞)2𝛼2
(𝜌2𝑒2
)
(2) + 𝑀2
1𝜌(2)𝐸kin = 𝑀2
1
(𝑐1𝑒1,(2) + 𝑐2𝑒2,(2) + 𝜌(2)𝐸kin
)
(52)
with 𝜌(2) = 𝑀2
1𝛼1𝜌1,(2) + 𝑀2
2𝛼2𝜌2,(2). Therefore, it follows
̄𝜇= 𝜇1 −𝒞2𝜇2
𝑀2
1
−̂𝜇= 𝒪
(
(𝜌̌𝐸−𝜌̌𝐸RS)2
𝑀2
1
)
= 𝒪(𝑀2
1).
(53)
Consequently, the nonlinear term ̄𝜇vanishes as 𝑀1 tends to 0 and can be treated explicitly without imposing a severe time step 
restriction. Note that for compressible ﬂow 𝑀1 ≈1, these terms are important to obtain the correct wave speeds and cannot be 
neglected.
Taking these considerations into account, the following subsystem will be treated explicitly
𝜕𝛼1
𝜕𝑡+ 𝒗⋅∇𝛼1 = 0,
(54a)
𝜕(𝛼1𝜌1)
𝜕𝑡
+ ∇⋅(𝛼1𝜌1𝒗1) = 0,
(54b)
𝜕(𝛼2𝜌2)
𝜕𝑡
+ ∇⋅(𝛼2𝜌2𝒗2) = 0,
(54c)
𝜕(𝜌𝒗)
𝜕𝑡
+ ∇⋅(𝜌𝒗⊗𝒗) = 0,
(54d)
𝜕𝒘
𝜕𝑡+ ∇⋅
([
𝒘⋅𝒗+
(
1 −2𝑐1
) ‖𝒘‖2
2
]
𝑰
)
+ (∇× 𝒘) × 𝒗= 0,
(54e)
𝜕(𝜌𝐸)
𝜕𝑡
+ ∇⋅
(
𝜌
[
𝒘⋅𝒗+ (1 −2𝑐1
) ‖𝒘‖2
2
]
𝑐1𝑐2𝒘
)
= 0.
(54f)
Written in compact notation, we have
𝜕𝒒
𝜕𝑡+ ∇⋅𝒇ex(𝒒) + 𝑩(𝒒)∇𝒒= 0.
(55)
Subsystem (54) is weakly hyperbolic, since it lacks one linearly independent eigenvector for the characteristic speed 𝜆𝒗
1. The complete 
list of characteristic speeds is given by
𝜆0 = 0,
𝜆𝒗
1 = 𝒗⋅𝒏(8×),
𝜆𝒗
2 = (𝒗+ (1 −2𝑐1)𝒘) ⋅𝒏.
(56)
Applying, e.g. the Rusanov numerical ﬂuxes, a numerical solution of the weakly hyperbolic system (54) can be obtained. Further, 
rewriting pressures and chemical potentials in terms of 𝜌𝐸and using the decomposition
𝜇= ̂𝜇+ ̄𝜇
and
𝜕𝜇𝑅𝑆=
(
𝜕𝜇
𝜕(𝜌𝐸)
)
RS
,
the implicitly treated subsystem is given by
𝜕𝛼1
𝜕𝑡= −
1
𝜏(𝛼)𝜌
(
𝜌1,ref
𝜌ref
𝑝1
𝑀2
1
−
𝜌2,ref
𝜌ref
𝑝2
𝑀2
2
)
,
(57a)
𝜕(𝛼1𝜌1)
𝜕𝑡
= 0,
(57b)


Journal of Computational Physics 498 (2024) 112696
9
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
𝜕(𝛼2𝜌2)
𝜕𝑡
= 0,
(57c)
𝜕(𝜌𝒗)
𝜕𝑡
+ ∇⋅((𝜙𝑝−1)(𝜌𝐸−𝜌𝐸kin
)𝑰+ 𝜌𝑐1𝑐2𝒘⊗𝒘) = 0,
(57d)
𝜕𝒘
𝜕𝑡+ ∇⋅
([
𝜕𝜇RS 𝜌𝐸+ ̂𝜇RS + ̄𝜇
]
𝑰
)
= −𝑐1𝑐2𝒘
𝜏(𝑤) ,
(57e)
𝜕(𝜌𝐸)
𝜕𝑡
+ ∇⋅
(
𝜌𝒗
(𝜌𝐸+ 𝑝
𝜌
)
+ 𝜇𝜌𝑐1𝑐2𝒘
)
= 0
(57f)
which yields the corresponding compact form
𝜕𝒒
𝜕𝑡+ ∇⋅𝒇im(𝒒) = 𝒓(𝒒).
(58)
In the following, we construct an IMEX scheme for subsystems (55) and (58). We start with the time semi-discrete scheme.
3.2. Time semi-discrete scheme
Let the time interval (0, 𝑇𝑓) be discretized by 𝑡𝑛= 𝑛Δ𝑡, where Δ𝑡denotes the time step subject to a time step restriction based on 
a CFL condition given by
Δ𝑡≤𝜈
Δ𝑥
max
𝒙∈Ω (|𝜆𝒗
1(𝒙,𝑡𝑛)|,|𝜆𝒗
2(𝒙,𝑡𝑛)|).
(59)
Therein, 𝜆𝒗
1, 𝜆𝒗
2 are the characteristic speeds of the explicit subsystem (56) evaluated at time level 𝑡𝑛. For a ﬁrst order scheme in 
time, we apply the forward Euler method for the explicit subsystem (55)
𝒒(1) = 𝒒𝑛−Δ𝑡∇⋅𝒇𝑒𝑥(𝒒𝑛) −Δ𝑡𝑩(𝒒𝑛) ⋅∇𝒒𝑛
(60)
and a backward Euler method for the implicit subsystem (57). We ﬁnd that there are still some nonlinear terms present in the implicit 
subsystem (57) yielding a nonlinear coupled system. Extending the approach from [4] for the Euler equations, we linearize certain 
ﬂux terms in time yielding the following time discretization
(𝛼1𝜌1)𝑛+1 = (𝛼1𝜌1)(1),
(61a)
(𝛼2𝜌2)𝑛+1 = (𝛼2𝜌2)(1),
(61b)
(𝜌𝒗)𝑛+1 = 𝜌𝒗(1) −Δ𝑡∇
(
(𝜙𝑛
𝑝−1)𝜌𝐸𝑛+1 + ̂𝑝𝑛)
−Δ𝑡∇⋅
((𝜌𝑐1𝑐2
)𝑛+1 𝒘𝑛⊗𝒘𝑛)
,
(61c)
𝒘𝑛+1 = 𝒘(1) −Δ𝑡∇
(
𝜕𝜇𝑛
RS(𝜌𝐸)𝑛+1 + ̂𝜇(1)
RS + ̄𝜇𝑛)
−Δ𝑡
𝜏(𝑤)
(𝑐1𝑐2
)𝑛+1 𝒘𝑛+1,
(61d)
(𝜌𝐸)𝑛+1 = (𝜌𝐸)(1) −Δ𝑡∇⋅
(
(𝜌𝒗)𝑛+1 (𝜌𝐸+ 𝑝)𝑛
𝜌𝑛+1
+ 𝜇𝑛(𝜌𝑐1𝑐2)𝑛+1𝒘𝑛+1
)
,
(61e)
where ̂𝑝𝑛= −(𝜙𝑛
𝑝−1)𝜌𝐸𝑛
kin. Rewriting the relative velocity equations with 𝜌𝑛+1 = (𝛼1𝜌1)𝑛+1 + (𝛼2𝜌2)𝑛+1 implies
𝒘𝑛+1 =
(
𝜏(𝑤)
𝜏(𝑤) + Δ𝑡(𝑐1𝑐2
)𝑛+1
)
𝒘(1) −
(
Δ𝑡𝜏(𝑤)
𝜏(𝑤) + Δ𝑡(𝑐1𝑐2
)𝑛+1
)
∇
(
𝜕𝜇𝑛
RS(𝜌𝐸)𝑛+1 + ̂𝜇𝑛
RS + ̄𝜇𝑛)
.
(62)
Substituting the relative velocity and momentum in the total energy equation yields a linear implicit equation for the total energy 
given by
(𝜌𝐸)𝑛+1 −Δ𝑡2∇⋅
((𝜌𝐸+ 𝑝)𝑛
𝜌𝑛+1
∇
(
(𝜙𝑛
𝑝−1)(𝜌𝐸)𝑛+1))
−Δ𝑡2∇⋅
((
𝜏(𝑤)𝜇𝑛(𝜌𝑐1𝑐2)𝑛+1
𝜏(𝑤) + Δ𝑡
(
𝑐1𝑐2
)𝑛+1
)
∇(𝜕𝜇𝑛
RS(𝜌𝐸)𝑛+1)
)
= (𝜌𝐸)(1) −Δ𝑡∇⋅
((𝜌𝐸+ 𝑝)𝑛
𝜌𝑛+1
(
𝜌𝒗(1) + Δ𝑡∇((𝜙𝑝−1)𝜌𝐸kin
)𝑛−Δ𝑡∇⋅
((𝜌𝑐1𝑐2
)𝑛+1 𝒘𝑛⊗𝒘𝑛)))
−Δ𝑡∇⋅
((
𝜏(𝑤)𝜇𝑛(𝜌𝑐1𝑐2)𝑛+1
𝜏(𝑤) + Δ𝑡
(
𝑐1𝑐2
)𝑛+1
)
𝒘(1) −
(
Δ𝑡𝜏(𝑤)𝜇𝑛(𝜌𝑐1𝑐2)𝑛+1
𝜏(𝑤) + Δ𝑡
(
𝑐1𝑐2
)𝑛+1
)
∇( ̂𝜇𝑛
RS + ̄𝜇𝑛)
)
.
(63)
Having obtained the total energy (𝜌𝐸)𝑛+1 we can successively update the relative velocity
𝒘𝑛+1 =
(
𝜏(𝑤)
𝜏(𝑤) + Δ𝑡
(
𝑐1𝑐2
)𝑛+1
)
(𝒘(1) −Δ𝑡∇𝜇𝑛+1)
(64)
and the momentum


Journal of Computational Physics 498 (2024) 112696
10
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
(𝜌𝒗)𝑛+1 = 𝜌𝒗(1) −Δ𝑡∇𝑝𝑛+1 −Δ𝑡∇⋅
((𝜌𝑐1𝑐2
)𝑛+1 𝒘𝑛+1 ⊗𝒘𝑛+1)
.
(65)
Finally, the volume fraction at the next time level is obtained from the pressure relaxation
𝜕𝛼1
𝜕𝑡= −
1
𝜏(𝛼)𝜌
(
𝑝1
𝑀2
1
−𝑝2
𝑀2
2
)
.
(66)
Rewriting the source term in terms of the state variables 𝒒, we ﬁnd
𝜕𝛼1
𝜕𝑡= −1
𝜏(𝛼)
((𝛾1 −1)𝑐𝑣,1𝑐1
𝛼1
−
(𝛾2 −1)𝑐𝑣,2𝑐2
1 −𝛼1
)
𝜌𝐸−𝜌𝐸kin
𝛼1𝜌1𝑐𝑣,1 + 𝛼2𝜌2𝑐𝑣,2
(67)
which is a nonlinear ordinary diﬀerential equation in 𝛼1 and can be solved implicitly applying the backward Euler scheme and 
the Newton algorithm to solve the nonlinear implicit system which concludes the time semi-discrete scheme. We proceed with the 
construction of the fully discrete scheme in the next section.
3.3. Fully discrete scheme
In time, we set as before 𝑡𝑛+1 = 𝑡𝑛+ Δ𝑡, where Δ𝑡obeys the CFL condition (59). In space, we consider a two-dimensional 
computational domain Ω divided into cells 𝐶𝐼= [𝑥1,𝑖−1∕2, 𝑥1,𝑖+1∕2] ×[𝑥2,𝑗−1∕2, 𝑥2,𝑗+1∕2] with 𝒙= (𝑥1, 𝑥2)𝑇. The common edge between 
two neighboring cells Ω𝐼and Ω𝐽is denoted by 𝜕Ω𝐼𝐽and the set of neighbors of Ω𝐼associated with the unit normal vector pointing 
from the cell Ω𝐼to Ω𝐽given by 𝒏𝐼𝐽is denoted by 𝒩𝐼. We consider a uniform mesh size Δ𝑥1, Δ𝑥2 in each direction and the barycenter 
of 𝐶𝐼is denoted by 𝒙𝐼= (𝑖Δ𝑥1, 𝑗Δ𝑥2) for 𝑖, 𝑗= 1, … , 𝑁. We use a ﬁnite volume framework, where the solution on the cell 𝐶𝐼at 
time 𝑡𝑛is approximated by the average given by
𝒒𝑛
𝐼≈
1
|Ω𝐼| ∫
Ω𝐼
𝒒(𝒙,𝑡𝑛) 𝑑𝒙.
(68)
A fully discrete ﬁnite volume (FV) method for (55) reads
𝒒(1)
𝐼= 𝒒𝑛
𝐼−Δ𝑡
∑
𝐾∈𝒩𝐼
|𝜕Ω𝐼𝐾|
|Ω𝐼|
(𝑭ex(𝒒𝑛
𝐼,𝒒𝑛
𝐾) ⋅𝒏𝐼𝐾+ 𝑫(𝒒𝑛
𝐼,𝒒𝑛
𝐾) ⋅𝒏𝐼𝐾
),
(69)
using a Rusanov numerical ﬂux
𝑭ex(𝒒𝑛
𝐼,𝒒𝑛
𝐾) ⋅𝒏𝐼𝐾= 1
2
(𝒇ex(𝒒𝑛
𝐼) + 𝒇ex(𝒒𝑛
𝐾)) ⋅𝒏𝐼𝐾−𝑠𝐼𝐾𝑰(𝒒𝑛
𝐾−𝒒𝑛
𝐼),
(70)
where 𝑠𝐼𝐾= max𝑘(|𝜆𝑘(𝒒𝐼)|, |𝜆𝑘(𝒒𝐾)|) denotes the maximum eigenvalue at the interface 𝜕Ω𝐼𝐾. The non-conservative product is 
approximated in the following way
𝑫(𝒒𝑛
𝐼,𝒒𝑛
𝐾) ⋅𝒏𝐼𝐾= 1
2𝑩(̃𝒒𝑛) ⋅(𝒒𝑛
𝐾−𝒒𝑛
𝐼),
̃𝒒= 1
2
(
𝒒𝑛
𝐾+ 𝒒𝑛
𝐼
)
.
(71)
The implicit elliptic equation for the total energy (63) is based on a centered ﬁnite diﬀerence approximation for the space discretiza-
tion and can be formulated on cell 𝐶𝐼in the following way
(𝜌𝐸)𝑛+1
𝐼
−Δ𝑡2 (ℒ𝑛
𝐼(𝜌𝐸)𝑛+1
𝐼
+ 𝒦𝑛
𝐼(𝜌𝐸)𝑛+1
𝐼
) =(𝜌𝐸)(1)
𝐼−Δ𝑡
∑
𝐾∈𝒩𝐼
|𝛿Ω𝐼𝐾|
|Ω𝐼| ℱ(𝒒(1)
𝐼,𝒒(1)
𝐾) ⋅𝒏𝐼𝐾
−Δ𝑡2ℒ𝑛
𝐼
(𝜌𝐸𝑛
kin
) −Δ𝑡2𝒦𝑛
𝐼
((𝜌𝐸)𝑛
RS −(𝜕𝜇𝑛
RS)−1 ̄𝜇𝑛).
(72)
Here, the weighted Laplacians are discretized as follows
ℒ𝐼(𝜌𝐸)𝐼=
∑
𝐾∈𝒩𝐼
|𝜕Ω𝐼𝐾|
|Ω𝐼| 𝐺1(𝒒𝐼,𝒒𝐾)[𝐻1(𝜌𝐸)](𝒒𝐼,𝒒𝐾),
𝒦𝐼(𝜌𝐸)𝐼=
∑
𝐾∈𝒩𝐼
|𝜕Ω𝐼𝐾|
|Ω𝐼| 𝐺2(𝒒𝐼,𝒒𝐾)[𝐻2(𝜌𝐸)](𝒒𝐼,𝒒𝐾)
(73)
with
𝐺𝑘(𝒒𝐼,𝒒𝐾) = 1
2
(𝑔𝑘(𝒒𝐼) + 𝑔𝑘(𝒒𝐾)),
𝐻𝑘(𝒒𝐼,𝒒𝐾) = |𝜕Ω𝐼𝐾|
|Ω𝐼|
(ℎ𝑘(𝒒𝐼) −ℎ𝑘(𝒒𝐾)),
𝑘= 1,2
(74)
where
𝑔1 = (𝜌𝐸+ 𝑝)𝑛
𝜌𝑛+1
,
ℎ1 = 𝜙𝑛
𝑝−1,
𝑔2 =
𝜏(𝑤)𝜇𝑛(𝜌𝑐1𝑐2)𝑛+1
𝜏(𝑤) + Δ𝑡
(
𝑐1𝑐2
)𝑛+1 ,
ℎ2 = 𝜕𝜇𝑛
RS.
(75)
The divergence terms are approximated as
ℱ(𝒒(1)
𝐼,𝒒(1)
𝐾) ⋅𝒏𝐼𝐾= 1
2
(
𝑔1(𝒒𝐼)𝜌𝒗(1)∗
𝐼
+ 𝑔1(𝒒𝐾)𝜌𝒗(1)∗
𝐾
)
⋅𝒏𝐼𝐾,
(76)


Journal of Computational Physics 498 (2024) 112696
11
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Table 1
Butcher tableaux of the ﬁrst and second order scheme.
IMEX-RK Scheme
𝑑
𝐴
𝑏𝑇
̃𝑑
̃𝐴
̃𝑏𝑇
Backward/Forward Euler
1
1
1
0
0
1
ARS(2,2,2)
0
0
0
0
𝛾
0
𝛾
0
1
0
1 −𝛾
𝛾
0
1 −𝛾
𝛾
0
0
0
0
𝛾
𝛾
0
0
1
𝛿
1 −𝛿
0
𝛿
1 −𝛿
0
𝛾= 1 −
1
√
2 , 𝛿= 1 −1
2𝛾
where 𝜌𝒗(1)∗contains the relative velocity ﬂux component ((𝜌𝑐1𝑐2)𝑛+1𝒘𝑛⊗𝒘𝑛)
with centered diﬀerences analogously to (76). The 
coeﬃcient matrix resulting from the linear equation (73) is strictly diagonal dominant. Therefore, the linear system of equations 
has a unique solution independent of the Mach number regime. Numerically it is solved by a preconditioned linear iterative solver 
GMRES provided by PetSc [2].
Once the energy 𝜌𝐸is computed at 𝑡𝑛+1, the relative velocity (64) and momentum (65) are updated consecutively by the FV 
method
𝒘𝑛+1
𝐼
= 𝒘(1)
𝐼−Δ𝑡
∑
𝐾∈𝒩𝐼
|𝜕Ω𝐼𝐾|
|Ω𝐼| 𝑭im
(𝒘)(𝒒𝑛+1
𝐼
,𝒒𝑛+1
𝐾) ⋅𝒏𝐼𝐾+ Δ𝑡𝒓(𝒘)(𝒒𝑛+1
𝐼
),
(77a)
𝜌𝒗𝑛+1
𝐼
= 𝜌𝒗(1)
𝐼−Δ𝑡
∑
𝐾∈𝒩𝐼
|𝜕Ω𝐼𝐾|
|Ω𝐼| 𝑭im
(𝜌𝒗)(𝒒𝑛+1
𝐼
,𝒒𝑛+1
𝐾) ⋅𝒏𝐼𝐾.
(77b)
The numerical ﬂux 𝑭im is constructed by the ﬁnite diﬀerence approximation deﬁned analogously as in (76) based on the implicit 
ﬂux 𝒇im. The update of the volume fraction is approximated by the backward Euler method
𝛼𝑛+1
𝐼
= 𝛼(1)
𝐼
+ Δ𝑡𝒓(𝛼)(𝒒𝑛+1
𝐼
),
(78)
where 𝒓(𝛼) denotes the pressure relaxation source term. Update (78) approximates on each cell the corresponding ordinary diﬀerential 
equation (ODE) independently. To solve the nonlinear implicit system arising from the backward Euler discretization of the ODE, a 
Newton algorithm is applied.
The steps of the reference solution implicit-explicit ﬁnite volume (RS-IMEX FV) scheme can be summarized as follows:
1. Compute the explicit update 𝒒(1)
𝐼
given by (69) based on the advective terms 𝒇ex and 𝑩with the numerical ﬂux (70) and the 
approximation of the non-conservative terms (71) under the material CFL condition (59).
2. Compute the implicit update 𝒒𝑛+1
𝐼
given by the following consecutive steps:
a) Solve the linear implicit equation (72) with centered elliptic operators (73) for the total energy (𝜌𝐸)𝑛+1
𝐼
based on a lineariza-
tion using reference states 𝒒𝑛
RS deﬁned in (47).
b) Compute ﬁrst the update of the relative velocity 𝒘𝑛+1
𝐼
given in (77a) and then the momentum 𝜌𝒗𝑛+1
𝐼
given in (77b) using the 
respective full nonlinear ﬂux components given in 𝒇im discretized with centered numerical ﬂuxes. Due to the knowledge of 
(𝜌𝐸)𝑛+1
𝐼
and the consecutive execution, both updates can be done explicitly.
c) Solve on each cell the nonlinear implicit system (78) for the volume fraction 𝛼𝑛+1
𝐼
arising from the implicit treatment of the 
pressure relaxation process 𝒓(𝛼) using a Newton algorithm.
3.4. Higher order extension
The above procedure ﬁts in the framework of IMEX Runge Kutta methods, using a forward Euler scheme for the explicit and an 
backward Euler scheme for the implicit subsystems. The corresponding Butcher tableaux are given in Table 1. For an 𝑠stage IMEX 
Runge Kutta method, the Butcher tableaux (𝐴, 𝑏, 𝑑) for the implicit and ( ̃𝐴, ̃𝑏, ̃𝑑) for the explicit parts are given by
𝐴=
⎡
⎢
⎢⎣
𝑎11
⋯
0
⋮
⋱
⋮
𝑎𝑠1
⋯
𝑎𝑠𝑠
⎤
⎥
⎥⎦
,
𝑏=
⎡
⎢
⎢⎣
𝑏1
⋮
𝑏𝑠
⎤
⎥
⎥⎦
𝑑=
⎡
⎢
⎢⎣
𝑑1
⋮
𝑑𝑠
⎤
⎥
⎥⎦
,
̃𝐴=
⎡
⎢
⎢⎣
̃𝑎11
⋯
0
⋮
⋱
⋮
̃𝑎𝑠1
⋯
̃𝑎𝑠𝑠
⎤
⎥
⎥⎦
,
̃𝑏=
⎡
⎢
⎢⎣
̃𝑏1
⋮
̃𝑏𝑠
⎤
⎥
⎥⎦
̃𝑑=
⎡
⎢
⎢⎣
̃𝑑1
⋮
̃𝑑𝑠
⎤
⎥
⎥⎦
.
(79)
Consequently, the IMEX method can be written as
𝒒𝑛+1
𝐼
= 𝒒𝑛
𝐼−Δ𝑡
𝑠
∑
𝑘=1
̃𝑏𝑘
(∇⋅𝒇ex(𝒒(𝑘)) + 𝑩(𝒒(𝑘)) ⋅∇𝒒(𝑘)) + 𝑏𝑘
(∇⋅𝒇im(𝒒(𝑘)) −𝒓(𝒒(𝑘))),
(80)
with the stages evaluated at time 𝑡(𝑘) = 𝑡𝑛+ 𝑑𝑘Δ𝑡


Journal of Computational Physics 498 (2024) 112696
12
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
𝒒(𝑘)
𝐼
= 𝒒𝑛
𝐼−Δ𝑡
𝑘−1
∑
𝑖=1
̃𝑎𝑘𝑖
(∇⋅𝒇ex(𝒒(𝑖)) + 𝑩(𝒒(𝑖)) ⋅∇𝒒(𝑘)) −Δ𝑡
𝑘
∑
𝑖=1
𝑎𝑘𝑖
(∇⋅𝒇im(𝒒(𝑖)) −𝒓(𝒒(𝑖))).
(81)
To be consistent with the asymptotic limit, we apply globally stiﬄy accurate (GSA) IMEX-RK methods for the time discretization. For 
the ﬁrst order method the forward/backward Euler method is applied, for the second order method ARS(2,2,2) is used, see Table 1. 
A second order method in space is achieved by a second order reconstruction with a minmod limiter in the Rusanov numerical ﬂux 
(70). In the implicit part, central ﬁnite diﬀerences yield second order accuracy. Note that for GSA Runge Kutta methods, the ﬁnal 
update (80) coincides with the last computational stage (81) and thus does not need to be performed.
4. Asymptotic preserving property
Motivated by the analysis in Section 2.2, we consider the case 𝑀1 = 𝑀2 = 𝑀≪1. For the cases 𝑀1 ≠𝑀2 and 𝑀1 ≈1 ≥𝑀2 > 0
we refer to the study of the isentropic case performed in [21]. The principle idea is the same and the proof can be performed along 
the lines presented in [21] combined with the analysis for the case 𝑀1 = 𝑀2 presented here.
Applying an analogous asymptotic analysis as in Section 2.2 on the semi-discrete scheme consisting of (60), (63), (64), (65), 
(78) and using well-prepared initial data as deﬁned in Deﬁnition 2.2, we can prove the asymptotic preserving (AP) property for the 
semi-discrete scheme.
Theorem 4.1 (Asymptotic preserving property). The ﬁrst order RS-IMEX FV scheme consisting of the explicit part (60), the linear implicit 
elliptic system (63) and the implicit updates (64), (65) and (78) is asymptotic preserving up to 𝒪(Δ𝑡). More precisely, for well-prepared initial 
data 𝒒0 ∈Ω𝑀
𝑤𝑝the RS-IMEX FV scheme yields a consistent approximation of limit equations (40) up to 𝒪(Δ𝑡).
Proof. For the proof of the AP property we refer a reader to Appendix A.
□
We want to point out, that the 𝒪(Δ𝑡) errors arising in the velocity equation and divergence free constraint at leading order are due 
to the non-constant volume fraction in the low Mach number limit. For single phase ﬂow, i.e. the Euler equations, or homogeneous 
mixtures, i.e. constant 𝛼, we obtain a stronger result of ∇⋅𝒗𝑛+1 = 𝒪(𝑀2).
Corollary 4.2 (Asymptotic preserving property for constant volume fraction). For constant volume fraction or single phase ﬂows, we have
𝒗𝑛+1
(0) = 𝒗(1)
(0) −Δ𝑡𝒗𝑛
(0) ⋅∇𝒗𝑛
(0) −Δ𝑡
∇𝑝𝑛+1
(2)
𝜌𝑛+1 .
(82)
Moreover, the energy update
(𝜌𝑒)𝑛+1
(0) = (𝜌𝑒)𝑛
(0) −Δ𝑡((𝜌𝑒+ 𝑝)𝑛
(0)𝒗𝑛+1
(0) )
(83)
and 𝑇𝑛+1
(0)
= 𝑇RS + 𝒪(𝑀2) yield ∇⋅𝒗𝑛+1
(0) = 𝒪(𝑀2) since (𝜌𝑒 + 𝑝)𝑛
(0) is constant. Consequently, for well-prepared initial data, the RS-IMEX FV 
scheme gives a consistent approximation of the limit equations as the Mach number tends to 0 independently of Δ𝑡.
Proof. The proof can be done following the lines of the proof of Theorem 4.1. Due to 𝜌𝑛and 𝜌𝑛+1 being a convex combination of 
the states 𝜌1,RS and 𝜌2,RS, they are constant. As a consequence, we obtain (82). Further, ∇⋅𝒗𝑛+1
(0) = 𝒪(𝑀2) since (𝜌𝑒 + 𝑝)𝑛
(0) is constant. 
As the initial data are well-prepared, i.e. 𝒒0 ∈Ω𝑀
𝑤𝑝, we also obtain recursively 𝒒𝑛∈Ω𝑀
𝑤𝑝for all successive time iterations 𝑛 > 0.
□
5. Numerical results
In this section, we illustrate by numerical experiments theoretical properties of the ﬁrst and second order RS-IMEX FV scheme, 
denoted respectively by RS-IMEX1 and RS-IMEX2, proposed in Section 3. All test cases are performed under the material CFL 
condition (59) based on the eigenvalues of the explicit subsystem (55) which are of the order of the advection scale. The initial 
conditions, if not mentioned otherwise, are given in dimensional form using the transformations (15), (35) and the deﬁnition of 
the Mach numbers (16). Whenever possible, we compare the numerical results obtained with our RS-IMEX FV schemes with an 
exact or explicit reference solution of the two-ﬂuid model with single temperature (10). Note that for a fully explicit scheme, the 
CFL condition depends directly on the Mach numbers 𝑀1, 𝑀2 and on the relaxation parameters 𝜏(𝛼), 𝜏(𝒘). Therefore, a fully explicit 
scheme which treats the relaxation source terms explicitly, is only comparable to the CFL condition (59) of the RS-IMEX FV scheme 
for sonic and supersonic ﬂows 𝑀1, 𝑀2 ≥1 and slow relaxation processes 𝜏(𝛼), 𝜏(𝒘) ≫1. Thus, for test cases with well-prepared initial 
data, 𝜏(𝒘) = 𝒪(𝑀2), the RS-IMEX FV scheme allows signiﬁcantly larger time steps in low Mach number regimes. Moreover, the 
case of pressure equilibrium which can be interpreted as “instantaneous” relaxation, is impossible to resolve with a purely explicit 
scheme. Thus, also in the case of an explicit scheme a nonlinear system has to be solved for 𝛼1 to guarantee 𝑝1 = 𝑝2. This underlines 
the necessity of a semi-implicit AP scheme in such situations.


Journal of Computational Physics 498 (2024) 112696
13
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
5.1. Numerical convergence study
To verify the experimental order of convergence (EOC) we construct an exact solution of the homogeneous two-ﬂuid single 
temperature model (10) given by a stationary vortex. It is obtained by considering zero radial velocities and a constant solution in 
angular direction, i.e.
𝑣𝑟= 0,
𝑤𝑟= 0,
𝜕
𝜕𝑡(⋅) = 0,
𝜕
𝜕𝜃(⋅) = 0.
(84)
In Appendix B, the two-phase model (10) without the relaxation source terms is written in polar coordinates (118). Applying (84), it 
reduces to
𝜕𝑝
𝜕𝑟=
𝛼1𝜌1𝑣𝜃
2
1 + 𝛼2𝜌2𝑣𝜃
2
2
𝑟
,
(85a)
𝜕
𝜕𝑟
(
𝑣𝜃
2
1 −𝑣𝜃
2
2
2
+ 𝜇1 −𝜇2
)
−𝑣𝜃
(1
𝑟
𝜕
𝜕𝑟
(
𝑟𝑤𝜃
))
= 0,
(85b)
with 𝑝 = 𝛼1𝑝1 + 𝛼2𝑝2, 𝑤𝜃= 𝑣𝜃1 −𝑣𝜃2, 𝑣𝜃= 𝑐1𝑣𝜃1 + 𝑐2𝑣𝜃2. We set the phase velocities and the proﬁle for the volume fraction to
𝑣𝜃𝑙= 𝑟𝑣𝑐,𝑙exp(𝜈𝒗,𝑙(1 −𝑟2))
and
𝛼1 = 𝑐𝛼+ 𝛼𝑐exp(𝜈𝛼(1 −𝑟2)),
respectively. This yields two equations for three unknowns 𝜌1, 𝜌2 and 𝑇. To eliminate one unknown, we set 𝜌2 = 𝑐𝜌𝜌1 with 𝑐𝜌being 
constant. Then, the unknowns 𝜌1 and 𝑇can be determined via the following system of ordinary diﬀerential equations
⎛
⎜
⎜
⎜⎝
𝛼1
𝜕𝑝1
𝜕𝜌1
+ 𝛼2𝑐𝜌
𝜕𝑝2
𝜕𝜌2
𝛼1
𝜕𝑝1
𝜕𝑇+ 𝛼2
𝜕𝑝2
𝜕𝑇
𝜕𝜇1
𝜕𝜌1
−𝑐𝜌
𝜕𝜇2
𝜕𝜌2
𝜕𝜇1
𝜕𝑇−𝜕𝜇2
𝜕𝑇
⎞
⎟
⎟
⎟⎠
⎛
⎜
⎜
⎜⎝
𝜕𝜌1
𝜕𝑟
𝜕𝑇
𝜕𝑟
⎞
⎟
⎟
⎟⎠
=
⎛
⎜
⎜
⎜
⎜
⎜⎝
𝛼1𝜌1𝑣𝜃
2
1 + 𝛼2𝜌2𝑣𝜃
2
2
𝑟
−𝑝1
𝜕𝛼1
𝜕𝑟+ 𝑝2
𝜕𝛼1
𝜕𝑟
−𝜕
𝜕𝑟
(
𝑣𝜃
2
1 −𝑣𝜃
2
2
2
)
+ 𝑣𝜃
(1
𝑟
𝜕
𝜕𝑟
(
𝑟𝑤𝜃
))
⎞
⎟
⎟
⎟
⎟
⎟⎠
.
(86)
Applying the ideal gas law yields
𝜕𝑝1
𝜕𝜌1
= (𝛾1 −1)𝑐𝑣,1𝑇,
𝜕𝑝2
𝜕𝜌2
= (𝛾2 −1)𝑐𝑣,2𝑇
(87a)
𝜕𝑝1
𝜕𝑇= (𝛾1 −1)𝑐𝑣,1𝜌1,
𝜕𝑝2
𝜕𝑇= (𝛾2 −1)𝑐𝑣,2𝜌2
(87b)
𝜕𝜇1
𝜕𝜌1
=
(𝛾1 −1)𝑐𝑣,1𝑇
𝜌1
,
𝜕𝜇2
𝜕𝜌1
=
(𝛾2 −1)𝑐𝑣,2𝑇
𝜌2
(87c)
𝜕𝜇1
𝜕𝑇= (𝛾1 −1)𝑐𝑣,1 −𝑠1,
𝜕𝜇1
𝜕𝑇= (𝛾2 −1).𝑐𝑣,2 −𝑠2
(87d)
To obtain the initial condition on the computational domain [−1, 1]2, we integrate (86) numerically with RK4, starting with the 
initial data 𝜌1 = 1, 𝜌2 = 1, 𝑇= 2. The parameters in order to obtain the velocities 𝑣𝑙,𝜃and the volume fraction 𝛼1 are set as
𝑐𝛼= 0.4,
𝛼𝑐= 10−4,
𝜈𝛼= 10,
𝑣𝑐,1 = 2 ⋅10−5,
𝑣𝑐,2 = 2.5 ⋅10−5,
𝜈𝒗,1 = 15,
𝜈𝒗,2 = 14.
(88)
This setting yields two diﬀerent phase velocities and consequently a non-zero relative velocity. To obtain a vortex in the compressible 
ﬂow regime, we assign the following material parameters
𝛾1 = 7
5,
𝛾2 = 5
3,
𝑐𝑣,1 = 1,
𝑐𝑣,2 = 1.
(89)
The maximal Mach number for phase 1 is 0.62, for phase 2 it is 0.21 and the maximal mixture Mach number is 0.54. Consequently, 
the ﬂow is compressible.
Since the sound speeds depend on the magnitude of the pressures which itself depend on 𝑐𝑣,𝑙, we scale 𝑐𝑣,𝑙with one over the 
Mach number 𝑀to achieve ﬂows in a desired Mach number regime. In the next test case, we set 𝑐𝑣,𝑙∕𝑀2 which yields a maximum 
Mach number of 0.018 for the phase 1 and 0.014 for the phase 2 and the mixture Mach number of 0.016. Setting
𝛾1 = 2,
𝛾2 = 2.8,
𝑐𝑣,1 = 20,
𝑐𝑣,2 = 20,
(90)
the vortex ﬂow is now weakly compressible. Note that, according to Deﬁnition 2.2, the initial data are ill-prepared since the phase 
densities are not constant. However, we see from Tables 2 and 3 that the numerical scheme RS-IMEX2 converges with the expected 
EOC of two for both Mach number regimes. The results are obtained with a material CFL condition (59) with 𝜈= 0.25.


Journal of Computational Physics 498 (2024) 112696
14
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Table 2
Two-ﬂuid stationary vortex: 𝐿1 error and EOC for the second order RS-IMEX FV scheme in 
the compressible regime with parameters given in (89).
16
32
64
128
𝛼1
6.31E-03
—
1.15E-03
2.45
2.20E-04
2.38
4.69E-05
2.23
𝜌1
2.98E-02
—
9.44E-03
1.65
2.24E-03
2.07
3.94E-04
2.50
𝜌2
2.78E-02
—
8.21E-03
1.75
1.60E-03
2.35
2.56E-04
2.64
𝒗1,1
5.51E-02
—
1.40E-02
1.98
2.44E-03
2.51
3.33E-04
2.87
𝒗2,1
5.51E-02
—
1.40E-02
1.97
2.45E-03
2.51
3.41E-04
2.84
𝒗1,2
6.85E-02
—
1.58E-02
2.11
2.50E-03
2.65
3.41E-04
2.87
𝒗2,2
6.85E-02
—
1.58E-02
2.11
2.49E-03
2.66
3.51E-04
2.82
𝑇
4.45E-02
—
1.84E-02
1.27
3.92E-03
2.23
6.54E-04
2.58
Table 3
Two-ﬂuid stationary vortex: 𝐿1 error and EOC for the second order RS-IMEX FV scheme in 
the weakly-compressible regime with parameters given in (90).
16
32
64
128
𝛼1
6.53E-03
—
1.29E-03
2.34
2.57E-04
2.32
5.42E-05
2.24
𝜌1
6.95E-03
—
1.70E-03
2.02
4.68E-04
1.86
1.19E-04
1.97
𝜌2
2.73E-02
—
1.20E-03
4.50
3.23E-04
1.89
8.17E-05
1.98
𝒗1,1
4.11E-01
—
1.32E-02
4.95
2.09E-03
2.66
3.08E-04
2.76
𝒗2,1
4.11E-01
—
1.33E-02
4.95
2.09E-03
2.66
3.08E-04
2.76
𝒗1,2
4.17E-01
—
2.64E-02
3.98
6.14E-03
2.10
1.41E-03
2.12
𝒗2,2
4.17E-01
—
2.64E-02
3.98
6.14E-03
2.10
1.41E-03
2.12
𝑇
2.50E-02
—
3.20E-03
2.96
8.24E-04
1.95
2.01E-04
2.03
Table 4
Initial condition for the 1D Riemann problems presented in Sec-
tion 5.2 with 𝛾1 = 1.4 and 𝛾2 = 2 and 𝑐𝑣,1 = 𝑐𝑣,2 = 1 on the domain 
[0, 1] with initial jump at 𝑥 = 0.5.
Test
𝑇𝑓
state
𝛼
𝜌1
𝜌2
𝒗1,1
𝒗2,1
𝑇
RP1
0.2
left
0.3
2
1.2
0
0
1.2
right
0.3
2
2
0
0
1
RP2
0.2
left
0.7
1
2
-1
-1
1
right
0.3
1
2
1
1
1
5.2. 1D Riemann problems
To test the ﬁrst and second order versions of the RS-IMEX FV scheme in a high Mach number regime, we consider two Riemann 
Problems (RPs) for the homogeneous system (10) omitting the pressure relaxation source term acting on the volume fraction (10a)
and the friction source term in the relative velocity equation (10e). The initial conﬁguration on the domain [0, 1] is given in Table 4
and the initial jump position is located at 𝑥 = 0.5. The ﬁrst RP (RP1) consists of an initial jump in density of phase two and the 
temperature where the volume fraction is kept constant. The second RP (RP2) is a double rarefaction test with an initial jump in 
the volume fraction resulting in a discontinuous mixture density and internal energy. In Fig. 1, we compare the results for the ﬁrst 
and second order RS-IMEX FV schemes using 2000 cells and the material CFL condition (59). For the RS-IMEX1 FV scheme we set 
𝜈= 0.8 and for the RS-IMEX2 FV scheme 𝜈= 0.4. This results in Δ𝑡 = 4 ⋅10−4 and 2 ⋅10−4, respectively. The reference solution 
was computed by a second order explicit SSP-RK2 FV scheme using 10000 cells resulting in Δ𝑡 = 7.65 ⋅10−6. Note that the CFL 
condition for the explicit scheme is dictated by the fastest wave speed arising in the model which depends on the sound speeds of the 
respective phases, [31]. Moreover a comparable time step of the explicit scheme for 2000 cells is 3.3 ⋅10−5 which is 10 times smaller 
than the one used for the IMEX schemes. Since there are no shear processes present, RP1 consists of 5 waves. The wave speeds and 
positions produced by both RS-IMEX FV schemes are in good agreement with the reference solution, where the ﬁrst order scheme is 
more diﬀusive on the fast waves than the second order scheme, for which we can observe small oscillations on the outermost fast 
traveling waves, see right panel of Fig. 1. Their appearance is local and does not impair the results on the material wave which is the 
focus of the simulation and is captured accurately by both RS-IMEX FV schemes since the chosen time step is oriented towards its 
accurate capturing only. Moreover, the phenomenon of spurious oscillations around discontinuities is a known problem for higher 
order numerical schemes. To fully resolve all waves, an acoustic time step can be chosen or additional artiﬁcial viscosity can be 
added in the explicit upwind part at the cost of more diﬀusive material waves.
The wave structure of RP2 is more intricate, as can be seen in Fig. 2. This is due to the initial jump in the volume fraction. It 
consists of the contact wave associated with 𝛼1 and three waves traveling to the left of the boundaries of the domain and two waves 
to the right. Note that due to the single temperature assumption, the wave propagation is not symmetric. We can observe, that the 
ﬁrst order RS-IMEX1 FV scheme is too diﬀusive in order to capture the complicated sequence of slower waves. On the other hand, the 
second order RS-IMEX2 FV scheme shows a great improvement in the capturing of the slower waves near the initial jump position.


Journal of Computational Physics 498 (2024) 112696
15
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Fig. 1. Numerical solutions of the homogeneous Riemann problem RP1 obtained at time 𝑇𝑓= 0.2 with constant volume fraction without relaxation source terms using 
the new ﬁrst and second order RS-IMEX FV schemes. The reference solution is computed by the explicit second order SSP-RK2 FV scheme. From top left to bottom 
right: Phase densities 𝜌1, 𝜌2, phase velocities 𝒗1,1, 𝒗2,1 and temperature 𝑇. Left: Computational domain 𝑥 ∈[0, 1]. Right: Zoom on 𝑥 ∈[0.21, 0.245].


Journal of Computational Physics 498 (2024) 112696
16
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Fig. 2. Numerical solutions of the homogeneous Riemann problem RP2 obtained at time 𝑇𝑓= 0.2 with an initial jump in the volume fraction without relaxation 
source terms using the new ﬁrst and second order RS-IMEX FV schemes. The reference solution is computed by the explicit second order SSP-RK2 FV scheme. From 
top to bottom: Phase densities 𝜌1, 𝜌2, phase velocities 𝒗1,1, 𝒗2,1 and temperature 𝑇. Left: Computational domain [0, 1]. Right: Zoom on material waves.


Journal of Computational Physics 498 (2024) 112696
17
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
5.3. Advection of a bubble
We consider a diagonally advected bubble initially centered at (𝑥0, 𝑦0) = (0.5, 0.5) with the radius 𝑟0 = 0.2. The computational 
domain is set to [0, 1] × [0, 1] and is discretized by 256 × 256 rectangular mesh cells. Further we apply periodic boundary conditions. 
The velocity ﬁelds are given by
𝒗1 = (1,1)𝑇,𝒗2 = (1,1)𝑇.
(91)
The bubble of phase 1 is moved through a second phase 2 which is modeled by a change in the volume fraction given in dependence 
of the radius 𝑟 =
√
(𝑥−𝑥0)2 + (𝑦−𝑦0)2. The initial volume fraction is given by
𝛼1(𝑟,0) = (𝛼𝐿−𝛼𝑅)
arctan(−𝜃(𝑟−𝑟0
))
𝜋
+ (𝛼𝐿+ 𝛼𝑅)
2
,
(92)
where 𝛼𝐿= 0.9, 𝛼𝑅= 0.1 and 𝜃= 2000. The parameter 𝜃indicates the diﬀusivity of the interface in the initial data. To have a bubble, 
that is initially in pressure equilibrium, we set 𝜌2 such that 𝑝1 = 𝑝2, i.e.
𝜌2 =
(𝛾1 −1)𝑐𝑣,1
(𝛾2 −1)𝑐𝑣,2
𝜌1,
(93)
where 𝜌1 = 2. To ensure that the phase-pressure equilibrium holds during the simulation, we set the relaxation parameter 𝜏(𝛼) = 10−16, 
i.e. “instantaneous” pressure relaxation. Further, we set the initial temperature to 𝑇= 2. Finally, we set 𝛾1 = 1.4, 𝛾2 = 2, 𝑐𝑣,1 = 1 and 
𝑐𝑣,2 in accordance with (20) by
𝑐𝑣,2 =
𝛾1(𝛾1 −1)𝑐𝑣,1
𝛾2(𝛾2 −1)
𝒞2.
(94)
As before, 𝒞denotes the ratio between the Mach numbers and will be used to adjust the ﬂow regimes. Note, that the relative velocity 
equation for initially constant chemical potentials does not reduce to pure advection, but creates perturbations in 𝒘. Therefore, to 
decrease these perturbations which can interfere with the advection of the bubble, we assume a high friction by setting 𝜏(𝒘) = 10−8
and 𝜏(𝒘) = 10−12 depending on the Mach number regimes associated with 𝒞= 10 and 𝒞= 50, respectively. This leads to the Mach 
numbers 𝑀1,max = 1.336 and 𝑀2,max = 1.336 ⋅10−1 for the ﬁrst case and 𝑀1,max = 1.336 and 𝑀2,max = 2.67 ⋅10−2 for the second 
case. The bubble is evolved up to the ﬁnal time 𝑇𝑓= 1 when the bubble is back in its initial position. In Fig. 3 the volume fraction 
𝛼together with the mixture Mach number (16) is plotted along the diagonal for the ﬁrst and second order schemes. Both schemes 
use a material CFL condition (59) with 𝜈= 0.5 for the ﬁrst order scheme and 𝜈= 0.25 for the second order scheme. The numerical 
solutions are in good agreement with the initial data. The RS-IMEX1 FV scheme is quite diﬀusive whereas the RS-IMEX2 FV scheme 
captures well the initial conﬁguration. Note further, that the mixture Mach number changes rapidly from ≈1.23 inside the bubble 
to ≈0.38 outside of the bubble and from ≈1.24 to ≈0.36 for 𝒞= 10 and 𝒞= 50, respectively. Even though the phase Mach 
number 𝑀2 is signiﬁcantly smaller in the second case, the mixture Mach number does not change due to the averaging with respect 
to the mass fraction. It is therefore not a good indicator to the individual Mach number regimes that determine the scales in the 
model.
5.4. Kelvin Helmholtz instability
We modify a set-up from [23,22] for the single phase Euler equations to the single temperature two-phase model (10). It describes 
two phases ﬂowing in opposite directions which creates the Kelvin Helmholtz instability. We apply periodic boundary conditions and 
set the computational domain to [0, 1] × [0, 1]. The two ﬂuids are characterized by 𝛾1 = 2 and 𝛾2 = 1.4, respectively. Further 𝜌1 = 1
and 𝜌2 is set according to (93) in such a way that the initial condition is in pressure equilibrium. Furthermore, we require both ﬂuids 
to have the same Mach number. We set 𝑐𝑣,1 = 1∕𝜀2 and 𝑐𝑣,2 with 𝒞= 1 according to (94). Setting 𝑇= 12.5 with 𝜀 = 1 yields the 
maximal initial Mach number 𝑀= 10−1, and choosing 𝜀 = 0.1 yields the maximal initial Mach number 𝑀= 3 ⋅10−2. To ensure that 
the ﬂow stays in pressure equilibrium, we set 𝜏(𝛼) = 10−16 and in accordance with the well-prepared initial data, we set 𝜏(𝒘) = 𝑀2. 
Initially, we choose the same phase velocities, deﬁned as
𝒗1,1 = 𝒗2,1 =
⎧
⎪
⎪
⎨
⎪
⎪⎩
𝑣𝐿−𝑣𝑚exp((𝑦−0.25)∕𝐿),
if
0 ≤𝑦< 0.25
𝑣𝑅+ 𝑣𝑚exp(−(𝑦−0.25)∕𝐿),
if
0.25 ≤𝑦< 0.5
𝑣𝑅+ 𝑣𝑚exp((𝑦−0.75)∕𝐿),
if
0.5 ≤𝑦< 0.75
𝑣𝐿−𝑣𝑚exp(−(𝑦−0.75)∕𝐿),
if
0.75 ≤𝑦≤1
,
(95)
where 𝑣𝐿= 0.5, 𝑣𝑅= −0.5, 𝑣𝑚= (𝑣𝐿−𝑣𝑅)∕2 and 𝐿 = 0.025. In 𝑦-direction we apply an initial perturbation 𝒗1,2 = 𝒗2,2 = 10−2 sin(4𝜋𝑥)
which yields an initial relative velocity 𝒘= 0 and divergence free velocity ﬁeld ∇⋅𝒗= 0.


Journal of Computational Physics 498 (2024) 112696
18
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Fig. 3. Numerical solutions of the diagonally advected bubble obtained at time 𝑇𝑓= 1 by the ﬁrst and second order RS-IMEX FV schemes displayed along the diagonal 
from [0, 0] to [1, 1]. Top: Case 𝒞= 10. Bottom: Case 𝒞= 50. Left: Volume fraction 𝛼1. Right: Mixture Mach number 𝑀mix.
Table 5
Kelvin-Helmholtz instability: 𝐿1 error of 
the phase densities for diﬀerent Mach num-
bers computed on a mesh with 512 × 512
grid cells at ﬁnal time 𝑇𝑓= 3.
𝑀
𝜌1
𝜌2
10−1
1.896 ⋅10−3
1.327 ⋅10−3
3 ⋅10−2
5.037 ⋅10−4
3.525 ⋅10−4
The volume fraction is set as
𝛼1 =
⎧
⎪
⎪
⎨
⎪
⎪⎩
𝛼𝐿−𝛼𝑚exp((𝑦−0.25)∕𝐿),
if
0 ≤𝑦< 0.25
𝛼𝑅+ 𝛼𝑚exp(−(𝑦−0.25)∕𝐿),
if
0.25 ≤𝑦< 0.5
𝛼𝑅+ 𝛼𝑚exp((𝑦−0.75)∕𝐿),
if
0.5 ≤𝑦< 0.75
𝛼𝐿−𝛼𝑚exp(−(𝑦−0.75)∕𝐿),
if
0.75 ≤𝑦≤1
(96)
where 𝛼𝐿= 0.9, 𝛼𝑅= 0.2 and 𝛼𝑚= (𝛼𝑅−𝛼𝐿)∕8. In Fig. 4 numerical solutions computed by the second order RS-IMEX FV scheme 
for the passively transported volume fraction for the Mach numbers 10−1 and 3 ⋅10−2 are depicted. Two diﬀerent grids consisting of 
256 ×256 and 512 ×512 mesh cells and the material CFL condition (59) 𝜈= 0.25 were used. The ﬁnal time is 𝑇𝑓= 3. One can observe 
that despite dealing with a mixture of inviscid ﬂuids, the mesh reﬁnement does not yield new small scale vortices. The latter is 
typical for the Kelvin-Helmholtz instabilities in an ideal ﬂuid. This is due to the fact that we solve numerically the non-homogeneous 
system (17), i.e. physical dissipation is included due to the relative velocity equation. Therefore, only large vortices are present which 
corresponds to the frequency modes of the initial data. Moreover, since the initial data are well-prepared in the sense of (2.2), the 
𝐿1 errors in the phase densities decrease with the Mach number. We refer to Table 5 that validates the AP property of the RS-IMEX 
FV scheme.


Journal of Computational Physics 498 (2024) 112696
19
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Fig. 4. Kelvin-Helmholtz instability: numerical solutions for the passively transported volume fraction 𝛼1 obtained at time 𝑇𝑓= 3 for the two-ﬂuid single temperature 
model. The numerical solution is obtained by the new second order RS-IMEX FV scheme. Top panel: 𝑀max = 10−1. Bottom panel: 𝑀max = 3 ⋅10−2. Left column: 
256 × 256 grid. Right column: 512 × 512 grid. (For interpretation of the colors in the ﬁgure(s), the reader is referred to the web version of this article.)
6. Conclusions
We have derived an analyzed a new implicit-explicit ﬁnite volume (RS-IMEX FV) scheme for a single-temperature SHTC model. 
We note that the two-ﬂuid model allows two velocities and pressures. Further, it includes two dissipative mechanisms: phase pressure 
and velocity relaxations. In the proposed scheme these processes are treated diﬀerently. The relative velocity relaxation term is linear 
and is resolved as a part of the implicit sub-system, whereas the pressure relaxation is strongly nonlinear and therefore is treated 
separately by the Newton method.
Our RS-IMEX FV method is constructed in such a way that acoustic-type waves are linearized around a suitably chosen reference 
state (RS) and approximated implicitly in time and by means of central ﬁnite diﬀerences in space. The remaining advective-type 
waves are approximated explicitly in time and by means of the Rusanov FV method. The RS-IMEX FV scheme is suitable for all Mach 
number ﬂows, but in particular it is asymptotic preserving in the low Mach number ﬂow regimes.
Many multi-phase ﬂows, such as granular or sediment transport ﬂows, can be modeled within the single-temperature ap-
proximation. In turn, many of these ﬂows are weakly compressible and therefore impose severe time step restrictions if solved 
with a time-explicit numerical scheme. Therefore, the proposed RS-IMEX FV scheme is suitable to model various environmental 
ﬂows.
The proposed method was tested on a number of test cases for low and moderately high Mach number ﬂows demonstrating the 
capability of the scheme to properly capture both regimes. The theoretical second order accuracy of the scheme was conﬁrmed on 
a stationary vortex test case. We compared the second order scheme against its ﬁrst order variant which showed that the second 
order scheme yields more accurate approximations of discontinuities. Finally, the asymptotic preserving property was veriﬁed by 
approximating the Kelvin-Helmholtz instability with well-prepared initial data.
CRediT authorship contribution statement
Mária Lukáˇcová-Medvid’ová: Conceptualization, Methodology, Writing – original draft, Writing – review & editing. Ilya 
Peshkov: Conceptualization, Methodology, Writing – original draft, Writing – review & editing. Andrea Thomann: Conceptual-
ization, Methodology, Software, Visualization, Writing – original draft, Writing – review & editing.


Journal of Computational Physics 498 (2024) 112696
20
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
Declaration of competing interest
The authors declare that they have no known competing ﬁnancial interests or personal relationships that could have appeared to 
inﬂuence the work reported in this paper.
Data availability
Data will be made available on request.
Acknowledgements
A.T. and M.L. have been partially supported by the Gutenberg Research College, JGU Mainz. Further, M.L. is grateful for the 
support of the Mainz Institute of Multiscale Modelling. I.P. is a member of the Gruppo Nazionale per il Calcolo Scientiﬁco of the 
Istituto Nazionale di Alta Matematica (INdAM GNCS) and acknowledges the ﬁnancial support received from the Italian Ministry of 
Education, University and Research (MIUR) in the frame of the Departments of Excellence Initiative 2018–2022 attributed to the 
Department of Civil, Environmental and Mechanical Engineering (DICAM) of the University of Trento (Grant No. L.232/2016).
Appendix A. Proof of Theorem 4.1
We will show the AP property for the ﬁrst order time semi-discrete scheme (60) - (61). Indeed, to obtain a consistent approxi-
mation of the limit equations, an appropriate time discretization is essential. Thereby we will use techniques that were developed in 
the context of the AP proof for the Euler equations, see for instance [3], and for the isentropic two-phase subsystem, see [21]. For 
simplicity we consider without loss of generality 
𝜌1,ref
𝜌ref = 1 and 
𝜌2,ref
𝜌ref = 1.
Let the initial data be well-prepared, i.e. 𝒒0 ∈Ω𝑤𝑝as given in Deﬁnition 2.2. We assume that at time level 𝑡𝑛we have the Mach 
number expansion for each phase 𝑙= 1, 2
𝜌𝑛
𝑙= 𝜌𝑙,RS + 𝒪(𝑀2),
𝑇𝑛= 𝑇RS + 𝒪(𝑀2)
𝜌𝑙,RS,𝑇RS = const.,
(97)
in pressure equilibrium up to order 𝒪(𝑀3), see Assumption 2.1,
𝑝𝑛
1,(𝑘) = 𝑝𝑛
1,(𝑘),
𝑘= 0,2,
𝑝𝑛
1,(0) = 𝑝1(𝜌1,RS,𝑇RS),
𝑝𝑛
2,(0) = 𝑝2(𝜌2,RS,𝑇RS) = const.
(98)
Further, for the velocities we assume in concordance with Deﬁnition 2.2 that
𝒗𝑛= 𝒗𝑛
(0) + 𝒪(𝑀2),
∇⋅𝑢𝑛
(0) = 0,
𝒘𝑛= 𝒪(𝑀2),
𝜏(𝒘) = 𝑀2.
(99)
Moreover, we assume that the data at the next time level has the Mach number expansion (21) for the phase densities, temperature 
and mixture velocity leading to the Mach number expansions (24) for pressures, chemical potentials and internal phase energies. 
Our aim is to show that the ﬁrst order IMEX FV method yields a consistent approximation of the incompressible Euler system with 
variable volume fraction (40). To obtain this goal, we show that 𝒒𝑛+1 ∈Ω𝑀
𝑤𝑝, where the divergence free property of the velocity ﬁeld 
is fulﬁlled up to a 𝒪(Δ𝑡) term.
Plugging the expansion (97) at level 𝑡𝑛into the explicit update (60) we directly have for the volume fraction
𝛼(1)
1
= 𝛼𝑛
1 −Δ𝑡𝒗𝑛
(0) ⋅∇𝛼𝑛
1.
(100)
Rewriting equations for (𝛼1𝜌1)(1), (𝛼2𝜌2)(1) in terms of 𝜌𝒗and 𝒘and using (100), ∇⋅𝒗𝑛
(0) = 0 and 𝒘𝑛
(0) = 0, we have at leading order
𝛼(1)
1 𝜌(1)
1,(0) = 𝛼𝑛
1𝜌1,RS −Δ𝑡𝜌1,RS𝒗𝑛
(0) ⋅∇𝛼𝑛
1 −Δ𝑡∇⋅(𝜌𝑛
(0)𝑐1𝑐2𝒘𝑛
(0)) = 𝛼(1)
1 𝜌1,RS,
(101)
thus 𝜌(1)
1,(0) = 𝜌1,RS. With the same strategy we obtain 𝜌𝑛+1
1,(1) = 0. Analogously, we obtain 𝜌(1)
2,(0) = 𝜌2,RS and 𝜌(1)
2,(1) = 0. Summarizing, the 
phase densities satisfy the expansion (97) at the intermediate time level 𝑡(1). Using 𝒘𝑛
(0) = 0 and the evolution of the volume fraction 
(100), we obtain for the momentum and relative velocity equations
(𝜌𝒗)(1)
(0) = (𝜌𝒗)𝑛
(0) −Δ𝑡𝒗𝑛
(0) ⋅∇(𝜌𝒗)𝑛
(0),
(102a)
𝒘(1)
(0) = 0,
𝒘(1)
(1) = 0.
(102b)
Multiplying the energy equation in the explicit update (60) by 𝑀2 and using the notation (51), yields
(𝛼1𝜌1𝑒1)(1) + (𝛼2𝜌2𝑒2)(1) + 𝑀2(𝜌𝐸kin)(1) = (𝛼1𝜌1𝑒1)𝑛+ (𝛼2𝜌2𝑒2)𝑛+ 𝑀2(𝜌𝐸kin)𝑛+ 𝒪(𝑀2).
(103)
For the leading order terms of the internal energy, we obtain directly
(𝛼1(𝜌1𝑒1)(0) + 𝛼2(𝜌2𝑒2)(0)
)(1) = (𝛼1(𝜌1𝑒1)(0) + (𝛼2(𝜌2𝑒2)(0)
)𝑛


Journal of Computational Physics 498 (2024) 112696
21
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
which completes the analysis of the explicit part (60).
For the implicit part, we will follow the reasoning of [3], where the AP property is shown for the Euler equations analyzing the 
structure of the implicit elliptic operator. Since 𝛼1𝜌1 does not change during the implicit part, the expansion of the phase densities 
at 𝑡𝑛+1 fulﬁlls (97). Therefore, we obtain 𝜌𝑛+1
1,(0) = 𝜌1,RS and analogously 𝜌𝑛+1
2,(0) = 𝜌2,RS for the second phase density.
Next, we analyze the elliptic update of the total energy (63). Analogously to the fully discrete operators ℒ𝐼and 𝒦𝐼in (73), we 
deﬁne semi-discrete operators
𝐿ℎ= ∇⋅
((
(𝛼1𝜌1𝑒1)𝑛+ (𝛼2𝜌2𝑒2)𝑛+ 𝑀2(𝜌𝐸kin)𝑛+ 𝛼𝑛
1𝑝𝑛
1 + 𝛼𝑛
2𝑝𝑛
2
𝜌𝑛+1
)
∇(𝜙𝑛
𝑝−1)
)
,
(104)
𝐾ℎ= ∇⋅
(
𝜏(𝒘) (𝜇𝑛
1 −𝜇2
2)(𝜌𝑐1𝑐2)𝑛+1
𝜏(𝒘) + Δ𝑡(𝑐1𝑐2)𝑛+1 ∇(𝜕𝜇𝑛
RS
)
)
.
(105)
Note that with (42) we have 𝐿ℎ= 𝒪(1). From (46) and 𝜏(𝒘) = 𝒪(𝑀2) it follows 𝐾ℎ= 𝒪(𝑀2). Using the notation as in (51), we deﬁne
𝜌̌𝐸= (𝛼1𝜌1𝑒1 + 𝛼2𝜌2𝑒2 + 𝑀2𝜌𝐸kin
),
̌𝑝= (𝛼1𝑝1 + 𝛼2𝑝2)
̌𝜇= 𝜇1 −𝜇2.
(106)
Now taking into account the scaling of 𝒘𝑛given in (99), we write the implicit update for the total energy (73) as
(
𝑰−Δ𝑡2
𝑀2 (𝐿ℎ+ 𝐾ℎ)
)
(𝜌̌𝐸)𝑛+1 = (𝜌̌𝐸)𝑛−Δ𝑡∇ℎ⋅((𝜌̌𝐸+ ̌𝑝)𝑛𝒗(1)) −Δ𝑡2
𝑀2 𝐿ℎ(𝑀2𝜌𝐸𝑛
kin) −Δ𝑡2
𝑀2 𝐾ℎ(𝜌̌𝐸)𝑛
RS + 𝒪(𝑀2).
(107)
The operators 𝐿ℎand 𝐾ℎare symmetric, positive deﬁnite and the inverse of 𝐴 = 𝑰−Δ𝑡2
𝑀2 (𝐿ℎ+𝐾ℎ) exists. Consequently, system (107)
has a unique solution for any 𝑀> 0. Similar as in [3], we obtain that the eigenvalues of 𝐴−1 are 1 and 𝒪(𝑀2). Applying analogous 
arguments as in [3, Lem. 4.6], we derive
(𝜌𝑒)𝑛+1 = (𝜌𝑒)𝑛−Δ𝑡∇ℎ⋅((𝜌𝑒+ 𝛼1𝑝1 + 𝛼2𝑝2
)𝑛𝒗𝑛) + 𝒪(𝑀2).
(108)
Focusing on the leading order terms and using the evolution of the volume fraction (100), ∇⋅𝒗𝑛
(0) = 0, see (99), 𝑝𝑛
1,(0) = 𝑝𝑛
2,(0), see (98), 
and EOS (7), yields for the temperature the following expansion
(
𝛼(1)
1 𝜌1,RS𝑐𝑣,1 + 𝛼(1)
2 𝜌2,RS𝑐𝑣,2
)
𝑇𝑛+1
(0)
=(𝛼𝑛
1𝜌1,RS𝑐𝑣,1 + 𝛼𝑛
2𝜌2,RS𝑐𝑣,2
)𝑇RS −Δ𝑡𝒗𝑛
(0)
(𝜌1,RS𝑐𝑣,1 −𝜌2,RS𝑐𝑣,2
)𝑇RS∇𝛼𝑛
1 + 𝒪(𝑀2)
=
(
𝛼(1)
1 𝜌1,RS𝑐𝑣,1 + 𝛼(1)
2 𝜌2,RS𝑐𝑣,2
)
𝑇RS + 𝒪(𝑀2).
Since the factor 𝛼(1)
1 𝜌1,RS𝑐𝑣,1 + 𝛼(1)
2 𝜌2,RS𝑐𝑣,2 is positive and independent of 𝑀, we derive 𝑇𝑛+1
(0)
= 𝑇RS + 𝒪(𝑀2), thus the temperature 
has a correct asymptotic expansion. Moreover, in the limit as 𝑀→0, we obtain 𝑇= 𝑇RS. Further, the update of the relative velocity 
(64) and momentum (65) yield
𝒘𝑛+1
(0) = 0,
∇𝜇𝑛+1
(0) = 0,
∇𝜇𝑛+1
(1) = 0,
(109)
(𝜌𝒗)𝑛+1
(0) = (𝜌𝒗)(1)
(0) −Δ𝑡∇𝑝𝑛+1
(2) ,
∇𝑝𝑛+1
(0) = 0,
∇𝑝𝑛+1
(1) = 0.
(110)
Since the mass densities of the phases are not evolved at the implicit step, it holds 𝜌𝑛+1 = (𝛼1𝜌1)(1) + (𝛼2𝜌2)(1). Using the volume 
fraction, we can rewrite the momentum equation as
𝒗𝑛+1
(0) = 𝒗(1)
(0) −Δ𝑡𝜌𝑛
𝜌𝑛+1 𝒗𝑛
(0) ⋅∇𝒗𝑛
(0) −Δ𝑡
∇𝑝𝑛+1
(2)
𝜌𝑛+1
(111)
which is consistent with the low Mach number limit (40) up to a 𝒪(Δ𝑡) term. From the energy equation (61e) we obtain
(𝜌𝑒)𝑛+1
(0) = (𝜌𝑒)𝑛
(0) −Δ𝑡∇⋅
((
(𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS) + 𝑝𝑛
(0)
)
𝒗𝑛+1
(0)
)
+ 𝒪(𝑀2).
(112)
Using the deﬁnition of the internal mixture energy, we obtain
𝛼(1)
1 (𝜌𝑒)1,RS + 𝛼(1)
2 (𝜌𝑒)2,RS = 𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS
−Δ𝑡∇⋅
((
(𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS) + 𝑝𝑛
(0)
)
𝒗𝑛+1
(0)
)
+ 𝒪(𝑀2).
Applying the evolution of the volume fraction (100) and ∇𝑝𝑛
(0) = 0, we obtain
(𝛼𝑛
1 −Δ𝑡𝒗𝑛
(0) ⋅∇𝛼𝑛
1)(𝜌𝑒)1,RS + (𝛼𝑛
2 + Δ𝑡𝒗𝑛
(0) ⋅∇𝛼𝑛
1)(𝜌𝑒)2,RS = 𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS
−Δ𝑡𝒗𝑛+1
(0) ⋅∇((𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS))
−Δ𝑡
(
(𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS) + 𝑝𝑛
(0)
)
∇⋅𝒗𝑛+1
(0) + 𝒪(𝑀2),


Journal of Computational Physics 498 (2024) 112696
22
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
which reduces to
(𝒗𝑛
(0) −𝒗𝑛+1
(0) ) ⋅∇(𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS
) =
(
(𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS) + 𝑝𝑛
(0)
)
∇⋅𝒗𝑛+1
(0) + 𝒪(𝑀2).
The left hand side is of order 𝒪(Δ𝑡) and the factor (𝛼𝑛
1(𝜌𝑒)1,RS + 𝛼𝑛
2(𝜌𝑒)2,RS) + 𝑝𝑛
(0) is positive. Consequently, we obtain the result 
∇⋅𝒗𝑛+1
(0) = 𝒪(Δ𝑡) + 𝒪(𝑀2).
Finally, we apply the pressure relaxation on the volume fraction, and we obtain
𝑝𝑛+1
1
= 𝑝𝑛+1
2
−𝜏(𝛼)𝜌𝑛+1𝑀2
(
𝛼𝑛+1 −𝛼𝑛
Δ𝑡
)
.
(113)
Thus 𝑝𝑛+1
1,(0) = 𝑝𝑛+1
2,(0) and 𝑝𝑛+1
1,(2) = 𝑝𝑛+1
2,(2) since 𝑝𝑛+1
1,(1) = 𝑝𝑛+1
2,(1) = 0. Note that this is due to phase densities and the temperature fulﬁlling 
expansion (97) at the new time level 𝑡𝑛+1. Moreover, 𝜌𝑛+1( 𝛼𝑛+1−𝛼𝑛
Δ𝑡
) = 𝒪(1) with respect to the Mach number, since the time step is 
independent of the Mach number as well.
This concludes the proof.
Appendix B. Polar coordinates
We consider a continuous solution of the homogeneous part of system (10) without relaxation source terms. Let the Cartesian 
coordinates in 2D be denoted by 𝑥 = (𝑥1, 𝑥2). We deﬁne the polar coordinates in terms of radius 𝑟and angle 𝜃as
𝑥1 = 𝑟cos(𝜃),
𝑥2 = 𝑟sin(𝜃).
(114)
The velocity based quantities are deﬁned by
𝑣1 = 𝑣𝑟cos(𝜃) −𝑣𝜃sin(𝜃),
𝑤1 = 𝑤𝑟cos(𝜃) −𝑤𝜃sin(𝜃),
(115a)
𝑣2 = 𝑣𝑟cos(𝜃) + 𝑣𝜃sin(𝜃),
𝑤2 = 𝑤𝑟cos(𝜃) + 𝑤𝜃sin(𝜃).
(115b)
Using
𝜕𝑥1
𝜕𝑟= cos(𝜃),
𝜕𝑥1
𝜕𝜃= −sin(𝜃)
𝑟
,
𝜕𝑥2
𝜕𝑟= sin(𝜃),
𝜕𝑥1
𝜕𝜃= cos(𝜃)
𝑟
,
(116)
we obtain for
𝒒= (𝛼1,𝛼1𝜌1,𝛼2𝜌2,𝑣𝑟,𝑣𝜃,𝑤𝑟,𝑤𝜃,𝜌𝐸)𝑇
(117)
the following system in polar coordinates
𝜕𝛼1
𝜕𝑡+ 𝑣𝑟
𝑟
𝜕
𝜕𝑟
(𝑟𝛼1
) + 𝑣𝜃
𝑟
𝜕
𝜕𝜃𝛼1 = 0,
(118a)
𝜕(𝛼1𝜌1)
𝜕𝑡
+ 1
𝑟
𝜕
𝜕𝑟
(𝑟𝛼1𝜌1𝑣1,𝑟
) + 1
𝑟
𝜕
𝜕𝜃
(𝛼𝜌1𝑣1,𝜃
) = 0,
(118b)
𝜕(𝛼2𝜌2)
𝜕𝑡
+ 1
𝑟
𝜕
𝜕𝑟
(
𝑟𝛼2𝜌2𝑣1,𝑟
)
+ 1
𝑟
𝜕
𝜕𝜃
(
𝛼2𝜌2𝑣2,𝜃
)
= 0,
(118c)
𝜕(𝜌𝑣𝑟)
𝜕𝑡
+ 1
𝑟
𝜕
𝜕𝑟
(𝑟(𝜌𝑣𝑟
2 + 𝜌𝑐1𝑐2𝑤𝑟
2 + 𝑝))
+ 1
𝑟
𝜕
𝜕𝜃
(𝜌𝑣𝑟𝑣𝜃+ 𝜌𝑐1𝑐2𝑤𝑟𝑤𝜃
) = 𝜌𝑣𝜃
2 + 𝜌𝑐1𝑐2𝑤𝜃
2 + 𝑝
𝑟
,
(118d)
𝜕(𝜌𝑣𝜃)
𝜕𝑡
+ 1
𝑟
𝜕
𝜕𝑟
(𝑟(𝜌𝑣𝑟𝑣𝜃+ 𝜌𝑐1𝑐2𝑤𝑟𝑤𝜃
))
+ 1
𝑟
𝜕
𝜕𝜃
(
𝜌𝑣𝜃
2 + 𝑝+ 𝜌𝑐1𝑐2𝑤𝜃
2)
= −𝜌𝑣𝑟𝑣𝜃+ 𝜌𝑐1𝑐2𝑤𝑟𝑤𝜃
𝑟
,
(118e)
𝜕𝑤𝑟
𝜕𝑡+ 𝜕
𝜕𝑟
(
𝑣𝑟𝑤𝑟+ 𝑣𝜃𝑤𝜃+ (1 −2𝑐1)
𝑤2
𝑟+ 𝑤𝜃
2
2
+ 𝜇1 −𝜇2
)
+ 𝑣𝜃
(1
𝑟
𝜕
𝜕𝜃𝑤𝑟−1
𝑟
𝜕
𝜕𝑟
(𝑟𝑤𝜃
))
= 0,
(118f)
𝜕𝑤𝜃
𝜕𝑡+ 1
𝑟
𝜕
𝜕𝜃
(
𝑣𝑟𝑤𝑟+ 𝑣𝜃𝑤𝜃+ (1 −2𝑐1)𝑤𝑟
2 + 𝑤𝜃
2
2
+ 𝜇1 −𝜇2
)
+ 𝑣𝑟
(1
𝑟
𝜕
𝜕𝑟
(𝑟𝑤𝜃
) −1
𝑟
𝜕
𝜕𝜃𝑤𝑟
)
= 0,
(118g)


Journal of Computational Physics 498 (2024) 112696
23
M. Lukáˇcová-Medvid’ová, I. Peshkov and A. Thomann
𝜕(𝜌𝐸)
𝜕𝑡
+ 1
𝑟
𝜕
𝜕𝑟
(
𝑟
(
𝑣𝑟(𝜌𝐸+ 𝑝) + 𝜌
[
𝑣𝑟𝑤𝑟+ 𝑣𝜃𝑤𝜃+ (1 −2𝑐1)𝑤𝑟2 + 𝑤𝜃2
2
]
𝑐1𝑐2𝑤𝑟
))
+ 1
𝑟
𝜕
𝜕𝜃
(
𝑣𝜃(𝜌𝐸+ 𝑝) + 𝜌
[
𝑣𝑟𝑤𝑟+ 𝑣𝜃𝑤𝜃+ (1 −2𝑐1)𝑤𝑟
2 + 𝑤𝜃
2
2
]
𝑐1𝑐2𝑤𝜃
)
= 0.
(118h)
References
[1] M.R. Baer, J.W. Nunziato, A two-phase mixture theory for the deﬂagration-to-detonation transition (DDT) in reactive granular materials, Int. J. Multiph. Flow 
12 (6) (1986) 861–889.
[2] S. Balay, S. Abhyankar, M.F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, A. Dener, V. Eijkhout, W.D. Gropp, D. Karpeyev, D. Kaushik, M.G. Knepley, 
D.A. May, L. Curfman McInnes, R. Tran Mills, T. Munson, K. Rupp, P. Sanan, B.F. Smith, S. Zampini, H. Zhang, PETSc users manual, Technical Report ANL-95/11 
- Revision 3.13, Argonne National Laboratory, 2020.
[3] G. Bispen, M. Lukáˇcová-Medvid’ová, L. Yelash, Asymptotic preserving IMEX ﬁnite volume schemes for low Mach number Euler equations with gravitation, J. 
Comput. Phys. 335 (2017) 222–248.
[4] S. Boscarino, G. Russo, L. Scandurra, All Mach number second order semi-implicit scheme for the Euler equations of gas dynamics, J. Sci. Comput. 77 (2) (2018) 
850–884.
[5] S. Chiocchetti, I. Peshkov, S. Gavrilyuk, M. Dumbser, High order ADER schemes and GLM curl cleaning for a ﬁrst order hyperbolic formulation of compressible 
ﬂow with surface tension, J. Comput. Phys. 426 (2021) 109898.
[6] F. Cordier, P. Degond, A. Kumbaro, An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations, J. Comput. Phys. 231 (17) (2012) 
5685–5704.
[7] P. Degond, M. Tang, All speed scheme for the low Mach number limit of the isentropic Euler equations, Commun. Comput. Phys. 10 (1) (2011) 1–31.
[8] S. Dellacherie, Analysis of Godunov type schemes applied to the compressible Euler system at low Mach number, J. Comput. Phys. 229 (4) (2010) 978–1016.
[9] D.A. Drew, R.T. Lahey, The virtual mass and lift force on a sphere in rotating and straining inviscid ﬂow, Int. J. Multiph. Flow 13 (1) (1987) 113–121.
[10] N. Favrie, S.L. Gavrilyuk, Diﬀuse interface model for compressible ﬂuid – compressible elastic–plastic solid interaction, J. Comput. Phys. 231 (7) (2012) 
2695–2723.
[11] S.K. Godunov, An interesting class of quasilinear systems, Dokl. Akad. Nauk SSSR 139 (3) (1961) 521–523.
[12] S.K. Godunov, E.I. Romenskii, Elements of Continuum Mechanics and Conservation Laws, Kluwer Academic/Plenum Publishers, 2003.
[13] S.K. Godunov, E.I. Romensky, Thermodynamics, conservation laws and symmetric forms of diﬀerential equations in mechanics of continuous media, 95 (1995) 
19–31.
[14] H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low Mach number limit, Comput. Fluids 28 (1) (1999) 63–86.
[15] S. Jin, Eﬃcient asymptotic-preserving (AP) schemes for some multiscale kinetic equations, SIAM J. Sci. Comput. 21 (2) (1999) 441–454.
[16] K. Kaiser, J. Schütz, R. Schöbel, S. Noelle, A new stable splitting for the isentropic Euler equations, J. Sci. Comput. 70 (3) (2017) 1390–1407.
[17] S. Klainerman, A. Majda, Singular limits of quasilinear hyperbolic systems with large parameters and the incompressible limit of compressible ﬂuids, Commun. 
Pure Appl. Math. 34 (4) (1981) 481–524.
[18] R. Klein, Semi-implicit extension of a Godunov-type scheme based on low Mach number asymptotics I: one-dimensional ﬂow, J. Comput. Phys. 121 (2) (1995) 
213–237.
[19] V. Kuˇcera, M. Lukáˇcová-Medvid’ová, S. Noelle, J. Schütz, Asymptotic properties of a class of linearly implicit schemes for weakly compressible Euler equations, 
Numer. Math. 150 (1) (2022) 79–103.
[20] G. La Spina, M. de’ Michieli Vitturi, E. Romenski, A compressible single-temperature conservative two-phase model with phase transitions, Int. J. Numer. 
Methods Fluids 41 (5) (2014) 282–311.
[21] M. Lukáˇcová-Medvid’ová, G. Puppo, A. Thomann, An all Mach number ﬁnite volume method for isentropic two-phase ﬂow, J. Numer. Math. 31 (3) (2023) 
175–204.
[22] C.P. McNally, W. Lyra, J.-C. Passy, A well-posed Kelvin–Helmholtz instability test and comparison, Astrophys. J. Suppl. Ser. 201 (2) (2012) 18.
[23] F. Miczek, F.K. Röpke, P.V.F. Edelmann, New numerical solver for ﬂows at various Mach numbers, Astron. Astrophys. 576 (2015) A50.
[24] S. Noelle, G. Bispen, K.R. Arun, M. Lukáˇcová-Medvid’ová, C.-D. Munz, A weakly asymptotic preserving low Mach number scheme for the Euler equations of gas 
dynamics, SIAM J. Sci. Comput. 36 (6) (2014) B989–B1024.
[25] J.H. Park, C.D. Munz, Multiple pressure variables methods for ﬂuid ﬂow at all Mach numbers, Int. J. Numer. Methods Fluids 49 (2005) 905–931.
[26] M. Pavelka, V. Klika, M. Grmela, Multiscale Thermo-Dynamics, De Gruyter, Berlin, Boston, 2018.
[27] B. Re, R. Abgrall, A pressure-based method for weakly compressible two-phase ﬂows under a Baer–Nunziato type model with generic equations of state and 
pressure and velocity disequilibrium, Int. J. Numer. Methods Fluids 94 (8) (2022) 1183–1232.
[28] E. Romenski, A.A. Belozerov, I.M. Peshkov, Conservative formulation for compressible multiphase ﬂows, Q. Appl. Math. 74 (1) (2016) 113–136.
[29] E. Romenski, D. Drikakis, E. Toro, Conservative models and numerical methods for compressible two-phase ﬂow, J. Sci. Comput. 42 (1) (2010) 68.
[30] E. Romenski, A.D. Resnyansky, E.F. Toro, Conservative hyperbolic formulation for compressible two-phase ﬂow with diﬀerent phase pressures and temperatures, 
Q. Appl. Math. 65 (2007) 259–279.
[31] E. Romenski, E.F. Toro, Compressible two-phase ﬂows: two-pressure models and numerical methods, Comput. Fluid Dyn. J. 13 (2004) 403–416.
[32] R. Saurel, R. Abgrall, A multiphase Godunov method for compressible multiﬂuid and multiphase ﬂows, J. Comput. Phys. 150 (2) (1999) 425–467.
[33] R. Saurel, C. Pantano, Diﬀuse-interface capturing methods for compressible two-phase ﬂows, Annu. Rev. Fluid Mech. 50 (1) (2018) 105–130.
[34] A. Thomann, M. Dumbser, Thermodynamically compatible discretization of a compressible two-ﬂuid model with two entropy inequalities, J. Sci. Comput. 97 (1) 
(2023) 9.
[35] J. Zeifang, J. Schütz, K. Kaiser, A. Beck, M. Lukáˇcová-Medvid’ová, S. Noelle, A novel full-Euler low Mach number IMEX splitting, Commun. Comput. Phys. 27 
(2020) 292–320.
