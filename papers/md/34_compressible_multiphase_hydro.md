Computers and Fluids 233 (2022) 105243
Available online 24 November 2021
0045-7930/© 2021 Elsevier Ltd. All rights reserved.
Contents lists available at ScienceDirect
Computers and Fluids
journal homepage: www.elsevier.com/locate/compfluid
Numerical simulation of compressible multiphase hydrodynamic problems
using reduced five-equation model on body-fitted grids
Yao Hong a, Benlong Wang a,b, Hua Liu a,b,∗
a Department of Engineering Mechanics, Shanghai Jiao Tong University, Shanghai, 200240, China
b MOE Key Laboratory of Hydrodynamics, Shanghai Jiao Tong University, Shanghai, 200240, China
A R T I C L E
I N F O
Keywords:
Multiphase
Godunov
Body-fitted grid
Underwater explosion
Slamming
A B S T R A C T
This paper focuses on the numerical simulation of compressible multiphase hydrodynamic problems on body-
fitted mapped grids. Assuming the mixture of water and gas with homogeneous mixture properties, the
reduced five-equation model (Kapila et al., 2001) is utilized to simulate the flow. Meanwhile, to circumvent
the difficulties during computation such as preserving the volume fraction positivity, the pressure relaxation
method (Saurel et al., 2009) is adopted to improve the robustness. The method is extended by considering the
gravity effect and axisymmetric flows which are absent in the original model. A Godunov method based on
the monotone upstream-centered scheme for conservation laws (MUSCL) is utilized to discretize the governing
equations and the HLLC approximate Riemann solver is used to compute the numerical flux across a mesh
cell face. Source terms representing the effect of gravitational and axisymmetric flow are calculated by the
Runge–Kutta method. Several test cases including water–air shock tubes, dam break, underwater explosion,
water entry of a hemisphere and an oblique cylinder are presented in the paper. The results obtained agree well
with the exact solutions, experimental results, and other numerical computations, demonstrating the capability
of the current method to deal with multiphase hydrodynamic problems involving complex geometries.
1. Introduction
Multiphase problems arise in many natural and industrial situations.
For example, underwater explosions, water entry of space capsules
and seaplanes, wave impacts on coastal structures, etc. In these cir-
cumstances, the structure may undergo a violent impact in a very
short time which will challenge the strength of the structure severely.
Numerous marine and offshore structures are damaged or destroyed
every year which causes serious economic loss and threatens thousands
of lives. Hence, the accurate prediction of impact loads is needed in the
design of vessels, offshore structures, etc. In a comprehensive review
concerning the slamming problems, Dias and Ghidaglia [1] suggested
a method to classify the impact loads into three elementary loading
process (ELPs) including the direct impact (ELP1), the building jets
along the structure (ELP2), and the compression of entrapped or escap-
ing gas (ELP3) [2]. It was suggested that liquid compressibility and gas
compressibility play an important role in ELP1 and ELP3 respectively.
The direct impact ELP1 occurs when the liquid is stopped by the solid
and a hemispherical pressure wave will spread at the sound speed in
the liquid. This phenomenon is especially obvious in a high-speed water
entry problem and is studied by many researchers through different
methods [3–6]. The gas compressibility is crucial in ELP3 and it will
∗Corresponding author at: Department of Engineering Mechanics, Shanghai Jiao Tong University, Shanghai, 200240, China.
E-mail address: hliu@sjtu.edu.cn (H. Liu).
cause an oscillating load on the structure. This phenomenon could be
observed in the case of a breaking wave when the crest reaches the wall
firstly to form an enclosed air pocket [7,8].
In some cases, the effect of compressibility of fluids is enhanced. For
example, in a liquefied natural gas (LNG) ship bubbles will be entrained
into the fluid after several cycles of the flip-through phenomenon. In
water entry problems, the cushion between the structure and the free
surface will collapse into bubbles and escape into the water. Moreover,
small air bubbles do exist in the ocean due to many biological and
physical phenomena. The plumes of bubbles have void fractions of
air exceeding 10% or more in breaking waves [9]. According to the
Wood’s formula [10], the sound speed of the mixture of gas and water
is sensitive to the volume fraction of gas. A small number of bubbles
will reduce the sound speed of the mixture significantly which implies
the necessity of taking the compressibility effect into account. Mai et al.
[11] and Elhimer et al. [12] conducted experiments concerning the
aerated water entry (water with bubbles) of a plate and a cone respec-
tively. The results show that both the local pressure and impact force
are reduced considerably when bubbles are introduced into the water.
In breaking wave experiments the variation of impact loads due to the
existence of bubbles is also of interest. The problem is investigated
https://doi.org/10.1016/j.compfluid.2021.105243
Received 28 April 2019; Received in revised form 3 November 2021; Accepted 10 November 2021


Computers and Fluids 233 (2022) 105243
2
Y. Hong et al.
experimentally and numerically by Kimmoun et al. [13], Mai et al.
[8], and Bredmose et al. [7]. All the results show that the dispersed
bubbles will lower the impact loads and prolong the impact duration. In
summary, a compressible multiphase solver is needed when addressing
these hydrodynamic problems.
For a compressible two-phase flow, the Baer–Nunziato (BN) model
[14], consisting of seven equations in one-dimensional flows (two mass
conservation equations, two momentum equations, two energy equa-
tions, and one topological equation) is one option. This model describes
the flow of a two-phase mixture comprehensively and considers the
non-equilibrium state of the two fluids. The numerical methods for solv-
ing this model could be found in the work of Andrianov and Warnecke
[15], Schwendeman et al. [16], and Deledicque and Papalexandris
[17]. Some Riemann solvers such as the HLLC [18], HLLEM [19] and
Osher [20] are also developed for this model. However, the original
BN model is numerically complex and expensive to solve. A simpler
and cheaper model derived from the BN model is desired in the
present study. Depending on the spatial and temporal scales required
to resolve the fields, different models derived from the BN model could
be classified into two typical scenarios, the multifluid and multiphase
flow models. In multifluid flows, the size of both phase volumes can
be resolved by the available computational mesh, and the interface
separating the two phases is accurately captured. In multiphase flows,
one phase is finely dispersed in a carrier phase and the typical size of
the fine phase is too small for any reasonable computational grid to
resolve its fluid details [21]. Considering the appearance of bubbles
in the hydrodynamic phenomena mentioned above, the reduced five-
equation model proposed by Kapila [22] for multiphase problems is
adopted.
The Kapila model assumes the two phases sharing the same velocity
and pressure. The work of Murrone and Guillard [23] demonstrates that
this simplified model could be derived from the Baer–Nunziato model
using an asymptotic analysis in the limit of zero relaxation times. A new
deduction method could be found in the work of Saurel et al. [24]. The
Kapila model is used extensively and improved after its proposal. For
example, LeMartelot et al. [25] adopted an implicit scheme to solve
this model at a low Mach number and verified it through a cavitation
test. Pelanti and Shyue [26] improved this model by including thermal
and chemical potential relaxation terms to account for heat and mass
transfer processes. An interface sharpening method is also proposed
by So et al. [27] to improve the numerical solution. Schmidmayer et al.
[28] improved the model by including the capillary effect in the model.
Although this cheaper model reduces the computational cost, the non-
conservative terms in the volume fraction equation arise from the
pressure equilibrium condition pose other numerical difficulties during
computations. For instance, preserving the volume fraction positivity
in the presence of shocks and strong expansion waves is difficult.
Non-monotonic behavior of the sound speed will result in inaccurate
wave transmission across diffuse interfaces. Besides, non-conservative
equations are usually not able to correctly conserve physical quan-
tities. To overcome the above problems, Saurel et al. [29] proposed
a relaxation method, which is proved to be robust and accurate in
most situations. In this method, the pressure equilibrium assumption
is relaxed and a pressure non-equilibrium model is developed. The
solution of this non-equilibrium model will converge to that of the
Kapila model by a pressure relaxation procedure. Some recent work
to handle the non-conservative terms could be found in [30–32].
Although this pressure relaxation method is further developed after
its proposal [25,26], to the best knowledge of authors, there are a
few studies simulating the multiphase flows by this pressure relaxation
method on body-fitted grids. When facing the structures with complex
geometries in engineering problems, the body-fitted grid is a good so-
lution to discretize the domain. For example, in water entry problems,
the pressure exerted on the body surface could be resolved accurately
which is crucial to the estimation of impact force. Moreover, coupled
with the overlapping mesh technique, the motion of the body could be
simulated easily. For instance, Banks et al. [33] studied the flows in a
cylindrical vessel on body-fitted grids by solving the Euler equations.
Nguyen et al. [34,35] use the body-fitted grids to investigate the impact
and ricochet behavior of a water entry body by solving the three-
equation model. A body-fitted grid can be generated to follow closely
the boundaries of the physical space and it has a mapping defined
from the physical domain to a logically rectangular computational do-
main [36]. The governing equations should be rewritten when solving
problems on body-fitted grids. However, the flow can be resolved very
accurately at the boundaries. In this paper, we present a method to
simulate the multiphase flows on body-fitted grids with the pressure
relaxation method. The relaxation method is extended by considering
the effect of gravity force and axisymmetric flows, which makes it
more useful when dealing with practical engineering problems like dam
break problems and water entry problems of revolution bodies. The
current method adopts a Godunov type discretization scheme on the
computational domain and the fluxes across a mesh cell are evaluated
by the HLLC approximate Riemann solver. To improve the accuracy
of the solution, the MUSCL-Hancock method is utilized. Source terms
denoting the gravitational and axisymmetric flows are integrated by a
total variation diminishing (TVD) Runge–Kutta method. The method is
proved to be accurate and robust. We will demonstrate this through a
series of numerical tests including water–air shock tubes, dam break,
water entry of a hemisphere and an oblique cylinder. Some libraries
used in the solver such as the structured mesh generation and parallel
computing are part of the Overture set of codes [37].
The summary of this work is as follows. The reduced two-phase
model and its counterpart of the pressure relaxation method are de-
scribed in Section 2.1. The numerical discretization schemes based on
the Godunov method are presented in Section 2.2. The MUSCL-Hancock
reconstruction method used to improve the accuracy of the solution is
shown in Section 2.3. The HLLC Riemann solver to compute the flux
on the cell is presented in Section 2.4. The handling of source terms is
explained in Section 2.5 and a summary of the solution procedure is
presented in Section 2.6. In Section 3, a series of numerical tests are
presented to verify this multiphase solver. Conclusions are drawn in
Section 4.
2. Numerical method
2.1. Governing equations
Assuming the two components of the water–air mixture sharing the
same pressure and velocity, the Kapila model in 3D flows could be
described as
𝜕(𝛼𝜌)l
𝜕𝑡
+ 𝜕(𝛼𝜌)l𝑢
𝜕𝑥
+ 𝜕(𝛼𝜌)l𝑣
𝜕𝑦
+ 𝜕(𝛼𝜌)l𝑤
𝜕𝑧
= 0
𝜕(𝛼𝜌)g
𝜕𝑡
+
𝜕(𝛼𝜌)g𝑢
𝜕𝑥
+
𝜕(𝛼𝜌)g𝑣
𝜕𝑦
+
𝜕(𝛼𝜌)g𝑤
𝜕𝑧
= 0
𝜕𝜌𝑢
𝜕𝑡+ 𝜕(𝜌𝑢2 + 𝑝)
𝜕𝑥
+ 𝜕𝜌𝑢𝑣
𝜕𝑦
+ 𝜕𝜌𝑢𝑤
𝜕𝑧
= 0
𝜕𝜌𝑣
𝜕𝑡+ 𝜕𝜌𝑢𝑣
𝜕𝑥
+ 𝜕(𝜌𝑣2 + 𝑝)
𝜕𝑦
+ 𝜕𝜌𝑣𝑤
𝜕𝑧
= 0
𝜕𝜌𝑤
𝜕𝑡
+ 𝜕𝜌𝑢𝑤
𝜕𝑥
+ 𝜕𝜌𝑣𝑤
𝜕𝑦
+ 𝜕(𝜌𝑤2 + 𝑝)
𝜕𝑧
= 0
𝜕𝜌𝐸
𝜕𝑡
+ 𝜕(𝜌𝐸+ 𝑝)𝑢
𝜕𝑥
+ 𝜕(𝜌𝐸+ 𝑝)𝑣
𝜕𝑦
+ 𝜕(𝜌𝐸+ 𝑝)𝑤
𝜕𝑧
= 0
𝜕𝛼l
𝜕𝑡+ 𝑢𝜕𝛼l
𝜕𝑥+ 𝑣𝜕𝛼l
𝜕𝑦+ 𝑤𝜕𝛼l
𝜕𝑧=
𝜌g𝑐2
g −𝜌l𝑐2
l
𝜌l𝑐2
l
𝛼l +
𝜌g𝑐2g
𝛼g
( 𝜕𝑢
𝜕𝑥+ 𝜕𝑣
𝜕𝑦+ 𝜕𝑤
𝜕𝑧)
,
(1)
where 𝛼l and 𝛼g are volume fractions of water and gas, satisfying
𝛼l + 𝛼g = 1; 𝜌l, 𝜌g, and 𝜌= 𝛼l𝜌l + 𝛼g𝜌g are the densities of water, gas,
and mixture respectively; 𝑢, 𝑣, and 𝑤are the velocities of the mixture
along 𝑥, 𝑦, and 𝑧axes; 𝑡is the time; 𝐸= 𝑒+ 1
2(𝑢2 +𝑣2 +𝑤2) is the specific


Computers and Fluids 233 (2022) 105243
3
Y. Hong et al.
total energy of the mixture, where 𝑒is the specific internal energy of
the mixture; 𝑝is the pressure of mixture; 𝑐l and 𝑐g are the sound speed
velocities of water and gas respectively. For the present work, the two
components of the mixture (i.e. water and gas) follow the stiffened-gas
equation of state
𝜌𝑘𝑒𝑘= 𝑝𝑘+ 𝛾𝑘𝜋𝑘
𝛾𝑘−1
,
(2)
where the subscript 𝑘= l, g represents water and gas respectively;
constant 𝛾is a polytropic constant and constant 𝜋is a reference
pressure. The sound speed of each component can be estimated by
𝑐𝑘=
√𝛾𝑘
𝜌𝑘
(𝑝𝑘+ 𝜋𝑘),
(3)
and the sound speed of the mixture at mechanical equilibrium state
follows the formula of Wood [10]
1
𝜌𝑐2 =
𝛼l
𝜌l𝑐2
l
+
𝛼g
𝜌g𝑐2
g
.
(4)
Once the internal energy of each component is obtained, the internal
energy of the mixture 𝑒can be calculated as
𝜌𝑒= 𝛼l𝜌l𝑒l + 𝛼g𝜌g𝑒g.
(5)
With the help of Eq. (2), the resulting mixture equation of state is
𝑝=
𝜌𝑒−( 𝛼l𝛾l𝜋l
𝛾l−1 +
𝛼g𝛾g𝜋g
𝛾g−1 )
𝛼l
𝛾l−1 +
𝛼g
𝛾g−1
.
(6)
For axisymmetric flow problems involving structures like spheres and
so on, a reduced axisymmetric model is preferred sometimes. It could
not only save the computational time and resources, but also reduce the
difficulty of mesh generation. In addition, the effect of gravity may need
to be included when dealing with some problems. If the axisymmetric
flow (𝑤= 0) or gravity effect is taken into account, then a source term
−𝑣
𝑦
[
𝛼l𝜌l, 𝛼g𝜌g, 𝜌𝑢, 𝜌𝑣, 0, 𝜌𝐸+ 𝑝,
𝜌l𝑐2
l −𝜌g𝑐2
g
𝜌l𝑐2
l
𝛼l +
𝜌g𝑐2𝑔
𝛼g
]𝑇
or [0, 0, 0, −𝜌𝑔, 0, −𝜌𝑔𝑣, 0]𝑇
(7)
should be added to the right of Eq. (1), where 𝑥and 𝑦are the axial
and radial coordinates; 𝑔the gravitational acceleration is along the −𝑦
direction [38].
The Kapila model is obtained as the asymptotic limit of the BN
model in the limit of both stiff velocity and pressure relaxation. The
main difficulty of this model comes from the pressure equilibrium con-
dition, which results in the non-conservative equation for the volume
fraction. The convergence of a numerical scheme to the exact solution
is difficult because of the non-conservative terms [39]. The compress-
ibility of water is weak compared with gas, it is not easy to keep the
volume fraction positive in the presence of shocks or strong rarefaction
waves [40]. Saurel et al. [29] proposed a method to circumvent these
difficulties, in which the pressure equilibrium assumption is relaxed
and a pressure non-equilibrium model is developed. It results in a single
velocity, non-conservative hyperbolic model with two energy equations
involving relaxation terms. The hyperbolic part is first solved without
relaxation terms with a Godunov type method, then the stiff pressure
relaxation terms are solved with a nonlinear algebraic system free of
parameters. This model is a step-model to solve the Kapila model and
it has better properties for numerical approximations. For instance,
the positivity of volume fraction is preserved easily and the mixture
sound speed has a monotonic behavior. It is proved that with proper
relaxation process, solutions of the Kapila model will be recovered. The
governing equations of this new model can be expressed by
𝜕
𝜕𝑡𝐔+ 𝜕
𝜕𝑥1
𝐅𝟏(𝐔)+ 𝜕
𝜕𝑥2
𝐅𝟐(𝐔)+ 𝜕
𝜕𝑥3
𝐅𝟑(𝐔)+𝐇(𝑈)div(𝐮) = 𝐒(𝐔) + 𝐑(𝐔), (8)
where
𝐔=
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝛼l𝜌l
𝛼g𝜌g
𝜌𝐮
𝜌𝐸
𝛼l
𝛼l𝜌l𝑒l
𝛼g𝜌g𝑒g
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 𝐅𝑛(𝐔) =
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝛼l𝜌l𝑢𝑛
𝛼g𝜌g𝑢𝑛
𝜌𝑢𝑛𝐮+ 𝑝𝐞𝑛
(𝜌𝐸+ 𝑝)𝑢𝑛
𝛼l𝑢𝑛
𝛼l𝜌l𝑒l𝑢𝑛
𝛼g𝜌g𝑒g𝑢𝑛
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
,
𝐇(𝐔) =
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
𝟎
0
−𝛼l
𝛼l𝑝l
𝛼g𝑝g
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 𝐑(𝐔) =
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
𝟎
0
𝜇(𝑝l −𝑝g)
−𝑝𝐼𝜇(𝑝l −𝑝g)
𝑝𝐼𝜇(𝑝l −𝑝g)
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
, 𝑛= 1, 2, 3.
The source term 𝐒(𝐔) is absent unless an axisymmetric flow or a
problem concerning gravity is considered. Under these circumstances,
the source term 𝐒(𝐔) can be expressed as
𝐒(𝐔) = −𝑢2
𝑥2
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝛼l𝜌l
𝛼g𝜌g
𝜌𝑢1
𝜌𝑢2
0
𝜌𝐸+ 𝑝
0
𝛼l𝜌l𝑒l + 𝛼l𝑝l
𝛼g𝜌g𝑒g + 𝛼g𝑝g
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
or
⎡
⎢
⎢
⎢
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
0
−𝜌𝑔
0
−𝜌𝑔𝑢2
0
0
0
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
.
(9)
In Eqs. (8)–(9), the coordinate 𝐱= (𝑥1, 𝑥2, 𝑥3) and the velocity 𝐮=
(𝑢1, 𝑢2, 𝑢3); 𝐞𝑛is the unit vector of the 𝑛th coordinate; 𝜇is the pressure
relaxation coefficient; 𝑝𝐼=
𝜌l𝑐l𝑝g+𝜌g𝑐g𝑝l
𝜌l𝑐l+𝜌g𝑐g
is the interfacial pressure. The
solution of this new model will converge to that of the Kapila model
through a pressure relaxation process in the limit 𝜇→+∞[29]. The
phase pressure 𝑝l and 𝑝g will be equal to the mixture pressure 𝑝after
the relaxation. A detailed discussion on this is presented in Section 2.5.
This model is an interim model towards the solving of the reduced five-
equation model and equipped with a monotonic mixture sound speed
(also called frozen sound speed) [41]
𝑐2
𝑓= (𝛼𝜌)l
𝜌
𝑐2
l +
(𝛼𝜌)g
𝜌
𝑐2
g.
(10)
The numerical simulation is carried out in the computational do-
main of a mesh. In the present work, the mesh generation library of
open-source software Overture is utilized [37]. To solve Eq. (8), a
fractional-step scheme
𝐔𝑛+1
𝐢
= 𝑆𝑟(𝛥𝑡)𝑆𝑠(𝛥𝑡)𝑆ℎ(𝛥𝑡)𝐔𝑛
𝐢
(11)
is used to advance the numerical integration at point 𝐱𝑖from a time 𝑡𝑛
to 𝑡𝑛+1 = 𝑡𝑛+ 𝛥𝑡. The operator 𝑆ℎrepresents the numerical solution of
𝜕
𝜕𝑡𝐔+
𝜕
𝜕𝑥1
𝐅𝟏(𝐔) +
𝜕
𝜕𝑥2
𝐅𝟐(𝐔) +
𝜕
𝜕𝑥3
𝐅𝟑(𝐔) + 𝐇(𝐔)div(𝐮) = 0.
(12)
This equation neglects the source terms on the right side of Eq. (8).
A Godunov method is utilized to solve it and more details will be
presented in Section 2.2. The operators 𝑆𝑠and 𝑆𝑟represent the inte-
grations of
d
d𝑡𝐔= 𝐒(𝐔)
(13)
and
d
d𝑡𝐔= 𝐑(𝐔)
(14)
respectively. The operator 𝑆𝑠only occurs if the gravity effect or axisym-
metric flow is considered. The operator 𝑆𝑟corresponds to the pressure
relaxation step [29].


Computers and Fluids 233 (2022) 105243
4
Y. Hong et al.
2.2. Godunov type discretization method
The open-source library provided by Overture is used for mesh gen-
eration in the present study. In a body-fitted grid, the grid boundaries
are conformal to the body surface. However, when involving a complex
physical domain, it is not easy to generate one such grid to cover the
whole domain. For such problems, the overlapping grid consisting of
one Cartesian grid covering the background and one or more body-
fitted grids denoting the body, is used for simulation. In the overlapping
region, boundary conditions for one grid are obtained by interpolating
the solution from the other (overlapped) grid [42]. Once the mesh is
generated, the mapping defined from the physical domain 𝐱to the unit
computational domain 𝐫is known. To solve Eq. (12), one needs to
rewrite it in computational space first. The following equation
𝜕
𝜕𝑡𝐔+ 1
𝐽
𝜕
𝜕𝑟1
𝐅1(𝐔) + 1
𝐽
𝜕
𝜕𝑟2
𝐅2(𝐔) + 1
𝐽
𝜕
𝜕𝑟3
𝐅3(𝐔) + 1
𝐽𝐇(𝐔)div(𝐮) = 0 (15)
is a counterpart of Eq. (12) in the computational domain, where
𝐅𝑛(𝐔) = 𝐽⋅𝑟𝑛,𝑚𝐅𝑚(𝐔),
𝑢𝑛= 𝐽⋅𝑟𝑛,𝑚𝑢𝑚
(𝑛, 𝑚= 1, 2, 3).
In Eq. (15), 𝐽is the Jacobian of the transformation matrix [𝐱𝐫] and
𝑟𝑛,𝑚= 𝜕𝑟𝑛
𝜕𝑥𝑚. Once the grid is generated, the metrics of the mapping and
these variables are already known.
Thus the discretization of the equation 𝐔𝐢= 𝑆ℎ(𝛥𝑡)𝐔𝐢is as follows:
𝐔𝐢= 𝐔𝐢−
𝛥𝑡
𝐽𝐢𝛥𝑟1
(𝐅1,𝑖1+1∕2,𝑖2,𝑖3 −𝐅1,𝑖1−1∕2,𝑖2,𝑖3)
−
𝛥𝑡
𝐽𝐢𝛥𝑟2
(𝐅2,𝑖1,𝑖2+1∕2,𝑖3 −𝐅2,𝑖1,𝑖2−1∕2,𝑖3)
−
𝛥𝑡
𝐽𝐢𝛥𝑟3
(𝐅3,𝑖1,𝑖2,𝑖3+1∕2 −𝐅3,𝑖1,𝑖2,𝑖3−1∕2)
−
𝛥𝑡
𝐽𝐢𝛥𝑟1
𝐇𝐢(𝑢1,𝑖1+1∕2,𝑖2,𝑖3 −𝑢1,𝑖1−1∕2,𝑖2,𝑖3)
−
𝛥𝑡
𝐽𝐢𝛥𝑟2
𝐇𝐢(𝑢2,𝑖1,𝑖2+1∕2,𝑖3 −𝑢2,𝑖1,𝑖2−1∕2,𝑖3)
−
𝛥𝑡
𝐽𝐢𝛥𝑟3
𝐇𝐢(𝑢3,𝑖1,𝑖2,𝑖3+1∕2 −𝑢3,𝑖1,𝑖2,𝑖3−1∕2),
(16)
where 𝐅𝑛,𝐢𝑛±1∕2 and 𝑢𝑛,𝐢𝑛±1∕2 are contravariant numerical fluxes and
velocities at the boundary respectively as shown in Fig. 1. For example,
𝐅1,𝑖1+1∕2,𝑖2,𝑖3 and 𝑢1,𝑖1+1∕2,𝑖2,𝑖3 represent the flux and velocity on the
imaginary boundary between points (𝑖1, 𝑖2, 𝑖3) and (𝑖1 +1, 𝑖2, 𝑖3). In the
non-conservative terms, 𝐇𝐢= 𝐇(𝐔𝐢) is defined at the node following the
work of Saurel et al. [29]. The fluxes can be calculated by the HLLC
Riemann solver described in Section 2.4. If the initial states inputted
into the Riemann solver are the flow states of points (𝑖1, 𝑖2, 𝑖3) and
(𝑖1+1, 𝑖2, 𝑖3), then the discretization Eq. (16) is of first order. To improve
the accuracy of numerical results, one needs to use the extrapolation
states on the boundary of each mesh as the input of a Riemann solver.
In the present work, the MUSCL-Hancock method is adopted.
2.3. MUSCL-hancock method
The numerical flux in Eq. (16) can be calculated in many ways. In
this work, we choose the HLLC Riemann solver [43] to achieve this
target. To improve the accuracy of the solution, we use the MUSCL-
Hancock method to reconstruct the data before the HLLC Riemann
solver is called on. In the MUSCL method, the solution is assumed to
be regular which means the abundant equation concerning total energy
in Eq. (8) can be omitted. Rearranging Eq. (8) in terms of primitive
variables 𝐰= [𝛼l, 𝜌l, 𝜌g, 𝑢1, 𝑢2, 𝑢3, 𝑝l, 𝑝g]T, one may obtain
𝜕𝐰
𝜕𝑡+ 𝑎(𝐰) 𝜕𝐰
𝜕𝑥1
+ 𝑏(𝐰) 𝜕𝐰
𝜕𝑥2
+ 𝑐(𝐰) 𝜕𝐰
𝜕𝑥3
= 0,
(17)
Fig. 1. The computational stencil.
where
𝑎(𝐰) =
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝑢1
0
0
0
0
0
0
0
0
𝑢1
0
𝜌l
0
0
0
0
0
0
𝑢1
𝜌g
0
0
0
0
𝑝l−𝑝g
𝜌
0
0
𝑢1
0
0
𝛼l
𝜌
1−𝛼l
𝜌
0
0
0
0
𝑢1
0
0
0
0
0
0
0
0
𝑢1
0
0
0
0
0
𝜌l𝑐2
l
0
0
𝑢1
0
0
0
0
𝜌g𝑐2
g
0
0
0
𝑢1
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
,
𝑏(𝐰) =
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝑢2
0
0
0
0
0
0
0
0
𝑢2
0
0
𝜌l
0
0
0
0
0
𝑢2
0
𝜌g
0
0
0
0
0
0
𝑢2
0
0
0
0
𝑝l−𝑝g
𝜌
0
0
0
𝑢2
0
𝛼l
𝜌
1−𝛼l
𝜌
0
0
0
0
0
𝑢2
0
0
0
0
0
0
𝜌l𝑐2
l
0
𝑢2
0
0
0
0
0
𝜌g𝑐2
g
0
0
𝑢2
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
,
𝑐(𝐰) =
⎡
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢
⎢⎣
𝑢3
0
0
0
0
0
0
0
0
𝑢3
0
0
0
𝜌l
0
0
0
0
𝑢3
0
0
𝜌g
0
0
0
0
0
𝑢3
0
0
0
0
0
0
0
0
𝑢3
0
0
0
𝑝l−𝑝g
𝜌
0
0
0
0
𝑢3
𝛼l
𝜌
1−𝛼l
𝜌
0
0
0
0
0
𝜌l𝑐2
l
𝑢3
0
0
0
0
0
0
𝜌g𝑐2
g
0
𝑢3
⎤
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥
⎥⎦
.
Transforming from the physical domain 𝐱to computation domain 𝐫, we
have:
𝜕𝐰
𝜕𝑡+ 𝐴(𝐰) 𝜕𝐰
𝜕𝑟1
+ 𝐵(𝐰) 𝜕𝐰
𝜕𝑟2
+ 𝐶(𝐰) 𝜕𝐰
𝜕𝑟3
= 0,
(18)
𝐴(𝐰) = 𝑟1,1𝑎(𝐰) + 𝑟1,2𝑏(𝐰) + 𝑟1,3𝑐(𝐰),
𝐵(𝐰) = 𝑟2,1𝑎(𝐰) + 𝑟2,2𝑏(𝐰) + 𝑟2,3𝑐(𝐰),
𝐶(𝐰) = 𝑟3,1𝑎(𝐰) + 𝑟3,2𝑏(𝐰) + 𝑟3,3𝑐(𝐰).
For a computational cell 𝐢at time 𝑡= 𝑡𝑛, the slope vectors on the
negative and positive directions of the 𝑚th (𝑚= 1, 2, 3) axis can be
expressed by
𝛥𝑟𝑚,−
𝐢
= 𝐰𝑛
𝐢−𝐰𝑛
𝐢𝑚−1, 𝛥𝑟𝑚,+
𝐢
= 𝐰𝑛
𝐢𝑚+1 −𝐰𝑛
𝐢.
(19)
To prevent spurious oscillations in the vicinity of strong gradients, the
MINMOD limiter is used to limit the slopes as
𝛥𝑟𝑚
𝐢
=
{
max[0, min(𝛥𝑟𝑚,−
𝐢
, 𝛥𝑟𝑚,+
𝐢
)], 𝛥𝑟𝑚,+
𝐢
> 0
min[0, max(𝛥𝑟𝑚,−
𝐢
, 𝛥𝑟𝑚,+
𝐢
)], 𝛥𝑟𝑚,+
𝐢
< 0
.
(20)


Computers and Fluids 233 (2022) 105243
5
Y. Hong et al.
Fig. 2. HLLC solution structure of the Riemann problem.
Thus the boundary extrapolated values are
𝐰𝑟𝑚,−
𝐢
= 𝐰𝑛
𝐢−1
2𝛥𝑟𝑚
𝐢, 𝐰𝑟𝑚,+
𝐢
= 𝐰𝑛
𝐢+ 1
2𝛥𝑟𝑚
𝐢.
(21)
Finally they are evolved by a time step 1
2𝛥𝑡
𝐰𝑙,𝑛+1∕2
𝐢
=𝐰𝑙
𝐢+ 1
2
𝛥𝑡
𝛥𝑟1
𝐴(𝐰𝑛
𝐢)[𝐰𝑟1,−
𝐢
−𝐰𝑟1,+
𝐢
] + 1
2
𝛥𝑡
𝛥𝑟2
𝐵(𝐰𝑛
𝐢)[𝐰𝑟2,−
𝐢
−𝐰𝑟2,+
𝐢
]
+ 1
2
𝛥𝑡
𝛥𝑟3
𝐶(𝐰𝑛
𝐢)[𝐰𝑟3,−
𝐢
−𝐰𝑟3,+
𝐢
],
(22)
in which 𝑙= 𝑟𝑚, ±. These values at the cell faces are then used to com-
pute the numerical fluxes in Eq. (16). For example, 𝐰𝑟1,+
𝑖1,𝑖2,𝑖3, 𝐰𝑟1,−
𝑖1+1,𝑖2,𝑖3 are
the initial left and right states of a Riemann problem to compute the
flux 𝐅1,𝑖1+1∕2,𝑖2,𝑖3. The initial states of other fluxes can be obtained in a
similar way.
2.4. HLLC Riemann solver
With the left and right initial states obtained, we need a Riemann
solver to compute the numerical fluxes in Eq. (16). The HLLC Riemann
solver [43] is adopted to compute the flux in the present work and we
will take the calculation of 𝐅1,𝑖1+1∕2,𝑖2,𝑖3 as an example to provide more
details. The solution structure of this solver is presented in Fig. 2. Three
waves (left, contact, and right waves) separate the 𝑟1 −𝑡domain into
four regions. The flow variables in left and right regions correspond
to the results of 𝐰𝑟1,+
𝑖1,𝑖2,𝑖3 and 𝐰𝑟1−,
𝑖1+1,𝑖2,𝑖3 respectively. The variables in the
middle two regions are unknown temporarily but will be given later.
Once the parameters in the four regions are known, the flux can be
calculated depending on which region the 𝑡axis is located in, since
𝐅1,𝑖1+1∕2,𝑖2,𝑖3 = 𝐅1(𝑟1 = 0).
For convenience, we define 𝑢′
𝑛= 𝑟𝑛,𝑚𝑢𝑚. 𝑆𝐿, 𝑆𝑀, and 𝑆𝑅are the
speeds of left, contact, and right waves respectively. The wave speeds
of these three waves are estimated by
⎧
⎪
⎪
⎪
⎨
⎪
⎪
⎪⎩
𝑆𝐿= min(𝑢′
1,𝐿−𝑐𝐿
√
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3, 𝑢′
1,𝑅−𝑐𝑅
√
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3)
𝑆𝑀=
𝜌𝑅𝑢′
1,𝑅(𝑆𝑅−𝑢′
1,𝑅) −𝜌𝐿𝑢′
1,𝐿(𝑆𝐿−𝑢′
1,𝐿) + (𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3)(𝑝𝐿−𝑝𝑅)
𝜌𝑅(𝑆𝑅−𝑢′
1,𝑅) −𝜌𝐿(𝑆𝐿−𝑢′
1,𝐿)
𝑆𝑅= max(𝑢′
1,𝐿+ 𝑐𝐿
√
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3, 𝑢′
1,𝑅+ 𝑐𝑅
√
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3)
,
(23)
in which 𝑐𝐿and 𝑐𝑅are the mixture sound speed calculated by Eq. (10).
Once the wave speeds are determined, the flow state in the four regions
can be calculated as follows
⎧
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎨
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪
⎪⎩
𝛼l,∗𝐾= 𝛼l,𝐾
𝜌l,∗𝐾= 𝜌l,𝐾
𝑆𝐾−𝑢′
1,𝐾
𝑆𝐾−𝑆𝑀
𝜌g,∗𝐾= 𝜌g,𝐾
𝑆𝐾−𝑢′
1,𝐾
𝑆𝐾−𝑆𝑀
𝑢1,∗𝐾=
𝑆𝑀𝑟1,1 + 𝑢1,𝐾𝑟2
1,2 + 𝑢1,𝐾𝑟2
1,3 −𝑢2,𝐾𝑟1,1𝑟1,2 −𝑢3,𝐾𝑟1,1𝑟1,3
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3
𝑢2,∗𝐾=
𝑆𝑀𝑟1,2 + 𝑢2,𝐾𝑟2
1,1 + 𝑢2,𝐾𝑟2
1,3 −𝑢1,𝐾𝑟1,1𝑟1,2 −𝑢3,𝐾𝑟1,2𝑟1,3
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3
𝑢3,∗𝐾=
𝑆𝑀𝑟1,3 + 𝑢3,𝐾𝑟2
1,1 + 𝑢3,𝐾𝑟2
1,2 −𝑢1,𝐾𝑟1,1𝑟1,3 −𝑢2,𝐾𝑟1,2𝑟1,3
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3
𝑝∗= 𝑝𝐿+
𝜌𝐿(𝑆𝑀−𝑢′
1,𝐿)(𝑆𝐿−𝑢′
1,𝐿)
𝑟2
1,1 + 𝑟2
1,2 + 𝑟2
1,3
𝐸∗,𝐾=
𝜌𝐾𝐸𝐾(𝑢′
1,𝐾−𝑆𝐾) + 𝑝𝐾𝑢′
1,𝐾−𝑝∗𝑆𝑀
𝜌∗𝐾(𝑆𝑀−𝑆𝐾)
𝑝l,∗= (𝑝l,𝐿+ 𝜋l)
(𝛾l −1)𝜌l,𝐿−(𝛾l + 1)𝜌l,∗𝐿
(𝛾l −1)𝜌l,∗𝐿−(𝛾l + 1)𝜌l,𝐿
−𝜋l
𝑝g,∗= (𝑝g,𝐿+ 𝜋g)
(𝛾g −1)𝜌g,𝐿−(𝛾g + 1)𝜌g,∗𝐿
(𝛾g −1)𝜌g,∗𝐿−(𝛾g + 1)𝜌g,𝐿
−𝜋g
, 𝐾= 𝐿or 𝑅.
(24)
then the variables needed for estimating the numerical flux 𝐅1,𝑖1+1∕2,𝑖2,𝑖3
and velocity 𝑢1,𝑖1+1∕2,𝑖2,𝑖3 are all prepared.
2.5. Source terms and relaxation step
When simulating an axisymmetric or gravitational flow, the source
term 𝐒(𝐔) ≠𝟎. To compute the contribution of this term to the solution,
one needs to solve the Eq. (13). There are many ways to solve ordinary
differential equations. In the present paper, we use a third-order TVD
Runge–Kutta scheme [46].
In the pressure relaxation step (i.e. operator 𝑆𝑟), one needs to solve
Eq. (14) to update the volume fraction of fluid l and g. Instead of solving
ordinary differential equations, an equivalent pressure relaxation solver
is built, resulting in a nonlinear algebraic system free of relaxation
parameters [29]. During this step, (𝛼𝜌)𝑘(𝑘= l, g) is regarded as a
constant and we define a new parameter 𝜈= 1∕𝜌for convenience. To
obtain the new volume fraction 𝛼l and 𝛼g, one needs to solve
⎧
⎪
⎨
⎪⎩
𝛴(𝛼𝜌)𝑘𝜈𝑘(𝑝) = 1
𝜈𝑘(𝑝) = 𝜈0
𝑘
𝑝0
𝑘+ 𝛾𝑘𝜋𝑘+ (𝛾𝑘−1)𝑝
𝑝+ 𝛾𝑘𝜋𝑘+ (𝛾𝑘−1)𝑝
,
(25)
where the superscript denotes the initial condition and the relaxed
pressure 𝑝is the only unknown. The initial conditions are estimated
from the solution of Eq. (15). Since (𝛼𝜌)𝑘is a constant, once the
density of fluid 𝑘is obtained, its corresponding volume fraction can
be estimated. The relax pressure 𝑝computed in Eq. (25) is not equal
to the equilibrium pressure 𝑝of the mixture in Eq. (1). It is only used
to calculate the volume fraction of component fluids at the mechanical
equilibrium state. Once the volume fraction 𝛼𝑘is updated, the pressure
of the mixture at equilibrium state should be estimated by Eq. (6) to
respect the conservation of total energy.
2.6. Solution procedure
The numerical method can be summarized as follows:
a. Use the MUSCL-Hancock method described in Section 2.3 to
reconstruct the initial states at the six cell faces.


Computers and Fluids 233 (2022) 105243
6
Y. Hong et al.
Fig. 3. Comparison of the present results (symbols) and the theory solution (lines), 1000 mesh cells. The theory solutions are from the exact Riemann solver for conservative
Euler equations [43,44].
Fig. 4. Comparison of the present results (symbols) and the theory solution (lines), 1000 mesh cells. Enlarged views of density and velocity. The theory solutions are from the
exact Riemann solver for conservative Euler equations [43,44].
b. Use the HLLC Riemann solver described in Section 2.4 to compute
numerical fluxes at each mesh cell.
c. Evolve all flow variables by the discretization scheme described
in Section 2.2.
d. Compute the source term if 𝐒(𝐔) ≠𝟎. Operate the pressure
relaxation step following the method described in Section 2.5. The flow
state at one time step is updated when this step is finished.
e. Go back to the first step and continue calculation for the next
time step.
3. Numerical results
In this section, some numerical tests are conducted to demonstrate
the capability of the current method to deal with multiphase flows.
First, two 1D shock tube problems of the water–air mixture are pre-
sented. Second, the dam break test (gravitational flow) is conducted to
show the ability to deal with low Mach number problems. Last, the
underwater explosion problem (axisymmetric flow) and water entry
problems of a hemisphere (axisymmetric flow) and an oblique cylinder
are investigated.
3.1. Water–air shock tube
In a shock tube problem, the tube is separated into two parts by an
imaginary membrane in the middle and two kinds of water–air shock
tubes are considered here. In the first one, the left and right parts of
the tube are filled with nearly pure gas and water with the membrane
located at 𝑥= 0.75 m. The initial density of water is 𝜌l = 𝜌water = 1000
kg∕m3 and the parameters of EOS are 𝛾l = 𝛾water = 4.4, 𝜋l = 𝜋water =
6 × 108 Pa. The initial density of air is 𝜌g = 𝜌air = 1 kg∕m3 and the
parameters of EOS are 𝛾g = 𝛾air = 1.4, 𝜋g = 𝜋air = 0 Pa. The initial
pressure of the left and right are 109 Pa and 105 Pa respectively. Due
to the limitation of the method, the left chamber has a small volume
fraction of air 𝛼g = 𝛼air = 10−6 while the right chamber has the same
volume fraction of water [29]. In this test, the source term 𝐒(𝐔) is zero.
Comparisons between the numerical results and the exact solutions
are presented in Figs. 3–4 at time 𝑡= 240 μs. The theoretical solutions
are from the exact Riemann solver for conservative Euler equations [43,
44], which may result in the difference of shock location between
the present numerical results and theory solutions. The mesh size is
1/1000 m and CFL=0.8. The generally good agreement verifies that
the present method is capable of resolving the wave speed around the
interface under high pressure wave.


Computers and Fluids 233 (2022) 105243
7
Y. Hong et al.
Fig. 5. Comparison of the present numerical results (symbols) and the results of Murrone and Guillard [23] (line), 1000 mesh cells. Every fifth data of the present results is
plotted for clarity.
Fig. 6. Configuration of the dam break test.
3.2. Water–air mixture shock tube
In the second case, the left and right parts of the tube are filled
with water–air mixture with a membrane located at 𝑥= 0.7 m. The
other initial conditions are the same as those in the first one except for
the volume fractions. The initial volume fraction of air in the left part
is 𝛼g = 𝛼air = 0.2 while the right part has the same volume fraction of
water. In the test, the source term 𝐒(𝐔) is zero.
The comparison between the present results and the numerical
results presented by Murrone and Guillard [23] is shown in Fig. 5. The
mesh size is 1/1000 m and the stopping time is 𝑡= 200 μs. In this
test, two CFL numbers 0.8 and 0.3 are considered. All the numerical
results are in a good agreement. The oscillation of volume fraction
near the contact zone is not spurious numerical oscillation but indeed
present in the solution [23]. The good agreement shown in this test
case proves that the present method is able to resolve the multiphase
mixture problem.
3.3. Dam break
The dam break problem is a classical test widely used in the vali-
dation of the free surface flow. This test can evaluate the accuracy of
the numerical model when dealing with flows involving gravitational
force. Therefore, the gravitational term in 𝐒(𝐔) is considered in the
test. Murrone and Guillard [23] firstly investigated this test using the
Kapila multiphase model with an acoustic solver developed especially
Fig. 7. Snapshots of pressure at different time for the dam break problem (units in
Pa). The black lines represent the free surface.
for low Mach number problems. The configuration of the test is shown
in Fig. 6. The computational domain is 0.5 m wide and 0.15 m high
with a water column of size 0.06 × 0.12 m2. The basic grid consists
of 150 × 45 mesh cells and adaptive mesh refinement (AMR) [47]


Computers and Fluids 233 (2022) 105243
8
Y. Hong et al.
Fig. 8. Comparison of the front position of the column between the present results
(dots) and the experimental results of Martin and Moyce [45] (crosses).
Fig. 9. Comparison of the height of the column between the present results (dots) and
the experimental results of Martin and Moyce [45] (crosses).
with refinement level 𝑙= 2 is implemented during computation. The
final mesh size during the computation is about 0.8 mm. Under the
effect of gravity 𝑔= 9.81 m∕s−2, the water column at rest initially will
collapse as time goes on. Fig. 7 illustrates the deformation of the surface
of the water column. The front position and the height of the water
column are of particular interest during computation. To estimate the
interface between air and water, the distribution of volume fraction is
interpolated along the bottom and side wall firstly. Then the location
of a point with the volume fraction 𝛼𝑙= 0.5 is regarded as the interface
at a time step. The numerical results are presented in Figs. 8–9. The
good agreement with the experimental results of Martin and Moyce
[45] indicating the method is capable of simulating free surface flow
and the effect of gravity.
3.4. 2D spherical underwater explosion in a cylinder
The numerical method proposed by Saurel et al. [29] has a close link
with the conventional barotropic cavitation models. It was suggested
that with a certain amount of gas injected into water in advance, this
model can also simulate the cavitation phenomenon caused by the un-
derwater explosion [48]. Thus the underwater explosion of a spherical
Fig. 10. The initial set-up of the underwater explosion inside a cylinder.
bubble inside a rigid cylinder is investigated herein. This problem has
been studied by many researchers with different methods [48–50].
The axisymmetric term in 𝐒(𝐔) is included in the test. The initial
set-up of the test is shown in Fig. 10. The rigid cylinder is 0.0889 m
wide and 0.2286 m high. In the center region of the container, there is
a high-pressure spherical bubble. The diameter of the bubble is 0.03 m,
the pressure 𝑝g = 20000 bar, the density 𝜌g = 1770 kg∕m3 and the EOS
parameters are 𝛾g = 2, 𝜋g = 0. The rest of the rigid cylinder is filled
with water at atmospheric pressure. The density of water is 𝜌l = 1000
kg∕m3 and the EOS parameters are 𝛾l = 7.15 and 𝜋l = 3 × 108 Pa.
Moreover, a small volume of air 𝛼g = 0.015 is contained in the ambient
water in advance [48]. Since this test is an axisymmetric problem, only
the left half is studied during computation and it consists of 71 × 361
grids uniformly distributed. The CFL number equals 0.8. Fig. 11 shows
the pressure contours at the respective times of 30, 60, 90 and 120 μs.
The shock reflected by the side walls will interact with the expanding
bubbles and generate rarefaction waves firstly. Then the shock waves
reflected from the side walls, bottom wall, and top wall will interact
and finally form a complicated flow inside the cylinder. The evolution
of pressure at the midpoint of side wall is presented in Fig. 12. The
numerical result of Jafarian and Pishevar [50], which is based on the
one-fluid model and an exact Riemann solver, shows a slight larger first
peak pressure and an earlier second impact. However, the isentropic
one-fluid model of Liu et al. [49] presents a later second impact. The
present numerical result is close to that of Ma et al. [48], which is also
based on the Kapila model.
3.5. Water entry of a hemisphere
Slamming force is of particular interest in the design of naval and
offshore engineering. The test of the water entry of a hemisphere is
performed here. In the test, the axisymmetric term in 𝐒(𝐔) is included.
The hemisphere is fixed and the water moves upward at a prescribed
velocity [48]. Besides, in the computation the change of impact velocity
is ignored which is feasible during the slamming period [51]. The initial
set-up is shown in Fig. 13(a). The diameter of the hemisphere is 0.3 m
and the impact velocity is 𝑉= 4 m∕s. To record the time history during
water entry, two pressure sensors are installed on points A and B,
which are 4 cm and 9 cm away from the axis. The computation domain
consists of two component mapped grids are shown in Fig. 13(b)
and the element cell is quad. The inner radius of the annulus grid is
0.15 m and the outer radius is 0.55 m. The rectangular grid, 1.5 m


Computers and Fluids 233 (2022) 105243
9
Y. Hong et al.
Fig. 11. Snapshots of pressure in the cylinder at time 𝑡= 30, 60, 90 and 120 μs (units in Pa). The red lines are the surface of the bubble. (For interpretation of the references to
color in this figure legend, the reader is referred to the web version of this article.)
Fig. 12. Comparison of the pressure at the midpoint of the cylinder wall. The black line
is the present result. The symbols are the numerical results of Liu et al. [49], Jafarian
and Pishevar [50] and Ma et al. [48].
long and 1 m wide, uniformly divided into 150 × 100 grids, is set
as the background grid. Different from the coarse grids shown for
simplicity, the annulus grid is divided with several different mesh sizes:
100 × 1000, 200 × 1000, 200 × 1500, and 200 × 2000.
Figs. 14–15 show the temporal pressure distribution coefficients
given by 𝐶𝑝= 𝑝∕0.5(𝜌𝑉2) at the two points. According to the results
of convergence tests, the mesh divides the annulus into 200 × 2000
grid points is adopted, in which the smallest mesh size is smaller
than 0.5 mm. The experimental results of Backer et al. [52] and 3D
numerical results by Nguyen and Park [53] and Wang and Soares [54]
are also presented. Although all the numerical results show smaller
peak pressures compared with the experimental results, the agreement
is acceptable in general. The pressure contours at two different instant
are shown in Fig. 16. At the initial stage of impact, high pressure
occurs at the head region of the hemisphere, and the largest impact
load appears at the root of the jet. As time advances, the impact loads
become smaller and more uniform.
3.6. Oblique water entry of a cylinder
The oblique water entry of a disk cylinder is investigated to verify
the present method when dealing with three-dimensional problems.
The setup of this problem is shown in Fig. 17(a). The impact velocity
𝑉= 30.48 m∕s and the situations with entry angle 𝜃= 30◦, 45◦, 60◦, 75◦
are studied. The diameter of the cylinder is 1.0 m and the length is
2.0 m. The overlapping grid consists of four component grids. A big
rectangular grid as the background, two small rectangular grids as the
patches of the bottom and top of the cylinder, and one revolved curvi-
linear grid as the body of the cylinder. This revolved grid is divided into
100 × 50 × 120 mesh cells. The three values are the number of the grids
in the axial, normal, and circumferential directions. The smallest mesh
size is less than 1 mm. For clarity, a coarse overlapping grid for this
problem is illustrated in Fig. 17(b). The nondimensional time and drag
force coefficients are respectively defined as
𝑡∗= 𝑉𝑡
𝐷, 𝐶𝐷=
𝐹
(1∕2)𝜌𝑉2𝐴,
(26)
where 𝐹denotes the total impact force along the moving direction; 𝐴is
the frontal area of the disk cylinder; 𝐷is the diameter of the cylinder.
In Fig. 18, the present numerical results are compared with the
experimental results of Baldwin [55] and the numerical results of Park
et al. [56]. The present numerical results show a good agreement with
the numerical results of Park et al. [56]. However, all the numerical
results show a discrepancy compared with the experimental results as
time moves on. This phenomenon is caused by the neglect of the viscous
force in the governing Eq. (8). The inertial force plays the dominant
role in a short duration after the impact and the viscous force could be
omitted. As the cylinder moves downward (see Fig. 19), the effect of
viscous force will emerge gradually and therefore leads to the difference
between the experimental results and the numerical results.
Different from the situation in a laboratory, many small bubbles ex-
isted and were detected in the ocean [9]. Many reasons can contribute
to the existence of bubbles, like white capping, breaking waves, and
biological processes. The vertical water entry of a plate and a cone are
investigated by Ma et al. [57] and Elhimer et al. [12] respectively. It
was found that the bubbles inside the water will influence the dynamic
behavior of fluids considerably. The local pressure can even be reduced
by more than 50% in some circumstances. To the best knowledge of
authors, there are a few studies concerning the inclined aerated water
entry problem. To illustrate the effect of aeration on the impact loads
of the inclined cylinder, three aeration levels: 𝛼g = 0.0%, 1.0% and 5.0%
are investigated, 𝛼g denoting the volume fraction of gas. The entry
angle 𝜃= 75◦and the results are shown in Fig. 20. The drag coefficients
suggesting that the impact loads will be reduced if there are bubbles


Computers and Fluids 233 (2022) 105243
10
Y. Hong et al.
Fig. 13. (a) The initial set-up of the water entry of a hemisphere (units in mm); (b) The schematic of the overlapping grid for the water entry of a hemisphere (a coarse one is
shown for clarity).
Fig. 14. Time history of pressure at point A. (a) Numerical results of different mesh sizes. (b) Numerical results (line) compared with the results of Backer et al. [52], Nguyen
and Park [53] and Wang and Soares [54] (symbols).
Fig. 15. Time history of pressure at point B. (a) Numerical results of different mesh sizes. (b) Numerical results (line) compared with the results of Backer et al. [52], Nguyen
and Park [53] and Wang and Soares [54] (symbols).
inside the water. The reduction effect is associated with the volume
fraction of the gas. Besides, the impact duration is also prolonged
as the void fraction increases. This phenomenon could be attributed
to the compressibility of gas which will cushion the slamming force.
Therefore, a multiphase compressible solver is needed when dealing
with this kind of problem.
4. Conclusions
In this paper, the Kapila multiphase model is utilized to simulate
hydrodynamic flows on body-fitted mapped grids. We use the pres-
sure relaxation method proposed by Saurel et al. [29] to cope with
the computational difficulties concerning the non-conservative terms


Computers and Fluids 233 (2022) 105243
11
Y. Hong et al.
Fig. 16. Pressure contours of the water impact of the hemisphere at different time (units in bar). The black lines represent the free surface.
Fig. 17. (a) The set-up of the oblique water entry of a disk cylinder; (b) The schematic of the overlapping grid for the oblique water entry of a disk cylinder (a coarse one is
shown for clarity).
Fig. 18. The evolution of drag force coefficient of the cylinder at different entry angles.
The experimental results of Baldwin [55] and the numerical results of Park et al. [56]
are depicted for comparison.
caused by the stiff pressure condition. Moreover, some situations absent
in the original model like axisymmetric flow and gravity effect are
also considered. A series of tests from 1D to 3D, including shock tube,
dam break, underwater explosion, water entry of a hemisphere and an
oblique cylinder are conducted. The numerical results agree well with
exact solutions, experimental results, and other independent numerical
results, proving the ability and accuracy of such method to deal with
a variety of multiphase problems. Moreover, the implementation of
body-fitted mapped grids makes this method practical especially for
engineering problems involving geometrically complicated structures.
In the future, efforts will be made to incorporate the moving grid
technique and include the cavitation effect.
CRediT authorship contribution statement
Yao Hong: Implementation of the numerical model and compu-
tation, Writing - original draft. Benlong Wang: Discussion on the
numerical results. Hua Liu: Theoretical analysis of the multiphase
hydrodynamic flows, Discussion on the numerical results, Revising the
manuscript, Supervision.
Declaration of competing interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to
influence the work reported in this paper.
Acknowledgments
This research is financially supported by the National Natural Sci-
ence Foundation of China (Grant Nos. 11632012 and 11772195). The
support from the State Key Laboratory of Ocean Engineering (Shanghai
Jiao Tong University, China) is appreciated.


Computers and Fluids 233 (2022) 105243
12
Y. Hong et al.
Fig. 19. The surface plot for the iso-value of 𝛼g = 0.5 at different time. The entry angle 𝜃= 45◦. The top row is the side view and the bottom row is the oblique view. Left:
𝑡∗= 0.38; middle: 𝑡∗= 0.56; right: 𝑡∗= 1.02.
Fig. 20. The evolution of drag force coefficient of the cylinder at different aeration
levels. The entry angle 𝜃= 75◦. The symbols are the experimental results of Baldwin
[55]. The lines are the present results corresponding to different aeration levels.
References
[1] Dias F, Ghidaglia J-M. Slamming: Recent progress in the evaluation of impact
pressures. Annu Rev Fluid Mech 2018;50:243–73.
[2] Lafeber W, Bogaert H, Brosset L, et al. Elementary loading processes (ELP)
involved in breaking wave impacts: findings from the sloshel project. In:
The twenty-second international offshore and polar engineering conference.
International Society of Offshore and Polar Engineers; 2012.
[3] Korobkin A. Blunt-body impact on the free surface of a compressible liquid. J
Fluid Mech 1994;263:319–42.
[4] Eroshin VA, Romanenkov NI, Serebryakov IV, Yakimov YL. Hydrodynamic forces
produced when blunt bodies strike the surface of a compressible fluid. Fluid Dyn
1980;15(6):829–35.
[5] Dyment A. Compressible liquid impact against a rigid body. J Fluids Eng
2015;137(3):031102.
[6] Hong Y, Wang B, Liu H. Numerical study of hydrodynamic loads at early stage
of vertical high-speed water entry of an axisymmetric blunt body. Phys Fluids
2019;31(10):102105.
[7] Bredmose H, Bullock G, Hogg A. Violent breaking wave impacts. Part 3. Effects
of scale and aeration. J Fluid Mech 2015;765:82–113.
[8] Mai T, Mai C, Raby A, Greaves D. Aeration effects on water-structure impacts:
Part 2. Wave impacts on a truncated vertical wall. Ocean Eng 2019;186:106053.
[9] Lamarre E, Melville WK. Air entrainment and dissipation in breaking waves.
Nature 1991;351(6326):469–72.
[10] Wood AB. A textbook of sound. Macmillan; 1955, p. 37.
[11] Mai T, Greaves D, Raby A. Aeration effects on impact: drop test of a flat
plate. In: The twenty-fourth international ocean and polar engineering conference
(ISOPE2014), Vol. 3, 2014.
[12] Elhimer M, Jacques N, Alaoui AEM, Gabillet C. The influence of aeration and
compressibility on slamming loads during cone water entry. J Fluids Struct
2017;70:24–46.
[13] Kimmoun O, Ratouis A, Brosset L, et al. Influence of a bubble curtain on the
impact of waves on a vertical wall. In: The twenty-second international offshore
and polar engineering conference. International Society of Offshore and Polar
Engineers; 2012.
[14] Baer MR, Nunziato JW. A two-phase mixture theory for the deflagration-to-
detonation transition (ddt) in reactive granular materials. Int J Multiph Flow
1986;12(6):861–89.
[15] Andrianov N, Warnecke G. The Riemann problem for the Baer–Nunziato
two-phase flow model. J Comput Phys 2004;195(2):434–64.
[16] Schwendeman DW, Wahle CW, Kapila AK. The Riemann problem and a high-
resolution godunov method for a model of compressible two-phase flow. J
Comput Phys 2006;212(2):490–526.
[17] Deledicque V, Papalexandris MV. An exact Riemann solver for compressible
two-phase flow models containing non-conservative products. J Comput Phys
2007;222(1):217–45.
[18] Tokareva S, Toro EF. HLLC-type Riemann solver for the Baer–Nunziato equations
of compressible two-phase flow. J Comput Phys 2010;229(10):3573–604.
[19] Dumbser M, Balsara DS. A new efficient formulation of the HLLEM Riemann
solver for general conservative and non-conservative hyperbolic systems. J
Comput Phys 2016;304:275–319.
[20] Dumbser M, Toro EF. A simple extension of the osher Riemann solver to
non-conservative hyperbolic systems. J Sci Comput 2011;48(1–3):70–88.
[21] Saurel R, Pantano C. Diffuse-interface capturing methods for compressible
two-phase flows. Annu Rev Fluid Mech 2018;50:105–30.
[22] Kapila AK, Menikoff R, Bdzil JB, Son SF, Stewart DS. Two-phase modeling of
deflagration-to-detonation transition in granular materials: Reduced equations.
Phys Fluids 2001;13(10):3002–24.
[23] Murrone A, Guillard H. A five equation reduced model for compressible two
phase flow problems. J Comput Phys 2005;202(2):664–98.
[24] Saurel R, Chinnayya A, Carmouze Q. Modelling compressible dense and dilute
two-phase flows. Phys Fluids 2017;29(6):063301.
[25] LeMartelot S, Nkonga B, Saurel R. Liquid and liquid–gas flows at all speeds. J
Comput Phys 2013;255:53–82.
[26] Pelanti M, Shyue K-M. A mixture-energy-consistent six-equation two-phase nu-
merical model for fluids with interfaces, cavitation and evaporation waves. J
Comput Phys 2014;259:331–57.
[27] So K, Hu X, Adams NA. Anti-diffusion interface sharpening technique for
two-phase compressible flow simulations. J Comput Phys 2012;231(11):4304–23.
[28] Schmidmayer K, Petitpas F, Daniel E, Favrie N, Gavrilyuk S. A model and
numerical method for compressible flows with capillary effects. J Comput Phys
2017;334:468–96.
[29] Saurel R, Petitpas F, Berry RA. Simple and efficient relaxation methods for inter-
faces separating compressible fluids, cavitating flows and shocks in multiphase
mixtures. J Comput Phys 2009;228(5):1678–712.
[30] Abgrall R, Bacigaluppi P. Design of a second-order fully explicit residual distri-
bution scheme for compressible multiphase flows. In: International conference
on finite volumes for complex applications. Springer; 2017, p. 257–64.
[31] Bacigaluppi P, Abgrall R, Kaman T. Hybrid explicit residual distribution scheme
for compressible multiphase flows. J Phys: Conf Ser 2017;821(1):012007.


Computers and Fluids 233 (2022) 105243
13
Y. Hong et al.
[32] Abgrall R, Bacigaluppi P, Tokareva S. A high-order nonconservative approach
for hyperbolic equations in fluid dynamics. Comput & Fluids 2018;169:10–22.
[33] Banks JW, Schwendeman DW, Kapila AK, Henshaw WD. A high-resolution
godunov method for compressible multi-material flow on overlapping grids. J
Comput Phys 2007;223(1):262–97.
[34] Nguyen V-T, Nguyen NT, Phan T-H, Park W-G. Efficient three-equation two-
phase model for free surface and water impact flows on a general curvilinear
body-fitted grid. Comput & Fluids 2020;196:104324.
[35] Nguyen V-T, Phan T-H, Park W-G. Modeling and numerical simulation of ricochet
and penetration of water entry bodies using an efficient free surface model. Int
J Mech Sci 2020;182:105726.
[36] Blazek J. Computational fluid dynamics: Principles and applications. Butterworth-
Heinemann; 2015.
[37] Brown DL, Henshaw WD, Quinlan DJ. Overture: An object-oriented framework
for solving partial differential Equations. Springer Berlin Heidelberg; 1997, p.
177–84.
[38] De Böck R, Tijsseling A, Koren B. A monotonicity-preserving higher-order
accurate finite-volume method for Kapila’s two-fluid flow model. Comput &
Fluids 2019;193:104272.
[39] Abgrall R, Karni S. A comment on the computation of non-conservative products.
J Comput Phys 2010;229(8):2759–63.
[40] Saurel R, Petitpas F, Abgrall R. Modelling phase transition in metastable liquids:
Application to cavitating and flashing flows.. J Fluid Mech 2008;607:313–50.
[41] De Lorenzo M, Lafon P, Pelanti M. A hyperbolic phase-transition model
with non-instantaneous eos-independent relaxation procedures. J Comput Phys
2019;379:279–308.
[42] Henshaw WD. Solving fluid flow problems on moving and adaptive overlapping
grids. In: Parallel computational fluid dynamics 2005. Elsevier; 2006, p. 21–30.
[43] Toro EF. Riemann solvers and numerical methods for fluid dynamics. Springer
Berlin; 2013, p. 87–114.
[44] Cocchi J, Saurel R, Loraud J. Treatment of interface problems with Godunov-type
schemes. Shock Waves 1996;5(6):347–57.
[45] Martin J, Moyce W. An experimental study of the collapse of fluid columns on a
rigid horizontal plane, in a medium of lower, but comparable, density. 5.. Philos
Trans R Soc Lond Ser A Math Phys Sci 1952;244(882):312–24.
[46] Gottlieb S, Shu CW. Total variation diminishing Runge-Kutta schemes. Math
Comp 1998;67(221):73–85.
[47] Henshaw WD, Schwendeman DW. An adaptive numerical scheme for high-speed
reactive flow on overlapping grids. J Comput Phys 2003;191(2):420–47.
[48] Ma ZH, Causon DM, Qian L, Gu HB, Mingham CG, Ferrer PM. A GPU based
compressible multiphase hydrocode for modelling violent hydrodynamic impact
problems. Comput & Fluids 2015;120:1–23.
[49] Liu TG, Khoo BC, Xie WF. Isentropic one-fluid modelling of unsteady cavitating
flow. J Comput Phys 2004;201(1):80–108.
[50] Jafarian A, Pishevar A. An exact multiphase Riemann solver for compressible
cavitating flows. Int J Multiph Flow 2017;88:152–66.
[51] Ng CO, Kot SC. Computations of water impact on a two-dimensional flat-
bottomed body with a volume-of-fluid method. Ocean Eng 1992;19(4):377–93.
[52] Backer GD, Vantorre M, Beels C, Pré JD, Victor S, Rouck JD, et al. Experi-
mental investigation of water impact on axisymmetric bodies. Appl Ocean Res
2009;31(3):143–56.
[53] Nguyen VT, Park WG. A volume-of-fluid (VOF) interface-sharpening method for
two-phase incompressible flows. Comput & Fluids 2017;152.
[54] Wang S, Soares CG. Numerical study on the water impact of 3D bodies by an
explicit finite element method. Ocean Eng 2014;78(3):73–88.
[55] Baldwin J. An experimental investigation of water entry (Analysis of phe-
nomenon generated by passage of projectile into water to determine effects of
accelerations, aerodynamic configurations, and surface motions) [Ph. D. Thesis],
1972.
[56] Park M-S, Jung Y-R, Park W-G. Numerical study of impact force and ricochet
behavior of high speed water-entry bodies. Comput & Fluids 2003;32(7):939–51.
[57] Ma Z, Causon D, Qian L, Mingham C, Mai T, Greaves D, et al. Pure and aerated
water entry of a flat plate. Phys Fluids 2016;28(1):016104.
