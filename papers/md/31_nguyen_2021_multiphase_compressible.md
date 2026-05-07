International Journal of Multiphase Flow 135 (2021) 103542 
Contents lists available at ScienceDirect 
International Journal of Multiphase Flow 
journal homepage: www.elsevier.com/locate/ijmulflow 
Numerical modeling of multiphase compressible ﬂows with the 
presence of shock waves using an interface-sharpening ﬁve-equation 
model 
Van-Tu Nguyen ∗, Thanh-Hoang Phan , Warn-Gyu Park ∗
School of Mechanical Engineering, Pusan National University, Busan, Korea 
a r t i c l e 
i n f o 
Article history: 
Received 2 March 2020 
Revised 28 October 2020 
Accepted 7 December 2020 
Available online 13 December 2020 
Keywords: 
Five-equation model 
Godunov-type scheme 
Compressible ﬂows 
Multiphase ﬂows 
Interface sharpening 
Shock capturing 
a b s t r a c t 
An accurate shock- and interface-capturing method is introduced for simulations of compressible multi- 
phase ﬂows. First, an associated Godunov-type numerical scheme is established for a ﬁve-equation two- 
phase model obtained from a seven-equation model by assuming a single velocity and a single pressure 
between two phases. The computational ﬁnite-volume Riemann solver using the scheme and computing 
algorithms is presented. Next, an interface-sharpening technique (IST) is extended for the compressible 
two-phase model to improve numerical simulations and correct diffusion errors. The modiﬁed IST was 
applied as postprocessing to correct the numerical diffusion error in the solution of the discretization 
scheme while maintaining a sharp interface with a desired thickness after each time step. A mixture- 
consistent interface regularization approach of all conservative variables is combined with the IST to ob- 
tain consistent thermodynamic laws for the mixture, ensuring the consistency of the variables in the 
correction process. Several examples of ﬂuid interface simulations including shock-tube, shock-bubble in- 
teractions, and underwater explosions were performed to demonstrate the accuracy and capability of the 
proposed method. Those compressible multiphase ﬂow problems are complicated by the presence of both 
shock waves and the dynamics of interfaces. Comparisons of the numerical method with theoretical re- 
sults and experimental data indicate that the present method can simulate interface dynamics with the 
presence of shock waves and large density differences. 
© 2020 Elsevier Ltd. All rights reserved. 
1. Introduction 
Compressible multiphase ﬂows with the presence of shock 
waves are present in many industrial engineering and medicine 
applications, such as underwater explosions ( Phan et al., 2019 ; 
Wu et al., 2019 ), combustion ( Kah et al., 2015 ; Kim and Kim, 2019 ), 
breakup of liquid jets at high speeds and cavitation erosion 
( Kim et al., 2013 ; RuiHan et al., 2019 ), and high-speed aero- 
dynamics and medical treatment ( Coralic and Colonius, 2014 ; 
Jagadeesh, 2008 ). Shock waves can be considered weak or strong 
shocks, depending on the value of the pressure jumps across the 
shock result in velocity and speciﬁc mass jumps. Additionally, the 
shocks are identiﬁed as normal or oblique shocks, compression or 
rarefaction shocks, and direct or reﬂected shocks ( Pontes et al., 
2019 ). The interaction of shock waves with the presence of in- 
∗Corresponding authors. 
E-mail addresses: nguyenvantu@live.com (V.-T. Nguyen), wgpark@pusan.ac.kr 
(W.-G. Park). 
terfaces such as a spherical or cylindrical gas volume of different 
densities modiﬁes the structural shape and amplitude of wave- 
fronts by reﬂection, refraction, diffraction, and scattering, which 
deforms the interface of ﬂuids ( Haas and Sturtevant, 1987 ). In ad- 
dition, during the collapse of gas bubbles, precursor and water- 
hammer shocks that arise from the re-entrant jet formation are 
generated. The shock waves can cause high pressures on walls rel- 
atively equivalent to the incident shock, thereby causing surface 
damages known as cavitation erosion ( Johnsen and Colonius, 2009 ; 
Tiwari et al., 2015 ). Owing to the mechanical changes and effective 
usage of their applications, numerical simulations of compressible 
multiphase ﬂows that capture shock waves as well to elucidate the 
underlying ﬂuid mechanisms have been performed extensively. 
The interface between two ﬂuids is important in the model- 
ing of multiphase ﬂows. State-of-the-art algorithms for the treat- 
ment of interfaces between immiscible ﬂuids are typically based 
on approaches where the numerical diffusion at the interfaces 
is eliminated, such as interface tracking ( Benson, 1992 ), inter- 
face reconstruction ( Nguyen and Park, 2016 ; Nguyen et al., 2014 ; 
https://doi.org/10.1016/j.ijmultiphaseﬂow.2020.103542 
0301-9322/© 2020 Elsevier Ltd. All rights reserved. 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Youngs, 1982 ), front tracking ( Glimm et al., 1998 ), level set meth- 
ods ( Osher and Sethian, 1988 ), compressive schemes ( Ubbink and 
Issa, 1999 ), and tangent of hyperbola for interface capturing 
(THINC) ( Xiao et al., 2005 ). Interface tracking methods are based 
on Lagrangian approaches, which determine the interface dynam- 
ics by deforming meshes to follow the interface location. The inter- 
face can remain sharp, but when the interface has a large deforma- 
tion and the mesh suffers large distortions, the approaches become 
ineﬃcient and eventually fail. Meanwhile, volume-of-ﬂuid (VOF)- 
based interface reconstruction and front tracking methods can de- 
termine the interface and restore a sharp proﬁle; they appear to 
be eﬃcient for complex and breaking interfaces ( Dang et al., 2019 ; 
Nguyen and Park, 2016 ; Vu et al., 2013 ). The interface position 
is determined to calculate the density ﬁeld and solve the ﬂuid 
dynamics equations for incompressible ﬂows; however, for com- 
pressible ﬂows, using only the interface position is insuﬃcient 
to solve both the density and internal energy of each ﬂuid in a 
mixed cell and ensure the consistency of thermodynamic laws for 
the mixture. Level-set methods have been developed for the ac- 
curate simulation of multiphase ﬂows; they are simple but less 
accurate, and reinitialization techniques must be used to enhance 
their precision ( Gibou et al., 2018 ; Osher and Sethian, 1988 ). Ad- 
ditionally, the methods require complex thermodynamic manage- 
ment (the ghost ﬂuid method) at the interface for ﬂows with high 
densities or pressure ratios ( Fedkiw et al., 1999 ). The compres- 
sive and THINC schemes are known as algebraic VOF approaches. 
The compressive methods use the information of the orientation 
of the interface to compute the ﬂuxes for the advection equation 
by using a high-resolution scheme and a high-order downwinding 
scheme. Although the methods are highly compressive and capa- 
ble of maintaining a sharp interface, they are often inclined to dis- 
tort or wrinkle the interface. Some improved techniques have been 
developed to alleviate the issue ( Heyns et al., 2013 ; Zhang et al., 
2014 ). Even though the techniques maintain a better-deﬁned inter- 
face shape while preventing artiﬁcial smearing of the interface, the 
accuracy of compressive algebraic VOF methods is generally found 
to be about an order of magnitude lower than the state of the art 
geometric VOF methods. THINC methods are relatively new meth- 
ods within algebraic VOF where a hyperbolic-tangent proﬁle is as- 
sumed for the phase indicator function within the cell containing 
the interface. The ﬂuxes of the VOF equation are computed alge- 
braically based on the proﬁle without geometrical reconstruction 
and the method can obtain accuracy close to geometric VOF meth- 
ods. The methods are widely used for the analysis of incompress- 
ible ﬂows and have been extended for compressible ﬂows ( Liu and 
Hu, 2017 ; Shyue and Xiao, 2014 ). There is still an open question 
that needs rigorous tests to assess both the robustness and perfor- 
mance of the methods for simulation of large-scale realistic prob- 
lems. 
Discontinuities of shock waves and interfaces between com- 
pressible ﬂuids with high-density differences can cause oscillations 
and large numerical diffusion errors, which become challenging 
for numerical simulations. By considering the ﬂow across a nor- 
mal shock wave and the density and pressure distribution across 
the shock, a discontinuous increase in density and pressure is evi- 
dent across the shock. If the nonconservation form of the govern- 
ing equations (primitive variable ﬂow models) were used to cal- 
culate this ﬂow, where the primary dependent variables are the 
primitive variables, such as density and pressure, then the equa- 
tions would exhibit a large discontinuity in the variables. It is chal- 
lenging to use the primitive variable models when resolving strong 
shock waves to maintain the correct shock speed; consequently, 
the shocks may appear in the wrong location and the solution may 
even become unstable (more discussions can be found in Chapter 
2 of the book ( Anderson, 2009 )). In fact, the use of the conser- 
vation form of the equations is important for the shock-capturing 
method as the conservation form of the equations does not exhibit 
any discontinuity in the dependent variables across the shock, and 
the computed ﬂow-ﬁeld results in single-phase ﬂows are generally 
smooth and stable. However, in multiphase ﬂows, fully conserva- 
tive ﬂow models are challenging at material interfaces where non- 
physical spurious oscillations inevitably occur owing to a nonphys- 
ical pressure update or negative volume fraction during numerical 
computations ( Abgrall, 1996 ; Abgrall and Karni, 2001 ). A nonlin- 
ear artiﬁcial diffusivity can be locally added in space to reduce the 
nonphysical spurious oscillations, but the oscillations are still large 
in the simulations ( Kawai and Terashima, 2011 ). In addition, to ob- 
tain a sharper interface resolution, the high-order schemes (e.g., 
weighted essentially non-oscillatory (WENO) schemes ( Jiang and 
Shu, 1996 )) are often used, but results obtained from the method 
have more numerical oscillations. Five-order ( Taylor et al., 2007 ) 
and six-order ( Hu et al., 2010 ) WENO-type schemes have been de- 
veloped and results reveal the schemes can reduce the dissipation 
signiﬁcantly. Moreover, such high-order schemes can be computa- 
tionally expensive in multiple dimensions, and stopping the pro- 
gressively more severe smearing for a longer simulation time is 
generally still not easy. Quasi-conservative ﬂow models known as 
the diffuse interface method (DIM), which combine the conserva- 
tion laws with a nonconservative scalar (volume fraction or other 
material property) advection equation, have been proven proﬁcient 
and simpler ( Goncalvès, 2013 ; Lin et al., 2017 ; Saurel and Pan- 
tano, 2018 ; Thornber et al., 2018 ). 
DIM models are suitable mathematical models that beneﬁt the 
numerical simulation of multiphase ﬂows. The most general form 
of DIM is the two-phase model ( Baer and Nunziato, 1986 ) that 
comprises at least seven equations, including two mass conserva- 
tion equations, two momentum equations, two energy equations, 
and one advection equation. The system is unconditionally hyper- 
bolic such that the system can treat both multiphase mixtures 
as well as interface problems between pure ﬂuids. The system of 
the two-ﬂuid model is closed by the use of its equation of states 
for each phase, this allows to predict ﬂuid ﬂows characterized by 
different thermodynamics. The two-ﬂuid model has advanced by 
the use of relaxation terms, i.e., inﬁnite relaxation parameters, in- 
stantaneous pressure, and velocity equilibrium, which enables the 
numerical treatment of interface problems ( Allaire et al., 2002 ; 
Ansari and Daramizadeh, 2013 ; Ha et al., 2015 ; Kapila et al., 2001 ; 
Kreeft and Koren, 2010 ; Murrone and Guillard, 2005 ; Nguyen and 
Dumbser, 2015 ; Richard et al., 2009 ). However, these models con- 
tain a large number of propagation waves, therefore it is still dif- 
ﬁcult to ﬁnd robust and eﬃcient methods to numerically solve 
this system. Also, their results are sensitive to the relaxation pro- 
cedures, leading to unstable issues in simulation. Less expensive 
DIM models, such as six-, ﬁve-, and three-equation models re- 
duced from the full seven-equation model, have been intensively 
developed and widely applied for various compressible multi- 
phase ﬂows ( Allaire et al., 2002 ; Ansari and Daramizadeh, 2013 ; 
Kapila et al., 2001 ; Kreeft and Koren, 2010 ; Murrone and Guil- 
lard, 2005 ; Nguyen et al., 2020 ; Richard et al., 2009 ). The three- 
equation two-phase model for free surface ﬂows has been im- 
plemented in a multidimensional general curvilinear coordinate 
framework with the novel idea of solving the governing equations 
only in the liquid regions, instead of solving the entire compu- 
tational domain as in classical Euler approaches ( Nguyen et al., 
2020 ). The six-equation model solves two mass balance equations 
(one for each of the phases), a momentum equation, two en- 
ergy equations (one for each of the phases), and a volume frac- 
tion advection equation. The ﬁve-equation models are similar to 
the six-equation models; however, they were obtained by assum- 
ing both single velocity and single pressure between two phases. 
Multiple alternatives to the ﬁve-equation models are available, in 
which the differences between these models are the different se- 
2 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
lection of equations of motion range, from conserving the total 
mass of the two ﬂuids versus the mass of each to transport- 
ing alternate scalar quantities ( Allaire et al., 2002 ; Kreeft and Ko- 
ren, 2010 ; Murrone and Guillard, 2005 ; Thornber et al., 2018 ). In 
summary, a ﬁve-equation model appears to be the preferred option 
of diffuse-interfaces for the simulation of compressible two-phase 
ﬂows with immiscible ﬂuids. As aforementioned, it is cast in a 
quasi-conservative form such that when standard shock-capturing 
schemes are used, the required physical quantities are conserved 
and nonphysical spurious oscillations at the material interfaces can 
be avoided. In addition, the advection equation for the volume 
fraction is not written in the conservative form and can be used 
to eﬃciently determine the position of the material interfaces be- 
tween two ﬂuids ( Nguyen and Park, 2017 ). 
Solving DIM models using standard shock-capturing schemes 
can capture unsteady shocks without nonphysical spurious os- 
cillations; however, the interfaces can become diffused and dis- 
torted without the appropriate numerical regularization, and key 
ﬂow features can be lost completely. Therefore, special treatments 
are required to reduce or correct this error. A combination of an 
interface-sharpening function and density correction is introduced 
as the source terms of advection and mass conservation equations. 
The correction algorithm is formulated based on a combination of 
both compressive and expansive terms with a stopping criterion, 
therefore that can restrict the thickness of the numerical diffused 
interface ( Shukla et al., 2010 ). However, for compressible ﬂows, 
thermodynamics is important and adds one more level of diﬃculty 
to an already complex problem in two-phase ﬂows, i.e., the model 
must maintain thermodynamic consistency at the interface. The 
correction algorithm of the interface without considering a regular- 
ization approach for all variables may cause inconsistent thermo- 
dynamic laws for the mixture. Correcting only the scale function 
and density is not suﬃciently compatible with the thermodynamic 
mixture models in the interface zone, which can result in the ac- 
cumulation of errors in space and time that are far from the in- 
terface. Anti-diffusion techniques formulated based on the low vis- 
cosity to prevent smear and diffusion errors suggested in ( So et al., 
2012 ) may satisfy this condition; however, based on low viscosity 
only, this technique cannot restrict the thickness of the numeri- 
cal diffused interface as the correction algorithm in ( Shukla et al., 
2010 ). In addition, the technique is intimately related to the un- 
derlying numerical scheme and it is therefore diﬃcult to general- 
ize to different discretizations, such as an increase in the order of 
accuracy. Recently, a general regularization approach for arbitrary 
numerical schemes proposed by Tiwari et al. ( Tiwari et al., 2013 ) 
can obtain consistent thermodynamic laws for the mixture and re- 
duce the interface diffusion error. Nevertheless, in this approach, 
the interface-sharpening function added in the source term of the 
advection equation can only reduce the interface diffusion error 
but cannot maintain a constant interface thickness everywhere at 
all times, which is a necessary feature for capturing discontinu- 
ities. Maintaining the interface thickness within several mesh cells 
is important for modeling shock. It eases the computation of shock 
or contact discontinuities in compressible two-phase applications 
and allows shock and contact discontinuity formation to be accu- 
rately predicted ( Saurel and Lemetayer, 2001 ). 
In this study, a ﬁve-equation two-phase model and its assump- 
tion of mixture rules similar to those in ( Allaire et al., 2002 ) were 
used for simulating the interfaces between compressible ﬂuids. 
Unlike the transport (advection) equation used in ( Allaire et al., 
2002 ), a Kapila term was considered in this study for the com- 
patibility of the assumption that the material derivatives of the 
phase entropies were zero ( Kapila et al., 2001 ; Murrone and Guil- 
lard, 2005 ). A high-resolution Godunov-type numerical scheme, 
which enables the solution of a problem called the Riemann prob- 
lem ( Ivings et al., 1998 ) to be obtained, was established for the 
ﬁve-equation two-phase model. The characteristics of the Jacobian 
matrix of the system was applied to construct the solution to the 
Riemann problem ( Cuong and Thanh, 2017 ). A high-order ﬁnite 
volume scheme based on monotone upstream-centered schemes 
for conservation law reconstruction (MUSCL) with limiters was 
used to extrapolate the Riemann variables at the cell faces. A third- 
order total variation diminishing Runge–Kutta scheme was adopted 
to obtain a time-accurate solution. The interface-sharpening tech- 
nique (IST) developed in ( Nguyen and Park, 2017 ) was extended 
for the simulation of compressible two-phase ﬂows with immis- 
cible ﬂuids by combining it with the mixture-consistent interface 
regularization approach of all the conservative variables proposed 
in ( Tiwari et al., 2013 ) to develop a postprocessing correction al- 
gorithm to improve numerical simulations. This correction process 
was applied as postprocessing to correct the numerical diffusion 
error in the solution of the discretization scheme, while maintain- 
ing a sharp interface and restricting the interface with the desired 
constant thickness after each time step. Consequently, the present 
method can (i) predict shock and contact discontinuity formation 
in two-phase compressible ﬂows with oscillation-free behaviors at 
material interfaces, (ii) obtain consistent thermodynamic laws for 
the mixture and ensure the consistency of the variables in the cor- 
rection process, (iii) correct the numerical diffusion error in the so- 
lution of the discretization scheme and maintain a sharp interface 
with a desired constant thickness. 
2. Mathematical formulations 
2.1. Governing equation 
The physical approach used in this study is based on a ﬁve- 
equation model for simulating interfaces between two immiscible 
compressible ﬂuids. This model is based on the typical conserva- 
tion laws for the mixture including mass, momentum, and energy 
balance, and supplemented with one advection equation. The mass 
balance equation in each phase is considered to discretely conserve 
the mass of each phase; therefore, it is always mass conservative 
with respect to each phase. The supplementation of the advection 
equation results in the sustained important property of phase mass 
conservation regardless of the numerical treatment of the color 
function used. The ﬁve-equation model is written as follows: 
∂ 
∂t ( α1 ρ1 ) + ∇ · ( α1 ρ1 u ) = 0 , 
(1) 
∂ 
∂t ( α2 ρ2 ) + ∇ · ( α2 ρ2 u ) = 0 , 
(2) 
∂ 
∂t ( ρu ) + ∇ · ( ρuu + pI ) = 0 , 
(3) 
∂ 
∂t ( ρE ) + ∇ · ( u ( ρE + p ) ) = 0 , 
(4) 
∂ α1 
∂t + u ∇ α1 = K∇ · u , 
(5) 
where the phasic volume fraction must satisfy the constraint α1 + 
α2 = 1 , ρi is the phasic density, u the velocity vector, p the pres- 
sure, and E the total energy. For compressible ﬂows, the Kapila 
term is considered for the compatibility of the assumption that the 
material derivatives of the phase entropies are zero ( Kapila et al., 
2001 ; Kreeft and Koren, 2010 ; Murrone and Guillard, 2005 ) 
K = α1 α2 
ρ2 c 2 
2 −ρ1 c 2 
1 

α1 ρ2 c 2 
2 + α2 ρ1 c 2 
1 

(6) 
The governing system (1)–(5) is a quasi-conservative ﬁve- 
equation model that is simpliﬁed from the Baer–Nunziato model, 
3 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
known as the general seven-equation model ( Baer and Nunzi- 
ato, 1986 ; Nguyen et al., 2020 ). The general system is closed by the 
use of its own equation of states (EOS) for each phase, which al- 
lows the treatment of ﬂuids characterized by vastly different ther- 
modynamics. The stiffened gas EOS for each phase is given by 
p i = ρi e i ( γi −1 ) −γi πi , 
(7) 
and temperature can be modeled as ( Le Métayer et al., 2005 ) 
T i = 
p i + πi 
( γi −1 ) C v ,i ρi 
. 
(8) 
The total energy per unit mass of each phase is deﬁned as 
ρi E i = ρi e i + 1 
2 ρi | u i | 2 , 
(9) 
where γi is the ratio of speciﬁc heats, πi is a material constant, and 
e i is the speciﬁc internal energy. 
For three-dimensional two-component ﬂows, the general model 
contains 11 equations; furthermore, because of the large number 
of waves they contain and the sensibility of the results with re- 
spect to the relaxation procedures, the model is expensive and nu- 
merically complex to solve. For a simpler implementation and less 
expensive computation, the ﬁve-equation model (1)-(5) simpliﬁed 
from the general model by assuming a single velocity and a single 
pressure between two phases was used. The equilibrium assump- 
tion implies that the quantities associated with the components 
are averaged to yield the corresponding mixture quantity. With the 
assumption of p = p 1 = p 2 , the generalized EOS for the mixture is 
given by ( Allaire et al., 2002 ; Deligant et al., 2015 ) 
p = ( γ −1 ) ρe −γ π, 
(10) 
where 
γ = 1 + 
1 
α1 
γ1 −1 + α2 
γ2 −1 
, and π = 
α1 γ1 π1 
γ1 −1 + α2 γ2 π2 
γ2 −1 
1 + α1 
γ1 −1 + α2 
γ2 −1 
(11) 
Accordingly, the quantities per unit volume are averaged by 
their respective volume fraction, α1 and α2 . The equilibrium mix- 
ture components are given by 
ρ = α1 ρ1 + α2 ρ2 
(12) 
ρe = α1 ρ1 e 1 + α2 ρ2 e 2 
(13) 
ρE = ρe + 1 
2 ρ| u | 2 
(14) 
To solve the governing Eqs. (1 –5) in a compact vector form, the 
nonconservative term in the advection equation for volume frac- 
tion (5) can be reformulated as 
u ∇ α1 = ∇ · u α1 −α1 ∇ · u . 
(15) 
All dependent variables are nondimensionalized using the free- 
stream conditions, and the governing Eqs. (1 –5) can be rewritten 
in a compact vector form as follows: 
∂Q 
∂t + ∇ · F ( Q ) = K ∇ · H ( Q ) , 
(16) 
where 
Q = [ α1 ρ1 , α2 ρ2 , ρu, ρv , ρw, ρE, α1 ] T 
is
the 
state 
vector, 
F (Q ) = ( F 1 , F 2 , F 3 ) 
is 
the 
ﬂux 
tensor, 
K = [ 0 , 0 , 0 , 0 , 0 , 0 , K + α1 ] T , and H (Q ) = ( H 1 , H 2 , H 3 ) is the 
nonconservative part in the right-hand side of the system. 
Here, 
F 1 = 
⎛ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎝ 
α1 ρ1 u 
α2 ρ2 u 
ρu 2 + p 
ρu v 
ρuw 
( ρE + p ) u 
α1 u 
⎞ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎠ 
; F 2 = 
⎛ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎝ 
α1 ρ1 v 
α2 ρ2 v 
ρu v 
ρv 2 + p 
ρv w 
( ρE + p ) v 
α1 v 
⎞ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎠ 
; F 3 = 
⎛ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎜ 
⎝ 
α1 ρ1 w 
α2 ρ2 w 
ρuw 
ρv w 
ρw 2 + p 
( ρE + p ) w 
α1 w 
⎞ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎟ 
⎠ 
;
H 1 = 
⎛ 
⎜ 
⎜ 
⎝ 
0 
0 
0 
0 
u 
⎞ 
⎟ 
⎟ 
⎠ ; H 2 = 
⎛ 
⎜ 
⎜ 
⎝ 
0 
0 
0 
0 
v 
⎞ 
⎟ 
⎟ 
⎠ ; H 3 = 
⎛ 
⎜ 
⎜ 
⎝ 
0 
0 
0 
0 
w 
⎞ 
⎟ 
⎟ 
⎠ . 
(17) 
2.2. Numerical method 
2.2.1. Eigensystem 
Governing equations expressed in a compact vector form (16) 
can be discretized in a general structured body-ﬁtted grid and 
solved by a ﬁnite-volume Riemann method based on an associ- 
ated Godunov-type numerical scheme, as reported in ( Nguyen and 
Park, 2015 ; Nguyen et al., 2016 ). In the schemes, the characteristic 
information of the governing equations is used to compute con- 
vective ﬂux derivatives. Hence, the ﬂux Jacobian matrix is divided 
into two subvectors that are associated with nonnegative and non- 
positive eigenvalues. The convective ﬂux vector is discretized us- 
ing a cell-centered ﬁnite-volume procedure, wherein extrapolated 
Riemann variables are obtained using the MUSCL procedure. The 
convective ﬂux in system (16) can be linearized as follows: 
∂Q 
∂t + A ∂Q 
∂x + B ∂Q 
∂y + C ∂Q 
∂z = Q ∇ · H ( Q ) . 
(18) 
The ﬂux Jacobian matrices are A = ∂ F 1 /∂Q , B = ∂ F 2 /∂Q , and 
C = ∂ F 3 /∂Q . 
The quasi-linear system (18) can be rewritten in terms of the 
primitive variables of W = [ ρ1 , ρ2 , u, v , w, p, α ] T as follows: 
∂W 
∂t + ˜ 
A ∂W 
∂x + ˜ 
B ∂W 
∂y + ˜ 
C ∂W 
∂z = Q ∇ · H ( Q ) , 
(19) 
where the system matrix ˜ 
A = [ ∂Q 
∂W ] −1 [ ∂ F 1 
∂W ] , ˜ 
B = [ ∂Q 
∂W ] −1 [ ∂ F 2 
∂W ] , and 
˜ 
C = [ ∂Q 
∂W ] −1 [ ∂ F 3 
∂W ] . 
The system matrix ˜ 
A is given as 
˜ 
A = 
⎡ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎣ 
u 
0 
α1 ρ1 
0 
0 
0 
0 
0 
u 
α2 ρ2 
0 
0 
0 
0 
0 
0 
u 
0 
0 
1 /ρ
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
0 
u 
0 
0 
0 
0 
ρc 2 
0 
0 
u 
0 
0 
0 
0 
0 
0 
0 
u 
⎤ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎦ 
;
˜ 
B = 
⎡ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎣ 
v 
0 
0 
α1 ρ1 
0 
0 
0 
0 
v 
0 
α2 ρ2 
0 
0 
0 
0 
0 
v 
0 
0 
0 
0 
0 
0 
0 
v 
0 
1 /ρ
0 
0 
0 
0 
0 
v 
0 
0 
0 
0 
0 
ρc 2 
0 
v 
0 
0 
0 
0 
0 
0 
0 
v 
⎤ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎦ 
;
˜ 
C = 
⎡ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎣ 
w 
0 
0 
0 
α1 ρ1 
0 
0 
0 
w 
0 
0 
α2 ρ2 
0 
0 
0 
0 
w 
0 
0 
0 
0 
0 
0 
0 
w 
0 
0 
0 
0 
0 
0 
0 
w 
1 /ρ
0 
0 
0 
0 
0 
ρc 2 
w 
0 
0 
0 
0 
0 
0 
0 
w 
⎤ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎦ 
(20) 
The system matrix ˜ 
A can be split into subvectors associated 
with the nonnegative and nonpositive eigenvalues as ˜ 
A = ˜ 
R ˜ 
R −1 . 
The eigenvalues are given in accordance with 
λ1 = λ2 = λ3 = λ4 = λ7 = u, λ5 = u + c, λ6 = u −c, 
(21) 
where the sound speed is deﬁned as 
c = 
 
γ ( p + π) 
ρ
. 
(22) 
4 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
The corresponding right eigenvector matrix of the system ma- 
trix ˜ 
A is given as 
˜ 
R = 
⎡ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎣ 
1 
0 
0 
0 
α1 ρ1 / ρc 2 
α1 ρ1 / ρc 2 
0 
0 
1 
0 
0 
α2 ρ2 / ρc 2 
α2 ρ2 / ρc 2 
0 
0 
0 
0 
0 
1 / ( ρc ) 
−1 / ( ρc ) 
0 
0 
0 
1 
0 
0 
0 
0 
0 
0 
0 
1 
u 
0 
0 
0 
0 
0 
0 
1 
1 
0 
0 
0 
0 
0 
0 
0 
1 
⎤ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎦ 
. 
(23) 
For the simulation of interfaces between compressible ﬂuids, 
the nonconservative set of variables in Eq. (19) and a common 
ﬂux over the boundary of two adjacent cells would cause a con- 
servation error. To obtain better conservation properties and cap- 
ture shock waves, the system is subsequently transformed back to 
the conservative set of variables. The ﬂux Jacobian matrix A in the 
system (18) can be computed as 
A = ∂Q 
∂W 
˜ 
A ∂W 
∂Q = 
 ∂Q 
∂W 
˜ 
R 

[ ] 

˜ 
R −1 ∂W 
∂Q 

, 
(24) 
with [ ] = diag ( λ1 , λ2 , λ3 , λ4 , λ5 , λ6 , λ7 ) and ∂W 
∂Q = [ ∂Q 
∂W ] −1 . 
B and C can be derived similarly. 
The Jacobian ∂Q 
∂W is given as 
∂Q 
∂W = 
⎡ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎢ 
⎣ 
α1 
0 
0 
0 
0 
0 
ρ1 
0 
α2 
0 
0 
0 
0 
−ρ2 
α1 u 
α2 u 
ρ
0 
0 
0 
( ρ1 −ρ2 ) u 
α1 v 
α2 v 
0 
ρ
0 
0 
( ρ1 −ρ2 ) v 
α1 w 
α2 w 
0 
0 
ρ
0 
( ρ1 −ρ2 ) w 
α1 | u | 2 / 2 
α2 | u | 2 / 2 
ρu 
ρv 
ρw 
1 / ( γ −1 ) 
( ρ1 −ρ2 ) | u | 2 / 2 
0 
0 
0 
0 
0 
0 
1 
⎤ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎥ 
⎦ 
, 
(25) 
where | u | 2 = u 2 + v 2 + w 2 
2.2.2. Discretization 
The convective ﬂux vector in the system (16) was discretized 
using a cell-centered ﬁnite volume procedure. Considering the con- 
vective ﬂux derivative in the x -direction, the following difference 
formula based on the Godunov-type numerical scheme was used: 
∂ F 1 
∂x = ( F 1 ) i +1 / 2 −( F 1 ) i −1 / 2 
∂x 
. 
(26) 
The numerical ﬂuxes in the cell face ﬁnite volume formulation 
can be expressed as ( Nguyen et al., 2020 ; Nguyen and Park, 2016 ; 
Nguyen et al., 2016 ; Toro, 2009 ) 
( F 1 ) i +1 / 2 = 1 
2 

F L + F R −S + 
Q R 
i +1 / 2 −Q L 
i +1 / 2 

, 
(27) 
where F L = F 1 ( Q L 
i +1 / 2 ) , F R = F 1 ( Q R 
i +1 / 2 ) , S + = max 
k =1 , 7 ( | λk | ) , and the 
eigenvalues λ are computed in accordance with Eq. (21) from the 
Roe-averaged values of Q L 
i +1 / 2 and Q R 
i +1 / 2 . 
Alternatively, the numerical ﬂuxes can be computed by solving 
a Riemann problem using the Harten–Lax–van Leer (HLL) approxi- 
mate Riemann solver: 
( F 1 ) i +1 / 2 = 
 F L 
F HLL 
F R 
if 0 ≤S L 
if S L ≤0 ≤S R 
if 0 ≥S R 
, 
(28) 
F HLL = 
S R F L −S L F R + S R S L 

Q R 
i +1 / 2 −Q L 
i +1 / 2 

S R −S L 
(29) 
where S L = min 
k =1 , 7 ( 0 , λL 
k , λR 
k ) and S R = max 
k =1 , 7 ( 0 , λL 
k , λR 
k ) . 
The 
procedure 
of 
spatial 
discretization 
deﬁned 
by 
Eq. (26) yields the overall second-order accuracy in multiple 
dimensions. For the extrapolated Riemann variables, a MUSCL 
procedure with third-order accuracy was employed, as follows: 
Q L 
i +1 / 2 = Q i + 1 
4 

( 1 −κ) 
Q i −1 / 2 φ( r L ) + ( 1 + κ) 
Q i +1 / 2 φ( 1 / r L ) 

, 
(30) 
and 
Q R 
i +1 / 2 = Q i +1 −1 
4 

( 1 −κ) 
Q i +3 / 2 φ( r R ) 
+ ( 1 + κ) 
Q i +1 / 2 φ( 1 / r R ) 

, 
(31) 
where the ratios of consecutive solutions are given by 
r L = 
Q i −1 / 2 

Q i +1 / 2 
, r R = 
Q i −1 / 2 

Q i +3 / 2 
, 
Q i +1 / 2 = Q i +1 −Q i . 
(32) 
The MUSCL procedures of (30) and (31) were performed using 
κ = 1 / 3 and a van Albada limiter φ. 
A third TVD Runge-Kutta method is used for time discretization. 
The system (16) can be temporally discretized with the ﬁrst order 
in the time solver as follows”
Q n +1 = Q n + 
tL 
Q n 
, L 
Q n 
= [ ∇ · F ( Q ) + Q ∇ · H ( Q ) ] 
n 
(33) 
With the discrete operator L ( Q n ) , the third-order Runge-Kutta 
method is expressed as 
Q ∗= Q n + 
tL ( Q n ) , 
(34) 
Q ∗∗= 3 
4 Q n + 1 
4 Q ∗+ 
t 
4 L ( Q ∗) , 
(35) 
Q n +1 = 1 
3 Q n + 2 
3 Q ∗∗+ 2
t 
3 L ( Q ∗∗) 
(36) 
For the stability constraints of the numerical scheme, the time 
step is limited according to the CFL stability restriction by the max- 
imum eigenvalue of the hyperbolic system as follows: 

t = 
CF L 
 3 
j=1 max | λk | 
, 
(37) 
where the dummy index j = 1, 2, and 3 corresponds to the three 
coordinate directions. The CFL number is in the range of 0 . 5 −1 . 0 
in the study. 
3. Interface-sharpening technique 
Standard numerical solvers often introduce additional interface 
smearing by numerical diffusion errors. To maintain a sharp inter- 
face, an interface sharpening technique developed in ( Nguyen and 
Park, 2017 ; Nguyen et al., 2018 ) was extended to be applied as 
postprocessing to the volume-fraction ﬁeld after each time step. 
This technique was applied independently of the discretization 
scheme of the governing system. For incompressible ﬂows, the fol- 
lowing interface-sharpening equation was solved after each physi- 
cal time step to suppress the diffusion error and maintain a sharp 
interface with the desired thickness: 
∂ α1 
∂τ
= −∇ ·  ˜ 
f ( α1 ) n −D ( ∇ α1 . n ) n 
, 
(38) 
5 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
where of ∇ · ( ∇ α1 . n ) is the diffusion term and the artiﬁcial com- 
pression ﬂux term, ˜ 
f ( α1 ) = α1 ( 1 −α1 ) was used to maintain the 
resolution of contact discontinuities in regions where 0 < α1 < 1 , 
and the local interface normal vector n was introduced into the 
interface-sharpening equation such that compression occurred in 
the normal direction of the interface. D is a length-scale of the or- 
der of the grid spacing for deﬁning the desired interface thickness 
and is deﬁned as D = V 1 / 3 / 2 , and V is the cell volume. 
For modeling incompressible ﬂows, only density is required to 
update after the volume-fraction ﬁeld is corrected ( Nguyen and 
Park, 2017 ; Nguyen et al., 2018 ). However, for compressible ﬂu- 
ids solved by the ﬁve-equation model, the state variable vector 
Q in system (16) is varied according to the change in α1 dur- 
ing the interface-sharpening process. Therefore, all the ﬂow vari- 
ables must be updated according to α1 to ensure consistency. The 
general regularization approach was proposed in ( Tiwari et al., 
2013 ) to obtain consistent thermodynamic laws for the mixture 
and reduce the interface diffusion error. However, in ( Tiwari et al., 
2013 ), the interface-sharpening function added in the source term 
of the advection equation can only reduce the interface diffu- 
sion error, i.e., it cannot maintain a constant interface thick- 
ness everywhere at all times. As aforementioned, an interface 
with a constant thickness within several mesh cells is neces- 
sary for capturing discontinuities, which has important implica- 
tions for the modeling of shock. It eases the computation of 
shock or contact discontinuities in compressible two-phase appli- 
cations and enables shock and contact discontinuity formation to 
be accurately predicted ( Saurel and Lemetayer, 2001 ). Hence, a 
modiﬁed interface-sharpening Eq. (39) is formulated based on a 
mixture-consistent interface regularization approach and interface- 
sharpening Eq. (38) . An iteration process is applied to solve the 
Eq. (39) . Consequently, this postprocessing procedure could main- 
tain the integrity of the thin interface between immiscible ﬂu- 
ids, reduce the numerical diffusion error in the solution of the 
discretization scheme, maintain a sharp interface with the de- 
sired constant thickness, and obtain consistent thermodynamic 
laws for the mixture after each time step. The modiﬁed interface- 
sharpening equation for updating variable vector Q in as follows: 
∂Q 
∂τ = ∂Q 
∂ α1 
S ( α1 ) , 
(39) 
where 
∂Q 
∂ α1 
= 

ρ1 , −ρ2 , ( ρ1 −ρ2 ) u 1 , ( ρ1 −ρ2 ) u 2 , ( ρ1 −ρ2 ) u 3 , 
( ρ1 −ρ2 ) | u | 2 
2 
+ ρ1 ε 1 −ρ2 ε 1 , 1 

, 
(40) 
and 
S ( α1 ) = −∇ ·  ˜ 
f ( α1 ) n −D ( ∇ α1 . n ) n 
. 
(41) 
The third TVD Runge–Kutta method Eq. (33 - (36) ) was used for 
time discretization of Eq. (39) , and the interface-sharpening cor- 
rection process required several iterations per physical time step 
to achieve a τ-steady state. The stopping criteria for the conver- 
gence of iteration is applied as in previous studies ( Nguyen and 
Park, 2017 ; Nguyen et al., 2018 ; Shukla et al., 2010 ) where the L ∞ 
norm of the volume fraction α1 is less than 10 −6 . The correction 
is applied each time step. While the diffusion error in each time 
step is small, it is found that the interface-sharpening correction is 
needed only one to three times per physical time step. The time 
consumption of this trivial process is small compare to the overall 
time for solving the governing system. 
One of the key challenges with capturing methods for inter- 
face ﬂows is being able to maintain velocity, pressure, and tem- 
perature equilibrium across material interfaces when the equa- 
tion of state changes. To assess this issue, the one-dimensional 
pure interface advection, which was extensively used for vali- 
dations of the mathematical models of two-phase systems, is 
computed to demonstrate the ability of the interface sharpen- 
ing approach for maintaining velocity, pressure, and tempera- 
ture equilibrium. The length of the computational domain is 
1.0 m and the initial interface between water and air is lo- 
cated at X = 0 . 5 m . The initial condition was set as ( ρ, u, p, γ ) = 

( 1 . 2 kg 
m 3 , 150 m 
s , 101325 Pa, 1 . 4 ) i f 0 ≤x ≤0 . 5 
( 10 0 0 kg/ m 3 , 150 m 
s , 101325 Pa, 4 . 4 ) else 
. The discretiza- 
tion is performed on a 400-cell grid and boundary conditions are 
constant states on both the right and left sides of the domain. 
Fig. 1 shows the interface proﬁles and numerical errors, veloc- 
ity u ∗= u 
U ∞ , pressure p ∗= 
p 
ρ∞ U 2 
∞ , and temperature T ∗= T 
T ∞ at 
t ∗= t U ∞ 
L ∞ = 0 . 26 ; ρ∞ = 10 0 0 kg 
m 3 , L ∞ = 1 m, and U ∞ = 150 m 
s . The 
comparison of interface proﬁles with and without using IST shows 
a signiﬁcant improvement in maintaining a sharp interface. Fur- 
thermore, the initial constant velocity, pressure, and temperature 
proﬁles are perfectly maintained with very small oscillation of the 
temperature about the order of 10 −12 at the interface. 
4. Numerical simulations and discussion 
4.1. Sod shock tube problem 
The one-dimensional stiff two-ﬂuid Reimann problem was con- 
sidered in this study. In this shock tube problem, the domain was 
deﬁned as [0, 1] and the initial condition was set as ( ρ, u, p, γ ) = 
{ ( 1 , 0 , 1 , 1 . 4 ) i f 0 ≤x ≤0 . 5 
( 0 . 125 , 0 , 0 . 1 , 1 . 667 ) else . The solutions to this problem were 
computed with and without the IST on a grid of 200 points and 
compared with exact solutions at t = 0.2, as shown in Fig. 2 . The 
computed solutions agree well with the exact solution. The solu- 
tion obtained using the IST maintains a constant interface thick- 
ness of a few grid points during the computation, while the result 
without IST shows the numerical diffusion reducing the accuracy 
of the solution. Using the IST, the density, pressure, and velocity 
ﬁelds are corrected and approximates the exact solution. 
Fig. 3 shows the solutions using the IST at higher grid resolu- 
tions, i.e., 40 0 and 80 0 points. As shown, the interface is resolved 
by a similar number of grid points on different grids sizes, and 
the numerical method achieves grid independence for this prob- 
lem. Furthermore, the accuracy of the method was assessed by an- 
alyzing the error norms, as shown in Table 1 , and these computa- 
tions have small errors of L 1 , L 2 , and L ∞ , as shown in the table. 
To further assess the accuracy of the numerical scheme, mass 
conservation, and momentum and energy balance over time for 
problems with shocks and interfaces were analyzed. By integrat- 
ing the conservation Eqs. (1) - (4) over the ﬂuid domain  and ap- 
plying the divergence theorem, we obtain the formulations of the 
mass, momentum, and energy conservation as follows: 
   

∂ 
∂t ( α1 ρ1 ) dV + 
  
S 
( α1 ρ1 u ) · n dS = 0 , 
(42) 
   

∂ 
∂t ( α2 ρ2 ) dV + 
  
S 
( α2 ρ2 u ) · n dS = 0 , 
(43) 
   

∂ 
∂t ( ρu ) dV + 
  
S 
( ρuu + pI ) · n dS = 0 , 
(44) 
   

∂ 
∂t ( ρE ) dV + 
  
S 
( u ( ρE + p ) ) · n dS = 0 . 
(45) 
We can evaluate these equations in order to check the mass, 
momentum, and energy conservation accuracy of the numerical 
6 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 1. One-dimensional interface advection solutions at t ∗= 0 . 26 . 
Table 1 
Computed error norms of L 1 , L 2 , L ∞ for sod shock tube problem. 
Grid 
size 
Density 
Velocity 
Pressure 
L 1 
L 2 
L ∞ 
L 1 
L 2 
L ∞ 
L 1 
L 2 
L ∞ 
100 
0.01570 
0.02588 
0.09676 
0.01662 
0.03069 
0.09992 
0.01297 
0.02503 
0.0960 
200 
0.00437 
0.01143 
0.09463 
0.00472 
0.01093 
0.07561 
0.00376 
0.00978 
0.06568 
400 
0.00246 
0.00620 
0.03707 
0.00307 
0.00716 
0.03125 
0.00279 
0.00753 
0.04604 
800 
0.00206 
0.00551 
0.04577 
0.00300 
0.00831 
0.08545 
0.00234 
0.00613 
0.02997 
scheme. For example, in Eq. (42) , the ﬁrst term is used to evalu- 
ate the total mass change of ﬂuid 1 in the domain, and the sec- 
ond term is the difference between the mass rates of ﬂow moving 
in and out of the domain. The left-hand sides of these equations 
must be equal to zero. In other words, the mass, momentum, and 
energy must balance. By using the theory of mass, momentum, and 
energy conservation, the accuracy of the numerical scheme for the 
problem can be assessed. The results of the analysis for four differ- 
ent grid resolutions are plotted in Fig. 4 . The results show that the 
conservatrion errors converge to zero when the grids are reﬁned, 
showing good accuracy of the numerical scheme. 
4.2. Shock-bubble-interaction problems 
In this section, two practical cases of a shock wave of Mach 
number 1.22 hitting an air-helium bubble and air-R22 bubble for 
cases I and II, respectively, were studied to examine the ability 
7 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 2. Sod shock tube solutions at t = 0.2. Comparison of numerical results with and without interface -sharpening on a grid of 200 points. 
Table 2 
Initial condition for the shock wave hitting an air-helium bubble and air-R22 bubble 
for cases I and II, respectively. 
γ
ρ ( kg/ m 3 ) 
u ( m/s ) 
v ( m/s ) 
p( Pa ) 
π( Pa ) 
Air 1 
1.4 
1.4 
0.0 
0.0 
100,000 
0.0 
Air 2 
1.4 
1.92691 
-114.42 
0.0 
156,980 
0.0 
Helium (case-I) 
1.648 
0.25463 
0.0 
0.0 
100,000 
0.0 
R22 (case-II) 
1.249 
4.41540 
0.0 
0.0 
100,000 
0.0 
of the numerical method in simulating the interaction of shock 
and sharp interfaces between two phases. These experiments, in- 
troduced by Haas and Sturtevant ( Haas and Sturtevant, 1987 ), 
have been extensively used as a benchmark for validations of 
mathematical and numerical methods, especially two-phase sys- 
tems for shock wave capturing ( Ansari and Daramizadeh, 2013 ; 
Deligant et al., 2015 ; Kreeft and Koren, 2010 ). Fig. 5 shows the 
schematics of the initial setup of these problems. In these experi- 
ments, bubbles were produced by inﬂating a cylindrical shape with 
thin walls made of a nitrocellulose membrane. In the simulations, 
the conﬁguration was considered as two dimensional, and the in- 
coming shock was introduced in front of the bubble. The dimen- 
sions of the computational domain were 267 mm × 89 mm and 
a grid of 1200 × 400 was used for both computations. The ﬂuid 
parameters and initial conditions are listed in Table 2 . In case-I, 
the cylindrical bubble was ﬁlled with helium, which is lighter than 
air; in case-II, the bubble was ﬁlled with R22 gas, which is heavier 
than air. The bubbles were at rest initially and surrounded by air, 
which was at rest as well. 
Fig. 6 shows a comparison of the numerical Schlieren-type re- 
sults on the right side obtained using the present method and the 
experimental photographs on the left side for the case-I of the 
air-helium bubble. The shock wave traveled from right to left, as 
shown in Fig. 5 . As the shock wave reached the air-helium bub- 
ble, the equilibrium state of the bubble was primarily perturbed 
through the air and toward the bubble. The refracted shock ap- 
peared and propagated ahead of the incoming shock, while the re- 
ﬂected wave was an expansion wave propagating back to the right 
( Fig. 6 (a)–(e)). It is clear that both the patterns and positions of the 
incoming, reﬂected, and refracted shocks are all well captured by 
the numerical method. The top and bottom parts of the incoming 
shock between the bubble and walls continue moving from right to 
left, while the middle part interacts with the bubble and becomes 
the reﬂected shock as a circular pattern moving backward to the 
right. The reﬂected shock exhibits a circular pattern because of the 
circular shape of the bubble. The refracted shock exhibits a circular 
shape as well, but it moves toward the left faster than the incom- 
ing shock and subsequently passes the bubble at t = 62 μs ( Fig. 6 
(c)). The refracted shock leaves the bubble and continues to pass 
through the air as a circular transmitted wave ( Figs. 6 (d)–(f)). The 
reﬂected and refracted shocks continue expanding and spreading 
and ﬁnally impact the top and bottom walls, resulting in a ripple- 
8 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 3. Sod shock tube solutions at t = 0.2; symbols are numerical results with interface-sharpening and solid lines are the exact solutions. Left column: grid resolution of 
400 points. Right column: grid resolution of 800 points. 
9 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 4. Mass (top), momentum (middle), and energy (bottom) balance results for a time-dependent study of a sod shock tube problem. 
10 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 5. Schematics of the initial setup of the shock–bubble interaction. 
like shock wave pattern. The bubble is displaced and moves toward 
the left owing to the impact of the incoming shock; additionally, 
the bubble shape is captured effectively by the numerical method. 
The effect of the interface between the two ﬂuids on the patterns 
and behaviors of shock waves is strong. The interface remained 
sharp, and the thickness was primarily constant during simulation. 
Fig. 7 shows the bubble shape at t = 346 μs and its zoon in view 
at the interface with the grid. The interface thickness was main- 
tained at within several grid points. Fig. 8 shows the evolution of 
air volume fraction and density distribution during this simulation 
at t = 32, 52, 62, 72, 82, 102, and 260.0 μs. The air volume frac- 
tion on the left side of the ﬁgure elucidates the behavior of the 
bubble owing to the impact of the shock wave during the simula- 
tion. The density distribution on the right side of the ﬁgure shows 
that the reﬂect shock decreases the density of the air 2 in the re- 
gion it passes (the vicinity at the right side of the bubble in the 
top three rows of the ﬁgure) and in contrast, the refracted shock 
increase the density of air 1 in the region it passes (the vicinity at 
the left side of the bubble in the top three rows of the ﬁgure). The 
density of helium inside the bubble is lighter than the sounded 
air, and the change in helium density is small. In the bottom three 
rows of the ﬁgure, as mentioned, the reﬂected and refracted shocks 
interact with the top and bottom walls, resulting in a ripple-like 
pattern of shock waves and an initial change in uniform density of 
air 1 and air 2. 
Fig. 9 shows a comparison of the numerical results obtained by 
the present method and the experimental photographs for case-II. 
In general, the patterns and positions of the shock waves plotted 
by Schlieren-type images by the numerical method on the mid- 
dle column matched well with the experimental photographs on 
the left column. It is similar to case-I, where the reﬂected and 
refracted shocks exhibit a curved pattern because of the circular 
shape of the bubble. However, as shown by the comparison be- 
tween cases I and II in Fig. 10 , the refracted shock pattern inside 
the bubble in case-II is clearer and the density change of the R22 
bubble is larger owing to the larger density of the R22 bubble com- 
pared with that of air. The density and shape of the bubbles not 
only deform the pattern of the shocks but also affect the speed of 
the shocks. The refracted shock in case-I diverged and spread to- 
ward the left faster than the incoming shock, while the refracted 
shock in case-II converged and transmitted slower than the incom- 
ing shock. In case-II, the convergence of the refracted shock caused 
the density peak to exceed thrice the initial density inside the bub- 
ble ( Fig. 9 (d)). Two parts of the incoming shock in the air moved 
faster and ﬁnally left the bubble and intersected each other. Af- 
ter the intersection of the two parts of the incoming shock, the 
refracted shock inside the R22 bubble converged in the left-most 
point of the bubble and subsequently expanded radially as a trans- 
mitted shock ( Figs. 9 (d)–(f)). The transmitted shock, which was of 
high velocity, was focused at the middle point and caused the in- 
terface of the bubble to bulge along the symmetry axis. 
4.3. Underwater explosion 
To further assess the numerical method for the analysis of mul- 
tiphase compressible ﬂows with high-density ratios and in the 
Fig. 6. Interaction of shock and air-helium bubble, shadow photographs of Haas 
and Sturtevant (left column) ( Haas and Sturtevant, 1987 ), patterns of Schlieren-type 
image obtained by current numerical method (right column) at t = 32 μs (a), 52 
μs (b), 62 μs (c), 72 μs (d), 82 μs I, 102 μs (f), and 260.0 μs (g). 
11 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 7. Predicted air volume fraction and helium bubble at t = 346 μs . 
Table 3 
Initial conditions for an underwater explosion near a free surface. 
γ
ρ ( kg/ m 3 ) 
p( Pa ) 
π ( Pa ) 
Air 
1.4 
1.225 
101,325 
0.0 
Explosion bubble 
1.4 
1250 
10 9 
0.0 
Water 
4.4 
1000 
101,325 
6 × 10 8 
presence of a strong shock wave, a two-dimensional underwater 
explosion near a free surface was simulated. This type of prob- 
lem has been widely investigated to demonstrate the ability of 
multiphase compressible systems ( Ansari and Daramizadeh, 2013 ; 
Daramizadeh and Ansari, 2015 ; Ge et al., 2019 ; Grove and 
Menikoff, 1990 ; Shukla et al., 2010 ). In this study, the simulation 
was established based on an experimental underwater explosion 
test by Kleine et al. ( Kleine et al., 2009 ). In this problem, the un- 
derwater explosion of 10 mg of silver azide (AgN3) equivalent to 
the energy of 25.5 J/m. The simulation was performed in a com- 
putational domain measuring 0 . 7 m × 0 . 7 m . Reﬂecting boundary 
conditions were applied on the side and bottom boundaries, and 
the top edge was extrapolated. The initial conditions and ﬂuid pa- 
rameters are listed in Table 3 . The initial bubble of a highly pres- 
surized gas was located 0.05 m below the free surface and the ﬂu- 
ids were at rest in the entire domain. 
A grid reﬁnement study of the underwater explosion was per- 
formed using four different grid including Grid I ( 40 0 × 40 0 ), Grid 
II (8 00 × 800 ), Grid III ( 1200 × 1200 ), and Grid IV (16 00 ×
1600 ). The initial bubble was modeled as a high-pressure of 10 9 
Pa with a diameter of the bubble D = 0.01 m. The ratio of the 
initial diameter of the bubble and the size of the domain is very 
small, therefore the grid resolution needs to be ﬁne enough to ob- 
tain grid independence. Fig. 11 shows the initial bubble setup with 
the four grids. It shows that when using Grid I and Grid II the in- 
terfaces are very thick and do not have high enough resolution, 
while the results on ﬁner grids including Grid III and Grid IV show 
good resolutions. A CFL number of 1.0 is used in these simulations. 
Fig. 12 shows a comparison of the predicted bubble and free sur- 
face proﬁles at t = 0.5 ms using the four grids. A grid convergence 
is obtained since the results using Grid III and Grid IV are very 
close to each other. Fig. 13 shows the bubble shape and free sur- 
Fig. 8. Evolution of air volume fraction (left column) and nondimensional density 
distribution ρ/ ρair 1 (right column) during interaction of shock and air-helium bub- 
ble at t = 32, 52, 62, 72, 82, 102, and 260.0 μs, from top to bottom. 
face proﬁles at t = 1 ms after the explosion and its zoom-in view 
at the interface with the grid. A very good resolution was obtained. 
Fig. 14 shows predicted density change presented in the form of 
Schlieren-type images at an early stage after the underwater explo- 
sion to assess the inﬂuence of grid resolution on the shock wave 
captured at the early stage. The results show acceptable simula- 
tions using the two ﬁnest grids. 
Fig. 15 shows a qualitative comparison between the Schlieren- 
type images and the pressure distribution plotted from the 
present simulation and the reference results ( Ansari and 
Daramizadeh, 2013 ). The propagation process of a strong shock 
wave in the ﬂuid ﬁeld is clearly illustrated in this ﬁgure, showing 
a good agreement. Results in the top row show that high pressure 
in the gas bubble functioned as a tension wave at the bubble inter- 
face impacting the surrounding water, causing a radial shock wave 
and pressure pulse. The shock wave radially expanded outward 
into the water and impacted the free surface. After hitting the free 
surface, the pressure pulse was transmitted into the air inducing 
a weak refracted wave, and simultaneously, a reﬂected expansion 
wave from the free surface moved back into the water. The results 
in the middle and bottom rows show that the reﬂected wave 
12 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 9. Interaction of shock and air-R22 bubble (case-II), shadow photographs of Haas and Sturtevant (left column) ( Haas and Sturtevant, 1987 ), patterns of numerical 
Schlieren-type results (middle column), and density distribution ρ/ ρ_(air 1) (right column) at t = 55 μs (a), 115 μs (b), 135 μs (c), 187 μs (d), 247 μs (e), and 318 μs (f). 
13 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 10. Comparison of shock patterns between case-I of air-helium bubble at 
t = 52 and 102 μs (left column), and case-II of air-R22 bubble at t = 55 and 135 
μs (right column). 
caused a very low pressurized region of the water underneath 
the free surface. This issue was discussed and similar results were 
shown in other studies ( Ge et al., 2019 ; Koukouvinis et al., 2016 ). 
Since the evaporation is not considered in this simulation, such 
pressures are naturally predicted by the stiffness gas equation of 
state. Fig. 16 shows shock wave patterns complicated by reﬂected 
Fig. 12. A comparison of the predicted bubble and free surface proﬁles at t = 0.5 
ms after the underwater explosion near a free surface. 
waves, interfaces, and walls. The radial wave expanded and inter- 
acted with the closed free surface ﬁrst, then the reﬂected wave 
from the free surface hit back to the bubble resulting in another 
Fig. 11. Initial proﬁle of bubble using different grids. 
14 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 13. Predicted bubble and free surface proﬁles and grid resolution with the interface thickness at t = 1 ms after the underwater explosion near a free surface. 
Fig. 14. Predicted density change presented in the form of Schlieren-type images at early stage after the underwater explosion near a free surface. 
15 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 15. Underwater explosion of AgN3 near free surface. Density change presented in the form of Schlieren-type images (left column) ( Ansari and Daramizadeh, 2013 ), the 
Schlieren-type images (middle column) and nondimensional pressure distribution ( p/ ( 10 9 −p 0 ) ) (right column) from present study. Time intervals between shown frames is 
14 μs. 
16 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 16. Numerical Schlieren-type images of an underwater explosion of AgN3 at times (a) t = 0.2 ms, (b) t = 0.25 ms, (c) t = 0.5 ms, (d) t = 1.0 ms, (e) t = 1.5 ms, and (f) t = 2.0 
ms. The dashed red lines illustrate the free surface. 
17 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 17. Bubble and free surface deformations, dimensionless kinematic energy, and velocity vectors in an underwater explosion near a free surface at t = 0.02 ms, 1.0 ms, 
6.0 ms, 10 ms, ms, and ms. 
18 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 18. The initial set-up condition for 3D simulation of the underwater explosion 
near a free surface. 
reﬂected wave from the bubble impacting the free surface again. 
The back and forth interactions of the reﬂected waves, bubble, 
and free surface result in a complicated wave pattern shown 
in Fig. 16 (a). The primary shock wave impacts the bottom and 
sidewalls and subsequently, the ﬁrst reﬂected shock wave from 
the free surface also hits the walls, while others reﬂected waves 
become weak during their propagation in the liquid. The three 
waves reﬂected from the walls and the reﬂected wave from the 
free surface intersected each other. The wave patterns and the 
interactions of the bubble and reﬂected wave from the bottom 
are captured in Figs. 16 (a)-(c), which conﬁrms the capability of 
the present method for capturing complex shock waves. The wave 
patterns became more complex subsequently and combined with 
the strong waves, the ripple-like pattern of waves appeared, as 
shown in Figs. 16 (d)–(f). The interface lines are also plotted in 
this ﬁgure to show the growth of the explosive bubble during the 
simulation. Due to the effect of the free surface, the bubble shape 
deviates from the initial radial shape and assumes an oval shape. 
The bubble continues to expand largely due to the high pressure 
of the initial bubble. 
To further assess the ability of the method to capture more 
complex behavior of a bubble collapse with the liquid jet forma- 
tion and a water column formed on the water surface, the initial 
pressure of the bubble is set to 10 7 Pa. Fig. 17 shows dimensionless 
kinetic energy and velocity vectors as well as interface proﬁles of 
the bubble and free surface during the simulation. The bubble ex- 
pands radially in the early stage ( Fig. 17 (a)), and the bubble con- 
tinues to grow in an oval shape, while a water hump is generated 
as well as high kinetic energy in the region below the free sur- 
face ( Fig. 17 (b)). The bubble grows and approaches a maximum 
size, and low kinetic energy is shown in Fig. 17 (c). The bubble 
collapses and a downward water jet with high energy is formed 
on the upper surface of the bubble, while a water hump continues 
to grow higher due to a direct consequence of the lower inertia 
of the ﬂuid towards the free surface ( Fig. 17 (d)-(f)). From the ve- 
locity ﬁeld, it is clear that the water ﬂow between the top of the 
bubble and the free surface is redirected upwards and downwards, 
and the bubble ﬁnally completely collapsed. The process of bub- 
ble collapse and the development of a high-speed water jet can be 
well predicted by the proposed method. 
Lately, a three-dimensional (3D) simulation of the underwater 
problem was performed to compare the numerical results with the 
experimental results by Kleine et al. ( Kleine et al., 2009 ). To simu- 
late the problem symmetry, a quarter of the underwater explosion 
(see Fig. 18 ) was applied with the symmetry boundary conditions 
along the relevant planes. A Cartesian uniform grid with sizes of 
24.3 million grid points and the domain of 0 . 2 m × 0 . 2 m result 
in the grid resolution equivalent to the grid III in the grid reﬁne- 
ment study of this problem. Fig. 19 shows a qualitative comparison 
between the experimental results and the present simulation with 
time intervals between shown frames is 14 μs. The 3D simulation 
results further conﬁrm the capability of the numerical method for 
the analysis of multiphase compressible ﬂows with high-density 
ratios and in the presence of a strong shock wave. 
19 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Fig. 19. Three-dimensional results of the underwater explosion of AgN3 near a free surface. Experimental photographs of Kleine et al. ( Kleine et al., 2009 ) (left column), 
density change presented in the form of Schlieren-type images and nondimensional pressure distribution ( p/ ( 10 9 −p 0 ) ) from present study (right column). Time intervals 
between shown frames is 14 μs. 
20 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
5. Conclusion 
In this study, a high-resolution shock- and interface-capturing 
model was developed to solve multiphase compressible ﬂows with 
a sharp interface between two immiscible ﬂuids and the pres- 
ence of shock waves. The main idea was to integrate the IST de- 
veloped in ( Nguyen and Park, 2017 ) into a ﬁve-equation model 
to correct smear and diffusion errors in diffuse interfaces and 
maintain a sharp interface. A mixture-consistent interface reg- 
ularization approach of all conservative variables proposed in 
( Tiwari et al., 2013 ) was improved and reconstructed as a postpro- 
cessing algorithm. Consequently, the method could maintain con- 
sistent thermodynamic laws for the mixture and ensure the con- 
sistency of the variables in the interface-sharpening process. The 
phase masses, momentum, and energy were corrected consistently 
through the correction process of the sharpened volume-fraction 
ﬁeld, which maintained the interface thickness during the simu- 
lation. This is important for the modeling of compressible mul- 
tiphase ﬂows, particularly solution procedures using conservative 
variables. Additionally, the interface-sharpening method can be ex- 
tended for three-phase ﬂow simulations, as presented and veriﬁed 
in ( Nguyen et al., 2018 ). 
A high-resolution Godunov-type numerical scheme has been 
constructed for a ﬁve-equation two-phase model. The computa- 
tional ﬁnite-volume Riemann solver together with computing al- 
gorithms in a hyperbolic vector form resulted in simulations with 
oscillation-free behaviors at material interfaces. The developed 
method was validated by several examples of ﬂuid interface sim- 
ulations including shock–tube and shock–bubble interactions, as 
well as underwater explosion. The shock–tube results from the 
developed method agreed well exact solutions and the analysis 
of computations indicate small errors. The shock wave patterns 
captured by the numerical method were compared comprehen- 
sively with experiments related to shock–bubble interaction prob- 
lems and the compressible ﬂows of underwater explosions with 
high-density ratios. Extension of the method for studies of shock 
wave emission and cavitation bubble dynamics during formation 
and break processes with high-speed microjets and high temper- 
atures, which are the main factors of cavitation erosion, will be 
considered in future studies. 
Declaration of Competing Interest 
The authors declare that they have no known competing ﬁnan- 
cial interests or personal relationships that could have appeared to 
inﬂuence the work reported in this paper. 
Acknowledgment 
This work was supported by Basic Science Research Program 
through the National Research Foundation of Korea (NRF) funded 
by the Ministry of Education (No. 2020R1I1A1A01072475), and the 
Human Resources Development program (No. 20184030202060) of 
the Korea Institute of Energy Technology Evaluation and Planning 
(KETEP) grant funded by the Korea government Ministry of Trade, 
Industry and Energy. 
References 
Abgrall, R. , 1996. How to Prevent Pressure Oscillations in Multicomponent Flow Cal- 
culations: A Quasi Conservative Approach. Journal of Computational Physics 125, 
150–160 . 
Abgrall, R. , Karni, S. , 2001. Computations of Compressible Multiﬂuids. Journal of 
Computational Physics 169, 594–623 . 
Allaire, G. , Clerc, S. , Kokh, S. , 2002. A Five-Equation Model for the Simulation of 
Interfaces between Compressible Fluids. Journal of Computational Physics 181, 
577–616 . 
Anderson Jr, J., 2009. Governing Equations of Fluid Dynamics. In: Wendt, J.F. 
(Ed.), Computational Fluid Dynamics. Springer, Berlin, pp. 15–51. doi: 10.1007/ 
978- 3- 540- 85056- 4 _ 2 . 
Ansari, M.R. , Daramizadeh, A. , 2013. Numerical simulation of compressible 
two-phase ﬂow using a diffuse interface method. Int J Heat Fluid Fl 42, 
209–223 . 
Baer, M.R. , Nunziato, J.W. , 1986. A two-phase mixture theory for the deﬂagration–
to-detonation transition (ddt) in reactive granular materials. Int J Multiphas 
Flow 12, 861–889 . 
Benson, D.J. , 1992. Computational methods in Lagrangian and Eulerian hydrocodes. 
Comput Method Appl M 99, 235–394 . 
Coralic, V. , Colonius, T. , 2014. Finite-volume WENO scheme for viscous compressible 
multicomponent ﬂows. Journal of Computational Physics 274, 95–121 . 
Cuong, D.H. , Thanh, M.D. , 2017. Building a Godunov-type numerical scheme for a 
model of two-phase ﬂows. Comput Fluids 148, 69–81 . 
Dang, S.T. , Meese, E.A. , Morud, J.C. , Johansen, S.T. , 2019. Numerical approach for 
generic three-phase ﬂow based on cut-cell and ghost ﬂuid methods. Int J Numer 
Meth Fl 91, 419–447 . 
Daramizadeh, A. , Ansari, M.R. , 2015. Numerical simulation of underwater explosion 
near air–water free surface using a ﬁve-equation reduced model. Ocean Eng 110, 
25–35 . 
Deligant, M. , Specklin, M. , Khelladi, S. , 2015. A naturally anti-diffusive compressible 
two phases Kapila model with boundedness preservation coupled to a high or- 
der ﬁnite volume solver. Comput Fluids 114, 265–273 . 
Fedkiw, R.P. , Aslam, T. , Merriman, B. , Osher, S. , 1999. A Non-oscillatory Eulerian Ap- 
proach to Interfaces in Multimaterial Flows (the Ghost Fluid Method). Journal 
of Computational Physics 152, 457–492 . 
Ge, L. , Zhang, A.-M. , Zhang, Z.-Y. , Wang, S.-P. , 2019. Numerical simulation of com- 
pressible multiﬂuid ﬂows using an adaptive positivity-preserving RKDG-GFM 
approach. Int J Numer Meth Fl 91, 615–636 . 
Gibou, F. , Fedkiw, R. , Osher, S. , 2018. A review of level-set methods and some recent 
applications. Journal of Computational Physics 353, 82–109 . 
Glimm, J. , Graham, M.J. , Grove, J. , Li, X.L. , Smith, T.M. , Tan, D. , Tangerman, F. , 
Zhang, Q. , 1998. Front tracking in two and three dimensions. Comput Math Appl 
35, 1–11 . 
Goncalvès, E. , 2013. Numerical study of expansion tube problems: Toward the sim- 
ulation of cavitation. Comput Fluids 72, 1–19 . 
Grove, J.W. , Menikoff, R. , 1990. Anomalous reﬂection of a shock wave at a ﬂuid in- 
terface. J Fluid Mech 219, 313–336 . 
Ha, C.-T. , Park, W.-G. , Jung, C.-M. , 2015. Numerical simulations of compressible ﬂows 
using multi-ﬂuid models. Int J Multiphas Flow 74, 5–18 . 
Haas, J.F. , Sturtevant, B. , 1987. Interaction of weak shock waves with cylindrical and 
spherical gas inhomogeneities. J Fluid Mech 181, 41–76 . 
Heyns, J.A. , Malan, A.G. , Harms, T.M. , Oxtoby, O.F. , 2013. Development of a compres- 
sive surface capturing formulation for modelling free-surface ﬂow by using the 
volume-of-ﬂuid approach. Int J Numer Meth Fl 71, 788–804 . 
Hu, X.Y. , Wang, Q. , Adams, N.A. , 2010. An adaptive central-upwind weighted essen- 
tially non-oscillatory scheme. Journal of Computational Physics 229, 8952–8965 . 
Ivings, M.J. , Causon, D.M. , Toro, E.F. , 1998. On Riemann solvers for compressible liq- 
uids. Int J Numer Meth Fl 28, 395–418 . 
Jagadeesh, G. , 2008. Industrial applications of shock waves. Proceedings of the Insti- 
tution of Mechanical Engineers, Part G: Journal of Aerospace Engineering 222, 
575–583 . 
Jiang, G.-S. , Shu, C.-W. , 1996. Eﬃcient Implementation of Weighted ENO Schemes. 
Journal of Computational Physics 126, 202–228 . 
Johnsen, E. , Colonius, T.I.M. , 2009. Numerical simulations of non-spherical bubble 
collapse. J Fluid Mech 629, 231–262 . 
Kah, D. , Emre, O. , Tran, Q.H. , de Chaisemartin, S. , Jay, S. , Laurent, F. , Massot, M. , 
2015. High order moment method for polydisperse evaporating sprays with 
mesh movement: Application to internal combustion engines. Int J Multiphas 
Flow 71, 38–65 . 
Kapila, A.K. , Menikoff, R. , Bdzil, J.B. , Son, S.F. , Stewart, D.S. , 2001. Two-phase mod- 
eling of deﬂagration-to-detonation transition in granular materials: Reduced 
equations. Phys Fluids 13, 3002–3024 . 
Kawai, S. , Terashima, H. , 2011. A high-resolution scheme for compressible multicom- 
ponent ﬂows with shock waves. Int J Numer Meth Fl 66, 1207–1225 . 
Kim, D. , Kim, J. , 2019. Numerical method to simulate detonative combustion of hy- 
drogen-air mixture in a containment. Eng Appl Comp Fluid 13, 938–953 . 
Kim, K.-H. , Chahine, G. , Franc, J.-P. , Karimi, A. , 2013. Advanced Experimental and 
Numerical Techniques for Cavitation Erosion Prediction. Springer Dordrecht Hei- 
delberg New York, London . 
Kleine, H. , Tepper, S. , Takehara, K. , Etoh, T.G. , Hiraki, K. , 2009. Cavitation induced by 
low-speed underwater impact. In: Hannemann, K., Seiler, F. (Eds.), Shock Waves. 
Springer Berlin Heidelberg, Berlin, Heidelberg, pp. 895–900 . 
Koukouvinis, P. , Gavaises, M. , Supponen, O. , Farhat, M. , 2016. Simulation of bubble 
expansion and collapse in the vicinity of a free surface. Phys Fluids 28, 052103 . 
Kreeft, J.J. , Koren, B. , 2010. A new formulation of Kapila’s ﬁve-equation model for 
compressible two-ﬂuid ﬂow, and its numerical treatment. Journal of Computa- 
tional Physics 229, 6220–6242 . 
Le Métayer, O. , Massoni, J. , Saurel, R. , 2005. Modelling evaporation fronts with reac- 
tive Riemann solvers. Journal of Computational Physics 205, 567–610 . 
Lin, J. , Ding, H. , Lu, X. , Wang, P. , 2017. A Comparison Study of Numerical Methods 
for Compressible Two-Phase Flows. Advances in Applied Mathematics and Me- 
chanics 9, 1111–1132 . 
Liu, C. , Hu, C. , 2017. Adaptive THINC-GFM for compressible multi-medium ﬂows. 
Journal of Computational Physics 342, 43–65 . 
21 


V.-T. Nguyen, T.-H. Phan and W.-G. Park 
International Journal of Multiphase Flow 135 (2021) 103542 
Murrone, A. , Guillard, H. , 2005. A ﬁve equation reduced model for compressible two 
phase ﬂow problems. Journal of Computational Physics 202, 664–698 . 
Nguyen, N.T. , Dumbser, M. , 2015. A path-conservative ﬁnite volume scheme for 
compressible multi-phase ﬂows with surface tension. Appl Math Comput 271, 
959–978 . 
Nguyen, V.-T. , Nguyen, N.T. , Phan, T.-H. , Park, W.-G. , 2020. Eﬃcient three-equation 
two-phase model for free surface and water impact ﬂows on a general curvilin- 
ear body-ﬁtted grid. Comput Fluids, 104324 . 
Nguyen, V.-T. , Park, W.-G. , 2015. A free surface ﬂow solver for complex three-di- 
mensional water impact problems based on the VOF method. Int J Numer Meth 
Fl 82, 3–34 . 
Nguyen, V.-T. , Park, W.-G. , 2016. A free surface ﬂow solver for complex three-di- 
mensional water impact problems based on the VOF method. Int J Numer Meth 
Fl 82, 3–34 . 
Nguyen, V.-T. , Park, W.-G. , 2017. A volume-of-ﬂuid (VOF) interface-sharpening 
method for two-phase incompressible ﬂows. Comput Fluids 152, 104–119 . 
Nguyen, V.-T. , Thang, V.-D. , Park, W.-G. , 2018. A novel sharp interface capturing 
method for two- and three-phase incompressible ﬂows. Comput Fluids 172, 
147–161 . 
Nguyen, V.-T. , Vu, D.-T. , Park, W.-G. , Jung, C.-M. , 2016. Navier–Stokes solver for wa- 
ter entry bodies with moving Chimera grid method in 6DOF motions. Comput 
Fluids 140, 19–38 . 
Nguyen, V.-T. , Vu, D.-T. , Park, W.-G. , Jung, Y.-R. , 2014. Numerical analysis of wa- 
ter impact forces using a dual-time pseudo-compressibility method and vol- 
ume-of-ﬂuid interface tracking algorithm. Comput Fluids 103, 18–33 . 
Osher, S. , Sethian, J.A. , 1988. Fronts propagating with curvature-dependent speed: 
Algorithms based on Hamilton-Jacobi formulations. Journal of Computational 
Physics 79, 12–49 . 
Phan, T.-H., Nguyen, V.-T., Park, W.-G., 2019. Numerical study on dynamics of an 
underwater explosion bubble based on compressible homogeneous mixture 
model. Comput Fluids 191, 104262. doi: 10.1016/j.compﬂuid.2019.104262 . 
Pontes, J. , Mangiavacchi, N. , Anjos, G.R. , 2019. An Introduction to Compressible Flows 
with Applications: Quasi-One-Dimensional Approximation and General Formu- 
lation for Subsonic, Transonic and Supersonic Flows. SpringerBriefs in Mathe- 
matics . 
Richard, S. , Petitpas, F. , Berry, R.A. , 2009. Simple and eﬃcient relaxation methods for 
interfaces separating compressible ﬂuids, cavitating ﬂows and shocks in multi- 
phase mixtures. Journal of Computational Physics 228, 1678–1712 . 
RuiHan, Tao , A-ManZhang, L. , ShuaiLi , 2019. A three-dimensional modeling for coa- 
lescence of multiple cavitation bubbles near a rigid wall. Phys Fluids 31, 062107 . 
Saurel, R. , Lemetayer, O. , 2001. A multiphase model for compressible ﬂows with in- 
terfaces, shocks, detonation waves and cavitation. J Fluid Mech 431, 239–271 . 
Saurel, R. , Pantano, C. , 2018. Diffuse-Interface Capturing Methods for Compressible 
Two-Phase Flows. Annu Rev Fluid Mech 50, 105–130 . 
Shukla, R.K. , Pantano, C. , Freund, J.B. , 2010. An interface capturing method for the 
simulation of multi-phase compressible ﬂows. Journal of Computational Physics 
229, 7411–7439 . 
Shyue, K.-M. , Xiao, F. , 2014. An Eulerian interface sharpening algorithm for com- 
pressible two-phase ﬂow: The algebraic THINC approach. Journal of Computa- 
tional Physics 268, 326–354 . 
So, K.K. , Hu, X.Y. , Adams, N.A. , 2012. Anti-diffusion interface sharpening technique 
for two-phase compressible ﬂow simulations. Journal of Computational Physics 
231, 4304–4323 . 
Taylor, E.M. , Wu, M. , Martín, M.P. , 2007. Optimization of nonlinear error for 
weighted essentially non-oscillatory methods in direct numerical simulations of 
compressible turbulence. Journal of Computational Physics 223, 384–397 . 
Thornber, B. , Groom, M. , Youngs, D. , 2018. A ﬁve-equation model for the simulation 
of miscible and viscous compressible ﬂuids. Journal of Computational Physics 
372, 256–280 . 
Tiwari, A. , Freund, J.B. , Pantano, C. , 2013. A diffuse interface model with immiscibil- 
ity preservation. Journal of Computational Physics 252, 290–309 . 
Tiwari, A. , Pantano, C. , Freund, J.B. , 2015. Growth-and-collapse dynamics of small 
bubble clusters near a wall. J Fluid Mech 775, 1–23 . 
Toro, E.F. , 2009. Riemann Solvers and Numerical Methods for Fluid Dynamics. A 
practical Introduction. Springer-Verlag, Berlin Heidelberg . 
Ubbink, O. , Issa, R.I. , 1999. A Method for Capturing Sharp Fluid Interfaces on Arbi- 
trary Meshes. Journal of Computational Physics 153, 26–50 . 
Vu, T.V. , Homma, S. , Tryggvason, G. , Wells, J.C. , Takakura, H. , 2013. Computations 
of breakup modes in laminar compound liquid jets in a coﬂowing ﬂuid. Int J 
Multiphas Flow 49, 58–69 . 
Wu, W.B. , Zhang, A.M. , Liu, Y.L. , Wang, S.P. , 2019. Local discontinuous Galerkin 
method for far-ﬁeld underwater explosion shock wave and cavitation. Appl 
Ocean Res 87, 102–110 . 
Xiao, F. , Honma, Y. , Kono, T. , 2005. A simple algebraic interface capturing scheme 
using hyperbolic tangent function. Int J Numer Meth Fl 48, 1023–1040 . 
Youngs, D.L. , 1982. Time dependent multi material ﬂow with large ﬂuid distortion. 
Num. Methods for Fluid Dynamics, N.Y 273–285 . 
Zhang, D. , Jiang, C. , Liang, D. , Chen, Z. , Yang, Y. , Shi, Y. , 2014. A reﬁned vol- 
ume-of-ﬂuid algorithm for capturing sharp ﬂuid interfaces on arbitrary meshes. 
Journal of Computational Physics 274, 709–736 . 
22 
