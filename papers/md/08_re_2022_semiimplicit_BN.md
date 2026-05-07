
## Non-equilibrium Model for Weakly Compressible Multi-component Flows: the Hyperbolic Operator

Barbara Re �and R´emi Abgrall

Universit¨a Z¨urich, Institute of Mathematics, Z¨urich, CH8057, Switzerland, barbara.re@math.uzh.ch

Abstract. We present a novel pressure-based method for weakly compressible multiphase flows, based on a non-equilibrium Baer and Nunziato-type model. In this work, we describe the hyperbolic operator, thus we do not consider relaxation terms. The acoustic part of the governing equations is treated implicitly to avoid the severe restriction on the time step imposed by the CFL condition at low-Mach. Particular care is taken to discretize the non-conservative terms to avoid spurious oscillations across multi-material interfaces. The absence of oscillations and the agreement with analytical or published solutions is demonstrated in simplified test cases, which confirm the validity of the proposed approach as a building block on which developing more accurate and comprehensive methods.

Keywords: Non-equilibrium Multiphase Flows, Low-Mach, Diffuse Interface Methods


### 1 Introduction

Weakly compressible multiphase (or multi-component) flows may arise in several applications, as for instance the transport of CO2 for carbon capture and storage (CCS) [1], in incidental configurations of water nuclear power plants [2], or in additive manufacturing. The former application, in particular, has motivated the present work. In normal-operating conditions, the transport pipelines are primary designed to contain liquid or super-critical CO2, but multiphase flows may occur due to fluctuating CO2 supply or during transient events [1]. The transported flow may contain also diverse impurities, whose state is not necessarily the same of the main component. In this regime, although the velocity is considerably smaller than the speed of sound, i.e. the Mach number is very low, compressibility effects cannot be neglected. The design, safety assessment and performance analysis of these and similar applications could undoubtedly benefit from efficient and robust computation fluid dynamics (CFD) tools. From the numerical point of view, the standard schemes widely used for inviscid compressible flows fields containing shock waves, contact discontinuities, and strong rarefactions, are inadequate for weakly compressible flows, even singlephase ones. The causes of this failure, usually referred to as low-Mach limit, are


# arXiv:1911.00270v1  [physics.comp-ph]  1 Nov 2019

2 B. Re, R. Abgrall

multiple. First, the use of an explicit time integration scheme increases dramatically the computational time. Indeed, at low Mach, the problem is stiff and the acoustic effects result in a severe time step restriction imposed by the CFL condition [3]. A further difficulty concerns accuracy. Generally, approaching the limit M = 0, the numerical solution of the compressible Euler equations does not converge to solutions satisfying the equations for incompressible flows, except for a restricted class of special initial data (the so-called well-prepared case). In particular, when a flux difference splitting method is used, the discrete scheme cannot recover the correct scaling of the pressure with the Mach number and this error increases as the Mach number goes to zero [4].

The low-Mach limit has been investigated since long time in single-phase CFD and the literature presents several contributions on how to develop a numerical method that provides accuracy and efficiency for weakly compressible flows. The proposed answers can be grouped roughly into two main strategies. The first one improves the low-Mach behavior of compressible schemes by means of preconditioning or artificial viscosity [5]. This approach leads to problem-specific solutions, which can be hardly generalized, and increases the computational burden. Alternatively, compressibility can be introduced into the incompressible schemes, by treating implicitly the acoustics and circumventing the issues related to the stiffness of the governing equations [6]. This latter research field was pioneered by Harlow and Amsden [7,8] and it has represented the core of several recent and outgoing research activities aimed at developing Mach-uniform schemes, e.g., [9,10,11,12]. In this work, we exploit some of the techniques proposed within this branch of research to build a robust and flexible method for weakly compressible, non-equilibrium multiphase flows.

While conceiving a CFD tool for multiphase flows, two of the main challenges that we have to handle are the multiscale nature of the field and the presence of the dynamic interfaces that separate distinct fluids (or phases). These issues can be addressed in numerous ways: among them, we can distinguish interface-tracking methods and interface capturing methods. The former ones, as for instance level-set, volume-of-fluid, front-tracking, and ad hoc arbitrary Lagrangian-Eulerian methods, fully resolve the interfaces and smooth out the fluid properties across them. The latter ones dynamically capture the interface as part of the numerical solution, by treating each fluid as a separate continuous. These methods are also known as multi-fluid or diffuse interface method [13]. This work focuses on this second class, which is more suitable to deal with the dynamic creation of interfaces. More specifically, we adopt a Baer and Nunziato (BN)-type model [14] (also known as 7-equation), which copes with the multitude of scales characterizing multiphase flows by means of averaging procedures. Assuming non-equilibrium of phasic pressures, velocities and internal energies, it allows the most general description of multiphase flows and the description of each component through its own thermodynamic model.

Some research activities has been already devoted to weakly compressible multiphase flows, but most of them focus on simplified BN-type models in which equilibrium in the pressure [15] and also in the velocity and the temperature [16]

Weakly Compressible Multi-component Flows: the Hyperbolic Operator 3

is imposed. On the contrary, Coquel et al. [2] adopt a full non-equilibrium BNtype model, but they make the simplifying assumption of isentropic flows. A part from the broader generality, a nice property that motivates the use of BN model is the hyberbolicity, which is not guaranteed by the 6-equation one-pressure model. In this work, we work with the symmetric variant proposed by Saurel and Abgrall [17], which includes pressure and velocity relaxation terms to model how the phasic equilibrium is reached at the interface. The goal of the paper is to present a preliminary 1D version of a novel numerical method for weakly compressible, non-equilibrium multiphase flows. Accordingly, we present how we have derived the model and the numerical discretization in Sec. 2. Particular care has been devoted to the discretization of the non-conservative terms involving the gradient of the volume fraction, which couple the evolution of the different phases. Then, in Sec. 3, we verify its validity in simplified test cases, aimed at recovering the single-phase behavior and verifying the correct discretization of non-conservative terms, i.e. the absence of spurious oscillations across multi-material interfaces. Finally, Sec. 4 draws the conclusions and outlines the future steps towards the long-term work.


### 2 Numerical Method

A winning strategy to attain an efficient numerical method for weakly compressible flows consists in approximating the sonic terms, which in the low-Mach limit are associated with an almost infinite sound propagation rate, in an implicit fashion. Typically, the methods that pursue this strategy for single-phase flows exploit a staggered description of the flow variables to avoid spurious pressure oscillations, and they solve the governing equations in a segregate way by updating the variables to the next time level through intermediate sub-steps. In low-Mach regimes, pressure-based methods are considered more suitable than density-based ones, because of the weak coupling between these two variables. Indeed, the density is approximately constant, while the pressure not, so if density is used as primary variable, small errors on it results in large errors in the pressure. In the followings, we illustrate how we have conceived a pressure-based, segregated strategy for non-equilibrium multiphase flows.

2.1 The continuous model

The first key ingredient of the proposed model is a pressure decomposition, which is required to retrieve the correct order of pressure fluctuations and to converge to correct approximation of the incompressible Euler equations [3]. We replicate here the choice made in [9] and we define the dimensionless pressure as


$$
�P −�Pr
ρru2r
(1)
$$

4 B. Re, R. Abgrall

where ρ is the density, u the velocity, and the subscript r and the accent�· indicate reference and dimensional quantities, respectively. Implicitly, this scaling filters out the acoustics, as it postulates an asymptotic expansion of the pressure as

P = P (0) + M 2 r P (1) + O(M 3 r ) with M 2 r = �ρr�u2 r �Pr

where Mr is a reference Mach number expressing the overall compressibility of the flow field. We can interpret this scalding also as a decomposition between the thermodynamic pressure ( �Pr) and a component that satisfies the momentum equation [3,10]. We want to apply this scaling to the Baer and Nunziato model. As told before, we work with the symmetric variant proposed by Saurel and Abgrall [17], which gives the possibility to use the same equations and the same numerical methods at all computational cells. In this preliminary work, we focus on the hyperbolic part, so we do not consider the relaxation terms. Consequently, considering only 2 phases (denoted with subscripts i and i∗), the dimensional form of the governing equations in 1D is


$$
∂αi
$$


$$
∂�t + �uI
∂αi
$$


$$
∂�x = 0
(2a)
$$


$$
∂(αi�ρi)
$$


$$
∂�t
+ ∂(αi �mi)
$$


$$
∂�x
(2b)
$$


$$
∂(αi �mi)
$$


$$
∂�t + ∂(αi �mi�ui + αi �Pi)
$$


$$
∂�x
∂αi
$$


$$
∂�x
(2c)
$$


$$
∂(αi �Ei)
$$


$$
∂�t + ∂ � αi( �Ei + �Pi) �mi/�ρi �
$$


$$
∂�x
∂αi
$$


$$
∂�x
(2d)
$$

where the last three equations are repeated also for the phase i∗. This system is written for the volume fraction α and for the conservative variables ρ, m, and E = e + m2/(2ρ) of each phase. In addition to the standard Euler equations, the system includes transport terms that stem from the averaging process that underlies BN-type model. To close the system, we have to define the interfacial pressure and velocity (PI and uI). Following [17], we define them as the weighted averages among the phases, i.e. PI = �

i αiPi and uI = � i αimi/(αiρi), although different alternatives are available in the literature. The system (2) has to be complemented by a thermodynamic model for each fluid. In this preliminary work, we use the stiffened gas model, but we leave the formulation as general as possible, to be able to easily add more accurate equations of state in the future. Since our goal is the development of a pressure-based method, we need to formulate the governing equations in primitive variables instead of conservative ones. We are aware of the possible problems that may arise from working with a non-conservative formulation, but at the moment our targets are low-Mach flows, so not experiencing strong shock waves; then a local correction is sufficient to converge to the correct weak solution. Recently, this idea has been

Weakly Compressible Multi-component Flows: the Hyperbolic Operator 5

successfully applied to multiphase flows modeled according to the simplified 5equations model [18]. Therefore, we substitute Eq (2d) with the following:


$$
αi
∂�Pi
$$


$$
∂�t + α�ui
∂�Pi
$$


$$
∂�x + αi�ρi�c2
∂�ui
$$


$$
∂�x = �ρi�c2
i,I(�uI −�ui)∂αi
$$


$$
∂�x
(3)
$$

where ci is the standard speed of sound of phase i and ci,I is a kind of interfacial

speed of sound, which we have defined as c2 i,I = χi+κi PI+ei


$$
ρi
, with χ =
∂P
$$


$$
∂ρ �
$$

e and


$$
κ =
�∂P
∂e
$$

ρ. We highlight that this model, as all 7-equation BN-type ones, does not require to define a speed of sound for the mixture, which can be considered as an open-issue in multiphase research. The final step to formulate the target model is to make dimensionless the BN-type model given by (2a)–(2c), and (3) according to the scaling given by (1). The variables that do not involve pressure are scaled as usual, according to the reference density, velocity, and length (Lr). On the other hand, the scaling of the thermodynamic variables, and in particular of the speed of sound, requires more care. Inserting Eq. (1) into the definition of �c2 i , we have


$$
�c2 = � χ + κP + e
$$


$$
ρ
$$


$$
� �u2 r + κ � Pr ρ�ρr = � c2 + κPr
$$


$$
ρ
$$


$$
� �u2 r (4)
$$

where c2 = χ+κ P +e

ρ has the same expression of the dimensional speed of sound,

and Pr = �Pr/(�u2 r �ρr is the dimensionless reference pressure. A similar result is obtained also for �c2 i,I. Omitting all the passages for brevity, the dimensionless formulation reads


$$
∂αi
$$


$$
∂t + uI
∂αi
$$


$$
∂x = 0
(5a)
$$


$$
∂(αiρi)
$$


$$
∂t
+ ∂(αimi)
$$


$$
∂x
(5b)
$$


$$
∂(αimi)
$$


$$
∂t
+ ∂(αimiui + αiPi)
$$


$$
∂x
∂αi
$$


$$
∂x
(5c)
$$

M 2 r


$$
αi
∂Pi
$$


$$
∂t + αiui
∂Pi
$$


$$
∂x + αiρic2
∂ui
$$


$$
∂x
$$


$$
� = M 2 r ρic2 i,I
$$


$$
� (uI −ui)∂αi
$$


$$
∂x
$$

�


$$
+ κi
$$


$$
� uI ∂αi
$$


$$
∂x −∂(αiui)
$$


$$
∂x
$$


$$
� . (5d)
$$

We can observe that the first three equations are identical to their dimensional counterparts, while the pressure equation has a special expression, thanks to which, as Mr →0, we have ∂αi


$$
∂t + ∂(αiui)
$$

∂x = 0. This can be viewed as the multiphase counterpart of the kinematic constraint for the incompressible single phase flows ∇· u = 0.

2.2 The discretization

As common in low-Mach schemes for single phase flows, we exploit a staggered description of the flow variables to avoid spurious pressure oscillations. This

6 B. Re, R. Abgrall

means that the scalar, thermodynamic variables are stored are the cell centered, while the vectorial, velocity-related quantities are stored at the cell faces. Under this configuration, it is natural to solve the governing equations in a segregate way, as the computational cells for the momentum equation are different from the ones for the density and energy equation.

The semi-implicit temporal discretization. The solution at time level tn+1

is computed through intermediate sub-steps in which the convective velocity is approximated explicitly. Moreover, the computation of the momentum is split into two sub-steps: first an intermediate momentum m∗is computed by approximating explicitly the pressure in (5c), then it is updated when the pressure P n+1

is known. More precisely, the following steps are performed for each phase:

i) the density equation (5b) is solved by using un i ; ii) the volume fraction equation (5a) is solved by using un I (only for the phase i, then αi∗= 1 −αi); iii) the intermediate momentum (αm)∗= (αρi)n+1u∗ i is computed solving


$$
(αimi)∗−(αimi)n
$$


$$
∆t + ∂ � (αimi)∗un i + αn+1 i P n i �
$$


$$
∂x = P n I ∂αn+1 i ∂x ; (6)
$$

iv) the pressure-equation (5d) is solved by treating implicitly the velocity divergence (which is mandatory to avoid too severe restrictions on the time step) and explicitly the terms related to the speed of sound, that is


$$
M 2 r αn+1 i
$$


$$
�P n+1 i −P n i ∆t + u∗ i ∂P n+1 i ∂x
$$


$$
� + � M 2 r ρic2 i + κi �n αn+1 i ∂un+1 i ∂x
$$


$$
r ρic2
i,I + κi
�n (uI −ui)∗∂αn+1
∂x
(7)
$$

v) the momentum is updated by solving the difference between (5c) and (6):


$$
(αimi)n+1 −(αimi)∗
$$


$$
∆t + ∂ �� (αimi)n+1 −(αimi)∗� un i �
$$


$$
∂x
$$


$$
+ ∂ � αn+1 i (P n+1 i −P n i ) �
$$


$$
∂x
−P n
I )∂αn+1
∂x
(8)
$$

The staggered spatial discretization. We use a pretty standard first-order, finite-volume discretization with Rusanov fluxes, so we highlight here only the peculiar features of the proposed scheme. Given a 1D domain, we divide it in N s j scalar cells of uniform width ∆xj and N v j = N s j + 1 cells for the vectorial variables, whose width ∆xj+ 1

2 is the equal to ∆xj for the internal cells, except for the first and the last cells which have width ∆xj/2. The equations for α, αρ, P are solved over the scalar (node-centered) cells Cj, while the equations for αm are solved over the vector cells Cj+ 1

2 . When a mapping

Weakly Compressible Multi-component Flows: the Hyperbolic Operator 7

from the vectorial to the scalar cells, or vice versa, is required, we use a weighted average according to the cell sizes, which, in this preliminary work, reduces to a simple mean as we use uniform grids. We denote the spatially discrete values by a further subscript j or j + 1

2, e.g., (αi)n j is the volume fraction of phase i at cell j at time step n. The staggered configurations facilitates the computation of some integral terms: for instance, the velocity on the face between the cells Cj and Cj+1 is unequivocally uj+ 1

2 . Similarly, we can easily discretize the pressure gradient over the cell Cj+ 1

2 through a central difference between Pj+1 and Pj. Particular care is mandatory while discretizing the non-conservative terms, which involves the gradient of α. As known, a superficial discretization of these products may lead to the onset of oscillations across interfaces separating fluids with different material properties [19]. In compressible multiphase flows, a guideline is provided by the so-called Abgrall’s criterion (or pressure nondisturbance condition), which states that “a two-phase flow, uniform in pressure and velocity must remain uniform on the same variables during its temporal evolution” [20]. So, applying the derived discretization to a uniform flow, we can obtain an oscillation-free discretization of the non-conservative terms. This procedure clearly depends on the adopted discretization scheme, but it has been already applied to BN-type models in previous works of Saurel et al. [17,21]. According to the previous remarks, the discrete equations solve at each time step are


$$
i) (αiρi)n+1
= (αiρi)n
j −
∆t
∆xj
$$


$$
� F rus,ρ j+ 1
$$


$$
2 −F rus,ρ
j−1
$$

2


$$
� with (9)
$$


$$
F rus,ρ j+ 1
$$

2 = 1


$$
2 � (αiρi)n+1 j+1 + (αiρi)n+1 j � (ui)n j+ 1
$$


$$
2 −1
$$


$$
2 � (αiρi)n+1 j+1 −(αiρi)n+1 j � |(ui)n j+ 1
$$

2 |

ii) (αi)n+1 j = (αi)n j − ∆t ∆xj Hu(αn+1 i , un I )j (10)

where Hu is the discretization of the non-conservative term. It is obtained by combining this equation with the one at step i): starting from uniform density and velocity, we impose that the density remains constant, obtaining

Hu(αn+1 i , un I )j = 1


$$
2 �� (αi)n+1 j+1 −(αi)n+1 j−1 � (uI)n j
$$


$$
−|(uI)n j | � (αi)n+1 j+1 −2(αi)n+1 j + (αi)n+1 j−1 ��
$$

iii) (αimi)∗ j+ 1


$$
2 = (αimi)n j+ 1
$$


$$
2 −
∆t
∆xj+ 1
$$

2

�� F rus,m j+1 −F rus,m j �


$$
+ � (αi)n+1 j+1 (Pi)n j+1 −(αi)n+1 j (Pi)n j � −HP (αn+1 i , P n I , u∗ i )∗ j+ 1
$$

2


$$
� (11)
$$

where F rus,m j is the standard Rusanov flux for the conservative variables (αimi)∗and the convective velocities (ui)n, while the approximation (HP )∗ j+ 1

2 of the non-conservative term is obtained by imposing the Abgrall’s criterion:


$$
(HP )∗ j+ 1
$$

2 = (PI)n j+ 1


$$
2 � (αi)n+1 j+1 −(αi)n+1 j � + Kvol j+ 1
$$


$$
2 (ui)∗ j+ 1
$$


$$
2 � F rus,ρ,¯u j+1 −F rus,ρ,¯u j �
$$

8 B. Re, R. Abgrall

where Kvol j+ 1


$$
2 = [1−∆xj+ 1
$$

2 � ∆xj] and F rus,ρ,¯u j+1 is defined as in (9) but between the mapped densities (αiρi)j+ 1


$$
2 and (αiρi)j−1
$$

2 . The last term of HP results from the mapping of (αiρj) at the left-hand side,1 but it is non-null only at the vector boundary cells, where ∆xj+ 1


$$
2 ̸= ∆xj.
$$

iv) M 2 r (αi)n+1 j � (Pi)n+1 j −(Pi)n j �∆xj ∆t =


$$
− M 2 r (αi)n+1 j 2 �� (Pi)n+1 j+1 −(Pi)n+1 j �� (ui)∗ j+ 1
$$


$$
2 −|(ui)∗ j+ 1
$$

2 | �


$$
+ � (Pi)n+1 j −(Pi)n+1 j−1 �� (ui)∗ j−1
$$


$$
2 + |(ui)∗ j−1
$$

2 | ��


$$
− � M 2 r (ρi)n j ((ci)n j )2 + (κi)n j � (αi)n+1 j � (ui)n+1 j+ 1
$$


$$
2 −(ui)n+1 j−1
$$

2

�


$$
+ � M 2 r (ρi)n j ((ci,I)n j )2 + (κi)n j � Hu � (u∗ I −u∗ i ), αn+1 i �
$$


$$
j (12)
$$

which has been derived from (7) re-writing u∗∂P n+1


$$
∂x
= ∂P n+1u∗
$$


$$
∂x
−P ∗∂u∗
$$

∂x and using the Rusanov approximation. For analogy with the density equation, the non-conservative term is discretized by the same operator Hu defined in step ii). v) we observe that if in Eq. (8) we define δαimi = (αiρi)n+1(un+1 i −u∗ i ) and δP = P n+1 −P n, it has the same shape of (6). Therefore we obtain the same discretization as in step iii).

We highlight that the velocity in the second right-hand side term of (12) has to be discretized implicitly, but at step iv) the velocities (ui)n+1 are still unknown. However, we can derive an approximation from the expression of step v), discharging the differences in the convective term. Thus, we use

(ui)n+1 j+ 1


$$
2 = (ui)∗ j+ 1
$$


$$
∆t
∆xj+ 1
$$

2


$$
� − � (αi)n+1 j+1 (δPi)j+1 −(αi)n+1 j (δPi)n+1 j �
$$


$$
+ (δPI)n+1 j+ 1
$$


$$
2 � (αi)n+1 j+1 −(αi)n+1 j ��� (αiρi)n+1 j+ 1
$$


$$
2 . (13)
$$

However, the previous expression contains the interfacial velocity P n+1 I , which depends on the pressure at time tn+1 of both phases. Consequently, the use of (13) in (12), i.e. the implicit discretization of the acoustic part, makes the solution of the pressure equations coupled among all phases.


### 3 Results and Discussion

We have tested the proposed numerical method on some simplified problems to verify its correctness. In the presented tests, we use the stiffened gas equation of state, which describes molecular agitation and repulsive effects in a simplified way, but facilitates the analytical solution of some simple problems [22].

1 The discretization of HP is obtained by assuming uniform (ui)n and (Pi)n in (11) and substituting into its left-hand side (αimi)∗ j+ 1


$$
2 = (αiρi)n+1 j+ 1
$$

2 the discretization of

(αiρi)n+1 j+ 1

2 from (9).

Weakly Compressible Multi-component Flows: the Hyperbolic Operator 9


$$
γ
P∞(Pa)
$$

Phase 1 4.4 6.0 108

Phase 2 1.4 0.0

a) Fluids of tests 1 and 3.


$$
cp (Jkg−1K−1)
γ
P∞(Pa)
$$

Phase 1 4186 2.8 8.5 108

Phase 2 1004 1.4 0.0

b) Fluids of test 2. Table 1. Thermodynamic parameters of the two fluids. cp is the isobaric specific heat capacity, γ is the ratio between the isobaric and the isochoric specific heat capacities, and P∞is a parameter of the stiffened gas model. Values are taken from [21,15].

Test 1. To verify the numerical verification of Abgrall’s criterion and the stability at CFL> 1, the first test is a simple transport of a volume fraction variation in a uniform pressure and velocity field. The fluids and the initial conditions are described in Table 1 (left) and Fig. 1. The final time is 1 ms, divided in 80 time steps. The resulting acoustic CFL conditions (i.e., considering u + c) are 10 for fluid 1 and 2.7 for fluid 2. The initial conditions are correctly preserved and no spurious oscillations are generated, as demonstrated in Fig. 1.

-0.5 -0.25 0 0.25 0.5 Position (m)

0.9

1

1.1

Pressure, (bar)

-0.5 -0.25 0 0.25 0.5 Position (m)

99

100

101

Velocity, (m/s)

-0.5 -0.25 0 0.25 0.5 Position (m)

0

0.5

1

Volume fraction)

-0.5 -0.25 0 0.25 0.5 Position (m)

1049

1050

1051

1052

Density phase 1, (kg/m3)

1.1

1.15

1.2

1.25

1.3

Density phase 2, (kg/m3)

Phase 1 Phase 2


> **Fig. 1. Test 1: volume fraction transport in a uniform pressure and velocity field, solution at the final time tF = 1 ms. The domain is x = −0.5 m ≤x ≤0.5 m, discretized with N s j = 500 scalar cells. Initial conditions are: u0 = 100 m/s, P 0 = 105 Pa, ρ0 1 = 1050 kg/m3 and ρ0 2 = 1.2 kg/m3. The volume fraction profile is initially centered at x = 0, with α1 = 0.3 at left and α1 = 0.7 at right.**

Test 2. The second test is a water/air shock-tube with uniform volume fraction α1 = 0.5. Initially, the left and right chambers have different pressures. The difference in pressure is moderate, in order to avoid excessively high Mach numbers. Because of the absence of relaxation terms, each fluid evolves as a single phase, so it is possible to compute the analytical solution, given the fluid parameters in Table 1 and initial conditions in Fig. 2. A good agreement is achieved, even if the rarefaction fans are smeared, but we can expect it since the discretization is

10 B. Re, R. Abgrall

only first-order accurate. We highlight that the pressure variation in the liquid (phase 1) is an acoustic wave and it moves at the speed of sound c1 ≈1525 m/s. Such a high propagation speed makes the capability to implicitly treat acoustics of paramount importance to avoid instabilities or an excessively small time step.

-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 Position (m)

50

60

70

80

90

100

Pressure, (bar)

-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 Position (m)

0

0.5

1

1.5

2

Velocity 1, (m/s)

0

20

40

60

80

100

120

Velocity 2, (m/s)

-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 Position (m)

1020

1025

1030

1035

Density phase 1, (kg/m3)

50

60

70

80

90

Density phase 2, (kg/m3)

-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 Position (m)

0

0.1

0.2

0.3

Mach number

Phase 1, analytical Phase 1, numerical

Phase 2, analytical Phase 2, numerical


> **Fig. 2. Test 2: water/air shock-tube with no phase mixing, solution at the final time tF = 0.2 ms, after 100 time steps. The grid spacing is ∆xj = 0.002 as in the other tests. The initial pressures are P 0 L = 100 bar (left) and P 0 R = 50 bar (right), while the temperature is T 0 = 308.15 K, uniform. The volume fraction is α = 0.5 everywhere. The numerical solution is compared to the analytical one for each phase.**

Test 3. Finally, we reproduce a multiphase test for which the results without relaxation terms are available in [21]. The fluids are the same as in Test 1, but now a pressure difference between the left and right state is also imposed. We can observe a good agreement with results in [21], expect for the pressure profile of phase 1. This difference can be explained by the fact that they use a simplified Riemann solver according to which no pressure wave is present in this phase. Furthermore, we notice a strange behavior of the numerical solution across the interface: in each phase, one point is not aligned with the neighboring ones. We believe that it could be generated by the mapping from the scalar to the vector grid and that it could be smooth down by adding the relaxation terms, however we will further investigate this issue.


### 4 Conclusions

In this work, we have presented a semi-implicit method for multiphase flows based on the Baer and Nunziato model, which avoids severe time step restric-

Weakly Compressible Multi-component Flows: the Hyperbolic Operator 11

-0.6 -0.4 -0.2 0 0.2 0.4 0.6 Position (m)

0

2

4

6

8

10

Pressure, (bar)

-0.6 -0.4 -0.2 0 0.2 0.4 0.6 Position (m)

-1

0

1

2

Velocity 1, (m/s)

0

200

400

600

Velocity 2, (m/s)

-0.6 -0.4 -0.2 0 0.2 0.4 0.6 Position (m)

0

0.2

0.4

0.6

0.8

1

Volume fraction)

-0.6 -0.4 -0.2 0 0.2 0.4 0.6 Position (m)

1020

1030

1040

1050

1060

Density phase 1, (kg/m3)

1

2

3

4

5

Density phase 2, (kg/m3)

Phase 1 Phase 2


> **Fig. 3. Test 3: “Smooth shock tube test case” as in [21], solution at the final time tF = 0.35 ms, after 700 time steps. The initial pressures are P 0 L = 10 bar (left) and P 0 R = 1 bar (right) and the velocity is 0 for both phases. The densities ρ0 1 = 1050 kg/m3**

and ρ0 2 = 1.2 kg/m3 are uniform along the domain.

tions at low-Mach regimes. We have proposed a pressure-based method, which, thanks to a special scaling of the pressure, recovers in the limit Mr →0 the incompressible formulation of the governing equations. Moreover, we have taken particular care in the discretization of the non-conservative terms, in order to avoid spurious oscillations across the multi-material interfaces.

The preliminary tests have confirmed the implicit treatment of acoustic and the fulfillment of the Abgrall’s criterion. These results indicate the applicability of the propose approach and allow us to pass to the next step of the longterm work. Currently, we are working on the inclusion of the relaxation terms. Future steps are the implementation of more accurate thermodynamic models, of secondand high-order discretization. The flexibility of the BN-type model will allow also the investigation of flows containing more than 2 phases.

Acknowledgments. This publication has been produced with support from the NCCS Centre, performed under the Norwegian research program Centres for Environment-friendly Energy Research (FME). The authors acknowledge the following partners for their contributions: Aker Solutions, Ansaldo Energia, CoorsTek Membrane Sciences, Emgs, Equinor, Gassco, Krohne, Larvik Shipping, Norcem, Norwegian Oil and Gas, Quad Geometrics, Shell, Total, V˚ar Energi, and the Research Council of Norway (257579/E20).

12 B. Re, R. Abgrall


## References

1. S.T. Munkejord, M. Hammer, International Journal of Greenhouse Gas Control 37, 398 (2015) 2. F. Coquel, J.M. H´erard, K. Saleh, J. Comput. Phys. 330, 401 (2017). DOI 10. 1016/j.jcp.2016.11.017 3. I. Wenneker, A. Segal, P. Wesseling, International Journal for Numerical Methods in Fluids 40(9), 1209 (2002). DOI 10.1002/fld.417 4. H. Guillard, C. Viozat, Computers & Fluids 28(1), 63 (1999). DOI 10.1016/ S0045-7930(98)00017-6 5. H. Guillard, A. Murrone, Computers and Fluids 33(4), 655 (2004). DOI 10.1016/ j.compfluid.2003.07.001 6. F. Xiao, J. Comput. Phys. 195(2), 629 (2004). DOI 10.1016/j.jcp.2003.10.014 7. F.H. Harlow, A.A. Amsden, J. Comput. Phys. 3(1), 80 (1968). DOI 10.1016/ 0021-9991(68)90007-7 8. F.H. Harlow, A.A. Amsden, J. Comput. Phys. 8(2), 197 (1971). DOI 10.1016/ 0021-9991(71)90002-7 9. H. Bijl, P. Wesseling, Journal of Computational Physics 141(2), 153 (1998). DOI 10.1006/JCPH.1998.5914 10. C.D. Munz, S. Roller, R. Klein, K.J. Geratz, Comput. Fluids 32(2), 173 (2003). DOI 10.1016/S0045-7930(02)00010-5 11. N. Kwatra, J. Su, J.T. Gr´etarsson, R. Fedkiw, Journal of Computational Physics 228(11), 4146 (2009). DOI 10.1016/J.JCP.2009.02.027 12. J. Ventosa-Molina, J. Chiva, O. Lehmkuhl, J. Muela, C.D. P´erez-Segarra, A. Oliva, Int. J. Numer. Methods Fluids 84(6), 309 (2017). DOI 10.1002/fld.4350 13. R. Saurel, C. Pantano, Annual Review of Fluid Mechanics 50, 105 (2018). DOI 10.1146/annurev-fluid-122316-050109 14. M.R. Baer, J.W. Nunziato, Int. J. Multiphase Flow 6, 861 (1986) 15. A.K. Pandare, H. Luo, J. Comput. Phys. 371, 67 (2018). DOI 10.1016/j.jcp.2018. 05.018 16. M. Bernard, S. Dellacherie, G. Faccanoni, B. Grec, Y. Penel, ESAIM Math. Model. Numer. Anal. 48(6), 1639 (2014). DOI 10.1051/m2an/2014015 17. R. Saurel, R. Abgrall, Journal of Computational Physics 150(2), 425 (1999). DOI 10.1006/JCPH.1999.6187 18. R. Abgrall, P. Bacigaluppi, S. Tokareva, Computers & Fluids (2017). DOI 10. 1016/J.COMPFLUID.2017.08.019 19. R. Abgrall, S. Karni, J. Comput. Phys. 229(8), 2759 (2010). DOI 10.1016/j.jcp. 2009.12.015 20. R. Abgrall, J. Comput. Phys. 125(1), 150 (1996). DOI 10.1006/JCPH.1996.0085 21. R. Saurel, A. Chinnayya, Q. Carmouze, Phys. Fluids 29(6), 1 (2017). DOI 10. 1063/1.4985289 22. K.K. Haller, Y. Ventikos, D. Poulikakos, J. Appl. Phys. 93(5), 3090 (2003). DOI 10.1063/1.1543649

