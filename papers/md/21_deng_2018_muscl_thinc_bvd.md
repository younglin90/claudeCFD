Journal of Computational Physics 371 (2018) 945–966


### Contents lists available at ScienceDirect


# Journal of Computational Physics


### www.elsevier.com/locate/jcp


# High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces


# Xi Deng a, Satoshi Inaba a, Bin Xie a, Keh-Ming Shyue b, Feng Xiao a,∗

a Department of Mechanical Engineering, Tokyo Institute of Technology, 2-12-1 Ookayama, Meguro-ku, Tokyo, 152-8550, Japan b Department of Mathematics, National Taiwan University, Taipei 106, Taiwan


## a r t i c l e i n f o a b s t r a c t

Article history: Received 26 October 2017 Received in revised form 23 February 2018 Accepted 25 March 2018 Available online 28 March 2018

Keywords: Compressible multiphase flows Five-equation model Interface capturing THINC reconstruction BVD algorithm Finite volume method

We present in this work a new reconstruction scheme, so-called MUSCL-THINC-BVD scheme, to solve the five-equation model for interfacial two phase flows. This scheme employs the traditional shock capturing MUSCL (Monotone Upstream-centered Schemes for Conservation Law) scheme as well as the interface sharpening THINC (Tangent of Hyperbola for INterface Capturing) scheme as two building-blocks of spatial reconstruction on the BVD (boundary variation diminishing) principle that minimizes the variations (jumps) of the reconstructed variables at cell boundaries, and thus effectively reduces the dissipation error in numerical solutions. The MUSCL-THINC-BVD scheme is implemented to the volume fraction and other state variables under the same finite volume framework, which realizes the consistency among volume fraction and other physical variables. Numerical results of benchmark tests show that the present method is able to capture the material interface as a well-defined sharp jump in volume fraction, and obtain numerical solutions of superior quality in comparison to other existing methods. The proposed scheme is a simple and effective method of practical significance for simulating compressible interfacial multiphase flows.


### © 2018 Elsevier Inc. All rights reserved.


### 1. Introduction

Compressible multiphase fluid dynamics is one of active and challenging research areas of great importance in both theoretical studies and industrial applications. For example, shock/interface interactions are thought to be crucial to the instability and evolution of material interfaces that separate different fluids as observed in a wide spectrum of phenomena [2]. The material interfaces greatly complicate the physics and make problems formidably difficult for analytical and experimental approaches in many cases, where numerical simulation turns out to be the most effective approach to provide quantitative information to elucidate the fundamental mechanisms behind the complex phenomena of multiphase flows.

In comparison to the computation of single phase flow, development of numerical methods for multiphase flow faces more challenging tasks. The major complexity comes from the moving interfaces between different fluids that usually associate with strong discontinuities, singular forces and phase changes in some cases. Given the numerical methods developed for multiphase incompressible flows with interfaces having been reaching a relatively mature stage, the numerical solvers for compressible interfacial multiphase flows are apparently insufficient. For incompressible multiphase flows with moving


# * Corresponding author. E-mail address: xiao .f .aa @m .titech .ac .jp (F. Xiao).

https://doi.org/10.1016/j.jcp.2018.03.036 0021-9991/© 2018 Elsevier Inc. All rights reserved.

946 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966

interfaces where the density and other physical properties, e.g. viscosity and thermal conductivity, are constant in each fluid, the one-fluid model [1] can be implemented in a straightforward manner with an assumption that the physical fields change monotonically across the interface region. Thus, provided with an indication function which identifies the moving interface, one can uniquely determine the physical property fields for the whole computational domain. Some indication functions, such as volume of fluid (VOF) function [3–5] and level set function [6–8], have been proposed and proved to be able to capture moving interfaces with compact thickness and geometrical faithfulness if solved by advanced numerical algorithms. However, substantial barrier exists when implementing the one-fluid model to compressible interfacial multiphase flows.


### The new difficulties we face when applying the one-fluid model1 to compressible interfacial multiphase flows lie in two aspects:

(I) Density and energy in compressible flow have to be solved along with the indication function, and special formulations are required to maintain the physical consistency which results in a balanced state among all variables for the interface cell where a well-defined interface falls in; (II) The numerical dissipation in the so-called high-resolution schemes designed for solving single phase compressible flows involving shock waves tends to smear out discontinuities including the material interfaces in numerical solutions, which is fatal to simulations of interfacial multiphase flows even if the schemes can produce acceptable results in single phase cases.

For issue (I) mentioned above, mixing or averaging models that consist of Euler or Navier–Stokes equations along with the equations of interface-indication functions have been derived and widely used as an efficient approximation to the state of the interface cell where two or more species co-exist. A simple single-fluid model was reported in [9–11] for interfacial multiphase compressible flows using either explicit time marching or semi-implicit pressure-projection solution procedure. The latter results in a unified formulation for solving both compressible and incompressible multiphase flows. As the primitive variables are solved in these models, the conservation properties are not guaranteed, and thus might not be suitable for high-Mach flows involving shock waves. Conservative formulations, which have been well-established for single phase compressible flows with shock waves, however may lead to spurious oscillations in pressure or other thermal fields [12,13]. It was found that special treatments are required in transporting the material interface and mixing/averaging the state variables to find the mixed state of fluids in the interfacial cell that satisfies pressure balance across material interface for multiple polytropic and stiff gases [14–18], van der Waals [19] and Mie–Grüneisen equations of state (EOS) [20]. A more general five-equation model [21] was developed for a wide range fluids. These models apply to multiphase compressible flows with either spread or sharp interfaces. See [22] for a recent review on the models of this sort. We make use of the five-equation model in the present work as the PDE (partial differential equation) set to develop our numerical method, which can be applied to other extended system as well.

Provided the SEF models with some desired properties, such as hyperbolicity, conservation and well-balanced mixing closure without spurious oscillations in thermal variables, we can in principle implement numerical methods for single phase compressible flow (e.g. the standard shock-capturing schemes) to solve these multiphase models. TVD (Total Variation Diminishing) schemes, such as the MUSCL (Monotone Upstream-centered Schemes for Conservation Law) scheme [23], can solve discontinuities without numerical oscillations, which is of paramount importance to ensure the physical fields to be bounded and monotonic in the transition region. However, TVD schemes suffer from excessive numerical dissipation, which brings the problem (II) listed above to us. The intrinsic numerical dissipation smears out the flow structures including the discontinuities in mass fraction or volume fraction which is used to represent the material interfaces. Consequently, material interfaces are continuously blurred and smeared out, which is not acceptable in many applications, especially for the simulations that need long-term computation. As a remedy, using higher order schemes, like WENO (Weighted Essentially Non-Oscillatory) scheme [24], to solve compressible multiphase flows is also found in the literature [25,26,55,56], where numerical dissipation is largely reduced, and the moving interfaces, as well as other flow structures, can be resolved with significantly improved accuracy. However, implementing high order schemes might generate numerical oscillations for compressible multiphase flows with complex EOS as discussed in [25], where even though the reconstructions were carried out in terms of the characteristic variables to reduce numerical oscillation, spurious disturbances are still observed when waves are reflected from the material interface. In a more recent work [27], an intermediate state was introduced at each cell edge in characteristic decomposition to suppress numerical oscillations and stabilize computation. Furthermore, high order monotonicity-preserving scheme [28] was used to ensure the bounded value for volume fraction. It is noted that numerical dissipation even reduced in WENO and other high-order schemes still remains in conventional Eulerian advection schemes, which might be problematic in long-term simulations. In general, the implementation of high order shock capturing schemes to interfacial compressible multiphase flows demands further investigations.

There are different numerical methods to identify and compute moving interfaces in compressible multiphase flows, such as [29–31] on moving meshes and [32–36] on fixed meshes. As aforementioned, the VOF-type methods that use

1 More precisely, it should be called single-state model or single-equivalent-fluid (SEF) model[21]. We call such model SEF in the present paper.

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 947

the volume fraction or mass fraction as the identification function of moving interface have been popularly used as well, which are referred to as interface-capturing methods in our context. Interface-capturing methods resolve the interfaces on fixed Eulerian grids and use advection schemes to transport the volume/mass fraction functions. It is well known that conventional Eulerian advection schemes have intrinsic numerical dissipation and tend to smear out the jumps in volume fraction or mass fraction functions which are used to identify the material interfaces between different fluids. In order to keep the compact thickness of material interfaces during computation, special numerical techniques are needed to steepen the jumps in the volume or mass fraction fields. For example, in [37–39] the advection equation of the interface function is treated by artificial compression method. As a post-processing approach, anti-diffusion techniques have also been introduced in [40] and [41]. An alternative approach is to reconstruct the volume fraction under the finite volume framework by using special functions. The THINC (Tangent of Hyperbola for INterface Capturing) method, for example, uses the hyperbolic tangent function [42] to capture the jumps in volume fraction. By virtue of the desirable characteristics of the hyperbolic tangent function in mimicking the jump-like profile of the volume fraction field, the sharp interface can be accurately captured in a simple and efficient way [43,27]. However, when applying interface-sharping methods to the SEF models of multiphase compressible flows, velocity and pressure oscillations may occur across the interface [39,43,41,38,44] due to the inconsistency between the physical variables and the volume fraction field with sharpened or compressed jumps. As stated in [38,39], in contrast to incompressible flows where the densities of fluids are fixed, artificial interface-sharpening scheme cannot be applied alone to volume fraction function in compressible multiphase flows. Modifications to other physical variables have to be made to maintain the consistency among the sharpened volume fraction and other physical fields [38,39,41,44]. In [43], a homogeneous reconstruction has been proposed where the reconstructed volume fraction is used to extrapolate the remaining conservative variables across the interface to ensure the mechanical consistency across the isolated material interfaces.

This work presents a novel methodology to resolve problem (II) addressed above. To alleviate the numerical dissipation in high-resolution schemes for shock capturing, a new scheme for spatial reconstruction has been devised to reduce numerical dissipation so as to maintain the sharpness of the jumps in volume fraction that identify the moving interfaces. The scheme, so-called MUSCL-THINC-BVD, implements the underlying idea of the boundary variation diminishing (BVD) algorithm [45,46] with the traditional MUSCL (Monotone Upstream-centered Schemes for Conservation Laws) scheme and the interface-sharpening THINC scheme as two building-blocks for reconstruction. The BVD algorithm chooses a reconstruction function between MUSCL and THINC, so that the variations (jumps) of the reconstructed variables at cell boundaries are minimized, which effectively removes the numerical dissipations in numerical solutions. More substantially, MUSCL-THINCBVD scheme is applied to the volume fraction and other state variables simultaneously in a finite volume framework, so that the consistency can be realized among the physical variables and volume fraction throughout the solution procedure. Hence the manipulations required in other existing methods to adjust the physical variables to be consistent with the volume fraction, are not needed in the present method. The numerical model is formulated under a standard finite volume framework with a Riemann solver in the wave propagation form [47]. The numerical tests verify the capability of the present method in capturing the material interface as a well-resolved sharp jump in volume fraction. The numerical results for a wide range of benchmark tests in one, two and three dimensions show superior solution quality competitive to other existing methods.

This paper is organized as follows. In Section 2, the governing equations of the five-equation model for two-phase flow with moving interfaces are described. In Section 3, after a brief review of the finite volume method with the Riemann solver in wave-propagation form for solving the quasi-conservative five-equation model, the details of the new MUSCL-THINC-BVD scheme for spatial reconstruction are presented. Numerical results of benchmark tests are presented in comparison with other high-resolution methods in section 4. Some concluding remarks end the paper in Section 5.


### 2. Mathematical model


### 2.1. Governing equations

In this work, the inviscid compressible two-component flows are formulated by the five-equation model developed in [21]. By assuming that the material interface is in equilibrium of mixed pressure and velocity, the five-equation model consists of two continuity equations for phasic mass, a momentum equation, an energy equation and an advection equation of volume fraction as follows


## ∂ ∂t (α1ρ1) + ∇· (α1ρ1u) = 0,


## ∂ ∂t (α2ρ2) + ∇· (α2ρ2u) = 0,


## ∂ ∂t (ρu) + ∇· (ρu ⊗u) + ∇p = 0,


## ∂E


## ∂t + ∇· (Eu + pu) = 0,


# ∂α1


# ∂t + u · ∇α1 = 0,


## (1)

948 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966

where ρk and αk ∈[0, 1] denote in turn the kth phasic density and volume fraction for k = 1, 2, u the vector of particle velocity, p the mixture pressure and E the total energy. When considering more than two-phases, the five-equation model can be extended by supplementing additional continuity equations and volume fraction advection equations for each new phase.


### 2.2. Closure strategy


### To close the system, the fluid of each phase is assumed to satisfy the Mie–Grüneisen equation of state,


# pk (ρk,ek) = p∞,k(ρk) + ρk�k(ρk) � ek −e∞,k(ρk) � , (2)

where �k = (1/ρk)(∂pk/∂ek)|ρk is the Grüneisen coefficient, and p∞,k, e∞,k are the properly chosen states of the pressure and internal energy along some reference curves (e.g., along an isentrope or other empirically fitting curves) in order to match the experimental data of the examined material [48]. Usually, parameters �k, p∞,k and e∞,k can be taken as functions only of the density. This equation of state can be employed to approximate a wide variety of materials including some gaseous or solid explosives and solid metals under high pressure.


### The conservativeness constraints lead to the mixing formula for volume fraction, density and internal energy as follows,


# α1 + α2 = 1, α1ρ1 + α2ρ2 = ρ, α1ρ1e1 + α2ρ2e2 = ρe.


## (3)


### Derived in [20], the mixture Grüneisen coefficient, pressure p∞and internal energy e∞can be expressed as


# α1 �1(ρ1) + α2 �2(ρ2) = 1 �,


# α1ρ1e∞,1(ρ1) + α2ρ2e∞,2(ρ2) = ρe∞,


# α1 p∞,1(ρ1)


# �1(ρ1) + α2 p∞,2(ρ2)


# �2(ρ2) = p∞(ρ)


# �(ρ) ,


## (4)


### under the isobaric assumption. The mixture pressure is then calculated by


## p =


## �


# ρe −

2 �

k=1 αkρke∞,k(ρk) +

2 �


$$
αk
p∞,k(ρk)
$$


# �k(ρk)


## �� 2 �

k=1


# αk �k(ρk). (5)

It should be noted that the mixing rules of Eq. (4) and Eq. (5) ensure that the mixed pressure is free of spurious oscillations across the material interfaces [14,18,16,20,19]. Following the five equations model under isobaric closure [21], the sound speed of mixture could be calculated as the volumetric average of the phasic sound speeds as


# c2 = α1c2 1 + α2c2 2. (6)


### 3. Numerical methods

For the sake of simplicity, the numerical method in one dimension is introduced in this section. Our numerical method can be extended to multidimensions on structured grids directly in the dimension-wise reconstruction fashion. We will first review the finite volume method in the wave propagation form [47] used in this work and then give details of the new MUSCL-THINC-BVD reconstruction scheme. Compared to the classical finite-volume method, the model system which is a quasi-conservative system of equations can be approximated in a consistent and accurate manner with wave propagation method [49,50,43]. However, being a spatial reconstruction scheme for hyperbolic fluxes, MUSCL-THINC-BVD can be implemented in any finite volume framework straightforwardly.


### 3.1. Wave propagation method


### We rewrite the one dimensional quasi-conservative five-equation model (1) as


## ∂q


## ∂t + ∂f(q)


## ∂x + B(q)∂q


## ∂x = 0, (7)


### where the vectors of physical variables q and flux functions f are

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 949


# q = (α1ρ1,α2ρ2,ρu, E,α1)T ,


# f = (α1ρ1u,α2ρ2u,ρuu + p, Eu + pu, 0)T , (8)


### respectively. The matrix B is defined as


## B = diag(0, 0, 0, 0, u), (9)


### where u denotes the velocity component in x direction.

We divide the computational domain into N non-overlapping cell elements, Ci : x ∈[xi−1/2, xi+1/2], i = 1, 2, ..., N, using a uniform grid with the spacing �x = xi+1/2 −xi−1/2. For a standard finite volume method, the volume-integrated average value ¯qi(t) in cell Ci is defined as


## ¯qi(t) ≈1


## �x

xi+1/2 �


$$
xi−1/2
$$


## q(x,t) dx. (10)


## Denoting all the spatial discretization terms in (7) by L(¯q(t)), the semi-discrete version of the finite volume formulation can be expressed as a system of ordinary differential equations (ODEs)


## ∂¯q(t)


## ∂t = L (¯q(t)). (11)


### In the wave-propagation method, the spatial discretization for cell Ci is computed by


## L (¯qi(t)) = −1


## �x


## � A+�qi−1/2 + A−�qi+1/2 + A�qi � (12)

where A+�qi−1/2 and A−�qi+1/2 are the rightand left-moving fluctuations, respectively, and A�qi is the total fluctuation within Ci. Riemann problems are solved to determine these fluctuations. The rightand left-moving fluctuations can be calculated by


## A±�qi−1/2 =

3 �

k=1


## � sk � qL i−1/2, qR i−1/2 ��± W k � qL i−1/2, qR i−1/2 � , (13)

where moving speeds sk and the jumps W k (k = 1, 2, 3) of three propagating discontinuities can be solved by the Riemann solver [51] with the reconstructed values qL i−1/2 and qR i−1/2 computed from the reconstruction functions ˜qi−1(x) and ˜qi(x) to the left and right sides of cell edge xi−1/2, respectively. Similarly, the total fluctuation can be determined by


## A�qi =

3 �

k=1


## � sk � qR i−1/2, qL i+1/2 ��± W k � qR i−1/2, qL i+1/2 � . (14)


### We will describe with details about the reconstructions to get these values, qL i−1/2 and qR i−1/2, at cell boundaries in the next subsection as the core part of this paper.


### In practice, given the reconstructed values qL i−1/2 and qR i−1/2, the minimum and maximum moving speeds


## s1(qL i−1/2, qR i−1/2) and s3(qL i−1/2, qR i−1/2) are computed as


## s1 = min{uL i−1/2 −cL i−1/2, uR i−1/2 −cR i−1/2},


## s3 = max{uL i−1/2 + cL i−1/2, uR i−1/2 + cR i−1/2}, (15)

where cL i−1/2 and cR i−1/2 are the sound speeds calculated by reconstructed values qL i−1/2 and qR i−1/2 respectively. Then the speed of the middle wave is estimated by the HLLC Riemann solver [51] as


## s2 = pR i−1/2 −pL i−1/2 + ρL i−1/2uL i−1/2(s1 −uL i−1/2) −ρ R i−1/2uR i−1/2(s3 −uR i−1/2)


# ρL i−1/2(s1 −uL i−1/2) −ρ R i−1/2(s3 −uR i−1/2) . (16)


### The left-side intermediate state variables q∗L i−1/2 is evaluated by


## q∗L i−1/2 = (uL i−1/2 −s1)qL i−1/2 + (pL i−1/2nL i−1/2 −p∗ i−1/2n∗ i−1/2)


## s2 −s1 , (17)

950 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


### where the vector uL i−1/2 = (uL i−1/2, uL i−1/2, uL i−1/2, uL i−1/2, s2), nL i−1/2 = (0, 0, 1, uL i−1/2, 0), n∗ i−1/2 = (0, 0, 1, s2, 0) and the intermediate pressure may be estimated as


## p∗ i−1/2 = ρL i−1/2(uL i−1/2 −s1)(uL i−1/2 −s2) + pL i−1/2 = ρ R i−1/2(uR i−1/2 −s1)(uR i−1/2 −s2) + pR i−1/2. (18)


### Analogously, the right-side intermediate state variables q∗R i−1/2 is


## q∗R i−1/2 = (uR i−1/2 −s3)qR i−1/2 + (pR i−1/2nR i−1/2 −p∗ i−1/2n∗ i−1/2)


## s2 −s3 . (19)


## Then we calculate the jumps W k(qL i−1/2, qR i−1/2) as


## W 1 = q∗L i−1/2 −qL i−1/2, W 2 = q∗R i−1/2 −q∗L i−1/2, W 3 = qR i−1/2 −q∗R i−1/2.


## (20)

For the non-conservative term, the jump for the volume fraction is simply zero for both W 1, W 3 and αR 1 −αL 1 for W 2 from above equations. Given the spatial discretization, we employ the three-stage third-order SSP (Strong Stability-Preserving) Runge–Kutta scheme [52]


## ¯q∗= ¯qn + �tL �¯qn� ,


## ¯q∗∗= 3


## 4 ¯qn + 1


## 4 ¯q∗+ 1


## 4�tL �¯q∗� ,


## ¯qn+1 = 1


## 3 ¯qn + 2


## 3 ¯q∗+ 2


## 3�tL �¯q∗∗� ,


## (21)

to solve the time evolution ODEs, where ¯q∗and ¯q∗∗denote the intermediate values at the sub-steps. For temporal integration, we have also tested the Euler first-order explicit and second order Runge–Kutta schemes, which produce stable numerical results as the third order SSP Runge–Kutta does. It is observed that the results from first and second-order temporal schemes are slightly diffusive.


### 3.2. MUSCL-THINC-BVD reconstruction

In the previous subsection, the boundary values, qL i−1/2 and qR i−1/2, are left to be determined, which will be presented in this subsection. We denote any single variable for reconstruction by q, which can be primitive variable, conservative variable or characteristic variable.

The values qL i−1/2 and qR i+1/2 at cell boundaries are computed from the piecewise reconstruction functions ˜qi(x) in cell Ci. In the present work, the MUSCL-THINC-BVD reconstruction scheme is designed to capture both smooth and non-smooth solutions. The BVD algorithm makes use of the MUSCL scheme [23] and the THINC scheme [42] as the candidates for spatial reconstruction.


## In the MUSCL scheme, a piecewise linear function is constructed from the volume-integrated average values ¯qi, which reads


# ˜qi(x)MU SC L = ¯qi + σi(x −xi), (22)


# where x ∈[xi−1/2, xi+1/2] and σi is the slope defined at the cell center xi = 1

2(xi−1/2 + xi+1/2). To prevent numerical oscillation, a slope limiter [23,50] is used to get numerical solutions satisfying the TVD property. The reconstructed values at cell boundaries from MUSCL reconstruction are denoted as qR,MU SC L i−1/2 = ˜qi(xi−1/2)MU SC L and qL,MU SC L i+1/2 = ˜qi(xi+1/2)MU SC L. The MUSCL scheme, in spite of popular use in various numerical models, has excessive numerical dissipation and tends to smear out flow structures, which might be a fatal drawback in simulating interfacial multiphase flows.

Being another reconstruction candidate, the THINC scheme [42,53] uses the hyperbolic tangent function, which is a differentiable and monotone function that fits well a step-like discontinuity. The THINC reconstruction function is written as


## ˜qi(x)T H INC = ¯qmin + ¯qmax


## 2


## �


## 1 + θ tanh � β � x −xi−1/2 xi+1/2 −xi−1/2 −˜xi


## ��� , (23)

where ¯qmin = min(¯qi−1, ¯qi+1), ¯qmax = max(¯qi−1, ¯qi+1) −¯qmin and θ = sgn(¯qi+1 −¯qi−1). The jump thickness is controlled by parameter β. Same as [45], β = 1.6 performs well for all numerical tests presented in this paper. When β becomes too smaller, the results tend to be more diffusive while a larger β enforces the anti-diffusion effect and tends to steepen the jumps in the numerical solutions. From our numerical experiments, a β valued from 1.4 to 2.0 is able to give good or acceptable results. In our numerical tests shown later a constant value of β is used in one dimensional problems. For

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 951


> **Fig. 1. Illustration of one possible situation corresponding to |qL,MU SC L i−1/2 −qR,T H INC i−1/2 | + |qL,T H INC i+1/2 −qR,MU SC L i+1/2 | when calculating T BV T H INC i,min .**

multidimensional cases, β depends on the discontinuity orientation. Following the work in [53], parameter in x direction is computed by βx = β|nx| where nx is the x component of the unit normal of the discontinuity. Likewise, the parameters in y and z directions, βy and βz, can be determined with the corresponding components of the unit normal vector ny and nz in a similar way. In order to keep simplicity, a traditional Young’s algorithm described in [54] is used to calculate the unit normal. The unknown ˜xi, which represents the location of the jump center, is computed from ¯qi = 1 �x �xi+1/2 xi−1/2 ˜qi(x)T H INC dx.


### Then the reconstructed values at cell boundaries by THINC function can be expressed by


## qL,T H INC i+1/2 = ˜qi(xi+1/2)T H INC = ¯qmin + ¯qmax


## 2


## �


## 1 + θ tanh(β) + A


## 1 + A tanh(β)


## � ,


## qR,T H INC i−1/2 = ˜qi(xi−1/2)T H INC = ¯qmin + ¯qmax


## 2 (1 + θ A),


## (24)


## where A = B/ cosh(β)−1

tanh(β) , B = exp(θ β(2 C −1)), and C = ¯qi −¯qmin + ϵ


# ¯qmax + ϵ is a mapping factor to project the physical fields onto


# [0, 1]. A small positive, ϵ = 10−20 is introduced to prevent arithmetic failure.

The final reconstruction function is determined by the BVD algorithm, which chooses the reconstruction function between ˜qi(x)MU SC L and ˜qi(x)T H INC so that the variations of the reconstructed values at cell boundaries are minimized. BVD algorithm prefers the THINC reconstruction ˜qi(x)T H INC within a cell where a discontinuity exists. It is sensible that the THINC reconstruction should only be employed when a discontinuity is detected. In practice, we make use of the following conditions as an additional criterion to implement the THINC reconstruction


## δ < C < 1 −δ and (¯qi+1 −¯qi)(¯qi −¯qi−1) > 0, (25)


## where δ is a small positive.


### In present work, we use a modified BVD algorithm that determines the reconstruction function by


## ˜qi(x)BV D = �˜qi(x)T H INC if δ < C < 1 −δ, and (¯qi+1 −¯qi)(¯qi −¯qi−1) > 0, and T BV T H INC i,min < T BV MU SC L i,min ˜qi(x)MU SC L otherwise , (26)


### where the minimum value of total boundary variation (TBV) T BV P i,min for reconstruction function ˜qi(x)P is computed by


## T BV P i,min = min(|qL,MU SC L i−1/2 −qR,P i−1/2| + |qL,P i+1/2 −qR,MU SC L i+1/2 |,|qL,T H INC i−1/2 −qR,P i−1/2| + |qL,P i+1/2 −qR,T H INC i+1/2 |,


## |qL,MU SC L i−1/2 −qR,P i−1/2| + |qL,P i+1/2 −qR,T H INC i+1/2 |,|qL,T H INC i−1/2 −qR,P i−1/2| + |qL,P i+1/2 −qR,MU SC L i+1/2 |), (27)


### where P stands for either T H INC or MU SC L.

Hence, THINC reconstruction function will be employed in the targeted cell if the minimum TBV value of THINC is smaller than that of MUSCL. In Fig. 1, we illustrate one possible situation corresponding to |qL,MU SC L i−1/2 −qR,T H INC i−1/2 | + |qL,T H INC i+1/2 −

qR,MU SC L i+1/2 | when evaluating T BV T H INC i,min . As stated in [45], the BVD algorithm will realize the polynomial interpolation for smooth solution while for discontinuous solution a step like function will be preferred. It is noted that the present BVD algorithm, (26) and (27), is slightly different from that in [45]. The present BVD algorithm minimizes the total BVs at two ends of the target cell. Our numerical tests show that the present BVD algorithm can effectively reduce numerical dissipations and prevent the flow structures from being smeared out as that in [45].

952 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 2. Numerical errors in density computed by the pure MUSCL scheme (a) and the MUSCL-THINC-BVD scheme (b). Solid and dashed lines indicate the slopes of −1 and −2 respectively.**

We have evaluated the numerical errors and convergence rate of the MUSCL-THINC-BVD scheme using the meshrefinement tests. Propagation of acoustic waves in a single gas is considered. The initial condition is set as the same as [26], where an initial perturbation is added to density and pressure field by


# ρ0 = 1 + ηh(x), u0 = 0, p0 = 1


# γ + ηh(x), (28)

where the perturbation function is specified as h(x) = sin8(πx) and η = 10−4. The convergence rate is studied by refining the mesh from N = 10 to N = 80 and evaluated for the density field. The L1 errors for both MUSCL scheme and MUSCLTHINC-BVD scheme are summarized in Fig. 2. As expected, the MUSCL-THINC-BVD scheme realizes the convergence rate of the polynomial-based reconstruction candidate for smooth solutions, and shows a convergence behavior similar to the MUSCL scheme. As discussed and demonstrated in [45], the BVD algorithm prefers the high-order polynomial reconstruction for smooth solution, and using a high-order polynomial reconstruction, such as the WENO scheme in [45], leads to high-order convergence rate. Even though the MUSCL scheme is used in the present BVD algorithm and the convergence rate is thus under 2nd order, the numerical errors in the results of MUSCL-THINC-BVD scheme are significantly reduced compared to the pure MUSCL scheme. Shown in [45], as well as the numerical results for multiphase flows in this paper, by minimizing the reconstructed differences across cell boundaries, a BVD scheme can effectively reduce numerical dissipation and thus better resolve vortical structures, such as the Kelvin–Helmholtz associated with density discontinuities.

It is noted that the MUSCL-THINC-BVD scheme can be used for single-phase flows as showcased in [45] and get substantially improved solution quality. For the SEF model of multiphase flows, if a mixing/averaging model, such as those in [14–21], is designed to maintain the physical balance across material interfaces, implementing MUSCL-THINC-BVD scheme to the resulting SEF model will not generate spurious oscillations in the numerical solution.

As shown later in the numerical tests, discontinuities including material interface can be resolved by the MUSCL-THINCBVD scheme with substantially reduced numerical dissipation in comparison with other existing methods. The material interface can be captured sharply, while any extra step, like anti-diffusion or other artificial interface sharpening techniques used in the existing works [27,43,38,39], is not needed here. More importantly, the MUSCL-THINC-BVD scheme is applied not only to the volume fraction but also to other physical variables, such as the phasic density, which automatically leads to the consistency among the reconstructed physical fields. As observed in our numerical results, no spurious numerical oscillation is generated in the vicinity of material interfaces. It is usually not trivial to prevent numerical oscillations for other anti-diffusion or artificial compression methods aforementioned. For example, in [44,39,43] anti-diffusion post-processing steps are required to adjust the state variables across the material interfaces to get around the oscillations.


### 4. Numerical results

Numerical tests in one-, twoand three-dimensions are presented in this section to verify the proposed MUSCL-THINCBVD scheme in comparison with the WENO scheme. Here we use the WENO scheme in [24] which is one of representative high order shock-capturing schemes. We denote it as WENO-JS in our tests. As addressed in [25], the WENO reconstruction should be implemented for characteristic fields in order to reduce the numerical oscillations, which is not an easy task for

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 953


> **Fig. 3. Numerical results for the passive advection test of a square liquid column at time t = 10 ms. The solid line is the exact solution and the points show the computed solution with 200 mesh cells obtained using different methods. We denote the numerical result from MUSCl-THINC-BVD by MUSCL-BVD and that from [24] by WENO-JS.**


> **Table 1 Material quantities for copper (k = 1) and explosive (k = 2) in Cochran–Chan equation of state (29).**

k ρ0k (kg/m3) B1k (GPa) B2k (GPa) E1k E2k γk Cvk

1 8900 145.67 147.75 2.99 1.99 3 393 J/kg·K 2 1840 12.87 13.42 4.1 3.1 1.93 1087 J/kg·K


### complex equations of state [27]. Whereas, it is found that the MUSCL-THINC-BVD scheme can prevent numerical oscillations even it is implemented to the primitive variables.

The one dimensional tests were conducted on a workstation with a single CPU (Intel(R) Xeon(R) CPU E5-2687W, 3.10GHZ), while two and three dimensional computations were accelerated with a NVIDIA GTX980ti GPU (graphics processing unit) card.


### 4.1. Passive advection of a square liquid column

To evaluate the ability of the proposed scheme to capture interface as well as to maintain the equilibrium of velocity and pressure fields, a simple interface-only problem in one dimension is considered in this test. The problem consists of a square liquid column in gas transported with a uniform velocity u = u0 = 102 m/s under equilibrium pressure p = p0 = 105 Pa in a tube of one meter. For initial condition, liquid is set in the region of x ∈[0.4, 0.6] m and gas is filled elsewhere. We set initially the volume fraction of liquid α1 = 1 −ϵ for the liquid region and α1 = ϵ in the gas region, and the volume fraction of gas is then α2 = 1 −α1. The small positive ϵ is set 10−8 in numerical tests in this paper. The densities for the liquid and gas phases are ρ1 = 103 kg/m3 and ρ2 = 1 kg/m3, respectively.


### To model the thermodynamic behavior of liquid and gas, we use the EOS of the stiffened gas where the materialdependent parameters in (2) are


# �k = γk −1, p∞,k = γkBk and e∞,k = 0,


# with the constants being γ1 = 4.4, B1 = 6 × 108 Pa for the liquid and γ2 = 1.4, B2 = 0 for the gas respectively.

The computations using WENO-JS and MUSCL-THINC-BVD are carried out separately. Periodic boundary condition is used on the left and right boundaries during the computations. Fig. 3 shows numerical results of partial density and pressure fields at time t = 10 ms using a 200-cell mesh with C F L = 0.5. It is obvious that MUSCl-THINC-BVD can solve the sharp interface within only two cells while the WENO scheme, in spite of high-order accuracy, excessively diffuses the interface due to the intrinsic numerical dissipation around discontinuities as other conventional shock capturing schemes. Meanwhile, MUSCL-THINC-BVD can retain the correct pressure equilibrium and particle velocity without spurious oscillations across the interfaces. Any extra procedure is not conducted to sharpen the interface, which is used in other existing works to keep the steepness of the jump in volume fraction field to identify the interface. The MUSCL-THINC-BVD reconstruction is implemented to other state variables, which remains the thermo-dynamical consistency among the physical fields.

954 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 4. Numerical results for a two-phase (solid explosive-copper) impact problem at time t = 85 μs. The solid line is the fine grid solution computed on a mesh of 5000 cells by MUSCL, and the points show the solutions with 200 meshes.**


### 4.2. Two-material impact problem

Following [57,20], the two-phase impact benchmark problem is computed. At the beginning, there is a right-moving copper (phase 1) plate with the speed u1 = 1500 m/s interacting with a solid explosive (phase 2) at rest on the right of the plate under the uniform atmospheric condition which has pressure p0 = 105 Pa and temperature T0 = 300 K throughout the domain. The material properties of the copper and (solid) explosive are modeled by the Cochran–Chan equation of state where in (2) we set the same �k as in the stiffened gas case, but with p∞,k, e∞,k defined by


# p∞,k(ρk) = B1k


# �ρ0k ρk


## �−E1k −B2k


# �ρ0k ρk


## �−E2k ,


# e∞,k(ρk) = −B1k ρ0k (1 −E1k)


# ��ρ0k ρk


## �1−E1k −1


## �


## + B2k ρ0k (1 −E2k)


# ��ρ0k ρk


## �1−E2k −1


## �


## −CvkT0.


## (29)


# Here γk, B1k, B2k, E1k, E2k, Cvk, and ρ0k are material-dependent quantities, see Table 1 for a typical set of values for copper and explosive considered.

The solution of this test is characterized by a left-moving shock wave to the copper, a right-moving shock waves to the inert explosive, and a material interface lying in between that separates these two different materials. We solved this

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 955


> **Fig. 5. Comparisons of numerical results of shock/interface interaction problem between MUSCL-THINC-BVD and WENO schemes at t = 0.07.**


> **Table 2 Comparison between WENO and MUSCL-THINC-BVD regarding to the elapse time for one dimensional tests.**

Test 4.1 Test 4.2 Test 4.3

WENO-JS 9.90 s 3.02 s 8.90 s MUSCL-THINC-BVD 4.23 s 1.74 s 3.72 s

problem with a 200-cell grid and C F L = 0.5 up to t = 85 μs. Fig. 4 shows the results for the partial densities, velocity, and the copper volume fraction of both WENO and MUSCL-THINC-BVD for comparison. Again, MUSCL-THINC-BVD can keep sharp interface without spurious numerical oscillations in velocity fields. It should be noted that due to complicated EOS, characteristic decomposition is conducted as in [27] when implementing the WENO scheme. In previous work [43], there is a slight overshoot on the partial density α1ρ1 on the left of the interface when using THINC method for the volume fraction. This oscillation is not observed in present study due to the consistency in MUSCL-THINC-BVD reconstructions for physical fields.


### 4.3. Shock interface interaction problem

The interaction between a strong shock wave in helium and an air/helium interface has been studied. Typically, such problem is very challenging for some interface tracking methods. For example, the schemes which are not conservative on

956 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 6. Numerical results for a planar Mach 1.22 shock wave in air interacting with a circular R22 gas bubble. Comparisons are made among MUSCLTHINC-BVD, MUSCL and WENO-JS schemes. Displayed are the Schlieren images of density variations at different instants. In each panel, the result of MUSCL-THINC-BVD (upper half) is plotted against the results of MUSCL or WENO-JS (lower half).**

discrete level may miscalculate the position and speed of the waves resulted from the interaction [58]. The initial conditions are the same as [25], where a Mach 8.96 shock wave is generated and traveling in helium toward a material interface with air which is moving toward the shock wave simultaneously. The detail initial configuration is given by


# (α1ρ1, α2ρ2, u0, p0, α1) =


## ⎧ ⎨


## ⎩


## (0.386, 0, 26.59, 100, 1) for −1 ≤x < −0.8 (0.1, 0, −0.5, 1, 1) for −0.8 ≤x < −0.2 (0, 1, −0.5, 1, 0) for −0.2 ≤x < 1 . (30)

The calculation domain is [−1, 1] which is divided by 200 uniform mesh cells. The solutions at t = 0.07 were computed with the CFL number of 0.1. The comparisons of numerical results between MUSCL-THINC-BVD and WENO schemes are presented in Fig. 5. The results from MUSCL-THINC-BVD show much superior solution quality in resolving material interface without obvious numerical oscillations, while some oscillations are observed in the pressure and velocity fields in the results of WENO scheme in the region of the reflected shock wave even although efforts have been made to implement reconstructions to the characteristic variables as also reported in [25].

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 957


> **Fig. 7. Continued part of Fig. 6.**

In order to evaluate the computational efficiency of the MUSCL-THINC-BVD method, we also compare the computational cost in Table 2 for the 1D numerical tests shown above. Without decomposing state variables into characteristic fields, the computation cost of MUSCL-THINC-BVD is thus about half of the WENO scheme.


### 4.4. Two dimensional shock-bubble interactions

In this widely used benchmark test [40,59,60,41,61,62], we investigate the interactions between a shock wave of Mach 1.22 in air and a cylindrical bubble of refrigerant-22 (R22) gas. The experimental results can be referred in [63]. A planar rightward-moving Mach 1.22 shock wave in air impacts a stationary R22 gas bubble with radius r0 = 25 mm. In this numerical test, both air and R22 are modeled as perfect gases. Inside the R22 gas bubble, the state variables are


# (ρ1,ρ2, u, v, p,α1) = (3.863 kg/m3, 1.225 kg/m3, 0, 0, 1.01325 × 105 Pa, 1 −ε),


### while outside the bubble the corresponding parameters are


# (ρ1,ρ2, u, v, p,α1) = (3.863 kg/m3, 1.225 kg/m3, 0, 0, 1.01325 × 105 Pa,ε)

958 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 8. The Schlieren-like images of density fields computed by the anti-diffusion interface sharpening technique [41] (the upper half in each panel) and the MUSCL-THINC-BVD scheme (the lower half in each panel) respectively with the same grid resolution (400 cells along the diameter of initial bubble).**


### in the pre-shock region and


# (ρ1,ρ2, u, v, p,α1) = (3.863 kg/m3, 1.686 kg/m3, 113.5 m/s, 0, 1.59 × 105 Pa,ε)


# in the post-shock region respectively, where ε = 10−8. The mesh size is �x = �y = 1

8 mm which corresponds to a gridresolution of 400 cells across the bubble diameter. Zero-gradient boundary conditions are imposed at the left and right boundaries while symmetric boundaries are imposed at the top and bottom boundaries. Schlieren-type images of density gradient, |∇ρ|, at different time instants are presented in Figs. 6–7, in which comparisons are made among WENO, MUSCL and MUSCL-THINC-BVD schemes. The MUSCL-THINC-BVD scheme maintains much better the compact thickness of the material interfaces and gives large-scale flow structures similar to the results computed from the WENO scheme. Moreover, MUSCL-THINC-BVD scheme is able to reproduce finer flow structures due to largely reduced numerical dissipation. As one of the important features of the flow field, the instability develops along the interface, which then rolls up and produces small filaments as shown in Fig. 6. These fine structures tend to be smeared out by numerical schemes with large numerical dissipation [41] unless high-resolution computational meshes are used. Not only the well-resolved material interface, we can also observe that the reflected shock waves and transmitted shock waves can be captured more clearly by MUSCL-THINCBVD scheme in comparison with the original MUSCL schemes and competitive to the 5th-order WENO shock-capturing scheme. The resolution quality has been improved remarkably by MUSCL-THINC-BVD scheme to reproduce the complex flow features which are easily diffused out by conventional shock capturing schemes.

We compared our numerical results with the published works in literature which were computed on same or finer grids. In Fig. 8, we plot our results on a coarse mesh where the initial diameter is resolved by 400 cells against the results computed by the anti-diffusion interface sharpening technique [41] on the same grid resolution. We further made comparisons in Fig. 9 with the multi-scale sharp interface simulation on a finer grid [62] where 1150 cells are used for the initial bubble diameter. It can be observed that similar small-scale structures have been re-produced by the MUSCl-THINC-BVD scheme with coarser grid resolution.

It should be noted that the Euler equations used to create the numerical results do not include the physical viscosity and surface tension. In the absence of physical viscosity, the solution of Euler equations does not converge, and the vortices will be continuously enforced when refined grid resolution is used [64]. So, it is hard to distinguish the physically-true structures and those due to the numerical schemes. The significance of the schemes with reduced numerical dissipation lies in the real-case applications where the physical dissipation (due to molecular or turbulent mixing) plays an important role. With reduced numerical dissipation, the true flow structures can be more faithfully reproduced under the effects of the physical dissipations.

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 959


> **Fig. 9. Same as Fig. 8, but with the upper half in each panel replaced by the results in [62] where a finer grid (1500 cells along the diameter of initial bubble) was used.**


### 4.5. Underwater explosion

We consider the underwater explosion problem which has been used in [39,38,40,65,60]. This test case involves complicated interactions of an air cavity generated from an initial high pressure region with a planar water-air interface lying above it. The computational domain is [−2, 2] × [−1.5, 2.5] m2. The cylindrical air cavity of 0.24 m in diameter is initially centered at (0, −0.3) with high pressure 109 Pa and high density 1250 kg/m3. The planar water-air interface is in equilibrium under standard atmospheric condition at y = 0. The thermodynamic behavior of water and air is modeled by the stiffened gas with the same equation of state as in Section 4.1. The transparent boundary condition is imposed for the top, left and right boundaries, while for the bottom boundary the reflection condition is implemented. We conducted the simulation on a coarse uniform mesh with 600 × 600 by different schemes.

In Fig. 10, the numerical Schlieren diagrams for the mixture density computed from different schemes at several instants are displayed. The conventional shock-capturing schemes (both MUSCL and the 5th-order WENO) are not able to prevent the water-air interfaces from smearing out, while MUSCL-THINC-BVD scheme can capture the interface sharply throughout the computation. Moreover, compared with original MUSCL scheme, flow structures including transmitted and reflected waves are much better resolved by MUSCL-THINC-BVD scheme, which produces competitive results with high order WENO schemes.

In Fig. 11, the volume fraction contour is presented. The initial circular underwater bubble evolves into an oval-like shape, which is in agreement with previous works [65,60,38]. It is observed that the material interfaces are sharply resolved so that the thin water bridge between the expanding bubble and the ambient air remains even in the later stage of the process, which is quite challenging for other existing methods. For example, the interface sharpening technique reported in [39] fails in resolving this thin bridge with the same grid resolution. As a quantitative comparison, we plot density profile along x = 0 cross-section in Fig. 12 against the results in [38,39] that use finer grid resolution. With a coarser grid, the present results are in good agreement with those in the existing works.

To further illustrate the superiority of the proposed scheme in resolving the material interface, the computation is conducted to a larger time of t = 3.16 ms, which has never been reported in the existing literatures. The material interfaces calculated by different schemes and the distributions of VOF function along x = 0 cross section are presented in Fig. 13–14.

960 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 10. Schlieren images of density for the underwater explosion test at the instants t = 0.95 ms, 1.26 ms, 1.58 ms, 1.90 ms (from top to bottom) computed by MUSCL, WENO-JS and MUSCL-THINC-BVD schemes respectively on a uniform grid of 600 × 600 cells.**

The MUSCL-THINC-BVD method keeps the compact thickness of the transition layer of the interface, while the MUSCL method smears out the interface over a wider band of mesh cells. For this long-time simulation, the intrinsic numerical dissipation of the conventional Eulerian high-resolution schemes for compressible flows diffuses the material interface to an unacceptable extent. However, the resolution of material interface has been substantially improved by the MUSCL-THINCBVD method.


### 4.6. Interaction of Mach 6 shock in air with a water column

This test was suggested in [39] to simulate the interaction of a Mach 6 shock in air with a water column, which has been also used in other works as a benchmark to examine numerical models for multiphase compressible flows [66,65]. The computational domain for this problem is [0, 8] × [−1, 1] discretized by a 2000 × 500 uniform grid. Initially, a right moving Mach 6.0 plane shock is set at x = 1.0. A water column with the diameter 1.124 is centered at (2, 0). Reflection bound-

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 961


> **Fig. 11. Volume fraction contour at different instants for underwater explosion test using the MUSCL-THINC-BVD scheme on a uniform grid with 600 × 600 cells.**


> **Fig. 12. Distribution of density field along x = 0 cross-section for underwater explosion test at t = 0.2 ms. The mesh size of the present computation is h = 0.0067. Comparisons are made with the published work [38,39] where the cell size is h = 0.005.**

aries are imposed for the top and bottom boundaries while transparent boundary conditions are set for the left and right boundaries. The Schlieren diagrams of density at different instants of the numerical solution computed by MUSCL-TINC-BVD scheme are shown in Fig. 15. Compared with Fig. 11 in [39] where fifth-order WENO scheme was applied with artificial compression technique for moving interface, the proposed scheme in present work not only captures the sharp material interface but also resolves more delicate flow structures on the same computational grid.

962 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 13. The material interface calculated by different schemes at t = 3.16 ms for underwater explosion test. Comparisons are made between MUSCL(a) and MUSCL-THINC-BVD(b) schemes.**


> **Fig. 14. Distribution of VOF function along x = 0 cross-section for underwater explosion test at t = 2.55, 3.16 ms. Comparisons are made between MUSCL and MUSCL-THINC-BVD schemes.**


### 4.7. 3D air and helium bubble interaction

Extending the present numerical method to three dimensions is straightforward. Our 3D code has been generated with the CUDA (Compute Unified Device Architecture) toolkit, which can be executed on hardwares with GPU accelerators. We verified the 3D code by solving the air shock and helium bubble interaction benchmark test. A Mach 1.5 shock wave in the air interacts with a helium bubble. Same as the experimental conditions in [67], the density for the air is 1.29 kg/m3 and 0.167 kg/m3 for the helium bubble. The computational domain is 0.305 m long, 0.08 m wide and 0.08 m high. The bubble radius is 0.02 m. The domain is divided by a Cartesian grid of uniform cell, �x = �y = �z = 1 2200 . We plot the surface of the deformed helium bubble, as well as the density and pressure fields on the central cross sections in Fig. 16. The plane shock impacts the helium bubble and deforms it axis-symmetrically into a twin-donut shape associated with complicated flow structures. The moving interfaces that separate air and helium are well resolved without numerical smearing. The comparisons with the experimental observations reported in [67] are shown in Fig. 17. Even though the density difference across the interfaces is relatively smaller in this test case, the bubble shapes, as well as the density disturbances at different instants, are reproduced by MUSCL-THINC-BVD scheme with adequate accuracy.

X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966 963


> **Fig. 15. Schlieren images of density fields for the Mach 6.0 shock-water interaction benchmark test at instants t = 0.5, 0.75, 0.89, 1.1, 1.5, and 2.15 solved by the MUSCL-THINC-BVD scheme on a 2000 × 500 uniform grid.**


### 5. Conclusion remarks

In this work, we implement MUSCL-THINC-BVD scheme to simulate compressible multiphase flows by solving the fiveequation model. This scheme can resolve discontinuous solutions with much less numerical dissipation in comparison with other existing methods, which enables to solve moving interfaces of compact thickness without additional “anti-diffusion” or “artificial compression” manipulation. The MUSCL-THINC-BVD scheme is applied to volume fraction function and other physical variables as a normal finite volume scheme, and the consistency among different physical fields can be realized without any post adjustment.

964 X. Deng et al. / Journal of Computational Physics 371 (2018) 945–966


> **Fig. 16. Numerical results for 3D air-shock and helium-bubble interactions. Displayed are the color maps of density and pressure fields on the central cross-sections and the iso-surface of the volume fraction of 0.5 that represents the moving interface. From top to bottom are the numerical results at 136 μs, 274 μs and 346 μs respectively. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)**

The present method has been verified with widely used benchmark tests in comparison with other existing methods. Numerical results of the tests show a remarkable improvement in solution quality. Compared with the highorder shock-capturing schemes, the new method shows competitive or superior numerical results but with less computational cost. This work provides an effective and practical approach to simulate compressible interfacial multiphase flows.


### Acknowledgement


### This work has been supported in part by JSPS KAKENHI Grant Numbers 15H03916, 15J09915, 17K18838 and MOST 105-2115-M-002-013-MY2.

