arXiv:2303.10020v1  [physics.flu-dyn]  17 Mar 2023
A ﬁve-point TENO scheme with adaptive dissipation based
on a new scale sensor
Haohan Huanga,1, Tian Liangb,1, Lin Fua,b,c,d,∗
aDepartment of Mathematics, The Hong Kong University of Science and Technology, Clear Water
Bay, Kowloon, Hong Kong
bDepartment of Mechanical and Aerospace Engineering, The Hong Kong University of Science
and Technology, Clear Water Bay, Kowloon, Hong Kong
cHKUST Shenzhen-Hong Kong Collaborative Innovation Research Institute, Futian, Shenzhen,
China
dShenzhen Research Institute, The Hong Kong University of Science and Technology, Shenzhen,
China
Abstract
In this paper, a new ﬁve-point targeted essentially non-oscillatory (TENO) scheme
with adaptive dissipation is proposed. With the standard TENO weighting strategy,
the cut-oﬀparameter CT determines the nonlinear numerical dissipation of the re-
sultant TENO scheme. Moreover, according to the dissipation-adaptive TENO5-A
scheme, the choice of the cut-oﬀparameter CT highly depends on the eﬀective scale
sensor. However, the scale sensor in TENO5-A can only roughly detect the discon-
tinuity locations instead of evaluating the local ﬂow wavenumber as desired. In this
work, a new ﬁve-point scale sensor, which can estimate the local ﬂow wavenumber
accurately, is proposed to further improve the performance of TENO5-A. In combi-
nation with a hyperbolic tangent function, the new scale sensor is deployed to the
TENO5-A framework for adapting the cut-oﬀparameter CT, i.e., the local nonlinear
dissipation, according to the local ﬂow wavenumber. Overall, suﬃcient numerical
∗Corresponding author.
Email address: linfu@ust.hk (Lin Fu)
1The ﬁrst two authors contributed equally.
Preprint submitted to XXX
March 20, 2023


dissipation is generated to capture discontinuities, whereas a minimum amount of
dissipation is delivered for better resolving the smooth ﬂows. A set of benchmark
cases is simulated to demonstrate the performance of the new TENO5-A scheme.
Keywords:
TENO, WENO, PDEs, Hyperbolic conservation laws, Low-dissipation
schemes
1. Introduction
For hyperbolic conservation laws, one of the most diﬃcult issues is the develop-
ment of high-order numerical schemes with the capability of capturing discontinuities
sharply and preserving the high-order accuracy in smooth regions. The essentially
non-oscillatory (ENO) [1] scheme has attracted lots of attention since it was proposed.
Among a set of candidate ﬂuxes, the ENO scheme selects the smoothest ﬂux. Unlike
ENO, the weighted ENO (WENO) scheme proposed by Liu et al. [2] uses a non-
linear convex combination of all candidate ﬂuxes, including the non-smooth ﬂuxes.
This weighting strategy ensures the high-order accuracy in the smooth regions and
the ENO property near discontinuities. After that, Jiang and Shu [3] propose the
WENO5-JS scheme by introducing a new smoothness indicator and a novel ﬁnite-
diﬀerence framework. However, further investigation demonstrates that the accuracy
order of the WENO5-JS scheme degenerates near the critical points. To remedy this
drawback, the WENO5-M [4] scheme remaps the weights calculated by WENO5-JS,
and the WENO5-Z [5] scheme introduces a new weighting strategy by employing
the global smoothness indicator. In addition, WENO5-JS generally produces exces-
sive numerical dissipation, which may smear the small-scale structures in the ﬂow
ﬁeld. The WENO-Z+ [6] scheme enhances the contribution of the less smooth can-
didate stencil ﬂux to reduce the numerical dissipation. Recently, Sun et al. [7][8]
devise a method to optimize a class of ﬁnite diﬀerence schemes with the Minimized
2


Dispersion and Controllable Dissipation (MDCD) properties by two independent pa-
rameters. More recently, Sun et al. [9] and Li et al. [10] present a ﬁnite diﬀerence
scheme with minimum dispersion and adaptive dissipation (MDAD) properties by
establishing a correlation between the local wavenumber and numerical dissipation.
Diﬀerent from altering the coeﬃcients of the background linear schemes, the fourth-
and ﬁfth-order weighted compact nonlinear schemes (WCNS) [11] are developed by
employing the compact schemes as the background schemes. The Hermite WENO
(HWENO) [12] schemes are proposed based on the Hermite polynomials. Beneﬁting
from the compactness of the reconstruction in these schemes, the three-point recon-
struction can generate a ﬁfth-order accuracy scheme. Furthermore, Cai et al. [13]
apply the positivity-preserving techniques in the ﬁnite volume HWENO schemes for
enhancing numerical stability. However, for the HWENO schemes, both the function
values and the ﬁrst-order derivatives need to be evolved in time and utilized in the
reconstruction, which is nontrivial in terms of practical implementations. After that,
Li et al. [14] introduce the multi-resolution HWENO schemes that only reconstruct
the function values and obtain the ﬁrst-order derivatives by the high-order linear
polynomials. Overall, the main drawback of the compact schemes is that a global
tridiagonal matrix needs to be solved at each time step, rendering them less eﬃcient.
Other variants include the central WENO (CWENO) schemes [15][16][17][18][19][20],
the WENO-AO [21] and WENO-ZQ [22] schemes, the WENO scheme with automatic
dissipation adjustment [23], and etc.
Diﬀerent from the WENO weighting strategy, Fu et al. [24][25][26][27][28] pro-
pose a family of TENO schemes for solving hyperbolic conservation laws. The TENO
schemes introduce a threshold parameter CT to assess whether the contribution of
one candidate stencil could be incorporated into the ﬁnal ﬂux computation. The
beneﬁt of this concept is that the TENO schemes can restore the background linear
3


schemes exactly in the smooth regions. Nevertheless, the standard TENO schemes
cannot deploy adaptive numerical dissipation in diﬀerent regions, i.e., low numerical
dissipation to resolve the high-wavenumber ﬂows and suﬃcient numerical dissipation
to capture discontinuities.
Later, Fu et al. [29][30] propose a series of TENO-A
schemes by adapting the threshold parameter CT based on a discontinuity sensor
proposed by Ren et al. [31]. However, the primary shortcoming of the disconti-
nuity sensor is its inability to evaluate the local ﬂow wavenumber accurately. De-
spite the fact that Su et al.
[32] and Sun et al.
[9] construct a six-point scale
sensor that can assess the local ﬂow wavenumber accurately, the sensor cannot be
applied directly to the ﬁve-point scheme due to the limited number of available sten-
cil points.
In addition to adapting CT for the diﬀerent ﬂow scales, by replacing
the polynomial reconstruction with a non-polynomial jump-like THINC reconstruc-
tion [33][34], the TENO5-THINC [35] scheme deploys the standard TENO scheme
in the smooth regions and the non-polynomial THINC reconstruction for resolving
discontinuities based on a novel discontinuity-detection criterion. The performance
of the TENO-family schemes has been extensively demonstrated for the compressible
gas dynamics [36][37][38][39][40][41][42], the multiphase ﬂows [43], the ideal magne-
tohydrodynamics (MHD) ﬂows [44], the turbulent ﬂows [45][46][47][48][49][50] and
the ﬂuid-structure-acoustics interactions [51], etc. For more details about TENO
schemes, please refer to [52].
In this paper, a new ﬁve-point TENO scheme with adaptive dissipation is pro-
posed by developing a novel ﬁve-point scale sensor. The main framework of the new
scheme is based on the TENO5-A scheme. The numerical dissipation-related param-
eter CT is determined by the local ﬂow wavenumber evaluated by the new ﬁve-point
scale sensor. Additionally, a hyperbolic tangent function is employed to map the es-
timated wavenumber to a limited interval due to the unboundedness of the evaluated
4


wavenumber. The new scheme achieves the adaptive dissipation control according to
the local ﬂow wavenumber, i.e., low dissipation is deployed in the low wavenumber
regions, comparatively high dissipation is delivered in the high wavenumber regions,
and adequate dissipation is generated for capturing discontinuities. To demonstrate
the performance of the new scheme, a set of 1D and 2D challenging benchmark cases
with broadband ﬂow length scales and shockwaves is simulated.
The remainder of the paper is organized as follows. (i) In section 2, the basic con-
cept of the standard TENO5 scheme for scalar conservation law is brieﬂy reviewed;
(ii) In section 3, the new ﬁve-point scale sensor and the new TENO scheme are pro-
posed in detail; (iii) In section 4, the performance of the new scheme is demonstrated
by simulating a set of benchmark cases; (iv) Concluding remarks are given in the
last section.
2. Brief review of the TENO scheme
In the following sections, we consider the one-dimensional scalar hyperbolic con-
servation law
∂u
∂t + ∂f(u)
∂x
= 0,
(1)
where u denotes the solution and f is the ﬂux function. For hyperbolic conservation
laws, ∂f(u)
∂u
denotes the characteristic signal speed. Without loss of generality, the
characteristic speed is assumed to be ∂f(u)
∂u
> 0 in the following analysis. Then, a
system of ordinary diﬀerential equations is formed by discretizing Eq. (1),
dui
dt = −∂f
∂x

x=xi
, i = 0, · · · , N.
(2)
∂f
∂x

x=xi can be approximated by a conservative ﬁnite-diﬀerence scheme as
∂f
∂x

x=xi
=
1
∆x(hi+1/2 −hi−1/2),
(3)
5


where h(x) is an implicit function of f(x), and deﬁned as
f(x) =
1
∆x
Z x+∆x/2
x−∆x/2
h(ξ)dξ.
(4)
Furthermore, ∂f
∂x

x=xi can be numerically approximated following
dui
dt ≈−1
∆x

bfi+1/2 −bfi−1/2

,
(5)
where bfi±1/2 denotes the numerical ﬂux and can be approximated by a convex com-
bination of K −2 candidate-stencil ﬂuxes,
bfi+1/2 =
K−3
X
k=0
wk bfk,i+1/2.
(6)
The candidate stencil arrangement of the TENO scheme is shown in Fig. 1. The
TENO scheme employs a set of candidate stencils with incremental width and ensures
that each candidate stencil contains at least one upwind point. The stencil width rk
for the K-point scheme is summarized as
{rk} =

















{3, 3, 3, 4, . . ., K + 2
2

,
|
{z
}
0,...,K−3
if mod (K, 2) = 0,
{3, 3, 3, 4, . . ., K + 1
2
|
{z
}
0,...,K−3
},
if mod (K, 2) = 1.
(7)
A (rk −1)−degree polynomial can be constructed corresponding to the candidate
stencil Sk, as
h(x) ≈ˆfk(x) =
rk−1
X
l=0
al,kxl,
(8)
where the coeﬃcients al,k are determined by satisfying Eq. (4). The numerical ﬂux at
the cell interface i + 1
2 can be approximated by the polynomial ˆfk(x) corresponding
6


to each candidate stencil Sk. The smoothness measure of the k-th candidate stencil
is deﬁned as
γk = (C +
τK
βk,rk + ε)q, k = 0, . . . , K −3,
(9)
where ε = 10−40 to avoid the zero denominator.
Diﬀerent from the standard
WENO5-JS scheme, for a stronger separation between diﬀerent ﬂow scales, C = 1
and q = 6 are adopted. Following [3], βk,rk can be deﬁned as
βk,rk =
rk−1
X
j=1
∆x2j−1
Z xi+1/2
xi−1/2
 dj
dxj ˆfk(x)
2
dx.
(10)
Then, to implement the ENO-like stencil selection strategy, the smoothness measure
is normalized as
χk =
γk
PK−3
i=0 γi
.
(11)
Unlike the WENO schemes [3], TENO schemes either abandon the non-smooth sten-
cils completely or apply the smooth ones with the optimal linear weights for the ﬁnal
reconstruction. Speciﬁcally, a sharp cut-oﬀfunction is deﬁned as
δk =





0,
if χk < CT,
1,
otherwise,
(12)
where CT is a constant in the standard TENO schemes [25]. It is noted that, instead
of being a constant, CT can be adjusted dynamically to further control the numerical
dissipation, as shown in the TENO-A schemes [29][30]. At last, the ﬁnal nonlinear
weight of each candidate stencil can be computed by
wk =
dkδk
PK−3
i=0 diδi
, k = 0, . . . , K −3,
(13)
where dk represents the optimal weight of candidate stencil Sk. For obtaining the
ﬁnal K−th order scheme in the smooth regions, the values of dk are shown in Table 1.
7


Table 1: Optimal weight dk of each candidate stencil for achieving the global K−th order scheme.
Order
d0
d1
d2
d3
d4
d5
K = 3
1
K = 4
3
6
3
6
K = 5
6
10
3
10
1
10
K = 6
9
20
6
20
1
20
4
20
K = 7
18
35
9
35
3
35
4
35
1
35
K = 8
30
70
18
70
4
70
12
70
1
70
5
70
Then, the ﬁnal high-order reconstruction for the numerical ﬂux at the cell interface
i + 1
2 is assembled as
ˆf K
i+1/2 =
K−3
X
k=0
wk ˆfk,i+1/2.
(14)
Especially, for the standard TENO5 scheme, the candidate stencils involve {S0, S1, S2}.
With some algebraic derivations, the candidate numerical ﬂuxes at the cell interface
i + 1
2 can be explicitly given by
ˆf0,i+1/2 = 1
6 (−fi−1 + 5fi + 2fi+1) ,
ˆf1,i+1/2 = 1
6 (2fi + 5fi+1 −fi+2) ,
ˆf2,i+1/2 = 1
6 (2fi−2 −7fi−1 + 11fi) .
(15)
Then, according to Eq. (10), the explicit formulas for the smoothness indicators of
the three candidate stencils are given as
β0 = 1
4 (fi−1 −fi+1)2 + 13
12 (fi−1 −2fi + fi+1)2 ,
β1 = 1
4 (3fi −4fi+1 + fi+2)2 + 13
12 (fi −2fi+1 + fi+2)2 ,
β2 = 1
4 (fi−2 −4fi−1 + 3fi)2 + 13
12 (fi−2 −2fi−1 + fi)2 .
(16)
8


Following [5][25], the global smoothness measure of the ﬁfth-order TENO5 scheme
is deﬁned as τ5 = |β1 −β2|, and the cut-oﬀparameter CT is set as 10−5 by spectral
analysis.
For the temporal integration of the resulting ODE Eq. (2), the third-order strong-
stability-preserving (SSP) Runge–Kutta scheme is utilized, which can be written as
u(1) = un + ∆tL (un) ,
u(2) = 3
4un + 1
4u(1) + 1
4∆tL
 u(1)
,
un+1 = 1
3un + 2
3u(2) + 2
3∆tL
 u(2)
.
(17)
xi-3
xi-2
xi-1
xi
xi+1
xi+2
xi+3
xi+4
xi+1/2
Figure 1: Candidate stencils of the high-order TENO scheme for reconstructing the cell interface
ﬂux at i + 1
2 [25]. The characteristic speed is assumed to be ∂f(u)
∂u
> 0. The stencil arrangement
and the corresponding candidate stencil schemes for the scenario with ∂f(u)
∂u
< 0 can be obtained
by symmetry at i + 1
2.
9


3. The new ﬁve-point scale sensor and new TENO5-A scheme
This section is divided into two parts.
We will ﬁrst review the discontinuity
sensor utilized in the standard TENO5-A scheme and propose a new ﬁve-point scale
sensor. Then, a new TENO5-A scheme will be proposed by incorporating this new
scale sensor.
3.1. The new ﬁve-point scale sensor
The discontinuity sensor m deployed in the standard TENO5-A scheme is pro-
posed by Ren et al. [31] as
m = 1 −min(1, ηi+1/2
Cr
),
(18)
where
ηi =
|2Dfi+1/2Dfi−1/2| + ε
(Dfi+1/2)2 + (Dfi−1/2)2 + ε,
ε =
0.9Cr
1 −0.9Cr
ξ2,
Dfi+1/2 = fi+1 −fi,
(19)
and the corresponding parameters are ξ = 10−3, Cr = 0.24, and ηi+1/2 = min(ηi−1, ηi, ηi+1).
This sensor can roughly locate the discontinuities, but is incapable of estimating the
speciﬁc wavenumber. In the standard TENO5-A scheme, CT is adjusted dynami-
cally to tailor the nonlinear numerical dissipation according to the smoothness of
local ﬂow scales as









g(m) = (1 −m)4(1 + 4m),
β = α1 −α2(1 −g(m)),
CT = 10−⌊¯β⌋,
(20)
where ⌊⌋is the Gauss bracket, g(m) is a smoothing-kernel based mapping function,
the parameter α1 = 10.0 and α2 = 5.0. When m ≈1, g(m) ≈0 and CT ≈10α2−α1,
which is typical for robust shock-capturing with strong nonlinear adaptation. When
10


m ≈0, g(m) ≈1 and CT decreases to 10−α1, which is suitable for resolving the high-
wavenumber physical ﬂuctuations. With a proper choice of parameter α1 and α2,
the TENO5-A scheme performs signiﬁcantly better than the counterpart standard
TENO5 scheme in terms of resolving the small-scale ﬂow structures [29][36].
In this work, a novel ﬁve-point scale sensor will be proposed for evaluating the
local ﬂow wavenumber accurately to further enhance the performance of the TENO5-
A scheme.
According to [10], a six-point scale sensor is proposed based on the
Taylor-series expansion, i.e.,
f(x) −f(x0) =
∞
X
p=1
1
p!f (p)(x0)∆xp,
(21)
and for each order derivative, the variation of the solution is deﬁned as
∆fp = f (p)(x0)∆xp.
(22)
Since the lower order derivatives play more important roles in the Taylor-series ex-
pansion than the higher order ones for smooth ﬂow scales, it is reasonable to express
the local scale of the solution by the ratio of ∆fp at diﬀerent orders as
KESW =
s
|∆f3|
|∆f1|
or
KESW =
s
|∆f4|
|∆f2|.
(23)
The eﬀectiveness of this scale sensor for estimating the scaled wavenumber can be
analyzed as follows. Considering a pure sine function
f(x) = A sin(ωx + ϕ),
(24)
11


the derivatives of this function are given by
f (1)(x) = Aω cos(ωx + ϕ),
f (2)(x) = −Aω2 sin(ωx + ϕ),
f (3)(x) = −Aω3 cos(ωx + ϕ),
f (4)(x) = Aω4 sin(ωx + ϕ),
(25)
and the theoretical scaled wavenumber KSW is computed by
KSW = ω∆x.
(26)
Then, it can be straightforwardly deduced that
KSW =
s
|f (3)|
|f (1)|∆x
or
KSW =
s
|f (4)|
|f (2)|∆x,
(27)
which is equal to
KSW =
s
|∆f3|
|∆f1|
or
KSW =
s
|∆f4|
|∆f2|.
(28)
Note that, when the two formulas are deployed separately, singular values may appear
near the critical points and the inﬂection points. To deal with this problem, the
two formulas in Eq. (23) are combined to achieve better performance for practical
simulations as
KESW =
s
|∆f3| + |∆f4|
|∆f1| + |∆f2| + ε1
,
(29)
where ε1 = 10−12 denotes a small number for avoiding the zero denominator. Addi-
tionally, the ﬁnal numerical results are not sensitive to the choice of ε1 as long as it
is a small value.
Diﬀerent from the six-point sensor developed in [10], due to insuﬃcient stencil
points in the ﬁve-point TENO5-A scheme, we propose to estimate each order deriva-
tive at xi instead of xi+1/2. In practice, evaluating the wavenumber at xi or xi+1/2 is
12


almost identical. Finally, the ﬁve-point scale sensor can be written as
KESW =
s
|∆f3,i| + |∆f4,i|
|∆f1,i| + |∆f2,i| + ε1
,
(30)
where
∆f1,i = 1
12fi−2 −2
3fi−1 + 2
3fi+1 −1
12fi+2,
∆f2,i = −1
12fi−2 + 4
3fi−1 −5
2fi + 4
3fi+1 −1
12fi+2,
∆f3,i = −1
2fi−2 + fi−1 −fi+1 + 1
2fi+2,
∆f4,i = fi−2 −4fi−1 + 6fi −4fi+1 + fi+2.
(31)
To demonstrate the performance of the newly proposed ﬁve-point scale sensor, a
set of functions is considered to compare with the classical sensor in the standard
TENO5-A scheme, i.e.,
(a)
f = sin(20πx),
−1 ≤x ≤1;
(b)
f =





sin(12πx) −2,
−1 ≤x < 0,
sin (24.5πx) + 2,
0 ≤x ≤1;
(c)
f =





0,
−1 ≤x < 0,
ex−1 sin(32πx),
0 ≤x ≤1;
(d)
f = sin(2πex+1x),
−1 ≤x ≤1.
The computational results are shown in Fig. 2. It can be seen that the wavenum-
ber of critical points computed by Ren’s method has distinct values from other points
in the regions with the same theoretical wavenumber. Additionally, in case (b), the
13


wavenumber calculated by Ren’s method at the discontinuity is smaller than that at
some critical points. In contrast, compared to the exact wavenumber distribution,
the newly proposed ﬁve-point scale sensor can estimate the local ﬂow wavenumber
including that at critical points accurately. Moreover, the estimated wavenumber at
discontinuities diﬀers signiﬁcantly from the other regions, which allows for deploying
larger dissipation for sharp shock-capturing.
-1
-0.5
0
0.5
1
-2
0
2
-2
0
2
-1
-0.5
0
0.5
1
-2
0
2
-5
0
5
-1
-0.5
0
0.5
1
-1
-0.5
0
0.5
1
-4
-2
0
2
4
-1
-0.5
0
0.5
1
-1
0
1
-2
0
2
Figure 2: Functional value distributions of the Ren’s sensor used in the standard TENO5-A scheme
[31] and the newly proposed ﬁve-point scale sensor in this work.
3.2. The new ﬁve-point TENO-A scheme
As mentioned above, the standard TENO5-A scheme determines the threshold
parameter CT based on the discontinuity sensor m in Eq. (18). Due to the unbound-
edness of the new scale sensor, an extra function, similar to g(m) in TENO5-A, is
14


needed to deploy the new scale sensor in the framework of TENO5-A. A hyperbolic
tangent function is chosen due to its rigorous boundedness and good smoothness.
The details of the new TENO5-A scheme are as follows.
Diﬀerent from the standard TENO5 scheme [24], the sixth-order global smooth-
ness measure τ5 in the weighting strategy is taken as [29][27]
τ5 =
1
5040
5788f 2
i−2 + fi−2 (−45681fi−1 + 64843fi −38947fi+1 + 8209fi+2) + fi−1 (93483fi−1
−275836fi + 173498fi+1 −38947fi+2) + fi (210993fi −275836fi+1 + 64843fi+2)
+fi+1 (93483fi+1 −45681fi+2) + 5788f 2
i+2
 .
(32)
For adaptive numerical dissipation, CT is determined by the newly proposed ﬁve-
point scale sensor,









g(KESW) = tanh(1.01KESW),
¯β = α1 −α2(g(KESW)),
CT = 10−⌊¯β⌋,
(33)
where KESW is computed by Eq. (30), the parameter α1 = 10 and α2 = 5. Here,
the function tanh(·) is introduced for achieving the boundedness between 0 and 1.
Then, the cut-oﬀfunction is similarly deﬁned as
δk =





0,
if χk < CT,
1,
otherwise.
(34)
At last, the ﬁnal weight of each candidate stencil is given by
wk =
dkδk
P2
i=0 diδi
, k = 0, 1, 2,
(35)
where the optimal linear weights are optimized by spectral analysis, and given as d0 =
0.5065006634, d1 = 0.3699651429, and d2 = 0.1235341937 [29]. The left formulas are
15


the same as the standard TENO5-A scheme and are not shown here for brevity.
Additionally, the dispersion and dissipation property of the present scheme can be
analysed by the approximated dispersion relation (ADR) analysis [53][54]. As shown
in Fig. 3, both the dispersion and dissipation properties of the present scheme are
better than TENO5. And the optimal background linear scheme can be restored
exactly up to an intermediate wavenumber.
0
0.5
1
1.5
2
2.5
3
0
0.5
1
1.5
2
2.5
3
Spectral
Upwind linear 5th order
Optimized linear 3rd order
TENO5
Present
0
0.5
1
1.5
2
2.5
3
-1.2
-1
-0.8
-0.6
-0.4
-0.2
0
Spectral
Upwind linear 5th order
Optimized linear 3rd order
TENO5
Present
Figure 3: Dispersion (left) and dissipation (right) properties of the upwind 5th-order linear scheme,
the optimized linear 3rd-order scheme, TENO5 scheme, and the present scheme, where the op-
timized linear 3rd-order scheme is from the combination of candidate stencils with the chosen
optimized parameters d0 = 0.5065006634, d1 = 0.3699651429, and d2 = 0.1235341937 [29].
4. Numerical validation
In this section, a set of benchmark cases is simulated by WENO5-Z, TENO5,
TENO5-A, and the present ﬁve-point scheme to access the performance. The ideal
gas equation p = (γ −1)ρe with γ = 1.4 is employed to close the Euler equations.
16


Unless otherwise speciﬁed, the CFL number is set as 0.4, the Rusanov scheme [55]
is used for ﬂux splitting, and the Roe average is utilized for characteristic decompo-
sition. In terms of the time integration, the third-order strong-stability-preserving
(SSP) Runge–Kutta [56] scheme is chosen as default.
All the grids used in this
section are Cartesian and uniform.
4.1. Accuracy test
Considering the linear advection problem with a smooth initial condition to verify
the accuracy order of the proposed scheme in the smooth regions, the governing
equation and the initial condition are given as
∂u
∂t + ∂u
∂x = 0, u(x, 0) = sin(πx), 0 ≤x ≤2.
(36)
The grid resolution is chosen as N = 20, 40, 80, 160, 320, and 640, respectively. ∆t is
set as ∆x
p
3, where p is the theoretical accuracy order.
The statistics of the L1 and L∞errors and the corresponding accuracy order
are shown in Table 2. As expected, TENO5-A and the present scheme show third-
order convergence. This case veriﬁes that both TENO5-A and the newly proposed
scheme can restore the optimal background linear scheme in the smooth regions,
and the accuracy order of the optimal background linear scheme is third with the
given optimal linear weights in the TENO5-A scheme [29]. Since the TENO schemes
either abandon the non-smooth stencils completely or apply the smooth ones with
the optimal linear weights for the ﬁnal reconstruction, meanwhile, this case only
contains smooth functions, TENO5-A and the newly proposed scheme both restore
the optimal background linear scheme exactly. Therefore, TENO5-A and the newly
proposed scheme have the same performance in this case.
17


Table 2: The error statistics and the corresponding accuracy orders of TENO5-A and the present
scheme.
Scheme
N
L1 error
L1 order
L∞error
L∞order
TENO5-A
20
2.03E-03
-
3.28E-03
-
40
2.86E-04
2.83
4.58E-04
2.84
80
3.71E-05
2.95
5.89E-05
2.96
160
4.69E-06
2.98
7.41E-06
2.99
320
5.89E-07
2.99
9.27E-07
3.00
640
7.37E-08
3.00
1.16E-07
3.00
Present
20
2.03E-03
-
3.28E-03
-
40
2.86E-04
2.83
4.58E-04
2.84
80
3.71E-05
2.95
5.89E-05
2.96
160
4.69E-06
2.98
7.41E-06
2.99
320
5.89E-07
2.99
9.27E-07
3.00
640
7.37E-08
3.00
1.16E-07
3.00
4.2. Linear advection of multiple waves
This case is taken from [3] and we solve the linear advection equation
∂u
∂t + ∂u
∂x = 0,
(37)
with the initial condition given as
u(x, 0) =





















1
6[G(x −1, β, z −θ) + G(x −1, β, z + θ) + 4G(x −1, β, z)],
if 0.2 ≤x < 0.4,
1,
if 0.6 ≤x ≤0.8,
1 −|10(x −1.1)|,
if 1.0 ≤x ≤1.2,
1
6[F(x −1, α, a −θ) + F(x −1, α, a + θ) + 4F(x −1, α, a)],
if 1.4 ≤x < 1.6,
0,
otherwise,
(38)
where
G(x, β, z) = e−β(x−z)2, F(x, α, a) =
p
max (1 −α2(x −a)2, 0).
(39)
18


The parameters in Eq. (38) and Eq. (39) are
a = 0.5, z = −0.7, θ = 0.005, α = 10, β = log 2
36θ2 .
(40)
The initial condition consists of a Gaussian pulse, a square wave, a sharp triangle
wave, and a half ellipse arranged from the left to the right in the computational
domain x ∈[0, 2]. The equation is solved by a uniform grid with N = 200, and the
ﬁnal evolution time is t = 2 and 18, respectively. The exact solution is the theoretical
solution of the linear advection equation with a constant speed of propagation.
As shown in Fig. 4, with the short time evolution, the performance of TENO5-A
and the present scheme does not have obvious diﬀerences. However, for the Gaussian
pulse and the sharp triangle wave, both TENO5-A and the present scheme capture
them more sharply than WENO5-Z and TENO5. In terms of the long-time evolution
where the numerical error accumulates substantially, as shown in Fig. 5, the result of
TENO5-A has obvious oscillations even in the smooth regions. The present scheme
exhibits notable advantages in preserving the overall shape and capturing the square
wave sharply.
4.3. Shock-tube problem
From section 4.3 to section 4.5, the one-dimensional Euler equations are solved,
which can be written as





ρ
ρu
E





t
+





ρu
ρu2 + p
u(E + p)





x
= 0,
(41)
where ρ denotes the density, u is the velocity, E is the total energy, and p is the
pressure.
19


x
u
0
0.5
1
1.5
2
0
0.2
0.4
0.6
0.8
1
1.2
1.4
Exact
WENO5-Z
TENO5
TENO5-A
Present
Figure 4: Linear advection of multiple waves: u distributions from WENO5-Z, TENO5, TENO5-A,
and the present scheme at the simulation time t = 2. The spatial discretization is on 200 uniform
grid points. “Exact” denotes the theoretical solution of the linear advection equation with the
constant speed of propagation.
In this section, two typical shock-tube problems are solved to validate the shock-
capturing capability of the proposed scheme.
The initial condition for the Sod’s problem [57] is
(ρ, u, p) =





(1, 0, 1),
if 0 ≤x < 0.5,
(0.125, 0, 0.1),
if 0.5 ≤x ≤1,
(42)
and the ﬁnal simulation time is set as t = 0.2.
20


x
u
0
0.5
1
1.5
2
0
0.2
0.4
0.6
0.8
1
1.2
1.4
Exact
TENO5-A
Present
Figure 5: Linear advection of multiple waves: u distributions from TENO5-A and the present
scheme at the simulation time t = 18. The spatial discretization is on 200 uniform grid points.
“Exact” denotes the theoretical solution of the linear advection equation with the constant speed
of propagation. The numerical error accumulates substantially after a long time evolution.
The initial condition for the Lax’s problem [58] is
(ρ, u, p) =





(0.445, 0.698, 3.528),
if 0 ≤x < 0.5,
(0.5, 0, 0.5710),
if 0.5 ≤x ≤1,
(43)
and the ﬁnal simulation time is set as t = 0.14.
These two cases are solved by WENO5-Z, TENO5, TENO5-A, and the present
scheme with the resolution of N = 100.
The results are shown in Fig. 6.
The
“Exact” reference is the theoretical solution of the corresponding Riemann problem.
21


The results show that all these schemes considered can capture the discontinuities
without artiﬁcial oscillations.
x
Density
0
0.2
0.4
0.6
0.8
1
0
0.2
0.4
0.6
0.8
1
Exact
WENO5-Z
TENO5
TENO5-A
Present
x
Density
0
0.2
0.4
0.6
0.8
1
0.4
0.6
0.8
1
1.2
1.4
Exact
WENO5-Z
TENO5
TENO5-A
Present
Figure 6: Shock-tube problem: the computed density distributions of the Sod’s problem (left) and
the computed density distributions of the Lax’s problem (right). Discretization is both on 100
uniform grid points. “Exact” both denotes the theoretical solution of the corresponding Riemann
problem.
4.4. Interacting blast waves
The two-blast-wave interaction taken from Woodward and Colella [59] is consid-
ered. The initial condition is
(ρ, u, p) =













(1, 0, 1000),
if 0 ≤x < 0.1,
(1, 0, 0.01),
if 0.1 ≤x < 0.9,
(1, 0, 100),
if 0.9 ≤x ≤1.
(44)
The reﬂective boundary condition is used at x = 0 and x = 1. Meanwhile, it is
solved by a uniform grid with N = 400 and the ﬁnal evolution time t = 0.038.
The Roe scheme with entropy-ﬁx is utilized for the numerical ﬂux splitting, and the
22


CFL number is set as 0.35 for good robustness. The exact solution is solved by the
WENO5-JS scheme with N = 2000.
As shown in Fig. 7, near the density peak at x = 0.78, TENO5-A and the present
scheme perform better than WENO5-Z and TENO5.
x
Density
0
0.2
0.4
0.6
0.8
1
0
2
4
6
Exact
WENO5-Z
TENO5
TENO5-A
Present
x
Density
0.65
0.7
0.75
0.8
3
3.5
4
4.5
5
5.5
6
6.5
Exact
WENO5-Z
TENO5
TENO5-A
Present
Figure 7: Interacting blast waves: the computed density distributions from various schemes (left)
and the zoomed-in view (right). Discretization is on 400 uniform grid points. “Exact” denotes the
solution solved by WENO5-JS with 2000 grid points.
4.5. Shock–density wave interaction
This case is taken from Shu and Osher [60]. The initial condition is
(ρ, u, p) =





(3.857, 2.629, 10.333),
if 0 ≤x < 1,
(1 + 0.2 sin(5(x −5)), 0, 1),
if 1 ≤x < 10.
(45)
This case is designed by simulating a Mach 3 shock interacting with a sine wave.
The computed density ρ is plotted at t = 1.8 with N = 200, and the exact solution
is solved by the WENO5-JS scheme with N = 2000.
23


As shown in Fig. 8, all the considered schemes are capable of capturing the
acoustic waves.
In terms of the density distribution, TENO5-A and the present
scheme show a higher resolution than WENO5-Z and TENO5 in resolving the high-
wavenumber physical ﬂuctuations. The standard TENO5-A scheme also generates
slight overshoots around x = 3.25.
For the computed velocity distribution, the
present scheme features the best resolution in maintaining the amplitude of the
velocity proﬁle from x = 4.3 to 5.6. Both the zoomed-in views show that the present
scheme has lower numerical dissipation than the other schemes.
4.6. Double Mach reﬂection of a strong shock
In this section, the two-dimensional Euler equations are solved, which can be
written as








ρ
ρu
ρv
E








t
+








ρu
ρu2 + p
ρuv
u(E + p)








x
+








ρv
ρuv
ρv2 + p
v(E + p)








y
= 0,
(46)
where u and v denote the velocity along x- and y-direction, respectively.
The initial condition is given as
(ρ, u, v, p) =





(1.4, 0, 0, 1),
if y < 1.732(x −0.1667),
(8, 7.145, −4.125, 116.8333),
otherwise.
(47)
The initial condition describes a Mach 10 shockwave moving from left to right with
an incidence angle of 60◦. As for the boundary condition, the inﬂow and outﬂow
boundary conditions are implemented for the left and right sides of the computational
domain, respectively. For the top side, the boundary condition follows the exact
solution of a Mach 10 moving shockwave. In terms of the bottom side, the boundary
24


x
Density
0
2
4
6
8
10
0
0.5
1
1.5
2
2.5
3
3.5
4
4.5
Exact
WENO5-Z
TENO5
TENO5-A
Present
x
Density
4
4.5
5
5.5
6
6.5
7
7.5
8
3
3.5
4
4.5
5
Exact
WENO5-Z
TENO5
TENO5-A
Present
x
Velocity
4.5
5
5.5
2.4
2.45
2.5
2.55
2.6
2.65
2.7
2.75
2.8
Exact
WENO5-Z
TENO5
TENO5-A
Present
x
Velocity
0
2
4
6
8
10
0
0.5
1
1.5
2
2.5
3
Exact
WENO5-Z
TENO5
TENO5-A
Present
Figure 8: Shock–density wave interaction: solutions from various schemes. Top: the computed
density distributions (left) and the zoomed-in view (right). Bottom: the computed velocity distri-
butions (left) and the zoomed-in view (right). Discretization is on 200 uniform grid points. “Exact”
denotes the solution solved by WENO5-JS with 2000 grid points.
25


condition in the region from x = 0 to x = 0.1667 follows the post-shock condition,
whereas that in the remaining region from x = 0.1667 to x = 4.0 follows the reﬂective
wall condition. The ﬁnal evolution time is t = 0.2, and the grid resolution is 1024 ×
256.
This result of this case is very sensitive to the dissipation and dispersion properties
of the deployed numerical scheme. Fig. 9 plots the density contours from TENO5-
A and the present scheme. Overall speaking, the present scheme generates much
less numerical noise than TENO5-A behind the incident moving shockwave. Fig. 10
only presents the region of [2, 3] × [0, 0.9] for the ease of comparison. Each scheme
exhibits a signiﬁcantly diﬀerent performance. The present scheme resolves the richest
small-scale structures, indicating the lowest built-in numerical dissipation. Besides,
TENO5-A generates too much numerical noise and fewer small-scale structures than
the new scheme. Unlike the above two schemes, both WENO5-Z and TENO5 produce
excessive numerical dissipation, but TENO5 still performs better than WENO5-Z.
In order to demonstrate the computational eﬃciency of the present scheme, ta-
ble 3 shows the statistics of computational costs with WENO5-Z, TENO5, TENO5-
A, and the present scheme. It can be seen that the computational eﬀort is approxi-
mately the same for TENO5-A and the present scheme.
WENO5-Z
TENO5
TENO5-A
Present
512 × 128 [s]
54.99
55.10
78.47
80.41
Table 3: The computational time statistics with diﬀerent numerical schemes.
26


x
y
0
0.2
0.4
0.6
0.8
1
1.2
1.4
1.6
1.8
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
Present
x
y
0
0.2
0.4
0.6
0.8
1
1.2
1.4
1.6
1.8
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
TENO5-A
Figure 9: Double Mach reﬂection of a strong shock: density contours computed from TENO5-A
and the present scheme. Both plots are drawn with 43 contourlines between 1.887 and 20.9. The
resolution is 1024 × 256.
4.7. Rayleigh-Taylor instability
The inviscid Rayleigh-Taylor instability case proposed by Xu and Shu [61] is
considered, where the two-dimensional Euler equations with gravity are solved, i.e.,








ρ
ρu
ρv
E








t
+








ρu
ρu2 + p
ρuv
u(E + p)








x
+








ρv
ρuv
ρv2 + p
v(E + p)








y
=








0
0
ρg
ρvg








,
(48)
here, g = 1 denotes the gravity.
27


x
y
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
Present
x
y
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
TENO5-A
x
y
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
WENO5-Z
x
y
2
2.2
2.4
2.6
2.8
3
0
0.2
0.4
0.6
0.8
TENO5
Figure 10: Same as Fig. 9, but the zoomed-in view from various schemes.
28


and the initial condition is
(ρ, u, v, p) =





(2, 0, −0.025c cos(8πx), 1 + 2y),
if 0 ≤y < 0.5,
(1, 0, −0.025c cos(8πx), y + 1.5),
if 0.5 ≤y ≤1,
(49)
where c =
q
γp
ρ is the sound speed with γ = 5
3, and the computational domain is
deﬁned as [0, 0.25] × [0, 1]. For the left and right sides of the computational domain,
the reﬂective boundary condition is enforced. In terms of the bottom and top sides,
constant primitive variables (ρ, u, v, p) = (2, 0, 0, 1) and (ρ, u, v, p) = (1, 0, 0, 2.5) are
imposed, respectively. The ﬁnal evolution time is t = 1.95 and the gird resolution is
96 × 384. Especially, the Roe scheme is used for ﬂux splitting.
As shown in Fig. 11, the present scheme resolves much richer vortical structures
than the other schemes, indicating its low numerical dissipation. It is noted that
the solutions from TENO5, TENO5-A, and the present scheme are not symmetric,
which is mainly due to the fact that the low numerical dissipation cannot suppress
the numerical disturbances from the machine round-oﬀerrors [62][63].
5. Conclusions
In this paper, a ﬁve-point TENO scheme with adaptive dissipation based on a
new scale sensor is proposed for solving hyperbolic conservation laws. Compared
to the discontinuity sensor in the standard TENO5-A scheme, the new scale sen-
sor is capable of evaluating the local ﬂow wavenumber. Beneﬁting from that, the
proposed scheme adapts its dissipation according to the local ﬂow wavenumber. A
set of benchmark cases is simulated, and the performance of the proposed scheme is
summarized as follows.
The proposed scheme can restore the optimal background linear scheme in the
smooth regions without degeneration.
The proposed scheme exhibits overall less
29


WENO5-Z
TENO5
TENO5-A
Present
Figure 11: Rayleigh-Taylor instability: density contours computed from various schemes. All plots
are drawn with 43 contourlines between 0.9 and 2.2. The resolution is 96 × 384.
30


numerical dissipation than the standard TENO5-A scheme while preserving the dis-
continuity more sharply. In the case of “double Mach reﬂection of a strong shock”,
the proposed scheme resolves more small-scale structures in the blow-up region and
generates signiﬁcantly less numerical noise than TENO5-A; In the case of the long-
time evolution of the multiple waves, the advantage of the proposed scheme is obvious
in terms of suppressing the artiﬁcial numerical oscillations.
The proposed scheme is easy to be implemented into an existing code since the
same stencil points are utilized as the classical WENO5-JS scheme. And it has much
lower dissipation than the existing ﬁve-point shock-capturing schemes. The present
idea can also be extended for the very high-order TENO schemes with adaptive
dissipation control. Our future work will also deploy the proposed schemes in more
complicated simulations, including the MHD and multiphase ﬂows.
It is further
noted that we do not see obvious barriers to extend the present scheme to turbulent
ﬂow simulations, and the relevant work will be reported in a separate forthcoming
paper.
Declaration of Competing Interest
The authors declare that they have no known competing ﬁnancial interests or
personal relationships that could have appeared to inﬂuence the work reported in
this paper.
Data availability
The data that support the ﬁndings of this study are available on request from
the corresponding author, LF.
31


Acknowledgements
L.F. acknowledges the fund from National Key R&D Program of China (No.
2022YFA1004500), the fund from Research Grants Council (RGC) of the Government
of Hong Kong Special Administrative Region (HKSAR) with RGC/ECS Project (No.
26200222), the fund from Guangdong Basic and Applied Basic Research Founda-
tion (No. 2022A1515011779), the fund from the Project of Hetao Shenzhen-Hong
Kong Science and Technology Innovation Cooperation Zone (No. HZQB-KCZYB-
2020083), and the fund from Key Laboratory of Computational Aerodynamics, AVIC
Aerodynamics Research Institute.
References
[1] A. Harten, B. Engquist, S. Osher, S. R. Chakravarthy, Uniformly high order
accurate essentially non-oscillatory schemes, III, in: Upwind and high-resolution
schemes, Springer, 1987, pp. 218–290.
[2] X.-D. Liu, S. Osher, T. Chan, Weighted essentially non-oscillatory schemes,
Journal of computational physics 115 (1) (1994) 200–212.
[3] G.-S. Jiang, C.-W. Shu, Eﬃcient implementation of weighted ENO schemes,
Journal of computational physics 126 (1) (1996) 202–228.
[4] A. K. Henrick, T. D. Aslam, J. M. Powers, Mapped weighted essentially non-
oscillatory schemes: achieving optimal order near critical points, Journal of
Computational Physics 207 (2) (2005) 542–567.
[5] R. Borges, M. Carmona, B. Costa, W. S. Don, An improved weighted essentially
non-oscillatory scheme for hyperbolic conservation laws, Journal of Computa-
tional Physics 227 (6) (2008) 3191–3211.
32


[6] F. Acker, R. d. R. Borges, B. Costa, An improved WENO-Z scheme, Journal of
Computational Physics 313 (2016) 726–753.
[7] Z.-S. Sun, Y.-X. Ren, C. Larricq, S.-y. Zhang, Y.-c. Yang, A class of ﬁnite
diﬀerence schemes with low dispersion and controllable dissipation for DNS
of compressible turbulence, Journal of computational physics 230 (12) (2011)
4616–4635.
[8] Z.-s. Sun, L. Luo, Y.-x. Ren, S.-y. Zhang, A sixth order hybrid ﬁnite diﬀerence
scheme based on the minimized dispersion and controllable dissipation tech-
nique, Journal of Computational Physics 270 (2014) 238–254.
[9] Z. Sun, Y. Hu, Y. Ren, K. Mao, An Optimal Finite Diﬀerence Scheme with Mini-
mized Dispersion and Adaptive Dissipation Considering the Spectral Properties
of the Fully Discrete Scheme, Journal of Scientiﬁc Computing 89 (2) (2021)
1–32.
[10] Y. Li, C. Chen, Y.-X. Ren, A class of high-order ﬁnite diﬀerence schemes with
minimized dispersion and adaptive dissipation for solving compressible ﬂows,
Journal of Computational Physics 448 (2022) 110770.
[11] X. Deng, H. Zhang, Developing high-order weighted compact nonlinear schemes,
Journal of Computational Physics 165 (1) (2000) 22–44.
[12] J. Qiu, C.-W. Shu, Hermite WENO schemes and their application as limiters for
Runge–Kutta discontinuous Galerkin method: one-dimensional case, Journal of
Computational Physics 193 (1) (2004) 115–135.
[13] X. Cai, X. Zhang, J. Qiu, Positivity-preserving high order ﬁnite volume HWENO
33


schemes for compressible Euler equations, Journal of Scientiﬁc Computing 68 (2)
(2016) 464–483.
[14] J. Li, C.-W. Shu, J. Qiu, Multi-resolution HWENO schemes for hyperbolic
conservation laws, Journal of Computational Physics 446 (2021) 110653.
[15] D. Levy, G. Puppo, G. Russo, Central WENO schemes for hyperbolic systems
of conservation laws, ESAIM: Mathematical Modelling and Numerical Analysis-
Mod´elisation Math´ematique et Analyse Num´erique 33 (3) (1999) 547–571.
[16] D. Levy, G. Puppo, G. Russo, Compact central WENO schemes for multidimen-
sional conservation laws, SIAM Journal on Scientiﬁc Computing 22 (2) (2000)
656–672.
[17] P. Tsoutsanis, M. Dumbser, Arbitrary high order central non-oscillatory schemes
on mixed-element unstructured meshes, Computers & Fluids 225 (2021) 104961.
[18] P. Tsoutsanis, M. S. S. P. Kumar, P. S. Farmakis, A relaxed a posteriori MOOD
algorithm for multicomponent compressible ﬂows using high-order ﬁnite-volume
methods on unstructured meshes, Applied Mathematics and Computation 437
(2023) 127544.
[19] P. Tsoutsanis, E. M. Adebayo, A. C. Merino, A. P. Arjona, M. Skote, CWENO
ﬁnite-volume interface capturing schemes for multicomponent ﬂows using un-
structured meshes, Journal of Scientiﬁc Computing 89 (2021) 1–27.
[20] V. Maltsev, D. Yuan, K. W. Jenkins, M. Skote, P. Tsoutsanis, Hybrid discontin-
uous Galerkin-ﬁnite volume techniques for compressible ﬂows on unstructured
meshes, Journal of Computational Physics 473 (2023) 111755.
34


[21] D. S. Balsara, S. Garain, C.-W. Shu, An eﬃcient class of WENO schemes with
adaptive order, Journal of Computational Physics 326 (2016) 780–804.
[22] J. Zhu, J. Qiu, A new ﬁfth order ﬁnite diﬀerence WENO scheme for solving
hyperbolic conservation laws, Journal of Computational Physics 318 (2016) 110–
121.
[23] J. Fern´andez-Fidalgo, L. Ram´ırez, P. Tsoutsanis, I. Colominas, X. Nogueira,
A reduced-dissipation WENO scheme with automatic dissipation adjustment,
Journal of Computational Physics 425 (2021) 109749.
[24] L. Fu, X. Y. Hu, N. A. Adams, A family of high-order targeted ENO schemes
for compressible-ﬂuid simulations, Journal of Computational Physics 305 (2016)
333–359.
[25] L. Fu, A hybrid method with TENO based discontinuity indicator for hyperbolic
conservation laws, Computer Physics Communications 26 (4) (2019) 973–1007.
[26] L. Fu, A very-high-order TENO scheme for all-speed gas dynamics and turbu-
lence, Computer Physics Communications 244 (2019) 117–131.
[27] L. Fu, X. Y. Hu, N. A. Adams, Targeted ENO schemes with tailored resolution
property for hyperbolic conservation laws, Journal of Computational Physics
349 (2017) 97–121.
[28] L. Fu, Very-high-order TENO schemes with adaptive accuracy order and adap-
tive dissipation control, Computer Methods in Applied Mechanics and Engi-
neering 387 (2021) 114193.
35


[29] L. Fu, X. Y. Hu, N. A. Adams, Improved ﬁve-and six-point targeted essentially
nonoscillatory schemes with adaptive dissipation, AIAA Journal 57 (3) (2019)
1143–1158.
[30] L. Fu, X. Hu, N. A. Adams, A targeted ENO scheme as implicit model for tur-
bulent and genuine subgrid scales, Communications in Computational Physics
26 (2) (2019) 311–345.
[31] Y.-X. Ren, H. Zhang, et al., A characteristic-wise hybrid compact-WENO
scheme for solving hyperbolic conservation laws, Journal of Computational
Physics 192 (2) (2003) 365–386.
[32] Y. Su, Y. Li, Y.-X. Ren, A sixth-order ﬁnite diﬀerence scheme with the mini-
mized dispersion and adaptive dissipation for solving compressible ﬂow, arXiv
preprint arXiv:2110.14482 (2021).
[33] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme
using hyperbolic tangent function, International journal for numerical methods
in ﬂuids 48 (9) (2005) 1023–1040.
[34] X. Deng, Z.-h. Jiang, P. Vincent, F. Xiao, C. Yan, A new paradigm of
dissipation-adjustable, multi-scale resolving schemes for compressible ﬂows,
Journal of Computational Physics (2022) 111287.
[35] S. Takagi, L. Fu, H. Wakimura, F. Xiao, A novel high-order low-dissipation
TENO-THINC scheme for hyperbolic conservation laws, Journal of Computa-
tional Physics 452 (2022) 110899.
[36] J. Peng, S. Liu, S. Li, K. Zhang, Y. Shen, An eﬃcient targeted ENO scheme with
36


local adaptive dissipation for compressible ﬂow simulation, Journal of Compu-
tational Physics 425 (2021) 109902.
[37] Y. Li, L. Fu, N. A. Adams, A low-dissipation shock-capturing framework with
ﬂexible nonlinear dissipation control, Journal of Computational Physics 428
(2021) 109960.
[38] K. Fardipour, K. Mansour, Development of targeted compact nonlinear scheme
with increasingly high order of accuracy, Progress in Computational Fluid Dy-
namics, an International Journal 20 (1) (2020) 1–19.
[39] R. Tan, A. Ooi, Two Dimensional Analysis and Optimization of Hybrid MDCD-
TENO Schemes, Journal of Scientiﬁc Computing 90 (1) (2022) 1–33.
[40] Z.-F. Meng, A.-M. Zhang, P.-P. Wang, F.-R. Ming, B. C. Khoo, A targeted
essentially non-oscillatory (TENO) SPH method and its applications in hydro-
dynamics, Ocean Engineering 243 (2022) 110100.
[41] T. Hiejima, A high-order weighted compact nonlinear scheme for compressible
ﬂows, Computers & Fluids 232 (2022) 105199.
[42] C.-C. Ye, P.-J.-Y. Zhang, Z.-H. Wan, D.-J. Sun, An alternative formulation of
targeted ENO scheme for hyperbolic conservation laws, Computers & Fluids
238 (2022) 105368.
[43] O. Haimovich, S. H. Frankel, Numerical simulations of compressible multicom-
ponent and multiphase ﬂow using a high-order targeted ENO (TENO) ﬁnite-
volume method, Computers & Fluids 146 (2017) 105–116.
[44] L. Fu, Q. Tang, High-order low-dissipation targeted ENO schemes for ideal
magnetohydrodynamics, Journal of Scientiﬁc Computing 80 (1) (2019) 692–716.
37


[45] A. Hamzehloo, D. J. Lusher, S. Laizet, N. D. Sandham, On the performance
of WENO/TENO schemes to resolve turbulence in DNS/LES of high-speed
compressible ﬂows, International Journal for Numerical Methods in Fluids 93 (1)
(2021) 176–196.
[46] D. J. Lusher, N. D. Sandham, Shock-wave/boundary-layer interactions in transi-
tional rectangular duct ﬂows, Flow, Turbulence and Combustion 105 (2) (2020)
649–670.
[47] E. Motheau, J. Wakeﬁeld, Investigation of ﬁnite-volume methods to capture
shocks and turbulence spectra in compressible ﬂows, Communications in Applied
Mathematics and Computational Science 15 (1) (2020) 1–36.
[48] D. J. Lusher, N. D. Sandham, Assessment of low-dissipative shock-capturing
schemes for the compressible Taylor–Green vortex, AIAA Journal 59 (2) (2021)
533–545.
[49] M. Di Renzo, J. Urzay, Direct numerical simulation of a hypersonic transitional
boundary layer at suborbital enthalpies, Journal of Fluid Mechanics 912 (2021).
[50] A. Gillespie, N. D. Sandham, Shock Train Response to High-Frequency Back-
pressure Forcing, AIAA Journal 60 (6) (2022) 3736–3748.
[51] L. Wang, F.-B. Tian, J. C. Lai, An immersed boundary method for ﬂuid–
structure–acoustics interactions involving large deformations and complex ge-
ometries, Journal of Fluids and Structures 95 (2020) 102993.
[52] L. Fu, Review of the High-Order TENO Schemes for Compressible Gas Dynam-
ics and Turbulence, Archives of Computational Methods in Engineering (2023).
doi:10.1007/s11831-022-09877-7.
38


[53] S. Pirozzoli, On the spectral properties of shock-capturing schemes, Journal of
Computational Physics 219 (2) (2006) 489–497.
[54] G. Zhao, M. Sun, A. Memmolo, S. Pirozzoli, A general framework for the evalu-
ation of shock-capturing schemes, Journal of Computational Physics 376 (2019)
924–936.
[55] V. V. Rusanov, Calculation of interaction of non-steady shock waves with ob-
stacles, USSR Computational Mathematics and Mathematical Physics (1961)
267 – 279.
[56] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time
discretization methods, SIAM review 43 (1) (2001) 89–112.
[57] G. A. Sod, A survey of several ﬁnite diﬀerence methods for systems of nonlinear
hyperbolic conservation laws, Journal of Computational Physics 27 (1978) 1–31.
[58] P. D. Lax, Weak solutions of nonlinear hyperbolic equations and their numerical
computation, Communications on Pure and Applied Mathematics 7 (1954) 159–
193.
[59] P. Woodward, The numerical simulation of two-dimensional ﬂuid ﬂow with
strong shocks, Journal of Computational Physics 54 (1984) 115–173.
[60] C.-W. Shu, S. Osher, Eﬃcient implementation of essentially non-oscillatory
shock-capturing schemes,
II, in:
Upwind and High-Resolution Schemes,
Springer, 1989, pp. 328–374.
[61] Z. Xu, C. W. Shu, Anti-diﬀusive ﬂux corrections for high order ﬁnite diﬀerence
WENO schemes, Journal of Computational Physics 205 (2005) 458–485.
39


[62] N. Fleischmann, S. Adami, N. A. Adams, Numerical symmetry-preserving tech-
niques for low-dissipation shock-capturing schemes, Computers & Fluids 189
(2019) 94–107.
[63] H. Wakimura, S. Takagi, F. Xiao, Symmetry-preserving enforcement of low-
dissipation method based on boundary variation diminishing principle, Com-
puters & Fluids 233 (2022) 105227.
40
