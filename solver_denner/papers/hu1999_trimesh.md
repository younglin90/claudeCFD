![](_page_0_Picture_1.jpeg)

# Weighted Essentially Non-oscillatory Schemes on Triangular Meshes<sup>1</sup>

Changqing Hu<sup>∗</sup> and Chi-Wang Shu†

*Division of Applied Mathematics, Brown University, Providence, Rhode Island 02912* E-mail: <sup>∗</sup>hu@cfm.brown.edu and †shu@cfm.brown.edu

Received June 16, 1998; revised November 24, 1998

In this paper we construct high-order weighted essentially non-oscillatory schemes on two-dimensional unstructured meshes (triangles) in the finite volume formulation. We present third-order schemes using a combination of linear polynomials and fourthorder schemes using a combination of quadratic polynomials. Numerical examples are shown to demonstrate the accuracies and robustness of the methods for shock calculations. °<sup>c</sup> 1999 Academic Press

*Key Words:* weighted essentially non-oscillatory schemes; unstructured mesh; high-order accuracy; shock calculations.

### **1. INTRODUCTION**

ENO (essentially non-oscillatory) schemes (Harten *et al.* [16], Shu and Osher [28, 29]) have been successfully applied to solve hyperbolic conservation laws and other convection dominated problems, for example in simulating shock turbulence interactions (Shu and Osher [29], Shu *et al.* [30], and Adams and Shariff [2]), in the direct simulation of compressible turbulence (Shu *et al.* [30], Walsteijn [35], and Ladeinde *et al.* [20]), in solving the relativistic hydrodynamics equations (Dolezal and Wong [8]), in shock vortex interactions and other gas dynamics problems (Casper and Atkins [6] and Erlebacher *et al.* [10]), in incompressible flow calculations (E and Shu [9] and Harabetian *et al.* [13]), in solving the viscoelasticity equations with fading memory (Shu and Zeng [31]), in semiconductor device simulation (Fatemi *et al.* [11] and Jerome and Shu [17, 18]), and in image processing and level set methods (Osher and Sethian [24], Sethian [26], and Siddiqi *et al.* [32]). The original ENO paper by Harten *et al.* [16] was for a one-dimensional finite volume formulation. Later,

<sup>1</sup> Research was supported in part by ARO Grant DAAG55-97-1-0318; NSF Grants DMS-9804985, ECS-9627849, and INT-9601084; NASA Langley Grant NAG-1-2070 and Contract NAS1-97046 while the second author was in residence at ICASE, NASA Langley Research Center, Hampton, VA 23681-2199, and AFOSR Grant F49620-96-1-0150.

![](_page_0_Picture_11.jpeg)

this finite volume formulation of ENO schemes was extended to two-dimensional structured meshes by Harten [14] and by Casper [5], and to unstructured triangular meshes by Abgrall [1], Harten and Chakravarthy [15], and Sonar [34]. Finite volume ENO schemes based on a staggered grid and Lax–Friedrichs formulation were given in Bianco *et al.* [4]. Although finite difference versions of ENO schemes [28, 29] are more efficient for multidimensional calculations, finite volume schemes have the advantage of easy handling of complicated geometry by arbitrary triangulations.

Weighted ENO (WENO) schemes were developed later to improve upon ENO schemes, in Liu *et al.* [23] and Jiang and Shu [19]. Advantages of WENO schemes over ENO include the smoothness of numerical fluxes, better steady-state convergence, and better accuracy using the same stencils. Levy *et al.* [22] designed one-dimensional finite volume WENO schemes based on a staggered grid and Lax–Friedrichs formulation.

For a review of ENO and WENO schemes, see [27].

Recently, Friedrich [12] constructed WENO schemes on unstructured meshes using a covolume formulation as in Abgrall [1]. The WENO schemes in [12] only achieve the same order of accuracy as the corresponding ENO schemes when the same set of stencils is considered. This is not optimal, as was known in Jiang and Shu [19] for structured meshes.

In this paper, we present higher order WENO schemes on triangular meshes when using the same set of ENO stencils. We will construct third-order schemes using a combination of two-dimensional linear polynomials and fourth-order schemes using a combination of two-dimensional quadratic polynomials.

We will first sketch the procedure to construct the high-order linear schemes. The formulation at this stage is important to accommodate nonlinear WENO weights later. We then describe the thirdand fourth-order WENO schemes. Numerical examples will be given, to demonstrate the accuracy and resolution of the constructed schemes. Concluding remarks are included at the end.

#### **2. FINITE VOLUME FORMULATION**

In this paper we solve the two-dimensional conservation law

$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} + \frac{\partial g(u)}{\partial y} = 0 \tag{2.1}$$

using the finite volume formulation. Computational control volumes are simply triangles.

Taking the triangle4*<sup>i</sup>* as our control volume, we formulate the semi-discrete finite volume scheme of Eq. (2.1) as

$$\frac{d}{dt}\bar{u}_i(t) + \frac{1}{|\Delta_i|} \int_{\partial \Delta_i} F \cdot n \, ds = 0, \tag{2.2}$$

where *u*¯*i*(*t*) is the cell average of *u* on the cell 4*i*, *F* = ( *f*, *g*)*<sup>T</sup>* , and *n* is the outward unit normal of the triangle boundary ∂4*<sup>i</sup>* .

The line integral in (2.2) is discretized by a *q*-point Gaussian integration formula,

$$\int_{\partial \Delta_i} F \cdot n \, ds \approx |\partial \Delta_i| \sum_{j=1}^q \omega_j F(u(G_j, t)) \cdot n, \tag{2.3}$$

and  $F(u(G_j, t)) \cdot n$  is replaced by a numerical flux. The simple Lax–Friedrichs flux is used in all our numerical experiments, which is given by

$$F(u(G_j,t)) \cdot n \approx \frac{1}{2} [(F(u^-(G_j,t)) + F(u^+(G_j,t))) \cdot n - \alpha(u^+(G_j,t) - u^-(G_j,t))],$$
(2.4)

where  $\alpha$  is taken as an upper bound for the eigenvalues of the Jacobian in the *n* direction, and  $u^-$  and  $u^+$  are the values of *u* inside the triangle and outside the triangle (inside the neighboring triangle) at the Gaussian point.

Since we are constructing schemes up to fourth-order accuracy, two-point Gaussian q=2 is used, which has  $G_1=cP_1+(1-c)P_2$ ,  $G_2=cP_2+(1-c)P_1$ ,  $c=\frac{1}{2}+\frac{\sqrt{3}}{6}$  and  $\omega_1=\omega_2=\frac{1}{2}$  for the line with endpoints  $P_1$  and  $P_2$ .

#### 3. RECONSTRUCTION AND LINEAR SCHEMES

Let  $P^k$  denote the set of two-dimensional polynomials of degree less than or equal to k. The reconstruction problem, from cell averages to point values, is as follows: given a smooth function u, and a triangulation with triangles  $\{\Delta_0, \Delta_1, \ldots, \Delta_N\}$ , we would like to construct, for each triangle  $\Delta_i$ , a polynomial p(x, y) in  $P^k$  that has the same mean value as u on  $\Delta_i$ , and is a (k + 1)th-order approximation to u on the cell  $\Delta_i$ . The mean value of a function u(x, y) on a cell  $\Delta_i$  is defined as

$$\bar{u}_i \equiv \frac{1}{|\Delta_i|} \int_{\Delta_i} u(x, y) \, dx \, dy. \tag{3.1}$$

In order to determine  $K = \frac{(k+1)(k+2)}{2}$  degrees of freedom in a kth degree polynomial p, we need to use the information of at least K triangles. In addition to  $\Delta_i$  itself, we may take its K-1 neighboring cells, and we rename these K triangles as  $S_i = \{\Omega_1, \Omega_2, \ldots, \Omega_K\}$ ,  $S_i$  is called a stencil for the triangle  $\Delta_i$ . If we require that p has the mean value  $\bar{u}_j$  on  $\Omega_j$  for all  $1 \le j \le K$ , we will get a  $K \times K$  linear system. If this linear system has a unique solution,  $S_i$  is called an *admissible* stencil. Of course, in practice, we also have to worry about any ill-conditioned linear system even if it is invertible. For linear polynomials k = 1, a stencil formed by  $\Delta_i$  and two of its neighbors is admissible for most triangulations.

#### 3.1. Third-Order Reconstruction

To construct a third-order linear scheme (a scheme is called linear if it is linear when applied to a linear equation with constant coefficients) as a starting point for the WENO procedure, we need a quadratic polynomial reconstruction. Notice that, as a linear scheme, the stencil of this quadratic polynomial depends not on the solution, but only on the local geometry of the mesh. It seems that one robust way is the least-square reconstruction suggested by Barth and Frederickson [3]. For the control volume of triangle  $\Delta_0$  (see Fig. 3.1), let  $\Delta_i$ ,  $\Delta_j$ ,  $\Delta_k$  be its three neighbors, and  $\Delta_{ia}$ ,  $\Delta_{ib}$  be the two neighbors (other than  $\Delta_0$ ) of  $\Delta_i$ , and so on. We determine the quadratic polynomial  $p^2$  by requiring that it have the same cell average as u on  $\Delta_0$ , and also it matches the cell averages of u on the triangles in the set

$$\{\Delta_i, \Delta_{ia}, \Delta_{ib}, \Delta_j, \Delta_{ja}, \Delta_{jb}, \Delta_k, \Delta_{ka}, \Delta_{kb}\},\$$

![](_page_3_Figure_1.jpeg)

**FIG. 3.1.** A typical stencil.

in a *least-square* sense (as this is an overdetermined system). Notice that some of the neighbors' neighbors (4*ia*, 4*ib*, 4*ja*,...) may coincide. For example, 4*ib* might be the same as 4*ja*. This, however, does not affect the least-square procedure to determine *p*2.

A key step in building a high-order WENO scheme based on lower order polynomials is carried out in the following. We want to construct several linear polynomials whose weighted average will give the same result as the quadratic reconstruction *p*<sup>2</sup> at each quadrature point (the weights are different for different quadrature points).

We first build the following nine linear polynomials by agreeing with the cell averages of *u* on the following stencils: *p*<sup>1</sup> (on triangles: 0, *j*, *k*), *p*<sup>2</sup> (on triangles: 0, *k*,*i*), *p*<sup>3</sup> (on triangles: 0,*i*, *j*), *p*<sup>4</sup> (on triangles: 0,*i*,*ia*), *p*<sup>5</sup> (on triangles: 0,*i*,*ib*), *p*<sup>6</sup> (on triangles: 0, *j*, *ja*), *p*<sup>7</sup> (on triangles: 0, *j*, *jb*), *p*<sup>8</sup> (on triangles: 0, *k*, *ka*), and *p*<sup>9</sup> (on triangles: 0, *k*, *kb*).

For each quadrature point (*x <sup>G</sup>*, *y<sup>G</sup>*), we want to find the linear weights γ*s*, which are constants depending only on the local geometry of the mesh, such that the linear polynomial obtained from a linear combination of these *ps*,

$$R(x, y) = \sum_{s=1}^{9} \gamma_s p_s(x, y), \tag{3.2}$$

satisfies

$$R(x^G, y^G) = p^2(x^G, y^G),$$
 (3.3)

where *p*<sup>2</sup> is defined before using the least-square procedure, for arbitrary choices of cell averages

$$\{\bar{u}_0, \bar{u}_i, \bar{u}_j, \bar{u}_k, \bar{u}_{ia}, \bar{u}_{ib}, \bar{u}_{ja}, \bar{u}_{jb}, \bar{u}_{ka}, \bar{u}_{kb}\}.$$
 (3.4)

Since both the left side and the right side of the equality (3.3) are linear in the cell averages (3.4), for the equality to hold for arbitrary *u*¯'s in (3.4) one must have all 10 coefficients of the *u*¯'s to be identically zero (when all terms are moved to one side of the equality), which leads to 10 linear equations for the nine weights γ*s*. This looks like an overdetermined system, but is in fact underdetermined of rank 8, allowing for one degree of freedom in the choice of the nine γ*s*.

Before explaining this, we first look at a simpler but illustrative one-dimensional example. Let us denote *Ij* , *j* = 0, 1, 2, as three equal-sized consecutive intervals. The two linear polynomials *ps*, where *p*<sup>1</sup> agrees with *u* on cell averages in the intervals *I*<sup>0</sup> and *I*1, and *p*<sup>2</sup> agrees with *u* on cell averages in the intervals *I*<sup>1</sup> and *I*2, give the following two second-order approximations to the value of *u* at the point *x*<sup>3</sup>/<sup>2</sup> (the boundary of *I*<sup>1</sup> and *I*2):

$$-\frac{1}{2}\bar{u}_0 + \frac{3}{2}\bar{u}_1, \qquad \frac{1}{2}\bar{u}_1 + \frac{1}{2}\bar{u}_2. \tag{3.5}$$

The quadratic polynomial *p*2, which agrees with *u* on cell averages in the intervals *I*0, *I*1, and *I*2, gives the following third-order approximation to the value of *u* at the point *x*<sup>3</sup>/2:

$$-\frac{1}{6}\bar{u}_0 + \frac{5}{6}\bar{u}_1 + \frac{1}{3}\bar{u}_1. \tag{3.6}$$

We would like to find γ*<sup>s</sup>* such that

$$\gamma_1 \left( -\frac{1}{2} \bar{u}_0 + \frac{3}{2} \bar{u}_1 \right) + \gamma_2 \left( \frac{1}{2} \bar{u}_1 + \frac{1}{2} \bar{u}_2 \right) = -\frac{1}{6} \bar{u}_0 + \frac{5}{6} \bar{u}_1 + \frac{1}{3} \bar{u}_1 \tag{3.7}$$

for arbitrary *u*¯'s. This leads to the three equations

$$-\frac{1}{2}\gamma_1 = -\frac{1}{6}, \qquad \frac{3}{2}\gamma_1 + \frac{1}{2}\gamma_2 = \frac{5}{6}, \qquad \frac{1}{2}\gamma_2 = \frac{1}{3},$$

for the two unknowns γ<sup>1</sup> and γ2. It looks like an overdetermined system but is in fact rank 2 and has a unique solution

$$\gamma_1 = \frac{1}{3}, \qquad \gamma_2 = \frac{2}{3}.$$

The reason can be understood if we ask for the validity of the equality (3.7) in the cases of *u* = 1, *u* = *x*, and *u* = *x* 2. Clearly if (3.7) holds in these three cases then it holds for arbitrary choices of *u*¯'s. The crucial observation is that (3.7) holds for both *u* = 1 and *u* = *x* as long as γ<sup>1</sup> +γ<sup>2</sup> = 1, as all three expressions in (3.5) and (3.6) reproduce linear functions exactly. Hence the equality (3.7) is valid for all the three cases *u* = 1, *u* = *x*, and *u* = *x*<sup>2</sup> with only two conditions: γ<sup>1</sup> + γ<sup>2</sup> = 1 and another one obtained when *u* = *x*2, resulting in a solvable 2 × 2 system for γ*s*.

The same argument can be applied in the current two-dimensional case. Although there are 10 linear equations for the nine weights γ*<sup>s</sup>* resulting from equality (3.3), we should notice that equality (3.3) is valid for all three cases *u* = 1, *u* = *x*, and *u* = *y* under only one constraint on γ*s*, namely P<sup>9</sup> *<sup>s</sup>*=<sup>1</sup> γ*<sup>s</sup>* = 1, again because *ps*(*x*) and *p*<sup>2</sup>(*x*) all reproduce linear functions exactly. Thus we can eliminate 2 equations from the 10, resulting in a rank 8 system with one degree of freedom in the solution for γ*s*. In practice, we obtain the solution γ*<sup>s</sup>* for *s* ≥ 2 with γ<sup>1</sup> as the degree of freedom.

The first effort we would like to make is to use this degree of freedom to obtain a set of nonnegative γ*s*, which is important for the WENO procedure to be developed later for shock calculations. Unfortunately, it turns out that, for many triangulations, this is impossible. Some grouping is needed and will be discussed later.

#### 3.2. Fourth-Order Reconstruction

To construct a fourth-order linear scheme as a starting point for the WENO procedure, we need a cubic polynomial reconstruction, which has 10 degrees of freedom. We thus only consider the case where ia, ib, ja, jb, ka, kb are distinct in the stencil (see Fig. 3.1). We construct the cubic polynomial  $p^3$  by requiring that its cell average agree with that of u on each triangle in the 10-triangle stencil shown in Fig. 3.1. It seems that for most triangulations this reconstruction is possible.

Again, the key step in building a high-order WENO scheme based on lower order polynomials is carried out in the following. We would like to construct several quadratic polynomials whose weighted average will give the same result as the cubic reconstruction  $p^3$  at each quadrature point (the weights are different for different quadrature points). The following six quadratic polynomials are constructed by having the same cell averages as u on the corresponding triangles:

```
q_1 (on triangles: 0, i, ia, ib, k, kb), q_2 (on triangles: 0, i, ia, ib, j, ja), q_3 (on triangles: 0, j, ja, jb, i, ib), q_4 (on triangles: 0, j, ja, jb, k, ka), q_5 (on triangles: 0, k, ka, kb, j, jb), q_6 (on triangles: 0, k, ka, kb, i, ia).
```

For each quadrature point  $(x^G, y^G)$ , we would like to find the linear weights such that the linear combination of these  $q_s$ ,

$$Q(x, y) = \sum_{s=1}^{6} \gamma_s q_s(x, y),$$
 (3.8)

satisfies

$$Q(x^{G}, y^{G}) = p^{3}(x^{G}, y^{G})$$
(3.9)

for all  $\bar{u}$ 's.

As before, (3.9) results in 10 linear equations for the six unknowns  $\gamma_s$ , which are the coefficients of the 10 cell averages  $\bar{u}$ 's in (3.4). This looks like a grossly overdetermined system, but it is in fact underdetermined with rank 5, thus allowing a solution for  $\gamma_s$  with one degree of freedom. A crucial observation is again that (3.9) is valid for all the six cases  $u=1,x,y,x^2,xy,y^2$  under just one constraint on the  $\gamma_s$ , namely  $\sum_{s=1}^9 \gamma_s = 1$ , because  $q_s(x)$  and  $p^3(x)$  all reproduce quadratic functions exactly. We can thus eliminate 5 equations from the 10, resulting in a rank 5 system with one degree of freedom in the solution for  $\gamma_s$ . In practice, we obtain the solution  $\gamma_s$  for  $s \geq 2$  with  $\gamma_1$  as the degree of freedom.

Again, the first effort we would like to make is to use this degree of freedom to obtain a set of non-negative  $\gamma_s$ , which is important for the WENO procedure to be developed later for shock calculations. This has been performed for the mostly near-uniform meshes used in the numerical examples.

#### 3.3. Accuracy Test for the Linear Schemes

From the thirdand the fourth-order reconstructions, we can now obtain the thirdand the fourth-order linear schemes for (2.1) by replacing  $u^-(G_j, t)$  in (2.4) with the reconstructed values  $R(x^G, y^G)$  in (3.2) or  $Q(x^G, y^G)$  in (3.8), respectively. Similarly,  $u^+(G_j, t)$  is replaced with the reconstructed values in the neighboring cell  $\Delta_i$ .

![](_page_6_Figure_2.jpeg)

**FIG. 3.2.** Uniform mesh with *h* = <sup>2</sup> <sup>5</sup> for accuracy test.

For the temporal discretization, the third-order TVD Runge–Kutta scheme of Shu and Osher [28] is used. For the fourth-order scheme, we use 4*t* = (4*x*)<sup>4</sup>/<sup>3</sup> to achieve fourthorder accuracy in time, but only for the examples of the accuracy test.

# EXAMPLE 3.1. Two-dimensional linear equation

$$u_t + u_x + u_y = 0, (3.10)$$

with the initial condition *u*0(*x*, *y*) = sin( <sup>π</sup> <sup>2</sup> (*x* + *y*)), −2 ≤ *x* ≤ 2, −2 ≤ *y* ≤ 2, and periodic boundary conditions.

We first use uniform triangular meshes which are obtained by adding one diagonal line in each rectangle, shown in Fig. 3.2 for the coarsest case *h* = <sup>2</sup> <sup>5</sup> . In Table 3.1, the accuracy results are shown for both the third-order scheme (from the combination of linear polynomials) and the fourth-order scheme (from the combination of quadratic polynomials), where *h* is the length of the rectangles, at *t* = 2.0. The errors presented are those of the cell averages of *u*.

We then use non-uniform meshes, shown in Fig. 3.3 for the coarsest case *h* = *h*<sup>0</sup> = 1, where *h* is just an average length. The refinement of the meshes is done in a uniform way, namely by cutting each triangle into four smaller similar ones. The accuracy result is shown in Table 3.2. We remark that *h* here is only a rough indicator of the mesh size. There are more triangles here than in the uniform mesh cases with the same *h*.

**TABLE 3.1 Accuracy for 2D Linear Equation, Uniform Meshes, Linear Schemes**

|      |          | P1 (third order) |          |       |          |       | P2 (fourth order) |       |
|------|----------|------------------|----------|-------|----------|-------|-------------------|-------|
| h    | L1 error | Order            | L∞ error | Order | L1 error | Order | L∞ error          | Order |
| 2/5  | 1.80E-01 | —                | 2.79E-01 | —     | 1.40E-02 | —     | 2.17E-02          | —     |
| 1/5  | 2.81E-02 | 2.68             | 4.37E-02 | 2.68  | 9.11E-04 | 3.94  | 1.41E-03          | 3.94  |
| 1/10 | 3.65E-03 | 2.95             | 5.72E-03 | 2.93  | 5.57E-05 | 4.03  | 8.72E-05          | 4.02  |
| 1/20 | 4.60E-04 | 2.99             | 7.22E-04 | 2.99  | 3.43E-06 | 4.02  | 5.39E-06          | 4.02  |
| 1/40 | 5.76E-05 | 3.00             | 9.05E-05 | 3.00  | 2.12E-07 | 4.02  | 3.34E-07          | 4.01  |
| 1/80 | 7.21E-06 | 3.00             | 1.13E-05 | 3.00  | 1.32E-08 | 4.01  | 2.07E-08          | 4.01  |

![](_page_7_Figure_1.jpeg)

**FIG. 3.3.** Non-uniform mesh with h = 1 for accuracy test.

## EXAMPLE 3.2. Two-dimensional Burgers' equation

$$u_t + \left(\frac{u^2}{2}\right)_x + \left(\frac{u^2}{2}\right)_y = 0,$$
 (3.11)

with the initial condition  $u_0(x, y) = 0.3 + 0.7 \sin(\frac{\pi}{2}(x + y)), -2 \le x \le 2, -2 \le y \le 2$ , and periodic boundary conditions.

We first use the same uniform triangular meshes as in Example 3.1, shown in Fig. 3.2 for the coarsest case  $h=\frac{2}{5}$ . In Table 3.3, the accuracy results are shown for both the third-order scheme and the fourth-order scheme, at  $t=0.5/\pi^2$  when the solution is still smooth. The errors presented are those of the point values at the six quadrature points of each triangle, as in this nonlinear case it is easier to obtain the exact solution of the PDE in point values than in cell averages.

We then use the same non-uniform meshes as in Example 3.1, shown in Fig. 3.3 for the coarsest case. The accuracy result is shown in Table 3.4.

EXAMPLE 3.3. Two-dimensional vortex evolution problem for the Euler equations. See [27] for a description of this problem. We consider the compressible Euler equations of gas dynamics

$$\xi_t + f(\xi)_x + g(\xi)_y = 0,$$
 (3.12)

TABLE 3.2
Accuracy for 2D Linear Equation, Non-uniform Meshes, Linear Schemes

|          |             | $P^1$ (thir | d order)           |       | $P^2$ (fourth order) |       |                    |       |
|----------|-------------|-------------|--------------------|-------|----------------------|-------|--------------------|-------|
| h        | $L^1$ error | Order       | $L^{\infty}$ error | Order | $L^1$ error          | Order | $L^{\infty}$ error | Order |
| $h_0/2$  | 1.21E-01    | _           | 2.25E-01           | _     | 4.95E-03             | _     | 1.73E-02           | _     |
| $h_0/4$  | 1.81E-02    | 2.74        | 3.74E-02           | 2.59  | 2.90E-04             | 4.09  | 1.42E-03           | 3.61  |
| $h_0/8$  | 2.36E-03    | 2.94        | 5.39E-03           | 2.80  | 2.21E-05             | 3.71  | 8.32E-05           | 4.09  |
| $h_0/16$ | 3.00E-04    | 2.98        | 7.19E-04           | 2.91  | 1.29E-06             | 4.10  | 5.09E-06           | 4.03  |
| $h_0/32$ | 3.78E-05    | 2.99        | 9.40E-05           | 2.94  | 7.76E-08             | 4.06  | 3.16E-07           | 4.01  |
| $h_0/64$ | 4.75E-06    | 2.99        | 1.22E-05           | 2.95  | 4.75E-09             | 4.03  | 1.95E-08           | 4.02  |

|      |          |       | P1 (third order) |       | P2 (fourth order) |       |          |       |
|------|----------|-------|------------------|-------|-------------------|-------|----------|-------|
| h    | L1 error | Order | L∞ error         | Order | L1 error          | Order | L∞ error | Order |
| 2/5  | 2.67E-02 | —     | 7.75E-02         | —     | 8.63E-03          | —     | 2.18E-02 | —     |
| 1/5  | 3.65E-03 | 2.87  | 1.16E-02         | 2.74  | 6.08E-04          | 3.83  | 1.70E-03 | 3.68  |
| 1/10 | 4.60E-04 | 2.99  | 1.52E-03         | 2.93  | 3.97E-05          | 3.94  | 1.16E-04 | 3.87  |
| 1/20 | 5.75E-05 | 3.00  | 1.91E-04         | 2.99  | 2.51E-06          | 3.98  | 7.37E-06 | 3.98  |
| 1/40 | 7.18E-06 | 3.01  | 2.38E-05         | 3.01  | 1.57E-07          | 4.00  | 4.62E-07 | 4.00  |
| 1/80 | 8.96E-07 | 3.00  | 2.97E-06         | 3.00  | 9.83E-09          | 4.00  | 2.89E-08 | 4.00  |

**TABLE 3.3 Accuracy for 2D Burgers' Equation, Uniform Meshes, Linear Schemes**

where

$$\xi = (\rho, \rho u, \rho v, E),$$
  

$$f(\xi) = (\rho u, \rho u^2 + p, \rho u v, u(E+p)),$$
  

$$g(\xi) = (\rho v, \rho u v, \rho v^2 + p, v(E+p)).$$

Here ρ is the density, (*u*, v) is the velocity, *E* is the total energy, *p* is the pressure, and

$$E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho(u^2 + v^2),$$

with γ = 1.4.

The mean flow is ρ = 1, *p* = 1, and (*u*, v) = (1, 1). We add, to the mean flow, an isentropic vortex (perturbations in (*u*, v) and the temperature *T* = *p*/ρ, no perturbation in the entropy *S* = *p*/ρ<sup>γ</sup> )

$$(\delta u, \delta v) = \frac{\epsilon}{2\pi} e^{0.5(1-r^2)} (-\bar{y}, \bar{x})$$
$$\delta T = -\frac{(\gamma - 1)\epsilon^2}{8\gamma \pi^2} e^{1-r^2}, \qquad \delta S = 0,$$

where (*x*¯, *y*¯) = (*x* − 5, *y* − 5), *r* <sup>2</sup> = *x*¯ <sup>2</sup> + *y*¯ 2, and the vortex strength ² = 5.

**TABLE 3.4 Accuracy for 2D Burgers' Equation, Non-uniform Meshes, Linear Schemes**

|       |          |       | P1 (third order) |       | P2 (fourth order)<br>L∞ error<br>L1 error<br>Order<br>3.96E-03<br>—<br>1.88E-02 |      |          |       |
|-------|----------|-------|------------------|-------|---------------------------------------------------------------------------------|------|----------|-------|
| h     | L1 error | Order | L∞ error         | Order |                                                                                 |      |          | Order |
| h0/2  | 1.69E-02 | —     | 7.95E-01         | —     |                                                                                 |      |          | —     |
| h0/4  | 2.23E-03 | 2.92  | 1.23E-02         | 2.69  | 2.87E-04                                                                        | 3.79 | 2.17E-03 | 3.12  |
| h0/8  | 2.84E-04 | 2.97  | 1.69E-03         | 2.86  | 1.90E-05                                                                        | 3.92 | 1.81E-04 | 3.58  |
| h0/16 | 3.57E-05 | 2.99  | 2.22E-04         | 2.93  | 1.20E-06                                                                        | 3.99 | 1.34E-05 | 3.77  |
| h0/32 | 4.48E-06 | 2.99  | 3.00E-05         | 2.89  | 7.57E-08                                                                        | 3.99 | 1.00E-06 | 3.74  |
| h0/64 | 5.63E-07 | 2.99  | 4.26E-06         | 2.82  | 4.75E-09                                                                        | 4.00 | 7.57E-08 | 3.72  |

TABLE 3.5

Accuracy for 2D Euler Equation of Smooth Vortex Evolution, Uniform Meshes,
Linear Schemes

|      |             | $P^1$ (thir | d order)           |       | 5.26E-03 — 7.89E-02<br>7.36E-04 2.84 1.62E-02 |       |                    |       |
|------|-------------|-------------|--------------------|-------|-----------------------------------------------|-------|--------------------|-------|
| h    | $L^1$ error | Order       | $L^{\infty}$ error | Order | $L^1$ error                                   | Order | $L^{\infty}$ error | Order |
| 1    | 1.65E-02    | _           | 2.60E-01           | _     | 5.26E-03                                      | _     | 7.89E-02           | _     |
| 1/2  | 6.31E-03    | 1.39        | 1.21E-01           | 1.10  | 7.36E-04                                      | 2.84  | 1.62E-02           | 2.28  |
| 1/4  | 1.31E-03    | 2.27        | 2.53E-02           | 2.26  | 5.40E-05                                      | 3.77  | 1.03E-03           | 3.98  |
| 1/8  | 2.21E-04    | 2.57        | 4.66E-03           | 2.44  | 2.32E-06                                      | 4.54  | 5.36E-05           | 4.26  |
| 1/16 | 2.98E-05    | 2.89        | 6.44E-04           | 2.86  | 1.10E-07                                      | 4.40  | 2.48E-06           | 4.43  |
| 1/32 | 3.77E-06    | 2.98        | 8.23E-05           | 2.97  | 6.37E-09                                      | 4.11  | 1.25E-07           | 4.31  |

The computational domain is taken as  $[0, 10] \times [0, 10]$ , and periodic boundary conditions are used.

It is clear that the exact solution of the Euler equation with the above initial and boundary conditions is just the passive convection of the vortex with the mean velocity.

The reconstruction procedure is applied to each component of the solution  $\xi$ . We first compute the solution to t=2.0 for the accuracy test. The meshes are the same as those in Example 3.1 suitably scaled for the new spatial domain. The accuracy results are shown in Table 3.5 for the uniform meshes and Table 3.6 for the non-uniform meshes. The errors presented are those of the cell averages of  $\rho$ .

We then fix the mesh at  $h = \frac{1}{8}$  (uniform) and compute the long-time evolution of the vortex. Figure 3.4 is the result by the third-order scheme at t = 0 and after 1, 5, and 10 time periods, and Fig. 3.5 is the result by the fourth-order scheme. We show the line cut through the center of the vortex for the density  $\rho$ . It is easy to see the difference between the thirdand fourth-order schemes. The fourth-order scheme gives almost no dissipation even after 10 periods, while the dissipation is quite noticeable for the long-time results of the third-order scheme. The mesh chosen is on the borderline of resolution for the fourth-order scheme and does not give enough resolution for long-time simulation of the third-order scheme

TABLE 3.6

Accuracy for 2D Euler Equation of Smooth Vortex Evolution, Non-uniform Meshes,

Linear Schemes

|          |             | P <sup>1</sup> (thir | d order)           |       |             | P <sup>2</sup> (four | th order)          |       |
|----------|-------------|----------------------|--------------------|-------|-------------|----------------------|--------------------|-------|
| h        | $L^1$ error | Order                | $L^{\infty}$ error | Order | $L^1$ error | Order                | $L^{\infty}$ error | Order |
| $h_0/2$  | 1.81E-02    | _                    | 2.98E-01           | _     | 7.00E-03    | _                    | 8.16E-02           | _     |
| $h_0/4$  | 7.74E-03    | 1.28                 | 1.44E-01           | 1.05  | 1.18E-03    | 2.57                 | 1.61E-02           | 2.34  |
| $h_0/8$  | 1.67E-03    | 2.21                 | 2.47E-02           | 2.54  | 8.17E-05    | 3.85                 | 1.31E-03           | 3.62  |
| $h_0/16$ | 2.86E-04    | 2.55                 | 4.79E-03           | 2.37  | 4.70E-06    | 4.12                 | 1.10E-04           | 3.57  |
| $h_0/32$ | 3.94E-05    | 2.86                 | 7.95E-04           | 2.59  | 2.68E-07    | 4.13                 | 7.73E-06           | 3.83  |
| $h_0/64$ | 5.07E-06    | 2.96                 | 1.25E-04           | 2.67  | 1.56E-08    | 4.10                 | 5.99E-07           | 3.69  |

![](_page_10_Figure_2.jpeg)

**FIG. 3.4.** 2D vortex evolution: third-order linear scheme.

#### **4. WENO RECONSTRUCTION AND WENO SCHEMES**

In this section, we will introduce non-linear weights to make the resulting schemes suitable for shock computations. To ensure stability near shocks, we need non-negative weights, and thus we need non-negative linear weights to start with. As there is one degree of freedom in the choice of linear weights for both the third-order and fourth-order cases described in the previous section, this will be explored to produce positive linear weights. In the case of the third-order scheme, grouping of polynomials is also used to achieve positivity.

## 4.1. *Positivity of Linear Weights for the Third-Order Scheme*

We will use the same notation as in the previous section. In (3.2), by consistency, P<sup>9</sup> *<sup>s</sup>*=<sup>1</sup> γ*<sup>s</sup>* = 1. We want to group these nine linear polynomials into three groups,

$$\sum_{s=1}^{9} \gamma_s p_s(x, y) = \sum_{s=1}^{3} \tilde{\gamma}_s \tilde{p}_s(x, y),$$

![](_page_10_Figure_9.jpeg)

**FIG. 3.5.** 2D vortex evolution: fourth-order linear scheme.

each *p*˜*s*(*x*, *y*) being still a linear polynomial and a second-order approximation to *u*, with positive coefficients ˜γ*<sup>s</sup>* ≥ 0. We also require the stencils corresponding to the three new linear polynomials *p*˜*s*(*x*, *y*) to be reasonably separated, so that when shocks are present, not all stencils will contain the shock under normal situations.

The grouping we will introduce in the following works for most triangulations. There are, however, cases when it does give some negative coefficients with very small magnitudes, but it does not seem to affect the stability of WENO schemes built upon them.2

For the first quadrature point on side *i* (*G*<sup>1</sup> in Fig. 3.1), Group 1 contains *p*<sup>2</sup> (0, *k*,*i*), *p*<sup>4</sup> (0,*i*,*ia*), and *p*<sup>5</sup> (0,*i*,*ib*),

$$\tilde{p}_1 = (\gamma_2 p_2 + \gamma_4 p_4 + \gamma_5 p_5)/(\gamma_2 + \gamma_4 + \gamma_5), \qquad \tilde{\gamma}_1 = \gamma_2 + \gamma_4 + \gamma_5;$$

Group 2 contains *p*<sup>3</sup> (0,*i*, *j*), *p*<sup>6</sup> (0, *j*, *ja*), and *p*<sup>7</sup> (0, *j*, *jb*),

$$\tilde{p}_2 = (\gamma_3 p_3 + \gamma_6 p_6 + \gamma_7 p_7)/(\gamma_3 + \gamma_6 + \gamma_7), \qquad \tilde{\gamma}_2 = \gamma_3 + \gamma_6 + \gamma_7;$$

Group 3 contains *p*<sup>1</sup> (0, *j*, *k*), *p*<sup>8</sup> (0, *k*, *ka*), and *p*<sup>9</sup> (0, *k*, *kb*),

$$\tilde{p}_3 = (\gamma_1 p_1 + \gamma_8 p_8 + \gamma_9 p_9)/(\gamma_1 + \gamma_8 + \gamma_9), \qquad \tilde{\gamma}_3 = \gamma_1 + \gamma_8 + \gamma_9.$$

The resulting linear polynomial

$$\tilde{R}(x, y) = \sum_{s=1}^{3} \tilde{\gamma}_s \tilde{p}_s(x, y)$$
(4.1)

is identical to *R*(*x*, *y*)in (3.2) and in most cases the coefficients ˜γ*<sup>s</sup>* can be made non-negative by suitably choosing the value of the degree of freedom γ1, through the solution of a group of three linear inequalities for γ1.

We remark that for practical implementation, it is the five constants *ai* which depend on the local geometry only, such that

$$\tilde{p}_1(x^{G^1}, y^{G^1}) = a_1 \bar{u}_0 + a_2 \bar{u}_i + a_3 \bar{u}_k + a_4 \bar{u}_{ia} + a_5 \bar{u}_{ib}, \tag{4.2}$$

that have to be precomputed and stored once the mesh is generated. We do not need to store any information about the polynomial *p*˜ <sup>1</sup> itself.

For the second quadrature point on side *i* (*G*<sup>2</sup> in Fig. 3.1), Group 1 contains *p*<sup>3</sup> (0,*i*, *j*), *p*<sup>4</sup> (0,*i*,*ia*), and *p*<sup>5</sup> (0,*i*,*ib*), with the combination coefficient ˜γ<sup>1</sup> = γ<sup>3</sup> + γ<sup>4</sup> + γ5; Group 2 contains *p*<sup>2</sup> (0, *k*,*i*), *p*<sup>8</sup> (0, *k*, *ka*), and *p*<sup>9</sup> (0, *k*, *kb*), with the combination coefficient γ˜2 = γ<sup>2</sup> + γ<sup>8</sup> + γ9; Group 3 contains *p*<sup>1</sup> (0, *j*, *k*), *p*<sup>6</sup> (0, *j*, *ja*), and *p*<sup>7</sup> (0, *j*, *jb*), with combination coefficient ˜γ<sup>3</sup> = γ<sup>1</sup> + γ<sup>6</sup> + γ7. We can do the same thing for the other two sides(*j*, *k*).

<sup>2</sup> Recently we have learned, in a joint project with R. Biswas and J. Djomehri on using adaptive methods with WENO schemes, that this grouping may fail near adaptively refined regions where triangle sizes change very abruptly. New grouping techniques to overcome this are under investigation.

## 4.2. *Positivity of Linear Weights for the Fourth-Order Scheme*

From Section 3.2, there is a degree of freedom (which was chosen to be γ1) in the determination of the linear weights γ*<sup>s</sup>* in (3.8). Our objective is to use this degree of freedom to obtain non-negative linear weights. This again amounts to solving a group of six linear inequalities for γ1. We have found out through our numerical experiments that positivity of linear weights can only be achieved for nearly uniform meshes. In this paper only such meshes are used for the fourth-order WENO scheme. The difficulty in doing a grouping here similar to the third-order case is that the stencil to each group would then be quite large, and hence shocks may be included in all stencils. Good grouping strategies are still under investigation.

## 4.3. *Smoothness Indicators and Nonlinear Weights*

We finally come to the point of smooth indicators and nonlinear weights. For this we follow exactly Jiang and Shu [19]. For a polynomial *p*(*x*, *y*) with degree up to *k*, we define the following measurement for smoothness,

$$S = \sum_{1 \le |\alpha| \le k} \int_{\Delta} |\Delta|^{|\alpha|-1} (D^{\alpha} p(x, y))^2 dx dy, \tag{4.3}$$

where α is a multi-index and *D* is the derivative operator; for example, when α = (1, 2) then |α| = 3 and *D*<sup>α</sup> *p*(*x*, *y*) = ∂*p*<sup>3</sup>(*x*, *y*)/∂*x*∂*y*2. The nonlinear weights are then defined as

$$\omega_j = \frac{\tilde{\omega}_j}{\sum_i \tilde{\omega}_i}, \qquad \tilde{\omega}_i = \frac{\gamma_i}{(\epsilon + S_i)^2},$$
(4.4)

where γ*<sup>i</sup>* is the *i*th coefficient in the linear combination of polynomials (i.e., the ˜γ*<sup>s</sup>* in (4.1) for the third-order case and the γ*<sup>s</sup>* in (3.8) for the fourth-order case), *Si* is the measurement of smoothness of the *i*th polynomial *pi*(*x*, *y*) (i.e., the *p*˜*<sup>s</sup>* in (4.1) for the third-order case and the *qs* in (3.8) for the fourth-order case), and ² is a small positive number which we take as ² = 10<sup>−</sup><sup>3</sup> for all the numerical experiments in this paper. The numerical results are not very sensitive to the choice of ² in a range from 10<sup>−</sup><sup>2</sup> to 10<sup>−</sup>6. In general, larger ² gives better accuracy for smooth problems but may generate small oscillations for shocks. Smaller ² is more friendly to shocks. The nonlinear weights ω*<sup>j</sup>* in (4.4) would then replace the linear weights γ*<sup>j</sup>* to form a WENO reconstruction.

We emphasize that the smoothness measurements (4.3) are quadratic functions of the cell averages in the stencil. For example, it is the 10 constants *bi* and *ci* , which depend on the local geometry only, such that

$$S = (b_1 \bar{u}_0 + b_2 \bar{u}_i + b_3 \bar{u}_k + b_4 \bar{u}_{ia} + b_5 \bar{u}_{ib})^2 + (c_1 \bar{u}_0 + c_2 \bar{u}_i + c_3 \bar{u}_k + c_4 \bar{u}_{ia} + c_5 \bar{u}_{ib})^2$$
(4.5)

for the smoothness measurements (4.3) of *p*˜ <sup>1</sup> in (4.1), that have to be precomputed and stored once the mesh is generated. We do not need to store any information about the polynomial *p*˜ <sup>1</sup> itself.

# 4.4. *Extension to the Euler Systems*

There are two ways to extend the previous results to systems. One is to do so component by component. This is easy to implement and cost effective, and it seems to work well for the third-order scheme. We will use componentwise methods for all numerical examples with the third-order scheme. Another extension method is by the characteristic decomposition. We will give a brief description in the following.

Let us take one side of the triangle which has the outward unit normal (*nx* , *ny* ). Let *A* be some average Jacobian at one quadrature point,

$$A = n_x \frac{\partial f}{\partial u} + n_y \frac{\partial g}{\partial u}.$$
 (4.6)

For Euler systems, the Roe's mean matrix [25] is used. Denote by *R* the matrix of right eigenvectors and *L* the matrix of left eigenvectors of *A*. Then the scalar triangular WENO scheme can be applied to each of the characteristic fields, i.e., to each component of the vector v = *L u*. With the reconstructed point values v, we define our reconstructed point values *u* by *u* = *R*v.

# 4.5. *Algorithm Flowchart and Parallel Implementation*

We summarize the practical aspects and operation and storage costs of implementing the method:

- Generate a triangular mesh.
- Compute and store the following mesh-dependent constants:
- —The constant coefficients in the linear combinations of cell averages to recover values of *u* at Gaussian points, for each stencil. Examples of these constants are given in (4.2). There are 5 × 3 × 6 = 90 such constants per triangle for the third-order case (5 for each of the 3 grouped linear polynomials *p*˜*<sup>s</sup>* in (4.1), for each of the 6 quadrature points), and 6 × 6 × 6 = 216 such constants per triangle for the fourth-order case (6 for each of the 6 quadratic polynomials *q*˜*<sup>s</sup>* in (3.8), for each of the 6 quadrature points).
- —For the third-order case, the constant coefficients in the linear combinations inside each of the squares of the smoothness indicator *S* in (4.3), for each stencil. Examples of these constants are given in (4.5). There are 10 × 3 × 6 = 180 such constants per triangle for the third-order case (10 for each of the 3 grouped linear polynomials *p*˜*<sup>s</sup>* in (4.1), for each of the 6 quadrature points). For the fourth-order case, the constant coefficients in the combination of all the quadratic terms of the smoothness indicator *S* in (4.3), for each polynomial *qs* in (3.8). There are 21 × 6 = 126 such constants per triangle (21 quadratic terms, such as *u*¯ <sup>2</sup> <sup>0</sup>, *u*¯ <sup>0</sup>*u*¯*i*,... , out of the 6 *u*¯'s in the stencil, for each of the 6 quadratic polynomials *qs* in (3.8)).
- —The total storage requirement per triangle: 270 numbers for the third-order case and 342 numbers for the fourth-order case.
  - Start the time iteration. For each time stage:
- —For each triangle, compute the lower order reconstructions for each quadrature point and the nonlinear weights, using the prestored constants.
- ∗ For the third-order case, per triangle, the number of multiplications is 306 and the number of additions is 234. This is because for each quadrature point there are 3 reconstructions, for the 3 *p*˜*<sup>s</sup>* in (4.1), each with 5 multiplications and 4 additions in (4.2). It

also involves 3 evaluations of the smooth indicator (4.3), for the 3 *p*˜*<sup>s</sup>* in (4.1), each with 12 multiplications and 9 additions in (4.5).

- ∗ For the fourth-order case, per triangle, the number of multiplications is 468 and the number of additions is 300. This is because for each quadrature point there are 6 reconstructions, for the 6 *qs* in (3.8), each with 6 multiplications and 5 additions. There are also 6 evaluations of the smooth indicator (4.3) for the whole triangle, for the 6 *qs* in (3.8), each with 42 multiplications and 20 additions.
- —Compute the numerical flux (2.4) at each quadrature point, then form the residue and forward in time.
- —The total costs per triangle per residue evaluation, including the WENO reconstruction and flux evaluation (2.4), are 390 multiplications/divisions, 285 additions/subtractions, and 6 flux evaluations for the third-order scheme, and 594 multiplications/divisions, 375 additions/subtractions, and 6 flux evaluations for the fourth-order scheme. In our implementation on a SUN Ultra1 workstation, the third-order WENO scheme takes 33.6 µs per triangle per residue evaluation, while the fourth-order WENO scheme takes 35.9 µs. This is about six times the CPU time needed for a linear scheme of the same order of accuracy implemented in a most economical way.

As an explicit method, the WENO schemes constructed above are easily implemented on an IBM SP-2 parallel computer. The parallel efficiency is over 90% when 16 processors are used. Most of the numerical examples with large meshes in the next section are obtained with the SP-2 parallel computer using 16 processors at the Center for Fluid Mechanics of Brown University.

#### **5. NUMERICAL EXAMPLES**

We will implement the thirdand fourth-order WENO schemes developed in the previous sections to some two-dimensional test problems. First, the same accuracy tests as in the linear weights case are given for linear, Burgers' equation and the smooth vortex problem. Next, some non-smooth problems for two-dimensional Euler equations are tested.

## 5.1. *Accuracy Test for Triangular WENO Schemes*

We use the same examples as in Section 3.3 to test the accuracy of thirdand fourth-order triangular WENO schemes constructed in the previous section.

- EXAMPLE 5.1. Two-dimensional linear equation as defined in Example 3.1, Eq. (3.10). The accuracy results are shown in Table 5.1 for the uniform meshes and in Table 5.2 for the non-uniform meshes. We can see that the correct orders of accuracy are obtained by the thirdand fourth-order WENO methods.
- EXAMPLE 5.2. Two-dimensional Burgers' equation as defined in Example 3.2, Eq. (3.11). The accuracy results are shown in Table 5.3 for the uniform meshes and in Table 5.4 for the non-uniform meshes. We can see that the correct orders of accuracy are again obtained by the thirdand fourth-order WENO methods.

To demonstrate the application for shock computations, we continue the the above calculation to *t* = 5/π<sup>2</sup> when discontinuities develop. Figure 5.1 is the result for *h* = 1/20 of

**TABLE 5.1 Accuracy for 2D Linear Equation, Uniform Meshes, WENO Schemes**

|      |          |       | P1 (third order) |       |          |       | P2 (fourth order) |       |
|------|----------|-------|------------------|-------|----------|-------|-------------------|-------|
| h    | L1 error | Order | L∞ error         | Order | L1 error | Order | L∞ error          | Order |
| 2/5  | 2.66E-01 | —     | 4.30E-01         | —     | 1.38E-02 | —     | 2.94E-02          | —     |
| 1/5  | 8.11E-02 | 1.71  | 1.93E-01         | 1.16  | 1.80E-03 | 2.94  | 2.74E-03          | 3.42  |
| 1/10 | 2.65E-02 | 1.62  | 6.16E-02         | 1.65  | 8.87E-05 | 4.34  | 1.46E-04          | 4.23  |
| 1/20 | 2.68E-03 | 3.31  | 8.77E-03         | 2.81  | 4.34E-06 | 4.35  | 7.11E-06          | 4.36  |
| 1/40 | 1.44E-04 | 4.22  | 4.88E-04         | 4.17  | 2.30E-07 | 4.24  | 3.71E-07          | 4.26  |
| 1/80 | 8.05E-06 | 4.16  | 2.40E-05         | 4.35  | 1.34E-08 | 4.10  | 2.12E-08          | 4.13  |

**TABLE 5.2 Accuracy for 2D Linear Equation, Non-uniform Meshes, WENO Schemes**

|       |          |       | P1 (third order) |       |          |       | P2 (fourth order) |       |  |
|-------|----------|-------|------------------|-------|----------|-------|-------------------|-------|--|
| h     | L1 error | Order | L∞ error         | Order | L1 error | Order | L∞ error          | Order |  |
| h0/2  | 2.79E-01 | —     | 5.28E-01         | —     | 1.77E-02 | —     | 6.41E-02          | —     |  |
| h0/4  | 8.43E-02 | 1.73  | 2.32E-01         | 1.19  | 8.85E-04 | 4.32  | 3.07E-03          | 4.38  |  |
| h0/8  | 2.53E-02 | 1.74  | 7.47E-02         | 1.64  | 4.08E-05 | 4.44  | 1.43E-04          | 4.42  |  |
| h0/16 | 2.24E-03 | 3.50  | 1.14E-02         | 2.71  | 1.82E-06 | 4.49  | 6.37E-06          | 4.49  |  |
| h0/32 | 1.18E-04 | 4.25  | 6.83E-04         | 4.06  | 8.95E-08 | 4.35  | 3.36E-07          | 4.25  |  |
| h0/64 | 6.21E-06 | 4.25  | 3.15E-05         | 4.44  | 4.92E-09 | 4.19  | 2.00E-08          | 4.07  |  |

**TABLE 5.3 Accuracy for 2D Burgers' Equation, Uniform Meshes, WENO Schemes**

|      |          |       | P1 (third order) |       | P2 (fourth order) |       |          |       |
|------|----------|-------|------------------|-------|-------------------|-------|----------|-------|
| h    | L1 error | Order | L∞ error         | Order | L1 error          | Order | L∞ error | Order |
| 2/5  | 2.76E-02 | —     | 8.18E-02         | —     | 8.64E-03          | —     | 2.106-02 | —     |
| 1/5  | 4.63E-03 | 2.58  | 1.20E-02         | 2.77  | 6.05E-04          | 3.84  | 1.73E-03 | 3.60  |
| 1/10 | 6.97E-04 | 2.73  | 2.16E-03         | 2.47  | 3.94E-05          | 3.94  | 1.18E-04 | 3.87  |
| 1/20 | 7.12E-05 | 3.29  | 1.90E-04         | 3.51  | 2.50E-06          | 3.98  | 7.42E-06 | 3.99  |
| 1/40 | 7.63E-06 | 3.22  | 2.36E-05         | 3.01  | 1.57E-07          | 3.99  | 4.63E-07 | 4.00  |
| 1/80 | 9.08E-07 | 3.07  | 2.96E-06         | 3.00  | 9.83E-09          | 4.00  | 2.89E-08 | 4.00  |

**TABLE 5.4 Accuracy for 2D Burgers' Equation, Non-uniform Meshes, WENO Schemes**

|       |          |       | P1 (third order) |       |          |       | P2 (fourth order) |       |
|-------|----------|-------|------------------|-------|----------|-------|-------------------|-------|
| h     | L1 error | Order | L∞ error         | Order | L1 error | Order | L∞ error          | Order |
| h0/2  | 2.01E-02 | —     | 9.16E-02         | —     | 4.18E-03 | —     | 2.376-02          | —     |
| h0/4  | 3.85E-03 | 2.38  | 1.80E-02         | 2.35  | 2.90E-04 | 3.85  | 2.61E-03          | 3.18  |
| h0/8  | 5.79E-04 | 2.73  | 3.39E-03         | 2.41  | 1.85E-05 | 3.97  | 1.92E-04          | 3.77  |
| h0/16 | 5.34E-05 | 3.44  | 3.55E-04         | 3.26  | 1.18E-06 | 3.97  | 1.35E-05          | 3.83  |
| h0/32 | 5.12E-06 | 3.38  | 2.95E-05         | 3.59  | 7.45E-08 | 3.99  | 9.99E-07          | 3.76  |
| h0/64 | 5.82E-07 | 3.14  | 4.23E-06         | 2.80  | 4.67E-09 | 4.00  | 7.56E-08          | 3.72  |

![](_page_16_Figure_2.jpeg)

**FIG. 5.1.** 2D Burgers' equation: *t* = 5/π2, uniform mesh.

a uniform mesh. Figure 5.2 is the result for *h* = *h*0/16 of a non-uniform mesh. We can see that the shock transitions are sharp and non-oscillatory.

EXAMPLE 5.3. Two-dimensional vortex evolution problem as defined in Example 3.3. The accuracy results are shown in Table 5.5 for the uniform meshes and in Table 5.6 for the non-uniform meshes. We can see that the correct orders of accuracy are again obtained by the thirdand fourth-order WENO methods.

Figures 5.3 and 5.4 are the results for the long-time evolution of the vortex. These results are similar to those obtained with the linear schemes in Figs. 3.4 and 3.5.

## 5.2. *Riemann Problems of Euler Equations*

The two-dimensional triangular WENO methods are applied to one-dimensional shock tube problems. We consider the solution of the Euler equations (3.12) in a domain of [−1, 1] × [0, 0.2] with a triangulation of 101 vertices in the *x*-direction and 11 vertices in the *y*-direction. The velocity in the *y*-direction is zero, and periodic boundary condition is used in the *y*-direction. Part of the mesh is shown in Fig. 5.5. The pictures shown

![](_page_16_Figure_9.jpeg)

**FIG. 5.2.** 2D Burgers' equation: *t* = 5/π2, non-uniform mesh.

**TABLE 5.5 Accuracy for 2D Euler Equation of Smooth Vortex Evolution, Uniform Meshes, WENO Schemes**

|      |          | P1 (third order)<br>P2 (fourth order) |          |       | L1 error<br>Order |      |          |       |
|------|----------|---------------------------------------|----------|-------|-------------------|------|----------|-------|
| h    | L1 error | Order                                 | L∞ error | Order |                   |      | L∞ error | Order |
| 1    | 1.87E-02 | —                                     | 2.95E-01 | —     | 1.30E-02          | —    | 2.05E-01 | —     |
| 1/2  | 1.01E-02 | 0.89                                  | 2.09E-01 | 0.50  | 2.50E-03          | 2.38 | 4.45E-02 | 2.49  |
| 1/4  | 2.78E-03 | 1.86                                  | 6.37E-02 | 1.71  | 1.79E-04          | 3.80 | 3.29E-03 | 3.76  |
| 1/8  | 6.47E-04 | 2.10                                  | 3.05E-02 | 1.06  | 6.92E-06          | 4.69 | 1.96E-04 | 4.07  |
| 1/16 | 8.74E-05 | 2.89                                  | 8.14E-03 | 1.91  | 2.03E-07          | 5.09 | 4.95E-06 | 5.31  |
| 1/32 | 7.10E-06 | 3.62                                  | 5.66E-04 | 3.85  | 7.83E-09          | 4.70 | 1.96E-07 | 4.66  |

below are obtained by extracting the data along the central cut line for 101 equally spaced points.

We consider the following Riemann-type initial conditions:

$$(\rho, u, p) = \begin{cases} (\rho_L, u_L, p_L) & \text{if } x \le 0 \\ (\rho_R, u_R, p_R) & \text{if } x > 0. \end{cases}$$

The first test case is Sod's problem [33]. The initial data are

$$(\rho_L, u_L, p_L) = (1, 0, 1), \qquad (\rho_R, u_R, p_R) = (0.125, 0, 0.1).$$

Density at *t* = 0.40 is shown in the first three plots in Fig. 5.6.

The second test case is the Riemann problem proposed by Lax [21]:

$$(\rho_L, u_L, p_L) = (0.445, 0.698, 3.528), \qquad (\rho_R, u_R, p_R) = (0.5, 0, 0.571).$$

Density at *t* = 0.26 is shown in the last three plots in Fig. 5.6.

We can observe a better resolution of the fourth-order scheme over the third-order one, and also a less oscillatory result from the characteristic version of the fourth-order scheme over the component version.

**TABLE 5.6 Accuracy for 2D Euler Equation of Smooth Vortex Evolution, Non-uniform Meshes, WENO Schemes**

|       | P1 (third order) |       |          |       | P2 (fourth order) |       |          |       |
|-------|------------------|-------|----------|-------|-------------------|-------|----------|-------|
| h     | L1 error         | Order | L∞ error | Order | L1 error          | Order | L∞ error | Order |
| h0/2  | 2.12E-02         | —     | 3.33E-01 | —     | 1.84E-02          | —     | 2.14E-01 | —     |
| h0/4  | 1.28E-02         | 0.73  | 2.27E-01 | 0.55  | 2.80E-03          | 2.69  | 3.43E-02 | 2.64  |
| h0/8  | 3.84E-03         | 1.74  | 6.85E-02 | 1.73  | 2.12E-04          | 3.72  | 6.57E-03 | 2.38  |
| h0/16 | 8.32E-04         | 2.21  | 3.02E-02 | 1.18  | 1.09E-05          | 4.28  | 5.91E-04 | 3.48  |
| h0/32 | 1.26E-04         | 2.72  | 5.64E-03 | 2.42  | 3.76E-07          | 4.86  | 1.97E-05 | 4.91  |
| h0/64 | 1.16E-05         | 3.44  | 6.19E-04 | 3.19  | 1.66E-08          | 4.50  | 6.78E-07 | 4.86  |

![](_page_18_Figure_2.jpeg)

**FIG. 5.3.** 2D vortex evolution: third-order WENO scheme.

## 5.3. *A Mach 3 Wind Tunnel with a Step*

This problem is from [36]. We solve the Euler equations (3.12) in a wind tunnel of 1 length unit wide and 3 length units long. The step is 0.2 length units high and is located 0.6 length units from the left end of the tunnel. Initially, a right-going Mach 3 flow is used. Reflective boundary conditions are applied along the walls of the tunnel, and inflow and outflow boundary conditions are used at the entrance and the exit.

The corner of the step is a singularity point. [36] uses an assumption of nearly steady flow in the region near the corner to fix this singularity. In this paper, we do not modify our method near the corner; instead we adopt the same technique as the one used in [7], namely refining the mesh near the corner and using the same scheme in the whole domain.

We use the third-order scheme for this problem. Four meshes have been used; see Fig. 5.7. For first mesh, the triangle size away from the corner is roughly equal to a rectangular element case of 4*x* = 4*y* = <sup>1</sup> <sup>40</sup> , while it is one-quarter that near the corner. For the second mesh, the triangle size away from the corner is the same as in the first mesh, but it is one-eighth that near the corner. The third mesh has a triangle size of 4*x* = 4*y* = <sup>1</sup> <sup>80</sup> away

![](_page_18_Figure_8.jpeg)

**FIG. 5.4.** 2D vortex evolution: fourth-order WENO scheme.

![](_page_19_Figure_1.jpeg)

**FIG. 5.5.** Mesh for the Riemann problems.

![](_page_19_Figure_3.jpeg)

**FIG. 5.6.** Riemann problems of Euler equations: density.

![](_page_20_Figure_2.jpeg)

**FIG. 5.7.** Triangulations for the forward step problem: part near the corner.

from the corner, and it is one-quarter that near the corner. The last mesh has a triangle size of  $\triangle x = \triangle y = \frac{1}{160}$  away from the corner, and it is one-half that near the corner. Figure 5.8 is the contour picture for the density at time t = 4.0. It is clear that with more triangles near the corner the artifacts from the singularity decrease significantly.

![](_page_21_Figure_1.jpeg)

**FIG. 5.8.** Forward step problem: 30 contours from 0.32 to 6.15.

![](_page_22_Figure_2.jpeg)

FIG. 5.9. Triangulation for the double Mach reflection.

## 5.4. Double Mach Reflection

This problem (Fig. 5.9) is also from [36]. We solve the Euler equations (3.12) in a computational domain of  $[0, 4] \times [0, 1]$ . A reflecting wall lies at the bottom of the domain starting from  $x = \frac{1}{6}$ . Initially a right-moving Mach 10 shock is located at  $x = \frac{1}{6}$ , y = 0, making a 60° angle with the x-axis. The reflective boundary condition is used at the wall, while for the rest of the bottom boundary (the part from x = 0 to  $x = \frac{1}{6}$ ), the exact postshock condition is imposed. At the top boundary, the flow values are set to describe the exact motion of the Mach 10 shock. The results shown are at t = 0.2.

We test both the thirdand the fourth-order schemes. Four triangle sizes are used; they are roughly equal to rectangular element cases of  $\Delta x = \Delta y = \frac{1}{50}$  (Fig. 5.10),  $\Delta x = \Delta y = \frac{1}{100}$  (Fig. 5.11),  $\Delta x = \Delta y = \frac{1}{200}$  (Figs. 5.12 and 5.13), and  $\Delta x = \Delta y = \frac{1}{400}$  (Figs. 5.14 and 5.15), respectively. For the third-order scheme, we use both uniform triangular mesh (equilateral triangles) and locally refined triangular mesh (the refined region has the above triangle sizes; Fig. 5.9 shows the region  $[0,2] \times [0,1]$  of such a mesh of  $\Delta x = \Delta y = \frac{1}{50}$  locally). Figures 5.10–5.15 provide further examples for the fourth order, we use uniform triangular mesh only. For the cases of  $\Delta x = \Delta y = \frac{1}{200}$  and  $\Delta x = \Delta y = \frac{1}{400}$ , we present both the picture of whole region ( $[0,3] \times [0,1]$ ) and a blow-up region around the double Mach stems. All pictures are the density contours with 30 equally spaced contour lines from 1.5 to 21.5. We can clearly see that the fourth-order scheme captures the complicated flow structure under the triple Mach stem much better than the third-order scheme, and the characteristic version is much less oscillatory than the component version for the fourth-order scheme. We refer to [7] for similar results obtained with discontinuous Galerkin methods.

#### 6. CONCLUDING REMARKS

We have presented the development of thirdand fourth-order WENO schemes based on linear and quadratic polynomials for 2D triangle meshes. Accuracy and stability issues are considered in the design of the schemes and verified numerically. Numerical examples show the improvement of resolution using high-order schemes.

![](_page_23_Figure_1.jpeg)

**FIG. 5.10.** Double Mach reflection: *h* = <sup>1</sup> , *t* = 0.2.

![](_page_24_Figure_2.jpeg)

**FIG. 5.11.** Double Mach reflection: *h* = <sup>1</sup> <sup>100</sup> , *t* = 0.2.

![](_page_25_Figure_1.jpeg)

**FIG. 5.12.** Double Mach reflection: *h* = <sup>1</sup> , *t* = 0.2.

![](_page_26_Figure_2.jpeg)

**FIG. 5.13.** Double Mach reflection: *h* = <sup>1</sup> <sup>200</sup> , *t* = 0.2 (blow-up).

![](_page_27_Figure_1.jpeg)

**FIG. 5.14.** Double Mach reflection: *h* = <sup>1</sup> , *t* = 0.2.

![](_page_28_Figure_2.jpeg)

**FIG. 5.15.** Double Mach reflection: *h* = <sup>1</sup> <sup>400</sup> , *t* = 0.2 (blow-up).

#### **REFERENCES**

- 1. R. Abgrall, On essentially non-oscillatory schemes on unstructured meshes: Analysis and implementation, *J. Comput. Phys.* **114**, 45 (1994).
- 2. N. Adams and K. Shariff, A high-resolution hybrid compact-ENO scheme for shock-turbulence interaction problems, *J. Comput. Phys.* **127**, 27 (1996).
- 3. T. Barth and P. Frederickson, High order solution of the Euler equations on unstructured grids using quadratic reconstruction, AIAA Paper No. 90-0013.
- 4. F. Bianco, G. Puppo, and G. Russo, High order central schemes for hyperbolic systems of conservation laws, *SIAM J. Sci. Comput.*, to appear.
- 5. J. Casper, Finite-volume implementation of high-order essentially non-oscillatory schemes in two dimensions, *AIAA J.* **30**, 2829 (1992).
- 6. J. Casper and H. Atkins, A finite-volume high-order ENO scheme for two dimensional hyperbolic systems, *J. Comput. Phys.* **106**, 62 (1993).
- 7. B. Cockburn and C.-W. Shu, The Runge–Kutta discontinuous Galerkin method for conservation laws. V. Multidimensional systems, *J. Comput. Phys.* **141**, 199 (1998).
- 8. A. Dolezal and S. Wong, Relativistic hydrodynamics and essentially non-oscillatory shock capturing schemes, *J. Comput. Phys.* **120**, 266 (1995).
- 9. W. E and C.-W. Shu, A numerical resolution study of high order essentially non-oscillatory schemes applied to incompressible flow, *J. Comput. Phys.* **110**, 39 (1994).
- 10. G. Erlebacher, Y. Hussaini, and C.-W. Shu, Interaction of a shock with a longitudinal vortex, *J. Fluid Mech.* **337**, 129 (1997).
- 11. E. Fatemi, J. Jerome, and S. Osher, Solution of the hydrodynamic device model using high order non-oscillatory shock capturing algorithms, *IEEE Trans. Computer-Aided Design Integrated Circuits Syst.* **10**, 232 (1991).
- 12. O. Friedrichs, Weighted essentially non-oscillatory schemes for the interpolation of mean values on unstructured grids, *J. Comput. Phys.* **144**, 194 (1998).
- 13. E. Harabetian, S. Osher, and C.-W. Shu, An Eulerian approach for vortex motion using a level set regularization procedure, *J. Comput. Phys.* **127**, 15 (1996).
- 14. A. Harten, Preliminary results on the extension of ENO schemes to two dimensional problems, in *Proceedings of the International Conference on Hyperbolic Problems, Saint-Etienne, 1986*.
- 15. A. Harten and S. Chakravarthy, *Multi-dimensional ENO Schemes for General Geometries*, ICASE Report No. 91-76 (1991).
- 16. A. Harten, B. Engquist, S. Osher, and S. Chakravarthy, Uniformly high order essentially non-oscillatory schemes, III, *J. Comput. Phys.* **71**, 231 (1987).
- 17. J. Jerome and C.-W. Shu, Energy models for one-carrier transport in semiconductor devices, in *IMA Volumes in Mathematics and Its Applications, Vol. 59*, edited by W. Coughran, J. Cole, P. Lloyd, and J. White (Springer-Verlag, Berlin/New York, 1994), p. 185.
- 18. J. Jerome and C.-W. Shu, Transport effects and characteristic modes in the modeling and simulation of submicron devices, *IEEE Trans. Computer-Aided Design Integrated Circuits Syst.* **14**, 917 (1995).
- 19. G. Jiang and C.-W. Shu, Efficient implementation of weighted ENO schemes, *J. Comput. Phys.* **126**, 202 (1996).
- 20. F. Ladeinde, E. O'Brien, X. Cai, and W. Liu, Advection by polytropic compressible turbulence, *Phys. Fluids* **7**, 2848 (1995).
- 21. P. D. Lax, Weak solutions of nonlinear hyperbolic equations and their numerical computation, *Commun. Pure Appl. Math.* **7**, 159 (1954).
- 22. D. Levy, G. Puppo, and G. Russo, Central WENO schemes for hyperbolic systems of conservation laws, *Math. Modelling Numer. Anal.*, to appear.
- 23. X.-D. Liu, S. Osher, and T. Chan, Weighted essentially non-oscillatory schemes, *J. Comput. Phys.* **115**, 200 (1994).
- 24. S. Osher and J. Sethian, Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton– Jacobi formulation, *J. Comput. Phys.* **79**, 12 (1988).

- 25. P. L. Roe, Approximate Riemann solvers, parameter vectors, and difference schemes, *J. Comput. Phys.* **43**, 357 (1981).
- 26. J. Sethian, *Level Set Methods: Evolving Interfaces in Geometry, Fluid Dynamics, Computer Vision, and Material Science*, Cambridge Monographs on Applied and Computational Mathematics (Cambridge Univ. Press, New York, 1996).
- 27. C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws, in *Advanced Numerical Approximation of Nonlinear Hyperbolic Equations*, edited by A. Quarteroni, Editor, Lecture Notes in Mathematics, CIME subseries (Springer-Verlag, Berlin/New York, to appear); ICASE Report 97-65.
- 28. C.-W. Shu and S. Osher, Efficient implementation of essentially non-oscillatory shock capturing schemes, *J. Comput. Phys.* **77**, 439 (1988).
- 29. C.-W. Shu and S. Osher, Efficient implementation of essentially non-oscillatory shock capturing schemes, II, *J. Comput. Phys.* **83**, 32 (1989).
- 30. C.-W. Shu, T. A. Zang, G. Erlebacher, D. Whitaker, and S. Osher, High order ENO schemes applied to twoand three-dimensional compressible flow, *Appl. Numer. Math.* **9**, 45 (1992).
- 31. C.-W. Shu and Y. Zeng, High order essentially non-oscillatory scheme for viscoelasticity with fading memory, *Quart. Appl. Math.* **55**, 459 (1997).
- 32. K. Siddiqi, B. Kimia, and C.-W. Shu, Geometric shock-capturing ENO schemes for subpixel interpolation, computation and curve evolution, *Comput. Vision Graphics Image Processing Graphical Models Image Process.* **59**, 278 (1997).
- 33. G. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, *J. Comput. Phys.* **27**, 1 (1978).
- 34. T. Sonar, On the construction of essentially non-oscillatory finite volume approximations to hyperbolic conservation laws on general triangulations: Polynomial recovery, accuracy and stencil selection, *Comput. Methods Appl. Mech. Engrg.* **140**, 157 (1997).
- 35. F. Walsteijn, Robust numerical methods for 2D turbulence, *J. Comput. Phys.* **114**, 129 (1994).
- 36. P. Woodward and P. Colella, The numerical simulation of two-dimensional fluid flow with strong shocks, *J. Comput. Phys.* **54**, 115 (1984).