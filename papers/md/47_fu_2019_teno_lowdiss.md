Contents lists available at [ScienceDirect](http://www.elsevier.com/locate/cpc)

# Computer Physics Communications

journal homepage: [www.elsevier.com/locate/cpc](http://www.elsevier.com/locate/cpc)

![](_page_0_Picture_5.jpeg)

# A low-dissipation finite-volume method based on a new TENO shock-capturing scheme

![](_page_0_Picture_7.jpeg)

Lin Fu [∗](#page-0-0)

*Center for Turbulence Research, Stanford University, Stanford, CA 94305, USA*

# a r t i c l e i n f o

*Article history:* Received 13 March 2018 Received in revised form 30 August 2018 Accepted 8 October 2018 Available online 19 October 2018

*Keywords:* TENO High-order schemes Shock-capturing schemes Low-dissipation schemes Finite-volume method

### a b s t r a c t

In Fu et al. (2016), a family of high-order TENO shock-capturing schemes has been proposed for compressible fluid simulations within a finite-difference framework. With the TENO weighting strategy, each candidate stencil is either applied for the final reconstruction with its optimal weight or discarded completely when crossed by discontinuities. In this paper, with the observation that the local flow scales can be judged to be smooth or non-smooth explicitly, we propose a novel low-dissipation finitevolume method based on a new TENO reconstruction. Firstly, a new ENO-like stencil selection paradigm, which adapts between three three-point small stencils and a large candidate stencil, is proposed. The resulting TENO scheme inherits the low-dissipation advantage of original TENO schemes and can be extended to arbitrarily high-order reconstruction without significant complexity increase. The optimal background linear scheme on the three small stencils and that on the large stencil can be optimized either approaching high-order accuracy or better spectral properties separately. Secondly, within the finitevolume framework, a ''low-dissipation'' Riemann solver is applied for flux computing when the large candidate stencils for both the leftand right-side reconstruction are judged as smooth whereas a robust "dissipative" Riemann solver is adopted when one large candidate stencil crosses discontinuities. Since the numerical dissipation from both the reconstruction stage and the flux computing stage can be tuned according to the TENO weighting strategy, the proposed finite-volume method is less-dissipative and provides additional flexibility to handle challenging simulations. A set of benchmark cases is simulated to assess the performance of proposed method.

© 2018 Elsevier B.V. All rights reserved.

# **1. Introduction**

Compressible fluid dynamics governed by hyperbolic conservation laws plays an important role in modern engineering industries. To understand the flow physics by computational fluid dynamics has been popular over the last several decades. Moreover, several numerical methods, e.g. the finite-difference method [[1](#page-14-0)], the finite-volume method [\[2\]](#page-14-1), discontinuous Galerkin (DG) method [\[3](#page-14-2)], have been proposed to solve the hyperbolic conservation laws [[4](#page-14-3)]. Due to the challenge that practical flows involve widespread spatial and temporal length scales, further improvements of these state-of-the-art methods are attractive. Compared with the finite-difference method, the finite-volume method is more widely employed in engineering community owing to the advantages of handling different types of meshes and maintaining the local conservation. With a proper choice of test function, the DG method can degenerate to standard finite-volume method. Inspired by the pioneering work of Kolgan [\[5\]](#page-14-4) and Godunov [[6\]](#page-14-5), the classical finite-volume method consists of two typical procedures, i.e. reconstructing the high-order data at cell interface and solving a local Riemann problem.

In terms of the high-order reconstruction, several concepts, e.g. the total variation diminishing (TVD) scheme [[7](#page-14-6)[,8](#page-14-7)], the essentially non-oscillatory (ENO) scheme [[9\]](#page-14-8) and the weighted essentially non-oscillatory (WENO) scheme [[10](#page-14-9)], have been proposed. In particular, the ENO [[9](#page-14-8)] and WENO [[10](#page-14-9)] schemes have gained widespread popularity due to the high-order accuracy and the sharp shock-capturing property. Jiang and Shu [\[11\]](#page-14-10) propose new smoothness indicators by minimizing the *L*<sup>2</sup> norm of the derivatives of the reconstructed polynomials and construct the fifthorder WENO-JS scheme. Afterwards, lots of efforts focus on further improving the performance of classical WENO-JS schemes, e.g. the WENO-M [[12\]](#page-14-11) scheme by remapping the nonlinear weights of WENO-JS [[11](#page-14-10)] to satisfy the sufficient criteria of fifth-order accuracy at critical points, the WENO-Z [[13](#page-14-12)] scheme by exploiting a high-order global reference to design the smoothness indicators, the hybrid WENO scheme [\[14](#page-14-13)], the very high-order WENO scheme [[15](#page-14-14)], the WENO scheme with modified weighting strategy [[16\]](#page-14-15) etc.

Godunov [[6](#page-14-5)] first presents a first-order method which is capable of capturing discontinuities without spurious oscillations for Euler

<sup>∗</sup> Corresponding author. *E-mail address:* [linfu@stanford.edu.](mailto:linfu@stanford.edu)

equations. The key is to compute the exact or approximate solution of a Riemann problem at the cell interface. While the exact solution is achievable in principle, it is computationally expensive due to the iterative procedure. Consequently, developing effective and efficient approximate Riemann solver becomes attractive. Roughly speaking, the classical approximate Riemann solvers can be categorized as the flux-vector splitting (FVS) methods and fluxdifference splitting (FDS) methods. The FVS methods, e.g. the Van Leer splitting [17], the Steger-Warming splitting [18] and the advection upstream splitting method (AUSM) [19], are highly efficient. The FDS methods, e.g. the Roe [20] scheme, the Harten, Lax and Van Leer (HLL) [21] and HLLC [22,23] scheme, typically feature higher resolution [24]. In comparison with the HLL scheme, which yields exact resolution of isolated shock waves, the HLLC scheme can additionally resolve the contact wave with an appropriate choice of signal velocities [22]. Based on these low-dissipation Riemann solvers, Titarev and Toro [25] develop the very-highorder finite-volume WENO scheme for three-dimensional conservation laws by imposing a suitable Gaussian numerical quadrature. In [26], the performance of the multi-dimensional finite-volume WENO method with/without Gaussian quadrature has been investigated systematically for both the smooth flows and the nonsmooth shocked flows.

The performance of classical finite-volume methods may be limited by two problems: (1) the high-order reconstruction schemes are unnecessarily dissipative that even the smooth flow scales are smeared significantly; (2) there are no universal Riemann solvers applicable to all sophisticated simulations with rich lengthscales, e.g. while a dissipative numerical flux benefits the shockwave capturing, the smooth flow scales are dissipated either. In this paper, we propose a low-dissipation finite-volume method to address these two issues. Firstly, based on the work of Fu et al. [27-30], a new set of high-order TENO scheme is proposed. The candidate stencils include three small three-point stencils and one large candidate stencil. With the ENO-like stencil selection, if the large stencil is judged as smooth, the final reconstruction is determined by the large candidate stencil independently; otherwise, the adaptation within the three small stencils is invoked to handle the nonlinear scales. The resulting TENO scheme inherits the advantages of original TENO schemes [27] and is low dissipative. Secondly, a new flux computing strategy based on the TENO indicator is proposed. When the large candidate stencils for both the leftand right-side reconstruction are judged as smooth, a "low dissipative" Riemann flux is employed to resolve the smooth scales; whereas, when the nonlinear adaptation is employed in the TENO reconstruction, a "dissipative" Riemann flux is adopted to avoid spurious oscillations. This new strategy provides additional flexibility in choosing different Riemann solvers in a unified framework. While a "dissipative" Riemann flux can stabilize the simulations with strong shocks, the smooth scales are separated and resolved with low numerical dissipation. Consequently, the numerical dissipation resulting from both the reconstruction stage and the flux evaluation stage is controllable.

The remaining of this paper is organized as follows. In Section 2, the classical finite-volume framework including the canonical Riemann solvers is briefly introduced. In Sections 3 and 4, the new low-dissipation finite-volume method is proposed. The new high-order TENO scheme and the novel flux computing strategy are developed in detail. In Section 5, the extension to multi-dimensional problems and unstructured meshes is discussed. In Section 6, a set of benchmark cases involving both strong discontinuities and rich flow scales is simulated to assess the performance of proposed low-dissipation method. Concluding remarks are given in the last section.

#### 2. Standard finite-volume method

In this section, the classical finite-volume method and the sources of numerical dissipation are discussed.

#### 2.1. Concepts of finite-volume method

For the facilitation of presentation, the one-dimensional scalar hyperbolic conservation law

$$\frac{\partial u}{\partial t} + \frac{\partial}{\partial x} f(u) = 0 \tag{1}$$

is considered as the prototype hereinafter. u(x,t) denotes the conservative variable and f(u) denotes the flux function.  $\alpha = \frac{\partial f(u)}{\partial u}$  denotes the characteristic velocity.

After discretizing Eq. (1) on a uniform non-overlapping cell elements, e.g.  $I_i = [x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}]$  and  $\Delta x = x_{i+\frac{1}{2}} - x_{i-\frac{1}{2}}, i = 0, \dots, N$ ,

a system of ordinary differential equations

$$\frac{d\bar{u}_i}{dt} = -\frac{1}{\Delta x} \int_{x_i - \Delta x/2}^{x_i + \Delta x/2} \frac{\partial f}{\partial x} dx, \qquad i = 0, \dots, N$$
 (2)

where

$$\bar{u}_i(t) = \frac{1}{\Delta x} \int_{x_i - \Delta x/2}^{x_i + \Delta x/2} u(x, t) dx$$
 (3)

denotes the volume-averaged conservative variable in cell element  $I_i$ , is formed and can be marched by explicit Runge-Kutta method [31] to obtain a physically-consistent weak solution. Eq. (2) can be further approximated by a conservative finite-volume meth-od as

$$\frac{d\bar{u}_i}{dt} \approx -\frac{1}{\Delta x} (\widehat{f}_{i+1/2} - \widehat{f}_{i-1/2}), \tag{4}$$

where the numerical flux  $\hat{f}_{i+1/2}$  at the cell interface is computed by a Riemann solver [32]

$$\widehat{f}_{i+\frac{1}{2}} = f_{i+\frac{1}{2}}^{\text{Riemann}}(u_{i+\frac{1}{2}}^L, u_{i+\frac{1}{2}}^R), \tag{5}$$

and the left-side data  $u^L_{i+\frac{1}{2}}$  and the right-side data  $u^R_{i+\frac{1}{2}}$  are com-

puted by the leftand right-biased reconstruction, respectively. Based on this framework, Godunov [6] presented a first-order scheme which is capable of capturing discontinuities without spurious oscillations for Euler equations. After that, the state-of-the-art researches focus on achieving the high-order accuracy in smooth regions while preserving the shock-capturing property near discontinuities [33].

# 2.2. Numerical dissipation source

Although the exact solution of a Riemann problem at the cell interface can be solved, approximate Riemann solvers are widely adopted due to the high efficiency [32]. While there exist many different variants of approximate riemann solvers, e.g. the Rusanov (local LxF) flux [32,34], the Roe flux [20] and the HLLC flux [23], they can be formulated into a general form as [35]

$$f_{i+\frac{1}{2}}^{\text{Riemann}}(u_{i+\frac{1}{2}}^{L}, u_{i+\frac{1}{2}}^{R}) = \frac{1}{2}(f(u_{i+\frac{1}{2}}^{L}) + f(u_{i+\frac{1}{2}}^{R}) - \left|\tilde{\partial}_{i+\frac{1}{2}}\right|(u_{i+\frac{1}{2}}^{R} - u_{i+\frac{1}{2}}^{L})),$$
(6)

where  $\tilde{\delta}_{i+\frac{1}{2}}$  denotes the characteristic signal velocity evaluated at the cell interface.

Eq. (6) consists of a central scheme,

$$\frac{1}{2}(f(u_{i+\frac{1}{2}}^{L}) + f(u_{i+\frac{1}{2}}^{R})),\tag{7}$$

![](_page_2_Figure_2.jpeg)

**Fig. 1.** Sketch of the new framework to construct candidate stencils towards high-order reconstruction. For present paper, the case K=5 is considered. The left-biased reconstruction is taken as an example without loss of generality. The right-biased reconstruction can be obtained by symmetry.

which is non-dissipative and an upwinding part

$$\left|\tilde{\partial}_{i+\frac{1}{2}}\right| (u_{i+\frac{1}{2}}^R - u_{i+\frac{1}{2}}^L),\tag{8}$$

which denotes the numerical dissipation. Consequently, the numerical dissipation originates from two sources, i.e. the high-order reconstruction which determines the difference between  $u^R_{i+\frac{1}{2}}$  and  $u^L_{i+\frac{1}{2}}$  and the Riemann solver which determines the scaling coefficient  $\left|\tilde{\delta}_{i+\frac{1}{2}}\right|$ .

In smooth regions, the higher-order reconstruction tends to generate smaller data jump between  $u_{i+\frac{1}{2}}^R$  and  $u_{i+\frac{1}{2}}^L$  and thus in-

duces less numerical dissipation. Meanwhile, the equivalent scaling coefficient  $\left|\tilde{\delta}_{i+\frac{1}{2}}\right|$  of distinct Riemann solvers may be quite different and correspondingly different nonlinear dissipation is produced. In particular, when the characteristic signal velocity is set as  $\left|\tilde{\delta}_{i+\frac{1}{2}}\right|=0$ , the overall numerical discretization is non-dissipative.

Note that above analyses are valid for both scalar and system of hyperbolic conservation laws.

# 3. Low-dissipation finite-volume method

In this section, the low-dissipation finite-volume method, which involves a new high-order TENO reconstruction scheme and a novel flux computing strategy, is developed in detail. Note that discussions in this section are based on the one-dimensional conservation law of Eq. (1).

#### 3.1. The new high-order TENO reconstruction

#### 3.1.1. The reconstruction candidate stencils

As shown in Fig. 1, the new reconstruction framework includes three three-point small stencils and a large stencil. The core idea is that the large candidate stencil is employed for the final reconstruction in smooth regions while the adaptation between the three small candidates is invoked to enforce the ENO property in non-smooth regions.

The numerical robustness of classical fifth-order WENO-JS sche -me can be recovered near discontinuities as the first three small stencils are identical to that of WENO-JS. Since the total candidate stencil number is four regardless of *K*, the complexity of resulting TENO scheme is almost independent of the accuracy order.

To achieve high-order reconstruction, a polynomial distribution is assumed on each candidate stencil, i.e.  $u_k(x) \approx \hat{u}_k(x) = \sum_{l=0}^{r-1} a_{l,k} x^l$  and r denotes the stencil width. The coefficients  $a_l$  are determined by solving the yielded system of linear algebraic equations after substituting  $u_k(x)$  into Eq. (3) and evaluating the integral functions at the stencil nodes. In terms of the five-point scheme,

the reconstructed conservative variable at the cell interface  $i + \frac{1}{2}$  can be given [27]

$$u_{0,i+1/2}^{L} = \frac{1}{6}(-\bar{u}_{i-1} + 5\bar{u}_{i} + 2\bar{u}_{i+1}),$$

$$u_{1,i+1/2}^{L} = \frac{1}{6}(2\bar{u}_{i} + 5\bar{u}_{i+1} - \bar{u}_{i+2}),$$

$$u_{2,i+1/2}^{L} = \frac{1}{6}(2\bar{u}_{i-2} - 7\bar{u}_{i-1} + 11\bar{u}_{i}),$$

$$u_{3,i+1/2}^{L} = \frac{6}{10}u_{0,i+1/2}^{L} + \frac{3}{10}u_{1,i+1/2}^{L} + \frac{1}{10}u_{2,i+1/2}^{L},$$

$$(9)$$

where the reconstruction  $u_{3,i+1/2}^{L}$  is a standard fifth-order upwind scheme.

#### 3.1.2. Strong scale separation

To separate smooth scales from discontinuities effectively, the smoothness indicators are defined as [27]

$$\gamma_k = \left(C + \frac{\tau_K}{\beta_{k,r} + \varepsilon}\right)^q, \ k = 0, \dots, 3, \tag{10}$$

where the parameter C=1, q=6 and  $\varepsilon=10^{-40}$ . Following Jiang and Shu [11],  $\beta_{k,r}$  can be evaluated as

$$\beta_{k,r} = \sum_{j=1}^{r-1} \Delta x^{2j-1} \int_{x_{i-1/2}}^{x_{i+1/2}} \left( \frac{d^j}{dx^j} \hat{u}_k(x) \right)^2 dx. \tag{11}$$

The explicit formulas for the five-point reconstruction are given as

$$\begin{split} \beta_{0,3} &= \frac{1}{4} (\bar{u}_{i-1} - \bar{u}_{i+1})^2 + \frac{13}{12} (\bar{u}_{i-1} - 2\bar{u}_i + \bar{u}_{i+1})^2, \\ \beta_{1,3} &= \frac{1}{4} (3\bar{u}_i - 4\bar{u}_{i+1} + \bar{u}_{i+2})^2 + \frac{13}{12} (\bar{u}_i - 2\bar{u}_{i+1} + \bar{u}_{i+2})^2, \\ \beta_{2,3} &= \frac{1}{4} (\bar{u}_{i-2} - 4\bar{u}_{i-1} + 3\bar{u}_i)^2 + \frac{13}{12} (\bar{u}_{i-2} - 2\bar{u}_{i-1} + \bar{u}_i)^2, \\ \beta_{3,5} &= \frac{1}{5040} [\bar{u}_{i+2} (6908\bar{u}_{i+2} - 51001\bar{u}_{i+1} + 67923\bar{u}_i \\ &- 38947\bar{u}_{i-1} + 8209\bar{u}_{i-2}) \\ &+ \bar{u}_{i+1} (104963\bar{u}_{i+1} - 299076\bar{u}_i + 179098\bar{u}_{i-1} \\ &- 38947\bar{u}_{i-2}) + \bar{u}_i (231153\bar{u}_i \\ &- 299076\bar{u}_{i-1} + 67923\bar{u}_{i-2}) + \bar{u}_{i-1} (104963\bar{u}_{i-1} \\ &- 51001\bar{u}_{i-2}) + 6908\bar{u}_{i-2}\bar{u}_{i-2}]. \end{split}$$

For K = 5,  $\tau_5$  is the global reference smoothness indicator and can be devised as [28]

$$\tau_5 = \left| \beta_{3,5} - \frac{1}{6} (\beta_{1,3} + \beta_{2,3} + 4\beta_{0,3}) \right| = O(\Delta x^6), \tag{13}$$

with sixth-order accuracy in smooth regions.

## 3.1.3. The new ENO-like stencil selection

The smoothness indicators are first normalized as

$$\chi_k = \frac{\gamma_k}{\sum_{k=0}^3 \gamma_k} \tag{14}$$

and then filtered by a sharp cutoff function

$$\delta_k = \begin{cases} 0, & \text{if } \chi_k < C_T, \\ 1, & \text{otherwise.} \end{cases}$$
 (15)

where the cut-off parameter  $C_T$  determines the nonlinear adaptation and  $C_T = 10^{-5}$  for K = 5 [27].

The core of original ENO-like stencil selection [27] is that only the candidate crossed by discontinuities is discarded whereas others are applied for the final reconstruction with the corresponding optimal weight. Based on the reconstruction framework in Fig. 1, a new ENO-like stencil selection is proposed as follows.

If the large candidate stencil is judged to be smooth, i.e.  $\delta_3 = 1$ , the high-order reconstructed conservative variable evaluated at cell face  $i + \frac{1}{2}$  is given as

$$u_{i+1/2}^{L} = u_{3,i+1/2}^{L}, (16)$$

where  $u_{3,i+1/2}^L$  is the optimal reconstruction scheme based on the large stencil. Otherwise, if  $\delta_3 = 0$ , the final reconstructed data is given by the nonlinear combination of small candidate stencils

$$u_{i+1/2}^{L} = \sum_{k=0}^{2} w_k u_{k,i+1/2}^{L}, \tag{17}$$

$$w_k = \frac{\alpha_k}{\sum_{k=0}^2 \alpha_k}, \ \alpha_k = d_k \delta_k, \tag{18}$$

and  $d_k$  is optimized such that the combined linear scheme from  $\bigcup_{k=0}^{2} S_k$  achieves the maximum accuracy order. For K=5,  $d_0=$  $\frac{6}{10}$ ,  $d_1=\frac{3}{10}$  and  $d_2=\frac{1}{10}$ . Note that, the coefficients  $d_k$  can also be optimized to obtain

better spectral property, see [36]. Moreover, although not studied in this paper, the candidate reconstruction scheme  $u_{3,i+1/2}^L$  can be optimized independent of the coefficients  $d_k$ .

## 3.2. The new flux computing strategy

For the classical finite-volume method, one certain Riemann solver is employed to evaluate the numerical flux at the cell interface based on the reconstructed data  $u_{i+1/2}^L$  and  $u_{i+1/2}^R$ . The

Riemann solver guarantees that perturbations are propagated in an upwind manner and consequently numerical dissipation is generated. While numerical dissipation is beneficial for shockcapturing, the smooth scales may be smeared simultaneously. It is difficult to develop one universal Riemann solver which delivers uniformly "good" performance for all flow scales [32].

In this paper, a new flux computing strategy is proposed to

In this paper, a new flux computing strategy is proposed to improve the performance as 
$$f_{i+\frac{1}{2}}^{\text{Riemann}}(u_{i+\frac{1}{2}}^L, u_{i+\frac{1}{2}}^R) = \begin{cases} f_{i+\frac{1}{2}}^{\text{low}}(u_{i+\frac{1}{2}}^L, u_{i+\frac{1}{2}}^R), & \text{if } \delta_3^L = 1 \text{ and } \delta_3^R = 1, \\ f_{i+\frac{1}{2}}^{\text{Riemann}}(u_{i+\frac{1}{2}}^L, u_{i+\frac{1}{2}}^R), & \text{otherwise,} \end{cases}$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R D_\rho}{1 + D_\rho},$$

$$\tilde{u} = \frac{u_L + u_R$$

where  $f_{i+\frac{1}{2}}^{\mathrm{low}}$  and  $f_{i+\frac{1}{2}}^{\mathrm{high}}$  denote the relatively "low-dissipation" and

the relatively "dissipative" Riemann flux respectively.

The basic idea is that a "low-dissipation" Riemann flux, even the non-dissipative central flux, is employed when the local flow scales are judged as smooth, whereas a dissipative Riemann flux is adopted for non-smooth scales. Here, it is based on the priority that TENO schemes can separate the nonsmooth scales from smooth scales effectively, which is verified by the approximate dispersion relation (ADR) analysis, see Fig. 3 in [27]. For low to intermediate wavenumbers. TENO schemes recover the optimal linear scheme exactly. The nonlinear adaptation is activated for high-wavenumber scales and discontinuities.

#### 4. Euler equations and HLLC Riemann solver

Considering the Euler equations, the conservative forms are written as

$$\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{u}) = 0, \tag{20}$$

$$\frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla \cdot (\rho\mathbf{u}\mathbf{u} + p\delta) = 0, \tag{21}$$

$$\frac{\partial E}{\partial t} + \nabla \cdot (\mathbf{u}(E+p)) = 0, \tag{22}$$

where  $\rho$  and **u** denote the density and the velocity vector, E = $\rho e + \frac{\rho \dot{\mathbf{u}} \cdot \mathbf{u}}{2}$  is the total energy. To close the equations, the ideal-gas equation of state  $p = (\gamma - 1)\rho e$  with  $\gamma$  as the ratio of specific heats is employed. Without loss of generality, the governing equations can be rewritten as

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F} = 0, \tag{23}$$

where  $\mathbf{U} = [\rho, \rho \mathbf{u}, E]^T$ .

Hereafter, the HLLC [23] approximate Riemann solver, which has been demonstrated to be robust and reliable for Euler equations, is briefly reviewed. The approximate Riemann solution for the discontinuous left and right state  $\mathbf{U}_L$ ,  $\mathbf{U}_R$  is

$$\mathbf{U} = \begin{cases} \mathbf{U}_{L}, & \text{if } S_{L} > 0, \\ \mathbf{U}_{L}^{*}, & \text{if } S_{L} \leq 0 < S_{M}, \\ \mathbf{U}_{R}^{*}, & \text{if } S_{M} \leq 0 \leq S_{R}, \\ \mathbf{U}_{R}, & \text{if } S_{R} < 0, \end{cases}$$

$$(24)$$

and the corresponding flux is

$$\mathbf{F}(\mathbf{U}_{L}, \mathbf{U}_{R}) = \begin{cases} \mathbf{F}_{L}, & \text{if } S_{L} > 0, \\ \mathbf{F}_{L}^{*} = \mathbf{F}_{L} + S_{L}(\mathbf{U}_{L}^{*} - \mathbf{U}_{L}), & \text{if } S_{L} \leq 0 < S_{M}, \\ \mathbf{F}_{R}^{*} = \mathbf{F}_{R} + S_{R}(\mathbf{U}_{R}^{*} - \mathbf{U}_{R}), & \text{if } S_{M} \leq 0 \leq S_{R}, \\ \mathbf{F}_{R}, & \text{if } S_{R} < 0. \end{cases}$$

$$(25)$$

where  $\mathbf{F}_L = \mathbf{F}(\mathbf{U}_L)$  and  $\mathbf{F}_R = \mathbf{F}(\mathbf{U}_R)$ . Concerning the x-component as an example, the acoustic wave-speed  $S_L$  and  $S_R$  are evaluated

$$S_L = \min(u_L - c_L, \tilde{u} - \tilde{c}), \ S_R = \max(u_R + c_R, \tilde{u} + \tilde{c}), \tag{26}$$

$$\begin{cases}
D_{\rho} = \sqrt{\frac{\rho_{R}}{\rho_{L}}}, \\
\tilde{u} = \frac{u_{L} + u_{R} D_{\rho}}{1 + D_{\rho}}, \\
\tilde{H} = \frac{H_{L} + H_{R} D_{\rho}}{1 + D_{\rho}}, \\
\tilde{c} = \sqrt{(\gamma - 1)[\tilde{H} - \frac{1}{2}\tilde{u}^{2}]}.
\end{cases} (27)$$

In order to determine the state  $\mathbf{U}_L^*$  and  $\mathbf{U}_R^*$  in the star region, it

$$S_M = u^* = u_I^* = u_P^*, \ p^* = p_I^* = p_P^*.$$
 (28)

With algebraic operations, the contact wave speed is devised as

$$S_{M} = \frac{\rho_{R} u_{R} (S_{R} - u_{R}) - \rho_{L} u_{L} (S_{L} - u_{L}) + p_{L} - p_{R}}{\rho_{R} (S_{R} - u_{R}) - \rho_{L} (S_{L} - u_{L})},$$

$$p^{*} = \rho_{L} (u_{L} - S_{L}) (u_{L} - S_{M}) + p_{L},$$
(29)

and

$$\rho_{K}^{*} = \rho_{K} \frac{S_{K} - u_{K}}{S_{K} - S_{M}} 
\rho_{K}^{*} u_{K}^{*} = \frac{(S_{K} - u_{K})\rho_{K} u_{K} + p^{*} - p_{K}}{S_{K} - S_{M}} 
\rho_{K}^{*} v_{K}^{*} = \frac{(S_{K} - u_{K})\rho_{K} v_{K}}{S_{K} - S_{M}} , K = L, R.$$

$$\rho_{K}^{*} w_{K}^{*} = \frac{(S_{K} - u_{K})\rho_{K} w_{K}}{S_{K} - S_{M}} 
E_{K}^{*} = \frac{(S_{K} - u_{K})E_{K} - p_{K} u_{K} + p^{*}S_{M}}{S_{K} - S_{M}}$$
(30)

It is proved that the first-order finite-volume method with above HLLC flux preserves the positivity of physical variables with the estimate of acoustic wavespeed following Eq. (26) and under the CFL condition  $\lambda ||u| + a||_{\infty} \le 1$  [22].

In the following, the HLLC Riemann solver is adopted as the building block of  $f_{i+\frac{1}{2}}^{high}$  and the non-dissipative central flux, i.e. Eq. (7), is employed as the building block of  $f_{i+\frac{1}{2}}^{low}$  in Eq. (19). Other choices are straightforward in the proposed framework.

# 5. Analysis and discussions

#### 5.1. The extension to multi-dimensional problems

Considering the extension to multi-dimensional problems, the Gaussian integral over cell faces is necessary to achieve genuinely high-order accuracy for smooth flows [25]. However, compared with the finite-difference method of same accuracy order, the computational cost increases significantly due to the Gaussian integral rule of finite-volume method. Recently, Zhang et al. [26] demonstrate that, for smooth nonlinear systems, the finite-volume WENO method with mid-point rule is only second-order accurate. For high precision simulation of smooth flows, the finitevolume WENO method with Gaussian integral rule could take less computational time to reach the same error threshold than that with mid-point rule. However, the resolution for more general non-smooth shocked problems is often comparable for schemes in both classes with the same building blocks and meshes, despite of the difference in their formal order of accuracy. Furthermore, the cost of the finite-volume WENO method with mid-point rule is significantly less than that with Gaussian integral rule. Motivated by [26], Sun et al. [35] investigate the performances of nonlinear shock-capturing schemes based on the finite-volume method with mid-point rule.

For Cartesian meshes, the present implementation of multidimensional TENO scheme with finite-volume method follows [26, 35] as we mainly concern the resolution property of TENO scheme for under-resolved non-smooth flows. The detailed algorithms are as follows

(1) using the two-dimensional cell-averaged data  $\bar{u}_{i,j}$ , perform the one-dimensional high-order TENO reconstruction in the y direction to obtain the approximate data at the cell interface  $y=y_{j+\frac{1}{2}}$ , i.e.  $u^L_{i,j+\frac{1}{2}}$  and  $u^R_{i,j+\frac{1}{2}}$  based on stencils biased to the left and to the right respectively.

(2) using the two-dimensional cell-averaged data  $\bar{u}_{i,j}$ , perform the one-dimensional high-order TENO reconstruction in the x direction to obtain the approximate data at the cell interface  $x=x_{i+\frac{1}{2}}$ , i.e.  $u_{i+\frac{1}{2},j}^L$  and  $u_{i+\frac{1}{2},j}^R$  based on stencils biased to the left and to the right respectively.

(3) compute the approximate fluxes

$$\widehat{f}_{i+\frac{1}{2},j} = f_{i+\frac{1}{2},j}^{\text{Riemann}}(u_{i+\frac{1}{2},j}^L, u_{i+\frac{1}{2},j}^R), \ \widehat{g}_{i,j+\frac{1}{2}} = g_{i,j+\frac{1}{2}}^{\text{Riemann}}(u_{i,j+\frac{1}{2}}^L, u_{i,j+\frac{1}{2}}^R).$$
(31)

(4) form the finite-volume scheme with mid-point rule [26,35]

$$\frac{d\bar{u}_{i,j}}{dt} = -\frac{1}{\Delta x} (\widehat{f}_{i+1/2,j} - \widehat{f}_{i-1/2,j}) - \frac{1}{\Delta y} (\widehat{g}_{i,j+1/2} - \widehat{g}_{i,j-1/2}), \tag{32}$$

and march it by explicit Runge-Kutta method [31].

Note that, for one-dimensional problem, the high-order accuracy does not degenerate in this framework. And the proposed low-dissipation TENO reconstruction scheme and the flux computing strategy can be extended to the finite-volume method with Gaussian integral rule straightforwardly, if deemed necessary.

Considering more complex geometries, the proposed reconstruction scheme and flux computing strategy can be generalized to the curvilinear meshes. By expanding Eq. (23), the three-dimensional Euler equations can be further written as

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}_x}{\partial x} + \frac{\partial \mathbf{F}_y}{\partial y} + \frac{\partial \mathbf{F}_z}{\partial z} = 0, \tag{33}$$

where  $\mathbf{F}_x$ ,  $\mathbf{F}_y$  and  $\mathbf{F}_z$  denote the physical flux in the x-, yand zcoordinate direction, respectively. This set of hyperbolic conservation laws can be transformed into the curvilinear coordinates ( $\tau$ ,  $\xi$ ,  $\eta$ ,  $\zeta$ ) as

$$\frac{\partial \tilde{\mathbf{Q}}}{\partial \tau} + \frac{\partial \tilde{\mathbf{E}}}{\partial \xi} + \frac{\partial \tilde{\mathbf{F}}}{\partial \eta} + \frac{\partial \tilde{\mathbf{G}}}{\partial \zeta} = 0, \tag{34}$$

where

$$\tilde{\mathbf{Q}} = \mathbf{U}/\mathbf{I},\tag{35}$$

$$\tilde{\mathbf{E}} = \tilde{\xi}_t \mathbf{U} + \tilde{\xi}_x \mathbf{F}_x + \tilde{\xi}_y \mathbf{F}_y + \tilde{\xi}_z \mathbf{F}_z, \tag{36}$$

$$\tilde{\mathbf{F}} = \tilde{\eta}_t \mathbf{U} + \tilde{\eta}_x \mathbf{F}_x + \tilde{\eta}_v \mathbf{F}_v + \tilde{\eta}_z \mathbf{F}_z, \tag{37}$$

$$\tilde{\mathbf{G}} = \tilde{\zeta}_t \mathbf{U} + \tilde{\zeta}_x \mathbf{F}_x + \tilde{\zeta}_y \mathbf{F}_y + \tilde{\zeta}_z \mathbf{F}_z. \tag{38}$$

The Jacobian J and the standard metrics can be computed by high-order finite-difference schemes, see e.g. [38–40]. The details are not given explicitly to avoid duplications.

Consequently, the proposed reconstruction scheme and flux computing strategy can be invoked in the coordinates  $(\tau, \xi, \eta, \zeta)$  similar to the way for the Cartesian coordinates. The midpoint rule is employed for simplicity as the resolution property for shocked flows is the main focus of this paper. As shown in [26], the conclusion is not affected whether the Gaussian integral rule is imposed or not for such type non-smooth simulations.

#### 5.2. The extension to unstructured meshes

On top of an existing high-order finite-volume WENO scheme on unstructured meshes, the implementation of proposed TENO reconstruction and flux evaluation strategy should have no fundamental problems due to the similar mathematical formulas [41]. In addition to the low-dissipation property, the proposed framework may provide one solution to resolve the stubborn problem of classical unstructured WENO scheme [41–43], i.e. (1) for specific local geometry and computing mesh, there may be no desired optimal linear weights for candidate stencils to assemble the high-order reconstruction in smooth regions; (2) negative optimal linear weights may appear for candidate stencils and lead to unstable reconstruction.

For the proposed framework, the high-order reconstruction is guaranteed by choosing the embedded large candidate stencil with the new TENO type stencil selection. The nonlinear convex combination of three small candidates is mainly designed to recover the ENO property in non-smooth regions. Consequently, rather than determined by forming the high-order reconstruction, the optimal linear weights of these three small candidates can be any positive parameters with the condition that  $\sum_{k=0}^{2} d_k = 1$ , e.g. the candidate stencil biased to central reconstruction may be assigned

![](_page_5_Figure_2.jpeg)

**Fig. 2.** Convergence of the *L*∞ error from the TENO5 scheme.

a significantly larger weight to maintain an overall low-dissipation scheme.

For Cartesian mesh, one straightforward choice is to design the three optimal weights to achieve the fifth-order reconstruction as the above mentioned problems do not exist and the spectral property of fifth-order scheme is adequate in practice.

# **6. Numerical validation**

The high-dimensional problem is treated by a dimension-bydimension manner [[44\]](#page-14-40). The high-order TENO reconstruction is employed in characteristic space to effectively suppress spurious oscillations. The third-order strong-stability-preserving (SSP) Runge–Kutta method [\[31\]](#page-14-28)

$$\mathbf{U}^{(1)} = \mathbf{U}^{n} + \Delta t L(\mathbf{U}^{n}), 
\mathbf{U}^{(2)} = \frac{3}{4} \mathbf{U}^{n} + \frac{1}{4} \mathbf{U}^{(1)} + \frac{1}{4} \Delta t L(\mathbf{U}^{(1)}), 
\mathbf{U}^{n+1} = \frac{1}{3} \mathbf{U}^{n} + \frac{2}{3} \mathbf{U}^{(2)} + \frac{2}{3} \Delta t L(\mathbf{U}^{(2)})$$
(39)

is adopted for the time integration of governing equations with a CFL number of 0.4 following [[28\]](#page-14-33). All parameters in the TENO scheme are not tuned in order to assess the numerical robustness.

Hereafter, TENO5-FV and TENO5-LFV denote the proposed TENO5 (*q* = 6) scheme in combination with standard and lowdissipation finite-volume method, respectively. WENO5-Z-FV denotes the fifth-order WENO5-Z (*q* = 2 [[13](#page-14-12)[,45](#page-14-41)]) scheme in combination with standard finite-volume method.

#### *6.1. Accuracy test problem*

The one-dimensional Gaussian pulse advection problem [[46\]](#page-14-42) is considered. The linear advection equation

$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0, (40)$$

with initial condition

$$u_0(x) = e^{-300(x - x_c)^2}, \ x_c = 0.5,$$
 (41)

is solved in a computational domain 0 ≤ *x* ≤ 1. Periodic boundary condition is imposed at *x* = 0 and *x* = 1.

![](_page_5_Figure_17.jpeg)

**Fig. 3.** Convergence of the *L*∞ error from the TENO5 scheme for the test function with second-order critical points.

![](_page_5_Figure_19.jpeg)

**Fig. 4.** Convergence of the *L*∞ error from the TENO5 scheme for 1D inviscid nonlinear Burger's equation.

As shown in [Fig.](#page-5-1) [2](#page-5-1), the theoretical accuracy order is achieved for the proposed TENO5 scheme.

We further consider a test function

$$u_0(x) = e^{0.75(x-1)}x^3, \ x \in [-1, 1],$$
 (42)

which has a critical point of order two at *x* = 0. As shown in [Fig.](#page-5-2) [3,](#page-5-2) unlike the classical WENO-JS and WENO-Z schemes [\[47\]](#page-14-43), even with the second-order critical point, the desired convergence order is achieved for the TENO5 scheme.

At last, we consider the 1D nonlinear Burgers equation

$$u_t + (\frac{u^2}{2})_x = 0, (43)$$

![](_page_6_Figure_2.jpeg)

Fig. 5. Shock-tube simulations: the Lax's problem (top) and the Sod's problem (bottom). Left: density profile. Right: velocity profile. Discretization on 100 uniformly distributed grid points.

with a sine wave  $u(x,0)=0.5+\sin(\pi x)$  as the initial condition. The simulation is terminated at  $t=0.5/\pi$ , when the solution is smooth. The computational domain is [0,2] with periodic boundary condition. In order to remove machine round-off errors, the sensitivity parameter is set as  $\epsilon=10^{-13}$ .

As shown in Fig. 4, for the nonlinear advection simulation, the theoretical accuracy order is obtained for the TENO5 scheme.

# 6.2. Shock-tube problem

Two shock-tube test problems, i.e. the Lax's problem [48] and the Sod's problem [49], are simulated. The initial condition for Lax's problem is

$$(\rho, u, p) = \begin{cases} (0.445, 0.689, 3.528), & \text{if } 0 \le x < 0.5, \\ (0.5, 0, 0.5710), & \text{if } 0.5 \le x \le 1, \end{cases}$$
(44)

and the final simulation time is t = 0.14.

The initial condition for Sod's problem is

$$(\rho, u, p) = \begin{cases} (1, 0, 1), & \text{if } 0 \le x < 0.5, \\ (0.125, 0, 0.1), & \text{if } 0.5 \le x \le 1, \end{cases}$$
 (45)

and the final simulation time is t = 0.2.

As shown in Fig. 5, both the proposed TENO5-FV and TENO5-LFV scheme perform well in resolving shock and contact discontinuities.

We further consider a modified Sod's problem with initial condition [32]

$$(\rho, u, p) = \begin{cases} (1, 0.75, 1), & \text{if } 0 \le x < 0.5, \\ (0.125, 0, 0.1), & \text{if } 0.5 \le x \le 1, \end{cases}$$
 (46)

and the final simulation time is t = 0.2.

The solution has a right shock wave, a right traveling contact wave and a left sonic rarefaction wave [32]. As shown in Fig. 6, the results from both two proposed schemes agree with the analytical solutions well.

# 6.3. Interacting blast waves

The two-blast-wave interaction taken from [50] is considered. The initial condition is

$$(\rho, u, p) = \begin{cases} (1, 0, 1000), & \text{if } 0 \le x < 0.1, \\ (1, 0, 0.01), & \text{if } 0.1 \le x < 0.8, \\ (1, 0, 100), & \text{if } 0.8 \le x \le 1. \end{cases}$$

$$(47)$$

![](_page_7_Figure_2.jpeg)

**Fig. 6.** Shock-tube simulation: the modified Sod's problem. Top left: density profile; top right: velocity profile; bottom: pressure profile. Discretization on 100 uniformly distributed grid points.

![](_page_7_Figure_4.jpeg)

**Fig. 7.** Interacting blast waves problem: solutions from the TENO5-FV, TENO5-LFV and WENO5-Z-FV scheme. Density distribution (left) and a zoom-in view of the density profile (right). Discretization on 400 uniform grid points. The simulation time *t* = 0.038.

![](_page_8_Figure_2.jpeg)

**Fig. 8.** Shu–Osher shock density-wave interaction problem: solutions from the TENO5-FV, TENO5-LFV and WENO5-Z-FV scheme. Density distribution (left) and a zoom-in view of the density distribution (right). Discretization on 200 uniformly distributed grid points.

![](_page_8_Figure_4.jpeg)

**Fig. 9.** Titarev–Toro shock-density wave interaction problem: solutions from the TENO5-FV, TENO5-LFV and WENO5-Z-FV scheme. Density distribution (left) and a zoom-in view of the density distribution (right). Discretization on 1000 uniformly distributed grid points.

The simulation is performed on a uniform mesh with *N* = 400. The ''exact'' solution is computed by the fifth-order WENO-JS scheme on a uniform mesh with *N* = 2500.

As shown in [Fig.](#page-7-1) [7,](#page-7-1) for this strong blast-wave interaction problem, the resolved solutions from the TENO5-FV and TENO5-LFV scheme agree well with the reference. The present results at resolution of 400 are much better than that from the latest five-point WENO-ZQ scheme at resolution of 800 (see Fig. 3.7 of [[16](#page-14-15)]), and are also comparable to that from the original TENO6 scheme at the same resolution [[27](#page-14-26)]. WENO5-Z-FV is much more dissipative than both TENO5-FV and TENO5-LFV.

## *6.4. Shu–Osher problem*

This case is proposed by Shu and Osher [[51](#page-14-47)]. A one-dimensional Mach 3 shock wave interacts with a perturbed density field generating both small-scale structures and discontinuities. The initial condition is

$$(\rho, u, p) = \begin{cases} (3.857, 2.629, 10.333), & \text{if } 0 \le x < 1, \\ (1 + 0.2\sin(5(x - 5)), 0, 1), & \text{if } 1 \le x \le 10. \end{cases}$$
 (48)

The computational domain is [0,10] with *N* = 200 uniformly distributed mesh cells and the final evolution time is *t* = 1.8. The reference solution is obtained by the fifth-order WENO scheme with *N* = 2000.

As shown in [Fig.](#page-8-0) [8,](#page-8-0) while the shocklets are captured sharply by both schemes, TENO5-LFV performs better than TENO5-FV in resolving the high-wavenumber fluctuations. WENO5-Z-FV is the most dissipative scheme.

A variant of Shu–Osher problem [[51](#page-14-47)] has been proposed by Titarev and Toro [\[25\]](#page-14-24) to test a severely oscillatory wave interacting with shock discontinuity. The initial condition is

$$(\rho, u, p) = \begin{cases} (1.515695, 0.523346, 1.805), & \text{if } 0 \le x < 0.5, \\ (1 + 0.1\sin(20\pi(x - 5)), 0, 1), & \text{if } 0.5 \le x \le 10. \end{cases}$$

$$(49)$$

The computational domain is [0,10] with *N* = 1000 uniformly distributed mesh cells and the final evolution time is *t* = 5. The reference solution is obtained by the fifth-order WENO-JS scheme with *N* = 5000.

![](_page_9_Figure_2.jpeg)

**Fig. 10.** Double Mach reflection of a strong shock: density contours from the WENO5-Z-FV, TENO5-FV and TENO5-LFV scheme at simulation time t=0.2. Resolution at  $512 \times 128$  (top),  $800 \times 200$  (middle) and  $1024 \times 256$  (bottom). This figure is drawn with 43 density contours between 1.887 and 20.9.

(50)

As shown in Fig. 9, the TENO5-LFV scheme performs significantly better than TENO5-FV in resolving the density fluctuations. TENO5-FV performs better than WENO5-Z-FV. Note that, the result from present TENO5-LFV scheme at resolution of 1000 is comparable to that from the high-order finite-volume WENO-FORCE scheme at resolution of 2000, see Fig. 6 of [25].

#### 6.5. Double mach reflection of a strong shock

This two-dimensional case is taken from Woodward and Colella [50]. The initial condition is

$$(\rho, u, v, p) = \begin{cases} (1.4, 0, 0, 1), & \text{if } y < 1.732(x - 0.1667), \\ (8, 7.145, -4.125, 116.8333), & \text{otherwise.} \end{cases}$$

The computational domain is  $[0, 4] \times [0, 1]$  and the simulation time is t = 0.2. Initially, a right-moving Mach 10 shock wave is placed at x = 0.1667 with an incident angle of  $60^{\circ}$  to the x-axis. The post-shock condition is imposed from x = 0 to x = 0.1667 whereas a reflecting wall condition is enforced from x = 0.1667 to x = 4 at the better

As shown in Fig. 10, at all three resolutions, the proposed TENO5-LFV scheme resolves much finer small-scale structures than TENO5-FV near the Mach stems, implying its low-dissipation property. At the same resolution, TENO5-FV performs better than WENO5-Z-FV. With resolution of  $800 \times 200$ , the result from present five-point TENO5-LFV scheme is much better than that from the nineand eleven-point WENO-Z scheme (see Fig. 10 of [45]). Moreover, TENO5-LFV delivers similar results at resolution of  $1024 \times 256$  in comparison with that from the eleven-point WENO-Z scheme at resolution of  $1600 \times 400$  [45]. The TENO5-FV and TENO5-LFV scheme perform better than the original

![](_page_10_Picture_2.jpeg)

**Fig. 11.** Rayleigh–Taylor instability: density contours from the WENO5-Z-FV, TENO5-FV and TENO5-LFV scheme. Resolution of  $64 \times 256$ . The simulation is run until t = 1.95.

TENO5 and TENO5-opt scheme at the same resolution, see Fig. 17 of [27]. Compared with the classical finite-volume WENO-JS method with/without Gaussian quadrature, the present methods resolve much more small-scale structures with a lower resolution, see Fig. 4 and Fig. 5 of [26].

#### 6.6. Rayleigh-Taylor instability

The initial condition is [52]

$$(\rho, u, v, p) = \begin{cases} (2, 0, -0.025c\cos(8\pi x), 1 + 2y), & \text{if } 0 \le y < 1/2, \\ (1, 0, -0.025c\cos(8\pi x), y + 3/2), & \text{if } 1/2 \le y \le 1, \end{cases} (51)$$

where the sound speed  $c=\sqrt{\gamma\frac{p}{\rho}}$  with  $\gamma=\frac{5}{3}$ . The computational domain is  $[0,0.25]\times[0,1]$ . Reflective boundary conditions are imposed at the left and right side of the domain. Constant primitive variables  $(\rho,u,v,p)=(2,0,0,1)$  and  $(\rho,u,v,p)=(1,0,0,2.5)$  are set for the bottom and top boundary.

For inviscid simulations, the smaller magnitude of numerical dissipation from high-order shock-capturing scheme results in finer small-scale structures. As shown in Fig. 11, TENO5-LFV generates much finer vortical structures than TENO5-FV and WENO5-Z-FV. While the low-dissipation shock-capturing schemes tend to break the flow symmetry (see e.g. Fig. 13 of [53]), this property can be improved by removing the round-off error with larger sensitivity parameter for the proposed TENO scheme as shown in Fig. 12. Note that the resolution property can also be improved by increasing the sensitivity parameter.

![](_page_10_Picture_10.jpeg)

**Fig. 12.** Rayleigh–Taylor instability: density contours from the TENO5-FV scheme with sensitivity parameter  $\varepsilon=10^{-6}$  (left),  $\varepsilon=10^{-8}$  (middle) and  $\varepsilon=10^{-10}$  (right). Resolution of  $64\times256$ . The simulation is run until t=1.95.

## 6.7. 2D Riemann problem

We consider the 2D Riemann problem of configuration 3 [54, 55]. The initial condition is

$$(\rho, u, v, p) = \begin{cases} (1.5, 0, 0, 1.5), & \text{if } x > 0.5, y > 0.5, \\ (0.5323, 1.206, 0, 0.3), & \text{if } x < 0.5, y > 0.5, \\ (0.138, 1.206, 1.206, 0.029), & \text{if } x < 0.5, y < 0.5, \\ (0.5323, 0, 1.206, 0.3), & \text{if } x > 0.5, y < 0.5. \end{cases}$$
(52)

The final computational time is t = 0.3. The computational domain is  $[0, 1] \times [0, 1]$ . The mesh resolution is  $512 \times 512$ .

As shown in Fig. 13, the TENO5-LFV scheme is less dissipative than TENO5-FV and WENO5-Z-FV. TENO5-FV performs better than WENO5-Z-FV in predicting the small-scale structures.

We further investigate the 2D Riemann problem of configuration 12 [54,55]. The initial condition is

$$(\rho, u, v, p) = \begin{cases} (0.5313, 0, 0, 0.4), & \text{if } x > 0.5, y > 0.5, \\ (1, 0.7276, 0, 1), & \text{if } x < 0.5, y > 0.5, \\ (0.8, 0, 0, 1), & \text{if } x < 0.5, y < 0.5, \\ (1, 0, 0.7276, 1), & \text{if } x > 0.5, y < 0.5. \end{cases}$$
(53)

The final computational time is t=0.25. The computational domain is  $[0,1]\times[0,1]$ . The mesh resolution is  $1024\times1024$ .

As shown in Fig. 14, the same conclusion holds.

![](_page_11_Figure_2.jpeg)

**Fig. 13.** 2D Riemann problem with configuration 3: density distributions. The numerical results are from the WENO5-Z-FV scheme (top left), the TENO5-FV scheme (top right) and the TENO5-LFV scheme (bottom). The final simulation time is t = 0.3.

# 6.8. Compressible isotropic turbulence decay

The under-resolved simulation of three-dimensional compressible isotropic turbulence decay at turbulent Mach number  $M_{t_0} = \frac{\sqrt{3}u_{rms,0}}{\langle c \rangle} = 0.6$  with  $u_{rms} = \sqrt{\langle u_i u_i \rangle/3}$  is challenging [36,56–58]. The computational domain is  $[0,2\pi] \times [0,2\pi] \times [0,2\pi]$  with periodic boundary conditions. The initial Reynolds number is set as  $\mathrm{Re}_{\lambda_0} = \frac{\langle \rho \rangle u_{rms,0} \lambda_0}{\langle \mu \rangle} = 100$ , where the Taylor-micro-scale is defined  $\lambda = \frac{1}{3} \sum_i \sqrt{\frac{\langle u_i^2 \rangle}{\langle u_i^2 \rangle}} = 0.5$ . The other initial parameters are set as

the same as that in [56].

As shown in Fig. 15, while the TENO5-FV scheme performs significantly better than WENO5-Z-FV in predicting the turbulence statistics, TENO5-LFV further improves the resolution property of TENO5-FV.

#### 6.9. Bow shock flow

For another canonical case, we consider the stationary bow shock flow to verify the performance of proposed methods for problems involving curved physical boundaries. A cylinder of unit radius is positioned at the origin of physical x-y plane and the computational domain is  $[0,1] \times [0,1]$  on the  $\xi - \eta$  plane. The mapping between these two domains follows [11]

$$\begin{cases} x = (R_x - (R_x - 1)\xi)\cos(\theta(2\eta - 1)) \\ y = (R_y - (R_y - 1)\xi)\sin(\theta(2\eta - 1)) \end{cases},$$
 (54)

where  $R_x=3$ ,  $R_y=6$  and  $\theta=\frac{5\pi}{12}$ . The flow is initialized by a constant Mach 3 flow moving toward the cylinder from the left [59]. The reflective boundary condition is imposed at  $\xi=1$  and the outflow boundary condition is applied at the other two boundaries. A uniform grid of resolution  $120\times160$  in the domain  $(\xi,\eta)$  is employed. To avoid possible shock-induced instabilities [60,61], the HLL Riemann solver [21] is applied as the "dissipative" flux instead of HLLC [23]. The simulations are marched until the steady state is achieved.

As shown in Fig. 16, the proposed numerical methods perform well in capturing the steady bow shock on the curved mesh and the results agree with that from WENO-Z scheme. Similar to the one-dimensional shock-tube cases, the difference between results from TENO5-FV and TENO5-LFV is not distinguishable.

![](_page_12_Figure_2.jpeg)

**Fig. 14.** 2D Riemann problem with configuration 12: density distributions. The numerical results are from the WENO5-Z-FV scheme (top left), the TENO5-FV scheme (top right) and the TENO5-LFV scheme (bottom). The final simulation time is *t* = 0.25.

# **7. Conclusions**

In this paper, we propose a low-dissipation finite-volume method for hyperbolic conservation laws. The contributions of this paper are summarized as follows

- A family of tailored high-order TENO schemes is proposed. The high-order reconstruction is achieved by adaptively combining three three-point stencils and one large candidate stencil. A new ENO-like stencil selection strategy is proposed. The large candidate stencil is selected as the final reconstruction when it is judged as smooth; otherwise, the nonlinear adaptation between the three small stencils formulates the final reconstruction to enforce ENO property. The resulting TENO scheme inherits the low-dissipation advantage of original TENO schemes and can be extended to arbitrarily highorder reconstruction without significant complexity increase.
- According to the TENO weighting strategy, when the large candidate stencils for both the leftand right-side reconstruction are judged as smooth, a ''low-dissipation'' Riemann solver, e.g. even the non-dissipative central flux, is applied for the flux computing at the cell interface; otherwise, an alternative ''dissipative'' Riemann solver is adopted for the flux

- integrating. The choice of ''low-dissipation'' and ''dissipative'' Riemann solver is flexible.
- A set of benchmark cases including strong discontinuities and broadband fluctuations is simulated. Numerical experiments demonstrate that the proposed method is low-dissipative while capable of capturing discontinuities sharply.
- The new proposed TENO reconstruction scheme can be straight forwardly extended to finite-difference framework by assigning the point flux in the finite-difference method identical to the cell average of the finite-volume method. The advantages of proposed TENO reconstruction schemes remain the same for the finite-difference framework.
- As the general building blocks of a finite-volume method, the present new low-dissipation high-order reconstruction scheme and flux computing strategy can be implemented on top of or extended to existing high-order finite-volume WENO frameworks, e.g. the finite-volume WENO scheme with unstructured meshes or with curvilinear meshes. Gaussian integral rule may be imposed to achieve high-order convergence for smooth flows, if deemed necessary.

The proposed new TENO scheme can be extended to arbitrarily high-order reconstruction and will be reported separately.

![](_page_13_Figure_2.jpeg)

**Fig. 15.** Compressible isotropic turbulence decay: comparisons of temporal evolution of different quantities on a 64<sup>3</sup> mesh. Circles denote the filtered DNS data from that produced on a 256<sup>3</sup> mesh [[56\]](#page-14-52).

![](_page_14_Picture_2.jpeg)

Fig. 16. Bow shock flow. Left: the curvilinear mesh (top) and the density distribution from WENO-Z (bottom). Middle: the density distribution from TENO5-FV (top) and TENO5-LFV (bottom) with  $\varepsilon = 10^{-40}$ . Right: the density distribution from TENO5-FV (top) and TENO5-LFV (bottom) with  $\varepsilon = 10^{-10}$ . This figure is drawn with 20 density contours between 1.5 and 5.5. Note that, the WENO-Z and TENO5-FV scheme also employ the HLL Riemann solver [21] for fair comparisons.

#### Acknowledgments

The author has been funded by U.S. Air Force Office of Scientific Research (AFOSR) and Predictive Science Academic Alliance Program (PSAAP).

#### References

- [1] G.D. Smith, Numerical Solution of Partial Differential Equations: Finite Difference Methods, Oxford university press, 1985.
- [2] R.J. LeVeque, Finite Volume Methods for Hyperbolic Problems, vol. 31, Cambridge university press, 2002.
- [3] B. Cockburn, G.E. Karniadakis, C.-W. Shu, Discontinuous Galerkin Methods, Springer, 2000, pp. 3-50.
- [4] C.-W. Shu, J. Comput. Phys. 316 (2016) 598-613.
- [5] V. Kolgan, J. Comput. Phys. 230 (7) (2011) 2384-2390.
- [6] S.K. Godunov, Mat. Sb. 89 (3) (1959) 271-306.

- [7] A. Harten, I. Comput. Phys. 49 (1983) 357-393.
- [8] K.H. Kim, C. Kim, J. Comput. Phys. 208 (2) (2005) 527-569.
- [9] A. Harten, B. Engquist, S. Osher, S.R. Chakravarthy, J. Comput. Phys. 71 (1987) 231-303
- [10] X.D. Liu, S. Osher, T. Chan, J. Comput. Phys. 115 (1994) 200-212.
- [11] G.S. Jiang, C.-W. Shu, J. Comput. Phys. 126 (1) (1996) 202-228.
- [12] A.K. Henrick, T. Aslam, J.M. Powers, J. Comput. Phys. 207 (2005) 542-567.
- [13] R. Borges, M. Carmona, B. Costa, W.S. Don, J. Comput. Phys. 227 (2008) 3191-3211
- [14] Y.-X. Ren, M. Liu, H. Zhang, J. Comput. Phys. 192 (2003) 365-386.
- [15] G. Gerolymos, D. Sénéchal, I. Vallet, J. Comput. Phys. 228 (2009) 8481-8524.
- [16] J. Zhu, J. Qiu, J. Comput. Phys. 318 (2016) 110-121.
- [17] B. Van Leer, Upwind and High-Resolution Schemes, Springer, 1997, pp. 80-89.
- [18] J.L. Steger, R. Warming, J. Comput. Phys. 40 (2) (1981) 263–293.
- [19] M.-S. Liou, C.J. Steffen, J. Comput. Phys. 107 (1) (1993) 23–39.
- [20] P.L. Roe, J. Comput. Phys. 43 (2) (1981) 357–372.
- [21] A. Harten, P.D. Lax, B. Van Leer, Upwind and High-Resolution Schemes, Springer, 1997, pp. 53–79.
- [22] P. Batten, N. Clarke, C. Lambert, D. Causon, SIAM J. Sci. Comput. 18 (6) (1997) 1553-1570
- [23] E.F. Toro, M. Spruce, W. Speares, Shock Waves 4 (1) (1994) 25–34.
- [24] M.-S. Liou, J. Comput. Phys. 129 (2) (1996) 364–382.
- [25] V.A. Titarev, E.F. Toro, J. Comput. Phys. 201 (1) (2004) 238–260.
- [26] R. Zhang, M. Zhang, C.-W. Shu, Commun. Comput. Phys. 9 (03) (2011) 807-827
- [27] L. Fu, X.Y. Hu, N.A. Adams, J. Comput. Phys. 305 (2016) 333-359.
- [28] L. Fu, X.Y. Hu, N.A. Adams, J. Comput. Phys. 349 (2017) 97–121.
- [29] L. Fu, X.Y. Hu, N.A. Adams, J. Comput. Phys. 374 (2018) 724-751, http://dx.doi. org/10.1016/j.jcp.2018.07.043.
- [30] L. Fu, X.Y. Hu, N.A. Adams, Commun. Comput. Phys. (2018) in press.
- [31] S. Gottlieb, C.-W. Shu, E. Tadmor, SIAM Rev. 43 (1) (2001) 89–112.
- [32] E.F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer Science & Business Media, 2013.
- [33] C.-W. Shu, SIAM Rev. 51 (1) (2009) 82–126.
- [34] C.-W. Shu, S. Osher, J. Comput. Phys. 77 (2) (1988) 439-471.
- [35] Z. Sun, S. Inaba, F. Xiao, J. Comput. Phys. 322 (2016) 309-325.
- [36] G.M. Arshed, K.A. Hoffmann, J. Comput. Phys. 246 (2013) 58-77.
- [37] B. Einfeldt, C.-D. Munz, P.L. Roe, B. Sjögreen, J. Comput. Phys. 92 (2) (1991) 273-295
- [38] B. Sjögreen, H.C. Yee, M. Vinokur, J. Comput. Phys. 265 (2014) 211–220.
- [39] Y. Jiang, C.-W. Shu, M. Zhang, Methods Appl. Anal. 21 (1) (2014) 001–030.
- [40] T. Nonomura, N. Iizuka, K. Fujii, Comput. & Fluids 39 (2) (2010) 197–214.
- [41] C. Hu, C.-W. Shu, J. Comput. Phys. 150 (1) (1999) 97–127.
- [42] M. Dumbser, M. Käser, J. Comput. Phys. 221 (2) (2007) 693-723.
- [43] J. Shi, C. Hu, C.-W. Shu, J. Comput. Phys. 175 (1) (2002) 108-127. [44] J. Casper, H. Atkins, J. Comput. Phys. 106 (1) (1993) 62-76.
- - [45] M. Castro, B. Costa, W.S. Don, J. Comput. Phys. 230 (5) (2011) 1766-1792.
- [46] N.K. Yamaleev, M.H. Carpenter, J. Comput. Phys. 228 (11) (2009) 4248–4272.
- W.S. Don, R. Borges, J. Comput. Phys. 250 (2013) 347-372. [48] P.D. Lax, Comm. Pure Appl. Math. 7 (1954) 159-193.
- [49] G.A. Sod, J. Comput. Phys. 27 (1978) 1-31.
- [50] P. Woodward, J. Comput. Phys. 54 (1984) 115–173.
- [51] C.W. Shu, S. Osher, J. Comput. Phys. 83 (1989) 32-78.
- [52] Z. Xu, C.W. Shu, J. Comput. Phys. 205 (2005) 458-485.
- [53] Z.-S. Sun, L. Luo, Y.-X. Ren, S.-Y. Zhang, J. Comput. Phys. 270 (2014) 238-254.
- [54] P.D. Lax, X.D. Liu, SIAM J. Sci. Comput. 19 (2) (1998) 319–340.
- [55] A. Kurganov, E. Tadmor, Numer. Methods Partial Differential Equations 18 (5) (2002) 584-608.
- [56] E. Johnsen, J. Larsson, A.V. Bhagatwala, W.H. Cabot, P. Moin, B.J. Olson, P.S. Rawat, S.K. Shankar, B. Sjögreen, H. Yee, et al., J. Comput. Phys. 229 (4) (2010) 1213-1237
- S. Kawai, S.K. Shankar, S.K. Lele, J. Comput. Phys. 229 (5) (2010) 1739–1762.
- [58] D. Kotov, H. Yee, A. Wray, B. Sjögreen, A. Kritsuk, J. Comput. Phys. 307 (2016) 189-202.
- [59] S. Tan, C.-W. Shu, J. Comput. Phys. 229 (21) (2010) 8144–8166.
- [60] L. Fu, Z.H. Gao, X.D. Zhang, ACTA Aerodyn. Sin. 32 (1) (2014) 116-122.
- [61] M. Pandolfi, D. D'Ambrosio, J. Comput. Phys. 166 (2) (2001) 271-301.